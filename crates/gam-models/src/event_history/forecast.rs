//! History-conditioned forecasting, the predictive probability integral
//! transform, and the smoothed latent state of a subject.
//!
//! A forecast takes the Laplace posterior of the subject's latent state at
//! its exit and runs the sequential Gaussian filter over future quadrature
//! nodes with zero counts. With the compensator restricted to a set of marks
//! `C`, the product of the normalisers up to a node is `P(no event of any
//! mark in C before it | history)`, and `w_d · S_C(t⁻) · E_pred[λ_d(t)]`
//! summed over the nodes is the probability that the first event of mark
//! `d ∈ C` happens by the horizon before any other mark of `C`. Every mark's
//! risk therefore uses `C = {d} ∪ terminal marks`: the probability of a first
//! occurrence of `d` by the horizon while the subject is still alive. The
//! survival is `C = terminal marks`. Risk-set membership is data: a
//! once-only mark the subject already had carries no risk.
//!
//! A window may also start from the stationary prior at given covariate
//! values (`population_forecast`): with population values that is the
//! population tier, with a subject's own score it is what the model says
//! before any history is seen; a subject's own forecast is the same window
//! continued from its smoothed state. The tiers are one model conditioned
//! on more, with no weight between them chosen.
//!
//! The predictive PIT of an event is `1 − P(no event of any mark at risk in
//! (t_prev, t_event] | history)`, which the same filter yields as the product
//! of the normalisers of the zero-count nodes between consecutive events; it
//! is uniform when the model is right.

use super::cohort::{
    CohortNodes, CovariateSegment, EventHistoryCohort, EventHistoryError, MarkKind,
    SubjectHistory, expand_nodes,
};
use super::family::EventHistoryFit;
use super::laplace::{self, Gaussian, SubjectInputs, expected_intensity, filter_pass};
use gam_terms::smooth::build_term_collection_design;
use ndarray::Array2;

/// A forecast for one subject: per horizon and mark, the probability of a
/// first occurrence of the mark by the horizon before any terminal event,
/// given the history.
#[derive(Clone, Debug)]
pub struct RiskForecast {
    pub horizons: Vec<f64>,
    /// `horizons × marks`: `P(first event of d in (exit, horizon] before any
    /// terminal event | history)`; `NaN` for a once-only mark the subject
    /// already had.
    pub risk: Array2<f64>,
    /// `P(no terminal event in (exit, horizon] | history)`.
    pub survival: Vec<f64>,
    /// `horizons × marks`: the expected number of events of the mark by the
    /// horizon before any terminal event. Equals `risk` for once-only and
    /// terminal marks; `NaN` where `risk` is.
    pub expected_counts: Array2<f64>,
}

/// The smoothed latent state of one subject on its own nodes.
#[derive(Clone, Debug)]
pub struct LatentPath {
    pub times: Vec<f64>,
    /// `nodes × atoms` posterior means.
    pub means: Array2<f64>,
    /// Per node, the `atoms × atoms` posterior covariance, row-major and
    /// concatenated.
    pub covariances: Vec<f64>,
}

fn single_subject_nodes(
    fit: &EventHistoryFit,
    cohort: &EventHistoryCohort,
    history: &SubjectHistory,
) -> Result<CohortNodes, EventHistoryError> {
    let mut one = EventHistoryCohort {
        mark_names: cohort.mark_names.clone(),
        mark_kinds: cohort.mark_kinds.clone(),
        covariate_names: cohort.covariate_names.clone(),
        covariates: cohort.covariates.clone(),
        subjects: vec![history.clone()],
    };
    one.validate()?;
    expand_nodes(&one, fit.quadrature_order)
}

/// Node log-intensities `η⁰` (design × coefficients + offset) for every
/// mark on a node set.
fn node_eta0(fit: &EventHistoryFit, nodes: &CohortNodes) -> Result<Vec<f64>, EventHistoryError> {
    let marks = fit.marks();
    let total = nodes.total_nodes;
    let mut eta0 = vec![0.0; total * marks];
    for d in 0..marks {
        let design = build_term_collection_design(nodes.node_data.view(), &fit.frozen_specs[d])
            .map_err(|error| EventHistoryError::Fit {
                reason: format!("prediction design for mark {d}: {error}"),
            })?;
        let beta = fit.mark_coefficients(d);
        if design.design.ncols() != beta.len() {
            return Err(EventHistoryError::Fit {
                reason: format!(
                    "prediction design for mark {d} has {} columns, the fit has {} coefficients",
                    design.design.ncols(),
                    beta.len()
                ),
            });
        }
        let dense = design
            .design
            .try_to_dense_arc("event-history prediction design")
            .map_err(|error| EventHistoryError::Fit {
                reason: error.to_string(),
            })?;
        for row in 0..total {
            let mut value = design.affine_offset[row];
            for (j, x) in dense.row(row).iter().enumerate() {
                value += x * beta[j];
            }
            eta0[row * marks + d] = value;
        }
    }
    Ok(eta0)
}

fn latent_parameters(fit: &EventHistoryFit) -> (Vec<f64>, Vec<f64>) {
    let marks = fit.marks();
    let atoms = fit.rank();
    let mut loadings = Vec::with_capacity(marks * atoms);
    for d in 0..marks {
        for k in 0..atoms {
            loadings.push(fit.loadings[[d, k]]);
        }
    }
    (loadings, fit.log_rates.clone())
}

/// The observed history's nodes, `η⁰`, and the Laplace posterior at them.
fn observed_posterior(
    fit: &EventHistoryFit,
    cohort: &EventHistoryCohort,
    history: &SubjectHistory,
) -> Result<(CohortNodes, Vec<f64>, laplace::Smoother), EventHistoryError> {
    let nodes = single_subject_nodes(fit, cohort, history)?;
    let eta0 = node_eta0(fit, &nodes)?;
    let (loadings, log_rates) = latent_parameters(fit);
    let inputs = SubjectInputs {
        nodes: &nodes.subjects[0],
        eta0: &eta0,
        loadings: &loadings,
        log_rates: &log_rates,
        time_scale: fit.time_scale,
    };
    let mode = laplace::find_mode(&inputs, None)?;
    let smoother = laplace::smoother(&inputs, &mode)?;
    Ok((nodes, eta0, smoother))
}

/// The zero-count window a forecast runs over: its pseudo-subject (the
/// history's events carried as prior history so the risk sets are right)
/// and the state it continues from, with the time that state was smoothed
/// at (`None`: the stationary prior).
struct Window<'a> {
    subject: SubjectHistory,
    initial: Option<(&'a Gaussian, f64)>,
    /// Marks the forecast subject is at risk for at the window's start.
    at_risk: Vec<bool>,
}

fn validate_horizons(horizons: &[f64], start: f64) -> Result<(), EventHistoryError> {
    if horizons.is_empty() || horizons.windows(2).any(|w| !(w[1] > w[0])) || !(horizons[0] > start) {
        return Err(EventHistoryError::InvalidInput {
            reason: "horizons must be strictly increasing and later than the window's start".to_string(),
        });
    }
    Ok(())
}

/// Run every filter pass a window needs and assemble the per-mark risks.
fn window_forecast(
    fit: &EventHistoryFit,
    cohort: &EventHistoryCohort,
    window: Window<'_>,
    horizons: &[f64],
) -> Result<RiskForecast, EventHistoryError> {
    let marks = fit.marks();
    let kinds = &cohort.mark_kinds;
    let atoms = fit.rank();
    let mut future_cohort = EventHistoryCohort {
        mark_names: cohort.mark_names.clone(),
        mark_kinds: cohort.mark_kinds.clone(),
        covariate_names: cohort.covariate_names.clone(),
        covariates: cohort.covariates.clone(),
        subjects: vec![window.subject],
    };
    future_cohort.validate()?;
    let future = expand_nodes(&future_cohort, fit.quadrature_order)?;
    let eta_future = node_eta0(fit, &future)?;
    let future_nodes = &future.subjects[0];
    let (loadings, log_rates) = latent_parameters(fit);
    let inputs = SubjectInputs {
        nodes: future_nodes,
        eta0: &eta_future,
        loadings: &loadings,
        log_rates: &log_rates,
        time_scale: fit.time_scale,
    };
    let terminal: Vec<bool> = kinds.iter().map(|k| *k == MarkKind::Terminal).collect();
    // Under the compensator restricted to a set `C`: the survival of `C`,
    // the cumulative first-occurrence probability of every mark of `C`
    // (the node's event probability `1 − c_n` shared among the marks of `C`
    // in proportion to their expected exposure-weighted intensity, so the
    // first occurrences of `C` and its survival sum to one exactly), and
    // the expected count of every mark, `Σ w S(t⁻) E[λ]`.
    // The window's first node sits after the time the initial state was
    // smoothed at; the state is propagated across that gap.
    let continuation = window
        .initial
        .map(|(state, at)| (state, future_nodes.times[0] - at));
    let accumulate = |compensated: &[bool]| -> Result<(Vec<f64>, Array2<f64>, Array2<f64>), EventHistoryError> {
        let pass = filter_pass(&inputs, continuation, compensated)?;
        let mut survival = vec![0.0; horizons.len()];
        let mut first = Array2::<f64>::zeros((horizons.len(), marks));
        let mut expected = Array2::<f64>::zeros((horizons.len(), marks));
        let mut log_survival = 0.0_f64;
        let mut first_counts = vec![0.0; marks];
        let mut expected_counts = vec![0.0; marks];
        let mut horizon = 0;
        for n in 0..future_nodes.len() {
            let t = future_nodes.times[n];
            while horizon < horizons.len() && t > horizons[horizon] {
                survival[horizon] = log_survival.exp();
                for d in 0..marks {
                    first[[horizon, d]] = first_counts[d];
                    expected[[horizon, d]] = expected_counts[d];
                }
                horizon += 1;
            }
            let survival_before = log_survival.exp();
            let mut mass = vec![0.0; marks];
            let mut total = 0.0;
            for d in 0..marks {
                let w = future_nodes.exposures[[n, d]];
                if w == 0.0 {
                    continue;
                }
                let intensity = expected_intensity(
                    eta_future[n * marks + d],
                    &loadings[d * atoms..(d + 1) * atoms],
                    &pass.predicted[n],
                );
                mass[d] = w * intensity;
                expected_counts[d] += survival_before * mass[d];
                if compensated[d] {
                    total += mass[d];
                }
            }
            let event_probability = -pass.log_normalisers[n].exp_m1();
            if total > 0.0 {
                for d in 0..marks {
                    if compensated[d] {
                        first_counts[d] += survival_before * event_probability * mass[d] / total;
                    }
                }
            }
            log_survival += pass.log_normalisers[n];
        }
        while horizon < horizons.len() {
            survival[horizon] = log_survival.exp();
            for d in 0..marks {
                first[[horizon, d]] = first_counts[d];
                expected[[horizon, d]] = expected_counts[d];
            }
            horizon += 1;
        }
        Ok((survival, first, expected))
    };
    let (survival, first_terminal, expected_terminal) = accumulate(&terminal)?;
    let mut risk = Array2::<f64>::from_elem((horizons.len(), marks), f64::NAN);
    let mut expected = Array2::<f64>::from_elem((horizons.len(), marks), f64::NAN);
    for d in 0..marks {
        if !window.at_risk[d] {
            continue;
        }
        match kinds[d] {
            MarkKind::Terminal => {
                for h in 0..horizons.len() {
                    risk[[h, d]] = first_terminal[[h, d]];
                    expected[[h, d]] = first_terminal[[h, d]];
                }
            }
            MarkKind::Once | MarkKind::Recurrent => {
                let mut own = terminal.clone();
                own[d] = true;
                let (_, first, _) = accumulate(&own)?;
                for h in 0..horizons.len() {
                    risk[[h, d]] = first[[h, d]];
                    expected[[h, d]] = if kinds[d] == MarkKind::Once {
                        first[[h, d]]
                    } else {
                        expected_terminal[[h, d]]
                    };
                }
            }
        }
    }
    Ok(RiskForecast {
        horizons: horizons.to_vec(),
        risk,
        survival,
        expected_counts: expected,
    })
}

/// Forecast one subject beyond its observed exit at absolute `horizons`,
/// with covariate row `future_row` in force over the forecast window: the
/// history is smoothed into the latent state at exit, and the window
/// continues from there.
pub fn forecast(
    fit: &EventHistoryFit,
    cohort: &EventHistoryCohort,
    history: &SubjectHistory,
    horizons: &[f64],
    future_row: usize,
) -> Result<RiskForecast, EventHistoryError> {
    let marks = fit.marks();
    let kinds = &cohort.mark_kinds;
    if kinds.len() != marks {
        return Err(EventHistoryError::InvalidInput {
            reason: format!("cohort declares {} mark kinds for a fit with {marks} marks", kinds.len()),
        });
    }
    validate_horizons(horizons, history.exit)?;
    if future_row >= cohort.covariates.nrows() {
        return Err(EventHistoryError::InvalidInput {
            reason: format!(
                "future covariate row {future_row} is outside the {}-row table",
                cohort.covariates.nrows()
            ),
        });
    }
    if history
        .events
        .iter()
        .any(|e| kinds[e.mark] == MarkKind::Terminal)
    {
        return Err(EventHistoryError::InvalidInput {
            reason: format!("subject {:?} had a terminal event; there is nothing to forecast", history.id),
        });
    }
    let (observed, _, smoother) = observed_posterior(fit, cohort, history)?;
    let atoms = fit.rank();
    let last = observed.subjects[0].len() - 1;
    let last_time = observed.subjects[0].times[last];
    let state = smoother.at(last, atoms);
    // Every horizon is a breakpoint so the cumulative sums land on horizons.
    let mut subject = SubjectHistory {
        id: format!("{}::forecast", history.id),
        entry: history.exit,
        exit: horizons[horizons.len() - 1],
        events: history
            .events
            .iter()
            .map(|e| super::cohort::Event {
                time: history.exit.min(e.time),
                mark: e.mark,
            })
            .collect(),
        segments: vec![CovariateSegment {
            start: history.exit,
            row: future_row,
        }],
    };
    subject.segments.extend(horizons[..horizons.len() - 1].iter().map(|&t| CovariateSegment {
        start: t,
        row: future_row,
    }));
    let at_risk: Vec<bool> = (0..marks).map(|d| cohort.at_risk(history, d, history.exit)).collect();
    window_forecast(
        fit,
        cohort,
        Window {
            subject,
            initial: Some((&state, last_time)),
            at_risk,
        },
        horizons,
    )
}

/// Forecast a subject with no observed history from covariate values alone
/// (in the cohort's column order), over `(start, horizons]`: the latent
/// state starts at its stationary prior. With population covariate values
/// this is the population tier of the information hierarchy; with a
/// subject's own covariates (a risk score, say) it is what the model says
/// before any history has been observed. The subject is at risk for every
/// mark.
pub fn population_forecast(
    fit: &EventHistoryFit,
    cohort: &EventHistoryCohort,
    covariates: &[f64],
    start: f64,
    horizons: &[f64],
) -> Result<RiskForecast, EventHistoryError> {
    let marks = fit.marks();
    if covariates.len() != cohort.covariates.ncols() {
        return Err(EventHistoryError::InvalidInput {
            reason: format!(
                "population forecast needs {} covariate values, got {}",
                cohort.covariates.ncols(),
                covariates.len()
            ),
        });
    }
    if covariates.iter().any(|v| !v.is_finite()) {
        return Err(EventHistoryError::InvalidInput {
            reason: "population forecast covariates must be finite".to_string(),
        });
    }
    validate_horizons(horizons, start)?;
    // A covariate table with the given values as its one row.
    let mut one_row = cohort.clone();
    one_row.subjects.clear();
    one_row.covariates = Array2::from_shape_vec((1, covariates.len()), covariates.to_vec())
        .map_err(|e| EventHistoryError::InvalidInput { reason: e.to_string() })?;
    let mut subject = SubjectHistory {
        id: "population".to_string(),
        entry: start,
        exit: horizons[horizons.len() - 1],
        events: Vec::new(),
        segments: vec![CovariateSegment { start, row: 0 }],
    };
    subject.segments.extend(horizons[..horizons.len() - 1].iter().map(|&t| CovariateSegment {
        start: t,
        row: 0,
    }));
    let window = Window {
        subject,
        initial: None,
        at_risk: vec![true; marks],
    };
    window_forecast(fit, &one_row, window, horizons)
}

/// Predictive PIT of every observed event of a subject, in time order.
pub fn predictive_pit(
    fit: &EventHistoryFit,
    cohort: &EventHistoryCohort,
    history: &SubjectHistory,
) -> Result<Vec<f64>, EventHistoryError> {
    let marks = fit.marks();
    let (loadings, log_rates) = latent_parameters(fit);
    let nodes = single_subject_nodes(fit, cohort, history)?;
    let eta0 = node_eta0(fit, &nodes)?;
    let subject = &nodes.subjects[0];
    let pass = filter_pass(
        &SubjectInputs {
            nodes: subject,
            eta0: &eta0,
            loadings: &loadings,
            log_rates: &log_rates,
            time_scale: fit.time_scale,
        },
        None,
        &vec![true; marks],
    )?;
    let mut pits = Vec::new();
    let mut log_survival: f64 = 0.0;
    for n in 0..subject.len() {
        let is_event = (0..marks).any(|d| subject.counts[[n, d]] > 0.0);
        if is_event {
            let events = (0..marks)
                .map(|d| subject.counts[[n, d]])
                .sum::<f64>()
                .round() as usize;
            let pit = 1.0 - log_survival.exp();
            for _ in 0..events.max(1) {
                pits.push(pit);
            }
            log_survival = 0.0;
        } else {
            log_survival += pass.log_normalisers[n];
        }
    }
    Ok(pits)
}

/// The smoothed latent state of one subject on its own nodes: the Laplace
/// posterior given the whole history.
pub fn latent_state(
    fit: &EventHistoryFit,
    cohort: &EventHistoryCohort,
    history: &SubjectHistory,
) -> Result<LatentPath, EventHistoryError> {
    let (nodes, _, smoother) = observed_posterior(fit, cohort, history)?;
    let atoms = fit.rank();
    let subject = &nodes.subjects[0];
    let n = subject.len();
    let mut means = Array2::<f64>::zeros((n, atoms));
    for node in 0..n {
        for k in 0..atoms {
            means[[node, k]] = smoother.means[node * atoms + k];
        }
    }
    Ok(LatentPath {
        times: subject.times.clone(),
        means,
        covariances: smoother.covariances,
    })
}

/// The follow-up average of one subject's latent state as a posterior
/// Gaussian: mean per atom and `atoms × atoms` covariance (row-major).
pub fn latent_exposure(
    fit: &EventHistoryFit,
    cohort: &EventHistoryCohort,
    history: &SubjectHistory,
) -> Result<(Vec<f64>, Vec<f64>), EventHistoryError> {
    let nodes = single_subject_nodes(fit, cohort, history)?;
    let eta0 = node_eta0(fit, &nodes)?;
    let (loadings, log_rates) = latent_parameters(fit);
    let inputs = SubjectInputs {
        nodes: &nodes.subjects[0],
        eta0: &eta0,
        loadings: &loadings,
        log_rates: &log_rates,
        time_scale: fit.time_scale,
    };
    let mode = laplace::find_mode(&inputs, None)?;
    let Gaussian { mean, cov } = laplace::exposure(&inputs, &mode)?;
    Ok((mean, cov))
}

/// Kolmogorov–Smirnov distance of a PIT sample from the uniform law.
pub fn kolmogorov_smirnov_uniform(pits: &[f64]) -> f64 {
    if pits.is_empty() {
        return 0.0;
    }
    let mut sorted = pits.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let n = sorted.len() as f64;
    let mut distance = 0.0_f64;
    for (i, &u) in sorted.iter().enumerate() {
        let lower = i as f64 / n;
        let upper = (i + 1) as f64 / n;
        distance = distance.max((u - lower).abs()).max((upper - u).abs());
    }
    distance
}
