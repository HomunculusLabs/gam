//! History-conditioned forecasting, the predictive probability integral
//! transform, and the smoothed latent state of a subject.
//!
//! A forecast window runs the sequential Gaussian filter over future
//! quadrature nodes with zero counts, continued either from the smoothed
//! state at a subject's exit or from the stationary prior at given
//! covariate values. With the compensator restricted to a set of marks `C`,
//! the product of the normalisers up to a node is `P(no event of any mark
//! in C before it | history)`, and each node's event probability shared
//! among the marks of `C` in proportion to their expected exposure-weighted
//! intensity accumulates to the probability that the first event of a mark
//! happens by the horizon before any other mark of `C`. Every mark's risk
//! therefore uses `C = {d} ∪ terminal marks`: a first occurrence of `d` by
//! the horizon while the subject is still alive. Risk-set membership is
//! data, so a once-only mark the subject already has carries no risk.
//!
//! The three tiers of a forecast are three conditionings of one model, with
//! no weight between them chosen: the stationary prior at population
//! covariate values, the same filter at a subject's own covariates, and the
//! filter continued from the subject's smoothed state.
//!
//! The predictive PIT of an event is `1 − P(no event of any mark at risk in
//! (t_prev, t_event] | history)`, which the same filter yields as the
//! product of the normalisers of the zero-count nodes between consecutive
//! events; it is uniform when the model is right.

use super::cohort::{
    CohortNodes, CovariateSegment, Event, EventHistoryCohort, EventHistoryError, MarkKind,
    SubjectHistory, expand_nodes,
};
use super::family::EventHistoryFit;
use super::laplace::{self, Gaussian, SubjectInputs, expected_intensity, filter_pass};
use gam_terms::smooth::build_term_collection_design;
use ndarray::{Array1, Array2, ArrayView2};

/// One piece of a forecast window during which the covariates are constant:
/// from `start` until the next segment's `start` (or the last horizon).
#[derive(Clone, Debug, PartialEq)]
pub struct FutureSegment {
    pub start: f64,
    /// Covariate values in the cohort's column order.
    pub covariates: Vec<f64>,
}

/// A forecast request for a subject whose history the model has seen.
pub struct ForecastRequest<'a> {
    /// The subject's observed history.
    pub history: &'a SubjectHistory,
    /// Absolute horizon times, strictly increasing and after the exit time.
    pub horizons: &'a [f64],
    /// The covariate path over the forecast window; the first segment must
    /// start at or before the subject's exit.
    pub future: &'a [FutureSegment],
}

/// A forecast request for a subject with no observed history: the latent
/// state starts at its stationary prior. With population covariate values
/// this is the population tier of the information hierarchy; with a
/// subject's own covariates it is what the model says before any history.
pub struct PopulationForecastRequest<'a> {
    /// Time the forecast window opens.
    pub start: f64,
    /// Absolute horizon times, strictly increasing and after `start`.
    pub horizons: &'a [f64],
    /// The covariate path over the window; the first segment must start at
    /// or before `start`.
    pub future: &'a [FutureSegment],
}

/// A forecast: per horizon, the first-occurrence risk of every mark, the
/// survival of the terminal marks, and the expected count of every mark.
#[derive(Clone, Debug)]
pub struct Forecast {
    pub horizons: Vec<f64>,
    /// `horizons × marks`: `P(first event of d in the window by the horizon,
    /// before any terminal event | history)`. `NaN` for a mark the subject
    /// is not at risk for.
    pub risk: Array2<f64>,
    /// `P(no terminal event by the horizon | history)`.
    pub survival: Vec<f64>,
    /// `horizons × marks`: the expected number of events of the mark before
    /// any terminal event. Equals `risk` for once-only and terminal marks.
    pub expected_counts: Array2<f64>,
}

/// The predictive PIT of one observed event, with the model's probability
/// that the event carried each mark.
#[derive(Clone, Debug)]
pub struct EventPit {
    pub time: f64,
    pub mark: usize,
    /// `1 − P(no event of any mark at risk since the previous event)`.
    pub pit: f64,
    /// Per mark, the probability that an event at this instant is of that
    /// mark, in proportion to the expected exposure-weighted intensities.
    pub mark_probabilities: Vec<f64>,
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
        covariate_levels: cohort.covariate_levels.clone(),
        covariates: cohort.covariates.clone(),
        subjects: vec![history.clone()],
    };
    one.validate()?;
    expand_nodes(&one, fit.quadrature_order, fit.mesh_refinement)
}

/// Node log-intensities `η⁰` (design × coefficients + offset) for every mark
/// on a row matrix (covariate columns then time), index `row * marks + d`.
fn node_eta0(fit: &EventHistoryFit, rows: ArrayView2<'_, f64>) -> Result<Vec<f64>, EventHistoryError> {
    let marks = fit.marks();
    let total = rows.nrows();
    let mut eta0 = vec![0.0; total * marks];
    for d in 0..marks {
        let design = build_term_collection_design(rows, &fit.frozen_specs[d]).map_err(|error| {
            EventHistoryError::Fit {
                reason: format!("prediction design for mark {d}: {error}"),
            }
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

fn validate_window(
    fit: &EventHistoryFit,
    horizons: &[f64],
    future: &[FutureSegment],
    start: f64,
) -> Result<(), EventHistoryError> {
    if horizons.is_empty()
        || horizons.windows(2).any(|w| !(w[1] > w[0]))
        || !(horizons[0] > start)
    {
        return Err(EventHistoryError::InvalidInput {
            reason: "horizons must be strictly increasing and later than the window's start".to_string(),
        });
    }
    if future.is_empty() {
        return Err(EventHistoryError::InvalidInput {
            reason: "a forecast window needs at least one covariate segment".to_string(),
        });
    }
    let width = fit.nodes.node_data.ncols() - 1;
    for segment in future {
        if segment.covariates.len() != width {
            return Err(EventHistoryError::InvalidInput {
                reason: format!(
                    "a future segment carries {} covariate values, the fit has {width}",
                    segment.covariates.len()
                ),
            });
        }
        if !segment.start.is_finite() || segment.covariates.iter().any(|v| !v.is_finite()) {
            return Err(EventHistoryError::InvalidInput {
                reason: "a future segment must carry finite values".to_string(),
            });
        }
    }
    if !(future[0].start <= start) {
        return Err(EventHistoryError::InvalidInput {
            reason: format!(
                "the first future segment starts at {} but the window opens at {start}",
                future[0].start
            ),
        });
    }
    Ok(())
}

/// The window's node expansion and its per-node `η⁰`.
///
/// Every horizon is a breakpoint of the pseudo-subject, so the cumulative
/// sums land exactly on the horizons; the covariate path is the caller's
/// future segments, and the subject's own prior events are carried so the
/// risk sets are the ones it actually faces.
fn window_nodes(
    fit: &EventHistoryFit,
    cohort: &EventHistoryCohort,
    start: f64,
    horizons: &[f64],
    future: &[FutureSegment],
    prior_events: &[Event],
    label: &str,
) -> Result<(CohortNodes, Vec<f64>), EventHistoryError> {
    let width = future[0].covariates.len();
    let mut table = Array2::<f64>::zeros((future.len(), width));
    for (i, segment) in future.iter().enumerate() {
        for (j, v) in segment.covariates.iter().enumerate() {
            table[[i, j]] = *v;
        }
    }
    let mut segments: Vec<CovariateSegment> = future
        .iter()
        .enumerate()
        .map(|(row, segment)| CovariateSegment {
            start: segment.start.max(start).min(if row == 0 { start } else { f64::INFINITY }),
            row,
        })
        .collect();
    segments[0].start = start;
    for (row, segment) in future.iter().enumerate().skip(1) {
        segments[row].start = segment.start;
    }
    segments.retain(|s| s.row == 0 || (s.start > start && s.start < horizons[horizons.len() - 1]));
    // Every horizon but the last is a breakpoint too, under the covariate
    // row in force there.
    let mut extra: Vec<CovariateSegment> = Vec::new();
    for &t in &horizons[..horizons.len() - 1] {
        let row = segments
            .iter()
            .rev()
            .find(|s| s.start <= t)
            .map(|s| s.row)
            .unwrap_or(0);
        extra.push(CovariateSegment { start: t, row });
    }
    segments.extend(extra);
    segments.sort_by(|a, b| a.start.total_cmp(&b.start));
    segments.dedup_by(|a, b| a.start == b.start);
    let subject = SubjectHistory {
        id: format!("{label}::window"),
        entry: start,
        exit: horizons[horizons.len() - 1],
        events: prior_events
            .iter()
            .map(|e| Event {
                time: start.min(e.time),
                mark: e.mark,
            })
            .collect(),
        segments,
    };
    let mut window = EventHistoryCohort {
        mark_names: cohort.mark_names.clone(),
        mark_kinds: cohort.mark_kinds.clone(),
        covariate_names: cohort.covariate_names.clone(),
        covariate_levels: cohort.covariate_levels.clone(),
        covariates: table,
        subjects: vec![subject],
    };
    window.validate()?;
    let nodes = expand_nodes(&window, fit.quadrature_order, fit.mesh_refinement)?;
    let eta0 = node_eta0(fit, nodes.node_data.view())?;
    Ok((nodes, eta0))
}

/// Run every filter pass a window needs and assemble the per-mark risks.
fn window_forecast(
    fit: &EventHistoryFit,
    cohort: &EventHistoryCohort,
    nodes: &CohortNodes,
    eta0: &[f64],
    initial: Option<(&Gaussian, f64)>,
    at_risk: &[bool],
    horizons: &[f64],
) -> Result<Forecast, EventHistoryError> {
    let marks = fit.marks();
    let kinds = &cohort.mark_kinds;
    let atoms = fit.rank();
    let window_nodes = &nodes.subjects[0];
    let (loadings, log_rates) = latent_parameters(fit);
    let inputs = SubjectInputs {
        nodes: window_nodes,
        eta0,
        loadings: &loadings,
        log_rates: &log_rates,
        time_scale: fit.time_scale,
    };
    // The window's first node sits after the time the initial state was
    // smoothed at; the state is propagated across that gap.
    let continuation = initial.map(|(state, at)| (state, window_nodes.times[0] - at));
    let terminal: Vec<bool> = kinds.iter().map(|k| *k == MarkKind::Terminal).collect();
    let accumulate = |compensated: &[bool]| -> Result<(Vec<f64>, Array2<f64>, Array2<f64>), EventHistoryError> {
        let pass = filter_pass(&inputs, continuation, compensated)?;
        let mut survival = vec![0.0; horizons.len()];
        let mut first = Array2::<f64>::zeros((horizons.len(), marks));
        let mut expected = Array2::<f64>::zeros((horizons.len(), marks));
        let mut log_survival = 0.0_f64;
        let mut first_counts = vec![0.0; marks];
        let mut expected_counts = vec![0.0; marks];
        let mut horizon = 0;
        for n in 0..window_nodes.len() {
            let t = window_nodes.times[n];
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
                let w = window_nodes.exposures[[n, d]];
                if w == 0.0 {
                    continue;
                }
                let intensity = expected_intensity(
                    eta0[n * marks + d],
                    &loadings[d * atoms..(d + 1) * atoms],
                    &pass.predicted[n],
                );
                mass[d] = w * intensity;
                expected_counts[d] += survival_before * mass[d];
                if compensated[d] {
                    total += mass[d];
                }
            }
            // The node's own event probability, shared among the
            // compensated marks by their expected intensity, so the first
            // occurrences and the survival of that set sum to one exactly.
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
        if !at_risk[d] {
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
    Ok(Forecast {
        horizons: horizons.to_vec(),
        risk,
        survival,
        expected_counts: expected,
    })
}

/// The Laplace posterior of one subject's latent path at the fit.
fn observed_posterior(
    fit: &EventHistoryFit,
    cohort: &EventHistoryCohort,
    history: &SubjectHistory,
) -> Result<(CohortNodes, laplace::Smoother), EventHistoryError> {
    let nodes = single_subject_nodes(fit, cohort, history)?;
    let eta0 = node_eta0(fit, nodes.node_data.view())?;
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
    Ok((nodes, smoother))
}

/// Forecast one subject beyond its observed exit: its history is smoothed
/// into its latent state, and the window continues from there.
pub fn forecast(
    fit: &EventHistoryFit,
    cohort: &EventHistoryCohort,
    request: &ForecastRequest<'_>,
) -> Result<Forecast, EventHistoryError> {
    let marks = fit.marks();
    let kinds = &cohort.mark_kinds;
    if kinds.len() != marks {
        return Err(EventHistoryError::InvalidInput {
            reason: format!("cohort declares {} mark kinds for a fit with {marks} marks", kinds.len()),
        });
    }
    let history = request.history;
    validate_window(fit, request.horizons, request.future, history.exit)?;
    if history.terminal_event(kinds).is_some() {
        return Err(EventHistoryError::InvalidInput {
            reason: format!("subject {:?} had a terminal event; there is nothing to forecast", history.id),
        });
    }
    let (observed, smoother) = observed_posterior(fit, cohort, history)?;
    let atoms = fit.rank();
    let last = observed.subjects[0].len() - 1;
    let last_time = observed.subjects[0].times[last];
    let state = smoother.at(last, atoms);
    let (nodes, eta0) = window_nodes(
        fit,
        cohort,
        history.exit,
        request.horizons,
        request.future,
        &history.events,
        &history.id,
    )?;
    let at_risk: Vec<bool> = (0..marks).map(|d| history.at_risk(d, history.exit, kinds)).collect();
    window_forecast(
        fit,
        cohort,
        &nodes,
        &eta0,
        Some((&state, last_time)),
        &at_risk,
        request.horizons,
    )
}

/// Forecast a subject with no observed history from its covariates alone:
/// the latent state starts at its stationary prior.
pub fn population_forecast(
    fit: &EventHistoryFit,
    cohort: &EventHistoryCohort,
    request: &PopulationForecastRequest<'_>,
) -> Result<Forecast, EventHistoryError> {
    let marks = fit.marks();
    validate_window(fit, request.horizons, request.future, request.start)?;
    let (nodes, eta0) = window_nodes(
        fit,
        cohort,
        request.start,
        request.horizons,
        request.future,
        &[],
        "population",
    )?;
    window_forecast(fit, cohort, &nodes, &eta0, None, &vec![true; marks], request.horizons)
}

/// Predictive PIT of every observed event of a subject, in time order.
pub fn predictive_pit(
    fit: &EventHistoryFit,
    cohort: &EventHistoryCohort,
    history: &SubjectHistory,
) -> Result<Vec<EventPit>, EventHistoryError> {
    let marks = fit.marks();
    let atoms = fit.rank();
    let (loadings, log_rates) = latent_parameters(fit);
    let nodes = single_subject_nodes(fit, cohort, history)?;
    let eta0 = node_eta0(fit, nodes.node_data.view())?;
    let subject = &nodes.subjects[0];
    let inputs = SubjectInputs {
        nodes: subject,
        eta0: &eta0,
        loadings: &loadings,
        log_rates: &log_rates,
        time_scale: fit.time_scale,
    };
    let pass = filter_pass(&inputs, None, &vec![true; marks])?;
    let mut pits = Vec::new();
    let mut log_survival: f64 = 0.0;
    for n in 0..subject.len() {
        let counts: Vec<f64> = (0..marks).map(|d| subject.counts[[n, d]]).collect();
        if counts.iter().any(|c| *c > 0.0) {
            let mut mass = vec![0.0; marks];
            let mut total = 0.0;
            for d in 0..marks {
                let w = subject.exposures[[n, d]];
                if w == 0.0 {
                    continue;
                }
                mass[d] = w * expected_intensity(
                    eta0[n * marks + d],
                    &loadings[d * atoms..(d + 1) * atoms],
                    &pass.predicted[n],
                );
                total += mass[d];
            }
            let probabilities: Vec<f64> = if total > 0.0 {
                mass.iter().map(|m| m / total).collect()
            } else {
                vec![f64::NAN; marks]
            };
            let pit = 1.0 - log_survival.exp();
            for (d, count) in counts.iter().enumerate() {
                for _ in 0..count.round() as usize {
                    pits.push(EventPit {
                        time: subject.times[n],
                        mark: d,
                        pit,
                        mark_probabilities: probabilities.clone(),
                    });
                }
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
    let (nodes, smoother) = observed_posterior(fit, cohort, history)?;
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
    let eta0 = node_eta0(fit, nodes.node_data.view())?;
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

/// Kolmogorov-Smirnov distance of a PIT sample from the uniform law, or
/// `None` when there are no events to test.
pub fn kolmogorov_smirnov_uniform(pits: &[f64]) -> Option<f64> {
    if pits.is_empty() {
        return None;
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
    Some(distance)
}

/// The fitted per-mark population log-intensity on the training nodes.
pub fn training_eta(fit: &EventHistoryFit) -> Array2<f64> {
    let marks = fit.marks();
    let total = fit.nodes.total_nodes;
    let mut out = Array2::<f64>::zeros((total, marks));
    for d in 0..marks {
        let eta: &Array1<f64> = fit.mark_eta(d);
        for row in 0..total {
            out[[row, d]] = eta[row];
        }
    }
    out
}
