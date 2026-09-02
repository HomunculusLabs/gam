//! History-conditioned forecasting and the predictive probability integral
//! transform, both exact expectations under the filtered latent state.
//!
//! A forecast runs the forward filter over the observed history, then over
//! future quadrature nodes with zero counts and the compensator restricted to
//! the absorbing marks. The product of the future normalisers is the
//! probability that no absorbing event has fired by a horizon; the expected
//! count of any mark by a horizon accumulates `w · S(t⁻) · E_pred[λ_d(t)]`
//! over the future nodes. For an absorbing mark this is its cumulative
//! incidence.
//!
//! The predictive PIT of an event is `1 − P(no event of any mark in
//! (t_prev, t_event] | history)`, which the same filter yields as the product
//! of the normalisers of the zero-count nodes between consecutive events; it
//! is uniform when the model is right.

use super::cohort::{
    CohortNodes, CovariateSegment, EventHistoryCohort, EventHistoryError, SubjectHistory,
    expand_nodes,
};
use super::family::EventHistoryFit;
use super::marginal::{SubjectInputs, forward_filter};
use gam_terms::smooth::build_term_collection_design;
use ndarray::{Array1, Array2};

/// A forecast request for one subject.
pub struct ForecastRequest<'a> {
    /// The subject's observed history (its covariate rows index `cohort`).
    pub history: &'a SubjectHistory,
    /// Absolute horizon times, strictly increasing and after the exit time.
    pub horizons: &'a [f64],
    /// Which marks end follow-up when they fire.
    pub absorbing: &'a [bool],
    /// Covariate row in force over the forecast window.
    pub future_row: usize,
}

/// A forecast: per horizon, the probability that no absorbing event has
/// fired, and the expected count of every mark (its cumulative incidence
/// when the mark is absorbing).
#[derive(Clone, Debug)]
pub struct Forecast {
    pub horizons: Vec<f64>,
    pub survival: Vec<f64>,
    pub expected_counts: Array2<f64>,
}

fn single_subject_nodes(
    fit: &EventHistoryFit,
    cohort: &EventHistoryCohort,
    history: &SubjectHistory,
) -> Result<CohortNodes, EventHistoryError> {
    let mut one = EventHistoryCohort {
        mark_names: cohort.mark_names.clone(),
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
    let atoms = fit.atoms();
    let mut loadings = Vec::with_capacity(marks * atoms);
    for d in 0..marks {
        for k in 0..atoms {
            loadings.push(fit.loadings[[d, k]]);
        }
    }
    (loadings, fit.log_rates.clone())
}

/// Forecast one subject beyond its observed exit.
pub fn forecast(
    fit: &EventHistoryFit,
    cohort: &EventHistoryCohort,
    request: &ForecastRequest<'_>,
) -> Result<Forecast, EventHistoryError> {
    let marks = fit.marks();
    if request.absorbing.len() != marks {
        return Err(EventHistoryError::InvalidInput {
            reason: format!(
                "absorbing mask has {} entries for {marks} marks",
                request.absorbing.len()
            ),
        });
    }
    if request.horizons.is_empty()
        || request
            .horizons
            .windows(2)
            .any(|w| !(w[1] > w[0]))
        || !(request.horizons[0] > request.history.exit)
    {
        return Err(EventHistoryError::InvalidInput {
            reason: "horizons must be strictly increasing and later than the exit time".to_string(),
        });
    }
    if request.future_row >= cohort.covariates.nrows() {
        return Err(EventHistoryError::InvalidInput {
            reason: format!(
                "future covariate row {} is outside the {}-row table",
                request.future_row,
                cohort.covariates.nrows()
            ),
        });
    }
    let (loadings, log_rates) = latent_parameters(fit);
    let gh = fit.family.gauss_hermite();
    // Observed history.
    let observed = single_subject_nodes(fit, cohort, request.history)?;
    let eta_observed = node_eta0(fit, &observed)?;
    let all_marks = vec![true; marks];
    let observed_pass = forward_filter(
        &SubjectInputs {
            nodes: &observed.subjects[0],
            eta0: &eta_observed,
            loadings: &loadings,
            log_rates: &log_rates,
            time_scale: fit.time_scale,
            gh,
            continuation_gap: 0.0,
        },
        None,
        &all_marks,
    )?;
    let last = observed.subjects[0].len() - 1;
    let last_time = observed.subjects[0].times[last];
    // Future nodes: one pseudo-subject over (exit, last horizon] with no events.
    let future_history = SubjectHistory {
        id: format!("{}::forecast", request.history.id),
        entry: request.history.exit,
        exit: request.horizons[request.horizons.len() - 1],
        events: Vec::new(),
        segments: vec![CovariateSegment {
            start: request.history.exit,
            row: request.future_row,
        }],
    };
    let mut future_cohort = EventHistoryCohort {
        mark_names: cohort.mark_names.clone(),
        covariate_names: cohort.covariate_names.clone(),
        covariates: cohort.covariates.clone(),
        subjects: vec![future_history],
    };
    // Every horizon is a breakpoint so the cumulative sums land on horizons.
    let mut breakpoints: Vec<f64> = request.horizons.to_vec();
    breakpoints.pop();
    future_cohort.subjects[0].segments.extend(breakpoints.iter().map(|&t| CovariateSegment {
        start: t,
        row: request.future_row,
    }));
    future_cohort.validate()?;
    let future = expand_nodes(&future_cohort, fit.quadrature_order)?;
    let eta_future = node_eta0(fit, &future)?;
    let future_nodes = &future.subjects[0];
    let pass = forward_filter(
        &SubjectInputs {
            nodes: future_nodes,
            eta0: &eta_future,
            loadings: &loadings,
            log_rates: &log_rates,
            time_scale: fit.time_scale,
            gh,
            continuation_gap: future_nodes.times[0] - last_time,
        },
        Some((&observed_pass.grids[last], &observed_pass.alpha[last])),
        request.absorbing,
    )?;
    let atoms = fit.atoms();
    let mut survival = vec![0.0; request.horizons.len()];
    let mut expected = Array2::<f64>::zeros((request.horizons.len(), marks));
    let mut log_survival: f64 = 0.0;
    let mut counts = vec![0.0; marks];
    let mut horizon = 0;
    for n in 0..future_nodes.len() {
        let t = future_nodes.times[n];
        while horizon < request.horizons.len() && t > request.horizons[horizon] {
            survival[horizon] = log_survival.exp();
            for d in 0..marks {
                expected[[horizon, d]] = counts[d];
            }
            horizon += 1;
        }
        let w = future_nodes.exposures[n];
        let grid = &pass.grids[n];
        let predicted = &pass.predicted[n];
        let survival_before = log_survival.exp();
        for d in 0..marks {
            let mut intensity = 0.0;
            for i in 0..grid.size() {
                let mut eta = eta_future[n * marks + d];
                for k in 0..atoms {
                    eta += loadings[d * atoms + k] * grid.coordinate(i, k);
                }
                intensity += grid.weights[i] * predicted[i] * eta.exp();
            }
            counts[d] += w * survival_before * intensity;
        }
        log_survival += pass.log_normalisers[n];
    }
    while horizon < request.horizons.len() {
        survival[horizon] = log_survival.exp();
        for d in 0..marks {
            expected[[horizon, d]] = counts[d];
        }
        horizon += 1;
    }
    Ok(Forecast {
        horizons: request.horizons.to_vec(),
        survival,
        expected_counts: expected,
    })
}

/// Predictive PIT of every event of a subject, in time order.
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
    let pass = forward_filter(
        &SubjectInputs {
            nodes: subject,
            eta0: &eta0,
            loadings: &loadings,
            log_rates: &log_rates,
            time_scale: fit.time_scale,
            gh: fit.family.gauss_hermite(),
            continuation_gap: 0.0,
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

/// Convenience: the fitted per-mark linear predictor on the training nodes.
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
