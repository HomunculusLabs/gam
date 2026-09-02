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
//!
//! The same window run from the stationary prior instead of a filtered state
//! ([`population_forecast`]) gives the lower tiers of the information
//! hierarchy: with population covariate values it is the population risk;
//! with a subject's own covariates (a risk score) it is what the model says
//! before any history is observed; the history-conditioned forecast updates
//! it. No weight between the tiers is chosen by hand — each is the same
//! probability model conditioned on more.

use super::chain::Grid;
use super::cohort::{
    CohortNodes, CovariateSegment, EventHistoryCohort, EventHistoryError, SubjectHistory,
    expand_nodes,
};
use super::family::EventHistoryFit;
use super::marginal::{SubjectInputs, forward_filter, log_intensity};
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

/// A forecast for a subject with no observed history: the forward filter
/// starts from the stationary prior of the latent state at `start` and sees
/// only the covariates, so this is the population tier when the covariates
/// are population values, and the covariate-only tier (a new subject with a
/// risk score but no history) otherwise. A subject's own forecast,
/// [`forecast`], is the same window started from its filtered state.
pub struct PopulationForecastRequest<'a> {
    /// Time the forecast window opens.
    pub start: f64,
    /// Absolute horizon times, strictly increasing and after `start`.
    pub horizons: &'a [f64],
    /// Which marks end follow-up when they fire.
    pub absorbing: &'a [bool],
    /// Covariate values in force over the window, in the cohort's column
    /// order.
    pub covariates: &'a [f64],
}

/// The latent state a forecast window continues from: the filtered density
/// on its grid at the time of the last observed node.
struct FilteredState<'a> {
    grid: &'a Grid<f64>,
    density: &'a [f64],
    time: f64,
}

/// One zero-count window: a covariate table with the row in force, the
/// window's start, its horizons, the absorbing mask, and the state it
/// continues from (a filtered state, or the stationary prior when `None`).
struct Window<'a> {
    covariates: Array2<f64>,
    row: usize,
    id: &'a str,
    start: f64,
    horizons: &'a [f64],
    absorbing: &'a [bool],
    initial: Option<FilteredState<'a>>,
}

/// Run the zero-count forward filter over a window and read off survival
/// and expected counts at its horizons.
fn forecast_window(
    fit: &EventHistoryFit,
    cohort: &EventHistoryCohort,
    window: Window<'_>,
) -> Result<Forecast, EventHistoryError> {
    let Window {
        covariates,
        row,
        id,
        start,
        horizons,
        absorbing,
        initial,
    } = window;
    let (mark_names, covariate_names) = (&cohort.mark_names, &cohort.covariate_names);
    let marks = fit.marks();
    if absorbing.len() != marks {
        return Err(EventHistoryError::InvalidInput {
            reason: format!("absorbing mask has {} entries for {marks} marks", absorbing.len()),
        });
    }
    if horizons.is_empty() || horizons.windows(2).any(|w| !(w[1] > w[0])) || !(horizons[0] > start)
    {
        return Err(EventHistoryError::InvalidInput {
            reason: format!(
                "horizons must be strictly increasing and later than the window start {start}"
            ),
        });
    }
    let (loadings, log_rates) = latent_parameters(fit);
    let gh = fit.family.gauss_hermite();
    // Future nodes: one pseudo-subject over (start, last horizon] with no
    // events; every horizon is a breakpoint so the cumulative sums land on
    // horizons.
    let mut segments = vec![CovariateSegment { start, row }];
    segments.extend(horizons[..horizons.len() - 1].iter().map(|&t| CovariateSegment { start: t, row }));
    let mut future_cohort = EventHistoryCohort {
        mark_names: mark_names.to_vec(),
        covariate_names: covariate_names.to_vec(),
        covariates,
        subjects: vec![SubjectHistory {
            id: id.to_string(),
            entry: start,
            exit: horizons[horizons.len() - 1],
            events: Vec::new(),
            segments,
        }],
    };
    future_cohort.validate()?;
    let future = expand_nodes(&future_cohort, fit.quadrature_order)?;
    let eta_future = node_eta0(fit, &future)?;
    let future_nodes = &future.subjects[0];
    let continuation_gap = initial
        .as_ref()
        .map_or(0.0, |state| future_nodes.times[0] - state.time);
    let pass = forward_filter(
        &SubjectInputs {
            nodes: future_nodes,
            eta0: &eta_future,
            loadings: &loadings,
            log_rates: &log_rates,
            time_scale: fit.time_scale,
            gh,
            continuation_gap,
        },
        initial.as_ref().map(|state| (state.grid, state.density)),
        absorbing,
    )?;
    let atoms = fit.atoms();
    let mut survival = vec![0.0; horizons.len()];
    let mut expected = Array2::<f64>::zeros((horizons.len(), marks));
    let mut log_survival: f64 = 0.0;
    let mut counts = vec![0.0; marks];
    let mut horizon = 0;
    for n in 0..future_nodes.len() {
        let t = future_nodes.times[n];
        while horizon < horizons.len() && t > horizons[horizon] {
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
        let mut z = vec![0.0; atoms];
        for d in 0..marks {
            let loadings_d = &loadings[d * atoms..(d + 1) * atoms];
            let mut intensity = 0.0;
            for i in 0..grid.size() {
                for (k, zk) in z.iter_mut().enumerate() {
                    *zk = *grid.coordinate(i, k);
                }
                let eta = log_intensity(&eta_future[n * marks + d], loadings_d, &z);
                intensity += grid.weights[i] * predicted[i] * eta.exp();
            }
            counts[d] += w * survival_before * intensity;
        }
        log_survival += pass.log_normalisers[n];
    }
    while horizon < horizons.len() {
        survival[horizon] = log_survival.exp();
        for d in 0..marks {
            expected[[horizon, d]] = counts[d];
        }
        horizon += 1;
    }
    Ok(Forecast {
        horizons: horizons.to_vec(),
        survival,
        expected_counts: expected,
    })
}

/// Forecast one subject beyond its observed exit: its history is filtered
/// into its latent state, and the window continues from there.
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
            gh: fit.family.gauss_hermite(),
            continuation_gap: 0.0,
        },
        None,
        &all_marks,
    )?;
    let last = observed.subjects[0].len() - 1;
    let id = format!("{}::forecast", request.history.id);
    forecast_window(
        fit,
        cohort,
        Window {
            covariates: cohort.covariates.clone(),
            row: request.future_row,
            id: &id,
            start: request.history.exit,
            horizons: request.horizons,
            absorbing: request.absorbing,
            initial: Some(FilteredState {
                grid: &observed_pass.grids[last],
                density: &observed_pass.alpha[last],
                time: observed.subjects[0].times[last],
            }),
        },
    )
}

/// Forecast a subject with no observed history from its covariates alone:
/// the latent state starts at its stationary prior. With population
/// covariate values this is the population tier of the information
/// hierarchy; with a subject's own covariates (a risk score, say) it is what
/// the model says before any history has been observed.
pub fn population_forecast(
    fit: &EventHistoryFit,
    cohort: &EventHistoryCohort,
    request: &PopulationForecastRequest<'_>,
) -> Result<Forecast, EventHistoryError> {
    let width = cohort.covariate_names.len();
    if request.covariates.len() != width {
        return Err(EventHistoryError::InvalidInput {
            reason: format!(
                "{} covariate values for a cohort with {width} covariate columns",
                request.covariates.len()
            ),
        });
    }
    let mut covariates = Array2::<f64>::zeros((1, width));
    for (j, &v) in request.covariates.iter().enumerate() {
        covariates[[0, j]] = v;
    }
    forecast_window(
        fit,
        cohort,
        Window {
            covariates,
            row: 0,
            id: "population::forecast",
            start: request.start,
            horizons: request.horizons,
            absorbing: request.absorbing,
            initial: None,
        },
    )
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
