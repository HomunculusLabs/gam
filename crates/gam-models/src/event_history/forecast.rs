//! History-conditioned forecasting, the population tier of the same
//! forecast, and the predictive probability integral transform, all
//! expectations under the latent state.
//!
//! A forecast filters the subject's own history into its latent state and
//! then integrates the killed process forward: the survival to a horizon is
//! `E[exp(−∫ Λ_T(t) dt)]` with `Λ_T` the total intensity of the terminal
//! marks, and the expected count of a mark by a horizon is the chronological
//! integral of the sub-density `m_d(t) = E[S(t) λ_d(t)]`, where `S(t)` is the
//! survival of the killed process up to `t` along the same latent path.
//! For a terminal mark that integral is its cumulative incidence; for a
//! once-only mark it is the probability of a first occurrence before
//! termination, with the mark's own hazard added to the killing; for a
//! recurrent mark it is the expected number of events before termination.
//!
//! The chronology is real: the survival at an interior quadrature time is a
//! Gauss-Legendre integral over the elapsed time up to that point, computed
//! by its own filter chain from the state at the start of the mesh cell, so
//! every reported probability is a proper quadrature of a well-defined
//! integral, and the identity `Σ_{d terminal} F_d(h) = 1 − S(h)` holds to
//! quadrature accuracy.
//!
//! The same window run from the stationary prior instead of a filtered
//! state ([`population_forecast`]) gives the lower tiers of the information
//! hierarchy: with population covariate values it is the population risk;
//! with a subject's own covariates (a risk score) it is what the model says
//! before any history is observed; the history-conditioned forecast updates
//! it. No weight between the tiers is chosen by hand — each is the same
//! probability model conditioned on more.
//!
//! The predictive PIT of an event is `1 − P(no event of any mark in
//! (t_prev, t_event] | history)`, which the filter yields as the product of
//! the normalisers of the zero-count nodes between consecutive events. Under
//! the model, the sequence of PITs of a subject is a Rosenblatt transform of
//! its event times: independent uniforms, across events and across
//! subjects (the time-rescaling theorem). The predictive mark probabilities
//! at each event complete the diagnostic for marked processes.

use super::chain::Grid;
use super::cohort::{
    CohortNodes, CovariateSegment, EventHistoryCohort, EventHistoryError, MarkKind,
    SubjectHistory, SubjectNodes, cell_rule, expand_nodes, mesh_cells,
};
use super::family::EventHistoryFit;
use super::marginal::{ForwardPass, SubjectInputs, expected_intensities, forward_filter};
use gam_terms::smooth::build_term_collection_design;
use ndarray::{Array1, Array2, ArrayView2};

/// One piece of a forecast window during which the covariates are
/// constant: from `start` until the next segment's start (or the last
/// horizon). `covariates` holds one value per covariate column, with
/// categorical covariates as their level codes.
#[derive(Clone, Debug, PartialEq)]
pub struct FutureSegment {
    pub start: f64,
    pub covariates: Vec<f64>,
}

/// A forecast request for one subject.
pub struct ForecastRequest<'a> {
    /// The subject's observed history (its covariate rows index the cohort).
    pub history: &'a SubjectHistory,
    /// Absolute horizon times, strictly increasing and after the exit time.
    pub horizons: &'a [f64],
    /// The covariate path over the forecast window. Empty holds the row in
    /// force at exit; otherwise a first segment starting after exit is
    /// preceded by that row.
    pub future: &'a [FutureSegment],
}

/// A forecast for a subject with no observed history: the latent state
/// starts at its stationary prior at `start`, so this is the population
/// tier when the covariates are population values and the covariate-only
/// tier (a new subject with a risk score) otherwise.
pub struct PopulationForecastRequest<'a> {
    /// Time the forecast window opens.
    pub start: f64,
    /// Absolute horizon times, strictly increasing and after `start`.
    pub horizons: &'a [f64],
    /// The covariate path over the window; the first segment must start at
    /// or before `start`.
    pub future: &'a [FutureSegment],
}

/// A forecast: per horizon, the probability that no terminal event has
/// fired, and the expected count of every mark (its cumulative incidence
/// when terminal, its first-occurrence probability when once-only).
#[derive(Clone, Debug)]
pub struct Forecast {
    pub horizons: Vec<f64>,
    pub survival: Vec<f64>,
    pub expected_counts: Array2<f64>,
}

/// The predictive PIT of one observed event and the predictive probability
/// of each mark given that an event happened then.
#[derive(Clone, Debug)]
pub struct EventPit {
    pub time: f64,
    pub mark: usize,
    pub pit: f64,
    pub mark_probabilities: Vec<f64>,
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

/// Population node log-intensities `η⁰` (design × coefficients + offset)
/// for every mark on a row matrix (covariate columns then time), index
/// `row * marks + d`.
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
    let atoms = fit.atoms();
    let mut loadings = Vec::with_capacity(marks * atoms);
    for d in 0..marks {
        for k in 0..atoms {
            loadings.push(fit.loadings[[d, k]]);
        }
    }
    (loadings, fit.log_rates.clone())
}

/// A filtered latent state: the grid and density at a time.
#[derive(Clone)]
struct LatentState {
    grid: Grid<f64>,
    alpha: Vec<f64>,
    time: f64,
}

/// The filtered latent state after a subject's observed history.
fn observed_state(
    fit: &EventHistoryFit,
    cohort: &EventHistoryCohort,
    history: &SubjectHistory,
    loadings: &[f64],
    log_rates: &[f64],
) -> Result<LatentState, EventHistoryError> {
    let marks = fit.marks();
    let observed = single_subject_nodes(fit, cohort, history)?;
    let eta0 = node_eta0(fit, observed.node_data.view())?;
    let all_marks = vec![true; marks];
    let mut pass = forward_filter(
        &SubjectInputs {
            nodes: &observed.subjects[0],
            eta0: &eta0,
            loadings,
            log_rates,
            time_scale: fit.time_scale,
            gh: fit.family.gauss_hermite(),
            continuation_gap: 0.0,
            designs: None,
        },
        None,
        &all_marks,
    )?;
    let last = observed.subjects[0].len() - 1;
    Ok(LatentState {
        grid: pass.grids.pop().expect("at least one node"),
        alpha: pass.alpha.pop().expect("at least one node"),
        time: observed.subjects[0].times[last],
    })
}

/// A chain of zero-count nodes for the future filter.
fn future_chain(times: &[f64], weights: &[f64], exposed: &[bool], marks: usize) -> SubjectNodes {
    let n = times.len();
    let mut exposures = Array2::<f64>::zeros((n, marks));
    for (i, &w) in weights.iter().enumerate() {
        for d in 0..marks {
            if exposed[d] {
                exposures[[i, d]] = w;
            }
        }
    }
    SubjectNodes {
        first_row: 0,
        times: times.to_vec(),
        gaps: times.windows(2).map(|w| w[1] - w[0]).collect(),
        weights: weights.to_vec(),
        exposures,
        counts: Array2::zeros((n, marks)),
        covariate_rows: vec![0; n],
    }
}

fn validate_horizons(horizons: &[f64], start: f64) -> Result<(), EventHistoryError> {
    if horizons.is_empty()
        || horizons.iter().any(|h| !h.is_finite())
        || horizons.windows(2).any(|w| !(w[1] > w[0]))
        || !(horizons[0] > start)
    {
        return Err(EventHistoryError::InvalidInput {
            reason: format!(
                "horizons must be finite, strictly increasing and later than the window's start {start}"
            ),
        });
    }
    Ok(())
}

fn validate_future(
    cohort: &EventHistoryCohort,
    future: &[FutureSegment],
    last_horizon: f64,
) -> Result<(), EventHistoryError> {
    let n_cov = cohort.covariates.ncols();
    for (i, segment) in future.iter().enumerate() {
        if !segment.start.is_finite() || segment.covariates.len() != n_cov {
            return Err(EventHistoryError::InvalidInput {
                reason: format!(
                    "future segment {i} needs a finite start and {n_cov} covariate values, got start {} and {} values",
                    segment.start,
                    segment.covariates.len()
                ),
            });
        }
        if i > 0 && !(segment.start > future[i - 1].start) {
            return Err(EventHistoryError::InvalidInput {
                reason: "future segments must have strictly increasing starts".to_string(),
            });
        }
        if segment.start >= last_horizon {
            return Err(EventHistoryError::InvalidInput {
                reason: format!(
                    "future segment {i} starts at {}, at or after the last horizon",
                    segment.start
                ),
            });
        }
        for (j, value) in segment.covariates.iter().enumerate() {
            if !value.is_finite() {
                return Err(EventHistoryError::InvalidInput {
                    reason: format!("future segment {i} has a non-finite covariate value at column {j}"),
                });
            }
            let levels = &cohort.covariate_levels[j];
            if !levels.is_empty() && (value.fract() != 0.0 || *value < 0.0 || *value >= levels.len() as f64)
            {
                return Err(EventHistoryError::InvalidInput {
                    reason: format!(
                        "future segment {i} has code {value} for categorical covariate {:?} with {} levels",
                        cohort.covariate_names[j],
                        levels.len()
                    ),
                });
            }
        }
    }
    Ok(())
}

/// The killed-process integration of one forecast window from a latent
/// state (`None` for the stationary prior) over a covariate path.
#[allow(clippy::too_many_arguments)]
fn run_window(
    fit: &EventHistoryFit,
    cohort: &EventHistoryCohort,
    initial: Option<&LatentState>,
    start: f64,
    horizons: &[f64],
    segments: Vec<CovariateSegment>,
    table: &Array2<f64>,
    at_risk: &[bool],
    label: &str,
) -> Result<Forecast, EventHistoryError> {
    let marks = fit.marks();
    let atoms = fit.atoms();
    let kinds = &cohort.mark_kinds;
    let n_cov = cohort.covariates.ncols();
    let (loadings, log_rates) = latent_parameters(fit);
    let last_horizon = horizons[horizons.len() - 1];
    // Every horizon is a mesh breakpoint, so cell ends land on horizons.
    let mut segments = segments;
    for &h in &horizons[..horizons.len() - 1] {
        if !segments.iter().any(|s| s.start == h) {
            let row = segments
                .iter()
                .rev()
                .find(|s| s.start <= h)
                .map(|s| s.row)
                .expect("a segment starts at the window's start");
            segments.push(CovariateSegment { start: h, row });
        }
    }
    segments.sort_by(|a, b| a.start.total_cmp(&b.start));
    let pseudo = SubjectHistory {
        id: label.to_string(),
        entry: start,
        exit: last_horizon,
        events: Vec::new(),
        segments,
    };
    let (gl_nodes, gl_weights) = gam_math::special::gauss_legendre(fit.quadrature_order);
    let cells = mesh_cells(&pseudo, false, fit.mesh_refinement);
    let q = fit.quadrature_order;
    // Evaluation points: the outer nodes of every cell, then the inner nodes
    // of every outer node's own chronological rule from the cell's start.
    let mut point_times: Vec<f64> = Vec::new();
    let mut point_rows: Vec<usize> = Vec::new();
    let mut outer: Vec<Vec<(f64, f64, usize)>> = Vec::with_capacity(cells.len());
    let mut inner: Vec<Vec<Vec<(f64, f64, usize)>>> = Vec::with_capacity(cells.len());
    let mut push_point = |t: f64| -> usize {
        point_times.push(t);
        point_rows.push(pseudo.covariate_row_at(t, false));
        point_times.len() - 1
    };
    for &(left, right) in &cells {
        let outer_rule: Vec<(f64, f64)> = cell_rule(left, right, &gl_nodes, &gl_weights).collect();
        let outer_nodes: Vec<(f64, f64, usize)> = outer_rule
            .iter()
            .map(|&(t, w)| (t, w, push_point(t)))
            .collect();
        let mut inner_cell = Vec::with_capacity(q);
        for &(t_j, _, _) in &outer_nodes {
            let rule: Vec<(f64, f64)> = cell_rule(left, t_j, &gl_nodes, &gl_weights).collect();
            let chain: Vec<(f64, f64, usize)> = rule
                .iter()
                .map(|&(s, v)| (s, v, push_point(s)))
                .collect();
            inner_cell.push(chain);
        }
        outer.push(outer_nodes);
        inner.push(inner_cell);
    }
    let mut rows = Array2::<f64>::zeros((point_times.len(), n_cov + 1));
    for (i, (&t, &row)) in point_times.iter().zip(point_rows.iter()).enumerate() {
        for j in 0..n_cov {
            rows[[i, j]] = table[[row, j]];
        }
        rows[[i, n_cov]] = t;
    }
    let eta0 = node_eta0(fit, rows.view())?;
    let eta_at = |point: usize| -> &[f64] { &eta0[point * marks..(point + 1) * marks] };
    let gh = fit.family.gauss_hermite();

    // One killed run under a killing set: the survival at the end of every
    // cell and the sub-density `m_d(t) = S(t) E[λ_d(t)]` at every outer node.
    let killed_run = |killing: &[bool]| -> Result<(Vec<f64>, Vec<Vec<f64>>), EventHistoryError> {
        let exposed: Vec<bool> = (0..marks).map(|d| killing[d] && at_risk[d]).collect();
        let mut state: Option<LatentState> = initial.cloned();
        let mut log_s = 0.0_f64;
        let mut cell_log_s = Vec::with_capacity(cells.len());
        let mut sub_density: Vec<Vec<f64>> = Vec::with_capacity(cells.len() * q);
        let filter = |state: &Option<LatentState>,
                      times: &[f64],
                      weights: &[f64],
                      eta: &[f64]|
         -> Result<ForwardPass<f64>, EventHistoryError> {
            let nodes = future_chain(times, weights, &exposed, marks);
            forward_filter(
                &SubjectInputs {
                    nodes: &nodes,
                    eta0: eta,
                    loadings: &loadings,
                    log_rates: &log_rates,
                    time_scale: fit.time_scale,
                    gh,
                    continuation_gap: state.as_ref().map_or(0.0, |s| times[0] - s.time),
                    designs: None,
                },
                state.as_ref().map(|s| (&s.grid, s.alpha.as_slice())),
                &exposed,
            )
        };
        for (c, outer_nodes) in outer.iter().enumerate() {
            for (j, &(t_j, _, point_j)) in outer_nodes.iter().enumerate() {
                let chain = &inner[c][j];
                let mut times: Vec<f64> = chain.iter().map(|&(s, _, _)| s).collect();
                let mut weights: Vec<f64> = chain.iter().map(|&(_, v, _)| v).collect();
                times.push(t_j);
                weights.push(0.0);
                let mut chain_eta = Vec::with_capacity(times.len() * marks);
                for &(_, _, point) in chain {
                    chain_eta.extend_from_slice(eta_at(point));
                }
                chain_eta.extend_from_slice(eta_at(point_j));
                let pass = filter(&state, &times, &weights, &chain_eta)?;
                let log_s_j: f64 = log_s + pass.log_normalisers.iter().sum::<f64>();
                let last = times.len() - 1;
                let intensities = expected_intensities(
                    &pass.grids[last],
                    &pass.predicted[last],
                    eta_at(point_j),
                    &loadings,
                    marks,
                    atoms,
                );
                sub_density.push(
                    (0..marks)
                        .map(|d| if at_risk[d] { log_s_j.exp() * intensities[d] } else { 0.0 })
                        .collect(),
                );
            }
            // Advance the state across the cell along its own rule.
            let times: Vec<f64> = outer_nodes.iter().map(|&(t, _, _)| t).collect();
            let weights: Vec<f64> = outer_nodes.iter().map(|&(_, w, _)| w).collect();
            let mut chain_eta = Vec::with_capacity(times.len() * marks);
            for &(_, _, point) in outer_nodes {
                chain_eta.extend_from_slice(eta_at(point));
            }
            let mut pass = filter(&state, &times, &weights, &chain_eta)?;
            log_s += pass.log_normalisers.iter().sum::<f64>();
            state = Some(LatentState {
                grid: pass.grids.pop().expect("cell has nodes"),
                alpha: pass.alpha.pop().expect("cell has nodes"),
                time: times[times.len() - 1],
            });
            cell_log_s.push(log_s);
        }
        Ok((cell_log_s, sub_density))
    };

    let terminal: Vec<bool> = kinds.iter().map(|k| *k == MarkKind::Terminal).collect();
    let (cell_log_s, base_density) = killed_run(&terminal)?;
    // Once-only marks are killed by their own hazard as well.
    let mut once_density: Vec<Option<Vec<Vec<f64>>>> = vec![None; marks];
    for d in 0..marks {
        if kinds[d] == MarkKind::Once && at_risk[d] {
            let mut killing = terminal.clone();
            killing[d] = true;
            once_density[d] = Some(killed_run(&killing)?.1);
        }
    }
    let n_h = horizons.len();
    let mut survival = vec![0.0; n_h];
    let mut expected = Array2::<f64>::zeros((n_h, marks));
    let mut counts = vec![0.0; marks];
    let mut horizon = 0usize;
    let mut point = 0usize;
    for (c, &(_, right)) in cells.iter().enumerate() {
        for &(_, w_j, _) in outer[c].iter() {
            for d in 0..marks {
                let density = match &once_density[d] {
                    Some(run) => run[point][d],
                    None => base_density[point][d],
                };
                counts[d] += w_j * density;
            }
            point += 1;
        }
        if horizon < n_h && right == horizons[horizon] {
            survival[horizon] = cell_log_s[c].exp();
            for d in 0..marks {
                expected[[horizon, d]] = counts[d];
            }
            horizon += 1;
        }
    }
    if horizon != n_h {
        return Err(EventHistoryError::NumericalFailure {
            reason: format!("forecast mesh reached {horizon} of {n_h} horizons"),
        });
    }
    Ok(Forecast {
        horizons: horizons.to_vec(),
        survival,
        expected_counts: expected,
    })
}

/// The covariate table extended by the future segments' rows, and the
/// segments as rows of that table starting at `window_start` (with
/// `initial_row` in force before the first future segment, if any).
fn future_table(
    cohort: &EventHistoryCohort,
    future: &[FutureSegment],
    window_start: f64,
    initial_row: Option<usize>,
) -> Result<(Array2<f64>, Vec<CovariateSegment>), EventHistoryError> {
    let n_cov = cohort.covariates.ncols();
    let base_rows = cohort.covariates.nrows();
    let mut table = Array2::<f64>::zeros((base_rows + future.len(), n_cov));
    table
        .slice_mut(ndarray::s![..base_rows, ..])
        .assign(&cohort.covariates);
    let mut segments: Vec<CovariateSegment> = Vec::new();
    if future.first().is_none_or(|s| s.start > window_start) {
        let row = initial_row.ok_or_else(|| EventHistoryError::InvalidInput {
            reason: format!(
                "the covariate path must cover the window from its start {window_start}: the first future segment starts at {:?}",
                future.first().map(|s| s.start)
            ),
        })?;
        segments.push(CovariateSegment {
            start: window_start,
            row,
        });
    }
    for (i, segment) in future.iter().enumerate() {
        for (j, value) in segment.covariates.iter().enumerate() {
            table[[base_rows + i, j]] = *value;
        }
        let start = segment.start.max(window_start);
        // Two segments at or before the start collapse onto the later one.
        if let Some(last) = segments.last_mut()
            && last.start == start
        {
            last.row = base_rows + i;
        } else {
            segments.push(CovariateSegment {
                start,
                row: base_rows + i,
            });
        }
    }
    Ok((table, segments))
}

/// Forecast one subject beyond its observed exit.
pub fn forecast(
    fit: &EventHistoryFit,
    cohort: &EventHistoryCohort,
    request: &ForecastRequest<'_>,
) -> Result<Forecast, EventHistoryError> {
    let marks = fit.marks();
    let kinds = &cohort.mark_kinds;
    let history = request.history;
    if kinds.len() != marks || fit.mark_kinds != *kinds {
        return Err(EventHistoryError::InvalidInput {
            reason: "the forecast cohort's mark kinds differ from the fit's".to_string(),
        });
    }
    validate_horizons(request.horizons, history.exit)?;
    let last_horizon = request.horizons[request.horizons.len() - 1];
    validate_future(cohort, request.future, last_horizon)?;
    // A subject whose follow-up ended with a terminal event has no future:
    // its survival is zero and nothing more can happen.
    if history.terminal_event(kinds).is_some() {
        return Ok(Forecast {
            horizons: request.horizons.to_vec(),
            survival: vec![0.0; request.horizons.len()],
            expected_counts: Array2::zeros((request.horizons.len(), marks)),
        });
    }
    let (loadings, log_rates) = latent_parameters(fit);
    let state = observed_state(fit, cohort, history, &loadings, &log_rates)?;
    let (table, segments) = future_table(
        cohort,
        request.future,
        history.exit,
        Some(history.covariate_row_at(history.exit, false)),
    )?;
    let at_risk: Vec<bool> = (0..marks)
        .map(|d| match kinds[d] {
            MarkKind::Recurrent | MarkKind::Terminal => true,
            MarkKind::Once => !history.events.iter().any(|e| e.mark == d),
        })
        .collect();
    run_window(
        fit,
        cohort,
        Some(&state),
        history.exit,
        request.horizons,
        segments,
        &table,
        &at_risk,
        &format!("{}::forecast", history.id),
    )
}

/// Forecast a subject with no observed history from covariate values alone.
pub fn population_forecast(
    fit: &EventHistoryFit,
    cohort: &EventHistoryCohort,
    request: &PopulationForecastRequest<'_>,
) -> Result<Forecast, EventHistoryError> {
    let marks = fit.marks();
    if cohort.mark_kinds.len() != marks || fit.mark_kinds != cohort.mark_kinds {
        return Err(EventHistoryError::InvalidInput {
            reason: "the forecast cohort's mark kinds differ from the fit's".to_string(),
        });
    }
    if !request.start.is_finite() {
        return Err(EventHistoryError::InvalidInput {
            reason: "the window's start must be finite".to_string(),
        });
    }
    validate_horizons(request.horizons, request.start)?;
    let last_horizon = request.horizons[request.horizons.len() - 1];
    validate_future(cohort, request.future, last_horizon)?;
    let (table, segments) = future_table(cohort, request.future, request.start, None)?;
    let at_risk = vec![true; marks];
    run_window(
        fit,
        cohort,
        None,
        request.start,
        request.horizons,
        segments,
        &table,
        &at_risk,
        "population::forecast",
    )
}

/// Predictive PIT of every event of a subject, in time order, with the
/// predictive mark probabilities at each event.
pub fn predictive_pit(
    fit: &EventHistoryFit,
    cohort: &EventHistoryCohort,
    history: &SubjectHistory,
) -> Result<Vec<EventPit>, EventHistoryError> {
    let marks = fit.marks();
    let atoms = fit.atoms();
    let kinds = &cohort.mark_kinds;
    let (loadings, log_rates) = latent_parameters(fit);
    let nodes = single_subject_nodes(fit, cohort, history)?;
    let eta0 = node_eta0(fit, nodes.node_data.view())?;
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
            designs: None,
        },
        None,
        &vec![true; marks],
    )?;
    let mut pits = Vec::new();
    let mut log_survival: f64 = 0.0;
    for n in 0..subject.len() {
        if !subject.is_event(n) {
            log_survival += pass.log_normalisers[n];
            continue;
        }
        let t = subject.times[n];
        let pit = -log_survival.exp_m1();
        let slack = 64.0 * f64::EPSILON;
        if !pit.is_finite() || pit < -slack || pit > 1.0 + slack {
            return Err(EventHistoryError::NumericalFailure {
                reason: format!(
                    "subject {:?}: predictive survival to the event at {t} is {}, outside [0, 1]",
                    history.id,
                    log_survival.exp()
                ),
            });
        }
        let pit = pit.clamp(0.0, 1.0);
        let intensities = expected_intensities(
            &pass.grids[n],
            &pass.predicted[n],
            &eta0[n * marks..(n + 1) * marks],
            &loadings,
            marks,
            atoms,
        );
        let at_risk: Vec<f64> = (0..marks)
            .map(|d| if history.at_risk(d, t, kinds) { intensities[d] } else { 0.0 })
            .collect();
        let total: f64 = at_risk.iter().sum();
        let mark_probabilities: Vec<f64> = if total > 0.0 {
            at_risk.iter().map(|v| v / total).collect()
        } else {
            vec![0.0; marks]
        };
        for d in 0..marks {
            let copies = subject.counts[[n, d]].round() as usize;
            for _ in 0..copies {
                pits.push(EventPit {
                    time: t,
                    mark: d,
                    pit,
                    mark_probabilities: mark_probabilities.clone(),
                });
            }
        }
        log_survival = 0.0;
    }
    Ok(pits)
}

/// Kolmogorov–Smirnov distance of a PIT sample from the uniform law, or
/// `None` for an empty sample (no events carry no information).
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
