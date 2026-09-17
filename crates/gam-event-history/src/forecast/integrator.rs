//! One killed-process integrator for every event-history forecast engine.
//!
//! A forecast integrates a killed point process forward over a window. Per
//! killed run it tracks the survival `S(t)` against the marks that kill the
//! run, and the expected count of every mark the run reports:
//! `∫ S(t) E[λ_d(t) | alive] dt`. An engine (the grid filter of the
//! event-history model, a particle route of the joint model) supplies each
//! run's integral over one cell. This module owns everything that makes the
//! result a checked number.
//!
//! - The mesh. Level-0 cells run between the window's breakpoints: its start,
//!   the covariate changes and the last horizon. Horizons are output times. A
//!   horizon inside an accepted cell is reached by the same rule from that
//!   cell's start, so the mesh never depends on which horizons were requested.
//! - Survival and incidence are one evolution (`coupled`). The survival plus
//!   the killing marks' incidences equals the survival at the window's start,
//!   to roundoff, at any resolution.
//! - Acceptance. A cell's time error is the gap between the cell and its two
//!   halves, for the survival and for every count. Halving continues while that
//!   gap exceeds the larger of two bounds on the quantity's error: the other
//!   error component the engine measured for it (its latent quadrature against
//!   the next rung, a Monte Carlo spread) and its roundoff floor. No tolerance
//!   is set: time is refined exactly until it no longer dominates the other
//!   known error. The other components are measured and reported, never refined
//!   here. Every accepted component, and the roundoff of every value produced,
//!   accumulates into the error returned with each value.
//! - Termination at derived limits only. A cell at the resolution of its
//!   endpoints is refused, and so is a halving whose time gap did not contract.

use crate::cohort::EventHistoryError;

/// One killed run of a forecast window: the marks whose hazards kill it and
/// the marks whose expected counts it reports. An event-history window has a
/// base run, killed by the terminal marks and reporting them and the recurrent
/// marks. It also has one run for each once-only mark still at risk, killed by
/// that mark's own hazard as well.
pub(crate) struct KilledRun {
    pub exposed: Vec<bool>,
    pub reported: Vec<bool>,
}

/// An engine's integral of one run over one cell, relative to the survival at
/// the cell's start. It carries the log survival decrement `L = ln S(b)/S(a)`,
/// the cell's quadrature of every mark's sub-density
/// `S(t)/S(a) · E[λ_d(t) | alive]`, and the engine's state at the cell's end.
pub(crate) struct CellSums<S> {
    pub log_decrement: f64,
    pub sub_densities: Vec<f64>,
    pub state: S,
}

/// One run's integral over one cell with survival and incidence coupled,
/// relative to the survival at the cell's start. The increments are `None`
/// when the cell is too coarse for its sub-density to be represented at all.
pub(crate) struct CellIntegral<S> {
    pub log_decrement: f64,
    pub increments: Option<Vec<f64>>,
    pub state: S,
}

/// The absolute gaps between two integrals of the same cell for one run,
/// relative to the survival at the cell's start: the gap in the survival
/// factor and the gap in every count increment.
pub(crate) struct RunGaps {
    pub survival: f64,
    pub counts: Vec<f64>,
}

/// Where a killed run stands at a time: the engine's state, the log survival,
/// the reported counts, and the error those have accumulated.
#[derive(Clone)]
pub(crate) struct RunPosition<S> {
    pub state: S,
    pub log_survival: f64,
    pub counts: Vec<f64>,
    pub survival_error: f64,
    pub count_errors: Vec<f64>,
}

/// What a forecast engine supplies: its killed runs, each run's integral over
/// one cell, the other measured components of a cell's error, and the
/// roundoff a cell integral carries.
pub(crate) trait KilledProcess {
    /// Where the engine stands at a cell boundary.
    type State: Clone;

    fn runs(&self) -> &[KilledRun];

    /// Every run's cell sums over `[left, right]` from `from`. An
    /// `EventHistoryError::LostPositivity` refusal says the cell is too coarse
    /// for the engine's representation. A finer cell resolves that.
    fn cell(
        &self,
        left: f64,
        right: f64,
        from: &[Self::State],
    ) -> Result<Vec<CellSums<Self::State>>, EventHistoryError>;

    /// Every run's gaps in `coarse`, the cell's integral, under the engine's
    /// own refinement on axes other than time: a latent quadrature against its
    /// next rung, or a Monte Carlo spread. `None` when time is the only axis.
    fn other_error(
        &self,
        left: f64,
        right: f64,
        from: &[Self::State],
        coarse: &[CellIntegral<Self::State>],
    ) -> Result<Option<Vec<RunGaps>>, EventHistoryError>;

    /// The relative roundoff a cell integral carries.
    fn roundoff(&self) -> f64;
}

/// A run's cell sums as one evolution of survival and incidence.
///
/// The marks that kill the run share the survival decrement `−expm1(L)` in
/// proportion to their sub-density sums. So the survival and the killing
/// marks' incidences sum to the survival at the cell's start to roundoff,
/// whatever the resolution, and a constant hazard is exact on any mesh.
///
/// A reported mark that does not kill the run keeps its sub-density sum. A run
/// that nothing kills keeps its survival exactly, whatever roundoff its
/// engine's normalisers carry.
///
/// A killing sub-density whose sum is zero is represented as zero, so the
/// killing marks gain nothing, as long as the survival did not move either.
/// A survival that fell while every outer node's sub-density underflowed is a
/// cell too coarse to represent, and its increments are `None`.
pub(crate) fn coupled<S>(run: &KilledRun, sums: CellSums<S>) -> CellIntegral<S> {
    let kills = run.exposed.contains(&true);
    let decrement = -sums.log_decrement.exp_m1();
    let killing: f64 = sums
        .sub_densities
        .iter()
        .zip(&run.exposed)
        .filter(|(_, exposed)| **exposed)
        .map(|(density, _)| *density)
        .sum();
    let representable = !kills || killing > 0.0 || decrement == 0.0;
    let increments = representable.then(|| {
        (0..run.reported.len())
            .map(|d| {
                if !run.reported[d] {
                    0.0
                } else if kills && run.exposed[d] {
                    if killing > 0.0 {
                        decrement * sums.sub_densities[d] / killing
                    } else {
                        0.0
                    }
                } else {
                    sums.sub_densities[d]
                }
            })
            .collect()
    });
    CellIntegral {
        log_decrement: if kills { sums.log_decrement } else { 0.0 },
        increments,
        state: sums.state,
    }
}

/// Every run's gaps between two integrals of the same cell. An increment that
/// cannot be represented is an infinite gap.
pub(crate) fn integral_gaps<S>(
    trial: &[CellIntegral<S>],
    reference: &[CellIntegral<S>],
    marks: usize,
) -> Vec<RunGaps> {
    trial
        .iter()
        .zip(reference)
        .map(|(t, r)| RunGaps {
            survival: (t.log_decrement.exp() - r.log_decrement.exp()).abs(),
            counts: match (&t.increments, &r.increments) {
                (Some(a), Some(b)) => a.iter().zip(b).map(|(x, y)| (x - y).abs()).collect(),
                _ => vec![f64::INFINITY; marks],
            },
        })
        .collect()
}

/// Two consecutive cells' integrals as one, relative to the survival at the
/// first one's start.
fn joined<S>(first: &CellIntegral<S>, second: CellIntegral<S>) -> CellIntegral<S> {
    let factor = first.log_decrement.exp();
    let increments = match (&first.increments, second.increments) {
        (Some(a), Some(b)) => Some(a.iter().zip(&b).map(|(x, y)| x + factor * y).collect()),
        _ => None,
    };
    CellIntegral {
        log_decrement: first.log_decrement + second.log_decrement,
        increments,
        state: second.state,
    }
}

impl<S: Clone> RunPosition<S> {
    fn opening(state: S, marks: usize) -> Self {
        RunPosition {
            state,
            log_survival: 0.0,
            counts: vec![0.0; marks],
            survival_error: 0.0,
            count_errors: vec![0.0; marks],
        }
    }

    /// This position carried across an accepted cell by `integral`. The errors
    /// gain the cell's error components `gaps` and the roundoff
    /// `roundoff · (1 + |value|)` of every value the cell produced, while the
    /// error already carried scales with what it multiplies. A survival that
    /// has underflowed carries no further gaps.
    fn advanced(&self, integral: &CellIntegral<S>, gaps: &RunGaps, roundoff: f64) -> Self {
        let survival = self.log_survival.exp();
        let scaled = |gap: f64| if survival > 0.0 { survival * gap } else { 0.0 };
        let mut counts = self.counts.clone();
        let mut count_errors = self.count_errors.clone();
        if let Some(increments) = &integral.increments {
            for d in 0..counts.len() {
                counts[d] += survival * increments[d];
                count_errors[d] += self.survival_error * increments[d]
                    + scaled(gaps.counts[d])
                    + roundoff * (1.0 + counts[d].abs());
            }
        }
        let log_survival = self.log_survival + integral.log_decrement;
        RunPosition {
            state: integral.state.clone(),
            log_survival,
            counts,
            survival_error: self.survival_error * integral.log_decrement.exp()
                + scaled(gaps.survival)
                + roundoff * (1.0 + log_survival.exp()),
            count_errors,
        }
    }
}

/// A cell waiting to be integrated: its bounds, its integral when that was
/// computed from the current position, and the largest absolute time gap of
/// the cell it halves.
struct Pending<S> {
    left: f64,
    right: f64,
    coarse: Option<Vec<CellIntegral<S>>>,
    parent_time_gap: f64,
}

/// A cell an engine's representation could not hold has not been resolved.
/// A finer cell can resolve it. Any other failure is the forecast's own.
fn resolved<T>(result: Result<T, EventHistoryError>) -> Result<Option<T>, EventHistoryError> {
    match result {
        Ok(value) => Ok(Some(value)),
        Err(EventHistoryError::LostPositivity { .. }) => Ok(None),
        Err(error) => Err(error),
    }
}

/// The largest ratio, over every run's survival and counts, of the time gap
/// (the cell against its halves) to what else bounds that quantity's error.
/// That bound is the engine's other measured component or, where that is
/// smaller, the value's roundoff floor `roundoff · (1 + |value|)`. Returns the
/// ratio, the gaps an accepted cell adds to the error (relative to the survival
/// at the cell's start), and the largest absolute time gap.
fn time_excess<S>(
    position: &[RunPosition<S>],
    coarse: &[CellIntegral<S>],
    fine: &[CellIntegral<S>],
    other: Option<&[RunGaps]>,
    roundoff: f64,
    marks: usize,
) -> Result<(f64, Vec<RunGaps>, f64), EventHistoryError> {
    let time = integral_gaps(coarse, fine, marks);
    let mut excess = 0.0_f64;
    let mut largest = 0.0_f64;
    let mut totals = Vec::with_capacity(time.len());
    for (r, ((p, f), t)) in position.iter().zip(fine).zip(&time).enumerate() {
        let survival = p.log_survival.exp();
        if survival > 0.0 {
            largest = t
                .counts
                .iter()
                .fold(largest.max(survival * t.survival), |acc, gap| acc.max(survival * gap));
        }
        let other_gaps = other.and_then(|gaps| gaps.get(r));
        let units = |gap: f64, other_gap: f64, value: f64| -> Result<f64, EventHistoryError> {
            if survival == 0.0 {
                return Ok(0.0);
            }
            let bound = (survival * other_gap).max(roundoff * (1.0 + value.abs()));
            let units = if other_gap.is_finite() { survival * gap / bound } else { f64::INFINITY };
            if units.is_nan() {
                return Err(EventHistoryError::NumericalFailure {
                    reason: "a forecast cell's survival or counts are not finite".to_string(),
                });
            }
            Ok(units)
        };
        let survival_other = other_gaps.map_or(0.0, |gaps| gaps.survival);
        excess = excess.max(units(
            t.survival,
            survival_other,
            survival * f.log_decrement.exp(),
        )?);
        let mut counts = Vec::with_capacity(marks);
        for d in 0..marks {
            let other_gap = other_gaps.map_or(0.0, |gaps| gaps.counts[d]);
            let increment = f.increments.as_ref().map_or(0.0, |increments| increments[d]);
            excess = excess.max(units(t.counts[d], other_gap, p.counts[d] + survival * increment)?);
            counts.push(t.counts[d] + other_gap);
        }
        totals.push(RunGaps {
            survival: t.survival + survival_other,
            counts,
        });
    }
    Ok((excess, totals, largest))
}

/// Integrate every run of `process` from `opening` over the window whose
/// level-0 breakpoints are `breakpoints` (its start, covariate changes and last
/// horizon). Returns every run's position at each of `horizons`, which must be
/// increasing and within the window.
pub(crate) fn integrate_window<P: KilledProcess>(
    process: &P,
    opening: Vec<P::State>,
    breakpoints: &[f64],
    horizons: &[f64],
) -> Result<Vec<Vec<RunPosition<P::State>>>, EventHistoryError> {
    let marks = process.runs().first().map_or(0, |run| run.reported.len());
    let mut position: Vec<RunPosition<P::State>> = opening
        .into_iter()
        .map(|state| RunPosition::opening(state, marks))
        .collect();
    let mut reached = Vec::with_capacity(horizons.len());
    for pair in breakpoints.windows(2) {
        let inside: Vec<f64> = horizons
            .iter()
            .copied()
            .filter(|&h| h > pair[0] && h <= pair[1])
            .collect();
        let (end, at_horizons) = integrate_interval(process, position, pair[0], pair[1], &inside, marks)?;
        position = end;
        reached.extend(at_horizons);
    }
    if reached.len() != horizons.len() {
        return Err(EventHistoryError::NumericalFailure {
            reason: format!(
                "the forecast integration reached {} of {} horizons",
                reached.len(),
                horizons.len()
            ),
        });
    }
    Ok(reached)
}

/// Integrate every run over `[left, right]` from `from`. Returns the positions
/// at `right` and at every horizon of `horizons` (increasing, in
/// `(left, right]`).
fn integrate_interval<P: KilledProcess>(
    process: &P,
    from: Vec<RunPosition<P::State>>,
    left: f64,
    right: f64,
    horizons: &[f64],
    marks: usize,
) -> Result<(Vec<RunPosition<P::State>>, Vec<Vec<RunPosition<P::State>>>), EventHistoryError> {
    let runs = process.runs();
    let couple = |sums: Vec<CellSums<P::State>>| -> Vec<CellIntegral<P::State>> {
        runs.iter().zip(sums).map(|(run, s)| coupled(run, s)).collect()
    };
    let mut position = from;
    let mut reached = Vec::with_capacity(horizons.len());
    let mut next_horizon = 0usize;
    let mut pending = vec![Pending {
        left,
        right,
        coarse: None,
        parent_time_gap: f64::INFINITY,
    }];
    while let Some(cell) = pending.pop() {
        let (a, b) = (cell.left, cell.right);
        let middle = a + 0.5 * (b - a);
        if !(middle > a && middle < b) {
            return Err(EventHistoryError::NumericalFailure {
                reason: format!(
                    "the forecast cell [{a}, {b}] is at the resolution of its endpoints and its survival and counts are still unresolved"
                ),
            });
        }
        let at: Vec<P::State> = position.iter().map(|p| p.state.clone()).collect();
        let coarse = match cell.coarse {
            Some(integral) => Some(integral),
            None => resolved(process.cell(a, b, &at))?.map(&couple),
        };
        let first = resolved(process.cell(a, middle, &at))?.map(&couple);
        let fine = match &first {
            Some(halves) => {
                let halfway: Vec<P::State> = halves.iter().map(|half| half.state.clone()).collect();
                resolved(process.cell(middle, b, &halfway))?.map(|sums| {
                    halves
                        .iter()
                        .zip(couple(sums))
                        .map(|(half, second)| joined(half, second))
                        .collect::<Vec<_>>()
                })
            }
            None => None,
        };
        let measured = match (&coarse, &fine) {
            (Some(c), Some(f)) => {
                let other = process.other_error(a, b, &at, c)?;
                Some(time_excess(&position, c, f, other.as_deref(), process.roundoff(), marks)?)
            }
            _ => None,
        };
        match (measured, fine) {
            (Some((excess, gaps, _)), Some(fine)) if excess <= 1.0 => {
                let entering = position;
                position = entering
                    .iter()
                    .zip(&fine)
                    .zip(&gaps)
                    .map(|((p, integral), g)| p.advanced(integral, g, process.roundoff()))
                    .collect();
                while next_horizon < horizons.len() && horizons[next_horizon] <= b {
                    let h = horizons[next_horizon];
                    if h == b {
                        reached.push(position.clone());
                    } else {
                        reached.push(integrate_interval(process, entering.clone(), a, h, &[], marks)?.0);
                    }
                    next_horizon += 1;
                }
            }
            (measured, _) => {
                // The absolute gap, not its ratio to a bound the engine
                // re-measures on every cell and that shrinks with it.
                let time_gap = measured.as_ref().map_or(f64::INFINITY, |m| m.2);
                if cell.parent_time_gap.is_finite() && !(time_gap < cell.parent_time_gap) {
                    return Err(EventHistoryError::NumericalFailure {
                        reason: format!(
                            "halving the forecast cell [{a}, {b}] did not contract its time error: gap {time_gap:.3e} after {:.3e}",
                            cell.parent_time_gap
                        ),
                    });
                }
                pending.push(Pending {
                    left: middle,
                    right: b,
                    coarse: None,
                    parent_time_gap: time_gap,
                });
                pending.push(Pending {
                    left: a,
                    right: middle,
                    coarse: first,
                    parent_time_gap: time_gap,
                });
            }
        }
    }
    Ok((position, reached))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn terminal_and_recurrent() -> KilledRun {
        KilledRun {
            exposed: vec![true, false],
            reported: vec![true, true],
        }
    }

    #[test]
    fn a_cell_whose_killing_hazard_underflowed_adds_nothing_and_stays_representable() {
        // A terminal hazard whose every outer-node sub-density underflowed
        // (η⁰ near −800), with the survival unmoved: nothing was killed, so the
        // cell is resolved with zero incidence, not an infinite gap.
        let integral = coupled(
            &terminal_and_recurrent(),
            CellSums {
                log_decrement: 0.0,
                sub_densities: vec![0.0, 0.3],
                state: (),
            },
        );
        assert_eq!(integral.increments, Some(vec![0.0, 0.3]));
        assert_eq!(integral.log_decrement, 0.0);
    }

    #[test]
    fn a_survival_that_fell_with_no_represented_sub_density_is_unresolved() {
        let integral = coupled(
            &terminal_and_recurrent(),
            CellSums {
                log_decrement: -800.0,
                sub_densities: vec![0.0, 0.0],
                state: (),
            },
        );
        assert!(integral.increments.is_none());
    }

    #[test]
    fn killing_marks_share_the_survival_decrement() {
        let run = KilledRun {
            exposed: vec![true, true, false],
            reported: vec![true, true, true],
        };
        let log_decrement = -3.0_f64;
        let integral = coupled(
            &run,
            CellSums {
                log_decrement,
                sub_densities: vec![0.61, 0.29, 1.7],
                state: (),
            },
        );
        let increments = integral.increments.expect("representable");
        // The survival factor and the killing marks' incidences sum to one.
        assert!((log_decrement.exp() + increments[0] + increments[1] - 1.0).abs() <= 4.0 * f64::EPSILON);
        assert!((increments[0] / increments[1] - 0.61 / 0.29).abs() <= 64.0 * f64::EPSILON);
        assert_eq!(increments[2], 1.7);
    }
}
