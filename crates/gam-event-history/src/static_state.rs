//! Adaptive integration for genuinely static Gaussian frailties.
//!
//! A static factor's likelihood over a run of nodes depends on the state only
//! through [`Statistics`]. Two calculations read it, on different grids:
//! - The whole-history integral ([`filter`]) places one grid from the whole
//!   observed likelihood. The chronological updates on that grid telescope to
//!   the same integral for every event placement, so the pass's total and its
//!   final state are resolved. Its intermediate normalisers and densities are
//!   not: a grid that resolves the final posterior need not resolve the broad
//!   posterior of an early prefix, so those partial products are not filtering
//!   probabilities (#2962).
//! - The chronological predictive quantities ([`spells`]) are ratios of prefix
//!   integrals, each resolved on a grid placed from its own prefix, with the
//!   prior's absolute mass intact.

use crate::chain::{Grid, log_sum_exp, normal_density};
use crate::cohort::EventHistoryError;
use crate::marginal::{ForwardPass, Spell, SubjectInputs, centred_baseline, condition, node_likelihood};
use crate::scalar::{add_real, div, exp, ln, sqrt};
use gam_math::nested_dual::JetField;

fn failure(reason: &str) -> EventHistoryError {
    EventHistoryError::NumericalFailure { reason: reason.to_string() }
}

/// Whether every atom is a genuinely static frailty: a zero rate, whose
/// transition across any gap is the identity.
pub(crate) fn is_static<S: JetField>(rates: &[S]) -> bool {
    !rates.is_empty() && rates.iter().all(|r| r.value() == 0.0)
}

pub(crate) fn filter<S: JetField>(inputs: &SubjectInputs<'_, S>, initial: Option<(&Grid<S>, &[S])>,
    compensated: &[bool]) -> Result<ForwardPass<S>, EventHistoryError> {
    let like = &inputs.eta0[0];
    let marks = inputs.nodes.counts.ncols();
    let atoms = inputs.rates.len();
    let (grid, mut density) = match initial {
        Some((grid, density)) => (grid.clone(), density.to_vec()),
        None => {
            let grid = posterior_grid(inputs, compensated)?;
            let density = prior(&grid, like);
            (grid, density)
        }
    };
    let mut pass = ForwardPass { grids: Vec::new(), alpha: Vec::new(), predicted: Vec::new(), log_normalisers: Vec::new() };
    for n in 0..inputs.nodes.len() {
        let likelihood = node_likelihood(&grid, &inputs.eta0[n * marks..(n + 1) * marks],
            inputs.loadings, &inputs.nodes.counts.row(n).to_vec(), &inputs.nodes.exposure_row(n),
            Some(compensated), inputs.log_normaliser.map(|m| &m[n * marks..(n + 1) * marks]), marks, atoms, false);
        let (updated, mass) = condition(&grid, &density, &likelihood.ell, likelihood.shift, "static frailty")?;
        pass.predicted.push(density);
        pass.grids.push(grid.clone());
        pass.log_normalisers.push(add_real(&ln(&mass), likelihood.shift));
        pass.alpha.push(updated.clone());
        density = updated;
    }
    Ok(pass)
}

/// The spells of a static factor's follow-up (see [`crate::marginal::spells`]).
/// With `Z` the integral `∫ φ L` over a prefix of nodes, a spell's survival is
/// `Z(before its end) / Z(through its start)`, and at an event the expected
/// intensity of mark `d` is `exp(η⁰_d − log M_d) Z_d / Z(before the event)`,
/// where `Z_d` adds one event of mark `d`. Every integral is placed and resolved
/// from its own prefix, so no later node enters a spell.
pub(crate) fn spells(inputs: &SubjectInputs<'_, f64>, compensated: &[bool]) -> Result<Vec<Spell>, EventHistoryError> {
    let nodes = inputs.nodes;
    let marks = nodes.counts.ncols();
    let atoms = inputs.rates.len();
    let mut statistics = Statistics::empty(inputs);
    let mut opened = statistics.log_integral(inputs)?;
    let mut spells = Vec::new();
    let mut open = false;
    for n in 0..nodes.len() {
        if !nodes.is_event(n) {
            statistics.push(inputs, compensated, n);
            open = true;
            continue;
        }
        let before = statistics.log_integral(inputs)?;
        let mut intensities = Vec::with_capacity(marks);
        for d in 0..marks {
            let base = centred_baseline(&inputs.eta0[n * marks + d], &inputs.loadings[d * atoms..(d + 1) * atoms],
                inputs.log_normaliser.map(|m| &m[n * marks + d]));
            intensities.push((base + statistics.with_event(inputs, d).log_integral(inputs)? - before).exp());
        }
        spells.push(Spell { node: n, log_survival: before - opened, intensities: Some(intensities) });
        statistics.push(inputs, compensated, n);
        opened = statistics.log_integral(inputs)?;
        open = false;
    }
    if open {
        spells.push(Spell { node: nodes.len() - 1, log_survival: statistics.log_integral(inputs)? - opened, intensities: None });
    }
    Ok(spells)
}

fn solve<S: JetField>(matrix: &[S], rhs: &[S]) -> Result<Vec<S>, EventHistoryError> {
    let n = rhs.len();
    let zero = rhs[0].constant_like(0.0);
    let mut lower = vec![zero.clone(); n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut value = matrix[i * n + j].clone();
            for k in 0..j { value = value.sub(&lower[i * n + k].mul(&lower[j * n + k])); }
            lower[i * n + j] = if i == j {
                if !(value.value().is_finite() && value.value() > 0.0) {
                    return Err(failure("static posterior precision is not positive definite"));
                }
                sqrt(&value)
            } else { div(&value, &lower[j * n + j]) };
        }
    }
    let mut out = rhs.to_vec();
    for i in 0..n {
        for j in 0..i { out[i] = out[i].sub(&lower[i * n + j].mul(&out[j])); }
        out[i] = div(&out[i], &lower[i * n + i]);
    }
    for i in (0..n).rev() {
        for j in i + 1..n { out[i] = out[i].sub(&lower[j * n + i].mul(&out[j])); }
        out[i] = div(&out[i], &lower[i * n + i]);
    }
    Ok(out)
}

pub(crate) fn prior<S: JetField>(grid: &Grid<S>, like: &S) -> Vec<S> {
    let zero = like.constant_like(0.0);
    let unit = like.constant_like(1.0);
    (0..grid.size()).map(|i| (0..grid.dimension()).fold(unit.clone(), |p, k|
        p.mul(&normal_density(grid.coordinate(i, k), &zero, &unit)))).collect()
}

/// What a static factor's likelihood over a run of nodes depends on the state
/// through. Up to a constant, `ln L(z) = linear · z − Σ_d exp(log_hazard_d +
/// a_d · z)`, with `linear = Σ_n Σ_d y_{nd} a_d` and `log_hazard_d` the log of
/// `Σ_n w_{nd} exp(η⁰_{nd} − log M_{nd})` over the nodes where mark `d` is
/// compensated and exposed. The constant `Σ y (η⁰ − log M)` moves no grid and
/// cancels from every ratio of integrals over the same events.
#[derive(Clone)]
struct Statistics<S> {
    linear: Vec<S>,
    log_hazards: Vec<Option<S>>,
}

/// `ln w_{nd} + η⁰_{nd} − log M_{nd}` when mark `d` is compensated and exposed
/// at node `n`.
fn hazard_term<S: JetField>(inputs: &SubjectInputs<'_, S>, compensated: &[bool], n: usize, d: usize) -> Option<S> {
    let marks = inputs.nodes.counts.ncols();
    let atoms = inputs.rates.len();
    let exposure = inputs.nodes.exposures[[n, d]];
    (compensated[d] && exposure > 0.0).then(|| add_real(&centred_baseline(&inputs.eta0[n * marks + d],
        &inputs.loadings[d * atoms..(d + 1) * atoms], inputs.log_normaliser.map(|m| &m[n * marks + d])), exposure.ln()))
}

impl<S: JetField> Statistics<S> {
    /// No nodes: the prior alone.
    fn empty(inputs: &SubjectInputs<'_, S>) -> Self {
        let zero = inputs.eta0[0].constant_like(0.0);
        Statistics { linear: vec![zero; inputs.rates.len()], log_hazards: vec![None; inputs.nodes.counts.ncols()] }
    }

    /// Every node of the history at once.
    fn whole(inputs: &SubjectInputs<'_, S>, compensated: &[bool]) -> Self {
        let atoms = inputs.rates.len();
        let mut statistics = Self::empty(inputs);
        for d in 0..inputs.nodes.counts.ncols() {
            let loadings = &inputs.loadings[d * atoms..(d + 1) * atoms];
            let mut terms = Vec::new();
            for n in 0..inputs.nodes.len() {
                for k in 0..atoms {
                    statistics.linear[k] = statistics.linear[k].add(&loadings[k].scale(inputs.nodes.counts[[n, d]]));
                }
                terms.extend(hazard_term(inputs, compensated, n, d));
            }
            if !terms.is_empty() {
                statistics.log_hazards[d] = Some(log_sum_exp(&terms));
            }
        }
        statistics
    }

    /// Append node `n`.
    fn push(&mut self, inputs: &SubjectInputs<'_, S>, compensated: &[bool], n: usize) {
        let atoms = inputs.rates.len();
        for d in 0..inputs.nodes.counts.ncols() {
            for k in 0..atoms {
                self.linear[k] = self.linear[k].add(&inputs.loadings[d * atoms + k].scale(inputs.nodes.counts[[n, d]]));
            }
            if let Some(term) = hazard_term(inputs, compensated, n, d) {
                self.log_hazards[d] = Some(match self.log_hazards[d].take() {
                    Some(log_hazard) => log_sum_exp(&[log_hazard, term]),
                    None => term,
                });
            }
        }
    }

    /// The same nodes with one more event of mark `d`, which carries no exposure.
    fn with_event(&self, inputs: &SubjectInputs<'_, S>, d: usize) -> Self {
        let atoms = inputs.rates.len();
        let mut out = self.clone();
        for k in 0..atoms {
            out.linear[k] = out.linear[k].add(&inputs.loadings[d * atoms + k]);
        }
        out
    }

    /// The product grid at the mode of `φ L`, scaled by its curvature.
    fn place(&self, inputs: &SubjectInputs<'_, S>) -> Result<Grid<S>, EventHistoryError> {
        let atoms = inputs.rates.len();
        let like = &inputs.eta0[0];
        let zero = like.constant_like(0.0);
        let evaluate = |z: &[S]| {
            let mut value = zero.clone();
            let mut gradient = self.linear.to_vec();
            let mut precision = vec![zero.clone(); atoms * atoms];
            for k in 0..atoms {
                value = value.add(&self.linear[k].mul(&z[k])).sub(&z[k].mul(&z[k]).scale(0.5));
                gradient[k] = gradient[k].sub(&z[k]);
                precision[k * atoms + k] = like.constant_like(1.0);
            }
            for (d, log_hazard) in self.log_hazards.iter().enumerate() {
                if let Some(log_hazard) = log_hazard {
                    let a = &inputs.loadings[d * atoms..(d + 1) * atoms];
                    let log_rate = a.iter().zip(z).fold(log_hazard.clone(), |acc, (a, z)| acc.add(&a.mul(z)));
                    let rate = exp(&log_rate);
                    value = value.sub(&rate);
                    for k in 0..atoms {
                        gradient[k] = gradient[k].sub(&rate.mul(&a[k]));
                        for j in 0..atoms {
                            precision[k * atoms + j] = precision[k * atoms + j].add(&rate.mul(&a[k]).mul(&a[j]));
                        }
                    }
                }
            }
            (value, gradient, precision)
        };
        let mut means = vec![zero.clone(); atoms];
        // Fixed iteration depth carries parameter sensitivities through the
        // converged mode, including when its primal value has stopped moving.
        for _ in 0..24 {
            let (value, gradient, precision) = evaluate(&means);
            let step = solve(&precision, &gradient)?;
            let mut scale = 1.0;
            let mut next: Vec<S> = means.iter().zip(&step).map(|(z, d)| z.add(d)).collect();
            let mut accepted = false;
            for _ in 0..40 {
                let proposed = evaluate(&next).0.value();
                if proposed.is_finite() && proposed >= value.value() - 16.0 * f64::EPSILON * (1.0 + value.value().abs()) {
                    accepted = true;
                    break;
                }
                scale *= 0.5;
                next = means.iter().zip(&step).map(|(z, d)| z.add(&d.scale(scale))).collect();
            }
            if !accepted { return Err(failure("static posterior grid placement did not converge")); }
            means = next;
        }
        let (value, gradient, precision) = evaluate(&means);
        if !value.value().is_finite() || gradient.iter().any(|g| !g.value().is_finite() || g.value().abs() > 1e-8) {
            return Err(failure("static posterior grid placement has an unresolved score"));
        }
        let mut scales = Vec::with_capacity(atoms);
        for k in 0..atoms {
            let mut unit = vec![zero.clone(); atoms];
            unit[k] = like.constant_like(1.0);
            scales.push(sqrt(&solve(&precision, &unit)?[k]));
        }
        Ok(Grid::new(inputs.gh, &means, &scales, like))
    }

    /// `ln ∫ φ(z) exp(linear · z − Σ_d exp(log_hazard_d + a_d · z)) dz`, on the
    /// grid placed from these statistics.
    fn log_integral(&self, inputs: &SubjectInputs<'_, S>) -> Result<S, EventHistoryError> {
        let grid = self.place(inputs)?;
        let atoms = grid.dimension();
        let half_log_tau = 0.5 * (2.0 * std::f64::consts::PI).ln();
        let terms: Vec<S> = (0..grid.size()).map(|i| {
            let mut value = ln(&grid.weights[i]);
            for k in 0..atoms {
                let z = grid.coordinate(i, k);
                value = add_real(&value.add(&self.linear[k].mul(z)).sub(&z.mul(z).scale(0.5)), -half_log_tau);
            }
            for (d, log_hazard) in self.log_hazards.iter().enumerate() {
                if let Some(log_hazard) = log_hazard {
                    let log_rate = (0..atoms).fold(log_hazard.clone(),
                        |acc, k| acc.add(&inputs.loadings[d * atoms + k].mul(grid.coordinate(i, k))));
                    value = value.sub(&exp(&log_rate));
                }
            }
            value
        }).collect();
        Ok(log_sum_exp(&terms))
    }
}

pub(crate) fn posterior_grid<S: JetField>(
    inputs: &SubjectInputs<'_, S>, compensated: &[bool],
) -> Result<Grid<S>, EventHistoryError> {
    Statistics::whole(inputs, compensated).place(inputs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chain::GaussHermite;
    use crate::cohort::{CovariateSegment, Event, EventHistoryCohort, MarkKind, SubjectHistory, expand_nodes};
    use crate::marginal::spells;
    use ndarray::Array2;

    /// The first event of the #2962 fixture; the later ones follow 0.05 apart.
    const FIRST: f64 = 0.0005;

    /// The fit's Gauss-Hermite order and the two orders its certificate steps
    /// to, `2G − 1`.
    const ORDERS: [usize; 3] = [9, 17, 33];

    fn emit(line: &str) {
        use std::io::Write;
        let mut out = std::io::stdout().lock();
        if out.write_all(line.as_bytes()).is_ok() && out.write_all(b"\n").is_ok() {
            return;
        }
    }

    fn event_time(k: usize) -> f64 {
        FIRST + 0.05 * k as f64
    }

    /// The fixture's first `events` events, all of mark 0, followed to `exit`.
    fn history(events: usize, exit: f64) -> SubjectHistory {
        SubjectHistory {
            id: "2962".to_string(),
            entry: 0.0,
            exit,
            events: (0..events).map(|k| Event { time: event_time(k), mark: 0 }).collect(),
            segments: vec![CovariateSegment { start: 0.0, row: 0 }],
        }
    }

    /// Every spell's PIT and mark probabilities as `predictive_pit` reads them:
    /// one static standard-normal factor, marks `(η⁰_d, a_d)` at rate
    /// `exp(η⁰_d − a_d²/2 + a_d z)`, no risk-set normaliser.
    fn native(marks: &[(f64, f64)], subject: SubjectHistory, order: usize) -> Vec<(f64, Vec<f64>)> {
        let count = marks.len();
        let cohort = EventHistoryCohort {
            mark_names: (0..count).map(|d| format!("m{d}")).collect(),
            mark_kinds: vec![MarkKind::Recurrent; count],
            covariate_names: Vec::new(),
            covariate_levels: Vec::new(),
            covariates: Array2::zeros((1, 0)),
            subjects: vec![subject],
        };
        let nodes = expand_nodes(&cohort, 9, 0).expect("nodes");
        let subject = &nodes.subjects[0];
        let eta0: Vec<f64> = marks.iter().map(|m| m.0).cycle().take(subject.len() * count).collect();
        let loadings: Vec<f64> = marks.iter().map(|m| m.1).collect();
        let gh = GaussHermite::new(order).expect("rule");
        let inputs = SubjectInputs {
            nodes: subject,
            eta0: &eta0,
            loadings: &loadings,
            rates: &[0.0],
            time_scale: 1.0,
            gh: &gh,
            continuation_gap: 0.0,
            designs: None,
            log_normaliser: None,
        };
        spells(&inputs, &vec![true; count])
            .expect("spells")
            .into_iter()
            .map(|spell| {
                let probabilities = spell.intensities.map_or_else(Vec::new, |intensities| {
                    let total: f64 = intensities.iter().sum();
                    intensities.iter().map(|v| v / total).collect()
                });
                (-spell.log_survival.exp_m1(), probabilities)
            })
            .collect()
    }

    /// Every spell at each order in [`ORDERS`], from the fixture's first
    /// `events` events followed to `exit`.
    fn at_orders(marks: &[(f64, f64)], events: usize, exit: f64) -> Vec<Vec<(f64, Vec<f64>)>> {
        ORDERS.iter().map(|&order| native(marks, history(events, exit), order)).collect()
    }

    /// The error scale of a value at the fit's order: its change across the
    /// certificate's two steps.
    fn refinement(values: [f64; 3]) -> f64 {
        (values[0] - values[1]).abs() + (values[1] - values[2]).abs()
    }

    /// The error scale of the value at the certificate's top order: its last
    /// refinement change, `|v17 − v33|`, which bounds that order's error when a
    /// refinement step does not grow the error.
    fn last_step(values: [f64; 3]) -> f64 {
        (values[1] - values[2]).abs()
    }

    /// `ln ∫_{−L}^{L} exp(f(z)) dz` by the 20-point Gauss-Legendre rule on
    /// `cells` equal cells, with a bound on the computed logarithm's roundoff.
    ///
    /// The bound is the canonical running error bound (Higham, *Accuracy and
    /// Stability*, ch. 3), accumulated in the summation loop itself.
    /// - `f` returns its value and the summed magnitudes `m` of the operands it
    ///   forms.
    /// - A log term `t_i = ln w_i + f(z_i)` is off by at most `ε (|ln w_i| + m_i)`.
    /// - `u_i = exp(t_i − shift)` has relative error at most `ε r_i`, with
    ///   `r_i = 1 + |ln w_i| + m_i + |t_i| + |shift|`.
    /// - Adding `u_i` to the partial sum gives `s_i` and adds `ε s_i`.
    /// - So the sum is off by at most `Σ_i ε (u_i r_i + s_i)`.
    /// - The logarithm divides that by `s_T`, and restoring the shift adds
    ///   `ε (|ln s_T| + |shift|)`.
    fn log_integral(f: &dyn Fn(f64) -> (f64, f64), half_width: f64, cells: usize) -> (f64, f64) {
        let (nodes, weights) = gam_math::special::gauss_legendre(20);
        let width = 2.0 * half_width / cells as f64;
        let mut terms = Vec::with_capacity(cells * nodes.len());
        for c in 0..cells {
            let left = -half_width + width * c as f64;
            for (x, w) in nodes.iter().zip(weights.iter()) {
                let log_weight = (0.5 * width * w).ln();
                let (value, magnitude) = f(left + 0.5 * width * (1.0 + x));
                terms.push((log_weight + value, log_weight.abs() + magnitude));
            }
        }
        let shift = terms.iter().fold(f64::NEG_INFINITY, |a, t| a.max(t.0));
        let (mut sum, mut error) = (0.0_f64, 0.0_f64);
        for &(t, magnitude) in &terms {
            let u = (t - shift).exp();
            sum += u;
            error += f64::EPSILON * (u * (1.0 + magnitude + t.abs() + shift.abs()) + sum);
        }
        let log_sum = sum.ln();
        (log_sum + shift, error / sum + f64::EPSILON * (log_sum.abs() + shift.abs()))
    }

    /// The spell after `events` events over `(start, end]`, by one-dimensional
    /// integration at one resolution: its PIT and the mark probabilities at
    /// `end`, each with its roundoff bound (see [`log_integral`]).
    fn resolved(marks: &[(f64, f64)], events: usize, start: f64, end: f64, half_width: f64, cells: usize)
        -> (f64, f64, Vec<f64>, Vec<f64>) {
        // `r_d(z) = η⁰_d − a_d²/2 + a_d z` and the summed magnitudes of its operands.
        let log_rate = |d: usize, z: f64| {
            let (eta, a) = marks[d];
            (eta - 0.5 * a * a + a * z, eta.abs() + 0.5 * a * a + (a * z).abs())
        };
        // The prior times the likelihood of the first `events` events with
        // exposure `exposure`, and the summed magnitudes of its operands. A hazard
        // `exposure e^{r}` carries relative error `ε (1 + m_r)`.
        let prefix = |exposure: f64, z: f64| {
            let (r0, m0) = log_rate(0, z);
            let (mut value, mut magnitude) = (-0.5 * z * z + events as f64 * r0, 0.5 * z * z + events as f64 * m0);
            for d in 0..marks.len() {
                let (r, m) = log_rate(d, z);
                let hazard = exposure * r.exp();
                value -= hazard;
                magnitude += hazard * (1.0 + m) + value.abs();
            }
            (value, magnitude)
        };
        let (opened, opened_roundoff) = log_integral(&|z| prefix(start, z), half_width, cells);
        let (before, before_roundoff) = log_integral(&|z| prefix(end, z), half_width, cells);
        let tilted: Vec<(f64, f64)> = (0..marks.len())
            .map(|d| log_integral(&|z| {
                let (value, magnitude) = prefix(end, z);
                let (r, m) = log_rate(d, z);
                (value + r, magnitude + m + (value + r).abs())
            }, half_width, cells))
            .collect();
        // The log survival `before − opened` is off by at most both bounds plus
        // `ε (|before| + |opened|)`; `expm1` scales that by the survival and adds
        // `ε` of its own result.
        let pit = -(before - opened).exp_m1();
        let pit_roundoff = (before - opened).exp()
            * (before_roundoff + opened_roundoff + f64::EPSILON * (before.abs() + opened.abs()))
            + f64::EPSILON * pit.abs();
        // The probabilities normalise `v_d = exp(t_d − shift)` by their running
        // sum. Each `v_d` has relative error `ρ_d = δ_d + ε (1 + |t_d| + |shift|)`,
        // and the sum's absolute error accumulates in the same loop.
        let shift = tilted.iter().fold(f64::NEG_INFINITY, |a, t| a.max(t.0));
        let (mut total, mut error) = (0.0_f64, 0.0_f64);
        let mut relative = Vec::with_capacity(marks.len());
        for &(t, roundoff) in &tilted {
            let v = (t - shift).exp();
            let rho = roundoff + f64::EPSILON * (1.0 + t.abs() + shift.abs());
            total += v;
            error += v * rho + f64::EPSILON * total;
            relative.push(rho);
        }
        let probabilities: Vec<f64> = tilted.iter().map(|t| (t.0 - shift).exp() / total).collect();
        let probability_roundoffs: Vec<f64> = probabilities
            .iter()
            .zip(relative.iter())
            .map(|(p, rho)| p * (rho + error / total) + f64::EPSILON * p)
            .collect();
        (pit, pit_roundoff, probabilities, probability_roundoffs)
    }

    struct Reference {
        pit: f64,
        pit_error: f64,
        probabilities: Vec<f64>,
        probability_errors: Vec<f64>,
    }

    /// The spell resolved with its own error scale: the value at `L = 12` on
    /// 192 cells, its change when every cell is halved and when the interval
    /// widens to `L = 16`, and the roundoff bounds of the fine and wide values.
    fn reference(marks: &[(f64, f64)], events: usize, start: f64, end: f64) -> Reference {
        let coarse = resolved(marks, events, start, end, 12.0, 96);
        let fine = resolved(marks, events, start, end, 12.0, 192);
        let wide = resolved(marks, events, start, end, 16.0, 256);
        Reference {
            pit: fine.0,
            pit_error: (fine.0 - coarse.0).abs() + (wide.0 - fine.0).abs() + fine.1 + wide.1,
            probability_errors: (0..marks.len())
                .map(|d| (fine.2[d] - coarse.2[d]).abs() + (wide.2[d] - fine.2[d]).abs() + fine.3[d] + wide.3[d])
                .collect(),
            probabilities: fine.2,
        }
    }

    /// Record a failure unless `value` and `other` are the same float, bit for bit.
    fn identical(failures: &mut Vec<String>, what: &str, value: f64, other: f64) {
        if value.to_bits() != other.to_bits() {
            failures.push(format!("{what}: {value:.17e} vs {other:.17e}, not bit-identical"));
        }
    }

    /// Record a failure unless `value` and `other` agree within `bar`.
    fn check(failures: &mut Vec<String>, what: &str, value: f64, other: f64, bar: f64) {
        let gap = (value - other).abs();
        if !(gap <= bar) {
            failures.push(format!("{what}: {value:.12e} vs {other:.12e}, gap {gap:.3e} above the error scale {bar:.3e}"));
        }
    }

    /// #2962 at fixed parameters: one static factor, a recurrent mark at rate
    /// `20 exp(z − ½)`, twenty events in a unit window, the first at 0.0005; a
    /// second fixture adds a never-firing mark at rate `exp(−z − ½)`, so the
    /// mark probabilities are not trivially one. Every spell of the full
    /// history must equal the same spell from the history known at its end, and
    /// from that history censored at 1, bit for bit at every order: the histories
    /// share every node before the spell's end and a spell reads nothing else.
    /// Every spell must also agree with one-dimensional integration: at the
    /// certificate's top order within its last refinement change plus the
    /// oracle's own error, and at the fit's order within the change across the
    /// whole sweep plus that bound. Neither arm relies on the sweep's errors
    /// keeping one sign.
    #[test]
    fn static_spells_do_not_depend_on_later_history_2962() {
        let one = [(20.0_f64.ln(), 1.0)];
        let two = [(20.0_f64.ln(), 1.0), (0.0, -1.0)];
        let mut failures: Vec<String> = Vec::new();
        for (fixture, marks) in [("one mark", &one[..]), ("two marks", &two[..])] {
            let full = at_orders(marks, 20, 1.0);
            assert!(full.iter().all(|order| order.len() == 21), "{fixture}: twenty event spells and the censored tail");
            for k in 0..21 {
                let (label, events, start, end) = if k < 20 {
                    (format!("spell {}", k + 1), k, if k == 0 { 0.0 } else { event_time(k - 1) }, event_time(k))
                } else {
                    ("the censored tail".to_string(), 20, event_time(19), 1.0)
                };
                let exact = reference(marks, events, start, end);
                let what = format!("{fixture}, {label}");
                let pits = [full[0][k].0, full[1][k].0, full[2][k].0];
                emit(&format!(
                    "[2962] {what}: PIT {:.12e} at order 9, {:.12e} at 17, {:.12e} at 33; oracle {:.12e} ± {:.3e}",
                    pits[0],
                    pits[1],
                    pits[2],
                    exact.pit,
                    exact.pit_error
                ));
                check(&mut failures, &format!("{what}: PIT at order 33 against the oracle"), pits[2], exact.pit,
                    last_step(pits) + exact.pit_error);
                check(&mut failures, &format!("{what}: PIT at order 9 against the oracle"), pits[0], exact.pit,
                    refinement(pits) + last_step(pits) + exact.pit_error);
                for d in 0..full[0][k].1.len() {
                    let probabilities = [full[0][k].1[d], full[1][k].1[d], full[2][k].1[d]];
                    check(&mut failures, &format!("{what}: mark {d} probability at order 33 against the oracle"),
                        probabilities[2], exact.probabilities[d], last_step(probabilities) + exact.probability_errors[d]);
                    check(&mut failures, &format!("{what}: mark {d} probability at order 9 against the oracle"),
                        probabilities[0], exact.probabilities[d],
                        refinement(probabilities) + last_step(probabilities) + exact.probability_errors[d]);
                }
                if k == 20 {
                    continue;
                }
                for (variant, exit) in [("ending at its event", end), ("censored after it", 1.0)] {
                    let known = at_orders(marks, k + 1, exit);
                    for (order, (ours, theirs)) in ORDERS.iter().zip(full.iter().zip(known.iter())) {
                        identical(&mut failures, &format!("{what}, order {order}: PIT against the history {variant}"),
                            ours[k].0, theirs[k].0);
                        for d in 0..ours[k].1.len() {
                            identical(&mut failures,
                                &format!("{what}, order {order}: mark {d} probability against the history {variant}"),
                                ours[k].1[d], theirs[k].1[d]);
                        }
                    }
                }
            }
        }
        assert!(failures.is_empty(), "#2962 static spells:\n{}", failures.join("\n"));
    }
}
