//! Exact marginal likelihood of one subject's event history over its latent
//! chain, with the exact gradient and Hessian in every parameter the subject
//! sees: its node log-intensities `η⁰`, the loadings `a`, and the log-rates
//! `ρ`.
//!
//! The chain is marginalised by forward filtering on the adaptive product
//! Gauss-Hermite grid of [`super::chain`]; derivatives come from the two
//! identities that hold for any latent-variable model whose complete-data
//! log-likelihood `L_c` is a sum of node terms and gap terms:
//!
//! ```text
//! ∂ℓ/∂θ     = E[∂L_c/∂θ | y]                                   (Fisher)
//! ∂²ℓ/∂θ∂θᵀ = E[∂²L_c/∂θ∂θᵀ | y] + Cov(∂L_c/∂θ | y)             (Louis)
//! ```
//!
//! Every expectation and covariance reduces to node marginals, consecutive
//! pairwise marginals, and all-pairs covariances of node functions, which the
//! backward operator propagates in one sweep per node. Gap functions are
//! quadratic in the two endpoint states, so their conditional expectations
//! are polynomial moments of the same operators.

use super::chain::{
    AtomTransition, GaussHermite, Grid, OperatorFamily, apply_axis, backward_axis_bases,
    forward_operators, interpolate_at_inner_points, log_sum_exp, normal_density,
};
use super::cohort::{EventHistoryError, SubjectNodes};
use super::scalar::{add_real, div, exp, ln, recip, sqrt, square};
use gam_math::nested_dual::JetField;
use std::collections::HashMap;

/// Everything one subject's marginal needs, in the caller's scalar type.
pub(crate) struct SubjectInputs<'a, S> {
    pub nodes: &'a SubjectNodes,
    /// Node log-intensities without the latent part, index `n * marks + d`.
    pub eta0: &'a [S],
    /// Loadings, index `d * atoms + k`.
    pub loadings: &'a [S],
    /// `ln(rate · time_scale)` per atom.
    pub log_rates: &'a [S],
    pub time_scale: f64,
    pub gh: &'a GaussHermite,
    /// Elapsed time between a supplied filtered state and the first node;
    /// zero unless [`forward_filter`] continues from an earlier state.
    pub continuation_gap: f64,
}

/// Marginal log-likelihood and its derivatives in the subject-local parameter
/// vector `[η⁰ (nodes × marks) | a (marks × atoms) | ρ (atoms)]`.
pub(crate) struct SubjectOutput<S> {
    pub loglik: S,
    pub gradient: Vec<S>,
    /// Row-major `P × P`; empty when derivatives were not requested.
    pub hessian: Vec<S>,
}

/// Relative floor on a quadrature-estimated posterior variance, guarding the
/// grid placement against a posterior that collapsed below the resolution of
/// the current grid.
const VARIANCE_FLOOR_RELATIVE: f64 = 1e-12;

/// A filtered or predicted density value below this fraction of its grid's
/// peak is interpolation noise (the Lagrange-based operator is a signed sum,
/// accurate to about `ε · peak` in absolute terms), and the future-likelihood
/// ratio that multiplies it in the smoothed marginal is exponentially large
/// exactly there. Such points carry no smoothed mass. `1e-11` sits two
/// orders above the noise of a product grid of a few hundred points and
/// discards mass far below any discretisation error.
const DENSITY_NOISE_RELATIVE: f64 = 1e-11;

/// The noise floor of a density on a grid: `DENSITY_NOISE_RELATIVE` of its
/// largest value.
fn density_floor<S: JetField>(values: &[S]) -> f64 {
    DENSITY_NOISE_RELATIVE * values.iter().map(|v| v.value()).fold(0.0, f64::max)
}

fn numerical(reason: impl Into<String>) -> EventHistoryError {
    EventHistoryError::NumericalFailure {
        reason: reason.into(),
    }
}

/// A bivariate polynomial in the start and end coordinates of one atom
/// across one gap, degree at most four in each variable.
#[derive(Clone)]
struct GapPolynomial<S> {
    /// `c[a * 5 + b]` multiplies `z^a z'^b`.
    c: Vec<S>,
    /// Structural support: which coefficients were ever set. A coefficient
    /// whose value happens to be zero may still carry derivative channels,
    /// so sparsity is tracked by construction, never read off the value.
    present: Vec<bool>,
}

impl<S: JetField> GapPolynomial<S> {
    fn zero(like: &S) -> Self {
        Self {
            c: vec![like.constant_like(0.0); 25],
            present: vec![false; 25],
        }
    }

    fn set(&mut self, a: usize, b: usize, value: S) {
        self.c[a * 5 + b] = value;
        self.present[a * 5 + b] = true;
    }

    fn get(&self, a: usize, b: usize) -> &S {
        &self.c[a * 5 + b]
    }

    /// Whether the coefficient of `z^a z'^b` is structurally absent.
    fn absent(&self, a: usize, b: usize) -> bool {
        !self.present[a * 5 + b]
    }

    fn scaled(&self, factor: &S) -> Self {
        Self {
            c: self.c.iter().map(|v| v.mul(factor)).collect(),
            present: self.present.clone(),
        }
    }

    fn add(&self, other: &Self) -> Self {
        Self {
            c: self
                .c
                .iter()
                .zip(other.c.iter())
                .map(|(x, y)| x.add(y))
                .collect(),
            present: self
                .present
                .iter()
                .zip(other.present.iter())
                .map(|(x, y)| x | y)
                .collect(),
        }
    }

    /// Product, exact when the total degree stays at most four per variable.
    fn mul(&self, other: &Self) -> Self {
        let zero = self.c[0].constant_like(0.0);
        let mut out = Self {
            c: vec![zero; 25],
            present: vec![false; 25],
        };
        for a in 0..5 {
            for b in 0..5 {
                if self.absent(a, b) {
                    continue;
                }
                let x = self.get(a, b);
                for a2 in 0..5 - a {
                    for b2 in 0..5 - b {
                        if other.absent(a2, b2) {
                            continue;
                        }
                        let y = other.get(a2, b2);
                        let idx = (a + a2) * 5 + (b + b2);
                        out.c[idx] = out.c[idx].add(&x.mul(y));
                        out.present[idx] = true;
                    }
                }
            }
        }
        out
    }
}

/// The score `t = ∂ ln p(z'|z) / ∂ρ` of one atom across one gap and its own
/// derivative `∂t/∂ρ`, as polynomials in the start state `z` and the
/// standardised innovation `u = (z' − φz)/√(1 − φ²)`.
///
/// In these coordinates every coefficient stays bounded as the gap shrinks
/// (`κ → 0`, `φ → 1`, `q → 0`); the monomial expansion in `(z, z')` carries
/// coefficients of order `1/q²` that cancel to `O(1)` and destroy the
/// quadrature moments in floating point.
fn gap_score_polynomials<S: JetField>(
    transition: &AtomTransition<S>,
    like: &S,
) -> (GapPolynomial<S>, GapPolynomial<S>) {
    let phi = &transition.phi;
    let v = &transition.innovation;
    let inv_v = recip(v);
    let inv_root_v = sqrt(&inv_v);
    let inv_v2 = square(&inv_v);
    let phi2 = square(phi);
    // dL/dφ = (φ/v)(1 − u²) + z u / √v
    let mut dl = GapPolynomial::zero(like);
    dl.set(0, 0, phi.mul(&inv_v));
    dl.set(0, 2, phi.mul(&inv_v).neg());
    dl.set(1, 1, inv_root_v.clone());
    // d²L/dφ² = (1+φ²)/v² − z²/v + 4φ z u / v^{3/2} − (1+3φ²) u²/v²
    let mut d2l = GapPolynomial::zero(like);
    d2l.set(0, 0, add_real(&phi2, 1.0).mul(&inv_v2));
    d2l.set(2, 0, inv_v.neg());
    d2l.set(1, 1, phi.mul(&inv_v).mul(&inv_root_v).scale(4.0));
    d2l.set(0, 2, add_real(&phi2.scale(3.0), 1.0).mul(&inv_v2).neg());
    let t = dl.scaled(&transition.dphi);
    let dt = d2l
        .scaled(&square(&transition.dphi))
        .add(&dl.scaled(&transition.d2phi));
    (t, dt)
}

/// The score `∂ ln p(z'|z)/∂ρ` of one atom across a gap of dimensionless
/// length `kappa`, and its derivative in `ρ`, as flat coefficient vectors
/// `c[a * 5 + b]` multiplying `z^a u^b` with `u = (z' − φz)/√(1 − φ²)`.
pub fn transition_score_polynomials(kappa: f64) -> (Vec<f64>, Vec<f64>) {
    let transition = AtomTransition::new(&kappa);
    let (t, dt) = gap_score_polynomials(&transition, &0.0);
    (t.c, dt.c)
}

fn pointwise<S: JetField>(a: &[S], b: &[S]) -> Vec<S> {
    a.iter().zip(b.iter()).map(|(x, y)| x.mul(y)).collect()
}

fn weighted_sum<S: JetField>(weights: &[S], values: &[S]) -> S {
    weights
        .iter()
        .zip(values.iter())
        .fold(weights[0].constant_like(0.0), |acc, (w, v)| acc.add(&w.mul(v)))
}

/// Pairwise (tree) sum of `terms`: rounding error grows like `log₂ n`
/// rather than `n`, which keeps a log-likelihood summed over thousands of
/// node terms resolvable at the level a Newton acceptance test needs.
pub(crate) fn pairwise_sum<S: JetField>(terms: &[S], zero: &S) -> S {
    match terms.len() {
        0 => zero.clone(),
        1 => terms[0].clone(),
        2 => terms[0].add(&terms[1]),
        n => {
            let (left, right) = terms.split_at(n / 2);
            pairwise_sum(left, zero).add(&pairwise_sum(right, zero))
        }
    }
}

fn weighted_sum3<S: JetField>(weights: &[S], a: &[S], b: &[S]) -> S {
    weights
        .iter()
        .zip(a.iter().zip(b.iter()))
        .fold(weights[0].constant_like(0.0), |acc, (w, (x, y))| {
            acc.add(&w.mul(x).mul(y))
        })
}

/// Node log-likelihood pieces at one grid.
struct NodeLikelihood<S> {
    /// `exp(η_{nd})` at every grid point, index `d * size + i`.
    expeta: Vec<S>,
    /// `Σ_d y η − w e^η` at every grid point.
    ell: Vec<S>,
    /// `max_i ell[i].value()`.
    shift: f64,
}

fn node_likelihood<S: JetField>(
    grid: &Grid<S>,
    eta0: &[S],
    loadings: &[S],
    counts: &[f64],
    exposure: f64,
    compensated: Option<&[bool]>,
    marks: usize,
    atoms: usize,
) -> NodeLikelihood<S> {
    let size = grid.size();
    let mut expeta = Vec::with_capacity(marks * size);
    let mut ell = vec![eta0[0].constant_like(0.0); size];
    for d in 0..marks {
        let exposure = if compensated.is_none_or(|mask| mask[d]) {
            exposure
        } else {
            0.0
        };
        for i in 0..size {
            let mut eta = eta0[d].clone();
            for k in 0..atoms {
                eta = eta.add(&loadings[d * atoms + k].mul(grid.coordinate(i, k)));
            }
            let e = exp(&eta);
            let y = counts[d];
            if y != 0.0 {
                ell[i] = ell[i].add(&eta.scale(y));
            }
            if exposure != 0.0 {
                ell[i] = ell[i].sub(&e.scale(exposure));
            }
            expeta.push(e);
        }
    }
    let shift = ell
        .iter()
        .map(|v| v.value())
        .fold(f64::NEG_INFINITY, f64::max);
    NodeLikelihood { expeta, ell, shift }
}

/// Posterior mean and variance of every atom under `alpha` on `grid`, the
/// variance floored relative to the grid's own scale.
fn posterior_moments<S: JetField>(grid: &Grid<S>, alpha: &[S]) -> (Vec<S>, Vec<S>) {
    let atoms = grid.dimension();
    let mut means = Vec::with_capacity(atoms);
    let mut variances = Vec::with_capacity(atoms);
    for k in 0..atoms {
        let zs: Vec<S> = (0..grid.size())
            .map(|i| grid.coordinate(i, k).clone())
            .collect();
        let mean = weighted_sum3(&grid.weights, alpha, &zs);
        let centred: Vec<S> = zs.iter().map(|z| square(&z.sub(&mean))).collect();
        let mut variance = weighted_sum3(&grid.weights, alpha, &centred);
        let floor = VARIANCE_FLOOR_RELATIVE * square(&grid.axes[k].sigma).value();
        if !(variance.value() > floor) {
            variance = variance.constant_like(floor);
        }
        means.push(mean);
        variances.push(variance);
    }
    (means, variances)
}

/// Multiply a predicted density on `grid` by the node factor
/// `exp(ell − shift)` and normalise; returns the filtered density and the
/// normaliser `c`.
fn condition<S: JetField>(
    grid: &Grid<S>,
    predicted: &[S],
    ell: &[S],
    shift: f64,
    label: &str,
) -> Result<(Vec<S>, S), EventHistoryError> {
    let mut raw: Vec<S> = predicted
        .iter()
        .zip(ell.iter())
        .map(|(p, e)| p.mul(&exp(&add_real(e, -shift))))
        .collect();
    let c = weighted_sum(&grid.weights, &raw);
    if !(c.value() > 0.0) || !c.value().is_finite() {
        return Err(numerical(format!(
            "{label}: normaliser is not positive ({})",
            c.value()
        )));
    }
    let inverse = recip(&c);
    for v in raw.iter_mut() {
        *v = v.mul(&inverse);
    }
    Ok((raw, c))
}

/// `exp(ell − shift) / c` at every grid point.
fn node_factors<S: JetField>(likelihood: &NodeLikelihood<S>, normaliser: &S) -> Vec<S> {
    let inverse = recip(normaliser);
    likelihood
        .ell
        .iter()
        .map(|e| exp(&add_real(e, -likelihood.shift)).mul(&inverse))
        .collect()
}

/// One filtered node: its grid, the operators that reached it, the predicted
/// and filtered densities on it, and the node's likelihood pieces.
struct FilteredNode<S> {
    grid: Grid<S>,
    /// Transitions across the gap that led here; empty at the first node.
    transitions: Vec<AtomTransition<S>>,
    /// Forward operators from the previous grid; empty at the first node.
    forward: OperatorFamily<S>,
    predicted: Vec<S>,
    alpha: Vec<S>,
    normaliser: S,
    /// `exp(ell − shift) / c` at every grid point.
    factors: Vec<S>,
    likelihood: NodeLikelihood<S>,
}

/// Where a node's grid goes: at the posterior mean, with the predictive
/// spread.
///
/// The forward operator is exact for `envelope × polynomial`, so what it
/// interpolates is the filtered density divided by the grid's Gaussian
/// envelope. On a grid placed at the *predicted* moments that ratio is the
/// node's likelihood factor itself — after an event, an exponential tilt
/// `exp(a z)` — and a degree-`G−1` interpolant of an exponential across a
/// hull of `±x_max √2 σ` has truncation error that grows like
/// `(a · hull)^G / G!`, astronomically large at the orders the quadrature
/// certificate reaches. Re-centring the grid on the posterior mean absorbs
/// the tilt into the envelope (a tilted Gaussian is a shifted Gaussian).
///
/// The spread stays the predictive one. The Poisson node factor is
/// log-concave with at most linear growth in `z`, so the posterior is
/// dominated by a shifted Gaussian of the predictive variance: against that
/// envelope the ratio is a bounded, mild tilt, and both the interpolation and
/// the quadrature are benign at any order. Scaling the envelope down to the
/// posterior variance would leave the integrand's Gaussian tails outside it
/// and break the quadrature instead (a survival forecast above one is the
/// symptom). The variance still narrows across nodes, through the predictive
/// recursion. The mean read off the predictive grid is accurate even when
/// interpolation from it would not be: Gauss-Hermite integrates the tilted
/// envelope itself to near machine precision.
fn filter_start<S: JetField>(
    gh: &GaussHermite,
    like: &S,
    atoms: usize,
    node_terms: &dyn Fn(&Grid<S>) -> NodeLikelihood<S>,
    label: &str,
) -> Result<FilteredNode<S>, EventHistoryError> {
    let zero: Vec<S> = (0..atoms).map(|_| like.constant_like(0.0)).collect();
    let unit: Vec<S> = (0..atoms).map(|_| like.constant_like(1.0)).collect();
    let prior_density = |grid: &Grid<S>| -> Vec<S> {
        (0..grid.size())
            .map(|i| {
                let mut density = like.constant_like(1.0);
                for k in 0..atoms {
                    density = density.mul(&normal_density(grid.coordinate(i, k), &zero[k], &unit[k]));
                }
                density
            })
            .collect()
    };
    let prior_grid = Grid::new(gh, &zero, &unit, like);
    let rough = node_terms(&prior_grid);
    let (rough_alpha, _) = condition(
        &prior_grid,
        &prior_density(&prior_grid),
        &rough.ell,
        rough.shift,
        label,
    )?;
    let (means, _) = posterior_moments(&prior_grid, &rough_alpha);
    let grid = Grid::new(gh, &means, &unit, like);
    let predicted = prior_density(&grid);
    let likelihood = node_terms(&grid);
    let (alpha, normaliser) = condition(&grid, &predicted, &likelihood.ell, likelihood.shift, label)?;
    let factors = node_factors(&likelihood, &normaliser);
    Ok(FilteredNode {
        grid,
        transitions: Vec::new(),
        forward: OperatorFamily {
            per_axis: Vec::new(),
            order: gh.order,
        },
        predicted,
        alpha,
        normaliser,
        factors,
        likelihood,
    })
}

/// Predict the filtered state on `previous_grid` across one gap and condition
/// on a node, on a grid placed at the posterior mean with the predictive
/// spread (see [`filter_start`]).
fn filter_step<S: JetField>(
    gh: &GaussHermite,
    like: &S,
    previous_grid: &Grid<S>,
    previous_alpha: &[S],
    transitions: Vec<AtomTransition<S>>,
    forward_power: u8,
    node_terms: &dyn Fn(&Grid<S>) -> NodeLikelihood<S>,
    label: &str,
) -> Result<FilteredNode<S>, EventHistoryError> {
    let atoms = transitions.len();
    let (means, variances) = posterior_moments(previous_grid, previous_alpha);
    let centres: Vec<S> = (0..atoms)
        .map(|k| transitions[k].phi.mul(&means[k]))
        .collect();
    let scales: Vec<S> = (0..atoms)
        .map(|k| {
            sqrt(
                &square(&transitions[k].phi)
                    .mul(&variances[k])
                    .add(&transitions[k].innovation),
            )
        })
        .collect();
    let predictive = Grid::new(gh, &centres, &scales, like);
    let rough_forward = forward_operators(gh, previous_grid, &predictive, &transitions, 0);
    let rough = node_terms(&predictive);
    let (rough_alpha, _) = condition(
        &predictive,
        &rough_forward.plain(previous_alpha),
        &rough.ell,
        rough.shift,
        label,
    )?;
    let (means, _) = posterior_moments(&predictive, &rough_alpha);
    let grid = Grid::new(gh, &means, &scales, like);
    let forward = forward_operators(gh, previous_grid, &grid, &transitions, forward_power);
    let predicted = forward.plain(previous_alpha);
    let likelihood = node_terms(&grid);
    let (alpha, normaliser) = condition(&grid, &predicted, &likelihood.ell, likelihood.shift, label)?;
    let factors = node_factors(&likelihood, &normaliser);
    Ok(FilteredNode {
        grid,
        transitions,
        forward,
        predicted,
        alpha,
        normaliser,
        factors,
        likelihood,
    })
}

/// Evaluate one subject's exact marginal log-likelihood and, when requested,
/// its exact gradient and Hessian.
pub(crate) fn subject_marginal<S: JetField>(
    inputs: &SubjectInputs<'_, S>,
    derivatives: bool,
) -> Result<SubjectOutput<S>, EventHistoryError> {
    let nodes = inputs.nodes;
    let n_nodes = nodes.len();
    let marks = nodes.counts.ncols();
    let atoms = inputs.log_rates.len();
    let gh = inputs.gh;
    if n_nodes == 0 || marks == 0 {
        return Err(numerical("subject marginal needs at least one node and one mark"));
    }
    if inputs.eta0.len() != n_nodes * marks || inputs.loadings.len() != marks * atoms {
        return Err(numerical("subject marginal received mismatched parameter slices"));
    }
    let like = &inputs.eta0[0];
    let counts_rows: Vec<Vec<f64>> = (0..n_nodes)
        .map(|n| nodes.counts.row(n).to_vec())
        .collect();

    // ---- forward filter ------------------------------------------------
    let mut grids: Vec<Grid<S>> = Vec::with_capacity(n_nodes);
    let mut alpha: Vec<Vec<S>> = Vec::with_capacity(n_nodes);
    let mut lik: Vec<Vec<S>> = Vec::with_capacity(n_nodes);
    let mut likelihoods: Vec<NodeLikelihood<S>> = Vec::with_capacity(n_nodes);
    let mut transitions: Vec<Vec<AtomTransition<S>>> = Vec::with_capacity(n_nodes);
    let mut forward_ops: Vec<OperatorFamily<S>> = Vec::with_capacity(n_nodes);
    let mut normalisers: Vec<S> = Vec::with_capacity(n_nodes);
    let mut predicted: Vec<Vec<S>> = Vec::with_capacity(n_nodes);
    let forward_power: u8 = if derivatives { 2 } else { 0 };
    let node_terms = |grid: &Grid<S>, n: usize| -> NodeLikelihood<S> {
        node_likelihood(
            grid,
            &inputs.eta0[n * marks..(n + 1) * marks],
            inputs.loadings,
            &counts_rows[n],
            nodes.exposures[n],
            None,
            marks,
            atoms,
        )
    };
    let first = filter_start(gh, like, atoms, &|grid| node_terms(grid, 0), "first node")?;
    let mut node_loglik: Vec<S> = Vec::with_capacity(n_nodes);
    node_loglik.push(ln(&first.normaliser).add(&like.constant_like(first.likelihood.shift)));
    normalisers.push(first.normaliser);
    predicted.push(first.predicted);
    lik.push(first.factors);
    likelihoods.push(first.likelihood);
    alpha.push(first.alpha);
    grids.push(first.grid);
    for n in 0..n_nodes - 1 {
        let gap_transitions: Vec<AtomTransition<S>> = (0..atoms)
            .map(|k| {
                let kappa = exp(&inputs.log_rates[k]).scale(nodes.gaps[n] / inputs.time_scale);
                AtomTransition::new(&kappa)
            })
            .collect();
        let step = filter_step(
            gh,
            like,
            &grids[n],
            &alpha[n],
            gap_transitions,
            forward_power,
            &|grid| node_terms(grid, n + 1),
            &format!("node {}", n + 1),
        )?;
        node_loglik.push(ln(&step.normaliser).add(&like.constant_like(step.likelihood.shift)));
        normalisers.push(step.normaliser);
        predicted.push(step.predicted);
        lik.push(step.factors);
        likelihoods.push(step.likelihood);
        alpha.push(step.alpha);
        grids.push(step.grid);
        transitions.push(step.transitions);
        forward_ops.push(step.forward);
    }
    let loglik = pairwise_sum(&node_loglik, &like.constant_like(0.0));
    if !loglik.value().is_finite() {
        return Err(numerical("subject marginal log-likelihood is not finite"));
    }
    let p_total = n_nodes * marks + marks * atoms + atoms;
    if !derivatives {
        return Ok(SubjectOutput {
            loglik,
            gradient: Vec::new(),
            hessian: Vec::new(),
        });
    }

    // ---- backward pass: smoothed transition kernels ----------------------
    // The future likelihood `lik_{n+1} β_{n+1}` is an exponential in the state
    // after an event, so it is never interpolated as a value: its logarithm
    // is interpolated (relative accuracy), and everything carried backward is
    // a smoothed conditional expectation `E[f | z_n, data]` — bounded,
    // smooth, and transported by the backward-smoothed transition kernel
    // `P(z_{n+1} | z_n, data) ∝ N(z_{n+1}; φ z_n, q) lik_{n+1} β_{n+1}`, whose
    // self-normalised Gauss-Hermite weights make every transfer a stochastic
    // matrix. Nothing grows along the chain.
    let n_gaps = n_nodes.saturating_sub(1);
    let inner_count = grids[0].size();
    let log_inner_weights: Vec<f64> = (0..inner_count)
        .map(|l| {
            let mut rest = l;
            let mut acc = 0.0;
            for _ in 0..atoms {
                acc += gh.normal_weights[rest % gh.order].ln();
                rest /= gh.order;
            }
            acc
        })
        .collect();
    let inner_innovation = |l: usize, e: &[u8]| -> f64 {
        let mut rest = l;
        let mut acc = 1.0;
        for &power in e.iter() {
            let u = std::f64::consts::SQRT_2 * gh.nodes[rest % gh.order];
            rest /= gh.order;
            acc *= u.powi(i32::from(power));
        }
        acc
    };
    let mut innovation_exponents: Vec<Vec<u8>> = Vec::new();
    for k in 0..atoms {
        for b in 0..=4u8 {
            let mut e = vec![0u8; atoms];
            e[k] = b;
            innovation_exponents.push(e);
        }
        for j in (k + 1)..atoms {
            for b in 1..=2u8 {
                for b2 in 1..=2u8 {
                    let mut e = vec![0u8; atoms];
                    e[k] = b;
                    e[j] = b2;
                    innovation_exponents.push(e);
                }
            }
        }
    }
    let mut log_beta: Vec<Vec<S>> = vec![Vec::new(); n_nodes];
    log_beta[n_nodes - 1] = vec![like.constant_like(0.0); grids[n_nodes - 1].size()];
    // Per gap `n`: the stochastic transfer `transfer[i * size_{n+1} + j]`
    // taking values on grid n+1 to smoothed expectations on grid n, and the
    // smoothed innovation moments `E[Π_k u_k^{e_k} | z_i, data]`.
    let mut transfers: Vec<Vec<S>> = vec![Vec::new(); n_gaps];
    let mut innovation_moments: Vec<HashMap<Vec<u8>, Vec<S>>> = vec![HashMap::new(); n_gaps];
    for n in (0..n_gaps).rev() {
        let grid = &grids[n];
        let next = &grids[n + 1];
        let size = grid.size();
        let next_size = next.size();
        // The node's log-likelihood is an explicit formula, so it is
        // evaluated exactly at every inner point; only the smoother residual
        // `log β_{n+1}` is interpolated.
        let bases = backward_axis_bases(gh, grid, next, &transitions[n]);
        let at_inner = interpolate_at_inner_points(gh.order, &bases, &log_beta[n + 1]);
        let log_c = ln(&normalisers[n + 1]);
        let node_log_lik = |zeta: &[S]| -> S {
            let mut ell = like.constant_like(0.0);
            for d in 0..marks {
                let mut eta = inputs.eta0[(n + 1) * marks + d].clone();
                for k in 0..atoms {
                    eta = eta.add(&inputs.loadings[d * atoms + k].mul(&zeta[k]));
                }
                let y = counts_rows[n + 1][d];
                if y != 0.0 {
                    ell = ell.add(&eta.scale(y));
                }
                if nodes.exposures[n + 1] != 0.0 {
                    ell = ell.sub(&exp(&eta).scale(nodes.exposures[n + 1]));
                }
            }
            add_real(&ell, -likelihoods[n + 1].shift).sub(&log_c)
        };
        let spreads: Vec<S> = transitions[n]
            .iter()
            .map(|t| sqrt(&t.innovation.scale(2.0)))
            .collect();
        let mut log_beta_n = Vec::with_capacity(size);
        let mut weights: Vec<S> = Vec::with_capacity(size * inner_count);
        for i in 0..size {
            let terms: Vec<S> = (0..inner_count)
                .map(|l| {
                    let mut rest_i = i;
                    let mut rest_l = l;
                    let zeta: Vec<S> = (0..atoms)
                        .map(|k| {
                            let point = grid.axes[k].points[rest_i % gh.order].clone();
                            let x = gh.nodes[rest_l % gh.order];
                            rest_i /= gh.order;
                            rest_l /= gh.order;
                            transitions[n][k].phi.mul(&point).add(&spreads[k].scale(x))
                        })
                        .collect();
                    add_real(
                        &node_log_lik(&zeta).add(&at_inner[i * inner_count + l]),
                        log_inner_weights[l],
                    )
                })
                .collect();
            let log_total = log_sum_exp(&terms);
            for term in terms.iter() {
                weights.push(exp(&term.sub(&log_total)));
            }
            log_beta_n.push(log_total);
        }
        let mut transfer = vec![like.constant_like(0.0); size * next_size];
        for i in 0..size {
            let mut current: Vec<S> = weights[i * inner_count..(i + 1) * inner_count].to_vec();
            let mut rest = i;
            for (axis, basis) in bases.iter().enumerate() {
                let i_axis = rest % gh.order;
                rest /= gh.order;
                // matrix[j * G + l] = basis of target j at inner node l of source i_axis.
                let mut matrix = vec![like.constant_like(0.0); gh.order * gh.order];
                for l in 0..gh.order {
                    for j in 0..gh.order {
                        matrix[j * gh.order + l] = basis[(i_axis * gh.order + l) * gh.order + j].clone();
                    }
                }
                current = apply_axis(&current, &matrix, gh.order, axis);
            }
            transfer[i * next_size..(i + 1) * next_size].clone_from_slice(&current);
        }
        let mut moments = HashMap::new();
        for e in innovation_exponents.iter() {
            let values: Vec<S> = (0..size)
                .map(|i| {
                    (0..inner_count).fold(like.constant_like(0.0), |acc, l| {
                        acc.add(&weights[i * inner_count + l].scale(inner_innovation(l, e)))
                    })
                })
                .collect();
            moments.insert(e.clone(), values);
        }
        log_beta[n] = log_beta_n;
        transfers[n] = transfer;
        innovation_moments[n] = moments;
    }
    // `β` alone overflows on a wide hull (it is a future-likelihood ratio,
    // astronomically large where the filtered density is astronomically
    // small), so it is never exponentiated on its own: the smoothed marginal
    // `α β` is formed in log space, and a point whose filtered density has
    // underflowed carries no smoothed mass.
    let smoothed_all: Vec<Vec<S>> = (0..n_nodes)
        .map(|n| {
            let floor = density_floor(&alpha[n]);
            alpha[n]
                .iter()
                .zip(log_beta[n].iter())
                .map(|(a, log_b)| {
                    if a.value() > floor {
                        exp(&ln(a).add(log_b))
                    } else {
                        a.constant_like(0.0)
                    }
                })
                .collect()
        })
        .collect();

    // ---- node functions -------------------------------------------------
    // Slot layout per node: [s_d (marks)] [s_d z_k (marks × atoms)] [gap slot (atoms)].
    let f0 = marks + marks * atoms;
    let f_total = f0 + atoms;
    let mut left: Vec<Vec<Vec<S>>> = Vec::with_capacity(n_nodes);
    let mut right: Vec<Vec<Vec<S>>> = Vec::with_capacity(n_nodes);
    let mut mean_node: Vec<Vec<S>> = Vec::with_capacity(n_nodes);
    // E[c_{nd}], E[c_{nd} z_k], E[c_{nd} z_k z_j]
    let mut expected_curvature: Vec<Vec<S>> = Vec::with_capacity(n_nodes);
    let mut expected_curvature_z: Vec<Vec<S>> = Vec::with_capacity(n_nodes);
    let mut expected_curvature_zz: Vec<Vec<S>> = Vec::with_capacity(n_nodes);
    for n in 0..n_nodes {
        let grid = &grids[n];
        let size = grid.size();
        let exposure = nodes.exposures[n];
        let smoothed = &smoothed_all[n];
        let mut left_n = Vec::with_capacity(f_total);
        let mut right_n = Vec::with_capacity(f_total);
        let mut means = Vec::with_capacity(f_total);
        let mut ec = Vec::with_capacity(marks);
        let mut ecz = Vec::with_capacity(marks * atoms);
        let mut eczz = Vec::with_capacity(marks * atoms * atoms);
        let mut node_functions: Vec<Vec<S>> = Vec::with_capacity(f0);
        for d in 0..marks {
            let y = counts_rows[n][d];
            let score: Vec<S> = (0..size)
                .map(|i| {
                    let e = &likelihoods[n].expeta[d * size + i];
                    add_real(&e.scale(-exposure), y)
                })
                .collect();
            let curvature: Vec<S> = (0..size)
                .map(|i| likelihoods[n].expeta[d * size + i].scale(exposure))
                .collect();
            ec.push(weighted_sum3(&grid.weights, smoothed, &curvature));
            for k in 0..atoms {
                let zk: Vec<S> = (0..size).map(|i| grid.coordinate(i, k).clone()).collect();
                let cz = pointwise(&curvature, &zk);
                ecz.push(weighted_sum3(&grid.weights, smoothed, &cz));
                for j in 0..atoms {
                    let zj: Vec<S> = (0..size).map(|i| grid.coordinate(i, j).clone()).collect();
                    let czz = pointwise(&cz, &zj);
                    eczz.push(weighted_sum3(&grid.weights, smoothed, &czz));
                }
            }
            node_functions.push(score);
        }
        for d in 0..marks {
            for k in 0..atoms {
                let zk: Vec<S> = (0..size).map(|i| grid.coordinate(i, k).clone()).collect();
                node_functions.push(pointwise(&node_functions[d], &zk));
            }
        }
        for f in node_functions.iter() {
            means.push(weighted_sum3(&grid.weights, smoothed, f));
            left_n.push(f.clone());
            right_n.push(f.clone());
        }
        let zeros = vec![like.constant_like(0.0); size];
        for _ in 0..atoms {
            means.push(like.constant_like(0.0));
            left_n.push(zeros.clone());
            right_n.push(zeros.clone());
        }
        left.push(left_n);
        right.push(right_n);
        mean_node.push(means);
        expected_curvature.push(ec);
        expected_curvature_z.push(ecz);
        expected_curvature_zz.push(eczz);
    }

    // ---- gap functions --------------------------------------------------
    let mut mean_gap: Vec<Vec<S>> = vec![vec![like.constant_like(0.0); atoms]; n_gaps];
    let mut expected_dt: Vec<Vec<S>> = vec![vec![like.constant_like(0.0); atoms]; n_gaps];
    // same-gap E[t_k t_j] including k == j
    let mut same_gap_products: Vec<Vec<S>> =
        vec![vec![like.constant_like(0.0); atoms * atoms]; n_gaps];
    for g in 0..n_gaps {
        let grid = &grids[g];
        let next = &grids[g + 1];
        let smoothed = &smoothed_all[g];
        // `backward_moments[e]` is the smoothed innovation moment
        // `E[Π_k u_k^{e_k} | z, data]` on grid g; `forward_moments[(k, a, b)]`
        // is `∫ A(z'|z) u_k^b z_k^a α_g(z) dz` on grid g+1.
        let backward_moments = &innovation_moments[g];
        let mut forward_moments: HashMap<(usize, u8, u8), Vec<S>> = HashMap::new();
        for k in 0..atoms {
            for b in 0..=2u8 {
                let mut e = vec![0u8; atoms];
                e[k] = b;
                for a in 0..=2u8 {
                    let zk_power: Vec<S> = (0..grid.size())
                        .map(|i| {
                            let z = grid.coordinate(i, k);
                            let mut value = alpha[g][i].clone();
                            for _ in 0..a {
                                value = value.mul(z);
                            }
                            value
                        })
                        .collect();
                    forward_moments.insert((k, a, b), forward_ops[g].apply(&e, &zk_power));
                }
            }
        }
        let polys: Vec<(GapPolynomial<S>, GapPolynomial<S>)> = (0..atoms)
            .map(|k| gap_score_polynomials(&transitions[g][k], like))
            .collect();
        let unit = |k: usize, b: u8| -> Vec<u8> {
            let mut e = vec![0u8; atoms];
            e[k] = b;
            e
        };
        for k in 0..atoms {
            let (t, dt) = &polys[k];
            let zk: Vec<S> = (0..grid.size()).map(|i| grid.coordinate(i, k).clone()).collect();
            let powers_start: Vec<Vec<S>> = (0..=4usize)
                .map(|a| {
                    zk.iter()
                        .map(|z| {
                            let mut v = z.constant_like(1.0);
                            for _ in 0..a {
                                v = v.mul(z);
                            }
                            v
                        })
                        .collect()
                })
                .collect();
            // Σ_{a,b} c[a][b] z^a E[u^b lik β | z] on grid g.
            let start_function = |poly: &GapPolynomial<S>, max_degree: usize| -> Vec<S> {
                let mut out = vec![like.constant_like(0.0); grid.size()];
                for a in 0..=max_degree {
                    for b in 0..=max_degree {
                        if poly.absent(a, b) {
                            continue;
                        }
                        let coefficient = poly.get(a, b);
                        let moment = &backward_moments[&unit(k, b as u8)];
                        for i in 0..grid.size() {
                            out[i] = out[i].add(
                                &coefficient
                                    .mul(&powers_start[a][i])
                                    .mul(&moment[i]),
                            );
                        }
                    }
                }
                out
            };
            let t_bar = start_function(t, 2);
            let dt_bar = start_function(dt, 2);
            let tt_bar = start_function(&t.mul(t), 4);
            mean_gap[g][k] = weighted_sum3(&grid.weights, smoothed, &t_bar);
            expected_dt[g][k] = weighted_sum3(&grid.weights, smoothed, &dt_bar);
            same_gap_products[g][k * atoms + k] =
                weighted_sum3(&grid.weights, smoothed, &tt_bar);
            right[g][f0 + k] = t_bar;
            // Ã on grid g+1: lik · Σ_{a,b} c[a][b] ∫ A(z'|z) u^b z^a α(z) dz.
            let mut a_tilde = vec![like.constant_like(0.0); next.size()];
            for a in 0..=2usize {
                for b in 0..=2usize {
                    if t.absent(a, b) {
                        continue;
                    }
                    let coefficient = t.get(a, b);
                    let moment = &forward_moments[&(k, a as u8, b as u8)];
                    for i in 0..next.size() {
                        a_tilde[i] = a_tilde[i].add(&coefficient.mul(&moment[i]));
                    }
                }
            }
            // As a ratio to the predicted density: `ã / p̂` is the
            // forward-smoothed polynomial expectation, bounded where the
            // predicted density has any mass.
            let floor = density_floor(&predicted[g + 1]);
            left[g + 1][f0 + k] = a_tilde
                .iter()
                .zip(predicted[g + 1].iter())
                .map(|(numerator, density)| {
                    if density.value() > floor {
                        div(numerator, density)
                    } else {
                        numerator.constant_like(0.0)
                    }
                })
                .collect();
        }
        // cross-atom same-gap products
        for k in 0..atoms {
            for j in (k + 1)..atoms {
                let (tk, _) = &polys[k];
                let (tj, _) = &polys[j];
                let mut out = vec![like.constant_like(0.0); grid.size()];
                for a in 0..=2usize {
                    for b in 0..=2usize {
                        if tk.absent(a, b) {
                            continue;
                        }
                        let ck = tk.get(a, b);
                        for a2 in 0..=2usize {
                            for b2 in 0..=2usize {
                                if tj.absent(a2, b2) {
                                    continue;
                                }
                                let cj = tj.get(a2, b2);
                                let mut e = vec![0u8; atoms];
                                e[k] = b as u8;
                                e[j] = b2 as u8;
                                let moment = &backward_moments[&e];
                                let coefficient = ck.mul(cj);
                                for i in 0..grid.size() {
                                    let mut zpow = coefficient.clone();
                                    let zk = grid.coordinate(i, k);
                                    let zj = grid.coordinate(i, j);
                                    for _ in 0..a {
                                        zpow = zpow.mul(zk);
                                    }
                                    for _ in 0..a2 {
                                        zpow = zpow.mul(zj);
                                    }
                                    out[i] = out[i].add(&zpow.mul(&moment[i]));
                                }
                            }
                        }
                    }
                }
                let value = weighted_sum3(&grid.weights, smoothed, &out);
                same_gap_products[g][k * atoms + j] = value.clone();
                same_gap_products[g][j * atoms + k] = value;
            }
        }
    }

    // ---- all-pairs table -------------------------------------------------
    let pair_index = |n: usize, m: usize| -> usize { m * (m + 1) / 2 + n };
    let mut table: Vec<Vec<S>> = vec![Vec::new(); n_nodes * (n_nodes + 1) / 2];
    // `left[n][a]` is a bounded function on grid n (`f` for a node function,
    // `ã / p̂` for the gap into n) and the carried functions are smoothed
    // conditional expectations `E[f_m | z_n, data]`, so the contraction is an
    // expectation under the smoothed marginal.
    let weighted_left: Vec<Vec<Vec<S>>> = (0..n_nodes)
        .map(|n| {
            left[n]
                .iter()
                .map(|f| pointwise(&grids[n].weights, &pointwise(f, &smoothed_all[n])))
                .collect()
        })
        .collect();
    let transfer_apply = |n: usize, values: &[S]| -> Vec<S> {
        let rows = grids[n].size();
        let cols = grids[n + 1].size();
        (0..rows)
            .map(|i| {
                (0..cols).fold(like.constant_like(0.0), |acc, j| {
                    acc.add(&transfers[n][i * cols + j].mul(&values[j]))
                })
            })
            .collect()
    };
    for m in 0..n_nodes {
        let mut carried: Vec<Vec<S>> = right[m].clone();
        let contract = |n: usize, functions: &[Vec<S>]| -> Vec<S> {
            let mut out = Vec::with_capacity(f_total * f_total);
            for a in 0..f_total {
                for b in 0..f_total {
                    out.push(
                        weighted_left[n][a]
                            .iter()
                            .zip(functions[b].iter())
                            .fold(like.constant_like(0.0), |acc, (w, v)| acc.add(&w.mul(v))),
                    );
                }
            }
            out
        };
        table[pair_index(m, m)] = contract(m, &carried);
        for n in (0..m).rev() {
            carried = carried.iter().map(|h| transfer_apply(n, h)).collect();
            table[pair_index(n, m)] = contract(n, &carried);
        }
    }

    // ---- assemble --------------------------------------------------------
    let eta_index = |n: usize, d: usize| n * marks + d;
    let a_index = |d: usize, k: usize| n_nodes * marks + d * atoms + k;
    let rho_index = |k: usize| n_nodes * marks + marks * atoms + k;
    let slot_a = |d: usize, k: usize| marks + d * atoms + k;

    let expectation = |n: usize, fa: usize, m: usize, fb: usize| -> S {
        if n <= m {
            table[pair_index(n, m)][fa * f_total + fb].clone()
        } else {
            table[pair_index(m, n)][fb * f_total + fa].clone()
        }
    };
    let cov_nodes = |n: usize, fa: usize, m: usize, fb: usize| -> S {
        expectation(n, fa, m, fb).sub(&mean_node[n][fa].mul(&mean_node[m][fb]))
    };
    let cov_node_gap = |n: usize, f: usize, g: usize, k: usize| -> S {
        let e = if n <= g {
            expectation(n, f, g, f0 + k)
        } else {
            expectation(g + 1, f0 + k, n, f)
        };
        e.sub(&mean_node[n][f].mul(&mean_gap[g][k]))
    };
    let cov_gap_gap = |g: usize, k: usize, g2: usize, j: usize| -> S {
        let e = if g < g2 {
            expectation(g + 1, f0 + k, g2, f0 + j)
        } else if g > g2 {
            expectation(g2 + 1, f0 + j, g, f0 + k)
        } else {
            same_gap_products[g][k * atoms + j].clone()
        };
        e.sub(&mean_gap[g][k].mul(&mean_gap[g2][j]))
    };

    let mut gradient = vec![like.constant_like(0.0); p_total];
    for n in 0..n_nodes {
        for d in 0..marks {
            gradient[eta_index(n, d)] = mean_node[n][d].clone();
            for k in 0..atoms {
                let idx = a_index(d, k);
                gradient[idx] = gradient[idx].add(&mean_node[n][slot_a(d, k)]);
            }
        }
    }
    for g in 0..n_gaps {
        for k in 0..atoms {
            let idx = rho_index(k);
            gradient[idx] = gradient[idx].add(&mean_gap[g][k]);
        }
    }

    let mut hessian = vec![like.constant_like(0.0); p_total * p_total];
    let mut set = |i: usize, j: usize, value: S| {
        hessian[i * p_total + j] = value.clone();
        hessian[j * p_total + i] = value;
    };
    // η-η
    for n in 0..n_nodes {
        for d in 0..marks {
            for m in n..n_nodes {
                for d2 in 0..marks {
                    if m == n && d2 < d {
                        continue;
                    }
                    let mut value = cov_nodes(n, d, m, d2);
                    if m == n && d2 == d {
                        value = value.sub(&expected_curvature[n][d]);
                    }
                    set(eta_index(n, d), eta_index(m, d2), value);
                }
            }
        }
    }
    // η-a
    for n in 0..n_nodes {
        for d in 0..marks {
            for d2 in 0..marks {
                for k in 0..atoms {
                    let mut value = like.constant_like(0.0);
                    for m in 0..n_nodes {
                        value = value.add(&cov_nodes(n, d, m, slot_a(d2, k)));
                    }
                    if d2 == d {
                        value = value.sub(&expected_curvature_z[n][d * atoms + k]);
                    }
                    set(eta_index(n, d), a_index(d2, k), value);
                }
            }
        }
    }
    // a-a
    for d in 0..marks {
        for k in 0..atoms {
            for d2 in 0..marks {
                for j in 0..atoms {
                    if a_index(d2, j) < a_index(d, k) {
                        continue;
                    }
                    let mut value = like.constant_like(0.0);
                    for n in 0..n_nodes {
                        for m in 0..n_nodes {
                            value = value.add(&cov_nodes(n, slot_a(d, k), m, slot_a(d2, j)));
                        }
                    }
                    if d2 == d {
                        for n in 0..n_nodes {
                            value = value
                                .sub(&expected_curvature_zz[n][(d * atoms + k) * atoms + j]);
                        }
                    }
                    set(a_index(d, k), a_index(d2, j), value);
                }
            }
        }
    }
    // η-ρ and a-ρ and ρ-ρ
    for k in 0..atoms {
        for n in 0..n_nodes {
            for d in 0..marks {
                let mut value = like.constant_like(0.0);
                for g in 0..n_gaps {
                    value = value.add(&cov_node_gap(n, d, g, k));
                }
                set(eta_index(n, d), rho_index(k), value);
            }
        }
        for d in 0..marks {
            for j in 0..atoms {
                let mut value = like.constant_like(0.0);
                for n in 0..n_nodes {
                    for g in 0..n_gaps {
                        value = value.add(&cov_node_gap(n, slot_a(d, j), g, k));
                    }
                }
                set(a_index(d, j), rho_index(k), value);
            }
        }
        for j in k..atoms {
            let mut value = like.constant_like(0.0);
            for g in 0..n_gaps {
                for g2 in 0..n_gaps {
                    value = value.add(&cov_gap_gap(g, k, g2, j));
                }
            }
            if j == k {
                for g in 0..n_gaps {
                    value = value.add(&expected_dt[g][k]);
                }
            }
            set(rho_index(k), rho_index(j), value);
        }
    }
    Ok(SubjectOutput {
        loglik,
        gradient,
        hessian,
    })
}

/// One completed forward filter: per-node grids, filtered densities, the
/// predicted (pre-update) densities, and the per-node log normalisers
/// `ln c_n + m_n`, whose running sum is the log predictive probability of the
/// observed counts.
pub(crate) struct ForwardPass<S> {
    pub grids: Vec<Grid<S>>,
    pub alpha: Vec<Vec<S>>,
    pub predicted: Vec<Vec<S>>,
    pub log_normalisers: Vec<S>,
}

/// Forward filter only, optionally continuing from a filtered state and
/// optionally restricting the compensator to a subset of marks (a forecast
/// conditions on the absorbing marks not having fired).
pub(crate) fn forward_filter<S: JetField>(
    inputs: &SubjectInputs<'_, S>,
    initial: Option<(&Grid<S>, &[S])>,
    compensated: &[bool],
) -> Result<ForwardPass<S>, EventHistoryError> {
    let nodes = inputs.nodes;
    let n_nodes = nodes.len();
    let marks = nodes.counts.ncols();
    let atoms = inputs.log_rates.len();
    let gh = inputs.gh;
    if n_nodes == 0 || marks == 0 || compensated.len() != marks {
        return Err(numerical("forward filter needs nodes, marks and a compensator mask"));
    }
    let like = &inputs.eta0[0];
    let mut grids: Vec<Grid<S>> = Vec::with_capacity(n_nodes);
    let mut alpha: Vec<Vec<S>> = Vec::with_capacity(n_nodes);
    let mut predicted: Vec<Vec<S>> = Vec::with_capacity(n_nodes);
    let mut log_normalisers: Vec<S> = Vec::with_capacity(n_nodes);
    let node_terms = |grid: &Grid<S>, n: usize| -> NodeLikelihood<S> {
        node_likelihood(
            grid,
            &inputs.eta0[n * marks..(n + 1) * marks],
            inputs.loadings,
            &nodes.counts.row(n).to_vec(),
            nodes.exposures[n],
            Some(compensated),
            marks,
            atoms,
        )
    };
    let transitions_across = |gap: f64| -> Vec<AtomTransition<S>> {
        (0..atoms)
            .map(|k| AtomTransition::new(&exp(&inputs.log_rates[k]).scale(gap / inputs.time_scale)))
            .collect()
    };
    // Node 0: either the stationary prior or a continuation of a filtered state.
    let first = match initial {
        None => filter_start(gh, like, atoms, &|grid| node_terms(grid, 0), "forecast first node")?,
        Some((grid, filtered)) => filter_step(
            gh,
            like,
            grid,
            filtered,
            transitions_across(inputs.continuation_gap),
            0,
            &|grid| node_terms(grid, 0),
            "forecast first node",
        )?,
    };
    log_normalisers.push(ln(&first.normaliser).add(&like.constant_like(first.likelihood.shift)));
    predicted.push(first.predicted);
    alpha.push(first.alpha);
    grids.push(first.grid);
    for n in 0..n_nodes - 1 {
        let step = filter_step(
            gh,
            like,
            &grids[n],
            &alpha[n],
            transitions_across(nodes.gaps[n]),
            0,
            &|grid| node_terms(grid, n + 1),
            &format!("forecast node {}", n + 1),
        )?;
        log_normalisers.push(ln(&step.normaliser).add(&like.constant_like(step.likelihood.shift)));
        predicted.push(step.predicted);
        alpha.push(step.alpha);
        grids.push(step.grid);
    }
    Ok(ForwardPass {
        grids,
        alpha,
        predicted,
        log_normalisers,
    })
}
