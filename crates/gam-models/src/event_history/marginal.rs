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
    AtomTransition, GaussHermite, Grid, SeparableOperator, backward_operator, forward_operator,
    normal_density,
};
use super::cohort::{EventHistoryError, SubjectNodes};
use super::scalar::{add_real, exp, ln, recip, sqrt, square};
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
}

impl<S: JetField> GapPolynomial<S> {
    fn zero(like: &S) -> Self {
        Self {
            c: vec![like.constant_like(0.0); 25],
        }
    }

    fn set(&mut self, a: usize, b: usize, value: S) {
        self.c[a * 5 + b] = value;
    }

    fn get(&self, a: usize, b: usize) -> &S {
        &self.c[a * 5 + b]
    }

    fn scaled(&self, factor: &S) -> Self {
        Self {
            c: self.c.iter().map(|v| v.mul(factor)).collect(),
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
        }
    }

    /// Product, exact when the total degree stays at most four per variable.
    fn mul(&self, other: &Self) -> Self {
        let zero = self.c[0].constant_like(0.0);
        let mut out = Self {
            c: vec![zero; 25],
        };
        for a in 0..5 {
            for b in 0..5 {
                let x = self.get(a, b);
                if x.value() == 0.0 {
                    continue;
                }
                for a2 in 0..5 - a {
                    for b2 in 0..5 - b {
                        let y = other.get(a2, b2);
                        if y.value() == 0.0 {
                            continue;
                        }
                        let idx = (a + a2) * 5 + (b + b2);
                        out.c[idx] = out.c[idx].add(&x.mul(y));
                    }
                }
            }
        }
        out
    }
}

/// The score `t = ∂ ln p(z'|z) / ∂ρ` of one atom across one gap and its own
/// derivative `∂t/∂ρ`, both as gap polynomials.
fn gap_score_polynomials<S: JetField>(
    transition: &AtomTransition<S>,
    like: &S,
) -> (GapPolynomial<S>, GapPolynomial<S>) {
    let phi = &transition.phi;
    let v = &transition.innovation;
    let inv_v = recip(v);
    let inv_v2 = square(&inv_v);
    let inv_v3 = inv_v2.mul(&inv_v);
    let phi2 = square(phi);
    // dL/dφ = φ/v + (1+φ²)/v² · z z' − φ/v² · (z² + z'²)
    let mut dl = GapPolynomial::zero(like);
    dl.set(0, 0, phi.mul(&inv_v));
    dl.set(1, 1, add_real(&phi2, 1.0).mul(&inv_v2));
    let minus_phi_v2 = phi.mul(&inv_v2).neg();
    dl.set(2, 0, minus_phi_v2.clone());
    dl.set(0, 2, minus_phi_v2);
    // d²L/dφ² = 1/v + 2φ²/v² + 2φ(3+φ²)/v³ · z z' − (1+3φ²)/v³ · (z² + z'²)
    let mut d2l = GapPolynomial::zero(like);
    d2l.set(0, 0, inv_v.add(&phi2.mul(&inv_v2).scale(2.0)));
    d2l.set(1, 1, phi.mul(&add_real(&phi2, 3.0)).mul(&inv_v3).scale(2.0));
    let minus = add_real(&phi2.scale(3.0), 1.0).mul(&inv_v3).neg();
    d2l.set(2, 0, minus.clone());
    d2l.set(0, 2, minus);
    let t = dl.scaled(&transition.dphi);
    let dt = d2l
        .scaled(&square(&transition.dphi))
        .add(&dl.scaled(&transition.d2phi));
    (t, dt)
}

/// The gap score polynomials as flat coefficient vectors, for tests.
pub(crate) fn gap_score_polynomials_for_test(
    transition: &AtomTransition<f64>,
) -> (Vec<f64>, Vec<f64>) {
    let (t, dt) = gap_score_polynomials(transition, &0.0);
    (t.c, dt.c)
}

/// Values of the monomial `Π_k z_k^{e_k}` on a grid.
fn monomial<S: JetField>(grid: &Grid<S>, exponents: &[u8]) -> Vec<S> {
    (0..grid.size())
        .map(|flat| {
            let mut value = grid.weights[0].constant_like(1.0);
            for (axis, &e) in exponents.iter().enumerate() {
                let z = grid.coordinate(flat, axis);
                for _ in 0..e {
                    value = value.mul(z);
                }
            }
            value
        })
        .collect()
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
    marks: usize,
    atoms: usize,
) -> NodeLikelihood<S> {
    let size = grid.size();
    let mut expeta = Vec::with_capacity(marks * size);
    let mut ell = vec![eta0[0].constant_like(0.0); size];
    for d in 0..marks {
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
    let mut forward_ops: Vec<SeparableOperator<S>> = Vec::with_capacity(n_nodes);
    let mut backward_ops: Vec<SeparableOperator<S>> = Vec::with_capacity(n_nodes);
    let mut loglik = like.constant_like(0.0);

    let zero_centres: Vec<S> = (0..atoms).map(|_| like.constant_like(0.0)).collect();
    let unit_scales: Vec<S> = (0..atoms).map(|_| like.constant_like(1.0)).collect();
    let first_grid = Grid::new(gh, &zero_centres, &unit_scales, like);
    let first_lik = node_likelihood(
        &first_grid,
        &inputs.eta0[0..marks],
        inputs.loadings,
        &counts_rows[0],
        nodes.exposures[0],
        marks,
        atoms,
    );
    let mut raw: Vec<S> = (0..first_grid.size())
        .map(|i| {
            let mut density = like.constant_like(1.0);
            for k in 0..atoms {
                let z = first_grid.coordinate(i, k);
                density = density.mul(&normal_density(
                    z,
                    &like.constant_like(0.0),
                    &like.constant_like(1.0),
                ));
            }
            density.mul(&exp(&add_real(&first_lik.ell[i], -first_lik.shift)))
        })
        .collect();
    let c0 = weighted_sum(&first_grid.weights, &raw);
    if !(c0.value() > 0.0) || !c0.value().is_finite() {
        return Err(numerical("first-node normaliser is not positive"));
    }
    let inv_c0 = recip(&c0);
    for v in raw.iter_mut() {
        *v = v.mul(&inv_c0);
    }
    loglik = loglik.add(&ln(&c0)).add(&like.constant_like(first_lik.shift));
    lik.push(
        first_lik
            .ell
            .iter()
            .map(|e| exp(&add_real(e, -first_lik.shift)).mul(&inv_c0))
            .collect(),
    );
    likelihoods.push(first_lik);
    alpha.push(raw);
    grids.push(first_grid);

    for n in 0..n_nodes - 1 {
        let grid = &grids[n];
        let a = &alpha[n];
        // posterior moments per axis
        let mut centres = Vec::with_capacity(atoms);
        let mut scales = Vec::with_capacity(atoms);
        let mut gap_transitions = Vec::with_capacity(atoms);
        for k in 0..atoms {
            let zs: Vec<S> = (0..grid.size()).map(|i| grid.coordinate(i, k).clone()).collect();
            let mean = weighted_sum3(&grid.weights, a, &zs);
            let centred: Vec<S> = zs.iter().map(|z| square(&z.sub(&mean))).collect();
            let mut variance = weighted_sum3(&grid.weights, a, &centred);
            let floor = VARIANCE_FLOOR_RELATIVE * square(&grid.axes[k].sigma).value();
            if !(variance.value() > floor) {
                variance = variance.constant_like(floor);
            }
            let kappa = exp(&inputs.log_rates[k]).scale(nodes.gaps[n] / inputs.time_scale);
            let transition = AtomTransition::new(&kappa);
            let predicted_mean = transition.phi.mul(&mean);
            let predicted_variance = square(&transition.phi)
                .mul(&variance)
                .add(&transition.innovation);
            centres.push(predicted_mean);
            scales.push(sqrt(&predicted_variance));
            gap_transitions.push(transition);
        }
        let next_grid = Grid::new(gh, &centres, &scales, like);
        let forward = forward_operator(gh, grid, &next_grid, &gap_transitions);
        let backward = backward_operator(gh, grid, &next_grid, &gap_transitions);
        let predicted = forward.apply(a);
        let next_lik = node_likelihood(
            &next_grid,
            &inputs.eta0[(n + 1) * marks..(n + 2) * marks],
            inputs.loadings,
            &counts_rows[n + 1],
            nodes.exposures[n + 1],
            marks,
            atoms,
        );
        let mut next_alpha: Vec<S> = predicted
            .iter()
            .zip(next_lik.ell.iter())
            .map(|(p, e)| p.mul(&exp(&add_real(e, -next_lik.shift))))
            .collect();
        let c = weighted_sum(&next_grid.weights, &next_alpha);
        if !(c.value() > 0.0) || !c.value().is_finite() {
            return Err(numerical(format!(
                "normaliser at node {} is not positive ({})",
                n + 1,
                c.value()
            )));
        }
        let inv_c = recip(&c);
        for v in next_alpha.iter_mut() {
            *v = v.mul(&inv_c);
        }
        loglik = loglik
            .add(&ln(&c))
            .add(&like.constant_like(next_lik.shift));
        lik.push(
            next_lik
                .ell
                .iter()
                .map(|e| exp(&add_real(e, -next_lik.shift)).mul(&inv_c))
                .collect(),
        );
        likelihoods.push(next_lik);
        alpha.push(next_alpha);
        grids.push(next_grid);
        transitions.push(gap_transitions);
        forward_ops.push(forward);
        backward_ops.push(backward);
    }
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

    // ---- backward pass ---------------------------------------------------
    let mut beta: Vec<Vec<S>> = vec![Vec::new(); n_nodes];
    beta[n_nodes - 1] = vec![like.constant_like(1.0); grids[n_nodes - 1].size()];
    for n in (0..n_nodes - 1).rev() {
        let carried = pointwise(&lik[n + 1], &beta[n + 1]);
        beta[n] = backward_ops[n].apply(&carried);
    }

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
        let smoothed: Vec<S> = pointwise(&alpha[n], &beta[n]);
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
            ec.push(weighted_sum3(&grid.weights, &smoothed, &curvature));
            for k in 0..atoms {
                let zk: Vec<S> = (0..size).map(|i| grid.coordinate(i, k).clone()).collect();
                let cz = pointwise(&curvature, &zk);
                ecz.push(weighted_sum3(&grid.weights, &smoothed, &cz));
                for j in 0..atoms {
                    let zj: Vec<S> = (0..size).map(|i| grid.coordinate(i, j).clone()).collect();
                    let czz = pointwise(&cz, &zj);
                    eczz.push(weighted_sum3(&grid.weights, &smoothed, &czz));
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
            means.push(weighted_sum3(&grid.weights, &smoothed, f));
            left_n.push(pointwise(&alpha[n], f));
            right_n.push(pointwise(&beta[n], f));
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
    let n_gaps = n_nodes.saturating_sub(1);
    let mut mean_gap: Vec<Vec<S>> = vec![vec![like.constant_like(0.0); atoms]; n_gaps];
    let mut expected_dt: Vec<Vec<S>> = vec![vec![like.constant_like(0.0); atoms]; n_gaps];
    // same-gap E[t_k t_j] including k == j
    let mut same_gap_products: Vec<Vec<S>> =
        vec![vec![like.constant_like(0.0); atoms * atoms]; n_gaps];
    for g in 0..n_gaps {
        let grid = &grids[g];
        let next = &grids[g + 1];
        let carried = pointwise(&lik[g + 1], &beta[g + 1]);
        let mut backward_moments: HashMap<Vec<u8>, Vec<S>> = HashMap::new();
        let mut forward_moments: HashMap<Vec<u8>, Vec<S>> = HashMap::new();
        let mut needed: Vec<Vec<u8>> = Vec::new();
        for k in 0..atoms {
            for b in 0..=4u8 {
                let mut e = vec![0u8; atoms];
                e[k] = b;
                needed.push(e);
            }
            for j in (k + 1)..atoms {
                for b in 1..=2u8 {
                    for b2 in 1..=2u8 {
                        let mut e = vec![0u8; atoms];
                        e[k] = b;
                        e[j] = b2;
                        needed.push(e);
                    }
                }
            }
        }
        for e in needed {
            if backward_moments.contains_key(&e) {
                continue;
            }
            let mono = monomial(next, &e);
            let value = backward_ops[g].apply(&pointwise(&mono, &carried));
            backward_moments.insert(e, value);
        }
        for k in 0..atoms {
            for a in 0..=2u8 {
                let mut e = vec![0u8; atoms];
                e[k] = a;
                if forward_moments.contains_key(&e) {
                    continue;
                }
                let mono = monomial(grid, &e);
                let value = forward_ops[g].apply(&pointwise(&mono, &alpha[g]));
                forward_moments.insert(e, value);
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
            let zk_next: Vec<S> = (0..next.size())
                .map(|i| next.coordinate(i, k).clone())
                .collect();
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
            let powers_end: Vec<Vec<S>> = (0..=2usize)
                .map(|b| {
                    zk_next
                        .iter()
                        .map(|z| {
                            let mut v = z.constant_like(1.0);
                            for _ in 0..b {
                                v = v.mul(z);
                            }
                            v
                        })
                        .collect()
                })
                .collect();
            // Evaluate Σ_{a,b} c[a][b] z^a Mb[b] on grid g for a polynomial.
            let start_function = |poly: &GapPolynomial<S>, max_degree: usize| -> Vec<S> {
                let mut out = vec![like.constant_like(0.0); grid.size()];
                for a in 0..=max_degree {
                    for b in 0..=max_degree {
                        let coefficient = poly.get(a, b);
                        if coefficient.value() == 0.0 {
                            continue;
                        }
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
            mean_gap[g][k] = weighted_sum3(&grid.weights, &alpha[g], &t_bar);
            expected_dt[g][k] = weighted_sum3(&grid.weights, &alpha[g], &dt_bar);
            same_gap_products[g][k * atoms + k] =
                weighted_sum3(&grid.weights, &alpha[g], &tt_bar);
            right[g][f0 + k] = t_bar;
            // Ã on grid g+1: lik · Σ_{a,b} c[a][b] z'^b Mf[a]
            let mut a_tilde = vec![like.constant_like(0.0); next.size()];
            for a in 0..=2usize {
                let moment = &forward_moments[&unit(k, a as u8)];
                for b in 0..=2usize {
                    let coefficient = t.get(a, b);
                    if coefficient.value() == 0.0 {
                        continue;
                    }
                    for i in 0..next.size() {
                        a_tilde[i] = a_tilde[i].add(
                            &coefficient
                                .mul(&powers_end[b][i])
                                .mul(&moment[i]),
                        );
                    }
                }
            }
            left[g + 1][f0 + k] = pointwise(&a_tilde, &lik[g + 1]);
        }
        // cross-atom same-gap products
        for k in 0..atoms {
            for j in (k + 1)..atoms {
                let (tk, _) = &polys[k];
                let (tj, _) = &polys[j];
                let mut out = vec![like.constant_like(0.0); grid.size()];
                for a in 0..=2usize {
                    for b in 0..=2usize {
                        let ck = tk.get(a, b);
                        if ck.value() == 0.0 {
                            continue;
                        }
                        for a2 in 0..=2usize {
                            for b2 in 0..=2usize {
                                let cj = tj.get(a2, b2);
                                if cj.value() == 0.0 {
                                    continue;
                                }
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
                let value = weighted_sum3(&grid.weights, &alpha[g], &out);
                same_gap_products[g][k * atoms + j] = value.clone();
                same_gap_products[g][j * atoms + k] = value;
            }
        }
    }

    // ---- all-pairs table -------------------------------------------------
    let pair_index = |n: usize, m: usize| -> usize { m * (m + 1) / 2 + n };
    let mut table: Vec<Vec<S>> = vec![Vec::new(); n_nodes * (n_nodes + 1) / 2];
    for m in 0..n_nodes {
        let mut carried: Vec<Vec<S>> = right[m].clone();
        let contract = |n: usize, functions: &[Vec<S>]| -> Vec<S> {
            let grid = &grids[n];
            let mut out = Vec::with_capacity(f_total * f_total);
            for a in 0..f_total {
                for b in 0..f_total {
                    out.push(weighted_sum3(&grid.weights, &left[n][a], &functions[b]));
                }
            }
            out
        };
        table[pair_index(m, m)] = contract(m, &carried);
        for n in (0..m).rev() {
            carried = carried
                .iter()
                .map(|h| backward_ops[n].apply(&pointwise(&lik[n + 1], h)))
                .collect();
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
    let masked_exposure = |n: usize, d: usize| -> f64 {
        if compensated[d] { nodes.exposures[n] } else { 0.0 }
    };
    let node_terms = |grid: &Grid<S>, n: usize| -> (Vec<S>, f64) {
        let size = grid.size();
        let mut ell = vec![like.constant_like(0.0); size];
        for d in 0..marks {
            let y = nodes.counts[[n, d]];
            let w = masked_exposure(n, d);
            if y == 0.0 && w == 0.0 {
                continue;
            }
            for i in 0..size {
                let mut eta = inputs.eta0[n * marks + d].clone();
                for k in 0..atoms {
                    eta = eta.add(&inputs.loadings[d * atoms + k].mul(grid.coordinate(i, k)));
                }
                if y != 0.0 {
                    ell[i] = ell[i].add(&eta.scale(y));
                }
                if w != 0.0 {
                    ell[i] = ell[i].sub(&exp(&eta).scale(w));
                }
            }
        }
        let shift = ell.iter().map(|v| v.value()).fold(f64::NEG_INFINITY, f64::max);
        (ell, shift)
    };
    // Node 0: either the stationary prior or a continuation of a filtered state.
    let (grid0, prior0): (Grid<S>, Vec<S>) = match initial {
        None => {
            let centres: Vec<S> = (0..atoms).map(|_| like.constant_like(0.0)).collect();
            let scales: Vec<S> = (0..atoms).map(|_| like.constant_like(1.0)).collect();
            let grid = Grid::new(gh, &centres, &scales, like);
            let density: Vec<S> = (0..grid.size())
                .map(|i| {
                    let mut density = like.constant_like(1.0);
                    for k in 0..atoms {
                        density = density.mul(&normal_density(
                            grid.coordinate(i, k),
                            &like.constant_like(0.0),
                            &like.constant_like(1.0),
                        ));
                    }
                    density
                })
                .collect();
            (grid, density)
        }
        Some((grid, filtered)) => {
            // Propagate the supplied state across the continuation gap to the first node.
            let mut centres = Vec::with_capacity(atoms);
            let mut scales = Vec::with_capacity(atoms);
            let mut transitions = Vec::with_capacity(atoms);
            for k in 0..atoms {
                let zs: Vec<S> = (0..grid.size()).map(|i| grid.coordinate(i, k).clone()).collect();
                let mean = weighted_sum3(&grid.weights, filtered, &zs);
                let centred: Vec<S> = zs.iter().map(|z| square(&z.sub(&mean))).collect();
                let mut variance = weighted_sum3(&grid.weights, filtered, &centred);
                let floor = VARIANCE_FLOOR_RELATIVE * square(&grid.axes[k].sigma).value();
                if !(variance.value() > floor) {
                    variance = variance.constant_like(floor);
                }
                let kappa = exp(&inputs.log_rates[k])
                    .scale(inputs.continuation_gap / inputs.time_scale);
                let transition = AtomTransition::new(&kappa);
                centres.push(transition.phi.mul(&mean));
                scales.push(sqrt(
                    &square(&transition.phi).mul(&variance).add(&transition.innovation),
                ));
                transitions.push(transition);
            }
            let next = Grid::new(gh, &centres, &scales, like);
            let forward = forward_operator(gh, grid, &next, &transitions);
            let density = forward.apply(filtered);
            (next, density)
        }
    };
    let (ell0, shift0) = node_terms(&grid0, 0);
    let mut raw: Vec<S> = prior0
        .iter()
        .zip(ell0.iter())
        .map(|(p, e)| p.mul(&exp(&add_real(e, -shift0))))
        .collect();
    let c0 = weighted_sum(&grid0.weights, &raw);
    if !(c0.value() > 0.0) || !c0.value().is_finite() {
        return Err(numerical("forecast normaliser at the first node is not positive"));
    }
    let inv = recip(&c0);
    for v in raw.iter_mut() {
        *v = v.mul(&inv);
    }
    log_normalisers.push(ln(&c0).add(&like.constant_like(shift0)));
    predicted.push(prior0);
    alpha.push(raw);
    grids.push(grid0);
    for n in 0..n_nodes - 1 {
        let grid = &grids[n];
        let a = &alpha[n];
        let mut centres = Vec::with_capacity(atoms);
        let mut scales = Vec::with_capacity(atoms);
        let mut transitions = Vec::with_capacity(atoms);
        for k in 0..atoms {
            let zs: Vec<S> = (0..grid.size()).map(|i| grid.coordinate(i, k).clone()).collect();
            let mean = weighted_sum3(&grid.weights, a, &zs);
            let centred: Vec<S> = zs.iter().map(|z| square(&z.sub(&mean))).collect();
            let mut variance = weighted_sum3(&grid.weights, a, &centred);
            let floor = VARIANCE_FLOOR_RELATIVE * square(&grid.axes[k].sigma).value();
            if !(variance.value() > floor) {
                variance = variance.constant_like(floor);
            }
            let kappa = exp(&inputs.log_rates[k]).scale(nodes.gaps[n] / inputs.time_scale);
            let transition = AtomTransition::new(&kappa);
            centres.push(transition.phi.mul(&mean));
            scales.push(sqrt(
                &square(&transition.phi).mul(&variance).add(&transition.innovation),
            ));
            transitions.push(transition);
        }
        let next = Grid::new(gh, &centres, &scales, like);
        let forward = forward_operator(gh, grid, &next, &transitions);
        let density = forward.apply(a);
        let (ell, shift) = node_terms(&next, n + 1);
        let mut raw: Vec<S> = density
            .iter()
            .zip(ell.iter())
            .map(|(p, e)| p.mul(&exp(&add_real(e, -shift))))
            .collect();
        let c = weighted_sum(&next.weights, &raw);
        if !(c.value() > 0.0) || !c.value().is_finite() {
            return Err(numerical(format!(
                "forecast normaliser at node {} is not positive",
                n + 1
            )));
        }
        let inv = recip(&c);
        for v in raw.iter_mut() {
            *v = v.mul(&inv);
        }
        log_normalisers.push(ln(&c).add(&like.constant_like(shift)));
        predicted.push(density);
        alpha.push(raw);
        grids.push(next);
    }
    Ok(ForwardPass {
        grids,
        alpha,
        predicted,
        log_normalisers,
    })
}
