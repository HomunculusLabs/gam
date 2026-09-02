//! Laplace evidence of one subject's event history over its latent path,
//! with the exact gradient of that evidence in every parameter the subject
//! sees: its node log-intensities `η⁰`, the loadings `a`, and the log-rates
//! `ρ`.
//!
//! The log-intensity of mark `d` at a node with latent state `z ∈ ℝ^K` is
//!
//! ```text
//! η_d(z) = η⁰_d − ½ Σ_k a_{dk}² + Σ_k a_{dk} z_k
//! ```
//!
//! The atoms are stationary and standard at every time, so
//! `E_z exp(Σ_k a_{dk} z_k) = exp(½ Σ_k a_{dk}²)` and the shift cancels it:
//! `exp(η⁰_d)` is the population-average intensity whatever the loadings,
//! and the latent term is the individual deviation from it.
//!
//! The complete-data log-density of the path `z = (z_0, …, z_{N−1})` is
//!
//! ```text
//! F(z; θ) = Σ_n Σ_d [y_{nd} η_{nd}(z_n) − w_{nd} e^{η_{nd}(z_n)}]
//!         − ½ Σ_k [ z_{0k}² + Σ_n (z_{n+1,k} − φ_{nk} z_{nk})² / q_{nk} + ln q_{nk} ]
//! ```
//!
//! with `φ = e^{−κ}`, `q = 1 − φ²`, `κ = rate · gap`. Every node term is
//! concave in `z` (a Poisson log-link is concave in its linear predictor)
//! and the path prior is a Gaussian with positive-definite block-tridiagonal
//! precision, so `F` is strictly concave: the mode `ẑ` is unique and Newton
//! with a line search reaches it from anywhere. The evidence is
//!
//! ```text
//! ℓ(θ) = F(ẑ; θ) − ½ log|H(ẑ; θ)|,    H = −∇²_z F = Q + Λ(ẑ),
//! ```
//!
//! the `(2π)^{NK/2}` of the Laplace integral cancelling the prior's
//! normaliser exactly. `Q` is the path precision (tridiagonal per atom) and
//! `Λ` is block diagonal with `Λ_n = Σ_d w_{nd} μ_{nd} a_d a_dᵀ`, so `H` is
//! block tridiagonal with `K × K` blocks and everything below is `O(N K³)`:
//! polynomial in the number of atoms, never exponential.
//!
//! The gradient is the exact derivative of the Laplace objective through the
//! implicit dependence of the mode on `θ`:
//!
//! ```text
//! dℓ/dθ = ∂_θ F − ½ tr(H⁻¹ ∂_θ H) − vᵀ ∂_θ ∇_z F,    v = H⁻¹ c,
//! c_{nk} = ½ tr(H⁻¹ ∂_{z_{nk}} H) = ½ Σ_d w_{nd} μ_{nd} a_{dk} · a_dᵀ Σ_{nn} a_d,
//! ```
//!
//! which needs only the block-tridiagonal part of `Σ = H⁻¹` (the smoother
//! marginals and their one-lag cross-covariances) and one extra solve. The
//! whole computation is generic over a [`JetField`] scalar: on a
//! [`super::scalar::Tangent`] seeded with coefficient directions, the
//! tangent channels of this gradient are exact Hessian columns; on a tangent
//! over a one- or two-direction dual base they are the directional
//! derivatives of the Hessian the outer LAML solve needs. Before the
//! evaluation the mode is lifted into the scalar and refined by two Newton
//! steps: Newton's iteration on a jet is exact to first order after one step
//! and to third order after two, which covers every channel used here.
//!
//! Across a short gap the transition precision `1/q` is enormous and the
//! entries of the path precision cancel to `O(1)`, so nothing below is ever
//! formed as a difference of `1/q`-sized terms. The factorisation carries
//! each node's own precision `R_n` (the pivot without the forward coupling)
//! through the identity `1/q − (φ/q)² D⁻¹ = Φ⁻¹ (R⁻¹ + q/φ²)⁻¹ Φ⁻¹`, and
//! the rate gradient is written in innovation coordinates: the posterior
//! variance `V` of the increment `z' − φz`, its covariance `W` with the
//! state, and the mode residual `r = ẑ' − φẑ`, each `O(q)` or smaller by
//! construction.
//!
//! The sequential Gaussian filter at the end of the file serves forecasts
//! and the predictive PIT: predict the state across a gap, then update it at
//! the node by the Laplace approximation of one node's posterior, so the
//! normalisers multiply to the predictive probability of the observed counts.

use super::cohort::{EventHistoryError, SubjectNodes};
use super::scalar::{add_real, exp, ln, recip, sqrt, square};
use gam_math::nested_dual::JetField;

/// Everything one subject's evidence needs, in the caller's scalar type.
pub(crate) struct SubjectInputs<'a, S> {
    pub nodes: &'a SubjectNodes,
    /// Node log-intensities without the latent part, index `n * marks + d`.
    pub eta0: &'a [S],
    /// Loadings, index `d * atoms + k`.
    pub loadings: &'a [S],
    /// `ln(rate · time_scale)` per atom.
    pub log_rates: &'a [S],
    pub time_scale: f64,
}

impl<S: JetField> SubjectInputs<'_, S> {
    pub fn atoms(&self) -> usize {
        self.log_rates.len()
    }

    pub fn marks(&self) -> usize {
        self.nodes.counts.ncols()
    }

    fn validate(&self) -> Result<(), EventHistoryError> {
        let n = self.nodes.len();
        let marks = self.marks();
        if n == 0 || marks == 0 {
            return Err(numerical("subject evidence needs at least one node and one mark"));
        }
        if self.eta0.len() != n * marks || self.loadings.len() != marks * self.atoms() {
            return Err(numerical("subject evidence received mismatched parameter slices"));
        }
        if !(self.time_scale.is_finite() && self.time_scale > 0.0) {
            return Err(numerical("subject evidence needs a positive finite time scale"));
        }
        Ok(())
    }

    /// `−½ Σ_k a_{dk}²` per mark: the shift that makes `exp(η⁰_d)` the
    /// population-average intensity.
    fn shifts(&self) -> Vec<S> {
        let atoms = self.atoms();
        let like = &self.eta0[0];
        (0..self.marks())
            .map(|d| {
                self.loadings[d * atoms..(d + 1) * atoms]
                    .iter()
                    .fold(like.constant_like(0.0), |acc, a| acc.sub(&square(a).scale(0.5)))
            })
            .collect()
    }
}

/// The evidence and its gradient in the subject-local parameter vector
/// `[η⁰ (nodes × marks) | a (marks × atoms) | ρ (atoms)]`.
pub(crate) struct SubjectEvidence<S> {
    pub loglik: S,
    /// Empty when derivatives were not requested.
    pub gradient: Vec<S>,
}

fn numerical(reason: impl Into<String>) -> EventHistoryError {
    EventHistoryError::NumericalFailure {
        reason: reason.into(),
    }
}

// ---- small dense matrices (row-major k × k) --------------------------------

fn dot<S: JetField>(a: &[S], b: &[S]) -> S {
    a.iter()
        .zip(b.iter())
        .fold(a[0].constant_like(0.0), |acc, (x, y)| acc.add(&x.mul(y)))
}

fn matvec<S: JetField>(a: &[S], x: &[S], k: usize) -> Vec<S> {
    (0..k).map(|r| dot(&a[r * k..(r + 1) * k], x)).collect()
}

/// Lower Cholesky factor `L` with `a = L Lᵀ`.
fn cholesky<S: JetField>(a: &[S], k: usize, label: &str) -> Result<Vec<S>, EventHistoryError> {
    let zero = a[0].constant_like(0.0);
    let mut l = vec![zero; k * k];
    for j in 0..k {
        let mut d = a[j * k + j].clone();
        for m in 0..j {
            d = d.sub(&square(&l[j * k + m]));
        }
        if !(d.value() > 0.0) || !d.value().is_finite() {
            return Err(numerical(format!(
                "{label}: Cholesky pivot {j} is {} (not positive)",
                d.value()
            )));
        }
        let root = sqrt(&d);
        let inverse_root = recip(&root);
        l[j * k + j] = root;
        for i in (j + 1)..k {
            let mut s = a[i * k + j].clone();
            for m in 0..j {
                s = s.sub(&l[i * k + m].mul(&l[j * k + m]));
            }
            l[i * k + j] = s.mul(&inverse_root);
        }
    }
    Ok(l)
}

/// Solve `L Lᵀ x = b` in place.
fn cholesky_solve<S: JetField>(l: &[S], k: usize, x: &mut [S]) {
    for i in 0..k {
        let mut s = x[i].clone();
        for m in 0..i {
            s = s.sub(&l[i * k + m].mul(&x[m]));
        }
        x[i] = s.mul(&recip(&l[i * k + i]));
    }
    for i in (0..k).rev() {
        let mut s = x[i].clone();
        for m in (i + 1)..k {
            s = s.sub(&l[m * k + i].mul(&x[m]));
        }
        x[i] = s.mul(&recip(&l[i * k + i]));
    }
}

fn cholesky_logdet<S: JetField>(l: &[S], k: usize) -> S {
    (0..k).fold(l[0].constant_like(0.0), |acc, i| acc.add(&ln(&l[i * k + i]).scale(2.0)))
}

fn cholesky_inverse<S: JetField>(l: &[S], k: usize) -> Vec<S> {
    let zero = l[0].constant_like(0.0);
    let one = l[0].constant_like(1.0);
    let mut inverse = vec![zero.clone(); k * k];
    for j in 0..k {
        let mut column: Vec<S> = (0..k).map(|i| if i == j { one.clone() } else { zero.clone() }).collect();
        cholesky_solve(l, k, &mut column);
        for i in 0..k {
            inverse[i * k + j] = column[i].clone();
        }
    }
    inverse
}

// ---- transitions -------------------------------------------------------------

/// Beyond this `κ`, `e^{-κ}` underflows to exactly zero in `f64`, so `φ`
/// is exactly zero and every quantity derived from the transition is exactly
/// constant; holding `κ` there keeps `κ φ` from becoming `∞ · 0`.
const KAPPA_SATURATION: f64 = 746.0;

/// Transition of one unit-variance Ornstein–Uhlenbeck atom across a gap of
/// dimensionless length `κ = rate · gap`: `z' | z ~ N(φ z, q)`.
pub(crate) struct Transition<S> {
    pub phi: S,
    pub q: S,
    pub inv_q: S,
    pub kappa: S,
}

pub(crate) fn transition<S: JetField>(log_rate: &S, gap: f64, time_scale: f64) -> Transition<S> {
    let scaled = exp(log_rate).scale(gap / time_scale);
    let kappa = if scaled.value() >= KAPPA_SATURATION {
        scaled.constant_like(KAPPA_SATURATION)
    } else {
        scaled
    };
    let k = kappa.value();
    let e = (-k).exp();
    let e2 = (-2.0 * k).exp();
    let phi = kappa.compose_unary([e, -e, e, -e, e]);
    let q = kappa.compose_unary([-(-2.0 * k).exp_m1(), 2.0 * e2, -4.0 * e2, 8.0 * e2, -16.0 * e2]);
    let inv_q = recip(&q);
    Transition {
        phi,
        q,
        inv_q,
        kappa,
    }
}

fn transitions<S: JetField>(inputs: &SubjectInputs<'_, S>) -> Vec<Vec<Transition<S>>> {
    inputs
        .nodes
        .gaps
        .iter()
        .map(|&gap| {
            inputs
                .log_rates
                .iter()
                .map(|rho| transition(rho, gap, inputs.time_scale))
                .collect()
        })
        .collect()
}

// ---- the complete-data objective at a state ----------------------------------

/// `F(z)`, its gradient, and the blocks of `H = −∇²F` at a path `z`.
struct Assembly<S> {
    objective: S,
    /// Per node, `K` entries.
    gradient: Vec<Vec<S>>,
    /// Per node, `K × K`: the node's likelihood precision `Λ_n`, plus the
    /// stationary prior `I` at the first node. The gap terms of the prior
    /// (`diag(1/q_{n−1})` backward, `diag(φ_n²/q_n)` forward) are added by
    /// the factorisation in their stable form.
    own: Vec<Vec<S>>,
    /// Per gap, `K` entries: `H_{n,n+1} = diag(−φ/q)`.
    off: Vec<Vec<S>>,
    /// Intensities `μ_{nd}` at `z`, index `n * marks + d`.
    mu: Vec<S>,
}

fn assemble<S: JetField>(
    inputs: &SubjectInputs<'_, S>,
    shifts: &[S],
    transitions: &[Vec<Transition<S>>],
    z: &[Vec<S>],
) -> Assembly<S> {
    let nodes = inputs.nodes;
    let n = nodes.len();
    let marks = inputs.marks();
    let atoms = inputs.atoms();
    let like = &inputs.eta0[0];
    let zero = like.constant_like(0.0);
    let mut objective = zero.clone();
    let mut gradient: Vec<Vec<S>> = vec![vec![zero.clone(); atoms]; n];
    let mut own: Vec<Vec<S>> = vec![vec![zero.clone(); atoms * atoms]; n];
    let mut off: Vec<Vec<S>> = Vec::with_capacity(n.saturating_sub(1));
    let mut mu = Vec::with_capacity(n * marks);
    for node in 0..n {
        for d in 0..marks {
            let a = &inputs.loadings[d * atoms..(d + 1) * atoms];
            let mut eta = inputs.eta0[node * marks + d].add(&shifts[d]);
            for k in 0..atoms {
                eta = eta.add(&a[k].mul(&z[node][k]));
            }
            let m = exp(&eta);
            let y = nodes.counts[[node, d]];
            let w = nodes.exposures[[node, d]];
            if y != 0.0 {
                objective = objective.add(&eta.scale(y));
            }
            if w != 0.0 {
                objective = objective.sub(&m.scale(w));
            }
            if y != 0.0 || w != 0.0 {
                let score = add_real(&m.scale(-w), y);
                let curvature = m.scale(w);
                for k in 0..atoms {
                    gradient[node][k] = gradient[node][k].add(&score.mul(&a[k]));
                    let ca = curvature.mul(&a[k]);
                    for j in 0..atoms {
                        own[node][k * atoms + j] = own[node][k * atoms + j].add(&ca.mul(&a[j]));
                    }
                }
            }
            mu.push(m);
        }
    }
    // Stationary start: z_0 ~ N(0, I).
    for k in 0..atoms {
        objective = objective.sub(&square(&z[0][k]).scale(0.5));
        gradient[0][k] = gradient[0][k].sub(&z[0][k]);
        own[0][k * atoms + k] = add_real(&own[0][k * atoms + k], 1.0);
    }
    for gap in 0..n.saturating_sub(1) {
        let mut off_gap = Vec::with_capacity(atoms);
        for k in 0..atoms {
            let t = &transitions[gap][k];
            let residual = z[gap + 1][k].sub(&t.phi.mul(&z[gap][k]));
            let scaled = residual.mul(&t.inv_q);
            objective = objective
                .sub(&residual.mul(&scaled).scale(0.5))
                .sub(&ln(&t.q).scale(0.5));
            gradient[gap][k] = gradient[gap][k].add(&t.phi.mul(&scaled));
            gradient[gap + 1][k] = gradient[gap + 1][k].sub(&scaled);
            off_gap.push(t.phi.mul(&t.inv_q).neg());
        }
        off.push(off_gap);
    }
    Assembly {
        objective,
        gradient,
        own,
        off,
        mu,
    }
}

// ---- block-tridiagonal factorisation ------------------------------------------

/// Block `LDLᵀ` of the symmetric positive-definite block-tridiagonal `H`
/// whose off-diagonal blocks are diagonal (one atom couples only to itself
/// across a gap): pivots `D_n = R_n + diag(φ_n²/q_n)`, multipliers
/// `L_n = H_{n+1,n} D_n⁻¹`.
///
/// `R_n`, the node's own precision, is carried without cancellation:
/// `R_0 = I + Λ_0` and `R_n = Λ_n + M_{n−1}` with
/// `M = diag(1/q) − diag(φ/q) D⁻¹ diag(φ/q) = Φ⁻¹ (R⁻¹ + diag(q/φ²))⁻¹ Φ⁻¹`,
/// the information-filter form of the Schur update, so a short gap's
/// `1/q`-sized terms never meet as a difference.
struct BlockFactor<S> {
    k: usize,
    pivot_chol: Vec<Vec<S>>,
    pivot_inv: Vec<Vec<S>>,
    /// Per node, `k × k`: `R_n`.
    own: Vec<Vec<S>>,
    /// Per gap, `k × k`.
    lower: Vec<Vec<S>>,
    /// Per gap, `k` entries.
    off: Vec<Vec<S>>,
}

impl<S: JetField> BlockFactor<S> {
    fn new(
        own: &[Vec<S>],
        off: &[Vec<S>],
        transitions: &[Vec<Transition<S>>],
        k: usize,
    ) -> Result<Self, EventHistoryError> {
        let n = own.len();
        let mut pivot_chol = Vec::with_capacity(n);
        let mut pivot_inv = Vec::with_capacity(n);
        let mut owns = Vec::with_capacity(n);
        let mut lower = Vec::with_capacity(n.saturating_sub(1));
        let mut carried: Option<Vec<S>> = None;
        for node in 0..n {
            // R_n = Λ_n + M_{n−1}: the node's own precision.
            let mut r = own[node].clone();
            if let Some(m) = carried.take() {
                for (value, add) in r.iter_mut().zip(m.iter()) {
                    *value = value.add(add);
                }
            }
            let mut pivot = r.clone();
            if node + 1 < n {
                for atom in 0..k {
                    let t = &transitions[node][atom];
                    pivot[atom * k + atom] = pivot[atom * k + atom].add(&square(&t.phi).mul(&t.inv_q));
                }
            }
            let chol = cholesky(&pivot, k, "latent-path Hessian")?;
            let inverse = cholesky_inverse(&chol, k);
            if node + 1 < n {
                let mut multiplier = inverse.clone();
                for row in 0..k {
                    for c in 0..k {
                        multiplier[row * k + c] = multiplier[row * k + c].mul(&off[node][row]);
                    }
                }
                lower.push(multiplier);
                // M_n = diag(1/q) − diag(φ/q) D_n⁻¹ diag(φ/q), the backward
                // prior term the next node sees after the Schur update. Across
                // a short gap both terms are 1/q-sized and cancel to O(1), so
                // it is formed as Φ⁻¹ (R_n⁻¹ + diag(q/φ²))⁻¹ Φ⁻¹ whenever every
                // atom's φ is above one half; a gap long enough for some φ to
                // fall below that has q of order one and the direct form is
                // exact as written (and finite at φ = 0).
                let short_gap = (0..k).all(|atom| transitions[node][atom].phi.value() > 0.5);
                let m = if short_gap {
                    let r_chol = cholesky(&r, k, "latent-path own precision")?;
                    let mut shifted = cholesky_inverse(&r_chol, k);
                    for atom in 0..k {
                        let t = &transitions[node][atom];
                        shifted[atom * k + atom] =
                            shifted[atom * k + atom].add(&t.q.mul(&recip(&square(&t.phi))));
                    }
                    let shifted_chol = cholesky(&shifted, k, "latent-path information update")?;
                    let mut m = cholesky_inverse(&shifted_chol, k);
                    for row in 0..k {
                        for c in 0..k {
                            let scale = recip(&transitions[node][row].phi.mul(&transitions[node][c].phi));
                            m[row * k + c] = m[row * k + c].mul(&scale);
                        }
                    }
                    m
                } else {
                    let mut m = vec![r[0].constant_like(0.0); k * k];
                    for row in 0..k {
                        for c in 0..k {
                            m[row * k + c] = inverse[row * k + c]
                                .mul(&off[node][row])
                                .mul(&off[node][c])
                                .neg();
                        }
                        m[row * k + row] = m[row * k + row].add(&transitions[node][row].inv_q);
                    }
                    m
                };
                carried = Some(m);
            }
            pivot_chol.push(chol);
            pivot_inv.push(inverse);
            owns.push(r);
        }
        Ok(Self {
            k,
            pivot_chol,
            pivot_inv,
            own: owns,
            lower,
            off: off.to_vec(),
        })
    }

    fn logdet(&self) -> S {
        let like = &self.pivot_chol[0][0];
        self.pivot_chol
            .iter()
            .fold(like.constant_like(0.0), |acc, chol| acc.add(&cholesky_logdet(chol, self.k)))
    }

    /// `H⁻¹ b` for a per-node right-hand side.
    fn solve(&self, b: &[Vec<S>]) -> Vec<Vec<S>> {
        let k = self.k;
        let n = b.len();
        let mut y: Vec<Vec<S>> = Vec::with_capacity(n);
        y.push(b[0].clone());
        for node in 0..n - 1 {
            let carried = matvec(&self.lower[node], &y[node], k);
            let next: Vec<S> = b[node + 1]
                .iter()
                .zip(carried.iter())
                .map(|(x, c)| x.sub(c))
                .collect();
            y.push(next);
        }
        for (node, value) in y.iter_mut().enumerate() {
            cholesky_solve(&self.pivot_chol[node], k, value);
        }
        let mut x = y;
        for node in (0..n - 1).rev() {
            let mut t: Vec<S> = (0..k).map(|r| self.off[node][r].mul(&x[node + 1][r])).collect();
            cholesky_solve(&self.pivot_chol[node], k, &mut t);
            for r in 0..k {
                x[node][r] = x[node][r].sub(&t[r]);
            }
        }
        x
    }

    /// The diagonal blocks `Σ_{nn}` of `Σ = H⁻¹`, and per gap the posterior
    /// covariance `V_n` of the increment `z_{n+1} − Φ z_n` and its covariance
    /// `W_n` with `z_n`, all as sums of positive-semidefinite pieces:
    ///
    /// ```text
    /// Σ_nn = D_n⁻¹ + A Σ_{n+1,n+1} Aᵀ,           A = D_n⁻¹ Φ diag(1/q),
    /// V_n  = B Σ_{n+1,n+1} Bᵀ + Φ D_n⁻¹ Φ,        B = Φ D_n⁻¹ R_n Φ⁻¹ = I − Φ A,
    /// W_n  = B Σ_{n+1,n+1} Aᵀ − Φ D_n⁻¹.
    /// ```
    fn inverse_blocks(&self, transitions: &[Vec<Transition<S>>]) -> (Vec<Vec<S>>, Vec<Vec<S>>, Vec<Vec<S>>) {
        let k = self.k;
        let n = self.pivot_chol.len();
        let zero = self.pivot_chol[0][0].constant_like(0.0);
        let matmul = |x: &[S], y: &[S]| -> Vec<S> {
            let mut out = vec![zero.clone(); k * k];
            for r in 0..k {
                for c in 0..k {
                    let mut acc = zero.clone();
                    for m in 0..k {
                        acc = acc.add(&x[r * k + m].mul(&y[m * k + c]));
                    }
                    out[r * k + c] = acc;
                }
            }
            out
        };
        let transpose = |x: &[S]| -> Vec<S> { (0..k * k).map(|idx| x[(idx % k) * k + idx / k].clone()).collect() };
        let mut diag: Vec<Vec<S>> = vec![Vec::new(); n];
        let mut increment: Vec<Vec<S>> = vec![Vec::new(); n.saturating_sub(1)];
        let mut cross: Vec<Vec<S>> = vec![Vec::new(); n.saturating_sub(1)];
        diag[n - 1] = self.pivot_inv[n - 1].clone();
        for node in (0..n - 1).rev() {
            let inverse = &self.pivot_inv[node];
            let phi: Vec<&S> = (0..k).map(|atom| &transitions[node][atom].phi).collect();
            // A = D_n⁻¹ Φ diag(1/q): column c scaled by −off_c = φ_c / q_c.
            let a: Vec<S> = (0..k * k)
                .map(|idx| inverse[idx].mul(&self.off[node][idx % k]).neg())
                .collect();
            // B = I − Φ A = Φ D_n⁻¹ R_n Φ⁻¹; the second form has no
            // cancellation across a short gap, the first is finite at φ = 0.
            let short_gap = (0..k).all(|atom| phi[atom].value() > 0.5);
            let b = if short_gap {
                let mut b = matmul(inverse, &self.own[node]);
                for r in 0..k {
                    for c in 0..k {
                        b[r * k + c] = b[r * k + c].mul(phi[r]).mul(&recip(phi[c]));
                    }
                }
                b
            } else {
                let mut b = vec![zero.clone(); k * k];
                for r in 0..k {
                    for c in 0..k {
                        b[r * k + c] = phi[r].mul(&a[r * k + c]).neg();
                    }
                    b[r * k + r] = add_real(&b[r * k + r], 1.0);
                }
                b
            };
            let next = &diag[node + 1];
            let a_t = transpose(&a);
            let b_t = transpose(&b);
            let next_a_t = matmul(next, &a_t);
            let mut own = matmul(&a, &next_a_t);
            for (o, i) in own.iter_mut().zip(inverse.iter()) {
                *o = o.add(i);
            }
            let mut v = matmul(&b, &matmul(next, &b_t));
            let mut w = matmul(&b, &next_a_t);
            for r in 0..k {
                for c in 0..k {
                    let phi_inv_phi = phi[r].mul(&inverse[r * k + c]).mul(phi[c]);
                    v[r * k + c] = v[r * k + c].add(&phi_inv_phi);
                    w[r * k + c] = w[r * k + c].sub(&phi[r].mul(&inverse[r * k + c]));
                }
            }
            diag[node] = own;
            increment[node] = v;
            cross[node] = w;
        }
        (diag, increment, cross)
    }
}

fn newton_step<S: JetField>(
    assembly: &Assembly<S>,
    transitions: &[Vec<Transition<S>>],
    k: usize,
) -> Result<Vec<Vec<S>>, EventHistoryError> {
    let factor = BlockFactor::new(&assembly.own, &assembly.off, transitions, k)?;
    Ok(factor.solve(&assembly.gradient))
}

// ---- mode ------------------------------------------------------------------------

/// The unique maximiser of the complete-data objective, by Newton's method
/// with a backtracking line search. `start` warm-starts from an earlier
/// mode; the result does not depend on it.
pub(crate) fn find_mode(
    inputs: &SubjectInputs<'_, f64>,
    start: Option<&[f64]>,
) -> Result<Vec<f64>, EventHistoryError> {
    inputs.validate()?;
    let n = inputs.nodes.len();
    let atoms = inputs.atoms();
    if atoms == 0 {
        return Ok(Vec::new());
    }
    let shifts = inputs.shifts();
    let transitions = transitions(inputs);
    let mut z: Vec<Vec<f64>> = match start {
        Some(s) if s.len() == n * atoms => (0..n).map(|i| s[i * atoms..(i + 1) * atoms].to_vec()).collect(),
        _ => vec![vec![0.0; atoms]; n],
    };
    let mut assembly = assemble(inputs, &shifts, &transitions, &z);
    if !assembly.objective.is_finite() {
        z = vec![vec![0.0; atoms]; n];
        assembly = assemble(inputs, &shifts, &transitions, &z);
        if !assembly.objective.is_finite() {
            return Err(numerical("latent-path objective is not finite at the prior mean"));
        }
    }
    // The evidence's gradient is exact only at an exact mode: it assumes
    // `∇_z F(ẑ) = 0`, so an error `δ` in the mode leaves the objective
    // right to `O(δ²)` but the gradient wrong to `O(δ)`. A rule that stops
    // when the Newton DECREMENT reaches the objective's roundoff floor
    // stops at `δ ≈ √floor`, and the outer solver — which asks for a
    // stationarity of `1e-11` relative — then grinds against gradient noise
    // it cannot descend. So the mode is taken to where it cannot move at
    // all in `f64`: the loop runs until the accepted step is below the
    // resolution of `z` itself, which Newton's quadratic convergence
    // reaches within two or three iterations of the decrement floor.
    // Newton polish steps taken after the decrement fell below the
    // objective's own resolution. Convergence is quadratic there, so three
    // drive the residual to the floor of `f64`; letting the loop run on
    // instead burns its whole budget against a step that has stopped
    // shrinking.
    let mut polish = 0usize;
    for _ in 0..200 {
        let step = newton_step(&assembly, &transitions, atoms)?;
        let decrement: f64 = 0.5
            * assembly
                .gradient
                .iter()
                .zip(step.iter())
                .map(|(g, s)| g.iter().zip(s.iter()).map(|(a, b)| a * b).sum::<f64>())
                .sum::<f64>();
        if !decrement.is_finite() {
            return Err(numerical("latent-path Newton decrement is not finite"));
        }
        let floor = 8.0 * f64::EPSILON * (1.0 + assembly.objective.abs());
        let scale = z.iter().flatten().fold(0.0_f64, |m, v| m.max(v.abs()));
        let step_inf = step.iter().flatten().fold(0.0_f64, |m, v| m.max(v.abs()));
        if step_inf <= 8.0 * f64::EPSILON * (1.0 + scale) {
            return Ok(z.concat());
        }
        // Near the mode the decrement is below the objective's own
        // resolution and no line search can measure a gain; the full Newton
        // step is exact there, so it is taken on the step's own evidence.
        if decrement <= floor {
            for (zn, sn) in z.iter_mut().zip(step.iter()) {
                for (a, b) in zn.iter_mut().zip(sn.iter()) {
                    *a += b;
                }
            }
            polish += 1;
            if polish >= 3 {
                return Ok(z.concat());
            }
            assembly = assemble(inputs, &shifts, &transitions, &z);
            if !assembly.objective.is_finite() {
                return Err(numerical("latent-path objective left the finite range at the mode"));
            }
            continue;
        }
        let mut t = 1.0;
        loop {
            let trial: Vec<Vec<f64>> = z
                .iter()
                .zip(step.iter())
                .map(|(zn, sn)| zn.iter().zip(sn.iter()).map(|(a, b)| a + t * b).collect())
                .collect();
            let candidate = assemble(inputs, &shifts, &transitions, &trial);
            let predicted = decrement * (2.0 * t - t * t);
            if candidate.objective.is_finite()
                && candidate.objective - assembly.objective >= 0.25 * predicted - floor
            {
                z = trial;
                assembly = candidate;
                break;
            }
            t *= 0.5;
            if t < 1e-12 {
                return Err(numerical("latent-path line search made no progress"));
            }
        }
    }
    Err(numerical("latent-path Newton did not converge in 200 iterations"))
}

// ---- evidence and gradient ------------------------------------------------------

fn lift<S: JetField>(like: &S, mode: &[f64], n: usize, atoms: usize) -> Vec<Vec<S>> {
    (0..n)
        .map(|i| (0..atoms).map(|k| like.constant_like(mode[i * atoms + k])).collect())
        .collect()
}

/// The mode refined by `steps` Newton iterations in the caller's scalar,
/// with the assembly and factorisation at it.
struct Refined<S> {
    z: Vec<Vec<S>>,
    assembly: Assembly<S>,
    factor: BlockFactor<S>,
}

fn refined<S: JetField>(
    inputs: &SubjectInputs<'_, S>,
    shifts: &[S],
    transitions: &[Vec<Transition<S>>],
    mode: &[f64],
    steps: usize,
) -> Result<Refined<S>, EventHistoryError> {
    let n = inputs.nodes.len();
    let atoms = inputs.atoms();
    if mode.len() != n * atoms {
        return Err(numerical(format!(
            "latent mode has {} entries, expected {}",
            mode.len(),
            n * atoms
        )));
    }
    let mut z = lift(&inputs.eta0[0], mode, n, atoms);
    let mut assembly = assemble(inputs, shifts, transitions, &z);
    for _ in 0..steps {
        let step = newton_step(&assembly, transitions, atoms)?;
        for (zn, sn) in z.iter_mut().zip(step.iter()) {
            for (a, b) in zn.iter_mut().zip(sn.iter()) {
                *a = a.add(b);
            }
        }
        assembly = assemble(inputs, shifts, transitions, &z);
    }
    let factor = BlockFactor::new(&assembly.own, &assembly.off, transitions, atoms)?;
    Ok(Refined {
        z,
        assembly,
        factor,
    })
}

/// Number of Newton refinements of the lifted mode: exact jets through
/// third order, the highest any derivative channel here carries.
const REFINEMENTS: usize = 2;

/// The evidence at the mode `mode` (found by [`find_mode`] on the `f64`
/// values of the same inputs) and, when requested, its exact gradient.
pub(crate) fn evidence<S: JetField>(
    inputs: &SubjectInputs<'_, S>,
    mode: &[f64],
    derivatives: bool,
) -> Result<SubjectEvidence<S>, EventHistoryError> {
    inputs.validate()?;
    let nodes = inputs.nodes;
    let n = nodes.len();
    let marks = inputs.marks();
    let atoms = inputs.atoms();
    let like = &inputs.eta0[0];
    let zero = like.constant_like(0.0);
    if atoms == 0 {
        // No latent state: the Poisson-process log-likelihood itself.
        let mut loglik = zero.clone();
        let mut gradient = Vec::new();
        for node in 0..n {
            for d in 0..marks {
                let eta = &inputs.eta0[node * marks + d];
                let y = nodes.counts[[node, d]];
                let w = nodes.exposures[[node, d]];
                let m = exp(eta);
                if y != 0.0 {
                    loglik = loglik.add(&eta.scale(y));
                }
                if w != 0.0 {
                    loglik = loglik.sub(&m.scale(w));
                }
                if derivatives {
                    gradient.push(add_real(&m.scale(-w), y));
                }
            }
        }
        return Ok(SubjectEvidence { loglik, gradient });
    }
    let shifts = inputs.shifts();
    let transitions = transitions(inputs);
    let Refined {
        z,
        assembly,
        factor,
    } = refined(inputs, &shifts, &transitions, mode, REFINEMENTS)?;
    let loglik = assembly.objective.sub(&factor.logdet().scale(0.5));
    if !loglik.value().is_finite() {
        return Err(numerical("subject evidence is not finite"));
    }
    if !derivatives {
        return Ok(SubjectEvidence {
            loglik,
            gradient: Vec::new(),
        });
    }
    let (sigma, increment, cross) = factor.inverse_blocks(&transitions);
    // Per (node, mark): p = Σ_nn a_d, m = a_dᵀ p, s = y − w μ, c = w μ.
    let mut p: Vec<Vec<S>> = Vec::with_capacity(n * marks);
    let mut m: Vec<S> = Vec::with_capacity(n * marks);
    let mut s: Vec<S> = Vec::with_capacity(n * marks);
    let mut c: Vec<S> = Vec::with_capacity(n * marks);
    for node in 0..n {
        for d in 0..marks {
            let a = &inputs.loadings[d * atoms..(d + 1) * atoms];
            let pv = matvec(&sigma[node], a, atoms);
            let mv = dot(a, &pv);
            let y = nodes.counts[[node, d]];
            let w = nodes.exposures[[node, d]];
            let mu = &assembly.mu[node * marks + d];
            s.push(add_real(&mu.scale(-w), y));
            c.push(mu.scale(w));
            p.push(pv);
            m.push(mv);
        }
    }
    // c_{nk} = ½ Σ_d c_{nd} m_{nd} a_{dk};  v = H⁻¹ c.
    let cvec: Vec<Vec<S>> = (0..n)
        .map(|node| {
            (0..atoms)
                .map(|k| {
                    (0..marks).fold(zero.clone(), |acc, d| {
                        let idx = node * marks + d;
                        if nodes.exposures[[node, d]] == 0.0 {
                            return acc;
                        }
                        acc.add(&c[idx].mul(&m[idx]).mul(&inputs.loadings[d * atoms + k]).scale(0.5))
                    })
                })
                .collect()
        })
        .collect();
    let v = factor.solve(&cvec);
    let mut gradient = vec![zero.clone(); n * marks + marks * atoms + atoms];
    let a_offset = n * marks;
    let rho_offset = a_offset + marks * atoms;
    for node in 0..n {
        for d in 0..marks {
            let idx = node * marks + d;
            let y = nodes.counts[[node, d]];
            let w = nodes.exposures[[node, d]];
            if y == 0.0 && w == 0.0 {
                continue;
            }
            let a = &inputs.loadings[d * atoms..(d + 1) * atoms];
            let av = dot(a, &v[node]);
            // g_{η⁰} = s − ½ c m + c (a · v_n)
            gradient[idx] = s[idx]
                .sub(&c[idx].mul(&m[idx]).scale(0.5))
                .add(&c[idx].mul(&av));
            for k in 0..atoms {
                let zeta = z[node][k].sub(&a[k]);
                // g_a = s ζ − ½ c (ζ m + 2 p_k) + c ζ (a · v_n) − s v_{nk}
                let term = s[idx]
                    .mul(&zeta)
                    .sub(&c[idx].mul(&zeta.mul(&m[idx]).add(&p[idx][k].scale(2.0))).scale(0.5))
                    .add(&c[idx].mul(&zeta).mul(&av))
                    .sub(&s[idx].mul(&v[node][k]));
                let slot = a_offset + d * atoms + k;
                gradient[slot] = gradient[slot].add(&term);
            }
        }
    }
    for gap in 0..n.saturating_sub(1) {
        for k in 0..atoms {
            let t = &transitions[gap][k];
            let (zn, zn1) = (&z[gap][k], &z[gap + 1][k]);
            // Innovation coordinates: the mode residual r = ẑ' − φ ẑ, the
            // same contrast of the correction vector, and the gap's
            // posterior increment variance V and state covariance W.
            let r = zn1.sub(&t.phi.mul(zn));
            let rv = v[gap + 1][k].sub(&t.phi.mul(&v[gap][k]));
            let kk = k * atoms + k;
            let scaled_r = r.mul(&t.inv_q);
            // ∂_ρ of the gap term: −κφ [ r z/q − r² φ/q² + φ/q ].
            let dphi = scaled_r
                .mul(zn)
                .sub(&scaled_r.mul(&scaled_r).mul(&t.phi))
                .add(&t.phi.mul(&t.inv_q));
            let explicit = t.kappa.mul(&t.phi).mul(&dphi).neg();
            // ½ tr(Σ ∂_ρ Q) = ½ [ ∂_ρ(1/q) V + 2 (κφ/q) W ] with
            // ∂_ρ(1/q) = −2κφ²/q².
            let d_inv_q = t.kappa.mul(&square(&t.phi)).mul(&square(&t.inv_q)).scale(-2.0);
            let kappa_phi_over_q = t.kappa.mul(&t.phi).mul(&t.inv_q);
            let trace = d_inv_q
                .mul(&increment[gap][kk])
                .add(&kappa_phi_over_q.mul(&cross[gap][kk]).scale(2.0))
                .scale(0.5);
            // vᵀ ∂_ρQ ẑ = r_v r ∂_ρ(1/q) + (κφ/q) (v_n r + r_v ẑ_n).
            let vterm = rv
                .mul(&r)
                .mul(&d_inv_q)
                .add(&kappa_phi_over_q.mul(&v[gap][k].mul(&r).add(&rv.mul(zn))));
            let slot = rho_offset + k;
            gradient[slot] = gradient[slot].add(&explicit).sub(&trace).add(&vterm);
        }
    }
    Ok(SubjectEvidence { loglik, gradient })
}

// ---- smoother -------------------------------------------------------------------

/// The Laplace posterior of one subject's latent path and the derived
/// posterior-mean intensities and residual scores.
pub struct Smoother {
    /// Posterior mean per node and atom, `n * atoms + k`.
    pub means: Vec<f64>,
    /// Posterior covariance per node, `n * atoms² + k * atoms + j`.
    pub covariances: Vec<f64>,
    /// Posterior-mean intensity `E[λ_{nd} | data]`, index `n * marks + d`.
    pub intensity: Vec<f64>,
}

impl Smoother {
    /// The posterior at node `node` as a Gaussian.
    pub(crate) fn at(&self, node: usize, atoms: usize) -> Gaussian {
        Gaussian {
            mean: self.means[node * atoms..(node + 1) * atoms].to_vec(),
            cov: self.covariances[node * atoms * atoms..(node + 1) * atoms * atoms].to_vec(),
        }
    }
}

/// The Laplace posterior at the mode.
pub(crate) fn smoother(inputs: &SubjectInputs<'_, f64>, mode: &[f64]) -> Result<Smoother, EventHistoryError> {
    inputs.validate()?;
    let nodes = inputs.nodes;
    let n = nodes.len();
    let marks = inputs.marks();
    let atoms = inputs.atoms();
    if atoms == 0 {
        let intensity = (0..n * marks).map(|i| inputs.eta0[i].exp()).collect();
        return Ok(Smoother {
            means: Vec::new(),
            covariances: Vec::new(),
            intensity,
        });
    }
    let shifts = inputs.shifts();
    let transitions = transitions(inputs);
    let Refined {
        z,
        assembly,
        factor,
    } = refined(inputs, &shifts, &transitions, mode, 0)?;
    let (sigma, _, _) = factor.inverse_blocks(&transitions);
    let mut intensity = Vec::with_capacity(n * marks);
    for node in 0..n {
        for d in 0..marks {
            let a = &inputs.loadings[d * atoms..(d + 1) * atoms];
            let variance = dot(a, &matvec(&sigma[node], a, atoms));
            intensity.push(assembly.mu[node * marks + d] * (0.5 * variance).exp());
        }
    }
    Ok(Smoother {
        means: z.concat(),
        covariances: sigma.concat(),
        intensity,
    })
}

/// The follow-up average of the latent state, `z̄ = Σ_n ω_n z_n` with
/// `ω_n` the node weights normalised to one, as a posterior Gaussian: mean
/// `Σ ω_n ẑ_n`, covariance `Ωᵀ H⁻¹ Ω` (exact under the Laplace posterior;
/// one block-tridiagonal solve per atom).
pub(crate) fn exposure(inputs: &SubjectInputs<'_, f64>, mode: &[f64]) -> Result<Gaussian, EventHistoryError> {
    inputs.validate()?;
    let nodes = inputs.nodes;
    let n = nodes.len();
    let atoms = inputs.atoms();
    let total: f64 = nodes.weights.iter().sum();
    if !(total > 0.0) {
        return Err(numerical("latent exposure needs a positive follow-up length"));
    }
    if atoms == 0 {
        return Ok(Gaussian {
            mean: Vec::new(),
            cov: Vec::new(),
        });
    }
    let shifts = inputs.shifts();
    let transitions = transitions(inputs);
    let refined = refined(inputs, &shifts, &transitions, mode, 0)?;
    let omega: Vec<f64> = nodes.weights.iter().map(|w| w / total).collect();
    let mean: Vec<f64> = (0..atoms)
        .map(|k| (0..n).map(|node| omega[node] * refined.z[node][k]).sum())
        .collect();
    let mut cov = vec![0.0; atoms * atoms];
    for j in 0..atoms {
        let rhs: Vec<Vec<f64>> = (0..n)
            .map(|node| (0..atoms).map(|k| if k == j { omega[node] } else { 0.0 }).collect())
            .collect();
        let column = refined.factor.solve(&rhs);
        for k in 0..atoms {
            cov[k * atoms + j] = (0..n).map(|node| omega[node] * column[node][k]).sum();
        }
    }
    Ok(Gaussian { mean, cov })
}

// ---- sequential Gaussian filter --------------------------------------------------

/// A Gaussian state of the atoms: mean (`K`) and covariance (`K × K`).
#[derive(Clone, Debug)]
pub(crate) struct Gaussian {
    pub mean: Vec<f64>,
    pub cov: Vec<f64>,
}

impl Gaussian {
    pub fn standard(atoms: usize) -> Self {
        let mut cov = vec![0.0; atoms * atoms];
        for k in 0..atoms {
            cov[k * atoms + k] = 1.0;
        }
        Self {
            mean: vec![0.0; atoms],
            cov,
        }
    }

    /// The state propagated across a gap.
    fn predict(&self, transitions: &[Transition<f64>]) -> Self {
        let atoms = self.mean.len();
        let mean: Vec<f64> = (0..atoms).map(|k| transitions[k].phi * self.mean[k]).collect();
        let mut cov = vec![0.0; atoms * atoms];
        for k in 0..atoms {
            for j in 0..atoms {
                cov[k * atoms + j] = transitions[k].phi * transitions[j].phi * self.cov[k * atoms + j];
            }
            cov[k * atoms + k] += transitions[k].q;
        }
        Self { mean, cov }
    }
}

/// One completed sequential filter: the predicted state at every node and
/// the per-node log normalisers `ln ∫ e^{ℓ_n(z)} N(z; predicted) dz`, whose
/// sum is the log predictive probability of the observed counts.
pub(crate) struct FilterPass {
    pub predicted: Vec<Gaussian>,
    pub log_normalisers: Vec<f64>,
}

/// Run the sequential filter, optionally continuing from a state `gap`
/// before the first node, with the compensator restricted to `compensated`
/// marks (counts of every mark are always conditioned on).
pub(crate) fn filter_pass(
    inputs: &SubjectInputs<'_, f64>,
    initial: Option<(&Gaussian, f64)>,
    compensated: &[bool],
) -> Result<FilterPass, EventHistoryError> {
    inputs.validate()?;
    let nodes = inputs.nodes;
    let n = nodes.len();
    let marks = inputs.marks();
    let atoms = inputs.atoms();
    if compensated.len() != marks {
        return Err(numerical("filter needs one compensator flag per mark"));
    }
    let shifts = inputs.shifts();
    let transitions = transitions(inputs);
    let mut predicted = Vec::with_capacity(n);
    let mut log_normalisers = Vec::with_capacity(n);
    let mut state = match initial {
        None => Gaussian::standard(atoms),
        Some((start, gap)) => {
            let across: Vec<Transition<f64>> = inputs
                .log_rates
                .iter()
                .map(|rho| transition(rho, gap, inputs.time_scale))
                .collect();
            start.predict(&across)
        }
    };
    for node in 0..n {
        if node > 0 {
            state = state.predict(&transitions[node - 1]);
        }
        predicted.push(state.clone());
        let exposure = |d: usize| if compensated[d] { nodes.exposures[[node, d]] } else { 0.0 };
        let informative = (0..marks).any(|d| nodes.counts[[node, d]] != 0.0 || exposure(d) != 0.0);
        if !informative {
            log_normalisers.push(0.0);
            continue;
        }
        if atoms == 0 {
            let mut value = 0.0;
            for d in 0..marks {
                let eta = inputs.eta0[node * marks + d];
                value += nodes.counts[[node, d]] * eta - exposure(d) * eta.exp();
            }
            log_normalisers.push(value);
            continue;
        }
        // Laplace update of one node: maximise ℓ_n(z) − ½ (z − m)ᵀ P⁻¹ (z − m).
        let prior_chol = cholesky(&state.cov, atoms, "filter predicted covariance")?;
        let precision = cholesky_inverse(&prior_chol, atoms);
        let prior_logdet = cholesky_logdet(&prior_chol, atoms);
        let node_terms = |z: &[f64]| -> (f64, Vec<f64>, Vec<f64>) {
            let mut value = 0.0;
            let mut grad = vec![0.0; atoms];
            let mut hess = precision.clone();
            for d in 0..marks {
                let y = nodes.counts[[node, d]];
                let w = exposure(d);
                if y == 0.0 && w == 0.0 {
                    continue;
                }
                let a = &inputs.loadings[d * atoms..(d + 1) * atoms];
                let eta = inputs.eta0[node * marks + d] + shifts[d] + dot(a, z);
                let m = eta.exp();
                value += y * eta - w * m;
                let score = y - w * m;
                for k in 0..atoms {
                    grad[k] += score * a[k];
                    for j in 0..atoms {
                        hess[k * atoms + j] += w * m * a[k] * a[j];
                    }
                }
            }
            let centred: Vec<f64> = (0..atoms).map(|k| z[k] - state.mean[k]).collect();
            let pulled = matvec(&precision, &centred, atoms);
            value -= 0.5 * dot(&centred, &pulled);
            for k in 0..atoms {
                grad[k] -= pulled[k];
            }
            (value, grad, hess)
        };
        let mut z = state.mean.clone();
        let (mut value, mut grad, mut hess) = node_terms(&z);
        let mut stalls = 0usize;
        let mut converged = false;
        for _ in 0..200 {
            let chol = cholesky(&hess, atoms, "filter node curvature")?;
            let mut step = grad.clone();
            cholesky_solve(&chol, atoms, &mut step);
            let decrement = 0.5 * dot(&grad, &step);
            let floor = 8.0 * f64::EPSILON * (1.0 + value.abs());
            if !decrement.is_finite() {
                return Err(numerical("filter Newton decrement is not finite"));
            }
            if decrement <= floor {
                converged = true;
                break;
            }
            let mut t = 1.0;
            loop {
                let trial: Vec<f64> = (0..atoms).map(|k| z[k] + t * step[k]).collect();
                let (tv, tg, th) = node_terms(&trial);
                if tv.is_finite() && tv - value >= 0.25 * decrement * (2.0 * t - t * t) - floor {
                    stalls = if tv - value <= floor { stalls + 1 } else { 0 };
                    z = trial;
                    value = tv;
                    grad = tg;
                    hess = th;
                    break;
                }
                t *= 0.5;
                if t < 1e-12 {
                    return Err(numerical("filter line search made no progress"));
                }
            }
            if stalls >= 2 {
                converged = true;
                break;
            }
        }
        if !converged {
            return Err(numerical("filter node update did not converge"));
        }
        let chol = cholesky(&hess, atoms, "filter node curvature")?;
        log_normalisers.push(value - 0.5 * prior_logdet - 0.5 * cholesky_logdet(&chol, atoms));
        state = Gaussian {
            mean: z,
            cov: cholesky_inverse(&chol, atoms),
        };
    }
    Ok(FilterPass {
        predicted,
        log_normalisers,
    })
}

/// `E[λ_{d}]` at a node under a Gaussian state: `exp(η⁰ + shift + a·m + ½ aᵀ P a)`.
pub(crate) fn expected_intensity(eta0: f64, loadings_d: &[f64], state: &Gaussian) -> f64 {
    let atoms = loadings_d.len();
    if atoms == 0 {
        return eta0.exp();
    }
    let shift: f64 = -0.5 * loadings_d.iter().map(|a| a * a).sum::<f64>();
    let variance = dot(loadings_d, &matvec(&state.cov, loadings_d, atoms));
    (eta0 + shift + dot(loadings_d, &state.mean) + 0.5 * variance).exp()
}
