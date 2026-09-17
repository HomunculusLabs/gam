//! The declared-law anchor of the survival marginal-slope index (gam#2923).
//!
//! The family's defining identity is that `q(t, a)` is the MARGINAL survival
//! index: `E_z[Φ(−η) | a] = Φ(−q)`. The closed-form lowering
//! `η = q·√(1 + rᵀΣr) + rᵀz` is that identity solved under `z | a ~ N(0, Σ)`
//! and nothing else. On a declared finite law `F_a` — nodes `u_k`, weights
//! `w_k` on the scalar the slope projects the score onto — the identity is
//! instead the anchoring equation
//!
//! ```text
//!     Σ_k w_k Φ(−(α + b·u_k)) = Φ(−q),        b = s_f·g,
//! ```
//!
//! whose left side is continuous, strictly decreasing in `α` and runs from 1
//! to 0, so `α(q, b)` exists and is unique for every `(q, b)`
//! (`Descent.Portability.MarginalAnchor.exists_unique_anchor`, stated for the
//! logit link; the argument is link-agnostic). `α` replaces `q·c` in the row
//! index, and its derivatives come from implicit differentiation of the same
//! equation (`anchor_deriv_eq`): with `G(α, q, b) = Σ_k w_k Φ(−(α + b u_k)) −
//! Φ(−q)`,
//!
//! ```text
//!     α_θ = −G_θ / G_α = −Σ_k w_k φ(η_k) ∂_θ(b u_k) / Σ_k w_k φ(η_k),   η_k = α + b u_k,
//! ```
//!
//! and the second and third orders likewise.
//!
//! This module owns three things and nothing else: the root itself
//! ([`solve_anchor`]), its closed-form derivatives through order three
//! ([`AnchorDerivatives`]) for the direct order-two lowering, and its lift onto
//! any [`JetField`] ([`anchor_jet`]) for the higher-order towers. The Gaussian
//! closed form is the special case of every one of them: on a Gaussian
//! quadrature grid the root is `q·√(1 + b²)` to quadrature tolerance, which is
//! what `anchor_tests::gaussian_grid_reproduces_the_closed_form` pins.

use gam_math::nested_dual::JetField;
use gam_math::probability::{normal_cdf, normal_logcdf, normal_pdf};
use crate::monotone_root::solve_monotone_root;

/// `|log-residual|` at which the anchoring equation counts as solved. The
/// residual is `log Σ_k w_k Φ(∓η_k) − log Φ(∓q)` on whichever tail is the
/// smaller, i.e. the RELATIVE error of the anchored marginal probability on
/// the side where relative error is the meaningful one; the `4·ε` floor keeps
/// the contract honest where that log is itself within round-off of zero.
pub(crate) const ANCHOR_LOG_RESIDUAL_TOL: f64 = 1e-13;

/// The largest `|log-residual|` a returned root may carry: a relative error of
/// `1e-10` on the smaller marginal tail, three orders above the target the
/// solver refines to, so a bracket-width stop is admitted and a stalled
/// search is not.
pub(crate) const ANCHOR_RESIDUAL_ACCEPTANCE: f64 = 1e-10;

const LOG_SQRT_TAU: f64 = 0.918_938_533_204_672_8;

/// A finite law on the projected score, borrowed from wherever the family
/// materialised it: nodes ascending, positive weights summing to one, and the
/// log-weights the solver's log-space residual reads.
#[derive(Clone, Copy)]
pub(crate) struct AnchorGrid<'a> {
    pub(crate) nodes: &'a [f64],
    pub(crate) weights: &'a [f64],
    pub(crate) log_weights: &'a [f64],
}

impl<'a> AnchorGrid<'a> {
    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.nodes.len()
    }
}

/// An owned finite law: what the family materialises once per fit (one for a
/// global law, one per row for a local one) and hands out as [`AnchorGrid`]s.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct AnchorGridOwned {
    pub(crate) nodes: Vec<f64>,
    pub(crate) weights: Vec<f64>,
    pub(crate) log_weights: Vec<f64>,
}

impl AnchorGridOwned {
    pub(crate) fn from_grid(grid: &crate::bms::EmpiricalZGrid) -> Self {
        Self::new(grid.nodes.clone(), grid.weights.clone())
    }

    pub(crate) fn new(nodes: Vec<f64>, weights: Vec<f64>) -> Self {
        let log_weights = weights.iter().map(|w| w.ln()).collect();
        Self {
            nodes,
            weights,
            log_weights,
        }
    }

    #[inline]
    pub(crate) fn view(&self) -> AnchorGrid<'_> {
        AnchorGrid {
            nodes: &self.nodes,
            weights: &self.weights,
            log_weights: &self.log_weights,
        }
    }
}

/// The declared latent law a fit anchors on, materialised once per family.
///
/// `kind` is the persisted object — the same [`LatentMeasureKind`] the
/// Bernoulli family writes and the shared predictor replays — and the grids
/// are its per-row projection onto the scalar the shared slope reads: one
/// grid for a global law, one per training row for a local mixture.
#[derive(Clone, Debug)]
pub(crate) struct SurvivalLatentLaw {
    pub(crate) kind: crate::bms::LatentMeasureKind,
    grids: LawGrids,
}

#[derive(Clone, Debug)]
enum LawGrids {
    Global(AnchorGridOwned),
    PerRow(Vec<AnchorGridOwned>),
}

impl SurvivalLatentLaw {
    /// Materialise the law's per-row grids for `n` training rows. `None` for
    /// the standard-normal law, on which the closed form is the anchor.
    pub(crate) fn from_kind(
        kind: &crate::bms::LatentMeasureKind,
        n: usize,
    ) -> Result<Option<Self>, String> {
        use crate::bms::LatentMeasureKind;
        kind.validate("survival marginal-slope declared latent law")?;
        let grids = match kind {
            LatentMeasureKind::StandardNormal => return Ok(None),
            LatentMeasureKind::GlobalEmpirical { grid } => {
                LawGrids::Global(AnchorGridOwned::from_grid(grid))
            }
            LatentMeasureKind::LocalEmpirical { .. } => {
                let mut per_row = Vec::with_capacity(n);
                for row in 0..n {
                    let grid = kind.empirical_grid_for_training_row(row)?.ok_or_else(|| {
                        format!(
                            "survival marginal-slope local latent law produced no grid for row {row}"
                        )
                    })?;
                    per_row.push(AnchorGridOwned::from_grid(&grid));
                }
                LawGrids::PerRow(per_row)
            }
        };
        Ok(Some(Self {
            kind: kind.clone(),
            grids,
        }))
    }

    /// The law of row `row`.
    #[inline]
    pub(crate) fn row(&self, row: usize) -> AnchorGrid<'_> {
        match &self.grids {
            LawGrids::Global(grid) => grid.view(),
            LawGrids::PerRow(grids) => grids[row].view(),
        }
    }
}

/// The Gaussian closed form, which is the anchor's seed and — on a Gaussian
/// law — its value.
#[inline]
pub(crate) fn gaussian_anchor(q: f64, observed_slope: f64) -> f64 {
    q * (1.0 + observed_slope * observed_slope).sqrt()
}

/// Solve the anchoring equation for `α` at the marginal index `q` and the
/// OBSERVED slope `b` (frailty scale already applied).
///
/// Bracketed Newton–Halley on a log-space residual: with `S(α) = Σ_k w_k
/// Φ(−η_k)` the residual is `log S − log Φ(−q)` when `Φ(−q) ≤ ½` and
/// `log(1 − S) − log Φ(q)` otherwise, so the tolerance is always a relative
/// error on the smaller of the two marginal tails. Both sides are strictly
/// monotone in `α`, the log-sum-exp keeps the sum finite where every node has
/// underflowed, and the shared solver brackets by doubling from the Gaussian
/// closed-form seed whenever the Newton probe is not trusted.
pub(crate) fn solve_anchor(q: f64, observed_slope: f64, grid: AnchorGrid<'_>) -> Result<f64, String> {
    if !(q.is_finite() && observed_slope.is_finite()) {
        return Err(format!(
            "survival marginal-slope anchor requires finite q and slope, got q={q}, b={observed_slope}"
        ));
    }
    let survival_side = q >= 0.0;
    let log_target = if survival_side {
        normal_logcdf(-q)
    } else {
        normal_logcdf(q)
    };
    let eval = |alpha: f64| -> Result<(f64, f64, f64), String> {
        anchor_log_residual(alpha, observed_slope, grid, survival_side, log_target)
    };
    let seed = gaussian_anchor(q, observed_slope);
    let tol = ANCHOR_LOG_RESIDUAL_TOL.max(4.0 * f64::EPSILON);
    let (root, _, residual) = solve_monotone_root(
        eval,
        seed,
        "survival marginal-slope anchor",
        tol,
        64,
        48,
    )
    .map_err(|error| error.to_string())?;
    // The shared solver also stops on bracket width, so a converged root can
    // sit a few multiples of `tol·|F′|` off the residual target; what this
    // guards against is a root handed back with a residual that is not small
    // at all, which the solver's own contract does not exclude.
    if !(residual.abs() <= ANCHOR_RESIDUAL_ACCEPTANCE) {
        return Err(format!(
            "survival marginal-slope anchor solve did not converge: log-residual={residual:.3e} at α={root:.6} (q={q:.6}, b={observed_slope:.6})"
        ));
    }
    Ok(root)
}

/// `(F, F′, F″)` of the log-space anchoring residual at `α`.
fn anchor_log_residual(
    alpha: f64,
    observed_slope: f64,
    grid: AnchorGrid<'_>,
    survival_side: bool,
    log_target: f64,
) -> Result<(f64, f64, f64), String> {
    let m = grid.len();
    // log Σ_k w_k Φ(∓η_k) by log-sum-exp, and the density ratios
    // `m_k = w_k φ(η_k) / Σ` that the derivatives read.
    let mut log_terms: [f64; 512] = [0.0; 512];
    let mut heap: Vec<f64> = Vec::new();
    let terms: &mut [f64] = if m <= 512 {
        &mut log_terms[..m]
    } else {
        heap.resize(m, 0.0);
        &mut heap[..]
    };
    let mut log_max = f64::NEG_INFINITY;
    for k in 0..m {
        let eta = alpha + observed_slope * grid.nodes[k];
        let log_cdf = if survival_side {
            normal_logcdf(-eta)
        } else {
            normal_logcdf(eta)
        };
        let term = grid.log_weights[k] + log_cdf;
        terms[k] = term;
        if term > log_max {
            log_max = term;
        }
    }
    if !log_max.is_finite() {
        return Err(format!(
            "survival marginal-slope anchor residual is non-finite at α={alpha}, b={observed_slope}"
        ));
    }
    let mut sum = 0.0;
    for term in terms.iter() {
        sum += (term - log_max).exp();
    }
    let log_sum = log_max + sum.ln();
    let value = log_sum - log_target;
    // Density ratios: m_k = w_k φ(η_k) / S, each formed in log space.
    let mut ratio_sum = 0.0;
    let mut ratio_eta_sum = 0.0;
    for k in 0..m {
        let eta = alpha + observed_slope * grid.nodes[k];
        let log_ratio = grid.log_weights[k] - 0.5 * eta * eta - LOG_SQRT_TAU - log_sum;
        let ratio = if log_ratio > -740.0 { log_ratio.exp() } else { 0.0 };
        ratio_sum += ratio;
        ratio_eta_sum += ratio * eta;
    }
    // Survival side: S′ = −Σ w φ, S″ = Σ w η φ.
    // Complement side: C′ = Σ w φ, C″ = −Σ w η φ.
    let (first, second) = if survival_side {
        (-ratio_sum, ratio_eta_sum - ratio_sum * ratio_sum)
    } else {
        (ratio_sum, -ratio_eta_sum - ratio_sum * ratio_sum)
    };
    // `F′` is mathematically non-zero on both sides; where every density ratio
    // underflowed the monotone solver only needs its sign to bracket.
    let first = if first == 0.0 {
        if survival_side {
            -f64::MIN_POSITIVE
        } else {
            f64::MIN_POSITIVE
        }
    } else {
        first
    };
    if !(value.is_finite() && first.is_finite() && second.is_finite()) {
        return Err(format!(
            "survival marginal-slope anchor residual state is non-finite: F={value}, F′={first}, F″={second} at α={alpha}"
        ));
    }
    Ok((value, first, second))
}

/// The anchor and its derivatives in `(q, b)` through order three, by implicit
/// differentiation of `G(α, q, b) = Σ_k w_k Φ(−(α + b u_k)) − Φ(−q) = 0`.
///
/// Third order is the highest the direct order-two lowering of the anchored
/// frame reads: the exit-time rate feature is `α̇₁ = α_q(q₁, b)·q̇₁`, whose
/// curvature in the primaries is a third derivative of `α`. Everything above
/// that comes through [`anchor_jet`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct AnchorDerivatives {
    pub(crate) alpha: f64,
    pub(crate) a_q: f64,
    pub(crate) a_b: f64,
    pub(crate) a_qq: f64,
    pub(crate) a_qb: f64,
    pub(crate) a_bb: f64,
    pub(crate) a_qqq: f64,
    pub(crate) a_qqb: f64,
    pub(crate) a_qbb: f64,
    pub(crate) a_bbb: f64,
}

/// Which of `(q, b)` an implicit-derivative index names.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Var {
    Q,
    B,
}

impl AnchorDerivatives {
    /// Solve the anchor and differentiate it.
    pub(crate) fn solve(q: f64, observed_slope: f64, grid: AnchorGrid<'_>) -> Result<Self, String> {
        let alpha = solve_anchor(q, observed_slope, grid)?;
        Self::at(alpha, q, observed_slope, grid)
    }

    /// Differentiate at an already solved anchor.
    pub(crate) fn at(
        alpha: f64,
        q: f64,
        observed_slope: f64,
        grid: AnchorGrid<'_>,
    ) -> Result<Self, String> {
        // Moments of the law against the derivatives of `F(x) = Φ(−x)`:
        //   F′ = −φ, F″ = xφ, F‴ = (1 − x²)φ,
        // each weighted by `u_k^j` for the `b`-derivatives.
        let mut p = [0.0_f64; 4];
        let mut s2 = [0.0_f64; 4];
        let mut s3 = [0.0_f64; 4];
        for (&u, &w) in grid.nodes.iter().zip(grid.weights.iter()) {
            let eta = alpha + observed_slope * u;
            let pdf = w * normal_pdf(eta);
            let mut power = 1.0;
            for j in 0..4 {
                p[j] += pdf * power;
                s2[j] += pdf * eta * power;
                s3[j] += pdf * (1.0 - eta * eta) * power;
                power *= u;
            }
        }
        let phi_q = normal_pdf(q);
        // Partial derivatives of G by order. In `q` alone: `−F^{(n)}(q) =
        // −(−1)^n He_{n−1}(q) φ(q)`; in `(α, b)` the law's moments above, by
        // total order and by the number of `b`'s. Order zero is never read.
        let g_q = [0.0, phi_q, -q * phi_q, -(1.0 - q * q) * phi_q];
        let g_ab = [[0.0; 4], [-p[0], -p[1], -p[2], -p[3]], s2, s3];
        // `na` in α, `nq` in q, `nb` in b; mixed q/(α, b) derivatives vanish.
        let g = |na: usize, nq: usize, nb: usize| -> f64 {
            if nq > 0 {
                if na + nb > 0 { 0.0 } else { g_q[nq] }
            } else {
                g_ab[na + nb][nb]
            }
        };
        let g_alpha = g(1, 0, 0);
        if !(g_alpha.is_finite() && g_alpha < 0.0) {
            return Err(format!(
                "survival marginal-slope anchor has no finite gradient: G_α={g_alpha:e} at α={alpha}, q={q}, b={observed_slope}"
            ));
        }
        let idx = |v: Var| -> (usize, usize) {
            match v {
                Var::Q => (1, 0),
                Var::B => (0, 1),
            }
        };
        // G with `na` α-derivatives and the listed (q, b) variables.
        let gp = |na: usize, vars: &[Var]| -> f64 {
            let mut nq = 0;
            let mut nb = 0;
            for v in vars {
                let (dq, db) = idx(*v);
                nq += dq;
                nb += db;
            }
            g(na, nq, nb)
        };
        let first = |x: Var| -> f64 { -gp(0, &[x]) / g_alpha };
        let a1 = |x: Var| first(x);
        let second = |x: Var, y: Var| -> f64 {
            -(gp(0, &[x, y]) + gp(1, &[x]) * a1(y) + gp(1, &[y]) * a1(x) + gp(2, &[]) * a1(x) * a1(y))
                / g_alpha
        };
        let third = |x: Var, y: Var, z: Var| -> f64 {
            let ax = a1(x);
            let ay = a1(y);
            let az = a1(z);
            let axy = second(x, y);
            let axz = second(x, z);
            let ayz = second(y, z);
            -(gp(0, &[x, y, z])
                + gp(1, &[x, y]) * az
                + (gp(1, &[x, z]) + gp(2, &[x]) * az) * ay
                + gp(1, &[x]) * ayz
                + (gp(1, &[y, z]) + gp(2, &[y]) * az) * ax
                + gp(1, &[y]) * axz
                + (gp(2, &[z]) + gp(3, &[]) * az) * ax * ay
                + gp(2, &[]) * (axz * ay + ax * ayz)
                + (gp(1, &[z]) + gp(2, &[]) * az) * axy)
                / g_alpha
        };
        let out = Self {
            alpha,
            a_q: first(Var::Q),
            a_b: first(Var::B),
            a_qq: second(Var::Q, Var::Q),
            a_qb: second(Var::Q, Var::B),
            a_bb: second(Var::B, Var::B),
            a_qqq: third(Var::Q, Var::Q, Var::Q),
            a_qqb: third(Var::Q, Var::Q, Var::B),
            a_qbb: third(Var::Q, Var::B, Var::B),
            a_bbb: third(Var::B, Var::B, Var::B),
        };
        Ok(out)
    }
}

// ── Faà-di-Bruno stacks of the unary leaves the jet lift composes ──────────

/// `[Φ(−x), F′, F″, F‴, F⁗]` with `F(x) = Φ(−x)`: `F^{(n)} = (−1)^n He_{n−1}(x) φ(x)`.
#[inline]
fn survival_cdf_stack(x: f64) -> [f64; 5] {
    let pdf = normal_pdf(x);
    [
        normal_cdf(-x),
        -pdf,
        x * pdf,
        (1.0 - x * x) * pdf,
        (x * x * x - 3.0 * x) * pdf,
    ]
}

/// `[F′, F″, F‴, F⁗, F⁽⁵⁾]` of the same `F`: the stack of `−φ`.
#[inline]
fn survival_cdf_prime_stack(x: f64) -> [f64; 5] {
    let pdf = normal_pdf(x);
    let x2 = x * x;
    [
        -pdf,
        x * pdf,
        (1.0 - x2) * pdf,
        (x2 * x - 3.0 * x) * pdf,
        -(x2 * x2 - 6.0 * x2 + 3.0) * pdf,
    ]
}

/// `[φ, φ′, φ″, φ‴, φ⁗]`: `φ^{(n)} = (−1)^n He_n(x) φ(x)`.
#[inline]
fn pdf_stack(x: f64) -> [f64; 5] {
    let pdf = normal_pdf(x);
    let x2 = x * x;
    [
        pdf,
        -x * pdf,
        (x2 - 1.0) * pdf,
        -(x2 * x - 3.0 * x) * pdf,
        (x2 * x2 - 6.0 * x2 + 3.0) * pdf,
    ]
}

/// `[1/x, −1/x², 2/x³, −6/x⁴, 24/x⁵]`.
#[inline]
fn reciprocal_stack(x: f64) -> [f64; 5] {
    let inv = 1.0 / x;
    let inv2 = inv * inv;
    let inv3 = inv2 * inv;
    [inv, -inv2, 2.0 * inv3, -6.0 * inv3 * inv, 24.0 * inv3 * inv2]
}

/// The anchor as a jet in whatever `q` and `b` carry.
///
/// Newton's iteration in the truncated jet algebra: from the exact real root
/// (every derivative channel zero) one step is exact through first order, two
/// through third, three through seventh — enough for every tower this family
/// evaluates. The value channel is then pinned back to the solver's root with
/// [`JetField::with_value`], so the value path and every jet path evaluate the
/// row at bitwise the same anchor. On `f64` the three steps cost three grid
/// sweeps and change nothing.
///
/// `alpha_star` MUST be the converged root at `(q.value(), b.value())`; the
/// caller solves it once and lifts it here.
pub(crate) fn anchor_jet<T: JetField>(alpha_star: f64, q: &T, b: &T, grid: AnchorGrid<'_>) -> T {
    let mut alpha = q.constant_like(alpha_star);
    if std::mem::size_of::<T>() == std::mem::size_of::<f64>() {
        // A carrier the size of one `f64` has no derivative channels (every
        // jet carries at least one beside its value): the root is the whole
        // answer, and three more grid sweeps would only re-derive it.
        return alpha;
    }
    for _ in 0..3 {
        // G = Σ_k w_k F(η_k) − F(q),   G_α = Σ_k w_k F′(η_k).
        let mut g = q.compose_unary(survival_cdf_stack(q.value())).neg();
        let mut g_alpha = q.constant_like(0.0);
        for (&u, &w) in grid.nodes.iter().zip(grid.weights.iter()) {
            let eta = alpha.add(&b.scale(u));
            let eta_value = eta.value();
            g = g.add(&eta.compose_unary(survival_cdf_stack(eta_value)).scale(w));
            g_alpha = g_alpha.add(&eta.compose_unary(survival_cdf_prime_stack(eta_value)).scale(w));
        }
        let step = g.mul(&g_alpha.compose_unary(reciprocal_stack(g_alpha.value())));
        alpha = alpha.sub(&step);
    }
    alpha.with_value(alpha_star)
}

/// `∂α/∂q = φ(q) / Σ_k w_k φ(η_k)` as a jet, at an anchor jet already lifted
/// by [`anchor_jet`]. This is the factor the exit-time rate feature
/// `α̇₁ = α_q · q̇₁` carries on a time-constant slope.
pub(crate) fn anchor_q_derivative_jet<T: JetField>(
    alpha: &T,
    q: &T,
    b: &T,
    grid: AnchorGrid<'_>,
) -> T {
    let phi_q = q.compose_unary(pdf_stack(q.value()));
    let mut density = q.constant_like(0.0);
    for (&u, &w) in grid.nodes.iter().zip(grid.weights.iter()) {
        let eta = alpha.add(&b.scale(u));
        density = density.add(&eta.compose_unary(pdf_stack(eta.value())).scale(w));
    }
    phi_q.mul(&density.compose_unary(reciprocal_stack(density.value())))
}

#[cfg(test)]
mod anchor_tests {
    use super::super::test_support::{gauss_hermite_probabilists, skewed_grid};
    use super::*;
    use gam_math::jet_scalar::{JetScalar, Order2};
    use gam_math::jet_tower::Tower4;

    /// `∂α/∂b = −Σ_k w_k φ(η_k) u_k / Σ_k w_k φ(η_k)` as a jet: the factor a
    /// follow-up-varying slope's rate `ḃ₁` enters `α̇₁` with.
    fn anchor_b_derivative_jet<T: JetField>(alpha: &T, q: &T, b: &T, grid: AnchorGrid<'_>) -> T {
        let mut density = q.constant_like(0.0);
        let mut moment = q.constant_like(0.0);
        for (&u, &w) in grid.nodes.iter().zip(grid.weights.iter()) {
            let eta = alpha.add(&b.scale(u));
            let pdf = eta.compose_unary(pdf_stack(eta.value()));
            density = density.add(&pdf.scale(w));
            moment = moment.add(&pdf.scale(w * u));
        }
        moment.mul(&density.compose_unary(reciprocal_stack(density.value()))).neg()
    }

    fn owned(nodes: Vec<f64>, weights: Vec<f64>) -> AnchorGridOwned {
        AnchorGridOwned::new(nodes, weights)
    }

    fn marginal(alpha: f64, b: f64, grid: AnchorGrid<'_>) -> f64 {
        grid.nodes
            .iter()
            .zip(grid.weights.iter())
            .map(|(&u, &w)| w * normal_cdf(-(alpha + b * u)))
            .sum()
    }

    #[test]
    fn gauss_hermite_law_integrates_moments() {
        for m in [2usize, 5, 16, 33, 65] {
            let (nodes, weights) = gauss_hermite_probabilists(m).expect("grid");
            assert_eq!(nodes.len(), m);
            let total: f64 = weights.iter().sum();
            assert!((total - 1.0).abs() < 1e-12, "m={m} total={total}");
            for k in 1..m {
                assert!(nodes[k] > nodes[k - 1], "m={m} nodes must ascend");
            }
            let mean: f64 = nodes.iter().zip(&weights).map(|(u, w)| u * w).sum();
            let var: f64 = nodes.iter().zip(&weights).map(|(u, w)| u * u * w).sum();
            assert!(mean.abs() < 1e-12, "m={m} mean={mean}");
            assert!((var - 1.0).abs() < 1e-10, "m={m} var={var}");
            if m >= 4 {
                let fourth: f64 = nodes.iter().zip(&weights).map(|(u, w)| u.powi(4) * w).sum();
                assert!((fourth - 3.0).abs() < 1e-9, "m={m} fourth={fourth}");
            }
        }
    }

    /// The anchoring equation is solved: the anchored marginal survival is
    /// `Φ(−q)` to the log-residual tolerance, across both tails and slopes of
    /// both signs, on a law that is nothing like Gaussian.
    #[test]
    fn anchor_solves_the_anchoring_equation_on_a_skewed_law() {
        let grid = skewed_grid();
        for &q in &[-3.5, -1.2, -0.3, 0.0, 0.4, 1.7, 3.2, 5.5] {
            for &b in &[-1.3, -0.4, 0.0, 0.25, 0.9, 2.1] {
                let alpha = solve_anchor(q, b, grid.view()).expect("anchor");
                let lhs = marginal(alpha, b, grid.view());
                let rhs = normal_cdf(-q);
                let smaller = rhs.min(1.0 - rhs);
                assert!(
                    (lhs - rhs).abs() <= 1e-10 * smaller.max(1e-300),
                    "q={q} b={b}: Σ w Φ(−η) = {lhs:e} vs Φ(−q) = {rhs:e}"
                );
            }
        }
    }

    /// Gaussian is the special case: on a Gauss–Hermite law the anchor is the
    /// closed form `q·√(1 + b²)` to quadrature tolerance, and its derivatives
    /// are the closed form's.
    #[test]
    fn gaussian_grid_reproduces_the_closed_form() {
        let (nodes, weights) = gauss_hermite_probabilists(65).expect("grid");
        let grid = owned(nodes, weights);
        for &q in &[-2.5, -0.7, 0.0, 0.6, 2.2] {
            for &b in &[-1.1, -0.3, 0.0, 0.45, 1.6] {
                let d = AnchorDerivatives::solve(q, b, grid.view()).expect("anchor");
                let c = (1.0 + b * b).sqrt();
                assert!((d.alpha - q * c).abs() < 1e-9, "q={q} b={b}: α={} vs {}", d.alpha, q * c);
                assert!((d.a_q - c).abs() < 1e-8, "q={q} b={b}: α_q={} vs {c}", d.a_q);
                assert!((d.a_b - q * b / c).abs() < 1e-8, "q={q} b={b}: α_b={} vs {}", d.a_b, q * b / c);
                assert!(d.a_qq.abs() < 1e-8, "q={q} b={b}: α_qq={}", d.a_qq);
                assert!((d.a_qb - b / c).abs() < 1e-7, "q={q} b={b}: α_qb={} vs {}", d.a_qb, b / c);
                assert!((d.a_bb - q / (c * c * c)).abs() < 1e-7, "q={q} b={b}: α_bb={} vs {}", d.a_bb, q / c.powi(3));
            }
        }
    }

    /// Every closed-form implicit derivative through order three matches a
    /// central difference of the solved anchor on the skewed law.
    #[test]
    fn implicit_derivatives_match_finite_differences() {
        let grid = skewed_grid();
        let solve = |q: f64, b: f64| solve_anchor(q, b, grid.view()).expect("anchor");
        // Central differences carry an `h²` truncation term; at `1e-4` it sits
        // below the tolerances while round-off on the closed-form derivatives
        // (each exact to ~1e-13) stays near `1e-9`.
        let h = 1e-4;
        for &q in &[-1.4, -0.2, 0.5, 1.9] {
            for &b in &[-0.8, 0.1, 0.7, 1.5] {
                let d = AnchorDerivatives::solve(q, b, grid.view()).expect("derivatives");
                let f = |dq: f64, db: f64| solve(q + dq, b + db);
                // First order: five-point stencils.
                let fd_q = (-f(2.0 * h, 0.0) + 8.0 * f(h, 0.0) - 8.0 * f(-h, 0.0) + f(-2.0 * h, 0.0)) / (12.0 * h);
                let fd_b = (-f(0.0, 2.0 * h) + 8.0 * f(0.0, h) - 8.0 * f(0.0, -h) + f(0.0, -2.0 * h)) / (12.0 * h);
                assert!((d.a_q - fd_q).abs() < 1e-8 * (1.0 + fd_q.abs()), "q={q} b={b}: α_q {} vs fd {fd_q}", d.a_q);
                assert!((d.a_b - fd_b).abs() < 1e-8 * (1.0 + fd_b.abs()), "q={q} b={b}: α_b {} vs fd {fd_b}", d.a_b);
                // Second order from the closed-form first derivatives.
                let dq = |dq: f64, db: f64| AnchorDerivatives::solve(q + dq, b + db, grid.view()).expect("d");
                let fd_qq = (dq(h, 0.0).a_q - dq(-h, 0.0).a_q) / (2.0 * h);
                let fd_qb = (dq(0.0, h).a_q - dq(0.0, -h).a_q) / (2.0 * h);
                let fd_bq = (dq(h, 0.0).a_b - dq(-h, 0.0).a_b) / (2.0 * h);
                let fd_bb = (dq(0.0, h).a_b - dq(0.0, -h).a_b) / (2.0 * h);
                assert!((d.a_qq - fd_qq).abs() < 1e-6 * (1.0 + fd_qq.abs()), "q={q} b={b}: α_qq {} vs fd {fd_qq}", d.a_qq);
                assert!((d.a_qb - fd_qb).abs() < 1e-6 * (1.0 + fd_qb.abs()), "q={q} b={b}: α_qb {} vs fd {fd_qb}", d.a_qb);
                assert!((d.a_qb - fd_bq).abs() < 1e-6 * (1.0 + fd_bq.abs()), "q={q} b={b}: α_qb {} vs fd(b,q) {fd_bq}", d.a_qb);
                assert!((d.a_bb - fd_bb).abs() < 1e-6 * (1.0 + fd_bb.abs()), "q={q} b={b}: α_bb {} vs fd {fd_bb}", d.a_bb);
                // Third order from the closed-form second derivatives.
                let fd_qqq = (dq(h, 0.0).a_qq - dq(-h, 0.0).a_qq) / (2.0 * h);
                let fd_qqb = (dq(0.0, h).a_qq - dq(0.0, -h).a_qq) / (2.0 * h);
                let fd_qbb = (dq(0.0, h).a_qb - dq(0.0, -h).a_qb) / (2.0 * h);
                let fd_bbb = (dq(0.0, h).a_bb - dq(0.0, -h).a_bb) / (2.0 * h);
                let fd_qbq = (dq(h, 0.0).a_qb - dq(-h, 0.0).a_qb) / (2.0 * h);
                assert!((d.a_qqq - fd_qqq).abs() < 1e-5 * (1.0 + fd_qqq.abs()), "q={q} b={b}: α_qqq {} vs fd {fd_qqq}", d.a_qqq);
                assert!((d.a_qqb - fd_qqb).abs() < 1e-5 * (1.0 + fd_qqb.abs()), "q={q} b={b}: α_qqb {} vs fd {fd_qqb}", d.a_qqb);
                assert!((d.a_qqb - fd_qbq).abs() < 1e-5 * (1.0 + fd_qbq.abs()), "q={q} b={b}: α_qqb {} vs fd(qb,q) {fd_qbq}", d.a_qqb);
                assert!((d.a_qbb - fd_qbb).abs() < 1e-5 * (1.0 + fd_qbb.abs()), "q={q} b={b}: α_qbb {} vs fd {fd_qbb}", d.a_qbb);
                assert!((d.a_bbb - fd_bbb).abs() < 1e-5 * (1.0 + fd_bbb.abs()), "q={q} b={b}: α_bbb {} vs fd {fd_bbb}", d.a_bbb);
            }
        }
    }

    /// The jet lift carries the same derivatives as the closed form through
    /// order two (`Order2<2>`) and through order four (`Tower4<2>`), and its
    /// value channel is bitwise the solver's root.
    #[test]
    fn anchor_jet_matches_closed_form_and_finite_differences() {
        let grid = skewed_grid();
        for &q in &[-1.1, 0.3, 1.6] {
            for &b in &[-0.6, 0.2, 1.2] {
                let d = AnchorDerivatives::solve(q, b, grid.view()).expect("derivatives");
                let qj = Order2::<2>::variable(q, 0);
                let bj = Order2::<2>::variable(b, 1);
                let aj = anchor_jet(d.alpha, &qj, &bj, grid.view());
                assert_eq!(aj.value().to_bits(), d.alpha.to_bits(), "value channel is the root");
                let tower = aj.0;
                assert!((tower.g[0] - d.a_q).abs() < 1e-10 * (1.0 + d.a_q.abs()), "jet α_q {} vs {}", tower.g[0], d.a_q);
                assert!((tower.g[1] - d.a_b).abs() < 1e-10 * (1.0 + d.a_b.abs()), "jet α_b {} vs {}", tower.g[1], d.a_b);
                assert!((tower.h[0][0] - d.a_qq).abs() < 1e-9 * (1.0 + d.a_qq.abs()), "jet α_qq {} vs {}", tower.h[0][0], d.a_qq);
                assert!((tower.h[0][1] - d.a_qb).abs() < 1e-9 * (1.0 + d.a_qb.abs()), "jet α_qb {} vs {}", tower.h[0][1], d.a_qb);
                assert!((tower.h[1][1] - d.a_bb).abs() < 1e-9 * (1.0 + d.a_bb.abs()), "jet α_bb {} vs {}", tower.h[1][1], d.a_bb);

                let qt = Tower4::<2>::variable(q, 0);
                let bt = Tower4::<2>::variable(b, 1);
                let at = anchor_jet(d.alpha, &qt, &bt, grid.view());
                assert!((at.t3[0][0][0] - d.a_qqq).abs() < 1e-8 * (1.0 + d.a_qqq.abs()), "tower α_qqq {} vs {}", at.t3[0][0][0], d.a_qqq);
                assert!((at.t3[0][0][1] - d.a_qqb).abs() < 1e-8 * (1.0 + d.a_qqb.abs()), "tower α_qqb {} vs {}", at.t3[0][0][1], d.a_qqb);
                assert!((at.t3[0][1][1] - d.a_qbb).abs() < 1e-8 * (1.0 + d.a_qbb.abs()), "tower α_qbb {} vs {}", at.t3[0][1][1], d.a_qbb);
                assert!((at.t3[1][1][1] - d.a_bbb).abs() < 1e-8 * (1.0 + d.a_bbb.abs()), "tower α_bbb {} vs {}", at.t3[1][1][1], d.a_bbb);
                // Fourth order against a central difference of the third.
                let h = 1e-4;
                let third = |dq: f64, db: f64| AnchorDerivatives::solve(q + dq, b + db, grid.view()).expect("d");
                let fd_qqqq = (third(h, 0.0).a_qqq - third(-h, 0.0).a_qqq) / (2.0 * h);
                let fd_qqqb = (third(0.0, h).a_qqq - third(0.0, -h).a_qqq) / (2.0 * h);
                let fd_qqbb = (third(0.0, h).a_qqb - third(0.0, -h).a_qqb) / (2.0 * h);
                let fd_qbbb = (third(0.0, h).a_qbb - third(0.0, -h).a_qbb) / (2.0 * h);
                let fd_bbbb = (third(0.0, h).a_bbb - third(0.0, -h).a_bbb) / (2.0 * h);
                assert!((at.t4[0][0][0][0] - fd_qqqq).abs() < 1e-5 * (1.0 + fd_qqqq.abs()), "tower α_qqqq {} vs fd {fd_qqqq}", at.t4[0][0][0][0]);
                assert!((at.t4[0][0][0][1] - fd_qqqb).abs() < 1e-5 * (1.0 + fd_qqqb.abs()), "tower α_qqqb {} vs fd {fd_qqqb}", at.t4[0][0][0][1]);
                assert!((at.t4[0][0][1][1] - fd_qqbb).abs() < 1e-5 * (1.0 + fd_qqbb.abs()), "tower α_qqbb {} vs fd {fd_qqbb}", at.t4[0][0][1][1]);
                assert!((at.t4[0][1][1][1] - fd_qbbb).abs() < 1e-5 * (1.0 + fd_qbbb.abs()), "tower α_qbbb {} vs fd {fd_qbbb}", at.t4[0][1][1][1]);
                assert!((at.t4[1][1][1][1] - fd_bbbb).abs() < 1e-5 * (1.0 + fd_bbbb.abs()), "tower α_bbbb {} vs fd {fd_bbbb}", at.t4[1][1][1][1]);

                // The q- and b-derivative jets agree with the closed form through order two.
                let aq = anchor_q_derivative_jet(&aj, &qj, &bj, grid.view()).0;
                assert!((aq.v - d.a_q).abs() < 1e-10 * (1.0 + d.a_q.abs()));
                assert!((aq.g[0] - d.a_qq).abs() < 1e-9 * (1.0 + d.a_qq.abs()), "α_q jet g[q] {} vs {}", aq.g[0], d.a_qq);
                assert!((aq.g[1] - d.a_qb).abs() < 1e-9 * (1.0 + d.a_qb.abs()), "α_q jet g[b] {} vs {}", aq.g[1], d.a_qb);
                assert!((aq.h[0][0] - d.a_qqq).abs() < 1e-8 * (1.0 + d.a_qqq.abs()), "α_q jet h[qq] {} vs {}", aq.h[0][0], d.a_qqq);
                assert!((aq.h[0][1] - d.a_qqb).abs() < 1e-8 * (1.0 + d.a_qqb.abs()), "α_q jet h[qb] {} vs {}", aq.h[0][1], d.a_qqb);
                assert!((aq.h[1][1] - d.a_qbb).abs() < 1e-8 * (1.0 + d.a_qbb.abs()), "α_q jet h[bb] {} vs {}", aq.h[1][1], d.a_qbb);
                let ab = anchor_b_derivative_jet(&aj, &qj, &bj, grid.view()).0;
                assert!((ab.v - d.a_b).abs() < 1e-10 * (1.0 + d.a_b.abs()));
                assert!((ab.g[0] - d.a_qb).abs() < 1e-9 * (1.0 + d.a_qb.abs()), "α_b jet g[q] {} vs {}", ab.g[0], d.a_qb);
                assert!((ab.g[1] - d.a_bb).abs() < 1e-9 * (1.0 + d.a_bb.abs()), "α_b jet g[b] {} vs {}", ab.g[1], d.a_bb);
                assert!((ab.h[1][1] - d.a_bbb).abs() < 1e-8 * (1.0 + d.a_bbb.abs()), "α_b jet h[bb] {} vs {}", ab.h[1][1], d.a_bbb);
            }
        }
    }

    /// Deep tails: the log-space residual keeps the solve finite and exact
    /// where the linear-space marginal has long underflowed.
    #[test]
    fn anchor_survives_the_deep_tails() {
        let grid = skewed_grid();
        for &q in &[-7.5, 7.5] {
            for &b in &[-1.5, 0.0, 1.5] {
                let alpha = solve_anchor(q, b, grid.view()).expect("anchor");
                assert!(alpha.is_finite());
                // On the survival side the anchored survival must equal Φ(−q)
                // relatively; on the complement side the anchored failure
                // probability must equal Φ(q) relatively.
                let survival = marginal(alpha, b, grid.view());
                if q >= 0.0 {
                    let target = normal_cdf(-q);
                    assert!(((survival - target) / target).abs() < 1e-9, "q={q} b={b}: {survival:e} vs {target:e}");
                } else {
                    let failure: f64 = grid
                        .nodes
                        .iter()
                        .zip(grid.weights.iter())
                        .map(|(&u, &w)| w * normal_cdf(alpha + b * u))
                        .sum();
                    let target = normal_cdf(q);
                    assert!(((failure - target) / target).abs() < 1e-9, "q={q} b={b}: {failure:e} vs {target:e}");
                }
            }
        }
    }
}
