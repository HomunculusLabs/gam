//! Gaussian expectations of elementwise activations, with analytic derivatives.
//!
//! A known MLP block `F(z) = Σ_j u_j σ(b_j + w_jᵀ z)` under a declared Gaussian
//! intervention law reads every unit through a Gaussian pre-activation (#2946).
//! The best response of a retained subspace smooths each unit,
//! `T_v σ(t) = E σ(t + √v E)` with `E ~ N(0, 1)`, and the variance it explains
//! pairs units through `K_σ = E[σ(X) σ(Y)]` for jointly Gaussian `X`, `Y`.
//! This module owns both expectations for ReLU and for the exact GELU
//! `σ(t) = t Φ(t)`: the smoothing with its t-derivatives of every order, and
//! the pair kernel with Price's covariance derivative
//! `∂_r K_σ = E[σ'(X) σ'(Y)]`. Every value is a closed form in `Φ`, `φ` and
//! elementary functions; nothing here is a quadrature.
//!
//! The approximate GELUs (`gelu_new`, `gelu_pytorch_tanh`) are different
//! activations and have no primitive here.
//!
//! # Smoothing
//!
//! ReLU with `v > 0`, `s = √v`, `u = t/s`:
//! `T = t Φ(u) + s φ(u)`, `T' = Φ(u)`, and for `k ≥ 2`
//! `T⁽ᵏ⁾ = φ⁽ᵏ⁻²⁾(u)/sᵏ⁻¹ = (−1)ᵏ He_{k−2}(u) φ(u)/sᵏ⁻¹`.
//!
//! Exact GELU with `A = 1 + v`, `S = √A`, `u = t/S`. For `X ~ N(t, v)` and an
//! independent `N ~ N(0, 1)`, `E Φ(X) = P(N ≤ X) = Φ(u)`, and Stein's lemma
//! `E[(X − t) g(X)] = v E g'(X)` gives
//! `T = t E Φ(X) + v E φ(X) = t Φ(u) + (v/S) φ(u)`.
//! Differentiating under the expectation, `T'' = E σ''(X)` with
//! `σ''(x) = (2 − x²) φ(x)`, and `φ(x) N(x; t, v) = N(t; 0, A) N(x; t/A, v/A)`
//! gives `T'' = (A φ(u) − φ''(u))/(A S)`. Hence, for `k ≥ 1`,
//! `T⁽ᵏ⁾ = (A φ⁽ᵏ⁻²⁾(u) − φ⁽ᵏ⁾(u))/(A Sᵏ⁻¹) = (−1)ᵏ φ(u) (A He_{k−2}(u) − He_k(u))/(A Sᵏ⁻¹)`,
//! where `k = 1` reads `φ⁽⁻¹⁾ = Φ`: `T' = Φ(u) + u φ(u)/A`.
//!
//! Both smoothings solve the heat equation `∂_v T = ½ ∂_t² T`.
//!
//! Left of the origin (`u < 0`) the values and slopes are formed from the
//! log-CDF slope `λ(u) = φ(u)/Φ(u)` and its correction `q(u) = λ(u) + u`, as
//! `Φ(u) = φ(u)/λ` and `1 + u/λ = q/λ`: ReLU `T = s φ(u) q/λ`, `T' = φ(u)/λ`;
//! exact GELU `T = S φ(u) (q/λ − 1/A)`, `T' = φ(u) (1/λ + u/A)`. There
//! `t Φ(u) + s φ(u)` cancels like `u²`, and `Φ(u) = ½ erfc(−u/√2)` multiplies
//! the rounding of its argument by `u²`; `λ` and `q = −f''/λ` (`f = ln Φ`, so
//! `f'' = −λ q`) come from the log-CDF owner at full relative precision.
//!
//! # Pair kernel
//!
//! `X ~ N(b, v)`, `Y ~ N(c, w)`, `Cov(X, Y) = r`.
//!
//! Exact GELU, with `A = 1 + v`, `B = 1 + w`, `Δ = AB − r²`. Write
//! `Φ(X) = P(N₁ ≤ X)` and `Φ(Y) = P(N₂ ≤ Y)` for independent standard normals,
//! so `K = E[X Y h(U, V)]` with `U = X − N₁`, `V = Y − N₂` and
//! `h = 1{U ≥ 0} 1{V ≥ 0}`. `(U, V)` has means `(b, c)`, variances `(A, B)` and
//! covariance `r`, so `H(b, c) = E h = Φ₂(b/√A, c/√B; r/√(AB))`. Second-order
//! Gaussian integration by parts gives
//! `K = (bc + r) H + (cv + br) H_b + (bw + cr) H_c + vr H_bb + (vw + r²) H_bc + rw H_cc`,
//! and `H_bb = −(b H_b + r H_bc)/A`, `H_cc = −(c H_c + r H_bc)/B` reduce it to
//! `K = (bc + r) H + (cv + br/A) H_b + (bw + cr/B) H_c + (vw + r² q) H_bc`, with
//! `q = 1 − v/A − w/B` and `H_bc = exp(−(B b² − 2rbc + A c²)/(2Δ))/(2π √Δ)`.
//! Price's theorem (`∂_r H = H_bc`, `∂_r H_b = −b̃ H_bc`, `∂_r H_c = −c̃ H_bc`,
//! `∂_r H_bc = (r/Δ + b̃ c̃) H_bc`, with `b̃ = (Bb − rc)/Δ`, `c̃ = (Ac − rb)/Δ`)
//! differentiates it to
//! `∂_r K = H + (b H_b + r H_bc)/A + (c H_c + r H_bc)/B + (r/Δ + b̃ c̃) H_bc`.
//! At zero mean `H = ¼ + arcsin(r/√(AB))/(2π) = arccos(−r/√(AB))/(2π)`, so
//! `K = r H + (vw + r² q)/(2π √Δ)` and `∂_r K = H + r (1/A + 1/B + 1/Δ)/(2π √Δ)`.
//!
//! ReLU by the same integration by parts with `h = 1{X ≥ 0} 1{Y ≥ 0}`:
//! `K = (bc + r) H + cv H_b + bw H_c + (vw − r²) H_bc` with
//! `H = Φ₂(b/√v, c/√w; r/√(vw))`, and `∂_r K = H`. At zero mean, with
//! `ρ = r/√(vw) = cos θ`, this is the arc-cosine kernel
//! `K = √(vw) (sin θ + (π − θ) cos θ)/(2π)` with `∂_r K = (π − θ)/(2π)`.
//!
//! The biased kernels need the bivariate normal CDF `Φ₂`, whose owner is moving
//! into this crate; [`pair_kernel`] refuses nonzero means until it has.

use crate::probability::{normal_cdf, normal_logcdf_derivatives, normal_pdf};
use std::f64::consts::TAU;
use std::fmt;

/// An elementwise activation whose Gaussian expectations are closed forms.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GaussianActivation {
    /// `σ(t) = max(t, 0)`.
    Relu,
    /// The exact GELU `σ(t) = t Φ(t)`, not its `tanh` or sigmoid approximations.
    ExactGelu,
}

/// A refused Gaussian activation expectation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GaussianActivationError {
    /// A pre-activation variance must be finite and nonnegative.
    InvalidVariance { variance: f64 },
    /// A mean, covariance or evaluation point must be finite.
    NonFiniteArgument { value: f64 },
    /// ReLU with zero smoothing variance has no t-derivative of order ≥ 2 at
    /// its kink `t = 0`: the second derivative there is a Dirac mass.
    UnsmoothedReluKink { order: usize },
    /// A biased pair kernel needs the bivariate normal CDF `Φ₂`.
    BiasedPairNeedsBivariateNormalCdf { mean_x: f64, mean_y: f64 },
}

impl fmt::Display for GaussianActivationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidVariance { variance } => write!(
                formatter,
                "a Gaussian pre-activation variance must be finite and nonnegative, got {variance}"
            ),
            Self::NonFiniteArgument { value } => write!(
                formatter,
                "a Gaussian pre-activation mean, covariance or evaluation point must be finite, got {value}"
            ),
            Self::UnsmoothedReluKink { order } => write!(
                formatter,
                "ReLU without smoothing variance has no t-derivative of order {order} at its kink t = 0"
            ),
            Self::BiasedPairNeedsBivariateNormalCdf { mean_x, mean_y } => write!(
                formatter,
                "the pair kernel at means ({mean_x}, {mean_y}) needs the bivariate normal CDF, whose owner has not moved into gam-math"
            ),
        }
    }
}

impl std::error::Error for GaussianActivationError {}

/// Fills `derivatives[k]` with `∂ᵏ/∂tᵏ T_v σ(t)`, `T_v σ(t) = E σ(t + √v E)`,
/// for every `k < derivatives.len()`. `T_v σ⁽ᵏ⁾(t) = E σ⁽ᵏ⁾(t + √v E)` is also
/// the Gaussian mean of the k-th derivative of the activation.
///
/// With zero variance ReLU returns the activation itself and the Gaussian
/// limit of its slope, `lim_{v→0} Φ(t/√v)`, which is `½` at the kink. Its
/// higher derivatives are zero away from the kink and refused at it. Once
/// `φ(u)` underflows, the derivatives of order ≥ 2, each `φ(u)` times a
/// polynomial in `u`, are returned as the zero it underflows to.
pub fn gaussian_smoothing_derivatives(
    activation: GaussianActivation,
    t: f64,
    variance: f64,
    derivatives: &mut [f64],
) -> Result<(), GaussianActivationError> {
    validate_finite(t)?;
    validate_variance(variance)?;
    match activation {
        GaussianActivation::Relu => relu_smoothing(t, variance, derivatives),
        GaussianActivation::ExactGelu => {
            exact_gelu_smoothing(t, variance, derivatives);
            Ok(())
        }
    }
}

/// The joint law of two Gaussian pre-activations: `X ~ N(mean_x, variance_x)`,
/// `Y ~ N(mean_y, variance_y)` and `Cov(X, Y) = covariance`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PreactivationPair {
    pub mean_x: f64,
    pub mean_y: f64,
    pub variance_x: f64,
    pub variance_y: f64,
    pub covariance: f64,
}

/// `K = E[σ(X) σ(Y)]` and its covariance derivative.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PairKernel {
    /// `K = E[σ(X) σ(Y)]`.
    pub value: f64,
    /// `∂K/∂r = E[σ'(X) σ'(Y)]` at fixed means and variances (Price's theorem).
    pub covariance_derivative: f64,
}

/// The pair kernel `K_σ(b, c; v, w, r) = E[σ(X) σ(Y)]` with `∂_r K_σ`.
///
/// The kernel is evaluated at the metric projection of the covariance onto
/// the Cauchy-Schwarz interval `|r| ≤ √(vw)`. A Gram matrix computed in
/// floating point leaves that interval only by its rounding, and a reader
/// lying in a retained subspace puts `r` on its boundary, so a refusal there
/// would refuse exact subspaces. The projection moves `K` by at most
/// `sup|σ'|² (|r| − √(vw))`, because `|∂_r K| = |E σ'(X) σ'(Y)| ≤ sup|σ'|²`:
/// `1` for ReLU and `(Φ(√2) + √2 φ(√2))² < 1.28` for the exact GELU.
///
/// A zero-variance pre-activation is constant. For ReLU at zero mean it sits
/// on the kink, where the slope is its Gaussian limit `½`, as in
/// [`gaussian_smoothing_derivatives`].
pub fn pair_kernel(
    activation: GaussianActivation,
    pair: PreactivationPair,
) -> Result<PairKernel, GaussianActivationError> {
    validate_finite(pair.mean_x)?;
    validate_finite(pair.mean_y)?;
    validate_variance(pair.variance_x)?;
    validate_variance(pair.variance_y)?;
    validate_finite(pair.covariance)?;
    if pair.mean_x != 0.0 || pair.mean_y != 0.0 {
        return Err(GaussianActivationError::BiasedPairNeedsBivariateNormalCdf {
            mean_x: pair.mean_x,
            mean_y: pair.mean_y,
        });
    }
    let law = ProjectedCovariance::new(pair.variance_x, pair.variance_y, pair.covariance);
    Ok(match activation {
        GaussianActivation::Relu => relu_zero_mean_pair_kernel(&law),
        GaussianActivation::ExactGelu => exact_gelu_zero_mean_pair_kernel(&law),
    })
}

fn validate_finite(value: f64) -> Result<(), GaussianActivationError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(GaussianActivationError::NonFiniteArgument { value })
    }
}

fn validate_variance(variance: f64) -> Result<(), GaussianActivationError> {
    if variance.is_finite() && variance >= 0.0 {
        Ok(())
    } else {
        Err(GaussianActivationError::InvalidVariance { variance })
    }
}

fn relu_smoothing(
    t: f64,
    variance: f64,
    derivatives: &mut [f64],
) -> Result<(), GaussianActivationError> {
    if variance == 0.0 {
        if t == 0.0 && derivatives.len() > 2 {
            return Err(GaussianActivationError::UnsmoothedReluKink { order: 2 });
        }
        for (order, slot) in derivatives.iter_mut().enumerate() {
            *slot = match order {
                0 => t.max(0.0),
                1 if t > 0.0 => 1.0,
                1 if t < 0.0 => 0.0,
                1 => 0.5,
                _ => 0.0,
            };
        }
        return Ok(());
    }
    let scale = variance.sqrt();
    let u = t / scale;
    let density = normal_pdf(u);
    let (value, probability) = if u < 0.0 {
        let (reciprocal_slope, correction) = left_tail_mills(u);
        (scale * density * correction, density * reciprocal_slope)
    } else {
        let probability = normal_cdf(u);
        (t * probability + scale * density, probability)
    };
    if let Some(slot) = derivatives.get_mut(0) {
        *slot = value;
    }
    if let Some(slot) = derivatives.get_mut(1) {
        *slot = probability;
    }
    fill_relu_curvature(derivatives, u, density, scale);
    Ok(())
}

/// `(1/λ(u), q(u)/λ(u))` for `u < 0`, with the log-CDF slope `λ = φ/Φ` and its
/// correction `q = λ + u`, read from `f'' = −λ q` of the log-CDF owner. Both
/// carry full relative precision however deep the tail, so `Φ(u) = φ(u)/λ` and
/// `1 + u/λ = q/λ` need no subtraction.
fn left_tail_mills(u: f64) -> (f64, f64) {
    let log_cdf_derivatives = normal_logcdf_derivatives(u);
    let reciprocal_slope = log_cdf_derivatives[1].recip();
    (
        reciprocal_slope,
        -log_cdf_derivatives[2] * reciprocal_slope * reciprocal_slope,
    )
}

/// `derivatives[k] = (−1)ᵏ He_{k−2}(u) φ(u)/sᵏ⁻¹` for `k ≥ 2`.
fn fill_relu_curvature(derivatives: &mut [f64], u: f64, density: f64, scale: f64) {
    if density == 0.0 {
        derivatives.iter_mut().skip(2).for_each(|slot| *slot = 0.0);
        return;
    }
    let mut hermite_previous = 0.0;
    let mut hermite_current = 1.0;
    let mut factor = density / scale;
    for (order, slot) in derivatives.iter_mut().enumerate().skip(2) {
        *slot = factor * hermite_current;
        let degree = (order - 2) as f64;
        let hermite_next = u * hermite_current - degree * hermite_previous;
        hermite_previous = hermite_current;
        hermite_current = hermite_next;
        factor = -factor / scale;
    }
}

fn exact_gelu_smoothing(t: f64, variance: f64, derivatives: &mut [f64]) {
    let total = 1.0 + variance;
    let root_total = total.sqrt();
    let u = t / root_total;
    let density = normal_pdf(u);
    let (value, slope) = if u < 0.0 {
        let (reciprocal_slope, correction) = left_tail_mills(u);
        (
            root_total * density * (correction - total.recip()),
            density * (reciprocal_slope + u / total),
        )
    } else {
        let probability = normal_cdf(u);
        (
            t * probability + variance / root_total * density,
            probability + u * density / total,
        )
    };
    if let Some(slot) = derivatives.get_mut(0) {
        *slot = value;
    }
    if let Some(slot) = derivatives.get_mut(1) {
        *slot = slope;
    }
    fill_exact_gelu_curvature(derivatives, u, density, total, root_total);
}

/// `derivatives[k] = (−1)ᵏ φ(u) (A He_{k−2}(u) − He_k(u))/(A Sᵏ⁻¹)` for `k ≥ 2`.
fn fill_exact_gelu_curvature(
    derivatives: &mut [f64],
    u: f64,
    density: f64,
    total: f64,
    root_total: f64,
) {
    if density == 0.0 {
        derivatives.iter_mut().skip(2).for_each(|slot| *slot = 0.0);
        return;
    }
    let mut hermite_lower = 1.0;
    let mut hermite_middle = u;
    let mut hermite_upper = u.mul_add(u, -1.0);
    let mut factor = density / (total * root_total);
    for (order, slot) in derivatives.iter_mut().enumerate().skip(2) {
        *slot = factor * (total * hermite_lower - hermite_upper);
        let hermite_next = u * hermite_upper - order as f64 * hermite_middle;
        hermite_lower = hermite_middle;
        hermite_middle = hermite_upper;
        hermite_upper = hermite_next;
        factor = -factor / root_total;
    }
}

/// A covariance projected onto the Cauchy-Schwarz interval of its variances.
struct ProjectedCovariance {
    variance_x: f64,
    variance_y: f64,
    /// `√v √w`.
    scale: f64,
    /// `r` projected onto `[−√(vw), √(vw)]`.
    covariance: f64,
    /// `r/√(vw)`, and `0` when a variance is zero.
    correlation: f64,
    /// `vw − r² ≥ 0`, with both squares carried exactly.
    determinant: f64,
}

impl ProjectedCovariance {
    fn new(variance_x: f64, variance_y: f64, covariance: f64) -> Self {
        let scale = variance_x.sqrt() * variance_y.sqrt();
        if scale == 0.0 {
            return Self {
                variance_x,
                variance_y,
                scale,
                covariance: 0.0,
                correlation: 0.0,
                determinant: 0.0,
            };
        }
        let ratio = covariance / scale;
        if ratio.abs() > 1.0 {
            let correlation = ratio.signum();
            return Self {
                variance_x,
                variance_y,
                scale,
                covariance: correlation * scale,
                correlation,
                determinant: 0.0,
            };
        }
        let product = variance_x * variance_y;
        let square = covariance * covariance;
        let determinant = (product - square)
            + (variance_x.mul_add(variance_y, -product) - covariance.mul_add(covariance, -square));
        Self {
            variance_x,
            variance_y,
            scale,
            covariance,
            correlation: ratio,
            determinant: determinant.max(0.0),
        }
    }
}

/// `K = √(vw) (sin θ + ρ (π − θ))/(2π)` and `∂_r K = (π − θ)/(2π)`, `ρ = cos θ`.
///
/// `sin θ` is formed from the same rounded `ρ` as `arccos(−ρ)`, so as `ρ → −1`
/// the rounding of `ρ` cancels between the two terms to first order.
fn relu_zero_mean_pair_kernel(law: &ProjectedCovariance) -> PairKernel {
    if law.scale == 0.0 {
        return PairKernel {
            value: 0.0,
            covariance_derivative: 0.25,
        };
    }
    let correlation = law.correlation;
    let orthant = (-correlation).acos() / TAU;
    let sine = ((1.0 - correlation) * (1.0 + correlation)).sqrt();
    PairKernel {
        value: law.scale * (sine / TAU + correlation * orthant),
        covariance_derivative: orthant,
    }
}

/// `K = r H + (vw + r² q)/(2π √Δ)` and `∂_r K = H + r (1/A + 1/B + 1/Δ)/(2π √Δ)`
/// with `H = arccos(−r/√(AB))/(2π)`. `vw + r² q = (vw − r²) + r² (1/A + 1/B)`
/// sums two nonnegative terms.
fn exact_gelu_zero_mean_pair_kernel(law: &ProjectedCovariance) -> PairKernel {
    let total_x = 1.0 + law.variance_x;
    let total_y = 1.0 + law.variance_y;
    let covariance = law.covariance;
    let discriminant = 1.0 + law.variance_x + law.variance_y + law.determinant;
    let correlation = (covariance / (total_x.sqrt() * total_y.sqrt())).clamp(-1.0, 1.0);
    let orthant = (-correlation).acos() / TAU;
    let density = 1.0 / (TAU * discriminant.sqrt());
    let inverse_totals = total_x.recip() + total_y.recip();
    let coupling = law.determinant + covariance * covariance * inverse_totals;
    PairKernel {
        value: covariance * orthant + coupling * density,
        covariance_derivative: orthant
            + covariance * density * (inverse_totals + discriminant.recip()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quadrature::{
        GaussHermiteRule, gauss_hermite_rule, symmetric_tridiagonal_eigen_first_components,
    };
    use std::f64::consts::{PI, SQRT_2};

    /// A bound on the floating-point operations one closed form here, or one
    /// integrand evaluation, chains: none takes more than 64 steps, and each
    /// adds at most one ulp of relative error to first order.
    const EVALUATION_OPERATIONS: f64 = 64.0;

    /// A quadrature estimate and a derived bound on its error.
    #[derive(Clone, Copy, Debug)]
    struct Reference {
        value: f64,
        bound: f64,
    }

    /// One quadrature pass: the sum and the sum of its summands' magnitudes.
    #[derive(Clone, Copy, Debug)]
    struct Pass {
        sum: f64,
        absolute: f64,
    }

    /// A rule checked against its doubled order. `|Q_n − Q_2n|` bounds
    /// `|I − Q_2n|` once the rule converges geometrically, which the analytic
    /// integrands here guarantee. Recursive summation of `summands` terms errs
    /// by at most `(summands − 1) ε Σ|term|` (Higham, ASNA §4.2), Golub-Welsch
    /// weights carry `O(nodes ε)` relative error, and every summand's
    /// evaluation adds at most `EVALUATION_OPERATIONS ε` relative.
    fn doubled_order(
        coarse: Pass,
        fine: Pass,
        summands: usize,
        nodes: usize,
        truncation: f64,
    ) -> Reference {
        let rounding = (summands as f64 + nodes as f64 + EVALUATION_OPERATIONS)
            * f64::EPSILON
            * fine.absolute;
        Reference {
            value: fine.sum,
            bound: (coarse.sum - fine.sum).abs() + rounding + truncation,
        }
    }

    /// First-order rounding of a closed form whose terms add up to `magnitude`.
    fn closed_form_rounding(magnitude: f64) -> f64 {
        EVALUATION_OPERATIONS * f64::EPSILON * magnitude
    }

    /// The Gauss-Legendre rule on `[−1, 1]` by Golub-Welsch.
    fn legendre_rule(node_count: usize) -> (Vec<f64>, Vec<f64>) {
        let diagonal = vec![0.0; node_count];
        let off_diagonal = (1..node_count)
            .map(|index| {
                let degree = index as f64;
                degree / (4.0 * degree * degree - 1.0).sqrt()
            })
            .collect::<Vec<_>>();
        let (nodes, first_components) =
            symmetric_tridiagonal_eigen_first_components(&diagonal, &off_diagonal)
                .expect("Gauss-Legendre Golub-Welsch");
        let weights = first_components
            .iter()
            .map(|component| 2.0 * component * component)
            .collect();
        (nodes, weights)
    }

    /// `∫_lower^upper f` by a Gauss-Legendre rule; `integrand` returns the
    /// value and a bound on its magnitude for the rounding model.
    fn legendre_pass(
        rule: &(Vec<f64>, Vec<f64>),
        lower: f64,
        upper: f64,
        integrand: impl Fn(f64) -> (f64, f64),
    ) -> Pass {
        let half_width = 0.5 * (upper - lower);
        let midpoint = 0.5 * (upper + lower);
        let mut pass = Pass {
            sum: 0.0,
            absolute: 0.0,
        };
        for (node, weight) in rule.0.iter().zip(&rule.1) {
            let (value, magnitude) = integrand(midpoint + half_width * node);
            pass.sum += weight * half_width * value;
            pass.absolute += weight * half_width.abs() * magnitude;
        }
        pass
    }

    /// `E f(E)` for `E ~ N(0, 1)` by the Gauss-Hermite rule.
    fn hermite_pass(rule: &GaussHermiteRule, integrand: impl Fn(f64) -> (f64, f64)) -> Pass {
        let mut pass = Pass {
            sum: 0.0,
            absolute: 0.0,
        };
        for (node, weight) in rule.nodes.iter().zip(&rule.weights) {
            let (value, magnitude) = integrand(SQRT_2 * node);
            let normalized = weight / PI.sqrt();
            pass.sum += normalized * value;
            pass.absolute += normalized * magnitude;
        }
        pass
    }

    /// `Σ_j |c_j| xʲ` for `He_degree(x) = Σ_j c_j xʲ`: the magnitude the
    /// Hermite recurrence rounds against.
    fn absolute_hermite(degree: usize, x: f64) -> f64 {
        let x = x.abs();
        let mut previous = 0.0;
        let mut current = 1.0;
        for index in 0..degree {
            let next = x * current + index as f64 * previous;
            previous = current;
            current = next;
        }
        current
    }

    /// The summed magnitudes of the closed-form terms of `T⁽ᵏ⁾`.
    fn smoothing_magnitude(
        activation: GaussianActivation,
        t: f64,
        variance: f64,
        order: usize,
    ) -> f64 {
        match activation {
            GaussianActivation::Relu if variance == 0.0 => match order {
                0 => t.abs(),
                1 => 1.0,
                _ => 0.0,
            },
            GaussianActivation::Relu => {
                let scale = variance.sqrt();
                let u = t / scale;
                match order {
                    0 => t.abs() * normal_cdf(u) + scale * normal_pdf(u),
                    1 => normal_cdf(u),
                    _ => {
                        normal_pdf(u) * absolute_hermite(order - 2, u)
                            / scale.powi(order as i32 - 1)
                    }
                }
            }
            GaussianActivation::ExactGelu => {
                let total = 1.0 + variance;
                let root_total = total.sqrt();
                let u = t / root_total;
                match order {
                    0 => t.abs() * normal_cdf(u) + variance / root_total * normal_pdf(u),
                    1 => normal_cdf(u) + u.abs() * normal_pdf(u) / total,
                    _ => {
                        normal_pdf(u)
                            * (total * absolute_hermite(order - 2, u)
                                + absolute_hermite(order, u))
                            / (total * root_total.powi(order as i32 - 1))
                    }
                }
            }
        }
    }

    fn smoothing(
        activation: GaussianActivation,
        t: f64,
        variance: f64,
        orders: usize,
    ) -> Vec<f64> {
        let mut derivatives = vec![f64::NAN; orders];
        gaussian_smoothing_derivatives(activation, t, variance, &mut derivatives)
            .expect("valid smoothing law");
        derivatives
    }

    fn zero_mean(variance_x: f64, variance_y: f64, covariance: f64) -> PreactivationPair {
        PreactivationPair {
            mean_x: 0.0,
            mean_y: 0.0,
            variance_x,
            variance_y,
            covariance,
        }
    }

    /// `E[ReLU⁽ᵏ⁾(t + √v E)]`, `k ∈ {0, 1}`, as `s^p ∫_ℓ^∞ (e − ℓ)^p φ(e) de`
    /// with `ℓ = −t/√v` and `p = 1 − k`. The kink sits on the endpoint, so the
    /// integrand is analytic on the domain; the domain is truncated to where
    /// the Gaussian mass lives, and each discarded tail is bounded in closed
    /// form and added to the error bound.
    fn relu_smoothing_reference(
        t: f64,
        variance: f64,
        order: usize,
        node_count: usize,
    ) -> Reference {
        // Truncation reaches choose the domain, not the tolerance: the bound
        // of every discarded tail joins the reference's error bound.
        const GAUSSIAN_REACH: f64 = 10.0;
        const EXPONENTIAL_REACH: f64 = 45.0;
        let scale = variance.sqrt();
        let lower = -t / scale;
        let power = (1 - order) as i32;
        let (start, end, truncation) = if lower <= 0.0 {
            // Φ(−G) ≤ φ(G)/G. Above G: ∫ (e − ℓ) φ = φ(G) + |ℓ| Φ(−G) and
            // ∫ φ = Φ(−G). Below −G (when ℓ < −G): ∫ (e − ℓ) φ ≤ |ℓ| Φ(−G)
            // and ∫ φ ≤ Φ(−G).
            let tail = normal_pdf(GAUSSIAN_REACH) / GAUSSIAN_REACH;
            let bound = if power == 1 {
                normal_pdf(GAUSSIAN_REACH) + 2.0 * lower.abs() * tail
            } else {
                2.0 * tail
            };
            (lower.max(-GAUSSIAN_REACH), GAUSSIAN_REACH, bound)
        } else {
            // φ(ℓ + y) ≤ φ(ℓ) min(e^{−y²/2}, e^{−ℓy}) for y ≥ 0, integrated
            // against y^p beyond the reach Y.
            let reach = GAUSSIAN_REACH.min(EXPONENTIAL_REACH / lower);
            let gaussian_tail = (-0.5 * reach * reach).exp();
            let exponential_tail = (-lower * reach).exp();
            let bound = if power == 1 {
                normal_pdf(lower)
                    * gaussian_tail
                        .min(exponential_tail * (reach / lower + 1.0 / (lower * lower)))
            } else {
                normal_pdf(lower) * (gaussian_tail / reach).min(exponential_tail / lower)
            };
            (lower, lower + reach, bound)
        };
        let integrand = |e: f64| {
            let value = (e - lower).powi(power) * normal_pdf(e);
            (value, value.abs())
        };
        let coarse = legendre_pass(&legendre_rule(node_count), start, end, integrand);
        let fine = legendre_pass(&legendre_rule(2 * node_count), start, end, integrand);
        let reference = doubled_order(coarse, fine, 2 * node_count, 2 * node_count, truncation);
        let factor = scale.powi(power);
        Reference {
            value: factor * reference.value,
            bound: factor * reference.bound,
        }
    }

    /// `σ⁽ᵏ⁾(x)` of the exact GELU for `k ≤ 2`, with its terms' magnitude.
    fn exact_gelu_derivative(x: f64, order: usize) -> (f64, f64) {
        let density = normal_pdf(x);
        let probability = normal_cdf(x);
        match order {
            0 => (x * probability, (x * probability).abs()),
            1 => (probability + x * density, probability + (x * density).abs()),
            _ => ((2.0 - x * x) * density, (2.0 + x * x) * density),
        }
    }

    /// `E[σ(X) σ(Y)]` and `E[σ'(X) σ'(Y)]` for the exact GELU at zero mean by
    /// the tensor Gauss-Hermite rule on `X = √v E₁`,
    /// `Y = √w (ρ E₁ + √(1 − ρ²) E₂)`. The inner sum over `E₂` is completed per
    /// outer node, so recursive summation rounds over `2n` summands, not `n²`.
    fn exact_gelu_zero_mean_pair_passes(
        rule: &GaussHermiteRule,
        variance_x: f64,
        variance_y: f64,
        covariance: f64,
    ) -> (Pass, Pass) {
        let scale = variance_x.sqrt() * variance_y.sqrt();
        let correlation = if scale == 0.0 {
            0.0
        } else {
            (covariance / scale).clamp(-1.0, 1.0)
        };
        let complement = ((1.0 - correlation) * (1.0 + correlation)).sqrt();
        let root_x = variance_x.sqrt();
        let root_y = variance_y.sqrt();
        let normalized = rule
            .weights
            .iter()
            .map(|weight| weight / PI.sqrt())
            .collect::<Vec<_>>();
        let mut kernel = Pass {
            sum: 0.0,
            absolute: 0.0,
        };
        let mut derivative = kernel;
        for (first_node, first_weight) in rule.nodes.iter().zip(&normalized) {
            let first = SQRT_2 * first_node;
            let (activation_x, activation_magnitude_x) = exact_gelu_derivative(root_x * first, 0);
            let (slope_x, slope_magnitude_x) = exact_gelu_derivative(root_x * first, 1);
            let mut inner_kernel = Pass {
                sum: 0.0,
                absolute: 0.0,
            };
            let mut inner_derivative = inner_kernel;
            for (second_node, second_weight) in rule.nodes.iter().zip(&normalized) {
                let y = root_y * (correlation * first + complement * SQRT_2 * second_node);
                let (activation_y, activation_magnitude_y) = exact_gelu_derivative(y, 0);
                let (slope_y, slope_magnitude_y) = exact_gelu_derivative(y, 1);
                inner_kernel.sum += second_weight * activation_y;
                inner_kernel.absolute += second_weight * activation_magnitude_y;
                inner_derivative.sum += second_weight * slope_y;
                inner_derivative.absolute += second_weight * slope_magnitude_y;
            }
            kernel.sum += first_weight * activation_x * inner_kernel.sum;
            kernel.absolute += first_weight * activation_magnitude_x * inner_kernel.absolute;
            derivative.sum += first_weight * slope_x * inner_derivative.sum;
            derivative.absolute += first_weight * slope_magnitude_x * inner_derivative.absolute;
        }
        (kernel, derivative)
    }

    /// `E[X₊ Y₊]` at zero mean by the angular Gauss-Legendre rule over the
    /// wedge where both pre-activations are positive. In polar coordinates of
    /// the standard pair `(E₁, E₂)`, `X = √v R cos φ` and `Y = √w R cos(φ − θ)`
    /// with `cos θ = ρ`; the radial moment `∫₀^∞ R³ e^{−R²/2} dR = 2` leaves
    /// `(√(vw)/π) ∫₀^{θ'} sin ψ sin(θ' − ψ) dψ`, `θ' = π − θ = arccos(−ρ)`,
    /// whose integrand is nonnegative and carries no cancellation.
    fn relu_zero_mean_pair_reference(
        variance_x: f64,
        variance_y: f64,
        covariance: f64,
        node_count: usize,
    ) -> Reference {
        let scale = variance_x.sqrt() * variance_y.sqrt();
        let correlation = (covariance / scale).clamp(-1.0, 1.0);
        let opening = (-correlation).acos();
        let integrand = |x: f64| {
            let value = (0.5 * opening * (1.0 + x)).sin() * (0.5 * opening * (1.0 - x)).sin();
            (value, value.abs())
        };
        let coarse = legendre_pass(&legendre_rule(node_count), -1.0, 1.0, integrand);
        let fine = legendre_pass(&legendre_rule(2 * node_count), -1.0, 1.0, integrand);
        let reference = doubled_order(coarse, fine, 2 * node_count, 2 * node_count, 0.0);
        let factor = 0.5 * opening * scale / PI;
        Reference {
            value: factor * reference.value,
            bound: factor * reference.bound,
        }
    }

    #[test]
    fn relu_smoothing_matches_a_kink_free_legendre_reference_into_the_tails() {
        // (t, v): the lower tail at u = −24 and u = −30 (the continued-fraction
        // side of the log-CDF owner), the upper tail at u = 1.5e5, the kink, a
        // vanishing smoothing (v = 1e-12 at u = −1) and moderate laws.
        let laws = [
            (-12.0, 0.25),
            (-30.0, 1.0),
            (-3.0, 1.0),
            (0.0, 2.0),
            (0.7, 0.3),
            (6.0, 1.0),
            (1.5, 1.0e-10),
            (-1.0e-6, 1.0e-12),
            (7.0, 9.0),
        ];
        for (t, variance) in laws {
            let closed = smoothing(GaussianActivation::Relu, t, variance, 2);
            for order in 0..2 {
                let reference = relu_smoothing_reference(t, variance, order, 64);
                let tolerance = reference.bound
                    + closed_form_rounding(smoothing_magnitude(
                        GaussianActivation::Relu,
                        t,
                        variance,
                        order,
                    ));
                let discrepancy = (closed[order] - reference.value).abs();
                assert!(
                    discrepancy <= tolerance,
                    "ReLU T^({order}) at t = {t}, v = {variance}: closed form {} against reference {} (discrepancy {discrepancy:e}, derived tolerance {tolerance:e})",
                    closed[order],
                    reference.value
                );
            }
        }
    }

    #[test]
    fn exact_gelu_smoothing_matches_gauss_hermite_into_the_tails() {
        // Φ(t + √v E) grows like exp(v y²) off the real axis, so Gauss-Hermite
        // converges roughly like exp(−2n/(1 + v)): 256 and 512 nodes resolve
        // every v ≤ 16 here, and the doubled order checks it.
        let coarse_rule = gauss_hermite_rule(256).expect("256-node Gauss-Hermite rule");
        let fine_rule = gauss_hermite_rule(512).expect("512-node Gauss-Hermite rule");
        let laws = [
            (-8.0, 0.5),
            (-40.0, 4.0),
            (-3.0, 4.0),
            (0.0, 0.0),
            (0.4, 1.0e-10),
            (2.5, 16.0),
            (7.0, 1.0),
            (-1.0, 9.0),
        ];
        for (t, variance) in laws {
            let closed = smoothing(GaussianActivation::ExactGelu, t, variance, 3);
            for order in 0..3 {
                let integrand = |e: f64| exact_gelu_derivative(t + variance.sqrt() * e, order);
                let reference = doubled_order(
                    hermite_pass(&coarse_rule, integrand),
                    hermite_pass(&fine_rule, integrand),
                    512,
                    512,
                    0.0,
                );
                let tolerance = reference.bound
                    + closed_form_rounding(smoothing_magnitude(
                        GaussianActivation::ExactGelu,
                        t,
                        variance,
                        order,
                    ));
                let discrepancy = (closed[order] - reference.value).abs();
                assert!(
                    discrepancy <= tolerance,
                    "exact GELU T^({order}) at t = {t}, v = {variance}: closed form {} against reference {} (discrepancy {discrepancy:e}, derived tolerance {tolerance:e})",
                    closed[order],
                    reference.value
                );
            }
        }
    }

    #[test]
    fn smoothing_solves_the_heat_equation_in_integral_form() {
        // T⁽ᵏ⁾_{v₂}(t) − T⁽ᵏ⁾_{v₁}(t) = ½ ∫_{v₁}^{v₂} T⁽ᵏ⁺²⁾_v(t) dv. Both
        // smoothings are analytic in v on a neighbourhood of [v₁, v₂] whose
        // points have positive real part, so Gauss-Legendre converges
        // geometrically.
        const LOWER_VARIANCE: f64 = 0.2;
        const UPPER_VARIANCE: f64 = 3.0;
        let coarse_rule = legendre_rule(32);
        let fine_rule = legendre_rule(64);
        for activation in [GaussianActivation::Relu, GaussianActivation::ExactGelu] {
            for t in [-2.5, 0.3, 4.0] {
                let upper = smoothing(activation, t, UPPER_VARIANCE, 4);
                let lower = smoothing(activation, t, LOWER_VARIANCE, 4);
                for order in 0..4 {
                    let integrand = |variance: f64| {
                        let derivatives = smoothing(activation, t, variance, order + 3);
                        (
                            0.5 * derivatives[order + 2],
                            0.5 * smoothing_magnitude(activation, t, variance, order + 2),
                        )
                    };
                    let reference = doubled_order(
                        legendre_pass(&coarse_rule, LOWER_VARIANCE, UPPER_VARIANCE, integrand),
                        legendre_pass(&fine_rule, LOWER_VARIANCE, UPPER_VARIANCE, integrand),
                        64,
                        64,
                        0.0,
                    );
                    let increment = upper[order] - lower[order];
                    let tolerance = reference.bound
                        + closed_form_rounding(
                            smoothing_magnitude(activation, t, UPPER_VARIANCE, order)
                                + smoothing_magnitude(activation, t, LOWER_VARIANCE, order),
                        );
                    let discrepancy = (increment - reference.value).abs();
                    assert!(
                        discrepancy <= tolerance,
                        "{activation:?} heat equation for T^({order}) at t = {t}: increment {increment} against ½∫T^({}) = {} (discrepancy {discrepancy:e}, derived tolerance {tolerance:e})",
                        order + 2,
                        reference.value
                    );
                    if order == 0 && t == 0.3 {
                        // Positive control: the same increment against the unhalved
                        // integral is rejected by the same tolerance.
                        let control = (increment - 2.0 * reference.value).abs();
                        assert!(
                            control > tolerance,
                            "{activation:?} heat-equation control at t = {t}: discrepancy {control:e} within tolerance {tolerance:e}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn smoothing_derivatives_integrate_to_the_orders_below() {
        // T⁽ᵏ⁾(t₂) − T⁽ᵏ⁾(t₁) = ∫_{t₁}^{t₂} T⁽ᵏ⁺¹⁾(t) dt, with entire integrands.
        const START: f64 = -3.0;
        const END: f64 = 2.0;
        let coarse_rule = legendre_rule(48);
        let fine_rule = legendre_rule(96);
        for activation in [GaussianActivation::Relu, GaussianActivation::ExactGelu] {
            for variance in [0.5, 4.0] {
                let start = smoothing(activation, START, variance, 6);
                let end = smoothing(activation, END, variance, 6);
                for order in 0..5 {
                    let integrand = |t: f64| {
                        (
                            smoothing(activation, t, variance, order + 2)[order + 1],
                            smoothing_magnitude(activation, t, variance, order + 1),
                        )
                    };
                    let reference = doubled_order(
                        legendre_pass(&coarse_rule, START, END, integrand),
                        legendre_pass(&fine_rule, START, END, integrand),
                        96,
                        96,
                        0.0,
                    );
                    let increment = end[order] - start[order];
                    let tolerance = reference.bound
                        + closed_form_rounding(
                            smoothing_magnitude(activation, END, variance, order)
                                + smoothing_magnitude(activation, START, variance, order),
                        );
                    let discrepancy = (increment - reference.value).abs();
                    assert!(
                        discrepancy <= tolerance,
                        "{activation:?} T^({order}) increment on [{START}, {END}] at v = {variance}: {increment} against ∫T^({}) = {} (discrepancy {discrepancy:e}, derived tolerance {tolerance:e})",
                        order + 1,
                        reference.value
                    );
                }
                // Positive control: the value's increment against the integral of
                // the second derivative is rejected.
                let integrand = |t: f64| {
                    (
                        smoothing(activation, t, variance, 3)[2],
                        smoothing_magnitude(activation, t, variance, 2),
                    )
                };
                let mismatched = doubled_order(
                    legendre_pass(&coarse_rule, START, END, integrand),
                    legendre_pass(&fine_rule, START, END, integrand),
                    96,
                    96,
                    0.0,
                );
                let control = (end[0] - start[0] - mismatched.value).abs();
                assert!(
                    control > mismatched.bound,
                    "{activation:?} derivative-chain control at v = {variance}: discrepancy {control:e} within {:e}",
                    mismatched.bound
                );
            }
        }
    }

    #[test]
    fn relu_smoothing_without_variance_is_the_activation_with_its_gaussian_limit_slope() {
        for (t, value, slope) in [(-2.0, 0.0, 0.0), (0.0, 0.0, 0.5), (3.0, 3.0, 1.0)] {
            let mut derivatives = [f64::NAN; 2];
            gaussian_smoothing_derivatives(GaussianActivation::Relu, t, 0.0, &mut derivatives)
                .expect("ReLU value and slope without smoothing");
            assert_eq!(derivatives, [value, slope], "ReLU at t = {t} with v = 0");
        }
        let mut away_from_kink = [f64::NAN; 4];
        gaussian_smoothing_derivatives(GaussianActivation::Relu, 1.5, 0.0, &mut away_from_kink)
            .expect("ReLU curvature away from the kink");
        assert_eq!(away_from_kink, [1.5, 1.0, 0.0, 0.0]);
        let mut at_kink = [f64::NAN; 3];
        assert_eq!(
            gaussian_smoothing_derivatives(GaussianActivation::Relu, 0.0, 0.0, &mut at_kink),
            Err(GaussianActivationError::UnsmoothedReluKink { order: 2 })
        );
        // As v → 0, T_v ReLU(t) − t₊ = √v L(|t|/√v) with the unit normal loss
        // L(x) = φ(x) − x Φ(−x) ∈ [0, φ(0)], so the smoothing reaches the
        // activation within √v φ(0).
        for variance in [1.0e-4, 1.0e-10, 1.0e-16] {
            for t in [-1.0e-3, 0.0, 2.0e-3] {
                let value = smoothing(GaussianActivation::Relu, t, variance, 1)[0];
                let rounding = closed_form_rounding(smoothing_magnitude(
                    GaussianActivation::Relu,
                    t,
                    variance,
                    0,
                ));
                let excess = value - t.max(0.0);
                assert!(
                    excess >= -rounding && excess <= variance.sqrt() * normal_pdf(0.0) + rounding,
                    "ReLU smoothing at t = {t}, v = {variance}: excess {excess:e} over t₊ outside [0, √v φ(0)]"
                );
            }
        }
    }

    #[test]
    fn relu_zero_mean_pair_kernel_matches_the_wedge_reference_up_to_the_correlation_limits() {
        let laws = [
            (1.0, 1.0, 0.3),
            (0.25, 4.0, -0.9),
            (9.0, 4.0, 5.994),
            (16.0, 9.0, -6.0),
            (4.0, 4.0, 4.0 * (1.0 - 1.0e-12)),
            (4.0, 4.0, 4.0),
            (4.0, 4.0, -4.0 * (1.0 - 1.0e-6)),
            (4.0, 4.0, -4.0 * (1.0 - 1.0e-12)),
            (4.0, 4.0, -4.0),
            (1.0e-300, 1.0, 0.5e-150),
        ];
        for (variance_x, variance_y, covariance) in laws {
            let closed = pair_kernel(
                GaussianActivation::Relu,
                zero_mean(variance_x, variance_y, covariance),
            )
            .expect("zero-mean ReLU pair kernel");
            let reference = relu_zero_mean_pair_reference(variance_x, variance_y, covariance, 16);
            let scale = variance_x.sqrt() * variance_y.sqrt();
            let correlation = (covariance / scale).clamp(-1.0, 1.0);
            let sine = ((1.0 - correlation) * (1.0 + correlation)).sqrt();
            let magnitude =
                scale * (sine / TAU + correlation.abs() * closed.covariance_derivative);
            let tolerance = reference.bound + closed_form_rounding(magnitude);
            let discrepancy = (closed.value - reference.value).abs();
            assert!(
                discrepancy <= tolerance,
                "ReLU K at v = {variance_x}, w = {variance_y}, r = {covariance}: closed form {} against reference {} (discrepancy {discrepancy:e}, derived tolerance {tolerance:e})",
                closed.value,
                reference.value
            );
        }
        // The orthant probability at the correlation limits and independence.
        for (covariance, expected) in [(6.0, 0.5), (-6.0, 0.0), (0.0, 0.25)] {
            let closed = pair_kernel(GaussianActivation::Relu, zero_mean(4.0, 9.0, covariance))
                .expect("zero-mean ReLU pair kernel");
            assert_eq!(closed.covariance_derivative, expected, "ReLU ∂_r K at r = {covariance}");
        }
        // Positive control: the kernel with θ in place of π − θ is rejected.
        let reference = relu_zero_mean_pair_reference(1.0, 1.0, 0.3, 16);
        let wrong = (0.91_f64.sqrt() + 0.3 * 0.3_f64.acos()) / TAU;
        let control = (wrong - reference.value).abs();
        assert!(
            control > reference.bound + closed_form_rounding(wrong),
            "ReLU wedge control: discrepancy {control:e} within {:e}",
            reference.bound
        );
    }

    #[test]
    fn exact_gelu_zero_mean_pair_kernel_and_price_derivative_match_gauss_hermite() {
        // X = √v E₁ reads Φ on the scale 1/√v, so the rule converges roughly
        // like exp(−2n/(1 + v)); the large-norm law (ρ_G = r/√(AB) → 1) takes
        // 1024/2048 nodes and the rest 256/512. The doubled order checks it.
        let moderate = (
            gauss_hermite_rule(256).expect("256-node Gauss-Hermite rule"),
            gauss_hermite_rule(512).expect("512-node Gauss-Hermite rule"),
        );
        let wide = (
            gauss_hermite_rule(1024).expect("1024-node Gauss-Hermite rule"),
            gauss_hermite_rule(2048).expect("2048-node Gauss-Hermite rule"),
        );
        let laws = [
            (1.0, 1.0, 0.3, &moderate),
            (0.25, 4.0, -0.9, &moderate),
            (9.0, 4.0, 5.994, &moderate),
            (16.0, 9.0, -12.0 * (1.0 - 1.0e-8), &moderate),
            (4.0, 4.0, 4.0 * (1.0 - 1.0e-12), &moderate),
            (4.0, 4.0, 4.0, &moderate),
            (4.0, 4.0, -4.0, &moderate),
            (1.0e-12, 1.0, 0.5e-6, &moderate),
            (0.0, 2.0, 0.0, &moderate),
            (64.0, 64.0, 64.0 * 0.999, &wide),
        ];
        for (variance_x, variance_y, covariance, rules) in laws {
            let closed = pair_kernel(
                GaussianActivation::ExactGelu,
                zero_mean(variance_x, variance_y, covariance),
            )
            .expect("zero-mean exact GELU pair kernel");
            let (coarse_kernel, coarse_derivative) =
                exact_gelu_zero_mean_pair_passes(&rules.0, variance_x, variance_y, covariance);
            let (fine_kernel, fine_derivative) =
                exact_gelu_zero_mean_pair_passes(&rules.1, variance_x, variance_y, covariance);
            let nodes = rules.1.nodes.len();
            let kernel = doubled_order(coarse_kernel, fine_kernel, 2 * nodes, 2 * nodes, 0.0);
            let derivative =
                doubled_order(coarse_derivative, fine_derivative, 2 * nodes, 2 * nodes, 0.0);
            // |r H| ≤ |r|/2, 0 ≤ vw + r² q ≤ vw + 2r² and 2π√Δ ≥ 2π bound the
            // closed form's terms; likewise H ≤ ½ and 1/A + 1/B + 1/Δ ≤ 3.
            let kernel_magnitude = 0.5 * covariance.abs()
                + (variance_x * variance_y + 2.0 * covariance * covariance) / TAU;
            let derivative_magnitude = 0.5 + 3.0 * covariance.abs() / TAU;
            let kernel_tolerance = kernel.bound + closed_form_rounding(kernel_magnitude);
            let derivative_tolerance =
                derivative.bound + closed_form_rounding(derivative_magnitude);
            let kernel_discrepancy = (closed.value - kernel.value).abs();
            let derivative_discrepancy = (closed.covariance_derivative - derivative.value).abs();
            assert!(
                kernel_discrepancy <= kernel_tolerance,
                "exact GELU K at v = {variance_x}, w = {variance_y}, r = {covariance}: closed form {} against reference {} (discrepancy {kernel_discrepancy:e}, derived tolerance {kernel_tolerance:e})",
                closed.value,
                kernel.value
            );
            assert!(
                derivative_discrepancy <= derivative_tolerance,
                "exact GELU ∂_r K at v = {variance_x}, w = {variance_y}, r = {covariance}: closed form {} against E[σ'σ'] {} (discrepancy {derivative_discrepancy:e}, derived tolerance {derivative_tolerance:e})",
                closed.covariance_derivative,
                derivative.value
            );
            if variance_x == 9.0 {
                // Positive controls: the kernel without the r² q coupling and the
                // derivative without Price's density term are both rejected.
                let total_x = 1.0 + variance_x;
                let total_y = 1.0 + variance_y;
                let discriminant = total_x * total_y - covariance * covariance;
                let density = 1.0 / (TAU * discriminant.sqrt());
                let coupling = 1.0 / total_x + 1.0 / total_y - 1.0;
                let wrong_kernel = closed.value - covariance * covariance * coupling * density;
                let wrong_derivative = closed.covariance_derivative
                    - covariance * density * (1.0 / total_x + 1.0 / total_y + 1.0 / discriminant);
                assert!(
                    (wrong_kernel - kernel.value).abs() > kernel_tolerance,
                    "exact GELU kernel control within tolerance {kernel_tolerance:e}"
                );
                assert!(
                    (wrong_derivative - derivative.value).abs() > derivative_tolerance,
                    "exact GELU derivative control within tolerance {derivative_tolerance:e}"
                );
            }
        }
    }

    #[test]
    fn pair_kernel_covariance_derivative_matches_a_richardson_checked_difference() {
        // Central differences D(h) = K_r + c h² + O(h⁴): |D(h) − D(h/2)| bounds
        // the truncation of D(h/2), and each kernel's rounding enters divided by
        // the step. The step is a design choice; the tolerance comes from the
        // halved-step check.
        let laws = [(1.0, 1.0, 0.3), (9.0, 4.0, -3.0), (0.25, 4.0, 0.8), (16.0, 9.0, 11.0)];
        for activation in [GaussianActivation::Relu, GaussianActivation::ExactGelu] {
            for (variance_x, variance_y, covariance) in laws {
                let kernel = |value: f64| {
                    pair_kernel(activation, zero_mean(variance_x, variance_y, value))
                        .expect("zero-mean pair kernel")
                };
                let scale = variance_x.sqrt() * variance_y.sqrt();
                let step = 1.0e-3 * scale;
                let difference =
                    |h: f64| (kernel(covariance + h).value - kernel(covariance - h).value) / (2.0 * h);
                let coarse = difference(step);
                let half_step = 0.5 * step;
                let fine = difference(half_step);
                let reach = covariance.abs() + step;
                let magnitude = match activation {
                    GaussianActivation::Relu => scale / TAU + 0.5 * reach,
                    GaussianActivation::ExactGelu => {
                        0.5 * reach + (variance_x * variance_y + 2.0 * reach * reach) / TAU
                    }
                };
                let tolerance = (coarse - fine).abs()
                    + closed_form_rounding(magnitude) / half_step
                    + fine.abs() * f64::EPSILON * reach / half_step;
                let closed = kernel(covariance).covariance_derivative;
                let discrepancy = (closed - fine).abs();
                assert!(
                    discrepancy <= tolerance,
                    "{activation:?} ∂_r K at v = {variance_x}, w = {variance_y}, r = {covariance}: closed form {closed} against difference {fine} (discrepancy {discrepancy:e}, derived tolerance {tolerance:e})"
                );
                // Positive control: the orthant term alone, with Price's coupling
                // (GELU) or the complementary angle (ReLU) wrong, is rejected.
                let correlation = covariance / scale;
                let wrong = match activation {
                    GaussianActivation::Relu => correlation.acos() / TAU,
                    GaussianActivation::ExactGelu => {
                        (-covariance / ((1.0 + variance_x) * (1.0 + variance_y)).sqrt()).acos()
                            / TAU
                    }
                };
                assert!(
                    (wrong - fine).abs() > tolerance,
                    "{activation:?} Richardson control at r = {covariance}: within tolerance {tolerance:e}"
                );
            }
        }
    }

    #[test]
    fn pair_kernel_projects_the_covariance_onto_the_cauchy_schwarz_interval() {
        for activation in [GaussianActivation::Relu, GaussianActivation::ExactGelu] {
            for sign in [1.0, -1.0] {
                let boundary = pair_kernel(activation, zero_mean(4.0, 9.0, sign * 6.0))
                    .expect("boundary covariance");
                let beyond = pair_kernel(activation, zero_mean(4.0, 9.0, sign * 6.0 * (1.0 + 1.0e-9)))
                    .expect("covariance beyond Cauchy-Schwarz");
                assert_eq!(beyond, boundary, "{activation:?} projection at sign {sign}");
            }
            // A constant pre-activation: K = σ(0) E σ(Y) = 0 and ∂_r K = σ'(0) E σ'(Y) = ¼.
            for covariance in [0.0, 0.1] {
                assert_eq!(
                    pair_kernel(activation, zero_mean(0.0, 2.0, covariance)),
                    Ok(PairKernel {
                        value: 0.0,
                        covariance_derivative: 0.25
                    }),
                    "{activation:?} with a zero variance and r = {covariance}"
                );
            }
        }
        assert_eq!(
            pair_kernel(GaussianActivation::Relu, zero_mean(4.0, 9.0, 6.0)),
            Ok(PairKernel {
                value: 3.0,
                covariance_derivative: 0.5
            })
        );
    }

    #[test]
    fn invalid_laws_are_refused_with_typed_errors() {
        let mut derivatives = [0.0; 2];
        assert_eq!(
            gaussian_smoothing_derivatives(GaussianActivation::ExactGelu, 0.0, -1.0, &mut derivatives),
            Err(GaussianActivationError::InvalidVariance { variance: -1.0 })
        );
        assert!(matches!(
            gaussian_smoothing_derivatives(GaussianActivation::Relu, 0.0, f64::NAN, &mut derivatives),
            Err(GaussianActivationError::InvalidVariance { .. })
        ));
        assert_eq!(
            gaussian_smoothing_derivatives(GaussianActivation::Relu, f64::INFINITY, 1.0, &mut derivatives),
            Err(GaussianActivationError::NonFiniteArgument {
                value: f64::INFINITY
            })
        );
        assert_eq!(
            pair_kernel(GaussianActivation::Relu, zero_mean(1.0, f64::INFINITY, 0.0)),
            Err(GaussianActivationError::InvalidVariance {
                variance: f64::INFINITY
            })
        );
        let biased = PreactivationPair {
            mean_x: 0.5,
            ..zero_mean(1.0, 1.0, 0.2)
        };
        assert_eq!(
            pair_kernel(GaussianActivation::ExactGelu, biased),
            Err(GaussianActivationError::BiasedPairNeedsBivariateNormalCdf {
                mean_x: 0.5,
                mean_y: 0.0
            })
        );
    }
}
