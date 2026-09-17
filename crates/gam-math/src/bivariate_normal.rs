//! The bivariate standard normal distribution function
//! `Φ₂(h, k; ρ) = P(X ≤ h, Y ≤ k)`, for standard normal `X, Y` with correlation
//! `ρ`, together with its density and its analytic partials (#2946).
//!
//! # Core rule, `|ρ| ≤ ½`
//!
//! Plackett's identity `∂_ρ Φ₂ = φ₂` integrated from `ρ = 0` with `r = sin θ`
//! (Drezner and Wesolowsky 1990) gives
//!
//! `Φ₂(h, k; ρ) = Φ(h)Φ(k) + (1/2π) ∫₀^α f(θ) dθ`, with `α = asin ρ`.
//!
//! The integrand is written with non-negative coefficients only:
//!
//! `f(θ) = exp(−a / cos²θ − b / (1 + σ sin θ))`,
//! with `a = (|h| − |k|)²/2`, `b = |hk|` and `σ = sign(hk)`.
//!
//! This is `(h² − 2hk sin θ + k²)/(2cos²θ)` split as `(|h| − |k|)²/2 + |hk|(1 − σ sin θ)`
//! over `(1 − sin θ)(1 + sin θ)`. It never cancels and never forms `∞ − ∞`.
//!
//! **Uniform bound.** On the diamond `|Re θ| + |Im θ| ≤ π/2`:
//! - `Re(1 + σ sin θ) = 1 + σ sin x cosh y ≥ 1 − cos t cosh t ≥ 0`, with `t = π/2 − |x| ≥ |y|`, because
//!   `cos t cosh t ≤ 1` on `[0, π/2]`;
//! - `Re cos²θ ≥ 0` follows from the same inequality.
//!
//! So `|f| ≤ 1` there for every `(h, k)`.
//!
//! **Order.** Map `θ = α(1 + x)/2`. The Bernstein ellipse `E_r` of `[−1, 1]` stays inside the diamond iff
//! `(|α|/2)(1 + √((r² + r⁻²)/2)) ≤ π/2`. The largest core angle `|α| = π/6` gives `r² + r⁻² = 50`.
//! Gauss-Legendre with `n` nodes misses `∫₋₁¹` by at most `(64/15) r⁻²ⁿ/(r² − 1)` (Trefethen 2008,
//! Thm 4.5). The order is the smallest `n` whose bound is at most `2ε`, the rounding scale of the same
//! sum (its weights sum to 2 and `0 < f ≤ 1`). A higher order would resolve digits the sum cannot hold.
//! The result is `n = 9` (`core_order`), and the core's truncation is at most `(|α|/4π)·2ε ≤ ε/12`.
//!
//! # Reductions, `|ρ| > ½`
//!
//! Rotate to independent `U = (X − Y)/√(2(1 − ρ))` and `V = (X + Y)/√(2(1 + ρ))`, and write
//! `c± = √((1 ± ρ)/2)`.
//!
//! - **`ρ > ½`.** The region `{X ≤ h, Y ≤ k}` splits along `U = u* = (h − k)/(2c₋)`:
//!   `Φ₂(h, k; ρ) = Φ₂(u*, k; −c₋) + Φ₂(−u*, h; −c₋)`.
//! - **`ρ < −½`.** The same region is `{V ≤ w*, lo(V) ≤ U ≤ hi(V)}`, with `w* = (h + k)/(2c₊)`:
//!   `Φ₂(h, k; ρ) = Φ₂(w*, h; c₊) − Φ₂(w*, −k; −c₊)`.
//!
//! Both are exact. The new correlations have modulus `√((1 − |ρ|)/2) < ½`, so every evaluation lands on the
//! core. `½` is not a knob: it is the fixed point `τ = √((1 − τ)/2)`, the smallest core domain the map
//! closes on.
//!
//! As `|ρ| → 1` the pieces tend to independence (`c → 0`). That is the regime where a fixed Gauss-Legendre
//! rule on `θ ∈ [0, asin ρ]` loses digits: the integrand's essential singularity at `θ = ±π/2` reaches the
//! interval. Genz (2004) handles `|ρ| > 0.925` with `s = √(1 − r²)`. That leaves `exp(−(h − k)²/2s²)` singular
//! at the endpoint `s = 0`, where no Bernstein ellipse exists. The rotation removes the endpoint instead.
//!
//! # Contract
//!
//! - **Truncation.** At most `ε/12` for `|ρ| ≤ ½`, and at most `ε/6` elsewhere.
//! - **Rounding.** On top of that, rounding of a few ulps in `Φ`, the sum and the final combination.
//! - **Exact branches.** Infinite bounds and `ρ ∈ {−1, 0, 1}` are exact special cases.
//! - **Projection.** Every result is projected onto `[0, 1]`, which contains the truth, so the projection never
//!   increases the error.
//! - **Relative accuracy.** It is not claimed for `Φ₂ ≲ ε`. The sum `Φ(h)Φ(k) + T` cancels when `ρ < 0` in the
//!   lower tails, and so do the `ρ < −½` difference and the negative-correlation pieces of `ρ > ½`.

use crate::probability::{normal_cdf, normal_pdf};
use crate::special::gauss_legendre;
use libm::{erf, erfc};
use std::f64::consts::{FRAC_PI_6, PI, SQRT_2};
use std::fmt;
use std::sync::LazyLock;

/// A bivariate normal argument outside the distribution's domain.
#[derive(Clone, Debug, PartialEq)]
pub enum BivariateNormalError {
    /// A bound was `NaN`. Both infinities are admissible bounds.
    NanBound { name: &'static str },
    /// The correlation was `NaN` or outside `[−1, 1]`.
    CorrelationOutsideUnitInterval { rho: f64 },
    /// The density and the partials exist only for `|ρ| < 1`.
    SingularCorrelation { rho: f64 },
}

impl fmt::Display for BivariateNormalError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NanBound { name } => {
                write!(formatter, "bivariate normal bound {name} is NaN")
            }
            Self::CorrelationOutsideUnitInterval { rho } => write!(
                formatter,
                "bivariate normal correlation must lie in [-1, 1], got {rho}"
            ),
            Self::SingularCorrelation { rho } => write!(
                formatter,
                "bivariate normal density needs |rho| < 1, got {rho}"
            ),
        }
    }
}

impl std::error::Error for BivariateNormalError {}

impl From<BivariateNormalError> for String {
    fn from(error: BivariateNormalError) -> Self {
        error.to_string()
    }
}

/// The analytic partials of `Φ₂(h, k; ρ)`:
/// - `∂_h Φ₂ = φ(h) Φ((k − ρh)/√(1 − ρ²))`, and symmetrically for `k`;
/// - `∂_ρ Φ₂ = φ₂(h, k; ρ)` (Plackett).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BivariateNormalPartials {
    pub d_h: f64,
    pub d_k: f64,
    pub d_rho: f64,
}

/// The largest `|asin ρ|` the core rule evaluates: `asin ½`.
const CORE_MAX_ANGLE: f64 = FRAC_PI_6;

/// A Gauss-Legendre rule stored on `[0, 1]`: nodes `(1 + x)/2` and the weights of `[−1, 1]`.
struct CoreRule {
    unit_nodes: Vec<f64>,
    weights: Vec<f64>,
}

impl CoreRule {
    fn with_order(order: usize) -> Self {
        let (nodes, weights) = gauss_legendre(order);
        Self {
            unit_nodes: nodes.iter().map(|node| 0.5 * (1.0 + node)).collect(),
            weights,
        }
    }
}

static CORE_RULE: LazyLock<CoreRule> = LazyLock::new(|| CoreRule::with_order(core_order()));

/// `r²` of the largest Bernstein ellipse whose image stays inside `|Re θ| + |Im θ| ≤ π/2` at the largest
/// core angle: `r² + r⁻² = 2s`, with `s = (π/α − 1)²`.
fn core_bernstein_radius_squared() -> f64 {
    let s = (PI / CORE_MAX_ANGLE - 1.0).powi(2);
    s + (s * s - 1.0).sqrt()
}

/// The smallest Gauss-Legendre order whose Trefethen bound `(64/15) r⁻²ⁿ/(r² − 1)` for an integrand bounded
/// by 1 on `E_r` is at most `2ε`, the rounding scale of the rule's own sum.
fn core_order() -> usize {
    let radius_squared = core_bernstein_radius_squared();
    let target = 2.0 * f64::EPSILON;
    ((64.0 / (15.0 * target * (radius_squared - 1.0))).ln() / radius_squared.ln()).ceil() as usize
}

fn validate(bounds: &[(&'static str, f64)], rho: f64) -> Result<(), BivariateNormalError> {
    if let Some(bound) = bounds.iter().find(|bound| bound.1.is_nan()) {
        return Err(BivariateNormalError::NanBound { name: bound.0 });
    }
    if !(-1.0..=1.0).contains(&rho) {
        return Err(BivariateNormalError::CorrelationOutsideUnitInterval { rho });
    }
    Ok(())
}

/// `P(lower ≤ Z ≤ upper)` for standard normal `Z`, always as a difference of same-side tails. So a small
/// interval deep in either tail keeps its relative digits.
fn normal_interval_probability(lower: f64, upper: f64) -> f64 {
    if !(lower < upper) {
        return 0.0;
    }
    let probability = if lower >= 0.0 {
        0.5 * (erfc(lower / SQRT_2) - erfc(upper / SQRT_2))
    } else if upper <= 0.0 {
        0.5 * (erfc(-upper / SQRT_2) - erfc(-lower / SQRT_2))
    } else {
        0.5 * (erf(upper / SQRT_2) - erf(lower / SQRT_2))
    };
    probability.max(0.0)
}

/// The exact value when a bound is infinite, or when `ρ` is singular or zero. Otherwise `None`.
fn exact_branch(h: f64, k: f64, rho: f64) -> Option<f64> {
    if h == f64::NEG_INFINITY || k == f64::NEG_INFINITY {
        Some(0.0)
    } else if h == f64::INFINITY {
        Some(normal_cdf(k))
    } else if k == f64::INFINITY {
        Some(normal_cdf(h))
    } else if rho == 1.0 {
        Some(normal_cdf(h.min(k)))
    } else if rho == -1.0 {
        Some(normal_interval_probability(-k, h))
    } else if rho == 0.0 {
        Some(normal_cdf(h) * normal_cdf(k))
    } else {
        None
    }
}

/// Drezner-Wesolowsky on `rule` for `|ρ| ≤ ½` (see the module docs for the integrand).
fn core_cdf(h: f64, k: f64, rho: f64, rule: &CoreRule) -> f64 {
    if let Some(value) = exact_branch(h, k, rho) {
        return value;
    }
    let alpha = rho.asin();
    let a = 0.5 * (h.abs() - k.abs()).powi(2);
    let b = (h * k).abs();
    // `σ sin θ = sin(σθ)`, while `cos²θ` does not see the sign.
    let signed_alpha = if h * k < 0.0 { -alpha } else { alpha };
    let mut sum = 0.0;
    for (&unit, &weight) in rule.unit_nodes.iter().zip(&rule.weights) {
        let sine = (signed_alpha * unit).sin();
        sum += weight * (-(a / (1.0 - sine * sine) + b / (1.0 + sine))).exp();
    }
    (normal_cdf(h) * normal_cdf(k) + alpha * sum / (4.0 * PI)).clamp(0.0, 1.0)
}

/// `Φ₂(h, k; ρ)` on `rule`, through the exact reductions onto the core domain.
fn cdf_on_rule(h: f64, k: f64, rho: f64, rule: &CoreRule) -> f64 {
    if let Some(value) = exact_branch(h, k, rho) {
        return value;
    }
    if rho > 0.5 {
        let c = (0.5 * (1.0 - rho)).sqrt();
        let split = (h - k) / (2.0 * c);
        (core_cdf(split, k, -c, rule) + core_cdf(-split, h, -c, rule)).min(1.0)
    } else if rho < -0.5 {
        let c = (0.5 * (1.0 + rho)).sqrt();
        let split = (h + k) / (2.0 * c);
        // Taking `h ≤ k` centres the conditional interval of `U` at or below zero. So the two pieces are
        // never both near one.
        let (low, high) = if h <= k { (h, k) } else { (k, h) };
        (core_cdf(split, low, c, rule) - core_cdf(split, -high, -c, rule)).clamp(0.0, 1.0)
    } else {
        core_cdf(h, k, rho, rule)
    }
}

/// `Φ₂(h, k; ρ) = P(X ≤ h, Y ≤ k)`. Bounds may be infinite, and `ρ ∈ [−1, 1]`. The error contract is in the
/// module docs.
pub fn bivariate_normal_cdf(h: f64, k: f64, rho: f64) -> Result<f64, BivariateNormalError> {
    validate(&[("h", h), ("k", k)], rho)?;
    Ok(cdf_on_rule(h, k, rho, &CORE_RULE))
}

/// `P(X ≤ h, lower ≤ Y ≤ upper)`.
///
/// A finite interval is a difference of two distribution functions taken on the tail side of zero: the
/// reflection `Y → −Y` when `lower ≥ 0`. So a small interval in the upper tail keeps its mass instead of
/// being subtracted from `Φ(h)`.
pub fn bivariate_normal_interval_probability(
    h: f64,
    lower: f64,
    upper: f64,
    rho: f64,
) -> Result<f64, BivariateNormalError> {
    validate(&[("h", h), ("lower", lower), ("upper", upper)], rho)?;
    if !(lower < upper) {
        return Ok(0.0);
    }
    let rule = &*CORE_RULE;
    let probability = if lower == f64::NEG_INFINITY {
        cdf_on_rule(h, upper, rho, rule)
    } else if upper == f64::INFINITY {
        cdf_on_rule(h, -lower, -rho, rule)
    } else if lower >= 0.0 {
        cdf_on_rule(h, -lower, -rho, rule) - cdf_on_rule(h, -upper, -rho, rule)
    } else {
        cdf_on_rule(h, upper, rho, rule) - cdf_on_rule(h, lower, rho, rule)
    };
    Ok(probability.clamp(0.0, 1.0))
}

/// `(1 − ρ)(1 + ρ)`, each factor exact in binary64 near its own singular end.
fn one_minus_rho_squared(rho: f64) -> f64 {
    (1.0 - rho) * (1.0 + rho)
}

/// `φ₂(h, k; ρ)` for `|ρ| < 1`. The quadratic form is carried with non-negative terms, as in the core rule
/// with `sin θ = ρ`.
fn density(h: f64, k: f64, rho: f64) -> f64 {
    if h.is_infinite() || k.is_infinite() {
        return 0.0;
    }
    let determinant = one_minus_rho_squared(rho);
    let a = 0.5 * (h.abs() - k.abs()).powi(2);
    let b = (h * k).abs();
    let signed_rho = if h * k < 0.0 { -rho } else { rho };
    (-(a / determinant + b / (1.0 + signed_rho))).exp() / (2.0 * PI * determinant.sqrt())
}

/// `φ₂(h, k; ρ)`, the bivariate standard normal density, for `|ρ| < 1`.
pub fn bivariate_normal_pdf(h: f64, k: f64, rho: f64) -> Result<f64, BivariateNormalError> {
    validate(&[("h", h), ("k", k)], rho)?;
    if rho.abs() == 1.0 {
        return Err(BivariateNormalError::SingularCorrelation { rho });
    }
    Ok(density(h, k, rho))
}

/// `φ(x) Φ((y − ρx)/√(1 − ρ²))`. The numerator is `(y − x) + x(1 − ρ)` for `ρ ≥ 0` and `(y + x) − x(1 + ρ)`
/// for `ρ < 0`, so it keeps its digits as `|ρ| → 1`.
fn conditional_partial(x: f64, y: f64, rho: f64) -> f64 {
    let weight = normal_pdf(x);
    if weight == 0.0 {
        return 0.0;
    }
    let numerator = if rho >= 0.0 {
        (y - x) + x * (1.0 - rho)
    } else {
        (y + x) - x * (1.0 + rho)
    };
    weight * normal_cdf(numerator / one_minus_rho_squared(rho).sqrt())
}

/// The analytic partials `(∂_h, ∂_k, ∂_ρ)` of `Φ₂(h, k; ρ)`, for `|ρ| < 1`.
pub fn bivariate_normal_cdf_partials(
    h: f64,
    k: f64,
    rho: f64,
) -> Result<BivariateNormalPartials, BivariateNormalError> {
    validate(&[("h", h), ("k", k)], rho)?;
    if rho.abs() == 1.0 {
        return Err(BivariateNormalError::SingularCorrelation { rho });
    }
    Ok(BivariateNormalPartials {
        d_h: conditional_partial(h, k, rho),
        d_k: conditional_partial(k, h, rho),
        d_rho: density(h, k, rho),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::TAU;

    /// The rounding allowance of one production call.
    ///
    /// Per core evaluation:
    /// - `Φ(h)Φ(k)`: each `erfc` is within one ulp, so the product is within `3ε`;
    /// - each weighted node: the weight (≤ 2ε at these orders), the exponent (`ε·x e⁻ˣ ≤ ε` for each of four
    ///   operations) and `exp` (1 ulp) put it within `7ε` of its weight. The weights sum to 2, under the
    ///   prefactor `|α|/4π ≤ 1/24`;
    /// - the final addition: `ε`.
    ///
    /// A reduction combines two evaluations, and the truncation adds `ε/6`.
    fn contract() -> f64 {
        let per_evaluation = (3.0 + 7.0 * 2.0 / 24.0 + 1.0) * f64::EPSILON;
        2.0 * per_evaluation + f64::EPSILON / 6.0
    }

    /// Neumaier's compensated sum, returned with `Σ|term|`. The rounding left over is the terms' own,
    /// at most a few ε of that magnitude.
    fn compensated_sum(terms: impl IntoIterator<Item = f64>) -> (f64, f64) {
        let (mut sum, mut compensation, mut magnitude) = (0.0_f64, 0.0_f64, 0.0_f64);
        for term in terms {
            let next = sum + term;
            compensation += if sum.abs() >= term.abs() {
                (sum - next) + term
            } else {
                (term - next) + sum
            };
            sum = next;
            magnitude += term.abs();
        }
        (sum + compensation, magnitude)
    }

    /// Composite Gauss-Legendre with `order` nodes per panel over `breaks`. It returns the integral and
    /// `Σ|term|`.
    fn composite(breaks: &[f64], order: usize, integrand: &dyn Fn(f64) -> f64) -> (f64, f64) {
        let (nodes, weights) = gauss_legendre(order);
        compensated_sum(breaks.windows(2).flat_map(|panel| {
            let (middle, half) = (0.5 * (panel[0] + panel[1]), 0.5 * (panel[1] - panel[0]));
            nodes
                .iter()
                .zip(weights.iter())
                .map(move |(&node, &weight)| half * weight * integrand(middle + half * node))
                .collect::<Vec<_>>()
        }))
    }

    /// An independent reference: the conditional form `∫_{−40}^{h} φ(x) Φ((k − ρx)/√(1 − ρ²)) dx`.
    ///
    /// - Panels: unit panels, plus panels graded geometrically about the conditional step `x = k/ρ`, of
    ///   width `√(1 − ρ²)/|ρ|`. So the step is resolved however close `|ρ|` is to 1.
    /// - Truncation: the mass below −40 is `Φ(−40)`, which is zero in binary64 (asserted by the caller).
    fn conditional_reference(h: f64, k: f64, rho: f64, order: usize) -> (f64, f64) {
        let (lower, upper) = (-40.0, h.min(40.0));
        if upper <= lower {
            return (0.0, 0.0);
        }
        let s = one_minus_rho_squared(rho).sqrt();
        let mut breaks = vec![lower, upper];
        let mut unit = lower + 1.0;
        while unit < upper {
            breaks.push(unit);
            unit += 1.0;
        }
        let (centre, width) = (k / rho, s / rho.abs());
        let mut offset = width;
        while offset < 80.0 {
            breaks.extend([centre - offset, centre + offset]);
            offset *= 2.0;
        }
        breaks.push(centre);
        breaks.retain(|point| *point >= lower && *point <= upper);
        breaks.sort_by(f64::total_cmp);
        breaks.dedup();
        composite(&breaks, order, &|x| {
            let numerator = if rho >= 0.0 {
                (k - x) + x * (1.0 - rho)
            } else {
                (k + x) - x * (1.0 + rho)
            };
            normal_pdf(x) * normal_cdf(numerator / s)
        })
    }

    /// The fixed-order Drezner-Wesolowsky rule on `θ ∈ [0, asin ρ]`, the form this module replaces. It is
    /// the positive control of the reference comparison.
    fn fixed_order_drezner_wesolowsky(h: f64, k: f64, rho: f64, order: usize) -> f64 {
        let (nodes, weights) = gauss_legendre(order);
        let alpha = rho.asin();
        let sum = nodes.iter().zip(weights.iter()).fold(0.0, |acc, (&node, &weight)| {
            let sine = (0.5 * alpha * (1.0 + node)).sin();
            acc + weight * ((sine * h * k - 0.5 * (h * h + k * k)) / (1.0 - sine * sine)).exp()
        });
        normal_cdf(h) * normal_cdf(k) + alpha * sum / (4.0 * PI)
    }

    #[test]
    fn core_order_is_the_smallest_meeting_the_bernstein_bound() {
        // The diamond claim rests on `cos t cosh t ≤ 1` over `[0, π/2]`.
        for step in 0..=1000 {
            let t = 0.5 * PI * f64::from(step) / 1000.0;
            assert!(t.cos() * t.cosh() <= 1.0 + f64::EPSILON, "t={t}");
        }
        let radius_squared = core_bernstein_radius_squared();
        let reach = 0.5
            * CORE_MAX_ANGLE
            * (1.0 + (0.5 * (radius_squared + 1.0 / radius_squared)).sqrt());
        assert!((reach - 0.5 * PI).abs() <= 8.0 * f64::EPSILON, "reach={reach}");
        let bound = |order: usize| {
            64.0 / 15.0 * radius_squared.powi(-(order as i32)) / (radius_squared - 1.0)
        };
        let order = core_order();
        assert_eq!(order, 9);
        assert!(bound(order) <= 2.0 * f64::EPSILON);
        assert!(bound(order - 1) > 2.0 * f64::EPSILON);
        assert_eq!(CORE_RULE.weights.len(), order);
    }

    #[test]
    fn core_rule_meets_its_truncation_bound_against_doubled_order() {
        let bounds = [-6.0, -3.0, -1.0, -0.25, 0.0, 0.5, 1.5, 3.0, 6.0];
        let rhos = [-0.5_f64, -0.35, -0.1, 0.1, 0.35, 0.5];
        let doubled = CoreRule::with_order(2 * core_order());
        let weak = CoreRule::with_order(4);
        let mut weak_exceeds = false;
        for &h in &bounds {
            for &k in &bounds {
                for &rho in &rhos {
                    let prefactor = rho.asin().abs() / (4.0 * PI);
                    // Truncation 2ε, plus the rounding of both sums (7ε a node) and of the two additions.
                    let allowance = prefactor
                        * (2.0 + 7.0 * (core_order() + doubled.weights.len()) as f64)
                        * f64::EPSILON
                        + 2.0 * f64::EPSILON;
                    let reference = core_cdf(h, k, rho, &doubled);
                    let production = core_cdf(h, k, rho, &CORE_RULE);
                    assert!(
                        (production - reference).abs() <= allowance,
                        "h={h} k={k} rho={rho} production={production:e} doubled={reference:e}"
                    );
                    weak_exceeds |= (core_cdf(h, k, rho, &weak) - reference).abs() > allowance;
                }
            }
        }
        assert!(weak_exceeds, "an order-4 rule must fail the same allowance somewhere");
    }

    #[test]
    fn cdf_matches_an_independent_conditional_reference_for_every_correlation() {
        assert_eq!(normal_cdf(-40.0), 0.0);
        let bounds = [-8.0, -4.0, -1.5, -0.3, 0.0, 0.7, 2.0, 5.0];
        let rhos = [
            -(1.0 - 1.0e-12),
            -(1.0 - 1.0e-6),
            -0.99,
            -0.925,
            -0.7,
            -0.5000001,
            -0.5,
            -0.3,
            -1.0e-12,
            1.0e-12,
            0.3,
            0.5,
            0.5000001,
            0.7,
            0.925,
            0.99,
            1.0 - 1.0e-6,
            1.0 - 1.0e-12,
        ];
        let check = |h: f64, k: f64, rho: f64| {
            let (reference, magnitude) = conditional_reference(h, k, rho, 40);
            let coarse = conditional_reference(h, k, rho, 20).0;
            let allowance = contract() + 4.0 * f64::EPSILON * magnitude + (reference - coarse).abs();
            let production = bivariate_normal_cdf(h, k, rho).unwrap();
            ((production - reference).abs(), allowance, reference)
        };
        for &h in &bounds {
            for &k in &bounds {
                for &rho in &rhos {
                    let (error, allowance, reference) = check(h, k, rho);
                    assert!(
                        error <= allowance,
                        "h={h} k={k} rho={rho} reference={reference:e} error={error:e} allowance={allowance:e}"
                    );
                }
            }
        }
        // Positive control: a nearly singular correlation with a nearly equal pair. There, the fixed
        // 20-point rule on θ misses the boundary layer at θ = π/2, while the rotated form does not.
        let (h, k, rho) = (0.25, 0.2505, 1.0 - 1.0e-12);
        let (error, allowance, reference) = check(h, k, rho);
        assert!(error <= allowance, "production error={error:e}");
        let fixed = fixed_order_drezner_wesolowsky(h, k, rho, 20);
        assert!(
            (fixed - reference).abs() > allowance,
            "the fixed 20-point rule must fail here: fixed={fixed:e} reference={reference:e}"
        );
    }

    #[test]
    fn reductions_agree_with_the_core_at_the_fixed_point() {
        let bounds = [-5.0, -1.2, 0.0, 0.8, 3.0];
        for &h in &bounds {
            for &k in &bounds {
                let positive = core_cdf(h, k, 0.5, &CORE_RULE);
                let split = h - k;
                let halved = core_cdf(split, k, -0.5, &CORE_RULE) + core_cdf(-split, h, -0.5, &CORE_RULE);
                assert!((positive - halved).abs() <= 1.5 * contract(), "h={h} k={k}");
                let negative = core_cdf(h, k, -0.5, &CORE_RULE);
                let (low, high) = if h <= k { (h, k) } else { (k, h) };
                let difference =
                    core_cdf(h + k, low, 0.5, &CORE_RULE) - core_cdf(h + k, -high, -0.5, &CORE_RULE);
                assert!((negative - difference).abs() <= 1.5 * contract(), "h={h} k={k}");
            }
        }
    }

    #[test]
    fn partials_integrate_back_to_the_distribution_function() {
        let panels = |lower: f64, upper: f64| -> Vec<f64> {
            (0..=8).map(|i| lower + (upper - lower) * f64::from(i) / 8.0).collect()
        };
        let agree = |exact: f64, integrand: &dyn Fn(f64) -> f64, breaks: &[f64]| {
            let (integral, magnitude) = composite(breaks, 40, integrand);
            let coarse = composite(breaks, 20, integrand).0;
            let allowance = 2.0 * contract() + 4.0 * f64::EPSILON * magnitude + (integral - coarse).abs();
            (exact - integral).abs() <= allowance
        };
        let cdf = |h: f64, k: f64, rho: f64| bivariate_normal_cdf(h, k, rho).unwrap();
        let partials = |h: f64, k: f64, rho: f64| bivariate_normal_cdf_partials(h, k, rho).unwrap();
        for &(h0, h1, k, rho) in &[
            (-3.0, 1.0, 0.4, 0.3),
            (-2.0, 2.0, -1.0, -0.8),
            (-1.0, 0.5, 0.5, 0.97),
            (0.0, 3.0, 1.0, -0.99),
        ] {
            let exact = cdf(h1, k, rho) - cdf(h0, k, rho);
            assert!(agree(exact, &|h| partials(h, k, rho).d_h, &panels(h0, h1)), "d_h {h0} {h1} {k} {rho}");
            assert!(agree(exact, &|h| partials(k, h, rho).d_k, &panels(h0, h1)), "d_k {h0} {h1} {k} {rho}");
        }
        for &(h, k, rho0, rho1) in &[
            (0.3, -0.7, -0.6, 0.4),
            (-1.2, -1.0, 0.2, 0.9),
            (1.5, -1.4, -0.95, -0.5),
        ] {
            let exact = cdf(h, k, rho1) - cdf(h, k, rho0);
            assert!(agree(exact, &|rho| partials(h, k, rho).d_rho, &panels(rho0, rho1)), "d_rho {h} {k}");
            assert!(agree(exact, &|rho| bivariate_normal_pdf(h, k, rho).unwrap(), &panels(rho0, rho1)));
        }
        // Positive control: the conditional partial with the correlation's sign flipped must fail.
        let (h0, h1, k, rho) = (-3.0, 1.0, 0.4, 0.3);
        let wrong = |h: f64| conditional_partial(h, k, -rho);
        assert!(!agree(cdf(h1, k, rho) - cdf(h0, k, rho), &wrong, &panels(h0, h1)));
    }

    #[test]
    fn near_singular_correlation_approaches_the_limit_at_the_derived_rate() {
        // |Φ₂(ρ) − Φ₂(±1)| ≤ ∫ dr/(2π√(1 − r²)) = acos|ρ|/2π, and Φ₂ is non-decreasing in ρ.
        for &(h, k) in &[(0.4, 0.4), (-1.0, 2.0), (1.3, -0.2), (-2.5, -2.4), (2.0, -2.0)] {
            let upper = bivariate_normal_cdf(h, k, 1.0).unwrap();
            let lower = bivariate_normal_cdf(h, k, -1.0).unwrap();
            for gap in [1.0e-2_f64, 1.0e-6, 1.0e-10, 1.0e-14] {
                let rho = 1.0 - gap;
                let rate = rho.acos() / TAU + contract();
                let near_upper = bivariate_normal_cdf(h, k, rho).unwrap();
                let near_lower = bivariate_normal_cdf(h, k, -rho).unwrap();
                assert!(near_upper <= upper + contract() && upper - near_upper <= rate, "h={h} k={k} gap={gap}");
                assert!(near_lower >= lower - contract() && near_lower - lower <= rate, "h={h} k={k} gap={gap}");
            }
        }
    }

    #[test]
    fn domain_errors_and_exact_branches() {
        assert_eq!(
            bivariate_normal_cdf(f64::NAN, 0.0, 0.1),
            Err(BivariateNormalError::NanBound { name: "h" })
        );
        assert!(matches!(
            bivariate_normal_cdf(0.0, 0.0, 1.01),
            Err(BivariateNormalError::CorrelationOutsideUnitInterval { .. })
        ));
        assert!(bivariate_normal_cdf(0.0, 0.0, f64::NAN).is_err());
        assert!(bivariate_normal_interval_probability(0.0, f64::NAN, 1.0, 0.2).is_err());
        assert!(matches!(
            bivariate_normal_pdf(0.0, 0.0, 1.0),
            Err(BivariateNormalError::SingularCorrelation { .. })
        ));
        assert!(bivariate_normal_cdf_partials(0.0, 0.0, -1.0).is_err());
        let (h, k) = (-0.35, 0.8);
        assert_eq!(bivariate_normal_cdf(h, k, 0.0).unwrap(), normal_cdf(h) * normal_cdf(k));
        assert_eq!(bivariate_normal_cdf(h, f64::INFINITY, 0.7).unwrap(), normal_cdf(h));
        assert_eq!(bivariate_normal_cdf(f64::NEG_INFINITY, k, -0.7).unwrap(), 0.0);
        // Sheppard's closed form at the origin.
        for rho in [-(1.0_f64 - 1.0e-13), -0.7, -0.2, 0.45, 0.8, 1.0 - 1.0e-13] {
            let expected = 0.25 + rho.asin() / TAU;
            let actual = bivariate_normal_cdf(0.0, 0.0, rho).unwrap();
            assert!((actual - expected).abs() <= contract(), "rho={rho} actual={actual:e}");
        }
    }

    #[test]
    fn interval_probability_retains_upper_tail_mass() {
        let expected = normal_cdf(-10.0) - normal_cdf(-12.0);
        for (h, rho, fraction) in [(f64::INFINITY, 0.3, 1.0), (0.0, 0.0, 0.5), (12.0, 1.0, 1.0), (0.0, -1.0, 1.0)] {
            let actual = bivariate_normal_interval_probability(h, 10.0, 12.0, rho).unwrap();
            assert!((actual / (fraction * expected) - 1.0).abs() <= 8.0 * f64::EPSILON, "h={h} rho={rho}");
        }
        let semi_infinite = bivariate_normal_interval_probability(0.0, 10.0, f64::INFINITY, 0.0).unwrap();
        assert!((semi_infinite / (0.5 * normal_cdf(-10.0)) - 1.0).abs() <= 8.0 * f64::EPSILON);
        let singular = bivariate_normal_cdf(12.0, -10.0, -1.0).unwrap();
        assert!((singular / expected - 1.0).abs() <= 8.0 * f64::EPSILON);
        for &(h, lower, upper, rho) in &[(0.3, -1.0, 0.5, 0.6), (-0.4, 0.2, 2.5, -0.8), (1.1, -3.0, -0.5, 0.95)] {
            let actual = bivariate_normal_interval_probability(h, lower, upper, rho).unwrap();
            let difference = bivariate_normal_cdf(h, upper, rho).unwrap() - bivariate_normal_cdf(h, lower, rho).unwrap();
            assert!((actual - difference).abs() <= 2.0 * contract(), "h={h} [{lower}, {upper}] rho={rho}");
        }
    }
}
