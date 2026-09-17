//! Gaussian smoothing kernels of the SiLU gate, for the finite-response
//! calculation of a SwiGLU block (#2946 R9).
//!
//! A SwiGLU unit (Qwen3 `hidden_act = "silu"`) is `u·(c + A)·s(X)`, with the gate
//! `s(t) = t·σ(t)` (`σ` is the logistic function) and `(A, X)` jointly Gaussian
//! under the declared intervention law. Gaussian integration by parts reduces
//! every moment the response calculation needs to Gaussian expectations of the
//! gate's derivatives, and the reduction is EXACT:
//!
//! * the conditional mean given the retained coordinates,
//!   `E[(α + A)·s(t + W)] = α·T_v s(t) + κ·T_v s'(t)`, with
//!   `T_v f(t) = E f(t + √v·E)`, `v = Var W` and `κ = Cov(A, W)`
//!   ([`gated_conditional_mean`]);
//! * the pair second moment under the coupling `Z' = PZ + (I − P)Z̃`: a six-term
//!   contraction of the table `K_pq = E[s^(p)(X_j)·s^(q)(X_k)]` against the
//!   up-projection covariances ([`gated_pair_second_moment`]).
//!
//! Their derivatives are exact too: `∂_v T_v f = ½·T_v f''` (the heat equation)
//! and `∂_r K_pq = K_{p+1,q+1}` (Price's theorem). One table through `p = 3`
//! carries every value and every leg of the frame gradient.
//!
//! The smoothing itself is not exact. SiLU has no finite closed-form Gaussian
//! smoothing: the logistic-normal integral has only an algebraically convergent
//! Mittag-Leffler series. So `T_v s^(p)` and `K_pq` are computed by the truncated
//! trapezoid rule, and a DERIVED error bound is returned beside every value:
//!
//! * `σ` has poles at `iπ(2m + 1)`, so the integrand in the standard-normal
//!   variable is analytic in the strip `|Im x| < π/√v`;
//! * Poisson summation bounds the error of the infinite trapezoid sum by
//!   `2M/(e^{2πa/h} − 1)` for any strip half-width `a` on which
//!   `∫|w(x + ib)|dx ≤ M` (Trefethen & Weideman, SIAM Review 56, 2014,
//!   Theorem 5.1). Its two-dimensional analogue is `M·(S₁S₂ − 1)` with
//!   `S = coth(πa/h)`;
//! * the dropped nodes are bounded by Gaussian tail integrals of a real-line
//!   envelope.
//!
//! The rule targets f64 resolution at the envelope scale, `ε·R` with
//! `R ≥ E|integrand|`. The step and the node count are the least that meet the
//! bound; no tolerance is chosen by hand. Gauss-Hermite is not used here: for an
//! integrand analytic only in a strip it converges like `e^{−2a√n}`, and at
//! `√v = 5` it needs four to five times the trapezoid's nodes. The derivation is
//! on #2946.

use std::f64::consts::{FRAC_2_SQRT_PI, FRAC_PI_2, PI, SQRT_2};
use std::fmt;

use crate::probability::{normal_cdf, normal_pdf};
use crate::special::logistic;

/// The derivative orders `p = 0..=3` of `s^(p)` that the kernels tabulate: the
/// value, plus the three derivatives read by the Stein reductions and their
/// frame-gradient legs.
pub const SILU_TABLE_ORDERS: usize = 4;

/// `E|x| = √(2/π)` for a standard normal `x`.
const MEAN_ABS_STANDARD_NORMAL: f64 = FRAC_2_SQRT_PI / SQRT_2;

/// A declared Gaussian law that the SiLU kernels cannot evaluate.
#[derive(Clone, Debug, PartialEq)]
pub enum GaussianGatedError {
    /// Every argument must be finite.
    NonFiniteArgument { name: &'static str, value: f64 },
    /// A variance must be non-negative.
    NegativeVariance { name: &'static str, value: f64 },
    /// `r² > v·w`: no Gaussian pair has this covariance.
    CovarianceOutsideCone { v: f64, w: f64, r: f64 },
}

impl fmt::Display for GaussianGatedError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFiniteArgument { name, value } => {
                write!(formatter, "SiLU Gaussian kernel argument {name} is not finite: {value}")
            }
            Self::NegativeVariance { name, value } => {
                write!(formatter, "SiLU Gaussian kernel variance {name} is negative: {value}")
            }
            Self::CovarianceOutsideCone { v, w, r } => write!(
                formatter,
                "SiLU pair covariance r = {r} violates r² ≤ v·w for variances v = {v}, w = {w}"
            ),
        }
    }
}

impl std::error::Error for GaussianGatedError {}

fn require_finite(name: &'static str, value: f64) -> Result<(), GaussianGatedError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(GaussianGatedError::NonFiniteArgument { name, value })
    }
}

fn require_variance(name: &'static str, value: f64) -> Result<(), GaussianGatedError> {
    require_finite(name, value)?;
    if value < 0.0 {
        Err(GaussianGatedError::NegativeVariance { name, value })
    } else {
        Ok(())
    }
}

/// `[s, s', s'', s''']` at a real `u`, where `s(u) = u·σ(u)`.
///
/// `σ' = σ(1 − σ)`, `σ'' = σ'(1 − 2σ)`, `σ''' = σ' − 6σ'²`, and
/// `s^(p) = p·σ^(p−1) + u·σ^(p)`. The complement `1 − σ(u)` is evaluated
/// directly as `σ(−u)`, so every product has factors that are accurate in their
/// own tail, and no entry loses relative accuracy away from its sign changes.
/// This is the one SiLU jet: the Gaussian kernels here and every native forward
/// pass and pullback of a SwiGLU gate read it.
pub fn silu_derivatives(u: f64) -> [f64; SILU_TABLE_ORDERS] {
    let sigma = logistic(u);
    let complement = logistic(-u);
    let first = sigma * complement;
    let second = first * (complement - sigma);
    let third = first * (1.0 - 6.0 * first);
    [
        u * sigma,
        sigma + u * first,
        2.0 * first + u * second,
        3.0 * second + u * third,
    ]
}

/// Uniform bounds of `|s^(p)(z)|` on the band `|Im z| ≤ ω`, for `0 ≤ ω < π`.
///
/// Entry `p` is `(constant, slope)`, with
/// `|s^(p)(z)| ≤ constant + slope·|Re z − center|` for every `z` in the band.
/// Write `u = Re z` and `c = cos(ω/2)`:
///
/// * `|σ(z)| ≤ S`, with `S = 1` for `ω ≤ π/2` and `1/sin ω` above. Reason:
///   `|1 + e^{−z}|² = 1 + ϱ² + 2ϱ·cos ω` is least at `ϱ = −cos ω`.
/// * `|σ'(z)| = 1/(4|cosh(z/2)|²) ≤ C = 1/(4c²)`. Reason:
///   `|cosh(x + iy)|² = sinh²x + cos²y`.
/// * `|1 − 2σ(z)| = |tanh(z/2)| ≤ T = max(1, tan(ω/2))`, so `|σ''| ≤ C·T`.
/// * `|z·σ'(z)| ≤ (|u| + ω)/(u² + 4c²)`, using `sinh²y ≥ y²`. Its maximum over
///   `u` is `D₁ = (√(ω² + 4c²) + ω)/(8c²)`. From there `|z·σ''| ≤ T·D₁`, and,
///   because `σ''' = σ'(1 − 6σ')`, `|z·σ'''| ≤ (1 + 6C)·D₁`.
/// * `s' = σ + zσ'`, `s'' = 2σ' + zσ''` and `s''' = 3σ'' + zσ'''` are bounded
///   by constants. `s = zσ` grows: `|u·σ(z)| ≤ |u|/(e^{|u|} − 1) ≤ 1` for
///   `u ≤ 0`, so `|s(z)| ≤ S·(max(u, 0) + ω) + 1`, and
///   `max(u, 0) ≤ max(center, 0) + |u − center|`.
///
/// Every bound increases with `ω`, so it holds on the whole band. At `ω = 0` it
/// is a real-line envelope.
fn silu_envelope(omega: f64, center: f64) -> [(f64, f64); SILU_TABLE_ORDERS] {
    let logistic_bound = if omega <= FRAC_PI_2 {
        1.0
    } else {
        omega.sin().recip()
    };
    let half_cosine = (0.5 * omega).cos();
    let half_cosine_squared = half_cosine * half_cosine;
    let slope_bound = 0.25 / half_cosine_squared;
    let tanh_bound = (0.5 * omega).tan().max(1.0);
    let z_slope_bound = ((omega * omega + 4.0 * half_cosine_squared).sqrt() + omega)
        / (8.0 * half_cosine_squared);
    [
        (
            logistic_bound * (center.max(0.0) + omega) + 1.0,
            logistic_bound,
        ),
        (logistic_bound + z_slope_bound, 0.0),
        (2.0 * slope_bound + tanh_bound * z_slope_bound, 0.0),
        (
            3.0 * slope_bound * tanh_bound + (1.0 + 6.0 * slope_bound) * z_slope_bound,
            0.0,
        ),
    ]
}

/// `ℓ = ln(1/ε)`: the nats of resolution an f64 result carries.
fn resolution_nats() -> f64 {
    -f64::EPSILON.ln()
}

/// The pole budget `ω* = πℓ/(ℓ + m)` of a strip, where `m = 4` is the highest
/// pole order in the table (that of `s'''`).
///
/// The Poisson bound holds for every strip narrower than the poles. This choice
/// is the leading-order maximiser of the step `2πa/ln(M/τ)` when
/// `M ∝ (π − ω)^{−m}`: it moves only the node count, never the validity.
fn strip_pole_budget() -> f64 {
    let nats = resolution_nats();
    PI * nats / (nats + SILU_TABLE_ORDERS as f64)
}

/// The Gaussian cap `√(2ℓ)` of a strip: `|φ(x + ib)| = φ(x)·e^{b²/2}` makes
/// `ln M` grow like `a²/2`, and `a/(ℓ + a²/2)` is largest at `a = √(2ℓ)`.
fn strip_gaussian_cap() -> f64 {
    (2.0 * resolution_nats()).sqrt()
}

/// One-sided tail integrals `∫_L^∞ φ(x)·xⁿ dx` for `n = 0, 1, 2`, beyond the last
/// node `L`: `Q(L)`, `φ(L)` and `L·φ(L) + Q(L)`.
///
/// They bound the dropped trapezoid terms `h·Σ_{kh > L} φ(kh)·(kh)ⁿ`, because
/// each summand decreases past `√2`. A rule whose last node lies closer in has no
/// tail bound.
fn standard_normal_tails(last_node: f64) -> [f64; 3] {
    if last_node < SQRT_2 {
        return [f64::INFINITY; 3];
    }
    let upper = normal_cdf(-last_node);
    let density = normal_pdf(last_node);
    [upper, density, last_node * density + upper]
}

/// One axis of a truncated trapezoid rule in a standard-normal variable: nodes
/// `k·step` for `|k| ≤ half_count`, weights `step·φ(k·step)`, and the strip
/// half-width on which its Poisson bound is stated.
#[derive(Clone, Copy, Debug)]
struct TrapezoidAxis {
    strip: f64,
    step: f64,
    half_count: usize,
}

impl TrapezoidAxis {
    /// The axis at `step` with the least node count whose tail meets its target.
    /// `exceeds(axis)` must be monotone: true below some count, false from there
    /// on. The tail integrals decrease with the last node, so every caller's
    /// predicate is. The count is bracketed by doubling, then bisected.
    fn least_count(strip: f64, step: f64, exceeds: impl Fn(&TrapezoidAxis) -> bool) -> Self {
        let at = |half_count: usize| TrapezoidAxis {
            strip,
            step,
            half_count,
        };
        let mut fitting = (SQRT_2 / step).ceil() as usize;
        if !exceeds(&at(fitting)) {
            return at(fitting);
        }
        let mut failing = fitting;
        while exceeds(&at(fitting)) {
            failing = fitting;
            fitting *= 2;
        }
        while fitting - failing > 1 {
            let middle = failing + (fitting - failing) / 2;
            if exceeds(&at(middle)) {
                failing = middle;
            } else {
                fitting = middle;
            }
        }
        at(fitting)
    }

    fn node(&self, index: i64) -> f64 {
        index as f64 * self.step
    }

    fn indices(&self) -> std::ops::RangeInclusive<i64> {
        let half_count = self.half_count as i64;
        -half_count..=half_count
    }

    fn last_node(&self) -> f64 {
        self.node(self.half_count as i64)
    }

    /// `S − 1 = 2q/(1 − q)` with `q = e^{−2πa/h}`: the aliased mass of the
    /// infinite sum per unit of strip mass `M`.
    fn aliasing(&self) -> f64 {
        2.0 / (2.0 * PI * self.strip / self.step).exp_m1()
    }

    /// A bound on the full-line sums `h·Σ_k φ(kh)·|kh|ⁿ` for `n = 0, 1, 2`.
    ///
    /// By Poisson summation, `h·Σφ(kh) = Σ_m e^{−2π²m²/h²} ≤ 1 + 2/(e^{2π²/h²} − 1)`.
    /// The terms of `h·Σφ(kh)(kh)² = Σ_m (1 − 4π²m²/h²)·e^{−2π²m²/h²}` are each
    /// at most those of the zeroth sum. And `|x| ≤ (1 + x²)/2`.
    fn full_line_sum(&self) -> f64 {
        1.0 + 2.0 / (2.0 * PI * PI / (self.step * self.step)).exp_m1()
    }
}

/// Gaussian smoothings `T_v s^(p)(t) = E[s^(p)(t + √v·E)]` for `E ~ N(0, 1)` and
/// `p = 0..=3`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SiluSmoothing {
    /// `T_v s^(p)(t)`. Smoothing commutes with differentiation, so this is also
    /// `(T_v s)^(p)(t)`.
    pub derivatives: [f64; SILU_TABLE_ORDERS],
    /// A derived bound on the discretisation and truncation error of each entry.
    /// Rounding is excluded.
    pub quadrature_bound: [f64; SILU_TABLE_ORDERS],
}

/// Gaussian smoothing of the SiLU gate and its first three derivatives.
///
/// `v = 0` is exact. Otherwise the trapezoid rule's step is the largest, and its
/// node count the least, that bound every entry's error by `ε·R_p`, where
/// `R_p ≥ E|s^(p)(X)|` is the envelope scale of the entry.
pub fn silu_gaussian_smoothing(t: f64, v: f64) -> Result<SiluSmoothing, GaussianGatedError> {
    require_finite("t", t)?;
    require_variance("v", v)?;
    if v == 0.0 {
        return Ok(SiluSmoothing {
            derivatives: silu_derivatives(t),
            quadrature_bound: [0.0; SILU_TABLE_ORDERS],
        });
    }
    let axis = smoothing_rule(t, v);
    Ok(SiluSmoothing {
        derivatives: smoothing_sums(t, v, &axis),
        quadrature_bound: smoothing_bound(t, v, &axis),
    })
}

/// `M_p = e^{a²/2}·(A_p + B_p·√(2/π))`: the strip mass `∫|w(x + ib)|dx`, for
/// `|b| ≤ a`, of `w(x) = φ(x)·s^(p)(t + √v·x)`, taken from the band envelope at
/// `ω = √v·a`. At `a = 0` it is the real-line envelope scale
/// `R_p ≥ E|s^(p)(X)|`.
fn smoothing_strip_mass(t: f64, root_v: f64, strip: f64) -> [f64; SILU_TABLE_ORDERS] {
    let gaussian_growth = (0.5 * strip * strip).exp();
    silu_envelope(root_v * strip, t).map(|(constant, slope)| {
        gaussian_growth * (constant + slope * root_v * MEAN_ABS_STANDARD_NORMAL)
    })
}

/// Both tails `2·(A_p·Q(L) + B_p·φ(L))` of the real-line envelope beyond the last
/// node `L`.
fn smoothing_tail(t: f64, root_v: f64, last_node: f64, order: usize) -> f64 {
    let (constant, slope) = silu_envelope(0.0, t)[order];
    let tails = standard_normal_tails(last_node);
    2.0 * (constant * tails[0] + slope * root_v * tails[1])
}

/// The least rule meeting `ε·R_p` in every entry. Half the target goes to the
/// Poisson bound, which fixes the step, and half to the tails, which fixes the
/// node count.
fn smoothing_rule(t: f64, v: f64) -> TrapezoidAxis {
    let root_v = v.sqrt();
    let strip = (strip_pole_budget() / root_v).min(strip_gaussian_cap());
    let strip_mass = smoothing_strip_mass(t, root_v, strip);
    let scale = smoothing_strip_mass(t, root_v, 0.0);
    // 2M/(e^{2πa/h} − 1) ≤ ε·R/2  ⇔  2πa/h ≥ ln(1 + 4M/(ε·R)).
    let aliasing_nats = (0..SILU_TABLE_ORDERS)
        .map(|order| (4.0 * strip_mass[order] / (f64::EPSILON * scale[order])).ln_1p())
        .fold(0.0, f64::max);
    TrapezoidAxis::least_count(strip, 2.0 * PI * strip / aliasing_nats, |axis| {
        (0..SILU_TABLE_ORDERS).any(|order| {
            smoothing_tail(t, root_v, axis.last_node(), order) > 0.5 * f64::EPSILON * scale[order]
        })
    })
}

fn smoothing_bound(t: f64, v: f64, axis: &TrapezoidAxis) -> [f64; SILU_TABLE_ORDERS] {
    let root_v = v.sqrt();
    let strip_mass = smoothing_strip_mass(t, root_v, axis.strip);
    let aliasing = axis.aliasing();
    std::array::from_fn(|order| {
        strip_mass[order] * aliasing + smoothing_tail(t, root_v, axis.last_node(), order)
    })
}

fn smoothing_sums(t: f64, v: f64, axis: &TrapezoidAxis) -> [f64; SILU_TABLE_ORDERS] {
    let root_v = v.sqrt();
    let mut sums = [0.0; SILU_TABLE_ORDERS];
    for index in axis.indices() {
        let node = axis.node(index);
        let weight = axis.step * normal_pdf(node);
        let derivatives = silu_derivatives(t + root_v * node);
        for (sum, derivative) in sums.iter_mut().zip(derivatives) {
            *sum += weight * derivative;
        }
    }
    sums
}

/// Mixed Gaussian moments `E[s^(p)(X)·s^(q)(Y)]` for `p, q = 0..=3`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SiluPairMoments {
    /// `moments[p][q] = E[s^(p)(X)·s^(q)(Y)]`. The pair kernel is
    /// `K_s = moments[0][0]`, and Price's theorem gives `∂_r K_s = moments[1][1]`.
    pub moments: [[f64; SILU_TABLE_ORDERS]; SILU_TABLE_ORDERS],
    /// A derived bound on the discretisation and truncation error of each entry.
    /// Rounding is excluded.
    pub quadrature_bound: [[f64; SILU_TABLE_ORDERS]; SILU_TABLE_ORDERS],
}

/// The SiLU pair kernel table `K_pq(b, c; v, w, r)` of a Gaussian pair, with
/// `X ~ N(b, v)`, `Y ~ N(c, w)` and `Cov(X, Y) = r`.
///
/// The argument order is that of `K_σ(b, c; v, w, r)` on #2946. `r² ≤ v·w` is
/// required exactly: a caller whose inputs round across the cone owns that
/// rounding.
pub fn silu_pair_moments(
    b: f64,
    c: f64,
    v: f64,
    w: f64,
    r: f64,
) -> Result<SiluPairMoments, GaussianGatedError> {
    require_finite("b", b)?;
    require_finite("c", c)?;
    require_variance("v", v)?;
    require_variance("w", w)?;
    require_finite("r", r)?;
    if r * r > v * w {
        return Err(GaussianGatedError::CovarianceOutsideCone { v, w, r });
    }
    if v >= w {
        return Ok(ordered_pair_moments(b, c, v, w, r));
    }
    let swapped = ordered_pair_moments(c, b, w, v, r);
    Ok(SiluPairMoments {
        moments: transpose(&swapped.moments),
        quadrature_bound: transpose(&swapped.quadrature_bound),
    })
}

fn transpose(
    table: &[[f64; SILU_TABLE_ORDERS]; SILU_TABLE_ORDERS],
) -> [[f64; SILU_TABLE_ORDERS]; SILU_TABLE_ORDERS] {
    std::array::from_fn(|p| std::array::from_fn(|q| table[q][p]))
}

/// The Cholesky coordinates of the pair: `X = b + √v·e₁` and
/// `Y = c + β·e₁ + γ·e₂`, with `β = r/√v` and `γ² = w − r²/v`. The larger
/// variance comes first, so the node product scales with `√(v·w − r²)`.
#[derive(Clone, Copy, Debug)]
struct PairCoordinates {
    b: f64,
    c: f64,
    root_v: f64,
    beta: f64,
    gamma: f64,
}

impl PairCoordinates {
    /// The coordinates of a law with `v > 0`, `w ≤ v` and `r² ≤ v·w`.
    fn of(b: f64, c: f64, v: f64, w: f64, r: f64) -> Self {
        let root_v = v.sqrt();
        PairCoordinates {
            b,
            c,
            root_v,
            beta: r / root_v,
            gamma: ((v * w - r * r) / v).sqrt(),
        }
    }
}

/// The tensor trapezoid rule of a pair.
#[derive(Clone, Copy, Debug)]
struct PairRule {
    first: TrapezoidAxis,
    /// `None` when `γ = 0`: then `Y` depends on `e₁` alone, and `e₂` integrates
    /// out exactly.
    second: Option<TrapezoidAxis>,
}

fn ordered_pair_moments(b: f64, c: f64, v: f64, w: f64, r: f64) -> SiluPairMoments {
    if v == 0.0 {
        // `w ≤ v` and `r² ≤ v·w`, so both coordinates are constants.
        let first = silu_derivatives(b);
        let second = silu_derivatives(c);
        return SiluPairMoments {
            moments: std::array::from_fn(|p| std::array::from_fn(|q| first[p] * second[q])),
            quadrature_bound: [[0.0; SILU_TABLE_ORDERS]; SILU_TABLE_ORDERS],
        };
    }
    let coordinates = PairCoordinates::of(b, c, v, w, r);
    let rule = pair_rule(&coordinates);
    SiluPairMoments {
        moments: pair_sums(&coordinates, &rule),
        quadrature_bound: pair_bound(&coordinates, &rule),
    }
}

/// The coefficients `[c₀₀, c₁₀, c₂₀, c₀₁, c₁₁]` of the product envelope
/// `|s^(p)(X)·s^(q)(Y)| ≤ c₀₀ + c₁₀|x₁| + c₂₀x₁² + c₀₁|x₂| + c₁₁|x₁||x₂|`, on the
/// polystrip with bands `ω_X = √v·a₁` and `ω_Y = |β|·a₁ + γ·a₂`.
fn pair_envelope(
    coordinates: &PairCoordinates,
    first_strip: f64,
    second_strip: f64,
) -> [[[f64; 5]; SILU_TABLE_ORDERS]; SILU_TABLE_ORDERS] {
    let slope = coordinates.beta.abs();
    let first = silu_envelope(coordinates.root_v * first_strip, coordinates.b);
    let second = silu_envelope(
        slope * first_strip + coordinates.gamma * second_strip,
        coordinates.c,
    );
    std::array::from_fn(|p| {
        std::array::from_fn(|q| {
            let (first_constant, first_slope) = first[p];
            let (second_constant, second_slope) = second[q];
            let first_scale = first_slope * coordinates.root_v;
            let second_along_first = second_slope * slope;
            let second_along_second = second_slope * coordinates.gamma;
            [
                first_constant * second_constant,
                first_constant * second_along_first + first_scale * second_constant,
                first_scale * second_along_first,
                first_constant * second_along_second,
                first_scale * second_along_second,
            ]
        })
    })
}

/// The mean of the product envelope, using `E|x| = √(2/π)`, `E x² = 1` and
/// `E|x₁||x₂| = 2/π`.
fn pair_envelope_mean(coefficients: &[f64; 5]) -> f64 {
    let mean_abs = MEAN_ABS_STANDARD_NORMAL;
    coefficients[0]
        + (coefficients[1] + coefficients[3]) * mean_abs
        + coefficients[2]
        + coefficients[4] * mean_abs * mean_abs
}

/// The mass of the dropped nodes from the real-line product envelope, as
/// `[region |x₁| > L₁, region |x₂| > L₂, their sum]`. The union bound counts the
/// overlap twice. Inside each region the other axis is summed over the full line.
fn pair_tails(coefficients: &[f64; 5], rule: &PairRule) -> [f64; 3] {
    let [constant, along_first, first_squared, along_second, cross] = *coefficients;
    let first_tails = standard_normal_tails(rule.first.last_node());
    let first_sum = rule.first.full_line_sum();
    let (second_tails, second_sum) = match &rule.second {
        Some(axis) => (standard_normal_tails(axis.last_node()), axis.full_line_sum()),
        None => ([0.0; 3], 1.0),
    };
    let first_region = 2.0
        * second_sum
        * ((constant + along_second) * first_tails[0]
            + (along_first + cross) * first_tails[1]
            + first_squared * first_tails[2]);
    let second_region = 2.0
        * first_sum
        * ((constant + along_first + first_squared) * second_tails[0]
            + (along_second + cross) * second_tails[1]);
    [first_region, second_region, first_region + second_region]
}

/// The strip half-widths of a pair. The poles of `s(X)` need `√v·a₁ < π`, and
/// those of `s(Y)` need `|β|·a₁ + γ·a₂ < π`. The node product scales as
/// `1/(a₁a₂)`, so `a₁` takes the balanced share `|β|·a₁ = ω*/2`, unless the
/// Gaussian cap on `a₂` leaves it more.
fn pair_strips(coordinates: &PairCoordinates) -> (f64, f64) {
    let budget = strip_pole_budget();
    let cap = strip_gaussian_cap();
    let slope = coordinates.beta.abs();
    let gamma = coordinates.gamma;
    let mut first_strip = (budget / coordinates.root_v).min(cap);
    if slope > 0.0 {
        let share = if gamma > 0.0 {
            (0.5 * budget / slope).max((budget - gamma * cap) / slope)
        } else {
            budget / slope
        };
        first_strip = first_strip.min(share);
    }
    let second_strip = if gamma > 0.0 {
        ((budget - slope * first_strip) / gamma).min(cap)
    } else {
        0.0
    };
    (first_strip, second_strip)
}

/// `M_pq = e^{(a₁² + a₂²)/2}·E[product envelope on the polystrip]`.
fn pair_strip_mass(
    coordinates: &PairCoordinates,
    first_strip: f64,
    second_strip: f64,
) -> [[f64; SILU_TABLE_ORDERS]; SILU_TABLE_ORDERS] {
    let growth = (0.5 * (first_strip * first_strip + second_strip * second_strip)).exp();
    pair_envelope(coordinates, first_strip, second_strip)
        .map(|row| row.map(|coefficients| growth * pair_envelope_mean(&coefficients)))
}

/// The least tensor rule meeting `ε·R_pq` in every entry. Half the target goes to
/// the Poisson bound and a quarter to each truncation region.
fn pair_rule(coordinates: &PairCoordinates) -> PairRule {
    let (first_strip, second_strip) = pair_strips(coordinates);
    let two_axes = coordinates.gamma > 0.0;
    let strip_mass = pair_strip_mass(coordinates, first_strip, second_strip);
    let scale_envelope = pair_envelope(coordinates, 0.0, 0.0);
    let scale = scale_envelope.map(|row| row.map(|coefficients| pair_envelope_mean(&coefficients)));
    // M·(S₁S₂ − 1) ≤ ε·R/2 with one q on both axes. S₁S₂ − 1 = 4q/(1 − q)² equals
    // θ at q = θ/(√(1 + θ) + 1)². With e₂ absent, 2q/(1 − q) = θ at q = θ/(2 + θ).
    let mut aliasing_nats = 0.0f64;
    for (mass_row, scale_row) in strip_mass.iter().zip(&scale) {
        for (mass, entry_scale) in mass_row.iter().zip(scale_row) {
            let theta = 0.5 * f64::EPSILON * entry_scale / mass;
            let nats = if two_axes {
                2.0 * ((1.0 + theta).sqrt() + 1.0).ln() - theta.ln()
            } else {
                (2.0 / theta).ln_1p()
            };
            aliasing_nats = aliasing_nats.max(nats);
        }
    }
    let region_exceeds = |rule: &PairRule, region: usize| {
        (0..SILU_TABLE_ORDERS).any(|p| {
            (0..SILU_TABLE_ORDERS).any(|q| {
                pair_tails(&scale_envelope[p][q], rule)[region] > 0.25 * f64::EPSILON * scale[p][q]
            })
        })
    };
    // Each truncation region depends on its own axis's count and only on the other
    // axis's step, so the two counts are found independently.
    let first_step = 2.0 * PI * first_strip / aliasing_nats;
    let second_axis = two_axes.then(|| TrapezoidAxis {
        strip: second_strip,
        step: 2.0 * PI * second_strip / aliasing_nats,
        half_count: 0,
    });
    let first = TrapezoidAxis::least_count(first_strip, first_step, |axis| {
        region_exceeds(
            &PairRule {
                first: *axis,
                second: second_axis,
            },
            0,
        )
    });
    let second = second_axis.map(|stepped| {
        TrapezoidAxis::least_count(stepped.strip, stepped.step, |axis| {
            region_exceeds(
                &PairRule {
                    first,
                    second: Some(*axis),
                },
                1,
            )
        })
    });
    PairRule { first, second }
}

fn pair_bound(
    coordinates: &PairCoordinates,
    rule: &PairRule,
) -> [[f64; SILU_TABLE_ORDERS]; SILU_TABLE_ORDERS] {
    let second_strip = rule.second.map_or(0.0, |axis| axis.strip);
    let strip_mass = pair_strip_mass(coordinates, rule.first.strip, second_strip);
    let scale_envelope = pair_envelope(coordinates, 0.0, 0.0);
    let first_aliasing = rule.first.aliasing();
    let second_aliasing = rule.second.map_or(0.0, |axis| axis.aliasing());
    let aliasing = first_aliasing + second_aliasing + first_aliasing * second_aliasing;
    std::array::from_fn(|p| {
        std::array::from_fn(|q| {
            strip_mass[p][q] * aliasing + pair_tails(&scale_envelope[p][q], rule)[2]
        })
    })
}

fn pair_sums(
    coordinates: &PairCoordinates,
    rule: &PairRule,
) -> [[f64; SILU_TABLE_ORDERS]; SILU_TABLE_ORDERS] {
    let mut sums = [[0.0; SILU_TABLE_ORDERS]; SILU_TABLE_ORDERS];
    for first_index in rule.first.indices() {
        let node = rule.first.node(first_index);
        let weight = rule.first.step * normal_pdf(node);
        let first = silu_derivatives(coordinates.b + coordinates.root_v * node);
        let center = coordinates.c + coordinates.beta * node;
        let inner = match &rule.second {
            Some(axis) => {
                let mut inner = [0.0; SILU_TABLE_ORDERS];
                for second_index in axis.indices() {
                    let second_node = axis.node(second_index);
                    let second_weight = axis.step * normal_pdf(second_node);
                    let second = silu_derivatives(center + coordinates.gamma * second_node);
                    for (sum, value) in inner.iter_mut().zip(second) {
                        *sum += second_weight * value;
                    }
                }
                inner
            }
            None => silu_derivatives(center),
        };
        for (row, first_value) in sums.iter_mut().zip(first) {
            for (sum, inner_value) in row.iter_mut().zip(inner) {
                *sum += weight * first_value * inner_value;
            }
        }
    }
    sums
}

/// A quantity together with a derived bound on its quadrature error. Rounding is
/// excluded.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BoundedValue {
    pub value: f64,
    pub quadrature_bound: f64,
}

/// The conditional mean of one gated unit, with its exact first derivatives.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GatedMean {
    /// `α·T_v s(t) + κ·T_v s'(t)`.
    pub value: BoundedValue,
    /// `∂/∂α = T_v s(t)`.
    pub d_alpha: BoundedValue,
    /// `∂/∂t = α·T_v s'(t) + κ·T_v s''(t)`.
    pub d_t: BoundedValue,
    /// `∂/∂κ = T_v s'(t)`.
    pub d_kappa: BoundedValue,
    /// `∂/∂v = ½·(α·T_v s''(t) + κ·T_v s'''(t))`, by the heat equation.
    pub d_v: BoundedValue,
}

/// `E[(α + A)·s(t + W)]` for a zero-mean Gaussian pair `(A, W)` with `Var W = v`
/// and `Cov(A, W) = κ`.
///
/// Take a gated unit `(aᵀz + c)·s(wᵀz + b)` with `Z ~ N(0, I)` and retained
/// projector `P`. Its conditional mean given `PZ = Pz` is this function at
/// `α = aᵀPz + c`, `t = wᵀPz + b`, `v = wᵀ(I − P)w` and `κ = aᵀ(I − P)w`:
/// `(I − P)Z` is independent of `PZ`, and Stein's lemma
/// `E[A·h(W)] = κ·E h'(W)` gives the second term exactly. `P = 0` gives the
/// unconditional mean.
pub fn gated_conditional_mean(
    alpha: f64,
    t: f64,
    v: f64,
    kappa: f64,
) -> Result<GatedMean, GaussianGatedError> {
    require_finite("alpha", alpha)?;
    require_finite("kappa", kappa)?;
    let smoothing = silu_gaussian_smoothing(t, v)?;
    let entry = |order: usize| BoundedValue {
        value: smoothing.derivatives[order],
        quadrature_bound: smoothing.quadrature_bound[order],
    };
    let stein = |order: usize| BoundedValue {
        value: alpha * smoothing.derivatives[order] + kappa * smoothing.derivatives[order + 1],
        quadrature_bound: alpha.abs() * smoothing.quadrature_bound[order]
            + kappa.abs() * smoothing.quadrature_bound[order + 1],
    };
    let curvature = stein(2);
    Ok(GatedMean {
        value: stein(0),
        d_alpha: entry(0),
        d_t: stein(1),
        d_kappa: entry(1),
        d_v: BoundedValue {
            value: 0.5 * curvature.value,
            quadrature_bound: 0.5 * curvature.quadrature_bound,
        },
    })
}

/// The up-projection side of a pair of gated units `(c_j + A_j)·s(X_j)` and
/// `(c_k + A_k)·s(X_k)`: the offsets, and every covariance of the zero-mean `A`
/// with the gates and with each other. The gates' own law is the argument of
/// [`silu_pair_moments`].
///
/// Under the coupling `Z' = PZ + (I − P)Z̃` of `V(P)`, with `A_j = a_jᵀZ`,
/// `A_k = a_kᵀZ'`, `X_j = b_j + w_jᵀZ` and `X_k = b_k + w_kᵀZ'`, the fields are:
/// `kappa_j = a_jᵀw_j`, `kappa_k = a_kᵀw_k`, `lambda_jk = a_jᵀPw_k`,
/// `lambda_kj = a_kᵀPw_j` and `rho = a_jᵀPa_k`. The gates have `v = ‖w_j‖²`,
/// `w = ‖w_k‖²` and `r = w_jᵀPw_k`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GatedPairLaw {
    pub offset_j: f64,
    pub offset_k: f64,
    pub kappa_j: f64,
    pub kappa_k: f64,
    pub lambda_jk: f64,
    pub lambda_kj: f64,
    pub rho: f64,
}

impl GatedPairLaw {
    /// The six Stein coefficients `(p, q, coefficient)` of `K_pq` in the pair second
    /// moment.
    pub fn stein_coefficients(&self) -> [(usize, usize, f64); 6] {
        [
            (0, 0, self.offset_j * self.offset_k + self.rho),
            (
                1,
                0,
                self.offset_j * self.lambda_kj + self.offset_k * self.kappa_j,
            ),
            (
                0,
                1,
                self.offset_j * self.kappa_k + self.offset_k * self.lambda_jk,
            ),
            (2, 0, self.kappa_j * self.lambda_kj),
            (
                1,
                1,
                self.kappa_j * self.kappa_k + self.lambda_jk * self.lambda_kj,
            ),
            (0, 2, self.lambda_jk * self.kappa_k),
        ]
    }
}

/// The second moment of a pair of gated units, with its exact legs in the
/// covariances that move with `P`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GatedPairMoment {
    /// `E[(c_j + A_j)·s(X_j)·(c_k + A_k)·s(X_k)]`.
    pub value: BoundedValue,
    /// `∂/∂r`: the same contraction one Price order up.
    pub d_r: BoundedValue,
    /// `∂/∂ρ = K₀₀`.
    pub d_rho: BoundedValue,
    /// `∂/∂λ_jk = c_k·K₀₁ + λ_kj·K₁₁ + κ_k·K₀₂`.
    pub d_lambda_jk: BoundedValue,
    /// `∂/∂λ_kj = c_j·K₁₀ + κ_j·K₂₀ + λ_jk·K₁₁`.
    pub d_lambda_kj: BoundedValue,
}

/// The second moment of a pair of gated units, contracted from the gates' mixed
/// moments.
///
/// Gaussian integration by parts, applied twice to `h(X) = s(X_j)·s(X_k)`, gives
/// `E[A·h] = Σ_m Cov(A, X_m)·E ∂_m h` and
/// `E[A_jA_k·h] = Cov(A_j, A_k)·E h + Σ_{m,n} Cov(A_j, X_m)·Cov(A_k, X_n)·E ∂_m∂_n h`.
/// These hold for any covariance, singular ones included. With
/// `K_pq = E[s^(p)(X_j)·s^(q)(X_k)]`:
///
/// `M = (c_jc_k + ρ)·K₀₀ + (c_jλ_kj + c_kκ_j)·K₁₀ + (c_jκ_k + c_kλ_jk)·K₀₁
///    + κ_jλ_kj·K₂₀ + (κ_jκ_k + λ_jkλ_kj)·K₁₁ + λ_jkκ_k·K₀₂`.
///
/// Every covariance that moves with `P` enters linearly or bilinearly, so the
/// returned legs are exact.
pub fn gated_pair_second_moment(law: &GatedPairLaw, moments: &SiluPairMoments) -> GatedPairMoment {
    let contract = |terms: &[(usize, usize, f64)], shift: usize| BoundedValue {
        value: terms
            .iter()
            .map(|&(p, q, coefficient)| coefficient * moments.moments[p + shift][q + shift])
            .sum(),
        quadrature_bound: terms
            .iter()
            .map(|&(p, q, coefficient)| {
                coefficient.abs() * moments.quadrature_bound[p + shift][q + shift]
            })
            .sum(),
    };
    let stein = law.stein_coefficients();
    GatedPairMoment {
        value: contract(&stein, 0),
        d_r: contract(&stein, 1),
        d_rho: contract(&[(0, 0, 1.0)], 0),
        d_lambda_jk: contract(
            &[
                (0, 1, law.offset_k),
                (1, 1, law.lambda_kj),
                (0, 2, law.kappa_k),
            ],
            0,
        ),
        d_lambda_kj: contract(
            &[
                (1, 0, law.offset_j),
                (2, 0, law.kappa_j),
                (1, 1, law.lambda_jk),
            ],
            0,
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::probability::standard_normal_quantile;
    use crate::special::gauss_legendre;

    const ORDERS: usize = SILU_TABLE_ORDERS;

    /// Declared family-wise false-alarm rate of a Monte Carlo agreement test: at
    /// most one noise-driven red per thousand independent seeds.
    const MONTE_CARLO_FALSE_ALARM_RATE: f64 = 1.0e-3;

    /// Laws spanning the regimes: a wide gate at the knee; gates deep in either
    /// tail; a narrow gate, where the Gaussian cap binds; a very wide gate, where
    /// the strip is narrow; and a nearly degenerate gate.
    const SMOOTHING_LAWS: [(f64, f64); 6] = [
        (0.0, 25.0),
        (3.0, 1.0),
        (-6.0, 4.0),
        (12.0, 0.25),
        (-1.5, 100.0),
        (0.7, 1.0e-4),
    ];

    /// The reference's truncation half-width. Its tail bound is added to each
    /// tolerance.
    const REFERENCE_HALF_WIDTH: f64 = 12.0;

    /// Real-line bounds of `|s^(n)(u)|`. Orders `0..=3` are `silu_envelope(0, u)`.
    /// Order 4 comes from `s'''' = 4σ''' + uσ''''` with `σ'''' = σ''(1 − 12σ')`,
    /// giving `|s''''| ≤ 4·(C + 6C²) + T·(1 + 12C)·D₁ = 3.5` at `ω = 0`. Higher
    /// orders use Cauchy's estimate on the circle of radius `π/2`, where `|σ| ≤ 1`
    /// and `|s(z)| ≤ max(u, 0) + π + 1`:
    /// `|s^(n)(u)| ≤ n!·(2/π)ⁿ·(max(u, 0) + π + 1)`.
    fn real_envelope(order: usize, u: f64) -> f64 {
        if order < ORDERS {
            return silu_envelope(0.0, u)[order].0;
        }
        if order == ORDERS {
            return 3.5;
        }
        cauchy_factor(order) * (u.max(0.0) + PI + 1.0)
    }

    /// `real_envelope(n, u)` for `n = 0..=4`, from one envelope evaluation.
    fn real_envelopes(u: f64) -> [f64; ORDERS + 1] {
        let base = silu_envelope(0.0, u);
        [base[0].0, base[1].0, base[2].0, base[3].0, real_envelope(ORDERS, u)]
    }

    fn cauchy_factor(order: usize) -> f64 {
        let factorial: f64 = (1..=order).map(|k| k as f64).product();
        factorial * (2.0 / PI).powi(order as i32)
    }

    /// `E[envelope_n(X)²]` for `X ~ N(t, v)`, from `max(X, 0) ≤ max(t, 0) + |X − t|`.
    fn envelope_second_moment(order: usize, t: f64, v: f64) -> f64 {
        let (constant, slope) = if order < ORDERS {
            silu_envelope(0.0, t)[order]
        } else if order == ORDERS {
            (3.5, 0.0)
        } else {
            let factor = cauchy_factor(order);
            (factor * (t.max(0.0) + PI + 1.0), factor)
        };
        constant * constant
            + 2.0 * constant * slope * v.sqrt() * MEAN_ABS_STANDARD_NORMAL
            + slope * slope * v
    }

    /// First-order rounding of a computed sum `Σ_k W_k·s^(p)(u_k)` with
    /// `u_k = t + √v·x_k`:
    /// * recursive summation adds at most `(N − 1)ε·Σ|terms|`;
    /// * a weight carries `(3 + x²)ε` relative, because `x` rounds and `φ`
    ///   magnifies a relative change of `x` by `x²`;
    /// * `u` carries `ε·(|t| + 2√v|x|)` absolute, which `s^(p)` magnifies by
    ///   `|s^(p+1)|`;
    /// * the derivative formula adds `8ε` relative over its operations.
    ///
    /// Terms are bounded by the real-line envelopes. `weight_ulps` carries a
    /// quadrature rule's own weight error.
    fn sum_rounding(
        t: f64,
        root_v: f64,
        weight_ulps: f64,
        weighted_nodes: &[(f64, f64)],
    ) -> [f64; ORDERS] {
        let count = weighted_nodes.len() as f64;
        let mut allowance = [0.0; ORDERS];
        for &(node, weight) in weighted_nodes {
            let u = t + root_v * node;
            for (order, entry) in allowance.iter_mut().enumerate() {
                *entry += weight.abs()
                    * ((count + 10.0 + weight_ulps + node * node) * real_envelope(order, u)
                        + (t.abs() + 2.0 * root_v * node.abs()) * real_envelope(order + 1, u));
            }
        }
        allowance.map(|entry| f64::EPSILON * entry)
    }

    fn trapezoid_nodes(axis: &TrapezoidAxis) -> Vec<(f64, f64)> {
        axis.indices()
            .map(|index| {
                let node = axis.node(index);
                (node, axis.step * normal_pdf(node))
            })
            .collect()
    }

    fn smoothing_rounding(t: f64, v: f64, axis: &TrapezoidAxis) -> [f64; ORDERS] {
        sum_rounding(t, v.sqrt(), 0.0, &trapezoid_nodes(axis))
    }

    /// Composite Gauss-Legendre nodes and weights for `∫_{−L}^{L} φ(x)·f(x)dx` on
    /// `panels` equal panels.
    fn legendre_panels(
        half_width: f64,
        panels: usize,
        rule: &(Vec<f64>, Vec<f64>),
    ) -> Vec<(f64, f64)> {
        let panel_half = half_width / panels as f64;
        let mut nodes = Vec::with_capacity(panels * rule.0.len());
        for panel in 0..panels {
            let center = -half_width + panel_half * (2 * panel + 1) as f64;
            for (abscissa, weight) in rule.0.iter().zip(&rule.1) {
                let node = center + panel_half * abscissa;
                nodes.push((node, panel_half * weight * normal_pdf(node)));
            }
        }
        nodes
    }

    /// Panel half-width of a Gauss-Legendre reference: half the distance from the
    /// real axis to the nearest pole, and at most ½, so `φ` is resolved. Each
    /// panel's Bernstein ellipse of parameter 4 then clears the poles, and 16 nodes
    /// leave about `4^{−32} ≈ 5e−20` of the panel's mass. The tests confirm
    /// convergence against doubled panels.
    fn legendre_panel_half_width(pole_distance: f64) -> f64 {
        (0.5 * pole_distance).min(0.5)
    }

    /// The composite Gauss-Legendre reference smoothing on `[−L, L]`, with its
    /// rounding. `gauss_legendre(16)` weights hold about `9ε`.
    fn legendre_smoothing(
        t: f64,
        v: f64,
        panels: usize,
        rule: &(Vec<f64>, Vec<f64>),
    ) -> ([f64; ORDERS], [f64; ORDERS]) {
        let root_v = v.sqrt();
        let nodes = legendre_panels(REFERENCE_HALF_WIDTH, panels, rule);
        let mut sums = [0.0; ORDERS];
        for &(node, weight) in &nodes {
            for (sum, value) in sums.iter_mut().zip(silu_derivatives(t + root_v * node)) {
                *sum += weight * value;
            }
        }
        (sums, sum_rounding(t, root_v, 9.0, &nodes))
    }

    #[test]
    fn silu_smoothing_meets_its_derived_bound_against_an_independent_reference() {
        let legendre = gauss_legendre(16);
        for (t, v) in SMOOTHING_LAWS {
            let root_v = v.sqrt();
            let axis = smoothing_rule(t, v);
            let values = smoothing_sums(t, v, &axis);
            let bound = smoothing_bound(t, v, &axis);
            let rounding = smoothing_rounding(t, v, &axis);
            let scale = smoothing_strip_mass(t, root_v, 0.0);
            let panels =
                (REFERENCE_HALF_WIDTH / legendre_panel_half_width(PI / root_v)).ceil() as usize;
            let (reference, reference_rounding) = legendre_smoothing(t, v, panels, &legendre);
            let (refined, refined_rounding) = legendre_smoothing(t, v, 2 * panels, &legendre);
            // ℓ → h = 2πa/ℓ → 2πa/h round-trips through five operations, moving ℓ by at
            // most 5ε relative, and e^ℓ by at most 5εℓ; ln_1p adds ℓ·ε more.
            let aliasing_nats = 2.0 * PI * axis.strip / axis.step;
            let round_trip = 1.0 + 8.0 * f64::EPSILON * aliasing_nats;
            for order in 0..ORDERS {
                let reference_tail = smoothing_tail(t, root_v, REFERENCE_HALF_WIDTH, order);
                let reference_change = (reference[order] - refined[order]).abs();
                assert!(
                    reference_change <= reference_rounding[order] + refined_rounding[order],
                    "t={t} v={v} p={order}: doubling the reference panels moved it by \
                     {reference_change:e} > rounding {:e} + {:e}",
                    reference_rounding[order],
                    refined_rounding[order]
                );
                let error = (values[order] - reference[order]).abs();
                let allowance =
                    bound[order] + rounding[order] + reference_rounding[order] + reference_tail;
                assert!(
                    error <= allowance,
                    "t={t} v={v} p={order}: |rule − reference| = {error:e} exceeds bound {:e} \
                     + rounding {:e} + reference rounding {:e} + reference tail {reference_tail:e}",
                    bound[order],
                    rounding[order],
                    reference_rounding[order]
                );
                assert!(
                    bound[order] <= f64::EPSILON * scale[order] * round_trip,
                    "t={t} v={v} p={order}: bound {:e} misses the contract ε·R = {:e}",
                    bound[order],
                    f64::EPSILON * scale[order]
                );
            }
        }
    }

    #[test]
    fn silu_smoothing_is_stable_under_step_halving_and_detects_an_under_resolved_rule() {
        for (t, v) in SMOOTHING_LAWS {
            let root_v = v.sqrt();
            let axis = smoothing_rule(t, v);
            let halved = TrapezoidAxis {
                step: 0.5 * axis.step,
                half_count: 2 * axis.half_count,
                ..axis
            };
            let coarse = TrapezoidAxis {
                step: 4.0 * axis.step,
                half_count: axis.half_count.div_ceil(4),
                ..axis
            };
            let values = smoothing_sums(t, v, &axis);
            let bound = smoothing_bound(t, v, &axis);
            let rounding = smoothing_rounding(t, v, &axis);
            let halved_values = smoothing_sums(t, v, &halved);
            let halved_bound = smoothing_bound(t, v, &halved);
            let halved_rounding = smoothing_rounding(t, v, &halved);
            let coarse_values = smoothing_sums(t, v, &coarse);
            let coarse_bound = smoothing_bound(t, v, &coarse);
            let coarse_rounding = smoothing_rounding(t, v, &coarse);
            let scale = smoothing_strip_mass(t, root_v, 0.0);
            let mut largest_excess = f64::NEG_INFINITY;
            for order in 0..ORDERS {
                let halving_change = (values[order] - halved_values[order]).abs();
                let halving_allowance =
                    bound[order] + halved_bound[order] + rounding[order] + halved_rounding[order];
                assert!(
                    halving_change <= halving_allowance,
                    "t={t} v={v} p={order}: halving the step moved the value by {halving_change:e} \
                     > {halving_allowance:e}"
                );
                let coarse_error = (coarse_values[order] - values[order]).abs();
                let coarse_allowance =
                    coarse_bound[order] + bound[order] + rounding[order] + coarse_rounding[order];
                assert!(
                    coarse_error <= coarse_allowance,
                    "t={t} v={v} p={order}: the quarter-density rule is off by {coarse_error:e}, \
                     beyond its own bound {coarse_allowance:e}"
                );
                largest_excess = largest_excess.max(
                    coarse_error
                        - (f64::EPSILON * scale[order] + rounding[order] + coarse_rounding[order]),
                );
            }
            assert!(
                largest_excess > 0.0,
                "t={t} v={v}: the quarter-density rule met the production contract, so this \
                 comparison cannot detect an under-resolved rule (largest excess {largest_excess:e})"
            );
        }
    }

    #[test]
    fn silu_smoothing_holds_its_exact_symmetries() {
        // s(t) − s(−t) = t, so s is t/2 plus an even function. Hence
        // T_v s(t) − T_v s(−t) = t, T_v s'(t) + T_v s'(−t) = 1, T_v s'' is even and
        // T_v s''' is odd.
        for (t, v) in SMOOTHING_LAWS {
            let plus_axis = smoothing_rule(t, v);
            let minus_axis = smoothing_rule(-t, v);
            let plus = smoothing_sums(t, v, &plus_axis);
            let minus = smoothing_sums(-t, v, &minus_axis);
            let plus_bound = smoothing_bound(t, v, &plus_axis);
            let minus_bound = smoothing_bound(-t, v, &minus_axis);
            let plus_rounding = smoothing_rounding(t, v, &plus_axis);
            let minus_rounding = smoothing_rounding(-t, v, &minus_axis);
            let residuals = [
                plus[0] - minus[0] - t,
                plus[1] + minus[1] - 1.0,
                plus[2] - minus[2],
                plus[3] + minus[3],
            ];
            for (order, residual) in residuals.iter().enumerate() {
                let allowance = plus_bound[order]
                    + minus_bound[order]
                    + plus_rounding[order]
                    + minus_rounding[order]
                    + f64::EPSILON * (t.abs() + 1.0);
                assert!(
                    residual.abs() <= allowance,
                    "t={t} v={v} p={order}: symmetry residual {residual:e} exceeds {allowance:e}"
                );
            }
        }
        let exact = silu_gaussian_smoothing(1.25, 0.0).expect("a degenerate gate is exact");
        assert_eq!(exact.derivatives, silu_derivatives(1.25));
        assert_eq!(exact.quadrature_bound, [0.0; ORDERS]);
        assert!(matches!(
            silu_gaussian_smoothing(0.0, -1.0),
            Err(GaussianGatedError::NegativeVariance { .. })
        ));
        assert!(matches!(
            silu_pair_moments(0.0, 0.0, 1.0, 1.0, 1.5),
            Err(GaussianGatedError::CovarianceOutsideCone { .. })
        ));
    }

    /// First-order rounding of a computed pair table. Recursive summation over the
    /// `N₁·N₂` terms, the weights of both axes and both derivative formulas add the
    /// same counts as `sum_rounding`, with `weight_ulps` for each axis's weights.
    /// Each coordinate's perturbation is magnified by the next derivative of its own
    /// factor.
    fn pair_rounding(
        coordinates: &PairCoordinates,
        weight_ulps: f64,
        first_nodes: &[(f64, f64)],
        second_nodes: &[(f64, f64)],
    ) -> [[f64; ORDERS]; ORDERS] {
        let relative_base = (first_nodes.len() * second_nodes.len()) as f64 + 22.0 + 2.0 * weight_ulps;
        let slope = coordinates.beta.abs();
        let mut allowance = [[0.0; ORDERS]; ORDERS];
        for &(first_node, first_weight) in first_nodes {
            let first_envelopes = real_envelopes(coordinates.b + coordinates.root_v * first_node);
            let first_perturbation =
                coordinates.b.abs() + 2.0 * coordinates.root_v * first_node.abs();
            for &(second_node, second_weight) in second_nodes {
                let second_envelopes = real_envelopes(
                    coordinates.c + coordinates.beta * first_node + coordinates.gamma * second_node,
                );
                let second_perturbation = coordinates.c.abs()
                    + 2.0 * slope * first_node.abs()
                    + 2.0 * coordinates.gamma * second_node.abs();
                let weight = (first_weight * second_weight).abs();
                let relative = relative_base + first_node * first_node + second_node * second_node;
                for (p, row) in allowance.iter_mut().enumerate() {
                    for (q, entry) in row.iter_mut().enumerate() {
                        *entry += weight
                            * (relative * first_envelopes[p] * second_envelopes[q]
                                + first_perturbation * first_envelopes[p + 1] * second_envelopes[q]
                                + second_perturbation * first_envelopes[p] * second_envelopes[q + 1]);
                    }
                }
            }
        }
        allowance.map(|row| row.map(|entry| f64::EPSILON * entry))
    }

    /// The pair table through the production path, with its rounding, for a law in
    /// the production orientation `v ≥ w`, `v > 0`.
    fn pair_with_rounding(
        b: f64,
        c: f64,
        v: f64,
        w: f64,
        r: f64,
    ) -> (SiluPairMoments, [[f64; ORDERS]; ORDERS]) {
        assert!(v >= w && v > 0.0, "the helper takes the production orientation");
        let coordinates = PairCoordinates::of(b, c, v, w, r);
        let rule = pair_rule(&coordinates);
        let second_nodes = match &rule.second {
            Some(axis) => trapezoid_nodes(axis),
            None => vec![(0.0, 1.0)],
        };
        let rounding = pair_rounding(
            &coordinates,
            0.0,
            &trapezoid_nodes(&rule.first),
            &second_nodes,
        );
        let moments = silu_pair_moments(b, c, v, w, r).expect("a valid pair law");
        (moments, rounding)
    }

    #[test]
    fn silu_pair_moments_meet_their_bound_against_an_independent_reference() {
        let legendre = gauss_legendre(16);
        // A correlated pair with unequal scales, knee offsets on both sides and ρ = 0.6.
        let (b, c, v, w): (f64, f64, f64, f64) = (0.8, -1.3, 4.0, 2.25);
        let r = 0.6 * (v * w).sqrt();
        let (table, rounding) = pair_with_rounding(b, c, v, w, r);
        let coordinates = PairCoordinates::of(b, c, v, w, r);
        let first_pole = PI / coordinates.root_v.max(coordinates.beta.abs());
        let second_pole = PI / coordinates.gamma;
        let reference = |refinement: usize| {
            let first_panels = refinement
                * (REFERENCE_HALF_WIDTH / legendre_panel_half_width(first_pole)).ceil() as usize;
            let second_panels = refinement
                * (REFERENCE_HALF_WIDTH / legendre_panel_half_width(second_pole)).ceil() as usize;
            let first_nodes = legendre_panels(REFERENCE_HALF_WIDTH, first_panels, &legendre);
            let second_nodes = legendre_panels(REFERENCE_HALF_WIDTH, second_panels, &legendre);
            let mut sums = [[0.0; ORDERS]; ORDERS];
            for &(first_node, first_weight) in &first_nodes {
                let first = silu_derivatives(b + coordinates.root_v * first_node);
                let center = c + coordinates.beta * first_node;
                let mut inner = [0.0; ORDERS];
                for &(second_node, second_weight) in &second_nodes {
                    let second = silu_derivatives(center + coordinates.gamma * second_node);
                    for (sum, value) in inner.iter_mut().zip(second) {
                        *sum += second_weight * value;
                    }
                }
                for (row, first_value) in sums.iter_mut().zip(first) {
                    for (sum, inner_value) in row.iter_mut().zip(inner) {
                        *sum += first_weight * first_value * inner_value;
                    }
                }
            }
            (
                sums,
                pair_rounding(&coordinates, 9.0, &first_nodes, &second_nodes),
            )
        };
        let (reference_sums, reference_rounding) = reference(1);
        let (refined_sums, refined_rounding) = reference(2);
        let scale_envelope = pair_envelope(&coordinates, 0.0, 0.0);
        let tails = standard_normal_tails(REFERENCE_HALF_WIDTH);
        for p in 0..ORDERS {
            for q in 0..ORDERS {
                // Truncating both reference axes at L drops the two regions beyond L, each
                // weighted over the other axis by its full-line integral 1.
                let reference_tail = 4.0
                    * scale_envelope[p][q].iter().sum::<f64>()
                    * (tails[0] + tails[1] + tails[2]);
                let reference_change = (reference_sums[p][q] - refined_sums[p][q]).abs();
                assert!(
                    reference_change <= reference_rounding[p][q] + refined_rounding[p][q],
                    "p={p} q={q}: doubling the reference panels moved it by {reference_change:e}"
                );
                let error = (table.moments[p][q] - reference_sums[p][q]).abs();
                let allowance = table.quadrature_bound[p][q]
                    + rounding[p][q]
                    + reference_rounding[p][q]
                    + reference_tail;
                assert!(
                    error <= allowance,
                    "p={p} q={q}: |rule − reference| = {error:e} exceeds bound {:e} + rounding {:e} \
                     + reference rounding {:e} + tail {reference_tail:e}",
                    table.quadrature_bound[p][q],
                    rounding[p][q],
                    reference_rounding[p][q]
                );
            }
        }
    }

    #[test]
    fn silu_pair_moments_reduce_to_their_exact_limits() {
        let (b, c, v, w) = (0.8, -1.3, 4.0, 1.0);
        // r = 0 factorises into the one-dimensional smoothings.
        let (independent, independent_rounding) = pair_with_rounding(b, c, v, w, 0.0);
        let first_axis = smoothing_rule(b, v);
        let second_axis = smoothing_rule(c, w);
        let first = smoothing_sums(b, v, &first_axis);
        let second = smoothing_sums(c, w, &second_axis);
        let first_bound = smoothing_bound(b, v, &first_axis);
        let second_bound = smoothing_bound(c, w, &second_axis);
        let first_rounding = smoothing_rounding(b, v, &first_axis);
        let second_rounding = smoothing_rounding(c, w, &second_axis);
        for p in 0..ORDERS {
            for q in 0..ORDERS {
                let first_error = first_bound[p] + first_rounding[p];
                let second_error = second_bound[q] + second_rounding[q];
                let product = first[p] * second[q];
                let allowance = independent.quadrature_bound[p][q]
                    + independent_rounding[p][q]
                    + first[p].abs() * second_error
                    + second[q].abs() * first_error
                    + first_error * second_error;
                let residual = (independent.moments[p][q] - product).abs();
                assert!(
                    residual <= allowance,
                    "p={p} q={q}: at r = 0 the table holds {} against the product {product}, \
                     residual {residual:e} > {allowance:e}",
                    independent.moments[p][q]
                );
            }
        }
        // Equal variances: swapping the roles of X and Y transposes the table, through a
        // different Cholesky coordinate system.
        let r_equal = 0.45;
        let (forward, forward_rounding) = pair_with_rounding(b, c, v, v, r_equal);
        let (backward, backward_rounding) = pair_with_rounding(c, b, v, v, r_equal);
        for p in 0..ORDERS {
            for q in 0..ORDERS {
                let residual = (forward.moments[p][q] - backward.moments[q][p]).abs();
                let allowance = forward.quadrature_bound[p][q]
                    + backward.quadrature_bound[q][p]
                    + forward_rounding[p][q]
                    + backward_rounding[q][p];
                assert!(
                    residual <= allowance,
                    "p={p} q={q}: transposed tables differ by {residual:e} > {allowance:e}"
                );
            }
        }
        // |ρ| → 1: the two-axis rule approaches the exact one-axis rule at r = √(v·w) = 2.
        // |∂_r K_pq| = |K_{p+1,q+1}| is at most the product of the constant envelopes of
        // orders p + 1 and q + 1.
        let r_edge = 2.0;
        let relative_gap = 1.0e-8;
        let (edge, edge_rounding) = pair_with_rounding(b, c, v, w, r_edge);
        let (near, near_rounding) = pair_with_rounding(b, c, v, w, r_edge * (1.0 - relative_gap));
        for p in 0..ORDERS {
            for q in 0..ORDERS {
                let lipschitz = real_envelope(p + 1, 0.0) * real_envelope(q + 1, 0.0);
                let residual = (edge.moments[p][q] - near.moments[p][q]).abs();
                let allowance = edge.quadrature_bound[p][q]
                    + near.quadrature_bound[p][q]
                    + edge_rounding[p][q]
                    + near_rounding[p][q]
                    + lipschitz * r_edge * relative_gap;
                assert!(
                    residual <= allowance,
                    "p={p} q={q}: at |ρ| = 1 − 1e−8 the table moved by {residual:e} > {allowance:e}"
                );
            }
        }
    }

    const DIMENSION: usize = 5;

    /// SplitMix64 feeding Box-Muller: a deterministic Gaussian stream, so each Monte
    /// Carlo verdict is reproducible.
    struct GaussianStream {
        state: u64,
        spare: Option<f64>,
    }

    impl GaussianStream {
        fn new(seed: u64) -> Self {
            Self { state: seed, spare: None }
        }

        fn next_u64(&mut self) -> u64 {
            self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }

        fn uniform(&mut self) -> f64 {
            ((self.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
        }

        fn normal(&mut self) -> f64 {
            if let Some(value) = self.spare.take() {
                return value;
            }
            let radius = (-2.0 * self.uniform().ln()).sqrt();
            let angle = 2.0 * PI * self.uniform();
            self.spare = Some(radius * angle.sin());
            radius * angle.cos()
        }

        fn vector(&mut self) -> [f64; DIMENSION] {
            std::array::from_fn(|_| self.normal())
        }
    }

    fn dot(left: &[f64; DIMENSION], right: &[f64; DIMENSION]) -> f64 {
        left.iter().zip(right).map(|(a, b)| a * b).sum()
    }

    fn add_scaled(
        left: &[f64; DIMENSION],
        scale: f64,
        right: &[f64; DIMENSION],
    ) -> [f64; DIMENSION] {
        std::array::from_fn(|index| left[index] + scale * right[index])
    }

    /// An orthonormal frame by Gram-Schmidt. Its projector is `P = QQᵀ`.
    fn frame(columns: [[f64; DIMENSION]; 2]) -> [[f64; DIMENSION]; 2] {
        let first_norm = dot(&columns[0], &columns[0]).sqrt();
        let first = columns[0].map(|entry| entry / first_norm);
        let residual = add_scaled(&columns[1], -dot(&columns[1], &first), &first);
        let residual_norm = dot(&residual, &residual).sqrt();
        [first, residual.map(|entry| entry / residual_norm)]
    }

    fn project(frame: &[[f64; DIMENSION]; 2], vector: &[f64; DIMENSION]) -> [f64; DIMENSION] {
        let along_first = add_scaled(&[0.0; DIMENSION], dot(&frame[0], vector), &frame[0]);
        add_scaled(&along_first, dot(&frame[1], vector), &frame[1])
    }

    /// Sample mean and its standard error.
    fn mean_and_standard_error(samples: &[f64]) -> (f64, f64) {
        let count = samples.len() as f64;
        let mean = samples.iter().sum::<f64>() / count;
        let variance =
            samples.iter().map(|sample| (sample - mean).powi(2)).sum::<f64>() / (count - 1.0);
        (mean, (variance / count).sqrt())
    }

    /// The standard-error multiple of a family of `checks` two-sided agreement tests
    /// at the declared false-alarm rate (Bonferroni).
    fn agreement_multiple(checks: usize) -> f64 {
        standard_normal_quantile(1.0 - MONTE_CARLO_FALSE_ALARM_RATE / (2.0 * checks as f64))
            .expect("a probability inside (0, 1)")
    }

    fn silu(u: f64) -> f64 {
        silu_derivatives(u)[0]
    }

    #[test]
    fn gated_conditional_mean_matches_monte_carlo_of_a_synthetic_unit() {
        let up = [0.6, 0.9, -0.5, 1.0, 0.2];
        let up_offset = 0.6;
        let gate = [0.5, 0.8, -0.6, 0.9, 0.4];
        let gate_offset = -0.3;
        let retained = frame([[1.0, 0.5, 0.0, -0.5, 0.2], [0.0, 1.0, 1.0, 0.3, -0.4]]);
        let points = [
            [0.0; DIMENSION],
            [1.2, -0.7, 0.4, 0.0, 0.9],
            [-1.5, 0.3, -1.1, 0.8, -0.2],
        ];
        let multiple = agreement_multiple(points.len());
        let up_discarded = add_scaled(&up, -1.0, &project(&retained, &up));
        let gate_discarded = add_scaled(&gate, -1.0, &project(&retained, &gate));
        let v_perp = dot(&gate_discarded, &gate_discarded);
        let kappa_perp = dot(&up_discarded, &gate_discarded);
        let mut stream = GaussianStream::new(0x2946_0009);
        for point in points {
            let retained_point = project(&retained, &point);
            let alpha = dot(&up, &retained_point) + up_offset;
            let t = dot(&gate, &retained_point) + gate_offset;
            let mean =
                gated_conditional_mean(alpha, t, v_perp, kappa_perp).expect("a valid gated law");
            let samples: Vec<f64> = (0..(1usize << 18))
                .map(|_| {
                    let draw = stream.vector();
                    let discarded = add_scaled(&draw, -1.0, &project(&retained, &draw));
                    let z = add_scaled(&retained_point, 1.0, &discarded);
                    (dot(&up, &z) + up_offset) * silu(dot(&gate, &z) + gate_offset)
                })
                .collect();
            let (monte_carlo, standard_error) = mean_and_standard_error(&samples);
            let band = multiple * standard_error + mean.value.quadrature_bound;
            let stein_gap = (monte_carlo - mean.value.value).abs();
            assert!(
                stein_gap <= band,
                "point {point:?}: Stein mean {} vs Monte Carlo {monte_carlo} differ by \
                 {stein_gap:e} > {multiple:.3}·SE {standard_error:e}",
                mean.value.value
            );
            // Positive control: the plug-in mean α·T_v s(t) drops the Stein term
            // κ·T_v s'(t). The fixture must make that term resolvable, and the
            // comparison must reject the plug-in mean.
            let stein_term = kappa_perp * mean.d_kappa.value;
            assert!(
                stein_term.abs() > 2.0 * band,
                "fixture: the Stein term {stein_term:e} is not resolvable against the band {band:e}"
            );
            let plug_in = alpha * mean.d_alpha.value;
            assert!(
                (monte_carlo - plug_in).abs() > band,
                "point {point:?}: the plug-in mean {plug_in} was not rejected \
                 (Monte Carlo {monte_carlo})"
            );
        }
    }

    #[test]
    fn gated_pair_second_moment_matches_monte_carlo_under_the_coupling() {
        let up_j = [0.6, 0.9, -0.5, 1.0, 0.2];
        let gate_j = [0.5, 0.8, -0.6, 0.9, 0.4];
        let up_k = [-0.4, 0.7, 0.9, -0.2, 0.6];
        let gate_k = [0.3, -0.6, 0.8, 0.5, -0.7];
        let (offset_j, bias_j, offset_k, bias_k) = (0.6, -0.3, -0.5, 0.4);
        // The retained frame spans a_j + w_k and a_k + w_j, so the coupled covariances
        // are large.
        let retained = frame([add_scaled(&up_j, 1.0, &gate_k), add_scaled(&up_k, 1.0, &gate_j)]);
        let law = GatedPairLaw {
            offset_j,
            offset_k,
            kappa_j: dot(&up_j, &gate_j),
            kappa_k: dot(&up_k, &gate_k),
            lambda_jk: dot(&up_j, &project(&retained, &gate_k)),
            lambda_kj: dot(&up_k, &project(&retained, &gate_j)),
            rho: dot(&up_j, &project(&retained, &up_k)),
        };
        let moments = silu_pair_moments(
            bias_j,
            bias_k,
            dot(&gate_j, &gate_j),
            dot(&gate_k, &gate_k),
            dot(&gate_j, &project(&retained, &gate_k)),
        )
        .expect("a valid pair law");
        let moment = gated_pair_second_moment(&law, &moments);
        let mut stream = GaussianStream::new(0x2946_0010);
        let samples: Vec<f64> = (0..(1usize << 19))
            .map(|_| {
                let z = stream.vector();
                let fresh = stream.vector();
                // Z' = PZ + (I − P)Z̃ = Z̃ + P(Z − Z̃).
                let coupled = add_scaled(
                    &fresh,
                    1.0,
                    &project(&retained, &add_scaled(&z, -1.0, &fresh)),
                );
                let unit_j = (dot(&up_j, &z) + offset_j) * silu(dot(&gate_j, &z) + bias_j);
                let unit_k =
                    (dot(&up_k, &coupled) + offset_k) * silu(dot(&gate_k, &coupled) + bias_k);
                unit_j * unit_k
            })
            .collect();
        let (monte_carlo, standard_error) = mean_and_standard_error(&samples);
        let multiple = agreement_multiple(1);
        let band = multiple * standard_error + moment.value.quadrature_bound;
        let gap = (monte_carlo - moment.value.value).abs();
        assert!(
            gap <= band,
            "Stein pair moment {} vs Monte Carlo {monte_carlo} differ by {gap:e} > \
             {multiple:.3}·SE {standard_error:e}",
            moment.value.value
        );
        // Positive control: dropping the up-projection/gate cross covariances λ must be
        // rejected, and the fixture must make them resolvable.
        let dropped = gated_pair_second_moment(
            &GatedPairLaw {
                lambda_jk: 0.0,
                lambda_kj: 0.0,
                ..law
            },
            &moments,
        );
        let legs = moment.value.value - dropped.value.value;
        assert!(
            legs.abs() > 2.0 * band,
            "fixture: the λ legs {legs:e} are not resolvable against the band {band:e}"
        );
        assert!(
            (monte_carlo - dropped.value.value).abs() > band,
            "the pair moment without its λ legs ({}) was not rejected (Monte Carlo {monte_carlo})",
            dropped.value.value
        );
    }

    /// A central difference's error: truncation `δ²/6·sup|f'''|` plus evaluation
    /// noise `(e₊ + e₋)/(2δ)`. The step minimising their sum is
    /// `δ = (3e/sup|f'''|)^{1/3}`.
    fn central_difference_step(noise: f64, third_derivative_bound: f64) -> f64 {
        (3.0 * noise / third_derivative_bound).cbrt()
    }

    /// Quadrature bound plus rounding of one gated-mean evaluation.
    fn mean_noise(alpha: f64, t: f64, v: f64, kappa: f64) -> f64 {
        let axis = smoothing_rule(t, v);
        let bound = smoothing_bound(t, v, &axis);
        let rounding = smoothing_rounding(t, v, &axis);
        alpha.abs() * (bound[0] + rounding[0]) + kappa.abs() * (bound[1] + rounding[1])
    }

    /// Quadrature bound plus rounding of a Stein contraction `Σ c·K_{p+s,q+s}`,
    /// with each coefficient's own two-operation rounding and six-term summation.
    fn contraction_noise(
        law: &GatedPairLaw,
        table: &SiluPairMoments,
        table_rounding: &[[f64; ORDERS]; ORDERS],
        shift: usize,
    ) -> f64 {
        law.stein_coefficients()
            .iter()
            .map(|&(p, q, coefficient)| {
                let entry = table.moments[p + shift][q + shift];
                coefficient.abs()
                    * (table.quadrature_bound[p + shift][q + shift]
                        + table_rounding[p + shift][q + shift]
                        + 16.0 * f64::EPSILON * entry.abs())
            })
            .sum()
    }

    #[test]
    fn gated_mean_and_pair_legs_match_central_differences() {
        let (alpha, t, v, kappa) = (0.7, -0.4, 2.5, 1.1);
        let mean = gated_conditional_mean(alpha, t, v, kappa).expect("a valid gated law");
        let rounding = smoothing_rounding(t, v, &smoothing_rule(t, v));
        let value_at = |t: f64, v: f64| {
            gated_conditional_mean(alpha, t, v, kappa)
                .expect("a valid gated law")
                .value
                .value
        };

        // ∂_t: the third t-derivative of the mean is α·T s''' + κ·T s'''', within the
        // constant envelopes.
        let third_t = alpha.abs() * real_envelope(3, 0.0) + kappa.abs() * real_envelope(4, 0.0);
        let step_t = central_difference_step(mean_noise(alpha, t, v, kappa), third_t);
        let difference_t = (value_at(t + step_t, v) - value_at(t - step_t, v)) / (2.0 * step_t);
        let allowance_t = step_t * step_t / 6.0 * third_t
            + (mean_noise(alpha, t + step_t, v, kappa) + mean_noise(alpha, t - step_t, v, kappa))
                / (2.0 * step_t)
            + mean.d_t.quadrature_bound
            + alpha.abs() * rounding[1]
            + kappa.abs() * rounding[2];
        assert!(
            (difference_t - mean.d_t.value).abs() <= allowance_t,
            "∂_t: analytic {} vs central difference {difference_t} differ by more than \
             {allowance_t:e}",
            mean.d_t.value
        );

        // ∂_v: ∂³_v T_v f = ⅛·T_v f⁽⁶⁾ by the heat equation, bounded through Cauchy's
        // estimate at the widest variance the difference visits, 1.5·v.
        let cauchy_mean = |order: usize| envelope_second_moment(order, t, 1.5 * v).sqrt();
        let third_v = 0.125 * (alpha.abs() * cauchy_mean(6) + kappa.abs() * cauchy_mean(7));
        let step_v = central_difference_step(mean_noise(alpha, t, v, kappa), third_v).min(0.5 * v);
        let difference_v = (value_at(t, v + step_v) - value_at(t, v - step_v)) / (2.0 * step_v);
        let allowance_v = step_v * step_v / 6.0 * third_v
            + (mean_noise(alpha, t, v + step_v, kappa) + mean_noise(alpha, t, v - step_v, kappa))
                / (2.0 * step_v)
            + mean.d_v.quadrature_bound
            + 0.5 * (alpha.abs() * rounding[2] + kappa.abs() * rounding[3]);
        assert!(
            (difference_v - mean.d_v.value).abs() <= allowance_v,
            "∂_v: analytic {} vs central difference {difference_v} differ by more than \
             {allowance_v:e}",
            mean.d_v.value
        );
        // Positive control: a wrong heat-equation factor would be resolved.
        assert!(
            (difference_v - 2.0 * mean.d_v.value).abs() > allowance_v,
            "∂_v: this comparison cannot tell ½·T s'' from T s''"
        );

        // ∂_r of the pair moment: the third r-derivative is the contraction three Price
        // orders up, each K bounded by Cauchy-Schwarz over the envelopes.
        let (b, c, var_j, var_k, r) = (0.3, -0.6, 2.0, 1.44, 0.7);
        let law = GatedPairLaw {
            offset_j: 0.5,
            offset_k: -0.4,
            kappa_j: 0.9,
            kappa_k: 0.6,
            lambda_jk: 0.35,
            lambda_kj: -0.25,
            rho: 0.2,
        };
        let third_r: f64 = law
            .stein_coefficients()
            .iter()
            .map(|&(p, q, coefficient)| {
                coefficient.abs()
                    * envelope_second_moment(p + 3, b, var_j).sqrt()
                    * envelope_second_moment(q + 3, c, var_k).sqrt()
            })
            .sum();
        let (table, table_rounding) = pair_with_rounding(b, c, var_j, var_k, r);
        let moment = gated_pair_second_moment(&law, &table);
        let step_r = central_difference_step(
            contraction_noise(&law, &table, &table_rounding, 0),
            third_r,
        )
        .min(0.5 * ((var_j * var_k).sqrt() - r.abs()));
        let (forward_table, forward_rounding) = pair_with_rounding(b, c, var_j, var_k, r + step_r);
        let (backward_table, backward_rounding) =
            pair_with_rounding(b, c, var_j, var_k, r - step_r);
        let difference_r = (gated_pair_second_moment(&law, &forward_table).value.value
            - gated_pair_second_moment(&law, &backward_table).value.value)
            / (2.0 * step_r);
        let allowance_r = step_r * step_r / 6.0 * third_r
            + (contraction_noise(&law, &forward_table, &forward_rounding, 0)
                + contraction_noise(&law, &backward_table, &backward_rounding, 0))
                / (2.0 * step_r)
            + contraction_noise(&law, &table, &table_rounding, 1);
        assert!(
            (difference_r - moment.d_r.value).abs() <= allowance_r,
            "∂_r: analytic {} vs central difference {difference_r} differ by more than \
             {allowance_r:e}",
            moment.d_r.value
        );

        // The covariance legs: M is affine in each of ρ, λ_jk and λ_kj with the others
        // held, so a unit central difference is exact up to rounding. Each contraction
        // (value or leg) rounds by at most 16ε of its terms' magnitudes: two operations
        // per coefficient, one product and six-term summation.
        let value_with =
            |shifted: &GatedPairLaw| gated_pair_second_moment(shifted, &table).value.value;
        let magnitude = |shifted: &GatedPairLaw| -> f64 {
            shifted
                .stein_coefficients()
                .iter()
                .map(|&(p, q, coefficient)| (coefficient * table.moments[p][q]).abs())
                .sum()
        };
        let k = &table.moments;
        let legs = [
            (
                "ρ",
                moment.d_rho.value,
                k[0][0].abs(),
                GatedPairLaw { rho: law.rho + 1.0, ..law },
                GatedPairLaw { rho: law.rho - 1.0, ..law },
            ),
            (
                "λ_jk",
                moment.d_lambda_jk.value,
                (law.offset_k * k[0][1]).abs()
                    + (law.lambda_kj * k[1][1]).abs()
                    + (law.kappa_k * k[0][2]).abs(),
                GatedPairLaw { lambda_jk: law.lambda_jk + 1.0, ..law },
                GatedPairLaw { lambda_jk: law.lambda_jk - 1.0, ..law },
            ),
            (
                "λ_kj",
                moment.d_lambda_kj.value,
                (law.offset_j * k[1][0]).abs()
                    + (law.kappa_j * k[2][0]).abs()
                    + (law.lambda_jk * k[1][1]).abs(),
                GatedPairLaw { lambda_kj: law.lambda_kj + 1.0, ..law },
                GatedPairLaw { lambda_kj: law.lambda_kj - 1.0, ..law },
            ),
        ];
        for (name, analytic, leg_magnitude, raised, lowered) in legs {
            let difference = 0.5 * (value_with(&raised) - value_with(&lowered));
            let rounding_allowance = 16.0
                * f64::EPSILON
                * (0.5 * (magnitude(&raised) + magnitude(&lowered)) + leg_magnitude);
            let residual = (difference - analytic).abs();
            assert!(
                residual <= rounding_allowance,
                "∂_{name}: analytic {analytic} vs central difference {difference} differ by \
                 {residual:e} > {rounding_allowance:e}"
            );
        }
    }
}
