//! Scalar special-function primitives shared across the workspace.
//!
//! These are pure (`std`/`libm`-only) numeric kernels with no upward crate
//! dependencies, so they live in the lowest crate (`gam-math`) and can be
//! consumed by any term/basis/inference code without inducing an SCC edge.

/// Numerically stable `C(n,k) = n! / (k!·(n−k)!)` as `f64`.  Uses the
/// symmetry `C(n,k) = C(n, n−k)` to keep the loop count `min(k, n−k)`
/// and the multiplicative recurrence `C(n,j+1) = C(n,j)·(n−j)/(j+1)`,
/// avoiding the overflow of separate factorial evaluations.  Returns
/// `0.0` for `k > n` and exact integer results within `2^53`.
#[inline]
pub fn binomial_coefficient_f64(n: usize, k: usize) -> f64 {
    if k > n {
        return 0.0;
    }
    if k == 0 || k == n {
        return 1.0;
    }
    let k_eff = k.min(n - k);
    // Carry the recurrence in u128, not f64. At step `j` the running product
    // equals the integer `C(n, j)`, which is always divisible by the next
    // denominator `(j + 1)` (the partial product of `(j+1)` consecutive
    // integers `(n−j)…(n)` is divisible by `(j+1)!`), so each integer division
    // is exact and no rounding accumulates. The earlier all-`f64` recurrence
    // divided in floating point, where `(n−j)/(j+1)` is generally inexact, and
    // the drift pushed results off the true integer well below `2^53`
    // (e.g. `C(54,24)` came back one short). Converting the exact `u128` at the
    // end is bit-exact for every value at or below `2^53`.
    let mut num: u128 = 1;
    for j in 0..k_eff {
        match num.checked_mul((n - j) as u128) {
            Some(scaled) => num = scaled / (j as u128 + 1),
            None => {
                // The true coefficient overflows u128 — astronomically above
                // `2^53`, where the exactness contract no longer applies.
                // Finish the (now necessarily inexact) recurrence in f64.
                let mut out = num as f64;
                for jj in j..k_eff {
                    out = out * (n - jj) as f64 / (jj + 1) as f64;
                }
                return out;
            }
        }
    }
    num as f64
}

#[inline]
fn horner_polynomial(x: f64, coeffs: &[f64]) -> f64 {
    coeffs.iter().rev().fold(0.0, |acc, &c| acc * x + c)
}

/// Evaluate `(Σ_k coeffs[k]·x^k) · exp(−x)` without overflow.  For moderate
/// `x ≤ 600` uses Horner + `exp(−x)` directly; for very large `x` rewrites
/// `xᵈ · exp(−x) = exp(d·ln x − x)` and runs Horner in `1/x`, which keeps
/// both the polynomial sum and its multiplier inside double range.  Returns
/// `0.0` for non-finite `x` or empty `coeffs`.
#[inline]
pub fn stable_polynomial_times_exp_neg(x: f64, coeffs: &[f64]) -> f64 {
    if coeffs.is_empty() || !x.is_finite() {
        return 0.0;
    }
    // Below this argument `(-x).exp()` is still well-resolved, so the direct
    // Horner-times-exp form is both accurate and cheapest. Above it the factor
    // underflows toward zero and we switch to the convergent asymptotic tail
    // series to retain the leading significant digits.
    const DIRECT_EXP_SWITCH: f64 = 600.0;
    if x <= DIRECT_EXP_SWITCH {
        return horner_polynomial(x, coeffs) * (-x).exp();
    }

    let inv_x = x.recip();
    let mut tail = 0.0;
    for &c in coeffs {
        tail = tail * inv_x + c;
    }
    let degree = (coeffs.len() - 1) as f64;
    let scale = (degree * x.ln() - x).exp();
    scale * tail
}

/// Argument at which the modified-Bessel evaluation switches from the ascending
/// power series to the large-argument (Hankel) asymptotic expansion.
///
/// Both branches are accurate to a few ulp here, which is what leaves the
/// crossover free of a visible seam. Below it the ascending series is exact up
/// to rounding, because every one of its terms is positive and nothing cancels.
/// Above it the asymptotic expansion's optimal-truncation error is `O(e^{−2x})`
/// — below `5e−18` already at `x = 20`, and shrinking from there.
///
/// The former implementation used the single-precision Abramowitz & Stegun
/// 9.8.1–9.8.4 minimax polynomials (crossover 3.75), whose stated accuracy is
/// `|ε| < 2e−7`. That is seven digits short of `f64` and it was the accuracy
/// floor of everything derived from them: `I1/I0` carried `1e−6` relative
/// error, the von-Mises ARD gradient channel `x·(I1/I0 − 1)` carried `4e−6`,
/// and the ARD log-precision curvature carried `6e−3` — a 0.6% error in a
/// quantity an outer Newton step consumes as an exact second derivative, with
/// visible jumps across both the 3.75 and the 30 branch seams.
const BESSEL_ASYMPTOTIC_THRESHOLD: f64 = 20.0;

/// Loop bound for the ascending series. It converges for every argument; below
/// the crossover 36 terms always suffice, so the cap only bounds the loop for a
/// non-finite argument that slipped past the guards.
const BESSEL_SERIES_MAX_TERMS: usize = 128;

/// Loop bound for the asymptotic expansion. The expansion is divergent, so it
/// is truncated at its own smallest term long before this for every argument it
/// is used at; the cap also keeps the coefficient recurrence itself in range.
const BESSEL_ASYMPTOTIC_MAX_TERMS: usize = 64;

/// Ascending power series `(I0(x) − 1, I1(x))` for finite `x ≥ 0`.
///
/// `I0(x) = Σ_k (x/2)^{2k}/(k!)²` and `I1(x) = (x/2)·Σ_k (x/2)^{2k}/(k!(k+1)!)`,
/// both carried by the ratio recurrence rather than by separate factorials.
/// Every term is positive, so neither sum cancels and each is correct to within
/// the accumulated rounding of its own additions.
///
/// `I0 − 1` is returned instead of `I0` so the caller can take `ln_1p`: as
/// `x → 0` the wanted `log I0(x) ≈ x²/4` falls below the resolution of
/// `1 + x²/4`, and forming `I0` first would round it away entirely.
fn bessel_ascending_series(ax: f64) -> (f64, f64) {
    let half = 0.5 * ax;
    let quarter_square = half * half;
    let mut term_i0 = 1.0_f64;
    let mut i0_minus_one = 0.0_f64;
    let mut term_i1 = 1.0_f64;
    let mut sum_i1 = 1.0_f64;
    for k in 1..=BESSEL_SERIES_MAX_TERMS {
        let kf = k as f64;
        term_i0 *= quarter_square / (kf * kf);
        term_i1 *= quarter_square / (kf * (kf + 1.0));
        i0_minus_one += term_i0;
        sum_i1 += term_i1;
        if term_i0 <= f64::EPSILON * (1.0 + i0_minus_one) && term_i1 <= f64::EPSILON * sum_i1 {
            break;
        }
    }
    (i0_minus_one, half * sum_i1)
}

/// One evaluation of the large-argument (Hankel) asymptotic expansions of `I0`
/// and `I1`, kept in the combinations the callers need so that every leading
/// term cancels ANALYTICALLY here instead of in floating point.
///
/// With `I_ν(x) ~ e^x/√(2πx) · Σ_k (−1)^k a_k(ν) x^{−k}` and
/// `a_k(ν) = ∏_{j=1}^{k} (4ν² − (2j−1)²) / (k!·8^k)`, write `c_k = (−1)^k a_k(0)`
/// and `b_k = (−1)^k a_k(1)`; then `c_0 = b_0 = 1` and both families obey a
/// two-term ratio recurrence, so no coefficient table is needed.
struct BesselAsymptotic {
    /// `S0 = Σ_{k≥0} c_k x^{−k}`, so `I0(x) = e^x S0 / √(2πx)`.
    s0: f64,
    /// `S1 = Σ_{k≥0} b_k x^{−k}`, so `I1/I0 = S1/S0`.
    s1: f64,
    /// `N = Σ_{k≥1} (b_k − c_k) x^{−(k−1)}`, so `d1 = x(I1/I0 − 1) = N/S0`.
    ///
    /// The `k = 0` terms of `S1` and `S0` are both exactly `1`, so they are
    /// dropped symbolically and the difference series starts at its own leading
    /// term `b_1 − c_1 = −1/2` — which is precisely the `d1 → −½` limit. No
    /// near-equal quantities are ever subtracted at run time.
    n: f64,
    /// `x²·S0′ = Σ_{k≥1} (−k) c_k x^{−(k−1)}`.
    s0_scaled_derivative: f64,
    /// `x²·N′ = Σ_{k≥2} −(k−1)(b_k − c_k) x^{−(k−2)}`.
    ///
    /// Both derivative accumulators carry the common `x²` factored out, which
    /// is what keeps `c″(log x) = (x²N′·S0 − N·x²S0′)/(x·S0²)` representable —
    /// and non-zero — out to the largest finite argument, where the unscaled
    /// `x^{−k}` factors would have underflowed to zero.
    n_scaled_derivative: f64,
}

fn bessel_asymptotic_series(ax: f64) -> BesselAsymptotic {
    let inverse = 1.0 / ax;
    let mut c = 1.0_f64;
    let mut b = 1.0_f64;
    let mut acc = BesselAsymptotic {
        s0: 1.0,
        s1: 1.0,
        n: 0.0,
        s0_scaled_derivative: 0.0,
        n_scaled_derivative: 0.0,
    };
    // `x^{−(k−2)}` and `x^{−(k−1)}` at the current `k`, carried as their own
    // running products so that a power which has overflowed is never multiplied
    // by one which has underflowed.
    let mut power_two_back = ax;
    let mut power_one_back = 1.0_f64;
    let mut smallest = f64::INFINITY;
    for k in 1..=BESSEL_ASYMPTOTIC_MAX_TERMS {
        let kf = k as f64;
        let odd = 2.0 * kf - 1.0;
        c *= odd * odd / (8.0 * kf);
        b *= (odd * odd - 4.0) / (8.0 * kf);
        let power = power_one_back * inverse;
        let term_c = c * power;
        // The expansion is asymptotic, not convergent: past its smallest term
        // every further term makes the answer worse. Stopping there is what
        // realises the `O(e^{−2x})` optimal-truncation error. The negated
        // comparison also stops on a NaN argument.
        if !(term_c.abs() <= smallest) {
            break;
        }
        smallest = term_c.abs();
        let difference = b - c;
        let curvature_term = (kf - 1.0) * difference * power_two_back;
        acc.s0 += term_c;
        acc.s1 += b * power;
        acc.n += difference * power_one_back;
        acc.s0_scaled_derivative -= kf * c * power_one_back;
        if k >= 2 {
            acc.n_scaled_derivative -= curvature_term;
        }
        // `n_scaled_derivative` carries the largest power of the four sums, so
        // once ITS increment is negligible every other one is too.
        let scale = acc.n_scaled_derivative.abs().max(acc.n.abs());
        if k >= 3 && curvature_term.abs() <= f64::EPSILON * scale {
            break;
        }
        power_two_back = power_one_back;
        power_one_back = power;
    }
    acc
}

/// Overflow-free centered Bessel value, ratio, and log-scale derivative.
///
/// For `x = |eta|`, returns
/// `(log I0(x) - x, I1(x) / I0(x), x d/dx[log I0(x) - x])`. The third term is
/// the stable form of `x·(I1/I0 - 1)`: it approaches `-½` instead of becoming
/// `x·0` after the ordinary ratio rounds to one. Centering the logarithm by its
/// leading `x` term likewise prevents catastrophic cancellation.
pub fn bessel_i0_centered_terms(eta: f64) -> (f64, f64, f64) {
    let ax = eta.abs();
    if ax.is_nan() {
        return (f64::NAN, f64::NAN, f64::NAN);
    }
    if ax.is_infinite() {
        // `−½ log(2πx) → −∞`, `I1/I0 → 1`, and the centered log-derivative
        // holds its exact `−½` limit.
        return (f64::NEG_INFINITY, 1.0, -0.5);
    }
    if ax < BESSEL_ASYMPTOTIC_THRESHOLD {
        let (i0_minus_one, i1) = bessel_ascending_series(ax);
        let ratio = i1 / (1.0 + i0_minus_one);
        return (i0_minus_one.ln_1p() - ax, ratio, ax * (ratio - 1.0));
    }
    let series = bessel_asymptotic_series(ax);
    (
        // `log I0(x) − x = −½ log(2πx) + log S0`. The `2πx` product is split so
        // it cannot overflow just short of the largest finite argument.
        series.s0.ln() - 0.5 * (std::f64::consts::TAU.ln() + ax.ln()),
        series.s1 / series.s0,
        series.n / series.s0,
    )
}

/// Stable centered Bessel terms when only `log(|eta|)` is representable.
///
/// For a finite representable `|eta|`, this is exactly
/// [`bessel_i0_centered_terms`]. Beyond the float range, inverse-`eta`
/// corrections are themselves below float resolution, so the limiting terms
/// `log I0(eta)-eta = -½ log(2 pi eta)` and
/// `eta d/deta[log I0(eta)-eta] = -½` are the correctly rounded result.
pub fn bessel_i0_centered_terms_from_log_abs(log_abs_eta: f64) -> (f64, f64, f64) {
    if log_abs_eta.is_nan() {
        return (f64::NAN, f64::NAN, f64::NAN);
    }
    if log_abs_eta == f64::NEG_INFINITY {
        return (0.0, 0.0, 0.0);
    }
    if log_abs_eta <= f64::MAX.ln() {
        return bessel_i0_centered_terms(log_abs_eta.exp());
    }
    (-0.5 * (std::f64::consts::TAU.ln() + log_abs_eta), 1.0, -0.5)
}

/// Second log-scale derivative of the centered Bessel primitive:
/// `d²/d(log η)²[log I0(η) − η]`, i.e. the derivative of the third term `d1`
/// returned by [`bessel_i0_centered_terms`] (`d1 = η d/dη[log I0(η) − η]`).
///
/// Writing `s = log η`, `r = I1(η)/I0(η)`, and `c(s) = log I0(η) − η`, the first
/// log-derivative is `c'(s) = d1 = η(r − 1)`. Differentiating again and using
/// the modified-Bessel ratio ODE `r'(η) = 1 − r/η − r²` gives the exact closed
/// form `c''(s) = −η + η²(1 − r²)`. That direct form is numerically unusable
/// for moderate/large `η`: its two terms each grow like `η` and cancel to
/// `O(1/η)`, so the ratio's `~ε_poly` approximation error is amplified by `η²`.
/// The algebraically identical rearrangement in terms of the STABLE third term
///
/// `c''(s) = −η(2·d1 + 1) − d1²`
///
/// cancels safely instead: `d1 → −½` with `2·d1 + 1 → 0`, so the amplification
/// drops to `η·δd1`. It is also, by construction, the exact derivative of the
/// SAME `d1` the outer gradient's periodic-ARD normalizer channel reports, so
/// gradient and Hessian differentiate one quantity. Beyond the float range
/// `c'(s) → −½` (constant) so `c''(s) → 0`; likewise `η → 0` gives `c''(s) → 0`.
/// The von-Mises ARD log-precision normalizer `n[−η + log I0(η)]` therefore has
/// `∂²/∂(log α)² = n · c''(log η)` up to the affine `log η = log α + const` shift.
///
/// Three regimes, each chosen so that nothing cancels in it:
///
/// * `η ≥ 20`: read `c''(s) = η(N′S0 − N S0′)/S0²` straight off the asymptotic
///   expansion ([`BesselAsymptotic`]). Its two products are `3/16` and `1/16` at
///   leading order — a benign ratio, where the `−η(2d1+1)` and `d1²` of the
///   closed form both approach `¼` and cancel down to `1/(8η)`.
/// * `1 ≤ η < 20`: the closed form, with that `¼` removed symbolically. Writing
///   `q = d1 + ½` (which decays like `−1/(8η)`), `d1² = q² − q + ¼` and
///   `−2ηq − ¼ = −2η(q + 1/(8η))`, leaving `c''(s) = −2η(q + 1/(8η)) + q − q²`
///   with no constant term for the answer to be dwarfed by.
/// * `η < 1`: `c''(s) = −η + η²(1 − r²) = −η·[1 + d1(1 + r)]`, whose bracket
///   tends to `1`. The rearrangement above would instead subtract two numbers
///   that both tend to `¼` while the answer itself tends to `−η`.
pub fn bessel_i0_centered_second_log_derivative_from_log_abs(log_abs_eta: f64) -> f64 {
    if log_abs_eta.is_nan() {
        return f64::NAN;
    }
    if log_abs_eta == f64::NEG_INFINITY {
        return 0.0;
    }
    if log_abs_eta > f64::MAX.ln() {
        return 0.0;
    }
    let eta = log_abs_eta.exp();
    if eta >= BESSEL_ASYMPTOTIC_THRESHOLD {
        let series = bessel_asymptotic_series(eta);
        return (series.n_scaled_derivative * series.s0 - series.n * series.s0_scaled_derivative)
            / (eta * series.s0 * series.s0);
    }
    let (_centered, ratio, d1) = bessel_i0_centered_terms(eta);
    if eta < 1.0 {
        return -eta * (1.0 + d1 * (1.0 + ratio));
    }
    let q = d1 + 0.5;
    -2.0 * eta * (q + 0.125 / eta) + q - q * q
}

/// Overflow-free `(log I0(eta) - |eta|, I1(|eta|) / I0(|eta|))`.
///
/// Centering the logarithm by its leading `|eta|` term is essential whenever a
/// likelihood cancels the Bessel growth against an equally large quadratic,
/// as in a Gaussian-blurred circle. The large-argument branch never forms
/// `exp(|eta|)`, and therefore remains finite beyond the ordinary exponential
/// overflow threshold and up to the largest finite `f64`.
pub fn bessel_i0_log_minus_abs_and_ratio(eta: f64) -> (f64, f64) {
    let (centered_log_i0, ratio, _) = bessel_i0_centered_terms(eta);
    (centered_log_i0, ratio)
}

/// Overflow-free `(log I0(eta), I1(|eta|) / I0(|eta|))`.
///
/// Consumers whose formulas cancel the leading `|eta|` term should use
/// [`bessel_i0_log_minus_abs_and_ratio`] directly, rather than forming that
/// cancellation after this function returns.
pub fn bessel_i0_log_and_ratio(eta: f64) -> (f64, f64) {
    let (centered_log_i0, ratio) = bessel_i0_log_minus_abs_and_ratio(eta);
    (eta.abs() + centered_log_i0, ratio)
}

/// Gauss-Legendre nodes and weights on `[-1, 1]` for `n` points, computed via
/// Newton iteration on the Legendre-polynomial roots (Bonnet's three-term
/// recurrence, cosine initial guess). Returns `(nodes, weights)` with nodes
/// ascending; for odd `n` the central node is exactly `0.0`.
///
/// Canonical home for the routine previously triplicated in
/// `gam-terms/basis/closed_form_penalty.rs`, `gam-model-kernels/
/// cubic_cell_kernel.rs`, and `gam-models/survival/base.rs`; this copy keeps
/// the tightest of their Newton settings (200-iteration cap, `1e-15`
/// convergence).
pub fn gauss_legendre(n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut tmp: Vec<(f64, f64)> = Vec::with_capacity(n);
    let half = n.div_ceil(2);
    for i in 0..half {
        let mut z = (std::f64::consts::PI * (i as f64 + 0.75) / (n as f64 + 0.5)).cos();
        let mut pp = 0.0_f64;
        for _ in 0..200 {
            let mut p1 = 1.0_f64;
            let mut p2 = 0.0_f64;
            for j in 0..n {
                let p3 = p2;
                p2 = p1;
                p1 = ((2.0 * j as f64 + 1.0) * z * p2 - j as f64 * p3) / (j as f64 + 1.0);
            }
            pp = n as f64 * (z * p1 - p2) / (z * z - 1.0);
            let z_prev = z;
            z = z_prev - p1 / pp;
            if (z - z_prev).abs() < 1e-15 {
                break;
            }
        }
        let w = 2.0 / ((1.0 - z * z) * pp * pp);
        // For odd n the central node is at z = 0; record once.
        if !n.is_multiple_of(2) && i == half - 1 {
            tmp.push((0.0, w));
        } else {
            tmp.push((-z.abs(), w));
            tmp.push((z.abs(), w));
        }
    }
    tmp.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let mut nodes = Vec::with_capacity(n);
    let mut weights = Vec::with_capacity(n);
    for (z, w) in tmp.into_iter().take(n) {
        nodes.push(z);
        weights.push(w);
    }
    (nodes, weights)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn centered_bessel_log_is_finite_and_derivative_consistent() {
        for eta in [0.25_f64, 1.0, 3.74, 3.76, 12.0, 900.0] {
            let (centered, ratio, scaled_derivative) = bessel_i0_centered_terms(eta);
            assert!(centered.is_finite());
            assert!((0.0..=1.0).contains(&ratio));

            // The tolerances below are sized by what a CENTRAL DIFFERENCE can
            // resolve — roundoff `ε·|f|/h` plus truncation `h²·f'''/6` — not by
            // what the evaluator happens to achieve. The A&S polynomials this
            // replaced needed `1e-6`/`2e-5` here; the series/asymptotic pair
            // leaves the finite difference itself as the limiting error.
            let h = 1.0e-4 * eta.max(1.0);
            let (plus, _) = bessel_i0_log_and_ratio(eta + h);
            let (minus, _) = bessel_i0_log_and_ratio(eta - h);
            let derivative = (plus - minus) / (2.0 * h);
            assert!(
                (derivative - ratio).abs() <= 1.0e-8,
                "d/dη log I0 mismatch at eta={eta}: analytic={ratio}, finite_difference={derivative}"
            );

            let log_step = 1.0e-5_f64;
            let (centered_plus, _, _) = bessel_i0_centered_terms(eta * log_step.exp());
            let (centered_minus, _, _) = bessel_i0_centered_terms(eta * (-log_step).exp());
            let finite_difference = (centered_plus - centered_minus) / (2.0 * log_step);
            assert!(
                (finite_difference - scaled_derivative).abs() < 1.0e-8,
                "centered Bessel value/gradient mismatch at eta={eta}: analytic={scaled_derivative}, finite_difference={finite_difference}"
            );
        }
        for eta in [1.0e20_f64, 1.0e100, 1.0e300] {
            let (centered, ratio, scaled_derivative) = bessel_i0_centered_terms(eta);
            let asymptotic = -0.5 * (std::f64::consts::TAU * eta).ln();
            assert!(centered.is_finite() && ratio.is_finite());
            // The `log S0` remainder is below `1e-20` at these arguments, so the
            // only admissible gap is the differing association of the two `log`
            // groupings — a few ulp of a number of size ~`log η`.
            assert!(
                (centered - asymptotic).abs() < 1.0e-13,
                "large-eta centered log must equal -½log(2πη); eta={eta:e}, centered={centered}, asymptotic={asymptotic}"
            );
            assert!(
                (scaled_derivative + 0.5).abs() < 1.0e-15,
                "large-eta centered derivative must retain its -1/2 limit; eta={eta:e}, derivative={scaled_derivative}"
            );
        }

        assert_eq!(bessel_i0_centered_terms(0.0), (0.0, 0.0, 0.0));

        let log_eta = 1_200.0;
        let (centered, ratio, scaled_derivative) = bessel_i0_centered_terms_from_log_abs(log_eta);
        assert!(centered.is_finite());
        assert_eq!(ratio, 1.0);
        assert_eq!(scaled_derivative, -0.5);
        assert_eq!(centered, -0.5 * (std::f64::consts::TAU.ln() + log_eta));
    }

    #[test]
    fn centered_bessel_second_log_derivative_matches_finite_difference() {
        // c''(log η) must be the derivative of the third term (c'(log η)) of
        // `bessel_i0_centered_terms`, across small, mid, and large arguments.
        // c''(log η) is the log-derivative of the STABLE third term `d1` (the
        // quantity the outer gradient's ARD normalizer channel reports), so the
        // self-consistent reference is a central difference of that same term.
        // The sweep straddles every seam this function has ever had: the retired
        // A&S 3.75 and 30 seams, and the live 1.0 (small-η rearrangement) and
        // 20.0 (series/asymptotic) ones.
        let first_log_derivative = |x: f64| bessel_i0_centered_terms(x).2;
        for eta in [
            0.02_f64, 0.05, 0.25, 0.999, 1.0, 1.001, 2.0, 3.5, 4.0, 8.0, 19.9, 20.1, 29.9, 30.1,
        ] {
            let log_eta = eta.ln();
            let analytic = bessel_i0_centered_second_log_derivative_from_log_abs(log_eta);

            let log_step = 1.0e-6_f64;
            let first_plus = first_log_derivative(eta * log_step.exp());
            let first_minus = first_log_derivative(eta * (-log_step).exp());
            let finite_difference = (first_plus - first_minus) / (2.0 * log_step);
            // `ε·|d1|/log_step ≈ 1e-10` of central-difference roundoff is the
            // floor here; the analytic value is far better than that. The old
            // `5e-5 + 1e-3·|analytic|` band was three orders wider than the
            // finite difference could even be wrong by — it was sized to the
            // 0.6% error the A&S polynomials put into `analytic`.
            assert!(
                (analytic - finite_difference).abs() < 1.0e-8 + 1.0e-6 * analytic.abs(),
                "centered Bessel second log-derivative mismatch at eta={eta}: \
                 analytic={analytic}, finite_difference={finite_difference}"
            );
        }
        // Large-η decay: the normalizer curvature vanishes like the leading
        // asymptotic term 1/(8η) (its Hessian contribution is then negligible
        // beside the ∝α energy term), stays finite and positive, and the
        // overflow-free gateway rounds it to exactly zero past the float range.
        // Held against THREE terms of the expansion rather than one, so the
        // admissible band is the size of the first omitted term (`≲ 2/η⁴`)
        // instead of a 25% shrug.
        for eta in [50.0_f64, 200.0, 1.0e4] {
            let c2 = bessel_i0_centered_second_log_derivative_from_log_abs(eta.ln());
            let inverse = 1.0 / eta;
            let expansion = inverse * (0.125 + inverse * (0.25 + inverse * (75.0 / 128.0)));
            assert!(
                c2 > 0.0 && (c2 - expansion).abs() < 8.0 * inverse.powi(4),
                "large-eta centered second derivative must track its own expansion; \
                 eta={eta}, c2={c2}, expansion={expansion}"
            );
        }
        // η → 0 and the overflow-free large-|η| gateway both round to 0.
        assert_eq!(
            bessel_i0_centered_second_log_derivative_from_log_abs(f64::NEG_INFINITY),
            0.0
        );
        assert_eq!(
            bessel_i0_centered_second_log_derivative_from_log_abs(1_200.0),
            0.0
        );
    }

    /// Every quantity `bessel_i0_centered_terms` and the second log-derivative
    /// return, against an INDEPENDENT 60-decimal-digit evaluation of the same
    /// closed forms (`mpmath.besseli`, `mpmath.diff`), rounded to `f64`.
    ///
    /// This is the assertion the module lacked. Everything else here is a
    /// self-consistency check — a finite difference of the evaluator against
    /// the evaluator — and a self-consistent evaluator can be uniformly wrong.
    /// The A&S 9.8.x polynomials this replaced were exactly that: internally
    /// consistent to the last digit and off the true value by up to `4e-6` in
    /// `d1` and `6e-3` in the curvature, with steps at their branch seams. No
    /// test in the tree compared them to anything but themselves.
    #[test]
    fn bessel_primitives_match_independent_high_precision_reference() {
        // (η, log I0(η) − η, I1(η)/I0(η), η(I1/I0 − 1), d²/d(log η)²[log I0 − η])
        const REFERENCE: [[f64; 5]; 24] = [
            [
                1e-06,
                -9.9999975e-07,
                4.999999999999375e-07,
                -9.999995e-07,
                -9.99999e-07,
            ],
            [
                0.001,
                -0.000999750000015625,
                0.0004999999375000105,
                -0.0009995000000625,
                -0.00099900000025,
            ],
            [
                0.05,
                -0.049375097629132,
                0.024992190753810217,
                -0.048750390462309494,
                -0.04750156152399669,
            ],
            [
                0.25,
                -0.23443561468661894,
                0.12403350191792471,
                -0.21899162452051882,
                -0.18846151934987648,
            ],
            [
                0.5,
                -0.4384502808145187,
                0.24249961258080194,
                -0.378750193709599,
                -0.2647015155254598,
            ],
            [
                1.0,
                -0.7640856414928213,
                0.4463899658965345,
                -0.5536100341034655,
                -0.19926400165310923,
            ],
            [
                2.0,
                -1.1760064585170438,
                0.697774657964008,
                -0.604450684071984,
                0.05244210681284669,
            ],
            [
                3.75,
                -1.5396457880279808,
                0.8531704594530685,
                -0.5506107770509933,
                0.0764086000777509,
            ],
            [
                5.0,
                -1.6953182241774665,
                0.8933831370440852,
                -0.5330843147795739,
                0.0466642611317311,
            ],
            [
                8.0,
                -1.941895744572186,
                0.9352354935294386,
                -0.5181160517644912,
                0.02141258513583364,
            ],
            [
                12.0,
                -2.1504975008971563,
                0.9573814053952422,
                -0.5114231352570932,
                0.01260162289404047,
            ],
            [
                17.0,
                -2.327961358737179,
                0.9701275885919403,
                -0.5078309939370159,
                0.008361475455484893,
            ],
            [
                19.5,
                -2.397561575434808,
                0.9740118676091061,
                -0.5067685816224307,
                0.007160287955186735,
            ],
            [
                19.999999,
                -2.410389546426233,
                0.9746705066059314,
                -0.5065898425518784,
                0.006960420318717729,
            ],
            [
                20.0,
                -2.4103895717557258,
                0.9746705078898071,
                -0.5065898422038575,
                0.006960419930170057,
            ],
            [
                20.000001,
                -2.410389597085217,
                0.9746705091736827,
                -0.5065898418558366,
                0.006960419541622429,
            ],
            [
                25.0,
                -2.5232719950007563,
                0.9797914534905159,
                -0.5052136627371017,
                0.005442291838848013,
            ],
            [
                30.0,
                -2.615298566828064,
                0.9831895553653361,
                -0.5043133390399173,
                0.004468398461442669,
            ],
            [
                64.0,
                -2.996411436485784,
                0.9921564935488112,
                -0.5019844128760834,
                0.002016497368136742,
            ],
            [
                150.0,
                -3.423420049648141,
                0.9966610736828279,
                -0.5008389475758167,
                0.0008446213361703931,
            ],
            [
                900.0,
                -4.319996948727984,
                0.9994442899516907,
                -0.5001390434784159,
                0.0001391983371050074,
            ],
            [
                10000.0,
                -5.524096218567699,
                0.999949998749875,
                -0.5000125012501954,
                1.2502500586100053e-05,
            ],
            [
                1000000.0,
                -7.826693687186747,
                0.999999499999875,
                -0.500000125000125,
                1.2500025000058594e-07,
            ],
            [
                1000000000000.0,
                -14.734449091168822,
                0.9999999999995,
                -0.500000000000125,
                1.2500000000025e-13,
            ],
        ];

        // Sized from the arithmetic, not from the outcome. The value and ratio
        // are read off sums that cannot cancel, so they land within a few ulp.
        // `d1` inherits `1/(1 − I1/I0)` ≈ 40x amplification at the top of the
        // ascending-series branch. The curvature inherits a further ≈ 2η from
        // `d1`'s ABSOLUTE error there, which peaks just under the crossover.
        const CENTERED_TOL: f64 = 4.0e-15;
        const RATIO_TOL: f64 = 4.0e-15;
        const D1_TOL: f64 = 1.0e-13;
        const CURVATURE_TOL: f64 = 1.0e-10;

        for [eta, want_centered, want_ratio, want_d1, want_curvature] in REFERENCE {
            let (centered, ratio, d1) = bessel_i0_centered_terms(eta);
            let curvature = bessel_i0_centered_second_log_derivative_from_log_abs(eta.ln());
            let relative = |got: f64, want: f64| (got - want).abs() / want.abs();
            assert!(
                relative(centered, want_centered) < CENTERED_TOL,
                "log I0({eta}) − {eta}: got {centered:.17e}, want {want_centered:.17e}"
            );
            assert!(
                relative(ratio, want_ratio) < RATIO_TOL,
                "I1/I0({eta}): got {ratio:.17e}, want {want_ratio:.17e}"
            );
            assert!(
                relative(d1, want_d1) < D1_TOL,
                "η(I1/I0 − 1) at {eta}: got {d1:.17e}, want {want_d1:.17e}"
            );
            assert!(
                relative(curvature, want_curvature) < CURVATURE_TOL,
                "c''(log η) at {eta}: got {curvature:.17e}, want {want_curvature:.17e}"
            );
        }
    }

    /// The three returned terms are computed from DIFFERENT representations on
    /// the asymptotic branch — `ratio` from `S1/S0`, `d1` from the symbolically
    /// pre-cancelled difference series — so their defining relations are a real
    /// cross-check there, not a tautology. Both must hold to within the
    /// cancellation the naive form suffers and the pre-cancelled one avoids.
    #[test]
    fn bessel_centered_terms_satisfy_their_defining_relations() {
        for eta in [
            0.5_f64, 1.0, 5.0, 12.0, 19.999, 20.0, 20.001, 25.0, 64.0, 900.0, 1.0e6, 1.0e12,
        ] {
            let (_centered, ratio, d1) = bessel_i0_centered_terms(eta);
            // d1 ≡ η(I1/I0 − 1). Forming it this way subtracts two numbers that
            // agree to `1/(2η)`, so it is only good to `≈ ε·η` — which is the
            // whole reason `d1` is carried separately.
            let naive = eta * (ratio - 1.0);
            assert!(
                (d1 - naive).abs() <= 8.0 * f64::EPSILON * eta,
                "d1 must equal η(I1/I0 − 1) at eta={eta}: d1={d1:.17e}, naive={naive:.17e}"
            );

            // c''(s) ≡ −η(2·d1 + 1) − d1², the rearrangement's starting point.
            // Both sides are fed the log-round-tripped argument the function
            // itself sees, so the ONLY admissible difference is the rounding of
            // the rearranged grouping: the two intermediate products are of
            // size `2η|d1|` and `d1²`, so a few ulp of those is the budget.
            //
            // Only checked where the naive form still HAS digits. Its two terms
            // both approach ¼ and cancel down to `1/(8η)`, which costs `≈ 8εη²`
            // in relative terms — already `1e-12` at η = 64 and total loss by
            // `η ≈ 1e8`. That collapse is the whole reason for the rearrangement,
            // so asserting agreement past it would assert nothing.
            if eta <= 64.0 {
                let round_tripped = eta.ln().exp();
                let (_, _, same_d1) = bessel_i0_centered_terms(round_tripped);
                let curvature = bessel_i0_centered_second_log_derivative_from_log_abs(eta.ln());
                let naive = -round_tripped * (2.0 * same_d1 + 1.0) - same_d1 * same_d1;
                let budget =
                    8.0 * f64::EPSILON * (2.0 * round_tripped * same_d1.abs() + same_d1 * same_d1);
                assert!(
                    (curvature - naive).abs() <= budget,
                    "c'' must equal −η(2d1+1) − d1² at eta={eta}: \
                     c2={curvature:.17e}, naive={naive:.17e}, budget={budget:.3e}"
                );
            }

            // `I1 < I0` for every η > 0, so `I1/I0 ∈ (0,1)` and `d1 < 0`. `d1`
            // is NOT monotone: it falls to a global minimum
            // `−0.608891247247801…` at `η = 1.702379944878764…` (the root of
            // `d(d1)/dη`) before rising back to its `−½` limit, so it crosses
            // `−½` once and the useful two-sided bound is that minimum.
            assert!((0.0..1.0).contains(&ratio), "I1/I0({eta})={ratio} ∉ (0,1)");
            assert!(
                (-0.608_891_247_247_802..0.0).contains(&d1),
                "η(I1/I0 − 1) at {eta} is {d1}, outside (min d1, 0)"
            );
        }
    }

    /// A branch crossover must not be observable in the output. The retired A&S
    /// pair stepped by `4e-6` in `d1` and `2e-4` in the curvature at its own
    /// 3.75 seam — a jump discontinuity in the objective and gradient an outer
    /// optimizer differentiates through.
    #[test]
    fn bessel_branch_crossovers_have_no_step() {
        // Every seam the implementation has ever carried.
        for seam in [1.0_f64, 3.75, 20.0, 30.0] {
            let delta = 1.0e-11 * seam;
            let (below_c, below_r, below_d1) = bessel_i0_centered_terms(seam - delta);
            let (above_c, above_r, above_d1) = bessel_i0_centered_terms(seam + delta);
            let below_c2 =
                bessel_i0_centered_second_log_derivative_from_log_abs((seam - delta).ln());
            let above_c2 =
                bessel_i0_centered_second_log_derivative_from_log_abs((seam + delta).ln());

            // Over `2δ` the true functions can move by at most `2δ·|f'|`, and
            // every derivative here is bounded by 1 in magnitude. Anything past
            // that plus a few ulp is a step, not a slope.
            let slope_budget = 2.0 * delta + 1.0e-14;
            assert!(
                (above_c - below_c).abs() < slope_budget,
                "centered log steps at the {seam} seam: {below_c:.17e} -> {above_c:.17e}"
            );
            assert!(
                (above_r - below_r).abs() < slope_budget,
                "I1/I0 steps at the {seam} seam: {below_r:.17e} -> {above_r:.17e}"
            );
            assert!(
                (above_d1 - below_d1).abs() < slope_budget,
                "d1 steps at the {seam} seam: {below_d1:.17e} -> {above_d1:.17e}"
            );
            assert!(
                (above_c2 - below_c2).abs() < slope_budget,
                "c'' steps at the {seam} seam: {below_c2:.17e} -> {above_c2:.17e}"
            );
        }
    }

    /// Non-finite and boundary arguments keep their documented limits, and no
    /// series loop can run away on them.
    #[test]
    fn bessel_primitives_handle_boundary_arguments() {
        let (centered, ratio, d1) = bessel_i0_centered_terms(f64::INFINITY);
        assert_eq!((centered, ratio, d1), (f64::NEG_INFINITY, 1.0, -0.5));
        let (centered, ratio, d1) = bessel_i0_centered_terms(f64::NEG_INFINITY);
        assert_eq!((centered, ratio, d1), (f64::NEG_INFINITY, 1.0, -0.5));

        let (centered, ratio, d1) = bessel_i0_centered_terms(f64::NAN);
        assert!(centered.is_nan() && ratio.is_nan() && d1.is_nan());
        assert!(bessel_i0_centered_second_log_derivative_from_log_abs(f64::NAN).is_nan());

        // I0 and I1 are even/odd, so every returned term is a function of |η|.
        for eta in [0.5_f64, 5.0, 25.0, 1.0e6] {
            assert_eq!(
                bessel_i0_centered_terms(-eta),
                bessel_i0_centered_terms(eta)
            );
        }
    }

    #[test]
    fn gauss_legendre_integrates_polynomials_exactly() {
        // An n-point rule is exact for polynomials of degree ≤ 2n−1.
        for n in [1usize, 2, 3, 5, 8, 40, 64] {
            let (nodes, weights) = gauss_legendre(n);
            assert_eq!(nodes.len(), n);
            assert_eq!(weights.len(), n);
            assert!(nodes.windows(2).all(|w| w[0] < w[1]), "nodes ascending");
            if !n.is_multiple_of(2) {
                assert_eq!(nodes[n / 2], 0.0, "odd-n central node is exact zero");
            }
            let total: f64 = weights.iter().sum();
            assert!((total - 2.0).abs() < 1e-13, "∫1 dx = 2, got {total}");
            if n >= 2 {
                let x2: f64 = nodes.iter().zip(&weights).map(|(x, w)| w * x * x).sum();
                assert!((x2 - 2.0 / 3.0).abs() < 1e-13, "∫x² dx = 2/3, got {x2}");
            }
        }
    }

    #[test]
    fn binom_k_exceeds_n_returns_zero() {
        assert_eq!(binomial_coefficient_f64(3, 5), 0.0);
        assert_eq!(binomial_coefficient_f64(0, 1), 0.0);
        assert_eq!(binomial_coefficient_f64(10, 11), 0.0);
    }

    #[test]
    fn binom_k_zero_returns_one() {
        assert_eq!(binomial_coefficient_f64(0, 0), 1.0);
        assert_eq!(binomial_coefficient_f64(5, 0), 1.0);
        assert_eq!(binomial_coefficient_f64(100, 0), 1.0);
    }

    #[test]
    fn binom_k_equals_n_returns_one() {
        assert_eq!(binomial_coefficient_f64(1, 1), 1.0);
        assert_eq!(binomial_coefficient_f64(5, 5), 1.0);
        assert_eq!(binomial_coefficient_f64(20, 20), 1.0);
    }

    #[test]
    fn binom_small_exact_values() {
        assert_eq!(binomial_coefficient_f64(5, 2), 10.0);
        assert_eq!(binomial_coefficient_f64(10, 3), 120.0);
        assert_eq!(binomial_coefficient_f64(20, 10), 184_756.0);
        assert_eq!(binomial_coefficient_f64(6, 3), 20.0);
    }

    #[test]
    fn binom_symmetry() {
        assert_eq!(
            binomial_coefficient_f64(10, 3),
            binomial_coefficient_f64(10, 7)
        );
        assert_eq!(
            binomial_coefficient_f64(20, 5),
            binomial_coefficient_f64(20, 15)
        );
        assert_eq!(
            binomial_coefficient_f64(54, 24),
            binomial_coefficient_f64(54, 30)
        );
    }

    #[test]
    fn binom_c54_24_is_exact() {
        // The u128-recurrence fix restored this value (old f64 recurrence
        // returned 1_402_659_561_581_459, one short of the true integer).
        assert_eq!(binomial_coefficient_f64(54, 24), 1_402_659_561_581_460.0);
    }

    #[test]
    fn poly_exp_empty_coeffs_returns_zero() {
        assert_eq!(stable_polynomial_times_exp_neg(1.0, &[]), 0.0);
        assert_eq!(stable_polynomial_times_exp_neg(0.0, &[]), 0.0);
        assert_eq!(stable_polynomial_times_exp_neg(700.0, &[]), 0.0);
    }

    #[test]
    fn poly_exp_nonfinite_x_returns_zero() {
        assert_eq!(
            stable_polynomial_times_exp_neg(f64::INFINITY, &[1.0, 2.0]),
            0.0
        );
        assert_eq!(
            stable_polynomial_times_exp_neg(f64::NEG_INFINITY, &[1.0, 2.0]),
            0.0
        );
        assert_eq!(stable_polynomial_times_exp_neg(f64::NAN, &[1.0]), 0.0);
    }

    #[test]
    fn poly_exp_constant_at_zero() {
        // At x=0: poly(0) = coeffs[0], exp(0)=1 → result = coeffs[0].
        assert_eq!(stable_polynomial_times_exp_neg(0.0, &[5.0]), 5.0);
        assert_eq!(stable_polynomial_times_exp_neg(0.0, &[3.0, 1.0, 2.0]), 3.0);
    }

    #[test]
    fn poly_exp_constant_poly_direct_path() {
        // x=2.0 < 600: direct Horner * exp(-x).
        let x = 2.0;
        let got = stable_polynomial_times_exp_neg(x, &[3.0]);
        let expected = 3.0 * (-x).exp();
        assert!(
            (got - expected).abs() < 1e-14,
            "got={got} expected={expected}"
        );
    }

    #[test]
    fn poly_exp_linear_poly_direct_path() {
        // coeffs = [a, b] → poly = a + b*x.
        let x = 1.5;
        let (a, b) = (2.0, 3.0);
        let got = stable_polynomial_times_exp_neg(x, &[a, b]);
        let expected = (a + b * x) * (-x).exp();
        assert!(
            (got - expected).abs() < 1e-14,
            "got={got} expected={expected}"
        );
    }

    #[test]
    fn poly_exp_constant_poly_asymptotic_path() {
        // x=700 > 600: asymptotic path. For poly = [1.0], result = exp(-700).
        let x = 700.0_f64;
        let got = stable_polynomial_times_exp_neg(x, &[1.0]);
        let expected = (-x).exp();
        let rel = (got - expected).abs() / expected;
        assert!(rel < 1e-12, "got={got} expected={expected} rel={rel}");
    }

    #[test]
    fn poly_exp_quadratic_asymptotic_path() {
        // x=620 > 600: poly = x^2 (coeffs=[0,0,1]). Result = x^2 * exp(-x).
        // x=800 would underflow to 0.0 in both the asymptotic path and the
        // reference, making the relative-error check degenerate; x=620 keeps
        // the result in the normal f64 range (~10^-264) while still exercising
        // the asymptotic branch (threshold is x=600).
        let x = 620.0_f64;
        let got = stable_polynomial_times_exp_neg(x, &[0.0, 0.0, 1.0]);
        let expected = (2.0 * x.ln() - x).exp();
        let rel = (got - expected).abs() / expected.abs();
        assert!(rel < 1e-12, "got={got} expected={expected} rel={rel}");
    }
}
