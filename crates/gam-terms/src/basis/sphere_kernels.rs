//! Closed-form and spectral zonal Wahba kernels on S².
//!
//! This module owns the scalar/SIMD kernel dispatch for intrinsic sphere
//! smooths. Callers in `basis` handle data validation, coordinate transforms,
//! and matrix assembly.

use super::BasisError;
use super::polylog::{dilog_unit, trilog_unit};
use super::sphere_half_angle::HalfAngleSeparation;
use super::sphere_spec::SphereWahbaKernel;
use super::sphere_spectral::{
    pseudo_s2_truncated_coefficients, sobolev_s2_truncated_coefficients,
    sphere_truncated_spectral_derivative_eval, sphere_truncated_spectral_eval,
};

/// Exact coincident-point value `K_m^{pseudo}(γ = 0)` of the pseudo-spline
/// Wahba kernel, in closed form.
///
/// Unlike the Sobolev kernel at `m = 1`, this limit is FINITE and known
/// exactly, so the diagonal of a pseudo-spline Gram matrix is a theorem rather
/// than a regularization choice. From the spectral coefficients that
/// [`super::sphere_spectral::pseudo_s2_truncated_coefficients`] builds,
///
/// ```text
/// K_m(0) = Σ_{ℓ≥1} c_ℓ,   c_ℓ = 2 / (4π · Π_{k=1..m+1} (ℓ + k)),
/// ```
///
/// and the reciprocal-Pochhammer sum telescopes
/// (`Σ_{j≥0} 1/((j+a)···(j+a+p−1)) = 1 / ((p−1)·(a)···(a+p−2))`, here with
/// `a = 2` and `p = m+1`) to `Σ_{ℓ≥1} 1/((ℓ+1)···(ℓ+m+1)) = 1/(m·(m+1)!)`.
/// Hence
///
/// ```text
/// K_m^{pseudo}(0) = 1 / (2π · m · (m+1)!)
/// ```
///
/// giving `1/(4π)`, `1/(24π)`, `1/(144π)`, `1/(960π)` for `m = 1..4`. Each was
/// checked against the truncated spectral sum to its own truncation tail.
///
/// `m` is clamped to `4` for `m > 4` so this agrees with the `_` fallback arm
/// of [`wahba_sphere_kernel_pseudo_from_cos`], which evaluates the `m = 4`
/// polynomial for any `m ≥ 4`.
#[inline]
pub(crate) fn wahba_sphere_kernel_pseudo_coincident(m: usize) -> f64 {
    let m_eff = m.clamp(1, 4);
    let factorial = (1..=(m_eff + 1)).map(|k| k as f64).product::<f64>();
    1.0 / (2.0 * std::f64::consts::PI * (m_eff as f64) * factorial)
}

/// Pseudo-spline Wahba kernel on S² (mgcv `makeR`-style closed form), as a
/// function of the half-angle separation `u = sin²(γ/2)`.
#[inline]
pub(crate) fn wahba_sphere_kernel_pseudo(sep: HalfAngleSeparation, m: usize) -> f64 {
    // `w` IS the half-angle separation: the closed forms below are polynomials
    // in `w = (1 − cos γ)/2 = sin²(γ/2)` with a `√w` and a `ln(1 + 1/√w)`, so
    // the kernel takes `u` straight through.
    //
    // Coincident points are the ONLY inputs the old `z.max(f64::EPSILON*1e-4)`
    // floor could ever bind on, and there the kernel has an exact finite limit.
    // Taking the analytic limit here is bit-identical to the floored form at
    // every input the old `cos γ` route could produce, where a positive `u` was
    // quantized to at least `2⁻⁵⁴ ≈ 5.6e-17` — three and a half orders above
    // that floor, which sat at `1.11e-20` in `w` — and it replaces a 4.2e-10
    // relative error at coincidence (the floor's `-2√w` term, `O(√floor)`) with
    // the exact value. Now that `u` arrives in haversine form (#2489) it can be
    // genuinely tiny rather than quantized, which is what makes the floor's
    // absence load-bearing rather than merely tidy.
    //
    // Shrinking the floor rather than removing it would also have recovered
    // the limits (down to `f64::MIN_POSITIVE`), but not unconditionally: half
    // a step further, at the smallest subnormal, `√w` underflows to zero,
    // `1.0/c0` is `+∞`, and the `2·a·w` term becomes `∞ · 0 = NaN`. The
    // analytic limit has no such cliff, and needs no constant to state.
    let w = sep.u;
    if w <= 0.0 {
        return wahba_sphere_kernel_pseudo_coincident(m);
    }
    let c0 = w.sqrt();
    let a = (1.0 + 1.0 / c0).ln();
    let c = 2.0 * c0;
    let two_pi = 2.0 * std::f64::consts::PI;
    match m {
        1 => {
            let q1 = 2.0 * a * w - c + 1.0;
            (q1 - 0.5) / two_pi
        }
        2 => {
            let w2 = w * w;
            let q2 = a * (6.0 * w2 - 2.0 * w) - 3.0 * c * w + 3.0 * w + 0.5;
            (q2 / 2.0 - 1.0 / 6.0) / two_pi
        }
        3 => {
            let w2 = w * w;
            let w3 = w2 * w;
            let q3 = (a * (60.0 * w3 - 36.0 * w2) + 30.0 * w2 + c * (8.0 * w - 30.0 * w2)
                - 3.0 * w
                + 1.0)
                / 3.0;
            (q3 / 6.0 - 1.0 / 24.0) / two_pi
        }
        _ => {
            let w2 = w * w;
            let w3 = w2 * w;
            let w4 = w3 * w;
            let q4 = a * (70.0 * w4 - 60.0 * w3 + 6.0 * w2)
                + 35.0 * w3 * (1.0 - c)
                + c * 55.0 * w2 / 3.0
                - 12.5 * w2
                - w / 3.0
                + 0.25;
            (q4 / 24.0 - 1.0 / 120.0) / two_pi
        }
    }
}

/// Coincident-point limit of `dK_m^{pseudo}/du` at `u = sin²(γ/2) = 0`, in
/// closed form — where it exists.
///
/// From the spectral representation `K = Σ_{ℓ≥1} c_ℓ P_ℓ(cos γ)` with
/// `c_ℓ = 2 / (4π · Π_{k=1..m+1}(ℓ + k))` (see
/// [`wahba_sphere_kernel_pseudo_coincident`]) and `P'_ℓ(1) = ℓ(ℓ+1)/2`,
///
/// ```text
/// dK/du|₀ = −2 · dK/d(cos γ)|₀ = −Σ_{ℓ≥1} ℓ(ℓ+1) c_ℓ
///         = −(1/2π) · Σ_{ℓ≥1} ℓ / ((ℓ+2)(ℓ+3)···(ℓ+m+1)).
/// ```
///
/// Splitting `ℓ = (ℓ+2) − 2` and telescoping each half with
/// `Σ_{ℓ≥1} 1/((ℓ+a)···(ℓ+a+p−1)) = 1/((p−1)·(1+a)···(a+p−1))` gives
///
/// ```text
/// Σ_{ℓ≥1} ℓ / ((ℓ+2)···(ℓ+m+1)) = 6/((m−2)(m+1)!) − 4/((m−1)(m+1)!)
///                                = 2(m+1) / ((m−2)(m−1)(m+1)!)
/// ```
///
/// so
///
/// ```text
/// dK_m^{pseudo}/du|₀ = −(m+1) / (π (m−2)(m−1)(m+1)!)
/// ```
///
/// giving `−1/(12π)` for `m = 3` and `−1/(144π)` for `m = 4`. The `m − 2`
/// factor is the statement that the sum DIVERGES for `m ∈ {1, 2}`: those
/// kernels have a genuine cusp at coincidence (`m = 1` diverges like
/// `−1/(2π√u)` from the `−2√u` term, `m = 2` logarithmically), so `None` is
/// returned and the caller must resolve the cusp rather than being handed a
/// finite lie. Both limits were checked against the `w → 0` limit of the
/// polynomial in [`wahba_sphere_kernel_pseudo_derivative_dhav`] and against the
/// truncated spectral sum.
#[inline]
fn wahba_sphere_kernel_pseudo_derivative_coincident(m: usize) -> Option<f64> {
    let m_eff = m.clamp(1, 4);
    if m_eff < 3 {
        return None;
    }
    let m_f = m_eff as f64;
    let factorial = (1..=(m_eff + 1)).map(|k| k as f64).product::<f64>();
    Some(-(m_f + 1.0) / (std::f64::consts::PI * (m_f - 2.0) * (m_f - 1.0) * factorial))
}

/// Exact derivative `dK_m^{pseudo}/du` of the pseudo-spline Wahba kernel
/// [`wahba_sphere_kernel_pseudo`] with respect to `u = sin²(γ/2)`.
///
/// The forward kernel is a polynomial in `w = u` with the auxiliary terms
/// `c0 = sqrt(w)`, `c = 2 c0`, and `a = ln(1 + 1/c0)`; this differentiates it
/// in `w` directly. The `u` form is the one the jet wants — see
/// [`super::sphere_half_angle::half_angle_partials`] for why the `cos γ` chain
/// cannot compute the cusp gradient.
///
/// At exact coincidence `m ∈ {3, 4}` have finite limits (returned by
/// [`wahba_sphere_kernel_pseudo_derivative_coincident`]) and `m ∈ {1, 2}` are
/// genuinely `−∞`. Returning the infinity is deliberate: it used to be masked
/// by a `w.max(f64::EPSILON * 1e-4)` floor that reported
/// `-1/(2π·1.05e-10) = −1.5e9` for a cusp of infinite slope, which is neither
/// the limit nor a diagnosable value. Callers that need the JET at coincidence
/// resolve it there, where `∂u/∂φ` is exactly `0` and the cusp's one-sided
/// gradients differ only in sign.
#[inline]
pub(crate) fn wahba_sphere_kernel_pseudo_derivative_dhav(
    sep: HalfAngleSeparation,
    m: usize,
) -> f64 {
    let w = sep.u;
    if w <= 0.0 {
        return wahba_sphere_kernel_pseudo_derivative_coincident(m).unwrap_or(f64::NEG_INFINITY);
    }
    let c0 = w.sqrt();
    let a = (1.0 + 1.0 / c0).ln();
    let c = 2.0 * c0;
    let two_pi = 2.0 * std::f64::consts::PI;
    let da_dw = -1.0 / (2.0 * c0 * c0 * (c0 + 1.0));
    let dc_dw = 1.0 / c0;
    let dk_dw = match m {
        1 => {
            let dq1_dw = 2.0 * a + 2.0 * w * da_dw - dc_dw;
            dq1_dw / two_pi
        }
        2 => {
            let dq2_dw = da_dw * (6.0 * w * w - 2.0 * w) + a * (12.0 * w - 2.0)
                - 3.0 * (dc_dw * w + c)
                + 3.0;
            (dq2_dw / 2.0) / two_pi
        }
        3 => {
            let w2 = w * w;
            let w3 = w2 * w;
            let dinner_dw = da_dw * (60.0 * w3 - 36.0 * w2)
                + a * (180.0 * w2 - 72.0 * w)
                + 60.0 * w
                + (dc_dw * (8.0 * w - 30.0 * w2) + c * (8.0 - 60.0 * w))
                - 3.0;
            let dq3_dw = dinner_dw / 3.0;
            (dq3_dw / 6.0) / two_pi
        }
        _ => {
            let w2 = w * w;
            let w3 = w2 * w;
            let w4 = w3 * w;
            let dq4_dw = da_dw * (70.0 * w4 - 60.0 * w3 + 6.0 * w2)
                + a * (280.0 * w3 - 180.0 * w2 + 12.0 * w)
                + 35.0 * (3.0 * w2 * (1.0 - c) - w3 * dc_dw)
                + (55.0 / 3.0) * (dc_dw * w2 + c * 2.0 * w)
                - 25.0 * w
                - 1.0 / 3.0;
            (dq4_dw / 24.0) / two_pi
        }
    };
    dk_dw
}

// ============================================================================
// Wahba/Sobolev kernel on S²
// ============================================================================
//
// `K_m^{Sobolev}(gamma) = (1/4pi) * sum_{l >= 1} (2l + 1)
// * [l(l + 1)]^{-m} * P_l(cos gamma)`.
//
// For `m in {1, 2, 3}` we use the closed-form expressions derived in
// Beatson & zu Castell, "Thinplate Splines on the Sphere", SIGMA 14 (2018)
// 083 (Section 6.2). For `m = 4`, we fall back to a truncated Legendre series.

/// Sobolev `K_m^{Sobolev}` reproducing kernel on S², closed-form for
/// `m in {1, 2, 3}` plus spectral series for `m = 4`, as a function of the
/// half-angle separation.
///
/// Every closed form here is a function of `u = sin²(γ/2)` and
/// `v = cos²(γ/2) = 1 − u`, which is why the separation is carried as the pair:
/// `m = 1` needs `−ln u` (singular at coincidence) and `m = 2` needs `Li₂(v)`
/// (whose argument vanishes at the antipode). Taking `v` as `1.0 - u` instead
/// destroys the antipodal end — at `cos γ = −1 + 1e-16`, `u` rounds to `1.0`
/// and `1.0 - u` is `0`, reporting an exact antipode for a pair that is not one.
#[inline]
pub(crate) fn wahba_sphere_kernel_sobolev(sep: HalfAngleSeparation, m: usize) -> f64 {
    let four_pi = 4.0 * std::f64::consts::PI;
    let pi2_6 = std::f64::consts::PI * std::f64::consts::PI / 6.0;
    // No `f64::EPSILON * 1.0e-4` floor on either half. Both polylogarithms
    // already carry their endpoints exactly (`Li₂(0) = Li₃(0) = 0`,
    // `Li₂(1) = π²/6`, `Li₃(1) = ζ₃`), so the only thing the floor did was to
    // keep `u.ln()` off `-∞` — and it did that by evaluating the kernel at a
    // separation of `1.5e-10` rad instead of at `0`, which is a choice of
    // resolution, not a limit (#2469, #2475). The two places `ln u` appears are
    // handled on their own terms below.
    let u = sep.u;
    let one_minus_u = sep.v;
    match m {
        // `K_1 = (−ln u − 1)/4π` is genuinely `+∞` at coincidence: the m = 1
        // Sobolev Gram diagonal does not exist, which is exactly why
        // `validate_spherical_wahba_gram_request` refuses to build one. Letting
        // the infinity through is the honest report — every public entry point
        // checks `is_finite` — where the floor returned `45.0/4π` and looked
        // like an answer.
        1 => (-u.ln() - 1.0) / four_pi,
        2 => (dilog_unit(one_minus_u) + 1.0 - pi2_6) / four_pi,
        3 => {
            const ZETA3: f64 = 1.2020569031595942853997381615114499907649862923404988817922;
            let li3_u = trilog_unit(u);
            let li2_one_minus_u = dilog_unit(one_minus_u);
            // `ln(u)·Li₂(u)` is `−∞ · 0` at coincidence with the REMOVABLE
            // limit `0`, since `Li₂(u) = u + u²/4 + … = O(u)` and `u ln u → 0`.
            // Resolving it analytically is what makes `K_3(0) = (2ζ₃ − 2)/4π`
            // exact rather than floor-dependent.
            let cross = if u <= 0.0 {
                0.0
            } else {
                u.ln() * dilog_unit(u)
            };
            (-2.0 * li3_u - li2_one_minus_u + cross + 2.0 * ZETA3 + pi2_6 - 2.0) / four_pi
        }
        _ => wahba_sphere_kernel_sobolev_spectral(sep.cos_gamma(), m),
    }
}

/// Spectral Legendre-series evaluation of the Sobolev kernel
/// `K_m^{Sobolev}(gamma) = (1/4pi) sum_{l >= 1} (2l+1) *
/// [l(l+1)]^{-m} * P_l(cos gamma)`.
#[inline]
pub(crate) fn wahba_sphere_kernel_sobolev_spectral(cos_gamma: f64, m: usize) -> f64 {
    let l_max = match m {
        1 => 4096_usize,
        2 => 256,
        3 => 128,
        _ => 96,
    };
    let x = cos_gamma.clamp(-1.0, 1.0);
    let m_i = m as i32;
    let four_pi = 4.0 * std::f64::consts::PI;
    let mut p_l_minus_1 = 1.0_f64;
    let mut p_l = x;
    let mut sum = 3.0 * p_l / (four_pi * 2.0_f64.powi(m_i));
    for l in 1..l_max {
        let p_l_plus_1 =
            ((2 * l + 1) as f64 * x * p_l - (l as f64) * p_l_minus_1) / ((l + 1) as f64);
        let ell = (l + 1) as f64;
        let eigen = (ell * (ell + 1.0)).powi(m_i);
        let weight = (2.0 * ell + 1.0) / four_pi;
        sum += weight * p_l_plus_1 / eigen;
        p_l_minus_1 = p_l;
        p_l = p_l_plus_1;
    }
    sum
}

/// Evaluate the Wahba sphere reproducing kernel at a single half-angle
/// separation.
#[inline]
pub(crate) fn wahba_sphere_kernel_kind(
    sep: HalfAngleSeparation,
    penalty_order: usize,
    kernel: SphereWahbaKernel,
) -> Result<f64, BasisError> {
    if !(1..=4).contains(&penalty_order) {
        crate::bail_invalid_basis!(
            "spherical spline penalty_order must be one of 1, 2, 3, 4; got {penalty_order}"
        );
    }
    let value = wahba_sphere_kernel_kind_unchecked(sep, penalty_order, kernel);
    if !value.is_finite() {
        crate::bail_invalid_basis!("spherical spline kernel produced a non-finite value");
    }
    Ok(value)
}

/// The kernel dispatch itself, with `penalty_order` and finiteness already
/// established (or, on the SIMD path, established once for the whole vector).
///
/// This is the single place the four [`SphereWahbaKernel`] variants are mapped
/// to their evaluators; the scalar and SIMD entry points differ only in how they
/// loop over it.
#[inline]
fn wahba_sphere_kernel_kind_unchecked(
    sep: HalfAngleSeparation,
    penalty_order: usize,
    kernel: SphereWahbaKernel,
) -> f64 {
    match kernel {
        SphereWahbaKernel::Sobolev => wahba_sphere_kernel_sobolev(sep, penalty_order),
        SphereWahbaKernel::Pseudo => wahba_sphere_kernel_pseudo(sep, penalty_order),
        SphereWahbaKernel::SobolevTruncated { lmax } => {
            let coeffs = sobolev_s2_truncated_coefficients(lmax as usize, penalty_order);
            sphere_truncated_spectral_eval(sep.cos_gamma(), &coeffs)
        }
        SphereWahbaKernel::PseudoTruncated { lmax } => {
            let coeffs = pseudo_s2_truncated_coefficients(lmax as usize, penalty_order);
            sphere_truncated_spectral_eval(sep.cos_gamma(), &coeffs)
        }
    }
}

/// SIMD lane-wise evaluation over four half-angle separations. Both Sobolev and
/// pseudo-spline branches are scalar-per-lane because the closed forms contain
/// non-vector elementary and polylogarithm calls; what the vector form buys is
/// the *separation* arithmetic (see
/// [`super::sphere_half_angle::half_angle_separation`]), which is pure `+ − ×`.
#[inline]
pub(crate) fn wahba_sphere_kernel_simd_kind(
    u: wide::f64x4,
    v: wide::f64x4,
    penalty_order: usize,
    kernel: SphereWahbaKernel,
) -> wide::f64x4 {
    use wide::f64x4;
    if !(1..=4).contains(&penalty_order) {
        return f64x4::from(f64::NAN);
    }
    let zero = f64x4::ZERO;
    let u_lanes = u.fast_max(zero).fast_min(f64x4::ONE).to_array();
    let v_lanes = v.fast_max(zero).fast_min(f64x4::ONE).to_array();
    let mut out = [0.0_f64; 4];
    for lane in 0..4 {
        let sep = HalfAngleSeparation {
            u: u_lanes[lane],
            v: v_lanes[lane],
        };
        out[lane] = wahba_sphere_kernel_kind_unchecked(sep, penalty_order, kernel);
    }
    f64x4::from(out)
}

/// Spectral derivative of the Sobolev sphere kernel w.r.t. `cos gamma`.
/// Exact closed-form derivative `dK_m^{Sobolev}/d(cos gamma)` for
/// `m in {1, 2, 3}`, differentiating the SAME polylogarithm closed forms used
/// by [`wahba_sphere_kernel_sobolev_closed_form`] so the design jet aligns with
/// the forward design to full precision (the slowly-convergent spectral
/// derivative series below was accurate enough for the kernel VALUE but lost
/// ~1.8 relative error on its DERIVATIVE at low `m`).
///
/// With `u = (1 - cos gamma)/2`, `du/d(cos gamma) = -1/2`:
///   m=1: K = (-ln u - 1)/(4π)              ⇒ dK/du = -1/(4π u)
///   m=2: K = (Li₂(1-u) + 1 - π²/6)/(4π)    ⇒ dK/du = ln(u)/((1-u)·4π)
///   m=3: K = (-2Li₃(u) - Li₂(1-u) + ln(u)·Li₂(u) + 2ζ₃ + π²/6 - 2)/(4π)
///        ⇒ dK/du = [-Li₂(u)/u - ln(u)/(1-u) - ln(u)·ln(1-u)/u]/(4π)
/// using d Li₂(z)/dz = -ln(1-z)/z and d Li₃(z)/dz = Li₂(z)/z.
#[inline]
fn wahba_sphere_kernel_sobolev_closed_form_derivative_dhav(
    sep: HalfAngleSeparation,
    m: usize,
) -> f64 {
    let four_pi = 4.0 * std::f64::consts::PI;
    // No floor on `u` or `v`. The sole caller only reaches this closed form
    // below the COINCIDENT pole, which bounds `u = sin²(γ/2)` away from `0`
    // by `5e-11` — nine orders above the `f64::EPSILON * 1.0e-4` floor these
    // lines used to carry (a factor of 2.3e9), so it could never bind.
    // Flooring here was dead arithmetic that read as if the singularities were
    // being handled (#2469, #2475 site 4).
    //
    // `1 - u` is carried as the separation's own `v = cos²(γ/2)` rather than as
    // `1.0 - u`, because `v` is the quantity that vanishes at the ANTIPODE and
    // only this form resolves it. Going through `u` instead destroys the
    // antipodal end outright — at `cos γ = -1 + 1e-16`, `1 - cos γ` rounds to
    // `2.0`, so `u` rounds to `1.0` and `1.0 - u` is `0`, reporting an exact
    // antipode for a pair that is not one. Both halves reach here already taken
    // from the side where they are small and exact (#2489).
    let u = sep.u;
    let v = sep.v;
    // assert!, not debug_assert!: the ban-scanner forbids debug_assert (silent
    // in release → debug/release divergence). An O(1) comparison in front of a
    // dilogarithm is free.
    //
    // Only `u > 0` is asserted. `v == 0` is the antipode, which is an ordinary
    // interior point of all three closed forms — every one of them is finite
    // and smooth there — and is handled by the removable-singularity arms
    // below rather than excluded.
    assert!(
        u > 0.0,
        "closed-form Sobolev derivative called at the coincident pole \
         (u = sin²(γ/2) = {u}); the caller's POLE_LIMIT_THRESHOLD guard is \
         supposed to make this unreachable"
    );
    // `ln u`, taken from whichever of the two exact halves is the small one.
    // `ln_1p(-v)` keeps full relative accuracy as `v → 0` (where `ln u → 0` and
    // is about to be divided by `v`); `u.ln()` keeps it as `u → 0`, where
    // `ln_1p` would have to reconstitute a tiny `1 - v` and lose the digits.
    let ln_u = if v <= 0.5 { (-v).ln_1p() } else { u.ln() };
    // `ln(1-u) = ln v`, by the same rule mirrored. Taking it as `v.ln()` near
    // the COINCIDENT end costs relative accuracy for exactly the reason `ln u`
    // costs it near the antipodal end: at `u = 5e-11` the true `ln v` is
    // `-5e-11`, but `v` can only carry `1 - 5e-11` to an absolute `1.1e-16`, so
    // the answer arrives with `2.2e-6` relative error. That error was reaching
    // the m=3 derivative — measured 2.4e-6 against a 40-digit reference at
    // `cos γ = 1 - 1e-10`, against `< 2e-14` everywhere else on the branch.
    let ln_v = if u <= 0.5 { (-u).ln_1p() } else { v.ln() };
    // `ln(u)/(1-u)` is `0/0` at the antipode with the finite limit `-1`:
    // `ln(1-v)/v = -1 - v/2 - v²/3 - …`. This is the factor that carries the
    // antipodal limit of BOTH the m=2 and the m=3 form.
    let ln_u_over_v = if v == 0.0 { -1.0 } else { ln_u / v };
    let dk_du = match m {
        1 => -1.0 / (four_pi * u),
        2 => ln_u_over_v / four_pi,
        3 => {
            let li2_u = dilog_unit(u);
            // `ln(u)·ln(1-u)/u` is `0·(-∞)` at the antipode and vanishes there
            // like `v·ln v`; nothing cancels in it for `v > 0`.
            let cross = if v == 0.0 { 0.0 } else { ln_u * ln_v / u };
            (-li2_u / u - ln_u_over_v - cross) / four_pi
        }
        // SAFETY: the sole caller
        // `wahba_sphere_kernel_sobolev_derivative_dhav` dispatches to this
        // closed form only inside `(1..=3).contains(&m)`, so
        // any other `m` is a caller-contract violation (a programming error,
        // not runtime data), and panicking surfaces it instead of returning a
        // silently-wrong derivative.
        other => {
            panic!("closed-form Sobolev derivative only defined for m in {{1,2,3}}; got m={other}")
        }
    };
    dk_du
}

/// `dK_m^{Sobolev}/du` with respect to the half-angle separation
/// `u = sin²(γ/2)`.
pub(crate) fn wahba_sphere_kernel_sobolev_derivative_dhav(
    sep: HalfAngleSeparation,
    m: usize,
) -> f64 {
    const POLE_LIMIT_THRESHOLD: f64 = 1.0e-10;
    // `u = (1 − cos γ)/2`, so the historical `cos γ ≤ 1 − POLE_LIMIT_THRESHOLD`
    // guard is exactly `u ≥ POLE_LIMIT_THRESHOLD/2`. Stating it in `u` keeps the
    // boundary where it was while letting the caller hand in a `u` that is
    // accurate below it (#2489) instead of one quantized to multiples of `ε/4`.
    const POLE_LIMIT_U: f64 = 0.5 * POLE_LIMIT_THRESHOLD;

    // m in {1,2,3} use the exact polylog closed-form derivative so the jet
    // matches the closed-form forward kernel; m=4 falls back to the spectral
    // series (the forward m=4 kernel is itself spectral). Leave the closed form
    // near the COINCIDENT pole, where `dK/du` carries the genuine `1/u` (m=1)
    // and `ln u` (m=2, m=3) singularities of the Sobolev kernel.
    //
    // The guard is ONE-SIDED, and used not to be. Only `cos γ → +1` is a pole
    // of these kernels; `cos γ → -1` is an ordinary interior point where all
    // three derivatives are finite, smooth, and elementary:
    //
    // ```text
    //   m=1, m=2:  dK/d(cos γ)|_{cos γ = -1} = 1/(8π)          = 3.9788735772973834e-2
    //   m=3:       dK/d(cos γ)|_{cos γ = -1} = (π²/6 - 1)/(8π) = 2.5661111176813525e-2
    // ```
    //
    // Routing the antipode to the spectral branch was not a conservative
    // choice, it was wrong, because term-by-term differentiation of a Legendre
    // series need not converge where the series itself does. At `m = 1` the
    // differentiated terms `(2ℓ+1)(-1)^{ℓ+1}/8π` GROW, so the branch summed a
    // divergent alternating series and returned its `l_max`-th partial sum:
    // `Σ_{ℓ≤L}(-1)^{ℓ+1}(2ℓ+1) = -L` exactly, i.e. `-4096/(8π) = -162.9747`
    // where the answer is `+0.0397887`. Wrong sign, 4096x magnitude, and a pure
    // function of the truncation constant — doubling `l_max` doubles it.
    //
    // The region is not measure-zero either: `|cos γ| > 1 - 1e-10` at the
    // antipodal end is every pair within 1.4e-5 rad (~3 arcsec) of antipodal,
    // and the previous behaviour stepped from `+0.0398` to `-162.97` across
    // that boundary. Antipodal pairs are what farthest-point centre selection
    // actively seeks out, which is the same reason the `Li₃` accuracy near
    // `z = 1` mattered (see `polylog`'s module docs).
    //
    // m=4 keeps both poles: its differentiated terms decay like `ℓ^-4`, so the
    // spectral limit converges there and is the only form available.
    if (1..=3).contains(&m) && sep.u >= POLE_LIMIT_U {
        return wahba_sphere_kernel_sobolev_closed_form_derivative_dhav(sep, m);
    }

    let l_max = match m {
        1 => 4096_usize,
        2 => 256,
        3 => 128,
        _ => 96,
    };
    let x = sep.cos_gamma();
    let m_i = m as i32;
    let four_pi = 4.0 * std::f64::consts::PI;
    // ONE sweep, valid on the closed interval including both poles. There is no
    // pole special-case and no `1 - x²` floor, because the derivative is taken
    // from the recurrence that has no pole:
    //
    // ```text
    //   P'_ℓ(x) = (2ℓ - 1)·P_{ℓ-1}(x) + P'_{ℓ-2}(x),    P'_0 = 0, P'_1 = 1
    // ```
    //
    // The form this replaces, `P'_ℓ = ℓ(P_{ℓ-1} - x·P_ℓ)/(1 - x²)`, is a
    // removable `0/0` at `x = ±1` — which is why it needed a floor AND a
    // separate pole branch — and it is already losing digits well before it
    // gets there: its numerator subtracts two `O(1)` Legendre values to leave
    // `O((ℓ+1)(1-x))`, so the relative error grows like `ε/(1-x²)`. Measured on
    // a truncated Sobolev kernel (`lmax=64`, `m=2`) against a 40-digit
    // reference:
    //
    // ```text
    //   |x|          0.99      0.9999    1-1e-8    1-1e-9    1-1e-10
    //   quotient    7.4e-16    1.5e-13   7.0e-10   3.3e-8    6.5e-8
    //   recurrence  4.3e-16    1.3e-17   1.8e-15   3.0e-15   2.2e-15
    // ```
    //
    // so the old code was handing the pole branch a value that had already
    // decayed to eight digits by the time the threshold caught it. The
    // recurrence needs no catching: at `x = ±1` it reproduces the closed pole
    // formula `P'_ℓ(±1) = (±1)^{ℓ+1}·ℓ(ℓ+1)/2` to 5e-16, so the branch it
    // replaces was computing the same number by a second route.
    let mut p_prev = 1.0_f64; // P_{ℓ-2}, seeded at P_0
    let mut p_curr = x; // P_{ℓ-1}, seeded at P_1
    let mut d_prev = 0.0_f64; // P'_{ℓ-2}, seeded at P'_0
    let mut d_curr = 1.0_f64; // P'_{ℓ-1}, seeded at P'_1
    let mut sum = 3.0 * d_curr / (four_pi * 2.0_f64.powi(m_i));
    for l in 2..=l_max {
        let ell = l as f64;
        let two_l_minus_1 = 2.0 * ell - 1.0;
        let d_next = two_l_minus_1 * p_curr + d_prev;
        let p_next = (two_l_minus_1 * x * p_curr - (ell - 1.0) * p_prev) / ell;
        let eigen = (ell * (ell + 1.0)).powi(m_i);
        let weight = (2.0 * ell + 1.0) / four_pi;
        sum += weight * d_next / eigen;
        p_prev = p_curr;
        p_curr = p_next;
        d_prev = d_curr;
        d_curr = d_next;
    }
    // The spectral sweep produces `dK/d(cos γ)`; `d(cos γ)/du = −2`.
    -2.0 * sum
}

/// Unified `dK/du` for any [`SphereWahbaKernel`] kind, against the half-angle
/// separation `u = sin²(γ/2)`.
///
/// This is the form the design jet wants: paired with
/// [`super::sphere_half_angle::half_angle_partials`] it computes the `|γ|` cusp
/// gradient as a product of two finite factors, where the `cos γ` chain has to
/// recover it from `∞ · 0` (#2489).
#[inline]
pub(crate) fn wahba_sphere_kernel_derivative_dhav_kind(
    sep: HalfAngleSeparation,
    penalty_order: usize,
    kernel: SphereWahbaKernel,
) -> f64 {
    match kernel {
        SphereWahbaKernel::Sobolev => {
            wahba_sphere_kernel_sobolev_derivative_dhav(sep, penalty_order)
        }
        SphereWahbaKernel::Pseudo => wahba_sphere_kernel_pseudo_derivative_dhav(sep, penalty_order),
        SphereWahbaKernel::SobolevTruncated { lmax } => {
            let coeffs = sobolev_s2_truncated_coefficients(lmax as usize, penalty_order);
            -2.0 * sphere_truncated_spectral_derivative_eval(sep.cos_gamma(), &coeffs)
        }
        SphereWahbaKernel::PseudoTruncated { lmax } => {
            let coeffs = pseudo_s2_truncated_coefficients(lmax as usize, penalty_order);
            -2.0 * sphere_truncated_spectral_derivative_eval(sep.cos_gamma(), &coeffs)
        }
    }
}
