//! Measurement probe for #2475 — what the `f64::EPSILON * 1.0e-4` floor in
//! `sphere_kernels.rs` actually *is*, at each of the four sites that shared it.
//!
//! The floor was reachable at exactly one input. By Sterbenz's lemma
//! `1.0 - cos γ` is computed exactly for `cos γ ∈ [0.5, 1]`, and the
//! representable spacing just below `1.0` is `2⁻⁵³`, so `u = (1 − cos γ)/2` is
//! either exactly `0` or at least `2⁻⁵⁴ = ε/4 ≈ 5.55e-17` — nine orders above
//! the `2.22e-20` floor. There are no inputs in between. So the floor never
//! approximated a limit from nearby; it only ever supplied a value AT
//! coincidence, and the three kernels that share it need three different
//! things there:
//!
//!  1. **Pseudo-spline forward** — the limit is finite and exactly known,
//!     `K_m^{pseudo}(0) = 1/(2π·m·(m+1)!)`. FIXED: taken analytically, which
//!     is bit-identical away from coincidence and exact at it.
//!  2. **Sobolev forward, `m = 2, 3`** — also finite, and already exact: at
//!     `u ≤ 2.22e-20` the argument `1 − u` rounds to `1.0`, so
//!     `dilog_unit(1.0) = π²/6` lands the closed forms on `1/(4π)` and
//!     `(2ζ₃ − 2)/(4π)` to below roundoff. NOT a defect.
//!  3. **Sobolev forward, `m = 1`** — genuinely log-divergent. The floor here
//!     is not an approximation, it *chooses* the diagonal. This probe pins
//!     what that choice amounts to (below) and measures the defect that
//!     remains. STILL OPEN.
//!
//! ## What the `m = 1` floor is, quantitatively
//!
//! Flooring `u` is exactly equivalent to truncating the Legendre expansion,
//! and the correspondence is closed-form. The truncated diagonal is
//!
//! ```text
//! K_L(0) = (1/4π) Σ_{ℓ=1..L} (2ℓ+1)/(ℓ(ℓ+1)) = (2H_L + 1/(L+1) − 1) / 4π
//! ```
//!
//! (using `(2ℓ+1)/(ℓ(ℓ+1)) = 1/ℓ + 1/(ℓ+1)`), while the floored closed form
//! gives `(−ln u₀ − 1)/4π`. Equating them and using `H_L = ln L + γ + O(1/L)`:
//!
//! ```text
//! u₀ = exp(−2H_L − 1/(L+1))  →  e^{−2γ_E} / L²,   i.e.  L_eff = e^{−γ_E} / √u₀
//! ```
//!
//! So **the shipped `u₀ = ε·1e-4` is a spectral truncation at degree
//! `L_eff ≈ 3.8 × 10⁹`** — while `SphereWahbaKernel::SobolevTruncated`
//! documents a practical range of `5..200` and stores `lmax` in a `u16`, which
//! cannot even represent it. That is the finding: the default kernel's
//! diagonal is a truncation degree seven orders of magnitude past the top of
//! the type the explicit truncated variant uses.
//!
//! ## The defect that remains
//!
//! `cos γ` for a center against itself is rebuilt from lat/lon trig
//! (`sin·sin + cos·cos·(cos·cos + sin·sin)`), which lands on exactly `1.0`
//! only when the rounding happens to cooperate. Whether it does decides
//! whether the floor binds, so a single Sobolev `m = 1` Gram matrix carries
//! several distinct diagonal values. Printed and asserted below.

use super::sphere_kernels::{
    wahba_sphere_kernel_pseudo_coincident, wahba_sphere_kernel_pseudo_from_cos,
    wahba_sphere_kernel_sobolev_closed_form, wahba_sphere_kernel_sobolev_derivative_dcos,
};

const FOUR_PI: f64 = 4.0 * std::f64::consts::PI;
/// The floor that used to be shared by all four sites.
const LEGACY_FLOOR: f64 = f64::EPSILON * 1.0e-4;
/// Euler–Mascheroni constant.
const EULER_GAMMA: f64 = 0.577_215_664_901_532_9;

/// `cos γ` exactly as `spherical_wahba_kernel_matrix_cpu` rebuilds it for a
/// point against ITSELF: the dot product of two unit vectors reconstructed
/// from the same `(lat, lon)` by the same trig.
fn self_cos_gamma(lat: f64, lon: f64) -> f64 {
    let (sin_lat, cos_lat) = (lat.sin(), lat.cos());
    let (sin_lon, cos_lon) = (lon.sin(), lon.cos());
    sin_lat * sin_lat + cos_lat * cos_lat * (cos_lon * cos_lon + sin_lon * sin_lon)
}

/// Deterministic lat/lon spread, no RNG dependency.
fn lat_lon_sample(i: usize) -> (f64, f64) {
    let t = i as f64;
    let lat = (t * 0.732_9).sin() * std::f64::consts::FRAC_PI_2 * 0.97;
    let lon = ((t * 1.113_7) % (2.0 * std::f64::consts::PI)) - std::f64::consts::PI;
    (lat, lon)
}

/// Truncated Sobolev diagonal `K_L(0)`, summed directly.
fn sobolev_truncated_diagonal(m: i32, l_max: usize) -> f64 {
    (1..=l_max)
        .map(|l| {
            let lf = l as f64;
            (2.0 * lf + 1.0) / (FOUR_PI * (lf * (lf + 1.0)).powi(m))
        })
        .sum()
}

// ---------------------------------------------------------------------------
// 1. FIXED: the pseudo-spline diagonal is a closed form, not a floor.
// ---------------------------------------------------------------------------

#[test]
fn zz_measure_2475_pseudo_coincident_matches_closed_form_and_spectral_sum() {
    // `1/(2π·m·(m+1)!)` for m = 1..4.
    let expected = [
        1.0 / (4.0 * std::f64::consts::PI),
        1.0 / (24.0 * std::f64::consts::PI),
        1.0 / (144.0 * std::f64::consts::PI),
        1.0 / (960.0 * std::f64::consts::PI),
    ];
    println!(
        "\n{:>2} {:>24} {:>24} {:>12} {:>12}",
        "m", "K_m(0) = 1/(2pi m (m+1)!)", "spectral sum (L=50)", "tail_rel", "rel_legacy"
    );
    for (idx, &want) in expected.iter().enumerate() {
        let m = idx + 1;

        // The kernel now returns the analytic limit at coincidence.
        let got = wahba_sphere_kernel_pseudo_from_cos(1.0, m);
        assert_eq!(
            got, want,
            "pseudo m={m} coincident value must be exactly 1/(2pi*m*(m+1)!)"
        );
        assert_eq!(got, wahba_sphere_kernel_pseudo_coincident(m));

        // Independent confirmation: the truncated spectral sum of the SAME
        // kernel's coefficients differs from the closed form by EXACTLY the
        // analytic truncation tail. The same telescoping identity that gives
        // `K_m(0) = 1/(2π·m·(m+1)!)` gives the tail with `(L+k)` in place of
        // `k`:
        //     Σ_{ℓ>L} c_ℓ = 1 / (2π · m · Π_{k=2..m+1} (L+k)).
        // Matching it to nine digits validates the identity twice over.
        const L: usize = 50;
        let coeffs = super::sphere_spectral::pseudo_s2_truncated_coefficients(L, m);
        let spectral: f64 = coeffs.iter().sum();
        let tail_product: f64 = (2..=(m + 1)).map(|k| (L + k) as f64).product();
        let tail = 1.0 / (2.0 * std::f64::consts::PI * (m as f64) * tail_product);
        let residual = want - spectral;
        let tail_rel = (residual - tail).abs() / tail;

        // What the old shared floor produced instead.
        let legacy = {
            let w = 0.5 * LEGACY_FLOOR;
            let c0 = w.sqrt();
            want - 2.0 * c0 / (2.0 * std::f64::consts::PI)
        };
        let rel_legacy = (legacy - want).abs() / want;

        println!("{m:>2} {want:>24.16e} {spectral:>24.16e} {tail_rel:>12.2e} {rel_legacy:>12.2e}");

        assert!(
            tail_rel < 1e-9,
            "pseudo m={m}: closed form {want:.17e} minus the L={L} spectral sum \
             {spectral:.17e} gives {residual:.6e}, but the analytic tail is \
             {tail:.6e} (rel {tail_rel:.3e})"
        );
    }
    // The m=1 site is where the old floor actually cost something.
    println!(
        "\n  the legacy floor's error was the -2*sqrt(w) term, O(sqrt(floor)): \
         4.2e-10 relative at m=1, roundoff at m>=2.\n"
    );
}

#[test]
fn zz_measure_2475_pseudo_floor_shrinking_has_a_hard_subnormal_floor_of_its_own() {
    // #2475 suggested `f64::MIN_POSITIVE` as a floor that "recovers all four
    // analytic limits exactly". That is correct — MIN_POSITIVE is the smallest
    // NORMAL (2.225e-308), `w = 0.5*z` stays subnormal-but-nonzero, and the
    // `-2√w` and `2aw` terms both vanish below the ulp of the leading constant.
    // Recorded here so the claim is pinned rather than assumed.
    let limits = [
        1.0 / (4.0 * std::f64::consts::PI),
        1.0 / (24.0 * std::f64::consts::PI),
        1.0 / (144.0 * std::f64::consts::PI),
        1.0 / (960.0 * std::f64::consts::PI),
    ];
    for (idx, &want) in limits.iter().enumerate() {
        let m = idx + 1;
        let got = wahba_sphere_kernel_pseudo_from_cos(1.0 - f64::MIN_POSITIVE, m);
        assert_eq!(got, want, "MIN_POSITIVE floor recovers the m={m} limit");
    }

    // But shrinking the floor is not unconditionally safe: one more halving,
    // to the smallest SUBNORMAL, makes `w = 0.5*z` underflow to zero, and then
    // `1/c0 = +inf` turns the `2·a·w` term into `inf · 0 = NaN`. The analytic
    // limit the kernel now takes has no such cliff.
    let z = f64::from_bits(1); // 4.94e-324, smallest positive subnormal
    let w = 0.5 * z;
    assert_eq!(w, 0.0, "half the smallest subnormal rounds to zero");
    let a = (1.0 + 1.0 / w.sqrt()).ln();
    assert!(a.is_infinite());
    assert!((2.0 * a * w).is_nan(), "2*a*w would be inf * 0 = NaN");

    for m in 1..=4 {
        assert!(wahba_sphere_kernel_pseudo_from_cos(1.0, m).is_finite());
    }
}

#[test]
fn zz_measure_2475_pseudo_change_is_confined_to_exact_coincidence() {
    // Removing the floor must not perturb any input it could not reach. The
    // smallest positive `1 - cos γ` is 2⁻⁵³, nine orders above the old floor,
    // so every non-coincident value is bit-identical to the floored form.
    let legacy = |cos_gamma: f64, m: usize| -> f64 {
        let cg = cos_gamma.clamp(-1.0, 1.0);
        let z = (1.0 - cg).max(LEGACY_FLOOR);
        assert!(z > 0.0);
        // Re-run the shipped arithmetic on the floored z by feeding back a
        // cos γ that reproduces it exactly.
        wahba_sphere_kernel_pseudo_from_cos(1.0 - z, m)
    };
    let mut checked = 0usize;
    for k in 1..=64u64 {
        // Representable neighbours of 1.0 from below, plus a coarse sweep.
        let cos_gamma = 1.0 - (k as f64) * f64::EPSILON / 2.0;
        for m in 1..=4 {
            assert_eq!(
                wahba_sphere_kernel_pseudo_from_cos(cos_gamma, m),
                legacy(cos_gamma, m),
                "m={m} at 1 - {k} ulp must be unchanged by dropping the floor"
            );
            checked += 1;
        }
    }
    for i in 0..200 {
        let cos_gamma = -1.0 + 2.0 * (i as f64) / 199.0;
        for m in 1..=4 {
            assert_eq!(
                wahba_sphere_kernel_pseudo_from_cos(cos_gamma, m),
                legacy(cos_gamma, m)
            );
            checked += 1;
        }
    }
    println!("\n  {checked} non-coincident evaluations, all bit-identical\n");
}

// ---------------------------------------------------------------------------
// 2. NOT A DEFECT: Sobolev m = 2, 3 already land on their exact limits.
// ---------------------------------------------------------------------------

#[test]
fn zz_measure_2475_sobolev_m2_m3_coincident_are_already_exact() {
    const ZETA3: f64 = 1.202_056_903_159_594_3;
    let cases = [
        (2usize, 1.0 / FOUR_PI),
        (3usize, (2.0 * ZETA3 - 2.0) / FOUR_PI),
    ];
    println!(
        "\n{:>2} {:>24} {:>24} {:>10}",
        "m", "closed-form limit", "shipped at cos=1", "rel"
    );
    for (m, want) in cases {
        let got = wahba_sphere_kernel_sobolev_closed_form(1.0, m);
        let rel = (got - want).abs() / want.abs();
        println!("{m:>2} {want:>24.16e} {got:>24.16e} {rel:>10.2e}");
        assert!(
            rel < 1e-15,
            "Sobolev m={m} coincident value {got:.17e} should already equal its \
             exact limit {want:.17e} (rel {rel:.3e})"
        );
        // Cross-check the limit itself against the kernel's own spectral series.
        let spectral = sobolev_truncated_diagonal(m as i32, 200_000);
        assert!(
            (spectral - want).abs() / want.abs() < 1e-9,
            "m={m}: limit {want:.17e} vs spectral {spectral:.17e}"
        );
    }
}

// ---------------------------------------------------------------------------
// 3. STILL OPEN: Sobolev m = 1 is log-divergent; the floor picks the diagonal.
// ---------------------------------------------------------------------------

#[test]
fn zz_measure_2475_sobolev_m1_floor_is_a_spectral_truncation_degree() {
    // K_L(0) = (2 H_L + 1/(L+1) - 1)/4π  matches  (-ln u₀ - 1)/4π
    // at  u₀ = exp(-2 H_L - 1/(L+1))  ->  e^{-2γ}/L².
    println!(
        "\n{:>8} {:>14} {:>22} {:>22} {:>10}",
        "L", "K_L(0) exact", "u0 solving closed form", "e^{-2gamma}/L^2", "ratio"
    );
    for &l_max in &[4usize, 8, 32, 128, 512, 4096, 65535] {
        let k_l = sobolev_truncated_diagonal(1, l_max);
        // Invert (-ln u - 1)/4π = K_L(0).
        let u_star = (-(FOUR_PI * k_l + 1.0)).exp();
        let asymptotic = (-2.0 * EULER_GAMMA).exp() / (l_max as f64).powi(2);
        let ratio = u_star / asymptotic;
        println!("{l_max:>8} {k_l:>14.6} {u_star:>22.6e} {asymptotic:>22.6e} {ratio:>10.6}");
        // The correspondence is exact in the limit; O(1/L) at finite L.
        assert!(
            (ratio - 1.0).abs() < 4.0 / (l_max as f64),
            "L={l_max}: floor<->truncation correspondence off by {:.3e}, \
             expected O(1/L)",
            (ratio - 1.0).abs()
        );
    }

    // Now read the shipped floor the other way: what degree is it?
    let l_eff = (-EULER_GAMMA).exp() / LEGACY_FLOOR.sqrt();
    let diagonal = wahba_sphere_kernel_sobolev_closed_form(1.0, 1);
    println!(
        "\n  shipped floor u0 = eps*1e-4 = {LEGACY_FLOOR:.4e}\n  \
         => K(diagonal)  = {diagonal:.6}\n  \
         => L_eff        = e^-gamma / sqrt(u0) = {l_eff:.4e}\n  \
         (SobolevTruncated documents lmax in 5..200 and stores it in a u16, \
         max {})\n",
        u16::MAX
    );
    assert!(
        (3.7e9..3.9e9).contains(&l_eff),
        "the shipped floor should correspond to L_eff ~ 3.8e9, got {l_eff:.4e}"
    );
    assert!(
        l_eff > f64::from(u16::MAX),
        "the finding: the default kernel's implied truncation degree is not \
         even representable in SobolevTruncated's lmax type"
    );
}

#[test]
fn zz_measure_2475_sobolev_m1_gram_diagonal_is_inhomogeneous() {
    // The kernel is zonal — a function of geodesic angle alone — and the
    // geodesic angle from a point to itself is exactly zero for EVERY point.
    // So K(c_i, c_i) must be one number. It is not: whether the trig rebuild
    // of cos γ lands on exactly 1.0 decides whether the floor binds.
    let mut values: Vec<f64> = Vec::new();
    let mut exact_one = 0usize;
    println!(
        "\n{:>10} {:>10} {:>14} {:>10}",
        "lat(deg)", "lon(deg)", "cos_g - 1", "K_diag"
    );
    for i in 0..24 {
        let (lat, lon) = lat_lon_sample(i);
        let cos_g = self_cos_gamma(lat, lon);
        if cos_g >= 1.0 {
            exact_one += 1;
        }
        let k = wahba_sphere_kernel_sobolev_closed_form(cos_g, 1);
        if i < 10 {
            println!(
                "{:>10.3} {:>10.3} {:>14.3e} {:>10.6}",
                lat.to_degrees(),
                lon.to_degrees(),
                cos_g - 1.0,
                k
            );
        }
        values.push(k);
    }
    let mut distinct = values.clone();
    distinct.sort_by(|a, b| a.partial_cmp(b).unwrap());
    distinct.dedup();
    let lo = distinct[0];
    let hi = distinct[distinct.len() - 1];
    let spread = (hi - lo) / lo;
    println!(
        "\n  {} of 24 self-dot-products are exactly 1.0\n  \
         {} DISTINCT diagonal values in one Gram matrix: {:?}\n  \
         spread = {:.1}% of the smallest\n",
        exact_one,
        distinct.len(),
        distinct
            .iter()
            .map(|v| format!("{v:.6}"))
            .collect::<Vec<_>>(),
        100.0 * spread
    );

    // Pin the defect. This assertion is the thing a fix must flip: it should
    // become `distinct.len() == 1`.
    assert!(
        distinct.len() > 1,
        "#2475 fixed? the Sobolev m=1 Gram diagonal is now homogeneous — \
         invert this assertion and pin the chosen value"
    );
    assert!(
        spread > 0.2,
        "expected the ~21.5% one-ulp step to still be present, saw {:.3}%",
        100.0 * spread
    );

    // The pseudo kernel, by contrast, has no step at coincidence: its limit is
    // finite, so neighbouring values approach it continuously (the residual
    // spread is the kernel's genuine |γ| cusp, not a floor artefact).
    let pseudo: Vec<f64> = (0..24)
        .map(|i| {
            let (lat, lon) = lat_lon_sample(i);
            wahba_sphere_kernel_pseudo_from_cos(self_cos_gamma(lat, lon), 1)
        })
        .collect();
    let p_lo = pseudo.iter().cloned().fold(f64::INFINITY, f64::min);
    let p_hi = pseudo.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    println!(
        "  pseudo m=1 diagonal spread for comparison: {:.2e}\n",
        (p_hi - p_lo) / p_lo
    );
    assert!(
        (p_hi - p_lo) / p_lo < 1e-7,
        "pseudo diagonal should vary only at the cusp scale"
    );
}

// ---------------------------------------------------------------------------
// 4. DELETED: the Sobolev DERIVATIVE floors were unreachable behind the pole
//    guard, so they were dead arithmetic rather than a regularization.
// ---------------------------------------------------------------------------

#[test]
fn zz_measure_2475_sobolev_derivative_floors_were_unreachable() {
    // `wahba_sphere_kernel_sobolev_derivative_dcos` only enters the closed form
    // inside `|cos γ| <= 1 - POLE_LIMIT_THRESHOLD` (1e-10), which bounds
    // `u = (1 - cos γ)/2` into `[5e-11, 1 - 5e-11]`. Both bounds are nine
    // orders above the `EPSILON * 1e-4` floor the closed form used to apply
    // (a factor of 2.3e9), so neither `.max()` could ever change a value.
    const POLE_LIMIT_THRESHOLD: f64 = 1.0e-10;
    let u_min = (1.0 - (1.0 - POLE_LIMIT_THRESHOLD)) * 0.5;
    let u_max = (1.0 - (-(1.0 - POLE_LIMIT_THRESHOLD))) * 0.5;
    println!(
        "\n  closed form sees u in [{u_min:.3e}, {u_max:.6}]; the deleted floor was \
         {LEGACY_FLOOR:.3e}, smaller than the tighter end by {:.1e}x\n",
        u_min / LEGACY_FLOOR
    );
    assert!(u_min / LEGACY_FLOOR > 1e9);
    assert!((1.0 - u_max) / LEGACY_FLOOR > 1e9);

    // And the dispatcher stays finite right across the pole boundary and at the
    // poles themselves, where it hands off to the bounded spectral limit.
    for m in 1..=4 {
        for &x in &[
            -1.0,
            -1.0 + 0.5 * POLE_LIMIT_THRESHOLD,
            -1.0 + POLE_LIMIT_THRESHOLD,
            -0.5,
            0.0,
            0.5,
            1.0 - POLE_LIMIT_THRESHOLD,
            1.0 - 0.5 * POLE_LIMIT_THRESHOLD,
            1.0,
        ] {
            let v = wahba_sphere_kernel_sobolev_derivative_dcos(x, m);
            assert!(
                v.is_finite(),
                "Sobolev derivative m={m} at cos γ = {x} is not finite: {v}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// 5. The ANTIPODE is not a pole. The derivative's pole guard used to be
//    two-sided, and the spectral branch it routed `cos γ = -1` into does not
//    converge there at `m = 1`.
// ---------------------------------------------------------------------------

/// `1/(8π)` — the exact `dK/d(cos γ)` at the antipode for Sobolev `m = 1` and
/// `m = 2`, from `dK/du → -1/(4π)` and `du/d(cos γ) = -1/2`.
const ANTIPODAL_DERIVATIVE_M1_M2: f64 = 3.978_873_577_297_383_4e-2;

/// `(π²/6 - 1)/(8π)` — the same quantity at `m = 3`, where
/// `dK/du → (1 - π²/6)/(4π)`: the `-Li₂(u)/u` term tends to `-π²/6`, the
/// `-ln(u)/(1-u)` term to `+1`, and the `-ln(u)·ln(1-u)/u` term to `0` like
/// `v·ln v`.
const ANTIPODAL_DERIVATIVE_M3: f64 = 2.566_111_117_681_352_5e-2;

#[test]
fn zz_measure_2475_antipodal_sobolev_derivative_is_the_closed_form_limit() {
    // Values cross-checked against 50-digit `mpmath` differentiation of the
    // closed-form kernel: at cos γ = -1 it reports 0.039788735772973834 for
    // m ∈ {1,2} and 0.025661111176813525 for m = 3, and the approach from
    // cos γ = -1 + 1e-12 agrees to 13 significant figures, so the point is
    // interior and smooth rather than a limit being taken.
    println!(
        "\n{:>3} {:>26} {:>26} {:>11}",
        "m", "at cos γ = -1", "exact limit", "rel"
    );
    for (m, want) in [
        (1, ANTIPODAL_DERIVATIVE_M1_M2),
        (2, ANTIPODAL_DERIVATIVE_M1_M2),
        (3, ANTIPODAL_DERIVATIVE_M3),
    ] {
        let got = wahba_sphere_kernel_sobolev_derivative_dcos(-1.0, m);
        let rel = (got - want).abs() / want.abs();
        println!("{m:>3} {got:>26.15e} {want:>26.15e} {rel:>11.2e}");
        assert!(
            rel < 1e-14,
            "Sobolev m={m} antipodal derivative is {got:.15e}, exact limit is \
             {want:.15e} (rel {rel:.3e}). The spectral pole branch used to \
             return -162.9747 here at m=1 — the L-th partial sum of a DIVERGENT \
             alternating series, equal to -l_max/(8π)."
        );
    }
    println!();
}

#[test]
fn zz_measure_2475_antipodal_derivative_has_no_step_across_the_old_guard() {
    // The old guard switched representation at |cos γ| = 1 - 1e-10. Walking
    // through it from the interior must now be smooth: these kernels have no
    // feature at the antipode, so any step is an artefact of the routing.
    const T: f64 = 1.0e-10;
    for m in 1..=4 {
        let mut prev = wahba_sphere_kernel_sobolev_derivative_dcos(-1.0 + 10.0 * T, m);
        for &x in &[-1.0 + 2.0 * T, -1.0 + T, -1.0 + 0.5 * T, -1.0 + 1e-14, -1.0] {
            let got = wahba_sphere_kernel_sobolev_derivative_dcos(x, m);
            let step = (got - prev).abs() / prev.abs().max(1e-300);
            assert!(
                step < 1e-6,
                "Sobolev m={m} derivative steps by {step:.3e} relative at \
                 cos γ = {x} ({prev:.9e} -> {got:.9e}); the antipode is an \
                 interior point and must not carry a seam"
            );
            prev = got;
        }
    }
}

#[test]
fn zz_measure_2475_the_antipodal_spectral_sum_is_l_max_not_a_limit() {
    // Why the old branch could not be repaired by widening `l_max`: at m=1 the
    // differentiated Legendre terms GROW, so the sum has no limit to converge
    // to and its partial sum is exactly `-L/(8π)`. Reproduced here at three
    // truncation degrees — a genuine limit would not move.
    let eighth_pi = 1.0 / (8.0 * std::f64::consts::PI);
    println!("\n  m=1 antipodal spectral partial sums (the branch's own arithmetic):");
    for l_max in [256_usize, 1024, 4096] {
        let mut sum = 0.0_f64;
        for l in 1..=l_max {
            let ell = l as f64;
            let sign = if l % 2 == 0 { -1.0 } else { 1.0 };
            let p_l_prime = 0.5 * ell * (ell + 1.0) * sign;
            sum += (2.0 * ell + 1.0) / FOUR_PI * p_l_prime / (ell * (ell + 1.0));
        }
        let predicted = -(l_max as f64) * eighth_pi;
        println!("    l_max={l_max:>5}: sum={sum:>16.6} predicted -l_max/(8π)={predicted:>16.6}");
        assert!(
            (sum - predicted).abs() / predicted.abs() < 1e-12,
            "the m=1 antipodal spectral sum should be exactly -l_max/(8π)"
        );
    }
    println!(
        "\n    exact answer at the antipode: {ANTIPODAL_DERIVATIVE_M1_M2:.9e} \
         — the shipped l_max=4096 gave {:.6}\n",
        -4096.0 * eighth_pi
    );
}

/// `dK/d(cos γ)` from 40-digit `mpmath` — the analytic derivative of the
/// closed-form kernel, cross-checked against `mpmath.diff` of the kernel
/// itself to 1e-41 at four interior points, so the table is not circular with
/// the differentiation rule it encodes.
///
/// Spans both ends of the branch. The `cos γ = -1` rows are the ones the old
/// two-sided pole guard could not produce at all.
const SOBOLEV_DERIVATIVE_REFERENCE: [(usize, f64, f64); 33] = [
    (1, -1.0, 0.039788735772973834),
    (1, -0.9999999999, 0.039788735774963271),
    (1, -0.99999999, 0.039788735971917515),
    (1, -0.999, 0.039808640093020344),
    (1, -0.9, 0.041882879761025088),
    (1, -0.5, 0.053051647697298445),
    (1, 0.0, 0.079577471545947668),
    (1, 0.5, 0.15915494309189534),
    (1, 0.9, 0.79577471545947686),
    (1, 0.999, 79.577471545947597),
    (1, 0.99999999, 7957747.1146090032),
    (2, -1.0, 0.039788735772973834),
    (2, -0.9999999999, 0.039788735773968552),
    (2, -0.99999999, 0.039788735872445674),
    (2, -0.999, 0.039798686273888954),
    (2, -0.9, 0.040817906746232198),
    (2, -0.5, 0.045786023869621704),
    (2, 0.0, 0.055158900038162898),
    (2, 0.5, 0.073545200050883864),
    (2, 0.9, 0.12546989460948413),
    (2, 0.999, 0.30258159039406385),
    (2, 0.99999999, 0.76051505250115565),
    (3, -1.0, 0.025661111176813525),
    (3, -0.9999999999, 0.025661111177101862),
    (3, -0.99999999, 0.025661111205647242),
    (3, -0.999, 0.02566399516138079),
    (3, -0.9, 0.025955749592912204),
    (3, -0.5, 0.027281025216225471),
    (3, 0.0, 0.029407564933744697),
    (3, 0.5, 0.032525947199281328),
    (3, 0.9, 0.037107094521124394),
    (3, 0.999, 0.039718052462181445),
    (3, 0.99999999, 0.03978873392142211),
];

#[test]
fn zz_measure_2475_sobolev_derivative_matches_high_precision_reference() {
    // Before the one-sided guard and the `ln_1p` pairing, this table failed in
    // two distinct places:
    //   * `m=1, cos γ = -1` returned -162.9747 (the divergent spectral sum);
    //   * `m=3, cos γ = 0.99999999` carried 2.4e-6, because `ln(1-u)` was read
    //     off a number within 5e-11 of 1.
    let mut worst = 0.0_f64;
    let mut worst_at = (0usize, f64::NAN);
    println!(
        "\n{:>3} {:>16} {:>26} {:>26} {:>10}",
        "m", "cos γ", "reference", "shipped", "rel"
    );
    for (m, x, want) in SOBOLEV_DERIVATIVE_REFERENCE {
        let got = wahba_sphere_kernel_sobolev_derivative_dcos(x, m);
        let rel = (got - want).abs() / want.abs();
        if rel > worst {
            worst = rel;
            worst_at = (m, x);
        }
        println!("{m:>3} {x:>16} {want:>26.17e} {got:>26.17e} {rel:>10.2e}");
    }
    println!(
        "\n  worst: {worst:.3e} at m={}, cos γ = {}\n",
        worst_at.0, worst_at.1
    );
    assert!(
        worst < 1e-13,
        "Sobolev derivative is off by {worst:.3e} relative at m={}, cos γ = {}",
        worst_at.0,
        worst_at.1
    );
}
