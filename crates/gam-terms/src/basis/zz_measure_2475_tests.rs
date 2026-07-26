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
    wahba_sphere_kernel_sobolev_closed_form,
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
