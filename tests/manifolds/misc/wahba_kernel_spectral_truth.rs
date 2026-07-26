//! Verify each Wahba closed-form kernel against the spectral ground truth.
//!
//! The Wahba reproducing kernel on S² with smoothness order m is
//!     K_m(p, q) = (1 / 4π) · Σ_{l ≥ 1} (2l + 1) · [l(l + 1)]^(-m) · P_l(cos γ)
//! where γ = arc-distance(p, q) and P_l is the unnormalized Legendre
//! polynomial of degree l. The l-th term scales like l^{1 − 2m}, so the
//! series converges absolutely for m ≥ 2 and is only conditionally
//! convergent (and divergent at γ = 0) for m = 1. Truncating at the L below
//! gives the "truth" against which the closed-form kernel is exact (up to an
//! additive constant the closed form may subtract for normalization).
//!
//! `m = 1` has no untruncated closed form to test: it is refused at every
//! public matrix entry point (#2475) because its Gram diagonal does not exist.
//! Its two facts are carried separately below — the refusal itself, and the
//! exactness of `SobolevTruncated { lmax }` against the same partial sum.
//!
//! For each m ∈ {1, 2, 3, 4} we evaluate at several γ angles and compare
//!   closed_form(γ) − closed_form(π/2)  vs  spectral(γ) − spectral(π/2)
//! The π/2 offset cancels any additive constant difference, isolating
//! the shape-only error of the kernel implementation.

use gam::basis::{SphereWahbaKernel, spherical_wahba_kernel_matrix_with_kind};
use ndarray::array;

fn legendre_p(l: usize, x: f64) -> f64 {
    // Standard 3-term recurrence: P_0=1, P_1=x, (l+1)P_{l+1} = (2l+1)·x·P_l − l·P_{l-1}
    if l == 0 {
        return 1.0;
    }
    if l == 1 {
        return x;
    }
    let mut pkm1 = 1.0_f64;
    let mut pk = x;
    for k in 1..l {
        let pkp1 = ((2 * k + 1) as f64 * x * pk - (k as f64) * pkm1) / ((k + 1) as f64);
        pkm1 = pk;
        pk = pkp1;
    }
    pk
}

fn spectral_kernel(cos_gamma: f64, m: usize, l_max: usize) -> f64 {
    let mut sum = 0.0_f64;
    for l in 1..=l_max {
        let lf = l as f64;
        let eig = (lf * (lf + 1.0)).powi(m as i32);
        let weight = (2.0 * lf + 1.0) / (4.0 * std::f64::consts::PI);
        sum += weight * legendre_p(l, cos_gamma) / eig;
    }
    sum
}

/// Use the public `spherical_wahba_kernel_matrix` to evaluate K(p, q) for a
/// single (p, q) pair. We construct two single-row coordinate arrays.
fn closed_form_kernel_with_kind(cos_gamma: f64, m: usize, kernel: SphereWahbaKernel) -> f64 {
    // Place point A at the north pole (lat=90°, lon=0°), point B at colatitude
    // γ measured from the pole. Then cos(angular_distance) = sin(latB) (the
    // colatitude formula). Pick latB such that sin(latB_rad) = cos_gamma.
    // (radians=true so lat is in radians directly.)
    let lat_b = cos_gamma.asin(); // radians
    let p = array![[std::f64::consts::FRAC_PI_2, 0.0_f64]];
    let q = array![[lat_b, 0.0_f64]];
    let k = spherical_wahba_kernel_matrix_with_kind(p.view(), q.view(), m, true, kernel)
        .expect("kernel evaluation");
    k[(0, 0)]
}

fn closed_form_kernel(cos_gamma: f64, m: usize) -> f64 {
    closed_form_kernel_with_kind(cos_gamma, m, SphereWahbaKernel::Sobolev)
}

/// The probe angles, shared by the untruncated shape comparison and the m=1
/// truncated comparison. γ = 0 is excluded (the m=1 series diverges there and
/// the closed form is +∞) and γ = π is a measure-zero exact match.
const PROBE_GAMMAS: [f64; 7] = [0.3, 0.6, 1.0, std::f64::consts::FRAC_PI_2, 1.8, 2.4, 2.9];

fn run_compare(m: usize) -> f64 {
    assert!(
        m >= 2,
        "m=1 has no untruncated closed form; see the m=1 tests"
    );
    // Probe angles γ in (0, π); skip γ=0 and γ=π (measure zero, exact match).
    let gammas = PROBE_GAMMAS;
    // L_MAX for the spectral reference, chosen so the reference's truncation
    // tail sits below the tolerance each caller asserts:
    //   m=2 → 4_000   (≈ 1e-10)
    //   m=3 → 1_000   (≈ 1e-12)
    //   m≥4 →   200   (≈ 1e-14)
    let l_max = match m {
        2 => 4_000_usize,
        3 => 1_000,
        _ => 200,
    };
    let cos_pi2 = 0.0_f64; // γ = π/2 → cos = 0
    let closed_ref = closed_form_kernel(cos_pi2, m);
    let spectral_ref = spectral_kernel(cos_pi2, m, l_max);
    let mut max_abs = 0.0_f64;
    for &gamma in &gammas {
        let cg = gamma.cos();
        let closed_delta = closed_form_kernel(cg, m) - closed_ref;
        let spectral_delta = spectral_kernel(cg, m, l_max) - spectral_ref;
        let abs = (closed_delta - spectral_delta).abs();
        if abs > max_abs {
            max_abs = abs;
        }
        eprintln!(
            "[wahba-m{m}] γ={gamma:.3} closed_Δ={closed_delta:+.6e} spectral_Δ={spectral_delta:+.6e} \
             abs_err={abs:.3e}",
        );
    }
    max_abs
}

/// The untruncated `m = 1` kernel has no public evaluator, and that is the
/// point: `K_1 = (−ln u − 1)/4π` is log-singular at coincidence, so its Gram
/// diagonal does not exist and every matrix entry point refuses the family
/// (#2475). Equivalently, `H¹(S²)` has no bounded point evaluation — Sobolev
/// embedding on a 2-manifold needs `s > 1` — so `m = 1` is not a reproducing
/// kernel on S² and there is no truncation-free object to compare against.
///
/// What CAN be pinned exactly is the shipped alternative: at a stated
/// resolution `lmax = L`, `SobolevTruncated { L }` must equal
/// `Σ_{ℓ=1..L} (2ℓ+1)[ℓ(ℓ+1)]⁻¹ P_ℓ(cos γ) / 4π` — the same partial sum this
/// file already computes independently through its own Legendre recurrence.
/// Comparing at the SAME truncation removes the reference's truncation tail
/// entirely, which is what made the old untruncated m=1 check a 1e-4 gate: the
/// tolerance measured the reference, not the kernel. Here the only error left
/// is roundoff in two different summation orders of the same L terms.
#[test]
fn wahba_m1_truncated_matches_its_own_spectral_partial_sum() {
    const LMAX: u16 = 200;
    let mut worst_rel = 0.0_f64;
    for gamma in PROBE_GAMMAS {
        let cg = gamma.cos();
        let shipped =
            closed_form_kernel_with_kind(cg, 1, SphereWahbaKernel::SobolevTruncated { lmax: LMAX });
        let reference = spectral_kernel(cg, 1, LMAX as usize);
        let rel = (shipped - reference).abs() / reference.abs();
        eprintln!(
            "[wahba-m1-trunc] γ={gamma:.3} shipped={shipped:+.17e} reference={reference:+.17e} \
             rel={rel:.3e}"
        );
        worst_rel = worst_rel.max(rel);
    }
    // Both sides run the same 3-term Legendre recurrence over L = 200 degrees.
    // The recurrence's forward error grows with the number of steps, so the
    // achievable bound is O(L·ε) rather than O(ε); 8·L·ε leaves the constant
    // room without admitting a term-level disagreement.
    let bound = 8.0 * f64::from(LMAX) * f64::EPSILON;
    assert!(
        worst_rel <= bound,
        "SobolevTruncated {{ lmax = {LMAX} }} disagrees with its own spectral partial sum by \
         {worst_rel:.3e} > {bound:.3e}"
    );
}

/// The untruncated `m = 1` family must be refused rather than silently given a
/// diagonal — the contract this file's m=1 arm used to depend on.
#[test]
fn wahba_m1_untruncated_is_refused_with_its_remedy() {
    let p = array![[std::f64::consts::FRAC_PI_2, 0.0_f64]];
    let q = array![[0.5_f64, 0.0_f64]];
    let error = spherical_wahba_kernel_matrix_with_kind(
        p.view(),
        q.view(),
        1,
        true,
        SphereWahbaKernel::Sobolev,
    )
    .expect_err("untruncated Sobolev m=1 has no Gram diagonal");
    let message = error.to_string();
    assert!(
        message.contains("log-singular") && message.contains("SobolevTruncated"),
        "the refusal must name both the defect and the explicit-resolution remedy; got: {message}"
    );
}

#[test]
fn wahba_m2_closed_matches_spectral_truth() {
    let err = run_compare(2);
    assert!(
        err < 1e-7,
        "Wahba m=2 closed-form disagrees with spectral truth by {err:.3e}"
    );
}

#[test]
fn wahba_m3_closed_matches_spectral_truth() {
    let err = run_compare(3);
    assert!(
        err < 1e-8,
        "Wahba m=3 closed-form disagrees with spectral truth by {err:.3e}"
    );
}

#[test]
fn wahba_m4_closed_matches_spectral_truth() {
    let err = run_compare(4);
    assert!(
        err < 1e-8,
        "Wahba m=4 kernel disagrees with spectral truth by {err:.3e}",
    );
}
