//! #2747 diagnostic experiment: is the monotone descent of `V_p(κ)` a
//! CURVATURE preference, or the criterion sliding down the RANGE direction that
//! the fill-invariance rule leaks into `dκ`?
//!
//! The constant-curvature kernel is `exp(−d_κ(x,y)/ℓ)`. It has TWO scalar
//! degrees of freedom in its exponent — the curvature κ and the range ℓ — and
//! they are strongly confounded: to first order `d_κ = d_0·(1 + κ·a(x,y))`, so
//! the MEAN of `a` acts exactly like a rescaling of `ℓ` and only the VARIATION
//! of `a` across pairs is genuine curvature.
//!
//! The shipped smooth exposes only κ as an outer coordinate. `ℓ` is pinned to a
//! heuristic (`ℓ_ref` = median chart spacing, doubled) and then κ-corrected by
//! the fill rule `L(κ)`. So the criterion is evaluated on the ONE-dimensional
//! curve `(κ, L(κ))` through the `(κ, ℓ)` plane, and
//!
//! ```text
//!   dV/dκ |_fill  =  ∂V/∂κ  +  ∂V/∂ℓ · L′(κ)
//! ```
//!
//! The second term is the leak. On the profile curve `ℓ̂(κ)` it is identically
//! zero by the envelope theorem; on the fill curve it is zero only if `ℓ_ref`
//! happens to be the optimal range. This experiment measures both terms.
//!
//! Diagnostic only: it PRINTS, it does not assert a fix.

use gam::basis::{
    CenterStrategy, ConstantCurvatureBasisSpec, ConstantCurvatureIdentifiability,
    build_constant_curvature_basis, constant_curvature_effective_length,
};
use gam::gaussian_reml::gaussian_reml_multi_closed_form;
use gam::utils::splitmix64;
use ndarray::{Array1, Array2};

fn next_unit(state: &mut u64) -> f64 {
    (splitmix64(state) >> 11) as f64 / (1u64 << 53) as f64
}
fn next_gauss(state: &mut u64) -> f64 {
    let u1 = next_unit(state).max(1.0e-12);
    let u2 = next_unit(state);
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

fn spec_at(kappa: f64, centers: usize, length_scale: f64) -> ConstantCurvatureBasisSpec {
    ConstantCurvatureBasisSpec {
        center_strategy: CenterStrategy::FarthestPoint {
            num_centers: centers,
        },
        kappa,
        kappa_fixed: false,
        length_scale,
        double_penalty: false,
        identifiability: ConstantCurvatureIdentifiability::CenterSumToZero,
    }
}

/// The shipped Gaussian REML criterion at an explicit `(κ, ℓ_ref)`, with λ
/// profiled by the production closed-form solver. Returns `(V, edf, rss)`.
fn reml_at(
    feats: &Array2<f64>,
    y: &Array1<f64>,
    kappa: f64,
    centers: usize,
    ell_ref: f64,
) -> Option<(f64, f64, f64)> {
    let spec = spec_at(kappa, centers, ell_ref);
    let basis = build_constant_curvature_basis(feats.view(), &spec).ok()?;
    let xs = basis.design.to_dense();
    let (n, p) = xs.dim();
    let mut design = Array2::<f64>::ones((n, p + 1));
    design.slice_mut(ndarray::s![.., 1..]).assign(&xs);
    let mut penalty = Array2::<f64>::zeros((p + 1, p + 1));
    penalty
        .slice_mut(ndarray::s![1.., 1..])
        .assign(&basis.active_penalties[0].matrix);
    let y2 = y.view().insert_axis(ndarray::Axis(1));
    let fit =
        gaussian_reml_multi_closed_form(design.view(), y2, penalty.view(), None, None).ok()?;
    let rss: f64 = y
        .iter()
        .zip(fit.fitted.column(0).iter())
        .map(|(a, b)| (a - b) * (a - b))
        .sum();
    Some((fit.reml_score, fit.edf, rss))
}

/// The auto `ℓ_ref` the shipped builder would pick for this cloud.
fn auto_ell_ref(feats: &Array2<f64>, centers: usize) -> f64 {
    let spec = spec_at(0.0, centers, 0.0);
    let basis = build_constant_curvature_basis(feats.view(), &spec).expect("basis at kappa=0");
    match &basis.metadata {
        gam::basis::BasisMetadata::ConstantCurvature { length_scale, .. } => *length_scale,
        _ => panic!("constant-curvature metadata"),
    }
}

/// `n` chart points in a radius-`radius` disk, with `y` an exact member of the
/// κ⋆-span built at `truth_ell` (the fixture builds it at the AUTO ℓ_ref; the
/// whole point here is to vary that).
fn dataset_in_span(
    n: usize,
    kappa_star: f64,
    radius: f64,
    noise_sd: f64,
    truth_ell: f64,
    centers: usize,
    seed: u64,
) -> (Array2<f64>, Array1<f64>) {
    let mut st = seed;
    let mut feats = Array2::<f64>::zeros((n, 2));
    let mut noise = Array1::<f64>::zeros(n);
    for i in 0..n {
        let (x1, x2) = loop {
            let a = 2.0 * next_unit(&mut st) - 1.0;
            let b = 2.0 * next_unit(&mut st) - 1.0;
            if a * a + b * b <= 1.0 {
                break (a * radius, b * radius);
            }
        };
        feats[(i, 0)] = x1;
        feats[(i, 1)] = x2;
        noise[i] = next_gauss(&mut st);
    }
    let truth_spec = spec_at(kappa_star, centers, truth_ell);
    let basis = build_constant_curvature_basis(feats.view(), &truth_spec).expect("truth basis");
    let design = basis.design.to_dense();
    let mut y = Array1::<f64>::zeros(n);
    for j in 0..design.ncols() {
        let w = 1.0 / (1.0 + j as f64);
        for i in 0..n {
            y[i] += w * design[(i, j)];
        }
    }
    let mean = y.iter().sum::<f64>() / n as f64;
    let sd = (y.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n as f64).sqrt();
    for i in 0..n {
        y[i] = (y[i] - mean) / sd + noise_sd * noise[i];
    }
    (feats, y)
}

/// THE EXPERIMENT. For each planted `κ⋆` and each planted truth RANGE, sweep κ
/// across the shipped box and report, at every κ:
///
/// * `V_fill`  — the shipped criterion, i.e. `V(κ, ℓ_ref)` with the builder's
///               fill rule mapping `ℓ_ref ↦ L(κ)` internally;
/// * `V_prof`  — `min over a log-grid of ℓ_ref of V(κ, ℓ_ref)`, the range-profiled
///               criterion, and the minimizing `ℓ̂(κ)`.
///
/// If `V_fill` is monotone while `V_prof` has an interior minimum at `κ⋆`, the
/// descent is the range leak and not a curvature preference.
#[test]
fn probe_range_profiled_kappa_criterion_versus_the_fill_slice() {
    gam::init_parallelism();
    let radius = 0.6_f64;
    let centers = 6usize;
    let n = 120usize;
    let seed = 0x5EED_0944_0000_0000_u64;

    for kappa_star in [-1.0_f64, 0.0, 1.0] {
        for truth_ell_mult in [0.5_f64, 1.0, 2.0] {
            // Probe cloud first, to learn the auto ℓ_ref this cloud yields.
            let (feats0, _) = dataset_in_span(n, 0.0, radius, 0.0, 1.0, centers, seed);
            let ell_ref = auto_ell_ref(&feats0, centers);
            let truth_ell = ell_ref * truth_ell_mult;
            let (feats, y) = dataset_in_span(
                n,
                kappa_star,
                radius,
                0.10,
                truth_ell,
                centers,
                seed,
            );
            let mut max_r2 = 0.0_f64;
            for row in feats.outer_iter() {
                max_r2 = max_r2.max(row.dot(&row));
            }
            let cap = 0.5 / max_r2;
            eprintln!(
                "\n=== κ⋆={kappa_star:+.2}  truth_ℓ={truth_ell:.5} ({truth_ell_mult}×ℓ_ref={ell_ref:.5})  \
                 cap=±{cap:.4} ==="
            );
            eprintln!(
                "  kappa      L_fill      V_fill        edf_f   | ell_hat     V_prof        edf_p"
            );
            let mut best_fill = (f64::INFINITY, f64::NAN);
            let mut best_prof = (f64::INFINITY, f64::NAN, f64::NAN);
            for i in 0..=24 {
                let kappa = -cap + 2.0 * cap * (i as f64) / 24.0;
                let Some((v_fill, edf_f, _)) = reml_at(&feats, &y, kappa, centers, 0.0) else {
                    eprintln!("  k={kappa:<9.4} refused");
                    continue;
                };
                let l_fill = constant_curvature_effective_length(
                    feats.view(),
                    feats.view(),
                    ell_ref,
                    kappa,
                )
                .unwrap_or(f64::NAN);
                if v_fill < best_fill.0 {
                    best_fill = (v_fill, kappa);
                }
                let mut inner = (f64::INFINITY, f64::NAN, f64::NAN);
                for j in 0..=40 {
                    let mult = (-1.6_f64 + 3.2 * (j as f64) / 40.0).exp();
                    let ell = ell_ref * mult;
                    if let Some((v, edf, _)) = reml_at(&feats, &y, kappa, centers, ell)
                        && v < inner.0
                    {
                        inner = (v, ell, edf);
                    }
                }
                if inner.0 < best_prof.0 {
                    best_prof = (inner.0, kappa, inner.1);
                }
                eprintln!(
                    "  k={kappa:<9.4} L={l_fill:<11.5} V={v_fill:<13.6} edf={edf_f:<7.3} | \
                     ell={:<11.5} V={:<13.6} edf={:<7.3}",
                    inner.1, inner.0, inner.2
                );
            }
            eprintln!(
                "  -> fill argmin κ={:+.4} (interior={})   |   profiled argmin κ={:+.4} \
                 (interior={}), ℓ̂={:.5}   [truth {kappa_star:+.2}]",
                best_fill.1,
                best_fill.1.abs() < cap * 0.999,
                best_prof.1,
                best_prof.1.abs() < cap * 0.999,
                best_prof.2
            );
        }
    }
}
