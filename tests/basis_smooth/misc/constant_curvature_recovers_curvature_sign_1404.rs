//! #1404 / #1464 regression guard: the constant-curvature (`curv`) smooth's
//! fitted curvature estimand must IDENTIFY THE SIGN of the true curvature — a
//! negative κ for hyperbolic-shaped data and a positive κ for spherical-shaped
//! data, instead of railing to the positive chart bound for every dataset.
//!
//! This drives the production free-curvature fit, whose continuous
//! response-minus-reference fair profile selects κ before the baseline fit. It
//! therefore guards the user-visible estimand without substituting either the
//! deleted plain-RSS profiler or the raw fixed-fit diagnostic.
//!
//! Reference-as-truth: the response is generated from gam's own
//! constant-curvature kernel and every assertion is on gam's fitted κ.

use gam::estimate::FitOptions;
use gam::smooth::{
    ShapeConstraint, SmoothBasisSpec, SmoothTermSpec, SpatialLengthScaleOptimizationOptions,
    TermCollectionSpec, fit_term_collectionwith_spatial_length_scale_optimization,
    get_constant_curvature_kappa,
};
use gam::terms::basis::{
    CenterStrategy, ConstantCurvatureBasisSpec, ConstantCurvatureIdentifiability,
    constant_curvature_kernel_matrix, realized_constant_curvature_length_scale,
};
use gam::types::LikelihoodSpec;
use ndarray::{Array1, Array2};

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}
fn next_unit(state: &mut u64) -> f64 {
    (splitmix64(state) >> 11) as f64 / (1u64 << 53) as f64
}
fn next_gauss(state: &mut u64) -> f64 {
    let u1 = next_unit(state).max(1.0e-12);
    let u2 = next_unit(state);
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

/// Reproducible data on a disk of radius 0.45.
fn disk_points(n: usize, seed: u64) -> Array2<f64> {
    let mut st = seed;
    let mut pts = Array2::<f64>::zeros((n, 2));
    let mut filled = 0usize;
    while filled < n {
        let a = 2.0 * next_unit(&mut st) - 1.0;
        let b = 2.0 * next_unit(&mut st) - 1.0;
        if a * a + b * b > 1.0 {
            continue;
        }
        pts[[filled, 0]] = a * 0.45;
        pts[[filled, 1]] = b * 0.45;
        filled += 1;
    }
    pts
}

/// A curvature-shaped response: a smooth radial signal whose kernel "shape" is
/// generated at the TRUE κ (via the geodesic-exponential kernel), plus noise.
/// The shape — not the amplitude — carries the curvature sign.
fn curved_response(data: &Array2<f64>, kappa_true: f64, ell: f64, seed: u64) -> Array1<f64> {
    // A single radial bump centered at the origin under the true geometry.
    let center = Array2::from_shape_vec((1, 2), vec![0.0, 0.0]).unwrap();
    let k = constant_curvature_kernel_matrix(data.view(), center.view(), kappa_true, ell).unwrap();
    let mut st = seed ^ 0xD1B5_4A32;
    let mut y = Array1::<f64>::zeros(data.nrows());
    for i in 0..data.nrows() {
        y[i] = 2.0 * k[[i, 0]] - 1.0 + 0.02 * next_gauss(&mut st);
    }
    y
}

/// Fit the free-curvature production model for a given true curvature.
fn fitted_kappa(data: &Array2<f64>, ell_ref: f64, kappa_true: f64) -> f64 {
    let y = curved_response(data, kappa_true, ell_ref, 11);
    let resolved_spec = TermCollectionSpec {
        linear_terms: Vec::new(),
        random_effect_terms: Vec::new(),
        smooth_terms: vec![SmoothTermSpec {
            name: "curvature".to_string(),
            basis: SmoothBasisSpec::ConstantCurvature {
                feature_cols: vec![0, 1],
                spec: ConstantCurvatureBasisSpec {
                    // A modest farthest-point center set keeps each analytic
                    // fair-profile evaluation cheap while resolving the signal.
                    center_strategy: CenterStrategy::FarthestPoint { num_centers: 12 },
                    kappa: 0.0,
                    kappa_fixed: false,
                    length_scale: ell_ref,
                    double_penalty: false,
                    identifiability: ConstantCurvatureIdentifiability::CenterSumToZero,
                },
            },
            shape: ShapeConstraint::None,
            joint_null_rotation: None,
        }],
    };
    let weights = Array1::<f64>::ones(data.nrows());
    let offset = Array1::<f64>::zeros(data.nrows());
    let options = FitOptions::default();
    let fitted = fit_term_collectionwith_spatial_length_scale_optimization(
        data.view(),
        y,
        weights,
        offset,
        &resolved_spec,
        LikelihoodSpec::gaussian_identity(),
        &options,
        &SpatialLengthScaleOptimizationOptions {
            pilot_subsample_threshold: 0,
            ..SpatialLengthScaleOptimizationOptions::default()
        },
    )
    .expect("free-curvature production fit");
    get_constant_curvature_kappa(&fitted.resolvedspec, 0)
        .expect("fitted constant-curvature term must retain kappa")
}

#[test]
fn curv_production_estimand_identifies_curvature_sign_both_ways() {
    let data = disk_points(220, 0xC0FF_EE12);
    // κ=0 reference length (auto chart spacing) — the L(κ) target is pinned to it.
    let ell_ref = realized_constant_curvature_length_scale(data.view(), 0.0).unwrap();

    let k_hyp = fitted_kappa(&data, ell_ref, -2.0);
    let k_sph = fitted_kappa(&data, ell_ref, 2.0);
    eprintln!("[#1404] curvature-sign recovery: hyperbolic κ̂={k_hyp:.2}  spherical κ̂={k_sph:.2}");

    assert!(
        k_hyp < 0.0,
        "hyperbolic truth (κ⋆=−2) must recover NEGATIVE curvature; got κ̂={k_hyp} \
         (the #1464 bug rails this to the +chart bound)"
    );
    assert!(
        k_sph > 0.0,
        "spherical truth (κ⋆=+2) must recover POSITIVE curvature; got κ̂={k_sph}"
    );
    // The two signs must be genuinely DISTINGUISHED, not a coincidence of one bound.
    assert!(
        k_hyp < k_sph,
        "curvature estimand must separate hyperbolic from spherical truth: \
         hyperbolic κ̂={k_hyp} should be below spherical κ̂={k_sph}"
    );
}
