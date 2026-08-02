//! #2496 routing regression: the public fixed-κ diagnostic must be exactly a
//! pinned complete production fit, not a basis-local Gaussian approximation.
//!
//! The fixture deliberately includes an independent spline smooth beside the
//! constant-curvature term. A basis-local shortcut necessarily omits that
//! smooth's likelihood contribution and smoothing-parameter coordinate. The
//! penalized-scale algebra itself is covered beside the canonical Gaussian REML
//! solver; this test covers only the production routing contract.

use gam::basis::{
    BSplineBasisSpec, BSplineBoundaryConditions, BSplineIdentifiability, BSplineKnotSpec,
    CenterStrategy, ConstantCurvatureBasisSpec, ConstantCurvatureIdentifiability,
    OneDimensionalBoundary,
};
use gam::estimate::FitOptions;
use gam::smooth::{
    ShapeConstraint, SmoothBasisSpec, SmoothTermSpec, SpatialLengthScaleOptimizationOptions,
    TermCollectionSpec, fit_term_collectionwith_spatial_length_scale_optimization,
    fixed_kappa_profiled_reml_score,
};
use gam::types::LikelihoodSpec;
use ndarray::{Array1, Array2};

fn deterministic_fixture() -> (Array2<f64>, Array1<f64>) {
    let n = 72usize;
    let mut data = Array2::<f64>::zeros((n, 3));
    let mut y = Array1::<f64>::zeros(n);
    for i in 0..n {
        let angle = std::f64::consts::TAU * (i as f64 + 0.5) / n as f64;
        let radius = 0.10 + 0.44 * ((i * 29 % 71) as f64 / 70.0);
        let x1 = radius * angle.cos();
        let x2 = radius * angle.sin();
        let z = (i * 37 % 73) as f64 / 72.0;
        data[(i, 0)] = x1;
        data[(i, 1)] = x2;
        data[(i, 2)] = z;
        y[i] = 1.4 * (3.2 * x1).sin()
            + 0.9 * (4.1 * x2).cos()
            + 0.7 * x1 * x2
            + 0.8 * (std::f64::consts::TAU * z).sin()
            + 0.04 * (0.73 * i as f64).sin();
    }
    (data, y)
}

fn collection_spec() -> TermCollectionSpec {
    TermCollectionSpec {
        linear_terms: Vec::new(),
        random_effect_terms: Vec::new(),
        smooth_terms: vec![
            SmoothTermSpec {
                name: "curvature".to_string(),
                basis: SmoothBasisSpec::ConstantCurvature {
                    feature_cols: vec![0, 1],
                    spec: ConstantCurvatureBasisSpec {
                        center_strategy: CenterStrategy::FarthestPoint { num_centers: 14 },
                        kappa: 0.0,
                        kappa_fixed: false,
                        length_scale: 0.55,
                        length_scale_fixed: true,
                        double_penalty: false,
                        identifiability: ConstantCurvatureIdentifiability::CenterSumToZero,
                    },
                },
                shape: ShapeConstraint::None,
                joint_null_rotation: None,
            },
            SmoothTermSpec {
                name: "independent_spline".to_string(),
                basis: SmoothBasisSpec::BSpline1D {
                    feature_col: 2,
                    spec: BSplineBasisSpec {
                        degree: 3,
                        penalty_order: 2,
                        knotspec: BSplineKnotSpec::Generate {
                            data_range: (0.0, 1.0),
                            num_internal_knots: 7,
                        },
                        double_penalty: false,
                        identifiability: BSplineIdentifiability::WeightedSumToZero {
                            weights: None,
                        },
                        boundary_conditions: BSplineBoundaryConditions::default(),
                        boundary: OneDimensionalBoundary::Open,
                    },
                },
                shape: ShapeConstraint::None,
                joint_null_rotation: None,
            },
        ],
    }
}

#[test]
fn fixed_kappa_score_equals_independently_pinned_full_production_fit() {
    let (data, y) = deterministic_fixture();
    let weights = Array1::<f64>::ones(y.len());
    let offset = Array1::<f64>::zeros(y.len());
    let options = FitOptions::default();
    let kappa = 0.65;
    let spec = collection_spec();

    let routed = fixed_kappa_profiled_reml_score(
        data.view(),
        y.view(),
        weights.view(),
        offset.view(),
        &spec,
        0,
        kappa,
        LikelihoodSpec::gaussian_identity(),
        &options,
    )
    .expect("public fixed-kappa diagnostic");

    let mut pinned_spec = spec.clone();
    let SmoothBasisSpec::ConstantCurvature {
        spec: pinned_basis, ..
    } = &mut pinned_spec.smooth_terms[0].basis
    else {
        unreachable!("fixture starts with one constant-curvature term");
    };
    pinned_basis.kappa = kappa;
    let direct = fit_term_collectionwith_spatial_length_scale_optimization(
        data.view(),
        y,
        weights,
        offset,
        &pinned_spec,
        LikelihoodSpec::gaussian_identity(),
        &options,
        &SpatialLengthScaleOptimizationOptions {
            enabled: false,
            ..SpatialLengthScaleOptimizationOptions::default()
        },
    )
    .expect("independently pinned complete production fit");
    let direct_score = direct.fit.reml_score();

    assert!(
        direct.design.penalties.len() >= 2,
        "the fixture must retain independent constant-curvature and spline penalty coordinates"
    );
    assert_eq!(
        direct.fit.lambdas.len(),
        direct.design.penalties.len(),
        "the complete production fit must optimize the entire materialized penalty chart"
    );
    assert!(
        direct.fit.stable_penalty_term > 1.0e-6,
        "fixture must exercise a material fitted penalty contribution, got {}",
        direct.fit.stable_penalty_term,
    );
    let direct_score = direct_score
        .filter(|score| score.is_finite())
        .expect("independently pinned production fit must expose a canonical REML/LAML score");
    assert!(
        (routed - direct_score).abs() <= 1.0e-11 * (1.0 + direct_score.abs()),
        "fixed-kappa diagnostic {routed:.16e} diverged from independently pinned \
         production fit {direct_score:.16e}",
    );
}
