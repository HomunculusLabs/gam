//! #2676 end-to-end: a Matérn joint spatial smooth whose penalty map carries an
//! EXACT linear redundancy must still fit and still certify.
//!
//! This is the shape of the three `geo_disease_*_matern` bench refusals, at a
//! size that runs in a test. The refusals were:
//!
//! * `INDEFINITE CURVATURE AT INTERIOR OPTIMUM (curvature floor did not clear)`
//!   with `interior lambda_min = -5.048e-6` and `railed = []`, and
//! * `rho Hessian has negative curvature -1.452e-6 below the outer certificate's
//!   own resolution floor 1.444e-6 … a genuine contradiction rather than an
//!   unresolvable direction` — decided by a **0.55%** margin.
//!
//! Both were the sign of a rounding residual on a direction where the
//! ρ-curvature is `Σ_k g_k t_k²` by the chain rule, and nothing else. See
//! `gam_solve::penalty_invariance` for the derivation.
//!
//! The two assertions here are a pair and neither is meaningful alone:
//!
//! 1. **Non-vacuity.** The realized penalty map of this design really does have
//!    a one-dimensional invariance. Without this the fit below could pass for
//!    the boring reason that there was nothing to deflate.
//! 2. **The fix.** The fit completes, with inference requested so the
//!    smoothing-correction inverse runs too.

use gam::estimate::FitOptions;
use gam::smooth::{
    ShapeConstraint, SmoothBasisSpec, SmoothTermSpec, SpatialLengthScaleOptimizationOptions,
    TermCollectionSpec,
};
use gam::terms::basis::{
    CenterStrategy, MaternBasisSpec, MaternIdentifiability, MaternLengthScale, MaternNu,
};
use gam::types::{InverseLink, LikelihoodSpec, ResponseFamily, StandardLink};
use gam::{FitRequest, FitResult, StandardFitRequest};
use ndarray::Array1;

const N_ROWS: usize = 900;
const N_PCS: usize = 6;
const CENTERS: usize = 10;
const SEED: u64 = 20260226;

fn spec() -> TermCollectionSpec {
    TermCollectionSpec {
        linear_terms: vec![],
        random_effect_terms: vec![],
        smooth_terms: vec![SmoothTermSpec {
            name: "geo".to_string(),
            basis: SmoothBasisSpec::Matern {
                feature_cols: (0..N_PCS).collect(),
                spec: MaternBasisSpec {
                    center_strategy: CenterStrategy::EqualMassCovarRepresentative {
                        num_centers: CENTERS,
                    },
                    periodic: None,
                    // Omitted `length_scale=` — the same `Auto` the formula
                    // surface produces, which is what routes the fit through
                    // the joint spatial κ search the #2676 refusals live in.
                    length_scale: MaternLengthScale::auto(),
                    nu: MaternNu::FiveHalves,
                    include_intercept: false,
                    double_penalty: false,
                    identifiability: MaternIdentifiability::default(),
                    aniso_log_scales: None,
                },
                input_scale: None,
            },
            shape: ShapeConstraint::None,
            joint_null_rotation: None,
        }],
    }
}

/// The design's realized penalty map carries an exact one-dimensional
/// invariance — i.e. this fixture is the #2676 configuration and not a fit that
/// happens to work.
///
/// The invariance is computed from the penalty map ALONE (the null space of the
/// Gram of the augmented operators), with no tolerance chosen: the rank
/// boundary is the eigensolver's own Weyl backward error.
#[test]
fn the_matern_joint_spatial_penalty_map_really_is_redundant_2676() {
    let (x, _) = gam::test_support::synthetic::geo_disease_columns(N_ROWS, SEED);
    let x = x.slice(ndarray::s![.., ..N_PCS]).to_owned();
    let design = gam::smooth::build_term_collection_design(x.view(), &spec())
        .expect("the Matern joint spatial design must build");

    let specs: Vec<gam::terms::PenaltySpec> = design
        .penalties
        .iter()
        .map(|penalty| gam::terms::PenaltySpec::Block {
            local: penalty.local.clone(),
            col_range: penalty.col_range.clone(),
            prior_mean: penalty.prior_mean.clone(),
            structure_hint: penalty.structure_hint.clone(),
            op: penalty.op.clone(),
        })
        .collect();
    let p = design.design.ncols();
    let (canonical, _) = gam::terms::construction::canonicalize_penalty_specs(
        &specs,
        &vec![0usize; specs.len()],
        p,
        "#2676 acceptance",
    )
    .expect("canonicalization of a built design must succeed");

    let invariance =
        gam::solver::penalty_invariance::PenaltyMapInvariance::from_canonical_penalties(
            &canonical, p,
        )
        .expect("the penalty-map Gram must decompose");
    assert_eq!(
        invariance.dimension(),
        1,
        "this fixture exists because the Matern joint spatial penalty map has exactly one exact \
         linear redundancy (k={}, ranges={:?}); if that is no longer true the acceptance below \
         proves nothing and this test must be re-derived, not relaxed",
        canonical.len(),
        canonical
            .iter()
            .map(|c| (c.col_range.start, c.col_range.end))
            .collect::<Vec<_>>(),
    );

    // And the redundancy is genuinely a NULL of the map, not a near-one: the
    // combination it names assembles to zero to the Gram's own resolution.
    let w = invariance.lambda_basis().column(0).to_owned();
    let mut assembled = ndarray::Array2::<f64>::zeros((p, p));
    for (index, penalty) in canonical.iter().enumerate() {
        let start = penalty.col_range.start;
        for row in 0..penalty.local.nrows() {
            for col in 0..penalty.local.ncols() {
                assembled[[start + row, start + col]] += w[index] * penalty.local[[row, col]];
            }
        }
    }
    let residual = assembled.iter().copied().map(f64::abs).fold(0.0_f64, f64::max);
    let scale = canonical
        .iter()
        .flat_map(|c| c.local.iter().copied().map(f64::abs))
        .fold(0.0_f64, f64::max);
    assert!(
        residual <= 1.0e-10 * scale.max(1.0),
        "sum_i w_i S_i must vanish (got max|entry| = {residual:.3e} against a penalty scale of \
         {scale:.3e})"
    );
}

/// THE ACCEPTANCE: the same design fits, with inference, without the curvature
/// certificate refusing on a direction it has no information about.
///
/// Before the deflation landed, whether this refused was decided by the sign of
/// a rounding residual — it refused on the bench hosts and admitted on others,
/// with nothing about the fit differing between the two. That is why the
/// assertion here is bare success: the failure mode is a hard `Err`, and a
/// verdict that used to be a coin flip is now a property of the penalty map.
#[test]
fn a_redundant_penalty_map_still_fits_and_certifies_2676() {
    let (x_full, y) = gam::test_support::synthetic::geo_disease_columns(N_ROWS, SEED);
    let x = x_full.slice(ndarray::s![.., ..N_PCS]).to_owned();
    let n = y.len();

    let result = gam::fit_model(FitRequest::Standard(StandardFitRequest {
        data: gam::solver::fit_orchestration::StandardFitData::shared(x),
        y: std::sync::Arc::new(y),
        weights: std::sync::Arc::new(Array1::ones(n)),
        offset: std::sync::Arc::new(Array1::zeros(n)),
        spec: spec(),
        family: LikelihoodSpec::new(
            ResponseFamily::Binomial,
            InverseLink::Standard(StandardLink::Logit),
        ),
        estimate_tweedie_p: false,
        options: FitOptions {
            // Inference on, so the smoothing-correction inverse
            // (`invert_identified_rho_hessian`, this issue's path 2) runs
            // alongside the outer certificate (path 1).
            compute_inference: true,
            ..FitOptions::default()
        },
        kappa_options: SpatialLengthScaleOptimizationOptions::default(),
        wiggle: None,
        coefficient_groups: Vec::new(),
        penalty_block_gamma_priors: Vec::new(),
        latent_coord: None,
    }));

    match result {
        Ok(FitResult::Standard(fit)) => {
            assert!(
                fit.fit.beta.iter().all(|value: &f64| value.is_finite()),
                "every coefficient of a certified fit must be finite"
            );
        }
        Ok(_) => panic!("the standard request must produce a standard result"),
        Err(error) => panic!(
            "a Matern joint spatial fit whose penalty map carries an exact redundancy must not \
             refuse: {error}"
        ),
    }
}
