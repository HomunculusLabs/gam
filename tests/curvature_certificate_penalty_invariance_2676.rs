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
//! 1. **The fix.** The fit completes, with inference requested so the
//!    smoothing-correction inverse runs alongside the outer certificate.
//! 2. **Non-vacuity**, taken on the design the fit ACTUALLY REALIZED and
//!    returned — not on a separately-built one, which is a different object on
//!    this route (the joint spatial freeze resolves the length scale, and the
//!    penalty map moves with it; measured: a cold `build_term_collection_design`
//!    of this same spec has no redundancy at all). Without this the fit above
//!    could pass for the boring reason that there was nothing to deflate.

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

// Chosen by sweeping `examples/repro2676_geo_disease_matern` over
// `(centers, n, n_pcs)`: the smallest cell measured that BOTH carries the exact
// redundancy and completes (2.7 s in release). The redundancy is a property of
// the `(centers, n_pcs)` pair rather than of `n` — `(10, 6)` has none,
// `(10, 16)` and `(24, 16)` do — so neither constant is arbitrary and neither
// may be tuned to make a red test green.
const N_ROWS: usize = 1500;
const N_PCS: usize = 16;
const CENTERS: usize = 10;
const SEED: u64 = 20260226;

fn spec() -> TermCollectionSpec {
    TermCollectionSpec {
        linear_terms: vec![],
        random_effect_terms: vec![],
        smooth_terms: vec![SmoothTermSpec {
            frozen_parametric_residualization: None,
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

/// Every pairwise matrix cosine `tr(S_i S_j)/sqrt(tr(S_i²) tr(S_j²))` of a
/// realized penalty bundle, for the failure message. A cosine of exactly 1 is
/// the redundancy signature the pairwise screen already reports; the Gram null
/// space is the general form of it.
fn pair_cosines(
    canonical: &[gam::terms::construction::CanonicalPenalty],
) -> Vec<(usize, usize, f64)> {
    let norms: Vec<f64> = canonical
        .iter()
        .map(|penalty| penalty.local.iter().map(|value| value * value).sum::<f64>())
        .collect();
    let mut out = Vec::new();
    for i in 0..canonical.len() {
        for j in (i + 1)..canonical.len() {
            if canonical[i].col_range != canonical[j].col_range
                || norms[i] == 0.0
                || norms[j] == 0.0
            {
                continue;
            }
            let dot: f64 = canonical[i]
                .local
                .iter()
                .zip(canonical[j].local.iter())
                .map(|(a, b)| a * b)
                .sum();
            out.push((i, j, dot / (norms[i] * norms[j]).sqrt()));
        }
    }
    out
}

fn canonicalize(
    design: &gam::smooth::TermCollectionDesign,
) -> (Vec<gam::terms::construction::CanonicalPenalty>, usize) {
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
    .expect("canonicalization of a realized design must succeed");
    (canonical, p)
}

/// THE ACCEPTANCE: a Matérn joint spatial fit whose penalty map carries an exact
/// redundancy completes, with inference, and the design it realized is then
/// checked to actually carry that redundancy.
///
/// Before the deflation landed, whether this refused was decided by the sign of
/// a rounding residual — it refused on the bench hosts and admitted on others,
/// with nothing about the fit differing between the two. That is why the fit
/// assertion is bare success: the failure mode is a hard `Err`.
#[test]
fn a_redundant_penalty_map_still_fits_and_certifies_2676() {
    let (x, y) = gam::test_support::synthetic::geo_disease_columns(N_ROWS, SEED);
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

    let fitted = match result {
        Ok(FitResult::Standard(fit)) => fit,
        Ok(_) => panic!("the standard request must produce a standard result"),
        Err(error) => panic!(
            "a Matern joint spatial fit whose penalty map carries an exact redundancy must not \
             refuse: {error}"
        ),
    };
    assert!(
        fitted.fit.beta.iter().all(|value: &f64| value.is_finite()),
        "every coefficient of a certified fit must be finite"
    );

    // Non-vacuity, on the design the fit REALIZED.
    let (canonical, p) = canonicalize(&fitted.design);
    let invariance =
        gam::solver::penalty_invariance::PenaltyMapInvariance::from_canonical_penalties(
            &canonical, p,
        )
        .expect("the penalty-map Gram of a realized design must decompose");
    assert_eq!(
        invariance.dimension(),
        1,
        "this fixture exists because the realized Matern joint spatial penalty map has exactly \
         one exact linear redundancy; if that is no longer true the acceptance above proves \
         nothing and this test must be re-derived, not relaxed. k={}, p={p}, ranges={:?}, \
         pair cosines={:?}",
        canonical.len(),
        canonical
            .iter()
            .map(|c| (c.col_range.start, c.col_range.end))
            .collect::<Vec<_>>(),
        pair_cosines(&canonical),
    );

    // And the redundancy is genuinely a NULL of the map, not a near-one: the
    // combination it names assembles to zero.
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
    let residual = assembled
        .iter()
        .copied()
        .map(f64::abs)
        .fold(0.0_f64, f64::max);
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
