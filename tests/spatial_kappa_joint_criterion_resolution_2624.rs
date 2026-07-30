//! #2624 gate: the exact-joint spatial length-scale optimizer must reach a
//! CERTIFIED stationary optimum on the production 1-D hybrid Duchon geometry at
//! a realistic outer-iteration budget.
//!
//! What this pins, and why an end-to-end fit is the right instrument for it.
//!
//! On the #1033 n-free kappa-trial path the design is never realized at the
//! trial psi, so the Gaussian deviance cannot be formed from rows and is
//! recovered from the k-space sufficient statistics as
//! `z^T W z - 2 qb^T b + qb^T G qb`. On this geometry that is a 7.1-order
//! cancellation -- `z^T W z = 3.13466938668704074e2` against a converged
//! `D_p = 4.0e-5` -- and the profiled-Gaussian REML criterion then multiplies
//! the RELATIVE error of `D_p` by `(n - M_p)/2`, because its entire `D_p`
//! dependence is the single term `((n - M_p)/2) ln(2 pi D_p/(n - M_p))`. At
//! `n = 600` that is a 300x amplifier.
//!
//! With the contraction accumulated plainly it carried 88 ulp of its own
//! magnitude, which is 1.25e-7 relative to `D_p` and so 3.7e-5 on the
//! criterion: at a FIXED theta the criterion moved 3.79e-5 across twelve
//! consecutive evaluations while every other criterion atom stayed monotone to
//! 1e-12 -- 1e6 times more than the theta drift over those calls could produce.
//! A backtracking line search cannot tell any trial step from that, so the two
//! multistart seeds that reach the best objectives both died
//! `StepSizeTooSmall after 50 attempts` a factor of 1.6 and 4.3 short of their
//! own stationarity band, the only seed that converged cleanly converged to a
//! far worse local optimum, and the fit was refused by the
//! objective-monotonicity certificate with `initial=-2.569819e3,
//! final=-1.463112e3` -- the joint phase returning a point 1107 WORSE than its
//! own starting point.
//!
//! The symptom is therefore a whole-fit property (which optimum the multistart
//! selects, and whether the joint phase improves on its own baseline), not a
//! property of one call, which is why the gate is an end-to-end fit rather than
//! a unit assertion on the deviance. It is fast because a converged arm exits
//! instead of running five seeds to exhaustion: ~2 s converged against ~10 s
//! refused.
//!
//! The budget is deliberately BELOW the production default
//! (`SpatialLengthScaleOptimizationOptions::default().max_outer_iter == 80`), so
//! this is not a gate a budget increase could satisfy: these arms fail
//! identically at `max_outer_iter` 60 and 200, with bit-identical
//! `theta_checkpoint`s, which is what separates them from the budget-limited
//! arms of #2624 (0/13 certify at 15, 8/13 at 60, 9/13 at 200).

use gam::{
    FitRequest, FitResult, StandardFitRequest,
    basis::{
        CenterStrategy, DuchonBasisSpec, DuchonNullspaceOrder, DuchonOperatorPenaltySpec,
        OneDimensionalBoundary, SpatialIdentifiability,
    },
    estimate::FitOptions,
    smooth::{
        ShapeConstraint, SmoothBasisSpec, SmoothTermSpec, SpatialLengthScaleOptimizationOptions,
        TermCollectionSpec,
    },
    types::{InverseLink, LikelihoodSpec, ResponseFamily, StandardLink},
};
use ndarray::{Array1, Array2};

/// The #2624 reproducer's own geometry: 1-D hybrid Duchon (`length_scale =
/// Some`), Gaussian identity, 12 farthest-point centers, auto-standardized axis.
/// This is the configuration on which the n-free κ lane is complete, i.e. the
/// one that actually takes the sufficient-statistic deviance path under test.
fn spec_1d(length_scale: f64) -> TermCollectionSpec {
    TermCollectionSpec {
        linear_terms: vec![],
        random_effect_terms: vec![],
        smooth_terms: vec![SmoothTermSpec {
            name: "duchon_1d".to_string(),
            basis: SmoothBasisSpec::Duchon {
                feature_cols: vec![0],
                spec: DuchonBasisSpec {
                    radial_reparam: None,
                    center_strategy: CenterStrategy::FarthestPoint { num_centers: 12 },
                    periodic: None,
                    length_scale: Some(length_scale),
                    power: 1.0,
                    nullspace_order: DuchonNullspaceOrder::Linear,
                    identifiability: SpatialIdentifiability::default(),
                    // `None` routes the isotropic κ optimizer, the n-free-arming
                    // case; `Some(_)` would route the per-axis optimizer, which
                    // does not exercise this lane.
                    aniso_log_scales: None,
                    operator_penalties: DuchonOperatorPenaltySpec::all_active(),
                    boundary: OneDimensionalBoundary::Open,
                },
                input_scale: None,
            },
            shape: ShapeConstraint::None,
            joint_null_rotation: None,
        }],
    }
}

fn fit_options() -> FitOptions {
    FitOptions {
        resource_policy: gam_runtime::resource::ResourcePolicy::default_library(),
        latent_cloglog: None,
        mixture_link: None,
        optimize_mixture: false,
        sas_link: None,
        optimize_sas: false,
        compute_inference: false,
        skip_rho_posterior_inference: false,
        max_iter: 30,
        tol: 1e-6,
        nullspace_dims: vec![],
        linear_constraints: None,
        firth_bias_reduction: false,
        adaptive_regularization: None,
        penalty_shrinkage_floor: None,
        rho_prior: Default::default(),
        kronecker_penalty_system: None,
        kronecker_factored: None,
        // The κ-trial phase is the measurand; a resumed foreign terminal
        // certificate from the shared on-disk warm store would decide the
        // outcome instead of this process (#2625).
        persist_warm_start_disk: false,
    }
}

fn run_arm(length_scale: f64) -> Result<(), String> {
    let n = 600usize;
    let mut x = Array2::<f64>::zeros((n, 1));
    let mut y = Array1::<f64>::zeros(n);
    for i in 0..n {
        let t = (i as f64) / (n as f64 - 1.0) * 6.0 - 3.0;
        x[[i, 0]] = t;
        y[i] = t.sin();
    }
    let weights = Array1::ones(n);
    let offset = Array1::zeros(n);
    let kappa_options = SpatialLengthScaleOptimizationOptions {
        enabled: true,
        max_outer_iter: 60,
        rel_tol: 1e-5,
        log_step: std::f64::consts::LN_2,
        min_length_scale: 1e-2,
        max_length_scale: 1e2,
        pilot_subsample_threshold: 0,
    };
    let result = gam::fit_model(FitRequest::Standard(StandardFitRequest {
        data: gam::solver::fit_orchestration::StandardFitData::shared(x),
        y: std::sync::Arc::new(y),
        weights: std::sync::Arc::new(weights),
        offset: std::sync::Arc::new(offset),
        spec: spec_1d(length_scale),
        family: LikelihoodSpec::new(
            ResponseFamily::Gaussian,
            InverseLink::Standard(StandardLink::Identity),
        ),
        options: fit_options(),
        kappa_options,
        wiggle: None,
        coefficient_groups: Vec::new(),
        penalty_block_gamma_priors: Vec::new(),
        latent_coord: None,
        estimate_tweedie_p: false,
    }))
    .map_err(|error| format!("{error:?}").replace('\n', " | "))?;
    match result {
        FitResult::Standard(standard) => {
            let beta_norm = standard
                .fit
                .beta
                .iter()
                .map(|value| value * value)
                .sum::<f64>()
                .sqrt();
            // A refusal is the failure this gate is about, but a "converged"
            // fit that collapsed the smooth to zero would pass a bare Ok check
            // while being just as wrong. The target `y = sin(t)` has unit-order
            // amplitude, so a smooth that tracks it carries a non-trivial
            // coefficient norm.
            if !(beta_norm > 1e-3) {
                return Err(format!(
                    "fit certified but the smooth collapsed: ||beta|| = {beta_norm:.3e}"
                ));
            }
            Ok(())
        }
        _ => Err("expected a Standard fit result from a standard fit request".to_string()),
    }
}

/// `length_scale = 1.0` and `1.2` are two of the four #2624 arms that are NOT
/// budget-limited: they refuse identically at `max_outer_iter` 60 and 200 with
/// bit-identical checkpoints, so their refusal is the criterion-resolution
/// defect and not the fixture's iteration cap.
#[test]
fn exact_joint_spatial_kappa_certifies_on_the_hybrid_duchon_arms_2624() {
    for &length_scale in &[1.0_f64, 1.2] {
        if let Err(reason) = run_arm(length_scale) {
            panic!(
                "exact-joint spatial kappa failed to certify a stationary optimum at \
                 length_scale = {length_scale} with max_outer_iter = 60 (below the \
                 production default of 80), on the fixture whose n-free deviance \
                 recovery #2624 stabilized: {reason}"
            );
        }
    }
}
