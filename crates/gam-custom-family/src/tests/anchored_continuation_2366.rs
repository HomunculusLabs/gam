//! #2366: the inner mode must be a FUNCTION of ρ, not a functional of the seed.
//!
//! The profiled outer criterion is `V(ρ) = ℓ_p(θ̂(ρ), ρ)`. When `ℓ_p(·, ρ)` is
//! nonconvex its `argmin` is a set, so `V` is a function of ρ only once a
//! selection rule fixes which element is meant. `anchored_continuation_seed`
//! supplies that rule: `θ̂(ρ)` is the endpoint of the continuation from the
//! effective-df-floor anchor, where the term sits on its penalty nullspace and
//! the mode is unique.
//!
//! These tests are built on a fixture whose two modes are known in closed form,
//! and — critically — they include a CONTROL that proves the fixture actually
//! discriminates. A seed-invariance assertion on a unimodal problem would pass
//! no matter what the code did.
use super::*;
use crate::fit::{
    AnchoredContinuationRefusal, ContinuationRefinement, anchored_continuation_seed,
    continuation_refinement_decision,
};
use crate::penalty_labels::penalty_label_layout_with_joint;

/// A one-coefficient family with two known, unequal modes.
///
/// The log-likelihood is `−w(β)` for the tilted double well
///
/// ```text
///     w(β) = (β² − 1)² + c·β,      c = TILT > 0
/// ```
///
/// which has a deep well near `β = −1` (value `≈ −c`) and a shallow well near
/// `β = +1` (value `≈ +c`), separated by a barrier at `β = 0`. The observed
/// information `−d²ℓ/dβ² = 12β² − 4` is genuinely indefinite between the wells,
/// so this is a real nonconvex inner problem rather than a convex one dressed up
/// as one, and `exact_newton_joint_hessian_beta_dependent` is honestly `true`.
///
/// Under the ridge penalty `½λβ²` the barrier is convexified for `λ > 4`, and
/// the unique mode there is `β ≈ −c/(λ−4) < 0`: the maximally-smoothed anchor
/// sits on the DEEP well's side. That is the whole point of anchoring — the
/// continuation from maximal smoothing tracks the deep well, while a caller's
/// coefficients on the other side of the barrier fall into the shallow one.
#[derive(Clone)]
struct TiltedDoubleWellFamily {
    tilt: f64,
}

impl TiltedDoubleWellFamily {
    fn beta(block_states: &[ParameterBlockState]) -> Result<f64, String> {
        block_states
            .first()
            .ok_or_else(|| "missing block 0".to_string())?
            .beta
            .first()
            .copied()
            .ok_or_else(|| "missing coefficient".to_string())
    }
}

impl CustomFamily for TiltedDoubleWellFamily {
    fn evaluate(&self, block_states: &[ParameterBlockState]) -> Result<FamilyEvaluation, String> {
        let beta = Self::beta(block_states)?;
        let well = beta * beta - 1.0;
        Ok(FamilyEvaluation {
            log_likelihood: -(well * well + self.tilt * beta),
            blockworking_sets: vec![BlockWorkingSet::ExactNewton {
                // dℓ/dβ, and the observed information −d²ℓ/dβ².
                gradient: array![-(4.0 * beta * beta * beta - 4.0 * beta + self.tilt)],
                hessian: SymmetricMatrix::Dense(array![[12.0 * beta * beta - 4.0]]),
            }],
        })
    }

    fn exact_newton_joint_hessian_beta_dependent(&self) -> bool {
        true
    }

    fn exact_newton_joint_hessian(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<Option<Array2<f64>>, String> {
        let beta = Self::beta(block_states)?;
        Ok(Some(array![[12.0 * beta * beta - 4.0]]))
    }

    fn exact_newton_joint_hessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        direction: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        let beta = Self::beta(block_states)?;
        let step = direction.first().copied().unwrap_or(0.0);
        Ok(Some(array![[24.0 * beta * step]]))
    }
}

const TILT: f64 = 0.3;

/// #2661 witness: accepting any strict decrease makes the refinement loop
/// operationally unbounded. A 0.999 contraction passes that old predicate on
/// every round, while each round doubles the number of full corrector solves.
#[test]
fn arbitrarily_slow_progress_is_not_a_continuation_contraction_certificate_2661() {
    let refusal = continuation_refinement_decision(4, Some(1.0), 0.999, 1e-12)
        .expect_err("a 0.999 discrepancy ratio must terminate with a typed refusal");
    match refusal {
        AnchoredContinuationRefusal::ContractionPremiseViolated {
            steps,
            previous_discrepancy,
            discrepancy,
            observed_factor,
            required_max_factor,
        } => {
            assert_eq!(steps, 4);
            assert_eq!(previous_discrepancy.to_bits(), 1.0_f64.to_bits());
            assert_eq!(discrepancy.to_bits(), 0.999_f64.to_bits());
            assert_eq!(observed_factor.to_bits(), 0.999_f64.to_bits());
            assert_eq!(required_max_factor.to_bits(), 0.5_f64.to_bits());
        }
        other => panic!("slow contraction produced the wrong typed refusal: {other:?}"),
    }
}

#[test]
fn exact_half_contraction_remains_admissible_2661() {
    let decision = continuation_refinement_decision(4, Some(1.0), 0.5, 1e-12)
        .expect("the documented half-contraction boundary must remain admissible");
    assert_eq!(decision, ContinuationRefinement::Refine);
}

fn double_well_spec(initial_beta: f64) -> ParameterBlockSpec {
    ParameterBlockSpec {
        name: "well".to_string(),
        design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(array![[1.0]])),
        offset: array![0.0],
        penalties: vec![PenaltyMatrix::Dense(array![[1.0]])],
        nullspace_dims: vec![0],
        initial_log_lambdas: array![0.0],
        initial_beta: Some(array![initial_beta]),
        gauge_priority: 100,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    }
}

fn double_well_options() -> BlockwiseFitOptions {
    BlockwiseFitOptions {
        inner_max_cycles: 200,
        inner_tol: 1e-10,
        outer_max_iter: 50,
        outer_tol: 1e-8,
        outer_rel_cost_tol: None,
        rho_lower_bound: -10.0,
        ridge_floor: 1e-8,
        ridge_policy: RidgePolicy::positive_part_approximate_objective(),
        use_remlobjective: true,
        compute_covariance: false,
        use_outer_hessian: false,
        screening_max_inner_iterations: None,
        outer_inner_max_iterations: None,
        seed_screening: false,
        early_exit_threshold: None,
        outer_score_subsample: None,
        auto_outer_subsample: false,
        outer_eval_context: None,
        cache_session: None,
        persistent_warm_start_store: None,
        cache_mirror_sessions: Vec::new(),
        joint_penalties: None,
        independent_prior_factor_labels: Vec::new(),
        screen_initial_rho: false,
    }
}

/// Solve the inner problem at `rho` exactly the way the outer search does, from
/// the seed carried in `specs` — i.e. the pre-#2366 "whatever the caller handed
/// us" mode.
fn cold_direct_mode(
    family: &TiltedDoubleWellFamily,
    specs: &[ParameterBlockSpec],
    rho: f64,
) -> f64 {
    let options = double_well_options();
    let penalty_counts: Vec<usize> = specs.iter().map(|spec| spec.penalties.len()).collect();
    let layout = penalty_label_layout_with_joint(specs, penalty_counts, Vec::new())
        .expect("single-penalty label layout");
    let eval = outerobjectivegradienthessian_labeled(
        family,
        specs,
        &options,
        &layout,
        &array![rho],
        None,
        &gam_problem::RhoPrior::Flat,
        EvalMode::ValueOnly,
    )
    .expect("cold-direct inner solve");
    assert!(
        eval.inner_converged,
        "cold-direct inner solve must converge"
    );
    eval.warm_start.block_beta[0][0]
}

fn continuation_mode(
    family: &TiltedDoubleWellFamily,
    specs: &[ParameterBlockSpec],
    rho: f64,
) -> f64 {
    let options = double_well_options();
    let penalty_counts: Vec<usize> = specs.iter().map(|spec| spec.penalties.len()).collect();
    let layout = penalty_label_layout_with_joint(specs, penalty_counts, Vec::new())
        .expect("single-penalty label layout");
    let certified = anchored_continuation_seed(
        family,
        specs,
        &options,
        &layout,
        &gam_problem::RhoPrior::Flat,
        &array![EFFECTIVE_DF_CEILING],
        &array![rho],
    )
    .expect("the continuation from the maximally-smoothed anchor must reach the target rho");
    assert!(
        certified.certificate.endpoint_discrepancy <= certified.certificate.inner_tolerance,
        "a returned continuation seed must carry its endpoint-invariance certificate"
    );
    certified.warm_start.block_beta[0][0]
}

/// CONTROL. Without a selection rule the mode is a functional of the seed: two
/// callers who ask the same question get different answers.
///
/// If this test ever starts failing because the two seeds agree, the fixture has
/// stopped discriminating and the two tests below prove nothing — so this one is
/// load-bearing, not decorative.
#[test]
fn cold_direct_mode_depends_on_the_seed_2366() {
    let family = TiltedDoubleWellFamily { tilt: TILT };
    let target_rho = -6.0;
    let from_positive = cold_direct_mode(&family, &[double_well_spec(2.0)], target_rho);
    let from_negative = cold_direct_mode(&family, &[double_well_spec(-2.0)], target_rho);
    assert!(
        from_positive > 0.5,
        "a seed inside the shallow well should stay in it; got {from_positive}"
    );
    assert!(
        from_negative < -0.5,
        "a seed inside the deep well should stay in it; got {from_negative}"
    );
}

/// The continuation endpoint is the SAME mode no matter what the caller seeds,
/// which is exactly the statement that `θ̂` is a function of ρ.
#[test]
fn anchored_continuation_mode_is_independent_of_the_seed_2366() {
    let family = TiltedDoubleWellFamily { tilt: TILT };
    let target_rho = -6.0;
    let from_positive = continuation_mode(&family, &[double_well_spec(2.0)], target_rho);
    let from_negative = continuation_mode(&family, &[double_well_spec(-2.0)], target_rho);
    let from_origin = continuation_mode(&family, &[double_well_spec(0.0)], target_rho);
    // Bitwise, not "close": a nonconvex profiled objective has no tolerance in
    // which two different branches are interchangeable provenance.
    assert_eq!(
        from_positive.to_bits(),
        from_negative.to_bits(),
        "continuation endpoints differ across seeds: {from_positive} vs {from_negative}"
    );
    assert_eq!(
        from_positive.to_bits(),
        from_origin.to_bits(),
        "continuation endpoints differ across seeds: {from_positive} vs {from_origin}"
    );
}

/// The end-to-end property: a whole production fit is a function of the model
/// and the data, not of the coefficients the caller happened to pass in.
///
/// This is the statement #2363 wants for cache state, obtained here from the
/// definition rather than from a per-family patch: the persistent cache seeds β,
/// and once β is selected by the continuation instead of by the seed, a warm
/// cache can change how fast the fit is reached but not where it lands.
/// The measured residual across seeds, `3.4e-14` relative, is far below what the
/// inner corrector resolves and far above zero. The cross-seed bound is stated
/// as the inner tolerance for that reason; a repeat-run control pins down which
/// of the two possible explanations applies, so the bound cannot quietly hide a
/// returning branch dependence. See the discussion on #2366.
#[test]
fn production_fit_is_independent_of_the_caller_seed_2366() {
    let family = TiltedDoubleWellFamily { tilt: TILT };
    let options = double_well_options();
    let fit = |seed: f64| {
        let result = fit_custom_family(&family, &[double_well_spec(seed)], &options)
            .expect("double-well fit");
        (result.block_states[0].beta[0], result.log_lambdas[0])
    };

    // CONTROL: two runs from the SAME seed must be bitwise identical. Without
    // this, a cross-seed bound stated at a tolerance could be satisfied by a
    // run-to-run wobble that has nothing to do with the seed, and the test would
    // stop measuring what it claims to measure.
    let (repeat_a, _) = fit(2.0);
    let (repeat_b, _) = fit(2.0);
    assert_eq!(
        repeat_a.to_bits(),
        repeat_b.to_bits(),
        "two fits of the same problem from the same seed disagree ({repeat_a} vs \
         {repeat_b}), so this fixture cannot attribute any difference to the seed"
    );

    let (from_positive, rho_positive) = fit(2.0);
    let (from_negative, rho_negative) = fit(-2.0);
    let cross_seed_gap = (from_positive - from_negative).abs();
    // The certified smoothing parameter is asserted alongside the coefficient
    // because they fail separately: an earlier revision of this fix left β
    // agreeing to 3.4e-14 while ρ still moved by 1.3e-13, which is how the
    // remaining leak (the stall guard's cold pulse falling back to the caller's
    // coefficients) was found. Both are bitwise now, and both are checked so
    // that channel cannot silently reopen.
    assert_eq!(
        rho_positive.to_bits(),
        rho_negative.to_bits(),
        "the certified smoothing parameter depends on the caller's seed: \
         {rho_positive} vs {rho_negative}"
    );

    // The qualitative property: both seeds select the SAME branch. Before the
    // anchored continuation these two seeds converged into opposite wells, so
    // this gap was ≈ 2. Sign agreement is the branch statement; the magnitude
    // bound is what makes it a fit-level statement rather than a sign check.
    assert!(
        from_positive < 0.0 && from_negative < 0.0,
        "both fits should land on the anchor's branch (beta < 0); got \
         {from_positive} and {from_negative}"
    );
    assert_eq!(
        from_positive.to_bits(),
        from_negative.to_bits(),
        "the fitted coefficient depends on the caller's seed: {from_positive} \
         vs {from_negative} (gap {cross_seed_gap:.3e})"
    );
}

/// The mode the rule selects is the DEEP well — the anchor's own branch — not
/// merely a consistent one.
///
/// A rule that consistently picked the worse mode would satisfy the invariance
/// test above while making every fit worse, so the selection has to be checked
/// against the closed-form geometry as well.
#[test]
fn anchored_continuation_selects_the_anchor_branch_2366() {
    let family = TiltedDoubleWellFamily { tilt: TILT };
    let target_rho = -6.0;
    let selected = continuation_mode(&family, &[double_well_spec(2.0)], target_rho);
    assert!(
        selected < -0.5,
        "the anchor is convexified around beta<0, so its branch is the deep well; got {selected}"
    );
    // The deep well is genuinely the better mode: w(-1) = -c < +c = w(+1).
    let shallow = cold_direct_mode(&family, &[double_well_spec(2.0)], target_rho);
    let objective = |beta: f64| {
        let well = beta * beta - 1.0;
        well * well + TILT * beta + 0.5 * (target_rho.exp()) * beta * beta
    };
    assert!(
        objective(selected) < objective(shallow),
        "selected mode {selected} (obj {}) should beat the seed-dependent mode {shallow} (obj {})",
        objective(selected),
        objective(shallow)
    );
}
