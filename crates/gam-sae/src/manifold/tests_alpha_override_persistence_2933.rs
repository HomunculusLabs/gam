//! #2933 F06 — minting a fit must not change the objective it certified.
//!
//! For an ordered Beta--Bernoulli assignment, `rho.log_lambda_sparse` is the
//! concentration offset while the concentration is effectively learnable. `into_fitted`
//! used to rewrite a raw-learnable mode to its resolved concentration and zero that
//! coordinate. Without an override the loss survived, but the persisted
//! `(alpha, learnable_alpha, log_lambda_sparse)` triple then reloaded `α_base` in place of
//! the fitted `α_base·exp(ρ)`.
//!
//! #2933 F45 — with the concentration fixed, by the mode or by a per-fit override, the prior
//! is the complete Beta--Bernoulli prior at weight one plus its constant partition, the rho
//! layout carries no sparse coordinate, and `log_lambda_sparse` is an unread placeholder. An
//! override-pinned learnable mode and its fixed-concentration twin are one objective.
//!
//! Every expected concentration here comes from the model definition, and every expected
//! prior value from an [`OrderedBetaBernoulliPenalty`] built directly at that concentration,
//! so no assertion reads the resolution it checks.

#![cfg(test)]

use super::tests::planted_circle_embedded;
use super::tests_recovery_split_780::gamma_fd_tiny_fixture;
use super::tests_startup_validation_1782::{Topo, build_term};
use super::*;
use crate::assignment::{assignment_prior_value_weighted, gate_logit_jacobian_value_weighted};
use gam_terms::analytic_penalties::{AnalyticPenalty, OrderedBetaBernoulliPenalty};
use ndarray::{Array1, Array2, array};

const BASE_ALPHA: f64 = 1.7;
const OVERRIDE_ALPHA: f64 = 0.6;
const TEMPERATURE: f64 = 1.0;

/// The concentration from the model definition at `ρ_sparse = ln 3`.
fn expected_concentration(learnable: bool, override_alpha: Option<f64>) -> f64 {
    match (learnable, override_alpha) {
        (_, Some(alpha)) => alpha,
        (true, None) => BASE_ALPHA * 3.0,
        (false, None) => BASE_ALPHA,
    }
}

/// The complete prior `P(concentration)` at weight one, with its partition, at `logits`,
/// with nothing resolved from rho.
fn expected_prior_value(logits: &Array2<f64>, concentration: f64) -> f64 {
    let flat = Array1::from_iter(logits.iter().copied());
    let penalty =
        OrderedBetaBernoulliPenalty::new(logits.ncols(), concentration, TEMPERATURE, false);
    let no_rho = Array1::<f64>::zeros(0);
    penalty.value(flat.view(), no_rho.view())
        + penalty
            .log_partition(flat.view(), no_rho.view())
            .expect("the untempered prior has a partition")
            .0
}

fn relative_gap(a: f64, b: f64) -> f64 {
    (a - b).abs() / (1.0 + a.abs().max(b.abs()))
}

fn max_abs(values: &Array2<f64>) -> f64 {
    values.iter().fold(0.0_f64, |acc, value| acc.max(value.abs()))
}

fn fixed_rho_objective(
    z: &Array2<f64>,
    term: SaeManifoldTerm,
    rho: &SaeManifoldRho,
) -> SaeManifoldOuterObjective {
    let mut objective = SaeManifoldOuterObjective::new(
        term,
        z.clone(),
        None,
        rho.clone(),
        8,
        0.04,
        1.0e-6,
        1.0e-6,
    );
    objective
        .fit_at_fixed_rho(rho.flat_coordinates().view())
        .unwrap_or_else(|error| panic!("fixed-rho fit must converge: {error}"));
    objective
}

/// A two-atom circle term under `mode` with the per-fit concentration override installed.
fn configured_term(
    z: &Array2<f64>,
    topology: Topo,
    mode: AssignmentMode,
    override_alpha: Option<f64>,
) -> SaeManifoldTerm {
    let (mut term, _) = build_term(z.view(), 2, topology, mode);
    let config = term.fit_config();
    term.set_fit_config(SaeFitConfig {
        ordered_beta_bernoulli_alpha_override: override_alpha,
        ..config
    });
    term
}

/// Fit at a fixed `ρ_sparse = ln 3`, mint the fit, then compare the prior concentration, the
/// full penalized loss, the criterion, assignments and reconstruction before and after
/// finalization, and (where the payload can represent the fit) after reloading the persisted
/// scalars.
fn assert_finalization_preserves_the_objective(learnable: bool, override_alpha: Option<f64>) {
    let label = format!("learnable_alpha={learnable}, override={override_alpha:?}");
    let z = planted_circle_embedded(24, 4, 0.03);
    let k = 2;
    let mode = AssignmentMode::ordered_beta_bernoulli(TEMPERATURE, BASE_ALPHA, learnable);
    let term = configured_term(&z, Topo::Circle, mode, override_alpha);
    let rho = SaeManifoldRho::new(3.0_f64.ln(), 0.0, vec![array![0.0]; k])
        .for_assignment(&term.assignment);
    let objective = fixed_rho_objective(&z, term, &rho);

    let concentration = expected_concentration(learnable, override_alpha);
    let learned = learnable && override_alpha.is_none();
    let rho_before = objective.current_rho.clone();
    assert_eq!(
        rho_before.sparse_flat_index().is_some(),
        learned,
        "{label}: before finalization a sparse coordinate exists exactly while the concentration \
         is learned"
    );
    let logits_before = objective.term.assignment.logits.clone();
    let loss_before = objective
        .term
        .loss(z.view(), &rho_before)
        .unwrap_or_else(|error| panic!("{label}: loss before finalization: {error}"));
    let criterion_before = objective
        .terminal_penalized_quasi_laplace_criterion
        .expect("a fixed-rho fit stamps its criterion");
    let assignments_before = objective.term.assignment.try_assignments()
        .expect("finite fitted logits give assignments");
    let reconstruction_before = objective.term.try_fitted()
        .expect("the fitted state reconstructs");
    let prior_expected = expected_prior_value(&logits_before, concentration);
    // The loss prior term also carries the gate-logit change of variables, which reads
    // neither the concentration nor the placeholder.
    let loss_prior_expected =
        prior_expected + gate_logit_jacobian_value_weighted(&objective.term.assignment, None);
    assert!(
        relative_gap(loss_before.assignment_sparsity, loss_prior_expected) <= 1.0e-12,
        "{label}: before finalization the fit scores the prior term as {:.12e}, but \
         P(concentration {concentration}) plus the logit Jacobian is {loss_prior_expected:.12e}",
        loss_before.assignment_sparsity
    );

    let fitted = objective
        .into_fitted()
        .unwrap_or_else(|error| panic!("{label}: into_fitted: {error}"));
    assert_eq!(
        fitted.term.assignment.logits, logits_before,
        "{label}: finalization must not move the logits"
    );
    let recomputed = fitted
        .term
        .loss(z.view(), &fitted.rho)
        .unwrap_or_else(|error| panic!("{label}: loss after finalization: {error}"));
    for (what, value) in [
        ("returned loss", fitted.loss.assignment_sparsity),
        ("recomputed loss", recomputed.assignment_sparsity),
    ] {
        assert!(
            relative_gap(value, loss_prior_expected) <= 1.0e-12,
            "{label}: after finalization the {what} scores the prior term as {value:.12e}, but \
             P(concentration {concentration}) plus the logit Jacobian is \
             {loss_prior_expected:.12e}"
        );
    }
    assert_eq!(
        fitted.rho.flat_coordinates(),
        rho_before.flat_coordinates(),
        "{label}: the certified rho coordinates must leave finalization unchanged"
    );
    assert_eq!(
        (
            fitted.rho.assignment_strength_layout,
            fitted.term.assignment.ordered_beta_bernoulli_alpha_override,
        ),
        (rho_before.assignment_strength_layout, override_alpha),
        "{label}: finalization must rewrite neither the layout nor the override"
    );
    assert!(
        matches!(
            fitted.term.assignment.mode,
            AssignmentMode::OrderedBetaBernoulli { alpha, learnable_alpha, .. }
                if alpha == BASE_ALPHA && learnable_alpha == learnable
        ),
        "{label}: finalization must not rewrite the mode: {:?}",
        fitted.term.assignment.mode
    );
    assert_eq!(
        fitted.rho.sparse_flat_index().is_some(),
        learned,
        "{label}: after finalization a sparse coordinate exists exactly while the concentration \
         is learned"
    );
    // Chart canonicalization is image-frozen and objective-gated, so everything outside
    // the prior may move only at that roundoff.
    let loss_gap = relative_gap(fitted.loss.total(), loss_before.total());
    assert!(
        loss_gap <= 1.0e-6,
        "{label}: full penalized loss {:.12e} before finalization, {:.12e} after \
         (relative gap {loss_gap:.3e}, charts_canonicalized={})",
        loss_before.total(),
        fitted.loss.total(),
        fitted.charts_canonicalized
    );
    assert_eq!(
        fitted.term.assignment.try_assignments()
        .expect("finite fitted logits give assignments"),
        assignments_before,
        "{label}: finalization must not change assignments"
    );
    let reconstruction_after = fitted.term.try_fitted()
        .expect("the fitted state reconstructs");
    let reconstruction_gap = max_abs(&(&reconstruction_after - &reconstruction_before));
    assert!(
        reconstruction_gap <= 1.0e-6 * (1.0 + max_abs(&reconstruction_before)),
        "{label}: reconstruction moved by {reconstruction_gap:.3e} through finalization"
    );

    // The criterion the fit reports must describe the returned (term, rho): re-mint it
    // through the same authoritative fixed-rho lane.
    let reevaluated = fixed_rho_objective(&z, fitted.term.clone(), &fitted.rho);
    let criterion_after = reevaluated
        .terminal_penalized_quasi_laplace_criterion
        .expect("a fixed-rho fit stamps its criterion");
    let criterion_tolerance = 1.0e-6 * (1.0 + criterion_before.abs());
    assert!(
        (criterion_after - criterion_before).abs() <= criterion_tolerance,
        "{label}: reported criterion {criterion_before:.12e}, but the returned state scores \
         {criterion_after:.12e}"
    );
    assert!(
        relative_gap(fitted.penalized_quasi_laplace_criterion, criterion_before) == 0.0,
        "{label}: into_fitted must carry the certified criterion"
    );

    if override_alpha.is_none() {
        // `ManifoldSaePayload` stores the request's `alpha` and `learnable_alpha` beside the
        // returned `log_lambda_sparse`, and frozen-decoder OOS rebuilds the mode and rho from
        // exactly those scalars. The payload has no override field, so only the
        // override-free fits round-trip through it.
        let mut reloaded = fitted.term.assignment.clone();
        reloaded.mode = AssignmentMode::ordered_beta_bernoulli(TEMPERATURE, BASE_ALPHA, learnable);
        let reloaded_rho = SaeManifoldRho::with_per_atom_smooth(
            fitted.rho.log_lambda_sparse,
            fitted.rho.log_lambda_smooth.clone(),
            fitted.rho.log_ard.clone(),
        )
        .for_assignment(&reloaded);
        let reloaded_prior = assignment_prior_value_weighted(&reloaded, &reloaded_rho, None)
            .unwrap_or_else(|error| panic!("{label}: reloaded prior: {error}"));
        assert!(
            relative_gap(reloaded_prior, prior_expected) <= 1.0e-12,
            "{label}: the reloaded payload scores the prior as {reloaded_prior:.12e}, but the \
             fit scored P(concentration {concentration}) = {prior_expected:.12e} (persisted \
             log_lambda_sparse={})",
            fitted.rho.log_lambda_sparse
        );
    }
}

#[test]
fn fixed_alpha_finalizes_without_changing_the_objective_2933() {
    assert_finalization_preserves_the_objective(false, None);
}

#[test]
fn overridden_fixed_alpha_finalizes_without_changing_the_objective_2933() {
    assert_finalization_preserves_the_objective(false, Some(OVERRIDE_ALPHA));
}

#[test]
fn learnable_alpha_finalizes_and_reloads_its_fitted_concentration_2933() {
    assert_finalization_preserves_the_objective(true, None);
}

#[test]
fn overridden_learnable_alpha_finalizes_at_the_override_concentration_2933() {
    assert_finalization_preserves_the_objective(true, Some(OVERRIDE_ALPHA));
}

/// #2933 F45 — the outer objective owns the flat layout, and a fit whose concentration is
/// fixed has no sparse coordinate to optimize: nothing in its prior reads one. Before the
/// change every ordered Beta--Bernoulli fit minted `log_lambda_sparse` as a prior-weight
/// coordinate.
#[test]
fn fixed_concentration_fits_expose_no_sparse_coordinate_2933() {
    let z = planted_circle_embedded(24, 4, 0.03);
    let k = 2;
    for (learnable, override_alpha) in [(true, None), (false, None), (true, Some(OVERRIDE_ALPHA))]
    {
        let mode = AssignmentMode::ordered_beta_bernoulli(TEMPERATURE, BASE_ALPHA, learnable);
        let term = configured_term(&z, Topo::Circle, mode, override_alpha);
        let objective = SaeManifoldOuterObjective::new(
            term,
            z.clone(),
            None,
            SaeManifoldRho::new(3.0_f64.ln(), 0.0, vec![array![0.0]; k]),
            8,
            0.04,
            1.0e-6,
            1.0e-6,
        );
        let learned = learnable && override_alpha.is_none();
        assert_eq!(
            objective.baseline_rho.sparse_flat_index().is_some(),
            learned,
            "learnable_alpha={learnable}, override={override_alpha:?}: the objective must carry \
             a sparse coordinate exactly while the concentration is learned"
        );
        assert_eq!(
            objective.baseline_rho.flat_coordinates().len(),
            usize::from(learned) + 2 * k,
            "learnable_alpha={learnable}, override={override_alpha:?}: flat layout is \
             sparse + K smoothness + K ARD"
        );
    }
}

/// A rho bound for one effective concentration is refused, not silently rebound, once the
/// assignment disagrees: an override installed after binding removes the coordinate the flat
/// layout still carries, and clearing it restores one the layout lacks. The unbound constructor
/// tag and a correctly bound layout are admitted.
#[test]
fn a_layout_contradicting_the_effective_concentration_is_refused_2933() {
    let mut assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        Array2::<f64>::zeros((3, 2)),
        vec![Array2::<f64>::zeros((3, 1)); 2],
        vec![LatentManifold::Euclidean; 2],
        AssignmentMode::ordered_beta_bernoulli(TEMPERATURE, BASE_ALPHA, true),
    )
    .expect("zero logits and matching coordinate blocks build an assignment");
    let unbound = SaeManifoldRho::new(0.0, 0.0, vec![Array1::zeros(1); 2]);
    let learned = unbound.clone().for_assignment(&assignment);
    assignment
        .validate_rho_domain(&learned)
        .expect("a layout bound for the learnable concentration is admitted");
    assignment.set_ordered_beta_bernoulli_alpha_override(Some(OVERRIDE_ALPHA));
    let refusal = assignment
        .validate_rho_domain(&learned)
        .expect_err("an override installed after binding must be refused");
    assert!(refusal.contains("#2933 F45"), "{refusal}");
    let fixed = unbound.clone().for_assignment(&assignment);
    assignment
        .validate_rho_domain(&fixed)
        .expect("a layout bound after the override is admitted");
    assignment
        .validate_rho_domain(&unbound)
        .expect("the unbound constructor layout is admitted");
    assignment.set_ordered_beta_bernoulli_alpha_override(None);
    assert!(
        assignment.validate_rho_domain(&fixed).is_err(),
        "clearing the override restores a coordinate the fixed layout lacks"
    );
    let softmax = SaeAssignment::from_blocks_with_mode_and_manifolds(
        Array2::<f64>::zeros((3, 2)),
        vec![Array2::<f64>::zeros((3, 1)); 2],
        vec![LatentManifold::Euclidean; 2],
        AssignmentMode::softmax(TEMPERATURE),
    )
    .expect("zero logits and matching coordinate blocks build an assignment");
    assert!(
        softmax.validate_rho_domain(&fixed).is_err(),
        "an ordered Beta--Bernoulli layout does not describe a softmax assignment"
    );
}

/// A fixed concentration puts no coordinate into the prior, so both log-strength trace routes
/// are exactly zero for the fixed mode and for an override-pinned learnable mode. The
/// learnable control at the same concentration (`ρ_sparse = 0`, base `OVERRIDE_ALPHA`) scores
/// the same objective on the same converged cache and must keep a live trace, so the zeros
/// are not a trace function that returns nothing.
#[test]
fn fixed_concentration_log_strength_traces_are_exact_zeros_2933() {
    let (mut twin, target, mut rho) = gamma_fd_tiny_fixture();
    twin.assignment.mode = AssignmentMode::ordered_beta_bernoulli(0.7, OVERRIDE_ALPHA, false);
    rho.log_lambda_sparse = 0.0;
    let (_value, _loss, cache) = twin
        .penalized_quasi_laplace_criterion_with_cache(
            target.view(),
            &rho,
            None,
            5,
            0.4,
            1.0e-6,
            1.0e-6,
        )
        .expect("converged cache for the fixed-concentration twin");
    let mut pinned = twin.clone();
    pinned.assignment.mode = AssignmentMode::ordered_beta_bernoulli(0.7, BASE_ALPHA, true);
    pinned
        .assignment
        .set_ordered_beta_bernoulli_alpha_override(Some(OVERRIDE_ALPHA));
    let mut control = twin.clone();
    control.assignment.mode = AssignmentMode::ordered_beta_bernoulli(0.7, OVERRIDE_ALPHA, true);

    let solver = DeflatedArrowSolver::plain(&cache);
    let k_border = cache.k;
    assert!(k_border > 0, "the fixture must carry a decoder border");
    let sqrt_k = (k_border as f64).sqrt();
    let probes: Vec<Array1<f64>> = (0..k_border)
        .map(|j| {
            let mut probe = Array1::<f64>::zeros(k_border);
            probe[j] = sqrt_k;
            probe
        })
        .collect();
    let sinv: Vec<Array1<f64>> = probes
        .iter()
        .map(|probe| cache.schur_inverse_apply(probe.view()).expect("schur solve"))
        .collect();
    let traces = |term: &SaeManifoldTerm| {
        let dense = term
            .assignment_log_strength_hessian_trace(&rho, &cache, &solver)
            .expect("dense log-strength trace");
        let probed = term
            .assignment_log_strength_hessian_trace_from_probes(
                &rho,
                &cache,
                &probes,
                &sinv,
                EvidenceOperator::Majorizer,
            )
            .expect("from-probes log-strength trace");
        (dense, probed)
    };
    let (control_dense, control_probed) = traces(&control);
    assert!(
        control_dense.abs() > 1.0e-6 && control_probed.abs() > 1.0e-6,
        "the learnable control must keep a live log-concentration trace: dense \
         {control_dense:.6e}, from probes {control_probed:.6e}"
    );
    for (label, term) in [("fixed twin", &twin), ("override-pinned learnable", &pinned)] {
        assert_eq!(traces(term), (0.0, 0.0), "{label}");
    }
}

/// The typed resolution names, for all four configurations, the concentration the prior uses
/// and whether `log_lambda_sparse` carries it.
#[test]
fn prior_parameters_name_the_concentration_and_its_learnability_2933() {
    for learnable in [false, true] {
        for override_alpha in [None, Some(OVERRIDE_ALPHA)] {
            let mut assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
                Array2::<f64>::zeros((3, 2)),
                vec![Array2::<f64>::zeros((3, 1)); 2],
                vec![LatentManifold::Euclidean; 2],
                AssignmentMode::ordered_beta_bernoulli(TEMPERATURE, BASE_ALPHA, learnable),
            )
            .expect("zero logits and matching coordinate blocks build an assignment");
            assignment.set_ordered_beta_bernoulli_alpha_override(override_alpha);
            let rho = SaeManifoldRho::new(3.0_f64.ln(), 0.0, vec![Array1::zeros(1); 2])
                .for_assignment(&assignment);
            let parameters = assignment
                .ordered_beta_bernoulli_prior_parameters(&rho)
                .expect("rho = ln 3 is inside every configuration's domain")
                .expect("an ordered Beta--Bernoulli mode resolves prior parameters");
            let concentration = expected_concentration(learnable, override_alpha);
            let label = format!("learnable_alpha={learnable}, override={override_alpha:?}");
            assert!(
                relative_gap(parameters.concentration, concentration) <= 1.0e-14,
                "{label}: concentration {} != {concentration}",
                parameters.concentration
            );
            assert_eq!(
                parameters.concentration_is_learnable,
                learnable && override_alpha.is_none(),
                "{label}"
            );
        }
    }
}

/// `(sparse target and upper face when present, whole upper face)` of the reactive entry box
/// for one ordered Beta--Bernoulli configuration on a fixed Euclidean fixture.
fn reactive_sparse_face(
    mode: AssignmentMode,
    override_alpha: Option<f64>,
) -> (Option<(f64, f64)>, Array1<f64>) {
    let z = planted_circle_embedded(24, 4, 0.03);
    let k = 2;
    let term = configured_term(&z, Topo::Euclidean, mode, override_alpha);
    let rho = SaeManifoldRho::new(0.01_f64.ln(), 0.0, vec![array![0.0]; k])
        .for_assignment(&term.assignment);
    let upper = super::outer_objective::reactive_rho_domain_upper(&term, &rho, TEMPERATURE)
        .unwrap_or_else(|error| panic!("reactive box for override={override_alpha:?}: {error}"));
    let face = rho
        .sparse_flat_index()
        .map(|index| (rho.flat_coordinates()[index], upper[index]));
    (face, upper)
}

/// The reactive entry box follows the layout. An override-pinned learnable mode and its
/// fixed-concentration twin define one objective with no sparse coordinate, so they receive
/// one box. The learnable mode without an override, which keeps the native-curvature cap on
/// its concentration coordinate, shows that the cap is live on this fixture.
#[test]
fn overridden_learnable_alpha_reactive_box_matches_the_fixed_twin_2933() {
    let (control_face, _) = reactive_sparse_face(
        AssignmentMode::ordered_beta_bernoulli(TEMPERATURE, BASE_ALPHA, true),
        None,
    );
    let (control_target, control_face) =
        control_face.expect("a learnable concentration owns the sparse coordinate");
    assert!(
        control_face > control_target + 1.0,
        "the native-curvature cap must be live on this fixture: sparse target \
         {control_target:.6}, capped face {control_face:.6}"
    );
    let (twin_face, twin_box) = reactive_sparse_face(
        AssignmentMode::ordered_beta_bernoulli(TEMPERATURE, OVERRIDE_ALPHA, false),
        None,
    );
    let (pinned_face, pinned_box) = reactive_sparse_face(
        AssignmentMode::ordered_beta_bernoulli(TEMPERATURE, BASE_ALPHA, true),
        Some(OVERRIDE_ALPHA),
    );
    assert_eq!(twin_face, None, "a fixed concentration has no sparse face");
    assert_eq!(pinned_face, None, "an override-pinned concentration has no sparse face");
    assert_eq!(
        pinned_box, twin_box,
        "override-pinned learnable mode and its fixed twin define one objective and must \
         receive one reactive box"
    );
}
