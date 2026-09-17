//! #2933 F06 — minting a fit must not change the objective it certified.
//!
//! For an ordered Beta--Bernoulli assignment, `rho.log_lambda_sparse` is the
//! concentration offset while the concentration is effectively learnable, and the
//! prior weight otherwise. `into_fitted` used to rewrite a raw-learnable mode to its
//! resolved concentration and zero that coordinate. Under a per-fit override the
//! concentration never read the coordinate, so the rewrite turned
//! `exp(ρ)·P(α_override)` into `P(α_override)`. Without an override the loss survived,
//! but the persisted `(alpha, learnable_alpha, log_lambda_sparse)` triple then
//! reloaded `α_base` in place of the fitted `α_base·exp(ρ)`.
//!
//! Every expected concentration and weight here comes from the model definition,
//! and every expected prior value from an [`OrderedBetaBernoulliPenalty`] built
//! directly at those two numbers, so no assertion reads the resolution it checks.

#![cfg(test)]

use super::tests::planted_circle_embedded;
use super::tests_recovery_split_780::gamma_fd_tiny_fixture;
use super::tests_startup_validation_1782::{Topo, build_term};
use super::*;
use crate::assignment::{
    assignment_prior_value_weighted, gate_logit_jacobian_value_weighted,
    ordered_beta_bernoulli_psd_majorizer_third_channels_weighted,
};
use gam_terms::analytic_penalties::{AnalyticPenalty, OrderedBetaBernoulliPenalty};
use ndarray::{Array1, Array2, array};

const BASE_ALPHA: f64 = 1.7;
const OVERRIDE_ALPHA: f64 = 0.6;
const TEMPERATURE: f64 = 1.0;

/// `(concentration, weight)` from the model definition at `ρ_sparse = ln 3`.
fn expected_prior_parameters(learnable: bool, override_alpha: Option<f64>) -> (f64, f64) {
    match (learnable, override_alpha) {
        (_, Some(alpha)) => (alpha, 3.0),
        (true, None) => (BASE_ALPHA * 3.0, 1.0),
        (false, None) => (BASE_ALPHA, 3.0),
    }
}

/// `weight · P(concentration)` at `logits`, with nothing resolved from rho.
fn expected_prior_value(logits: &Array2<f64>, concentration: f64, weight: f64) -> f64 {
    let flat = Array1::from_iter(logits.iter().copied());
    let penalty =
        OrderedBetaBernoulliPenalty::new(logits.ncols(), concentration, TEMPERATURE, false);
    weight * penalty.value(flat.view(), Array1::<f64>::zeros(0).view())
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
        .fit_at_fixed_rho(rho.to_flat().view())
        .unwrap_or_else(|error| panic!("fixed-rho fit must converge: {error}"));
    objective
}

/// Fit at a fixed `ρ_sparse = ln 3`, mint the fit, then compare prior concentration and
/// weight, the full penalized loss, the criterion, assignments and reconstruction before
/// and after finalization, and (where the payload can represent the fit) after reloading
/// the persisted scalars.
fn assert_finalization_preserves_the_objective(learnable: bool, override_alpha: Option<f64>) {
    let label = format!("learnable_alpha={learnable}, override={override_alpha:?}");
    let z = planted_circle_embedded(24, 4, 0.03);
    let k = 2;
    let mode = AssignmentMode::ordered_beta_bernoulli(TEMPERATURE, BASE_ALPHA, learnable);
    let (mut term, _) = build_term(z.view(), k, Topo::Circle, mode);
    let config = term.fit_config();
    term.set_fit_config(SaeFitConfig {
        ordered_beta_bernoulli_alpha_override: override_alpha,
        ..config
    });
    let rho = SaeManifoldRho::new(3.0_f64.ln(), 0.0, vec![array![0.0]; k]).for_assignment(mode);
    let objective = fixed_rho_objective(&z, term, &rho);

    let (concentration, weight) = expected_prior_parameters(learnable, override_alpha);
    let rho_before = objective.current_rho.clone();
    let logits_before = objective.term.assignment.logits.clone();
    let loss_before = objective
        .term
        .loss(z.view(), &rho_before)
        .unwrap_or_else(|error| panic!("{label}: loss before finalization: {error}"));
    let criterion_before = objective
        .terminal_penalized_quasi_laplace_criterion
        .expect("a fixed-rho fit stamps its criterion");
    let assignments_before = objective.term.assignment.try_assignments().unwrap();
    let reconstruction_before = objective.term.try_fitted().unwrap();
    let prior_expected = expected_prior_value(&logits_before, concentration, weight);
    assert!(
        prior_expected.abs() >= 1.0e-2,
        "{label}: the fixture's prior value {prior_expected:.3e} is too small to resolve a \
         rescaled weight"
    );
    // The loss prior term also carries the gate-logit change of variables, which reads
    // neither the concentration nor the weight.
    let loss_prior_expected =
        prior_expected + gate_logit_jacobian_value_weighted(&objective.term.assignment, None);
    assert!(
        relative_gap(loss_before.assignment_sparsity, loss_prior_expected) <= 1.0e-12,
        "{label}: before finalization the fit scores the prior term as {:.12e}, but \
         weight {weight} · P(concentration {concentration}) plus the logit Jacobian is \
         {loss_prior_expected:.12e}",
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
             weight {weight} · P(concentration {concentration}) plus the logit Jacobian is \
             {loss_prior_expected:.12e}"
        );
    }
    assert_eq!(
        fitted.rho.to_flat(),
        rho_before.to_flat(),
        "{label}: the certified rho coordinate must leave finalization unchanged"
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
        fitted.term.assignment.try_assignments().unwrap(),
        assignments_before,
        "{label}: finalization must not change assignments"
    );
    let reconstruction_after = fitted.term.try_fitted().unwrap();
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
    if weight != 1.0 {
        assert!(
            (1.0 - 1.0 / weight) * prior_expected.abs() >= 100.0 * criterion_tolerance,
            "{label}: a lost prior weight must be resolvable by the criterion comparison"
        );
    }

    if override_alpha.is_none() {
        // `ManifoldSaePayload` stores the request's `alpha` and `learnable_alpha` beside the
        // returned `log_lambda_sparse`, and frozen-decoder OOS rebuilds the mode and rho from
        // exactly those scalars. The payload has no override field, so only the
        // override-free fits round-trip through it.
        let reloaded_mode =
            AssignmentMode::ordered_beta_bernoulli(TEMPERATURE, BASE_ALPHA, learnable);
        let mut reloaded = fitted.term.assignment.clone();
        reloaded.mode = reloaded_mode;
        let reloaded_rho = SaeManifoldRho::with_per_atom_smooth(
            fitted.rho.log_lambda_sparse,
            fitted.rho.log_lambda_smooth.clone(),
            fitted.rho.log_ard.clone(),
        )
        .for_assignment(reloaded_mode);
        let reloaded_prior = assignment_prior_value_weighted(&reloaded, &reloaded_rho, None)
            .unwrap_or_else(|error| panic!("{label}: reloaded prior: {error}"));
        assert!(
            relative_gap(reloaded_prior, prior_expected) <= 1.0e-12,
            "{label}: the reloaded payload scores the prior as {reloaded_prior:.12e}, but the \
             fit scored weight {weight} · P(concentration {concentration}) = \
             {prior_expected:.12e} (persisted log_lambda_sparse={})",
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
fn overridden_learnable_alpha_keeps_its_prior_weight_through_finalization_2933() {
    assert_finalization_preserves_the_objective(true, Some(OVERRIDE_ALPHA));
}

/// Under an override the sparse coordinate is the prior weight even when the mode was
/// built learnable. A fixed-mode twin at the override concentration is the same
/// objective, so both assignment log-strength trace routes must agree with it on the
/// same converged cache.
#[test]
fn overridden_learnable_alpha_log_strength_traces_match_the_fixed_twin_2933() {
    let (mut twin, target, mut rho) = gamma_fd_tiny_fixture();
    twin.assignment.mode = AssignmentMode::ordered_beta_bernoulli(0.7, OVERRIDE_ALPHA, false);
    let feasible = [1.0_f64, 1.5, 2.0, 2.5, 3.0, 0.5, 0.0, -0.5]
        .into_iter()
        .find(|&log_strength| {
            let mut probe = twin.clone();
            let mut probe_rho = rho.clone();
            probe_rho.log_lambda_sparse = log_strength;
            probe
                .penalized_quasi_laplace_criterion_with_cache(
                    target.view(),
                    &probe_rho,
                    None,
                    5,
                    0.4,
                    1.0e-6,
                    1.0e-6,
                )
                .is_ok()
        })
        .expect("no PD-region rho for the fixed-alpha twin");
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
        .expect("converged cache at the PD rho");
    let mut pinned = twin.clone();
    pinned.assignment.mode = AssignmentMode::ordered_beta_bernoulli(0.7, BASE_ALPHA, true);
    pinned
        .assignment
        .set_ordered_beta_bernoulli_alpha_override(Some(OVERRIDE_ALPHA));

    // Non-vacuity: the weight and concentration majorizer derivatives differ exactly on
    // slots whose row-local diagonal is kept and whose rank-one coefficient is live.
    let k_atoms = twin.k_atoms();
    let channels = ordered_beta_bernoulli_psd_majorizer_third_channels_weighted(
        &twin.assignment,
        &rho,
        None,
    )
    .expect("ordered prior channels")
    .expect("a free-logit ordered prior carries channels");
    let live_slots = (0..channels.diagonal_term.len())
        .filter(|&slot| {
            channels.diagonal_term[slot] > 0.0
                && channels.mass_hessian_coefficient[slot % k_atoms] * channels.z_jac[slot].powi(2)
                    != 0.0
        })
        .count();
    assert!(
        live_slots > 0,
        "the fixture must keep at least one majorized logit slot with a live rank-one term"
    );

    let solver = DeflatedArrowSolver::plain(&cache);
    let twin_dense = twin
        .assignment_log_strength_hessian_trace(&rho, &cache, &solver)
        .expect("fixed twin dense trace");
    let pinned_dense = pinned
        .assignment_log_strength_hessian_trace(&rho, &cache, &solver)
        .expect("override-pinned dense trace");
    assert!(
        relative_gap(pinned_dense, twin_dense) <= 1.0e-12,
        "dense log-strength trace: override-pinned learnable mode {pinned_dense:.12e}, \
         fixed twin {twin_dense:.12e}"
    );

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
    let twin_probed = twin
        .assignment_log_strength_hessian_trace_from_probes(
            &rho,
            &cache,
            &probes,
            &sinv,
            EvidenceOperator::Majorizer,
        )
        .expect("fixed twin from-probes trace");
    let pinned_probed = pinned
        .assignment_log_strength_hessian_trace_from_probes(
            &rho,
            &cache,
            &probes,
            &sinv,
            EvidenceOperator::Majorizer,
        )
        .expect("override-pinned from-probes trace");
    assert!(
        relative_gap(pinned_probed, twin_probed) <= 1.0e-12,
        "from-probes log-strength trace: override-pinned learnable mode {pinned_probed:.12e}, \
         fixed twin {twin_probed:.12e}"
    );
}

/// The typed resolution names, for all four configurations, which quantity the sparse
/// coordinate carries and what concentration and weight the prior uses.
#[test]
fn prior_parameters_name_the_concentration_and_the_weight_2933() {
    use crate::assignment::OrderedBetaBernoulliSparseCoordinate;
    for learnable in [false, true] {
        for override_alpha in [None, Some(OVERRIDE_ALPHA)] {
            let mut assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
                Array2::<f64>::zeros((3, 2)),
                vec![Array2::<f64>::zeros((3, 1)); 2],
                vec![LatentManifold::Euclidean; 2],
                AssignmentMode::ordered_beta_bernoulli(TEMPERATURE, BASE_ALPHA, learnable),
            )
            .unwrap();
            assignment.set_ordered_beta_bernoulli_alpha_override(override_alpha);
            let rho = SaeManifoldRho::new(3.0_f64.ln(), 0.0, vec![Array1::zeros(1); 2])
                .for_assignment(assignment.mode);
            let parameters = assignment
                .ordered_beta_bernoulli_prior_parameters(&rho)
                .unwrap()
                .expect("an ordered Beta--Bernoulli mode resolves prior parameters");
            let (concentration, weight) = expected_prior_parameters(learnable, override_alpha);
            let label = format!("learnable_alpha={learnable}, override={override_alpha:?}");
            assert!(
                relative_gap(parameters.concentration, concentration) <= 1.0e-14,
                "{label}: concentration {} != {concentration}",
                parameters.concentration
            );
            assert!(
                relative_gap(parameters.weight, weight) <= 1.0e-14,
                "{label}: weight {} != {weight}",
                parameters.weight
            );
            let expected_coordinate = if learnable && override_alpha.is_none() {
                OrderedBetaBernoulliSparseCoordinate::LogConcentrationOffset
            } else {
                OrderedBetaBernoulliSparseCoordinate::LogPenaltyWeight
            };
            assert_eq!(parameters.sparse_coordinate, expected_coordinate, "{label}");
        }
    }
}

/// `(sparse target, sparse upper face, whole upper face)` of the reactive entry box for
/// one ordered Beta--Bernoulli configuration on a fixed Euclidean fixture.
fn reactive_sparse_face(
    mode: AssignmentMode,
    override_alpha: Option<f64>,
) -> (f64, f64, Array1<f64>) {
    let z = planted_circle_embedded(24, 4, 0.03);
    let k = 2;
    let (mut term, _) = build_term(z.view(), k, Topo::Euclidean, mode);
    let config = term.fit_config();
    term.set_fit_config(SaeFitConfig {
        ordered_beta_bernoulli_alpha_override: override_alpha,
        ..config
    });
    let rho = SaeManifoldRho::new(0.01_f64.ln(), 0.0, vec![array![0.0]; k]).for_assignment(mode);
    let index = rho
        .sparse_flat_index()
        .expect("an ordered Beta--Bernoulli prior owns a sparse coordinate");
    let upper = super::outer_objective::reactive_rho_domain_upper(&term, &rho, TEMPERATURE)
        .unwrap_or_else(|error| panic!("reactive box for override={override_alpha:?}: {error}"));
    (rho.to_flat()[index], upper[index], upper)
}

/// The reactive entry box follows what the sparse coordinate carries. An override-pinned
/// learnable mode and its fixed-concentration twin define one objective, so they must
/// receive one box. The learnable mode without an override, which keeps the
/// native-curvature cap, shows that the cap is live on this fixture.
#[test]
fn overridden_learnable_alpha_reactive_box_matches_the_fixed_twin_2933() {
    let (control_target, control_face, _) = reactive_sparse_face(
        AssignmentMode::ordered_beta_bernoulli(TEMPERATURE, BASE_ALPHA, true),
        None,
    );
    assert!(
        control_face > control_target + 1.0,
        "the native-curvature cap must be live on this fixture: sparse target \
         {control_target:.6}, capped face {control_face:.6}"
    );
    let (_, twin_face, twin_box) = reactive_sparse_face(
        AssignmentMode::ordered_beta_bernoulli(TEMPERATURE, OVERRIDE_ALPHA, false),
        None,
    );
    let (_, pinned_face, pinned_box) = reactive_sparse_face(
        AssignmentMode::ordered_beta_bernoulli(TEMPERATURE, BASE_ALPHA, true),
        Some(OVERRIDE_ALPHA),
    );
    assert_eq!(
        pinned_box, twin_box,
        "override-pinned learnable mode (sparse face {pinned_face:.6}) and its fixed twin \
         (sparse face {twin_face:.6}) define one objective and must receive one reactive box"
    );
}
