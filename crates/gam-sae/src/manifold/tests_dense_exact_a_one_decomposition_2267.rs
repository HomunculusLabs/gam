//! #2267 — a dense outer evaluation decomposes its state's exact observed information once.
//!
//! The dense criterion prices `½log|A|` off one pencil eigensystem of `A`. Its gradient used
//! to decompose the same `A` again for each consumer entered without it: the rank-charge
//! derivative's dispersion, the log-determinant channel geometry, and a second rank-charge
//! derivative inside those channels. Probe 1161362 read one 7692-dimensional state
//! decomposed four times before its 900 s cap, about 140 s each, and the complete value and
//! gradient would have decomposed it once more. The evaluation now owns the block and hands
//! it to every consumer.
//!
//! The count pin reads the production counter the `[SAE-EXACT-DENSE]` line prints. Its
//! mutant is the no-geometry rank-charge route, a `None` block at the assembler's rank-charge
//! call, which must fail it at 1 gradient decomposition. The identity pin compares the
//! received block against a decomposition at the evaluation's own cache, which is what the
//! gradient paid for before: every priced derivative must agree bit for bit.

use super::tests_sparse_curvature_operator_2500::threshold_gate_tiny_fixture;
use super::*;
use ndarray::Array1;

/// A frozen fixed state: no inner iterations, so the value materializes `A` at the
/// fixture's own state and descends no saddle.
const INNER_MAX_ITER: usize = 0;
const LEARNING_RATE: f64 = 0.4;
const RIDGE: f64 = 1.0e-6;

fn bits(values: &Array1<f64>) -> Vec<u64> {
    values.iter().map(|value| value.to_bits()).collect()
}

#[test]
fn a_dense_outer_evaluation_decomposes_its_state_once_2267() {
    let (term, target, rho) = threshold_gate_tiny_fixture(false);
    let mut objective = SaeManifoldOuterObjective::new(
        term,
        target,
        None,
        rho,
        INNER_MAX_ITER,
        LEARNING_RATE,
        RIDGE,
        RIDGE,
    );
    let route_rho = objective.baseline_rho.clone();
    let entered = super::construction::exact_a_pencil_decompositions_on_this_thread();
    let evaluation = objective
        .evaluate_outer_criterion_route(&route_rho, true, false)
        .expect("#2267: the fixture prices a dense criterion at its own rho");
    let priced = super::construction::exact_a_pencil_decompositions_on_this_thread();
    let gradient = objective
        .analytic_gradient_for_outer_evaluation(&route_rho, &evaluation)
        .expect("#2267: the dense route differentiates its own evaluation");
    let differentiated = super::construction::exact_a_pencil_decompositions_on_this_thread();
    // The bits let a lane compare this value and gradient against a build before #2267.
    println!(
        "[#2267] decompositions: value {}, gradient {}; cost bits {:016x}; gradient bits {:?}",
        priced - entered,
        differentiated - priced,
        evaluation.cost.to_bits(),
        gradient
            .iter()
            .map(|value| format!("{:016x}", value.to_bits()))
            .collect::<Vec<_>>(),
    );
    assert!(
        gradient.iter().any(|value| *value != 0.0),
        "#2267: the fixture's gradient must be live, else a skipped derivative decomposes nothing"
    );
    assert_eq!(
        priced - entered,
        1,
        "#2267: the dense value decomposes its state once"
    );
    assert_eq!(
        differentiated - priced,
        0,
        "#2267: the gradient reads the block its value priced and decomposes nothing"
    );
}

#[test]
fn the_evaluations_block_prices_what_a_fresh_decomposition_prices_2267() {
    let (mut term, target, rho) = threshold_gate_tiny_fixture(false);
    let (_value, loss, cache, received) = term
        .penalized_quasi_laplace_criterion_with_geometry(
            target.view(),
            &rho,
            None,
            INNER_MAX_ITER,
            LEARNING_RATE,
            RIDGE,
            RIDGE,
            true,
        )
        .expect("#2267: the fixture prices a dense criterion at its own rho");
    let received = received.expect("#2267: the dense criterion hands out the block it priced");
    // What the gradient decomposed before #2267: `A` materialized again at this cache.
    let fresh = term
        .materialize_dense_exact_a_geometry(&rho, target.view(), &cache)
        .expect("#2267: a fresh decomposition at the evaluation's state");

    // Before #2267 the rank-charge dispersion took the admission route.
    let admission_rank_charge = term
        .production_rank_charge_derivative(target.view(), &rho, &loss, &cache, None)
        .expect("#2267: the admission-routed rank-charge derivative");
    let received_rank_charge = term
        .production_rank_charge_derivative(target.view(), &rho, &loss, &cache, Some(&received))
        .expect("#2267: the rank-charge derivative off the evaluation's block");
    assert!(
        received_rank_charge.direct_rho.iter().any(|value| *value != 0.0)
            || received_rank_charge.theta.t.iter().any(|value| *value != 0.0),
        "#2267: the rank-charge derivative must be live on this fixture, else the identity \
         compares zeros"
    );
    for (label, received_values, admission_values) in [
        (
            "direct rho",
            &received_rank_charge.direct_rho,
            &admission_rank_charge.direct_rho,
        ),
        (
            "theta t",
            &received_rank_charge.theta.t,
            &admission_rank_charge.theta.t,
        ),
        (
            "theta beta",
            &received_rank_charge.theta.beta,
            &admission_rank_charge.theta.beta,
        ),
    ] {
        assert_eq!(
            bits(received_values),
            bits(admission_values),
            "#2267: the rank charge's {label} moved with the dispersion's divergence route"
        );
    }

    let received_channels = term
        .dense_exact_a_logdet_channels(
            target.view(),
            &rho,
            &cache,
            &received,
            &received_rank_charge.theta,
        )
        .expect("#2267: the log-determinant channels off the evaluation's block");
    let fresh_channels = term
        .dense_exact_a_logdet_channels(
            target.view(),
            &rho,
            &cache,
            &fresh,
            &admission_rank_charge.theta,
        )
        .expect("#2267: the log-determinant channels off a fresh decomposition");
    assert!(
        received_channels.logdet_trace.iter().any(|value| *value != 0.0)
            && received_channels.theta_adjoint.t.iter().any(|value| *value != 0.0),
        "#2267: the log-determinant trace and Γ must be live on this fixture"
    );
    for (label, received_values, fresh_values) in [
        (
            "log-determinant trace",
            &received_channels.logdet_trace,
            &fresh_channels.logdet_trace,
        ),
        (
            "Γ t",
            &received_channels.theta_adjoint.t,
            &fresh_channels.theta_adjoint.t,
        ),
        (
            "Γ beta",
            &received_channels.theta_adjoint.beta,
            &fresh_channels.theta_adjoint.beta,
        ),
        (
            "stationarity adjoint t",
            &received_channels.stationarity_adjoint.t,
            &fresh_channels.stationarity_adjoint.t,
        ),
        (
            "stationarity adjoint beta",
            &received_channels.stationarity_adjoint.beta,
            &fresh_channels.stationarity_adjoint.beta,
        ),
    ] {
        assert_eq!(
            bits(received_values),
            bits(fresh_values),
            "#2267: the {label} read off the evaluation's block differs from a fresh decomposition"
        );
    }

    let lambda_smooth = rho
        .lambda_smooth_vec()
        .expect("#2267: the fixture's smoothing strengths are finite");
    let solver = term
        .outer_gradient_arrow_solver(&cache, &lambda_smooth)
        .expect("#2267: the evaluation's outer solver factors");
    let components = |geometry: &DenseExactAGeometry| {
        term.analytic_outer_rho_gradient_components_with_bundle(
            target.view(),
            &rho,
            &loss,
            &cache,
            &solver,
            None,
            None,
            Some(geometry),
        )
        .expect("#2267: the dense gradient components")
    };
    let from_received = components(&received);
    let from_fresh = components(&fresh);
    for (label, received_values, fresh_values) in [
        ("explicit", &from_received.explicit, &from_fresh.explicit),
        (
            "log-determinant trace",
            &from_received.logdet_trace,
            &from_fresh.logdet_trace,
        ),
        ("occam", &from_received.occam, &from_fresh.occam),
        (
            "implicit response",
            &from_received.third_order_correction,
            &from_fresh.third_order_correction,
        ),
    ] {
        assert_eq!(
            bits(received_values),
            bits(fresh_values),
            "#2267: the gradient's {label} read off the evaluation's block differs from a fresh \
             decomposition"
        );
    }
}
