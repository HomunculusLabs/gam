#![cfg(test)]
//! #2933 F03 — a ThresholdGate fit's outer hypergradient must be the derivative
//! of the criterion its route ranks, on every route that ranks one.
//!
//! Both production value routes price the exact observed information `½log|A|`.
//! The dense direct-logdet route used to exclude `ThresholdGate` from its exact-A
//! derivative channels, so with no evidence bundle it contracted the majorizer
//! `B`'s selected inverse and differentiated `∂B` while the value ranked `A`.
//! The gap `½·d/dρ log|I + B⁻¹ΔC|` is not a tolerance question.
//!
//! The arbiter here is the COMPLETE reconverged scalar criterion: every
//! finite-difference endpoint re-runs the inner fit from the same starting state
//! through the production entry point `evaluate_outer_criterion_route`, so no
//! derivative routine is compared against another that shares its inverse.
//!
//! Every endpoint declares the collapse-prevention gates the central root adopted
//! (#2933 F05). An objective holds one gate set for its whole hyperparameter solve,
//! and its analytic gradient differentiates with that set held. An endpoint that
//! chose its own gates at its own root would add `L_w·W_θ·θ̂_ρ`, a motion no
//! objective's value contains.

use super::tests_sparse_curvature_operator_2500::threshold_gate_tiny_fixture;
use super::*;
use ndarray::{Array1, Array2};

const INNER_MAX_ITER: usize = 40;

fn outer_objective(
    term: SaeManifoldTerm,
    target: &Array2<f64>,
    rho: &SaeManifoldRho,
) -> SaeManifoldOuterObjective {
    SaeManifoldOuterObjective::new(
        term,
        target.clone(),
        None,
        rho.clone(),
        INNER_MAX_ITER,
        0.4,
        1.0e-6,
        1.0e-6,
    )
}

#[test]
fn threshold_gate_route_gradients_differentiate_the_reconverged_criterion_2933() {
    let mut failures = Vec::new();
    let mut compared = 0usize;
    for straddle in [false, true] {
        let (term, target, rho) = threshold_gate_tiny_fixture(straddle);
        for (route, direct_logdet_admitted) in [("dense", true), ("streaming", false)] {
            let mut objective = outer_objective(term.clone(), &target, &rho);
            let flat = objective.baseline_rho.to_flat();
            let at = objective
                .baseline_rho
                .from_flat(flat.view())
                .expect("#2933 F03: the objective owns its typed rho layout");
            let evaluation =
                match objective.evaluate_outer_criterion_route(&at, direct_logdet_admitted, false) {
                    Ok(evaluation) => evaluation,
                    Err(err) => {
                        failures.push(format!("straddle={straddle} route={route}: value refused: {err}"));
                        continue;
                    }
                };
            let gradient = match objective.analytic_gradient_for_outer_evaluation(&at, &evaluation) {
                Ok(gradient) => gradient,
                Err(err) => {
                    failures.push(format!("straddle={straddle} route={route}: gradient refused: {err}"));
                    continue;
                }
            };
            println!(
                "[#2933 F03] straddle={straddle} route={route} cost={:.12e} ‖g‖∞={:.6e}",
                evaluation.cost,
                gradient.iter().fold(0.0_f64, |acc, value| acc.max(value.abs())),
            );
            // The set the central root adopted; every endpoint prices the same objective.
            let gates = objective.term.collapse_prevention_gates();
            let reconverged_cost = |flat: &Array1<f64>| -> Result<f64, String> {
                let mut endpoint_term = term.clone();
                endpoint_term.declare_collapse_prevention_gates(&gates);
                let mut endpoint = outer_objective(endpoint_term, &target, &rho);
                let at = endpoint.baseline_rho.from_flat(flat.view())?;
                endpoint
                    .evaluate_outer_criterion_route(&at, direct_logdet_admitted, false)
                    .map(|evaluation| evaluation.cost)
                    .map_err(|err| err.to_string())
            };
            for coordinate in 0..flat.len() {
                let mut differences = Vec::new();
                for step in [1.0e-3, 2.5e-4] {
                    let mut plus = flat.clone();
                    plus[coordinate] += step;
                    let mut minus = flat.clone();
                    minus[coordinate] -= step;
                    match (reconverged_cost(&plus), reconverged_cost(&minus)) {
                        (Ok(up), Ok(down)) => differences.push((step, (up - down) / (2.0 * step))),
                        (Err(err), _) | (_, Err(err)) => println!(
                            "[#2933 F03] straddle={straddle} route={route} coord={coordinate} \
                             h={step:.1e}: endpoint refused: {err}"
                        ),
                    }
                }
                for &(step, fd) in &differences {
                    println!(
                        "[#2933 F03] straddle={straddle} route={route} coord={coordinate} \
                         h={step:.1e} fd={fd:.10e} analytic={:.10e} gap={:.3e}",
                        gradient[coordinate],
                        (gradient[coordinate] - fd).abs(),
                    );
                }
                let Some(&(_, fd)) = differences.last() else {
                    failures.push(format!(
                        "straddle={straddle} route={route} coord={coordinate}: no finite difference"
                    ));
                    continue;
                };
                let gap = (gradient[coordinate] - fd).abs();
                if gap > 1.0e-4 * (1.0 + fd.abs()) {
                    failures.push(format!(
                        "straddle={straddle} route={route} coord={coordinate}: analytic \
                         {:.10e} vs reconverged fd {fd:.10e} (gap {gap:.3e})",
                        gradient[coordinate]
                    ));
                }
                compared += 1;
            }
        }
    }
    println!("[#2933 F03] compared {compared} coordinates; {} failures", failures.len());
    assert!(
        failures.is_empty(),
        "#2933 F03: a ThresholdGate route returned a hypergradient that is not the \
         derivative of the criterion it ranks:\n{}",
        failures.join("\n")
    );
}
