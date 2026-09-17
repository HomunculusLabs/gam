//! #2935 — the κ entry of the outer ρ-gradient is the derivative of the criterion
//! each route reports.
//!
//! A constant-curvature atom's criterion moves with its raw sectional curvature κ
//! only through the penalty Gram `S(κ)`: the penalty energy `½λ<B, S B>`, the
//! observed information `½log|A|`, the rank charge's basis EDF `tr(G(G+λS)⁻¹)`, the
//! smoothing-prior normalizer `½r·log|S|_+`, and the fitted state that re-converges
//! against all of them. Each route's analytic κ entry must match a central
//! difference of that same route's `cost`, re-converged at both endpoints by one
//! objective that holds its declared collapse-prevention gates (#2933 F05).
#![cfg(test)]
use super::*;
use ndarray::{Array1, Array2};

const KAPPA: f64 = 0.3;

/// One constant-curvature tangent-chart atom observed with a small deterministic
/// residual, so the exact observed information carries residual curvature.
fn curvature_fixture() -> (SaeManifoldTerm, Array2<f64>, SaeManifoldRho) {
    let n = 24usize;
    let p = 3usize;
    let coords = Array2::from_shape_fn((n, 2), |(row, axis)| {
        let angle = std::f64::consts::TAU * 0.618_033_988_75 * row as f64;
        let radius = 0.15 + 0.45 * (row as f64 + 0.5) / n as f64;
        if axis == 0 {
            radius * angle.cos()
        } else {
            radius * angle.sin()
        }
    });
    let plan = SaeAtomGeometryPlan::new(
        SaeAtomBasisKind::Poincare,
        2,
        SaeBasisResolution::Polynomial {
            degree: SAE_EUCLIDEAN_PATCH_MAX_DEGREE,
        },
        SaeReferenceMetricPlan::ConstantCurvatureChart {
            kappa: KAPPA,
            reference_coords: coords.clone(),
        },
    )
    .expect("constant-curvature plan");
    let bundle = plan
        .evaluate_bundle(coords.view())
        .expect("reference rows lie inside the chart");
    let m = bundle.basis_values.ncols();
    let decoder = Array2::from_shape_fn((m, p), |(basis_col, out_col)| {
        0.4 * ((1 + basis_col + 3 * out_col) as f64).sin()
    });
    let mut target = bundle.basis_values.dot(&decoder);
    for ((row, col), value) in target.indexed_iter_mut() {
        *value += 0.02 * ((7 * row + 3 * col) as f64).sin();
    }
    let atom = SaeManifoldAtom::new_with_provided_function_gram(
        "curvature",
        SaeAtomBasisKind::Poincare,
        2,
        bundle.basis_values,
        bundle.basis_jacobian,
        decoder,
        bundle.reference_penalty,
    )
    .expect("atom from the plan's own bundle")
    .with_basis_second_jet(bundle.evaluator)
    .with_geometry_plan(plan)
    .expect("installed Gram is the plan's Gram");
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        Array2::<f64>::zeros((n, 1)),
        vec![coords],
        vec![LatentManifold::Euclidean],
        AssignmentMode::softmax(1.0),
    )
    .expect("one coordinate block for one atom");
    let term = SaeManifoldTerm::new(vec![atom], assignment).expect("single-atom term");
    let rho = SaeManifoldRho::new(-1.0, -1.0, vec![Array1::from_elem(2, -1.0)]);
    (term, target, rho)
}

/// Install `S(κ)` and `∂S/∂κ` on the objective's atom, as the production outer
/// walk does before it evaluates a trial curvature.
fn install_curvature(objective: &mut SaeManifoldOuterObjective, kappa: f64) {
    let prepared = objective.term.atoms[0]
        .prepare_constant_curvature(kappa)
        .expect("the trial curvature lies inside the chart's domain");
    objective.term.atoms[0].commit_prepared_constant_curvature(prepared);
}

/// The route's own converged `cost` at curvature `kappa`.
fn route_cost_at_curvature(
    objective: &mut SaeManifoldOuterObjective,
    anchor: &SaeManifoldRho,
    direct_logdet_admitted: bool,
    kappa: f64,
) -> f64 {
    let mut endpoint = anchor.clone();
    endpoint.kappa[0] = kappa;
    install_curvature(objective, kappa);
    objective
        .evaluate_outer_criterion_route(&endpoint, direct_logdet_admitted, false)
        .expect("the route prices the criterion at a curvature endpoint")
        .cost
}

fn kappa_gradient_is_the_derivative_of_the_route_value(direct_logdet_admitted: bool, route: &str) {
    let (term, target, rho) = curvature_fixture();
    let mut objective =
        SaeManifoldOuterObjective::new(term, target, None, rho, 40, 0.4, 1.0e-6, 1.0e-6);
    let anchor = objective.baseline_rho.clone();
    let flat = anchor
        .kappa_flat_index(0)
        .expect("the curvature atom owns an outer coordinate");
    assert_eq!(
        anchor.kappa[0].to_bits(),
        KAPPA.to_bits(),
        "the objective seeds κ from the atom's geometry plan"
    );
    install_curvature(&mut objective, KAPPA);
    let evaluation = objective
        .evaluate_outer_criterion_route(&anchor, direct_logdet_admitted, false)
        .expect("the route prices the criterion at the anchor");
    let gradient = objective
        .analytic_gradient_for_outer_evaluation(&anchor, &evaluation)
        .expect("the route differentiates the criterion at the anchor");
    let analytic = gradient[flat];

    let mut central = |step: f64| {
        let plus = route_cost_at_curvature(&mut objective, &anchor, direct_logdet_admitted, KAPPA + step);
        let minus =
            route_cost_at_curvature(&mut objective, &anchor, direct_logdet_admitted, KAPPA - step);
        (plus - minus) / (2.0 * step)
    };
    let step = 1.0e-3_f64;
    let coarse = central(step);
    let fine = central(0.5 * step);
    let richardson = (4.0 * fine - coarse) / 3.0;
    let spread = (coarse - fine).abs();
    println!(
        "[#2935 {route}] cost={:.10e} analytic dV/dκ={analytic:.10e} \
         central(h)={coarse:.10e} central(h/2)={fine:.10e} richardson={richardson:.10e}",
        evaluation.cost
    );
    assert!(
        analytic.is_finite() && richardson.is_finite(),
        "{route}: non-finite κ derivative (analytic {analytic}, difference {richardson})"
    );
    assert!(
        spread <= 1.0e-2 * fine.abs().max(1.0),
        "{route}: the two central differences disagree ({coarse} vs {fine}), so the endpoints \
         straddle a discrete stratum or the inner solve is not converged tightly enough to \
         difference"
    );
    assert!(
        fine.abs() > 1.0e-3,
        "{route}: the criterion must move materially with κ for this check to mean anything \
         (slope {fine})"
    );
    let tolerance = 10.0 * spread + 1.0e-6 * richardson.abs().max(1.0);
    assert!(
        (analytic - richardson).abs() <= tolerance,
        "{route}: analytic κ entry {analytic} is not the derivative of the route's own value \
         ({richardson}; |Δ| = {}, tolerance {tolerance})",
        (analytic - richardson).abs()
    );
}

#[test]
fn dense_route_kappa_gradient_is_the_derivative_of_its_value_2935() {
    kappa_gradient_is_the_derivative_of_the_route_value(true, "dense exact-A");
}

#[test]
fn streaming_route_kappa_gradient_is_the_derivative_of_its_value_2935() {
    kappa_gradient_is_the_derivative_of_the_route_value(false, "streaming");
}
