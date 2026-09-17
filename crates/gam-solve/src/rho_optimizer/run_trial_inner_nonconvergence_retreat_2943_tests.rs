//! gam#2943: a trial point whose inner solve did not certify is valley escape,
//! not the end of a fit.
//!
//! The objective refuses every trial past θ = 1.5 with the refusal the
//! custom-family producer builds, with its cycle budget and carrying block
//! recorded, and its certified optimum at θ = 1 lies outside that region. The
//! outer search must step away from the refused trials and certify, and the
//! refusal must never become the fit-ending variant, which only the boundary
//! that hands a fit to its caller mints.
//!
//! The cost is log cosh(θ − 1), whose curvature peaks at the optimum and
//! vanishes far from it, so a secant model built from the far seed
//! underestimates the curvature and steps past the optimum into the refused
//! region. A quadratic cost cannot test retreat: its secant model is exact after
//! one step and lands on the optimum without ever trying a refused point.

use super::*;
use ndarray::{Array1, array};
use std::sync::{Arc, Mutex};

fn uncertified_trial() -> EstimationError {
    EstimationError::CustomFamily(gam_problem::CustomFamilyError::InnerSolveNotConverged {
        cycles: 8,
        terminal: None,
        kkt_residual: Some(1.081e3),
        kkt_tol: Some(5.352e-2),
        theta_dim: 1,
        rho_dim: 1,
        psi_dim: 0,
        cycle_budget: Some(8),
        carrying_block: Some("slope_surface".to_string()),
    })
}

const REFUSED_ABOVE: f64 = 1.5;

/// Records the trial, then refuses it past `REFUSED_ABOVE` or returns
/// log cosh(θ − 1) and its derivative.
fn evaluate(trail: &Mutex<Vec<f64>>, theta: &Array1<f64>) -> Result<(f64, f64), EstimationError> {
    trail
        .lock()
        .expect("only the fixture's own closures lock the trail, and none panics holding it")
        .push(theta[0]);
    if theta[0] > REFUSED_ABOVE {
        return Err(uncertified_trial());
    }
    let offset = theta[0] - 1.0;
    Ok((offset.cosh().ln(), offset.tanh()))
}

#[test]
fn an_uncertified_trial_retreats_and_the_search_certifies_elsewhere_2943() {
    assert!(uncertified_trial().is_trial_point_infeasible());
    let trail = Arc::new(Mutex::new(Vec::new()));
    let problem = OuterProblem::new(1)
        .with_gradient(Derivative::Analytic)
        .with_hessian(DeclaredHessianForm::Unavailable)
        .with_initial_rho(array![-2.0])
        .with_max_iter(60);
    let mut objective = problem.build_objective(
        (),
        {
            let trail = Arc::clone(&trail);
            move |_: &mut (), theta: &Array1<f64>| evaluate(&trail, theta).map(|(cost, _)| cost)
        },
        {
            let trail = Arc::clone(&trail);
            move |_: &mut (), theta: &Array1<f64>| {
                let (cost, slope) = evaluate(&trail, theta)?;
                Ok(OuterEval {
                    cost,
                    gradient: array![slope],
                    hessian: HessianValue::Unavailable,
                    inner_beta_hint: None,
                })
            }
        },
        None::<fn(&mut ())>,
        None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
    );

    let result = problem
        .run(&mut objective, "uncertified trial retreat #2943")
        .unwrap_or_else(|error| {
            panic!("the search must step away from uncertified trials and certify, got {error}")
        });
    let trials = trail
        .lock()
        .expect("the search has returned, so no closure holds the trail");
    assert!(
        trials.iter().any(|&theta| theta > REFUSED_ABOVE),
        "the fixture must reach the refused region, or it proves nothing about retreat; trials {trials:?}"
    );
    assert!(
        (result.rho[0] - 1.0).abs() < 1.0e-3,
        "the search must certify the optimum at θ = 1 outside the refused region, got {:?}",
        result.rho
    );
}
