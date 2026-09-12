#![cfg(test)]
//! #2629 scope item 2 — settle the custom-family engine's row of the objective
//! table by MEASUREMENT.
//!
//! #2629 lists seven outer-objective families and asks which of them carry the
//! soft ρ-guard barrier that #2545 taught the certificate to subtract. Three of
//! those rows — `gamlss mean-wiggle`, `spatial-adaptive`, and `custom family` —
//! are the same evaluator seen from three call sites:
//! [`evaluate_custom_family_joint_hyper_owned`]. So one measurement settles
//! three rows, and it is this one.
//!
//! The issue's own evidence for those rows was a call-graph argument:
//! `RemlState::build_prior` is the only site that adds
//! `soft_rho_guard_prior_atom`'s gradient to a criterion, its only callers are
//! `RemlState` methods, and this engine holds no `RemlState`. That argument is
//! correct, and it is still an argument about code rather than about numbers.
//! The issue said what would settle it — *"evaluate each path's ρ-gradient at a
//! saturated ρ and look for the 1.3333e-7 floor"* — and this file reads that
//! number from the engine's own published outer gradient.
//!
//! What a floor here would have meant: every railed coordinate of every gamlss,
//! spatial-adaptive, and custom-family fit carrying a standing
//! `|Pg| ≥ w·a = 1.3333e-7` that no amount of convergence clears, and three more
//! objectives owing the seam a publication.
//!
//! [`evaluate_custom_family_joint_hyper_owned`]: crate::psi_hyper::evaluate_custom_family_joint_hyper_owned

use super::*;
use crate::tests::{OneBlockGaussianFamily, test_design_hyper_layout};
use ndarray::{Array1, Array2};
use gam_problem::EvalMode;

/// The saturated-ρ ladder #2450 established and #2545/#2629 measure on. `ρ ≥ 21`
/// is past the REML part's own tail on the reference fixture, and `RHO_BOUND = 30`
/// is the deepest point the box admits. Four rungs make the pencil's constancy
/// three independent ratios, an observation rather than a definition.
const SATURATED_RHO_LADDER: [f64; 4] = [21.0, 24.0, 27.0, 30.0];

/// A Gaussian one-block fixture with a real λ→∞ face: an unpenalized intercept
/// plus three penalized basis columns, so sending ρ to the box bound drives the
/// fit onto the penalty's null space rather than onto nothing.
///
/// Deliberately NOT the degenerate `[[1.0]]` design most of this crate's
/// fixtures use. A 1×1 problem has no face to decay along, so its ρ-gradient
/// would be identically zero at every rung — which the classifier would answer
/// correctly (`AbsentBelowTheFloor`) but which would prove nothing, since a
/// criterion that carried the barrier on THAT fixture would still show the
/// floor and a criterion that did not would show zero. The whole point is to
/// measure a fixture where a floor would be visible.
fn gaussian_face_fixture() -> (OneBlockGaussianFamily, Vec<ParameterBlockSpec>) {
    const N: usize = 64;
    const P: usize = 4;
    let mut design = Array2::<f64>::zeros((N, P));
    let mut y = Array1::<f64>::zeros(N);
    for i in 0..N {
        let t = (i as f64 + 0.5) / N as f64;
        let x = -1.5 + 3.0 * t;
        design[[i, 0]] = 1.0;
        design[[i, 1]] = x;
        design[[i, 2]] = x * x;
        design[[i, 3]] = (2.1 * x).sin();
        // A signal with real curvature, so the penalized columns carry weight
        // and the criterion's tail constant is not numerically zero.
        y[i] = 0.4 + 0.8 * x - 0.3 * x * x + 0.5 * (2.1 * x).sin();
    }
    // Penalize everything but the intercept: a rank-3 penalty with a 1-D null
    // space, the standard smooth-term shape.
    let mut penalty = Array2::<f64>::zeros((P, P));
    for j in 1..P {
        penalty[[j, j]] = 1.0;
    }
    let specs = vec![ParameterBlockSpec {
        name: "gaussian_face".to_string(),
        design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(design)),
        offset: Array1::zeros(N),
        penalties: vec![PenaltyMatrix::Dense(penalty)],
        nullspace_dims: vec![1],
        initial_log_lambdas: Array1::zeros(1),
        initial_beta: Some(Array1::zeros(P)),
        gauge_priority: 100,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    }];
    (OneBlockGaussianFamily { y }, specs)
}

/// Build the saturated-ρ ladder from the shared custom-family evaluator: one
/// `(ρ, g)` rung per probe, carrying the SIGNED outer ρ-gradient exactly as the
/// engine reports it, with nothing subtracted.
fn custom_family_rho_ladder(probes: &[f64]) -> Vec<(f64, f64)> {
    let (family, specs) = gaussian_face_fixture();
    let options = BlockwiseFitOptions {
        use_remlobjective: true,
        use_outer_hessian: false,
        compute_covariance: false,
        ..BlockwiseFitOptions::default()
    };
    let hyper_layout = test_design_hyper_layout(vec![vec![]]);
    probes
        .iter()
        .map(|&probe| {
            let rho = Array1::from_elem(1, probe);
            let owned = crate::psi_hyper::evaluate_custom_family_joint_hyper_owned(
                &family,
                &specs,
                &options,
                &rho,
                &hyper_layout,
                None,
                EvalMode::ValueAndGradient,
            )
            .unwrap_or_else(|e| {
                panic!("the custom-family engine must evaluate at rho={probe}: {e}")
            });
            assert!(
                owned.result.inner_converged,
                "rho={probe}: the outer gradient is an ENVELOPE derivative and is \
                 only valid at a stationary beta-hat. A non-converged rung would \
                 make the ladder a reading of the inner solver, not of the \
                 criterion"
            );
            assert_eq!(
                owned.result.gradient.len(),
                1,
                "this fixture declares exactly one rho coordinate"
            );
            (probe, owned.result.gradient[0])
        })
        .collect()
}

/// The face pencil `ĉ_j = g_j·e^{ρ_j}` of a ladder, as `(spread, mean)` with
/// `spread = (max|ĉ| − min|ĉ|)/max|ĉ|`. `None` when the pencil is not a finite
/// single-signed run, so no face constant exists to be spread.
///
/// A bare `c·e^{−ρ}` face makes the pencil one constant. A floor `F` added at
/// every rung moves rung `j` by `F·e^{ρ_j}`, which grows by `e^9 ≈ 8100` across
/// the ladder, so no floor can sit inside a small spread.
fn face_pencil(ladder: &[(f64, f64)]) -> Option<(f64, f64)> {
    let pencil: Vec<f64> = ladder.iter().map(|&(rho, g)| g * rho.exp()).collect();
    let finite = pencil.iter().all(|c| c.is_finite());
    let single_signed = pencil.iter().all(|c| *c > 0.0) || pencil.iter().all(|c| *c < 0.0);
    if pencil.is_empty() || !finite || !single_signed {
        return None;
    }
    let max = pencil.iter().fold(0.0f64, |acc, c| acc.max(c.abs()));
    let min = pencil.iter().fold(f64::MAX, |acc, c| acc.min(c.abs()));
    let mean = pencil.iter().sum::<f64>() / pencil.len() as f64;
    Some(((max - min) / max, mean))
}

fn render(ladder: &[(f64, f64)]) -> String {
    ladder
        .iter()
        .map(|&(rho, g)| format!("(rho={rho:.0}, g={g:+.6e})"))
        .collect::<Vec<_>>()
        .join(" ")
}

/// **The measurement.** The shared custom-family outer evaluator — the route
/// `gamlss mean-wiggle`, `spatial-adaptive`, and `custom family` all take — does
/// NOT carry the soft ρ-guard barrier, so `None` is the correct answer for all
/// three at [`OuterObjective::soft_rho_guard_gradient`].
///
/// The engine's outer gradient on the saturated ladder IS a bare face: its pencil
/// is one constant to `1e-4`. Measured c = 2.230365e3 with a pencil spread of
/// 2.08e-6 across three three-e-fold steps, 1.69e-6 at ρ=21 down to 2.09e-10 at
/// ρ=30: six significant figures of `c·e^{−ρ}`. The barrier would add
/// `w·a·tanh(a·ρ) ≈ 1.3333e-7` at every rung and move the ρ=30 pencil by
/// `1.3333e-7·e^30 ≈ 1.42e6` against that `c`, a spread near one.
///
/// The engine's own construction agrees, and is worth naming since it is the
/// mechanism behind the number:
/// `evaluate_custom_family_joint_hyper_owned` passes `gam_problem::RhoPrior::Flat`
/// into `evaluate_custom_family_hyper_internal` unconditionally, and the
/// ρ-prior machinery it reaches (`psi_hyper`'s `has_configured_rho_prior`) is the
/// CONFIGURED prior only. The soft guard has no path into it.
///
/// [`OuterObjective::soft_rho_guard_gradient`]: gam_solve::rho_optimizer::OuterObjective::soft_rho_guard_gradient
#[test]
fn the_custom_family_engine_carries_no_soft_rho_guard_floor_2629() {
    let ladder = custom_family_rho_ladder(&SATURATED_RHO_LADDER);
    let rendered = render(&ladder);
    let pencil = face_pencil(&ladder);
    eprintln!("[#2629-table] custom-family engine: pencil (spread, c) = {pencil:?} | {rendered}");
    let Some((spread, constant)) = pencil else {
        panic!(
            "this fixture's face is LIVE across the ladder (1.69e-6 down to 2.09e-10), \
             so the bare pencil c = g*e^rho must be a finite single-signed run. A sign \
             change or a zero rung means the gradient is not the face. Ladder: {rendered}"
        );
    };
    assert!(
        spread <= 1.0e-4,
        "#2629's table lists `gamlss mean-wiggle`, `spatial-adaptive` and `custom \
         family` as carrying no soft rho-guard barrier. The bare pencil c = g*e^rho \
         must be CONSTANT to say the gradient IS the face and nothing else is hiding \
         in it; got spread {spread:.3e} on c={constant:.6e}. Ladder: {rendered}"
    );

    // The control that keeps the gate above from passing for the wrong reason: a
    // ladder whose pencil could not show a floor cannot certify that one is absent.
    // Add a constant floor equal to the deepest rung's own gradient, about 600 times
    // smaller than the barrier's w·a, to the same measurements, and require the
    // pencil test to refuse it.
    let floor = ladder.last().expect("the ladder reaches RHO_BOUND").1.abs();
    let floored: Vec<(f64, f64)> = ladder.iter().map(|&(rho, g)| (rho, g + floor)).collect();
    let floored_pencil = face_pencil(&floored);
    eprintln!(
        "[#2629-control] custom-family engine + floor {floor:.6e}: \
         pencil (spread, c) = {floored_pencil:?}"
    );
    assert!(
        floored_pencil.is_none_or(|(floored_spread, _)| floored_spread > 1.0e-4),
        "adding a constant floor of {floor:.6e} to every rung must break the pencil's \
         constancy. If it does not, this ladder is blind to a floor far smaller than \
         the barrier and cannot certify the barrier's absence. Got {floored_pencil:?} \
         on {}",
        render(&floored)
    );
}
