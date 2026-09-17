#![cfg(test)]
//! #2933 F36 — the joint fitted-response divergence `tr(∂f̂/∂y)` of the SAE
//! reconstruction against direct re-solves of the perturbed inner problem.
//!
//! The divergence of a penalized fit is the trace of its response Jacobian, and
//! the oracle here measures that trace directly. It perturbs one target entry by
//! `±h`, drives the penalized stationarity `∇L(θ; y ± h·e) = 0` from the fitted
//! state to its roundoff plateau with exact-A Newton steps, and differences the
//! fitted value of that entry. The root of that loop is fixed by the assembled
//! gradient alone; the exact stationarity solve only chooses the path to it, so
//! the oracle does not score the operator against itself. The production inner
//! driver stops inside a `1e-5` relative KKT band, which is coarser than the
//! response being differenced, so the re-solve runs to roundoff instead.
//! Collapse-prevention gates stay frozen at the fitted state: the same basin and
//! the same objective the divergence describes.
//!
//! The per-axis completion this replaced summed scalar fractions
//! `htt/(htt+c+V'') − htt/(htt+V'')` from diagonal curvatures, so an off-diagonal
//! residual curvature `A = [[1, c], [c, 1]]` returned zero where the trace
//! correction is `2/(1−c²) − 2`. The fixtures carry what that routine could not
//! see: a two-dimensional torus chart (mixed coordinate curvature), two atoms on
//! the same rows (cross-atom coupling), free gate logits, an atom with ARD
//! disabled, and ARD rows on the concave side of a periodic axis.

use super::*;
use crate::basis::TorusHarmonicEvaluator;
use crate::manifold::arrow_solver::SaeArrowVector;
use gam_terms::latent::LatentManifold;
use ndarray::{Array1, Array2, ArrayView2, array};
use std::sync::Arc;

/// Largest number of exact-A Newton steps one re-solve may take. From a state
/// one `h` away from a root, quadratic convergence reaches roundoff in a few.
const ROOT_POLISH_STEPS: usize = 40;

/// Central-difference step on a target entry. The targets are of order one, so
/// the truncation error `h²·f'''/6` sits far below the tolerance below.
const FD_STEP: f64 = 1.0e-4;

/// Relative agreement required between the divergence and the re-solved
/// response trace. The truncation error at `FD_STEP` and the roundoff of a root
/// driven to its plateau are both orders of magnitude smaller.
const RELATIVE_TOLERANCE: f64 = 1.0e-4;

/// A root is admitted only once its gradient has reached the arithmetic floor
/// of an order-one objective on a few dozen scalar observations.
const ROOT_GRADIENT_CEILING: f64 = 1.0e-10;

fn gradient(system: &ArrowSchurSystem) -> SaeArrowVector {
    SaeArrowVector {
        t: system
            .rows
            .iter()
            .flat_map(|row| row.gt.iter().copied())
            .collect(),
        beta: system.gb.clone(),
    }
}

/// Drive `∇L(θ; target) = 0` from the term's current state by exact-A Newton
/// steps until the gradient norm stops halving, and return the undamped
/// evidence factorization at that root with the root's gradient norm.
fn polish_to_root(
    term: &mut SaeManifoldTerm,
    target: ArrayView2<'_, f64>,
    rho: &SaeManifoldRho,
) -> (ArrowFactorCache, f64) {
    let options = term.evidence_factor_options();
    let mut previous = f64::INFINITY;
    for _ in 0..ROOT_POLISH_STEPS {
        let system = term
            .assemble_arrow_schur(target, rho, None)
            .expect("the arrow system assembles at every re-solve iterate");
        let g = gradient(&system);
        let norm = (g.t.dot(&g.t) + g.beta.dot(&g.beta)).sqrt();
        let (_, _, cache) = solve_arrow_newton_step_with_options(&system, 0.0, 0.0, &options)
            .expect("the majorizer factors undamped at every re-solve iterate");
        if !(norm < 0.5 * previous) {
            return (cache, norm);
        }
        let step = term
            .solve_exact_stationarity(rho, target, &cache, &g)
            .expect("the exact stationarity pseudoinverse solves at every re-solve iterate");
        term.apply_newton_step((-&step.t).view(), (-&step.beta).view(), 1.0)
            .expect("the exact Newton step applies to the term state");
        previous = norm;
    }
    panic!("the exact-A Newton re-solve did not reach its roundoff plateau in {ROOT_POLISH_STEPS} steps");
}

/// `Σ_{i,c} ∂f̂_ic/∂y_ic` by central differences of fits re-solved to their roots.
fn resolved_divergence(
    base: &SaeManifoldTerm,
    target: &Array2<f64>,
    rho: &SaeManifoldRho,
) -> f64 {
    let (n, p) = target.dim();
    let mut divergence = 0.0_f64;
    for row in 0..n {
        for col in 0..p {
            let mut fitted = [0.0_f64; 2];
            for (side, sign) in [1.0_f64, -1.0].into_iter().enumerate() {
                let mut term = base.clone();
                let mut perturbed = target.clone();
                perturbed[[row, col]] += sign * FD_STEP;
                let (_, norm) = polish_to_root(&mut term, perturbed.view(), rho);
                assert!(
                    norm <= ROOT_GRADIENT_CEILING,
                    "the re-solve at entry ({row}, {col}), side {sign} stalled at ‖g‖={norm:.3e}"
                );
                let residual = term
                    .reconstruction_residual(perturbed.view(), rho)
                    .expect("the re-solved fit has a residual");
                fitted[side] = residual[[row, col]] + perturbed[[row, col]];
            }
            divergence += (fitted[0] - fitted[1]) / (2.0 * FD_STEP);
        }
    }
    divergence
}

fn assert_agrees(label: &str, value: f64, resolved: f64) {
    let gap = (value - resolved).abs();
    assert!(
        gap <= RELATIVE_TOLERANCE * resolved.abs().max(1.0),
        "{label}: {value:.9e} against the re-solved response trace {resolved:.9e} (gap {gap:.3e})"
    );
}

/// The softmax witness: two circle atoms on every row, gated by one softmax, so
/// the logits are free and couple the atoms through the simplex.
fn softmax_two_circle_state() -> (SaeManifoldTerm, Array2<f64>, SaeManifoldRho, ArrowFactorCache) {
    let (mut term, target, mut rho) = crate::manifold::tests::gamma_fd_tiny_fixture();
    // The moderate-penalty basin where this fixture's exact observed information
    // is positive definite and its residual curvature is live.
    rho.log_lambda_sparse = 0.0;
    for value in rho.log_lambda_smooth.iter_mut() {
        *value = -1.0;
    }
    for axis in rho.log_ard.iter_mut() {
        for value in axis.iter_mut() {
            *value = -1.0;
        }
    }
    term.penalized_quasi_laplace_criterion_with_cache(
        target.view(),
        &rho,
        None,
        40,
        0.4,
        1.0e-6,
        1.0e-6,
    )
    .expect("the softmax fixture converges in its moderate-penalty basin");
    term.streaming_gates_frozen = true;
    let (cache, norm) = polish_to_root(&mut term, target.view(), &rho);
    assert!(
        norm <= ROOT_GRADIENT_CEILING,
        "the softmax fixture's root stalled at ‖g‖={norm:.3e}"
    );
    (term, target, rho, cache)
}

/// The ordered Beta--Bernoulli witness: a torus atom (two coupled chart axes,
/// ARD on both, rows on both sides of each axis) and a circle atom with ARD
/// disabled, on the same rows with independent free logits.
fn obb_torus_and_circle_state() -> (SaeManifoldTerm, Array2<f64>, SaeManifoldRho, ArrowFactorCache) {
    let n = 12usize;
    let p = 3usize;
    let torus = Arc::new(
        TorusHarmonicEvaluator::new(2, 1).expect("one harmonic per axis is a valid torus basis"),
    );
    let circle = Arc::new(
        PeriodicHarmonicEvaluator::new(3).expect("an odd harmonic count is a valid periodic basis"),
    );
    let torus_coords = Array2::<f64>::from_shape_fn((n, 2), |(row, axis)| {
        let (rate, offset) = [(0.137, 0.05), (0.241, 0.43)][axis];
        (row as f64 * rate + offset).rem_euclid(1.0)
    });
    let circle_coords =
        Array2::<f64>::from_shape_fn((n, 1), |(row, _)| (0.11 + row as f64 / n as f64).rem_euclid(1.0));
    let (torus_phi, torus_jet) = torus
        .evaluate(torus_coords.view())
        .expect("the torus coordinates are wrapped into the unit period");
    let (circle_phi, circle_jet) = circle
        .evaluate(circle_coords.view())
        .expect("the circle coordinates are wrapped into the unit period");
    let torus_width = torus_phi.ncols();
    let circle_width = circle_phi.ncols();
    let torus_decoder = Array2::<f64>::from_shape_fn((torus_width, p), |(basis, out)| {
        0.35 * ((3 * basis + out) as f64 * 0.7 + 0.2).sin()
    });
    let circle_decoder = Array2::<f64>::from_shape_fn((circle_width, p), |(basis, out)| {
        0.3 * ((2 * basis + 3 * out) as f64 * 0.9 + 0.5).cos()
    });
    let mut target = torus_phi.dot(&torus_decoder) * 0.8 + circle_phi.dot(&circle_decoder) * 0.6;
    for row in 0..n {
        for out in 0..p {
            target[[row, out]] += 0.2 * (1.7 * row as f64 + 2.3 * out as f64).sin();
        }
    }
    let torus_atom = SaeManifoldAtom::new_with_provided_function_gram(
        "torus".to_string(),
        SaeAtomBasisKind::Torus,
        2,
        torus_phi,
        torus_jet,
        torus_decoder,
        Array2::<f64>::eye(torus_width),
    )
    .expect("torus atom: basis width, latent dimension and decoder shape agree")
    .with_basis_second_jet(torus);
    let circle_atom = SaeManifoldAtom::new_with_provided_function_gram(
        "circle".to_string(),
        SaeAtomBasisKind::Periodic,
        1,
        circle_phi,
        circle_jet,
        circle_decoder,
        Array2::<f64>::eye(circle_width),
    )
    .expect("circle atom: basis width, latent dimension and decoder shape agree")
    .with_basis_second_jet(circle);
    let logits = Array2::<f64>::from_shape_fn((n, 2), |(row, atom)| {
        if atom == 0 { 1.2 } else { 0.4 + 0.05 * row as f64 }
    });
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        logits,
        vec![torus_coords, circle_coords],
        vec![
            LatentManifold::Product(vec![
                LatentManifold::Circle { period: 1.0 },
                LatentManifold::Circle { period: 1.0 },
            ]),
            LatentManifold::Circle { period: 1.0 },
        ],
        AssignmentMode::ordered_beta_bernoulli(0.7, 1.0, false),
    )
    .expect("assignment: one logit column and one coordinate block per atom");
    let mut term = SaeManifoldTerm::new(vec![torus_atom, circle_atom], assignment)
        .expect("term: every atom's basis width matches its assignment block");
    // ARD on both torus axes at a precision comparable to the data curvature, so
    // the rows on the concave side of an axis carry a live negative prior
    // curvature; ARD disabled on the circle atom.
    let mut rho = SaeManifoldRho::new(0.0, -1.0, vec![array![1.0, 1.0], Array1::<f64>::zeros(0)]);
    term.run_joint_fit_arrow_schur(target.view(), &mut rho, None, 80, 1.0, 1.0e-7, 1.0e-7)
        .expect("the torus and circle fixture fits");
    term.streaming_gates_frozen = true;
    let (cache, norm) = polish_to_root(&mut term, target.view(), &rho);
    assert!(
        norm <= ROOT_GRADIENT_CEILING,
        "the torus and circle fixture's root stalled at ‖g‖={norm:.3e}"
    );
    let concave_rows = (0..n)
        .filter(|&row| {
            (0..2).any(|axis| (std::f64::consts::TAU * term.assignment.coords[0].row(row)[axis]).cos() < 0.0)
        })
        .count();
    assert!(
        concave_rows > 0,
        "the fixture must place ARD rows on the concave side of a periodic axis"
    );
    (term, target, rho, cache)
}

#[test]
fn softmax_divergence_matches_the_resolved_response_2933() {
    let (term, target, rho, cache) = softmax_two_circle_state();
    let response = term
        .fitted_response_divergence(target.view(), &rho, &cache)
        .expect("the softmax fixture admits the exact spectral divergence");
    assert!(
        matches!(response.estimator, FittedResponseDivergenceEstimator::ExactSpectral),
        "a fixture this small is on the exact spectral route, got {:?}",
        response.estimator
    );
    let resolved = resolved_divergence(&term, &target, &rho);
    eprintln!(
        "[#2933 F36 softmax] divergence={:.9e} resolved={resolved:.9e}",
        response.divergence
    );
    assert!(
        resolved > 1.0,
        "the re-solved response trace {resolved} must be material for the comparison to mean anything"
    );
    assert_agrees("softmax exact spectral divergence", response.divergence, resolved);
}

#[test]
fn torus_and_ard_free_divergence_prices_the_dispersion_2933() {
    let (term, target, rho, cache) = obb_torus_and_circle_state();
    let resolved = resolved_divergence(&term, &target, &rho);
    let loss = term
        .loss(target.view(), &rho)
        .expect("the fitted state has a loss");
    let residual = term
        .reconstruction_residual(target.view(), &rho)
        .expect("the fitted state has a residual");
    let dispersion = term
        .reconstruction_dispersion(&loss, &cache, &rho, Some(residual.view()))
        .expect("the fitted state prices a dispersion");
    // With no row metric the raw output noise variance is `RSS/(N·p − edf)`, and
    // the dispersion charges no selection degrees of freedom, so the priced EDF is
    // the within-basin response trace alone.
    let rss: f64 = residual.iter().map(|value| value * value).sum();
    let n_scalar = target.len() as f64;
    let priced_edf = n_scalar - rss / dispersion.raw_output_noise_variance;
    eprintln!("[#2933 F36 torus] priced edf={priced_edf:.9e} resolved={resolved:.9e}");
    assert!(
        resolved > 1.0 && resolved < n_scalar - 1.0,
        "the re-solved response trace {resolved} must be material and leave residual dof"
    );
    assert_agrees("the EDF reconstruction_dispersion prices", priced_edf, resolved);
    let response = term
        .fitted_response_divergence(target.view(), &rho, &cache)
        .expect("the torus fixture admits the exact spectral divergence");
    assert_agrees("torus exact spectral divergence", response.divergence, resolved);
}

#[test]
fn hutchinson_divergence_brackets_the_resolved_response_2933() {
    let (mut term, target, rho, cache) = obb_torus_and_circle_state();
    let resolved = resolved_divergence(&term, &target, &rho);
    // No host memory admits the dense eigensystem, so the matrix-free estimator
    // must carry the divergence.
    term.host_available_bytes = 0;
    let response = term
        .fitted_response_divergence(target.view(), &rho, &cache)
        .expect("the matrix-free estimator solves where the dense route is refused");
    let FittedResponseDivergenceEstimator::Hutchinson {
        probes,
        standard_error,
    } = response.estimator
    else {
        panic!(
            "without an admitted eigensystem the divergence must be the Hutchinson estimate, \
             got {:?}",
            response.estimator
        );
    };
    eprintln!(
        "[#2933 F36 Hutchinson] divergence={:.9e} standard error={standard_error:.3e} \
         probes={probes} resolved={resolved:.9e}",
        response.divergence
    );
    assert!(
        standard_error.is_finite() && standard_error > 0.0 && standard_error < 0.5 * resolved,
        "a Hutchinson estimate with standard error {standard_error} cannot resolve a trace of \
         {resolved}"
    );
    // Four standard errors of a fixed-seed estimate: a deterministic bracket whose
    // width is the estimator's own sampling error, not a tuned band.
    assert!(
        (response.divergence - resolved).abs() <= 4.0 * standard_error,
        "the Hutchinson divergence {} is more than four standard errors ({standard_error:.3e}) \
         from the re-solved response trace {resolved}",
        response.divergence
    );
}
