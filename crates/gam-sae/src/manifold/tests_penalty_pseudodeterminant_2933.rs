//! #2933 F26 — the dense criterion's smoothing-prior normalizer is the full
//! pseudo-determinant `½·Σ_k r_k·log|λ_k S_k|_+`, not only its `log λ` part.
//!
//! The decisive representation check is `S → c·S`, `λ → λ/c`. It leaves the
//! prior precision `λS`, the fitted state and `log|A|` unchanged, so the
//! criterion must not move. With only `½·r·rank(S)·log λ` it moved by
//! `½·Σ_k r_k·rank(S_k)·log c`.
#![cfg(test)]
use super::*;
use crate::manifold::tests_sparse_curvature_operator_2500::threshold_gate_tiny_fixture;
use ndarray::{Array1, Array2};

const RESCALINGS: [f64; 2] = [100.0, 1.0e-3];

/// The same model with every atom's penalty written as `c·S` and its strength as
/// `λ/c`.
fn rescaled_penalty_representation(
    term: &SaeManifoldTerm,
    rho: &SaeManifoldRho,
    c: f64,
) -> (SaeManifoldTerm, SaeManifoldRho) {
    let mut scaled = term.clone();
    for atom in scaled.atoms.iter_mut() {
        let basis_values = atom.basis_values.clone();
        let basis_jacobian = atom.basis_jacobian.clone();
        let decoder = atom.decoder_coefficients().clone();
        let penalty = atom.smooth_penalty() * c;
        atom.install_reparameterized_basis(basis_values, basis_jacobian, decoder, penalty)
            .expect("a positive multiple of a PSD Gram is a valid Gram");
    }
    let mut scaled_rho = rho.clone();
    for log_lambda in scaled_rho.log_lambda_smooth.iter_mut() {
        *log_lambda -= c.ln();
    }
    (scaled, scaled_rho)
}

/// `½·Σ_k r_k·rank(S_k)·log c`, what a normalizer carrying only its `log λ` part
/// moves by under the rescaling. Each tolerance below is asserted far beneath it,
/// so an invariance assertion cannot pass on a fixture where the defect is silent.
fn log_lambda_only_shift(term: &SaeManifoldTerm, c: f64) -> f64 {
    term.atoms
        .iter()
        .map(|atom| {
            let rank = SaeManifoldTerm::symmetric_rank(atom.smooth_penalty())
                .expect("fixture penalty is square");
            0.5 * atom.border_frame_rank() as f64 * rank as f64 * c.ln()
        })
        .sum()
}

/// `½·Σ_k r_k·log|λ_k S_k|_+`, read off a plain eigendecomposition of each
/// precision with the textbook numerical-rank cutoff `m·ε·σ_max`, independent of
/// the REML threshold the production normalizer reads.
fn half_log_pseudodeterminant_of_precisions(term: &SaeManifoldTerm, rho: &SaeManifoldRho) -> f64 {
    let mut acc = 0.0_f64;
    for (atom_idx, atom) in term.atoms.iter().enumerate() {
        let precision = atom.smooth_penalty() * rho.log_lambda_smooth[atom_idx].exp();
        let (eigenvalues, _) = precision.eigh(Side::Lower).expect("symmetric precision");
        let largest = eigenvalues.iter().copied().fold(0.0_f64, f64::max);
        let cutoff = eigenvalues.len() as f64 * f64::EPSILON * largest;
        let log_pseudodeterminant: f64 = eigenvalues
            .iter()
            .filter(|&&value| value > cutoff)
            .map(|value| value.ln())
            .sum();
        acc += 0.5 * atom.border_frame_rank() as f64 * log_pseudodeterminant;
    }
    acc
}

#[test]
fn occam_normalizer_is_the_log_pseudodeterminant_of_the_prior_precision_2933() {
    let (term, _target, rho) = threshold_gate_tiny_fixture(false);
    for c in RESCALINGS {
        let shift = log_lambda_only_shift(&term, c);
        assert!(
            shift.abs() > 1.0,
            "the rescaling by {c} must be able to move a log-λ-only normalizer (shift {shift})"
        );
        let (scaled, scaled_rho) = rescaled_penalty_representation(&term, &rho, c);
        for (label, arm_term, arm_rho) in [("S", &term, &rho), ("c·S", &scaled, &scaled_rho)] {
            let occam = arm_term.reml_occam_term(arm_rho).expect("occam normalizer");
            let expected = half_log_pseudodeterminant_of_precisions(arm_term, arm_rho);
            assert!(
                (occam - expected).abs() <= 1.0e-10 * expected.abs().max(1.0),
                "{label} arm (c = {c}): occam {occam} is not ½·Σ r·log|λS|_+ = {expected}"
            );
        }
    }
}

#[test]
fn dense_criterion_is_invariant_under_compensated_penalty_rescaling_2933() {
    let (term, target, rho) = threshold_gate_tiny_fixture(false);
    // A zero inner budget prices the criterion at the fixture state, so both arms
    // see the identical `(t, β)` and the only difference is how `λS` is written.
    let (base_value, _, _) = term
        .clone()
        .penalized_quasi_laplace_criterion_with_cache(
            target.view(),
            &rho,
            None,
            0,
            0.4,
            1.0e-6,
            1.0e-6,
        )
        .expect("criterion at the fixture state");
    for c in RESCALINGS {
        let (mut scaled, scaled_rho) = rescaled_penalty_representation(&term, &rho, c);
        let (value, _, _) = scaled
            .penalized_quasi_laplace_criterion_with_cache(
                target.view(),
                &scaled_rho,
                None,
                0,
                0.4,
                1.0e-6,
                1.0e-6,
            )
            .expect("criterion at the rescaled representation");
        let shift = log_lambda_only_shift(&term, c);
        let tolerance = 1.0e-8 * base_value.abs().max(1.0);
        assert!(
            tolerance < 1.0e-3 * shift.abs(),
            "tolerance {tolerance} must sit far below the log-λ-only shift {shift}"
        );
        assert!(
            (value - base_value).abs() <= tolerance,
            "S → {c}·S, λ → λ/{c} moved the dense criterion from {base_value} to {value} \
             (Δ = {}); a log-λ-only normalizer moves it by {shift}",
            value - base_value
        );
    }
}

/// One constant-curvature tangent-chart atom at curvature `kappa`, with a fixed
/// set of reference rows spread over a disc of radius 0.7.
fn constant_curvature_term(kappa: f64) -> (SaeManifoldTerm, SaeManifoldRho) {
    let n = 12usize;
    let p = 2usize;
    let coords = Array2::from_shape_fn((n, 2), |(row, axis)| {
        let angle = std::f64::consts::TAU * row as f64 / n as f64;
        let radius = 0.3 + 0.2 * (row % 3) as f64;
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
            kappa,
            reference_coords: coords.clone(),
        },
    )
    .expect("constant-curvature plan");
    let bundle = plan
        .evaluate_bundle(coords.view())
        .expect("reference rows lie inside the chart");
    let m = bundle.basis_values.ncols();
    let decoder = Array2::from_shape_fn((m, p), |(basis_col, out_col)| {
        0.2 * ((basis_col + 2 * out_col) as f64 + 0.5).sin()
    });
    let atom = SaeManifoldAtom::new_with_provided_function_gram(
        "constant_curvature",
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
    assert!(
        atom.smooth_penalty_kappa_derivative().is_some(),
        "a constant-curvature plan installs dS/dκ"
    );
    let mode = AssignmentMode::softmax(1.0);
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        Array2::<f64>::zeros((n, 1)),
        vec![coords],
        vec![LatentManifold::Euclidean],
        mode,
    )
    .expect("one coordinate block for one atom");
    let term = SaeManifoldTerm::new(vec![atom], assignment).expect("single-atom term");
    let rho = SaeManifoldRho::new(0.0, 0.3, vec![Array1::<f64>::zeros(2)])
        .for_assignment(mode)
        .with_curvature(vec![(0, kappa)]);
    (term, rho)
}

#[test]
fn occam_normalizer_follows_a_curvature_parameterised_penalty_2933() {
    let (hyperbolic, hyperbolic_rho) = constant_curvature_term(-0.8);
    let (spherical, spherical_rho) = constant_curvature_term(0.9);
    let expected = half_log_pseudodeterminant_of_precisions(&spherical, &spherical_rho)
        - half_log_pseudodeterminant_of_precisions(&hyperbolic, &hyperbolic_rho);
    assert!(
        expected.abs() > 1.0e-2,
        "the two curvatures must give materially different penalty pseudo-determinants \
         (Δ = {expected})"
    );
    let moved = spherical.reml_occam_term(&spherical_rho).expect("occam at κ = 0.9")
        - hyperbolic
            .reml_occam_term(&hyperbolic_rho)
            .expect("occam at κ = -0.8");
    assert!(
        (moved - expected).abs() <= 1.0e-9 * expected.abs().max(1.0),
        "occam moved by {moved} between κ = -0.8 and κ = 0.9, but ½·r·Δlog|λS(κ)|_+ = {expected}"
    );
}

#[test]
fn kappa_occam_channel_is_the_derivative_of_the_normalizer_2933() {
    let kappa = 0.4_f64;
    let (term, rho) = constant_curvature_term(kappa);
    let channels = term
        .reml_occam_kappa_derivative(&rho)
        .expect("κ Occam channel");
    let flat = rho.kappa_flat_index(0).expect("curvature coordinate");
    assert_eq!(channels.len(), 1, "one curvature atom, one channel");
    assert_eq!(channels[0].0, flat, "the channel lands on the κ coordinate");
    let step = 1.0e-4_f64;
    let (plus, plus_rho) = constant_curvature_term(kappa + step);
    let (minus, minus_rho) = constant_curvature_term(kappa - step);
    let central = (plus.reml_occam_term(&plus_rho).expect("occam at κ + h")
        - minus.reml_occam_term(&minus_rho).expect("occam at κ − h"))
        / (2.0 * step);
    assert!(
        central.abs() > 1.0e-3,
        "the normalizer must move with κ for this check to mean anything (slope {central})"
    );
    assert!(
        (channels[0].1 - central).abs() <= 1.0e-6 * central.abs().max(1.0),
        "κ Occam channel {} disagrees with the central difference {central}",
        channels[0].1
    );
}
