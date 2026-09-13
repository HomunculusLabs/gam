//! Fit-free controls for the crosscoder transport law (gam#2234).
//!
//! `measure_atom_transport_between` projects the source layer's decoded circle
//! onto the target layer's image and fits the phase-shift law `t' = s·t + φ`.
//! Each fixture installs one exact period-1 circle atom whose downstream decoder
//! block is related to the anchor block by a known isometry of the circle, so
//! every reported quantity has a closed form derived here, not read from the
//! code under test: the sign `s`, the phase `φ`, a unit circular `R²`, every
//! transport sample the integrals evaluate lying on the law, the honest-units
//! drift `‖B_tgt − B_src‖_F / √(‖B_src‖_F·‖B_tgt‖_F)`, and zero principal angles
//! between the two images (both span the same output plane). An atom the law does
//! not describe reports a typed undefined status, not an error.
//!
//! The anchor decodes `q(t) = (sin 2πt, cos 2πt)` and the downstream block is
//! stored in the weighted fit space `√λ·B`, so the fixtures also exercise the
//! honest-units division.

#![cfg(test)]

use super::*;
use ndarray::Array2;
use std::sync::Arc;

const N_ROWS: usize = 32;
const BLOCK_LOG_LAMBDA: f64 = 1.4;
const PHASE: f64 = 0.173_205_080_756_887_73;

/// How the downstream block decodes the anchor's circle.
#[derive(Clone, Copy)]
enum Planted {
    /// `q(t + φ)`: a rotation of the chart.
    Shift(f64),
    /// `q(φ − t)`: a reflection of the chart.
    Reflection(f64),
}

fn crosscoder_term(planted: Planted) -> (SaeManifoldTerm, CrosscoderLayout) {
    let evaluator = Arc::new(PeriodicHarmonicEvaluator::new(3).expect("order-1 periodic basis"));
    let coords =
        Array2::<f64>::from_shape_fn((N_ROWS, 1), |(row, _)| row as f64 / N_ROWS as f64);
    let (phi, jet) = evaluator
        .evaluate(coords.view())
        .expect("basis at the fixture coordinates");
    let sqrt_lambda = (0.5 * BLOCK_LOG_LAMBDA).exp();

    // Basis rows are [1, sin 2πt, cos 2πt]; output columns 0..2 are the anchor
    // and 2..4 the downstream block. `sin_row[c]` / `cos_row[c]` are the
    // coefficients of the sin / cos basis rows in downstream output column c.
    let mut decoder = Array2::<f64>::zeros((3, 4));
    decoder[[1, 0]] = 1.0;
    decoder[[2, 1]] = 1.0;
    let (sin_row, cos_row) = match planted {
        Planted::Shift(phase) => {
            let angle = std::f64::consts::TAU * phase;
            // sin(θ + a) = sinθ·cos a + cosθ·sin a;  cos(θ + a) = cosθ·cos a − sinθ·sin a.
            ([angle.cos(), -angle.sin()], [angle.sin(), angle.cos()])
        }
        Planted::Reflection(phase) => {
            let angle = std::f64::consts::TAU * phase;
            // sin(a − θ) = sin a·cosθ − cos a·sinθ;  cos(a − θ) = cos a·cosθ + sin a·sinθ.
            ([-angle.cos(), angle.sin()], [angle.sin(), angle.cos()])
        }
    };
    for column in 0..2 {
        decoder[[1, 2 + column]] = sqrt_lambda * sin_row[column];
        decoder[[2, 2 + column]] = sqrt_lambda * cos_row[column];
    }

    let atom = SaeManifoldAtom::new_with_provided_function_gram(
        "transport-law-circle",
        SaeAtomBasisKind::Periodic,
        1,
        phi,
        jet,
        decoder,
        Array2::<f64>::eye(3),
    )
    .expect("circle atom")
    .with_basis_evaluator(evaluator);
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        Array2::<f64>::zeros((N_ROWS, 1)),
        vec![coords],
        vec![LatentManifold::Circle { period: 1.0 }],
        AssignmentMode::softmax(1.0),
    )
    .expect("single-atom assignment");
    let mut term = SaeManifoldTerm::new(vec![atom], assignment).expect("crosscoder term");
    let layout = CrosscoderLayout::new(
        2,
        vec![2],
        vec!["downstream".into()],
        vec![BLOCK_LOG_LAMBDA],
    )
    .expect("two-layer layout");
    term.set_crosscoder_layout(layout.clone())
        .expect("install the layout");
    (term, layout)
}

/// Distance on the period-1 circle.
fn circular_gap(a: f64, b: f64) -> f64 {
    let d = (a - b).rem_euclid(1.0);
    d.min(1.0 - d)
}

/// The closed-form drift of a block `[[0, 0], sin_row, cos_row]` against the
/// anchor `[[0, 0], [1, 0], [0, 1]]`, both of Frobenius norm √2.
fn rotation_drift(phase: f64) -> f64 {
    // ‖R(a) − I‖_F = 2√(1 − cos a) = 2√2·|sin(a/2)|, divided by √(√2·√2).
    2.0 * (std::f64::consts::PI * phase).sin().abs()
}

fn reflection_drift() -> f64 {
    // ‖F(a) − I‖_F² = (cos a + 1)² + 2 sin² a + (cos a − 1)² = 4, divided by √2.
    std::f64::consts::SQRT_2
}

/// The measured report, or a panic naming why the fixture was not measured.
fn measured(label: &str, status: Result<AtomTransportStatus, String>) -> AtomTransportReport {
    match status {
        Ok(AtomTransportStatus::Measured(report)) => report,
        Ok(AtomTransportStatus::Undefined { reason }) => {
            panic!("{label}: a periodic circle atom must be measured, got undefined: {reason}")
        }
        Err(error) => panic!("{label}: transport measurement failed: {error}"),
    }
}

fn assert_isometry(
    label: &str,
    report: &AtomTransportReport,
    expected_sign: f64,
    expected_phase: f64,
    expected_drift: f64,
) {
    let (sign, phase) = report.phase_shift;
    assert_eq!(
        sign, expected_sign,
        "{label}: phase-law sign {sign} != planted {expected_sign}"
    );
    assert!(
        circular_gap(phase, expected_phase) <= 1.0e-9,
        "{label}: phase {phase} is {} turns from the planted {expected_phase}",
        circular_gap(phase, expected_phase)
    );
    assert!(
        (-0.5..0.5).contains(&phase),
        "{label}: phase {phase} must be wrapped to [-1/2, 1/2)"
    );
    assert!(
        1.0 - report.phase_r2 <= 1.0e-9,
        "{label}: a planted isometry must be an exact phase law, got phase_r2 = {}",
        report.phase_r2
    );
    assert!(
        !report.transport_grid.is_empty(),
        "{label}: the transport integrals evaluated no sample"
    );
    let mut previous = f64::NEG_INFINITY;
    for &(t, t_prime) in &report.transport_grid {
        assert!(
            (0.0..1.0).contains(&t) && t > previous,
            "{label}: samples must be strictly increasing inside [0, 1), got t = {t} after {previous}"
        );
        previous = t;
        let on_law = expected_sign * t + expected_phase;
        assert!(
            circular_gap(t_prime, on_law) <= 1.0e-9,
            "{label}: sample t = {t} transported to {t_prime}, the law says {on_law}"
        );
    }
    assert!(
        (report.drift - expected_drift).abs() <= 1.0e-12 * expected_drift.max(1.0),
        "{label}: drift {} != closed form {expected_drift}",
        report.drift
    );
    assert_eq!(report.principal_angles.len(), 2, "{label}: two rank-2 images");
    for angle in &report.principal_angles {
        assert!(
            angle.abs() <= 1.0e-6,
            "{label}: both images span the output plane, got principal angle {angle}"
        );
    }
    assert!(
        report.deviation_locus().is_some(),
        "{label}: a non-empty grid must name the law's worst locus"
    );
}

#[test]
fn planted_rotation_is_recovered_in_both_directions() {
    let (term, layout) = crosscoder_term(Planted::Shift(PHASE));
    // The downstream block decodes q(t' + φ), so the anchor point q(t) lands at
    // t' = t − φ; the reverse direction lands at t' = t + φ.
    let forward = measured(
        "rotation anchor->block",
        measure_atom_transport_between(
            &term,
            &layout,
            0,
            CrosscoderLayer::Anchor,
            CrosscoderLayer::Block(0),
        ),
    );
    assert_isometry("rotation anchor->block", &forward, 1.0, -PHASE, rotation_drift(PHASE));
    let reverse = measured(
        "rotation block->anchor",
        measure_atom_transport_between(
            &term,
            &layout,
            0,
            CrosscoderLayer::Block(0),
            CrosscoderLayer::Anchor,
        ),
    );
    assert_isometry("rotation block->anchor", &reverse, 1.0, PHASE, rotation_drift(PHASE));
}

#[test]
fn planted_reflection_reports_a_negative_sign() {
    let (term, layout) = crosscoder_term(Planted::Reflection(PHASE));
    // The downstream block decodes q(φ − t'), so q(t) lands at t' = φ − t in
    // both directions (a reflection is an involution).
    for (label, source, target) in [
        ("reflection anchor->block", CrosscoderLayer::Anchor, CrosscoderLayer::Block(0)),
        ("reflection block->anchor", CrosscoderLayer::Block(0), CrosscoderLayer::Anchor),
    ] {
        let report = measured(
            label,
            measure_atom_transport_between(&term, &layout, 0, source, target),
        );
        assert_isometry(label, &report, -1.0, PHASE, reflection_drift());
    }
}

#[test]
fn half_turn_at_the_wrap_boundary_is_an_exact_law() {
    // φ = ½ sits exactly on the wrap boundary of [−½, ½), where an unwrapped
    // phase average would split the samples between ±½.
    let (term, layout) = crosscoder_term(Planted::Shift(0.5));
    let report = measured(
        "half turn",
        measure_atom_transport_between(
            &term,
            &layout,
            0,
            CrosscoderLayer::Anchor,
            CrosscoderLayer::Block(0),
        ),
    );
    assert_isometry("half turn", &report, 1.0, 0.5, rotation_drift(0.5));
}

#[test]
fn transport_refuses_an_absent_block() {
    let (term, layout) = crosscoder_term(Planted::Shift(PHASE));
    let absent = measure_atom_transport_between(
        &term,
        &layout,
        0,
        CrosscoderLayer::Anchor,
        CrosscoderLayer::Block(1),
    );
    assert!(
        matches!(&absent, Err(message) if message.contains("block index")),
        "a block the layout does not have must be refused, got {absent:?}"
    );
}

#[test]
fn an_atom_short_of_the_fitted_homotopy_endpoint_is_undefined_not_an_error() {
    let (mut term, layout) = crosscoder_term(Planted::Shift(PHASE));
    term.atoms[0].homotopy_eta = 0.5;
    let status = measure_atom_transport_between(
        &term,
        &layout,
        0,
        CrosscoderLayer::Anchor,
        CrosscoderLayer::Block(0),
    );
    assert!(
        matches!(
            &status,
            Ok(AtomTransportStatus::Undefined { reason }) if reason.contains("homotopy eta")
        ),
        "an atom the phase-shift law does not describe must report a typed undefined status, \
         got {status:?}"
    );
}
