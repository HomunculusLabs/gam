#![cfg(test)]
//! #2933 F07 — the dense exact-A lane classifies every eigenvalue of the raw reduced Schur
//! `S_A` in its pencil `S_A w = μ Y w` ([`exact_a_reduced_pencil_operands`]), on the band the
//! dense joint route classifies on, and prices `log|Y| + Σ ln μ + Σ ln κ`.
//!
//! Every fixture is built from rows of width 2 against a border of at most three columns,
//! so `S_A`, `Y`, the substituted form `s` and each verdict are available in closed form. A
//! row whose majorizer block `B_tt = A_tt − ΔC_tt` is flat along a coordinate carries the
//! evidence factor's spectral pin there, which is where a nonzero `s` comes from.

use super::*;
use ndarray::array;

fn options() -> ArrowSolveOptions {
    ArrowSolveOptions::direct()
        .with_newton_schur_tikhonov(SPECTRAL_DEFLATION_REL_FLOOR)
        .with_indefinite_refusing_evidence_unit_deflation(SPECTRAL_DEFLATION_REL_FLOOR)
}

fn lane() -> SurrogateLaneState {
    let mut lane = SurrogateLaneState::new(SurrogateLaneConfig {
        num_probes: 4,
        seed: 0x2933,
        rel_tol: 1.0e-10,
        cg_rel_tol: 1.0e-12,
        deflation_subspace_iters: 1,
        deflation_target_std_err_rel: 1.0,
    });
    lane.request_logdet_derivative_bundle();
    lane.request_inverse_probes();
    lane
}

/// One exact-A row: `A_tt`, `A_tβ`, and the carrier `ΔC_tt`, `ΔC_tβ` on every border column.
struct Row {
    a_tt: Array2<f64>,
    a_tbeta: Array2<f64>,
    delta_tt: Array2<f64>,
    delta_tbeta: Array2<f64>,
}

/// An exact-A system with shared block `A_ββ = a_bb`, no clamp, and `remainder = A_ββ − B_ββ`.
fn exact_a_system(
    rows: Vec<Row>,
    a_bb: Array2<f64>,
    remainder: Option<Array2<f64>>,
) -> ArrowSchurSystem {
    let k = a_bb.nrows();
    let mut system = ArrowSchurSystem::new(rows.len(), 2, k);
    for (block, row) in system.rows.iter_mut().zip(rows.iter()) {
        block.htt.assign(&row.a_tt);
        block.htbeta.assign(&row.a_tbeta);
    }
    system.hbb.assign(&a_bb);
    let columns: Arc<[usize]> = (0..k).collect::<Vec<_>>().into();
    system.exact_a_classification = Some(ExactAClassificationGeometry {
        rows: rows
            .into_iter()
            .map(|row| ExactAClassificationRow {
                delta_tt: row.delta_tt,
                delta_tbeta: row.delta_tbeta,
                border_columns: Arc::clone(&columns),
                clamp_diag: Array1::<f64>::zeros(2),
            })
            .collect::<Vec<_>>()
            .into(),
        border_remainder: remainder
            .map(|remainder| Arc::new(DensePenaltyOp(remainder)) as Arc<dyn BetaPenaltyOp>),
    });
    system
}

/// The admitted dense lane, with the derivative bundle and the EFS pairs requested.
fn evaluate(system: &ArrowSchurSystem) -> (MatrixFreeArrowEvidenceEvaluation, SurrogateLaneState) {
    let mut lane = lane();
    let evaluation = matrix_free_arrow_evidence_evaluation(
        system,
        0.0,
        0.0,
        &options(),
        4,
        1,
        0x2933,
        &mut lane,
        true,
    )
    .expect("the fixture is priced, not refused");
    (evaluation, lane)
}

fn operands(
    system: &ArrowSchurSystem,
    evaluation: &MatrixFreeArrowEvidenceEvaluation,
) -> ExactAReducedPencilOperands {
    exact_a_reduced_pencil_operands(system, &evaluation.factor_cache.htt_factors)
        .expect("the carrier is installed")
}

/// `tr(S̃⁻¹)` off the evaluation's derivative bundle and off its EFS pairs.
fn inverse_traces(lane: &mut SurrogateLaneState) -> (f64, f64) {
    let identity = |v: ArrayView1<f64>| v.to_owned();
    let bundle = lane
        .take_logdet_derivative_bundle()
        .expect("the requested derivative bundle");
    assert!(
        bundle.exact_inverse_vectors().is_some(),
        "the dense lane's bundle spans the inverse of the operator it priced"
    );
    let derivative = bundle
        .directional_derivative(&identity)
        .expect("a finite derivative");
    let (probes, sinv) = lane.take_inverse_probes().expect("the requested EFS pairs");
    let efs = hutchinson_reduced_schur_inverse_trace(&probes, &sinv, &identity)
        .expect("an EFS trace");
    (derivative, efs)
}

/// `S_A = A_ββ − Σ A_tβᵀ A_tt⁻¹ A_tβ` through the evaluation's own row factors.
fn raw_reduced_schur(system: &ArrowSchurSystem, factors: &ArrowFactorSlab) -> Array2<f64> {
    let mut schur = system.hbb.clone();
    for (index, row) in system.rows.iter().enumerate() {
        schur -= &row
            .htbeta
            .t()
            .dot(&cholesky_solve_matrix(factors.factor(index), &row.htbeta));
    }
    schur
}

/// `κ₂` of a symmetric positive-definite matrix.
fn condition(matrix: &Array2<f64>) -> f64 {
    let (values, _) = matrix.eigh(Side::Lower).expect("a symmetric eigensolve");
    let (low, high) = values
        .iter()
        .fold((f64::INFINITY, 0.0_f64), |(low, high), &value| {
            (low.min(value.abs()), high.max(value.abs()))
        });
    high / low
}

/// The quotient fixture's rows (see
/// `a_declared_quotient_direction_and_an_unused_atom_read_unit_curvature_2933_f07`), with
/// their cross blocks written in border coordinates `ξ`, `β = Tξ`.
fn three_column_rows(coupling: f64, transform: &Array2<f64>) -> Vec<Row> {
    vec![
        Row {
            a_tt: array![[2.0, 0.0], [0.0, 1.0]],
            a_tbeta: array![[0.0, 0.0, 0.0], [0.0, coupling, 0.0]].dot(transform),
            delta_tt: array![[0.0, 0.0], [0.0, 1.0]],
            delta_tbeta: array![[0.0, 0.0, 0.0], [0.0, coupling, 0.0]].dot(transform),
        },
        Row {
            a_tt: array![[0.5, 0.0], [0.0, 3.0]],
            a_tbeta: array![[0.0, 0.0, 1.0], [0.0, 0.0, 0.0]].dot(transform),
            delta_tt: array![[-0.5, 0.0], [0.0, 0.0]],
            delta_tbeta: Array2::<f64>::zeros((2, 3)),
        },
    ]
}

/// Two rows of width 2 against a 3-column border, the shared block moved by `shift`.
/// `ΔC_tt = I/4` makes the exact blocks stiffer than their majorizers, so `Y ≠ S_A` and the
/// pencil is not the identity, while every block stays well conditioned.
fn full_rank_system(shift: &Array2<f64>) -> ArrowSchurSystem {
    let row = |diagonal: f64, phase: f64| Row {
        a_tt: array![[2.0 + diagonal, 0.3], [0.3, 1.5]],
        a_tbeta: Array2::from_shape_fn((2, 3), |(i, j)| {
            0.4 * (phase + ((i + 1) * (j + 1)) as f64).cos()
        }),
        delta_tt: Array2::<f64>::eye(2) * 0.25,
        delta_tbeta: Array2::<f64>::zeros((2, 3)),
    };
    let a_bb = array![[3.0, 0.2, 0.0], [0.2, 4.0, 0.2], [0.0, 0.2, 5.0]] + shift;
    exact_a_system(vec![row(0.0, 0.0), row(1.0, 0.7)], a_bb, None)
}

/// Positive control: a full-rank state with an empty band prices exactly `log|S_A|`, since
/// `det S_A = det Y · Π μ`, and its bundle and EFS pairs span `S_A⁻¹`. On such a state main's
/// dense lane priced `Σ ln λ(S_A)`, so the printed difference is the change against main.
#[test]
fn a_full_rank_state_prices_log_s_a_with_an_empty_band_2933_f07() {
    let system = full_rank_system(&Array2::<f64>::zeros((3, 3)));
    let (evaluation, mut lane) = evaluate(&system);
    assert!(
        evaluation.exact_a_reduced_band_directions.is_empty()
            && lane.logdet_derivative_band_directions().is_empty(),
        "a full-rank state has an empty band: {:?}",
        evaluation.exact_a_reduced_band_directions
    );
    assert_eq!(evaluation.exact_a_clamp_basin_directions, 0);

    let schur = raw_reduced_schur(&system, &evaluation.factor_cache.htt_factors);
    let lower = cholesky_lower(&schur).expect("the fixture's S_A is positive definite");
    let expected = (0..3).map(|axis| 2.0 * lower[[axis, axis]].ln()).sum::<f64>();
    let expected_trace = cholesky_solve_matrix(&lower, &Array2::<f64>::eye(3))
        .diag()
        .sum();
    let metric = operands(&system, &evaluation).metric;
    // The whitened operator `L⁻¹S_A L⁻ᵀ` carries a backward error `dim·ε·κ(Y)·‖Ŝ‖`, and each
    // of the `dim` logarithms moves by at most that over its own `μ`, which `κ(S_A)` bounds.
    let dim = 3.0_f64;
    let bar = dim * dim * f64::EPSILON * condition(&metric) * condition(&schur);
    let delta = (evaluation.log_det_schur - expected).abs();
    println!(
        "[pencil lane positive control] log|S_A| {expected:.15e}, lane {:.15e}: max|Δ| against \
         main's full-rank value {delta:.3e}, bar {bar:.3e}",
        evaluation.log_det_schur
    );
    assert!(
        delta <= bar,
        "the pencil value {:.15e} departs from log|S_A| {expected:.15e} by {delta:.3e} > {bar:.3e}",
        evaluation.log_det_schur
    );
    let (derivative, efs) = inverse_traces(&mut lane);
    println!(
        "[pencil lane positive control] tr(S_A⁻¹) {expected_trace:.15e}: bundle {derivative:.15e}, \
         EFS {efs:.15e}"
    );
    assert!(
        (derivative - expected_trace).abs() <= bar * expected_trace,
        "bundle tr(S⁻¹) {derivative:.15e} against {expected_trace:.15e}"
    );
    assert!(
        (efs - expected_trace).abs() <= bar * expected_trace,
        "EFS tr(S⁻¹) {efs:.15e} against {expected_trace:.15e}"
    );
}

/// A direction whose exact curvature is roundoff against unit majorizer curvature is returned
/// as a typed band direction and priced at the metric's own curvature, with a finite value and
/// no `1/μ` amplification in the bundle. An unused atom, whose block holds only its penalty,
/// reads `μ = 1` with `s = 0`.
#[test]
fn a_null_reduced_direction_is_a_typed_band_direction_2933_f07() {
    // Column 0 carries exact curvature 5.859e-15, the direction routes' seed state priced raw
    // (job 1152405), against majorizer curvature 1: the remainder is `5.859e-15 − 1`. Column
    // 1 is an unused atom, identical in `A` and `B`. No row couples to the border.
    const SEED_STATE_CURVATURE: f64 = 5.859e-15;
    let row = Row {
        a_tt: Array2::<f64>::eye(2),
        a_tbeta: Array2::<f64>::zeros((2, 2)),
        delta_tt: Array2::<f64>::zeros((2, 2)),
        delta_tbeta: Array2::<f64>::zeros((2, 2)),
    };
    let system = exact_a_system(
        vec![row],
        array![[SEED_STATE_CURVATURE, 0.0], [0.0, 5.0]],
        Some(array![[SEED_STATE_CURVATURE - 1.0, 0.0], [0.0, 0.0]]),
    );
    let (evaluation, mut lane) = evaluate(&system);
    let band = &evaluation.exact_a_reduced_band_directions;
    println!("[pencil lane band] {band:?}, log|S̃| {:.15e}", evaluation.log_det_schur);
    assert_eq!(band.len(), 1, "one band direction: {band:?}");
    let direction = &band[0];
    // Every operand here is an exact small integer or the seed curvature, so the arithmetic
    // is exact up to one ε of the unit metric.
    let bar = 4.0 * f64::EPSILON;
    assert_eq!(direction.origin, ReducedBandOrigin::Pencil);
    assert!(
        (direction.curvature - SEED_STATE_CURVATURE).abs() <= bar,
        "band curvature {:e}",
        direction.curvature
    );
    assert!(direction.edge >= exact_a_pencil_floor() && direction.substituted_stiffness == 0.0);
    assert!(
        (direction.direction[0].abs() - 1.0).abs() <= bar && direction.direction[1].abs() <= bar,
        "the band direction is the seed column, Y-normalized: {:?}",
        direction.direction
    );
    // `log|Y| + ln 5/5` with `Y = diag(1, 5)`: the band direction adds `ln 1 = 0`.
    let expected = 5.0_f64.ln();
    assert!(
        evaluation.log_det_schur.is_finite() && (evaluation.log_det_schur - expected).abs() <= bar,
        "log|S̃| {:.15e} against ln 5",
        evaluation.log_det_schur
    );
    // The lane records the band direction its bundle carries, where the bundle's derivative
    // is missing the channel `wᵀ(dY − dS_A)w`.
    let recorded = lane.logdet_derivative_band_directions();
    println!("[pencil lane band] bundle band directions: {}", recorded.len());
    assert_eq!(recorded.len(), 1, "the lane records the bundle's band direction");
    assert_eq!(recorded[0].direction, direction.direction);
    // No amplification: the priced inverse is `diag(1/1, 1/5)`, not `1/5.859e-15`.
    let (derivative, efs) = inverse_traces(&mut lane);
    assert!(
        (derivative - 1.2).abs() <= bar && (efs - 1.2).abs() <= bar,
        "tr(S̃⁻¹) must be 1 + 1/5: bundle {derivative:e}, EFS {efs:e}"
    );

    let schur = raw_reduced_schur(&system, &evaluation.factor_cache.htt_factors);
    let operands = operands(&system, &evaluation);
    let unused_atom = schur[[1, 1]] / operands.metric[[1, 1]];
    assert!(
        (unused_atom - 1.0).abs() <= bar
            && schur[[0, 1]] == 0.0
            && operands.metric[[0, 1]] == 0.0
            && operands.substituted_metric.iter().all(|&value| value == 0.0),
        "the unused atom reads μ = {unused_atom:e} with s = 0: metric {:?}, substituted {:?}",
        operands.metric,
        operands.substituted_metric
    );
}

/// A declared quotient direction reads `μ = 1` and is priced `ln 1 = 0`; an unused atom
/// reads `μ = 1`; installing the quotient leaves the other directions' `Y` and `s` unchanged.
#[test]
fn a_declared_quotient_direction_and_an_unused_atom_read_unit_curvature_2933_f07() {
    // Column 0: an unused atom, `A_ββ = B_ββ = 4`.
    // Column 1: row 0 couples along the coordinate its majorizer block is flat in
    //   (`B_tt = diag(2, 0)`, `A_tt = diag(2, 1)`, `ΔC_tβ = A_tβ = a·e₂`), so the evidence
    //   factor's spectral pin substitutes `a²`: `S_A = h₁ − a²`, `Y = h₁ + a²`, `s = a²`.
    // Column 2: row 1 couples through a concave remainder (`A_tt = 1/2`, `B_tt = 1`,
    //   `A_tβ = B_tβ = 1`): `S_A = h₂ − 2`, `Y = h₂ + 4 − 4 = h₂`.
    // The quotient is column 2.
    let (coupling, h1, h2) = (0.5_f64, 2.0_f64, 4.0_f64);
    let a_bb = array![[4.0, 0.0, 0.0], [0.0, h1, 0.0], [0.0, 0.0, h2]];
    let identity = Array2::<f64>::eye(3);
    let plain_system = exact_a_system(three_column_rows(coupling, &identity), a_bb.clone(), None);
    let mut quotient_system = exact_a_system(three_column_rows(coupling, &identity), a_bb, None);
    quotient_system
        .set_beta_gauge_quotient(
            ArrowBetaGaugeQuotient::new(vec![array![0.0, 0.0, 1.0]]).expect("a unit direction"),
        )
        .expect("the quotient spans the border");
    let (plain, _) = evaluate(&plain_system);
    let (quotiented, _) = evaluate(&quotient_system);
    let plain_operands = operands(&plain_system, &plain);
    let quotient_operands = operands(&quotient_system, &quotiented);
    let a_sq = coupling * coupling;
    let bar = 16.0 * f64::EPSILON;
    let expected_metric = |quotient: f64| array![[4.0, 0.0, 0.0], [0.0, h1 + a_sq, 0.0], [0.0, 0.0, quotient]];
    let expected_substituted = array![[0.0, 0.0, 0.0], [0.0, a_sq, 0.0], [0.0, 0.0, 0.0]];
    for (label, forms, quotient_metric) in [
        ("no quotient", &plain_operands, h2),
        ("quotient", &quotient_operands, 1.0),
    ] {
        let metric_gap = (&forms.metric - &expected_metric(quotient_metric))
            .iter()
            .fold(0.0_f64, |acc, value| acc.max(value.abs()));
        let substituted_gap = (&forms.substituted_metric - &expected_substituted)
            .iter()
            .fold(0.0_f64, |acc, value| acc.max(value.abs()));
        println!(
            "[pencil lane quotient] {label}: max|Y − closed form| {metric_gap:.3e}, \
             max|s − closed form| {substituted_gap:.3e}"
        );
        assert!(
            metric_gap <= bar && substituted_gap <= bar,
            "{label}: Y {:?}, s {:?}",
            forms.metric,
            forms.substituted_metric
        );
    }
    // Every direction is retained: `μ = (1, (h₁ − a²)/(h₁ + a²), (h₂ − 2)/h₂)` without the
    // quotient and `(1, (h₁ − a²)/(h₁ + a²), 1)` with it, none inside its edge.
    assert!(plain.exact_a_reduced_band_directions.is_empty());
    assert!(quotiented.exact_a_reduced_band_directions.is_empty());
    let expected_plain = (4.0 * (h1 - a_sq) * (h2 - 2.0)).ln();
    let expected_quotient = (4.0 * (h1 - a_sq)).ln();
    println!(
        "[pencil lane quotient] log|S̃|: no quotient {:.15e} (expected {expected_plain:.15e}), \
         quotient {:.15e} (expected {expected_quotient:.15e})",
        plain.log_det_schur, quotiented.log_det_schur
    );
    assert!(
        (plain.log_det_schur - expected_plain).abs() <= bar
            && (quotiented.log_det_schur - expected_quotient).abs() <= bar,
        "the quotient direction must price ln 1 = 0"
    );
}

/// `β = Tξ` with a nonorthogonal `T` moves `S_A`, `Y`, `s` and `E` by congruence, so the
/// pencil's curvatures and verdicts are unchanged, the band direction maps back as `w = T w′`,
/// and the value moves by `2 ln|det T|`. Only `‖z‖²`, and so `τ`, depends on the coordinates,
/// and `τ` binds nowhere on this state.
#[test]
fn a_nonorthogonal_border_reparametrization_is_a_congruence_2933_f07() {
    // The quotient fixture with column 0 replaced by the seed-state band direction: a band
    // direction, a partially pinned retained direction and a concave-remainder direction.
    const SEED_STATE_CURVATURE: f64 = 5.859e-15;
    let (coupling, h1, h2) = (0.5_f64, 2.0_f64, 4.0_f64);
    let a_bb = array![[SEED_STATE_CURVATURE, 0.0, 0.0], [0.0, h1, 0.0], [0.0, 0.0, h2]];
    let remainder = array![[SEED_STATE_CURVATURE - 1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]];
    let transform = array![[1.0, 0.5, 0.0], [0.0, 1.0, 0.3], [0.2, 0.0, 1.0]];
    let determinant = 1.0_f64 + 0.5 * 0.3 * 0.2;
    let congruent = |t: &Array2<f64>| {
        exact_a_system(
            three_column_rows(coupling, t),
            t.t().dot(&a_bb).dot(t),
            Some(t.t().dot(&remainder).dot(t)),
        )
    };
    let base_system = congruent(&Array2::<f64>::eye(3));
    let moved_system = congruent(&transform);
    let (base, _) = evaluate(&base_system);
    let (moved, _) = evaluate(&moved_system);
    let base_band = &base.exact_a_reduced_band_directions;
    let moved_band = &moved.exact_a_reduced_band_directions;
    assert_eq!(base_band.len(), 1, "base band {base_band:?}");
    assert_eq!(moved_band.len(), 1, "moved band {moved_band:?}");

    // Backward error of the whitened pencil in either coordinates: `dim·ε·κ(TᵀT)·κ(Y)` in `μ`
    // units. The value's logarithms divide it by the smallest retained `μ`, 1/2 here, and the
    // band direction's alignment divides it by the band's spectral gap, also 1/2.
    let dim = 3.0_f64;
    let metric = operands(&base_system, &base).metric;
    let mu_bar = dim * dim * f64::EPSILON * condition(&transform.t().dot(&transform)) * condition(&metric);
    let value_bar = dim * mu_bar / 0.5;
    let direction_bar = mu_bar / 0.5;
    let value_gap = (moved.log_det_schur - base.log_det_schur - 2.0 * determinant.ln()).abs();
    let mapped = transform.dot(&moved_band[0].direction);
    let alignment_gap = (&mapped - &base_band[0].direction)
        .iter()
        .fold(0.0_f64, |acc, value| acc.max(value.abs()))
        .min(
            (&mapped + &base_band[0].direction)
                .iter()
                .fold(0.0_f64, |acc, value| acc.max(value.abs())),
        );
    let curvature_gap = (moved_band[0].curvature - base_band[0].curvature).abs();
    println!(
        "[pencil lane congruence] log|S̃| {:.15e} -> {:.15e}, 2 ln|det T| {:.15e}: gap \
         {value_gap:.3e} (bar {value_bar:.3e}); band μ {:e} -> {:e}: gap {curvature_gap:.3e} \
         (bar {mu_bar:.3e}); max|T w′ ∓ w| {alignment_gap:.3e} (bar {direction_bar:.3e})",
        base.log_det_schur,
        moved.log_det_schur,
        2.0 * determinant.ln(),
        base_band[0].curvature,
        moved_band[0].curvature,
    );
    // `s` is a quadratic form in `μ` units, so it carries the same backward error as `μ`.
    let substituted_gap =
        (moved_band[0].substituted_stiffness - base_band[0].substituted_stiffness).abs();
    println!(
        "[pencil lane congruence] band s {:e} -> {:e}: gap {substituted_gap:.3e} (bar {mu_bar:.3e})",
        base_band[0].substituted_stiffness, moved_band[0].substituted_stiffness,
    );
    assert!(value_gap <= value_bar, "the value is not a congruence");
    assert!(curvature_gap <= mu_bar, "the band curvature moved");
    assert!(alignment_gap <= direction_bar, "the band direction does not map back");
    assert!(substituted_gap <= mu_bar, "the band direction's substituted stiffness moved");
}

/// A direction crossing its band edge moves the value by `ln(edge)`: continuous at a full pin
/// (`s = 1`), `ln s` at a partial pin, `ln √ε` at the bare floor. The two discontinuous paths
/// are what an outer search comparing values across the edge would read.
#[test]
fn a_band_edge_crossing_moves_the_value_by_the_log_of_its_edge_2933_f07() {
    // One row, one border column. Row 0's majorizer block is flat along its second coordinate
    // (`B_tt = diag(2, 0)`, `A_tt = diag(2, 1)`) and `ΔC_tβ = A_tβ = a·e₂`, so the evidence
    // factor's spectral pin substitutes `s = a²` along the border. With `A_ββ = μ + s` and the
    // remainder `μ + 2s − 1`: `S_A = μ`, `Y = 1`.
    let state = |curvature: f64, substituted: f64| {
        let coupling = substituted.sqrt();
        exact_a_system(
            vec![Row {
                a_tt: array![[2.0, 0.0], [0.0, 1.0]],
                a_tbeta: array![[0.0], [coupling]],
                delta_tt: array![[0.0, 0.0], [0.0, 1.0]],
                delta_tbeta: array![[0.0], [coupling]],
            }],
            array![[curvature + substituted]],
            Some(array![[curvature + 2.0 * substituted - 1.0]]),
        )
    };
    // A step many digits above the pencil's resolution and far inside either stratum.
    let step = f64::EPSILON.sqrt().sqrt();
    for substituted in [1.0_f64, 0.5, 0.0] {
        let edge = exact_a_pencil_floor().max(substituted);
        let (inside, _) = evaluate(&state(edge * (1.0 - step), substituted));
        let (outside, _) = evaluate(&state(edge * (1.0 + step), substituted));
        let jump = outside.log_det_schur - inside.log_det_schur;
        println!(
            "[pencil lane band-edge path] s = {substituted}: edge {edge:.6e}; inside {:.12e} \
             ({} band), outside {:.12e} ({} band); jump {jump:.6e} against ln(edge) {:.6e}, \
             ½·jump {:.6e} at ½log|A|",
            inside.log_det_schur,
            inside.exact_a_reduced_band_directions.len(),
            outside.log_det_schur,
            outside.exact_a_reduced_band_directions.len(),
            edge.ln(),
            0.5 * jump,
        );
        assert_eq!(inside.exact_a_reduced_band_directions.len(), 1);
        assert_eq!(outside.exact_a_reduced_band_directions.len(), 0);
        // Outside prices `ln Y + ln μ`, inside `ln Y`, so the jump is `ln(edge) + ln(1 + step)`.
        assert!(
            (jump - edge.ln()).abs() <= 2.0 * step,
            "s = {substituted}: jump {jump:e} against ln(edge) {:e}",
            edge.ln()
        );
    }
}

/// A negative pencil direction with no clamp to restore is a saddle, refused with the typed
/// marker rather than priced.
#[test]
fn a_negative_direction_beyond_its_clamp_refuses_as_a_saddle_2933_f07() {
    // `A_tt = 1`, `A_tβ = 1`, `A_ββ = 1/2`, `B_tt = 3`: `S_A = −1/2`, `Y = 3/2`, no clamp.
    let system = exact_a_system(
        vec![Row {
            a_tt: array![[1.0, 0.0], [0.0, 1.0]],
            a_tbeta: array![[1.0], [0.0]],
            delta_tt: array![[-2.0, 0.0], [0.0, 0.0]],
            delta_tbeta: array![[0.0], [0.0]],
        }],
        array![[0.5]],
        None,
    );
    let mut lane = lane();
    let refusal = matrix_free_arrow_evidence_evaluation(
        &system,
        0.0,
        0.0,
        &options(),
        4,
        1,
        0x2933,
        &mut lane,
        true,
    )
    .err()
    .expect("a negative direction with no clamp is a saddle")
    .to_string();
    assert!(
        ArrowSchurError::rendered_is_indefinite_evidence(&refusal),
        "the saddle refusal carries the typed marker: {refusal}"
    );
}

/// With an empty band the bundle is the derivative of the value: it matches a central
/// difference of the lane's own value along border perturbations, and main's bundle formula
/// `Σ vᵢᵀDvᵢ/λᵢ` off the ordinary eigenpairs of `S_A`, which the printed difference states.
#[test]
fn a_full_rank_bundle_matches_its_central_difference_and_main_s_formula_2933_f07() {
    let base = full_rank_system(&Array2::<f64>::zeros((3, 3)));
    let (evaluation, mut lane) = evaluate(&base);
    let schur = raw_reduced_schur(&base, &evaluation.factor_cache.htt_factors);
    let metric = operands(&base, &evaluation).metric;
    let dim = 3.0_f64;
    // The value's arithmetic bar, as in the positive control.
    let value_bar = dim * dim * f64::EPSILON * condition(&metric) * condition(&schur);
    let bundle = lane
        .take_logdet_derivative_bundle()
        .expect("the requested derivative bundle");
    let (main_values, main_vectors) = schur.eigh(Side::Lower).expect("a symmetric eigensolve");
    let inverse = cholesky_solve_matrix(
        &cholesky_lower(&schur).expect("the fixture's S_A is positive definite"),
        &Array2::<f64>::eye(3),
    );
    // The step balances a central difference's truncation `h²/3·|tr((S⁻¹D)³)|` against its
    // roundoff `value_bar/h`.
    let step = f64::EPSILON.cbrt();
    let perturbations = [
        array![[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        array![[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        array![[0.5, 0.2, 0.0], [0.2, -0.3, 0.1], [0.0, 0.1, 0.8]],
    ];
    let mut main_gap = 0.0_f64;
    for (index, perturbation) in perturbations.iter().enumerate() {
        let apply = |v: ArrayView1<f64>| perturbation.dot(&v);
        let derivative = bundle
            .directional_derivative(&apply)
            .expect("a finite derivative");
        let (plus, _) = evaluate(&full_rank_system(&(perturbation * step)));
        let (minus, _) = evaluate(&full_rank_system(&(perturbation * -step)));
        assert!(
            plus.exact_a_reduced_band_directions.is_empty()
                && minus.exact_a_reduced_band_directions.is_empty(),
            "the central difference stays on the empty-band stratum"
        );
        let central = (plus.log_det_schur - minus.log_det_schur) / (2.0 * step);
        let product = inverse.dot(perturbation);
        let third = product.dot(&product).dot(&product).diag().sum();
        let central_bar = step * step / 3.0 * third.abs() + value_bar / step;
        let terms: Vec<f64> = (0..3)
            .map(|axis| {
                let vector = main_vectors.column(axis);
                vector.dot(&perturbation.dot(&vector)) / main_values[axis]
            })
            .collect();
        let main = terms.iter().sum::<f64>();
        let main_bar = value_bar * terms.iter().map(|term| term.abs()).sum::<f64>();
        let gap = (derivative - main).abs();
        main_gap = main_gap.max(gap);
        println!(
            "[pencil lane bundle FD] perturbation {index}: bundle {derivative:.15e}, central \
             {central:.15e} (bar {central_bar:.3e}), main's formula {main:.15e}: |Δ| {gap:.3e} \
             (bar {main_bar:.3e})"
        );
        assert!(
            (derivative - central).abs() <= central_bar,
            "perturbation {index}: bundle {derivative:e} against central difference {central:e}"
        );
        assert!(
            gap <= main_bar,
            "perturbation {index}: bundle {derivative:e} against main's formula {main:e}"
        );
    }
    println!("[pencil lane bundle FD] max|Δ| against main's bundle formula {main_gap:.3e}");
}

/// Dense admission is counted per lane: the majorizer lane keeps main's six-block count, bit
/// for bit, so a `UnitDeflation` route's dense admission does not move, and only the exact-A
/// pencil lane counts eight blocks.
#[test]
fn dense_lane_admission_is_counted_per_lane_2933_f07() {
    const MAIN_MAJORIZER_LANE_BLOCKS: usize = 6;
    const PENCIL_LANE_BLOCKS: usize = 8;
    for k in [0usize, 1, 72, 288, 2048] {
        let block = k * k * std::mem::size_of::<f64>();
        assert_eq!(
            dense_lane_reduced_schur_peak_bytes(k),
            Some(MAIN_MAJORIZER_LANE_BLOCKS * block),
            "k = {k}: the majorizer lane's admission is main's"
        );
        assert_eq!(
            dense_lane_exact_a_pencil_peak_bytes(k),
            Some(PENCIL_LANE_BLOCKS * block),
            "k = {k}: the pencil lane's admission"
        );
    }
    assert_eq!(dense_lane_reduced_schur_peak_bytes(usize::MAX), None);
    assert_eq!(dense_lane_exact_a_pencil_peak_bytes(usize::MAX), None);

    // The largest border a route admits at a byte budget. The support route's limit under
    // `dense_lane_reduced_schur_peak_bytes` must equal the one main's six-block count gives.
    let largest_admitted = |budget: usize, peak: &dyn Fn(usize) -> Option<usize>| {
        let (mut low, mut high) = (0usize, 1usize << 24);
        while low < high {
            let mid = low + (high - low).div_ceil(2);
            if peak(mid).is_some_and(|bytes| bytes <= budget) {
                low = mid;
            } else {
                high = mid - 1;
            }
        }
        low
    };
    let main_majorizer_lane = |k: usize| {
        k.checked_mul(k)?
            .checked_mul(std::mem::size_of::<f64>())?
            .checked_mul(MAIN_MAJORIZER_LANE_BLOCKS)
    };
    for budget in [256usize << 20, 4 << 30, 64 << 30] {
        let support = largest_admitted(budget, &dense_lane_reduced_schur_peak_bytes);
        let main = largest_admitted(budget, &main_majorizer_lane);
        let pencil = largest_admitted(budget, &dense_lane_exact_a_pencil_peak_bytes);
        println!(
            "[dense lane admission] budget {budget} bytes: support route admits k ≤ {support} \
             (main {main}); pencil lane k ≤ {pencil}"
        );
        assert_eq!(support, main, "budget {budget}: the support route's admitted limit moved");
    }
}
