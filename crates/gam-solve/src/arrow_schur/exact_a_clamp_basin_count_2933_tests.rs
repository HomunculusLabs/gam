#![cfg(test)]
//! #2933 F27 S3: the matrix-free evidence evaluation reports how many directions the
//! shared exact-A classifier priced at their clamp basin.
//!
//! A basin price moves with the fitted state through the clamp and its direction, so a
//! derivative that holds the conditioning fixed is exact only when nothing was priced
//! there. The support-sparse criterion reads this count to decide whether it may return
//! its gradient. Each fixture's verdict is known by construction: a one-coordinate row
//! against a one-column border, where `S_A = A_ββ − A_tβ²/A_tt` and the row's own
//! curvature are available in closed form.

use super::*;
use ndarray::array;

fn options() -> ArrowSolveOptions {
    ArrowSolveOptions::direct()
        .with_newton_schur_tikhonov(SPECTRAL_DEFLATION_REL_FLOOR)
        .with_indefinite_refusing_evidence_unit_deflation(SPECTRAL_DEFLATION_REL_FLOOR)
}

fn lane() -> SurrogateLaneState {
    SurrogateLaneState::new(SurrogateLaneConfig {
        num_probes: 4,
        seed: 0x2933,
        rel_tol: 1.0e-10,
        cg_rel_tol: 1.0e-12,
        deflation_subspace_iters: 1,
        deflation_target_std_err_rel: 1.0,
    })
}

/// `A_tt = a_tt`, `A_tβ = 1`, `A_ββ = a_bb`, with `ΔC_tt = delta_tt` and clamp `clamp`.
fn exact_a_system(a_tt: f64, a_bb: f64, delta_tt: f64, clamp: f64) -> ArrowSchurSystem {
    let mut system = ArrowSchurSystem::new(1, 1, 1);
    system.rows[0].htt[[0, 0]] = a_tt;
    system.rows[0].htbeta[[0, 0]] = 1.0;
    system.hbb[[0, 0]] = a_bb;
    system.exact_a_classification = Some(ExactAClassificationGeometry {
        rows: vec![ExactAClassificationRow {
            delta_tt: array![[delta_tt]],
            delta_tbeta: Array2::<f64>::zeros((1, 0)),
            border_columns: Arc::from([] as [usize; 0]),
            clamp_diag: array![clamp],
        }]
        .into(),
        border_remainder: None,
    });
    system
}

fn evaluate(system: &ArrowSchurSystem, dense_admitted: bool) -> MatrixFreeArrowEvidenceEvaluation {
    matrix_free_arrow_evidence_evaluation(
        system,
        0.0,
        0.0,
        &options(),
        4,
        1,
        0x2933,
        &mut lane(),
        dense_admitted,
    )
    .expect("the fixture is priced, not refused")
}

#[test]
fn evidence_evaluation_counts_clamp_basin_prices_2933_f27() {
    for dense_admitted in [true, false] {
        // Positive-definite control: `S_A = 2 − 1/1 = 1`, and nothing is priced.
        let control = evaluate(&exact_a_system(1.0, 2.0, 0.0, 0.0), dense_admitted);
        assert_eq!(
            control.exact_a_clamp_basin_directions, 0,
            "dense_admitted={dense_admitted}: a positive-definite exact A prices no basin"
        );
        assert!(
            control.log_det_schur.abs() <= 1.0e-7,
            "dense_admitted={dense_admitted}: control log|S_A| = {:.12e}, expected 0",
            control.log_det_schur
        );

        // Reduced-Schur basin, the #2515 fixture: `S_A = 0.5 − 1 = −0.5`. The lifted
        // direction has majorizer curvature 1.5 and clamp 2, so its basin price is 1.5.
        let reduced = evaluate(&exact_a_system(1.0, 0.5, -2.0, 2.0), dense_admitted);
        assert_eq!(
            reduced.exact_a_clamp_basin_directions, 1,
            "dense_admitted={dense_admitted}: the negative reduced direction is one basin price"
        );
        assert!(
            (reduced.log_det_schur - 1.5_f64.ln()).abs() <= 1.0e-7,
            "dense_admitted={dense_admitted}: basin log|S_A| = {:.12e}, expected ln 1.5",
            reduced.log_det_schur
        );

        // Row basin: `A_tt = −0.5`, `B_tt = 1.5`, clamp 1, so the row is priced at 0.5. The
        // reduced Schur is then `5 − 1/0.5 = 3`, resolved positive.
        let row = evaluate(&exact_a_system(-0.5, 5.0, -2.0, 1.0), dense_admitted);
        assert_eq!(
            row.exact_a_clamp_basin_directions, 1,
            "dense_admitted={dense_admitted}: the negative row direction is one basin price"
        );
        assert!(
            (row.log_det_tt - 0.5_f64.ln()).abs() <= 1.0e-12,
            "dense_admitted={dense_admitted}: row log|A_tt| = {:.12e}, expected ln 0.5",
            row.log_det_tt
        );
        assert!(
            (row.log_det_schur - 3.0_f64.ln()).abs() <= 1.0e-7,
            "dense_admitted={dense_admitted}: row-fixture log|S_A| = {:.12e}, expected ln 3",
            row.log_det_schur
        );
    }
}
