#![cfg(test)]
//! #2933 F27 S3: an exact-A classification row carries its cross correction on its
//! own border columns.
//!
//! The support-sparse lane's correction `A_tβ − B_tβ = −r_o·∂_aφ_m` touches only a
//! row's active decoder blocks. One column set shared by every row would make each
//! row carry the whole border, `O(N·q·K)` per classified direction. The references
//! below are assembled densely in the test from the raw operands, independently of
//! the carrier walk.

use super::*;
use ndarray::array;

const K: usize = 4;
const DELTA_TT: [f64; 2] = [-0.7, 0.4];
const CLAMP: [f64; 2] = [0.2, 0.0];

/// `(A_tt, A_tβ)` of each one-coordinate row.
fn a_rows() -> [(f64, [f64; K]); 2] {
    [(3.0, [0.5, -1.0, 0.25, 2.0]), (2.5, [1.5, 0.0, -0.6, 0.8])]
}

/// Row 0 corrects border columns 1 and 3, row 1 columns 0 and 2.
fn sparse_carriers() -> [(Vec<usize>, Vec<f64>); 2] {
    [(vec![1, 3], vec![0.3, -0.4]), (vec![0, 2], vec![-0.9, 0.35])]
}

fn exact_a_system(carriers: &[(Vec<usize>, Vec<f64>); 2]) -> ArrowSchurSystem {
    let mut system = ArrowSchurSystem::new(2, 1, K);
    for (row, (a_tt, a_tbeta)) in a_rows().iter().enumerate() {
        system.rows[row].htt[[0, 0]] = *a_tt;
        for column in 0..K {
            system.rows[row].htbeta[[0, column]] = a_tbeta[column];
        }
    }
    for column in 0..K {
        system.hbb[[column, column]] = 4.0 + column as f64;
    }
    system.hbb[[0, 3]] = 0.5;
    system.hbb[[3, 0]] = 0.5;
    system.exact_a_classification = Some(ExactAClassificationGeometry {
        rows: carriers
            .iter()
            .enumerate()
            .map(|(row, (columns, values))| ExactAClassificationRow {
                delta_tt: array![[DELTA_TT[row]]],
                delta_tbeta: Array2::from_shape_vec((1, values.len()), values.clone())
                    .expect("one carrier row per latent coordinate"),
                border_columns: columns.clone().into(),
                clamp_diag: array![CLAMP[row]],
            })
            .collect::<Vec<_>>()
            .into(),
        border_remainder: None,
    });
    system
}

/// `B_raw` cross row `A_tβ − ΔC_tβ` on the whole border.
fn dense_b_tbeta(row: usize) -> [f64; K] {
    let (_, mut b_tbeta) = a_rows()[row];
    let (columns, values) = &sparse_carriers()[row];
    for (column, value) in columns.iter().zip(values.iter()) {
        b_tbeta[*column] -= value;
    }
    b_tbeta
}

/// The exact-A Schur graph `t(β) = −A_tt⁻¹ A_tβ β` of one row, as a border covector.
fn graph(row: usize) -> [f64; K] {
    let (a_tt, a_tbeta) = a_rows()[row];
    a_tbeta.map(|entry| -entry / a_tt)
}

#[test]
fn exact_a_row_carrier_names_its_own_border_columns_2933_f27() {
    let sparse = exact_a_system(&sparse_carriers());
    let factors = CpuBatchedBlockSolver
        .factor_blocks(&sparse.rows, 0.0, sparse.d, true)
        .expect("positive row blocks factor");
    let direction = array![0.3, -1.2, 0.7, 0.45];

    let mut expected_majorizer = direction.dot(&sparse.hbb.dot(&direction));
    let mut expected_clamp = 0.0_f64;
    let mut expected_metric = sparse.hbb.clone();
    let mut expected_clamp_metric = Array2::<f64>::zeros((K, K));
    for row in 0..2 {
        let g = graph(row);
        let b_tbeta = dense_b_tbeta(row);
        let b_tt = a_rows()[row].0 - DELTA_TT[row];
        let t = g.iter().zip(direction.iter()).map(|(gc, dc)| gc * dc).sum::<f64>();
        let cross = b_tbeta.iter().zip(direction.iter()).map(|(bc, dc)| bc * dc).sum::<f64>();
        expected_majorizer += b_tt * t * t + 2.0 * t * cross;
        expected_clamp += CLAMP[row] * t * t;
        for i in 0..K {
            for j in 0..K {
                expected_metric[[i, j]] += b_tt * g[i] * g[j] + g[i] * b_tbeta[j] + b_tbeta[i] * g[j];
                expected_clamp_metric[[i, j]] += CLAMP[row] * g[i] * g[j];
            }
        }
    }

    let (majorizer, clamp) =
        exact_a_reduced_direction_metrics(&sparse, &factors, 0.0, direction.view())
            .expect("sparse carrier classifies a direction");
    assert!(
        (majorizer - expected_majorizer).abs() <= 1.0e-12 * (1.0 + expected_majorizer.abs()),
        "v'B_raw v {majorizer:.15e} != dense reference {expected_majorizer:.15e}"
    );
    assert!(
        (clamp - expected_clamp).abs() <= 1.0e-12 * (1.0 + expected_clamp.abs()),
        "v'E v {clamp:.15e} != dense reference {expected_clamp:.15e}"
    );

    let classification = exact_a_reduced_classification(&sparse, &factors)
        .expect("sparse carrier classifies the reduced border")
        .expect("the carrier is installed");
    for i in 0..K {
        for j in 0..K {
            assert!(
                (classification.majorizer_metric[[i, j]] - expected_metric[[i, j]]).abs()
                    <= 1.0e-12 * (1.0 + expected_metric[[i, j]].abs()),
                "majorizer metric [{i},{j}] {:.15e} != dense reference {:.15e}",
                classification.majorizer_metric[[i, j]],
                expected_metric[[i, j]]
            );
            assert!(
                (classification.clamp_metric[[i, j]] - expected_clamp_metric[[i, j]]).abs()
                    <= 1.0e-12 * (1.0 + expected_clamp_metric[[i, j]].abs()),
                "clamp metric [{i},{j}] {:.15e} != dense reference {:.15e}",
                classification.clamp_metric[[i, j]],
                expected_clamp_metric[[i, j]]
            );
        }
    }

    // The same correction zero-padded onto the whole border classifies identically.
    let padded = sparse_carriers().map(|(columns, values)| {
        let mut full = vec![0.0_f64; K];
        for (column, value) in columns.iter().zip(values.iter()) {
            full[*column] = *value;
        }
        ((0..K).collect::<Vec<_>>(), full)
    });
    let (padded_majorizer, padded_clamp) =
        exact_a_reduced_direction_metrics(&exact_a_system(&padded), &factors, 0.0, direction.view())
            .expect("full-border carrier classifies a direction");
    assert!(
        (padded_majorizer - majorizer).abs() <= 1.0e-13 * (1.0 + majorizer.abs())
            && (padded_clamp - clamp).abs() <= 1.0e-13 * (1.0 + clamp.abs()),
        "zero-padded carrier ({padded_majorizer:.15e}, {padded_clamp:.15e}) != sparse \
         ({majorizer:.15e}, {clamp:.15e})"
    );

    // Free to disagree: the same values on swapped columns are another operator.
    let swapped = [(vec![3, 1], vec![0.3, -0.4]), (vec![0, 2], vec![-0.9, 0.35])];
    let (swapped_majorizer, _) =
        exact_a_reduced_direction_metrics(&exact_a_system(&swapped), &factors, 0.0, direction.view())
            .expect("swapped carrier classifies a direction");
    assert!(
        (swapped_majorizer - majorizer).abs() > 1.0e-2,
        "carrier columns must decide which border entries are corrected: swapped \
         {swapped_majorizer:.15e} vs {majorizer:.15e}"
    );
}
