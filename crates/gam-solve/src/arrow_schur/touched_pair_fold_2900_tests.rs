//! #2900 — the parallel reduced-Schur fold stores only the `(a, b)` pairs its rows
//! touch, and gives the same f64 values as the dense zero-seeded `k×k` chunk partials
//! it replaced, except for the sign of a zero entry.
//!
//! The serial in-place reduction is the control: it associates the reduction sum
//! across chunk boundaries differently, so it must NOT be bit-equal to the chunked
//! fold on this fixture, which shows the equality below can fail.

#![cfg(test)]

use super::*;
use crate::arrow_schur::newton_step::{
    SchurReductionKind, factor_blocks_for_system, subtract_row_schur_contribution,
};
use crate::arrow_schur::reduced_solve::{
    SCHUR_MATVEC_PARALLEL_ROW_MIN, fold_row_chunk_partials, reduce_row_schur_contributions,
};
use crate::arrow_schur::solve_options::{ArrowEvidencePolicy, CpuBatchedBlockSolver};
use ndarray::{Array1, Array2, ArrayView1};
use std::sync::Arc;

/// Rows with `active` of `k` border columns each, `H_tt = 4·I`, and fixed couplings
/// installed through the opaque row operator.
fn sparse_row_system(n: usize, active: usize, k: usize) -> ArrowSchurSystem {
    let mut state = 0x9E37_79B9_7F4A_7C15_u64;
    let mut mark = vec![usize::MAX; k];
    let mut supports: Vec<u32> = Vec::with_capacity(n * active);
    for row in 0..n {
        let first = supports.len();
        while supports.len() - first < active {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let atom = ((state >> 33) % k as u64) as usize;
            if mark[atom] != row {
                mark[atom] = row;
                supports.push(atom as u32);
            }
        }
        supports[first..].sort_unstable();
    }
    let couplings: Vec<f64> = (0..n * active * active)
        .map(|idx| 0.1 * (((idx.wrapping_mul(2_654_435_761)) % 1000) as f64 / 1000.0 - 0.5))
        .collect();
    let supports = Arc::new(supports);
    let couplings = Arc::new(couplings);
    let mut sys =
        ArrowSchurSystem::new_with_per_row_dims_empty_hbb_and_htbeta_cols(vec![active; n], k, 0);
    for row in 0..n {
        for r in 0..active {
            sys.rows[row].htt[[r, r]] = 4.0;
        }
    }
    sys.hbb = Array2::<f64>::eye(k) * 20.0;
    let (forward_supports, forward_couplings) = (Arc::clone(&supports), Arc::clone(&couplings));
    let (transpose_supports, transpose_couplings) = (Arc::clone(&supports), Arc::clone(&couplings));
    // Each row's supports are distinct, so its couplings are exactly the row's entries.
    let row_norm_bounds: Arc<[f64]> = couplings
        .chunks(active * active)
        .map(|row_couplings| frobenius_norm_upper_bound(row_couplings.iter().copied()))
        .collect();
    sys.set_row_htbeta_operator(
        move |row: usize, x: ArrayView1<'_, f64>, out: &mut Array1<f64>| {
            for r in 0..active {
                let mut acc = 0.0_f64;
                for ci in 0..active {
                    acc += forward_couplings[(row * active + r) * active + ci]
                        * x[forward_supports[row * active + ci] as usize];
                }
                out[r] += acc;
            }
        },
        move |row: usize, v: ArrayView1<'_, f64>, out: &mut Array1<f64>| {
            for ci in 0..active {
                let mut acc = 0.0_f64;
                for r in 0..active {
                    acc += transpose_couplings[(row * active + r) * active + ci] * v[r];
                }
                out[transpose_supports[row * active + ci] as usize] += acc;
            }
        },
        // Each apply accumulates `active` terms, one more for the transpose's addition.
        RowHtbetaDeclaration {
            row_norm_bounds,
            apply_depth: active + 1,
        },
    );
    sys
}

fn bit_equal_up_to_zero_sign(left: &Array2<f64>, right: &Array2<f64>) -> bool {
    left.dim() == right.dim() && left.iter().zip(right.iter()).all(|(p, q)| p == q)
}

#[test]
fn touched_pair_fold_matches_the_dense_chunk_partials_up_to_zero_sign_2900() {
    let (n, active, k) = (1024usize, 6usize, 48usize);
    let sys = sparse_row_system(n, active, k);
    let backend = CpuBatchedBlockSolver;
    let factors = factor_blocks_for_system(
        &sys,
        0.0,
        ArrowEvidencePolicy::Strict,
        &backend,
        gam_gpu::GpuPolicy::Off,
    )
    .expect("row factors")
    .factors;
    assert!(
        n >= SCHUR_MATVEC_PARALLEL_ROW_MIN && rayon::current_thread_index().is_none(),
        "the parallel chunk fold must be the route under test"
    );

    for kind in [SchurReductionKind::Direct, SchurReductionKind::SqrtBa] {
        let mut touched = Array2::<f64>::eye(k) * 20.0;
        reduce_row_schur_contributions(
            &sys,
            &factors,
            &backend,
            kind,
            &mut touched,
            gam_gpu::GpuPolicy::Off,
        )
        .expect("touched-pair chunk fold");

        // The route it replaced: one zero-seeded dense `k×k` partial per chunk,
        // folded entry by entry in chunk order.
        let mut dense = Array2::<f64>::eye(k) * 20.0;
        fold_row_chunk_partials(
            n,
            || Array2::<f64>::zeros((k, k)),
            |partial| partial.fill(0.0),
            |i, partial| {
                subtract_row_schur_contribution(
                    &sys,
                    i,
                    &sys.rows[i],
                    factors.factor(i),
                    &backend,
                    kind,
                    partial,
                )
            },
            |partial| {
                for a in 0..k {
                    for b in 0..k {
                        dense[[a, b]] += partial[[a, b]];
                    }
                }
            },
        )
        .expect("dense chunk fold");

        let mut serial = Array2::<f64>::eye(k) * 20.0;
        rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .expect("one-thread pool")
            .install(|| {
                reduce_row_schur_contributions(
                    &sys,
                    &factors,
                    &backend,
                    kind,
                    &mut serial,
                    gam_gpu::GpuPolicy::Off,
                )
            })
            .expect("serial in-place reduction");

        assert!(
            bit_equal_up_to_zero_sign(&touched, &dense),
            "{kind:?}: the touched-pair fold departs from the dense chunk partials"
        );
        assert!(
            !bit_equal_up_to_zero_sign(&serial, &dense),
            "{kind:?}: control failed: the serial reduction is bit-equal to the chunked fold, \
             so bit equality cannot tell the two routes apart on this fixture"
        );
    }
}
