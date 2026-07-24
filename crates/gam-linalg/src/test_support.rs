//! Low-level numerical test fixtures shared across the workspace.
//!
//! The operator fixtures exercise `gam-linalg`'s design machinery, so they live
//! here — the crate that owns [`LinearOperator`], [`DesignMatrix`], and friends
//! — rather than in a downstream crate. Following the workspace convention for
//! `test_support` modules (and matching the root `gam` crate), this is a plain
//! always-compiled `pub mod`: feature gates and `#[cfg(test)]` module gates are
//! banned here, and a `cfg(test)` module would be invisible to downstream
//! crates' test builds anyway. The contents are `pub`, so they are reachable
//! (no dead-code lint) yet only ever called from `#[cfg(test)]` code.
//!
//! [`fd_checker`] carries the same argument one level down: a central-difference
//! derivative check is `ndarray` in, `ndarray` out and owns no model-layer type,
//! so it belongs to the leaf that owns the dense-array seam. Keeping it here
//! means a crate cross-checking an analytic derivative pulls one leaf dependency
//! it already has instead of the whole model layer.
//!
//! [`paired_holdout_partition`] is likewise model-free deterministic numerical
//! test design. Keeping its row-ranking primitive here lets downstream quality
//! tests share one paired partition without making its invariant test compile
//! the reference bridge and the entire model stack.

pub mod fd_checker;

use crate::matrix::{DenseDesignMatrix, DenseDesignOperator, DesignMatrix, LinearOperator};
use gam_runtime::resource::MatrixMaterializationError;
use ndarray::{Array1, Array2, Axis, s};
use std::ops::Range;
use std::sync::Arc;

/// One exact-cardinality train/test partition for a paired quality comparison.
#[derive(Clone, Debug, PartialEq)]
pub struct PairedHoldout {
    /// Source-row indices used to fit both implementations.
    pub train: Vec<usize>,
    /// Source-row indices scored for both implementations.
    pub test: Vec<usize>,
    /// Full-width external-tool mask: `1.0` for test, `0.0` for train.
    pub mask: Vec<f64>,
}

/// Build a reproducible, exact-cardinality paired holdout partition.
///
/// Rows are ranked by a SplitMix64 score keyed by `split_key`; the lowest
/// `round(n * holdout_fraction)` scores form the test set. Ranking instead of
/// thresholding a pseudo-random value keeps every split the same size. That
/// matters when per-split metrics are averaged: every term then represents the
/// same amount of held-out evidence, and small fixtures cannot accidentally
/// produce a degenerate fold.
///
/// The returned row indices retain source order. A quality test must use the
/// same returned partition for both the implementation under test and its
/// reference tool.
pub fn paired_holdout_partition(
    n: usize,
    holdout_fraction: f64,
    split_key: u64,
) -> PairedHoldout {
    assert!(n >= 2, "paired holdout needs at least two rows, got {n}");
    assert!(
        holdout_fraction.is_finite() && 0.0 < holdout_fraction && holdout_fraction < 1.0,
        "paired holdout fraction must be finite and strictly between zero and one, got {holdout_fraction}"
    );

    let test_len = (n as f64 * holdout_fraction).round() as usize;
    assert!(
        0 < test_len && test_len < n,
        "paired holdout fraction {holdout_fraction} yields {test_len} test rows for n={n}"
    );

    const GOLDEN_RATIO: u64 = 0x9E3779B97F4A7C15;
    let score = |row: usize| {
        let mut z = (row as u64)
            .wrapping_add(split_key.wrapping_mul(GOLDEN_RATIO))
            .wrapping_add(GOLDEN_RATIO);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    };

    let mut ranked: Vec<(u64, usize)> = (0..n).map(|row| (score(row), row)).collect();
    ranked.sort_unstable();

    let mut held_out = vec![false; n];
    for &(_, row) in &ranked[..test_len] {
        held_out[row] = true;
    }

    let train = (0..n).filter(|&row| !held_out[row]).collect();
    let test = (0..n).filter(|&row| held_out[row]).collect();
    let mask = held_out
        .into_iter()
        .map(|is_test| if is_test { 1.0 } else { 0.0 })
        .collect();
    PairedHoldout { train, test, mask }
}

/// A dense-backed [`LinearOperator`] that refuses to materialize itself.
///
/// It services every operator-aware code path (`apply`, `apply_transpose`,
/// `row_chunk_into`, `diag_xtw_x`) but panics from [`to_dense`](DenseDesignOperator::to_dense).
/// Wrapping a design in this fixture turns "a code path densified when it should
/// have stayed lazy" — the regression we guard against — into a hard test
/// failure instead of a silent, slow correctness-preserving fallback.
#[derive(Clone)]
struct NoDensifyOperator {
    dense: Array2<f64>,
}

impl LinearOperator for NoDensifyOperator {
    fn nrows(&self) -> usize {
        self.dense.nrows()
    }

    fn ncols(&self) -> usize {
        self.dense.ncols()
    }

    fn apply(&self, vector: &Array1<f64>) -> Array1<f64> {
        self.dense.dot(vector)
    }

    fn apply_transpose(&self, vector: &Array1<f64>) -> Array1<f64> {
        self.dense.t().dot(vector)
    }

    fn diag_xtw_x(&self, weights: &Array1<f64>) -> Result<Array2<f64>, String> {
        if weights.len() != self.nrows() {
            return Err(format!(
                "NoDensifyOperator weight length mismatch: weights={}, nrows={}",
                weights.len(),
                self.nrows()
            ));
        }
        let weighted = &self.dense * &weights.view().insert_axis(Axis(1));
        Ok(self.dense.t().dot(&weighted))
    }
}

impl DenseDesignOperator for NoDensifyOperator {
    fn row_chunk_into(
        &self,
        rows: Range<usize>,
        mut out: ndarray::ArrayViewMut2<'_, f64>,
    ) -> Result<(), MatrixMaterializationError> {
        out.assign(&self.dense.slice(s![rows, ..]));
        Ok(())
    }

    fn to_dense(&self) -> Array2<f64> {
        // `NoDensifyOperator` is a test fixture asserting that
        // operator-aware code paths never densify.
        // SAFETY: a call here means a code path under test bypassed
        // `row_chunk_into` and tried to materialize — the regression
        // this fixture is designed to catch.
        panic!("NoDensifyOperator must stay lazy")
    }
}

/// Build an operator-backed [`DesignMatrix`] from a dense array that will panic
/// if any consumer tries to densify it. See [`NoDensifyOperator`].
pub fn no_densify_design(dense: Array2<f64>) -> DesignMatrix {
    DesignMatrix::from(DenseDesignMatrix::from(Arc::new(NoDensifyOperator {
        dense,
    })))
}

#[cfg(test)]
mod tests {
    use super::{no_densify_design, paired_holdout_partition};
    use ndarray::array;

    /// Regression guard for #1566: `no_densify_design` must live in `gam-linalg`
    /// (the crate that owns the operator traits) and yield an operator-backed
    /// design that services the lazy paths without ever materializing. If the
    /// fixture is dropped or moved back out of this crate, this test stops
    /// compiling in the very lib-test phase the issue was about.
    #[test]
    fn no_densify_design_is_operator_backed_and_stays_lazy() {
        let design = no_densify_design(array![[1.0, 2.0], [3.0, 4.0]]);
        assert!(design.as_dense_ref().is_none(), "must not be materialized");
        assert!(!design.is_materialized_dense());
        assert!(design.is_operator_backed());
        assert_eq!(design.nrows(), 2);
        assert_eq!(design.ncols(), 2);

        // Operator-aware paths still work: y = X·β and lazy row chunks.
        let beta = array![1.0, -1.0];
        let got = design.dot(&beta);
        assert!((got[0] - (-1.0)).abs() < 1e-12); // 1·1 + 2·(-1)
        assert!((got[1] - (-1.0)).abs() < 1e-12); // 3·1 + 4·(-1)
        let chunk = design
            .try_row_chunk(0..2)
            .expect("row chunk must stay lazy, not densify");
        assert_eq!(chunk, array![[1.0, 2.0], [3.0, 4.0]]);
    }

    /// The whole point of the fixture: any code path that tries to collapse it to
    /// a dense matrix trips a hard panic, turning a silent densification
    /// regression into a test failure.
    #[test]
    #[should_panic(expected = "operator-backed design")]
    fn no_densify_design_rejects_materialization() {
        let design = no_densify_design(array![[1.0, 2.0], [3.0, 4.0]]);
        design.as_dense_cow();
    }

    #[test]
    fn paired_holdout_is_exact_reproducible_and_partitioned() {
        let first = paired_holdout_partition(221, 0.20, 17);
        let replay = paired_holdout_partition(221, 0.20, 17);
        let other = paired_holdout_partition(221, 0.20, 18);

        assert_eq!(first, replay);
        assert_ne!(first.test, other.test);
        assert_eq!(first.test.len(), 44);
        assert_eq!(first.train.len(), 177);
        assert_eq!(first.mask.len(), 221);

        let mut memberships = vec![0usize; 221];
        for &row in &first.train {
            memberships[row] += 1;
            assert_eq!(first.mask[row], 0.0);
        }
        for &row in &first.test {
            memberships[row] += 1;
            assert_eq!(first.mask[row], 1.0);
        }
        assert!(memberships.into_iter().all(|count| count == 1));
    }
}
