//! Shared coefficient-space Hessian cost models for operator-aware families.
//!
//! Several families (GAMLSS location-scale variants, the Bernoulli and survival
//! marginal-slope kernels, and the conditional transformation model) expose
//! their joint inner Hessian as a row-streaming matrix-free operator. When the
//! joint route is matrix-free
//! ([`crate::custom_family::JointHessianWork::matrix_free_route`]) the honest
//! per-evaluation work is one operator product rather than the `O(n · p²)`
//! dense joint-Hessian assembly modelled by
//! [`crate::custom_family::joint_coupled_coefficient_hessian_cost`].
//!
//! The "report the operator product when the route is matrix-free, else report
//! the dense build" decision was historically copy-pasted across every such
//! family. This module is the single source of truth for that branch so a
//! change to either work figure touches exactly one site.

use crate::custom_family::{JointHessianWork, ParameterBlockSpec};

/// Operator-aware coefficient-space Hessian cost: one operator product when
/// `work` routes the `p_total`-coefficient joint solve matrix-free, the dense
/// build otherwise. Families whose work differs structurally supply their own
/// [`JointHessianWork`] while sharing the branch.
pub(crate) fn operator_aware_hessian_cost(work: JointHessianWork, p_total: u64) -> u64 {
    if work.matrix_free_route(p_total as usize) {
        work.apply
    } else {
        work.build
    }
}

/// Operator-aware coefficient-space Hessian cost for the common joint-coupled
/// case: a row pullback over `n` rows and `p_total = Σ_b p_b` coefficients,
/// whose dense build is the joint-coupled `n · p_total²`.
///
/// This is the shared body for every GAMLSS location-scale variant and the
/// Bernoulli/survival marginal-slope kernels. `n` is the family's observation
/// count (`self.y.len()` / `self.n`); `specs` are the assembled parameter
/// blocks.
pub(crate) fn joint_coupled_operator_aware_hessian_cost(n: u64, specs: &[ParameterBlockSpec]) -> u64 {
    let p_total: u64 = specs
        .iter()
        .map(|s| s.design.ncols() as u64)
        .fold(0u64, |acc, p| acc.saturating_add(p));
    operator_aware_hessian_cost(JointHessianWork::row_pullback(n, p_total), p_total)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::custom_family::{ParameterBlockSpec, joint_coupled_coefficient_hessian_cost};
    use gam_linalg::matrix::{DenseDesignMatrix, DesignMatrix};
    use ndarray::Array2;

    fn spec_with_ncols(p: usize) -> ParameterBlockSpec {
        ParameterBlockSpec {
            design: DesignMatrix::Dense(DenseDesignMatrix::from(Array2::<f64>::zeros((1, p)))),
            ..ParameterBlockSpec::defaults()
        }
    }

    #[test]
    fn operator_aware_picks_dense_for_small_problem() {
        // p = 4 coefficients over n = 100 rows is well inside the p ≤ 3n dense turn.
        let (p, n) = (4u64, 100u64);
        let work = JointHessianWork::row_pullback(n, p);
        assert!(!work.matrix_free_route(p as usize));
        assert_eq!(operator_aware_hessian_cost(work, p), n * p * p);
    }

    #[test]
    fn operator_aware_picks_matrix_free_for_wide_problem() {
        // p = 512 coefficients over one row is far past the p ≤ 3n dense turn.
        let (p, n) = (512u64, 1u64);
        let work = JointHessianWork::row_pullback(n, p);
        assert!(work.matrix_free_route(p as usize));
        assert_eq!(operator_aware_hessian_cost(work, p), 2 * n * p);
    }

    #[test]
    fn operator_aware_branch_is_exactly_the_gate() {
        // A row pullback's route turns at p = 3n: across a grid spanning the turn
        // the cost is the build inside it and one operator product past it.
        for &p in &[1u64, 64, 128, 299, 300, 301, 1024] {
            for &n in &[1u64, 100, 50_000] {
                let expected = if p > 3 * n { 2 * n * p } else { n * p * p };
                assert_eq!(
                    operator_aware_hessian_cost(JointHessianWork::row_pullback(n, p), p),
                    expected,
                    "p={p} n={n}"
                );
            }
        }
    }

    #[test]
    fn joint_coupled_small_returns_dense_n_p_squared() {
        // Small problem -> dense branch -> n * p_total^2, summed over block ncols.
        let specs = [spec_with_ncols(3), spec_with_ncols(5)];
        let n = 10u64;
        let p_total = 8u64; // 3 + 5
        assert!(p_total <= 3 * n);
        assert_eq!(
            joint_coupled_operator_aware_hessian_cost(n, &specs),
            joint_coupled_coefficient_hessian_cost(n, &specs)
        );
        assert_eq!(
            joint_coupled_operator_aware_hessian_cost(n, &specs),
            n * p_total * p_total
        );
    }

    #[test]
    fn joint_coupled_wide_returns_one_operator_product() {
        // p_total = 600 over n = 4 rows is past p ≤ 3n -> one product, 2 * n * p_total.
        let specs = [spec_with_ncols(300), spec_with_ncols(300)];
        let n = 4u64;
        let p_total = 600u64;
        assert!(p_total > 3 * n);
        assert_eq!(
            joint_coupled_operator_aware_hessian_cost(n, &specs),
            2 * n * p_total
        );
    }

    #[test]
    fn joint_coupled_empty_specs_is_zero() {
        // No blocks => p_total == 0 => both candidate costs are 0.
        let specs: [ParameterBlockSpec; 0] = [];
        assert_eq!(joint_coupled_operator_aware_hessian_cost(1234, &specs), 0);
    }

    #[test]
    fn joint_coupled_p_total_sums_block_ncols() {
        // The dense cost must reflect the summed ncols across all blocks, not a
        // single block: three blocks of 2 columns => p_total = 6.
        let specs = [spec_with_ncols(2), spec_with_ncols(2), spec_with_ncols(2)];
        let n = 7u64;
        let p_total = 6u64;
        assert!(p_total <= 3 * n);
        assert_eq!(
            joint_coupled_operator_aware_hessian_cost(n, &specs),
            n * p_total * p_total
        );
    }
}
