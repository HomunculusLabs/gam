use super::*;
use crate::pirls::PirlsWorkspace;

/// A small sparse design may use the dense fast path only while densification
/// costs at most four times the sparse Gram arithmetic.  With density `d`,
/// dense `X'WX` performs approximately `1 / d²` as much multiply-accumulate
/// work as row-wise sparse assembly, so this is the `d >= 1/2` boundary.
///
/// This is deliberately separate from the dense byte budget: fitting in memory
/// does not make multiplying structural zeroes computationally sensible.
const DENSE_FAST_PATH_MIN_DESIGN_DENSITY: f64 = 0.5;

fn dense_fast_path_is_compute_admissible(n_obs: usize, p: usize, nnz_x: usize) -> bool {
    let dense_cells = n_obs.saturating_mul(p);
    if dense_cells == 0 {
        return false;
    }
    (nnz_x as f64) / (dense_cells as f64) >= DENSE_FAST_PATH_MIN_DESIGN_DENSITY
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum GeometryBackendKind {
    DenseSpectral,
    SparseExactSpd,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum HessianEvalStrategyKind {
    SpectralExact,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct HessianStrategyDecision {
    pub(super) strategy: HessianEvalStrategyKind,
}

impl<'a> RemlState<'a> {
    pub(super) fn selecthessian_strategy_policy(
        &self,
        bundle: &EvalShared,
    ) -> HessianStrategyDecision {
        // When the sparse-exact backend produced the PIRLS result, prefer
        // the sparse Hessian path for consistency (avoids dense→sparse
        // round-trip that loses sparsity structure).
        if bundle.backend_kind() == GeometryBackendKind::SparseExactSpd {
            return HessianStrategyDecision {
                strategy: HessianEvalStrategyKind::SpectralExact,
            };
        }
        HessianStrategyDecision {
            strategy: HessianEvalStrategyKind::SpectralExact,
        }
    }

    /// Coefficient count below which a problem is considered "small" for the
    /// dense fast-path: at this width a dense p×p Gram/Hessian is at most a few
    /// hundred KB, so the sparse machinery's overhead is not worth paying.
    pub(crate) const SMALL_P_DENSE_THRESHOLD: usize = 256;

    /// Upper-triangle density of the penalized Hessian above which the sparse
    /// exact-SPD backend loses its advantage and we fall back to dense: once
    /// >10% of entries are nonzero, sparse factorization fill-in and bookkeeping
    /// cost more than a dense Cholesky of the same dimension.
    pub(crate) const SPARSE_HESSIAN_MAX_DENSITY: f64 = 0.10;

    pub(super) fn select_reml_geometry(
        &self,
        rho: &Array1<f64>,
    ) -> Result<SparseRemlDecision, EstimationError> {
        let lambdas =
            Array1::from_vec(gam_problem::checked_exp_log_strengths(rho.iter().copied())?);
        let p = self.p;
        let has_dense_constraints =
            self.linear_constraints.is_some() || self.coefficient_lower_bounds.is_some();
        let x_sparse = self.x.as_sparse();
        let nnz_x = x_sparse.map(|s| s.val().len()).unwrap_or(0);
        let dense_backend =
            |reason: &'static str,
             nnz_h_upper_est: Option<usize>,
             density_h_upper_est: Option<f64>| SparseRemlDecision {
                geometry: RemlGeometry::DenseSpectral,
                reason,
                p,
                nnz_x,
                nnz_h_upper_est,
                density_h_upper_est,
            };

        if self.config.firth_bias_reduction {
            // Route ALL Firth-active fits (Logit or otherwise) through the
            // dense-spectral backend.  The sparse bundle assembly at
            // `prepare_sparse_eval_bundlewithkey` factors
            // `X'WX + S_λ + barrier`, WITHOUT subtracting the Jeffreys
            // Hessian `H_φ`.  The dense path at
            // `prepare_dense_eval_bundlewithkey` (runtime.rs:2175-2194)
            // explicitly subtracts `H_φ` from `h_total` before caching,
            // so `bundle.h_total = X'WX + S_λ − H_φ (+ barrier)`.
            //
            // The `FirthAwareGlmDerivatives` provider returns
            // `dH/dρ = A_k + D_β(X'WX − H_φ)[v_k]`, including the
            // Firth correction term `−D(H_φ)[B_k]`.  If the sparse
            // logdet operator is factored from `X'WX + S_λ` but the
            // derivative provider differentiates `X'WX + S_λ − H_φ`,
            // then `log|H|` and `trace_logdet(dH/dρ)` live on different
            // Hessian surfaces — the REML cost and its ρ-gradient
            // would no longer be consistent (see `reml_laml_evaluate`
            // in unified.rs, where both `log|H|` and
            // `trace_logdet(dH/dρ)` come from the same `hop`, yet
            // `dH/dρ` is assembled via `effective_deriv
            // .hessian_derivative_correction_result(&neg_v_i)`).
            //
            // Subtracting `H_φ` inside the sparse factoring path would
            // require materialising `H_φ` densely (Firth H_φ is a
            // generally-dense p×p matrix — it's defined on the
            // identifiable column-space of `X`), which would destroy
            // the sparsity that justified going sparse in the first
            // place.  Routing to dense is the only cost-gradient-
            // consistent option.
            return Ok(dense_backend("firth_bias_reduction_active", None, None));
        }
        if has_dense_constraints {
            return Ok(dense_backend("constraints_present", None, None));
        }
        let Some(x_sparse) = x_sparse else {
            return Ok(dense_backend("design_not_sparse", None, None));
        };
        let Some(block_count) = self.sparse_penalty_block_count else {
            return Ok(dense_backend("penalty_blocks_not_separable", None, None));
        };
        // Small-problem dense fast-path.  Both memory and arithmetic must make
        // densification admissible: the byte budget prevents an oversized
        // materialization, while the density gate prevents a small but mostly
        // structural-zero design from paying dense Gram work.  Sparse designs
        // rejected by either gate continue to the Hessian-pattern authority
        // below; the early-out never decides sparsity by allocation size alone.
        const SMALL_NP_DENSE_BUDGET: usize = 4_000_000;
        let n_obs = self.y.len();
        if p < Self::SMALL_P_DENSE_THRESHOLD
            && n_obs.saturating_mul(p) < SMALL_NP_DENSE_BUDGET
            && dense_fast_path_is_compute_admissible(n_obs, p, nnz_x)
        {
            return Ok(dense_backend(
                "p_below_threshold_small_and_compute_dense",
                None,
                None,
            ));
        }

        let mut s_lambda = Array2::<f64>::zeros((self.p, self.p));
        for (k, cp) in self.canonical_penalties.iter().enumerate() {
            if k < lambdas.len() && lambdas[k] != 0.0 {
                cp.accumulate_weighted(&mut s_lambda, lambdas[k]);
            }
        }
        let mut workspace = PirlsWorkspace::new(self.y.len(), self.p, 0, 0);
        Ok(
            match workspace.sparse_penalized_system_stats(x_sparse, &s_lambda) {
                Ok(stats)
                    if stats.density_upper < Self::SPARSE_HESSIAN_MAX_DENSITY
                        && block_count > 0 =>
                {
                    SparseRemlDecision {
                        geometry: RemlGeometry::SparseExactSpd,
                        reason: "sparse_exact_spd",
                        p,
                        nnz_x,
                        nnz_h_upper_est: Some(stats.nnz_h_upper),
                        density_h_upper_est: Some(stats.density_upper),
                    }
                }
                Ok(stats) => dense_backend(
                    "penalized_hessian_too_dense",
                    Some(stats.nnz_h_upper),
                    Some(stats.density_upper),
                ),
                Err(_) => dense_backend("sparse_stats_failed", None, None),
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::dense_fast_path_is_compute_admissible;

    #[test]
    fn ancient_dna_shape_cannot_densify_merely_because_its_bytes_fit_2413() {
        let n_obs = 13_897;
        let p = 227;
        let nnz_x = n_obs * 22;

        assert!(n_obs * p < 4_000_000, "fixture must fit the old byte gate");
        assert!(
            !dense_fast_path_is_compute_admissible(n_obs, p, nnz_x),
            "a 9.7%-dense design pays roughly 106x dense Gram arithmetic"
        );
    }

    #[test]
    fn dense_fast_path_compute_gate_has_an_explicit_four_x_work_boundary_2413() {
        assert!(dense_fast_path_is_compute_admissible(100, 20, 1_000));
        assert!(!dense_fast_path_is_compute_admissible(100, 20, 999));
        assert!(!dense_fast_path_is_compute_admissible(0, 20, 0));
        assert!(!dense_fast_path_is_compute_admissible(100, 0, 0));
    }
}
