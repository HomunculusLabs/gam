use super::*;
use crate::pirls::PirlsWorkspace;

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

        let mut s_lambda = Array2::<f64>::zeros((self.p, self.p));
        for (k, cp) in self.canonical_penalties.iter().enumerate() {
            if k < lambdas.len() && lambdas[k] != 0.0 {
                cp.accumulate_weighted(&mut s_lambda, lambdas[k]);
            }
        }
        let mut workspace = PirlsWorkspace::new(self.y.len(), self.p);
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
    use super::*;
    // The canonical Gaussian-identity fixture spec is owned by the REML module's
    // own test tree. `use super::*` reaches the REML module itself, which does
    // not re-export its `#[cfg(test)] mod tests` items, so the helper has to be
    // named explicitly — by relative path, because this file is `#[path]`-
    // included and therefore has no stable absolute module path.
    use super::super::tests::gaussian_identity_glm_spec;
    use faer::sparse::{SparseColMat, Triplet};
    use ndarray::{Array1, Array2, array};

    #[test]
    fn small_sparse_design_routes_by_penalized_hessian_structure_2413() {
        let n = 8;
        let p = 4;
        let triplets: Vec<_> = (0..n)
            .flat_map(|row| {
                (0..p)
                    .filter(move |&col| col != row % p)
                    .map(move |col| Triplet::new(row, col, 1.0))
            })
            .collect();
        let x = SparseColMat::try_new_from_triplets(n, p, &triplets)
            .expect("sparse design should build");
        let y = Array1::<f64>::zeros(n);
        let weights = Array1::<f64>::ones(n);
        let offset = Array1::<f64>::zeros(n);
        let config = RemlConfig::external(gaussian_identity_glm_spec(), 1e-8, false);
        let penalty = gam_terms::construction::CanonicalPenalty::from_dense_root(Array2::eye(p), p);
        let state = RemlState::newwith_offset(
            y.view(),
            x,
            weights.view(),
            offset.view(),
            vec![penalty],
            p,
            &config,
            Some(vec![0]),
            None,
            None,
        )
        .expect("REML state should build");

        let decision = state
            .select_reml_geometry(&array![0.0])
            .expect("geometry decision should succeed");

        assert!(matches!(decision.geometry, RemlGeometry::DenseSpectral));
        assert_eq!(decision.reason, "penalized_hessian_too_dense");
        assert_eq!(decision.nnz_x, triplets.len());
        assert!(
            decision
                .density_h_upper_est
                .is_some_and(|density| density > RemlState::SPARSE_HESSIAN_MAX_DENSITY),
            "the decision must come from measured Hessian structure"
        );
        assert!(decision.nnz_h_upper_est.is_some());
    }
}
