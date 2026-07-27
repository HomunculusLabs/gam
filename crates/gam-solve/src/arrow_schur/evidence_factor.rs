//! Evidence factorization of a bordered-arrow system — the operation that is
//! NOT a Newton solve.
//!
//! A REML/LAML criterion needs three things from the joint Hessian at a
//! stationary inner state: the undamped per-row Cholesky factors, the joint
//! log-determinant `log|H| = Σ_i log|H_tt^(i)| + log|S|`, and an operator it can
//! apply `H⁻¹` with. It needs no step, no right-hand side, no trust region and
//! no preconditioner ladder.
//!
//! Every caller that wanted this used to ask
//! [`solve_arrow_newton_step_with_options`] and throw the step away. That is not
//! merely wasteful — it is *wrong at the border*:
//!
//! * `ArrowSolverMode::InexactPCG` deliberately never forms a dense `k × k`
//!   reduced-Schur factor, so the cache it returns carries
//!   `schur_factor = None` and `schur_factor_is_undamped = false`. With `k > 0`
//!   that makes [`ArrowFactorCache::arrow_log_det`] return `None` and
//!   [`matrix_free_arrow_inverse_apply`] refuse the cache outright — the
//!   criterion cannot evaluate at all.
//! * The one route that DID produce a matrix-free `log|S|` lived inside
//!   `try_device_arrow_direct`, i.e. behind a CUDA device. On a CPU-only host
//!   the matrix-free evidence log-determinant simply did not exist (#2576,
//!   sibling of #2573).
//! * The discarded step still paid for a `JacobiPreconditioner` build, which at
//!   the massive-`K` SAE border is `O(n·K)` — minutes of the criterion's
//!   wall-clock spent producing a vector nobody reads.
//!
//! This module names the operation instead. [`factor_arrow_evidence_rows`]
//! performs exactly the row-side work — the undamped per-row factorization
//! under the evidence policy, `O(Σ_i d_i³)` — and returns the row log-determinant
//! together with the factors. The BORDER log-determinant is deliberately the
//! caller's: `log|S|` admits several routes (exact dense Cholesky at small `k`,
//! matrix-free Stochastic Lanczos Quadrature, the frozen rational surrogate
//! whose derivative is exact), and which one a criterion may use is a property
//! of that criterion — a criterion that also needs `∂log|S|/∂ρ` must pick the
//! route whose value it can differentiate. [`ArrowEvidenceRowFactorization::
//! into_cache`] then seals the two halves into one [`ArrowFactorCache`] that
//! reports itself as undamped evidence, so the matrix-free inverse/trace
//! primitives accept it.

use super::*;

/// Undamped per-row evidence factorization of a bordered-arrow system: every
/// part of `log|H|` and of the `H⁻¹` operator that does NOT involve the border.
pub struct ArrowEvidenceRowFactorization {
    /// Per-row lower-triangular Cholesky factors of the UNDAMPED `H_tt^(i)`,
    /// produced under the request's evidence policy (so a gauge/spectral null is
    /// pinned to unit stiffness, contributing `log 1 = 0`).
    pub htt_factors: ArrowFactorSlab,
    /// `Σ_i log|H_tt^(i)|` over those factors.
    pub row_log_det: f64,
    /// Row-local gauge/spectral deflation metadata the outer ρ/θ traces need to
    /// restrict to the kept subspace.
    pub gauge_deflated_directions: usize,
    pub deflated_row_directions: Vec<Vec<Array1<f64>>>,
    pub deflation_row_spectra: Vec<Option<RowDeflationSpectrum>>,
}

/// Factor `sys` for EVIDENCE: undamped per-row Cholesky under the request's
/// evidence policy, and nothing else.
///
/// No step is solved, no right-hand side is consulted, no reduced-Schur
/// preconditioner is built and no trust region is applied. Cost is
/// `O(Σ_i d_i³)`, independent of the border width `k`.
///
/// The returned `row_log_det` is `Σ_i Σ_j 2·log L_ii[j,j]`. A row factor with a
/// non-positive or non-finite pivot is a genuine infeasibility signal (the same
/// one [`probe_undamped_evidence_row_factors`] surfaces), so it is refused here
/// rather than folded into a `NaN` log-determinant downstream.
pub fn factor_arrow_evidence_rows(
    sys: &ArrowSchurSystem,
    options: &ArrowSolveOptions,
) -> Result<ArrowEvidenceRowFactorization, ArrowSchurError> {
    if options.streaming_chunk_size.is_some() {
        return Err(ArrowSchurError::SchurFactorFailed {
            reason: "streaming Arrow-Schur assembly does not materialize the per-row factors an \
                     evidence factorization is made of"
                .to_string(),
        });
    }
    if !sys.cross_row_penalties.is_empty() {
        return Err(ArrowSchurError::SchurFactorFailed {
            reason: "cross-row latent curvature is not row-block-diagonal, so its joint Hessian \
                     has no per-row evidence factorization; that system needs its own \
                     matrix-free evidence carrier"
                .to_string(),
        });
    }
    let backend = CpuBatchedBlockSolver;
    let factored = factor_blocks_for_system(
        sys,
        0.0,
        options.evidence_policy.factors_undamped_evidence(),
        &backend,
        options.gpu_policy,
    )?;
    let mut row_log_det = 0.0_f64;
    for factor in factored.factors.iter() {
        for index in 0..factor.nrows() {
            let pivot = factor[[index, index]];
            if !(pivot.is_finite() && pivot > 0.0) {
                return Err(ArrowSchurError::SchurFactorFailed {
                    reason: format!(
                        "undamped evidence row factor carries a non-positive pivot {pivot}; \
                         the joint Hessian is not positive definite at this state"
                    ),
                });
            }
            row_log_det += 2.0 * pivot.ln();
        }
    }
    Ok(ArrowEvidenceRowFactorization {
        htt_factors: factored.factors,
        row_log_det,
        gauge_deflated_directions: factored.gauge_deflated_directions,
        deflated_row_directions: factored.deflated_row_directions,
        deflation_row_spectra: factored.deflation_row_spectra,
    })
}

impl ArrowEvidenceRowFactorization {
    /// Seal the row half and a caller-computed border log-determinant into one
    /// evidence [`ArrowFactorCache`].
    ///
    /// `schur_log_det` is `log|S|` for `S = H_ββ − Σ_i H_βt^(i)(H_tt^(i))⁻¹
    /// H_tβ^(i)`, by whatever route the caller's criterion can also
    /// differentiate. For `k == 0` there is no border and the argument must be
    /// zero.
    ///
    /// The cache reports `schur_factor_is_undamped = true` with
    /// `schur_factor = None`: it holds no dense border factor at all, and every
    /// piece it does hold is the undamped evidence one. That is exactly the
    /// predicate the matrix-free inverse/trace primitives test — they never read
    /// a dense factor, they need to know the ROW factors and the operator agree
    /// on being undamped evidence.
    pub fn into_cache(
        self,
        sys: &ArrowSchurSystem,
        schur_log_det: f64,
    ) -> Result<ArrowFactorCache, ArrowSchurError> {
        if !schur_log_det.is_finite() {
            return Err(ArrowSchurError::SchurFactorFailed {
                reason: format!(
                    "evidence reduced-Schur log-determinant must be finite; got {schur_log_det}"
                ),
            });
        }
        if sys.k == 0 && schur_log_det != 0.0 {
            return Err(ArrowSchurError::SchurFactorFailed {
                reason: format!(
                    "an arrow system with no border has no reduced Schur; got \
                     log|S| = {schur_log_det}"
                ),
            });
        }
        let htbeta_estimated_bytes =
            estimated_htbeta_bytes(sys.rows.len(), sys.d, sys.k).unwrap_or(usize::MAX);
        let htbeta = if let Some(op) = sys.htbeta_matvec.as_ref() {
            let transpose_op = sys.htbeta_transpose_matvec.as_ref().ok_or_else(|| {
                ArrowSchurError::SchurFactorFailed {
                    reason: "matrix-free H_tbeta evidence cache requires the sparse transpose \
                             installed by ArrowSchurSystem::set_row_htbeta_operator"
                        .to_string(),
                }
            })?;
            ArrowHtbetaCache::Matvec {
                op: Arc::clone(op),
                transpose_op: Arc::clone(transpose_op),
                estimated_bytes: htbeta_estimated_bytes,
            }
        } else if htbeta_estimated_bytes <= ARROW_FACTOR_CACHE_HTBETA_BUDGET_BYTES {
            ArrowHtbetaCache::Dense {
                blocks: sys
                    .rows
                    .iter()
                    .map(|row| row.htbeta.clone())
                    .collect::<Vec<_>>()
                    .into(),
                estimated_bytes: htbeta_estimated_bytes,
            }
        } else {
            ArrowHtbetaCache::Disabled {
                estimated_bytes: htbeta_estimated_bytes,
            }
        };
        Ok(ArrowFactorCache {
            htt_factors: self.htt_factors.clone(),
            htt_factors_undamped: ArrowUndampedFactors::Owned(self.htt_factors),
            schur_factor: None,
            schur_factor_is_undamped: true,
            beta_schur_deflation: None,
            joint_hessian_log_det: Some(self.row_log_det + schur_log_det),
            solver_mode: ArrowSolverMode::InexactPCG,
            ridge_t: 0.0,
            ridge_beta: 0.0,
            htbeta,
            d: sys.d,
            row_dims: Arc::clone(&sys.row_dims),
            row_offsets: Arc::clone(&sys.row_offsets),
            k: sys.k,
            manifold_mode_fingerprint: sys.manifold_mode_fingerprint,
            row_hessian_fingerprint: sys.current_row_hessian_fingerprint(),
            pcg_diagnostics: ArrowPcgDiagnostics::default(),
            gauge_deflated_directions: self.gauge_deflated_directions,
            deflated_row_directions: Arc::from(self.deflated_row_directions),
            deflation_row_spectra: Arc::from(self.deflation_row_spectra),
            beta_gauge_quotient: sys.beta_gauge_quotient.clone(),
        })
    }
}
