use super::*;

/// Default inner P-IRLS tolerance floor.
///
/// The inner Newton iteration certifies the coefficient mode against this
/// (scale-aware) tolerance independently of the outer REML tolerance. Coupling
/// the two collapses two unrelated convergence concepts: when a user dials the
/// outer tolerance up to e.g. 1e-3 to make the smoothing-parameter search
/// coarser, the inner solve becomes coarse too, returning betas whose
/// stationarity residual is ~1e-3·scale rather than the floating-point noise
/// floor. Outer derivatives then read those imprecise betas as if they were
/// the true mode and accumulate error. Keeping the inner floor at 1e-6 lets
/// the outer loop relax without contaminating the coefficient certificate.
pub(crate) const PIRLS_INNER_TOLERANCE_FLOOR: f64 = 1e-6;

#[derive(Clone)]
pub(crate) struct RemlConfig {
    pub(crate) likelihood: GlmLikelihoodSpec,
    pub(crate) link_kind: InverseLink,
    pub(crate) pirls_convergence_tolerance: f64,
    pub(crate) max_iterations: usize,
    pub(crate) reml_convergence_tolerance: f64,
    pub(crate) firth_bias_reduction: bool,
}

impl RemlConfig {
    pub(crate) fn external(
        likelihood: GlmLikelihoodSpec,
        reml_tol: f64,
        firth_bias_reduction: bool,
    ) -> Self {
        // Inner P-IRLS certifies the coefficient mode against
        // `pirls_convergence_tolerance`; the outer REML iteration certifies
        // the smoothing-parameter optimum against `reml_convergence_tolerance`.
        // These are different concepts and must not be coupled. The inner
        // tolerance is at most the outer tolerance (so a user who *tightens*
        // the outer also tightens the inner), but never coarser than the
        // floor — a coarse outer must not silently pollute the inner mode.
        let pirls_tol = reml_tol.min(PIRLS_INNER_TOLERANCE_FLOOR);
        let link_kind = likelihood.spec.link.clone();
        Self {
            likelihood,
            link_kind,
            pirls_convergence_tolerance: pirls_tol,
            max_iterations: 0,
            reml_convergence_tolerance: reml_tol,
            firth_bias_reduction,
        }
        .with_max_iterations(300)
    }

    pub(crate) fn with_max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    pub(crate) fn link_function(&self) -> LinkFunction {
        self.link_kind.link_function()
    }

    pub(crate) fn as_pirls_config(&self) -> pirls::PirlsConfig {
        pirls::PirlsConfig {
            likelihood: self.likelihood.clone(),
            link_kind: self.link_kind.clone(),
            max_iterations: self.max_iterations,
            convergence_tolerance: self.pirls_convergence_tolerance,
            firth_bias_reduction: self.firth_bias_reduction,
            // Caller (the REML runtime) populates this hint just before
            // each `execute_pirls_if_needed` call from the cached final
            // λ of the previous successful PIRLS solve.
            initial_lm_lambda: None,
            // Arrow-Schur structured-inner-solve descriptor. Not used by
            // the standard REML→PIRLS path (β-only); set by the latent
            // driver (`crate::latent_inner::LatentInnerSolver`)
            // which assembles the per-row (t, β) bordered system
            // externally. Default `None` preserves back-compat.
            arrow_schur: None,
        }
    }
}
/// Small ridge added to the rho-space LAML Hessian before inversion, for
/// numerical stability when smoothing parameters are weakly identified.
///
/// **Stabilization semantics:** this ridge is a
/// [`gam_problem::StabilizationKind::NumericalPerturbation`] (not an
/// `ExplicitPrior`). It enters only the inverse used to build `V_rho` for
/// the smoothing-correction propagation step. It does NOT enter the LAML
/// objective, its gradient, the saved coefficients, or any user-visible
/// summary — the rho-Hessian itself is recomputed from first principles
/// in every place that consults it. Classified as
/// [`gam_problem::StabilizationKind::NumericalPerturbation`]; no ledger
/// record is emitted at this site because the perturbation never escapes the
/// local `V_rho` inverse (it touches no saved coefficient, objective, or
/// user-visible summary).
/// Minimum penalized-deviance floor, expressed as a fraction of the
/// problem's own deviance scale (the weighted null deviance `D₀`, see
/// [`smooth_floor_dp`]). The floor exists only to keep the profiled
/// dispersion `φ̂ = D_p/(n−M_p)` strictly positive when a smooth fits the
/// data essentially perfectly (`D_p ↓ 0`), so it must trigger on the
/// *relative* smallness `D_p/D₀`, never on an absolute magnitude — an
/// absolute floor silently breaks the exact scale-equivariance of the
/// Gaussian REML fit under a response rescale `y → a·y` (#1127).
pub(crate) const DP_FLOOR: f64 = 1e-12;
/// Width of the smooth transition region for the deviance floor, also as a
/// fraction of the deviance scale `D₀`.
const DP_FLOOR_SMOOTH_WIDTH: f64 = 1e-8;

// Unified rho bound corresponding to lambda in [exp(-RHO_BOUND), exp(RHO_BOUND)].
// Additional headroom reduces frequent contact with the hard box constraints.
pub const RHO_BOUND: f64 = 30.0;
// Soft interior prior on rho near the box boundaries.
pub(crate) const RHO_SOFT_PRIOR_WEIGHT: f64 = 1e-6;
pub(crate) const RHO_SOFT_PRIOR_SHARPNESS: f64 = 4.0;
// Adaptive cubature guardrails for bounded correction latency.
pub(crate) const AUTO_CUBATURE_MAX_RHO_DIM: usize = 12;
pub(crate) const AUTO_CUBATURE_MAX_EIGENVECTORS: usize = 4;
pub(crate) const AUTO_CUBATURE_TARGET_VAR_FRAC: f64 = 0.95;
pub(crate) const AUTO_CUBATURE_MAX_BETA_DIM: usize = 1600;
pub(crate) const AUTO_CUBATURE_BOUNDARY_MARGIN: f64 = 2.0;

/// Smooth, differentiable approximation of `max(dp, floor)` where the floor
/// and the width of the smoothing band are taken **relative to the supplied
/// deviance `scale`** (the weighted null deviance `D₀` of the response).
///
/// Returns the smoothed value, first derivative, and second derivative with
/// respect to `dp`.
///
/// # Why the floor must be relative (issue #1127)
///
/// The penalized deviance `D_p = Σ wᵢ(yᵢ−μ̂ᵢ)² + β̂ᵀSβ̂` is exactly quadratic
/// in the response, so under a multiplicative rescale `y → a·y` it scales as
/// `D_p → a²·D_p`. The profiled Gaussian REML criterion depends on `D_p` only
/// through `log D_p` (the `(ν/2)·log(2πφ̂)` term, `φ̂ = D_p/ν`), so the rescale
/// shifts the cost by the *additive constant* `ν·log a` and leaves the
/// ρ-gradient — hence the selected `λ̂`, the EDF, and `ŝ(x)/a` — exactly
/// invariant. An **absolute** floor destroys this: when `a` is small enough
/// that `D_p` enters the fixed band (e.g. `D_p ≈ 3.6e-11` at `a = 1e-6` with
/// a band of width `1e-8`), `dp_c` is spuriously inflated toward the absolute
/// floor, `log dp_c` stops tracking `2·log a + const`, and the optimizer
/// converges at an over-smoothed `λ̂` — reshaping, not merely rescaling, the
/// smooth. Scaling both the floor and its width by `D₀ ∝ a²` makes the band a
/// fixed *fraction* of the deviance, so `smooth_floor_dp(a²·dp, a²·D₀) =
/// a²·smooth_floor_dp(dp, D₀)` exactly and equivariance is restored.
///
/// `scale = 1.0` recovers the historical absolute floor byte-for-byte, which
/// is the correct default for callers without a Gaussian response scale in
/// hand (the floor is consumed only on the profiled-Gaussian path).
pub(crate) fn smooth_floor_dp(dp: f64, scale: f64) -> (f64, f64, f64) {
    let scale = if scale.is_finite() && scale > 0.0 {
        scale
    } else {
        1.0
    };
    let floor = DP_FLOOR * scale;
    let tau = (DP_FLOOR_SMOOTH_WIDTH * scale).max(f64::MIN_POSITIVE);
    let scaled = (dp - floor) / tau;

    let softplus = if scaled > 20.0 {
        scaled + (-scaled).exp()
    } else if scaled < -20.0 {
        scaled.exp()
    } else {
        (1.0 + scaled.exp()).ln()
    };

    let sigma = if scaled >= 0.0 {
        let exp_neg = (-scaled).exp();
        1.0 / (1.0 + exp_neg)
    } else {
        let exp_pos = scaled.exp();
        exp_pos / (1.0 + exp_pos)
    };

    let dp_c = floor + tau * softplus;
    let dp_cgrad2 = sigma * (1.0 - sigma) / tau;
    (dp_c, sigma, dp_cgrad2)
}

/// Compute the smoothing parameter uncertainty correction matrix `V_corr = J * V_rho * J^T`.
///
/// This implements the Wood et al. (2016) correction for smoothing parameter uncertainty.
/// The corrected covariance for `beta` is: `V*_beta = V_beta + J * V_rho * J^T`.
/// where:
/// - `V_beta = H^{-1}` (conditional covariance treating `lambda` as fixed)
/// - `J = d(beta)/d(rho)` (Jacobian wrt log-smoothing parameters)
/// - `V_rho = (d^2 LAML / d rho^2)^{-1}` (outer covariance)
///
/// Returns the correction matrix in the ORIGINAL coefficient basis.
///
/// Full correction reference.
/// Let `rho ~ N(mu, Sigma)` with `mu = rho_hat`, `Sigma = V_rho`,
/// and define:
/// - `A(rho) = H_rho^{-1}`
/// - `b(rho) = beta_hat_rho`
///
/// The exact Gaussian-mixture identity is:
///   `Var(beta) = E[A(rho)] + Var(b(rho))`.
///
/// Around `mu`, this routine keeps the first-order terms:
///
///   `E[A(rho)]      ~= A(mu) = Hmu^{-1}`
///   `Var(b(rho))    ~= J Sigma J^T`
///   `Var(beta)      ~= Hmu^{-1} + J V_rho J^T`.
///
/// Equivalent first-order propagation around the outer optimum `rho*`:
///
///   `Var(beta_hat) ~= Var(beta_hat | rho_hat) + (d beta_hat / d rho) Var(rho_hat) (d beta_hat / d rho)^T`
///                  `= V_beta + J V_rho J^T`.
///
/// Components:
///   `J[:,k] = d(beta_hat)/d(rho_k) = -H^{-1}(A_k beta_hat),  A_k = exp(rho_k) S_k`
///   `V_rho  = (d^2 V / d rho^2 at rho*)^{-1}`
///
/// Exact non-Gaussian V_ρ^{-1} requires the full Hessian with:
///   - tr(H^{-1}H_{kℓ})
///   - tr(H^{-1}H_k H^{-1}H_ℓ)
///   - pseudo-det second derivatives in S
///   - and H_{kℓ} terms containing fourth-likelihood derivatives.
///
/// This routine obtains V_ρ^{-1} from the analytic rho-space Hessian selected
/// by `compute_lamlhessian_consistent`, then inverts its explicitly identified
/// subspace without perturbing the matrix. If exact geometry is unavailable,
/// the typed status records why; no substitute Hessian is used.
///
/// Notes on omitted higher-order terms:
/// - The exact `E[A(rho)]` and `Var(b(rho))` can be written with the Gaussian
///   smoothing/heat operator `exp(0.5 * Delta_Sigma)` (equivalently Wick/Isserlis
///   contractions of high-order derivatives).
/// - Those infinite-series corrections are not expanded in this routine.
pub(crate) struct SmoothingCorrectionComputation {
    pub correction: Option<Array2<f64>>,
    pub hessian_rho: Option<Array2<f64>>,
    /// Regularized inverse outer Hessian `Cov(rho_hat)` in the same rho ordering
    /// as the fitted smoothing-parameter vector. This exposes the #740 quantity
    /// to LR Bartlett inference without changing the production algebra that
    /// computes it.
    pub rho_covariance: Option<Array2<f64>>,
    /// Identified-subspace rank of the rho-Hessian inverse used to build
    /// `correction`. `Some(n)` if the matrix was SPD and fully inverted;
    /// `Some(r)` with `r < n` if the pseudo-inverse dropped non-identified
    /// directions; `None` when no inversion was attempted or it failed before
    /// producing a usable V_ρ. Downstream consumers (e.g. auto-cubature)
    /// use this to decide whether higher-order corrections are even
    /// meaningful — they aren't when V_ρ is rank-deficient.
    pub active_rank: Option<usize>,
    pub status: SmoothingCorrectionStatus,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum SmoothingCorrectionStatus {
    Computed,
    NotApplicableNoSmoothingParameters,
    ZeroNoIdentifiedOuterDirections,
    Unavailable(SmoothingCorrectionUnavailable),
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum SmoothingCorrectionUnavailable {
    ObjectiveInnerHessian {
        error: String,
    },
    InnerHessianDimension {
        rows: usize,
        cols: usize,
        coefficients: usize,
    },
    InnerHessianNotPositiveDefinite,
    SensitivitySolve,
    OuterHessian {
        error: String,
    },
    OuterHessianInverse { error: String },
    PenaltyDimension {
        rho: usize,
        lambdas: usize,
        canonical_penalties: usize,
    },
    PenaltyStructure { error: String },
    NonFiniteCorrection,
}

/// Certified inverse of the rho-space LAML Hessian. A pseudoinverse is admitted
/// only for zero directions whose count is independently certified by the
/// structural penalty map; positive curvature is never truncated and negative
/// curvature is never salvaged as covariance.
#[derive(Debug)]
pub(crate) struct InvertedRhoHessian {
    pub inverse: Array2<f64>,
    pub active_rank: usize,
    pub structural_zero: usize,
    /// Directions dropped because their curvature sits under the outer loop's
    /// own gradient noise floor (#2428). Distinct from `structural_zero`.
    pub below_gradient_floor: usize,
    pub used_structural_pseudoinverse: bool,
    pub eigenvalue_backward_error_bound: f64,
    pub eigenvalues: Array1<f64>,
    pub eigenvectors: Array2<f64>,
    pub classifications: Vec<EigenClassification>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum EigenClassification {
    Active,
    StructuralZero,
    /// Curvature indistinguishable from zero at the accuracy the OUTER LOOP
    /// itself demonstrated when it accepted this ρ̂ (#2428).
    ///
    /// Not a structural zero of the penalty map: it is created dynamically by
    /// a smoothing parameter running to its saturation limit, where the term
    /// is numerically pinned to its own null space and the REML profile goes
    /// flat in that coordinate. The count is therefore NOT knowable a priori
    /// and is excluded from the structural-nullity identity below.
    BelowGradientFloor,
}

/// Assemble `Q J Vρ Jᵀ Qᵀ` through a rectangular square-root factor.
///
/// `invert_identified_rho_hessian` has already classified every retained
/// eigendirection as strictly positive and resolved. Therefore
///
/// ```text
/// Vρ = U_active diag(1 / σ_active) U_activeᵀ
/// Q J Vρ Jᵀ Qᵀ = B Bᵀ,
/// B = Q J U_active diag(1 / sqrt(σ_active)).
/// ```
///
/// Forming the covariance as the Gram matrix `B Bᵀ` preserves its defining
/// positive-semidefinite structure in floating-point arithmetic. In
/// particular, each diagonal is accumulated as a sum of squares. The previous
/// `(J Vρ) Jᵀ` followed by `Q (…) Qᵀ` route accumulated signed cancellation in
/// two generic matrix products. When a true correction diagonal was zero or
/// much smaller than the matrix's signed off-diagonal scale, that route could
/// emit a negative diagonal of order ε; a small conditional covariance need
/// not dominate that error when the corrected covariance is assembled.
fn smoothing_correction_gram(
    jacobian_trans: &Array2<f64>,
    qs: &Array2<f64>,
    eigenvalues: &Array1<f64>,
    eigenvectors: &Array2<f64>,
    classifications: &[EigenClassification],
) -> Array2<f64> {
    let rho_dimension = jacobian_trans.ncols();
    assert_eq!(eigenvalues.len(), rho_dimension);
    assert_eq!(eigenvectors.dim(), (rho_dimension, rho_dimension));
    assert_eq!(classifications.len(), rho_dimension);
    assert_eq!(qs.ncols(), jacobian_trans.nrows());

    let active_directions: Vec<usize> = classifications
        .iter()
        .enumerate()
        .filter_map(|(index, class)| {
            matches!(class, EigenClassification::Active).then_some(index)
        })
        .collect();
    let mut factor_trans =
        Array2::<f64>::zeros((jacobian_trans.nrows(), active_directions.len()));
    for (factor_column, &eigen_index) in active_directions.iter().enumerate() {
        let eigenvalue = eigenvalues[eigen_index];
        assert!(
            eigenvalue.is_finite() && eigenvalue > 0.0,
            "an active rho-Hessian eigendirection must have finite positive curvature"
        );
        let scale = eigenvalue.sqrt().recip();
        let direction = jacobian_trans.dot(&eigenvectors.column(eigen_index));
        factor_trans
            .column_mut(factor_column)
            .assign(&direction.mapv(|value| value * scale));
    }

    let factor_orig = qs.dot(&factor_trans);
    let mut correction = factor_orig.dot(&factor_orig.t());
    gam_linalg::matrix::symmetrize_in_place(&mut correction);
    // Make the Gram diagonal's non-negativity explicit rather than relying on
    // a backend-specific GEMM diagonal accumulation path.
    for index in 0..factor_orig.nrows() {
        let row = factor_orig.row(index);
        correction[[index, index]] = row.dot(&row);
    }
    correction
}

fn eigenpair_backward_error_bound(
    matrix: &Array2<f64>,
    eigenvalues: &Array1<f64>,
    eigenvectors: &Array2<f64>,
) -> Result<f64, String> {
    let n = matrix.nrows();
    if matrix.ncols() != n || eigenvalues.len() != n || eigenvectors.dim() != (n, n) {
        return Err("eigendecomposition dimensions do not match the symmetric matrix".into());
    }
    if !matrix.iter().all(|value| value.is_finite())
        || !eigenvalues.iter().all(|value| value.is_finite())
        || !eigenvectors.iter().all(|value| value.is_finite())
    {
        return Err("eigendecomposition contains a non-finite value".into());
    }
    let matrix_scale = matrix
        .iter()
        .copied()
        .map(f64::abs)
        .fold(0.0_f64, f64::max);
    let mut max_residual_norm = 0.0_f64;
    for column in 0..n {
        let vector = eigenvectors.column(column);
        let residual = matrix.dot(&vector) - &vector.mapv(|value| value * eigenvalues[column]);
        max_residual_norm = max_residual_norm.max(residual.dot(&residual).sqrt());
    }
    let arithmetic_bound = 64.0 * n.max(1) as f64 * f64::EPSILON * matrix_scale;
    Ok(max_residual_norm.max(arithmetic_bound))
}

fn penalty_map_structural_nullity(
    canonical: &[gam_terms::construction::CanonicalPenalty],
    coefficient_dimension: usize,
) -> Result<usize, String> {
    use gam_linalg::faer_ndarray::FaerEigh;

    let k = canonical.len();
    if k == 0 {
        return Ok(0);
    }
    for (index, penalty) in canonical.iter().enumerate() {
        let block_dimension = penalty.col_range.end.saturating_sub(penalty.col_range.start);
        if penalty.col_range.end > coefficient_dimension
            || penalty.local.dim() != (block_dimension, block_dimension)
        {
            return Err(format!(
                "canonical penalty {index} has range {:?}, local shape {:?}, coefficient dimension {coefficient_dimension}",
                penalty.col_range,
                penalty.local.dim()
            ));
        }
    }
    // Gram matrix of the unscaled derivative maps S_k. Positive lambdas only
    // rescale columns and therefore cannot change this structural rank.
    let mut gram = Array2::<f64>::zeros((k, k));
    for i in 0..k {
        for j in i..k {
            let start = canonical[i].col_range.start.max(canonical[j].col_range.start);
            let end = canonical[i].col_range.end.min(canonical[j].col_range.end);
            let mut inner = 0.0_f64;
            for global_row in start..end {
                for global_col in start..end {
                    inner += canonical[i].local[[
                        global_row - canonical[i].col_range.start,
                        global_col - canonical[i].col_range.start,
                    ]] * canonical[j].local[[
                        global_row - canonical[j].col_range.start,
                        global_col - canonical[j].col_range.start,
                    ]];
                }
            }
            gram[[i, j]] = inner;
            gram[[j, i]] = inner;
        }
    }
    let (eigenvalues, eigenvectors) = gram
        .eigh(faer::Side::Lower)
        .map_err(|error| format!("penalty-map Gram eigendecomposition failed: {error}"))?;
    let zero_bound = eigenpair_backward_error_bound(&gram, &eigenvalues, &eigenvectors)?;
    let minimum = eigenvalues.iter().copied().fold(f64::INFINITY, f64::min);
    if minimum < -zero_bound {
        let neg_zero_bound = -zero_bound;
        return Err(format!(
            "penalty-map Gram matrix has negative eigenvalue {minimum:.3e} below backward-error bound {neg_zero_bound:.3e}"
        ));
    }
    let rank = eigenvalues
        .iter()
        .filter(|&&eigenvalue| eigenvalue > zero_bound)
        .count();
    Ok(k - rank)
}

/// Invert the ρ-Hessian on the subspace where its curvature is actually
/// resolvable, judged by the SAME standard the outer certificate used to accept
/// this ρ̂ (#2428).
///
/// `outer_gradient` is the outer loop's residual gradient at the certified
/// point, per ρ coordinate; pass an empty array when it is unavailable. The
/// outer certificate admits a point as a minimum by testing that
/// `H + diag(|g|)` is PSD off the railed coordinates
/// (`certificate_hessian_is_psd_off_railed_above_gradient_floor`): a curvature
/// smaller than the residual gradient in that coordinate is below the noise
/// floor of the very machinery that produced both numbers, so it cannot be
/// called negative. Along eigenvector `v` that test is exactly
/// `σ + Σ_k |g_k| v_k² ≥ 0`, so the per-direction floor is `Σ_k |g_k| v_k²`,
/// evaluated here on the eigenvectors we already have.
///
/// Applying any WEAKER standard here than the certificate applied there lets the
/// two subsystems reach opposite verdicts on one matrix at one converged point —
/// which is exactly #2428: the outer loop certified the fit, and this inversion
/// then destroyed it over an eigenvalue 19x below the residual gradient.
///
/// Directions under that floor are dropped from the inverse rather than
/// regularized. That is the correct limit, not a convenience: they arise when a
/// smoothing parameter saturates, and there `∂β̂/∂ρ → 0`, so the direction
/// contributes nothing to `J·V_ρ·Jᵀ` however large `1/σ` would have been.
pub(crate) fn invert_identified_rho_hessian(
    hessian_rho: &Array2<f64>,
    expected_structural_nullity: usize,
    outer_gradient: &Array1<f64>,
) -> Result<InvertedRhoHessian, String> {
    let n = hessian_rho.nrows();
    if expected_structural_nullity > n {
        return Err(format!(
            "structural nullity {expected_structural_nullity} exceeds rho dimension {n}"
        ));
    }
    if !outer_gradient.is_empty() && outer_gradient.len() != n {
        return Err(format!(
            "outer gradient has {} coordinate(s) but the rho Hessian is {n}x{n}",
            outer_gradient.len()
        ));
    }
    // Reject a non-finite Hessian before the eigensolver sees it: a NaN spectrum
    // would otherwise classify as "unresolvable" rather than as the hard input
    // defect it is.
    gam_linalg::utils::validate_finite_symmetric_matrix(hessian_rho, "rho Hessian")
        .map_err(|error| error.to_string())?;

    let (eigenvalues, eigenvectors) = hessian_rho
        .eigh(faer::Side::Lower)
        .map_err(|error| format!("rho-Hessian eigendecomposition failed: {error}"))?;
    let zero_bound = eigenpair_backward_error_bound(hessian_rho, &eigenvalues, &eigenvectors)?;

    // Per-direction resolution floor: the eigensolver's own backward error, or
    // the outer certificate's gradient floor along this eigenvector, whichever
    // admits more. Both are measurements, not tolerances chosen by hand.
    let direction_floor = |i: usize| -> f64 {
        if outer_gradient.is_empty() {
            return zero_bound;
        }
        let v = eigenvectors.column(i);
        let mut floor = 0.0_f64;
        for k in 0..n {
            floor += outer_gradient[k].abs() * v[k] * v[k];
        }
        if floor.is_finite() { floor.max(zero_bound) } else { zero_bound }
    };

    for i in 0..n {
        let sigma = eigenvalues[i];
        let floor = direction_floor(i);
        if sigma < -floor {
            return Err(format!(
                "rho Hessian has negative curvature {sigma:.3e} below the outer certificate's own \
                 resolution floor {floor:.3e} on that direction (eigensolver backward error \
                 {zero_bound:.3e}); the outer loop certified this point as a minimum, so this is a \
                 genuine contradiction rather than an unresolvable direction"
            ));
        }
    }

    let mut inverse = Array2::<f64>::zeros((n, n));
    let mut projector = Array2::<f64>::zeros((n, n));
    let mut classifications = Vec::with_capacity(n);
    let mut active_rank = 0usize;
    let mut structural_zero = 0usize;
    let mut below_gradient_floor = 0usize;

    for i in 0..n {
        let sigma = eigenvalues[i];
        let floor = direction_floor(i);
        let class = if sigma > floor {
            EigenClassification::Active
        } else if sigma.abs() <= zero_bound {
            // Zero to the eigensolver's own backward error: this is the
            // structural kind the penalty map counts.
            EigenClassification::StructuralZero
        } else {
            EigenClassification::BelowGradientFloor
        };
        classifications.push(class);
        match class {
            EigenClassification::Active => {
                active_rank += 1;
                let inv_lambda = 1.0 / sigma;
                if !inv_lambda.is_finite() {
                    return Err(format!(
                        "positive rho curvature {sigma:.3e} has an unrepresentable reciprocal"
                    ));
                }
                let v = eigenvectors.column(i);
                for row in 0..n {
                    for col in 0..n {
                        inverse[[row, col]] += inv_lambda * v[row] * v[col];
                        projector[[row, col]] += v[row] * v[col];
                    }
                }
            }
            EigenClassification::StructuralZero => structural_zero += 1,
            EigenClassification::BelowGradientFloor => below_gradient_floor += 1,
        }
    }
    // The penalty map certifies HOW MANY directions must be null; it cannot say
    // which eigenpair each one lands on, and once a rail saturates, a structural
    // zero and a saturation null are indistinguishable by magnitude (both sit
    // under the assembly noise, which is far above eigensolver backward error).
    // So the identity to enforce is that the Hessian exhibits AT LEAST the
    // certified number of null directions — finding fewer contradicts the
    // penalty map and is a real defect. Finding more is a saturated rail, which
    // is a property of this ρ̂, not of the penalty map, and is expected.
    let identified_null = structural_zero + below_gradient_floor;
    if identified_null < expected_structural_nullity {
        return Err(format!(
            "rho Hessian has only {identified_null} null direction(s) ({structural_zero} within eigensolver backward error, {below_gradient_floor} under the outer gradient floor), but the penalty map certifies {expected_structural_nullity}"
        ));
    }

    // Every direction resolvable: return the Cholesky-certified inverse, which
    // is what this function has always returned for a strictly positive
    // definite ρ-Hessian. Keeping that path verbatim means no fit that succeeds
    // today moves by a single ulp — the eigen route below engages only where the
    // old code aborted.
    if active_rank == n {
        let certified =
            gam_linalg::utils::certified_spd_inverse(hessian_rho, "unperturbed rho Hessian")
                .map_err(|error| error.to_string())?;
        return Ok(InvertedRhoHessian {
            inverse: certified.into_inverse(),
            active_rank: n,
            structural_zero: 0,
            below_gradient_floor: 0,
            used_structural_pseudoinverse: false,
            eigenvalue_backward_error_bound: zero_bound,
            eigenvalues,
            eigenvectors,
            classifications,
        });
    }

    gam_linalg::matrix::symmetrize_in_place(&mut inverse);
    let matrix_max_abs = gam_linalg::utils::validate_finite_symmetric_matrix(
        hessian_rho,
        "structurally singular rho Hessian",
    )
    .map_err(|error| error.to_string())?;
    let residual = hessian_rho.dot(&inverse) - &projector;
    gam_linalg::utils::certify_linear_system_residual(
        n,
        matrix_max_abs,
        &projector,
        &inverse,
        &residual,
        "rho-Hessian structural pseudoinverse",
    )
    .map_err(|error| error.to_string())?;

    Ok(InvertedRhoHessian {
        inverse,
        active_rank,
        structural_zero,
        below_gradient_floor,
        used_structural_pseudoinverse: true,
        eigenvalue_backward_error_bound: zero_bound,
        eigenvalues,
        eigenvectors,
        classifications,
    })
}

/// Cosine threshold above which two penalty matrices are treated as the
/// structural-redundancy signature in [INDEF-HESS] diagnostics. Pairs with
/// cosine above this AND a dominant-negative eigenvector concentrated on
/// the pair's antisymmetric direction trigger the headline
/// `structural_redundancy_detected` line.
const INDEF_HESS_STRUCTURAL_REDUNDANCY_COS: f64 = 0.999;

/// Penalty-count crossover at which the [INDEF-HESS] pair dump switches from
/// the full O(k²) grid to top-3 pairs only. Bounds log volume on large-scale
/// rho_dim while keeping the per-pair detail useful for small models.
const INDEF_HESS_PAIR_DUMP_GRID_MAX_K: usize = 16;

/// Number of top-cosine pairs to dump when `n_pen > INDEF_HESS_PAIR_DUMP_GRID_MAX_K`.
const INDEF_HESS_PAIR_DUMP_TOP_N: usize = 3;

/// Diagnostic emitted whenever the post-fit rho-Hessian has at least one
/// non-identified direction (active_rank < n). Reports the eigendecomposition,
/// the dominant-negative eigenvector, per-eigenpair classification, and pairwise
/// penalty cosines tr(SᵢSⱼ)/√(tr(Sᵢ²)·tr(Sⱼ²)). A pair cosine ≈ 1.0 combined
/// with the negative eigenvector concentrated on that pair's antisymmetric
/// direction is the structural Z₂-saddle signature.
///
/// Output is capped: when the penalty count exceeds 16, only the top-3
/// highest-cosine pairs are dumped instead of the full O(k²) grid. When the
/// structural-redundancy signature is detected, a single headline line is
/// emitted with the offending pair, cosine, and antisymmetric projection.
fn dump_indefinite_rho_hessian_diagnostic(
    hessian_rho: &Array2<f64>,
    final_rho: &Array1<f64>,
    canonical: &[gam_terms::construction::CanonicalPenalty],
    inverted: Option<&InvertedRhoHessian>,
) {
    let k = hessian_rho.nrows();
    if k == 0 {
        return;
    }

    // Reuse the eigendecomposition already computed by the inverter when present
    // (the slow path always populates it). Only recompute on the rare paths
    // where the diagnostic is called without an `InvertedRhoHessian` (e.g. the
    // eigendecomposition-failed bail in `compute_smoothing_correction`).
    let (eigenvalues_owned, eigenvectors_owned);
    let (eigenvalues_ref, eigenvectors_ref) = match inverted {
        Some(inv) if !inv.eigenvalues.is_empty() && !inv.eigenvectors.is_empty() => {
            (&inv.eigenvalues, &inv.eigenvectors)
        }
        _ => match hessian_rho.eigh(faer::Side::Lower) {
            Ok((evals, evecs)) => {
                eigenvalues_owned = evals;
                eigenvectors_owned = evecs;
                (&eigenvalues_owned, &eigenvectors_owned)
            }
            Err(err) => {
                log::warn!("[INDEF-HESS] eigendecomposition failed: {err}");
                return;
            }
        },
    };

    log::warn!("[INDEF-HESS] rho={:?}", final_rho.as_slice().unwrap_or(&[]),);
    log::warn!(
        "[INDEF-HESS] eigenvalues={:?}",
        eigenvalues_ref.as_slice().unwrap_or(&[]),
    );
    if let Some(inv) = inverted {
        log::warn!(
            "[INDEF-HESS] active_rank={}/{} structural_zero={} below_gradient_floor={} eigenvalue_backward_error_bound={:.3e}",
            inv.active_rank,
            k,
            inv.structural_zero,
            inv.below_gradient_floor,
            inv.eigenvalue_backward_error_bound,
        );
        if !inv.classifications.is_empty() {
            let labels: Vec<&'static str> = inv
                .classifications
                .iter()
                .map(|c| match c {
                    EigenClassification::Active => "A",
                    EigenClassification::StructuralZero => "Z",
                    EigenClassification::BelowGradientFloor => "G",
                })
                .collect();
            log::warn!(
                "[INDEF-HESS] classifications={:?} (A=active Z=structurally certified zero \
                 G=under the outer loop's own gradient floor, i.e. a saturation null)",
                labels,
            );
        }
    }

    let mut neg_idx = 0usize;
    let mut min_eig = f64::INFINITY;
    for (i, &v) in eigenvalues_ref.iter().enumerate() {
        if v < min_eig {
            min_eig = v;
            neg_idx = i;
        }
    }
    let v_neg = eigenvectors_ref.column(neg_idx);
    log::warn!(
        "[INDEF-HESS] negative_eigenvalue={:.4e} eigenvector={:?}",
        min_eig,
        v_neg.as_slice().unwrap_or(&[]),
    );

    let n_pen = canonical.len();
    let mut tr_aa = vec![0.0_f64; n_pen];
    for i in 0..n_pen {
        let local = &canonical[i].local;
        let mut s = 0.0;
        for r in 0..local.nrows() {
            for c in 0..local.ncols() {
                s += local[[r, c]] * local[[r, c]];
            }
        }
        tr_aa[i] = s;
    }
    log::warn!(
        "[INDEF-HESS] penalty_count={} ranges={:?} ranks={:?}",
        n_pen,
        (0..n_pen)
            .map(|i| (canonical[i].col_range.start, canonical[i].col_range.end))
            .collect::<Vec<_>>(),
        (0..n_pen).map(|i| canonical[i].rank()).collect::<Vec<_>>(),
    );

    // Collect compatible pairs with their cosines.
    struct PairCos {
        i: usize,
        j: usize,
        cos: f64,
        antisym_proj: f64,
    }
    let mut pairs: Vec<PairCos> = Vec::new();
    for i in 0..n_pen {
        for j in (i + 1)..n_pen {
            let ci = &canonical[i];
            let cj = &canonical[j];
            if ci.col_range != cj.col_range {
                continue;
            }
            let local_i = &ci.local;
            let local_j = &cj.local;
            let mut dot = 0.0;
            for r in 0..local_i.nrows() {
                for c in 0..local_i.ncols() {
                    dot += local_i[[r, c]] * local_j[[r, c]];
                }
            }
            let cos = if tr_aa[i] > 0.0 && tr_aa[j] > 0.0 {
                dot / (tr_aa[i].sqrt() * tr_aa[j].sqrt())
            } else {
                f64::NAN
            };
            let antisym_proj = if v_neg.len() == n_pen {
                (v_neg[i] - v_neg[j]) / std::f64::consts::SQRT_2
            } else {
                f64::NAN
            };
            pairs.push(PairCos {
                i,
                j,
                cos,
                antisym_proj,
            });
        }
    }

    // Headline: structural redundancy detection. Pair cosine above the
    // structural-redundancy threshold AND the dominant-negative eigenvector's
    // top-2 absolute components on indices (i, j) with opposite signs.
    if min_eig < 0.0 && v_neg.len() == n_pen {
        for p in &pairs {
            if !(p.cos > INDEF_HESS_STRUCTURAL_REDUNDANCY_COS) {
                continue;
            }
            let mut indexed: Vec<(usize, f64)> = v_neg
                .iter()
                .enumerate()
                .map(|(idx, &val)| (idx, val))
                .collect();
            indexed.sort_by(|a, b| {
                b.1.abs()
                    .partial_cmp(&a.1.abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            if indexed.len() < 2 {
                continue;
            }
            let top0 = indexed[0].0;
            let top1 = indexed[1].0;
            let (a, b) = if top0 == p.i && top1 == p.j {
                (indexed[0].1, indexed[1].1)
            } else if top0 == p.j && top1 == p.i {
                (indexed[1].1, indexed[0].1)
            } else {
                continue;
            };
            if a * b >= 0.0 {
                continue;
            }
            log::warn!(
                "[INDEF-HESS] structural_redundancy_detected pair=({},{}) cos={:.6} antisym_proj={:.4e}",
                p.i,
                p.j,
                p.cos,
                p.antisym_proj,
            );
            break;
        }
    }

    // Cap output: dump the full grid when small, otherwise only the top-N
    // highest-cosine pairs.
    if n_pen <= INDEF_HESS_PAIR_DUMP_GRID_MAX_K {
        for p in &pairs {
            log::warn!(
                "[INDEF-HESS] pair=({},{}) cos={:.6} tr_ii={:.4e} tr_jj={:.4e} v_neg[i]-v_neg[j]/sqrt2={:.4e}",
                p.i,
                p.j,
                p.cos,
                tr_aa[p.i],
                tr_aa[p.j],
                p.antisym_proj,
            );
        }
        // Note: we no longer log a "ranges_differ" line per skipped pair to
        // keep the diagnostic O(k). The headline pair already captures intent.
    } else {
        let mut top: Vec<&PairCos> = pairs.iter().filter(|p| p.cos.is_finite()).collect();
        top.sort_by(|a, b| {
            b.cos
                .abs()
                .partial_cmp(&a.cos.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        for p in top.iter().take(INDEF_HESS_PAIR_DUMP_TOP_N) {
            log::warn!(
                "[INDEF-HESS] top_pair=({},{}) cos={:.6} tr_ii={:.4e} tr_jj={:.4e} v_neg[i]-v_neg[j]/sqrt2={:.4e}",
                p.i,
                p.j,
                p.cos,
                tr_aa[p.i],
                tr_aa[p.j],
                p.antisym_proj,
            );
        }
    }
}

/// `outer_gradient` is the outer loop's residual gradient at the certified ρ̂,
/// per coordinate (empty when unavailable). It is the resolution floor the
/// ρ-Hessian's definiteness must be judged against — see
/// [`invert_identified_rho_hessian`] and #2428.
pub(crate) fn compute_smoothing_correction(
    reml_state: &RemlState<'_>,
    final_rho: &Array1<f64>,
    lambdas: &Array1<f64>,
    final_fit: &pirls::PirlsResult,
    outer_gradient: &Array1<f64>,
) -> SmoothingCorrectionComputation {
    use gam_linalg::faer_ndarray::FaerCholesky;

    let n_rho = final_rho.len();
    if n_rho == 0 {
        return SmoothingCorrectionComputation {
            correction: None,
            hessian_rho: None,
            rho_covariance: None,
            active_rank: None,
            status: SmoothingCorrectionStatus::NotApplicableNoSmoothingParameters,
        };
    }

    let n_coeffs_trans = final_fit.beta_transformed.len();
    let ct = &final_fit.reparam_result.canonical_transformed;
    if lambdas.len() != n_rho || ct.len() != n_rho {
        return SmoothingCorrectionComputation {
            correction: None,
            hessian_rho: None,
            rho_covariance: None,
            active_rank: None,
            status: SmoothingCorrectionStatus::Unavailable(
                SmoothingCorrectionUnavailable::PenaltyDimension {
                    rho: n_rho,
                    lambdas: lambdas.len(),
                    canonical_penalties: ct.len(),
                },
            ),
        };
    }
    let structural_nullity = match penalty_map_structural_nullity(ct, n_coeffs_trans) {
        Ok(nullity) => nullity,
        Err(error) => {
            return SmoothingCorrectionComputation {
                correction: None,
                hessian_rho: None,
                rho_covariance: None,
                active_rank: None,
                status: SmoothingCorrectionStatus::Unavailable(
                    SmoothingCorrectionUnavailable::PenaltyStructure { error },
                ),
            };
        }
    };

    // Step 1: Compute the Jacobian J = d(beta)/d(rho) in transformed space.
    //
    // Exact implicit-function identity at the inner optimum:
    //   dβ̂/dρ_k = -H^{-1}(S_k^ρ (β̂ - μ_k)),   S_k^ρ = λ_k S_k,
    //   λ_k = exp(ρ_k).
    //
    // In transformed coordinates with root penalties S_k = R_kᵀR_k:
    //   S_k (β̂ - μ_k) = R_kᵀ(R_k (β̂ - μ_k)),
    // so each Jacobian column is one linear solve with H.

    // Use the same objective-consistent inner Hessian surface used by REML:
    // - non-Firth: H = X'W_HX + S (+ stabilization if present)
    // - Firth logit: H_total = H - d²Phi/dβ²
    // Conclusion:
    //   J[:,k] = dβ̂/dρ_k must use the Jacobian of the actual stationarity
    //   system G*(β,ρ)=0, i.e. H_total for Firth-adjusted fits. Using only
    //   X'W_HX+S here would be inconsistent with the fitted objective and would
    //   misstate smoothing-parameter uncertainty propagation.
    let h_trans = match reml_state.objective_innerhessian(final_rho) {
        Ok(hessian) => hessian,
        Err(error) => {
            return SmoothingCorrectionComputation {
                correction: None,
                hessian_rho: None,
                rho_covariance: None,
                active_rank: None,
                status: SmoothingCorrectionStatus::Unavailable(
                    SmoothingCorrectionUnavailable::ObjectiveInnerHessian {
                        error: error.to_string(),
                    },
                ),
            };
        }
    };

    // The IFT solve below feeds length-`n_coeffs_trans` right-hand sides into
    // the Cholesky factor of `h_trans`, and faer asserts `rhs.len() == factor.n()`.
    // A Hessian that does not match the coefficient dimension (e.g. a degenerate
    // 0×0 placeholder from a geometry backend that failed to materialize a real
    // dense inner Hessian) would otherwise abort the whole fit inside the solve.
    if h_trans.nrows() != n_coeffs_trans || h_trans.ncols() != n_coeffs_trans {
        log::warn!(
            "smoothing-correction inner Hessian shape {}x{} does not match coefficient dimension {}; skipping.",
            h_trans.nrows(),
            h_trans.ncols(),
            n_coeffs_trans
        );
        return SmoothingCorrectionComputation {
            correction: None,
            hessian_rho: None,
            rho_covariance: None,
            active_rank: None,
            status: SmoothingCorrectionStatus::Unavailable(
                SmoothingCorrectionUnavailable::InnerHessianDimension {
                    rows: h_trans.nrows(),
                    cols: h_trans.ncols(),
                    coefficients: n_coeffs_trans,
                },
            ),
        };
    }

    // Factor the Hessian for solving
    let h_chol = match h_trans.cholesky(faer::Side::Lower) {
        Ok(c) => c,
        Err(_) => {
            log::warn!("Cholesky decomposition failed for smoothing correction; skipping.");
            return SmoothingCorrectionComputation {
                correction: None,
                hessian_rho: None,
                rho_covariance: None,
                active_rank: None,
                status: SmoothingCorrectionStatus::Unavailable(
                    SmoothingCorrectionUnavailable::InnerHessianNotPositiveDefinite,
                ),
            };
        }
    };

    let beta_trans = final_fit.beta_transformed.as_ref();
    // Build the stationarity-gradient derivative matrix G_ρ where column k is
    // ∂g(β,ρ)/∂ρ_k = λ_k S_k(β - μ_k), then delegate the IFT solve
    // dβ/dρ = -H⁻¹G_ρ to the canonical evidence helper. This keeps the
    // coefficient-space prediction correction and the joint-evidence
    // Arrow-Schur path on the same hand-derived IFT identity.
    let mut dg_drho_trans = Array2::<f64>::zeros((n_coeffs_trans, n_rho));
    // Per-ρ_k support: the coefficient range its stationarity-gradient
    // derivative ∂g/∂ρ_k is nonzero on. Each column is block-local (only the
    // k-th penalty block), so this is exactly cp.col_range; structurally
    // inactive columns keep an empty support and the cone-of-influence solve
    // skips them entirely (their sensitivity is identically zero). See #779.
    let mut col_supports: Vec<std::ops::Range<usize>> = vec![0..0; n_rho];
    for k in 0..n_rho {
        let cp = &ct[k];
        if cp.rank() == 0 {
            continue;
        }
        // S_k(β - μ) — block-local: R^T (R (β[block] - μ)), embedded into p-vector.
        let r = &cp.col_range;
        col_supports[k] = r.start..r.end;
        let beta_block = beta_trans.slice(s![r.start..r.end]);
        let centered = &beta_block - &cp.prior_mean;
        let r_beta = cp.root.dot(&centered);
        for a in 0..cp.block_dim() {
            dg_drho_trans[[r.start + a, k]] = lambdas[k]
                * (0..cp.rank())
                    .map(|row| cp.root[[row, a]] * r_beta[row])
                    .sum::<f64>();
        }
    }
    // Lazy/local cone-of-influence propagation (#779): confine each column's
    // sensitivity to the coupling component of `h_trans` containing the moved
    // penalty block, and skip structurally inactive columns. Exact on a
    // block-decoupled Hessian (entries outside the cone are identically zero)
    // and identical to the full joint solve on a fully coupled Hessian.
    let jacobian_trans =
        match crate::sensitivity::FitSensitivity::from_faer_cholesky(&h_chol, n_coeffs_trans)
            .mode_response_coned(h_trans.view(), dg_drho_trans.view(), &col_supports)
        {
            Some(jacobian) => jacobian,
            None => {
                log::warn!(
                    "IFT beta-rho sensitivity solve failed for smoothing correction; skipping."
                );
                return SmoothingCorrectionComputation {
                    correction: None,
                    hessian_rho: None,
                    rho_covariance: None,
                    active_rank: None,
                    status: SmoothingCorrectionStatus::Unavailable(
                        SmoothingCorrectionUnavailable::SensitivitySolve,
                    ),
                };
            }
        };

    // Step 2: Build V_rho by inverting the LAML Hessian in rho-space.
    // The authoritative inner-strategy path chooses the rho-space Hessian
    // evaluation policy here. Unified may still perform local numerical
    // salvage inside the exact branch, but the branch choice itself no longer
    // lives inline at the call site.
    let mut hessian_rho = match reml_state.compute_lamlhessian_consistent(final_rho) {
        Ok(h) => h,
        Err(err) => {
            log::warn!(
                "LAML Hessian unavailable ({}); skipping smoothing correction.",
                err
            );
            return SmoothingCorrectionComputation {
                correction: None,
                hessian_rho: None,
                rho_covariance: None,
                active_rank: None,
                status: SmoothingCorrectionStatus::Unavailable(
                    SmoothingCorrectionUnavailable::OuterHessian {
                        error: err.to_string(),
                    },
                ),
            };
        }
    };

    // Symmetrize the Hessian
    gam_linalg::matrix::symmetrize_in_place(&mut hessian_rho);

    // Step 3: invert the exact, unperturbed Hessian on its explicitly
    // identified spectral subspace. A diagonal ridge would change V_rho and
    // therefore the covariance estimand while being invisible in the result.
    let inverted =
        match invert_identified_rho_hessian(&hessian_rho, structural_nullity, outer_gradient) {
        Ok(inverse) => inverse,
        Err(error) => {
            log::warn!("Exact LAML rho-Hessian inversion failed: {error}");
            dump_indefinite_rho_hessian_diagnostic(
                &hessian_rho,
                final_rho,
                &final_fit.reparam_result.canonical_transformed,
                None,
            );
            return SmoothingCorrectionComputation {
                correction: None,
                hessian_rho: Some(hessian_rho),
                rho_covariance: None,
                active_rank: None,
                status: SmoothingCorrectionStatus::Unavailable(
                    SmoothingCorrectionUnavailable::OuterHessianInverse { error },
                ),
            };
        }
    };

    let n_rho_total = hessian_rho.nrows();
    if inverted.active_rank == 0 {
        // Every direction is independently certified as a structural zero of
        // the penalty map, so J·V_ρ·Jᵀ is mathematically zero.
        log::info!(
            "LAML rho Hessian has no identified directions (active_rank=0/{}, structural_zero={}, eigenvalue_backward_error_bound={:.3e}); smoothing correction is exactly zero.",
            n_rho_total,
            inverted.structural_zero,
            inverted.eigenvalue_backward_error_bound,
        );
        dump_indefinite_rho_hessian_diagnostic(
            &hessian_rho,
            final_rho,
            &final_fit.reparam_result.canonical_transformed,
            Some(&inverted),
        );
        return SmoothingCorrectionComputation {
            correction: None,
            hessian_rho: Some(hessian_rho),
            rho_covariance: Some(inverted.inverse),
            active_rank: Some(0),
            status: SmoothingCorrectionStatus::ZeroNoIdentifiedOuterDirections,
        };
    }

    if inverted.active_rank < n_rho_total {
        log::info!(
            "LAML rho Hessian is not fully identified (active_rank={}/{}, structural_zero={}, below_gradient_floor={}, eigenvalue_backward_error_bound={:.3e}); using its certified structural pseudoinverse. `below_gradient_floor` counts saturation nulls: directions whose curvature is under the outer loop\'s own residual gradient, so they carry no resolvable rho-variance (#2428).",
            inverted.active_rank,
            n_rho_total,
            inverted.structural_zero,
            inverted.below_gradient_floor,
            inverted.eigenvalue_backward_error_bound,
        );
        dump_indefinite_rho_hessian_diagnostic(
            &hessian_rho,
            final_rho,
            &final_fit.reparam_result.canonical_transformed,
            Some(&inverted),
        );
    }

    let used_structural_pseudoinverse = inverted.used_structural_pseudoinverse;
    let active_rank_used = inverted.active_rank;
    if used_structural_pseudoinverse {
        log::debug!(
            "Applied rank-deficient pseudo-inverse on identified rho-Hessian subspace before smoothing correction."
        );
    }

    // Step 4: Compute V_corr through the identified square-root factor of V_rho.
    //
    // This is the first-order smoothing-parameter uncertainty inflation:
    //   Var(β̂) ≈ Var(β̂|ρ̂) + (dβ̂/dρ) Var(ρ̂) (dβ̂/dρ)ᵀ.
    //
    // Here:
    //   J = dβ̂/dρ,  J[:,k] = -H^{-1}(A_k β̂),
    //   V_ρ = (∇²_{ρρ}V)^{-1} evaluated at the final ρ.
    let qs = &final_fit.reparam_result.qs;
    let v_corr_orig = smoothing_correction_gram(
        &jacobian_trans,
        qs,
        &inverted.eigenvalues,
        &inverted.eigenvectors,
        &inverted.classifications,
    );
    let rho_covariance = inverted.inverse;

    // Validate the result
    if !v_corr_orig.iter().all(|v| v.is_finite()) {
        log::warn!("Non-finite values in smoothing correction matrix; skipping.");
        return SmoothingCorrectionComputation {
            correction: None,
            hessian_rho: Some(hessian_rho),
            rho_covariance: Some(rho_covariance),
            active_rank: Some(active_rank_used),
            status: SmoothingCorrectionStatus::Unavailable(
                SmoothingCorrectionUnavailable::NonFiniteCorrection,
            ),
        };
    }
    SmoothingCorrectionComputation {
        correction: Some(v_corr_orig),
        hessian_rho: Some(hessian_rho),
        rho_covariance: Some(rho_covariance),
        active_rank: Some(active_rank_used),
        status: SmoothingCorrectionStatus::Computed,
    }
}

#[cfg(test)]
mod smoothing_correction_gram_tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn zero_mode_response_produces_bit_exact_zero_covariance_2490() {
        let jacobian = Array2::<f64>::zeros((3, 2));
        let qs = array![
            [1.0_f64, 0.25, -0.5],
            [0.0, 0.75, 0.125],
            [0.5, -0.25, 1.0],
        ];
        let eigenvalues = array![0.5_f64, 2.0];
        let eigenvectors = Array2::<f64>::eye(2);
        let classifications = vec![
            EigenClassification::Active,
            EigenClassification::Active,
        ];

        let correction = smoothing_correction_gram(
            &jacobian,
            &qs,
            &eigenvalues,
            &eigenvectors,
            &classifications,
        );

        assert!(
            correction.iter().all(|&value| value == 0.0),
            "a zero coefficient response to rho must produce an exact zero covariance, got {correction:?}"
        );
        assert_eq!(
            gam_problem::se_from_covariance(&correction).unwrap(),
            Array1::<f64>::zeros(3),
        );
    }

    #[test]
    fn factor_gram_matches_dense_congruence_with_nonnegative_diagonal_2490() {
        let jacobian = array![
            [1.0_f64, -2.0],
            [3.0, 4.0],
            [-5.0, 6.0],
        ];
        let qs = array![
            [1.0_f64, 0.0, 0.0],
            [0.0, 0.8, -0.6],
            [0.0, 0.6, 0.8],
        ];
        let inv_sqrt_two = 2.0_f64.sqrt().recip();
        let eigenvectors = array![
            [inv_sqrt_two, -inv_sqrt_two],
            [inv_sqrt_two, inv_sqrt_two],
        ];
        let eigenvalues = array![0.25_f64, 4.0];
        let classifications = vec![
            EigenClassification::Active,
            EigenClassification::Active,
        ];

        let correction = smoothing_correction_gram(
            &jacobian,
            &qs,
            &eigenvalues,
            &eigenvectors,
            &classifications,
        );
        let inverse = eigenvectors
            .dot(&Array2::from_diag(&eigenvalues.mapv(f64::recip)))
            .dot(&eigenvectors.t());
        let expected = qs
            .dot(&jacobian)
            .dot(&inverse)
            .dot(&jacobian.t())
            .dot(&qs.t());

        let covariance_scale = expected
            .iter()
            .copied()
            .map(f64::abs)
            .fold(0.0_f64, f64::max);
        let arithmetic_bound =
            64.0 * correction.nrows().max(1) as f64 * f64::EPSILON * covariance_scale;
        for (&actual, &reference) in correction.iter().zip(expected.iter()) {
            assert!((actual - reference).abs() <= arithmetic_bound);
        }
        assert!(correction.diag().iter().all(|&value| value >= 0.0));
    }
}
