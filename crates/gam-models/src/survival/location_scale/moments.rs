use super::*;
use gam_solve::gauge::Gauge;

pub(crate) fn survival_response_moment_block_ranges(
    p_time: usize,
    p_t: usize,
    p_ls: usize,
    pw: usize,
) -> (
    std::ops::Range<usize>,
    std::ops::Range<usize>,
    std::ops::Range<usize>,
    Option<std::ops::Range<usize>>,
) {
    let time = 0..p_time;
    let threshold = time.end..time.end + p_t;
    let log_sigma = threshold.end..threshold.end + p_ls;
    let wiggle = (pw > 0).then_some(log_sigma.end..log_sigma.end + pw);
    (time, threshold, log_sigma, wiggle)
}

pub(crate) fn projected_survival_response_moment_covariance(
    covariance: &Array2<f64>,
    a_h: &Array1<f64>,
    a_t: &Array1<f64>,
    a_ls: &Array1<f64>,
    p_time: usize,
    p_t: usize,
    p_ls: usize,
) -> [[f64; 3]; 3] {
    let (time, threshold, log_sigma, _) =
        survival_response_moment_block_ranges(p_time, p_t, p_ls, 0);
    let cov_hh = covariance.slice(s![time.start..time.end, time.start..time.end]);
    let cov_tt = covariance.slice(s![
        threshold.start..threshold.end,
        threshold.start..threshold.end
    ]);
    let cov_ll = covariance.slice(s![
        log_sigma.start..log_sigma.end,
        log_sigma.start..log_sigma.end
    ]);
    let cov_ht = covariance.slice(s![time.start..time.end, threshold.start..threshold.end]);
    let cov_hl = covariance.slice(s![time.start..time.end, log_sigma.start..log_sigma.end]);
    let cov_tl = covariance.slice(s![
        threshold.start..threshold.end,
        log_sigma.start..log_sigma.end
    ]);
    let var_h = a_h.dot(&cov_hh.dot(a_h));
    let var_t = a_t.dot(&cov_tt.dot(a_t));
    let var_ls = a_ls.dot(&cov_ll.dot(a_ls));
    let cov_ht_i = a_h.dot(&cov_ht.dot(a_t));
    let cov_hl_i = a_h.dot(&cov_hl.dot(a_ls));
    let cov_tl_i = a_t.dot(&cov_tl.dot(a_ls));
    [
        [var_h, cov_ht_i, cov_hl_i],
        [cov_ht_i, var_t, cov_tl_i],
        [cov_hl_i, cov_tl_i, var_ls],
    ]
}

pub(crate) fn covariance3_to_array2(cov: [[f64; 3]; 3]) -> Array2<f64> {
    let mut out = Array2::<f64>::zeros((3, 3));
    for i in 0..3 {
        for j in 0..3 {
            out[[i, j]] = cov[i][j];
        }
    }
    out
}

pub(crate) fn symmetrize_and_clip_covariance(cov: &Array2<f64>) -> Array2<f64> {
    let mut out = cov.clone();
    for i in 0..out.nrows() {
        out[[i, i]] = out[[i, i]].max(0.0);
        for j in (i + 1)..out.ncols() {
            let avg = 0.5 * (out[[i, j]] + out[[j, i]]);
            out[[i, j]] = avg;
            out[[j, i]] = avg;
        }
    }
    out
}

pub(crate) struct LowRankGaussianFactor {
    pub(crate) factor: Array2<f64>,
}

// Exact projected-Gaussian handling for possibly singular covariance blocks.
// We integrate over the active standard-normal coordinates rather than adding
// jitter or inverting the covariance directly.
pub(crate) fn factorize_psd_covariance(
    covariance: &Array2<f64>,
    label: &str,
) -> Result<LowRankGaussianFactor, String> {
    let covariance = symmetrize_and_clip_covariance(covariance);
    let (eigenvalues, eigenvectors_full) = covariance
        .eigh(faer::Side::Lower)
        .map_err(|e| format!("{label} eigendecomposition failed: {e}"))?;
    let max_abs_eigenvalue = eigenvalues
        .iter()
        .fold(0.0_f64, |acc, &ev| acc.max(ev.abs()));
    let tol = (max_abs_eigenvalue * PSD_EIGENVALUE_REL_TOL).max(PSD_EIGENVALUE_ABS_FLOOR);
    if eigenvalues.iter().any(|&ev| ev < -tol) {
        return Err(SurvivalLocationScaleError::InvalidConfiguration {
            reason: format!(
                "{label} is not positive semidefinite: minimum eigenvalue {:.3e}",
                eigenvalues
                    .iter()
                    .fold(f64::INFINITY, |acc, &ev| acc.min(ev))
            ),
        }
        .into());
    }

    let active = eigenvalues
        .iter()
        .enumerate()
        .filter_map(|(idx, &ev)| (ev > tol).then_some((idx, ev.sqrt())))
        .collect::<Vec<_>>();
    let mut factor = Array2::<f64>::zeros((covariance.nrows(), active.len()));
    for (col, (idx, sqrt_ev)) in active.into_iter().enumerate() {
        factor
            .column_mut(col)
            .assign(&(&eigenvectors_full.column(idx) * sqrt_ev));
    }

    Ok(LowRankGaussianFactor { factor })
}

fn apply_low_rank_gaussian_factor(
    mean: &Array1<f64>,
    factor: &Array2<f64>,
    z: &[f64],
) -> Array1<f64> {
    let mut x = mean.clone();
    for row in 0..x.len() {
        for (col, &latent) in z.iter().enumerate() {
            x[row] += factor[[row, col]] * latent;
        }
    }
    x
}

fn low_rank_normal_expectation_pair_from_factor<F>(
    quadctx: &crate::quadrature::QuadratureContext,
    mean: &Array1<f64>,
    factor: &Array2<f64>,
    max_n: usize,
    label: &str,
    integrand: F,
) -> Result<(f64, f64), String>
where
    F: Fn(&Array1<f64>) -> Result<(f64, f64), String>,
{
    if factor.nrows() != mean.len() {
        return Err(SurvivalLocationScaleError::DimensionMismatch {
            reason: format!(
                "{label} factor has {} rows for a mean of length {}",
                factor.nrows(),
                mean.len()
            ),
        }
        .into());
    }
    match factor.ncols() {
        0 => integrand(mean),
        1 => crate::quadrature::normal_expectation_nd_adaptive_result::<1, _, _, String>(
            quadctx,
            [0.0],
            [[1.0]],
            max_n,
            |z| {
                let latent = [z[0]];
                integrand(&apply_low_rank_gaussian_factor(mean, factor, &latent))
            },
        ),
        2 => crate::quadrature::normal_expectation_nd_adaptive_result::<2, _, _, String>(
            quadctx,
            [0.0, 0.0],
            [[1.0, 0.0], [0.0, 1.0]],
            max_n,
            |z| {
                let latent = [z[0], z[1]];
                integrand(&apply_low_rank_gaussian_factor(mean, factor, &latent))
            },
        ),
        3 => crate::quadrature::normal_expectation_nd_adaptive_result::<3, _, _, String>(
            quadctx,
            [0.0, 0.0, 0.0],
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            max_n,
            |z| {
                let latent = [z[0], z[1], z[2]];
                integrand(&apply_low_rank_gaussian_factor(mean, factor, &latent))
            },
        ),
        rank => Err(SurvivalLocationScaleError::InternalInvariant {
            reason: format!("{label} unexpectedly has rank {rank} > 3"),
        }
        .into()),
    }
}

pub(crate) fn low_rank_normal_expectation_pair_3d_result<F>(
    quadctx: &crate::quadrature::QuadratureContext,
    mu: [f64; 3],
    covariance: [[f64; 3]; 3],
    max_n: usize,
    label: &str,
    integrand: F,
) -> Result<(f64, f64), String>
where
    F: Fn([f64; 3]) -> Result<(f64, f64), String>,
{
    let factorization = factorize_psd_covariance(&covariance3_to_array2(covariance), label)?;
    low_rank_normal_expectation_pair_from_factor(
        quadctx,
        &Array1::from_vec(mu.to_vec()),
        &factorization.factor,
        max_n,
        label,
        |x| integrand([x[0], x[1], x[2]]),
    )
}

// Exact response moments stay in the original Gaussian coordinates for an
// unconstrained fit. A link-wiggle fit instead uses the multivariate
// cone-pushforward rule carried from `gam-solve`: its discrete coordinates are
// feasible coefficient vectors and its Gaussian residual lies in the cone's
// tangent space.
pub(crate) fn exact_survival_response_moments_row(
    input: &SurvivalLocationScalePredictInput,
    fit: &UnifiedFitResult,
    covariance: &Array2<f64>,
    constrained_rule: Option<(
        &gam_solve::constrained_posterior::ConstrainedPosteriorNodeRule,
        &Gauge,
    )>,
    x_threshold_dense: &Array2<f64>,
    x_log_sigma_dense: &Array2<f64>,
    row: usize,
    quadctx: &crate::quadrature::QuadratureContext,
) -> Result<(f64, f64), String> {
    if input.time_wiggle_ncols > 0 {
        return Err(SurvivalLocationScaleError::InvalidConfiguration { reason: "predict_survival_location_scale: exact response moments are not implemented for time-wiggle models"
                .to_string(), }.into());
    }

    let beta_time = fit.beta_time();
    let beta_threshold = fit.beta_threshold();
    let beta_log_sigma = fit.beta_log_sigma();
    let beta_link_wiggle = fit.beta_link_wiggle();
    let p_time = beta_time.len();
    let p_t = beta_threshold.len();
    let p_ls = beta_log_sigma.len();
    let pw = beta_link_wiggle.as_ref().map_or(0, |beta| beta.len());
    let (time, threshold, log_sigma, wiggle) =
        survival_response_moment_block_ranges(p_time, p_t, p_ls, pw);

    let a_h = input.x_time_exit.row(row).to_owned();
    let a_t = x_threshold_dense.row(row).to_owned();
    let a_ls = x_log_sigma_dense.row(row).to_owned();

    let mu_h = a_h.dot(&beta_time) + input.eta_time_offset_exit[row];
    let mu_t = a_t.dot(&beta_threshold) + input.eta_threshold_offset[row];
    let mu_ls = a_ls.dot(&beta_log_sigma) + input.eta_log_sigma_offset[row];
    let mu = [mu_h, mu_t, mu_ls];
    let cov_htl = projected_survival_response_moment_covariance(
        covariance, &a_h, &a_t, &a_ls, p_time, p_t, p_ls,
    );

    if let (Some(_), Some(wiggle_range)) = (beta_link_wiggle.as_ref(), wiggle) {
        let knots = input
            .link_wiggle_knots
            .as_ref()
            .or(fit.artifacts.survival_link_wiggle_knots.as_ref())
            .ok_or_else(|| {
                "predict_survival_location_scale: link-wiggle coefficients are missing knot metadata"
                    .to_string()
            })?;
        let degree = input
            .link_wiggle_degree
            .or(fit.artifacts.survival_link_wiggle_degree)
            .ok_or_else(|| {
                "predict_survival_location_scale: link-wiggle coefficients are missing degree metadata"
                    .to_string()
            })?;

        let (node_rule, gauge) = constrained_rule.ok_or_else(|| {
            SurvivalLocationScaleError::InvalidConfiguration {
                reason: "predict_survival_location_scale: a link-wiggle posterior requires \
                         its fitted inequality-truncated geometry"
                    .to_string(),
            }
        })?;
        let output_dimension = 3 + pw;
        let p_total = covariance.nrows();
        let mut raw_projection = Array2::<f64>::zeros((output_dimension, p_total));
        raw_projection
            .slice_mut(s![0, time.start..time.end])
            .assign(&a_h);
        raw_projection
            .slice_mut(s![1, threshold.start..threshold.end])
            .assign(&a_t);
        raw_projection
            .slice_mut(s![2, log_sigma.start..log_sigma.end])
            .assign(&a_ls);
        for j in 0..pw {
            raw_projection[[3 + j, wiggle_range.start + j]] = 1.0;
        }
        let mut raw_offset = Array1::<f64>::zeros(output_dimension);
        raw_offset[0] = input.eta_time_offset_exit[row];
        raw_offset[1] = input.eta_threshold_offset[row];
        raw_offset[2] = input.eta_log_sigma_offset[row];
        let (active_projection, active_offset) =
            gauge.restrict_design_and_offset(&raw_projection, &raw_offset);
        let pushforward = node_rule.affine_pushforward(&active_projection, &active_offset)?;
        // Every link-wiggle coefficient has its own non-negativity wall, so the
        // independent Gaussian remainder must have exactly zero variance in
        // those coordinates. Certify that invariant before discarding the
        // round-off left by `CΣC' - CGWG'(CG)'`; then integrate only the true
        // three-coordinate response tangent. Letting an eigensolver turn that
        // subtraction residue back into a wiggle displacement would forfeit the
        // support guarantee precisely at a wall node.
        let tangent_scale = pushforward
            .residual_covariance
            .diag()
            .iter()
            .fold(0.0_f64, |scale, &value| scale.max(value.abs()));
        let tangent_zero_tolerance =
            PSD_EIGENVALUE_ABS_FLOOR + PSD_EIGENVALUE_REL_TOL * tangent_scale;
        for i in 3..output_dimension {
            for j in 0..output_dimension {
                let value = pushforward.residual_covariance[[i, j]];
                if value.abs() > tangent_zero_tolerance {
                    return Err(SurvivalLocationScaleError::InternalInvariant {
                        reason: format!(
                            "survival constrained response tangent moves link-wiggle \
                             coordinate {} through its wall: covariance ({i},{j})={value:.6e}, \
                             zero tolerance={tangent_zero_tolerance:.6e}",
                            i - 3
                        ),
                    }
                    .into());
                }
            }
        }
        let response_tangent_covariance = pushforward
            .residual_covariance
            .slice(s![0..3, 0..3])
            .to_owned();
        let residual = factorize_psd_covariance(
            &response_tangent_covariance,
            "survival constrained response-moment tangent covariance",
        )?;

        let mut first = gam_linalg::utils::KahanSum::default();
        let mut second = gam_linalg::utils::KahanSum::default();
        let mut mass = gam_linalg::utils::KahanSum::default();
        for node in &pushforward.nodes {
            let beta_node = node.conditional_mean.slice(s![3..]);
            if let Some((j, &value)) = beta_node
                .iter()
                .enumerate()
                .find(|(_, value)| **value < 0.0)
            {
                return Err(SurvivalLocationScaleError::InternalInvariant {
                    reason: format!(
                        "constraint-normal node escaped the link-wiggle cone at coordinate \
                         {j}: {value:.6e}"
                    ),
                }
                .into());
            }
            let response_mean = node.conditional_mean.slice(s![0..3]).to_owned();
            let pair = low_rank_normal_expectation_pair_from_factor(
                quadctx,
                &response_mean,
                &residual.factor,
                15,
                "survival constrained response-moment tangent covariance",
                |x| {
                    let q0 = survival_q0_from_eta(x[1], x[2]);
                    let q0_arr = Array1::from_vec(vec![q0]);
                    let basis = survival_wiggle_basis_with_options(
                        q0_arr.view(),
                        knots,
                        degree,
                        BasisOptions::value(),
                    )?;
                    if basis.ncols() != beta_node.len() {
                        return Err(SurvivalLocationScaleError::DimensionMismatch { reason: format!(
                            "predict_survival_location_scale: link-wiggle basis/beta mismatch: {} vs {}",
                            basis.ncols(),
                            beta_node.len()
                        ) }.into());
                    }
                    let warp = basis.row(0).dot(&beta_node);
                    if warp < 0.0 {
                        return Err(SurvivalLocationScaleError::InternalInvariant {
                            reason: format!(
                                "a feasible link-wiggle node produced negative scalar warp \
                                 {warp:.6e}"
                            ),
                        }
                        .into());
                    }
                    let p = inverse_link_survival_prob_checked(
                        &input.inverse_link,
                        x[0] + q0 + warp,
                    )?;
                    Ok((p, p * p))
                },
            )?;
            first.add(node.weight * pair.0);
            second.add(node.weight * pair.1);
            mass.add(node.weight);
        }
        let mass = mass.sum();
        if !(mass.is_finite() && mass > 0.0) {
            return Err(SurvivalLocationScaleError::InternalInvariant {
                reason: format!(
                    "survival constrained response-moment node rule has invalid mass {mass:?}"
                ),
            }
            .into());
        }
        return Ok((
            (first.sum() / mass).clamp(0.0, 1.0),
            (second.sum() / mass).clamp(0.0, 1.0),
        ));
    }

    low_rank_normal_expectation_pair_3d_result(
        quadctx,
        mu,
        cov_htl,
        15,
        "survival response-moment projected covariance",
        |x| {
            let p = inverse_link_survival_prob_checked(
                &input.inverse_link,
                x[0] + survival_q0_from_eta(x[1], x[2]),
            )?;
            Ok((p, p * p))
        },
    )
    .map(|(first, second)| (first.clamp(0.0, 1.0), second.clamp(0.0, 1.0)))
}

pub(crate) fn exact_survival_response_moments(
    input: &SurvivalLocationScalePredictInput,
    fit: &UnifiedFitResult,
    covariance: &Array2<f64>,
) -> Result<(Array1<f64>, Array1<f64>), String> {
    validate_predict_inverse_link(&input.inverse_link)?;

    let n = input.x_time_exit.nrows();
    let p_time = fit.beta_time().len();
    let p_t = fit.beta_threshold().len();
    let p_ls = fit.beta_log_sigma().len();
    let pw = fit.beta_link_wiggle().map_or(0, |beta| beta.len());
    let p_total = p_time + p_t + p_ls + pw;
    if covariance.nrows() != p_total || covariance.ncols() != p_total {
        return Err(SurvivalLocationScaleError::DimensionMismatch { reason: format!(
            "predict_survival_location_scale: covariance shape mismatch: got {}x{}, expected {}x{}",
            covariance.nrows(),
            covariance.ncols(),
            p_total,
            p_total
        ) }.into());
    }
    if input.x_time_exit.ncols() != p_time {
        return Err(SurvivalLocationScaleError::DimensionMismatch {
            reason: format!(
                "predict_survival_location_scale: time design/beta mismatch: {} vs {}",
                input.x_time_exit.ncols(),
                p_time
            ),
        }
        .into());
    }
    if input.eta_time_offset_exit.len() != n
        || input.x_threshold.nrows() != n
        || input.eta_threshold_offset.len() != n
        || input.x_log_sigma.nrows() != n
        || input.eta_log_sigma_offset.len() != n
    {
        return Err(SurvivalLocationScaleError::DimensionMismatch {
            reason: "predict_survival_location_scale: row mismatch across inputs".to_string(),
        }
        .into());
    }

    let constrained_node_rule = if pw > 0 {
        let geometry = fit.geometry.as_ref().ok_or_else(|| {
            SurvivalLocationScaleError::InvalidConfiguration {
                reason: "predict_survival_location_scale: link-wiggle response moments require \
                         fitted coefficient geometry"
                    .to_string(),
            }
        })?;
        let constrained = geometry.constrained_posterior.as_ref().ok_or_else(|| {
            SurvivalLocationScaleError::InvalidConfiguration {
                reason: "predict_survival_location_scale: link-wiggle response moments require \
                         the fitted inequality-truncated posterior identity"
                    .to_string(),
            }
        })?;
        if geometry.coefficient_gauge.raw_total() != p_total {
            return Err(SurvivalLocationScaleError::DimensionMismatch {
                reason: format!(
                    "predict_survival_location_scale: coefficient gauge has raw width {}, \
                     expected {p_total}",
                    geometry.coefficient_gauge.raw_total()
                ),
            }
            .into());
        }
        let ambient = gam_linalg::utils::certified_spd_inverse(
            geometry.penalized_hessian.as_array(),
            "survival constrained response-moment ambient precision",
        )
        .map_err(|error| SurvivalLocationScaleError::InvalidConfiguration {
            reason: format!(
                "predict_survival_location_scale: constrained response moments require the \
                 exact ambient covariance: {error}"
            ),
        })?
        .into_inverse();
        Some(
            gam_solve::constrained_posterior::ConstrainedPosteriorNodeRule::new(
                &ambient,
                constrained,
            )?,
        )
    } else {
        None
    };
    let constrained_gauge = fit
        .geometry
        .as_ref()
        .map(|geometry| &geometry.coefficient_gauge);
    let x_threshold_dense = input.x_threshold.to_dense_arc();
    let x_log_sigma_dense = input.x_log_sigma.to_dense_arc();
    let mut first = Array1::<f64>::zeros(n);
    let mut second = Array1::<f64>::zeros(n);
    // Build a single QuadratureContext up front and share it across all
    // chunks.  Per-chunk construction wastes work (each chunk's first call
    // re-derives the Gauss-Hermite rule from scratch via OnceLock) and risks
    // the OnceLock-inside-rayon deadlock pattern (see repo memory) if the
    // rule init were ever to spawn nested parallel work. Warm the rule size
    // that the per-row evaluator actually uses (15 for the projected Gaussian
    // tangent) so worker threads only hit the cached rule lookup.
    let quadctx = crate::quadrature::QuadratureContext::new();
    {
        // Warm GH rule caches on the calling thread with cheap probes.
        crate::quadrature::normal_expectation_nd_adaptive_result::<1, _, _, String>(
            &quadctx,
            [0.0_f64],
            [[1.0_f64]],
            15,
            |_x: [f64; 1]| Ok((0.0_f64, 0.0_f64)),
        )?;
    }
    if n >= SURVIVAL_ROW_PARALLEL_THRESHOLD {
        let first_slice = first
            .as_slice_mut()
            .expect("fresh Array1 response moments are contiguous");
        let second_slice = second
            .as_slice_mut()
            .expect("fresh Array1 response moments are contiguous");
        let quadctx_ref = &quadctx;
        first_slice
            .par_chunks_mut(SURVIVAL_ROW_PARALLEL_CHUNK)
            .zip(second_slice.par_chunks_mut(SURVIVAL_ROW_PARALLEL_CHUNK))
            .enumerate()
            .try_for_each(
                |(chunk_idx, (first_chunk, second_chunk))| -> Result<(), String> {
                    let row_start = chunk_idx * SURVIVAL_ROW_PARALLEL_CHUNK;
                    for offset in 0..first_chunk.len() {
                        let row = row_start + offset;
                        let (m1, m2) = exact_survival_response_moments_row(
                            input,
                            fit,
                            covariance,
                            constrained_node_rule.as_ref().zip(constrained_gauge),
                            &x_threshold_dense,
                            &x_log_sigma_dense,
                            row,
                            quadctx_ref,
                        )?;
                        first_chunk[offset] = m1;
                        second_chunk[offset] = m2;
                    }
                    Ok(())
                },
            )?;
    } else {
        for row in 0..n {
            let (m1, m2) = exact_survival_response_moments_row(
                input,
                fit,
                covariance,
                constrained_node_rule.as_ref().zip(constrained_gauge),
                &x_threshold_dense,
                &x_log_sigma_dense,
                row,
                &quadctx,
            )?;
            first[row] = m1;
            second[row] = m2;
        }
    }
    Ok((first, second))
}

/// Exact affine map from the fitted location-scale coefficient frame into the
/// saved/reporting frame.
///
/// The inner fit sees the reduced time block and the active tails of the
/// threshold and log-sigma blocks. Saved coefficients expand all three back to
/// their raw layouts. Keeping this as a single [`Gauge`] is essential: the
/// conditional covariance pushes forward through it, while the penalized
/// Hessian remains in the active frame and is paired with this map for exact
/// post-fit row-Jacobian pullback.
pub(crate) fn survival_location_scale_finalization_gauge(
    time_gauge: &Gauge,
    p_threshold_reduced: usize,
    p_threshold_full: usize,
    threshold_fixed_cols: usize,
    p_log_sigma_reduced: usize,
    p_log_sigma_full: usize,
    log_sigma_fixed_cols: usize,
    p_linkwiggle: Option<usize>,
) -> Result<Gauge, String> {
    if time_gauge.n_blocks() != 1 {
        return Err(SurvivalLocationScaleError::InvalidConfiguration {
            reason: format!(
                "survival location-scale finalization expected a single-block time gauge, got {} blocks",
                time_gauge.n_blocks()
            ),
        }
        .into());
    }
    let p_time_reduced = time_gauge.reduced_total();
    let p_time_full = time_gauge.raw_total();
    if threshold_fixed_cols + p_threshold_reduced != p_threshold_full {
        return Err(SurvivalLocationScaleError::InvalidConfiguration { reason: format!(
            "survival location-scale covariance lift threshold dimensions are inconsistent: fixed={}, reduced={}, full={}",
            threshold_fixed_cols, p_threshold_reduced, p_threshold_full
        ) }.into());
    }
    if log_sigma_fixed_cols + p_log_sigma_reduced != p_log_sigma_full {
        return Err(SurvivalLocationScaleError::InvalidConfiguration { reason: format!(
            "survival location-scale covariance lift log-sigma dimensions are inconsistent: fixed={}, reduced={}, full={}",
            log_sigma_fixed_cols, p_log_sigma_reduced, p_log_sigma_full
        ) }.into());
    }
    // Raw↔canonical reconciliation at the time sub-block. The time Gauge lifts
    // the inner solver's ACTIVE (reduced, canonical-gauge) time coefficients
    // back to the RAW time layout, so its linear map must
    // be at least as tall as it is wide — the active block can never carry more
    // columns than the raw block it expands into. If a future canonicalization
    // ever produces a map whose active width exceeds the raw width (the
    // raw-vs-active drift behind the historical `[N,N] → [N-1,N-1]` finalization
    // panic, #735), surface it here as a structured DimensionMismatch instead of
    // letting the downstream block assignment fault with a bare ndarray
    // broadcast. The threshold/log_sigma offsets are already validated above via
    // `*_fixed_cols + reduced == full`; this is the matching guard for the one
    // time block whose map is a dense matrix rather than a fixed-column offset.
    if p_time_reduced > p_time_full {
        return Err(SurvivalLocationScaleError::DimensionMismatch {
            reason: format!(
                "survival location-scale covariance lift time map is wider than tall: \
             active(reduced)={p_time_reduced} exceeds raw(full)={p_time_full}; \
             the time identifiability Gauge must map reduced→raw"
            ),
        }
        .into());
    }
    time_gauge.validate().map_err(|reason| {
        SurvivalLocationScaleError::InvalidConfiguration {
            reason: format!("survival location-scale time gauge is invalid: {reason}"),
        }
        .to_string()
    })?;

    let fixed_tail_transform = |full: usize, fixed: usize, reduced: usize| {
        let mut t = Array2::<f64>::zeros((full, reduced));
        for j in 0..reduced {
            t[[fixed + j, j]] = 1.0;
        }
        t
    };
    let p_linkwiggle_width = p_linkwiggle.unwrap_or(0);
    let p_full = p_time_full + p_threshold_full + p_log_sigma_full + p_linkwiggle_width;
    let mut affine_shift = Array1::<f64>::zeros(p_full);
    affine_shift
        .slice_mut(s![0..p_time_full])
        .assign(&time_gauge.affine_shift);
    let mut block_transforms = vec![
        time_gauge.block_transform(0),
        fixed_tail_transform(p_threshold_full, threshold_fixed_cols, p_threshold_reduced),
        fixed_tail_transform(p_log_sigma_full, log_sigma_fixed_cols, p_log_sigma_reduced),
    ];
    if let Some(width) = p_linkwiggle {
        block_transforms.push(Array2::<f64>::eye(width));
    }
    let joint_gauge = Gauge::from_block_transforms_with_shift(&block_transforms, affine_shift);
    let p_reduced = p_time_reduced + p_threshold_reduced + p_log_sigma_reduced + p_linkwiggle_width;
    assert_eq!(joint_gauge.raw_total(), p_full);
    assert_eq!(joint_gauge.reduced_total(), p_reduced);
    Ok(joint_gauge)
}

pub(crate) fn lift_conditional_covariance(
    cov_reduced: &Array2<f64>,
    finalization_gauge: &Gauge,
) -> Result<Array2<f64>, String> {
    finalization_gauge.validate().map_err(|reason| {
        SurvivalLocationScaleError::InvalidConfiguration {
            reason: format!("survival location-scale finalization gauge is invalid: {reason}"),
        }
        .to_string()
    })?;
    let p_reduced = finalization_gauge.reduced_total();
    if cov_reduced.nrows() != p_reduced || cov_reduced.ncols() != p_reduced {
        return Err(SurvivalLocationScaleError::DimensionMismatch { reason: format!(
            "survival location-scale covariance lift expected active matrix {p_reduced}x{p_reduced}, got {}x{}",
            cov_reduced.nrows(),
            cov_reduced.ncols()
        ) }.into());
    }
    Ok(finalization_gauge.lift_covariance(cov_reduced))
}
