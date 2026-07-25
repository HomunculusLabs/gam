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
    pub(crate) eigenvectors: Array2<f64>,
    pub(crate) inv_sqrt_eigenvalues: Array1<f64>,
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
    let mut eigenvectors = Array2::<f64>::zeros((covariance.nrows(), active.len()));
    let mut inv_sqrt_eigenvalues = Array1::<f64>::zeros(active.len());
    for (col, (idx, sqrt_ev)) in active.into_iter().enumerate() {
        eigenvectors
            .column_mut(col)
            .assign(&eigenvectors_full.column(idx));
        factor
            .column_mut(col)
            .assign(&(&eigenvectors_full.column(idx) * sqrt_ev));
        inv_sqrt_eigenvalues[col] = 1.0 / sqrt_ev;
    }

    Ok(LowRankGaussianFactor {
        factor,
        eigenvectors,
        inv_sqrt_eigenvalues,
    })
}

pub(crate) fn apply_low_rank_gaussian_factor3(
    mu: [f64; 3],
    factor: &Array2<f64>,
    z: &[f64],
) -> [f64; 3] {
    let mut x = mu;
    for row in 0..3 {
        for (col, &latent) in z.iter().enumerate() {
            x[row] += factor[[row, col]] * latent;
        }
    }
    x
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
    F: Fn([f64; 3], &[f64]) -> Result<(f64, f64), String>,
{
    let factorization = factorize_psd_covariance(&covariance3_to_array2(covariance), label)?;
    match factorization.factor.ncols() {
        0 => integrand(mu, &[]),
        1 => crate::quadrature::normal_expectation_nd_adaptive_result::<1, _, _, String>(
            quadctx,
            [0.0],
            [[1.0]],
            max_n,
            |z| {
                let latent = [z[0]];
                integrand(
                    apply_low_rank_gaussian_factor3(mu, &factorization.factor, &latent),
                    &latent,
                )
            },
        ),
        2 => crate::quadrature::normal_expectation_nd_adaptive_result::<2, _, _, String>(
            quadctx,
            [0.0, 0.0],
            [[1.0, 0.0], [0.0, 1.0]],
            max_n,
            |z| {
                let latent = [z[0], z[1]];
                integrand(
                    apply_low_rank_gaussian_factor3(mu, &factorization.factor, &latent),
                    &latent,
                )
            },
        ),
        3 => crate::quadrature::normal_expectation_nd_adaptive_result::<3, _, _, String>(
            quadctx,
            [0.0, 0.0, 0.0],
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            max_n,
            |z| {
                let latent = [z[0], z[1], z[2]];
                integrand(
                    apply_low_rank_gaussian_factor3(mu, &factorization.factor, &latent),
                    &latent,
                )
            },
        ),
        rank => Err(SurvivalLocationScaleError::InternalInvariant {
            reason: format!("{label} unexpectedly has rank {rank} > 3"),
        }
        .into()),
    }
}

/// Symmetric fraction-to-boundary clip of ONE coordinate's realized
/// displacement in a `β ≥ 0` cone-constrained block (#2390, pattern from
/// #2375).
///
/// Returns `α_j · d_j` for the largest `α_j ∈ [0, 1]` that keeps BOTH
/// `β̂_j + α_j·d_j` and `β̂_j − α_j·d_j` on the half-line `β_j ≥ 0`:
///
/// ```text
///   α_j = min( 1,  max(β̂_j, 0) / |d_j| ),        α_j = 1 when d_j = 0
///   α_j · d_j = d_j                    if |d_j| ≤ max(β̂_j, 0)
///             = sign(d_j)·max(β̂_j, 0)  otherwise
/// ```
///
/// The clipped form is returned rather than the factor because the second
/// branch is then EXACT: `β̂_j ± sign(d_j)·β̂_j` is `0` or `2·β̂_j` with no
/// rounding, so no realized coordinate can land an ulp below the wall the way
/// `β̂_j + (β̂_j/|d_j|)·d_j` can.
///
/// Depending only on `|d_j|` makes the clip sign-symmetric (`clip(β̂_j, −d_j) =
/// −clip(β̂_j, d_j)`), so a symmetric quadrature rule displaced by the clipped
/// amount stays symmetric about `β̂` and the posterior mean of every linear
/// functional is exactly unbiased. A coordinate already pinned at its wall
/// (`β̂_j ≤ 0` from round-off, with `d_j ≠ 0`) collapses ITS OWN displacement to
/// zero rather than admitting an infeasible vector; an interior coordinate
/// (`|d_j| ≤ β̂_j`) is returned bit-for-bit unchanged. A non-finite `d_j` is
/// passed through so it fails loudly downstream instead of being silently
/// sanitized to the wall.
///
/// # Why per coordinate, and not one factor for the whole block
///
/// The cone the fit certifies for the monotone link-wiggle block is `A = I`,
/// `b = 0` (`monotone_wiggle_nonnegative_constraints`) — a Cartesian product
/// of independent half-lines. Coordinate `j`'s wall constrains coordinate `j`
/// and nothing else, so the feasibility clip is separable. A single global
/// `min_j` factor is the fraction-to-boundary rule for a step along ONE ray of
/// a coupled polytope; #2375's spherical-radial cubature nodes
/// (`β̂ ± α_k·f_{·,k}`) genuinely have that shape and correctly carry one
/// factor per direction. The conditional-mean displacement here does not: each
/// coordinate of `regression · z` moves independently, and a global factor
/// lets the tightest coordinate's wall govern every other coordinate.
///
/// That is not conservatism, it is silent erasure. The terminal covariance is
/// already computed on the ACTIVE FACE (`Σ = Z (ZᵀHZ)⁻¹ Zᵀ`, zero rows and
/// columns for constraints tight within `ACTIVE_SET_WORKING_FACE_TOL = 1e-10`),
/// so a genuinely PINNED coordinate arrives with an exactly-zero covariance row
/// and contributes `d_j = 0` — it never binds a global minimum in the first
/// place. The coordinates that DO bind are the near-wall but still-slack ones
/// just outside that band, and the inner constrained solve's own KKT tolerance
/// band is `1e-6·scale + 1e-10` (see `MONOTONE_WIGGLE_ACTIVE_SET_TOL`), so
/// that region is routinely occupied. For such a coordinate
/// `max(β̂_j, 0)/|d_j|` is on the order of `1e-8`; a global factor multiplies
/// EVERY coordinate's displacement by it, freezing the conditional mean at
/// `β̂_w` for every latent node. The response-moment integral then degenerates
/// to a plug-in at the mode and drops the entire link-wiggle ↔
/// `(h, threshold, log σ)` cross-covariance — a wrong number with no symptom,
/// which is the #2385 shape all over again.
#[inline]
pub(crate) fn cone_clipped_coordinate_displacement(beta: f64, displacement: f64) -> f64 {
    let wall = beta.max(0.0);
    if displacement.abs() > wall {
        wall.copysign(displacement)
    } else {
        displacement
    }
}

/// `[0, ∞)`-truncated Gaussian expectation of a fallible pair integrand
/// (#2390 layer 2): `E[f(w) | w ≥ 0]` for `w ~ N(mean, sd²)`.
///
/// The feasible image of the monotone I-spline cone under a non-negative basis
/// row is exactly `w ≥ 0`, so the scalar link-wiggle integral must not spend
/// mass on `w < 0` — predictor values no feasible model produces.
///
/// # The rule
///
/// Work in the standardized coordinate `y = (w − mean)/sd`, so with
/// `r = mean/sd` the retained set is `y ≥ −r` and `w = sd·(y − (−r))` is the
/// distance above the wall. The quadrature integrates the OFFSET ABOVE THE WALL
/// `δ = y + r` on a finite interval, against the Gaussian weight, and divides by
/// the same quadrature's own mass:
///
/// ```text
///   E[f | w ≥ 0]  =  Σ ω_i f(sd·δ_i)  /  Σ ω_i ,
///   ω_i = weight_i · exp(−½·(y_i² − y_ref²)),   y_i = −r + δ_i
/// ```
///
/// Three properties this buys, each of which the previous formulation lacked:
///
/// * **No singularity.** The integrand is `f` against a Gaussian on a bounded
///   interval — entire wherever `f` is. Gauss–Legendre converges geometrically.
/// * **No cancellation.** The sample is `sd·δ` built from the offset directly,
///   never `mean + sd·y`, which for a wall far below the mean is a difference of
///   two large nearly-equal numbers.
/// * **Exact measure.** Dividing by the quadrature's own mass makes a constant
///   integrand integrate to itself bit-exactly and cancels the endpoint
///   truncation, so no residual `1 − ε` bias is carried into the moments.
///
/// The scale `y_ref = clamp(0, y_lo, y_hi)` is where the Gaussian peaks on the
/// interval; factoring it out of every weight keeps them `O(1)` even 100σ into
/// the tail, where the unscaled `φ(y)` would underflow. It cancels in the ratio.
///
/// # Why the limits are taken in log space
///
/// `y_hi` is the point beyond which the CONDITIONAL survival is `ε`, obtained
/// without ever forming a probability-space difference:
/// `y_hi = −Φ⁻¹(exp(ln ε + ln Φ(r)))` through
/// [`gam_math::probability::standard_normal_quantile_from_log_cdf`]. That is
/// what keeps a deeply negative mean from collapsing: at `r = −10` the interval
/// is the genuine sliver `[10, 13.2]`, not the point mass that `1 − Φ(10) = 0`
/// would suggest. The lower limit is `max(−r, −y_hi)` — the wall, unless the
/// wall sits so far below the mean that the Gaussian's own lower tail dies
/// first, in which case extending to it would only stretch the interval and
/// waste resolution on mass of size `ε`.
///
/// # What was wrong before
///
/// The previous rule integrated `∫₀^∞ f(w(s)) e^{−s} ds` in the log-survival
/// coordinate `s`, with `w(s) = sd·(r − Φ⁻¹(exp(ln Φ(r) − s)))`. That removed
/// the inverse-CDF endpoint singularity of a uniform-probability rule, but it
/// introduced a worse one: `w(s)` grows like `sd·√(2(s − ln Φ(r)))`, a branch
/// point at `s = ln Φ(r)`, which is just OUTSIDE `[0, s_max]` and approaches the
/// endpoint as `r` grows. That collapses the Bernstein ellipse and with it
/// Gauss–Legendre's convergence rate. Measured against the closed form at 32
/// nodes: `1.9e-7` relative at `r = 0.43`, and `1.5e-3` at `r = 5` — the
/// DEEP-INTERIOR case, which the old documentation claimed was "bit-identical to
/// the current rule within f64" and which is the regime production actually
/// lives in (a wiggle coefficient comfortably inside the cone). The rule above
/// is at machine precision across the same cases at the same node count.
pub(crate) fn truncated_nonnegative_normal_expectation_pair(
    mean: f64,
    sd: f64,
    f: impl Fn(f64) -> Result<(f64, f64), String>,
) -> Result<(f64, f64), String> {
    if !mean.is_finite() {
        return Err(format!(
            "truncated-normal expectation requires a finite mean, got {mean}"
        ));
    }
    if sd == 0.0 {
        return f(mean.max(0.0));
    }
    if !(sd.is_finite() && sd > 0.0) {
        return Err(format!(
            "truncated-normal expectation requires a finite non-negative sd, got {sd}"
        ));
    }
    let standardized_mean = mean / sd;
    if standardized_mean == f64::INFINITY {
        return f(mean);
    }
    if standardized_mean == f64::NEG_INFINITY {
        return f(0.0);
    }
    let log_retained = gam_math::probability::normal_logcdf(standardized_mean);
    let log_omitted = f64::EPSILON.ln() + log_retained;
    let upper = -gam_math::probability::standard_normal_quantile_from_log_cdf(log_omitted)
        .map_err(|e| format!("truncated-normal log-quantile at log_p={log_omitted}: {e}"))?;
    if !upper.is_finite() {
        return Err(format!(
            "truncated-normal expectation could not locate its upper limit for \
             mean={mean}, sd={sd} (log-quantile returned {upper})"
        ));
    }
    let wall = -standardized_mean;
    let lower = wall.max(-upper);
    // A sliver narrower than the rounding of its own endpoints carries no
    // information. `upper` solves a log-CDF equation whose residual is divided
    // by a Mills ratio of order `|wall|`, so it is itself determined only to
    // within an ulp or so of `wall`; integrating a width below that would be
    // integrating the endpoint's own rounding error. There the conditional law
    // IS the wall to everything f64 can represent — at `mean = −1e6, sd = 1e-3`
    // the exact `E[w | w ≥ 0] = σ²/|μ|` is `1e-12` against a wall coordinate of
    // `1e9`, whose ulp is `1.2e-7`. This is NOT the deep-tail collapse #2390
    // removed: at `r = −10` the sliver is `[10, 13.2]`, wider than its rounding
    // by fourteen orders, and is integrated rather than collapsed.
    if upper - lower <= 4.0 * f64::EPSILON * wall.abs().max(1.0) {
        return f(mean.max(0.0));
    }
    let reference = 0.0_f64.clamp(lower, upper);
    let (nodes, weights) = gam_math::special::gauss_legendre(32);
    let half_width = 0.5 * (upper - lower);
    let midpoint = 0.5 * (upper + lower);
    let mut first = 0.0;
    let mut second = 0.0;
    let mut mass = 0.0;
    for (t, wgt) in nodes.iter().zip(weights.iter()) {
        let y = half_width * t + midpoint;
        // `(y − y_ref)(y + y_ref)` rather than `y² − y_ref²`: deep in the tail
        // both squares are large and nearly equal, and the difference of the
        // squares loses the exponent's leading digits. The factored form keeps
        // it to full relative precision.
        let weight =
            half_width * wgt * (-0.5 * (y - reference) * (y + reference)).exp();
        let (f1, f2) = f(sd * (y - wall))?;
        first += weight * f1;
        second += weight * f2;
        mass += weight;
    }
    if !(mass.is_finite() && mass > 0.0) {
        return Err(format!(
            "truncated-normal expectation resolved no conditional mass on [{lower}, {upper}] \
             for mean={mean}, sd={sd}"
        ));
    }
    Ok((first / mass, second / mass))
}

// Exact response moments must stay in the original Gaussian coordinates:
// [h, threshold, log_sigma] for non-wiggle predictions, with a nested
// conditional Gaussian over the scalar link-wiggle contribution when present.
pub(crate) fn exact_survival_response_moments_row(
    input: &SurvivalLocationScalePredictInput,
    fit: &UnifiedFitResult,
    covariance: &Array2<f64>,
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

    if let (Some(beta_w), Some(wiggle_range)) = (beta_link_wiggle.as_ref(), wiggle) {
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

        let htl_factor = factorize_psd_covariance(
            &covariance3_to_array2(cov_htl),
            "survival response-moment projected covariance",
        )?;

        let cov_wy = {
            let mut out = Array2::<f64>::zeros((pw, 3));
            let cov_wh = covariance
                .slice(s![
                    wiggle_range.start..wiggle_range.end,
                    time.start..time.end
                ])
                .to_owned();
            let cov_wt = covariance
                .slice(s![
                    wiggle_range.start..wiggle_range.end,
                    threshold.start..threshold.end
                ])
                .to_owned();
            let cov_wl = covariance
                .slice(s![
                    wiggle_range.start..wiggle_range.end,
                    log_sigma.start..log_sigma.end
                ])
                .to_owned();
            out.column_mut(0).assign(&cov_wh.dot(&a_h));
            out.column_mut(1).assign(&cov_wt.dot(&a_t));
            out.column_mut(2).assign(&cov_wl.dot(&a_ls));
            out
        };
        let cov_ww = covariance
            .slice(s![
                wiggle_range.start..wiggle_range.end,
                wiggle_range.start..wiggle_range.end
            ])
            .to_owned();
        let mut regression = cov_wy.dot(&htl_factor.eigenvectors);
        for col in 0..regression.ncols() {
            let scale = htl_factor.inv_sqrt_eigenvalues[col];
            regression
                .column_mut(col)
                .mapv_inplace(|value| value * scale);
        }
        let cov_cond =
            symmetrize_and_clip_covariance(&(cov_ww - regression.dot(&regression.t().to_owned())));

        return low_rank_normal_expectation_pair_3d_result(
            quadctx,
            mu,
            cov_htl,
            15,
            "survival response-moment projected covariance",
            |x, z| {
                // #2390 (#2385 instance, pattern from #2375): `cond_mean` is a
                // REALIZED coefficient vector for the cone-constrained
                // link-wiggle block (`β_w ≥ 0`, the structural monotone
                // I-spline warp the fit certified). A cone coordinate at or
                // near its wall has `β̂_w,j ≈ 0`, so an unconstrained
                // conditional displacement manufactures a warp the model does
                // not admit. Clip each coordinate's displacement by its OWN
                // symmetric fraction-to-boundary factor: the cone is a product
                // of independent half-lines, so coordinate `j`'s wall binds
                // coordinate `j` alone and a single global factor would let the
                // tightest wall freeze the whole block (see
                // `cone_clipped_coordinate_displacement`). The clip depends
                // only on `|d_j|`, so the realized vector at `−z` is the exact
                // mirror of the one at `+z` and the rule stays symmetric about
                // `β̂` (the posterior mean of linear functionals stays exactly
                // unbiased). An interior `β̂` with modest spread leaves every
                // coordinate untouched, recovering the unconstrained rule
                // verbatim.
                let mut cond_mean = beta_w.to_owned();
                for j in 0..pw {
                    let mut displacement = 0.0;
                    for (col, &latent) in z.iter().enumerate() {
                        displacement += regression[[j, col]] * latent;
                    }
                    cond_mean[j] +=
                        cone_clipped_coordinate_displacement(beta_w[j], displacement);
                }
                let q0 = survival_q0_from_eta(x[1], x[2]);
                let q0_arr = Array1::from_vec(vec![q0]);
                let basis = survival_wiggle_basis_with_options(
                    q0_arr.view(),
                    knots,
                    degree,
                    BasisOptions::value(),
                )?;
                if basis.ncols() != cond_mean.len() {
                    return Err(SurvivalLocationScaleError::DimensionMismatch { reason: format!(
                        "predict_survival_location_scale: link-wiggle basis/beta mismatch: {} vs {}",
                        basis.ncols(),
                        cond_mean.len()
                    ) }.into());
                }
                let b = basis.row(0).to_owned();
                let w_mean = b.dot(&cond_mean);
                let w_var = b.dot(&cov_cond.dot(&b)).max(0.0);
                // #2390 layer 2: the wiggle contribution's feasible image is
                // exactly `w ≥ 0` (non-negative I-spline basis row against the
                // β_w ≥ 0 cone), so the scalar integral runs over the
                // `[0, ∞)`-truncated conditional Gaussian — never over
                // predictor values no feasible model produces.
                truncated_nonnegative_normal_expectation_pair(
                    w_mean,
                    w_var.sqrt(),
                    |w| {
                        let p = inverse_link_survival_prob_checked(
                            &input.inverse_link,
                            x[0] + q0 + w,
                        )?;
                        Ok((p, p * p))
                    },
                )
            },
        )
        .map(|(first, second)| (first.clamp(0.0, 1.0), second.clamp(0.0, 1.0)));
    }

    low_rank_normal_expectation_pair_3d_result(
        quadctx,
        mu,
        cov_htl,
        15,
        "survival response-moment projected covariance",
        |x, _| {
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

    let x_threshold_dense = input.x_threshold.to_dense_arc();
    let x_log_sigma_dense = input.x_log_sigma.to_dense_arc();
    let mut first = Array1::<f64>::zeros(n);
    let mut second = Array1::<f64>::zeros(n);
    // Build a single QuadratureContext up front and share it across all
    // chunks.  Per-chunk construction wastes work (each chunk's first call
    // re-derives the Gauss-Hermite rule from scratch via OnceLock) and risks
    // the OnceLock-inside-rayon deadlock pattern (see repo memory) if the
    // rule init were ever to spawn nested parallel work.  Warm the rule sizes
    // that the per-row evaluator actually uses (15 for the projected 3D
    // path, 21 for the 1D wiggle fallback) so the worker threads only hit
    // the cached rule lookup.
    let quadctx = crate::quadrature::QuadratureContext::new();
    {
        // Warm GH rule caches on the calling thread with cheap probes.
        crate::quadrature::normal_expectation_nd_adaptive_result::<1, _, _, String>(
            &quadctx,
            [0.0_f64],
            [[1.0_f64]],
            21,
            |_x: [f64; 1]| Ok((0.0_f64, 0.0_f64)),
        )?;
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
