//! First two moments of the constrained (truncated) Laplace posterior.
//!
//! # What the posterior is when inequality constraints are active
//!
//! A shape- or box-constrained fit restricts the PRIOR SUPPORT to the feasible
//! set `C = {β : Aβ ≥ b}` (a user asserting monotonicity is asserting that the
//! non-monotone coefficient vectors have prior mass zero). The posterior is
//! therefore the penalized likelihood restricted to `C` and renormalized:
//!
//! ```text
//! π(β | y) ∝ exp(−ℓ_p(β)) · 1_C(β)
//! ```
//!
//! Expanding `ℓ_p` at the constrained mode `β̂` — where the KKT conditions give
//! `g := ∇ℓ_p(β̂) = A_actᵀλ` with `λ ≥ 0` and `H := ∇²ℓ_p(β̂)` — and completing
//! the square gives, to Laplace order,
//!
//! ```text
//! π ≈ N(β; β_unc, Σ) truncated to C,   β_unc = β̂ − Σ·∇ℓ_p(β̂),   Σ = φ·H⁻¹
//! ```
//!
//! The Gaussian is centred at the UNCONSTRAINED centre `β_unc`, not at the
//! boundary mode: truncating a Gaussian does not move its pre-truncation mean,
//! so a law centred at the KKT mode is a different (half-normal-shaped)
//! distribution. This is the same law `sample_truncated_gaussian_posterior`
//! draws from, and for the same reason.
//!
//! # The exact decomposition this module computes
//!
//! Let `A` be the retained constraint rows (`q × p`), `W = A Σ Aᵀ` and
//! `G = Σ Aᵀ W⁻¹`, and let `P = I − G A` be the `Σ⁻¹`-orthogonal projector onto
//! `null(A)`. Split the deviation `d = β − β_unc` as `d = P d + G(A d)` and set
//! `t = P d`, `u = A β − b = (A β_unc − b) + A d`. Under the UNTRUNCATED law:
//!
//! * `Cov(t, u) = P Σ Aᵀ = Σ Aᵀ − Σ Aᵀ W⁻¹ (A Σ Aᵀ) = 0`, so `t` and `u` are
//!   independent;
//! * `E[t] = 0` and `Cov(t) = P Σ Pᵀ = Z(Zᵀ Σ⁻¹ Z)⁻¹Zᵀ` for any basis `Z` of
//!   `null(A)`;
//! * `u ~ N(A β_unc − b, W)`.
//!
//! Feasibility is exactly `u ≥ 0`, which constrains ONLY `u`. Because `t` is
//! independent of `u`, conditioning leaves `t` untouched, so the truncated
//! moments are exactly
//!
//! ```text
//! E_π[β]   = β_unc + G·(E[u] − E_untrunc[u])
//! Cov_π[β] = Σ − G·(W − Cov[u])·Gᵀ
//! ```
//!
//! with `(E[u], Cov[u])` the first two moments of the `q`-dimensional Gaussian
//! `N(A β_unc − b, W)` restricted to the orthant `u ≥ 0`.
//!
//! # Why the two obvious answers are both wrong
//!
//! Reading the covariance formula at its two endpoints:
//!
//! * `Cov[u] := W` (no truncation) returns `Σ` — the full unconstrained
//!   covariance. This is the answer that ignores the constraint entirely, and
//!   it over-states the spread along every constrained direction.
//! * `Cov[u] := 0` returns `Σ − G W Gᵀ = P Σ Pᵀ`, the active-face reduction
//!   `Z(ZᵀHZ)⁻¹Zᵀ`, which reports EXACTLY ZERO variance for a fully pinned
//!   coordinate. That is the `λ → ∞` limit: it is correct only for a
//!   constraint whose multiplier is infinite.
//!
//! An EQUALITY / gauge / identifiability constraint genuinely deletes a degree
//! of freedom and zero variance is right for it — but those are absorbed into
//! the basis in this codebase, so the deleted direction is not a coordinate at
//! all. An INEQUALITY does not delete a direction, it halves one: the posterior
//! along the constraint normal is supported on a half-line, its mode sits at
//! the endpoint, and its variance does not. In the scalar case `X ~ N(μ,σ²)`
//! truncated to `[0,∞)` with `α = −μ/σ` and `λ(α) = φ(α)/Φ̄(α)`,
//! `Var(X) = σ²(1 + αλ − λ²)`, which is `0.3634σ²` when the mode sits exactly
//! on the bound and decays as `σ²/α²` only as the multiplier diverges.
//!
//! Since truncating a Gaussian to a convex set can only shrink its covariance,
//! `P Σ Pᵀ ≺ Cov_π[β] ≺ Σ` strictly at every finite multiplier.
//!
//! # No tightness predicate
//!
//! Nothing here classifies a row as "active". A row enters through its
//! standardized slack `s_j = (a_jᵀβ_unc − b_j)/sd_j` and its contribution
//! varies smoothly with it, so a row at slack `1e-9` and a row at slack `0`
//! give nearly the same answer instead of differing by a full `σ²`. The only
//! cut is dropping rows whose truncated mass `Φ̄(s_j)` is below double-precision
//! resolution, a bound read off `f64::EPSILON` rather than tuned.

use gam_math::probability::{
    normal_cdf, normal_logsf, signed_probit_logcdf_and_mills_ratio, standard_normal_quantile,
    standard_normal_quantile_from_log_cdf,
};
use gam_problem::LinearInequalityConstraints;
use ndarray::{Array1, Array2, ArrayView2};
use serde::{Deserialize, Serialize};

/// Relative accuracy demanded of the orthant-moment cubature, measured against
/// the PRE-TRUNCATION scale `sd_i = sqrt(W_ii)` so the criterion is invariant
/// to how the constraint rows happen to be scaled.
///
/// The target is set by what the number is FOR. The cubature resolves `Δ`, the
/// variance the truncation removes, and the reported variance is `Σ − GΔGᵀ`
/// with the removed part never exceeding the total. A relative error `ε` in `Δ`
/// therefore moves a reported variance by at most `ε` relative, and a reported
/// standard error by at most `ε/2`. At `1e-3` no reported interval half-width
/// can move in its fourth significant digit — far below the Laplace
/// approximation's own error, and far below any resolution a credible interval
/// carries. Demanding more is not free: the orthant integrand is unbounded at
/// the cube boundary (the second moment grows like `log 1/(1−x)`), so the
/// quasi-Monte-Carlo rate is closer to `N⁻¹` than the `N⁻²` a bounded
/// integrand would give, and every extra digit costs two decades of nodes.
const ORTHANT_MOMENT_RELATIVE_TOLERANCE: f64 = 1e-3;

/// Cubature point count at the first pass. Each subsequent pass EXTENDS the
/// node set to twice its length — the Kronecker sequence is a prefix sequence,
/// so refinement reuses every node already evaluated — until the moments agree
/// to [`ORTHANT_MOMENT_RELATIVE_TOLERANCE`]. This is a starting point rather
/// than a budget.
const ORTHANT_MOMENT_INITIAL_POINTS: usize = 1 << 11;

/// Node count past which the cubature is declared non-convergent and the caller
/// gets an error rather than an unconverged covariance. Silently reporting the
/// last iterate would ship an uncertified number into every interval built from
/// this fit.
const ORTHANT_MOMENT_MAXIMUM_POINTS: usize = 1 << 20;

/// The low-rank correction that turns the unconstrained Laplace covariance into
/// the truncated-posterior covariance.
///
/// Carrying the factored form rather than a dense `p × p` matrix lets a
/// consumer that never materializes `Σ` (the factorized inference path, the
/// prediction backends) apply the same correction with `q` extra solves:
/// `xᵀΣ_π x = xᵀΣx − ‖Δ^{1/2} Gᵀ x‖²`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConstrainedPosteriorCorrection {
    /// `G = Σ Aᵀ W⁻¹`, `p × q`.
    pub lift: Array2<f64>,
    /// `Δ = W − Cov[u] ⪰ 0`, `q × q`: the variance the truncation removes from
    /// the constraint-normal coordinates.
    pub removed_normal_variance: Array2<f64>,
    /// `E[u] − E_untrunc[u]`, `q`: how far truncation moves the posterior mean
    /// in constraint-normal coordinates. Positive componentwise for a
    /// half-line coordinate — the posterior mean is interior even when the mode
    /// is not — and of either sign for a coordinate that also carries an upper
    /// limit, where the far face pulls the mean back down.
    pub normal_mean_shift: Array1<f64>,
    /// Indices, into the caller's constraint system, of the rows retained.
    pub rows: Vec<usize>,
    /// Upper limit on each retained coordinate: `u_k ≤ normal_upper_limits[k]`,
    /// with `f64::INFINITY` where the coordinate is a half-line.
    ///
    /// A two-sided coefficient bound `l ≤ β_j ≤ u` arrives as two rows whose
    /// normals are exactly anti-parallel. The second carries no constraint-normal
    /// DIRECTION the first does not already carry, so the rank filter drops it —
    /// correctly, as a direction. It is still a constraint, and this is where it
    /// is kept (#2523).
    ///
    /// `#[serde(default)]` with an empty vector reading as "every retained
    /// coordinate is a half-line" is the encoding of a model saved before upper
    /// limits existed, which is exactly what those models meant. Live
    /// constructions always carry one entry per retained row.
    #[serde(default)]
    pub normal_upper_limits: Vec<f64>,
}

impl ConstrainedPosteriorCorrection {
    /// `Σ ← Σ − G Δ Gᵀ`, in place. The correction is rank `q`, so this never
    /// allocates a second `p × p` matrix next to the one being corrected.
    pub fn apply_to_covariance_in_place(&self, covariance: &mut Array2<f64>) {
        let scaled = self.lift.dot(&self.removed_normal_variance);
        let p = covariance.nrows();
        for i in 0..p {
            for j in 0..=i {
                let removed = scaled.row(i).dot(&self.lift.row(j));
                covariance[[i, j]] -= removed;
                if i != j {
                    covariance[[j, i]] = covariance[[i, j]];
                }
            }
        }
    }

    /// `Σ_π = Σ − G Δ Gᵀ`.
    pub fn apply_to_covariance(&self, covariance: &Array2<f64>) -> Array2<f64> {
        let mut corrected = covariance.clone();
        self.apply_to_covariance_in_place(&mut corrected);
        corrected
    }

    /// `diag(G Δ Gᵀ)` — the per-coefficient variance the truncation removes,
    /// for consumers that only ever build the covariance diagonal.
    pub fn removed_variance_diagonal(&self) -> Array1<f64> {
        let scaled = self.lift.dot(&self.removed_normal_variance);
        let p = self.lift.nrows();
        let mut diagonal = Array1::<f64>::zeros(p);
        for i in 0..p {
            diagonal[i] = scaled.row(i).dot(&self.lift.row(i));
        }
        diagonal
    }

    /// `E_π[β] = β_unc + G·(E[u] − E_untrunc[u])`.
    pub fn posterior_mean(&self, unconstrained_center: &Array1<f64>) -> Array1<f64> {
        unconstrained_center + &self.lift.dot(&self.normal_mean_shift)
    }

    /// Upper limit per retained coordinate, materializing the legacy encoding of
    /// [`Self::normal_upper_limits`].
    pub fn upper_limits(&self) -> Vec<f64> {
        if self.normal_upper_limits.is_empty() {
            vec![f64::INFINITY; self.rows.len()]
        } else {
            self.normal_upper_limits.clone()
        }
    }
}

/// Persisted identity of an inequality-truncated Laplace posterior.
///
/// These three objects must remain distinct:
///
/// * `mode` is the feasible optimizer solution and the reflective sampler's
///   valid starting point;
/// * `unconstrained_center` is the centre of the ambient Gaussian before
///   truncation and therefore the reflective sampler's target centre;
/// * the user-facing coefficient vector is the ambient centre plus the
///   retained correction's normal-coordinate mean shift (or exactly the
///   ambient centre when truncation is invisible at f64 resolution).
///
/// Keeping the two locations next to the factored moment correction prevents a
/// saved model from re-deriving either location from row evidence or from
/// treating the reported posterior mean as though it were the optimizer mode.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConstrainedPosteriorGeometry {
    /// Exact inequality system `Aβ ≥ b` in the same coefficient frame as the
    /// locations, correction lift, and ambient precision.
    pub constraints: LinearInequalityConstraints,
    pub mode: Array1<f64>,
    pub unconstrained_center: Array1<f64>,
    /// Moment correction when at least one inequality changes the answer at
    /// f64 resolution. `None` still records that an inequality system was
    /// fitted; it means the ambient centre is far enough inside every row that
    /// truncation is numerically invisible.
    pub correction: Option<ConstrainedPosteriorCorrection>,
}

impl ConstrainedPosteriorGeometry {
    pub fn posterior_mean(&self) -> Array1<f64> {
        self.correction
            .as_ref()
            .map(|correction| correction.posterior_mean(&self.unconstrained_center))
            .unwrap_or_else(|| self.unconstrained_center.clone())
    }

    pub fn validate_for_dimension(&self, dimension: usize) -> Result<(), String> {
        if self.constraints.a.ncols() != dimension
            || self.constraints.a.nrows() != self.constraints.b.len()
        {
            return Err(format!(
                "constrained posterior inequalities have shape {}x{} with {} bounds, expected {dimension} columns",
                self.constraints.a.nrows(),
                self.constraints.a.ncols(),
                self.constraints.b.len()
            ));
        }
        if self.mode.len() != dimension || self.unconstrained_center.len() != dimension {
            return Err(format!(
                "constrained posterior locations have lengths mode={} and center={}, expected {dimension}",
                self.mode.len(),
                self.unconstrained_center.len()
            ));
        }
        if self
            .mode
            .iter()
            .chain(self.unconstrained_center.iter())
            .chain(self.constraints.a.iter())
            .chain(self.constraints.b.iter())
            .any(|value| !value.is_finite())
        {
            return Err("constrained posterior geometry contains a non-finite value".to_string());
        }
        if let Some(correction) = self.correction.as_ref() {
            let q = correction.lift.ncols();
            if correction.lift.nrows() != dimension {
                return Err(format!(
                    "constrained posterior lift has {} rows, expected {dimension}",
                    correction.lift.nrows()
                ));
            }
            if correction.removed_normal_variance.dim() != (q, q)
                || correction.normal_mean_shift.len() != q
                || correction.rows.len() != q
            {
                return Err(format!(
                    "constrained posterior normal geometry is inconsistent: lift={}x{q}, removed={:?}, mean={}, rows={}",
                    correction.lift.nrows(),
                    correction.removed_normal_variance.dim(),
                    correction.normal_mean_shift.len(),
                    correction.rows.len()
                ));
            }
            let mut unique_rows = correction.rows.clone();
            unique_rows.sort_unstable();
            unique_rows.dedup();
            if unique_rows.len() != q
                || unique_rows
                    .iter()
                    .any(|&row| row >= self.constraints.a.nrows())
            {
                return Err(format!(
                    "constrained posterior retained rows {:?} are not unique valid indices for {} inequalities",
                    correction.rows,
                    self.constraints.a.nrows()
                ));
            }
            if correction
                .lift
                .iter()
                .chain(correction.removed_normal_variance.iter())
                .chain(correction.normal_mean_shift.iter())
                .any(|value| !value.is_finite())
            {
                return Err(
                    "constrained posterior correction contains a non-finite value".to_string()
                );
            }
            if !correction.normal_upper_limits.is_empty()
                && correction.normal_upper_limits.len() != q
            {
                return Err(format!(
                    "constrained posterior carries {} upper limits for {q} retained rows",
                    correction.normal_upper_limits.len()
                ));
            }
            // `+∞` is the half-line coordinate and is admissible; anything at or
            // below the wall would make the retained region empty.
            if correction
                .normal_upper_limits
                .iter()
                .any(|limit| !(*limit > 0.0))
            {
                return Err(format!(
                    "constrained posterior upper limits must be positive, got {:?}",
                    correction.normal_upper_limits
                ));
            }
        }
        Ok(())
    }
}

/// Equal-tailed interval for one linear projection of an inequality-truncated
/// Gaussian posterior.
///
/// `ambient_covariance` is the pre-truncation covariance `Σ` in the active
/// coefficient frame and `contrast` defines the scalar `cᵀβ`.  The affine
/// shift of a saved coefficient gauge is deliberately not accepted here:
/// callers add that deterministic shift to both returned endpoints.
///
/// The decomposition in this module makes the projection
///
/// ```text
/// cᵀβ = cᵀβ_unc + cᵀt + (Gᵀc)ᵀ(u - E_untrunc[u]),
/// ```
///
/// where `cᵀt` is an independent scalar Gaussian and `u` is the retained
/// orthant-truncated Gaussian.  The interval therefore comes from the quantiles
/// of that convolution, not from `posterior_mean ± z·posterior_sd`.
pub fn constrained_projection_equal_tailed_interval(
    ambient_covariance: &Array2<f64>,
    geometry: &ConstrainedPosteriorGeometry,
    contrast: &Array1<f64>,
    level: f64,
) -> Result<(f64, f64), String> {
    let p = contrast.len();
    geometry.validate_for_dimension(p)?;
    if ambient_covariance.dim() != (p, p) {
        return Err(format!(
            "constrained projection interval needs a {p}x{p} ambient covariance, got {:?}",
            ambient_covariance.dim()
        ));
    }
    if !(level.is_finite() && level > 0.0 && level < 1.0) {
        return Err(format!(
            "constrained projection interval level must lie in (0, 1), got {level}"
        ));
    }
    if ambient_covariance.iter().any(|value| !value.is_finite())
        || contrast.iter().any(|value| !value.is_finite())
    {
        return Err(
            "constrained projection interval received a non-finite covariance or contrast"
                .to_string(),
        );
    }

    let ambient_mean = contrast.dot(&geometry.unconstrained_center);
    let sigma_c = ambient_covariance.dot(contrast);
    let ambient_variance = contrast.dot(&sigma_c);
    let covariance_scale = ambient_covariance
        .diag()
        .iter()
        .map(|value| value.abs())
        .fold(f64::MIN_POSITIVE, f64::max);
    let contrast_scale = contrast.dot(contrast).max(f64::MIN_POSITIVE);
    let variance_floor =
        (p.max(1) as f64) * f64::EPSILON * covariance_scale * contrast_scale;
    if ambient_variance < -variance_floor || !ambient_variance.is_finite() {
        return Err(format!(
            "constrained projection interval has invalid ambient variance {ambient_variance:.6e}"
        ));
    }
    let ambient_variance = ambient_variance.max(0.0);
    let alpha = 0.5 * (1.0 - level);

    let Some(correction) = geometry.correction.as_ref() else {
        let sd = ambient_variance.sqrt();
        if sd == 0.0 {
            return Ok((ambient_mean, ambient_mean));
        }
        let z = standard_normal_quantile(1.0 - alpha)
            .map_err(|error| format!("constrained projection normal quantile: {error}"))?;
        return Ok((ambient_mean - z * sd, ambient_mean + z * sd));
    };

    let q = correction.rows.len();
    let mut normal_center = Array1::<f64>::zeros(q);
    let mut normal_covariance = Array2::<f64>::zeros((q, q));
    let mut sigma_a = Array2::<f64>::zeros((p, q));
    for (position, &row) in correction.rows.iter().enumerate() {
        let a = geometry.constraints.a.row(row);
        normal_center[position] = a.dot(&geometry.unconstrained_center)
            - geometry.constraints.b[row];
        sigma_a
            .column_mut(position)
            .assign(&ambient_covariance.dot(&a));
    }
    for i in 0..q {
        let ai = geometry.constraints.a.row(correction.rows[i]);
        for j in 0..=i {
            let value = ai.dot(&sigma_a.column(j));
            normal_covariance[[i, j]] = value;
            normal_covariance[[j, i]] = value;
        }
    }

    let projection_lift = correction.lift.t().dot(contrast);
    let normal_component_variance =
        projection_lift.dot(&normal_covariance.dot(&projection_lift));
    let residual_variance = ambient_variance - normal_component_variance;
    let residual_floor = (p.max(q).max(1) as f64)
        * f64::EPSILON
        * ambient_variance.max(normal_component_variance).max(f64::MIN_POSITIVE);
    if residual_variance < -residual_floor || !residual_variance.is_finite() {
        return Err(format!(
            "constrained projection decomposition produced residual variance \
             {residual_variance:.6e} from ambient {ambient_variance:.6e}"
        ));
    }
    let residual_variance = residual_variance.max(0.0);
    let posterior_mean =
        ambient_mean + projection_lift.dot(&correction.normal_mean_shift);
    let upper_limits = correction.upper_limits();
    if upper_limits.len() != q {
        return Err(format!(
            "constrained projection interval: {q} retained rows carry {} upper limits",
            upper_limits.len()
        ));
    }
    if q == 1 && residual_variance == 0.0 && projection_lift[0] != 0.0 {
        let scalar_quantile = |probability: f64| -> Result<f64, String> {
            let normal_probability = if projection_lift[0] > 0.0 {
                probability
            } else {
                1.0 - probability
            };
            let value = scalar_truncated_quantile(
                normal_center[0],
                normal_covariance[[0, 0]],
                upper_limits[0],
                normal_probability,
            )?;
            Ok(ambient_mean + projection_lift[0] * (value - normal_center[0]))
        };
        return Ok((scalar_quantile(alpha)?, scalar_quantile(1.0 - alpha)?));
    }
    let nodes = converged_projection_nodes(
        &normal_center,
        &normal_covariance,
        &upper_limits,
        &projection_lift,
        ambient_mean,
    )?;
    let lower = projection_quantile(
        &nodes,
        residual_variance,
        alpha,
        posterior_mean,
        ambient_variance.sqrt(),
    )?;
    let upper = projection_quantile(
        &nodes,
        residual_variance,
        1.0 - alpha,
        posterior_mean,
        ambient_variance.sqrt(),
    )?;
    Ok((lower, upper))
}

/// Quantile of `N(mean, variance)` restricted to `[0, upper]`.
fn scalar_truncated_quantile(
    mean: f64,
    variance: f64,
    upper: f64,
    probability: f64,
) -> Result<f64, String> {
    if !(variance.is_finite() && variance > 0.0) {
        return Err(format!(
            "scalar truncated quantile needs positive finite variance, got {variance:?}"
        ));
    }
    if !(probability.is_finite() && probability > 0.0 && probability < 1.0) {
        return Err(format!(
            "scalar truncated quantile probability must lie in (0, 1), got {probability}"
        ));
    }
    if !(upper > 0.0) {
        return Err(format!(
            "scalar truncated quantile needs the upper limit above the wall, got {upper:?}"
        ));
    }
    let sd = variance.sqrt();
    let alpha = -mean / sd;
    if !upper.is_finite() {
        // P(Z > z | Z >= alpha) = (1-p) P(Z >= alpha). Work entirely in
        // log-survival space so a deeply pinned face never forms `1-Phi(alpha)`.
        let log_tail = (1.0 - probability).ln() + normal_logsf(alpha);
        let z = -standard_normal_quantile_from_log_cdf(log_tail)
            .map_err(|error| format!("scalar truncated quantile: {error}"))?;
        return Ok(mean + sd * z);
    }
    let beta = (upper - mean) / sd;
    // Same reflection as the moments: put the retained slab in the upper tail so
    // its mass is a difference of directly-evaluated tail probabilities. Under
    // the reflection the probability runs the other way.
    let reflect = alpha + beta < 0.0;
    let (low, high, probability) = if reflect {
        (-beta, -alpha, 1.0 - probability)
    } else {
        (alpha, beta, probability)
    };
    let log_tail_low = normal_logsf(low);
    let removed = normal_logsf(high) - log_tail_low;
    // `Φ̄(z) = Φ̄(low)·(1 − p(1 − e^removed))`, the inversion the cubature uses.
    let log_tail = log_tail_low + (-probability * -removed.exp_m1()).ln_1p();
    let z = -standard_normal_quantile_from_log_cdf(log_tail)
        .map_err(|error| format!("scalar truncated quantile: {error}"))?;
    let z = z.clamp(low, high);
    // Reflected, `z` standardizes `−X` about `−mean`, so `X = mean − sd·z`.
    Ok(if reflect {
        mean - sd * z
    } else {
        mean + sd * z
    })
}

/// Build the truncated-posterior correction for a fit carrying linear
/// inequality constraints, or `None` when no constraint row is close enough to
/// the posterior centre to move the answer at double precision.
///
/// * `covariance` — `Σ`, the PRE-TRUNCATION posterior covariance on the same
///   coefficient frame as `constraints` and `unconstrained_center`. This is the
///   dispersion-scaled `φ·H⁻¹`: truncation is a statement about the posterior's
///   own spread, so it must be applied in the scaled metric, not to `H⁻¹`.
/// * `unconstrained_center` — `β_unc = β̂ − Σ·∇ℓ_p(β̂)`.
/// * `constraints` — `A β ≥ b`.
///
/// `None` is returned when every row's standardized slack exceeds the
/// resolution horizon, which includes the case of a fit whose constraints are
/// all inactive. Callers must then report `Σ` unchanged, bit for bit.
pub fn constrained_posterior_correction_from_covariance(
    covariance: &Array2<f64>,
    unconstrained_center: &Array1<f64>,
    constraints: &LinearInequalityConstraints,
) -> Result<Option<ConstrainedPosteriorCorrection>, String> {
    let p = covariance.nrows();
    if covariance.ncols() != p {
        return Err(format!(
            "constrained posterior correction needs a square covariance, got {}x{}",
            covariance.nrows(),
            covariance.ncols()
        ));
    }
    if constraints.a.ncols() != p {
        return Err(format!(
            "constrained posterior correction: covariance is {p}x{p} but the constraint \
             system has {} columns",
            constraints.a.ncols()
        ));
    }
    let sigma_times_at = covariance.dot(&constraints.a.t());
    constrained_posterior_correction(sigma_times_at.view(), unconstrained_center, constraints)
}

/// Same correction for a caller that never materializes `Σ`.
///
/// Everything the decomposition needs from the covariance is the `p × m` block
/// `Σ Aᵀ` — column `j` is `Σ a_j`, `W_ij = a_iᵀ(Σ a_j)`, and the lift is
/// `(Σ Aᵀ)W⁻¹` — so a factorized inference path supplies `m` solves instead of
/// a `p × p` inverse.
pub fn constrained_posterior_correction(
    sigma_times_constraint_transpose: ArrayView2<'_, f64>,
    unconstrained_center: &Array1<f64>,
    constraints: &LinearInequalityConstraints,
) -> Result<Option<ConstrainedPosteriorCorrection>, String> {
    let p = sigma_times_constraint_transpose.nrows();
    if sigma_times_constraint_transpose.ncols() != constraints.a.nrows() {
        return Err(format!(
            "constrained posterior correction: the constraint system has {} rows but \
             Sigma·Aᵀ has {} columns",
            constraints.a.nrows(),
            sigma_times_constraint_transpose.ncols()
        ));
    }
    if unconstrained_center.len() != p {
        return Err(format!(
            "constrained posterior correction: Sigma·Aᵀ has {p} rows but the centre has \
             length {}",
            unconstrained_center.len()
        ));
    }
    if constraints.a.ncols() != p {
        return Err(format!(
            "constrained posterior correction: Sigma·Aᵀ has {p} rows but the constraint \
             system has {} columns",
            constraints.a.ncols()
        ));
    }

    // A row whose remaining feasible mass `Φ̄(s)` is below `f64::EPSILON` cannot
    // change any moment at double precision, so the horizon is read off the
    // machine epsilon rather than chosen.
    let slack_horizon = -standard_normal_quantile(f64::EPSILON)
        .map_err(|error| format!("resolution horizon for the constraint slack: {error}"))?;

    // Order candidates by standardized slack so the greedy rank filter below
    // keeps the rows that bind hardest when a face carries redundant rows. That
    // ordering is a STATISTICAL choice and it stays: two near-parallel rows with
    // different offsets are not the same constraint, and the tighter one
    // dominates, so retaining the slacker of a pair would quietly relax the
    // constraint by a multiple of its own standard deviation. Ordering by pivot
    // magnitude instead would reveal the conditioning but pay exactly that cost.
    let mut candidates: Vec<(usize, f64, Array1<f64>)> = Vec::new();
    for row_index in 0..constraints.a.nrows() {
        let row = constraints.a.row(row_index).to_owned();
        let sigma_row = sigma_times_constraint_transpose
            .column(row_index)
            .to_owned();
        let variance = row.dot(&sigma_row);
        if !(variance.is_finite() && variance > 0.0) {
            // The constraint normal has no posterior spread at all: the fit
            // cannot move along it, so the truncation removes nothing.
            continue;
        }
        let slack = (row.dot(unconstrained_center) - constraints.b[row_index]) / variance.sqrt();
        if !slack.is_finite() {
            return Err(format!(
                "constraint row {row_index} produced a non-finite standardized slack"
            ));
        }
        if slack < slack_horizon {
            candidates.push((row_index, slack, sigma_row));
        }
    }
    if candidates.is_empty() {
        return Ok(None);
    }
    candidates.sort_by(|left, right| {
        left.1
            .partial_cmp(&right.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.0.cmp(&right.0))
    });

    // The retention floor is the accuracy the retained face must deliver, and
    // the first pass asks for exactly the accuracy this module reports its
    // moments to. When the assembled face misses that, the floor is raised by
    // the amount it missed by and the face rebuilt — the identity departure
    // scales like `ε / (pivot / diagonal)`, so the overshoot IS the factor the
    // floor is short by.
    //
    // This terminates by construction and without an iteration budget: the
    // demanded accuracy at least halves every pass, so once it falls below
    // `f64::EPSILON` the floor exceeds the diagonal itself and no row can be
    // retained. That is at most `log2(ORTHANT_MOMENT_RELATIVE_TOLERANCE /
    // f64::EPSILON)` passes — about forty — independent of how many constraint
    // rows the system carries.
    let mut demanded_accuracy = ORTHANT_MOMENT_RELATIVE_TOLERANCE;
    let mut first_pass = true;
    while demanded_accuracy >= f64::EPSILON {
        let Some(face) = assemble_retained_face(
            &candidates,
            demanded_accuracy,
            constraints,
            unconstrained_center,
        )?
        else {
            if first_pass {
                return Ok(None);
            }
            return Err(format!(
                "no constraint face survives the accuracy its own lift must deliver: raising \
                 the retention floor to {demanded_accuracy:.3e} relative left no retained row"
            ));
        };
        first_pass = false;
        // `G = Σ Aᵀ W⁻¹` solved through the factor built above, one column of
        // `Gᵀ` at a time: `W Gᵀ_col = (Σ Aᵀ)ᵀ_col`.
        let lift = cholesky_solve_right(&face.factor, &face.sigma_at)?;
        let departure = lift_identity_departure(&lift, constraints, &face.rows)?;
        if departure > ORTHANT_MOMENT_RELATIVE_TOLERANCE {
            if face.rows.len() == 1 {
                return Err(format!(
                    "a single retained constraint row still misses the identity that defines \
                     its lift: max|A G - I| = {departure:.6e} exceeds \
                     {ORTHANT_MOMENT_RELATIVE_TOLERANCE:.1e}, which one row cannot be \
                     ill-conditioned enough to cause"
                ));
            }
            // The departure scales like `ε / (pivot / diagonal)`, so the amount
            // the face missed by IS the factor its floor was short by.
            let overshoot = (departure / ORTHANT_MOMENT_RELATIVE_TOLERANCE).max(2.0);
            demanded_accuracy /= overshoot;
            continue;
        }

        let q = face.rows.len();
        let mut normal_center = Array1::<f64>::zeros(q);
        for (position, &row_index) in face.rows.iter().enumerate() {
            normal_center[position] =
                constraints.a.row(row_index).dot(unconstrained_center) - constraints.b[row_index];
        }

        let (normal_mean, normal_covariance) = box_truncated_moments(
            &normal_center,
            &face.upper,
            &face.w,
            face.factor.view(),
        )?;

        let mut removed = &face.w - &normal_covariance;
        symmetrize_in_place(&mut removed);
        certify_removed_variance(&removed, &face.w)?;

        return Ok(Some(ConstrainedPosteriorCorrection {
            lift,
            removed_normal_variance: removed,
            normal_mean_shift: normal_mean - normal_center,
            rows: face.rows,
            normal_upper_limits: face.upper,
        }));
    }
    Err(
        "the constraint-normal lift never reached the accuracy it is certified to, even with \
         the retention floor raised to double-precision resolution"
            .to_string(),
    )
}

/// One assembled constraint face: the retained rows in acceptance order, the
/// lower Cholesky factor of `W = A Σ Aᵀ` built while retaining them, `W` itself,
/// and the `Σ Aᵀ` block restricted to those rows.
struct RetainedFace {
    rows: Vec<usize>,
    factor: Array2<f64>,
    w: Array2<f64>,
    sigma_at: Array2<f64>,
    /// Upper limit per retained coordinate, `f64::INFINITY` for a half-line.
    upper: Vec<f64>,
}

/// Greedy pivoted-Cholesky rank filter on `W = A Σ Aᵀ`, walking the candidates
/// in their slack order.
///
/// The factor is built incrementally here and handed back, so the face is
/// factorized exactly once: a second factorization of the same `W` under a
/// different guard would let one matrix be judged by two standards, and near the
/// retention floor those two standards disagree.
fn assemble_retained_face(
    candidates: &[(usize, f64, Array1<f64>)],
    demanded_accuracy: f64,
    constraints: &LinearInequalityConstraints,
    unconstrained_center: &Array1<f64>,
) -> Result<Option<RetainedFace>, String> {
    let columns = constraints.a.ncols();
    // Each of `cross[k]`, `W_kk` and `diagonal` is one length-`p` inner product
    // of a constraint row against a `Σ` column, so each carries the standard
    // `γ_p ≈ p·ε` dot-product rounding; the anti-parallel test below compares
    // two products of such quantities, which propagates to about `4(p+1)ε`.
    // Nothing here is fitted to a fixture: it is the resolution at which "the
    // same direction, reversed" stops being decidable in double precision.
    let antiparallel_tolerance = 4.0 * (columns as f64 + 1.0) * f64::EPSILON;
    let mut rows: Vec<usize> = Vec::new();
    let mut sigma_a_columns: Vec<Array1<f64>> = Vec::new();
    let mut upper: Vec<f64> = Vec::new();
    let mut w_accepted = Array2::<f64>::zeros((0, 0));
    let mut factor = Array2::<f64>::zeros((0, 0));
    for (row_index, _, sigma_row) in candidates {
        let row = constraints.a.row(*row_index);
        let accepted = rows.len();
        let diagonal = row.dot(sigma_row);
        let mut cross = Array1::<f64>::zeros(accepted);
        for (position, column) in sigma_a_columns.iter().enumerate() {
            cross[position] = row.dot(column);
        }
        // Forward-substitute the new column through the accepted factor.
        let mut new_column = Array1::<f64>::zeros(accepted);
        for i in 0..accepted {
            let mut sum = cross[i];
            for k in 0..i {
                sum -= factor[[i, k]] * new_column[k];
            }
            new_column[i] = sum / factor[[i, i]];
        }
        let pivot = diagonal - new_column.dot(&new_column);
        // `pivot / diagonal` is the squared sine of the angle between this row's
        // constraint normal and the span of the rows accepted before it, in the
        // `Σ` metric. The lift `G = Σ Aᵀ W⁻¹` is solved through this same factor,
        // so its relative error grows like `ε · diagonal / pivot`. A floor at the
        // bare DETECTABILITY limit — `pivot ≈ ε · diagonal`, i.e. "reject only a
        // row that is dependent to the last bit" — therefore retains rows whose
        // lift carries no correct digits: measured 1.3e-2 relative error against
        // an exact rational reference at `pivot = 2 ε · diagonal`.
        //
        // So the floor is the one that keeps the retained face's own numerical
        // error under the accuracy demanded of it, and dropping instead costs
        // `O(θ)` with `θ` the angle between the two normals — below `5e-7`
        // radians at the first pass's floor — so a dropped row imposes no
        // constraint the retained one does not already impose.
        //
        // This is necessary and NOT sufficient, which is why the caller checks
        // the assembled face and raises `demanded_accuracy` when it falls short:
        // `min pivot / diagonal` is the smallest pivot of the correlation matrix
        // and bounds its smallest eigenvalue only when the elimination is ordered
        // by pivot magnitude. This walk is ordered by slack, so every row can
        // clear the floor while the face as a whole does not.
        let rank_floor = (accepted + 1) as f64 * f64::EPSILON * diagonal / demanded_accuracy;
        if !(pivot.is_finite() && pivot > rank_floor) {
            // Redundant AS A DIRECTION. That is not the same as redundant as a
            // CONSTRAINT, and the two come apart exactly at an anti-parallel
            // row: `l ≤ β_j ≤ u` arrives as `e_jᵀβ ≥ l` and `−e_jᵀβ ≥ −u`, and
            // the second adds no constraint-normal direction while halving the
            // support. Dropping it reports a one-sided posterior for a
            // two-sided bound (#2523).
            //
            // A row PARALLEL to an accepted one is genuinely implied, and the
            // slack ordering is what makes that true rather than hoped: rows are
            // walked by ascending standardized slack and `sd` scales with the
            // row, so the accepted row's `slack/sd` is the smaller, which is
            // exactly the statement that its wall is the binding one. Those
            // still drop, unchanged.
            record_opposed_face_limit(
                *row_index,
                &cross,
                diagonal,
                &w_accepted,
                &rows,
                constraints,
                unconstrained_center,
                antiparallel_tolerance,
                &mut upper,
            )?;
            continue;
        }
        let mut grown = Array2::<f64>::zeros((accepted + 1, accepted + 1));
        grown
            .slice_mut(ndarray::s![..accepted, ..accepted])
            .assign(&factor);
        for i in 0..accepted {
            grown[[accepted, i]] = new_column[i];
        }
        grown[[accepted, accepted]] = pivot.sqrt();
        factor = grown;

        let mut grown_w = Array2::<f64>::zeros((accepted + 1, accepted + 1));
        grown_w
            .slice_mut(ndarray::s![..accepted, ..accepted])
            .assign(&w_accepted);
        for i in 0..accepted {
            grown_w[[accepted, i]] = cross[i];
            grown_w[[i, accepted]] = cross[i];
        }
        grown_w[[accepted, accepted]] = diagonal;
        w_accepted = grown_w;

        rows.push(*row_index);
        sigma_a_columns.push(sigma_row.clone());
        upper.push(f64::INFINITY);
    }
    if rows.is_empty() {
        return Ok(None);
    }

    let q = rows.len();
    let p = sigma_a_columns[0].len();
    let mut sigma_at = Array2::<f64>::zeros((p, q));
    for (position, column) in sigma_a_columns.iter().enumerate() {
        sigma_at.column_mut(position).assign(column);
    }
    Ok(Some(RetainedFace {
        rows,
        factor,
        w: w_accepted,
        sigma_at,
        upper,
    }))
}

/// Keep the far wall of a two-sided bound that the rank filter has just refused
/// as a direction.
///
/// The refused row `a_r` carries no constraint-normal direction beyond the
/// accepted ones. When it is the OPPOSITE face of one of them — `a_r = −γ a_k`
/// for some `γ > 0`, up to a remainder with no posterior variance — it still
/// bounds that coordinate, from above:
///
/// ```text
/// a_rᵀβ ≥ b_r   ⟺   a_kᵀβ ≤ (ν − b_r)/γ   ⟺   u_k ≤ δ/γ,
/// ```
///
/// with `u_k = a_kᵀβ − b_k`, `ν = (a_r + γ a_k)ᵀβ` (almost surely constant,
/// precisely because its posterior variance is the pivot that just failed) and
/// `δ = (a_rᵀβ_unc − b_r) + γ(a_kᵀβ_unc − b_k)`.
///
/// The test is run in the `Σ` metric on quantities the filter has already
/// formed: `a_r = −γ a_k` makes the correlation `cross_k/√(W_kk·diagonal)`
/// exactly `−1`, and `γ = −cross_k/W_kk`. Running it in that metric rather than
/// on the raw rows is not a convenience — two rows that differ by a direction
/// with no posterior spread impose the same constraint on this posterior, and
/// the `Σ` metric is what sees that.
///
/// Rows that are refused for any other reason are left dropped, which is what
/// they were before. That is a narrower repair than "every dependent row", and
/// deliberately so: a row depending on two or more accepted normals at once cuts
/// the face along a diagonal, which no per-coordinate limit can represent.
#[allow(clippy::too_many_arguments)]
fn record_opposed_face_limit(
    row_index: usize,
    cross: &Array1<f64>,
    diagonal: f64,
    w_accepted: &Array2<f64>,
    rows: &[usize],
    constraints: &LinearInequalityConstraints,
    unconstrained_center: &Array1<f64>,
    antiparallel_tolerance: f64,
    upper: &mut [f64],
) -> Result<(), String> {
    let mut opposed: Option<(usize, f64, f64)> = None;
    for position in 0..rows.len() {
        let w_kk = w_accepted[[position, position]];
        let scale = (w_kk * diagonal).sqrt();
        if !(scale.is_finite() && scale > 0.0) {
            continue;
        }
        let correlation = cross[position] / scale;
        if correlation + 1.0 > antiparallel_tolerance {
            continue;
        }
        let gamma = -cross[position] / w_kk;
        if !(gamma.is_finite() && gamma > 0.0) {
            continue;
        }
        // Two accepted rows cannot both be anti-parallel to this one without
        // being parallel to each other, which the filter already refused; take
        // the most opposed and let the identity gate catch a face that is not.
        if opposed.is_none_or(|(_, best, _)| correlation < best) {
            opposed = Some((position, correlation, gamma));
        }
    }
    let Some((position, _, gamma)) = opposed else {
        return Ok(());
    };
    let accepted_row = rows[position];
    let delta = (constraints.a.row(row_index).dot(unconstrained_center)
        - constraints.b[row_index])
        + gamma
            * (constraints.a.row(accepted_row).dot(unconstrained_center)
                - constraints.b[accepted_row]);
    let limit = delta / gamma;
    if !(limit.is_finite() && limit > 0.0) {
        return Err(format!(
            "constraint rows {accepted_row} and {row_index} bound the same coefficient \
             direction from opposite sides with no width between them (upper limit \
             {limit:.6e} above the lower wall): the retained region is empty or a single \
             point, which is an equality constraint and not a posterior this module can \
             report moments for"
        ));
    }
    if limit < upper[position] {
        upper[position] = limit;
    }
    Ok(())
}

/// `max |A G - I|` over the retained rows.
///
/// `G = Σ Aᵀ W⁻¹` satisfies `A G = I` on those rows EXACTLY, because
/// `A (Σ Aᵀ) = W` by construction of `W`. The departure from that identity is
/// therefore not a modelling approximation: it is precisely the accuracy the
/// retained face's conditioning destroyed, measured on the object that is
/// actually used rather than on a proxy for it. Across a sweep of near-parallel
/// constraint normals it tracked the true error in `G` — against an exact
/// rational reference — to three significant digits at every angle.
fn lift_identity_departure(
    lift: &Array2<f64>,
    constraints: &LinearInequalityConstraints,
    rows: &[usize],
) -> Result<f64, String> {
    let q = rows.len();
    let mut departure = 0.0_f64;
    for (i, &row_index) in rows.iter().enumerate() {
        let row = constraints.a.row(row_index);
        for j in 0..q {
            let entry = row.dot(&lift.column(j));
            let target = if i == j { 1.0 } else { 0.0 };
            let deviation = (entry - target).abs();
            if !deviation.is_finite() {
                return Err(format!(
                    "the constraint-normal lift is not finite at retained row {row_index}, \
                     constraint-normal coordinate {j}"
                ));
            }
            departure = departure.max(deviation);
        }
    }
    Ok(departure)
}

/// Solve `X W = B` for `X` given the lower Cholesky factor `L` of the symmetric
/// `W = L Lᵀ`, i.e. return `B W⁻¹`. `W` is symmetric so `X = (W⁻¹ Bᵀ)ᵀ`.
fn cholesky_solve_right(factor: &Array2<f64>, b: &Array2<f64>) -> Result<Array2<f64>, String> {
    let q = factor.nrows();
    if b.ncols() != q {
        return Err(format!(
            "constraint-normal solve: factor is {q}x{q} but the right-hand side has {} columns",
            b.ncols()
        ));
    }
    let rows = b.nrows();
    let mut out = Array2::<f64>::zeros((rows, q));
    let mut work = Array1::<f64>::zeros(q);
    for r in 0..rows {
        for i in 0..q {
            let mut sum = b[[r, i]];
            for k in 0..i {
                sum -= factor[[i, k]] * work[k];
            }
            work[i] = sum / factor[[i, i]];
        }
        for i in (0..q).rev() {
            let mut sum = work[i];
            for k in (i + 1)..q {
                sum -= factor[[k, i]] * out[[r, k]];
            }
            out[[r, i]] = sum / factor[[i, i]];
        }
    }
    Ok(out)
}

/// Refuse a correction that is not a genuine variance REMOVAL. `Δ = W − Cov[u]`
/// must be positive semidefinite (truncation cannot inflate a Gaussian's
/// covariance) and must not exceed `W` (it cannot remove more variance than
/// there was). Either failure means the cubature returned something that is not
/// the moment of a distribution, which is a numerical failure and not a number
/// to report.
fn certify_removed_variance(removed: &Array2<f64>, w: &Array2<f64>) -> Result<(), String> {
    let q = removed.nrows();
    // Scale-free bound: both `Δ` and `W − Δ = Cov[u]` are checked against the
    // cubature's own accuracy in the pre-truncation metric.
    let slack = ORTHANT_MOMENT_RELATIVE_TOLERANCE * (q as f64);
    for i in 0..q {
        let scale = w[[i, i]];
        if removed[[i, i]] < -slack * scale {
            return Err(format!(
                "truncated orthant moments inflated the constraint-normal variance at index {i} \
                 (removed {:.6e} against scale {scale:.6e}); truncation cannot increase a \
                 Gaussian covariance",
                removed[[i, i]]
            ));
        }
        if removed[[i, i]] > (1.0 + slack) * scale {
            return Err(format!(
                "truncated orthant moments removed more variance than exists at index {i} \
                 (removed {:.6e} against scale {scale:.6e})",
                removed[[i, i]]
            ));
        }
        for j in 0..q {
            if !removed[[i, j]].is_finite() {
                return Err(format!(
                    "truncated orthant moments produced a non-finite entry at ({i},{j})"
                ));
            }
        }
    }
    Ok(())
}

fn symmetrize_in_place(matrix: &mut Array2<f64>) {
    let n = matrix.nrows();
    for i in 0..n {
        for j in (i + 1)..n {
            let averaged = 0.5 * (matrix[[i, j]] + matrix[[j, i]]);
            matrix[[i, j]] = averaged;
            matrix[[j, i]] = averaged;
        }
    }
}

/// First two moments of `u ~ N(mean, covariance)` restricted to the box
/// `0 ≤ u ≤ upper`, where `upper_i = f64::INFINITY` makes coordinate `i` the
/// half-line the orthant is built from.
///
/// One dimension has the closed form and is evaluated exactly. Higher
/// dimensions use the Genz separation-of-variables transformation, under which
/// EVERY moment is an integral of the same integrand over the unit cube — so a
/// single cubature delivers the normalizing probability, the mean and the second
/// moment together, instead of the `O(q²)` separate orthant probabilities the
/// Tallis face/edge recursion would need. The transformation is already a
/// product of intervals; a finite upper limit changes only where each interval
/// ends, which is why a box costs the same cubature as an orthant.
fn box_truncated_moments(
    mean: &Array1<f64>,
    upper: &[f64],
    covariance: &Array2<f64>,
    factor: ArrayView2<'_, f64>,
) -> Result<(Array1<f64>, Array2<f64>), String> {
    let q = mean.len();
    if covariance.nrows() != q || covariance.ncols() != q {
        return Err(format!(
            "truncated moments: mean has length {q} but the covariance is {}x{}",
            covariance.nrows(),
            covariance.ncols()
        ));
    }
    if upper.len() != q {
        return Err(format!(
            "truncated moments: mean has length {q} but {} upper limits were supplied",
            upper.len()
        ));
    }
    if upper.iter().any(|limit| !(*limit > 0.0)) {
        return Err(format!(
            "truncated moments: every upper limit must sit strictly above its wall, got {upper:?}"
        ));
    }
    if q == 1 {
        return scalar_truncated_moments(mean[0], covariance[[0, 0]], upper[0]);
    }
    if factor.nrows() != q || factor.ncols() != q {
        return Err(format!(
            "orthant moments: the constraint-normal covariance is {q}x{q} but the Cholesky \
             factor supplied with it is {}x{}",
            factor.nrows(),
            factor.ncols()
        ));
    }

    let generator = kronecker_generator(q);
    let mut accumulator = OrthantAccumulator::new(q);
    let mut evaluated = 0usize;
    let mut previous: Option<(Array1<f64>, Array2<f64>)> = None;
    loop {
        let target = if evaluated == 0 {
            ORTHANT_MOMENT_INITIAL_POINTS
        } else {
            evaluated * 2
        };
        accumulate_orthant_nodes(
            &mut accumulator,
            mean,
            upper,
            factor,
            &generator,
            evaluated,
            target,
        )?;
        evaluated = target;
        let current = accumulator.moments()?;
        if let Some(ref last) = previous
            && moment_relative_change(last, &current, covariance)
                <= ORTHANT_MOMENT_RELATIVE_TOLERANCE
        {
            return Ok(current);
        }
        if evaluated >= ORTHANT_MOMENT_MAXIMUM_POINTS {
            let change = previous
                .as_ref()
                .map(|last| moment_relative_change(last, &current, covariance))
                .unwrap_or(f64::INFINITY);
            return Err(format!(
                "truncated moments for a {q}-dimensional constraint face did not converge: \
                 relative moment change {change:.3e} still exceeds \
                 {ORTHANT_MOMENT_RELATIVE_TOLERANCE:.1e} at {evaluated} cubature nodes"
            ));
        }
        previous = Some(current);
    }
}

/// Running log-scaled first and second moment accumulator for the cubature.
///
/// Node weights span hundreds of decades between a barely-truncated face and a
/// deeply pinned one, so the accumulators carry an explicit log scale and are
/// rescaled whenever a heavier node arrives. Accumulating the weights directly
/// would underflow the whole face to zero and leave the normalized moments as
/// `0/0`.
struct OrthantAccumulator {
    log_scale: f64,
    weight_sum: f64,
    weighted_mean: Array1<f64>,
    weighted_second: Array2<f64>,
}

trait OrthantNodeSink {
    fn push(&mut self, log_weight: f64, point: &Array1<f64>);
}

impl OrthantAccumulator {
    fn new(q: usize) -> Self {
        Self {
            log_scale: f64::NEG_INFINITY,
            weight_sum: 0.0,
            weighted_mean: Array1::zeros(q),
            weighted_second: Array2::zeros((q, q)),
        }
    }

    fn push(&mut self, log_weight: f64, point: &Array1<f64>) {
        let q = point.len();
        if log_weight > self.log_scale {
            let rescale = (self.log_scale - log_weight).exp();
            self.weight_sum *= rescale;
            self.weighted_mean *= rescale;
            self.weighted_second *= rescale;
            self.log_scale = log_weight;
        }
        let weight = (log_weight - self.log_scale).exp();
        self.weight_sum += weight;
        for i in 0..q {
            self.weighted_mean[i] += weight * point[i];
            for j in 0..=i {
                self.weighted_second[[i, j]] += weight * point[i] * point[j];
            }
        }
    }

    fn moments(&self) -> Result<(Array1<f64>, Array2<f64>), String> {
        if !(self.weight_sum.is_finite() && self.weight_sum > 0.0) {
            return Err(format!(
                "orthant cubature accumulated no feasible mass (weight sum {:?}); the \
                 constraint face has no representable interior",
                self.weight_sum
            ));
        }
        let q = self.weighted_mean.len();
        let mean = &self.weighted_mean / self.weight_sum;
        let mut covariance = Array2::<f64>::zeros((q, q));
        for i in 0..q {
            for j in 0..=i {
                let centered = self.weighted_second[[i, j]] / self.weight_sum - mean[i] * mean[j];
                covariance[[i, j]] = centered;
                covariance[[j, i]] = centered;
            }
        }
        Ok((mean, covariance))
    }
}

impl OrthantNodeSink for OrthantAccumulator {
    fn push(&mut self, log_weight: f64, point: &Array1<f64>) {
        OrthantAccumulator::push(self, log_weight, point);
    }
}

/// Evaluate Genz nodes `first..last` of the Kronecker sequence and fold them
/// into `accumulator`.
fn accumulate_orthant_nodes<S: OrthantNodeSink>(
    accumulator: &mut S,
    mean: &Array1<f64>,
    upper: &[f64],
    factor: ArrayView2<'_, f64>,
    generator: &[f64],
    first: usize,
    last: usize,
) -> Result<(), String> {
    let q = mean.len();
    let mut z = Array1::<f64>::zeros(q);
    let mut point = Array1::<f64>::zeros(q);
    for node in first..last {
        let offset = node as f64 + 0.5;
        let mut log_weight = 0.0f64;
        for i in 0..q {
            let mut bound = -mean[i];
            for j in 0..i {
                bound -= factor[[i, j]] * z[j];
            }
            let wall = bound / factor[[i, i]];
            // Tent-periodized Kronecker lattice. The raw sequence leaves the
            // integrand non-periodic across the cube face, which costs the
            // lattice rule most of its rate; folding `x ↦ 1 − |2x − 1|`
            // preserves the uniform measure and periodizes it.
            let lattice = {
                let raw = offset * generator[i];
                let fractional = raw - raw.floor();
                1.0 - (2.0 * fractional - 1.0).abs()
            };
            if !upper[i].is_finite() {
                let log_tail = normal_logsf(wall);
                if !log_tail.is_finite() {
                    // The remaining feasible mass along this coordinate
                    // underflowed to zero: the node contributes nothing and
                    // cannot be renormalized, so drop it rather than propagate a
                    // NaN.
                    log_weight = f64::NEG_INFINITY;
                    break;
                }
                log_weight += log_tail;
                // `Φ̄(z_i) = (1 − x_i)·Φ̄(lower)` inverted on the upper tail, so
                // a deeply pinned coordinate never forms `1 − Φ(·)` in
                // probability space. Both factors can round to one (an inactive
                // coordinate at the very edge of the lattice cell), which would
                // ask for `Φ̄⁻¹(1)`; the smallest representable log-probability
                // answers that with the far-left endpoint, which is what the
                // region actually is there.
                let log_fraction = (1.0 - lattice).max(f64::MIN_POSITIVE).ln();
                let log_upper_tail = log_fraction + log_tail;
                let resolved = if log_upper_tail < 0.0 {
                    log_upper_tail
                } else {
                    -f64::MIN_POSITIVE
                };
                z[i] = -standard_normal_quantile_from_log_cdf(resolved)
                    .map_err(|error| format!("orthant cubature coordinate {i}: {error}"))?;
                continue;
            }

            // Bounded coordinate. The conditional interval is `[wall, ceiling]`
            // and `ceiling − wall = upper_i / L_ii` exactly, so the width never
            // goes through a subtraction of two conditional means.
            let ceiling = wall + upper[i] / factor[[i, i]];
            // Reflect an interval that sits in the LOWER tail. Both `Φ̄` values
            // are then within rounding of one and their difference — the
            // interval's entire mass — would be computed as a cancellation
            // between them. Under `z ↦ −z` the same interval is `[−ceiling,
            // −wall]` with both endpoints in the upper tail, where `Φ̄` is
            // evaluated directly. This is the regime a two-sided bound reaches
            // whenever the unconstrained fit lands beyond the far wall, which is
            // exactly when such a bound is worth declaring.
            let reflect = wall + ceiling < 0.0;
            let (low, high) = if reflect {
                (-ceiling, -wall)
            } else {
                (wall, ceiling)
            };
            let log_tail_low = normal_logsf(low);
            let log_tail_high = normal_logsf(high);
            if !log_tail_low.is_finite() {
                log_weight = f64::NEG_INFINITY;
                break;
            }
            // `removed ≤ 0` is the log of the fraction of the half-line's mass
            // that the far wall takes away.
            let removed = log_tail_high - log_tail_low;
            let log_mass = log_tail_low + log1mexp(removed);
            if !log_mass.is_finite() {
                // The slab is narrower than double precision can resolve at this
                // conditional position; it carries no representable mass.
                log_weight = f64::NEG_INFINITY;
                break;
            }
            log_weight += log_mass;
            // `Φ̄(z) = Φ̄(low)·(1 − x(1 − e^removed))`: the same upper-tail
            // inversion as the half-line, with the retained fraction shortened
            // to the slab.
            let retained = (-lattice * (-removed.exp_m1())).ln_1p();
            let log_upper_tail = log_tail_low + retained;
            let resolved = if log_upper_tail < 0.0 {
                log_upper_tail
            } else {
                -f64::MIN_POSITIVE
            };
            let sampled = -standard_normal_quantile_from_log_cdf(resolved)
                .map_err(|error| format!("truncated cubature coordinate {i}: {error}"))?;
            // The inversion is exact in probability space, so an excursion past
            // either endpoint is rounding in `Φ̄⁻¹` alone; the node belongs to
            // the interval by construction and is placed there.
            let clamped = sampled.clamp(low, high);
            z[i] = if reflect { -clamped } else { clamped };
        }
        if !log_weight.is_finite() {
            continue;
        }
        for i in 0..q {
            let mut value = mean[i];
            for j in 0..=i {
                value += factor[[i, j]] * z[j];
            }
            point[i] = value;
        }
        accumulator.push(log_weight, &point);
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct WeightedProjectionNode {
    conditional_mean: f64,
    weight: f64,
}

struct ProjectionNodeAccumulator<'a> {
    moments: OrthantAccumulator,
    normal_center: &'a Array1<f64>,
    projection_lift: &'a Array1<f64>,
    ambient_mean: f64,
    nodes: Vec<(f64, f64)>,
}

impl<'a> ProjectionNodeAccumulator<'a> {
    fn new(
        normal_center: &'a Array1<f64>,
        projection_lift: &'a Array1<f64>,
        ambient_mean: f64,
    ) -> Self {
        Self {
            moments: OrthantAccumulator::new(normal_center.len()),
            normal_center,
            projection_lift,
            ambient_mean,
            nodes: Vec::new(),
        }
    }

    fn normalized_nodes(self) -> Result<Vec<WeightedProjectionNode>, String> {
        let max_log_weight = self
            .nodes
            .iter()
            .map(|(_, log_weight)| *log_weight)
            .fold(f64::NEG_INFINITY, f64::max);
        if !max_log_weight.is_finite() {
            return Err(
                "orthant projection cubature accumulated no finite node weight".to_string(),
            );
        }
        let weight_sum = self
            .nodes
            .iter()
            .map(|(_, log_weight)| (*log_weight - max_log_weight).exp())
            .sum::<f64>();
        if !(weight_sum.is_finite() && weight_sum > 0.0) {
            return Err(format!(
                "orthant projection cubature has invalid normalized weight sum {weight_sum:?}"
            ));
        }
        Ok(self
            .nodes
            .into_iter()
            .map(|(conditional_mean, log_weight)| WeightedProjectionNode {
                conditional_mean,
                weight: (log_weight - max_log_weight).exp() / weight_sum,
            })
            .collect())
    }
}

impl OrthantNodeSink for ProjectionNodeAccumulator<'_> {
    fn push(&mut self, log_weight: f64, point: &Array1<f64>) {
        self.moments.push(log_weight, point);
        let conditional_mean = self.ambient_mean
            + self
                .projection_lift
                .iter()
                .zip(point.iter().zip(self.normal_center.iter()))
                .map(|(&lift, (&value, &center))| lift * (value - center))
                .sum::<f64>();
        self.nodes.push((conditional_mean, log_weight));
    }
}

fn converged_projection_nodes(
    mean: &Array1<f64>,
    covariance: &Array2<f64>,
    upper: &[f64],
    projection_lift: &Array1<f64>,
    ambient_mean: f64,
) -> Result<Vec<WeightedProjectionNode>, String> {
    let q = mean.len();
    if covariance.dim() != (q, q) || projection_lift.len() != q || upper.len() != q {
        return Err(format!(
            "truncated projection geometry mismatch: mean={q}, covariance={:?}, lift={}, \
             upper limits={}",
            covariance.dim(),
            projection_lift.len(),
            upper.len()
        ));
    }
    let factor = gam_linalg::triangular::cholesky_factor_in_place(
        covariance.view(),
        gam_linalg::triangular::CholeskyGuard::FiniteStrict,
    )
    .ok_or_else(|| {
        "orthant projection: the constraint-normal covariance is not numerically positive definite"
            .to_string()
    })?;
    let generator = kronecker_generator(q);
    let mut accumulator = ProjectionNodeAccumulator::new(mean, projection_lift, ambient_mean);
    let mut evaluated = 0usize;
    let mut previous: Option<(Array1<f64>, Array2<f64>)> = None;
    loop {
        let target = if evaluated == 0 {
            ORTHANT_MOMENT_INITIAL_POINTS
        } else {
            evaluated * 2
        };
        accumulate_orthant_nodes(
            &mut accumulator,
            mean,
            upper,
            factor.view(),
            &generator,
            evaluated,
            target,
        )?;
        evaluated = target;
        let current = accumulator.moments.moments()?;
        if let Some(ref last) = previous
            && moment_relative_change(last, &current, covariance)
                <= ORTHANT_MOMENT_RELATIVE_TOLERANCE
        {
            return accumulator.normalized_nodes();
        }
        if evaluated >= ORTHANT_MOMENT_MAXIMUM_POINTS {
            let change = previous
                .as_ref()
                .map(|last| moment_relative_change(last, &current, covariance))
                .unwrap_or(f64::INFINITY);
            return Err(format!(
                "orthant projection for a {q}-dimensional constraint face did not converge: \
                 relative moment change {change:.3e} still exceeds \
                 {ORTHANT_MOMENT_RELATIVE_TOLERANCE:.1e} at {evaluated} cubature nodes"
            ));
        }
        previous = Some(current);
    }
}

fn projection_quantile(
    nodes: &[WeightedProjectionNode],
    residual_variance: f64,
    probability: f64,
    posterior_mean: f64,
    ambient_sd: f64,
) -> Result<f64, String> {
    if nodes.is_empty() {
        return Err("orthant projection quantile received no cubature nodes".to_string());
    }
    if residual_variance == 0.0 {
        let mut ordered = nodes.to_vec();
        ordered.sort_by(|left, right| left.conditional_mean.total_cmp(&right.conditional_mean));
        let mut cumulative = 0.0;
        for node in &ordered {
            cumulative += node.weight;
            if cumulative >= probability {
                return Ok(node.conditional_mean);
            }
        }
        return Ok(ordered
            .last()
            .expect("non-empty projection node set")
            .conditional_mean);
    }

    let residual_sd = residual_variance.sqrt();
    let cdf = |value: f64| {
        nodes
            .iter()
            .map(|node| {
                node.weight * normal_cdf((value - node.conditional_mean) / residual_sd)
            })
            .sum::<f64>()
    };
    let mut step = ambient_sd.max(residual_sd).max(f64::MIN_POSITIVE);
    let mut lower = posterior_mean - step;
    let mut upper = posterior_mean + step;
    while cdf(lower) > probability {
        step *= 2.0;
        lower = posterior_mean - step;
        if !lower.is_finite() {
            return Err(format!(
                "orthant projection quantile could not bracket lower probability {probability}"
            ));
        }
    }
    step = ambient_sd.max(residual_sd).max(f64::MIN_POSITIVE);
    while cdf(upper) < probability {
        step *= 2.0;
        upper = posterior_mean + step;
        if !upper.is_finite() {
            return Err(format!(
                "orthant projection quantile could not bracket upper probability {probability}"
            ));
        }
    }

    let resolution = f64::EPSILON.sqrt() * ambient_sd.max(residual_sd);
    loop {
        let midpoint = lower + 0.5 * (upper - lower);
        if midpoint == lower || midpoint == upper || upper - lower <= resolution {
            return Ok(midpoint);
        }
        if cdf(midpoint) < probability {
            lower = midpoint;
        } else {
            upper = midpoint;
        }
    }
}

/// `log(1 − exp(d))` for `d ≤ 0`, evaluated on whichever of the two branches
/// keeps the cancellation out of the result.
///
/// `d` is always the log of the fraction of a half-line's mass that an upper
/// limit removes, so `d = −∞` is the unbounded coordinate and returns exactly
/// `0`. That exactness is what makes an infinite upper limit reproduce the
/// half-line arithmetic bit for bit rather than merely closely.
fn log1mexp(d: f64) -> f64 {
    if d == f64::NEG_INFINITY {
        return 0.0;
    }
    if d >= 0.0 {
        return f64::NEG_INFINITY;
    }
    if d > -std::f64::consts::LN_2 {
        (-d.exp_m1()).ln()
    } else {
        (-d.exp()).ln_1p()
    }
}

/// Closed-form moments of `N(mean, variance)` restricted to `[0, upper]`.
///
/// `upper = f64::INFINITY` is the half-line, and takes the inverse-Mills branch
/// unchanged — a one-sided bound is not routed through the two-sided formula and
/// then hoped to agree with itself.
fn scalar_truncated_moments(
    mean: f64,
    variance: f64,
    upper: f64,
) -> Result<(Array1<f64>, Array2<f64>), String> {
    if !(variance.is_finite() && variance > 0.0) {
        return Err(format!(
            "scalar truncated moments need a positive finite variance, got {variance:?}"
        ));
    }
    let sd = variance.sqrt();
    // `alpha` is the truncation point in standardized units; the feasible half
    // is `z ≥ alpha`, and `mills` is the inverse Mills ratio `φ(α)/Φ̄(α)`
    // obtained on the numerically stable `Φ` branch by reflection.
    let alpha = -mean / sd;
    if !upper.is_finite() {
        let mills = signed_probit_logcdf_and_mills_ratio(-alpha).1;
        if !(mills.is_finite() && mills >= 0.0) {
            return Err(format!(
                "scalar truncated moments: inverse Mills ratio at {alpha} is {mills:?}"
            ));
        }
        let truncated_mean = mean + sd * mills;
        let truncated_variance = variance * (1.0 + alpha * mills - mills * mills);
        if !(truncated_variance.is_finite() && truncated_variance >= 0.0) {
            return Err(format!(
                "scalar truncated moments produced variance {truncated_variance:?} at \
                 standardized truncation point {alpha}"
            ));
        }
        return Ok((
            Array1::from_elem(1, truncated_mean),
            Array2::from_elem((1, 1), truncated_variance),
        ));
    }
    if !(upper > 0.0) {
        return Err(format!(
            "scalar truncated moments need the upper limit above the wall, got {upper:?}"
        ));
    }
    let beta = (upper - mean) / sd;
    // Reflect so the retained interval sits in the upper tail: every difference
    // below is then between two directly-evaluated tail quantities instead of
    // between two numbers within rounding of one.
    let reflect = alpha + beta < 0.0;
    let (low, high, centre) = if reflect {
        (-beta, -alpha, -mean)
    } else {
        (alpha, beta, mean)
    };
    let log_tail_low = normal_logsf(low);
    let log_tail_high = normal_logsf(high);
    let log_mass = log_tail_low + log1mexp(log_tail_high - log_tail_low);
    if !log_mass.is_finite() {
        return Err(format!(
            "scalar truncated moments: the interval [0, {upper:.6e}] around mean {mean:.6e} \
             with standard deviation {sd:.6e} carries no representable mass"
        ));
    }
    // `φ(high)/φ(low) = exp(½(low² − high²))`, factored as a difference of
    // squares so the exponent is not the cancellation of two large numbers. The
    // reflection above makes it non-positive.
    let log_density_ratio = 0.5 * (low - high) * (low + high);
    let density_ratio = log_density_ratio.exp();
    let scale = (-0.5 * low * low - 0.5 * (2.0 * std::f64::consts::PI).ln() - log_mass).exp();
    let first = scale * -log_density_ratio.exp_m1();
    let second = scale * (low - high * density_ratio);
    let truncated_mean = centre + sd * first;
    let truncated_variance = variance * (1.0 + second - first * first);
    if !(truncated_variance.is_finite() && truncated_variance >= 0.0) {
        return Err(format!(
            "scalar truncated moments produced variance {truncated_variance:?} on the \
             standardized interval [{low}, {high}]"
        ));
    }
    Ok((
        Array1::from_elem(1, if reflect { -truncated_mean } else { truncated_mean }),
        Array2::from_elem((1, 1), truncated_variance),
    ))
}

/// Largest relative moment change between two cubature passes, measured in the
/// pre-truncation scale `sd_i = sqrt(W_ii)` so the criterion does not depend on
/// how the constraint rows happen to be scaled.
fn moment_relative_change(
    previous: &(Array1<f64>, Array2<f64>),
    current: &(Array1<f64>, Array2<f64>),
    w: &Array2<f64>,
) -> f64 {
    let q = current.0.len();
    let mut worst = 0.0f64;
    for i in 0..q {
        let sd_i = w[[i, i]].sqrt();
        worst = worst.max((current.0[i] - previous.0[i]).abs() / sd_i);
        for j in 0..q {
            let sd_j = w[[j, j]].sqrt();
            worst =
                worst.max((current.1[[i, j]] - previous.1[[i, j]]).abs() / (sd_i * sd_j));
        }
    }
    worst
}

/// Kronecker (Richtmyer) lattice generator `α_i = frac(√p_i)` over the primes.
/// Deterministic and table-free: the sequence is reproduced from the primes
/// themselves, so the reported covariance does not depend on a stored vector of
/// magic direction numbers or on any random seed.
fn kronecker_generator(dimension: usize) -> Vec<f64> {
    let mut generator = Vec::with_capacity(dimension);
    let mut candidate = 2u64;
    while generator.len() < dimension {
        if is_prime(candidate) {
            let root = (candidate as f64).sqrt();
            generator.push(root - root.floor());
        }
        candidate += 1;
    }
    generator
}

fn is_prime(value: u64) -> bool {
    if value < 2 {
        return false;
    }
    let mut divisor = 2u64;
    while divisor * divisor <= value {
        if value % divisor == 0 {
            return false;
        }
        divisor += 1;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;


    /// Independent reference: Simpson quadrature of `N(mean, variance)`
    /// restricted to `[0, ∞)`, with the density rescaled by its value at the
    /// truncation point so a deeply pinned centre does not underflow.
    fn quadrature_truncated_moments(mean: f64, variance: f64) -> (f64, f64) {
        let sd = variance.sqrt();
        let alpha = -mean / sd;
        let panels = 400_000usize;
        let upper = alpha + 60.0;
        let step = (upper - alpha) / panels as f64;
        let mut mass = 0.0f64;
        let mut first = 0.0f64;
        let mut second = 0.0f64;
        for index in 0..=panels {
            let z = alpha + step * index as f64;
            let simpson = if index == 0 || index == panels {
                1.0
            } else if index % 2 == 1 {
                4.0
            } else {
                2.0
            };
            let density = (-(z * z - alpha * alpha) / 2.0).exp();
            mass += simpson * density;
            first += simpson * density * z;
            second += simpson * density * z * z;
        }
        let m1 = first / mass;
        let m2 = second / mass;
        (mean + sd * m1, variance * (m2 - m1 * m1))
    }

    /// The scalar closed form against the textbook truncated-normal moments at
    /// the three regimes the estimand argument turns on.
    #[test]
    fn scalar_truncated_moments_match_the_closed_form_at_every_regime() {
        // Mode exactly on the bound: half-normal, variance (1 - 2/pi) sigma^2.
        let (mean, variance) = scalar_truncated_moments(0.0, 1.0, f64::INFINITY).expect("half normal");
        let expected_mean = (2.0 / std::f64::consts::PI).sqrt();
        assert!(
            (mean[0] - expected_mean).abs() < 1e-12,
            "half-normal mean {} vs {expected_mean}",
            mean[0]
        );
        let expected_variance = 1.0 - 2.0 / std::f64::consts::PI;
        assert!(
            (variance[[0, 0]] - expected_variance).abs() < 1e-12,
            "half-normal variance {} vs {expected_variance}",
            variance[[0, 0]]
        );
        assert!(
            variance[[0, 0]] > 0.36 && variance[[0, 0]] < 0.37,
            "a coefficient whose mode sits exactly on its bound keeps a THIRD of its \
             unconstrained variance, not zero: got {}",
            variance[[0, 0]]
        );

        // Strongly pinned: checked against an INDEPENDENT quadrature of the
        // truncated density rather than against an asymptotic, because the
        // leading `sigma^2/alpha^2` term carries an O(alpha^-4) deficit that a
        // tolerance would have to absorb.
        for center in [-2.0, -4.0, -8.0] {
            let (deep_mean, deep) = scalar_truncated_moments(center, 1.0, f64::INFINITY).expect("deep tail");
            let (reference_mean, reference_variance) = quadrature_truncated_moments(center, 1.0);
            assert!(
                (deep_mean[0] - reference_mean).abs() < 1e-9 * reference_mean.abs().max(1.0),
                "closed-form mean {} vs quadrature {reference_mean} at centre {center}",
                deep_mean[0]
            );
            assert!(
                (deep[[0, 0]] / reference_variance - 1.0).abs() < 1e-8,
                "closed-form variance {} vs quadrature {reference_variance} at centre {center}",
                deep[[0, 0]]
            );
            assert!(
                deep[[0, 0]] > 0.0,
                "a finite multiplier never gives zero variance, got {} at centre {center}",
                deep[[0, 0]]
            );
        }
        // ...and it does head to zero like sigma^2/alpha^2, which is the ONLY
        // limit in which the active-face answer becomes correct.
        let (_, at_eight) = scalar_truncated_moments(-8.0, 1.0, f64::INFINITY).expect("deep tail");
        assert!(
            at_eight[[0, 0]] * 64.0 > 0.9 && at_eight[[0, 0]] * 64.0 < 1.0,
            "variance times alpha^2 should approach one from below, got {}",
            at_eight[[0, 0]] * 64.0
        );

        // Constraint far away: the moments relax back to the untruncated ones,
        // but only to the order of the tail mass the constraint still removes —
        // a bound five standard deviations below the centre still moves the mean
        // by `sd·φ(5)/Φ(5) ≈ 3e-6`, which is exactly the smooth dependence on
        // slack that makes a tightness predicate unnecessary.
        let (far_mean, far_variance) = scalar_truncated_moments(10.0, 4.0, f64::INFINITY).expect("inactive");
        let (reference_mean, reference_variance) = quadrature_truncated_moments(10.0, 4.0);
        assert!(
            (far_mean[0] - reference_mean).abs() < 1e-9,
            "inactive-bound mean {} vs quadrature {reference_mean}",
            far_mean[0]
        );
        assert!(
            (far_variance[[0, 0]] - reference_variance).abs() < 1e-9,
            "inactive-bound variance {} vs quadrature {reference_variance}",
            far_variance[[0, 0]]
        );
        assert!(
            (far_mean[0] - 10.0).abs() < 1e-5 && far_mean[0] > 10.0,
            "a bound five sd away moves the mean by the tail mass and no more, got {}",
            far_mean[0]
        );
        assert!(
            (far_variance[[0, 0]] - 4.0).abs() < 1e-4 && far_variance[[0, 0]] < 4.0,
            "a bound five sd away shrinks the variance by the tail mass and no more, got {}",
            far_variance[[0, 0]]
        );
    }

    #[test]
    fn equal_tailed_projection_interval_is_asymmetric_for_a_half_normal() {
        let covariance = array![[1.0]];
        let center = array![0.0];
        let constraints =
            LinearInequalityConstraints::new(array![[1.0]], array![0.0]).expect("constraint");
        let correction =
            constrained_posterior_correction_from_covariance(&covariance, &center, &constraints)
                .expect("correction")
                .expect("active half-space");
        let geometry = ConstrainedPosteriorGeometry {
            constraints,
            mode: array![0.0],
            unconstrained_center: center,
            correction: Some(correction),
        };
        let (lower, upper) = constrained_projection_equal_tailed_interval(
            &covariance,
            &geometry,
            &array![1.0],
            0.95,
        )
        .expect("equal-tailed interval");

        // For Z | Z>=0, F(z)=2 Phi(z)-1. The equal-tailed endpoints are
        // Phi^-1((1+p)/2), p in {0.025, 0.975}.
        let expected_lower = standard_normal_quantile(0.5125).expect("lower quantile");
        let expected_upper = standard_normal_quantile(0.9875).expect("upper quantile");
        assert!(
            (lower - expected_lower).abs() < 2e-3,
            "half-normal lower endpoint {lower} vs {expected_lower}"
        );
        assert!(
            (upper - expected_upper).abs() < 2e-3,
            "half-normal upper endpoint {upper} vs {expected_upper}"
        );
        let posterior_mean = (2.0 / std::f64::consts::PI).sqrt();
        assert!(
            (posterior_mean - lower) < (upper - posterior_mean),
            "the exact skew interval must not collapse back to mean +/- z*sd"
        );
    }

    #[test]
    fn equal_tailed_projection_sweep_has_exact_mass_and_repairs_the_short_symmetric_band() {
        let covariance = array![[1.0]];
        let constraints =
            LinearInequalityConstraints::new(array![[1.0]], array![0.0]).expect("constraint");
        let alpha = 0.025;
        let ambient_width =
            2.0 * standard_normal_quantile(1.0 - alpha).expect("ambient quantile");
        let mut saw_repaired_short_symmetric_band = false;

        for center_value in [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0] {
            let center = array![center_value];
            let correction = constrained_posterior_correction_from_covariance(
                &covariance,
                &center,
                &constraints,
            )
            .expect("correction")
            .expect("finite lower truncation");
            let posterior_variance =
                1.0 - correction.removed_variance_diagonal()[0];
            let geometry = ConstrainedPosteriorGeometry {
                constraints: constraints.clone(),
                mode: array![center_value.max(0.0)],
                unconstrained_center: center,
                correction: Some(correction),
            };
            let (lower, upper) = constrained_projection_equal_tailed_interval(
                &covariance,
                &geometry,
                &array![1.0],
                0.95,
            )
            .expect("equal-tailed interval");

            let mass_below_bound = normal_cdf(-center_value);
            let retained_mass = 1.0 - mass_below_bound;
            let truncated_cdf = |value: f64| {
                (normal_cdf(value - center_value) - mass_below_bound) / retained_mass
            };
            assert!(
                (truncated_cdf(lower) - alpha).abs() < 2e-8
                    && (truncated_cdf(upper) - (1.0 - alpha)).abs() < 2e-8,
                "centre {center_value}: endpoints [{lower}, {upper}] do not enclose exact \
                 posterior mass 0.95"
            );
            assert!(
                lower >= 0.0,
                "centre {center_value}: lower endpoint {lower} escaped the saved cone"
            );
            assert!(
                upper - lower <= ambient_width + 1e-10,
                "centre {center_value}: truncation widened [{lower}, {upper}] beyond the \
                 ambient Gaussian interval"
            );

            if center_value == 3.0 {
                let symmetric_width = 2.0
                    * standard_normal_quantile(1.0 - alpha).expect("symmetric quantile")
                    * posterior_variance.sqrt();
                assert!(
                    upper - lower > symmetric_width,
                    "the exact 3-SE interval must repair the moment-matched symmetric interval's \
                     short, under-covering band: exact width {}, symmetric width {symmetric_width}",
                    upper - lower
                );
                saw_repaired_short_symmetric_band = true;
            }
        }

        assert!(
            saw_repaired_short_symmetric_band,
            "the sweep must include its 3-SE regression cell"
        );
    }

    /// A constraint row whose normal is nearly a combination of the accepted
    /// ones must be dropped — and the reason matters. It is not dropped because
    /// it is undetectable: its pivot sits two hundred times above the bare
    /// `ε·diagonal` limit at which an exactly dependent row stops being
    /// distinguishable. It is dropped because retaining it reports a lift
    /// `G = Σ Aᵀ W⁻¹` whose error exceeds the accuracy this module certifies its
    /// own moments to, and a wrong lift is worse than a missing row that imposes
    /// nothing the retained one does not already impose.
    ///
    /// Both arms are asserted so the gate discriminates: a filter that never
    /// drops fails the near-degenerate arm, one that always drops fails the
    /// resolvable arm.
    #[test]
    fn a_constraint_row_below_the_lift_accuracy_floor_is_dropped_though_detectable() {
        let identity = Array2::<f64>::eye(4);
        let center = Array1::<f64>::zeros(4);

        let mut resolvable = Array2::<f64>::zeros((3, 4));
        resolvable[[0, 0]] = 1.0;
        resolvable[[1, 1]] = 1.0;
        resolvable[[2, 2]] = 1.0;
        let constraints = LinearInequalityConstraints::new(resolvable, Array1::<f64>::zeros(3))
            .expect("orthogonal constraint rows");
        let correction =
            constrained_posterior_correction_from_covariance(&identity, &center, &constraints)
                .expect("orthogonal face")
                .expect("an active face at zero slack");
        assert_eq!(
            correction.rows,
            vec![0, 1, 2],
            "three mutually independent constraint normals must all be retained"
        );

        // Row 1 is row 0 rotated by `sine` in the `Σ` metric, so its pivot is
        // exactly `sine²` against a diagonal of `1 + sine²`.
        let sine = 3.0e-7;
        let pivot = sine * sine;
        let diagonal = 1.0 + pivot;
        let detectability_limit = 2.0 * f64::EPSILON * diagonal;
        assert!(
            pivot > detectability_limit,
            "the fixture must be DETECTABLE, or the drop below proves nothing: pivot \
             {pivot:e} against the bare rank limit {detectability_limit:e}"
        );
        assert!(
            pivot < detectability_limit / ORTHANT_MOMENT_RELATIVE_TOLERANCE,
            "the fixture must sit below the accuracy the first pass demands"
        );

        let mut degenerate = Array2::<f64>::zeros((3, 4));
        degenerate[[0, 0]] = 1.0;
        degenerate[[1, 0]] = 1.0;
        degenerate[[1, 1]] = sine;
        degenerate[[2, 2]] = 1.0;
        let constraints = LinearInequalityConstraints::new(degenerate, Array1::<f64>::zeros(3))
            .expect("near-parallel constraint rows");
        let correction =
            constrained_posterior_correction_from_covariance(&identity, &center, &constraints)
                .expect("near-degenerate face")
                .expect("an active face at zero slack");
        assert_eq!(
            correction.rows,
            vec![0, 2],
            "the near-parallel row must be dropped: retaining it reports a lift whose own \
             defining identity A·G = I fails by more than the certified accuracy"
        );
    }

    /// The retention floor is necessary and NOT sufficient, so the assembled
    /// face has to be checked and the floor raised until it delivers. Because
    /// the filter walks candidates in slack order rather than in pivot order —
    /// a statistical choice, since two near-parallel rows with different offsets
    /// are not the same constraint and the tighter one dominates — its per-row
    /// pivots do not reveal the assembled face's conditioning. Every pivot can
    /// clear the floor while the face as a whole does not.
    ///
    /// A Vandermonde face in clustered nodes is the sharp case: seven rows whose
    /// effective rank is five, all well inside the slack horizon so nothing is
    /// dropped for statistical reasons. Measured `max|A G − I|` on the face this
    /// fixture produces:
    ///
    /// * bare detectability floor, no check — `3.26e-1`, 326× the accuracy this
    ///   module reports its moments to;
    /// * one pass at the derived floor — `1.21e-1`, still 121× over;
    /// * the floor raised by the amount it missed by — `3.53e-5`, inside, in two
    ///   passes.
    ///
    /// So this gate fails on the shipped filter, fails on a single-pass floor
    /// change, and passes only when the realized lift governs the retained face.
    #[test]
    fn the_retained_face_satisfies_the_identity_that_defines_its_lift() {
        const ROWS: usize = 7;
        const DIMENSION: usize = 8;
        const DEGREE: usize = 5;
        const SPACING: f64 = 1.0e-2;

        let mut a = Array2::<f64>::zeros((ROWS, DIMENSION));
        for row in 0..ROWS {
            let node = row as f64 * SPACING;
            for power in 0..DEGREE {
                a[[row, power]] = node.powi(power as i32);
            }
        }
        let constraints = LinearInequalityConstraints::new(a.clone(), Array1::<f64>::zeros(ROWS))
            .expect("clustered Vandermonde rows");

        // Place the centre so every row sits at ~7 standardized units of slack:
        // inside the resolution horizon, so each row is a genuine candidate and
        // nothing is dropped for being statistically irrelevant, while the
        // truncation itself is nearly invisible — which keeps this gate a
        // statement about the lift rather than about the cubature.
        let covariance = Array2::<f64>::eye(DIMENSION);
        let mut center = Array1::<f64>::zeros(DIMENSION);
        center[0] = 7.0;
        for row in 0..ROWS {
            let normal = a.row(row);
            let slack = normal.dot(&center) / normal.dot(&normal).sqrt();
            assert!(
                slack < 8.12 && slack > 6.0,
                "row {row} must be a candidate inside the resolution horizon, got slack {slack}"
            );
        }

        let correction =
            constrained_posterior_correction_from_covariance(&covariance, &center, &constraints)
                .expect("clustered Vandermonde face")
                .expect("an active face inside the horizon");

        assert!(
            correction.rows.len() < ROWS,
            "the fixture must exercise the filter: all {ROWS} rows were retained"
        );
        assert!(
            correction.rows.len() >= 2,
            "the face must not collapse to a single row, or the identity below is vacuous: \
             retained {:?}",
            correction.rows
        );

        let mut departure = 0.0_f64;
        for (i, &row_index) in correction.rows.iter().enumerate() {
            for j in 0..correction.rows.len() {
                let entry = a.row(row_index).dot(&correction.lift.column(j));
                let target = if i == j { 1.0 } else { 0.0 };
                departure = departure.max((entry - target).abs());
            }
        }
        assert!(
            departure <= ORTHANT_MOMENT_RELATIVE_TOLERANCE,
            "the reported lift must satisfy A·G = I, the identity it is defined by, to the \
             accuracy this module certifies its moments to: max|A G - I| = {departure:e} on \
             the retained rows {:?}",
            correction.rows
        );
    }

    /// The cubature must reproduce the closed form when the orthant factorizes
    /// into independent coordinates, which is the only multivariate case with
    /// an exact answer to check against. The bound is the module's own
    /// certified accuracy, [`ORTHANT_MOMENT_RELATIVE_TOLERANCE`], measured on
    /// the pre-truncation scale — asserting tighter would assert something the
    /// algorithm does not promise.
    #[test]
    fn cubature_reproduces_independent_coordinates_within_its_certified_accuracy() {
        let mean = array![-0.5, 0.25, -1.5];
        let covariance = array![[2.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 1.0]];
        let factor = gam_linalg::triangular::cholesky_factor_in_place(
            covariance.view(),
            gam_linalg::triangular::CholeskyGuard::FiniteStrict,
        )
        .expect("independent orthant covariance factors");
        let (moment_mean, moment_covariance) =
            box_truncated_moments(&mean, &vec![f64::INFINITY; mean.len()], &covariance, factor.view())
                .expect("independent orthant");
        for i in 0..3 {
            let (exact_mean, exact_variance) =
                scalar_truncated_moments(mean[i], covariance[[i, i]], f64::INFINITY).expect("scalar");
            let scale = covariance[[i, i]].sqrt();
            assert!(
                (moment_mean[i] - exact_mean[0]).abs()
                    < ORTHANT_MOMENT_RELATIVE_TOLERANCE * scale,
                "coordinate {i} mean {} vs exact {}",
                moment_mean[i],
                exact_mean[0]
            );
            assert!(
                (moment_covariance[[i, i]] - exact_variance[[0, 0]]).abs()
                    < ORTHANT_MOMENT_RELATIVE_TOLERANCE * covariance[[i, i]],
                "coordinate {i} variance {} vs exact {}",
                moment_covariance[[i, i]],
                exact_variance[[0, 0]]
            );
            for j in 0..3 {
                if i != j {
                    assert!(
                        moment_covariance[[i, j]].abs()
                            < ORTHANT_MOMENT_RELATIVE_TOLERANCE
                                * scale
                                * covariance[[j, j]].sqrt(),
                        "independent coordinates must stay uncorrelated under an orthant \
                         truncation, got {} at ({i},{j})",
                        moment_covariance[[i, j]]
                    );
                }
            }
        }
    }

    /// The whole point of the module: the correction lands strictly between the
    /// two answers the two fit paths ship today.
    #[test]
    fn correction_lands_strictly_between_full_space_and_active_face() {
        let covariance = array![[1.0, 0.4], [0.4, 1.0]];
        let constraints =
            LinearInequalityConstraints::new(array![[1.0, 0.0]], array![0.0]).expect("cone");
        // Unconstrained centre BELOW the bound: the constrained mode is pinned.
        let center = array![-0.6, 0.3];
        let correction = constrained_posterior_correction_from_covariance(&covariance, &center, &constraints)
            .expect("correction")
            .expect("an active row");
        let truncated = correction.apply_to_covariance(&covariance);

        // Active-face answer: the same formula with the normal variance removed
        // in full.
        let mut face = covariance.clone();
        let full_removal = correction.lift.dot(&array![[1.0]]).dot(&correction.lift.t());
        face -= &full_removal;

        assert!(
            truncated[[0, 0]] > face[[0, 0]] + 1e-6,
            "truncated variance {} must exceed the active-face answer {}",
            truncated[[0, 0]],
            face[[0, 0]]
        );
        assert!(
            truncated[[0, 0]] < covariance[[0, 0]] - 1e-6,
            "truncated variance {} must fall below the unconstrained answer {}",
            truncated[[0, 0]],
            covariance[[0, 0]]
        );
        assert!(
            face[[0, 0]].abs() < 1e-12,
            "the active-face answer for a single pinned coordinate is exactly zero, got {}",
            face[[0, 0]]
        );
        assert!(
            correction.normal_mean_shift[0] > 0.0,
            "truncation moves the posterior mean INTO the feasible region, shift was {}",
            correction.normal_mean_shift[0]
        );
    }

    /// A constraint far from the posterior centre must leave the covariance
    /// untouched, so unconstrained-in-practice fits keep their exact bytes.
    #[test]
    fn inactive_constraints_produce_no_correction() {
        let covariance = array![[1.0, 0.0], [0.0, 1.0]];
        let constraints =
            LinearInequalityConstraints::new(array![[1.0, 0.0]], array![0.0]).expect("cone");
        let center = array![40.0, 0.0];
        let correction = constrained_posterior_correction_from_covariance(&covariance, &center, &constraints)
            .expect("correction");
        assert!(
            correction.is_none(),
            "a bound 40 posterior standard deviations away cannot move any moment at double \
             precision"
        );
    }

    /// A duplicated constraint row must not make `W` singular.
    #[test]
    fn redundant_rows_are_dropped_by_the_rank_filter() {
        let covariance = array![[1.0, 0.2], [0.2, 1.0]];
        let constraints = LinearInequalityConstraints::new(
            array![[1.0, 0.0], [2.0, 0.0], [0.0, 1.0]],
            array![0.0, 0.0, 0.0],
        )
        .expect("cone");
        let center = array![-0.2, -0.3];
        let correction = constrained_posterior_correction_from_covariance(&covariance, &center, &constraints)
            .expect("correction")
            .expect("active rows");
        assert_eq!(
            correction.rows.len(),
            2,
            "the duplicated half-space must be filtered out, kept rows {:?}",
            correction.rows
        );
    }

    /// The correction must never inflate a variance or drive one negative.
    #[test]
    fn corrected_covariance_stays_between_zero_and_the_unconstrained_answer() {
        let covariance = array![
            [1.0, 0.3, 0.1],
            [0.3, 1.2, -0.2],
            [0.1, -0.2, 0.8]
        ];
        let constraints = LinearInequalityConstraints::new(
            array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            array![0.0, 0.0],
        )
        .expect("cone");
        for center in [
            array![-2.0, -1.0, 0.5],
            array![0.0, 0.0, 0.0],
            array![-0.1, 0.4, -3.0],
        ] {
            let correction = constrained_posterior_correction_from_covariance(&covariance, &center, &constraints)
                .expect("correction")
                .expect("active rows");
            let truncated = correction.apply_to_covariance(&covariance);
            for i in 0..3 {
                assert!(
                    truncated[[i, i]] > 0.0,
                    "coordinate {i} lost all variance at centre {center:?}: {}",
                    truncated[[i, i]]
                );
                assert!(
                    truncated[[i, i]] <= covariance[[i, i]] + 1e-9,
                    "coordinate {i} gained variance at centre {center:?}: {} vs {}",
                    truncated[[i, i]],
                    covariance[[i, i]]
                );
            }
            let diagonal = correction.removed_variance_diagonal();
            for i in 0..3 {
                assert!(
                    (diagonal[i] - (covariance[[i, i]] - truncated[[i, i]])).abs() < 1e-9,
                    "the diagonal-only accessor must agree with the dense correction at {i}"
                );
            }
        }
    }
    // ---------------------------------------------------------------- #2523

    /// The two rows a `linear(min=l, max=u)` term emits: `+e_0ᵀβ ≥ l` and
    /// `−e_0ᵀβ ≥ −u`, exactly as `gam-terms/src/smooth/term_design.rs` builds
    /// them.
    fn two_sided_bound_rows(lower: f64, upper: f64, columns: usize) -> LinearInequalityConstraints {
        let mut a = Array2::<f64>::zeros((2, columns));
        a[[0, 0]] = 1.0;
        a[[1, 0]] = -1.0;
        LinearInequalityConstraints::new(a, array![lower, -upper])
            .expect("two-sided bound rows")
    }

    /// Independent reference: Simpson quadrature of `N(mean, variance)`
    /// restricted to `[0, upper]`, built directly from the density rather than
    /// from any tail function the code under test also uses.
    fn quadrature_box_moments(mean: f64, variance: f64, upper: f64) -> (f64, f64) {
        let sd = variance.sqrt();
        let low = -mean / sd;
        let high = (upper - mean) / sd;
        let reference = if low <= 0.0 && 0.0 <= high {
            0.0
        } else if high < 0.0 {
            high
        } else {
            low
        };
        let panels = 400_000usize;
        let step = (high - low) / panels as f64;
        let (mut mass, mut first, mut second) = (0.0f64, 0.0f64, 0.0f64);
        for index in 0..=panels {
            let z = low + step * index as f64;
            let simpson = if index == 0 || index == panels {
                1.0
            } else if index % 2 == 1 {
                4.0
            } else {
                2.0
            };
            let density = (-(z * z - reference * reference) / 2.0).exp();
            mass += simpson * density;
            first += simpson * density * z;
            second += simpson * density * z * z;
        }
        let m1 = first / mass;
        let m2 = second / mass;
        (mean + sd * m1, variance * (m2 - m1 * m1))
    }

    /// The defect: a row that is anti-parallel to an accepted one carries no new
    /// constraint-normal DIRECTION and is still a constraint. It must survive as
    /// the coordinate's upper limit rather than be dropped as redundant.
    ///
    /// The fixture is the one measured on #2523 — `Σ = I`, `0 ≤ β₀ ≤ 2`, ambient
    /// centre `0.6` — where both walls sit far inside the resolution horizon
    /// (`0.6` and `1.4` standardized against `8.126`), so neither is discarded
    /// as statistically irrelevant.
    #[test]
    fn two_sided_coefficient_bound_keeps_its_far_wall_2523() {
        let columns = 3;
        let covariance = Array2::<f64>::eye(columns);
        let centre = array![0.6, 0.0, 0.0];
        let constraints = two_sided_bound_rows(0.0, 2.0, columns);
        let correction = constrained_posterior_correction_from_covariance(
            &covariance,
            &centre,
            &constraints,
        )
        .expect("two-sided correction")
        .expect("an active two-sided bound corrects the posterior");

        assert_eq!(
            correction.rows.len(),
            1,
            "the anti-parallel row adds no direction, so exactly one is retained"
        );
        let limits = correction.upper_limits();
        assert_eq!(limits.len(), 1);
        assert!(
            (limits[0] - 2.0).abs() < 1e-12,
            "the far wall of [0, 2] must arrive as the coordinate's upper limit, got {}",
            limits[0]
        );

        // The retained law is now `u ~ N(0.6, 1)` on `[0, 2]`, not on `[0, ∞)`.
        // Both moments must be the bounded ones.
        let (bounded_mean, bounded_variance) =
            quadrature_box_moments(0.6, 1.0, 2.0);
        let (half_line_mean, half_line_variance) = quadrature_truncated_moments(0.6, 1.0);
        let reported_mean = 0.6 + correction.normal_mean_shift[0];
        let reported_variance = 1.0 - correction.removed_normal_variance[[0, 0]];
        assert!(
            (reported_mean - bounded_mean).abs() < 1e-6,
            "reported mean {reported_mean} must be the [0,2] mean {bounded_mean}, \
             not the [0,inf) mean {half_line_mean}"
        );
        assert!(
            (reported_variance - bounded_variance).abs() < 1e-6,
            "reported variance {reported_variance} must be the [0,2] variance \
             {bounded_variance}, not the [0,inf) variance {half_line_variance}"
        );
        // Discrimination: the two laws are far apart, so agreeing with one is
        // evidence against the other rather than a bound both would clear.
        assert!(
            (bounded_mean - half_line_mean).abs() > 0.1
                && (bounded_variance - half_line_variance).abs() > 0.1,
            "the fixture must separate the two answers: means {bounded_mean} vs \
             {half_line_mean}, variances {bounded_variance} vs {half_line_variance}"
        );
    }

    /// Control for the test above: push the far wall past the resolution horizon
    /// and the answer must return to the half-line one BIT FOR BIT. A coordinate
    /// with no reachable upper limit is not merely close to the old arithmetic,
    /// it takes it.
    #[test]
    fn a_far_wall_beyond_the_horizon_restores_the_half_line_answer_exactly() {
        let columns = 3;
        let covariance = Array2::<f64>::eye(columns);
        let centre = array![0.6, 0.0, 0.0];
        let two_sided = constrained_posterior_correction_from_covariance(
            &covariance,
            &centre,
            &two_sided_bound_rows(0.0, 40.0, columns),
        )
        .expect("wide two-sided correction")
        .expect("the lower wall is still active");

        let mut lower_only = Array2::<f64>::zeros((1, columns));
        lower_only[[0, 0]] = 1.0;
        let one_sided = constrained_posterior_correction_from_covariance(
            &covariance,
            &centre,
            &LinearInequalityConstraints::new(lower_only, array![0.0]).expect("lower wall"),
        )
        .expect("one-sided correction")
        .expect("an active lower bound corrects the posterior");

        assert_eq!(two_sided.rows.len(), 1);
        assert_eq!(
            two_sided.upper_limits(),
            vec![f64::INFINITY],
            "a wall 39.4 standard deviations away is not a candidate at all"
        );
        assert_eq!(
            two_sided.normal_mean_shift[0], one_sided.normal_mean_shift[0],
            "no reachable upper limit must reproduce the half-line mean shift exactly"
        );
        assert_eq!(
            two_sided.removed_normal_variance[[0, 0]],
            one_sided.removed_normal_variance[[0, 0]],
            "no reachable upper limit must reproduce the half-line variance exactly"
        );
    }

    /// The regime a two-sided bound is declared FOR: the unconstrained fit lands
    /// beyond the far wall, so the retained slab sits deep in a tail. This is
    /// what the reflection inside the cubature and the closed form exist for; a
    /// sign error there puts the posterior mean outside its own box.
    #[test]
    fn a_box_the_unconstrained_centre_overshoots_stays_inside_itself() {
        let columns = 2;
        let covariance = Array2::<f64>::eye(columns);
        // Ambient centre at -3 with the box [-1, 1]: the coordinate
        // `u = beta_0 + 1` has untruncated mean -2 and lives on [0, 2].
        let centre = array![-3.0, 0.0];
        let correction = constrained_posterior_correction_from_covariance(
            &covariance,
            &centre,
            &two_sided_bound_rows(-1.0, 1.0, columns),
        )
        .expect("overshooting correction")
        .expect("both walls bind");

        let limits = correction.upper_limits();
        assert!(
            (limits[0] - 2.0).abs() < 1e-12,
            "the slab is two units wide, got {}",
            limits[0]
        );
        let reported_mean = -2.0 + correction.normal_mean_shift[0];
        let reported_variance = 1.0 - correction.removed_normal_variance[[0, 0]];
        assert!(
            reported_mean > 0.0 && reported_mean < limits[0],
            "the posterior mean of a law supported on [0, {}] cannot sit outside it, \
             got {reported_mean}",
            limits[0]
        );
        // Popoviciu: any law on an interval of width `w` has variance at most
        // `w²/4`. A one-sided answer here would report ~0.06 on a coordinate the
        // box confines to at most 1.0, so this separates them.
        assert!(
            reported_variance > 0.0 && reported_variance <= limits[0] * limits[0] / 4.0,
            "variance {reported_variance} exceeds the width bound for [0, {}]",
            limits[0]
        );
        let (expected_mean, expected_variance) = quadrature_box_moments(-2.0, 1.0, 2.0);
        assert!(
            (reported_mean - expected_mean).abs() < 1e-6
                && (reported_variance - expected_variance).abs() < 1e-6,
            "deep-tail slab moments {reported_mean}/{reported_variance} against the \
             independent quadrature {expected_mean}/{expected_variance}"
        );
    }

    /// The two branches of the scalar closed form are separate code, so they are
    /// checked against each other where they must agree: an upper limit far
    /// enough out that it removes no representable mass.
    #[test]
    fn the_two_sided_scalar_form_meets_the_mills_branch_at_a_distant_wall() {
        for &(mean, variance) in &[(0.6f64, 1.0f64), (-2.5, 1.0), (0.0, 4.0), (3.0, 0.25)] {
            let sd: f64 = variance.sqrt();
            let distant = mean + 40.0 * sd;
            let (bounded_mean, bounded_variance) =
                scalar_truncated_moments(mean, variance, distant).expect("bounded");
            let (open_mean, open_variance) =
                scalar_truncated_moments(mean, variance, f64::INFINITY).expect("half line");
            assert!(
                (bounded_mean[0] - open_mean[0]).abs() <= 1e-12 * open_mean[0].abs().max(1.0),
                "mean {} vs {} at mean={mean} variance={variance}",
                bounded_mean[0],
                open_mean[0]
            );
            assert!(
                (bounded_variance[[0, 0]] - open_variance[[0, 0]]).abs()
                    <= 1e-12 * open_variance[[0, 0]].abs().max(1.0),
                "variance {} vs {} at mean={mean} variance={variance}",
                bounded_variance[[0, 0]],
                open_variance[[0, 0]]
            );
        }
    }

    /// The two-sided closed form across the regimes the reflection switches on,
    /// against Simpson quadrature of the density itself.
    #[test]
    fn the_two_sided_scalar_form_matches_an_independent_quadrature() {
        for &(mean, variance, upper) in &[
            (0.6f64, 1.0f64, 2.0f64),
            (-1.5, 1.0, 0.5),
            (3.0, 0.25, 0.4),
            (-4.0, 1.0, 0.2),
            (0.05, 1.0, 0.1),
            (-2.0, 1.0, 2.0),
            (0.5, 9.0, 12.0),
        ] {
            let (moment_mean, moment_variance) =
                scalar_truncated_moments(mean, variance, upper).expect("two-sided moments");
            let (reference_mean, reference_variance) =
                quadrature_box_moments(mean, variance, upper);
            let scale = variance.sqrt();
            assert!(
                (moment_mean[0] - reference_mean).abs() < 1e-9 * scale,
                "mean {} vs {reference_mean} at mean={mean} variance={variance} upper={upper}",
                moment_mean[0]
            );
            assert!(
                (moment_variance[[0, 0]] - reference_variance).abs() < 1e-9 * variance,
                "variance {} vs {reference_variance} at mean={mean} variance={variance} \
                 upper={upper}",
                moment_variance[[0, 0]]
            );
            assert!(
                moment_mean[0] > 0.0 && moment_mean[0] < upper,
                "the mean of a law on [0, {upper}] must lie inside it, got {}",
                moment_mean[0]
            );
        }
    }

    /// The multivariate cubature over a genuine box, against the same
    /// coordinatewise reference on a product law where the two agree exactly.
    /// A diagonal covariance makes the box-truncated joint the product of its
    /// box-truncated marginals, so the reference needs no second cubature.
    #[test]
    fn the_box_cubature_reproduces_a_product_law_it_cannot_shortcut() {
        let mean = array![0.4, -1.2, 0.9];
        let covariance = Array2::from_diag(&array![1.0, 0.5, 2.0]);
        let upper = vec![1.5, 0.8, f64::INFINITY];
        let factor = gam_linalg::triangular::cholesky_factor_in_place(
            covariance.view(),
            gam_linalg::triangular::CholeskyGuard::FiniteStrict,
        )
        .expect("diagonal factor");
        let (cubature_mean, cubature_covariance) =
            box_truncated_moments(&mean, &upper, &covariance, factor.view()).expect("box moments");
        for i in 0..mean.len() {
            let (reference_mean, reference_variance) =
                scalar_truncated_moments(mean[i], covariance[[i, i]], upper[i])
                    .expect("marginal closed form");
            let sd = covariance[[i, i]].sqrt();
            assert!(
                (cubature_mean[i] - reference_mean[0]).abs()
                    < ORTHANT_MOMENT_RELATIVE_TOLERANCE * sd,
                "coordinate {i} mean {} vs {}",
                cubature_mean[i],
                reference_mean[0]
            );
            assert!(
                (cubature_covariance[[i, i]] - reference_variance[[0, 0]]).abs()
                    < ORTHANT_MOMENT_RELATIVE_TOLERANCE * covariance[[i, i]],
                "coordinate {i} variance {} vs {}",
                cubature_covariance[[i, i]],
                reference_variance[[0, 0]]
            );
        }
        // Independence survives truncation to a product region, so every
        // off-diagonal must vanish. This is what a mis-indexed upper limit would
        // break first.
        for i in 0..mean.len() {
            for j in 0..mean.len() {
                if i == j {
                    continue;
                }
                let sd = (covariance[[i, i]] * covariance[[j, j]]).sqrt();
                assert!(
                    cubature_covariance[[i, j]].abs() < ORTHANT_MOMENT_RELATIVE_TOLERANCE * sd,
                    "a product law truncated to a box stays a product law: entry ({i},{j}) \
                     is {}",
                    cubature_covariance[[i, j]]
                );
            }
        }
    }

    /// The reflection inside the cubature, gated where it decides the answer
    /// rather than where it is merely present.
    ///
    /// A slab sitting `d` standard deviations BELOW the mean has both endpoints
    /// deep in the lower tail, where `Φ̄` is within rounding of one and the
    /// slab's whole mass is the difference between them. Measured against a
    /// Simpson reference on a diagonal `q = 2` law — where the box-truncated
    /// joint is exactly the product of its box-truncated marginals, so the
    /// reference needs no second cubature — the unreflected arithmetic holds to
    /// `d = 9` and then fails completely: at `d = 12` it returns zero for a mean
    /// of 1.9019, and by `d = 40` every node has underflowed and there is no
    /// mass left to normalize. Reflected, the error is 2.2e-5 at `d = 12` and
    /// keeps FALLING with depth, reaching 6.1e-6 at `d = 40`.
    ///
    /// `d = 12` is therefore the shallowest depth at which this test can tell
    /// the two apart, which is why it is the depth used.
    #[test]
    fn a_slab_twelve_deviations_below_the_mean_keeps_its_mass() {
        let depth = 12.0_f64;
        let mean = array![depth, depth];
        let covariance = Array2::<f64>::eye(2);
        let upper = vec![2.0, 2.0];
        let factor = gam_linalg::triangular::cholesky_factor_in_place(
            covariance.view(),
            gam_linalg::triangular::CholeskyGuard::FiniteStrict,
        )
        .expect("identity factor");
        let (cubature_mean, cubature_covariance) =
            box_truncated_moments(&mean, &upper, &covariance, factor.view())
                .expect("a slab deep in a tail still carries mass");
        let (reference_mean, reference_variance) = quadrature_box_moments(depth, 1.0, 2.0);
        // The density rises across the whole slab, so the mean sits near the far
        // wall; a lost slab would report 0 and a mis-signed reflection would
        // report the mirror image near the near wall.
        assert!(
            reference_mean > 1.85 && reference_mean < 2.0,
            "the fixture must place the mean near the far wall, got {reference_mean}"
        );
        for i in 0..2 {
            assert!(
                (cubature_mean[i] - reference_mean).abs() < 1e-3,
                "coordinate {i} mean {} against the Simpson reference {reference_mean}",
                cubature_mean[i]
            );
            assert!(
                (cubature_covariance[[i, i]] - reference_variance).abs() < 1e-3,
                "coordinate {i} variance {} against the Simpson reference {reference_variance}",
                cubature_covariance[[i, i]]
            );
        }
    }

    /// A two-sided bound with no width between its walls is an equality
    /// constraint, and this module reports moments of a density. It refuses
    /// rather than reporting the moments of a point.
    #[test]
    fn coincident_two_sided_walls_are_refused_not_collapsed() {
        let columns = 2;
        let covariance = Array2::<f64>::eye(columns);
        let centre = array![0.5, 0.0];
        let error = constrained_posterior_correction_from_covariance(
            &covariance,
            &centre,
            &two_sided_bound_rows(0.25, 0.25, columns),
        )
        .expect_err("an empty slab has no posterior to report");
        assert!(
            error.contains("no width between them"),
            "the refusal must name the geometry, got: {error}"
        );
    }

    /// The interval path reads the same box the moments do. An equal-tailed
    /// interval for a bounded coordinate cannot leave the coordinate's own
    /// bounds — which is exactly the confidently-wrong report #2523 describes.
    #[test]
    fn a_two_sided_projection_interval_stays_within_its_own_bounds() {
        let columns = 2;
        let covariance = Array2::<f64>::eye(columns);
        let centre = array![0.6, 0.0];
        let constraints = two_sided_bound_rows(0.0, 1.0, columns);
        let correction = constrained_posterior_correction_from_covariance(
            &covariance,
            &centre,
            &constraints,
        )
        .expect("correction")
        .expect("active");
        let geometry = ConstrainedPosteriorGeometry {
            constraints,
            mode: array![0.6, 0.0],
            unconstrained_center: centre,
            correction: Some(correction),
        };
        let (low, high) = constrained_projection_equal_tailed_interval(
            &covariance,
            &geometry,
            &array![1.0, 0.0],
            0.95,
        )
        .expect("two-sided projection interval");
        assert!(
            low >= -1e-9 && high <= 1.0 + 1e-9,
            "a coefficient declared to lie in [0, 1] cannot be reported in [{low}, {high}]"
        );
        assert!(low < high, "the interval must be non-degenerate");
    }
}

/// Coverage gate for #2417.
///
/// The estimand is settled by COVERAGE against a known truth, not by the two
/// fit paths agreeing: two paths agreeing on a wrong covariance is not
/// progress. Each cell simulates from a known constrained model, refits with
/// the production constrained-quadratic solver, and measures what fraction of
/// nominal-95% intervals actually contain the truth under four procedures:
///
/// * **full space** `Σ = φH⁻¹` centred at the constrained mode — what the PIRLS
///   path reported before this change, with no reference to the active geometry;
/// * **active face** `Z(ZᵀHZ)⁻¹Zᵀ` centred at the mode — what the blockwise path
///   reports, reproduced here with the SAME tightness predicate it uses
///   (`scaled slack ≤ ACTIVE_SET_WORKING_FACE_TOL`, `covariance.rs:1583-1592`);
/// * **truncated** `Σ − G(W − Cov[u])Gᵀ` centred at the mode — what this module
///   computes and what the fit now reports;
/// * **truncated, mean-centred** — the same covariance around the truncated
///   posterior MEAN rather than the mode. Not what the fit ships (moving the
///   reported coefficients is out of scope for #2417); measured so the size of
///   the mode-vs-mean effect is on the record rather than asserted.
///
/// Among procedures that reach nominal coverage, expected interval length is
/// the tie-break.
#[cfg(test)]
mod coverage_gate_tests {
    use super::*;
    use gam_linalg::triangular::{CholeskyGuard, cholesky_factor_in_place, cholesky_solve_vector};

    /// Deterministic SplitMix64 — the gate must produce the same numbers on
    /// every host, so no external RNG and no thread-local state.
    struct SplitMix64 {
        state: u64,
    }

    impl SplitMix64 {
        fn new(seed: u64) -> Self {
            Self { state: seed }
        }
        fn next_u64(&mut self) -> u64 {
            self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = self.state;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^ (z >> 31)
        }
        fn unit(&mut self) -> f64 {
            ((self.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
        }
        fn normal(&mut self) -> f64 {
            let (u1, u2) = (self.unit().max(1.0e-12), self.unit());
            (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
        }
    }

    /// Realized coverage and mean half-width of one interval procedure.
    struct CoverageTally {
        covered: usize,
        replicates: usize,
        total_half_width: f64,
    }

    impl CoverageTally {
        fn new() -> Self {
            Self {
                covered: 0,
                replicates: 0,
                total_half_width: 0.0,
            }
        }
        fn record(&mut self, center: f64, half_width: f64, truth: f64) {
            self.replicates += 1;
            self.total_half_width += half_width;
            if (truth - center).abs() <= half_width {
                self.covered += 1;
            }
        }
        fn coverage(&self) -> f64 {
            self.covered as f64 / self.replicates as f64
        }
        fn mean_half_width(&self) -> f64 {
            self.total_half_width / self.replicates as f64
        }
    }

    /// The four procedures the gate compares, in one bundle.
    struct CellResult {
        full_space: CoverageTally,
        active_face: CoverageTally,
        truncated: CoverageTally,
        truncated_mean_centred: CoverageTally,
        pinned_fraction: f64,
    }

    /// Two-sided nominal level the gate reports against.
    const NOMINAL_HALF_WIDTH_MULTIPLIER: f64 = 1.959_963_984_540_054;
    const NOMINAL_COVERAGE: f64 = 0.95;

    /// `Σ = σ²(XᵀX)⁻¹` for a fixed design.
    fn gaussian_posterior_covariance(gram: &Array2<f64>, noise_variance: f64) -> Array2<f64> {
        let p = gram.nrows();
        let factor = cholesky_factor_in_place(gram.view(), CholeskyGuard::FiniteStrict)
            .expect("simulation design is full rank");
        let mut covariance = Array2::<f64>::zeros((p, p));
        for j in 0..p {
            let mut unit = Array1::<f64>::zeros(p);
            unit[j] = 1.0;
            let column = cholesky_solve_vector(&factor, &unit);
            for i in 0..p {
                covariance[[i, j]] = noise_variance * column[i];
            }
        }
        covariance
    }

    /// Rows of `A β ≥ b` that are tight at `beta`, under the SAME scaled-slack
    /// predicate the blockwise covariance path applies at `β̂`.
    fn tight_rows_at(constraints: &LinearInequalityConstraints, beta: &Array1<f64>) -> Vec<usize> {
        let mut tight = Vec::new();
        for row_index in 0..constraints.a.nrows() {
            let row = constraints.a.row(row_index).to_owned();
            let norm = row.dot(&row).sqrt();
            if norm > 0.0
                && (row.dot(beta) - constraints.b[row_index]) / norm
                    <= crate::active_set::ACTIVE_SET_WORKING_FACE_TOL
            {
                tight.push(row_index);
            }
        }
        tight
    }

    /// `Σ_face = Σ − ΣA_tᵀ(A_tΣA_tᵀ)⁻¹A_tΣ` on the tight rows: the active-face
    /// reduction written as the same low-rank removal, so the comparator and
    /// the estimand under test differ ONLY in whether the constraint-normal
    /// variance is removed in full or only by the truncated part.
    fn active_face_variance(
        covariance: &Array2<f64>,
        constraints: &LinearInequalityConstraints,
        tight: &[usize],
        index: usize,
    ) -> f64 {
        if tight.is_empty() {
            return covariance[[index, index]];
        }
        let q = tight.len();
        let mut sigma_at = Array2::<f64>::zeros((covariance.nrows(), q));
        for (position, &row_index) in tight.iter().enumerate() {
            let column = covariance.dot(&constraints.a.row(row_index).to_owned());
            sigma_at.column_mut(position).assign(&column);
        }
        let mut normal = Array2::<f64>::zeros((q, q));
        for (i, &row_i) in tight.iter().enumerate() {
            for j in 0..q {
                normal[[i, j]] = constraints
                    .a
                    .row(row_i)
                    .to_owned()
                    .dot(&sigma_at.column(j).to_owned());
            }
        }
        let Some(factor) = cholesky_factor_in_place(normal.view(), CholeskyGuard::FiniteStrict)
        else {
            // A rank-deficient tight face pins every direction it spans; the
            // face answer for this coordinate is zero variance.
            return 0.0;
        };
        let row = sigma_at.row(index).to_owned();
        let solved = cholesky_solve_vector(&factor, &row);
        covariance[[index, index]] - row.dot(&solved)
    }

    /// One simulation cell: a fixed design, a known truth strictly inside
    /// `A β ≥ 0`, and `replicates` refits under Gaussian noise with a KNOWN
    /// noise scale, so the comparison isolates the covariance question from
    /// dispersion estimation.
    fn run_cell(
        design: &Array2<f64>,
        truth: &Array1<f64>,
        constraints: &LinearInequalityConstraints,
        reported_index: usize,
        noise_sd: f64,
        replicates: usize,
        seed: u64,
    ) -> CellResult {
        let n = design.nrows();
        let p = design.ncols();
        let gram = design.t().dot(design);
        let covariance = gaussian_posterior_covariance(&gram, noise_sd * noise_sd);
        let mut rng = SplitMix64::new(seed);
        let mut result = CellResult {
            full_space: CoverageTally::new(),
            active_face: CoverageTally::new(),
            truncated: CoverageTally::new(),
            truncated_mean_centred: CoverageTally::new(),
            pinned_fraction: 0.0,
        };
        let mean_response = design.dot(truth);
        let mut pinned = 0usize;

        for _ in 0..replicates {
            let mut response = Array1::<f64>::zeros(n);
            for i in 0..n {
                response[i] = mean_response[i] + noise_sd * rng.normal();
            }
            let rhs = design.t().dot(&response);
            let start = crate::active_set::feasible_point_for_linear_constraints(constraints, p)
                .expect("the simulation cone has an interior");
            let (beta_hat, _) = crate::active_set::solve_quadratic_with_linear_constraints(
                &gram,
                &rhs,
                &start,
                constraints,
                None,
            )
            .expect("constrained quadratic solve");

            let full_half_width =
                NOMINAL_HALF_WIDTH_MULTIPLIER * covariance[[reported_index, reported_index]].sqrt();
            result.full_space.record(
                beta_hat[reported_index],
                full_half_width,
                truth[reported_index],
            );

            let tight = tight_rows_at(constraints, &beta_hat);
            if !tight.is_empty() {
                pinned += 1;
            }
            let face_variance =
                active_face_variance(&covariance, constraints, &tight, reported_index);
            result.active_face.record(
                beta_hat[reported_index],
                NOMINAL_HALF_WIDTH_MULTIPLIER * face_variance.max(0.0).sqrt(),
                truth[reported_index],
            );

            // `β_unc = β̂ − Σ ∇ℓ_p(β̂)` with `∇ℓ_p(β̂) = XᵀXβ̂ − Xᵀy`. For this
            // Gaussian cell that is exactly the unconstrained least-squares
            // solution — the centre a truncated Gaussian keeps.
            let penalized_gradient = gram.dot(&beta_hat) - &rhs;
            let center = &beta_hat
                - &(covariance.dot(&penalized_gradient) / (noise_sd * noise_sd));
            let correction =
                constrained_posterior_correction_from_covariance(&covariance, &center, constraints)
                    .expect("truncated correction");
            let (truncated_half_width, truncated_center) = match correction {
                None => (full_half_width, beta_hat[reported_index]),
                Some(ref correction) => {
                    let variance = covariance[[reported_index, reported_index]]
                        - correction.removed_variance_diagonal()[reported_index];
                    (
                        NOMINAL_HALF_WIDTH_MULTIPLIER * variance.max(0.0).sqrt(),
                        correction.posterior_mean(&center)[reported_index],
                    )
                }
            };
            result.truncated.record(
                beta_hat[reported_index],
                truncated_half_width,
                truth[reported_index],
            );
            result.truncated_mean_centred.record(
                truncated_center,
                truncated_half_width,
                truth[reported_index],
            );
        }
        result.pinned_fraction = pinned as f64 / replicates as f64;
        result
    }

    fn report_cell(label: &str, cell: &CellResult) {
        eprintln!(
            "[#2417 coverage] {label}: nominal {NOMINAL_COVERAGE:.2}, {} replicates, mode pinned \
             in {:.1}% of them",
            cell.full_space.replicates,
            100.0 * cell.pinned_fraction
        );
        for (name, tally) in [
            ("full space          ", &cell.full_space),
            ("active face         ", &cell.active_face),
            ("truncated           ", &cell.truncated),
            ("truncated+mean shift", &cell.truncated_mean_centred),
        ] {
            eprintln!(
                "[#2417 coverage]   {name} coverage {:.4}  mean half-width {:.5}",
                tally.coverage(),
                tally.mean_half_width()
            );
        }
    }

    /// A single box bound with the truth half a standard error inside the
    /// feasible region: the regime where the constrained mode pins in about a
    /// third of replicates, so the active-face answer reports a ZERO-WIDTH
    /// interval a third of the time and cannot possibly cover.
    #[test]
    fn box_bound_at_half_a_standard_error_separates_the_three_covariances() {
        let n = 60;
        let mut rng = SplitMix64::new(20_417);
        let mut design = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            design[[i, 0]] = 1.0;
            design[[i, 1]] = rng.normal();
        }
        let gram = design.t().dot(&design);
        let noise_sd = 1.0;
        let covariance = gaussian_posterior_covariance(&gram, noise_sd * noise_sd);
        let standard_error = covariance[[1, 1]].sqrt();
        let truth = Array1::from_vec(vec![0.3, 0.5 * standard_error]);
        let constraints =
            LinearInequalityConstraints::new(ndarray::array![[0.0, 1.0]], ndarray::array![0.0])
                .expect("nonnegativity bound");

        let cell = run_cell(&design, &truth, &constraints, 1, noise_sd, 4000, 91_137);
        report_cell("box bound, truth 0.5 se", &cell);

        // The mode pins often enough that a zero-width interval is not a corner
        // case; if it stopped pinning the cell would stop testing anything.
        assert!(
            cell.pinned_fraction > 0.2,
            "the cell must actually exercise the boundary, pinned fraction {:.3}",
            cell.pinned_fraction
        );
        assert!(
            cell.active_face.coverage() < 0.80,
            "the active-face covariance must under-cover catastrophically here — it reports a \
             zero-width interval whenever the mode pins — but coverage was {:.4}",
            cell.active_face.coverage()
        );
        assert!(
            cell.truncated.coverage() >= NOMINAL_COVERAGE - 0.01,
            "the truncated covariance must reach nominal coverage, got {:.4}",
            cell.truncated.coverage()
        );
        assert!(
            cell.truncated_mean_centred.coverage() >= NOMINAL_COVERAGE - 0.01,
            "and it must still reach it once the centre moves to the truncated posterior \
             mean, got {:.4}",
            cell.truncated_mean_centred.coverage()
        );
        assert!(
            cell.truncated.mean_half_width() < 0.85 * cell.full_space.mean_half_width(),
            "the truncated covariance must buy its coverage with materially SHORTER intervals \
             than the full-space answer: {:.5} vs {:.5}",
            cell.truncated.mean_half_width(),
            cell.full_space.mean_half_width()
        );
        assert!(
            cell.full_space.coverage() >= NOMINAL_COVERAGE,
            "the full-space covariance over-covers by construction, got {:.4}",
            cell.full_space.coverage()
        );
    }

    /// The truth pushed further from the bound, where the mode is pinned less
    /// often but is much further from the truth when it is. This cell is the
    /// counterexample to narrowing the covariance ALONE.
    ///
    /// Measured here: the truncated covariance around the constrained MODE
    /// covers 0.873 against a nominal 0.95 — worse than both the full-space
    /// answer (0.976) and the active face (0.910) — while the SAME covariance
    /// around the truncated posterior MEAN covers 0.966 at an interval 16%
    /// shorter than full space. Truncating the spread without moving the
    /// location keeps the interval centred on a point the posterior says is its
    /// least-likely feasible value, then makes it narrower. The two halves of
    /// the estimand are not separable, and this test exists so that fact cannot
    /// be lost: a covariance-only change is a REGRESSION here.
    #[test]
    fn narrowing_the_covariance_without_moving_the_mean_is_a_regression() {
        let n = 60;
        let mut rng = SplitMix64::new(31_417);
        let mut design = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            design[[i, 0]] = 1.0;
            design[[i, 1]] = rng.normal();
        }
        let gram = design.t().dot(&design);
        let noise_sd = 1.0;
        let covariance = gaussian_posterior_covariance(&gram, noise_sd * noise_sd);
        let standard_error = covariance[[1, 1]].sqrt();
        let truth = Array1::from_vec(vec![-0.2, 1.5 * standard_error]);
        let constraints =
            LinearInequalityConstraints::new(ndarray::array![[0.0, 1.0]], ndarray::array![0.0])
                .expect("nonnegativity bound");

        let cell = run_cell(&design, &truth, &constraints, 1, noise_sd, 4000, 47_903);
        report_cell("box bound, truth 1.5 se", &cell);

        assert!(
            cell.truncated.coverage() < NOMINAL_COVERAGE - 0.02,
            "this cell exists BECAUSE the mode-centred truncated interval under-covers here; \
             if it stopped doing so the counterexample would no longer be testing anything, \
             got {:.4}",
            cell.truncated.coverage()
        );
        assert!(
            cell.truncated.coverage() < cell.active_face.coverage(),
            "the point of the cell: narrowing the covariance while leaving the interval \
             centred on the mode is worse than the active-face answer it replaces, {:.4} vs \
             {:.4}",
            cell.truncated.coverage(),
            cell.active_face.coverage()
        );
        assert!(
            cell.truncated_mean_centred.coverage() >= NOMINAL_COVERAGE - 0.01,
            "moving the centre to the truncated posterior mean recovers nominal coverage with \
             the same covariance, got {:.4}",
            cell.truncated_mean_centred.coverage()
        );
        assert!(
            cell.truncated_mean_centred.mean_half_width() < cell.full_space.mean_half_width(),
            "and it does so with shorter intervals than the full-space answer: {:.5} vs {:.5}",
            cell.truncated_mean_centred.mean_half_width(),
            cell.full_space.mean_half_width()
        );
    }

    /// Two coupled bounds, so the correction runs through the multivariate
    /// orthant cubature rather than the scalar closed form.
    #[test]
    fn two_coupled_bounds_exercise_the_orthant_cubature() {
        let n = 80;
        let mut rng = SplitMix64::new(74_211);
        let mut design = Array2::<f64>::zeros((n, 3));
        for i in 0..n {
            design[[i, 0]] = 1.0;
            let shared = rng.normal();
            design[[i, 1]] = shared;
            // Correlated with column 1, so the two bounds are coupled and the
            // constraint-normal covariance `W` is not diagonal.
            design[[i, 2]] = 0.7 * shared + 0.7 * rng.normal();
        }
        let gram = design.t().dot(&design);
        let noise_sd = 1.0;
        let covariance = gaussian_posterior_covariance(&gram, noise_sd * noise_sd);
        let truth = Array1::from_vec(vec![
            0.25,
            0.5 * covariance[[1, 1]].sqrt(),
            0.5 * covariance[[2, 2]].sqrt(),
        ]);
        let constraints = LinearInequalityConstraints::new(
            ndarray::array![[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            ndarray::array![0.0, 0.0],
        )
        .expect("two nonnegativity bounds");

        let cell = run_cell(&design, &truth, &constraints, 1, noise_sd, 600, 55_301);
        report_cell("two coupled bounds, truth 0.5 se", &cell);

        assert!(
            cell.active_face.coverage() < 0.85,
            "the active-face covariance must under-cover here too, got {:.4}",
            cell.active_face.coverage()
        );
        assert!(
            cell.truncated.coverage() >= NOMINAL_COVERAGE - 0.03,
            "the truncated covariance must reach nominal coverage through the orthant \
             cubature, got {:.4}",
            cell.truncated.coverage()
        );
        assert!(
            cell.truncated.mean_half_width() < cell.full_space.mean_half_width(),
            "shorter intervals at nominal coverage: {:.5} vs {:.5}",
            cell.truncated.mean_half_width(),
            cell.full_space.mean_half_width()
        );
    }

}
