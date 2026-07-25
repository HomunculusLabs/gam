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
    normal_logsf, signed_probit_logcdf_and_mills_ratio, standard_normal_quantile,
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
    /// in constraint-normal coordinates. Strictly positive componentwise for an
    /// active face — the posterior mean is interior even when the mode is not.
    pub normal_mean_shift: Array1<f64>,
    /// Indices, into the caller's constraint system, of the rows retained.
    pub rows: Vec<usize>,
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
        }
        Ok(())
    }
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
    // keeps the rows that bind hardest when a face carries redundant rows.
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

    // Greedy pivoted-Cholesky rank filter on `W = A Σ Aᵀ`. A row that is a
    // linear combination of already-accepted rows adds no constraint-normal
    // direction; keeping it would make `W` singular and `W⁻¹` meaningless.
    let mut rows: Vec<usize> = Vec::new();
    let mut sigma_a_columns: Vec<Array1<f64>> = Vec::new();
    let mut offsets: Vec<f64> = Vec::new();
    let mut w_accepted = Array2::<f64>::zeros((0, 0));
    let mut factor = Array2::<f64>::zeros((0, 0));
    for (row_index, _, sigma_row) in candidates {
        let row = constraints.a.row(row_index);
        let accepted = rows.len();
        let diagonal = row.dot(&sigma_row);
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
        let rank_floor = (accepted + 1) as f64 * f64::EPSILON * diagonal;
        if !(pivot.is_finite() && pivot > rank_floor) {
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

        rows.push(row_index);
        sigma_a_columns.push(sigma_row);
        offsets.push(constraints.b[row_index]);
    }
    if rows.is_empty() {
        return Ok(None);
    }

    let q = rows.len();
    let mut sigma_at = Array2::<f64>::zeros((p, q));
    for (position, column) in sigma_a_columns.iter().enumerate() {
        sigma_at.column_mut(position).assign(column);
    }
    // `G = Σ Aᵀ W⁻¹` solved through the factor built above, one column of `Gᵀ`
    // at a time: `W Gᵀ_col = (Σ Aᵀ)ᵀ_col`.
    let lift = cholesky_solve_right(&factor, &sigma_at)?;

    let mut normal_center = Array1::<f64>::zeros(q);
    for (position, &row_index) in rows.iter().enumerate() {
        normal_center[position] =
            constraints.a.row(row_index).dot(unconstrained_center) - offsets[position];
    }

    let (normal_mean, normal_covariance) = orthant_truncated_moments(&normal_center, &w_accepted)?;

    let mut removed = &w_accepted - &normal_covariance;
    symmetrize_in_place(&mut removed);
    certify_removed_variance(&removed, &w_accepted)?;

    Ok(Some(ConstrainedPosteriorCorrection {
        lift,
        removed_normal_variance: removed,
        normal_mean_shift: normal_mean - normal_center,
        rows,
    }))
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

/// First two moments of `u ~ N(mean, covariance)` restricted to the orthant
/// `u ≥ 0`.
///
/// One dimension has the closed form and is evaluated exactly. Higher
/// dimensions use the Genz separation-of-variables transformation, under which
/// EVERY moment is an integral of the same integrand over the unit cube — so a
/// single cubature delivers the normalizing orthant probability, the mean and
/// the second moment together, instead of the `O(q²)` separate orthant
/// probabilities the Tallis face/edge recursion would need.
fn orthant_truncated_moments(
    mean: &Array1<f64>,
    covariance: &Array2<f64>,
) -> Result<(Array1<f64>, Array2<f64>), String> {
    let q = mean.len();
    if covariance.nrows() != q || covariance.ncols() != q {
        return Err(format!(
            "orthant moments: mean has length {q} but the covariance is {}x{}",
            covariance.nrows(),
            covariance.ncols()
        ));
    }
    if q == 1 {
        return scalar_truncated_moments(mean[0], covariance[[0, 0]]);
    }

    let factor = gam_linalg::triangular::cholesky_factor_in_place(
        covariance.view(),
        gam_linalg::triangular::CholeskyGuard::FiniteStrict,
    )
    .ok_or_else(|| {
        "orthant moments: the constraint-normal covariance W = AΣAᵀ is not numerically \
         positive definite"
            .to_string()
    })?;

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
            factor.view(),
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
                "orthant moments for a {q}-dimensional constraint face did not converge: \
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

/// Evaluate Genz nodes `first..last` of the Kronecker sequence and fold them
/// into `accumulator`.
fn accumulate_orthant_nodes(
    accumulator: &mut OrthantAccumulator,
    mean: &Array1<f64>,
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
            let lower = bound / factor[[i, i]];
            let log_tail = normal_logsf(lower);
            if !log_tail.is_finite() {
                // The remaining feasible mass along this coordinate underflowed
                // to zero: the node contributes nothing and cannot be
                // renormalized, so drop it rather than propagate a NaN.
                log_weight = f64::NEG_INFINITY;
                break;
            }
            log_weight += log_tail;
            // Tent-periodized Kronecker lattice. The raw sequence leaves the
            // integrand non-periodic across the cube face, which costs the
            // lattice rule most of its rate; folding `x ↦ 1 − |2x − 1|`
            // preserves the uniform measure and periodizes it.
            let lattice = {
                let raw = offset * generator[i];
                let fractional = raw - raw.floor();
                1.0 - (2.0 * fractional - 1.0).abs()
            };
            // `Φ̄(z_i) = (1 − x_i)·Φ̄(lower)` inverted on the upper tail, so a
            // deeply pinned coordinate never forms `1 − Φ(·)` in probability
            // space. Both factors can round to one (an inactive coordinate at
            // the very edge of the lattice cell), which would ask for `Φ̄⁻¹(1)`;
            // the smallest representable log-probability answers that with the
            // far-left endpoint, which is what the region actually is there.
            let log_fraction = (1.0 - lattice).max(f64::MIN_POSITIVE).ln();
            let log_upper_tail = log_fraction + log_tail;
            let resolved = if log_upper_tail < 0.0 {
                log_upper_tail
            } else {
                -f64::MIN_POSITIVE
            };
            z[i] = -standard_normal_quantile_from_log_cdf(resolved)
                .map_err(|error| format!("orthant cubature coordinate {i}: {error}"))?;
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

/// Closed-form moments of `N(mean, variance)` restricted to `[0, ∞)`.
fn scalar_truncated_moments(
    mean: f64,
    variance: f64,
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
    Ok((
        Array1::from_elem(1, truncated_mean),
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
        let (mean, variance) = scalar_truncated_moments(0.0, 1.0).expect("half normal");
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
            let (deep_mean, deep) = scalar_truncated_moments(center, 1.0).expect("deep tail");
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
        let (_, at_eight) = scalar_truncated_moments(-8.0, 1.0).expect("deep tail");
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
        let (far_mean, far_variance) = scalar_truncated_moments(10.0, 4.0).expect("inactive");
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
        let (moment_mean, moment_covariance) =
            orthant_truncated_moments(&mean, &covariance).expect("independent orthant");
        for i in 0..3 {
            let (exact_mean, exact_variance) =
                scalar_truncated_moments(mean[i], covariance[[i, i]]).expect("scalar");
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
