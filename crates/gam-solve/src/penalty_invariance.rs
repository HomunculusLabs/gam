//! The exact invariance a penalized criterion has in its own smoothing
//! parameters, and the subspace a curvature certificate must therefore refuse
//! to judge (#2676).
//!
//! # The invariance
//!
//! Every criterion in this crate — REML, LAML, the profiled Gaussian score,
//! their Firth/Jeffreys variants — depends on the smoothing parameters
//! `lambda` ONLY through the assembled penalty
//!
//! ```text
//!     P_lambda(beta) = sum_i lambda_i (beta - mu_i)' S_i (beta - mu_i).
//! ```
//!
//! So if a vector `w` satisfies `sum_i w_i S_i = 0` **and** the two companion
//! conditions its prior means impose (see [`PenaltyMapInvariance`]), then
//! `P_{lambda + s w} = P_lambda` identically in `s`, and the criterion is
//! EXACTLY constant along that line of `lambda`. Nothing about the fit, the
//! data, or the family enters: it is a property of the penalty map alone.
//!
//! Such a `w` is precisely a null vector of the Gram matrix
//! `G_ij = <A_i, A_j>_F` of the penalty operators, because
//! `w' G w = ||sum_i w_i A_i||_F^2`.
//!
//! # Why the certificate has to know
//!
//! `rho = log lambda` is a nonlinear reparameterisation, so for any smooth `V`
//!
//! ```text
//!     H_rho = diag(lambda) H_lambda diag(lambda) + diag(g_rho)          (*)
//! ```
//!
//! holds exactly — the second term is pure chain rule and carries no curvature.
//! Lift `w` to rho by `t = diag(lambda)^{-1} w`. Then `H_lambda w = 0` gives
//!
//! ```text
//!     t' H_rho t = sum_k (g_rho)_k t_k^2        EXACTLY, at every point,
//! ```
//!
//! whose magnitude is bounded by `sum_k |(g_rho)_k| t_k^2` — which is
//! *verbatim* the per-direction gradient floor
//! [`crate::estimate::smoothing_correction::invert_identified_rho_hessian`]
//! and the `H + diag(|g|)` test in [`crate::rho_optimizer::run`] compare
//! against. A direction of this subspace does not sit NEAR the decision
//! boundary of those gates; it sits ON it, by identity, and which side it
//! lands on is decided by the disagreement between the gradient evaluation and
//! the Hessian evaluation.
//!
//! Measured on `geo_disease_matern` (`examples/repro2676_geo_disease_matern.rs`):
//! `sigma = 2.0930992e-5`, `sum_k g_k v_k^2 = 2.0946774e-5`, intrinsic
//! `-1.578e-8` — the identity holding to `7.5e-4` relative, with the gate's
//! whole verdict riding on the sign of that residual.
//!
//! The repair is therefore NOT a wider floor. The comparison is degenerate, not
//! under-resolved: **deflate the subspace, then apply the existing, unchanged
//! rule to its complement**, so that no direction is judged by a test whose
//! boundary it occupies identically.
//!
//! # What deflating cannot hide
//!
//! * *Genuine curvature.* On the deflated subspace the ρ-curvature is
//!   `sum_k g_k t_k^2` — a pure function of the gradient, which the outer loop
//!   has already certified against its own stationarity bound. There is no
//!   second-order information there to lose. #2665's `lambda_min = -1.6e3`
//!   saddle is not in it and still refuses.
//! * *A rho-prior's curvature.* When a prior on `rho` is present the criterion
//!   is no longer exactly flat along the lift, and (*) picks up the prior's own
//!   second derivative. Every prior this crate offers — `Normal`,
//!   `GammaPrecision` (`rate·e^rho`), and the PC prior
//!   (`rho/2 + theta·e^{-rho/2}`) — is CONVEX in rho, so that addition is
//!   positive semidefinite. Deflating can only discard a direction whose
//!   curvature was `artifact + (something >= 0)`; the only way it could have
//!   refused is if the artifact dominated, which is the very round-off verdict
//!   this module exists to remove.
//!
//! # What it does NOT claim
//!
//! Nothing here says the deflated directions are minima. It says they are not
//! MEASUREMENTS — the sibling distinction `gam_math::score_opt` already draws
//! when it reports that a request is unsatisfiable rather than that a bound was
//! violated (#2614).

use gam_terms::construction::CanonicalPenalty;
use ndarray::{Array1, Array2};

/// The exact null space of the penalty map, in `lambda` coordinates.
///
/// Built from the canonical penalties alone. No tolerance is chosen: the rank
/// boundary is decided by the eigensolver's own backward error under Weyl's law
/// (#2690), the same instrument
/// [`crate::estimate::smoothing_correction`] already uses for this Gram.
#[derive(Debug, Clone)]
pub struct PenaltyMapInvariance {
    /// Orthonormal columns spanning `null(G)`, shape `k x d`.
    basis: Array2<f64>,
    /// The Weyl resolution the rank boundary was decided at.
    resolution: f64,
}

impl PenaltyMapInvariance {
    /// Dimension of the invariance, i.e. the certified structural nullity of
    /// the penalty map. This is the number `k - rank(G)` the smoothing
    /// correction's nullity identity is denominated in.
    pub fn dimension(&self) -> usize {
        self.basis.ncols()
    }

    /// The resolution the rank boundary was decided at.
    pub fn resolution(&self) -> f64 {
        self.resolution
    }

    /// Orthonormal basis of `null(G)` in `lambda` coordinates, `k x d`.
    pub fn lambda_basis(&self) -> &Array2<f64> {
        &self.basis
    }

    /// Build from the canonical penalty bundle.
    ///
    /// The Gram is taken over the AUGMENTED operators
    ///
    /// ```text
    ///     A_i = [[ S_i,        -S_i mu_i        ],
    ///            [ -mu_i' S_i,  mu_i' S_i mu_i  ]]
    /// ```
    ///
    /// so that `sum_i w_i A_i = 0` is equivalent to the FULL centered quadratic
    /// `sum_i lambda_i (beta - mu_i)' S_i (beta - mu_i)` being invariant along
    /// `w` — the quadratic, linear and constant parts all at once. With zero
    /// prior means (the overwhelmingly common case) `A_i` is `S_i` bordered by
    /// zeros and the Gram is bit-identical to the plain `tr(S_i S_j)` one, so
    /// this generalisation cannot move any existing verdict; with nonzero
    /// means it can only ADD conditions, i.e. shrink the invariance, which is
    /// the conservative direction.
    pub fn from_canonical_penalties(
        canonical: &[CanonicalPenalty],
        coefficient_dimension: usize,
    ) -> Result<Self, String> {
        use gam_linalg::faer_ndarray::FaerEigh;

        let k = canonical.len();
        if k == 0 {
            return Ok(Self {
                basis: Array2::zeros((0, 0)),
                resolution: 0.0,
            });
        }
        for (index, penalty) in canonical.iter().enumerate() {
            let block_dimension = penalty.col_range.end.saturating_sub(penalty.col_range.start);
            if penalty.col_range.end > coefficient_dimension
                || penalty.local.dim() != (block_dimension, block_dimension)
                || penalty.prior_mean.len() != block_dimension
            {
                return Err(format!(
                    "canonical penalty {index} has range {:?}, local shape {:?}, prior mean length \
                     {}, coefficient dimension {coefficient_dimension}",
                    penalty.col_range,
                    penalty.local.dim(),
                    penalty.prior_mean.len(),
                ));
            }
        }

        // The centering vectors c_i = S_i mu_i, in GLOBAL coefficient
        // coordinates so that overlapping blocks add correctly, and the
        // scalars q_i = mu_i' S_i mu_i.
        let mut centering = Array2::<f64>::zeros((k, coefficient_dimension));
        let mut quadratic = Array1::<f64>::zeros(k);
        for (index, penalty) in canonical.iter().enumerate() {
            if penalty.prior_mean.iter().all(|value| *value == 0.0) {
                continue;
            }
            let start = penalty.col_range.start;
            let block = penalty.col_range.end - start;
            for row in 0..block {
                let mut accumulated = 0.0_f64;
                for col in 0..block {
                    accumulated += penalty.local[[row, col]] * penalty.prior_mean[col];
                }
                centering[[index, start + row]] = accumulated;
                quadratic[index] += penalty.prior_mean[row] * accumulated;
            }
        }

        // Gram of the unscaled augmented maps. Positive lambdas only rescale
        // the columns of the map and therefore cannot change this rank.
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
                // The two border blocks of A_i contribute 2 c_i . c_j, the
                // corner contributes q_i q_j.
                let mut border = 0.0_f64;
                for column in 0..coefficient_dimension {
                    border += centering[[i, column]] * centering[[j, column]];
                }
                inner += 2.0 * border + quadratic[i] * quadratic[j];
                gram[[i, j]] = inner;
                gram[[j, i]] = inner;
            }
        }

        let (eigenvalues, eigenvectors) = gram
            .eigh(faer::Side::Lower)
            .map_err(|error| format!("penalty-map Gram eigendecomposition failed: {error}"))?;
        // The Gram is assembled analytically, so its rank boundary is judged by
        // the Weyl law (#2690), never by the finite-difference one.
        let resolution = crate::estimate::smoothing_correction::eigenpair_backward_error_bound(
            &gram,
            &eigenvalues,
            &eigenvectors,
        )?
        .resolution();
        let minimum = eigenvalues.iter().copied().fold(f64::INFINITY, f64::min);
        if minimum < -resolution {
            let negated = -resolution;
            return Err(format!(
                "penalty-map Gram matrix has negative eigenvalue {minimum:.3e} below \
                 backward-error bound {negated:.3e}"
            ));
        }
        let null_columns: Vec<usize> = (0..k)
            .filter(|&index| eigenvalues[index] <= resolution)
            .collect();
        let mut basis = Array2::<f64>::zeros((k, null_columns.len()));
        for (target, &source) in null_columns.iter().enumerate() {
            basis
                .column_mut(target)
                .assign(&eigenvectors.column(source));
        }
        Ok(Self { basis, resolution })
    }

    /// Lift the invariance to the outer coordinate vector.
    ///
    /// `t = diag(lambda)^{-1} w` is the tangent, at `lambda`, of the curve in
    /// `rho` that the straight line `lambda + s w` traces. `theta_dimension`
    /// and `rho_offset` embed it into a wider outer vector (the exact-joint
    /// spatial route optimises `theta = (rho, psi)`; a mixture/SAS route
    /// appends link coordinates), leaving every non-`rho` coordinate zero:
    /// those coordinates are not part of the invariance, and the identity above
    /// holds for the embedded vector because its non-`rho` components vanish.
    ///
    /// Returns `None` when there is nothing to deflate, so callers stay on the
    /// bit-identical legacy path.
    pub fn theta_directions(
        &self,
        lambdas: &Array1<f64>,
        theta_dimension: usize,
        rho_offset: usize,
    ) -> Option<Array2<f64>> {
        let k = self.basis.nrows();
        if self.basis.ncols() == 0 || k == 0 {
            return None;
        }
        if lambdas.len() != k || rho_offset + k > theta_dimension {
            return None;
        }
        if !lambdas.iter().all(|value| value.is_finite() && *value > 0.0) {
            return None;
        }
        let mut lifted = Array2::<f64>::zeros((theta_dimension, self.basis.ncols()));
        for column in 0..self.basis.ncols() {
            for row in 0..k {
                lifted[[rho_offset + row, column]] = self.basis[[row, column]] / lambdas[row];
            }
        }
        orthonormalize_columns(&lifted)
    }
}

/// Modified Gram-Schmidt with a relative drop tolerance, returning `None` when
/// nothing survives.
///
/// The drop tolerance is the classical loss-of-orthogonality scale
/// `64 * n * EPSILON` relative to the incoming column norm — the same
/// arithmetic-floor coefficient the sibling
/// [`crate::estimate::smoothing_correction::eigenpair_backward_error_bound`]
/// uses, kept identical so the two places that decide "this is round-off" agree.
/// A column whose residual against the accepted basis has fallen that far is
/// numerically dependent, and keeping it would admit a direction determined
/// entirely by round-off.
pub fn orthonormalize_columns(columns: &Array2<f64>) -> Option<Array2<f64>> {
    let rows = columns.nrows();
    let mut accepted: Vec<Array1<f64>> = Vec::with_capacity(columns.ncols());
    let drop_relative = 64.0 * (rows.max(1) as f64) * f64::EPSILON;
    for index in 0..columns.ncols() {
        let mut vector = columns.column(index).to_owned();
        let incoming = vector.dot(&vector).sqrt();
        if !(incoming > 0.0) || !incoming.is_finite() {
            continue;
        }
        // Twice is enough (Kahan-Parlett): one pass can lose orthogonality on a
        // nearly dependent column, two passes recover it or reveal dependence.
        for _ in 0..2 {
            for basis_vector in &accepted {
                let projection = vector.dot(basis_vector);
                vector.scaled_add(-projection, basis_vector);
            }
        }
        let residual = vector.dot(&vector).sqrt();
        if !residual.is_finite() || residual <= drop_relative * incoming {
            continue;
        }
        vector.mapv_inplace(|value| value / residual);
        accepted.push(vector);
    }
    if accepted.is_empty() {
        return None;
    }
    let mut basis = Array2::<f64>::zeros((rows, accepted.len()));
    for (column, vector) in accepted.iter().enumerate() {
        basis.column_mut(column).assign(vector);
    }
    Some(basis)
}

/// Orthonormal basis of the subspace a curvature certificate is entitled to
/// judge: the orthogonal complement of `span({e_k : k in excluded} U deflate)`.
///
/// Returns `None` when the complement is empty (nothing left to judge) and
/// `Some(Z)` with `Z' Z = I` otherwise. With `deflate = None` this is exactly
/// the indicator basis of the un-excluded coordinates, so `Z' H Z` is the
/// interior sub-block the certificate has always taken — bit for bit.
pub fn judged_subspace_basis(
    dimension: usize,
    excluded: &[usize],
    deflate: Option<&Array2<f64>>,
) -> Option<Array2<f64>> {
    if dimension == 0 {
        return None;
    }
    let excluded_set: std::collections::BTreeSet<usize> =
        excluded.iter().copied().filter(|k| *k < dimension).collect();
    let interior: Vec<usize> = (0..dimension)
        .filter(|k| !excluded_set.contains(k))
        .collect();
    if interior.is_empty() {
        return None;
    }
    let deflate = match deflate {
        Some(matrix) if matrix.nrows() == dimension && matrix.ncols() > 0 => matrix,
        // Nothing to deflate: the interior indicator basis, which reproduces
        // the historical sub-block extraction exactly.
        _ => {
            let mut basis = Array2::<f64>::zeros((dimension, interior.len()));
            for (column, &row) in interior.iter().enumerate() {
                basis[[row, column]] = 1.0;
            }
            return Some(basis);
        }
    };
    // Project the deflation directions onto the interior coordinates and
    // re-orthonormalise: an excluded coordinate is already removed from the
    // judged space, so only the interior part of each direction can constrain
    // what is left.
    let mut restricted = Array2::<f64>::zeros((interior.len(), deflate.ncols()));
    for column in 0..deflate.ncols() {
        for (target, &row) in interior.iter().enumerate() {
            restricted[[target, column]] = deflate[[row, column]];
        }
    }
    let interior_dimension = interior.len();
    let orthonormal = match orthonormalize_columns(&restricted) {
        Some(basis) => basis,
        None => {
            let mut basis = Array2::<f64>::zeros((dimension, interior_dimension));
            for (column, &row) in interior.iter().enumerate() {
                basis[[row, column]] = 1.0;
            }
            return Some(basis);
        }
    };
    // Complement inside the interior block, taken as the range of the
    // orthogonal projector `P = I - Q Q'`. `P` is symmetric with spectrum
    // exactly {0, 1}, so selecting eigenvalues above 1/2 is a decision with an
    // O(1) margin rather than one taken at round-off scale — the whole point of
    // this module is to stop deciding things at round-off scale.
    use gam_linalg::faer_ndarray::FaerEigh;
    let mut projector = Array2::<f64>::eye(interior_dimension);
    projector -= &orthonormal.dot(&orthonormal.t());
    gam_linalg::matrix::symmetrize_in_place(&mut projector);
    let (eigenvalues, eigenvectors) = projector.eigh(faer::Side::Lower).ok()?;
    let kept: Vec<usize> = (0..interior_dimension)
        .filter(|&index| eigenvalues[index] > 0.5)
        .collect();
    if kept.is_empty() {
        return None;
    }
    let mut basis = Array2::<f64>::zeros((dimension, kept.len()));
    for (column, &source) in kept.iter().enumerate() {
        for (target, &row) in interior.iter().enumerate() {
            basis[[row, column]] = eigenvectors[[target, source]];
        }
    }
    Some(basis)
}

/// Compress a symmetric matrix onto the judged subspace: `Z' H Z`.
pub fn compress_to_judged_subspace(matrix: &Array2<f64>, basis: &Array2<f64>) -> Array2<f64> {
    let mut compressed = basis.t().dot(matrix).dot(basis);
    gam_linalg::matrix::symmetrize_in_place(&mut compressed);
    compressed
}

#[cfg(test)]
#[path = "penalty_invariance_tests.rs"]
mod penalty_invariance_tests;
