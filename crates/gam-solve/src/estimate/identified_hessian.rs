//! Post-fit inference on the penalized Hessian's identified subspace (#2901 V22).
//!
//! `H = XᵀWX + S_λ` with both terms PSD is singular exactly on
//! `null(X) ∩ null(S_λ)`: coefficient directions no observation and no penalty
//! pins down, such as an intercept column that every partition-of-unity spline
//! block reproduces. PIRLS solves such an `H` min-norm, and the criterion scores
//! `½log|H|₊` over the eigenvectors that
//! `RemlState::intrinsic_hessian_pseudo_logdet_parts_from_eigensystem` keeps.
//! When the strict Cholesky of `H` refuses, post-fit inference takes its traces,
//! covariances, influence and IFT solves against `H⁺ = U·M⁻¹·Uᵀ` over that same
//! kept set, so a fit is summarized on the subspace it was scored on.

use super::EstimationError;
use super::reml::RemlState;
use super::reml::reml_outer_engine::DenseSpectralOperator;
use faer::Side;
use gam_linalg::faer_ndarray::FaerEigh;
use gam_linalg::utils::{CertifiedSymmetricSolveError, certify_linear_system_residual};
use ndarray::Array2;

/// `H⁺` in spectral form over the identified subspace of a dense penalized Hessian.
pub(crate) struct IdentifiedHessianInverse {
    /// The kept eigenvectors `U`, one column per identified direction.
    basis: Array2<f64>,
    /// `M⁻¹ = diag(1/σ_a)` over the kept eigenvalues `σ_a`.
    reduced_inverse: Array2<f64>,
}

impl IdentifiedHessianInverse {
    /// Decompose `hessian` and keep exactly the directions the criterion keeps
    /// for a penalty of rank `penalty_rank`. An eigenvalue below `−p·ε·‖H‖₂`,
    /// PIRLS's own rounding band, is material indefiniteness that `XᵀWX + S_λ`
    /// cannot have, so it is refused rather than dropped.
    pub(crate) fn from_dense(
        hessian: &Array2<f64>,
        penalty_rank: usize,
    ) -> Result<Self, EstimationError> {
        let mut symmetric = hessian.clone();
        gam_linalg::matrix::symmetrize_in_place(&mut symmetric);
        let (eigenvalues, eigenvectors) = symmetric
            .eigh(Side::Lower)
            .map_err(EstimationError::EigendecompositionFailed)?;
        let eigenvalues = eigenvalues.to_vec();
        let rounding_band = DenseSpectralOperator::rounding_band(&eigenvalues);
        let min_eigenvalue = eigenvalues.iter().copied().fold(f64::INFINITY, f64::min);
        if !(min_eigenvalue >= -rounding_band) {
            return Err(EstimationError::HessianNotPositiveDefinite { min_eigenvalue });
        }
        let kept = RemlState::intrinsic_hessian_pseudo_logdet_parts_from_eigensystem(
            &eigenvalues,
            &eigenvectors,
            penalty_rank,
        )?
        .1
        .ok_or_else(|| {
            EstimationError::RemlOptimizationFailed(
                "the penalized Hessian has no positive curvature, so no coefficient direction \
                 is identified"
                    .to_string(),
            )
        })?;
        Ok(Self {
            basis: kept.u_s,
            reduced_inverse: kept.h_proj_inverse,
        })
    }

    /// The number of identified directions.
    pub(crate) fn rank(&self) -> usize {
        self.reduced_inverse.nrows()
    }

    /// The same inverse for `Qs·H·Qsᵀ` with orthogonal `Qs`: the eigenvalues are
    /// unchanged and every kept eigenvector maps through `Qs`.
    pub(crate) fn rotated(&self, qs: &Array2<f64>) -> Self {
        Self {
            basis: qs.dot(&self.basis),
            reduced_inverse: self.reduced_inverse.clone(),
        }
    }

    /// The IFT sensitivity operator `U·M⁻¹·Uᵀ`.
    pub(crate) fn sensitivity(&self) -> crate::sensitivity::FitSensitivity<'_> {
        crate::sensitivity::FitSensitivity::from_projected(&self.basis, &self.reduced_inverse)
    }

    /// `H⁺·B`.
    pub(crate) fn apply(&self, rhs: &Array2<f64>) -> Array2<f64> {
        let mut coordinates = self.basis.t().dot(rhs);
        for (mut row, &inverse) in coordinates
            .rows_mut()
            .into_iter()
            .zip(self.reduced_inverse.diag().iter())
        {
            row *= inverse;
        }
        self.basis.dot(&coordinates)
    }

    /// `U·Uᵀ·B`, the part of `B` that lies in the identified subspace.
    pub(crate) fn project(&self, rhs: &Array2<f64>) -> Array2<f64> {
        self.basis.dot(&self.basis.t().dot(rhs))
    }

    /// Solve `H·X = U·Uᵀ·B` min-norm and certify the residual against `U·Uᵀ·B`,
    /// the right-hand side a solve on the identified subspace can represent.
    pub(crate) fn certified_solve(
        &self,
        hessian: &Array2<f64>,
        rhs: &Array2<f64>,
        label: &str,
    ) -> Result<Array2<f64>, CertifiedSymmetricSolveError> {
        let solution = self.apply(rhs);
        let projected = self.project(rhs);
        let residual = hessian.dot(&solution) - &projected;
        certify_linear_system_residual(
            hessian.nrows(),
            max_abs_entry(hessian),
            &projected,
            &solution,
            &residual,
            label,
        )?;
        Ok(solution)
    }

    /// `H⁺` itself, certified by `H·H⁺ = U·Uᵀ`.
    pub(crate) fn certified_inverse(
        &self,
        hessian: &Array2<f64>,
        label: &str,
    ) -> Result<Array2<f64>, CertifiedSymmetricSolveError> {
        let identity = Array2::<f64>::eye(hessian.nrows());
        let mut inverse = self.apply(&identity);
        gam_linalg::matrix::symmetrize_in_place(&mut inverse);
        let projector = self.project(&identity);
        let residual = hessian.dot(&inverse) - &projector;
        certify_linear_system_residual(
            hessian.nrows(),
            max_abs_entry(hessian),
            &projector,
            &inverse,
            &residual,
            label,
        )?;
        Ok(inverse)
    }
}

fn max_abs_entry(matrix: &Array2<f64>) -> f64 {
    matrix
        .iter()
        .fold(0.0_f64, |acc, value| acc.max(value.abs()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    /// A rank-2 PSD Hessian in three dimensions, rotated off the axes: the
    /// identified inverse is its Moore–Penrose pseudo-inverse, and a right-hand
    /// side with a null-space component solves to the min-norm answer.
    #[test]
    fn identified_inverse_is_the_pseudo_inverse_of_a_singular_hessian() {
        let s = 0.5_f64.sqrt();
        let q = array![[s, s, 0.0], [-s, s, 0.0], [0.0, 0.0, 1.0]];
        let spectrum = array![[3.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]];
        let pseudo_spectrum = array![[1.0 / 3.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]];
        let hessian = q.dot(&spectrum).dot(&q.t());
        let pseudo_inverse = q.dot(&pseudo_spectrum).dot(&q.t());

        let inverse = IdentifiedHessianInverse::from_dense(&hessian, 2).unwrap();
        assert_eq!(inverse.rank(), 2);
        let computed = inverse
            .certified_inverse(&hessian, "identified inverse test")
            .unwrap();
        for (value, expected) in computed.iter().zip(pseudo_inverse.iter()) {
            assert!((value - expected).abs() < 1e-12, "{computed:?} vs {pseudo_inverse:?}");
        }

        let rhs = array![[1.0], [2.0], [5.0]];
        let solution = inverse
            .certified_solve(&hessian, &rhs, "identified solve test")
            .unwrap();
        let expected = pseudo_inverse.dot(&rhs);
        for (value, reference) in solution.iter().zip(expected.iter()) {
            assert!((value - reference).abs() < 1e-12, "{solution:?} vs {expected:?}");
        }
    }
}
