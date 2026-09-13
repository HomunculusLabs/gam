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
//!
//! That kept set is a step function of ρ: `½log|H|₊` jumps by `½ln σ` where a
//! direction crosses the rounding band. [`certify_fitted_identified_rank`]
//! certifies at finalization that the rank is the same over the outer
//! certificate's own Newton step, so the derivative certificate at ρ̂ describes
//! a criterion that is smooth where the certificate applies.

use super::EstimationError;
use super::reml::RemlState;
use super::reml::reml_outer_engine::DenseSpectralOperator;
use faer::Side;
use gam_linalg::faer_ndarray::FaerEigh;
use gam_linalg::utils::{CertifiedSymmetricSolveError, certify_linear_system_residual};
use ndarray::{Array1, Array2, ArrayView1, s};

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

/// How far the penalized Hessian's spectrum can move over a smoothing-parameter
/// step (#2901 V22).
///
/// Along `ρ ↦ ρ + δρ` the criterion's Hessian is
/// `H' = Σ_k e^{δρ_k}·λ_k S_k + XᵀW'X`. Every `λ_k S_k ⪰ 0`, so with
/// `t = max_k |δρ_k|` the penalty satisfies `e^{−t}·S_λ ⪯ S'_λ ⪯ e^{t}·S_λ`
/// exactly, for any finite step. The curvature weights move through `β̂`, and a
/// row with `W_i > 0` moves by at most `m·W_i`, `m = max_i |ΔW_i|/W_i`. In
/// Loewner order
///
/// ```text
///   s₋·H − E₋ ⪯ H' ⪯ s₊·H + E₊,   s₊ = max(e^t, 1 + m),   s₋ = min(e^{−t}, 1 − m),
/// ```
///
/// where `E±` collects the rows whose weight is not positive:
/// `‖E±‖₂ ≤ a = ‖Σ_{W_i ≤ 0} v_i·x_i x_iᵀ‖₂` with
/// `v_i = max(s₊ − 1, 1 − s₋)·|W_i| + |ΔW_i|`. By Weyl every eigenvalue `σ` of
/// `H` is carried into `[s₋σ − a, s₊σ + a]`. The bound is relative: a railed λ
/// moves the directions it dominates in proportion to themselves, and never
/// charges its magnitude to a direction it does not reach.
#[derive(Clone, Copy, Debug)]
pub(crate) struct HessianSpectrumMotion {
    /// `s₋`.
    lower_factor: f64,
    /// `s₊`.
    upper_factor: f64,
    /// `a`.
    additive: f64,
    /// Whether every weight is positive or still at zero, with `m ≤ 1`: then
    /// `XᵀW'X ⪰ 0` everywhere the step reaches, so `H' ⪰ S'_λ`.
    weights_stay_nonnegative: bool,
}

impl HessianSpectrumMotion {
    /// The motion over a step whose largest coordinate is `step_radius`, for
    /// curvature `weights` whose first-order motion over the step is at most
    /// `weight_motion`, row by row. `nonpositive_row_norm(v)` is
    /// `‖Xᵀdiag(v)X‖₂` for a nonnegative `v` that vanishes on every row with a
    /// positive weight; it is asked only when some other row carries mass.
    pub(crate) fn over_step(
        step_radius: f64,
        weights: ArrayView1<'_, f64>,
        weight_motion: ArrayView1<'_, f64>,
        nonpositive_row_norm: impl FnOnce(&Array1<f64>) -> Result<f64, EstimationError>,
    ) -> Result<Self, EstimationError> {
        if !(step_radius.is_finite() && step_radius >= 0.0) || weights.len() != weight_motion.len()
        {
            return Err(EstimationError::InvalidInput(format!(
                "Hessian spectrum motion needs a finite nonnegative step radius and one weight \
                 motion per row: radius {step_radius:.4e}, {} weights, {} motions",
                weights.len(),
                weight_motion.len()
            )));
        }
        let mut relative_motion = 0.0_f64;
        let mut still_or_positive = true;
        for (&weight, &motion) in weights.iter().zip(weight_motion.iter()) {
            if !(weight.is_finite() && motion.is_finite() && motion >= 0.0) {
                return Err(EstimationError::InvalidInput(format!(
                    "Hessian spectrum motion needs finite weights and finite nonnegative weight \
                     motions: weight {weight:.4e}, motion {motion:.4e}"
                )));
            }
            if weight > 0.0 {
                relative_motion = relative_motion.max(motion / weight);
            } else {
                still_or_positive &= weight == 0.0 && motion == 0.0;
            }
        }
        let growth = step_radius.exp();
        let upper_factor = growth.max(1.0 + relative_motion);
        let lower_factor = growth.recip().min(1.0 - relative_motion);
        let spread = (upper_factor - 1.0).max(1.0 - lower_factor);
        let mass: Array1<f64> = weights
            .iter()
            .zip(weight_motion.iter())
            .map(|(&weight, &motion)| {
                if weight > 0.0 {
                    0.0
                } else {
                    spread * weight.abs() + motion
                }
            })
            .collect();
        let additive = if mass.iter().any(|&value| value > 0.0) {
            nonpositive_row_norm(&mass)?
        } else {
            0.0
        };
        if !(additive.is_finite() && additive >= 0.0) {
            return Err(EstimationError::InvalidInput(format!(
                "Hessian spectrum motion: the nonpositive-weight row norm {additive:.4e} is not \
                 a finite nonnegative spectral norm"
            )));
        }
        Ok(Self {
            lower_factor,
            upper_factor,
            additive,
            weights_stay_nonnegative: still_or_positive && relative_motion <= 1.0,
        })
    }
}

/// The outer certificate's own Newton displacement `|H_ρ⁺·g|` at ρ̂, coordinate
/// by coordinate (#2901 V22): how far from ρ̂ the stationary point it vouches for
/// can lie. A coordinate certified on a rail is certified by its tail law, not
/// by a derivative at ρ̂, so it moves nothing here. The inverse is taken over the
/// interior block's eigenvalues outside that block's own rounding band: a
/// direction whose curvature the arithmetic does not resolve localizes nothing.
pub(crate) fn certificate_newton_displacement(
    hessian_rho: &Array2<f64>,
    gradient: &Array1<f64>,
    railed: &[usize],
) -> Result<Array1<f64>, EstimationError> {
    let dimension = gradient.len();
    if hessian_rho.dim() != (dimension, dimension) {
        return Err(EstimationError::InvalidInput(format!(
            "certificate Newton displacement: a {}x{} outer Hessian against {dimension} gradient \
             coordinates",
            hessian_rho.nrows(),
            hessian_rho.ncols()
        )));
    }
    let interior: Vec<usize> = (0..dimension)
        .filter(|index| !railed.contains(index))
        .collect();
    let mut displacement = Array1::<f64>::zeros(dimension);
    if interior.is_empty() {
        return Ok(displacement);
    }
    let mut block = Array2::from_shape_fn((interior.len(), interior.len()), |(row, column)| {
        hessian_rho[[interior[row], interior[column]]]
    });
    gam_linalg::matrix::symmetrize_in_place(&mut block);
    let (eigenvalues, eigenvectors) = block
        .eigh(Side::Lower)
        .map_err(EstimationError::EigendecompositionFailed)?;
    let rounding_band = DenseSpectralOperator::rounding_band(&eigenvalues.to_vec());
    let interior_gradient: Array1<f64> = interior.iter().map(|&index| gradient[index]).collect();
    let coordinates = eigenvectors.t().dot(&interior_gradient);
    let mut interior_step = Array1::<f64>::zeros(interior.len());
    for (column, (&sigma, &coordinate)) in eigenvalues.iter().zip(coordinates.iter()).enumerate() {
        if sigma.abs() > rounding_band {
            interior_step.scaled_add(coordinate / sigma, &eigenvectors.column(column));
        }
    }
    for (&index, &step) in interior.iter().zip(interior_step.iter()) {
        displacement[index] = step.abs();
    }
    if displacement.iter().all(|value| value.is_finite()) {
        Ok(displacement)
    } else {
        Err(EstimationError::InvalidInput(format!(
            "certificate Newton displacement is not finite: {displacement}"
        )))
    }
}

/// The identified rank of a penalized Hessian, certified constant over a step,
/// with the spectrum numbers it was judged at.
#[derive(Clone, Copy, Debug)]
pub(crate) struct IdentifiedRankCertificate {
    /// The number of identified coefficient directions.
    pub(crate) rank: usize,
    /// The smallest identified eigenvalue `σ_r`.
    pub(crate) smallest_identified: f64,
    /// The largest unidentified eigenvalue `σ_{r+1}`; `None` at full rank.
    pub(crate) largest_unidentified: Option<f64>,
    /// `H`'s rounding band `p·ε·‖H‖₂`.
    pub(crate) band: f64,
}

/// Certify that the identified rank of a penalized Hessian with `eigenvalues`,
/// for a penalty of rank `penalty_rank`, is the same at every point `motion`
/// reaches (#2901 V22).
///
/// The criterion prices `½log|H|₊` over the top
/// [`DenseSpectralOperator::identified_rank`] eigenvalues: those above
/// `p·ε·‖H‖₂`, and never fewer than `rank(S_λ)`. Over the step the smallest kept
/// eigenvalue can fall to `s₋σ_r − a` while the band rises to
/// `p·ε·(s₊‖H‖₂ + a)`, and the largest dropped eigenvalue can rise to
/// `s₊σ_{r+1} + a` while the band falls to `p·ε·(s₋‖H‖₂ − a)`. The rank is
/// constant when each stays on its own side of the band. At the floor
/// `rank = rank(S_λ)` the kept set cannot shrink while the curvature weights stay
/// nonnegative: `H' ⪰ S'_λ` gives `σ_{rank(S_λ)}(H') ≥ σ_{rank(S_λ)}(S'_λ) > 0`,
/// and `rank(S_λ)` does not depend on ρ.
pub(crate) fn certify_identified_rank_locally_constant(
    eigenvalues: &[f64],
    penalty_rank: usize,
    motion: HessianSpectrumMotion,
) -> Result<IdentifiedRankCertificate, EstimationError> {
    let coefficients = eigenvalues.len();
    let rank = DenseSpectralOperator::identified_rank(eigenvalues, penalty_rank);
    let band = DenseSpectralOperator::rounding_band(eigenvalues);
    let spectral_radius = eigenvalues
        .iter()
        .fold(0.0_f64, |acc, value| acc.max(value.abs()));
    let mut descending = eigenvalues.to_vec();
    descending.sort_by(|left, right| right.total_cmp(left));
    let Some(&smallest_identified) = rank.checked_sub(1).and_then(|index| descending.get(index))
    else {
        return Err(EstimationError::RemlOptimizationFailed(
            "the penalized Hessian has no positive curvature, so no coefficient direction is \
             identified"
                .to_string(),
        ));
    };
    let largest_unidentified = descending.get(rank).copied();
    let HessianSpectrumMotion {
        lower_factor,
        upper_factor,
        additive,
        weights_stay_nonnegative,
    } = motion;
    let resolution = coefficients as f64 * f64::EPSILON;
    let reachable_band = (
        resolution * (lower_factor * spectral_radius - additive).max(0.0),
        resolution * (upper_factor * spectral_radius + additive),
    );
    let reachable_smallest_identified = lower_factor * smallest_identified - additive;
    let reachable_largest_unidentified =
        largest_unidentified.map(|sigma| upper_factor * sigma + additive);
    let kept_set_holds = (rank == penalty_rank && weights_stay_nonnegative)
        || (lower_factor > 0.0 && reachable_smallest_identified > reachable_band.1);
    let dropped_set_holds = match reachable_largest_unidentified {
        Some(sigma) => sigma <= reachable_band.0,
        None => true,
    };
    if kept_set_holds && dropped_set_holds {
        Ok(IdentifiedRankCertificate {
            rank,
            smallest_identified,
            largest_unidentified,
            band,
        })
    } else {
        Err(EstimationError::IdentifiedRankNotLocallyConstant {
            rank,
            coefficients,
            smallest_identified,
            largest_unidentified,
            band,
            reachable_smallest_identified,
            reachable_largest_unidentified,
            reachable_band,
        })
    }
}

/// Certify at finalization that the fitted Hessian's identified rank is constant
/// over the outer certificate's own Newton step (#2901 V22), and return that
/// certificate with the step's largest coordinate.
///
/// `hessian` is PIRLS's dense penalized Hessian in its transformed basis;
/// `hessian_rho` and `gradient` are the outer certificate's curvature and
/// gradient at ρ̂, and `railed` lists the coordinates it certified on a rail. The
/// curvature weights move through `β̂`: `ΔW_i = c_i·x_iᵀΔβ` with
/// `Δβ = Σ_k δρ_k·∂β̂/∂ρ_k` and `∂β̂/∂ρ_k = −H⁺λ_kS_kβ̂` over the identified
/// subspace the criterion priced, so row `i` moves by at most
/// `|c_i|·Σ_k |δρ_k|·|x_iᵀ∂β̂/∂ρ_k|`. Rows whose weight is not positive are
/// charged by the exact spectral norm of their weighted rank-one sum.
pub(crate) fn certify_fitted_identified_rank(
    pirls: &crate::pirls::PirlsResult,
    hessian: &Array2<f64>,
    lambdas: &Array1<f64>,
    design: &gam_linalg::matrix::DesignMatrix,
    hessian_rho: &Array2<f64>,
    gradient: &Array1<f64>,
    railed: &[usize],
) -> Result<(IdentifiedRankCertificate, f64), EstimationError> {
    let displacement = certificate_newton_displacement(hessian_rho, gradient, railed)?;
    let step_radius = displacement.iter().fold(0.0_f64, |acc, value| acc.max(*value));
    let penalty_rank = pirls.reparam_result.e_transformed.nrows();
    let mut symmetric = hessian.clone();
    gam_linalg::matrix::symmetrize_in_place(&mut symmetric);
    let (eigenvalues, eigenvectors) = symmetric
        .eigh(Side::Lower)
        .map_err(EstimationError::EigendecompositionFailed)?;
    let eigenvalues = eigenvalues.to_vec();
    let rows = design.nrows();
    let weight_motion = if pirls.solve_c_nontrivial && step_radius > 0.0 {
        let rank = DenseSpectralOperator::identified_rank(&eigenvalues, penalty_rank);
        let mut order: Vec<usize> = (0..eigenvalues.len()).collect();
        order.sort_by(|&left, &right| eigenvalues[right].total_cmp(&eigenvalues[left]));
        let beta: &Array1<f64> = pirls.beta_transformed.as_ref();
        let qs = &pirls.reparam_result.qs;
        let mut eta_motion = Array1::<f64>::zeros(rows);
        for ((penalty, &lambda), &step) in pirls
            .reparam_result
            .canonical_transformed
            .iter()
            .zip(lambdas.iter())
            .zip(displacement.iter())
        {
            if step == 0.0 {
                continue;
            }
            let range = penalty.col_range.clone();
            let root_beta = penalty.root.dot(&beta.slice(s![range.clone()]));
            let mut penalty_gradient = Array1::<f64>::zeros(beta.len());
            penalty_gradient
                .slice_mut(s![range])
                .assign(&penalty.root.t().dot(&root_beta));
            let mut response = Array1::<f64>::zeros(beta.len());
            for &column in order.iter().take(rank) {
                let direction = eigenvectors.column(column);
                response.scaled_add(
                    -lambda * direction.dot(&penalty_gradient) / eigenvalues[column],
                    &direction,
                );
            }
            let row_response = design.apply_view(qs.dot(&response).view());
            eta_motion.zip_mut_with(&row_response, |motion, value| {
                *motion += step * value.abs();
            });
        }
        pirls
            .solve_c_array
            .iter()
            .zip(eta_motion.iter())
            .map(|(&derivative, &motion)| derivative.abs() * motion)
            .collect()
    } else {
        Array1::<f64>::zeros(rows)
    };
    let motion = HessianSpectrumMotion::over_step(
        step_radius,
        pirls.finalweights.view(),
        weight_motion.view(),
        |mass| {
            let columns = design.ncols();
            let mut gram = Array2::<f64>::zeros((columns, columns));
            for (row, &value) in mass.iter().enumerate() {
                if value > 0.0 {
                    design
                        .syr_row_into(row, value, &mut gram)
                        .map_err(EstimationError::InvalidInput)?;
                }
            }
            gam_linalg::matrix::symmetrize_in_place(&mut gram);
            let spectrum = gram
                .eigh(Side::Lower)
                .map_err(EstimationError::EigendecompositionFailed)?
                .0;
            Ok(spectrum
                .iter()
                .fold(0.0_f64, |acc, value| acc.max(value.abs())))
        },
    )?;
    certify_identified_rank_locally_constant(&eigenvalues, penalty_rank, motion)
        .map(|certificate| (certificate, step_radius))
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

    /// The motion over a step for positive curvature weights that do not move.
    fn still_weights_motion(step_radius: f64) -> HessianSpectrumMotion {
        let weights = Array1::from_elem(4, 1.0);
        let weight_motion = Array1::<f64>::zeros(4);
        HessianSpectrumMotion::over_step(step_radius, weights.view(), weight_motion.view(), |mass| {
            Ok(mass.sum())
        })
        .unwrap()
    }

    /// A structural null beside well-resolved directions: the rank is the same
    /// everywhere a tenth of a log-unit reaches.
    #[test]
    fn a_resolved_spectrum_certifies_its_rank_over_the_step() {
        let certificate =
            certify_identified_rank_locally_constant(&[3.0, 1.0, 0.5, 0.0], 2, still_weights_motion(0.1))
                .unwrap();
        assert_eq!(certificate.rank, 3);
        assert_eq!(certificate.largest_unidentified, Some(0.0));
    }

    /// An unidentified data direction just under the band is certified at a
    /// zero step, and refused once the step can lift it over the band.
    #[test]
    fn an_unidentified_direction_straddling_the_band_refuses() {
        let band = 3.0 * f64::EPSILON;
        let spectrum = [1.0, 0.3, 0.6 * band];
        let pointwise =
            certify_identified_rank_locally_constant(&spectrum, 2, still_weights_motion(0.0))
                .unwrap();
        assert_eq!(pointwise.rank, 2);
        let refusal =
            certify_identified_rank_locally_constant(&spectrum, 2, still_weights_motion(0.5))
                .unwrap_err();
        assert!(
            matches!(
                refusal,
                EstimationError::IdentifiedRankNotLocallyConstant { rank: 2, .. }
            ),
            "{refusal}"
        );
    }

    /// An identified data direction just over the band, above the penalty-rank
    /// floor, is certified at a zero step and refused once the step can push it
    /// under the band.
    #[test]
    fn an_identified_direction_straddling_the_band_refuses() {
        let band = 3.0 * f64::EPSILON;
        let spectrum = [1.0, 1.5 * band, 0.0];
        let pointwise =
            certify_identified_rank_locally_constant(&spectrum, 1, still_weights_motion(0.0))
                .unwrap();
        assert_eq!(pointwise.rank, 2);
        let refusal =
            certify_identified_rank_locally_constant(&spectrum, 1, still_weights_motion(0.5))
                .unwrap_err();
        assert!(
            matches!(
                refusal,
                EstimationError::IdentifiedRankNotLocallyConstant { rank: 2, .. }
            ),
            "{refusal}"
        );
    }

    /// A penalty railed near λ = 10¹⁵ dominates the spectrum and the band, but a
    /// data direction it does not reach stays far above that band and a
    /// structural null stays at zero, so the rank is certified.
    #[test]
    fn a_railed_penalty_does_not_charge_the_directions_it_does_not_reach() {
        let certificate = certify_identified_rank_locally_constant(
            &[1.0e15, 5.0e14, 1.0e3, 0.0],
            2,
            still_weights_motion(0.05),
        )
        .unwrap();
        assert_eq!(certificate.rank, 3);
    }

    /// A coordinate certified on a rail is certified by its tail, so only the
    /// interior Newton displacement enters the step.
    #[test]
    fn a_railed_coordinate_moves_nothing_in_the_certificate_step() {
        let hessian_rho = array![[2.0, 0.0], [0.0, 1.0e-3]];
        let gradient = array![1.0e-4, 5.0e-2];
        let railed = certificate_newton_displacement(&hessian_rho, &gradient, &[1]).unwrap();
        assert!((railed[0] - 5.0e-5).abs() < 1e-18, "{railed}");
        assert_eq!(railed[1], 0.0);
        let interior = certificate_newton_displacement(&hessian_rho, &gradient, &[]).unwrap();
        assert!((interior[1] - 50.0).abs() < 1e-10, "{interior}");
    }

    /// A row whose curvature weight is negative is charged additively, by the
    /// norm its row mass returns, and voids the penalty-rank floor.
    #[test]
    fn a_negative_weight_row_is_charged_additively() {
        let weights = array![1.0, -0.5];
        let weight_motion = array![0.1, 0.2];
        let motion =
            HessianSpectrumMotion::over_step(0.0, weights.view(), weight_motion.view(), |mass| {
                Ok(mass.sum())
            })
            .unwrap();
        assert!((motion.upper_factor - 1.1).abs() < 1e-15);
        assert!((motion.lower_factor - 0.9).abs() < 1e-15);
        assert!((motion.additive - 0.25).abs() < 1e-15);
        assert!(!motion.weights_stay_nonnegative);
    }
}
