//! Anderson acceleration of a fixed-point map.
//!
//! For an iteration `x_{k+1} = G(x_k)` whose residual `f(x) = G(x) - x`
//! contracts linearly, the plain iteration needs `log(tol)/log(ρ)` steps. When
//! `ρ` is close to one — an alternating block minimisation on a coupled problem
//! typically lands at `ρ ≈ 0.95`–`0.99` — that is hundreds of steps spent on
//! the tail, and each one costs a full pass over the data.
//!
//! Anderson acceleration (Anderson, *JACM* 1965; Walker & Ni, *SINUM* 2011)
//! replaces the next iterate with the affine combination of the last `m + 1`
//! images whose residual combination is smallest:
//!
//! ```text
//! γ  = argmin_γ ‖ f_k − ΔF_k γ ‖₂ ,   ΔF_k = [f_{k-m+1}−f_{k-m}, …, f_k−f_{k-1}]
//! x⁺ = g_k − ΔG_k γ ,                 ΔG_k = [g_{k-m+1}−g_{k-m}, …, g_k−g_{k-1}]
//! ```
//!
//! It is a multisecant quasi-Newton step on `f`: with `m` stored differences it
//! matches the secant condition on the `m`-dimensional subspace they span, and
//! it needs nothing from the map beyond its images — no Jacobian, no
//! directional derivative, no extra evaluation.
//!
//! # What the caller owes, and why the residual is an argument
//!
//! [`AndersonAccelerator::propose`] takes `(x_k, f_k)` rather than `(x_k, g_k)`
//! and reconstructs `g_k = x_k + f_k`. That is deliberate: on a state with
//! PERIODIC coordinates the map's image is projected back to a principal branch,
//! so `g_k − x_k` read literally can be a whole period where the actual step was
//! infinitesimal. A caller on such a state supplies the WRAPPED difference as
//! the residual, and every quantity this module forms — `g_k`, `ΔG`, `ΔF` — is
//! then built from lifted, consistent values.
//!
//! The caller also owns the safeguard. A proposal is a candidate, not a step:
//! Anderson has no descent guarantee, and the honest contract is that the caller
//! evaluates its own merit function at the candidate and keeps the plain iterate
//! when the candidate does not improve on it (Walker & Ni §5; Toth & Kelley,
//! *SINUM* 2015, for the safeguarded convergence theory). [`Self::reset`] is
//! there for the rejection path.

use crate::LinalgError;
use crate::faer_ndarray::FaerEigh;
use faer::Side;
use ndarray::{Array1, Array2};
use std::collections::VecDeque;

/// Bounded-history Anderson accelerator over a fixed-point map.
#[derive(Debug, Clone)]
pub struct AndersonAccelerator {
    depth: usize,
    regularization: f64,
    dimension: Option<usize>,
    /// `(g_{k-1}, f_{k-1})` — the previous image and residual, kept so the next
    /// call can form one new difference column.
    previous: Option<(Vec<f64>, Vec<f64>)>,
    /// Difference columns, oldest first, at most `depth` of each.
    image_differences: VecDeque<Vec<f64>>,
    residual_differences: VecDeque<Vec<f64>>,
}

impl AndersonAccelerator {
    /// `depth` is the number of stored difference columns `m`; `regularization`
    /// is the Tikhonov weight on the least squares, RELATIVE to the mean
    /// diagonal of `ΔFᵀΔF`, so it is dimensionless and scale-free.
    ///
    /// The regularisation is not optional insurance. `ΔF` is rank-deficient by
    /// construction whenever the fixed-point map has an exactly flat direction —
    /// a gauge orbit, for instance — because consecutive residuals then differ
    /// by nothing along it. An unregularised normal equation is singular there,
    /// and the extrapolation it produces is unbounded along the very direction
    /// that does not matter.
    pub fn new(depth: usize, regularization: f64) -> Result<Self, LinalgError> {
        if depth == 0 {
            return Err(LinalgError::InvalidInput(
                "Anderson acceleration requires a positive history depth".to_string(),
            ));
        }
        if !(regularization.is_finite() && regularization >= 0.0) {
            return Err(LinalgError::InvalidInput(format!(
                "Anderson regularization must be finite and non-negative, got {regularization}"
            )));
        }
        Ok(Self {
            depth,
            regularization,
            dimension: None,
            previous: None,
            image_differences: VecDeque::with_capacity(depth),
            residual_differences: VecDeque::with_capacity(depth),
        })
    }

    /// Number of stored difference columns — the effective order of the current
    /// multisecant model.
    pub fn history_len(&self) -> usize {
        self.residual_differences.len()
    }

    /// Forget the history, keeping the configuration.
    ///
    /// The caller resets after rejecting a proposal: the rejected candidate is
    /// not the iterate the next residual will be measured from, so keeping
    /// differences taken across that break would fit a secant model to a
    /// trajectory that never happened.
    pub fn reset(&mut self) {
        self.previous = None;
        self.image_differences.clear();
        self.residual_differences.clear();
    }

    /// Offer the current iterate and its residual; return the accelerated
    /// candidate when one is available.
    ///
    /// `Ok(None)` on the first call (no difference column yet) and whenever the
    /// least squares carries no usable information — an all-zero `ΔF`, or a
    /// candidate that is not finite. Those are ordinary states of a converging
    /// iteration, not errors: the caller simply takes the plain step.
    pub fn propose(
        &mut self,
        iterate: &[f64],
        residual: &[f64],
    ) -> Result<Option<Vec<f64>>, LinalgError> {
        if iterate.len() != residual.len() {
            return Err(LinalgError::InvalidInput(format!(
                "Anderson iterate width {} != residual width {}",
                iterate.len(),
                residual.len()
            )));
        }
        if iterate.is_empty() {
            return Err(LinalgError::InvalidInput(
                "Anderson acceleration requires a non-empty state".to_string(),
            ));
        }
        match self.dimension {
            None => self.dimension = Some(iterate.len()),
            Some(dimension) if dimension != iterate.len() => {
                return Err(LinalgError::InvalidInput(format!(
                    "Anderson state width changed from {dimension} to {}",
                    iterate.len()
                )));
            }
            Some(_) => {}
        }
        if iterate.iter().chain(residual).any(|value| !value.is_finite()) {
            return Err(LinalgError::InvalidInput(
                "Anderson acceleration requires a finite iterate and residual".to_string(),
            ));
        }

        let image: Vec<f64> = iterate
            .iter()
            .zip(residual)
            .map(|(x, f)| x + f)
            .collect();
        if let Some((previous_image, previous_residual)) = self.previous.as_ref() {
            let image_difference: Vec<f64> = image
                .iter()
                .zip(previous_image)
                .map(|(new, old)| new - old)
                .collect();
            let residual_difference: Vec<f64> = residual
                .iter()
                .zip(previous_residual)
                .map(|(new, old)| new - old)
                .collect();
            if self.residual_differences.len() == self.depth {
                self.image_differences.pop_front();
                self.residual_differences.pop_front();
            }
            self.image_differences.push_back(image_difference);
            self.residual_differences.push_back(residual_difference);
        }
        self.previous = Some((image.clone(), residual.to_vec()));

        let order = self.residual_differences.len();
        if order == 0 {
            return Ok(None);
        }
        let Some(coefficients) = self.solve_multisecant(residual, order)? else {
            return Ok(None);
        };
        let mut candidate = image;
        for (column, weight) in self.image_differences.iter().zip(coefficients.iter()) {
            for (value, difference) in candidate.iter_mut().zip(column) {
                *value -= weight * difference;
            }
        }
        if candidate.iter().any(|value| !value.is_finite()) {
            return Ok(None);
        }
        Ok(Some(candidate))
    }

    /// `γ = (ΔFᵀΔF + λ‖·‖ I)⁻¹ ΔFᵀ f_k`, solved through the symmetric
    /// eigendecomposition of an `order × order` matrix (`order ≤ depth`, so this
    /// is a handful of flops next to one evaluation of the map).
    ///
    /// The eigendecomposition rather than a Cholesky because the whole point of
    /// the regularisation is that the un-shifted matrix may be singular, and an
    /// eigendecomposition reports HOW singular: modes at or below the shift are
    /// dropped rather than inverted, so a flat direction contributes nothing
    /// instead of contributing a huge coefficient that the shift merely bounds.
    fn solve_multisecant(
        &self,
        residual: &[f64],
        order: usize,
    ) -> Result<Option<Array1<f64>>, LinalgError> {
        let mut normal = Array2::<f64>::zeros((order, order));
        for (left, left_column) in self.residual_differences.iter().enumerate() {
            for (right, right_column) in self.residual_differences.iter().enumerate().skip(left) {
                let entry: f64 = left_column
                    .iter()
                    .zip(right_column)
                    .map(|(a, b)| a * b)
                    .sum();
                normal[[left, right]] = entry;
                normal[[right, left]] = entry;
            }
        }
        let trace: f64 = (0..order).map(|index| normal[[index, index]]).sum();
        if !(trace.is_finite() && trace > 0.0) {
            return Ok(None);
        }
        let shift = self.regularization * trace / order as f64;
        for index in 0..order {
            normal[[index, index]] += shift;
        }
        let rhs = Array1::from_iter(self.residual_differences.iter().map(|column| {
            column
                .iter()
                .zip(residual)
                .map(|(difference, value)| difference * value)
                .sum::<f64>()
        }));
        let (eigenvalues, eigenvectors) = normal.eigh(Side::Lower).map_err(|error| {
            LinalgError::InvalidInput(format!(
                "Anderson multisecant eigendecomposition failed: {error}"
            ))
        })?;
        let scale = eigenvalues
            .iter()
            .map(|value| value.abs())
            .fold(0.0_f64, f64::max);
        let floor = shift.max(f64::EPSILON * scale * order as f64);
        let projected = eigenvectors.t().dot(&rhs);
        let mut spectral = Array1::<f64>::zeros(order);
        for mode in 0..order {
            if eigenvalues[mode] > floor {
                spectral[mode] = projected[mode] / eigenvalues[mode];
            }
        }
        let coefficients = eigenvectors.dot(&spectral);
        if coefficients.iter().any(|value| !value.is_finite()) {
            return Ok(None);
        }
        Ok(Some(coefficients))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A linear contraction `G(x) = A x + b` with spectral radius 0.975 — the
    /// rate measured on the support-sparse inner fixed point (#2575). Anderson
    /// with a history at least the problem's dimension is EXACT on an affine
    /// map after that many steps, so this pins both the algebra and the
    /// convention (`propose` takes the residual, not the image).
    #[test]
    fn an_affine_contraction_is_solved_in_dimension_steps() {
        let a = [[0.975_f64, 0.20, 0.0], [0.0, 0.90, 0.15], [0.10, 0.0, 0.80]];
        let b = [1.0_f64, -2.0, 0.5];
        let map = |x: &[f64]| -> Vec<f64> {
            (0..3)
                .map(|row| (0..3).map(|col| a[row][col] * x[col]).sum::<f64>() + b[row])
                .collect()
        };
        // Fixed point by plain iteration, run long enough to be exact to 1e-14.
        let mut reference = vec![0.0_f64; 3];
        for _ in 0..20_000 {
            reference = map(&reference);
        }

        let mut accelerator = AndersonAccelerator::new(3, 1.0e-12).expect("accelerator");
        let mut x = vec![0.0_f64; 3];
        let mut accelerated_error = f64::INFINITY;
        for _ in 0..6 {
            let g = map(&x);
            let residual: Vec<f64> = g.iter().zip(&x).map(|(g, x)| g - x).collect();
            x = match accelerator.propose(&x, &residual).expect("propose") {
                Some(candidate) => candidate,
                None => g,
            };
            accelerated_error = x
                .iter()
                .zip(&reference)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max);
        }
        assert!(
            accelerated_error < 1.0e-10,
            "six accelerated steps must solve a 3-dimensional affine map; error {accelerated_error:.3e}"
        );

        // The plain iteration at the same budget is nowhere near.
        let mut plain = vec![0.0_f64; 3];
        for _ in 0..6 {
            plain = map(&plain);
        }
        let plain_error = plain
            .iter()
            .zip(&reference)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            plain_error > 1.0e3 * accelerated_error,
            "the comparison is only meaningful if the plain rate is genuinely slow: \
             plain {plain_error:.3e} vs accelerated {accelerated_error:.3e}"
        );
    }

    /// An exactly flat direction — a gauge orbit — makes `ΔF` rank-deficient.
    /// The accelerator must still produce a finite, bounded proposal, and must
    /// not extrapolate along the flat direction just because the least squares
    /// cannot see it.
    #[test]
    fn a_flat_gauge_direction_does_not_blow_up_the_extrapolation() {
        // Coordinate 2 is invariant under the map and carries no residual.
        let map = |x: &[f64]| -> Vec<f64> { vec![0.98 * x[0] + 1.0, 0.95 * x[1] - 0.5, x[2]] };
        let mut accelerator = AndersonAccelerator::new(4, 1.0e-8).expect("accelerator");
        let mut x = vec![0.0_f64, 0.0, 7.0];
        for _ in 0..8 {
            let g = map(&x);
            let residual: Vec<f64> = g.iter().zip(&x).map(|(g, x)| g - x).collect();
            x = match accelerator.propose(&x, &residual).expect("propose") {
                Some(candidate) => candidate,
                None => g,
            };
            assert!(x.iter().all(|value| value.is_finite()), "{x:?}");
            assert_eq!(x[2], 7.0, "the flat coordinate must not move");
        }
        assert!((x[0] - 50.0).abs() < 1.0e-8, "x0 -> 1/(1-0.98) = 50");
        assert!((x[1] + 10.0).abs() < 1.0e-8, "x1 -> -0.5/(1-0.95) = -10");
    }

    #[test]
    fn the_configuration_and_the_state_width_are_validated() {
        assert!(AndersonAccelerator::new(0, 0.0).is_err());
        assert!(AndersonAccelerator::new(2, -1.0).is_err());
        assert!(AndersonAccelerator::new(2, f64::NAN).is_err());
        let mut accelerator = AndersonAccelerator::new(2, 1.0e-10).expect("accelerator");
        assert!(accelerator.propose(&[], &[]).is_err());
        assert!(accelerator.propose(&[1.0], &[1.0, 2.0]).is_err());
        assert!(accelerator.propose(&[1.0, 2.0], &[0.1, 0.2]).is_ok());
        assert!(
            accelerator.propose(&[1.0], &[0.1]).is_err(),
            "a state that changes width is a caller error, not a silent restart"
        );
        assert!(accelerator.propose(&[1.0, f64::NAN], &[0.1, 0.2]).is_err());
    }

    /// The first call has no difference column, so there is nothing to
    /// extrapolate from; `reset` returns to that state.
    #[test]
    fn the_history_starts_and_resets_empty() {
        let mut accelerator = AndersonAccelerator::new(2, 1.0e-10).expect("accelerator");
        assert_eq!(accelerator.history_len(), 0);
        assert!(
            accelerator
                .propose(&[1.0, 2.0], &[0.1, 0.2])
                .expect("propose")
                .is_none()
        );
        assert!(
            accelerator
                .propose(&[1.1, 2.2], &[0.05, 0.1])
                .expect("propose")
                .is_some()
        );
        assert_eq!(accelerator.history_len(), 1);
        accelerator.reset();
        assert_eq!(accelerator.history_len(), 0);
        assert!(
            accelerator
                .propose(&[1.15, 2.3], &[0.02, 0.05])
                .expect("propose")
                .is_none()
        );
    }
}
