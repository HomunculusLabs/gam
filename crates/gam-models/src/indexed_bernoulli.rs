//! Indexed multi-output Bernoulli GAM with joint LAML smoothing selection.
//!
//! Each output owns one coefficient block and one linear predictor, while the
//! response lives on an explicit subset of the Cartesian `(row, output)` grid.
//! Likelihood blocks are conditionally independent, so coefficient solves stay
//! block-separable. Penalty components carrying the same precision label share
//! one outer smoothing coordinate; the existing custom-family engine therefore
//! maximizes one joint Laplace marginal likelihood without constructing a dense
//! cross-output coefficient Hessian.

use crate::custom_family::{
    BlockWorkingSet, BlockwiseFitOptions, CustomFamily, CustomFamilyError, FamilyEvaluation,
    ParameterBlockSpec, ParameterBlockState, fit_custom_family,
};
use gam_model_kernels::bernoulli_link::{
    bernoulli_natural_jet, bernoulli_natural_observation,
};
use gam_problem::{EstimationError, OwnedSeparableCellMeasure};
use gam_solve::model_types::UnifiedFitResult;
use gam_spec::{InverseLink, StandardLink};
use ndarray::{Array1, Array2, ArrayView2};

/// Owned indexed Bernoulli response family.
///
/// Structural activity is model geometry: inactive cells are never evaluated,
/// so they may contain a non-finite placeholder in the dense response carrier.
/// Active cells, including active cells with numerical weight zero, must contain
/// a finite binomial proportion in `[0, 1]`.
#[derive(Clone, Debug)]
pub struct IndexedBernoulliFamily {
    y: Array2<f64>,
    measure: OwnedSeparableCellMeasure,
    link: InverseLink,
}

impl IndexedBernoulliFamily {
    pub fn new(
        y: ArrayView2<'_, f64>,
        measure: OwnedSeparableCellMeasure,
        link: InverseLink,
    ) -> Result<Self, EstimationError> {
        let (n_rows, n_outputs) = y.dim();
        if n_rows == 0 || n_outputs == 0 {
            return Err(EstimationError::InvalidInput(format!(
                "indexed Bernoulli response must be non-empty, got {n_rows}x{n_outputs}"
            )));
        }
        if (measure.n_rows(), measure.n_outputs()) != (n_rows, n_outputs) {
            return Err(EstimationError::InvalidInput(format!(
                "indexed Bernoulli measure geometry ({}, {}) does not match response geometry ({n_rows}, {n_outputs})",
                measure.n_rows(),
                measure.n_outputs(),
            )));
        }
        // Validate the bounded-link contract once at construction. Row kernels
        // still validate every realized eta because parameterized links can
        // leave their representable domain away from zero.
        bernoulli_natural_jet(0, 0.0, &link)?;
        for ((row, output), &response) in y.indexed_iter() {
            if measure.is_active(row, output)
                && !(response.is_finite() && (0.0..=1.0).contains(&response))
            {
                return Err(EstimationError::InvalidInput(format!(
                    "indexed Bernoulli response[{row},{output}] must be finite and in [0,1] because the cell is structurally active, got {response}"
                )));
            }
        }
        Ok(Self {
            y: y.to_owned(),
            measure,
            link,
        })
    }

    pub fn n_rows(&self) -> usize {
        self.y.nrows()
    }

    pub fn n_outputs(&self) -> usize {
        self.y.ncols()
    }

    pub fn link(&self) -> &InverseLink {
        &self.link
    }

    fn validate_states(&self, block_states: &[ParameterBlockState]) -> Result<(), String> {
        if block_states.len() != self.n_outputs() {
            return Err(format!(
                "indexed Bernoulli family requires one parameter block per output: got {}, expected {}",
                block_states.len(),
                self.n_outputs(),
            ));
        }
        for (output, state) in block_states.iter().enumerate() {
            if state.eta.len() != self.n_rows() {
                return Err(format!(
                    "indexed Bernoulli output {output} predictor has length {}, expected {}",
                    state.eta.len(),
                    self.n_rows(),
                ));
            }
        }
        Ok(())
    }

    fn evaluate_output(
        &self,
        output: usize,
        eta: &Array1<f64>,
        derivatives: bool,
    ) -> Result<(f64, Option<(Array1<f64>, Array1<f64>)>), String> {
        let mut log_likelihood = 0.0;
        let mut score = derivatives.then(|| Array1::<f64>::zeros(self.n_rows()));
        let mut curvature = derivatives.then(|| Array1::<f64>::zeros(self.n_rows()));
        for row in 0..self.n_rows() {
            let Some(weight) = self.measure.active_weight(row, output) else {
                continue;
            };
            if weight == 0.0 {
                continue;
            }
            let observation = bernoulli_natural_observation(
                row,
                self.y[[row, output]],
                eta[row],
                &self.link,
            )
            .map_err(|error| error.to_string())?;
            log_likelihood += weight * observation.log_likelihood;
            if let (Some(score), Some(curvature)) = (&mut score, &mut curvature) {
                score[row] = weight * observation.score;
                curvature[row] = weight * observation.negative_hessian;
            }
        }
        if !log_likelihood.is_finite() {
            return Err(format!(
                "indexed Bernoulli output {output} produced non-finite log likelihood {log_likelihood}"
            ));
        }
        Ok((log_likelihood, score.zip(curvature)))
    }
}

impl CustomFamily for IndexedBernoulliFamily {
    fn evaluate(&self, block_states: &[ParameterBlockState]) -> Result<FamilyEvaluation, String> {
        self.validate_states(block_states)?;
        let mut log_likelihood = 0.0;
        let mut blockworking_sets = Vec::with_capacity(self.n_outputs());
        for (output, state) in block_states.iter().enumerate() {
            let (value, derivatives) = self.evaluate_output(output, &state.eta, true)?;
            log_likelihood += value;
            let (score, curvature) = derivatives.ok_or_else(|| {
                format!("indexed Bernoulli output {output} omitted requested derivatives")
            })?;
            blockworking_sets.push(BlockWorkingSet::natural_diagonal_checked(score, curvature)?);
        }
        Ok(FamilyEvaluation {
            log_likelihood,
            blockworking_sets,
        })
    }

    fn log_likelihood_only(&self, block_states: &[ParameterBlockState]) -> Result<f64, String> {
        self.validate_states(block_states)?;
        block_states
            .iter()
            .enumerate()
            .try_fold(0.0, |total, (output, state)| {
                self.evaluate_output(output, &state.eta, false)
                    .map(|(value, _)| total + value)
            })
    }

    fn classical_deviance(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<Option<f64>, String> {
        let fitted = self.log_likelihood_only(block_states)?;
        let mut saturated = 0.0;
        for ((row, output), &response) in self.y.indexed_iter() {
            let Some(weight) = self.measure.active_weight(row, output) else {
                continue;
            };
            if weight == 0.0 {
                continue;
            }
            let unit = if response == 0.0 || response == 1.0 {
                0.0
            } else {
                response * response.ln() + (1.0 - response) * (-response).ln_1p()
            };
            saturated += weight * unit;
        }
        Ok(Some(2.0 * (saturated - fitted)))
    }

    fn likelihood_blocks_uncoupled(&self) -> bool {
        true
    }

    fn exact_newton_joint_hessian_beta_dependent(&self) -> bool {
        true
    }

    fn inner_coefficient_objective_is_globally_convex(&self) -> bool {
        matches!(
            self.link,
            InverseLink::Standard(
                StandardLink::Logit
                    | StandardLink::Probit
                    | StandardLink::CLogLog
                    | StandardLink::LogLog
            )
        )
    }

    fn output_channel_assignment(&self, specs: &[ParameterBlockSpec]) -> Option<Vec<usize>> {
        Some((0..specs.len()).collect())
    }

    fn diagonalworking_weights_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        block_index: usize,
        d_eta: &Array1<f64>,
    ) -> Result<Option<Array1<f64>>, String> {
        self.validate_states(block_states)?;
        let state = block_states.get(block_index).ok_or_else(|| {
            format!(
                "indexed Bernoulli curvature derivative block {block_index} is outside 0..{}",
                block_states.len()
            )
        })?;
        if d_eta.len() != self.n_rows() {
            return Err(format!(
                "indexed Bernoulli curvature direction has length {}, expected {}",
                d_eta.len(),
                self.n_rows(),
            ));
        }
        let mut derivative = Array1::<f64>::zeros(self.n_rows());
        for row in 0..self.n_rows() {
            let Some(weight) = self.measure.active_weight(row, block_index) else {
                continue;
            };
            if weight == 0.0 {
                continue;
            }
            let observation = bernoulli_natural_observation(
                row,
                self.y[[row, block_index]],
                state.eta[row],
                &self.link,
            )
            .map_err(|error| error.to_string())?;
            derivative[row] = weight * observation.negative_hessian_derivative * d_eta[row];
        }
        Ok(Some(derivative))
    }
}

/// Fit an indexed multi-output Bernoulli GAM and select every unfixed,
/// potentially shared precision by the canonical LAML outer optimizer.
pub fn fit_indexed_bernoulli_laml(
    family: &IndexedBernoulliFamily,
    specs: &[ParameterBlockSpec],
    options: &BlockwiseFitOptions,
) -> Result<UnifiedFitResult, CustomFamilyError> {
    if specs.len() != family.n_outputs() {
        return Err(CustomFamilyError::DimensionMismatch {
            reason: format!(
                "indexed Bernoulli fit requires one parameter block per output: got {}, expected {}",
                specs.len(),
                family.n_outputs(),
            ),
        });
    }
    fit_custom_family(family, specs, options)
}

#[cfg(test)]
mod tests {
    use super::*;
    use gam_problem::{IndexedCellSet, LikelihoodWeights, SeparableCellMeasure, StructuralCells};

    fn states(etas: &[Array1<f64>]) -> Vec<ParameterBlockState> {
        etas.iter()
            .map(|eta| ParameterBlockState {
                beta: Array1::zeros(0),
                eta: eta.clone(),
            })
            .collect()
    }

    #[test]
    fn inactive_nan_and_active_zero_weight_are_not_likelihood_rows() {
        let active = IndexedCellSet::from_cells(2, 2, vec![(0, 0), (0, 1), (1, 1)])
            .expect("valid activity set");
        let weights = ndarray::array![[1.0, 0.0], [9.0, 2.0]];
        let borrowed = SeparableCellMeasure::new(
            StructuralCells::Only(&active),
            LikelihoodWeights::ByCell(weights.view()),
        );
        let measure = borrowed.to_owned(2, 2).expect("owned measure");
        let y = ndarray::array![[1.0, 0.0], [f64::NAN, 0.0]];
        let family = IndexedBernoulliFamily::new(
            y.view(),
            measure,
            InverseLink::Standard(StandardLink::Logit),
        )
        .expect("indexed family");
        let point = states(&[ndarray::array![0.0, 700.0], ndarray::array![-700.0, 0.0]]);
        let evaluation = family.evaluate(&point).expect("finite indexed evaluation");

        assert_eq!(evaluation.log_likelihood, -3.0 * 2.0_f64.ln());
        let BlockWorkingSet::NaturalDiagonal {
            score,
            observed_curvature,
        } = &evaluation.blockworking_sets[0]
        else {
            panic!("indexed Bernoulli must expose natural diagonal geometry");
        };
        assert_eq!(score[1], 0.0);
        assert_eq!(observed_curvature[1], 0.0);
    }

    #[test]
    fn cloglog_tail_preserves_nonzero_score_at_zero_observed_curvature() {
        let family = IndexedBernoulliFamily::new(
            ndarray::array![[1.0]].view(),
            OwnedSeparableCellMeasure::uniform(1, 1),
            InverseLink::Standard(StandardLink::CLogLog),
        )
        .expect("cloglog family");
        let evaluation = family
            .evaluate(&states(&[ndarray::array![-1_000.0]]))
            .expect("representable natural tail");
        let BlockWorkingSet::NaturalDiagonal {
            score,
            observed_curvature,
        } = &evaluation.blockworking_sets[0]
        else {
            panic!("cloglog must expose natural diagonal geometry");
        };
        assert_eq!(score[0], 1.0);
        assert_eq!(observed_curvature[0], 0.0);
        assert_eq!(evaluation.log_likelihood, -1_000.0);
    }
}
