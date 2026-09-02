//! Shared custom-family engine for indexed separable natural likelihoods.
//!
//! A concrete program owns row data and evaluates one stable scalar row.  This
//! module owns the Cartesian response geometry, structural activity, numerical
//! likelihood weights, block validation, and the complete curvature tower used
//! by the inner Newton and outer LAML calculations.

use crate::custom_family::{
    BlockWorkingSet, CustomFamily, FamilyEvaluation, ParameterBlockSpec, ParameterBlockState,
};
use gam_model_kernels::natural_observation::NaturalDiagonalObservation;
use gam_problem::{EstimationError, OwnedSeparableCellMeasure};
use ndarray::Array1;

/// Stable scalar row program over an indexed `(row, output)` response grid.
pub trait IndexedNaturalDiagonalProgram {
    fn family_name(&self) -> &'static str;
    fn n_rows(&self) -> usize;
    fn n_outputs(&self) -> usize;
    fn measure(&self) -> &OwnedSeparableCellMeasure;

    fn observation(
        &self,
        row: usize,
        output: usize,
        eta: f64,
    ) -> Result<NaturalDiagonalObservation, EstimationError>;

    /// Return the classical deviance at a fitted log likelihood, or `None`
    /// when the family has no finite saturated row law.
    fn classical_deviance(&self, _fitted_log_likelihood: f64) -> Result<Option<f64>, String> {
        Ok(None)
    }

    /// Mathematical certificate for the complete unpenalized coefficient
    /// objective.  Penalties are positive semidefinite and preserve convexity.
    fn coefficient_objective_is_globally_convex(&self) -> bool {
        false
    }
}
/// One block-separable parameter vector per output, driven by a shared scalar
/// row program.
#[derive(Clone, Debug)]
pub struct IndexedNaturalDiagonalFamily<P> {
    program: P,
}

impl<P: IndexedNaturalDiagonalProgram> IndexedNaturalDiagonalFamily<P> {
    pub fn from_program(program: P) -> Result<Self, EstimationError> {
        let (n_rows, n_outputs) = (program.n_rows(), program.n_outputs());
        if n_rows == 0 || n_outputs == 0 {
            return Err(EstimationError::InvalidInput(format!(
                "{} response must be non-empty, got {n_rows}x{n_outputs}",
                program.family_name(),
            )));
        }
        let measure = program.measure();
        if (measure.n_rows(), measure.n_outputs()) != (n_rows, n_outputs) {
            return Err(EstimationError::InvalidInput(format!(
                "{} measure geometry ({}, {}) does not match response geometry ({n_rows}, {n_outputs})",
                program.family_name(),
                measure.n_rows(),
                measure.n_outputs(),
            )));
        }
        Ok(Self { program })
    }

    pub fn program(&self) -> &P {
        &self.program
    }

    pub fn n_rows(&self) -> usize {
        self.program.n_rows()
    }

    pub fn n_outputs(&self) -> usize {
        self.program.n_outputs()
    }

    fn validate_states(&self, block_states: &[ParameterBlockState]) -> Result<(), String> {
        if block_states.len() != self.n_outputs() {
            return Err(format!(
                "{} family requires one parameter block per output: got {}, expected {}",
                self.program.family_name(),
                block_states.len(),
                self.n_outputs(),
            ));
        }
        for (output, state) in block_states.iter().enumerate() {
            if state.eta.len() != self.n_rows() {
                return Err(format!(
                    "{} output {output} predictor has length {}, expected {}",
                    self.program.family_name(),
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
            let Some(weight) = self.program.measure().active_weight(row, output) else {
                continue;
            };
            if weight == 0.0 {
                continue;
            }
            let observation = self
                .program
                .observation(row, output, eta[row])
                .map_err(|error| error.to_string())?;
            log_likelihood += weight * observation.log_likelihood;
            if let (Some(score), Some(curvature)) = (&mut score, &mut curvature) {
                score[row] = weight * observation.score;
                curvature[row] = weight * observation.negative_hessian;
            }
        }
        if !log_likelihood.is_finite() {
            return Err(format!(
                "{} output {output} produced non-finite log likelihood {log_likelihood}",
                self.program.family_name(),
            ));
        }
        Ok((log_likelihood, score.zip(curvature)))
    }
}

impl<P: IndexedNaturalDiagonalProgram> CustomFamily for IndexedNaturalDiagonalFamily<P> {
    fn evaluate(&self, block_states: &[ParameterBlockState]) -> Result<FamilyEvaluation, String> {
        self.validate_states(block_states)?;
        let mut log_likelihood = 0.0;
        let mut blockworking_sets = Vec::with_capacity(self.n_outputs());
        for (output, state) in block_states.iter().enumerate() {
            let (value, derivatives) = self.evaluate_output(output, &state.eta, true)?;
            log_likelihood += value;
            let (score, curvature) = derivatives.ok_or_else(|| {
                format!(
                    "{} output {output} omitted requested derivatives",
                    self.program.family_name(),
                )
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
        self.program.classical_deviance(fitted)
    }

    fn likelihood_blocks_uncoupled(&self) -> bool {
        true
    }

    fn exact_newton_joint_hessian_beta_dependent(&self) -> bool {
        true
    }

    fn inner_coefficient_objective_is_globally_convex(&self) -> bool {
        self.program.coefficient_objective_is_globally_convex()
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
                "{} curvature derivative block {block_index} is outside 0..{}",
                self.program.family_name(),
                block_states.len(),
            )
        })?;
        if d_eta.len() != self.n_rows() {
            return Err(format!(
                "{} curvature direction has length {}, expected {}",
                self.program.family_name(),
                d_eta.len(),
                self.n_rows(),
            ));
        }
        let mut derivative = Array1::<f64>::zeros(self.n_rows());
        for row in 0..self.n_rows() {
            let Some(weight) = self.program.measure().active_weight(row, block_index) else {
                continue;
            };
            if weight == 0.0 {
                continue;
            }
            let observation = self
                .program
                .observation(row, block_index, state.eta[row])
                .map_err(|error| error.to_string())?;
            derivative[row] = weight * observation.negative_hessian_derivative * d_eta[row];
        }
        Ok(Some(derivative))
    }

    fn diagonalworking_weights_second_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        block_index: usize,
        d_eta_u: &Array1<f64>,
        d_eta_v: &Array1<f64>,
    ) -> Result<Option<Array1<f64>>, String> {
        self.validate_states(block_states)?;
        let state = block_states.get(block_index).ok_or_else(|| {
            format!(
                "{} second curvature derivative block {block_index} is outside 0..{}",
                self.program.family_name(),
                block_states.len(),
            )
        })?;
        for (name, direction) in [("u", d_eta_u), ("v", d_eta_v)] {
            if direction.len() != self.n_rows() {
                return Err(format!(
                    "{} second curvature direction {name} has length {}, expected {}",
                    self.program.family_name(),
                    direction.len(),
                    self.n_rows(),
                ));
            }
        }
        let mut derivative = Array1::<f64>::zeros(self.n_rows());
        for row in 0..self.n_rows() {
            let Some(weight) = self.program.measure().active_weight(row, block_index) else {
                continue;
            };
            if weight == 0.0 {
                continue;
            }
            let observation = self
                .program
                .observation(row, block_index, state.eta[row])
                .map_err(|error| error.to_string())?;
            derivative[row] = weight
                * observation.negative_hessian_second_derivative
                * d_eta_u[row]
                * d_eta_v[row];
        }
        Ok(Some(derivative))
    }
}
