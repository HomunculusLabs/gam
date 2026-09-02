//! Replication of one GAM coefficient block across a named output axis.
//!
//! Multi-response likelihoods commonly reuse one design and penalty geometry
//! for every endpoint. This compiler owns that repetition once: output names,
//! `(row, output)` offsets, coefficient seeds, and precision-sharing semantics
//! are validated before they become low-level [`ParameterBlockSpec`] values.

use crate::custom_family::{ParameterBlockSpec, PenaltyMatrix};
use gam_linalg::matrix::DesignMatrix;
use gam_problem::validate_log_strength;
use ndarray::{Array1, Array2};
use std::collections::BTreeSet;
use thiserror::Error;

/// Invalid replicated-output block geometry.
#[derive(Clone, Debug, Error, PartialEq)]
#[error("{reason}")]
pub struct OutputAxisError {
    reason: String,
}

impl OutputAxisError {
    fn new(reason: impl Into<String>) -> Self {
        Self {
            reason: reason.into(),
        }
    }

    pub fn reason(&self) -> &str {
        &self.reason
    }
}

/// How one penalty template maps onto output-specific physical penalties.
#[derive(Clone, Debug, PartialEq)]
pub enum OutputAxisPrecision {
    /// Each output receives an independent LAML smoothing coordinate.
    Independent,
    /// Every output shares the named LAML smoothing coordinate.
    Shared(String),
    /// Every output uses the same fixed physical log-precision.
    Fixed(f64),
}

/// One validated penalty template replicated across output blocks.
#[derive(Clone, Debug)]
pub struct OutputAxisPenalty {
    matrix: PenaltyMatrix,
    nullspace_dim: usize,
    initial_log_precision: f64,
    precision: OutputAxisPrecision,
}

impl OutputAxisPenalty {
    pub fn independent(
        matrix: PenaltyMatrix,
        nullspace_dim: usize,
        initial_log_precision: f64,
    ) -> Result<Self, OutputAxisError> {
        Self::new(
            matrix,
            nullspace_dim,
            initial_log_precision,
            OutputAxisPrecision::Independent,
        )
    }

    pub fn shared(
        matrix: PenaltyMatrix,
        nullspace_dim: usize,
        initial_log_precision: f64,
        label: impl Into<String>,
    ) -> Result<Self, OutputAxisError> {
        let label = label.into();
        if label.trim().is_empty() {
            return Err(OutputAxisError::new(
                "shared output-axis precision label must not be empty",
            ));
        }
        Self::new(
            matrix,
            nullspace_dim,
            initial_log_precision,
            OutputAxisPrecision::Shared(label),
        )
    }

    pub fn fixed(
        matrix: PenaltyMatrix,
        nullspace_dim: usize,
        log_precision: f64,
    ) -> Result<Self, OutputAxisError> {
        Self::new(
            matrix,
            nullspace_dim,
            log_precision,
            OutputAxisPrecision::Fixed(log_precision),
        )
    }

    fn new(
        matrix: PenaltyMatrix,
        nullspace_dim: usize,
        initial_log_precision: f64,
        precision: OutputAxisPrecision,
    ) -> Result<Self, OutputAxisError> {
        if matrix.precision_label().is_some() || matrix.fixed_log_lambda().is_some() {
            return Err(OutputAxisError::new(
                "output-axis penalty templates must be unlabeled and unfixed; declare replication precision through OutputAxisPenalty",
            ));
        }
        let dimension = matrix.dim();
        matrix.validate(dimension).map_err(|reason| {
            OutputAxisError::new(format!("invalid output-axis penalty template: {reason}"))
        })?;
        if nullspace_dim > dimension {
            return Err(OutputAxisError::new(format!(
                "output-axis penalty nullspace dimension {nullspace_dim} exceeds coefficient dimension {dimension}"
            )));
        }
        validate_log_strength(initial_log_precision).map_err(|error| {
            OutputAxisError::new(format!(
                "output-axis penalty initial log-precision: {error}"
            ))
        })?;
        Ok(Self {
            matrix,
            nullspace_dim,
            initial_log_precision,
            precision,
        })
    }

    pub fn precision(&self) -> &OutputAxisPrecision {
        &self.precision
    }

    fn realized_matrix(&self) -> PenaltyMatrix {
        match &self.precision {
            OutputAxisPrecision::Independent => self.matrix.clone(),
            OutputAxisPrecision::Shared(label) => {
                self.matrix.clone().with_precision_label(label.clone())
            }
            OutputAxisPrecision::Fixed(log_precision) => self
                .matrix
                .clone()
                .with_fixed_log_lambda(*log_precision),
        }
    }
}

/// A common design and penalty layout replicated over named outputs.
#[derive(Clone, Debug)]
pub struct OutputBlockAxis {
    output_names: Vec<String>,
    design: DesignMatrix,
    offsets: Array2<f64>,
    penalties: Vec<OutputAxisPenalty>,
    initial_coefficients: Option<Array2<f64>>,
    gauge_priority: u8,
}

impl OutputBlockAxis {
    /// Create a zero-offset axis. Output names are coefficient-block identities
    /// and therefore must be nonempty and unique.
    pub fn new(
        output_names: Vec<String>,
        design: DesignMatrix,
    ) -> Result<Self, OutputAxisError> {
        if output_names.is_empty() {
            return Err(OutputAxisError::new(
                "output block axis must contain at least one output",
            ));
        }
        let mut unique = BTreeSet::new();
        for (index, name) in output_names.iter().enumerate() {
            if name.trim().is_empty() {
                return Err(OutputAxisError::new(format!(
                    "output name at index {index} must not be empty"
                )));
            }
            if !unique.insert(name.as_str()) {
                return Err(OutputAxisError::new(format!(
                    "output block axis contains duplicate output name '{name}'"
                )));
            }
        }
        if design.nrows() == 0 || design.ncols() == 0 {
            return Err(OutputAxisError::new(format!(
                "output block axis design must be nonempty, got {}x{}",
                design.nrows(),
                design.ncols(),
            )));
        }
        let offsets = Array2::zeros((design.nrows(), output_names.len()));
        Ok(Self {
            output_names,
            design,
            offsets,
            penalties: Vec::new(),
            initial_coefficients: None,
            gauge_priority: 100,
        })
    }

    pub fn n_outputs(&self) -> usize {
        self.output_names.len()
    }

    pub fn coefficient_width(&self) -> usize {
        self.design.ncols()
    }

    /// Set known predictor offsets in `(row, output)` layout.
    pub fn with_offsets(mut self, offsets: Array2<f64>) -> Result<Self, OutputAxisError> {
        let expected = (self.design.nrows(), self.n_outputs());
        if offsets.dim() != expected {
            return Err(OutputAxisError::new(format!(
                "output-axis offsets have shape {:?}, expected {expected:?}",
                offsets.dim(),
            )));
        }
        if let Some(((row, output), value)) = offsets
            .indexed_iter()
            .find(|(_, value)| !value.is_finite())
        {
            return Err(OutputAxisError::new(format!(
                "output-axis offset[{row},{output}] must be finite, got {value}"
            )));
        }
        self.offsets = offsets;
        Ok(self)
    }

    /// Set coefficient seeds in `(coefficient, output)` layout.
    pub fn with_initial_coefficients(
        mut self,
        coefficients: Array2<f64>,
    ) -> Result<Self, OutputAxisError> {
        let expected = (self.coefficient_width(), self.n_outputs());
        if coefficients.dim() != expected {
            return Err(OutputAxisError::new(format!(
                "output-axis initial coefficients have shape {:?}, expected {expected:?}",
                coefficients.dim(),
            )));
        }
        if let Some(((coefficient, output), value)) = coefficients
            .indexed_iter()
            .find(|(_, value)| !value.is_finite())
        {
            return Err(OutputAxisError::new(format!(
                "output-axis initial coefficient[{coefficient},{output}] must be finite, got {value}"
            )));
        }
        self.initial_coefficients = Some(coefficients);
        Ok(self)
    }

    pub fn with_penalty(mut self, penalty: OutputAxisPenalty) -> Result<Self, OutputAxisError> {
        if penalty.matrix.dim() != self.coefficient_width() {
            return Err(OutputAxisError::new(format!(
                "output-axis penalty dimension {} does not match coefficient width {}",
                penalty.matrix.dim(),
                self.coefficient_width(),
            )));
        }
        self.penalties.push(penalty);
        Ok(self)
    }

    pub fn with_gauge_priority(mut self, gauge_priority: u8) -> Self {
        self.gauge_priority = gauge_priority;
        self
    }

    /// Compile one canonical single-channel parameter block per output.
    pub fn parameter_blocks(&self) -> Vec<ParameterBlockSpec> {
        (0..self.n_outputs())
            .map(|output| ParameterBlockSpec {
                name: self.output_names[output].clone(),
                design: self.design.clone(),
                offset: self.offsets.column(output).to_owned(),
                penalties: self
                    .penalties
                    .iter()
                    .map(OutputAxisPenalty::realized_matrix)
                    .collect(),
                nullspace_dims: self
                    .penalties
                    .iter()
                    .map(|penalty| penalty.nullspace_dim)
                    .collect(),
                initial_log_lambdas: Array1::from_iter(
                    self.penalties
                        .iter()
                        .map(|penalty| penalty.initial_log_precision),
                ),
                initial_beta: self
                    .initial_coefficients
                    .as_ref()
                    .map(|coefficients| coefficients.column(output).to_owned()),
                gauge_priority: self.gauge_priority,
                jacobian_callback: None,
                stacked_design: None,
                stacked_offset: None,
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn compiles_named_outputs_with_explicit_precision_semantics() {
        let design = DesignMatrix::from(array![[1.0, -1.0], [1.0, 1.0]]);
        let shared = OutputAxisPenalty::shared(
            PenaltyMatrix::Diagonal(array![0.0, 1.0]),
            1,
            -0.5,
            "shared_wiggle",
        )
        .expect("shared penalty");
        let fixed = OutputAxisPenalty::fixed(
            PenaltyMatrix::Diagonal(array![1.0, 0.0]),
            1,
            2.0,
        )
        .expect("fixed penalty");
        let blocks = OutputBlockAxis::new(vec!["left".into(), "right".into()], design)
            .expect("output axis")
            .with_offsets(array![[0.1, 0.2], [0.3, 0.4]])
            .expect("offset geometry")
            .with_initial_coefficients(array![[1.0, 2.0], [3.0, 4.0]])
            .expect("seed geometry")
            .with_penalty(shared)
            .expect("shared penalty geometry")
            .with_penalty(fixed)
            .expect("fixed penalty geometry")
            .parameter_blocks();

        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].name, "left");
        assert_eq!(blocks[1].offset, array![0.2, 0.4]);
        assert_eq!(blocks[1].initial_beta.as_ref(), Some(&array![2.0, 4.0]));
        for block in &blocks {
            assert_eq!(block.penalties[0].precision_label(), Some("shared_wiggle"));
            assert_eq!(block.penalties[1].fixed_log_lambda(), Some(2.0));
            assert_eq!(block.initial_log_lambdas, array![-0.5, 2.0]);
        }
    }

    #[test]
    fn rejects_ambiguous_or_misaligned_axis_geometry() {
        let design = DesignMatrix::from(array![[1.0], [1.0]]);
        assert!(
            OutputBlockAxis::new(vec!["same".into(), "same".into()], design.clone())
                .unwrap_err()
                .reason()
                .contains("duplicate")
        );
        assert!(
            OutputBlockAxis::new(vec!["one".into()], design)
                .expect("axis")
                .with_offsets(Array2::zeros((3, 1)))
                .unwrap_err()
                .reason()
                .contains("offsets have shape")
        );
    }
}
