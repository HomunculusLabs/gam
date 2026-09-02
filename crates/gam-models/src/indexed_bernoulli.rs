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
    BlockwiseFitOptions, CustomFamilyError, ParameterBlockSpec,
};
use crate::indexed_natural::{
    IndexedNaturalDiagonalFamily, IndexedNaturalDiagonalProgram, fit_indexed_natural_laml,
};
use gam_model_kernels::bernoulli_link::{
    bernoulli_natural_jet, bernoulli_natural_observation,
};
use gam_model_kernels::natural_observation::NaturalDiagonalObservation;
use gam_problem::{EstimationError, OwnedCellValues, OwnedSeparableCellMeasure};
use gam_solve::model_types::UnifiedFitResult;
use gam_spec::{InverseLink, StandardLink};

#[cfg(test)]
use crate::custom_family::{BlockWorkingSet, CustomFamily, ParameterBlockState};
#[cfg(test)]
use ndarray::Array1;

/// Owned indexed Bernoulli response family.
///
/// Structural activity is model geometry: inactive cells are never evaluated,
/// so the value field may contain a non-finite placeholder there.
/// Active cells, including active cells with numerical weight zero, must contain
/// a finite binomial proportion in `[0, 1]`.
#[derive(Clone, Debug)]
pub struct IndexedBernoulliProgram {
    y: OwnedCellValues,
    measure: OwnedSeparableCellMeasure,
    link: InverseLink,
}

/// Indexed Bernoulli response driven by the shared natural-diagonal engine.
pub type IndexedBernoulliFamily = IndexedNaturalDiagonalFamily<IndexedBernoulliProgram>;

impl IndexedNaturalDiagonalFamily<IndexedBernoulliProgram> {
    pub fn new(
        y: OwnedCellValues,
        measure: OwnedSeparableCellMeasure,
        link: InverseLink,
    ) -> Result<Self, EstimationError> {
        let (n_rows, n_outputs) = (y.n_rows(), y.n_outputs());
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
        measure.try_for_each_active(|row, output, _| {
            let response = y
                .value(row, output)
                .expect("response and measure shapes were matched above");
            if !(response.is_finite() && (0.0..=1.0).contains(&response)) {
                return Err(EstimationError::InvalidInput(format!(
                    "indexed Bernoulli response[{row},{output}] must be finite and in [0,1] because the cell is structurally active, got {response}"
                )));
            }
            Ok(())
        })?;
        Self::from_program(IndexedBernoulliProgram {
            y,
            measure,
            link,
        })
    }

    pub fn link(&self) -> &InverseLink {
        &self.program().link
    }
}

impl IndexedNaturalDiagonalProgram for IndexedBernoulliProgram {
    fn family_name(&self) -> &'static str {
        "indexed Bernoulli"
    }

    fn n_rows(&self) -> usize {
        self.y.n_rows()
    }

    fn n_outputs(&self) -> usize {
        self.y.n_outputs()
    }

    fn measure(&self) -> &OwnedSeparableCellMeasure {
        &self.measure
    }

    fn observation(
        &self,
        row: usize,
        output: usize,
        eta: f64,
    ) -> Result<NaturalDiagonalObservation, EstimationError> {
        bernoulli_natural_observation(
            row,
            self.y
                .value(row, output)
                .expect("validated response geometry"),
            eta,
            &self.link,
        )
        .map(Into::into)
    }

    fn classical_deviance(&self, fitted_log_likelihood: f64) -> Result<Option<f64>, String> {
        let mut saturated = 0.0;
        self.measure.try_for_each_active(|row, output, weight| {
            if weight == 0.0 {
                return Ok::<(), String>(());
            }
            let response = self
                .y
                .value(row, output)
                .expect("validated response geometry");
            let unit = if response == 0.0 || response == 1.0 {
                0.0
            } else {
                response * response.ln() + (1.0 - response) * (-response).ln_1p()
            };
            saturated += weight * unit;
            Ok(())
        })?;
        Ok(Some(2.0 * (saturated - fitted_log_likelihood)))
    }

    fn coefficient_objective_is_globally_convex(&self) -> bool {
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
}

/// Fit an indexed multi-output Bernoulli GAM and select every unfixed,
/// potentially shared precision by the canonical LAML outer optimizer.
pub fn fit_indexed_bernoulli_laml(
    family: &IndexedBernoulliFamily,
    specs: &[ParameterBlockSpec],
    options: &BlockwiseFitOptions,
) -> Result<UnifiedFitResult, CustomFamilyError> {
    fit_indexed_natural_laml(family, specs, options)
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
            OwnedCellValues::dense(y),
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
            OwnedCellValues::constant(1, 1, 1.0),
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

    #[test]
    fn sparse_event_overrides_equal_the_dense_response_likelihood() {
        let sparse = OwnedCellValues::constant_with_overrides(
            3,
            2,
            0.0,
            vec![(0, 1, 1.0), (2, 0, 1.0)],
        )
        .expect("sparse response");
        let dense = OwnedCellValues::dense(ndarray::array![[0.0, 1.0], [0.0, 0.0], [1.0, 0.0]]);
        let link = InverseLink::Standard(StandardLink::Logit);
        let sparse_family = IndexedBernoulliFamily::new(
            sparse,
            OwnedSeparableCellMeasure::uniform(3, 2),
            link.clone(),
        )
        .expect("sparse family");
        let dense_family = IndexedBernoulliFamily::new(
            dense,
            OwnedSeparableCellMeasure::uniform(3, 2),
            link,
        )
        .expect("dense family");
        let point = states(&[
            ndarray::array![-0.3, 0.2, 1.1],
            ndarray::array![0.7, -0.5, 0.4],
        ]);

        let sparse_evaluation = sparse_family.evaluate(&point).expect("sparse evaluation");
        let dense_evaluation = dense_family.evaluate(&point).expect("dense evaluation");
        assert_eq!(
            sparse_evaluation.log_likelihood,
            dense_evaluation.log_likelihood,
        );
        for (sparse_working, dense_working) in sparse_evaluation
            .blockworking_sets
            .iter()
            .zip(dense_evaluation.blockworking_sets.iter())
        {
            let BlockWorkingSet::NaturalDiagonal {
                score: sparse_score,
                observed_curvature: sparse_curvature,
            } = sparse_working
            else {
                panic!("sparse indexed response must retain natural geometry");
            };
            let BlockWorkingSet::NaturalDiagonal {
                score: dense_score,
                observed_curvature: dense_curvature,
            } = dense_working
            else {
                panic!("dense indexed response must retain natural geometry");
            };
            assert_eq!(sparse_score, dense_score);
            assert_eq!(sparse_curvature, dense_curvature);
        }
    }
}
