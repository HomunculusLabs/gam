//! Indexed multi-output point-process GAM with joint LAML selection.
//!
//! Counts and compensator exposures are separate indexed fields.  Exact event
//! nodes therefore use `(count > 0, exposure = 0)` directly, while quadrature
//! nodes use `(count = 0, exposure > 0)`; no `log(0)` Poisson offset or time
//! bin is required.  Structural risk-set activity and numerical subject/IPW
//! weights are carried independently by the response measure.

use crate::custom_family::{
    BlockwiseFitOptions, CustomFamilyError, ParameterBlockSpec,
};
use crate::indexed_natural::{
    IndexedNaturalDiagonalFamily, IndexedNaturalDiagonalProgram, fit_indexed_natural_laml,
};
use gam_model_kernels::natural_observation::NaturalDiagonalObservation;
use gam_model_kernels::point_process::point_process_natural_observation;
use gam_problem::{EstimationError, OwnedCellValues, OwnedSeparableCellMeasure};
use gam_solve::model_types::UnifiedFitResult;

/// Row data for the exact marked point-process likelihood.
#[derive(Clone, Debug)]
pub struct IndexedPointProcessProgram {
    counts: OwnedCellValues,
    exposures: OwnedCellValues,
    measure: OwnedSeparableCellMeasure,
}

/// Indexed marked point-process response driven by the shared natural engine.
pub type IndexedPointProcessFamily = IndexedNaturalDiagonalFamily<IndexedPointProcessProgram>;

impl IndexedNaturalDiagonalFamily<IndexedPointProcessProgram> {
    pub fn new(
        counts: OwnedCellValues,
        exposures: OwnedCellValues,
        measure: OwnedSeparableCellMeasure,
    ) -> Result<Self, EstimationError> {
        let shape = (counts.n_rows(), counts.n_outputs());
        if (exposures.n_rows(), exposures.n_outputs()) != shape {
            return Err(EstimationError::InvalidInput(format!(
                "indexed point-process exposure geometry ({}, {}) does not match count geometry {shape:?}",
                exposures.n_rows(),
                exposures.n_outputs(),
            )));
        }
        if (measure.n_rows(), measure.n_outputs()) != shape {
            return Err(EstimationError::InvalidInput(format!(
                "indexed point-process measure geometry ({}, {}) does not match count geometry {shape:?}",
                measure.n_rows(),
                measure.n_outputs(),
            )));
        }
        measure.try_for_each_active(|row, output, _| {
            let count = counts
                .value(row, output)
                .expect("count and measure shapes were matched above");
            if !(count.is_finite() && count >= 0.0) {
                return Err(EstimationError::InvalidInput(format!(
                    "indexed point-process count[{row},{output}] must be finite and non-negative because the cell is structurally active, got {count}",
                )));
            }
            let exposure = exposures
                .value(row, output)
                .expect("exposure and measure shapes were matched above");
            if !(exposure.is_finite() && exposure >= 0.0) {
                return Err(EstimationError::InvalidInput(format!(
                    "indexed point-process exposure[{row},{output}] must be finite and non-negative because the cell is structurally active, got {exposure}",
                )));
            }
            Ok(())
        })?;
        Self::from_program(IndexedPointProcessProgram {
            counts,
            exposures,
            measure,
        })
    }
}

impl IndexedNaturalDiagonalProgram for IndexedPointProcessProgram {
    fn family_name(&self) -> &'static str {
        "indexed point process"
    }

    fn n_rows(&self) -> usize {
        self.counts.n_rows()
    }

    fn n_outputs(&self) -> usize {
        self.counts.n_outputs()
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
        point_process_natural_observation(
            row,
            self.counts
                .value(row, output)
                .expect("validated count geometry"),
            self.exposures
                .value(row, output)
                .expect("validated exposure geometry"),
            eta,
        )
    }

    fn coefficient_objective_is_globally_convex(&self) -> bool {
        true
    }
}

/// Fit an indexed multi-output point-process GAM and select every unfixed,
/// potentially shared precision by the canonical LAML outer optimizer.
pub fn fit_indexed_point_process_laml(
    family: &IndexedPointProcessFamily,
    specs: &[ParameterBlockSpec],
    options: &BlockwiseFitOptions,
) -> Result<UnifiedFitResult, CustomFamilyError> {
    fit_indexed_natural_laml(family, specs, options)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::custom_family::{BlockWorkingSet, CustomFamily, ParameterBlockState};
    use gam_problem::{
        IndexedCellSet, LikelihoodWeights, SeparableCellMeasure, StructuralCells,
    };
    use ndarray::{Array1, array};

    fn states(etas: &[Array1<f64>]) -> Vec<ParameterBlockState> {
        etas.iter()
            .map(|eta| ParameterBlockState {
                beta: Array1::zeros(0),
                eta: eta.clone(),
            })
            .collect()
    }

    #[test]
    fn sparse_events_and_row_broadcast_exposure_equal_exact_process_law() {
        let counts = OwnedCellValues::constant_with_overrides(
            2,
            2,
            0.0,
            vec![(0, 1, 1.0)],
        )
        .expect("sparse event field");
        let family = IndexedPointProcessFamily::new(
            counts,
            OwnedCellValues::by_row(array![0.0, 2.5], 2),
            OwnedSeparableCellMeasure::uniform(2, 2),
        )
        .expect("point-process family");
        let eta = [array![0.3, -0.2], array![0.7, 0.4]];
        let evaluation = family.evaluate(&states(&eta)).expect("exact evaluation");
        let expected = 0.7 - 2.5 * ((-0.2_f64).exp() + 0.4_f64.exp());

        assert!((evaluation.log_likelihood - expected).abs() < 1.0e-14);
        let BlockWorkingSet::NaturalDiagonal {
            score,
            observed_curvature,
        } = &evaluation.blockworking_sets[1]
        else {
            panic!("point-process family must expose natural diagonal geometry");
        };
        assert_eq!(score[0], 1.0);
        assert_eq!(observed_curvature[0], 0.0);
        assert_eq!(score[1], -2.5 * 0.4_f64.exp());
        assert_eq!(observed_curvature[1], 2.5 * 0.4_f64.exp());
    }

    #[test]
    fn inactive_placeholders_and_active_zero_weights_are_never_evaluated() {
        let active = IndexedCellSet::from_cells(2, 2, vec![(0, 0), (0, 1), (1, 1)])
            .expect("activity set");
        let weights = array![[1.0, 0.0], [7.0, 2.0]];
        let measure = SeparableCellMeasure::new(
            StructuralCells::Only(&active),
            LikelihoodWeights::ByCell(weights.view()),
        )
        .to_owned(2, 2)
        .expect("owned measure");
        let family = IndexedPointProcessFamily::new(
            OwnedCellValues::dense(array![[1.0, 0.0], [f64::NAN, 0.0]]),
            OwnedCellValues::dense(array![[0.0, 1.0], [f64::NAN, 1.5]]),
            measure,
        )
        .expect("inactive placeholders are outside the likelihood");
        let evaluation = family
            .evaluate(&states(&[
                array![1_000.0, -8.0],
                array![1_000.0, 0.0],
            ]))
            .expect("finite active likelihood");
        assert_eq!(evaluation.log_likelihood, 1_000.0 - 3.0);
    }

    #[test]
    fn active_invalid_counts_and_exposures_are_rejected_at_construction() {
        let measure = OwnedSeparableCellMeasure::uniform(1, 1);
        let bad_count = IndexedPointProcessFamily::new(
            OwnedCellValues::constant(1, 1, -1.0),
            OwnedCellValues::constant(1, 1, 0.0),
            measure.clone(),
        )
        .expect_err("negative count");
        assert!(bad_count.to_string().contains("count[0,0]"));

        let bad_exposure = IndexedPointProcessFamily::new(
            OwnedCellValues::constant(1, 1, 0.0),
            OwnedCellValues::constant(1, 1, f64::NAN),
            measure,
        )
        .expect_err("non-finite exposure");
        assert!(bad_exposure.to_string().contains("exposure[0,0]"));
    }
}
