//! Structural geometry and likelihood measures for separable indexed responses.
//!
//! A response cell can be structurally absent (missing, not at risk, or outside
//! a declared output domain) independently of the numerical weight attached to
//! an observed cell. Conflating those concepts in a single zero-valued weight
//! loses the response geometry and makes it impossible to validate event/risk
//! sets or preserve them in a fitted model. This module keeps them distinct.

use ndarray::{ArrayView1, ArrayView2};
use std::error::Error;
use std::fmt::{Display, Formatter};

/// Invalid indexed-response geometry or likelihood measure.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IndexedResponseError {
    reason: String,
}

impl IndexedResponseError {
    fn new(reason: impl Into<String>) -> Self {
        Self {
            reason: reason.into(),
        }
    }

    /// Human-readable invariant violation.
    pub fn reason(&self) -> &str {
        &self.reason
    }
}

impl Display for IndexedResponseError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.reason)
    }
}

impl Error for IndexedResponseError {}

/// A sorted, duplicate-free set of `(row, output)` cells in CSR form.
///
/// The representation costs `O(n_rows + n_cells)` and supports both sparse
/// inclusion sets and sparse exclusion sets through [`StructuralCells`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IndexedCellSet {
    n_rows: usize,
    n_outputs: usize,
    row_offsets: Vec<usize>,
    output_indices: Vec<usize>,
}

impl IndexedCellSet {
    /// Construct from CSR arrays. Output indices inside every row must be
    /// strictly increasing; duplicates are rejected rather than silently
    /// changing the declared response geometry.
    pub fn new(
        n_rows: usize,
        n_outputs: usize,
        row_offsets: Vec<usize>,
        output_indices: Vec<usize>,
    ) -> Result<Self, IndexedResponseError> {
        if row_offsets.len() != n_rows.saturating_add(1) {
            return Err(IndexedResponseError::new(format!(
                "indexed cell row_offsets length {} does not equal n_rows + 1 = {}",
                row_offsets.len(),
                n_rows.saturating_add(1)
            )));
        }
        if row_offsets.first().copied() != Some(0) {
            return Err(IndexedResponseError::new(
                "indexed cell row_offsets must begin at zero",
            ));
        }
        if row_offsets.last().copied() != Some(output_indices.len()) {
            return Err(IndexedResponseError::new(format!(
                "indexed cell final row offset {:?} does not equal cell count {}",
                row_offsets.last(),
                output_indices.len()
            )));
        }
        for row in 0..n_rows {
            let start = row_offsets[row];
            let end = row_offsets[row + 1];
            if start > end || end > output_indices.len() {
                return Err(IndexedResponseError::new(format!(
                    "indexed cell row {row} has invalid CSR range {start}..{end} for {} cells",
                    output_indices.len()
                )));
            }
            let outputs = &output_indices[start..end];
            for (position, &output) in outputs.iter().enumerate() {
                if output >= n_outputs {
                    return Err(IndexedResponseError::new(format!(
                        "indexed cell row {row} output {output} is outside 0..{n_outputs}"
                    )));
                }
                if position > 0 && outputs[position - 1] >= output {
                    return Err(IndexedResponseError::new(format!(
                        "indexed cell outputs in row {row} must be strictly increasing; found {} then {output}",
                        outputs[position - 1]
                    )));
                }
            }
        }
        Ok(Self {
            n_rows,
            n_outputs,
            row_offsets,
            output_indices,
        })
    }

    /// Construct from unordered cell coordinates. Coordinates are sorted into
    /// canonical row-major order; duplicates remain an error.
    pub fn from_cells(
        n_rows: usize,
        n_outputs: usize,
        mut cells: Vec<(usize, usize)>,
    ) -> Result<Self, IndexedResponseError> {
        cells.sort_unstable();
        if let Some(pair) = cells.windows(2).find(|pair| pair[0] == pair[1]) {
            return Err(IndexedResponseError::new(format!(
                "indexed response cell ({}, {}) was declared more than once",
                pair[0].0, pair[0].1
            )));
        }
        let mut row_offsets = vec![0usize; n_rows.saturating_add(1)];
        let mut output_indices = Vec::with_capacity(cells.len());
        for (row, output) in cells {
            if row >= n_rows {
                return Err(IndexedResponseError::new(format!(
                    "indexed cell row {row} is outside 0..{n_rows}"
                )));
            }
            if output >= n_outputs {
                return Err(IndexedResponseError::new(format!(
                    "indexed cell row {row} output {output} is outside 0..{n_outputs}"
                )));
            }
            row_offsets[row + 1] += 1;
            output_indices.push(output);
        }
        for row in 0..n_rows {
            row_offsets[row + 1] += row_offsets[row];
        }
        Self::new(n_rows, n_outputs, row_offsets, output_indices)
    }

    /// Number of rows in the declared response grid.
    pub fn n_rows(&self) -> usize {
        self.n_rows
    }

    /// Number of outputs in the declared response grid.
    pub fn n_outputs(&self) -> usize {
        self.n_outputs
    }

    /// Number of represented cells.
    pub fn len(&self) -> usize {
        self.output_indices.len()
    }

    /// Whether the set contains no cells.
    pub fn is_empty(&self) -> bool {
        self.output_indices.is_empty()
    }

    /// Sorted output indices represented on `row`.
    pub fn row_outputs(&self, row: usize) -> Option<&[usize]> {
        if row >= self.n_rows {
            return None;
        }
        Some(&self.output_indices[self.row_offsets[row]..self.row_offsets[row + 1]])
    }

    /// Whether `(row, output)` belongs to the set.
    pub fn contains(&self, row: usize, output: usize) -> bool {
        self.row_outputs(row)
            .is_some_and(|outputs| outputs.binary_search(&output).is_ok())
    }

    fn validate_shape(
        &self,
        n_rows: usize,
        n_outputs: usize,
    ) -> Result<(), IndexedResponseError> {
        if (self.n_rows, self.n_outputs) != (n_rows, n_outputs) {
            return Err(IndexedResponseError::new(format!(
                "indexed cell set geometry ({}, {}) does not match response geometry ({n_rows}, {n_outputs})",
                self.n_rows, self.n_outputs
            )));
        }
        Ok(())
    }
}

/// Structural presence of cells in a separable response grid.
#[derive(Clone, Copy, Debug)]
pub enum StructuralCells<'a> {
    /// Every `(row, output)` cell is structurally present.
    All,
    /// A dense Boolean activity mask with shape `(N, M)`.
    Dense(ArrayView2<'a, bool>),
    /// Only cells in the sparse set are structurally present.
    Only(&'a IndexedCellSet),
    /// Every cell except those in the sparse set is structurally present.
    AllExcept(&'a IndexedCellSet),
}

/// Numerical likelihood weighting, independent of structural presence.
#[derive(Clone, Copy, Debug)]
pub enum LikelihoodWeights<'a> {
    /// Unit weight for every structurally present cell.
    Uniform,
    /// One weight per observation row, shared across outputs.
    ByRow(ArrayView1<'a, f64>),
    /// One weight per response cell, shape `(N, M)`.
    ByCell(ArrayView2<'a, f64>),
}

/// Structural activity plus numerical likelihood weights for a separable
/// response. An inactive cell has no likelihood contribution; an active cell
/// with weight zero remains an observed member of the response geometry.
#[derive(Clone, Copy, Debug)]
pub struct SeparableCellMeasure<'a> {
    /// Which response cells exist in the likelihood.
    pub structural: StructuralCells<'a>,
    /// Numerical measure applied to existing cells.
    pub likelihood_weights: LikelihoodWeights<'a>,
}

impl<'a> SeparableCellMeasure<'a> {
    /// All cells present with unit numerical weight.
    pub const fn uniform() -> Self {
        Self {
            structural: StructuralCells::All,
            likelihood_weights: LikelihoodWeights::Uniform,
        }
    }

    /// All cells present with a shared weight per row.
    pub const fn row_weighted(weights: ArrayView1<'a, f64>) -> Self {
        Self {
            structural: StructuralCells::All,
            likelihood_weights: LikelihoodWeights::ByRow(weights),
        }
    }

    /// Construct from an explicit structural layout and numerical measure.
    pub const fn new(
        structural: StructuralCells<'a>,
        likelihood_weights: LikelihoodWeights<'a>,
    ) -> Self {
        Self {
            structural,
            likelihood_weights,
        }
    }

    /// Validate every shape, weight, and sparse-set index against `(N, M)`.
    pub fn validate(
        &self,
        n_rows: usize,
        n_outputs: usize,
    ) -> Result<(), IndexedResponseError> {
        match self.structural {
            StructuralCells::All => {}
            StructuralCells::Dense(mask) => {
                if mask.dim() != (n_rows, n_outputs) {
                    return Err(IndexedResponseError::new(format!(
                        "structural cell mask shape {:?} does not match ({n_rows}, {n_outputs})",
                        mask.dim()
                    )));
                }
            }
            StructuralCells::Only(cells) | StructuralCells::AllExcept(cells) => {
                cells.validate_shape(n_rows, n_outputs)?;
            }
        }
        match self.likelihood_weights {
            LikelihoodWeights::Uniform => {}
            LikelihoodWeights::ByRow(weights) => {
                if weights.len() != n_rows {
                    return Err(IndexedResponseError::new(format!(
                        "row likelihood weights length {} does not match N={n_rows}",
                        weights.len()
                    )));
                }
                for (row, &weight) in weights.iter().enumerate() {
                    validate_weight(weight, format!("row likelihood weight[{row}]"))?;
                }
            }
            LikelihoodWeights::ByCell(weights) => {
                if weights.dim() != (n_rows, n_outputs) {
                    return Err(IndexedResponseError::new(format!(
                        "cell likelihood weights shape {:?} does not match ({n_rows}, {n_outputs})",
                        weights.dim()
                    )));
                }
                for ((row, output), &weight) in weights.indexed_iter() {
                    validate_weight(
                        weight,
                        format!("cell likelihood weight[{row},{output}]"),
                    )?;
                }
            }
        }
        Ok(())
    }

    /// Whether `(row, output)` is structurally present.
    pub fn is_active(&self, row: usize, output: usize) -> bool {
        match self.structural {
            StructuralCells::All => true,
            StructuralCells::Dense(mask) => mask[[row, output]],
            StructuralCells::Only(cells) => cells.contains(row, output),
            StructuralCells::AllExcept(cells) => !cells.contains(row, output),
        }
    }

    /// Numerical weight of an active cell, or `None` when the cell is
    /// structurally absent. Returning `Some(0.0)` preserves the distinction
    /// between a present zero-weight observation and an absent cell.
    pub fn active_weight(&self, row: usize, output: usize) -> Option<f64> {
        if !self.is_active(row, output) {
            return None;
        }
        Some(match self.likelihood_weights {
            LikelihoodWeights::Uniform => 1.0,
            LikelihoodWeights::ByRow(weights) => weights[row],
            LikelihoodWeights::ByCell(weights) => weights[[row, output]],
        })
    }
}

fn validate_weight(weight: f64, context: String) -> Result<(), IndexedResponseError> {
    if !(weight.is_finite() && weight >= 0.0) {
        return Err(IndexedResponseError::new(format!(
            "{context} must be finite and non-negative (got {weight})"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sparse_inclusion_and_exclusion_preserve_structural_geometry() {
        let cells = IndexedCellSet::from_cells(3, 4, vec![(2, 3), (0, 1), (2, 0)])
            .expect("valid sparse cell set");
        assert_eq!(cells.row_outputs(0), Some(&[1][..]));
        assert_eq!(cells.row_outputs(1), Some(&[][..]));
        assert_eq!(cells.row_outputs(2), Some(&[0, 3][..]));

        let only = SeparableCellMeasure::new(
            StructuralCells::Only(&cells),
            LikelihoodWeights::Uniform,
        );
        only.validate(3, 4).expect("matching inclusion geometry");
        assert_eq!(only.active_weight(2, 3), Some(1.0));
        assert_eq!(only.active_weight(2, 2), None);

        let except = SeparableCellMeasure::new(
            StructuralCells::AllExcept(&cells),
            LikelihoodWeights::Uniform,
        );
        except.validate(3, 4).expect("matching exclusion geometry");
        assert_eq!(except.active_weight(2, 3), None);
        assert_eq!(except.active_weight(2, 2), Some(1.0));
    }

    #[test]
    fn structural_absence_is_distinct_from_zero_likelihood_weight() {
        let excluded = IndexedCellSet::from_cells(1, 2, vec![(0, 0)])
            .expect("valid exclusion set");
        let weights = ndarray::array![0.0];
        let measure = SeparableCellMeasure::new(
            StructuralCells::AllExcept(&excluded),
            LikelihoodWeights::ByRow(weights.view()),
        );
        measure.validate(1, 2).expect("valid measure");
        assert_eq!(measure.active_weight(0, 0), None);
        assert_eq!(measure.active_weight(0, 1), Some(0.0));
    }

    #[test]
    fn duplicate_cells_and_malformed_measures_are_rejected() {
        let duplicate = IndexedCellSet::from_cells(2, 2, vec![(0, 1), (0, 1)])
            .expect_err("duplicate structural declarations must fail");
        assert!(duplicate.reason().contains("more than once"));

        let bad_weights = ndarray::array![[1.0, -1.0]];
        let measure = SeparableCellMeasure::new(
            StructuralCells::All,
            LikelihoodWeights::ByCell(bad_weights.view()),
        );
        let error = measure
            .validate(1, 2)
            .expect_err("negative likelihood weight must fail");
        assert!(error.reason().contains("non-negative"));
    }
}
