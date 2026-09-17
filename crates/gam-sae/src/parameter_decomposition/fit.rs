//! Joint fit of a manifold parameter decomposition (#2951).
//!
//! The loop is designed on the issue (comments 5716713394 and 5717011462):
//! * an anchored finite-intervention objective at fixed structure;
//! * conditionally Gaussian coefficient blocks marginalized exactly;
//! * a smooth outer criterion over labels, weights and the non-Gaussian blocks;
//! * witnesses from the separation oracle;
//! * structural proposals accepted on decoded code under a fidelity check.
//!
//! This file assembles those pieces and owns no numerical primitive. The anchor
//! belongs to `lift`, the field and its pullbacks to `field`, and REML to gam-solve.
//!
//! # Conditionally Gaussian coefficient block
//!
//! Take one parameter family whose basis matrices are in product form
//! `B_j = L_j R_jᵀ`, with `L_j` of shape `d_out × r_j`, `R_j` of shape
//! `d_in × r_j`, and `j = 1..K`. Hold the labels, weights and right factors fixed.
//! Under the residual anchor, readout row `r` at mask `m_r` on the current
//! intervened input `h_r` is
//!
//! ```text
//! y_r = m_Δ Θ_* h_r + Σ_j β_j(m_r) L_j R_jᵀ h_r,     β(m) = Σ_c (m_c − m_Δ) v_c.
//! ```
//!
//! So at fixed `(z, w, R, m, h)` the row is linear in the left factors. Output
//! coordinate `o` reads the design row whose block `j` is `β_j(m_r) R_jᵀ h_r`, at
//! columns `offset_j + a` with `offset_j = Σ_{k<j} r_k`. The function-space penalty
//!
//! ```text
//! Σ_jk S_jk ⟨L_j R_jᵀ, L_k R_kᵀ⟩_F = Σ_o Σ_jk S_jk L_{j,o}ᵀ (R_jᵀ R_k) L_{k,o}
//! ```
//!
//! is the same quadratic form for every output coordinate, with blocks
//! `S_jk R_jᵀ R_k`. The output coordinates are coordinates of one vector-valued
//! observation, so the block is exactly the shared-dispersion multi-response
//! Gaussian REML problem of [`gaussian_reml_multi_shared_dispersion_closed_form`].
//! Its posterior mean and smoothing strength are taken from that owner and never
//! refitted here. A dense basis matrix is the case `R_j = I_{d_in}`: the design row
//! is `β(m_r) ⊗ h_r` and the penalty is `S ⊗ I_{d_in}`.
//!
//! # Rows that are refused
//!
//! A row whose declared mask has `m_c = m_Δ` for every component executes `m_Δ Θ_*`
//! at every parameter value:
//! * at all-on it is the bit-identity guard (P16);
//! * with the residual removed and every component off it is the zero tensor.
//!
//! Neither observes anything this block or the outer criterion moves, so such a
//! row is refused. The refusal keys on the declared mask values, never on a
//! computed moment. `β(m_r)` can vanish by cancellation at the current labels and
//! weights while its derivatives in them do not, and such a row is admitted.

use std::fmt;

use gam_linalg::matrix::dense_rowwise_kronecker;
use gam_solve::estimate::EstimationError;
use gam_solve::gaussian_reml::{
    GaussianRemlMultiResult, gaussian_reml_multi_shared_dispersion_closed_form,
};
use ndarray::{Array2, ArrayView1, ArrayView2, s};

/// The rows of one conditionally Gaussian coefficient block.
#[derive(Clone, Copy, Debug)]
pub struct GaussianBlockRows<'a> {
    /// The declared component masks `m_c` of each row, `n × C`.
    pub component_masks: ArrayView2<'a, f64>,
    /// The declared residual mask `m_Δ` of each row, length `n`.
    pub residual_masks: ArrayView1<'a, f64>,
    /// The anchor moment `β(m_r) = Σ_c (m_c − m_Δ) v_c` of each row, `n × K`, read
    /// from the anchor.
    pub moments: ArrayView2<'a, f64>,
    /// The current intervened input `h_r` of the edited tensor, `n × d_in`: the
    /// input under every upstream edit of the row, never a cached clean input.
    pub inputs: ArrayView2<'a, f64>,
    /// The target readout minus the anchored native term `m_Δ Θ_* h_r`, `n × d_out`.
    pub responses: ArrayView2<'a, f64>,
}

/// A fitted conditionally Gaussian coefficient block.
#[derive(Clone, Debug)]
pub struct GaussianBlockFit {
    /// Posterior-mean left factors `L_j`, each `d_out × r_j`.
    pub left_factors: Vec<Array2<f64>>,
    /// The shared-dispersion REML fit of the block. Its coefficients are in design
    /// order (row `offset_j + a`, one column per output coordinate).
    pub reml: GaussianRemlMultiResult,
}

/// Why a conditionally Gaussian coefficient block was not fitted.
#[derive(Debug)]
pub enum GaussianBlockError {
    EmptyBlock {
        rows: usize,
        components: usize,
        basis: usize,
        input_dim: usize,
        output_dim: usize,
    },
    RowCountMismatch {
        what: &'static str,
        rows: usize,
        expected: usize,
    },
    RightFactorCount {
        basis: usize,
        right_factors: usize,
    },
    RightFactorShape {
        basis: usize,
        rows: usize,
        rank: usize,
        input_dim: usize,
    },
    PenaltyShape {
        basis: usize,
        rows: usize,
        cols: usize,
    },
    NonFinite {
        what: &'static str,
    },
    /// The row's declared mask has `m_c = m_Δ` for every component.
    NativeMultipleRow {
        row: usize,
    },
    /// The block's dense state exceeds the host in-core budget. `required_bytes`
    /// is `None` when the count overflows `usize`.
    AdmissionRefused {
        required_bytes: Option<usize>,
        budget_bytes: usize,
    },
    Reml(EstimationError),
}

impl fmt::Display for GaussianBlockError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyBlock {
                rows,
                components,
                basis,
                input_dim,
                output_dim,
            } => write!(
                f,
                "conditionally Gaussian block refused: it needs positive rows, components, basis \
                 size, input and output dimensions; got n={rows}, C={components}, K={basis}, \
                 d_in={input_dim}, d_out={output_dim}"
            ),
            Self::RowCountMismatch {
                what,
                rows,
                expected,
            } => write!(
                f,
                "conditionally Gaussian block refused: the {what} have {rows} rows but the moments \
                 have {expected}"
            ),
            Self::RightFactorCount {
                basis,
                right_factors,
            } => write!(
                f,
                "conditionally Gaussian block refused: {right_factors} right factors for {basis} \
                 basis matrices"
            ),
            Self::RightFactorShape {
                basis,
                rows,
                rank,
                input_dim,
            } => write!(
                f,
                "conditionally Gaussian block refused: right factor {basis} is {rows}×{rank}; it \
                 needs {input_dim} rows and a positive rank"
            ),
            Self::PenaltyShape { basis, rows, cols } => write!(
                f,
                "conditionally Gaussian block refused: the field penalty must be {basis}×{basis} \
                 to match the basis; got {rows}×{cols}"
            ),
            Self::NonFinite { what } => write!(
                f,
                "conditionally Gaussian block refused: the {what} contain a non-finite value"
            ),
            Self::NativeMultipleRow { row } => write!(
                f,
                "conditionally Gaussian block refused: row {row} declares m_c = m_Delta for every \
                 component, so it executes m_Delta Theta_* whatever the field, labels and weights \
                 are, and observes nothing the fit moves"
            ),
            Self::AdmissionRefused {
                required_bytes: Some(required),
                budget_bytes,
            } => write!(
                f,
                "conditionally Gaussian block refused: its dense state needs at least {required} \
                 bytes, above the host in-core budget of {budget_bytes} bytes"
            ),
            Self::AdmissionRefused {
                required_bytes: None,
                budget_bytes,
            } => write!(
                f,
                "conditionally Gaussian block refused: its dense byte count overflows usize \
                 (host in-core budget {budget_bytes} bytes)"
            ),
            Self::Reml(error) => write!(f, "conditionally Gaussian block REML failed: {error}"),
        }
    }
}

impl std::error::Error for GaussianBlockError {}

/// Fits one conditionally Gaussian coefficient block exactly.
///
/// * `right_factors` holds the fixed `R_j`, `d_in × r_j`, one per basis matrix.
/// * `field_penalty` is the `K × K` function-space penalty `S` of the family's field.
///
/// Refuses:
/// * an empty or mis-shaped block, or non-finite input;
/// * a row whose declared mask is a native multiple (see the module docs);
/// * a block whose dense state exceeds the host in-core budget.
///
/// Everything else is decided by the REML owner, whose refusals are returned
/// unchanged.
pub fn fit_gaussian_coefficient_block(
    rows: GaussianBlockRows<'_>,
    right_factors: &[ArrayView2<'_, f64>],
    field_penalty: ArrayView2<'_, f64>,
) -> Result<GaussianBlockFit, GaussianBlockError> {
    let n = rows.moments.nrows();
    let components = rows.component_masks.ncols();
    let basis = rows.moments.ncols();
    let input_dim = rows.inputs.ncols();
    let output_dim = rows.responses.ncols();
    if n == 0 || components == 0 || basis == 0 || input_dim == 0 || output_dim == 0 {
        return Err(GaussianBlockError::EmptyBlock {
            rows: n,
            components,
            basis,
            input_dim,
            output_dim,
        });
    }
    for (what, count) in [
        ("component masks", rows.component_masks.nrows()),
        ("residual masks", rows.residual_masks.len()),
        ("inputs", rows.inputs.nrows()),
        ("responses", rows.responses.nrows()),
    ] {
        if count != n {
            return Err(GaussianBlockError::RowCountMismatch {
                what,
                rows: count,
                expected: n,
            });
        }
    }
    if right_factors.len() != basis {
        return Err(GaussianBlockError::RightFactorCount {
            basis,
            right_factors: right_factors.len(),
        });
    }
    for (index, factor) in right_factors.iter().enumerate() {
        if factor.nrows() != input_dim || factor.ncols() == 0 {
            return Err(GaussianBlockError::RightFactorShape {
                basis: index,
                rows: factor.nrows(),
                rank: factor.ncols(),
                input_dim,
            });
        }
    }
    if field_penalty.nrows() != basis || field_penalty.ncols() != basis {
        return Err(GaussianBlockError::PenaltyShape {
            basis,
            rows: field_penalty.nrows(),
            cols: field_penalty.ncols(),
        });
    }
    for (what, values) in [
        ("component masks", rows.component_masks),
        ("moments", rows.moments),
        ("inputs", rows.inputs),
        ("responses", rows.responses),
        ("field penalty entries", field_penalty),
    ] {
        if values.iter().any(|value| !value.is_finite()) {
            return Err(GaussianBlockError::NonFinite { what });
        }
    }
    if rows.residual_masks.iter().any(|value| !value.is_finite()) {
        return Err(GaussianBlockError::NonFinite {
            what: "residual masks",
        });
    }
    if right_factors
        .iter()
        .any(|factor| factor.iter().any(|value| !value.is_finite()))
    {
        return Err(GaussianBlockError::NonFinite {
            what: "right factors",
        });
    }
    for row in 0..n {
        let residual = rows.residual_masks[row];
        if rows
            .component_masks
            .row(row)
            .iter()
            .all(|&mask| mask == residual)
        {
            return Err(GaussianBlockError::NativeMultipleRow { row });
        }
    }
    let ranks: Vec<usize> = right_factors.iter().map(|factor| factor.ncols()).collect();
    admit_dense_block(
        n,
        &ranks,
        output_dim,
        crate::manifold::sae_host_in_core_budget_bytes().0,
    )?;
    let design = block_design(rows.moments, rows.inputs, right_factors);
    let penalty = block_penalty(field_penalty, right_factors);
    let reml = gaussian_reml_multi_shared_dispersion_closed_form(
        design.view(),
        rows.responses,
        penalty.view(),
        None,
        None,
    )
    .map_err(GaussianBlockError::Reml)?;
    let left_factors = left_factors_from_design_order(reml.coefficients.view(), right_factors);
    Ok(GaussianBlockFit { left_factors, reml })
}

/// `[0, r_0, r_0 + r_1, …, Σ_j r_j]`: where each basis matrix's block starts.
fn design_offsets(right_factors: &[ArrayView2<'_, f64>]) -> Vec<usize> {
    let mut offsets = Vec::with_capacity(right_factors.len() + 1);
    let mut total = 0;
    offsets.push(total);
    for factor in right_factors {
        total += factor.ncols();
        offsets.push(total);
    }
    offsets
}

/// The design in design order: block `j` of row `r` is `β_j(m_r) R_jᵀ h_r`.
fn block_design(
    moments: ArrayView2<'_, f64>,
    inputs: ArrayView2<'_, f64>,
    right_factors: &[ArrayView2<'_, f64>],
) -> Array2<f64> {
    let offsets = design_offsets(right_factors);
    let mut design = Array2::<f64>::zeros((moments.nrows(), offsets[right_factors.len()]));
    for (j, factor) in right_factors.iter().enumerate() {
        let projected = inputs.dot(factor);
        let block = dense_rowwise_kronecker(moments.slice(s![.., j..j + 1]), projected.view());
        design
            .slice_mut(s![.., offsets[j]..offsets[j + 1]])
            .assign(&block);
    }
    design
}

/// The penalty in design order: block `(j, k)` is `S_jk R_jᵀ R_k`, the quadratic
/// form `Σ_jk S_jk ⟨L_j R_jᵀ, L_k R_kᵀ⟩_F` of one output coordinate.
fn block_penalty(field_penalty: ArrayView2<'_, f64>, right_factors: &[ArrayView2<'_, f64>]) -> Array2<f64> {
    let offsets = design_offsets(right_factors);
    let columns = offsets[right_factors.len()];
    let mut penalty = Array2::<f64>::zeros((columns, columns));
    for (j, left) in right_factors.iter().enumerate() {
        for (k, right) in right_factors.iter().enumerate() {
            let strength = field_penalty[[j, k]];
            let gram = left.t().dot(right);
            penalty
                .slice_mut(s![offsets[j]..offsets[j + 1], offsets[k]..offsets[k + 1]])
                .assign(&gram.mapv(|value| strength * value));
        }
    }
    penalty
}

/// Reads design-order coefficients (row `offset_j + a`, column `o`) as `L_j[[o, a]]`.
fn left_factors_from_design_order(
    coefficients: ArrayView2<'_, f64>,
    right_factors: &[ArrayView2<'_, f64>],
) -> Vec<Array2<f64>> {
    let output_dim = coefficients.ncols();
    let offsets = design_offsets(right_factors);
    right_factors
        .iter()
        .enumerate()
        .map(|(j, factor)| {
            Array2::from_shape_fn((output_dim, factor.ncols()), |(o, a)| {
                coefficients[[offsets[j] + a, o]]
            })
        })
        .collect()
}

/// A lower bound on the block's peak dense state in bytes, or `None` on overflow.
///
/// With `p = Σ_j r_j`, it counts what this file materializes together with the
/// state any dense eigen-REML on `p` coefficients must hold:
/// * the design, `n·p`, and the widest projected input `R_jᵀ h`, `n·max_j r_j`,
///   while it is folded into the design;
/// * the penalty in design order, `p²`;
/// * the Gram `XᵀWX` and its eigenbasis, `2p²`;
/// * the coefficients, `p·d_out`.
///
/// The REML owner's transient factorization workspace is on top of this. So the
/// bound is necessary, not sufficient: it refuses a block that cannot fit, and it
/// does not certify one that can.
fn dense_block_bytes(rows: usize, ranks: &[usize], outputs: usize) -> Option<usize> {
    let columns = ranks
        .iter()
        .try_fold(0usize, |total, &rank| total.checked_add(rank))?;
    let widest = ranks.iter().copied().max().unwrap_or(0);
    let design = rows.checked_mul(columns.checked_add(widest)?)?;
    let squares = columns.checked_mul(columns)?.checked_mul(3)?;
    let coefficients = columns.checked_mul(outputs)?;
    design
        .checked_add(squares)?
        .checked_add(coefficients)?
        .checked_mul(std::mem::size_of::<f64>())
}

fn admit_dense_block(
    rows: usize,
    ranks: &[usize],
    outputs: usize,
    budget_bytes: usize,
) -> Result<(), GaussianBlockError> {
    match dense_block_bytes(rows, ranks, outputs) {
        Some(required) if required <= budget_bytes => Ok(()),
        required_bytes => Err(GaussianBlockError::AdmissionRefused {
            required_bytes,
            budget_bytes,
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gam_linalg::roundoff::accumulation_growth;
    use ndarray::{Array1, array};

    const COMPONENTS: usize = 3;
    const BASIS: usize = 3;
    const INPUT_DIM: usize = 4;
    const OUTPUT_DIM: usize = 3;
    const RANK: usize = 2;
    const INPUT_ROWS: usize = 6;

    /// Basis evaluations `φ(z_c)` of three components (a quadratic basis at
    /// z = 0.5, −0.3, 0.8) and their weights on the scale gauge `Σ_c w_c = 1`.
    const COMPONENT_BASIS: [[f64; BASIS]; COMPONENTS] =
        [[1.0, 0.5, 0.25], [1.0, -0.3, 0.09], [1.0, 0.8, 0.64]];
    const COMPONENT_WEIGHTS: [f64; COMPONENTS] = [0.5, 0.3, 0.2];

    /// Declared component masks and the residual mask.
    type Mask = ([f64; COMPONENTS], f64);

    /// Residual-removed witness masks whose moments are linearly independent.
    const SUFFICIENCY_MASKS: [Mask; 3] = [
        ([1.0, 0.0, 0.0], 0.0),
        ([1.0, 1.0, 0.0], 0.0),
        ([0.0, 1.0, 1.0], 0.0),
    ];

    /// The all-on setting: every component and the residual kept.
    const ALL_ON: Mask = ([1.0; COMPONENTS], 1.0);

    /// `β(m) = Σ_c (m_c − m_Δ) w_c φ(z_c)`, the fixture's anchor moment.
    fn moment(mask: Mask) -> [f64; BASIS] {
        let mut beta = [0.0; BASIS];
        for component in 0..COMPONENTS {
            let scale = (mask.0[component] - mask.1) * COMPONENT_WEIGHTS[component];
            for j in 0..BASIS {
                beta[j] += scale * COMPONENT_BASIS[component][j];
            }
        }
        beta
    }

    /// Right factors `R_j` (`4 × 2`): full column rank, neither orthonormal nor
    /// mutually orthogonal, so every Gram `R_jᵀ R_k` matters.
    fn generic_right_factors() -> Vec<Array2<f64>> {
        (0..BASIS)
            .map(|j| {
                Array2::from_shape_fn((INPUT_DIM, RANK), |(i, a)| {
                    (0.9 * ((i + 1) * (a + 1)) as f64 + 0.4 * j as f64).cos()
                })
            })
            .collect()
    }

    fn planted_left(j: usize, o: usize, a: usize) -> f64 {
        0.4 + 0.3 * j as f64 - 0.2 * o as f64 + 0.15 * a as f64 * (j as f64 + 1.0)
    }

    struct Fixture {
        component_masks: Array2<f64>,
        residual_masks: Array1<f64>,
        moments: Array2<f64>,
        inputs: Array2<f64>,
        responses: Array2<f64>,
        right: Vec<Array2<f64>>,
        penalty: Array2<f64>,
    }

    impl Fixture {
        fn rows(&self) -> GaussianBlockRows<'_> {
            GaussianBlockRows {
                component_masks: self.component_masks.view(),
                residual_masks: self.residual_masks.view(),
                moments: self.moments.view(),
                inputs: self.inputs.view(),
                responses: self.responses.view(),
            }
        }

        fn right_views(&self) -> Vec<ArrayView2<'_, f64>> {
            self.right.iter().map(|factor| factor.view()).collect()
        }
    }

    /// Each mask is applied to the same six inputs. Responses are the planted
    /// factored block plus a deterministic perturbation, so the profiled REML
    /// deviance is positive.
    fn fixture(masks: &[Mask], right: Vec<Array2<f64>>) -> Fixture {
        let n = masks.len() * INPUT_ROWS;
        let mut component_masks = Array2::<f64>::zeros((n, COMPONENTS));
        let mut residual_masks = Array1::<f64>::zeros(n);
        let mut moments = Array2::<f64>::zeros((n, BASIS));
        let mut inputs = Array2::<f64>::zeros((n, INPUT_DIM));
        let mut responses = Array2::<f64>::zeros((n, OUTPUT_DIM));
        for (mask_index, &mask) in masks.iter().enumerate() {
            let beta = moment(mask);
            for x in 0..INPUT_ROWS {
                let row = mask_index * INPUT_ROWS + x;
                for component in 0..COMPONENTS {
                    component_masks[[row, component]] = mask.0[component];
                }
                residual_masks[row] = mask.1;
                for j in 0..BASIS {
                    moments[[row, j]] = beta[j];
                }
                for i in 0..INPUT_DIM {
                    inputs[[row, i]] = (0.7 * x as f64 + 1.1 * i as f64 + 0.3).sin();
                }
                for o in 0..OUTPUT_DIM {
                    let mut value = 0.01 * (2.1 * row as f64 + 0.9 * o as f64 + 0.4).sin();
                    for j in 0..BASIS {
                        for a in 0..right[j].ncols() {
                            let mut projected = 0.0;
                            for i in 0..INPUT_DIM {
                                projected += right[j][[i, a]] * inputs[[row, i]];
                            }
                            value += beta[j] * planted_left(j, o, a) * projected;
                        }
                    }
                    responses[[row, o]] = value;
                }
            }
        }
        // The second-difference penalty D₂ᵀD₂ on three coefficients has rank 1, so
        // constant and linear fields are unpenalized.
        let penalty = array![[1.0, -2.0, 1.0], [-2.0, 4.0, -2.0], [1.0, -2.0, 1.0]];
        Fixture {
            component_masks,
            residual_masks,
            moments,
            inputs,
            responses,
            right,
            penalty,
        }
    }

    #[test]
    fn block_fit_reads_left_factors_in_design_order() {
        let data = fixture(&SUFFICIENCY_MASKS, generic_right_factors());
        let right = data.right_views();
        let fit = fit_gaussian_coefficient_block(data.rows(), &right, data.penalty.view())
            .expect("the full-rank sufficiency fixture is admitted and fitted");
        assert_eq!(fit.left_factors.len(), BASIS);
        for factor in &fit.left_factors {
            assert_eq!(factor.dim(), (OUTPUT_DIM, RANK));
        }
        let coefficients = &fit.reml.coefficients;
        // Negative control: the blocks read in reverse basis order.
        let reversed: Vec<Array2<f64>> = (0..BASIS)
            .map(|j| {
                Array2::from_shape_fn((OUTPUT_DIM, RANK), |(o, a)| {
                    coefficients[[(BASIS - 1 - j) * RANK + a, o]]
                })
            })
            .collect();
        // Both sides sum the K·d_in·r monomials β_j h_i R_j[i,a] L_j[o,a], each with
        // three multiplications, so every rounding path has at most K·d_in·r + 3
        // operations (Higham γ).
        let growth = accumulation_growth(BASIS * INPUT_DIM * RANK + 3);
        let mut reversed_refuted = false;
        for row in 0..data.moments.nrows() {
            for o in 0..OUTPUT_DIM {
                let mut design_side = 0.0;
                for j in 0..BASIS {
                    for a in 0..RANK {
                        let mut projected = 0.0;
                        for i in 0..INPUT_DIM {
                            projected += data.inputs[[row, i]] * data.right[j][[i, a]];
                        }
                        design_side +=
                            data.moments[[row, j]] * projected * coefficients[[j * RANK + a, o]];
                    }
                }
                let contract = |left: &[Array2<f64>]| {
                    let mut value = 0.0;
                    let mut absolute = 0.0;
                    for j in 0..BASIS {
                        let mut inner = 0.0;
                        for i in 0..INPUT_DIM {
                            let mut entry = 0.0;
                            for a in 0..RANK {
                                entry += left[j][[o, a]] * data.right[j][[i, a]];
                                absolute += (data.moments[[row, j]]
                                    * left[j][[o, a]]
                                    * data.right[j][[i, a]]
                                    * data.inputs[[row, i]])
                                    .abs();
                            }
                            inner += entry * data.inputs[[row, i]];
                        }
                        value += data.moments[[row, j]] * inner;
                    }
                    (value, absolute)
                };
                let (tensor_side, absolute) = contract(fit.left_factors.as_slice());
                let band = 2.0 * growth * absolute;
                assert!(
                    (design_side - tensor_side).abs() <= band,
                    "row {row} output {o}: design order {design_side} vs Σ_j β_j L_j R_jᵀ h = \
                     {tensor_side} differ beyond the roundoff band {band}"
                );
                let (reversed_side, reversed_absolute) = contract(reversed.as_slice());
                if (design_side - reversed_side).abs() > 2.0 * growth * absolute.max(reversed_absolute) {
                    reversed_refuted = true;
                }
            }
        }
        assert!(
            reversed_refuted,
            "negative control: reading the blocks in reverse basis order must disagree with the \
             design beyond the roundoff band"
        );
    }

    #[test]
    fn block_penalty_is_the_function_space_penalty_of_each_output() {
        let data = fixture(&SUFFICIENCY_MASKS, generic_right_factors());
        let right = data.right_views();
        let fit = fit_gaussian_coefficient_block(data.rows(), &right, data.penalty.view())
            .expect("the full-rank sufficiency fixture is admitted and fitted");
        let columns = BASIS * RANK;
        let penalty = block_penalty(data.penalty.view(), &right);
        // Negative control: S ⊗ I_r, which wrongly treats every Gram R_jᵀ R_k as the identity.
        let gram_blind = Array2::from_shape_fn((columns, columns), |(a, b)| {
            if a % RANK == b % RANK {
                data.penalty[[a / RANK, b / RANK]]
            } else {
                0.0
            }
        });
        let coefficients = &fit.reml.coefficients;
        // Both sides sum the K²·d_in·r² monomials S_jk L_j[o,a] R_j[i,a] L_k[o,b] R_k[i,b],
        // each with four multiplications, so every rounding path has at most
        // K²·d_in·r² + 4 operations.
        let growth = accumulation_growth(BASIS * BASIS * INPUT_DIM * RANK * RANK + 4);
        let mut gram_blind_refuted = false;
        for o in 0..OUTPUT_DIM {
            let quadratic = |matrix: &Array2<f64>| {
                let mut value = 0.0;
                let mut absolute = 0.0;
                for a in 0..columns {
                    for b in 0..columns {
                        let term = coefficients[[a, o]] * matrix[[a, b]] * coefficients[[b, o]];
                        value += term;
                        absolute += term.abs();
                    }
                }
                (value, absolute)
            };
            let mut functional = 0.0;
            let mut absolute = 0.0;
            for j in 0..BASIS {
                for k in 0..BASIS {
                    for i in 0..INPUT_DIM {
                        let mut left_entry = 0.0;
                        let mut right_entry = 0.0;
                        for a in 0..RANK {
                            left_entry += fit.left_factors[j][[o, a]] * data.right[j][[i, a]];
                            right_entry += fit.left_factors[k][[o, a]] * data.right[k][[i, a]];
                        }
                        functional += data.penalty[[j, k]] * left_entry * right_entry;
                        for a in 0..RANK {
                            for b in 0..RANK {
                                absolute += (data.penalty[[j, k]]
                                    * fit.left_factors[j][[o, a]]
                                    * data.right[j][[i, a]]
                                    * fit.left_factors[k][[o, b]]
                                    * data.right[k][[i, b]])
                                    .abs();
                            }
                        }
                    }
                }
            }
            let band = 2.0 * growth * absolute;
            let (design, design_absolute) = quadratic(&penalty);
            assert!(
                (design - functional).abs() <= band,
                "output {o}: β'Pβ = {design} vs Σ_jk S_jk⟨L_j R_jᵀ, L_k R_kᵀ⟩ = {functional} \
                 differ beyond the roundoff band {band} (design-side absolute sum {design_absolute})"
            );
            let (blind, blind_absolute) = quadratic(&gram_blind);
            if (blind - functional).abs() > 2.0 * growth * absolute.max(blind_absolute) {
                gram_blind_refuted = true;
            }
        }
        assert!(
            gram_blind_refuted,
            "negative control: a penalty that ignores the right-factor Grams must disagree with \
             the function-space penalty beyond the roundoff band"
        );
    }

    #[test]
    fn identity_right_factors_reduce_to_the_dense_kronecker_block() {
        let identity: Vec<Array2<f64>> = (0..BASIS).map(|_| Array2::eye(INPUT_DIM)).collect();
        let data = fixture(&SUFFICIENCY_MASKS, identity);
        let right = data.right_views();
        let design = block_design(data.moments.view(), data.inputs.view(), &right);
        assert_eq!(
            design,
            dense_rowwise_kronecker(data.moments.view(), data.inputs.view()),
            "with R_j = I the design is β ⊗ h exactly: every product with an identity entry is exact"
        );
        let columns = BASIS * INPUT_DIM;
        let kronecker = Array2::from_shape_fn((columns, columns), |(a, b)| {
            if a % INPUT_DIM == b % INPUT_DIM {
                data.penalty[[a / INPUT_DIM, b / INPUT_DIM]]
            } else {
                0.0
            }
        });
        assert_eq!(
            block_penalty(data.penalty.view(), &right),
            kronecker,
            "with R_j = I the penalty is S ⊗ I exactly"
        );
        // Negative control: doubling one right factor moves the design.
        let doubled: Vec<Array2<f64>> = (0..BASIS)
            .map(|j| {
                if j == 0 {
                    Array2::eye(INPUT_DIM) * 2.0
                } else {
                    Array2::eye(INPUT_DIM)
                }
            })
            .collect();
        let doubled_views: Vec<ArrayView2<'_, f64>> =
            doubled.iter().map(|factor| factor.view()).collect();
        assert_ne!(
            block_design(data.moments.view(), data.inputs.view(), &doubled_views),
            design,
            "negative control: a non-identity right factor must change the design"
        );
    }

    #[test]
    fn rows_are_refused_on_the_declared_mask_never_on_a_vanishing_moment() {
        // Guard: all-on, and every component off with the residual removed, both
        // execute m_Δ Θ_* at every parameter value.
        for native_multiple in [ALL_ON, ([0.0; COMPONENTS], 0.0)] {
            let mut masks = SUFFICIENCY_MASKS.to_vec();
            masks.push(native_multiple);
            let data = fixture(&masks, generic_right_factors());
            let right = data.right_views();
            let refused = fit_gaussian_coefficient_block(data.rows(), &right, data.penalty.view());
            assert!(
                matches!(
                    refused,
                    Err(GaussianBlockError::NativeMultipleRow { row }) if row == 3 * INPUT_ROWS
                ),
                "mask {native_multiple:?}: the first native-multiple row must be refused, got {refused:?}"
            );
        }

        // Positive control: a residual-on ablation (component 3 removed from Θ_*).
        let mut ablation = SUFFICIENCY_MASKS.to_vec();
        ablation.push(([1.0, 1.0, 0.0], 1.0));
        let admitted = fixture(&ablation, generic_right_factors());
        let admitted_right = admitted.right_views();
        let fit =
            fit_gaussian_coefficient_block(admitted.rows(), &admitted_right, admitted.penalty.view());
        assert!(
            matches!(&fit, Ok(block) if block.reml.reml_score.is_finite()),
            "a residual-on ablation row must be admitted and fitted, got {fit:?}"
        );

        // A declared mask that differs from the residual mask, whose moment vanishes
        // by cancellation (as when v_1 + v_2 = 0 at the current labels and weights).
        // Its derivatives in the labels and weights do not vanish, so it is admitted.
        let mut cancelling = SUFFICIENCY_MASKS.to_vec();
        cancelling.push(([1.0, 1.0, 0.0], 0.0));
        let mut cancelled = fixture(&cancelling, generic_right_factors());
        cancelled.moments.row_mut(3 * INPUT_ROWS).fill(0.0);
        let cancelled_right = cancelled.right_views();
        let cancelled_fit = fit_gaussian_coefficient_block(
            cancelled.rows(),
            &cancelled_right,
            cancelled.penalty.view(),
        );
        assert!(
            matches!(&cancelled_fit, Ok(block) if block.reml.reml_score.is_finite()),
            "a row with a vanishing moment but a non-native mask must be admitted, got {cancelled_fit:?}"
        );
    }

    #[test]
    fn dense_block_admission_refuses_one_byte_below_its_ledger() {
        let rows = SUFFICIENCY_MASKS.len() * INPUT_ROWS;
        let ranks = [RANK; BASIS];
        let columns = BASIS * RANK;
        let expected = (rows * (columns + RANK) + 3 * columns * columns + columns * OUTPUT_DIM)
            * std::mem::size_of::<f64>();
        assert_eq!(
            dense_block_bytes(rows, &ranks, OUTPUT_DIM),
            Some(expected),
            "the ledger counts the design and widest projection, the penalty, the Gram and \
             eigenbasis, and the coefficients"
        );
        assert!(
            admit_dense_block(rows, &ranks, OUTPUT_DIM, expected).is_ok(),
            "a budget equal to the ledger admits the block"
        );
        let refused = admit_dense_block(rows, &ranks, OUTPUT_DIM, expected - 1);
        assert!(
            matches!(
                refused,
                Err(GaussianBlockError::AdmissionRefused { required_bytes: Some(required), budget_bytes })
                    if required == expected && budget_bytes == expected - 1
            ),
            "one byte below the ledger must refuse, got {refused:?}"
        );
        let overflow = admit_dense_block(usize::MAX, &[2], 1, usize::MAX);
        assert!(
            matches!(
                overflow,
                Err(GaussianBlockError::AdmissionRefused { required_bytes: None, .. })
            ),
            "an overflowing ledger must refuse, got {overflow:?}"
        );
    }
}
