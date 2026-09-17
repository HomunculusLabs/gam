//! Executed-stage receipts against the native lift (#2951 A12).
//!
//! A receipt compares one stage of a network executed by an external executor
//! (torch, through `gamfit.torch.parameter_interventions`) with the same stage
//! executed natively, entry by entry. Two IEEE evaluations `ŝ_ext` and `ŝ_nat` of
//! one exact stage value `s` satisfy `|ŝ_ext − ŝ_nat| ≤ |ŝ_ext − s| + |ŝ_nat − s|`,
//! so they agree when the discrepancy is at most the sum of two bands, each a
//! certified bound on its own evaluation's error.
//!
//! # The band of one evaluation
//!
//! A stage entry is a sum of terms, each a product of floats. Every rounded
//! operation on a term's path scales it by at most `1 ± u`, and a rounded product
//! applied to an accumulated sum scales every term inside it. With `k` the most
//! rounded operations on any term's path, the computed entry is within
//! `γ_k Σ|terms|` of the exact one, `γ_k = ku/(1 − ku)` (Higham, *Accuracy and
//! Stability of Numerical Algorithms*, 2nd ed., Lemma 3.1 and §3.1; the owner is
//! [`accumulation_growth`]). The order of additions inside one accumulation does
//! not change `k`: a binary tree over `n` leaves has depth at most `n − 1`, and a
//! fused multiply-add rounds once where the model counts two.
//!
//! `Σ|terms|` is accumulated under emulated upward rounding: each operation rounds
//! to nearest and then steps one float up, which lies at or above the exact result
//! of nonnegative operands. So the band never rests on a rounded-down magnitude.
//!
//! # What a band does not cover
//!
//! The model counts rounded binary64 operations in round-to-nearest. So a band is valid
//! only for an executor that evaluates the stage in IEEE binary64 with one rounding per
//! basic operation. A float32 or bfloat16 execution, or TF32 matrix multiplication,
//! rounds at a coarser precision that no binary64 band covers. A caller holding such an
//! execution refuses instead of comparing (mpd-verify on #2951).
//!
//! Two implementations of a special function (torch's `erf` against gam-math's
//! `erfc`) share no derivation here. A receipt evaluates the native activation at
//! the external executor's exported pre-activation, which isolates that difference
//! as a measured discrepancy. The measured discrepancy is reported beside the
//! band and never folded into it (#2951 correction C2).

use super::rewrite::ShapeMismatch;
use gam_linalg::roundoff::accumulation_growth;
use ndarray::{Array2, ArrayView1, ArrayView2};
use std::fmt;

/// A refused receipt computation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ReceiptRefusal {
    Shape(ShapeMismatch),
    /// An executed or band array has a non-finite entry.
    NonFinite {
        array: &'static str,
        row: usize,
        column: usize,
    },
}

impl From<ShapeMismatch> for ReceiptRefusal {
    fn from(mismatch: ShapeMismatch) -> Self {
        Self::Shape(mismatch)
    }
}

impl fmt::Display for ReceiptRefusal {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Shape(mismatch) => write!(formatter, "receipt shapes do not compose: {mismatch}"),
            Self::NonFinite { array, row, column } => write!(
                formatter,
                "the receipt's {array} has a non-finite entry at ({row}, {column})"
            ),
        }
    }
}

impl std::error::Error for ReceiptRefusal {}

fn check(what: &'static str, expected: usize, found: usize) -> Result<(), ShapeMismatch> {
    if expected == found {
        Ok(())
    } else {
        Err(ShapeMismatch {
            what,
            expected,
            found,
        })
    }
}

fn up(value: f64) -> f64 {
    value.next_up()
}

fn down(value: f64) -> f64 {
    value.next_down().max(0.0)
}

/// The certified band of one evaluation: `γ_k` rounded up times an upper bound on
/// the exact `Σ|terms|`, where `path_roundings = k` is the most rounded operations
/// on any term's path.
pub fn evaluation_band(path_roundings: usize, absolute_sum_upper: f64) -> f64 {
    up(up(accumulation_growth(path_roundings)) * absolute_sum_upper)
}

/// Per-entry band of one evaluation of `x Wᵀ + b` for each row `x` of `inputs`,
/// rows by outputs.
///
/// A term `w_jk x_k` rounds once as a product and then passes at most `d − 1`
/// additions among the products and one for the bias, so `k = d + 1` (`d` without
/// a bias).
pub fn affine_stage_band(
    weight: ArrayView2<'_, f64>,
    bias: Option<ArrayView1<'_, f64>>,
    inputs: ArrayView2<'_, f64>,
) -> Result<Array2<f64>, ShapeMismatch> {
    let (outputs, width) = weight.dim();
    check("affine input width", width, inputs.ncols())?;
    if let Some(bias) = bias {
        check("affine bias length", outputs, bias.len())?;
    }
    let path_roundings = width + usize::from(bias.is_some());
    let mut band = Array2::<f64>::zeros((inputs.nrows(), outputs));
    for (row, input) in inputs.outer_iter().enumerate() {
        for output in 0..outputs {
            let mut absolute_sum = bias.map_or(0.0, |bias| bias[output].abs());
            for (w, x) in weight.row(output).iter().zip(input.iter()) {
                absolute_sum = up(absolute_sum + up((w * x).abs()));
            }
            band[[row, output]] = evaluation_band(path_roundings, absolute_sum);
        }
    }
    Ok(band)
}

/// Per-entry band of one evaluation of `x (W + L diag(s) Rᵀ)ᵀ + b` for each row `x`
/// of `inputs`: a weight `W` (`p × d`) with a factored edit of `C` components, left
/// factor `L` (`p × C`), coefficients `s` and right factor `R` (`d × C`).
///
/// It covers both programs an executor runs:
/// * **dense:** form `W' = W + L diag(s) Rᵀ`, then `x W'ᵀ + b`;
/// * **matrix-free:** `x Wᵀ + ((x R) ⊙ s) Lᵀ + b`.
///
/// On either path a term `l_jc s_c r_kc x_k` rounds at most three times as a
/// product, `C − 1` times accumulating over components, `d − 1` times over inputs,
/// once where the edit meets `W` and once for the bias: `k = C + d + 3`. The terms
/// `w_jk x_k` take shorter paths.
pub fn factored_edit_stage_band(
    weight: ArrayView2<'_, f64>,
    left: ArrayView2<'_, f64>,
    coefficients: ArrayView1<'_, f64>,
    right: ArrayView2<'_, f64>,
    bias: Option<ArrayView1<'_, f64>>,
    inputs: ArrayView2<'_, f64>,
) -> Result<Array2<f64>, ShapeMismatch> {
    let (outputs, width) = weight.dim();
    let components = coefficients.len();
    check("edit left rows", outputs, left.nrows())?;
    check("edit left components", components, left.ncols())?;
    check("edit right rows", width, right.nrows())?;
    check("edit right components", components, right.ncols())?;
    check("edit input width", width, inputs.ncols())?;
    if let Some(bias) = bias {
        check("edit bias length", outputs, bias.len())?;
    }
    let mut band = Array2::<f64>::zeros((inputs.nrows(), outputs));
    let path_roundings = components + width + 3;
    let mut read = vec![0.0; components];
    for (row, input) in inputs.outer_iter().enumerate() {
        for (component, slot) in read.iter_mut().enumerate() {
            let mut absolute_sum = 0.0;
            for (r, x) in right.column(component).iter().zip(input.iter()) {
                absolute_sum = up(absolute_sum + up((r * x).abs()));
            }
            *slot = absolute_sum;
        }
        for output in 0..outputs {
            let mut absolute_sum = bias.map_or(0.0, |bias| bias[output].abs());
            for (w, x) in weight.row(output).iter().zip(input.iter()) {
                absolute_sum = up(absolute_sum + up((w * x).abs()));
            }
            for component in 0..components {
                let scale = up((left[[output, component]] * coefficients[component]).abs());
                absolute_sum = up(absolute_sum + up(scale * read[component]));
            }
            band[[row, output]] = evaluation_band(path_roundings, absolute_sum);
        }
    }
    Ok(band)
}

/// How one executed stage compares with its native execution.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StageAgreement {
    /// Every entry has `|external − native| ≤ external_band + native_band`, decided
    /// with the discrepancy rounded up and the band sum rounded down: a certified
    /// agreement.
    pub agrees: bool,
    /// Some entry has `|external − native| > external_band + native_band`, decided
    /// with the discrepancy rounded down and the band sum rounded up: a certified
    /// violation. When neither `agrees` nor `refutes` holds, an entry sits inside the
    /// rounding of its own comparison and the stage is unresolved.
    pub refutes: bool,
    /// `(row, column)` of the entry with the largest discrepancy-to-band ratio.
    pub witness: (usize, usize),
    /// The discrepancy at the witness, rounded up.
    pub discrepancy: f64,
    /// The band sum at the witness.
    pub band: f64,
    /// `discrepancy / band` at the witness. It is for reports only: `agrees` is
    /// decided entry by entry without this division.
    pub ratio: f64,
}

fn require_finite(array: &'static str, values: ArrayView2<'_, f64>) -> Result<(), ReceiptRefusal> {
    match values.indexed_iter().find(|(_, value)| !value.is_finite()) {
        Some(((row, column), _)) => Err(ReceiptRefusal::NonFinite { array, row, column }),
        None => Ok(()),
    }
}

/// Compares an external executor's stage output with the native one against their
/// bands.
pub fn compare_stage(
    external: ArrayView2<'_, f64>,
    native: ArrayView2<'_, f64>,
    external_band: ArrayView2<'_, f64>,
    native_band: ArrayView2<'_, f64>,
) -> Result<StageAgreement, ReceiptRefusal> {
    let (rows, columns) = native.dim();
    for (what, array) in [
        ("external", external),
        ("external band", external_band),
        ("native band", native_band),
    ] {
        check(what, rows, array.nrows())?;
        check(what, columns, array.ncols())?;
    }
    for (what, array) in [
        ("external output", external),
        ("native output", native),
        ("external band", external_band),
        ("native band", native_band),
    ] {
        require_finite(what, array)?;
    }
    let mut agreement = StageAgreement {
        agrees: true,
        refutes: false,
        witness: (0, 0),
        discrepancy: 0.0,
        band: 0.0,
        ratio: f64::NEG_INFINITY,
    };
    for ((row, column), &native_value) in native.indexed_iter() {
        // A finite difference of floats is zero only when they are equal, and then it
        // is exact. Otherwise the exact difference lies between the rounded one's
        // neighbours, and so does the exact band sum.
        let difference = (external[[row, column]] - native_value).abs();
        let band_sum = external_band[[row, column]] + native_band[[row, column]];
        let (discrepancy, discrepancy_lower) = if difference == 0.0 {
            (0.0, 0.0)
        } else {
            (up(difference), down(difference))
        };
        let band = down(band_sum);
        agreement.agrees &= discrepancy <= band;
        agreement.refutes |= discrepancy_lower > up(band_sum);
        let ratio = if discrepancy == 0.0 { 0.0 } else { discrepancy / band };
        if ratio > agreement.ratio {
            agreement.witness = (row, column);
            agreement.discrepancy = discrepancy;
            agreement.band = band;
            agreement.ratio = ratio;
        }
    }
    Ok(agreement)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, array};
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};

    fn uniform(rng: &mut StdRng, rows: usize, cols: usize) -> Array2<f64> {
        Array2::from_shape_simple_fn((rows, cols), || rng.random_range(-1.0..1.0))
    }

    /// `x Wᵀ + b` accumulated left to right.
    fn affine_forward(weight: &Array2<f64>, inputs: &Array2<f64>) -> Array2<f64> {
        Array2::from_shape_fn((inputs.nrows(), weight.nrows()), |(row, output)| {
            let mut sum = 0.0;
            for input in 0..weight.ncols() {
                sum += weight[[output, input]] * inputs[[row, input]];
            }
            sum
        })
    }

    /// `x Wᵀ + b` accumulated right to left.
    fn affine_reverse(weight: &Array2<f64>, inputs: &Array2<f64>) -> Array2<f64> {
        Array2::from_shape_fn((inputs.nrows(), weight.nrows()), |(row, output)| {
            let mut sum = 0.0;
            for input in (0..weight.ncols()).rev() {
                sum += weight[[output, input]] * inputs[[row, input]];
            }
            sum
        })
    }

    /// Two summation orders of the row `[2⁵³, 1, …, 1, −2⁵³]` against ones differ by
    /// fourteen: left to right each `+1` ties to even and is lost, right to left every
    /// partial sum is exact. The depth-`d` band covers it. Positive control: a band
    /// with the path depth dropped to one does not, so the depth count is load-bearing.
    #[test]
    fn affine_band_encloses_two_summation_orders_of_a_cancelling_row() {
        let width = 16;
        let mut weight = Array2::<f64>::ones((1, width));
        weight[[0, 0]] = 2.0_f64.powi(53);
        weight[[0, width - 1]] = -2.0_f64.powi(53);
        let inputs = Array2::<f64>::ones((1, width));
        let forward = affine_forward(&weight, &inputs);
        let reverse = affine_reverse(&weight, &inputs);
        assert_eq!((forward[[0, 0]], reverse[[0, 0]]), (0.0, 14.0));

        let band = affine_stage_band(weight.view(), None, inputs.view()).expect("shapes compose");
        let agreement = compare_stage(forward.view(), reverse.view(), band.view(), band.view())
            .expect("finite arrays");
        assert!(agreement.agrees && !agreement.refutes, "{agreement:?}");

        let absolute_sum = 2.0_f64.powi(54) + 14.0;
        let shallow = Array2::from_elem((1, 1), evaluation_band(1, absolute_sum));
        let refuted = compare_stage(forward.view(), reverse.view(), shallow.view(), shallow.view())
            .expect("finite arrays");
        assert!(
            !refuted.agrees && refuted.refutes,
            "a depth-one band must be refuted by fourteen: {refuted:?}"
        );
    }

    /// A factored edit executed densely and matrix-free, under signed and continuous
    /// coefficients, agrees within the factored-edit band. Positive control: the native
    /// output displaced at one entry by twice its band sum is refuted at that entry.
    #[test]
    fn factored_edit_band_encloses_dense_and_matrix_free_programs() {
        let (outputs, width, components, rows) = (5, 7, 3, 4);
        let mut rng = StdRng::seed_from_u64(2951);
        let weight = uniform(&mut rng, outputs, width);
        let left = uniform(&mut rng, outputs, components);
        let right = uniform(&mut rng, width, components);
        let inputs = uniform(&mut rng, rows, width);
        let bias: Array1<f64> = Array1::from_shape_simple_fn(outputs, || rng.random_range(-1.0..1.0));
        let coefficients = array![-1.0, -0.37, 0.5];

        let mut edited = weight.clone();
        for output in 0..outputs {
            for input in 0..width {
                let mut edit = 0.0;
                for component in 0..components {
                    edit += left[[output, component]] * coefficients[component] * right[[input, component]];
                }
                edited[[output, input]] += edit;
            }
        }
        let dense = affine_forward(&edited, &inputs) + &bias;

        let read = affine_forward(&right.t().to_owned(), &inputs) * &coefficients;
        let matrix_free = affine_forward(&weight, &inputs) + affine_forward(&left, &read) + &bias;

        let band = factored_edit_stage_band(
            weight.view(),
            left.view(),
            coefficients.view(),
            right.view(),
            Some(bias.view()),
            inputs.view(),
        )
        .expect("shapes compose");
        let agreement = compare_stage(dense.view(), matrix_free.view(), band.view(), band.view())
            .expect("finite arrays");
        assert!(agreement.agrees, "{agreement:?}");

        let mut displaced = matrix_free.clone();
        displaced[[2, 1]] += 2.0 * (band[[2, 1]] + band[[2, 1]]);
        let refuted = compare_stage(dense.view(), displaced.view(), band.view(), band.view())
            .expect("finite arrays");
        assert!(!refuted.agrees && refuted.refutes);
        assert_eq!(refuted.witness, (2, 1));
    }

    /// A discrepancy of exactly its band sum is decided by rounding alone, so neither
    /// agreement nor refutation is certified. Positive controls: a band sum of twice the
    /// discrepancy agrees, and one of half the discrepancy refutes.
    #[test]
    fn compare_stage_leaves_a_boundary_entry_unresolved() {
        let external = Array2::from_elem((1, 1), 0.0);
        let native = Array2::from_elem((1, 1), 1.0);
        let verdict = |half_band: f64| {
            let band = Array2::from_elem((1, 1), half_band);
            compare_stage(external.view(), native.view(), band.view(), band.view())
                .expect("finite arrays")
        };

        let boundary = verdict(0.5);
        assert!(!boundary.agrees && !boundary.refutes, "{boundary:?}");

        let wide = verdict(1.0);
        assert!(wide.agrees && !wide.refutes, "{wide:?}");

        let narrow = verdict(0.25);
        assert!(!narrow.agrees && narrow.refutes, "{narrow:?}");
    }

    /// Mismatched shapes and non-finite entries are refused, never compared.
    /// Positive control: the same arrays with matching shapes and finite entries compare.
    #[test]
    fn compare_stage_refuses_mismatched_shapes_and_non_finite_entries() {
        let values = Array2::<f64>::zeros((2, 3));
        let band = Array2::<f64>::zeros((2, 3));
        assert!(compare_stage(values.view(), values.view(), band.view(), band.view()).is_ok());

        let narrow = Array2::<f64>::zeros((2, 2));
        assert!(matches!(
            compare_stage(narrow.view(), values.view(), band.view(), band.view()),
            Err(ReceiptRefusal::Shape(..))
        ));

        let mut poisoned = values.clone();
        poisoned[[1, 2]] = f64::NAN;
        assert_eq!(
            compare_stage(poisoned.view(), values.view(), band.view(), band.view()),
            Err(ReceiptRefusal::NonFinite {
                array: "external output",
                row: 1,
                column: 2,
            })
        );
    }
}
