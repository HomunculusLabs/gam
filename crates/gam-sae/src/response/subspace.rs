//! The best retained response of a known MLP block, and the exact error of discarding the rest of its input
//! (#2946 R1, R2, R4, R6).
//!
//! # Entry points
//!
//! - [`KnownBlock::new`]`(readers, biases, writers, output_bias, metric, activation)` builds the block and caches
//!   `V(I)`.
//! - [`KnownBlock::retained_response`]`(frame, points)` is R1 at ambient points, output bias included.
//! - [`KnownBlock::explained_variance`], [`KnownBlock::discarded_error`] and
//!   [`KnownBlock::explained_variance_of_coordinates`] are R4 on a frame or on a coordinate set.
//! - [`KnownBlock::explained_variance_gradient`] is R6.
//! - [`covariance_rounding_band`] and [`frame_defect`] own the Cauchy–Schwarz rounding band a caller states to the
//!   pair kernels.
//!
//! # Block, law and frame
//!
//! The block is `F(z) = Σ_j u_j σ(b_j + w_jᵀ z) + c` with readers `W ∈ ℝ^{h×d}` (rows `w_jᵀ`), biases `b ∈ ℝ^h`,
//! writers `U ∈ ℝ^{p×h}` (columns `u_j`), output bias `c ∈ ℝ^p` and activation `σ`. The input law is `Z ~ N(0, I_d)`,
//! the output metric is `M ≻ 0`, and a frame `Q ∈ ℝ^{d×k}` with orthonormal columns spans the retained input
//! subspace, `P = Q Qᵀ`. The output bias moves every response and no variance.
//!
//! # R1, the best retained response
//!
//! `PZ` and `(I − P)Z` are independent, so the conditional mean given the retained input is
//!
//! ```text
//! F̄_P(Pz) = E[F(Z) | PZ = Pz] = Σ_j u_j T_{v_j⊥} σ(b_j + w_jᵀ P z) + c,   v_j⊥ = w_jᵀ (I − P) w_j,
//! ```
//!
//! with `T_v σ(t) = E σ(t + √v E)` the Gaussian smoothing of the activation.
//!
//! # R2, the error split
//!
//! For every `g`, `E‖F − g(PZ)‖²_M = E‖F − F̄_P‖²_M + E‖F̄_P − g(PZ)‖²_M`, because `F − F̄_P` is orthogonal to every
//! function of `PZ`. The discarded-input error and the fit error of a compact response are separate numbers.
//!
//! # R4, explained variance and discarded error
//!
//! Couple `Z' = PZ + (I − P)Z̃` with `Z̃` an independent copy of `Z`. Then `F̄_P(PZ) = E[F(Z') | Z]`, so
//! `E⟨F̄_P − c, F̄_P − c⟩_M = E⟨F(Z) − c, F(Z') − c⟩_M`, and `(b_j + w_jᵀZ, b_k + w_kᵀZ')` is jointly normal with
//! variances `v_j = ‖w_j‖²`, `v_k` and covariance `w_jᵀ P w_k`. Hence
//!
//! ```text
//! V(P) = E‖F̄_P − E F‖²_M = Σ_jk D_jk [K_σ(b_j, b_k; v_j, v_k, w_jᵀ P w_k) − m_j m_k],
//! ```
//!
//! with `D_jk = u_jᵀ M u_k`, `K_σ = E σ(X) σ(Y)` and `m_j = T_{v_j} σ(b_j)`. The cross terms `j ≠ k` are required.
//! R2 with `g = E F` gives `E(P) = E‖F − F̄_P‖²_M = V(I) − V(P) ≥ 0`. At `P = 0` the pair is independent and every
//! bracket vanishes, so `V(0) = 0` and `E(0) = V(I)`; `E(I) = 0`.
//!
//! # R6, the frame gradient
//!
//! Price's theorem gives `∂_r K_σ = E σ'(X) σ'(Y)`. With `B_jk = D_jk ∂_r K_σ(b_j, b_k; v_j, v_k, w_jᵀ P w_k)`, which
//! is symmetric, `dV = Σ_jk B_jk w_jᵀ (dQ Qᵀ + Q dQᵀ) w_k = 2 tr(dQᵀ Wᵀ B W Q)`. `V` depends on `Q` only through `P`,
//! so the Grassmann (horizontal) gradient is
//!
//! ```text
//! ∇_Q V = 2 (I − Q Qᵀ) Wᵀ B(P) W Q.
//! ```
//!
//! # Tiles
//!
//! The frame enters only through `R = W Q` (`h × k`) and the metric only through `D = Uᵀ M U`. `D` is read over
//! `j ≤ k` from the block's [`ReaderGram`], or, when the process-wide ledger declines that cache, streamed per tile
//! by [`fill_upper_rows`] over the same tiles, so both routes read the same words. `D` and the pair law are both
//! symmetric, so a pass evaluates only the pairs `k ≥ j` and counts each strictly upper term twice. A tile `J` of `t`
//! units starting at `s` forms `R_J R_{k≥s}ᵀ` and its weights `B_J` over `k ≥ s` (each `t × (h − s)`). It adds
//! `B_J R_{k≥s}` to its own rows of `B R`, and the strictly upper part transposed, `B_Jᵀ R_J`, to the rows `k ≥ s`.
//! No `d × d` matrix is formed, and the only resident `h × h` object is the packed half of `D`, charged to the
//! memory governor. The gradient is then `Wᵀ(B R) − Q (Qᵀ Wᵀ (B R))`.
//!
//! # Rounding of the pair law
//!
//! The exact law of a pair has `|w_jᵀ P w_k| ≤ √(v_j v_k)`. The computed triple `(v̂_j, v̂_k, r̂_jk)` can leave that
//! cone only by the rounding of its formation and by a frame's orthonormality defect, and the kernel projects a
//! covariance within the stated band [`covariance_rounding_band`] onto the boundary and refuses one beyond it. A frame
//! whose measured defect reaches 1 is not a frame and is refused ([`ResponseError::FrameNotOrthonormal`]).

use super::reader_gram::{ReaderGram, ReaderGramError, fill_upper_rows, upper_tile_rows};
use faer::Side;
use gam_linalg::faer_ndarray::{FaerCholesky, fast_ab, fast_abt, fast_atb};
use gam_linalg::roundoff::accumulation_growth;
use gam_math::gaussian_activation::{
    GaussianActivation, GaussianActivationError, PairKernel, PreactivationPair, gaussian_smoothing_derivatives,
    pair_kernel,
};
use gam_runtime::resource::byte_balanced_row_chunk;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis, s};
use rayon::prelude::*;
use std::fmt;
use std::sync::Arc;

/// A refusal of the retained-response operator.
#[derive(Debug, Clone, PartialEq)]
pub enum ResponseError {
    DimensionMismatch {
        context: &'static str,
        expected: usize,
        got: usize,
    },
    NonFinite {
        context: &'static str,
    },
    /// The metric must be exactly symmetric: `V` reads only its symmetric part, but the gradient formula `2 Wᵀ B W Q`
    /// holds only when `B`, and so `D = Uᵀ M U`, is symmetric.
    MetricNotSymmetric {
        row: usize,
        column: usize,
    },
    MetricNotPositiveDefinite {
        reason: String,
    },
    /// A frame of rank `k` in `ℝ^d` needs `k ≤ d` orthonormal columns.
    FrameWiderThanInput {
        rank: usize,
        input_dim: usize,
    },
    /// The frame's measured orthonormality defect [`frame_defect`] is at least 1, so it has no nearest orthonormal
    /// frame whose projector the operator could be exact for.
    FrameNotOrthonormal {
        defect: f64,
    },
    /// Retained coordinates must be strictly increasing and below the input dimension, so they name a projector.
    InvalidRetainedCoordinates {
        position: usize,
    },
    Kernel {
        context: &'static str,
        error: GaussianActivationError,
    },
    /// The reader Gram refused for a reason other than the ledger declining its footprint, which the block answers
    /// by streaming `D`.
    ReaderGram {
        error: ReaderGramError,
    },
}

impl fmt::Display for ResponseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DimensionMismatch {
                context,
                expected,
                got,
            } => write!(f, "{context}: expected {expected}, got {got}"),
            Self::NonFinite { context } => write!(f, "{context} holds a non-finite entry"),
            Self::MetricNotSymmetric { row, column } => write!(
                f,
                "output metric is not symmetric: entry ({row}, {column}) differs from its transpose"
            ),
            Self::MetricNotPositiveDefinite { reason } => {
                write!(f, "output metric is not positive definite: {reason}")
            }
            Self::FrameWiderThanInput { rank, input_dim } => write!(
                f,
                "a retained frame of rank {rank} cannot be orthonormal in input dimension {input_dim}"
            ),
            Self::FrameNotOrthonormal { defect } => write!(
                f,
                "retained frame is not orthonormal: its measured defect {defect} is at least 1"
            ),
            Self::InvalidRetainedCoordinates { position } => write!(
                f,
                "retained coordinate at position {position} is out of range or not strictly increasing"
            ),
            Self::Kernel { context, error } => write!(f, "{context}: {error}"),
            Self::ReaderGram { error } => write!(f, "reader Gram: {error}"),
        }
    }
}

impl std::error::Error for ResponseError {}

/// `V(P)`, `E(P)` and the horizontal frame gradient of `V` at one frame.
#[derive(Debug, Clone)]
pub struct FrameGradient {
    /// `V(P)`, the output variance the retained response explains.
    pub explained_variance: f64,
    /// `E(P) = V(I) − V(P)`, the error of discarding the input outside the frame.
    pub discarded_error: f64,
    /// `∇_Q V = 2 (I − Q Qᵀ) Wᵀ B(P) W Q`, a `d × k` horizontal tangent: the ascent direction of `V` and the descent
    /// direction of `E`.
    pub horizontal_gradient: Array2<f64>,
}

/// An energy together with a bound on its absolute error: the vocabulary every producer of a variance in `response/`
/// reports in, so a consumer decides "resolved from zero" against a derived band instead of a hand threshold.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BandedEnergy {
    /// The computed value.
    pub value: f64,
    /// A bound on `|computed − exact|`: the arithmetic's rounding plus the producer's accuracy contract.
    pub band: f64,
}

impl BandedEnergy {
    /// The exact zero, `V(∅)`.
    pub const ZERO: BandedEnergy = BandedEnergy {
        value: 0.0,
        band: 0.0,
    };

    /// Whether the exact value is resolved as positive: the computed value clears its band.
    pub fn resolved_positive(&self) -> bool {
        self.value > self.band
    }
}

/// `Σ added − Σ subtracted` with its band: the operands' bands plus `γ_{n−1} Σ|operand|`, the rounding of `n − 1`
/// additions of pre-formed terms (Higham, ASNA §3.1), to first order in `u`.
pub fn signed_sum(added: &[BandedEnergy], subtracted: &[BandedEnergy]) -> BandedEnergy {
    let mut value = 0.0;
    let mut absolute = 0.0;
    let mut band = 0.0;
    for term in added {
        value += term.value;
        absolute += term.value.abs();
        band += term.band;
    }
    for term in subtracted {
        value -= term.value;
        absolute += term.value.abs();
        band += term.band;
    }
    let operations = (added.len() + subtracted.len()).saturating_sub(1);
    BandedEnergy {
        value,
        band: band + accumulation_growth(operations) * absolute,
    }
}

/// How a computed covariance `r̂ = fl(Σ_a left_a right_a)` was formed, for [`covariance_rounding_band`]. Every field is
/// an absolute bound in the covariance's own units.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CovarianceFormation {
    /// The number of products in `r̂`.
    pub terms: usize,
    /// `‖left‖` and `‖right‖` of the computed rows.
    pub left_norm: f64,
    pub right_norm: f64,
    /// Bounds on the Euclidean distance from each computed row to the row formed exactly from the same inputs.
    pub left_row_error: f64,
    pub right_row_error: f64,
    /// A bound on `|left_exact · right_exact − r|`, with `r` the law's covariance: `0` when the exact rows are the law's
    /// own rows, and a frame's projector gap otherwise.
    pub law_gap: f64,
    /// Bounds on `|v̂ − v|` of the two computed variances handed to the kernel.
    pub variance_error_x: f64,
    pub variance_error_y: f64,
}

/// The `covariance_rounding` of a computed pair law: a bound on how far `|r̂|` can exceed `√(v̂_x v̂_y)` when the exact
/// law satisfies `|r| ≤ √(v_x v_y)`.
///
/// `|r̂| ≤ |r| + |r̂ − r|` and `√(v_x v_y) ≤ √((v̂_x + e_x)(v̂_y + e_y))`, while
/// `|r̂ − r| ≤ γ_terms ‖left‖‖right‖ + ‖δ_left‖‖right‖ + ‖left‖‖δ_right‖ + ‖δ_left‖‖δ_right‖ + law_gap`: the
/// recursive rounding of the product sum (Higham, ASNA §3.1), the propagated row formation errors `δ`, and the gap
/// between the exactly formed rows and the law. The band is their sum.
pub fn covariance_rounding_band(formation: &CovarianceFormation, variance_x: f64, variance_y: f64) -> f64 {
    let product_rounding = accumulation_growth(formation.terms) * formation.left_norm * formation.right_norm;
    let row_rounding = formation.left_row_error * formation.right_norm
        + formation.left_norm * formation.right_row_error
        + formation.left_row_error * formation.right_row_error;
    let variance_rounding = ((variance_x + formation.variance_error_x)
        * (variance_y + formation.variance_error_y))
        .sqrt()
        - (variance_x * variance_y).sqrt();
    product_rounding + row_rounding + formation.law_gap + variance_rounding
}

/// A measured bound `η ≥ ‖QᵀQ − I‖₂` on the orthonormality defect of a `d × k` frame, or infinity when no bound is
/// available.
///
/// The computed Gram `Ĝ = fl(QᵀQ)` errs entrywise by at most `γ_d ‖q_a‖‖q_b‖ ≤ γ_d (1 + η)`, so
/// `‖QᵀQ − Ĝ‖_F ≤ k γ_d (1 + η)`. The Frobenius norm of `Ĝ − I` rounds by at most `γ_{k²+1}` relative, so
/// `η ≤ n̂ (1 + γ_{k²+1}) + k γ_d (1 + η)`, which solves to the returned bound.
pub fn frame_defect(frame: ArrayView2<'_, f64>) -> f64 {
    let (input_dim, rank) = frame.dim();
    let formation = rank as f64 * accumulation_growth(input_dim);
    if !(formation < 1.0) {
        return f64::INFINITY;
    }
    let gram = fast_atb(&frame, &frame);
    let mut squares = 0.0;
    for row in 0..rank {
        for column in 0..rank {
            let entry = gram[[row, column]] - if row == column { 1.0 } else { 0.0 };
            squares += entry * entry;
        }
    }
    let measured = squares.sqrt() * (1.0 + accumulation_growth(rank * rank + 1));
    (measured + formation) / (1.0 - formation)
}

/// How a pair pass's reader coordinates were formed.
#[derive(Debug, Clone, Copy)]
enum CoordinateFormation {
    /// The coordinates are reader entries themselves: the readers, or selected reader columns.
    Copied,
    /// `R = fl(W Q)` for a frame with measured defect `frame_defect`.
    FrameProducts { frame_defect: f64 },
}

/// A known MLP block under the declared law `Z ~ N(0, I_d)`, with its output metric.
#[derive(Debug, Clone)]
pub struct KnownBlock {
    units: BlockUnits,
    total_variance: f64,
}

/// The per-unit data every pair term reads.
#[derive(Debug, Clone)]
struct BlockUnits {
    /// `W`, `h × d`.
    readers: Array2<f64>,
    /// `b`, length `h`.
    biases: Array1<f64>,
    /// `U`, `p × h`.
    writers: Array2<f64>,
    /// `c`, length `p`.
    output_bias: Array1<f64>,
    /// `M`, `p × p`.
    metric: Array2<f64>,
    /// `M U`, `p × h`, so a tile's `D_J = U_Jᵀ (M U)`.
    metric_writers: Array2<f64>,
    /// `D = Uᵀ M U` over `j ≤ k` when the process-wide ledger admitted it; `None` streams the same rows per tile.
    /// Clones of the block share one charged cache.
    reader_gram: Option<Arc<ReaderGram>>,
    activation: GaussianActivation,
    /// `v̂_j = fl(‖w_j‖²)`.
    reader_variances: Array1<f64>,
    /// `|v̂_j − v_j| ≤ γ_d v_j ≤ γ_d v̂_j / (1 − γ_d)`: the rounding of a `d`-term sum of squares.
    reader_variance_errors: Array1<f64>,
    /// `m_j = T_{v_j} σ(b_j) = E σ(b_j + w_jᵀ Z)`.
    unit_means: Array1<f64>,
}

impl KnownBlock {
    /// Build the block from readers `W` (`h × d`), biases `b` (`h`), writers `U` (`p × h`), the output bias `c`
    /// (`p`; zeros for a layer without one), a symmetric positive definite output metric `M` (`p × p`) and the
    /// activation. The total variance `V(I) = E‖F − E F‖²_M` is computed once here, since every `E(P)` reads it.
    pub fn new(
        readers: Array2<f64>,
        biases: Array1<f64>,
        writers: Array2<f64>,
        output_bias: Array1<f64>,
        metric: ArrayView2<'_, f64>,
        activation: GaussianActivation,
    ) -> Result<Self, ResponseError> {
        let (width, input_dim) = readers.dim();
        require_length("block biases", width, biases.len())?;
        require_length("block writer columns", width, writers.ncols())?;
        let output_dim = writers.nrows();
        require_length("block output bias", output_dim, output_bias.len())?;
        require_length("output metric rows", output_dim, metric.nrows())?;
        require_length("output metric columns", output_dim, metric.ncols())?;
        require_finite("block readers", readers.iter())?;
        require_finite("block biases", biases.iter())?;
        require_finite("block writers", writers.iter())?;
        require_finite("block output bias", output_bias.iter())?;
        require_finite("output metric", metric.iter())?;
        for row in 0..output_dim {
            for column in (row + 1)..output_dim {
                if metric[[row, column]] != metric[[column, row]] {
                    return Err(ResponseError::MetricNotSymmetric { row, column });
                }
            }
        }
        metric
            .cholesky(Side::Lower)
            .map_err(|error| ResponseError::MetricNotPositiveDefinite {
                reason: error.to_string(),
            })?;
        let metric_writers = fast_ab(&metric, &writers);
        // `M U` of finite factors can still overflow. Both routes to `D` read it, so it is refused here, once.
        require_finite("output metric times writers", metric_writers.iter())?;
        let reader_gram = match ReaderGram::new(writers.view(), metric_writers.view()) {
            Ok(gram) => Some(Arc::new(gram)),
            // A footprint the ledger declines, or one with no representable byte count, streams the same rows instead.
            Err(ReaderGramError::Admission { .. } | ReaderGramError::SizeOverflow { .. }) => None,
            Err(error) => return Err(ResponseError::ReaderGram { error }),
        };
        let reader_variances: Array1<f64> = readers.rows().into_iter().map(|row| row.dot(&row)).collect();
        let variance_growth = accumulation_growth(input_dim);
        let reader_variance_errors = reader_variances.mapv(|variance| variance_growth * variance / (1.0 - variance_growth));
        let unit_means = (0..width)
            .map(|unit| smoothing(activation, reader_variances[unit], biases[unit]))
            .collect::<Result<Array1<f64>, ResponseError>>()?;
        let units = BlockUnits {
            readers,
            biases,
            writers,
            output_bias,
            metric: metric.to_owned(),
            metric_writers,
            reader_gram,
            activation,
            reader_variances,
            reader_variance_errors,
            unit_means,
        };
        let total_variance = units.pair_pass(units.readers.view(), CoordinateFormation::Copied, None)?;
        Ok(Self {
            units,
            total_variance,
        })
    }

    /// Input dimension `d`.
    pub fn input_dim(&self) -> usize {
        self.units.readers.ncols()
    }

    /// Unit count `h`.
    pub fn width(&self) -> usize {
        self.units.readers.nrows()
    }

    /// Output dimension `p`.
    pub fn output_dim(&self) -> usize {
        self.units.writers.nrows()
    }

    pub fn activation(&self) -> GaussianActivation {
        self.units.activation
    }

    /// Readers `W`, `h × d`.
    pub fn readers(&self) -> ArrayView2<'_, f64> {
        self.units.readers.view()
    }

    /// Biases `b`, length `h`.
    pub fn biases(&self) -> ArrayView1<'_, f64> {
        self.units.biases.view()
    }

    /// Writers `U`, `p × h`.
    pub fn writers(&self) -> ArrayView2<'_, f64> {
        self.units.writers.view()
    }

    /// The output bias `c`, length `p`.
    pub fn output_bias(&self) -> ArrayView1<'_, f64> {
        self.units.output_bias.view()
    }

    /// The output metric `M`, `p × p`.
    pub fn metric(&self) -> ArrayView2<'_, f64> {
        self.units.metric.view()
    }

    /// `M U`, `p × h`.
    pub fn metric_writers(&self) -> ArrayView2<'_, f64> {
        self.units.metric_writers.view()
    }

    /// `V(I) = E‖F − E F‖²_M`, the output variance of the whole block.
    pub fn total_variance(&self) -> f64 {
        self.total_variance
    }

    /// R1: the best retained response `F̄_P(Pz)`, output bias included, at each row `z` of `points` (`n × d`), as an
    /// `n × p` matrix comparable with the executed block's outputs.
    pub fn retained_response(
        &self,
        frame: ArrayView2<'_, f64>,
        points: ArrayView2<'_, f64>,
    ) -> Result<Array2<f64>, ResponseError> {
        self.require_frame(frame)?;
        require_length("retained-response point columns", self.input_dim(), points.ncols())?;
        require_finite("retained-response points", points.iter())?;
        let units = &self.units;
        let coordinates = fast_ab(&units.readers, &frame);
        let discarded_variances = self.discarded_reader_variances(frame, coordinates.view());
        let rows = points.nrows();
        let mut response = Array2::<f64>::zeros((rows, self.output_dim()));
        let chunk = byte_balanced_row_chunk(self.width() + self.input_dim(), rows);
        for start in (0..rows).step_by(chunk) {
            let end = (start + chunk).min(rows);
            // Row `i` of `retained` is `zᵢᵀ Q`, so entry `(i, j)` of the product is `w_jᵀ P zᵢ`.
            let retained = fast_ab(&points.slice(s![start..end, ..]), &frame);
            let mut smoothed = fast_abt(&retained, &coordinates);
            smoothed
                .axis_iter_mut(Axis(0))
                .into_par_iter()
                .try_for_each(|mut row| {
                    for unit in 0..row.len() {
                        row[unit] = smoothing(
                            units.activation,
                            discarded_variances[unit],
                            units.biases[unit] + row[unit],
                        )?;
                    }
                    Ok::<(), ResponseError>(())
                })?;
            let mut tile_response = fast_abt(&smoothed, &units.writers);
            tile_response += &units.output_bias;
            response.slice_mut(s![start..end, ..]).assign(&tile_response);
        }
        Ok(response)
    }

    /// R4: `V(P)`, the output variance explained by the best response of the input inside `frame` (`d × k`).
    pub fn explained_variance(&self, frame: ArrayView2<'_, f64>) -> Result<f64, ResponseError> {
        let defect = self.require_frame(frame)?;
        let coordinates = fast_ab(&self.units.readers, &frame);
        self.units.pair_pass(
            coordinates.view(),
            CoordinateFormation::FrameProducts {
                frame_defect: defect,
            },
            None,
        )
    }

    /// R4 on a coordinate frame: `V(P_S)` for the projector onto the input coordinates `retained`, given strictly
    /// increasing. The covariances `r_jk = Σ_{i ∈ S} W_ji W_ki` are read from the selected reader columns, so no frame
    /// is formed.
    pub fn explained_variance_of_coordinates(&self, retained: &[usize]) -> Result<f64, ResponseError> {
        for (position, &coordinate) in retained.iter().enumerate() {
            let increasing = position == 0 || retained[position - 1] < coordinate;
            if coordinate >= self.input_dim() || !increasing {
                return Err(ResponseError::InvalidRetainedCoordinates { position });
            }
        }
        let coordinates = self.units.readers.select(Axis(1), retained);
        self.units
            .pair_pass(coordinates.view(), CoordinateFormation::Copied, None)
    }

    /// R4: `E(P) = V(I) − V(P)`, the error of discarding the input outside `frame`.
    pub fn discarded_error(&self, frame: ArrayView2<'_, f64>) -> Result<f64, ResponseError> {
        Ok(self.total_variance - self.explained_variance(frame)?)
    }

    /// R6: `V(P)`, `E(P)` and the horizontal gradient `2 (I − Q Qᵀ) Wᵀ B(P) W Q`, from one pass over the unit pairs.
    pub fn explained_variance_gradient(
        &self,
        frame: ArrayView2<'_, f64>,
    ) -> Result<FrameGradient, ResponseError> {
        let defect = self.require_frame(frame)?;
        let units = &self.units;
        let coordinates = fast_ab(&units.readers, &frame);
        let mut weighted_coordinates = Array2::<f64>::zeros(coordinates.dim());
        let explained_variance = units.pair_pass(
            coordinates.view(),
            CoordinateFormation::FrameProducts {
                frame_defect: defect,
            },
            Some(&mut weighted_coordinates),
        )?;
        let ambient = fast_atb(&units.readers, &weighted_coordinates);
        let in_frame = fast_atb(&frame, &ambient);
        let horizontal_gradient = (&ambient - &fast_ab(&frame, &in_frame)) * 2.0;
        Ok(FrameGradient {
            explained_variance,
            discarded_error: self.total_variance - explained_variance,
            horizontal_gradient,
        })
    }

    /// `v_j⊥ = ‖w_j − Q Qᵀ w_j‖²` for every unit, in tiles, with `coordinates = R = W Q`. It is formed from the residual
    /// itself rather than as `‖w_j‖² − ‖Qᵀ w_j‖²`, so it is a sum of squares and never negative.
    pub fn discarded_reader_variances(
        &self,
        frame: ArrayView2<'_, f64>,
        coordinates: ArrayView2<'_, f64>,
    ) -> Array1<f64> {
        let width = self.width();
        let mut variances = Array1::<f64>::zeros(width);
        let tile = byte_balanced_row_chunk(self.input_dim(), width);
        for start in (0..width).step_by(tile) {
            let end = (start + tile).min(width);
            let residual = &self.units.readers.slice(s![start..end, ..])
                - &fast_abt(&coordinates.slice(s![start..end, ..]), &frame);
            for (offset, row) in residual.axis_iter(Axis(0)).enumerate() {
                variances[start + offset] = row.dot(&row);
            }
        }
        variances
    }

    /// Validate a frame and return its measured defect [`frame_defect`], refusing a defect of at least 1.
    fn require_frame(&self, frame: ArrayView2<'_, f64>) -> Result<f64, ResponseError> {
        require_length("retained frame rows", self.input_dim(), frame.nrows())?;
        if frame.ncols() > self.input_dim() {
            return Err(ResponseError::FrameWiderThanInput {
                rank: frame.ncols(),
                input_dim: self.input_dim(),
            });
        }
        require_finite("retained frame", frame.iter())?;
        let defect = frame_defect(frame);
        if !(defect < 1.0) {
            return Err(ResponseError::FrameNotOrthonormal { defect });
        }
        Ok(defect)
    }
}

impl BlockUnits {
    /// One tiled pass over the unit pairs `j ≤ k` for reader coordinates `coordinates` (`h × c`, with covariance
    /// `r_jk = coordinates_j · coordinates_k`). It returns `Σ_jk D_jk [K_σ − m_j m_k]`, formed as
    /// `Σ_j (D_jj [K_jj − m_j²] + 2 Σ_{k>j} D_jk [K_jk − m_j m_k])` because `D` and the pair law are symmetric. When
    /// `weighted_coordinates` is given it writes `B R` into it (`h × c`, with `B_jk = D_jk ∂_r K_σ`, symmetric by
    /// construction): row `j` is `Σ_{k≥j} B_jk R_k + Σ_{i<j} B_ij R_i`. The tiles are the reader Gram's own
    /// ([`upper_tile_rows`]), so a streamed pass reads the cached rows' words. Units are summed in order and tiles merge
    /// in order, so neither the value nor `B R` depends on the thread count or on the route to `D`.
    fn pair_pass(
        &self,
        coordinates: ArrayView2<'_, f64>,
        formation: CoordinateFormation,
        mut weighted_coordinates: Option<&mut Array2<f64>>,
    ) -> Result<f64, ResponseError> {
        let (width, terms) = coordinates.dim();
        let input_dim = self.readers.ncols();
        let coordinate_norms: Array1<f64> = coordinates
            .rows()
            .into_iter()
            .map(|row| row.dot(&row).sqrt())
            .collect();
        let reader_norm_bounds = (&self.reader_variances + &self.reader_variance_errors).mapv(f64::sqrt);
        // A frame row `fl(w_jᵀ Q)` errs per entry by at most `γ_d ‖w_j‖ ‖q_a‖`, with `‖q_a‖² ≤ 1 + η`, and its exact
        // Gram differs from the nearest orthonormal frame's projector by at most `η ‖w_j‖ ‖w_k‖`.
        let (row_error_scale, projector_defect) = match formation {
            CoordinateFormation::Copied => (0.0, 0.0),
            CoordinateFormation::FrameProducts { frame_defect } => (
                accumulation_growth(input_dim) * (terms as f64 * (1.0 + frame_defect)).sqrt(),
                frame_defect,
            ),
        };
        if let Some(out) = weighted_coordinates.as_deref_mut() {
            out.fill(0.0);
        }
        let tile = upper_tile_rows(width);
        let mut unit_sums = Vec::with_capacity(width);
        let mut streamed_rows: Vec<f64> = Vec::new();
        let mut row_starts: Vec<usize> = Vec::with_capacity(tile + 1);
        for start in (0..width).step_by(tile) {
            let end = (start + tile).min(width);
            let rows = end - start;
            // Row `j ∈ J` against the columns `k ≥ start`, at entry `k − start`.
            let covariances = fast_abt(
                &coordinates.slice(s![start..end, ..]),
                &coordinates.slice(s![start.., ..]),
            );
            // Row `j` of `D` over `k ≥ j` starts at `row_starts[j − start]` of the tile's packed rows.
            row_starts.clear();
            let mut packed_length = 0;
            for unit in start..end {
                row_starts.push(packed_length);
                packed_length += width - unit;
            }
            row_starts.push(packed_length);
            if self.reader_gram.is_none() {
                streamed_rows.resize(packed_length, 0.0);
                fill_upper_rows(
                    self.writers.view(),
                    self.metric_writers.view(),
                    start,
                    end,
                    &mut streamed_rows,
                )
                .map_err(|error| ResponseError::ReaderGram { error })?;
            }
            // `B_J` over the columns `k ≥ start`, zero below each row's diagonal.
            let mut upper_weights = Array2::<f64>::zeros((rows, width - start));
            let sums = upper_weights
                .axis_iter_mut(Axis(0))
                .into_par_iter()
                .zip(covariances.axis_iter(Axis(0)).into_par_iter())
                .enumerate()
                .map(|(offset, (mut weight_row, covariance_row))| {
                    let unit = start + offset;
                    let metric_row = match &self.reader_gram {
                        Some(gram) => gram.upper_row(unit),
                        None => &streamed_rows[row_starts[offset]..row_starts[offset + 1]],
                    };
                    let mut diagonal = 0.0;
                    let mut off_diagonal = 0.0;
                    for (index, &metric_product) in metric_row.iter().enumerate() {
                        let other = unit + index;
                        let covariance_rounding = covariance_rounding_band(
                            &CovarianceFormation {
                                terms,
                                left_norm: coordinate_norms[unit],
                                right_norm: coordinate_norms[other],
                                left_row_error: row_error_scale * reader_norm_bounds[unit],
                                right_row_error: row_error_scale * reader_norm_bounds[other],
                                law_gap: projector_defect * reader_norm_bounds[unit] * reader_norm_bounds[other],
                                variance_error_x: self.reader_variance_errors[unit],
                                variance_error_y: self.reader_variance_errors[other],
                            },
                            self.reader_variances[unit],
                            self.reader_variances[other],
                        );
                        let moments = pair_moments(
                            self.activation,
                            PreactivationPair {
                                mean_x: self.biases[unit],
                                mean_y: self.biases[other],
                                variance_x: self.reader_variances[unit],
                                variance_y: self.reader_variances[other],
                                covariance: covariance_row[other - start],
                                covariance_rounding,
                            },
                        )?;
                        let term = metric_product
                            * (moments.value - self.unit_means[unit] * self.unit_means[other]);
                        if index == 0 {
                            diagonal = term;
                        } else {
                            off_diagonal += term;
                        }
                        weight_row[other - start] = metric_product * moments.covariance_derivative;
                    }
                    Ok(diagonal + 2.0 * off_diagonal)
                })
                .collect::<Result<Vec<f64>, ResponseError>>()?;
            unit_sums.extend(sums);
            if let Some(out) = weighted_coordinates.as_deref_mut() {
                // `Σ_{k≥j} B_jk R_k` for the rows `j ∈ J`.
                let own = fast_ab(&upper_weights, &coordinates.slice(s![start.., ..]));
                out.slice_mut(s![start..end, ..]).scaled_add(1.0, &own);
                // `Σ_{j∈J, j<k} B_jk R_j` for the rows `k ≥ start`: the strictly upper part of `B_J`, transposed.
                for offset in 0..rows {
                    upper_weights[[offset, offset]] = 0.0;
                }
                let mirrored = fast_atb(&upper_weights, &coordinates.slice(s![start..end, ..]));
                out.slice_mut(s![start.., ..]).scaled_add(1.0, &mirrored);
            }
        }
        Ok(unit_sums.iter().sum())
    }
}

/// `T_v σ(t)`, with the kernel's refusal carried as a [`ResponseError`].
pub fn smoothing(activation: GaussianActivation, variance: f64, location: f64) -> Result<f64, ResponseError> {
    let mut value = [0.0];
    gaussian_smoothing_derivatives(activation, location, variance, &mut value).map_err(|error| {
        ResponseError::Kernel {
            context: "Gaussian smoothing",
            error,
        }
    })?;
    Ok(value[0])
}

/// `K_σ` and `∂_r K_σ` for one unit pair, with the kernel's refusal carried as a [`ResponseError`]. The kernel owns
/// the projection of a rounded covariance onto its Cauchy–Schwarz interval, within the pair's stated
/// `covariance_rounding`.
pub fn pair_moments(activation: GaussianActivation, pair: PreactivationPair) -> Result<PairKernel, ResponseError> {
    pair_kernel(activation, pair).map_err(|error| ResponseError::Kernel {
        context: "Gaussian pair kernel",
        error,
    })
}

/// Refuse a length that differs from the one the block requires.
pub fn require_length(context: &'static str, expected: usize, got: usize) -> Result<(), ResponseError> {
    if expected == got {
        Ok(())
    } else {
        Err(ResponseError::DimensionMismatch {
            context,
            expected,
            got,
        })
    }
}

/// Refuse an input with a non-finite entry.
pub fn require_finite<'a>(
    context: &'static str,
    mut values: impl Iterator<Item = &'a f64>,
) -> Result<(), ResponseError> {
    if values.all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(ResponseError::NonFinite { context })
    }
}

#[cfg(test)]
#[path = "subspace_tests.rs"]
mod subspace_tests;

#[cfg(test)]
#[path = "subspace_route_tests.rs"]
mod subspace_route_tests;
