//! Composition across known blocks under the declared law (#2946).
//!
//! # Conditional means do not compose
//!
//! `E ReLU(Z)² = 1/2`, but `(E ReLU Z)² = 1/(2π)`. A second stage `F₂` after a block `F₁` reads the conditional LAW of
//! `Y = F₁(Z)` given the retained input, not its conditional mean.
//!
//! # The first block's law at a retained point
//!
//! With `F₁(z) = U σ(b + W z) + c₁` and `P = Q Qᵀ`, the independence of `PZ` and `(I − P)Z` gives, exactly,
//!
//! ```text
//! a = b + W Z | PZ = Pz  ~  N(μ, Σ⊥),   μ = b + W P z,   Σ⊥ = W (I − P) Wᵀ,
//! C_jl = Cov(σ(a_j), σ(a_l) | PZ = Pz) = K_σ(μ_j, μ_l; v_j⊥, v_l⊥, Σ⊥_jl) − τ_j τ_l,   τ_j = T_{v_j⊥} σ(μ_j),
//! ```
//!
//! with `K_σ` the biased pair kernel at the DISCARDED covariance. `Y | PZ` is the pushforward of that Gaussian by `U σ`:
//! known, and not Gaussian.
//!
//! # Exact compositions
//!
//! - An affine last stage `F₂(y) = A y + c`. `F₂∘F₁ = (A U) σ(b + W z) + A c₁ + c` is one block, so the retained
//!   response, `V(P)`, `E(P)` and the frame gradient of the composition are those of `(W, b, A U, A c₁ + c)`
//!   ([`compose_affine_stage`]).
//! - A quadratic readout `yᵀ A y`. `E[Yᵀ A Y | PZ] = Ȳᵀ A Ȳ + tr(A U C Uᵀ)` with `Ȳ = U τ + c₁`, so the mean composition
//!   `Ȳᵀ A Ȳ` fails by exactly `tr(A U C Uᵀ)` ([`quadratic_readout`]). For one ReLU unit at `P = 0` that is
//!   `1/2 − 1/(2π)`.
//! - Retained readers, `W (I − P) = 0`. `Y` is a function of `PZ`, so `E[F₂(F₁(Z)) | PZ] = F₂(F₁(Pz))` for every `F₂`.
//! - The residual form `F(z) = z + G(z)`. `F̄_P = Pz + Ḡ_P`, and since `E[(I − P)Z σ(b_j + w_jᵀZ)] = (I − P) w_j
//!   E σ'(b_j + w_jᵀZ)`, the energies carry a Stein cross term ([`residual_block_energies`]):
//!
//!   ```text
//!   E(P) = tr(M (I − P)) + 2 Σ_j u_jᵀ M (I − P) w_j (T_{v_j} σ)'(b_j) + E_G(P),   v_j = ‖w_j‖².
//!   ```
//!
//!   The cross term can be negative, and `E(P)` is an expected squared norm, so it never is. The error of `z + G(z)` is
//!   not the skip's error plus `G`'s.
//!
//! A nonlinear `F₂` reading discarded units needs an integral over `rank W (I − P)` dimensions of a non-smooth function,
//! which has no fixed-cost route.
//!
//! # The Gaussian closure (approximate)
//!
//! Replacing the law of `Y | PZ` by the Gaussian with its exact mean `U τ + c₁` and covariance `U C Uᵀ` gives
//!
//! ```text
//! F̂₂(Pz) = c₂ + Σ_k u₂ₖ T_{s_k²} σ₂(m_k),   m_k = b₂ₖ + w₂ₖᵀ (U τ + c₁),   s_k² = α_kᵀ C α_k,   α_k = Uᵀ w₂ₖ.
//! ```
//!
//! `m_k` and `s_k²` are the exact conditional mean and variance of the second block's pre-activation `c_k`; only its
//! shape is closed. The closure is exact for affine and quadratic stages and APPROXIMATE otherwise. Its error comes from
//! the cumulants of `c_k` beyond the second, and it is measured against the executed composition, never assumed
//! ([`GaussianClosureResponse`]).
//!
//! # The ReLU two-moment bound
//!
//! `ReLU(x) = (x + |x|)/2`, so the means cancel. `E|c| ∈ [|m|, √(m² + s²)]` for every law with mean `m` and variance
//! `s²` (Jensen, Cauchy–Schwarz), and two-point laws attain or approach both ends. Hence
//!
//! ```text
//! |E ReLU(c_k) − T_{s_k²} ReLU(m_k)| ≤ e_k = ½ max(√(m_k² + s_k²) − g_k, g_k − |m_k|),   g_k = E|ĉ_k|, ĉ_k ~ N(m_k, s_k²),
//! ```
//!
//! which is sharp over that class, and `‖E F₂ − F̂₂‖_M ≤ Σ_k ‖u₂ₖ‖_M e_k`. The bound holds on this domain only: a ReLU
//! second stage whose `m_k` and `s_k²` are exact to the kernels' accuracy. It is informative off the kink
//! (`e ≈ s²/(4|m|)` for `|m| ≫ s`) and equals the closure itself at `m = 0`. No deterministic bound is claimed for any
//! other second-stage activation.
//!
//! # Cost
//!
//! One retained point costs `h₁²` pair-kernel evaluations plus `O(h₁² q)` flops for `q` projected terms. Units are tiled,
//! so no `h₁ × h₁` matrix is resident.

use super::subspace::{
    CovarianceFormation, KnownBlock, ResponseError, covariance_rounding_band, frame_defect, pair_moments, require_finite,
    require_length, smoothing,
};
use gam_linalg::faer_ndarray::{fast_ab, fast_abt, fast_atb};
use gam_linalg::roundoff::accumulation_growth;
use gam_math::gaussian_activation::{GaussianActivation, PreactivationPair, gaussian_smoothing_derivatives};
use gam_math::probability::{normal_cdf, normal_pdf};
use gam_runtime::resource::byte_balanced_row_chunk;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis, s};
use rayon::prelude::*;

/// Compose an affine last stage `F₂(y) = A y + c` after a known block, with `matrix = A` (`q × p`), `offset = c` (`q`)
/// and the output metric `metric` on `ℝ^q`.
///
/// `F₂∘F₁ = (A U) σ(b + W z) + A c₁ + c` is itself one block, so every quantity of the retained-response operator is
/// exact for the composition, and the output bias drops out of `V(P)`, `E(P)` and the frame gradient.
pub fn compose_affine_stage(
    first: &KnownBlock,
    matrix: ArrayView2<'_, f64>,
    offset: ArrayView1<'_, f64>,
    metric: ArrayView2<'_, f64>,
) -> Result<KnownBlock, ResponseError> {
    require_length("affine stage columns", first.output_dim(), matrix.ncols())?;
    require_length("affine stage offset", matrix.nrows(), offset.len())?;
    require_finite("affine stage matrix", matrix.iter())?;
    require_finite("affine stage offset", offset.iter())?;
    KnownBlock::new(
        first.readers().to_owned(),
        first.biases().to_owned(),
        fast_ab(&matrix, &first.writers()),
        matrix.dot(&first.output_bias()) + &offset,
        metric,
        first.activation(),
    )
}

/// A quadratic readout `yᵀ A y` of a known block at retained points.
#[derive(Debug, Clone)]
pub struct QuadraticReadout {
    /// `E[F₁(Z)ᵀ A F₁(Z) | PZ = Pz]`, exact.
    pub exact: Array1<f64>,
    /// `F̄_{1,P}ᵀ A F̄_{1,P}`, the mean composition.
    pub mean_composition: Array1<f64>,
    /// `tr(A U C Uᵀ)`, the exact amount by which the mean composition fails.
    pub covariance_term: Array1<f64>,
}

/// The exact conditional mean of `F₁(Z)ᵀ A F₁(Z)` given the input inside `frame` (`d × k`), at each row of `points`
/// (`n × d`). `form` is `A` (`p × p`); only its symmetric part is read.
pub fn quadratic_readout(
    first: &KnownBlock,
    form: ArrayView2<'_, f64>,
    frame: ArrayView2<'_, f64>,
    points: ArrayView2<'_, f64>,
) -> Result<QuadraticReadout, ResponseError> {
    require_length("quadratic form rows", first.output_dim(), form.nrows())?;
    require_length("quadratic form columns", first.output_dim(), form.ncols())?;
    require_finite("quadratic form", form.iter())?;
    let law = DiscardedLaw::new(first, frame, points)?;
    let writers = first.writers();
    let mut conditional_means = fast_abt(&law.unit_means, &writers);
    conditional_means += &first.output_bias();
    let formed_means = fast_ab(&conditional_means, &form);
    let mean_composition: Array1<f64> = formed_means
        .rows()
        .into_iter()
        .zip(conditional_means.rows())
        .map(|(formed, mean)| formed.dot(&mean))
        .collect();
    // With `L = Uᵀ` and `R = (A U)ᵀ`, `Σ_k diag(Lᵀ C R)_k = Σ_jl (Uᵀ A U)_jl C_jl = tr(A U C Uᵀ)`.
    let formed_writers = fast_ab(&form, &writers);
    let covariance_term = law
        .covariance_diagonals(first, frame, writers.t(), formed_writers.t())?
        .sum_axis(Axis(1));
    let exact = &mean_composition + &covariance_term;
    Ok(QuadraticReadout {
        exact,
        mean_composition,
        covariance_term,
    })
}

/// `V(P)` and `E(P)` of the residual form `F(z) = z + G(z)`.
#[derive(Debug, Clone)]
pub struct ResidualEnergies {
    /// `V(P) = tr(M P) + 2 Σ_j u_jᵀ M P w_j (T_{v_j} σ)'(b_j) + V_G(P)`.
    pub explained_variance: f64,
    /// `E(P) = tr(M (I − P)) + 2 Σ_j u_jᵀ M (I − P) w_j (T_{v_j} σ)'(b_j) + E_G(P)`, an expected squared norm, so never
    /// negative.
    pub discarded_error: f64,
    /// The Stein cross term `2 Σ_j u_jᵀ M (I − P) w_j (T_{v_j} σ)'(b_j)` inside `discarded_error`, which can be negative.
    pub discarded_cross_term: f64,
}

/// The energies of the residual form `F(z) = z + G(z)` for the input inside `frame` (`d × k`). `block` is `G`, writing
/// into its own input space under the output metric `M` (`d × d`).
///
/// The skip must carry the block's declared input `z` itself. A pre-norm transformer block's skip carries the pre-norm
/// stream, not the post-norm input the declared law is placed on, so this form does not describe such a block's
/// residual output.
///
/// The discarded cross term is formed from the discarded reader parts `w_j − Q Qᵀ w_j`, tiled, rather than as a
/// difference of the full and retained sums.
pub fn residual_block_energies(
    block: &KnownBlock,
    frame: ArrayView2<'_, f64>,
) -> Result<ResidualEnergies, ResponseError> {
    require_length("residual block output dimension", block.input_dim(), block.output_dim())?;
    let explained_mlp = block.explained_variance(frame)?;
    let readers = block.readers();
    let metric = block.metric();
    let metric_writers = block.metric_writers();
    let activation = block.activation();
    // `(T_{v_j} σ)'(b_j) = E σ'(b_j + w_jᵀ Z)` with `v_j = ‖w_j‖²`.
    let slopes = readers
        .rows()
        .into_iter()
        .zip(block.biases().iter())
        .map(|(reader, &bias)| {
            let mut jet = [0.0; 2];
            gaussian_smoothing_derivatives(activation, bias, reader.dot(&reader), &mut jet).map_err(|error| {
                ResponseError::Kernel {
                    context: "Gaussian smoothing slope",
                    error,
                }
            })?;
            Ok(jet[1])
        })
        .collect::<Result<Array1<f64>, ResponseError>>()?;
    let total_skip = metric.diag().sum();
    let retained_skip: f64 = frame
        .columns()
        .into_iter()
        .zip(fast_ab(&metric, &frame).columns())
        .map(|(column, metric_column)| column.dot(&metric_column))
        .sum();
    let width = block.width();
    let coordinates = fast_ab(&readers, &frame);
    // Column `j` of `Qᵀ M U` dotted with row `j` of `R = W Q` is `u_jᵀ M P w_j`.
    let retained_writers = fast_atb(&frame, &metric_writers);
    let retained_cross: f64 = (0..width)
        .map(|unit| slopes[unit] * retained_writers.column(unit).dot(&coordinates.row(unit)))
        .sum();
    let tile = byte_balanced_row_chunk(block.input_dim(), width);
    let mut discarded_cross = 0.0;
    for start in (0..width).step_by(tile) {
        let end = (start + tile).min(width);
        let residual =
            &readers.slice(s![start..end, ..]) - &fast_abt(&coordinates.slice(s![start..end, ..]), &frame);
        for (offset, residual_reader) in residual.axis_iter(Axis(0)).enumerate() {
            let unit = start + offset;
            discarded_cross += slopes[unit] * residual_reader.dot(&metric_writers.column(unit));
        }
    }
    let discarded_cross_term = 2.0 * discarded_cross;
    Ok(ResidualEnergies {
        explained_variance: retained_skip + 2.0 * retained_cross + explained_mlp,
        discarded_error: (total_skip - retained_skip) + discarded_cross_term + (block.total_variance() - explained_mlp),
        discarded_cross_term,
    })
}

/// The Gaussian closure of a second block after a first, at retained points. APPROXIMATE: see the module docs.
#[derive(Debug, Clone)]
pub struct GaussianClosureResponse {
    /// `F̂₂(Pz) = c₂ + Σ_k u₂ₖ T_{s_k²} σ₂(m_k)`, `n × p₂`.
    pub approximate_response: Array2<f64>,
    /// `m_k = E[c_k | PZ = Pz]`, the exact conditional means of the second block's pre-activations, `n × h₂`.
    pub preactivation_means: Array2<f64>,
    /// `s_k² = Var(c_k | PZ = Pz)`, their exact conditional variances, `n × h₂`.
    pub preactivation_variances: Array2<f64>,
    /// `Σ_k ‖u₂ₖ‖_M e_k` at each point, a deterministic bound on `‖E[F₂(F₁(Z)) | PZ] − F̂₂‖_M`, present only when the
    /// second block's activation is ReLU.
    pub relu_moment_class_bound: Option<Array1<f64>>,
}

/// The Gaussian closure `F̂₂` of `second ∘ first` given the input inside `frame` (`d × k`), at each row of `points`
/// (`n × d`). The second block reads the first block's output: `second.input_dim() == first.output_dim()`.
pub fn gaussian_closure_response(
    first: &KnownBlock,
    second: &KnownBlock,
    frame: ArrayView2<'_, f64>,
    points: ArrayView2<'_, f64>,
) -> Result<GaussianClosureResponse, ResponseError> {
    require_length("second block input dimension", first.output_dim(), second.input_dim())?;
    let law = DiscardedLaw::new(first, frame, points)?;
    let first_writers = first.writers();
    let second_readers = second.readers();
    let mut conditional_means = fast_abt(&law.unit_means, &first_writers);
    conditional_means += &first.output_bias();
    let mut preactivation_means = fast_abt(&conditional_means, &second_readers);
    preactivation_means += &second.biases();
    // `α = Uᵀ W₂ᵀ` (`h₁ × h₂`), so `diag(αᵀ C α)` holds every `s_k²`.
    let reads = fast_atb(&first_writers, &second_readers.t());
    let mut preactivation_variances = law.covariance_diagonals(first, frame, reads.view(), reads.view())?;
    // A variance is never negative, so projecting a rounded value onto `[0, ∞)` never moves it farther from the exact one.
    preactivation_variances.mapv_inplace(|variance| variance.max(0.0));
    let activation = second.activation();
    let mut smoothed = preactivation_means.clone();
    smoothed
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .zip(preactivation_variances.axis_iter(Axis(0)).into_par_iter())
        .try_for_each(|(mut row, variances)| {
            for unit in 0..row.len() {
                row[unit] = smoothing(activation, variances[unit], row[unit])?;
            }
            Ok::<(), ResponseError>(())
        })?;
    let mut approximate_response = fast_abt(&smoothed, &second.writers());
    approximate_response += &second.output_bias();
    let relu_moment_class_bound = if matches!(activation, GaussianActivation::Relu) {
        let writer_norms = (&second.writers() * &second.metric_writers())
            .sum_axis(Axis(0))
            .mapv(f64::sqrt);
        Some(
            preactivation_means
                .rows()
                .into_iter()
                .zip(preactivation_variances.rows())
                .map(|(means, variances)| {
                    means
                        .iter()
                        .zip(variances.iter())
                        .zip(writer_norms.iter())
                        .map(|((&mean, &variance), &norm)| norm * relu_two_moment_bound(mean, variance))
                        .sum::<f64>()
                })
                .collect(),
        )
    } else {
        None
    };
    Ok(GaussianClosureResponse {
        approximate_response,
        preactivation_means,
        preactivation_variances,
        relu_moment_class_bound,
    })
}

/// `½ max(√(m² + s²) − g, g − |m|)` with `g = E|ĉ|`, `ĉ ~ N(m, s²)`: the sharp bound on `|E ReLU(c) − T_{s²} ReLU(m)|`
/// over every law of `c` with mean `m` and variance `s² = variance`.
///
/// Both branches are formed without cancellation against `|m|`: `g − |m| = 2s (φ(x) − x Φ(−x))` with `x = |m|/s`, and
/// `√(m² + s²) − |m| = s² / (√(m² + s²) + |m|)`.
pub fn relu_two_moment_bound(mean: f64, variance: f64) -> f64 {
    if variance == 0.0 {
        // `c = m` almost surely, and the closure is the executed value.
        return 0.0;
    }
    let deviation = variance.sqrt();
    let standardized = mean.abs() / deviation;
    let gaussian_excess = if standardized.is_finite() {
        2.0 * deviation * (normal_pdf(standardized) - standardized * normal_cdf(-standardized))
    } else {
        0.0
    };
    let second_moment_excess = variance / (mean.hypot(deviation) + mean.abs());
    0.5 * (second_moment_excess - gaussian_excess).max(gaussian_excess)
}

/// The first block's hidden pre-activation law at each retained point.
struct DiscardedLaw {
    /// `μ_ij = b_j + w_jᵀ P zᵢ`, `n × h`.
    means: Array2<f64>,
    /// `τ_ij = T_{v_j⊥} σ(μ_ij) = E[σ(a_j) | PZ = P zᵢ]`, `n × h`.
    unit_means: Array2<f64>,
    /// `v_j⊥ = ‖w_j − Q Qᵀ w_j‖²`, a sum of squares.
    discarded_variances: Array1<f64>,
    /// Bounds on `|v̂_j⊥ − w_jᵀ (I − P₀) w_j|`, with `P₀` the projector of the nearest orthonormal frame.
    discarded_variance_errors: Array1<f64>,
    /// Upper bounds on `‖w_j‖`.
    reader_norms: Array1<f64>,
    /// Bounds `δ_j` on `‖ê_j − (I − Q Qᵀ) w_j‖` for the residual row `ê_j = w_j − fl(fl(w_jᵀ Q) Qᵀ)`.
    residual_row_errors: Array1<f64>,
    /// `R = W Q`, `h × k`.
    coordinates: Array2<f64>,
    /// The measured bound `η ≥ ‖QᵀQ − I‖₂`.
    frame_defect: f64,
}

impl DiscardedLaw {
    fn new(
        block: &KnownBlock,
        frame: ArrayView2<'_, f64>,
        points: ArrayView2<'_, f64>,
    ) -> Result<Self, ResponseError> {
        let input_dim = block.input_dim();
        require_length("retained frame rows", input_dim, frame.nrows())?;
        if frame.ncols() > input_dim {
            return Err(ResponseError::FrameWiderThanInput {
                rank: frame.ncols(),
                input_dim,
            });
        }
        require_finite("retained frame", frame.iter())?;
        let defect = frame_defect(frame);
        if !(defect < 1.0) {
            return Err(ResponseError::FrameNotOrthonormal { defect });
        }
        require_length("retained point columns", input_dim, points.ncols())?;
        require_finite("retained points", points.iter())?;
        let readers = block.readers();
        let coordinates = fast_ab(&readers, &frame);
        let discarded_variances = block.discarded_reader_variances(frame, coordinates.view());
        // First-order formation bounds, with `γ_n` the accumulation growth, `‖Q‖₂ ≤ √(1 + η)` and
        // `‖Q‖_F ≤ √(k (1 + η))`:
        // - `R̂_j = fl(w_jᵀ Q)` errs by at most `γ_d ‖Q‖_F ‖w_j‖`, and `Qᵀ` carries that error by `‖Q‖₂`;
        // - `fl(R̂_j Qᵀ)` rounds by at most `γ_k ‖Q‖_F ‖R̂_j‖`;
        // - the subtraction rounds by at most `γ₁ (‖w_j‖ + ‖p̂_j‖)`.
        // The exact row `(I − Q Qᵀ) w_j` differs from the nearest orthonormal frame's law by `‖Q Qᵀ − P₀‖₂ ≤ η`, so its
        // squared norm differs by at most `η² ‖w_j‖²`.
        let rank = frame.ncols();
        let product_growth = accumulation_growth(input_dim);
        let recombination_growth = accumulation_growth(rank);
        let subtraction_growth = accumulation_growth(1);
        let column_scale = (1.0 + defect).sqrt();
        let frame_norm = (rank as f64 * (1.0 + defect)).sqrt();
        let reader_norms: Array1<f64> = readers
            .rows()
            .into_iter()
            .map(|row| (row.dot(&row) / (1.0 - product_growth)).sqrt())
            .collect();
        let residual_row_errors = reader_norms.mapv(|reader_norm| {
            let coordinate_error = product_growth * frame_norm * reader_norm;
            let coordinate_norm = column_scale * reader_norm + coordinate_error;
            let recombination_error = recombination_growth * frame_norm * coordinate_norm;
            let product_norm = column_scale * coordinate_norm + recombination_error;
            column_scale * coordinate_error + recombination_error + subtraction_growth * (reader_norm + product_norm)
        });
        // `|v̂⊥ − ‖ê‖²| ≤ γ_d ‖ê‖²`, `|‖ê‖² − ‖e‖²| ≤ δ (2 ‖ê‖ + δ)` and `|‖e‖² − v⊥| ≤ η² ‖w‖²`.
        let discarded_variance_errors = Array1::from_shape_fn(block.width(), |unit| {
            let residual_norm = (discarded_variances[unit] / (1.0 - product_growth)).sqrt();
            let row_error = residual_row_errors[unit];
            product_growth * residual_norm * residual_norm
                + row_error * (2.0 * residual_norm + row_error)
                + defect * defect * reader_norms[unit] * reader_norms[unit]
        });
        // Row `i` of `points Q` is `zᵢᵀ Q`, so entry `(i, j)` of the product with `Rᵀ` is `w_jᵀ P zᵢ`.
        let mut means = fast_abt(&fast_ab(&points, &frame), &coordinates);
        means += &block.biases();
        let activation = block.activation();
        let mut unit_means = means.clone();
        unit_means
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .try_for_each(|mut row| {
                for unit in 0..row.len() {
                    row[unit] = smoothing(activation, discarded_variances[unit], row[unit])?;
                }
                Ok::<(), ResponseError>(())
            })?;
        Ok(Self {
            means,
            unit_means,
            discarded_variances,
            discarded_variance_errors,
            reader_norms,
            residual_row_errors,
            coordinates,
            frame_defect: defect,
        })
    }

    /// `diag(Lᵀ C R)` at each retained point (`n × q`), for `left = L` and `right = R` (each `h × q`), where `C` is the
    /// unit covariance of the module docs.
    ///
    /// One tile of `t` units forms its rows of `Σ⊥` once for every point, as `(W_J − R_J Qᵀ) Wᵀ` (`t × h`); each point
    /// then holds one `h` row at a time. Points are summed tile by tile in order, so the value does not depend on the
    /// thread count.
    fn covariance_diagonals(
        &self,
        block: &KnownBlock,
        frame: ArrayView2<'_, f64>,
        left: ArrayView2<'_, f64>,
        right: ArrayView2<'_, f64>,
    ) -> Result<Array2<f64>, ResponseError> {
        let readers = block.readers();
        let activation = block.activation();
        let input_dim = block.input_dim();
        let product_growth = accumulation_growth(input_dim);
        let (points, width) = self.means.dim();
        let terms = left.ncols();
        let mut diagonals = Array2::<f64>::zeros((points, terms));
        let tile = byte_balanced_row_chunk(block.input_dim() + width, width);
        for start in (0..width).step_by(tile) {
            let end = (start + tile).min(width);
            let residual =
                &readers.slice(s![start..end, ..]) - &fast_abt(&self.coordinates.slice(s![start..end, ..]), &frame);
            let discarded_covariances = fast_abt(&residual, &readers);
            let residual_norms: Array1<f64> = residual
                .rows()
                .into_iter()
                .map(|row| (row.dot(&row) / (1.0 - product_growth)).sqrt())
                .collect();
            let contributions = (0..points)
                .into_par_iter()
                .map(|point| {
                    let mut contribution = Array1::<f64>::zeros(terms);
                    let mut covariance_row = Array1::<f64>::zeros(width);
                    for offset in 0..(end - start) {
                        let unit = start + offset;
                        for other in 0..width {
                            // `r̂⊥ = fl(ê_j · w_l)`: a `d`-term product of the residual row with the copied reader row.
                            let covariance_rounding = covariance_rounding_band(
                                &CovarianceFormation {
                                    terms: input_dim,
                                    left_norm: residual_norms[offset],
                                    right_norm: self.reader_norms[other],
                                    left_row_error: self.residual_row_errors[unit],
                                    right_row_error: 0.0,
                                    law_gap: self.frame_defect * self.reader_norms[unit] * self.reader_norms[other],
                                    variance_error_x: self.discarded_variance_errors[unit],
                                    variance_error_y: self.discarded_variance_errors[other],
                                },
                                self.discarded_variances[unit],
                                self.discarded_variances[other],
                            );
                            let second_moment = pair_moments(
                                activation,
                                PreactivationPair {
                                    mean_x: self.means[[point, unit]],
                                    mean_y: self.means[[point, other]],
                                    variance_x: self.discarded_variances[unit],
                                    variance_y: self.discarded_variances[other],
                                    covariance: discarded_covariances[[offset, other]],
                                    covariance_rounding,
                                },
                            )?
                            .value;
                            covariance_row[other] =
                                second_moment - self.unit_means[[point, unit]] * self.unit_means[[point, other]];
                        }
                        contribution += &(&left.row(unit) * &right.t().dot(&covariance_row));
                    }
                    Ok(contribution)
                })
                .collect::<Result<Vec<Array1<f64>>, ResponseError>>()?;
            for (point, contribution) in contributions.iter().enumerate() {
                let mut target = diagonals.row_mut(point);
                target += contribution;
            }
        }
        Ok(diagonals)
    }
}

#[cfg(test)]
#[path = "compose_tests.rs"]
mod compose_tests;
