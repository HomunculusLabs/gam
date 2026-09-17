//! A whole pre-norm transformer block executed under component masks (#2951).
//!
//! # The block
//!
//! A decoder block moves the residual stream rows `h` through two sublayers,
//! each reading its own normalization of the current stream:
//!
//! ```text
//! Sequential (Qwen3, Llama):  h₁ = h + A(N₁(h)),   h' = h₁ + F(N₂(h₁))
//! Parallel (GPT-NeoX):        h' = (F(N₂(h)) + A(N₁(h))) + h
//! ```
//!
//! `A` is the source's causal self-attention through its output projection
//! ([`super::attention`]), `F(x) = W₂ σ(W₁ x + b₁) + b₂` is the MLP write
//! ([`super::rewrite`]), and `N₁`, `N₂` are the source's RMSNorm or LayerNorm
//! ([`MaskedNorm`]). The residual additions, in the source's association, are
//! [`decoder_layer`]'s. This module owns no primitive and no residual order,
//! only the composition.
//!
//! # Exact replay under masks
//!
//! [`ComponentBlock`] executes the query and key projections through masked
//! component coordinates (P6) and both MLP weights through masked exact factors
//! (P5). Each sublayer is its owner's identity: for every real mask
//! (continuous, binary or signed) it computes, on the input it is handed, the
//! sublayer with the edited tensors `U diag(m) R`. The block hands each sublayer
//! the current stream, after every upstream masked write, never a cached clean
//! one, and each normalization reads its whole current row. So the block under
//! any mask is the source block with the edited tensors: an identity, not a
//! linearization. In the sequential layout a query or key mask reaches the MLP
//! only through `N₂` and `σ` of the summed stream; in the parallel layout it does
//! not reach the MLP input at all. Every block-level mask reaches the logits
//! through at least the final normalization (mpd-gated, #2951 comment
//! 5716738723), so the exact affine-logit adversary (P9) does not apply to it.
//!
//! With every mask at one both sublayers run the original tensors on their
//! original paths, so the block is bit-identical to [`NativeBlock::execute`].
//!
//! # Stages
//!
//! [`BlockExecution`] keeps each stage that a torch block exports as a module
//! output: the two normalized inputs, the attention execution with its
//! forward-error radius, the MLP pre-activation, activation and write, and the
//! layer output. A receipt compares an external executor with the native block
//! stage by stage (A12). The tests propagate the stage radii into a derived
//! replay band over the whole block.
//!
//! # Validity domain
//!
//! The block covers what its sublayers cover: query and key projections without
//! biases that feed the rotary embedding directly (no per-head query/key norm),
//! an MLP with a ReLU or exact GELU activation, and normalizations with the
//! source's own gain and bias. Masks on the value and output projections, on
//! normalization gains, and on a gated (SwiGLU) MLP are not carried here.

use super::attention::{
    AttentionExecution, AttentionProgramError, ComponentAttention, ComponentProjection,
    NativeAttention, QueryKeyMasks,
};
use super::gated_rewrite::{GatedRewriteError, MaskedNorm, ResidualLayout, decoder_layer};
use super::rewrite::{ComponentMlp, ComponentRead, MlpMask, NativeMlp, RewriteError};
use ndarray::{Array1, Array2, ArrayView2};
use std::fmt;

/// A source model's input normalization on its own tensors.
#[derive(Clone, Debug, PartialEq)]
pub enum NativeNorm {
    /// `Qwen3RMSNorm`: `w ⊙ h (mean(h²) + ε)^(-1/2)`.
    Rms { epsilon: f64, gain: Array1<f64> },
    /// `torch.nn.LayerNorm`: the RMSNorm of the centred row, plus `β`.
    Layer {
        epsilon: f64,
        gain: Array1<f64>,
        bias: Array1<f64>,
    },
}

impl NativeNorm {
    /// The normalization as its owner's [`MaskedNorm`], with the source's own
    /// gain and bias as the edited tensors.
    pub fn as_masked_norm(&self) -> MaskedNorm<'_> {
        match self {
            Self::Rms { epsilon, gain } => MaskedNorm::Rms {
                epsilon: *epsilon,
                gain: gain.view(),
            },
            Self::Layer {
                epsilon,
                gain,
                bias,
            } => MaskedNorm::Layer {
                epsilon: *epsilon,
                gain: gain.view(),
                bias: bias.view(),
            },
        }
    }
}

/// One of a decoder block's two sublayers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sublayer {
    Attention,
    Mlp,
}

impl fmt::Display for Sublayer {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Attention => "attention",
            Self::Mlp => "MLP",
        })
    }
}

/// A refused block execution.
#[derive(Debug)]
pub enum BlockError {
    /// A sublayer's input normalization refused the current stream.
    Norm {
        sublayer: Sublayer,
        error: GatedRewriteError,
    },
    /// The decoder layer refused a sublayer's write.
    Layer(GatedRewriteError),
    Attention(AttentionProgramError),
    Mlp(RewriteError),
    /// The decoder layer returned a stream without running this sublayer: an
    /// executor invariant.
    SublayerNotExecuted { sublayer: Sublayer },
}

impl From<GatedRewriteError> for BlockError {
    fn from(error: GatedRewriteError) -> Self {
        Self::Layer(error)
    }
}

impl From<AttentionProgramError> for BlockError {
    fn from(error: AttentionProgramError) -> Self {
        Self::Attention(error)
    }
}

impl From<RewriteError> for BlockError {
    fn from(error: RewriteError) -> Self {
        Self::Mlp(error)
    }
}

impl fmt::Display for BlockError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Norm { sublayer, error } => {
                write!(formatter, "the {sublayer} sublayer's normalization refused: {error}")
            }
            Self::Layer(error) => write!(formatter, "the decoder layer refused a write: {error}"),
            Self::Attention(error) => write!(formatter, "the attention sublayer refused: {error}"),
            Self::Mlp(error) => write!(formatter, "the MLP sublayer refused: {error}"),
            Self::SublayerNotExecuted { sublayer } => write!(
                formatter,
                "executor invariant: the decoder layer returned without running the {sublayer} sublayer"
            ),
        }
    }
}

impl std::error::Error for BlockError {}

/// Every stage of one executed block. Rows are positions throughout.
#[derive(Clone, Debug)]
pub struct BlockExecution {
    /// `N₁(h)`, the attention sublayer's input.
    pub attention_input: Array2<f64>,
    /// The attention sublayer on [`Self::attention_input`]. Its output, after the
    /// output projection, is the sublayer's write, with its forward-error radius.
    pub attention: AttentionExecution,
    /// `N₂` of the stream the MLP reads: `h + A` in the sequential layout, `h` in
    /// the parallel one.
    pub mlp_input: Array2<f64>,
    /// `W₁ x + b₁` on [`Self::mlp_input`], under the read-in mask.
    pub mlp_summed_input: Array2<f64>,
    /// `σ` of [`Self::mlp_summed_input`].
    pub mlp_activations: Array2<f64>,
    /// `W₂ a + b₂` on [`Self::mlp_activations`], under the write-out mask.
    pub mlp_write: Array2<f64>,
    /// The residual stream after both writes.
    pub output: Array2<f64>,
}

/// The MLP sublayer's stages on its normalized input.
struct MlpStage {
    summed_input: Array2<f64>,
    activations: Array2<f64>,
    write: Array2<f64>,
}

/// The decoder layer over `residual`, with each sublayer normalizing the stream
/// the layer hands it, and every stage kept.
fn execute_layer<Attend, Mlp>(
    layout: ResidualLayout,
    attention_norm: &NativeNorm,
    mlp_norm: &NativeNorm,
    residual: ArrayView2<'_, f64>,
    attend: Attend,
    mlp: Mlp,
) -> Result<BlockExecution, BlockError>
where
    Attend: FnOnce(ArrayView2<'_, f64>) -> Result<AttentionExecution, BlockError>,
    Mlp: FnOnce(ArrayView2<'_, f64>) -> Result<MlpStage, BlockError>,
{
    let mut attention_stage = None;
    let mut mlp_stage = None;
    let output = decoder_layer::<BlockError, _, _>(
        layout,
        residual,
        |stream| {
            let input = attention_norm
                .as_masked_norm()
                .apply(stream)
                .map_err(|error| BlockError::Norm {
                    sublayer: Sublayer::Attention,
                    error,
                })?;
            let execution = attend(input.view())?;
            let write = execution.output.clone();
            attention_stage = Some((input, execution));
            Ok(write)
        },
        |stream| {
            let input = mlp_norm
                .as_masked_norm()
                .apply(stream)
                .map_err(|error| BlockError::Norm {
                    sublayer: Sublayer::Mlp,
                    error,
                })?;
            let stage = mlp(input.view())?;
            let write = stage.write.clone();
            mlp_stage = Some((input, stage));
            Ok(write)
        },
    )?;
    let (attention_input, attention) = attention_stage.ok_or(BlockError::SublayerNotExecuted {
        sublayer: Sublayer::Attention,
    })?;
    let (mlp_input, stage) = mlp_stage.ok_or(BlockError::SublayerNotExecuted {
        sublayer: Sublayer::Mlp,
    })?;
    Ok(BlockExecution {
        attention_input,
        attention,
        mlp_input,
        mlp_summed_input: stage.summed_input,
        mlp_activations: stage.activations,
        mlp_write: stage.write,
        output,
    })
}

/// A decoder block on its original tensors.
#[derive(Clone, Debug)]
pub struct NativeBlock {
    layout: ResidualLayout,
    attention_norm: NativeNorm,
    attention: NativeAttention,
    mlp_norm: NativeNorm,
    mlp: NativeMlp,
}

impl NativeBlock {
    /// A block of the source's sublayers. Each width is refused where its owner
    /// reads it: the normalizations at their gain, the attention at its input, the
    /// MLP at its read-in, and the layer at each write.
    pub fn new(
        layout: ResidualLayout,
        attention_norm: NativeNorm,
        attention: NativeAttention,
        mlp_norm: NativeNorm,
        mlp: NativeMlp,
    ) -> Self {
        Self {
            layout,
            attention_norm,
            attention,
            mlp_norm,
            mlp,
        }
    }

    pub fn layout(&self) -> ResidualLayout {
        self.layout
    }

    /// The source block on its original tensors, over the residual rows at their
    /// absolute positions.
    pub fn execute(
        &self,
        residual: ArrayView2<'_, f64>,
        positions: &[i64],
    ) -> Result<BlockExecution, BlockError> {
        execute_layer(
            self.layout,
            &self.attention_norm,
            &self.mlp_norm,
            residual,
            |input| Ok(self.attention.execute(input, positions)?),
            |input| {
                let summed_input = self.mlp.summed_input(input).map_err(RewriteError::from)?;
                let activations = self
                    .mlp
                    .activate(summed_input.view())
                    .map_err(RewriteError::from)?;
                let write = self
                    .mlp
                    .written(activations.view())
                    .map_err(RewriteError::from)?;
                Ok(MlpStage {
                    summed_input,
                    activations,
                    write,
                })
            },
        )
    }
}

/// The masks of a [`ComponentBlock`]: the query and key component masks of its
/// attention and the factor masks of its MLP.
#[derive(Clone, Copy, Debug)]
pub struct BlockMasks<'a> {
    pub attention: &'a QueryKeyMasks,
    pub mlp: MlpMask<'a>,
}

/// A decoder block with its query and key projections and both MLP weights
/// executed through masked component coordinates (see the module documentation).
#[derive(Clone, Debug)]
pub struct ComponentBlock {
    layout: ResidualLayout,
    attention_norm: NativeNorm,
    attention: ComponentAttention,
    mlp_norm: NativeNorm,
    mlp: ComponentMlp,
}

impl ComponentBlock {
    /// Factors `native`'s sublayers: `query` and `key` must factor the attention's
    /// query and key projections exactly (as [`ComponentAttention::new`] requires),
    /// and both MLP weights are solved for their exact writes through `read_in` and
    /// `write_out` ([`ComponentMlp::new`]).
    pub fn new(
        native: NativeBlock,
        query: ComponentProjection,
        key: ComponentProjection,
        read_in: ComponentRead<'_>,
        write_out: ComponentRead<'_>,
    ) -> Result<Self, BlockError> {
        let NativeBlock {
            layout,
            attention_norm,
            attention,
            mlp_norm,
            mlp,
        } = native;
        Ok(Self {
            layout,
            attention_norm,
            attention: ComponentAttention::new(attention, query, key)?,
            mlp_norm,
            mlp: ComponentMlp::new(mlp, read_in, write_out)?,
        })
    }

    pub fn layout(&self) -> ResidualLayout {
        self.layout
    }

    /// The MLP sublayer and its exact factors.
    pub fn mlp(&self) -> &ComponentMlp {
        &self.mlp
    }

    /// The block under `masks`, over the residual rows at their absolute
    /// positions. All-on masks execute the original tensors on their original
    /// paths.
    pub fn execute(
        &self,
        masks: BlockMasks<'_>,
        residual: ArrayView2<'_, f64>,
        positions: &[i64],
    ) -> Result<BlockExecution, BlockError> {
        execute_layer(
            self.layout,
            &self.attention_norm,
            &self.mlp_norm,
            residual,
            |input| Ok(self.attention.execute(masks.attention, input, positions)?),
            |input| {
                let summed_input = self
                    .mlp
                    .summed_input(input, masks.mlp.read_in)
                    .map_err(RewriteError::from)?;
                let activations = self
                    .mlp
                    .native()
                    .activate(summed_input.view())
                    .map_err(RewriteError::from)?;
                let write = self
                    .mlp
                    .written(activations.view(), masks.mlp.write_out)
                    .map_err(RewriteError::from)?;
                Ok(MlpStage {
                    summed_input,
                    activations,
                    write,
                })
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parameter_decomposition::attention::{
        AffineProjection, AttentionGeometry, RotaryEmbedding, RotaryPairing,
    };
    use crate::parameter_decomposition::rewrite::{ComponentMask, ExactFactor};
    use gam_linalg::roundoff::{UNIT_ROUNDOFF, accumulation_growth};
    use gam_math::gaussian_activation::GaussianActivation;
    use ndarray::ArrayView1;
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};

    const WIDTH: usize = 4;
    const HIDDEN: usize = 6;
    const READ_IN_COMPONENTS: usize = 7;
    const WRITE_OUT_COMPONENTS: usize = 8;
    const QUERY_COMPONENTS: usize = 5;
    const KEY_COMPONENTS: usize = 4;
    const TOKENS: usize = 5;
    /// Qwen3's declared `rms_norm_eps`.
    const RMS_EPSILON: f64 = 1.0e-6;
    /// Pythia's declared `layer_norm_eps`.
    const LAYER_EPSILON: f64 = 1.0e-5;

    /// Two query heads sharing one key/value head, two rotated planes and two
    /// pass-through coordinates per head.
    fn geometry() -> AttentionGeometry {
        AttentionGeometry {
            model_dim: WIDTH,
            n_heads: 2,
            n_kv_heads: 1,
            head_dim: 6,
        }
    }

    fn rotary() -> RotaryEmbedding {
        RotaryEmbedding {
            pairing: RotaryPairing::HalfSplit,
            inverse_frequencies: vec![1.0, 0.25],
            attention_scaling: 1.25,
        }
    }

    /// Eighths in `[-1, 1]`. A product of three and a sum of a few dozen stay
    /// exact in f64, so every product of fixture tensors and masks, including each
    /// edited tensor `A diag(m) B`, is formed without rounding.
    fn eighths(rng: &mut StdRng, rows: usize, cols: usize) -> Array2<f64> {
        Array2::from_shape_simple_fn((rows, cols), || rng.random_range(-8..=8) as f64 / 8.0)
    }

    fn eighths_vector(rng: &mut StdRng, len: usize) -> Array1<f64> {
        Array1::from_shape_simple_fn(len, || rng.random_range(-8..=8) as f64 / 8.0)
    }

    /// A bias-free source projection: adding the zero bias is exact, so the
    /// block's arithmetic is that of the unbiased source.
    fn unbiased(weight: Array2<f64>) -> AffineProjection {
        let bias = Array1::zeros(weight.nrows());
        AffineProjection { weight, bias }
    }

    /// `A diag(m) B`.
    fn masked_product(write: ArrayView2<'_, f64>, mask: ArrayView1<'_, f64>, read: ArrayView2<'_, f64>) -> Array2<f64> {
        (&write * &mask).dot(&read)
    }

    struct Fixture {
        layout: ResidualLayout,
        activation: GaussianActivation,
        attention_gain: Array1<f64>,
        attention_bias: Array1<f64>,
        mlp_gain: Array1<f64>,
        mlp_bias: Array1<f64>,
        query_outputs: Array2<f64>,
        query_readins: Array2<f64>,
        key_outputs: Array2<f64>,
        key_readins: Array2<f64>,
        value: Array2<f64>,
        output: Array2<f64>,
        read_in_candidate: Array2<f64>,
        read_in: Array2<f64>,
        bias_in: Array1<f64>,
        write_out_candidate: Array2<f64>,
        write_out: Array2<f64>,
        bias_out: Array1<f64>,
        residual: Array2<f64>,
        positions: Vec<i64>,
    }

    impl Fixture {
        /// A block whose weights factor exactly through overcomplete reads,
        /// `W₁ = N R` and `W₂ = N̄ R̄`, so each exact write is its candidate. The
        /// sequential layout normalizes with RMSNorm (Qwen3), the parallel one with
        /// LayerNorm (GPT-NeoX).
        fn new(layout: ResidualLayout, activation: GaussianActivation, seed: u64) -> Self {
            let mut rng = StdRng::seed_from_u64(seed);
            let g = geometry();
            Self {
                layout,
                activation,
                attention_gain: eighths_vector(&mut rng, WIDTH),
                attention_bias: eighths_vector(&mut rng, WIDTH),
                mlp_gain: eighths_vector(&mut rng, WIDTH),
                mlp_bias: eighths_vector(&mut rng, WIDTH),
                query_outputs: eighths(&mut rng, g.query_dim(), QUERY_COMPONENTS),
                query_readins: eighths(&mut rng, QUERY_COMPONENTS, WIDTH),
                key_outputs: eighths(&mut rng, g.key_value_dim(), KEY_COMPONENTS),
                key_readins: eighths(&mut rng, KEY_COMPONENTS, WIDTH),
                value: eighths(&mut rng, g.key_value_dim(), WIDTH),
                output: eighths(&mut rng, WIDTH, g.query_dim()),
                read_in_candidate: eighths(&mut rng, HIDDEN, READ_IN_COMPONENTS),
                read_in: eighths(&mut rng, READ_IN_COMPONENTS, WIDTH),
                bias_in: eighths_vector(&mut rng, HIDDEN),
                write_out_candidate: eighths(&mut rng, WIDTH, WRITE_OUT_COMPONENTS),
                write_out: eighths(&mut rng, WRITE_OUT_COMPONENTS, HIDDEN),
                bias_out: eighths_vector(&mut rng, WIDTH),
                residual: Array2::from_shape_simple_fn((TOKENS, WIDTH), || {
                    rng.random_range(-16..=16) as f64 / 8.0
                }),
                positions: vec![2, 3, 5, 6, 9],
            }
        }

        fn norm(&self, gain: &Array1<f64>, bias: &Array1<f64>) -> NativeNorm {
            match self.layout {
                ResidualLayout::Sequential => NativeNorm::Rms {
                    epsilon: RMS_EPSILON,
                    gain: gain.clone(),
                },
                ResidualLayout::Parallel => NativeNorm::Layer {
                    epsilon: LAYER_EPSILON,
                    gain: gain.clone(),
                    bias: bias.clone(),
                },
            }
        }

        fn native_block(
            &self,
            query: Array2<f64>,
            key: Array2<f64>,
            read_in_weight: Array2<f64>,
            write_out_weight: Array2<f64>,
        ) -> NativeBlock {
            NativeBlock::new(
                self.layout,
                self.norm(&self.attention_gain, &self.attention_bias),
                NativeAttention::new(
                    geometry(),
                    rotary(),
                    1.0 / (geometry().head_dim as f64).sqrt(),
                    unbiased(query),
                    unbiased(key),
                    unbiased(self.value.clone()),
                    unbiased(self.output.clone()),
                )
                .expect("fixture attention tensors match the geometry"),
                self.norm(&self.mlp_gain, &self.mlp_bias),
                NativeMlp::new(
                    read_in_weight,
                    self.bias_in.clone(),
                    write_out_weight,
                    self.bias_out.clone(),
                    self.activation,
                )
                .expect("fixture MLP shapes compose"),
            )
        }

        /// The source block: every weight is its factors' product.
        fn native(&self) -> NativeBlock {
            self.native_block(
                self.query_outputs.dot(&self.query_readins),
                self.key_outputs.dot(&self.key_readins),
                self.read_in_candidate.dot(&self.read_in),
                self.write_out_candidate.dot(&self.write_out),
            )
        }

        fn component_block(&self) -> ComponentBlock {
            ComponentBlock::new(
                self.native(),
                ComponentProjection::new(self.query_outputs.clone(), self.query_readins.clone())
                    .expect("query factor shapes agree"),
                ComponentProjection::new(self.key_outputs.clone(), self.key_readins.clone())
                    .expect("key factor shapes agree"),
                ComponentRead {
                    read: self.read_in.view(),
                    candidate_write: self.read_in_candidate.view(),
                },
                ComponentRead {
                    read: self.write_out.view(),
                    candidate_write: self.write_out_candidate.view(),
                },
            )
            .expect("random overcomplete reads are resolved")
        }

        /// Direct execution with the edited tensors `U diag(m) R`.
        fn edited_block(&self, block: &ComponentBlock, masks: &FixtureMasks) -> NativeBlock {
            let (read_in, write_out) = (block.mlp().read_in(), block.mlp().write_out());
            self.native_block(
                masked_product(self.query_outputs.view(), masks.query.view(), self.query_readins.view()),
                masked_product(self.key_outputs.view(), masks.key.view(), self.key_readins.view()),
                masked_product(read_in.write(), masks.read_in.view(), read_in.read()),
                masked_product(write_out.write(), masks.write_out.view(), write_out.read()),
            )
        }
    }

    #[derive(Clone)]
    struct FixtureMasks {
        query: Array1<f64>,
        key: Array1<f64>,
        read_in: Array1<f64>,
        write_out: Array1<f64>,
    }

    impl FixtureMasks {
        /// Eighths: continuous in `[0, 1]`, binary, and signed in `[-3/2, 3/2]`.
        fn family(rng: &mut StdRng) -> [(&'static str, Self); 3] {
            let mut draw = |low: i32, high: i32, denominator: f64| {
                let mut vector = |len: usize| {
                    Array1::from_shape_simple_fn(len, || rng.random_range(low..=high) as f64 / denominator)
                };
                Self {
                    query: vector(QUERY_COMPONENTS),
                    key: vector(KEY_COMPONENTS),
                    read_in: vector(READ_IN_COMPONENTS),
                    write_out: vector(WRITE_OUT_COMPONENTS),
                }
            };
            [
                ("continuous", draw(0, 8, 8.0)),
                ("binary", draw(0, 1, 1.0)),
                ("signed", draw(-12, 12, 8.0)),
            ]
        }

        fn ones() -> Self {
            Self {
                query: Array1::ones(QUERY_COMPONENTS),
                key: Array1::ones(KEY_COMPONENTS),
                read_in: Array1::ones(READ_IN_COMPONENTS),
                write_out: Array1::ones(WRITE_OUT_COMPONENTS),
            }
        }

        fn attention(&self) -> QueryKeyMasks {
            QueryKeyMasks {
                query: self.query.clone(),
                key: self.key.clone(),
            }
        }

        fn block<'a>(&'a self, attention: &'a QueryKeyMasks) -> BlockMasks<'a> {
            BlockMasks {
                attention,
                mlp: MlpMask {
                    read_in: ComponentMask::Components(self.read_in.view()),
                    write_out: ComponentMask::Components(self.write_out.view()),
                },
            }
        }
    }

    /// `(|x| |B|ᵀ ⊙ |m|) |A|ᵀ` over nonnegative rows `x`: it bounds
    /// `|A diag(m) B| x` entrywise, and on `|x|` it is the absolute sum of the terms
    /// of `A diag(m) B x`.
    fn magnitude(factor: &ExactFactor, mask: ArrayView1<'_, f64>, rows: ArrayView2<'_, f64>) -> Array2<f64> {
        (rows.dot(&factor.read().mapv(f64::abs).t()) * &mask.mapv(f64::abs))
            .dot(&factor.write().mapv(f64::abs).t())
    }

    /// `|fl(a + b) − (a + b)| ≤ u |a + b| ≤ u |fl(a + b)| / (1 − u)`, entrywise.
    fn addition_rounding(sum: &Array2<f64>) -> Array2<f64> {
        sum.mapv(|value| UNIT_ROUNDOFF * value.abs() / (1.0 - UNIT_ROUNDOFF))
    }

    /// Rounding of `A diag(m) B x + b` on either route, for rows `x` of `rows`.
    ///
    /// The factored route passes each term `a_ic m_c b_cj x_j` through the product
    /// `b x`, `d − 1` additions, the mask, the product with `a`, `C − 1` additions and
    /// the bias: `C + d + 2` rounded operations (Higham ASNA Lemma 3.1, in any
    /// summation order). The edited tensor's route passes `d + 1` over terms bounded
    /// entrywise by the same magnitudes, since `|A diag(m) B| ≤ |A| |m| |B|`.
    fn affine_rounding(
        factor: &ExactFactor,
        mask: ArrayView1<'_, f64>,
        rows: ArrayView2<'_, f64>,
        bias: ArrayView1<'_, f64>,
    ) -> Array2<f64> {
        let growth = accumulation_growth(factor.components() + factor.read().ncols() + 2);
        (magnitude(factor, mask, rows.mapv(f64::abs).view()) + &bias.mapv(f64::abs))
            .mapv(|value| growth * value / (1.0 - growth))
    }

    /// The longest chain of rounded operations in either route band below, all
    /// on nonnegative operands:
    /// - the stream error takes 3;
    /// - its row norm takes `d + 1`, and the normalizer slope `d + 7`;
    /// - the MLP input error takes 3 more;
    /// - the read-in propagation takes `C + d + 1`, then 1;
    /// - the write-out propagation takes `C̄ + H + 1`, then 3;
    /// - the sum of the two routes takes 1.
    const BAND_OPERATIONS: usize =
        2 * WIDTH + READ_IN_COMPONENTS + HIDDEN + WRITE_OUT_COMPONENTS + 18;

    /// Divides a computed band by `1 − γ_(k+1)`, with `k` = [`BAND_OPERATIONS`] and
    /// the extra operation this division. Every operation of the band arithmetic
    /// scales its chain by a factor `(1 + δ)^(±1)`, or `(1 + δ)^(1/2)` for a square
    /// root, with `|δ| ≤ u`. So the computed band dominates its real value.
    fn dominate(band: Array2<f64>) -> Array2<f64> {
        let factor = 1.0 - accumulation_growth(BAND_OPERATIONS + 1);
        band.mapv(|value| value / factor)
    }

    /// Per output entry, a bound on the distance of one route's computed sequential
    /// RMSNorm ReLU block from the exact block with the edited tensors.
    ///
    /// Both routes normalize the same `h` with the same code, so they hand the
    /// attention identical bits `x = fl(N₁(h))`. Let `o*` be the exact sublayer at
    /// `x`, the same for both routes (P6). The reference is
    /// `s* = h + o*`, `h'* = s* + W₂(m̄) relu(W₁(m) N₂(s*) + b₁) + b₂`.
    /// - **Stream.** `|ŝ − s*| ≤ r̂ + u|ŝ|/(1 − u)`, with `r̂` the attention's
    ///   output radius (first order in its trigonometric input error).
    /// - **MLP input.** `|fl(N₂(ŝ)) − N₂(ŝ)| ≤ γ_(d+6) |N₂(ŝ)|`: the squares, `d − 1`
    ///   additions, the mean, `+ ε`, the square root, the reciprocal, the product
    ///   with the row and the gain. The Jacobian of `h ↦ h (‖h‖²/d + ε)^(-1/2)` has
    ///   spectral norm `ρ(h) = (‖h‖²/d + ε)^(-1/2)`: its eigenvalues are `ρ` on
    ///   `h^⊥` and `ρ ε/(‖h‖²/d + ε)` along `h`. So the mean value inequality bounds
    ///   `|N₂(ŝ)_i − N₂(s*)_i|` by `|w_i| sup ρ · ‖ŝ − s*‖₂`. The supremum is over
    ///   the segment, whose norms stay at least `‖ŝ‖/2` once `4‖ŝ − s*‖ ≤ ‖ŝ‖`,
    ///   which the band asserts.
    /// - **Pre-activation.** The route's own rounding ([`affine_rounding`]) plus
    ///   `|U| |m| |R|` times the MLP input error.
    /// - **Activation.** `relu` is exact and 1-Lipschitz, so the pre-activation
    ///   error carries through unchanged.
    /// - **Output.** The route's rounding of the write, `|Ū| |m̄| |R̄|` times the
    ///   activation error, the stream error, and the rounding of the last
    ///   addition.
    fn sequential_route_band(
        fixture: &Fixture,
        block: &ComponentBlock,
        masks: &FixtureMasks,
        execution: &BlockExecution,
    ) -> Array2<f64> {
        let stream = &fixture.residual + &execution.attention.output;
        let stream_error = &execution.attention.output_radius + &addition_rounding(&stream);
        let norm_growth = accumulation_growth(WIDTH + 6);
        let mut input_error = execution
            .mlp_input
            .mapv(|value| norm_growth * value.abs() / (1.0 - norm_growth));
        for token in 0..TOKENS {
            let displacement = stream_error.row(token).iter().map(|error| error * error).sum::<f64>().sqrt();
            let stream_norm = stream.row(token).iter().map(|value| value * value).sum::<f64>().sqrt();
            assert!(
                4.0 * displacement <= stream_norm,
                "row {token}: the stream error {displacement:e} must stay below a quarter of the row norm {stream_norm:e} for the normalizer slope bound"
            );
            let half = stream_norm / 2.0;
            let slope = (half * half / WIDTH as f64 + RMS_EPSILON).sqrt().recip();
            for (slot, gain) in input_error.row_mut(token).iter_mut().zip(fixture.mlp_gain.iter()) {
                *slot += gain.abs() * slope * displacement;
            }
        }
        let mlp = block.mlp();
        let hidden_error = affine_rounding(
            mlp.read_in(),
            masks.read_in.view(),
            execution.mlp_input.view(),
            fixture.bias_in.view(),
        ) + &magnitude(mlp.read_in(), masks.read_in.view(), input_error.view());
        affine_rounding(
            mlp.write_out(),
            masks.write_out.view(),
            execution.mlp_activations.view(),
            fixture.bias_out.view(),
        ) + &magnitude(mlp.write_out(), masks.write_out.view(), hidden_error.view())
            + &stream_error
            + &addition_rounding(&execution.output)
    }

    /// The parallel layout's route band. Both routes normalize the same `h` for
    /// both sublayers, so the MLP input carries no upstream error. The output
    /// `fl(fl(ŵ + ô) + h)` adds the write's error, the attention radius, and the
    /// rounding of the two additions.
    fn parallel_route_band(
        fixture: &Fixture,
        block: &ComponentBlock,
        masks: &FixtureMasks,
        execution: &BlockExecution,
    ) -> Array2<f64> {
        let mlp = block.mlp();
        let hidden_error = affine_rounding(
            mlp.read_in(),
            masks.read_in.view(),
            execution.mlp_input.view(),
            fixture.bias_in.view(),
        );
        let writes = &execution.mlp_write + &execution.attention.output;
        affine_rounding(
            mlp.write_out(),
            masks.write_out.view(),
            execution.mlp_activations.view(),
            fixture.bias_out.view(),
        ) + &magnitude(mlp.write_out(), masks.write_out.view(), hidden_error.view())
            + &execution.attention.output_radius
            + &addition_rounding(&writes)
            + &addition_rounding(&execution.output)
    }

    fn route_band(
        fixture: &Fixture,
        block: &ComponentBlock,
        masks: &FixtureMasks,
        execution: &BlockExecution,
    ) -> Array2<f64> {
        match fixture.layout {
            ResidualLayout::Sequential => sequential_route_band(fixture, block, masks, execution),
            ResidualLayout::Parallel => parallel_route_band(fixture, block, masks, execution),
        }
    }

    fn violations(left: &Array2<f64>, right: &Array2<f64>, band: &Array2<f64>) -> usize {
        left.iter()
            .zip(right.iter())
            .zip(band.iter())
            .filter(|((a, b), bound)| (*a - *b).abs() > **bound)
            .count()
    }

    fn stage_bits(execution: &BlockExecution) -> Vec<u64> {
        execution
            .attention_input
            .iter()
            .chain(execution.attention.output.iter())
            .chain(execution.attention.weights.iter())
            .chain(execution.mlp_input.iter())
            .chain(execution.mlp_summed_input.iter())
            .chain(execution.mlp_activations.iter())
            .chain(execution.mlp_write.iter())
            .chain(execution.output.iter())
            .map(|value| value.to_bits())
            .collect()
    }

    fn bits(values: &Array2<f64>) -> Vec<u64> {
        values.iter().map(|value| value.to_bits()).collect()
    }

    /// A1 over a whole block: for continuous, binary and signed masks, the
    /// component block equals direct execution with the edited tensors within the
    /// sum of the two routes' derived bands, in both layouts.
    #[test]
    fn a_masked_block_equals_the_edited_tensor_block_within_the_derived_band() {
        for (layout, seed) in [(ResidualLayout::Sequential, 2961), (ResidualLayout::Parallel, 2962)] {
            let fixture = Fixture::new(layout, GaussianActivation::Relu, seed);
            let block = fixture.component_block();
            assert_eq!(
                (block.mlp().read_in().write(), block.mlp().write_out().write()),
                (fixture.read_in_candidate.view(), fixture.write_out_candidate.view()),
                "{layout:?}: an exactly factored weight's exact write is its candidate, so the edited tensors are formed without rounding"
            );
            let mut rng = StdRng::seed_from_u64(seed + 100);
            for (kind, masks) in FixtureMasks::family(&mut rng) {
                let attention_masks = masks.attention();
                let component = block
                    .execute(masks.block(&attention_masks), fixture.residual.view(), &fixture.positions)
                    .expect("masked block");
                let edited = fixture
                    .edited_block(&block, &masks)
                    .execute(fixture.residual.view(), &fixture.positions)
                    .expect("edited-tensor block");
                let band = dominate(
                    route_band(&fixture, &block, &masks, &component) + &route_band(&fixture, &block, &masks, &edited),
                );
                assert_eq!(
                    violations(&component.output, &edited.output, &band),
                    0,
                    "{layout:?}, {kind} masks: the component block left the derived band around the edited-tensor block"
                );

                // Positive controls: a write-out mask moved by 1e-9, and a key mask
                // moved by 1/8 upstream of the second normalization, both leave the band.
                let mut moved_write_out = masks.clone();
                moved_write_out.write_out[3] += 1.0e-9;
                let mut moved_key = masks.clone();
                moved_key.key[1] += 0.125;
                for (control, moved) in [("write-out 1e-9", moved_write_out), ("key 1/8", moved_key)] {
                    let perturbed = fixture
                        .edited_block(&block, &moved)
                        .execute(fixture.residual.view(), &fixture.positions)
                        .expect("perturbed edited-tensor block");
                    let control_band = dominate(
                        route_band(&fixture, &block, &masks, &component)
                            + &route_band(&fixture, &block, &moved, &perturbed),
                    );
                    assert!(
                        violations(&component.output, &perturbed.output, &control_band) > 0,
                        "{layout:?}, {kind} masks: the band must resolve a {control} mask move"
                    );
                }
            }
        }
    }

    /// All-on masks run both sublayers' original tensors on their original paths,
    /// so every stage is bit-identical to the native block.
    #[test]
    fn all_on_masks_execute_the_native_block_bit_for_bit() {
        for (layout, seed) in [(ResidualLayout::Sequential, 2963), (ResidualLayout::Parallel, 2964)] {
            let fixture = Fixture::new(layout, GaussianActivation::ExactGelu, seed);
            let block = fixture.component_block();
            let native = fixture
                .native()
                .execute(fixture.residual.view(), &fixture.positions)
                .expect("native block");
            let ones = FixtureMasks::ones();
            let attention_masks = ones.attention();
            let all_on = BlockMasks {
                attention: &attention_masks,
                mlp: MlpMask {
                    read_in: ComponentMask::AllOn,
                    write_out: ComponentMask::AllOn,
                },
            };
            let executed = block
                .execute(all_on, fixture.residual.view(), &fixture.positions)
                .expect("all-on block");
            assert!(
                stage_bits(&executed) == stage_bits(&native),
                "{layout:?}: the all-on block must execute the original tensors on their original paths"
            );

            // Positive control: the factored MLP at the all-ones mask is algebraically
            // the same block but not the same bits.
            let factored = block
                .execute(ones.block(&attention_masks), fixture.residual.view(), &fixture.positions)
                .expect("all-ones factored block");
            assert!(
                stage_bits(&factored) != stage_bits(&native),
                "{layout:?}: the bit-identity check must distinguish the factored all-ones path"
            );
        }
    }

    /// The block's stages are its owners' stages on the current stream, composed in
    /// the source's order, bit for bit. A sequential MLP reads `N₂(h + A)`, and a
    /// parallel MLP reads `N₂(h)`. So a query/key mask moves a sequential block's MLP
    /// input and leaves a parallel block's bit-identical.
    #[test]
    fn each_sublayer_reads_the_current_stream_in_the_source_order() {
        for (layout, seed) in [(ResidualLayout::Sequential, 2965), (ResidualLayout::Parallel, 2966)] {
            let fixture = Fixture::new(layout, GaussianActivation::ExactGelu, seed);
            let block = fixture.component_block();
            let mut rng = StdRng::seed_from_u64(seed + 100);
            let [(kind, masks), ..] = FixtureMasks::family(&mut rng);
            let attention_masks = masks.attention();
            let execution = block
                .execute(masks.block(&attention_masks), fixture.residual.view(), &fixture.positions)
                .expect("masked block");

            let attention_input = fixture
                .norm(&fixture.attention_gain, &fixture.attention_bias)
                .as_masked_norm()
                .apply(fixture.residual.view())
                .expect("attention norm");
            let attention = ComponentAttention::new(
                fixture.native().attention,
                ComponentProjection::new(fixture.query_outputs.clone(), fixture.query_readins.clone())
                    .expect("query factor shapes agree"),
                ComponentProjection::new(fixture.key_outputs.clone(), fixture.key_readins.clone())
                    .expect("key factor shapes agree"),
            )
            .expect("attention factors match the geometry")
            .execute(&attention_masks, attention_input.view(), &fixture.positions)
            .expect("masked attention");
            let stream = match layout {
                ResidualLayout::Sequential => &fixture.residual + &attention.output,
                ResidualLayout::Parallel => fixture.residual.clone(),
            };
            let mlp_input = fixture
                .norm(&fixture.mlp_gain, &fixture.mlp_bias)
                .as_masked_norm()
                .apply(stream.view())
                .expect("MLP norm");
            let summed_input = block
                .mlp()
                .summed_input(mlp_input.view(), ComponentMask::Components(masks.read_in.view()))
                .expect("masked summed input");
            let activations = block
                .mlp()
                .native()
                .activate(summed_input.view())
                .expect("activations");
            let write = block
                .mlp()
                .written(activations.view(), ComponentMask::Components(masks.write_out.view()))
                .expect("masked write");
            let output = match layout {
                ResidualLayout::Sequential => &stream + &write,
                ResidualLayout::Parallel => &(&write + &attention.output) + &fixture.residual,
            };
            let composed = BlockExecution {
                attention_input,
                attention,
                mlp_input,
                mlp_summed_input: summed_input,
                mlp_activations: activations,
                mlp_write: write,
                output,
            };
            assert!(
                stage_bits(&execution) == stage_bits(&composed),
                "{layout:?}, {kind} masks: the block must be its owners' stages composed on the current stream"
            );

            let mut moved = masks.clone();
            moved.key[1] += 0.125;
            let moved_attention = moved.attention();
            let moved_execution = block
                .execute(moved.block(&moved_attention), fixture.residual.view(), &fixture.positions)
                .expect("moved-mask block");
            assert!(
                bits(&moved_execution.attention.output) != bits(&execution.attention.output),
                "{layout:?}: positive control: the moved key mask must reach the attention write"
            );
            let mlp_input_moved = bits(&moved_execution.mlp_input) != bits(&execution.mlp_input);
            assert_eq!(
                mlp_input_moved,
                layout == ResidualLayout::Sequential,
                "{layout:?}: a key mask must reach the MLP input exactly in the sequential layout"
            );
        }
    }

    /// Shapes are refused by the owner that reads them, typed by sublayer.
    #[test]
    fn a_block_refuses_a_normalization_of_another_width_and_positions_of_another_length() {
        let fixture = Fixture::new(ResidualLayout::Sequential, GaussianActivation::Relu, 2967);
        let masks = FixtureMasks::ones();
        let attention_masks = masks.attention();
        assert!(
            fixture
                .component_block()
                .execute(masks.block(&attention_masks), fixture.residual.view(), &fixture.positions)
                .is_ok(),
            "positive control: the fixture block executes"
        );

        let narrow_positions = &fixture.positions[..TOKENS - 1];
        assert!(
            matches!(
                fixture
                    .component_block()
                    .execute(masks.block(&attention_masks), fixture.residual.view(), narrow_positions),
                Err(BlockError::Attention(AttentionProgramError::Shape { tensor: "input", .. }))
            ),
            "positions of another length must be refused by the attention sublayer"
        );

        let wide = Fixture {
            mlp_gain: Array1::ones(WIDTH + 1),
            ..Fixture::new(ResidualLayout::Sequential, GaussianActivation::Relu, 2967)
        };
        assert!(
            matches!(
                wide.component_block()
                    .execute(masks.block(&attention_masks), wide.residual.view(), &wide.positions),
                Err(BlockError::Norm {
                    sublayer: Sublayer::Mlp,
                    error: GatedRewriteError::ShapeMismatch { .. },
                })
            ),
            "an MLP normalization gain of another width must be refused, typed by sublayer"
        );
    }
}
