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
//! The block covers what its sublayers cover: query and key projections that feed
//! the rotary embedding directly (no per-head query/key norm), an MLP with a ReLU
//! or exact GELU activation, and normalizations with the source's own gain and
//! bias. Masks on the value and output projections, on normalization gains, and on
//! a gated (SwiGLU) MLP are not carried by [`ComponentBlock`].
//!
//! # Attention-only layers
//!
//! An attention-only transformer's layer has no normalization, no MLP and no
//! bias: `h' = h + concat_h(z_h) W_Oᵀ`, with `z_h` the joint-softmax value read of
//! head `h` ([`NativeAttentionLayer`]). Its learned absolute positions enter the
//! embedding, so its rotary embedding is empty and the source's rotation leaves
//! the rows unchanged. Each of the four linear reads executes one of three ways
//! ([`ProjectionRead`]), all through the owner of matrix-free edits
//! ([`super::apply`]):
//!
//! * `Native`: the stored tensor on its original path.
//! * `Components(m)`: `U diag(m) R` over the component coordinates of the exact
//!   factor `W = U R` ([`ComponentAttentionLayer`]), with the native anchor at
//!   zero, so `U M R` is never formed.
//! * `Edited`: the stored tensor plus a factored edit `Δ` on exactly the rows whose
//!   absolute positions a [`PositionScope`] reaches. The other rows come from the
//!   same native product as the unedited layer, so they are its bits.
//!
//! Queries, keys and values go through the tensor-free attention core
//! ([`RotaryCausalAttention::attend_projected`]). Its head-mixed rows feed the
//! output read, and its forward-error radii are against the exact attention of the
//! rows it was handed. Every mask on Q, K, V and O is a real mask on the summed
//! read: the softmax and the value mix see the masked rows, never a sum of
//! per-component patterns.

use super::apply::{ApplyError, FactorView, apply_anchored_linear, native_linear};
use super::attention::{
    AttentionExecution, AttentionGeometry, AttentionProgramError, ComponentAttention,
    ComponentProjection, NativeAttention, ProjectedAttention, ProjectedRows, QueryKeyMasks,
    RotaryCausalAttention, RotaryEmbedding,
};
use super::gated_rewrite::{GatedRewriteError, MaskedNorm, ResidualLayout, decoder_layer};
use super::occurrence::PositionScope;
use super::rewrite::{
    ComponentMlp, ComponentRead, ExactFactor, FactorRefusal, MlpMask, NativeMlp, RewriteError,
};
use gam_runtime::resource::Governed;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
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
    /// An attention-only layer's residual rows do not match its positions and its
    /// model width.
    ResidualShape {
        expected: (usize, usize),
        found: (usize, usize),
    },
    /// A projection weight whose shape is not the layer geometry's.
    ProjectionShape {
        projection: AttentionProjection,
        expected: (usize, usize),
        found: (usize, usize),
    },
    /// A projection weight has no exact factor through its declared read.
    Factor {
        projection: AttentionProjection,
        refusal: FactorRefusal,
    },
    /// A matrix-free read refused its operands or its memory footprint.
    Apply {
        projection: AttentionProjection,
        error: ApplyError,
    },
    /// A component read on a layer that holds no component factors.
    NoComponentFactors { projection: AttentionProjection },
    /// An edited read at a negative absolute position, which no position scope can
    /// name.
    NegativePosition {
        projection: AttentionProjection,
        position: i64,
    },
    /// An edit declares a position that no row of the layer holds.
    EditPositionAbsent {
        projection: AttentionProjection,
        position: usize,
    },
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
            Self::ResidualShape { expected, found } => write!(
                formatter,
                "the attention-only layer's residual rows have shape {found:?}, its positions and width need {expected:?}"
            ),
            Self::ProjectionShape {
                projection,
                expected,
                found,
            } => write!(
                formatter,
                "the {projection} weight has shape {found:?}, the layer geometry needs {expected:?}"
            ),
            Self::Factor { projection, refusal } => {
                write!(formatter, "the {projection} weight has no exact factor: {refusal}")
            }
            Self::Apply { projection, error } => {
                write!(formatter, "the {projection} read refused: {error}")
            }
            Self::NoComponentFactors { projection } => write!(
                formatter,
                "a component read of the {projection} projection needs a layer with component factors"
            ),
            Self::NegativePosition {
                projection,
                position,
            } => write!(
                formatter,
                "an edited {projection} read at negative position {position}, which no position scope names"
            ),
            Self::EditPositionAbsent {
                projection,
                position,
            } => write!(
                formatter,
                "the {projection} edit declares position {position}, which no row of the layer holds"
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

/// One of an attention-only layer's four linear reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttentionProjection {
    Query,
    Key,
    Value,
    Output,
}

impl AttentionProjection {
    fn index(self) -> usize {
        match self {
            Self::Query => 0,
            Self::Key => 1,
            Self::Value => 2,
            Self::Output => 3,
        }
    }
}

impl fmt::Display for AttentionProjection {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Query => "query",
            Self::Key => "key",
            Self::Value => "value",
            Self::Output => "output",
        })
    }
}

/// A parameter edit read at declared positions: `Θ + Δ` on every row whose
/// absolute position `positions` reaches, and the stored tensor on every other row.
#[derive(Clone, Copy, Debug)]
pub struct ScopedEdit<'a> {
    /// `Δ = Σ_k u_k v_kᵀ` in the read's orientation (`occurrence`'s
    /// `ParameterEditRecord::delta_read_at`).
    pub edit: FactorView<'a>,
    pub positions: &'a PositionScope,
}

/// How one linear read of an attention-only layer executes (see the module
/// documentation).
#[derive(Clone, Copy, Debug)]
pub enum ProjectionRead<'a> {
    /// The stored tensor on its original path.
    Native,
    /// `U diag(m) R` over the component coordinates of a [`ComponentAttentionLayer`].
    /// Entries are any real numbers: continuous, binary or signed.
    Components(ArrayView1<'a, f64>),
    /// The stored tensor plus a factored edit at declared positions.
    Edited(ScopedEdit<'a>),
}

/// The reads of an attention-only layer's four projections.
#[derive(Clone, Copy, Debug)]
pub struct AttentionLayerReads<'a> {
    pub query: ProjectionRead<'a>,
    pub key: ProjectionRead<'a>,
    pub value: ProjectionRead<'a>,
    pub output: ProjectionRead<'a>,
}

impl AttentionLayerReads<'_> {
    /// Every projection on its stored tensor.
    pub fn native() -> Self {
        Self {
            query: ProjectionRead::Native,
            key: ProjectionRead::Native,
            value: ProjectionRead::Native,
            output: ProjectionRead::Native,
        }
    }
}

/// Every stage of one executed attention-only layer. Rows are positions.
#[derive(Debug)]
pub struct AttentionLayerExecution {
    /// `x W_Qᵀ` under the query read, `tokens × n_heads·head_dim`.
    pub queries: Governed<Array2<f64>>,
    /// `x W_Kᵀ` under the key read, `tokens × n_kv_heads·head_dim`.
    pub keys: Governed<Array2<f64>>,
    /// `x W_Vᵀ` under the value read, `tokens × n_kv_heads·head_dim`.
    pub values: Governed<Array2<f64>>,
    /// The scores, the attention pattern and the head-mixed rows `concat_h z_h`, each
    /// with its forward-error radius against the exact attention of the rows above.
    pub attention: ProjectedAttention,
    /// `concat_h(z_h) W_Oᵀ` under the output read: the layer's write, no residual.
    pub write: Governed<Array2<f64>>,
    /// `h + write`, the residual stream after the layer.
    pub output: Array2<f64>,
}

/// A norm-free, MLP-free, bias-free decoder layer `h' = h + concat_h(z_h) W_Oᵀ` on
/// its original tensors.
#[derive(Clone, Debug)]
pub struct NativeAttentionLayer {
    geometry: AttentionGeometry,
    attention: RotaryCausalAttention,
    query: Array2<f64>,
    key: Array2<f64>,
    value: Array2<f64>,
    output: Array2<f64>,
}

impl NativeAttentionLayer {
    /// The layer's attention core and its four weights in torch `Linear` layout:
    /// `query` is `n_heads·head_dim × model_dim`, `key` and `value` are
    /// `n_kv_heads·head_dim × model_dim`, `output` is `model_dim × n_heads·head_dim`.
    /// A source with learned absolute positions passes an empty rotary embedding.
    pub fn new(
        geometry: AttentionGeometry,
        rotary: RotaryEmbedding,
        score_scale: f64,
        query: Array2<f64>,
        key: Array2<f64>,
        value: Array2<f64>,
        output: Array2<f64>,
    ) -> Result<Self, BlockError> {
        let attention = RotaryCausalAttention::new(geometry, rotary, score_scale)?;
        let (model, query_dim, kv_dim) = (
            geometry.model_dim,
            geometry.query_dim(),
            geometry.key_value_dim(),
        );
        for (projection, weight, expected) in [
            (AttentionProjection::Query, &query, (query_dim, model)),
            (AttentionProjection::Key, &key, (kv_dim, model)),
            (AttentionProjection::Value, &value, (kv_dim, model)),
            (AttentionProjection::Output, &output, (model, query_dim)),
        ] {
            if weight.dim() != expected {
                return Err(BlockError::ProjectionShape {
                    projection,
                    expected,
                    found: weight.dim(),
                });
            }
        }
        Ok(Self {
            geometry,
            attention,
            query,
            key,
            value,
            output,
        })
    }

    pub fn geometry(&self) -> AttentionGeometry {
        self.geometry
    }

    /// The stored weight of one projection.
    pub fn weight(&self, projection: AttentionProjection) -> ArrayView2<'_, f64> {
        match projection {
            AttentionProjection::Query => self.query.view(),
            AttentionProjection::Key => self.key.view(),
            AttentionProjection::Value => self.value.view(),
            AttentionProjection::Output => self.output.view(),
        }
    }

    /// The layer under `reads`, over the residual rows at their absolute positions.
    /// A component read is refused: this layer holds no component factors.
    pub fn execute(
        &self,
        reads: AttentionLayerReads<'_>,
        residual: ArrayView2<'_, f64>,
        positions: &[i64],
    ) -> Result<AttentionLayerExecution, BlockError> {
        execute_attention_layer(self, None, reads, residual, positions)
    }
}

/// The exact factor `W = U R` of one projection, with `Rᵀ` held in the column layout
/// the matrix-free kernels read.
#[derive(Clone, Debug)]
struct ProjectionFactor {
    factor: ExactFactor,
    read_transpose: Array2<f64>,
}

/// An attention-only layer whose four projections can execute through masked
/// component coordinates.
#[derive(Clone, Debug)]
pub struct ComponentAttentionLayer {
    native: NativeAttentionLayer,
    factors: [ProjectionFactor; 4],
}

impl ComponentAttentionLayer {
    /// Solves each projection's exact write through its declared read (rewrite's
    /// overcomplete factor), refusing an uncovered or unresolved read.
    pub fn new(
        native: NativeAttentionLayer,
        query: ComponentRead<'_>,
        key: ComponentRead<'_>,
        value: ComponentRead<'_>,
        output: ComponentRead<'_>,
    ) -> Result<Self, BlockError> {
        let solve = |projection: AttentionProjection, read: ComponentRead<'_>| {
            ExactFactor::solve_write(native.weight(projection), read.read, read.candidate_write)
                .map(|factor| ProjectionFactor {
                    read_transpose: factor.read().t().to_owned(),
                    factor,
                })
                .map_err(|refusal| BlockError::Factor { projection, refusal })
        };
        let factors = [
            solve(AttentionProjection::Query, query)?,
            solve(AttentionProjection::Key, key)?,
            solve(AttentionProjection::Value, value)?,
            solve(AttentionProjection::Output, output)?,
        ];
        Ok(Self { native, factors })
    }

    /// The layer on its stored tensors.
    pub fn native(&self) -> &NativeAttentionLayer {
        &self.native
    }

    /// The exact factor `W = U R` of one projection.
    pub fn factor(&self, projection: AttentionProjection) -> &ExactFactor {
        &self.factors[projection.index()].factor
    }

    /// The layer under `reads`, over the residual rows at their absolute positions.
    pub fn execute(
        &self,
        reads: AttentionLayerReads<'_>,
        residual: ArrayView2<'_, f64>,
        positions: &[i64],
    ) -> Result<AttentionLayerExecution, BlockError> {
        execute_attention_layer(&self.native, Some(&self.factors), reads, residual, positions)
    }
}

fn execute_attention_layer(
    layer: &NativeAttentionLayer,
    factors: Option<&[ProjectionFactor; 4]>,
    reads: AttentionLayerReads<'_>,
    residual: ArrayView2<'_, f64>,
    positions: &[i64],
) -> Result<AttentionLayerExecution, BlockError> {
    let expected = (positions.len(), layer.geometry.model_dim);
    if residual.dim() != expected {
        return Err(BlockError::ResidualShape {
            expected,
            found: residual.dim(),
        });
    }
    let factor = |projection: AttentionProjection| factors.map(|all| &all[projection.index()]);
    let queries = read_projection(
        layer.weight(AttentionProjection::Query),
        factor(AttentionProjection::Query),
        AttentionProjection::Query,
        reads.query,
        residual,
        positions,
    )?;
    let keys = read_projection(
        layer.weight(AttentionProjection::Key),
        factor(AttentionProjection::Key),
        AttentionProjection::Key,
        reads.key,
        residual,
        positions,
    )?;
    let values = read_projection(
        layer.weight(AttentionProjection::Value),
        factor(AttentionProjection::Value),
        AttentionProjection::Value,
        reads.value,
        residual,
        positions,
    )?;
    let attention = layer.attention.attend_projected(
        ProjectedRows::exact(queries.view()),
        ProjectedRows::exact(keys.view()),
        ProjectedRows::exact(values.view()),
        positions,
    )?;
    let write = read_projection(
        layer.weight(AttentionProjection::Output),
        factor(AttentionProjection::Output),
        AttentionProjection::Output,
        reads.output,
        attention.mixed.view(),
        positions,
    )?;
    let mut output = residual.to_owned();
    output += &*write;
    Ok(AttentionLayerExecution {
        queries,
        keys,
        values,
        attention,
        write,
        output,
    })
}

/// One linear read of `rows`, whose row `r` sits at absolute position `positions[r]`.
fn read_projection(
    weight: ArrayView2<'_, f64>,
    factor: Option<&ProjectionFactor>,
    projection: AttentionProjection,
    read: ProjectionRead<'_>,
    rows: ArrayView2<'_, f64>,
    positions: &[i64],
) -> Result<Governed<Array2<f64>>, BlockError> {
    let refused = |error: ApplyError| BlockError::Apply { projection, error };
    match read {
        ProjectionRead::Native => native_linear(weight, rows).map_err(refused),
        ProjectionRead::Components(mask) => {
            let factor = factor.ok_or(BlockError::NoComponentFactors { projection })?;
            let components = FactorView::new(factor.factor.write(), factor.read_transpose.view())
                .map_err(refused)?;
            apply_anchored_linear(weight, 0.0, components, mask, rows).map_err(refused)
        }
        ProjectionRead::Edited(scoped) => {
            let mut row_positions = Vec::with_capacity(positions.len());
            for &position in positions {
                row_positions.push(
                    usize::try_from(position)
                        .ok()
                        .ok_or(BlockError::NegativePosition {
                            projection,
                            position,
                        })?,
                );
            }
            if let Some(&absent) = scoped
                .positions
                .positions()
                .and_then(|declared| declared.iter().find(|position| !row_positions.contains(position)))
            {
                return Err(BlockError::EditPositionAbsent {
                    projection,
                    position: absent,
                });
            }
            let mut written = native_linear(weight, rows).map_err(refused)?;
            let reached: Vec<usize> = row_positions
                .iter()
                .enumerate()
                .filter_map(|(row, position)| scoped.positions.reaches(*position).then_some(row))
                .collect();
            if reached.is_empty() {
                return Ok(written);
            }
            let selected = rows.select(Axis(0), &reached);
            let every_term = Array1::<f64>::ones(scoped.edit.term_count());
            let edited = apply_anchored_linear(weight, 1.0, scoped.edit, every_term.view(), selected.view())
                .map_err(refused)?;
            for (index, &row) in reached.iter().enumerate() {
                written.row_mut(row).assign(&edited.row(index));
            }
            Ok(written)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parameter_decomposition::attention::{AffineProjection, RotaryPairing};
    use crate::parameter_decomposition::rewrite::ComponentMask;
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

    const LAYER_WIDTH: usize = 8;
    const LAYER_HEAD_DIM: usize = 4;
    const LAYER_TOKENS: usize = 6;
    const LAYER_COMPONENTS: usize = 9;
    /// `1/sqrt(head_dim)` at `head_dim = 4`, exactly.
    const LAYER_SCORE_SCALE: f64 = 0.5;

    fn layer_geometry() -> AttentionGeometry {
        AttentionGeometry {
            model_dim: LAYER_WIDTH,
            n_heads: 2,
            n_kv_heads: 2,
            head_dim: LAYER_HEAD_DIM,
        }
    }

    /// Learned absolute positions: the source rotates no plane.
    fn no_rotary() -> RotaryEmbedding {
        RotaryEmbedding {
            pairing: RotaryPairing::HalfSplit,
            inverse_frequencies: Vec::new(),
            attention_scaling: 1.0,
        }
    }

    const PROJECTIONS: [AttentionProjection; 4] = [
        AttentionProjection::Query,
        AttentionProjection::Key,
        AttentionProjection::Value,
        AttentionProjection::Output,
    ];

    /// An attention-only layer whose four weights factor exactly through overcomplete
    /// reads in eighths, `W = N R`, so each exact write is its candidate and every edited
    /// tensor `U diag(m) R` is formed without rounding.
    struct LayerFixture {
        candidates: [Array2<f64>; 4],
        reads: [Array2<f64>; 4],
        residual: Array2<f64>,
        positions: Vec<i64>,
    }

    impl LayerFixture {
        fn new(seed: u64) -> Self {
            let mut rng = StdRng::seed_from_u64(seed);
            let candidates = [
                eighths(&mut rng, LAYER_WIDTH, LAYER_COMPONENTS),
                eighths(&mut rng, LAYER_WIDTH, LAYER_COMPONENTS),
                eighths(&mut rng, LAYER_WIDTH, LAYER_COMPONENTS),
                eighths(&mut rng, LAYER_WIDTH, LAYER_COMPONENTS),
            ];
            let reads = [
                eighths(&mut rng, LAYER_COMPONENTS, LAYER_WIDTH),
                eighths(&mut rng, LAYER_COMPONENTS, LAYER_WIDTH),
                eighths(&mut rng, LAYER_COMPONENTS, LAYER_WIDTH),
                eighths(&mut rng, LAYER_COMPONENTS, LAYER_WIDTH),
            ];
            Self {
                candidates,
                reads,
                // Thirds of eighths in `[-2, 2]`. With eighth-valued rows every linear read is exact
                // in f64, so two routes with the same algebra would share their bits: the bitwise
                // checks could not see a route change, and the derived bands would never meet rounding.
                residual: Array2::from_shape_simple_fn((LAYER_TOKENS, LAYER_WIDTH), || {
                    rng.random_range(-48..=48) as f64 / 24.0
                }),
                positions: (0..LAYER_TOKENS as i64).collect(),
            }
        }

        fn native_with(&self, weights: [Array2<f64>; 4]) -> NativeAttentionLayer {
            let [query, key, value, output] = weights;
            NativeAttentionLayer::new(
                layer_geometry(),
                no_rotary(),
                LAYER_SCORE_SCALE,
                query,
                key,
                value,
                output,
            )
            .expect("fixture weights match the geometry")
        }

        /// The source layer: every weight is its factors' product.
        fn native(&self) -> NativeAttentionLayer {
            self.native_with([0, 1, 2, 3].map(|index| self.candidates[index].dot(&self.reads[index])))
        }

        fn component(&self) -> ComponentAttentionLayer {
            let read = |index: usize| ComponentRead {
                read: self.reads[index].view(),
                candidate_write: self.candidates[index].view(),
            };
            ComponentAttentionLayer::new(self.native(), read(0), read(1), read(2), read(3))
                .expect("random overcomplete reads are resolved")
        }

        /// Direct execution with the edited tensors `U diag(m) R`.
        fn edited(&self, layer: &ComponentAttentionLayer, masks: &LayerMasks) -> NativeAttentionLayer {
            self.native_with(PROJECTIONS.map(|projection| {
                let factor = layer.factor(projection);
                masked_product(factor.write(), masks.of(projection), factor.read())
            }))
        }
    }

    #[derive(Clone)]
    struct LayerMasks {
        query: Array1<f64>,
        key: Array1<f64>,
        value: Array1<f64>,
        output: Array1<f64>,
    }

    impl LayerMasks {
        /// Eighths: continuous in `[0, 1]`, binary, and signed in `[-3/2, 3/2]`.
        fn family(rng: &mut StdRng) -> [(&'static str, Self); 3] {
            let mut draw = |low: i32, high: i32, denominator: f64| {
                let mut vector = || {
                    Array1::from_shape_simple_fn(LAYER_COMPONENTS, || rng.random_range(low..=high) as f64 / denominator)
                };
                Self {
                    query: vector(),
                    key: vector(),
                    value: vector(),
                    output: vector(),
                }
            };
            [
                ("continuous", draw(0, 8, 8.0)),
                ("binary", draw(0, 1, 1.0)),
                ("signed", draw(-12, 12, 8.0)),
            ]
        }

        fn of(&self, projection: AttentionProjection) -> ArrayView1<'_, f64> {
            match projection {
                AttentionProjection::Query => self.query.view(),
                AttentionProjection::Key => self.key.view(),
                AttentionProjection::Value => self.value.view(),
                AttentionProjection::Output => self.output.view(),
            }
        }

        fn reads(&self) -> AttentionLayerReads<'_> {
            AttentionLayerReads {
                query: ProjectionRead::Components(self.query.view()),
                key: ProjectionRead::Components(self.key.view()),
                value: ProjectionRead::Components(self.value.view()),
                output: ProjectionRead::Components(self.output.view()),
            }
        }
    }

    /// Rounding of `A diag(m) B x` through the matrix-free kernel on either route. The
    /// factored route passes each term through the products `b_c x` (`d` operations), the
    /// scale, the product with `a_c` and `C − 1` sums, and the accumulation into the tile:
    /// `C + d + 2`. The edited tensor's native product takes `d` operations over terms
    /// bounded entrywise by the same magnitudes.
    fn component_rounding(
        factor: &ExactFactor,
        mask: ArrayView1<'_, f64>,
        rows: ArrayView2<'_, f64>,
    ) -> Array2<f64> {
        let growth = accumulation_growth(factor.components() + factor.read().ncols() + 2);
        magnitude(factor, mask, rows.mapv(f64::abs).view()).mapv(|value| growth * value / (1.0 - growth))
    }

    /// Per output entry, a bound on one route's distance from the exact layer with the
    /// edited tensors. Each read's rounding band enters the owner's attention radius as that
    /// read's input radius (first order), so the mixed radius covers the exact attention at the
    /// exact reads. The output read adds its own rounding and `|U| |m| |R|` times the mixed
    /// radius, and the residual addition rounds once.
    fn attention_layer_route_band(
        layer: &ComponentAttentionLayer,
        masks: &LayerMasks,
        fixture: &LayerFixture,
        execution: &AttentionLayerExecution,
    ) -> Array2<f64> {
        let radius = |projection: AttentionProjection| {
            dominate(component_rounding(
                layer.factor(projection),
                masks.of(projection),
                fixture.residual.view(),
            ))
        };
        let (query_radius, key_radius, value_radius) = (
            radius(AttentionProjection::Query),
            radius(AttentionProjection::Key),
            radius(AttentionProjection::Value),
        );
        let attention = RotaryCausalAttention::new(layer_geometry(), no_rotary(), LAYER_SCORE_SCALE)
            .expect("the fixture attention core");
        let with_radius = attention
            .attend_projected(
                ProjectedRows {
                    values: execution.queries.view(),
                    radius: query_radius.view(),
                },
                ProjectedRows {
                    values: execution.keys.view(),
                    radius: key_radius.view(),
                },
                ProjectedRows {
                    values: execution.values.view(),
                    radius: value_radius.view(),
                },
                &fixture.positions,
            )
            .expect("finite projected rows");
        assert!(
            bits(&with_radius.mixed) == bits(&execution.attention.mixed),
            "the radius arithmetic must leave the mixed rows' bits alone"
        );
        let output = layer.factor(AttentionProjection::Output);
        component_rounding(output, masks.output.view(), execution.attention.mixed.view())
            + &magnitude(output, masks.output.view(), with_radius.mixed_radius.view())
            + &addition_rounding(&execution.output)
    }

    fn layer_stage_bits(execution: &AttentionLayerExecution) -> Vec<u64> {
        execution
            .queries
            .iter()
            .chain(execution.keys.iter())
            .chain(execution.values.iter())
            .chain(execution.attention.weights.iter())
            .chain(execution.attention.mixed.iter())
            .chain(execution.write.iter())
            .chain(execution.output.iter())
            .map(|value| value.to_bits())
            .collect()
    }

    fn row_bits(values: &Array2<f64>, row: usize) -> Vec<u64> {
        values.row(row).iter().map(|value| value.to_bits()).collect()
    }

    /// A1 for an attention-only layer: for continuous, binary and signed masks on Q, K, V and
    /// O, the component layer equals direct execution with the edited tensors within the sum of
    /// the two routes' derived bands.
    #[test]
    fn a_masked_attention_layer_equals_the_edited_tensor_layer_within_the_derived_band() {
        let fixture = LayerFixture::new(2981);
        let layer = fixture.component();
        for projection in PROJECTIONS {
            assert_eq!(
                layer.factor(projection).write(),
                fixture.candidates[projection.index()].view(),
                "the {projection} weight factors exactly through its candidate, so the edited tensors carry no rounding"
            );
        }
        let mut rng = StdRng::seed_from_u64(2982);
        for (kind, masks) in LayerMasks::family(&mut rng) {
            let component = layer
                .execute(masks.reads(), fixture.residual.view(), &fixture.positions)
                .expect("masked layer");
            let edited = fixture
                .edited(&layer, &masks)
                .execute(AttentionLayerReads::native(), fixture.residual.view(), &fixture.positions)
                .expect("edited-tensor layer");
            let band = dominate(
                attention_layer_route_band(&layer, &masks, &fixture, &component)
                    + &attention_layer_route_band(&layer, &masks, &fixture, &edited),
            );
            assert_eq!(
                violations(&component.output, &edited.output, &band),
                0,
                "{kind} masks: the component layer left the derived band around the edited-tensor layer"
            );

            // Positive controls: a value mask and an output mask moved by 1e-9 each leave the band.
            let mut moved_value = masks.clone();
            moved_value.value[2] += 1.0e-9;
            let mut moved_output = masks.clone();
            moved_output.output[5] += 1.0e-9;
            for (control, moved) in [("value 1e-9", moved_value), ("output 1e-9", moved_output)] {
                let perturbed = fixture
                    .edited(&layer, &moved)
                    .execute(AttentionLayerReads::native(), fixture.residual.view(), &fixture.positions)
                    .expect("perturbed edited-tensor layer");
                let control_band = dominate(
                    attention_layer_route_band(&layer, &masks, &fixture, &component)
                        + &attention_layer_route_band(&layer, &moved, &fixture, &perturbed),
                );
                assert!(
                    violations(&component.output, &perturbed.output, &control_band) > 0,
                    "{kind} masks: the band must resolve a {control} mask move"
                );
            }
        }
    }

    /// All-on reads run every projection on its stored tensor, so each stage is bit-identical
    /// to the native layer's.
    #[test]
    fn all_on_reads_execute_the_native_attention_layer_bit_for_bit() {
        let fixture = LayerFixture::new(2983);
        let layer = fixture.component();
        let native = fixture
            .native()
            .execute(AttentionLayerReads::native(), fixture.residual.view(), &fixture.positions)
            .expect("native layer");
        let all_on = layer
            .execute(AttentionLayerReads::native(), fixture.residual.view(), &fixture.positions)
            .expect("all-on component layer");
        assert!(
            layer_stage_bits(&all_on) == layer_stage_bits(&native),
            "all-on reads must execute the stored tensors on their original paths"
        );

        // Positive control: the factored all-ones value read is algebraically the same layer but
        // not the same bits.
        let ones = Array1::<f64>::ones(LAYER_COMPONENTS);
        let factored = layer
            .execute(
                AttentionLayerReads {
                    value: ProjectionRead::Components(ones.view()),
                    ..AttentionLayerReads::native()
                },
                fixture.residual.view(),
                &fixture.positions,
            )
            .expect("all-ones factored value read");
        assert!(
            layer_stage_bits(&factored) != layer_stage_bits(&native),
            "the bit-identity check must distinguish the factored all-ones value path"
        );
    }

    /// The occurrence test's shape: head 1 of layer 0 ablated in the output write, `Δ = −W_O[:,
    /// 4..8] e_{4..7}ᵀ`. Scoped at one declared row, it moves exactly that row of layer 0's write,
    /// every earlier logits row of a 2-layer stack keeps its bits, and the declared row equals
    /// the dense edited layer within the derived band. Scoped at every position, every row
    /// does.
    #[test]
    fn a_scoped_output_edit_moves_exactly_its_declared_rows_through_a_two_layer_stack() {
        let (first, second) = (LayerFixture::new(2984), LayerFixture::new(2985));
        let (layer0, layer1) = (first.native(), second.native());
        let mut rng = StdRng::seed_from_u64(2986);
        let unembed = eighths(&mut rng, 5, LAYER_WIDTH);
        let head_columns: Vec<usize> = (LAYER_HEAD_DIM..2 * LAYER_HEAD_DIM).collect();
        let output_weight = layer0.weight(AttentionProjection::Output).to_owned();
        let left = output_weight.select(Axis(1), &head_columns).mapv(|value| -value);
        let mut right = Array2::<f64>::zeros((LAYER_WIDTH, LAYER_HEAD_DIM));
        for (term, &column) in head_columns.iter().enumerate() {
            right[[column, term]] = 1.0;
        }
        let edit = FactorView::new(left.view(), right.view()).expect("finite edit factors");
        let stack = |reads: AttentionLayerReads<'_>| {
            let hidden = layer0
                .execute(reads, first.residual.view(), &first.positions)
                .expect("layer 0");
            let top = layer1
                .execute(AttentionLayerReads::native(), hidden.output.view(), &first.positions)
                .expect("layer 1");
            let logits = native_linear(unembed.view(), top.output.view())
                .expect("unembedding")
                .to_owned();
            (hidden, logits)
        };
        let (native_hidden, native_logits) = stack(AttentionLayerReads::native());
        let declared = 3_usize;
        let scope = PositionScope::declared(vec![declared]).expect("one declared position");
        let (scoped_hidden, scoped_logits) = stack(AttentionLayerReads {
            output: ProjectionRead::Edited(ScopedEdit {
                edit,
                positions: &scope,
            }),
            ..AttentionLayerReads::native()
        });
        for row in 0..LAYER_TOKENS {
            assert_eq!(
                row_bits(&native_hidden.write, row) == row_bits(&scoped_hidden.write, row),
                row != declared,
                "row {row}: an output edit scoped at {declared} must move exactly its declared row of the write"
            );
            if row < declared {
                assert!(
                    row_bits(&native_logits, row) == row_bits(&scoped_logits, row),
                    "logits row {row} precedes the edited row, so it must keep its bits"
                );
            }
        }
        assert!(
            row_bits(&native_logits, declared) != row_bits(&scoped_logits, declared),
            "positive control: the logits at the declared row move"
        );

        // The dense edited layer: `W_O + left rightᵀ` zeroes head 1's columns exactly.
        let ablated = &output_weight + &left.dot(&right.t());
        let dense_layer = first.native_with([
            layer0.weight(AttentionProjection::Query).to_owned(),
            layer0.weight(AttentionProjection::Key).to_owned(),
            layer0.weight(AttentionProjection::Value).to_owned(),
            ablated.clone(),
        ]);
        let dense = dense_layer
            .execute(AttentionLayerReads::native(), first.residual.view(), &first.positions)
            .expect("dense edited layer");
        assert!(
            bits(&dense.attention.mixed) == bits(&scoped_hidden.attention.mixed),
            "the edit reaches only the output read, so the mixed rows keep their bits"
        );
        // The scoped route forms `W_O x` (`H` operations) plus the terms `left (rightᵀ x)`
        // (`d + 1 + H + 1`), the dense route `W_O' x` (`H`); `H = d` here.
        let mixed_abs = dense.attention.mixed.mapv(f64::abs);
        let growth = accumulation_growth(2 * LAYER_WIDTH + 2);
        let band = dominate(
            (mixed_abs.dot(&output_weight.mapv(f64::abs).t())
                + &mixed_abs.dot(&right.mapv(f64::abs)).dot(&left.mapv(f64::abs).t())
                + &mixed_abs.dot(&ablated.mapv(f64::abs).t()))
            .mapv(|value| growth * value / (1.0 - growth)),
        );
        let row_violations = |left_rows: &Array2<f64>, right_rows: &Array2<f64>, row: usize| {
            (0..LAYER_WIDTH)
                .filter(|&column| (left_rows[[row, column]] - right_rows[[row, column]]).abs() > band[[row, column]])
                .count()
        };
        assert_eq!(
            row_violations(&scoped_hidden.write, &dense.write, declared),
            0,
            "the declared row of the scoped write must equal the dense edited layer within the derived band"
        );
        assert!(
            row_violations(&native_hidden.write, &dense.write, declared) > 0,
            "positive control: the unedited write at the declared row must leave the band"
        );

        let every = PositionScope::every();
        let global = layer0
            .execute(
                AttentionLayerReads {
                    output: ProjectionRead::Edited(ScopedEdit {
                        edit,
                        positions: &every,
                    }),
                    ..AttentionLayerReads::native()
                },
                first.residual.view(),
                &first.positions,
            )
            .expect("globally edited layer 0");
        assert_eq!(
            violations(&global.write, &dense.write, &band),
            0,
            "an output edit at every position must equal the dense edited layer within the derived band"
        );
    }

    /// Reads outside the layer's domain are refused, typed by projection.
    #[test]
    fn an_attention_layer_refuses_reads_outside_its_domain() {
        let fixture = LayerFixture::new(2987);
        let native = fixture.native();
        let layer = fixture.component();
        assert!(
            native
                .execute(AttentionLayerReads::native(), fixture.residual.view(), &fixture.positions)
                .is_ok(),
            "positive control: the fixture layer executes"
        );

        let ones = Array1::<f64>::ones(LAYER_COMPONENTS);
        assert!(
            matches!(
                native.execute(
                    AttentionLayerReads {
                        value: ProjectionRead::Components(ones.view()),
                        ..AttentionLayerReads::native()
                    },
                    fixture.residual.view(),
                    &fixture.positions,
                ),
                Err(BlockError::NoComponentFactors {
                    projection: AttentionProjection::Value
                })
            ),
            "a component read on a layer without factors must be refused"
        );

        let short = Array1::<f64>::ones(LAYER_COMPONENTS - 1);
        assert!(
            matches!(
                layer.execute(
                    AttentionLayerReads {
                        key: ProjectionRead::Components(short.view()),
                        ..AttentionLayerReads::native()
                    },
                    fixture.residual.view(),
                    &fixture.positions,
                ),
                Err(BlockError::Apply {
                    projection: AttentionProjection::Key,
                    error: ApplyError::Shape { .. }
                })
            ),
            "a mask of another length must be refused by the matrix-free read"
        );

        let (left, right) = (Array2::<f64>::ones((LAYER_WIDTH, 1)), Array2::<f64>::ones((LAYER_WIDTH, 1)));
        let edit = FactorView::new(left.view(), right.view()).expect("finite edit factors");
        let absent = PositionScope::declared(vec![LAYER_TOKENS]).expect("one declared position");
        assert!(
            matches!(
                native.execute(
                    AttentionLayerReads {
                        output: ProjectionRead::Edited(ScopedEdit {
                            edit,
                            positions: &absent,
                        }),
                        ..AttentionLayerReads::native()
                    },
                    fixture.residual.view(),
                    &fixture.positions,
                ),
                Err(BlockError::EditPositionAbsent {
                    projection: AttentionProjection::Output,
                    position,
                }) if position == LAYER_TOKENS
            ),
            "an edit declaring a position no row holds must be refused"
        );

        let first = PositionScope::declared(vec![0]).expect("one declared position");
        let shifted: Vec<i64> = fixture.positions.iter().map(|position| position - 1).collect();
        assert!(
            matches!(
                native.execute(
                    AttentionLayerReads {
                        output: ProjectionRead::Edited(ScopedEdit {
                            edit,
                            positions: &first,
                        }),
                        ..AttentionLayerReads::native()
                    },
                    fixture.residual.view(),
                    &shifted,
                ),
                Err(BlockError::NegativePosition {
                    projection: AttentionProjection::Output,
                    position: -1
                })
            ),
            "an edited read at a negative position must be refused"
        );

        let [query, key, value, output] =
            [0, 1, 2, 3].map(|index| fixture.candidates[index].dot(&fixture.reads[index]));
        assert!(
            matches!(
                NativeAttentionLayer::new(
                    layer_geometry(),
                    no_rotary(),
                    LAYER_SCORE_SCALE,
                    query,
                    key,
                    value,
                    output.select(Axis(1), &[0, 1, 2]),
                ),
                Err(BlockError::ProjectionShape {
                    projection: AttentionProjection::Output,
                    ..
                })
            ),
            "an output weight of another shape must be refused at construction"
        );
        assert_eq!(output.dim(), (LAYER_WIDTH, LAYER_WIDTH), "control: the fixture weight has the geometry's shape");
    }
}
