//! Exact attention under query-key component masks (#2951 P6).
//!
//! Factor the source's query and key projections through components,
//! `W_Q = U^Q R^Q` and `W_K = U^K R^K`, and mask the component coordinates:
//! `a_t = m^Q ⊙ R^Q x_t`, `b_s = m^K ⊙ R^K x_s`. With the source's rotary
//! embedding `R_p` at absolute position `p`, the score of query token `t` on key
//! token `s` in head `h` is
//!
//! ```text
//! S_ts = σ (U^Q_h a_t)ᵀ R_{p_t}ᵀ R_{p_s} (U^K_{g(h)} b_s)
//!      = Σ_ij a_{t,i} C_ij(p_s − p_t) b_{s,j},
//! C_ij(Δ) = σ u^{Q}_{h,i}ᵀ R_Δ u^{K}_{g(h),j},
//! ```
//!
//! because every `R_p` rotates the same planes, so `R_{p_t}ᵀ R_{p_s} = R_{p_s − p_t}`.
//! The kernel argument is the key position minus the query position; with a
//! nonzero sine part `C(Δ) ≠ C(−Δ)`, so the sign is part of the source, not a
//! convention to choose. `σ` is the source's score multiplier (its `scaling`) and
//! `g(h)` the key/value head the source's `repeat_kv` gives query head `h`.
//!
//! The scores then go through the source's causal mask and ONE joint softmax per
//! query row, and the weights read the source's value projection and output
//! projection. `softmax(Σ_c ℓ_c) ≠ Σ_c softmax(ℓ_c)`, so the components share the
//! normalization; a separately normalized sum of per-component patterns is a
//! different program (the tests refute it on a fixture).
//!
//! # The source's rotary convention
//!
//! The source rotates plane `(a, b)` of a head by `+p θ_p`:
//! `(x_a, x_b) ↦ (x_a cos − x_b sin, x_b cos + x_a sin)`, with `cos` and `sin`
//! both multiplied by the source's `attention_scaling`. Two pairings exist in
//! transformers:
//! - [`RotaryPairing::HalfSplit`]: `rotate_half` (`models/llama`, `models/qwen3`,
//!   `models/gpt_neox`), planes `(p, p + P)` over the first `2P` head coordinates;
//! - [`RotaryPairing::Interleaved`]: `rotate_every_two` (`models/gptj`), planes
//!   `(2p, 2p + 1)`.
//!
//! Head coordinates past the rotary dimension pass through unrotated and unscaled
//! (gpt_neox `rotary_ndims`). The frequencies are the source's `inv_freq` buffer
//! as exported, so every rope-scaling rule the source applied is already in them.
//!
//! # Validity domain
//!
//! The kernel identity needs the projection to feed the rotary embedding
//! directly. A per-head norm between them (Qwen3's `q_norm`/`k_norm`) is a
//! native nonlinearity on the summed query, and biases on the projections are
//! components this program does not carry; neither is represented here.
//!
//! # All-on and roundoff
//!
//! With every mask at one the program executes the original tensors on the
//! original path ([`NativeAttention::execute`]), bit for bit. Every other result
//! carries a forward-error radius against the exact value of its own program at
//! the given inputs: Higham's `γ_k · Σ|monomials|` over the rounded operations
//! (`gam_linalg::roundoff`), plus the trigonometric input error `u·|φ| + ulp` of
//! each angle under the platform libm's documented at-most-one-ulp `sin`, `cos`
//! and `exp`, propagated to first order. Two programs with the same exact value
//! agree within the sum of their radii.

use std::collections::BTreeMap;
use std::fmt;

use gam_linalg::roundoff::accumulation_growth;
use ndarray::{Array1, Array2, Array3, ArrayView1, ArrayView2};

/// Largest position magnitude whose differences stay exact in `f64`:
/// `|p_s − p_t| ≤ 2^53` is representable, so `Δ as f64` rounds nothing.
const EXACT_POSITION_LIMIT: i64 = 1 << 52;

#[derive(Clone, Debug, PartialEq)]
pub enum AttentionProgramError {
    /// A tensor, input or mask whose shape does not match the declared geometry.
    Shape {
        tensor: &'static str,
        expected: (usize, usize),
        found: (usize, usize),
    },
    /// Query heads share key/value heads in contiguous groups, so `n_kv_heads`
    /// must divide `n_heads`.
    KeyValueHeadsDoNotDivide { n_heads: usize, n_kv_heads: usize },
    /// The rotary planes need more coordinates than one head has.
    RotaryExceedsHead { rotary_dim: usize, head_dim: usize },
    /// A position whose differences are not exact in `f64`.
    PositionNotExact { position: i64 },
    /// A kernel displacement beyond `2^53`, which is not exact in `f64`.
    DisplacementNotExact { displacement: i64 },
    HeadOutOfRange { head: usize, n_heads: usize },
}

impl fmt::Display for AttentionProgramError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Shape {
                tensor,
                expected,
                found,
            } => write!(f, "{tensor} has shape {found:?}, the geometry needs {expected:?}"),
            Self::KeyValueHeadsDoNotDivide {
                n_heads,
                n_kv_heads,
            } => write!(f, "{n_kv_heads} key/value heads do not divide {n_heads} query heads"),
            Self::RotaryExceedsHead {
                rotary_dim,
                head_dim,
            } => write!(f, "rotary dimension {rotary_dim} exceeds head dimension {head_dim}"),
            Self::PositionNotExact { position } => write!(
                f,
                "position {position} exceeds 2^52, so position differences are not exact in f64"
            ),
            Self::DisplacementNotExact { displacement } => {
                write!(f, "displacement {displacement} exceeds 2^53 and is not exact in f64")
            }
            Self::HeadOutOfRange { head, n_heads } => {
                write!(f, "head {head} is out of range for {n_heads} heads")
            }
        }
    }
}

impl std::error::Error for AttentionProgramError {}

/// The source attention block's dimensions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AttentionGeometry {
    pub model_dim: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
}

impl AttentionGeometry {
    pub fn query_dim(&self) -> usize {
        self.n_heads * self.head_dim
    }

    pub fn key_value_dim(&self) -> usize {
        self.n_kv_heads * self.head_dim
    }

    /// The key/value head the source's `repeat_kv` gives query head `head`.
    pub fn key_value_head(&self, head: usize) -> usize {
        head / (self.n_heads / self.n_kv_heads)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RotaryPairing {
    HalfSplit,
    Interleaved,
}

/// The source's rotary embedding, as exported from the source module.
#[derive(Clone, Debug, PartialEq)]
pub struct RotaryEmbedding {
    pub pairing: RotaryPairing,
    /// The source's `inv_freq` buffer, one frequency per rotated plane.
    pub inverse_frequencies: Vec<f64>,
    /// The source's `attention_scaling`, multiplying both `cos` and `sin`.
    pub attention_scaling: f64,
}

impl RotaryEmbedding {
    pub fn rotary_dim(&self) -> usize {
        2 * self.inverse_frequencies.len()
    }

    /// Head coordinates `(a, b)` of rotated plane `plane`.
    pub fn plane(&self, plane: usize) -> (usize, usize) {
        match self.pairing {
            RotaryPairing::HalfSplit => (plane, plane + self.inverse_frequencies.len()),
            RotaryPairing::Interleaved => (2 * plane, 2 * plane + 1),
        }
    }
}

/// `cos φ`, `sin φ` at `φ = multiplier · θ`, with the error bound `η` on each:
/// the product rounds by at most `γ_1 |φ̂|` and libm adds at most one ulp, which
/// is at most `ε |value|`.
fn plane_trig(multiplier: f64, frequency: f64) -> (f64, f64, f64) {
    let angle = multiplier * frequency;
    let (sin, cos) = angle.sin_cos();
    let eta = accumulation_growth(1) * angle.abs() + f64::EPSILON * cos.abs().max(sin.abs());
    (cos, sin, eta)
}

/// One sequential inner product and the absolute sum of its terms.
fn inner_with_abs(row: ArrayView1<f64>, x: ArrayView1<f64>) -> (f64, f64) {
    row.iter()
        .zip(x.iter())
        .fold((0.0, 0.0), |(value, abs), (w, v)| (value + w * v, abs + (w * v).abs()))
}

/// The source's joint softmax over one query row, with a radius on each weight.
///
/// Logits within `r = max_s logit_radius[s]` of exact move each exact weight by
/// at most the factor `e^{±2r}`, so `|w̃_s − w_s| ≤ w̃_s·expm1(2r)`. Evaluating
/// `exp(ℓ_s − max ℓ) / Σ exp(ℓ − max ℓ)` rounds the shift by `γ_1|d_s|`, adds a
/// libm ulp to each exponential, `γ_{n−1}` to the positive sum and `γ_1` to the
/// division, which are relative errors on the weight.
pub fn joint_softmax_with_radius(logits: &[f64], logit_radius: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let top = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let shifted: Vec<f64> = logits.iter().map(|&l| l - top).collect();
    let exponentials: Vec<f64> = shifted.iter().map(|&d| d.exp()).collect();
    let normalizer: f64 = exponentials.iter().sum();
    let exponential_relative: Vec<f64> = shifted
        .iter()
        .map(|&d| accumulation_growth(1) * d.abs() + f64::EPSILON)
        .collect();
    let normalizer_relative = exponential_relative.iter().copied().fold(0.0, f64::max)
        + accumulation_growth(logits.len().saturating_sub(1));
    let logit_factor = (2.0 * logit_radius.iter().copied().fold(0.0, f64::max)).exp_m1();
    let weights: Vec<f64> = exponentials.iter().map(|&e| e / normalizer).collect();
    let radius = weights
        .iter()
        .zip(&exponential_relative)
        .map(|(&w, &rel)| w * (logit_factor + rel + normalizer_relative + accumulation_growth(1)))
        .collect();
    (weights, radius)
}

/// The source's causal self-attention block on its original tensors.
#[derive(Clone, Debug)]
pub struct NativeAttention {
    geometry: AttentionGeometry,
    rotary: RotaryEmbedding,
    score_scale: f64,
    query: Array2<f64>,
    key: Array2<f64>,
    value: Array2<f64>,
    output: Array2<f64>,
}

fn expect_shape(
    tensor: &'static str,
    found: (usize, usize),
    expected: (usize, usize),
) -> Result<(), AttentionProgramError> {
    if found == expected {
        Ok(())
    } else {
        Err(AttentionProgramError::Shape {
            tensor,
            expected,
            found,
        })
    }
}

/// A pre-softmax score matrix per head with its radius; causally masked entries
/// are `−∞` with radius zero.
struct ScoredHeads {
    scores: Array3<f64>,
    radius: Array3<f64>,
}

/// One program's executed attention block, each quantity with its forward-error
/// radius against the program's exact value.
#[derive(Clone, Debug)]
pub struct AttentionExecution {
    /// `heads × tokens × tokens` scores; `s > t` is causally masked to `−∞`.
    pub scores: Array3<f64>,
    pub score_radius: Array3<f64>,
    pub weights: Array3<f64>,
    pub weight_radius: Array3<f64>,
    /// `tokens × model_dim`, after the source's output projection.
    pub output: Array2<f64>,
    pub output_radius: Array2<f64>,
}

/// Per-token head-space vectors: the computed values, the absolute sums of their
/// monomials, and the first-order effect of the trigonometric input error.
struct HeadVectors {
    value: Array2<f64>,
    abs: Array2<f64>,
    trig: Array2<f64>,
}

/// `W x_t` for every token, with the absolute sum of each inner product's terms.
fn project(weights: ArrayView2<f64>, x: ArrayView2<f64>) -> (Array2<f64>, Array2<f64>) {
    let mut value = Array2::zeros((x.nrows(), weights.nrows()));
    let mut abs = Array2::zeros((x.nrows(), weights.nrows()));
    for (t, token) in x.outer_iter().enumerate() {
        for (row, w) in weights.outer_iter().enumerate() {
            (value[[t, row]], abs[[t, row]]) = inner_with_abs(w, token);
        }
    }
    (value, abs)
}

impl NativeAttention {
    pub fn new(
        geometry: AttentionGeometry,
        rotary: RotaryEmbedding,
        score_scale: f64,
        query: Array2<f64>,
        key: Array2<f64>,
        value: Array2<f64>,
        output: Array2<f64>,
    ) -> Result<Self, AttentionProgramError> {
        if geometry.n_kv_heads == 0 || geometry.n_heads % geometry.n_kv_heads != 0 {
            return Err(AttentionProgramError::KeyValueHeadsDoNotDivide {
                n_heads: geometry.n_heads,
                n_kv_heads: geometry.n_kv_heads,
            });
        }
        if rotary.rotary_dim() > geometry.head_dim {
            return Err(AttentionProgramError::RotaryExceedsHead {
                rotary_dim: rotary.rotary_dim(),
                head_dim: geometry.head_dim,
            });
        }
        let (model, query_dim, kv_dim) = (
            geometry.model_dim,
            geometry.query_dim(),
            geometry.key_value_dim(),
        );
        expect_shape("query", query.dim(), (query_dim, model))?;
        expect_shape("key", key.dim(), (kv_dim, model))?;
        expect_shape("value", value.dim(), (kv_dim, model))?;
        expect_shape("output", output.dim(), (model, query_dim))?;
        Ok(Self {
            geometry,
            rotary,
            score_scale,
            query,
            key,
            value,
            output,
        })
    }

    fn check_input(&self, x: ArrayView2<f64>, positions: &[i64]) -> Result<(), AttentionProgramError> {
        expect_shape(
            "input",
            x.dim(),
            (positions.len(), self.geometry.model_dim),
        )?;
        match positions
            .iter()
            .find(|p| p.unsigned_abs() > EXACT_POSITION_LIMIT as u64)
        {
            Some(&position) => Err(AttentionProgramError::PositionNotExact { position }),
            None => Ok(()),
        }
    }

    /// The source block on its original tensors: rotate queries and keys at their
    /// absolute positions, score, mask causally, softmax jointly, read the values
    /// and project the output.
    pub fn execute(
        &self,
        x: ArrayView2<f64>,
        positions: &[i64],
    ) -> Result<AttentionExecution, AttentionProgramError> {
        self.check_input(x, positions)?;
        let queries = self.rotate(project(self.query.view(), x), positions, self.geometry.n_heads);
        let keys = self.rotate(project(self.key.view(), x), positions, self.geometry.n_kv_heads);
        // One side's rounded operations: the `model_dim`-term projection, then
        // `α·cos`, the product with the coordinate and the difference. The score
        // multiplies the two sides (1), sums `head_dim` products and scales by σ (1).
        let depth = 2 * (self.geometry.model_dim + 3) + self.geometry.head_dim + 1;
        let growth = accumulation_growth(depth);
        let (heads, tokens, hd) = (self.geometry.n_heads, x.nrows(), self.geometry.head_dim);
        let mut scores = Array3::from_elem((heads, tokens, tokens), f64::NEG_INFINITY);
        let mut radius = Array3::zeros((heads, tokens, tokens));
        for head in 0..heads {
            let (qo, ko) = (head * hd, self.geometry.key_value_head(head) * hd);
            for t in 0..tokens {
                for s in 0..=t {
                    let (mut value, mut abs, mut trig) = (0.0, 0.0, 0.0);
                    for c in 0..hd {
                        let (q, k) = ((t, qo + c), (s, ko + c));
                        value += queries.value[q] * keys.value[k];
                        abs += queries.abs[q] * keys.abs[k];
                        trig += queries.trig[q] * keys.abs[k] + queries.abs[q] * keys.trig[k];
                    }
                    scores[[head, t, s]] = self.score_scale * value;
                    radius[[head, t, s]] = self.score_scale.abs() * (growth * abs + trig);
                }
            }
        }
        Ok(self.attend(ScoredHeads { scores, radius }, x))
    }

    /// The source's rotation of every head of `projected` at each token's absolute
    /// position.
    fn rotate(
        &self,
        (mut value, mut abs): (Array2<f64>, Array2<f64>),
        positions: &[i64],
        heads: usize,
    ) -> HeadVectors {
        let alpha = self.rotary.attention_scaling;
        let mut trig = Array2::zeros(value.dim());
        for (t, &position) in positions.iter().enumerate() {
            for (plane, &frequency) in self.rotary.inverse_frequencies.iter().enumerate() {
                let (cos, sin, eta) = plane_trig(position as f64, frequency);
                let (cos, sin) = (alpha * cos, alpha * sin);
                let (a, b) = self.rotary.plane(plane);
                for head in 0..heads {
                    let (a, b) = ((t, head * self.geometry.head_dim + a), (t, head * self.geometry.head_dim + b));
                    let (xa, xb) = (value[a], value[b]);
                    let (abs_a, abs_b) = (abs[a], abs[b]);
                    value[a] = xa * cos - xb * sin;
                    value[b] = xb * cos + xa * sin;
                    abs[a] = abs_a * cos.abs() + abs_b * sin.abs();
                    abs[b] = abs_b * cos.abs() + abs_a * sin.abs();
                    let perturbation = alpha.abs() * eta * (abs_a + abs_b);
                    trig[a] = perturbation;
                    trig[b] = perturbation;
                }
            }
        }
        HeadVectors { value, abs, trig }
    }

    /// Causal mask, joint softmax per query row, the value read and the output
    /// projection, propagating the score radius.
    fn attend(&self, scored: ScoredHeads, x: ArrayView2<f64>) -> AttentionExecution {
        let g = self.geometry;
        let (tokens, hd) = (x.nrows(), g.head_dim);
        let (values, value_abs) = project(self.value.view(), x);
        let value_growth = accumulation_growth(g.model_dim);
        let mut weights = Array3::zeros((g.n_heads, tokens, tokens));
        let mut weight_radius = Array3::zeros((g.n_heads, tokens, tokens));
        let mut mixed = Array2::zeros((tokens, g.query_dim()));
        let mut mixed_radius = Array2::zeros((tokens, g.query_dim()));
        for head in 0..g.n_heads {
            let (qo, vo) = (head * hd, g.key_value_head(head) * hd);
            for t in 0..tokens {
                let logits: Vec<f64> = (0..=t).map(|s| scored.scores[[head, t, s]]).collect();
                let logit_radius: Vec<f64> = (0..=t).map(|s| scored.radius[[head, t, s]]).collect();
                let (row, row_radius) = joint_softmax_with_radius(&logits, &logit_radius);
                let mix_growth = accumulation_growth(t + 1);
                for s in 0..=t {
                    weights[[head, t, s]] = row[s];
                    weight_radius[[head, t, s]] = row_radius[s];
                }
                for c in 0..hd {
                    let (mut value, mut abs, mut propagated) = (0.0, 0.0, 0.0);
                    for s in 0..=t {
                        let v = values[[s, vo + c]];
                        value += row[s] * v;
                        abs += (row[s] * v).abs();
                        propagated +=
                            row_radius[s] * v.abs() + row[s] * value_growth * value_abs[[s, vo + c]];
                    }
                    mixed[[t, qo + c]] = value;
                    mixed_radius[[t, qo + c]] = propagated + mix_growth * abs;
                }
            }
        }
        let output_growth = accumulation_growth(g.query_dim());
        let mut output = Array2::zeros((tokens, g.model_dim));
        let mut output_radius = Array2::zeros((tokens, g.model_dim));
        for t in 0..tokens {
            for (d, w) in self.output.outer_iter().enumerate() {
                let (value, abs) = inner_with_abs(w, mixed.row(t));
                let propagated: f64 = w
                    .iter()
                    .zip(mixed_radius.row(t).iter())
                    .map(|(o, r)| o.abs() * r)
                    .sum();
                output[[t, d]] = value;
                output_radius[[t, d]] = propagated + output_growth * abs;
            }
        }
        AttentionExecution {
            scores: scored.scores,
            score_radius: scored.radius,
            weights,
            weight_radius,
            output,
            output_radius,
        }
    }
}

/// A projection factored through components, `W = outputs · readins`.
#[derive(Clone, Debug)]
pub struct ComponentProjection {
    outputs: Array2<f64>,
    readins: Array2<f64>,
}

impl ComponentProjection {
    pub fn new(outputs: Array2<f64>, readins: Array2<f64>) -> Result<Self, AttentionProgramError> {
        expect_shape(
            "component readins",
            readins.dim(),
            (outputs.ncols(), readins.ncols()),
        )?;
        Ok(Self { outputs, readins })
    }

    pub fn components(&self) -> usize {
        self.outputs.ncols()
    }
}

/// Component masks: one control per query component and one per key component.
#[derive(Clone, Debug, PartialEq)]
pub struct QueryKeyMasks {
    pub query: Array1<f64>,
    pub key: Array1<f64>,
}

/// The component query-key kernel `C_ij(Δ)` of one head at one displacement,
/// `Δ = p_key − p_query`.
#[derive(Clone, Debug)]
pub struct QueryKeyKernel {
    pub head: usize,
    pub displacement: i64,
    /// `query components × key components`.
    pub values: Array2<f64>,
    pub radius: Array2<f64>,
}

/// The source attention block with its query and key projections executed
/// through masked component coordinates.
#[derive(Clone, Debug)]
pub struct ComponentAttention {
    native: NativeAttention,
    query: ComponentProjection,
    key: ComponentProjection,
}

/// Masked component coordinates contracted into head space. The absolute sums
/// bound every monomial `|U_ci| |m_i| |R_ij x_j|`.
fn masked_head_vectors(
    projection: &ComponentProjection,
    mask: ArrayView1<f64>,
    x: ArrayView2<f64>,
) -> HeadVectors {
    let (coordinates, coordinate_abs) = project(projection.readins.view(), x);
    let masked = &coordinates * &mask;
    let masked_abs = &coordinate_abs * &mask.mapv(f64::abs);
    let value = project(projection.outputs.view(), masked.view()).0;
    let abs = project(projection.outputs.mapv(f64::abs).view(), masked_abs.view()).0;
    let trig = Array2::zeros(value.dim());
    HeadVectors { value, abs, trig }
}

impl ComponentAttention {
    /// `query` and `key` must factor the source's query and key projections
    /// exactly (as `rewrite`'s overcomplete factor does); the all-on path executes
    /// `native`'s own tensors.
    pub fn new(
        native: NativeAttention,
        query: ComponentProjection,
        key: ComponentProjection,
    ) -> Result<Self, AttentionProgramError> {
        let g = native.geometry;
        expect_shape(
            "query component outputs",
            query.outputs.dim(),
            (g.query_dim(), query.components()),
        )?;
        expect_shape(
            "query component readins",
            query.readins.dim(),
            (query.components(), g.model_dim),
        )?;
        expect_shape(
            "key component outputs",
            key.outputs.dim(),
            (g.key_value_dim(), key.components()),
        )?;
        expect_shape(
            "key component readins",
            key.readins.dim(),
            (key.components(), g.model_dim),
        )?;
        Ok(Self { native, query, key })
    }

    /// The block under component masks. All-on executes the original tensors;
    /// any other mask scores through the component kernel in head space, then the
    /// source's causal mask, joint softmax, values and output projection.
    pub fn execute(
        &self,
        masks: &QueryKeyMasks,
        x: ArrayView2<f64>,
        positions: &[i64],
    ) -> Result<AttentionExecution, AttentionProgramError> {
        expect_shape(
            "query mask",
            (masks.query.len(), 1),
            (self.query.components(), 1),
        )?;
        expect_shape("key mask", (masks.key.len(), 1), (self.key.components(), 1))?;
        if masks.query.iter().chain(masks.key.iter()).all(|&m| m == 1.0) {
            return self.native.execute(x, positions);
        }
        self.native.check_input(x, positions)?;
        let queries = masked_head_vectors(&self.query, masks.query.view(), x);
        let keys = masked_head_vectors(&self.key, masks.key.view(), x);
        let g = self.native.geometry;
        let rotary = &self.native.rotary;
        let alpha2 = rotary.attention_scaling * rotary.attention_scaling;
        // Per side: the `model_dim`-term readin, the mask product and the
        // component contraction. Per plane: the coordinate product, the in-plane
        // sum, the `cos`/`sin` product and the plane's sum (4), then at most
        // `head_dim` plane additions, `α·α` and its product (2), the pass-through
        // addition (1) and σ (1).
        let depth = (g.model_dim + 1 + self.query.components())
            + (g.model_dim + 1 + self.key.components())
            + 4
            + g.head_dim
            + 4;
        let growth = accumulation_growth(depth);
        let (heads, tokens, hd) = (g.n_heads, x.nrows(), g.head_dim);
        let rotated = rotary.rotary_dim();
        let mut trig_by_displacement: BTreeMap<i64, Vec<(f64, f64, f64)>> = BTreeMap::new();
        let mut scores = Array3::from_elem((heads, tokens, tokens), f64::NEG_INFINITY);
        let mut radius = Array3::zeros((heads, tokens, tokens));
        for head in 0..heads {
            let (qo, ko) = (head * hd, g.key_value_head(head) * hd);
            for t in 0..tokens {
                for s in 0..=t {
                    let displacement = positions[s] - positions[t];
                    let trig = trig_by_displacement.entry(displacement).or_insert_with(|| {
                        rotary
                            .inverse_frequencies
                            .iter()
                            .map(|&frequency| plane_trig(displacement as f64, frequency))
                            .collect()
                    });
                    let (mut planes, mut planes_abs, mut planes_trig) = (0.0, 0.0, 0.0);
                    for (plane, &(cos, sin, eta)) in trig.iter().enumerate() {
                        let (a, b) = rotary.plane(plane);
                        let (qa, qb) = (queries.value[(t, qo + a)], queries.value[(t, qo + b)]);
                        let (ka, kb) = (keys.value[(s, ko + a)], keys.value[(s, ko + b)]);
                        let (aqa, aqb) = (queries.abs[(t, qo + a)], queries.abs[(t, qo + b)]);
                        let (aka, akb) = (keys.abs[(s, ko + a)], keys.abs[(s, ko + b)]);
                        planes += cos * (qa * ka + qb * kb) + sin * (qb * ka - qa * kb);
                        planes_abs += cos.abs() * (aqa * aka + aqb * akb) + sin.abs() * (aqb * aka + aqa * akb);
                        planes_trig += eta * (aqa + aqb) * (aka + akb);
                    }
                    let (mut pass, mut pass_abs) = (0.0, 0.0);
                    for c in rotated..hd {
                        pass += queries.value[(t, qo + c)] * keys.value[(s, ko + c)];
                        pass_abs += queries.abs[(t, qo + c)] * keys.abs[(s, ko + c)];
                    }
                    scores[[head, t, s]] = self.native.score_scale * (alpha2 * planes + pass);
                    radius[[head, t, s]] = self.native.score_scale.abs()
                        * (growth * (alpha2.abs() * planes_abs + pass_abs) + alpha2.abs() * planes_trig);
                }
            }
        }
        Ok(self.native.attend(ScoredHeads { scores, radius }, x))
    }

    /// `C_ij(Δ) = σ u^Q_{h,i}ᵀ R_Δ u^K_{g(h),j}`, evaluated plane by plane:
    /// `α² Σ_p [cos(Δθ_p)(u_a w_a + u_b w_b) + sin(Δθ_p)(u_b w_a − u_a w_b)]` plus
    /// the unrotated coordinates.
    pub fn kernel(&self, head: usize, displacement: i64) -> Result<QueryKeyKernel, AttentionProgramError> {
        let g = self.native.geometry;
        if head >= g.n_heads {
            return Err(AttentionProgramError::HeadOutOfRange {
                head,
                n_heads: g.n_heads,
            });
        }
        if displacement.unsigned_abs() > 2 * EXACT_POSITION_LIMIT as u64 {
            return Err(AttentionProgramError::DisplacementNotExact { displacement });
        }
        let rotary = &self.native.rotary;
        let alpha2 = rotary.attention_scaling * rotary.attention_scaling;
        let (hd, rotated) = (g.head_dim, rotary.rotary_dim());
        let (qo, ko) = (head * hd, g.key_value_head(head) * hd);
        let trig: Vec<(f64, f64, f64)> = rotary
            .inverse_frequencies
            .iter()
            .map(|&frequency| plane_trig(displacement as f64, frequency))
            .collect();
        // The coordinate product, the in-plane sum, the `cos`/`sin` product and
        // the plane's sum (4), at most `head_dim` plane additions, `α·α` and its
        // product (2), the pass-through addition (1) and σ (1).
        let growth = accumulation_growth(hd + 8);
        let (rows, cols) = (self.query.components(), self.key.components());
        let mut values = Array2::zeros((rows, cols));
        let mut radius = Array2::zeros((rows, cols));
        for i in 0..rows {
            let u = self.query.outputs.column(i);
            for j in 0..cols {
                let w = self.key.outputs.column(j);
                let (mut planes, mut planes_abs, mut planes_trig) = (0.0, 0.0, 0.0);
                for (plane, &(cos, sin, eta)) in trig.iter().enumerate() {
                    let (a, b) = rotary.plane(plane);
                    let (ua, ub, wa, wb) = (u[qo + a], u[qo + b], w[ko + a], w[ko + b]);
                    planes += cos * (ua * wa + ub * wb) + sin * (ub * wa - ua * wb);
                    planes_abs += cos.abs() * (ua * wa).abs()
                        + cos.abs() * (ub * wb).abs()
                        + sin.abs() * ((ub * wa).abs() + (ua * wb).abs());
                    planes_trig += eta * (ua.abs() + ub.abs()) * (wa.abs() + wb.abs());
                }
                let (mut pass, mut pass_abs) = (0.0, 0.0);
                for c in rotated..hd {
                    pass += u[qo + c] * w[ko + c];
                    pass_abs += (u[qo + c] * w[ko + c]).abs();
                }
                values[[i, j]] = self.native.score_scale * (alpha2 * planes + pass);
                radius[[i, j]] = self.native.score_scale.abs()
                    * (growth * (alpha2.abs() * planes_abs + pass_abs) + alpha2.abs() * planes_trig);
            }
        }
        Ok(QueryKeyKernel {
            head,
            displacement,
            values,
            radius,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    /// Small dyadic rationals. Every product and sum the fixture forms stays exact
    /// in f64, so the edited tensor `U diag(m) R` and both routes' head-space
    /// vectors carry no rounding; the routes differ only by the trigonometric and
    /// scoring arithmetic their radii bound.
    fn dyadic(rows: usize, cols: usize, salt: usize, denominator: f64) -> Array2<f64> {
        Array2::from_shape_fn((rows, cols), |(i, j)| {
            ((i * 7 + j * 3 + salt * 5 + i * j) % 9) as f64 / denominator - 4.0 / denominator
        })
    }

    fn masked_product(outputs: &Array2<f64>, mask: &Array1<f64>, readins: &Array2<f64>) -> Array2<f64> {
        Array2::from_shape_fn((outputs.nrows(), readins.ncols()), |(c, j)| {
            (0..mask.len())
                .map(|i| outputs[[c, i]] * mask[i] * readins[[i, j]])
                .sum()
        })
    }

    fn coordinates(readins: &Array2<f64>, mask: &Array1<f64>, token: ArrayView1<f64>) -> Array1<f64> {
        readins.dot(&token) * mask
    }

    struct Fixture {
        geometry: AttentionGeometry,
        rotary: RotaryEmbedding,
        score_scale: f64,
        query_outputs: Array2<f64>,
        query_readins: Array2<f64>,
        key_outputs: Array2<f64>,
        key_readins: Array2<f64>,
        value: Array2<f64>,
        output: Array2<f64>,
        x: Array2<f64>,
        positions: Vec<i64>,
    }

    const QUERY_COMPONENTS: usize = 5;
    const KEY_COMPONENTS: usize = 4;

    impl Fixture {
        /// Two query heads sharing one key/value head, two rotated planes and two
        /// pass-through coordinates per head, positions offset and with gaps.
        fn new(pairing: RotaryPairing) -> Self {
            let geometry = AttentionGeometry {
                model_dim: 4,
                n_heads: 2,
                n_kv_heads: 1,
                head_dim: 6,
            };
            Self {
                geometry,
                rotary: RotaryEmbedding {
                    pairing,
                    inverse_frequencies: vec![1.0, 0.25],
                    attention_scaling: 1.25,
                },
                score_scale: 1.0 / (geometry.head_dim as f64).sqrt(),
                query_outputs: dyadic(12, QUERY_COMPONENTS, 1, 8.0),
                query_readins: dyadic(QUERY_COMPONENTS, 4, 2, 8.0),
                key_outputs: dyadic(6, KEY_COMPONENTS, 3, 8.0),
                key_readins: dyadic(KEY_COMPONENTS, 4, 4, 8.0),
                value: dyadic(6, 4, 5, 8.0),
                output: dyadic(4, 12, 6, 8.0),
                x: dyadic(5, 4, 7, 4.0),
                positions: vec![2, 3, 5, 6, 9],
            }
        }

        fn native_with(&self, query: Array2<f64>, key: Array2<f64>) -> NativeAttention {
            NativeAttention::new(
                self.geometry,
                self.rotary.clone(),
                self.score_scale,
                query,
                key,
                self.value.clone(),
                self.output.clone(),
            )
            .expect("fixture tensors match the geometry")
        }

        /// Direct execution with the edited tensors `U diag(m) R`.
        fn edited(&self, masks: &QueryKeyMasks) -> NativeAttention {
            self.native_with(
                masked_product(&self.query_outputs, &masks.query, &self.query_readins),
                masked_product(&self.key_outputs, &masks.key, &self.key_readins),
            )
        }

        fn program(&self) -> ComponentAttention {
            ComponentAttention::new(
                self.edited(&all_on()),
                ComponentProjection::new(self.query_outputs.clone(), self.query_readins.clone())
                    .expect("query factor shapes agree"),
                ComponentProjection::new(self.key_outputs.clone(), self.key_readins.clone())
                    .expect("key factor shapes agree"),
            )
            .expect("factors match the geometry")
        }
    }

    fn all_on() -> QueryKeyMasks {
        QueryKeyMasks {
            query: Array1::ones(QUERY_COMPONENTS),
            key: Array1::ones(KEY_COMPONENTS),
        }
    }

    fn continuous_masks() -> QueryKeyMasks {
        QueryKeyMasks {
            query: array![0.375, 0.75, 0.5, 0.875, 0.25],
            key: array![0.625, 0.125, 1.0, 0.5],
        }
    }

    /// The largest `|a − b| / (radius_a + radius_b)` over causally admissible
    /// scores and weights and over the outputs.
    fn worst_ratio(a: &AttentionExecution, b: &AttentionExecution) -> f64 {
        let ratio = |x: f64, y: f64, rx: f64, ry: f64| (x - y).abs() / (rx + ry);
        let mut worst = 0.0_f64;
        for ((h, t, s), &score) in a.scores.indexed_iter() {
            if s <= t {
                let i = [h, t, s];
                worst = worst.max(ratio(score, b.scores[i], a.score_radius[i], b.score_radius[i]));
                worst = worst.max(ratio(a.weights[i], b.weights[i], a.weight_radius[i], b.weight_radius[i]));
            }
        }
        for ((t, d), &value) in a.output.indexed_iter() {
            let i = [t, d];
            worst = worst.max(ratio(value, b.output[i], a.output_radius[i], b.output_radius[i]));
        }
        worst
    }

    /// A11, A1: the component query-key program equals direct execution with the
    /// edited tensors for continuous, binary and signed masks, within the summed
    /// derived radii of the two routes; moving one key mask by 1/8 leaves them.
    #[test]
    fn component_program_equals_edited_tensor_execution_within_derived_radii() {
        let fixture = Fixture::new(RotaryPairing::HalfSplit);
        let program = fixture.program();
        let families = [
            ("continuous", continuous_masks()),
            (
                "binary",
                QueryKeyMasks {
                    query: array![1.0, 0.0, 1.0, 1.0, 0.0],
                    key: array![0.0, 1.0, 1.0, 1.0],
                },
            ),
            (
                "signed",
                QueryKeyMasks {
                    query: array![-0.5, 1.0, 0.25, -1.0, 0.75],
                    key: array![1.0, -0.375, 0.5, -0.25],
                },
            ),
        ];
        for (label, masks) in families {
            let component = program
                .execute(&masks, fixture.x.view(), &fixture.positions)
                .expect("component execution");
            let edited = fixture
                .edited(&masks)
                .execute(fixture.x.view(), &fixture.positions)
                .expect("edited-tensor execution");
            let agreement = worst_ratio(&component, &edited);
            assert!(
                agreement <= 1.0,
                "{label} masks: the component program leaves edited-tensor execution by {agreement} of the summed radii"
            );
            let mut moved = masks.clone();
            moved.key[0] += 0.125;
            let control = fixture
                .edited(&moved)
                .execute(fixture.x.view(), &fixture.positions)
                .expect("moved-mask execution");
            let separation = worst_ratio(&component, &control);
            assert!(
                separation > 1.0,
                "{label} masks: a key mask moved by 1/8 must leave the radii, got {separation}"
            );
        }
    }

    /// A1: all-on executes the original tensors bit for bit; a query entry moved by
    /// 2^-40 changes the bits, so the comparison sees a change of that size.
    #[test]
    fn all_on_masks_execute_the_original_tensors_bit_for_bit() {
        let fixture = Fixture::new(RotaryPairing::HalfSplit);
        let bits = |e: &AttentionExecution| -> Vec<u64> {
            e.scores
                .iter()
                .chain(e.weights.iter())
                .chain(e.output.iter())
                .map(|v| v.to_bits())
                .collect()
        };
        let executed = fixture
            .program()
            .execute(&all_on(), fixture.x.view(), &fixture.positions)
            .expect("all-on execution");
        let original = fixture
            .edited(&all_on())
            .execute(fixture.x.view(), &fixture.positions)
            .expect("original execution");
        assert_eq!(bits(&executed), bits(&original), "all-on must run the original tensors");
        let mut nudged_query = masked_product(&fixture.query_outputs, &all_on().query, &fixture.query_readins);
        nudged_query[[0, 1]] += 2.0_f64.powi(-40);
        let nudged = fixture
            .native_with(
                nudged_query,
                masked_product(&fixture.key_outputs, &all_on().key, &fixture.key_readins),
            )
            .execute(fixture.x.view(), &fixture.positions)
            .expect("nudged execution");
        assert_ne!(bits(&nudged), bits(&original), "a moved query entry must change the bits");
    }

    /// P6: `a_tᵀ C(p_s − p_t) b_s` reproduces the executed score; the kernel read
    /// at `p_t − p_s` does not.
    #[test]
    fn kernel_scores_read_key_minus_query_displacement() {
        let fixture = Fixture::new(RotaryPairing::HalfSplit);
        let program = fixture.program();
        let masks = continuous_masks();
        let executed = program
            .execute(&masks, fixture.x.view(), &fixture.positions)
            .expect("component execution");
        // Two products per term and `QUERY_COMPONENTS · KEY_COMPONENTS` additions.
        let growth = accumulation_growth(QUERY_COMPONENTS * KEY_COMPONENTS + 2);
        let (mut forward_worst, mut flipped_worst) = (0.0_f64, 0.0_f64);
        for head in 0..fixture.geometry.n_heads {
            for t in 0..fixture.positions.len() {
                let a = coordinates(&fixture.query_readins, &masks.query, fixture.x.row(t));
                for s in 0..=t {
                    let b = coordinates(&fixture.key_readins, &masks.key, fixture.x.row(s));
                    let read = |kernel: &QueryKeyKernel| -> (f64, f64) {
                        let (mut value, mut abs, mut propagated) = (0.0, 0.0, 0.0);
                        for i in 0..QUERY_COMPONENTS {
                            for j in 0..KEY_COMPONENTS {
                                value += a[i] * kernel.values[[i, j]] * b[j];
                                abs += (a[i] * kernel.values[[i, j]] * b[j]).abs();
                                propagated += (a[i] * b[j]).abs() * kernel.radius[[i, j]];
                            }
                        }
                        (value, propagated + growth * abs)
                    };
                    let displacement = fixture.positions[s] - fixture.positions[t];
                    let (value, radius) = read(&program.kernel(head, displacement).expect("kernel"));
                    let (flipped, flipped_radius) =
                        read(&program.kernel(head, -displacement).expect("flipped kernel"));
                    let (score, score_radius) = (executed.scores[[head, t, s]], executed.score_radius[[head, t, s]]);
                    forward_worst = forward_worst.max((value - score).abs() / (radius + score_radius));
                    flipped_worst = flipped_worst.max((flipped - score).abs() / (flipped_radius + score_radius));
                }
            }
        }
        assert!(
            forward_worst <= 1.0,
            "the kernel at p_key − p_query leaves the executed scores by {forward_worst} of the radii"
        );
        assert!(
            flipped_worst > 1.0,
            "the kernel at p_query − p_key must leave the executed scores, got {flipped_worst}"
        );
    }

    /// The source's pairing and sign on a hand-computed score: the query token at
    /// position 1 reads `e_0`, the key token at position 0 reads `e_1`. gptj's
    /// `rotate_every_two` rotates plane (0, 1), so `e_0 ↦ α(cos θ, sin θ)` and the
    /// score is `+σα² sin θ`; llama's `rotate_half` puts `e_0` and `e_1` in the
    /// different planes (0, 2) and (1, 3), so the score is zero.
    #[test]
    fn rotary_pairing_and_sign_follow_the_source() {
        let geometry = AttentionGeometry {
            model_dim: 4,
            n_heads: 1,
            n_kv_heads: 1,
            head_dim: 4,
        };
        let (frequency, score_scale, attention_scaling) = (0.75_f64, 0.5, 1.25);
        let x = array![[0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]];
        let positions = [0_i64, 1];
        let score = |pairing: RotaryPairing| -> (f64, f64) {
            let executed = NativeAttention::new(
                geometry,
                RotaryEmbedding {
                    pairing,
                    inverse_frequencies: vec![frequency, frequency],
                    attention_scaling,
                },
                score_scale,
                Array2::eye(4),
                Array2::eye(4),
                Array2::eye(4),
                Array2::eye(4),
            )
            .expect("identity block")
            .execute(x.view(), &positions)
            .expect("execution");
            (executed.scores[[0, 1, 0]], executed.score_radius[[0, 1, 0]])
        };
        let expected = score_scale * attention_scaling * attention_scaling * frequency.sin();
        // Three products and one libm ulp in the reference itself.
        let expected_error = (accumulation_growth(3) + f64::EPSILON) * expected.abs();
        let (interleaved, interleaved_radius) = score(RotaryPairing::Interleaved);
        let (half_split, half_split_radius) = score(RotaryPairing::HalfSplit);
        let tolerance = interleaved_radius + expected_error;
        assert!(
            (interleaved - expected).abs() <= tolerance,
            "interleaved score {interleaved}, expected {expected} within {tolerance}"
        );
        assert!(
            (interleaved + expected).abs() > tolerance,
            "the opposite rotation sign must be distinguishable from the source's"
        );
        assert!(
            half_split.abs() <= half_split_radius,
            "half-split score {half_split} must be zero within {half_split_radius}"
        );
        assert!(
            (half_split - expected).abs() > half_split_radius + expected_error,
            "the two pairings must be distinguishable on this fixture"
        );
    }

    /// A11 negative control: the mass-preserving mixture of separately normalized
    /// per-query-component patterns is refuted against the executed weights, while
    /// the joint softmax of the same summed component logits is not.
    #[test]
    fn separately_normalized_component_patterns_are_refuted() {
        let fixture = Fixture::new(RotaryPairing::HalfSplit);
        let program = fixture.program();
        let masks = continuous_masks();
        let executed = program
            .execute(&masks, fixture.x.view(), &fixture.positions)
            .expect("component execution");
        let components = QUERY_COMPONENTS as f64;
        let (mut joint_worst, mut separate_worst) = (0.0_f64, 0.0_f64);
        for head in 0..fixture.geometry.n_heads {
            for t in 1..fixture.positions.len() {
                let a = coordinates(&fixture.query_readins, &masks.query, fixture.x.row(t));
                let mut logits = vec![vec![0.0; t + 1]; QUERY_COMPONENTS];
                let mut logit_radius = vec![vec![0.0; t + 1]; QUERY_COMPONENTS];
                for s in 0..=t {
                    let b = coordinates(&fixture.key_readins, &masks.key, fixture.x.row(s));
                    let kernel = program
                        .kernel(head, fixture.positions[s] - fixture.positions[t])
                        .expect("kernel");
                    for i in 0..QUERY_COMPONENTS {
                        let (mut value, mut abs, mut propagated) = (0.0, 0.0, 0.0);
                        for j in 0..KEY_COMPONENTS {
                            value += a[i] * kernel.values[[i, j]] * b[j];
                            abs += (a[i] * kernel.values[[i, j]] * b[j]).abs();
                            propagated += (a[i] * b[j]).abs() * kernel.radius[[i, j]];
                        }
                        logits[i][s] = value;
                        logit_radius[i][s] = propagated + accumulation_growth(KEY_COMPONENTS + 2) * abs;
                    }
                }
                let summed: Vec<f64> = (0..=t)
                    .map(|s| (0..QUERY_COMPONENTS).map(|i| logits[i][s]).sum())
                    .collect();
                let summed_radius: Vec<f64> = (0..=t)
                    .map(|s| {
                        (0..QUERY_COMPONENTS).map(|i| logit_radius[i][s]).sum::<f64>()
                            + accumulation_growth(QUERY_COMPONENTS)
                                * (0..QUERY_COMPONENTS).map(|i| logits[i][s].abs()).sum::<f64>()
                    })
                    .collect();
                let (joint, joint_radius) = joint_softmax_with_radius(&summed, &summed_radius);
                let mut mixture = vec![0.0; t + 1];
                let mut mixture_radius = vec![0.0; t + 1];
                for i in 0..QUERY_COMPONENTS {
                    let (weights, radius) = joint_softmax_with_radius(&logits[i], &logit_radius[i]);
                    for s in 0..=t {
                        mixture[s] += weights[s] / components;
                        mixture_radius[s] +=
                            radius[s] / components + accumulation_growth(QUERY_COMPONENTS + 1) * weights[s] / components;
                    }
                }
                for s in 0..=t {
                    let (weight, radius) = (executed.weights[[head, t, s]], executed.weight_radius[[head, t, s]]);
                    joint_worst = joint_worst.max((joint[s] - weight).abs() / (joint_radius[s] + radius));
                    separate_worst = separate_worst.max((mixture[s] - weight).abs() / (mixture_radius[s] + radius));
                }
            }
        }
        assert!(
            joint_worst <= 1.0,
            "the joint softmax of the summed component logits leaves the executed weights by {joint_worst}"
        );
        assert!(
            separate_worst > 1.0,
            "separately normalized component patterns must be refuted, got {separate_worst}"
        );
    }
}
