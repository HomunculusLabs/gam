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
//! The source's projection biases are executed unmasked. `q_t = U^Q (m^Q ⊙ R^Q x_t) + b_Q`
//! equals edited-tensor execution with `W_Q(m)` and the source `b_Q`, and a
//! bias-free source carries zero biases. The score then gains the bias terms
//! `b_Qᵀ R_Δ k`, `qᵀ R_Δ b_K` and `b_Qᵀ R_Δ b_K`, while [`ComponentAttention::kernel`]
//! covers the component coordinates only. Masking bias reads through components
//! removes part of a bias; that is a different experiment, and this program does
//! not execute it.
//!
//! A per-head norm between the projection and the rotary embedding (Qwen3's
//! `q_norm`/`k_norm`, [`NativeAttention::with_query_key_norm`]) is a native
//! nonlinearity. It runs on each token's summed head rows, never per component,
//! through gam-sae's gated-rewrite owner. Because `w ⊙ (ν q) = ν (w ⊙ q)`, the gains
//! fold into the component outputs, and the score is
//! `ν^Q_{t,h} ν^K_{s,g(h)} a_tᵀ C^w(Δ) b_s`, where [`ComponentAttention::kernel`]
//! returns `C^w` and each `ν` is evaluated natively on the current input.
//!
//! # All-on and roundoff
//!
//! With every mask at one the program executes the original tensors on the
//! original path ([`NativeAttention::execute`]), bit for bit. Every other result
//! carries a forward-error radius against the exact value of its own program at
//! the given inputs, propagated entry by entry to first order. It collects:
//! - Higham's `γ_k · Σ|monomials|` over the rounded operations (`gam_linalg::roundoff`);
//! - each input's own radius;
//! - the trigonometric input error `u·|φ| + ulp` of each angle;
//! - the log-softmax evaluation radius from `gam_math::categorical`.
//!
//! Two programs with the same exact value agree within the sum of their radii.
//!
//! The libm terms rest on the platform's documented accuracy. glibc x86_64's "Known
//! Maximum Errors" puts `sin`, `cos` and `exp` within one ulp, which covers the MSI
//! build targets. The tests measure that bound against a double-double reference at
//! every point the fixtures evaluate. That checks the platform the tests run on; it
//! does not certify production inputs.

use std::collections::BTreeMap;
use std::fmt;

use gam_linalg::roundoff::accumulation_growth;
use gam_math::categorical::{CategoricalError, log_softmax_with_error};
use ndarray::{Array1, Array2, Array3, ArrayView1, ArrayView2, ShapeBuilder};
use serde::{Deserialize, Serialize};

use crate::parameter_decomposition::gated_rewrite::{GatedRewriteError, MaskedNorm, rms_normalizers};

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
    /// A score row that names no categorical distribution.
    Categorical(CategoricalError),
    /// The source's per-head query/key norm refused its rows.
    Norm(GatedRewriteError),
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
            Self::Categorical(error) => write!(f, "attention score row: {error}"),
            Self::Norm(error) => write!(f, "attention query/key norm: {error}"),
        }
    }
}

impl std::error::Error for AttentionProgramError {}

/// The source attention block's dimensions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum RotaryPairing {
    HalfSplit,
    Interleaved,
}

/// The source's rotary embedding, as exported from the source module.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
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

/// The source's joint softmax over one causally admissible query row, with a
/// radius on each weight.
///
/// The log weights and their evaluation radii `e_s` come from gam-math's
/// categorical owner, [`log_softmax_with_error`]. A logit within `r_s` of exact
/// moves `ℓ_s` by `r_s` and `lse ℓ` by at most `max_t r_t`, so each log weight is
/// within `b_s = r_s + max_t r_t + e_s` of exact. The weight `exp(log p_s)` adds
/// one libm ulp, so its radius is `w_s (expm1(b_s) + ε)`.
fn attention_weights(logits: &[f64], logit_radius: &[f64]) -> Result<(Vec<f64>, Vec<f64>), CategoricalError> {
    let (log_weights, evaluation_radius) = log_softmax_with_error(logits)?;
    let widest = logit_radius.iter().copied().fold(0.0, f64::max);
    let mut weights = Vec::with_capacity(logits.len());
    let mut radius = Vec::with_capacity(logits.len());
    for ((&log_weight, &evaluation), &own) in log_weights.iter().zip(&evaluation_radius).zip(logit_radius) {
        let weight = log_weight.exp();
        weights.push(weight);
        radius.push(weight * ((own + widest + evaluation).exp_m1() + f64::EPSILON));
    }
    Ok((weights, radius))
}

/// A source projection `W x + b`, as the source module stores it. A bias-free
/// source carries a zero bias, and adding `0.0` is exact.
#[derive(Clone, Debug)]
pub struct AffineProjection {
    pub weight: Array2<f64>,
    pub bias: Array1<f64>,
}

/// The source's causal rotary attention without its projections: the geometry,
/// the rotary embedding and the score multiplier, validated once at construction.
/// It holds no tensor, so a program node can carry it by serializing its three
/// inputs.
#[derive(Clone, Debug)]
pub struct RotaryCausalAttention {
    geometry: AttentionGeometry,
    rotary: RotaryEmbedding,
    score_scale: f64,
}

/// The source's per-head query/key RMS norm: its declared epsilon and the
/// `head_dim` gains of `q_norm` and `k_norm`.
#[derive(Clone, Debug)]
struct QueryKeyNorm {
    epsilon: f64,
    query_gain: Array1<f64>,
    key_gain: Array1<f64>,
}

/// The source's causal self-attention block on its original tensors.
#[derive(Clone, Debug)]
pub struct NativeAttention {
    attention: RotaryCausalAttention,
    query: AffineProjection,
    key: AffineProjection,
    value: AffineProjection,
    output: AffineProjection,
    query_key_norm: Option<QueryKeyNorm>,
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

/// Already-projected rows, `tokens × width`, with a per-entry forward-error
/// radius against their exact values.
#[derive(Clone, Copy, Debug)]
pub struct ProjectedRows<'a> {
    pub values: ArrayView2<'a, f64>,
    pub radius: ArrayView2<'a, f64>,
}

/// The one zero that exact rows view as their radius.
static EXACT_RADIUS: [f64; 1] = [0.0];

impl<'a> ProjectedRows<'a> {
    /// Rows taken as exact. The radius is a zero-stride view of one zero, so no
    /// `tokens × width` matrix is allocated.
    pub fn exact(values: ArrayView2<'a, f64>) -> Self {
        let radius = ArrayView2::from_shape(values.dim().strides((0, 0)), &EXACT_RADIUS[..])
            .expect("a zero-stride view reads only its one entry");
        Self { values, radius }
    }
}

/// The attention block before the output projection, each quantity with its
/// forward-error radius.
#[derive(Clone, Debug)]
pub struct ProjectedAttention {
    /// `heads × tokens × tokens` scores; `s > t` is causally masked to `−∞`.
    pub scores: Array3<f64>,
    pub score_radius: Array3<f64>,
    pub weights: Array3<f64>,
    pub weight_radius: Array3<f64>,
    /// `tokens × n_heads·head_dim`: each head's weighted value read, before the
    /// output projection.
    pub mixed: Array2<f64>,
    pub mixed_radius: Array2<f64>,
}

/// Per-token head-space rows with a per-entry forward-error radius.
struct HeadRows {
    value: Array2<f64>,
    radius: Array2<f64>,
}

/// `W x_t + b` for every token. The `model_dim`-term inner product and the bias
/// addition give the radius `γ_{d+1}` times the absolute sum of the terms.
fn project_affine(projection: &AffineProjection, x: ArrayView2<f64>) -> (Array2<f64>, Array2<f64>) {
    let growth = accumulation_growth(x.ncols() + 1);
    let mut value = Array2::zeros((x.nrows(), projection.weight.nrows()));
    let mut radius = Array2::zeros((x.nrows(), projection.weight.nrows()));
    for (t, token) in x.outer_iter().enumerate() {
        for (row, w) in projection.weight.outer_iter().enumerate() {
            let (inner, abs) = inner_with_abs(w, token);
            let bias = projection.bias[row];
            value[[t, row]] = inner + bias;
            radius[[t, row]] = growth * (abs + bias.abs());
        }
    }
    (value, radius)
}

/// The source's per-head RMS norm (Qwen3 `q_norm`, `k_norm`) on every head of
/// `tokens × heads·head_dim` rows, evaluated by the gated-rewrite owner on the
/// current summed rows.
///
/// For one head row `x` with radius `r`, `y = w ⊙ x ν` with
/// `ν = (mean x² + ε)^{-1/2}`. Since `∂ν/∂x_d = −ν³ x_d / d`, to first order
/// `|δy_c| ≤ |w_c| ν (r_c + ν² |x_c| Σ_d |x_d| r_d / d)`. Evaluating `ν` from the
/// stored row costs `γ_{d+4}` relative (the owner's bound) and the two products
/// `γ_2`, so the radius adds `γ_{d+6} |y_c|`.
fn normalize_heads(
    (values, radius): (Array2<f64>, Array2<f64>),
    heads: usize,
    head_dim: usize,
    epsilon: f64,
    gain: ArrayView1<'_, f64>,
) -> Result<(Array2<f64>, Array2<f64>), GatedRewriteError> {
    let (tokens, width) = values.dim();
    let mismatch = GatedRewriteError::ShapeMismatch {
        what: "per-head query/key rows",
        expected: (tokens * heads, head_dim),
        found: (tokens, width),
    };
    let per_head = values
        .into_shape_with_order((tokens * heads, head_dim))
        .ok()
        .ok_or(mismatch.clone())?;
    let per_head_radius = radius
        .into_shape_with_order((tokens * heads, head_dim))
        .ok()
        .ok_or(mismatch.clone())?;
    let normalized = MaskedNorm::Rms { epsilon, gain }.apply(per_head.view())?;
    let normalizers = rms_normalizers(per_head.view(), epsilon)?;
    let evaluation = accumulation_growth(head_dim + 6);
    let mut normalized_radius = Array2::zeros(normalized.dim());
    for (row, &nu) in normalizers.iter().enumerate() {
        let weighted = per_head
            .row(row)
            .iter()
            .zip(per_head_radius.row(row).iter())
            .map(|(x, r)| x.abs() * r)
            .sum::<f64>()
            / head_dim as f64;
        for c in 0..head_dim {
            normalized_radius[[row, c]] = gain[c].abs()
                * nu
                * (per_head_radius[[row, c]] + nu * nu * per_head[[row, c]].abs() * weighted)
                + evaluation * normalized[[row, c]].abs();
        }
    }
    let values = normalized
        .into_shape_with_order((tokens, width))
        .ok()
        .ok_or(mismatch.clone())?;
    let radius = normalized_radius
        .into_shape_with_order((tokens, width))
        .ok()
        .ok_or(mismatch)?;
    Ok((values, radius))
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

impl RotaryCausalAttention {
    /// Query heads share key/value heads in contiguous groups, and the rotated
    /// planes must fit in one head. Only these checks run; nothing is allocated.
    pub fn new(
        geometry: AttentionGeometry,
        rotary: RotaryEmbedding,
        score_scale: f64,
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
        Ok(Self {
            geometry,
            rotary,
            score_scale,
        })
    }

    fn check_positions(&self, positions: &[i64]) -> Result<(), AttentionProgramError> {
        match positions
            .iter()
            .find(|p| p.unsigned_abs() > EXACT_POSITION_LIMIT as u64)
        {
            Some(&position) => Err(AttentionProgramError::PositionNotExact { position }),
            None => Ok(()),
        }
    }

    /// The source's attention on already-projected rows: rotate queries and keys
    /// at their absolute positions, score, mask causally, softmax jointly and read
    /// the values. Queries are `tokens × n_heads·head_dim`; keys and values are
    /// `tokens × n_kv_heads·head_dim`. Each input radius enters to first order.
    pub fn attend_projected(
        &self,
        queries: ProjectedRows<'_>,
        keys: ProjectedRows<'_>,
        values: ProjectedRows<'_>,
        positions: &[i64],
    ) -> Result<ProjectedAttention, AttentionProgramError> {
        self.check_positions(positions)?;
        let g = self.geometry;
        let (heads, tokens, hd) = (g.n_heads, positions.len(), g.head_dim);
        // Dimensions only: an array of `&ProjectedRows` would tie the three
        // arguments' lifetimes together, and `ArrayView` is invariant in its lifetime.
        for (name, values_dim, radius_dim, width) in [
            ("query rows", queries.values.dim(), queries.radius.dim(), g.query_dim()),
            ("key rows", keys.values.dim(), keys.radius.dim(), g.key_value_dim()),
            ("value rows", values.values.dim(), values.radius.dim(), g.key_value_dim()),
        ] {
            expect_shape(name, values_dim, (tokens, width))?;
            expect_shape(name, radius_dim, (tokens, width))?;
        }
        let queries = self.rotate(queries, positions, heads);
        let keys = self.rotate(keys, positions, g.n_kv_heads);
        // The `head_dim` products and their additions, then σ.
        let growth = accumulation_growth(hd + 1);
        let mut scores = Array3::from_elem((heads, tokens, tokens), f64::NEG_INFINITY);
        let mut radius = Array3::zeros((heads, tokens, tokens));
        for head in 0..heads {
            let (qo, ko) = (head * hd, g.key_value_head(head) * hd);
            for t in 0..tokens {
                for s in 0..=t {
                    let (mut value, mut abs, mut propagated) = (0.0, 0.0, 0.0);
                    for c in 0..hd {
                        let (q, k) = ((t, qo + c), (s, ko + c));
                        value += queries.value[q] * keys.value[k];
                        abs += (queries.value[q] * keys.value[k]).abs();
                        propagated +=
                            queries.radius[q] * keys.value[k].abs() + queries.value[q].abs() * keys.radius[k];
                    }
                    scores[[head, t, s]] = self.score_scale * value;
                    radius[[head, t, s]] = self.score_scale.abs() * (propagated + growth * abs);
                }
            }
        }
        self.mix(ScoredHeads { scores, radius }, values)
    }

    /// The source's rotation of every head of `rows` at each token's absolute
    /// position. The radius carries the rows' own radius, the trigonometric input
    /// error and the rotation's rounding (`α·cos`, the product and the in-plane sum).
    fn rotate(&self, rows: ProjectedRows<'_>, positions: &[i64], heads: usize) -> HeadRows {
        let alpha = self.rotary.attention_scaling;
        let growth = accumulation_growth(3);
        let mut value = rows.values.to_owned();
        let mut radius = rows.radius.to_owned();
        for (t, &position) in positions.iter().enumerate() {
            for (plane, &frequency) in self.rotary.inverse_frequencies.iter().enumerate() {
                let (cos, sin, eta) = plane_trig(position as f64, frequency);
                let (cos, sin) = (alpha * cos, alpha * sin);
                let (a, b) = self.rotary.plane(plane);
                for head in 0..heads {
                    let offset = head * self.geometry.head_dim;
                    let (a, b) = ((t, offset + a), (t, offset + b));
                    let (xa, xb, ra, rb) = (value[a], value[b], radius[a], radius[b]);
                    value[a] = xa * cos - xb * sin;
                    value[b] = xb * cos + xa * sin;
                    let trig = alpha.abs() * eta * (xa.abs() + xb.abs());
                    radius[a] = ra * cos.abs()
                        + rb * sin.abs()
                        + trig
                        + growth * (xa.abs() * cos.abs() + xb.abs() * sin.abs());
                    radius[b] = rb * cos.abs()
                        + ra * sin.abs()
                        + trig
                        + growth * (xb.abs() * cos.abs() + xa.abs() * sin.abs());
                }
            }
        }
        HeadRows { value, radius }
    }

    /// Causal mask, joint softmax per query row and the value read, propagating
    /// the score and value radii.
    fn mix(&self, scored: ScoredHeads, values: ProjectedRows<'_>) -> Result<ProjectedAttention, AttentionProgramError> {
        let g = self.geometry;
        let (tokens, hd) = (values.values.nrows(), g.head_dim);
        let mut weights = Array3::zeros((g.n_heads, tokens, tokens));
        let mut weight_radius = Array3::zeros((g.n_heads, tokens, tokens));
        let mut mixed = Array2::zeros((tokens, g.query_dim()));
        let mut mixed_radius = Array2::zeros((tokens, g.query_dim()));
        for head in 0..g.n_heads {
            let (qo, vo) = (head * hd, g.key_value_head(head) * hd);
            for t in 0..tokens {
                let logits: Vec<f64> = (0..=t).map(|s| scored.scores[[head, t, s]]).collect();
                let logit_radius: Vec<f64> = (0..=t).map(|s| scored.radius[[head, t, s]]).collect();
                let (row, row_radius) =
                    attention_weights(&logits, &logit_radius).map_err(AttentionProgramError::Categorical)?;
                let mix_growth = accumulation_growth(t + 1);
                for s in 0..=t {
                    weights[[head, t, s]] = row[s];
                    weight_radius[[head, t, s]] = row_radius[s];
                }
                for c in 0..hd {
                    let (mut value, mut abs, mut propagated) = (0.0, 0.0, 0.0);
                    for s in 0..=t {
                        let v = values.values[[s, vo + c]];
                        value += row[s] * v;
                        abs += (row[s] * v).abs();
                        propagated += row_radius[s] * v.abs() + row[s] * values.radius[[s, vo + c]];
                    }
                    mixed[[t, qo + c]] = value;
                    mixed_radius[[t, qo + c]] = propagated + mix_growth * abs;
                }
            }
        }
        Ok(ProjectedAttention {
            scores: scored.scores,
            score_radius: scored.radius,
            weights,
            weight_radius,
            mixed,
            mixed_radius,
        })
    }

}

impl NativeAttention {
    pub fn new(
        geometry: AttentionGeometry,
        rotary: RotaryEmbedding,
        score_scale: f64,
        query: AffineProjection,
        key: AffineProjection,
        value: AffineProjection,
        output: AffineProjection,
    ) -> Result<Self, AttentionProgramError> {
        let attention = RotaryCausalAttention::new(geometry, rotary, score_scale)?;
        let (model, query_dim, kv_dim) = (
            geometry.model_dim,
            geometry.query_dim(),
            geometry.key_value_dim(),
        );
        for (weight_name, bias_name, projection, rows, cols) in [
            ("query weight", "query bias", &query, query_dim, model),
            ("key weight", "key bias", &key, kv_dim, model),
            ("value weight", "value bias", &value, kv_dim, model),
            ("output weight", "output bias", &output, model, query_dim),
        ] {
            expect_shape(weight_name, projection.weight.dim(), (rows, cols))?;
            expect_shape(bias_name, (projection.bias.len(), 1), (rows, 1))?;
        }
        Ok(Self {
            attention,
            query,
            key,
            value,
            output,
            query_key_norm: None,
        })
    }

    /// The source block's dimensions.
    pub fn geometry(&self) -> AttentionGeometry {
        self.attention.geometry
    }

    /// The source's rotary embedding.
    pub fn rotary(&self) -> &RotaryEmbedding {
        &self.attention.rotary
    }

    /// The source's query projection.
    pub fn query(&self) -> &AffineProjection {
        &self.query
    }

    /// The source's key projection.
    pub fn key(&self) -> &AffineProjection {
        &self.key
    }

    /// The source's value projection.
    pub fn value(&self) -> &AffineProjection {
        &self.value
    }

    /// The source's output projection.
    pub fn output(&self) -> &AffineProjection {
        &self.output
    }

    /// The source's multiplier on each query-key inner product.
    pub fn score_scale(&self) -> f64 {
        self.attention.score_scale
    }

    /// Whether the source normalizes each head's queries and keys before the rotary embedding
    /// ([`NativeAttention::with_query_key_norm`]). With a norm, a map that preserves every score
    /// must also commute with that norm, and a caller that projects rows itself would skip it.
    pub fn has_query_key_norm(&self) -> bool {
        self.query_key_norm.is_some()
    }

    /// The source's per-head query/key RMS norm between the projections and the
    /// rotary embedding (Qwen3 `q_norm`, `k_norm`): `w ⊙ h (mean(h²) + ε)^{-1/2}` on
    /// each head's rows, with the source's declared `ε` and gains.
    pub fn with_query_key_norm(
        self,
        epsilon: f64,
        query_gain: Array1<f64>,
        key_gain: Array1<f64>,
    ) -> Result<Self, AttentionProgramError> {
        let head_dim = self.attention.geometry.head_dim;
        expect_shape("query norm gain", (query_gain.len(), 1), (head_dim, 1))?;
        expect_shape("key norm gain", (key_gain.len(), 1), (head_dim, 1))?;
        Ok(Self {
            query_key_norm: Some(QueryKeyNorm {
                epsilon,
                query_gain,
                key_gain,
            }),
            ..self
        })
    }

    /// The source's per-head query norm, when the source has one.
    fn normalize_queries(
        &self,
        rows: (Array2<f64>, Array2<f64>),
    ) -> Result<(Array2<f64>, Array2<f64>), AttentionProgramError> {
        match &self.query_key_norm {
            None => Ok(rows),
            Some(norm) => normalize_heads(
                rows,
                self.attention.geometry.n_heads,
                self.attention.geometry.head_dim,
                norm.epsilon,
                norm.query_gain.view(),
            )
            .map_err(AttentionProgramError::Norm),
        }
    }

    /// The source's per-head key norm, when the source has one.
    fn normalize_keys(
        &self,
        rows: (Array2<f64>, Array2<f64>),
    ) -> Result<(Array2<f64>, Array2<f64>), AttentionProgramError> {
        match &self.query_key_norm {
            None => Ok(rows),
            Some(norm) => normalize_heads(
                rows,
                self.attention.geometry.n_kv_heads,
                self.attention.geometry.head_dim,
                norm.epsilon,
                norm.key_gain.view(),
            )
            .map_err(AttentionProgramError::Norm),
        }
    }

    fn check_input(&self, x: ArrayView2<f64>, positions: &[i64]) -> Result<(), AttentionProgramError> {
        expect_shape(
            "input",
            x.dim(),
            (positions.len(), self.attention.geometry.model_dim),
        )?;
        self.attention.check_positions(positions)
    }

    /// The source block on its original tensors: project queries, keys and
    /// values, attend, and project the output.
    pub fn execute(
        &self,
        x: ArrayView2<f64>,
        positions: &[i64],
    ) -> Result<AttentionExecution, AttentionProgramError> {
        self.check_input(x, positions)?;
        let (queries, query_radius) = self.normalize_queries(project_affine(&self.query, x))?;
        let (keys, key_radius) = self.normalize_keys(project_affine(&self.key, x))?;
        let (values, value_radius) = project_affine(&self.value, x);
        let projected = self.attention.attend_projected(
            ProjectedRows {
                values: queries.view(),
                radius: query_radius.view(),
            },
            ProjectedRows {
                values: keys.view(),
                radius: key_radius.view(),
            },
            ProjectedRows {
                values: values.view(),
                radius: value_radius.view(),
            },
            positions,
        )?;
        Ok(self.project_output(projected))
    }

    /// The source's output projection `W_O h_t + b_O` of the head-mixed rows.
    fn project_output(&self, projected: ProjectedAttention) -> AttentionExecution {
        let g = self.attention.geometry;
        let tokens = projected.mixed.nrows();
        // The `n_heads·head_dim`-term inner product and the bias addition.
        let growth = accumulation_growth(g.query_dim() + 1);
        let mut output = Array2::zeros((tokens, g.model_dim));
        let mut output_radius = Array2::zeros((tokens, g.model_dim));
        for t in 0..tokens {
            for (d, w) in self.output.weight.outer_iter().enumerate() {
                let (inner, abs) = inner_with_abs(w, projected.mixed.row(t));
                let bias = self.output.bias[d];
                let propagated: f64 = w
                    .iter()
                    .zip(projected.mixed_radius.row(t).iter())
                    .map(|(o, r)| o.abs() * r)
                    .sum();
                output[[t, d]] = inner + bias;
                output_radius[[t, d]] = propagated + growth * (abs + bias.abs());
            }
        }
        AttentionExecution {
            scores: projected.scores,
            score_radius: projected.score_radius,
            weights: projected.weights,
            weight_radius: projected.weight_radius,
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

/// Masked component coordinates contracted into head space, plus the source
/// bias, unmasked. The radius is `γ_k` times the absolute sum of the monomials
/// `|U_ci| |m_i| |R_ij x_j|` and `|b_c|`, where `k` counts the readin inner
/// product, the mask product, the contraction and the bias addition.
fn masked_head_rows(
    projection: &ComponentProjection,
    bias: &Array1<f64>,
    mask: ArrayView1<f64>,
    x: ArrayView2<f64>,
) -> HeadRows {
    let (coordinates, coordinate_abs) = project(projection.readins.view(), x);
    let masked = &coordinates * &mask;
    let masked_abs = &coordinate_abs * &mask.mapv(f64::abs);
    let contracted = project(projection.outputs.view(), masked.view()).0;
    let contracted_abs = project(projection.outputs.mapv(f64::abs).view(), masked_abs.view()).0;
    let growth = accumulation_growth(x.ncols() + 1 + projection.components() + 1);
    let value = &contracted + bias;
    let radius = (&contracted_abs + &bias.mapv(f64::abs)) * growth;
    HeadRows { value, radius }
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
        let g = native.attention.geometry;
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
        let HeadRows { value, radius } = masked_head_rows(&self.query, &self.native.query.bias, masks.query.view(), x);
        let (value, radius) = self.native.normalize_queries((value, radius))?;
        let queries = HeadRows { value, radius };
        let HeadRows { value, radius } = masked_head_rows(&self.key, &self.native.key.bias, masks.key.view(), x);
        let (value, radius) = self.native.normalize_keys((value, radius))?;
        let keys = HeadRows { value, radius };
        let (values, value_radius) = project_affine(&self.native.value, x);
        let g = self.native.attention.geometry;
        let rotary = &self.native.attention.rotary;
        let alpha2 = rotary.attention_scaling * rotary.attention_scaling;
        // Per plane: the coordinate product, the in-plane sum, the `cos`/`sin`
        // product and the plane's sum (4), then at most `head_dim` plane additions,
        // `α·α` and its product (2), the pass-through addition (1) and σ (1).
        let growth = accumulation_growth(g.head_dim + 8);
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
                    let (mut planes, mut planes_abs, mut planes_propagated, mut planes_trig) = (0.0, 0.0, 0.0, 0.0);
                    for (plane, &(cos, sin, eta)) in trig.iter().enumerate() {
                        let (a, b) = rotary.plane(plane);
                        let (qa, qb) = (queries.value[(t, qo + a)], queries.value[(t, qo + b)]);
                        let (ka, kb) = (keys.value[(s, ko + a)], keys.value[(s, ko + b)]);
                        let (rqa, rqb) = (queries.radius[(t, qo + a)], queries.radius[(t, qo + b)]);
                        let (rka, rkb) = (keys.radius[(s, ko + a)], keys.radius[(s, ko + b)]);
                        planes += cos * (qa * ka + qb * kb) + sin * (qb * ka - qa * kb);
                        planes_abs += cos.abs() * ((qa * ka).abs() + (qb * kb).abs())
                            + sin.abs() * ((qb * ka).abs() + (qa * kb).abs());
                        let direct = rqa * ka.abs() + qa.abs() * rka + rqb * kb.abs() + qb.abs() * rkb;
                        let crossed = rqb * ka.abs() + qb.abs() * rka + rqa * kb.abs() + qa.abs() * rkb;
                        planes_propagated += cos.abs() * direct + sin.abs() * crossed;
                        planes_trig += eta * (qa.abs() + qb.abs()) * (ka.abs() + kb.abs());
                    }
                    let (mut pass, mut pass_abs, mut pass_propagated) = (0.0, 0.0, 0.0);
                    for c in rotated..hd {
                        let (q, k) = ((t, qo + c), (s, ko + c));
                        pass += queries.value[q] * keys.value[k];
                        pass_abs += (queries.value[q] * keys.value[k]).abs();
                        pass_propagated +=
                            queries.radius[q] * keys.value[k].abs() + queries.value[q].abs() * keys.radius[k];
                    }
                    scores[[head, t, s]] = self.native.attention.score_scale * (alpha2 * planes + pass);
                    radius[[head, t, s]] = self.native.attention.score_scale.abs()
                        * (alpha2.abs() * (planes_propagated + planes_trig)
                            + pass_propagated
                            + growth * (alpha2.abs() * planes_abs + pass_abs));
                }
            }
        }
        let projected = self.native.attention.mix(
            ScoredHeads { scores, radius },
            ProjectedRows {
                values: values.view(),
                radius: value_radius.view(),
            },
        )?;
        Ok(self.native.project_output(projected))
    }

    /// `C_ij(Δ) = σ u^Q_{h,i}ᵀ R_Δ u^K_{g(h),j}`, evaluated plane by plane:
    /// `α² Σ_p [cos(Δθ_p)(u_a w_a + u_b w_b) + sin(Δθ_p)(u_b w_a − u_a w_b)]` plus
    /// the unrotated coordinates.
    pub fn kernel(&self, head: usize, displacement: i64) -> Result<QueryKeyKernel, AttentionProgramError> {
        let g = self.native.attention.geometry;
        if head >= g.n_heads {
            return Err(AttentionProgramError::HeadOutOfRange {
                head,
                n_heads: g.n_heads,
            });
        }
        if displacement.unsigned_abs() > 2 * EXACT_POSITION_LIMIT as u64 {
            return Err(AttentionProgramError::DisplacementNotExact { displacement });
        }
        let rotary = &self.native.attention.rotary;
        let alpha2 = rotary.attention_scaling * rotary.attention_scaling;
        let (hd, rotated) = (g.head_dim, rotary.rotary_dim());
        let (qo, ko) = (head * hd, g.key_value_head(head) * hd);
        let trig: Vec<(f64, f64, f64)> = rotary
            .inverse_frequencies
            .iter()
            .map(|&frequency| plane_trig(displacement as f64, frequency))
            .collect();
        // The gain products (1), the coordinate product, the in-plane sum, the
        // `cos`/`sin` product and the plane's sum (4), at most `head_dim` plane
        // additions, `α·α` and its product (2), the pass-through addition (1) and σ (1).
        let growth = accumulation_growth(hd + 9);
        let (query_gain, key_gain) = match &self.native.query_key_norm {
            Some(norm) => (norm.query_gain.clone(), norm.key_gain.clone()),
            None => (Array1::ones(hd), Array1::ones(hd)),
        };
        let (rows, cols) = (self.query.components(), self.key.components());
        let mut values = Array2::zeros((rows, cols));
        let mut radius = Array2::zeros((rows, cols));
        for i in 0..rows {
            let u: Array1<f64> = (0..hd).map(|c| query_gain[c] * self.query.outputs[[qo + c, i]]).collect();
            for j in 0..cols {
                let w: Array1<f64> = (0..hd).map(|c| key_gain[c] * self.key.outputs[[ko + c, j]]).collect();
                let (mut planes, mut planes_abs, mut planes_trig) = (0.0, 0.0, 0.0);
                for (plane, &(cos, sin, eta)) in trig.iter().enumerate() {
                    let (a, b) = rotary.plane(plane);
                    let (ua, ub, wa, wb) = (u[a], u[b], w[a], w[b]);
                    planes += cos * (ua * wa + ub * wb) + sin * (ub * wa - ua * wb);
                    planes_abs += cos.abs() * (ua * wa).abs()
                        + cos.abs() * (ub * wb).abs()
                        + sin.abs() * ((ub * wa).abs() + (ua * wb).abs());
                    planes_trig += eta * (ua.abs() + ub.abs()) * (wa.abs() + wb.abs());
                }
                let (mut pass, mut pass_abs) = (0.0, 0.0);
                for c in rotated..hd {
                    pass += u[c] * w[c];
                    pass_abs += (u[c] * w[c]).abs();
                }
                values[[i, j]] = self.native.attention.score_scale * (alpha2 * planes + pass);
                radius[[i, j]] = self.native.attention.score_scale.abs()
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
    use qd::Quad;

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
        query_bias: Array1<f64>,
        key_bias: Array1<f64>,
        value_bias: Array1<f64>,
        output_bias: Array1<f64>,
        query_key_norm: Option<(f64, Array1<f64>, Array1<f64>)>,
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
                query_bias: Array1::zeros(12),
                key_bias: Array1::zeros(6),
                value_bias: Array1::zeros(6),
                output_bias: Array1::zeros(4),
                query_key_norm: None,
            }
        }

        /// Qwen3's per-head query/key RMS norm, with dyadic gains in `[1/2, 3/2]`
        /// and a declared epsilon of `1/64`.
        fn normalized(self) -> Self {
            Self {
                query_key_norm: Some((
                    0.015625,
                    dyadic(6, 1, 12, 8.0).column(0).mapv(|gain| gain + 1.0),
                    dyadic(6, 1, 13, 8.0).column(0).mapv(|gain| gain + 1.0),
                )),
                ..self
            }
        }

        /// Source projection biases, as on Pythia's `query_key_value` and `dense`.
        /// They are dyadic, so the edited-tensor route stays exact.
        fn biased(self) -> Self {
            Self {
                query_bias: dyadic(12, 1, 8, 8.0).column(0).to_owned(),
                key_bias: dyadic(6, 1, 9, 8.0).column(0).to_owned(),
                value_bias: dyadic(6, 1, 10, 8.0).column(0).to_owned(),
                output_bias: dyadic(4, 1, 11, 8.0).column(0).to_owned(),
                ..self
            }
        }

        fn native_with(&self, query: Array2<f64>, key: Array2<f64>) -> NativeAttention {
            let affine = |weight: Array2<f64>, bias: &Array1<f64>| AffineProjection {
                weight,
                bias: bias.clone(),
            };
            let native = NativeAttention::new(
                self.geometry,
                self.rotary.clone(),
                self.score_scale,
                affine(query, &self.query_bias),
                affine(key, &self.key_bias),
                affine(self.value.clone(), &self.value_bias),
                affine(self.output.clone(), &self.output_bias),
            )
            .expect("fixture tensors match the geometry");
            match &self.query_key_norm {
                None => native,
                Some((epsilon, query_gain, key_gain)) => native
                    .with_query_key_norm(*epsilon, query_gain.clone(), key_gain.clone())
                    .expect("norm gains match the head dimension"),
            }
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
        let fixture = Fixture::new(RotaryPairing::HalfSplit).biased();
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
        let biased = program
            .execute(&continuous_masks(), fixture.x.view(), &fixture.positions)
            .expect("biased component execution");
        let unbiased = Fixture::new(RotaryPairing::HalfSplit)
            .edited(&continuous_masks())
            .execute(fixture.x.view(), &fixture.positions)
            .expect("bias-free execution");
        let bias_separation = worst_ratio(&biased, &unbiased);
        assert!(
            bias_separation > 1.0,
            "dropping the source biases must leave the radii, got {bias_separation}"
        );
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
        let identity = || AffineProjection {
            weight: Array2::eye(4),
            bias: Array1::zeros(4),
        };
        let score = |pairing: RotaryPairing| -> (f64, f64) {
            let executed = NativeAttention::new(
                geometry,
                RotaryEmbedding {
                    pairing,
                    inverse_frequencies: vec![frequency, frequency],
                    attention_scaling,
                },
                score_scale,
                identity(),
                identity(),
                identity(),
                identity(),
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
                let (joint, joint_radius) =
                    attention_weights(&summed, &summed_radius).expect("finite summed logits");
                let mut mixture = vec![0.0; t + 1];
                let mut mixture_radius = vec![0.0; t + 1];
                for i in 0..QUERY_COMPONENTS {
                    let (weights, radius) =
                        attention_weights(&logits[i], &logit_radius[i]).expect("finite component logits");
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

    /// Exact rows give bit for bit what explicit zero radii give. A nonzero query
    /// radius changes the bits, so the comparison sees the radius channel.
    #[test]
    fn exact_rows_match_explicit_zero_radii() {
        let fixture = Fixture::new(RotaryPairing::HalfSplit).biased();
        let native = fixture.edited(&all_on());
        let queries = project_affine(&native.query, fixture.x.view()).0;
        let keys = project_affine(&native.key, fixture.x.view()).0;
        let values = project_affine(&native.value, fixture.x.view()).0;
        let (zero_queries, zero_keys, zero_values) = (
            Array2::zeros(queries.dim()),
            Array2::zeros(keys.dim()),
            Array2::zeros(values.dim()),
        );
        let bits = |p: &ProjectedAttention| -> Vec<u64> {
            p.scores
                .iter()
                .chain(p.score_radius.iter())
                .chain(p.weights.iter())
                .chain(p.weight_radius.iter())
                .chain(p.mixed.iter())
                .chain(p.mixed_radius.iter())
                .map(|v| v.to_bits())
                .collect()
        };
        let explicit = native
            .attention
            .attend_projected(
                ProjectedRows {
                    values: queries.view(),
                    radius: zero_queries.view(),
                },
                ProjectedRows {
                    values: keys.view(),
                    radius: zero_keys.view(),
                },
                ProjectedRows {
                    values: values.view(),
                    radius: zero_values.view(),
                },
                &fixture.positions,
            )
            .expect("explicit zero radii");
        let exact = native
            .attention
            .attend_projected(
                ProjectedRows::exact(queries.view()),
                ProjectedRows::exact(keys.view()),
                ProjectedRows::exact(values.view()),
                &fixture.positions,
            )
            .expect("exact rows");
        assert_eq!(bits(&exact), bits(&explicit), "exact rows must equal explicit zero radii");
        let widened_queries = Array2::from_elem(queries.dim(), 0.125);
        let widened = native
            .attention
            .attend_projected(
                ProjectedRows {
                    values: queries.view(),
                    radius: widened_queries.view(),
                },
                ProjectedRows::exact(keys.view()),
                ProjectedRows::exact(values.view()),
                &fixture.positions,
            )
            .expect("widened query radius");
        assert_ne!(bits(&widened), bits(&explicit), "a nonzero query radius must change the radii");
    }

    /// Double-double unit roundoff: the low word's last place relative to the high
    /// word, `2^-104`.
    const QUAD_UNIT: f64 = f64::EPSILON * f64::EPSILON;

    fn ulp(value: f64) -> f64 {
        value.abs().next_up() - value.abs()
    }

    /// `sin x` or `cos x` by the alternating Taylor series in double-double. Its
    /// derived error is `QUAD_UNIT` times three rounded operations per term times the
    /// largest term, plus the first omitted term: the series alternates, and its terms
    /// decrease once the order exceeds `|x|`.
    fn quad_trig(x: f64, sine: bool) -> (Quad, f64) {
        let square = Quad::from_f64(x) * Quad::from_f64(x);
        let mut term = if sine {
            Quad::from_f64(x)
        } else {
            Quad::from_f64(1.0)
        };
        let mut order = if sine { 1.0_f64 } else { 0.0 };
        let mut sum = term;
        let mut largest = term.0.abs();
        let mut terms = 1usize;
        loop {
            let next = Quad::from_f64(0.0) - term * square / Quad::from_f64((order + 1.0) * (order + 2.0));
            order += 2.0;
            if order > x.abs() && next.0.abs() <= QUAD_UNIT * largest {
                return (sum, 3.0 * terms as f64 * QUAD_UNIT * largest + next.0.abs());
            }
            term = next;
            sum = sum + term;
            largest = largest.max(term.0.abs());
            terms += 1;
        }
    }

    /// `e^x` in double-double through the positive series; for `x < 0` it computes
    /// `1/e^{-x}`, so no term cancels. Once the order exceeds `2|x|` consecutive terms
    /// shrink by at least half, so the omitted tail is at most twice the first
    /// omitted term.
    fn quad_exp(x: f64) -> (Quad, f64) {
        let magnitude = x.abs();
        let mut term = Quad::from_f64(1.0);
        let mut sum = term;
        let mut order = 0.0_f64;
        let mut terms = 1usize;
        loop {
            order += 1.0;
            term = term * Quad::from_f64(magnitude) / Quad::from_f64(order);
            if order + 1.0 > 2.0 * magnitude && term.0 <= QUAD_UNIT * sum.0 {
                let relative = 3.0 * terms as f64 * QUAD_UNIT + 2.0 * term.0 / sum.0;
                return if x < 0.0 {
                    let inverse = Quad::from_f64(1.0) / sum;
                    (inverse, inverse.0 * (relative + QUAD_UNIT))
                } else {
                    (sum, sum.0 * relative)
                };
            }
            sum = sum + term;
            terms += 1;
        }
    }

    /// `|f_libm − f_ref| ≤ ulp(f_libm) + e_ref`, evaluated in double-double.
    fn within_one_ulp(computed: f64, (reference, reference_error): (Quad, f64)) -> bool {
        (Quad::from_f64(computed) - reference).0.abs() <= ulp(computed) + reference_error
    }

    /// The positive control: `computed` moved 2 ulp further from the reference,
    /// so its error is at least 2 ulp whichever side libm rounded to.
    fn moved_away(computed: f64, reference: Quad) -> f64 {
        if (Quad::from_f64(computed) - reference).0 >= 0.0 {
            computed.next_up().next_up()
        } else {
            computed.next_down().next_down()
        }
    }

    /// The radii's libm assumption, checked at every angle and exponent the fixtures
    /// evaluate: `sin` and `cos` at `p·θ` for `|p| ≤ 9` and `θ ∈ {1, 1/4, 3/4}`, and
    /// `exp` at every executed log weight. This checks the documented platform bound
    /// on the platform the tests run on. A value moved 2 ulp away must fail.
    #[test]
    fn libm_sin_cos_exp_are_within_one_ulp_at_the_fixture_points() {
        let mut checked = 0usize;
        for multiplier in -9..=9 {
            for frequency in [1.0, 0.25, 0.75] {
                let angle = f64::from(multiplier) * frequency;
                let (sin, cos) = angle.sin_cos();
                for (value, reference) in [(sin, quad_trig(angle, true)), (cos, quad_trig(angle, false))] {
                    assert!(
                        within_one_ulp(value, reference),
                        "libm trig at {angle}: {value} is beyond one ulp"
                    );
                    assert!(
                        !within_one_ulp(moved_away(value, reference.0), reference),
                        "a value moved 2 ulp away at {angle} must fail"
                    );
                    checked += 1;
                }
            }
        }
        let fixture = Fixture::new(RotaryPairing::HalfSplit).biased();
        let executed = fixture
            .program()
            .execute(&continuous_masks(), fixture.x.view(), &fixture.positions)
            .expect("component execution");
        for head in 0..fixture.geometry.n_heads {
            for t in 0..fixture.positions.len() {
                let row: Vec<f64> = (0..=t).map(|s| executed.scores[[head, t, s]]).collect();
                let (log_weights, evaluation_radius) = log_softmax_with_error(&row).expect("finite row");
                assert_eq!(log_weights.len(), evaluation_radius.len(), "one radius per log weight");
                for &log_weight in &log_weights {
                    let value = log_weight.exp();
                    let reference = quad_exp(log_weight);
                    assert!(
                        within_one_ulp(value, reference),
                        "libm exp at {log_weight}: {value} is beyond one ulp"
                    );
                    assert!(
                        !within_one_ulp(moved_away(value, reference.0), reference),
                        "a value moved 2 ulp away at {log_weight} must fail"
                    );
                    checked += 1;
                }
            }
        }
        assert!(checked > 0, "the libm check must evaluate at least one point");
    }

    /// Qwen3's per-head query/key norm. The component program still equals direct
    /// execution with the edited tensors within the summed radii, because the norm
    /// runs natively on the same summed head rows. Dropping the norm leaves the radii.
    #[test]
    fn component_program_with_query_key_norm_equals_edited_tensor_execution() {
        let fixture = Fixture::new(RotaryPairing::HalfSplit).biased().normalized();
        let masks = continuous_masks();
        let component = fixture
            .program()
            .execute(&masks, fixture.x.view(), &fixture.positions)
            .expect("component execution with the norm");
        let edited = fixture
            .edited(&masks)
            .execute(fixture.x.view(), &fixture.positions)
            .expect("edited-tensor execution with the norm");
        let agreement = worst_ratio(&component, &edited);
        assert!(
            agreement <= 1.0,
            "with the query/key norm the component program leaves edited-tensor execution by {agreement} of the summed radii"
        );
        let unnormalized = Fixture::new(RotaryPairing::HalfSplit)
            .biased()
            .edited(&masks)
            .execute(fixture.x.view(), &fixture.positions)
            .expect("execution without the norm");
        let separation = worst_ratio(&component, &unnormalized);
        assert!(
            separation > 1.0,
            "dropping the query/key norm must leave the radii, got {separation}"
        );
    }

    /// With the query/key norm the score is `ν^Q ν^K a_tᵀ C^w(p_s − p_t) b_s`: the
    /// kernel folds the gains into the outputs, and each `ν` is the owner's normalizer
    /// of a summed head row. Omitting the normalizers leaves the tolerance.
    #[test]
    fn kernel_with_query_key_norm_reads_normalized_scores() {
        let fixture = Fixture::new(RotaryPairing::HalfSplit).normalized();
        let program = fixture.program();
        let masks = continuous_masks();
        let executed = program
            .execute(&masks, fixture.x.view(), &fixture.positions)
            .expect("component execution with the norm");
        let epsilon = fixture
            .query_key_norm
            .as_ref()
            .map(|norm| norm.0)
            .expect("normalized fixture");
        let hd = fixture.geometry.head_dim;
        let normalizer = |row: ArrayView1<f64>| -> f64 {
            rms_normalizers(row.insert_axis(ndarray::Axis(0)), epsilon).expect("finite head row")[0]
        };
        // Two products per kernel term and the additions, then the two normalizer products.
        let growth = accumulation_growth(QUERY_COMPONENTS * KEY_COMPONENTS + 4);
        // Each normalizer is within the owner's `γ_{d+4}` of its real value.
        let normalizer_error = 2.0 * accumulation_growth(hd + 4);
        let (mut forward_worst, mut unnormalized_worst) = (0.0_f64, 0.0_f64);
        for head in 0..fixture.geometry.n_heads {
            let (qo, ko) = (head * hd, fixture.geometry.key_value_head(head) * hd);
            for t in 0..fixture.positions.len() {
                let a = coordinates(&fixture.query_readins, &masks.query, fixture.x.row(t));
                let query = fixture.query_outputs.dot(&a);
                let query_normalizer = normalizer(query.slice(ndarray::s![qo..qo + hd]));
                for s in 0..=t {
                    let b = coordinates(&fixture.key_readins, &masks.key, fixture.x.row(s));
                    let key = fixture.key_outputs.dot(&b);
                    let key_normalizer = normalizer(key.slice(ndarray::s![ko..ko + hd]));
                    let kernel = program
                        .kernel(head, fixture.positions[s] - fixture.positions[t])
                        .expect("kernel");
                    let (mut value, mut abs, mut propagated) = (0.0, 0.0, 0.0);
                    for i in 0..QUERY_COMPONENTS {
                        for j in 0..KEY_COMPONENTS {
                            value += a[i] * kernel.values[[i, j]] * b[j];
                            abs += (a[i] * kernel.values[[i, j]] * b[j]).abs();
                            propagated += (a[i] * b[j]).abs() * kernel.radius[[i, j]];
                        }
                    }
                    let scale = query_normalizer * key_normalizer;
                    let normalized = scale * value;
                    let tolerance = scale * (propagated + growth * abs)
                        + normalizer_error * normalized.abs()
                        + executed.score_radius[[head, t, s]];
                    let score = executed.scores[[head, t, s]];
                    forward_worst = forward_worst.max((normalized - score).abs() / tolerance);
                    unnormalized_worst = unnormalized_worst.max((value - score).abs() / tolerance);
                }
            }
        }
        assert!(
            forward_worst <= 1.0,
            "ν^Q ν^K aᵀ C^w b leaves the executed scores by {forward_worst} of the tolerance"
        );
        assert!(
            unnormalized_worst > 1.0,
            "the kernel without the normalizers must leave the executed scores, got {unnormalized_worst}"
        );
    }

    /// The read accessors return the parts the block was built from (geometry, rotary embedding,
    /// score scale and all four projections), and `has_query_key_norm` follows
    /// `with_query_key_norm`. Positive control: an edited query tensor compares unequal to the
    /// accessor's.
    #[test]
    fn accessors_return_the_source_parts() {
        let fixture = Fixture::new(RotaryPairing::HalfSplit).biased();
        let native = fixture.edited(&all_on());
        assert_eq!(native.geometry(), fixture.geometry, "geometry");
        assert_eq!(native.rotary(), &fixture.rotary, "rotary embedding");
        assert_eq!(native.query().bias, fixture.query_bias, "query bias");
        assert_eq!(native.key().bias, fixture.key_bias, "key bias");
        let source_query = masked_product(&fixture.query_outputs, &all_on().query, &fixture.query_readins);
        assert_eq!(native.query().weight, source_query, "query weight");
        let source_key = masked_product(&fixture.key_outputs, &all_on().key, &fixture.key_readins);
        assert_eq!(native.key().weight, source_key, "key weight");
        assert_eq!(native.value().weight, fixture.value, "value weight");
        assert_eq!(native.value().bias, fixture.value_bias, "value bias");
        assert_eq!(native.output().weight, fixture.output, "output weight");
        assert_eq!(native.output().bias, fixture.output_bias, "output bias");
        assert_eq!(native.score_scale(), fixture.score_scale, "score scale");
        assert!(
            !native.has_query_key_norm(),
            "a block built without q_norm/k_norm has no query/key norm"
        );
        let normalized = Fixture::new(RotaryPairing::HalfSplit).normalized().edited(&all_on());
        assert!(
            normalized.has_query_key_norm(),
            "with_query_key_norm must set the query/key norm"
        );
        let edited_query = masked_product(&fixture.query_outputs, &continuous_masks().query, &fixture.query_readins);
        assert_ne!(
            native.query().weight,
            edited_query,
            "the accessor must return the source query tensor, not an edited one"
        );
    }
}
