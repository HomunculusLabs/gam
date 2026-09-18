//! Residual genetic repair (gam#2924): a shrunk block of conditionally centred
//! genetic residual features inside the anchored Bernoulli marginal-slope
//! likelihood.
//!
//! # The model
//!
//! ```text
//!   η_i = c(a_i)·q(a_i) + s·( b(a_i)·z_i + βᵀ r_i ),      r_i = φ_i − E_ref[φ | S_i, A_i]
//!   c(a) = √(1 + s²· b̃(a)ᵀ Σ(a) b̃(a)),                     b̃(a) = (b(a), β)
//!   Σ(a) = Var((z, r) | a),   s = 1/√(1+σ²) the probit frailty scale
//! ```
//!
//! The scalar score kept ONE direction of the genome; a varying slope `b(a)`
//! rescales that direction and cannot turn it into a different one. `βᵀr` reads
//! the directions the score discarded, and the population value it removes from
//! squared risk is exactly `c_rᵀ Σ_r⁺ c_r` with `c_r = E[rY]`, `Σ_r = E[rrᵀ]`
//! (`Descent.Portability.ResidualGeneticRepair.residual_repair_law`).
//!
//! # Why the anchor integrates the WHOLE drive
//!
//! The marginal interpretation `E[p | a] = Φ(q(a))` is what keeps the baseline
//! surface meaningful. Under a conditionally Gaussian joint law of `(z, r)` the
//! anchoring equation `E_{z,r}[Φ(c·q + s(b z + βᵀr)) | a] = Φ(q)` is solved by
//! `c(a) = √(1 + s²·b̃ᵀΣ(a)b̃)` — the gam#2766 identity with the residual block
//! appended to the score vector. Adding `βᵀr` WITHOUT moving the anchor would
//! leak `Var(βᵀr | a)` into the baseline: the same defect as an unconditioned
//! score. With the anchor holding along every parameter path, the baseline and
//! the predictor-shape directions are Fisher-orthogonal under the declared law
//! (`Descent.Portability.MarginalAnchor.crossInformation_baseline_shape_zero`),
//! which is what makes `β` estimable without corrupting `q`.
//!
//! # Where the block lives in the row program
//!
//! The rigid marginal-slope kernel is a `RowKernel<2>` over the primaries
//! `(q, g)`, each a row-linear predictor of its block. The residual block cannot
//! be a third row-linear primary: `c(a)` depends on `β` through the quadratic
//! form `βᵀΣ_rr β`, which no per-row scalar `r_iᵀβ` carries. So the residual
//! COEFFICIENTS are the primaries — `RowKernel<2+K>` over `(q, g, β_1, …, β_K)`
//! with a unit design on the last `K` — exactly the survival multi-score
//! structure with an intercept-only slope surface per residual feature. Every
//! derivative channel (value, gradient, Hessian, the third- and fourth-order
//! contractions the REML/LAML outer objective reads) is then derived by the
//! canonical jet lowering of the ONE row program below; nothing here is a hand
//! schedule that could disagree with it.
//!
//! # The centring contract
//!
//! The features are supplied already centred on the reference law. The fit does
//! not centre them — that would silently absorb a level into the baseline — it
//! CHECKS `E_w[r_k | marginal-index span] = 0` with the same robust Rao score
//! test the score's conditional gate uses (gam#2768), at the same level, and
//! refuses with a typed reason when a column fails.
//!
//! # The penalty
//!
//! One ridge block `λ_r·‖β‖²` with its own REML/LAML smoothing parameter: not a
//! fixed prior, not a grid. Under `r ⟂ Y` the marginal likelihood drives
//! `λ_r → ∞` and `β → 0`, leaving the score-only fit unchanged.

use super::conditional_score_covariance::ConditionalScoreCovariance;
use super::family::*;
use super::gradient_paths::*;
use super::hessian_paths::BlockSlices;
use super::*;
use gam_math::jet_scalar::{JetScalar, SymmetricQuadraticCoefficients};
use ndarray::Zip;

/// Name of the residual block in the parameter-block list and in the
/// identifiability audit.
pub const RESIDUAL_BLOCK_NAME: &str = "residual_repair";

/// The widest residual block the const-generic row kernel is instantiated for.
/// The kernel's primary count is `2 + K`; each width is a separate monomorphic
/// instantiation of the generic row-kernel assembly, so the ladder is finite.
pub const MAX_RESIDUAL_COLUMNS: usize = 12;

/// Gauge priority of the residual block: below the parametric surfaces (a
/// shared direction is demoted out of the residual block, never out of the
/// marginal or slope surface), above the flex deviations.
pub(super) const GAUGE_PRIORITY_RESIDUAL: u8 = 110;

/// Why a residual block was refused. Each reason names the contract that was
/// not met; none is a solver failure.
#[derive(Clone, Debug, PartialEq)]
pub enum ResidualRepairRefusal {
    /// `E_w[r_k | marginal-index span] = 0` was rejected at level `alpha`.
    ColumnNotCentred {
        column: String,
        p_value: f64,
        alpha: f64,
    },
    /// A feature value was not finite.
    ColumnNonFinite { column: String, row: usize },
    /// A feature column has no weighted variation, so it carries no direction.
    ColumnConstant { column: String },
    /// The block is wider than the instantiated kernel ladder.
    WidthUnsupported { width: usize, max: usize },
    /// No residual columns were named.
    Empty,
    /// The feature matrix does not have one row per fitted observation.
    RowCountMismatch { rows: usize, expected: usize },
    /// The residual block is not lowered through the flexible (score-warp /
    /// link-deviation) row program.
    FlexBlocksUnsupported,
    /// A learned frailty scale is an outer axis the residual kernel does not
    /// differentiate; pass a fixed `frailty_sd` instead.
    LearnedFrailtyUnsupported,
    /// The empirical latent measure integrates `z` on a grid; integrating the
    /// joint `(z, r)` law on that grid is not implemented here.
    EmpiricalLatentMeasureUnsupported,
    /// The CTN Stage-1 influence absorber widens the marginal block; the
    /// residual kernel reads the marginal design at its raw width.
    InfluenceAbsorberUnsupported,
    /// The centring test of a column could not be evaluated.
    CentringTestUnavailable { column: String, reason: String },
    /// The joint law of `(z, r)` could not be estimated on the fit rows.
    JointCovarianceUnavailable { reason: String },
}

impl std::fmt::Display for ResidualRepairRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ColumnNotCentred {
                column,
                p_value,
                alpha,
            } => write!(
                f,
                "residual column '{column}' is not centred on the marginal-index span: the robust \
                 Rao score test of E_w[r | span] = 0 has p = {p_value:.3e} < {alpha:.1e}. Centre the \
                 feature on the reference law (r = φ − E_ref[φ | S, A]) before passing it; the fit \
                 will not absorb the level into the baseline"
            ),
            Self::ColumnNonFinite { column, row } => {
                write!(f, "residual column '{column}' is non-finite at row {row}")
            }
            Self::ColumnConstant { column } => write!(
                f,
                "residual column '{column}' has zero weighted variance on the fit data and carries no \
                 direction"
            ),
            Self::WidthUnsupported { width, max } => write!(
                f,
                "residual block has {width} columns; at most {max} are supported by the residual \
                 row kernel"
            ),
            Self::Empty => write!(f, "residual_columns is empty"),
            Self::RowCountMismatch { rows, expected } => write!(
                f,
                "residual feature matrix has {rows} rows but the fit has {expected} observations"
            ),
            Self::FlexBlocksUnsupported => write!(
                f,
                "residual_columns cannot be combined with linkwiggle(...) score-warp / \
                 link-deviation blocks: the residual block is lowered only through the rigid \
                 standard-normal row kernel"
            ),
            Self::LearnedFrailtyUnsupported => write!(
                f,
                "residual_columns requires a fixed frailty_sd (or none): a learned frailty scale is \
                 not differentiated by the residual row kernel"
            ),
            Self::EmpiricalLatentMeasureUnsupported => write!(
                f,
                "residual_columns requires the standard-normal latent measure: the score did not \
                 pass the conditional-normality gate and fell back to the empirical grid, on which \
                 the joint (z, r) anchor is not implemented"
            ),
            Self::InfluenceAbsorberUnsupported => write!(
                f,
                "residual_columns cannot be combined with an absorbed CTN Stage-1 influence block"
            ),
            Self::CentringTestUnavailable { column, reason } => write!(
                f,
                "the centring test of residual column '{column}' could not be evaluated: {reason}"
            ),
            Self::JointCovarianceUnavailable { reason } => write!(
                f,
                "the joint law of the score and the residual columns could not be estimated: \
                 {reason}"
            ),
        }
    }
}

impl std::error::Error for ResidualRepairRefusal {}

/// The residual block as the caller supplies it: named columns, already
/// centred on the reference law, one row per fitted observation.
#[derive(Clone, Debug)]
pub struct ResidualRepairSpec {
    pub columns: Vec<String>,
    /// `n × K`.
    pub features: Array2<f64>,
}

/// The fitted geometry of the residual block: what prediction needs to replay
/// the anchor, persisted with the model.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ResidualRepairGeometry {
    /// Column names, in coefficient order.
    pub columns: Vec<String>,
    /// Pooled `Var((z, r))` as a dense `(K+1) × (K+1)` matrix, coordinate `0`
    /// the latent score at its declared unit variance. This is the fit's
    /// summary covariance and the object the no-escalation anchor consumes.
    pub pooled_covariance: Vec<Vec<f64>>,
    /// The gam#2766 conditional model `Σ(a)` when the pairwise Rao gate fired
    /// on some `(z, r_j)` or `(r_j, r_k)` pair. `None` ⇒ every row uses the
    /// pooled matrix.
    pub conditional_covariance: Option<ConditionalScoreCovariance>,
    /// The centring-gate p-value of each column, in coefficient order.
    pub centring_pvalues: Vec<f64>,
}

impl ResidualRepairGeometry {
    /// `K`.
    pub fn width(&self) -> usize {
        self.columns.len()
    }

    /// Rebuild the row-wise covariance field over `(z, r)` for a block of
    /// marginal-design rows (fit or predict).
    pub fn covariance_field(
        &self,
        a_block: ArrayView2<'_, f64>,
    ) -> Result<ScoreCovarianceField, String> {
        let dim = self.width() + 1;
        let mut dense = Array2::<f64>::zeros((dim, dim));
        if self.pooled_covariance.len() != dim {
            return Err(format!(
                "residual repair pooled covariance has {} rows, expected {dim}",
                self.pooled_covariance.len()
            ));
        }
        for (i, row) in self.pooled_covariance.iter().enumerate() {
            if row.len() != dim {
                return Err(format!(
                    "residual repair pooled covariance row {i} has {} entries, expected {dim}",
                    row.len()
                ));
            }
            for (j, &value) in row.iter().enumerate() {
                dense[[i, j]] = value;
            }
        }
        let pooled = MarginalSlopeCovariance::full(dense)?;
        match self.conditional_covariance.as_ref() {
            None => Ok(ScoreCovarianceField::pooled(pooled)),
            Some(model) => ScoreCovarianceField::conditional(pooled, model.clone(), a_block),
        }
    }
}

/// The residual block bound to the training rows: what the family's row
/// kernel reads.
pub struct ResidualBlockRuntime {
    /// `n × K` training features.
    pub features: Array2<f64>,
    /// `Var((z, r) | a_i)` at every training row; `K + 1` coordinates.
    pub field: ScoreCovarianceField,
    pub geometry: ResidualRepairGeometry,
}

impl std::fmt::Debug for ResidualBlockRuntime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResidualBlockRuntime")
            .field("rows", &self.features.nrows())
            .field("width", &self.features.ncols())
            .field("conditional", &self.field.is_conditional())
            .finish()
    }
}

impl ResidualBlockRuntime {
    /// `K`.
    #[inline]
    pub fn width(&self) -> usize {
        self.features.ncols()
    }

    /// Validate, gate, and fit the joint covariance of `(z, r)` on the training
    /// rows.
    ///
    /// `z` is the score the row kernel will actually see (after every latent
    /// calibration), `a_block` the marginal-index span the gam#2768 gates
    /// condition on. Each column is checked for `E_w[r_k | span] = 0` including
    /// the level, with the robust Rao score statistic at
    /// [`AUTO_Z_CONDITIONAL_RAO_ALPHA`]; a failing column is a typed refusal.
    pub fn fit(
        spec: &ResidualRepairSpec,
        z: ArrayView1<'_, f64>,
        weights: ArrayView1<'_, f64>,
        a_block: ArrayView2<'_, f64>,
    ) -> Result<Self, ResidualRepairRefusal> {
        let n = z.len();
        let k = spec.features.ncols();
        if k == 0 || spec.columns.is_empty() {
            return Err(ResidualRepairRefusal::Empty);
        }
        if spec.columns.len() != k {
            return Err(ResidualRepairRefusal::RowCountMismatch {
                rows: spec.columns.len(),
                expected: k,
            });
        }
        if k > MAX_RESIDUAL_COLUMNS {
            return Err(ResidualRepairRefusal::WidthUnsupported {
                width: k,
                max: MAX_RESIDUAL_COLUMNS,
            });
        }
        if spec.features.nrows() != n || weights.len() != n || a_block.nrows() != n {
            return Err(ResidualRepairRefusal::RowCountMismatch {
                rows: spec.features.nrows(),
                expected: n,
            });
        }
        let total_weight = weights.iter().copied().sum::<f64>();
        for (col, name) in spec.columns.iter().enumerate() {
            let column = spec.features.column(col);
            if let Some(row) = column.iter().position(|v| !v.is_finite()) {
                return Err(ResidualRepairRefusal::ColumnNonFinite {
                    column: name.clone(),
                    row,
                });
            }
            let mean = column
                .iter()
                .zip(weights.iter())
                .map(|(&v, &w)| w * v)
                .sum::<f64>()
                / total_weight;
            let var = column
                .iter()
                .zip(weights.iter())
                .map(|(&v, &w)| w * (v - mean) * (v - mean))
                .sum::<f64>()
                / total_weight;
            let magnitude = column.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()));
            let resolution = gam_linalg::roundoff::accumulation_growth(n) * magnitude;
            if !(var.is_finite() && var.sqrt() > resolution) {
                return Err(ResidualRepairRefusal::ColumnConstant {
                    column: name.clone(),
                });
            }
        }

        // The centring gate. `[1 | ã]` — the level is part of the hypothesis,
        // so the intercept column stays in the tested span; the marginal
        // columns are centred so the statistic's rank is the span's.
        let basis = centring_gate_basis(a_block, weights);
        let mut centring_pvalues = Vec::with_capacity(k);
        for (col, name) in spec.columns.iter().enumerate() {
            let u: Vec<f64> = spec.features.column(col).to_vec();
            let p_value = robust_conditional_score_pvalue(basis.view(), &u, weights)
                .map_err(|reason| ResidualRepairRefusal::CentringTestUnavailable {
                    column: name.clone(),
                    reason,
                })?
                .unwrap_or(1.0);
            if p_value < AUTO_Z_CONDITIONAL_RAO_ALPHA {
                return Err(ResidualRepairRefusal::ColumnNotCentred {
                    column: name.clone(),
                    p_value,
                    alpha: AUTO_Z_CONDITIONAL_RAO_ALPHA,
                });
            }
            centring_pvalues.push(p_value);
        }

        // The joint law of (z, r): the pooled second central moment, escalated
        // to Σ(a) by the gam#2766 pairwise gate on the same span, then moved
        // onto the DECLARED law of the score. Read as the modified Cholesky
        // factorisation `Σ = L·diag(d₀, d₁, …)·Lᵀ` with the score first, `L`'s
        // first column is the regression of r on z and `d₁, …` the innovation
        // variances of r given z: together, the conditional law of r given z.
        // `d₀` is Var(z | a), which the latent measure declares rather than
        // estimates — N(0, 1) here — so it is set to 1 and nothing else moves.
        // At β = 0 the anchor is then the rigid kernel's √(1 + s²g²) exactly: a
        // block with nothing to read leaves the score-only fit where it was
        // instead of moving its baseline by the sampling error of Var(z).
        let unavailable =
            |reason: String| ResidualRepairRefusal::JointCovarianceUnavailable { reason };
        let mut scores = Array2::<f64>::zeros((n, k + 1));
        scores.column_mut(0).assign(&z);
        scores.slice_mut(s![.., 1..]).assign(&spec.features);
        let weights_owned = weights.to_owned();
        let sample = marginal_slope_covariance_from_scores(scores.view(), &weights_owned)
            .map_err(unavailable)?
            .to_dense();
        let pooled_dense = declare_unit_score_variance(&sample).map_err(unavailable)?;
        let pooled = MarginalSlopeCovariance::full(pooled_dense.clone()).map_err(unavailable)?;
        let conditional = ConditionalScoreCovariance::fit(scores.view(), weights, a_block)
            .map_err(unavailable)?
            .map(|mut model| {
                // Coordinate 0 of the MCD is the score: its innovation IS
                // Var(z | a). Declare it, keep every regression and every
                // residual innovation the data estimated.
                model.coordinates[0].log_innovation = vec![0.0];
                model.coordinates[0].log_innovation_range = [0.0, 0.0];
                model
            });
        let field = match conditional.as_ref() {
            None => ScoreCovarianceField::pooled(pooled),
            Some(model) => ScoreCovarianceField::conditional(pooled, model.clone(), a_block)
                .map_err(unavailable)?,
        };
        if let Some(model) = conditional.as_ref() {
            log::info!(
                "[BMS residual repair] joint (z, r) covariance escalated to Σ(a): {} pair(s) fired \
                 the conditional gate",
                model
                    .pair_pvalues
                    .iter()
                    .filter(|(_, _, p)| *p < AUTO_Z_CONDITIONAL_RAO_ALPHA)
                    .count()
            );
        }
        Ok(Self {
            features: spec.features.clone(),
            field,
            geometry: ResidualRepairGeometry {
                columns: spec.columns.clone(),
                pooled_covariance: pooled_dense
                    .rows()
                    .into_iter()
                    .map(|row| row.to_vec())
                    .collect(),
                conditional_covariance: conditional,
                centring_pvalues,
            },
        })
    }

    /// Rebind a persisted geometry to a block of rows (prediction, ALO replay).
    pub fn from_geometry(
        geometry: ResidualRepairGeometry,
        features: Array2<f64>,
        a_block: ArrayView2<'_, f64>,
    ) -> Result<Self, String> {
        if features.ncols() != geometry.width() {
            return Err(format!(
                "residual repair feature width {} does not match the saved block width {}",
                features.ncols(),
                geometry.width()
            ));
        }
        if features.nrows() != a_block.nrows() {
            return Err(format!(
                "residual repair feature rows {} do not match the conditioning rows {}",
                features.nrows(),
                a_block.nrows()
            ));
        }
        let field = geometry.covariance_field(a_block)?;
        Ok(Self {
            features,
            field,
            geometry,
        })
    }

    /// The ridge block spec: identity penalty on `β`, one REML/LAML coordinate.
    pub(super) fn block_spec(
        &self,
        rho: Array1<f64>,
        beta_hint: Option<Array1<f64>>,
    ) -> Result<ParameterBlockSpec, String> {
        let k = self.width();
        if rho.len() != 1 {
            return Err(format!(
                "residual repair block takes exactly one smoothing coordinate, got {}",
                rho.len()
            ));
        }
        let initial_beta = match beta_hint {
            Some(beta) if beta.len() == k => Some(beta),
            _ => Some(Array1::zeros(k)),
        };
        Ok(ParameterBlockSpec {
            name: RESIDUAL_BLOCK_NAME.to_string(),
            design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
                self.features.clone(),
            )),
            offset: Array1::zeros(self.features.nrows()),
            penalties: vec![PenaltyMatrix::Diagonal(Array1::from_elem(k, 1.0))],
            nullspace_dims: vec![0],
            initial_log_lambdas: rho,
            initial_beta,
            gauge_priority: GAUGE_PRIORITY_RESIDUAL,
            jacobian_callback: None,
            stacked_design: None,
            stacked_offset: None,
        })
    }
}

/// `[1 | ã]`: the intercept plus the weighted-centred marginal columns, the span
/// the centring gate tests against.
fn centring_gate_basis(a_block: ArrayView2<'_, f64>, weights: ArrayView1<'_, f64>) -> Array2<f64> {
    let n = a_block.nrows();
    let p = a_block.ncols();
    let total_weight = weights.iter().copied().sum::<f64>();
    let mut basis = Array2::<f64>::zeros((n, p + 1));
    basis.column_mut(0).fill(1.0);
    for col in 0..p {
        let column = a_block.column(col);
        let mean = column
            .iter()
            .zip(weights.iter())
            .map(|(&v, &w)| w * v)
            .sum::<f64>()
            / total_weight;
        Zip::from(basis.column_mut(col + 1))
            .and(column)
            .for_each(|out, &v| *out = v - mean);
    }
    basis
}

/// The joint covariance of `(z, r)` on the score's declared law: the sample
/// `Σ̂ = L·diag(d₀, d₁, …)·Lᵀ` with `d₀` replaced by the declared `Var(z) = 1`.
/// Only `L`'s first column, the regression `γ = Σ̂_r0/Σ̂_00` of r on z, and the
/// Schur complement `Σ̂_rr − Σ̂_r0Σ̂_0r/Σ̂_00`, the law of r given z, survive:
///
/// ```text
///   Σ_00 = 1,   Σ_r0 = γ,   Σ_rr = Σ̂_rr − Σ̂_r0Σ̂_0r/Σ̂_00 + γγᵀ
/// ```
///
/// A Schur complement of a PSD matrix plus a rank-one PSD term, so the result
/// is PSD whenever the sample is.
fn declare_unit_score_variance(sample: &Array2<f64>) -> Result<Array2<f64>, String> {
    let dim = sample.nrows();
    let score_variance = sample[[0, 0]];
    if !(score_variance.is_finite() && score_variance > 0.0) {
        return Err(format!("the score has weighted variance {score_variance}"));
    }
    let inv = score_variance.recip();
    // γγᵀ − Σ̂_r0Σ̂_0r/Σ̂_00 = Σ̂_r0Σ̂_0r·(1/Σ̂_00² − 1/Σ̂_00).
    let rank_one = inv * inv - inv;
    let mut declared = Array2::<f64>::zeros((dim, dim));
    declared[[0, 0]] = 1.0;
    for i in 1..dim {
        declared[[i, 0]] = sample[[i, 0]] * inv;
        declared[[0, i]] = sample[[0, i]] * inv;
        for j in 1..dim {
            declared[[i, j]] = sample[[i, j]] + sample[[i, 0]] * sample[[0, j]] * rank_one;
        }
    }
    Ok(declared)
}

/// Derivative stack of `u ↦ √(1 + u)` at `u`, given `f = √(1 + u)`.
#[inline]
fn sqrt1p_stack(f: f64) -> [f64; 5] {
    let inv = f.recip();
    let inv3 = inv * inv * inv;
    let inv5 = inv3 * inv * inv;
    let inv7 = inv5 * inv * inv;
    [
        f,
        0.5 * inv,
        -0.25 * inv3,
        0.375 * inv5,
        -0.9375 * inv7,
    ]
}

/// The row index and its first derivatives in plain `f64`: the value-only
/// path of the fit and the prediction replay share this one statement of the
/// anchor with the jet program below.
///
/// Returns `(η, ∂η/∂q_eta, ∂η/∂g, ∂η/∂β)`.
pub(crate) fn residual_row_index(
    marginal: &BernoulliMarginalLinkMap,
    g: f64,
    beta: &[f64],
    z: f64,
    r: &[f64],
    covariance: &MarginalSlopeCovariance,
    probit_scale: f64,
) -> Result<(f64, f64, f64, Vec<f64>), String> {
    let k = beta.len();
    if r.len() != k || covariance.dim() != k + 1 {
        return Err(format!(
            "residual row index dimension mismatch: beta={k}, r={}, covariance={}",
            r.len(),
            covariance.dim()
        ));
    }
    let s = probit_scale;
    let mut drive = Vec::with_capacity(k + 1);
    drive.push(s * g);
    drive.extend(beta.iter().map(|&b| s * b));
    let mut sigma_drive = vec![0.0; k + 1];
    covariance.multiply(&drive, &mut sigma_drive);
    let quad: f64 = drive.iter().zip(sigma_drive.iter()).map(|(a, b)| a * b).sum();
    if !(quad.is_finite() && quad >= -f64::EPSILON * (1.0 + quad.abs())) {
        return Err(format!(
            "residual row index: the anchor quadratic form b̃ᵀΣb̃ = {quad} is not admissible"
        ));
    }
    let c = (1.0 + quad.max(0.0)).sqrt();
    let inv_c = c.recip();
    let linear = drive[0] * z + drive[1..].iter().zip(r.iter()).map(|(b, x)| b * x).sum::<f64>();
    let eta = marginal.q * c + linear;
    let d_q = marginal.q1 * c;
    // ∂c/∂g = s·(Σ b̃)_0 / c ; ∂c/∂β_k = s·(Σ b̃)_{k+1} / c   (b̃ already carries s).
    let d_g = marginal.q * s * sigma_drive[0] * inv_c + s * z;
    let d_beta: Vec<f64> = (0..k)
        .map(|j| marginal.q * s * sigma_drive[j + 1] * inv_c + s * r[j])
        .collect();
    Ok((eta, d_q, d_g, d_beta))
}

/// Value-only row negative log-likelihood, bit-equivalent to the jet program's
/// value channel at the same row state.
pub(super) fn residual_row_neglog_only(
    family: &BernoulliMarginalSlopeFamily,
    runtime: &ResidualBlockRuntime,
    block_states: &[ParameterBlockState],
    row: usize,
) -> Result<f64, String> {
    let marginal = family.marginal_link_map(block_states[0].eta[row])?;
    let g = block_states[1].eta[row];
    let beta = block_states[2].beta.as_slice().ok_or("residual beta not contiguous")?;
    let r = runtime.features.row(row);
    let r = r.as_slice().ok_or("residual feature row not contiguous")?;
    let (eta, _, _, _) = residual_row_index(
        &marginal,
        g,
        beta,
        family.z[row],
        r,
        runtime.field.at_row(row),
        family.probit_frailty_scale(),
    )?;
    let w = family.weights[row];
    let sign = 2.0 * family.y[row] - 1.0;
    let stack = signed_probit_neglog_unary_stack(sign * eta, w);
    if !stack[0].is_finite() {
        return Err(format!(
            "residual repair row {row}: non-finite log Φ at η={eta}, y={}, w={w}",
            family.y[row]
        ));
    }
    Ok(stack[0])
}

// ── The row kernel ────────────────────────────────────────────────────

/// The residual-augmented rigid Bernoulli marginal-slope row kernel:
/// `RowKernel<K>` with `K = 2 + width`, primaries `(q_eta, g, β_1, …, β_width)`.
pub(super) struct BernoulliResidualRowKernel<const K: usize> {
    pub(super) family: BernoulliMarginalSlopeFamily,
    pub(super) block_states: Vec<ParameterBlockState>,
    pub(super) slices: BlockSlices,
    pub(super) runtime: Arc<ResidualBlockRuntime>,
}

impl<const K: usize> BernoulliResidualRowKernel<K> {
    pub(super) fn new(
        family: BernoulliMarginalSlopeFamily,
        block_states: Vec<ParameterBlockState>,
    ) -> Result<Self, String> {
        let runtime = family.residual.clone().ok_or_else(|| {
            "residual row kernel constructed on a family without a residual block".to_string()
        })?;
        if runtime.width() + 2 != K {
            return Err(format!(
                "residual row kernel width mismatch: block has {} columns, kernel is K={K}",
                runtime.width()
            ));
        }
        if family.flex_active() {
            return Err(ResidualRepairRefusal::FlexBlocksUnsupported.to_string());
        }
        if family.latent_measure.is_empirical() {
            return Err(ResidualRepairRefusal::EmpiricalLatentMeasureUnsupported.to_string());
        }
        family.validate_exact_block_state_shapes(&block_states)?;
        let slices = super::hessian_paths::block_slices(&family);
        if slices.residual.as_ref().map(|r| r.len()) != Some(K - 2) {
            return Err("residual row kernel: block slices carry no residual range".to_string());
        }
        Ok(Self {
            family,
            block_states,
            slices,
            runtime,
        })
    }

    #[inline]
    fn residual_range(&self) -> std::ops::Range<usize> {
        self.slices
            .residual
            .clone()
            .expect("residual kernel is constructed with a residual range")
    }
}

impl<const K: usize> gam_math::jet_tower::RowProgram<K> for BernoulliResidualRowKernel<K> {
    fn n_rows(&self) -> usize {
        self.family.y.len()
    }

    fn primaries(&self, row: usize) -> Result<[f64; K], String> {
        if row >= self.family.y.len() {
            return Err(format!("BernoulliResidualRowKernel: row {row} out of range"));
        }
        let mut p = [0.0_f64; K];
        p[0] = self.block_states[0].eta[row];
        p[1] = self.block_states[1].eta[row];
        let beta = &self.block_states[2].beta;
        for k in 0..K - 2 {
            p[2 + k] = beta[k];
        }
        Ok(p)
    }

    fn eval<S: JetScalar<K>>(&self, row: usize, p: &[S; K]) -> Result<S, String> {
        if row >= self.family.y.len() {
            return Err(format!("BernoulliResidualRowKernel: row {row} out of range"));
        }
        let marginal = self
            .family
            .marginal_link_map(self.block_states[0].eta[row])?;
        let s = self.family.probit_frailty_scale();
        let z = self.family.z[row];
        let y = self.family.y[row];
        let w = self.family.weights[row];
        // q = Φ⁻¹(Φ(η_m)) through the supplied link stack.
        let q = p[0].compose_unary([
            marginal.q,
            marginal.q1,
            marginal.q2,
            marginal.q3,
            marginal.q4,
        ]);
        // Observed drive coefficients b̃ = s·(g, β).
        let mut drive = [S::constant(0.0); K];
        for k in 1..K {
            drive[k - 1] = p[k].scale(s);
        }
        let drive = &drive[..K - 1];
        let covariance = self.runtime.field.at_row(row);
        let quad = S::symmetric_quadratic_form(drive, covariance);
        let quad_value = quad.value();
        if !(quad_value.is_finite() && quad_value >= -f64::EPSILON * (1.0 + quad_value.abs())) {
            return Err(format!(
                "residual repair row {row}: the anchor quadratic form b̃ᵀΣb̃ = {quad_value} is not \
                 admissible"
            ));
        }
        let c = quad.compose_unary(sqrt1p_stack((1.0 + quad_value.max(0.0)).sqrt()));
        let mut linear_weights = [0.0_f64; K];
        linear_weights[0] = z;
        let r = self.runtime.features.row(row);
        for k in 0..K - 2 {
            linear_weights[1 + k] = r[k];
        }
        let linear = S::linear_combination(drive, &linear_weights[..K - 1]);
        let eta = q.multiply_add(&c, &linear);
        let margin = eta.scale(2.0 * y - 1.0);
        let nll = margin.compose_unary(signed_probit_neglog_unary_stack(margin.value(), w));
        if !nll.value().is_finite() {
            return Err(format!(
                "residual repair row {row}: non-finite log Φ at η={}, y={y}, w={w}",
                eta.value()
            ));
        }
        Ok(nll)
    }
}

impl<const K: usize> crate::row_kernel::RowKernel<K> for BernoulliResidualRowKernel<K> {
    fn n_coefficients(&self) -> usize {
        self.slices.total
    }

    fn jacobian_action(&self, row: usize, d_beta: &[f64]) -> [f64; K] {
        let d_beta = ndarray::ArrayView1::from(d_beta);
        let mut out = [0.0_f64; K];
        out[0] = self
            .family
            .marginal_design
            .dot_row_view(row, d_beta.slice(s![self.slices.marginal.clone()]));
        out[1] = self
            .family
            .slope_design
            .dot_row_view(row, d_beta.slice(s![self.slices.slope.clone()]));
        let residual = self.residual_range();
        for k in 0..K - 2 {
            out[2 + k] = d_beta[residual.start + k];
        }
        out
    }

    fn jacobian_transpose_action(&self, row: usize, v: &[f64; K], out: &mut [f64]) {
        {
            let mut m = ndarray::ArrayViewMut1::from(&mut out[self.slices.marginal.clone()]);
            self.family
                .marginal_design
                .axpy_row_into(row, v[0], &mut m)
                .expect("marginal axpy dim mismatch");
        }
        {
            let mut g = ndarray::ArrayViewMut1::from(&mut out[self.slices.slope.clone()]);
            self.family
                .slope_design
                .axpy_row_into(row, v[1], &mut g)
                .expect("slope axpy dim mismatch");
        }
        let residual = self.residual_range();
        for k in 0..K - 2 {
            out[residual.start + k] += v[2 + k];
        }
    }

    fn add_pullback_hessian(&self, row: usize, h: &[[f64; K]; K], target: &mut Array2<f64>) {
        let marginal = self.slices.marginal.clone();
        let slope = self.slices.slope.clone();
        let residual = self.residual_range();
        self.family
            .marginal_design
            .syr_row_into_view(
                row,
                h[0][0],
                target.slice_mut(s![marginal.clone(), marginal.clone()]),
            )
            .expect("marginal syr dim mismatch");
        if h[0][1] != 0.0 {
            self.family
                .marginal_design
                .row_outer_into_view(
                    row,
                    &self.family.slope_design,
                    h[0][1],
                    target.slice_mut(s![marginal.clone(), slope.clone()]),
                )
                .expect("marginal-slope outer dim mismatch");
            self.family
                .slope_design
                .row_outer_into_view(
                    row,
                    &self.family.marginal_design,
                    h[0][1],
                    target.slice_mut(s![slope.clone(), marginal.clone()]),
                )
                .expect("slope-marginal outer dim mismatch");
        }
        self.family
            .slope_design
            .syr_row_into_view(row, h[1][1], target.slice_mut(s![slope.clone(), slope.clone()]))
            .expect("slope syr dim mismatch");
        // Cross terms with the unit-design residual primaries: one design row
        // scaled into a column (and its transpose) per residual coefficient.
        for k in 0..K - 2 {
            let col = residual.start + k;
            if h[0][2 + k] != 0.0 {
                self.family
                    .marginal_design
                    .axpy_row_into(row, h[0][2 + k], &mut target.slice_mut(s![marginal.clone(), col]))
                    .expect("marginal-residual axpy dim mismatch");
                self.family
                    .marginal_design
                    .axpy_row_into(row, h[0][2 + k], &mut target.slice_mut(s![col, marginal.clone()]))
                    .expect("residual-marginal axpy dim mismatch");
            }
            if h[1][2 + k] != 0.0 {
                self.family
                    .slope_design
                    .axpy_row_into(row, h[1][2 + k], &mut target.slice_mut(s![slope.clone(), col]))
                    .expect("slope-residual axpy dim mismatch");
                self.family
                    .slope_design
                    .axpy_row_into(row, h[1][2 + k], &mut target.slice_mut(s![col, slope.clone()]))
                    .expect("residual-slope axpy dim mismatch");
            }
        }
        for j in 0..K - 2 {
            for k in 0..K - 2 {
                target[[residual.start + j, residual.start + k]] += h[2 + j][2 + k];
            }
        }
    }

    fn add_diagonal_quadratic(&self, row: usize, h: &[[f64; K]; K], diag: &mut [f64]) {
        {
            let mut md = ndarray::ArrayViewMut1::from(&mut diag[self.slices.marginal.clone()]);
            self.family
                .marginal_design
                .squared_axpy_row_into(row, h[0][0], &mut md)
                .expect("marginal squared_axpy dim mismatch");
        }
        {
            let mut gd = ndarray::ArrayViewMut1::from(&mut diag[self.slices.slope.clone()]);
            self.family
                .slope_design
                .squared_axpy_row_into(row, h[1][1], &mut gd)
                .expect("slope squared_axpy dim mismatch");
        }
        let residual = self.residual_range();
        for k in 0..K - 2 {
            diag[residual.start + k] += h[2 + k][2 + k];
        }
    }

    /// `J·F` in one pass: two design GEMMs for the marginal and slope axes, and
    /// a broadcast of the residual rows of `F` for the unit-design axes.
    fn jacobian_action_matrix(&self, factor: ArrayView2<'_, f64>) -> Option<Array2<f64>> {
        let p_total = self.slices.total;
        if factor.nrows() != p_total {
            return None;
        }
        let n_rows = self.family.y.len();
        let rank = factor.ncols();
        let f_marg = factor
            .slice(s![self.slices.marginal.clone(), ..])
            .as_standard_layout()
            .into_owned();
        let f_slope = factor
            .slice(s![self.slices.slope.clone(), ..])
            .as_standard_layout()
            .into_owned();
        let jf_marg =
            crate::row_kernel::row_kernel_design_jf(&self.family.marginal_design, f_marg.view(), n_rows);
        let jf_slope =
            crate::row_kernel::row_kernel_design_jf(&self.family.slope_design, f_slope.view(), n_rows);
        let residual = self.residual_range();
        let mut axes = vec![(0usize, jf_marg), (1usize, jf_slope)];
        for k in 0..K - 2 {
            let mut broadcast = Array2::<f64>::zeros((n_rows, rank));
            let source = factor.row(residual.start + k);
            for mut out_row in broadcast.rows_mut() {
                out_row.assign(&source);
            }
            axes.push((2 + k, broadcast));
        }
        Some(crate::row_kernel::row_kernel_pack_jf_axes::<K>(n_rows, rank, axes))
    }
}

/// Dispatch a rigid-path consumer over the family's residual width. Each arm
/// instantiates the generic body with a different kernel type; a width above
/// [`MAX_RESIDUAL_COLUMNS`] is refused at construction, never here.
macro_rules! with_residual_kernel {
    ($family:expr, $states:expr, |$kern:ident| $body:expr) => {{
        match $family.residual_dim() {
            1 => {
                let $kern = $crate::bms::residual_repair::BernoulliResidualRowKernel::<3>::new(
                    $family.clone(),
                    $states.to_vec(),
                )?;
                $body
            }
            2 => {
                let $kern = $crate::bms::residual_repair::BernoulliResidualRowKernel::<4>::new(
                    $family.clone(),
                    $states.to_vec(),
                )?;
                $body
            }
            3 => {
                let $kern = $crate::bms::residual_repair::BernoulliResidualRowKernel::<5>::new(
                    $family.clone(),
                    $states.to_vec(),
                )?;
                $body
            }
            4 => {
                let $kern = $crate::bms::residual_repair::BernoulliResidualRowKernel::<6>::new(
                    $family.clone(),
                    $states.to_vec(),
                )?;
                $body
            }
            5 => {
                let $kern = $crate::bms::residual_repair::BernoulliResidualRowKernel::<7>::new(
                    $family.clone(),
                    $states.to_vec(),
                )?;
                $body
            }
            6 => {
                let $kern = $crate::bms::residual_repair::BernoulliResidualRowKernel::<8>::new(
                    $family.clone(),
                    $states.to_vec(),
                )?;
                $body
            }
            7 => {
                let $kern = $crate::bms::residual_repair::BernoulliResidualRowKernel::<9>::new(
                    $family.clone(),
                    $states.to_vec(),
                )?;
                $body
            }
            8 => {
                let $kern = $crate::bms::residual_repair::BernoulliResidualRowKernel::<10>::new(
                    $family.clone(),
                    $states.to_vec(),
                )?;
                $body
            }
            9 => {
                let $kern = $crate::bms::residual_repair::BernoulliResidualRowKernel::<11>::new(
                    $family.clone(),
                    $states.to_vec(),
                )?;
                $body
            }
            10 => {
                let $kern = $crate::bms::residual_repair::BernoulliResidualRowKernel::<12>::new(
                    $family.clone(),
                    $states.to_vec(),
                )?;
                $body
            }
            11 => {
                let $kern = $crate::bms::residual_repair::BernoulliResidualRowKernel::<13>::new(
                    $family.clone(),
                    $states.to_vec(),
                )?;
                $body
            }
            12 => {
                let $kern = $crate::bms::residual_repair::BernoulliResidualRowKernel::<14>::new(
                    $family.clone(),
                    $states.to_vec(),
                )?;
                $body
            }
            width => Err(format!(
                "residual repair block width {width} is outside the instantiated kernel ladder \
                 1..={}",
                $crate::bms::residual_repair::MAX_RESIDUAL_COLUMNS
            )),
        }
    }};
}
pub(super) use with_residual_kernel;

impl BernoulliMarginalSlopeFamily {
    /// `K`, the residual block width; `0` without a block.
    #[inline]
    pub(super) fn residual_dim(&self) -> usize {
        self.residual.as_ref().map_or(0, |r| r.width())
    }

    /// Whether the family carries a residual block.
    #[inline]
    pub(super) fn residual_active(&self) -> bool {
        self.residual.is_some()
    }

    /// Index of the residual block in the parameter-block list.
    #[inline]
    pub(super) fn residual_block_index(&self) -> Option<usize> {
        self.residual.as_ref().map(|_| 2)
    }

    /// Block-diagonal working sets for the inner block-coordinate solver,
    /// from the residual kernel's generic joint assembly.
    pub(super) fn evaluate_residual_block_diagonals(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<FamilyEvaluation, String> {
        let (log_likelihood, gradient, hessian) = with_residual_kernel!(self, block_states, |kern| {
            let rows = crate::row_kernel::RowSet::All;
            let cache = crate::row_kernel::build_row_kernel_cache(&kern, &rows)?;
            let ll = crate::row_kernel::row_kernel_log_likelihood(&cache, &rows);
            let gradient = crate::row_kernel::row_kernel_gradient(&kern, &cache, &rows);
            let hessian = crate::row_kernel::row_kernel_hessian_dense(&kern, &cache, &rows)?;
            Ok::<_, String>((ll, gradient, hessian))
        })?;
        let slices = super::hessian_paths::block_slices(self);
        let residual = slices
            .residual
            .clone()
            .ok_or("residual block diagonals requested without a residual range")?;
        let block = |range: std::ops::Range<usize>| BlockWorkingSet::ExactNewton {
            gradient: Self::exact_newton_score_from_objective_gradient(
                gradient.slice(s![range.clone()]).to_owned(),
            ),
            hessian: SymmetricMatrix::Dense(hessian.slice(s![range.clone(), range]).to_owned()),
        };
        Ok(FamilyEvaluation {
            log_likelihood,
            blockworking_sets: vec![
                block(slices.marginal.clone()),
                block(slices.slope.clone()),
                block(residual),
            ],
        })
    }
}

#[cfg(test)]
mod residual_repair_kernel_tests {
    //! Finite-difference gates on the residual row program: the jet-derived
    //! gradient and Hessian in primary space against central differences of
    //! the value channel, and the plain-`f64` index derivatives against the
    //! same program. These pin the ONE statement of the anchor that the fit,
    //! the value-only line search and the prediction replay all read.

    use super::*;
    use gam_math::jet_scalar::Order2;
    use gam_math::jet_tower::RowProgram;
    use gam_math::nested_dual::JetField;

    fn covariance(k: usize, seed: u64) -> MarginalSlopeCovariance {
        // A random SPD matrix with unit (0,0) entry, the score's variance.
        let mut state = seed;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state % 10_000) as f64 / 10_000.0 - 0.5
        };
        let dim = k + 1;
        let mut l = Array2::<f64>::zeros((dim, dim));
        for i in 0..dim {
            for j in 0..=i {
                l[[i, j]] = if i == j { 0.6 + 0.4 * next().abs() } else { 0.5 * next() };
            }
        }
        let mut sigma = l.dot(&l.t());
        let scale = sigma[[0, 0]].sqrt();
        sigma.mapv_inplace(|v| v / (scale * scale));
        MarginalSlopeCovariance::full(sigma).expect("SPD covariance")
    }

    #[test]
    fn plain_index_derivatives_match_central_differences() {
        let link = InverseLink::Standard(StandardLink::Probit);
        for (k, eta_m, g, z, s) in [(1usize, 0.3, 0.4, -0.7, 1.0), (3, -0.8, -0.5, 1.2, 0.8)] {
            let cov = covariance(k, 11 + k as u64);
            let beta: Vec<f64> = (0..k).map(|j| 0.3 - 0.2 * j as f64).collect();
            let r: Vec<f64> = (0..k).map(|j| 0.5 * (j as f64 + 1.0) - 0.9).collect();
            let marginal = bernoulli_marginal_link_map(&link, eta_m).unwrap();
            let (_, d_q, d_g, d_beta) =
                residual_row_index(&marginal, g, &beta, z, &r, &cov, s).unwrap();
            let h = 1.0e-5;
            let value = |eta_m: f64, g: f64, beta: &[f64]| {
                let marginal = bernoulli_marginal_link_map(&link, eta_m).unwrap();
                residual_row_index(&marginal, g, beta, z, &r, &cov, s).unwrap().0
            };
            let fd_q = (value(eta_m + h, g, &beta) - value(eta_m - h, g, &beta)) / (2.0 * h);
            let fd_g = (value(eta_m, g + h, &beta) - value(eta_m, g - h, &beta)) / (2.0 * h);
            assert!((fd_q - d_q).abs() < 1.0e-7, "k={k}: ∂η/∂q {d_q} vs fd {fd_q}");
            assert!((fd_g - d_g).abs() < 1.0e-7, "k={k}: ∂η/∂g {d_g} vs fd {fd_g}");
            for j in 0..k {
                let mut plus = beta.clone();
                let mut minus = beta.clone();
                plus[j] += h;
                minus[j] -= h;
                let fd = (value(eta_m, g, &plus) - value(eta_m, g, &minus)) / (2.0 * h);
                assert!(
                    (fd - d_beta[j]).abs() < 1.0e-7,
                    "k={k}: ∂η/∂β_{j} {} vs fd {fd}",
                    d_beta[j]
                );
            }
        }
    }

    /// A standalone row program with the same body as the kernel's `eval`,
    /// so the jet lowering can be gated without a family fixture.
    struct StandaloneRow<const K: usize> {
        marginal: BernoulliMarginalLinkMap,
        z: f64,
        y: f64,
        w: f64,
        s: f64,
        r: Vec<f64>,
        cov: MarginalSlopeCovariance,
        base: [f64; K],
    }

    impl<const K: usize> gam_math::jet_tower::RowProgram<K> for StandaloneRow<K> {
        fn n_rows(&self) -> usize {
            1
        }
        fn primaries(&self, row: usize) -> Result<[f64; K], String> {
            if row != 0 {
                return Err(format!("standalone row program has one row, got {row}"));
            }
            Ok(self.base)
        }
        fn eval<S: JetScalar<K>>(&self, row: usize, p: &[S; K]) -> Result<S, String> {
            if row != 0 {
                return Err(format!("standalone row program has one row, got {row}"));
            }
            let q = p[0].compose_unary([
                self.marginal.q,
                self.marginal.q1,
                self.marginal.q2,
                self.marginal.q3,
                self.marginal.q4,
            ]);
            let mut drive = [S::constant(0.0); K];
            for k in 1..K {
                drive[k - 1] = p[k].scale(self.s);
            }
            let drive = &drive[..K - 1];
            let quad = S::symmetric_quadratic_form(drive, &self.cov);
            let c = quad.compose_unary(sqrt1p_stack((1.0 + quad.value()).sqrt()));
            let mut lw = [0.0_f64; K];
            lw[0] = self.z;
            for k in 0..K - 2 {
                lw[1 + k] = self.r[k];
            }
            let linear = S::linear_combination(drive, &lw[..K - 1]);
            let eta = q.multiply_add(&c, &linear);
            let margin = eta.scale(2.0 * self.y - 1.0);
            Ok(margin.compose_unary(signed_probit_neglog_unary_stack(margin.value(), self.w)))
        }
    }

    fn gate_jet_against_fd<const K: usize>(row: &StandaloneRow<K>) {
        let (nll, grad, hess) = gam_math::jet_tower::program_row_kernel(row, 0).unwrap();
        // The value at a displaced point: the marginal link stack is expanded
        // about the displaced η, exactly as the kernel rebuilds it from the
        // block state at every evaluation.
        let link = InverseLink::Standard(StandardLink::Probit);
        let value_at = |p: [f64; K]| -> f64 {
            let displaced = StandaloneRow::<K> {
                marginal: bernoulli_marginal_link_map(&link, p[0]).unwrap(),
                z: row.z,
                y: row.y,
                w: row.w,
                s: row.s,
                r: row.r.clone(),
                cov: row.cov.clone(),
                base: p,
            };
            let jets: [Order2<K>; K] =
                std::array::from_fn(|a| <Order2<K> as JetScalar<K>>::constant(p[a]));
            displaced.eval(0, &jets).unwrap().value()
        };
        assert!((value_at(row.base) - nll).abs() < 1.0e-12);
        let h = 1.0e-5;
        for a in 0..K {
            let mut plus = row.base;
            let mut minus = row.base;
            plus[a] += h;
            minus[a] -= h;
            let fd = (value_at(plus) - value_at(minus)) / (2.0 * h);
            assert!(
                (fd - grad[a]).abs() < 2.0e-7 * (1.0 + fd.abs()),
                "K={K}: grad[{a}] {} vs fd {fd}",
                grad[a]
            );
            for b in 0..K {
                let mut pp = row.base;
                let mut pm = row.base;
                let mut mp = row.base;
                let mut mm = row.base;
                pp[a] += h;
                pp[b] += h;
                pm[a] += h;
                pm[b] -= h;
                mp[a] -= h;
                mp[b] += h;
                mm[a] -= h;
                mm[b] -= h;
                let fd2 = (value_at(pp) - value_at(pm) - value_at(mp) + value_at(mm)) / (4.0 * h * h);
                assert!(
                    (fd2 - hess[a][b]).abs() < 5.0e-5 * (1.0 + fd2.abs()),
                    "K={K}: hess[{a}][{b}] {} vs fd {fd2}",
                    hess[a][b]
                );
            }
        }
        // The plain-f64 index derivatives and the jet gradient agree through
        // the chain rule dℓ/dp = ℓ'(margin)·sign·∂η/∂p.
        let beta: Vec<f64> = row.base[2..].to_vec();
        let (eta, d_q, d_g, d_beta) =
            residual_row_index(&row.marginal, row.base[1], &beta, row.z, &row.r, &row.cov, row.s)
                .unwrap();
        let sign = 2.0 * row.y - 1.0;
        let stack = signed_probit_neglog_unary_stack(sign * eta, row.w);
        let outer = stack[1] * sign;
        assert!((grad[0] - outer * d_q).abs() < 1.0e-10);
        assert!((grad[1] - outer * d_g).abs() < 1.0e-10);
        for j in 0..K - 2 {
            assert!((grad[2 + j] - outer * d_beta[j]).abs() < 1.0e-10);
        }
    }

    #[test]
    fn jet_gradient_and_hessian_match_central_differences() {
        let link = InverseLink::Standard(StandardLink::Probit);
        let row3 = StandaloneRow::<3> {
            marginal: bernoulli_marginal_link_map(&link, 0.4).unwrap(),
            z: -0.6,
            y: 1.0,
            w: 1.3,
            s: 0.9,
            r: vec![0.7],
            cov: covariance(1, 5),
            base: [0.4, 0.3, -0.5],
        };
        gate_jet_against_fd(&row3);
        let row5 = StandaloneRow::<5> {
            marginal: bernoulli_marginal_link_map(&link, -1.1).unwrap(),
            z: 1.4,
            y: 0.0,
            w: 0.8,
            s: 1.0,
            r: vec![0.2, -1.1, 0.6],
            cov: covariance(3, 7),
            base: [-1.1, -0.35, 0.25, 0.4, -0.15],
        };
        gate_jet_against_fd(&row5);
    }

    #[test]
    fn anchor_reduces_to_frailty_identity_when_residuals_are_independent_of_the_score() {
        // Σ = blockdiag(1, Σ_rr): c = √(1 + s²(g² + βᵀΣ_rrβ)), the gam#2766
        // identity with the residual variance entering like a frailty.
        let link = InverseLink::Standard(StandardLink::Probit);
        let sigma_rr = [[0.5, 0.1], [0.1, 0.3]];
        let mut sigma = Array2::<f64>::eye(3);
        for i in 0..2 {
            for j in 0..2 {
                sigma[[1 + i, 1 + j]] = sigma_rr[i][j];
            }
        }
        let cov = MarginalSlopeCovariance::full(sigma).unwrap();
        let (g, beta, s, z, r) = (0.45, [0.3, -0.2], 0.85, 0.2, [1.0, -0.5]);
        let marginal = bernoulli_marginal_link_map(&link, 0.7).unwrap();
        let (eta, _, _, _) = residual_row_index(&marginal, g, &beta, z, &r, &cov, s).unwrap();
        let kappa = beta[0] * beta[0] * sigma_rr[0][0]
            + 2.0 * beta[0] * beta[1] * sigma_rr[0][1]
            + beta[1] * beta[1] * sigma_rr[1][1];
        let c = (1.0 + s * s * (g * g + kappa)).sqrt();
        let expected = 0.7 * c + s * (g * z + beta[0] * r[0] + beta[1] * r[1]);
        assert!((eta - expected).abs() < 1.0e-13, "{eta} vs {expected}");
    }

    #[test]
    fn centring_gate_refuses_a_shifted_column_and_admits_a_centred_one() {
        let n = 4000usize;
        let mut state = 0x9e3779b97f4a7c15_u64;
        let mut uniform = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        let mut normal = || {
            let u1 = uniform().max(1.0e-12);
            let u2 = uniform();
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        };
        let a: Vec<f64> = (0..n).map(|_| normal()).collect();
        let z: Vec<f64> = (0..n).map(|_| normal()).collect();
        let centred: Vec<f64> = (0..n).map(|_| normal()).collect();
        let shifted: Vec<f64> = (0..n).map(|i| 0.5 * a[i] + 0.3 * normal()).collect();
        let a_block = Array2::from_shape_vec((n, 1), a.clone()).unwrap();
        let weights = Array1::from_elem(n, 1.0);
        let z = Array1::from(z);
        let ok = ResidualRepairSpec {
            columns: vec!["r".into()],
            features: Array2::from_shape_vec((n, 1), centred).unwrap(),
        };
        let runtime = ResidualBlockRuntime::fit(&ok, z.view(), weights.view(), a_block.view())
            .expect("a centred column is admitted");
        assert_eq!(runtime.width(), 1);
        assert_eq!(runtime.field.dim(), 2);
        let bad = ResidualRepairSpec {
            columns: vec!["r".into()],
            features: Array2::from_shape_vec((n, 1), shifted).unwrap(),
        };
        let err = ResidualBlockRuntime::fit(&bad, z.view(), weights.view(), a_block.view())
            .expect_err("a column with conditional mean structure is refused");
        assert!(
            matches!(err, ResidualRepairRefusal::ColumnNotCentred { .. }),
            "{err}"
        );
    }

    #[test]
    fn declared_score_variance_keeps_the_law_of_r_given_z_and_the_rigid_anchor() {
        let sample = covariance(3, 23).to_dense().mapv(|v| 1.7 * v);
        let declared = declare_unit_score_variance(&sample).unwrap();
        assert_eq!(declared[[0, 0]], 1.0);
        let schur = |m: &Array2<f64>| {
            Array2::from_shape_fn((3, 3), |(i, j)| {
                m[[i + 1, j + 1]] - m[[i + 1, 0]] * m[[0, j + 1]] / m[[0, 0]]
            })
        };
        let (s_sample, s_declared) = (schur(&sample), schur(&declared));
        for i in 0..3 {
            let gamma_sample = sample[[i + 1, 0]] / sample[[0, 0]];
            assert!((declared[[i + 1, 0]] - gamma_sample).abs() < 1.0e-14);
            for j in 0..3 {
                assert!((s_sample[[i, j]] - s_declared[[i, j]]).abs() < 1.0e-13);
            }
        }
        // β = 0: the anchor is the score-only √(1 + s²g²), whatever Var̂(z) was.
        let link = InverseLink::Standard(StandardLink::Probit);
        let marginal = bernoulli_marginal_link_map(&link, -0.4).unwrap();
        let cov = MarginalSlopeCovariance::full(declared).unwrap();
        let (g, s, z) = (0.8, 0.9, 1.3);
        let (eta, _, _, _) =
            residual_row_index(&marginal, g, &[0.0; 3], z, &[0.5, -0.2, 1.1], &cov, s).unwrap();
        let rigid = marginal.q * (1.0 + s * s * g * g).sqrt() + s * g * z;
        assert!((eta - rigid).abs() < 1.0e-14, "{eta} vs {rigid}");
    }
}
