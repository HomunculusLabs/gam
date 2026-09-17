use super::*;

/// Cap on the number of coordinates at which a per-atom shape band is
/// materialized. The full per-atom decoder covariance is exact and exposed
/// regardless; this only bounds the cost of the convenience band, which is
/// evaluated at an evenly-strided subset of the atom's own on-atom coordinates.
pub const SHAPE_BAND_MAX_POINTS: usize = 512;

/// Entry budget for materializing one atom's dense `(M_k·p)²` decoder
/// covariance in the fit payload. Above it (LLM-scale ambient `p`) the band
/// quantities are computed exactly from the factored frame covariance and the
/// dense export is omitted (`decoder_covariance: None`) — the python reader
/// treats it as optional. 2^24 f64 entries = 128 MiB per atom.
pub const SAE_DECODER_COV_PAYLOAD_MAX_ENTRIES: usize = 1 << 24;

/// Posterior uncertainty of one fitted atom's manifold shape.
///
/// In the primary path — [`SaeManifoldTerm::assemble_shape_uncertainty`], and
/// after a structure/finalization change its final-state twin
/// [`SaeManifoldTerm::recompute_joint_shape_uncertainty`] — the covariance is
/// the φ-scaled β-block of the pseudo-inverse of the JOINT exact observed
/// information `A` (coordinates and the other atoms marginalized out), on the
/// identified space the exact-A criterion prices; see
/// [`SaeObservedInformationCovariance`]. The band is its closed-form
/// push-forward through the linear basis→ambient map `m_k(t) = Φ_k(t)·B_k`.
/// This is what the production fit returns.
///
/// When no such covariance exists — a streaming plan that cannot materialize
/// `A`, or a state where `A` has resolved negative curvature — every band field
/// is `None` and [`SaeShapeUncertainty::operator`] names the reason. Neither a
/// per-atom marginal nor the majorizer's inverse is substituted.
#[derive(Debug, Clone)]
pub struct SaeAtomShapeUncertainty {
    /// φ-scaled posterior covariance of this atom's decoder coefficients,
    /// `Cov(β_k) = φ·[A⁺]_ββ[block_k]`, shape `(M_k·p, M_k·p)` in the decoder's
    /// row-major `(basis, channel)` flat layout (flat index `b·p + c`).
    ///
    /// `None` when materializing it would exceed
    /// [`SAE_DECODER_COV_PAYLOAD_MAX_ENTRIES`] (LLM-scale ambient `p`: at
    /// `(M=8, p=2048)` the dense block is 2 GiB *per atom*, at
    /// `(M=16, p=5120)` ~50 GiB). The band quantities below are still exact
    /// in that case — they are computed directly from the factored
    /// `(M_k·r_k)²` frame covariance without ever lifting it.
    pub decoder_covariance: Option<Array2<f64>>,
    /// Coordinates at which the band is evaluated, shape `(G, d_k)`.
    pub band_coords: Option<Array2<f64>>,
    /// Fitted ambient point `m_k(t) = Φ_k(t)·B_k` at each band coordinate,
    /// shape `(G, p)`.
    pub band_mean: Option<Array2<f64>>,
    /// Posterior standard deviation of each ambient channel at each band
    /// coordinate, `sqrt(Var_c(t))` with
    /// `Var_c(t) = Σ_{b1,b2} Φ[b1] Φ[b2] Cov(β_k)[(b1,c),(b2,c)]`, shape
    /// `(G, p)`.
    ///
    /// This is the MODEL-BASED band (`Cov = φ̂ A⁺`), correct only when the
    /// working reconstruction likelihood is correctly specified. See
    /// [`Self::band_sd_robust`] for the misspecification-robust companion.
    pub band_sd: Option<Array2<f64>>,
    /// Sandwich (Godambe / robust) posterior standard deviation of each ambient
    /// channel, with the same shape as the model-based band, computed from the within-channel
    /// sandwich covariance `A_c⁻¹ J_cc A_c⁻¹` (see `super::sandwich`). Reported
    /// ALONGSIDE the model-based band, never in place of it; the two coincide
    /// when the information-matrix equality holds and diverge exactly to the
    /// degree the residuals violate the working likelihood (heteroskedastic
    /// tokens, LayerNorm structure, cross-channel template correlation).
    ///
    /// `None` when the robust band was not requested for this atom or could not
    /// be formed (e.g. the huge-`p` factored path where the dense within-channel
    /// meat is not materialized) — an honest gap, not a fabricated band.
    pub band_sd_robust: Option<Array2<f64>>,
}

/// The frame in which the reconstruction likelihood measures its residuals,
/// which fixes the units of [`SaeReconstructionDispersion::likelihood_dispersion`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SaeLikelihoodFrame {
    /// Isotropic data fit `½‖r_n‖²` (no metric, or a gauge-only metric): the
    /// dispersion is in squared output units over `n·p` scalar observations.
    RawOutput,
    /// Whitened data fit `½ r_nᵀ M_n r_n` with `M_n = U_n U_nᵀ` of rank
    /// `metric_rank`: the dispersion is dimensionless over `n·metric_rank`
    /// whitened scalar observations.
    Whitened { metric_rank: usize },
}

/// The two reconstruction noise scales of a fit, which answer different
/// questions and need not share units or values.
///
/// With an inverse-noise metric `M_n = Σ_n⁻¹` the joint Hessian
/// `H = JᵀMJ + P` already carries `Σ`'s variance scale: for `M = σ⁻²I` and a
/// linear decoder `H⁻¹ = σ²(XᵀX + σ²λS)⁻¹`. Multiplying that inverse by a raw
/// residual variance `≈ σ²` produces `σ⁴(…)⁻¹`, so the covariance multiplier is
/// the dimensionless metric-frame dispersion, never the raw one.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SaeReconstructionDispersion {
    /// Raw output-frame noise variance per scalar observation,
    /// `Σ_{i,c} r_{ic}² / (n·p − EDF)`, in squared output units whatever
    /// metric the likelihood uses. Consumers that compare against unwhitened
    /// output quantities read this: the Marchenko–Pastur rank edge (against the
    /// raw decoder Gram), the incoherence SNR and the per-atom inner fits.
    pub raw_output_noise_variance: f64,
    /// Dispersion `φ̂` of the working likelihood `exp(−½ Σ w_n r_nᵀ M_n r_n / φ)`,
    /// `2·data_fit / (n_likelihood − EDF)` in the frame named by
    /// [`Self::likelihood_frame`]. Equal to [`Self::raw_output_noise_variance`]
    /// when that frame is [`SaeLikelihoodFrame::RawOutput`].
    pub likelihood_dispersion: f64,
    /// Units and scalar-observation count of [`Self::likelihood_dispersion`].
    pub likelihood_frame: SaeLikelihoodFrame,
}

impl SaeReconstructionDispersion {
    /// Multiplier turning the metric-weighted inverse joint Hessian `H⁻¹` into
    /// the posterior covariance, `Cov = φ̂·H⁻¹`.
    ///
    /// The inner objective is `½ Σ w_n r_nᵀ M_n r_n + P(θ; λ)` with no `φ`, so
    /// each fitted penalty strength is the product `λ = φ·λ′` of a prior
    /// precision `λ′` and the likelihood dispersion. The posterior
    /// `exp(−½ rᵀMr/φ − P(θ; λ′))` then has Hessian `H/φ` and covariance exactly
    /// `φ·H⁻¹`: the prior scales with the dispersion. Holding `λ′ = λ` fixed
    /// instead would give `(JᵀMJ/φ + P″)⁻¹`, which is not a multiple of `H⁻¹`.
    pub fn posterior_covariance_scale(&self) -> f64 {
        self.likelihood_dispersion
    }
}

/// Selected inverse blocks of the exact observed information on its identified
/// space, one per atom (#2933 F33).
///
/// `A = ∇²_θθ L` is the Hessian of the penalized inner objective in the joint
/// `(t, β)` cache layout: Gauss–Newton curvature, the residual curvature
/// `Σ_{i,c} (WMr)_{ic} ∇²f_{ic}`, and the exact prior curvature. It is not the
/// positive majorizer `B` the inner Newton solve factors, whose Schur inverse
/// this replaced: with `B = 2` and a residual-curvature correction of `−1`,
/// `B⁻¹ = 0.5` while the observed-information variance is `A⁻¹ = 1`, and no
/// accuracy in solving `B` turns one into the other.
///
/// Block `k` is `[A⁺]_ββ[r_k, r_k]` over atom `k`'s border range `r_k`: every
/// latent coordinate and every other atom is marginalized through the whole
/// joint operator. `A⁺` is the pseudo-inverse over the directions the exact-A
/// value path retains (`λᵢ > rank_floor(i)`), so the covariance lives on the
/// same identified space the criterion prices. The blocks are unscaled;
/// [`SaeReconstructionDispersion::posterior_covariance_scale`] multiplies them
/// at assembly.
#[derive(Debug, Clone)]
pub struct SaeObservedInformationCovariance {
    /// `[A⁺]_ββ[r_k, r_k]` per atom, in the cache's border layout (the
    /// factored `(M_k·r_k)` block when frames are active).
    pub(crate) blocks: Vec<Array2<f64>>,
    /// Number of retained directions of `A`.
    pub(crate) identified_rank: usize,
    /// Dimension of the joint `(t, β)` operator.
    pub(crate) ambient_dim: usize,
}

/// The information a shape covariance is assembled from, or why none exists.
#[derive(Debug, Clone)]
pub enum SaeShapeInformation {
    /// Selected inverse blocks of the exact observed information.
    ObservedInformation(SaeObservedInformationCovariance),
    /// No covariance exists at this state or on this execution plan.
    Unavailable(SaeShapeCovarianceUnavailable),
}

/// Why a fit reports no shape covariance.
#[derive(Debug, Clone, PartialEq)]
pub enum SaeShapeCovarianceUnavailable {
    /// The admitted execution plan does not materialize the dense joint
    /// observed information, so no selected inverse of `A` is formed.
    NoDenseObservedInformation,
    /// `A` has resolved negative curvature: the state is not a mode of the
    /// penalized objective, so `A⁺` is not a Laplace covariance. The value path
    /// may still price such a basin through the majorizer's concave clamp, but
    /// that clamped operator has no probabilistic contract as a covariance.
    IndefiniteObservedInformation {
        negative_directions: usize,
        most_negative_curvature: f64,
    },
}

/// The operator a reported shape covariance inverts.
#[derive(Debug, Clone, PartialEq)]
pub enum SaeShapeCovarianceOperator {
    /// `φ̂·[A⁺]_ββ` on the identified space of the exact observed information.
    ObservedInformation {
        identified_rank: usize,
        ambient_dim: usize,
    },
    /// No covariance was produced; every band field is `None`.
    Unavailable(SaeShapeCovarianceUnavailable),
}

impl SaeShapeCovarianceOperator {
    /// Wire name of the operator, owned here so bindings marshal rather than map.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::ObservedInformation { .. } => "observed_information",
            Self::Unavailable(SaeShapeCovarianceUnavailable::NoDenseObservedInformation) => {
                "unavailable_no_dense_observed_information"
            }
            Self::Unavailable(
                SaeShapeCovarianceUnavailable::IndefiniteObservedInformation { .. },
            ) => "unavailable_indefinite_observed_information",
        }
    }
}

/// Posterior shape uncertainty for a whole SAE-manifold fit: one band per atom
/// plus the reconstruction noise scales, of which
/// [`SaeReconstructionDispersion::posterior_covariance_scale`] scales every
/// covariance. See [`SaeManifoldTerm::assemble_shape_uncertainty`].
#[derive(Debug, Clone)]
pub struct SaeShapeUncertainty {
    /// Raw and likelihood-frame reconstruction noise scales.
    pub dispersion: SaeReconstructionDispersion,
    /// The operator every atom's covariance inverts, or why none was produced.
    pub operator: SaeShapeCovarianceOperator,
    /// One entry per atom, in atom order.
    pub atoms: Vec<SaeAtomShapeUncertainty>,
}
