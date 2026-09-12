//! Analytic penalty registry manifests.
//!
//! Add a primitive by implementing [`PenaltyManifest`] for its concrete
//! penalty type here and registering it in [`analytic_penalty_registry`].
//!
//! Inlined into `gam-terms` during the #1521 crate carve. This module was the
//! last `analytic_penalties` submodule still bridged to the old monolith via a
//! `#[path = "../../../../src/terms/analytic_penalties/manifest.rs"]` shim; when
//! `83dd4411c` deleted the monolith `src/terms/` tree the shim's target vanished
//! and the crate stopped compiling (the trait + every `impl` + the
//! `analytic_penalty_registry!` macro consumed by `registry.rs` live here). The
//! content is unchanged from the monolith version except the explicit
//! `crate::terms::analytic_penalties::{…}` import list is replaced by the
//! crate-idiomatic `use super::*;` (every sibling submodule pulls the shared
//! penalty types this way — see the `mod.rs` re-export block).

use super::*;

pub trait PenaltyManifest: AnalyticPenalty {
    const KIND_TAG: &'static str;
    const PYTHON_WRAPPER: &'static str;

    fn dispatch_tier(&self) -> PenaltyTier {
        self.tier()
    }
}

impl PenaltyManifest for ARDPenalty {
    const KIND_TAG: &'static str = "ard";
    const PYTHON_WRAPPER: &'static str = "ARDPenalty";
}

impl PenaltyManifest for BlockOrthogonalityPenalty {
    const KIND_TAG: &'static str = "block_orthogonality";
    const PYTHON_WRAPPER: &'static str = "BlockOrthogonalityPenalty";
}

impl PenaltyManifest for BlockSparsityPenalty {
    const KIND_TAG: &'static str = "block_sparsity";
    const PYTHON_WRAPPER: &'static str = "BlockSparsityPenalty";
}

impl PenaltyManifest for DecoderIncoherencePenalty {
    const KIND_TAG: &'static str = "decoder_incoherence";
    const PYTHON_WRAPPER: &'static str = "DecoderIncoherencePenalty";
}

impl PenaltyManifest for HarmonicRoughnessPenalty {
    const KIND_TAG: &'static str = "harmonic_roughness";
    const PYTHON_WRAPPER: &'static str = "HarmonicRoughnessPenalty";
}

impl PenaltyManifest for OrderedBetaBernoulliPenalty {
    const KIND_TAG: &'static str = "ordered_beta_bernoulli";
    const PYTHON_WRAPPER: &'static str = "OrderedBetaBernoulliPenalty";
}

impl PenaltyManifest for IsometryPenalty {
    const KIND_TAG: &'static str = "isometry";
    const PYTHON_WRAPPER: &'static str = "IsometryPenalty";
}

impl PenaltyManifest for IvaeRidgeMeanGauge {
    const KIND_TAG: &'static str = "ivae_ridge_mean_gauge";
    const PYTHON_WRAPPER: &'static str = "IvaeRidgeMeanGauge";
}

impl PenaltyManifest for SmoothThresholdPenalty {
    const KIND_TAG: &'static str = "smooth_threshold";
    const PYTHON_WRAPPER: &'static str = "SmoothThresholdPenalty";
}

impl PenaltyManifest for MechanismSparsityPenalty {
    const KIND_TAG: &'static str = "mechanism_sparsity";
    const PYTHON_WRAPPER: &'static str = "MechanismSparsityPenalty";
}

impl PenaltyManifest for ShapeMonotonicityPenalty {
    const KIND_TAG: &'static str = "monotonicity";
    const PYTHON_WRAPPER: &'static str = "MonotonicityPenalty";
}

impl PenaltyManifest for NestedPrefixPenalty {
    const KIND_TAG: &'static str = "nested_prefix";
    const PYTHON_WRAPPER: &'static str = "NestedPrefixPenalty";
}

impl PenaltyManifest for NuclearNormPenalty {
    const KIND_TAG: &'static str = "nuclear_norm";
    const PYTHON_WRAPPER: &'static str = "NuclearNormPenalty";
}

impl PenaltyManifest for OrthogonalityPenalty {
    const KIND_TAG: &'static str = "orthogonality";
    const PYTHON_WRAPPER: &'static str = "OrthogonalityPenalty";
}

impl PenaltyManifest for ParametricRowPrecisionPriorPenalty {
    const KIND_TAG: &'static str = "parametric_row_precision_prior";
    const PYTHON_WRAPPER: &'static str = "ParametricAuxConditionalPriorPenalty";
}

impl PenaltyManifest for RowPrecisionPriorPenalty {
    const KIND_TAG: &'static str = "row_precision_prior";
    const PYTHON_WRAPPER: &'static str = "AuxConditionalPriorPenalty";
}

impl PenaltyManifest for ScadMcpPenalty {
    const KIND_TAG: &'static str = "scad_mcp";
    const PYTHON_WRAPPER: &'static str = "ScadMcpPenalty";
}

impl PenaltyManifest for SheafConsistencyPenalty {
    const KIND_TAG: &'static str = "sheaf_consistency";
    const PYTHON_WRAPPER: &'static str = "SheafConsistencyPenalty";
}

impl PenaltyManifest for SoftmaxAssignmentSparsityPenalty {
    const KIND_TAG: &'static str = "softmax_assignment_sparsity";
    const PYTHON_WRAPPER: &'static str = "SoftmaxAssignmentSparsityPenalty";
}

impl PenaltyManifest for SparsityPenalty {
    const KIND_TAG: &'static str = "sparsity";
    const PYTHON_WRAPPER: &'static str = "SparsityPenalty";
}

impl PenaltyManifest for TopKActivationPenalty {
    const KIND_TAG: &'static str = "topk_activation";
    const PYTHON_WRAPPER: &'static str = "TopKActivationPenalty";
}

impl PenaltyManifest for TotalVariationPenalty {
    const KIND_TAG: &'static str = "total_variation";
    const PYTHON_WRAPPER: &'static str = "TotalVariationPenalty";
}

#[macro_export]
macro_rules! analytic_penalty_registry {
    ($macro:ident) => {
        $macro! {
            register!(Isometry, IsometryPenalty);
            register!(Sparsity, SparsityPenalty);
            register!(SoftmaxAssignmentSparsity, SoftmaxAssignmentSparsityPenalty);
            register!(OrderedBetaBernoulli, OrderedBetaBernoulliPenalty);
            register!(Ard, ARDPenalty);
            register!(TopKActivation, TopKActivationPenalty);
            register!(SmoothThreshold, SmoothThresholdPenalty);
            register!(TotalVariation, TotalVariationPenalty);
            register!(HarmonicRoughness, HarmonicRoughnessPenalty);
            register!(NuclearNorm, NuclearNormPenalty);
            register!(BlockSparsity, BlockSparsityPenalty);
            register!(MechanismSparsity, MechanismSparsityPenalty);
            register!(Monotonicity, ShapeMonotonicityPenalty);
            register!(NestedPrefix, NestedPrefixPenalty);
            register!(RowPrecisionPrior, RowPrecisionPriorPenalty);
            register!(IvaeRidgeMeanGauge, IvaeRidgeMeanGauge);
            register!(ParametricRowPrecisionPrior, ParametricRowPrecisionPriorPenalty);
            register!(ScadMcp, ScadMcpPenalty);
            register!(BlockOrthogonality, BlockOrthogonalityPenalty);
            register!(DecoderIncoherence, DecoderIncoherencePenalty);
            register!(Orthogonality, OrthogonalityPenalty);
            register!(SheafConsistency, SheafConsistencyPenalty);
        }
    };
}
