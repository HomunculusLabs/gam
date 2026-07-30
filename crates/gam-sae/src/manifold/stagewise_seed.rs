//! Typed construction of the single-atom seed consumed by stagewise SAE.

use gam_terms::analytic_penalties::AnalyticPenaltyRegistry;
use ndarray::{ArrayView2, ArrayView3, ArrayView4};

use super::*;

pub struct SaeStagewiseSeedRequest<'a> {
    pub target: ArrayView2<'a, f64>,
    pub geometry_plans: &'a [SaeAtomGeometryPlan],
    pub basis_values: ArrayView3<'a, f64>,
    pub basis_jacobian: ArrayView4<'a, f64>,
    pub decoder_coefficients: ArrayView3<'a, f64>,
    pub smooth_penalties: ArrayView3<'a, f64>,
    pub initial_logits: ArrayView2<'a, f64>,
    pub initial_coords: ArrayView3<'a, f64>,
    pub alpha: f64,
    pub tau: f64,
    pub learnable_alpha: bool,
    pub assignment_kind: SaeFitAssignmentKind,
    pub sparsity_strength: f64,
    pub smoothness: f64,
    pub max_iter: usize,
    pub learning_rate: f64,
    pub ridge_ext_coord: f64,
    pub ridge_beta: f64,
    pub structured_whitening: bool,
    pub fisher_metric: Option<SaeFisherRowMetricRequest<'a>>,
}

pub struct SaeStagewiseSeedReport {
    pub base_term: SaeManifoldTerm,
    pub initial_rho: SaeManifoldRho,
}

