//! The event-history custom family: every subject's exact-marginal likelihood
//! assembled into the coefficient space of the per-mark smooth blocks and the
//! latent block, with every derivative the outer LAML evaluator asks for
//! produced by the same generic code under a directional dual scalar.

use super::chain::GaussHermite;
use super::cohort::{CohortNodes, EventHistoryCohort, EventHistoryError, expand_nodes};
use super::marginal::{SubjectInputs, subject_marginal};
use crate::custom_family::{
    BlockWorkingSet, BlockwiseFitOptions, CustomFamily, ExactNewtonJointGradientEvaluation,
    FamilyEvaluation, ParameterBlockSpec, ParameterBlockState, PenaltyMatrix,
    fit_custom_family,
};
use gam_linalg::matrix::{DenseDesignMatrix, DesignMatrix, SymmetricMatrix};
use gam_math::jet_scalar::{JetScalar, OneSeed, Order2, TwoSeed};
use gam_math::nested_dual::JetField;
use gam_model_api::families::custom_family::joint_coupled_coefficient_hessian_cost;
use gam_problem::CoefficientCoordinate;
use gam_solve::model_types::UnifiedFitResult;
use gam_terms::smooth::{TermCollectionDesign, TermCollectionSpec, build_term_collection_design};
use ndarray::{Array1, Array2};
use rayon::prelude::*;
use std::sync::{Arc, Mutex};

/// A scalar that can be seeded with up to two directions and read back.
pub(crate) trait Directional: JetField + Send + Sync {
    fn seeded(value: f64, u: f64, v: f64) -> Self;
    fn eps(&self) -> f64;
    fn eps_del(&self) -> f64;
}

#[inline]
fn validate_seed_directions(directions: [f64; 2]) {
    assert!(
        directions.into_iter().all(f64::is_finite),
        "event-history derivative seed directions must be finite",
    );
}

impl Directional for f64 {
    fn seeded(value: f64, u: f64, v: f64) -> Self {
        validate_seed_directions([u, v]);
        value
    }
    fn eps(&self) -> f64 {
        0.0
    }
    fn eps_del(&self) -> f64 {
        0.0
    }
}

fn scalar0(value: f64) -> Order2<0> {
    <Order2<0> as JetScalar<0>>::constant(value)
}

impl Directional for OneSeed<0> {
    fn seeded(value: f64, u: f64, v: f64) -> Self {
        validate_seed_directions([u, v]);
        OneSeed {
            base: scalar0(value),
            eps: scalar0(u),
        }
    }
    fn eps(&self) -> f64 {
        self.eps.value()
    }
    fn eps_del(&self) -> f64 {
        0.0
    }
}

impl Directional for TwoSeed<0> {
    fn seeded(value: f64, u: f64, v: f64) -> Self {
        TwoSeed {
            base: scalar0(value),
            eps: scalar0(u),
            del: scalar0(v),
            eps_del: scalar0(0.0),
        }
    }
    fn eps(&self) -> f64 {
        self.eps.value()
    }
    fn eps_del(&self) -> f64 {
        self.eps_del.value()
    }
}

/// A one-direction dual at `value` with `ε`-component `u`.
pub fn seeded_one(value: f64, u: f64) -> OneSeed<0> {
    OneSeed::seeded(value, u, 0.0)
}

/// A two-direction dual at `value` with `ε`-component `u` and `δ`-component `v`.
pub fn seeded_two(value: f64, u: f64, v: f64) -> TwoSeed<0> {
    TwoSeed::seeded(value, u, v)
}

fn is_zero<S: Directional>(x: &S) -> bool {
    x.value() == 0.0 && x.eps() == 0.0 && x.eps_del() == 0.0
}

/// A full joint evaluation in coefficient space. `hessian` is the negative
/// log-likelihood Hessian, the convention the custom-family engine expects.
#[derive(Clone, Debug)]
pub struct JointEvaluation {
    pub log_likelihood: f64,
    pub gradient: Array1<f64>,
    pub hessian: Array2<f64>,
}

/// The event-history family over a node-expanded cohort.
#[derive(Clone)]
pub struct EventHistoryFamily {
    nodes: Arc<CohortNodes>,
    /// Dense per-mark designs on the nodes, `n_obs × p_d`.
    designs: Vec<Arc<Array2<f64>>>,
    atoms: usize,
    gh: Arc<GaussHermite>,
    time_scale: f64,
    cache: Arc<Mutex<Option<(u64, Arc<JointEvaluation>)>>>,
}

impl EventHistoryFamily {
    pub fn new(
        nodes: Arc<CohortNodes>,
        designs: Vec<Arc<Array2<f64>>>,
        atoms: usize,
        gauss_hermite_order: usize,
        time_scale: f64,
    ) -> Result<Self, EventHistoryError> {
        if designs.len() != nodes.marks {
            return Err(EventHistoryError::InvalidInput {
                reason: format!(
                    "event-history family needs one design per mark: got {} designs for {} marks",
                    designs.len(),
                    nodes.marks
                ),
            });
        }
        for (d, design) in designs.iter().enumerate() {
            if design.nrows() != nodes.total_nodes {
                return Err(EventHistoryError::InvalidInput {
                    reason: format!(
                        "design for mark {d} has {} rows but the cohort has {} nodes",
                        design.nrows(),
                        nodes.total_nodes
                    ),
                });
            }
        }
        if !(time_scale.is_finite() && time_scale > 0.0) {
            return Err(EventHistoryError::InvalidInput {
                reason: "time scale must be finite and positive".to_string(),
            });
        }
        Ok(Self {
            nodes,
            designs,
            atoms,
            gh: Arc::new(GaussHermite::new(gauss_hermite_order)?),
            time_scale,
            cache: Arc::new(Mutex::new(None)),
        })
    }

    pub fn marks(&self) -> usize {
        self.nodes.marks
    }

    pub fn atoms(&self) -> usize {
        self.atoms
    }

    pub fn time_scale(&self) -> f64 {
        self.time_scale
    }

    pub fn gauss_hermite_order(&self) -> usize {
        self.gh.order
    }

    pub fn nodes(&self) -> &Arc<CohortNodes> {
        &self.nodes
    }

    pub(crate) fn gauss_hermite(&self) -> &Arc<GaussHermite> {
        &self.gh
    }

    /// The same family with a different Gauss-Hermite order.
    pub fn with_gauss_hermite_order(&self, order: usize) -> Result<Self, EventHistoryError> {
        Ok(Self {
            nodes: Arc::clone(&self.nodes),
            designs: self.designs.clone(),
            atoms: self.atoms,
            gh: Arc::new(GaussHermite::new(order)?),
            time_scale: self.time_scale,
            cache: Arc::new(Mutex::new(None)),
        })
    }

    /// Width of the latent block: loadings then log-rates.
    pub fn latent_width(&self) -> usize {
        self.marks() * self.atoms + self.atoms
    }

    fn block_widths(&self) -> Vec<usize> {
        let mut widths: Vec<usize> = self.designs.iter().map(|d| d.ncols()).collect();
        widths.push(self.latent_width());
        widths
    }

    fn block_offsets(&self) -> Vec<usize> {
        let mut offsets = Vec::with_capacity(self.marks() + 2);
        let mut acc = 0;
        for w in self.block_widths() {
            offsets.push(acc);
            acc += w;
        }
        offsets.push(acc);
        offsets
    }

    fn total_width(&self) -> usize {
        self.block_widths().iter().sum()
    }

    fn validate_states(&self, states: &[ParameterBlockState]) -> Result<(), String> {
        let marks = self.marks();
        if states.len() != marks + 1 {
            return Err(format!(
                "event-history family expects {} blocks (one per mark plus the latent block), got {}",
                marks + 1,
                states.len()
            ));
        }
        for (d, state) in states.iter().take(marks).enumerate() {
            if state.eta.len() != self.nodes.total_nodes {
                return Err(format!(
                    "mark {d} predictor has length {}, expected {} nodes",
                    state.eta.len(),
                    self.nodes.total_nodes
                ));
            }
            if state.beta.len() != self.designs[d].ncols() {
                return Err(format!(
                    "mark {d} has {} coefficients, expected {}",
                    state.beta.len(),
                    self.designs[d].ncols()
                ));
            }
        }
        if states[marks].beta.len() != self.latent_width() {
            return Err(format!(
                "latent block has {} coefficients, expected {}",
                states[marks].beta.len(),
                self.latent_width()
            ));
        }
        Ok(())
    }

    fn state_key(states: &[ParameterBlockState]) -> u64 {
        let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
        for state in states {
            for v in state.beta.iter().chain(state.eta.iter()) {
                hash ^= v.to_bits();
                hash = hash.wrapping_mul(0x0100_0000_01b3);
            }
        }
        hash
    }

    /// Value, gradient and Hessian (of the log-likelihood) in coefficient
    /// space, generic in the scalar so a seeded scalar yields directional
    /// derivatives.
    fn evaluate_generic<S: Directional>(
        &self,
        states: &[ParameterBlockState],
        u: Option<&Array1<f64>>,
        v: Option<&Array1<f64>>,
        derivatives: bool,
    ) -> Result<(S, Vec<S>, Vec<S>), String> {
        self.validate_states(states)?;
        let marks = self.marks();
        let atoms = self.atoms;
        let offsets = self.block_offsets();
        let total = self.total_width();
        let latent_offset = offsets[marks];
        for direction in [u, v].into_iter().flatten() {
            if direction.len() != total {
                return Err(format!(
                    "event-history direction has length {}, expected {total}",
                    direction.len()
                ));
            }
        }
        let component = |dir: Option<&Array1<f64>>, index: usize| -> f64 {
            dir.map_or(0.0, |d| d[index])
        };
        let latent_beta = &states[marks].beta;
        let loadings: Vec<S> = (0..marks * atoms)
            .map(|q| {
                S::seeded(
                    latent_beta[q],
                    component(u, latent_offset + q),
                    component(v, latent_offset + q),
                )
            })
            .collect();
        let log_rates: Vec<S> = (0..atoms)
            .map(|k| {
                let q = marks * atoms + k;
                S::seeded(
                    latent_beta[q],
                    component(u, latent_offset + q),
                    component(v, latent_offset + q),
                )
            })
            .collect();
        let designs = &self.designs;
        let subjects = &self.nodes.subjects;
        let gh = &self.gh;
        let time_scale = self.time_scale;
        let row_direction = |dir: Option<&Array1<f64>>, d: usize, row: usize| -> f64 {
            dir.map_or(0.0, |dir| {
                let design = &designs[d];
                let mut acc = 0.0;
                for (j, x) in design.row(row).iter().enumerate() {
                    acc += x * dir[offsets[d] + j];
                }
                acc
            })
        };
        let per_subject: Result<Vec<(S, Vec<S>, Vec<S>)>, String> = subjects
            .par_iter()
            .map(|subject| {
                let n = subject.len();
                let first = subject.first_row;
                let mut eta0 = Vec::with_capacity(n * marks);
                for node in 0..n {
                    let row = first + node;
                    for d in 0..marks {
                        eta0.push(S::seeded(
                            states[d].eta[row],
                            row_direction(u, d, row),
                            row_direction(v, d, row),
                        ));
                    }
                }
                let inputs = SubjectInputs {
                    nodes: subject,
                    eta0: &eta0,
                    loadings: &loadings,
                    log_rates: &log_rates,
                    time_scale,
                    gh,
                    continuation_gap: 0.0,
                };
                let local = subject_marginal(&inputs, derivatives).map_err(|e| e.to_string())?;
                if !derivatives {
                    return Ok((local.loglik, Vec::new(), Vec::new()));
                }
                let zero = local.loglik.constant_like(0.0);
                let latent_local = n * marks;
                let p_local = latent_local + marks * atoms + atoms;
                let local_index = |node: usize, d: usize| node * marks + d;
                let mut gradient = vec![zero.clone(); total];
                for d in 0..marks {
                    let design = &designs[d];
                    let width = design.ncols();
                    for node in 0..n {
                        let g = &local.gradient[local_index(node, d)];
                        if is_zero(g) {
                            continue;
                        }
                        let row = design.row(first + node);
                        for j in 0..width {
                            let x = row[j];
                            if x != 0.0 {
                                gradient[offsets[d] + j] = gradient[offsets[d] + j].add(&g.scale(x));
                            }
                        }
                    }
                }
                for q in 0..marks * atoms + atoms {
                    gradient[latent_offset + q] = local.gradient[latent_local + q].clone();
                }
                let mut hessian = vec![zero.clone(); total * total];
                let h = |i: usize, j: usize| -> &S { &local.hessian[i * p_local + j] };
                for d in 0..marks {
                    let design_d = &designs[d];
                    let width_d = design_d.ncols();
                    for d2 in d..marks {
                        let design_2 = &designs[d2];
                        let width_2 = design_2.ncols();
                        let mut w = vec![zero.clone(); n * width_2];
                        for node in 0..n {
                            for m in 0..n {
                                let hv = h(local_index(node, d), local_index(m, d2));
                                if is_zero(hv) {
                                    continue;
                                }
                                let row = design_2.row(first + m);
                                for j in 0..width_2 {
                                    let x = row[j];
                                    if x != 0.0 {
                                        w[node * width_2 + j] = w[node * width_2 + j].add(&hv.scale(x));
                                    }
                                }
                            }
                        }
                        for node in 0..n {
                            let row = design_d.row(first + node);
                            for i in 0..width_d {
                                let x = row[i];
                                if x == 0.0 {
                                    continue;
                                }
                                for j in 0..width_2 {
                                    let idx = (offsets[d] + i) * total + offsets[d2] + j;
                                    hessian[idx] = hessian[idx].add(&w[node * width_2 + j].scale(x));
                                }
                            }
                        }
                    }
                    for node in 0..n {
                        let row = design_d.row(first + node);
                        for q in 0..marks * atoms + atoms {
                            let hv = h(local_index(node, d), latent_local + q);
                            if is_zero(hv) {
                                continue;
                            }
                            for i in 0..width_d {
                                let x = row[i];
                                if x != 0.0 {
                                    let idx = (offsets[d] + i) * total + latent_offset + q;
                                    hessian[idx] = hessian[idx].add(&hv.scale(x));
                                }
                            }
                        }
                    }
                }
                for q in 0..marks * atoms + atoms {
                    for q2 in q..marks * atoms + atoms {
                        let idx = (latent_offset + q) * total + latent_offset + q2;
                        hessian[idx] = h(latent_local + q, latent_local + q2).clone();
                    }
                }
                Ok((local.loglik, gradient, hessian))
            })
            .collect();
        let per_subject = per_subject?;
        let zero = per_subject
            .first()
            .map(|(l, _, _)| l.constant_like(0.0))
            .ok_or_else(|| "event-history family has no subjects".to_string())?;
        let mut loglik = zero.clone();
        let mut gradient = vec![zero.clone(); if derivatives { total } else { 0 }];
        let mut hessian = vec![zero.clone(); if derivatives { total * total } else { 0 }];
        for (l, g, hh) in &per_subject {
            loglik = loglik.add(l);
            if derivatives {
                for (acc, x) in gradient.iter_mut().zip(g.iter()) {
                    *acc = acc.add(x);
                }
                for (acc, x) in hessian.iter_mut().zip(hh.iter()) {
                    *acc = acc.add(x);
                }
            }
        }
        if derivatives {
            for i in 0..total {
                for j in (i + 1)..total {
                    hessian[j * total + i] = hessian[i * total + j].clone();
                }
            }
        }
        Ok((loglik, gradient, hessian))
    }

    /// Full `f64` joint evaluation, cached on the state.
    pub fn joint_evaluation(
        &self,
        states: &[ParameterBlockState],
    ) -> Result<Arc<JointEvaluation>, String> {
        let key = Self::state_key(states);
        if let Ok(guard) = self.cache.lock()
            && let Some((k, value)) = guard.as_ref()
            && *k == key
        {
            return Ok(Arc::clone(value));
        }
        let (loglik, gradient, hessian) = self.evaluate_generic::<f64>(states, None, None, true)?;
        let total = self.total_width();
        let mut negative_hessian = Array2::<f64>::zeros((total, total));
        for i in 0..total {
            for j in 0..total {
                negative_hessian[[i, j]] = -hessian[i * total + j];
            }
        }
        let evaluation = Arc::new(JointEvaluation {
            log_likelihood: loglik,
            gradient: Array1::from(gradient),
            hessian: negative_hessian,
        });
        if let Ok(mut guard) = self.cache.lock() {
            *guard = Some((key, Arc::clone(&evaluation)));
        }
        Ok(evaluation)
    }

    /// Log-likelihood only (forward filter, no derivatives).
    pub fn log_likelihood(&self, states: &[ParameterBlockState]) -> Result<f64, String> {
        let (loglik, _, _) = self.evaluate_generic::<f64>(states, None, None, false)?;
        Ok(loglik)
    }

    /// `D_β H[u]` for the negative log-likelihood Hessian `H`.
    pub fn directional_hessian(
        &self,
        states: &[ParameterBlockState],
        u: &Array1<f64>,
    ) -> Result<Array2<f64>, String> {
        let total = self.total_width();
        let (_, _, hessian) = self.evaluate_generic::<OneSeed<0>>(states, Some(u), None, true)?;
        let mut out = Array2::<f64>::zeros((total, total));
        for i in 0..total {
            for j in 0..total {
                out[[i, j]] = -hessian[i * total + j].eps();
            }
        }
        Ok(out)
    }

    /// `D²_β H[u, v]` for the negative log-likelihood Hessian `H`.
    pub fn second_directional_hessian(
        &self,
        states: &[ParameterBlockState],
        u: &Array1<f64>,
        v: &Array1<f64>,
    ) -> Result<Array2<f64>, String> {
        let total = self.total_width();
        let (_, _, hessian) =
            self.evaluate_generic::<TwoSeed<0>>(states, Some(u), Some(v), true)?;
        let mut out = Array2::<f64>::zeros((total, total));
        for i in 0..total {
            for j in 0..total {
                out[[i, j]] = -hessian[i * total + j].eps_del();
            }
        }
        Ok(out)
    }
}

impl CustomFamily for EventHistoryFamily {
    fn evaluate(&self, block_states: &[ParameterBlockState]) -> Result<FamilyEvaluation, String> {
        let joint = self.joint_evaluation(block_states)?;
        let offsets = self.block_offsets();
        let mut blockworking_sets = Vec::with_capacity(offsets.len() - 1);
        for b in 0..offsets.len() - 1 {
            let range = offsets[b]..offsets[b + 1];
            let gradient = joint.gradient.slice(ndarray::s![range.clone()]).to_owned();
            let hessian = joint
                .hessian
                .slice(ndarray::s![range.clone(), range])
                .to_owned();
            blockworking_sets.push(BlockWorkingSet::ExactNewton {
                gradient,
                hessian: SymmetricMatrix::Dense(hessian),
            });
        }
        Ok(FamilyEvaluation {
            log_likelihood: joint.log_likelihood,
            blockworking_sets,
        })
    }

    fn log_likelihood_only(&self, block_states: &[ParameterBlockState]) -> Result<f64, String> {
        self.log_likelihood(block_states)
    }

    fn classical_deviance(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<Option<f64>, String> {
        self.validate_states(block_states)?;
        Ok(None)
    }

    fn exact_newton_joint_hessian_beta_dependent(&self) -> bool {
        true
    }

    fn has_explicit_joint_hessian(&self) -> bool {
        true
    }

    fn requires_joint_outer_hyper_path(&self) -> bool {
        true
    }

    fn inner_coefficient_objective_is_globally_convex(&self) -> bool {
        false
    }

    fn coefficient_hessian_cost(&self, specs: &[ParameterBlockSpec]) -> u64 {
        joint_coupled_coefficient_hessian_cost(self.nodes.total_nodes as u64, specs)
    }

    fn output_channel_assignment(&self, specs: &[ParameterBlockSpec]) -> Option<Vec<usize>> {
        Some((0..specs.len()).collect())
    }

    fn block_coefficient_coordinate(
        &self,
        block_states: &[ParameterBlockState],
        block_index: usize,
        block_spec: &ParameterBlockSpec,
    ) -> CoefficientCoordinate {
        // The family owns the chain rule from node predictors to coefficients
        // through its stored designs, so no block may be reparameterised.
        if block_index >= block_states.len() || block_spec.name.is_empty() {
            return CoefficientCoordinate::Structural;
        }
        CoefficientCoordinate::Structural
    }

    fn exact_newton_joint_hessian(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<Option<Array2<f64>>, String> {
        Ok(Some(self.joint_evaluation(block_states)?.hessian.clone()))
    }

    fn exact_newton_joint_loglik_gradient(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<Option<Array1<f64>>, String> {
        Ok(Some(self.joint_evaluation(block_states)?.gradient.clone()))
    }

    fn exact_newton_joint_gradient_evaluation(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
    ) -> Result<Option<ExactNewtonJointGradientEvaluation>, String> {
        if specs.len() != block_states.len() {
            return Err(format!(
                "event-history joint gradient: {} specs for {} block states",
                specs.len(),
                block_states.len()
            ));
        }
        let joint = self.joint_evaluation(block_states)?;
        Ok(Some(ExactNewtonJointGradientEvaluation {
            log_likelihood: joint.log_likelihood,
            gradient: joint.gradient.clone(),
        }))
    }

    fn exact_newton_joint_hessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        Ok(Some(self.directional_hessian(block_states, d_beta_flat)?))
    }

    fn exact_newton_joint_hessiansecond_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_u_flat: &Array1<f64>,
        d_betav_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        Ok(Some(self.second_directional_hessian(
            block_states,
            d_beta_u_flat,
            d_betav_flat,
        )?))
    }
}

/// Fit specification for an event-history model.
#[derive(Clone)]
pub struct EventHistorySpec {
    /// Number of latent Ornstein–Uhlenbeck atoms: the maximum. Each carries
    /// its own REML ridge and is switched off by the evidence when unneeded.
    pub atoms: usize,
    /// One covariate/time term collection per mark, or a single one shared by
    /// every mark. Feature columns index the node data matrix: the covariate
    /// table's columns followed by the node time.
    pub covariates: Vec<TermCollectionSpec>,
    /// Gauss-Legendre order per follow-up segment.
    pub quadrature_order: usize,
    /// Starting Gauss-Hermite order per latent axis; doubled until the fitted
    /// marginal log-likelihood is stable to `quadrature_tolerance`.
    pub gauss_hermite_order: usize,
    /// Relative stability tolerance on the fitted marginal log-likelihood
    /// between successive Gauss-Hermite orders.
    pub quadrature_tolerance: f64,
    pub options: BlockwiseFitOptions,
}

impl EventHistorySpec {
    pub fn new(atoms: usize, covariates: Vec<TermCollectionSpec>) -> Self {
        Self {
            atoms,
            covariates,
            quadrature_order: super::cohort::quadrature_order_for_degree(3),
            gauss_hermite_order: 9,
            quadrature_tolerance: 1e-6,
            options: BlockwiseFitOptions::default(),
        }
    }
}

/// Certificate that the Gauss-Hermite order resolved the fitted marginal.
#[derive(Clone, Debug)]
pub struct QuadratureCertificate {
    pub order: usize,
    pub checked_order: usize,
    pub log_likelihood: f64,
    pub log_likelihood_at_checked_order: f64,
}

/// A fitted event-history model.
pub struct EventHistoryFit {
    pub nodes: Arc<CohortNodes>,
    pub family: EventHistoryFamily,
    pub fit: UnifiedFitResult,
    /// Frozen per-mark term collections for prediction.
    pub frozen_specs: Vec<TermCollectionSpec>,
    pub designs: Vec<TermCollectionDesign>,
    /// Loadings `a[d, k]`.
    pub loadings: Array2<f64>,
    /// `ln(rate_k · time_scale)`.
    pub log_rates: Vec<f64>,
    /// Rates in the data's time unit.
    pub rates: Vec<f64>,
    /// REML log smoothing parameter of each atom's ridge.
    pub atom_log_lambdas: Vec<f64>,
    pub time_scale: f64,
    /// Gauss-Legendre order per follow-up segment used for the training nodes.
    pub quadrature_order: usize,
    pub quadrature: QuadratureCertificate,
}

impl EventHistoryFit {
    pub fn marks(&self) -> usize {
        self.nodes.marks
    }

    pub fn atoms(&self) -> usize {
        self.log_rates.len()
    }

    /// Fitted coefficients of the mark-`d` block.
    pub fn mark_coefficients(&self, d: usize) -> &Array1<f64> {
        &self.fit.block_states[d].beta
    }

    /// Node log-intensity offsets `η⁰` for mark `d` on the training nodes.
    pub fn mark_eta(&self, d: usize) -> &Array1<f64> {
        &self.fit.block_states[d].eta
    }
}

fn identity_pattern_design(n_obs: usize, width: usize) -> Result<Array2<f64>, EventHistoryError> {
    if n_obs < width {
        return Err(EventHistoryError::InvalidInput {
            reason: format!(
                "the cohort has {n_obs} nodes but the latent block needs at least {width}"
            ),
        });
    }
    let mut design = Array2::<f64>::zeros((n_obs, width));
    for i in 0..width {
        design[[i, i]] = 1.0;
    }
    Ok(design)
}

/// The latent block: loadings then log-rates, one ridge per atom covering
/// that atom's loadings and its log-rate, so an atom the evidence does not
/// support is pinned by its own penalty instead of leaving an unidentified
/// coordinate in the Laplace integral.
pub fn latent_block_spec(
    n_obs: usize,
    marks: usize,
    atoms: usize,
) -> Result<ParameterBlockSpec, EventHistoryError> {
    let width = marks * atoms + atoms;
    let design = identity_pattern_design(n_obs, width.max(1))?;
    let mut penalties = Vec::with_capacity(atoms);
    let mut nullspace_dims = Vec::with_capacity(atoms);
    for k in 0..atoms {
        let mut s = Array2::<f64>::zeros((width, width));
        for d in 0..marks {
            s[[d * atoms + k, d * atoms + k]] = 1.0;
        }
        s[[marks * atoms + k, marks * atoms + k]] = 1.0;
        penalties.push(PenaltyMatrix::Dense(s));
        nullspace_dims.push(width - marks - 1);
    }
    Ok(ParameterBlockSpec {
        name: "latent".to_string(),
        design: DesignMatrix::Dense(DenseDesignMatrix::from(Arc::new(design))),
        offset: Array1::zeros(n_obs),
        penalties,
        nullspace_dims,
        initial_log_lambdas: Array1::zeros(atoms),
        initial_beta: Some(Array1::zeros(width)),
        gauge_priority: 100,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    })
}

/// The block of one mark's covariate/time smooths.
pub fn mark_block_spec(name: &str, design: &TermCollectionDesign) -> ParameterBlockSpec {
    ParameterBlockSpec {
        name: name.to_string(),
        design: design.design.clone(),
        offset: design.affine_offset.clone(),
        penalties: design.penalties_as_penalty_matrix(),
        nullspace_dims: design.nullspace_dims.clone(),
        initial_log_lambdas: Array1::zeros(design.penalties.len()),
        initial_beta: None,
        gauge_priority: 150,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    }
}

/// Fit an event-history model to a cohort.
pub fn fit_event_history(
    cohort: &mut EventHistoryCohort,
    spec: &EventHistorySpec,
) -> Result<EventHistoryFit, EventHistoryError> {
    cohort.validate()?;
    let marks = cohort.marks();
    if spec.covariates.len() != 1 && spec.covariates.len() != marks {
        return Err(EventHistoryError::InvalidInput {
            reason: format!(
                "expected one covariate term collection or one per mark ({marks}), got {}",
                spec.covariates.len()
            ),
        });
    }
    if !(spec.quadrature_tolerance.is_finite() && spec.quadrature_tolerance > 0.0) {
        return Err(EventHistoryError::InvalidInput {
            reason: "quadrature tolerance must be finite and positive".to_string(),
        });
    }
    let nodes = Arc::new(expand_nodes(cohort, spec.quadrature_order)?);
    let time_scale = cohort.time_scale();
    let mut designs = Vec::with_capacity(marks);
    let mut dense = Vec::with_capacity(marks);
    let mut frozen_specs = Vec::with_capacity(marks);
    let mut block_specs = Vec::with_capacity(marks + 1);
    for d in 0..marks {
        let term_spec = if spec.covariates.len() == 1 {
            &spec.covariates[0]
        } else {
            &spec.covariates[d]
        };
        let design = build_term_collection_design(nodes.node_data.view(), term_spec).map_err(
            |error| EventHistoryError::Fit {
                reason: format!("design for mark {d}: {error}"),
            },
        )?;
        let frozen =
            crate::fit_orchestration::drivers::freeze_term_collection_from_design(term_spec, &design)
                .map_err(|error| EventHistoryError::Fit {
                    reason: format!("freezing mark {d} term collection: {error}"),
                })?;
        let dense_design = design
            .design
            .try_to_dense_arc("event-history mark design")
            .map_err(|error| EventHistoryError::Fit {
                reason: error.to_string(),
            })?;
        block_specs.push(mark_block_spec(&cohort.mark_names[d], &design));
        designs.push(design);
        dense.push(dense_design);
        frozen_specs.push(frozen);
    }
    block_specs.push(latent_block_spec(nodes.total_nodes, marks, spec.atoms)?);

    let mut order = spec.gauss_hermite_order.max(3);
    let mut family = EventHistoryFamily::new(
        Arc::clone(&nodes),
        dense.clone(),
        spec.atoms,
        order,
        time_scale,
    )?;
    let mut specs = block_specs;
    loop {
        let fit = fit_custom_family(&family, &specs, &spec.options).map_err(|error| {
            EventHistoryError::Fit {
                reason: format!("event-history LAML fit at Gauss-Hermite order {order}: {error}"),
            }
        })?;
        let value = family
            .log_likelihood(&fit.block_states)
            .map_err(|reason| EventHistoryError::Fit { reason })?;
        let checked_order = 2 * order - 1;
        let checked_family = family.with_gauss_hermite_order(checked_order)?;
        let checked = checked_family
            .log_likelihood(&fit.block_states)
            .map_err(|reason| EventHistoryError::Fit { reason })?;
        let stable = (value - checked).abs() <= spec.quadrature_tolerance * value.abs().max(1.0);
        if stable || spec.atoms == 0 {
            let latent = &fit.block_states[marks].beta;
            let mut loadings = Array2::<f64>::zeros((marks, spec.atoms));
            for d in 0..marks {
                for k in 0..spec.atoms {
                    loadings[[d, k]] = latent[d * spec.atoms + k];
                }
            }
            let log_rates: Vec<f64> = (0..spec.atoms)
                .map(|k| latent[marks * spec.atoms + k])
                .collect();
            let rates: Vec<f64> = log_rates.iter().map(|r| r.exp() / time_scale).collect();
            let n_lambda = fit.log_lambdas.len();
            let atom_log_lambdas: Vec<f64> = fit
                .log_lambdas
                .iter()
                .skip(n_lambda.saturating_sub(spec.atoms))
                .copied()
                .collect();
            return Ok(EventHistoryFit {
                nodes,
                family,
                fit,
                frozen_specs,
                designs,
                loadings,
                log_rates,
                rates,
                atom_log_lambdas,
                time_scale,
                quadrature_order: spec.quadrature_order,
                quadrature: QuadratureCertificate {
                    order,
                    checked_order,
                    log_likelihood: value,
                    log_likelihood_at_checked_order: checked,
                },
            });
        }
        if checked_order > 129 {
            return Err(EventHistoryError::NumericalFailure {
                reason: format!(
                    "the marginal log-likelihood did not stabilise across Gauss-Hermite orders (order {order}: {value}, order {checked_order}: {checked})"
                ),
            });
        }
        log::info!(
            "[event-history] Gauss-Hermite order {order} not stable ({value} vs {checked} at order {checked_order}); refitting"
        );
        order = checked_order;
        family = checked_family;
        for (block, state) in specs.iter_mut().zip(fit.block_states.iter()) {
            block.initial_beta = Some(state.beta.clone());
        }
    }
}
