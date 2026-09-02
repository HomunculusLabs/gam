//! The event-history custom family: every subject's marginal likelihood
//! assembled into the coefficient space of the per-mark smooth blocks and the
//! latent block, with every derivative the outer LAML evaluator asks for
//! produced by the same generic code under a directional dual scalar, and
//! the fit driver that refines the Gauss-Hermite order and the time mesh
//! until the fitted coefficients are stationary under refinement.

use super::chain::{GaussHermite, product_grid_size};
use super::cohort::{
    CohortNodes, EventHistoryCohort, EventHistoryError, MarkKind, design_rows, expand_nodes,
};
use super::covariance::{NewAtom, SubjectResiduals, best_new_atom};
use super::marginal::{
    LOST_POSITIVITY, SubjectInputs, expected_intensities, forward_filter, pairwise_sum,
    subject_marginal,
};
use super::scalar::Tangent;
use crate::custom_family::{
    BlockWorkingSet, BlockwiseFitOptions, CustomFamily, ExactNewtonJointGradientEvaluation,
    FamilyEvaluation, ParameterBlockSpec, ParameterBlockState, PenaltyMatrix, fit_custom_family,
};
use gam_linalg::matrix::{DenseDesignMatrix, DesignMatrix, SymmetricMatrix};
use gam_math::jet_scalar::{JetScalar, OneSeed, Order2, TwoSeed};
use gam_math::nested_dual::JetField;
use gam_model_api::families::custom_family::joint_coupled_coefficient_hessian_cost;
use gam_problem::CoefficientCoordinate;
use gam_solve::model_types::UnifiedFitResult;
use gam_terms::smooth::{
    TermCollectionDesign, TermCollectionSpec, build_term_collection_design,
    freeze_term_collection_from_design,
};
use ndarray::{Array1, Array2, ArrayView2, s};
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
    /// The last joint evaluation, keyed on the exact state it was made at.
    cache: Arc<Mutex<Option<(Vec<f64>, Arc<JointEvaluation>)>>>,
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
        product_grid_size(gauss_hermite_order, atoms)?;
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

    /// Width of the latent block: loadings then log-rates.
    pub fn latent_width(&self) -> usize {
        self.marks() * self.atoms + self.atoms
    }

    /// Whether the fit carries a latent block (no atoms means a plain
    /// Poisson-process GAM with the same node expansion).
    pub fn has_latent_block(&self) -> bool {
        self.atoms > 0
    }

    fn block_widths(&self) -> Vec<usize> {
        let mut widths: Vec<usize> = self.designs.iter().map(|d| d.ncols()).collect();
        if self.has_latent_block() {
            widths.push(self.latent_width());
        }
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

    /// Total coefficient count: every mark block then the latent block.
    pub fn total_width(&self) -> usize {
        self.block_widths().iter().sum()
    }

    fn validate_states(&self, states: &[ParameterBlockState]) -> Result<(), String> {
        let marks = self.marks();
        let expected = marks + usize::from(self.has_latent_block());
        if states.len() != expected {
            return Err(format!(
                "event-history family expects {expected} blocks (one per mark{}), got {}",
                if self.has_latent_block() {
                    " plus the latent block"
                } else {
                    ""
                },
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
        if self.has_latent_block() && states[marks].beta.len() != self.latent_width() {
            return Err(format!(
                "latent block has {} coefficients, expected {}",
                states[marks].beta.len(),
                self.latent_width()
            ));
        }
        Ok(())
    }

    /// The exact state a joint evaluation is keyed on: every coefficient and
    /// every node predictor, bit for bit. A hash alone would turn a collision
    /// into a silently wrong likelihood.
    fn state_key(states: &[ParameterBlockState]) -> Vec<f64> {
        let mut key = Vec::new();
        for state in states {
            key.extend(state.beta.iter().copied());
            key.extend(state.eta.iter().copied());
        }
        key
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
        let empty = Array1::<f64>::zeros(0);
        let latent_beta: &Array1<f64> = if self.has_latent_block() {
            &states[marks].beta
        } else {
            &empty
        };
        for direction in [u, v].into_iter().flatten() {
            if direction.len() != total {
                return Err(format!(
                    "event-history direction has length {}, expected {total}",
                    direction.len()
                ));
            }
            if let Some((index, value)) = direction
                .iter()
                .copied()
                .enumerate()
                .find(|(_, value)| !value.is_finite())
            {
                return Err(format!(
                    "event-history direction contains non-finite value {value} at coefficient {index}",
                ));
            }
        }
        let component = |dir: Option<&Array1<f64>>, index: usize| -> f64 {
            dir.map_or(0.0, |d| d[index])
        };
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
                let views: Vec<ArrayView2<'_, f64>> = designs
                    .iter()
                    .map(|design| design.slice(s![first..first + n, ..]))
                    .collect();
                let inputs = SubjectInputs {
                    nodes: subject,
                    eta0: &eta0,
                    loadings: &loadings,
                    log_rates: &log_rates,
                    time_scale,
                    gh,
                    continuation_gap: 0.0,
                    designs: derivatives.then_some(views.as_slice()),
                };
                let local = subject_marginal(&inputs, derivatives).map_err(|e| e.to_string())?;
                if derivatives && (local.gradient.len() != total || local.hessian.len() != total * total) {
                    return Err(format!(
                        "subject {} produced {} gradient entries for {total} coefficients",
                        first,
                        local.gradient.len()
                    ));
                }
                Ok((local.loglik, local.gradient, local.hessian))
            })
            .collect();
        let per_subject = per_subject?;
        let zero = per_subject
            .first()
            .map(|(l, _, _)| l.constant_like(0.0))
            .ok_or_else(|| "event-history family has no subjects".to_string())?;
        let subject_logliks: Vec<S> = per_subject.iter().map(|(l, _, _)| l.clone()).collect();
        let loglik = pairwise_sum(&subject_logliks, &zero);
        let mut gradient = vec![zero.clone(); if derivatives { total } else { 0 }];
        let mut hessian = vec![zero.clone(); if derivatives { total * total } else { 0 }];
        if derivatives {
            for (_, g, hh) in &per_subject {
                for (acc, x) in gradient.iter_mut().zip(g.iter()) {
                    *acc = acc.add(x);
                }
                for (acc, x) in hessian.iter_mut().zip(hh.iter()) {
                    *acc = acc.add(x);
                }
            }
        }
        Ok((loglik, gradient, hessian))
    }

    /// Exact gradient of the computed log-likelihood in coefficient space.
    ///
    /// The forward filter is replayed on forward-mode duals seeded with the
    /// design rows, so every tangent slot is the derivative of the very
    /// arithmetic that produced the value. The Fisher-identity gradient of the
    /// exact marginal differs from this by the quadrature error, and a
    /// trust-region Newton with a value-based acceptance test cannot converge
    /// on a gradient that is not the derivative of the value it tests.
    fn exact_gradient(&self, states: &[ParameterBlockState]) -> Result<Vec<f64>, String> {
        let total = self.total_width();
        let mut gradient = vec![0.0; total];
        if total <= 4 {
            self.exact_gradient_chunks::<4>(states, &mut gradient)?;
        } else if total <= 8 {
            self.exact_gradient_chunks::<8>(states, &mut gradient)?;
        } else if total <= 16 {
            self.exact_gradient_chunks::<16>(states, &mut gradient)?;
        } else if total <= 32 {
            self.exact_gradient_chunks::<32>(states, &mut gradient)?;
        } else {
            self.exact_gradient_chunks::<64>(states, &mut gradient)?;
        }
        Ok(gradient)
    }

    /// One `W`-wide sweep of tangent slots at a time over the coefficient
    /// vector, each sweep a full forward filter per subject.
    fn exact_gradient_chunks<const W: usize>(
        &self,
        states: &[ParameterBlockState],
        gradient: &mut [f64],
    ) -> Result<(), String> {
        let marks = self.marks();
        let atoms = self.atoms;
        let offsets = self.block_offsets();
        let total = self.total_width();
        let latent_offset = offsets[marks];
        let empty = Array1::<f64>::zeros(0);
        let latent_beta: &Array1<f64> = if self.has_latent_block() {
            &states[marks].beta
        } else {
            &empty
        };
        let designs = &self.designs;
        let gh = &self.gh;
        let time_scale = self.time_scale;
        for start in (0..total).step_by(W) {
            let end = (start + W).min(total);
            let seed = |slot: usize, grad: &mut [f64; W], x: f64| {
                if (start..end).contains(&slot) {
                    grad[slot - start] = x;
                }
            };
            let unit = |q: usize| -> Tangent<W> {
                let mut grad = [0.0; W];
                seed(latent_offset + q, &mut grad, 1.0);
                Tangent::seeded(latent_beta[q], grad)
            };
            let loadings: Vec<Tangent<W>> = (0..marks * atoms).map(unit).collect();
            let log_rates: Vec<Tangent<W>> = (0..atoms).map(|k| unit(marks * atoms + k)).collect();
            let per_subject: Result<Vec<[f64; W]>, String> = self
                .nodes
                .subjects
                .par_iter()
                .map(|subject| {
                    let n = subject.len();
                    let first = subject.first_row;
                    let mut eta0 = Vec::with_capacity(n * marks);
                    for node in 0..n {
                        let row = first + node;
                        for d in 0..marks {
                            let mut grad = [0.0; W];
                            for (j, x) in designs[d].row(row).iter().enumerate() {
                                seed(offsets[d] + j, &mut grad, *x);
                            }
                            eta0.push(Tangent::seeded(states[d].eta[row], grad));
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
                        designs: None,
                    };
                    let local = subject_marginal(&inputs, false).map_err(|e| e.to_string())?;
                    Ok(local.loglik.grad)
                })
                .collect();
            for g in per_subject? {
                for (slot, x) in g.iter().enumerate().take(end - start) {
                    gradient[start + slot] += x;
                }
            }
        }
        Ok(())
    }

    /// The martingale residuals of every subject at `states`: per node and
    /// mark the compensated score `s = y − w E[λ | past]` and its curvature
    /// `c = w E[λ | past]`, both under the FILTERED density, which is what
    /// makes them the increments of a martingale — uncorrelated under the
    /// null, with `Σ_n c_n` as the exact predictable variation the
    /// covariance score subtracts.
    pub(crate) fn residuals(
        &self,
        states: &[ParameterBlockState],
    ) -> Result<Vec<SubjectResiduals>, String> {
        self.validate_states(states)?;
        let marks = self.marks();
        let atoms = self.atoms;
        let empty = Array1::<f64>::zeros(0);
        let latent: &Array1<f64> = if self.has_latent_block() {
            &states[marks].beta
        } else {
            &empty
        };
        let loadings: Vec<f64> = latent.iter().take(marks * atoms).copied().collect();
        let log_rates: Vec<f64> = latent.iter().skip(marks * atoms).copied().collect();
        let gh = &self.gh;
        let time_scale = self.time_scale;
        let all_marks = vec![true; marks];
        self.nodes
            .subjects
            .par_iter()
            .map(|subject| {
                let n = subject.len();
                let first = subject.first_row;
                let mut eta0 = Vec::with_capacity(n * marks);
                for node in 0..n {
                    for d in 0..marks {
                        eta0.push(states[d].eta[first + node]);
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
                    designs: None,
                };
                let pass = forward_filter(&inputs, None, &all_marks).map_err(|e| e.to_string())?;
                let mut scores = Vec::with_capacity(n * marks);
                let mut curvatures = Vec::with_capacity(n * marks);
                for node in 0..n {
                    let intensities = expected_intensities(
                        &pass.grids[node],
                        &pass.predicted[node],
                        &eta0[node * marks..(node + 1) * marks],
                        &loadings,
                        marks,
                        atoms,
                    );
                    for d in 0..marks {
                        let c = subject.exposures[[node, d]] * intensities[d];
                        scores.push(subject.counts[[node, d]] - c);
                        curvatures.push(c);
                    }
                }
                Ok(SubjectResiduals {
                    times: subject.times.clone(),
                    scores,
                    curvatures,
                })
            })
            .collect()
    }

    /// Full `f64` joint evaluation, cached on the state: the value, its exact
    /// gradient, and the Louis-identity Hessian.
    ///
    /// The Hessian is Louis' identity evaluated by the same quadrature as the
    /// value; it agrees with the second derivative of the computed value to
    /// the quadrature error the fit's certificate bounds. The gradient is the
    /// exact derivative of the computed value. A Newton iteration with an
    /// exact gradient and a Hessian accurate to a small relative error
    /// converges at that relative rate, and the outer LAML's log-determinant
    /// term sees the same Hessian its directional derivatives are taken of.
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
        let (loglik, _, hessian) = self.evaluate_generic::<f64>(states, None, None, true)?;
        let gradient = self.exact_gradient(states)?;
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
            let gradient = joint.gradient.slice(s![range.clone()]).to_owned();
            let hessian = joint
                .hessian
                .slice(s![range.clone(), range])
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

    /// The marginal likelihood is not log-concave in the loadings: at zero
    /// loading the profile can be locally convex (a saddle of the negative
    /// log-likelihood), so the joint Newton needs the self-vanishing
    /// Levenberg–Marquardt damping the sibling latent families use.
    fn levenberg_on_ill_conditioning(&self) -> bool {
        true
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
    /// One covariate/time term collection per mark, or a single one shared by
    /// every mark. Feature columns index the node data matrix: the covariate
    /// table's columns followed by the node time.
    pub covariates: Vec<TermCollectionSpec>,
    /// Gauss-Legendre order per mesh cell.
    pub quadrature_order: usize,
    /// Starting Gauss-Hermite order per latent axis.
    pub gauss_hermite_order: usize,
    /// The certificate's tolerance: the largest shift any fitted coefficient
    /// may make under a refinement of the Gauss-Hermite order or of the time
    /// mesh, in units of that coefficient's posterior standard deviation.
    /// The order is doubled and the mesh halved until both shifts are below
    /// it, so a fit is never an artefact of its discretisation at a level
    /// the data could resolve. The default, a twentieth of a posterior
    /// standard deviation, is the scale at which a shift cannot change any
    /// inference the fit supports: it is a twentieth of the width the data
    /// itself leaves undetermined.
    pub quadrature_tolerance: f64,
    pub options: BlockwiseFitOptions,
}

impl EventHistorySpec {
    pub fn new(covariates: Vec<TermCollectionSpec>) -> Self {
        Self {
            covariates,
            quadrature_order: super::cohort::quadrature_order_for_degree(3),
            gauss_hermite_order: 9,
            quadrature_tolerance: 5e-2,
            options: BlockwiseFitOptions::default(),
        }
    }
}

/// One stationarity check of the fitted coefficients under a refinement.
///
/// The shift is the exact first-order move of the penalised mode: at the
/// fitted coefficients the current setting is stationary, so the refined
/// setting's mode sits at `β + V (g' − g)` with `g'` and `g` the two
/// settings' exact gradients at the same `β` and `V` the fit's own posterior
/// covariance — the inverse of the penalised Hessian, which is the operator
/// that turns a gradient discrepancy into a coefficient one. Measuring it
/// this way costs one gradient per candidate instead of a second fit, and it
/// is reported in the units the shift has to be judged in: posterior
/// standard deviations.
#[derive(Clone, Debug)]
pub struct RefinementCheck {
    /// The refined setting that was checked (a Gauss-Hermite order or a mesh
    /// refinement level).
    pub candidate: usize,
    /// The largest coefficient shift under the refinement, in posterior
    /// standard deviations.
    pub coefficient_shift: f64,
    /// The log-likelihood at the fitted coefficients under the refinement.
    pub log_likelihood: f64,
}

/// Certificate that the fitted coefficients are stationary under a
/// refinement of the latent quadrature and of the time mesh.
#[derive(Clone, Debug)]
pub struct QuadratureCertificate {
    pub gauss_hermite_order: usize,
    pub mesh_refinement: usize,
    pub log_likelihood: f64,
    /// The check against the next Gauss-Hermite order (`2·order − 1`).
    pub gauss_hermite: RefinementCheck,
    /// The check against the mesh with every cell halved.
    pub mesh: RefinementCheck,
}

/// A fitted event-history model.
pub struct EventHistoryFit {
    pub nodes: Arc<CohortNodes>,
    pub family: EventHistoryFamily,
    pub fit: UnifiedFitResult,
    pub mark_kinds: Vec<MarkKind>,
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
    /// Gauss-Legendre order per mesh cell used for the training nodes.
    pub quadrature_order: usize,
    /// Mesh refinement level of the training nodes.
    pub mesh_refinement: usize,
    pub quadrature: QuadratureCertificate,
    /// Every rank step the evidence judged, in order.
    pub rank_path: Vec<RankStep>,
    /// The decrease of the outer LAML criterion each accepted atom brought.
    pub atom_evidence: Vec<f64>,
}

impl EventHistoryFit {
    /// The rank of the latent covariance the evidence supports.
    pub fn rank(&self) -> usize {
        self.log_rates.len()
    }

    /// `C(0) = A Aᵀ`: the covariance of the marks' latent log-intensity
    /// deviations at one time. This is the reported latent object; the
    /// loadings are its factor coordinates, which two atoms of equal rate
    /// could rotate without changing the process.
    pub fn disease_covariance(&self) -> Array2<f64> {
        super::covariance::disease_covariance(&self.loadings)
    }

    /// `C(Δ)`: the same covariance across a lag of `lag` time units.
    pub fn temporal_covariance(&self, lag: f64) -> Array2<f64> {
        super::covariance::temporal_covariance(&self.loadings, &self.rates, lag)
    }

    /// Eigenvalues (descending) and eigenvectors (columns) of `C(0)`.
    pub fn eigenmodes(&self) -> Result<(Array1<f64>, Array2<f64>), EventHistoryError> {
        super::covariance::eigenmodes(&self.disease_covariance())
    }

    pub fn marks(&self) -> usize {
        self.nodes.marks
    }

    /// The number of atoms the fit was offered (which of them the data use
    /// is read from `loadings` and `atom_log_lambdas`).
    pub fn atoms(&self) -> usize {
        self.log_rates.len()
    }

    /// Fitted coefficients of the mark-`d` block: the population
    /// log-intensity surface's coefficients (see [`Self::mark_eta`]).
    pub fn mark_coefficients(&self, d: usize) -> &Array1<f64> {
        &self.fit.block_states[d].beta
    }

    /// Population log-intensity `η⁰` of mark `d` on the training nodes:
    /// `exp(η⁰)` is the intensity averaged over the latent state, since the
    /// latent term enters as `−½|a_d|² + a_d · z`, whose Gaussian mixing the
    /// shift cancels exactly (`docs/event-history.md` derives it).
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
/// coordinate in the Laplace integral. The ridge on the log-rate is an
/// empirical-Bayes prior centred on the cohort's own time scale whose
/// strength REML learns per atom.
///
/// Gauge: the likelihood is invariant under flipping the sign of an atom's
/// loadings (with its state) and under permuting atoms; with equal rates it
/// is also invariant under rotating the atoms jointly. The initial point
/// fixes the gauge: every loading starts at `+1` (one latent standard
/// deviation per unit of log-intensity, the parameterisation's natural
/// scale) and the log-rates start apart, each atom's memory half its
/// predecessor's, centred on the ridge centre — a permutation-symmetric
/// start would keep every atom identical under a deterministic Newton and
/// leave the loading matrix on the rotationally degenerate manifold.
pub fn latent_block_spec(
    n_obs: usize,
    marks: usize,
    atoms: usize,
    start: Option<&RankStart>,
) -> Result<ParameterBlockSpec, EventHistoryError> {
    if atoms == 0 {
        return Err(EventHistoryError::InvalidInput {
            reason: "a latent block needs at least one atom".to_string(),
        });
    }
    let width = marks * atoms + atoms;
    let design = identity_pattern_design(n_obs, width)?;
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
    // With a start, every atom below the new one keeps the values the fit
    // one rank down reached and the new one begins at the direction, scale
    // and rate the covariance score named — a start the data chose, not a
    // symmetric one every atom shares. Without a start (rank one from
    // nothing) the loadings begin at one latent standard deviation per unit
    // of log-intensity, the parameterisation's own scale, and the rates
    // spread around the cohort's time scale.
    let mut initial_beta = Array1::<f64>::zeros(width);
    let mut initial_log_lambdas = Array1::<f64>::zeros(atoms);
    match start {
        Some(start) => {
            let carried = match start.atom {
                Some(_) => atoms - 1,
                None => atoms,
            };
            if start.loadings.len() != marks * carried
                || start.log_rates.len() != carried
                || start.log_lambdas.len() != carried
            {
                return Err(EventHistoryError::InvalidInput {
                    reason: format!(
                        "a rank start for {atoms} atoms carries {} loadings, {} rates and {} ridges for {carried} atoms",
                        start.loadings.len(),
                        start.log_rates.len(),
                        start.log_lambdas.len()
                    ),
                });
            }
            for d in 0..marks {
                for k in 0..carried {
                    initial_beta[d * atoms + k] = start.loadings[d * carried + k];
                }
            }
            for k in 0..carried {
                initial_beta[marks * atoms + k] = start.log_rates[k];
                initial_log_lambdas[k] = start.log_lambdas[k];
            }
            if let Some(atom) = start.atom.as_ref() {
                for d in 0..marks {
                    initial_beta[d * atoms + carried] = atom.loading[d];
                }
                initial_beta[marks * atoms + carried] = atom.log_rate;
                // Empirical Bayes at the proposal: the ridge whose prior
                // variance is the start's own scale.
                let scale: f64 = atom.loading.iter().map(|a| a * a).sum::<f64>()
                    + atom.log_rate * atom.log_rate;
                initial_log_lambdas[carried] =
                    ((marks + 1) as f64 / scale.max(f64::MIN_POSITIVE)).ln();
            }
        }
        None => {
            for q in 0..marks * atoms {
                initial_beta[q] = 1.0;
            }
            for k in 0..atoms {
                initial_beta[marks * atoms + k] =
                    std::f64::consts::LN_2 * (k as f64 - 0.5 * (atoms as f64 - 1.0));
            }
        }
    }
    Ok(ParameterBlockSpec {
        name: "latent".to_string(),
        design: DesignMatrix::Dense(DenseDesignMatrix::from(Arc::new(design))),
        offset: Array1::zeros(n_obs),
        penalties,
        nullspace_dims,
        initial_log_lambdas,
        initial_beta: Some(initial_beta),
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

/// Fit an event-history model from a formula right-hand side shared by every
/// mark, such as `x + s(time)`.
pub fn fit_event_history_formula(
    cohort: &mut EventHistoryCohort,
    formula: &str,
    options: BlockwiseFitOptions,
) -> Result<EventHistoryFit, EventHistoryError> {
    cohort.validate()?;
    let mut spec = EventHistorySpec::new(Vec::new());
    spec.options = options;
    let rows = design_rows(cohort, spec.quadrature_order)?;
    let covariates = super::formula::covariate_spec_from_formula(formula, rows.view(), cohort)?;
    spec.covariates = vec![covariates];
    fit_event_history(cohort, &spec)
}

/// The family and its block specs at one (order, mesh refinement) setting.
struct Built {
    nodes: Arc<CohortNodes>,
    family: EventHistoryFamily,
    designs: Vec<TermCollectionDesign>,
    dense: Vec<Arc<Array2<f64>>>,
    specs: Vec<ParameterBlockSpec>,
}

impl Built {
    /// Block states at the given per-block coefficients (the node predictors
    /// recomputed from this setting's designs).
    fn states(&self, betas: &[Array1<f64>]) -> Vec<ParameterBlockState> {
        let marks = self.family.marks();
        let mut states = Vec::with_capacity(betas.len());
        for (b, beta) in betas.iter().enumerate() {
            let eta = if b < marks {
                self.dense[b].dot(beta) + &self.designs[b].affine_offset
            } else {
                Array1::zeros(self.nodes.total_nodes)
            };
            states.push(ParameterBlockState {
                beta: beta.clone(),
                eta,
            });
        }
        states
    }
}

/// Bytes the family's evaluation may hold at once: the transient `S × S`
/// backward kernel of one gap, the carried `P × S` conditional expectations,
/// the per-node densities and operators of every node, per parallel
/// subject, in the widest scalar the outer solve uses (four channels).
fn transient_footprint_bytes(
    order: usize,
    atoms: usize,
    max_nodes: usize,
    marks: usize,
    total_width: usize,
) -> Result<f64, EventHistoryError> {
    let s = product_grid_size(order, atoms)? as f64;
    let g = order as f64;
    let n = max_nodes as f64;
    let per_subject = s * s
        + total_width as f64 * s
        + n * s * (4.0 + marks as f64)
        + n * atoms as f64 * 3.0 * g * g;
    let channels = 4.0;
    let bytes = 8.0 * channels * per_subject * rayon::current_num_threads() as f64;
    Ok(bytes)
}

fn preflight(
    order: usize,
    atoms: usize,
    max_nodes: usize,
    marks: usize,
    total_width: usize,
) -> Result<(), EventHistoryError> {
    let bytes = transient_footprint_bytes(order, atoms, max_nodes, marks, total_width)?;
    let budget = gam_runtime::resource::ResourcePolicy::default_library()
        .max_single_materialization_bytes as f64;
    if bytes > budget {
        return Err(EventHistoryError::InvalidInput {
            reason: format!(
                "an event-history evaluation at Gauss-Hermite order {order} over {atoms} atoms ({} grid points) with {max_nodes} nodes per subject needs about {:.1} GiB across {} threads, above this machine's {:.1} GiB materialisation budget; offer fewer atoms or a shorter follow-up per subject",
                product_grid_size(order, atoms)?,
                bytes / f64::from(1u32 << 30),
                rayon::current_num_threads(),
                budget / f64::from(1u32 << 30)
            ),
        });
    }
    Ok(())
}

/// Fit an event-history model to a cohort.
/// What a rank-`K+1` fit starts from: the converged rank-`K` latent block
/// and the atom the covariance score proposed.
pub struct RankStart {
    /// The rank-`K` loadings, `marks × K` row-major.
    pub loadings: Vec<f64>,
    /// `ln(rate · T̄)` of every atom the fit already has.
    pub log_rates: Vec<f64>,
    /// The ridge each of those atoms ended on.
    pub log_lambdas: Vec<f64>,
    /// The atom the covariance score proposed, when this start grows the
    /// rank. `None` re-starts the same rank from its own converged values,
    /// which is what certifying an accepted fit needs.
    pub(crate) atom: Option<NewAtom>,
}

/// One step of the rank path: what the covariance score proposed and what
/// the evidence made of it.
#[derive(Clone, Debug)]
pub struct RankStep {
    /// Rank before the step.
    pub rank: usize,
    /// The score's top eigenvalue at the proposed rate: the second-order
    /// evidence slope along the proposed direction.
    pub score_eigenvalue: f64,
    /// The log-rate the proposal named.
    pub proposed_log_rate: f64,
    /// The proposal was held at the rate the node mesh can still resolve.
    pub at_resolution_limit: bool,
    /// Whether the rank-`K+1` model reached a certified optimum at all. A
    /// candidate that cannot be fitted is refused: a fit object may only
    /// come from a converged optimisation, so an atom whose model has no
    /// certified optimum is not one the evidence can be said to support.
    pub converged: bool,
    /// Decrease of the outer LAML criterion the refit achieved.
    pub evidence_gain: f64,
    pub accepted: bool,
}

/// The certified fit at ONE rank: the Gauss-Hermite order and the time mesh
/// are refined until no fitted coefficient moves by more than the
/// certificate's tolerance. `start` warm-starts every block from a fit one
/// rank down, with the new atom's loadings at the covariance score's
/// proposal.
fn fit_at_rank(
    cohort: &EventHistoryCohort,
    spec: &EventHistorySpec,
    atoms: usize,
    start: Option<&RankStart>,
    pinned: Option<(usize, usize)>,
) -> Result<EventHistoryFit, EventHistoryError> {
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
    let time_scale = cohort.time_scale();
    let mut options = spec.options.clone();
    options.compute_covariance = true;

    // The bases are built on the outcome-free design rows and frozen; every
    // mesh evaluates the frozen bases, so a refinement never changes what a
    // coefficient means.
    let rows = design_rows(cohort, spec.quadrature_order)?;
    let mut frozen_specs = Vec::with_capacity(marks);
    for d in 0..marks {
        let term_spec = if spec.covariates.len() == 1 {
            &spec.covariates[0]
        } else {
            &spec.covariates[d]
        };
        let design = build_term_collection_design(rows.view(), term_spec).map_err(|error| {
            EventHistoryError::Fit {
                reason: format!("design for mark {d}: {error}"),
            }
        })?;
        let frozen = freeze_term_collection_from_design(term_spec, &design).map_err(|error| {
            EventHistoryError::Fit {
                reason: format!("freezing mark {d} term collection: {error}"),
            }
        })?;
        frozen_specs.push(frozen);
    }
    let build = |order: usize, refinement: usize| -> Result<Built, EventHistoryError> {
        let nodes = Arc::new(expand_nodes(cohort, spec.quadrature_order, refinement)?);
        let mut designs = Vec::with_capacity(marks);
        let mut dense = Vec::with_capacity(marks);
        let mut specs = Vec::with_capacity(marks + 1);
        for d in 0..marks {
            let design = build_term_collection_design(nodes.node_data.view(), &frozen_specs[d])
                .map_err(|error| EventHistoryError::Fit {
                    reason: format!("design for mark {d} on the mesh: {error}"),
                })?;
            let dense_design = design
                .design
                .try_to_dense_arc("event-history mark design")
                .map_err(|error| EventHistoryError::Fit {
                    reason: error.to_string(),
                })?;
            specs.push(mark_block_spec(&cohort.mark_names[d], &design));
            designs.push(design);
            dense.push(dense_design);
        }
        if atoms > 0 {
            specs.push(latent_block_spec(nodes.total_nodes, marks, atoms, start)?);
        }
        let family = EventHistoryFamily::new(
            Arc::clone(&nodes),
            dense.clone(),
            atoms,
            order,
            time_scale,
        )?;
        preflight(order, atoms, nodes.max_subject_nodes(), marks, family.total_width())?;
        Ok(Built {
            nodes,
            family,
            designs,
            dense,
            specs,
        })
    };
    let warm = |built: &mut Built, fit: &UnifiedFitResult| {
        let mut cursor = 0usize;
        for (block, state) in built.specs.iter_mut().zip(fit.block_states.iter()) {
            block.initial_beta = Some(state.beta.clone());
            let count = block.initial_log_lambdas.len();
            if cursor + count <= fit.log_lambdas.len() {
                block.initial_log_lambdas = fit.log_lambdas.slice(s![cursor..cursor + count]).to_owned();
            }
            cursor += count;
        }
    };
    // The largest shift of any fitted coefficient under a refinement, in
    // posterior standard deviations, without refitting: at the fitted
    // coefficients the current setting is stationary, so the refined
    // setting's penalised mode moves by `V (g' − g)` to first order, with
    // the two settings' exact gradients taken at the same coefficients and
    // `V` the posterior covariance the fit publishes (the inverse penalised
    // Hessian). One gradient per candidate, and no second optimisation whose
    // own convergence would have to be certified before the certificate
    // could be read.
    let check = |candidate: &Built,
                 fit: &UnifiedFitResult,
                 current_gradient: &[f64],
                 covariance: &Array2<f64>,
                 sd: &[f64]|
     -> Result<RefinementCheck, EventHistoryError> {
        let betas: Vec<Array1<f64>> = fit.block_states.iter().map(|s| s.beta.clone()).collect();
        let states = candidate.states(&betas);
        let refined_gradient = candidate
            .family
            .exact_gradient(&states)
            .map_err(|reason| EventHistoryError::Fit { reason })?;
        if refined_gradient.len() != current_gradient.len() {
            return Err(EventHistoryError::Fit {
                reason: format!(
                    "certificate: the refined setting has {} coefficients, the fit {}",
                    refined_gradient.len(),
                    current_gradient.len()
                ),
            });
        }
        let discrepancy: Vec<f64> = refined_gradient
            .iter()
            .zip(current_gradient.iter())
            .map(|(refined, current)| refined - current)
            .collect();
        let mut shift = 0.0_f64;
        for (q, scale) in sd.iter().enumerate() {
            let move_q: f64 = (0..discrepancy.len())
                .map(|r| covariance[[q, r]] * discrepancy[r])
                .sum();
            shift = shift.max(move_q.abs() / scale);
        }
        let log_likelihood = candidate
            .family
            .log_likelihood(&states)
            .map_err(|reason| EventHistoryError::Fit { reason })?;
        Ok(RefinementCheck {
            candidate: 0,
            coefficient_shift: shift,
            log_likelihood,
        })
    };

    let mesh_ceiling = cohort.mesh_refinement_ceiling();
    // A candidate the rank path is judging is fitted at the SETTING the
    // incumbent was certified at, not at its own: the two criteria are
    // compared, so they must be the same functional. Only the fit that is
    // returned runs the refinement ladder.
    let (mut order, mut refinement) = match pinned {
        Some((order, refinement)) => (order, refinement),
        None => (spec.gauss_hermite_order.max(3), 0usize),
    };
    // The grid is raised once for a posterior it cannot represent. A second
    // raise would climb into the orders where the Lagrange interpolant's own
    // roundoff is the larger error, so a posterior that survives one raise is
    // reported rather than chased.
    let mut positivity_raises = 0usize;
    let mut built = build(order, refinement)?;
    loop {
        let fit = match fit_custom_family(&built.family, &built.specs, &options) {
            Ok(fit) => fit,
            Err(error) => {
                // A posterior the grid cannot represent (its interpolant goes
                // negative where the mass is) is answered by resolving the
                // grid, which is the ladder this driver already owns — not by
                // handing the caller a number the representation could not
                // carry. Anything else is the caller's to see.
                let message = error.to_string();
                let next_order = 2 * order - 1;
                let admissible = atoms > 0
                    && positivity_raises == 0
                    && GaussHermite::new(next_order).is_ok_and(|rule| {
                        rule.lebesgue_constant * f64::EPSILON * built.nodes.max_subject_nodes() as f64
                            <= spec.quadrature_tolerance
                    });
                if message.contains(LOST_POSITIVITY) && admissible {
                    log::info!(
                        "[event-history] Gauss-Hermite order {order} cannot represent a posterior on this cohort; raising it to {next_order}"
                    );
                    order = next_order;
                    positivity_raises += 1;
                    built = build(order, refinement)?;
                    continue;
                }
                return Err(EventHistoryError::Fit {
                    reason: format!(
                        "event-history LAML fit at Gauss-Hermite order {order}, mesh refinement {refinement}: {message}"
                    ),
                });
            }
        };
        let total = built.family.total_width();
        // The certificate turns a gradient discrepancy into a coefficient
        // one through the posterior covariance, and reads the result in
        // posterior standard deviations, so it needs the whole matrix.
        let covariance = fit
            .beta_covariance()
            .filter(|c| c.nrows() == total && c.ncols() == total)
            .ok_or_else(|| EventHistoryError::Fit {
                reason: format!(
                    "the fit carries no {total}×{total} posterior covariance, which the refinement certificate measures its shift in"
                ),
            })?
            .clone();
        let sd: Vec<f64> = (0..total)
            .map(|q| covariance[[q, q]].max(0.0).sqrt())
            .collect();
        if let Some(q) = sd.iter().position(|s| !(s.is_finite() && *s > 0.0)) {
            return Err(EventHistoryError::Fit {
                reason: format!(
                    "coefficient {q} has no finite positive scale to measure a refinement's shift in ({}); it is unidentified at the fitted mode",
                    sd[q]
                ),
            });
        }
        let value = fit.log_likelihood;
        let current_gradient = built
            .family
            .exact_gradient(&fit.block_states)
            .map_err(|reason| EventHistoryError::Fit { reason })?;
        // Gauss-Hermite refinement: admissible while the interpolant's
        // roundoff amplification stays below the certificate's tolerance.
        // Without a latent block there is no latent integral to certify.
        let next_order = 2 * order - 1;
        let max_nodes = built.nodes.max_subject_nodes();
        let gauss_hermite = if atoms == 0 || pinned.is_some() {
            RefinementCheck {
                candidate: order,
                coefficient_shift: 0.0,
                log_likelihood: value,
            }
        } else {
            let rule = GaussHermite::new(next_order)?;
            if rule.lebesgue_constant * f64::EPSILON * max_nodes as f64 > spec.quadrature_tolerance {
                return Err(EventHistoryError::NumericalFailure {
                    reason: format!(
                        "the Gauss-Hermite certificate cannot be checked at order {next_order}: the Lagrange interpolant's Lebesgue constant {:.3e} amplifies roundoff above the tolerance {} over {max_nodes} nodes; the latent integral at order {order} is uncertified",
                        rule.lebesgue_constant,
                        spec.quadrature_tolerance
                    ),
                });
            }
            let order_candidate = build(next_order, refinement)?;
            let mut gauss_hermite = check(&order_candidate, &fit, &current_gradient, &covariance, &sd)?;
            gauss_hermite.candidate = next_order;
            if gauss_hermite.coefficient_shift > spec.quadrature_tolerance {
                log::info!(
                    "[event-history] Gauss-Hermite order {order} moves the coefficients by {:.3} posterior sd at order {next_order}; refitting",
                    gauss_hermite.coefficient_shift
                );
                order = next_order;
                built = order_candidate;
                warm(&mut built, &fit);
                continue;
            }
            gauss_hermite
        };
        if pinned.is_some() {
            let mesh = RefinementCheck {
                candidate: refinement,
                coefficient_shift: 0.0,
                log_likelihood: value,
            };
            return Ok(assemble(
                cohort,
                spec,
                frozen_specs,
                time_scale,
                Assembled { built, fit, atoms, order, refinement, value, gauss_hermite, mesh },
            ));
        }
        if refinement + 1 > mesh_ceiling {
            return Err(EventHistoryError::NumericalFailure {
                reason: format!(
                    "the fitted coefficients are still moving under mesh refinement at level {refinement}, where every cell is already narrower than the shortest interval this cohort's own breakpoints distinguish; the fit is not resolved by refining time further"
                ),
            });
        }
        let mesh_candidate = build(order, refinement + 1)?;
        let mut mesh = check(&mesh_candidate, &fit, &current_gradient, &covariance, &sd)?;
        mesh.candidate = refinement + 1;
        if mesh.coefficient_shift > spec.quadrature_tolerance {
            log::info!(
                "[event-history] mesh refinement {refinement} moves the coefficients by {:.3} posterior sd at refinement {}; refitting",
                mesh.coefficient_shift,
                refinement + 1
            );
            refinement += 1;
            built = mesh_candidate;
            warm(&mut built, &fit);
            continue;
        }
        return Ok(assemble(
            cohort,
            spec,
            frozen_specs,
            time_scale,
            Assembled { built, fit, atoms, order, refinement, value, gauss_hermite, mesh },
        ));
    }
}

/// The fitted model at one rank and one certified discretisation.
struct Assembled {
    built: Built,
    fit: UnifiedFitResult,
    atoms: usize,
    order: usize,
    refinement: usize,
    value: f64,
    gauss_hermite: RefinementCheck,
    mesh: RefinementCheck,
}

fn assemble(
    cohort: &EventHistoryCohort,
    spec: &EventHistorySpec,
    frozen_specs: Vec<TermCollectionSpec>,
    time_scale: f64,
    assembled: Assembled,
) -> EventHistoryFit {
    let Assembled {
        built,
        fit,
        atoms,
        order,
        refinement,
        value,
        gauss_hermite,
        mesh,
    } = assembled;
    let marks = cohort.marks();
    let empty = Array1::<f64>::zeros(0);
    let latent: &Array1<f64> = if atoms > 0 {
        &fit.block_states[marks].beta
    } else {
        &empty
    };
    let mut loadings = Array2::<f64>::zeros((marks, atoms));
    for d in 0..marks {
        for k in 0..atoms {
            loadings[[d, k]] = latent[d * atoms + k];
        }
    }
    let log_rates: Vec<f64> = (0..atoms).map(|k| latent[marks * atoms + k]).collect();
    let rates: Vec<f64> = log_rates.iter().map(|r| r.exp() / time_scale).collect();
    let n_lambda = fit.log_lambdas.len();
    let atom_log_lambdas: Vec<f64> = fit
        .log_lambdas
        .iter()
        .skip(n_lambda.saturating_sub(atoms))
        .copied()
        .collect();
    let Built {
        nodes,
        family,
        designs,
        ..
    } = built;
    EventHistoryFit {
        nodes,
        family,
        fit,
        mark_kinds: cohort.mark_kinds.clone(),
        frozen_specs,
        designs,
        loadings,
        log_rates,
        rates,
        atom_log_lambdas,
        time_scale,
        quadrature_order: spec.quadrature_order,
        mesh_refinement: refinement,
        quadrature: QuadratureCertificate {
            gauss_hermite_order: order,
            mesh_refinement: refinement,
            log_likelihood: value,
            gauss_hermite,
            mesh,
        },
        rank_path: Vec::new(),
        atom_evidence: Vec::new(),
    }
}

/// Fit an event-history model, growing the rank of the latent covariance
/// from zero until the evidence refuses the next direction.
///
/// At each rank the fit is the certified one — the Gauss-Hermite order and
/// the time mesh are refined until no fitted coefficient moves by more than
/// the certificate's tolerance — and the next atom is proposed by the
/// covariance score of that fit's martingale residuals: the direction and
/// rate whose standardised evidence gain is largest, which is the most
/// evidence-improving covariance direction the current rank omits. The
/// candidate is refit and kept only when the outer LAML criterion decreases
/// by more than the solver's own resolution. Nothing about the latent
/// structure is chosen by hand: not the number of atoms, not their
/// directions, not their rates.
pub fn fit_event_history(
    cohort: &mut EventHistoryCohort,
    spec: &EventHistorySpec,
) -> Result<EventHistoryFit, EventHistoryError> {
    cohort.validate()?;
    let marks = cohort.marks();
    let time_scale = cohort.time_scale();
    let resolution = |criterion: f64| spec.options.outer_tol.max(f64::EPSILON * criterion.abs());
    // The rank path compares criteria, so every step of it runs at ONE
    // setting — the spec's own starting order and mesh. Only the fit that is
    // returned runs the refinement ladder, once, at the rank the evidence
    // chose. Certifying every candidate would run a ladder per rank and
    // certify models the caller never sees.
    let pin = Some((spec.gauss_hermite_order.max(3), 0usize));
    let mut fit = fit_at_rank(cohort, spec, 0, None, pin)?;
    let mut rank_path: Vec<RankStep> = Vec::new();
    let mut atom_evidence: Vec<f64> = Vec::new();
    loop {
        let rank = fit.atoms();
        let criterion = match fit.fit.reml_score() {
            Some(value) => value,
            None => break,
        };
        let residuals = fit
            .family
            .residuals(&fit.fit.block_states)
            .map_err(|reason| EventHistoryError::Fit { reason })?;
        let Some(atom) = best_new_atom(&residuals, marks, time_scale)? else {
            break;
        };
        let start = RankStart {
            loadings: fit.loadings.iter().copied().collect(),
            log_rates: fit.log_rates.clone(),
            log_lambdas: fit.atom_log_lambdas.clone(),
            atom: Some(atom.clone()),
        };
        let grown = fit_at_rank(cohort, spec, rank + 1, Some(&start), pin)
            .and_then(|candidate| match candidate.fit.reml_score() {
                Some(value) => Ok((candidate, value)),
                None => Err(EventHistoryError::Fit {
                    reason: format!("the rank-{} fit carries no outer LAML criterion", rank + 1),
                }),
            });
        let (candidate, next_criterion) = match grown {
            Ok(value) => value,
            Err(error) => {
                // No certified optimum at the next rank, so there is no
                // evidence for it to report: the path stops with the reason
                // recorded rather than failing the whole fit.
                log::info!(
                    "[event-history] rank {rank} → {}: no certified optimum, refused ({error})",
                    rank + 1
                );
                rank_path.push(RankStep {
                    rank,
                    score_eigenvalue: atom.eigenvalue,
                    proposed_log_rate: atom.log_rate,
                    at_resolution_limit: atom.at_resolution_limit,
                    converged: false,
                    evidence_gain: 0.0,
                    accepted: false,
                });
                break;
            }
        };
        let gain = criterion - next_criterion;
        let accepted = gain > resolution(criterion);
        log::info!(
            "[event-history] rank {rank} → {}: score eigenvalue {:.4e} at log-rate {:.3}{} (one-step variance {:.4e}), criterion {criterion:.6} → {next_criterion:.6} ({})",
            rank + 1,
            atom.eigenvalue,
            atom.log_rate,
            if atom.at_resolution_limit { " (the mesh's resolution limit)" } else { "" },
            atom.variance,
            if accepted { "accepted" } else { "refused" }
        );
        rank_path.push(RankStep {
            rank,
            score_eigenvalue: atom.eigenvalue,
            proposed_log_rate: atom.log_rate,
            at_resolution_limit: atom.at_resolution_limit,
            converged: true,
            evidence_gain: gain,
            accepted,
        });
        if !accepted {
            break;
        }
        atom_evidence.push(gain);
        fit = candidate;
    }
    {
        // The whole path ran pinned, so the winner has not been certified:
        // it runs the ladder once, from where it already is, and the
        // certificate the caller reads belongs to the model the caller gets.
        let start = RankStart {
            loadings: fit.loadings.iter().copied().collect(),
            log_rates: fit.log_rates.clone(),
            log_lambdas: fit.atom_log_lambdas.clone(),
            atom: None,
        };
        let rank = fit.rank();
        let start = (rank > 0).then_some(start);
        fit = fit_at_rank(cohort, spec, rank, start.as_ref(), None)?;
    }
    fit.rank_path = rank_path;
    fit.atom_evidence = atom_evidence;
    Ok(fit)
}
