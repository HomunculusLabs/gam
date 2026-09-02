//! The event-history custom family: every subject's Laplace evidence
//! assembled into the coefficient space of the per-mark smooth blocks and
//! the latent block, with every derivative the outer LAML evaluator asks for
//! produced by the same generic code under a directional dual scalar, and
//! the rank of the latent covariance grown from zero by the evidence.
//!
//! Nothing of size `(nodes × marks)²` is ever formed: a subject's evidence
//! gradient lives in its node log-intensities, and the Hessian columns are
//! read off the tangent channels of that gradient and contracted with the
//! design rows node by node.

use super::cohort::{CohortNodes, EventHistoryCohort, EventHistoryError, MarkKind, design_rows, expand_nodes};
use super::covariance::{NewAtom, SubjectResiduals, best_new_atom};
use super::laplace::{self, Gaussian, Smoother, SubjectInputs};
use super::scalar::Tangent;
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

/// A base scalar that can be seeded with up to two directions and read back.
pub(crate) trait Directional: JetField + Send + Sync + Copy {
    fn seeded(value: f64, u: f64, v: f64) -> Self;
    /// The channel a directional evaluation reads: the value itself on
    /// `f64`, the `ε` coefficient on a one-direction dual, the `εδ`
    /// coefficient on a two-direction dual.
    fn channel(&self) -> f64;
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
    fn channel(&self) -> f64 {
        *self
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
    fn channel(&self) -> f64 {
        self.eps.value()
    }
}

impl Directional for TwoSeed<0> {
    fn seeded(value: f64, u: f64, v: f64) -> Self {
        validate_seed_directions([u, v]);
        TwoSeed {
            base: scalar0(value),
            eps: scalar0(u),
            del: scalar0(v),
            eps_del: scalar0(0.0),
        }
    }
    fn channel(&self) -> f64 {
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

#[derive(Default)]
struct Cache {
    /// The state the cached modes and evaluation belong to.
    key: Option<u64>,
    modes: Option<Arc<Vec<Vec<f64>>>>,
    evaluation: Option<Arc<JointEvaluation>>,
    /// The last modes computed at any state: a warm start for the next
    /// mode search (the mode is unique, so the start changes nothing).
    warm: Option<Arc<Vec<Vec<f64>>>>,
}

/// The event-history family over a node-expanded cohort at a fixed number
/// of atoms.
#[derive(Clone)]
pub struct EventHistoryFamily {
    nodes: Arc<CohortNodes>,
    /// Dense per-mark designs on the nodes, `n_obs × p_d`.
    designs: Vec<Arc<Array2<f64>>>,
    /// `ln(rate · T̄)` of every atom.
    ///
    /// The rate is data, not a coefficient. It is learned — the covariance
    /// score's matched filter reads it off the residuals of the fit one rank
    /// down, and the evidence then judges the whole atom — but it is not
    /// fitted jointly with the loadings, because past the rate the node mesh
    /// can resolve the likelihood is EXACTLY flat in it: the transition
    /// correlation underflows, every faster rate is the same model, and a
    /// coefficient with no identified direction leaves the inner solve
    /// unable to certify a stationary point it has already reached.
    log_rates: Vec<f64>,
    time_scale: f64,
    cache: Arc<Mutex<Cache>>,
}

impl EventHistoryFamily {
    pub fn new(
        nodes: Arc<CohortNodes>,
        designs: Vec<Arc<Array2<f64>>>,
        log_rates: Vec<f64>,
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
        if log_rates.iter().any(|r| !r.is_finite()) {
            return Err(EventHistoryError::InvalidInput {
                reason: "every atom needs a finite log-rate".to_string(),
            });
        }
        Ok(Self {
            nodes,
            designs,
            log_rates,
            time_scale,
            cache: Arc::new(Mutex::new(Cache::default())),
        })
    }

    pub fn marks(&self) -> usize {
        self.nodes.marks
    }

    pub fn atoms(&self) -> usize {
        self.log_rates.len()
    }

    /// `ln(rate · T̄)` of every atom.
    pub fn log_rates(&self) -> &[f64] {
        &self.log_rates
    }

    pub fn time_scale(&self) -> f64 {
        self.time_scale
    }

    pub fn nodes(&self) -> &Arc<CohortNodes> {
        &self.nodes
    }

    /// Width of the latent block: the loadings.
    pub fn latent_width(&self) -> usize {
        self.marks() * self.atoms()
    }

    /// Whether the fit carries a latent block (no atoms means a plain
    /// Poisson-process GAM with the same node expansion).
    pub fn has_latent_block(&self) -> bool {
        self.atoms() > 0
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

    fn total_width(&self) -> usize {
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

    fn latent_beta<'a>(&self, states: &'a [ParameterBlockState]) -> &'a [f64] {
        if self.has_latent_block() {
            states[self.marks()].beta.as_slice().expect("contiguous latent block")
        } else {
            &[]
        }
    }

    /// The latent modes of every subject at `states`, from the cache when
    /// the state is the cached one, else computed (warm-started from the
    /// last modes) and cached.
    pub(crate) fn modes(&self, states: &[ParameterBlockState]) -> Result<Arc<Vec<Vec<f64>>>, String> {
        self.validate_states(states)?;
        let key = Self::state_key(states);
        let warm = {
            let guard = self.cache.lock().map_err(|_| "event-history cache poisoned".to_string())?;
            if guard.key == Some(key)
                && let Some(modes) = guard.modes.as_ref()
            {
                return Ok(Arc::clone(modes));
            }
            guard.warm.clone()
        };
        let marks = self.marks();

        let loadings = self.latent_beta(states);
        let log_rates = &self.log_rates;
        let time_scale = self.time_scale;
        let modes: Result<Vec<Vec<f64>>, String> = self
            .nodes
            .subjects
            .par_iter()
            .enumerate()
            .map(|(index, subject)| {
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
                    loadings,
                    log_rates,
                    time_scale,
                };
                let start = warm.as_ref().and_then(|w| w.get(index)).map(|v| v.as_slice());
                laplace::find_mode(&inputs, start).map_err(|e| format!("subject {index}: {e}"))
            })
            .collect();
        let modes = Arc::new(modes?);
        if let Ok(mut guard) = self.cache.lock() {
            guard.key = Some(key);
            guard.modes = Some(Arc::clone(&modes));
            guard.evaluation = None;
            guard.warm = Some(Arc::clone(&modes));
        }
        Ok(modes)
    }

    /// Log-likelihood only: the modes and the Laplace evidence, no derivatives.
    pub fn log_likelihood(&self, states: &[ParameterBlockState]) -> Result<f64, String> {
        let modes = self.modes(states)?;
        let marks = self.marks();

        let loadings = self.latent_beta(states);
        let log_rates = &self.log_rates;
        let time_scale = self.time_scale;
        let values: Result<Vec<f64>, String> = self
            .nodes
            .subjects
            .par_iter()
            .zip(modes.par_iter())
            .map(|(subject, mode)| {
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
                    loadings,
                    log_rates,
                    time_scale,
                };
                laplace::evidence(&inputs, mode, false)
                    .map(|e| e.loglik)
                    .map_err(|e| e.to_string())
            })
            .collect();
        Ok(pairwise_sum(&values?))
    }

    /// Value and exact gradient in coefficient space.
    fn value_and_gradient(
        &self,
        states: &[ParameterBlockState],
        modes: &[Vec<f64>],
    ) -> Result<(f64, Vec<f64>), String> {
        let marks = self.marks();
        let atoms = self.atoms();
        let offsets = self.block_offsets();
        let total = self.total_width();
        let latent_offset = offsets[marks];
        let loadings = self.latent_beta(states);
        let log_rates = &self.log_rates;
        let time_scale = self.time_scale;
        let designs = &self.designs;
        let per_subject: Result<Vec<(f64, Vec<f64>)>, String> = self
            .nodes
            .subjects
            .par_iter()
            .zip(modes.par_iter())
            .map(|(subject, mode)| {
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
                    loadings,
                    log_rates,
                    time_scale,
                };
                let local = laplace::evidence(&inputs, mode, true).map_err(|e| e.to_string())?;
                let mut gradient = vec![0.0; total];
                for node in 0..n {
                    for d in 0..marks {
                        let g = local.gradient[node * marks + d];
                        if g == 0.0 {
                            continue;
                        }
                        let row = designs[d].row(first + node);
                        for (j, x) in row.iter().enumerate() {
                            if *x != 0.0 {
                                gradient[offsets[d] + j] += g * x;
                            }
                        }
                    }
                }
                for q in 0..marks * atoms {
                    gradient[latent_offset + q] = local.gradient[n * marks + q];
                }
                Ok((local.loglik, gradient))
            })
            .collect();
        let per_subject = per_subject?;
        let values: Vec<f64> = per_subject.iter().map(|(l, _)| *l).collect();
        let mut gradient = vec![0.0; total];
        for (_, g) in &per_subject {
            for (acc, x) in gradient.iter_mut().zip(g.iter()) {
                *acc += x;
            }
        }
        Ok((pairwise_sum(&values), gradient))
    }

    /// The log-likelihood Hessian (or a directional derivative of it) in
    /// coefficient space: every column is the tangent channel of the exact
    /// evidence gradient along one coefficient direction, contracted with the
    /// design rows at the node level.
    fn hessian_generic<B: Directional>(
        &self,
        states: &[ParameterBlockState],
        u: Option<&Array1<f64>>,
        v: Option<&Array1<f64>>,
        modes: &[Vec<f64>],
    ) -> Result<Vec<f64>, String> {
        let total = self.total_width();
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
        let mut out = vec![0.0; total * total];
        if total <= 4 {
            self.hessian_chunks::<B, 4>(states, u, v, modes, &mut out)?;
        } else if total <= 8 {
            self.hessian_chunks::<B, 8>(states, u, v, modes, &mut out)?;
        } else if total <= 16 {
            self.hessian_chunks::<B, 16>(states, u, v, modes, &mut out)?;
        } else if total <= 32 {
            self.hessian_chunks::<B, 32>(states, u, v, modes, &mut out)?;
        } else {
            self.hessian_chunks::<B, 64>(states, u, v, modes, &mut out)?;
        }
        for i in 0..total {
            for j in (i + 1)..total {
                let mean = 0.5 * (out[i * total + j] + out[j * total + i]);
                out[i * total + j] = mean;
                out[j * total + i] = mean;
            }
        }
        Ok(out)
    }

    fn hessian_chunks<B: Directional, const W: usize>(
        &self,
        states: &[ParameterBlockState],
        u: Option<&Array1<f64>>,
        v: Option<&Array1<f64>>,
        modes: &[Vec<f64>],
        out: &mut [f64],
    ) -> Result<(), String> {
        let marks = self.marks();
        let atoms = self.atoms();
        let offsets = self.block_offsets();
        let total = self.total_width();
        let latent_offset = offsets[marks];
        let latent = self.latent_beta(states);
        let designs = &self.designs;
        let time_scale = self.time_scale;
        let component = |dir: Option<&Array1<f64>>, index: usize| -> f64 { dir.map_or(0.0, |d| d[index]) };
        let row_direction = |dir: Option<&Array1<f64>>, d: usize, row: usize| -> f64 {
            dir.map_or(0.0, |dir| {
                designs[d]
                    .row(row)
                    .iter()
                    .enumerate()
                    .map(|(j, x)| x * dir[offsets[d] + j])
                    .sum()
            })
        };
        for start in (0..total).step_by(W) {
            let end = (start + W).min(total);
            let seed = |slot: usize, grad: &mut [B; W], x: f64| {
                if (start..end).contains(&slot) {
                    grad[slot - start] = B::seeded(x, 0.0, 0.0);
                }
            };
            let latent_scalar = |q: usize| -> Tangent<B, W> {
                let value = B::seeded(
                    latent[q],
                    component(u, latent_offset + q),
                    component(v, latent_offset + q),
                );
                let mut grad = [B::seeded(0.0, 0.0, 0.0); W];
                seed(latent_offset + q, &mut grad, 1.0);
                Tangent::seeded(value, grad)
            };
            let loadings: Vec<Tangent<B, W>> = (0..marks * atoms).map(latent_scalar).collect();
            // The rates are data: they carry no coefficient direction and no
            // tangent slot, so every derivative channel through them is zero
            // by construction rather than by cancellation.
            let log_rates: Vec<Tangent<B, W>> = self
                .log_rates
                .iter()
                .map(|r| {
                    Tangent::seeded(B::seeded(*r, 0.0, 0.0), [B::seeded(0.0, 0.0, 0.0); W])
                })
                .collect();
            let block: Result<Vec<f64>, String> = self
                .nodes
                .subjects
                .par_iter()
                .zip(modes.par_iter())
                .map(|(subject, mode)| {
                    let n = subject.len();
                    let first = subject.first_row;
                    let mut eta0 = Vec::with_capacity(n * marks);
                    for node in 0..n {
                        let row = first + node;
                        for d in 0..marks {
                            let value = B::seeded(
                                states[d].eta[row],
                                row_direction(u, d, row),
                                row_direction(v, d, row),
                            );
                            let mut grad = [B::seeded(0.0, 0.0, 0.0); W];
                            for (j, x) in designs[d].row(row).iter().enumerate() {
                                seed(offsets[d] + j, &mut grad, *x);
                            }
                            eta0.push(Tangent::seeded(value, grad));
                        }
                    }
                    let inputs = SubjectInputs {
                        nodes: subject,
                        eta0: &eta0,
                        loadings: &loadings,
                        log_rates: &log_rates,
                        time_scale,
                    };
                    let local = laplace::evidence(&inputs, mode, true).map_err(|e| e.to_string())?;
                    let mut acc = vec![0.0; total * W];
                    for node in 0..n {
                        for d in 0..marks {
                            let g = &local.gradient[node * marks + d];
                            let row = designs[d].row(first + node);
                            for w in 0..(end - start) {
                                let value = g.grad[w].channel();
                                if value == 0.0 {
                                    continue;
                                }
                                for (j, x) in row.iter().enumerate() {
                                    if *x != 0.0 {
                                        acc[(offsets[d] + j) * W + w] += x * value;
                                    }
                                }
                            }
                        }
                    }
                    for q in 0..marks * atoms {
                        let g = &local.gradient[n * marks + q];
                        for w in 0..(end - start) {
                            acc[(latent_offset + q) * W + w] += g.grad[w].channel();
                        }
                    }
                    Ok(acc)
                })
                .try_reduce(
                    || vec![0.0; total * W],
                    |mut a, b| {
                        for (x, y) in a.iter_mut().zip(b.iter()) {
                            *x += y;
                        }
                        Ok(a)
                    },
                );
            let block = block?;
            for row in 0..total {
                for w in 0..(end - start) {
                    out[row * total + start + w] = block[row * W + w];
                }
            }
        }
        Ok(())
    }

    /// Full `f64` joint evaluation, cached on the state: the value, its exact
    /// gradient, and the exact Hessian.
    pub fn joint_evaluation(
        &self,
        states: &[ParameterBlockState],
    ) -> Result<Arc<JointEvaluation>, String> {
        let key = Self::state_key(states);
        if let Ok(guard) = self.cache.lock()
            && guard.key == Some(key)
            && let Some(value) = guard.evaluation.as_ref()
        {
            return Ok(Arc::clone(value));
        }
        let modes = self.modes(states)?;
        let (loglik, gradient) = self.value_and_gradient(states, &modes)?;
        let hessian = self.hessian_generic::<f64>(states, None, None, &modes)?;
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
        if let Ok(mut guard) = self.cache.lock()
            && guard.key == Some(key)
        {
            guard.evaluation = Some(Arc::clone(&evaluation));
        }
        Ok(evaluation)
    }

    /// `D_β H[u]` for the negative log-likelihood Hessian `H`.
    pub fn directional_hessian(
        &self,
        states: &[ParameterBlockState],
        u: &Array1<f64>,
    ) -> Result<Array2<f64>, String> {
        let modes = self.modes(states)?;
        let total = self.total_width();
        let hessian = self.hessian_generic::<OneSeed<0>>(states, Some(u), None, &modes)?;
        let mut out = Array2::<f64>::zeros((total, total));
        for i in 0..total {
            for j in 0..total {
                out[[i, j]] = -hessian[i * total + j];
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
        let modes = self.modes(states)?;
        let total = self.total_width();
        let hessian = self.hessian_generic::<TwoSeed<0>>(states, Some(u), Some(v), &modes)?;
        let mut out = Array2::<f64>::zeros((total, total));
        for i in 0..total {
            for j in 0..total {
                out[[i, j]] = -hessian[i * total + j];
            }
        }
        Ok(out)
    }

    fn subject_inputs<'a>(
        &'a self,
        states: &[ParameterBlockState],
        index: usize,
        eta0: &'a mut Vec<f64>,
        latent: &'a [f64],
    ) -> SubjectInputs<'a, f64> {
        let marks = self.marks();
        let subject = &self.nodes.subjects[index];
        eta0.clear();
        for node in 0..subject.len() {
            for d in 0..marks {
                eta0.push(states[d].eta[subject.first_row + node]);
            }
        }
        SubjectInputs {
            nodes: subject,
            eta0,
            loadings: latent,
            log_rates: &self.log_rates,
            time_scale: self.time_scale,
        }
    }

    /// The Laplace posterior of every subject's latent path at `states`.
    pub(crate) fn smoothers(&self, states: &[ParameterBlockState]) -> Result<Vec<Smoother>, String> {
        let modes = self.modes(states)?;
        let latent = self.latent_beta(states).to_vec();
        (0..self.nodes.subjects.len())
            .into_par_iter()
            .map(|index| {
                let mut eta0 = Vec::new();
                let inputs = self.subject_inputs(states, index, &mut eta0, &latent);
                laplace::smoother(&inputs, &modes[index]).map_err(|e| e.to_string())
            })
            .collect()
    }

    /// The residual scores and curvatures of every subject at `states`,
    /// under the posterior-mean intensities.
    pub(crate) fn residuals(&self, states: &[ParameterBlockState]) -> Result<Vec<SubjectResiduals>, String> {
        let marks = self.marks();
        let smoothers = self.smoothers(states)?;
        Ok(self
            .nodes
            .subjects
            .iter()
            .zip(smoothers.iter())
            .map(|(subject, smoother)| {
                let n = subject.len();
                let mut scores = Vec::with_capacity(n * marks);
                let mut curvatures = Vec::with_capacity(n * marks);
                for node in 0..n {
                    for d in 0..marks {
                        let w = subject.exposures[[node, d]];
                        let c = w * smoother.intensity[node * marks + d];
                        scores.push(subject.counts[[node, d]] - c);
                        curvatures.push(c);
                    }
                }
                SubjectResiduals {
                    times: subject.times.clone(),
                    scores,
                    curvatures,
                }
            })
            .collect())
    }

    /// The follow-up average of every subject's latent state as a Gaussian.
    pub(crate) fn exposures(&self, states: &[ParameterBlockState]) -> Result<Vec<Gaussian>, String> {
        let modes = self.modes(states)?;
        let latent = self.latent_beta(states).to_vec();
        (0..self.nodes.subjects.len())
            .into_par_iter()
            .map(|index| {
                let mut eta0 = Vec::new();
                let inputs = self.subject_inputs(states, index, &mut eta0, &latent);
                laplace::exposure(&inputs, &modes[index]).map_err(|e| e.to_string())
            })
            .collect()
    }
}

/// Pairwise (tree) sum: rounding error grows like `log₂ n` rather than `n`,
/// which keeps a log-likelihood summed over many subjects resolvable at the
/// level a Newton acceptance test needs.
pub(crate) fn pairwise_sum(terms: &[f64]) -> f64 {
    match terms.len() {
        0 => 0.0,
        1 => terms[0],
        2 => terms[0] + terms[1],
        n => {
            let (left, right) = terms.split_at(n / 2);
            pairwise_sum(left) + pairwise_sum(right)
        }
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

    /// The evidence is not log-concave in the loadings: at zero loading the
    /// profile can be locally convex (a saddle of the negative
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
    /// Quadrature order per mesh cell.
    pub quadrature_order: usize,
    /// Mesh refinement: every cell between breakpoints is split into
    /// `2^refinement` parts before the rule is applied.
    pub mesh_refinement: usize,
    pub options: BlockwiseFitOptions,
}

impl EventHistorySpec {
    pub fn new(covariates: Vec<TermCollectionSpec>) -> Self {
        Self {
            covariates,
            quadrature_order: super::cohort::quadrature_order_for_degree(3),
            mesh_refinement: 0,
            options: BlockwiseFitOptions::default(),
        }
    }
}

/// One step of the rank path: the atom the covariance score proposed and
/// what the evidence made of it.
#[derive(Clone, Debug)]
pub struct RankStep {
    /// Rank before the step.
    pub rank: usize,
    /// The score's top eigenvalue at the proposed rate: the second-order
    /// evidence slope along the proposed direction.
    pub score_eigenvalue: f64,
    /// Log-rate the proposal started at.
    pub proposed_log_rate: f64,
    /// The proposal was held at the rate the node mesh can still resolve:
    /// the residuals carry structure faster than the quadrature mesh.
    pub at_resolution_limit: bool,
    /// Whether the rank-`K+1` model reached a certified optimum at all. A
    /// candidate that cannot be fitted is refused: a fit object may only
    /// come from a converged optimisation, so an atom whose model has no
    /// certified optimum is not one the evidence can be said to support.
    pub converged: bool,
    /// Decrease of the outer LAML criterion the refit achieved; the step
    /// was accepted iff this exceeds the solver's own resolution.
    pub evidence_gain: f64,
    pub accepted: bool,
}

/// A fitted event-history model.
pub struct EventHistoryFit {
    pub nodes: Arc<CohortNodes>,
    pub family: EventHistoryFamily,
    pub fit: UnifiedFitResult,
    /// Frozen per-mark term collections for prediction.
    pub frozen_specs: Vec<TermCollectionSpec>,
    pub designs: Vec<TermCollectionDesign>,
    pub mark_kinds: Vec<MarkKind>,
    /// Factor coordinates of the latent covariance, `a[d, k]`. The
    /// covariance itself is the reported object: see
    /// [`Self::disease_covariance`].
    pub loadings: Array2<f64>,
    /// `ln(rate_k · T̄)` of every atom, as the covariance score named it.
    pub log_rates: Vec<f64>,
    /// Rates in the data's time unit.
    pub rates: Vec<f64>,
    /// REML log smoothing parameter of each atom's ridge.
    pub atom_log_lambdas: Vec<f64>,
    /// The decrease of the outer criterion each accepted atom brought.
    pub atom_evidence: Vec<f64>,
    /// Every rank step tried, accepted or refused.
    pub rank_path: Vec<RankStep>,
    pub time_scale: f64,
    /// Quadrature order per mesh cell used for the training nodes.
    pub quadrature_order: usize,
    /// Mesh refinement used for the training nodes.
    pub mesh_refinement: usize,
}

impl EventHistoryFit {
    pub fn marks(&self) -> usize {
        self.nodes.marks
    }

    /// The rank of the latent covariance the evidence supports.
    pub fn rank(&self) -> usize {
        self.log_rates.len()
    }

    /// Fitted coefficients of the mark-`d` block.
    pub fn mark_coefficients(&self, d: usize) -> &Array1<f64> {
        &self.fit.block_states[d].beta
    }

    /// Population-average node log-intensities `η⁰` for mark `d` on the
    /// training nodes.
    pub fn mark_eta(&self, d: usize) -> &Array1<f64> {
        &self.fit.block_states[d].eta
    }

    /// `C(0) = A Aᵀ`: the covariance of the marks' latent log-intensity
    /// deviations at one time.
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

    /// The Laplace posterior of every training subject's latent path.
    pub fn smoothers(&self) -> Result<Vec<Smoother>, EventHistoryError> {
        self.family
            .smoothers(&self.fit.block_states)
            .map_err(|reason| EventHistoryError::Fit { reason })
    }

    /// The follow-up average of every training subject's latent state as a
    /// posterior Gaussian (mean per atom, covariance atoms × atoms).
    pub fn latent_exposures(&self) -> Result<Vec<(Vec<f64>, Vec<f64>)>, EventHistoryError> {
        self.family
            .exposures(&self.fit.block_states)
            .map(|g| g.into_iter().map(|g| (g.mean, g.cov)).collect())
            .map_err(|reason| EventHistoryError::Fit { reason })
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
/// coordinate in the Laplace integral. `initial_beta` is the block's start
/// (`marks × atoms` loadings then `atoms` log-rates) and
/// `initial_log_lambdas` one ridge start per atom.
pub fn latent_block_spec(
    n_obs: usize,
    marks: usize,
    atoms: usize,
    initial_beta: Array1<f64>,
    initial_log_lambdas: Array1<f64>,
) -> Result<ParameterBlockSpec, EventHistoryError> {
    if atoms == 0 {
        return Err(EventHistoryError::InvalidInput {
            reason: "a latent block needs at least one atom".to_string(),
        });
    }
    let width = marks * atoms;
    if initial_beta.len() != width || initial_log_lambdas.len() != atoms {
        return Err(EventHistoryError::InvalidInput {
            reason: format!(
                "latent block start has {} coefficients and {} ridges, expected {width} and {atoms}",
                initial_beta.len(),
                initial_log_lambdas.len()
            ),
        });
    }
    let design = identity_pattern_design(n_obs, width)?;
    let mut penalties = Vec::with_capacity(atoms);
    let mut nullspace_dims = Vec::with_capacity(atoms);
    for k in 0..atoms {
        let mut s = Array2::<f64>::zeros((width, width));
        for d in 0..marks {
            s[[d * atoms + k, d * atoms + k]] = 1.0;
        }
        penalties.push(PenaltyMatrix::Dense(s));
        nullspace_dims.push(width - marks);
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

/// Warm-start every block spec from a converged fit: coefficients per block
/// and the smoothing parameters in block order.
fn warm_start(specs: &mut [ParameterBlockSpec], fit: &UnifiedFitResult) {
    let mut cursor = 0usize;
    for (block, state) in specs.iter_mut().zip(fit.block_states.iter()) {
        block.initial_beta = Some(state.beta.clone());
        let count = block.initial_log_lambdas.len();
        if cursor + count <= fit.log_lambdas.len() {
            block.initial_log_lambdas = fit
                .log_lambdas
                .slice(ndarray::s![cursor..cursor + count])
                .to_owned();
        }
        cursor += count;
    }
}

fn outer_criterion(fit: &UnifiedFitResult, rank: usize) -> Result<f64, EventHistoryError> {
    fit.reml_score().ok_or_else(|| EventHistoryError::Fit {
        reason: format!("the rank-{rank} fit carries no outer LAML criterion"),
    })
}

/// The converged state of the rank path.
struct RankState {
    family: EventHistoryFamily,
    fit: UnifiedFitResult,
    loadings: Vec<f64>,
    log_rates: Vec<f64>,
    atom_evidence: Vec<f64>,
    rank_path: Vec<RankStep>,
}

/// Grow the rank from zero until the evidence refuses the next direction.
fn grow_rank(
    nodes: &Arc<CohortNodes>,
    dense: &[Arc<Array2<f64>>],
    mark_specs: &[ParameterBlockSpec],
    marks: usize,
    time_scale: f64,
    options: &BlockwiseFitOptions,
) -> Result<RankState, EventHistoryError> {
    let fit_at = |rates: &[f64], specs: &[ParameterBlockSpec]| -> Result<(EventHistoryFamily, UnifiedFitResult), EventHistoryError> {
        let rank = rates.len();
        let family =
            EventHistoryFamily::new(Arc::clone(nodes), dense.to_vec(), rates.to_vec(), time_scale)?;
        let fit = fit_custom_family(&family, specs, options).map_err(|error| EventHistoryError::Fit {
            reason: format!("event-history LAML fit at rank {rank}: {error}"),
        })?;
        Ok((family, fit))
    };
    let (mut family, mut fit) = fit_at(&[], mark_specs)?;
    let mut loadings: Vec<f64> = Vec::new();
    let mut log_rates: Vec<f64> = Vec::new();
    let mut atom_evidence: Vec<f64> = Vec::new();
    let mut rank_path: Vec<RankStep> = Vec::new();
    loop {
        let rank = log_rates.len();
        let residuals = family
            .residuals(&fit.block_states)
            .map_err(|reason| EventHistoryError::Fit { reason })?;
        let Some(atom) = best_new_atom(&residuals, marks, time_scale)? else {
            break;
        };
        let criterion = outer_criterion(&fit, rank)?;
        let candidate = grow(&fit, mark_specs, &loadings, log_rates.len(), &atom, marks, nodes.total_nodes)?;
        let mut next_rates = log_rates.clone();
        next_rates.push(atom.log_rate);
        let grown = fit_at(&next_rates, &candidate).and_then(|(family, fit)| {
            let criterion = outer_criterion(&fit, rank + 1)?;
            Ok((family, fit, criterion))
        });
        let (next_family, next_fit, next_criterion) = match grown {
            Ok(value) => value,
            Err(error) => {
                // The rank-`K+1` model has no certified optimum, so there is
                // no evidence for it to report and the path stops here with
                // the reason recorded rather than failing the whole fit.
                log::info!(
                    "[event-history] rank {rank} → {}: no certified optimum, refused ({error})",
                    rank + 1
                );
                rank_path.push(RankStep {
                    rank,
                    score_eigenvalue: atom.eigenvalue,
                    proposed_log_rate: atom.log_rate,
                    at_resolution_limit: atom.at_resolution_limit,
                    evidence_gain: 0.0,
                    accepted: false,
                    converged: false,
                });
                break;
            }
        };
        let gain = criterion - next_criterion;
        // Two converged criteria differ by more than the solver's own
        // resolution only if the atom carries evidence.
        let accepted = gain > criterion_resolution(options, criterion);
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
            evidence_gain: gain,
            accepted,
            converged: true,
        });
        if !accepted {
            break;
        }
        loadings = next_fit.block_states[marks].beta.to_vec();
        log_rates = next_rates;
        atom_evidence.push(gain);
        family = next_family;
        fit = next_fit;
    }
    Ok(RankState {
        family,
        fit,
        loadings,
        log_rates,
        atom_evidence,
        rank_path,
    })
}

/// The resolution of the outer criterion: the solver's tolerance, or the
/// roundoff of the criterion's magnitude when that is larger.
fn criterion_resolution(options: &BlockwiseFitOptions, criterion: f64) -> f64 {
    options.outer_tol.max(f64::EPSILON * criterion.abs())
}

/// Fit an event-history model to a cohort, growing the rank of the latent
/// covariance from zero until the evidence refuses the next direction.
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
    let time_scale = cohort.time_scale();
    // The bases are built on the outcome-free design rows and frozen, so no
    // event time enters a data-adaptive basis and a coefficient means the
    // same thing on every mesh.
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
        let frozen =
            crate::fit_orchestration::drivers::freeze_term_collection_from_design(term_spec, &design)
                .map_err(|error| EventHistoryError::Fit {
                    reason: format!("freezing mark {d} term collection: {error}"),
                })?;
        frozen_specs.push(frozen);
    }
    let nodes = Arc::new(expand_nodes(cohort, spec.quadrature_order, spec.mesh_refinement)?);
    let mut designs = Vec::with_capacity(marks);
    let mut dense = Vec::with_capacity(marks);
    let mut mark_specs = Vec::with_capacity(marks);
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
        mark_specs.push(mark_block_spec(&cohort.mark_names[d], &design));
        designs.push(design);
        dense.push(dense_design);
    }
    let state = grow_rank(&nodes, &dense, &mark_specs, marks, time_scale, &spec.options)?;
    Ok(assemble_fit(cohort, spec, nodes, designs, frozen_specs, state))
}

fn assemble_fit(
    cohort: &EventHistoryCohort,
    spec: &EventHistorySpec,
    nodes: Arc<CohortNodes>,
    designs: Vec<TermCollectionDesign>,
    frozen_specs: Vec<TermCollectionSpec>,
    state: RankState,
) -> EventHistoryFit {
    let marks = cohort.marks();
    let time_scale = cohort.time_scale();
    let atoms = state.log_rates.len();
    let mut loading_matrix = Array2::<f64>::zeros((marks, atoms));
    for d in 0..marks {
        for k in 0..atoms {
            loading_matrix[[d, k]] = state.loadings[d * atoms + k];
        }
    }
    let rates: Vec<f64> = state.log_rates.iter().map(|r| r.exp() / time_scale).collect();
    let n_lambda = state.fit.log_lambdas.len();
    let atom_log_lambdas: Vec<f64> = state
        .fit
        .log_lambdas
        .iter()
        .skip(n_lambda.saturating_sub(atoms))
        .copied()
        .collect();
    EventHistoryFit {
        nodes,
        family: state.family,
        fit: state.fit,
        frozen_specs,
        designs,
        mark_kinds: cohort.mark_kinds.clone(),
        loadings: loading_matrix,
        log_rates: state.log_rates,
        rates,
        atom_log_lambdas,
        atom_evidence: state.atom_evidence,
        rank_path: state.rank_path,
        time_scale,
        quadrature_order: spec.quadrature_order,
        mesh_refinement: spec.mesh_refinement,
    }
}

/// The block specs of the rank-`K+1` candidate: every block warm-started
/// from the rank-`K` fit and the new atom's loadings appended to the latent
/// block at the score's direction and one-step variance, with a ridge start
/// at the empirical-Bayes value for that start. The atom's rate is not in
/// the block: it is the rate the covariance score named, and the family
/// carries it as data.
fn grow(
    fit: &UnifiedFitResult,
    mark_specs: &[ParameterBlockSpec],
    loadings: &[f64],
    rank: usize,
    atom: &NewAtom,
    marks: usize,
    n_obs: usize,
) -> Result<Vec<ParameterBlockSpec>, EventHistoryError> {
    let atoms = rank + 1;
    let mut specs: Vec<ParameterBlockSpec> = mark_specs.to_vec();
    warm_start(&mut specs, fit);
    let mut beta = Array1::<f64>::zeros(marks * atoms);
    for d in 0..marks {
        for k in 0..rank {
            beta[d * atoms + k] = loadings[d * rank + k];
        }
        beta[d * atoms + rank] = atom.loading[d];
    }
    let n_lambda = fit.log_lambdas.len();
    let mut lambdas = Array1::<f64>::zeros(atoms);
    for k in 0..rank {
        lambdas[k] = fit.log_lambdas[n_lambda - rank + k];
    }
    // Empirical Bayes at the proposal: the ridge whose prior variance is the
    // start's own scale, `λ = marks / |a|²`.
    let start_norm: f64 = atom.loading.iter().map(|a| a * a).sum::<f64>();
    lambdas[rank] = (marks as f64 / start_norm.max(f64::MIN_POSITIVE)).ln();
    specs.push(latent_block_spec(n_obs, marks, atoms, beta, lambdas)?);
    Ok(specs)
}
