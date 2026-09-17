//! Assignment gates and sparsity-prior helpers for the SAE manifold term.
//! Mechanically split from `sae_manifold.rs`.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use crate::manifold::SaeManifoldRho;
use gam_terms::analytic_penalties::{
    AnalyticPenalty, OrderedBetaBernoulliHessianDiagThirdChannels,
    OrderedBetaBernoulliLogitAdjointData, OrderedBetaBernoulliPenalty,
    SoftmaxAssignmentSparsityPenalty, resolve_learnable_weight,
};
use gam_terms::latent::{LatentCoordValues, LatentIdMode, LatentManifold};

/// Shared per-atom row support measure.
///
/// The weights are the fitted assignment masses `w_i = a_{ik}` for one atom:
/// non-negative, unnormalised, and on the same scale as the reconstruction gate.
/// Diagnostics should read atom occupancy through this object instead of
/// re-deriving hard owner sets or local soft-mass sums. Three sizes are exposed
/// because they answer different questions:
///
/// * [`Self::mass`] is the soft occupancy `Σ_i w_i`.
/// * [`Self::fisher_n`] is the reconstruction-information count `Σ_i w_i²`,
///   matching the rank-charge Gram `Φᵀdiag(w²)Φ`.
/// * [`Self::ess`] is the scale-invariant Kish effective support
///   `(Σ_i w_i)² / Σ_i w_i²`, the number of equally weighted rows represented by
///   the support distribution.
#[derive(Clone, Debug)]
pub struct SupportMeasure {
    atom_idx: usize,
    weights: Array1<f64>,
    mass: f64,
    fisher_n: f64,
}

impl SupportMeasure {
    #[must_use = "support construction error must be handled"]
    pub(crate) fn from_assignment(assignment: &SaeAssignment, atom_idx: usize) -> Result<Self, String> {
        let assignments = assignment.assignments();
        Self::from_assignment_matrix(assignments.view(), atom_idx)
    }

    #[must_use = "support construction error must be handled"]
    pub(crate) fn from_assignment_matrix(
        assignments: ArrayView2<'_, f64>,
        atom_idx: usize,
    ) -> Result<Self, String> {
        let (_n, k) = assignments.dim();
        if atom_idx >= k {
            return Err(format!(
                "SupportMeasure::from_assignment_matrix: atom {atom_idx} out of range K={k}"
            ));
        }
        let weights = assignments.column(atom_idx).to_owned();
        Self::from_weights(atom_idx, weights)
    }

    #[must_use = "support construction error must be handled"]
    pub(crate) fn from_weights(atom_idx: usize, weights: Array1<f64>) -> Result<Self, String> {
        let mut mass = 0.0_f64;
        let mut fisher_n = 0.0_f64;
        for (row, &w) in weights.iter().enumerate() {
            if !(w.is_finite() && w >= 0.0) {
                return Err(format!(
                    "SupportMeasure::from_weights: row {row} has invalid support weight {w}"
                ));
            }
            mass += w;
            fisher_n += w * w;
        }
        Ok(Self {
            atom_idx,
            weights,
            mass,
            fisher_n,
        })
    }

    pub fn atom_idx(&self) -> usize {
        self.atom_idx
    }

    pub fn weights(&self) -> ArrayView1<'_, f64> {
        self.weights.view()
    }

    pub fn len(&self) -> usize {
        self.weights.len()
    }

    pub fn is_empty(&self) -> bool {
        self.weights.is_empty()
    }

    pub fn mass(&self) -> f64 {
        self.mass
    }

    pub fn fisher_n(&self) -> f64 {
        self.fisher_n
    }

    pub fn ess(&self) -> f64 {
        if self.fisher_n > 0.0 {
            (self.mass * self.mass) / self.fisher_n
        } else {
            0.0
        }
    }

    pub fn weight(&self, row: usize) -> f64 {
        self.weights[row]
    }

    pub fn positive_rows(&self) -> Vec<usize> {
        self.weights
            .iter()
            .enumerate()
            .filter_map(|(row, &w)| if w > 0.0 { Some(row) } else { None })
            .collect()
    }
}

/// #976 Layer-1 guard: cap on one accepted iteration's assignment-logit
/// update, in units of the gate temperature τ (the gate's natural length
/// scale — every assignment mode reads logits through `σ(·/τ)` /
/// `softmax(·/τ)`). A 4τ move spans the gate's whole soft range, so healthy
/// convergence is never throttled, but no single inner iteration can carry a
/// gate from contention to numerically-zero support: a collapse takes
/// multiple accepted iterations, which guarantees the per-iteration
/// active-mass guard observes the decay before it completes. The clamp is
/// applied where the step is realised; when it binds, the realised objective
/// is evaluated on the clamped state, so the Armijo comparison stays
/// value-consistent (the unclamped quadratic model is merely conservative,
/// and step halvings shrink the trial below the cap).
pub(crate) const SAE_ASSIGNMENT_LOGIT_STEP_CAP_TAUS: f64 = 4.0;

/// #976 Layer-1 guard: re-seed budget per atom per joint fit. One second
/// chance from a fresh basin; a second breach means the collapse is (locally)
/// the objective's verdict at the current hyperparameters, which is recorded
/// as a terminal collapse event and left for the structure-search death move
/// to adjudicate — re-seeding in a loop would fight the optimizer.
pub(crate) const SAE_ATOM_COLLAPSE_RESEED_BUDGET: usize = 1;

/// #976 Layer-1 guard (decoder arm): an atom whose decoder block Frobenius norm
/// has fallen to this fraction of the dictionary's MEDIAN decoder norm carries
/// no material reconstruction signal — it has degenerated to (near-)zero output
/// and decodes the same nothing as every other collapsed atom. This is the
/// real-data K>1 failure that the gate-mass floor cannot see: the assignment
/// gates can stay spread across rows (mass guard satisfied) while the decoders
/// all collapse to ~0, giving EV≈0 and a rank-deficient per-row coordinate
/// Hessian on every row (the 0→K·n evidence-deflation jump). The statistic is a
/// RATIO to the dictionary median so it is scale-free and never fires for a
/// uniformly-small but well-conditioned decoder; only an atom that has fallen
/// far behind its peers is caught. By construction this is a no-op for K=1
/// (a single atom has no peer to fall behind, and the median equals its own
/// norm), so the K=1 path is byte-for-byte unchanged.
pub(crate) const SAE_ATOM_DECODER_NORM_COLLAPSE_RATIO: f64 = 1.0e-3;

/// #976 / #1117 K>1 robustness: bounded DICTIONARY-level multi-start budget for
/// the simultaneous co-collapse arm of
/// [`crate::manifold::SaeManifoldTerm::enforce_decoder_norm_guard`]).
/// Distinct from the per-atom [`SAE_ATOM_COLLAPSE_RESEED_BUDGET`] (= 1): that
/// budget governs reseeding ONE atom's gate logits against an optimizer that
/// keeps killing it, where a loop would fight the optimizer. A co-collapse
/// reseed is categorically different — it is a full-dictionary multi-start that
/// re-diversifies ALL atoms onto distinct principal directions of a FRESHLY
/// recomputed residual, so successive attempts explore genuinely different
/// basins. A single such reseed empirically cannot always break a K≥3 three-way
/// basin (identical (K, seed) flips EV≈0.40 ↔ 0.00), so this arm gets a small
/// bounded budget of independent multi-starts. It is consumed only after
/// iteration zero when the same-state certificate proves that all gated decoder
/// signals disappeared at floating-point resolution or #2362 proves structural
/// union-output-span collapse. Training EV is telemetry, so a healthy or merely
/// uncompetitive live-decoder fit never consumes this budget.
pub(crate) const SAE_DICTIONARY_COCOLLAPSE_RESEED_BUDGET: usize = 3;

/// Assignment prior/relaxation used by [`SaeAssignment`].
#[derive(Debug, Clone, Copy)]
pub enum AssignmentMode {
    /// Row-wise simplex assignment with entropy sparsity.
    Softmax { temperature: f64, sparsity: f64 },
    /// Deterministic sigmoid relaxation for an ordered independent
    /// Beta--Bernoulli active set:
    /// `a_k = σ(logit_k/temperature)`. These are independent Bernoulli gates,
    /// not mixture/simplex responsibilities. The ordered geometric mean schedule
    /// `π_k = (α/(α+1))^{k+1}` is scored once by the ordered Beta--Bernoulli prior; it is not
    /// multiplied into the final reconstructed function.
    OrderedBetaBernoulli {
        temperature: f64,
        alpha: f64,
        learnable_alpha: bool,
    },
    /// Smooth threshold-centered logistic gate
    /// `a_k = σ((logit_k − threshold) / temperature)`. Magnitude lives in the
    /// decoder curve `g_k(t) = φ(t)ᵀB_k`; this gate supplies a bounded
    /// activation in `(0, 1)`. Its derivative is exact on both sides of the
    /// threshold, so fitted values, data-fit Jacobians, priors, and Hessians are
    /// derivatives of one smooth objective.
    ThresholdGate { temperature: f64, threshold: f64 },
    /// Hard top-`k` support gate: the `k` atoms with the LARGEST routing logits
    /// in a row carry gate 1, every other atom carries gate 0 (ties broken
    /// toward the lower atom index, so the support is deterministic).
    ///
    /// Sparsity is BY CONSTRUCTION, not by penalty: there is no sparsity term
    /// in the objective, no gate logit in the inner system
    /// (`assignment_coord_dim() == 0` — at K = 32,000 this deletes 32k
    /// coordinates from the inner Newton), and no sparsity coordinate in the
    /// outer ρ search. This is deterministic fixed-cardinality support, not a
    /// probabilistic prior or a MAP approximation to one. The gate is per-row
    /// independent (couples rows through NOTHING), so fits stream
    /// chunk-invariantly at any K, and it is exchangeable across atom index.
    TopK { k: usize },
}

impl AssignmentMode {
    /// The family's stable snake_case name, for refusal text and provenance.
    /// Matching on the enum here means a family added later cannot be reported
    /// as an anonymous "non-softmax prior" by any message that uses this.
    #[must_use]
    pub fn family_label(&self) -> &'static str {
        match self {
            Self::Softmax { .. } => "softmax",
            Self::OrderedBetaBernoulli { .. } => "ordered_beta_bernoulli",
            Self::ThresholdGate { .. } => "threshold_gate",
            Self::TopK { .. } => "topk",
        }
    }

    #[must_use]
    pub fn softmax(temperature: f64) -> Self {
        Self::Softmax {
            temperature,
            sparsity: 1.0,
        }
    }

    #[must_use]
    pub fn ordered_beta_bernoulli(temperature: f64, alpha: f64, learnable_alpha: bool) -> Self {
        Self::OrderedBetaBernoulli {
            temperature,
            alpha,
            learnable_alpha,
        }
    }

    /// Construct the smooth threshold-centered logistic [`Self::ThresholdGate`].
    #[must_use]
    pub fn threshold_gate(temperature: f64, threshold: f64) -> Self {
        Self::ThresholdGate {
            temperature,
            threshold,
        }
    }

    /// Construct the hard top-`k` support gate ([`Self::TopK`]): sparsity by
    /// construction, zero gate coordinates in the inner system, per-row
    /// independent. `k` is clamped to at least 1 by the fit-time validator.
    #[must_use]
    pub fn top_k_support(k: usize) -> Self {
        Self::TopK { k }
    }

    pub fn temperature(&self) -> f64 {
        match *self {
            AssignmentMode::Softmax { temperature, .. }
            | AssignmentMode::OrderedBetaBernoulli { temperature, .. }
            | AssignmentMode::ThresholdGate { temperature, .. } => temperature,
            // The hard support gate has no relaxation, hence no temperature; the
            // unit value keeps generic temperature-logging paths well-defined.
            AssignmentMode::TopK { .. } => 1.0,
        }
    }

    pub(crate) fn set_temperature(&mut self, new_temperature: f64) -> Result<(), String> {
        if !(new_temperature.is_finite() && new_temperature > 0.0) {
            return Err(format!(
                "AssignmentMode: temperature must be finite and positive; got {new_temperature}"
            ));
        }
        match self {
            AssignmentMode::Softmax { temperature, .. }
            | AssignmentMode::OrderedBetaBernoulli { temperature, .. }
            | AssignmentMode::ThresholdGate { temperature, .. } => {
                *temperature = new_temperature;
            }
            // No relaxation to anneal: the hard support is temperature-free, so
            // annealing schedules pass through as a no-op.
            AssignmentMode::TopK { .. } => {}
        }
        Ok(())
    }

    pub(crate) fn validate(&self) -> Result<(), String> {
        let temperature = self.temperature();
        if !(temperature.is_finite() && temperature > 0.0) {
            return Err(format!(
                "AssignmentMode: temperature must be finite and positive; got {temperature}"
            ));
        }
        match *self {
            AssignmentMode::Softmax { sparsity, .. } => {
                if !(sparsity.is_finite() && sparsity > 0.0) {
                    return Err(format!(
                        "AssignmentMode::Softmax: sparsity must be finite and positive; got {sparsity}"
                    ));
                }
            }
            AssignmentMode::OrderedBetaBernoulli { alpha, .. } => {
                if !(alpha.is_finite() && alpha > 0.0) {
                    return Err(format!(
                        "AssignmentMode::OrderedBetaBernoulli: alpha must be finite and positive; got {alpha}"
                    ));
                }
            }
            AssignmentMode::ThresholdGate { threshold, .. } => {
                if !threshold.is_finite() {
                    return Err(format!(
                        "AssignmentMode::ThresholdGate: threshold must be finite; got {threshold}"
                    ));
                }
            }
            AssignmentMode::TopK { k } => {
                if k == 0 {
                    return Err(
                        "AssignmentMode::TopK: support size k must be at least 1".to_string()
                    );
                }
            }
        }
        Ok(())
    }

    /// Resolve the effective ordered independent Beta--Bernoulli concentration `α` for this mode.
    ///
    /// `per_fit_override` is the #1777 PER-FIT override (from
    /// [`SaeAssignment::ordered_beta_bernoulli_alpha_override`]) and is the source of truth when set.
    /// Otherwise the mode's canonical fixed `α` or learnable schedule is used.
    pub(crate) fn resolved_ordered_beta_bernoulli_alpha(
        &self,
        rho: &SaeManifoldRho,
        per_fit_override: Option<f64>,
    ) -> Option<f64> {
        match *self {
            AssignmentMode::OrderedBetaBernoulli {
                alpha,
                learnable_alpha,
                ..
            } => Some(if let Some(over) = per_fit_override {
                // #1777 — the per-fit override flattens the ordered geometric
                // prior π_k = (α/(α+1))^{k+1}
                // so all K atoms can contribute to the reconstruction (the
                // production α=1 gives a (0.5)^{k+1} schedule that structurally
                // caps atoms 4..K → effective-K≈3). Forces the fixed value,
                // bypassing the learnable schedule.
                over
            } else if learnable_alpha {
                resolve_learnable_weight(alpha, rho.log_lambda_sparse)
                    .expect("ordered Beta--Bernoulli rho must be validated before resolution")
            } else {
                alpha
            }),
            _ => None,
        }
    }
}

/// A gate configuration the dense assignment state refuses to evaluate, because no
/// derivative in the crate differentiates its forward map.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GateConfigurationRefusal {
    /// #2933 F04 — softmax routing with ungated background atoms (see
    /// [`SaeAssignment::ungated`]).
    SoftmaxWithUngatedAtoms { ungated_atoms: Vec<usize> },
}

impl std::fmt::Display for GateConfigurationRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SoftmaxWithUngatedAtoms { ungated_atoms } => write!(
                f,
                "SaeAssignment: softmax routing refuses ungated atoms {ungated_atoms:?} \
                 (#2933 F04): pinning a gate at 1 after a softmax over all K atoms is not the \
                 map its logit JVP, entropy prior, logit Jacobian and reference-logit chart \
                 differentiate, and a unit background plus a simplex over only the gated atoms \
                 is not implemented"
            ),
        }
    }
}

/// Per-row latent assignment state — the DENSE-CERTIFICATION / debug-and-research
/// lane state only (#985 / E1), NOT the production route.
///
/// This is the dense `N×K` routing representation. The production SAE path is the
/// sparse-code lane ([`crate::sparse_dict`]), whose per-row state is fixed-width
/// `(indices, codes)` and never materializes an `N×K` assignment; large-K public
/// fits are routed there by the front door ([`crate::front_door::admit_sae_fit`] /
/// [`crate::front_door::admit_dense_certification`], #14). The dense manifold
/// engine that owns this type is reached only for the small-`K` certification lane
/// (`K ≤ P`) and for overcomplete research fits at small `N`. A source-guard test
/// (`sparse_lane_constructs_no_dense_assignment`) locks the invariant that the
/// sparse lane constructs zero `SaeAssignment`s; `#[doc(hidden)]` keeps this dense
/// state off the public API surface to match the demotion.
///
/// The stored assignment parameter is `logits`; non-negative assignments are
/// derived by row-wise softmax, independent ordered Beta--Bernoulli sigmoid active indicators,
/// or threshold gate gates. Softmax logits are canonicalized to the reference chart
/// `logits[K - 1] = 0`, so the row-local Newton coordinates contain only the
/// first `K - 1` logits (`0` coordinates for `K = 1`). Gate-style modes keep
/// all `K` logits as identifiable scalar parameters. `coords[k]` holds
/// `t_{.,k}` for atom `k`.
#[derive(Debug, Clone)]
pub struct SaeAssignment {
    pub logits: Array2<f64>,
    pub coords: Vec<LatentCoordValues>,
    pub mode: AssignmentMode,
    /// #1026 — per-atom UNGATED flag (length `K`, default all-`false`). An
    /// ungated atom is the dense linear/background tier: its per-row gate is
    /// fixed at `a_k ≡ 1` (it contributes `γ_k(t_k)` to EVERY row, unweighted),
    /// it is excluded from the other atoms' gate (for the column-separable
    /// ordered Beta--Bernoulli / threshold gate modes the remaining atoms are computed independently, so
    /// they are unaffected), and its logit is NOT a free parameter — its
    /// logit-JVP and sparsity-prior gradient/curvature are all zero, leaving its
    /// logit slot an inert (ridge-regularized) null direction in the per-row
    /// Newton block. This lets the linear tier carry FULL-RANK reconstructible
    /// variance (`fitted = γ_ungated(x) + Σ_{gated} a_k·γ_k(x)`) so a linear SAE
    /// can reach the rank-(K·d) PCA ceiling, while the gated curved atoms still
    /// add sparse structure on the residual (#1026 routing-bound finding).
    ///
    /// Softmax refuses ungated atoms ([`GateConfigurationRefusal`], #2933 F04).
    /// Its gates share one simplex, so pinning one of them at `1` after a softmax
    /// over all `K` atoms is not the map its logit JVP, entropy prior, logit
    /// Jacobian and reference-logit chart differentiate: with decoder values 2
    /// and 3, atom 1 ungated and zero logits, the forward map is `2σ(ℓ) + 3` with
    /// slope `0.5`, while the all-`K` softmax JVP reads `−1`. A unit background
    /// plus a simplex over only the gated atoms is a different model, with its
    /// own reference chart and jets, and is not implemented.
    pub ungated: Vec<bool>,
    /// #1033 — AMORTIZED / FROZEN routing. When `Some`, this `(n, K)` matrix is a
    /// ρ-INVARIANT predicted routing (the amortized `x → logits` map distilled
    /// from the frozen dictionary): the gates are computed from THESE logits
    /// instead of the free `self.logits`, and the logits are NOT optimized by the
    /// inner Newton (their gradient/curvature/prior contributions are zeroed,
    /// exactly as for [`Self::ungated`]). This is the generalization of an ungated
    /// atom from "pin the gate at 1" to "pin the gate at the predicted value": it
    /// makes the per-row routing a fixed function of `x` + the frozen dictionary,
    /// so the outer ρ-search reuses ONE routing instead of re-solving per-row
    /// gates every outer eval — the n-independent-outer-loop lever (#1033). `None`
    /// is the historical free-logit path (bit-identical).
    pub frozen_logits: Option<Array2<f64>>,
    /// #1777 PER-FIT ordered Beta--Bernoulli-α override. `Some(α)` forces a fixed value and bypasses
    /// the learnable schedule for this assignment/fit. `None` uses the
    /// [`AssignmentMode`]'s canonical fixed `α` or learnable schedule. Read via
    /// `Self::resolved_ordered_beta_bernoulli_alpha`; set from the FFI through the term's
    /// `set_fit_config`.
    pub ordered_beta_bernoulli_alpha_override: Option<f64>,
}

impl SaeAssignment {
    #[must_use = "build error must be handled"]
    pub fn new(
        logits: Array2<f64>,
        coords: Vec<LatentCoordValues>,
        temperature: f64,
    ) -> Result<Self, String> {
        Self::with_mode(logits, coords, AssignmentMode::softmax(temperature))
    }

    #[must_use = "build error must be handled"]
    pub(crate) fn with_mode(
        mut logits: Array2<f64>,
        coords: Vec<LatentCoordValues>,
        mode: AssignmentMode,
    ) -> Result<Self, String> {
        mode.validate()?;
        let n = logits.nrows();
        let k = logits.ncols();
        if coords.len() != k {
            return Err(format!(
                "SaeAssignment::new: coords length {} must equal K={k}",
                coords.len()
            ));
        }
        for (atom, coord) in coords.iter().enumerate() {
            if coord.n_obs() != n {
                return Err(format!(
                    "SaeAssignment::new: coord atom {atom} has n_obs={} but logits has {n}",
                    coord.n_obs()
                ));
            }
        }
        for row in 0..n {
            validate_finite_logits(logits.row(row), row)?;
        }
        if matches!(mode, AssignmentMode::Softmax { .. }) {
            canonicalize_softmax_logits(&mut logits);
        }
        Ok(Self {
            logits,
            coords,
            mode,
            ungated: vec![false; k],
            frozen_logits: None,
            ordered_beta_bernoulli_alpha_override: None,
        })
    }

    /// Whether the per-row routing is FROZEN (amortized) rather than free-logit.
    pub(crate) fn routing_is_frozen(&self) -> bool {
        self.frozen_logits.is_some()
    }

    /// The active routing logits for `row`: the frozen/predicted logits when
    /// routing is frozen (#1033), else the free `self.logits`. This is the SINGLE
    /// source the gate value reads, so freezing routing changes every gate
    /// consistently.
    pub(crate) fn routing_logits_row(&self, row: usize) -> ArrayView1<'_, f64> {
        match self.frozen_logits {
            Some(ref f) => f.row(row),
            None => self.logits.row(row),
        }
    }

    /// Whether atom `k`'s logit is held fixed (not a free Newton parameter): true
    /// for an ungated atom (#1026, gate pinned at 1) OR when routing is frozen
    /// (#1033, gate pinned at the predicted value). Both share the same inert
    /// treatment — zero logit-JVP, zero sparsity-prior gradient/curvature, zero
    /// softmax majorizer — so the logit slot never moves.
    pub(crate) fn logit_is_fixed(&self, k: usize) -> bool {
        // TopK: NO logit is ever a free Newton parameter — the support is a
        // deterministic function of the routing logits (assignment_coord_dim
        // is 0), so every gate rides as a constant, exactly like frozen routing.
        matches!(self.mode, AssignmentMode::TopK { .. })
            || self.routing_is_frozen()
            || self.ungated.get(k).copied().unwrap_or(false)
    }

    /// Per-atom mask (length `K`) of [`Self::logit_is_fixed`] — the logit slots
    /// that are NOT free Newton parameters (ungated #1026 and/or frozen-routing
    /// #1033). Precompute once per assembly and pass to the logit-JVP fillers so
    /// the data-fit Jacobian zeroes those rows. Under frozen routing every entry
    /// is `true`; with only ungated atoms it equals `ungated`; otherwise all
    /// `false` (the historical free-logit path).
    pub(crate) fn fixed_logit_mask(&self) -> Vec<bool> {
        if matches!(self.mode, AssignmentMode::TopK { .. }) || self.routing_is_frozen() {
            vec![true; self.k_atoms()]
        } else {
            self.ungated.clone()
        }
    }

    /// Whether any atom is ungated (the #1026 background tier is engaged).
    pub(crate) fn has_ungated(&self) -> bool {
        self.ungated.iter().any(|&u| u)
    }

    /// Refuse a gate configuration whose forward map no derivative in the crate
    /// differentiates. The forward seam and [`Self::validate_rho_domain`] call
    /// this, so the value path and every prior/operator channel stop at the
    /// same refusal. Frozen routing is not refused here: it holds every logit,
    /// so the forward gates and all their logit derivatives are constant.
    pub(crate) fn validate_gate_configuration(&self) -> Result<(), GateConfigurationRefusal> {
        if matches!(self.mode, AssignmentMode::Softmax { .. }) && self.has_ungated() {
            return Err(GateConfigurationRefusal::SoftmaxWithUngatedAtoms {
                ungated_atoms: (0..self.ungated.len())
                    .filter(|&atom| self.ungated[atom])
                    .collect(),
            });
        }
        Ok(())
    }

    pub fn n_obs(&self) -> usize {
        self.logits.nrows()
    }

    pub fn k_atoms(&self) -> usize {
        self.logits.ncols()
    }

    pub(crate) fn total_coord_dim(&self) -> usize {
        self.coords.iter().map(|c| c.latent_dim()).sum()
    }

    pub(crate) fn assignment_coord_dim(&self) -> usize {
        match self.mode {
            AssignmentMode::Softmax { .. } => self.k_atoms().saturating_sub(1),
            AssignmentMode::OrderedBetaBernoulli { .. } | AssignmentMode::ThresholdGate { .. } => {
                self.k_atoms()
            }
            // Sparsity by construction: the support is a deterministic function
            // of the routing logits, so there are NO free gate coordinates in
            // the inner system.
            AssignmentMode::TopK { .. } => 0,
        }
    }

    pub fn row_block_dim(&self) -> usize {
        self.assignment_coord_dim() + self.total_coord_dim()
    }

    pub(crate) fn coord_offsets(&self) -> Vec<usize> {
        let mut out = Vec::with_capacity(self.k_atoms());
        let mut cursor = self.assignment_coord_dim();
        for coord in &self.coords {
            out.push(cursor);
            cursor += coord.latent_dim();
        }
        out
    }

    pub fn assignments(&self) -> Array2<f64> {
        let n = self.n_obs();
        let k = self.k_atoms();
        let mut out = Array2::<f64>::zeros((n, k));
        for row in 0..n {
            let a = self.assignments_row(row);
            for atom in 0..k {
                out[[row, atom]] = a[atom];
            }
        }
        out
    }

    pub(crate) fn assignments_row(&self, row: usize) -> Array1<f64> {
        self.try_assignments_row(row)
            .expect("assignment logits must be finite")
    }

    pub fn try_assignments_row(&self, row: usize) -> Result<Array1<f64>, String> {
        self.try_assignments_row_inner(row)
    }

    /// #1777 — the effective ordered independent Beta--Bernoulli `α` for this assignment at `rho`,
    /// honoring the PER-FIT [`Self::ordered_beta_bernoulli_alpha_override`] before the mode's
    /// canonical value or learnable schedule. The single seam every
    /// gate/jet/prior site reads so the per-fit override is applied consistently.
    /// `None` for non-ordered Beta--Bernoulli modes.
    pub(crate) fn resolved_ordered_beta_bernoulli_alpha(
        &self,
        rho: &SaeManifoldRho,
    ) -> Option<f64> {
        self.mode
            .resolved_ordered_beta_bernoulli_alpha(rho, self.ordered_beta_bernoulli_alpha_override)
    }

    /// Whether the ordered independent Beta--Bernoulli concentration α is a FREE outer parameter that
    /// varies with ρ (`rho.log_lambda_sparse`). α is learnable ONLY when the mode
    /// requests it AND no per-fit override pins it: an override forces the fixed
    /// value and bypasses the learnable
    /// schedule (see [`AssignmentMode::resolved_ordered_beta_bernoulli_alpha`]), so α's ρ-derivatives
    /// are then identically zero and every prior / log-det / IFT term must treat α
    /// as a constant to stay consistent with the forward gate. `false` for non-ordered Beta--Bernoulli
    /// modes. (#Bug6)
    pub(crate) fn effective_alpha_is_learnable(&self) -> bool {
        match self.mode {
            AssignmentMode::OrderedBetaBernoulli {
                learnable_alpha, ..
            } => learnable_alpha && self.ordered_beta_bernoulli_alpha_override.is_none(),
            _ => false,
        }
    }

    pub(crate) fn validate_rho_domain(&self, rho: &SaeManifoldRho) -> Result<(), String> {
        self.validate_gate_configuration()
            .map_err(|refusal| refusal.to_string())?;
        rho.validate_log_strength_domain()?;
        if let AssignmentMode::OrderedBetaBernoulli {
            alpha,
            learnable_alpha: true,
            ..
        } = self.mode
            && self.ordered_beta_bernoulli_alpha_override.is_none()
        {
            resolve_learnable_weight(alpha, rho.log_lambda_sparse).map_err(|error| {
                format!("ordered Beta--Bernoulli learnable concentration: {error}")
            })?;
        }
        Ok(())
    }

    pub(crate) fn learnable_alpha_rho_domain(&self) -> Result<Option<(f64, f64)>, String> {
        let AssignmentMode::OrderedBetaBernoulli {
            alpha,
            learnable_alpha: true,
            ..
        } = self.mode
        else {
            return Ok(None);
        };
        if self.ordered_beta_bernoulli_alpha_override.is_some() {
            return Ok(None);
        }
        gam_terms::analytic_penalties::learnable_weight_coordinate_domain(alpha)
    }

    /// #1777 — install (or clear, with `None`) the PER-FIT ordered Beta--Bernoulli-α override on this
    /// assignment. Source of truth used by `Self::resolved_ordered_beta_bernoulli_alpha`; the FFI
    /// reaches it through the term's `set_fit_config`.
    pub(crate) fn set_ordered_beta_bernoulli_alpha_override(&mut self, alpha: Option<f64>) {
        self.ordered_beta_bernoulli_alpha_override = alpha;
    }

    /// Post-#1033 the row gates are ρ-INVARIANT (frozen/predicted or free
    /// routing logits never read ρ), so the assignment APIs take no ρ — the
    /// signatures state the invariance instead of threading a dead parameter.
    /// (A previous "wiring contract" rejected ρ whose per-atom width differed
    /// from `k_atoms()`, but K legitimately moves mid-fit — births, deaths,
    /// compaction, topology-race candidates — while ρ updates lag, so that
    /// contract vetoed valid states and broke seed validation fleet-wide;
    /// bisected to 6297a7e9f.)
    fn try_assignments_row_inner(&self, row: usize) -> Result<Array1<f64>, String> {
        // #1033 — read the ACTIVE routing logits: the ρ-invariant frozen/predicted
        // logits when routing is frozen, else the free `self.logits`. This single
        // source makes the gate value ρ-invariant under frozen routing (the
        // amortized-routing lever) and bit-identical to the historical path when
        // not frozen.
        let routing = self.routing_logits_row(row);
        validate_finite_logits(routing, row)?;
        self.validate_gate_configuration()
            .map_err(|refusal| refusal.to_string())?;
        // Only Softmax collapses to a fixed assignment at K==1: its
        // assignment_coord_dim is K-1 = 0, so there is no free logit. OrderedBetaBernoulli and
        // threshold gate keep a free per-atom gate logit even at K==1
        // (assignment_coord_dim = K = 1), so they must fall through to their real
        // row functions or the logit would move the prior but not the gate.
        if self.k_atoms() == 1 && matches!(self.mode, AssignmentMode::Softmax { .. }) {
            return Ok(Array1::from_vec(vec![1.0]));
        }
        let mut row_gates = match self.mode {
            AssignmentMode::Softmax { temperature, .. } => softmax_row(routing, temperature),
            AssignmentMode::OrderedBetaBernoulli { temperature, .. } => {
                ordered_beta_bernoulli_row(routing, temperature)
            }
            AssignmentMode::ThresholdGate {
                temperature,
                threshold,
            } => threshold_gate_row(routing, temperature, threshold),
            AssignmentMode::TopK { k } => topk_row(routing, k),
        };
        // #1026 — ungated (background-tier) atoms have a fixed unit gate. For the
        // column-separable ordered Beta--Bernoulli / threshold gate modes the other atoms' gates are
        // computed independently above, so overwriting the ungated entries to 1.0
        // leaves the gated atoms exactly as they were; the ungated atom then
        // contributes `γ_k(t_k)` unweighted to every row. Softmax + ungated was
        // refused above (#2933 F04): overwriting one simplex entry is not the map
        // the softmax derivatives differentiate.
        if self.has_ungated() {
            for (k, gate) in row_gates.iter_mut().enumerate() {
                if self.ungated[k] {
                    *gate = 1.0;
                }
            }
        }
        Ok(row_gates)
    }

    /// #1557 — fill-into-caller-buffer twin of [`Self::try_assignments_row`].
    ///
    /// `out` must have length `k_atoms()`; it is fully overwritten with the same
    /// values the allocating variant would return. Every branch (early-return
    /// K==1 Softmax, the per-mode row math, the #1026 ungated overwrite) mirrors
    /// the allocating path exactly so the two are bit-identical.
    pub(crate) fn try_assignments_row_into(
        &self,
        row: usize,
        out: &mut [f64],
    ) -> Result<(), String> {
        // `out` is sized `k_atoms()` by every caller; the per-mode helpers below
        // fully overwrite indices `0..k_atoms()`.
        let routing = self.routing_logits_row(row);
        validate_finite_logits(routing, row)?;
        self.validate_gate_configuration()
            .map_err(|refusal| refusal.to_string())?;
        // Mirror the allocating early-return: only Softmax collapses to a fixed
        // unit assignment at K==1.
        if self.k_atoms() == 1 && matches!(self.mode, AssignmentMode::Softmax { .. }) {
            out[0] = 1.0;
            return Ok(());
        }
        match self.mode {
            AssignmentMode::Softmax { temperature, .. } => {
                softmax_row_into(routing, temperature, out)
            }
            AssignmentMode::OrderedBetaBernoulli { temperature, .. } => {
                ordered_beta_bernoulli_row_into(routing, temperature, out)
            }
            AssignmentMode::ThresholdGate {
                temperature,
                threshold,
            } => threshold_gate_row_into(routing, temperature, threshold, out),
            AssignmentMode::TopK { k } => topk_row_into(routing, k, out),
        };
        // #1026 — ungated (background-tier) atoms have a fixed unit gate, exactly
        // as in the allocating path.
        if self.has_ungated() {
            for (k, gate) in out.iter_mut().enumerate() {
                if self.ungated[k] {
                    *gate = 1.0;
                }
            }
        }
        Ok(())
    }

    pub(crate) fn persist_resolved_ordered_beta_bernoulli_alpha(
        &mut self,
        rho: &SaeManifoldRho,
    ) -> bool {
        let AssignmentMode::OrderedBetaBernoulli {
            temperature,
            alpha,
            learnable_alpha: true,
        } = self.mode
        else {
            return false;
        };
        let resolved_alpha = resolve_learnable_weight(alpha, rho.log_lambda_sparse)
            .expect("ordered Beta--Bernoulli rho must be validated before persistence");
        self.mode = AssignmentMode::OrderedBetaBernoulli {
            temperature,
            alpha: resolved_alpha,
            learnable_alpha: false,
        };
        true
    }

    pub(crate) fn try_assignments(&self) -> Result<Array2<f64>, String> {
        let n = self.n_obs();
        let k = self.k_atoms();
        let mut out = Array2::<f64>::zeros((n, k));
        for row in 0..n {
            let a = self.try_assignments_row(row)?;
            for atom in 0..k {
                out[[row, atom]] = a[atom];
            }
        }
        Ok(out)
    }

    /// Flatten extension coordinates in row-major SAE layout:
    /// `(assignment chart_i, t_i0[0..d_0], ..., t_iK[0..d_K])` for every row.
    /// Softmax contributes the first `K - 1` reference logits and omits the
    /// fixed reference logit; gate-style assignment modes contribute all `K`
    /// logits.
    pub(crate) fn flatten_ext_coords(&self) -> Array1<f64> {
        let n = self.n_obs();
        let q = self.row_block_dim();
        let k = self.k_atoms();
        let assignment_dim = self.assignment_coord_dim();
        let offsets = self.coord_offsets();
        let mut out = Array1::<f64>::zeros(n * q);
        for row in 0..n {
            let base = row * q;
            for atom in 0..assignment_dim {
                out[base + atom] = self.logits[[row, atom]];
            }
            for atom in 0..k {
                let d = self.coords[atom].latent_dim();
                let t_row = self.coords[atom].row(row);
                for axis in 0..d {
                    out[base + offsets[atom] + axis] = t_row[axis];
                }
            }
        }
        out
    }

    #[must_use = "build error must be handled"]
    pub fn from_blocks_with_mode_and_manifolds(
        logits: Array2<f64>,
        coord_blocks: Vec<Array2<f64>>,
        manifolds: Vec<LatentManifold>,
        mode: AssignmentMode,
    ) -> Result<Self, String> {
        if coord_blocks.len() != manifolds.len() {
            return Err(format!(
                "SaeAssignment::from_blocks_with_mode_and_manifolds: coord block length {} != manifold length {}",
                coord_blocks.len(),
                manifolds.len()
            ));
        }
        let coords = coord_blocks
            .iter()
            .zip(manifolds)
            .map(|(c, manifold)| {
                LatentCoordValues::from_matrix_with_manifold(c.view(), LatentIdMode::None, manifold)
            })
            .collect();
        Self::with_mode(logits, coords, mode)
    }
}

pub(crate) fn softmax_row(logits: ArrayView1<'_, f64>, temperature: f64) -> Array1<f64> {
    let k = logits.len();
    let inv_tau = 1.0 / temperature;
    let mut max_logit = f64::NEG_INFINITY;
    for &v in logits.iter() {
        max_logit = max_logit.max(v);
    }
    let mut out = Array1::<f64>::zeros(k);
    let mut sum = 0.0;
    for i in 0..k {
        let v = ((logits[i] - max_logit) * inv_tau).exp();
        out[i] = v;
        sum += v;
    }
    assert!(sum.is_finite() && sum > 0.0);
    for v in out.iter_mut() {
        *v /= sum;
    }
    out
}

pub(crate) fn validate_finite_logits(
    logits: ArrayView1<'_, f64>,
    row: usize,
) -> Result<(), String> {
    for (col, &v) in logits.iter().enumerate() {
        if !v.is_finite() {
            return Err(format!(
                "SaeAssignment: non-finite assignment logit at row {row}, atom {col}: {v}"
            ));
        }
    }
    Ok(())
}

pub(crate) fn canonicalize_softmax_logits(logits: &mut Array2<f64>) {
    let k = logits.ncols();
    if k == 0 {
        return;
    }
    if k == 1 {
        logits.fill(0.0);
        return;
    }
    for row in 0..logits.nrows() {
        let reference = logits[[row, k - 1]];
        for col in 0..k - 1 {
            logits[[row, col]] -= reference;
        }
        logits[[row, k - 1]] = 0.0;
    }
}

/// #1784 — K-aware default ordered Beta--Bernoulli concentration.
///
/// The independent-Beta prior-mean schedule `μ_k = (α/(α+1))^{k+1}` decays
/// GEOMETRICALLY in the atom INDEX, so a fixed small concentration (the
/// historical default `α = 1`, i.e. the `(0.5)^{k+1}` schedule) collapses to a
/// near-hard mask past atom ~3: a K-atom dictionary can then only ever place
/// mass on its first handful of atoms. That is exactly why the manifold SAE
/// UNDERFITS a linear dictionary of equal K on real activations, and why its
/// late atoms carry zero mass and leave the per-row joint Hessian rank-deficient
/// (the K = 128 `RemlConvergenceError`).
///
/// For a K-atom dictionary to actually USE all K atoms the ordered Beta--Bernoulli concentration must
/// scale with K. Choosing `α` so the LAST atom retains prior mass
/// `π_{K-1} = (α/(α+1))^K ≈ e^{-1}` spans the whole dictionary while keeping the
/// prior monotone (no atom is structurally masked). Solving
/// `(α/(α+1))^K = e^{-1}` gives
/// `α = 1/(exp(1/K) − 1) ≈ K − 1/2`. Floored at `1.0` so `K = 1` keeps the
/// historical `α = 1`.
pub fn default_ordered_beta_bernoulli_concentration_for_k_atoms(k_atoms: usize) -> f64 {
    let k = k_atoms.max(1) as f64;
    // π_{K-1} = (α/(α+1))^K = e^{-1}  ⇒  α = 1/(e^{1/K} − 1).
    let alpha = 1.0 / ((1.0 / k).exp() - 1.0);
    alpha.max(1.0)
}

/// Sigmoid activations for the ordered Beta--Bernoulli assignment model.
///
/// Ordered shrinkage belongs to the Beta--Bernoulli prior scored by
/// [`OrderedBetaBernoulliPenalty`], not as a second multiplicative factor on the final
/// reconstruction. Multiplying by the prior mean capped atom `k` at `mu_k < 1`
/// even when its learned gate approached one, double-counted the prior and made
/// the fitted function depend on atom index. The reconstruction gate is simply
/// `sigmoid(logit_k / temperature)`.
pub fn ordered_beta_bernoulli_row(logits: ArrayView1<'_, f64>, temperature: f64) -> Array1<f64> {
    let mut out = Array1::<f64>::zeros(logits.len());
    for i in 0..logits.len() {
        out[i] = gam_linalg::utils::stable_logistic(logits[i] / temperature);
    }
    out
}

pub fn threshold_gate_row(
    logits: ArrayView1<'_, f64>,
    temperature: f64,
    threshold: f64,
) -> Array1<f64> {
    let mut out = Array1::<f64>::zeros(logits.len());
    for i in 0..logits.len() {
        out[i] = gam_linalg::utils::stable_logistic((logits[i] - threshold) / temperature);
    }
    out
}

/// Hard top-`k` support row (the [`AssignmentMode::TopK`] gate): 1.0 for the
/// `k` largest routing logits in the row, 0.0 elsewhere. Ties break toward the
/// LOWER atom index so the support is deterministic. `k ≥ len` degenerates to
/// the all-active row. Logits are validated finite upstream.
pub fn topk_row(logits: ArrayView1<'_, f64>, k: usize) -> Array1<f64> {
    let mut out = Array1::<f64>::zeros(logits.len());
    topk_row_into(
        logits,
        k,
        out.as_slice_mut()
            .expect("freshly allocated 1-D array is contiguous"),
    );
    out
}

/// Fill-into-caller-buffer twin of [`topk_row`] — bit-identical values, no
/// allocation beyond the O(K) index scratch. Average O(K) via quickselect.
pub(crate) fn topk_row_into(logits: ArrayView1<'_, f64>, k: usize, out: &mut [f64]) {
    let n = logits.len();
    if k >= n {
        out[..n].fill(1.0);
        return;
    }
    out[..n].fill(0.0);
    let mut idx: Vec<usize> = (0..n).collect();
    // Larger logit first; equal logits fall back to index order so the
    // boundary atom is deterministic across runs and chunkings.
    idx.select_nth_unstable_by(k, |&a, &b| {
        logits[b]
            .partial_cmp(&logits[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    for &i in &idx[..k] {
        out[i] = 1.0;
    }
}

/// Exact numerical inverse of the softplus link `softplus(x) = log(1 + eˣ)`
/// (the forward direction is [`gam_linalg::utils::stable_softplus`], used by
/// the penalty implementations). This is the single source of truth for the
/// softplus⁻¹ reparameterization the SAE penalty FFI uses to map
/// a positive scale hyperparameter `β > 0` back to its raw pre-softplus
/// coordinate `raw = softplus⁻¹(β)` (the `raw_beta` of the parametric
/// row-precision / aux-conditional priors). Moved out of the pyffi shim
/// (`geometry_ffi::inverse_softplus_scalar`) so no numeric policy lives in the
/// FFI layer.
///
/// Domain / stability contract (preserved exactly from the shim):
///   * `value ≤ 0` or `NaN` → `NaN` (softplus is strictly positive, so its
///     inverse is undefined off the positive reals);
///   * `value > 30` uses the overflow-safe identity
///     `softplus⁻¹(v) = v + log1p(−e^{−v})` (`eᵛ` would overflow);
///   * otherwise the direct `log(e^v − 1) = ln(expm1(v))`.
#[must_use]
pub fn inverse_softplus(value: f64) -> f64 {
    if value <= 0.0 || value.is_nan() {
        f64::NAN
    } else if value > 30.0 {
        value + (-(-value).exp()).ln_1p()
    } else {
        value.exp_m1().ln()
    }
}

#[cfg(test)]
mod topk_support_gate_tests {
    // Contract tests for the [`AssignmentMode::TopK`] hard-support gate: the
    // support is EXACTLY the k largest routing logits (deterministic lower-index
    // tie-break), L0 is exactly k, the fill-into twin is bit-identical, and the
    // all-equal neutral support is the first k atoms.
    use super::*;

    #[test]
    fn topk_row_selects_exact_support_and_l0_is_k() {
        let logits = Array1::from(vec![0.3_f64, 0.9, 0.9, -1.0, 0.5]);
        let g = topk_row(logits.view(), 3);
        assert_eq!(g.to_vec(), vec![0.0, 1.0, 1.0, 0.0, 1.0]);
        assert_eq!(
            g.iter().filter(|&&v| v == 1.0).count(),
            3,
            "L0 must equal k exactly"
        );
        assert!(
            g.iter().all(|&v| v == 0.0 || v == 1.0),
            "gates are hard {{0,1}}"
        );
    }

    #[test]
    fn topk_boundary_tie_breaks_toward_lower_index() {
        let logits = Array1::from(vec![1.0_f64, 0.5, 0.5, 0.1]);
        let g = topk_row(logits.view(), 2);
        assert_eq!(
            g.to_vec(),
            vec![1.0, 1.0, 0.0, 0.0],
            "the tied boundary atom with the LOWER index wins deterministically"
        );
    }

    #[test]
    fn topk_row_into_is_bit_identical_and_k_ge_n_is_all_active() {
        let logits = Array1::from(vec![-0.2_f64, 3.0, 0.7, 0.7, -5.0, 2.2]);
        for k in [1usize, 2, 4, 6, 9] {
            let alloc = topk_row(logits.view(), k);
            let mut buf = vec![f64::NAN; logits.len()];
            topk_row_into(logits.view(), k, &mut buf);
            assert_eq!(
                alloc.to_vec(),
                buf,
                "into-twin must be bit-identical at k={k}"
            );
        }
        let all = topk_row(logits.view(), 99);
        assert!(
            all.iter().all(|&v| v == 1.0),
            "k >= n degenerates to all-active"
        );
    }

    #[test]
    fn topk_mode_carries_no_temperature_or_prior_knobs() {
        let mode = AssignmentMode::top_k_support(4);
        mode.validate().expect("k >= 1 validates");
        assert!(
            AssignmentMode::top_k_support(0).validate().is_err(),
            "k = 0 must be rejected"
        );
    }
}

// #1557 — fill-into-caller-buffer variants of the three per-mode row functions.
// These compute the EXACT SAME values as `softmax_row` / `ordered_beta_bernoulli_row` /
// `threshold_gate_row` (same arithmetic, same order of operations) but write into a
// caller-provided `&mut [f64]` slice instead of heap-allocating a fresh
// `Array1<f64>` per call. The hot per-row loops (loss eval, arrow/Schur row
// loops) call these with a reused scratch buffer, eliminating millions of tiny
// K-sized allocations while staying bit-identical to the allocating path.
// `out` must have length `logits.len()`; the slice is fully overwritten.

pub(crate) fn softmax_row_into(logits: ArrayView1<'_, f64>, temperature: f64, out: &mut [f64]) {
    let k = logits.len();
    let inv_tau = 1.0 / temperature;
    let mut max_logit = f64::NEG_INFINITY;
    for &v in logits.iter() {
        max_logit = max_logit.max(v);
    }
    let mut sum = 0.0;
    for i in 0..k {
        let v = ((logits[i] - max_logit) * inv_tau).exp();
        out[i] = v;
        sum += v;
    }
    assert!(sum.is_finite() && sum > 0.0);
    for v in out.iter_mut() {
        *v /= sum;
    }
}

pub(crate) fn ordered_beta_bernoulli_row_into(
    logits: ArrayView1<'_, f64>,
    temperature: f64,
    out: &mut [f64],
) {
    for i in 0..logits.len() {
        out[i] = gam_linalg::utils::stable_logistic(logits[i] / temperature);
    }
}

pub(crate) fn threshold_gate_row_into(
    logits: ArrayView1<'_, f64>,
    temperature: f64,
    threshold: f64,
    out: &mut [f64],
) {
    for i in 0..logits.len() {
        out[i] = gam_linalg::utils::stable_logistic((logits[i] - threshold) / temperature);
    }
}

pub(crate) fn fill_assignment_logit_jvp_rows(
    mode: AssignmentMode,
    logits: ArrayView1<'_, f64>,
    assignments: ArrayView1<'_, f64>,
    decoded: ArrayView2<'_, f64>,
    fitted: ArrayView1<'_, f64>,
    // #1026 — per-atom ungated flags (length `K`). An ungated atom's gate is
    // constant, so its logit-JVP row is identically zero (skipped below). Empty
    // ⇒ no atom is ungated (the historical path, bit-identical).
    ungated: &[bool],
    local_jac: &mut Array2<f64>,
) {
    let is_ungated = |k: usize| ungated.get(k).copied().unwrap_or(false);
    match mode {
        AssignmentMode::Softmax { temperature, .. } => {
            if assignments.len() == 1 {
                return;
            }
            // da_k/dl_j = a_k (1[k=j] - a_j) / tau, contracted against
            // the assignment-weighted fitted row. The dense row layout uses
            // the reference-logit chart, so only columns `0..K-1` are free;
            // the final reference logit is fixed at zero and has no row.
            // `fitted` is `Σ_k a_k γ_k` over the whole simplex, which is only
            // the right contraction because the forward seam refuses a softmax
            // with an ungated background (#2933 F04); the skipped columns here
            // are frozen routing's, where every logit is held.
            let inv_tau = 1.0 / temperature;
            for logit_col in 0..assignments.len() - 1 {
                if is_ungated(logit_col) {
                    continue;
                }
                for out_col in 0..fitted.len() {
                    local_jac[[logit_col, out_col]] = assignments[logit_col]
                        * (decoded[[logit_col, out_col]] - fitted[out_col])
                        * inv_tau;
                }
            }
        }
        AssignmentMode::OrderedBetaBernoulli { temperature, .. } => {
            // Posterior-mean Bernoulli gate `z_k = σ(l_k/τ)`; independent-Beta
            // shrinkage is scored once, in the ordered Beta--Bernoulli prior.
            let inv_tau = 1.0 / temperature;
            for logit_col in 0..assignments.len() {
                if is_ungated(logit_col) {
                    continue;
                }
                let a_k = assignments[logit_col];
                let dz = a_k * (1.0 - a_k) * inv_tau;
                for out_col in 0..fitted.len() {
                    local_jac[[logit_col, out_col]] = dz * decoded[[logit_col, out_col]];
                }
            }
        }
        AssignmentMode::ThresholdGate {
            temperature,
            threshold,
        } => {
            // Exact derivative of the smooth threshold-centered logistic gate.
            let inv_tau = 1.0 / temperature;
            for logit_col in 0..assignments.len() {
                if is_ungated(logit_col) {
                    continue;
                }
                let activation =
                    gam_linalg::utils::stable_logistic((logits[logit_col] - threshold) * inv_tau);
                let da = activation * (1.0 - activation) * inv_tau;
                for out_col in 0..fitted.len() {
                    local_jac[[logit_col, out_col]] = da * decoded[[logit_col, out_col]];
                }
            }
        }
        // Constant {0, 1} gates: zero data-fit logit derivative everywhere (no
        // logit is a free parameter — the caller's fixed-logit mask already
        // skips every column; this arm keeps the JVP identically zero).
        AssignmentMode::TopK { .. } => {}
    }
}

pub(crate) fn flat_logits(logits: ArrayView2<'_, f64>) -> Array1<f64> {
    let mut out = Array1::<f64>::zeros(logits.len());
    for row in 0..logits.nrows() {
        let start = row * logits.ncols();
        for col in 0..logits.ncols() {
            out[start + col] = logits[[row, col]];
        }
    }
    out
}

/// Build the ordered Beta--Bernoulli sparsity penalty used by every assignment-prior term at `rho`,
/// honoring #Bug6 (α is FIXED to the forward-gate value whenever an override
/// pins it — `effective_alpha_is_learnable`, `resolved_ordered_beta_bernoulli_alpha`) and #Bug4
/// (ungated atoms are inert columns excluded from value/gradient/curvature).
/// Returns `(penalty, rho_view)`; the fixed-α branch uses the `lambda_sparse`
/// weight convention with an empty `rho_view`.
fn ordered_beta_bernoulli_prior_penalty(
    assignment: &SaeAssignment,
    rho: &SaeManifoldRho,
    base_alpha: f64,
    temperature: f64,
    row_weights: Option<&[f64]>,
) -> Result<(OrderedBetaBernoulliPenalty, Array1<f64>), String> {
    let learnable = assignment.effective_alpha_is_learnable();
    let alpha_eff = if learnable {
        base_alpha
    } else {
        assignment
            .resolved_ordered_beta_bernoulli_alpha(rho)
            .unwrap_or(base_alpha)
    };
    // #991 design-honesty weights: the ordered Beta--Bernoulli prior is not row-separable (the
    // exact integrated scalar couples rows through the column active mass), so the weights are
    // installed ON the penalty — its value/grad/hessian/hvp/ρ- and third
    // channels all fold them identically (weighted mass `M_k = Σ w_i z_ik` and
    // active-mass Jacobian `u = w·J`), keeping every channel the exact derivative of one
    // weighted energy. `None` gives the unit-weight operator.
    let mut penalty =
        OrderedBetaBernoulliPenalty::new(assignment.k_atoms(), alpha_eff, temperature, learnable)
            .with_row_weights(row_weights);
    // #Bug4: ungated atoms have a pinned unit gate and a held-constant logit — they
    // are inert columns excluded from the sparsity energy and all its derivatives.
    if assignment.has_ungated() {
        penalty.fixed_columns = Some(assignment.ungated.clone());
    }
    let rho_view = if learnable {
        Array1::from_vec(vec![rho.log_lambda_sparse])
    } else {
        penalty.weight = rho.lambda_sparse()?;
        Array1::zeros(0)
    };
    Ok((penalty, rho_view))
}

/// Apply the exact ordered Beta--Bernoulli logit Hessian minus the diagonal
/// PSD majorizer installed in the Newton/Laplace operator.
///
/// The exact integrated marginal contributes a dense-within-column Hessian:
/// a negative rank-one active-mass term plus a row-local concrete-Jacobian
/// diagonal. The assembled operator keeps only the positive part of that
/// diagonal, because zero is a PSD Loewner majorizer of the negative rank-one
/// term. The stationarity IFT must nevertheless invert the exact scalar
/// Hessian, so `A - B` is applied here analytically and matrix-free. No dense
/// `N K × N K` matrix or persistent low-rank carrier is constructed.
/// #2330 Patch D — the ordered-Beta--Bernoulli prior structural data the exact-A
/// θ-adjoint contracts for `∂ΔC_obb/∂ℓ` (see
/// `OrderedBetaBernoulliPenalty::logit_theta_adjoint_data`). `None` when the mode
/// is not ordered-Beta--Bernoulli or routing is frozen (the prior curvature is
/// then ρ/θ-inert). The cache-layout contraction lives in gam-sae.
pub(crate) fn ordered_beta_bernoulli_logit_adjoint_data_weighted(
    assignment: &SaeAssignment,
    rho: &SaeManifoldRho,
    row_weights: Option<&[f64]>,
) -> Result<Option<OrderedBetaBernoulliLogitAdjointData>, String> {
    assignment.validate_rho_domain(rho)?;
    let AssignmentMode::OrderedBetaBernoulli {
        temperature, alpha, ..
    } = assignment.mode
    else {
        return Ok(None);
    };
    if assignment.routing_is_frozen() {
        return Ok(None);
    }
    for row in 0..assignment.n_obs() {
        validate_finite_logits(assignment.logits.row(row), row)?;
    }
    let (penalty, rho_view) =
        ordered_beta_bernoulli_prior_penalty(assignment, rho, alpha, temperature, row_weights)?;
    let target = flat_logits(assignment.logits.view());
    Ok(Some(
        penalty.logit_theta_adjoint_data(target.view(), rho_view.view()),
    ))
}

pub(crate) fn ordered_beta_bernoulli_exact_hessian_minus_majorizer_hvp_weighted(
    assignment: &SaeAssignment,
    rho: &SaeManifoldRho,
    row_weights: Option<&[f64]>,
    direction: ArrayView1<'_, f64>,
) -> Result<Array1<f64>, String> {
    assignment.validate_rho_domain(rho)?;
    let AssignmentMode::OrderedBetaBernoulli {
        temperature, alpha, ..
    } = assignment.mode
    else {
        return Err(
            "ordered Beta--Bernoulli exact-Hessian correction requires ordered assignment mode"
                .to_string(),
        );
    };
    let target = flat_logits(assignment.logits.view());
    if direction.len() != target.len() {
        return Err(format!(
            "ordered Beta--Bernoulli exact-Hessian direction has length {}; expected {}",
            direction.len(),
            target.len()
        ));
    }
    if !direction.iter().all(|value| value.is_finite()) {
        return Err("ordered Beta--Bernoulli exact-Hessian direction must be finite".to_string());
    }
    if assignment.routing_is_frozen() {
        return Ok(Array1::<f64>::zeros(target.len()));
    }
    for row in 0..assignment.n_obs() {
        validate_finite_logits(assignment.logits.row(row), row)?;
    }

    let (penalty, rho_view) =
        ordered_beta_bernoulli_prior_penalty(assignment, rho, alpha, temperature, row_weights)?;
    let mut delta = penalty.hvp(target.view(), rho_view.view(), direction);
    let channels = penalty.psd_majorizer_logit_third_channels(target.view(), rho_view.view());
    for index in 0..delta.len() {
        delta[index] -= channels.diagonal_term[index].max(0.0) * direction[index];
    }
    Ok(delta)
}

#[cfg(test)]
mod ordered_beta_bernoulli_exact_hessian_tests {
    use super::*;

    #[test]
    fn exact_hessian_minus_majorizer_hvp_matches_gradient_fd_and_keeps_cross_row_term() {
        let n = 4usize;
        let k = 2usize;
        let logits =
            Array2::from_shape_vec((n, k), vec![0.2, -0.3, 0.7, -0.1, 0.4, 0.5, -0.2, 0.6])
                .unwrap();
        let coords = vec![Array2::<f64>::zeros((n, 1)); k];
        let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
            logits,
            coords,
            vec![LatentManifold::Euclidean; k],
            AssignmentMode::ordered_beta_bernoulli(0.8, 1.7, false),
        )
        .unwrap();
        let rho = SaeManifoldRho::new(1.3_f64.ln(), 0.0, vec![Array1::zeros(1); k]);
        // Excite one logit only. The exact integrated marginal must still
        // produce nonzero output on other rows of the same atom column.
        let mut direction = Array1::<f64>::zeros(n * k);
        direction[0] = 0.7;
        let analytic = ordered_beta_bernoulli_exact_hessian_minus_majorizer_hvp_weighted(
            &assignment,
            &rho,
            None,
            direction.view(),
        )
        .unwrap();

        let (penalty, rho_view) =
            ordered_beta_bernoulli_prior_penalty(&assignment, &rho, 1.7, 0.8, None).unwrap();
        let target = flat_logits(assignment.logits.view());
        let step = 1.0e-6;
        let plus = &target + &(step * &direction);
        let minus = &target - &(step * &direction);
        let gradient_plus = penalty.grad_target(plus.view(), rho_view.view());
        let gradient_minus = penalty.grad_target(minus.view(), rho_view.view());
        let channels = penalty.psd_majorizer_logit_third_channels(target.view(), rho_view.view());
        for index in 0..analytic.len() {
            let exact_fd = (gradient_plus[index] - gradient_minus[index]) / (2.0 * step);
            let expected = exact_fd - channels.diagonal_term[index].max(0.0) * direction[index];
            assert!(
                (analytic[index] - expected).abs() <= 2.0e-7,
                "index {index}: analytic A-B={} expected={} exact_fd={exact_fd}",
                analytic[index],
                expected,
            );
        }
        assert!(
            analytic[2].abs() > 1.0e-6 && analytic[4].abs() > 1.0e-6,
            "a one-row direction must produce the exact cross-row rank-one action: {analytic:?}"
        );
    }
}

/// The assignment prior's value with #991 design-honesty per-row weights:
/// row `i`'s per-row prior contribution is scaled by `w_i` (mean-1). This is the
/// per-row latent prior's analog of the `√w_i`-weighted data likelihood and the
/// `w_i`-weighted `ard_value` — each retained row of a design-honest subsample
/// stands in for `w_i` population rows, so its routing prior carries `w_i` too.
/// `None` gives the unit-weight path. Softmax/threshold gate are row-separable;
/// ordered Beta--Bernoulli instead forms weighted active mass and effective row
/// count inside its integrated scalar. Every derivative uses that same measure.
///
/// The ThresholdGate value is the normalized negative log density `λz + log Z(λ)` per free
/// gate ([`ThresholdGateLogPartition`]), so its strength derivative carries `∂_ρ log Z`.
pub(crate) fn assignment_prior_value_weighted(
    assignment: &SaeAssignment,
    rho: &SaeManifoldRho,
    row_weights: Option<&[f64]>,
) -> Result<f64, String> {
    assignment.validate_rho_domain(rho)?;
    for row in 0..assignment.n_obs() {
        validate_finite_logits(assignment.logits.row(row), row)?;
    }
    let target = flat_logits(assignment.logits.view());
    if matches!(assignment.mode, AssignmentMode::Softmax { .. }) && assignment.k_atoms() == 1 {
        return Ok(0.0);
    }
    // #Bug4: under FROZEN routing every logit is inert (the gates come from the
    // ρ-invariant frozen predictor, not `self.logits`), so the whole assignment
    // sparsity prior is a constant with zero gradient/curvature — score it as 0 to
    // match the derivative-side treatment. (Softmax rejects frozen routing.)
    if assignment.routing_is_frozen() {
        return Ok(0.0);
    }
    Ok(match assignment.mode {
        AssignmentMode::Softmax {
            temperature,
            sparsity,
        } => {
            let penalty = SoftmaxAssignmentSparsityPenalty::new(assignment.k_atoms(), temperature)
                .with_row_weights(row_weights);
            let rho_view = Array1::from_vec(vec![rho.log_lambda_sparse + sparsity.ln()]);
            penalty.value(target.view(), rho_view.view())
        }
        AssignmentMode::OrderedBetaBernoulli {
            temperature, alpha, ..
        } => {
            let (penalty, rho_view) = ordered_beta_bernoulli_prior_penalty(
                assignment,
                rho,
                alpha,
                temperature,
                row_weights,
            )?;
            penalty.value(target.view(), rho_view.view())
        }
        AssignmentMode::ThresholdGate {
            temperature,
            threshold,
        } => {
            // Sparsity penalty and reconstruction use the same smooth
            // threshold-centered logistic gate as the gradient and Hessian.
            let sparsity_strength = rho.lambda_sparse()?;
            let gates = threshold_gate_free_gates(
                assignment,
                target.view(),
                threshold,
                temperature,
                row_weights,
            );
            sparsity_strength * gates.weighted_activation
                + gates.weight * ThresholdGateLogPartition::eval(sparsity_strength).value()
        }
        // Sparsity by construction: the fixed-|S| support IS the sparsity — there
        // is no penalty term, so the prior contributes exactly zero.
        AssignmentMode::TopK { .. } => 0.0,
    })
}

/// #991-weighted derivative of the assignment prior in its log strength. Every
/// assignment mode differentiates the same weighted scalar used by its value path.
pub(crate) fn assignment_prior_log_strength_derivative_weighted(
    assignment: &SaeAssignment,
    rho: &SaeManifoldRho,
    row_weights: Option<&[f64]>,
) -> Result<f64, String> {
    assignment.validate_rho_domain(rho)?;
    for row in 0..assignment.n_obs() {
        validate_finite_logits(assignment.logits.row(row), row)?;
    }
    let target = flat_logits(assignment.logits.view());
    if matches!(assignment.mode, AssignmentMode::Softmax { .. }) && assignment.k_atoms() == 1 {
        return Ok(0.0);
    }
    // #Bug4: frozen routing ⇒ inert prior ⇒ zero ρ-derivative.
    if assignment.routing_is_frozen() {
        return Ok(0.0);
    }
    Ok(match assignment.mode {
        AssignmentMode::Softmax { .. } => {
            return assignment_prior_value_weighted(assignment, rho, row_weights);
        }
        AssignmentMode::ThresholdGate {
            temperature,
            threshold,
        } => {
            // `∂_ρ[λ·Σ w·z + Σ w·log Z(λ)]`: the energy is degree one in `λ = e^ρ`, the
            // partition is not.
            let sparsity_strength = rho.lambda_sparse()?;
            let gates = threshold_gate_free_gates(
                assignment,
                target.view(),
                threshold,
                temperature,
                row_weights,
            );
            sparsity_strength * gates.weighted_activation
                + gates.weight
                    * ThresholdGateLogPartition::eval(sparsity_strength).log_strength_derivative()
        }
        AssignmentMode::OrderedBetaBernoulli {
            temperature, alpha, ..
        } => {
            // #Bug6: `ordered_beta_bernoulli_prior_penalty` picks the effective-α learnability (an
            // override forces the fixed-α value branch) and the #Bug4 ungated mask.
            let (penalty, rho_view) = ordered_beta_bernoulli_prior_penalty(
                assignment,
                rho,
                alpha,
                temperature,
                row_weights,
            )?;
            if penalty.learnable_alpha {
                penalty.grad_rho(target.view(), rho_view.view())[0]
            } else {
                penalty.value(target.view(), rho_view.view())
            }
        }
        // No prior term ⇒ no ρ-derivative (sparsity lives in the fixed support).
        AssignmentMode::TopK { .. } => 0.0,
    })
}

/// #991-weighted log-strength Hessian diagonal of the assignment prior. Every
/// assignment mode differentiates the same weighted scalar used by its value path.
pub(crate) fn assignment_prior_log_strength_hdiag_weighted(
    assignment: &SaeAssignment,
    rho: &SaeManifoldRho,
    row_weights: Option<&[f64]>,
) -> Result<Array1<f64>, String> {
    assignment.validate_rho_domain(rho)?;
    for row in 0..assignment.n_obs() {
        validate_finite_logits(assignment.logits.row(row), row)?;
    }
    let target = flat_logits(assignment.logits.view());
    if matches!(assignment.mode, AssignmentMode::Softmax { .. }) && assignment.k_atoms() == 1 {
        return Ok(Array1::<f64>::zeros(target.len()));
    }
    // #Bug4: frozen routing ⇒ inert prior ⇒ zero curvature everywhere.
    if assignment.routing_is_frozen() {
        return Ok(Array1::<f64>::zeros(target.len()));
    }
    match assignment.mode {
        AssignmentMode::Softmax {
            temperature,
            sparsity,
        } => {
            let penalty = SoftmaxAssignmentSparsityPenalty::new(assignment.k_atoms(), temperature)
                .with_row_weights(row_weights);
            let rho_view = Array1::from_vec(vec![rho.log_lambda_sparse + sparsity.ln()]);
            let mut d = penalty
                .hessian_diag(target.view(), rho_view.view())
                .ok_or_else(|| {
                    "softmax assignment log-strength hessian diag unavailable".to_string()
                })?;
            // #Bug4: the softmax array method is not internally column-masked, so
            // zero any fixed-logit (ungated) column's curvature diagonal to match
            // `assignment_prior_grad_hdiag_weighted`'s post-hoc masking.
            mask_fixed_logit_entries(assignment, &mut d);
            Ok(d)
        }
        AssignmentMode::ThresholdGate {
            temperature,
            threshold,
        } => {
            let sparsity_strength = rho.lambda_sparse()?;
            let inv_tau = 1.0 / temperature;
            let k = assignment.k_atoms();
            let mut d = Array1::<f64>::zeros(target.len());
            for idx in 0..target.len() {
                // #Bug4: ungated (inert) atoms carry no curvature.
                if assignment.logit_is_fixed(idx % k) {
                    continue;
                }
                // #991 — row `idx / k`'s design weight.
                let w_row = row_weights.map_or(1.0, |w| w[idx / k]);
                // #2520 — `∂/∂ρ_sparse` of the curvature `B` actually carries.
                // `smooth_psd_clamp` is homogeneous of degree 1 in its
                // prefactor and the prefactor carries `λ_sparse`, so that
                // derivative IS the majorizer; reading the shared seam is what
                // keeps this channel exact without a second derivation.
                d[idx] = ThresholdGateLogitCurvature::eval(
                    w_row * sparsity_strength,
                    target[idx],
                    threshold,
                    inv_tau,
                )
                .psd_majorizer_hess();
            }
            Ok(d)
        }
        AssignmentMode::OrderedBetaBernoulli {
            temperature, alpha, ..
        } => {
            let (penalty, rho_view) = ordered_beta_bernoulli_prior_penalty(
                assignment,
                rho,
                alpha,
                temperature,
                row_weights,
            )?;
            let mut d = if penalty.learnable_alpha {
                penalty.hessian_diag_log_alpha_derivative(target.view(), rho_view.view())
            } else {
                penalty
                    .hessian_diag(target.view(), rho_view.view())
                    .ok_or_else(|| {
                        "ordered Beta--Bernoulli assignment log-strength hessian diag unavailable"
                            .to_string()
                    })?
            };
            // #Bug4: zero the curvature diagonal of ungated (inert) columns so the
            // log-det ρ-trace never charges them (the array methods are not
            // internally column-masked).
            mask_fixed_logit_entries(assignment, &mut d);
            Ok(d)
        }
        // No prior term ⇒ zero curvature everywhere (mirrors the frozen-routing
        // early return; the support carries no free logits at all).
        AssignmentMode::TopK { .. } => Ok(Array1::<f64>::zeros(target.len())),
    }
}

/// The free gates of a ThresholdGate assignment: `Σ w_row·z` over every free logit with
/// `z = σ((ℓ − θ)/τ)`, and `Σ w_row` over the same logits.
struct ThresholdGateFreeGates {
    weighted_activation: f64,
    weight: f64,
}

fn threshold_gate_free_gates(
    assignment: &SaeAssignment,
    target: ArrayView1<'_, f64>,
    threshold: f64,
    temperature: f64,
    row_weights: Option<&[f64]>,
) -> ThresholdGateFreeGates {
    let k = assignment.k_atoms();
    let mut gates = ThresholdGateFreeGates {
        weighted_activation: 0.0,
        weight: 0.0,
    };
    for (idx, &logit) in target.iter().enumerate() {
        // #Bug4: skip ungated (inert) atoms' logits.
        if assignment.logit_is_fixed(idx % k) {
            continue;
        }
        // #991 — this row stands in for `w_i` population rows.
        let w_row = row_weights.map_or(1.0, |w| w[idx / k]);
        gates.weighted_activation +=
            w_row * gam_linalg::utils::stable_logistic((logit - threshold) / temperature);
        gates.weight += w_row;
    }
    gates
}

/// The log partition function of the ThresholdGate prior on one gate, and its derivative in
/// the log strength `ρ = ln λ`.
///
/// #2933 F45. The energy `λz` on `z ∈ (0, 1)` is a density only after dividing by
/// `Z(λ) = ∫₀¹ e^{−λz} dz = (1 − e^{−λ})/λ`, so each free gate's negative log prior is
/// `λz + log Z(λ)`, and with the change of variables [`GateLogitJacobian`] it integrates to one
/// over the logit for every `λ`, `θ` and `τ`. Without `log Z` the no-data mass is `Z(λ) < 1`
/// and the strength derivative lacks `λ·∂_λ log Z = λ/(e^λ − 1) − 1 = −λ·E[z]`, which is the
/// term that lets the criterion balance the fitted mean gate against the prior mean instead of
/// always lowering `λ`. A design-weighted row carries `w·log Z(λ)` beside its `w·λz`, as the
/// Jacobian carries `w·J`: it stands in for `w` population rows, each a normalized density.
///
/// `Z(λ) = exprel(−λ)`, evaluated by [`gam_math::special::log_exprel`]. The derivative is
/// `−(e^λ − 1 − λ)/(e^λ − 1)` through [`gam_math::special::expm1_minus_x`] up to `λ = 1/2`,
/// which never subtracts two terms near one as `λ → 0` (where it is `−λ/2`); above that it is
/// `λe^{−λ}/(1 − e^{−λ}) − 1`, which never forms `e^λ`. Both forms are accurate at the branch
/// point, the same one `gam_math` uses for its small-argument series.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ThresholdGateLogPartition {
    value: f64,
    log_strength_derivative: f64,
}

impl ThresholdGateLogPartition {
    pub(crate) fn eval(strength: f64) -> Self {
        let log_strength_derivative = if strength <= 0.5 {
            -gam_math::special::expm1_minus_x(strength) / strength.exp_m1()
        } else {
            strength * (-strength).exp() / -(-strength).exp_m1() - 1.0
        };
        Self {
            value: gam_math::special::log_exprel(-strength),
            log_strength_derivative,
        }
    }

    /// `log Z(λ) = ln[(1 − e^{−λ})/λ]`.
    pub(crate) fn value(self) -> f64 {
        self.value
    }

    /// `∂ log Z/∂ ln λ = λ/(e^λ − 1) − 1`.
    pub(crate) fn log_strength_derivative(self) -> f64 {
        self.log_strength_derivative
    }
}

/// The ThresholdGate sparsity prior's curvature at ONE logit, split into the
/// PSD majorizer the Newton/Schur factor declares and the non-positive
/// remainder that restores the exact signed curvature.
///
/// #2520. The exact second derivative of `λ·σ((ℓ−θ)/τ)` is
/// `λ·s·(1 − 2a)/τ²` with `a = σ((ℓ−θ)/τ)` and `s = a(1−a) ≥ 0`, which is
/// NEGATIVE for every logit above the threshold — the sigmoid penalty is
/// concave there. Written into `B` verbatim it made the per-row `H_tt` block
/// indefinite on exactly the atoms the gate had switched ON, and the
/// factorization then spectrally deflated those directions to unit stiffness:
/// #1419's pathology, on the one prior family that never received #1419's
/// treatment.
///
/// The split is [`SaeManifoldAtom`]'s periodic-ARD pair
/// (`psd_majorizer_hess` / `negative_hessian_remainder`) applied verbatim, and
/// it introduces NO new constant: the signed factor `1 − 2a ∈ (−1, 1)` is
/// dimensionless and lives on the same scale as the ARD axis's `cos κt ∈
/// [−1, 1]`, so #2339's derived softplus temperature transfers with its
/// derivation intact.
///
/// Homogeneity is load-bearing exactly as it is for ARD:
/// [`gam_linalg::utils::smooth_psd_clamp`] is degree-1 in its prefactor, and
/// the prefactor carries `λ_sparse`, so
/// `∂/∂ρ_sparse[majorizer] == majorizer` and the log-strength ρ-channel
/// ([`assignment_prior_log_strength_hdiag_weighted`]) stays exact by reading
/// the same seam rather than by a separate derivation.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ThresholdGateLogitCurvature {
    activation: f64,
    exact: f64,
    majorized: f64,
    exact_logit_derivative: f64,
    majorized_logit_derivative: f64,
}

impl ThresholdGateLogitCurvature {
    /// `strength` is the design-weighted prior strength `w_row·λ_sparse`; the
    /// caller keeps that convention so value, gradient and curvature share one
    /// weighting (#991).
    pub(crate) fn eval(strength: f64, logit: f64, threshold: f64, inv_tau: f64) -> Self {
        let activation = gam_linalg::utils::stable_logistic((logit - threshold) * inv_tau);
        let slope = activation * (1.0 - activation);
        // Non-negative magnitude, and the dimensionless signed factor it
        // multiplies. The clamp acts on the second and scales with the first.
        let magnitude = strength * slope * inv_tau * inv_tau;
        let signed = 1.0 - 2.0 * activation;
        // The logit derivative of BOTH curvatures, from the SAME evaluation, so
        // no theta-adjoint can differentiate a function no route installs.
        //
        // Writing `C` for the dimensionless clamp
        // ([`gam_linalg::utils::smooth_psd_clamp`] at unit prefactor) and using
        // `da/dl = s/tau`, `ds/dl = s(1-2a)/tau`, hence
        // `d(magnitude)/dl = strength*s*(1-2a)/tau^3` and `d(signed)/dl = -2s/tau`:
        //
        //   d(exact)/dl     = strength*s*(1-6a+6a^2)/tau^3        (`C` = identity)
        //   d(majorized)/dl = strength*s/tau^3 * [(1-2a)*C(1-2a) - 2s*C'(1-2a)]
        //
        // The two coincide wherever the clamp is inactive (`C(x) = x`, `C' = 1`
        // gives `(1-2a)^2 - 2s = 1-6a+6a^2` exactly); the majorized one is `0`
        // on the deep concave half where `B` carries a hard `0`; and at the seam
        // (`a = 1/2`) it is HALF the exact value -- the C-1 average of the hard
        // clamp's two one-sided slopes, which is what #2339's smoothing is for.
        let signed_clamp = gam_linalg::utils::smooth_psd_clamp(1.0, signed);
        let signed_clamp_slope = gam_linalg::utils::smooth_psd_clamp_slope(signed);
        let third_scale = strength * slope * inv_tau * inv_tau * inv_tau;
        Self {
            activation,
            exact: magnitude * signed,
            majorized: gam_linalg::utils::smooth_psd_clamp(magnitude, signed),
            exact_logit_derivative: third_scale
                * (1.0 - 6.0 * activation + 6.0 * activation * activation),
            majorized_logit_derivative: third_scale
                * (signed * signed_clamp - 2.0 * slope * signed_clamp_slope),
        }
    }

    /// `a = σ((ℓ−θ)/τ)`, shared with the gradient so both read one evaluation.
    pub(crate) fn activation(self) -> f64 {
        self.activation
    }

    /// Positive-semidefinite curvature written into `B`.
    pub(crate) fn psd_majorizer_hess(self) -> f64 {
        self.majorized
    }

    /// Signed correction with `psd_majorizer_hess + negative_hessian_remainder
    /// == exact` bit-for-bit, so `A = B + ΔC` is unchanged as an operator.
    /// Non-positive, because `softplus_{τ₀}(c) ≥ max(c, 0) ≥ c`.
    pub(crate) fn negative_hessian_remainder(self) -> f64 {
        self.exact - self.majorized
    }

    /// Logit derivative of [`Self::psd_majorizer_hess`] -- the theta-adjoint of
    /// the curvature `B` actually installs, and the ThresholdGate twin of
    /// `SaeManifoldTerm::ard_majorized_hessian_derivative`. It must exist
    /// because #2520 replaced the raw signed curvature in `B` with the clamp,
    /// which left the pre-#2520 third derivative `P'''` differentiating an
    /// operator no route installs any more.
    pub(crate) fn majorized_hess_logit_derivative(self) -> f64 {
        self.majorized_logit_derivative
    }

    /// Logit derivative of the UNCLAMPED signed curvature that `A = B + dC`
    /// carries: `P'''(l) = (lambda/tau^3)*s*(1-6a+6a^2)` (#1415).
    pub(crate) fn exact_hess_logit_derivative(self) -> f64 {
        self.exact_logit_derivative
    }

}

/// The logit Jacobian of a sigmoid gate's prior density (#2080).
///
/// The ordered Beta--Bernoulli and ThresholdGate priors are densities on the gate
/// `z = σ((ℓ − θ)/τ)`, while the inner solve and the Laplace evidence integrate over
/// the logit `ℓ`. Changing variables, `p(ℓ) = p(z)·|dz/dℓ| = p(z)·z(1 − z)/τ`, so the
/// penalized objective carries `−ln[z(1 − z)/τ]` per free gate. Without it the prior in
/// `ℓ` is improper: along a saturated logit the data slope and the prior slope both
/// decay like `e^{−|ℓ|/τ}`, the objective has no finite mode, and the evidence curvature
/// `λ_ℓ ∝ e^{−|ℓ|/τ}` drifts through the exact-A rank floor. On the #2080 wide-p fixture
/// 121–144 of 424 exact-A directions, every one a gate logit at `15 ≤ |ℓ/τ| < 50`,
/// crossed that floor between neighbouring ρ and moved the criterion by `½·|ln floor|`
/// per crossing (pool jobs 598388, 603989).
///
/// With `x = (ℓ − θ)/τ`: `J = |x| + 2·ln(1 + e^{−|x|}) + ln τ`, `J′ = (2z − 1)/τ`,
/// `J″ = 2z(1 − z)/τ²` and `J‴ = 2z(1 − z)(1 − 2z)/τ³`. `J″ ≥ 0` is the exact curvature,
/// so `B` carries it and `ΔC` gains no remainder. Nothing here depends on ρ, so no ρ
/// channel changes. Each gate carries its row's design weight (#991).
#[derive(Clone, Copy, Debug)]
pub(crate) struct GateLogitJacobian {
    value: f64,
    gradient: f64,
    curvature: f64,
    third: f64,
}

impl GateLogitJacobian {
    pub(crate) fn eval(weight: f64, logit: f64, threshold: f64, temperature: f64) -> Self {
        let inv_tau = 1.0 / temperature;
        let x = (logit - threshold) * inv_tau;
        // `z(1 − z) = e^{−|x|}/(1 + e^{−|x|})²` and `2z − 1 = tanh(x/2)`, both free of the
        // cancellation `1 − z` suffers where a gate saturates on: at `x = 25` that loses five
        // digits, and past `x ≈ 37` it rounds the slope to exactly zero.
        let tail = (-x.abs()).exp();
        let slope = tail / ((1.0 + tail) * (1.0 + tail));
        let centre = (0.5 * x).tanh();
        Self {
            value: weight * (x.abs() + 2.0 * tail.ln_1p() + temperature.ln()),
            gradient: weight * centre * inv_tau,
            curvature: weight * 2.0 * slope * inv_tau * inv_tau,
            third: -weight * 2.0 * slope * centre * inv_tau * inv_tau * inv_tau,
        }
    }

    /// `w·(−ln[z(1 − z)/τ])`.
    pub(crate) fn value(self) -> f64 {
        self.value
    }

    /// `w·(2z − 1)/τ`.
    pub(crate) fn gradient(self) -> f64 {
        self.gradient
    }

    /// `w·2z(1 − z)/τ²`: exact and non-negative.
    pub(crate) fn curvature(self) -> f64 {
        self.curvature
    }

    /// `w·2z(1 − z)(1 − 2z)/τ³`, the logit derivative of [`Self::curvature`].
    pub(crate) fn third(self) -> f64 {
        self.third
    }
}

/// `(θ, τ)` for a mode whose gates are per-logit sigmoids `z = σ((ℓ − θ)/τ)`, where
/// [`GateLogitJacobian`] applies. Softmax gates share one simplex per row, and TopK has
/// no free logits.
pub(crate) fn sigmoid_gate_frame(mode: &AssignmentMode) -> Option<(f64, f64)> {
    match *mode {
        AssignmentMode::OrderedBetaBernoulli { temperature, .. } => Some((0.0, temperature)),
        AssignmentMode::ThresholdGate {
            temperature,
            threshold,
        } => Some((threshold, temperature)),
        AssignmentMode::Softmax { .. } | AssignmentMode::TopK { .. } => None,
    }
}

/// One free gate's [`GateLogitJacobian`], or `None` for a fixed logit or a mode without
/// per-logit sigmoid gates, which carry no gate prior and so no change of variables.
fn gate_logit_jacobian_at(
    assignment: &SaeAssignment,
    row_weights: Option<&[f64]>,
    row: usize,
    atom: usize,
) -> Option<GateLogitJacobian> {
    let (threshold, temperature) = sigmoid_gate_frame(&assignment.mode)?;
    if assignment.routing_is_frozen() || assignment.logit_is_fixed(atom) {
        return None;
    }
    let weight = row_weights.map_or(1.0, |w| w[row]);
    Some(GateLogitJacobian::eval(
        weight,
        assignment.logits[[row, atom]],
        threshold,
        temperature,
    ))
}

/// The free logits of a softmax assignment and its temperature, or `None` when the mode is
/// not softmax or no logit is free.
///
/// A softmax row's prior is a density on the simplex, while the inner solve and the Laplace
/// evidence integrate over its free logits. The chart holds the reference logit `K − 1` at
/// zero, and frozen routing holds every logit, leaving no free set. An ungated atom never
/// reaches here: softmax refuses it ([`SaeAssignment::validate_gate_configuration`], #2933 F04).
/// For the free set `F`, with `R = 1 − Σ_{i∈F} z_i` the mass on the held atoms, the change of
/// variables has `|det ∂z_F/∂ℓ_F| = Π_{i∈F} z_i · R / τ^{|F|}` (#2080), so each row carries
/// `J = −Σ_{i∈F} ln z_i − ln R + |F|·ln τ`. With `c = |F| + 1`, `∂J/∂ℓ_j = (c·z_j − 1)/τ` and
/// `∂²J/∂ℓ_i∂ℓ_j = c·z_i(δ_ij − z_j)/τ²`, which is exact and PSD. Without it a minority atom's
/// probability runs to zero with no finite mode, the softmax twin of the saturated sigmoid
/// gate [`GateLogitJacobian`] fixes. Nothing here depends on ρ.
fn simplex_gate_frame(assignment: &SaeAssignment) -> Option<(Vec<usize>, f64)> {
    let AssignmentMode::Softmax { temperature, .. } = assignment.mode else {
        return None;
    };
    let free: Vec<usize> = (0..assignment.k_atoms().saturating_sub(1))
        .filter(|&atom| !assignment.logit_is_fixed(atom))
        .collect();
    (!free.is_empty()).then_some((free, temperature))
}

/// `ln Σ_{atom ∈ atoms} e^{ℓ_atom/τ}`, shifted by its largest term so no exponent overflows or
/// underflows to an infinite logarithm.
fn log_sum_exp_scaled(
    logits: ArrayView1<'_, f64>,
    atoms: impl Iterator<Item = usize> + Clone,
    inv_tau: f64,
) -> f64 {
    let top = atoms
        .clone()
        .map(|atom| logits[atom] * inv_tau)
        .fold(f64::NEG_INFINITY, f64::max);
    top + atoms
        .map(|atom| (logits[atom] * inv_tau - top).exp())
        .sum::<f64>()
        .ln()
}

/// One softmax row's `J = −Σ_{i∈F} ln z_i − ln R + |F|·ln τ` (see [`simplex_gate_frame`]), from
/// log-sum-exps so a minority atom's `ln z_i` and the held mass `ln R` stay finite.
fn simplex_gate_logit_jacobian_value(
    logits: ArrayView1<'_, f64>,
    free: &[usize],
    temperature: f64,
) -> f64 {
    let inv_tau = temperature.recip();
    let k = logits.len();
    let all = log_sum_exp_scaled(logits, 0..k, inv_tau);
    let held = log_sum_exp_scaled(logits, (0..k).filter(|atom| !free.contains(atom)), inv_tau);
    let free_log_mass: f64 = free.iter().map(|&atom| logits[atom] * inv_tau - all).sum();
    -free_log_mass - (held - all) + free.len() as f64 * temperature.ln()
}

/// `z_i(1 − z_i)` as `z_i·Σ_{k≠i} z_k`, free of the cancellation at a dominant atom.
fn simplex_spread(z: &Array1<f64>, i: usize) -> f64 {
    let others: f64 = (0..z.len()).filter(|&atom| atom != i).map(|atom| z[atom]).sum();
    z[i] * others
}

/// The gate prior's change-of-variables term in logit coordinates, summed over every free
/// sigmoid gate ([`GateLogitJacobian`]) or every softmax row ([`simplex_gate_frame`]).
pub(crate) fn gate_logit_jacobian_value_weighted(
    assignment: &SaeAssignment,
    row_weights: Option<&[f64]>,
) -> f64 {
    let mut total = 0.0_f64;
    if let Some((free, temperature)) = simplex_gate_frame(assignment) {
        for row in 0..assignment.n_obs() {
            let weight = row_weights.map_or(1.0, |w| w[row]);
            total += weight
                * simplex_gate_logit_jacobian_value(assignment.logits.row(row), &free, temperature);
        }
        return total;
    }
    let k = assignment.k_atoms();
    for row in 0..assignment.n_obs() {
        for atom in 0..k {
            if let Some(jacobian) = gate_logit_jacobian_at(assignment, row_weights, row, atom) {
                total += jacobian.value();
            }
        }
    }
    total
}

/// Gradient and curvature diagonal of [`gate_logit_jacobian_value_weighted`] per flat
/// `(row·K + atom)` logit, the layout of [`assignment_prior_grad_hdiag_weighted`]. A softmax
/// row's full curvature block is [`simplex_gate_logit_jacobian_row_block`].
pub(crate) fn gate_logit_jacobian_grad_hdiag_weighted(
    assignment: &SaeAssignment,
    row_weights: Option<&[f64]>,
) -> (Array1<f64>, Array1<f64>) {
    let k = assignment.k_atoms();
    let n = assignment.n_obs();
    let mut grad = Array1::<f64>::zeros(n * k);
    let mut curvature = Array1::<f64>::zeros(n * k);
    if let Some((free, temperature)) = simplex_gate_frame(assignment) {
        let inv_tau = temperature.recip();
        let count = (free.len() + 1) as f64;
        for row in 0..n {
            let weight = row_weights.map_or(1.0, |w| w[row]);
            let z = softmax_row(assignment.logits.row(row), temperature);
            for &atom in &free {
                grad[row * k + atom] = weight * (count * z[atom] - 1.0) * inv_tau;
                curvature[row * k + atom] =
                    weight * count * simplex_spread(&z, atom) * inv_tau * inv_tau;
            }
        }
        return (grad, curvature);
    }
    for row in 0..n {
        for atom in 0..k {
            if let Some(jacobian) = gate_logit_jacobian_at(assignment, row_weights, row, atom) {
                grad[row * k + atom] = jacobian.gradient();
                curvature[row * k + atom] = jacobian.curvature();
            }
        }
    }
    (grad, curvature)
}

/// One softmax row's Jacobian curvature `c·z_i(δ_ij − z_j)/τ²` over the chart's `K − 1` logit
/// slots, zero on held slots, or `None` when the assignment has no free softmax logit.
pub(crate) fn simplex_gate_logit_jacobian_row_block(
    assignment: &SaeAssignment,
    row_weights: Option<&[f64]>,
    row: usize,
) -> Option<Array2<f64>> {
    let (free, temperature) = simplex_gate_frame(assignment)?;
    let k = assignment.k_atoms();
    let inv_tau = temperature.recip();
    let scale = row_weights.map_or(1.0, |w| w[row]) * (free.len() + 1) as f64 * inv_tau * inv_tau;
    let z = softmax_row(assignment.logits.row(row), temperature);
    let mut block = Array2::<f64>::zeros((k - 1, k - 1));
    for &i in &free {
        block[[i, i]] = scale * simplex_spread(&z, i);
        for &j in &free {
            if j != i {
                block[[i, j]] = -scale * z[i] * z[j];
            }
        }
    }
    Some(block)
}

/// `c = |F| + 1` for a softmax assignment's free logits (see [`simplex_gate_frame`]), or `None`
/// when there is none.
pub(crate) fn simplex_gate_free_count(assignment: &SaeAssignment) -> Option<f64> {
    simplex_gate_frame(assignment).map(|(free, _)| (free.len() + 1) as f64)
}

/// `∂³J/∂ℓ_i∂ℓ_j∂ℓ_w = (c/τ³)·[z_i(δ_iw − z_w)δ_ij − z_i(δ_iw − z_w)z_j − z_i z_j(δ_jw − z_w)]`, the
/// logit derivative of one softmax row's Jacobian curvature, for the θ-adjoints. `z` is the
/// row's full softmax vector, `count` is [`simplex_gate_free_count`], and the caller masks held
/// logits.
pub(crate) fn simplex_gate_logit_jacobian_third(
    z: &[f64],
    i: usize,
    j: usize,
    w: usize,
    count: f64,
    inv_tau: f64,
) -> f64 {
    let delta = |a: usize, b: usize| if a == b { 1.0 } else { 0.0 };
    let dz_i = z[i] * (delta(i, w) - z[w]);
    let dz_j = z[j] * (delta(j, w) - z[w]);
    count * inv_tau * inv_tau * inv_tau * (dz_i * delta(i, j) - dz_i * z[j] - z[i] * dz_j)
}

/// The logit derivative of one gate's [`GateLogitJacobian`] curvature, for the θ-adjoints
/// that differentiate the logit diagonal of `B` or `A`.
pub(crate) fn gate_logit_jacobian_third_weighted(
    assignment: &SaeAssignment,
    row_weights: Option<&[f64]>,
    row: usize,
    atom: usize,
) -> f64 {
    gate_logit_jacobian_at(assignment, row_weights, row, atom).map_or(0.0, GateLogitJacobian::third)
}

/// The ΔC channel of the ThresholdGate prior: the non-positive remainder
/// `exact − majorizer` per flat `(row·K + atom)` logit, masked identically to
/// the curvature [`assignment_prior_grad_hdiag_weighted`] writes into `B`.
///
/// #2520. Reads the same [`ThresholdGateLogitCurvature`] seam and the same
/// [`mask_fixed_logit_entries`] rule as the majorizer, so `A = B + ΔC` cannot
/// drift by construction. Every non-ThresholdGate mode returns zeros: their
/// majorizers are exact, or their remainder travels its own channel (ordered
/// Beta--Bernoulli's rank-one HVP).
pub(crate) fn threshold_gate_negative_hessian_remainder_weighted(
    assignment: &SaeAssignment,
    rho: &SaeManifoldRho,
    row_weights: Option<&[f64]>,
) -> Result<Array1<f64>, String> {
    assignment.validate_rho_domain(rho)?;
    let target = flat_logits(assignment.logits.view());
    let mut remainder = Array1::<f64>::zeros(target.len());
    let AssignmentMode::ThresholdGate {
        temperature,
        threshold,
    } = assignment.mode
    else {
        return Ok(remainder);
    };
    for row in 0..assignment.n_obs() {
        validate_finite_logits(assignment.logits.row(row), row)?;
    }
    let sparsity_strength = rho.lambda_sparse()?;
    let inv_tau = 1.0 / temperature;
    let k = assignment.k_atoms();
    for idx in 0..target.len() {
        let w_row = row_weights.map_or(1.0, |w| w[idx / k]);
        let curvature = ThresholdGateLogitCurvature::eval(
            w_row * sparsity_strength,
            target[idx],
            threshold,
            inv_tau,
        );
        remainder[idx] = curvature.negative_hessian_remainder();
    }
    mask_fixed_logit_entries(assignment, &mut remainder);
    Ok(remainder)
}

/// Zero the entries of a flat `(n·K)` per-(row, atom) array whose atom is a FIXED
/// (ungated / frozen) logit, so an inert atom contributes nothing to the term.
/// (#Bug4) No-op when nothing is fixed.
fn mask_fixed_logit_entries(assignment: &SaeAssignment, arr: &mut Array1<f64>) {
    if !(assignment.has_ungated() || assignment.routing_is_frozen()) {
        return;
    }
    let k = assignment.k_atoms();
    for idx in 0..arr.len() {
        if assignment.logit_is_fixed(idx % k) {
            arr[idx] = 0.0;
        }
    }
}

/// #991-weighted log-strength/target mixed derivative of the assignment prior. The fixed-α
/// fall-through reuses the `w_i`-weighted gradient; the learnable-α ordered Beta--Bernoulli branch
/// uses the same weighted active mass as the value, gradient, and Hessian.
pub(crate) fn assignment_prior_log_strength_target_mixed_weighted(
    assignment: &SaeAssignment,
    rho: &SaeManifoldRho,
    row_weights: Option<&[f64]>,
) -> Result<Array1<f64>, String> {
    assignment.validate_rho_domain(rho)?;
    for row in 0..assignment.n_obs() {
        validate_finite_logits(assignment.logits.row(row), row)?;
    }
    let target = flat_logits(assignment.logits.view());
    if matches!(assignment.mode, AssignmentMode::Softmax { .. }) && assignment.k_atoms() == 1 {
        return Ok(Array1::<f64>::zeros(target.len()));
    }
    // #Bug4: frozen routing ⇒ inert prior ⇒ zero mixed derivative.
    if assignment.routing_is_frozen() {
        return Ok(Array1::<f64>::zeros(target.len()));
    }
    // #Bug6: the α-target mixed derivative only exists when α is EFFECTIVELY
    // learnable (mode-learnable AND not pinned by an override); otherwise α is a
    // constant and there is no log-α channel, so fall through to the grad_hdiag
    // (fixed-α) path.
    match assignment.mode {
        AssignmentMode::OrderedBetaBernoulli {
            temperature, alpha, ..
        } if assignment.effective_alpha_is_learnable() => {
            let (penalty, rho_view) = ordered_beta_bernoulli_prior_penalty(
                assignment,
                rho,
                alpha,
                temperature,
                row_weights,
            )?;
            let mut d = penalty.log_alpha_target_mixed_derivative(target.view(), rho_view.view());
            // #Bug4: inert columns carry no mixed derivative.
            mask_fixed_logit_entries(assignment, &mut d);
            Ok(d)
        }
        _ => Ok(assignment_prior_grad_hdiag_weighted(assignment, rho, row_weights)?.0),
    }
}

/// #991-weighted per-(row, atom) logit
/// gradient and Hessian diagonal of the assignment prior, each row scaled by its
/// design weight `w_i`. Softmax, threshold gate, and ordered Beta--Bernoulli
/// modes all use the same row weights in value, gradient, curvature, and outer
/// concentration derivatives.
///
/// The assembly (`construction_arrow_schur_assembly`) consumes THIS gradient
/// unchanged for `gt`; the softmax curvature written to `htt` is the per-row
/// Gershgorin/`row_psd_majorizer` block, which its call sites weight by folding
/// `w_row` into the `scale` they pass — so the softmax gradient and curvature
/// both carry `w_i` without any double application.
pub(crate) fn assignment_prior_grad_hdiag_weighted(
    assignment: &SaeAssignment,
    rho: &SaeManifoldRho,
    row_weights: Option<&[f64]>,
) -> Result<(Array1<f64>, Array1<f64>), String> {
    assignment.validate_rho_domain(rho)?;
    for row in 0..assignment.n_obs() {
        validate_finite_logits(assignment.logits.row(row), row)?;
    }
    let target = flat_logits(assignment.logits.view());
    let mut grad = Array1::<f64>::zeros(target.len());
    let mut diag = Array1::<f64>::zeros(target.len());
    if matches!(assignment.mode, AssignmentMode::Softmax { .. }) && assignment.k_atoms() == 1 {
        return Ok((grad, diag));
    }
    let (sparsity_grad, sparsity_diag) = match assignment.mode {
        AssignmentMode::Softmax {
            temperature,
            sparsity,
        } => {
            let penalty = SoftmaxAssignmentSparsityPenalty::new(assignment.k_atoms(), temperature)
                .with_row_weights(row_weights);
            let rho_view = Array1::from_vec(vec![rho.log_lambda_sparse + sparsity.ln()]);
            let g = penalty.grad_target(target.view(), rho_view.view());
            let d = penalty
                .hessian_diag(target.view(), rho_view.view())
                .ok_or_else(|| "softmax assignment hessian diag unavailable".to_string())?;
            (g, d)
        }
        AssignmentMode::OrderedBetaBernoulli {
            temperature, alpha, ..
        } => {
            // Scale the ordered Beta--Bernoulli assignment-sparsity prior by `lambda_sparse` in the
            // fixed-α branch (Softmax folds it into the penalty's rho coordinate;
            // threshold gate multiplies `sparsity_strength`). #Bug6: `ordered_beta_bernoulli_prior_penalty`
            // picks the EFFECTIVE-α learnability — an override pins α so the prior
            // uses the fixed-α weight convention and the resolved (override) α,
            // matching the forward gate — and installs the #Bug4 ungated mask. The
            // per-atom fixed-logit columns are additionally zeroed post-hoc below,
            // so the array (grad/hessian) methods need no internal column mask.
            let (penalty, rho_view) = ordered_beta_bernoulli_prior_penalty(
                assignment,
                rho,
                alpha,
                temperature,
                row_weights,
            )?;
            let g = penalty.grad_target(target.view(), rho_view.view());
            let d = penalty
                .hessian_diag(target.view(), rho_view.view())
                .ok_or_else(|| {
                    "ordered Beta--Bernoulli assignment hessian diag unavailable".to_string()
                })?;
            (g, d)
        }
        AssignmentMode::ThresholdGate {
            temperature,
            threshold,
        } => {
            // Gradient and exact diagonal Hessian of the sparsity value's
            // threshold-centered surrogate σ((l−θ)/τ), using the same
            // machine-precision support as the value path. Data-fit JVP support
            // is narrower and follows the hard forward gate.
            //
            // The `d` returned here is the curvature the arrow assembly writes
            // into `block.htt` VERBATIM.
            //
            // HISTORY — this block used to say `d` was the EXACT signed curvature
            // and that it was the one assignment/coordinate prior in the SAE inner
            // system that was not PSD-majorized first. `2956f601c` (#2520) ended
            // that: `d` is now `psd_majorizer_hess`, and the concave half travels
            // `threshold_gate_negative_hessian_remainder_weighted` into `ΔC`, so
            // `A = B + ΔC` is still the exact signed operator while `B` — the thing
            // that gets factored, and whose ½log|B| the criterion prices — is PSD
            // like `λ_k·S_k ⊗ I`, periodic ARD's `α·softplus_{τ₀}(cos κt)`,
            // softmax's Gershgorin radius, and
            // `ordered_beta_bernoulli_psd_majorized_hdiag`.
            //
            // A MEASUREMENT IN THIS COMMENT WAS FALSIFIED, and it is kept here
            // because the #2500 gates were written against it and still assert it:
            // "Measured on `threshold_gate_tiny_fixture(straddle = true)`: all ten
            // rows deflate exactly one direction each, and each one is the
            // negative-curvature logit." NO LONGER TRUE — run 30503262222 measures
            // `deflated_direction_count == 0` on BOTH arms of that fixture, which
            // is what `threshold_gate_sparse_operator_is_the_installed_exact_a_\
            // derivative_2500`, `..._is_not_the_raw_prior_on_deflated_rows_2500`
            // and `deflation_map_applies_to_every_row_local_curvature_\
            // coordinate_2500` report when they fail. The `1 − 2a < 0` indefinite
            // `H_tt` those rows deflated is no longer what is factored: `d` is the
            // majorizer. What is NOT established is the mechanism — note the
            // majorizer is not merely non-negative but EXACTLY ZERO above the
            // threshold (`|1 − 2a| ≫ τ₀ ≈ 1.44e-8` makes the softplus term
            // underflow), so a vanishing diagonal entry there would if anything
            // deflate MORE. Do not re-derive that story from this comment; measure.
            //
            // What #2520 did NOT do: teach
            // `materialize_ard_concave_clamp_diagonal` about this remainder, so a
            // mode whose only indefiniteness is the gate's own concave half is
            // still REFUSED rather than priced at its basin curvature under
            // #2336's E-attributability rule. That is a separate change — it moves
            // the `½log|B|` criterion of every ThresholdGate fit and wants its own
            // pre-registered A/B.
            let sparsity_strength = rho.lambda_sparse()?;
            let inv_tau = 1.0 / temperature;
            let k = assignment.k_atoms();
            let mut g = Array1::<f64>::zeros(target.len());
            let mut d = Array1::<f64>::zeros(target.len());
            for idx in 0..target.len() {
                // #991 — row `idx / k`'s design weight scales this row's prior
                // gradient AND curvature identically (both linear in strength).
                let w_row = row_weights.map_or(1.0, |w| w[idx / k]);
                let strength = w_row * sparsity_strength;
                let curvature = ThresholdGateLogitCurvature::eval(
                    strength,
                    target[idx],
                    threshold,
                    inv_tau,
                );
                let activation = curvature.activation();
                g[idx] = strength * activation * (1.0 - activation) * inv_tau;
                // #2520 — the PSD majorizer, not the exact signed curvature.
                // `ΔC` restores the concave half through
                // `threshold_gate_negative_hessian_remainder_weighted`, so the
                // EXACT operator `A = B + ΔC` is unchanged while `B` — the
                // thing that gets factored, and whose ½log|B| the criterion
                // prices — is positive semidefinite like every other
                // assignment/coordinate prior in the crate.
                d[idx] = curvature.psd_majorizer_hess();
            }
            (g, d)
        }
        // No sparsity prior and no free logits: zero gradient and curvature by
        // construction (every column is also masked as fixed below).
        AssignmentMode::TopK { .. } => (
            Array1::<f64>::zeros(target.len()),
            Array1::<f64>::zeros(target.len()),
        ),
    };
    grad += &sparsity_grad;
    diag += &sparsity_diag;
    // #1026/#1033 — a FIXED logit (an ungated atom's, or every atom's under
    // frozen routing) is not a free parameter, so it carries NO sparsity-prior
    // gradient or curvature. Zero its flat columns (`flat_logits` is row-major
    // `row*K + atom`) so the assembled `gt` and `htt` logit slots stay zero —
    // matching the zero logit-JVP. The column-separable ordered Beta--Bernoulli / threshold gate priors are
    // per-atom, so zeroing one atom's columns leaves the others' prior intact;
    // under frozen routing ALL atoms' logit columns are zeroed (the whole routing
    // is a fixed predicted function, not optimized).
    if assignment.has_ungated() || assignment.routing_is_frozen() {
        let k = assignment.k_atoms();
        for idx in 0..grad.len() {
            if assignment.logit_is_fixed(idx % k) {
                grad[idx] = 0.0;
                diag[idx] = 0.0;
            }
        }
    }
    Ok((grad, diag))
}

/// Build exact derivatives of the ordered Beta--Bernoulli PSD curvature
/// majorizer for the SAE log-det adjoint Γ, using the same penalty configuration —
/// `alpha`/`tau`/`learnable_alpha` and the `lambda_sparse` weight convention —
/// that [`assignment_prior_grad_hdiag_weighted`] assembles into `htt`. Returns
/// `None` for other assignment modes. Every row carries the #991 design-honesty
/// weight the assembled `htt` carried (the channels must differentiate the
/// same weighted operator; `z_jac` carries the weighted active-mass derivative
/// `u = w·J`).
pub(crate) fn ordered_beta_bernoulli_psd_majorizer_third_channels_weighted(
    assignment: &SaeAssignment,
    rho: &SaeManifoldRho,
    row_weights: Option<&[f64]>,
) -> Result<Option<OrderedBetaBernoulliHessianDiagThirdChannels>, String> {
    assignment.validate_rho_domain(rho)?;
    let AssignmentMode::OrderedBetaBernoulli {
        temperature, alpha, ..
    } = assignment.mode
    else {
        return Ok(None);
    };
    for row in 0..assignment.n_obs() {
        validate_finite_logits(assignment.logits.row(row), row)?;
    }
    let target = flat_logits(assignment.logits.view());
    // #Bug6: build with the EFFECTIVE-α learnability and weight convention that
    // `assignment_prior_grad_hdiag_weighted` uses, so an α override differentiates the same
    // fixed-α operator. Fixed-logit columns are zeroed post-hoc below (the channel
    // arrays are not internally column-masked).
    let (penalty, rho_view) =
        ordered_beta_bernoulli_prior_penalty(assignment, rho, alpha, temperature, row_weights)?;
    let mut channels = penalty.psd_majorizer_logit_third_channels(target.view(), rho_view.view());
    // #1026/#1033 — zero the log-det third-derivative channels of FIXED-logit
    // atoms (ungated, or all atoms under frozen routing) so the #1006 θ-adjoint
    // differentiates the SAME (fixed-logit-zeroed) `htt` that
    // `assignment_prior_grad_hdiag_weighted` assembled. `k_max` columns, row-major `N·K`
    // for the per-(row,atom) arrays and length-`K` for the per-column ones.
    if assignment.has_ungated() || assignment.routing_is_frozen() {
        let k = channels.k_max;
        for idx in 0..channels.z_jac.len() {
            if assignment.logit_is_fixed(idx % k) {
                channels.z_jac[idx] = 0.0;
                channels.local_logit_third[idx] = 0.0;
                channels.m_channel[idx] = 0.0;
                channels.diagonal_term[idx] = 0.0;
            }
        }
        for atom in 0..k {
            if assignment.logit_is_fixed(atom) {
                channels.mass_hessian_coefficient[atom] = 0.0;
                channels.mass_hessian_log_alpha_derivative[atom] = 0.0;
            }
        }
    }
    Ok(Some(channels))
}

#[cfg(test)]
mod support_measure_tests {
    use super::*;

    #[test]
    fn support_measure_reads_assignment_column() {
        let assignments =
            Array2::from_shape_vec((3, 2), vec![0.8, 0.2, 0.4, 0.6, 0.0, 1.0]).unwrap();
        let support = SupportMeasure::from_assignment_matrix(assignments.view(), 1).unwrap();
        assert!((support.mass() - 1.8).abs() < 1e-12);
        assert!((support.fisher_n() - 1.4).abs() < 1e-12);
        assert!((support.ess() - (1.8_f64 * 1.8 / 1.4)).abs() < 1e-12);
        assert_eq!(support.positive_rows(), vec![0usize, 1, 2]);
    }
}

#[cfg(test)]
mod ordered_alpha_domain_tests {
    use super::*;
    use gam_problem::{LOG_STRENGTH_MAX, LOG_STRENGTH_MIN};

    fn ordered_assignment(alpha: f64) -> SaeAssignment {
        SaeAssignment::from_blocks_with_mode_and_manifolds(
            Array2::<f64>::zeros((3, 2)),
            vec![Array2::<f64>::zeros((3, 1)); 2],
            vec![LatentManifold::Euclidean; 2],
            AssignmentMode::ordered_beta_bernoulli(0.8, alpha, true),
        )
        .unwrap()
    }

    #[test]
    fn learnable_ordered_alpha_tightens_sparse_rho_face_without_saturation() {
        let alpha = 1.7_f64;
        let assignment = ordered_assignment(alpha);
        let (lower, upper) = assignment
            .learnable_alpha_rho_domain()
            .unwrap()
            .expect("learnable ordered alpha owns the sparse rho coordinate");
        assert!(upper < LOG_STRENGTH_MAX);

        let legal = SaeManifoldRho::new(upper, 0.0, vec![Array1::zeros(1); 2])
            .for_assignment(assignment.mode);
        assignment
            .validate_rho_domain(&legal)
            .expect("closed effective-alpha upper face is legal");
        let invalid = SaeManifoldRho::new(upper + 1.0e-6, 0.0, vec![Array1::zeros(1); 2])
            .for_assignment(assignment.mode);
        assert!(assignment.validate_rho_domain(&invalid).is_err());

        assert_eq!(
            lower,
            LOG_STRENGTH_MIN - alpha.ln(),
            "lower face must be shifted by the base concentration too"
        );
    }
}

#[cfg(test)]
mod fill_into_buffer_1557_tests {
    //! #1557 — the fill-into-caller-buffer variant
    //! [`SaeAssignment::try_assignments_row_into`] must produce
    //! BIT-IDENTICAL output to the allocating
    //! [`SaeAssignment::try_assignments_row`] across every assignment
    //! mode (Softmax, OrderedBetaBernoulli, threshold gate), the #1026 ungated case, and the K==1
    //! edge. Exact `==` on f64 — not an approximate tolerance — because the
    //! `_into` path is a pure allocation-elision refactor and any numeric drift
    //! is a regression.
    use super::*;

    fn build(n: usize, k: usize, mode: AssignmentMode) -> SaeAssignment {
        // Deterministic, asymmetric logits/coords so every atom takes a distinct
        // value (no accidental ties masking an index bug).
        let logits = Array2::from_shape_fn((n, k), |(i, kk)| {
            0.37 + 0.11 * (i as f64) - 0.23 * (kk as f64)
        });
        let coords: Vec<Array2<f64>> = (0..k)
            .map(|_| Array2::from_shape_fn((n, 1), |(i, _)| 0.1 + 0.05 * (i as f64)))
            .collect();
        SaeAssignment::from_blocks_with_mode_and_manifolds(
            logits,
            coords,
            vec![LatentManifold::Euclidean; k],
            mode,
        )
        .unwrap()
    }

    fn assert_into_matches_alloc(a: &SaeAssignment) {
        let n = a.n_obs();
        let k = a.k_atoms();
        let mut scratch = vec![f64::NAN; k];
        for row in 0..n {
            let allocated = a.try_assignments_row(row).unwrap();
            // Pre-fill with NaN so a partial write (e.g. a threshold gate below-threshold
            // entry left untouched) is caught as a mismatch, not silently passed.
            for s in scratch.iter_mut() {
                *s = f64::NAN;
            }
            a.try_assignments_row_into(row, &mut scratch).unwrap();
            assert_eq!(allocated.len(), k);
            for kk in 0..k {
                assert_eq!(
                    allocated[kk], scratch[kk],
                    "row {row} atom {kk}: _into must be BIT-IDENTICAL to the allocating \
                     try_assignments_row; got {} vs {}",
                    allocated[kk], scratch[kk]
                );
            }
        }
    }

    #[test]
    fn softmax_into_is_bit_identical() {
        assert_into_matches_alloc(&build(7, 4, AssignmentMode::softmax(0.8)));
    }

    #[test]
    fn ordered_beta_bernoulli_into_is_bit_identical() {
        // Both learnable and fixed alpha exercise the resolved-alpha branch.
        assert_into_matches_alloc(&build(
            7,
            5,
            AssignmentMode::ordered_beta_bernoulli(0.6, 1.3, false),
        ));
        assert_into_matches_alloc(&build(
            7,
            5,
            AssignmentMode::ordered_beta_bernoulli(0.6, 1.3, true),
        ));
    }

    #[test]
    fn threshold_gate_into_is_bit_identical() {
        // Threshold chosen so SOME atoms fall below it (the untouched-entry path)
        // and some clear it (the sigmoid path) — both branches are exercised.
        assert_into_matches_alloc(&build(7, 5, AssignmentMode::threshold_gate(0.9, 0.2)));
    }

    #[test]
    fn k_equals_one_into_is_bit_identical() {
        // Softmax K==1 hits the fixed-unit early return; ordered Beta--Bernoulli/threshold gate K==1 keep a
        // free per-atom gate and fall through to the real row functions.
        assert_into_matches_alloc(&build(5, 1, AssignmentMode::softmax(1.0)));
        assert_into_matches_alloc(&build(
            5,
            1,
            AssignmentMode::ordered_beta_bernoulli(0.7, 1.0, false),
        ));
        assert_into_matches_alloc(&build(5, 1, AssignmentMode::threshold_gate(0.8, 0.1)));
    }
}

#[cfg(test)]
mod frozen_routing_1033_tests {
    //! #1033 — the FROZEN (amortized) routing mechanism: once installed, the
    //! per-row gate is a ρ-invariant function of the FROZEN predicted logits and
    //! is DECOUPLED from any subsequent update to the free `self.logits` (the
    //! inner-fit logit drift the outer ρ-search would otherwise re-incur every
    //! eval). These are deterministic mechanism invariants — no inner fit — so
    //! they pin the load-bearing freeze properties without the cluster.
    use super::*;

    fn ordered_beta_bernoulli_assignment(n: usize, k: usize) -> SaeAssignment {
        let logits = Array2::from_shape_fn((n, k), |(i, kk)| {
            0.3 + 0.05 * (i as f64) - 0.1 * (kk as f64)
        });
        let coords: Vec<Array2<f64>> = (0..k)
            .map(|_| Array2::from_shape_fn((n, 1), |(i, _)| (i as f64) * 0.1))
            .collect();
        // learnable_alpha = false: alpha is ρ-independent, isolating the routing.
        SaeAssignment::from_blocks_with_mode_and_manifolds(
            logits,
            coords,
            vec![LatentManifold::Euclidean; k],
            AssignmentMode::ordered_beta_bernoulli(0.5, 1.0, false),
        )
        .unwrap()
    }

    /// Freeze the free logits as the routing, as the amortized fit driver does.
    fn frozen(mut assignment: SaeAssignment) -> SaeAssignment {
        assignment.frozen_logits = Some(assignment.logits.clone());
        assignment
    }

    #[test]
    fn frozen_routing_decouples_gates_from_logit_updates_1033() {
        let (n, k) = (6usize, 3usize);
        let mut a = frozen(ordered_beta_bernoulli_assignment(n, k));
        assert!(a.routing_is_frozen());
        // Gates BEFORE mutating the free logits.
        let before: Vec<Array1<f64>> = (0..n).map(|r| a.try_assignments_row(r).unwrap()).collect();
        // Simulate an inner-fit logit update (what the ρ-search would otherwise do
        // every eval): perturb every free logit substantially.
        a.logits.mapv_inplace(|v| v + 5.0);
        let after: Vec<Array1<f64>> = (0..n).map(|r| a.try_assignments_row(r).unwrap()).collect();
        // FROZEN routing reads the snapshot, so the gates are UNCHANGED by the
        // free-logit perturbation — the routing is decoupled from inner-fit drift.
        for r in 0..n {
            for kk in 0..k {
                assert_eq!(
                    before[r][kk], after[r][kk],
                    "row {r} atom {kk}: frozen-routing gate must be UNCHANGED by a free-logit \
                     update (decoupled from inner-fit drift); {} vs {}",
                    before[r][kk], after[r][kk]
                );
            }
        }
    }

    #[test]
    fn frozen_routing_gates_are_rho_invariant_1033() {
        let (n, k) = (5usize, 2usize);
        let a = frozen(ordered_beta_bernoulli_assignment(n, k));
        // The ρ-invariance is now STRUCTURAL: the assignment APIs take no ρ
        // (the signature is the proof). What remains observable is purity —
        // repeated reads of a frozen row must be identical.
        for r in 0..n {
            let ga = a.try_assignments_row(r).unwrap();
            let gb = a.try_assignments_row(r).unwrap();
            for kk in 0..k {
                assert_eq!(
                    ga[kk], gb[kk],
                    "row {r} atom {kk}: frozen-routing gate must be ρ-INVARIANT (the n-independence \
                     lever); {} at ρ_a vs {} at ρ_b",
                    ga[kk], gb[kk]
                );
            }
        }
    }

    #[test]
    fn frozen_routing_fixes_all_logits_and_thaw_restores_free_path_1033() {
        let (n, k) = (4usize, 3usize);
        let mut a = frozen(ordered_beta_bernoulli_assignment(n, k));
        // Under frozen routing EVERY logit is fixed (not a free Newton coord).
        let mask = a.fixed_logit_mask();
        assert_eq!(mask.len(), k);
        assert!(
            mask.iter().all(|&f| f),
            "frozen routing must fix ALL logits"
        );
        for kk in 0..k {
            assert!(
                a.logit_is_fixed(kk),
                "atom {kk} logit must be fixed under frozen routing"
            );
        }
        // Thawing restores the free-logit path (no fixed logits, no ungated).
        a.frozen_logits = None;
        assert!(!a.routing_is_frozen());
        assert!(
            a.fixed_logit_mask().iter().all(|&f| !f),
            "thaw must restore the free-logit path"
        );
    }
}

#[cfg(test)]
mod ungated_background_jvp_2933_tests {
    //! #2933 F04 — an ungated atom's gate is pinned at 1 after the row's gates are computed.
    //! For the column-separable sigmoid gates that pin is a derivative-consistent unit
    //! background; after a softmax over all K atoms it is a different map from the one every
    //! softmax derivative differentiates. Every configuration the forward seam admits must
    //! have a production logit JVP equal to a central difference of the production forward
    //! seam, and the configuration it cannot differentiate must be refused.
    use super::*;

    /// Decoder rows `γ_k` (K ≤ 3, p = 2), distinct per atom so no contraction cancels.
    const DECODED: [[f64; 2]; 3] = [[2.0, -0.7], [3.0, 1.1], [-1.3, 0.4]];

    fn one_row_assignment(
        mode: AssignmentMode,
        logits: &[f64],
        ungated: Vec<bool>,
    ) -> SaeAssignment {
        let k = logits.len();
        let mut assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
            Array2::from_shape_vec((1, k), logits.to_vec()).unwrap(),
            vec![Array2::<f64>::zeros((1, 1)); k],
            vec![LatentManifold::Euclidean; k],
            mode,
        )
        .unwrap();
        assignment.ungated = ungated;
        assignment
    }

    /// `Σ_k a_k γ_k`, read through the production forward seam.
    fn fitted_row(assignment: &SaeAssignment, decoded: &Array2<f64>) -> Array1<f64> {
        let gates = assignment
            .try_assignments_row(0)
            .expect("an admitted configuration evaluates");
        decoded.t().dot(&gates)
    }

    #[test]
    fn gate_logit_jvp_matches_forward_difference_for_every_admitted_ungated_mix_2933() {
        let modes = [
            AssignmentMode::softmax(0.7),
            AssignmentMode::ordered_beta_bernoulli(0.8, 1.3, false),
            AssignmentMode::threshold_gate(0.9, 0.2),
        ];
        let base_logits = [0.4_f64, -0.6, 0.9];
        let step = 1.0e-5;
        let mut refused = 0usize;
        let mut admitted = 0usize;
        for mode in modes {
            for k in [2usize, 3] {
                let decoded = Array2::from_shape_fn((k, 2), |(atom, out)| DECODED[atom][out]);
                // Every mask, including atom `K − 1`: softmax's reference-logit atom.
                for mask in 0..(1usize << k) {
                    let ungated: Vec<bool> = (0..k).map(|atom| mask & (1 << atom) != 0).collect();
                    let label = format!("{} K={k} ungated={ungated:?}", mode.family_label());
                    let assignment = one_row_assignment(mode, &base_logits[..k], ungated.clone());
                    let softmax = matches!(mode, AssignmentMode::Softmax { .. });
                    let gates = match assignment.try_assignments_row(0) {
                        Ok(gates) => gates,
                        Err(refusal) => {
                            assert!(
                                softmax && mask != 0,
                                "{label}: refused a configuration with a consistent derivative: {refusal}"
                            );
                            let mut scratch = vec![f64::NAN; k];
                            assert!(
                                assignment.try_assignments_row_into(0, &mut scratch).is_err(),
                                "{label}: the fill-into seam must refuse what the allocating seam refuses"
                            );
                            refused += 1;
                            continue;
                        }
                    };
                    admitted += 1;
                    let fitted = decoded.t().dot(&gates);
                    let mut jac = Array2::<f64>::zeros((assignment.row_block_dim(), 2));
                    fill_assignment_logit_jvp_rows(
                        assignment.mode,
                        assignment.logits.row(0),
                        gates.view(),
                        decoded.view(),
                        fitted.view(),
                        &assignment.fixed_logit_mask(),
                        &mut jac,
                    );
                    let mut live_slope = 0.0_f64;
                    for slot in 0..assignment.assignment_coord_dim() {
                        let mut plus = assignment.clone();
                        let mut minus = assignment.clone();
                        plus.logits[[0, slot]] += step;
                        minus.logits[[0, slot]] -= step;
                        let difference = (fitted_row(&plus, &decoded)
                            - fitted_row(&minus, &decoded))
                            / (2.0 * step);
                        for out in 0..2 {
                            assert!(
                                (jac[[slot, out]] - difference[out]).abs()
                                    <= 1.0e-8 + 1.0e-6 * difference[out].abs(),
                                "{label}: logit slot {slot}, output {out}: JVP {} vs forward \
                                 difference {}",
                                jac[[slot, out]],
                                difference[out]
                            );
                            live_slope = live_slope.max(difference[out].abs());
                        }
                    }
                    if ungated.iter().any(|&u| !u) {
                        assert!(
                            live_slope > 5.0e-2,
                            "{label}: no live logit slope ({live_slope:e})"
                        );
                    }
                    // Row sums over the simplex portion only; a sigmoid background is exactly 1.
                    if softmax {
                        assert!(
                            (gates.sum() - 1.0).abs() <= 1.0e-12,
                            "{label}: softmax row mass {}",
                            gates.sum()
                        );
                    }
                    for atom in 0..k {
                        if ungated[atom] {
                            assert_eq!(gates[atom], 1.0, "{label}: ungated gate {atom}");
                        } else {
                            assert!(
                                gates[atom] > 0.0 && gates[atom] < 1.0,
                                "{label}: gated atom {atom} = {}",
                                gates[atom]
                            );
                        }
                    }
                }
            }
        }
        // Softmax refuses its 3 + 7 nonzero masks; each sigmoid family admits all 4 + 8.
        assert_eq!(refused, 10, "softmax with any ungated atom must be refused");
        assert_eq!(admitted, 2 + 2 * 12);
    }

    #[test]
    fn softmax_with_ungated_background_is_refused_not_differentiated_as_another_map_2933() {
        // The audit's two-atom counterexample: decoder values 2 and 3, atom 1 (softmax's
        // reference-logit atom) ungated, zero logits, unit temperature. The overwritten
        // forward map is `2σ(ℓ) + 3`, with slope 0.5 at zero; the all-K softmax JVP reads
        // `0.5·(2 − 4) = −1`.
        let assignment =
            one_row_assignment(AssignmentMode::softmax(1.0), &[0.0, 0.0], vec![false, true]);
        let decoded = Array2::from_shape_vec((2, 1), vec![2.0, 3.0]).unwrap();
        match assignment.try_assignments_row(0) {
            Err(refusal) => {
                let mut scratch = [f64::NAN; 2];
                assert!(assignment.try_assignments_row_into(0, &mut scratch).is_err());
                let rho = SaeManifoldRho::new(0.0, 0.0, vec![Array1::zeros(1); 2])
                    .for_assignment(assignment.mode);
                let rho_refusal = assignment
                    .validate_rho_domain(&rho)
                    .expect_err("the rho-domain seam must refuse too");
                assert_eq!(rho_refusal, refusal);
            }
            Ok(gates) => {
                let fitted = decoded.t().dot(&gates);
                let mut jac = Array2::<f64>::zeros((assignment.row_block_dim(), 1));
                fill_assignment_logit_jvp_rows(
                    assignment.mode,
                    assignment.logits.row(0),
                    gates.view(),
                    decoded.view(),
                    fitted.view(),
                    &assignment.fixed_logit_mask(),
                    &mut jac,
                );
                let step = 1.0e-5;
                let mut plus = assignment.clone();
                let mut minus = assignment.clone();
                plus.logits[[0, 0]] += step;
                minus.logits[[0, 0]] -= step;
                let difference = (fitted_row(&plus, &decoded)[0]
                    - fitted_row(&minus, &decoded)[0])
                    / (2.0 * step);
                assert!(
                    (difference - 0.5).abs() <= 1.0e-8,
                    "the admitted forward map's slope is {difference}"
                );
                assert!(
                    (jac[[0, 0]] - difference).abs() <= 1.0e-8,
                    "softmax with an ungated background was admitted, but its logit JVP {} is \
                     not the forward slope {difference}",
                    jac[[0, 0]]
                );
            }
        }
    }

    #[test]
    fn softmax_ungated_refusal_is_typed_and_names_the_ungated_atoms_2933() {
        let logits = [0.3_f64, -0.2, 0.5];
        let check = |mode: AssignmentMode, ungated: Vec<bool>| {
            one_row_assignment(mode, &logits, ungated).validate_gate_configuration()
        };
        let softmax = AssignmentMode::softmax(0.8);
        assert_eq!(
            check(softmax, vec![false, false, true]),
            Err(GateConfigurationRefusal::SoftmaxWithUngatedAtoms {
                ungated_atoms: vec![2]
            }),
            "the reference-logit atom K - 1 must be named"
        );
        assert_eq!(
            check(softmax, vec![true, false, true]),
            Err(GateConfigurationRefusal::SoftmaxWithUngatedAtoms {
                ungated_atoms: vec![0, 2]
            })
        );
        assert_eq!(check(softmax, vec![false; 3]), Ok(()));
        for mode in [
            AssignmentMode::ordered_beta_bernoulli(0.8, 1.3, false),
            AssignmentMode::threshold_gate(0.9, 0.2),
            AssignmentMode::top_k_support(2),
        ] {
            assert_eq!(
                check(mode, vec![true, false, true]),
                Ok(()),
                "{}",
                mode.family_label()
            );
        }
        // Frozen routing holds every logit, so its softmax gates are constant in the free
        // logits and stay admitted.
        let mut frozen = one_row_assignment(softmax, &logits, vec![false; 3]);
        frozen.frozen_logits = Some(frozen.logits.clone());
        assert_eq!(frozen.validate_gate_configuration(), Ok(()));
        // The String seams carry the typed refusal's text.
        let refused = one_row_assignment(softmax, &logits, vec![false, true, false]);
        let expected = GateConfigurationRefusal::SoftmaxWithUngatedAtoms {
            ungated_atoms: vec![1],
        }
        .to_string();
        assert_eq!(refused.try_assignments_row(0), Err(expected.clone()));
        let rho = SaeManifoldRho::new(0.0, 0.0, vec![Array1::zeros(1); 3]).for_assignment(softmax);
        assert_eq!(refused.validate_rho_domain(&rho), Err(expected));
    }
}

#[cfg(test)]
mod threshold_gate_partition_2933_tests {
    //! #2933 F45 — the ThresholdGate energy `λz` on a gate `z ∈ (0, 1)` is a density only with
    //! its partition `Z(λ) = (1 − e^{−λ})/λ`. With the logit change of variables, the production
    //! prior value must integrate to one over the logit for every strength, threshold and
    //! temperature, and its log-strength derivative must have zero mean under that prior: the
    //! score identity `E[∂_ρ(−log p)] = 0` holds for a normalized density and fails for an
    //! energy whose normalizer depends on `ρ`. A derivative check against the same unnormalized
    //! scalar cannot see the difference, so these integrate instead.
    use super::*;
    use ndarray::array;

    /// Strengths from the small-`λ` series regime through deep saturation.
    const STRENGTHS: [f64; 5] = [1.0e-11, 0.05, 2.0, 30.0, 150.0];
    /// `(θ, τ)` gate frames.
    const FRAMES: [(f64, f64); 3] = [(0.0, 1.0), (0.7, 0.35), (-1.2, 2.5)];

    fn gate_assignment(logits: Array2<f64>, threshold: f64, temperature: f64) -> SaeAssignment {
        let (n, k) = logits.dim();
        SaeAssignment::from_blocks_with_mode_and_manifolds(
            logits,
            vec![Array2::<f64>::zeros((n, 1)); k],
            vec![LatentManifold::Euclidean; k],
            AssignmentMode::threshold_gate(temperature, threshold),
        )
        .expect("one logit column, coordinate block and manifold per atom")
    }

    fn strength_rho(assignment: &SaeAssignment, strength: f64) -> SaeManifoldRho {
        SaeManifoldRho::new(strength.ln(), 0.0, vec![Array1::zeros(1); assignment.k_atoms()])
            .for_assignment(assignment.mode)
    }

    /// The production negative log prior in logit coordinates: prior value plus Jacobian.
    fn negative_log_prior(assignment: &SaeAssignment, rho: &SaeManifoldRho) -> f64 {
        assignment_prior_value_weighted(assignment, rho, None).expect("admitted prior value")
            + gate_logit_jacobian_value_weighted(assignment, None)
    }

    /// `Σ h·f(x)` over `x = (ℓ − θ)/τ ∈ [−60, 60]` at `h = 1/16`. Each integrand below is
    /// `e^{−λσ(x)}σ(x)σ(−x)/Z` times a factor bounded by `1 + λ`, with `1/Z ≤ 1 + λ`. It is
    /// analytic in `|Im x| < π`, and on `|Im x| ≤ π/2` `Re σ ≥ 0` keeps `|e^{−λσ}| ≤ 1`, so the
    /// trapezoid error is below `(1 + λ)²·e^{−2π(π/2)·16}` and the dropped tails below
    /// `2(1 + λ)²e^{−60}`: both far under the bars, which are set by rounding in the sum.
    fn trapezoid(mut f: impl FnMut(f64) -> f64) -> f64 {
        let h = 1.0 / 16.0;
        (-960_i32..=960).map(|i| h * f(f64::from(i) * h)).sum()
    }

    /// With no data, prior plus Jacobian is a normalized density over the logit. Before the
    /// partition its mass was `Z(λ)`: `0.9754` at `λ = 0.05`, `0.4323` at `λ = 2`, `1/30` and
    /// `1/150` above.
    #[test]
    fn threshold_gate_prior_integrates_to_one_over_its_logit_2933() {
        for (threshold, temperature) in FRAMES {
            let mut assignment = gate_assignment(Array2::zeros((1, 1)), threshold, temperature);
            for strength in STRENGTHS {
                let rho = strength_rho(&assignment, strength);
                let mass = temperature
                    * trapezoid(|x| {
                        assignment.logits[[0, 0]] = threshold + temperature * x;
                        (-negative_log_prior(&assignment, &rho)).exp()
                    });
                assert!(
                    (mass - 1.0).abs() <= 1.0e-11,
                    "ThresholdGate prior at λ={strength}, θ={threshold}, τ={temperature} has \
                     no-data mass {mass:.15e}, not one"
                );
            }
        }
    }

    /// The log-strength derivative has zero mean under the prior at every strength and frame,
    /// and its partition part is `λ·∂_λ log Z = λ/(e^λ − 1) − 1`: `2·(−0.3434823572503343)` at
    /// `λ = 2` (audit §12 check 35), and `−λ/2 + λ²/12` near zero, where the difference form
    /// `λ/expm1(λ) − 1` would carry an absolute error of `ε`, i.e. a relative error of `2ε/λ`.
    #[test]
    fn threshold_gate_log_strength_derivative_has_zero_prior_mean_2933() {
        for (threshold, temperature) in FRAMES {
            let mut assignment = gate_assignment(Array2::zeros((1, 1)), threshold, temperature);
            for strength in STRENGTHS {
                let rho = strength_rho(&assignment, strength);
                let mean = temperature
                    * trapezoid(|x| {
                        assignment.logits[[0, 0]] = threshold + temperature * x;
                        let derivative =
                            assignment_prior_log_strength_derivative_weighted(&assignment, &rho, None)
                                .expect("admitted log-strength derivative");
                        derivative * (-negative_log_prior(&assignment, &rho)).exp()
                    });
                assert!(
                    mean.abs() <= 1.0e-11 * (1.0 + strength),
                    "ThresholdGate log-strength derivative at λ={strength}, θ={threshold}, \
                     τ={temperature} has prior mean {mean:.15e}, not zero"
                );
            }
        }
        let assignment = gate_assignment(array![[0.4]], 0.0, 1.0);
        let gate = 1.0 / (1.0 + (-0.4_f64).exp());
        let partition_slope = |strength: f64| {
            let rho = strength_rho(&assignment, strength);
            let derivative =
                assignment_prior_log_strength_derivative_weighted(&assignment, &rho, None)
                    .expect("admitted log-strength derivative");
            (derivative - strength * gate) / strength
        };
        let at_two = partition_slope(2.0);
        assert!(
            (at_two + 0.3434823572503343).abs() <= 1.0e-13,
            "∂_λ log Z(2) = {at_two:.16e}, expected −0.3434823572503343"
        );
        let small = 1.0e-11;
        let at_small = partition_slope(small);
        assert!(
            (at_small + 0.5 - small / 12.0).abs() <= 1.0e-9,
            "∂_λ log Z({small:e}) = {at_small:.16e}, expected −1/2 + λ/12"
        );
    }

    /// Under row weights and an ungated atom, the value carries `Σ w·log Z(λ)` over exactly the
    /// free gates, and the log-strength derivative is a central difference of that value.
    #[test]
    fn threshold_gate_partition_follows_free_gates_and_row_weights_2933() {
        let logits = array![[1.3, -0.4], [-2.2, 0.8], [0.1, 4.0]];
        let weights = [0.5, 1.0, 2.0];
        let (threshold, temperature) = (0.3_f64, 0.6_f64);
        let mut assignment = gate_assignment(logits.clone(), threshold, temperature);
        assignment.ungated = vec![false, true];
        let strength = 1.7_f64;
        let rho = strength_rho(&assignment, strength);
        let activation: f64 = (0..3)
            .map(|row| weights[row] / (1.0 + (-(logits[[row, 0]] - threshold) / temperature).exp()))
            .sum();
        let free_weight: f64 = weights.iter().sum();
        let log_partition = (-(-strength).exp_m1() / strength).ln();
        let value = assignment_prior_value_weighted(&assignment, &rho, Some(&weights))
            .expect("admitted prior value");
        let expected = strength * activation + free_weight * log_partition;
        assert!(
            (value - expected).abs() <= 1.0e-13 * (1.0 + expected.abs()),
            "weighted ThresholdGate prior value {value:.15e}, expected λ·Σw·z + Σw·log Z = \
             {expected:.15e}"
        );
        let h = 1.0e-4;
        let shifted = |delta: f64| {
            let mut moved = rho.clone();
            moved.log_lambda_sparse += delta;
            assignment_prior_value_weighted(&assignment, &moved, Some(&weights))
                .expect("admitted prior value")
        };
        let difference = (shifted(h) - shifted(-h)) / (2.0 * h);
        let derivative =
            assignment_prior_log_strength_derivative_weighted(&assignment, &rho, Some(&weights))
                .expect("admitted log-strength derivative");
        assert!(
            (derivative - difference).abs() <= 1.0e-7 * (1.0 + derivative.abs()),
            "log-strength derivative {derivative:.12e} vs central difference {difference:.12e}"
        );
    }
}
