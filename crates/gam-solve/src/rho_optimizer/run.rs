use super::*;
use gam_problem::{StationarityRung, StationarityStandard};

use super::asymptote_certificate::{
    AsymptoteSample, AsymptoteSide, AsymptoteTolerances, AsymptoteVerdict, AsymptoteWindow,
    MIN_TAIL_SAMPLES, assess_coordinate,
};
use super::rail_face::{
    RailFaceLimit, RailFaceLimitOutcome, RailFaceProof, RailFaceVerdict, certify_rail_face,
};

pub(crate) const OPERATOR_TRUST_RESTART_RADIUS_FLOOR: f64 = 1.0e-6;

/// Inner coefficient state bound to one exact outer seed.
///
/// A cached coefficient vector is only a valid initialization for the outer
/// coordinate that produced it.  Keeping the coordinate beside the vector
/// prevents a multi-start run from silently reusing one basin's coefficients
/// at a different seed.
#[derive(Clone, Debug)]
pub(crate) struct BoundInnerSeed {
    pub(crate) theta: Array1<f64>,
    pub(crate) beta: Array1<f64>,
}

pub(crate) fn outer_theta_bitwise_eq(left: &Array1<f64>, right: &Array1<f64>) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right.iter())
            .all(|(left, right)| left.to_bits() == right.to_bits())
}

/// Install cached inner state only at the exact outer coordinate that owns it.
///
/// This function is called after a seed-attempt reset and immediately before
/// the literal seed can be evaluated. Generated multistart candidates never
/// inherit coefficients from another outer point.
pub(crate) fn install_matching_initial_inner_seed(
    obj: &mut dyn OuterObjective,
    config: &OuterConfig,
    seed: &Array1<f64>,
    context: &str,
) -> Result<(), EstimationError> {
    let Some(bound) = config.initial_inner_seed.as_ref() else {
        return Ok(());
    };
    if !outer_theta_bitwise_eq(&bound.theta, seed) {
        return Ok(());
    }
    match obj.seed_inner_state(&bound.beta)? {
        SeedOutcome::Installed => log::info!(
            "[CACHE] beta-warm context={} theta_dim={} beta_dim={} action=installed",
            context,
            bound.theta.len(),
            bound.beta.len(),
        ),
        SeedOutcome::NoSlot => log::warn!(
            "[CACHE] beta-warm context={} theta_dim={} beta_dim={} action=skip \
             reason=objective_has_no_inner_beta_slot",
            context,
            bound.theta.len(),
            bound.beta.len(),
        ),
        SeedOutcome::Incompatible => log::info!(
            "[CACHE] beta-warm context={} theta_dim={} beta_dim={} action=rho-only \
             reason=seed_beta_incompatible_with_inner_state",
            context,
            bound.theta.len(),
            bound.beta.len(),
        ),
    }
    Ok(())
}

/// Hold terminal evidence at full inner-solve fidelity.
///
/// Search-time REML evaluations may deliberately cap P-IRLS.  A terminal
/// certificate or final state installation is a different operation: it must
/// run with the cap lifted, then restore the scheduler's value after the
/// operation completes.  The objective reset performed by the callers clears
/// any evaluation/P-IRLS entries created under the search state before the
/// full-fidelity request is made.
pub(crate) struct TerminalInnerCapGuard<'a> {
    cap: &'a AtomicUsize,
    previous: usize,
}

impl<'a> TerminalInnerCapGuard<'a> {
    pub(crate) fn lift(feedback: &'a InnerProgressFeedback) -> Self {
        let cap = feedback.cap.as_ref();
        let previous = cap.swap(0, Ordering::Relaxed);
        Self { cap, previous }
    }
}

impl Drop for TerminalInnerCapGuard<'_> {
    fn drop(&mut self) {
        self.cap.store(self.previous, Ordering::Relaxed);
    }
}

/// Configuration for the outer optimization runner.
#[derive(Clone, Debug)]
pub(crate) struct OuterConfig {
    pub(crate) tolerance: f64,
    /// Optional override for the *relative-cost-decrease* convergence stop,
    /// decoupled from `tolerance`. `outer_gradient_tolerance` normally derives
    /// BOTH the absolute projected-gradient floor
    /// (`max(tolerance, scale·√ε_machine)`)
    /// AND the relative-cost stop (`rel_cost = tolerance`) from the single
    /// `tolerance`. That conflation forces a caller who needs a *tight absolute
    /// floor* (to resolve λ to the genuine REML optimum at large `n`, where the
    /// floor is `scale·√ε_machine`) to also accept a *tight rel-cost stop*,
    /// which on a flat REML ridge never trips and grinds the optimizer to `max_iter` —
    /// dozens of surplus O(D·p³) Laplace-derivative outer iterations (the #1082
    /// multinomial smooth-by-factor wall-clock blow-up). When `Some(r)`, the
    /// rel-cost stop uses `r` while the absolute floor keeps using `tolerance`
    /// via `objective_scale`, so accuracy (absolute floor) and perf (loose
    /// rel-cost) are selected independently. `None` preserves the legacy coupling
    /// (`rel_cost = tolerance`) for every existing path byte-for-byte.
    pub(crate) rel_cost_tolerance: Option<f64>,
    pub(crate) max_iter: usize,
    /// The model's canonical feasible outer domain. Every stationarity
    /// certificate and rail report reasons against this box.
    pub(crate) model_domain_bounds: Option<(Array1<f64>, Array1<f64>)>,
    /// A temporary algorithmic subspace used only by an active-set polish.
    /// It may narrow the model domain but never changes the feasible cone that
    /// screening or mint is allowed to certify.
    pub(crate) search_bounds_override: Option<(Array1<f64>, Array1<f64>)>,
    pub(crate) seed_config: gam_problem::SeedConfig,
    pub(crate) rho_bound: f64,
    pub(crate) heuristic_lambdas: Option<Vec<f64>>,
    pub(crate) initial_rho: Option<Array1<f64>>,
    pub(crate) initial_inner_seed: Option<BoundInnerSeed>,
    pub(crate) fallback_policy: FallbackPolicy,
    pub(crate) screening_cap: Option<Arc<AtomicUsize>>,
    pub(crate) screen_initial_rho: bool,
    /// Outer-aware inner-PIRLS iteration cap (sibling of `screening_cap`).
    /// When set, the BFGS bridge drives this atomic on every accepted
    /// gradient eval to coarsen the inner Newton solve at early outer iters
    /// (when ρ is far from converged) and lift it back to full as
    /// convergence approaches. Distinct from `screening_cap` in that it
    /// does NOT suppress cache writes / warm-start updates / KKT
    /// enforcement; it is purely a budget. See
    /// `RemlObjectiveState::outer_inner_cap` for dual-cap semantics.
    pub(crate) outer_inner_cap: Option<InnerProgressFeedback>,
    pub(crate) operator_initial_trust_radius: Option<f64>,
    pub(crate) arc_initial_regularization: Option<f64>,
    /// Optional scale factor for the objective's natural magnitude.
    /// Used to widen the absolute gradient-norm floor on objectives whose
    /// gradient lives on a non-unit scale (e.g. Gaussian-identity REML at
    /// large `n`, whose ∂/∂logλ inherits the O(n) likelihood constant).
    /// `None` falls back to the bare `tolerance` floor.
    pub(crate) objective_scale: Option<f64>,
    /// BFGS line-search infinity-norm cap applied to the leading `rho_dim`
    /// outer parameters (log-λ axes). Documented natural step for
    /// `log(lambda)` is ≈ 5 (`e^5 ≈ 148`-fold smoothing-parameter change
    /// per accepted outer iter — matches typical quasi-Newton direction
    /// magnitude on flat REML surfaces). Setting this `None` disables the
    /// rho-axis cap entirely.
    pub(crate) bfgs_step_cap: Option<f64>,
    /// BFGS line-search infinity-norm cap applied to the trailing `psi_dim`
    /// outer parameters (kappa / aniso-log-scale axes). Required because
    /// the kernel scale axes need much tighter control (`e^1 ≈ 2.7`-fold
    /// per iter is plenty) — using the rho-axis cap here lets the optimizer
    /// jump kappa by orders of magnitude per step and oscillate. Setting
    /// this `None` disables the psi-axis cap.
    pub(crate) bfgs_step_cap_psi: Option<f64>,
    /// Optional persistent-cache session. When `Some`, every finite objective
    /// evaluation is written through to disk (rate-limited, atomic-rename)
    /// and the best on-disk rho is prepended as a seed at the start of each
    /// plan attempt. Defaulted off so test-only paths skip filesystem I/O.
    pub(crate) cache_session: Option<Arc<CacheSession>>,
    /// Optional mirror cache sessions. Checkpoints and successful finalize
    /// writes are also written to each of these sessions (different keys,
    /// shared store). Used for hierarchical broadcast: the current best ρ is
    /// written to the exact-key (primary) AND the data-independent
    /// seed-prefix key so the next fit with related structure can warm-start
    /// from this one, even after an interrupted run.
    pub(crate) cache_mirror_sessions: Vec<Arc<CacheSession>>,
    pub(crate) rho_uncertainty_problem_size: crate::rho_uncertainty::RhoUncertaintyProblemSize,
    /// Converged exact outer Hessian `H(θ̂)` transferred from a prior
    /// structurally-matching fit via the persistent cache (a warm-start *hit*),
    /// in the full θ layout. When present and SPD, the BFGS host path seeds its
    /// iter-0 metric with `InitialMetric::DenseInverseHessian(H⁻¹)` so the first
    /// outer step is quasi-Newton instead of unscaled steepest descent — the
    /// dominant LOSO line-search-bracketing cost (each bracketing probe is a
    /// full inner joint-Newton re-solve). Strictly stronger than the scalar
    /// `1/‖g₀‖` metric: it carries the full anisotropic curvature, which across
    /// folds (one held-out point) is nearly identical to this fold's. Never
    /// changes the converged optimum — BFGS reaches `∇V=0` under any SPD initial
    /// metric. `None` on every cold-start / no-cache / pre-Hessian-schema path,
    /// which falls back to the scalar warm metric byte-for-byte.
    pub(crate) warm_start_outer_hessian: Option<Array2<f64>>,
    /// Per-ρ-coordinate structural keys, in the objective's NATIVE (formula)
    /// coordinate order, used to make the outer smoothing-parameter search
    /// invariant to the order the user wrote the smooth terms / tensor margins
    /// (#1538/#1539).
    ///
    /// When `Some` and the keys induce a non-identity canonical permutation,
    /// [`run_outer`] reorders the coordinate layout the optimizer sees into a
    /// stable canonical order (derived purely from the keys, never from the
    /// native position) before seeding/optimizing, and inverts the permutation
    /// on the returned ρ / gradient / Hessian so the caller still receives the
    /// native layout. Seeding, multistart and tie-breaking then all operate on
    /// the identical canonical layout for every term order, so both orders
    /// reach the same λ̂ and the same fitted surface. `None` (or an identity
    /// permutation) leaves the legacy native-order path byte-for-byte unchanged.
    pub(crate) rho_canonical_keys: Option<Vec<u64>>,
}

impl Default for OuterConfig {
    fn default() -> Self {
        Self {
            tolerance: 1e-5,
            rel_cost_tolerance: None,
            max_iter: 200,
            model_domain_bounds: None,
            search_bounds_override: None,
            seed_config: gam_problem::SeedConfig::default(),
            rho_bound: 30.0,
            heuristic_lambdas: None,
            initial_rho: None,
            initial_inner_seed: None,
            fallback_policy: FallbackPolicy::Automatic,
            screening_cap: None,
            screen_initial_rho: false,
            outer_inner_cap: None,
            operator_initial_trust_radius: None,
            arc_initial_regularization: None,
            objective_scale: None,
            bfgs_step_cap: None,
            bfgs_step_cap_psi: None,
            cache_session: None,
            cache_mirror_sessions: Vec::new(),
            rho_uncertainty_problem_size:
                crate::rho_uncertainty::RhoUncertaintyProblemSize::default(),
            warm_start_outer_hessian: None,
            rho_canonical_keys: None,
        }
    }
}

// ─── OuterProblem builder ─────────────────────────────────────────────
//
// Declarative builder for outer optimization problems.  Derives
// OuterCapability flags from high-level inputs (gradient/hessian
// availability, psi dimension, EFS eligibility) so call sites never
// hand-copy capability flags.

/// Declarative outer-problem builder.  Produces both the
/// [`OuterCapability`] (what the objective can provide) and the
/// [`OuterConfig`] (how the runner should behave) from a small set
/// of high-level declarations.
pub struct OuterProblem {
    n_params: usize,
    gradient: Derivative,
    hessian: DeclaredHessianForm,
    prefer_gradient_only: bool,
    disable_fixed_point: bool,
    psi_dim: usize,
    barrier_config: Option<BarrierConfig>,
    tolerance: f64,
    rel_cost_tolerance: Option<f64>,
    max_iter: usize,
    bounds: Option<(Array1<f64>, Array1<f64>)>,
    rho_bound: f64,
    seed_config: gam_problem::SeedConfig,
    heuristic_lambdas: Option<Vec<f64>>,
    initial_rho: Option<Array1<f64>>,
    fallback_policy: FallbackPolicy,
    screening_cap: Option<Arc<AtomicUsize>>,
    screen_initial_rho: bool,
    outer_inner_cap: Option<InnerProgressFeedback>,
    operator_initial_trust_radius: Option<f64>,
    arc_initial_regularization: Option<f64>,
    objective_scale: Option<f64>,
    bfgs_step_cap: Option<f64>,
    bfgs_step_cap_psi: Option<f64>,
    cache_session: Option<Arc<CacheSession>>,
    cache_mirror_sessions: Vec<Arc<CacheSession>>,
    rho_uncertainty_problem_size: crate::rho_uncertainty::RhoUncertaintyProblemSize,
    rho_canonical_keys: Option<Vec<u64>>,
}

impl OuterProblem {
    pub fn new(n_params: usize) -> Self {
        Self {
            n_params,
            gradient: Derivative::Unavailable,
            hessian: DeclaredHessianForm::Unavailable,
            // Closed-form/test objectives preserve explicit ARC availability;
            // production family-ladder entry points opt into #2359's
            // optimize-3/certify-4 protocol with
            // `with_prefer_gradient_only(true)`.
            prefer_gradient_only: false,
            disable_fixed_point: false,
            psi_dim: 0,
            barrier_config: None,
            tolerance: 1e-5,
            rel_cost_tolerance: None,
            max_iter: 200,
            bounds: None,
            rho_bound: 30.0,
            seed_config: gam_problem::SeedConfig::default(),
            heuristic_lambdas: None,
            initial_rho: None,
            fallback_policy: FallbackPolicy::Automatic,
            screening_cap: None,
            screen_initial_rho: false,
            outer_inner_cap: None,
            operator_initial_trust_radius: None,
            arc_initial_regularization: None,
            objective_scale: None,
            bfgs_step_cap: None,
            bfgs_step_cap_psi: None,
            cache_session: None,
            cache_mirror_sessions: Vec::new(),
            rho_uncertainty_problem_size:
                crate::rho_uncertainty::RhoUncertaintyProblemSize::default(),
            rho_canonical_keys: None,
        }
    }

    /// Supply per-ρ-coordinate structural keys (native/formula order) so the
    /// outer search is canonicalized to be invariant to the order the smooth
    /// terms / tensor margins were written (#1538/#1539). See
    /// [`OuterConfig::rho_canonical_keys`].
    pub fn with_rho_canonical_keys(mut self, keys: Option<Vec<u64>>) -> Self {
        self.rho_canonical_keys = keys;
        self
    }

    pub fn with_gradient(mut self, d: Derivative) -> Self {
        self.gradient = d;
        self
    }
    pub fn with_hessian(mut self, form: DeclaredHessianForm) -> Self {
        self.hessian = form;
        self
    }
    /// Choose whether analytic Hessian work is reserved for terminal
    /// certification (`true`, the generic derivative-ladder protocol) or may
    /// also be consumed by the search (`false`, for closed-form objectives).
    pub fn with_prefer_gradient_only(mut self, prefer_gradient_only: bool) -> Self {
        self.prefer_gradient_only = prefer_gradient_only;
        self
    }
    /// Forbid the planner from selecting EFS/HybridEfs, even when the
    /// objective implements `eval_efs()` and the coordinate structure would
    /// otherwise make pure/hybrid EFS eligible.
    ///
    /// Callers use this for families where the Wood-Fasiolo structural
    /// property is known not to hold (e.g. GAMLSS/location-scale with
    /// β-dependent joint Hessian), so EFS would stagnate and burn budget
    /// before the automatic cascade falls back to gradient-based BFGS.
    pub fn with_disable_fixed_point(mut self, disable: bool) -> Self {
        self.disable_fixed_point = disable;
        self
    }
    // MEASURE-JET ψ REGISTRATION: the engine below is already complete for a
    // 3-coordinate measure-jet ψ group (s, α, ln τ) — `psi_dim` is generic,
    // `with_bounds` carries the s ∈ (0, 2) box (the same convention matern κ
    // uses for its log-κ window; no logistic reparameterization exists or is
    // needed in-house), `with_bfgs_step_cap_psi` caps per-iteration ψ moves,
    // and `DirectionalHyperParam::new_compact` (solver/reml/mod.rs) carries
    // penalty-only first/second/cross jets with `is_penalty_like`
    // auto-derived from the identically-zero design drift (∂X/∂ψ ≡ 0).
    // Every remaining registration arm is formula-layer dispatch in
    // src/terms/smooth.rs (eligibility in
    // `spatial_term_supports_hyper_optimization`, dims in
    // `spatial_dims_per_term`, seed/bounds/write-back on
    // `SpatialLogKappaCoords`, the per-trial rebuild in
    // `apply_log_kappa_to_term`, and the derivative bundle in
    // `try_build_spatial_term_log_kappa_derivative`, which currently returns
    // `Ok(None)` for `SmoothBasisSpec::MeasureJet`) plus the
    // `build_measure_jet_basis_psi_derivatives` producer in
    // src/terms/basis/measure_jet_smooth.rs; both are owned by the
    // measure-jet terms actor. Registration stays gated on those arms — do
    // NOT add measure-jet-specific branches to this engine.
    pub fn with_psi_dim(mut self, dim: usize) -> Self {
        self.psi_dim = dim;
        self
    }
    pub fn with_barrier(mut self, cfg: Option<BarrierConfig>) -> Self {
        self.barrier_config = cfg;
        self
    }
    pub fn with_tolerance(mut self, tol: f64) -> Self {
        self.tolerance = tol;
        self
    }
    pub fn with_max_iter(mut self, n: usize) -> Self {
        self.max_iter = n;
        self
    }
    pub fn with_bounds(mut self, lo: Array1<f64>, hi: Array1<f64>) -> Self {
        self.bounds = Some((lo, hi));
        self
    }
    pub fn with_rho_bound(mut self, b: f64) -> Self {
        self.rho_bound = b;
        self
    }
    pub fn with_seed_config(mut self, sc: gam_problem::SeedConfig) -> Self {
        self.seed_config = sc;
        self
    }
    pub fn with_heuristic_lambdas(mut self, h: Vec<f64>) -> Self {
        self.heuristic_lambdas = Some(h);
        self
    }
    pub fn with_initial_rho(mut self, rho: Array1<f64>) -> Self {
        self.initial_rho = Some(rho);
        self
    }
    pub fn with_screening_cap(mut self, screening_cap: Arc<AtomicUsize>) -> Self {
        self.screening_cap = Some(screening_cap);
        self
    }
    /// Allow seed screening to rank the explicit initial rho against generated
    /// candidates even when the effective seed budget is one. The default keeps
    /// a user-provided initial point authoritative and avoids a separate
    /// screening pass.
    pub fn with_screen_initial_rho(mut self, screen_initial_rho: bool) -> Self {
        self.screen_initial_rho = screen_initial_rho;
        self
    }
    /// Wire the bidirectional inner-PIRLS feedback channel.
    ///
    /// The outer bridge writes a coarsened iteration cap into
    /// `feedback.cap` on every accepted gradient/Hessian eval; the inner
    /// solver writes back into `feedback.last_iters` /
    /// `feedback.last_converged` after each non-screening solve so the
    /// next outer iter's schedule can adapt to the inner solver's
    /// actual convergence behavior. Typical caller passes
    /// `InnerProgressFeedback {
    ///     cap: Arc::clone(&reml_state.outer_inner_cap),
    ///     last_iters: Arc::clone(&reml_state.last_inner_iters),
    ///     last_converged: Arc::clone(&reml_state.last_inner_converged),
    /// }` so the inner and outer observe the same atomics.
    pub fn with_outer_inner_cap(mut self, feedback: InnerProgressFeedback) -> Self {
        self.outer_inner_cap = Some(feedback);
        self
    }

    /// Wire a one-shot "re-evaluate the inner solve COLD" signal that the outer
    /// cost-stall guard raises when it grants a STUCK-stall escape (#2349).
    ///
    /// A profiled objective whose inner solve is warm-started along the outer
    /// trajectory can carry value HYSTERESIS on a near-flat inner ridge — the
    /// multinomial simplex-boundary regime where the softmax Fisher weight
    /// `diag(p) − ppᵀ` collapses is the motivating case: two warm starts
    /// converge to different ridge points whose Laplace `½log|H(β)|`, hence the
    /// profiled objective, differ by more than the outer descent resolution, so
    /// the optimizer's step-acceptance cannot separate real descent from that
    /// hysteresis and grinds to `max_iter` at a non-stationary point. Uncapping
    /// the inner cycle budget does not cure it (a fully converged warm solve
    /// still lands on the warm-biased ridge point); the objective must re-solve
    /// COLD to see a consistent surface.
    ///
    /// The caller shares this `Arc<AtomicBool>` with its objective closure and
    /// consults it there, re-solving the inner problem from a canonical seed
    /// (dropping the warm cache) whenever the flag is raised. The signal rides
    /// the internal inner-cap feedback channel, but its `cap` slot is a private
    /// throwaway so wiring the signal never perturbs the caller's own inner-cap
    /// scheduling (custom families hold their real inner cap separately).
    /// Objectives that do not warm-start, or never near-separate, simply never
    /// observe the flag raised.
    pub fn with_stuck_stall_cold_reeval_signal(self, signal: Arc<AtomicBool>) -> Self {
        self.with_outer_inner_cap(InnerProgressFeedback {
            cap: Arc::new(AtomicUsize::new(0)),
            accepted_iter: Arc::new(AtomicUsize::new(0)),
            // `last_iters == 0` ⇒ `snapshot()` returns `None` ⇒ no cap-schedule
            // adaptation is derived from this dummy; `last_converged == true`
            // matches the `None` default of `inner_solve_converged`, so
            // terminal-fidelity gating is byte-for-byte unchanged.
            last_iters: Arc::new(AtomicUsize::new(0)),
            last_converged: Arc::new(AtomicBool::new(true)),
            ift_residual: Arc::new(AtomicU64::new(f64::NAN.to_bits())),
            accept_rho: Arc::new(AtomicU64::new(f64::NAN.to_bits())),
            force_cold: signal,
        })
    }
    pub fn with_operator_initial_trust_radius(mut self, radius: Option<f64>) -> Self {
        self.operator_initial_trust_radius = sanitized_operator_trust_restart_radius(radius);
        self
    }

    /// Override the ARC initial cubic-regularization parameter sigma
    /// (default in `opt`: 1.0). Smaller sigma → less cubic penalty on the
    /// first step → larger first move on benign objectives. The matrix-
    /// free Newton-TR analog is `with_operator_initial_trust_radius`.
    ///
    /// Used by Gaussian-identity REML at large-scale n: the objective is
    /// quadratic-like in log-λ near the optimum (sigma is the right
    /// scale), and log-λ moves of 2–4 units in the early iters
    /// otherwise burn 4–8 iters of trust-region expansion before the
    /// model trusts the analytic Hessian.
    pub fn with_arc_initial_regularization(mut self, sigma: Option<f64>) -> Self {
        self.arc_initial_regularization = sigma.filter(|v| v.is_finite() && *v > 0.0);
        self
    }

    /// Set the objective's natural magnitude scale, used to derive an
    /// `n`-aware absolute gradient-norm floor. When set to `Some(s)`,
    /// the runner uses `abs_floor = max(tol, s * √ε_machine)` for the
    /// projected-gradient convergence check.
    ///
    /// Rationale: a fixed `abs = tol` (e.g. 1e-6) is appropriate when the
    /// objective and its gradient live on a unit scale, but Gaussian-
    /// identity REML carries an O(n) likelihood constant that flows into
    /// ∂/∂logλ. At large-scale n the floor becomes binding even when the
    /// relative-from-seed component (`rel_initial_grad * ‖g0‖`) declared
    /// convergence iters earlier — chasing sub-ULP changes in log-λ at
    /// the cost of repeated k²·n·p² analytic-Hessian assemblies.
    pub fn with_objective_scale(mut self, scale: Option<f64>) -> Self {
        self.objective_scale = scale.filter(|v| v.is_finite() && *v > 0.0);
        self
    }

    /// Decouple the *relative-cost-decrease* convergence stop from the
    /// absolute projected-gradient floor. By default both are derived from the
    /// single `with_tolerance` value (`abs = max(tol, scale·√ε_machine)`,
    /// `rel_cost = tol`). Supplying `Some(r)` here makes the rel-cost stop use
    /// `r` while the absolute floor keeps using `tolerance` (so a caller can
    /// keep a tight absolute floor for accuracy at large `n` AND a loose
    /// rel-cost stop for perf on a flat REML ridge — see #1082). `None` keeps
    /// the legacy coupling.
    pub fn with_rel_cost_tolerance(mut self, rel_cost: Option<f64>) -> Self {
        self.rel_cost_tolerance = rel_cost.filter(|v| v.is_finite() && *v > 0.0);
        self
    }

    /// Cap the infinity-norm displacement of BFGS cost-only line-search probes
    /// on the **rho axes** (the first `n_params - psi_dim` outer parameters,
    /// = log-λ). Also scales the initial inverse metric so the first trial
    /// direction respects the same local budget coordinate-wise. Documented
    /// natural step on log-λ is ≈ 5; tighter values throttle BFGS and starve
    /// convergence on flat REML valleys.
    pub fn with_bfgs_step_cap(mut self, cap: Option<f64>) -> Self {
        self.bfgs_step_cap = cap.filter(|v| v.is_finite() && *v > 0.0);
        self
    }

    /// Cap the infinity-norm displacement of BFGS cost-only line-search probes
    /// on the **psi axes** (the trailing `psi_dim` outer parameters, = kappa
    /// or anisotropic log-scales). Mirrors [`Self::with_bfgs_step_cap`] but
    /// scoped to kernel-scale parameters whose natural step is much smaller
    /// than log-λ (≈ ln 2 per iter keeps kappa from oscillating). Without
    /// this split, a uniform rho-scale cap lets psi explode while a uniform
    /// psi-scale cap throttles rho — both fail the survival-marginal-slope
    /// path at large scale, where rho needs |d|≈5 while psi wants |d|≤1.
    pub fn with_bfgs_step_cap_psi(mut self, cap: Option<f64>) -> Self {
        self.bfgs_step_cap_psi = cap.filter(|v| v.is_finite() && *v > 0.0);
        self
    }

    pub fn with_cache_session(mut self, session: Arc<CacheSession>) -> Self {
        self.cache_session = Some(session);
        self
    }

    /// Attach mirror cache sessions that receive a broadcast copy of
    /// the final-result finalize write. See
    /// [`OuterConfig::cache_mirror_sessions`].
    pub fn with_cache_mirror_sessions(mut self, sessions: Vec<Arc<CacheSession>>) -> Self {
        self.cache_mirror_sessions = sessions;
        self
    }

    pub fn with_problem_size(mut self, n_obs: usize, p_coefficients: usize) -> Self {
        self.rho_uncertainty_problem_size = crate::rho_uncertainty::RhoUncertaintyProblemSize {
            n_obs: Some(n_obs),
            p_coefficients: Some(p_coefficients),
        };
        self
    }

    /// Override the fallback policy. Default is [`FallbackPolicy::Automatic`].
    ///
    /// Set [`FallbackPolicy::Disabled`] when the caller requires the primary
    /// plan to stand on its own. Exact-Hessian objectives use this to ensure
    /// failures surface on the analytic geometry instead of being reinterpreted
    /// by a different optimizer class.
    pub fn with_fallback_policy(mut self, policy: FallbackPolicy) -> Self {
        self.fallback_policy = policy;
        self
    }

    /// Derive the capability flags from the builder state.
    /// `fixed_point_available` is set to `false` here; `build_objective`
    /// overrides it based on whether an EFS closure is actually provided.
    fn capability(&self) -> OuterCapability {
        OuterCapability {
            gradient: self.gradient,
            hessian: self.hessian,
            prefer_gradient_only: self.prefer_gradient_only,
            disable_fixed_point: self.disable_fixed_point,
            n_params: self.n_params,
            psi_dim: self.psi_dim,
            fixed_point_available: false,
            barrier_config: self.barrier_config.clone(),
        }
    }

    /// Derive the runner configuration from the builder state.
    pub(crate) fn config(&self) -> OuterConfig {
        OuterConfig {
            tolerance: self.tolerance,
            rel_cost_tolerance: self.rel_cost_tolerance,
            max_iter: self.max_iter,
            model_domain_bounds: self.bounds.clone(),
            search_bounds_override: None,
            seed_config: self.seed_config,
            rho_bound: self.rho_bound,
            heuristic_lambdas: self.heuristic_lambdas.clone(),
            initial_rho: self.initial_rho.clone(),
            initial_inner_seed: None,
            fallback_policy: self.fallback_policy,
            screening_cap: self.screening_cap.clone(),
            screen_initial_rho: self.screen_initial_rho,
            outer_inner_cap: self.outer_inner_cap.clone(),
            operator_initial_trust_radius: self.operator_initial_trust_radius,
            arc_initial_regularization: self.arc_initial_regularization,
            objective_scale: self.objective_scale,
            bfgs_step_cap: self.bfgs_step_cap,
            bfgs_step_cap_psi: self.bfgs_step_cap_psi,
            cache_session: self.cache_session.clone(),
            cache_mirror_sessions: self.cache_mirror_sessions.clone(),
            rho_uncertainty_problem_size: self.rho_uncertainty_problem_size,
            // Populated only by the persistent-cache resume path in `run` after
            // a warm-start hit decodes a converged outer Hessian.
            warm_start_outer_hessian: None,
            rho_canonical_keys: self.rho_canonical_keys.clone(),
        }
    }

    /// Construct a [`ClosureObjective`] with capability flags derived from the
    /// builder state **and** the closures actually provided.
    ///
    /// `fixed_point_available` is set to `true` when `efs_fn` is `Some`,
    /// regardless of whether `.with_efs()` was called.  This is the canonical
    /// way to create production objectives — it eliminates the drift risk of
    /// manually entering capability flags.
    pub fn build_objective<S, Fc, Fe, Fr, Fefs>(
        &self,
        state: S,
        cost_fn: Fc,
        eval_fn: Fe,
        reset_fn: Option<Fr>,
        efs_fn: Option<Fefs>,
    ) -> ClosureObjective<S, Fc, Fe, Fr, Fefs>
    where
        Fc: FnMut(&mut S, &Array1<f64>) -> Result<f64, EstimationError>,
        Fe: FnMut(&mut S, &Array1<f64>) -> Result<OuterEval, EstimationError>,
        Fr: FnMut(&mut S),
        Fefs: FnMut(&mut S, &Array1<f64>) -> Result<EfsEval, EstimationError>,
    {
        let mut cap = self.capability();
        // Derive fixed_point_available from whether the caller actually
        // provided an EFS hook, rather than relying on manual flags.
        cap.fixed_point_available = efs_fn.is_some();
        ClosureObjective {
            state,
            cap,
            cost_fn,
            eval_fn,
            eval_order_fn: None,
            reset_fn,
            efs_fn,
            fixed_point_certificate_fn: None,
            exact_polish_fn: None,
            rail_face_limit_fn: None,
            screening_proxy_fn: None::<fn(&mut S, &Array1<f64>) -> Result<f64, EstimationError>>,
            seed_fn: None::<fn(&mut S, &Array1<f64>) -> Result<SeedOutcome, EstimationError>>,
            terminal_eval_order: None,
        }
    }

    /// Construct a [`ClosureObjective`] with an order-aware evaluation hook.
    ///
    /// This lets the runner request first-order vs second-order work based on
    /// the active outer plan while preserving the legacy eager `eval_fn`.
    pub fn build_objective_with_eval_order<S, Fc, Fe, Feo, Fr, Fefs>(
        &self,
        state: S,
        cost_fn: Fc,
        eval_fn: Fe,
        eval_order_fn: Feo,
        reset_fn: Option<Fr>,
        efs_fn: Option<Fefs>,
    ) -> ClosureObjective<S, Fc, Fe, Fr, Fefs, Feo>
    where
        Fc: FnMut(&mut S, &Array1<f64>) -> Result<f64, EstimationError>,
        Fe: FnMut(&mut S, &Array1<f64>) -> Result<OuterEval, EstimationError>,
        Feo: FnMut(&mut S, &Array1<f64>, OuterEvalOrder) -> Result<OuterEval, EstimationError>,
        Fr: FnMut(&mut S),
        Fefs: FnMut(&mut S, &Array1<f64>) -> Result<EfsEval, EstimationError>,
    {
        let mut cap = self.capability();
        cap.fixed_point_available = efs_fn.is_some();
        ClosureObjective {
            state,
            cap,
            cost_fn,
            eval_fn,
            eval_order_fn: Some(eval_order_fn),
            reset_fn,
            efs_fn,
            fixed_point_certificate_fn: None,
            exact_polish_fn: None,
            rail_face_limit_fn: None,
            screening_proxy_fn: None::<fn(&mut S, &Array1<f64>) -> Result<f64, EstimationError>>,
            seed_fn: None::<fn(&mut S, &Array1<f64>) -> Result<SeedOutcome, EstimationError>>,
            terminal_eval_order: None,
        }
    }

    /// Construct a [`ClosureObjective`] with both an order-aware evaluation
    /// hook and a custom seed-screening ranking proxy. The proxy fires only
    /// when the cascade in `rank_seeds_with_screening` calls it; outside
    /// screening the regular cost path is unaffected.
    pub fn build_objective_with_screening_proxy<S, Fc, Fe, Feo, Fr, Fefs, Fsp>(
        &self,
        state: S,
        cost_fn: Fc,
        eval_fn: Fe,
        eval_order_fn: Feo,
        reset_fn: Option<Fr>,
        efs_fn: Option<Fefs>,
        screening_proxy_fn: Fsp,
    ) -> ClosureObjective<S, Fc, Fe, Fr, Fefs, Feo, Fsp>
    where
        Fc: FnMut(&mut S, &Array1<f64>) -> Result<f64, EstimationError>,
        Fe: FnMut(&mut S, &Array1<f64>) -> Result<OuterEval, EstimationError>,
        Feo: FnMut(&mut S, &Array1<f64>, OuterEvalOrder) -> Result<OuterEval, EstimationError>,
        Fr: FnMut(&mut S),
        Fefs: FnMut(&mut S, &Array1<f64>) -> Result<EfsEval, EstimationError>,
        Fsp: FnMut(&mut S, &Array1<f64>) -> Result<f64, EstimationError>,
    {
        let mut cap = self.capability();
        cap.fixed_point_available = efs_fn.is_some();
        ClosureObjective {
            state,
            cap,
            cost_fn,
            eval_fn,
            eval_order_fn: Some(eval_order_fn),
            reset_fn,
            efs_fn,
            fixed_point_certificate_fn: None,
            exact_polish_fn: None,
            rail_face_limit_fn: None,
            screening_proxy_fn: Some(screening_proxy_fn),
            seed_fn: None::<fn(&mut S, &Array1<f64>) -> Result<SeedOutcome, EstimationError>>,
            terminal_eval_order: None,
        }
    }

    /// Run the outer optimization with a given objective.
    pub fn run(
        &self,
        obj: &mut dyn OuterObjective,
        context: &str,
    ) -> Result<OuterResult, EstimationError> {
        let mut config = self.config();
        let objective_lower = obj.outer_domain_lower_bound()?;
        let objective_upper = obj.outer_domain_upper_bound()?;
        if objective_lower.is_some() || objective_upper.is_some() {
            install_objective_domain(&mut config, self.n_params, objective_lower, objective_upper)?;
        }
        let Some(session) = config.cache_session.clone() else {
            return run_outer(obj, &config, context);
        };
        let key_hex = session.key().to_hex();
        let short_key = &key_hex[..8.min(key_hex.len())];
        let mut had_hit = false;
        let mut cached_inner_seed: Option<BoundInnerSeed> = None;
        if let Some(loaded) = session.try_load_with_source() {
            match classify_cache_entry_for_outer(&loaded, self.n_params) {
                CacheSeedDecision::ExactFinal {
                    rho,
                    beta,
                    iterations,
                    prior_obj_display,
                } => {
                    log::info!(
                        "[CACHE] final-hit key={}.. context={} rho_dim={} prior_obj={:.6e} iter={} action=resume-and-recertify",
                        short_key,
                        context,
                        rho.len(),
                        prior_obj_display,
                        iterations,
                    );
                    config.initial_rho = Some(rho.clone());
                    config.screen_initial_rho = false;
                    if !beta.is_empty() {
                        cached_inner_seed = Some(BoundInnerSeed {
                            theta: rho,
                            beta: Array1::from_vec(beta),
                        });
                    }
                    had_hit = true;
                }
                CacheSeedDecision::Seed {
                    rho,
                    beta,
                    hessian,
                    prior_obj_display,
                    iteration,
                } => {
                    let beta_len = beta.len();
                    let beta_arr = if beta.is_empty() {
                        None
                    } else {
                        Some(Array1::from_vec(beta))
                    };
                    // Adopt the transferred converged outer Hessian only when it
                    // matches this fit's full-θ dimension; a dimension drift
                    // (structural change the cache key did not capture) falls
                    // back to the scalar warm metric in run_plan.
                    config.warm_start_outer_hessian = if self.hessian.is_analytic() {
                        hessian.and_then(|(dim, flat)| {
                            if dim == self.n_params && flat.len() == dim * dim {
                                Array2::from_shape_vec((dim, dim), flat).ok()
                            } else {
                                None
                            }
                        })
                    } else {
                        None
                    };
                    if config
                        .initial_rho
                        .as_ref()
                        .is_none_or(|initial| initial != rho)
                    {
                        log::info!(
                            "[CACHE] hit  key={}.. context={} rho_dim={} beta_dim={} prior_obj={:.6e} iter={}",
                            short_key,
                            context,
                            rho.len(),
                            beta_len,
                            prior_obj_display,
                            iteration,
                        );
                        config.initial_rho = Some(rho.clone());
                        config.screen_initial_rho = false;
                        had_hit = true;
                    } else {
                        log::info!(
                            "[CACHE] hit  key={}.. context={} rho_dim={} beta_dim={} already-aligned prior_obj={:.6e}",
                            short_key,
                            context,
                            rho.len(),
                            beta_len,
                            prior_obj_display,
                        );
                        had_hit = true;
                    }
                    if let Some(beta) = beta_arr {
                        cached_inner_seed = Some(BoundInnerSeed { theta: rho, beta });
                    }
                }
                CacheSeedDecision::Discard {
                    reason: "payload-shape-mismatch",
                    ..
                } => {
                    log::info!(
                        "[CACHE] skip key={}.. context={} reason=payload-shape-mismatch n_params={}",
                        short_key,
                        context,
                        self.n_params,
                    );
                }
                CacheSeedDecision::Discard {
                    reason,
                    prior_obj_display,
                    all_rho_finite,
                } => {
                    log::info!(
                        "[CACHE] skip key={}.. context={} reason={} prior_obj={:.6e} all_rho_finite={}",
                        short_key,
                        context,
                        reason,
                        prior_obj_display,
                        all_rho_finite.unwrap_or(false),
                    );
                }
            }
        } else {
            log::info!(
                "[CACHE] miss key={}.. context={} reason=fresh-fingerprint n_params={}",
                short_key,
                context,
                self.n_params,
            );
        }
        // Preserve the ownership relation between a cached coefficient vector
        // and the exact outer coordinate that produced it. The runner installs
        // this seed only after resetting for that bitwise-matching candidate;
        // it is never replayed at another generated seed.
        config.initial_inner_seed = cached_inner_seed;
        let mut checkpointing = CheckpointingObjective::new(
            obj,
            Arc::clone(&session),
            config.cache_mirror_sessions.clone(),
        );
        let result = run_outer(&mut checkpointing, &config, context);
        // Attach β only when a beta-bearing evaluation surfaced it at this
        // exact final ρ. Scalar terminal audits carry no β and may follow an
        // evaluation elsewhere; a bare "last β" would manufacture a false
        // (ρ, β) pair and let machine history select the next fit's basin
        // (#2486). A rho-only payload is slower to resume but remains honest.
        let final_beta = result
            .as_ref()
            .ok()
            .and_then(|result| checkpointing.inner_beta_for(&result.rho));
        if let Ok(result) = result.as_ref()
            && result.final_value.is_finite()
            && result.converged()
            && result
                .criterion_certificate
                .as_ref()
                .is_some_and(OuterCriterionCertificate::certifies)
            && let Some(bytes) = encode_iterate(
                &result.rho,
                final_beta.as_ref(),
                result.final_hessian.as_ref(),
                result.final_value,
                result.iterations as u64,
            )
        {
            let saved = session.finalize(
                &bytes,
                Some(result.final_value),
                Some(result.iterations as u64),
            );
            if saved {
                log::info!(
                    "[CACHE] save key={}.. context={} final_obj={:.6e} iter={} resumed={}",
                    short_key,
                    context,
                    result.final_value,
                    result.iterations,
                    had_hit,
                );
            }
            // Broadcast finalize to mirror keys. The seed-prefix mirror
            // exists so future fits with related-but-not-identical
            // structure can warm-start from this run via the dispatcher's
            // prefix lookup.
            for mirror in &config.cache_mirror_sessions {
                let mirror_saved = mirror.finalize(
                    &bytes,
                    Some(result.final_value),
                    Some(result.iterations as u64),
                );
                if mirror_saved {
                    let mirror_hex = mirror.key().to_hex();
                    log::info!(
                        "[CACHE] save key={}.. context={} mirror final_obj={:.6e} iter={}",
                        &mirror_hex[..8.min(mirror_hex.len())],
                        context,
                        result.final_value,
                        result.iterations,
                    );
                }
            }
        }
        result
    }

    /// Run the outer optimization and return an unforgeable certified-result
    /// carrier.  Callers that only need checkpoints or diagnostics should use
    /// [`Self::run`]; fit assembly after an optimized outer coordinate must use
    /// this boundary so a caller-constructed [`OuterResult`] cannot mint
    /// convergence provenance.
    pub fn run_certified(
        &self,
        obj: &mut dyn OuterObjective,
        context: &str,
    ) -> Result<CertifiedOuterResult, EstimationError> {
        let result = self.run(obj, context)?;
        CertifiedOuterResult::from_optimizer_result(result).map_err(|reason| {
            EstimationError::RemlOptimizationFailed(format!(
                "{context}: outer result failed certified-fit validation: {reason}"
            ))
        })
    }
}

/// Internal outcome of one planned solver/multistart attempt.
///
/// Exhausted checkpoints carry resumable work only. They never pass through
/// finalization, cache promotion, uncertainty diagnostics, or fitted-model
/// construction.
pub(crate) enum PlanRunOutcome {
    Converged(OuterResult),
    Exhausted(OuterResult),
}

/// Which certificate concluded a CONVERGED outer run (#2235/#2241).
///
/// `OuterResult.converged == true` bundles genuinely different endings, each
/// with its own certificate. Distinguishing them is pure evidence for the
/// caller's termination report — every variant is a converged fit. There is
/// deliberately no "budget/freeze" variant: exhaustion is a typed error
/// carrying the resume checkpoint, never a minted fit (SPEC 20; the #2235
/// forcing-function redesign deleted the freeze lanes).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum OuterConvergedVia {
    /// The bound-projected analytic gradient at the returned point cleared the
    /// solver's absolute/score-scaled stationarity tolerance.
    GradientStationary,
    /// Criterion-flat certificate (#2241/#2253): the criterion stalled over the
    /// cost-stall window and the residual projected gradient sits inside the
    /// flat certificate band — the score-relative stationarity bound
    /// (`flat_valley_converged_grad_bound`), the probe-noise-floor bound
    /// measured from the stall window's own value scatter, and/or the
    /// curvature-scaled Newton-decrement bound (`newton_predicted_decrease`),
    /// under which a residual above the gradient-magnitude bands is still
    /// stationary when the second-order-predicted improvement `½·gᵀH⁻¹g` is below
    /// the outer objective tolerance. `certificate_bound` is the operative
    /// (widened) bound the residual actually cleared.
    CriterionFlat {
        residual_grad_norm: f64,
        certificate_bound: f64,
    },
    /// Every optimized coordinate carried an explicit analytic fixed-point
    /// equation and the KKT-projected residual cleared the solver tolerance.
    FixedPointStationary {
        projected_residual_inf_norm: f64,
        certificate_bound: f64,
    },
    /// Fellner–Schall model-state fixed point (#2235 verdict 2): two
    /// consecutive outer evaluations restored the same banked incumbent, so a
    /// further outer update provably does not change the fitted state. The
    /// analytic first-order certificate is still taken at the incumbent.
    RecurrentIncumbent { consecutive_restores: usize },
    /// Stationary-at-asymptote (#2348 Inc 1 / #2299 layer 3): the interior
    /// (non-railed) coordinates are gradient-stationary, and every coordinate
    /// railed at the infinite-/zero-smoothing box bound is certified on a
    /// confirmed exponential tail (Thm 2.1) whose fitted model has reached the
    /// rail limit to within the estimand tolerance. The typed rail supersedes
    /// the generic gradient/criterion-flat verdict for a railed optimum.
    AsymptoteStationary { rails: usize },
}

impl OuterConvergedVia {
    /// Stable wire name for termination reports; the enum owns the vocabulary
    /// so bindings marshal instead of mapping.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::GradientStationary => "converged_stationary",
            Self::CriterionFlat { .. } => "converged_criterion_flat",
            Self::FixedPointStationary { .. } => "converged_fixed_point",
            Self::RecurrentIncumbent { .. } => "incumbent_stationary",
            Self::AsymptoteStationary { .. } => "converged_asymptote_rail",
        }
    }
}

/// Typed lifecycle of an outer optimization result.
///
/// A solver claim and an analytic certificate are different stages, but they
/// belong to one state machine. Encoding them in one enum makes it impossible
/// to retain a certified success verdict after a later certificate refusal.
#[derive(Clone, Copy, Debug, PartialEq)]
enum OuterTermination {
    /// The solver exhausted or refused without claiming convergence.
    Exhausted,
    /// The solver claimed convergence, but no terminal analytic certificate
    /// currently authorizes the point. `proposed_via` is reserved for the
    /// recurrent-incumbent fixed-point signal that certification must corroborate.
    SolverClaimed {
        proposed_via: Option<OuterConvergedVia>,
    },
    /// A terminal analytic certificate authorizes the point and records why.
    Certified(OuterConvergedVia),
}

impl OuterTermination {
    fn from_solver_claim(claimed: bool) -> Self {
        if claimed {
            Self::SolverClaimed { proposed_via: None }
        } else {
            Self::Exhausted
        }
    }

    fn solver_claimed_convergence(self) -> bool {
        !matches!(self, Self::Exhausted)
    }

    fn is_certified(self) -> bool {
        matches!(self, Self::Certified(_))
    }

    fn certified_via(self) -> Option<OuterConvergedVia> {
        match self {
            Self::Certified(via) => Some(via),
            Self::Exhausted | Self::SolverClaimed { .. } => None,
        }
    }

    fn proposed_via(self) -> Option<OuterConvergedVia> {
        match self {
            Self::SolverClaimed { proposed_via } => proposed_via,
            Self::Exhausted | Self::Certified(_) => None,
        }
    }

    /// Revoke any earlier screening certificate before measuring a new
    /// screening/mint verdict. Only the model-state recurrent-incumbent signal
    /// survives as a proposal; ordinary gradient/rail verdicts must be re-earned.
    fn begin_certification(&mut self) {
        let proposed_via = match *self {
            Self::SolverClaimed {
                proposed_via: Some(via @ OuterConvergedVia::RecurrentIncumbent { .. }),
            }
            | Self::Certified(via @ OuterConvergedVia::RecurrentIncumbent { .. }) => Some(via),
            Self::Exhausted
            | Self::SolverClaimed { .. }
            | Self::Certified(_) => None,
        };
        if !matches!(*self, Self::Exhausted) {
            *self = Self::SolverClaimed { proposed_via };
        }
    }

    fn certify(&mut self, via: OuterConvergedVia) {
        *self = Self::Certified(via);
    }

    /// A refused analytic pass may retain the factual solver claim for resume
    /// policy, but never a proposed or certified success verdict.
    fn refuse_certificate(&mut self) {
        if !matches!(*self, Self::Exhausted) {
            *self = Self::SolverClaimed { proposed_via: None };
        }
    }
}

/// Which lane actually produced an [`OuterResult`].
///
/// `OuterResult::solver_termination` is `None` whenever no `opt` solver
/// produced the result, and its own doc names three ways that happens — a
/// cache short-circuit, a synthesized checkpoint, the per-atom Fellner–Schall
/// lane — without recording WHICH. A refusal then reads
/// `termination=<no opt solver produced this result>` and the reader is left to
/// infer the lane from iteration counts, which is the #2465 shape: the decision
/// is made, and the basis for it is dropped by the emitter that holds it. On
/// the #1575 binomial fixture that absence is the whole question — a result at
/// `|Pg| = 5.991e-1` after 13 of 300 permitted outer iterations, PSD Hessian,
/// nothing railed, is a search that stopped with descent still available, and
/// "which lane stopped it" is the first thing anyone needs.
///
/// [`OuterResult::new`] defaults to [`Self::Solver`]; the gam-side lanes that
/// synthesize a result overwrite it at their construction site.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OuterResultOrigin {
    /// An `opt` solver ran and its solution was translated into this result.
    Solver,
    /// A seed was accepted as its own optimum with zero outer iterations.
    SeedAcceptedWithoutIteration,
    /// ARC exhausted its budget on a last iterate WORSE than the best feasible
    /// iterate it had seen, so the best iterate was substituted (#1371/#1476).
    ArcBestIterateSubstitution,
    /// ARC hit a run of infeasible probes with no synchronized Hessian, so a
    /// checkpoint was rebuilt from the stored best iterate.
    ArcInfeasibleStallCheckpoint,
    /// The BFGS cost-stall guard halted the search and published its best
    /// iterate, which was rebuilt into this result.
    BfgsCostStallExit,
    /// The per-atom Fellner–Schall frontier lane.
    PerAtomFellnerSchall,
    /// The parameter space is empty; there was nothing to optimize.
    EmptyParameterSpace,
    /// A caller-supplied point audited without running any optimizer.
    StationaryPointAudit,
}

/// Result of a completed outer optimization.
#[derive(Clone, Debug)]
pub struct OuterResult {
    /// Optimized log-smoothing parameters.
    pub rho: Array1<f64>,
    /// Final objective value.
    pub final_value: f64,
    /// Total outer iterations across all solver restarts.
    pub iterations: usize,
    /// Final gradient norm, when the solver computed an actual gradient.
    pub final_grad_norm: Option<f64>,
    /// Final gradient when the solver is gradient-based.
    pub final_gradient: Option<Array1<f64>>,
    /// Final Hessian when the solver tracks one.
    pub final_hessian: Option<Array2<f64>>,
    /// Single authoritative termination lifecycle. Private so downstream
    /// callers cannot manufacture convergence without the optimizer transition.
    termination: OuterTermination,
    /// Which plan was actually used (may differ from initial if fallback fired).
    pub plan_used: OuterPlan,
    /// Final trust radius for the internal operator trust-region solver.
    ///
    /// A non-converged operator-ARC attempt may be restarted by the budget
    /// ladder. Restarting only from the last θ but resetting the trust radius
    /// is not a warm start: it replays the same rejected large trial steps.
    /// Carry this globalization state so retries resume from the scale the
    /// previous attempt already learned.
    pub operator_trust_radius: Option<f64>,
    /// Why the internal operator trust-region solver stopped.
    ///
    /// Derived from [`Self::termination`] in
    /// `bridges::solution_into_outer_result`; do not set it independently
    /// or the two can disagree.
    pub operator_stop_reason: Option<OperatorTrustRegionStopReason>,
    /// Which test the underlying `opt` solver stopped on, and the
    /// quantity it was decided against.
    ///
    /// Distinct from [`OuterTermination`], which is gam's *certification*
    /// state machine (did a terminal analytic certificate authorize this
    /// point). This is the solver's own account of why it stopped
    /// searching, and the two answer different questions: a run can stop
    /// on a satisfied gradient test and still fail certification, or
    /// exhaust its budget and be certified by a later re-measurement.
    ///
    /// Carried from the solver rather than reconstructed here. Before
    /// `opt` reported this, `operator_stop_reason` was hand-populated on
    /// the matrix-free branch alone by matching the coarse
    /// `OptimizationStatus`, so every other route left it `None` — and
    /// `None` rendered identically to "there was nothing to say", which
    /// made #2547's "WHY it stopped is unrecorded" unanswerable without a
    /// new probe.
    ///
    /// `None` here means no `opt` solver produced this result at all (a
    /// cache short-circuit, a synthesized checkpoint, the per-atom
    /// Fellner–Schall lane) — an honest absence, not a dropped verdict.
    pub solver_termination: Option<TerminationReason>,
    /// First-order optimality self-audit at the returned point (#934).
    ///
    /// `None` when no analytic gradient was measured at termination
    /// (gradient-free solvers, cache-hit short-circuits, per-atom EFS) or
    /// when an audit probe failed to evaluate. Populated once by
    /// [`run_outer`] after the solver ladder returns, outside all hot loops.
    pub criterion_certificate: Option<OuterCriterionCertificate>,
    /// Probe-noise-floor gradient bound measured by the cost-stall guard at a
    /// halted stall (#2241): σ̂/Δ, the criterion's evaluation-noise floor over
    /// the stall window divided by the radius the accepted steps actually
    /// probed. Present only on results rebuilt from a cost-stall exit;
    /// [`certify_outer_optimality`] folds it into the stationarity bound so the
    /// final re-measured gradient is judged against the same flat certificate
    /// the guard granted.
    /// Why the solver's line search gave up, when it did (#2465).
    ///
    /// `opt` reports `LineSearchFailureReason` plus the attempt count on
    /// `BfgsError::LineSearchFailed`, and the two variants have opposite
    /// causes: `StepSizeTooSmall` means the direction WAS a descent direction
    /// and no usable step decreased the objective — an objective/gradient
    /// inconsistency or evaluation noise — while `MaxAttempts` means the
    /// bracketing never closed, a pathological landscape. The bridge used to
    /// drop both on the floor: a line-search failure whose last iterate is
    /// finite is returned as `Ok(non-converged)`, so the caller never sees the
    /// `Err` that carries them, and the certificate could say only
    /// `termination=line_search_failed(|g|=…)` — the verdict without the
    /// quantity it was decided against.
    ///
    /// `None` means no line search failed (or no `opt` solver produced this
    /// result at all).
    pub line_search_failure: Option<(LineSearchFailureReason, usize)>,
    pub flat_noise_grad_bound: Option<f64>,
    /// Post-fit PSIS diagnostic for whether sampled smoothing-parameter weights
    /// show evidence that plug-in REML/LAML intervals are unreliable. Populated
    /// once by [`run_outer`] when the exact rho Hessian is cheap enough to use.
    pub rho_uncertainty_diagnostic: Option<crate::rho_uncertainty::RhoUncertaintyDiagnostic>,
    /// Reseed point minted by a refused certification whose tail snap CONFIRMED
    /// an exponential tail (#2348 Inc 2b). A snap is a waypoint, never a
    /// candidate optimum: even an interior coordinate stationary before the
    /// snap can move when it is coupled to the tail coordinate (#2358). The
    /// plan runner retries ONCE from this point, allowing every coordinate to
    /// re-descend or remain on the rail before the natural certificate judges
    /// the result.
    pub tail_snap_reseed: Option<Array1<f64>>,
    /// Saddle-escape reseed point minted by a refused certification whose
    /// interior reduced Hessian is a certified strict saddle — small projected
    /// gradient, `hessian_psd = Some(false)`, no railed coordinate (#2357). A
    /// gradient-only convergence gate (ARC's, or the cost-stall guard's) can
    /// ARRIVE at such a saddle with its gradient already below tolerance and
    /// stop, even though the certified negative-curvature eigendirection is a
    /// strict descent direction the optimizer never took. This point is
    /// `ρ + α·v` for the most-negative-curvature eigenvector `v`, stepped off
    /// the saddle ridge to a strictly-lower objective; the plan runner reseeds
    /// the outer search ONCE from it (reseed gate closed so it cannot recurse),
    /// which lets the optimizer descend to the true PSD minimum exactly as an
    /// identical warm-started resume does by hand.
    pub saddle_escape_reseed: Option<Array1<f64>>,
    /// Wrong-rail pull-back reseed point minted by a refused certification whose
    /// coordinate sits AT the ρ box bound but whose clean-band probes prove the
    /// objective DECREASES as the coordinate moves INWARD (#2392). The outer
    /// search drove the coordinate to the wrong bound — its terminal gradient is
    /// deep-λ instrument noise, so the trust region never proposed the large
    /// inward move — while a drift-band-clean, above-noise-floor run of probes a
    /// few e-folds inside carries a pencil constant of the sign OPPOSITE the rail
    /// (descent points away from the bound, `∂V/∂ρ > 0` at an upper rail). This
    /// point moves that coordinate to its clean-band interior scale, where the
    /// gradient is informative again; the plan runner reseeds ONCE (gate closed)
    /// and the optimizer descends to the true interior optimum. Gated strictly on
    /// the opposite-sign clean-tail proof, so a GENUINE rail (descent toward the
    /// bound) never mints it and no real λ→∞ optimum is pulled off its rail.
    pub wrong_rail_reseed: Option<Array1<f64>>,
    /// Active-set reduction reseed minted by a refused certification whose
    /// INTERIOR is not stationary while a coordinate is railed at the ρ box with
    /// a deep-λ noise-floor gradient (#2392). The railed coordinate's
    /// ill-conditioned Hessian row poisons the joint Newton/ARC steps, so the
    /// interior cannot polish; freezing that coordinate at its bound and
    /// re-running lets the optimizer converge the interior in the well-conditioned
    /// REDUCED space. The reseed carries the frozen box (`bounds`, with
    /// `lower[k]==upper[k]==rail` for each frozen coordinate); the plan runner's
    /// re-certification under the ORIGINAL bounds then judges every pinned
    /// coordinate's KKT sign at the reduced optimum (an inward-feasible-descent
    /// gradient unfreezes it — no silent clamping of a coordinate that stops
    /// wanting the rail).
    pub active_set_reseed: Option<ActiveSetReseed>,
    /// `(noise_floor σ̂, probe_radius Δ)` the cost-stall guard measured over its
    /// stall window, when a stall produced this result. Present whether or not
    /// their ratio licensed a `flat_noise_grad_bound`: σ̂ is the per-step
    /// objective change the no-improvement window judged and Δ is the radius the
    /// accepted steps moved, and a window that filled because the search took
    /// microscopic steps is told from one that filled because the surface is
    /// flat by Δ, not by the verdict.
    pub cost_stall_probe_scale: Option<(f64, f64)>,
    /// Which lane produced this result. See [`OuterResultOrigin`].
    pub origin: OuterResultOrigin,
}

/// An active-set reduction reseed (#2392): re-run the outer search with a set of
/// railed coordinates FROZEN at their box bounds so the optimizer polishes the
/// interior in the reduced space.
#[derive(Clone, Debug)]
pub struct ActiveSetReseed {
    /// The reseed point: the refused checkpoint with the frozen coordinates
    /// pinned at their bounds (`rho[k] == bounds.0[k] == bounds.1[k]`).
    pub rho: Array1<f64>,
    /// The reduced-space box: `lower[k] == upper[k] == rail` for every frozen
    /// coordinate `k`, the original bounds elsewhere.
    pub bounds: (Array1<f64>, Array1<f64>),
}

impl OuterResult {
    pub fn new(
        rho: Array1<f64>,
        final_value: f64,
        iterations: usize,
        solver_claimed_convergence: bool,
        plan_used: OuterPlan,
    ) -> Self {
        Self {
            rho,
            final_value,
            iterations,
            final_grad_norm: None,
            final_gradient: None,
            final_hessian: None,
            termination: OuterTermination::from_solver_claim(solver_claimed_convergence),
            plan_used,
            operator_trust_radius: None,
            operator_stop_reason: None,
            solver_termination: None,
            criterion_certificate: None,
            line_search_failure: None,
            flat_noise_grad_bound: None,
            rho_uncertainty_diagnostic: None,
            tail_snap_reseed: None,
            saddle_escape_reseed: None,
            wrong_rail_reseed: None,
            active_set_reseed: None,
            cost_stall_probe_scale: None,
            origin: OuterResultOrigin::Solver,
        }
    }

    /// Whether this result owns a terminal analytic convergence certificate.
    pub fn converged(&self) -> bool {
        self.termination.is_certified()
    }

    /// Which analytic certificate concluded this run.
    pub fn converged_via(&self) -> Option<OuterConvergedVia> {
        self.termination.certified_via()
    }

    /// Whether the underlying solver claimed convergence before analytic
    /// certification. Certified results necessarily originated from a claim or
    /// an explicit stationary-point audit.
    pub(crate) fn solver_claimed_convergence(&self) -> bool {
        self.termination.solver_claimed_convergence()
    }

    /// Human-readable rendering of `final_grad_norm` for diagnostics. Returns
    /// `"n/a"` when no gradient was measured (gradient-free / cache-hit paths).
    pub fn final_grad_norm_report(&self) -> String {
        match self.final_grad_norm {
            Some(g) => format!("{g:.3e}"),
            None => "n/a".to_string(),
        }
    }
}

/// Validated evidence that an outer optimization terminated at a finite,
/// analytically certified optimum.
///
/// The inner [`OuterResult`] is private so downstream fit assembly cannot turn
/// a status boolean into convergence provenance. Construction consumes the
/// optimizer result and revalidates the certificate at the ownership boundary.
#[derive(Clone, Debug)]
pub struct CertifiedOuterResult {
    result: OuterResult,
}

impl CertifiedOuterResult {
    /// The sole constructor is reached from [`OuterProblem::run_certified`]
    /// after the optimizer has produced the result.  Keeping this private is
    /// load-bearing: `OuterResult` is also a public diagnostic/checkpoint
    /// payload, so a public conversion would let downstream code fabricate a
    /// certificate-shaped result without ever running an objective.
    fn from_optimizer_result(result: OuterResult) -> Result<Self, String> {
        if !result.converged() {
            return Err(format!(
                "outer optimization did not converge after {} iterations",
                result.iterations
            ));
        }
        if !result.final_value.is_finite() {
            return Err(format!(
                "outer optimization returned a non-finite objective: {}",
                result.final_value
            ));
        }
        if result.rho.iter().any(|value| !value.is_finite()) {
            return Err("outer optimization returned non-finite hyperparameters".to_string());
        }
        if result
            .final_grad_norm
            .is_some_and(|value| !value.is_finite() || value < 0.0)
        {
            return Err(format!(
                "outer optimization returned an invalid gradient norm: {:?}",
                result.final_grad_norm
            ));
        }
        let certificate = result
            .criterion_certificate
            .as_ref()
            .ok_or_else(|| "outer optimization returned no analytic certificate".to_string())?;
        if !certificate.certifies() {
            return Err(format!(
                "outer optimization certificate does not certify: {}",
                certificate.summary()
            ));
        }
        Ok(Self { result })
    }

    /// Exact optimizer-owned hyperparameter vector covered by the certificate.
    pub fn rho(&self) -> &Array1<f64> {
        &self.result.rho
    }

    pub fn iterations(&self) -> usize {
        self.result.iterations
    }

    pub fn final_value(&self) -> f64 {
        self.result.final_value
    }

    pub fn final_grad_norm(&self) -> Option<f64> {
        self.result.final_grad_norm
    }

    /// Exact analytic gradient re-measured by the optimizer-owned terminal
    /// certificate. Downstream selected-profile finalizers use this to prove
    /// that a retained objective payload is the one certified at `rho()`.
    pub fn final_gradient(&self) -> Option<&Array1<f64>> {
        self.result.final_gradient.as_ref()
    }

    pub fn criterion_certificate(&self) -> &OuterCriterionCertificate {
        self.result
            .criterion_certificate
            .as_ref()
            .expect("CertifiedOuterResult always owns a validated certificate")
    }

    /// The analytic outer ρ-Hessian measured at the certified point, when the
    /// certification retained one. This is the curvature evidence behind the
    /// certificate's PSD verdict — and the `V_ρ = H_ρ⁻¹` input to first-order
    /// smoothing-correction inflation (#2346).
    pub fn final_hessian(&self) -> Option<&Array2<f64>> {
        self.result.final_hessian.as_ref()
    }
}

#[cfg(test)]
mod certified_outer_result_tests {
    use super::*;

    #[test]
    fn caller_boolean_and_zero_gradient_cannot_mint_outer_authority() {
        let mut fabricated = OuterResult::new(
            Array1::from_vec(vec![0.0]),
            1.0,
            3,
            true,
            OuterPlan {
                solver: Solver::Bfgs,
                hessian_source: HessianSource::BfgsApprox,
            },
        );
        fabricated.final_grad_norm = Some(0.0);
        fabricated.final_gradient = Some(Array1::from_vec(vec![0.0]));
        fabricated.termination = OuterTermination::Certified(OuterConvergedVia::GradientStationary);

        let reason = CertifiedOuterResult::from_optimizer_result(fabricated)
            .expect_err("caller-written status and gradient must not mint a certificate");
        assert!(reason.contains("no analytic certificate"), "{reason}");
    }
}

/// Typed refusal from [`audit_stationary_point`]. The rejected point and every
/// analytic certificate field measured before refusal remain available to the
/// caller; `source` records why those measurements did not certify.
#[derive(Debug)]
pub struct OuterStationaryPointRejection {
    pub result: OuterResult,
    pub source: EstimationError,
}

impl std::fmt::Display for OuterStationaryPointRejection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.source, f)
    }
}

impl std::error::Error for OuterStationaryPointRejection {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.source)
    }
}

/// Apply the shared analytic outer-optimality authority to one caller-supplied
/// point without running an optimizer or taking a step.
///
/// The objective controls whether evaluating that point mutates its profiled
/// state. Callers auditing an already-installed inner state must put their
/// objective in a frozen evaluation mode before calling this function.
/// `iterations == 0` in the returned result is structural: no optimization loop
/// exists on this path.
pub fn audit_stationary_point(
    obj: &mut dyn OuterObjective,
    rho: Array1<f64>,
    context: &str,
) -> Result<OuterResult, OuterStationaryPointRejection> {
    let config = OuterConfig::default();
    let selected_plan = plan(&obj.capability());
    // There is intentionally no independent value-only probe. The analytic
    // sample is the authority being audited, and infinity records that no
    // optimizer-produced terminal value exists to compare against it.
    let mut result = OuterResult::new(rho, f64::INFINITY, 0, false, selected_plan);
    result.origin = OuterResultOrigin::StationaryPointAudit;
    match certify_outer_optimality(obj, &config, context, &mut result) {
        Ok(certificate) => {
            result.criterion_certificate = Some(certificate);
            Ok(result)
        }
        Err(source) => Err(OuterStationaryPointRejection { result, source }),
    }
}

// ─── First-order optimality certificate (#934) ────────────────────────
//
// The objective↔gradient desync bug genus (#748, #752, #808, #901, …) has a
// universal signature: at the returned "optimum" the optimizer claims
// convergence while the criterion is not actually stationary there (or the
// optimizer stalls and rails λ). The certificate makes the engine check
// itself, once, at θ̂, on every generic outer fit — purely from the ANALYTIC
// objective, per SPEC rule 2 (finite differences never run outside tests;
// the FD gradient oracle now lives in the test-only `fd_audit` module): the
// KKT-projected analytic gradient norm against the same score-relative
// stationarity bound the outer loop already uses to accept flat-valley
// stalls (#1690), a scaled PSD probe of the tracked outer Hessian, and the
// λ-rail facts every desync postmortem asks for. It is the runtime
// enforcement layer for the criterion-atom architecture (#931).
//
// A failed certificate REJECTS the fit as typed non-convergence — never a
// warn-and-continue diagnostic — so a nonstationary point can never be
// minted into a fit (SPEC rule 20).

/// Cholesky positive-SEMIdefiniteness probe for the (small, outer-dim) final
/// Hessian, with a roundoff-scale diagonal shift. Returns `None` when the
/// matrix is empty, non-square, or non-finite; `Some(false)` when the shifted
/// matrix has a non-positive pivot — i.e. the curvature is genuinely
/// indefinite, not merely semidefinite-within-noise.
///
/// The shift is `√ε · max(1, max|H_ii|)`: eigenvalues assembled through
/// O(‖H‖)-scaled arithmetic carry O(ε·‖H‖) roundoff, so a `√ε`-relative
/// margin cleanly separates a true negative direction from accumulated
/// floating-point noise on a flat (near-semidefinite) valley.
pub(crate) fn certificate_hessian_is_psd(hessian: &Array2<f64>) -> Option<bool> {
    let n = hessian.nrows();
    if n == 0 || hessian.ncols() != n || hessian.iter().any(|v| !v.is_finite()) {
        return None;
    }
    let max_diag = (0..n).fold(0.0_f64, |acc, j| acc.max(hessian[[j, j]].abs()));
    let shift = f64::EPSILON.sqrt() * max_diag.max(1.0);
    let mut chol = hessian.clone();
    for j in 0..n {
        chol[[j, j]] += shift;
    }
    for j in 0..n {
        for k in 0..j {
            let l_jk = chol[[j, k]];
            for i in j..n {
                chol[[i, j]] -= chol[[i, k]] * l_jk;
            }
        }
        let pivot = chol[[j, j]];
        if !(pivot > 0.0) || !pivot.is_finite() {
            return Some(false);
        }
        let inv_sqrt = 1.0 / pivot.sqrt();
        for i in j..n {
            chol[[i, j]] *= inv_sqrt;
        }
    }
    Some(true)
}

/// PSD verdict of the outer Hessian restricted to its UN-RAILED coordinates
/// (#2299 box-KKT reduced-Hessian / critical-cone gate).
///
/// A coordinate railed at ±`rho_bound` with an outward gradient is at the box-KKT
/// constrained optimum: its curvature direction is the flat/indefinite
/// infinite-smoothing plateau of a fully-saturated penalty (λ ~ 1e13), carrying
/// no feasible descent. Including it makes the FULL Hessian indefinite and used to
/// disable the very flatness certificate that exists to handle rails, so an
/// honest railed optimum ground to `max_iter` and refused. Judging PSD on the
/// INTERIOR (un-railed) sub-block is the standard reduced-Hessian condition: a
/// genuinely indefinite *interior* direction still keeps the sub-block non-PSD,
/// so this can never over-certify a real saddle. When every coordinate is railed
/// the interior is empty — there is no feasible curvature to certify and the rail
/// KKT signs are the whole certificate — so the empty sub-block is trivially PSD.
/// With no railed coordinate it is exactly [`certificate_hessian_is_psd`].
pub(crate) fn certificate_hessian_is_psd_off_railed(
    hessian: &Array2<f64>,
    railed: &[usize],
) -> Option<bool> {
    if railed.is_empty() {
        return certificate_hessian_is_psd(hessian);
    }
    let n = hessian.nrows();
    let railed_set: std::collections::BTreeSet<usize> = railed.iter().copied().collect();
    let interior: Vec<usize> = (0..n).filter(|k| !railed_set.contains(k)).collect();
    if interior.is_empty() {
        return Some(true);
    }
    let mut sub = Array2::<f64>::zeros((interior.len(), interior.len()));
    for (i, &ri) in interior.iter().enumerate() {
        for (j, &rj) in interior.iter().enumerate() {
            sub[[i, j]] = hessian[[ri, rj]];
        }
    }
    certificate_hessian_is_psd(&sub)
}

/// Interior-PSD verdict judged ABOVE the per-coordinate gradient-residue noise
/// floor (#2349): PSD of `H + diag(|g|)` restricted to the un-excluded
/// coordinates.
///
/// The assembled ρ-Hessian's tail entries carry the #2298 trace-pair
/// cancellation residue: when the `λ²V_λλ` pair cancels to roundoff, the
/// surviving diagonal entry is `λV_λ = g_k` — gradient magnitude, corrupted
/// sign (the same tie signature the tail-snap candidate band keys on).
/// Measured on the #2349 multinomial checkpoint: the sole interior coordinate
/// had `g₁ = −1.0228e-3`, `H₁₁ = −1.0216e-3` (ratio 0.999), and that single
/// sub-resolution entry was the entire `interior Hessian sub-block not PSD`
/// refusal — the full 6×6 spectrum was `[−1.02e-3, 0.135, …, 0.904]`.
///
/// # Why the floor is safe — instrument resolution, not step economics
///
/// The residue is `O(|g_k|)`, and the sub-block is extracted AFTER flooring, so
/// only the *judged* (un-excluded) coordinates' gradients ever enter: a railed
/// coordinate's large `|g|` (measured 1.40 on the #2349 checkpoint) cannot
/// inflate the floor. Weyl bounds what remains exactly:
///
/// ```text
///     λ_min(H) + min|g| ≤ λ_min(H + diag|g|) ≤ λ_min(H) + max|g|
/// ```
///
/// so the floor absorbs **at most `max_k |g_k|` over the judged coordinates**.
/// Where this verdict can mint, those coordinates have passed gradient
/// stationarity, so that is at most the stationarity bound. A negative
/// eigenvalue smaller than that is not distinguishable from zero *by the
/// instrument that produced it* — the assembled tail entry IS `λV_λ = g`
/// (ratio 0.999 as measured) — while a genuine saddle survives untouched: the
/// #2357 trace's `λ_min ≈ −0.5` against `|g| ≈ 1e-3` floors to `−0.4973`, and a
/// `−1.50e-2` direction still refuses at a `1e-2` bound.
///
/// ## Do not widen this floor beyond `diag(|g|)`
///
/// Earlier revisions of this argument (and #2349 rounds 6/7) justified the
/// floor by an "exploitable improvement `≲ g²/2|H| ≈ 5e-7`". Both halves are
/// wrong and the pair is misleading in the permissive direction:
///
/// * the quoted number is `g²/2 = 5.23e-7`; the quoted formula evaluates to
///   `g²/(2|H|) = 5.12e-4` on the same measured `g = 1.0228e-3`,
///   `|H| = 1.0216e-3` — a 979× discrepancy. Since `|H| ≈ g` at exactly the
///   point this floor serves, that formula collapses to `≈ g/2`;
/// * and `g²/2|H|` is the Newton decrement, which bounds the improvement along
///   a direction of POSITIVE curvature. Along curvature `−ε` the model
///   `−g·t − ½εt²` is unbounded below, so the quantity it purports to bound
///   does not exist in the regime this floor exists for; the step is limited by
///   the trust radius, not the curvature.
///
/// The Weyl bound above is the correct and checkable statement. Anything that
/// widens the floor must re-derive against it, not against `g²/2|H|`.
pub(crate) fn certificate_hessian_is_psd_off_railed_above_gradient_floor(
    hessian: &Array2<f64>,
    excluded: &[usize],
    gradient: &Array1<f64>,
) -> Option<bool> {
    let n = hessian.nrows();
    if gradient.len() != n {
        return certificate_hessian_is_psd_off_railed(hessian, excluded);
    }
    let mut floored = hessian.clone();
    for k in 0..n {
        floored[[k, k]] += gradient[k].abs();
    }
    certificate_hessian_is_psd_off_railed(&floored, excluded)
}

/// Measure the gradient-residue floor's clearance on the interior sub-block:
/// the sub-block's smallest eigenvalue as assembled, the floor it is judged
/// against (`max_k |g_k|` over exactly those coordinates), and whether
/// `H + diag(|g|)` is PSD there.
///
/// This RECORDS the verdict; it does not replace the raw measurement. See
/// [`certificate_hessian_is_psd_off_railed_above_gradient_floor`] for why the
/// Weyl bound makes the floor safe where it can mint, and
/// [`crate::model_types::CurvatureFloorClearance`] for why the two facts are
/// kept apart.
pub(crate) fn interior_curvature_floor_clearance(
    hessian: &Array2<f64>,
    excluded: &[usize],
    gradient: &Array1<f64>,
) -> Option<CurvatureFloorClearance> {
    use faer::Side;
    use gam_linalg::faer_ndarray::FaerEigh;

    let n = hessian.nrows();
    if n == 0 || hessian.ncols() != n || gradient.len() != n {
        return None;
    }
    let excluded_set: std::collections::BTreeSet<usize> = excluded.iter().copied().collect();
    let interior: Vec<usize> = (0..n).filter(|k| !excluded_set.contains(k)).collect();
    if interior.is_empty() {
        return None;
    }
    let m = interior.len();
    let mut sub = Array2::<f64>::zeros((m, m));
    for (i, &ri) in interior.iter().enumerate() {
        for (j, &rj) in interior.iter().enumerate() {
            sub[[i, j]] = 0.5 * (hessian[[ri, rj]] + hessian[[rj, ri]]);
        }
    }
    if sub.iter().any(|v| !v.is_finite()) {
        return None;
    }
    let interior_min_eigenvalue = sub
        .eigh(Side::Lower)
        .ok()?
        .0
        .iter()
        .fold(f64::INFINITY, |acc, v| acc.min(*v));
    if !interior_min_eigenvalue.is_finite() {
        return None;
    }
    // The floor is the largest gradient among EXACTLY the judged coordinates —
    // the excluded ones never enter, so a railed coordinate's large |g| cannot
    // inflate it.
    let gradient_floor = interior
        .iter()
        .fold(0.0_f64, |acc, &k| acc.max(gradient[k].abs()));
    let cleared = certificate_hessian_is_psd_off_railed_above_gradient_floor(
        hessian, excluded, gradient,
    ) == Some(true);
    Some(CurvatureFloorClearance {
        interior_min_eigenvalue,
        gradient_floor,
        cleared,
    })
}

/// Escape point off a certified strict saddle in the free (un-railed) subspace
/// (#2357, generalised to the box-constrained case in #2155).
///
/// A gradient-only outer convergence gate — ARC's own, or the cost-stall guard's
/// — can ARRIVE at a point that is first-order stationary (`‖Pg‖ ≤ bound`) yet
/// sits on genuinely indefinite curvature in its INTERIOR (un-railed) directions,
/// and stop there because its gradient already cleared tolerance. The mandatory
/// analytic certificate then refuses the point as `INDEFINITE CURVATURE AT
/// INTERIOR OPTIMUM` — a verdict `certificate_hessian_is_psd_off_railed` reaches
/// on the reduced Hessian restricted to the un-railed coordinates, so it fires
/// whether or not some other coordinate happens to be railed. Such a point is a
/// saddle, not a minimum: the most-negative-curvature eigenvector `v` of that
/// reduced Hessian is a strict, box-feasible descent direction the optimizer
/// never took. An
/// identical warm-started resume escapes it trivially (its fresh cubic step moves
/// off the ridge, which is why the resume converges where the cold run refuses);
/// this reproduces that escape deterministically by stepping `ρ ± α·v` to a
/// strictly-lower objective and handing the point back as a one-shot reseed.
///
/// Termination is guaranteed: along a direction of negative curvature
/// `vᵀHv = λ_min < 0` at a near-stationary gradient,
/// `f(ρ ± αv) = f(ρ) ± α(g·v) + ½α²λ_min + o(α²)` strictly decreases for small
/// enough `α` once the sign is chosen so the first-order term is non-positive, so
/// the finite backtracking below always finds a descending feasible point when
/// one exists inside the box.
///
/// Returns `None` (no reseed; the ordinary refusal proceeds) when the Hessian
/// carries no eigen-resolvable negative direction, or no bounded step along it
/// clears the box projection with a strict objective decrease. Restores the
/// objective's profiled inner state to `rho` before returning either way, so the
/// refusal path that follows measures the checkpoint rather than the last probe.
fn negative_curvature_escape_point(
    obj: &mut dyn OuterObjective,
    rho: &Array1<f64>,
    gradient: &Array1<f64>,
    hessian: &Array2<f64>,
    railed: &[usize],
    baseline_cost: f64,
    bounds: &(Array1<f64>, Array1<f64>),
    context: &str,
) -> Option<Array1<f64>> {
    use faer::Side;
    use gam_linalg::faer_ndarray::FaerEigh;

    let n = hessian.nrows();
    if n == 0 || hessian.ncols() != n || hessian.iter().any(|v| !v.is_finite()) {
        return None;
    }
    // The escape direction lives in the INTERIOR (un-railed) subspace — the exact
    // reduced Hessian / critical cone that `certificate_hessian_is_psd_off_railed`
    // judges for the PSD verdict. A coordinate railed at a box bound with an
    // outward KKT gradient is already at its constrained optimum; its curvature is
    // the flat/indefinite infinite-smoothing plateau (λ ~ 1e13) and carries no
    // feasible descent. Including it would let the step chase that spurious
    // direction and simply re-rail. Restricting to the un-railed block yields a
    // feasible descent that holds every rail fixed, so the escape generalises from
    // the fully-interior saddle to a box-constrained one whose free-direction
    // reduced Hessian is indefinite (#2357 → #2155). With no rail this is exactly
    // the full-Hessian eigenproblem as before.
    let railed_set: std::collections::BTreeSet<usize> = railed.iter().copied().collect();
    let interior: Vec<usize> = (0..n).filter(|k| !railed_set.contains(k)).collect();
    if interior.is_empty() {
        // Every coordinate is railed: there is no feasible interior direction and
        // the rail KKT signs are the whole certificate.
        return None;
    }
    let m = interior.len();
    let mut sub = Array2::<f64>::zeros((m, m));
    for (i, &ri) in interior.iter().enumerate() {
        for (j, &rj) in interior.iter().enumerate() {
            sub[[i, j]] = hessian[[ri, rj]];
        }
    }
    let (eigenvalues, eigenvectors) = match sub.eigh(Side::Lower) {
        Ok(pair) => pair,
        Err(err) => {
            log::warn!(
                "[CERTIFICATE] {context}: saddle-escape eigendecomposition failed ({err}); \
                 refusing at the checkpoint without a reseed"
            );
            return None;
        }
    };
    // The SAME √ε·‖H‖ margin `certificate_hessian_is_psd` uses to separate a
    // genuine negative eigenvalue from O(ε·‖H‖) assembly roundoff: only a truly
    // negative direction — not a flat / near-semidefinite one — carries a descent
    // the reseed can exploit. Measured on the interior sub-block's diagonal so the
    // threshold matches the reduced PSD verdict exactly.
    let max_diag = interior
        .iter()
        .fold(0.0_f64, |acc, &j| acc.max(hessian[[j, j]].abs()));
    let neg_margin = f64::EPSILON.sqrt() * max_diag.max(1.0);
    let mut min_idx = 0usize;
    for k in 1..eigenvalues.len() {
        if eigenvalues[k] < eigenvalues[min_idx] {
            min_idx = k;
        }
    }
    if !(eigenvalues[min_idx] < -neg_margin) {
        return None;
    }
    let v_sub = eigenvectors.column(min_idx);
    let dir_norm = v_sub.dot(&v_sub).sqrt();
    if !(dir_norm > 0.0) || !dir_norm.is_finite() {
        return None;
    }
    // Lift the interior eigenvector into the full ρ space, exactly zero on every
    // railed coordinate so the backtracking step below holds all rails fixed.
    let mut direction = Array1::<f64>::zeros(n);
    for (i, &ri) in interior.iter().enumerate() {
        direction[ri] = v_sub[i] / dir_norm;
    }
    // First-order-consistent sign: move against the (tiny) gradient's projection
    // onto `v` so the linear term never opposes the curvature descent. With a
    // stationary gradient the tie is arbitrary; the opposite sign is tried below
    // regardless, which also covers a `v` that projects straight out of the box.
    let primary_sign = if gradient.dot(&direction) > 0.0 {
        -1.0
    } else {
        1.0
    };
    // One e-fold in log-λ is a macroscopic step across the saddle ridge; ARC then
    // refines from wherever this lands, so the reseed only needs to leave the
    // ridge, not solve the problem. Backtrack so the box-projected point still
    // strictly descends.
    const ESCAPE_STEP_SCALES: [f64; 5] = [1.0, 0.5, 0.25, 0.125, 0.0625];
    // A strict-decrease floor at the objective's roundoff resolution: a reseed
    // that only matches the checkpoint to roundoff is not a real escape.
    let strict_floor = baseline_cost.abs().max(1.0) * (16.0 * f64::EPSILON);
    let mut best: Option<(f64, Array1<f64>)> = None;
    for sign in [primary_sign, -primary_sign] {
        for &alpha in ESCAPE_STEP_SCALES.iter() {
            let mut trial = rho.clone();
            for i in 0..n {
                trial[i] += sign * alpha * direction[i];
            }
            let trial = project_to_bounds(&trial, Some(bounds));
            // A fully box-clamped trial that lands back on ρ probes nothing.
            if outer_theta_bitwise_eq(&trial, rho) {
                continue;
            }
            if let Ok(cost) = obj.eval_cost(&trial)
                && cost.is_finite()
                && cost < baseline_cost - strict_floor
                && best.as_ref().is_none_or(|(c, _)| cost < *c)
            {
                best = Some((cost, trial));
            }
        }
        if best.is_some() {
            break;
        }
    }
    // Restore the profiled inner state to the checkpoint ρ so the refusal path
    // that follows measures the checkpoint, not the last probe.
    if let Err(err) = obj.eval_cost(rho) {
        log::warn!(
            "[CERTIFICATE] {context}: failed to restore the objective to the checkpoint \
             after saddle-escape probing: {err}"
        );
    }
    best.map(|(cost, point)| {
        log::info!(
            "[CERTIFICATE] {context}: interior strict saddle (λ_min={:.3e} < 0, |Pg| within \
             band); minting a negative-curvature escape reseed (objective {:.6e} → {:.6e}) for \
             one retry (#2357)",
            eigenvalues[min_idx],
            baseline_cost,
            cost,
        );
        point
    })
}

/// Second-order predicted objective decrease of a safeguarded Newton step at a
/// flat-valley cost-stall exit (#2253/#2249/#2015).
///
/// On a flat-valley cost-stall exit the outer criterion has provably stopped
/// improving (the cost-stall window fired), yet the re-measured projected
/// gradient can sit modestly above the score-relative flat band on a
/// weakly-identified small-n fit (measured: |Pg| ≈ 0.072 vs a score-relative
/// band ≈ 0.053 on an n=84/p=64 K=1 circle). Whether that residual is genuine
/// available descent is a SECOND-ORDER question. The improvement a safeguarded
/// Newton step buys is the Newton decrement over two:
///
///     Δpred = ½ · gᵀ H⁻¹ g,
///
/// the textbook Newton stopping quantity (Boyd–Vandenberghe §9.5). When `Δpred`
/// is below the outer objective tolerance, no step can reduce the criterion by
/// more than that tolerance and the point is stationary at the resolution the
/// criterion can be optimized — the mathematically correct "no further descent
/// possible" criterion.
///
/// This is curvature-scaled, not a constant: because `H⁻¹` weights each gradient
/// component by the inverse eigenvalue, a residual aligned with a NEAR-FLAT
/// Hessian eigenvector (a linear ramp that DOES carry real descent) inflates
/// `gᵀ H⁻¹ g` toward the roundoff-regularized `|g_flat|² / shift` and is
/// REJECTED; only a residual that is small along the well-curved directions and
/// nearly orthogonal to the flat ones certifies. An indefinite Hessian never
/// reaches here — the certificate's curvature gate (`certificate_hessian_is_psd`)
/// rejects a genuinely indefinite point independently, and this factorization
/// returns `None` on a non-PSD shifted factor so the caller falls back to the
/// gradient-only bound.
///
/// `hessian` and `grad` are the analytic outer Hessian and the KKT-PROJECTED
/// gradient at the certified point. The shift `√ε · max|H_jj|` matches
/// [`certificate_hessian_is_psd`] so the definiteness verdict and this decrement
/// agree on the same regularized operator. Returns `None` when the shapes are
/// malformed, an entry is non-finite, the shifted factor is not PD, or the
/// resulting quadratic form is negative (which a PD factor rules out; retained
/// as a roundoff guard).
pub(crate) fn newton_predicted_decrease(hessian: &Array2<f64>, grad: &Array1<f64>) -> Option<f64> {
    let n = hessian.nrows();
    if n == 0 || hessian.ncols() != n || grad.len() != n {
        return None;
    }
    if hessian.iter().any(|v| !v.is_finite()) || grad.iter().any(|v| !v.is_finite()) {
        return None;
    }
    let max_diag = (0..n).fold(0.0_f64, |acc, j| acc.max(hessian[[j, j]].abs()));
    let shift = f64::EPSILON.sqrt() * max_diag.max(1.0);
    // Lower Cholesky factor L of H + shift·I (same regularization the PSD probe
    // uses), computed in place.
    let mut l = hessian.clone();
    for j in 0..n {
        l[[j, j]] += shift;
    }
    for j in 0..n {
        for k in 0..j {
            let l_jk = l[[j, k]];
            for i in j..n {
                l[[i, j]] -= l[[i, k]] * l_jk;
            }
        }
        let pivot = l[[j, j]];
        if !(pivot > 0.0) || !pivot.is_finite() {
            return None;
        }
        let inv_sqrt = 1.0 / pivot.sqrt();
        for i in j..n {
            l[[i, j]] *= inv_sqrt;
        }
    }
    // Solve (L Lᵀ) d = g for d = H_s⁻¹ g: forward-substitute L y = g, then
    // back-substitute Lᵀ d = y.
    let mut y = grad.clone();
    for j in 0..n {
        let mut s = y[j];
        for k in 0..j {
            s -= l[[j, k]] * y[k];
        }
        y[j] = s / l[[j, j]];
    }
    let mut d = y;
    for j in (0..n).rev() {
        let mut s = d[j];
        for k in (j + 1)..n {
            s -= l[[k, j]] * d[k];
        }
        d[j] = s / l[[j, j]];
    }
    let quad = grad.dot(&d); // gᵀ H_s⁻¹ g ≥ 0 for a PD factor.
    if !quad.is_finite() || quad < 0.0 {
        return None;
    }
    Some(0.5 * quad)
}

/// Is outer coordinate `k` pinned within [`coordinate_rail_margin`] of either
/// of its own box bounds?
///
/// Factored out so the λ-block REPORT and the θ-wide certificate FACE below
/// cannot drift apart about what "railed" means for the same coordinate: they
/// differ only in which coordinates they scan, never in the test.
///
/// The margin is the shared width-capped one, so this is the exact-bound test on
/// [`rail_relaxed_bounds`]' relaxed endpoints. It used to be a flat
/// [`CERTIFICATE_RAIL_MARGIN`] while the residual projector capped the same
/// constant at a quarter-width, and a box narrower than `2 ×
/// CERTIFICATE_RAIL_MARGIN` — the raw-κ chart window on any standardised
/// feature set — was then covered end to end by its own two margin bands: every
/// κ read railed, flat κ = 0 included, and the certificate's reduced Hessian
/// lost every row it was supposed to judge (#2462).
///
/// Comparing against the relaxed endpoints rather than `|θ_k − bound|` also
/// keeps an infeasible coordinate railed. Under the absolute-value form a point
/// *outside* the box by more than the margin reported interior, which is the one
/// reading that can never be right.
fn outer_coordinate_is_railed(theta: &Array1<f64>, k: usize, config: &OuterConfig) -> bool {
    RailTest::evaluate(theta, k, config).is_railed()
}

/// One coordinate's rail test: the verdict together with the interval and the
/// margin it was decided against (#2465).
///
/// The predicate above computed `(lo, hi)`, derived a margin from them, compared
/// against the relaxed endpoints, and returned a bare `bool` — so `railed=[3]`
/// reached the reader with everything that produced it already destroyed, and
/// recovering the interval on #2462 took a thirteen-point seeding sweep.
///
/// #2462 made carrying it necessary rather than merely useful: the margin is now
/// [`coordinate_rail_margin`], **width-capped per coordinate**, so two
/// coordinates in the same fit can be judged railed against different margins.
/// `railed=[1, 3]` is no longer even one statement, and the flag alone cannot
/// say which band either coordinate met.
#[derive(Debug, Clone, Copy)]
pub(crate) struct RailTest {
    /// Index into the θ vector.
    pub(crate) index: usize,
    /// The coordinate's value at the judged point.
    pub(crate) theta: f64,
    /// The interval it was tested against. `None` means the configured box does
    /// not cover this coordinate at all — which is itself the reason the verdict
    /// is `false`, a distinction the bare bool erased.
    pub(crate) box_bounds: Option<(f64, f64)>,
    /// The width-capped margin in force for THIS coordinate, from
    /// [`coordinate_rail_margin`]. Zero when the box does not cover it.
    pub(crate) margin: f64,
}

impl RailTest {
    fn evaluate(theta: &Array1<f64>, k: usize, config: &OuterConfig) -> Self {
        let box_bounds = match config.model_domain_bounds.as_ref() {
            Some((lo, hi)) if k < lo.len() && k < hi.len() => Some((lo[k], hi[k])),
            Some(_) => None,
            None => Some((-config.rho_bound, config.rho_bound)),
        };
        Self {
            // Indexed, not `get`-ed: every caller scans an index range derived
            // from `theta.len()`, so an out-of-range k is a caller bug and the
            // predicate this replaced panicked on it. Softening that to a silent
            // `false` would turn a bug into a coordinate quietly reported
            // un-railed.
            theta: theta[k],
            index: k,
            margin: box_bounds.map_or(0.0, |(lo, hi)| coordinate_rail_margin(lo, hi)),
            box_bounds,
        }
    }

    /// Pinned at or past either relaxed endpoint. This is the ONLY definition of
    /// railed in this file; both the λ-block report and the θ-wide certificate
    /// face route through it, and the relaxed-endpoint form (rather than
    /// `|θ_k − bound|`) is what keeps an infeasible coordinate railed (#2462).
    pub(crate) fn is_railed(self) -> bool {
        match self.box_bounds {
            Some((lo, hi)) => self.theta <= lo + self.margin || self.theta >= hi - self.margin,
            None => false,
        }
    }
}

impl std::fmt::Display for RailTest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.box_bounds {
            Some((lo, hi)) => write!(
                f,
                "#{} theta={:.6e} box=[{:.6e}, {:.6e}] margin={:.3e} railed_at=(<={:.6e} or >={:.6e})",
                self.index,
                self.theta,
                lo,
                hi,
                self.margin,
                lo + self.margin,
                hi - self.margin,
            ),
            None => write!(
                f,
                "#{} theta={:.6e} box=NOT-COVERED-BY-CONFIGURED-BOUNDS",
                self.index, self.theta,
            ),
        }
    }
}

/// The certificate-facing facts for `indices`: the interval and margin each
/// railed coordinate was judged against (#2530).
///
/// Built from [`RailTest`], which is the single definition of railed, so the
/// certificate reports what the predicate actually decided rather than a second
/// derivation of it. A coordinate the configured box does not cover contributes
/// nothing: it was not judged against an interval, so there is no interval to
/// report.
pub(crate) fn railed_coordinate_facts(
    theta: &Array1<f64>,
    indices: &[usize],
    config: &OuterConfig,
) -> Vec<RailedCoordinateFact> {
    indices
        .iter()
        .filter_map(|&k| {
            let test = RailTest::evaluate(theta, k, config);
            test.box_bounds.map(|(lower, upper)| RailedCoordinateFact {
                index: test.index,
                theta: test.theta,
                lower,
                upper,
                margin: test.margin,
            })
        })
        .collect()
}

/// Render the rail tests for `indices`, so a refusal naming railed coordinates
/// also states the interval and margin each was judged against (#2465).
pub(crate) fn rail_test_summary(
    theta: &Array1<f64>,
    indices: &[usize],
    config: &OuterConfig,
) -> String {
    indices
        .iter()
        .map(|&k| RailTest::evaluate(theta, k, config).to_string())
        .collect::<Vec<_>>()
        .join(", ")
}

/// Smoothing coordinates (leading ρ block) railed against the outer box.
///
/// This is the **report**. `OuterCriterionCertificate::lambdas_railed` indexes
/// *smoothing parameters*, and every consumer reads it that way — gam-report's
/// "λ railed" warning, the pyffi certificate surface, `is_clean`. It is
/// deliberately NOT the set the certificate reasons with; see
/// [`certificate_railed_coordinates`].
pub(crate) fn certificate_railed_lambdas(
    rho: &Array1<f64>,
    rho_dim: usize,
    config: &OuterConfig,
) -> Vec<usize> {
    (0..rho_dim.min(rho.len()))
        .filter(|&k| outer_coordinate_is_railed(rho, k, config))
        .collect()
}

/// **Every** outer coordinate railed against its own box bound — the ρ block
/// *and* the trailing non-ρ blocks that a joint search carries in the same θ
/// vector under the same box: the spatial log-κ ψ coordinates and the
/// auxiliary coordinates.
///
/// This is the set every *decision* in [`certify_outer_optimality`] must use,
/// and using the λ-only report there instead was a real defect. The active-set
/// reasoning at a box-constrained optimum is a statement about coordinates, not
/// about what a coordinate happens to parameterize:
///
/// * the second-order condition only has to hold on the **feasible tangent
///   subspace**, so `certificate_hessian_is_psd_off_railed` deletes the railed
///   rows and columns. `layout.rho_dim()` is `n_params − psi_dim`, so a ψ
///   coordinate pinned on its data-derived κ window stayed *inside* that
///   sub-block and its saturated (and, near a window edge, routinely negative)
///   curvature row decided `hessian_psd = false` for the whole fit. That is
///   precisely the failure the block's own comment says must not happen — "a
///   rail-caused indefiniteness would disable the very certificate that exists
///   to certify a railed optimum (#2299)" — it was simply never extended past
///   the ρ block;
/// * the same omission hid the coordinate from the asymptote-rail mint
///   (#2348 Inc 1), from tail-snap, from `interior_curvature_floor_clearance`,
///   from the #2155 saddle escape, and from the #2392 wrong-rail pull-back and
///   active-set reduction. A κ search whose rail is a ψ coordinate could
///   therefore never be certified *by construction*, no matter how correct its
///   gradient was;
/// * and it made the refusal message actively misleading, because
///   `project_gradient_vector` DOES project every coordinate against its own
///   bound. So `|g|` could collapse to a small `|Pg|` while `railed=[]`
///   reported nothing responsible for the collapse — the exact disagreement
///   #979 measured from the other side and could not explain.
///
/// Relaxing nothing: a coordinate near a bound whose gradient still points
/// *into* the box keeps its feasible-descent component in `|Pg|`
/// ([`project_gradient_vector`] zeros only the outward half), so a genuinely
/// unconverged interior direction is still refused.
pub(crate) fn certificate_railed_coordinates(
    theta: &Array1<f64>,
    config: &OuterConfig,
) -> Vec<usize> {
    (0..theta.len())
        .filter(|&k| outer_coordinate_is_railed(theta, k, config))
        .collect()
}

/// Which term of the stationarity bound's `max` chain actually set it.
///
/// `certify_outer_optimality` does not compute *a* bound; it takes the maximum of
/// up to five independently-derived quantities, and until #2458 nothing recorded
/// which one won. That is why a `bound=1.000e0` and a `bound=5.68e-6` came out of
/// the same message shape and telling them apart required reading this file --
/// the emitted number is a DERIVATIVE of a construction-site value, computed
/// later, so grepping the construction sites cannot match an observed bound.
///
/// The rungs are not interchangeable calibrations of one quantity. Only
/// [`Self::CurvatureResolvability`] is derived from what a gradient of that size
/// does to the criterion: `|Pg|·√(τ/Δpred)` simplifies to `√(2·h·τ)` -- the `|Pg|`
/// cancels -- i.e. the gradient magnitude below which the criterion cannot resolve
/// descent at all. The others are gradient-magnitude tests, blind to how curvature
/// maps a gradient to an objective change, which is exactly what this file's own
/// comment at the curvature block already says. Recording the rung is the
/// prerequisite for holding every route to the derived one: it makes "which
/// machinery did this route happen to have" an observable rather than an
/// inference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StationarityBoundSource {
    /// `outer_stationarity_band_at(config, cost)` -- the raw band.
    SolverBand,
    /// `flat_valley_converged_grad_bound(cost)`, gated on a `CostStallFlatValley`
    /// exit, reported at the band its formula gives.
    FlatValleyScore,
    /// The SAME score-relative band reported at its absolute ceiling, because
    /// `FLAT_VALLEY_CONVERGED_REL_GRAD*(1 + |score|)` exceeded
    /// `FLAT_VALLEY_CONVERGED_ABS_GRAD_CAP` (#2456/#2465).
    ///
    /// Split from [`Self::FlatValleyScore`] because a saturated value is not
    /// the band its label names -- it is a constant, identical on objectives
    /// three orders apart, and that is exactly why a `bound=1.000e0` could not
    /// be reconciled against the objective printed beside it. The bound is
    /// unchanged; what changes is that the message now says which of the two
    /// the number is.
    FlatValleyScoreCeiling,
    /// The cost-stall guard's measured probe-noise floor `σ̂/Δ` (#2241). Diverges
    /// as the step collapses, which is the regime it fires in.
    ProbeNoiseFloor,
    /// `|Pg|·√(τ/Δpred)` = `√(2·h·τ)` (#2253/#2249/#2015/#2091) -- the only rung
    /// with a derivation from the criterion's own resolution.
    CurvatureResolvability,
    /// Twice the same-ρ spread between the run-recorded and certificate-time
    /// gradients (#2299): the measuring instrument's demonstrated noise.
    GradientReproducibility,
    /// `config.tolerance` judged against the EFS/fixed-point route's
    /// normalized residual `‖(θ⁺−θ)/scale‖_∞` -- not against a gradient norm
    /// at all. The route has no ladder: it exposes no analytic gradient, so
    /// none of the gradient-magnitude rungs above is even computable on it,
    /// and the certificate it mints (`OuterStationarityCertificate::FixedPoint`)
    /// already declares `bound: config.tolerance`. Recording it names the one
    /// thing the shared `bound=` field could not previously say -- that the
    /// number is a residual tolerance and the quantity beside it is a residual.
    ///
    /// Only the ONE refusal on that route which has actually formed the
    /// residual carries this. Its eight early exits formed none, and now say so.
    FixedPointResidual,
}

impl StationarityBoundSource {
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::SolverBand => "solver-band",
            Self::FlatValleyScore => "flat-valley-score",
            Self::FlatValleyScoreCeiling => "flat-valley-score-ceiling",
            Self::ProbeNoiseFloor => "probe-noise-floor",
            Self::CurvatureResolvability => "curvature-resolvability",
            Self::GradientReproducibility => "gradient-reproducibility",
            Self::FixedPointResidual => "fixed-point-residual",
        }
    }

    /// Whether this rung is the derived resolvability standard rather than a
    /// gradient-magnitude substitute adopted where that standard was unavailable.
    pub(crate) fn is_derived_standard(self) -> bool {
        matches!(self, Self::CurvatureResolvability)
    }

    /// The neutral projection carried on the refusal, so a red states its own
    /// rung instead of leaving it to be inferred from the numbers.
    pub(crate) fn provenance(self) -> StationarityRung {
        StationarityRung {
            label: self.label(),
            derived_standard: self.is_derived_standard(),
        }
    }
}

/// A stationarity bound bundled with the rung that produced it (#2458).
///
/// The bound and its provenance travelled separately before this: the bound was
/// a bare `f64` argument and the rung stayed a local in the certificate block,
/// so every refusal outside that block reported a number with no way to say
/// which standard it came from. Bundling them makes it impossible to pass one
/// without deciding the other.
///
/// The source is NOT optional, and there is no constructor for "a number with
/// no standard". An earlier revision carried an `unrecorded` escape that 22 of
/// the 30 refusal paths took; naming those 22 was the first repair, and it was
/// half right. Twenty of them do not have an unclassified bound — they have NO
/// bound. They refuse before any stationarity residual exists (a failed
/// terminal evaluation, a malformed gradient, a non-converged inner state), and
/// the configured constant they used to report was never weighed against the
/// point. Those now carry [`StationarityStandard::NoComparison`] and print no
/// bound at all; only a route that formed a residual constructs one of these.
#[derive(Debug, Clone, Copy)]
pub(crate) struct StationarityBound {
    value: f64,
    source: StationarityBoundSource,
}

impl StationarityBound {
    /// A bound the rung ladder produced, carrying the rung it came from.
    pub(crate) fn from_ladder(value: f64, source: StationarityBoundSource) -> Self {
        Self { value, source }
    }

    /// The EFS/fixed-point route's standard: `config.tolerance` against a
    /// normalized fixed-point residual
    /// ([`StationarityBoundSource::FixedPointResidual`]).
    ///
    /// This is the bound `certify_fixed_point_optimality` mints its
    /// `OuterStationarityCertificate::FixedPoint` against, so the refusal that
    /// weighs a formed residual reports the route's own declared standard
    /// rather than a gradient tolerance it never used.
    pub(crate) fn fixed_point_residual(config: &OuterConfig) -> Self {
        Self::from_ladder(
            config.tolerance,
            StationarityBoundSource::FixedPointResidual,
        )
    }

    pub(crate) fn value(self) -> f64 {
        self.value
    }

    /// Whether this value was derived by mapping the caller's residual through
    /// the caller's Hessian. Such a value cannot cross into a reduced face:
    /// the face must mint its own bound from its own Hessian and residual.
    pub(crate) fn requires_face_local_derivation(self) -> bool {
        self.source.is_derived_standard()
    }

    pub(crate) fn rung(self) -> StationarityRung {
        self.source.provenance()
    }
}

impl From<StationarityBound> for StationarityStandard {
    fn from(bound: StationarityBound) -> Self {
        Self::Measured {
            bound: bound.value(),
            rung: bound.rung(),
        }
    }
}

fn outer_nonconvergence_error(
    context: &str,
    reason: &str,
    result: &OuterResult,
    projected_grad_norm: Option<f64>,
    stationarity_standard: impl Into<StationarityStandard>,
) -> EstimationError {
    // Solver provenance, appended to every outer non-convergence.
    //
    // The certificate answers "is this point stationary" — and when it is not,
    // the next question is always "then why did the search STOP here", which
    // the message did not answer. A binomial/logit P-spline (#1575/#1561)
    // refuses with `|Pg|=7.5e-1` against a `1.0e-2` bound, PSD Hessian, nothing
    // railed, all coordinates 18 e-folds inside the box, "after 7 outer
    // iteration(s)" — and the 7 is the whole mystery. `converged` distinguishes
    // a solver that CLAIMED convergence (a tolerance desync, and the only case
    // the #2273/#2374 resume will retry) from one that ran out of budget;
    // `operator_stop_reason` separates a trust-region reject floor from a
    // flat-valley cost stall from an iteration budget. All three are already on
    // the result and cost nothing to print.
    let reason = format!(
        "{reason}; solver provenance: origin={:?}, plan={}, claimed_converged={}{}{}{}",
        // #2465 at a fifth site. `termination=<no opt solver produced this
        // result>` says an `opt` solver did not decide this, and stops there;
        // `origin` says WHICH lane did. Both are already on the result.
        result.origin,
        result.plan_used,
        result.solver_claimed_convergence(),
        // #2547: `stop_reason` is the coarse projection. `termination` is
        // the test the solver actually applied plus the quantity it was
        // judged against, so a reader gets "the L-infinity window fired at
        // a threshold of 8.9e-1" rather than "it stopped". A stop that
        // made no stationarity claim at all (an iteration budget, a
        // collapsed trust region) says so, instead of leaving that to be
        // inferred from a bare gradient norm nothing compared it to.
        result
            .solver_termination
            .map(|t| {
                let evidence = match t.stationarity_evidence() {
                    Some(e) => format!(
                        " [measured={:.6e} vs threshold={:.6e}, {:?}, {:?}]",
                        e.measured, e.threshold, e.norm, e.scaling
                    ),
                    None => " [made no stationarity claim]".to_string(),
                };
                format!(", termination={t}{evidence}")
            })
            .unwrap_or_else(|| ", termination=<no opt solver produced this result>".to_string()),
        result
            .operator_stop_reason
            .map(|stop| format!(", stop_reason={stop:?}"))
            .unwrap_or_default(),
        result
            .converged_via()
            .map(|via| format!(", converged_via={via:?}"))
            .unwrap_or_default(),
    );
    // #2465: the line search's own verdict, when one failed. `StepSizeTooSmall`
    // and `MaxAttempts` are different defects with different repairs, and
    // "line_search_failed" alone distinguishes neither.
    // The cost-stall guard's measured probe-noise floor, when it published one.
    // A stall halted against a noise bound and a stall halted against the
    // score-relative flat band are different verdicts, and only the first
    // carries this.
    let reason = match result.cost_stall_probe_scale {
        Some((noise_floor, probe_radius)) => format!(
            "{reason}, cost_stall_window=[noise_floor={noise_floor:.6e}, \
             probe_radius={probe_radius:.6e}, ratio={:.6e}]",
            noise_floor / probe_radius
        ),
        None => reason,
    };
    let reason = match result.flat_noise_grad_bound {
        Some(bound) => format!("{reason}, flat_noise_grad_bound={bound:.6e}"),
        None => reason,
    };
    let reason = match result.line_search_failure {
        Some((failure_reason, max_attempts)) => format!(
            "{reason}, line_search={failure_reason:?} after {max_attempts} attempt(s) \
             [StepSizeTooSmall = the direction descended but no step improved the \
             objective; MaxAttempts = the bracket never closed]"
        ),
        None => reason,
    };
    EstimationError::RemlDidNotConverge {
        context: context.to_string(),
        reason,
        iterations: result.iterations,
        final_value: result.final_value,
        projected_grad_norm,
        stationarity_standard: stationarity_standard.into(),
        rho_checkpoint: result.rho.to_vec(),
    }
}

/// Roundoff envelope for two independently assembled values of the same outer
/// criterion at the same point.
///
/// Value-only and derivative-bearing evaluation lanes may use different
/// kernels and reduction trees, so bitwise identity is not a valid contract.
/// Their relative disagreement must nevertheless stay below the square root of
/// machine epsilon: beyond that scale the derivative sample is not evidence
/// about the scalar objective the optimizer's value lane ranks.
pub fn outer_value_agreement_bound(value_only: f64, derivative_sample: f64) -> f64 {
    f64::EPSILON.sqrt() * value_only.abs().max(derivative_sample.abs()).max(1.0)
}

fn audit_outer_value_agreement(
    context: &str,
    value_only: f64,
    derivative_sample: f64,
    result: &mut OuterResult,
    projected_grad_norm: Option<f64>,
    stationarity_standard: impl Into<StationarityStandard>,
) -> Result<(), EstimationError> {
    let bound = outer_value_agreement_bound(value_only, derivative_sample);
    let disagreement = (value_only - derivative_sample).abs();
    if disagreement <= bound {
        return Ok(());
    }

    // The value-only lane is the scalar criterion authority. Preserve it on
    // the resumable checkpoint rather than the derivative lane's inconsistent
    // scalar; no certificate may be attached to this mixed evidence.
    result.final_value = value_only;
    Err(outer_nonconvergence_error(
        context,
        &format!(
            "cost-only value disagrees with analytic-sample value at the same outer point: \
             value-only={value_only:.16e}, analytic-sample={derivative_sample:.16e}, \
             disagreement={disagreement:.3e}, roundoff bound={bound:.3e}"
        ),
        result,
        projected_grad_norm,
        stationarity_standard,
    ))
}

fn certify_fixed_point_optimality(
    obj: &mut dyn OuterObjective,
    config: &OuterConfig,
    context: &str,
    result: &mut OuterResult,
    fidelity: CertificationFidelity,
) -> Result<OuterCriterionCertificate, EstimationError> {
    let layout = obj.capability().theta_layout();
    // A fixed-point residual is only a certificate for the scalar objective
    // when both lanes price that same objective at the same rho. Sample the
    // authoritative value lane first; the analytic fixed-point evaluator then
    // remains the installed terminal-state owner.
    //
    // MINT ONLY (#2359), for the same reason as on the analytic path: whether
    // the two lanes agree is terminal proof work, not a ranking signal, and on
    // this route it is the ONLY thing that distinguishes screening from mint —
    // there is no order-four ladder to reserve here.
    let value_only = match fidelity {
        CertificationFidelity::Screening => None,
        CertificationFidelity::Mint => {
            let sample = obj
                .eval_with_order(&result.rho, OuterEvalOrder::Value)
                .map_err(|err| {
                    outer_nonconvergence_error(
                        context,
                        &format!("terminal value-only certificate evaluation failed: {err}"),
                        result,
                        None,
                        StationarityStandard::NoComparison,
                    )
                })?
                .cost;
            if !sample.is_finite() {
                return Err(outer_nonconvergence_error(
                    context,
                    "terminal value-only certificate evaluation returned a non-finite objective value",
                    result,
                    None,
                    StationarityStandard::NoComparison,
                ));
            }
            Some(sample)
        }
    };
    let evaluation = obj
        .eval_fixed_point_certificate(&result.rho)
        .map_err(|err| {
            outer_nonconvergence_error(
                context,
                &format!("analytic fixed-point certificate evaluation failed: {err}"),
                result,
                None,
                StationarityStandard::NoComparison,
            )
        })?;
    if !inner_solve_converged(config.outer_inner_cap.as_ref()) {
        return Err(outer_nonconvergence_error(
            context,
            "terminal fixed-point evidence was evaluated at a non-converged inner state",
            result,
            None,
            StationarityStandard::NoComparison,
        ));
    }
    if evaluation.coordinates.len() != layout.n_params {
        return Err(outer_nonconvergence_error(
            context,
            &format!(
                "fixed-point certificate returned {} coordinates for an outer problem of dimension {}",
                evaluation.coordinates.len(),
                layout.n_params
            ),
            result,
            None,
            StationarityStandard::NoComparison,
        ));
    }
    if !evaluation.cost.is_finite() {
        return Err(outer_nonconvergence_error(
            context,
            "fixed-point certificate returned a non-finite objective value",
            result,
            None,
            StationarityStandard::NoComparison,
        ));
    }
    if let Some(value_only) = value_only {
        audit_outer_value_agreement(
            context,
            value_only,
            evaluation.cost,
            result,
            None,
            StationarityStandard::NoComparison,
        )?;
    }

    let mut normalized_updates = Vec::with_capacity(layout.n_params);
    let mut uncovered = Vec::new();
    for (index, coordinate) in evaluation.coordinates.iter().enumerate() {
        match coordinate {
            FixedPointCoordinateCertificate::Covered { update, scale }
                if update.is_finite() && scale.is_finite() && *scale > 0.0 =>
            {
                normalized_updates.push(*update / *scale);
            }
            FixedPointCoordinateCertificate::Covered { update, scale } => {
                uncovered.push(format!(
                    "coordinate {index} has invalid covered residual update={update} scale={scale}"
                ));
                normalized_updates.push(f64::NAN);
            }
            FixedPointCoordinateCertificate::Uncovered { reason } => {
                uncovered.push(format!("coordinate {index}: {reason}"));
                normalized_updates.push(f64::NAN);
            }
        }
    }
    if !uncovered.is_empty() {
        return Err(outer_nonconvergence_error(
            context,
            &format!(
                "fixed-point certificate lacks root-equivalent analytic coverage: {}",
                uncovered.join("; ")
            ),
            result,
            None,
            StationarityStandard::NoComparison,
        ));
    }

    let (lower, upper) = outer_model_domain_bounds_template(config, layout.n_params);
    let mut raw_inf = 0.0_f64;
    let mut projected_inf = 0.0_f64;
    for index in 0..layout.n_params {
        let update = normalized_updates[index];
        raw_inf = raw_inf.max(update.abs());
        // `update` is a signed descent/update direction, the negative of the
        // gradient convention used by `projected_gradient_norm`: at a lower
        // bound a negative update points out of the box, and at an upper bound
        // a positive update does. Only those infeasible multiplier components
        // are removed.
        let projected = if result.rho[index] <= lower[index] {
            update.max(0.0)
        } else {
            update
        };
        let projected = if result.rho[index] >= upper[index] {
            projected.min(0.0)
        } else {
            projected
        };
        projected_inf = projected_inf.max(projected.abs());
    }

    result.final_value = evaluation.cost;
    result.final_grad_norm = None;
    result.final_gradient = None;
    result.final_hessian = None;

    let certificate = OuterCriterionCertificate {
        stationarity: OuterStationarityCertificate::FixedPoint {
            residual_inf_norm: raw_inf,
            projected_residual_inf_norm: projected_inf,
            bound: config.tolerance,
            rung: StationarityBoundSource::FixedPointResidual.provenance().into(),
            covered_coordinates: layout.n_params,
        },
        // The EFS/fixed-point route exposes no analytic Hessian, so there was
        // a curvature question and nothing could answer it (#2561).
        curvature: CurvatureEvidence::NotAvailable,
        lambdas_railed: certificate_railed_lambdas(&result.rho, layout.rho_dim(), config),
        railed_facts: railed_coordinate_facts(
            &result.rho,
            &certificate_railed_lambdas(&result.rho, layout.rho_dim(), config),
            config,
        ),
        curvature_floor: None,
    };
    result.criterion_certificate = Some(certificate.clone());
    if !certificate.certifies() {
        return Err(outer_nonconvergence_error(
            context,
            &certificate.summary(),
            result,
            Some(projected_inf),
            StationarityBound::fixed_point_residual(config),
        ));
    }

    let via = match result.termination.proposed_via() {
        Some(via @ OuterConvergedVia::RecurrentIncumbent { .. }) => via,
        _ => OuterConvergedVia::FixedPointStationary {
            projected_residual_inf_norm: projected_inf,
            certificate_bound: config.tolerance,
        },
    };
    result.termination.certify(via);
    log::info!("[CERTIFICATE] {context}: {}", certificate.summary());
    Ok(certificate)
}

/// Build the mandatory analytic optimality certificate at the returned point.
///
/// The objective is evaluated at the selected point through both its
/// authoritative value-only lane and its analytic derivative lane. The two
/// values must agree within their roundoff envelope before the derivative
/// sample can certify the scalar objective. Missing, malformed, non-finite, or
/// split-objective evidence is non-convergence: an optimizer status bit cannot
/// substitute for a stationarity certificate. Exact analytic curvature is
/// checked when the objective declares it and can materialize it; BFGS/EFS
/// solver geometry is never mistaken for objective curvature.
/// Which derivative fidelity a certification pass is allowed to spend.
///
/// #2359: the generic REML/LAML Hessian consumes the row-family derivative
/// ladder through order FOUR while its analytic gradient stops at order three,
/// so order four is a mint-time cost, not a per-candidate one. A multi-start
/// screens every seed it starts — that screening is a first-order gate
/// (stationarity, KKT projection, rail facts), and `curvature_admissible()`
/// reads `hessian_psd != Some(false)`, so a `None` curvature verdict certifies
/// on stationarity alone. The one order-four evaluation belongs to the winner,
/// once, and its verdict is the one that mints.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CertificationFidelity {
    /// Per-candidate multi-start gate. Never spends order four.
    Screening,
    /// The single terminal mint audit. Spends order four when the objective
    /// declares an analytic Hessian.
    Mint,
}

pub(crate) fn certify_outer_optimality(
    obj: &mut dyn OuterObjective,
    config: &OuterConfig,
    context: &str,
    result: &mut OuterResult,
) -> Result<OuterCriterionCertificate, EstimationError> {
    certify_outer_optimality_with_fidelity(obj, config, context, result, CertificationFidelity::Mint)
}

pub(crate) fn certify_outer_optimality_with_fidelity(
    obj: &mut dyn OuterObjective,
    config: &OuterConfig,
    context: &str,
    result: &mut OuterResult,
    fidelity: CertificationFidelity,
) -> Result<OuterCriterionCertificate, EstimationError> {
    result.termination.begin_certification();
    let terminal_cap_guard = config
        .outer_inner_cap
        .as_ref()
        .map(TerminalInnerCapGuard::lift);
    if terminal_cap_guard.is_some() || obj.owns_terminal_coefficient_mode() {
        // `reset` is deliberately conditional on the presence of the cap
        // contract.  Those are the REML/mixture objectives whose search cache
        // can contain a coarse inner state; uncapped objectives retain their
        // ordinary stateful certification semantics.
        //
        // The `owns_terminal_coefficient_mode()` disjunct (#2334) closes the
        // gap for cap-less objectives that install an owned coefficient mode:
        // the certifying re-eval below must start from the same clean baseline
        // that `finalize_outer_result` used, so the mode's objective bitwise
        // matches the certified `final_value` even when the inner solve is
        // bimodal at `rho_star`.
        obj.reset();
    }
    let outcome =
        certify_outer_optimality_at_terminal_fidelity(obj, config, context, result, true, fidelity);
    drop(terminal_cap_guard);
    if outcome.is_err() {
        result.termination.refuse_certificate();
    }
    outcome
}

fn certify_outer_optimality_at_terminal_fidelity(
    obj: &mut dyn OuterObjective,
    config: &OuterConfig,
    context: &str,
    result: &mut OuterResult,
    allow_tail_snap: bool,
    fidelity: CertificationFidelity,
) -> Result<OuterCriterionCertificate, EstimationError> {
    let capability = obj.capability();
    let layout = capability.theta_layout();
    layout
        .validate_point_len(&result.rho, "outer certificate point")
        .map_err(|err| {
            EstimationError::RemlOptimizationFailed(format!(
                "{context}: invalid outer certificate point: {err}"
            ))
        })?;
    // Certification is an ownership boundary: reinstall the model-domain
    // derivative face before either screening or mint evaluates the selected
    // point. This makes the terminal audit independent of whichever
    // canonicalized plan or prior fit last touched this thread's IFT state, and
    // prevents an active-set search override from surviving as derivative
    // geometry. Screening and mint can differ in derivative order, never in the
    // model whose feasible directions they differentiate (#2514).
    let model_domain_bounds_for_derivatives =
        outer_model_domain_bounds_template(config, layout.n_params);
    crate::estimate::reml::outer_eval::record_current_outer_rho_model_upper_bounds_for_ift(
        &model_domain_bounds_for_derivatives.1,
    );
    if result.rho.iter().any(|value| !value.is_finite()) {
        return Err(outer_nonconvergence_error(
            context,
            "the selected checkpoint contains non-finite coordinates",
            result,
            None,
            StationarityStandard::NoComparison,
        ));
    }
    if layout.n_params == 0 {
        let value = obj.eval_cost(&result.rho).map_err(|err| {
            outer_nonconvergence_error(
                context,
                &format!("zero-dimensional final objective evaluation failed: {err}"),
                result,
                Some(0.0),
                StationarityStandard::NoComparison,
            )
        })?;
        if !value.is_finite() {
            return Err(outer_nonconvergence_error(
                context,
                "the zero-dimensional final objective is non-finite",
                result,
                Some(0.0),
                StationarityStandard::NoComparison,
            ));
        }
        let certificate = OuterCriterionCertificate {
            stationarity: OuterStationarityCertificate::AnalyticGradient {
                grad_norm: 0.0,
                projected_grad_norm: 0.0,
                bound: outer_gradient_tolerance(config).abs,
                // No smoothing estimand: the empty score is stationary by
                // construction, not by clearing this band (#2530).
                rung: StationarityRung::EMPTY_ESTIMAND.into(),
            },
            // No estimand, so no curvature exists to be admissible — the
            // second-order twin of the EMPTY_ESTIMAND rung above (#2561).
            curvature: CurvatureEvidence::NoEstimand,
            lambdas_railed: Vec::new(),
            railed_facts: Vec::new(),
            curvature_floor: None,
        };
        result.final_value = value;
        result.final_grad_norm = Some(0.0);
        result.final_gradient = Some(Array1::zeros(0));
        result.final_hessian = None;
        result
            .termination
            .certify(OuterConvergedVia::GradientStationary);
        result.criterion_certificate = Some(certificate.clone());
        return Ok(certificate);
    }
    if matches!(result.plan_used.solver, Solver::Efs | Solver::HybridEfs)
        && capability.gradient != Derivative::Analytic
    {
        return certify_fixed_point_optimality(obj, config, context, result, fidelity);
    }
    if capability.gradient != Derivative::Analytic {
        return Err(outer_nonconvergence_error(
            context,
            "the objective exposes no analytic gradient for final certification",
            result,
            None,
            StationarityStandard::NoComparison,
        ));
    }

    // Sample the scalar authority FIRST and leave the derivative-bearing
    // evaluator as the terminal state owner. This is one same-rho audit, not a
    // finite-difference derivative: it catches a split value/gradient
    // implementation without violating the production no-FD contract.
    //
    // The order is load-bearing in two independent ways, and #2583 briefly
    // inverted it:
    //
    //   * Every real outer objective's derivative lane is a warm-started inner
    //     SOLVE, so evaluating it ADVANCES the inner state. A value sample
    //     taken afterwards reads the state that solve left behind, not the
    //     state it priced. The value lane, by contrast, reads the installed
    //     state without advancing it, so value-then-derivative is the only
    //     order in which both samples price ONE inner state.
    //   * The mint's derivative-bearing evaluation must be the LAST evaluation
    //     of the certification, because it is the terminal coefficient-mode
    //     owner every downstream consumer reads (#2359). A trailing value-only
    //     request re-installs a value-only pass as that owner.
    //
    // Inverting it also made the audit VACUOUS on the one route it was aimed
    // at: the REML objective's outer-eval cache serves any `Value` request from
    // whatever entry the derivative call just wrote, so the audit compared a
    // number with itself and could not fire. Agreement by cache lookup is not
    // agreement between two assemblies of the criterion.
    //
    // The genuine #2583 hazard — two INNER SOLVES at one ρ, disagreeing at
    // inner-tolerance scale rather than at the √ε scale this bound is
    // calibrated for — is addressed where it lives: the value lane's bundle is
    // stored under the shared ρ key, so the derivative lane that follows reuses
    // that inner solution instead of re-solving from it.
    //
    // BOTH fidelities pay this, deliberately. It is tempting to reserve it for
    // the mint the way order four is reserved (#2359), but the two are not
    // alike: order four is evidence the filter does not consume, while the
    // value-agreement audit is a REFUSAL GATE on the very number the
    // multistart ranks by. `retain_best_outer_checkpoint` orders candidates on
    // `final_value`; admitting a candidate whose value lane disagrees with its
    // derivative lane lets a desynced candidate win the ranking on a number
    // nothing has validated, and mint then refuses the whole fit instead of a
    // runner-up carrying it. Pinned by
    // `analytic_hessian_candidate_screening_requires_only_first_order_evidence_2414`,
    // which asserts screening's orders are exactly [Value, ValueAndGradient].
    //
    // The EFS/fixed-point route is different and IS gated on fidelity — see
    // `certify_fixed_point_optimality`. It returns above this block, and there
    // screening and mint are otherwise identical work (no order-four ladder to
    // reserve), so running the audit twice buys nothing at all.
    let value_only = obj
        .eval_with_order(&result.rho, OuterEvalOrder::Value)
        .map_err(|err| {
            outer_nonconvergence_error(
                context,
                &format!("terminal value-only certificate evaluation failed: {err}"),
                result,
                result.final_grad_norm,
                StationarityStandard::NoComparison,
            )
        })?
        .cost;
    if !value_only.is_finite() {
        return Err(outer_nonconvergence_error(
            context,
            "terminal value-only certificate evaluation returned a non-finite objective value",
            result,
            result.final_grad_norm,
            StationarityStandard::NoComparison,
        ));
    }

    // Order four is reserved for the mint audit (#2359). A screening pass takes
    // the same first-order evidence the no-analytic-Hessian path already
    // certifies on, so the multi-start keeps its filter without building the
    // order-four family tower once per seed.
    // One boolean drives BOTH the request below and the requirement at the
    // analytic-Hessian block: a pass that does not ask for curvature must not
    // then refuse the candidate for not supplying it. Splitting them made every
    // screened candidate of an analytic-Hessian objective fail certification
    // with "declared analytic curvature but returned none at the final point" —
    // a statement about this pass's own eval order, not about the candidate.
    //
    // #2596 addendum, and the reason this line is no longer the whole story:
    // the reservation is sound for the CURVATURE VERDICT (a missing verdict is
    // permissive) but NOT for the stationarity BOUND, one of whose rungs the
    // same Hessian owns and where a missing rung is strictly tightening. The
    // screening pass therefore re-acquires the Hessian, for the bound alone,
    // when the un-widened band would refuse — see `screening_bound_curvature`
    // below. That escalation is the reason this boolean can stay
    // fidelity-gated: it still governs the ORDINARY path, and order four still
    // costs nothing on any candidate that clears its first-order band.
    let wants_analytic_hessian =
        capability.hessian.is_analytic() && matches!(fidelity, CertificationFidelity::Mint);
    let order = if wants_analytic_hessian {
        OuterEvalOrder::ValueGradientHessian
    } else {
        OuterEvalOrder::ValueAndGradient
    };
    let evaluation = obj.eval_with_order(&result.rho, order).map_err(|err| {
        outer_nonconvergence_error(
            context,
            &format!("analytic final-point evaluation failed: {err}"),
            result,
            result.final_grad_norm,
            StationarityStandard::NoComparison,
        )
    })?;

    if !inner_solve_converged(config.outer_inner_cap.as_ref()) {
        return Err(outer_nonconvergence_error(
            context,
            "terminal analytic evidence was evaluated at a non-converged inner state",
            result,
            None,
            StationarityStandard::NoComparison,
        ));
    }
    layout
        .validate_gradient_len(&evaluation.gradient, "outer certificate gradient")
        .map_err(|err| {
            outer_nonconvergence_error(
                context,
                &format!("malformed analytic final gradient: {err}"),
                result,
                None,
                StationarityStandard::NoComparison,
            )
        })?;
    if !evaluation.cost.is_finite() || evaluation.gradient.iter().any(|value| !value.is_finite()) {
        return Err(outer_nonconvergence_error(
            context,
            "the analytic final-point value or gradient is non-finite",
            result,
            None,
            StationarityStandard::NoComparison,
        ));
    }

    let bounds = outer_model_domain_bounds_template(config, layout.n_params);
    // A penalty creeping toward the ±rho_bound infinite-smoothing ceiling never reaches
    // it EXACTLY — each outer step only shrinks the gap, so it lands strictly inside the
    // box (the #2299 checkpoint sits at ρ=29.9938, not 30). `certificate_railed_lambdas`
    // then flags it railed via `CERTIFICATE_RAIL_MARGIN`, but the exact `x >= upper` /
    // `x <= lower` box-KKT projection treats it as INTERIOR and its outward pull inflates
    // |Pg| above the (tiny) stationarity bound — the fit refuses a genuine railed optimum.
    // Project the stationarity residual with the box endpoints relaxed inward by that SAME
    // rail margin, so "railed" means ONE thing to the detector AND the projector: a
    // within-tolerance coordinate whose gradient points OUT of the box has its KKT-multiplier
    // component removed rather than counted as a stationarity residual (#2299). The
    // projection only zeros the OUTWARD half (`.max(0.0)`/`.min(0.0)`), so a coordinate near
    // the bound that still has feasible-descent gradient keeps it and is never falsely
    // certified.
    let rail_projection_bounds = rail_relaxed_bounds(&bounds);
    let grad_norm = evaluation.gradient.dot(&evaluation.gradient).sqrt();
    // The terminal inner coefficients β(ρ̂), published by the REML bridge on
    // every eval (`inner_beta_hint`). Used to scale the estimand tolerance for
    // the asymptote-rail certificate (#2348 Inc 1).
    let terminal_beta = evaluation.inner_beta_hint.clone();
    // KKT-projected gradient VECTOR (not just its norm): the norm feeds the
    // stationarity certificate below, and the vector feeds the curvature-scaled
    // flat-valley Newton decrement (#2253/#2249/#2015) once the analytic Hessian
    // is in hand.
    let projected_gradient = project_gradient_vector(
        &result.rho,
        &evaluation.gradient,
        Some(&rail_projection_bounds),
    );
    let projected_grad_norm = projected_gradient.iter().map(|v| v * v).sum::<f64>().sqrt();
    // #2514: preserve the literal first-order geometry used by this pass without
    // dumping an O(p) vector for large outer problems. A temporary active-set
    // search box can differ from the model's feasible box; when that happens,
    // screening and mint may otherwise print identical rho/raw gradients while
    // silently projecting against different faces. Record every detector-active
    // or actually projected coordinate, capped only in the log payload.
    if log::log_enabled!(log::Level::Info) {
        const PROJECTION_RECORD_LIMIT: usize = 32;
        let mut active_or_projected = 0usize;
        let mut projection_records = Vec::new();
        for k in 0..result.rho.len() {
            let detector_railed = outer_coordinate_is_railed(&result.rho, k, config);
            let raw = evaluation.gradient[k];
            let projected = projected_gradient[k];
            if detector_railed || raw.to_bits() != projected.to_bits() {
                active_or_projected += 1;
                if projection_records.len() < PROJECTION_RECORD_LIMIT {
                    projection_records.push((
                        k,
                        result.rho[k],
                        bounds.0[k],
                        bounds.1[k],
                        rail_projection_bounds.0[k],
                        rail_projection_bounds.1[k],
                        raw,
                        projected,
                        detector_railed,
                    ));
                }
            }
        }
        if active_or_projected > 0 {
            log::info!(
                "[CERTIFICATE-PROJECTION] {context}: fidelity={fidelity:?},                  active_or_projected={active_or_projected},                  records(index,rho,model_lo,model_hi,projection_lo,projection_hi,raw_g,projected_g,detector_railed)={projection_records:?}{}",
                if active_or_projected > projection_records.len() {
                    " (truncated)"
                } else {
                    ""
                },
            );
        }
    }
    // Anchored at the criterion value of the point being JUDGED, which is what
    // the mgcv `magic` rule means and what the solver's own band deliberately
    // no longer does (#2613): see `outer_stationarity_band_at`.
    let solver_bound = outer_stationarity_band_at(config, evaluation.cost);
    let mut bound_source = StationarityBoundSource::SolverBand;
    let mut stationarity_bound = if matches!(
        result.operator_stop_reason,
        Some(OperatorTrustRegionStopReason::CostStallFlatValley)
    ) {
        let flat_valley = flat_valley_converged_grad_bound(evaluation.cost);
        if flat_valley > solver_bound {
            bound_source = if flat_valley_score_saturates(evaluation.cost) {
                StationarityBoundSource::FlatValleyScoreCeiling
            } else {
                StationarityBoundSource::FlatValleyScore
            };
            flat_valley
        } else {
            solver_bound
        }
    } else {
        solver_bound
    };
    // #2241 — a cost-stall exit carries the guard's measured probe-noise-floor
    // gradient bound σ̂/Δ. The certificate must judge the re-measured final
    // gradient against the same flat band the guard certified, or the guard's
    // noise-scale convergence would be granted in the loop and revoked here.
    if let Some(noise_bound) = result.flat_noise_grad_bound
        && noise_bound.is_finite()
    {
        if noise_bound > stationarity_bound {
            bound_source = StationarityBoundSource::ProbeNoiseFloor;
            stationarity_bound = noise_bound;
        }
    }
    audit_outer_value_agreement(
        context,
        value_only,
        evaluation.cost,
        result,
        Some(projected_grad_norm),
        StationarityBound::from_ladder(stationarity_bound, bound_source),
    )?;

    // The optimizer's own recorded best-iterate evidence, captured before the
    // fresh certificate-time measurement overwrites it below. Together with
    // `evaluation` this is a SECOND independent measurement of the objective
    // at the same ρ — the raw material for the gradient-reproducibility floor
    // further down, at zero additional objective evaluations.
    let run_recorded_gradient = result.final_gradient.take();
    let run_recorded_value = result.final_value;

    // Install measured first-order evidence before any fallible curvature
    // processing. If curvature is malformed, the retained resume checkpoint
    // still carries the exact value/gradient that caused certification to stop.
    result.final_value = evaluation.cost;
    result.final_grad_norm = Some(projected_grad_norm);
    result.final_gradient = Some(evaluation.gradient);

    // #2596 — a pass that spends LESS evidence must not produce a STRONGER
    // refusal than the pass that mints.
    //
    // The stationarity bound is a LADDER, and one of its rungs — the
    // curvature-resolvability widening below — is owned by the analytic outer
    // Hessian. `Screening` deliberately does not spend the order-four ladder
    // (#2359), and that reservation was argued on the CURVATURE CONJUNCT:
    // `curvature_admissible()` reads `hessian_psd != Some(false)`, so a `None`
    // curvature verdict certifies on stationarity alone. True — but the Hessian
    // ALSO owns a rung of the stationarity BOUND, and there `None` is not
    // permissive: it silently reverts screening to the un-widened solver band.
    // Screening therefore applied a strictly tighter standard than the mint, and
    // a candidate it refused was discarded rather than deferred.
    //
    // Measured (#2596, lognormal location-scale AFT with a double-penalty
    // `s(z, bs="tp", k=10)`): the BFGS converged to the correct interior optimum
    // ρ = (0.378, −4.975) at cost 4.1926 with |Pg| = 7.29e-5 against a solver
    // band of 5.19e-5 — refused by a factor of 1.4. Both interior seeds were
    // refused, the multi-start fell through to the seed lattice's
    // over-smoothing boundary candidate, and THAT certified vacuously (at a
    // railed corner the box-KKT projection makes |Pg| identically zero) and was
    // minted at cost 110.94. The published smoothing parameter's own LAML was
    // 26× worse than the one the optimizer had already measured, and the fitted
    // smooth carried none of its signal. Every sibling arm of the same suite
    // reached the mint and got the rung that would have saved this one
    // (`curvature-scaled flat-valley bound 1.537e-3 … widened from
    // gradient-band 2.359e-4`); which side of the band a fit lands on is which
    // side the last BFGS step stopped on, not a statistical distinction.
    //
    // So spend the ladder at screening too — but ONLY when the un-widened bound
    // would refuse, and ONLY for the bound. The escalated curvature never
    // reaches the curvature verdict, the rail certificate, or the tail-snap, so
    // this can only ever turn a screening refusal into a screening
    // certification and never the reverse. A fit that clears its first-order
    // band is byte-identical and pays nothing, so #2359's "order four exactly
    // once, at the mint" continues to hold for every healthy fit.
    let screening_bound_curvature = if !wants_analytic_hessian
        && capability.hessian.is_analytic()
        && projected_grad_norm > stationarity_bound
    {
        log::info!(
            "[CERTIFICATE] {context}: screening's first-order band would refuse \
             (|Pg|={projected_grad_norm:.3e} > bound={stationarity_bound:.3e}, rung={}); \
             spending the order-four ladder so this refusal is judged by the SAME \
             stationarity bound the mint applies (#2596)",
            bound_source.label(),
        );
        // A failed or malformed escalation is NOT a refusal of the candidate: it
        // only means this rung is unavailable, which is exactly the state the
        // pass was already in. Fall through to the un-widened comparison.
        match obj.eval_with_order(&result.rho, OuterEvalOrder::ValueGradientHessian) {
            Ok(escalated) => match escalated.hessian.materialize_dense() {
                Ok(Some(hessian))
                    if layout
                        .validate_hessian_shape(&hessian, "outer certificate Hessian")
                        .is_ok()
                        && hessian.iter().all(|value| value.is_finite()) =>
                {
                    Some(hessian)
                }
                _ => {
                    log::info!(
                        "[CERTIFICATE] {context}: the escalated order-four evaluation \
                         supplied no usable curvature; keeping the first-order bound"
                    );
                    None
                }
            },
            Err(error) => {
                log::info!(
                    "[CERTIFICATE] {context}: the escalated order-four evaluation failed \
                     ({error}); keeping the first-order bound"
                );
                None
            }
        }
    } else {
        None
    };

    let analytic_hessian = if wants_analytic_hessian {
        match evaluation.hessian.materialize_dense() {
            Ok(Some(hessian)) => {
                layout
                    .validate_hessian_shape(&hessian, "outer certificate Hessian")
                    .map_err(|err| {
                        outer_nonconvergence_error(
                            context,
                            &format!("malformed analytic final Hessian: {err}"),
                            result,
                            Some(projected_grad_norm),
                            StationarityBound::from_ladder(stationarity_bound, bound_source),
                        )
                    })?;
                if hessian.iter().any(|value| !value.is_finite()) {
                    return Err(outer_nonconvergence_error(
                        context,
                        "the analytic final Hessian contains non-finite entries",
                        result,
                        Some(projected_grad_norm),
                        StationarityBound::from_ladder(stationarity_bound, bound_source),
                    ));
                }
                Some(hessian)
            }
            Ok(None) => {
                return Err(outer_nonconvergence_error(
                    context,
                    "the objective declared analytic curvature but returned none at the final point",
                    result,
                    Some(projected_grad_norm),
                    StationarityBound::from_ladder(stationarity_bound, bound_source),
                ));
            }
            Err(err) => {
                return Err(outer_nonconvergence_error(
                    context,
                    &format!("analytic final Hessian could not be certified: {err}"),
                    result,
                    Some(projected_grad_norm),
                    StationarityBound::from_ladder(stationarity_bound, bound_source),
                ));
            }
        }
    } else {
        None
    };

    // Curvature-scaled stationarity (#2253/#2249/#2015/#2091). The re-measured
    // projected gradient can sit modestly ABOVE the score-relative / probe-noise
    // bands even though NO step reduces the objective by more than the outer
    // tolerance — a weakly-identified small-n fit reaches this by a flat-valley
    // cost-stall, and an *already-stationary* fit reaches it at iteration 0 when
    // the plan search exhausts without stepping (a 2-parameter Gaussian-linear
    // REML lands λ→0 at a genuine interior optimum whose |Pg|≈1e-7 sits just above
    // an absolute score·1e-9 gradient floor tighter than the REML gradient's
    // matrix-factorization round-off). Whether that residual is genuine descent is
    // a second-order question the flat bands above cannot answer: they are
    // gradient-magnitude tests, blind to how the local curvature maps a gradient
    // to an objective change. The Newton decrement `½·gᵀH⁻¹g` (see
    // `newton_predicted_decrease`) IS that map — the exact predicted improvement of
    // a safeguarded second-order step. When it is below the outer objective
    // tolerance, the point is stationary at the resolution the criterion can be
    // optimized ("no further descent possible"), independent of HOW the solver
    // stopped.
    //
    // Applied whenever a PSD-along-gradient analytic Hessian is in hand (NOT gated
    // to a specific exit reason: the decrement test is the certificate, the exit
    // reason is not). It can NEVER wrongly certify a fit with real available
    // descent: it only widens when `curvature_grad_bound > stationarity_bound`, so
    // a well-identified fit that already clears `solver_bound` is untouched; a
    // gradient aligned with a near-flat Hessian direction inflates the decrement
    // and is rejected; a globally indefinite Hessian is rejected independently by
    // the `hessian_psd` gate inside `certifies()`. The derived widening is a
    // genuine, direction-aware curvature-scaled GRADIENT bound — the largest ‖Pg‖
    // that, in this gradient's direction under this curvature, still predicts a
    // decrease of exactly `objective_tol` — not a constant bump: because the
    // decrement scales quadratically with ‖g‖ at fixed direction, that bound is
    // `‖Pg‖·√(objective_tol/Δpred)`, which clears the actual ‖Pg‖ iff
    // `Δpred ≤ objective_tol`.
    // `screening_bound_curvature` is the #2596 escalation: at `Mint` it is
    // always `None` and this reads `analytic_hessian` exactly as before; at
    // `Screening` it is `Some` only on the would-refuse path, and it feeds THIS
    // rung and nothing else.
    if let Some(hessian) = analytic_hessian
        .as_ref()
        .or(screening_bound_curvature.as_ref())
        && let Some(predicted_decrease) = newton_predicted_decrease(hessian, &projected_gradient)
        && predicted_decrease.is_finite()
        && predicted_decrease > 0.0
    {
        // The SAME relative cost floor the cost-stall guard used to declare the
        // criterion stalled (run_plan.rs), so certification asserts nothing
        // tighter than the loop already proved about this surface.
        let objective_tol = outer_rel_cost_floor(config) * (1.0 + evaluation.cost.abs());
        let curvature_grad_bound =
            projected_grad_norm * (objective_tol / predicted_decrease).sqrt();
        if curvature_grad_bound.is_finite() && curvature_grad_bound > stationarity_bound {
            log::info!(
                "[CERTIFICATE] {context}: curvature-scaled flat-valley bound {curvature_grad_bound:.3e} \
                 (|Pg|={projected_grad_norm:.3e}, Newton ½gᵀH⁻¹g={predicted_decrease:.3e} ≤ tol {objective_tol:.3e}) \
                 widened from gradient-band {stationarity_bound:.3e}"
            );
            stationarity_bound = curvature_grad_bound;
            bound_source = StationarityBoundSource::CurvatureResolvability;
        }
    }

    // Gradient-reproducibility floor (#2299 fully-saturated smooth). A
    // stationarity certificate cannot resolve below the reproducibility of its
    // own measuring instrument: at a rail-adjacent optimum (λ ~ 1e12, the term
    // collapsed onto its penalty null space, edf saturated) the analytic
    // gradient is a difference of enormous canceling log-det terms whose
    // evaluation drifts run to run, so |Pg| measures round-off, not slope —
    // observed as the SAME ρ returning |g| ∈ {2.5e-3 … 4.5e-2} across
    // consecutive evaluations while the objective stays flat to 1e-7.
    //
    // The certifier already holds TWO independent measurements at this ρ: the
    // optimizer's recorded best-iterate gradient (`run_recorded_gradient`) and
    // the fresh certificate-time `evaluation` — so the instrument's
    // demonstrated noise costs ZERO additional objective evaluations (scripted
    // test objectives keep their exact call counts). A REAL residual gradient
    // reproduces (spread ≈ 0, no widening — genuine descent can never be
    // masked, and a deterministic objective yields bit-identical pairs), while
    // cancellation noise decorrelates (spread ~ |Pg|). The widening is gated
    // on the two measurements' objective VALUES agreeing to the same relative
    // floor the cost-stall guard uses, and the PSD gate below is unchanged.
    if projected_grad_norm > stationarity_bound
        && let Some(prior_gradient) = run_recorded_gradient.as_ref()
        && layout
            .validate_gradient_len(prior_gradient, "outer run-recorded gradient")
            .is_ok()
        && prior_gradient.iter().all(|value| value.is_finite())
        && run_recorded_value.is_finite()
    {
        const GRADIENT_REPRODUCIBILITY_WIDENING: f64 = 2.0;
        let objective_tol = config
            .rel_cost_tolerance
            .unwrap_or(config.tolerance * 1.0e-2)
            .max(COST_STALL_REL_TOL_FLOOR)
            * (1.0 + evaluation.cost.abs());
        let cost_drift = (run_recorded_value - evaluation.cost).abs();
        let prior_projected =
            project_gradient_vector(&result.rho, prior_gradient, Some(&rail_projection_bounds));
        let spread = (&prior_projected - &projected_gradient)
            .iter()
            .map(|v| v * v)
            .sum::<f64>()
            .sqrt();
        let repro_bound = GRADIENT_REPRODUCIBILITY_WIDENING * spread;
        if cost_drift <= objective_tol
            && repro_bound.is_finite()
            && repro_bound > stationarity_bound
            && projected_grad_norm <= repro_bound
        {
            log::info!(
                "[CERTIFICATE] {context}: gradient-reproducibility floor widened the \
                 stationarity bound to {repro_bound:.3e} (|Pg|={projected_grad_norm:.3e}, \
                 same-ρ spread between the run-recorded and certificate-time gradients \
                 {spread:.3e}, cost drift {cost_drift:.3e} ≤ tol {objective_tol:.3e})"
            );
            stationarity_bound = repro_bound;
            bound_source = StationarityBoundSource::GradientReproducibility;
        }
    }

    // #2458/#2479 -- the bound's own provenance, emitted UNCONDITIONALLY rather
    // than only when a rung happens to widen. A certificate that does not carry
    // which of its five terms decided it can only be re-derived from source,
    // never audited; the same complaint this file's refusal messages make about
    // the fits they refuse. Fidelity and the exact rho identify whether a
    // screening verdict and the terminal mint measured the same candidate.
    // `derived_standard=false` is the actionable bit: it says this verdict rests
    // on a gradient-magnitude substitute because the resolvability form was
    // unavailable on this route, not because the problem called for it.
    log::info!(
        "[CERTIFICATE-BOUND] {context}: fidelity={fidelity:?}, rho={:?}, \
         bound {stationarity_bound:.6e} set by {} (derived_standard={}, \
         |Pg|={projected_grad_norm:.6e}, |g|={grad_norm:.6e}, \
         solver_band={solver_bound:.6e}, cost={:.6e})",
        result.rho.to_vec(),
        bound_source.label(),
        bound_source.is_derived_standard(),
        evaluation.cost,
    );

    // Large-step flatness certificate (#2299 fully-saturated smooth). After the
    // reproducibility floor a coordinate that has collapsed EXACTLY onto its
    // penalty null space (λ ~ 1e12, edf saturated) can still carry a projected
    // gradient component that is DETERMINISTIC cancellation bias from the
    // 1e12-conditioned logdet derivative (≈ ε·κ·scale). Being deterministic it
    // reproduces run to run, so the spread-keyed reproducibility floor above
    // cannot see it; and the Newton decrement anti-rescues, because the near-null
    // Hessian direction inflates gᵀH⁻¹g by design. The decisive question is
    // second-order-independent: does the criterion actually MOVE along that
    // coordinate at MACROSCOPIC scale? This block answers it directly — it probes
    // the objective a full e-fold in λ to either side of a near-null-curvature
    // coordinate and, for coordinates whose value is provably flat there, removes
    // their measured gradient component (numerical bias, not slope) from the
    // projected gradient before the bound test. A coordinate whose large-step
    // value MOVES is left untouched, so a genuine pseudologdet ramp still refuses.
    //
    // Gated as narrowly as possible: it runs only when the certificate would
    // OTHERWISE refuse on |Pg|, only with an analytic Hessian that is PSD-within-
    // noise in hand, and only probes coordinates whose curvature row is below the
    // roundoff floor — so a well-conditioned objective (every scripted mock at its
    // certification point) probes nothing and pays zero extra evaluations.
    // The coordinates railed at ±rho_bound (the infinite-smoothing ceiling). Their
    // saturated curvature direction makes the FULL Hessian indefinite, so the
    // flatness certificate below — and the final curvature gate — judge PSD on the
    // interior (un-railed) sub-block instead, or a rail-caused indefiniteness would
    // disable the very certificate that exists to certify a railed optimum (#2299).
    // The FACE the certificate reasons on: every θ coordinate against its own
    // bound, ρ and non-ρ alike. See `certificate_railed_coordinates` for why
    // the λ-block report cannot be used here.
    let certificate_railed = certificate_railed_coordinates(&result.rho, config);
    // The λ-block REPORT that ships on the certificate. Same predicate,
    // narrower scan, because `lambdas_railed` indexes smoothing parameters.
    let railed_lambda_block = certificate_railed_lambdas(&result.rho, layout.rho_dim(), config);

    // Typed stationary-at-asymptote rail certificate (#2348 Inc 1, #2299 layer 3,
    // #2337 Thm 2.1). Before falling through to the generic gradient/criterion-flat
    // verdict, POSITIVELY certify a railed optimum: the interior (non-railed)
    // coordinates are gradient-stationary, and each coordinate railed at the
    // infinite-/zero-smoothing box bound sits on a confirmed exponential tail whose
    // fitted model has already reached the rail limit to within the estimand
    // tolerance. This supersedes the untyped `lambdas_railed` flag with a proof that
    // the criterion improvement and coefficient travel still available by running to
    // the rail are both below tolerance.
    //
    // Computed in its own statement so the borrow of `analytic_hessian` ends before
    // the mint branch moves it onto the result. Gated to a genuinely railed optimum
    // with outward pull (`grad_norm` above the stationarity bound) and an analytic
    // Hessian: a well-conditioned interior fit, or a coordinate merely resting near a
    // bound with a vanishing gradient, probes nothing and keeps its ordinary verdict.
    let asymptote_objective_tol = config
        .rel_cost_tolerance
        .unwrap_or(config.tolerance * 1.0e-2)
        .max(COST_STALL_REL_TOL_FLOOR)
        * (1.0 + evaluation.cost.abs());
    let rail_outcome = match analytic_hessian.as_ref() {
        Some(hessian) if !certificate_railed.is_empty() && grad_norm > stationarity_bound => {
            Some(try_certify_asymptote_rail(
                obj,
                &AsymptoteRailInputs {
                    rho: &result.rho,
                    projected_gradient: &projected_gradient,
                    railed: &certificate_railed,
                    layout,
                    hessian,
                    bounds: &bounds,
                    terminal_beta: terminal_beta.as_ref(),
                    stationarity_bound: StationarityBound::from_ladder(stationarity_bound, bound_source),
                    objective_tol: asymptote_objective_tol,
                    context,
                },
            )?)
        }
        _ => None,
    };
    // A refused railed mint carries its typed decline reason into the final
    // refusal summary (mirroring the tail-snap decline note), so a railed
    // non-mint names the gate that refused instead of failing silently.
    let mut asymptote_rail_note: Option<String> = None;
    let mut probes_ran = rail_outcome.is_some();
    if let Some(outcome) = rail_outcome {
        match outcome {
            Err(reason) => asymptote_rail_note = Some(reason),
            Ok(minted) => {
                let (interior_projected_grad_norm, effective_interior_bound, rails) = minted;
                // The tail probes were derivative-bearing evaluations at probe
                // ρ's, so the EVALUATOR-side terminal-mode carrier now owns the
                // last probe, not the checkpoint (#2155 regression: every
                // custom-family at-point mint then failed the bitwise terminal
                // theta identity at fit assembly). Re-evaluate at the minted
                // point and ship ITS numbers as the terminal facts: the same
                // evaluation sets the evaluator carrier, so the optimizer
                // certificate and the owned mode are bitwise-identical by
                // construction. The certified stationarity facts (interior
                // norms, rails) remain the judged ones.
                let restored = obj
                    .eval_with_order(&result.rho, OuterEvalOrder::ValueAndGradient)
                    .map_err(|err| {
                        EstimationError::RemlOptimizationFailed(format!(
                            "{context}: failed to re-own the certified point after \
                             asymptote-rail probing: {err}"
                        ))
                    })?;
                result.final_value = restored.cost;
                let restored_projected = project_gradient_vector(
                    &result.rho,
                    &restored.gradient,
                    Some(&rail_projection_bounds),
                );
                result.final_grad_norm = Some(
                    restored_projected
                        .iter()
                        .map(|v| v * v)
                        .sum::<f64>()
                        .sqrt(),
                );
                result.final_gradient = Some(restored.gradient);
                let certificate = OuterCriterionCertificate {
                    stationarity: OuterStationarityCertificate::AsymptoteRail {
                        interior_projected_grad_norm,
                        // The bound that admitted the interior: the raw stationarity
                        // bound, or the curvature-scaled flat-valley widening when the
                        // interior sub-block's Newton decrement is below the loop's
                        // cost resolution (shared judgment with the Inc 2c mint).
                        bound: effective_interior_bound.value(),
                        rung: effective_interior_bound.rung().into(),
                        rails,
                    },
                    curvature: CurvatureEvidence::Measured { psd: true },
                    lambdas_railed: railed_lambda_block.clone(),
                    railed_facts: railed_coordinate_facts(
                        &result.rho,
                        &railed_lambda_block,
                        config,
                    ),
                    curvature_floor: None,
                };
                // Move the certified curvature onto the result; the mint path returns
                // immediately, so the fall-through below never observes the move.
                result.final_hessian = analytic_hessian;
                result.criterion_certificate = Some(certificate.clone());
                if !certificate.certifies() {
                    return Err(outer_nonconvergence_error(
                        context,
                        &certificate.summary(),
                        result,
                        Some(interior_projected_grad_norm),
                        StationarityBound::from_ladder(stationarity_bound, bound_source),
                    ));
                }
                result
                    .termination
                    .certify(OuterConvergedVia::AsymptoteStationary {
                        rails: certificate.stationarity.rails().len(),
                    });
                log::info!("[CERTIFICATE] {context}: {}", certificate.summary());
                return Ok(certificate);
            }
        }
    }

    let mut certified_projected_grad_norm = projected_grad_norm;
    if projected_grad_norm > stationarity_bound
        && let Some(hessian) = analytic_hessian.as_ref()
        && certificate_hessian_is_psd_off_railed(hessian, &certificate_railed) == Some(true)
    {
        let n = layout.n_params;
        // Curvature scale of the analytic outer Hessian: its dominant diagonal,
        // the same ‖H‖ scale `certificate_hessian_is_psd` and
        // `newton_predicted_decrease` regularize against. A coordinate's curvature
        // ROW is indistinguishable from the assembly's roundoff — it has no
        // curvature the arithmetic can resolve and has collapsed onto the penalty
        // null space — when its largest entry falls below the SAME √ε·‖H‖ margin
        // those two probes use to separate a real curvature direction from
        // O(ε·‖H‖) accumulation noise. This is the derivation of the threshold:
        // NULL_CURVATURE_REL = √ε (machine epsilon's square root, the assembled
        // Hessian's relative resolution), scaled by the Hessian's own max-diagonal
        // magnitude, floored at 1 exactly as the PSD/Newton shift is.
        let max_diag = (0..n).fold(0.0_f64, |acc, j| acc.max(hessian[[j, j]].abs()));
        let null_curvature_threshold = f64::EPSILON.sqrt() * max_diag.max(1.0);
        // The SAME relative cost floor the cost-stall guard and both widenings
        // above use: certification asserts nothing tighter about this surface's
        // macroscopic flatness than the loop already proved.
        let objective_tol = config
            .rel_cost_tolerance
            .unwrap_or(config.tolerance * 1.0e-2)
            .max(COST_STALL_REL_TOL_FLOOR)
            * (1.0 + evaluation.cost.abs());
        // One e-fold in log-λ per coordinate (ρ IS log-λ): the +δ/−δ pair spans e²
        // in λ, a macroscopic move across which no genuine descent slope can hide.
        const LARGE_STEP_DELTA: f64 = 1.0;
        let mut saturated_flat: Vec<usize> = Vec::new();
        let mut probe_reports: Vec<String> = Vec::new();
        let mut probed_any = false;
        let mut probe_failed = false;
        for k in 0..n {
            let row_inf = (0..n).fold(0.0_f64, |acc, j| acc.max(hessian[[k, j]].abs()));
            // Only near-null-curvature coordinates the measured gradient actually
            // loads on can be responsible for |Pg| exceeding the band; skip every
            // other coordinate, so no probe fires on a well-conditioned surface.
            if row_inf > null_curvature_threshold || projected_gradient[k] == 0.0 {
                continue;
            }
            let mut plus = result.rho.clone();
            plus[k] += LARGE_STEP_DELTA;
            let mut minus = result.rho.clone();
            minus[k] -= LARGE_STEP_DELTA;
            probed_any = true;
            let (Ok(cost_plus), Ok(cost_minus)) = (obj.eval_cost(&plus), obj.eval_cost(&minus))
            else {
                // A failed probe is not evidence of flatness — refuse to classify
                // (conservative) and leave |Pg| intact for the bound test.
                probe_failed = true;
                break;
            };
            if !cost_plus.is_finite() || !cost_minus.is_finite() {
                probe_failed = true;
                break;
            }
            let up = (cost_plus - evaluation.cost).abs();
            let down = (cost_minus - evaluation.cost).abs();
            if up <= objective_tol && down <= objective_tol {
                saturated_flat.push(k);
                probe_reports.push(format!("k={k} |ΔV|+={up:.3e} |ΔV|-={down:.3e}"));
            }
        }
        if !probe_failed && !saturated_flat.is_empty() {
            // Recompute |Pg| with the provably macroscopically-flat coordinates
            // removed: their measured gradient is deterministic cancellation bias,
            // not slope. Coordinates that moved keep their component and still count
            // against the bound.
            let reduced_sq = (0..n)
                .filter(|k| !saturated_flat.contains(k))
                .map(|k| projected_gradient[k] * projected_gradient[k])
                .sum::<f64>();
            certified_projected_grad_norm = reduced_sq.sqrt();
            let flat_list = saturated_flat
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(", ");
            let probe_summary = probe_reports.join("; ");
            log::info!(
                "[CERTIFICATE] {context}: large-step flatness certificate classified \
                 coordinate(s) [{flat_list}] saturated-flat (curvature row ≤ \
                 {null_curvature_threshold:.3e}, probed Δ=±{LARGE_STEP_DELTA} with \
                 {probe_summary}, cost-flat to tol {objective_tol:.3e}); projected \
                 gradient reduced from {projected_grad_norm:.3e} to \
                 {certified_projected_grad_norm:.3e}"
            );
        }
        // `eval_cost` warm-starts the inner solve, so the probes moved the objective
        // off the certified point. Restore it to ρ̂ once iff we actually probed, so
        // the downstream state (and the rho-uncertainty diagnostic) sees the fitted
        // point. A failure to re-evaluate the same ρ that certified moments ago is a
        // genuinely broken objective and refuses conservatively.
        if probed_any {
            obj.eval_cost(&result.rho).map_err(|err| {
                outer_nonconvergence_error(
                    context,
                    &format!(
                        "failed to restore the objective to the certified point after \
                         flatness probing: {err}"
                    ),
                    result,
                    Some(certified_projected_grad_norm),
                    StationarityBound::from_ladder(stationarity_bound, bound_source),
                )
            })?;
        }
    }

    let certificate = OuterCriterionCertificate {
        stationarity: OuterStationarityCertificate::AnalyticGradient {
            grad_norm,
            projected_grad_norm: certified_projected_grad_norm,
            bound: stationarity_bound,
            rung: bound_source.provenance().into(),
        },
        // The RAW measurement — unchanged, and what every consumer that asks
        // for a genuine PSD certificate keeps receiving.
        curvature: match analytic_hessian.as_ref() {
            Some(hessian) => CurvatureEvidence::from_measurement(
                certificate_hessian_is_psd_off_railed(hessian, &certificate_railed),
            ),
            // A screening pass deliberately declines the order-four ladder
            // (the documented design at `CertificationFidelity`); a Mint pass
            // reaching here simply has no analytic Hessian to test. Those were
            // the same `None` before #2561, which is why the design's own
            // promise — the winner's verdict is the one that mints — could not
            // be checked by anyone.
            None if matches!(fidelity, CertificationFidelity::Screening) => {
                CurvatureEvidence::NotSpent
            }
            None => CurvatureEvidence::NotAvailable,
        },
        lambdas_railed: railed_lambda_block.clone(),
        railed_facts: railed_coordinate_facts(&result.rho, &railed_lambda_block, config),
        // The floor's verdict on that same curvature, recorded beside it.
        curvature_floor: analytic_hessian.as_ref().and_then(|hessian| {
            interior_curvature_floor_clearance(hessian, &certificate_railed, &projected_gradient)
        }),
    };
    // Certify-time tail snap (#2348 Inc 2). About to refuse a point whose
    // residual gradient is carried by un-railed coordinates crawling a
    // CONFIRMED exponential tail toward the ρ-box (the one-e-fold-per-step
    // grind: the loop budget can exhaust strictly inside the box, where the
    // Inc 1 railed mint can never fire), snap those coordinates to their box
    // bound and publish that point as a one-shot optimization reseed. The
    // resumed optimizer lets coupled interior coordinates move before the
    // FULL Inc 1 rail discipline judges the result — this path grants nothing
    // by itself.
    let mut tail_snap_note: Option<String> = None;
    if allow_tail_snap
        && !certificate.certifies()
        && grad_norm > stationarity_bound
        && let Some(hessian) = analytic_hessian.as_ref()
    {
        probes_ran = true;
        match try_tail_snap_to_rail(
            obj,
            &AsymptoteRailInputs {
                rho: &result.rho,
                projected_gradient: &projected_gradient,
                railed: &certificate_railed,
                layout,
                hessian,
                bounds: &bounds,
                terminal_beta: terminal_beta.as_ref(),
                stationarity_bound: StationarityBound::from_ladder(stationarity_bound, bound_source),
                objective_tol: asymptote_objective_tol,
                context,
            },
        )? {
            TailSnapOutcome::TailStationaryAtPoint {
                rails,
                interior_projected_grad_norm,
                effective_interior_bound,
            } => {
                // #2348 Inc 2c: the confirmed tails extrapolate below the bound
                // AT the checkpoint — mint the typed asymptote certificate for
                // the point as it stands. `hessian_psd` is the interior
                // sub-block verdict established before probing (the full
                // matrix is expected non-PD from the noise-corrupted tail
                // entry); the stored bound is the one that actually certified
                // the interior (raw, or the sub-block curvature-scaled
                // flat-valley bound).
                //
                // Re-own the minted point first: the tail probes were
                // derivative-bearing evaluations at probe ρ's, so the
                // evaluator-side terminal-mode carrier owns the last probe —
                // shipping the pre-probe terminal numbers then fails the
                // bitwise terminal theta identity at custom-family fit
                // assembly (the #2155 all-links regression). One fresh
                // evaluation at the checkpoint sets the carrier AND supplies
                // the terminal facts, so both sides are bitwise-identical by
                // construction.
                let restored = obj
                    .eval_with_order(&result.rho, OuterEvalOrder::ValueAndGradient)
                    .map_err(|err| {
                        EstimationError::RemlOptimizationFailed(format!(
                            "{context}: failed to re-own the certified point after \
                             tail-snap probing: {err}"
                        ))
                    })?;
                result.final_value = restored.cost;
                let restored_projected = project_gradient_vector(
                    &result.rho,
                    &restored.gradient,
                    Some(&rail_projection_bounds),
                );
                result.final_grad_norm = Some(
                    restored_projected
                        .iter()
                        .map(|v| v * v)
                        .sum::<f64>()
                        .sqrt(),
                );
                result.final_gradient = Some(restored.gradient);
                let certificate = OuterCriterionCertificate {
                    stationarity: OuterStationarityCertificate::AsymptoteRail {
                        interior_projected_grad_norm,
                        bound: effective_interior_bound.value(),
                        rung: effective_interior_bound.rung().into(),
                        rails,
                    },
                    curvature: CurvatureEvidence::Measured { psd: true },
                    lambdas_railed: railed_lambda_block.clone(),
                    railed_facts: railed_coordinate_facts(
                        &result.rho,
                        &railed_lambda_block,
                        config,
                    ),
                    curvature_floor: None,
                };
                result.final_hessian = analytic_hessian;
                result.criterion_certificate = Some(certificate.clone());
                if !certificate.certifies() {
                    return Err(outer_nonconvergence_error(
                        context,
                        &certificate.summary(),
                        result,
                        Some(interior_projected_grad_norm),
                        effective_interior_bound,
                    ));
                }
                result
                    .termination
                    .certify(OuterConvergedVia::AsymptoteStationary {
                        rails: certificate.stationarity.rails().len(),
                    });
                log::info!(
                    "[CERTIFICATE] {context}: tail-stationary at the checkpoint \
                     (#2348 Inc 2c): {}",
                    certificate.summary()
                );
                return Ok(certificate);
            }
            TailSnapOutcome::ConfirmedNeedsReseed(snapped) => {
                log::info!(
                    "[CERTIFICATE] {context}: confirmed exponential tail on un-railed \
                     coordinate(s); publishing the snapped point {snapped} as a waypoint \
                     for one re-optimization retry (#2348 Inc 2b / #2358)"
                );
                tail_snap_note = Some(
                    "tail confirmed; retry seeded at the snapped rail waypoint".to_string(),
                );
                result.tail_snap_reseed = Some(snapped);
            }
            TailSnapOutcome::Declined(reason) => {
                tail_snap_note = Some(reason);
            }
        }
    }
    // Install the measured evidence before deciding its verdict.  A rejected
    // candidate is retained only as a resumable checkpoint, and that
    // checkpoint must carry the actual analytic residual/curvature evidence
    // that caused the rejection rather than the optimizer's stale terminal
    // status.
    result.final_hessian = analytic_hessian;
    result.criterion_certificate = Some(certificate.clone());
    if !certificate.certifies() {
        // Mint the #2392 reseeds fresh for THIS refused point: clear any value a
        // prior (multistart / pre-polish) certification of a different ρ left on
        // the result so the resume loop never consumes a stale pull-back/freeze.
        result.wrong_rail_reseed = None;
        result.active_set_reseed = None;
        // #2357 — saddle escape. The point is first-order stationary
        // (`is_stationary`: ‖Pg‖ ≤ bound) yet its INTERIOR reduced Hessian is a
        // certified strict saddle (`!curvature_admissible`). A railed coordinate
        // no longer waives this: `curvature_admissible` already reads the
        // off-railed reduced Hessian, and the escape descends only the free
        // (un-railed) directions while holding every rail fixed (#2155). That is
        // exactly the case a gradient-only
        // convergence gate mis-accepts: it arrived with the gradient already
        // below tolerance and stopped, leaving the certified negative-curvature
        // eigendirection — a strict descent — untaken. Mint a one-shot reseed
        // stepped off the ridge to a strictly-lower objective so the plan runner
        // can re-descend to the true PSD minimum, exactly as an identical
        // warm-started resume does by hand. Gated by `allow_tail_snap` (the same
        // one-shot reseed gate the tail snap rides) so the retry pass — which
        // runs with it `false` — can never recurse.
        if allow_tail_snap
            && certificate.is_stationary()
            && !certificate.curvature_not_refused()
            && let Some(hessian) = result.final_hessian.clone()
            && let Some(gradient) = result.final_gradient.clone()
        {
            let saddle_rho = result.rho.clone();
            let baseline_cost = result.final_value;
            // `curvature_admissible()` is `false` exactly when the REDUCED
            // (off-railed) Hessian is indefinite, so a railed coordinate no
            // longer waives the escape: it is passed through and held fixed while
            // the step descends the free-direction saddle (#2155). It must be
            // held fixed on the SAME face the reduction was taken on — the
            // certificate's λ-block report would leave a railed ψ free to be
            // stepped out of its box.
            result.saddle_escape_reseed = negative_curvature_escape_point(
                obj,
                &saddle_rho,
                &gradient,
                &hessian,
                &certificate_railed,
                baseline_cost,
                &bounds,
                context,
            );
        }
        // #2392 — wrong-rail pull-back and active-set reduction. A coordinate at
        // the ρ box whose deep-λ terminal gradient is instrument noise leaves the
        // outer search unable to move it: the trust region's local model is flat
        // there. Two evidence-gated one-shot reseeds recover the fit (both gated
        // by `allow_tail_snap` so the retry pass cannot recurse):
        //   (1) WRONG-RAIL PULL-BACK: the coordinate's clean-band probes (a few
        //       e-folds inside, above the noise floor) prove the objective
        //       DECREASES inward — it was driven to the wrong bound. Reseed it at
        //       its clean-band interior scale so the optimizer descends to the
        //       true interior optimum. Fires ONLY on the opposite-sign clean-tail
        //       proof, so a genuine λ→∞/λ→0 rail is never pulled off its bound.
        //   (2) ACTIVE-SET REDUCTION: no wrong rail, but the INTERIOR is not
        //       stationary while a rail is present — the railed coordinate's
        //       ill-conditioned Hessian row poisons the joint step. Freeze the
        //       KKT-ACTIVE rail(s) at their bound and re-run so the interior
        //       converges in the reduced space; the plan runner re-certifies the
        //       polished point under the ORIGINAL box, so the reduction can
        //       never redefine the feasible set.
        //
        //       "Active" is decided by the PROJECTOR, at freeze time, not by
        //       distance to a bound (#2454): a coordinate on its bound whose
        //       projected gradient survives still has feasible descent and is
        //       left free. This is where the un-freeze has to happen. Doing it
        //       after the reduced solve is not available — the retry is one-shot
        //       (`allow_tail_snap` is cleared on it, so no second reseed can be
        //       published) and the only path back off a frozen bound is the
        //       wrong-rail pull-back, which demands a clean opposite-sign
        //       exponential tail and declines on any coordinate that has none.
        // (1) takes precedence: a wrong rail must be pulled back, never frozen.
        if allow_tail_snap && !certificate_railed.is_empty() {
            let beta_norm = terminal_beta
                .as_ref()
                .map(|b| b.dot(b).sqrt())
                .filter(|v| v.is_finite())
                .unwrap_or(0.0);
            let mut rail_tol =
                AsymptoteTolerances::exp4_rail_bands(ASYMPTOTE_ESTIMAND_REL_TOL * (1.0 + beta_norm));
            rail_tol.tail_drift_rel = TAIL_SNAP_DRIFT_REL;
            let (lower, upper) = &bounds;
            let mut wrong_rail_point: Option<Array1<f64>> = None;
            for &k in certificate_railed.iter() {
                if k >= result.rho.len() || k >= lower.len() || k >= upper.len() {
                    continue;
                }
                let side = if (upper[k] - result.rho[k]).abs() <= (result.rho[k] - lower[k]).abs() {
                    AsymptoteSide::Upper
                } else {
                    AsymptoteSide::Lower
                };
                if let Some(target) = detect_wrong_rail_pullback(
                    obj,
                    &result.rho,
                    k,
                    side,
                    &rail_tol,
                    (lower[k], upper[k]),
                )? {
                    let mut reseed = result.rho.clone();
                    reseed[k] = target;
                    wrong_rail_point = Some(reseed);
                    break;
                }
            }
            if let Some(reseed) = wrong_rail_point {
                result.wrong_rail_reseed = Some(reseed);
            } else if let Some(hessian) = result.final_hessian.as_ref() {
                // Active-set reduction needs curvature to prove that the free
                // subspace is genuinely unpolished. Wrong-rail pull-back above
                // is a first-order tail-sign proof and deliberately does NOT:
                // gradient-only BFGS objectives can rail incorrectly too, and
                // withholding a valid pull-back merely because they do not
                // materialize H would make recovery depend on solver class.
                let interior_indices =
                    interior_face_indices(&projected_gradient, &certificate_railed);
                let interior_not_stationary = !interior_indices.is_empty()
                    && certify_interior_stationarity(
                        &projected_gradient,
                        &hessian,
                        &interior_indices,
                        StationarityBound::from_ladder(stationarity_bound, bound_source),
                        asymptote_objective_tol,
                    )
                    .is_err();
                if interior_not_stationary {
                    let mut froz_lower = lower.clone();
                    let mut froz_upper = upper.clone();
                    let mut reseed = result.rho.clone();
                    let mut froze_any = false;
                    for &k in certificate_railed.iter() {
                        if k >= reseed.len() {
                            continue;
                        }
                        // Freeze the KKT-ACTIVE set, not the proximity set
                        // (#2454). `certificate_railed` is a DISTANCE test —
                        // "within `CERTIFICATE_RAIL_MARGIN` of a bound" — and
                        // being near a bound says nothing about whether the
                        // constraint is active. The active set is the one the
                        // projector already decided: at a bound it keeps only
                        // the feasible-descent half, so a coordinate whose
                        // projected component survives is one the search can
                        // still move INWARD, and `interior_face_indices` (three
                        // statements up) has already classified it as interior
                        // for exactly that reason.
                        //
                        // Freezing it anyway pins the coordinate carrying the
                        // descent that made `interior_not_stationary` true in
                        // the first place, and the reduced solve then converges
                        // everything ELSE around it. Measured on #2454's Matérn
                        // monotone fixture, whose ψ seed sits on its box edge:
                        // ψ is frozen at −2.4849 with `∂V/∂ψ = −4.3164`
                        // (FD-confirmed), the three ρ converge to
                        // `‖g_ρ‖ = 1.03e-4`, the solver reports
                        // `claimed_converged=true` off that reduced norm, and
                        // the re-certification under the original box then
                        // refuses at `|Pg| = 4.316e0` against a `7.06e-4`
                        // bound — with ψ bit-identical to its seed after 26
                        // outer iterations, because it was never free to move.
                        //
                        // The un-freeze the reduction's contract promises ("a
                        // frozen coordinate whose gradient turns inward
                        // re-certifies under the ORIGINAL box") cannot rescue
                        // this: the gradient had ALREADY turned inward when the
                        // freeze was taken, so the reduced solve is asked to
                        // polish a face that was never active.
                        if projected_gradient
                            .get(k)
                            .is_some_and(|component| *component != 0.0)
                        {
                            log::info!(
                                "[ACTIVE-SET] {context}: coordinate {k} is within the rail \
                                 margin of its bound but its projected gradient is \
                                 {:.6e} (feasible descent remains), so it is INTERIOR and \
                                 stays free rather than being frozen (#2454)",
                                projected_gradient[k],
                            );
                            continue;
                        }
                        let rail = if (upper[k] - reseed[k]).abs() <= (reseed[k] - lower[k]).abs() {
                            upper[k]
                        } else {
                            lower[k]
                        };
                        reseed[k] = rail;
                        froz_lower[k] = rail;
                        froz_upper[k] = rail;
                        froze_any = true;
                    }
                    if froze_any {
                        result.active_set_reseed = Some(ActiveSetReseed {
                            rho: reseed,
                            bounds: (froz_lower, froz_upper),
                        });
                    }
                }
            }
        }
        // Carry the railed-mint and tail-snap decline evidence into the
        // refusal so a railed or budget-exhausted crawl explains which
        // certificate gate refused instead of failing silently.
        let mut summary = certificate.summary();
        // `summary()` prints `lambdas_railed`, which is the λ-block report. When
        // the face the certificate actually reasoned on is wider — a joint
        // [ρ, ψ] search with a κ coordinate on its data-derived window — say so,
        // or the refusal reads as `railed=[]` while the projector has already
        // discarded that coordinate's outward pull and nothing explains where
        // `|g|` went (#979).
        if certificate_railed.len() != railed_lambda_block.len() {
            summary = format!(
                "{summary}; outer coordinates railed (theta-wide, incl. non-rho blocks): \
                 {certificate_railed:?}"
            );
        }
        // #2465: `railed=[…]` without its box is unfalsifiable from the run
        // record. Every value here is live at the emission site.
        if !certificate_railed.is_empty() {
            summary = format!(
                "{summary}; rail tests: [{}]",
                rail_test_summary(&result.rho, &certificate_railed, config)
            );
        }
        // #2465 again, one level up: the `solver provenance` this refusal is
        // about to append reports the terminating `|g|` of the run that produced
        // `result`. When that run executed under an active-set reduction, its
        // box PINNED some coordinates (`lower == upper`) and its `|g|` therefore
        // ranges over the FREE ones only, while `|Pg|` above ranges over all of
        // θ under the model domain. The two are then not the same quantity, and
        // nothing in the string said so: on #2454's Matérn fixture the refusal
        // read `claimed_converged=true, gradient_tolerance(|g|=1.029235e-4 <
        // 5.487011e-4)` beside `|Pg|=4.316e0`, a four-order gap that is entirely
        // the pinned ψ coordinate and looks like a contradiction until the two
        // coordinate sets are named. Name them.
        if let Some((search_lower, search_upper)) = config.search_bounds_override.as_ref() {
            let pinned: Vec<usize> = (0..search_lower.len().min(search_upper.len()))
                .filter(|&k| search_lower[k] == search_upper[k])
                .collect();
            if !pinned.is_empty() {
                summary = format!(
                    "{summary}; NOTE the run that produced this point searched a REDUCED box \
                     with coordinate(s) {pinned:?} pinned (active-set reduction), so the \
                     solver-provenance |g| below ranges over the FREE coordinates only and is \
                     NOT comparable with the |Pg| above, which ranges over all of theta under \
                     the model domain"
                );
            }
        }
        if let Some(note) = asymptote_rail_note {
            summary = format!("{summary}; asymptote-rail declined: {note}");
        }
        let summary = match tail_snap_note {
            Some(note) => format!("{summary}; tail-snap declined: {note}"),
            None => summary,
        };
        return Err(outer_nonconvergence_error(
            context,
            &summary,
            result,
            Some(certified_projected_grad_norm),
            StationarityBound::from_ladder(stationarity_bound, bound_source),
        ));
    }

    // #2155 regression, the LAST carrier-stealing path: the rail-mint and
    // tail-snap attempts probe with derivative-bearing evaluations, and the
    // ORDINARY certificate can still certify after a declined attempt (e.g. a
    // KKT-railed projection whose raw gradient norm sits above the bound), so
    // this success would ship pre-probe terminal numbers while the evaluator's
    // terminal-mode carrier owns the last probe — refusing the bitwise theta
    // identity at custom-family fit assembly. Re-own the certified point with
    // one fresh evaluation and ship ITS numbers; the mint branches re-own for
    // themselves before their early returns, and the judged stationarity facts
    // above remain the measured pre-probe ones.
    if probes_ran {
        let restored = obj
            .eval_with_order(&result.rho, OuterEvalOrder::ValueAndGradient)
            .map_err(|err| {
                EstimationError::RemlOptimizationFailed(format!(
                    "{context}: failed to re-own the certified point after                      rail/tail probing: {err}"
                ))
            })?;
        result.final_value = restored.cost;
        let restored_projected = project_gradient_vector(
            &result.rho,
            &restored.gradient,
            Some(&rail_projection_bounds),
        );
        result.final_grad_norm = Some(
            restored_projected
                .iter()
                .map(|v| v * v)
                .sum::<f64>()
                .sqrt(),
        );
        result.final_gradient = Some(restored.gradient);
    }
    // #2235/#2241 — record WHICH certificate concluded this run. A
    // Fellner–Schall model-state fixed point was pre-stamped by the runner and
    // is preserved (this analytic certificate is its corroborating evidence);
    // otherwise the verdict is decided by which stationarity band the measured
    // projected gradient actually cleared: the solver's own tolerance
    // (gradient-stationary) or only the widened flat certificate band
    // (criterion-flat, #2241).
    let via = match result.termination.proposed_via() {
        Some(via @ OuterConvergedVia::RecurrentIncumbent { .. }) => via,
        _ if certified_projected_grad_norm <= solver_bound => {
            OuterConvergedVia::GradientStationary
        }
        _ => OuterConvergedVia::CriterionFlat {
            residual_grad_norm: certified_projected_grad_norm,
            certificate_bound: stationarity_bound,
        },
    };
    result.termination.certify(via);
    log::info!("[CERTIFICATE] {context}: {}", certificate.summary());
    Ok(certificate)
}

/// Estimand tolerance relative to the fitted coefficient scale for the
/// asymptote-rail certificate (#2348 Inc 1): the remaining coefficient travel
/// to the rail limit must fall below `ASYMPTOTE_ESTIMAND_REL_TOL·(1 + ‖β‖)` for
/// the fitted model to be certified equal to the rail-limit fit.
const ASYMPTOTE_ESTIMAND_REL_TOL: f64 = 1.0e-4;

/// Number of one-e-fold-in-`ρ` probes stepped back from a railed coordinate
/// toward the interior when reconstructing its exponential tail (#2348 Inc 1).
/// Enough to span both the finite-difference floor next to the rail (rejected)
/// and a confirmable-tail run further in.
// 18 e-folds: the window must REACH the finite-difference-clean constant-ĉ
// band from a coordinate railed AT the box ceiling. The fused-Hessian
// trajectory (#2348) rails fits at ρ=30 that previously stalled mid-box, and
// the #2299 fixture's clean band sits 13–16 e-folds inside — the old 12-probe
// window (sized for mid-box crawls) stopped one row short of it, so a fully
// confirmed tail declined with "no finite-difference-clean tail window". Six
// extra value+gradient evals, paid only at certification of railed fits.
const ASYMPTOTE_PROBE_COUNT: usize = 18;

/// Local confirmation resolution used only when the one-e-fold ladder cannot
/// find a clean run. A finite smoothing box can intersect a perfectly regular
/// asymptote before three whole e-folds of the leading-order tail are visible;
/// half-e-fold probes resolve that band without relaxing any certificate gate.
///
/// The probes remain equally spaced, which is load-bearing: the estimand
/// certificate interprets consecutive coefficient moves as one geometric
/// sequence. Six samples cover three e-folds, so the fallback still observes
/// curvature over a material interval instead of manufacturing constancy from
/// an arbitrarily small neighborhood.
const ASYMPTOTE_LOCAL_PROBE_DELTA: f64 = 0.5;
const ASYMPTOTE_LOCAL_PROBE_COUNT: usize = 6;

/// Read-only inputs to [`try_certify_asymptote_rail`], bundled so the certify
/// path passes one borrow rather than a long positional argument list.
struct AsymptoteRailInputs<'a> {
    rho: &'a Array1<f64>,
    projected_gradient: &'a Array1<f64>,
    railed: &'a [usize],
    /// Which slots of `rho` carry `log λ`. The active-set half of the
    /// certificate reads `railed` as-is; the tail law reads it through
    /// [`tail_law_coordinates`], which is the whole difference.
    layout: OuterThetaLayout,
    hessian: &'a Array2<f64>,
    bounds: &'a (Array1<f64>, Array1<f64>),
    terminal_beta: Option<&'a Array1<f64>>,
    /// The ladder bound the full problem was judged against, carrying the rung
    /// that set it. A gradient-magnitude rung is already in the exact projected
    /// residual currency used on the face. A curvature-derived rung is not:
    /// the interior judgment must REPLACE it with its own sub-block curvature
    /// bound, and the replacement's rung is what the resulting certificate —
    /// or refusal — must report (#2458/#2559).
    stationarity_bound: StationarityBound,
    /// The run's relative objective tolerance resolved at the certified cost —
    /// the same flat-valley floor the cost-stall guard and the curvature-scaled
    /// widening use. The tail-snap interior judgment applies the identical
    /// Newton-decrement criterion on the interior SUB-BLOCK (the full-Hessian
    /// widening is disabled exactly when a noise-corrupted tail entry makes the
    /// full matrix non-PD).
    objective_tol: f64,
    context: &'a str,
}

impl AsymptoteRailInputs<'_> {
    /// Split the railed set into the coordinates the exponential tail law
    /// speaks for and the ones it does not (#2453).
    ///
    /// Both halves stay in the active set: they are equally deleted from the
    /// interior gradient and the interior Hessian sub-block, because that
    /// reasoning is about a bound and not about a quantity. Only the first
    /// half may be *certified* by — or *required* to produce — a tail, since
    /// [`OuterThetaLayout::coordinate_is_log_smoothing`] is what makes
    /// `ĉ = ∓e^{±ρ}·∂V/∂ρ` a theorem rather than an arithmetic accident.
    ///
    /// The second half needs no tail: its bound is attainable, so the
    /// outward-gradient complementarity the projector already enforces is a
    /// complete first-order certificate there.
    fn tail_law_coordinates(&self) -> (Vec<usize>, Vec<usize>) {
        self.railed
            .iter()
            .copied()
            .partition(|&k| self.layout.coordinate_is_log_smoothing(k))
    }
}

/// Attempt the typed stationary-at-asymptote rail certificate (#2348 Inc 1).
///
/// Returns `Some((interior_projected_grad_norm, rails))` when the interior
/// (non-railed) coordinates are gradient-stationary, the interior Hessian
/// sub-block is PSD, and EVERY railed coordinate is certified on a confirmed
/// exponential tail whose fitted model has reached the rail limit to within the
/// estimand tolerance. Returns `None` (fall through to the generic verdict) on
/// any failure — a non-stationary interior, indefinite interior curvature, or
/// any railed coordinate whose tail is not confirmable. Never errors on a
/// refusal; the only `Err` is a genuinely broken objective that cannot restore
/// its inner state to the certified point after probing.
fn try_certify_asymptote_rail(
    obj: &mut dyn OuterObjective,
    inputs: &AsymptoteRailInputs<'_>,
) -> Result<Result<(f64, StationarityBound, Vec<RailCoordinate>), String>, EstimationError> {
    let rho = inputs.rho;
    let projected_gradient = inputs.projected_gradient;
    let railed = inputs.railed;
    // The interior (non-railed) coordinates must be stationary in their own
    // right: the asymptote certificate speaks only to the railed directions,
    // never rescues a still-descending interior. Judged by the SAME two-stage
    // criterion as the Inc 2c at-point mint: a gradient-magnitude bound may
    // judge the exact KKT-projected residual directly; a curvature-derived
    // bound is reminted from the interior sub-block before it may admit
    // anything. A fit whose remaining interior Newton step would improve the
    // cost by less than the loop's own cost resolution is at its interior
    // optimum, and the residual gradient is the deep-λ instrument noise floor
    // (evaluations beside a saturated rail share the rail's logdet noise).
    let interior_indices = interior_face_indices(projected_gradient, railed);
    let (interior_projected_grad_norm, effective_interior_bound) =
        match certify_interior_stationarity(
            projected_gradient,
            inputs.hessian,
            &interior_indices,
            inputs.stationarity_bound,
            inputs.objective_tol,
        ) {
            Ok(certified) => certified,
            Err(reason) => return Ok(Err(reason)),
        };
    // The interior sub-block (railed coordinates removed) must be admissible
    // curvature for a minimum. A rail-caused indefiniteness in the saturated
    // direction is expected and excluded; genuine interior negative curvature is
    // not, and refuses the certificate.
    if certificate_hessian_is_psd_off_railed_above_gradient_floor(
        inputs.hessian,
        railed,
        projected_gradient,
    ) != Some(true)
    {
        return Ok(Err("interior Hessian sub-block is not PSD".to_string()));
    }
    let beta_norm = inputs
        .terminal_beta
        .map(|b| b.dot(b).sqrt())
        .filter(|v| v.is_finite())
        .unwrap_or(0.0);
    let estimand_tol = ASYMPTOTE_ESTIMAND_REL_TOL * (1.0 + beta_norm);
    let mut tol = AsymptoteTolerances::exp4_rail_bands(estimand_tol);
    // Real REML tails hold ĉ to ~5e-3 relative, not the exp4 synthetic
    // characterization's 1e-3 (measured on the #2299 fixture during Inc 2c);
    // the tail-snap path already certifies against the widened band, and the
    // railed mint must judge the SAME physical tail by the same standard.
    tol.tail_drift_rel = TAIL_SNAP_DRIFT_REL;
    let (lower, upper) = inputs.bounds;

    // #2453: only the log-λ coordinates go to the tail law. A ψ rail stays in
    // the active set above (it is excluded from the interior gradient and the
    // interior sub-block just like any other bound-active coordinate) but is
    // certified by complementarity, not by an asymptote it does not have.
    let (tail_railed, box_railed) = inputs.tail_law_coordinates();
    if tail_railed.is_empty() {
        return Ok(Err(format!(
            "no railed coordinate parameterizes log λ; {} bound-active ψ coordinate(s) {:?} carry \
             no exponential tail to certify",
            box_railed.len(),
            box_railed,
        )));
    }
    if !box_railed.is_empty() {
        log::info!(
            "[CERTIFICATE] {}: {} bound-active ψ coordinate(s) {:?} held out of the tail law \
             (their box endpoints are attainable, so complementarity certifies them); tail law \
             runs on log-λ coordinate(s) {:?}",
            inputs.context,
            box_railed.len(),
            box_railed,
            tail_railed,
        );
    }

    // #2348 Inc 5 — the ANALYTIC face proof, first resort.
    //
    // Measuring a tail beside the ρ box asks the criterion for derivative
    // information exactly where its logdet pair cancels; and even when it
    // succeeds it speaks for ONE coordinate along ONE ray. The analytic route
    // forms the λ=∞ limit itself — the null-space-restricted fit, and the
    // exact first-order form of the logdet and trace terms there — and its
    // positive definiteness proves the whole face against every way of coming
    // off it. When the objective cannot form that limit, or the proof does not
    // hold, the measured-tail path below is unchanged.
    match try_certify_face_analytically(obj, inputs, &tail_railed, estimand_tol)? {
        Ok((rails, proof)) => {
            log::info!(
                "[CERTIFICATE] {}: analytic λ=∞ face proof on {} coordinate(s): λ_min(C)={:.6e} \
                 > margin {:.3e}, joint pencil ĉ={:.6e}, remaining value gap {:.3e}, estimand \
                 travel {:.3e}",
                inputs.context,
                rails.len(),
                proof.min_curvature,
                proof.curvature_margin,
                proof.joint_tail_constant,
                proof.value_gap,
                proof.estimand_travel,
            );
            return Ok(Ok((
                interior_projected_grad_norm,
                effective_interior_bound,
                rails,
            )));
        }
        Err(reason) => log::info!(
            "[CERTIFICATE] {}: analytic λ=∞ face proof declined ({reason}); measuring the tail \
             instead",
            inputs.context,
        ),
    }

    let mut rails: Vec<RailCoordinate> = Vec::new();
    let mut decline: Option<String> = None;
    let mut probed_any = false;
    for &k in tail_railed.iter() {
        if k >= rho.len() || k >= lower.len() || k >= upper.len() {
            decline = Some(format!("railed coordinate {k} outside the box layout"));
            break;
        }
        // Which rail: the box endpoint the coordinate sits nearest. `Upper`
        // (λ → ∞) probes step ρ downward into the tail; `Lower` (λ → 0) step up.
        let side = if (upper[k] - rho[k]).abs() <= (rho[k] - lower[k]).abs() {
            AsymptoteSide::Upper
        } else {
            AsymptoteSide::Lower
        };
        probed_any = true;
        match build_and_assess_rail_coordinate(obj, rho, k, side, &tol, (lower[k], upper[k]))? {
            Ok(rail) => rails.push(rail),
            Err(reason) => {
                decline = Some(reason);
                break;
            }
        }
    }

    // The probes warm-started the inner solve away from ρ̂; restore it so the
    // shipped fitted state (and the ρ-uncertainty diagnostic) sees the certified
    // point. A failure here is a genuinely broken objective, not a refusal.
    if probed_any {
        obj.eval_cost(rho).map_err(|err| {
            EstimationError::RemlOptimizationFailed(format!(
                "{}: failed to restore the objective to the certified point after \
                 asymptote-rail probing: {err}",
                inputs.context
            ))
        })?;
    }

    if let Some(reason) = decline {
        return Ok(Err(reason));
    }
    if rails.is_empty() {
        return Ok(Err(
            "no railed coordinate produced a certifiable tail".to_string()
        ));
    }
    Ok(Ok((
        interior_projected_grad_norm,
        effective_interior_bound,
        rails,
    )))
}

/// The analytic face law is falsified against production criterion VALUES, so
/// the admissible discrepancy is exactly the error sources the comparison is
/// made of: the run's own cost resolution (absolute, once per evaluation), the
/// law's `O(e^{−ρ})` second-order remainder (relative), and the digits the
/// Gram/Schur assembly loses. `FACE_LAW_ERROR_SLACK` widens that budget by the
/// factor a two-point difference accumulates — each evaluation carries its own
/// resolution error and the remainder enters both the predicted and the
/// measured side — so a CORRECT law is never refused by its own error bars,
/// while a wrong one (which misses by orders of magnitude, not by slack) still
/// is. The falsification only runs where the budget leaves a discriminating
/// band; where it does not, the analytic route declines rather than minting
/// unfalsifiable evidence.
const FACE_LAW_ERROR_SLACK: f64 = 4.0;

/// Floor on the falsification band.
///
/// The derived numerical budget can land many orders below the honest fidelity
/// of a closed form that deliberately does not model the criterion's
/// stabilization ridge or its reparameterization; holding the law to that
/// budget would reject a CORRECT law over a difference that changes no
/// decision. What the falsification exists to catch is a structurally wrong law
/// — a missing Schur term, the wrong dispersion convention, a sign error — and
/// those miss by orders of magnitude, not by slack (measured on a fixture whose
/// released directions genuinely earned their cost: 100%). Half the predicted
/// drop is the coarsest band that still separates those two worlds.
const FACE_LAW_ORDER_BAND: f64 = 0.5;

/// Prove the rail face analytically (#2348 Inc 5): ask the objective for the
/// exact λ→∞ limit, test the first-order form, and falsify the resulting law
/// against the production criterion before minting anything from it.
///
/// Returns the minted rail coordinates plus the proof, or a human-readable
/// decline that the caller logs before falling back to the measured tail.
fn try_certify_face_analytically(
    obj: &mut dyn OuterObjective,
    inputs: &AsymptoteRailInputs<'_>,
    tail_railed: &[usize],
    estimand_tol: f64,
) -> Result<Result<(Vec<RailCoordinate>, RailFaceProof), String>, EstimationError> {
    let rho = inputs.rho;
    let (lower, upper) = inputs.bounds;
    // The analytic limit is the INFINITE-smoothing face. A coordinate railed at
    // the zero-smoothing bound is the opposite limit (the penalty leaves the
    // model rather than pinning it) and belongs to the measured-tail path.
    for &k in tail_railed.iter() {
        if k >= rho.len() || k >= lower.len() || k >= upper.len() {
            return Ok(Err("railed coordinate outside the box layout".to_string()));
        }
        if (upper[k] - rho[k]).abs() > (rho[k] - lower[k]).abs() {
            return Ok(Err(format!(
                "coordinate {k} rails at the zero-smoothing bound; the analytic face limit \
                 covers λ→∞ only"
            )));
        }
    }
    let limit = match obj.rail_face_limit(rho, tail_railed)? {
        RailFaceLimitOutcome::Available(limit) => *limit,
        // The decline is typed, and the distinction is worth carrying into the
        // refusal: "outside this closed form" invites a different one, while
        // "the face is unavailable" is a statement about the face.
        RailFaceLimitOutcome::OutsideClosedForm { reason } => {
            return Ok(Err(format!("outside the analytic closed form: {reason}")));
        }
        RailFaceLimitOutcome::FaceUnavailable { reason } => {
            return Ok(Err(format!("the λ=∞ face is unavailable: {reason}")));
        }
    };
    let proof = match certify_rail_face(&limit) {
        RailFaceVerdict::Certified(proof) => proof,
        RailFaceVerdict::Refused { reason } => return Ok(Err(reason)),
    };
    // The face being optimal does not by itself mean the SHIPPED fit is the
    // limit fit; the estimand gate is the same one the measured path applies,
    // now answered by the exact first-order coefficient offset rather than a
    // geometric extrapolation of observed steps.
    if !(proof.estimand_travel <= estimand_tol) {
        return Ok(Err(format!(
            "λ=∞ face proven, but the shipped fit has not reached it: coefficient travel \
             {:.3e} > estimand tolerance {estimand_tol:.3e}",
            proof.estimand_travel
        )));
    }
    if let Err(reason) = falsify_face_law(obj, inputs, &limit, &proof)? {
        return Ok(Err(reason));
    }
    let rails: Vec<RailCoordinate> = limit
        .face
        .iter()
        .zip(limit.face_rho.iter())
        .zip(proof.tail_constants.iter())
        .map(|((&index, &rho_k), &tail_constant)| RailCoordinate {
            index,
            side: AsymptoteSide::Upper,
            tail_constant,
            // On the tail law the remaining value gap of one coordinate is
            // exactly `c_k·e^{−ρ_k}`; a coordinate the rest of the face has
            // already pinned reports `c_k = 0` because releasing it does not
            // move the criterion at all.
            value_gap: tail_constant * (-rho_k).exp(),
            estimand_travel_bound: proof.estimand_travel,
            // The face was PROVEN, so the standard it cleared is the form's own
            // eigen-backward-error margin — not a finite-difference floor, and
            // not a quantity comparable with one.
            evidence: RailTailEvidence::AnalyticFaceProof {
                min_curvature: proof.min_curvature,
                curvature_margin: proof.curvature_margin,
            },
        })
        .collect();
    Ok(Ok((rails, proof)))
}

/// Falsify the analytic face law against the production criterion.
///
/// The law predicts `V(ρ) − V_∞ = ½tr((Σλ_kQᵀS_kQ)⁻¹C)`, so pulling every face
/// coordinate back by `Δ` must raise the criterion by exactly
/// `gap·(e^{Δ} − 1)`. That is a VALUE comparison — no derivative, no
/// cancellation — and it costs two evaluations instead of a probe ladder.
///
/// `Δ` is not a knob: the measurement error is `resolution/(gap·(e^Δ−1))` and
/// the law's own remainder is `O(e^{Δ−ρ})`, so their sum is minimized at
/// `Δ* = ½(ln(resolution/gap) + ρ)`, where both equal `√(resolution·e^{−ρ}/gap)`.
fn falsify_face_law(
    obj: &mut dyn OuterObjective,
    inputs: &AsymptoteRailInputs<'_>,
    limit: &RailFaceLimit,
    proof: &RailFaceProof,
) -> Result<Result<(), String>, EstimationError> {
    const FACE_LAW_DOMAIN_MARGIN: f64 = 1.0e-6;
    let rho = inputs.rho;
    let (lower, _) = inputs.bounds;
    let gap = proof.value_gap;
    let resolution = inputs.objective_tol;
    if !(gap > 0.0) || !(resolution > 0.0) {
        return Ok(Err(format!(
            "the face law carries no resolvable value gap: gap={gap:.3e}, cost resolution \
             {resolution:.3e}"
        )));
    }
    let deepest = limit
        .face_rho
        .iter()
        .fold(f64::INFINITY, |acc, v| acc.min(*v));
    let ideal = 0.5 * ((resolution / gap).ln() + deepest);
    let room = limit
        .face
        .iter()
        .zip(limit.face_rho.iter())
        .map(|(&k, &rho_k)| rho_k - lower[k] - FACE_LAW_DOMAIN_MARGIN)
        .fold(f64::INFINITY, f64::min);
    let delta = ideal.min(room);
    // One e-fold is the natural unit of the law being tested; below that the
    // predicted change is not a statement about a tail.
    if !(delta >= 1.0) || !delta.is_finite() {
        return Ok(Err(format!(
            "no room inside the box to falsify the face law: Δ={delta:.3e} e-folds"
        )));
    }
    let predicted = gap * (delta.exp() - 1.0);
    let measurement_error = resolution / predicted;
    let remainder_error = (delta - deepest).exp();
    let assembly_error = f64::EPSILON.sqrt() * limit.form_conditioning.sqrt();
    let budget = measurement_error + remainder_error + assembly_error;
    let admissible = (FACE_LAW_ERROR_SLACK * budget).max(FACE_LAW_ORDER_BAND);
    if !(admissible < 1.0) {
        return Ok(Err(format!(
            "the face law cannot be falsified here: error budget {budget:.3e} (measurement \
             {measurement_error:.3e}, remainder {remainder_error:.3e}, assembly \
             {assembly_error:.3e}) leaves no discriminating band"
        )));
    }
    let baseline = obj.eval_cost(rho)?;
    let mut pulled_back = rho.clone();
    for &k in limit.face.iter() {
        pulled_back[k] -= delta;
    }
    let pulled_value = obj.eval_cost(&pulled_back)?;
    // Every exit below ships the certified point, so restore it before judging.
    obj.eval_cost(rho)?;
    if !baseline.is_finite() || !pulled_value.is_finite() {
        return Ok(Err(
            "the criterion is not finite at the falsification points".to_string()
        ));
    }
    let measured = pulled_value - baseline;
    let discrepancy = (predicted - measured).abs() / predicted;
    if discrepancy > admissible {
        return Ok(Err(format!(
            "the analytic face law does not reproduce the criterion: pulling the face back \
             {delta:.2} e-folds should raise V by {predicted:.6e}, measured {measured:.6e} \
             (relative {discrepancy:.3e} > admissible {admissible:.3e})"
        )));
    }
    Ok(Ok(()))
}

/// The coordinates a stationarity residual must still account for.
///
/// A coordinate leaves the interior only when the box has genuinely pinned it:
/// railed AND the projection zeroed its gradient, i.e. its entire pull was the
/// infeasible KKT multiplier. [`project_gradient_vector`] keeps a near-bound
/// coordinate's INWARD component on purpose — that component is feasible
/// descent — so dropping the row for *every* railed coordinate discards exactly
/// what the projector was written to preserve.
///
/// Deleting agrees with this only when every railed gradient points strictly
/// outward. `matern_nu_sweep_uniform_quality_on_sin1` is the counterexample
/// (#2471): coordinate 3 was reported railed while `1.2018e1` of its projected
/// gradient survived — 66.7x the stationarity bound, and 99.99% of the reported
/// `|Pg|`. With it deleted the interior norm read `1.986e-1`, i.e. 1.10x the
/// bound, which reads as "essentially converged" at a point still carrying that
/// much feasible descent. The certificate refused anyway, so the number misled
/// the reader rather than the verdict — but the ledger built on it classified a
/// genuine non-convergence as a railed-coordinate accounting artifact.
///
/// Since the projection either keeps a component unchanged or zeroes it, the
/// norm over these indices is exactly `‖Pg‖`. Where this differs from deleting
/// it includes MORE residual coordinates. Curvature evidence is deliberately
/// recomputed on that exact set: unlike the residual norm, a Newton decrement
/// is not transferable across Hessian subspaces or their regularization scales.
pub(crate) fn interior_face_indices(
    projected_gradient: &Array1<f64>,
    railed: &[usize],
) -> Vec<usize> {
    (0..projected_gradient.len())
        .filter(|k| !railed.contains(k) || projected_gradient[*k] != 0.0)
        .collect()
}

/// Interior stationarity judgment shared by the Inc 1 railed mint and the
/// Inc 2c at-point mint (#2348/#2559).
///
/// The supplied indices make `interior_grad_norm` the exact KKT-projected
/// residual of the face being judged: normally [`interior_face_indices`]
/// itself, and with already-proven tail coordinates removed on the Inc 2c
/// route. A gradient-magnitude ladder rung can therefore judge it directly.
/// [`StationarityBoundSource::CurvatureResolvability`] is different: its value
/// contains the caller's Hessian and Newton decrement, so reusing it here would
/// make the ordinary rail path's early comparison reduce to the caller's own
/// `Δpred <= objective_tol` test. It must instead be derived from `sub_h` and
/// `sub_g`. Exact zero is stationary without a curvature scale.
///
/// The face-local curvature path certifies only when the active-face Newton
/// step would improve the objective by at most `objective_tol` — the loop's
/// own cost resolution. Returns the bound that actually admitted the norm;
/// `Err` carries both caller and face-local evidence when descent remains.
pub(crate) fn certify_interior_stationarity(
    gradient: &Array1<f64>,
    hessian: &Array2<f64>,
    interior_indices: &[usize],
    stationarity_bound: StationarityBound,
    objective_tol: f64,
) -> Result<(f64, StationarityBound), String> {
    let interior_grad_norm = interior_indices
        .iter()
        .map(|&k| gradient[k] * gradient[k])
        .sum::<f64>()
        .sqrt();
    if interior_grad_norm <= stationarity_bound.value()
        && (interior_grad_norm == 0.0
            || !stationarity_bound.requires_face_local_derivation())
    {
        return Ok((interior_grad_norm, stationarity_bound));
    }
    let m = interior_indices.len();
    let mut sub_h = Array2::<f64>::zeros((m, m));
    let mut sub_g = Array1::<f64>::zeros(m);
    for (i, &ri) in interior_indices.iter().enumerate() {
        sub_g[i] = gradient[ri];
        for (j, &rj) in interior_indices.iter().enumerate() {
            sub_h[[i, j]] = hessian[[ri, rj]];
        }
    }
    match newton_predicted_decrease(&sub_h, &sub_g) {
        Some(predicted_decrease) if predicted_decrease.is_finite() && predicted_decrease > 0.0 => {
            if predicted_decrease <= objective_tol {
                let curvature_grad_bound =
                    interior_grad_norm * (objective_tol / predicted_decrease).sqrt();
                if curvature_grad_bound.is_finite() && curvature_grad_bound >= interior_grad_norm {
                    // The returned bound is no longer the caller's: it is the
                    // interior sub-block's own `|Pg_int|·√(τ/Δpred)`. Carrying
                    // the caller's rung with it would report a widened bound
                    // under the standard that did NOT set it (#2458).
                    return Ok((
                        interior_grad_norm,
                        StationarityBound::from_ladder(
                            curvature_grad_bound,
                            StationarityBoundSource::CurvatureResolvability,
                        ),
                    ));
                }
            }
            Err(format!(
                "interior not stationary: active-face |Pg|={interior_grad_norm:.3e}, \
                 caller bound {:.3e} from {}; active-face Newton decrement \
                 {predicted_decrease:.3e} > cost resolution {objective_tol:.3e}",
                stationarity_bound.value(),
                stationarity_bound.rung().label,
            ))
        }
        _ => Err(format!(
            "interior not stationary: active-face |Pg|={interior_grad_norm:.3e}, \
             caller bound {:.3e} from {}; the active-face Hessian and residual \
             yield no positive finite PD Newton decrement",
            stationarity_bound.value(),
            stationarity_bound.rung().label,
        )),
    }
}

/// Curvature-tie acceptance band for a certify-time tail-snap candidate
/// (#2348 Inc 2). On the #2337 Thm 2.1 exponential tail `V = V_∞ + c·e^{∓ρ}`
/// the coordinate's own curvature equals its gradient magnitude EXACTLY
/// (`H_kk = c·e^{∓ρ} = |g_k|`, unit decay rate in ρ = log λ), so `H_kk/|g_k| ≈ 1`
/// is a zero-cost analytic signature separating a live tail crawl from a
/// genuinely unconverged curved coordinate before any probe is spent. The band
/// tolerates the `O(e^{∓2ρ})` next-order term and assembly round-off; the
/// probing confirmation is the rigorous gate.
const TAIL_SNAP_CURVATURE_BAND: (f64, f64) = (0.25, 4.0);

/// Certify-time tail snap (#2348 Inc 2): when certification is about to refuse
/// a point whose gradient residual is carried entirely by coordinates crawling
/// an exponential tail TOWARD the ρ-box (the one-e-fold-per-Newton-step grind
/// the asymptote certificate exists to kill — the loop can exhaust its budget
/// strictly inside the box, where the Inc 1 railed mint can never fire),
/// positively confirm each such coordinate's tail from the current point and
/// return the point with those coordinates snapped to their box bound as an
/// optimization waypoint. The caller re-runs the outer search from that point;
/// only the resulting point is eligible for the Inc 1 rail certificate.
///
/// Refusal semantics mirror [`try_certify_asymptote_rail`]: any gate failure
/// returns `Ok(None)` (fall through to the ordinary refusal); the only `Err` is
/// a genuinely broken objective that cannot restore its inner state after
/// probing. Gates, in order of cost:
/// 1. candidate coordinates = un-railed, `|g_k|` above the stationarity bound,
///    positive own-curvature within [`TAIL_SNAP_CURVATURE_BAND`] of `|g_k|`
///    (the tail-law tie), with the rail side read from the gradient sign;
/// 2. the interior Hessian sub-block (railed + candidates excluded) is PSD;
/// 3. every candidate's tail is confirmed by the same probing engine the rail
///    mint uses (`CertifiedAtAsymptote` or `OnTailNotYetEquivalent`).
/// Outcome of a certify-time tail-snap attempt: a point already stationary on
/// its confirmed tail, a confirmed-tail waypoint requiring one optimization
/// retry (#2348 Inc 2b / #2358), or a human-readable decline reason carried
/// into the refusal summary.
#[derive(Debug)]
enum TailSnapOutcome {
    /// Every candidate's confirmed tail EXTRAPOLATES to a gradient already
    /// below the stationarity bound at the CURRENT point (#2348 Inc 2c): the
    /// coordinate is tail-stationary where it stands, and the measured local
    /// gradient is instrument noise (observed on the #2299 fixture: measured
    /// |g|=1.04e-2 at ρ=26.56 vs the clean-band extrapolation ĉ·e^{−ρ} ≈
    /// 1.9e-8, 100× below the bound). Mint the AsymptoteRail at this point —
    /// no snap, no reseed. `interior_projected_grad_norm` and the
    /// `effective_interior_bound` that certified it (the raw stationarity
    /// bound, or the interior sub-block's curvature-scaled flat-valley bound —
    /// the full-Hessian widening is disabled precisely because the noisy tail
    /// entry makes the full matrix non-PD) ride along for the certificate.
    TailStationaryAtPoint {
        rails: Vec<RailCoordinate>,
        interior_projected_grad_norm: f64,
        effective_interior_bound: StationarityBound,
    },
    /// Tails confirmed, but the current point is not already
    /// tail-stationary. The snapped rail point is only a waypoint: the plan
    /// runner must retry ONCE from it so coupled coordinates can reoptimize
    /// before certification.
    ConfirmedNeedsReseed(Array1<f64>),
    Declined(String),
}

/// Relative drift band for the tail-snap confirmation window, wider than the
/// exp4 characterization band (1e-3). The snap's evidentiary strength comes
/// from the EXTRAPOLATED-GAP margin, not the band tightness: a 1–2% spread in
/// `ĉ` across the clean run moves the extrapolated remaining gradient
/// `ĉ·e^{∓ρ}` by the same 1–2%, immaterial against the orders-of-magnitude
/// margin the at-point/stationarity decisions demand — while the true tail on
/// a REAL fixture still carries visible sub-percent curvature contamination at
/// probe depth (measured on #2299: ĉ ∈ {6544, 6565, 6574} over three e-folds,
/// drift 4.6e-3, against a wildly swinging noise region above).
const TAIL_SNAP_DRIFT_REL: f64 = 1.0e-2;

fn try_tail_snap_to_rail(
    obj: &mut dyn OuterObjective,
    inputs: &AsymptoteRailInputs<'_>,
) -> Result<TailSnapOutcome, EstimationError> {
    let rho = inputs.rho;
    let gradient = inputs.projected_gradient;
    let hessian = inputs.hessian;
    let (lower, upper) = inputs.bounds;
    let n = gradient.len();
    if rho.len() != n
        || hessian.nrows() != n
        || hessian.ncols() != n
        || lower.len() < n
        || upper.len() < n
    {
        return Ok(TailSnapOutcome::Declined("shape mismatch".to_string()));
    }

    let mut candidates: Vec<(usize, AsymptoteSide)> = Vec::new();
    let mut rejected: Vec<String> = Vec::new();
    for k in 0..n {
        if inputs.railed.contains(&k) {
            continue;
        }
        // #2453: snapping a coordinate to its bound and declaring the fit
        // finished is an act the tail law authorizes and nothing else does.
        // For a ψ coordinate — a curvature, a log length-scale — there is no
        // `λ = e^ρ` behind the box, so a gradient pointing at the endpoint is
        // just an unfinished search along that quantity, and pinning it there
        // would manufacture an optimum out of an arithmetic coincidence.
        if !inputs.layout.coordinate_is_log_smoothing(k) {
            rejected.push(format!(
                "k={k}: psi coordinate (rho_dim={}), no exponential tail law",
                inputs.layout.rho_dim()
            ));
            continue;
        }
        let g_k = gradient[k];
        let side = match AsymptoteSide::from_gradient(g_k, inputs.stationarity_bound.value()) {
            Some(side) => side,
            None => continue,
        };
        // A tail candidate must be DEEP toward the bound its gradient points
        // at — within the probe span of the box. The tail law is an asymptotic
        // statement; a coordinate sitting many probe-spans inside the interior
        // (every scripted mock optimum, every ordinary unconverged fit) has no
        // asymptote to confirm there, and probing it would spend a dozen
        // objective evaluations per would-refuse certification for nothing
        // (breaking eval-count-asserting harnesses along the way).
        let probe_span = ASYMPTOTE_PROBE_COUNT as f64;
        let deep_enough = match side {
            AsymptoteSide::Upper => upper[k] - rho[k] <= probe_span,
            AsymptoteSide::Lower => rho[k] - lower[k] <= probe_span,
        };
        if !deep_enough {
            rejected.push(format!(
                "k={k}: ρ={:.2} more than {probe_span:.0} e-folds inside the box",
                rho[k]
            ));
            continue;
        }
        let h_kk = hessian[[k, k]];
        // The tie is judged on |H_kk|/|g_k| — MAGNITUDE only. On the exact
        // tail `H_kk = |g_k|` (positive), but the assembled ρ-Hessian's tail
        // entry is `λV_λ + λ²V_λλ`, and when the `λ²V_λλ` trace pair cancels
        // to roundoff in the deep-smoothing regime (the #2298 rail-cancellation
        // class), what survives is `λV_λ = g_k` — magnitude right, SIGN
        // flipped (measured on the #2299 fixture: g=-1.040e-2, H_kk=-1.018e-2,
        // ratio -0.979). The sign at the tail is exactly the corrupted datum,
        // so it cannot gate; the probing confirmation is the rigorous test.
        let ratio = h_kk.abs() / g_k.abs();
        if !(TAIL_SNAP_CURVATURE_BAND.0..=TAIL_SNAP_CURVATURE_BAND.1).contains(&ratio) {
            rejected.push(format!(
                "k={k}: g={g_k:.3e} H_kk={h_kk:.3e} |ratio|={ratio:.3e} outside tie band"
            ));
            continue;
        }
        candidates.push((k, side));
    }
    if candidates.is_empty() {
        return Ok(TailSnapOutcome::Declined(if rejected.is_empty() {
            "no super-bound coordinate".to_string()
        } else {
            format!(
                "no candidate passed the curvature tie ({})",
                rejected.join("; ")
            )
        }));
    }

    // The curvature left after excluding the railed + candidate directions
    // must be admissible for a minimum; a genuinely indefinite interior
    // refuses before any probe is spent.
    let excluded: Vec<usize> = inputs
        .railed
        .iter()
        .copied()
        .chain(candidates.iter().map(|(k, _)| *k))
        .collect();
    if certificate_hessian_is_psd_off_railed_above_gradient_floor(hessian, &excluded, gradient)
        != Some(true)
    {
        return Ok(TailSnapOutcome::Declined(
            "interior Hessian sub-block not PSD".to_string(),
        ));
    }

    let beta_norm = inputs
        .terminal_beta
        .map(|b| b.dot(b).sqrt())
        .filter(|v| v.is_finite())
        .unwrap_or(0.0);
    let mut tol =
        AsymptoteTolerances::exp4_rail_bands(ASYMPTOTE_ESTIMAND_REL_TOL * (1.0 + beta_norm));
    tol.tail_drift_rel = TAIL_SNAP_DRIFT_REL;
    let mut decline: Option<String> = None;
    // Rails for candidates whose confirmed tail extrapolates to an
    // already-below-bound gradient at the CURRENT point; when every candidate
    // qualifies, the point is minted where it stands (#2348 Inc 2c).
    let mut at_point_rails: Vec<RailCoordinate> = Vec::new();
    for (k, side) in &candidates {
        let verdict = match probe_tail_window(obj, rho, *k, *side, &tol, (lower[*k], upper[*k]))? {
            (Some(window), rows) => match assess_coordinate(&window, &tol) {
                AsymptoteVerdict::CertifiedAtAsymptote {
                    side: assessed_side,
                    tail_constant,
                    estimand_travel_bound,
                    ..
                } => {
                    // Extrapolate the confirmed tail law to the current point:
                    // the TRUE remaining gradient there, immune to the local
                    // instrument noise the certificate measured.
                    let extrapolated_gap = match assessed_side {
                        AsymptoteSide::Upper => tail_constant * (-rho[*k]).exp(),
                        AsymptoteSide::Lower => tail_constant * rho[*k].exp(),
                    };
                    if extrapolated_gap.is_finite()
                        && extrapolated_gap <= inputs.stationarity_bound.value()
                    {
                        at_point_rails.push(RailCoordinate {
                            index: *k,
                            side: assessed_side,
                            tail_constant,
                            value_gap: extrapolated_gap,
                            estimand_travel_bound,
                            evidence: RailTailEvidence::ProbedTail {
                                noise_floor: tol.tail_noise_floor,
                                drift_band: tol.tail_drift_rel,
                            },
                        });
                    }
                    None
                }
                AsymptoteVerdict::OnTailNotYetEquivalent { .. } => None,
                AsymptoteVerdict::NoAsymptote { reason } => {
                    Some(format!("{reason}; probes: {rows}"))
                }
            },
            (None, rows) => Some(format!(
                "no finite-difference-clean tail run; probes: {rows}"
            )),
        };
        if let Some(reason) = verdict {
            decline = Some(format!("candidate k={k} tail unconfirmed: {reason}"));
            break;
        }
    }
    // #2349 round 7: a multi-coordinate rail face. When a candidate's OWN
    // one-dimensional tail law fails and several candidates ride out together,
    // the marginal law is the wrong object — overlapping penalties share range
    // space, so a lone coordinate's gradient saturates once the others
    // dominate the shared term (the measured #2349 ladder swept ĉ₀ across 8
    // orders of magnitude). The scalar section along the joint face direction
    // has the ordinary exponential tail; certify THAT with the same
    // discipline, and mint every face coordinate from the joint law.
    // A face confirmed through the JOINT fallback snaps as a WAYPOINT, never a
    // candidate optimum: the joint law certifies the direction of the optimum,
    // but individual face coordinates can hold interior optima once the others
    // sit railed (measured on the #2349 fixture: after snapping the 5-face,
    // coordinate 0's own gradient crossed zero near ρ₀ ≈ 7.5 — 4.5 e-folds
    // inside its snapped rail — while V dropped 3.86 from the checkpoint). The
    // reseed retry re-descends from the snapped point with the rails free to
    // hold or relax; a direct re-certification there would refuse exactly that
    // relaxation.
    if decline.is_some() && candidates.len() >= 2 {
        let (window, joint_rows) =
            probe_joint_tail_window(obj, rho, &candidates, &tol, (lower, upper))?;
        match window.as_ref().map(|w| assess_coordinate(w, &tol)) {
            Some(AsymptoteVerdict::CertifiedAtAsymptote {
                tail_constant,
                estimand_travel_bound,
                ..
            }) => {
                // Extrapolate the joint law back to the checkpoint: the true
                // remaining directional gradient there, immune to the local
                // instrument noise. All face gradients share one sign
                // structure along the face, so the joint gap bounds each
                // coordinate's own remaining gradient.
                let r0 = candidates
                    .iter()
                    .map(|(k, side)| match side {
                        AsymptoteSide::Upper => rho[*k],
                        AsymptoteSide::Lower => -rho[*k],
                    })
                    .sum::<f64>()
                    / candidates.len() as f64;
                let joint_gap = tail_constant * (-r0).exp();
                if joint_gap.is_finite() && joint_gap <= inputs.stationarity_bound.value() {
                    at_point_rails = candidates
                        .iter()
                        .map(|(k, side)| RailCoordinate {
                            index: *k,
                            side: *side,
                            tail_constant,
                            value_gap: joint_gap,
                            estimand_travel_bound,
                            evidence: RailTailEvidence::ProbedTail {
                                noise_floor: tol.tail_noise_floor,
                                drift_band: tol.tail_drift_rel,
                            },
                        })
                        .collect();
                }
                decline = None;
            }
            Some(AsymptoteVerdict::OnTailNotYetEquivalent { .. }) => {
                // Confirmed on the joint tail; travel not yet settled — the
                // face snaps/reseeds below exactly as a confirmed single
                // candidate would.
                decline = None;
            }
            Some(AsymptoteVerdict::NoAsymptote { reason }) => {
                // A returned window IS the law: it exists only when a
                // drift-band-clean, above-noise-floor, uniformly-positive
                // pencil-constant run of MIN_TAIL_SAMPLES was found, so the
                // only `NoAsymptote` reachable from it is the estimand
                // contraction gate — the β-steps in the retained (deep
                // interior) rows still move, i.e. the checkpoint is genuinely
                // NOT at the face limit yet (measured on the #2349 checkpoint:
                // ĉ settled to 34.2 over the last four probes while the crawl
                // was still travelling). That is the same state as
                // `OnTailNotYetEquivalent`: the law says WHERE the optimum is;
                // the snap below reoptimizes from the face, granting nothing
                // by itself.
                log::info!(
                    "[CERTIFICATE] joint {}-coordinate face: pencil-constant run \
                     confirmed but estimand not settled at the checkpoint \
                     ({reason}); snapping the face for re-optimization",
                    candidates.len(),
                );
                decline = None;
            }
            None => {
                decline = Some(format!(
                    "{}; joint {}-coordinate face: no finite-difference-clean run; joint probes: {joint_rows}",
                    decline.take().unwrap_or_default(),
                    candidates.len(),
                ));
            }
        }
    }
    // The probes warm-started the inner solve away from the checkpoint; every
    // exit below leaves the CURRENT point as the shipped state, so restore it
    // before returning. A failure here is a genuinely broken objective.
    obj.eval_cost(rho).map_err(|err| {
        EstimationError::RemlOptimizationFailed(format!(
            "{}: failed to restore the objective to the certified point after \
             tail-snap probing: {err}",
            inputs.context
        ))
    })?;
    if let Some(reason) = decline {
        return Ok(TailSnapOutcome::Declined(reason));
    }

    let interior_indices: Vec<usize> = interior_face_indices(gradient, inputs.railed)
        .into_iter()
        .filter(|k| !candidates.iter().any(|(c, _)| c == k))
        .collect();
    // #2348 Inc 2c: every candidate's confirmed tail already extrapolates
    // BELOW the stationarity bound at the current point — the fit is
    // tail-stationary where it stands and the measured local gradient is
    // instrument noise. Judge the interior with the shared two-stage
    // criterion (`certify_interior_stationarity`): the raw bound, then the
    // curvature-scaled flat-valley bound on the interior SUB-BLOCK (the
    // full-Hessian widening is unavailable here exactly because the
    // noise-corrupted tail entry makes the full matrix non-PD).
    if at_point_rails.len() == candidates.len() {
        if let Ok((interior_projected_grad_norm, effective_interior_bound)) =
            certify_interior_stationarity(
                gradient,
                hessian,
                &interior_indices,
                inputs.stationarity_bound,
                inputs.objective_tol,
            )
        {
            return Ok(TailSnapOutcome::TailStationaryAtPoint {
                rails: at_point_rails,
                interior_projected_grad_norm,
                effective_interior_bound,
            });
        }
        // Real interior descent remains: fall through to the reseed path so
        // one more optimizer pass polishes it.
    }

    let mut snapped = rho.clone();
    for (k, side) in &candidates {
        snapped[*k] = match side {
            AsymptoteSide::Upper => upper[*k],
            AsymptoteSide::Lower => lower[*k],
        };
    }

    // Tails confirmed, but a finite snap can change every coupled coordinate's
    // optimum even when its PRE-snap gradient was stationary (#2358 measured
    // the location-scale interior gradient jumping to 0.5). The snap proves a
    // direction and supplies a waypoint; only a resumed optimization and its
    // subsequent certificate can prove the endpoint.
    Ok(TailSnapOutcome::ConfirmedNeedsReseed(snapped))
}

/// Reconstruct one railed coordinate's exponential tail by probing the analytic
/// gradient back from the rail at coarse and, when needed, local resolution;
/// locate the longest finite-difference-clean run (rejecting the noise floor
/// adjacent to the rail); and assess it against the tail law (#2348 Inc 1 /
/// #2337 Thm 2.1). Returns the certified [`RailCoordinate`] or `None` if no
/// confirmable tail is found.
fn build_and_assess_rail_coordinate(
    obj: &mut dyn OuterObjective,
    rho: &Array1<f64>,
    coord: usize,
    side: AsymptoteSide,
    tol: &AsymptoteTolerances,
    domain: (f64, f64),
) -> Result<Result<RailCoordinate, String>, EstimationError> {
    let window = match probe_tail_window(obj, rho, coord, side, tol, domain)? {
        (Some(window), _) => window,
        (None, rows) => {
            // #2450: name the floor instead of leaving the reader to find it.
            // The criterion ALWAYS carries the soft rho-guard barrier
            // (`soft_rho_guard_prior_atom`), and a `log cosh` barrier's gradient
            // SATURATES to `w*a` rather than decaying, so once the REML tail
            // falls below it, `c_hat = -e^rho * dV/drho` diverges and no tail is
            // observable THROUGH the guard at any box width or probe count. That
            // is a different failure from a noisy or truncated window, and
            // reporting both as "no finite-difference-clean tail window" cost
            // real time: it reads as a fixture/conditioning problem when it is a
            // property of the objective. Measured: the floor is exactly
            // `w*a*tanh(a*rho)` = 1.3324e-7 at rho=30 against a predicted
            // 1.3324e-7 (five significant figures, every coordinate).
            let guard_floor = crate::estimate::RHO_SOFT_PRIOR_WEIGHT
                * (crate::estimate::RHO_SOFT_PRIOR_SHARPNESS / crate::estimate::RHO_BOUND);
            return Ok(Err(format!(
                "k={coord}: no finite-difference-clean tail window; note the soft \
                 rho-guard gradient saturates at w*a={guard_floor:.4e} (#2450), so any \
                 tail whose |dV/drho| is at or below that is unobservable through the \
                 guard rather than merely unclean; probes {rows}"
            )));
        }
    };
    match assess_coordinate(&window, tol) {
        AsymptoteVerdict::CertifiedAtAsymptote {
            side,
            tail_constant,
            value_gap,
            estimand_travel_bound,
        } => Ok(Ok(RailCoordinate {
            index: coord,
            side,
            tail_constant,
            value_gap,
            estimand_travel_bound,
            evidence: RailTailEvidence::ProbedTail {
                noise_floor: tol.tail_noise_floor,
                drift_band: tol.tail_drift_rel,
            },
        })),
        other => Ok(Err(format!("k={coord}: tail verdict {other:?}"))),
    }
}

/// Detect a WRONG-RAIL coordinate (#2392): one sitting AT its ρ box bound whose
/// clean-band probes prove the objective strictly DECREASES as the coordinate
/// moves INWARD — the outer search drove it to the wrong bound. Returns the
/// interior ρ to reseed the coordinate at (the deepest drift-clean probe, where
/// `|g|` is largest and the descent is most informative) when the proof holds,
/// else `None`.
///
/// # Proof condition (evidence-gated; cannot launder a genuine λ→∞ / λ→0 optimum)
///
/// Probe [`ASYMPTOTE_PROBE_COUNT`] e-folds inward and require a contiguous run of
/// at least [`MIN_TAIL_SAMPLES`] probes that is, at once:
/// 1. above the gradient interior floor, `|g| > interior_grad_tol` (so a probe
///    whose gradient has decayed into finite-difference cancellation next to the
///    rail is excluded rather than read as a settled tail);
/// 2. above the pencil-constant noise floor, `|ĉ| > tail_noise_floor`, where
///    `ĉ = side.tail_constant(ρ, g)` uses the coordinate's ACTUAL rail side;
/// 3. drift-band-clean in `ĉ` within `tail_drift_rel` (the same constant-pencil
///    band the genuine tail uses — `run_drift_within_band` keys on `|mean|`, so a
///    uniformly-negative run is judged on its magnitude); AND
/// 4. `ĉ` uniformly of the sign OPPOSITE a genuine tail — `ĉ < 0` — i.e. the
///    descent direction points AWAY from the bound (`∂V/∂ρ > 0` at an upper rail,
///    `∂V/∂ρ < 0` at a lower rail).
///
/// A genuine rail (descent TOWARD the bound) has `ĉ > 0` across the clean run and
/// never satisfies (4), so it is never pulled off its rail. This shares every
/// tolerance with the genuine asymptote path ([`AsymptoteTolerances`]); the ONLY
/// difference is the sign gate in (4).
fn detect_wrong_rail_pullback(
    obj: &mut dyn OuterObjective,
    rho: &Array1<f64>,
    coord: usize,
    side: AsymptoteSide,
    tol: &AsymptoteTolerances,
    domain: (f64, f64),
) -> Result<Option<f64>, EstimationError> {
    const PROBE_DELTA: f64 = 1.0;
    const PROBE_DOMAIN_MARGIN: f64 = 1.0e-6;
    // Upper rail (ρ → +∞): step ρ DOWN into the interior. Lower rail: step UP.
    let sign = match side {
        AsymptoteSide::Upper => -1.0,
        AsymptoteSide::Lower => 1.0,
    };
    // rows[r] is probe j=r+1: r=0 is CLOSEST to the rail, increasing r steps
    // further into the interior (larger |grad| on a clean run).
    let mut rows: Vec<(f64, f64)> = Vec::new();
    for j in 1..=ASYMPTOTE_PROBE_COUNT {
        let stepped = rho[coord] + sign * (j as f64) * PROBE_DELTA;
        if stepped <= domain.0 + PROBE_DOMAIN_MARGIN || stepped >= domain.1 - PROBE_DOMAIN_MARGIN {
            break;
        }
        let mut probe = rho.clone();
        probe[coord] = stepped;
        let eval = match obj.eval_with_order(&probe, OuterEvalOrder::ValueAndGradient) {
            Ok(eval) => eval,
            Err(_) => break,
        };
        if !eval.cost.is_finite() || coord >= eval.gradient.len() || !eval.gradient[coord].is_finite()
        {
            break;
        }
        rows.push((stepped, eval.gradient[coord]));
    }
    if rows.len() < MIN_TAIL_SAMPLES {
        return Ok(None);
    }
    let constants: Vec<f64> = rows
        .iter()
        .map(|(r, g)| side.tail_constant(*r, *g))
        .collect();
    // Wrong-rail element-clean: gradient above the interior floor, pencil
    // constant above the noise floor IN MAGNITUDE, and of the descent-inward
    // (negative) sign. Genuine tails have ĉ > 0 here and are excluded.
    let element_clean: Vec<bool> = rows
        .iter()
        .zip(&constants)
        .map(|((_, g), c)| {
            c.is_finite() && *c < -tol.tail_noise_floor && g.abs() > tol.interior_grad_tol
        })
        .collect();
    // Longest contiguous element-clean, drift-band-clean run; reseed the
    // coordinate at the DEEPEST such probe (largest |g|, closest to the true
    // interior optimum the descent points toward).
    let mut best: Option<(usize, usize)> = None;
    for a in 0..rows.len() {
        if !element_clean[a] {
            continue;
        }
        for b in a..rows.len() {
            if !element_clean[b] {
                break;
            }
            if b - a + 1 < MIN_TAIL_SAMPLES {
                continue;
            }
            if !run_drift_within_band(&constants[a..=b], tol.tail_drift_rel) {
                continue;
            }
            match best {
                Some((ba, bb)) if bb - ba + 1 >= b - a + 1 => {}
                _ => best = Some((a, b)),
            }
        }
    }
    Ok(best.map(|(_, b)| rows[b].0))
}

/// Probe one coordinate's tail toward the interior (the shared probing engine
/// of [`build_and_assess_rail_coordinate`] and the certify-time tail snap).
/// The one-e-fold ladder runs first; if it finds no clean run, a short
/// half-e-fold ladder resolves a narrower local band without mixing step sizes
/// in one estimand window. Returns the longest finite-difference-clean
/// constant-`ĉ` run (newest sample nearest `rho[coord]`), or `None` when neither
/// resolution contains at least [`MIN_TAIL_SAMPLES`] clean rows. The second
/// element dumps `(ρ, ∂V/∂ρ, ĉ)` evidence for every attempted resolution.
fn probe_tail_window(
    obj: &mut dyn OuterObjective,
    rho: &Array1<f64>,
    coord: usize,
    side: AsymptoteSide,
    tol: &AsymptoteTolerances,
    domain: (f64, f64),
) -> Result<(Option<AsymptoteWindow>, String), EstimationError> {
    let (coarse_window, coarse_rows) = probe_tail_window_at_resolution(
        obj,
        rho,
        coord,
        side,
        tol,
        domain,
        (1.0, ASYMPTOTE_PROBE_COUNT),
    )?;
    if coarse_window.is_some() {
        return Ok((coarse_window, coarse_rows));
    }

    // The coarse ladder is deliberately retained as the first pass: railed
    // dense REML fits can have several e-folds of cancellation noise beside
    // the box followed by a clean band far inside it. The local pass addresses
    // the complementary geometry exposed by #2358, where a modest finite box
    // contains a narrow but regular tail and unit steps skip over it.
    let (local_window, local_rows) = probe_tail_window_at_resolution(
        obj,
        rho,
        coord,
        side,
        tol,
        domain,
        (
            ASYMPTOTE_LOCAL_PROBE_DELTA,
            ASYMPTOTE_LOCAL_PROBE_COUNT,
        ),
    )?;
    Ok((
        local_window,
        format!("coarse[{coarse_rows}] local[{local_rows}]"),
    ))
}

/// Probe a single equally-spaced resolution of one coordinate's tail.
///
/// Keeping each returned window at one resolution is essential for
/// [`assess_coordinate`]: its coefficient-travel bound estimates a geometric
/// ratio from consecutive steps, which is only meaningful when their `Δρ`
/// values are identical.
fn probe_tail_window_at_resolution(
    obj: &mut dyn OuterObjective,
    rho: &Array1<f64>,
    coord: usize,
    side: AsymptoteSide,
    tol: &AsymptoteTolerances,
    domain: (f64, f64),
    resolution: (f64, usize),
) -> Result<(Option<AsymptoteWindow>, String), EstimationError> {
    let (probe_delta, probe_count) = resolution;
    // Strictly-inside guard for probes against the probed coordinate's own box
    // interval (#2388). The ρ-gradient assembly freezes any coordinate at (or
    // within 1e-8 of) its recorded upper bound to the #197 KKT projection — a
    // literal 0.0 — so a probe at or past a box bound samples the frozen-axis
    // convention, not the criterion's tail: a fabricated hard-zero tail that
    // the drift band can never confirm. Out-of-box points are outside the
    // λ-selection domain altogether; they are not evidence for or against a
    // tail, so the ladder stops at the last strictly-in-domain probe.
    const PROBE_DOMAIN_MARGIN: f64 = 1.0e-6;
    // Upper rail (ρ → +∞): step ρ DOWN into the tail. Lower rail: step UP.
    let sign = match side {
        AsymptoteSide::Upper => -1.0,
        AsymptoteSide::Lower => 1.0,
    };
    // rows[r] corresponds to probe j=r+1: r=0 is the point CLOSEST to the rail,
    // increasing r steps further into the interior (larger |grad|).
    let mut rows: Vec<(f64, f64, Option<Array1<f64>>)> = Vec::new();
    for j in 1..=probe_count {
        let stepped = rho[coord] + sign * (j as f64) * probe_delta;
        if stepped <= domain.0 + PROBE_DOMAIN_MARGIN || stepped >= domain.1 - PROBE_DOMAIN_MARGIN {
            break;
        }
        let mut probe = rho.clone();
        probe[coord] = stepped;
        let eval = match obj.eval_with_order(&probe, OuterEvalOrder::ValueAndGradient) {
            Ok(eval) => eval,
            // A failed probe is not evidence against a tail; stop probing and
            // assess whatever clean run the earlier probes established.
            Err(_) => break,
        };
        if !eval.cost.is_finite()
            || coord >= eval.gradient.len()
            || !eval.gradient[coord].is_finite()
        {
            break;
        }
        rows.push((probe[coord], eval.gradient[coord], eval.inner_beta_hint));
    }
    let rows_summary = rows
        .iter()
        .map(|(r, g, _)| {
            format!(
                "(ρ={r:.2}, g={g:.3e}, ĉ={:.3e})",
                side.tail_constant(*r, *g)
            )
        })
        .collect::<Vec<_>>()
        .join(" ");
    if rows.len() < MIN_TAIL_SAMPLES {
        return Ok((None, rows_summary));
    }

    // Per-row pencil constant ĉ and the element-clean predicate: ĉ above the
    // noise floor AND the gradient above the interior floor (so a row adjacent to
    // the rail, whose gradient has decayed into finite-difference cancellation, is
    // excluded rather than mistaken for a settled tail).
    let constants: Vec<f64> = rows
        .iter()
        .map(|(r, g, _)| side.tail_constant(*r, *g))
        .collect();
    let element_clean: Vec<bool> = rows
        .iter()
        .zip(&constants)
        .map(|((_, g, _), c)| {
            c.is_finite() && *c > tol.tail_noise_floor && g.abs() > tol.interior_grad_tol
        })
        .collect();

    // Longest contiguous run that is element-clean AND holds ĉ within the drift
    // band; ties broken toward the rail (smallest start) for the most settled
    // estimand.
    let mut best: Option<(usize, usize)> = None;
    for a in 0..rows.len() {
        if !element_clean[a] {
            continue;
        }
        for b in a..rows.len() {
            if !element_clean[b] {
                break;
            }
            if b - a + 1 < MIN_TAIL_SAMPLES {
                continue;
            }
            if !run_drift_within_band(&constants[a..=b], tol.tail_drift_rel) {
                continue;
            }
            let len = b - a + 1;
            match best {
                Some((ba, bb)) if bb - ba + 1 >= len => {}
                _ => best = Some((a, b)),
            }
        }
    }
    let (a, b) = match best {
        Some(run) => run,
        None => return Ok((None, rows_summary)),
    };

    // Build the window oldest → newest: newest (window `latest`) is the row
    // CLOSEST to the rail (r=a). A sample's coefficient move is ‖β(r) − β(r+1)‖,
    // the step from the next-farther retained row toward the rail.
    let mut window = AsymptoteWindow::with_capacity(b - a + 1);
    for r in (a..=b).rev() {
        let (rho_r, grad_r, beta_r) = &rows[r];
        let coef_step_norm = match (beta_r, rows.get(r + 1).map(|row| &row.2)) {
            (Some(cur), Some(Some(farther))) if cur.len() == farther.len() => {
                (cur - farther).iter().map(|v| v * v).sum::<f64>().sqrt()
            }
            _ => 0.0,
        };
        window.push(AsymptoteSample {
            rho: *rho_r,
            grad: *grad_r,
            coef_step_norm,
        });
    }

    Ok((Some(window), rows_summary))
}

/// Probe a JOINT multi-coordinate rail face (#2349 round 7 / #2348): step every
/// face coordinate one e-fold toward the interior TOGETHER and assess the
/// directional gradient along the outward face direction against the same
/// exponential tail law, noise floor, and drift band as the single-coordinate
/// window.
///
/// Why a joint law exists where the per-coordinate laws fail: for OVERLAPPING
/// penalties (e.g. the multinomial per-class family's coalesced pseudo-logdet
/// `½log|Σ_s λ_s M_s|₊`) several λs ride to ∞ on one face and share range
/// space. Moving ONE coordinate down leaves the shared term dominated by the
/// others, so that coordinate's own gradient saturates and its per-probe pencil
/// constant `ĉ_k = |g_k|e^{ρ_k}` sweeps orders of magnitude — measured on the
/// #2349 checkpoint: ĉ₀ spanning 4.5e2 → 1.0e-6 over the ladder, an honest
/// refusal of a law that genuinely does not hold marginally. Along the face
/// direction `u` (`u_k = +1` toward an upper rail, `−1` toward a lower rail)
/// the shared term moves coherently and the scalar objective section
/// `t ↦ V(ρ + t·u)` has the ordinary one-dimensional exponential tail; its
/// pencil constant is assessed with the pseudo-coordinate `r = mean_k(u_k ρ_k)`
/// and the directional derivative `g_u = Σ_{k∈face} u_k g_k = dV/dt`.
///
/// The window it returns speaks the [`assess_coordinate`] conventions
/// verbatim: on a genuine face `g_u < 0` at every interior probe (descent runs
/// outward), so the verdict side is `Upper` in the pseudo-coordinate
/// regardless of the mix of physical sides, and `ĉ = −e^{r}·g_u` recovers the
/// joint tail constant. Per-coordinate rails minted from it keep their own
/// physical [`AsymptoteSide`].
fn probe_joint_tail_window(
    obj: &mut dyn OuterObjective,
    rho: &Array1<f64>,
    face: &[(usize, AsymptoteSide)],
    tol: &AsymptoteTolerances,
    bounds: (&Array1<f64>, &Array1<f64>),
) -> Result<(Option<AsymptoteWindow>, String), EstimationError> {
    const PROBE_DELTA: f64 = 1.0;
    const PROBE_DOMAIN_MARGIN: f64 = 1.0e-6;
    let (lower, upper) = bounds;
    // Outward unit direction of the face; probes step INWARD (−u).
    let direction: Vec<(usize, f64)> = face
        .iter()
        .map(|(k, side)| {
            (
                *k,
                match side {
                    AsymptoteSide::Upper => 1.0,
                    AsymptoteSide::Lower => -1.0,
                },
            )
        })
        .collect();
    let r0 = direction
        .iter()
        .map(|(k, u)| u * rho[*k])
        .sum::<f64>()
        / direction.len() as f64;
    let mut rows: Vec<(f64, f64, Option<Array1<f64>>)> = Vec::new();
    for j in 1..=ASYMPTOTE_PROBE_COUNT {
        let step = (j as f64) * PROBE_DELTA;
        let mut probe = rho.clone();
        let mut in_domain = true;
        for (k, u) in &direction {
            let stepped = rho[*k] - u * step;
            if stepped <= lower[*k] + PROBE_DOMAIN_MARGIN
                || stepped >= upper[*k] - PROBE_DOMAIN_MARGIN
            {
                in_domain = false;
                break;
            }
            probe[*k] = stepped;
        }
        if !in_domain {
            break;
        }
        let eval = match obj.eval_with_order(&probe, OuterEvalOrder::ValueAndGradient) {
            Ok(eval) => eval,
            Err(_) => break,
        };
        if !eval.cost.is_finite() {
            break;
        }
        let mut g_u = 0.0;
        let mut finite = true;
        for (k, u) in &direction {
            match eval.gradient.get(*k) {
                Some(g) if g.is_finite() => g_u += u * g,
                _ => {
                    finite = false;
                    break;
                }
            }
        }
        if !finite {
            break;
        }
        rows.push((r0 - step, g_u, eval.inner_beta_hint));
    }
    let rows_summary = rows
        .iter()
        .map(|(r, g, _)| {
            format!(
                "(r={r:.2}, dV/dt={g:.3e}, ĉ={:.3e})",
                AsymptoteSide::Upper.tail_constant(*r, *g)
            )
        })
        .collect::<Vec<_>>()
        .join(" ");
    if rows.len() < MIN_TAIL_SAMPLES {
        return Ok((None, rows_summary));
    }
    let constants: Vec<f64> = rows
        .iter()
        .map(|(r, g, _)| AsymptoteSide::Upper.tail_constant(*r, *g))
        .collect();
    let element_clean: Vec<bool> = rows
        .iter()
        .zip(&constants)
        .map(|((_, g, _), c)| {
            c.is_finite() && *c > tol.tail_noise_floor && g.abs() > tol.interior_grad_tol
        })
        .collect();
    let mut best: Option<(usize, usize)> = None;
    for a in 0..rows.len() {
        if !element_clean[a] {
            continue;
        }
        for b in a..rows.len() {
            if !element_clean[b] {
                break;
            }
            if b - a + 1 < MIN_TAIL_SAMPLES {
                continue;
            }
            if !run_drift_within_band(&constants[a..=b], tol.tail_drift_rel) {
                continue;
            }
            let len = b - a + 1;
            match best {
                Some((ba, bb)) if bb - ba + 1 >= len => {}
                _ => best = Some((a, b)),
            }
        }
    }
    let (a, b) = match best {
        Some(run) => run,
        None => return Ok((None, rows_summary)),
    };
    let mut window = AsymptoteWindow::with_capacity(b - a + 1);
    for r in (a..=b).rev() {
        let (rho_r, grad_r, beta_r) = &rows[r];
        let coef_step_norm = match (beta_r, rows.get(r + 1).map(|row| &row.2)) {
            (Some(cur), Some(Some(farther))) if cur.len() == farther.len() => {
                (cur - farther).iter().map(|v| v * v).sum::<f64>().sqrt()
            }
            _ => 0.0,
        };
        window.push(AsymptoteSample {
            rho: *rho_r,
            grad: *grad_r,
            coef_step_norm,
        });
    }
    Ok((Some(window), rows_summary))
}

/// Whether a run of pencil constants holds constant within the relative drift
/// band `(max − min)/|mean| ≤ band` (deterministic, ordered).
fn run_drift_within_band(constants: &[f64], band: f64) -> bool {
    if constants.len() < MIN_TAIL_SAMPLES {
        return false;
    }
    let mut sum = 0.0_f64;
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for &c in constants {
        if !c.is_finite() {
            return false;
        }
        sum += c;
        lo = lo.min(c);
        hi = hi.max(c);
    }
    let mean = sum / constants.len() as f64;
    if !(mean.abs() > 0.0) {
        return false;
    }
    (hi - lo) / mean.abs() <= band
}

pub(crate) fn compute_rho_uncertainty_diagnostic(
    obj: &mut dyn OuterObjective,
    config: &OuterConfig,
    context: &str,
    result: &mut OuterResult,
) -> crate::rho_uncertainty::RhoUncertaintyDiagnostic {
    let terminal_cap_guard = config
        .outer_inner_cap
        .as_ref()
        .map(TerminalInnerCapGuard::lift);
    // Do not reset here. The diagnostic intentionally runs before terminal
    // installation and certification; the certificate must remain the final
    // owner of objective state. Holding cap=0 ensures proposal evaluations use
    // terminal fidelity, and the selected point is reinstalled immediately
    // afterward before the mint audit.
    let diagnostic =
        compute_rho_uncertainty_diagnostic_at_terminal_fidelity(obj, config, context, result);
    drop(terminal_cap_guard);
    diagnostic
}

fn compute_rho_uncertainty_diagnostic_at_terminal_fidelity(
    obj: &mut dyn OuterObjective,
    config: &OuterConfig,
    context: &str,
    result: &mut OuterResult,
) -> crate::rho_uncertainty::RhoUncertaintyDiagnostic {
    let cap = obj.capability();
    let layout = cap.theta_layout();
    let rho_dim = layout.rho_dim();
    let gate = crate::rho_uncertainty::RhoUncertaintyCostGate {
        sample_count: 32,
        problem_size: config.rho_uncertainty_problem_size,
    };
    if let Err(reason) = crate::rho_uncertainty::cost_gate_allows(rho_dim, gate) {
        return crate::rho_uncertainty::RhoUncertaintyDiagnostic::skipped(reason, 0);
    }
    if result.rho.len() != layout.n_params {
        return crate::rho_uncertainty::RhoUncertaintyDiagnostic::skipped(
            format!(
                "final outer point length {} does not match objective dimension {}",
                result.rho.len(),
                layout.n_params
            ),
            0,
        );
    }
    // The ρ-uncertainty diagnostic needs the EXACT outer Hessian, but it runs
    // BEFORE terminal certification so that the certificate remains the final
    // owner of objective state. Under the optimize-3/certify-4 protocol (#2359)
    // this diagnostic may therefore consume only curvature the SEARCH already
    // retained. It must never trigger its own `ValueGradientHessian` evaluation:
    // a gradient-only search has deliberately reserved that one order-four pass
    // for the mint gate. Such a search skips this optional diagnostic; the
    // terminal certificate still computes and persists exact curvature.
    if !cap.hessian.is_analytic() {
        return crate::rho_uncertainty::RhoUncertaintyDiagnostic::skipped(
            "outer Hessian is not analytic; rho-uncertainty diagnostic needs exact curvature",
            0,
        );
    }
    if result.plan_used.hessian_source != HessianSource::Analytic {
        return crate::rho_uncertainty::RhoUncertaintyDiagnostic::skipped(
            "search did not use exact outer curvature; order-four work is reserved for the terminal certificate",
            0,
        );
    }

    let hessian = match result.final_hessian.as_ref() {
        Some(hessian) => hessian.clone(),
        None => {
            return crate::rho_uncertainty::RhoUncertaintyDiagnostic::skipped(
                "search retained no exact outer Hessian; order-four work is reserved for the terminal certificate",
                0,
            );
        }
    };
    if hessian.nrows() != layout.n_params || hessian.ncols() != layout.n_params {
        return crate::rho_uncertainty::RhoUncertaintyDiagnostic::skipped(
            format!(
                "exact outer Hessian shape {}x{} does not match objective dimension {}",
                hessian.nrows(),
                hessian.ncols(),
                layout.n_params
            ),
            1,
        );
    }
    let mut hessian_rho = Array2::<f64>::zeros((rho_dim, rho_dim));
    for row in 0..rho_dim {
        for col in 0..rho_dim {
            hessian_rho[[row, col]] = hessian[[row, col]];
        }
    }
    let rho_hat = result.rho.slice(ndarray::s![..rho_dim]).to_owned();
    let theta_hat = result.rho.clone();
    let cost_hat = result.final_value;
    let diagnostic = {
        let mut served_hat_cost = false;
        let mut criterion = |rho: &Array1<f64>| -> Option<f64> {
            let is_hat = rho.len() == rho_hat.len()
                && rho
                    .iter()
                    .zip(rho_hat.iter())
                    .all(|(&left, &right)| left.to_bits() == right.to_bits());
            if is_hat && !served_hat_cost {
                served_hat_cost = true;
                return Some(cost_hat);
            }
            let mut theta = theta_hat.clone();
            for idx in 0..rho_dim {
                theta[idx] = rho[idx];
            }
            obj.eval_cost(&theta).ok()
        };
        crate::rho_uncertainty::rho_uncertainty_diagnostic(
            &rho_hat,
            &hessian_rho,
            gate,
            &mut criterion,
        )
    };
    match &diagnostic.status {
        crate::rho_uncertainty::RhoUncertaintyStatus::NoEvidenceOfHeavyTails => {
            log::info!(
                "[RHO uncertainty] {context}: no heavy-tail evidence at sampled rho proposals k_hat={:.3} evals={}",
                diagnostic.k_hat.unwrap_or(f64::NAN),
                diagnostic.n_evaluations,
            );
        }
        crate::rho_uncertainty::RhoUncertaintyStatus::HeavyTailsDetected { k_hat } => {
            log::warn!(
                "[RHO uncertainty] {context}: heavy rho-importance tail detected k_hat={:.3} evals={}",
                k_hat,
                diagnostic.n_evaluations,
            );
        }
        crate::rho_uncertainty::RhoUncertaintyStatus::Skipped { reason } => {
            log::info!("[RHO uncertainty] {context}: skipped ({reason})");
        }
    }
    diagnostic
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OperatorTrustRegionStopReason {
    Converged,
    RejectFloor,
    IterationBudget,
    /// The objective stopped changing on a criterion-flat surface. The
    /// in-loop guard may already have certified the score-relative residual or
    /// may have returned a non-stationary floor; either way the final analytic
    /// certificate needs this provenance to reproduce the guard's derived
    /// stationarity band exactly.
    CostStallFlatValley,
    /// Family returned a non-operator Hessian mid-flight after routing into
    /// the operator path. Best-effort `x_k` returned with this reason; the
    /// caller should consider re-fitting under a different solver class
    /// (e.g. BFGS gradient-only) instead of trusting the partial result.
    RoutingMismatch,
    /// The solver stopped because something failed, not because a test it
    /// stands behind was satisfied: the line search gave up, an objective
    /// evaluation failed, or the arithmetic went non-finite.
    ///
    /// These used to map to [`Self::Converged`], on the stated premise that
    /// they are "a hard failure the caller sees through the `Err` arm". That
    /// premise is false for the line-search case on the path it actually
    /// takes: `run_plan` turns a `BfgsError::LineSearchFailed` whose last
    /// iterate is finite into `Ok(non-converged)`, because that iterate is a
    /// usable checkpoint. The caller therefore sees no `Err`, and the coarse
    /// reason it does see said `Converged` — which is how a binomial/logit
    /// REML fit that never accepted a single step came to report
    /// `stop_reason=Converged after 1 outer iteration(s)` (#2614).
    ///
    /// Reporting-only today: no consumer branches on `Converged`, so this
    /// splits a label without moving a decision. Kept separate from
    /// [`Self::IterationBudget`] because "ran out of budget" and "could not
    /// take a step" call for different repairs.
    SolverFailure,
}

/// Run the outer smoothing-parameter optimization.
///
/// This is the single entry point that replaces the scattered optimizer wiring
/// across estimate.rs, joint.rs, and custom_family.rs. It:
///
/// 1. Queries and canonicalizes the objective's capability declaration.
/// 2. Calls `plan()` to select solver + hessian source.
/// 3. Logs the plan and the analytic derivative capabilities it will consume.
/// 4. Generates seed candidates.
/// 5. Runs the chosen solver on candidates in heuristic order up to budget.
/// 6. If the configured fallback policy allows it, re-plans with degraded
///    capabilities chosen centrally inside outer_strategy and retries.
/// 7. Returns the best result (including which plan was actually used).
///
/// Do not wrap `run_outer` calls in try/catch with ad-hoc solver recovery.
/// Callers should declare only the primary capability and, at most, whether
/// automatic fallback is enabled at all.
///
/// Bound on the certify-last checkpoint-resume loop (#2273/#2374). When a
/// solver CLAIMS convergence but the mandatory analytic certificate refuses,
/// the loop re-runs the outer search seeded AT the refused checkpoint (with a
/// fresh metric for gradient-only outers, since `final_hessian` is `None`)
/// while each resume strictly reduces the objective — real descent the claim
/// left unexploited. #2273 introduced this as a SINGLE retry for the
/// stale-tolerance desync (one reseed re-anchors the in-loop tolerance to the
/// terminal cost scale and certifies). #2374 generalized it to a
/// progress-bounded loop: a gradient-only `opt::Bfgs` outer in log-λ space can
/// exit its flat-valley `StallPolicy` at `‖g‖∞ ≤ tol·(1 + ‖ρ‖∞)` — a gate
/// inflated ~10× by a railed coordinate — reporting `Ok(converged)` at a
/// checkpoint whose projected gradient the un-inflated certificate correctly
/// rejects, and a single fresh-metric reseed rarely lands the optimum in one
/// hop (the transformation-survival LAML of #2373 needed two). This bound caps
/// how many such reseeds are attempted before the honest non-convergence is
/// surfaced; a fit that certifies on the first pass never enters the loop, and
/// a reseed that fails to reduce the objective (a genuine non-stationary floor
/// a fresh metric cannot escape) stops the loop immediately regardless of the
/// remaining budget.
const OUTER_CERTIFY_RESUME_BUDGET: usize = 16;

/// Max interior strict-saddle escape resumes (#2357/#2155). A genuine saddle is
/// cleared in one escape; the small cap keeps a pathological non-convergent
/// objective (e.g. a bimodal inner solve, #2363) from re-escaping a family of
/// shallow saddles until the general resume budget is spent.
pub(crate) const OUTER_SADDLE_ESCAPE_BUDGET: usize = 3;

/// Roundoff-relative scale below which a certify-last reseed's objective
/// reduction is numerical noise rather than exploited descent (#2374). A
/// fresh-metric BFGS restart seeded AT the refused checkpoint can only reduce
/// the objective from that checkpoint, so `retried == prior` (to roundoff)
/// means the restart found no descent — a genuine stationary floor — while a
/// false flat-valley stall yields a reduction orders of magnitude above this
/// scale. The progress gate MUST anchor on roundoff, not the much larger
/// cost-stall relative floor: a flat valley crawls out in per-reseed steps far
/// smaller than `rel_cost·(1 + |cost|)` (the transformation-survival LAML moves
/// ~4e-5 relative per reseed), and gating on that coarser floor stops the crawl
/// after a single hop and refuses a well-posed fit.
const CERTIFY_RESUME_PROGRESS_REL: f64 = 32.0 * f64::EPSILON;

pub(crate) fn run_outer(
    obj: &mut dyn OuterObjective,
    config: &OuterConfig,
    context: &str,
) -> Result<OuterResult, EstimationError> {
    // Permutation-invariant outer search (#1538/#1539). When the caller has
    // supplied per-coordinate structural keys that induce a non-identity
    // canonical order, run the ENTIRE outer pipeline (seeding, multistart,
    // optimization, and the #934 certificate / uncertainty audits) in that
    // canonical layout against a permuting wrapper, then map the result back to
    // the native layout. Seeding/tie-breaking then see byte-identical
    // coordinates for every term order, so both orders select the same λ̂.
    if let Some(keys) = config.rho_canonical_keys.as_ref()
        && let Some(perm) = canonical_permutation(keys)
    {
        let canonical_config = canonicalize_outer_config(config, &perm);
        let mut canonical_obj = CanonicalizedObjective::new(obj, perm.clone());
        let result = run_outer(&mut canonical_obj, &canonical_config, context)?;
        return Ok(outer_result_to_native(result, &perm));
    }
    let mut result = run_outer_uncertified(obj, config, context)?;
    if obj.begin_exact_polish() {
        // A sampled outer-derivative pilot is an optimization stage, never a
        // certifiable objective. Continue from its best checkpoint on the
        // family's exact full-data measure before the mandatory analytic
        // certificate. This transition is unconditional whenever the family
        // reports that a sample actually ran, so convergence before a nominal
        // phase budget cannot strand the optimizer on the stochastic surface
        // (#979: matrix-free TR stopped after 6 evaluations while the family
        // waited for a 12-evaluation counter).
        let pilot_iterations = result.iterations;
        let mut exact_config = config.clone();
        exact_config.initial_rho = Some(result.rho.clone());
        exact_config.heuristic_lambdas = None;
        exact_config.seed_config.max_seeds = 1;
        exact_config.seed_config.seed_budget = 1;
        exact_config.screen_initial_rho = false;
        exact_config.operator_initial_trust_radius = result.operator_trust_radius;
        exact_config.warm_start_outer_hessian = result.final_hessian.clone();
        log::info!(
            "[OUTER] {context}: sampled derivative pilot completed after {} iteration(s); \
             continuing from its checkpoint on the exact full-data measure",
            pilot_iterations,
        );
        let mut polished = run_outer_uncertified(obj, &exact_config, context)?;
        polished.iterations = polished.iterations.saturating_add(pilot_iterations);
        result = polished;
    }
    // Mandatory analytic optimality certificate (#934): once at the selected
    // point, outside every hot loop, for every solver path and every iteration
    // budget. Missing or failed evidence is typed non-convergence; there is no
    // max-iteration or logging-level bypass.
    //
    // #2273 STALE-TOLERANCE DESYNC RETRY.
    //
    // HISTORY, because the mechanism this was written for no longer exists.
    // The solver's in-loop threshold used to be resolved ONCE from the SEED's
    // cost scale (`rel_cost·(1+|seed_cost|)`) while this certificate re-derived
    // the same formula at the terminal point's own, often far smaller, cost. On
    // a perfectly-separated binomial the score plunges between the oversmoothed
    // heuristic seed and the first accepted step, so the solver declared
    // victory against a bound orders of magnitude looser than the one that then
    // refused it here (measured: |g|=8.1e-1 accepted in-loop vs bound 8.3e-3 at
    // certification, "NOT STATIONARY after 1 outer iteration"; the pass/fail
    // pattern was non-monotone in n because it tracked the seed-to-terminus
    // cost ratio, not identifiability). The retry removed that by construction:
    // re-seeded AT the refused checkpoint, the retry's seed cost IS the
    // certificate's cost.
    //
    // #2613 removed the anchor desync itself. The solver's band
    // (`outer_gradient_tolerance`) is now a function of the declared problem
    // and of nothing else, and this certificate's band
    // (`outer_stationarity_band_at`) is floored at it, so
    // `certificate_bound >= solver_bound` holds identically — a point the
    // solver legitimately converged at CANNOT be refused here for being above
    // its band. `certificate_band_never_undercuts_the_solver_band_2613` gates
    // that invariant directly.
    //
    // What the retry still covers is a different desync with the same shape: a
    // FIDELITY one. Search-time evaluations may run under the inner-PIRLS cap,
    // so the gradient the solver stopped on and the gradient re-measured here
    // at full inner fidelity are not the same number. Re-seeding at the refused
    // checkpoint still collapses that difference, for the same reason. Bounded
    // to a single retry; only fires when the solver CLAIMED convergence (a
    // budget-exhausted result is not a desync — its refusal is genuine).
    // CERTIFICATION-LAST FIT OWNERSHIP. The uncertainty diagnostic evaluates
    // proposal points after theta-hat and the terminal reinstallation
    // re-evaluates at `result.rho`, so any certificate measured BEFORE them
    // describes a state the caller never receives: on a nonconvex profile the
    // certificate-time inner mode and the finally-installed inner mode can sit
    // in different coefficient basins (measured on the cause-specific survival
    // gate as a stable bitwise mismatch, terminal 9.1931e2 vs certified
    // 9.1671e2, because the two paths prime the inner solve under different
    // eval orders). Running the diagnostic and the terminal installation
    // FIRST and certifying LAST makes the certificate's own evaluation the
    // final objective-state installer, so the sealed terminal identity fit
    // assembly binds against IS the certified evidence — bitwise, by
    // construction, independent of basin multiplicity.
    let certify_diagnose_and_install = |obj: &mut dyn OuterObjective,
                                        result: &mut OuterResult|
     -> Result<OuterCriterionCertificate, EstimationError> {
        result.rho_uncertainty_diagnostic = Some(compute_rho_uncertainty_diagnostic(
            obj, config, context, result,
        ));
        // Reinstall the selected point under cap=0 so the certificate below
        // measures the full-fidelity state belonging to `result.rho`, not
        // the diagnostic's final proposal (seeding beta alone does not
        // restore weights, factors, or link state). Reset forces a real
        // installation instead of an LRU value hit.
        let terminal_cap_guard = config
            .outer_inner_cap
            .as_ref()
            .map(TerminalInnerCapGuard::lift);
        // Reset is conditional on the cap contract, mirroring
        // `certify_outer_optimality`'s own doctrine: REML/mixture
        // objectives with a cap can hold a coarse search cache that must
        // not be installed as terminal state, while uncapped stateful
        // objectives (reactive-domain entries among them) retain the very
        // state their evaluation at `result.rho` depends on — an
        // unconditional reset here wiped it and made the certification
        // evaluation non-finite on the reactive fixture.
        //
        // OR-in the terminal-coefficient-mode ownership signal (#2334):
        // objectives that install an owned coefficient mode here but hold
        // their inner cap in a different field (custom families) leave
        // `outer_inner_cap` `None`, so the cap gate alone never fires and
        // `finalize` here could land in a different inner basin than the
        // certifying re-eval below — a spurious bitwise bind failure on a
        // bimodal inner solve. Forcing the reset for mode-owning objectives
        // makes both installations start from the same clean baseline.
        if terminal_cap_guard.is_some() || obj.owns_terminal_coefficient_mode() {
            obj.reset();
        }
        let terminal_installation = obj.finalize_outer_result(&result.rho, &result.plan_used);
        let terminal_inner_converged = inner_solve_converged(config.outer_inner_cap.as_ref());
        drop(terminal_cap_guard);
        terminal_installation?;
        if !terminal_inner_converged {
            return Err(outer_nonconvergence_error(
                context,
                "final outer state installation did not converge at full inner fidelity",
                result,
                result.final_grad_norm,
                StationarityStandard::NoComparison,
            ));
        }
        certify_outer_optimality(obj, config, context, result)
    };
    // Certify-last checkpoint-resume loop (#2273 stale-tolerance desync,
    // generalized by #2374). A solver that CLAIMS convergence but fails the
    // mandatory analytic certificate is re-run once per iteration seeded AT the
    // refused checkpoint — re-anchoring the in-loop tolerance to the terminal
    // cost scale and, for gradient-only outers (`final_hessian == None`),
    // restarting `opt::Bfgs` with a fresh inverse-Hessian metric that breaks the
    // flat-valley `StallPolicy` false stop the accumulated metric crawled into.
    // Looping (rather than the original single retry) matters because that stall
    // gate is inflated by `(1 + ‖ρ‖∞)` in log-λ space, so one fresh-metric
    // reseed rarely lands the optimum in a single hop. The loop stops the moment
    // certification passes; it also stops — regardless of remaining budget —
    // when a reseed fails to strictly reduce the objective, because a point that
    // a fresh-metric restart cannot improve is a genuine non-stationary floor
    // (or a true flat valley), not an exploitable false stall, and further
    // reseeds would only re-derive the same refusal. A result that never claimed
    // convergence (e.g. a budget-exhausted `MaxIterationsReached`) is refused
    // immediately with no reseed: its non-convergence is genuine.
    let mut resumes_remaining = OUTER_CERTIFY_RESUME_BUDGET;
    // Interior strict-saddle escapes are bounded separately and tightly: a real
    // saddle is cleared in one hop, so a handful of attempts is ample, while a
    // non-convergent bimodal-inner grind (#2155/#2363) is cut off well before it
    // exhausts the general resume budget (#2357).
    let mut saddle_escapes_remaining: usize = OUTER_SADDLE_ESCAPE_BUDGET;
    let certificate = loop {
        let claimed_converged = result.solver_claimed_convergence();
        match certify_diagnose_and_install(obj, &mut result) {
            Ok(certificate) => break certificate,
            Err(refusal) => {
                // #2357/#2155 — interior strict-saddle escape. When the refusal
                // is a first-order-stationary point whose reduced (off-railed)
                // Hessian is indefinite, the certificate publishes a
                // negative-curvature reseed stepped strictly BELOW the saddle
                // (`negative_curvature_escape_point`). Reseeding the resume at the
                // refused checkpoint itself would re-descend straight back to that
                // zero-gradient saddle — the #2273/#2374 stale-tolerance resume
                // anchors the tolerance and breaks flat-valley stalls, but it
                // cannot break a genuine saddle. Seed at the escape point instead,
                // off the ridge, and start from a FRESH outer metric so the
                // saddle's indefinite curvature is not transferred into the
                // restart. This is the run_outer-level consumer of the reseed that
                // the multistart-loop consumer (`run_outer_with_plan`) mints only
                // when a per-seed claim is already stationary; the terminal
                // certificate is where stationarity is reached for the binomial
                // link-wiggle families, so without this the reseed was minted and
                // then dropped.
                let saddle_escape_reseed = result.saddle_escape_reseed.take();
                let resume_from_saddle_escape = saddle_escape_reseed.is_some();
                // #2348 Inc 2b, completed (#2349 round 8): a confirmed-tail
                // snap that needs a re-descent publishes the snapped face as
                // `tail_snap_reseed` — previously minted and then DROPPED
                // (declared, set, never consumed), so every ConfirmedNeedsReseed
                // outcome fell through to the plain refusal. The joint tail law
                // is first-order evidence of WHERE the optimum is, so the retry
                // is warranted regardless of the solver's convergence claim,
                // exactly like the saddle-escape reseed (measured on the #2349
                // fixture: the face snap descends 3.86 with |Pg| dropping
                // 2.05 → 0.35; the retry lets over-snapped coordinates relax
                // back to their interior optima while the rest hold the rail).
                let tail_snap_reseed = if resume_from_saddle_escape {
                    result.tail_snap_reseed.take();
                    None
                } else {
                    result.tail_snap_reseed.take()
                };
                let resume_from_tail_snap = tail_snap_reseed.is_some();
                // #2392 — wrong-rail pull-back and active-set reduction reseeds,
                // consumed with LOWER precedence than the saddle/tail-snap
                // reseeds. A higher-precedence reseed DROPS them (take-and-discard)
                // so no stale reseed leaks into a later iteration, exactly as the
                // saddle escape drops a co-minted tail snap above. Both are
                // first-order evidence (a proven inward descent / a poisoned-rail
                // interior), so — like the tail snap — they fire regardless of the
                // solver's convergence claim.
                let higher_precedence_reseed = resume_from_saddle_escape || resume_from_tail_snap;
                let wrong_rail_reseed = if higher_precedence_reseed {
                    result.wrong_rail_reseed.take();
                    None
                } else {
                    result.wrong_rail_reseed.take()
                };
                let resume_from_wrong_rail = wrong_rail_reseed.is_some();
                let active_set_reseed = if higher_precedence_reseed || resume_from_wrong_rail {
                    result.active_set_reseed.take();
                    None
                } else {
                    result.active_set_reseed.take()
                };
                let resume_from_active_set = active_set_reseed.is_some();
                let active_set_rho = active_set_reseed.as_ref().map(|a| a.rho.clone());
                let active_set_bounds = active_set_reseed.map(|a| a.bounds);
                // A published reseed means the refused point IS first-order
                // stationary (the escape mint gate requires `is_stationary`), so
                // it is a genuine saddle escapable regardless of whether the
                // solver "claimed" convergence: for exact-Hessian link-wiggle
                // families the terminal certificate — not the in-loop gate — is
                // where stationarity is first reached, so they arrive here with
                // `converged == false` yet stationary. The #2273/#2374
                // stale-tolerance resume, which reseeds AT the refused checkpoint,
                // still requires a genuine convergence claim (a budget-exhausted
                // non-stationary iterate has no desync to remove).
                if (!claimed_converged
                    && !resume_from_saddle_escape
                    && !resume_from_tail_snap
                    && !resume_from_wrong_rail
                    && !resume_from_active_set)
                    || resumes_remaining == 0
                    || (resume_from_saddle_escape && saddle_escapes_remaining == 0)
                {
                    return Err(refusal);
                }
                resumes_remaining -= 1;
                if resume_from_saddle_escape {
                    // Genuine strict saddles are cleared in one escape (the fresh
                    // ARC step off the ridge descends straight to the PSD
                    // minimum). A SMALL cap stops a pathological objective — e.g. a
                    // bimodal inner solve whose warm re-descent keeps reporting a
                    // phantom improvement that the cold certificate cannot
                    // reproduce (#2155 / #2363) — from burning the whole resume
                    // budget re-escaping a family of shallow saddles that never
                    // certifies. Past the cap the honest refusal is taken.
                    saddle_escapes_remaining -= 1;
                }
                let prior_iterations = result.iterations;
                let prior_value = result.final_value;
                log::info!(
                    "[OUTER] {context}: analytic certification refused after \
                     {prior_iterations} iteration(s) (final_value={prior_value:.6e}); re-running \
                     seeded {} so the in-loop tolerance anchors to the terminal cost scale \
                     ({resumes_remaining} resume(s) left after this one; #2273/#2374/#2155)",
                    if resume_from_saddle_escape {
                        "off the negative-curvature saddle ridge"
                    } else if resume_from_tail_snap {
                        "at the confirmed-tail snapped face"
                    } else if resume_from_wrong_rail {
                        "at the wrong-rail coordinate's clean-band interior scale"
                    } else if resume_from_active_set {
                        "with the poisoned rail frozen so the interior polishes in the reduced box"
                    } else {
                        "at the refused checkpoint"
                    }
                );
                let mut retry_cfg = config.clone();
                retry_cfg.initial_rho = Some(
                    saddle_escape_reseed
                        .or(tail_snap_reseed)
                        .or(wrong_rail_reseed)
                        .or(active_set_rho)
                        .unwrap_or_else(|| result.rho.clone()),
                );
                // Active-set reduction (#2392): the polish runs in the REDUCED
                // (frozen) box so the interior converges without the railed
                // coordinate's ill-conditioned Hessian row poisoning the step. The
                // loop re-certifies the polished point under the ORIGINAL box at
                // the top of the next iteration (`certify_outer_optimality` reads
                // `model_domain_bounds`, which `search_bounds_override` cannot
                // redefine), so the reduction can narrow the SEARCH but never the
                // feasible set a certificate is judged against. `config.clone()`
                // reset each iteration, so the frozen box never persists past
                // this run.
                //
                // A coordinate is only frozen here if the projector had already
                // zeroed it (#2454), i.e. it was on an active constraint when the
                // freeze was taken. That test is at freeze time deliberately: the
                // retry is one-shot, so there is no second reduction to undo a
                // wrong freeze, and the only path back off a frozen bound is the
                // wrong-rail pull-back, which needs a clean opposite-sign
                // exponential tail and declines on any coordinate without one.
                if let Some(frozen_bounds) = active_set_bounds {
                    retry_cfg.search_bounds_override = Some(frozen_bounds);
                }
                retry_cfg.heuristic_lambdas = None;
                retry_cfg.seed_config.max_seeds = 1;
                retry_cfg.seed_config.seed_budget = 1;
                retry_cfg.screen_initial_rho = false;
                // Every reseed kind lands at a genuinely different point, so the
                // refused checkpoint's metric (trust radius, outer Hessian)
                // must not be transferred into the restart.
                let fresh_metric = resume_from_saddle_escape
                    || resume_from_tail_snap
                    || resume_from_wrong_rail
                    || resume_from_active_set;
                retry_cfg.operator_initial_trust_radius = if fresh_metric {
                    None
                } else {
                    result.operator_trust_radius
                };
                retry_cfg.warm_start_outer_hessian = if fresh_metric {
                    None
                } else {
                    result.final_hessian.clone()
                };
                obj.reset();
                match run_outer_uncertified(obj, &retry_cfg, context) {
                    Ok(mut retried) => {
                        retried.iterations = retried.iterations.saturating_add(prior_iterations);
                        // Progress gate. A fresh-metric reseed seeded AT the
                        // checkpoint can only descend from it, so a reduction at
                        // roundoff scale means it found no descent — a genuine
                        // stationary floor — while a false flat-valley stall
                        // yields a reduction orders of magnitude larger. Gate on
                        // roundoff (NOT the coarser cost-stall floor) so a valley
                        // that crawls out in tiny per-reseed steps is not cut off
                        // after one hop; stop only when a reseed truly stalls, so
                        // the next iteration certifies the best point once more
                        // and takes the honest refusal.
                        let improved = certify_resume_made_progress(
                            prior_value,
                            retried.final_value,
                            CERTIFY_RESUME_PROGRESS_REL,
                        );
                        result = retried;
                        if !improved {
                            resumes_remaining = 0;
                        }
                    }
                    // The reseed could not even run (e.g. the checkpoint is a
                    // hard refusal wall for the objective): surface the
                    // certification refusal from the point we started this
                    // iteration at, which carries the checkpoint evidence.
                    Err(_) => return Err(refusal),
                }
            }
        }
    };
    result.criterion_certificate = Some(certificate);
    Ok(result)
}

/// Build a CANONICAL-order copy of an [`OuterConfig`] for the
/// permutation-invariant outer search (#1538/#1539).
///
/// `perm[c]` is the native coordinate at canonical slot `c`. Every
/// per-coordinate config field (initial ρ seed, heuristic-λ seed, per-axis
/// bounds, transferred warm Hessian) is reordered native→canonical so the
/// optimizer's seeding and multistart operate entirely in canonical space;
/// scalar fields are copied verbatim. `rho_canonical_keys` is cleared so the
/// recursive [`run_outer`] frame runs the normal (identity-order) pipeline on
/// the already-canonical objective.
fn canonicalize_outer_config(config: &OuterConfig, perm: &[usize]) -> OuterConfig {
    // Permute a per-coordinate slice native→canonical; pass through any length
    // that does not match the permutation (defensive — should not occur).
    let permute_vec = |v: &[f64]| -> Vec<f64> {
        if v.len() == perm.len() {
            perm.iter().map(|&i| v[i]).collect()
        } else {
            v.to_vec()
        }
    };
    let permute_arr = |a: &Array1<f64>| -> Array1<f64> {
        if a.len() == perm.len() {
            Array1::from_iter(perm.iter().map(|&i| a[i]))
        } else {
            a.clone()
        }
    };
    let mut canonical = config.clone();
    canonical.rho_canonical_keys = None;
    if let Some(initial) = config.initial_rho.as_ref() {
        canonical.initial_rho = Some(permute_arr(initial));
    }
    if let Some(bound) = config.initial_inner_seed.as_ref() {
        canonical.initial_inner_seed = Some(BoundInnerSeed {
            theta: permute_arr(&bound.theta),
            beta: bound.beta.clone(),
        });
    }
    if let Some(h) = config.heuristic_lambdas.as_ref() {
        canonical.heuristic_lambdas = Some(permute_vec(h));
    }
    if let Some((lower, upper)) = config.model_domain_bounds.as_ref() {
        canonical.model_domain_bounds = Some((permute_arr(lower), permute_arr(upper)));
    }
    if let Some((lower, upper)) = config.search_bounds_override.as_ref() {
        canonical.search_bounds_override = Some((permute_arr(lower), permute_arr(upper)));
    }
    // A transferred dense outer Hessian is in native coordinate order; permute
    // it into canonical order so the BFGS warm metric stays aligned. (None on
    // the cold-start canonicalized path, so this is usually a no-op.)
    if let Some(h) = config.warm_start_outer_hessian.as_ref()
        && h.nrows() == perm.len()
        && h.ncols() == perm.len()
    {
        let mut hc = Array2::<f64>::zeros((perm.len(), perm.len()));
        for (a, &ia) in perm.iter().enumerate() {
            for (b, &ib) in perm.iter().enumerate() {
                hc[[a, b]] = h[[ia, ib]];
            }
        }
        canonical.warm_start_outer_hessian = Some(hc);
    }
    canonical
}

/// The solver ladder behind [`run_outer`], without the #934 self-audit.
pub(crate) fn run_outer_uncertified(
    obj: &mut dyn OuterObjective,
    config: &OuterConfig,
    context: &str,
) -> Result<OuterResult, EstimationError> {
    let cap = primary_capability_for_config(obj.capability(), config, context);
    cap.validate_layout(context)?;
    // #2370: reject a degenerate / inverted ρ-box up front, as a typed error.
    // Every downstream stage — the per-atom EFS path below and
    // `run_outer_with_plan` — projects seeds against these bounds with
    // `f64::clamp`, whose `min > max` (or NaN) precondition panics *inside the
    // Rust boundary* and surfaces as an opaque `GamError: ... panicked` across
    // the FFI, violating the fail-loudly contract. The configured box can invert
    // whenever an independently-derived upper bound drifts below the lower wall
    // (e.g. the custom-family effective-df ceiling vs. `rho_lower_bound`).
    // Validating the *effective* template here — the same one every consumer
    // reads — turns any such inversion into `EstimationError::InvalidInput`
    // regardless of how the bounds were constructed.
    {
        let (model_lo, model_hi) =
            outer_model_domain_bounds_template(config, cap.n_params);
        let (bound_lo, bound_hi) = outer_search_bounds_template(config, cap.n_params);
        if model_lo.len() != cap.n_params
            || model_hi.len() != cap.n_params
            || bound_lo.len() != cap.n_params
            || bound_hi.len() != cap.n_params
        {
            return Err(EstimationError::InvalidInput(format!(
                "{context}: outer bound dimension mismatch: parameters={},                  model_lower={}, model_upper={}, search_lower={}, search_upper={}",
                cap.n_params,
                model_lo.len(),
                model_hi.len(),
                bound_lo.len(),
                bound_hi.len(),
            )));
        }
        for i in 0..cap.n_params {
            if !(model_lo[i].is_finite() && model_hi[i].is_finite())
                || model_lo[i] > model_hi[i]
            {
                return Err(EstimationError::InvalidInput(format!(
                    "{context}: outer model-domain bounds are invalid at coordinate {i}:                      lower={}, upper={}",
                    model_lo[i], model_hi[i]
                )));
            }
            if bound_lo[i] < model_lo[i] || bound_hi[i] > model_hi[i] {
                return Err(EstimationError::InvalidInput(format!(
                    "{context}: outer search bounds escape the model domain at coordinate {i}:                      model=[{}, {}], search=[{}, {}]",
                    model_lo[i], model_hi[i], bound_lo[i], bound_hi[i]
                )));
            }
            if !(bound_lo[i].is_finite() && bound_hi[i].is_finite()) {
                return Err(EstimationError::InvalidInput(format!(
                    "{context}: outer rho bounds are non-finite at coordinate {i}: \
                     lower={}, upper={}",
                    bound_lo[i], bound_hi[i]
                )));
            }

            // Report a collapsed interval with BOTH walls. `outer_bounds` below
            // is the backstop and rejects the same condition, but its message
            // names only the coordinate. The panic this guard replaced printed
            // `min = -10.0, max = -11.855421656441532`, and those two numbers
            // are what made #2370 diagnosable from a bug report alone: they
            // identify WHICH pair of independently-derived bounds drifted, and
            // by how much. A typed error must not be a weaker diagnostic than
            // the panic it replaced.
            //
            // Two tests constrain this string: `inverted_rho_box_is_a_typed_
            // error_not_a_clamp_panic_2370` greps for the word "bound", and
            // `the_inverted_box_refusal_carries_both_bound_values_2370` pins
            // both numeric walls. Keep both when rewording.
            if bound_lo[i] > bound_hi[i] {
                return Err(EstimationError::InvalidInput(format!(
                    "{context}: outer rho bounds are inverted at coordinate {i}: \
                     lower bound {} exceeds upper bound {}",
                    bound_lo[i], bound_hi[i]
                )));
            }
        }
        outer_bounds(&bound_lo, &bound_hi)
            .map_err(|err| EstimationError::InvalidInput(format!("{context}: {err}")))?;
    }
    if let Some(initial_rho) = config.initial_rho.as_ref() {
        cap.theta_layout()
            .validate_point_len(initial_rho, "initial outer seed")
            .map_err(|err| match err {
                err => {
                    let message = err.into_message();
                    EstimationError::RemlOptimizationFailed(format!("{context}: {message}"))
                }
            })?;
    }
    // Frontier ρ-scaling auto-switch (#986): at per-atom-EFS-eligible frontier
    // rho dimension the decoupled per-atom fixed point is the primary outer
    // iteration; everything else falls through to the dense / standard path
    // below. Routed here so every entry point inherits it (magic by default).
    if let Some(result) = run_per_atom_efs_if_frontier(obj, config, context)? {
        if result.solver_claimed_convergence() {
            return Ok(result);
        }
        return Err(outer_nonconvergence_error(
            context,
            "per-atom EFS exhausted its iteration budget before the fixed-point step converged",
            &result,
            None,
            StationarityStandard::NoComparison,
        ));
    }

    if cap.n_params == 0 {
        let cost = obj.eval_cost(&Array1::zeros(0))?;
        let the_plan = plan(&cap);
        let mut result =
            outer_result_with_gradient_norm(Array1::zeros(0), cost, 0, Some(0.0), true, the_plan);
        result.origin = OuterResultOrigin::EmptyParameterSpace;
        return Ok(result);
    }

    // Build the ordered list of capabilities to attempt: primary first, then
    // any centrally-derived degraded capabilities. Aux direct-search has no
    // degraded ladder — a single attempt either succeeds or the failure is
    // surfaced to the caller.
    let fallback_attempts = match config.fallback_policy {
        FallbackPolicy::Automatic => automatic_fallback_attempts(&cap),
        FallbackPolicy::Disabled => Vec::new(),
    };
    let mut attempts: Vec<OuterCapability> = Vec::with_capacity(1 + fallback_attempts.len());
    attempts.push(cap.clone());
    for degraded in fallback_attempts {
        attempts.push(degraded);
    }

    let mut last_error: Option<EstimationError> = None;
    let mut best_checkpoint: Option<OuterResult> = None;

    for (attempt_idx, attempt_cap) in attempts.iter().enumerate() {
        let the_plan = plan(attempt_cap);
        if attempt_idx > 0 {
            log::debug!("[OUTER] {context}: primary plan failed; falling back to {the_plan}");
        }
        log_plan(context, attempt_cap, &the_plan);

        obj.reset();

        // ARC budget-exhaustion retry: when an Arc attempt runs out of
        // outer iterations, reseed a fresh Arc run from the previous
        // attempt's last ρ and trust radius. Inner caches (PIRLS LRU,
        // eval bundle, warm-start predictor, adaptive signals) are wiped
        // by `obj.reset()`; the operator-TR's Cauchy/Newton/CG state has
        // no resume API and is not preserved. The lever that changes for
        // the resumed run is the inner-PIRLS cap (uncapped via the
        // feedback handle), not `max_iter` — empirically the prior stall
        // was an inner-tolerance / model-fidelity issue, not an outer
        // budget shortfall, and doubling `max_iter` only replays the
        // same trajectory byte-for-byte. The retry is gated on observed
        // `‖g‖` progress so trajectories that made no headway fall
        // through to the degraded plan instead of replaying.
        let mut arc_retries_left: u32 = if matches!(the_plan.solver, Solver::Arc) {
            2
        } else {
            0
        };
        let mut retry_config: Option<OuterConfig> = None;
        // Tracks the previous ARC attempt's terminal `‖g‖`. The retry
        // gate compares attempt-over-attempt: if a retry didn't move
        // the gradient norm, the trajectory replayed (same seed, same
        // trust radius, cold caches, deterministic optimizer) and
        // further retries cannot help. First retry is unconditional
        // (no prior attempt to compare against).
        let mut prev_attempt_grad_norm: Option<f64> = None;

        let outcome = loop {
            // Bind the active config by cloning into a local owned value so
            // subsequent retry-config assignment does not collide with the
            // borrow used inside this iteration body.
            let active_config_owned: OuterConfig =
                retry_config.clone().unwrap_or_else(|| config.clone());
            let active_config: &OuterConfig = &active_config_owned;
            match run_outer_with_plan(obj, active_config, context, attempt_cap, &the_plan, true) {
                Ok(PlanRunOutcome::Converged(result)) => break Ok(result),
                Ok(PlanRunOutcome::Exhausted(result)) => {
                    if arc_retries_left == 0
                        || matches!(
                            result.operator_stop_reason,
                            Some(
                                OperatorTrustRegionStopReason::RejectFloor
                                    // #1690: a flat-valley cost-stall is a CONVERGED
                                    // cost plateau over the whole stall window, not a
                                    // budget shortfall. The ARC retry only reseeds
                                    // from the same last ρ with a reset trust radius
                                    // and the same deterministic operator state, so it
                                    // replays the identical trajectory and re-halts at
                                    // the same valley floor with the same |g| (verified
                                    // on the #1690 Gamma repro: two retries, each
                                    // returning |g|=0.3646 byte-for-byte). Treat it
                                    // like `RejectFloor` and stop — the genuine
                                    // stationarity verdict is reconciled downstream
                                    // against the authoritative shipped-β gradient
                                    // (`optimizer.rs`), and a non-stationary floor is
                                    // still reported non-converged. This skips the
                                    // wasted full-trajectory replay that dominated the
                                    // count-family slowdown.
                                    | OperatorTrustRegionStopReason::CostStallFlatValley
                            )
                        )
                    {
                        break Ok(result);
                    }
                    // Gate the retry on attempt-over-attempt `‖g‖`
                    // progress. The first retry is unconditional (no
                    // prior attempt). Subsequent retries fall through
                    // to the degraded plan when the gradient norm did
                    // not materially shrink — the deterministic
                    // optimizer with the same seed and trust radius
                    // would replay the same trajectory.
                    let Some(cur_grad_norm) = result.final_grad_norm else {
                        log::info!(
                            "[OUTER] {context}: ARC attempt exhausted budget at \
                             iter={} cost={:.6e} without a final gradient norm; \
                             falling through to degraded plan",
                            result.iterations,
                            result.final_value,
                        );
                        break Ok(result);
                    };
                    if let Some(prev_g) = prev_attempt_grad_norm {
                        let progressed = cur_grad_norm.is_finite()
                            && prev_g.is_finite()
                            && cur_grad_norm < 0.5 * prev_g;
                        if !progressed {
                            log::info!(
                                "[OUTER] {context}: ARC retry stalled at \
                                 iter={} cost={:.6e} |g|={:.6e} (prev |g|={:.6e}); \
                                 deterministic replay suspected, falling through \
                                 to degraded plan",
                                result.iterations,
                                result.final_value,
                                cur_grad_norm,
                                prev_g,
                            );
                            break Ok(result);
                        }
                    }
                    let next_trust_radius =
                        sanitized_operator_trust_restart_radius(result.operator_trust_radius);
                    log::info!(
                        "[OUTER] {context}: ARC attempt exhausted budget at \
                         iter={} cost={:.6e} |g|={:.6e}; resuming from last \
                         rho + trust_radius={:?}, inner-PIRLS uncapped \
                         (objective caches wiped; operator-TR Cauchy/Newton \
                         state is not resumable)",
                        result.iterations,
                        result.final_value,
                        cur_grad_norm,
                        next_trust_radius,
                    );
                    // Snapshot the cap-feedback handle before we
                    // reassign `retry_config` (which currently backs
                    // `active_config`'s borrow). `InnerProgressFeedback`
                    // is an Arc-wrapper bundle, so the clone is cheap.
                    let cap_feedback = active_config.outer_inner_cap.clone();
                    let mut next = active_config.clone();
                    prev_attempt_grad_norm = Some(cur_grad_norm);
                    next.initial_rho = Some(result.rho.clone());
                    next.operator_initial_trust_radius = next_trust_radius;
                    retry_config = Some(next);
                    arc_retries_left -= 1;
                    obj.reset();
                    // Lift any inner-PIRLS cap for the resumed run. The
                    // schedule's cold-start ladder (3/5/10) would
                    // re-coarsen exactly the inner solves whose tolerance
                    // is suspected to have starved the prior trajectory.
                    // The next outer iter consumes ρ near a near-stationary
                    // point where exact β / gradient / Hessian is the
                    // load-bearing input to the operator-TR geometry.
                    if let Some(feedback) = cap_feedback.as_ref() {
                        feedback.cap.store(0, Ordering::Relaxed);
                    }
                }
                Err(e) => break Err(e),
            }
        };

        match outcome {
            Ok(result) => {
                if result.solver_claimed_convergence() {
                    return Ok(result);
                }

                let improves_checkpoint = result.final_value.is_finite()
                    && best_checkpoint.as_ref().is_none_or(|checkpoint| {
                        !checkpoint.final_value.is_finite()
                            || result.final_value < checkpoint.final_value
                    });
                if improves_checkpoint {
                    best_checkpoint = Some(result);
                }

                let message = format!(
                    "{context}: attempt {} (plan={the_plan}) exhausted without convergence",
                    attempt_idx + 1
                );
                log::debug!("[OUTER] {message}; trying degraded fallback plan");
                last_error = Some(EstimationError::RemlOptimizationFailed(message));
            }
            Err(e) => {
                if e.is_fatal_outer_evaluation() {
                    return Err(e);
                }
                log::debug!(
                    "[OUTER] {context}: attempt {} (plan={the_plan}) failed: {e}",
                    attempt_idx + 1
                );
                last_error = Some(e);
            }
        }
    }

    if let Some(checkpoint) = best_checkpoint {
        // The solver ladder produced no result that its OWN internal
        // (raw-gradient) convergence test accepted — but that test cannot see a
        // railed or already-stationary optimum. At a smoothing parameter railed to
        // the ρ box floor (λ→0, e.g. an exact linear fit or a separated smooth),
        // the RAW gradient stays large along the railed axis — it "wants" to push
        // past the boundary — so the solver reports non-convergence and can take
        // zero steps, even though the KKT-PROJECTED gradient (which zeroes
        // outward-railed axes) is stationary and no feasible step reduces the
        // objective. Only the mandatory analytic certificate in `run_outer`
        // computes that projected gradient AND the curvature-scaled flat-valley
        // bound (½·gᵀH⁻¹g ≤ objective_tol), so IT, not this raw-gradient ladder, is
        // the sole authority on stationarity. Hand it the best finite checkpoint:
        // `certify_outer_optimality` mints iff the point is genuinely stationary
        // (interior, railed, or flat-valley) and returns typed non-convergence
        // otherwise, so a truly divergent fit is still rejected there.
        return Ok(checkpoint);
    }

    Err(last_error.unwrap_or_else(|| {
        EstimationError::RemlOptimizationFailed(format!("all plan attempts exhausted ({context})"))
    }))
}

// ─── Frontier ρ-scaling auto-switch (issue #986) ─────────────────────────
//
// ARD-per-atom assigns one smoothing coordinate per dictionary atom, so the
// ρ-vector reaches 10^4–10^5 coordinates. A dense outer quasi-Newton over that
// materializes an O(K²) Hessian and is impossible at scale. When the ρ-dimension
// is frontier-scale AND every coordinate is penalty-like with a working
// fixed-point hook, route the PRIMARY outer iteration to the per-atom decoupled
// EFS path (`crate::estimate::reml::per_atom_efs`) instead of the dense
// ARC/BFGS lane. The decision is auto-derived from the coordinate count alone —
// there is no flag — and it is additive: the dense path is unchanged for small K
// and for any objective that is not per-atom-EFS-eligible.

/// Whether this capability is in the frontier ρ-scaling regime where the
/// per-atom decoupled EFS primary should take over from the dense outer.
///
/// Delegates the eligibility decision to
/// [`crate::estimate::reml::per_atom_efs::per_atom_efs_eligible`], which
/// requires all-penalty-like coordinates, a working `eval_efs` hook,
/// fixed-point not disabled, and a frontier-scale ρ-dimension. This is the
/// single auto-switch predicate; `plan` keeps selecting the
/// dense or standard-EFS solver for everything below the frontier threshold.
pub fn is_per_atom_efs_frontier(cap: &OuterCapability) -> bool {
    crate::estimate::reml::per_atom_efs::per_atom_efs_eligible(cap)
}

/// Auto-switch entry point: when `cap` is frontier-scale per-atom-EFS-eligible,
/// run the per-atom decoupled EFS primary and return its [`OuterResult`];
/// otherwise return `Ok(None)` so the caller falls through to the existing dense
/// / standard-EFS path via [`OuterProblem::run`] / [`run_outer`].
///
/// Builds the same bounded seed and tolerance/budget the standard plan path
/// uses, picks the seed (initial-ρ if supplied, else the first generated
/// candidate — the per-atom fixed point is a contraction near the optimum and
/// does not need the multi-seed cascade the dense path runs for its non-convex
/// quasi-Newton surface), then drives the per-atom EFS loop. The shared-border
/// topology defaults to disjoint (every atom owns a private penalty block — the
/// common ARD-per-atom case); callers with a known arrow-border overlap can run
/// the module's `run_per_atom_efs` directly with a populated
/// `SharedBorderTopology`.
///
/// Additive: this function neither mutates nor bypasses the dense path; it is
/// the pre-dispatch shortcut [`run_outer`] calls before the dense ladder.
pub(crate) fn run_per_atom_efs_if_frontier(
    obj: &mut dyn OuterObjective,
    config: &OuterConfig,
    context: &str,
) -> Result<Option<OuterResult>, EstimationError> {
    let cap = primary_capability_for_config(obj.capability(), config, context);
    cap.validate_layout(context)?;
    if !is_per_atom_efs_frontier(&cap) {
        return Ok(None);
    }

    let the_plan = plan(&cap);
    let rho_dim = cap.theta_layout().rho_dim();

    let model_domain_bounds = outer_model_domain_bounds_template(config, cap.n_params);
    crate::estimate::reml::outer_eval::record_current_outer_rho_model_upper_bounds_for_ift(
        &model_domain_bounds.1,
    );
    let (lower, upper) = outer_search_bounds_template(config, cap.n_params);

    // Seed: cache/explicit initial ρ if present, otherwise the first generated
    // candidate. The per-atom multiplicative fixed point is locally
    // contractive, so a single seed suffices; the heavy multi-seed cascade
    // exists for the dense quasi-Newton's non-convex surface, not for EFS.
    let seed = match config.initial_rho.as_ref() {
        Some(initial) if initial.len() == cap.n_params => initial.clone(),
        _ => {
            let generated = crate::seeding::generate_rho_candidates(
                cap.n_params,
                config.heuristic_lambdas.as_deref(),
                &config.seed_config,
            )?;
            match generated.into_iter().next() {
                Some(first) => first,
                None => Array1::<f64>::zeros(cap.n_params),
            }
        }
    };

    log::info!(
        "[OUTER] {context}: frontier ρ-scaling (rho_dim={rho_dim}) → per-atom decoupled EFS primary"
    );

    let pa_cfg = crate::estimate::reml::per_atom_efs::PerAtomEfsConfig::new(
        config.tolerance,
        config.max_iter,
        lower,
        upper,
    );
    let topology = crate::estimate::reml::per_atom_efs::SharedBorderTopology::disjoint(rho_dim);

    obj.reset();
    install_matching_initial_inner_seed(obj, config, &seed, context)?;
    let result =
        crate::estimate::reml::per_atom_efs::run_per_atom_efs(obj, &seed, &pa_cfg, &topology)?;
    Ok(Some(result.into_outer_result(the_plan)))
}

#[cfg(test)]
#[path = "inverted_rho_box_tests.rs"]
mod inverted_rho_box_tests;

pub(crate) fn outer_bounds(lo: &Array1<f64>, hi: &Array1<f64>) -> Result<Bounds, EstimationError> {
    Bounds::new(lo.clone(), hi.clone(), 1e-6).map_err(|err| {
        EstimationError::InvalidInput(format!("outer rho bounds are invalid: {err}"))
    })
}

pub(crate) fn outer_model_domain_bounds_template(
    config: &OuterConfig,
    n: usize,
) -> (Array1<f64>, Array1<f64>) {
    config.model_domain_bounds.clone().unwrap_or_else(|| {
        (
            Array1::<f64>::from_elem(n, -config.rho_bound),
            Array1::<f64>::from_elem(n, config.rho_bound),
        )
    })
}

pub(crate) fn outer_search_bounds_template(
    config: &OuterConfig,
    n: usize,
) -> (Array1<f64>, Array1<f64>) {
    config
        .search_bounds_override
        .clone()
        .unwrap_or_else(|| outer_model_domain_bounds_template(config, n))
}

/// Intersect typed objective-domain faces with the caller's declared model
/// domain. The resulting box is the immutable feasible set used by every
/// stationarity certificate. Algorithmic active-set reduction is represented
/// separately by `search_bounds_override` and cannot redefine this domain.
pub(super) fn install_objective_domain(
    config: &mut OuterConfig,
    n_params: usize,
    objective_lower: Option<Array1<f64>>,
    objective_upper: Option<Array1<f64>>,
) -> Result<(), EstimationError> {
    let (mut lower, mut upper) = outer_model_domain_bounds_template(config, n_params);
    if lower.len() != n_params || upper.len() != n_params {
        return Err(EstimationError::InvalidInput(format!(
            "outer configured bounds dimension mismatch: parameters={n_params}, lower={}, upper={}",
            lower.len(),
            upper.len(),
        )));
    }
    if let Some(domain) = objective_lower.as_ref()
        && domain.len() != n_params
    {
        return Err(EstimationError::InvalidInput(format!(
            "outer objective-domain lower-bound dimension mismatch: parameters={n_params}, lower={}",
            domain.len()
        )));
    }
    if let Some(domain) = objective_upper.as_ref()
        && domain.len() != n_params
    {
        return Err(EstimationError::InvalidInput(format!(
            "outer objective-domain upper-bound dimension mismatch: parameters={n_params}, upper={}",
            domain.len()
        )));
    }
    for index in 0..n_params {
        if let Some(domain) = objective_lower.as_ref() {
            let value = domain[index];
            if !value.is_finite() {
                return Err(EstimationError::InvalidInput(format!(
                    "outer objective-domain lower bound[{index}] must be finite; got {value}"
                )));
            }
            lower[index] = lower[index].max(value);
        }
        if let Some(domain) = objective_upper.as_ref() {
            let value = domain[index];
            if !value.is_finite() {
                return Err(EstimationError::InvalidInput(format!(
                    "outer objective-domain upper bound[{index}] must be finite; got {value}"
                )));
            }
            upper[index] = upper[index].min(value);
        }
        if !(lower[index].is_finite() && upper[index].is_finite() && lower[index] < upper[index]) {
            return Err(EstimationError::InvalidInput(format!(
                "outer objective-domain intersection is empty or non-finite at coordinate {index}: lower={}, upper={}",
                lower[index], upper[index]
            )));
        }
    }
    config.model_domain_bounds = Some((lower, upper));
    config.search_bounds_override = None;
    Ok(())
}

pub(crate) fn outer_tolerance(value: f64) -> Result<Tolerance, EstimationError> {
    Tolerance::new(value)
        .map_err(|err| EstimationError::InvalidInput(format!("outer tolerance is invalid: {err}")))
}

/// The relative cost floor shared by the cost-stall guard, the curvature-scaled
/// flat-valley certificate, and the certify-last resume progress gate: nothing
/// tighter than what the in-loop stall detector already proved about the
/// surface. `rel_cost_tolerance` when set, else a small fraction of the absolute
/// tolerance, never below `COST_STALL_REL_TOL_FLOOR`.
pub(crate) fn outer_rel_cost_floor(config: &OuterConfig) -> f64 {
    config
        .rel_cost_tolerance
        .unwrap_or(config.tolerance * 1.0e-2)
        .max(COST_STALL_REL_TOL_FLOOR)
}

/// Whether a certify-last checkpoint reseed (#2273/#2374) exploited real descent.
///
/// A reseed that does not strictly reduce the outer objective past the shared
/// relative cost floor `rel_cost_floor·(1 + min(|prior|, |retried|))` is at a
/// genuine non-stationary floor (or a true flat valley) a fresh metric cannot
/// escape, so the resume loop must stop rather than spend its remaining budget
/// re-deriving the same refusal. Anchoring the floor on the SMALLER of the two
/// costs keeps a tiny uphill wobble from a metric restart from reading as
/// progress, and a non-finite retried value is never progress.
pub(crate) fn certify_resume_made_progress(
    prior_value: f64,
    retried_value: f64,
    rel_cost_floor: f64,
) -> bool {
    let floor = rel_cost_floor * (1.0 + prior_value.abs().min(retried_value.abs()));
    retried_value.is_finite() && retried_value < prior_value - floor
}

/// The user-requested outer precision, expressed relative to the criterion's
/// magnitude — the mgcv `magic` rule `‖g‖ ≤ τ·(1 + |V|)`.
///
/// This is a CONVERGENCE tolerance, not a resolution floor: at the default
/// `τ = 1e-5` it sits ~670× above the arithmetic floor `√ε`. What it needs from
/// the caller is `|V|`, and the whole content of #2613 is *which* `|V|`.
#[inline]
pub(crate) fn outer_cost_relative_tolerance(config: &OuterConfig) -> f64 {
    config.rel_cost_tolerance.unwrap_or(config.tolerance)
}

/// The arithmetic resolution of the declared objective scale.
///
/// A matrix-factorization REML/LAML score cannot resolve relative perturbations
/// below the forward-error scale √ε. Requiring a smaller absolute residual made
/// gradient-only / operator-curvature objectives impossible to certify unless
/// an unrelated Hessian or probe-noise rescue happened to be available (#2269).
#[inline]
fn outer_arithmetic_gradient_floor(config: &OuterConfig) -> f64 {
    config
        .objective_scale
        .map(|scale| config.tolerance.max(scale * f64::EPSILON.sqrt()))
        .unwrap_or(config.tolerance)
}

/// The stationarity band handed to the SOLVER, and to the cost-stall guard's
/// stationarity gate.
///
/// It is a function of the DECLARED problem and of nothing else. `opt` resolves
/// a `GradientTolerance` exactly once, at run start, against the seed cost —
/// so handing it a `rel_cost` component makes the band a function of where a
/// seed happened to land. On #2392's exponentially stiff recovery that produced
/// an eighteen-order spread across the seeds of ONE fit: a generated lattice
/// seed at `ρ = 1.0`, where the criterion is `1.79e13`, gave
///
///     termination=gradient_tolerance(|g|=1.522998e-4 < 1.792397e8)
///
/// i.e. the solver claimed convergence on the wrong rail against a threshold no
/// gradient can fail. A stationarity test that depends on the starting point is
/// not a stationarity test: two seeds converging to the same optimum must reach
/// the same verdict.
///
/// So the cost-relative term is anchored on `objective_scale` — a property of
/// the data (`n_obs` on the routes that set it), fixed for the whole fit. On
/// those routes this is magnitude-preserving, because a REML/LAML score is a
/// sum over `n` rows and `1 + |V| = O(n)` is what `1 + scale` already says.
///
/// When no scale is declared, gam does not know the criterion's magnitude, and
/// the honest band is the absolute tolerance the caller asked for. It does NOT
/// silently substitute a trajectory point for the thing it does not know. The
/// consequence — an outer loop that keeps stepping past the point a
/// score-relative band would have stopped at — is bounded by the cost-stall
/// guard, whose own score-relative rung
/// (`flat_valley_converged_grad_bound(best_value)`) is anchored at the BEST
/// iterate and is therefore the correctly-anchored version of the same idea.
///
/// The certificate keeps the point-anchored form, which is what mgcv means:
/// see [`outer_stationarity_band_at`].
pub(crate) fn outer_gradient_tolerance(config: &OuterConfig) -> GradientTolerance {
    let mut abs = outer_arithmetic_gradient_floor(config);
    if let Some(scale) = config.objective_scale {
        abs = abs.max(outer_cost_relative_tolerance(config) * (1.0 + scale));
    }
    GradientTolerance {
        abs,
        rel_initial_grad: None,
        // Never delegated: `opt`'s only anchor is the seed. See above.
        rel_cost: None,
        projected: true,
    }
}

/// The stationarity band a CERTIFICATE applies at the point it is judging.
///
/// Same formula, correct anchor: `cost_at_point` is the criterion value of the
/// candidate optimum, which is what the mgcv `magic` rule means by `f₀` and
/// what every consumer of a certificate reads the bound as. Unlike the solver's
/// band this one is resolved per point, so it costs nothing to anchor it right.
///
/// Floored at the SOLVER's band by construction. A certificate tighter than the
/// threshold the optimizer was told to reach manufactures the "solver claimed
/// convergence, certificate refused" family out of nothing but a disagreement
/// between two spellings of one tolerance: the solver stops exactly where it
/// was asked to and the certificate then declares the stop illegitimate. The
/// certificate may be LOOSER — that is what the score-relative widening is for
/// — but never stricter.
pub(crate) fn outer_stationarity_band_at(config: &OuterConfig, cost_at_point: f64) -> f64 {
    let solver_band = outer_gradient_tolerance(config).abs;
    if !cost_at_point.is_finite() {
        return solver_band;
    }
    solver_band.max(outer_cost_relative_tolerance(config) * (1.0 + cost_at_point.abs()))
}

pub(crate) fn outer_max_iterations(value: usize) -> Result<MaxIterations, EstimationError> {
    MaxIterations::new(value)
        .map_err(|err| EstimationError::InvalidInput(format!("outer max_iter is invalid: {err}")))
}

pub(crate) fn sanitized_operator_trust_restart_radius(radius: Option<f64>) -> Option<f64> {
    radius
        .filter(|value| value.is_finite() && *value > 0.0)
        .map(|value| value.max(OPERATOR_TRUST_RESTART_RADIUS_FLOOR))
}

pub(crate) fn bfgs_axis_step_caps(
    config: &OuterConfig,
    layout: OuterThetaLayout,
) -> Option<Array1<f64>> {
    if config.bfgs_step_cap.is_none() && config.bfgs_step_cap_psi.is_none() {
        return None;
    }
    let mut caps = Array1::from_elem(layout.n_params, f64::INFINITY);
    if let Some(cap) = config.bfgs_step_cap {
        for i in 0..layout.rho_dim() {
            caps[i] = cap;
        }
    }
    if let Some(cap) = config.bfgs_step_cap_psi {
        for i in layout.rho_dim()..layout.n_params {
            caps[i] = cap;
        }
    }
    Some(caps)
}

pub(crate) enum FixedPointOuterRunError {
    SeedRejected(EstimationError),
    ImmediateFallback(EstimationError),
    Failed(EstimationError),
}

pub(crate) fn run_fixed_point_outer_solver(
    obj: &mut dyn OuterObjective,
    layout: OuterThetaLayout,
    barrier_config: Option<BarrierConfig>,
    config: &OuterConfig,
    context: &str,
    seed: &Array1<f64>,
    the_plan: OuterPlan,
    label: &str,
    failure_prefix: &str,
) -> Result<OuterResult, FixedPointOuterRunError> {
    // Shared publication slot for the recurrent-restored-incumbent stop
    // (#2235 verdict 2): the bridge is moved into the driver, so the streak
    // count comes back through this cell and is stamped onto the returned
    // `OuterResult` below.
    let recurrent_incumbent_exit = Arc::new(Mutex::new(None));
    let mut objective = OuterFixedPointBridge {
        obj,
        layout,
        barrier_config,
        fixed_point_tolerance: config.tolerance,
        consecutive_psi_zero_iters: 0,
        last_restored_incumbent_streak: None,
        recurrent_incumbent_exit: Arc::clone(&recurrent_incumbent_exit),
    };
    let seed_sample = match objective.eval_step(seed) {
        Ok(sample) => sample,
        Err(err) if err.is_recoverable() => {
                        let message = err.into_message();
            let err = EstimationError::RemlOptimizationFailed(message);
            if requests_immediate_first_order_fallback(&err.to_string()) {
                return Err(FixedPointOuterRunError::ImmediateFallback(err));
            }
            return Err(FixedPointOuterRunError::SeedRejected(err));
        }
        Err(err) => {
                        let message = err.into_message();
            return Err(FixedPointOuterRunError::Failed(
                EstimationError::fatal_outer_evaluation(
                    "outer fixed-point seed evaluation",
                    EstimationError::RemlOptimizationFailed(message),
                ),
            ));
        }
    };
    let (lo, hi) = outer_search_bounds_template(config, layout.n_params);
    let bounds = outer_bounds(&lo, &hi).map_err(FixedPointOuterRunError::Failed)?;
    let tol = outer_tolerance(config.tolerance).map_err(FixedPointOuterRunError::Failed)?;
    let max_iter =
        outer_max_iterations(config.max_iter).map_err(FixedPointOuterRunError::Failed)?;
    let mut optimizer = FixedPoint::new(seed.clone(), objective)
        // Seed validation already paid the complete EFS inner solve. Reuse that
        // exact sample so iteration zero neither repeats the expensive solve nor
        // mistakes two evaluations at the identical rho for recurrent incumbent
        // evidence (#2241).
        .with_initial_sample(seed.clone(), seed_sample)
        .with_bounds(bounds)
        .with_tolerance(tol)
        .with_max_iterations(max_iter);
    match optimizer.run() {
        Ok(sol) => {
            let mut result = solution_into_outer_result(sol, true, the_plan);
            // Stamp the model-state fixed-point stop when the bridge published
            // one; `None` means the walk stopped through the ordinary
            // step-norm test instead.
            if let Some(consecutive_restores) =
                recurrent_incumbent_exit.lock().ok().and_then(|slot| *slot)
            {
                result.termination = OuterTermination::SolverClaimed {
                    proposed_via: Some(OuterConvergedVia::RecurrentIncumbent {
                        consecutive_restores,
                    }),
                };
            }
            Ok(result)
        }
        Err(FixedPointError::MaxIterationsReached { last_solution }) => {
            log::warn!(
                "[OUTER warning] {context}: {label} hit max_iter={} at final_value={:.6e} step_norm={:.3e}",
                config.max_iter,
                last_solution.final_value,
                last_solution.final_gradient_norm.unwrap_or(f64::NAN),
            );
            Ok(solution_into_outer_result(*last_solution, false, the_plan))
        }
        Err(FixedPointError::ObjectiveFailed { message })
            if requests_immediate_first_order_fallback(&message) =>
        {
            // The bridge raises `EFS_FIRST_ORDER_FALLBACK_MARKER` when the
            // fixed-point step is not a descent direction it can rescue — a
            // ψ-stagnation streak, or a step every halving rejected on both
            // the full vector and the ρ/τ-only fallback. That marker is a
            // ROUTING REQUEST ("abandon the fixed point, run the joint
            // gradient solver that enforces ∇_ψ V = 0"), not a defect: the
            // ladder `automatic_fallback_attempts` builds for an
            // analytic-gradient EFS/HybridEFS primary is exactly the
            // `disable_fixed_point` BFGS plan it is asking for.
            //
            // The seed evaluation above already honours it. Every LATER
            // iteration reached this arm instead, where the blanket
            // `fatal_outer_evaluation` classification short-circuits both
            // `run_outer_with_plan`'s own marker check and the attempt loop in
            // `run_outer_with_strategy` — so a HybridEFS search that descended
            // and then asked to hand over died fatally with the fallback plan
            // never attempted. Carry the same recoverable routing the seed
            // path uses, so the request is honoured wherever it is raised.
            Err(FixedPointOuterRunError::ImmediateFallback(
                EstimationError::RemlOptimizationFailed(message),
            ))
        }
        Err(FixedPointError::ObjectiveFailed { message }) => Err(FixedPointOuterRunError::Failed(
            EstimationError::fatal_outer_evaluation(
                "outer fixed-point evaluation",
                EstimationError::RemlOptimizationFailed(message),
            ),
        )),
        Err(e) => Err(FixedPointOuterRunError::Failed(
            EstimationError::RemlOptimizationFailed(format!("{failure_prefix}: {e:?}")),
        )),
    }
}

#[cfg(test)]
mod asymptote_rail_certify_tests {
    use super::*;
    use ndarray::array;

    /// Build a one-coordinate UPPER-rail tail-law objective: at ρ its gradient is
    /// `−c·e^{−ρ}` (so `ĉ = −e^{ρ}·grad = c` is constant) and its published inner
    /// β is `a·e^{−ρ}` (so consecutive-probe `‖Δβ‖` contracts geometrically).
    /// `drift_amp` ramps `ĉ` with ρ to model the finite-difference noise regime
    /// (a non-constant pencil constant that no drift band can confirm).
    fn upper_tail_objective(c: f64, a: f64, drift_amp: f64) -> impl OuterObjective {
        let problem = OuterProblem::new(1).with_gradient(Derivative::Analytic);
        problem.build_objective(
            (),
            move |_: &mut (), rho: &Array1<f64>| {
                let r = rho[0];
                let c_eff = c + drift_amp * r;
                Ok((c_eff * (-r).exp()).abs())
            },
            move |_: &mut (), rho: &Array1<f64>| {
                let r = rho[0];
                let c_eff = c + drift_amp * r;
                Ok(OuterEval {
                    cost: (c_eff * (-r).exp()).abs(),
                    gradient: array![-c_eff * (-r).exp()],
                    hessian: HessianValue::Unavailable,
                    inner_beta_hint: Some(array![a * (-r).exp()]),
                })
            },
            None::<fn(&mut ())>,
            None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
        )
    }

    /// An exact upper-rail exponential tail is certified: the reconstructed
    /// pencil constant is `c`, and the value-gap / estimand-travel are finite.
    #[test]
    fn asymptote_rail_mints_on_exact_tail_law() {
        let mut obj = upper_tail_objective(6723.0, 1.0, 0.0);
        let rho = array![29.9];
        let tol = AsymptoteTolerances::exp4_rail_bands(1.0e-2);
        let rail = build_and_assess_rail_coordinate(
            &mut obj,
            &rho,
            0,
            AsymptoteSide::Upper,
            &tol,
            (f64::NEG_INFINITY, f64::INFINITY),
        )
        .expect("probing the tail-law objective must not error")
        .expect("an exact exponential tail must certify a rail");
        assert_eq!(rail.index, 0);
        assert_eq!(rail.side, AsymptoteSide::Upper);
        assert!(
            (rail.tail_constant - 6723.0).abs() / 6723.0 < 1.0e-6,
            "recovered ĉ={} should equal c=6723",
            rail.tail_constant,
        );
        assert!(rail.value_gap.is_finite() && rail.value_gap >= 0.0);
        assert!(rail.estimand_travel_bound.is_finite() && rail.estimand_travel_bound >= 0.0);
    }

    /// A drifting pencil constant (finite-difference noise regime) never
    /// certifies: no finite-difference-clean run of the required length exists.
    #[test]
    fn asymptote_rail_refuses_on_drifting_constant() {
        let mut obj = upper_tail_objective(6723.0, 1.0, 3000.0);
        let rho = array![29.9];
        let tol = AsymptoteTolerances::exp4_rail_bands(1.0e-2);
        let verdict = build_and_assess_rail_coordinate(
            &mut obj,
            &rho,
            0,
            AsymptoteSide::Upper,
            &tol,
            (f64::NEG_INFINITY, f64::INFINITY),
        )
        .expect("probing must not error");
        assert!(
            verdict.is_err(),
            "a drifting ĉ must not certify a tail, got {verdict:?}",
        );
    }

    /// #2358: a finite smoothing box can expose fewer than three whole
    /// e-folds of the leading-order tail even though the compactified
    /// criterion is already regular there. For
    ///
    /// `V(ρ) = c·e⁻ρ + (d/2)·e⁻²ρ`,
    ///
    /// the pencil constant is `ĉ(ρ) = c + d·e⁻ρ`: the ordinary asymptotic law
    /// plus its first vanishing correction. Unit probes from ρ=10 step into
    /// enough correction curvature that no three-row clean run exists; the
    /// equal-spaced half-e-fold fallback resolves the local tail without
    /// changing the drift, estimand, sign, or noise gates.
    #[test]
    fn tail_probe_resolves_narrow_regular_band_before_finite_box_2358() {
        let (c, d, a) = (7.1_f64, 400.0_f64, 1.0e-3_f64);
        let problem = OuterProblem::new(1).with_gradient(Derivative::Analytic);
        let mut obj = problem.build_objective(
            (),
            move |_: &mut (), rho: &Array1<f64>| {
                let tau = (-rho[0]).exp();
                Ok(c * tau + 0.5 * d * tau * tau)
            },
            move |_: &mut (), rho: &Array1<f64>| {
                let tau = (-rho[0]).exp();
                Ok(OuterEval {
                    cost: c * tau + 0.5 * d * tau * tau,
                    gradient: array![-c * tau - d * tau * tau],
                    hessian: HessianValue::Unavailable,
                    inner_beta_hint: Some(array![a * tau]),
                })
            },
            None::<fn(&mut ())>,
            None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
        );
        let rho = array![10.0];
        let mut tol = AsymptoteTolerances::exp4_rail_bands(1.0e-2);
        tol.tail_drift_rel = TAIL_SNAP_DRIFT_REL;

        let (coarse, coarse_rows) = probe_tail_window_at_resolution(
            &mut obj,
            &rho,
            0,
            AsymptoteSide::Upper,
            &tol,
            (-10.0, 10.5),
            (1.0, ASYMPTOTE_PROBE_COUNT),
        )
        .expect("coarse probing must not error");
        assert!(
            coarse.is_none(),
            "unit probes must not manufacture a pure tail through the curved band: {coarse_rows}"
        );

        let (window, rows) = probe_tail_window(
            &mut obj,
            &rho,
            0,
            AsymptoteSide::Upper,
            &tol,
            (-10.0, 10.5),
        )
        .expect("multi-resolution probing must not error");
        let window = window.unwrap_or_else(|| {
            panic!("the half-e-fold fallback must resolve the regular tail: {rows}")
        });
        assert!(
            matches!(
                assess_coordinate(&window, &tol),
                AsymptoteVerdict::CertifiedAtAsymptote { .. }
            ),
            "the resolved tail must pass the unchanged asymptote gates"
        );
    }

    /// #2358: pre-snap interior stationarity cannot authorize direct
    /// certification at the rail. The second coordinate is stationary at the
    /// checkpoint but its mode depends on the tail coordinate, so moving the
    /// first coordinate changes the second coordinate's optimum. A confirmed
    /// snap must therefore publish a re-optimization waypoint even though the
    /// pre-snap interior gradient is exactly zero.
    #[test]
    fn coupled_coordinate_stationary_before_snap_still_reseeds_2358() {
        let c = 10.0_f64;
        let problem = OuterProblem::new(2).with_gradient(Derivative::Analytic);
        let mut obj = problem.build_objective(
            (),
            move |_: &mut (), rho: &Array1<f64>| {
                let tau = (-rho[0]).exp();
                let coupled = rho[1] - tau;
                Ok(c * tau + 0.5 * coupled * coupled)
            },
            move |_: &mut (), rho: &Array1<f64>| {
                let tau = (-rho[0]).exp();
                Ok(OuterEval {
                    cost: c * tau + 0.5 * (rho[1] - tau).powi(2),
                    gradient: array![
                        -c * tau + (rho[1] - tau) * tau,
                        rho[1] - tau
                    ],
                    hessian: HessianValue::Unavailable,
                    inner_beta_hint: Some(array![1.0e-3 * tau]),
                })
            },
            None::<fn(&mut ())>,
            None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
        );
        let rho = array![8.0, (-8.0_f64).exp()];
        let tail_gradient = -c * (-rho[0]).exp();
        let gradient = array![tail_gradient, 0.0];
        let hessian = array![[tail_gradient.abs(), 0.0], [0.0, 1.0]];
        let bounds = (Array1::from_elem(2, -12.0), Array1::from_elem(2, 12.0));

        let outcome = try_tail_snap_to_rail(
            &mut obj,
            &AsymptoteRailInputs {
                rho: &rho,
                projected_gradient: &gradient,
                railed: &[],
                layout: OuterThetaLayout::new(2, 0),
                hessian: &hessian,
                bounds: &bounds,
                terminal_beta: None,
                stationarity_bound: StationarityBound::from_ladder(1.0e-3, StationarityBoundSource::SolverBand),
                objective_tol: 1.0e-8,
                context: "coupled pre-snap stationarity guard",
            },
        )
        .expect("tail snap must not error");
        match outcome {
            TailSnapOutcome::ConfirmedNeedsReseed(snapped) => {
                assert_eq!(snapped, array![12.0, (-8.0_f64).exp()]);
            }
            other => panic!(
                "a coupled coordinate stationary only before the snap must reoptimize, got {other:?}"
            ),
        }
    }

    /// #2392 wrong-rail pull-back FIRES: a coordinate sitting at the UPPER bound
    /// whose clean-band probes carry a POSITIVE gradient (`∂V/∂ρ > 0`, so the
    /// pencil constant `ĉ = −e^{ρ}·g < 0` — descent points INWARD, away from the
    /// bound) was driven to the wrong rail. `detect_wrong_rail_pullback` returns
    /// an interior reseed target strictly below the coordinate's current ρ.
    #[test]
    fn wrong_rail_pullback_fires_on_inward_descent_2392() {
        // V(ρ) = −c·e^{−ρ} ⇒ ∂V/∂ρ = +c·e^{−ρ} > 0: the descent runs ρ DOWN, away
        // from the upper rail, and ĉ_upper = −e^{ρ}·(c·e^{−ρ}) = −c < 0 uniformly.
        let c = 6723.0;
        let problem = OuterProblem::new(1).with_gradient(Derivative::Analytic);
        let mut obj = problem.build_objective(
            (),
            move |_: &mut (), rho: &Array1<f64>| Ok(-c * (-rho[0]).exp()),
            move |_: &mut (), rho: &Array1<f64>| {
                Ok(OuterEval {
                    cost: -c * (-rho[0]).exp(),
                    gradient: array![c * (-rho[0]).exp()],
                    hessian: HessianValue::Unavailable,
                    inner_beta_hint: Some(array![(-rho[0]).exp()]),
                })
            },
            None::<fn(&mut ())>,
            None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
        );
        let rho = array![29.9];
        let tol = AsymptoteTolerances::exp4_rail_bands(1.0e-2);
        let target = detect_wrong_rail_pullback(
            &mut obj,
            &rho,
            0,
            AsymptoteSide::Upper,
            &tol,
            (-30.0, 30.0),
        )
        .expect("probing the wrong-rail objective must not error")
        .expect("an inward-descent rail must publish a pull-back target");
        assert!(
            target < rho[0] && target.is_finite(),
            "the reseed must move the coordinate INWARD (ρ down), got {target}",
        );
    }

    /// #2392 wrong-rail pull-back does NOT fire on a GENUINE upper-rail tail:
    /// `∂V/∂ρ < 0` ⇒ `ĉ > 0` ⇒ descent runs TOWARD the bound (a real λ→∞ optimum),
    /// which must never be pulled off its rail.
    #[test]
    fn wrong_rail_pullback_refuses_a_genuine_upper_tail_2392() {
        let mut obj = upper_tail_objective(6723.0, 1.0, 0.0);
        let rho = array![29.9];
        let tol = AsymptoteTolerances::exp4_rail_bands(1.0e-2);
        let verdict = detect_wrong_rail_pullback(
            &mut obj,
            &rho,
            0,
            AsymptoteSide::Upper,
            &tol,
            (-30.0, 30.0),
        )
        .expect("probing must not error");
        assert!(
            verdict.is_none(),
            "a genuine λ→∞ tail (ĉ>0) must not be pulled off its rail, got {verdict:?}",
        );
    }

    /// #2349: the interior-PSD gate must judge curvature above the
    /// gradient-residue noise floor. Fixture = the measured multinomial
    /// checkpoint shape: excluded tail candidates {0}, interior coordinate 1
    /// gradient-stationary (|g| = 1.0228e-3) with the corrupted tie-signature
    /// diagonal H₁₁ = −1.0216e-3 ≈ −|g₁| (the #2298 trace-pair residue — the
    /// entire measured 6×6 spectrum was PSD except this one sub-resolution
    /// entry). The raw gate refuses on the residue; the floored gate
    /// certifies; a GENUINE interior saddle (λ_min = −0.5 against the same
    /// tiny gradient) still refuses under the floor.
    #[test]
    fn interior_psd_gate_floors_tail_residue_but_keeps_genuine_saddles_2349() {
        let hessian = array![[0.2828, 0.0004], [0.0004, -1.0216e-3]];
        let gradient = array![-1.057, -1.0228e-3];
        let excluded = [0usize];
        assert_eq!(
            certificate_hessian_is_psd_off_railed(&hessian, &excluded),
            Some(false),
            "raw gate must see the corrupted sub-resolution entry as indefinite"
        );
        assert_eq!(
            certificate_hessian_is_psd_off_railed_above_gradient_floor(
                &hessian, &excluded, &gradient
            ),
            Some(true),
            "the gradient floor must absorb the O(|g|) trace-pair residue"
        );
        let saddle = array![[0.2828, 0.0004], [0.0004, -0.5]];
        assert_eq!(
            certificate_hessian_is_psd_off_railed_above_gradient_floor(
                &saddle, &excluded, &gradient
            ),
            Some(false),
            "a genuine interior saddle dwarfs the bound-scale floor and refuses"
        );
    }

    /// THE GUARANTEE, not an instance of it. Weyl bounds the floored spectrum by
    /// `λ_min(H) + min|g| ≤ λ_min(H + diag|g|) ≤ λ_min(H) + max|g|`, so the floor
    /// can absorb AT MOST `max_k |g_k|` over the JUDGED coordinates. Therefore
    /// any interior spectrum whose most negative direction exceeds that floor
    /// must still refuse — for every such spectrum, not merely for the one
    /// saddle that happened to be measured.
    ///
    /// Swept over curvatures spanning six orders and gradients spanning four,
    /// including the pair where they are within a factor of two of each other
    /// (the regime the floor exists to serve, and the only place the verdict is
    /// genuinely close). The excluded coordinate carries a deliberately huge
    /// gradient: the sub-block is extracted AFTER flooring, so it must never
    /// reach the floor.
    #[test]
    fn gradient_floor_absorbs_at_most_max_interior_gradient_weyl_bound() {
        let excluded = [0usize];
        for &lambda_min in &[-5.0e-1, -1.5e-2, -1.0e-3, -1.0e-5, -1.0e-7] {
            for &g_interior in &[1.0e-7, 1.0e-5, 1.0e-3, 1.0e-2] {
                // The railed coordinate's gradient is four orders above every
                // interior one; if it ever entered the floor the sweep would
                // certify everything.
                let gradient = array![-1.4017, g_interior];
                let hessian = array![[0.2828, 0.0004], [0.0004, lambda_min]];
                let floored = certificate_hessian_is_psd_off_railed_above_gradient_floor(
                    &hessian, &excluded, &gradient,
                );
                if g_interior < lambda_min.abs() {
                    assert_eq!(
                        floored,
                        Some(false),
                        "Weyl: max|g_int|={g_interior:.1e} < |λ_min|={:.1e} means the floored \
                         sub-block is still indefinite, so the gate MUST refuse",
                        lambda_min.abs()
                    );
                }
                // And the recorded clearance must report the same verdict
                // against the same floor, so the certificate's evidence and its
                // gate can never disagree.
                let clearance =
                    interior_curvature_floor_clearance(&hessian, &excluded, &gradient)
                        .expect("a finite 1×1 interior sub-block has a clearance");
                assert_eq!(
                    clearance.gradient_floor, g_interior,
                    "the floor must be the largest JUDGED gradient — the excluded \
                     coordinate's 1.4017 must never enter it"
                );
                assert!(
                    (clearance.interior_min_eigenvalue - lambda_min).abs()
                        <= 1.0e-12 * lambda_min.abs(),
                    "recorded λ_min {} should be the interior sub-block's own {lambda_min}",
                    clearance.interior_min_eigenvalue
                );
                assert_eq!(
                    Some(clearance.cleared),
                    floored,
                    "the recorded verdict and the gate must be the same judgment"
                );
            }
        }
    }

    /// Joint-face objective `V(ρ) = c·e^{−(ρ₀+ρ₁)/2}` — the algebraic skeleton
    /// of an OVERLAPPING-penalty λ→∞ face (the coalesced pseudo-logdet's
    /// shared range space couples the two coordinates, so each marginal
    /// gradient `g_k = −(c/2)e^{−(ρ₀+ρ₁)/2}` decays in the JOINT coordinate
    /// only). Along either single coordinate the pencil constant
    /// `ĉ_k = |g_k|e^{ρ_k}` sweeps a factor `e^{1/2}` per e-fold — far outside
    /// any drift band — while along the face direction it is exactly constant.
    fn joint_face_objective(c: f64, a: f64) -> impl OuterObjective {
        let problem = OuterProblem::new(2).with_gradient(Derivative::Analytic);
        problem.build_objective(
            (),
            move |_: &mut (), rho: &Array1<f64>| Ok(c * (-(rho[0] + rho[1]) / 2.0).exp()),
            move |_: &mut (), rho: &Array1<f64>| {
                let v = c * (-(rho[0] + rho[1]) / 2.0).exp();
                Ok(OuterEval {
                    cost: v,
                    gradient: array![-0.5 * v, -0.5 * v],
                    hessian: HessianValue::Unavailable,
                    inner_beta_hint: Some(array![a * (-(rho[0] + rho[1]) / 2.0).exp()]),
                })
            },
            None::<fn(&mut ())>,
            None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
        )
    }

    /// #2349 round 7: the joint multi-coordinate rail face. The marginal
    /// single-coordinate tail law honestly fails on an overlapping-penalty
    /// face (measured on the multinomial checkpoint: ĉ₀ swept 8 orders of
    /// magnitude), so tail-snap must fall back to the joint face direction,
    /// certify the one-dimensional joint law there, and snap the whole face.
    #[test]
    fn joint_face_tail_certifies_where_single_coordinate_law_drifts_2349() {
        let c = 1.2 * (7.5_f64).exp();
        let rho = array![8.0, 7.0];
        let tol = {
            let mut t = AsymptoteTolerances::exp4_rail_bands(1.0e-2);
            t.tail_drift_rel = TAIL_SNAP_DRIFT_REL;
            t
        };
        let bounds = (Array1::from_elem(2, -12.0), Array1::from_elem(2, 12.0));

        // The marginal law must genuinely fail first — otherwise this test
        // would pass vacuously with the joint path never exercised.
        let mut obj = joint_face_objective(c, 1.0e-9);
        let (single_window, _) = probe_tail_window(
            &mut obj,
            &rho,
            0,
            AsymptoteSide::Upper,
            &tol,
            (bounds.0[0], bounds.1[0]),
        )
        .expect("single-coordinate probing must not error");
        assert!(
            single_window.is_none(),
            "the marginal pencil constant drifts e^(1/2) per e-fold and must not \
             produce a finite-difference-clean run"
        );

        // The joint window recovers the exact face constant.
        let (joint_window, _) = probe_joint_tail_window(
            &mut obj,
            &rho,
            &[(0, AsymptoteSide::Upper), (1, AsymptoteSide::Upper)],
            &tol,
            (&bounds.0, &bounds.1),
        )
        .expect("joint probing must not error");
        let window = joint_window.expect("the joint face law is exactly exponential");
        match assess_coordinate(&window, &tol) {
            AsymptoteVerdict::CertifiedAtAsymptote { tail_constant, .. } => {
                assert!(
                    (tail_constant - c).abs() / c < 1.0e-6,
                    "joint pencil constant must recover c: got {tail_constant}, want {c}"
                );
            }
            other => panic!("joint face must certify, got {other:?}"),
        }

        // End-to-end: tail-snap declines the marginal law, falls back to the
        // joint face, and snaps BOTH coordinates to their upper rails.
        let g = -0.5 * c * (-7.5_f64).exp();
        let gradient = array![g, g];
        let hessian = array![[g.abs(), 0.0], [0.0, g.abs()]];
        let outcome = try_tail_snap_to_rail(
            &mut obj,
            &AsymptoteRailInputs {
                rho: &rho,
                projected_gradient: &gradient,
                railed: &[],
                layout: OuterThetaLayout::new(2, 0),
                hessian: &hessian,
                bounds: &bounds,
                terminal_beta: None,
                stationarity_bound: StationarityBound::from_ladder(1.0e-3, StationarityBoundSource::SolverBand),
                objective_tol: 1.0e-8,
                context: "joint-face guard test",
            },
        )
        .expect("tail snap must not error");
        match outcome {
            TailSnapOutcome::ConfirmedNeedsReseed(snapped) => {
                assert_eq!(
                    snapped,
                    array![12.0, 12.0],
                    "both face coordinates must snap to their upper rails"
                );
            }
            other => panic!(
                "the joint face must confirm and publish a reseed waypoint, got {other:?}"
            ),
        }
    }

    /// #2349 e2e shape: the joint pencil-constant run confirms (measured on
    /// the multinomial checkpoint: ĉ settling 62.7 → … → 34.2) while the
    /// coefficient steps in the retained deep-interior rows are NOT yet
    /// geometrically contracting — the crawl was cut mid-travel. A confirmed
    /// law with an unsettled estimand must SNAP the face for re-optimization
    /// (the single-coordinate `OnTailNotYetEquivalent` semantics), not
    /// decline.
    #[test]
    fn joint_face_with_unsettled_estimand_snaps_for_reoptimization_2349() {
        let c = 1.2 * (7.5_f64).exp();
        // Non-contracting coefficient hints: constant per-probe steps (β moves
        // linearly in r), so coef_step_ratio has q = 1 and the estimand gate
        // refuses while the pencil constant is exact.
        let problem = OuterProblem::new(2).with_gradient(Derivative::Analytic);
        let mut obj = problem.build_objective(
            (),
            move |_: &mut (), rho: &Array1<f64>| Ok(c * (-(rho[0] + rho[1]) / 2.0).exp()),
            move |_: &mut (), rho: &Array1<f64>| {
                let v = c * (-(rho[0] + rho[1]) / 2.0).exp();
                Ok(OuterEval {
                    cost: v,
                    gradient: array![-0.5 * v, -0.5 * v],
                    hessian: HessianValue::Unavailable,
                    inner_beta_hint: Some(array![0.1 * (rho[0] + rho[1])]),
                })
            },
            None::<fn(&mut ())>,
            None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
        );
        let rho = array![8.0, 7.0];
        let g = -0.5 * c * (-7.5_f64).exp();
        let gradient = array![g, g];
        let hessian = array![[g.abs(), 0.0], [0.0, g.abs()]];
        let bounds = (Array1::from_elem(2, -12.0), Array1::from_elem(2, 12.0));
        let outcome = try_tail_snap_to_rail(
            &mut obj,
            &AsymptoteRailInputs {
                rho: &rho,
                projected_gradient: &gradient,
                railed: &[],
                layout: OuterThetaLayout::new(2, 0),
                hessian: &hessian,
                bounds: &bounds,
                terminal_beta: None,
                stationarity_bound: StationarityBound::from_ladder(1.0e-3, StationarityBoundSource::SolverBand),
                objective_tol: 1.0e-8,
                context: "joint-face unsettled-estimand guard test",
            },
        )
        .expect("tail snap must not error");
        match outcome {
            TailSnapOutcome::ConfirmedNeedsReseed(snapped) => {
                assert_eq!(
                    snapped,
                    array![12.0, 12.0],
                    "a confirmed joint law with unsettled estimand must snap the face"
                );
            }
            other => panic!("expected a face snap, got {other:?}"),
        }
    }

    /// A pair of super-bound coordinates that do NOT share a face (independent
    /// laws with strongly drifting joint section) must still decline — the
    /// joint fallback cannot manufacture a certificate where no joint law
    /// holds.
    #[test]
    fn joint_face_fallback_refuses_a_non_face_2349() {
        // V = c₀·e^{−ρ₀} + drift·ρ₀·e^{−ρ₀} + c₁·e^{−2ρ₁}: coordinate 0's own
        // law is corrupted by the drift term and coordinate 1 decays at a
        // DIFFERENT exponential rate, so neither the marginals nor the joint
        // direction carry a constant pencil.
        let problem = OuterProblem::new(2).with_gradient(Derivative::Analytic);
        let (c0, drift, c1) = (3.0e3, 2.0e3, 5.0e2);
        let mut obj = problem.build_objective(
            (),
            move |_: &mut (), rho: &Array1<f64>| {
                Ok((c0 + drift * rho[0]) * (-rho[0]).exp() + c1 * (-2.0 * rho[1]).exp())
            },
            move |_: &mut (), rho: &Array1<f64>| {
                let e0 = (-rho[0]).exp();
                let e1 = (-2.0 * rho[1]).exp();
                Ok(OuterEval {
                    cost: (c0 + drift * rho[0]) * e0 + c1 * e1,
                    gradient: array![
                        drift * e0 - (c0 + drift * rho[0]) * e0,
                        -2.0 * c1 * e1
                    ],
                    hessian: HessianValue::Unavailable,
                    inner_beta_hint: Some(array![1.0e-9 * e0]),
                })
            },
            None::<fn(&mut ())>,
            None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
        );
        let rho = array![8.0, 7.0];
        let g0 = drift * (-8.0_f64).exp() - (c0 + drift * 8.0) * (-8.0_f64).exp();
        let g1 = -2.0 * c1 * (-14.0_f64).exp();
        let gradient = array![g0, g1];
        let hessian = array![[g0.abs(), 0.0], [0.0, g1.abs()]];
        let bounds = (Array1::from_elem(2, -12.0), Array1::from_elem(2, 12.0));
        let outcome = try_tail_snap_to_rail(
            &mut obj,
            &AsymptoteRailInputs {
                rho: &rho,
                projected_gradient: &gradient,
                railed: &[],
                layout: OuterThetaLayout::new(2, 0),
                hessian: &hessian,
                bounds: &bounds,
                terminal_beta: None,
                stationarity_bound: StationarityBound::from_ladder(1.0e-9, StationarityBoundSource::SolverBand),
                objective_tol: 1.0e-8,
                context: "non-face guard test",
            },
        )
        .expect("tail snap must not error");
        match outcome {
            TailSnapOutcome::Declined(reason) => {
                assert!(
                    reason.contains("joint 2-coordinate face"),
                    "the decline must carry the joint-face evidence, got: {reason}"
                );
            }
            other => panic!("a non-face must decline, got {other:?}"),
        }
    }

    /// #2388: the tail-probe ladder must never step the probed coordinate
    /// outside its own box interval. Past a box bound the ρ-gradient assembly
    /// reports the #197 frozen-axis projection — a literal `0.0` — so an
    /// out-of-box probe fabricates a hard-zero tail row (`1.531e0 → 0.000e0` in
    /// one e-fold in the #2388 evidence) that the drift band can never confirm,
    /// and the fit refuses. The ladder must stop at the last strictly-in-domain
    /// probe, and the in-domain rows alone must still confirm an exact tail.
    #[test]
    fn tail_probe_ladder_never_leaves_the_coordinate_box_2388() {
        let c = 6723.0_f64;
        // Deep enough that the in-domain ladder keeps a healthy-gradient run
        // (probes at |g| below the interior floor are rightly judged unclean),
        // shallow enough that the 18-probe ladder would cross it without the
        // domain clip.
        let box_lower = 12.0_f64;
        let probed = std::sync::Arc::new(std::sync::Mutex::new(Vec::<f64>::new()));
        let probed_in_eval = std::sync::Arc::clone(&probed);
        let problem = OuterProblem::new(1).with_gradient(Derivative::Analytic);
        let mut obj = problem.build_objective(
            (),
            move |_: &mut (), rho: &Array1<f64>| Ok((c * (-rho[0]).exp()).abs()),
            move |_: &mut (), rho: &Array1<f64>| {
                let r = rho[0];
                probed_in_eval.lock().expect("probe log").push(r);
                // Below the box the assembly's frozen-axis convention reports a
                // fabricated zero gradient — exactly the #2388 evidence shape.
                let grad = if r <= box_lower + 1.0e-8 {
                    0.0
                } else {
                    -c * (-r).exp()
                };
                Ok(OuterEval {
                    cost: (c * (-r).exp()).abs(),
                    gradient: array![grad],
                    hessian: HessianValue::Unavailable,
                    inner_beta_hint: Some(array![(-r).exp()]),
                })
            },
            None::<fn(&mut ())>,
            None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
        );
        let rho = array![29.9];
        let tol = AsymptoteTolerances::exp4_rail_bands(1.0e-2);
        let rail = build_and_assess_rail_coordinate(
            &mut obj,
            &rho,
            0,
            AsymptoteSide::Upper,
            &tol,
            (box_lower, 30.0),
        )
        .expect("probing must not error")
        .expect("the in-domain rows alone must certify the exact tail");
        assert!(
            (rail.tail_constant - c).abs() / c < 1.0e-6,
            "recovered ĉ={} should equal c={c}",
            rail.tail_constant,
        );
        let seen = probed.lock().expect("probe log").clone();
        assert!(
            !seen.is_empty() && seen.iter().all(|&r| r > box_lower),
            "no probe may leave the λ-selection domain (lower bound {box_lower}): {seen:?}",
        );
    }

    /// Build a two-coordinate objective: coordinate 0 follows the upper-rail tail
    /// law; coordinate 1 (interior) is gradient-flat.
    fn upper_tail_with_interior(c: f64, a: f64) -> impl OuterObjective {
        let problem = OuterProblem::new(2).with_gradient(Derivative::Analytic);
        problem.build_objective(
            (),
            move |_: &mut (), rho: &Array1<f64>| Ok((c * (-rho[0]).exp()).abs()),
            move |_: &mut (), rho: &Array1<f64>| {
                let r = rho[0];
                Ok(OuterEval {
                    cost: (c * (-r).exp()).abs(),
                    gradient: array![-c * (-r).exp(), 0.0],
                    hessian: HessianValue::Unavailable,
                    inner_beta_hint: Some(array![a * (-r).exp(), 0.0]),
                })
            },
            None::<fn(&mut ())>,
            None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
        )
    }

    /// The interior-PSD gate is load-bearing: with a positive-definite interior
    /// sub-block the confirmed tail mints, but a genuinely indefinite interior
    /// curvature refuses the rail certificate even though the tail is clean.
    #[test]
    fn asymptote_rail_requires_psd_interior_sub_block() {
        let rho = array![29.9, 0.0];
        let projected = array![0.0, 0.0];
        let bounds = (array![-30.0, -30.0], array![30.0, 30.0]);
        let railed = [0usize];

        let mut obj = upper_tail_with_interior(6723.0, 1.0);
        let hessian_psd = array![[1.0, 0.0], [0.0, 2.0]];
        let inputs_psd = AsymptoteRailInputs {
            rho: &rho,
            projected_gradient: &projected,
            railed: &railed,
            layout: OuterThetaLayout::new(2, 0),
            hessian: &hessian_psd,
            bounds: &bounds,
            terminal_beta: None,
            stationarity_bound: StationarityBound::from_ladder(1.0e-6, StationarityBoundSource::SolverBand),
            objective_tol: 1.0e-5,
            context: "asymptote-rail psd test",
        };
        let minted = try_certify_asymptote_rail(&mut obj, &inputs_psd)
            .expect("certification must not error");
        let (interior_norm, effective_bound, rails) =
            minted.expect("PSD interior + confirmed tail must mint");
        assert!(interior_norm <= 1.0e-6);
        assert!(
            effective_bound.value() >= interior_norm,
            "the admitting bound must cover the interior norm"
        );
        assert_eq!(rails.len(), 1);
        assert_eq!(rails[0].index, 0);

        let hessian_indefinite = array![[1.0, 0.0], [0.0, -2.0]];
        let inputs_indefinite = AsymptoteRailInputs {
            hessian: &hessian_indefinite,
            ..inputs_psd
        };
        let refused = try_certify_asymptote_rail(&mut obj, &inputs_indefinite)
            .expect("certification must not error");
        assert!(
            refused.is_err(),
            "indefinite interior curvature must refuse the rail certificate, got {refused:?}",
        );
        let reason = refused.unwrap_err();
        assert!(
            reason.contains("not PSD") || reason.contains("interior"),
            "the decline must name the refusing gate, got: {reason}"
        );
    }

    /// #2453: what authorizes the asymptote certificate is the coordinate
    /// being `log λ`, not the numbers looking exponential.
    ///
    /// Holding the fixture bit-identical — same objective, same ρ, same
    /// gradient, same PSD Hessian, same box, same railed set, the same
    /// textbook-clean `ĉ·e^{−ρ}` tail that the test above mints on — and
    /// flipping ONE declared fact, that the coordinate carries a sectional
    /// curvature rather than a log-smoothing parameter, must refuse. The
    /// alternative is a certificate that reports `value_gap = ĉ·e^{−κ}` as
    /// "the exact remaining criterion value-gap" for a quantity that is not
    /// in an exponent of anything.
    #[test]
    fn asymptote_rail_refuses_a_psi_coordinate_with_a_perfect_tail() {
        let rho = array![29.9, 0.0];
        let projected = array![0.0, 0.0];
        let bounds = (array![-30.0, -30.0], array![30.0, 30.0]);
        let railed = [0usize];
        let hessian = array![[1.0, 0.0], [0.0, 2.0]];

        let mut obj = upper_tail_with_interior(6723.0, 1.0);
        let as_psi = AsymptoteRailInputs {
            rho: &rho,
            projected_gradient: &projected,
            railed: &railed,
            // Both slots declared ψ: rho_dim = 0, so coordinate 0 is a
            // design-moving quantity whose box endpoint is attainable.
            layout: OuterThetaLayout::new(2, 2),
            hessian: &hessian,
            bounds: &bounds,
            terminal_beta: None,
            stationarity_bound: StationarityBound::from_ladder(
                1.0e-6,
                StationarityBoundSource::SolverBand,
            ),
            objective_tol: 1.0e-5,
            context: "asymptote-rail psi-identity test",
        };
        let refused =
            try_certify_asymptote_rail(&mut obj, &as_psi).expect("certification must not error");
        let reason = refused.expect_err(
            "a psi coordinate must not be certified at an asymptote it has no law for",
        );
        assert!(
            reason.contains("parameterizes log λ"),
            "the decline must name the coordinate's identity as the refusing gate, got: {reason}"
        );

        // The control: the identical numbers under a log-λ declaration still
        // mint, so the refusal above is the identity and nothing else.
        let as_rho = AsymptoteRailInputs {
            layout: OuterThetaLayout::new(2, 0),
            ..as_psi
        };
        let minted = try_certify_asymptote_rail(&mut obj, &as_rho)
            .expect("certification must not error")
            .expect("the same tail under a log-λ declaration must mint");
        assert_eq!(minted.2.len(), 1);
        assert_eq!(minted.2[0].index, 0);
    }

    /// #2453: the same identity gate on the snap path. Tail-snap MOVES the
    /// coordinate to its bound and republishes that as the search's next
    /// point, so applying it to a ψ coordinate does not just mislabel an
    /// optimum — it relocates the fit to one.
    #[test]
    fn tail_snap_refuses_a_psi_coordinate() {
        let c = 6723.0_f64;
        let mut obj = upper_tail_with_interior(c, 1.0);
        let rho = array![8.0_f64, 0.0];
        let tail_gradient = -c * (-rho[0]).exp();
        let gradient = array![tail_gradient, 0.0];
        let hessian = array![[tail_gradient.abs(), 0.0], [0.0, 1.0]];
        let bounds = (Array1::from_elem(2, -12.0), Array1::from_elem(2, 12.0));

        let outcome = try_tail_snap_to_rail(
            &mut obj,
            &AsymptoteRailInputs {
                rho: &rho,
                projected_gradient: &gradient,
                railed: &[],
                layout: OuterThetaLayout::new(2, 2),
                hessian: &hessian,
                bounds: &bounds,
                terminal_beta: None,
                stationarity_bound: StationarityBound::from_ladder(
                    1.0e-3,
                    StationarityBoundSource::SolverBand,
                ),
                objective_tol: 1.0e-8,
                context: "tail-snap psi-identity test",
            },
        )
        .expect("tail snap must not error");
        match outcome {
            TailSnapOutcome::Declined(reason) => assert!(
                reason.contains("psi coordinate"),
                "the decline must name the coordinate's identity, got: {reason}"
            ),
            other => panic!("a psi coordinate must never be snapped to its bound, got {other:?}"),
        }
    }
}

#[cfg(test)]
mod certify_resume_progress_tests {
    //! Unit coverage for the certify-last checkpoint-resume progress gate
    //! (#2374). The generalized loop keeps reseeding at the refused checkpoint
    //! only while each reseed exploits real descent; `certify_resume_made_progress`
    //! is the exact predicate that decides "real descent" vs "genuine floor", so
    //! pinning it directly pins the loop's termination contract independent of the
    //! solver dynamics that produce the reseeds.
    use super::{
        CERTIFY_RESUME_PROGRESS_REL, OuterConfig, certify_resume_made_progress,
        outer_rel_cost_floor,
    };

    fn config_with_rel_cost(rel_cost: Option<f64>, tolerance: f64) -> OuterConfig {
        OuterConfig {
            tolerance,
            rel_cost_tolerance: rel_cost,
            ..OuterConfig::default()
        }
    }

    #[test]
    fn rel_cost_floor_prefers_explicit_then_scaled_tolerance_never_below_hard_floor() {
        // Explicit relative tolerance wins verbatim.
        let explicit = config_with_rel_cost(Some(1.0e-3), 1.0e-5);
        assert_eq!(outer_rel_cost_floor(&explicit), 1.0e-3);
        // Absent, it derives from a small fraction of the absolute tolerance.
        let derived = config_with_rel_cost(None, 1.0e-2);
        assert!((outer_rel_cost_floor(&derived) - 1.0e-4).abs() <= 1.0e-16);
        // But never below the shared hard floor, however tight the tolerances.
        let tiny = config_with_rel_cost(Some(1.0e-30), 1.0e-30);
        assert_eq!(outer_rel_cost_floor(&tiny), super::COST_STALL_REL_TOL_FLOOR);
    }

    // ── Helper math (arbitrary floor) ────────────────────────────────────

    #[test]
    fn strict_descent_past_the_floor_is_progress() {
        let floor = 1.0e-4;
        // A drop far larger than floor·(1+|cost|)≈0.034 at cost scale ~1e2.
        assert!(certify_resume_made_progress(342.0, 300.0, floor));
        // A drop of many orders is trivially progress.
        assert!(certify_resume_made_progress(1.0e6, 1.0e3, floor));
    }

    #[test]
    fn flat_or_uphill_reseed_is_not_progress() {
        let floor = 1.0e-4;
        // Exactly equal: no descent.
        assert!(!certify_resume_made_progress(100.0, 100.0, floor));
        // Uphill: a metric restart that landed worse is never progress.
        assert!(!certify_resume_made_progress(100.0, 100.5, floor));
        // A reduction SMALLER than the passed floor is within the flat band.
        let cost = 1.0e4;
        let sub_floor = floor * (1.0 + cost) * 0.5;
        assert!(!certify_resume_made_progress(cost, cost - sub_floor, floor));
    }

    #[test]
    fn floor_anchors_on_the_smaller_cost_magnitude() {
        // With prior≈0 and a large-magnitude retried, anchoring on the smaller
        // (prior) magnitude keeps the floor tight so a genuine tiny descent near a
        // small optimum still registers, rather than being swamped by |retried|.
        let floor = 1.0e-4;
        // prior tiny-positive, retried strictly below it by more than floor·(1+0):
        assert!(certify_resume_made_progress(1.0e-2, 1.0e-3, floor));
    }

    #[test]
    fn non_finite_retried_is_never_progress() {
        let floor = 1.0e-4;
        assert!(!certify_resume_made_progress(100.0, f64::NAN, floor));
        assert!(!certify_resume_made_progress(100.0, f64::INFINITY, floor));
        assert!(!certify_resume_made_progress(100.0, f64::NEG_INFINITY, floor));
    }

    // ── Production gate (roundoff floor) ─────────────────────────────────
    //
    // These pin the ACTUAL gate the loop runs (`CERTIFY_RESUME_PROGRESS_REL`),
    // which must admit the tiny per-reseed descent a flat valley crawls out in
    // and reject only numerical noise — the exact distinction the earlier
    // cost-stall-floor gate got wrong (#2374: it stopped the survival LAML crawl
    // after one hop and refused a well-posed fit).

    #[test]
    fn roundoff_gate_admits_the_tiny_flat_valley_crawl_step() {
        let rel = CERTIFY_RESUME_PROGRESS_REL;
        // The transformation-survival LAML moves ~4e-5 relative per reseed:
        // cost 342.0730 → 342.0580 is ~4.4e-5 relative, far above roundoff.
        assert!(certify_resume_made_progress(342.0730, 342.0580, rel));
        // Even a 1e-6 relative step at cost ~455 (the two-smooth cohort scale)
        // is real descent under the roundoff gate — the coarse cost-stall floor
        // (~1e-6·456 ≈ 4.6e-4) would have wrongly rejected it.
        assert!(certify_resume_made_progress(455.40, 455.40 - 5.0e-4, rel));
    }

    #[test]
    fn roundoff_gate_rejects_noise_and_non_descent() {
        let rel = CERTIFY_RESUME_PROGRESS_REL;
        // A bitwise-identical reseed (genuine floor: BFGS found no descent from
        // the checkpoint) is not progress.
        assert!(!certify_resume_made_progress(455.40, 455.40, rel));
        // A reduction at the roundoff scale is noise, not descent: a few ULPs at
        // cost ~455 sits below CERTIFY_RESUME_PROGRESS_REL·(1+455).
        let noise = 4.0 * f64::EPSILON * (1.0 + 455.40);
        assert!(!certify_resume_made_progress(455.40, 455.40 - noise, rel));
        // Uphill / non-finite are never progress.
        assert!(!certify_resume_made_progress(455.40, 455.41, rel));
        assert!(!certify_resume_made_progress(455.40, f64::NAN, rel));
    }
}
