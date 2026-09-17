//! Grouped LAML smoothing selection for the support-sparse TopK term.
//!
//! A separate smoothing coordinate for every atom makes the outer problem grow
//! as `O(K)` even when thousands of atoms share one declared function space.
//! The sparse lane instead shares a variance component by exact
//! `(basis kind, latent dimension)` family. The generic optimizer therefore
//! sees the number of heterogeneous families, while the inner model still has
//! distinct decoder functions and coordinates for every occupied atom.
//!
//! ## The criterion is ONE functional (#2576), and it is not the dense one (#2933 F27)
//!
//! `2·cost(ρ) = log|S| − log|S_ρ|₊ + df·(1 + ln(τ·D_p/df))`, where
//! `S = H_ββ − Σ_i H_βt^(i) H_tt^(ⁱ)⁻¹ H_tβ^(ⁱ)` is the reduced decoder Schur
//! complement of the Gauss–Newton system. The row-coordinate normalizer
//! `Σ_i log|H_tt^(ⁱ)|` is removed, so the coordinates are profiled rather than
//! integrated. The Gaussian dispersion is profiled out of the penalized deviance
//! `D_p`, and the penalty pseudo-determinant `log|S_ρ|₊` carries its base term.
//!
//! The dense SAE criterion is a different statistical objective. It keeps the
//! coordinate block inside the observed information `log|A|` (#2668), scores the
//! penalized loss at unit dispersion, adds a realised-rank charge, and has no
//! `2π` constant. Both price the base pseudo-determinant (#2933 F26). A hard-TopK
//! request that crosses `K = P`
//! therefore changes criteria. The report carries a [`SaeCriterionScore`] of kind
//! [`SaeCriterionKind::ProfiledGaussianLaml`], [`SaeCriterionScore::difference`]
//! refuses a dense score, and [`SaeSupportLamlComponents`] names every term.
//!
//! The normalizer comes from the evidence lane ([`SurrogateLaneState`]) the dense
//! manifold criterion also runs. Where the dense `k × k` reduced Schur's complete
//! eigensystem fits the in-core ledger, it takes the exact `log|S|` off one
//! eigendecomposition; otherwise it evaluates the #2080 frozen rational surrogate
//! (#2731). Either way the value and its fixed-state directional derivative are one
//! representation; the profiled derivative additionally carries the exact implicit
//! response of the converged inner state.
//!
//! This lane previously took its factor cache from a Newton STEP it discarded
//! and estimated that trace with a SECOND, independent Hutchinson family — its
//! own probes, its own unshifted `S⁻¹` solves, one per smoothing group. That
//! cost the evaluation's whole wall-clock, left value and gradient free to
//! describe different functions, and never ran at all: the step entry returns
//! no reduced-Schur log-determinant under `InexactPCG`, so the criterion refused
//! on every host without a CUDA device.

use std::collections::BTreeMap;

use gam_problem::{DeclaredHessianForm, Derivative, EstimationError, HessianValue, OuterEval};
use gam_solve::rho_optimizer::{
    OuterCapability, OuterCriterionCertificate, OuterEvalOrder, OuterObjective, OuterProblem,
    SeedOutcome,
};
use ndarray::{Array1, Array2, ArrayView1};

use super::*;
use crate::front_door::{SaeCriterionKind, SaeCriterionScore};
use crate::migration_ledger::{
    BirthSeed, MoveEvidence, MoveReason, MoveStage, SaeMigrationLedger,
};

const SUPPORT_LAML_CONTEXT: &str = "support-sparse TopK grouped LAML";

/// The one field of the shared SAE evidence-surrogate policy this lane may not
/// inherit: the bar the frozen `log|S|` plan's Hutchinson error must clear
/// before its deflation rank stops growing.
///
/// The shared value is `0.1 · SAE_MANIFOLD_INNER_OBJECTIVE_STALL_REL_TOL` = 1e-9
/// of `|log|S|| + 1`. **That is unreachable at overcomplete border widths, and
/// unreachable here means REFUSED**, not merely slow:
/// `rational_reduced_schur_plan_derived` doubles the deflation rank until the
/// bar clears and refuses with `Err` when its ceiling is exhausted, which this
/// lane turns into a typed evidence failure. Measured on a small overcomplete
/// chart (N=2000, P=32, K=59, border 5056, `log|S| ≈ 1.4e4`): the bare
/// estimator's relative error bar is 3.7e-3 at 8 probes and 1.9e-3 at 16, so the
/// shared bar asks for roughly a hundredfold variance reduction that peeling 128
/// of 5056 directions, the ladder's ceiling when this was measured, could not
/// deliver. The ceiling is now the border itself, lowered only by the plan
/// storage memory admits (#2731), so the bar is either refused there or bought
/// with a basis spanning nearly the whole border.
///
/// The reachable bar, derived from the probe count rather than borrowed from an
/// inner-solve stall tolerance: `√(2/m)`, the relative standard error a
/// `±1`-Rademacher Hutchinson estimator has in the WORST case
/// (`Var(zᵀAz) = 2‖A_off‖_F²`, attained when the off-diagonal mass matches the
/// trace). Asking for exactly that says "deflate only when this operator's
/// spectrum is worse than the theory says an `m`-probe estimator can be" — the
/// bar fires on a pathological spectrum and stays out of the way otherwise,
/// which is the only thing a variance-reduction ladder should be doing.
///
/// This does not weaken the criterion's usable accuracy, because a FROZEN plan
/// draws the SAME probes at every ρ: the estimator's error is a nearly constant
/// offset across the smoothing search, not per-ρ jitter, and the search ranks
/// differences. That common-random-numbers property is the whole reason the
/// plan is frozen, and it is what makes a worst-case-bound bar the right one
/// here rather than an accuracy compromise.
fn support_laml_deflation_target_std_err_rel() -> f64 {
    (2.0 / SCHUR_SLQ_LOGDET_PROBES as f64).sqrt()
}

/// The coarsest quadrature this lane will ever ask for, and the accuracy its
/// one-off pilot runs at.
///
/// `√(2/m)/m`: the worst-case relative standard error a `±1`-Rademacher
/// Hutchinson estimator can have (`√(2/m)`, attained when the off-diagonal
/// Frobenius mass matches the trace) divided by the probe count. It is a
/// CEILING, not the working value — see
/// [`support_laml_measured_quadrature_tolerance`] for why an a-priori bound
/// cannot produce a safe working value here.
fn support_laml_coarsest_quadrature_tolerance() -> f64 {
    support_laml_deflation_target_std_err_rel() / SCHUR_SLQ_LOGDET_PROBES as f64
}

/// Accuracy asked of the surrogate's QUADRATURE — and of the quadrature only —
/// derived from this operator's MEASURED Hutchinson resolution.
///
/// The two deterministic knobs in this surrogate look alike and are not. Both
/// read `1.0e-8` in the shared policy, and only one of them should move:
///
/// * The **quadrature** truncation and step are fixed once, in the frozen plan.
///   Their error is a smooth, ρ-independent BIAS in `log|S|` — the same
///   displacement at every ρ the outer search visits. A bias the estimator's
///   own variance swamps is a bias nobody can measure, and asking for one five
///   orders under that variance is not free: the node count grows like
///   `log(1/tol)` and the surrogate's cost is `m × nodes` shifted solves.
///   Measured on a small overcomplete chart (border 5056, `λ_min/λ_max = 1e-8`
///   by the deflation-floor convention, hence a twelve-decade padded window):
///   `1e-8` sizes **81** nodes.
/// * The **shifted-CG residual** is not bias. Each solve's iteration count
///   varies with ρ, so its error is JITTER — a non-smooth `O(δ)` wobble in a
///   criterion the outer quasi-Newton differentiates and line-searches. What
///   that must beat is the outer search's step sizes, not the probe count, so
///   `cg_rel_tol` stays at the shared lane's value.
///   (`support_outer_logdet_gradient_matches_fd_of_its_own_surrogate` is the
///   gate that catches loosening it: a central difference at `h = 1e-5`
///   amplifies value jitter by `1/2h = 5e4`.)
///
/// **Why the bias budget has to be measured.** The requirement is
/// `m·δ ≲ σ/√m`: the bias, which the average does not shrink because it is
/// identical in every term, must stay under the stochastic error, which the
/// average does shrink. Substituting the a-priori worst case for `σ` moves the
/// bound the WRONG WAY — it is an upper bound on `σ`, so it yields an upper
/// bound on the ALLOWED `δ`, and a safe working value needs a lower bound on
/// `σ` instead. There is none: `σ` is zero for a diagonal operator. Measured
/// here, `σ/√m` is 1.9e-3 at 16 probes, two orders under the `√(2/m)` bound —
/// so the bound-derived `√(2/m)/m = 7.8e-3` sits ABOVE the noise it was meant
/// to hide under, not below it.
///
/// So measure it. The surrogate reports `std_err` — its own realized error bar
/// — and a rank-0 pilot is cheap at the ceiling tolerance (18 nodes rather than
/// 81). Crucially the pilot's COARSE quadrature does not corrupt the number it
/// is measuring: a quadrature bias is common to every probe and cancels out of
/// the across-probe spread that `std_err` is. The pilot's shifted solves DO run
/// at the working `cg_rel_tol`, because per-probe solve jitter would not cancel.
///
/// The working budget is then `σ̂/m`, clamped to `[shared rel_tol, ceiling]` so
/// it is never tighter than the shared policy would have asked nor looser than
/// the worst case admits. Measured: `σ̂ ≈ 1.4e-3` at 32 probes gives `4.4e-5`
/// and 39 nodes, against 81.
fn support_laml_measured_quadrature_tolerance(
    system: &ArrowSchurSystem,
    htt_factors: &ArrowFactorSlab,
    seed: u64,
) -> Result<f64, EstimationError> {
    let shared = sae_surrogate_lane_config();
    let ceiling = support_laml_coarsest_quadrature_tolerance();
    let timer = std::time::Instant::now();
    let (plan, pilot) = rational_reduced_schur_log_det(
        system,
        htt_factors,
        0.0,
        &CpuBatchedBlockSolver,
        None,
        None,
        shared.num_probes,
        seed,
        ceiling,
        shared.cg_rel_tol,
    )
    .ok_or_else(|| {
        outer_error(format!(
            "support LAML could not measure its reduced-Schur log-determinant resolution: the \
             rank-0 pilot surrogate did not evaluate on a border of width {}",
            system.k
        ))
    })?;
    let measured_relative_std_err = pilot.std_err / (pilot.estimate.abs() + 1.0);
    if !(measured_relative_std_err.is_finite() && measured_relative_std_err >= 0.0) {
        return Err(outer_error(format!(
            "support LAML pilot surrogate reported a non-finite error bar {} against estimate {}",
            pilot.std_err, pilot.estimate
        )));
    }
    let budget = (measured_relative_std_err / shared.num_probes as f64)
        .clamp(shared.rel_tol.min(ceiling), ceiling);
    log::info!(
        "support LAML quadrature budget: pilot log|S| = {:.6e}, measured relative error bar \
         {:.3e} at {} probes -> quadrature tolerance {:.3e} (shared policy {:.3e}, ceiling \
         {:.3e}); pilot {} total shifted-CG iterations, {} rational nodes, deflation rank {}, \
         {:.1}s",
        pilot.estimate,
        measured_relative_std_err,
        shared.num_probes,
        budget,
        shared.rel_tol,
        ceiling,
        pilot.cg_iterations,
        plan.nodes.len(),
        pilot.deflation_basis.len(),
        timer.elapsed().as_secs_f64(),
    );
    Ok(budget)
}

fn outer_error(message: impl Into<String>) -> EstimationError {
    EstimationError::RemlOptimizationFailed(message.into())
}

#[derive(Clone, Debug)]
pub struct SaeSupportSmoothingLayout {
    pub atom_group: Vec<usize>,
    pub group_keys: Vec<String>,
}

impl SaeSupportSmoothingLayout {
    pub(crate) fn from_term(term: &SaeSupportSparseTerm) -> Self {
        let mut keys = BTreeMap::<String, ()>::new();
        let atom_keys = term
            .atoms
            .iter()
            .map(|atom| {
                format!(
                    "{}:d{}",
                    sae_atom_basis_kind_name(atom.basis_kind()),
                    atom.latent_dim()
                )
            })
            .collect::<Vec<_>>();
        for key in &atom_keys {
            keys.insert(key.clone(), ());
        }
        let group_keys = keys.into_keys().collect::<Vec<_>>();
        let index = group_keys
            .iter()
            .enumerate()
            .map(|(group, key)| (key.clone(), group))
            .collect::<BTreeMap<_, _>>();
        let atom_group = atom_keys.iter().map(|key| index[key]).collect();
        Self {
            atom_group,
            group_keys,
        }
    }

    pub fn expand(&self, rho: &Array1<f64>) -> Result<Vec<f64>, String> {
        if rho.len() != self.group_keys.len() {
            return Err(format!(
                "SaeSupportSmoothingLayout::expand: rho length {} != groups {}",
                rho.len(),
                self.group_keys.len()
            ));
        }
        let lambdas =
            gam_problem::checked_exp_log_strengths(rho.iter().copied()).map_err(|error| {
                format!("SaeSupportSmoothingLayout::expand: invalid log strength: {error}")
            })?;
        Ok(self
            .atom_group
            .iter()
            .map(|&group| lambdas[group])
            .collect())
    }
}

/// Inner fixed-point iteration budget for the support-sparse engine.
///
/// This is a different quantity from the outer budget and must not be derived
/// from it. `max_outer_iter` counts quasi-Newton steps of the grouped-LAML
/// search over log-smoothing; `max_inner_iter` counts alternating
/// decoder/coordinate cycles spent reaching a stationary point *within a single*
/// outer evaluation. One outer step consumes a whole inner solve, so tying the
/// two together silently caps the inner solve at the length of the outer search
/// — and since [`SaeSupportSparseTerm::solve_fixed_point`] requires two
/// consecutive candidate cycles before it may report success, a small outer
/// budget then makes convergence unreachable rather than merely slow.
///
/// Both drivers of this engine read this one declaration: the tiered driver
/// through `Tier2SupportConfig`, and the public support-sparse fit entry through
/// its FFI request. They previously carried independent values, which is how they
/// came to disagree 4:1.
pub const SAE_SUPPORT_INNER_FIXED_POINT_MAX_ITER: usize = 256;

pub struct SaeSupportOuterRequest {
    pub term: SaeSupportSparseTerm,
    pub target: Array2<f64>,
    pub initial_smoothness: f64,
    pub ard_precisions: Vec<Vec<f64>>,
    pub max_outer_iter: usize,
    pub max_inner_iter: usize,
    pub trust_radius: f64,
    pub random_state: u64,
}

pub struct SaeSupportOuterReport {
    pub term: SaeSupportSparseTerm,
    pub smoothing_layout: SaeSupportSmoothingLayout,
    pub log_lambda_groups: Array1<f64>,
    pub lambda_smooth: Vec<f64>,
    pub ard_precisions: Vec<Vec<f64>>,
    /// The terminal criterion, typed as the profiled-Gaussian LAML this lane
    /// minimizes. It does not compare with a dense quasi-Laplace score (#2933 F27).
    pub criterion: SaeCriterionScore,
    /// The named terms `criterion` is assembled from.
    pub criterion_components: SaeSupportLamlComponents,
    pub fixed_point: SaeSupportFixedPointReport,
    pub outer_iterations: usize,
    pub outer_certificate: OuterCriterionCertificate,
    /// The relative tolerance the inner fixed point certified to:
    /// [`SaeSupportSparseTerm::fixed_point_tolerance`] of the term the search started
    /// from.
    pub inner_tolerance: f64,
}

struct PenaltySpectrum {
    rank_by_group: Vec<usize>,
    log_pdet_base_by_group: Vec<f64>,
    total_rank: usize,
}

/// The named terms of one support LAML value (#2933 F27):
/// `2·cost = reduced_log_det − penalty_log_pdet
///   + residual_df·(1 + ln(2π·penalized_deviance/residual_df))`.
///
/// The terms show where this criterion departs from the dense quasi-Laplace score.
/// The curvature is the Gauss–Newton reduced Schur with the row block profiled
/// out. The scale is profiled out of the penalized deviance rather than held at
/// one, which brings `2π`. No realised-rank charge enters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SaeSupportLamlComponents {
    /// `log|S|` of the Gauss–Newton reduced decoder Schur complement.
    /// `Σ_i log|H_tt^(i)|` is not in it.
    pub reduced_log_det: f64,
    /// `log|λS|₊ = Σ_g (log|S_g|₊ + rank_g·ρ_g)`, base pseudo-determinant included.
    pub penalty_log_pdet: f64,
    /// Response cells less the estimated unpenalized decoder directions.
    pub residual_df: f64,
    /// `D_p = 2·penalized_objective`: the residual sum of squares plus every
    /// penalty the inner solve descends.
    pub penalized_deviance: f64,
}

impl SaeSupportLamlComponents {
    /// `½·df·(1 + ln(2π·D_p/df))`: the Gaussian likelihood with its dispersion
    /// profiled out.
    pub fn profiled_dispersion_term(&self) -> f64 {
        0.5 * self.residual_df
            * (1.0
                + (std::f64::consts::TAU * self.penalized_deviance / self.residual_df).ln())
    }

    /// The support LAML value these terms assemble.
    pub fn value(&self) -> f64 {
        0.5 * (self.reduced_log_det - self.penalty_log_pdet
            + self.residual_df
                * (1.0
                    + (std::f64::consts::TAU * self.penalized_deviance / self.residual_df)
                        .ln()))
    }
}

#[derive(Clone)]
struct SupportOuterEvaluation {
    cost: f64,
    components: SaeSupportLamlComponents,
    gradient: Array1<f64>,
    lambda_smooth: Vec<f64>,
    fixed_point: SaeSupportFixedPointReport,
}

#[derive(Clone)]
struct CachedSupportOuterEvaluation {
    rho: Array1<f64>,
    state_revision: u64,
    available_order: OuterEvalOrder,
    evaluation: SupportOuterEvaluation,
}

impl CachedSupportOuterEvaluation {
    fn matches(
        &self,
        rho: &Array1<f64>,
        state_revision: u64,
        requested_order: OuterEvalOrder,
    ) -> bool {
        self.state_revision == state_revision
            && self.rho.len() == rho.len()
            && self
                .rho
                .iter()
                .zip(rho.iter())
                .all(|(cached, requested)| cached.to_bits() == requested.to_bits())
            && match self.available_order {
                OuterEvalOrder::Value => requested_order == OuterEvalOrder::Value,
                OuterEvalOrder::ValueAndGradient => {
                    requested_order != OuterEvalOrder::ValueGradientHessian
                }
                OuterEvalOrder::ValueGradientHessian => true,
            }
    }
}

struct SaeSupportOuterObjective {
    term: SaeSupportSparseTerm,
    initial_term: SaeSupportSparseTerm,
    target: Array2<f64>,
    layout: SaeSupportSmoothingLayout,
    spectrum: PenaltySpectrum,
    ard_precisions: Vec<Vec<f64>>,
    max_inner_iter: usize,
    inner_tolerance: f64,
    trust_radius: f64,
    random_state: u64,
    last_evaluation: Option<CachedSupportOuterEvaluation>,
    /// Identity of the mutable fitted term underlying `last_evaluation`.
    /// Every uncached solve and reset advances it before touching the state, so
    /// a cache entry can never survive a failed or intervening evaluation.
    state_revision: u64,
    /// Uncached-solve counter. Unconditional rather than `#[cfg(test)]`:
    /// the ban scanner rejects `#[cfg(test)]` on a `src/` item (it is not a
    /// dead-code escape hatch), and one `usize` per outer solve is not worth a
    /// cfg split.
    uncached_evaluations: usize,
    /// The FROZEN reduced-Schur log-determinant surrogate for this outer solve
    /// — the same #2080 lane the dense manifold criterion runs on.
    ///
    /// Built once, on the first evaluation, and reused at every subsequent ρ.
    /// The plan is the criterion's identity: its probes, quadrature nodes, and
    /// deflation basis fix WHICH function of ρ the outer search descends.
    /// Rebuilding it per ρ would evaluate a different function at every point,
    /// and the exact directional derivative the gradient contracts would then
    /// be the exact gradient of something nobody evaluated twice.
    logdet_surrogate: Option<SurrogateLaneState>,
}

fn penalty_spectrum(
    term: &SaeSupportSparseTerm,
    layout: &SaeSupportSmoothingLayout,
) -> Result<PenaltySpectrum, String> {
    let groups = layout.group_keys.len();
    let mut rank_by_group = vec![0usize; groups];
    let mut log_pdet_base_by_group = vec![0.0; groups];
    for (atom_idx, atom) in term.atoms.iter().enumerate() {
        let symmetric = (atom.smooth_penalty() + &atom.smooth_penalty().t()) * 0.5;
        let (values, _) = symmetric
            .eigh(Side::Lower)
            .map_err(|error| format!("support smooth-penalty eigendecomposition: {error}"))?;
        // The positive eigenspace is counted on gam-solve's REML rule, the one the
        // dense criterion's `symmetric_rank` reads for the same `S_k`, so both lanes
        // price one rank and one pseudo-logdet. Every install path already refuses a
        // Gram that is not positive semidefinite
        // (`SaeManifoldAtom::validate_reference_function_gram`), so a computed
        // eigenvalue at or below the threshold is a null mode here, not a refusal.
        let threshold =
            gam_solve::estimate::reml::reml_outer_engine::positive_eigenvalue_threshold(
                &values.to_vec(),
            );
        let group = layout.atom_group[atom_idx];
        for value in values.iter().copied().filter(|value| *value > threshold) {
            rank_by_group[group] = rank_by_group[group]
                .checked_add(term.output_dim())
                .ok_or_else(|| "support penalty rank overflow".to_string())?;
            log_pdet_base_by_group[group] += term.output_dim() as f64 * value.ln();
        }
    }
    let total_rank = rank_by_group.iter().sum();
    Ok(PenaltySpectrum {
        rank_by_group,
        log_pdet_base_by_group,
        total_rank,
    })
}

/// #2576: the evidence quotient for atoms no row selects. Such an atom's decoder
/// block in the reduced Schur is exactly `λ S ⊗ I_p`, decoupled from every row and
/// every other atom, so its penalty's null space is a border direction nothing
/// identifies (job 631950: spectrum [-2.26e-13, 1.40e3], atom 10 selected by no row).
/// The quotient pins those directions, where they add `log 1 = 0`. On its range the
/// atom adds p times the log pseudo-determinant of `λS` to `log|S|`, and the penalty
/// log pseudo-determinant adds the same amount, so it cancels. Null modes are counted on the rule
/// `penalty_spectrum` counts rank on, so the returned count is exactly what
/// `beta_nullity` has to drop.
fn unused_atom_null_quotient(
    term: &SaeSupportSparseTerm,
) -> Result<Option<(gam_solve::arrow_schur::ArrowBetaGaugeQuotient, usize)>, String> {
    let unused = term.atoms_without_rows();
    if unused.is_empty() {
        return Ok(None);
    }
    let (beta_offsets, beta_dim) = term.beta_layout()?;
    let width = term.output_dim();
    let mut null_modes: Vec<(usize, Array1<f64>)> = Vec::new();
    for &atom in &unused {
        let penalty = term.atoms[atom].smooth_penalty();
        let symmetric = (penalty + &penalty.t()) * 0.5;
        let (values, vectors) = symmetric
            .eigh(Side::Lower)
            .map_err(|error| format!("support unused-atom penalty eigendecomposition: {error}"))?;
        let threshold =
            gam_solve::estimate::reml::reml_outer_engine::positive_eigenvalue_threshold(
                &values.to_vec(),
            );
        for (mode, &value) in values.iter().enumerate() {
            if value <= threshold {
                null_modes.push((atom, vectors.column(mode).to_owned()));
            }
        }
    }
    let pinned = null_modes.len() * width;
    if pinned == 0 {
        return Ok(None);
    }
    // One dense border-width vector per pinned direction: admitted against the
    // in-core ledger before any of them is allocated.
    let carrier_bytes = (pinned as u128)
        .saturating_mul(beta_dim as u128)
        .saturating_mul(std::mem::size_of::<f64>() as u128);
    let budget = crate::manifold::sae_host_in_core_budget_bytes().0 as u128;
    if carrier_bytes > budget {
        return Err(format!(
            "support LAML evidence: atoms {unused:?} select no row, and pinning their {pinned} \
             unidentified penalty directions of border width {beta_dim} needs {carrier_bytes} \
             bytes against an in-core budget of {budget} bytes"
        ));
    }
    let mut directions = Vec::with_capacity(pinned);
    for (atom, vector) in &null_modes {
        for channel in 0..width {
            let mut direction = Array1::<f64>::zeros(beta_dim);
            for (basis, &value) in vector.iter().enumerate() {
                direction[beta_offsets[*atom] + basis * width + channel] = value;
            }
            directions.push(direction);
        }
    }
    gam_solve::arrow_schur::ArrowBetaGaugeQuotient::new(directions)
        .map(|quotient| Some((quotient, pinned)))
}

impl SaeSupportOuterObjective {
    fn beta_layout(&self) -> Result<(Vec<usize>, usize), EstimationError> {
        self.term.beta_layout().map_err(outer_error)
    }

    fn penalty_energy_by_group(&self, lambda_smooth: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; self.layout.group_keys.len()];
        for atom in 0..self.term.k_atoms() {
            let sb = self.term.atoms[atom]
                .smooth_penalty()
                .dot(self.term.atoms[atom].decoder_coefficients());
            let energy = self.term.atoms[atom]
                .decoder_coefficients()
                .iter()
                .zip(sb.iter())
                .map(|(left, right)| left * right)
                .sum::<f64>();
            out[self.layout.atom_group[atom]] += lambda_smooth[atom] * energy;
        }
        out
    }

    /// `∂S/∂ρ_g` as a matvec: the EXACT reduced-Schur derivative along one
    /// smoothing coordinate.
    ///
    /// The chain is short and it is worth stating in full, because it is what
    /// lets this lane drop its separate trace estimator entirely. The joint
    /// Hessian's dependence on the smoothing coordinates is confined to the
    /// β block: `H_tt^(i)` is the Gauss–Newton latent curvature plus the ARD
    /// prior and `H_tβ^(i)` is the cross term, and neither carries a λ. So on
    /// `S = H_ββ − Σ_i H_βt^(i)(H_tt^(i))⁻¹H_tβ^(i)` only the first term moves,
    /// and since `λ_g = exp(ρ_g)`,
    ///
    /// ```text
    /// ∂S/∂ρ_g = ∂H_ββ/∂ρ_g = Σ_{k ∈ g} λ_k (S_k ⊗ I_P) = the group-g penalty apply.
    /// ```
    ///
    /// This is the EXPLICIT, frozen-inner-state derivative. The coordinates and
    /// decoder fitted at the inner optimum also move with ρ, so the complete
    /// profiled derivative adds the implicit-function response assembled in
    /// [`SaeSupportSparseTerm::support_reduced_logdet_profile_adjoint`]. Keeping
    /// this operator restricted to the explicit penalty direction makes that
    /// split auditable: no state-response term can be counted twice.
    fn schur_derivative_matvec(
        &self,
        group: usize,
        lambda_smooth: &[f64],
        beta_offsets: &[usize],
        beta_dim: usize,
        vector: ArrayView1<'_, f64>,
    ) -> Array1<f64> {
        let output_dim = self.term.output_dim();
        let mut out = Array1::<f64>::zeros(beta_dim);
        for atom in 0..self.term.k_atoms() {
            if self.layout.atom_group[atom] != group {
                continue;
            }
            let m = self.term.atoms[atom].basis_size();
            let offset = beta_offsets[atom];
            let lambda = lambda_smooth[atom];
            let penalty = self.term.atoms[atom].smooth_penalty();
            for left in 0..m {
                for right in 0..m {
                    let weight = lambda * penalty[[left, right]];
                    for channel in 0..output_dim {
                        out[offset + left * output_dim + channel] +=
                            weight * vector[offset + right * output_dim + channel];
                    }
                }
            }
        }
        out
    }

    /// The coordinate-profiled Laplace normalizer `log|S|`, and the bundle whose
    /// contraction against any `∂S` is that value's exact fixed-state
    /// directional derivative.
    ///
    /// This lane used to obtain its factor cache from
    /// `solve_arrow_newton_step_with_options`, which solves a Newton step it
    /// then discarded. That was not merely wasteful: `InexactPCG` never forms a
    /// dense `k × k` reduced-Schur factor, so the cache came back with
    /// `schur_factor_is_undamped = false`, `arrow_log_det()` returned `None`,
    /// and the criterion could not evaluate AT ALL on a host without a CUDA
    /// device — the only place a matrix-free `log|S|` was reachable from that
    /// entry. Meanwhile the discarded step still paid for a
    /// `JacobiPreconditioner` build, `O(n·K)` at the overcomplete border, which
    /// is where the reported minutes of silent burn actually went (#2576).
    ///
    /// What a criterion needs instead is the evidence factorization:
    /// `Σ_i log|H_tt^(i)|` from the undamped per-row Cholesky, and `log|S|` by a
    /// route it can also DIFFERENTIATE. That route already exists and is already
    /// production on the dense manifold lane:
    /// [`gam_solve::arrow_schur::matrix_free_arrow_evidence_evaluation`] driven by a
    /// [`SurrogateLaneState`]. It takes the exact `log|S|` off one eigendecomposition
    /// where the dense `k × k` block fits in core, and the #2080 frozen rational
    /// surrogate otherwise (#2731). It runs on CPU and device alike, and its value
    /// and gradient are one functional by construction. The support lane joins it
    /// rather than growing a second
    /// evidence policy beside it; [`Self::schur_derivative_matvec`] supplies the
    /// explicit-ρ half and the support-term adjoint supplies the fitted-state
    /// half.
    fn evidence_log_det(
        &mut self,
        system: &ArrowSchurSystem,
    ) -> Result<(f64, RationalLogdetDerivativeBundle), EstimationError> {
        if system.k == 0 {
            return Err(outer_error(
                "support LAML requires a decoder border to select smoothing against; the \
                 assembled arrow system has none",
            ));
        }
        // One SAE evidence-surrogate policy, shared with the dense manifold
        // lane, with exactly two fields DERIVED rather than inherited — and the
        // derivation happens ONCE, here, at the first ρ, so the whole outer
        // search descends a single functional.
        let seed = self.random_state;
        if self.logdet_surrogate.is_none() {
            // `evidence_factorization = true` must match what the lane itself
            // will use, or the pilot would measure a different operator from
            // the one the frozen plan is built on. It does: the lane runs
            // `ArrowEvidencePolicy::UnitDeflation`
            // (`with_evidence_unit_deflation` below), and
            // `factors_undamped_evidence()` is `!matches!(self, Strict)` — true
            // for every policy except `Strict`, which this lane never selects.
            let htt_factors = CpuBatchedBlockSolver
                .factor_blocks(&system.rows, 0.0, system.d, true)
                .map_err(|error| {
                    outer_error(format!(
                        "support LAML undamped evidence row factorization: {error}"
                    ))
                })?;
            let rel_tol =
                support_laml_measured_quadrature_tolerance(system, &htt_factors, seed)?;
            self.logdet_surrogate = Some(SurrogateLaneState::new(SurrogateLaneConfig {
                // The seed is the caller's: a support fit's `random_state` is
                // what makes ITS criterion bit-reproducible, and two fits of the
                // same data at different seeds must be able to disagree about
                // their probes without disagreeing about their policy.
                seed,
                deflation_target_std_err_rel: support_laml_deflation_target_std_err_rel(),
                rel_tol,
                ..sae_surrogate_lane_config()
            }));
        }
        let lane = self
            .logdet_surrogate
            .as_mut()
            .expect("the surrogate lane was just installed");
        // Ask for the derivative representation BEFORE the value, so a failed
        // evaluation can never be paired with a previous operator's gradient.
        lane.request_logdet_derivative_bundle();
        let timer = std::time::Instant::now();
        // The lane factors the Gauss–Newton majorizer, PSD by construction, so a reduced-Schur
        // eigenvalue below the unit-deflation floor is the rounding image of a numerically null
        // direction. Under `UnitDeflation` both lanes pin it to unit stiffness, `log 1 = 0`: the
        // dense route off its eigendecomposition (024816108), the rational route through its Ritz
        // conditioning (e8fa40de7). `PositiveDefinite` refused it instead: the pair-chart ring fit
        // in job 623388 stopped on the spectrum [-6.161538e-16, 2.584721e1] (dim 24) (#2576).
        let options = ArrowSolveOptions::inexact_pcg()
            .with_evidence_unit_deflation(gam_solve::arrow_schur::SPECTRAL_DEFLATION_REL_FLOOR);
        // #2731: where the dense `k × k` reduced Schur's complete eigensystem fits the
        // cgroup-aware in-core ledger, the lane takes the exact `log|S|` off one
        // eigendecomposition and its derivative bundle is the exact `tr(S⁻¹·D)`;
        // otherwise it walks the frozen rational surrogate. The route prices its own peak
        // with `dense_lane_reduced_schur_peak_bytes`, the one admission rule the criterion
        // lane uses as well.
        let dense_reduced_schur_admitted =
            gam_solve::arrow_schur::dense_lane_reduced_schur_peak_bytes(system.k)
                .is_some_and(|bytes| bytes <= crate::manifold::sae_host_in_core_budget_bytes().0);
        let evaluated = gam_solve::arrow_schur::matrix_free_arrow_evidence_evaluation(
            system,
            0.0,
            0.0,
            &options,
            SCHUR_SLQ_LOGDET_PROBES,
            SCHUR_SLQ_LOGDET_LANCZOS_STEPS,
            SCHUR_SLQ_LOGDET_SEED,
            lane,
            dense_reduced_schur_admitted,
        );
        let (row_log_det, schur_log_det) = match evaluated {
            Ok(evaluation) => (evaluation.log_det_tt, evaluation.log_det_schur),
            Err(error) => {
                drop(lane.take_logdet_derivative_bundle());
                // #2576: job 627150 refused here after an adopted support move, on a
                // reduced Schur with a null eigenvalue at roundoff. An atom no row
                // selects leaves a decoder block that holds only its penalty, whose
                // null space the data cannot identify, so the refusal names them.
                let unused = self.term.atoms_without_rows();
                return Err(outer_error(format!(
                    "support LAML matrix-free evidence log-determinant: {error}; atoms no row \
                     selects: {} {:?}",
                    unused.len(),
                    unused,
                )));
            }
        };
        let bundle = lane.take_logdet_derivative_bundle().ok_or_else(|| {
            outer_error(
                "support LAML evidence evaluation did not emit its value's derivative bundle, \
                 so no smoothing gradient can be minted from it",
            )
        })?;
        // The expensive half of one outer evaluation lives in this call, and
        // before #2576 it emitted nothing at all — six minutes of fourteen busy
        // cores between two log lines is what kept the cost invisible.
        if dense_reduced_schur_admitted {
            log::info!(
                "support LAML evidence: border {}, row log|H_tt| = {:.6e}, dense exact log|S| = \
                 {:.6e} from one eigendecomposition, {:.1}s",
                system.k,
                row_log_det,
                schur_log_det,
                timer.elapsed().as_secs_f64(),
            );
        } else {
            let metrics = bundle.evaluation_metrics();
            log::info!(
                "support LAML evidence: border {}, row log|H_tt| = {:.6e}, surrogate log|S| = \
                 {:.6e}; {} total shifted-CG iterations, {} rational nodes, deflation rank {}, \
                 {:.1}s",
                system.k,
                row_log_det,
                schur_log_det,
                metrics.cg_iterations,
                metrics.node_count,
                metrics.deflation_rank,
                timer.elapsed().as_secs_f64(),
            );
        }
        // Coordinates are nuisance parameters profiled by the inner solve. As
        // in `rank_adjusted_quasi_laplace_complexity`, their row-block
        // determinant is removed from the criterion; retaining it here would
        // charge arbitrary decoder scaling and, worse, would pair a moving row
        // value with a derivative that differentiates only the reduced Schur
        // operator.
        Ok((schur_log_det, bundle))
    }

    fn evaluate_for_order(
        &mut self,
        rho: &Array1<f64>,
        requested_order: OuterEvalOrder,
    ) -> Result<SupportOuterEvaluation, EstimationError> {
        if let Some(cached) = &self.last_evaluation
            && cached.matches(rho, self.state_revision, requested_order)
        {
            return Ok(cached.evaluation.clone());
        }

        // Invalidate before the inner solve mutates `term`: a refused
        // evaluation must never leave the previous point reusable against a
        // different fitted state. Wrapping is safe because the sole cache entry
        // is gone before the revision advances.
        self.last_evaluation = None;
        self.state_revision = self.state_revision.wrapping_add(1);
        self.uncached_evaluations += 1;
        let evaluation = self.evaluate_uncached(rho)?;
        // The implementation always computes the analytic gradient alongside
        // the value. A value request therefore populates a gradient-capable
        // entry; a declared VGH request additionally establishes that the
        // unavailable-Hessian result is the requested maximum capability.
        let available_order = match requested_order {
            OuterEvalOrder::ValueGradientHessian => OuterEvalOrder::ValueGradientHessian,
            OuterEvalOrder::Value | OuterEvalOrder::ValueAndGradient => {
                OuterEvalOrder::ValueAndGradient
            }
        };
        self.last_evaluation = Some(CachedSupportOuterEvaluation {
            rho: rho.clone(),
            state_revision: self.state_revision,
            available_order,
            evaluation: evaluation.clone(),
        });
        Ok(evaluation)
    }

    fn evaluate(&mut self, rho: &Array1<f64>) -> Result<SupportOuterEvaluation, EstimationError> {
        self.evaluate_for_order(rho, OuterEvalOrder::ValueAndGradient)
    }

    fn evaluate_uncached(
        &mut self,
        rho: &Array1<f64>,
    ) -> Result<SupportOuterEvaluation, EstimationError> {
        let lambda_smooth = self.layout.expand(rho).map_err(outer_error)?;
        let fixed_point = self
            .term
            .solve_fixed_point(
                self.target.view(),
                &lambda_smooth,
                &self.ard_precisions,
                self.max_inner_iter,
                self.inner_tolerance,
                self.trust_radius,
            )
            .map_err(outer_error)?;
        let mut system = self
            .term
            .assemble_arrow_schur(self.target.view(), &lambda_smooth, &self.ard_precisions)
            .map_err(outer_error)?;
        // #2576: atoms no row selects leave border directions nothing identifies. The
        // evidence prices the quotient without them (`unused_atom_null_quotient`),
        // and `beta_nullity` below drops the same count.
        let pinned_dims = match unused_atom_null_quotient(&self.term).map_err(outer_error)? {
            Some((quotient, pinned)) => {
                system.set_beta_gauge_quotient(quotient).map_err(outer_error)?;
                pinned
            }
            None => 0,
        };
        let (reduced_logdet, logdet_derivative) = self.evidence_log_det(&system)?;
        // Gaussian dispersion argument = the PENALIZED deviance
        //   D_p(ρ) = ‖y − ŷ‖² + β̂ᵀ S_ρ β̂ + (every other penalty the inner solve descends),
        // NOT the raw residual sum of squares. This mirrors the canonical dense
        // manifold path, whose profiled-scale data term ranks `loss.total()`
        // (data_fit + smoothness + ard + sparsity) — the full penalized loss —
        // precisely so the envelope theorem makes the analytic outer gradient
        // exact (construction_quasi_laplace.rs:357). `penalized_objective` returns
        // ½·D_p (½‖y−ŷ‖² + ½Σ_k λ_k β̂ᵀS_kβ̂ + ARD), so 2× recovers D_p. At the inner
        // optimum β̂ minimizes the full penalized objective, hence the envelope
        // theorem gives d D_p/dρ_g = ∂_ρ_g D_p|_{β̂} = Σ_{k∈g} λ_k β̂ᵀS_kβ̂ = energy[g]
        // — the exact numerator the gradient below already forms. With the RAW rss
        // instead, d(rss)/dρ_g carries an implicit H⁻¹ envelope term ≠ energy[g],
        // so value and gradient would descend different functions (the desync bug).
        let deviance = 2.0
            * self
                .term
                .penalized_objective(self.target.view(), &lambda_smooth, &self.ard_precisions)
                .map_err(outer_error)?;
        if !(deviance.is_finite() && deviance > 0.0) {
            return Err(outer_error(format!(
                "support LAML requires positive finite penalized deviance; got {deviance}"
            )));
        }
        let (beta_offsets, beta_dim) = self.beta_layout()?;
        // The pinned directions of atoms no row selects are not estimated, so they are
        // not charged as fitted unpenalized coefficients.
        let beta_nullity = beta_dim
            .checked_sub(self.spectrum.total_rank)
            .and_then(|nullity| nullity.checked_sub(pinned_dims))
            .ok_or_else(|| outer_error("support smooth penalty rank exceeds beta dimension"))?;
        let data_dim = self
            .term
            .n_obs()
            .checked_mul(self.term.output_dim())
            .ok_or_else(|| outer_error("support LAML data dimension overflow"))?;
        if data_dim <= beta_nullity {
            return Err(outer_error(format!(
                "support LAML requires more response cells than unpenalized decoder coefficients; got {data_dim} <= {beta_nullity}"
            )));
        }
        let residual_df = (data_dim - beta_nullity) as f64;
        let mut penalty_logdet = 0.0;
        for group in 0..self.layout.group_keys.len() {
            penalty_logdet += self.spectrum.log_pdet_base_by_group[group]
                + self.spectrum.rank_by_group[group] as f64 * rho[group];
        }
        let components = SaeSupportLamlComponents {
            reduced_log_det: reduced_logdet,
            penalty_log_pdet: penalty_logdet,
            residual_df,
            penalized_deviance: deviance,
        };
        let cost = components.value();
        // The surrogate's directional derivative is the exact FIXED-STATE
        // derivative of the very `log|S|` that entered `cost` above — same
        // probes, quadrature nodes, frozen deflation basis and shifted solves.
        // A profiled criterion needs one more term: S also moves because the
        // converged decoder/coordinates move with rho. The support term forms
        // `Gamma = d log|S|/d theta` from this same bundle and solves the exact
        // inner stationarity adjoint once; omitting that IFT response is the
        // value/gradient split that made every descent direction point uphill.
        // The lane used to estimate this trace with a SECOND, independent
        // Hutchinson family (its own 16 probes, its own unshifted `S⁻¹` solves,
        // one per group), which both cost the whole evaluation's wall-clock and
        // left the value and gradient free to describe different functions.
        let energy = self.penalty_energy_by_group(&lambda_smooth);
        let profile_adjoint = self
            .term
            .support_reduced_logdet_profile_adjoint(
                self.target.view(),
                &self.ard_precisions,
                &system,
                &logdet_derivative.vectors,
            )
            .map_err(outer_error)?;
        let mut gradient = Array1::<f64>::zeros(self.layout.group_keys.len());
        for group in 0..gradient.len() {
            let derivative_matvec = |vector: ArrayView1<f64>| -> Array1<f64> {
                self.schur_derivative_matvec(
                    group,
                    &lambda_smooth,
                    &beta_offsets,
                    beta_dim,
                    vector,
                )
            };
            let logdet_group_derivative = logdet_derivative
                .directional_derivative(&derivative_matvec)
                .ok_or_else(|| {
                    outer_error(format!(
                        "support LAML reduced-Schur surrogate produced no derivative for \
                         smoothing group {group} ({})",
                        self.layout.group_keys[group]
                    ))
                })?;
            // `g_rho = d(∇_theta L)/d rho` lives only in the decoder
            // block and equals the group penalty applied to beta. With
            // `a = A^+ Gamma`, the implicit logdet response is
            // `-1/2 <a, g_rho>` (one adjoint, then one cheap contraction per
            // smoothing group).
            let mut profile_response = 0.0_f64;
            for atom in 0..self.term.k_atoms() {
                if self.layout.atom_group[atom] != group {
                    continue;
                }
                let m = self.term.atoms[atom].basis_size();
                let offset = beta_offsets[atom];
                let lambda = lambda_smooth[atom];
                let sb = self.term.atoms[atom]
                    .smooth_penalty()
                    .dot(self.term.atoms[atom].decoder_coefficients());
                for basis in 0..m {
                    for output in 0..self.term.output_dim() {
                        profile_response += profile_adjoint.beta
                            [offset + basis * self.term.output_dim() + output]
                            * lambda
                            * sb[[basis, output]];
                    }
                }
            }
            gradient[group] = 0.5
                * (logdet_group_derivative - self.spectrum.rank_by_group[group] as f64
                    + residual_df * energy[group] / deviance
                    - profile_response);
        }
        if !cost.is_finite() || gradient.iter().any(|value| !value.is_finite()) {
            return Err(outer_error(
                "support LAML produced a non-finite value or gradient",
            ));
        }
        Ok(SupportOuterEvaluation {
            cost,
            components,
            gradient,
            lambda_smooth,
            fixed_point,
        })
    }
}

impl OuterObjective for SaeSupportOuterObjective {
    fn capability(&self) -> OuterCapability {
        OuterCapability {
            gradient: Derivative::Analytic,
            hessian: DeclaredHessianForm::Unavailable,
            n_params: self.layout.group_keys.len(),
            psi_dim: 0,
            fixed_point_available: false,
            barrier_config: None,
            prefer_gradient_only: true,
            disable_fixed_point: true,
        }
    }

    fn eval_cost(&mut self, rho: &Array1<f64>) -> Result<f64, EstimationError> {
        self.evaluate_for_order(rho, OuterEvalOrder::Value)
            .map(|evaluation| evaluation.cost)
    }

    fn eval(&mut self, rho: &Array1<f64>) -> Result<OuterEval, EstimationError> {
        let evaluation = self.evaluate_for_order(rho, OuterEvalOrder::ValueAndGradient)?;
        Ok(OuterEval {
            cost: evaluation.cost,
            gradient: evaluation.gradient,
            hessian: HessianValue::Unavailable,
            inner_beta_hint: None,
        })
    }

    fn eval_with_order(
        &mut self,
        rho: &Array1<f64>,
        order: OuterEvalOrder,
    ) -> Result<OuterEval, EstimationError> {
        let evaluation = self.evaluate_for_order(rho, order)?;
        match order {
            OuterEvalOrder::Value => Ok(OuterEval::value_only(evaluation.cost, rho.len(), None)),
            OuterEvalOrder::ValueAndGradient | OuterEvalOrder::ValueGradientHessian => {
                Ok(OuterEval {
                    cost: evaluation.cost,
                    gradient: evaluation.gradient,
                    hessian: HessianValue::Unavailable,
                    inner_beta_hint: None,
                })
            }
        }
    }

    fn reset(&mut self) {
        self.term = self.initial_term.clone();
        self.last_evaluation = None;
        self.state_revision = self.state_revision.wrapping_add(1);
    }

    fn seed_inner_state(&mut self, beta: &Array1<f64>) -> Result<SeedOutcome, EstimationError> {
        if beta.iter().any(|value| !value.is_finite()) {
            return Err(outer_error(
                "support outer seed contains a non-finite value",
            ));
        }
        Ok(SeedOutcome::NoSlot)
    }
}

/// The grouped smoothing coordinates' domain (#2812) at the entry state. Each
/// atom's decoder data Gram against its native smooth penalty gives generalized
/// eigenvalues `γ_j` on the penalty range. A group's eigenvalues are the union
/// over its atoms, and its domain is `[ln(√ε·γ_min), ln(γ_max/√ε)]`: past either
/// face every direction's share of the ρ-gradient is under its own round-off,
/// so a railed strength is a structural result. A group whose penalized
/// directions carry no data curvature keeps the precision box. This replaces
/// the outer engine's ±30 fallback on this lane (SPEC rule 20, #2902 row 8).
fn support_smoothing_domain(
    term: &SaeSupportSparseTerm,
    layout: &SaeSupportSmoothingLayout,
) -> Result<(Array1<f64>, Array1<f64>), String> {
    use gam_solve::estimate::rho_domain;
    let groups = layout.group_keys.len();
    let mut gammas_by_group = vec![Vec::<f64>::new(); groups];
    for atom_idx in 0..term.k_atoms() {
        let gram = term.atom_decoder_gram(atom_idx)?;
        if let Some(gammas) =
            rho_domain::penalty_range_gammas_from_gram(&gram, term.atoms[atom_idx].smooth_penalty())
        {
            gammas_by_group[layout.atom_group[atom_idx]].extend(gammas);
        }
    }
    let mut lower = Array1::<f64>::zeros(groups);
    let mut upper = Array1::<f64>::zeros(groups);
    for (group, gammas) in gammas_by_group.iter().enumerate() {
        let (lo, hi) =
            rho_domain::coordinate_domain(rho_domain::resolvability_interval(gammas), None);
        lower[group] = lo;
        upper[group] = hi;
    }
    Ok((lower, upper))
}

/// Select topology-grouped smoothing strengths through the shared generic
/// outer optimizer. Only a terminal point with an analytic stationarity
/// certificate and a recurring raw inner fixed point is returned.
pub fn run_sae_support_outer(
    request: SaeSupportOuterRequest,
) -> Result<SaeSupportOuterReport, EstimationError> {
    if !(request.initial_smoothness.is_finite() && request.initial_smoothness > 0.0) {
        return Err(outer_error(format!(
            "support outer initial_smoothness must be finite and positive; got {}",
            request.initial_smoothness
        )));
    }
    let layout = SaeSupportSmoothingLayout::from_term(&request.term);
    if layout.group_keys.is_empty() {
        return Err(outer_error(
            "support outer requires at least one smoothing group",
        ));
    }
    let spectrum = penalty_spectrum(&request.term, &layout).map_err(outer_error)?;
    let (rho_lower, rho_upper) =
        support_smoothing_domain(&request.term, &layout).map_err(outer_error)?;
    // This Gaussian LAML criterion is a sum over response cells, not a
    // unit-scale scalar.  Declare that natural scale to the shared outer
    // engine so its projected-gradient stopping band has the same units as
    // the score.  Without it BFGS used the bare absolute 1e-5 default even
    // when the score-relative certificate at the same point was already
    // decisive; on the small overcomplete Tier-2 witness that drove Strong
    // Wolfe into sub-1e-8 rho probes whose predicted score change was below
    // floating-point cancellation, with each probe paying for a full inner
    // fixed point and rational evidence evaluation.
    let objective_scale = request.target.len() as f64;
    let initial_term = request.term.clone();
    // The inner fixed point certifies to the resolution its own objective has, not to
    // a caller's number, so every driver of this engine gets the same derived value.
    let inner_tolerance = request.term.fixed_point_tolerance();
    let mut objective = SaeSupportOuterObjective {
        term: request.term,
        initial_term,
        target: request.target,
        layout: layout.clone(),
        spectrum,
        ard_precisions: request.ard_precisions.clone(),
        max_inner_iter: request.max_inner_iter,
        inner_tolerance,
        trust_radius: request.trust_radius,
        random_state: request.random_state,
        last_evaluation: None,
        state_revision: 0,
        uncached_evaluations: 0,
        logdet_surrogate: None,
    };
    // The caller's smoothness, placed in the domain the search has.
    let initial_log_smoothness = request.initial_smoothness.ln();
    let initial_rho = Array1::from_shape_fn(layout.group_keys.len(), |group| {
        initial_log_smoothness.max(rho_lower[group]).min(rho_upper[group])
    });
    let problem = OuterProblem::new(layout.group_keys.len())
        .with_gradient(Derivative::Analytic)
        .with_hessian(DeclaredHessianForm::Unavailable)
        .with_prefer_gradient_only(true)
        .with_disable_fixed_point(true)
        .with_objective_scale(Some(objective_scale))
        .with_bounds(rho_lower, rho_upper)
        .with_initial_rho(initial_rho)
        .with_max_iter(request.max_outer_iter.max(1));
    let outer = problem.run(&mut objective, SUPPORT_LAML_CONTEXT)?;
    let certificate = outer
        .criterion_certificate
        .clone()
        .filter(OuterCriterionCertificate::certifies)
        .ok_or_else(|| {
            outer_error(format!(
                "support outer returned without an analytic stationarity certificate after {} iterations",
                outer.iterations
            ))
        })?;
    let terminal = objective.evaluate(&outer.rho)?;
    if !terminal.fixed_point.recurred {
        return Err(outer_error(
            "support outer terminal inner state did not recur",
        ));
    }
    Ok(SaeSupportOuterReport {
        term: objective.term,
        smoothing_layout: layout,
        log_lambda_groups: outer.rho,
        lambda_smooth: terminal.lambda_smooth,
        ard_precisions: request.ard_precisions,
        criterion: SaeCriterionScore::new(SaeCriterionKind::ProfiledGaussianLaml, terminal.cost),
        criterion_components: terminal.components,
        fixed_point: terminal.fixed_point,
        outer_iterations: outer.iterations,
        outer_certificate: certificate,
        inner_tolerance,
    })
}

/// Request for [`fit_sae_support_sparse`]: one overcomplete hard-TopK
/// support-sparse fit of an uncentered `N×P` target.
pub struct SaeSupportSparseFitRequest<'a> {
    /// Activations `N×P`. The fit centers them on their column mean.
    pub target: ndarray::ArrayView2<'a, f64>,
    /// Per-atom chart family. `"auto"` entries resolve through
    /// [`resolve_support_auto_atoms`].
    pub atom_basis: Vec<String>,
    /// Per-atom public dimension.
    pub atom_dim: Vec<usize>,
    /// Per-row TopK support width.
    pub support_k: usize,
    /// Initial isotropic smoothing strength seeding the outer LAML search.
    pub initial_smoothness: f64,
    /// Outer (smoothing-selection) iteration budget.
    pub max_outer_iter: usize,
    /// Inner fixed-point iteration budget.
    pub max_inner_iter: usize,
    /// Inner coordinate trust radius.
    pub trust_radius: f64,
    /// Deterministic seed for the support routing and the evidence probes.
    pub random_state: u64,
}

/// A converged overcomplete support-sparse fit, in target coordinates.
pub struct SaeSupportSparseFit {
    /// The certified outer report: converged term, selected smoothing, criterion
    /// and both certificates.
    pub outer: SaeSupportOuterReport,
    /// Atoms requested, before the seed prunes zero-mass atoms.
    pub requested_atoms: usize,
    /// Indices of the requested atoms the seed retained.
    pub retained_atom_indices: Vec<usize>,
    /// Resolved chart family of each retained atom.
    pub atom_basis: Vec<String>,
    /// Public dimension of each retained atom.
    pub atom_dim: Vec<usize>,
    /// Column mean the target was centered on, length `P`.
    pub training_mean: Vec<f64>,
    /// In-sample reconstruction `training_mean + Ĉ`, `N×P`.
    pub fitted: Array2<f64>,
    /// `1 − RSS/TSS` against the training mean; `1` when the target has no
    /// variance about its mean.
    pub reconstruction_r2: f64,
    /// Every birth and death of this fit. The support lane has no reseed: each
    /// atom is born once, at the support seed, from the centered rows' own
    /// projections, and an atom no row selected is pruned at that boundary. A
    /// support move re-routes rows among the retained atoms and can leave one with
    /// no row; that atom stays in the term, reconstructs nothing, and is recorded
    /// as a death at the end. So births are the retained atoms, deaths are every
    /// requested atom not live at the end, and `pc_reseed_events` is `0`.
    pub migration: SaeMigrationLedger,
}

/// The one production path for an overcomplete (`K > P`) hard-TopK manifold SAE
/// fit (#2023). It resolves the `"auto"` portfolio, admits the support-sparse
/// lane, centers on the training mean, seeds the support routing and the charts,
/// and selects the grouped-LAML smoothing through [`run_sae_support_outer`]. The
/// public FFI entry, the tiered Tier-2 refinement and the code-space pair chart
/// all fit through it, so none of them carries its own copy of the cadence.
pub fn fit_sae_support_sparse(
    request: SaeSupportSparseFitRequest<'_>,
) -> Result<SaeSupportSparseFit, String> {
    let (n_obs, output_dim) = request.target.dim();
    let requested_atoms = request.atom_basis.len();
    let mut resolved_basis = request.atom_basis;
    resolve_support_auto_atoms(&mut resolved_basis);
    let effective_dims = sae_support_effective_atom_dims(&resolved_basis, &request.atom_dim)?;
    let d_max = effective_dims.iter().copied().max().unwrap_or(1);
    let admission = crate::front_door::admit_topk_manifold(
        n_obs,
        output_dim,
        requested_atoms,
        d_max,
        request.support_k,
    )?;
    if admission.lane != crate::front_door::SaeFitLane::CurvedStreaming {
        return Err(format!(
            "support-sparse fit requires CurvedStreaming admission; got {:?}",
            admission.lane
        ));
    }
    let training_mean = request
        .target
        .mean_axis(ndarray::Axis(0))
        .ok_or_else(|| "support-sparse fit requires positive rows".to_string())?
        .to_vec();
    let centered = Array2::from_shape_fn((n_obs, output_dim), |(row, column)| {
        request.target[[row, column]] - training_mean[column]
    });
    let seed = build_sae_support_seed(SaeSupportSeedRequest {
        target: centered.view(),
        atom_basis: &resolved_basis,
        atom_dim: &request.atom_dim,
        support_k: request.support_k,
        random_state: request.random_state,
        admission,
    })?;
    let retained_atom_indices = seed.retained_atom_indices;
    let atom_basis = retained_atom_indices
        .iter()
        .map(|&atom| resolved_basis[atom].clone())
        .collect::<Vec<_>>();
    let atom_dim = retained_atom_indices
        .iter()
        .map(|&atom| request.atom_dim[atom])
        .collect::<Vec<_>>();
    let term_seed = build_sae_support_term_seed(SaeSupportTermSeedRequest {
        assignment: seed.assignment,
        atom_basis: atom_basis.clone(),
        atom_dim: atom_dim.clone(),
        output_dim,
        random_state: request.random_state,
    })?;
    let ard_precisions = (0..term_seed.term.k_atoms())
        .map(|atom| vec![1.0; term_seed.term.assignment.atom_coord_dim(atom)])
        .collect::<Vec<_>>();
    let outer = run_sae_support_outer(SaeSupportOuterRequest {
        term: term_seed.term,
        target: centered,
        initial_smoothness: request.initial_smoothness,
        ard_precisions,
        max_outer_iter: request.max_outer_iter,
        max_inner_iter: request.max_inner_iter,
        trust_radius: request.trust_radius,
        random_state: request.random_state,
    })
    .map_err(|error| error.to_string())?;
    let mut fitted = outer.term.reconstruct()?;
    let mut residual_ss = 0.0_f64;
    let mut total_ss = 0.0_f64;
    for row in 0..n_obs {
        for column in 0..output_dim {
            fitted[[row, column]] += training_mean[column];
            let truth = request.target[[row, column]];
            residual_ss += (truth - fitted[[row, column]]).powi(2);
            total_ss += (truth - training_mean[column]).powi(2);
        }
    }
    let reconstruction_r2 = if total_ss > 0.0 {
        1.0 - residual_ss / total_ss
    } else {
        1.0
    };
    let mut migration = SaeMigrationLedger::new();
    migration.birth(
        MoveStage::Curved,
        BirthSeed::ResidualFactor,
        retained_atom_indices.len(),
        Some(0),
        MoveEvidence::none(),
        outer.criterion.value(),
    );
    let pruned = requested_atoms - retained_atom_indices.len();
    if pruned > 0 {
        migration.death(
            MoveStage::Curved,
            MoveReason::DeadRouting,
            pruned,
            Some(0),
            MoveEvidence::none(),
            outer.criterion.value(),
        );
    }
    // A support move re-routes rows among the retained atoms and keeps every atom
    // in the term, so one it left with no row is a death after the seed.
    let mut occupied = vec![false; outer.term.k_atoms()];
    for row in 0..outer.term.n_obs() {
        for &atom in outer.term.assignment.support_indices(row) {
            occupied[atom as usize] = true;
        }
    }
    let emptied = occupied.iter().filter(|&&used| !used).count();
    if emptied > 0 {
        migration.death(
            MoveStage::Curved,
            MoveReason::DeadRouting,
            emptied,
            None,
            MoveEvidence::none(),
            outer.criterion.value(),
        );
    }
    log::info!(
        "support-sparse migration ledger: {} births, {} deaths, {} refusals, {} \
         principal-component reseeds",
        migration.n_births,
        migration.n_deaths,
        migration.n_refusals,
        migration.pc_reseed_events,
    );
    Ok(SaeSupportSparseFit {
        outer,
        requested_atoms,
        retained_atom_indices,
        atom_basis,
        atom_dim,
        training_mean,
        fitted,
        reconstruction_r2,
        migration,
    })
}

/// A public overcomplete support-sparse fit together with its linear-bulk census.
pub struct SaeSupportSparseCensusedFit {
    /// The curved fit of the Tier-0-centered target.
    pub fit: SaeSupportSparseFit,
    /// The linear bulk at the derived width and its code-space census
    /// ([`crate::tiered::linear_bulk_census`]), or why it refused. It audits the
    /// curved fit and never changes it, so a refusal is reported, not raised.
    pub linear_bulk_census: Result<crate::tiered::TieredFitReport, String>,
}

impl SaeSupportSparseCensusedFit {
    /// The census as a payload record, or `{"refused": reason}`.
    #[must_use]
    pub fn census_json(&self) -> serde_json::Value {
        match &self.linear_bulk_census {
            Ok(report) => report.census_json(),
            Err(reason) => serde_json::json!({ "refused": reason }),
        }
    }
}

/// The public overcomplete support-sparse entry (#2023 lead ruling). It runs the
/// curved fit of the Tier-0-centered target through [`fit_sae_support_sparse`], plus
/// the Tier-1 linear bulk at the derived width with its code-space census: the
/// adjudicated account, in bits, of which linear communities would curve.
pub fn fit_sae_support_sparse_with_census(
    request: SaeSupportSparseFitRequest<'_>,
) -> Result<SaeSupportSparseCensusedFit, String> {
    let target = request.target;
    let support_k = request.support_k;
    let fit = fit_sae_support_sparse(request)?;
    let d_max = sae_support_effective_atom_dims(&fit.atom_basis, &fit.atom_dim)?
        .into_iter()
        .max()
        .unwrap_or(1);
    let linear_bulk_census = crate::tiered::linear_bulk_census(target, d_max, support_k);
    match &linear_bulk_census {
        Ok(report) => log::info!(
            "support-sparse linear-bulk census: {} blocks of size {}, EV {:.6}, {} of {} \
             communities curve ({:.1} bits saved)",
            report.tier1.block_utilization.len(),
            report.tier1.block_size,
            report.tier1.explained_variance,
            report.code_space.n_accepted,
            report.code_space.n_communities,
            report.code_space.dl_saved_bits,
        ),
        Err(reason) => log::warn!("support-sparse linear-bulk census refused: {reason}"),
    }
    Ok(SaeSupportSparseCensusedFit {
        fit,
        linear_bulk_census,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assignment_state::{SaeAssignmentAtomSpec, SaeAssignmentState};
    use ndarray::array;
    use std::sync::Arc;

    fn atom(
        name: &str,
        kind: SaeAtomBasisKind,
        d: usize,
        evaluator: Arc<dyn SaeBasisSecondJet>,
        coords: &[f64],
        decoder: Array2<f64>,
    ) -> SaeManifoldAtom {
        let coord = Array2::from_shape_vec((1, d), coords.to_vec()).expect("coords");
        let (phi, jet) = evaluator.evaluate(coord.view()).expect("evaluate");
        let m = phi.ncols();
        SaeManifoldAtom::new_with_provided_function_gram(
            name,
            kind,
            d,
            phi,
            jet,
            decoder,
            Array2::eye(m),
        )
        .expect("atom")
        .with_basis_second_jet(evaluator)
    }

    /// Two-group heterogeneous support fixture (K = 2 atoms > P = 1 response
    /// cells) whose smoothing selection is the exact regime the outer criterion
    /// desync afflicts. The target leaves a genuine residual so the penalized
    /// deviance and every per-group penalty energy are strictly positive.
    ///
    /// The periodic latent coordinate is seeded AWAY from a quarter period. With
    /// one active row per atom the exact ridge decoder block makes the fitted row
    /// interpolate the target, so the likelihood coordinate Jacobian
    /// `b·phi'(t)` collapses and the coordinate normal equation is carried
    /// entirely by the von-Mises PSD majorizer `max(alpha·cos(kappa·t), 0)`. That
    /// majorizer is exactly zero at `t = p/4` (period `p`), where the prior
    /// gradient `(alpha/kappa)·sin(kappa·t)` is still maximal — a zero-curvature
    /// block with a non-zero right-hand side, which `solve_psd_minimum_norm`
    /// (correctly) refuses as an RHS in the normal-equation null space. Seeding at
    /// `t = 0.1` (period 1) keeps `cos(kappa·t) = 0.809 > 0`, and because the
    /// von-Mises prior pulls the coordinate toward its mode at `t = 0`, every
    /// later iterate moves further from `p/4`, so the coordinate block stays
    /// strictly positive definite all the way to the recurring fixed point.
    fn build_objective() -> SaeSupportOuterObjective {
        let periodic_eval: Arc<dyn SaeBasisSecondJet> =
            Arc::new(PeriodicHarmonicEvaluator::new(3).expect("periodic"));
        let patch_eval: Arc<dyn SaeBasisSecondJet> =
            Arc::new(EuclideanPatchEvaluator::new(2, 1).expect("patch"));
        let atoms = vec![
            atom(
                "circle",
                SaeAtomBasisKind::Periodic,
                1,
                periodic_eval,
                &[0.3],
                array![[0.2], [1.1], [-0.4]],
            ),
            atom(
                "plane",
                SaeAtomBasisKind::Linear,
                2,
                patch_eval,
                &[0.1, -0.2],
                array![[0.3], [2.0], [-1.0]],
            ),
        ];
        let specs = vec![
            SaeAssignmentAtomSpec {
                latent_dim: 1,
                manifold: SaeAtomBasisKind::Periodic.latent_manifold(1),
                retraction: gam_problem::LatentRetractionRegistry::all_euclidean(),
            },
            SaeAssignmentAtomSpec::euclidean(2),
        ];
        let state = SaeAssignmentState::from_topk_support_heterogeneous(
            2,
            2,
            1,
            specs,
            vec![vec![0], vec![1]],
            vec![vec![9.0], vec![-4.0]],
            vec![vec![0.1], vec![3.0, 1.0]],
        )
        .expect("state");
        let term = SaeSupportSparseTerm::new(atoms, state).expect("term");
        let layout = SaeSupportSmoothingLayout::from_term(&term);
        assert_eq!(layout.group_keys.len(), 2, "fixture must expose two groups");
        let spectrum = penalty_spectrum(&term, &layout).expect("spectrum");
        let initial_term = term.clone();
        SaeSupportOuterObjective {
            term,
            initial_term,
            target: array![[1.4], [4.3]],
            layout,
            spectrum,
            ard_precisions: vec![vec![1.0], vec![1.0, 1.0]],
            max_inner_iter: 5000,
            inner_tolerance: 1.0e-9,
            trust_radius: 1.0,
            random_state: 0xC0FF_EE00_D15E_A5E5,
            last_evaluation: None,
            state_revision: 0,
            uncached_evaluations: 0,
            logdet_surrogate: None,
        }
    }

    /// #2634 — the first-order bridge asks for a value during line search and
    /// then for value+gradient at the accepted point. This objective computes
    /// both in one evidence pass, so an exact repeated rho must consume that
    /// pass once. A one-bit-different rho is the negative control: the cache is
    /// exact, never a tolerance-based warm-start shortcut.
    #[test]
    fn identical_value_then_gradient_evaluates_support_laml_once_2634() {
        let mut objective = build_objective();
        let rho = array![0.0, 0.0];
        let value = objective
            .eval_with_order(&rho, OuterEvalOrder::Value)
            .expect("value evaluation");
        assert_eq!(objective.uncached_evaluations, 1);

        let value_and_gradient = objective
            .eval_with_order(&rho, OuterEvalOrder::ValueAndGradient)
            .expect("the full evaluation reuses the value pass");
        assert_eq!(
            objective.uncached_evaluations, 1,
            "an identical rho must not repeat the inner solve and evidence pass"
        );
        assert_eq!(value.cost.to_bits(), value_and_gradient.cost.to_bits());
        assert!(
            value_and_gradient.gradient.iter().all(|value| value.is_finite()),
            "the reused full evaluation must retain its analytic gradient"
        );

        let mut adjacent = rho.clone();
        adjacent[0] = f64::from_bits(rho[0].to_bits() + 1);
        objective
            .eval_with_order(&adjacent, OuterEvalOrder::Value)
            .expect("one-bit-adjacent rho evaluation");
        assert_eq!(
            objective.uncached_evaluations, 2,
            "a one-bit rho change must invalidate the exact cache"
        );
    }

    /// #2634 root gate: a smoothing gradient is the derivative of the fully
    /// profiled production criterion, not the partial derivative obtained by
    /// freezing the coordinates and decoder at the centre point. The latter
    /// dropped the `Gamma^T theta_hat_rho` Laplace response and, together with
    /// the spurious row-coordinate determinant, reported an uphill direction
    /// as descent on the filed Tier-2 route.
    #[test]
    fn profiled_support_outer_gradient_matches_refitted_value_fd_2634() {
        let mut objective = build_objective();
        let base = array![0.4_f64.ln(), 2.2_f64.ln()];
        let analytic = objective
            .evaluate(&base)
            .expect("profiled centre evaluation")
            .gradient;
        let h = 2.0e-4;
        for group in 0..base.len() {
            let mut plus = base.clone();
            let mut minus = base.clone();
            plus[group] += h;
            minus[group] -= h;

            // Both legs start from the same pristine inner state. The frozen
            // rational plan intentionally survives reset so centre and legs
            // are values of one deterministic surrogate.
            objective.reset();
            let value_plus = objective
                .evaluate(&plus)
                .expect("profiled plus evaluation")
                .cost;
            objective.reset();
            let value_minus = objective
                .evaluate(&minus)
                .expect("profiled minus evaluation")
                .cost;
            let finite_difference = (value_plus - value_minus) / (2.0 * h);
            let gap = (analytic[group] - finite_difference).abs();
            let scale = 1.0 + analytic[group].abs().max(finite_difference.abs());
            eprintln!(
                "#2634 profiled FD group {group} ({}): analytic={:.12e}, central_fd={finite_difference:.12e}, gap={gap:.3e}",
                objective.layout.group_keys[group],
                analytic[group],
            );
            assert!(
                gap <= 5.0e-4 * scale,
                "group {group} ({}): profiled analytic gradient {:.9e} != refitted production-value FD {finite_difference:.9e} (gap {gap:.3e}, scale {scale:.3e})",
                objective.layout.group_keys[group],
                analytic[group],
            );
        }
    }

    /// Solve the inner fixed point cleanly from the initial term at `rho`
    /// (rebuilding the whole cache — never freezing it, per the FD-gate rule)
    /// and read off the penalized deviance `D_p = 2·penalized_objective` and the
    /// raw residual sum of squares at that converged inner optimum.
    fn deviance_and_rss(objective: &mut SaeSupportOuterObjective, rho: &Array1<f64>) -> (f64, f64) {
        objective.reset();
        let lambda = objective.layout.expand(rho).expect("expand");
        objective
            .term
            .solve_fixed_point(
                objective.target.view(),
                &lambda,
                &objective.ard_precisions,
                objective.max_inner_iter,
                objective.inner_tolerance,
                objective.trust_radius,
            )
            .expect("inner fixed point");
        let deviance = 2.0
            * objective
                .term
                .penalized_objective(objective.target.view(), &lambda, &objective.ard_precisions)
                .expect("penalized objective");
        let residual = objective
            .term
            .raw_residual(objective.target.view())
            .expect("raw residual");
        let rss = residual.iter().map(|value| value * value).sum::<f64>();
        (deviance, rss)
    }

    /// Decisive oracle for the value↔gradient desync. The outer value feeds the
    /// penalized deviance `D_p` into the Gaussian dispersion term, and the
    /// analytic gradient's dispersion channel is `½·residual_df·energy[g]/D_p`
    /// with `energy[g] = Σ_{k∈g} λ_k β̂ᵀ S_k β̂ = penalty_energy_by_group`. Since
    /// `residual_df` and `D_p` are common factors, value/gradient consistency of
    /// that channel is EXACTLY the envelope identity `d D_p/dρ_g = energy[g]`.
    /// Central-differencing the production `D_p` (with a full clean inner re-solve
    /// at each ρ±h) must reproduce the production `energy[g]` — this is the FD
    /// oracle the SPEC allows in tests. The same test also confirms that the RAW
    /// residual sum of squares does NOT satisfy the identity (its derivative
    /// carries the implicit `H⁻¹` envelope term), so a revert to raw RSS is caught.
    #[test]
    fn support_penalized_deviance_derivative_equals_penalty_energy() {
        let mut objective = build_objective();
        // λ deliberately away from 1 in both groups so the raw-RSS derivative and
        // the penalized-deviance derivative are unmistakably different functions.
        let base = array![0.35_f64.ln(), 2.8_f64.ln()];
        let groups = objective.layout.group_keys.len();

        // Production `energy[g]` at the base inner optimum.
        objective.reset();
        let lambda_base = objective.layout.expand(&base).expect("expand");
        objective
            .term
            .solve_fixed_point(
                objective.target.view(),
                &lambda_base,
                &objective.ard_precisions,
                objective.max_inner_iter,
                objective.inner_tolerance,
                objective.trust_radius,
            )
            .expect("base inner fixed point");
        let energy = objective.penalty_energy_by_group(&lambda_base);
        assert_eq!(energy.len(), groups);
        assert!(
            energy.iter().all(|value| value.is_finite() && *value > 0.0),
            "fixture must exercise strictly positive per-group penalty energy: {energy:?}"
        );

        let h = 1.0e-4;
        let energy_scale = energy
            .iter()
            .fold(0.0_f64, |acc, value| acc.max(value.abs()));
        let mut max_raw_gap = 0.0_f64;
        for g in 0..groups {
            let mut plus = base.clone();
            let mut minus = base.clone();
            plus[g] += h;
            minus[g] -= h;
            let (dev_plus, rss_plus) = deviance_and_rss(&mut objective, &plus);
            let (dev_minus, rss_minus) = deviance_and_rss(&mut objective, &minus);
            let deviance_derivative = (dev_plus - dev_minus) / (2.0 * h);
            let raw_rss_derivative = (rss_plus - rss_minus) / (2.0 * h);

            // (1) Envelope identity: d D_p/dρ_g == energy[g] (value↔gradient match).
            let envelope_gap = (deviance_derivative - energy[g]).abs();
            assert!(
                envelope_gap <= 1.0e-6 * (1.0 + energy[g].abs()),
                "group {g}: penalized-deviance derivative {deviance_derivative:.9e} \
                 disagrees with penalty energy {:.9e} (gap {envelope_gap:.3e})",
                energy[g]
            );

            // (2) Raw RSS is a DIFFERENT function: its derivative must not be the
            // penalty energy, proving the raw-RSS gradient (the fixed defect) desyncs.
            max_raw_gap = max_raw_gap.max((raw_rss_derivative - energy[g]).abs());
        }
        assert!(
            max_raw_gap > 1.0e-2 * energy_scale.max(1.0e-3),
            "raw-RSS derivative must visibly differ from the penalty energy so the \
             desync is caught (max gap {max_raw_gap:.3e}, energy scale {energy_scale:.3e})"
        );
    }

    /// Full-production oracle: the analytic gradient returned by `evaluate` must
    /// match a central difference of the production value `cost`, restricted to
    /// the dispersion channel that the desync corrupted. The joint log-det term is
    /// a fixed-probe Hutchinson estimate (deterministic but not exact), so we
    /// isolate the dispersion channel by subtracting the exact analytic log-det
    /// and penalty-log-det contributions — both computed from the same production
    /// quantities — leaving `½·residual_df·(1+ln(τ·D_p/df))`, whose FD must equal
    /// the analytic `½·residual_df·energy[g]/D_p`.
    #[test]
    fn support_outer_value_dispersion_channel_matches_gradient() {
        let mut objective = build_objective();
        let base = array![0.4_f64.ln(), 2.2_f64.ln()];
        let groups = objective.layout.group_keys.len();

        // residual_df is a ρ-independent constant of the fixture.
        let (_, beta_dim) = objective.beta_layout().expect("beta layout");
        let beta_nullity = beta_dim - objective.spectrum.total_rank;
        let data_dim = objective.term.n_obs() * objective.term.output_dim();
        let residual_df = (data_dim - beta_nullity) as f64;
        assert!(residual_df > 0.0);

        let dispersion_value =
            |objective: &mut SaeSupportOuterObjective, rho: &Array1<f64>| -> f64 {
                let (deviance, _) = deviance_and_rss(objective, rho);
                0.5 * residual_df * (1.0 + (std::f64::consts::TAU * deviance / residual_df).ln())
            };

        // Analytic dispersion-channel gradient from production quantities.
        objective.reset();
        let lambda_base = objective.layout.expand(&base).expect("expand");
        objective
            .term
            .solve_fixed_point(
                objective.target.view(),
                &lambda_base,
                &objective.ard_precisions,
                objective.max_inner_iter,
                objective.inner_tolerance,
                objective.trust_radius,
            )
            .expect("base inner fixed point");
        let deviance_base = 2.0
            * objective
                .term
                .penalized_objective(
                    objective.target.view(),
                    &lambda_base,
                    &objective.ard_precisions,
                )
                .expect("penalized objective");
        let energy = objective.penalty_energy_by_group(&lambda_base);

        let h = 1.0e-4;
        for g in 0..groups {
            let analytic = 0.5 * residual_df * energy[g] / deviance_base;
            let mut plus = base.clone();
            let mut minus = base.clone();
            plus[g] += h;
            minus[g] -= h;
            let fd = (dispersion_value(&mut objective, &plus)
                - dispersion_value(&mut objective, &minus))
                / (2.0 * h);
            assert!(
                (analytic - fd).abs() <= 1.0e-6 * (1.0 + analytic.abs()),
                "group {g}: analytic dispersion gradient {analytic:.9e} != FD {fd:.9e}"
            );
        }
    }

    /// #2576 regression, the blunt one: the criterion must PRODUCE A NUMBER.
    ///
    /// Before this issue it could not, on any host without a CUDA device. The
    /// lane took its factor cache from `solve_arrow_newton_step_with_options`
    /// under `InexactPCG`, which never forms a dense reduced-Schur factor, so
    /// `schur_factor_is_undamped` stayed false, `compute_undamped_arrow_log_det`
    /// returned `None` for `k > 0`, and `arrow_log_det()` refused —
    /// `"support LAML factor cache has no joint log determinant"` on the very
    /// first outer evaluation, before the trace estimator the issue was filed
    /// about was ever reached. The whole grouped-LAML engine was dead code with
    /// no test that would notice, because every existing test isolated one
    /// channel and none called `evaluate`.
    ///
    /// This test calls it. It is also the CPU-only gate: nothing here builds a
    /// device operator, so it exercises the host lane that used to have no
    /// matrix-free `log|S|` at all.
    #[test]
    fn support_outer_evaluate_mints_a_finite_value_and_gradient() {
        let mut objective = build_objective();
        let rho = array![0.4_f64.ln(), 2.2_f64.ln()];
        let evaluation = objective
            .evaluate(&rho)
            .expect("the grouped-LAML criterion must evaluate on a CPU-only host");
        assert!(
            evaluation.cost.is_finite(),
            "criterion value must be finite, got {}",
            evaluation.cost
        );
        assert_eq!(evaluation.gradient.len(), objective.layout.group_keys.len());
        assert!(
            evaluation.gradient.iter().all(|value| value.is_finite()),
            "every smoothing gradient component must be finite, got {:?}",
            evaluation.gradient
        );
        assert!(
            evaluation.fixed_point.recurred,
            "the fixture's inner fixed point must recur so the evaluation is at a \
             stationary inner state"
        );

        // The frozen surrogate is the criterion's identity. Its invariant is
        // stated against a FIXED operator: evaluating the plan twice on one
        // assembled system must be bit-identical, because the probes, the
        // quadrature nodes and the deflation basis are all fixed at the first
        // build. A plan rebuilt per call would make the outer search descend a
        // different function at every point, and the exact directional
        // derivative would be the gradient of something nobody evaluated twice.
        let lambda = objective.layout.expand(&rho).expect("expand");
        let system = objective
            .term
            .assemble_arrow_schur(
                objective.target.view(),
                &lambda,
                &objective.ard_precisions,
            )
            .expect("assemble arrow schur");
        let first = objective
            .evidence_log_det(&system)
            .expect("frozen surrogate")
            .0;
        let second = objective
            .evidence_log_det(&system)
            .expect("frozen surrogate")
            .0;
        assert_eq!(
            first, second,
            "the frozen log|S| surrogate must be bit-reproducible on one operator"
        );

        // A second full `evaluate` at the same ρ re-enters `solve_fixed_point`
        // from the state the first one left, so β̂ moves by a few ulps and the
        // criterion moves with it. That is the inner solve's reproducibility,
        // not the surrogate's, so this limb is a tight relative bound rather
        // than an equality — the equality above already pins the surrogate.
        let again = objective.evaluate(&rho).expect("second evaluation");
        assert!(
            (again.cost - evaluation.cost).abs() <= 1.0e-12 * evaluation.cost.abs().max(1.0),
            "criterion moved between two evaluations at one ρ: {} vs {}",
            evaluation.cost,
            again.cost
        );
        for group in 0..evaluation.gradient.len() {
            let gap = (again.gradient[group] - evaluation.gradient[group]).abs();
            assert!(
                gap <= 1.0e-10 * evaluation.gradient[group].abs().max(1.0),
                "group {group} gradient moved between two evaluations at one ρ: {} vs {}",
                evaluation.gradient[group],
                again.gradient[group]
            );
        }
    }

    /// #2576's decisive oracle for the channel that REPLACED the Hutchinson
    /// trace estimator.
    ///
    /// The lane no longer estimates `tr(H⁻¹ ∂H/∂ρ_g)` with its own probe family.
    /// It contracts the log-determinant surrogate's own derivative bundle
    /// against `∂S/∂ρ_g`, on the claim that
    ///
    /// Only `H_ββ` carries an explicit smoothing coordinate, so at fixed
    /// inner state `∂S/∂ρ_g = Σ_{k∈g} λ_k (S_k ⊗ I_P)` —
    /// `schur_derivative_matvec`.
    ///
    /// Both claims are tested at once by central-differencing the production
    /// `evidence_log_det` — the coordinate-profiled `log|S|` — through λ alone
    /// at a FROZEN inner state, against the analytic contraction. A wrong
    /// `∂S/∂ρ_g` (missing the `⊗ I_P`, wrong group mask, `λ` instead of
    /// `∂λ/∂ρ = λ`) fails; a plan rebuilt between the FD legs fails,
    /// because then the two legs would be values of two different functions.
    #[test]
    fn support_outer_logdet_gradient_matches_fd_of_its_own_surrogate() {
        let mut objective = build_objective();
        let base = array![0.4_f64.ln(), 2.2_f64.ln()];
        let groups = objective.layout.group_keys.len();

        // One clean inner solve, then FREEZE: the log-det channel's contract is
        // the partial derivative through λ at a fixed inner state, which is
        // exactly what the LAML gradient's trace term is.
        objective.reset();
        let lambda_base = objective.layout.expand(&base).expect("expand");
        objective
            .term
            .solve_fixed_point(
                objective.target.view(),
                &lambda_base,
                &objective.ard_precisions,
                objective.max_inner_iter,
                objective.inner_tolerance,
                objective.trust_radius,
            )
            .expect("base inner fixed point");

        let assemble = |objective: &SaeSupportOuterObjective, rho: &Array1<f64>| {
            let lambda = objective.layout.expand(rho).expect("expand");
            let system = objective
                .term
                .assemble_arrow_schur(
                    objective.target.view(),
                    &lambda,
                    &objective.ard_precisions,
                )
                .expect("assemble arrow schur");
            (system, lambda)
        };

        let (base_system, _) = assemble(&objective, &base);
        let (base_logdet, bundle) = objective
            .evidence_log_det(&base_system)
            .expect("base evidence log-determinant");
        assert!(base_logdet.is_finite());

        let (beta_offsets, beta_dim) = objective.beta_layout().expect("beta layout");
        let analytic = (0..groups)
            .map(|group| {
                bundle
                    .directional_derivative(&|vector: ArrayView1<f64>| {
                        objective.schur_derivative_matvec(
                            group,
                            &lambda_base,
                            &beta_offsets,
                            beta_dim,
                            vector,
                        )
                    })
                    .expect("surrogate directional derivative")
            })
            .collect::<Vec<_>>();

        let h = 1.0e-5;
        for group in 0..groups {
            let mut plus = base.clone();
            let mut minus = base.clone();
            plus[group] += h;
            minus[group] -= h;
            let (system_plus, _) = assemble(&objective, &plus);
            let (system_minus, _) = assemble(&objective, &minus);
            let value_plus = objective
                .evidence_log_det(&system_plus)
                .expect("perturbed evidence log-determinant")
                .0;
            let value_minus = objective
                .evidence_log_det(&system_minus)
                .expect("perturbed evidence log-determinant")
                .0;
            let fd = (value_plus - value_minus) / (2.0 * h);
            let gap = (analytic[group] - fd).abs();
            assert!(
                gap <= 1.0e-5 * (1.0 + fd.abs()),
                "group {group} ({}): analytic ∂log|S|/∂ρ = {:.9e} disagrees with the \
                 central difference of the SAME frozen surrogate {fd:.9e} (gap {gap:.3e})",
                objective.layout.group_keys[group],
                analytic[group],
            );
            // A positive-semidefinite ∂S with a strictly positive-rank penalty
            // block can only INCREASE log|S|; a sign flip in the contraction is
            // the failure this catches independently of the FD magnitude.
            assert!(
                analytic[group] > 0.0,
                "group {group}: an SPD ∂S/∂ρ must raise log|S|, got {:.9e}",
                analytic[group]
            );
        }
    }

    /// #2933 F27 — force the dense and the support-sparse route on ONE hard-TopK
    /// model at ONE fitted state and compare what each scores.
    ///
    /// The model is `N = 8`, `P = 2`, `K = 2` (so `K = P` and the dense lane admits
    /// it), top-1 support: a periodic circle atom on rows 0..4 and a linear plane on
    /// rows 4..8. The support route converges its inner fixed point. That state
    /// (routing, coordinates, decoders) is transferred into a dense `SaeManifoldTerm`
    /// and priced through the dense criterion's frozen lane (`inner_max_iter = 0`), so
    /// both values are evaluated at the same θ̂.
    ///
    /// Two positive controls show the transfer is exact and the support value is
    /// rebuilt from its production pieces: the dense `data_fit` equals ½·RSS of the
    /// support term, and the frozen support value equals `evaluate`'s cost. If the
    /// two routes were two factorizations of one scalar, the values and their
    /// explicit hypergradients at this state would agree. They must disagree by more
    /// than a floor, and the typed scores must refuse to compare.
    #[test]
    fn dense_and_support_topk_routes_score_different_criteria_at_one_state_2933_f27() {
        let n = 8usize;
        let p = 2usize;
        let circle_decoder = array![[0.1, -0.2], [0.8, 0.3], [-0.5, 0.6], [0.2, 0.1], [-0.1, 0.25]];
        let plane_decoder = array![[0.4, -0.3], [1.2, 0.2], [-0.3, 0.9]];
        let circle_coords = [0.03, 0.10, -0.06, 0.14];
        let plane_coords = [[0.3, -0.1], [-0.2, 0.4], [0.1, 0.2], [-0.4, -0.3]];
        let circle_eval = PeriodicHarmonicEvaluator::new(5).expect("periodic");
        let plane_eval = EuclideanPatchEvaluator::new(2, 1).expect("patch");

        // Target: the seed decode plus a deterministic perturbation, so the
        // residual, the penalty energy and the dispersion are all genuine.
        let mut target = Array2::<f64>::zeros((n, p));
        for row in 0..n {
            let decoded = if row < 4 {
                let coord = Array2::from_shape_vec((1, 1), vec![circle_coords[row]]).expect("c");
                circle_eval.evaluate(coord.view()).expect("eval").0.dot(&circle_decoder)
            } else {
                let coord =
                    Array2::from_shape_vec((1, 2), plane_coords[row - 4].to_vec()).expect("c");
                plane_eval.evaluate(coord.view()).expect("eval").0.dot(&plane_decoder)
            };
            for column in 0..p {
                target[[row, column]] = decoded[[0, column]]
                    + 0.3 * ((1.7 * row as f64 + 0.9 * column as f64).sin());
            }
        }

        let atoms = vec![
            atom(
                "circle",
                SaeAtomBasisKind::Periodic,
                1,
                Arc::new(circle_eval.clone()),
                &[circle_coords[0]],
                circle_decoder.clone(),
            ),
            atom(
                "plane",
                SaeAtomBasisKind::Linear,
                2,
                Arc::new(plane_eval.clone()),
                &plane_coords[0],
                plane_decoder.clone(),
            ),
        ];
        let specs = vec![
            SaeAssignmentAtomSpec {
                latent_dim: 1,
                manifold: SaeAtomBasisKind::Periodic.latent_manifold(1),
                retraction: gam_problem::LatentRetractionRegistry::all_euclidean(),
            },
            SaeAssignmentAtomSpec::euclidean(2),
        ];
        let indices = (0..n).map(|row| vec![u32::from(row >= 4)]).collect::<Vec<_>>();
        let gates = vec![vec![1.0]; n];
        let coords = (0..n)
            .map(|row| {
                if row < 4 {
                    vec![circle_coords[row]]
                } else {
                    plane_coords[row - 4].to_vec()
                }
            })
            .collect::<Vec<_>>();
        let state = SaeAssignmentState::from_topk_support_heterogeneous(
            n, 2, 1, specs, indices, gates, coords,
        )
        .expect("state");
        let term = SaeSupportSparseTerm::new(atoms, state).expect("term");
        let layout = SaeSupportSmoothingLayout::from_term(&term);
        assert_eq!(layout.group_keys.len(), 2, "one smoothing group per atom");
        let spectrum = penalty_spectrum(&term, &layout).expect("spectrum");
        let initial_term = term.clone();
        let mut objective = SaeSupportOuterObjective {
            term,
            initial_term,
            target: target.clone(),
            layout,
            spectrum,
            ard_precisions: vec![vec![1.0], vec![1.0, 1.0]],
            max_inner_iter: 5000,
            inner_tolerance: 1.0e-9,
            trust_radius: 1.0,
            random_state: 0xC0FF_EE00_D15E_A5E5,
            last_evaluation: None,
            state_revision: 0,
            uncached_evaluations: 0,
            logdet_surrogate: None,
        };
        let rho = array![0.4_f64.ln(), 2.2_f64.ln()];
        let evaluation = objective.evaluate(&rho).expect("support route evaluates");
        assert!(evaluation.fixed_point.recurred, "support inner state must recur");
        let components = evaluation.components;

        // Support value at the FROZEN state, rebuilt from production pieces.
        let residual_df = components.residual_df;
        let support_frozen = |objective: &mut SaeSupportOuterObjective, rho: &Array1<f64>| {
            let lambda = objective.layout.expand(rho).expect("expand");
            let system = objective
                .term
                .assemble_arrow_schur(objective.target.view(), &lambda, &objective.ard_precisions)
                .expect("assemble");
            let reduced_log_det = objective.evidence_log_det(&system).expect("evidence").0;
            let penalty_log_pdet = (0..rho.len())
                .map(|group| {
                    objective.spectrum.log_pdet_base_by_group[group]
                        + objective.spectrum.rank_by_group[group] as f64 * rho[group]
                })
                .sum::<f64>();
            let penalized_deviance = 2.0
                * objective
                    .term
                    .penalized_objective(
                        objective.target.view(),
                        &lambda,
                        &objective.ard_precisions,
                    )
                    .expect("penalized objective");
            SaeSupportLamlComponents {
                reduced_log_det,
                penalty_log_pdet,
                residual_df,
                penalized_deviance,
            }
            .value()
        };
        let support_rebuilt = support_frozen(&mut objective, &rho);

        // Transfer the converged state into the dense engine.
        let lambda = objective.layout.expand(&rho).expect("expand");
        let support_term = &objective.term;
        let mut dense_atoms = Vec::with_capacity(2);
        let mut coord_blocks = Vec::with_capacity(2);
        let mut manifolds = Vec::with_capacity(2);
        let mut logits = Array2::<f64>::from_elem((n, 2), -6.0);
        for row in 0..n {
            logits[[row, support_term.assignment.support_indices(row)[0] as usize]] = 6.0;
        }
        for atom_idx in 0..2 {
            let d = support_term.atoms[atom_idx].latent_dim();
            let mut block = Array2::<f64>::zeros((n, d));
            for row in 0..n {
                if support_term.assignment.support_indices(row)[0] as usize == atom_idx {
                    for (axis, &value) in
                        support_term.assignment.coords_for_slot(row, 0).iter().enumerate()
                    {
                        block[[row, axis]] = value;
                    }
                }
            }
            let evaluator: Arc<dyn SaeBasisSecondJet> = if atom_idx == 0 {
                Arc::new(circle_eval.clone())
            } else {
                Arc::new(plane_eval.clone())
            };
            let (phi, jet) = evaluator.evaluate(block.view()).expect("dense basis");
            let m = phi.ncols();
            let kind = support_term.atoms[atom_idx].basis_kind().clone();
            manifolds.push(kind.latent_manifold(d));
            dense_atoms.push(
                SaeManifoldAtom::new_with_provided_function_gram(
                    support_term.atoms[atom_idx].name.clone(),
                    kind,
                    d,
                    phi,
                    jet,
                    support_term.atoms[atom_idx].decoder_coefficients().to_owned(),
                    Array2::eye(m),
                )
                .expect("dense atom")
                .with_basis_second_jet(evaluator),
            );
            coord_blocks.push(block);
        }
        let mode = AssignmentMode::top_k_support(1);
        let assignment =
            SaeAssignment::from_blocks_with_mode_and_manifolds(logits, coord_blocks, manifolds, mode)
                .expect("dense assignment");
        let dense_term = SaeManifoldTerm::new(dense_atoms, assignment).expect("dense term");
        let dense_rho_at = |log_lambda: &[f64]| {
            SaeManifoldRho::with_per_atom_smooth(
                0.0,
                log_lambda.to_vec(),
                vec![Array1::zeros(1), Array1::zeros(2)],
            )
            .for_assignment(mode)
        };
        let log_lambda = lambda.iter().map(|value| value.ln()).collect::<Vec<_>>();
        let dense_frozen = |log_lambda: &[f64]| {
            let mut frozen = dense_term.clone();
            frozen
                .penalized_quasi_laplace_criterion_with_refine_policy(
                    target.view(),
                    &dense_rho_at(log_lambda),
                    None,
                    0,
                    1.0,
                    1.0e-6,
                    1.0e-6,
                    false,
                )
                .map_err(|error| format!("{error:?}"))
        };
        let dense_value = dense_frozen(&log_lambda);

        let rss = support_term
            .raw_residual(target.view())
            .expect("raw residual")
            .iter()
            .map(|value| value * value)
            .sum::<f64>();
        let smooth_energy = objective.penalty_energy_by_group(&lambda).iter().sum::<f64>();

        // Explicit fixed-state hypergradients of both values, per atom.
        let h = 1.0e-4;
        let mut support_gradient = [0.0_f64; 2];
        let mut dense_gradient = [f64::NAN; 2];
        for atom_idx in 0..2 {
            let group = objective.layout.atom_group[atom_idx];
            let mut plus = rho.clone();
            let mut minus = rho.clone();
            plus[group] += h;
            minus[group] -= h;
            support_gradient[atom_idx] = (support_frozen(&mut objective, &plus)
                - support_frozen(&mut objective, &minus))
                / (2.0 * h);
            let mut dense_plus = log_lambda.clone();
            let mut dense_minus = log_lambda.clone();
            dense_plus[atom_idx] += h;
            dense_minus[atom_idx] -= h;
            if let (Ok((value_plus, _)), Ok((value_minus, _))) =
                (dense_frozen(&dense_plus), dense_frozen(&dense_minus))
            {
                dense_gradient[atom_idx] = (value_plus - value_minus) / (2.0 * h);
            }
        }

        eprintln!(
            "[#2933 F27] support: cost={:.12e} rebuilt={support_rebuilt:.12e} \
             reduced_log_det={:.9e} penalty_log_pdet={:.9e} base_pdet={:?} residual_df={} \
             D_p={:.9e} profiled_dispersion_term={:.9e} rss={rss:.9e} smooth_energy={smooth_energy:.9e} \
             explicit_gradient={support_gradient:?} profiled_gradient={:?}",
            evaluation.cost,
            components.reduced_log_det,
            components.penalty_log_pdet,
            objective.spectrum.log_pdet_base_by_group,
            components.residual_df,
            components.penalized_deviance,
            components.profiled_dispersion_term(),
            evaluation.gradient,
        );
        match &dense_value {
            Ok((value, loss)) => eprintln!(
                "[#2933 F27] dense: V={value:.12e} loss.total={:.9e} data_fit={:.9e} \
                 smoothness={:.9e} ard={:.9e} complexity(V-loss)={:.9e} \
                 explicit_gradient={dense_gradient:?}",
                loss.total(),
                loss.data_fit,
                loss.smoothness,
                loss.ard,
                value - loss.total(),
            ),
            Err(error) => eprintln!("[#2933 F27] dense frozen lane refused: {error}"),
        }

        let (dense_value, dense_loss) = dense_value.expect("dense route prices the same state");
        // Positive controls: one state, and a support value rebuilt from its pieces.
        assert!(
            (dense_loss.data_fit - 0.5 * rss).abs() <= 1.0e-10 * (1.0 + rss),
            "state transfer is not exact: dense data_fit {:.12e} vs ½·RSS {:.12e}",
            dense_loss.data_fit,
            0.5 * rss
        );
        assert!(
            (support_rebuilt - evaluation.cost).abs() <= 1.0e-10 * (1.0 + evaluation.cost.abs()),
            "frozen support value {support_rebuilt:.12e} != evaluate cost {:.12e}",
            evaluation.cost
        );

        // Disagreement limbs: two criteria, not two factorizations of one.
        let value_gap = (dense_value - evaluation.cost).abs();
        assert!(
            value_gap > 1.0e-2 * (1.0 + dense_value.abs().max(evaluation.cost.abs())),
            "dense {dense_value:.12e} and support {:.12e} values agree at one state",
            evaluation.cost
        );
        let gradient_scale = support_gradient
            .iter()
            .chain(dense_gradient.iter())
            .fold(0.0_f64, |acc, value| acc.max(value.abs()));
        let gradient_gap = (0..2)
            .map(|atom_idx| (dense_gradient[atom_idx] - support_gradient[atom_idx]).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            dense_gradient.iter().all(|value| value.is_finite()),
            "dense explicit hypergradient must evaluate at the frozen state: {dense_gradient:?}"
        );
        assert!(
            gradient_gap > 1.0e-2 * (1.0 + gradient_scale),
            "explicit hypergradients agree at one state: dense {dense_gradient:?} support \
             {support_gradient:?}"
        );

        // Typed identity: the admission names two kinds, and their scores refuse.
        let dense_kind = crate::front_door::SaeFitLane::DenseCertification
            .criterion_kind()
            .expect("dense lane names its criterion");
        let support_kind = crate::front_door::SaeFitLane::CurvedStreaming
            .criterion_kind()
            .expect("support lane names its criterion");
        assert_ne!(dense_kind, support_kind);
        let dense_score = SaeCriterionScore::new(dense_kind, dense_value);
        let support_score = SaeCriterionScore::new(support_kind, evaluation.cost);
        assert!(dense_score.difference(&support_score).is_err());
    }
}
