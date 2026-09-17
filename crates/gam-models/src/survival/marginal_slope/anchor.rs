//! The declared-law anchor of the survival marginal-slope index (gam#2923).
//!
//! The family's defining identity is that `q(t, a)` is the MARGINAL survival
//! index: `E_z[Φ(−η) | a] = Φ(−q)`. The closed-form lowering
//! `η = q·√(1 + rᵀΣr) + rᵀz` is that identity solved under `z | a ~ N(0, Σ)`
//! and nothing else. On a declared finite law `F_a` — nodes `u_k`, weights
//! `w_k` on the scalar the slope projects the score onto — the identity is
//! instead the anchoring equation
//!
//! ```text
//!     Σ_k w_k Φ(−(α + b·u_k)) = Φ(−q),        b = s_f·g,
//! ```
//!
//! whose left side is continuous, strictly decreasing in `α` and runs from 1
//! to 0, so `α(q, b)` exists and is unique for every `(q, b)`
//! (`Descent.Portability.MarginalAnchor.exists_unique_anchor`, stated for the
//! logit link; the argument is link-agnostic). `α` replaces `q·c` in the row
//! index, and its derivatives come from implicit differentiation of the same
//! equation (`anchor_deriv_eq`): with `G(α, q, b) = Σ_k w_k Φ(−(α + b u_k)) −
//! Φ(−q)`,
//!
//! ```text
//!     α_θ = −G_θ / G_α = −Σ_k w_k φ(η_k) ∂_θ(b u_k) / Σ_k w_k φ(η_k),   η_k = α + b u_k,
//! ```
//!
//! and the second and third orders likewise.
//!
//! This module owns three things and nothing else: the root itself
//! ([`solve_anchor`], and [`solve_anchor_in_slot`] through a row's warm-start
//! slot), its closed-form derivatives through order three
//! ([`AnchorDerivatives`]) for the direct order-two lowering, and its Taylor
//! table through order five ([`AnchorTaylor`]) with that table's lift onto any
//! [`JetField`] for the higher-order carriers (gam#2928). The Gaussian
//! closed form is the special case of every one of them: on a Gaussian
//! quadrature grid the root is `q·√(1 + b²)` to quadrature tolerance, which is
//! what `anchor_tests::gaussian_grid_reproduces_the_closed_form` pins.

use super::SurvivalInterceptSlotKind;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use gam_linalg::utils::splitmix64_hash;
use gam_math::nested_dual::JetField;
use gam_math::probability::{
    normal_cdf, normal_logcdf, normal_pdf, signed_probit_logcdf_and_mills_ratio,
};
use smallvec::SmallVec;

/// `|log-residual|` at which the anchoring equation counts as solved. The
/// residual is `log Σ_k w_k Φ(∓η_k) − log Φ(∓q)` on whichever tail is the
/// smaller, i.e. the RELATIVE error of the anchored marginal probability on
/// the side where relative error is the meaningful one. Where one float step
/// of `α` moves the residual by more than this, a root is held to what that
/// step resolves instead ([`anchor_residual_resolution`]).
pub(crate) const ANCHOR_LOG_RESIDUAL_TOL: f64 = 1e-13;

/// A finite law on the projected score, borrowed from wherever the family
/// materialised it: nodes ascending, positive weights summing to one, and the
/// log-weights the solver's log-space residual reads.
#[derive(Clone, Copy)]
pub(crate) struct AnchorGrid<'a> {
    pub(crate) nodes: &'a [f64],
    pub(crate) weights: &'a [f64],
    pub(crate) log_weights: &'a [f64],
}

impl<'a> AnchorGrid<'a> {
    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.nodes.len()
    }
}

/// An owned finite law: what the family materialises once per fit (one for a
/// global law, one per row for a local one) and hands out as [`AnchorGrid`]s.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct AnchorGridOwned {
    pub(crate) nodes: Vec<f64>,
    pub(crate) weights: Vec<f64>,
    pub(crate) log_weights: Vec<f64>,
}

impl AnchorGridOwned {
    pub(crate) fn from_grid(grid: &crate::bms::EmpiricalZGrid) -> Self {
        Self::new(grid.nodes.clone(), grid.weights.clone())
    }

    pub(crate) fn new(nodes: Vec<f64>, weights: Vec<f64>) -> Self {
        let log_weights = weights.iter().map(|w| w.ln()).collect();
        Self {
            nodes,
            weights,
            log_weights,
        }
    }

    #[inline]
    pub(crate) fn view(&self) -> AnchorGrid<'_> {
        AnchorGrid {
            nodes: &self.nodes,
            weights: &self.weights,
            log_weights: &self.log_weights,
        }
    }
}

/// The declared latent law a fit anchors on, materialised once per family.
///
/// `kind` is the persisted object — the same [`LatentMeasureKind`] the
/// Bernoulli family writes and the shared predictor replays — and the grids
/// are its per-row projection onto the scalar the shared slope reads: one
/// grid for a global law, one per training row for a local mixture.
#[derive(Clone, Debug)]
pub(crate) struct SurvivalLatentLaw {
    pub(crate) kind: crate::bms::LatentMeasureKind,
    grids: LawGrids,
    /// The roots solved on this law (gam#2928), shared by every family the
    /// search builds on it.
    roots: Arc<AnchorRootCache>,
}

#[derive(Clone, Debug)]
enum LawGrids {
    Global(AnchorGridOwned),
    PerRow(Vec<AnchorGridOwned>),
    /// The joint law of a `K ≥ 2` score vector (gam#2929). It has no scalar
    /// grid: each row projects it onto its own slope vector, and only the
    /// per-score vector program reads it.
    Joint(Arc<super::JointLatentLawRuntime>),
}

impl SurvivalLatentLaw {
    /// Materialise the law's per-row grids for `n` training rows. `None` for
    /// the standard-normal law, on which the closed form is the anchor.
    pub(crate) fn from_kind(
        kind: &crate::bms::LatentMeasureKind,
        n: usize,
    ) -> Result<Option<Self>, String> {
        use crate::bms::LatentMeasureKind;
        kind.validate("survival marginal-slope declared latent law")?;
        let grids = match kind {
            LatentMeasureKind::StandardNormal => return Ok(None),
            LatentMeasureKind::GlobalEmpirical { grid } => {
                LawGrids::Global(AnchorGridOwned::from_grid(grid))
            }
            LatentMeasureKind::LocalEmpirical { .. } => {
                let mut per_row = Vec::with_capacity(n);
                for row in 0..n {
                    let grid = kind.empirical_grid_for_training_row(row)?.ok_or_else(|| {
                        format!(
                            "survival marginal-slope local latent law produced no grid for row {row}"
                        )
                    })?;
                    per_row.push(AnchorGridOwned::from_grid(&grid));
                }
                LawGrids::PerRow(per_row)
            }
        };
        let shared_across_rows = matches!(grids, LawGrids::Global(_));
        Ok(Some(Self {
            kind: kind.clone(),
            grids,
            roots: Arc::new(AnchorRootCache::new(n, shared_across_rows)),
        }))
    }

    /// A joint law of the score vector (gam#2929). `kind` is the primary
    /// score's measure, which is what named the law's configuration. It carries
    /// no root slots: the per-score vector program solves its own roots.
    pub(crate) fn from_joint(
        kind: crate::bms::LatentMeasureKind,
        runtime: super::JointLatentLawRuntime,
    ) -> Self {
        Self {
            kind,
            grids: LawGrids::Joint(Arc::new(runtime)),
            roots: Arc::new(AnchorRootCache::new(0, false)),
        }
    }

    /// The joint law, when this is one.
    #[inline]
    pub(crate) fn joint(&self) -> Option<&super::JointLatentLawRuntime> {
        match &self.grids {
            LawGrids::Joint(runtime) => Some(runtime),
            LawGrids::Global(_) | LawGrids::PerRow(_) => None,
        }
    }

    /// Row `row`'s law with the roots already solved on it.
    #[inline]
    pub(crate) fn row_context(&self, row: usize) -> AnchorRowContext<'_> {
        AnchorRowContext {
            grid: self.row(row),
            roots: Some(&self.roots),
        }
    }

    /// The law of row `row`.
    ///
    /// A joint law has no scalar grid. The per-score vector lanes that read it
    /// never ask for one, and every consumer of the scalar frames refuses a
    /// per-score family before solving on a row's grid, so the empty grid only
    /// keeps this match total without inventing a law the fit did not declare.
    #[inline]
    pub(crate) fn row(&self, row: usize) -> AnchorGrid<'_> {
        match &self.grids {
            LawGrids::Global(grid) => grid.view(),
            LawGrids::PerRow(grids) => grids[row].view(),
            LawGrids::Joint(_) => AnchorGrid {
                nodes: &[],
                weights: &[],
                log_weights: &[],
            },
        }
    }
}

/// What a row hands the anchored frame: its law, and the law's root slots
/// (gam#2928).
#[derive(Clone, Copy)]
pub(crate) struct AnchorRowContext<'a> {
    pub(crate) grid: AnchorGrid<'a>,
    /// The roots already solved on this law, when the caller keeps them.
    pub(crate) roots: Option<&'a AnchorRootCache>,
}

/// The Gaussian closed form, which is the anchor's seed and — on a Gaussian
/// law — its value.
#[inline]
pub(crate) fn gaussian_anchor(q: f64, observed_slope: f64) -> f64 {
    q * (1.0 + observed_slope * observed_slope).sqrt()
}

/// Entries in a global law's cross-row root table: a power of two.
const SHARED_ROOT_SLOTS: usize = 256;

/// One stored anchor: the exact inputs of the equation it solved and its
/// root, published under a sequence lock so a reader sees all three from one
/// completed write or nothing.
#[derive(Debug)]
struct RootSlot {
    sequence: AtomicU64,
    q: AtomicU64,
    slope: AtomicU64,
    root: AtomicU64,
}

impl RootSlot {
    fn empty() -> Self {
        let nan = f64::NAN.to_bits();
        Self {
            sequence: AtomicU64::new(0),
            q: AtomicU64::new(nan),
            slope: AtomicU64::new(nan),
            root: AtomicU64::new(nan),
        }
    }

    /// `(q bits, b bits, root)` from one completed write, or `None` while a
    /// writer holds the slot or finished a write during the read.
    fn read(&self) -> Option<(u64, u64, f64)> {
        let before = self.sequence.load(Ordering::Acquire);
        if before & 1 == 1 {
            return None;
        }
        let q = self.q.load(Ordering::Acquire);
        let slope = self.slope.load(Ordering::Acquire);
        let root = f64::from_bits(self.root.load(Ordering::Acquire));
        (self.sequence.load(Ordering::Acquire) == before).then_some((q, slope, root))
    }

    /// Publish a solved anchor. A writer that finds the slot held skips it:
    /// the slot is a cache, and the other writer's root serves as well.
    fn write(&self, q: u64, slope: u64, root: f64) {
        let before = self.sequence.load(Ordering::Acquire);
        if before & 1 == 1
            || self
                .sequence
                .compare_exchange(before, before + 1, Ordering::AcqRel, Ordering::Acquire)
                .is_err()
        {
            return;
        }
        self.q.store(q, Ordering::Release);
        self.slope.store(slope, Ordering::Release);
        self.root.store(root.to_bits(), Ordering::Release);
        self.sequence.store(before + 2, Ordering::Release);
    }
}

/// The roots already solved on one law (gam#2928): two slots per training
/// row and, for a global law, a cross-row table.
///
/// A stored root answers only for bitwise the inputs it was solved for. The
/// residual reads nothing else: `q`, the OBSERVED slope `b` (frailty scale
/// already applied), and the row's grid, which is fixed for the life of the
/// law that owns this cache — a re-estimated law is a new law with a new
/// cache. A hash selects a cross-row entry; equality of the stored inputs
/// decides a hit, and any other stored root is only a seed.
#[derive(Debug)]
pub(crate) struct AnchorRootCache {
    entry: Vec<RootSlot>,
    exit: Vec<RootSlot>,
    /// `Some` for a global law, on which every row solves the same equation
    /// for the same inputs; a per-row law gives each row its own equation.
    shared: Option<Vec<RootSlot>>,
    shared_index: fn(u64, u64) -> u64,
}

impl AnchorRootCache {
    pub(crate) fn new(n: usize, shared_across_rows: bool) -> Self {
        Self::with_shared_index(n, shared_across_rows, |q, slope| {
            splitmix64_hash(q ^ splitmix64_hash(slope))
        })
    }

    /// [`Self::new`] with the cross-row table indexed by `shared_index`.
    fn with_shared_index(
        n: usize,
        shared_across_rows: bool,
        shared_index: fn(u64, u64) -> u64,
    ) -> Self {
        let slots = |count: usize| (0..count).map(|_| RootSlot::empty()).collect::<Vec<_>>();
        Self {
            entry: slots(n),
            exit: slots(n),
            shared: shared_across_rows.then(|| slots(SHARED_ROOT_SLOTS)),
            shared_index,
        }
    }

    fn slot(&self, row: usize, kind: SurvivalInterceptSlotKind) -> Option<&RootSlot> {
        match kind {
            SurvivalInterceptSlotKind::Entry => self.entry.get(row),
            SurvivalInterceptSlotKind::Exit => self.exit.get(row),
        }
    }

    fn shared_slot(&self, q: u64, slope: u64) -> Option<&RootSlot> {
        let table = self.shared.as_ref()?;
        table.get((self.shared_index)(q, slope) as usize & (SHARED_ROOT_SLOTS - 1))
    }
}

/// [`solve_anchor`] through the law's root slots (gam#2928).
///
/// Every consumer of a row at one coefficient iterate — the value, the direct
/// order-two lowering, and each directional jet the outer derivatives seed —
/// asks for the root of bitwise the same equation, so the row's slot answers
/// without evaluating the residual. On a global law, rows asking the same
/// equation — every row entering at the time origin shares its entry index —
/// find one row's root in the cross-row table. At a new iterate the slot's
/// root is the previous iterate's, one Newton probe away after a Newton step.
/// A root stored by a rejected trial far from this iterate can mislead the
/// seed, so a refused warm solve retries from the closed form, which depends
/// only on `(q, b)` and cannot be poisoned.
pub(crate) fn solve_anchor_in_slot(
    q: f64,
    observed_slope: f64,
    context: AnchorRowContext<'_>,
    row: usize,
    slot: SurvivalInterceptSlotKind,
) -> Result<f64, String> {
    let Some(cache) = context.roots else {
        return solve_anchor(q, observed_slope, context.grid);
    };
    let inputs = (q.to_bits(), observed_slope.to_bits());
    let own = cache.slot(row, slot);
    let mut seed = None;
    if let Some((stored_q, stored_slope, root)) = own.and_then(RootSlot::read)
        && root.is_finite()
    {
        if (stored_q, stored_slope) == inputs {
            return Ok(root);
        }
        seed = Some(root);
    }
    let shared = cache.shared_slot(inputs.0, inputs.1);
    if let Some((stored_q, stored_slope, root)) = shared.and_then(RootSlot::read)
        && root.is_finite()
        && (stored_q, stored_slope) == inputs
    {
        if let Some(own) = own {
            own.write(inputs.0, inputs.1, root);
        }
        return Ok(root);
    }
    let root = match seed {
        Some(seed) => match solve_anchor_from(q, observed_slope, context.grid, seed) {
            Ok(root) => root,
            Err(warm_failure) => {
                log::debug!(
                    "[survival-marginal-slope anchor] row {row}: the slot's seed {seed} did not \
                     solve ({warm_failure}); retrying from the closed form"
                );
                solve_anchor(q, observed_slope, context.grid)?
            }
        },
        None => solve_anchor(q, observed_slope, context.grid)?,
    };
    if let Some(own) = own {
        own.write(inputs.0, inputs.1, root);
    }
    if let Some(shared) = shared {
        shared.write(inputs.0, inputs.1, root);
    }
    Ok(root)
}

/// Solve the anchoring equation for `α` at the marginal index `q` and the
/// OBSERVED slope `b` (frailty scale already applied).
///
/// Bracketed Newton–Halley on a log-space residual: with `S(α) = Σ_k w_k
/// Φ(−η_k)` the residual is `log S − log Φ(−q)` when `Φ(−q) ≤ ½` and
/// `log(1 − S) − log Φ(q)` otherwise, so the tolerance is always a relative
/// error on the smaller of the two marginal tails. Both sides are strictly
/// monotone in `α`, the log-sum-exp keeps the sum finite where every node has
/// underflowed, and the solve searches from the Gaussian closed-form seed.
///
/// The solve is a safeguarded Halley iteration on a sign bracket (gam#2928):
/// Halley's step where it agrees with Newton's, a doubling search until both
/// sides of the root are seen, bisection whenever a step leaves the bracket or
/// fails to halve. It returns only an `α` whose residual meets
/// [`anchor_residual_resolution`] — the tolerance, or the residual one float
/// step of `α` resolves where that is coarser — and refuses otherwise. It does
/// not stop on bracket width, so no returned root carries a residual above
/// that criterion.
pub(crate) fn solve_anchor(q: f64, observed_slope: f64, grid: AnchorGrid<'_>) -> Result<f64, String> {
    solve_anchor_from(q, observed_slope, grid, gaussian_anchor(q, observed_slope))
}

/// [`solve_anchor`] from an explicit seed.
pub(crate) fn solve_anchor_from(
    q: f64,
    observed_slope: f64,
    grid: AnchorGrid<'_>,
    seed: f64,
) -> Result<f64, String> {
    solve_anchor_with(q, observed_slope, grid, seed, anchor_log_residual)
}

/// The residual a solve reads: `(F, F′, F″)` at `(α, b, grid, survival side,
/// log target)`.
type AnchorResidual = fn(f64, f64, AnchorGrid<'_>, bool, f64) -> Result<(f64, f64, f64), String>;

/// [`solve_anchor_from`] on an explicit residual form, so the log-space form
/// can be solved on exactly the production path as an oracle.
fn solve_anchor_with(
    q: f64,
    observed_slope: f64,
    grid: AnchorGrid<'_>,
    seed: f64,
    residual: AnchorResidual,
) -> Result<f64, String> {
    if !(q.is_finite() && observed_slope.is_finite() && seed.is_finite()) {
        return Err(format!(
            "survival marginal-slope anchor requires finite q, slope and seed, got q={q}, b={observed_slope}, seed={seed}"
        ));
    }
    let survival_side = q >= 0.0;
    let log_target = if survival_side {
        normal_logcdf(-q)
    } else {
        normal_logcdf(q)
    };
    // `F` is strictly decreasing in `α` on the survival side and strictly
    // increasing on the complement side.
    let increasing = !survival_side;
    // The sign bracket: `below < root < above` once each side has been seen.
    let mut below = f64::NEG_INFINITY;
    let mut above = f64::INFINITY;
    let mut alpha = seed;
    let mut step_cap = (0.25 * (1.0 + seed.abs())).max(1.0);
    let mut last_step = f64::INFINITY;
    for _ in 0..ANCHOR_SOLVE_MAX_EVALUATIONS {
        let (value, first, second) = residual(alpha, observed_slope, grid, survival_side, log_target)?;
        if value.abs() <= anchor_residual_resolution(alpha, first) {
            return Ok(alpha);
        }
        let root_is_above = if increasing { value < 0.0 } else { value > 0.0 };
        if root_is_above {
            below = alpha;
        } else {
            above = alpha;
        }
        let toward = if root_is_above { 1.0 } else { -1.0 };
        // Halley's step where it agrees in direction with Newton's, else
        // Newton's; a step that does not point at the root is replaced by the
        // capped search step.
        let newton = -value / first;
        let halley = -2.0 * value * first / (2.0 * first * first - value * second);
        let mut step = if halley.is_finite() && halley * newton > 0.0 {
            halley
        } else {
            newton
        };
        if !(step.is_finite() && step * toward > 0.0) {
            step = toward * step_cap;
        }
        let next = if below.is_finite() && above.is_finite() {
            let midpoint = below + 0.5 * (above - below);
            let candidate = alpha + step;
            // Bisect whenever the step leaves the bracket or is not at most
            // half the step before it, so the bracket reaches adjacent floats
            // in bounded work however the curvature behaves.
            let next = if candidate > below && candidate < above && step.abs() <= 0.5 * last_step {
                candidate
            } else {
                midpoint
            };
            last_step = (next - alpha).abs();
            if !(next > below && next < above) {
                return Err(format!(
                    "survival marginal-slope anchor bracketed its root between adjacent floats \
                     [{below:e}, {above:e}] without resolving the residual (q={q}, b={observed_slope})"
                ));
            }
            next
        } else if step.abs() > step_cap {
            // One side unseen: search toward it with a doubling cap.
            let capped = alpha + toward * step_cap;
            step_cap *= 2.0;
            capped
        } else {
            alpha + step
        };
        alpha = next;
    }
    Err(format!(
        "survival marginal-slope anchor did not resolve its residual in {ANCHOR_SOLVE_MAX_EVALUATIONS} \
         evaluations (q={q}, b={observed_slope}, seed={seed}, bracket [{below:e}, {above:e}])"
    ))
}

/// Residual evaluations a solve may spend before it refuses: enough for a
/// doubling search across the whole `f64` range and a bisection down to
/// adjacent floats.
const ANCHOR_SOLVE_MAX_EVALUATIONS: usize = 256;

/// The `|F|` a root is held to (gam#2928): [`ANCHOR_LOG_RESIDUAL_TOL`], or —
/// where one floating-point step of `α` already moves the residual by more
/// than that — the residual that step resolves, `|F′(α)|·ulp(α)`. The float
/// nearest the root always meets it, so a solve stops on this criterion or
/// refuses; it never stops on a bracket width.
#[inline]
pub(crate) fn anchor_residual_resolution(alpha: f64, first: f64) -> f64 {
    let magnitude = alpha.abs();
    ANCHOR_LOG_RESIDUAL_TOL.max(first.abs() * (magnitude.next_up() - magnitude))
}

/// The smallest marginal tail the linear-space residual sums directly. Every
/// node term is a positive probability, so the sum keeps each term's relative
/// accuracy; below this floor a term can have underflowed against a sum it is
/// no longer negligible in, and the log-space form takes over.
const LINEAR_RESIDUAL_FLOOR: f64 = 1e-250;

/// `(F, F′, F″)` of the log-space anchoring residual at `α` (gam#2928).
///
/// The residual is always `log T(α) − log Φ(∓q)` with `T` the marginal on the
/// SMALLER tail — `Σ_k w_k Φ(−η_k)` against `Φ(−q) ≤ ½` when `q ≥ 0`,
/// `Σ_k w_k Φ(η_k)` against `Φ(q) ≤ ½` otherwise — so acceptance at
/// `|F| ≤ 1e-13` is a relative error on that tail on both forms, and nothing
/// subtracts nearly equal survival probabilities where `S` sits near one.
/// `T` is summed in linear space whenever it sits above
/// [`LINEAR_RESIDUAL_FLOOR`]: every node term is a positive probability from
/// its own `erfc`, so the sum keeps each term's relative accuracy, at one
/// `erfc` and one density per node where the log-space form pays a scaled
/// complementary error function, its logarithm and a log-sum-exp exponential.
/// The two forms agree to rounding, and deep tails keep the log-space form.
fn anchor_log_residual(
    alpha: f64,
    observed_slope: f64,
    grid: AnchorGrid<'_>,
    survival_side: bool,
    log_target: f64,
) -> Result<(f64, f64, f64), String> {
    // T = Σ_k w_k Φ(∓η_k), and the density sums Σ w φ(η), Σ w η φ(η).
    let mut tail = 0.0;
    let mut density = 0.0;
    let mut density_eta = 0.0;
    for (&u, &w) in grid.nodes.iter().zip(grid.weights.iter()) {
        let eta = alpha + observed_slope * u;
        tail += w * normal_cdf(if survival_side { -eta } else { eta });
        let weighted_pdf = w * normal_pdf(eta);
        density += weighted_pdf;
        density_eta += weighted_pdf * eta;
    }
    if !(tail.is_finite() && tail >= LINEAR_RESIDUAL_FLOOR) {
        return anchor_log_residual_in_log_space(
            alpha,
            observed_slope,
            grid,
            survival_side,
            log_target,
        );
    }
    let value = tail.ln() - log_target;
    let ratio_sum = density / tail;
    let ratio_eta_sum = density_eta / tail;
    // Survival side: T′/T = −Σ w φ / T, (log T)″ = Σ w η φ / T − (T′/T)².
    // Complement side: T′/T = Σ w φ / T, (log T)″ = −Σ w η φ / T − (T′/T)².
    let (first, second) = if survival_side {
        (-ratio_sum, ratio_eta_sum - ratio_sum * ratio_sum)
    } else {
        (ratio_sum, -ratio_eta_sum - ratio_sum * ratio_sum)
    };
    if !(value.is_finite() && first.is_finite() && first != 0.0 && second.is_finite()) {
        return anchor_log_residual_in_log_space(
            alpha,
            observed_slope,
            grid,
            survival_side,
            log_target,
        );
    }
    Ok((value, first, second))
}

/// [`anchor_log_residual`] with the marginal tail formed by log-sum-exp, for
/// tails below [`LINEAR_RESIDUAL_FLOOR`].
fn anchor_log_residual_in_log_space(
    alpha: f64,
    observed_slope: f64,
    grid: AnchorGrid<'_>,
    survival_side: bool,
    log_target: f64,
) -> Result<(f64, f64, f64), String> {
    let m = grid.len();
    // Per node, from one shared `erfcx`/`erfc` evaluation: the log term
    // `log w_k + log Φ(∓η_k)` and the Mills ratio `φ(η_k) / Φ(∓η_k)`
    // (gam#2928).
    // Inline storage, filled rather than zeroed: a law of up to 128 nodes
    // touches no allocator and no per-evaluation memset.
    let mut terms: SmallVec<[f64; 128]> = SmallVec::with_capacity(m);
    let mut mills: SmallVec<[f64; 128]> = SmallVec::with_capacity(m);
    let mut log_max = f64::NEG_INFINITY;
    for k in 0..m {
        let eta = alpha + observed_slope * grid.nodes[k];
        let (log_cdf, mills_ratio) =
            signed_probit_logcdf_and_mills_ratio(if survival_side { -eta } else { eta });
        let term = grid.log_weights[k] + log_cdf;
        terms.push(term);
        mills.push(mills_ratio);
        if term > log_max {
            log_max = term;
        }
    }
    if !log_max.is_finite() {
        return Err(format!(
            "survival marginal-slope anchor residual is non-finite at α={alpha}, b={observed_slope}"
        ));
    }
    // log Σ_k w_k Φ(∓η_k) by log-sum-exp; each term is overwritten by its
    // scaled weight `e_k = exp(term_k − max)`.
    let mut sum = 0.0;
    for term in terms.iter_mut() {
        *term = (*term - log_max).exp();
        sum += *term;
    }
    let log_sum = log_max + sum.ln();
    let value = log_sum - log_target;
    // Density ratios `m_k = w_k φ(η_k) / S = (e_k / Σ_j e_j)·φ(η_k)/Φ(∓η_k)`:
    // the log-sum-exp weight times the Mills ratio, with no second
    // exponential per node.
    let mut ratio_sum = 0.0;
    let mut ratio_eta_sum = 0.0;
    for k in 0..m {
        let eta = alpha + observed_slope * grid.nodes[k];
        let ratio = terms[k] / sum * mills[k];
        ratio_sum += ratio;
        ratio_eta_sum += ratio * eta;
    }
    // Survival side: S′ = −Σ w φ, S″ = Σ w η φ.
    // Complement side: C′ = Σ w φ, C″ = −Σ w η φ.
    let (first, second) = if survival_side {
        (-ratio_sum, ratio_eta_sum - ratio_sum * ratio_sum)
    } else {
        (ratio_sum, -ratio_eta_sum - ratio_sum * ratio_sum)
    };
    // `F′` is mathematically non-zero on both sides; where every density ratio
    // underflowed the monotone solver only needs its sign to bracket.
    let first = if first == 0.0 {
        if survival_side {
            -f64::MIN_POSITIVE
        } else {
            f64::MIN_POSITIVE
        }
    } else {
        first
    };
    if !(value.is_finite() && first.is_finite() && second.is_finite()) {
        return Err(format!(
            "survival marginal-slope anchor residual state is non-finite: F={value}, F′={first}, F″={second} at α={alpha}"
        ));
    }
    Ok((value, first, second))
}

/// The anchor and its derivatives in `(q, b)` through order three, by implicit
/// differentiation of `G(α, q, b) = Σ_k w_k Φ(−(α + b u_k)) − Φ(−q) = 0`.
///
/// Third order is the highest the direct order-two lowering of the anchored
/// frame reads: the exit-time rate feature is `α̇₁ = α_q(q₁, b)·q̇₁`, whose
/// curvature in the primaries is a third derivative of `α`. Everything above
/// that comes through [`AnchorTaylor`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct AnchorDerivatives {
    pub(crate) alpha: f64,
    pub(crate) a_q: f64,
    pub(crate) a_b: f64,
    pub(crate) a_qq: f64,
    pub(crate) a_qb: f64,
    pub(crate) a_bb: f64,
    pub(crate) a_qqq: f64,
    pub(crate) a_qqb: f64,
    pub(crate) a_qbb: f64,
    pub(crate) a_bbb: f64,
}

/// Which of `(q, b)` an implicit-derivative index names.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Var {
    Q,
    B,
}

impl AnchorDerivatives {
    /// Differentiate at an already solved anchor.
    pub(crate) fn at(
        alpha: f64,
        q: f64,
        observed_slope: f64,
        grid: AnchorGrid<'_>,
    ) -> Result<Self, String> {
        // Moments of the law against the derivatives of `F(x) = Φ(−x)`:
        //   F′ = −φ, F″ = xφ, F‴ = (1 − x²)φ,
        // each weighted by `u_k^j` for the `b`-derivatives.
        let mut p = [0.0_f64; 4];
        let mut s2 = [0.0_f64; 4];
        let mut s3 = [0.0_f64; 4];
        for (&u, &w) in grid.nodes.iter().zip(grid.weights.iter()) {
            let eta = alpha + observed_slope * u;
            let pdf = w * normal_pdf(eta);
            let mut power = 1.0;
            for j in 0..4 {
                p[j] += pdf * power;
                s2[j] += pdf * eta * power;
                s3[j] += pdf * (1.0 - eta * eta) * power;
                power *= u;
            }
        }
        let phi_q = normal_pdf(q);
        // Partial derivatives of G by order. In `q` alone: `−F^{(n)}(q) =
        // −(−1)^n He_{n−1}(q) φ(q)`; in `(α, b)` the law's moments above, by
        // total order and by the number of `b`'s. Order zero is never read.
        let g_q = [0.0, phi_q, -q * phi_q, -(1.0 - q * q) * phi_q];
        let g_ab = [[0.0; 4], [-p[0], -p[1], -p[2], -p[3]], s2, s3];
        // `na` in α, `nq` in q, `nb` in b; mixed q/(α, b) derivatives vanish.
        let g = |na: usize, nq: usize, nb: usize| -> f64 {
            if nq > 0 {
                if na + nb > 0 { 0.0 } else { g_q[nq] }
            } else {
                g_ab[na + nb][nb]
            }
        };
        let g_alpha = g(1, 0, 0);
        if !(g_alpha.is_finite() && g_alpha < 0.0) {
            return Err(format!(
                "survival marginal-slope anchor has no finite gradient: G_α={g_alpha:e} at α={alpha}, q={q}, b={observed_slope}"
            ));
        }
        let idx = |v: Var| -> (usize, usize) {
            match v {
                Var::Q => (1, 0),
                Var::B => (0, 1),
            }
        };
        // G with `na` α-derivatives and the listed (q, b) variables.
        let gp = |na: usize, vars: &[Var]| -> f64 {
            let mut nq = 0;
            let mut nb = 0;
            for v in vars {
                let (dq, db) = idx(*v);
                nq += dq;
                nb += db;
            }
            g(na, nq, nb)
        };
        let first = |x: Var| -> f64 { -gp(0, &[x]) / g_alpha };
        let a1 = |x: Var| first(x);
        let second = |x: Var, y: Var| -> f64 {
            -(gp(0, &[x, y]) + gp(1, &[x]) * a1(y) + gp(1, &[y]) * a1(x) + gp(2, &[]) * a1(x) * a1(y))
                / g_alpha
        };
        let third = |x: Var, y: Var, z: Var| -> f64 {
            let ax = a1(x);
            let ay = a1(y);
            let az = a1(z);
            let axy = second(x, y);
            let axz = second(x, z);
            let ayz = second(y, z);
            -(gp(0, &[x, y, z])
                + gp(1, &[x, y]) * az
                + (gp(1, &[x, z]) + gp(2, &[x]) * az) * ay
                + gp(1, &[x]) * ayz
                + (gp(1, &[y, z]) + gp(2, &[y]) * az) * ax
                + gp(1, &[y]) * axz
                + (gp(2, &[z]) + gp(3, &[]) * az) * ax * ay
                + gp(2, &[]) * (axz * ay + ax * ayz)
                + (gp(1, &[z]) + gp(2, &[]) * az) * axy)
                / g_alpha
        };
        let out = Self {
            alpha,
            a_q: first(Var::Q),
            a_b: first(Var::B),
            a_qq: second(Var::Q, Var::Q),
            a_qb: second(Var::Q, Var::B),
            a_bb: second(Var::B, Var::B),
            a_qqq: third(Var::Q, Var::Q, Var::Q),
            a_qqb: third(Var::Q, Var::Q, Var::B),
            a_qbb: third(Var::Q, Var::B, Var::B),
            a_bbb: third(Var::B, Var::B, Var::B),
        };
        Ok(out)
    }
}

/// `∂α/∂q = φ(q) / Σ_k w_k φ(η_k)` at a solved anchor: the factor the exit rate
/// `α̇₁ = α_q·q̇₁` carries on the value path.
pub(crate) fn anchor_q_derivative(
    alpha: f64,
    q: f64,
    observed_slope: f64,
    grid: AnchorGrid<'_>,
) -> Result<f64, String> {
    let density: f64 = grid
        .nodes
        .iter()
        .zip(grid.weights.iter())
        .map(|(&u, &w)| w * normal_pdf(alpha + observed_slope * u))
        .sum();
    if !(density.is_finite() && density > 0.0) {
        return Err(format!(
            "survival marginal-slope anchor density underflowed: Σ w φ(η)={density:e} at α={alpha}, q={q}, b={observed_slope}"
        ));
    }
    Ok(normal_pdf(q) / density)
}

/// The order of the anchor's Taylor table. The order-≤4 carriers read `α`
/// through order four and the exit rate `α̇₁ = α_q·q̇₁` through order four,
/// which is `α` through order five.
const ANCHOR_TAYLOR_ORDER: usize = 5;

/// Coefficient slots per axis of the table: degrees `0..=ANCHOR_TAYLOR_ORDER`.
const TAYLOR_SLOTS: usize = ANCHOR_TAYLOR_ORDER + 1;

/// `n!` for `n ≤ 5`: the factor between a Taylor coefficient and the
/// derivative stack a carrier composes.
const FACTORIAL: [f64; TAYLOR_SLOTS] = [1.0, 1.0, 2.0, 6.0, 24.0, 120.0];

/// `scale·[·, F′, F″, F‴, F⁗, F⁽⁵⁾](x)` with `F(x) = Φ(−x)`, `F^{(n)} =
/// (−1)^n He_{n−1}(x) φ(x)`. The value slot is never read and holds zero.
#[inline]
fn survival_cdf_derivative_stack(x: f64, scale: f64) -> [f64; TAYLOR_SLOTS] {
    let pdf = scale * normal_pdf(x);
    let x2 = x * x;
    [
        0.0,
        -pdf,
        x * pdf,
        (1.0 - x2) * pdf,
        (x2 * x - 3.0 * x) * pdf,
        -(x2 * x2 - 6.0 * x2 + 3.0) * pdf,
    ]
}

/// A polynomial in `(δq, δb)` truncated at total degree five: `0[i][j]` is the
/// coefficient of `δq^i·δb^j`, and every slot with `i + j > 5` stays zero.
#[derive(Clone, Copy)]
struct TruncatedBivariate([[f64; TAYLOR_SLOTS]; TAYLOR_SLOTS]);

impl TruncatedBivariate {
    const ZERO: Self = Self([[0.0; TAYLOR_SLOTS]; TAYLOR_SLOTS]);

    const ONE: Self = {
        let mut one = [[0.0; TAYLOR_SLOTS]; TAYLOR_SLOTS];
        one[0][0] = 1.0;
        Self(one)
    };

    /// The product, truncated at total degree five.
    fn mul(&self, other: &Self) -> Self {
        let mut out = Self::ZERO;
        for i in 0..TAYLOR_SLOTS {
            for j in 0..TAYLOR_SLOTS - i {
                let left = self.0[i][j];
                for k in 0..TAYLOR_SLOTS - i - j {
                    for l in 0..TAYLOR_SLOTS - i - j - k {
                        out.0[i + k][j + l] += left * other.0[k][l];
                    }
                }
            }
        }
        out
    }

    /// `self += scale·other·δb^shift`, truncated at total degree five.
    fn add_scaled_shifted(&mut self, scale: f64, other: &Self, shift: usize) {
        for i in 0..TAYLOR_SLOTS - shift {
            for j in 0..TAYLOR_SLOTS - shift - i {
                self.0[i][j + shift] += scale * other.0[i][j];
            }
        }
    }
}

/// The anchor's Taylor table at a solved root (gam#2928): `c[i][j] =
/// ∂_q^i ∂_b^j α / (i!·j!)` through total order five.
///
/// Implicit differentiation of `H(α, b) = Σ_k w_k F(α + b u_k) = F(q)`,
/// `F(x) = Φ(−x)`, in the truncated polynomial algebra. Every partial of `H`
/// is a moment of the law, `H_{rs} = ∂_α^r ∂_b^s H = Σ_k w_k u_k^s
/// F^{(r+s)}(η_k)`, so one pass over the nodes gives the equation's whole
/// Taylor polynomial, and the fixed point
///
/// ```text
///     δα ← −[ Σ_{(r,s) ∉ {(0,0),(1,0)}} H_{rs}·δα^r·δb^s / (r!·s!) − Σ_n F^{(n)}(q)·δq^n / n! ] / H_{10}
/// ```
///
/// fixes one more total degree per pass, so five passes are exact through
/// order five. [`AnchorDerivatives`] is the same object through order three
/// in closed form; `anchor_tests` pins the two against each other and against
/// Newton's iteration in the jet algebra.
#[derive(Clone, Copy, Debug)]
pub(crate) struct AnchorTaylor {
    coefficients: [[f64; TAYLOR_SLOTS]; TAYLOR_SLOTS],
}

impl AnchorTaylor {
    /// The table at an already solved anchor.
    pub(crate) fn at(
        alpha: f64,
        q: f64,
        observed_slope: f64,
        grid: AnchorGrid<'_>,
    ) -> Result<Self, String> {
        // moments[n][s] = Σ_k w_k u_k^s F^{(n)}(η_k), 1 ≤ n ≤ 5, s ≤ n.
        let mut moments = [[0.0_f64; TAYLOR_SLOTS]; TAYLOR_SLOTS];
        for (&u, &w) in grid.nodes.iter().zip(grid.weights.iter()) {
            let stack = survival_cdf_derivative_stack(alpha + observed_slope * u, w);
            let mut power = 1.0;
            for s in 0..TAYLOR_SLOTS {
                for n in s.max(1)..TAYLOR_SLOTS {
                    moments[n][s] += stack[n] * power;
                }
                power *= u;
            }
        }
        let h_alpha = moments[1][0];
        if !(h_alpha.is_finite() && h_alpha < 0.0) {
            return Err(format!(
                "survival marginal-slope anchor has no finite gradient: H_α={h_alpha:e} at α={alpha}, q={q}, b={observed_slope}"
            ));
        }
        let target = survival_cdf_derivative_stack(q, 1.0);
        let mut delta = TruncatedBivariate::ZERO;
        for _ in 0..ANCHOR_TAYLOR_ORDER {
            let mut rest = TruncatedBivariate::ZERO;
            for n in 1..TAYLOR_SLOTS {
                rest.0[n][0] = -target[n] / FACTORIAL[n];
            }
            let mut power = TruncatedBivariate::ONE;
            for r in 0..TAYLOR_SLOTS {
                if r > 0 {
                    power = power.mul(&delta);
                }
                for s in 0..TAYLOR_SLOTS - r {
                    if (r, s) != (0, 0) && (r, s) != (1, 0) {
                        let scale = moments[r + s][s] / (FACTORIAL[r] * FACTORIAL[s]);
                        rest.add_scaled_shifted(scale, &power, s);
                    }
                }
            }
            for i in 0..TAYLOR_SLOTS {
                for j in 0..TAYLOR_SLOTS - i {
                    delta.0[i][j] = -rest.0[i][j] / h_alpha;
                }
            }
        }
        let mut coefficients = delta.0;
        coefficients[0][0] = alpha;
        if !coefficients.iter().flatten().all(|c| c.is_finite()) {
            return Err(format!(
                "survival marginal-slope anchor Taylor table is non-finite at α={alpha}, q={q}, b={observed_slope}"
            ));
        }
        Ok(Self { coefficients })
    }

    /// `α` over any carrier: the table composed with `(q − q*, b − b*)`.
    /// `delta_b` is the observed-slope carrier with its value channel zero.
    #[inline]
    pub(crate) fn lift<T: JetField>(&self, q: &T, delta_b: &T) -> T {
        self.lift_q_derivative_of_order(q, delta_b, 0)
    }

    /// `∂α/∂q` over any carrier, from the same table one order higher in `q`.
    #[inline]
    pub(crate) fn lift_q_derivative<T: JetField>(&self, q: &T, delta_b: &T) -> T {
        self.lift_q_derivative_of_order(q, delta_b, 1)
    }

    /// `∂_q^shift α` by Horner's rule in `δb` over univariate compositions in
    /// `q`: `Σ_j p_j(q)·δb^j`, with `p_j`'s derivative stack
    /// `(i + shift)!·c[i + shift][j]`. Carriers truncate at order four, so the
    /// `j = 4` slice is the last one any carrier reads, and the value channel
    /// is `shift!·c[shift][0]` exactly because `δb` carries a zero value.
    fn lift_q_derivative_of_order<T: JetField>(&self, q: &T, delta_b: &T, shift: usize) -> T {
        let slice = |j: usize| -> [f64; 5] {
            std::array::from_fn(|i| {
                let source = i + shift;
                if source + j <= ANCHOR_TAYLOR_ORDER {
                    FACTORIAL[source] * self.coefficients[source][j]
                } else {
                    0.0
                }
            })
        };
        let mut acc = q.compose_unary(slice(4));
        for j in (0..4).rev() {
            acc = q.compose_unary(slice(j)).add(&delta_b.mul(&acc));
        }
        acc
    }
}

#[cfg(test)]
mod anchor_tests {
    use super::super::test_support::{gauss_hermite_probabilists, skewed_grid};
    use super::*;
    use gam_math::jet_scalar::{JetScalar, Order2};
    use gam_math::jet_tower::Tower4;

    // ── The jet-algebra oracle the Taylor table is pinned against ──────────

    /// `[Φ(−x), F′, F″, F‴, F⁗]` with `F(x) = Φ(−x)`: `F^{(n)} = (−1)^n He_{n−1}(x) φ(x)`.
    fn survival_cdf_stack(x: f64) -> [f64; 5] {
        let pdf = normal_pdf(x);
        [
            normal_cdf(-x),
            -pdf,
            x * pdf,
            (1.0 - x * x) * pdf,
            (x * x * x - 3.0 * x) * pdf,
        ]
    }

    /// `[F′, F″, F‴, F⁗, F⁽⁵⁾]` of the same `F`: the stack of `−φ`.
    fn survival_cdf_prime_stack(x: f64) -> [f64; 5] {
        let pdf = normal_pdf(x);
        let x2 = x * x;
        [
            -pdf,
            x * pdf,
            (1.0 - x2) * pdf,
            (x2 * x - 3.0 * x) * pdf,
            -(x2 * x2 - 6.0 * x2 + 3.0) * pdf,
        ]
    }

    /// `[φ, φ′, φ″, φ‴, φ⁗]`: `φ^{(n)} = (−1)^n He_n(x) φ(x)`.
    fn pdf_stack(x: f64) -> [f64; 5] {
        let pdf = normal_pdf(x);
        let x2 = x * x;
        [
            pdf,
            -x * pdf,
            (x2 - 1.0) * pdf,
            -(x2 * x - 3.0 * x) * pdf,
            (x2 * x2 - 6.0 * x2 + 3.0) * pdf,
        ]
    }

    /// `[1/x, −1/x², 2/x³, −6/x⁴, 24/x⁵]`.
    fn reciprocal_stack(x: f64) -> [f64; 5] {
        let inv = 1.0 / x;
        let inv2 = inv * inv;
        let inv3 = inv2 * inv;
        [inv, -inv2, 2.0 * inv3, -6.0 * inv3 * inv, 24.0 * inv3 * inv2]
    }

    /// The anchor as a jet in whatever `q` and `b` carry, by Newton's
    /// iteration in the truncated jet algebra: from the exact real root one
    /// step is exact through first order, two through third, three through
    /// seventh. The value channel is pinned back to the solver's root.
    /// `alpha_star` MUST be the converged root at `(q.value(), b.value())`.
    fn anchor_jet<T: JetField>(alpha_star: f64, q: &T, b: &T, grid: AnchorGrid<'_>) -> T {
        let mut alpha = q.constant_like(alpha_star);
        for _ in 0..3 {
            // G = Σ_k w_k F(η_k) − F(q),   G_α = Σ_k w_k F′(η_k).
            let mut g = q.compose_unary(survival_cdf_stack(q.value())).neg();
            let mut g_alpha = q.constant_like(0.0);
            for (&u, &w) in grid.nodes.iter().zip(grid.weights.iter()) {
                let eta = alpha.add(&b.scale(u));
                let eta_value = eta.value();
                g = g.add(&eta.compose_unary(survival_cdf_stack(eta_value)).scale(w));
                g_alpha =
                    g_alpha.add(&eta.compose_unary(survival_cdf_prime_stack(eta_value)).scale(w));
            }
            let step = g.mul(&g_alpha.compose_unary(reciprocal_stack(g_alpha.value())));
            alpha = alpha.sub(&step);
        }
        alpha.with_value(alpha_star)
    }

    /// `∂α/∂q = φ(q) / Σ_k w_k φ(η_k)` as a jet, at an anchor jet already
    /// lifted by [`anchor_jet`].
    fn anchor_q_derivative_jet<T: JetField>(alpha: &T, q: &T, b: &T, grid: AnchorGrid<'_>) -> T {
        let phi_q = q.compose_unary(pdf_stack(q.value()));
        let mut density = q.constant_like(0.0);
        for (&u, &w) in grid.nodes.iter().zip(grid.weights.iter()) {
            let eta = alpha.add(&b.scale(u));
            density = density.add(&eta.compose_unary(pdf_stack(eta.value())).scale(w));
        }
        phi_q.mul(&density.compose_unary(reciprocal_stack(density.value())))
    }

    /// `∂α/∂b = −Σ_k w_k φ(η_k) u_k / Σ_k w_k φ(η_k)` as a jet: the factor a
    /// follow-up-varying slope's rate `ḃ₁` enters `α̇₁` with.
    fn anchor_b_derivative_jet<T: JetField>(alpha: &T, q: &T, b: &T, grid: AnchorGrid<'_>) -> T {
        let mut density = q.constant_like(0.0);
        let mut moment = q.constant_like(0.0);
        for (&u, &w) in grid.nodes.iter().zip(grid.weights.iter()) {
            let eta = alpha.add(&b.scale(u));
            let pdf = eta.compose_unary(pdf_stack(eta.value()));
            density = density.add(&pdf.scale(w));
            moment = moment.add(&pdf.scale(w * u));
        }
        moment.mul(&density.compose_unary(reciprocal_stack(density.value()))).neg()
    }

    fn owned(nodes: Vec<f64>, weights: Vec<f64>) -> AnchorGridOwned {
        AnchorGridOwned::new(nodes, weights)
    }

    /// Solve the anchor and differentiate it.
    fn solve_derivatives(q: f64, b: f64, grid: AnchorGrid<'_>) -> Result<AnchorDerivatives, String> {
        AnchorDerivatives::at(solve_anchor(q, b, grid)?, q, b, grid)
    }

    fn marginal(alpha: f64, b: f64, grid: AnchorGrid<'_>) -> f64 {
        grid.nodes
            .iter()
            .zip(grid.weights.iter())
            .map(|(&u, &w)| w * normal_cdf(-(alpha + b * u)))
            .sum()
    }

    #[test]
    fn gauss_hermite_law_integrates_moments() {
        for m in [2usize, 5, 16, 33, 65] {
            let (nodes, weights) = gauss_hermite_probabilists(m).expect("grid");
            assert_eq!(nodes.len(), m);
            let total: f64 = weights.iter().sum();
            assert!((total - 1.0).abs() < 1e-12, "m={m} total={total}");
            for k in 1..m {
                assert!(nodes[k] > nodes[k - 1], "m={m} nodes must ascend");
            }
            let mean: f64 = nodes.iter().zip(&weights).map(|(u, w)| u * w).sum();
            let var: f64 = nodes.iter().zip(&weights).map(|(u, w)| u * u * w).sum();
            assert!(mean.abs() < 1e-12, "m={m} mean={mean}");
            assert!((var - 1.0).abs() < 1e-10, "m={m} var={var}");
            if m >= 4 {
                let fourth: f64 = nodes.iter().zip(&weights).map(|(u, w)| u.powi(4) * w).sum();
                assert!((fourth - 3.0).abs() < 1e-9, "m={m} fourth={fourth}");
            }
        }
    }

    /// The anchoring equation is solved: the anchored marginal survival is
    /// `Φ(−q)` to the log-residual tolerance, across both tails and slopes of
    /// both signs, on a law that is nothing like Gaussian.
    #[test]
    fn anchor_solves_the_anchoring_equation_on_a_skewed_law() {
        let grid = skewed_grid();
        for &q in &[-3.5, -1.2, -0.3, 0.0, 0.4, 1.7, 3.2, 5.5] {
            for &b in &[-1.3, -0.4, 0.0, 0.25, 0.9, 2.1] {
                let alpha = solve_anchor(q, b, grid.view()).expect("anchor");
                let lhs = marginal(alpha, b, grid.view());
                let rhs = normal_cdf(-q);
                let smaller = rhs.min(1.0 - rhs);
                assert!(
                    (lhs - rhs).abs() <= 1e-10 * smaller.max(1e-300),
                    "q={q} b={b}: Σ w Φ(−η) = {lhs:e} vs Φ(−q) = {rhs:e}"
                );
            }
        }
    }

    /// Gaussian is the special case: on a Gauss–Hermite law the anchor is the
    /// closed form `q·√(1 + b²)` to quadrature tolerance, and its derivatives
    /// are the closed form's.
    #[test]
    fn gaussian_grid_reproduces_the_closed_form() {
        let (nodes, weights) = gauss_hermite_probabilists(65).expect("grid");
        let grid = owned(nodes, weights);
        for &q in &[-2.5, -0.7, 0.0, 0.6, 2.2] {
            for &b in &[-1.1, -0.3, 0.0, 0.45, 1.6] {
                let d = solve_derivatives(q, b, grid.view()).expect("anchor");
                let c = (1.0 + b * b).sqrt();
                assert!((d.alpha - q * c).abs() < 1e-9, "q={q} b={b}: α={} vs {}", d.alpha, q * c);
                assert!((d.a_q - c).abs() < 1e-8, "q={q} b={b}: α_q={} vs {c}", d.a_q);
                assert!((d.a_b - q * b / c).abs() < 1e-8, "q={q} b={b}: α_b={} vs {}", d.a_b, q * b / c);
                assert!(d.a_qq.abs() < 1e-8, "q={q} b={b}: α_qq={}", d.a_qq);
                assert!((d.a_qb - b / c).abs() < 1e-7, "q={q} b={b}: α_qb={} vs {}", d.a_qb, b / c);
                assert!((d.a_bb - q / (c * c * c)).abs() < 1e-7, "q={q} b={b}: α_bb={} vs {}", d.a_bb, q / c.powi(3));
            }
        }
    }

    /// The declared grid has to be sized to the drive `b`: a Gauss–Hermite law
    /// reproduces the closed form `q·√(1 + b²)` only to a tolerance that
    /// shrinks with its node count, and at slopes of 2–4 a 16-node law is
    /// visibly wrong where 64 nodes are at 1e-4 and 128 at round-off.
    #[test]
    fn gauss_hermite_grid_size_sets_the_closed_form_tolerance() {
        let mut previous = f64::INFINITY;
        for (m, bound) in [(16_usize, 5e-1), (64, 2e-3), (128, 1e-5)] {
            let (nodes, weights) = gauss_hermite_probabilists(m).expect("grid");
            let grid = owned(nodes, weights);
            let mut worst = 0.0_f64;
            for &b in &[2.0, 3.0, 4.0] {
                for &q in &[-1.0, 0.5, 2.0] {
                    let alpha = solve_anchor(q, b, grid.view()).expect("anchor");
                    worst = worst.max((alpha - q * (1.0 + b * b).sqrt()).abs());
                }
            }
            eprintln!(
                "[2923 grid] m={m}: max |α − q√(1+b²)| over b ∈ {{2, 3, 4}}, q ∈ {{−1, 0.5, 2}} = {worst:.3e}"
            );
            assert!(worst < bound, "m={m}: {worst:.3e} exceeds {bound:.1e}");
            assert!(
                worst < previous,
                "m={m}: the tolerance must shrink with the node count ({worst:.3e} vs {previous:.3e})"
            );
            previous = worst;
        }
    }

    /// Every closed-form implicit derivative through order three matches a
    /// central difference of the solved anchor on the skewed law.
    #[test]
    fn implicit_derivatives_match_finite_differences() {
        let grid = skewed_grid();
        let solve = |q: f64, b: f64| solve_anchor(q, b, grid.view()).expect("anchor");
        // Central differences carry an `h²` truncation term; at `1e-4` it sits
        // below the tolerances while round-off on the closed-form derivatives
        // (each exact to ~1e-13) stays near `1e-9`.
        let h = 1e-4;
        for &q in &[-1.4, -0.2, 0.5, 1.9] {
            for &b in &[-0.8, 0.1, 0.7, 1.5] {
                let d = solve_derivatives(q, b, grid.view()).expect("derivatives");
                let f = |dq: f64, db: f64| solve(q + dq, b + db);
                // First order: five-point stencils.
                let fd_q = (-f(2.0 * h, 0.0) + 8.0 * f(h, 0.0) - 8.0 * f(-h, 0.0) + f(-2.0 * h, 0.0)) / (12.0 * h);
                let fd_b = (-f(0.0, 2.0 * h) + 8.0 * f(0.0, h) - 8.0 * f(0.0, -h) + f(0.0, -2.0 * h)) / (12.0 * h);
                assert!((d.a_q - fd_q).abs() < 1e-8 * (1.0 + fd_q.abs()), "q={q} b={b}: α_q {} vs fd {fd_q}", d.a_q);
                assert!((d.a_b - fd_b).abs() < 1e-8 * (1.0 + fd_b.abs()), "q={q} b={b}: α_b {} vs fd {fd_b}", d.a_b);
                // Second order from the closed-form first derivatives.
                let dq = |dq: f64, db: f64| solve_derivatives(q + dq, b + db, grid.view()).expect("d");
                let fd_qq = (dq(h, 0.0).a_q - dq(-h, 0.0).a_q) / (2.0 * h);
                let fd_qb = (dq(0.0, h).a_q - dq(0.0, -h).a_q) / (2.0 * h);
                let fd_bq = (dq(h, 0.0).a_b - dq(-h, 0.0).a_b) / (2.0 * h);
                let fd_bb = (dq(0.0, h).a_b - dq(0.0, -h).a_b) / (2.0 * h);
                assert!((d.a_qq - fd_qq).abs() < 1e-6 * (1.0 + fd_qq.abs()), "q={q} b={b}: α_qq {} vs fd {fd_qq}", d.a_qq);
                assert!((d.a_qb - fd_qb).abs() < 1e-6 * (1.0 + fd_qb.abs()), "q={q} b={b}: α_qb {} vs fd {fd_qb}", d.a_qb);
                assert!((d.a_qb - fd_bq).abs() < 1e-6 * (1.0 + fd_bq.abs()), "q={q} b={b}: α_qb {} vs fd(b,q) {fd_bq}", d.a_qb);
                assert!((d.a_bb - fd_bb).abs() < 1e-6 * (1.0 + fd_bb.abs()), "q={q} b={b}: α_bb {} vs fd {fd_bb}", d.a_bb);
                // Third order from the closed-form second derivatives.
                let fd_qqq = (dq(h, 0.0).a_qq - dq(-h, 0.0).a_qq) / (2.0 * h);
                let fd_qqb = (dq(0.0, h).a_qq - dq(0.0, -h).a_qq) / (2.0 * h);
                let fd_qbb = (dq(0.0, h).a_qb - dq(0.0, -h).a_qb) / (2.0 * h);
                let fd_bbb = (dq(0.0, h).a_bb - dq(0.0, -h).a_bb) / (2.0 * h);
                let fd_qbq = (dq(h, 0.0).a_qb - dq(-h, 0.0).a_qb) / (2.0 * h);
                assert!((d.a_qqq - fd_qqq).abs() < 1e-5 * (1.0 + fd_qqq.abs()), "q={q} b={b}: α_qqq {} vs fd {fd_qqq}", d.a_qqq);
                assert!((d.a_qqb - fd_qqb).abs() < 1e-5 * (1.0 + fd_qqb.abs()), "q={q} b={b}: α_qqb {} vs fd {fd_qqb}", d.a_qqb);
                assert!((d.a_qqb - fd_qbq).abs() < 1e-5 * (1.0 + fd_qbq.abs()), "q={q} b={b}: α_qqb {} vs fd(qb,q) {fd_qbq}", d.a_qqb);
                assert!((d.a_qbb - fd_qbb).abs() < 1e-5 * (1.0 + fd_qbb.abs()), "q={q} b={b}: α_qbb {} vs fd {fd_qbb}", d.a_qbb);
                assert!((d.a_bbb - fd_bbb).abs() < 1e-5 * (1.0 + fd_bbb.abs()), "q={q} b={b}: α_bbb {} vs fd {fd_bbb}", d.a_bbb);
            }
        }
    }

    /// The jet lift carries the same derivatives as the closed form through
    /// order two (`Order2<2>`) and through order four (`Tower4<2>`), and its
    /// value channel is bitwise the solver's root.
    #[test]
    fn anchor_jet_matches_closed_form_and_finite_differences() {
        let grid = skewed_grid();
        for &q in &[-1.1, 0.3, 1.6] {
            for &b in &[-0.6, 0.2, 1.2] {
                let d = solve_derivatives(q, b, grid.view()).expect("derivatives");
                let qj = Order2::<2>::variable(q, 0);
                let bj = Order2::<2>::variable(b, 1);
                let aj = anchor_jet(d.alpha, &qj, &bj, grid.view());
                assert_eq!(aj.value().to_bits(), d.alpha.to_bits(), "value channel is the root");
                let tower = aj.0;
                assert!((tower.g[0] - d.a_q).abs() < 1e-10 * (1.0 + d.a_q.abs()), "jet α_q {} vs {}", tower.g[0], d.a_q);
                assert!((tower.g[1] - d.a_b).abs() < 1e-10 * (1.0 + d.a_b.abs()), "jet α_b {} vs {}", tower.g[1], d.a_b);
                assert!((tower.h[0][0] - d.a_qq).abs() < 1e-9 * (1.0 + d.a_qq.abs()), "jet α_qq {} vs {}", tower.h[0][0], d.a_qq);
                assert!((tower.h[0][1] - d.a_qb).abs() < 1e-9 * (1.0 + d.a_qb.abs()), "jet α_qb {} vs {}", tower.h[0][1], d.a_qb);
                assert!((tower.h[1][1] - d.a_bb).abs() < 1e-9 * (1.0 + d.a_bb.abs()), "jet α_bb {} vs {}", tower.h[1][1], d.a_bb);

                let qt = Tower4::<2>::variable(q, 0);
                let bt = Tower4::<2>::variable(b, 1);
                let at = anchor_jet(d.alpha, &qt, &bt, grid.view());
                assert!((at.t3[0][0][0] - d.a_qqq).abs() < 1e-8 * (1.0 + d.a_qqq.abs()), "tower α_qqq {} vs {}", at.t3[0][0][0], d.a_qqq);
                assert!((at.t3[0][0][1] - d.a_qqb).abs() < 1e-8 * (1.0 + d.a_qqb.abs()), "tower α_qqb {} vs {}", at.t3[0][0][1], d.a_qqb);
                assert!((at.t3[0][1][1] - d.a_qbb).abs() < 1e-8 * (1.0 + d.a_qbb.abs()), "tower α_qbb {} vs {}", at.t3[0][1][1], d.a_qbb);
                assert!((at.t3[1][1][1] - d.a_bbb).abs() < 1e-8 * (1.0 + d.a_bbb.abs()), "tower α_bbb {} vs {}", at.t3[1][1][1], d.a_bbb);
                // Fourth order against a central difference of the third.
                let h = 1e-4;
                let third = |dq: f64, db: f64| solve_derivatives(q + dq, b + db, grid.view()).expect("d");
                let fd_qqqq = (third(h, 0.0).a_qqq - third(-h, 0.0).a_qqq) / (2.0 * h);
                let fd_qqqb = (third(0.0, h).a_qqq - third(0.0, -h).a_qqq) / (2.0 * h);
                let fd_qqbb = (third(0.0, h).a_qqb - third(0.0, -h).a_qqb) / (2.0 * h);
                let fd_qbbb = (third(0.0, h).a_qbb - third(0.0, -h).a_qbb) / (2.0 * h);
                let fd_bbbb = (third(0.0, h).a_bbb - third(0.0, -h).a_bbb) / (2.0 * h);
                assert!((at.t4[0][0][0][0] - fd_qqqq).abs() < 1e-5 * (1.0 + fd_qqqq.abs()), "tower α_qqqq {} vs fd {fd_qqqq}", at.t4[0][0][0][0]);
                assert!((at.t4[0][0][0][1] - fd_qqqb).abs() < 1e-5 * (1.0 + fd_qqqb.abs()), "tower α_qqqb {} vs fd {fd_qqqb}", at.t4[0][0][0][1]);
                assert!((at.t4[0][0][1][1] - fd_qqbb).abs() < 1e-5 * (1.0 + fd_qqbb.abs()), "tower α_qqbb {} vs fd {fd_qqbb}", at.t4[0][0][1][1]);
                assert!((at.t4[0][1][1][1] - fd_qbbb).abs() < 1e-5 * (1.0 + fd_qbbb.abs()), "tower α_qbbb {} vs fd {fd_qbbb}", at.t4[0][1][1][1]);
                assert!((at.t4[1][1][1][1] - fd_bbbb).abs() < 1e-5 * (1.0 + fd_bbbb.abs()), "tower α_bbbb {} vs fd {fd_bbbb}", at.t4[1][1][1][1]);

                // The q- and b-derivative jets agree with the closed form through order two.
                let aq = anchor_q_derivative_jet(&aj, &qj, &bj, grid.view()).0;
                assert!((aq.v - d.a_q).abs() < 1e-10 * (1.0 + d.a_q.abs()));
                assert!((aq.g[0] - d.a_qq).abs() < 1e-9 * (1.0 + d.a_qq.abs()), "α_q jet g[q] {} vs {}", aq.g[0], d.a_qq);
                assert!((aq.g[1] - d.a_qb).abs() < 1e-9 * (1.0 + d.a_qb.abs()), "α_q jet g[b] {} vs {}", aq.g[1], d.a_qb);
                assert!((aq.h[0][0] - d.a_qqq).abs() < 1e-8 * (1.0 + d.a_qqq.abs()), "α_q jet h[qq] {} vs {}", aq.h[0][0], d.a_qqq);
                assert!((aq.h[0][1] - d.a_qqb).abs() < 1e-8 * (1.0 + d.a_qqb.abs()), "α_q jet h[qb] {} vs {}", aq.h[0][1], d.a_qqb);
                assert!((aq.h[1][1] - d.a_qbb).abs() < 1e-8 * (1.0 + d.a_qbb.abs()), "α_q jet h[bb] {} vs {}", aq.h[1][1], d.a_qbb);
                let ab = anchor_b_derivative_jet(&aj, &qj, &bj, grid.view()).0;
                assert!((ab.v - d.a_b).abs() < 1e-10 * (1.0 + d.a_b.abs()));
                assert!((ab.g[0] - d.a_qb).abs() < 1e-9 * (1.0 + d.a_qb.abs()), "α_b jet g[q] {} vs {}", ab.g[0], d.a_qb);
                assert!((ab.g[1] - d.a_bb).abs() < 1e-9 * (1.0 + d.a_bb.abs()), "α_b jet g[b] {} vs {}", ab.g[1], d.a_bb);
                assert!((ab.h[1][1] - d.a_bbb).abs() < 1e-8 * (1.0 + d.a_bbb.abs()), "α_b jet h[bb] {} vs {}", ab.h[1][1], d.a_bbb);
            }
        }
    }

    /// The Taylor table is the closed-form implicit derivatives through order
    /// three, and its lift onto the full order-four tower is Newton's iteration
    /// in the jet algebra, for `α` and for `α_q` — which reads the table's
    /// fifth order (gam#2928).
    #[test]
    fn taylor_table_matches_the_closed_form_and_the_jet_oracle() {
        let grid = skewed_grid();
        for &q in &[-2.1, -0.6, 0.3, 1.6, 3.4] {
            for &b in &[-1.2, -0.2, 0.0, 0.7, 1.9] {
                let d = solve_derivatives(q, b, grid.view()).expect("derivatives");
                let taylor = AnchorTaylor::at(d.alpha, q, b, grid.view()).expect("table");
                let c = &taylor.coefficients;
                let close = |left: f64, right: f64, tol: f64, what: &str| {
                    assert!(
                        (left - right).abs() <= tol * (1.0 + right.abs()),
                        "q={q} b={b} {what}: {left:+.15e} vs {right:+.15e}"
                    );
                };
                close(c[1][0], d.a_q, 1e-12, "table α_q");
                close(c[0][1], d.a_b, 1e-12, "table α_b");
                close(2.0 * c[2][0], d.a_qq, 1e-11, "table α_qq");
                close(c[1][1], d.a_qb, 1e-11, "table α_qb");
                close(2.0 * c[0][2], d.a_bb, 1e-11, "table α_bb");
                close(6.0 * c[3][0], d.a_qqq, 1e-10, "table α_qqq");
                close(2.0 * c[2][1], d.a_qqb, 1e-10, "table α_qqb");
                close(2.0 * c[1][2], d.a_qbb, 1e-10, "table α_qbb");
                close(6.0 * c[0][3], d.a_bbb, 1e-10, "table α_bbb");

                let qt = Tower4::<2>::variable(q, 0);
                let bt = Tower4::<2>::variable(b, 1);
                let delta_b = bt.compose_unary([0.0, 1.0, 0.0, 0.0, 0.0]);
                let lifted = taylor.lift(&qt, &delta_b);
                let oracle = anchor_jet(d.alpha, &qt, &bt, grid.view());
                assert_eq!(lifted.value().to_bits(), d.alpha.to_bits(), "the lift's value is the root");
                let lifted_q = taylor.lift_q_derivative(&qt, &delta_b);
                let oracle_q = anchor_q_derivative_jet(&oracle, &qt, &bt, grid.view());
                for (what, left, right) in [("α", &lifted, &oracle), ("α_q", &lifted_q, &oracle_q)] {
                    close(left.value(), right.value(), 1e-12, &format!("{what} value"));
                    for i in 0..2 {
                        close(left.g[i], right.g[i], 1e-11, &format!("{what} g[{i}]"));
                        for j in 0..2 {
                            close(left.h[i][j], right.h[i][j], 1e-10, &format!("{what} h[{i}][{j}]"));
                            for k in 0..2 {
                                close(left.t3[i][j][k], right.t3[i][j][k], 1e-9, &format!("{what} t3[{i}][{j}][{k}]"));
                                for l in 0..2 {
                                    close(
                                        left.t4[i][j][k][l],
                                        right.t4[i][j][k][l],
                                        1e-8,
                                        &format!("{what} t4[{i}][{j}][{k}][{l}]"),
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// A row's slot answers only for bitwise the inputs its root was solved
    /// for: a sentinel stored under `(q, b)` comes back for exactly `(q, b)`,
    /// one bit off in either input is a different equation and is solved, a
    /// warm solve is published under its own inputs, and a root stored far
    /// away cannot poison the solve it seeds.
    #[test]
    fn anchor_slot_answers_only_its_own_equation_and_survives_a_poisoned_seed() {
        let grid = skewed_grid();
        let cache = AnchorRootCache::new(4, false);
        let context = AnchorRowContext {
            grid: grid.view(),
            roots: Some(&cache),
        };
        let slot = SurvivalInterceptSlotKind::Exit;
        let (q, b) = (0.7_f64, 1.1_f64);

        let cold = solve_anchor_in_slot(q, b, context, 2, slot).expect("cold solve");
        assert_eq!(cold.to_bits(), solve_anchor(q, b, grid.view()).expect("closed form").to_bits());
        // Stored under exactly these inputs the slot IS the answer: a sentinel
        // comes back, which no solve could produce.
        let own = cache.slot(2, slot).expect("row 2 slot");
        own.write(q.to_bits(), b.to_bits(), 123.0);
        assert_eq!(solve_anchor_in_slot(q, b, context, 2, slot).expect("reuse"), 123.0);
        for (nudged_q, nudged_b) in [
            (f64::from_bits(q.to_bits() + 1), b),
            (q, f64::from_bits(b.to_bits() + 1)),
        ] {
            own.write(q.to_bits(), b.to_bits(), 123.0);
            let nudged = solve_anchor_in_slot(nudged_q, nudged_b, context, 2, slot).expect("nudged");
            let fresh = solve_anchor(nudged_q, nudged_b, grid.view()).expect("fresh nudged");
            assert!(
                nudged != 123.0 && (nudged - fresh).abs() <= 1e-10 * (1.0 + fresh.abs()),
                "a one-bit-different equation must be solved: {nudged} vs {fresh}"
            );
        }

        cache.slot(1, slot).expect("row 1 slot").write(q.to_bits(), b.to_bits(), cold);
        let warm = solve_anchor_in_slot(0.7003, 1.0996, context, 1, slot).expect("warm solve");
        let fresh = solve_anchor(0.7003, 1.0996, grid.view()).expect("fresh solve");
        assert!((warm - fresh).abs() <= 1e-10 * (1.0 + fresh.abs()), "warm {warm} vs fresh {fresh}");
        assert_eq!(
            cache.slot(1, slot).and_then(RootSlot::read).map(|(sq, sb, root)| (sq, sb, root.to_bits())),
            Some((0.7003_f64.to_bits(), 1.0996_f64.to_bits(), warm.to_bits())),
            "the warm solve is published under its own inputs"
        );

        cache.slot(3, slot).expect("row 3 slot").write(5.0_f64.to_bits(), 0.2_f64.to_bits(), 1e6);
        let poisoned = solve_anchor_in_slot(-0.4, 0.9, context, 3, slot).expect("poisoned seed");
        let reference = solve_anchor(-0.4, 0.9, grid.view()).expect("reference");
        assert!(
            (poisoned - reference).abs() <= 1e-10 * (1.0 + reference.abs()),
            "poisoned {poisoned} vs reference {reference}"
        );
    }

    /// Rows asking a global law the same equation share one root through the
    /// cross-row table. A hash only selects an entry: with every equation
    /// forced onto one entry by a colliding test hasher, a different equation
    /// is never a hit. A per-row law keeps no table at all.
    #[test]
    fn anchor_cross_row_table_decides_hits_by_exact_inputs() {
        let grid = skewed_grid();
        let cache = AnchorRootCache::with_shared_index(4, true, |_, _| 7);
        let context = AnchorRowContext {
            grid: grid.view(),
            roots: Some(&cache),
        };
        let slot = SurvivalInterceptSlotKind::Entry;
        let (q, b) = (-1.3_f64, 0.8_f64);

        let first = solve_anchor_in_slot(q, b, context, 0, slot).expect("row 0");
        // A sentinel stored for (q, b) reaches row 1 asking the same equation
        // without a solve, and is published into row 1's own slot.
        cache
            .shared_slot(q.to_bits(), b.to_bits())
            .expect("global law table")
            .write(q.to_bits(), b.to_bits(), 77.0);
        assert_eq!(solve_anchor_in_slot(q, b, context, 1, slot).expect("row 1"), 77.0);
        assert_eq!(
            cache.slot(1, slot).and_then(RootSlot::read),
            Some((q.to_bits(), b.to_bits(), 77.0))
        );

        // A different equation hashed onto the same entry is solved on its own.
        let (other_q, other_b) = (0.4_f64, -0.5_f64);
        let colliding = solve_anchor_in_slot(other_q, other_b, context, 2, slot).expect("row 2");
        let reference = solve_anchor(other_q, other_b, grid.view()).expect("reference");
        assert_eq!(colliding.to_bits(), reference.to_bits(), "a colliding entry is never a hit");
        assert!((first - reference).abs() > 1e-3, "the two equations have different roots");

        assert!(
            AnchorRootCache::new(4, false).shared_slot(q.to_bits(), b.to_bits()).is_none(),
            "a per-row law keeps no cross-row table"
        );
    }

    /// The linear-space residual is the log-space residual to rounding wherever
    /// its tail clears the floor, on both sides and across slopes, and a tail
    /// below the floor is handed to the log-space form unchanged (gam#2928).
    #[test]
    fn linear_space_residual_matches_the_log_space_residual() {
        let grid = skewed_grid();
        let mut compared = 0usize;
        for &q in &[-6.0, -2.2, -0.4, 0.0, 0.9, 3.1, 7.5] {
            let survival_side = q >= 0.0;
            let log_target = if survival_side {
                normal_logcdf(-q)
            } else {
                normal_logcdf(q)
            };
            for &b in &[-2.5, -0.7, 0.0, 0.35, 1.6, 4.0] {
                for &offset in &[-3.0, -0.5, 0.0, 0.4, 2.5] {
                    let alpha = gaussian_anchor(q, b) + offset;
                    let linear = anchor_log_residual(alpha, b, grid.view(), survival_side, log_target)
                        .expect("linear-space residual");
                    let logged =
                        anchor_log_residual_in_log_space(alpha, b, grid.view(), survival_side, log_target)
                            .expect("log-space residual");
                    // Rounding, not a picked tolerance: the log-space form rounds each
                    // node's log term — about |log T| in size where it carries weight —
                    // before exponentiating it, and either form sums m node terms, so
                    // the density moments agree to this relative rounding and F, a
                    // difference of two logarithms, to this absolute rounding.
                    let log_tail = logged.0 + log_target;
                    let rounding = (grid.view().len() as f64 + 4.0)
                        * f64::EPSILON
                        * (1.0 + log_tail.abs() + log_target.abs());
                    assert!(
                        (linear.0 - logged.0).abs() <= rounding,
                        "q={q} b={b} α={alpha} F: linear {:+.17e} vs log {:+.17e} (rounding {rounding:.3e})",
                        linear.0,
                        logged.0
                    );
                    assert!(
                        (linear.1 - logged.1).abs() <= rounding * logged.1.abs(),
                        "q={q} b={b} α={alpha} F′: linear {:+.17e} vs log {:+.17e} (rounding {rounding:.3e})",
                        linear.1,
                        logged.1
                    );
                    // F″ = ±Σ m η − (Σ m)² subtracts nearly equal numbers in the tails
                    // (both near 170 against F″ ≈ −1 at q = −6, b = −2.5), which
                    // amplifies either form's rounding a hundredfold. What each form
                    // computes is the moment Σ m η = ±(F″ + F′²): compare it before the
                    // subtraction, allowing the recovery's own rounding.
                    let moment = |r: (f64, f64, f64)| r.2 + r.1 * r.1;
                    let recovery = 2.0
                        * f64::EPSILON
                        * (linear.2.abs() + linear.1 * linear.1 + logged.2.abs() + logged.1 * logged.1);
                    assert!(
                        (moment(linear) - moment(logged)).abs() <= rounding * moment(logged).abs() + recovery,
                        "q={q} b={b} α={alpha} Σ m η: linear {:+.17e} vs log {:+.17e} (rounding {rounding:.3e})",
                        moment(linear),
                        moment(logged)
                    );
                    compared += 1;
                }
            }
        }
        assert_eq!(compared, 7 * 6 * 5);
        // A tail far below the floor (every node past Φ(−44) ≈ 1e-423, which
        // underflows): the two forms are the same evaluation.
        let (q, b, alpha) = (38.0, 0.5, 45.0);
        let log_target = normal_logcdf(-q);
        let deep = anchor_log_residual(alpha, b, grid.view(), true, log_target).expect("deep tail");
        let reference =
            anchor_log_residual_in_log_space(alpha, b, grid.view(), true, log_target).expect("deep reference");
        assert_eq!(
            (deep.0.to_bits(), deep.1.to_bits(), deep.2.to_bits()),
            (reference.0.to_bits(), reference.1.to_bits(), reference.2.to_bits()),
            "below the floor the residual is the log-space form"
        );
    }

    /// Roots solved on the linear-space residual meet the solve's acceptance
    /// criterion measured on the log-space form, and so do the log-space form's
    /// own roots, across `q ∈ [−8, 8]` and slopes to ±4 on both laws, with both
    /// marginal tails `Φ(−q) < 1e-6` and `Φ(−q) > 1 − 1e-6` in the set and shown
    /// to run the linear form (gam#2928). The two roots then differ by at most
    /// what those residuals allow: `F` is monotone, so by the mean value theorem
    /// `|α_lin − α_log| ≤ (|F(α_lin)| + |F(α_log)|) / min |F′|` over the
    /// interval between them. Nothing here is a hand-picked bound: the one
    /// allowance is the rounding of the reference evaluation itself, a sum of
    /// `m` positive terms and a logarithm, `(m + 4)·ε·(1 + |log target|)`.
    #[test]
    fn linear_space_roots_match_the_log_space_roots_across_both_tails() {
        let (nodes, weights) = gauss_hermite_probabilists(65).expect("grid");
        let gaussian = owned(nodes, weights);
        let skewed = skewed_grid();
        let qs = [-8.0, -6.0, -4.8, -3.0, -1.0, -0.2, 0.0, 0.3, 1.5, 3.3, 4.8, 6.5, 8.0];
        assert!(qs.iter().any(|&q| normal_cdf(-q) < 1e-6), "a small survival tail is in the set");
        assert!(qs.iter().any(|&q| normal_cdf(-q) > 1.0 - 1e-6), "a small event tail is in the set");
        let mut worst_gap = 0.0_f64;
        let mut worst_bound_use = 0.0_f64;
        let mut linear_tail_cases = 0usize;
        for (label, grid) in [("gaussian", &gaussian), ("skewed", &skewed)] {
            for &q in &qs {
                let survival_side = q >= 0.0;
                let log_target = if survival_side {
                    normal_logcdf(-q)
                } else {
                    normal_logcdf(q)
                };
                let noise = (grid.view().len() as f64 + 4.0) * f64::EPSILON * (1.0 + log_target.abs());
                for &b in &[-4.0, -1.6, -0.5, 0.0, 0.5, 1.6, 4.0] {
                    let seed = gaussian_anchor(q, b);
                    let linear = solve_anchor_with(q, b, grid.view(), seed, anchor_log_residual)
                        .unwrap_or_else(|e| panic!("{label} q={q} b={b}: linear solve: {e}"));
                    let logged =
                        solve_anchor_with(q, b, grid.view(), seed, anchor_log_residual_in_log_space)
                            .unwrap_or_else(|e| panic!("{label} q={q} b={b}: log solve: {e}"));
                    let reference = |alpha: f64| {
                        anchor_log_residual_in_log_space(alpha, b, grid.view(), survival_side, log_target)
                            .expect("reference residual")
                    };
                    let (f_linear, d_linear, _) = reference(linear);
                    let (f_logged, d_logged, _) = reference(logged);
                    assert!(
                        f_linear.abs() <= anchor_residual_resolution(linear, d_linear) + noise,
                        "{label} q={q} b={b}: linear root {linear:+.17e} has reference residual {f_linear:.3e}"
                    );
                    assert!(
                        f_logged.abs() <= anchor_residual_resolution(logged, d_logged),
                        "{label} q={q} b={b}: log root {logged:+.17e} has reference residual {f_logged:.3e}"
                    );
                    let (_, d_mid, _) = reference(linear + 0.5 * (logged - linear));
                    let min_slope = d_linear.abs().min(d_logged.abs()).min(d_mid.abs());
                    let bound = (f_linear.abs() + f_logged.abs() + 2.0 * noise) / min_slope;
                    let gap = (linear - logged).abs();
                    assert!(
                        gap <= bound,
                        "{label} q={q} b={b}: linear root {linear:+.17e} vs log root {logged:+.17e}: gap {gap:.3e} exceeds the residual bound {bound:.3e}"
                    );
                    worst_gap = worst_gap.max(gap / logged.abs().max(1.0));
                    if bound > 0.0 {
                        worst_bound_use = worst_bound_use.max(gap / bound);
                    }
                    let target_tail = normal_cdf(-q).min(normal_cdf(q));
                    if target_tail < 1e-6 {
                        let tail: f64 = grid
                            .nodes
                            .iter()
                            .zip(grid.weights.iter())
                            .map(|(&u, &w)| {
                                let eta = linear + b * u;
                                w * normal_cdf(if q >= 0.0 { -eta } else { eta })
                            })
                            .sum();
                        assert!(
                            tail >= LINEAR_RESIDUAL_FLOOR,
                            "{label} q={q} b={b}: the tail case must run the linear form (T={tail:e})"
                        );
                        linear_tail_cases += 1;
                    }
                }
            }
        }
        eprintln!(
            "[2928 residual] max |α_linear − α_log|/max(1, |α|) = {worst_gap:.3e}, max gap/bound = {worst_bound_use:.3} over {} solves; {linear_tail_cases} tail solves on the linear form",
            2 * qs.len() * 7
        );
        assert!(linear_tail_cases >= 2 * 4 * 7, "every tail case ran the linear form");
    }

    /// The solve must not refuse anywhere a line search can wander: wide
    /// slopes, indices deep in either tail, on both laws.
    #[test]
    fn anchor_solves_across_the_whole_trial_range() {
        let (nodes, weights) = gauss_hermite_probabilists(65).expect("grid");
        let gaussian = owned(nodes, weights);
        let skewed = skewed_grid();
        let mut failures = Vec::new();
        for (label, grid) in [("gaussian", &gaussian), ("skewed", &skewed)] {
            for &q in &[-12.0, -8.0, -4.0, -1.0, 0.0, 1.0, 4.0, 8.0, 12.0] {
                for &b in &[-8.0, -3.0, -1.0, -0.1, 0.0, 0.1, 1.0, 3.0, 8.0, 20.0] {
                    if let Err(reason) = solve_anchor(q, b, grid.view()) {
                        failures.push(format!("{label} q={q} b={b}: {reason}"));
                    }
                }
            }
        }
        assert!(failures.is_empty(), "anchor solves refused:\n{}", failures.join("\n"));
    }

    /// Deep tails: the log-space residual keeps the solve finite and exact
    /// where the linear-space marginal has long underflowed.
    #[test]
    fn anchor_survives_the_deep_tails() {
        let grid = skewed_grid();
        for &q in &[-7.5, 7.5] {
            for &b in &[-1.5, 0.0, 1.5] {
                let alpha = solve_anchor(q, b, grid.view()).expect("anchor");
                assert!(alpha.is_finite());
                // On the survival side the anchored survival must equal Φ(−q)
                // relatively; on the complement side the anchored failure
                // probability must equal Φ(q) relatively.
                let survival = marginal(alpha, b, grid.view());
                if q >= 0.0 {
                    let target = normal_cdf(-q);
                    assert!(((survival - target) / target).abs() < 1e-9, "q={q} b={b}: {survival:e} vs {target:e}");
                } else {
                    let failure: f64 = grid
                        .nodes
                        .iter()
                        .zip(grid.weights.iter())
                        .map(|(&u, &w)| w * normal_cdf(alpha + b * u))
                        .sum();
                    let target = normal_cdf(q);
                    assert!(((failure - target) / target).abs() < 1e-9, "q={q} b={b}: {failure:e} vs {target:e}");
                }
            }
        }
    }
}
