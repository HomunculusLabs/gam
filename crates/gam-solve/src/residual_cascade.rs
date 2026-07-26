//! Multiresolution residual cascade for scattered 2-3D smooths at huge n
//! (compute-first primitive #3, #1032; siblings: the 1-D scan in
//! [`crate::spline_scan`], the 2-D grid in
//! [`gam_terms::grid_spline_2d`]).
//!
//! Model. In metric-scaled coordinates `z = diag(metric)·x` the smooth is
//!   `f(z) = P(z)'γ + Σ_l Σ_j c_{l,j} · φ((z − ξ_{l,j})/δ_l)`,
//! an unpenalized linear polynomial layer `P = {1, z_1, …, z_d}` at the root
//! plus, per level `l = 0..L`, compactly supported Wendland bumps
//! `φ(r) = (1−r)₊⁴(4r+1)` (positive definite and C² on ℝ³) of support radius
//! `δ_l = OVERLAP·h_l` planted on the NEW centers of a nested net with
//! covering radius `h_l = h₀·2^{−l}`. Coefficients are a-priori independent,
//! `c_{l,j} ~ N(0, τ²·4^{−l(s−d/2)})` — the standard multilevel frame whose
//! diagonal prior norm is equivalent to the Sobolev-`s` (semi)norm on
//! quasi-uniform nested nets (Narcowich–Ward inverse estimates + Le Gia–
//! Wendland multilevel stability; `d/2 < s ≤ (d+3)/2`, the native smoothness
//! of the Wendland-(3,1) bump). The assembled claim is certified in-test
//! against a dense kernel solve on small n (#904 style), not assumed.
//!
//! Nets. Each level's center set is a greedy hash-grid ε-net scanned in data
//! order, seeded with the previous level's net: covering radius ≤ h_l over
//! the data AND separation ≥ h_l — the same quasi-uniformity guarantees
//! farthest-point sampling gives, at O(n) per level (each point checks the
//! 3^d neighboring cells of one hash grid of cell size h_l). Nets are nested
//! (`Ξ_0 ⊂ Ξ_1 ⊂ …`); a center carries a bump only at its birth level.
//!
//! Fit. With `W = diag(w)`, `D = diag(0 on the polynomial layer, d_l =
//! 4^{l(s−d/2)} on level-l bumps)` and `λ = σ²/τ²`, the posterior mode solves
//! `(X'WX + λD)c = X'Wy`. `X` is sparse — a row touches the O(1) bumps per
//! level whose supports cover it, O(qL) nonzeros — and is held in CSR. For
//! moderate column counts (`m ≤ DENSE_GRAM_MAX`) the normal equations are
//! solved by dense Cholesky with the EXACT log-determinant (same route as the
//! grid sibling); beyond that the solve is preconditioned CG with the two-level
//! additive-Schwarz coarse-space preconditioner `P = blockdiag(A_CC,
//! diag(A_FF))`. The multilevel Wendland frame is redundant across scales — a
//! coarse bump and the fine bumps in its support are strongly correlated — so
//! the data-fit Gram `X'WX` couples levels and a pure-diagonal preconditioner
//! leaves a conditioning that GROWS with the number of data-identified levels
//! (hence with n). The coarse space `C` (polynomial layer + the data-dominated
//! coarsest levels, see `coarse_space_cols`) is solved EXACTLY by a small dense
//! Cholesky and the penalty-dominated fine tail `F` — where `A_ll ≈ λ d_l I` is
//! already uniformly conditioned — by its Jacobi diagonal. That deflation is
//! what makes `P^{−1/2}(X'WX+λD)P^{−1/2}` uniformly conditioned, so the CG
//! iteration count is genuinely n-independent (the in-test gate asserts an
//! ADDITIVE bound across a 4× n jump, not a multiplicative one). Every CG solve
//! reports its relative residual `‖b − Ac‖/‖b‖`: a computable backward-error
//! certificate (`c` solves a system perturbed by no more than that fraction)
//! inherited by every linear functional of the solution.
//!
//! REML. λ maximizes the profiled-σ² restricted criterion
//!   `ℓ_R(λ) = −½[ log|X'WX+λD| − log|λD|₊ + (n−d−1)·log σ̂²(λ) ] + const`,
//! `log|λD|₊ = r·logλ + Σ_j log d_j` over the `r` penalized columns and
//! `σ̂² = (y'Wy − c'X'Wy)/(n−d−1)` — the same shape as the siblings, with the
//! penalty-logdet constant kept so criteria are comparable across cascade
//! depths. Eliminating the polynomial null block once gives the
//! penalty-whitened Schur complement `B`, for which the normalized determinant
//! is `log|G₀₀| + log|I+B/λ|`. Its spectrum is exact on the dense route and
//! represented by one fixed-probe Lanczos quadrature on the iterative route.
//! Thus the score, gradient, and curvature are analytic functions of log λ
//! with the SAME spectral nodes at every trial. Rigorous derivative enclosures
//! isolate every stationary interval before safeguarded root refinement; both
//! bounded-domain endpoints are compared exactly, with no basin-selecting
//! lattice.
//!
//! The same elimination puts the profiled RESIDUAL in the same form, and this is
//! what makes the criterion solve-free at every λ on BOTH routes. With
//! `β = D^{−1/2}(b₁ − G₁₀G₀₀^{−1}b₀)` and `S_k(λ) = β'(B+λI)^{−k}β`, the residual
//! is `R = anchor − S₁` and its three `log λ` derivatives are built from
//! `S₂, S₃, S₄`; `anchor = y'Wy − b₀'G₀₀^{−1}b₀` is the part no λ can move. On
//! the dense route the eigenbasis projects β directly. Past the cap, `S_k(λ)` is
//! `∫(θ+λ)^{−k} dμ(θ)` for the measure β induces on `spec(B)`, so ONE Lanczos run
//! seeded with β (rather than with a Rademacher probe) returns the Jacobi matrix
//! of its Golub–Meurant Gauss rule — the same `(node, weight)` shape the dense
//! route stores, so both routes then evaluate one expression. That rule is
//! admitted only when it has earned it: either the Krylov space has consumed
//! `rank(B) ≤ n − nullity` and is therefore invariant (the rule is then exact for
//! every kernel), or two NESTED rules — free, since `T_m`'s leading block is
//! `T_j` — agree to `√ε` over the whole domain, which for a completely monotone
//! kernel bounds the coarser rule's error exactly. Uncertified, the route falls
//! back to two PCG solves per λ and says so in the fit certificate: at the bottom
//! of the domain `X'WX + λD` is numerically singular and no solve can be
//! certified there, which is why the quadrature exists (#2503).
//!
//! Refinement certificate. After fitting L levels, the candidate level L+1
//! is constructed (O(n)) and the EXACT objective decrease available from
//! adding it is bounded: for the penalized objective `F(c) = ‖√W(y−Xc)‖² +
//! λc'Dc`, appending columns `X₂` with penalty `λd_{L+1}I` decreases the
//! minimum by `g'S⁻¹g`, `g = X₂'W r̂`, `S` the Schur complement; since
//! `A₁₁ ⪰ X₁'WX₁` and `X₂'W^{1/2}·proj·W^{1/2}X₂ ⪯ X₂'WX₂`, `S ⪰ λd_{L+1}I`,
//! so the decrease is at most `‖X₂'W r̂‖²/(λ·d_{L+1})` — a computable
//! discretization certificate. The cascade refines (adds the level, refits,
//! re-selects λ) until that bound drops below `REFINE_TOL` of the penalized
//! residual, the net stops producing new centers (every point is a center),
//! or the level/center caps are reached: certified-or-fallback, the same
//! discipline as the radial-profile GL ladder.
//!
//! Posterior. Coefficient covariance is `σ²(X'WX+λD)^{−1}`; pointwise
//! prediction variance routes the basis row through one (certified) solve.
//! Exact posterior samples come from perturb-and-solve: `c_s = A^{−1}(X'Wy +
//! σ(X'W^{1/2}z₁ + √λ D^{1/2}z₂))` with iid standard-normal `z₁, z₂` has
//! mean `ĉ` and covariance exactly `σ²A^{−1}` (deterministically seeded; one
//! certified solve per sample).
//!
//! Payoff. Build O(n·(L + 3^d)), fit O(nnz · iters) per λ trial with
//! n-independent iters — O(n log n) end to end, against the dense n×k kernel
//! Gram + O(k³) per trial that duchon/matern pay today. Gap behavior is
//! mechanical: levels wider than a gap keep support across it (polynomial +
//! coarse bumps bridge), finer levels have no data and revert to their prior
//! variance, so the posterior mean bridges instead of sagging while the
//! variance grows into the gap.

use std::collections::HashMap;
use std::sync::Arc;

use faer::Side;
use gam_linalg::faer_ndarray::FaerEigh;
use gam_math::score_opt::{
    AffineRemlProfile, ClosedInterval, DerivativeEnclosure, ScoreJet, ScoreSample,
    maximize_score_1d,
};
use gam_terms::grid_spline_2d::{chol_solve, cholesky_logdet};
use ndarray::Array2;

/// Bump support radius as a multiple of the level's covering radius:
/// `δ_l = OVERLAP·h_l`. Separation ≥ h_l caps the bumps covering a point at
/// a packing constant per level (O(q) row nonzeros per level).
const OVERLAP: f64 = 2.0;
/// Root covering radius as a fraction of the largest scaled axis range.
const H0_FRACTION: f64 = 0.5;
/// Levels in the initial cascade before refinement certificates run.
const INITIAL_LEVELS: usize = 3;
/// Hard cap on cascade depth (h shrinks 2^16-fold below the root).
const MAX_LEVELS: usize = 16;
/// Hard cap on total centers across all levels.
const MAX_CENTERS: usize = 200_000;
/// Refinement stops when the exact next-level gain bound falls below this
/// fraction of the penalized residual.
const REFINE_TOL: f64 = 1e-3;

/// Column count up to which the normal equations go through dense Cholesky
/// (exact logdet, no iteration); above it, PCG + SLQ. 1536² doubles ≈ 18 MB.
const DENSE_GRAM_MAX: usize = 1536;

/// PCG convergence: relative residual ‖b − Ac‖/‖b‖ (the backward-error
/// certificate) demanded of every solve, and the iteration cap past which
/// the solve is an error rather than a silent approximation. The certification
/// suite gates the iterative route at 1e-9; asking for more burns matvecs
/// without strengthening any downstream certificate.
const CG_RTOL: f64 = 1e-9;
const CG_MAX_ITERS: usize = 4000;

/// Coarse-space additive-Schwarz preconditioner controls (issue #1032: the
/// "BPX/level-diagonal preconditioned CG, n-independent iters" spec).
///
/// The multilevel Wendland frame is redundant across scales — a coarse bump and
/// the fine bumps inside its support are strongly correlated — so the data-fit
/// Gram `X'WX` couples levels and a pure-diagonal (Jacobi) preconditioner leaves
/// a conditioning that grows with the number of *data-identified* levels, hence
/// with `n` (more rows ⇒ finer levels carry data ⇒ another collinear coarse
/// scale the diagonal can't decouple). The cure is the textbook two-level
/// additive Schwarz coarse space: solve the coarse block — the polynomial layer
/// plus every level the penalty has NOT yet made diagonally dominant — EXACTLY,
/// and precondition the remaining penalty-dominated fine levels (where
/// `A_ll ≈ λ d_l I` is already uniformly conditioned) by their Jacobi diagonal.
///
/// A level is "data-dominated" while `λ d_l < COARSE_DOMINANCE · median diag
/// (X'WX) over the level`. Because columns are laid out poly, level-0, level-1,
/// … and `d_l` increases while the per-level data weight decreases, the
/// data-dominated levels are exactly the coarsest prefix `[0, ncoarse)`, so the
/// coarse space is a contiguous column prefix and the cut is a single scan. The
/// crossover level grows only as `½ log₄(n/λ)` — `ncoarse = O(√(n/λ))` columns —
/// so the exact coarse factorization stays small against the sparse matvecs at
/// every n the primitive serves. [`COARSE_SPACE_MAX`] caps it as a safety valve
/// (past the cap the finer data-dominated levels fall back to Jacobi and the
/// iteration count rises, but the CG residual certificate still guarantees the
/// solve); [`MIN_COARSE_LEVELS`] always deflates the two coarsest scales, which
/// are near-collinear with the polynomial layer at every λ.
const COARSE_DOMINANCE: f64 = 4.0;
/// Safety ceiling on the exact-coarse column count. It must NOT bind at the n
/// the primitive serves: the n-independent iteration count rests on the coarse
/// block containing the WHOLE data-dominated prefix (`O(√(n/λ))` columns), so a
/// cap that truncates that prefix is exactly what makes the iteration count
/// climb with n (a finer data-dominated level demoted to Jacobi cannot be
/// decoupled from the coarse scales it is collinear with). At the n-scales the
/// iterative route engages (tens of thousands of rows → a ≈1.4k-column
/// prefix) this is non-binding headroom; it only triggers in the genuinely
/// degenerate case the quasi-uniformity guard is meant to catch first. The
/// realized coarse factorization runs at the actual prefix length, not the cap,
/// so the ceiling costs nothing until it fires.
const COARSE_SPACE_MAX: usize = 4096;
const MIN_COARSE_LEVELS: usize = 2;

/// Quasi-uniformity guard (issue #1032, caveat 2). The BPX n-independent CG
/// iteration bound rests on the nested ε-nets being quasi-uniform *in the
/// metric-scaled coordinates `z = diag(metric)·x` the bumps live in*. The
/// greedy net guarantees covering ≤ h and separation ≥ h in `z` by
/// construction, so the only way the BPX norm-equivalence constant blows up is
/// when the metric is so anisotropic that the metric-scaled point cloud is
/// effectively degenerate along a direction — the data collapses onto a lower
/// dimension in `z`, the root covering radius `h₀ = ½·max_a range_a` swamps the
/// collapsed axis, the level-`l` bumps overlap pathologically, and the
/// preconditioner constant (hence the iteration count) grows without an
/// n-independent bound. The realized symptom is `solve_iters` climbing toward
/// [`CG_MAX_ITERS`]; this guard detects the *cause* up front from the
/// metric-scaled per-axis spread so the auto-route can fall back to the dense
/// kernel BEFORE paying an unbounded iterative solve, rather than discovering
/// the blow-up only after `CG_MAX_ITERS` work.
///
/// Condition measure: the ratio of the largest to smallest metric-scaled
/// per-axis standard deviation (a scale-free aspect ratio of the scaled
/// cloud). Past this threshold the net is no longer quasi-uniform in every
/// direction and the BPX bound is not trustworthy. Derived, not a knob: a
/// `10³` aspect ratio means the collapsed axis carries <0.1% of the dominant
/// axis's variation, at which point its bumps span the whole cloud and the
/// multilevel hierarchy degenerates to a single ill-conditioned level.
const QUASI_UNIFORMITY_MAX_ASPECT: f64 = 1.0e3;

/// SLQ controls: fixed Rademacher probes (shared across λ trials) and the
/// Lanczos depth per probe (full reorthogonalization; early exit on
/// breakdown).
const SLQ_PROBES: usize = 24;
const SLQ_LANCZOS_STEPS: usize = 48;
/// Live bytes the profiled-residual quadrature's full-reorthogonalization basis
/// may occupy past the dense cap (see [`Core::residual_quadrature_budget`]). The
/// basis is `steps x rank` doubles, and it is the only quantity in that run that
/// grows with both; the matvecs and the tridiagonal eigensolve are negligible
/// beside it.
///
/// The number is chosen so the run can REACH its Krylov ceiling, because that is
/// where the rule becomes exact and stopping short of it buys nothing at all: a
/// budget of `0.9 * ceiling` pays 90% of the work and then falls back to the solve.
/// The binding requirement is therefore `ceiling * rank * 8` bytes, measured on the
/// designs the #2503 integration fixtures actually build:
///
/// ```text
///   n =  800, level 6   ceiling  797   rank  7387    47 MB
///   n = 1200, level 8   ceiling 1197   rank 30879   296 MB
///   n = 2500, level 7   ceiling 2497   rank 16565   331 MB
///   n = 6000, level 7   ceiling 5997   rank  8432   404 MB
/// ```
///
/// 512 MiB covers all of them and still bounds the run on the
/// hundred-thousand-column designs `MAX_CENTERS` permits, where the ceiling is
/// unreachable at any budget worth paying and the route keeps the solve.
const RESIDUAL_QUADRATURE_BASIS_BYTES: usize = 512 << 20;

/// Deterministic seed for the SLQ probes and posterior samples.
const RNG_SEED: u64 = 0x1032_CA5C_ADE0_5EED;

/// Floor for eigenvalues/pivots before the system is declared singular.
const EIG_FLOOR: f64 = 1e-300;

// ───────────────────────────── deterministic RNG ────────────────────────────

/// SplitMix64: tiny, deterministic, full-period stream generator.
struct SplitMix64(u64);

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        SplitMix64(seed)
    }

    fn next_u64(&mut self) -> u64 {
        gam_linalg::utils::splitmix64(&mut self.0)
    }

    /// Uniform in (0, 1): 53-bit mantissa, shifted off zero.
    fn next_unit(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) / 9_007_199_254_740_992.0
    }

    /// Standard normal via Box–Muller.
    fn next_normal(&mut self) -> f64 {
        let u1 = self.next_unit();
        let u2 = self.next_unit();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }

    /// Rademacher ±1.
    fn next_sign(&mut self) -> f64 {
        if self.next_u64() & 1 == 0 { 1.0 } else { -1.0 }
    }
}

// ─────────────────────────────── hash grids ─────────────────────────────────

/// Integer cell of a point at a given cell width (coordinates are already
/// metric-scaled and shifted to be ≥ 0, so indices are small and exact).
#[inline]
fn cell_of(z: &[f64; 3], dim: usize, width: f64) -> (i32, i32, i32) {
    let mut c = [0_i32; 3];
    for a in 0..dim {
        c[a] = (z[a] / width).floor() as i32;
    }
    (c[0], c[1], c[2])
}

/// Hash grid over a point set: cell → indices. Lookup scans the 3^d
/// neighborhood, which covers every point within one cell width.
struct HashGrid {
    width: f64,
    dim: usize,
    cells: HashMap<(i32, i32, i32), Vec<u32>>,
}

impl HashGrid {
    fn new(width: f64, dim: usize) -> Self {
        HashGrid {
            width,
            dim,
            cells: HashMap::new(),
        }
    }

    fn insert(&mut self, idx: u32, z: &[f64; 3]) {
        let key = cell_of(z, self.dim, self.width);
        self.cells.entry(key).or_default().push(idx);
    }

    /// Visit every stored index in the 3^d cells around `z` (deterministic
    /// order: lexicographic cells, insertion order within a cell).
    fn for_neighbors(&self, z: &[f64; 3], mut visit: impl FnMut(u32)) {
        let (c0, c1, c2) = cell_of(z, self.dim, self.width);
        let d2 = if self.dim > 2 { 1 } else { 0 };
        let d1 = if self.dim > 1 { 1 } else { 0 };
        for i0 in -1..=1_i32 {
            for i1 in -d1..=d1 {
                for i2 in -d2..=d2 {
                    if let Some(bucket) = self.cells.get(&(c0 + i0, c1 + i1, c2 + i2)) {
                        for &idx in bucket {
                            visit(idx);
                        }
                    }
                }
            }
        }
    }
}

#[inline]
fn dist2(a: &[f64; 3], b: &[f64; 3], dim: usize) -> f64 {
    let mut s = 0.0;
    for k in 0..dim {
        let d = a[k] - b[k];
        s += d * d;
    }
    s
}

/// Wendland-(3,1) bump `(1−r)₊⁴(4r+1)`: positive definite on ℝ^d, d ≤ 3,
/// C², native space H^{(d+3)/2}.
#[inline]
fn wendland(r: f64) -> f64 {
    if r >= 1.0 {
        return 0.0;
    }
    let v = 1.0 - r;
    let v2 = v * v;
    v2 * v2 * (4.0 * r + 1.0)
}

// ───────────────────────────── design assembly ──────────────────────────────

/// One resolution level: its NEW centers (scaled coordinates), covering
/// radius, support radius, prior precision weight, and a lookup grid of cell
/// width δ_l over those centers.
struct Level {
    h: f64,
    delta: f64,
    /// Prior precision weight `d_l = 4^{l(s−d/2)}` (prior variance τ²/d_l).
    weight: f64,
    centers: Vec<[f64; 3]>,
    /// First flat column index of this level's coefficients.
    col_offset: usize,
    grid: HashGrid,
}

/// Immutable fitted-design core shared between the design handle and fits.
struct Core {
    dim: usize,
    metric: [f64; 3],
    /// Lower corner / range of the scaled bounding box (polynomial layer
    /// coordinates are `2(z − lo)/range − 1` for conditioning).
    z_lo: [f64; 3],
    z_range: [f64; 3],
    sobolev_s: f64,
    levels: Vec<Level>,
    /// Full nested net Ξ_L (scaled coords), retained so the candidate level
    /// L+1 can extend it without re-deriving coarser levels.
    net: Vec<[f64; 3]>,
    /// Total columns: `dim + 1` polynomial + all level centers.
    m: usize,
    /// CSR design rows (column-sorted within a row).
    row_ptr: Vec<usize>,
    col_idx: Vec<u32>,
    vals: Vec<f64>,
    /// Inputs retained for matvecs, residuals, and refinement.
    w: Vec<f64>,
    y: Vec<f64>,
    /// Scaled data coordinates (shifted to the box corner).
    z: Vec<[f64; 3]>,
    /// `X'Wy`, `y'Wy`, `diag(X'WX)`.
    rhs: Vec<f64>,
    ytwy: f64,
    gram_diag: Vec<f64>,
    /// Per-column prior precision weight (0 on the polynomial layer).
    pen_diag: Vec<f64>,
    /// `Σ_j log d_j` over penalized columns (the λ-free part of log|λD|₊,
    /// kept so REML criteria compare across cascade depths).
    pen_logdet_const: f64,
    /// Dense upper-triangular `X'WX` when `m ≤ DENSE_GRAM_MAX` (row-major
    /// m×m, lower mirror filled at solve time); None on the iterative route.
    dense_gram: Option<Vec<f64>>,
    /// Predict-only factored precision: the lower Cholesky factor `L` of
    /// `A = X'WX + λD` at the FIT's λ, populated only on a core rebuilt from a
    /// persisted [`ResidualCascadeState`] (where the training CSR is dropped).
    /// When present, `solve_coeff` replays the posterior-variance solve through
    /// this factor instead of the absent training design; `None` on a
    /// training-built core, which solves through `dense_gram`/PCG as usual.
    predict_chol: Option<Vec<f64>>,
}

/// Solver route a fit took for its log-determinant.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LogdetMethod {
    /// Dense Cholesky: exact.
    DenseExact,
    /// Diagonal control variate + stochastic Lanczos quadrature on fixed
    /// deterministic probes.
    Slq,
}

/// Route the profiled residual `R(λ)` and its three `log λ` derivative moments
/// `S₂, S₃, S₄` took during REML selection — reported WITH the quantity that
/// decided it, because two of the three arms carry a convergence certificate and
/// the third is the refusal to issue one, and which a fit received is not
/// otherwise readable from the outside.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ResidualMomentMethod {
    /// Read off the dense Schur eigenbasis under the sizing cap: exact, with no
    /// linear solve at any λ.
    DenseExact,
    /// Golub–Meurant quadrature seeded with the whitened right-hand side,
    /// admitted because it CONVERGED: either the Krylov space closed (the rule is
    /// then exact) or two nested rules agree to the search's own resolution over
    /// the whole λ domain. No linear solve at any λ either way.
    ConvergedQuadrature {
        /// Nodes in the admitted rule.
        steps: usize,
        /// Nodes in the coarser rule it was charged against (`steps / 2`), or `0`
        /// when the Krylov space closed and no comparison was needed.
        coarse_steps: usize,
        /// Penalized Schur rank, the hard ceiling on `steps`.
        rank: usize,
        /// Step budget the growth loop was allowed, before that ceiling.
        budget: usize,
        /// Conservative geometric extrapolation of the rule's REMAINING relative
        /// error, over `S₁..S₄` and the whole λ domain, from three nested rules.
        /// `0` when the Krylov space closed and the rule is exact outright.
        tail_estimate: f64,
        /// The resolution `tail_estimate` was charged against.
        target: f64,
        /// `‖r_m‖ / maxᵢ|αᵢ|`: the Krylov residual against the operator scale.
        /// At roundoff the space closed and the rule is exact outright.
        relative_tail: f64,
        /// `|Σⱼ wⱼ / ‖β‖² − 1|`: a Gauss rule's weights sum to its measure's
        /// mass, so this is a free end-to-end check on the Jacobi pipeline.
        mass_defect: f64,
        /// Fraction of `‖β‖²` dropped as null-space mass. `β ⊥ null(B)` holds
        /// EXACTLY, so this is roundoff, and its size is the evidence for that.
        dropped_mass_fraction: f64,
    },
    /// No rule converged inside the budget, so `S₁..S₄` are re-derived from two
    /// solves of `A = X'WX + λD` at every λ the search visits — which is where
    /// the λ at the bottom of the domain cannot be solved at all (#2503).
    Solved {
        steps: usize,
        rank: usize,
        budget: usize,
        tail_estimate: f64,
        target: f64,
        relative_tail: f64,
    },
}

/// Computable certificates attached to a fit.
#[derive(Clone, Copy, Debug)]
pub struct CascadeCertificate {
    /// Backward error of the coefficient solve: ‖b − Aĉ‖/‖b‖ (0 on the dense
    /// route).
    pub solve_rel_residual: f64,
    /// CG iterations of the coefficient solve (0 on the dense route); the
    /// n-independence gate watches this.
    pub solve_iters: usize,
    /// Route the log-determinant took.
    pub logdet_method: LogdetMethod,
    /// Route the profiled residual's `λ` moments took, on a fit whose `λ` was
    /// SELECTED by the REML criterion. `None` on a fixed-λ fit, which evaluates
    /// no criterion and therefore takes no such route.
    pub residual_moments: Option<ResidualMomentMethod>,
}

/// Discretization certificate of the refinement loop: the exact upper bound
/// on the penalized-objective decrease available from one more level.
#[derive(Clone, Copy, Debug)]
pub struct RefinementCertificate {
    /// `‖X_{L+1}'W r̂‖² / (λ·d_{L+1})` at the accepted fit.
    pub next_level_gain_bound: f64,
    /// The absolute tolerance it was compared against (`REFINE_TOL·rss_pen`).
    pub tolerance: f64,
}

/// A structural limit that prevented the cascade from assessing or adding the
/// next resolution level. These are never convergence certificates: if the
/// requested gain tolerance has not passed, they produce
/// [`ResidualCascadeError::Underresolved`] instead of a fit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RefinementObstruction {
    /// The representation reached its supported maximum number of levels.
    LevelCapacity {
        levels: usize,
        maximum_levels: usize,
    },
    /// Extending the nested net would exceed its supported center capacity.
    CenterCapacity {
        centers: usize,
        maximum_centers: usize,
    },
}

impl std::fmt::Display for RefinementObstruction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            Self::LevelCapacity {
                levels,
                maximum_levels,
            } => write!(
                f,
                "level capacity reached ({levels} of {maximum_levels} levels)"
            ),
            Self::CenterCapacity {
                centers,
                maximum_centers,
            } => write!(
                f,
                "center capacity exceeded ({centers} centers for capacity {maximum_centers})"
            ),
        }
    }
}

/// Result of assessing the candidate level immediately finer than a fitted
/// design. Empty-net exhaustion is distinct from representation capacity:
/// only the former proves that the remaining gain is exactly zero.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum NextLevelAssessment {
    /// The nested net produced no new centers, so the next-level gain is zero.
    EmptyNet,
    /// The complete candidate level was assessed and has this gain bound.
    GainBound(f64),
    /// A representation limit was reached. `gain_bound` is the computed bound
    /// when the candidate could be assessed (level capacity), and positive
    /// infinity when center capacity prevented a complete assessment.
    CapacityExceeded {
        obstruction: RefinementObstruction,
        gain_bound: f64,
    },
}

/// Multiresolution residual-cascade design: nested nets, sparse design,
/// diagonal multilevel prior — everything needed to evaluate the REML
/// criterion and solve at any λ.
pub struct ResidualCascadeDesign {
    core: Arc<Core>,
}

/// Fitted cascade with factored-by-solve posterior access.
pub struct ResidualCascadeFit {
    core: Arc<Core>,
    /// Dense-route prediction factor at the fit's λ. When present, pointwise
    /// variance uses this one Cholesky factor instead of refactoring the same
    /// precision matrix for every prediction point.
    predict_chol: Option<Vec<f64>>,
    /// Coefficients: `dim+1` polynomial entries, then level blocks.
    pub coeff: Vec<f64>,
    /// Selected (or supplied) log smoothing parameter `log λ = log σ²/τ²`.
    log_lambda: f64,
    /// Profiled (or supplied) observation variance σ².
    pub sigma2: f64,
    /// Restricted log-likelihood at the fit, up to λ- and data-independent
    /// additive constants (exact REML differences across λ on the dense
    /// route; SLQ-estimated on the iterative route).
    pub restricted_loglik: f64,
    /// Penalized residual quadratic `y'Wy − c'X'Wy`.
    pub rss_pen: f64,
    /// Solve/logdet certificates.
    pub certificate: CascadeCertificate,
    /// Present when the fit came from the refinement loop.
    pub refinement: Option<RefinementCertificate>,
}

/// Opaque work checkpoint carried by an underresolved cascade result.
///
/// The current finite-resolution iterate is deliberately private: callers can
/// inspect its numerical evidence, but cannot turn an uncertified iterate into
/// a [`ResidualCascadeFit`]. The retained design and coefficients allow a
/// future refinement backend to resume the work without minting a partial fit.
pub struct ResidualCascadeCheckpoint {
    iterate: ResidualCascadeFit,
}

impl ResidualCascadeCheckpoint {
    fn new(iterate: ResidualCascadeFit) -> Self {
        Self { iterate }
    }

    /// Number of levels already fitted in this checkpoint.
    pub fn num_levels(&self) -> usize {
        self.iterate.num_levels()
    }

    /// Number of centers already fitted in this checkpoint.
    pub fn num_centers(&self) -> usize {
        self.iterate.num_centers()
    }

    /// REML-selected log smoothing parameter of the retained iterate.
    pub fn log_lambda(&self) -> f64 {
        self.iterate.log_lambda
    }

    /// Penalized residual used to scale the requested refinement tolerance.
    pub fn rss_pen(&self) -> f64 {
        self.iterate.rss_pen
    }

    /// Linear-solve evidence attached to the retained iterate.
    pub fn certificate(&self) -> CascadeCertificate {
        self.iterate.certificate
    }
}

impl std::fmt::Debug for ResidualCascadeCheckpoint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResidualCascadeCheckpoint")
            .field("num_levels", &self.num_levels())
            .field("num_centers", &self.num_centers())
            .field("log_lambda", &self.log_lambda())
            .field("rss_pen", &self.rss_pen())
            .field("certificate", &self.certificate())
            .finish_non_exhaustive()
    }
}

/// Typed failure of the magic-default cascade fit.
#[derive(Debug)]
pub enum ResidualCascadeError {
    /// Invalid input or a numerical failure in design construction/optimization.
    Computation(String),
    /// Refinement could not meet its requested tolerance before a structural
    /// capacity was reached. The checkpoint preserves all completed work while
    /// remaining unusable as a public fit.
    Underresolved {
        checkpoint: ResidualCascadeCheckpoint,
        gain_bound: f64,
        requested_tolerance: f64,
        obstruction: RefinementObstruction,
    },
}

impl From<String> for ResidualCascadeError {
    fn from(reason: String) -> Self {
        Self::Computation(reason)
    }
}

impl std::fmt::Display for ResidualCascadeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Computation(reason) => f.write_str(reason),
            Self::Underresolved {
                checkpoint,
                gain_bound,
                requested_tolerance,
                obstruction,
            } => write!(
                f,
                "residual cascade underresolved after {} levels: next-level gain bound \
                 {gain_bound:.6e} exceeds requested tolerance {requested_tolerance:.6e}; \
                 {obstruction}",
                checkpoint.num_levels()
            ),
        }
    }
}

impl std::error::Error for ResidualCascadeError {}

/// One resolution level's geometry in a persisted snapshot: the data needed to
/// rebuild a [`Level`] (its lookup grid, bumps, and column block) without the
/// training rows. Centers are flattened `dim`-major (`dim` floats per center).
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct LevelState {
    pub h: f64,
    pub delta: f64,
    pub weight: f64,
    pub col_offset: u64,
    /// `dim·n_centers` scaled-coordinate floats, center-major.
    pub centers: Vec<f64>,
}

/// Serializable snapshot of a [`ResidualCascadeFit`] (#1032 persistence
/// prerequisite). Holds everything `predict` needs and NOTHING about the
/// training rows:
/// - MEAN: the nested geometry (`dim`/`metric`/box/`sobolev_s` + per-level
///   centers/δ/weights/col-offsets) and the root polynomial layer are all that
///   `basis_row_scaled`·`coeff` reads;
/// - VARIANCE: the factored precision `predict_chol` — the lower Cholesky factor
///   `L` of `A = X'WX + λD` at the fit's λ — which the posterior-variance solve
///   `x'A⁻¹x` replays against (the training design that originally assembled `A`
///   is dropped).
///
/// `from_state` rebuilds a predict-capable fit whose `Core` carries empty
/// training CSR and `predict_chol = Some(L)`; `solve_coeff` then routes the
/// variance solve through `L`. The reconstructed fit cannot be re-fit or
/// resampled (it has no rows), only predicted from — exactly the persistence
/// contract.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ResidualCascadeState {
    pub dim: u64,
    /// Per-axis metric scaling (length 3; trailing entries are 1 for `dim < 3`).
    pub metric: [f64; 3],
    pub z_lo: [f64; 3],
    pub z_range: [f64; 3],
    pub sobolev_s: f64,
    pub levels: Vec<LevelState>,
    /// Total column count `dim + 1 + Σ centers`.
    pub m: u64,
    /// `Σ_j log d_j` over penalized columns (kept so restored REML scalars stay
    /// comparable across cascade depths).
    pub pen_logdet_const: f64,
    /// Posterior-mode coefficients (length `m`).
    pub coeff: Vec<f64>,
    pub log_lambda: f64,
    pub sigma2: f64,
    pub restricted_loglik: f64,
    pub rss_pen: f64,
    /// Lower Cholesky factor `L` of `A = X'WX + λD` at the fit's λ, `m × m`
    /// row-major — the factored precision the variance solve replays through.
    pub predict_chol: Vec<f64>,
}

/// Forward substitution `L y = b` (lower factor, row-major) into `out`.
fn forward_sub_into(l: &[f64], p: usize, b: &[f64], out: &mut [f64]) {
    for i in 0..p {
        let mut s = b[i];
        for t in 0..i {
            s -= l[i * p + t] * out[t];
        }
        out[i] = s / l[i * p + i];
    }
}

/// Back substitution `Lᵀ z = y` (lower factor, row-major) into `out`.
fn back_sub_into(l: &[f64], p: usize, y: &[f64], out: &mut [f64]) {
    for i in (0..p).rev() {
        let mut s = y[i];
        for t in i + 1..p {
            s -= l[t * p + i] * out[t];
        }
        out[i] = s / l[i * p + i];
    }
}

/// Coarse-space additive-Schwarz preconditioner for the iterative route
/// (issue #1032). `A = X'WX + λD` is preconditioned by the symmetric positive
/// definite block-diagonal `P = blockdiag(A_CC, diag(A_FF))`, where the coarse
/// index set `C = [0, ncoarse)` is the polynomial layer plus the data-dominated
/// (coarsest) levels and `F` the penalty-dominated fine tail — see the
/// [`COARSE_DOMINANCE`]/[`COARSE_SPACE_MAX`] docs for why this delivers
/// n-independent CG iteration counts where the pure-Jacobi diagonal does not.
///
/// `solve` applies `P⁻¹` (exact coarse Cholesky solve ⊕ fine Jacobi). For the
/// SLQ log-determinant the symmetric factor `R = blockdiag(L_CC, diag√A_FF)`
/// with `P = R Rᵀ` is exposed through `apply_r_inv`/`apply_r_inv_t`, and
/// `log|P| = log|A_CC| + Σ_F log A_jj`.
struct Preconditioner {
    /// First fine column; coarse block is the principal `[0, ncoarse)` submatrix.
    ncoarse: usize,
    /// Lower Cholesky factor of the coarse block `A_CC` (`ncoarse × ncoarse`).
    coarse_chol: Vec<f64>,
    /// `log|A_CC|` (exact).
    coarse_logdet: f64,
    /// `1/A_jj` on the fine columns `[ncoarse, m)`.
    inv_fine: Vec<f64>,
    /// `1/√A_jj` on the fine columns (the `R⁻¹`/`R⁻ᵀ` fine scaling).
    inv_sqrt_fine: Vec<f64>,
    /// `Σ_F log A_jj` (the fine part of `log|P|`).
    fine_logdet: f64,
}

impl Preconditioner {
    /// `out = P⁻¹ r`: exact coarse solve on `[0, ncoarse)`, Jacobi on the tail.
    fn solve(&self, r: &[f64], out: &mut [f64]) {
        let nc = self.ncoarse;
        let zc = chol_solve(&self.coarse_chol, nc, &r[..nc]);
        out[..nc].copy_from_slice(&zc);
        for (k, o) in out[nc..].iter_mut().enumerate() {
            *o = r[nc + k] * self.inv_fine[k];
        }
    }

    /// `out = R⁻ᵀ v` (coarse: `L_CCᵀ` back-solve; fine: `/√A_jj`).
    fn apply_r_inv_t(&self, v: &[f64], out: &mut [f64]) {
        let nc = self.ncoarse;
        back_sub_into(&self.coarse_chol, nc, &v[..nc], &mut out[..nc]);
        for (k, o) in out[nc..].iter_mut().enumerate() {
            *o = v[nc + k] * self.inv_sqrt_fine[k];
        }
    }

    /// `out = R⁻¹ v` (coarse: `L_CC` forward-solve; fine: `/√A_jj`).
    fn apply_r_inv(&self, v: &[f64], out: &mut [f64]) {
        let nc = self.ncoarse;
        forward_sub_into(&self.coarse_chol, nc, &v[..nc], &mut out[..nc]);
        for (k, o) in out[nc..].iter_mut().enumerate() {
            *o = v[nc + k] * self.inv_sqrt_fine[k];
        }
    }

    /// `log|P| = log|A_CC| + Σ_F log A_jj`.
    fn logdet(&self) -> f64 {
        self.coarse_logdet + self.fine_logdet
    }
}

/// One positive-semidefinite eigenmode of the penalty-whitened Schur
/// complement. `weight == 1` on the dense exact route; on the large route it
/// is the fixed-probe Lanczos quadrature weight. The weights sum to the
/// penalized rank, so constants have the same null-recovery limit on both
/// routes.
#[derive(Clone, Copy)]
struct CascadeSpectralMode {
    eigenvalue: f64,
    weight: f64,
}

/// One Lanczos run's Jacobi matrix on the penalty-whitened Schur complement
/// `B`, plus everything a Golub–Meurant quadrature rule needs from it.
///
/// See [`Core::schur_lanczos`]. `alpha` and `beta` are `T_m`'s diagonal and
/// off-diagonal, with `beta.len() == alpha.len() - 1` whenever `alpha` is
/// non-empty, which is the shape [`symmetric_tridiagonal_eigen`] reads.
struct SchurLanczos {
    /// `alpha_1..alpha_m` — the diagonal of `T_m`.
    alpha: Vec<f64>,
    /// `beta_1..beta_{m-1}` — the off-diagonal INSIDE `T_m`.
    beta: Vec<f64>,
    /// `||r_m||`, the `(m+1, m)` entry of the untruncated Jacobi matrix. It is
    /// the residual of the Krylov approximation, so `tail == 0` means
    /// `K_m(B, start)` is `B`-invariant and the Gauss rule is EXACT.
    tail: f64,
    /// `max_i |alpha_i|` — a Rayleigh-quotient lower bound on `||B||` over the
    /// Krylov space, and the scale the invariance certificate is stated in.
    spectral_scale: f64,
    /// `||start||^2`, the total mass of the quadrature measure: `Sum_j w_j`.
    start_norm_sq: f64,
    /// `tail` sits at the run's own roundoff floor, or the run consumed the full
    /// penalized rank. Either way `K_m(B, start)` is (numerically) invariant and
    /// the Gauss rule reproduces `start' g(B) start` for EVERY analytic `g` and
    /// every `lambda` — not approximately, but to the arithmetic's own floor.
    invariant: bool,
}

/// Lambda-independent spectral representation of the profiled REML score.
///
/// Partition the normal matrix into the polynomial null space `0` and the
/// penalized cascade columns `1`. Eliminating the null block gives
///
/// `|G + lambda D| / |lambda D|_+ = |G00| |I + B/lambda|`,
///
/// with `B = D^(-1/2) (G11 - G10 G00^(-1) G01) D^(-1/2)`. Consequently every
/// determinant mode is an analytic logistic function of `log(lambda)`. The
/// representation is built once, rather than re-running a basin-selecting
/// lattice of lambda-dependent factorizations.
///
/// The same elimination puts the PROFILED RESIDUAL in the same form wherever
/// the eigenbasis survives the construction — see [`CascadeResidualForm`].
struct CascadeRemlProfile<'a> {
    core: &'a Core,
    null_logdet: f64,
    modes: Vec<CascadeSpectralMode>,
    residual: CascadeResidualForm,
}

/// The lambda-independent spectral form of the PROFILED RESIDUAL
/// `R(lambda) = y'Wy - b'A(lambda)^(-1) b`.
///
/// The same null-space elimination and penalty whitening that turns the
/// determinant into a mode sum does the same thing to the residual. In the
/// Schur eigenbasis `B = V Theta V'`, `A(lambda)` acts as `Theta + lambda I`,
/// so with
///
/// `p = V' D^(-1/2) (b1 - G10 G00^(-1) b0)`   and
/// `S_k(lambda) = sum_i p_i^2 / (theta_i + lambda)^k`,
///
/// `R = anchor_energy - S1`, and the three quadratic forms the score jet needs
/// are the next three moments of that same sum:
///
/// `c'Dc = S2`, `(Dc)'A^(-1)(Dc) = S3`, `u'Du = S4` for `u = A^(-1) D c`.
///
/// `anchor_energy = y'Wy - b0' G00^(-1) b0` is the part of the residual no
/// lambda can move.
struct CascadeResidualSpectrum {
    /// `theta_i`, the Schur eigenvalue of mode `i` — the SAME numbers the
    /// determinant modes carry.
    eigenvalue: Vec<f64>,
    /// Every mode's penalty scale, which is exactly `1` because the Schur
    /// complement was whitened by `D^(-1/2)` before it was decomposed. It is
    /// materialized because [`AffineRemlProfile`] takes the pencil
    /// `h_i = g_i + lambda s_i` as two parallel slices.
    penalty: Vec<f64>,
    /// `p_i^2`, the squared projection of the null-eliminated, penalty-whitened
    /// right-hand side onto mode `i`.
    projected_square: Vec<f64>,
    /// `y'Wy - b0' G00^(-1) b0`, as the single-response slice
    /// [`AffineRemlProfile`] expects.
    anchor_energy: [f64; 1],
}

impl CascadeResidualSpectrum {
    /// `(R, S2, S3, S4)` at `lambda`. Every `theta_i` is nonnegative by
    /// construction and `lambda` is strictly positive, so every denominator is
    /// strictly positive; the caller still rejects a nonpositive `R`, which is a
    /// statement about the DATA rather than about this arithmetic.
    fn moments(&self, lambda: f64) -> (f64, f64, f64, f64) {
        let [s1, s2, s3, s4] = self.moment_sums(lambda);
        (self.anchor_energy[0] - s1, s2, s3, s4)
    }

    /// `[S_1, S_2, S_3, S_4]` at `lambda`, before the anchor subtraction that
    /// turns `S_1` into the profiled residual.
    ///
    /// Exposed separately because `S_1` must NOT be recovered from `R` by
    /// undoing that subtraction. At the top of the search domain
    /// `lambda ~ theta_max/sqrt(eps)`, so `S_1 ~ ||beta||^2 sqrt(eps)/theta_max`
    /// and `anchor - S_1` loses about eight digits of `S_1`; recovering it would
    /// leave a `sqrt(eps)` relative error, which is exactly the resolution the
    /// nested-rule certificate is charged at. Every `S_k` is a sum of strictly
    /// positive terms, so read directly it carries no cancellation at all.
    fn moment_sums(&self, lambda: f64) -> [f64; 4] {
        let mut sums = [0.0_f64; 4];
        for (&theta, &projected_square) in self.eigenvalue.iter().zip(&self.projected_square) {
            let h = theta + lambda;
            let mut term = projected_square;
            for sum in &mut sums {
                term /= h;
                *sum += term;
            }
        }
        sums
    }
}

/// What the profiled-residual quadrature certified about ITSELF, past the dense
/// cap. See [`Core::iterative_residual_spectrum`].
#[derive(Clone, Copy, Debug)]
struct ResidualQuadratureCertificate {
    /// Nodes in the rule this certificate is about.
    steps: usize,
    /// Nodes in the nested coarser rule it was charged against; `0` when the
    /// Krylov space closed and no comparison was needed.
    coarse_steps: usize,
    /// Penalized Schur rank, the hard ceiling on `steps`.
    rank: usize,
    /// Step budget the growth loop was allowed, before that ceiling.
    budget: usize,
    /// `||r_m|| / max_i |alpha_i|`: the Krylov residual against the operator
    /// scale. At roundoff, `K_m(B, beta)` is invariant and the rule is exact.
    relative_tail: f64,
    /// Conservative geometric extrapolation of the rule's remaining relative
    /// error over `S_1..S_4` and the whole lambda domain, from three nested rules.
    tail_estimate: f64,
    /// The resolution `tail_estimate` had to reach.
    target: f64,
    /// The rule may be used: it either closed the Krylov space or met `target`.
    certified: bool,
    /// `|sum_j w_j / ||beta||^2 - 1|` — the free mass self-check of a Gauss rule.
    mass_defect: f64,
    /// Fraction of `||beta||^2` that landed on roundoff-level nodes and was
    /// dropped as null-space mass.
    dropped_mass_fraction: f64,
}

impl ResidualQuadratureCertificate {
    /// This certificate as the route it authorizes.
    fn method(&self) -> ResidualMomentMethod {
        let Self {
            steps,
            coarse_steps,
            rank,
            budget,
            relative_tail,
            tail_estimate,
            target,
            certified,
            mass_defect,
            dropped_mass_fraction,
        } = *self;
        if certified {
            ResidualMomentMethod::ConvergedQuadrature {
                steps,
                coarse_steps,
                rank,
                budget,
                tail_estimate,
                target,
                relative_tail,
                mass_defect,
                dropped_mass_fraction,
            }
        } else {
            ResidualMomentMethod::Solved {
                steps,
                rank,
                budget,
                tail_estimate,
                target,
                relative_tail,
            }
        }
    }
}

/// Extrapolated estimate of the profiled-residual quadrature's REMAINING relative
/// error, from three NESTED Gauss rules for the same measure, over `S_1..S_4` and
/// the whole `log lambda` domain.
///
/// An estimate, stated as one: it is a rate fitted to two observed gaps and
/// summed, not an inequality. What makes it usable as an admission test rather
/// than a hope is (a) the rate it fits is the one the Gauss error for a resolvent
/// is KNOWN to decay at, (b) the fit errs high — see below — and (c) it is charged
/// against the exact dense eigenbasis at every budget on the production ladder by
/// `the_quadrature_tail_estimate_bounds_the_error_against_the_exact_spectrum_2503`.
///
/// WHY NOT TWO RULES. `(theta + lambda)^-k` has positive even derivatives, so the
/// standard Gauss error representation is one-signed and every rule
/// UNDER-estimates its integral: `G_j <= S_k` for every `j`. Hence
/// `G_m - G_{m/2} <= S_k - G_{m/2}` — the gap between two nested rules is an exact
/// LOWER bound on the coarser one's error. That direction is rigorous and it is
/// also not what a certificate needs: a small gap says the two errors are nearly
/// EQUAL, which is equally consistent with both being small and with both being
/// stuck. A two-rule agreement test is blind to stagnation by construction.
///
/// WHAT THREE RULES BUY. All the gaps are of one sign, so the remaining error is
/// the TAIL of the gap series, and three rules are enough to fit the decay that
/// tail obeys. The decay is not assumed: the Gauss-rule error for a resolvent
/// falls like `((sqrt(kappa) - 1)/(sqrt(kappa) + 1))^{2m}` — geometric in the node
/// count — so with `err(j) = C x^{4j/m}` sampled at `j = m/4, m/2, m` and
/// `x = r^{m/4}`,
///
/// ```text
/// g1 = G_{m/2} - G_{m/4} = C(x - x^2)      g2 = G_m - G_{m/2} = C(x^2 - x^4)
/// rho = |g2|/|g1| = x(1 + x)               tail = C x^4 = |g2| * x^2/(1 - x^2)
/// ```
///
/// so `x = (sqrt(1 + 4 rho) - 1)/2` recovers the rate from the two observed gaps
/// and the tail follows in closed form. The direction of the residual
/// approximation is the safe one: the effective rate IMPROVES with `m` once the
/// extreme Ritz values converge, so a rate fitted on `[m/4, m/2]` over-states the
/// tail beyond `m`.
///
/// `x >= 1` — no contraction — has no tail to extrapolate, and there the last
/// OBSERVED movement is reported instead. That decides correctly at both ends
/// without a special case: a rule that is genuinely stuck is still moving by more
/// than the target and is refused, while a rule that has already converged, whose
/// two gaps sit near the arithmetic floor and whose RATIO is therefore pure noise,
/// is not refused for the noise in a ratio. This is the stagnation case a two-rule
/// agreement test cannot see at all.
///
/// A gap already at the arithmetic's own floor (`eps * m * |G_m|`, the rounding of
/// the sum being compared) is converged, not stagnant, and contributes nothing.
///
/// WHICH QUANTITIES. The four moments, not the profiled residual `R = anchor -
/// S_1`, and `S_1` is read directly rather than recovered from `R` — see
/// [`CascadeResidualSpectrum::moment_sums`]. An error of `sqrt(eps)` relative on
/// `S_1` is an error of at most `sqrt(eps) * anchor` absolute on `R`, which is the
/// statement worth making about a difference that can cancel to nine digits.
///
/// GRID. Each `S_k` is a positive mixture of `(theta + lambda)^-k`, so
/// `|d log S_k / d log lambda| = k * (weighted mean of lambda/(theta+lambda)) <= k
/// <= 4`: every moment moves by at most a factor `e^4` per e-fold of `lambda`.
/// Four samples per e-fold resolve that, and the endpoints are always sampled.
fn residual_quadrature_tail_estimate(
    fine: &CascadeResidualSpectrum,
    mid: &CascadeResidualSpectrum,
    coarse: &CascadeResidualSpectrum,
    steps: usize,
    (lo, hi): (f64, f64),
) -> Result<f64, String> {
    if !(lo.is_finite() && hi.is_finite() && lo < hi) {
        return Err(format!(
            "residual cascade: invalid quadrature certification domain [{lo}, {hi}]"
        ));
    }
    let cells = (4.0 * (hi - lo)).ceil().max(8.0);
    if !cells.is_finite() {
        return Err(format!(
            "residual cascade: unbounded quadrature certification domain [{lo}, {hi}]"
        ));
    }
    let cells = cells as usize;
    let rounding = f64::EPSILON * steps.max(1) as f64;
    let mut worst = 0.0_f64;
    let mut sampled = 0usize;
    for step in 0..=cells {
        let log_lambda = lo + (hi - lo) * step as f64 / cells as f64;
        let lambda = log_lambda.exp();
        if !(lambda.is_finite() && lambda > 0.0) {
            continue;
        }
        sampled += 1;
        let fine_moments = fine.moment_sums(lambda);
        let mid_moments = mid.moment_sums(lambda);
        let coarse_moments = coarse.moment_sums(lambda);
        for index in 0..4 {
            let (value, previous, earlier) =
                (fine_moments[index], mid_moments[index], coarse_moments[index]);
            if !(value.is_finite() && previous.is_finite() && earlier.is_finite()) {
                return Err(format!(
                    "residual cascade: non-finite nested-rule moment S{} at log lambda \
                     {log_lambda} ({value}, {previous}, {earlier})",
                    index + 1
                ));
            }
            let magnitude = value.abs();
            if !(magnitude > 0.0) {
                // All three rules report an identically zero moment: nothing is
                // moving and there is nothing to extrapolate.
                if previous != 0.0 || earlier != 0.0 {
                    return Ok(f64::INFINITY);
                }
                continue;
            }
            let recent = (value - previous).abs();
            if recent <= rounding * magnitude {
                // The rule stopped moving at the arithmetic's own floor.
                continue;
            }
            let older = (previous - earlier).abs();
            let ratio = recent / older;
            // `x` solves `x^2 + x = ratio`: the per-quarter-budget decay rate the
            // two observed gaps imply.
            let rate = 0.5 * ((1.0 + 4.0 * ratio).sqrt() - 1.0);
            worst = worst.max(if rate < 1.0 {
                let square = rate * rate;
                recent * square / ((1.0 - square) * magnitude)
            } else {
                // NOT CONTRACTING: the geometric model does not apply and there
                // is nothing to extrapolate, so what is reported is the last
                // OBSERVED movement and no more. That is the honest quantity, and
                // it decides correctly at both ends without a special case. A rule
                // that is genuinely stuck is still moving by more than the target,
                // so this refuses it; a rule that has converged and whose two gaps
                // are both near the arithmetic floor — where their RATIO is pure
                // noise and routinely exceeds one — is not refused for the noise
                // in a ratio when its movement is already inside the target.
                recent / magnitude
            });
        }
    }
    if sampled == 0 {
        // No representable `lambda` on the declared domain, so the rules were
        // never compared. Agreement on zero samples is not agreement.
        return Err(format!(
            "residual cascade: the quadrature certification domain [{lo}, {hi}] contains no \
             representable lambda, so no nested-rule comparison was made"
        ));
    }
    Ok(worst)
}

/// One Golub-Meurant Gauss rule read off a Lanczos run's Jacobi matrix, with the
/// two self-checks its construction hands over for free.
struct ResidualGaussRule {
    spectrum: CascadeResidualSpectrum,
    /// `|sum_j w_j / ||beta||^2 - 1|`.
    mass_defect: f64,
    /// Fraction of `||beta||^2` dropped by the two node floors.
    dropped_mass_fraction: f64,
}

/// Where the profiled residual and its three log-lambda derivatives come from.
///
/// All three arms describe the SAME function of lambda; they differ only in what
/// the design's Schur decomposition left behind. Under the dense sizing cap the
/// determinant spectrum comes from a full eigendecomposition, so the eigen-BASIS
/// exists and the residual is a closed-form sum over exactly the modes the
/// determinant already uses — no linear solve at any lambda, and the whole score
/// is O(rank) per trial after the one decomposition.
///
/// Past the cap the determinant is a fixed-probe Hutchinson quadrature whose
/// nodes carry no basis to project the right-hand side onto — but the RESIDUAL
/// does not need one. It is a single quadratic form of a single known vector, so
/// one Lanczos run seeded with that vector gives the Golub–Meurant Gauss rule
/// for `S_k(lambda) = beta'(B + lambda I)^-k beta`, in the same node/weight
/// shape the dense route stores. When that run exhausts the Krylov space the
/// rule is EXACT, and the whole score is again solve-free at every lambda; when
/// it does not, the rule's derivative moments are not approximately right but
/// useless (#2503), so the route falls back to solving rather than shipping a
/// stationarity certificate stated in numbers that are 80% wrong.
enum CascadeResidualForm {
    /// Exact eigenbasis projection under the dense cap. Interval-extendable via
    /// [`CascadeRemlProfile::affine_view`], because the determinant modes on
    /// this route are the SAME unit-weight modes.
    Spectral(CascadeResidualSpectrum),
    /// The Golub–Meurant quadrature past the dense cap, WITH the certificate it
    /// earned. `spectrum` is `Some` exactly when that certificate says the
    /// Krylov space closed, and `None` — meaning "solve at every lambda" — when
    /// it did not; the certificate travels either way, so the refusal to use a
    /// quadrature carries the numbers that refused it.
    ///
    /// Never affine-viewable even when exact: this route's DETERMINANT modes are
    /// Hutchinson Ritz nodes with fractional weights, unrelated to the residual
    /// run's nodes.
    Quadrature {
        spectrum: Option<CascadeResidualSpectrum>,
        certificate: ResidualQuadratureCertificate,
    },
}

impl CascadeResidualForm {
    /// The lambda-independent spectral form, when this route has one.
    fn spectrum(&self) -> Option<&CascadeResidualSpectrum> {
        match self {
            Self::Spectral(spectrum) => Some(spectrum),
            Self::Quadrature { spectrum, .. } => spectrum.as_ref(),
        }
    }

    /// This route, as the fit certificate reports it.
    fn method(&self) -> ResidualMomentMethod {
        match self {
            Self::Spectral(_) => ResidualMomentMethod::DenseExact,
            Self::Quadrature { certificate, .. } => certificate.method(),
        }
    }
}

/// What a REML-SELECTED fit inherits from the profile that selected it: the
/// normalized log-determinant already evaluated at the chosen λ (so the fit does
/// not redo it), and the route the profiled residual's λ moments took (so the
/// fit certificate can report which of the three it was). A fixed-λ fit
/// evaluates no criterion and passes `None`.
#[derive(Clone, Copy, Debug)]
struct CascadeSelectionProvenance {
    normalized_logdet: f64,
    residual_moments: ResidualMomentMethod,
}

struct CascadeScoreEvaluation {
    jet: ScoreJet,
    /// `log|G + lambda D| - rank(D) log(lambda) - log|D|_+`.
    normalized_logdet: f64,
}

/// The determinant half of the score at one `log lambda`.
struct DeterminantParts {
    /// `log|G + lambda D| - rank(D) log(lambda) - log|D|_+`.
    normalized_logdet: f64,
    /// `d/d log lambda`: `-sum_i w_i t_i` with `t_i = theta_i/(theta_i+lambda)`.
    /// Nonpositive, and INCREASING in `log lambda` because every `t_i` falls.
    first: f64,
    /// `d^2/d log lambda^2`: `sum_i w_i t_i (1-t_i)`. Nonnegative.
    second: f64,
}

/// Machine-resolved bounded domain containing every determinant transition
/// `lambda ~ theta`. Outside it, every positive mode is within `sqrt(epsilon)` of
/// its analytic small- or large-lambda limit. The bounds scale with the actual
/// design spectrum rather than a fixed log-lambda window.
///
/// A free function rather than a profile method because the profiled RESIDUAL's
/// convergence certificate has to be charged over exactly this interval, and the
/// residual is built while the profile is being assembled — the interval is a
/// function of the determinant modes alone, so it is available at that point.
fn log_lambda_domain_from_modes(modes: &[CascadeSpectralMode]) -> Result<(f64, f64), String> {
    let mut smallest = f64::INFINITY;
    let mut largest = 0.0_f64;
    for mode in modes {
        if mode.weight > 0.0 && mode.eigenvalue > 0.0 {
            smallest = smallest.min(mode.eigenvalue);
            largest = largest.max(mode.eigenvalue);
        }
    }
    if !(smallest.is_finite() && smallest > 0.0 && largest.is_finite() && largest > 0.0) {
        return Err(
            "residual cascade: the data identify no positive penalized Schur mode; log lambda is not estimable"
                .into(),
        );
    }
    let log_relative_resolution = f64::EPSILON.sqrt().ln();
    let lo = (smallest.ln() + log_relative_resolution).max(f64::MIN_POSITIVE.ln());
    let hi = (largest.ln() - log_relative_resolution).min(f64::MAX.ln());
    if !(lo.is_finite() && hi.is_finite() && lo < hi) {
        return Err(format!(
            "residual cascade: invalid spectrum-derived log-lambda domain [{lo}, {hi}]"
        ));
    }
    Ok((lo, hi))
}

impl CascadeRemlProfile<'_> {
    fn log_lambda_domain(&self) -> Result<(f64, f64), String> {
        log_lambda_domain_from_modes(&self.modes)
    }

    /// This profile as the affine spectral REML score it is, when the residual
    /// is spectral.
    ///
    /// With `h_i(lambda) = theta_i + lambda` the cascade's dense-route score is
    /// term for term an [`AffineRemlProfile`]: `sum log h_i - rank log lambda`
    /// is the normalized log-determinant, `R = anchor - sum p_i^2/h_i` is the
    /// profiled residual, and there is one response. The point of saying so is
    /// the ENCLOSURE. `AffineRemlProfile::enclose` evaluates the mode kernels on
    /// an interval lambda, so it is a genuine interval extension whose width
    /// collapses with the cell; [`CascadeRemlProfile::enclose`] can only pad the
    /// endpoint jets with global Lipschitz constants, and that pad does not
    /// collapse — see its own note on why the search could not terminate.
    ///
    /// [`CascadeResidualForm::Quadrature`] is DELIBERATELY excluded even though
    /// it carries the same spectral shape. `AffineRemlProfile` computes the
    /// determinant from the modes it is handed — `sum_i log h_i - rank log
    /// lambda` — and past the dense cap the determinant is a Hutchinson
    /// quadrature over 24 independent probes with fractional weights, which is
    /// neither the residual run's node set nor unit-weight. Handing it the
    /// residual nodes would silently substitute one determinant for another.
    fn affine_view(&self) -> Result<Option<AffineRemlProfile<'_>>, String> {
        let CascadeResidualForm::Spectral(spectrum) = &self.residual else {
            return Ok(None);
        };
        let core = self.core;
        AffineRemlProfile::new(
            &spectrum.eigenvalue,
            &spectrum.penalty,
            &spectrum.projected_square,
            &spectrum.anchor_energy,
            (core.y.len() - core.nullity()) as f64,
            // Every whitened mode carries penalty scale 1, so the penalized
            // determinant rank is the full Schur rank.
            spectrum.penalty.len(),
            self.null_logdet,
        )
        .map(Some)
        .map_err(|error| format!("residual cascade: affine spectral profile rejected: {error}"))
    }

    /// The normalized log-determinant and its first two `log lambda`
    /// derivatives.
    ///
    /// `O(modes)` and free of linear algebra on every route, which is what lets
    /// [`Self::enclose`] have the determinant half of the jet at both cell
    /// endpoints without an evaluation of its own.
    fn determinant_parts(&self, log_lambda: f64, lambda: f64) -> DeterminantParts {
        let mut parts = DeterminantParts {
            normalized_logdet: self.null_logdet,
            first: 0.0,
            second: 0.0,
        };
        for mode in &self.modes {
            let theta = mode.eigenvalue;
            let weight = mode.weight;
            if theta == 0.0 || weight == 0.0 {
                continue;
            }
            // Stable forms for log(1 + theta/lambda) and
            // t=theta/(lambda+theta), including widely separated scales.
            let log_theta = theta.ln();
            parts.normalized_logdet += weight
                * if log_theta > log_lambda {
                    (log_theta - log_lambda) + (log_lambda - log_theta).exp().ln_1p()
                } else {
                    (log_theta - log_lambda).exp().ln_1p()
                };
            let t = if theta > lambda {
                1.0 / (1.0 + lambda / theta)
            } else {
                theta / (lambda + theta)
            };
            parts.first -= weight * t;
            parts.second += weight * t * (1.0 - t);
        }
        parts
    }

    fn evaluate(&self, log_lambda: f64) -> Result<CascadeScoreEvaluation, String> {
        let lambda = gam_problem::checked_exp_log_strength(log_lambda)
            .map_err(|error| format!("residual cascade: {error}"))?;

        let core = self.core;
        // R = y'Wy - b'A^-1b. With A' = lambda D,
        // R' = lambda c'Dc and
        // R'' = lambda c'Dc - 2 lambda^2 (Dc)'A^-1(Dc).
        // The third derivative is retained to justify the analytic enclosure
        // used below; it needs no third solve because the last quadratic is
        // u'Du for u=A^-1Dc.
        let (rss, penalty_energy, inverse_penalty_energy, third_energy) = match self
            .residual
            .spectrum()
        {
            // The decomposition that produced the determinant modes produced
            // these three quadratic forms too. Reading them off it costs
            // O(rank); re-deriving them cost a fresh O(m^3) factorization of
            // `A = X'WX + λD` at EVERY λ the certified search visits.
            Some(spectrum) => spectrum.moments(lambda),
            None => {
                // ONE factorization of `A` for BOTH right-hand sides below; the
                // matrix is the same at this λ and only the right-hand side
                // differs.
                //
                // Every failure here is a failure to evaluate the criterion at a
                // λ the search chose, and on this route the reason the criterion
                // needs a solve at all is that no quadrature converged. So the
                // refusal carries the certificate that refused: without it the
                // message names the SOLVER ("CG failed to reach 1e-9") when the
                // decision that put a solve in the path was made much earlier
                // and elsewhere. That is the archaeology #2503 opens with.
                let solved = |error: String| -> String {
                    format!(
                        "{error}; the profiled residual is on the SOLVE route at this λ because                          the Golub–Meurant quadrature did not converge: {:?}",
                        self.residual.method()
                    )
                };
                let solver = core.coeff_solver(lambda).map_err(solved)?;
                let coeff = solver.solve(core, lambda, &core.rhs).map_err(solved)?;
                let dc: Vec<f64> = coeff
                    .iter()
                    .zip(core.pen_diag.iter())
                    .map(|(&c, &d)| d * c)
                    .collect();
                let penalty_energy = coeff
                    .iter()
                    .zip(dc.iter())
                    .map(|(&c, &v)| c * v)
                    .sum::<f64>();
                let u = solver.solve(core, lambda, &dc).map_err(solved)?;
                let inverse_penalty_energy =
                    dc.iter().zip(u.iter()).map(|(&a, &b)| a * b).sum::<f64>();
                let third_energy = u
                    .iter()
                    .zip(core.pen_diag.iter())
                    .map(|(&v, &d)| d * v * v)
                    .sum::<f64>();
                (
                    core.rss_pen(&coeff),
                    penalty_energy,
                    inverse_penalty_energy,
                    third_energy,
                )
            }
        };
        if !(rss.is_finite() && rss > 0.0) {
            return Err(format!(
                "residual cascade: degenerate penalized residual {rss}"
            ));
        }
        let rss_d1 = lambda * penalty_energy;
        let lambda2 = lambda * lambda;
        let rss_d2 = rss_d1 - 2.0 * lambda2 * inverse_penalty_energy;
        let rss_d3 =
            rss_d1 - 6.0 * lambda2 * inverse_penalty_energy + 6.0 * lambda2 * lambda * third_energy;

        let DeterminantParts {
            normalized_logdet,
            first: determinant_d1,
            second: determinant_d2,
        } = self.determinant_parts(log_lambda, lambda);

        let dof = (core.y.len() - core.nullity()) as f64;
        let rss_log_d1 = rss_d1 / rss;
        let rss_log_d2 = rss_d2 / rss - rss_log_d1 * rss_log_d1;
        let rss_log_d3 = rss_d3 / rss - 3.0 * rss_d1 * rss_d2 / (rss * rss)
            + 2.0 * rss_log_d1 * rss_log_d1 * rss_log_d1;
        if !(rss_log_d3.is_finite()) {
            return Err(format!(
                "residual cascade: non-finite analytic residual derivative at log lambda {log_lambda}"
            ));
        }
        let jet = ScoreJet {
            value: -0.5 * (normalized_logdet + dof * (rss / dof).ln()),
            derivative: -0.5 * (determinant_d1 + dof * rss_log_d1),
            curvature: -0.5 * (determinant_d2 + dof * rss_log_d2),
            // This profile's enclosure pads with the CLOSED-FORM `third_abs_bound`
            // Lipschitz constant rather than the endpoint third derivative, so it
            // never reads this field; the exact `rss_log_d3` above is retained
            // only as the analyticity check that justifies that bound.
            third: 0.0,
        };
        if !(jet.value.is_finite() && jet.derivative.is_finite() && jet.curvature.is_finite()) {
            return Err(format!(
                "residual cascade: non-finite REML jet at log lambda {log_lambda}: value {}, derivative {}, curvature {}",
                jet.value, jet.derivative, jet.curvature
            ));
        }
        Ok(CascadeScoreEvaluation {
            jet,
            normalized_logdet,
        })
    }

    /// Outer derivative ranges for the route with no eigenbasis: the INTERSECTION
    /// of an additive Lipschitz pad and a multiplicative spectral bracket.
    ///
    /// Both are outer enclosures of the same two derivatives, so intersecting
    /// them is again one — and they fail in opposite places, which is the whole
    /// reason both are here.
    ///
    /// THE PAD. Each determinant mode has `|f''| <= 1/4` and `|f'''| <= 1/4`.
    /// After the null-space elimination the profiled residual is a positive
    /// mixture of `lambda/(theta+lambda)` kernels plus a lambda-independent
    /// residual, so its log has `|g''| <= 2`, `|g'''| <= 6` (the loose moment
    /// bounds for variables in `[0,1]`). Endpoint jets plus these bounds enclose
    /// the interval without sampling it, and the jets arrive as the search's own
    /// `left`/`right` SAMPLES, so this function evaluates the profile zero
    /// times. The pad is tight where the score has real curvature — around the
    /// optimum, where certifying a unique root actually happens.
    ///
    /// WHERE THE PAD FAILS. Its radius is `C·width` with `C` of order the
    /// residual degrees of freedom, and it shrinks only as fast as the cell.
    /// [`Self::log_lambda_domain`] deliberately runs `ln(1/sqrt(eps))` past the
    /// extreme Schur eigenvalues, and out there `f'` has decayed to order
    /// `rank·sqrt(eps)` — while the search's resolution floor is also
    /// `sqrt(eps)`, so `C·width` AT THE FLOOR is larger than the derivative it
    /// is meant to bracket. The tail is then neither dismissible (the pad
    /// straddles zero) nor refinable (the floor is reached), and the search
    /// grinds toward a `ScoreSearchError::Unresolved` it cannot avoid. That is
    /// a search that does not terminate on its own domain, not a slow one.
    ///
    /// THE BRACKET, which has no floor because it is RELATIVE. Write
    /// `f' = -(D1 + dof·rho)/2` with `D1 = -sum_i w_i t_i`,
    /// `t_i = theta_i/(theta_i+lambda)` and `rho = R'/R > 0`.
    ///
    /// * every `t_i` falls with `log lambda`, so `D1` RISES: `D1` on the cell is
    ///   bracketed by its two endpoint values exactly, with no bound at all;
    /// * `d log t_i/dx = -(1-t_i)` and `d log[t_i(1-t_i)]/dx = 2t_i - 1`, both
    ///   in `[-1, 1]`, and a positive mixture's log-derivative is a convex
    ///   combination of its parts', so `D2` and `R'` each satisfy
    ///   `|d log(.)/dx| <= 1`;
    /// * `R` rises, so `d log rho/dx = d log R'/dx - rho <= 1`, giving
    ///   `rho(x) <= rho(a)e^w` and `rho(x) >= rho(b)e^{-w}`;
    /// * `|R''| <= R'` mode by mode, so `sigma = R''/R - rho^2` lies in
    ///   `[-rho(1+rho), rho]`.
    ///
    /// Every one of those bounds is proportional to the quantity it bounds, so
    /// in a tail where `f'` merely has a constant sign the bracket excludes zero
    /// at a width that does not depend on how far the tail runs.
    fn enclose(&self, left: ScoreSample, right: ScoreSample) -> Result<DerivativeEnclosure, String> {
        let (lo, hi) = (left.x, right.x);
        if !(lo.is_finite() && hi.is_finite() && lo <= hi) {
            return Err(format!(
                "residual cascade: invalid score-enclosure interval [{lo}, {hi}]"
            ));
        }
        let width = hi - lo;
        let dof = (self.core.y.len() - self.core.nullity()) as f64;
        let pad = self.lipschitz_pad(left, right, width);
        let Some(bracket) = self.multiplicative_bracket(left, right, width, dof)? else {
            return Ok(pad);
        };
        // Two OUTER enclosures of the same real number must overlap — both
        // contain the endpoint derivatives, if nothing else. A disjoint pair
        // means one of them is not an outer bound, and an enclosure that is not
        // an outer bound does not fail loudly downstream: it lets the search
        // discard a cell that held a stationary point and return a certified
        // wrong answer. Refuse here instead of narrowing to whichever one is
        // left, which would be choosing a winner between two derivations with
        // no evidence about which is sound.
        let derivative = intersect(pad.derivative, bracket.derivative)
            .ok_or_else(|| disjoint("derivative", lo, hi, pad.derivative, bracket.derivative))?;
        let curvature = intersect(pad.curvature, bracket.curvature)
            .ok_or_else(|| disjoint("curvature", lo, hi, pad.curvature, bracket.curvature))?;
        Ok(DerivativeEnclosure {
            derivative,
            curvature,
        })
    }

    /// The additive half of [`Self::enclose`], on its own — the enclosure this
    /// module used to return, kept separable so its tail stall can be asserted
    /// against the bracket that repairs it rather than described.
    fn lipschitz_pad(
        &self,
        left: ScoreSample,
        right: ScoreSample,
        width: f64,
    ) -> DerivativeEnclosure {
        let rank = (self.core.m - self.core.nullity()) as f64;
        let dof = (self.core.y.len() - self.core.nullity()) as f64;
        let curvature_abs_bound = 0.5 * (0.25 * rank + 2.0 * dof);
        let third_abs_bound = 0.5 * (0.25 * rank + 6.0 * dof);
        let derivative_radius = curvature_abs_bound * width;
        let curvature_radius = third_abs_bound * width;
        DerivativeEnclosure {
            derivative: ClosedInterval::outward(
                (left.derivative - derivative_radius).min(right.derivative - derivative_radius),
                (left.derivative + derivative_radius).max(right.derivative + derivative_radius),
            ),
            curvature: ClosedInterval::outward(
                (left.curvature - curvature_radius).min(right.curvature - curvature_radius),
                (left.curvature + curvature_radius).max(right.curvature + curvature_radius),
            ),
        }
    }

    /// The relative bracket described on [`Self::enclose`], or `None` when the
    /// endpoint residual log-slopes cannot be recovered to a definite sign (see
    /// below), in which case the pad stands alone and nothing is claimed.
    fn multiplicative_bracket(
        &self,
        left: ScoreSample,
        right: ScoreSample,
        width: f64,
        dof: f64,
    ) -> Result<Option<DerivativeEnclosure>, String> {
        // Per endpoint: (D1, D2, rho lower estimate, rho upper estimate).
        let mut parts = [(0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64); 2];
        for (slot, sample) in parts.iter_mut().zip([left, right]) {
            let lambda = gam_problem::checked_exp_log_strength(sample.x)
                .map_err(|error| format!("residual cascade: {error}"))?;
            let determinant = self.determinant_parts(sample.x, lambda);
            // `evaluate` formed `derivative = -(D1 + dof*rho)/2` from THIS
            // determinant value — recomputed here bitwise, same inputs, same
            // code — so `-2*derivative` returns that sum exactly (halving and
            // doubling are exact) and one subtraction recovers `dof*rho`. The
            // recovery's error is the roundoff of the sum it undoes plus its
            // own and the division: at most four roundings of magnitude
            // `|D1| + |2*derivative|`. That is a rounding COUNT, not a slack to
            // tune, and it is carried in both directions.
            let total = -2.0 * sample.derivative;
            let recovered = (total - determinant.first) / dof;
            let slack = 4.0 * f64::EPSILON * (determinant.first.abs() + total.abs()) / dof;
            *slot = (
                determinant.first,
                determinant.second,
                recovered - slack,
                recovered + slack,
            );
        }
        let (first_lo, second_lo, _, rho_left_upper) = parts[0];
        let (first_hi, second_hi, rho_right_lower, rho_right_upper) = parts[1];

        // `rho > 0` is a theorem (`R' = lambda c'Dc > 0`), but a recovery that
        // cannot place it strictly above zero has lost it to cancellation and
        // can no longer carry a RELATIVE bound. Decline rather than guess.
        let growth = width.exp();
        if !(rho_left_upper > 0.0 && rho_right_upper > 0.0 && growth.is_finite()) {
            return Ok(None);
        }
        // `rho(x) <= rho(a)e^w` integrating the `<= 1` side forward from the
        // left endpoint; `rho(x) >= rho(b)e^-w` integrating it back from the
        // right. Only those two directions are free of `rho` itself.
        let rho_hi = rho_left_upper.max(rho_right_upper) * growth;
        let rho_lo = (rho_right_lower / growth).max(0.0);

        // D1 rises across the cell, so its endpoint values bracket it exactly.
        let derivative = ClosedInterval::outward(
            -0.5 * (first_hi + dof * rho_hi),
            -0.5 * (first_lo + dof * rho_lo),
        );

        // D2 > 0 at both ends forces D2 > 0 throughout (a zero inside would make
        // `|d log D2/dx| <= 1` impossible on a finite cell), which is what lets
        // the relative bound apply; otherwise only `D2 >= 0` is available.
        let (d2_lo, d2_hi) = if second_lo > 0.0 && second_hi > 0.0 {
            (second_lo.max(second_hi) / growth, second_lo.min(second_hi) * growth)
        } else {
            (0.0, 0.25 * self.modes.iter().map(|mode| mode.weight).sum::<f64>())
        };
        let curvature = ClosedInterval::outward(
            -0.5 * (d2_hi + dof * rho_hi),
            -0.5 * (d2_lo - dof * rho_hi * (1.0 + rho_hi)),
        );
        Ok(Some(DerivativeEnclosure {
            derivative,
            curvature,
        }))
    }
}

/// The tighter of two outer enclosures of the same quantity, or `None` if they
/// are disjoint — which is a contradiction, not a tight answer.
fn intersect(a: ClosedInterval, b: ClosedInterval) -> Option<ClosedInterval> {
    let merged = ClosedInterval::new(a.lo.max(b.lo), a.hi.min(b.hi));
    (merged.lo <= merged.hi).then_some(merged)
}

fn disjoint(
    what: &str,
    lo: f64,
    hi: f64,
    pad: ClosedInterval,
    bracket: ClosedInterval,
) -> String {
    format!(
        "residual cascade: the Lipschitz pad and the multiplicative bracket give DISJOINT \
         {what} enclosures on [{lo}, {hi}] ({pad:?} versus {bracket:?}); two outer bounds of \
         one quantity cannot be disjoint, so one of them is not an outer bound"
    )
}

/// A coefficient solver pinned to one λ (see [`Core::coeff_solver`]). The dense
/// arms hold the Cholesky factor so repeated right-hand sides at that λ cost
/// only triangular solves; the iterative arm holds the coarse-space
/// preconditioner, which is likewise a function of λ alone and whose coarse
/// block is an `O(n·q_C²) + O(q_C³)` assembly and factorization — paid once per
/// λ, not once per right-hand side.
enum CoeffSolver<'a> {
    Cached(&'a [f64]),
    Factored(Vec<f64>),
    Iterative(Preconditioner),
}

impl CoeffSolver<'_> {
    fn solve(&self, core: &Core, lambda: f64, b: &[f64]) -> Result<Vec<f64>, String> {
        match self {
            Self::Cached(l) => Ok(chol_solve(l, core.m, b)),
            Self::Factored(l) => Ok(chol_solve(l, core.m, b)),
            Self::Iterative(preconditioner) => core
                .pcg_with(lambda, preconditioner, b, None)
                .map(|(coeff, _, _)| coeff),
        }
    }
}

impl Core {
    #[inline]
    fn dense_gram_entry(&self, row: usize, col: usize) -> Option<f64> {
        let gram = self.dense_gram.as_ref()?;
        let (i, j) = if row <= col { (row, col) } else { (col, row) };
        Some(gram[i * self.m + j])
    }

    /// Factor the unpenalized polynomial Gram block. It is tiny (`dim+1 <= 4`)
    /// on every route and is the exact anchor for the Schur complement.
    fn null_gram_factor(&self) -> Result<(Vec<f64>, f64), String> {
        let q = self.nullity();
        let mut gram = vec![0.0; q * q];
        if self.dense_gram.is_some() {
            for i in 0..q {
                for j in i..q {
                    let value = self.dense_gram_entry(i, j).expect("dense Gram exists");
                    gram[i * q + j] = value;
                    gram[j * q + i] = value;
                }
            }
        } else {
            for row in 0..self.w.len() {
                let lo = self.row_ptr[row];
                let hi = self.row_ptr[row + 1];
                for ea in lo..hi {
                    let ca = self.col_idx[ea] as usize;
                    if ca >= q {
                        break;
                    }
                    let weighted = self.w[row] * self.vals[ea];
                    for eb in ea..hi {
                        let cb = self.col_idx[eb] as usize;
                        if cb >= q {
                            break;
                        }
                        gram[ca * q + cb] += weighted * self.vals[eb];
                    }
                }
            }
            for i in 0..q {
                for j in i + 1..q {
                    gram[j * q + i] = gram[i * q + j];
                }
            }
        }
        let logdet = cholesky_logdet(&mut gram, q).map_err(|error| {
            format!("residual cascade: polynomial null-space factorization failed: {error}")
        })?;
        Ok((gram, logdet))
    }

    /// Apply the penalty-whitened Schur complement `B` without materializing
    /// the data Gram. Scratch buffers are supplied by the Lanczos caller so
    /// each iteration remains allocation-free apart from the tiny null solve.
    fn schur_whitened_matvec(
        &self,
        null_chol: &[f64],
        input: &[f64],
        output: &mut [f64],
        full: &mut [f64],
        gram_full: &mut [f64],
        projected_null: &mut [f64],
    ) {
        let q = self.nullity();
        full.fill(0.0);
        for (i, &value) in input.iter().enumerate() {
            full[q + i] = value / self.pen_diag[q + i].sqrt();
        }
        self.matvec(0.0, full, gram_full);
        let null_coeff = chol_solve(null_chol, q, &gram_full[..q]);
        full.fill(0.0);
        full[..q].copy_from_slice(&null_coeff);
        self.matvec(0.0, full, projected_null);
        for i in 0..output.len() {
            output[i] = (gram_full[q + i] - projected_null[q + i]) / self.pen_diag[q + i].sqrt();
        }
    }

    /// One full-reorthogonalization Lanczos run on the penalty-whitened Schur
    /// complement `B`, from a caller-supplied start vector.
    ///
    /// The determinant sweep (a Rademacher probe per run) and the profiled
    /// residual (one run seeded with the whitened right-hand side) are the same
    /// Krylov process on the same operator, and they read the same Jacobi matrix
    /// afterwards. Sharing ONE implementation is not tidiness: an accuracy gate
    /// on the residual quadrature that measured a copy of this recurrence would
    /// certify a routine that does not ship.
    ///
    /// The returned `T_m` is the Jacobi matrix of the Gauss quadrature rule for
    /// the measure `mu` that `start` induces on the spectrum of `B`, so
    /// `start' g(B) start ≈ ||start||^2 · e_1' g(T_m) e_1` for every analytic
    /// `g` — the Golub–Meurant rule. Two things decide whether that `≈` is an
    /// `=`, and both ride on the returned [`SchurLanczos`]: `tail`, the `(m+1, m)`
    /// entry the truncation dropped, and `dimension_ceiling`, the caller's bound
    /// on how many dimensions `K(B, start)` can have at all — reaching it leaves
    /// the space nothing to grow into, whatever roundoff has left in `tail`.
    fn schur_lanczos(
        &self,
        null_chol: &[f64],
        start: &[f64],
        max_steps: usize,
        dimension_ceiling: usize,
    ) -> Result<SchurLanczos, String> {
        let nullity = self.nullity();
        let rank = self.m - nullity;
        if start.len() != rank {
            return Err(format!(
                "residual cascade: Lanczos start vector carries {} entries against penalized \
                 Schur rank {rank}",
                start.len()
            ));
        }
        // `Sum_j w_j = ||start||^2` exactly, which is the free mass self-check
        // every caller applies to the weights it derives from this run.
        let start_norm_sq = start.iter().map(|value| value * value).sum::<f64>();
        let start_norm = start_norm_sq.sqrt();
        if !(start_norm.is_finite() && start_norm > 0.0) {
            return Err(format!(
                "residual cascade: Lanczos start vector has non-positive norm {start_norm}"
            ));
        }
        let steps = max_steps.min(rank);
        let mut full = vec![0.0; self.m];
        let mut gram_full = vec![0.0; self.m];
        let mut projected_null = vec![0.0; self.m];
        let mut matvec = vec![0.0; rank];
        // The reorthogonalization basis, `steps x rank` row-major in ONE allocation
        // rather than a `Vec` of `Vec`s. This is the run's whole memory footprint
        // (see `RESIDUAL_QUADRATURE_BASIS_BYTES`) and, at these sizes, its whole
        // cost: the loop below streams it end to end at every step, so it is
        // bandwidth-bound and the per-vector indirection was pure overhead.
        // Reserving it up front also removes the reallocation-and-copy that a
        // growing `Vec` of a few hundred megabytes otherwise pays repeatedly. The
        // arithmetic and its ORDER are unchanged, so this is bit-identical.
        let mut basis: Vec<f64> = Vec::with_capacity(steps.saturating_mul(rank));
        let mut q: Vec<f64> = start.iter().map(|&value| value / start_norm).collect();
        let mut q_previous: Option<Vec<f64>> = None;
        let mut alpha: Vec<f64> = Vec::with_capacity(steps);
        let mut beta: Vec<f64> = Vec::with_capacity(steps.saturating_sub(1));
        // `max_i |alpha_i|` is a lower bound on `||B||` restricted to the Krylov
        // space (every `alpha_i` is a Rayleigh quotient) and it only rises, which
        // is exactly what the break floor below needs. Stating that floor against
        // the CURRENT `alpha_i` instead — as this recurrence originally did — makes
        // it COLLAPSE in the regime it exists to detect: once the Krylov space has
        // consumed the operator's range, the remaining iterates are roundoff, the
        // Rayleigh quotients go to zero with them, and the floor chases the
        // residual down instead of catching it. Measured on the #2503 `n = 2500`
        // fixture, where `rank = 16565` but `rank(B) <= n - nullity = 2497`, so
        // 85% of the space is null: the run ground to 2023 steps without ever
        // reporting invariance and produced a Ritz value at `-7.9e-11`.
        let mut spectral_scale = 0.0_f64;
        let mut tail = 0.0_f64;
        // A Krylov space that has consumed its whole DIMENSION is invariant: there
        // is nothing left for it to grow into. `dimension_ceiling` is the caller's
        // bound on that dimension — `rank` is the trivial one, but for a start
        // vector inside `range(B)` the binding bound is `rank(B) <= n - nullity`
        // (`B = Z'WZ` with `W^(1/2) Z = (I - P) W^(1/2) X_1` and `P` of rank
        // `nullity`), which on a bounding-box-filled cascade is an order of
        // magnitude below `rank`. Reaching it makes the Gauss rule exact whatever
        // the accumulated roundoff has left in `tail`.
        let mut invariant = steps >= dimension_ceiling;
        for step in 0..steps {
            self.schur_whitened_matvec(
                null_chol,
                &q,
                &mut matvec,
                &mut full,
                &mut gram_full,
                &mut projected_null,
            );
            let diagonal = matvec
                .iter()
                .zip(q.iter())
                .map(|(&a, &b)| a * b)
                .sum::<f64>();
            alpha.push(diagonal);
            spectral_scale = spectral_scale.max(diagonal.abs());
            let mut residual = matvec.clone();
            for i in 0..rank {
                residual[i] -= diagonal * q[i];
            }
            if let Some(previous) = &q_previous {
                let previous_beta = beta.last().copied().unwrap_or(0.0);
                for i in 0..rank {
                    residual[i] -= previous_beta * previous[i];
                }
            }
            basis.extend_from_slice(&q);
            for direction in basis.chunks_exact(rank) {
                let projection = residual
                    .iter()
                    .zip(direction.iter())
                    .map(|(&a, &b)| a * b)
                    .sum::<f64>();
                for (value, &component) in residual.iter_mut().zip(direction) {
                    *value -= projection * component;
                }
            }
            let norm = residual
                .iter()
                .map(|value| value * value)
                .sum::<f64>()
                .sqrt();
            if !norm.is_finite() {
                return Err(
                    "residual cascade: Schur-spectrum Lanczos produced a non-finite norm".into(),
                );
            }
            tail = norm;
            let rounding_floor =
                f64::EPSILON * alpha.len() as f64 * spectral_scale.max(f64::MIN_POSITIVE);
            if norm <= rounding_floor {
                invariant = true;
                break;
            }
            if step + 1 == steps {
                break;
            }
            beta.push(norm);
            q_previous = Some(std::mem::replace(&mut q, residual));
            for value in &mut q {
                *value /= norm;
            }
        }
        Ok(SchurLanczos {
            alpha,
            beta,
            tail,
            spectral_scale,
            start_norm_sq,
            invariant,
        })
    }

    /// Exact Schur spectrum under the dense sizing cap, WITH the eigenbasis it
    /// is computed from.
    ///
    /// `eigh` returns the eigenvectors whether or not the caller keeps them.
    /// Dropping them here used to leave the residual half of the same score with
    /// no way to reach the decomposition, so `evaluate` re-derived it as a fresh
    /// Cholesky of `A = X'WX + λD` at every λ the certified search visited —
    /// an O(m^3) factorization per trial, standing in for a projection this
    /// factorization had already made available.
    fn dense_cascade_spectrum(
        &self,
        null_chol: &[f64],
    ) -> Result<(Vec<CascadeSpectralMode>, CascadeResidualSpectrum), String> {
        let q = self.nullity();
        let rank = self.m - q;
        let mut schur = Array2::<f64>::zeros((rank, rank));
        let mut cross = vec![0.0; q];
        for j in 0..rank {
            for (k, value) in cross.iter_mut().enumerate() {
                *value = self.dense_gram_entry(k, q + j).expect("dense Gram exists");
            }
            let projected = chol_solve(null_chol, q, &cross);
            for i in 0..=j {
                let mut value = self
                    .dense_gram_entry(q + i, q + j)
                    .expect("dense Gram exists");
                for (k, &coefficient) in projected.iter().enumerate() {
                    value -=
                        self.dense_gram_entry(q + i, k).expect("dense Gram exists") * coefficient;
                }
                value /= (self.pen_diag[q + i] * self.pen_diag[q + j]).sqrt();
                schur[(i, j)] = value;
                schur[(j, i)] = value;
            }
        }
        let (eigenvalues, eigenvectors) = schur.eigh(Side::Lower).map_err(|error| {
            format!("residual cascade: Schur-complement eigendecomposition failed: {error}")
        })?;
        let scale = eigenvalues
            .iter()
            .copied()
            .map(f64::abs)
            .fold(0.0, f64::max);
        let roundoff = f64::EPSILON * rank.max(1) as f64 * scale.max(f64::MIN_POSITIVE);
        // A mode inside the decomposition's OWN roundoff floor is a null
        // direction of the whitened design, not a small positive one. The floor
        // is the same quantity the semidefiniteness check below is stated in;
        // reading it in one direction only ("this is not really negative") and
        // not the other ("so it is not really positive either") is what lets a
        // noise-level eigenvalue set the small-lambda end of the search domain
        // and divide into the residual there.
        let certified = |eigenvalue: f64| {
            if eigenvalue > roundoff {
                eigenvalue
            } else {
                0.0
            }
        };
        let modes = eigenvalues
            .iter()
            .copied()
            .enumerate()
            .map(|(index, eigenvalue)| {
                if !eigenvalue.is_finite() || eigenvalue < -roundoff {
                    Err(format!(
                        "residual cascade: penalty-whitened Schur mode {index} is not positive semidefinite ({eigenvalue})"
                    ))
                } else {
                    Ok(CascadeSpectralMode {
                        eigenvalue: certified(eigenvalue),
                        weight: 1.0,
                    })
                }
            })
            .collect::<Result<Vec<_>, String>>()?;

        // The same null elimination and penalty whitening, applied to the
        // right-hand side instead of to the Gram: `beta = D^(-1/2)(b1 - G10
        // G00^(-1) b0)`, then projected onto the eigenbasis above.
        let null_solved = chol_solve(null_chol, q, &self.rhs[..q]);
        let mut whitened = vec![0.0_f64; rank];
        for (i, value) in whitened.iter_mut().enumerate() {
            let mut entry = self.rhs[q + i];
            for (k, &coefficient) in null_solved.iter().enumerate() {
                entry -= self.dense_gram_entry(q + i, k).expect("dense Gram exists") * coefficient;
            }
            *value = entry / self.pen_diag[q + i].sqrt();
        }
        // A null mode carries NO response energy, exactly. The Schur complement
        // and the whitened right-hand side are built from the same design `Z`:
        // `B = Z'WZ` and `beta = Z'Wy`, so `Bv = 0` gives `Zv = 0` and hence
        // `v'beta = (Zv)'Wy = 0`. What the arithmetic returns for such a mode is
        // roundoff — and the residual sum divides it by `theta + lambda`, which
        // at the bottom of the search domain is SMALLER than that roundoff. On a
        // 558-column cascade the three null modes carried `p^2 ~ 3e-16` against
        // `lambda ~ 4e-19` and drove the profiled residual to -764 where the
        // mathematics bounds it below by the unpenalized residual sum of
        // squares. Restoring the exact identity is not a tolerance.
        let mut projected_square = vec![0.0_f64; rank];
        for (j, square) in projected_square.iter_mut().enumerate() {
            if certified(eigenvalues[j]) == 0.0 {
                continue;
            }
            let mut projection = 0.0;
            for (i, &value) in whitened.iter().enumerate() {
                projection += eigenvectors[(i, j)] * value;
            }
            *square = projection * projection;
        }
        let anchor_energy = self.ytwy
            - self.rhs[..q]
                .iter()
                .zip(null_solved.iter())
                .map(|(&b, &c)| b * c)
                .sum::<f64>();
        if !(anchor_energy.is_finite() && projected_square.iter().all(|v| v.is_finite())) {
            return Err(format!(
                "residual cascade: non-finite spectral residual representation (anchor {anchor_energy})"
            ));
        }
        Ok((
            modes,
            CascadeResidualSpectrum {
                eigenvalue: eigenvalues.iter().copied().map(certified).collect(),
                penalty: vec![1.0; rank],
                projected_square,
                anchor_energy: [anchor_energy],
            },
        ))
    }

    /// Fixed-probe Lanczos quadrature of the lambda-independent Schur
    /// spectrum. Unlike the previous lambda-dependent SLQ call, its nodes and
    /// weights define one smooth analytic score across the entire search
    /// domain, so differentiating the scalar kernels is exact for the score
    /// being optimized.
    fn iterative_cascade_spectrum(
        &self,
        null_chol: &[f64],
    ) -> Result<Vec<CascadeSpectralMode>, String> {
        let q0 = self.nullity();
        let rank = self.m - q0;
        let steps = SLQ_LANCZOS_STEPS.min(rank);
        let mut modes = Vec::with_capacity(SLQ_PROBES * steps);

        for probe in 0..SLQ_PROBES {
            let mut rng =
                SplitMix64::new(RNG_SEED ^ (probe as u64).wrapping_mul(0xD134_2543_DE82_EF95));
            // Unit-entry Rademacher probe. `schur_lanczos` normalizes it, and
            // `||probe||^2 = rank` exactly (a sum of `rank` ones), so the
            // Hutchinson scaling below reads that norm rather than restating it.
            let probe_vector = (0..rank).map(|_| rng.next_sign()).collect::<Vec<_>>();
            let SchurLanczos {
                alpha,
                beta,
                start_norm_sq,
                ..
            } = self.schur_lanczos(null_chol, &probe_vector, steps, rank)?;
            let (eigenvalues, first_components) = symmetric_tridiagonal_eigen(&alpha, &beta)?;
            let scale = eigenvalues
                .iter()
                .copied()
                .map(f64::abs)
                .fold(0.0, f64::max);
            let roundoff = f64::EPSILON * alpha.len().max(1) as f64 * scale.max(f64::MIN_POSITIVE);
            for (index, (&eigenvalue, &first)) in
                eigenvalues.iter().zip(first_components.iter()).enumerate()
            {
                if !eigenvalue.is_finite() || eigenvalue < -roundoff {
                    return Err(format!(
                        "residual cascade: Schur-spectrum Ritz value {index} is not positive semidefinite ({eigenvalue})"
                    ));
                }
                let weight = start_norm_sq * first * first / SLQ_PROBES as f64;
                if !(weight.is_finite() && weight >= 0.0) {
                    return Err(format!(
                        "residual cascade: invalid Schur-spectrum quadrature weight {weight}"
                    ));
                }
                modes.push(CascadeSpectralMode {
                    // Same reading of the same floor as the dense route: a Ritz
                    // value inside the quadrature's own roundoff is a null
                    // direction, not a small positive mode. Admitting it as
                    // positive lets it set the small-lambda end of
                    // `log_lambda_domain`, which is how the search comes to
                    // demand a solve of `X'WX + λD` at a λ that leaves the
                    // matrix numerically singular.
                    eigenvalue: if eigenvalue > roundoff {
                        eigenvalue
                    } else {
                        0.0
                    },
                    weight,
                });
            }
        }
        Ok(modes)
    }

    /// `beta = D^(-1/2)(b1 - G10 G00^(-1) b0)` and
    /// `anchor_energy = y'Wy - b0' G00^(-1) b0`: the null-eliminated,
    /// penalty-whitened right-hand side and the part of the profiled residual no
    /// lambda can move.
    ///
    /// Identical in exact arithmetic to what [`Self::dense_cascade_spectrum`]
    /// builds inline, but routed through [`Self::matvec`] instead of
    /// `dense_gram_entry`, so it is available past the dense cap. `matvec(0, v)`
    /// applies the FULL `X'WX`, and only its `1`-block rows are read — that is
    /// `G10 (G00^(-1) b0)`, the cross term, with no dense Gram formed.
    fn whitened_residual_rhs(&self, null_chol: &[f64]) -> (Vec<f64>, f64) {
        let q = self.nullity();
        let rank = self.m - q;
        let null_coeff = chol_solve(null_chol, q, &self.rhs[..q]);
        let mut full = vec![0.0; self.m];
        full[..q].copy_from_slice(&null_coeff);
        let mut cross = vec![0.0; self.m];
        self.matvec(0.0, &full, &mut cross);
        let beta = (0..rank)
            .map(|i| (self.rhs[q + i] - cross[q + i]) / self.pen_diag[q + i].sqrt())
            .collect::<Vec<_>>();
        let anchor_energy = self.ytwy
            - self.rhs[..q]
                .iter()
                .zip(null_coeff.iter())
                .map(|(&b, &c)| b * c)
                .sum::<f64>();
        (beta, anchor_energy)
    }

    /// One Golub-Meurant Gauss rule from the leading `steps x steps` block of a
    /// beta-seeded Lanczos run's Jacobi matrix.
    ///
    /// `T_m`'s leading `j x j` block IS `T_j`, the Jacobi matrix the same run
    /// would have produced had it stopped at step `j` — the Lanczos recurrence is
    /// forward. So every nested rule of a single run is available for the cost of
    /// one tridiagonal eigendecomposition, which is what makes the convergence
    /// certificate in [`Self::iterative_residual_spectrum`] free of a second run.
    ///
    /// Nodes are the Ritz values `theta_j`, weights are `||beta||^2 tau_j^2` with
    /// `tau_j` the first component of Ritz vector `j` — exactly the
    /// `(eigenvalue, projected_square)` pair [`CascadeResidualSpectrum`] stores,
    /// so the iterative route populates the same struct and inherits
    /// [`CascadeResidualSpectrum::moments`] unchanged.
    ///
    /// TWO NODE FLOORS, both derived rather than tuned.
    ///
    /// The first is the dense route's, read the same way: a node inside the
    /// decomposition's own roundoff (`eps*m*theta_max`) is a NULL direction of the
    /// whitened design, and a null direction carries no response energy exactly
    /// (`Bv = 0` gives `Zv = 0` and hence `v'beta = 0`).
    ///
    /// The second is on the WEIGHT, and it exists because the first is not
    /// sufficient for a RITZ value. Measured (#2503, `side=14 levels=4`, rank 203,
    /// 96 steps): one node at `theta = 2.65e-11` — `1.8e-12` of `theta_max`, but
    /// 80x ABOVE the eigenvalue floor, so that floor passes it — carrying weight
    /// `8.9e-27 ||beta||^2`. At the bottom of the search domain
    /// (`lambda ~ 2.9e-11`) that single node contributes `w/(theta+lambda)^4` and
    /// `S_4` comes out `3.6e7` RELATIVE off while `S_2` is still right to `6e-9`.
    /// The catastrophe #2503 attributed to quadrature truncation is one node whose
    /// weight is pure roundoff and whose position happens to sit under `lambda`.
    ///
    /// The floor is the SQUARE of the roundoff in the quantity being squared. A
    /// Ritz vector's first component comes out of a QL sweep that accumulates
    /// `O(m)` plane rotations, so a component that should be zero comes out at
    /// `~eps*m`; its square, times the mass, is `(eps*m)^2 ||beta||^2`. Measured
    /// against that prediction across three fixtures and eleven step budgets:
    /// every spurious weight landed in `[1e-258, 4e-27] ||beta||^2` — i.e. at or
    /// below `(eps*m)^2` — while the smallest GENUINE mass the rules carried was
    /// `4.1e-14 ||beta||^2`, thirteen orders above the floor. The earlier
    /// `eps*m*||beta||^2` floor did over-drop that genuine mass, and the symptom
    /// was diagnostic: it GREW with `m`, so a longer run dropped more, which no
    /// convergent process can be right about.
    fn residual_gauss_rule(
        &self,
        run: &SchurLanczos,
        steps: usize,
        anchor_energy: f64,
        measure_mass: f64,
    ) -> Result<ResidualGaussRule, String> {
        let steps = steps.min(run.alpha.len());
        let (ritz, first_components) = symmetric_tridiagonal_eigen(
            &run.alpha[..steps],
            &run.beta[..steps.saturating_sub(1)],
        )?;
        let scale = ritz.iter().copied().map(f64::abs).fold(0.0, f64::max);
        let count = steps.max(1) as f64;
        let eigenvalue_floor = f64::EPSILON * count * scale.max(f64::MIN_POSITIVE);
        let component_roundoff = f64::EPSILON * count;
        let mass_floor = component_roundoff * component_roundoff * measure_mass;

        let mut eigenvalue = Vec::with_capacity(ritz.len());
        let mut projected_square = Vec::with_capacity(ritz.len());
        let mut total_mass = 0.0_f64;
        let mut dropped_mass = 0.0_f64;
        // A Ritz value is not an eigenvalue: it comes out of a Lanczos recurrence
        // and a QL sweep whose backward error grows with the step count, so the
        // threshold at which negativity stops being arithmetic and starts being a
        // statement about `B` is NOT the eigenvalue floor. Below `sqrt(eps)*theta_max`
        // — the resolution this module works at throughout, the same one
        // `log_lambda_domain_from_modes` pads with — a negative Ritz value is
        // indistinguishable from the zero it is approximating, and is clamped to
        // the null direction it represents. Above it, the penalty-whitened Schur
        // complement is genuinely indefinite, which is a defect and not roundoff.
        let indefinite = f64::EPSILON.sqrt() * scale.max(f64::MIN_POSITIVE);
        for (index, (&theta, &first)) in ritz.iter().zip(first_components.iter()).enumerate() {
            if !theta.is_finite() || theta < -indefinite {
                return Err(format!(
                    "residual cascade: profiled-residual Ritz value {index} of {steps} is not \
                     positive semidefinite ({theta}); the whitened Schur complement is \
                     indefinite beyond the run's own resolution {indefinite}"
                ));
            }
            let weight = run.start_norm_sq * first * first;
            if !(weight.is_finite() && weight >= 0.0) {
                return Err(format!(
                    "residual cascade: invalid profiled-residual quadrature weight {weight}"
                ));
            }
            total_mass += weight;
            if theta <= eigenvalue_floor || weight <= mass_floor {
                dropped_mass += weight;
                eigenvalue.push(0.0);
                projected_square.push(0.0);
            } else {
                eigenvalue.push(theta);
                projected_square.push(weight);
            }
        }
        // The weights of ANY Gauss rule sum to the measure's total mass, so
        // `sum_j w_j = ||beta||^2` is a free self-check on the whole Jacobi
        // pipeline — the recurrence, the reorthogonalization and the tridiagonal
        // eigensolver at once. It is charged against the accumulated rounding of
        // the sum it checks, not against a tuned slack.
        let mass_defect = ((total_mass - measure_mass) / measure_mass).abs();
        let mass_tolerance = 8.0 * f64::EPSILON * count;
        if !(mass_defect <= mass_tolerance) {
            return Err(format!(
                "residual cascade: profiled-residual quadrature weights sum to {total_mass} \
                 against the measure mass {measure_mass} (relative defect {mass_defect} over \
                 tolerance {mass_tolerance}); the Gauss rule for a measure of mass m has weights \
                 summing to m, so the Jacobi matrix or its eigendecomposition is wrong"
            ));
        }
        let modes = eigenvalue.len();
        Ok(ResidualGaussRule {
            spectrum: CascadeResidualSpectrum {
                eigenvalue,
                penalty: vec![1.0; modes],
                projected_square,
                anchor_energy: [anchor_energy],
            },
            mass_defect,
            dropped_mass_fraction: dropped_mass / measure_mass,
        })
    }

    /// The profiled residual's spectral form past the dense cap, by Golub-Meurant
    /// quadrature of the SAME Schur operator the determinant sweep runs on — or
    /// `None` when no rule earned the right to be used.
    ///
    /// `S_k(lambda) = beta'(B + lambda I)^-k beta` is `integral (theta +
    /// lambda)^-k dmu(theta)` for the measure `mu` that `beta` induces on
    /// `spec(B)`. One Lanczos run seeded with `beta` (rather than with a
    /// Rademacher probe) gives the Jacobi matrix of the `m`-node Gauss rule for
    /// that measure; see [`Self::residual_gauss_rule`] for the rule itself and its
    /// two node floors. Where a rule is admitted, the whole score is solve-free at
    /// every `lambda`, exactly as on the dense route after #2455 — and the
    /// domain-endpoint `lambda` that no PCG can solve (#2503) is never solved at.
    ///
    /// WHAT ADMITS A RULE, and why it is not the step count.
    ///
    /// A Gauss rule accurate in VALUE need not be accurate in its
    /// lambda-DERIVATIVES: the nodes are placed to integrate one kernel, and
    /// `(theta + lambda)^-k` grows more peaked at the bottom of the spectrum with
    /// every power of `k`. `S_2, S_3, S_4` ARE the score's first three
    /// `log lambda` derivatives, so a rule may not be admitted on `R` alone. Two
    /// admissions are available and both are properties of the run, not of a
    /// calibrated budget:
    ///
    /// 1. THE KRYLOV SPACE CLOSED (`SchurLanczos::invariant`). Then
    ///    `(B + lambda I)^-k beta` lies inside `K_m` for every `k` and every
    ///    `lambda`, and the rule reproduces the spectral sum outright.
    /// 2. THE NESTED LADDER HAS CONTRACTED. Every Gauss rule for a completely
    ///    monotone kernel UNDER-estimates its integral (the `(2m)`-th derivative
    ///    of `(theta+lambda)^-k` is positive, so the standard error
    ///    representation is one-signed), so the rules at `m/4`, `m/2`, `m` — free,
    ///    since `T_m`'s leading block IS `T_j` — rise toward `S_k` with all their
    ///    gaps of one sign, and the remaining error is the tail of those gaps.
    ///    [`residual_quadrature_tail_estimate`] extrapolates that tail
    ///    geometrically and REFUSES when the last two gaps do not contract, which
    ///    is the stagnation case a bare two-rule agreement test cannot see. The
    ///    estimate must fall below the resolution the score search itself works at
    ///    — `sqrt(eps)`, the same constant `log_lambda_domain_from_modes` pads
    ///    with and `fit_reml` refines to. Measured against the exact dense
    ///    eigenbasis at every budget on this ladder, over three designs: where the
    ///    estimate passed, the rule was within `1e-12`.
    ///
    /// The budget GROWS geometrically until one of those fires, so the accepted
    /// rule is the smallest that passes rather than the largest affordable. That
    /// matters in both directions: past roughly 60% of the penalized rank the run
    /// starts producing near-null ghost nodes (measured: rank 473, from 256 steps
    /// on), so the cheapest passing rule is also the cleanest one.
    ///
    /// Refuses by returning `None` rather than by erroring: the caller then solves
    /// at every `lambda`, which is what shipped, and the certificate says so.
    fn iterative_residual_spectrum(
        &self,
        null_chol: &[f64],
        domain: (f64, f64),
    ) -> Result<(Option<CascadeResidualSpectrum>, ResidualQuadratureCertificate), String> {
        let rank = self.m - self.nullity();
        let ceiling = self.residual_krylov_ceiling();
        let budget = self.residual_quadrature_budget();
        let target = f64::EPSILON.sqrt();
        let (beta, anchor_energy) = self.whitened_residual_rhs(null_chol);
        if !(anchor_energy.is_finite() && beta.iter().all(|value| value.is_finite())) {
            return Err(format!(
                "residual cascade: non-finite whitened residual right-hand side (anchor \
                 {anchor_energy})"
            ));
        }
        let measure_mass = beta.iter().map(|value| value * value).sum::<f64>();
        if !(measure_mass > 0.0) {
            // No response energy outside the polynomial null space: the profiled
            // residual is the anchor at every lambda. A zero measure is exactly
            // integrated by the empty rule, so this is certified, not degraded.
            return Ok((
                Some(CascadeResidualSpectrum {
                    eigenvalue: Vec::new(),
                    penalty: Vec::new(),
                    projected_square: Vec::new(),
                    anchor_energy: [anchor_energy],
                }),
                ResidualQuadratureCertificate {
                    steps: 0,
                    coarse_steps: 0,
                    rank,
                    budget,
                    relative_tail: 0.0,
                    tail_estimate: 0.0,
                    target,
                    certified: true,
                    mass_defect: 0.0,
                    dropped_mass_fraction: 0.0,
                },
            ));
        }
        let mut steps = SLQ_LANCZOS_STEPS.min(budget);
        loop {
            let run = self.schur_lanczos(null_chol, &beta, steps, ceiling)?;
            let taken = run.alpha.len();
            let fine = self.residual_gauss_rule(&run, taken, anchor_energy, measure_mass)?;
            let coarse_steps = taken / 4;
            let tail_estimate = if run.invariant {
                // A closed Krylov space makes the rule exact for every kernel;
                // there is nothing left for a nested comparison to add.
                0.0
            } else if coarse_steps == 0 {
                // Fewer than four nodes leaves no nested ladder to extrapolate
                // along, and "no evidence" is not "converged".
                f64::INFINITY
            } else {
                let mid =
                    self.residual_gauss_rule(&run, taken / 2, anchor_energy, measure_mass)?;
                let coarse =
                    self.residual_gauss_rule(&run, coarse_steps, anchor_energy, measure_mass)?;
                residual_quadrature_tail_estimate(
                    &fine.spectrum,
                    &mid.spectrum,
                    &coarse.spectrum,
                    taken,
                    domain,
                )?
            };
            let certificate = ResidualQuadratureCertificate {
                steps: taken,
                coarse_steps: if run.invariant { 0 } else { coarse_steps },
                rank,
                budget,
                relative_tail: run.tail / run.spectral_scale.max(f64::MIN_POSITIVE),
                tail_estimate,
                target,
                certified: run.invariant || tail_estimate <= target,
                mass_defect: fine.mass_defect,
                dropped_mass_fraction: fine.dropped_mass_fraction,
            };
            if certificate.certified {
                return Ok((Some(fine.spectrum), certificate));
            }
            if taken >= budget {
                return Ok((None, certificate));
            }
            steps = steps.saturating_mul(2).min(budget);
        }
    }

    /// Lanczos steps the profiled-residual quadrature may grow to past the dense
    /// cap.
    ///
    /// Not an accuracy dial — a rule is admitted by its own convergence, not by
    /// reaching a step count — so this bounds how large a Krylov space we are
    /// willing to REORTHOGONALIZE before declining to certify. The binding
    /// resource is the basis itself (`steps x rank` doubles held live), so the
    /// bound is stated as that memory and the number in the code is the one being
    /// reasoned about.
    ///
    /// Two structural ceilings apply on top. `rank` is the obvious one. The other
    /// is `n - nullity`: `B = Z'WZ` for an `n x rank` whitened design, so
    /// `rank(B) <= n - nullity`, and in exact arithmetic `K_m(B, beta)` cannot
    /// grow past that dimension. Steps beyond it are spent entirely inside the
    /// numerical null space, which is where the ghost nodes the weight floor has
    /// to clean up come from.
    fn residual_quadrature_budget(&self) -> usize {
        let rank = self.m - self.nullity();
        let by_memory =
            RESIDUAL_QUADRATURE_BASIS_BYTES / (size_of::<f64>() * rank.max(1)).max(1);
        by_memory
            .max(SLQ_LANCZOS_STEPS)
            .min(self.residual_krylov_ceiling())
    }

    /// Largest dimension `K_m(B, beta)` can reach, so reaching it proves the space
    /// is `B`-invariant and the Gauss rule EXACT.
    ///
    /// `B = D^(-1/2)(G11 - G10 G00^(-1) G01) D^(-1/2) = Z'WZ` with
    /// `W^(1/2) Z = (I - P) W^(1/2) X_1` and `P` the `W`-projector onto the
    /// polynomial block, so `rank(B) <= n - nullity`; and `beta = Z'Wy` lies in
    /// `range(B)`, so the Krylov space cannot leave it. On a cascade whose net
    /// fills the bounding BOX rather than the data cloud this is the binding bound
    /// by an order of magnitude — measured on the `n = 800` 2-D fixture at
    /// refinement level 6: `rank = 7387` against `n - nullity = 797`, because 89%
    /// of the columns are void-filling centres the data cannot pin. Reading the
    /// ceiling as `rank` there made the invariance test unreachable and sent the
    /// route back to the solve at full budget.
    fn residual_krylov_ceiling(&self) -> usize {
        let rank = self.m - self.nullity();
        rank.min(self.y.len().saturating_sub(self.nullity())).max(1)
    }

    fn reml_profile(&self) -> Result<CascadeRemlProfile<'_>, String> {
        let (null_chol, null_logdet) = self.null_gram_factor()?;
        let (modes, residual) = if self.dense_gram.is_some() {
            let (modes, spectrum) = self.dense_cascade_spectrum(&null_chol)?;
            (modes, CascadeResidualForm::Spectral(spectrum))
        } else {
            let modes = self.iterative_cascade_spectrum(&null_chol)?;
            // The residual's convergence certificate is charged over exactly the
            // interval the search will visit, so the determinant modes — which
            // define that interval — are built first.
            let domain = log_lambda_domain_from_modes(&modes)?;
            let (spectrum, certificate) = self.iterative_residual_spectrum(&null_chol, domain)?;
            (
                modes,
                CascadeResidualForm::Quadrature {
                    spectrum,
                    certificate,
                },
            )
        };
        Ok(CascadeRemlProfile {
            core: self,
            null_logdet,
            modes,
            residual,
        })
    }

    /// Scale a raw point into shifted metric coordinates.
    fn scale_point(&self, x: &[f64]) -> [f64; 3] {
        let mut z = [0.0_f64; 3];
        for a in 0..self.dim {
            z[a] = self.metric[a] * x[a] - self.z_lo[a];
        }
        z
    }

    /// Sparse basis row at a scaled point: polynomial layer then every bump
    /// whose support covers it, as (column, value) pairs sorted by column.
    fn basis_row_scaled(&self, z: &[f64; 3]) -> Vec<(usize, f64)> {
        let mut row = Vec::with_capacity(self.dim + 1 + self.levels.len() * 8);
        row.push((0, 1.0));
        for a in 0..self.dim {
            row.push((a + 1, 2.0 * z[a] / self.z_range[a] - 1.0));
        }
        for level in &self.levels {
            let start = row.len();
            level.grid.for_neighbors(z, |j| {
                let c = &level.centers[j as usize];
                let r = dist2(z, c, self.dim).sqrt() / level.delta;
                let v = wendland(r);
                if v > 0.0 {
                    row.push((level.col_offset + j as usize, v));
                }
            });
            row[start..].sort_unstable_by_key(|&(col, _)| col);
        }
        row
    }

    /// `out = (X'WX + λD)·v` through the CSR rows: O(nnz).
    fn matvec(&self, lambda: f64, v: &[f64], out: &mut [f64]) {
        for (o, (&d, &x)) in out.iter_mut().zip(self.pen_diag.iter().zip(v.iter())) {
            *o = lambda * d * x;
        }
        for i in 0..self.w.len() {
            let lo = self.row_ptr[i];
            let hi = self.row_ptr[i + 1];
            let mut t = 0.0;
            for e in lo..hi {
                t += self.vals[e] * v[self.col_idx[e] as usize];
            }
            t *= self.w[i];
            for e in lo..hi {
                out[self.col_idx[e] as usize] += self.vals[e] * t;
            }
        }
    }

    /// Jacobi / level-diagonal preconditioner: `diag(X'WX) + λ·diag(λD)`.
    /// Levels share a constant prior weight, so this IS the level-block
    /// (BPX-flavored) diagonal in the multilevel frame.
    /// Coarse column count of the additive-Schwarz coarse space at `λ`: the
    /// polynomial layer plus the longest prefix of data-dominated levels
    /// (`λ d_l < COARSE_DOMINANCE · median diag(X'WX) over the level`), with the
    /// two coarsest levels always deflated and the total capped at
    /// [`COARSE_SPACE_MAX`]. Because `d_l` rises while the per-level data weight
    /// falls, the data-dominated set is a contiguous prefix, so one scan from the
    /// coarsest level finds the cut. (See [`COARSE_DOMINANCE`].)
    fn coarse_space_cols(&self, lambda: f64) -> usize {
        let mut ncoarse = self.nullity();
        let mut buf: Vec<f64> = Vec::new();
        for (li, level) in self.levels.iter().enumerate() {
            let a = level.col_offset;
            let b = a + level.centers.len();
            if b <= a {
                continue;
            }
            if b > COARSE_SPACE_MAX {
                break;
            }
            let dominated = if li < MIN_COARSE_LEVELS {
                true
            } else {
                buf.clear();
                buf.extend_from_slice(&self.gram_diag[a..b]);
                buf.sort_unstable_by(|x, y| x.partial_cmp(y).unwrap());
                let gram_median = buf[buf.len() / 2];
                lambda * level.weight < COARSE_DOMINANCE * gram_median
            };
            if dominated {
                ncoarse = b;
            } else {
                break;
            }
        }
        // Keep at least one fine column so the split is well-defined; if every
        // level is coarse the iterative route is degenerate anyway and the dense
        // route would have been taken, but guard regardless.
        let ncoarse = ncoarse.min(self.m);
        // Debug-only coarse-space layout trace (#1032). Gated on the log level so
        // the per-call string build stays out of this preconditioner hot path,
        // and routed through `log` (an `eprintln!` here trips the src banned-macro
        // gate and broke the build).
        if log::log_enabled!(log::Level::Debug) {
            let mut s = String::new();
            for (li, level) in self.levels.iter().enumerate() {
                let a = level.col_offset;
                let b = a + level.centers.len();
                let mut buf: Vec<f64> = self.gram_diag[a..b].to_vec();
                buf.sort_unstable_by(|x, y| x.partial_cmp(y).unwrap());
                let med = if buf.is_empty() {
                    0.0
                } else {
                    buf[buf.len() / 2]
                };
                let coarse = b <= ncoarse;
                s.push_str(&format!(
                    " L{li}[{}c off{a} w={:.2e} λw={:.2e} med={:.2e} {}]",
                    level.centers.len(),
                    level.weight,
                    lambda * level.weight,
                    med,
                    if coarse { "C" } else { "F" }
                ));
            }
            log::debug!(
                "[1032-COARSE] λ={lambda:.3e} m={} ncoarse={ncoarse} cap={COARSE_SPACE_MAX}{s}",
                self.m
            );
        }
        ncoarse
    }

    /// Build the coarse-space additive-Schwarz preconditioner at `λ`: assemble
    /// and factor the coarse block `A_CC` from the CSR (coarse columns are the
    /// prefix `[0, ncoarse)`, and each CSR row is column-sorted, so a row's
    /// coarse entries are its leading run), then the Jacobi diagonal on the fine
    /// tail. `O(n · q_C²) + O(ncoarse³)` — paid once per `λ`, not per CG step.
    fn build_preconditioner(&self, lambda: f64) -> Result<Preconditioner, String> {
        let m = self.m;
        let nc = self.coarse_space_cols(lambda);
        let mut acc = vec![0.0_f64; nc * nc];
        for i in 0..self.w.len() {
            let lo = self.row_ptr[i];
            let hi = self.row_ptr[i + 1];
            // Leading run of coarse columns (CSR rows are column-sorted).
            let mut end = lo;
            while end < hi && (self.col_idx[end] as usize) < nc {
                end += 1;
            }
            for ea in lo..end {
                let ca = self.col_idx[ea] as usize;
                let va = self.w[i] * self.vals[ea];
                for eb in ea..end {
                    let cb = self.col_idx[eb] as usize;
                    acc[ca * nc + cb] += va * self.vals[eb];
                }
            }
        }
        for i in 0..nc {
            for j in i + 1..nc {
                acc[j * nc + i] = acc[i * nc + j];
            }
        }
        for i in 0..nc {
            acc[i * nc + i] += lambda * self.pen_diag[i];
        }
        let coarse_logdet = cholesky_logdet(&mut acc, nc)?;
        let mut inv_fine = Vec::with_capacity(m - nc);
        let mut inv_sqrt_fine = Vec::with_capacity(m - nc);
        let mut fine_logdet = 0.0;
        for j in nc..m {
            let p = self.gram_diag[j] + lambda * self.pen_diag[j];
            if !(p.is_finite() && p > EIG_FLOOR) {
                return Err(format!(
                    "residual cascade: non-positive preconditioner diagonal {p} at column {j}"
                ));
            }
            inv_fine.push(1.0 / p);
            inv_sqrt_fine.push(1.0 / p.sqrt());
            fine_logdet += p.ln();
        }
        Ok(Preconditioner {
            ncoarse: nc,
            coarse_chol: acc,
            coarse_logdet,
            inv_fine,
            inv_sqrt_fine,
            fine_logdet,
        })
    }

    /// Preconditioned CG on `(X'WX + λD)c = b` to relative residual CG_RTOL.
    /// Returns the solution with its backward-error certificate.
    fn pcg(
        &self,
        lambda: f64,
        b: &[f64],
        warm: Option<&[f64]>,
    ) -> Result<(Vec<f64>, f64, usize), String> {
        let prec = self.build_preconditioner(lambda)?;
        self.pcg_with(lambda, &prec, b, warm)
    }

    /// [`Core::pcg`] against a preconditioner the caller already built at this
    /// λ. The preconditioner depends on λ and nothing else, so a caller with
    /// several right-hand sides at one λ builds it once.
    fn pcg_with(
        &self,
        lambda: f64,
        prec: &Preconditioner,
        b: &[f64],
        warm: Option<&[f64]>,
    ) -> Result<(Vec<f64>, f64, usize), String> {
        let m = self.m;
        let b_norm = b.iter().map(|v| v * v).sum::<f64>().sqrt();
        if b_norm == 0.0 {
            return Ok((vec![0.0; m], 0.0, 0));
        }
        let mut zv = vec![0.0; m];
        let mut x = match warm {
            Some(x0) => {
                if x0.len() != m {
                    return Err(format!(
                        "residual cascade: warm-start length {} != system size {m}",
                        x0.len()
                    ));
                }
                x0.to_vec()
            }
            None => {
                prec.solve(b, &mut zv);
                zv.clone()
            }
        };
        let mut r = vec![0.0; m];
        self.matvec(lambda, &x, &mut r);
        for (ri, &bi) in r.iter_mut().zip(b.iter()) {
            *ri = bi - *ri;
        }
        prec.solve(&r, &mut zv);
        let mut p_dir = zv.clone();
        let mut rz: f64 = r.iter().zip(zv.iter()).map(|(&a, &c)| a * c).sum();
        let mut ap = vec![0.0; m];
        let max_iters = CG_MAX_ITERS;
        for iter in 0..max_iters {
            let r_norm = r.iter().map(|v| v * v).sum::<f64>().sqrt();
            if r_norm <= CG_RTOL * b_norm {
                return Ok((x, r_norm / b_norm, iter));
            }
            self.matvec(lambda, &p_dir, &mut ap);
            let pap: f64 = p_dir.iter().zip(ap.iter()).map(|(&a, &c)| a * c).sum();
            if !(pap.is_finite() && pap > 0.0) {
                return Err(format!(
                    "residual cascade: CG curvature breakdown (p'Ap = {pap}) at iteration {iter}"
                ));
            }
            let alpha = rz / pap;
            for j in 0..m {
                x[j] += alpha * p_dir[j];
                r[j] -= alpha * ap[j];
            }
            prec.solve(&r, &mut zv);
            let rz_new: f64 = r.iter().zip(zv.iter()).map(|(&a, &c)| a * c).sum();
            let beta = rz_new / rz;
            rz = rz_new;
            for j in 0..m {
                p_dir[j] = zv[j] + beta * p_dir[j];
            }
        }
        Err(format!(
            "residual cascade: CG failed to reach relative residual {CG_RTOL} within \
             {CG_MAX_ITERS} iterations (the coarse-space additive-Schwarz preconditioner should \
             make this n-independent; this indicates a degenerate design)"
        ))
    }

    /// Expand the cached dense upper Gram + λD into a full symmetric matrix.
    fn dense_system(&self, lambda: f64) -> Option<Vec<f64>> {
        let gram = self.dense_gram.as_ref()?;
        let m = self.m;
        let mut a = vec![0.0; m * m];
        for i in 0..m {
            for j in i..m {
                let mut v = gram[i * m + j];
                if i == j {
                    v += lambda * self.pen_diag[i];
                }
                a[i * m + j] = v;
                a[j * m + i] = v;
            }
        }
        Some(a)
    }

    /// Exact log-determinant of `X'WX + λD` by dense Cholesky. Errors when
    /// the design is past the dense sizing cap.
    fn logdet_dense(&self, lambda: f64) -> Result<f64, String> {
        let mut a = self.dense_system(lambda).ok_or_else(|| {
            format!(
                "residual cascade: dense logdet requested past the sizing cap \
                 (m = {} > {DENSE_GRAM_MAX})",
                self.m
            )
        })?;
        cholesky_logdet(&mut a, self.m)
    }

    /// SLQ log-determinant: exact control variate `log|P|` (the coarse-space
    /// additive-Schwarz preconditioner's own log-determinant — `log|A_CC|` plus
    /// the fine Jacobi `Σ_F log A_jj`) plus stochastic Lanczos quadrature for
    /// `tr log(R⁻¹ A R⁻ᵀ)`, `P = R Rᵀ`, on fixed deterministic Rademacher probes
    /// shared across every λ (common random numbers ⇒ the REML criterion is a
    /// smooth deterministic function of λ). The same coarse deflation that makes
    /// the PCG iteration count n-independent makes `R⁻¹ A R⁻ᵀ` uniformly
    /// conditioned, so the Lanczos quadrature converges in a depth-independent
    /// number of steps too.
    fn logdet_slq(&self, lambda: f64) -> Result<f64, String> {
        let m = self.m;
        let prec = self.build_preconditioner(lambda)?;
        let logdet = prec.logdet();
        // M·v = R⁻¹ A R⁻ᵀ v (eigenvalues of P^{−1/2} A P^{−1/2}) without forming M.
        let mut scratch_in = vec![0.0; m];
        let mut scratch_out = vec![0.0; m];
        let mut vbuf = vec![0.0; m];
        let mut trace_est = 0.0;
        let steps = SLQ_LANCZOS_STEPS.min(m);
        let mut basis: Vec<Vec<f64>> = Vec::with_capacity(steps);
        for probe in 0..SLQ_PROBES {
            let mut rng =
                SplitMix64::new(RNG_SEED ^ (probe as u64).wrapping_mul(0xD134_2543_DE82_EF95));
            let mut q = vec![0.0; m];
            for qj in q.iter_mut() {
                *qj = rng.next_sign();
            }
            let z_norm2 = m as f64;
            let inv_norm = 1.0 / (m as f64).sqrt();
            for qj in q.iter_mut() {
                *qj *= inv_norm;
            }
            // Lanczos with full reorthogonalization.
            basis.clear();
            let mut alpha = Vec::with_capacity(steps);
            let mut beta: Vec<f64> = Vec::with_capacity(steps);
            let mut q_prev: Option<Vec<f64>> = None;
            for _step in 0..steps {
                // v = R⁻¹ A R⁻ᵀ q.
                prec.apply_r_inv_t(&q, &mut scratch_in);
                self.matvec(lambda, &scratch_in, &mut scratch_out);
                prec.apply_r_inv(&scratch_out, &mut vbuf);
                let mut v: Vec<f64> = vbuf.clone();
                let a: f64 = v.iter().zip(q.iter()).map(|(&x, &y)| x * y).sum();
                alpha.push(a);
                for j in 0..m {
                    v[j] -= a * q[j];
                }
                if let Some(prev) = &q_prev {
                    let b_prev = beta.last().copied().unwrap_or(0.0);
                    for j in 0..m {
                        v[j] -= b_prev * prev[j];
                    }
                }
                // Full reorthogonalization against the stored basis.
                basis.push(q.clone());
                for qb in &basis {
                    let proj: f64 = v.iter().zip(qb.iter()).map(|(&x, &y)| x * y).sum();
                    for j in 0..m {
                        v[j] -= proj * qb[j];
                    }
                }
                let b: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
                if !(b.is_finite()) {
                    return Err("residual cascade: Lanczos breakdown (non-finite norm)".into());
                }
                if b < 1e-13 {
                    break;
                }
                beta.push(b);
                q_prev = Some(std::mem::replace(&mut q, v));
                for qj in q.iter_mut() {
                    *qj /= b;
                }
            }
            beta.truncate(alpha.len().saturating_sub(1));
            let (theta, tau) = symmetric_tridiagonal_eigen(&alpha, &beta)?;
            let mut quad = 0.0;
            for (&t, &w0) in theta.iter().zip(tau.iter()) {
                if !(t.is_finite() && t > EIG_FLOOR) {
                    return Err(format!(
                        "residual cascade: non-positive Ritz value {t} in SLQ (system not PD)"
                    ));
                }
                quad += w0 * w0 * t.ln();
            }
            trace_est += z_norm2 * quad;
        }
        Ok(logdet + trace_est / SLQ_PROBES as f64)
    }

    /// Log-determinant through the route the sizing contract picks.
    fn logdet(&self, lambda: f64) -> Result<(f64, LogdetMethod), String> {
        if self.dense_gram.is_some() {
            Ok((self.logdet_dense(lambda)?, LogdetMethod::DenseExact))
        } else {
            Ok((self.logdet_slq(lambda)?, LogdetMethod::Slq))
        }
    }

    /// The coefficient solver at a FIXED λ, obtained once so that several
    /// right-hand sides can share one factorization or one preconditioner.
    ///
    /// Neither `A = X'WX + λD` nor its preconditioner depends on the right-hand
    /// side, so a caller that needs two solves at the same λ must not pay two
    /// O(m³) Cholesky factorizations — or two coarse-block assemblies — for it.
    /// [`CascadeResidualForm::Solved`] needs exactly that pair (`A⁻¹b` and then
    /// `A⁻¹Dc`), and going through `solve_coeff`/`pcg` twice rebuilt the
    /// identical operator both times.
    fn coeff_solver(&self, lambda: f64) -> Result<CoeffSolver<'_>, String> {
        if let Some(l) = &self.predict_chol {
            return Ok(CoeffSolver::Cached(l));
        }
        if let Some(mut a) = self.dense_system(lambda) {
            cholesky_logdet(&mut a, self.m)?;
            return Ok(CoeffSolver::Factored(a));
        }
        Ok(CoeffSolver::Iterative(self.build_preconditioner(lambda)?))
    }

    /// Coefficient solve at λ: dense Cholesky when cached, else certified PCG.
    fn solve_coeff(
        &self,
        lambda: f64,
        b: &[f64],
        warm: Option<&[f64]>,
    ) -> Result<(Vec<f64>, f64, usize), String> {
        // A core rebuilt from a persisted state carries no training design, only
        // the factored precision `L` of `A = X'WX + λD` at the fit's λ. Replay
        // the solve through it (exact — predict always solves at that same λ).
        if let Some(l) = &self.predict_chol {
            return Ok((chol_solve(l, self.m, b), 0.0, 0));
        }
        if let Some(mut a) = self.dense_system(lambda) {
            cholesky_logdet(&mut a, self.m)?;
            return Ok((chol_solve(&a, self.m, b), 0.0, 0));
        }
        self.pcg(lambda, b, warm)
    }

    /// Assemble the lower Cholesky factor `L` of `A = X'WX + λD` as a dense
    /// `m × m` row-major matrix — the factored precision a persisted predict
    /// replays its posterior-variance solve through. Uses the cached dense Gram
    /// when present; otherwise scatters the CSR row outer products into the
    /// upper triangle (one O(nnz·q) pass), the same assembly `build` uses under
    /// the sizing cap, just without the cap. Factoring is O(m³) — paid once at
    /// snapshot time, not per predict.
    fn assemble_predict_factor(&self, lambda: f64) -> Result<Vec<f64>, String> {
        let m = self.m;
        let mut a = vec![0.0_f64; m * m];
        if let Some(gram) = &self.dense_gram {
            for i in 0..m {
                for j in i..m {
                    let v = gram[i * m + j];
                    a[i * m + j] = v;
                    a[j * m + i] = v;
                }
            }
        } else {
            for i in 0..self.w.len() {
                let lo = self.row_ptr[i];
                let hi = self.row_ptr[i + 1];
                for ea in lo..hi {
                    let ca = self.col_idx[ea] as usize;
                    let va = self.w[i] * self.vals[ea];
                    for eb in ea..hi {
                        let cb = self.col_idx[eb] as usize;
                        a[ca * m + cb] += va * self.vals[eb];
                    }
                }
            }
            // Mirror the upper triangle into the lower.
            for i in 0..m {
                for j in i + 1..m {
                    a[j * m + i] = a[i * m + j];
                }
            }
        }
        for (i, d) in self.pen_diag.iter().enumerate() {
            a[i * m + i] += lambda * d;
        }
        cholesky_logdet(&mut a, m)?;
        Ok(a)
    }

    /// Penalized residual quadratic at a solution: `y'Wy − c'X'Wy`.
    fn rss_pen(&self, coeff: &[f64]) -> f64 {
        let mut quad = 0.0;
        for (c, r) in coeff.iter().zip(self.rhs.iter()) {
            quad += c * r;
        }
        self.ytwy - quad
    }

    /// Number of unpenalized (polynomial) columns.
    fn nullity(&self) -> usize {
        self.dim + 1
    }

    /// Working residual `r_i = y_i − (Xc)_i`.
    fn residuals(&self, coeff: &[f64]) -> Vec<f64> {
        let n = self.y.len();
        let mut r = Vec::with_capacity(n);
        for i in 0..n {
            let mut fit = 0.0;
            for e in self.row_ptr[i]..self.row_ptr[i + 1] {
                fit += self.vals[e] * coeff[self.col_idx[e] as usize];
            }
            r.push(self.y[i] - fit);
        }
        r
    }
}

// ──────────────────── symmetric tridiagonal eigensolver ─────────────────────

/// Eigenvalues and FIRST eigenvector components of a symmetric tridiagonal
/// matrix (diag `d`, off-diagonal `e`), by implicit-shift QL with the
/// first-row vector carried through the rotations — exactly what Lanczos
/// quadrature needs.
fn symmetric_tridiagonal_eigen(d: &[f64], e: &[f64]) -> Result<(Vec<f64>, Vec<f64>), String> {
    let n = d.len();
    if n == 0 {
        return Ok((Vec::new(), Vec::new()));
    }
    let mut diag = d.to_vec();
    let mut off = vec![0.0; n];
    off[..n - 1].copy_from_slice(&e[..n - 1]);
    let mut first = vec![0.0; n];
    first[0] = 1.0;
    for l in 0..n {
        let mut iter = 0;
        loop {
            // Find a negligible off-diagonal to split at.
            let mut msplit = n - 1;
            for mm in l..n - 1 {
                let dd = diag[mm].abs() + diag[mm + 1].abs();
                if off[mm].abs() <= f64::EPSILON * dd {
                    msplit = mm;
                    break;
                }
            }
            if msplit == l {
                break;
            }
            iter += 1;
            if iter > 60 {
                return Err("residual cascade: tridiagonal QL failed to converge".into());
            }
            let mut g = (diag[l + 1] - diag[l]) / (2.0 * off[l]);
            let mut r = g.hypot(1.0);
            g = diag[msplit] - diag[l] + off[l] / (g + r.copysign(g));
            let (mut s, mut c) = (1.0, 1.0);
            let mut p = 0.0;
            let mut broke_early = false;
            for i in (l..msplit).rev() {
                let mut f = s * off[i];
                let b = c * off[i];
                r = f.hypot(g);
                off[i + 1] = r;
                if r == 0.0 {
                    diag[i + 1] -= p;
                    off[msplit] = 0.0;
                    broke_early = true;
                    break;
                }
                s = f / r;
                c = g / r;
                g = diag[i + 1] - p;
                r = (diag[i] - g) * s + 2.0 * c * b;
                p = s * r;
                diag[i + 1] = g + p;
                g = c * r - b;
                // Carry the first-row eigenvector components.
                f = first[i + 1];
                first[i + 1] = s * first[i] + c * f;
                first[i] = c * first[i] - s * f;
            }
            if broke_early {
                continue;
            }
            diag[l] -= p;
            off[l] = g;
            off[msplit] = 0.0;
        }
    }
    Ok((diag, first))
}

// ───────────────────────────── net construction ─────────────────────────────

/// Extend a nested net to covering radius `h` over the DOMAIN: first every data
/// point further than `h` from the (seeded) net becomes a new center, then every
/// cell of the `h`-grid over the bounding box `[0, box_hi]` whose centre is not
/// yet within `h` of the net is filled with a synthetic center. O((n + box
/// cells)·3^d). Returns the new centers.
///
/// Covering the box, not merely the data cloud, is what the multilevel Wendland
/// norm-equivalence (Narcowich–Ward inverse estimates + Le Gia–Wendland
/// multilevel stability) actually requires: the nested centres must be
/// quasi-uniform over the domain Ω. In data-dense regions every cell is already
/// covered by a data center, so the fill is a no-op there; in a data void it
/// plants the fine centres whose coefficients carry no data and revert to the
/// prior — the mechanism by which the posterior mean bridges a gap (coarse
/// data-pinned bumps) while the posterior variance GROWS into it (fine void
/// bumps the data cannot pin). The synthetic centres carry (almost) no data
/// rows, so their Gram diagonal is ~0 and they land in the penalty-dominated
/// fine block where the Jacobi preconditioner is exact — they neither perturb
/// the coarse factorization nor the n-independent iteration count.
fn extend_net(
    net: &mut Vec<[f64; 3]>,
    points: &[[f64; 3]],
    dim: usize,
    h: f64,
    box_hi: &[f64; 3],
) -> Vec<[f64; 3]> {
    let mut grid = HashGrid::new(h, dim);
    for (idx, c) in net.iter().enumerate() {
        grid.insert(idx as u32, c);
    }
    let h2 = h * h;
    let mut new_centers = Vec::new();
    let try_add = |net: &mut Vec<[f64; 3]>,
                   grid: &mut HashGrid,
                   new_centers: &mut Vec<[f64; 3]>,
                   p: &[f64; 3]| {
        let mut covered = false;
        grid.for_neighbors(p, |j| {
            if !covered && dist2(p, &net[j as usize], dim) <= h2 {
                covered = true;
            }
        });
        if !covered {
            let idx = net.len() as u32;
            net.push(*p);
            grid.insert(idx, p);
            new_centers.push(*p);
        }
    };
    for p in points {
        try_add(net, &mut grid, &mut new_centers, p);
        if net.len() > MAX_CENTERS {
            return new_centers;
        }
    }
    // Fill the bounding box so the net covers the domain, not just the data.
    //
    // The box has ~`(box_hi/h)^dim` cells, so the fill cost grows like
    // `(2^l)^dim` as the covering radius `h = h₀·2^{-l}` shrinks with the
    // level `l`. At fine levels below the data spacing that is an explosion
    // (every sub-data-spacing cell of the whole domain becomes a synthetic
    // center), which is unbounded work the caller never needs: once the net
    // crosses `MAX_CENTERS` the build path errors and the auto-route's typed
    // next-level assessment reports center-capacity underresolution. So
    // cap the fill IN the loop — stop planting synthetic centers the moment
    // the net exceeds the cap rather than materializing the entire fine-level
    // box first. Coarse levels (few cells, never near the cap) keep the full
    // quasi-uniform domain fill and the polynomial-bridge gap behavior intact.
    let mut cells = [1_i64; 3];
    for a in 0..dim {
        cells[a] = (box_hi[a] / h).ceil() as i64 + 1;
    }
    let mut c = [0.0_f64; 3];
    'fill: for i0 in 0..cells[0] {
        c[0] = (i0 as f64 + 0.5) * h;
        for i1 in 0..cells[1] {
            if dim > 1 {
                c[1] = (i1 as f64 + 0.5) * h;
            }
            for i2 in 0..cells[2] {
                if dim > 2 {
                    c[2] = (i2 as f64 + 0.5) * h;
                }
                try_add(net, &mut grid, &mut new_centers, &c);
                if net.len() > MAX_CENTERS {
                    break 'fill;
                }
            }
        }
    }
    new_centers
}

impl ResidualCascadeDesign {
    /// Build the cascade design: validate, scale by the metric, grow `levels`
    /// nested nets, and assemble the sparse design plus its sufficient
    /// statistics in O(n·(levels + 3^d)).
    ///
    /// `xs` holds one slice per axis (2 or 3 of them), `metric` the positive
    /// per-axis scaling of the learned metric, `sobolev_s` the Sobolev order
    /// of the equivalent (semi)norm — must satisfy `d/2 < s ≤ (d+3)/2` (the
    /// Wendland-(3,1) native smoothness).
    pub fn build(
        xs: &[&[f64]],
        y: &[f64],
        w: &[f64],
        metric: &[f64],
        sobolev_s: f64,
        levels: usize,
    ) -> Result<Self, String> {
        let dim = xs.len();
        if !(dim == 2 || dim == 3) {
            return Err(format!(
                "residual cascade: built for scattered 2-3D smooths, got {dim} axes"
            ));
        }
        let n = y.len();
        if w.len() != n || xs.iter().any(|x| x.len() != n) {
            return Err(format!(
                "residual cascade: length mismatch (y={n}, w={}, axes={:?})",
                w.len(),
                xs.iter().map(|x| x.len()).collect::<Vec<_>>()
            ));
        }
        if n <= dim + 1 {
            return Err(format!(
                "residual cascade: needs more than {} rows for the profiled REML degrees of \
                 freedom, got {n}",
                dim + 1
            ));
        }
        if metric.len() != dim || metric.iter().any(|&s| !(s.is_finite() && s > 0.0)) {
            return Err(format!(
                "residual cascade: metric must be {dim} finite positive scales, got {metric:?}"
            ));
        }
        if !(sobolev_s > dim as f64 / 2.0 && sobolev_s <= (dim as f64 + 3.0) / 2.0) {
            return Err(format!(
                "residual cascade: sobolev_s must lie in (d/2, (d+3)/2] = ({}, {}] for the \
                 Wendland-(3,1) bump, got {sobolev_s}",
                dim as f64 / 2.0,
                (dim as f64 + 3.0) / 2.0
            ));
        }
        if levels == 0 || levels > MAX_LEVELS {
            return Err(format!(
                "residual cascade: levels must be in 1..={MAX_LEVELS}, got {levels}"
            ));
        }
        for i in 0..n {
            if !(y[i].is_finite() && w[i].is_finite() && w[i] > 0.0)
                || xs.iter().any(|x| !x[i].is_finite())
            {
                return Err(format!(
                    "residual cascade: non-finite or non-positive input at row {i}"
                ));
            }
        }
        // Scaled, corner-shifted coordinates.
        let mut z_lo = [f64::INFINITY; 3];
        let mut z_hi = [f64::NEG_INFINITY; 3];
        for a in 0..dim {
            for &v in xs[a] {
                let s = metric[a] * v;
                z_lo[a] = z_lo[a].min(s);
                z_hi[a] = z_hi[a].max(s);
            }
        }
        let mut z_range = [1.0_f64; 3];
        let mut max_range = 0.0_f64;
        for a in 0..dim {
            if !(z_hi[a] > z_lo[a]) {
                return Err(format!(
                    "residual cascade: degenerate axis {a} bounding box [{}, {}]",
                    z_lo[a], z_hi[a]
                ));
            }
            z_range[a] = z_hi[a] - z_lo[a];
            max_range = max_range.max(z_range[a]);
        }
        for a in dim..3 {
            z_lo[a] = 0.0;
        }
        let z: Vec<[f64; 3]> = (0..n)
            .map(|i| {
                let mut p = [0.0_f64; 3];
                for a in 0..dim {
                    p[a] = metric[a] * xs[a][i] - z_lo[a];
                }
                p
            })
            .collect();
        let mut metric3 = [1.0_f64; 3];
        metric3[..dim].copy_from_slice(metric);

        let h0 = H0_FRACTION * max_range;
        let mut net: Vec<[f64; 3]> = Vec::new();
        let mut level_specs = Vec::with_capacity(levels);
        let mut col = dim + 1;
        let mut pen_logdet_const = 0.0;
        for l in 0..levels {
            let h = h0 * 0.5_f64.powi(l as i32);
            let new_centers = extend_net(&mut net, &z, dim, h, &z_range);
            if net.len() > MAX_CENTERS {
                return Err(format!(
                    "residual cascade: center cap {MAX_CENTERS} exceeded at level {l}"
                ));
            }
            let weight = level_weight(l, sobolev_s, dim);
            pen_logdet_const += new_centers.len() as f64 * weight.ln();
            let delta = OVERLAP * h;
            let mut grid = HashGrid::new(delta, dim);
            for (j, c) in new_centers.iter().enumerate() {
                grid.insert(j as u32, c);
            }
            let col_offset = col;
            col += new_centers.len();
            level_specs.push(Level {
                h,
                delta,
                weight,
                centers: new_centers,
                col_offset,
                grid,
            });
        }
        let m = col;

        // CSR assembly + sufficient statistics in one pass.
        let mut row_ptr = Vec::with_capacity(n + 1);
        row_ptr.push(0_usize);
        let mut col_idx: Vec<u32> = Vec::new();
        let mut vals: Vec<f64> = Vec::new();
        let mut rhs = vec![0.0_f64; m];
        let mut gram_diag = vec![0.0_f64; m];
        let mut ytwy = 0.0_f64;
        let probe_core = CoreScaffold {
            dim,
            z_range,
            levels: &level_specs,
        };
        for i in 0..n {
            let row = probe_core.basis_row(&z[i]);
            for &(c, v) in &row {
                col_idx.push(c as u32);
                vals.push(v);
                rhs[c] += w[i] * y[i] * v;
                gram_diag[c] += w[i] * v * v;
            }
            ytwy += w[i] * y[i] * y[i];
            row_ptr.push(col_idx.len());
        }
        let mut pen_diag = vec![0.0_f64; m];
        for level in &level_specs {
            for j in 0..level.centers.len() {
                pen_diag[level.col_offset + j] = level.weight;
            }
        }

        // Dense Gram cache under the sizing cap: O(n·q²) scatter of row outer
        // products into the upper triangle.
        let dense_gram = if m <= DENSE_GRAM_MAX {
            let mut gram = vec![0.0_f64; m * m];
            for i in 0..n {
                let lo = row_ptr[i];
                let hi = row_ptr[i + 1];
                for ea in lo..hi {
                    let ca = col_idx[ea] as usize;
                    let va = w[i] * vals[ea];
                    for eb in ea..hi {
                        gram[ca * m + col_idx[eb] as usize] += va * vals[eb];
                    }
                }
            }
            Some(gram)
        } else {
            None
        };

        Ok(ResidualCascadeDesign {
            core: Arc::new(Core {
                dim,
                metric: metric3,
                z_lo,
                z_range,
                sobolev_s,
                levels: level_specs,
                net,
                m,
                row_ptr,
                col_idx,
                vals,
                w: w.to_vec(),
                y: y.to_vec(),
                z,
                rhs,
                ytwy,
                gram_diag,
                pen_diag,
                pen_logdet_const,
                dense_gram,
                predict_chol: None,
            }),
        })
    }

    /// Number of resolution levels.
    pub fn num_levels(&self) -> usize {
        self.core.levels.len()
    }

    /// Aspect ratio of the metric-scaled point cloud: the ratio of the largest
    /// to smallest per-axis standard deviation of the scaled coordinates `z`.
    /// This is the metric-condition measure the quasi-uniformity guard (issue
    /// #1032, caveat 2) keys on — see [`QUASI_UNIFORMITY_MAX_ASPECT`]. A value
    /// near 1 is an isotropic (benign) cloud; a large value means the metric
    /// has collapsed the data onto a lower-dimensional sheet in `z`, breaking
    /// the BPX n-independent iteration bound.
    pub fn metric_scaled_aspect_ratio(&self) -> f64 {
        let dim = self.core.dim;
        let n = self.core.z.len();
        if dim == 0 || n == 0 {
            return 1.0;
        }
        let mut mean = [0.0_f64; 3];
        for p in &self.core.z {
            for a in 0..dim {
                mean[a] += p[a];
            }
        }
        for m in mean.iter_mut().take(dim) {
            *m /= n as f64;
        }
        let mut var = [0.0_f64; 3];
        for p in &self.core.z {
            for a in 0..dim {
                let d = p[a] - mean[a];
                var[a] += d * d;
            }
        }
        let mut sd_lo = f64::INFINITY;
        let mut sd_hi = 0.0_f64;
        for v in var.iter().take(dim) {
            let sd = (v / n as f64).sqrt();
            sd_lo = sd_lo.min(sd);
            sd_hi = sd_hi.max(sd);
        }
        if !(sd_lo > 0.0 && sd_lo.is_finite()) {
            // A collapsed axis (zero scaled spread) is maximally degenerate.
            return f64::INFINITY;
        }
        sd_hi / sd_lo
    }

    /// Quasi-uniformity certificate (issue #1032, caveat 2): `true` iff the
    /// metric-scaled cloud is isotropic enough that the BPX n-independent CG
    /// iteration bound is trustworthy. When this returns `false` the auto-route
    /// MUST fall back to the dense kernel path rather than pay an iterative
    /// solve whose iteration count is no longer n-independent — the CG residual
    /// certificate would still *catch* a mis-solve at [`CG_MAX_ITERS`], but the
    /// guard prevents the silent O(n·iters) blow-up up front.
    pub fn quasi_uniformity_certified(&self) -> bool {
        self.metric_scaled_aspect_ratio() <= QUASI_UNIFORMITY_MAX_ASPECT
    }

    /// Number of columns `ncoarse` in the additive-Schwarz coarse space at `log
    /// λ` (the polynomial layer plus the data-dominated coarsest levels). The
    /// iterative-route preconditioner solves the principal `[0, ncoarse)` block
    /// of `A = X'WX + λD` exactly and Jacobi-preconditions the fine tail; exposed
    /// so the conditioning oracle can reconstruct that block-arrow preconditioner
    /// from the public dense system and certify it is uniformly conditioned in
    /// depth. See [`COARSE_DOMINANCE`].
    pub fn coarse_space_cols(&self, log_lambda: f64) -> Result<usize, String> {
        let lambda = gam_problem::checked_exp_log_strength(log_lambda)
            .map_err(|error| format!("residual cascade: {error}"))?;
        Ok(self.core.coarse_space_cols(lambda))
    }

    /// Total coefficient count (`dim + 1` polynomial + all centers).
    pub fn num_coeffs(&self) -> usize {
        self.core.m
    }

    /// Structural nonzero count of the sparse design `X` (its CSR size). Each
    /// iterative-route PCG iteration applies the operator `A = XᵀWX + λD` as two
    /// CSR products against `X`, so its per-iteration cost is `Θ(nnz(X))`; the
    /// certified sparse-solve work is therefore `solve_iters · num_nonzeros()`,
    /// the figure the residual-cascade complexity certificate compares against
    /// the dense `m³/3` factorization cost. Zero on a predict-only core rebuilt
    /// from a persisted snapshot (the training CSR is intentionally dropped).
    pub fn num_nonzeros(&self) -> usize {
        self.core.col_idx.len()
    }

    /// Total centers across all levels.
    pub fn num_centers(&self) -> usize {
        self.core.m - self.core.nullity()
    }

    /// NEW centers of one level in ORIGINAL (unscaled) coordinates.
    pub fn centers(&self, level: usize) -> Vec<Vec<f64>> {
        let lv = &self.core.levels[level];
        lv.centers
            .iter()
            .map(|c| {
                (0..self.core.dim)
                    .map(|a| (c[a] + self.core.z_lo[a]) / self.core.metric[a])
                    .collect()
            })
            .collect()
    }

    /// Sparse basis row at a raw point, as (column, value) pairs sorted by
    /// column within each block — the exact row the fit used for training
    /// rows, exposed so oracles can assemble the dense system independently.
    pub fn basis_row(&self, x: &[f64]) -> Result<Vec<(usize, f64)>, String> {
        self.check_point(x)?;
        Ok(self.core.basis_row_scaled(&self.core.scale_point(x)))
    }

    fn check_point(&self, x: &[f64]) -> Result<(), String> {
        if x.len() != self.core.dim || x.iter().any(|v| !v.is_finite()) {
            return Err(format!(
                "residual cascade: point must be {} finite coordinates, got {x:?}",
                self.core.dim
            ));
        }
        Ok(())
    }

    /// Exact penalty quadratic `c'Dc` (unit-λ multilevel prior energy).
    pub fn penalty_value(&self, coeff: &[f64]) -> Result<f64, String> {
        if coeff.len() != self.core.m {
            return Err(format!(
                "residual cascade: coefficient length {} != {}",
                coeff.len(),
                self.core.m
            ));
        }
        Ok(coeff
            .iter()
            .zip(self.core.pen_diag.iter())
            .map(|(&c, &d)| d * c * c)
            .sum())
    }

    /// Exact dense log-determinant of `X'WX + λD` (errors past the sizing
    /// cap) — exposed for the in-test SLQ-vs-exact oracle.
    pub fn logdet_exact(&self, log_lambda: f64) -> Result<f64, String> {
        let lambda = gam_problem::checked_exp_log_strength(log_lambda)
            .map_err(|error| format!("residual cascade: {error}"))?;
        self.core.logdet_dense(lambda)
    }

    /// SLQ log-determinant estimate on the fixed deterministic probes —
    /// exposed for the in-test SLQ-vs-exact oracle.
    pub fn logdet_slq(&self, log_lambda: f64) -> Result<f64, String> {
        let lambda = gam_problem::checked_exp_log_strength(log_lambda)
            .map_err(|error| format!("residual cascade: {error}"))?;
        self.core.logdet_slq(lambda)
    }

    /// The bounded `log λ` domain the certified REML search runs on — every
    /// determinant transition `λ ≈ θ`, padded by `ln(1/√ε)` past the extreme Schur
    /// modes.
    ///
    /// Exposed because the ENDPOINTS are where a criterion evaluation is hardest:
    /// `maximize_score_1d` evaluates the lower boundary before anything else, and
    /// on the iterative route that is the λ at which `X'WX + λD` is numerically
    /// singular (#2503). A gate on "the criterion is evaluable everywhere the
    /// search will look" needs to know where that is, rather than hard-coding a λ
    /// read out of one failure's message.
    ///
    /// Rebuilds the whole REML profile, exactly as [`Self::criterion`] does — both
    /// are single-shot oracles, not loop bodies. Past the dense cap that includes
    /// the determinant sweep and the residual quadrature, so calling either in a λ
    /// loop pays the profile per λ; [`Self::fit_reml`] builds it once.
    pub fn log_lambda_domain(&self) -> Result<(f64, f64), String> {
        self.core.reml_profile()?.log_lambda_domain()
    }

    /// Profiled-σ² REML criterion at `log λ` (differences across λ are exact
    /// REML differences on the dense route; one fixed spectral quadrature is
    /// used past the cap).
    pub fn criterion(&self, log_lambda: f64) -> Result<f64, String> {
        Ok(self.core.reml_profile()?.evaluate(log_lambda)?.jet.value)
    }

    /// Fit at a FIXED `log λ`, with σ² either supplied or profiled.
    pub fn fit_at(
        &self,
        log_lambda: f64,
        sigma2: Option<f64>,
    ) -> Result<ResidualCascadeFit, String> {
        self.fit_at_with_warm(log_lambda, sigma2, None, None)
    }

    fn fit_at_with_warm(
        &self,
        log_lambda: f64,
        sigma2: Option<f64>,
        warm: Option<&[f64]>,
        selection: Option<CascadeSelectionProvenance>,
    ) -> Result<ResidualCascadeFit, String> {
        let core = &self.core;
        let lambda = gam_problem::checked_exp_log_strength(log_lambda)
            .map_err(|error| format!("residual cascade: {error}"))?;
        let (coeff, rel_res, iters) = core.solve_coeff(lambda, &core.rhs, warm)?;
        let rss_pen = core.rss_pen(&coeff);
        let dof = (core.y.len() - core.nullity()) as f64;
        let sigma2 = match sigma2 {
            Some(s) => {
                if !(s.is_finite() && s > 0.0) {
                    return Err(format!("residual cascade: invalid sigma2 {s}"));
                }
                s
            }
            None => {
                if !(rss_pen > 0.0) {
                    return Err(format!(
                        "residual cascade: degenerate penalized residual {rss_pen}"
                    ));
                }
                rss_pen / dof
            }
        };
        let r = (core.m - core.nullity()) as f64;
        let (logdet, logdet_method) = match selection.map(|s| s.normalized_logdet) {
            Some(normalized) => (
                normalized + r * log_lambda + core.pen_logdet_const,
                if core.dense_gram.is_some() {
                    LogdetMethod::DenseExact
                } else {
                    LogdetMethod::Slq
                },
            ),
            None => core.logdet(lambda)?,
        };
        // Full restricted log-likelihood at this (λ, σ²) up to λ- and σ-free
        // constants; at the profiled σ̂² the quadratic collapses to `dof`.
        let restricted_loglik = -0.5
            * (logdet - r * log_lambda - core.pen_logdet_const
                + dof * sigma2.ln()
                + rss_pen / sigma2);
        let predict_chol = if core.dense_gram.is_some() {
            Some(core.assemble_predict_factor(lambda)?)
        } else {
            None
        };
        Ok(ResidualCascadeFit {
            core: Arc::clone(&self.core),
            predict_chol,
            coeff,
            log_lambda,
            sigma2,
            restricted_loglik,
            rss_pen,
            certificate: CascadeCertificate {
                solve_rel_residual: rel_res,
                solve_iters: iters,
                logdet_method,
                residual_moments: selection.map(|s| s.residual_moments),
            },
            refinement: None,
        })
    }

    /// Fit with `log λ` selected by the profiled REML criterion. Every
    /// stationary interval in the bounded domain is isolated from analytic
    /// derivative enclosures, refined by safeguarded Newton/bisection, and
    /// compared with both exact boundary candidates. The large route uses one
    /// lambda-independent fixed-probe spectral profile, so it is the same
    /// smooth deterministic score at every trial.
    pub fn fit_reml(&self) -> Result<ResidualCascadeFit, String> {
        let profile = self.core.reml_profile()?;
        let (log_lambda_lo, log_lambda_hi) = profile.log_lambda_domain()?;
        let resolution = f64::EPSILON.sqrt();
        let failed = |error: &dyn std::fmt::Display| {
            format!("residual cascade: REML stationary isolation failed: {error}")
        };
        // Both arms run the SAME certified isolation on the SAME domain; they
        // differ only in the enclosure oracle the design can honestly supply.
        let selected_log_lambda = match profile.affine_view()? {
            Some(affine) => {
                affine
                    .maximize(log_lambda_lo, log_lambda_hi, resolution)
                    .map_err(|error| failed(&error))?
                    .optimum
                    .x
            }
            None => {
                maximize_score_1d(
                    log_lambda_lo,
                    log_lambda_hi,
                    resolution,
                    |log_lambda| {
                        profile
                            .evaluate(log_lambda)
                            .map(|evaluation| evaluation.jet)
                    },
                    |left, right| profile.enclose(left, right),
                )
                .map_err(|error| failed(&error))?
                .optimum
                .x
            }
        };
        let selected = profile.evaluate(selected_log_lambda)?;
        self.fit_at_with_warm(
            selected_log_lambda,
            None,
            None,
            Some(CascadeSelectionProvenance {
                normalized_logdet: selected.normalized_logdet,
                residual_moments: profile.residual.method(),
            }),
        )
    }

    /// Assess the candidate level L+1 at this fit's λ. A complete candidate
    /// reports the exact upper bound `‖X₂'W r̂‖² / (λ·d_{L+1})` on its
    /// penalized-objective decrease (see the module header for the Schur-
    /// complement argument). Empty-net exhaustion and representation capacity
    /// are different typed outcomes because only an empty net certifies zero
    /// remaining gain.
    pub fn assess_next_level(
        &self,
        fit: &ResidualCascadeFit,
    ) -> Result<NextLevelAssessment, String> {
        let core = &self.core;
        if !Arc::ptr_eq(core, &fit.core) {
            return Err("residual cascade: fit does not belong to this design".into());
        }
        let next_l = core.levels.len();
        let h = core.levels[next_l - 1].h * 0.5;
        let mut net = core.net.clone();
        let candidates = extend_net(&mut net, &core.z, core.dim, h, &core.z_range);
        if candidates.is_empty() {
            return Ok(NextLevelAssessment::EmptyNet);
        }
        if net.len() > MAX_CENTERS {
            return Ok(NextLevelAssessment::CapacityExceeded {
                obstruction: RefinementObstruction::CenterCapacity {
                    centers: net.len(),
                    maximum_centers: MAX_CENTERS,
                },
                // The cap stopped candidate construction before every column
                // could contribute to ‖X₂'Wr̂‖². Infinity is the honest
                // conservative upper bound; a finite partial sum would not
                // certify the omitted columns.
                gain_bound: f64::INFINITY,
            });
        }
        let delta = OVERLAP * h;
        let mut grid = HashGrid::new(delta, core.dim);
        for (j, c) in candidates.iter().enumerate() {
            grid.insert(j as u32, c);
        }
        let r = core.residuals(&fit.coeff);
        let mut g = vec![0.0_f64; candidates.len()];
        for (i, zi) in core.z.iter().enumerate() {
            let wr = core.w[i] * r[i];
            grid.for_neighbors(zi, |j| {
                let rad = dist2(zi, &candidates[j as usize], core.dim).sqrt() / delta;
                g[j as usize] += wr * wendland(rad);
            });
        }
        let g2: f64 = g.iter().map(|v| v * v).sum();
        let d_next = level_weight(next_l, core.sobolev_s, core.dim);
        let lambda = gam_problem::checked_exp_log_strength(fit.log_lambda)
            .map_err(|error| format!("residual cascade refinement: {error}"))?;
        let gain_bound = g2 / (lambda * d_next);
        if next_l >= MAX_LEVELS {
            Ok(NextLevelAssessment::CapacityExceeded {
                obstruction: RefinementObstruction::LevelCapacity {
                    levels: next_l,
                    maximum_levels: MAX_LEVELS,
                },
                gain_bound,
            })
        } else {
            Ok(NextLevelAssessment::GainBound(gain_bound))
        }
    }
}

/// Prior precision weight of level `l`: `4^{l(s−d/2)}`.
fn level_weight(l: usize, sobolev_s: f64, dim: usize) -> f64 {
    (4.0_f64).powf(l as f64 * (sobolev_s - dim as f64 / 2.0))
}

/// Lightweight view used during assembly, before the Core exists: shares the
/// exact basis-row logic with [`Core::basis_row_scaled`] so the assembled CSR
/// and later prediction rows cannot drift apart.
struct CoreScaffold<'a> {
    dim: usize,
    z_range: [f64; 3],
    levels: &'a [Level],
}

impl CoreScaffold<'_> {
    fn basis_row(&self, z: &[f64; 3]) -> Vec<(usize, f64)> {
        let mut row = Vec::with_capacity(self.dim + 1 + self.levels.len() * 8);
        row.push((0, 1.0));
        for a in 0..self.dim {
            row.push((a + 1, 2.0 * z[a] / self.z_range[a] - 1.0));
        }
        for level in self.levels {
            let start = row.len();
            level.grid.for_neighbors(z, |j| {
                let c = &level.centers[j as usize];
                let r = dist2(z, c, self.dim).sqrt() / level.delta;
                let v = wendland(r);
                if v > 0.0 {
                    row.push((level.col_offset + j as usize, v));
                }
            });
            row[start..].sort_unstable_by_key(|&(col, _)| col);
        }
        row
    }
}

impl ResidualCascadeFit {
    pub fn log_lambda(&self) -> f64 {
        self.log_lambda
    }

    pub fn lambda(&self) -> f64 {
        gam_problem::checked_exp_log_strength(self.log_lambda)
            .expect("ResidualCascadeFit construction validates its private log strength")
    }

    /// Posterior `(mean, variance)` at a raw point: the sparse basis row
    /// dotted with the coefficients, and `σ̂²·x'(X'WX+λD)^{−1}x` through one
    /// certified solve.
    pub fn predict(&self, x: &[f64]) -> Result<(f64, f64), String> {
        let core = &self.core;
        if x.len() != core.dim || x.iter().any(|v| !v.is_finite()) {
            return Err(format!(
                "residual cascade: prediction point must be {} finite coordinates, got {x:?}",
                core.dim
            ));
        }
        let row = core.basis_row_scaled(&core.scale_point(x));
        let mut mean = 0.0;
        let mut dense_row = vec![0.0_f64; core.m];
        for &(c, v) in &row {
            mean += v * self.coeff[c];
            dense_row[c] += v;
        }
        let lambda = gam_problem::checked_exp_log_strength(self.log_lambda)
            .map_err(|error| format!("residual cascade fit: {error}"))?;
        let zsol = if let Some(l) = &self.predict_chol {
            chol_solve(l, core.m, &dense_row)
        } else {
            core.solve_coeff(lambda, &dense_row, None)?.0
        };
        let mut quad = 0.0;
        for (a, b) in dense_row.iter().zip(zsol.iter()) {
            quad += a * b;
        }
        Ok((mean, self.sigma2 * quad))
    }

    /// EXACT posterior coefficient samples by perturb-and-solve:
    /// `c_s = A^{−1}(X'Wy + σ(X'W^{1/2}z₁ + √λ D^{1/2}z₂))` has mean ĉ and
    /// covariance exactly `σ̂²A^{−1}`. Deterministically seeded; one certified
    /// solve per sample (warm-started at the mode).
    pub fn sample_coefficients(&self, n_samples: usize) -> Result<Vec<Vec<f64>>, String> {
        let core = &self.core;
        let lambda = gam_problem::checked_exp_log_strength(self.log_lambda)
            .map_err(|error| format!("residual cascade fit: {error}"))?;
        let sigma = self.sigma2.sqrt();
        let sqrt_lambda = lambda.sqrt();
        let n = core.y.len();
        let mut rng = SplitMix64::new(RNG_SEED ^ 0xA11C_E5A_u64);
        let mut samples = Vec::with_capacity(n_samples);
        for _ in 0..n_samples {
            let mut b = core.rhs.clone();
            // X'W^{1/2} z₁: one CSR pass with per-row factor √w_i·z₁_i.
            for i in 0..n {
                let f = sigma * core.w[i].sqrt() * rng.next_normal();
                for e in core.row_ptr[i]..core.row_ptr[i + 1] {
                    b[core.col_idx[e] as usize] += f * core.vals[e];
                }
            }
            // √λ D^{1/2} z₂ on the penalized columns.
            for (bj, &dj) in b.iter_mut().zip(core.pen_diag.iter()) {
                if dj > 0.0 {
                    *bj += sigma * sqrt_lambda * dj.sqrt() * rng.next_normal();
                }
            }
            let (c, _, _) = core.solve_coeff(lambda, &b, Some(&self.coeff))?;
            samples.push(c);
        }
        Ok(samples)
    }

    /// Number of resolution levels in the fitted cascade.
    pub fn num_levels(&self) -> usize {
        self.core.levels.len()
    }

    /// Total coefficient count.
    pub fn num_coeffs(&self) -> usize {
        self.core.m
    }

    /// Total centers across all fitted resolution levels.
    pub fn num_centers(&self) -> usize {
        self.core.m - self.core.nullity()
    }

    /// Snapshot the fit for persistence (#1032). Assembles the factored
    /// precision `L` of `A = X'WX + λD` at the fit's λ (O(m³) once) and copies
    /// the nested geometry + coefficients, dropping all training rows. The
    /// resulting [`ResidualCascadeState`] is predict-complete: `from_state`
    /// replays the posterior mean+variance bit-for-bit.
    pub fn to_state(&self) -> Result<ResidualCascadeState, String> {
        let core = &self.core;
        let lambda = gam_problem::checked_exp_log_strength(self.log_lambda)
            .map_err(|error| format!("residual cascade fit: {error}"))?;
        let predict_chol = if let Some(l) = &self.predict_chol {
            l.clone()
        } else if let Some(l) = &core.predict_chol {
            l.clone()
        } else {
            core.assemble_predict_factor(lambda)?
        };
        let dim = core.dim;
        let levels = core
            .levels
            .iter()
            .map(|level| {
                let mut centers = Vec::with_capacity(level.centers.len() * dim);
                for c in &level.centers {
                    centers.extend_from_slice(&c[..dim]);
                }
                LevelState {
                    h: level.h,
                    delta: level.delta,
                    weight: level.weight,
                    col_offset: level.col_offset as u64,
                    centers,
                }
            })
            .collect();
        Ok(ResidualCascadeState {
            dim: dim as u64,
            metric: core.metric,
            z_lo: core.z_lo,
            z_range: core.z_range,
            sobolev_s: core.sobolev_s,
            levels,
            m: core.m as u64,
            pen_logdet_const: core.pen_logdet_const,
            coeff: self.coeff.clone(),
            log_lambda: self.log_lambda,
            sigma2: self.sigma2,
            restricted_loglik: self.restricted_loglik,
            rss_pen: self.rss_pen,
            predict_chol,
        })
    }

    /// Rebuild a predict-capable fit from a snapshot (#1032). Validates shape,
    /// finiteness, the Sobolev/Wendland window, strictly-positive level weights
    /// and box ranges, the column accounting (`m = dim+1 + Σ centers`, matching
    /// `col_offset`s), positive σ², and that `predict_chol` is a valid `m × m`
    /// lower factor (positive pivots) — so a corrupt payload fails here, not in
    /// a later `predict`. The restored `Core` has empty training CSR and
    /// `predict_chol = Some(L)`; its `predict` reads only geometry (mean) and
    /// the factor (variance), replaying both exactly.
    pub fn from_state(state: &ResidualCascadeState) -> Result<Self, String> {
        let dim = state.dim as usize;
        if !(dim == 2 || dim == 3) {
            return Err(format!(
                "residual cascade state: dim must be 2 or 3, got {dim}"
            ));
        }
        if !(state.sobolev_s > dim as f64 / 2.0 && state.sobolev_s <= (dim as f64 + 3.0) / 2.0) {
            return Err(format!(
                "residual cascade state: sobolev_s {} outside the Wendland window ({}, {}]",
                state.sobolev_s,
                dim as f64 / 2.0,
                (dim as f64 + 3.0) / 2.0
            ));
        }
        for a in 0..dim {
            if !(state.metric[a].is_finite() && state.metric[a] > 0.0) {
                return Err(format!(
                    "residual cascade state: metric axis {a} must be finite positive, got {}",
                    state.metric[a]
                ));
            }
            if !(state.z_range[a].is_finite()
                && state.z_range[a] > 0.0
                && state.z_lo[a].is_finite())
            {
                return Err(format!(
                    "residual cascade state: degenerate box on axis {a} (lo={}, range={})",
                    state.z_lo[a], state.z_range[a]
                ));
            }
        }
        let m = state.m as usize;
        let mut metric3 = [1.0_f64; 3];
        metric3[..dim].copy_from_slice(&state.metric[..dim]);
        let mut z_lo = [0.0_f64; 3];
        let mut z_range = [1.0_f64; 3];
        z_lo[..dim].copy_from_slice(&state.z_lo[..dim]);
        z_range[..dim].copy_from_slice(&state.z_range[..dim]);

        // Rebuild the levels and their lookup grids from the flattened centers,
        // checking the column accounting matches the polynomial layer + blocks.
        let mut levels = Vec::with_capacity(state.levels.len());
        let mut net: Vec<[f64; 3]> = Vec::new();
        let mut pen_diag = vec![0.0_f64; m];
        let mut expected_offset = dim + 1;
        for (li, ls) in state.levels.iter().enumerate() {
            if !(ls.h.is_finite() && ls.h > 0.0 && ls.delta.is_finite() && ls.delta > 0.0) {
                return Err(format!(
                    "residual cascade state: level {li} has non-positive h/delta ({}, {})",
                    ls.h, ls.delta
                ));
            }
            if !(ls.weight.is_finite() && ls.weight > 0.0) {
                return Err(format!(
                    "residual cascade state: level {li} has non-positive prior weight {}",
                    ls.weight
                ));
            }
            if ls.centers.len() % dim != 0 {
                return Err(format!(
                    "residual cascade state: level {li} centers length {} not a multiple of dim {dim}",
                    ls.centers.len()
                ));
            }
            let n_centers = ls.centers.len() / dim;
            let col_offset = ls.col_offset as usize;
            if col_offset != expected_offset {
                return Err(format!(
                    "residual cascade state: level {li} col_offset {col_offset} ≠ expected {expected_offset}"
                ));
            }
            let mut grid = HashGrid::new(ls.delta, dim);
            let mut centers = Vec::with_capacity(n_centers);
            for j in 0..n_centers {
                let mut c = [0.0_f64; 3];
                for a in 0..dim {
                    let v = ls.centers[j * dim + a];
                    if !v.is_finite() {
                        return Err(format!(
                            "residual cascade state: non-finite center coordinate at level {li}, center {j}"
                        ));
                    }
                    c[a] = v;
                }
                grid.insert(j as u32, &c);
                centers.push(c);
                net.push(c);
                let col = col_offset + j;
                if col >= m {
                    return Err(format!(
                        "residual cascade state: level {li} column {col} exceeds m {m}"
                    ));
                }
                pen_diag[col] = ls.weight;
            }
            expected_offset = col_offset + n_centers;
            levels.push(Level {
                h: ls.h,
                delta: ls.delta,
                weight: ls.weight,
                centers,
                col_offset,
                grid,
            });
        }
        if expected_offset != m {
            return Err(format!(
                "residual cascade state: column accounting mismatch (dim+1+Σcenters = {expected_offset} ≠ m {m})"
            ));
        }
        if state.coeff.len() != m {
            return Err(format!(
                "residual cascade state: coeff length {} ≠ m {m}",
                state.coeff.len()
            ));
        }
        if state.predict_chol.len() != m * m {
            return Err(format!(
                "residual cascade state: predict_chol must be m×m = {m}² = {}, got {}",
                m * m,
                state.predict_chol.len()
            ));
        }
        for (i, v) in state
            .coeff
            .iter()
            .chain(state.predict_chol.iter())
            .enumerate()
        {
            if !v.is_finite() {
                return Err(format!("residual cascade state: non-finite entry at {i}"));
            }
        }
        for g in 0..m {
            let piv = state.predict_chol[g * m + g];
            if !(piv.is_finite() && piv > 0.0) {
                return Err(format!(
                    "residual cascade state: non-positive Cholesky pivot {piv} at index {g}"
                ));
            }
        }
        gam_problem::validate_log_strength(state.log_lambda)
            .map_err(|error| format!("residual cascade state: {error}"))?;
        if !(state.sigma2.is_finite()
            && state.sigma2 > 0.0
            && state.restricted_loglik.is_finite()
            && state.rss_pen.is_finite())
        {
            return Err(format!(
                "residual cascade state: invalid scalars (log_lambda={}, sigma2={}, restricted_loglik={}, rss_pen={})",
                state.log_lambda, state.sigma2, state.restricted_loglik, state.rss_pen
            ));
        }
        let core = Core {
            dim,
            metric: metric3,
            z_lo,
            z_range,
            sobolev_s: state.sobolev_s,
            levels,
            net,
            m,
            row_ptr: Vec::new(),
            col_idx: Vec::new(),
            vals: Vec::new(),
            w: Vec::new(),
            y: Vec::new(),
            z: Vec::new(),
            rhs: Vec::new(),
            ytwy: 0.0,
            gram_diag: Vec::new(),
            pen_diag,
            pen_logdet_const: state.pen_logdet_const,
            dense_gram: None,
            predict_chol: Some(state.predict_chol.clone()),
        };
        Ok(ResidualCascadeFit {
            core: Arc::new(core),
            predict_chol: None,
            coeff: state.coeff.clone(),
            log_lambda: state.log_lambda,
            sigma2: state.sigma2,
            restricted_loglik: state.restricted_loglik,
            rss_pen: state.rss_pen,
            certificate: CascadeCertificate {
                solve_rel_residual: 0.0,
                solve_iters: 0,
                logdet_method: LogdetMethod::DenseExact,
                // A core rebuilt from a persisted state carries no training
                // design, so it can neither re-derive nor replay the criterion
                // that selected this λ — the selection provenance is not
                // reconstructible here and is reported as absent rather than
                // guessed from the route the rebuilt core happens to be on.
                residual_moments: None,
            },
            refinement: None,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum RefinementDecision {
    Converged {
        gain_bound: f64,
    },
    Refine,
    Underresolved {
        gain_bound: f64,
        obstruction: RefinementObstruction,
    },
}

/// Turn the typed next-level assessment into the only three legal refinement
/// transitions. In particular, a capacity limit can yield a fit only when its
/// already-computed gain bound independently passes the requested tolerance.
fn decide_refinement(
    assessment: NextLevelAssessment,
    requested_tolerance: f64,
) -> RefinementDecision {
    match assessment {
        NextLevelAssessment::EmptyNet => RefinementDecision::Converged { gain_bound: 0.0 },
        NextLevelAssessment::GainBound(gain_bound) if gain_bound <= requested_tolerance => {
            RefinementDecision::Converged { gain_bound }
        }
        NextLevelAssessment::GainBound(_) => RefinementDecision::Refine,
        NextLevelAssessment::CapacityExceeded {
            gain_bound,
            obstruction: _,
        } if gain_bound <= requested_tolerance => RefinementDecision::Converged { gain_bound },
        NextLevelAssessment::CapacityExceeded {
            gain_bound,
            obstruction,
        } => RefinementDecision::Underresolved {
            gain_bound,
            obstruction,
        },
    }
}

/// Fit the full magic-default cascade: start at [`INITIAL_LEVELS`], REML-fit,
/// and refine (add a level, refit, re-select λ) until the exact next-level
/// gain bound certifies that one more level cannot move the penalized
/// objective by more than [`REFINE_TOL`] of the penalized residual. A genuinely
/// empty next-level net certifies zero remaining gain; a level/center capacity
/// reached before the tolerance passes is a typed
/// [`ResidualCascadeError::Underresolved`] carrying the retained work and its
/// evidence, never a fit.
pub fn fit_residual_cascade(
    xs: &[&[f64]],
    y: &[f64],
    w: &[f64],
    metric: &[f64],
    sobolev_s: f64,
) -> Result<ResidualCascadeFit, ResidualCascadeError> {
    let mut levels = INITIAL_LEVELS;
    loop {
        let design = ResidualCascadeDesign::build(xs, y, w, metric, sobolev_s, levels)?;
        // Quasi-uniformity guard (issue #1032, caveat 2): if the metric has
        // collapsed the cloud onto a near-degenerate sheet in scaled
        // coordinates, the BPX iteration bound no longer holds. Refuse the
        // iterative solve up front with a typed signal so the auto-route falls
        // back to the dense kernel BEFORE paying an unbounded CG, rather than
        // grinding to CG_MAX_ITERS. (The guard is checked at the root level
        // only — refinement adds finer nets to the SAME scaled cloud, so the
        // aspect ratio is invariant under added levels.)
        if levels == INITIAL_LEVELS && !design.quasi_uniformity_certified() {
            return Err(format!(
                "residual cascade: metric-scaled aspect ratio {:.3e} exceeds the \
                 quasi-uniformity ceiling {QUASI_UNIFORMITY_MAX_ASPECT:.0e}; the BPX \
                 iteration bound is not trustworthy on this (near-degenerate) metric — \
                 fall back to the dense kernel path",
                design.metric_scaled_aspect_ratio()
            )
            .into());
        }
        let mut fit = design.fit_reml()?;
        // The realized CG iteration count at this cascade depth is the runtime
        // tell of the BPX n-independence bound (issue #1032 caveat: a count
        // creeping toward CG_MAX_ITERS means the quasi-uniformity guard's static
        // aspect-ratio check was too lenient for this cloud). It is exposed
        // STRUCTURALLY rather than over stderr: the per-depth count and backward
        // error ride on `fit.certificate` (`solve_iters` — 0 on the dense route,
        // the PCG count on the iterative route — and `solve_rel_residual`), so a
        // caller that wants to watch the bound reads them off the returned fit
        // instead of scraping log lines. (A library solve never writes to
        // stderr.)
        let assessment = design.assess_next_level(&fit)?;
        let requested_tolerance = REFINE_TOL * fit.rss_pen;
        match decide_refinement(assessment, requested_tolerance) {
            RefinementDecision::Converged { gain_bound } => {
                fit.refinement = Some(RefinementCertificate {
                    next_level_gain_bound: gain_bound,
                    tolerance: requested_tolerance,
                });
                return Ok(fit);
            }
            RefinementDecision::Refine => {
                levels += 1;
            }
            RefinementDecision::Underresolved {
                gain_bound,
                obstruction,
            } => {
                return Err(ResidualCascadeError::Underresolved {
                    checkpoint: ResidualCascadeCheckpoint::new(fit),
                    gain_bound,
                    requested_tolerance,
                    obstruction,
                });
            }
        }
    }
}

#[cfg(test)]
mod refinement_decision_tests {
    use super::*;

    const TOLERANCE: f64 = 0.25;

    #[test]
    fn only_empty_or_passing_bound_converges() {
        assert_eq!(
            decide_refinement(NextLevelAssessment::EmptyNet, TOLERANCE),
            RefinementDecision::Converged { gain_bound: 0.0 }
        );
        assert_eq!(
            decide_refinement(NextLevelAssessment::GainBound(0.2), TOLERANCE),
            RefinementDecision::Converged { gain_bound: 0.2 }
        );
        assert_eq!(
            decide_refinement(NextLevelAssessment::GainBound(0.3), TOLERANCE),
            RefinementDecision::Refine
        );
    }

    #[test]
    fn capacity_above_tolerance_is_underresolved() {
        let obstruction = RefinementObstruction::LevelCapacity {
            levels: MAX_LEVELS,
            maximum_levels: MAX_LEVELS,
        };
        assert_eq!(
            decide_refinement(
                NextLevelAssessment::CapacityExceeded {
                    obstruction,
                    gain_bound: 0.3,
                },
                TOLERANCE,
            ),
            RefinementDecision::Underresolved {
                gain_bound: 0.3,
                obstruction,
            }
        );

        let center_obstruction = RefinementObstruction::CenterCapacity {
            centers: MAX_CENTERS + 1,
            maximum_centers: MAX_CENTERS,
        };
        assert_eq!(
            decide_refinement(
                NextLevelAssessment::CapacityExceeded {
                    obstruction: center_obstruction,
                    gain_bound: f64::INFINITY,
                },
                TOLERANCE,
            ),
            RefinementDecision::Underresolved {
                gain_bound: f64::INFINITY,
                obstruction: center_obstruction,
            }
        );
    }

    #[test]
    fn capacity_does_not_block_an_independently_passing_bound() {
        assert_eq!(
            decide_refinement(
                NextLevelAssessment::CapacityExceeded {
                    obstruction: RefinementObstruction::LevelCapacity {
                        levels: MAX_LEVELS,
                        maximum_levels: MAX_LEVELS,
                    },
                    gain_bound: 0.2,
                },
                TOLERANCE,
            ),
            RefinementDecision::Converged { gain_bound: 0.2 }
        );
    }

    /// A 2-D fixture small enough to stay under the dense sizing cap, with a
    /// response that is smooth plus a deterministic wobble so the profiled
    /// residual is not degenerate at any λ.
    fn dense_fixture(side: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut x1 = Vec::with_capacity(side * side);
        let mut x2 = Vec::with_capacity(side * side);
        let mut y = Vec::with_capacity(side * side);
        for i in 0..side {
            for j in 0..side {
                let a = i as f64 / (side - 1) as f64;
                let b = j as f64 / (side - 1) as f64;
                x1.push(a);
                x2.push(b);
                y.push((2.3 * a).sin() + (1.7 * b).cos() + 0.07 * ((3 * i + 5 * j) % 7) as f64);
            }
        }
        (x1, x2, y)
    }

    /// Tight clusters with empty space between them, deterministic.
    ///
    /// `extend_net` fills the whole bounding BOX, not just the data, so a cloud
    /// with genuine voids puts cascade columns where no row supports them. Those
    /// columns are annihilated by the design exactly, which is what produces the
    /// numerically null Schur modes. A regular grid never does — every bump has
    /// data under it — which is why the other fixture cannot cover that edge.
    fn clustered_fixture() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let centers = [(0.12_f64, 0.13_f64), (0.86, 0.20), (0.45, 0.88)];
        let mut x1 = Vec::new();
        let mut x2 = Vec::new();
        let mut y = Vec::new();
        let mut state = 0x2455_u64;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 11) as f64) / ((1u64 << 53) as f64)
        };
        for (cx, cy) in centers {
            for _ in 0..60 {
                let a = cx + 0.06 * (next() - 0.5);
                let b = cy + 0.06 * (next() - 0.5);
                x1.push(a);
                x2.push(b);
                y.push((3.0 * a).sin() + (2.0 * b).cos() + 0.05 * (next() - 0.5));
            }
        }
        (x1, x2, y)
    }

    /// The two [`CascadeResidualForm`] arms are the same function of λ.
    ///
    /// The spectral arm reads the profiled residual and its three quadratic
    /// forms off the Schur decomposition; the solved arm re-derives them from a
    /// factorization of `A = X'WX + λD`. If they ever disagree the criterion is
    /// route-dependent, which is the defect the spectral arm exists to remove —
    /// so the agreement is asserted directly rather than inferred from the
    /// scores that consume it.
    ///
    /// The bound is the textbook forward-error of the comparator, not a tuned
    /// number: the solved arm's Cholesky solve carries `O(m)·eps·cond(A)`, and
    /// `cond(A) = (θ_max + λ)/(θ_min + λ)` is available exactly from the same
    /// spectrum. Nothing here is free to be widened without changing that claim.
    #[test]
    fn spectral_and_solved_residual_forms_agree() {
        let (x1, x2, y) = dense_fixture(6);
        let weights = vec![1.0; y.len()];
        let axes: [&[f64]; 2] = [&x1, &x2];
        let design = ResidualCascadeDesign::build(&axes, &y, &weights, &[1.0, 1.0], 2.0, 2)
            .expect("cascade design");
        let core = &design.core;
        assert!(core.dense_gram.is_some(), "fixture must take the dense route");
        let profile = core.reml_profile().expect("spectral profile");
        let CascadeResidualForm::Spectral(spectrum) = &profile.residual else {
            panic!("the dense route must carry the spectral residual form");
        };

        let smallest = spectrum
            .eigenvalue
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
        let largest = spectrum
            .eigenvalue
            .iter()
            .copied()
            .fold(0.0_f64, f64::max);

        for log_lambda in [-6.0_f64, -2.0, 0.0, 2.0, 6.0] {
            let lambda = log_lambda.exp();
            let (rss, penalty_energy, inverse_penalty_energy, third_energy) =
                spectrum.moments(lambda);

            let solver = core.coeff_solver(lambda).expect("factored solver");
            let coeff = solver.solve(core, lambda, &core.rhs).expect("first solve");
            let dc: Vec<f64> = coeff
                .iter()
                .zip(core.pen_diag.iter())
                .map(|(&c, &d)| d * c)
                .collect();
            let u = solver.solve(core, lambda, &dc).expect("second solve");
            let solved = [
                core.rss_pen(&coeff),
                coeff.iter().zip(dc.iter()).map(|(&c, &v)| c * v).sum(),
                dc.iter().zip(u.iter()).map(|(&a, &b)| a * b).sum(),
                u.iter()
                    .zip(core.pen_diag.iter())
                    .map(|(&v, &d)| d * v * v)
                    .sum(),
            ];
            let spectral = [rss, penalty_energy, inverse_penalty_energy, third_energy];
            let names = ["R", "c'Dc", "(Dc)'A^-1(Dc)", "u'Du"];

            let condition = (largest + lambda) / (smallest + lambda);
            // The three quadratic forms are sums of positive terms, so their
            // relative error is the solve's own `O(m)·eps·cond(A)`. `R` is not:
            // BOTH routes form it by subtracting a fitted energy from an anchor
            // energy, so its relative error carries that cancellation's own
            // condition number, `anchor/|R|`. Charging the sum of the two is the
            // honest comparator bound.
            let cancellation = spectrum.anchor_energy[0] / rss.abs().max(f64::MIN_POSITIVE);
            let bounds = [
                core.m as f64 * f64::EPSILON * (condition + cancellation),
                core.m as f64 * f64::EPSILON * condition,
                core.m as f64 * f64::EPSILON * condition,
                core.m as f64 * f64::EPSILON * condition,
            ];
            for (((&a, &b), name), bound) in
                spectral.iter().zip(solved.iter()).zip(names).zip(bounds)
            {
                let gap = (a - b).abs() / b.abs().max(f64::MIN_POSITIVE);
                assert!(
                    gap <= bound,
                    "{name} disagrees at log lambda {log_lambda}: spectral {a}, solved {b} \
                     (relative {gap:e} exceeds the comparator's own forward error {bound:e} \
                      at cond(A) = {condition:e}, cancellation = {cancellation:e})"
                );
            }
        }
    }

    /// The cascade's own closed form and the [`AffineRemlProfile`] the search
    /// actually runs on are one score.
    ///
    /// `fit_reml` isolates the optimum with the affine profile (for its interval
    /// extension) while `criterion` and the selected `normalized_logdet` come
    /// from [`CascadeRemlProfile::evaluate`]. Two implementations of one
    /// quantity is exactly the arrangement that lets a criterion drift, so the
    /// two are held to agreement in value, slope and curvature here. The bound
    /// is the summation roundoff of the mode sums both perform — `rank·eps`
    /// relative — and nothing about the fixture is free to widen it.
    #[test]
    fn affine_view_is_the_same_score_as_the_cascade_jet() {
        let (x1, x2, y) = dense_fixture(6);
        let weights = vec![1.0; y.len()];
        let axes: [&[f64]; 2] = [&x1, &x2];
        let design = ResidualCascadeDesign::build(&axes, &y, &weights, &[1.0, 1.0], 2.0, 2)
            .expect("cascade design");
        let profile = design.core.reml_profile().expect("spectral profile");
        let affine = profile
            .affine_view()
            .expect("affine view")
            .expect("the dense route must expose an affine view");
        let (lo, hi) = profile.log_lambda_domain().expect("domain");
        let rank = (design.core.m - design.core.nullity()) as f64;
        let bound = rank * f64::EPSILON;

        for step in 0..=8 {
            let log_lambda = lo + (hi - lo) * step as f64 / 8.0;
            let cascade = profile.evaluate(log_lambda).expect("cascade jet").jet;
            let spectral = affine.evaluate(log_lambda).expect("affine jet");
            for (name, a, b) in [
                ("value", cascade.value, spectral.value),
                ("derivative", cascade.derivative, spectral.derivative),
                ("curvature", cascade.curvature, spectral.curvature),
            ] {
                let gap = (a - b).abs() / (1.0 + b.abs());
                assert!(
                    gap <= bound,
                    "{name} disagrees at log lambda {log_lambda}: cascade {a}, affine {b} \
                     (relative {gap:e} exceeds the shared mode-sum roundoff {bound:e})"
                );
            }
        }
    }

    /// Both replacement enclosures dismiss a tail cell the Lipschitz pad cannot.
    ///
    /// At the top of `log_lambda_domain` the score's derivative has decayed to
    /// order `rank·sqrt(eps)`, while the pad's radius at the search's own
    /// resolution floor is `C·sqrt(eps)` with `C` of order the residual degrees
    /// of freedom. The pad therefore straddles zero at a width the search cannot
    /// go below — a search that cannot terminate, not a slow one. Both cures
    /// have widths that collapse with the cell instead: the affine interval
    /// extension on the dense route, and the multiplicative bracket that
    /// [`CascadeRemlProfile::enclose`] intersects into the pad everywhere else.
    /// This asserts the difference at the resolution floor rather than
    /// describing it.
    #[test]
    fn tail_cell_the_lipschitz_pad_cannot_dismiss_is_dismissed_by_both_cures() {
        let (x1, x2, y) = dense_fixture(6);
        let weights = vec![1.0; y.len()];
        let axes: [&[f64]; 2] = [&x1, &x2];
        let design = ResidualCascadeDesign::build(&axes, &y, &weights, &[1.0, 1.0], 2.0, 2)
            .expect("cascade design");
        let profile = design.core.reml_profile().expect("spectral profile");
        let affine = profile
            .affine_view()
            .expect("affine view")
            .expect("the dense route must expose an affine view");
        let (_, hi) = profile.log_lambda_domain().expect("domain");

        let lo = hi - f64::EPSILON.sqrt();
        let left = sample_at(&profile, lo);
        let right = sample_at(&profile, hi);

        let pad = profile.lipschitz_pad(left, right, hi - lo);
        assert!(
            pad.derivative.contains_zero(),
            "the fixture no longer reproduces the pad's tail stall: {pad:?}"
        );
        assert!(
            pad.derivative.hi - pad.derivative.lo > 10.0 * left.derivative.abs(),
            "the stall is that the pad is WIDER than the derivative it brackets; \
             derivative {}, pad {:?}",
            left.derivative,
            pad.derivative
        );

        for (name, enclosure) in [
            (
                "affine interval extension",
                affine.enclose(lo, hi).expect("interval extension").derivative,
            ),
            (
                "multiplicative bracket (intersected)",
                profile.enclose(left, right).expect("enclosure").derivative,
            ),
        ] {
            assert!(
                !enclosure.contains_zero(),
                "{name} must exclude zero on a floor-width tail cell where the \
                 derivative is {} and the pad reports {:?}; got {enclosure:?}",
                left.derivative,
                pad.derivative
            );
        }
    }

    /// The multiplicative bracket is an OUTER bound, checked against the score
    /// it claims to bracket, on a design that HAS numerically null Schur modes.
    ///
    /// A too-tight enclosure does not fail loudly — it makes the search discard
    /// a cell that contained a stationary point and return a certified wrong
    /// answer. So the bracket is charged against densely sampled truth on cells
    /// spanning the whole domain and every width from the resolution floor up.
    ///
    /// The deep fixture is deliberate: the null modes are the edge where the
    /// bracket's premises are thinnest (`R'` is a positive mixture only over the
    /// modes that survive the roundoff floor), so the containment claim is made
    /// where it is hardest, not where it is easiest. The test asserts the
    /// fixture still reaches that regime, so it cannot quietly stop covering it.
    #[test]
    fn enclosure_contains_the_derivatives_it_brackets_with_null_modes() {
        let (x1, x2, y) = clustered_fixture();
        let weights = vec![1.0; y.len()];
        let axes: [&[f64]; 2] = [&x1, &x2];
        let design = ResidualCascadeDesign::build(&axes, &y, &weights, &[1.0, 1.0], 2.0, 3)
            .expect("cascade design");
        let profile = design.core.reml_profile().expect("spectral profile");
        let null_modes = profile
            .modes
            .iter()
            .filter(|mode| mode.eigenvalue == 0.0)
            .count();
        assert!(
            null_modes > 0,
            "fixture no longer reaches the null-mode regime this test exists to cover \
             ({} modes, all positive)",
            profile.modes.len()
        );
        check_containment_over(&profile);
    }

    /// The same containment claim on the small, well-conditioned fixture the
    /// other tests use — the two together bracket the range of designs the
    /// enclosure has to serve.
    #[test]
    fn enclosure_contains_the_derivatives_it_brackets() {
        let (x1, x2, y) = dense_fixture(6);
        let weights = vec![1.0; y.len()];
        let axes: [&[f64]; 2] = [&x1, &x2];
        let design = ResidualCascadeDesign::build(&axes, &y, &weights, &[1.0, 1.0], 2.0, 2)
            .expect("cascade design");
        let profile = design.core.reml_profile().expect("spectral profile");
        check_containment_over(&profile);
    }

    fn check_containment_over(profile: &CascadeRemlProfile<'_>) {
        let (domain_lo, domain_hi) = profile.log_lambda_domain().expect("domain");
        let span = domain_hi - domain_lo;

        for cell in 0..24 {
            let width = span * 0.5_f64.powi(cell as i32 % 12);
            let start = domain_lo + (span - width) * (cell / 12) as f64;
            let (lo, hi) = (start, (start + width).min(domain_hi));
            if !(hi > lo) {
                continue;
            }
            let left = sample_at(profile, lo);
            let right = sample_at(profile, hi);
            let enclosure = profile.enclose(left, right).expect("enclosure");
            for step in 0..=32 {
                let x = lo + (hi - lo) * step as f64 / 32.0;
                let jet = profile.evaluate(x).expect("jet").jet;
                assert!(
                    enclosure.derivative.contains(jet.derivative),
                    "derivative {} at {x} escapes {:?} on cell [{lo}, {hi}]",
                    jet.derivative,
                    enclosure.derivative
                );
                assert!(
                    enclosure.curvature.contains(jet.curvature),
                    "curvature {} at {x} escapes {:?} on cell [{lo}, {hi}]",
                    jet.curvature,
                    enclosure.curvature
                );
            }
        }
    }

    fn sample_at(profile: &CascadeRemlProfile<'_>, x: f64) -> ScoreSample {
        let jet = profile.evaluate(x).expect("cascade jet").jet;
        ScoreSample {
            x,
            value: jet.value,
            derivative: jet.derivative,
            curvature: jet.curvature,
            third: jet.third,
        }
    }

    #[test]
    fn dense_spectral_profile_matches_factorization_and_analytic_slope() {
        let (x1, x2, y) = dense_fixture(6);
        let weights = vec![1.0; y.len()];
        let axes: [&[f64]; 2] = [&x1, &x2];
        let design = ResidualCascadeDesign::build(&axes, &y, &weights, &[1.0, 1.0], 2.0, 2)
            .expect("cascade design");
        assert!(design.core.dense_gram.is_some());
        let profile = design.core.reml_profile().expect("spectral profile");
        let rank = (design.core.m - design.core.nullity()) as f64;
        let dof = (design.core.y.len() - design.core.nullity()) as f64;

        for log_lambda in [-4.0, 0.0, 3.0] {
            let evaluation = profile.evaluate(log_lambda).expect("analytic score");
            let lambda = log_lambda.exp();
            let logdet = design.core.logdet_dense(lambda).expect("dense logdet");
            let coefficients = design
                .core
                .solve_coeff(lambda, &design.core.rhs, None)
                .expect("dense solve")
                .0;
            let rss = design.core.rss_pen(&coefficients);
            let direct = -0.5
                * (logdet - rank * log_lambda - design.core.pen_logdet_const
                    + dof * (rss / dof).ln());
            assert!(
                (evaluation.jet.value - direct).abs() <= f64::EPSILON.sqrt() * (1.0 + direct.abs()),
                "spectral/direct score mismatch at {log_lambda}: {} versus {direct}",
                evaluation.jet.value,
            );

            // Finite differences are confined to this oracle test. The
            // production optimizer consumes the hand-derived score jet above.
            //
            // The comparator has to be built for a NOISY evaluator, and this
            // one is: `profile.evaluate` runs a spectral solve, so its value
            // carries roughly 1e-12 of evaluation noise rather than being exact
            // to the last bit.
            //
            // That moves the optimal step. `h = eps^(1/3)` is optimal only when
            // the sole error is representation roundoff; against noise `v` the
            // central-difference error is `v/h + (h²/6)·S3`, minimized at
            // `h ~ (3v/S3)^(1/3) ~ 1e-4` — three orders ABOVE `eps^(1/3)`.
            // Measured at `eps^(1/3) = 6.06e-6`: D(h) = 5.025511346842242,
            // D(h/2) = 5.025511611442345. The two stencils disagree by 2.6e-7
            // and the FINER one is FARTHER from the analytic slope. Truncation
            // shrinks with h and cannot do that; noise amplified by `1/h` does
            // exactly that. The step was too small, not too crude.
            //
            // So: `h = 1e-4`, and Richardson there. The `h²` term cancels
            // exactly (leaving O(h⁴) ~ 1e-16, negligible whatever `S3` is) and
            // the noise floor is `~3v/h ~ 3e-8`, inside the unchanged
            // `sqrt(eps)·(1+|S'|) ~ 9e-8` bound. The bound is not relaxed; the
            // comparator is made accurate enough to be charged against it.
            let central = |step: f64| -> f64 {
                let right = profile
                    .evaluate(log_lambda + step)
                    .expect("right score")
                    .jet
                    .value;
                let left = profile
                    .evaluate(log_lambda - step)
                    .expect("left score")
                    .jet
                    .value;
                (right - left) / (2.0 * step)
            };
            let step = 1.0e-4;
            let coarse = central(step);
            let fine = central(0.5 * step);
            let numerical_slope = (4.0 * fine - coarse) / 3.0;
            assert!(
                (evaluation.jet.derivative - numerical_slope).abs()
                    <= f64::EPSILON.sqrt() * (1.0 + numerical_slope.abs()),
                "analytic slope mismatch at {log_lambda}: {} versus {numerical_slope} \
                 (Richardson of h={step:e} → {coarse}, h/2 → {fine})",
                evaluation.jet.derivative,
            );
        }
    }
    /// A fixture whose rank is far above the quadrature's accepted step count,
    /// so the truncated regime is what gets measured, while staying under the
    /// dense cap so an exact comparator survives.
    fn truncated_regime_fixture() -> (Vec<f64>, Vec<f64>, Vec<f64>, usize) {
        let (x1, x2, y) = dense_fixture(28);
        (x1, x2, y, 5)
    }

    fn cascade_core(
        side_data: (&[f64], &[f64], &[f64]),
        levels: usize,
    ) -> ResidualCascadeDesign {
        let (x1, x2, y) = side_data;
        let weights = vec![1.0; y.len()];
        let axes: [&[f64]; 2] = [x1, x2];
        ResidualCascadeDesign::build(&axes, y, &weights, &[1.0, 1.0], 2.0, levels)
            .expect("cascade design")
    }

    /// #2503 measurement — what the beta-seeded Golub–Meurant residual quadrature
    /// accepts, at what step count, and how far its moments then sit from the
    /// exact dense eigenbasis over the whole log-lambda domain.
    #[test]
    fn zz_measure_residual_quadrature_admission_ladder_2503() {
        for (side, levels) in [
            (6usize, 2usize),
            (10, 3),
            (14, 3),
            (22, 3),
            (14, 4),
            (20, 5),
            (28, 5),
        ] {
            let (x1, x2, y) = dense_fixture(side);
            let design = cascade_core((&x1, &x2, &y), levels);
            let core = &design.core;
            if core.dense_gram.is_none() {
                println!("#2503 side={side} levels={levels} m={}: past the dense cap", core.m);
                continue;
            }
            let (null_chol, _) = core.null_gram_factor().expect("null factor");
            let (modes, exact) = core
                .dense_cascade_spectrum(&null_chol)
                .expect("dense spectrum");
            let domain = log_lambda_domain_from_modes(&modes).expect("domain");
            let (spectrum, certificate) = core
                .iterative_residual_spectrum(&null_chol, domain)
                .expect("quadrature");
            let rank = core.m - core.nullity();
            let anchor = exact.anchor_energy[0];
            let worst = spectrum.as_ref().map(|spectrum| {
                let mut worst = [0.0_f64; 4];
                for step in 0..=192 {
                    let lambda =
                        (domain.0 + (domain.1 - domain.0) * step as f64 / 192.0).exp();
                    let t = exact.moments(lambda);
                    let g = spectrum.moments(lambda);
                    worst[0] = worst[0].max((g.0 - t.0).abs() / anchor.abs());
                    for (k, (tv, gv)) in
                        [(t.1, g.1), (t.2, g.2), (t.3, g.3)].into_iter().enumerate()
                    {
                        worst[k + 1] =
                            worst[k + 1].max((gv - tv).abs() / tv.abs().max(f64::MIN_POSITIVE));
                    }
                }
                worst
            });
            println!(
                "#2503 side={side} levels={levels} n={} m={} rank={rank} budget={} steps={} \
                 coarse={} tail_estimate={:.3e} target={:.3e} certified={} tail={:.2e} \
                 mass_defect={:.2e} dropped={:.2e}",
                y.len(),
                core.m,
                certificate.budget,
                certificate.steps,
                certificate.coarse_steps,
                certificate.tail_estimate,
                certificate.target,
                certificate.certified,
                certificate.relative_tail,
                certificate.mass_defect,
                certificate.dropped_mass_fraction,
            );
            match worst {
                Some(worst) => println!(
                    "#2503   versus exact dense: dR/anchor={:.3e} S2={:.3e} S3={:.3e} S4={:.3e}",
                    worst[0], worst[1], worst[2], worst[3]
                ),
                None => println!("#2503   no rule admitted; the route solves at every lambda"),
            }
        }
    }

    /// The admitted quadrature IS the dense eigenbasis residual, to the resolution
    /// its own certificate claims.
    ///
    /// This is the substitution gate: past the dense cap the profiled residual and
    /// its three `log λ` derivative moments come from a Golub–Meurant rule instead
    /// of an exact projection, and if the two disagree anywhere in the domain the
    /// REML criterion is route-dependent — which is the defect the spectral form
    /// exists to remove.
    ///
    /// The fixture asserts its own premise. `rank ≫ steps` is what puts the rule
    /// in the TRUNCATED regime; at `steps == rank` the Krylov space is the whole
    /// space and the rule reproduces the spectral sum by dimension alone, so a gate
    /// that silently drifted into that case would measure arithmetic and not
    /// quadrature. #2503's first accuracy gate did exactly that.
    #[test]
    fn admitted_residual_quadrature_matches_the_dense_eigenbasis_2503() {
        let (x1, x2, y, levels) = truncated_regime_fixture();
        let design = cascade_core((&x1, &x2, &y), levels);
        let core = &design.core;
        assert!(
            core.dense_gram.is_some(),
            "the exact comparator needs the dense route"
        );
        let (null_chol, _) = core.null_gram_factor().expect("null factor");
        let (modes, exact) = core
            .dense_cascade_spectrum(&null_chol)
            .expect("dense spectrum");
        let domain = log_lambda_domain_from_modes(&modes).expect("domain");
        let (spectrum, certificate) = core
            .iterative_residual_spectrum(&null_chol, domain)
            .expect("quadrature");
        let rank = core.m - core.nullity();
        assert!(
            certificate.certified,
            "the quadrature must certify on this fixture: {certificate:?}"
        );
        assert!(
            certificate.steps * 2 < rank,
            "premise: the accepted rule must be TRUNCATED (steps {} against rank {rank}), else \
             this gate measures arithmetic and not quadrature",
            certificate.steps
        );
        assert!(
            certificate.tail_estimate <= certificate.target,
            "the extrapolated tail must reach the search resolution: {certificate:?}"
        );
        let spectrum = spectrum.expect("a certified rule is returned");

        // The comparator is the exact projection, and the bound is the resolution
        // the certificate claims — not a widened number. `R` is charged
        // ABSOLUTELY against the anchor energy: `R = anchor − S₁` cancels to nine
        // digits at the bottom of an over-complete cascade's domain, so a relative
        // bound there would be a statement about cancellation.
        let anchor = exact.anchor_energy[0];
        let resolution = f64::EPSILON.sqrt();
        for step in 0..=192 {
            let log_lambda = domain.0 + (domain.1 - domain.0) * step as f64 / 192.0;
            let lambda = log_lambda.exp();
            let (r_exact, s2_exact, s3_exact, s4_exact) = exact.moments(lambda);
            let (r_gauss, s2_gauss, s3_gauss, s4_gauss) = spectrum.moments(lambda);
            assert!(
                (r_gauss - r_exact).abs() <= resolution * anchor.abs(),
                "profiled residual disagrees at log lambda {log_lambda}: {r_gauss} versus \
                 {r_exact} (anchor {anchor})"
            );
            for (name, gauss, exact_value) in [
                ("S2", s2_gauss, s2_exact),
                ("S3", s3_gauss, s3_exact),
                ("S4", s4_gauss, s4_exact),
            ] {
                assert!(
                    (gauss - exact_value).abs() <= resolution * exact_value.abs(),
                    "{name} disagrees at log lambda {log_lambda}: {gauss} versus {exact_value} \
                     ({certificate:?})"
                );
            }
        }
    }

    /// The admitted quadrature and the SOLVE it replaces are the same function of
    /// λ, measured on the iterative route itself.
    ///
    /// Every other gate here charges the quadrature against the dense eigenbasis,
    /// which means charging it on designs small enough to HAVE one. This one runs
    /// past the dense sizing cap — where there is no eigenbasis, no `dense_gram`,
    /// and the only other way to obtain `S₁..S₄` is the pair of PCG solves the
    /// shipped route used to perform at every λ. If those two disagree, the REML
    /// criterion is route-dependent, which is the defect the spectral form exists
    /// to remove; and this is the only angle from which that can be checked where
    /// it actually matters.
    ///
    /// The λ is chosen where PCG is well conditioned, because the comparator is
    /// the thing with the error bar: `cond(B + λI) ≤ (θmax + λ)/λ`, and the solve
    /// carries `CG_RTOL` backward error, so its forward error on `S₁` is about
    /// `CG_RTOL · cond` and on `S₃, S₄` — which pass through a second solve —
    /// about its square. The bound below is that product with the measured
    /// conditioning substituted, not a widened number.
    #[test]
    fn the_quadrature_and_the_solve_it_replaces_agree_past_the_dense_cap_2503() {
        let (x1, x2, y) = dense_fixture(56);
        let design = cascade_core((&x1, &x2, &y), 6);
        let core = &design.core;
        assert!(
            core.dense_gram.is_none(),
            "premise: this fixture must be PAST the dense sizing cap (m = {}), or the solve is \
             not the comparator this gate is about",
            core.m
        );
        let (null_chol, _) = core.null_gram_factor().expect("null factor");
        let modes = core
            .iterative_cascade_spectrum(&null_chol)
            .expect("determinant modes");
        let domain = log_lambda_domain_from_modes(&modes).expect("domain");
        let (spectrum, certificate) = core
            .iterative_residual_spectrum(&null_chol, domain)
            .expect("quadrature");
        let spectrum = spectrum.unwrap_or_else(|| {
            panic!("the past-cap quadrature must certify on this fixture: {certificate:?}")
        });
        let theta_max = spectrum.eigenvalue.iter().copied().fold(0.0, f64::max);

        for log_lambda in [0.0_f64, 1.0, 2.0, 3.0] {
            let lambda = log_lambda.exp();
            let quadrature = spectrum.moment_sums(lambda);

            // Exactly what `CascadeRemlProfile::evaluate` does on the solve route.
            let solver = core.coeff_solver(lambda).expect("iterative solver");
            let coeff = solver
                .solve(core, lambda, &core.rhs)
                .expect("first certified solve");
            let dc: Vec<f64> = coeff
                .iter()
                .zip(core.pen_diag.iter())
                .map(|(&c, &d)| d * c)
                .collect();
            let u = solver
                .solve(core, lambda, &dc)
                .expect("second certified solve");
            let anchor = spectrum.anchor_energy[0];
            let solved = [
                anchor - core.rss_pen(&coeff),
                coeff.iter().zip(dc.iter()).map(|(&c, &v)| c * v).sum(),
                dc.iter().zip(u.iter()).map(|(&a, &b)| a * b).sum(),
                u.iter()
                    .zip(core.pen_diag.iter())
                    .map(|(&v, &d)| d * v * v)
                    .sum(),
            ];

            let conditioning = (theta_max + lambda) / lambda;
            for (k, (&got, &comparator)) in quadrature.iter().zip(solved.iter()).enumerate() {
                // One solve for `S_1, S_2`; two for `S_3, S_4`.
                let solves = if k < 2 { 1u32 } else { 2 };
                let bound = CG_RTOL * conditioning.powi(solves as i32) * comparator.abs();
                assert!(
                    (got - comparator).abs() <= bound,
                    "S{} disagrees between the quadrature and the solve it replaces at log \
                     lambda {log_lambda}: {got} versus {comparator} (bound {bound}, conditioning \
                     {conditioning}, {certificate:?})",
                    k + 1
                );
            }
        }
    }

    /// A scattered cloud with a bounding-box-filled net, sized so the iterative
    /// route is engaged and the penalized rank is far above the reachable Krylov
    /// dimension — the regime the #2503 integration fixtures live in.
    fn scattered_fixture(n: usize, seed: u64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut rng = SplitMix64::new(seed);
        let mut x1 = Vec::with_capacity(n);
        let mut x2 = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);
        for _ in 0..n {
            let a = rng.next_unit();
            let b = rng.next_unit();
            x1.push(a);
            x2.push(b);
            let truth = (2.0 * std::f64::consts::PI * a).sin()
                * (2.0 * std::f64::consts::PI * b).cos();
            y.push(truth + 0.1 * rng.next_normal());
        }
        (x1, x2, y)
    }

    /// Past the dense cap, a scattered design must EXHAUST its Krylov space inside
    /// the budget — because on these designs that is the only route to admission,
    /// and without admission the route falls back to the solve and #2503 returns.
    ///
    /// The measurement behind this gate is the reason the budget is stated the way
    /// it is. On every past-cap fixture the nested-rule tail estimate stays at
    /// `O(1)` at EVERY budget, and always for the same reason: the worst point is
    /// the domain's lower endpoint, where `S_k` is dominated by the smallest
    /// spectral modes and a truncated rule sees none of them. Contrast the dense
    /// regime, where the estimate reaches `1e-10` by 96–192 nodes. So past the cap
    /// the extrapolation never fires and the rule is admitted by exhaustion alone —
    /// which is not a weaker outcome but a stronger one, since an exhausted Krylov
    /// space makes the rule exact for every kernel and every λ.
    ///
    /// That makes the budget load-bearing in a way a step count is not: it must
    /// reach `min(rank, n − nullity)`, and a budget at 90% of that pays 90% of the
    /// work and then falls back to the solve anyway. The fixtures below are the
    /// three shapes the #2503 integration reds build, at the first level past the
    /// cap.
    #[test]
    fn past_cap_designs_exhaust_their_krylov_space_inside_the_budget_2503() {
        for (n, levels) in [(1200usize, 6usize), (2500, 6), (6000, 6)] {
            let (x1, x2, y) = scattered_fixture(n, 0x2503_0001);
            let weights = vec![1.0; y.len()];
            let axes: [&[f64]; 2] = [&x1, &x2];
            let design = ResidualCascadeDesign::build(&axes, &y, &weights, &[1.0, 1.0], 2.0, levels)
                .expect("cascade design");
            let core = &design.core;
            assert!(
                core.dense_gram.is_none(),
                "premise: n={n} levels={levels} must be PAST the dense cap (m = {})",
                core.m
            );
            let (null_chol, _) = core.null_gram_factor().expect("null factor");
            let modes = core
                .iterative_cascade_spectrum(&null_chol)
                .expect("determinant modes");
            let domain = log_lambda_domain_from_modes(&modes).expect("domain");
            let rank = core.m - core.nullity();
            let ceiling = core.residual_krylov_ceiling();
            let budget = core.residual_quadrature_budget();
            assert_eq!(
                budget, ceiling,
                "n={n} levels={levels}: the budget must reach the Krylov ceiling (rank {rank}); \
                 stopping short of it pays the work and still falls back to the solve"
            );
            let (spectrum, certificate) = core
                .iterative_residual_spectrum(&null_chol, domain)
                .expect("quadrature");
            println!(
                "#2503 n={n} levels={levels} m={} rank={rank} ceiling={ceiling} steps={} \
                 tail={:.2e} tail_estimate={:.3e} certified={} dropped={:.2e}",
                core.m,
                certificate.steps,
                certificate.relative_tail,
                certificate.tail_estimate,
                certificate.certified,
                certificate.dropped_mass_fraction,
            );
            assert!(
                certificate.certified && spectrum.is_some(),
                "n={n} levels={levels}: the past-cap quadrature must be admitted, else the \
                 criterion is evaluated by a solve at every λ and the domain endpoint refuses \
                 (#2503): {certificate:?}"
            );
            assert!(
                certificate.relative_tail <= f64::EPSILON * certificate.steps as f64,
                "n={n} levels={levels}: admission here must come from EXHAUSTION — the Krylov \
                 residual against the operator scale must be at the arithmetic floor: \
                 {certificate:?}"
            );
            assert!(
                certificate.steps <= ceiling,
                "the run cannot exceed the reachable dimension: {certificate:?}"
            );
        }
    }

    /// A Krylov space that has reached `rank(B) <= n - nullity` is invariant even
    /// when its numerical `tail` says otherwise, and the rule there IS the exact
    /// spectrum.
    ///
    /// This is the ceiling that decides whether the iterative route can ever leave
    /// the solve. A bounding-box-filled cascade has far more columns than the data
    /// can pin — measured on #2503's own `n = 800` 2-D fixture at refinement level
    /// 6, `rank = 7387` against `n - nullity = 797`, so 89% of the whitened Schur
    /// spectrum is exactly zero. Reading the invariance ceiling as `rank` made the
    /// test unreachable and the route fell back to the solve at full budget, which
    /// is exactly the wall this issue is about.
    ///
    /// The claim being gated is that the ceiling is a THEOREM and not an
    /// optimism: `B = Z'WZ` with `W^{1/2}Z = (I − P)W^{1/2}X₁` and `P` of rank
    /// `nullity`, so `rank(B) ≤ n − nullity`; `β = Z'Wy ∈ range(B)`; and the
    /// Krylov space cannot leave `range(B)`. The fixture puts the ceiling strictly
    /// below `rank` and charges the rule at the ceiling against the exact dense
    /// eigenbasis over the whole domain.
    #[test]
    fn a_krylov_space_at_the_rank_ceiling_reproduces_the_exact_spectrum_2503() {
        let (x1, x2, y) = dense_fixture(20);
        let design = cascade_core((&x1, &x2, &y), 5);
        let core = &design.core;
        assert!(core.dense_gram.is_some());
        let rank = core.m - core.nullity();
        let ceiling = core.residual_krylov_ceiling();
        assert!(
            ceiling < rank,
            "premise: this fixture must have MORE penalized columns than the data can pin \
             (rank {rank}, ceiling {ceiling}), or the ceiling is not being exercised"
        );
        assert_eq!(
            ceiling,
            y.len() - core.nullity(),
            "the ceiling must be `n - nullity` when that is the binding bound"
        );

        let (null_chol, _) = core.null_gram_factor().expect("null factor");
        let (modes, exact) = core
            .dense_cascade_spectrum(&null_chol)
            .expect("dense spectrum");
        let (lo, hi) = log_lambda_domain_from_modes(&modes).expect("domain");
        let (beta, anchor) = core.whitened_residual_rhs(&null_chol);
        let mass = beta.iter().map(|value| value * value).sum::<f64>();
        let run = core
            .schur_lanczos(&null_chol, &beta, ceiling, ceiling)
            .expect("lanczos");
        assert_eq!(run.alpha.len(), ceiling, "the run must reach the ceiling");
        assert!(
            run.invariant,
            "a run that consumed the whole reachable dimension must report invariance \
             (tail/scale {})",
            run.tail / run.spectral_scale
        );
        let rule = core
            .residual_gauss_rule(&run, ceiling, anchor, mass)
            .expect("gauss rule");

        let resolution = f64::EPSILON.sqrt();
        for step in 0..=192 {
            let lambda = (lo + (hi - lo) * step as f64 / 192.0).exp();
            let (r_exact, s2, s3, s4) = exact.moments(lambda);
            let (r_gauss, g2, g3, g4) = rule.spectrum.moments(lambda);
            assert!(
                (r_gauss - r_exact).abs() <= resolution * anchor.abs(),
                "profiled residual at the ceiling disagrees at lambda {lambda}: {r_gauss} \
                 versus {r_exact}"
            );
            for (name, got, truth) in [("S2", g2, s2), ("S3", g3, s3), ("S4", g4, s4)] {
                assert!(
                    (got - truth).abs() <= resolution * truth.abs(),
                    "{name} at the ceiling disagrees at lambda {lambda}: {got} versus {truth}"
                );
            }
        }
    }

    /// A Ritz node whose WEIGHT is roundoff must not reach the spectrum, because
    /// `(θ + λ)^{-k}` will amplify it by `λ^{-k}` at the bottom of the domain.
    ///
    /// The mechanism, measured: on this fixture at 96 steps one node lands at
    /// `θ = 2.6e-11` — above the eigenvalue roundoff floor, so that floor passes
    /// it — carrying weight `8.9e-27·‖β‖²`. At `λ ≈ 2.9e-11` it contributes
    /// `w/(θ+λ)⁴` and `S₄` comes out `3.6e7` RELATIVE off while `S₂` is still
    /// right to `6e-9`. #2503 read that as quadrature truncation and concluded the
    /// approach was refuted; it is one node of pure roundoff.
    ///
    /// The test builds BOTH rules from the SAME Lanczos run — the shipped one and
    /// one with the weight floor removed — so it names the mechanism rather than
    /// asserting a number that some other change could also produce.
    #[test]
    fn roundoff_weight_nodes_cannot_poison_the_derivative_moments_2503() {
        let (x1, x2, y) = dense_fixture(14);
        let design = cascade_core((&x1, &x2, &y), 4);
        let core = &design.core;
        assert!(core.dense_gram.is_some());
        let (null_chol, _) = core.null_gram_factor().expect("null factor");
        let (modes, exact) = core
            .dense_cascade_spectrum(&null_chol)
            .expect("dense spectrum");
        let (lo, hi) = log_lambda_domain_from_modes(&modes).expect("domain");
        let (beta, anchor) = core.whitened_residual_rhs(&null_chol);
        let mass = beta.iter().map(|value| value * value).sum::<f64>();
        let steps = 96;
        let run = core
            .schur_lanczos(&null_chol, &beta, steps, core.residual_krylov_ceiling())
            .expect("lanczos");
        assert_eq!(run.alpha.len(), steps, "premise: the run must not close early");
        let shipped = core
            .residual_gauss_rule(&run, steps, anchor, mass)
            .expect("gauss rule");

        // The same run, with ONLY the eigenvalue floor — what the weight floor is
        // being charged against.
        let (ritz, first) =
            symmetric_tridiagonal_eigen(&run.alpha, &run.beta[..steps - 1]).expect("eigen");
        let scale = ritz.iter().copied().map(f64::abs).fold(0.0, f64::max);
        let eigenvalue_floor = f64::EPSILON * steps as f64 * scale;
        let mut eigenvalue = Vec::with_capacity(steps);
        let mut projected_square = Vec::with_capacity(steps);
        let mut poison = None;
        for (&theta, &component) in ritz.iter().zip(first.iter()) {
            let weight = run.start_norm_sq * component * component;
            if theta <= eigenvalue_floor {
                eigenvalue.push(0.0);
                projected_square.push(0.0);
                continue;
            }
            if theta < lo.exp() {
                poison = Some((theta, weight / mass));
            }
            eigenvalue.push(theta);
            projected_square.push(weight);
        }
        let unfloored = CascadeResidualSpectrum {
            eigenvalue,
            penalty: vec![1.0; steps],
            projected_square,
            anchor_energy: [anchor],
        };
        let (theta, relative_weight) = poison.expect(
            "premise: the eigenvalue floor alone must leave a node below the domain's smallest \
             lambda, which is the node this test is about",
        );
        let component_roundoff = f64::EPSILON * steps as f64;
        assert!(
            relative_weight <= component_roundoff * component_roundoff,
            "premise: that node's weight must be the SQUARE of the roundoff in a Ritz vector's \
             first component, `(eps*m)^2 = {}` (theta {theta}, w/||beta||^2 {relative_weight})",
            component_roundoff * component_roundoff
        );

        let bottom = lo.exp();
        let (_, _, _, s4_exact) = exact.moments(bottom);
        let (_, _, _, s4_unfloored) = unfloored.moments(bottom);
        let (_, _, _, s4_shipped) = shipped.spectrum.moments(bottom);
        let unfloored_error = (s4_unfloored - s4_exact).abs() / s4_exact.abs();
        let shipped_error = (s4_shipped - s4_exact).abs() / s4_exact.abs();
        assert!(
            unfloored_error > 1.0,
            "premise: without the weight floor S4 must be wrong by more than 100% at the domain \
             bottom (got {unfloored_error}); if it is not, this fixture no longer exercises the \
             mechanism"
        );
        assert!(
            shipped_error <= f64::EPSILON.sqrt(),
            "the weight floor must restore S4 at the domain bottom: relative error \
             {shipped_error} against {unfloored_error} unfloored"
        );
        assert!(
            shipped.dropped_mass_fraction <= f64::EPSILON,
            "the mass the floor dropped must itself be roundoff: {}",
            shipped.dropped_mass_fraction
        );
        // ...and dropping it must not disturb the moments the rule got right.
        for step in 0..=64 {
            let lambda = (lo + (hi - lo) * step as f64 / 64.0).exp();
            let (r_exact, s2_exact, ..) = exact.moments(lambda);
            let (r_gauss, s2_gauss, ..) = shipped.spectrum.moments(lambda);
            assert!(
                (r_gauss - r_exact).abs() <= f64::EPSILON.sqrt() * anchor.abs()
                    && (s2_gauss - s2_exact).abs() <= f64::EPSILON.sqrt() * s2_exact.abs(),
                "the floored rule must keep the moments it already had right at lambda {lambda}"
            );
        }
    }

    /// The admission rule's load-bearing claim: when the geometric tail estimate
    /// over three nested Gauss rules falls below `sqrt(eps)`, the finest rule
    /// really is that close to the EXACT spectrum.
    ///
    /// The certificate is a self-comparison — it never sees the truth — so the
    /// inference from "the ladder has contracted" to "the rule is right" is the
    /// thing that has to be measured. It rests on two facts and one model: every
    /// Gauss rule for a completely monotone kernel under-estimates its integral,
    /// so the ladder rises toward the truth; the gaps are therefore all of one
    /// sign and the remaining error is their tail; and the tail is extrapolated
    /// geometrically from the last two gaps, refusing outright when they do not
    /// contract. This charges that inference against the dense eigenbasis at every
    /// budget on the PRODUCTION ladder, over three designs — including the budgets
    /// the certificate REFUSES, where it must be the refusal that is right.
    #[test]
    fn the_quadrature_tail_estimate_bounds_the_error_against_the_exact_spectrum_2503() {
        for (side, levels) in [(14usize, 4usize), (20, 5), (28, 5)] {
            let (x1, x2, y) = dense_fixture(side);
            let design = cascade_core((&x1, &x2, &y), levels);
            let core = &design.core;
            assert!(core.dense_gram.is_some());
            let (null_chol, _) = core.null_gram_factor().expect("null factor");
            let (modes, exact) = core
                .dense_cascade_spectrum(&null_chol)
                .expect("dense spectrum");
            let (lo, hi) = log_lambda_domain_from_modes(&modes).expect("domain");
            let (beta, anchor) = core.whitened_residual_rhs(&null_chol);
            let mass = beta.iter().map(|value| value * value).sum::<f64>();
            let rank = core.m - core.nullity();
            let target = f64::EPSILON.sqrt();
            let ceiling = core.residual_krylov_ceiling();
            let budget = core.residual_quadrature_budget();
            let mut certified_at_least_once = false;
            let mut refused_at_least_once = false;

            // The same start and the same geometric growth
            // `iterative_residual_spectrum` walks, so what is charged here ships.
            let mut steps = SLQ_LANCZOS_STEPS.min(budget);
            loop {
                let run = core
                    .schur_lanczos(&null_chol, &beta, steps, ceiling)
                    .expect("lanczos");
                let taken = run.alpha.len();
                let rule = |nodes: usize| {
                    core.residual_gauss_rule(&run, nodes, anchor, mass)
                        .expect("gauss rule")
                        .spectrum
                };
                let (fine, mid, coarse) = (rule(taken), rule(taken / 2), rule(taken / 4));
                let estimate =
                    residual_quadrature_tail_estimate(&fine, &mid, &coarse, taken, (lo, hi))
                        .expect("tail estimate");

                let mut worst = 0.0_f64;
                for step in 0..=96 {
                    let lambda = (lo + (hi - lo) * step as f64 / 96.0).exp();
                    let truth = exact.moment_sums(lambda);
                    let got = fine.moment_sums(lambda);
                    for (truth, got) in truth.into_iter().zip(got) {
                        worst = worst.max((got - truth).abs() / truth.abs().max(f64::MIN_POSITIVE));
                    }
                }

                if run.invariant || estimate <= target {
                    certified_at_least_once = true;
                    assert!(
                        worst <= target,
                        "side={side} levels={levels} steps={taken}: the tail estimate was \
                         {estimate} (target {target}, invariant {}) but the rule is {worst} from \
                         the exact spectrum — the admission rule's inference is unsound",
                        run.invariant
                    );
                } else {
                    refused_at_least_once = true;
                }
                if taken >= budget {
                    break;
                }
                steps = (steps * 2).min(budget);
            }
            assert!(
                certified_at_least_once,
                "side={side} levels={levels}: the growth ladder must reach a certified rule \
                 inside the budget ({budget} of rank {rank}), or the route can never leave the \
                 solve"
            );
            assert!(
                refused_at_least_once,
                "side={side} levels={levels}: the ladder must also contain a REFUSED budget, or \
                 the criterion is not being exercised"
            );
        }
    }

}
