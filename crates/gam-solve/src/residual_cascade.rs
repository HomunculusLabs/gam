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
//! with the SAME spectral nodes at every trial. The dense route has a rigorous
//! interval extension that isolates every stationary interval before
//! safeguarded root refinement.
//!
//! Which designs get that proof is set by `CERTIFIED_SPECTRUM_MAX`, NOT by the
//! dense Gram cache. The exact spectrum is a dense symmetric eigendecomposition
//! of the `rank × rank` Schur complement, so its bound is that
//! decomposition's transient memory; the Gram cache's bound (`DENSE_GRAM_MAX`)
//! is a much tighter LIFETIME budget for a per-design array. Past the cache the
//! Schur source is assembled from the CSR rows for the duration of the
//! decomposition and dropped, so certification continues well past it — which
//! is what lets a cascade whose refinement crosses 1536 columns finish at all
//! (#2546). Past the SPECTRUM budget the route deliberately refuses automatic
//! λ selection: a fixed-probe SLQ log-determinant has no exact-real outer
//! enclosure, and neither does an exact factorization, which returns a number at
//! one λ and no enclosure over a λ cell. Closing the separate β-seeded
//! residual Krylov space can make the residual exact, but cannot enclose that
//! determinant; a merely converged residual tail is numerical point evidence
//! only. Fixed-λ fits remain available on the iterative route.
//!
//! The same elimination puts the profiled RESIDUAL in the same form, making the
//! diagnostic criterion solve-free at every λ whenever its residual rule is
//! admitted. With
//! `β = D^{−1/2}(b₁ − G₁₀G₀₀^{−1}b₀)` and `S_k(λ) = β'(B+λI)^{−k}β`, the residual
//! is `R = anchor − S₁` and its three `log λ` derivatives are built from
//! `S₂, S₃, S₄`; `anchor = y'Wy − b₀'G₀₀^{−1}b₀` is the part no λ can move. On
//! the dense route the eigenbasis projects β directly. Past the cap, `S_k(λ)` is
//! `∫(θ+λ)^{−k} dμ(θ)` for the measure β induces on `spec(B)`, so ONE Lanczos run
//! seeded with β (rather than with a Rademacher probe) returns the Jacobi matrix
//! of its Golub–Meurant Gauss rule — the same `(node, weight)` shape the dense
//! route stores, so both routes then evaluate one expression. That rule is
//! admitted only when it has earned point-evaluation trust: either the Krylov
//! space has consumed `rank(B) ≤ n − nullity` and is therefore invariant (the
//! rule is then exact for every kernel), or the gaps over the nested
//! `m/4, m/2, m` rules contract enough that their geometric tail estimate is at
//! most `√ε` over the whole domain. A failed admission is an error; the
//! ill-conditioned two-PCG-solves-per-λ fallback is not revived (#2503).
//! Neither residual admission encloses the independent fixed-probe SLQ
//! determinant, so neither authorizes automatic iterative-route REML (#2513).
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
//! or the level/center caps are reached: certified-or-typed-refusal, the same
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
use faer::sparse::{SparseColMat, SymbolicSparseColMat};
use gam_linalg::faer_ndarray::FaerEigh;
use gam_math::score_opt::{
    AffineRemlProfile, ScoreJet, certified_ln_positive,
};
use gam_linalg::sparse_exact::{
    SparseExactFactor, factorize_sparse_spd_strict, logdet_from_factor, solve_sparse_spd,
    sparse_spd_factor_nnz,
};
use gam_terms::grid_spline_2d::{chol_solve, cholesky_logdet};
use ndarray::{Array1, Array2};

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
///
/// This sizes a PERSISTENT per-design cache: `Core::dense_gram` is held for the
/// whole life of the design and reused by every solve and every λ. It does NOT
/// bound the certified REML proof — see [`CERTIFIED_SPECTRUM_MAX`], which is a
/// transient budget and reaches further.
const DENSE_GRAM_MAX: usize = 1536;

/// Column count up to which the CERTIFIED automatic-REML proof is available.
///
/// The proof is not a log-determinant, and that is the whole reason it needs its
/// own bound. It is the λ-INDEPENDENT Schur spectrum built by
/// [`Core::dense_cascade_spectrum`]: with `B = D^{−1/2}(G₁₁ − G₁₀G₀₀^{−1}G₀₁)
/// D^{−1/2} = VΘV'`, every determinant mode and every residual moment is an
/// analytic kernel of `θ_i + λ`, which is what lets
/// [`AffineRemlProfile::enclose`] produce genuine INTERVAL extensions of the
/// score value and its first two derivatives — the objects the KKT root and the
/// global candidate ordering are certified in.
///
/// An exact factorization of `X'WX + λD` does not substitute for that, however
/// exact it is. A factorization — dense or sparse-direct — is a POINTWISE
/// object: it returns a number at one λ and supports no enclosure over a λ cell,
/// so it can certify neither a score sign on an interval nor a stationary point.
/// (The former endpoint-jet/global-Lipschitz enclosure that tried to bridge that
/// gap did not collapse in saturated tails and was removed; see
/// [`CascadeRemlProfile::affine_view`].) The requirement is therefore ALL
/// eigenvalues of the `rank × rank` whitened Schur complement together with its
/// eigenvectors, one of which is projected onto the whitened response — a dense
/// symmetric eigendecomposition, and no sparse route replaces it.
///
/// So the bound is that eigendecomposition's LIVE MEMORY, and the width is
/// DERIVED from it: `sqrt(CERTIFIED_SPECTRUM_BYTES / (blocks · 8))` = 2896
/// columns at a 512 MiB peak over the [`CERTIFIED_SPECTRUM_BLOCKS`] `m × m`
/// blocks the route was measured to hold, all freed as soon as the modes are
/// extracted. Being a transient rather than a lifetime cache is why this budget
/// sits 1.9× further out in columns (3.6× in memory) than [`DENSE_GRAM_MAX`]:
/// past that cap the Gram is materialized from the CSR design for the duration
/// of the decomposition and dropped, instead of being kept for the fit.
///
/// Time is not the binding resource here and is not what the number is derived
/// from: the decomposition is `O(rank³)` and is paid ONCE per cascade depth
/// (`fit_reml` builds the profile once and the certified search then evaluates
/// mode sums, `O(modes)` per trial, with no linear algebra at all).
const CERTIFIED_SPECTRUM_MAX: usize =
    (CERTIFIED_SPECTRUM_BYTES / (CERTIFIED_SPECTRUM_BLOCKS * size_of::<f64>())).isqrt();

/// Live memory the certified spectral proof may hold at its peak. The largest
/// transient this crate asks of a workstation; [`CERTIFIED_SPECTRUM_MAX`] is
/// derived from it rather than chosen, so moving the budget moves the width and
/// the two cannot drift apart.
const CERTIFIED_SPECTRUM_BYTES: usize = 512 * 1024 * 1024;

/// `m × m` f64 blocks simultaneously resident at the certified route's peak.
///
/// MEASURED, not counted. The obvious inventory is four — the upper Gram the
/// Schur complement is assembled from, the Schur complement itself, the
/// eigenvector matrix `eigh` returns, and one working copy — and that is wrong,
/// because `eigh` is `faer`'s self-adjoint EVD behind `FaerEigh` and its
/// tridiagonalization allocates workspace this crate never names.
///
/// Re-measured at `b8745892a` with each width in its own process, the marginal
/// high-water mark over an `m = 891 -> 2038` step is **6.41 - 6.84** blocks
/// (54.4-60.4 MiB -> 218.7-235.7 MiB), flat across `RAYON_NUM_THREADS` 1/2/4/8,
/// so the declared count is the next power of two above it. The earlier reading
/// of 7.44 blocks (79.4 MiB -> 270.1 MiB) was taken in a shared process and is
/// superseded; see `zz_measure_certified_spectrum_peak_memory_2546` for why that
/// distinction decides the number.
///
/// That test re-measures it and fails if it ever exceeds this, because a count
/// below the realized one would let [`CERTIFIED_SPECTRUM_MAX`] admit a width
/// that overruns [`CERTIFIED_SPECTRUM_BYTES`].
///
/// The count is measured NEAR the cap and is not claimed constant in `m`. It is
/// not: with the cap lifted experimentally, the same refinement ladder reached
/// `m = 8435` at 26 GB resident (≈46 blocks) and the next level past it at
/// 198 GB before being stopped. So the budget holds where the cap admits and the
/// growth past it is superlinear in the block count as well as in `m²` — which is
/// the quantitative reason the cap is not simply raised to whatever a given
/// fixture asks for.
const CERTIFIED_SPECTRUM_BLOCKS: usize = 8;

/// Memory budget for the exact sparse-direct factor of `A = X'WX + λD`, stated
/// as nonzeros of `L`.
///
/// Past [`DENSE_GRAM_MAX`] the design is still SPARSE — a row touches the `O(1)`
/// bumps per level whose supports cover it, `O(qL)` nonzeros — so `A` is sparse
/// and has an exact sparse Cholesky. Nothing about a fixed-λ log-determinant
/// requires iteration or a stochastic estimate; what it requires is that the
/// FILL-IN fit in memory, and `nnz(A)` does not predict `nnz(L)`. So the
/// realized fill of the AMD ordering is measured by a symbolic pass before any
/// numeric work is committed, and compared against this budget.
///
/// The number is that budget divided by the factor's own per-entry cost: the
/// simplicial factor stores one `f64` value and one `usize` row index per
/// nonzero, 16 bytes, and 256 MiB of factor is the largest this route will pay,
/// so `256·2^20 / 16 = 16·2^20` nonzeros. Beyond it no exact factorization is
/// available at all and the log-determinant falls back to the stochastic
/// estimate — reported as [`LogdetMethod::Slq`], and never underwriting a proof,
/// since [`ResidualCascadeDesign::fit_reml`] refuses far below this width.
const SPARSE_FACTOR_MAX_NNZ: usize = 16 * 1024 * 1024;

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
/// metric-scaled per-axis spread so the selected route can refuse BEFORE
/// paying an unbounded iterative solve, rather than discovering
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
/// budget of `0.9 * ceiling` pays 90% of the work and then refuses point evaluation.
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
/// unreachable at any budget worth paying and the point route refuses rather
/// than reviving the ill-conditioned solve.
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
    /// Exact dense linear algebra: either the dense Cholesky of `X'WX + λD` at
    /// this λ, or the λ-independent Schur eigendecomposition the certified REML
    /// profile is built from. Both are exact; which one ran depends on whether a
    /// λ was fixed or selected.
    DenseExact,
    /// Exact sparse-direct Cholesky of `X'WX + λD` at this λ (AMD-ordered
    /// simplicial LLᵀ); the log-determinant is `2·Σ log L_jj`. Available past
    /// the dense Gram cache, where the design is still sparse.
    SparseExact,
    /// Diagonal control variate + stochastic Lanczos quadrature on fixed
    /// deterministic probes. NOT exact — the only route that is not, and taken
    /// only when the sparse factor's fill-in exceeds
    /// [`SPARSE_FACTOR_MAX_NNZ`].
    Slq,
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
    /// Number of original training rows. Restored fits retain this scalar even
    /// though their prediction-only core intentionally drops the row arrays.
    training_sample_size: std::num::NonZeroUsize,
    /// Dense-route prediction factor at the fit's λ. When present, pointwise
    /// variance uses this one Cholesky factor instead of refactoring the same
    /// precision matrix for every prediction point.
    predict_chol: Option<Vec<f64>>,
    /// Exact sparse-direct factor of `A = X'WX + λD` at THIS fit's λ, held when
    /// the design is past the dense Gram cache. The posterior variance is one
    /// solve per prediction point; replaying it through this factor is exact and
    /// `O(nnz(L))`, where the alternative is a fresh PCG per point whose
    /// backward error the point then inherits.
    predict_sparse: Option<Arc<SparseExactFactor>>,
    /// Coefficients: `dim+1` polynomial entries, then level blocks.
    pub coeff: Vec<f64>,
    /// Selected (or supplied) log smoothing parameter `log λ = log σ²/τ²`.
    log_lambda: f64,
    /// Profiled (or supplied) observation variance σ².
    pub sigma2: f64,
    /// Restricted log-likelihood at the fit, up to λ- and data-independent
    /// additive constants. Exact on every route whose log-determinant is exact
    /// — dense Cholesky, the certified Schur spectrum, or the sparse direct
    /// factor — and SLQ-estimated only when the sparse factor's fill-in
    /// exceeded its budget, which the fit's `logdet_method` reports.
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
    /// Automatic smoothing-parameter selection needs mathematical outer
    /// enclosures of the score value and derivatives over λ CELLS, which only
    /// the λ-independent Schur spectrum provides. Past
    /// [`CERTIFIED_SPECTRUM_MAX`] the dense eigendecomposition that spectrum is
    /// made of exceeds its memory budget, and what remains is a fixed-probe
    /// stochastic quadrature — a pointwise estimate, not an enclosure. Exact or
    /// numerically converged residual evidence cannot repair that gap, and
    /// neither can an exact factorization at a point.
    RemlScoreProofUnavailable {
        columns: usize,
        certified_spectrum_max: usize,
    },
    /// Stationary structure could not be isolated even though the score was
    /// certified flat at its representable value resolution.
    RemlOptimumResolutionFlat {
        lo: f64,
        hi: f64,
        max_score_gap: f64,
        score_resolution: f64,
    },
    /// The certified 1-D score search could not decompose the λ domain within
    /// the subdivision budget derived from that domain and the requested
    /// resolution, so no λ was selected.
    ///
    /// Carries the identifiability of the design because that is the cause
    /// whenever `rank > identifiable`: past the data's own rank the profiled
    /// residual is an interpolation and the score is flat by rank deficiency
    /// over whole stretches of λ, where there is no stationary point to isolate
    /// and no derivative sign to exclude a cell by — so the search subdivides
    /// every cell it reaches and its cost is exponential in the domain's
    /// subdivision depth (#2546). `rank <= identifiable` means the budget was
    /// hit for some other flat-criterion reason and the numbers say so.
    RemlScoreSearchUndecomposable {
        columns: usize,
        rank: usize,
        identifiable: usize,
        subdivisions: usize,
        budget: usize,
        log_lambda_lo: f64,
        log_lambda_hi: f64,
    },
    /// Rounded candidate ordering is wider than its certified comparison
    /// resolution, so no unique representative may be fitted.
    RemlValueOrderingUnresolved {
        maximum_excess: f64,
        comparison_resolution: f64,
    },
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
            Self::RemlScoreProofUnavailable {
                columns,
                certified_spectrum_max,
            } => write!(
                f,
                "residual cascade: automatic REML proof unavailable because the certified \
                 Schur eigendecomposition does not fit its memory budget ({columns} columns \
                 exceed {certified_spectrum_max}); without the lambda-independent spectrum the \
                 score has only a fixed-probe stochastic quadrature, which is a pointwise \
                 estimate and not an interval enclosure, so score signs, KKT roots, and global \
                 candidate ordering are uncertified even when the separate residual quadrature \
                 converges; use an explicitly fixed log lambda"
            ),
            Self::RemlOptimumResolutionFlat {
                lo,
                hi,
                max_score_gap,
                score_resolution,
            } => write!(
                f,
                "residual cascade: REML optimum is value-resolved but not stationary on \
                 [{lo}, {hi}] (maximum score gap {max_score_gap}, score resolution \
                 {score_resolution})"
            ),
            Self::RemlScoreSearchUndecomposable {
                columns,
                rank,
                identifiable,
                subdivisions,
                budget,
                log_lambda_lo,
                log_lambda_hi,
            } => {
                write!(
                    f,
                    "residual cascade: the certified REML score search spent {subdivisions} cell \
                     subdivisions on log lambda in [{log_lambda_lo}, {log_lambda_hi}] without \
                     decomposing it, exceeding the budget {budget} derived from that domain and \
                     the requested resolution"
                )?;
                if rank > identifiable {
                    write!(
                        f,
                        "; the design is rank deficient against its data — {rank} penalized Schur \
                         modes ({columns} columns) against {identifiable} identifiable directions \
                         — so the profiled residual interpolates and the score is flat by rank \
                         deficiency, with no stationary point to isolate; refine less, or fix log \
                         lambda explicitly"
                    )
                } else {
                    write!(
                        f,
                        "; the design is identified ({rank} penalized Schur modes from {columns} \
                         columns against {identifiable} identifiable directions), so the flat \
                         criterion has some other cause and the budget is reporting it rather \
                         than diagnosing it"
                    )
                }
            }
            Self::RemlValueOrderingUnresolved {
                maximum_excess,
                comparison_resolution,
            } => write!(
                f,
                "residual cascade: selected REML representative can trail another exact \
                 candidate by {maximum_excess}, beyond comparison resolution \
                 {comparison_resolution}"
            ),
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
/// prerequisite). Holds everything `predict` and sample-size-based reporting
/// need and no training row values:
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
    /// Original training row count. Required on the wire.
    pub training_sample_size: std::num::NonZeroU64,
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

/// Numerical evidence for the profiled-residual point quadrature past the dense
/// cap.
///
/// This is deliberately not a REML proof object. Krylov invariance makes the
/// residual rule exact, but a geometric tail estimate is only point-evaluation
/// evidence. Neither case encloses the independent stochastic determinant, so
/// [`ResidualCascadeDesign::fit_reml`] refuses the entire iterative route before
/// this evidence can participate in fit certification.
#[derive(Clone, Copy, Debug)]
struct ResidualQuadratureEvidence {
    /// Nodes in the rule this evidence is about.
    steps: usize,
    /// Nodes in the nested coarser rule it was charged against; `0` when the
    /// Krylov space closed and no comparison was needed.
    coarse_steps: usize,
    /// Penalized Schur rank. The reachable Krylov dimension may be smaller,
    /// because `rank(B) <= n - nullity`.
    rank: usize,
    /// Step budget the growth loop was allowed, before that ceiling.
    budget: usize,
    /// `||r_m|| / max_i |alpha_i|`: the Krylov residual against the operator
    /// scale. At roundoff, `K_m(B, beta)` is invariant and the rule is exact.
    relative_tail: f64,
    /// Geometric estimate of the rule's remaining relative error over
    /// `S_1..S_4` and the whole lambda domain, from three nested rules.
    tail_estimate: f64,
    /// The resolution `tail_estimate` had to reach.
    target: f64,
    /// Whether the Krylov space closed, making the residual rule exact.
    invariant: bool,
    /// Whether the rule may be used for the diagnostic point criterion. This is
    /// never sufficient to authorize automatic REML.
    accepted_for_point_evaluation: bool,
    /// `|sum_j w_j / ||beta||^2 - 1|` — the free mass self-check of a Gauss rule.
    mass_defect: f64,
    /// Fraction of `||beta||^2` that landed on roundoff-level nodes and was
    /// dropped as null-space mass.
    dropped_mass_fraction: f64,
}

/// Extrapolated estimate of the profiled-residual quadrature's REMAINING relative
/// error, from three NESTED Gauss rules for the same measure, over `S_1..S_4` and
/// the whole `log lambda` domain.
///
/// An estimate, stated as one: it is a rate fitted to two observed gaps and
/// summed, not an inequality. It can authorize only the diagnostic point
/// criterion represented by [`ResidualQuadratureEvidence`], never automatic
/// REML: the independent stochastic determinant still lacks exact-real
/// value/derivative enclosures. The exact-dense comparison in
/// `the_quadrature_tail_estimate_bounds_the_error_against_the_exact_spectrum_2503`
/// charges the estimate against its intended numerical use at every production
/// budget.
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
/// Both forms describe the SAME function of lambda; they differ only in what
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
/// shape the dense route stores. Krylov invariance is exact residual evidence;
/// a contracting nested-rule tail can also support the explicitly diagnostic point
/// criterion, but is never promoted to an exact-real fit certificate. If neither
/// condition holds, construction refuses instead of retrying every point with
/// the ill-conditioned solve that caused #2503.
enum CascadeResidualForm {
    /// Exact eigenbasis projection under the dense cap. Interval-extendable via
    /// [`CascadeRemlProfile::affine_view`], because the determinant modes on
    /// this route are the SAME unit-weight modes.
    Spectral(CascadeResidualSpectrum),
    /// The Golub–Meurant point quadrature past the dense cap, admitted only for
    /// diagnostic score evaluation after its own numerical evidence passes.
    ///
    /// Never affine-viewable: this route's DETERMINANT modes are Hutchinson Ritz
    /// nodes with fractional weights, unrelated to the residual run's nodes.
    Quadrature(CascadeResidualSpectrum),
}

impl CascadeResidualForm {
    /// The lambda-independent spectral form, when this route has one.
    fn spectrum(&self) -> &CascadeResidualSpectrum {
        match self {
            Self::Spectral(spectrum) | Self::Quadrature(spectrum) => spectrum,
        }
    }
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
/// point-evaluation evidence has to be charged over exactly this interval, and
/// the residual is built while the profile is being assembled — the interval is
/// a function of the determinant modes alone, so it is available at that point.
fn certified_log_lambda_domain_from_modes(
    modes: &[CascadeSpectralMode],
) -> Result<(f64, f64), String> {
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
    let log_relative_resolution = certified_ln_positive(f64::EPSILON.sqrt()).ok_or_else(|| {
        "residual cascade: could not enclose the spectral-domain resolution".to_string()
    })?;
    let log_smallest = certified_ln_positive(smallest).ok_or_else(|| {
        "residual cascade: could not enclose the smallest spectral transition".to_string()
    })?;
    let log_largest = certified_ln_positive(largest).ok_or_else(|| {
        "residual cascade: could not enclose the largest spectral transition".to_string()
    })?;
    let minimum_log = certified_ln_positive(f64::MIN_POSITIVE).ok_or_else(|| {
        "residual cascade: could not enclose the minimum-normal logarithm".to_string()
    })?;
    let maximum_log = certified_ln_positive(f64::MAX).ok_or_else(|| {
        "residual cascade: could not enclose the maximum-finite logarithm".to_string()
    })?;
    let lo = log_smallest
        .add(log_relative_resolution)
        .lo
        .max(minimum_log.lo);
    let hi = log_largest
        .sub(log_relative_resolution)
        .hi
        .min(maximum_log.lo);
    if !(lo.is_finite() && hi.is_finite() && lo < hi) {
        return Err(format!(
            "residual cascade: invalid spectrum-derived log-lambda domain [{lo}, {hi}]"
        ));
    }
    Ok((lo, hi))
}

impl CascadeRemlProfile<'_> {
    fn log_lambda_domain(&self) -> Result<(f64, f64), String> {
        certified_log_lambda_domain_from_modes(&self.modes)
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
    /// collapses with the cell. The former endpoint-jet/global-Lipschitz
    /// enclosure did not collapse fast enough in saturated tails and has been
    /// removed.
    ///
    /// [`CascadeResidualForm::Quadrature`] is deliberately excluded even though
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
            // Every whitened mode carries penalty scale 1, and the certified-null
            // modes were already dropped when the spectrum was built (see
            // `dense_cascade_spectrum`), so the penalized determinant rank is the
            // number of POSITIVE Schur modes — which is what makes this
            // enclosure's width track the score's own and not the arithmetic's
            // failure to cancel `Z·log λ` against `rank·log λ`.
            spectrum.penalty.len(),
            self.null_logdet,
        )
        .map(Some)
        .map_err(|error| format!("residual cascade: affine spectral profile rejected: {error}"))
    }

    /// The normalized log-determinant and its first two `log lambda`
    /// derivatives.
    ///
    /// `O(modes)` and free of linear algebra on every route.
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
        // Both admitted forms are lambda-independent spectral sums. The
        // iterative constructor requires either an invariant Krylov space or a
        // sufficiently small nested-rule tail estimate instead of reviving the
        // ill-conditioned per-lambda solved fallback from #2503.
        let (rss, penalty_energy, inverse_penalty_energy, _third_energy) =
            self.residual.spectrum().moments(lambda);
        if !(rss.is_finite() && rss > 0.0) {
            return Err(format!(
                "residual cascade: degenerate penalized residual {rss}"
            ));
        }
        let rss_d1 = lambda * penalty_energy;
        let lambda2 = lambda * lambda;
        let rss_d2 = rss_d1 - 2.0 * lambda2 * inverse_penalty_energy;
        let DeterminantParts {
            normalized_logdet,
            first: determinant_d1,
            second: determinant_d2,
        } = self.determinant_parts(log_lambda, lambda);

        let dof = (core.y.len() - core.nullity()) as f64;
        let rss_log_d1 = rss_d1 / rss;
        let rss_log_d2 = rss_d2 / rss - rss_log_d1 * rss_log_d1;
        let jet = ScoreJet {
            value: -0.5 * (normalized_logdet + dof * (rss / dof).ln()),
            derivative: -0.5 * (determinant_d1 + dof * rss_log_d1),
            curvature: -0.5 * (determinant_d2 + dof * rss_log_d2),
            // The cascade criterion API does not consume a third derivative;
            // certified dense-route search evaluates the affine interval
            // extension directly, so no endpoint third derivative is needed.
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

}

/// Read `(row, col)` of a symmetric `m × m` Gram held as its row-major UPPER
/// triangle — the encoding of both `Core::dense_gram` and
/// [`Core::assemble_upper_gram`].
#[inline]
fn upper_gram_entry(gram: &[f64], m: usize, row: usize, col: usize) -> f64 {
    let (i, j) = if row <= col { (row, col) } else { (col, row) };
    gram[i * m + j]
}

impl Core {
    #[inline]
    fn dense_gram_entry(&self, row: usize, col: usize) -> Option<f64> {
        let gram = self.dense_gram.as_ref()?;
        Some(upper_gram_entry(gram, self.m, row, col))
    }

    /// Scatter the CSR design's row outer products into a fresh row-major UPPER
    /// `m × m` `X'WX`: one `O(nnz·q)` pass, the same assembly `build` performs
    /// under [`DENSE_GRAM_MAX`], just without the cap and without retaining the
    /// result on the design.
    fn assemble_upper_gram(&self) -> Vec<f64> {
        let m = self.m;
        let mut gram = vec![0.0_f64; m * m];
        for i in 0..self.w.len() {
            let lo = self.row_ptr[i];
            let hi = self.row_ptr[i + 1];
            for ea in lo..hi {
                let ca = self.col_idx[ea] as usize;
                let weighted = self.w[i] * self.vals[ea];
                for eb in ea..hi {
                    gram[ca * m + self.col_idx[eb] as usize] += weighted * self.vals[eb];
                }
            }
        }
        gram
    }

    /// The upper `X'WX` the certified spectral proof reads: the persistent cache
    /// when the design is narrow enough to carry one, otherwise a transient
    /// assembly from the CSR rows.
    ///
    /// `None` is the one honest refusal left — the design is wider than
    /// [`CERTIFIED_SPECTRUM_MAX`], so the dense eigendecomposition the proof is
    /// made of does not fit its memory budget. Crossing [`DENSE_GRAM_MAX`] alone
    /// no longer forfeits the proof.
    fn spectral_gram(&self) -> Option<std::borrow::Cow<'_, [f64]>> {
        if let Some(gram) = &self.dense_gram {
            return Some(std::borrow::Cow::Borrowed(gram.as_slice()));
        }
        if self.m > CERTIFIED_SPECTRUM_MAX {
            return None;
        }
        if self.w.is_empty() {
            // A core rebuilt from a persisted state keeps only a factored
            // precision; there is no CSR design left to assemble a Gram from,
            // and assembling one anyway would return zeros.
            return None;
        }
        Some(std::borrow::Cow::Owned(self.assemble_upper_gram()))
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

    /// Exact Schur spectrum, WITH the eigenbasis it is computed from.
    ///
    /// `gram` is the row-major UPPER `m × m` `X'WX` — [`Self::spectral_gram`]
    /// supplies either the persistent cache or a transient assembly, so this
    /// routine is available at every width inside
    /// [`CERTIFIED_SPECTRUM_MAX`] rather than only under [`DENSE_GRAM_MAX`].
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
        gram: &[f64],
    ) -> Result<(Vec<CascadeSpectralMode>, CascadeResidualSpectrum), String> {
        let m = self.m;
        let q = self.nullity();
        let rank = m - q;
        let mut schur = Array2::<f64>::zeros((rank, rank));
        let mut cross = vec![0.0; q];
        for j in 0..rank {
            for (k, value) in cross.iter_mut().enumerate() {
                *value = upper_gram_entry(gram, m, k, q + j);
            }
            let projected = chol_solve(null_chol, q, &cross);
            for i in 0..=j {
                let mut value = upper_gram_entry(gram, m, q + i, q + j);
                for (k, &coefficient) in projected.iter().enumerate() {
                    value -= upper_gram_entry(gram, m, q + i, k) * coefficient;
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
                entry -= upper_gram_entry(gram, m, q + i, k) * coefficient;
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
        // Certified-NULL modes are dropped, not carried as zeros, and the
        // penalized determinant rank drops with them.
        //
        // Exact, not an approximation: with `Z` null modes,
        // `Σ_i log(θ_i+λ) − rank·log λ = Σ_{θ>0} log(θ+λ) + Z·log λ − rank·log λ
        //  = Σ_{θ>0} log(θ+λ) − (rank−Z)·log λ`,
        // and a null mode's response energy is exactly zero (`Bv = 0` gives
        // `Zv = 0`, so `v'β = (Zv)'Wy = 0`), so the residual sum is unchanged
        // too. `determinant_parts` already skips `θ == 0` on the SCALAR path for
        // the same reason.
        //
        // Carrying them costs nothing on the scalar path and real width on the
        // INTERVAL path, which is why this is not cosmetic.
        // `AffineRemlProfile::enclose` evaluates the same expression in interval
        // arithmetic, where `Z·[log λ] − rank·[log λ]` does NOT cancel: it returns
        // a width proportional to `(Z + rank)·width(log λ)` where the real
        // function's width is proportional to the number of POSITIVE modes. On a
        // rank-deficient wide cascade — `m` columns against `n` rows with `m ≫ n`,
        // which is what box-filling nets produce on a small sample — `Z` is almost
        // all of `rank`, so every score enclosure is inflated by that ratio.
        //
        // What this does NOT do is make such a design certifiable, and the
        // measurement says so: a 36-row / 1725-column design still spins in
        // `AffineRemlProfile::enclose` under `maximize_score_1d` past 900 s with
        // all 1692 nulls dropped and only 33 modes left. So the inflation was not
        // that design's blocker; the unidentified end of its spectrum is, and that
        // is a separate defect. Dropping the nulls is kept because it is exact and
        // strictly tightens every enclosure, not because it fixed that case.
        let mut kept_eigenvalue = Vec::with_capacity(rank);
        let mut kept_projected_square = Vec::with_capacity(rank);
        for (index, &eigenvalue) in eigenvalues.iter().enumerate() {
            if certified(eigenvalue) > 0.0 {
                kept_eigenvalue.push(certified(eigenvalue));
                kept_projected_square.push(projected_square[index]);
            }
        }
        let kept = kept_eigenvalue.len();
        Ok((
            modes,
            CascadeResidualSpectrum {
                eigenvalue: kept_eigenvalue,
                penalty: vec![1.0; kept],
                projected_square: kept_projected_square,
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
    /// comparison in [`Self::iterative_residual_spectrum`] free of a second run.
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
        // `certified_log_lambda_domain_from_modes` pads with — a negative Ritz value is
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
    /// point-evaluation admissions are available and both are properties of the
    /// run, not of a calibrated budget:
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
    ///    estimate must fall below the diagnostic point resolution —
    ///    `sqrt(eps)`, the same constant `certified_log_lambda_domain_from_modes` uses for
    ///    endpoint padding. Measured against the exact dense eigenbasis at every
    ///    budget on this ladder, over three designs: where the estimate passed,
    ///    the rule was within `1e-12`.
    ///
    /// The budget GROWS geometrically until one of those fires, so the accepted
    /// rule is the smallest that passes rather than the largest affordable. That
    /// matters in both directions: past roughly 60% of the penalized rank the run
    /// starts producing near-null ghost nodes (measured: rank 473, from 256 steps
    /// on), so the cheapest passing rule is also the cleanest one.
    ///
    /// Returns `None` when neither admission holds. The caller turns that into a
    /// refusal; it never revives the per-`lambda` solve fallback.
    fn iterative_residual_spectrum(
        &self,
        null_chol: &[f64],
        domain: (f64, f64),
    ) -> Result<(Option<CascadeResidualSpectrum>, ResidualQuadratureEvidence), String> {
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
            // integrated by the empty rule, so this evidence is exact.
            return Ok((
                Some(CascadeResidualSpectrum {
                    eigenvalue: Vec::new(),
                    penalty: Vec::new(),
                    projected_square: Vec::new(),
                    anchor_energy: [anchor_energy],
                }),
                ResidualQuadratureEvidence {
                    steps: 0,
                    coarse_steps: 0,
                    rank,
                    budget,
                    relative_tail: 0.0,
                    tail_estimate: 0.0,
                    target,
                    invariant: true,
                    accepted_for_point_evaluation: true,
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
            let evidence = ResidualQuadratureEvidence {
                steps: taken,
                coarse_steps: if run.invariant { 0 } else { coarse_steps },
                rank,
                budget,
                relative_tail: run.tail / run.spectral_scale.max(f64::MIN_POSITIVE),
                tail_estimate,
                target,
                invariant: run.invariant,
                accepted_for_point_evaluation: run.invariant || tail_estimate <= target,
                mass_defect: fine.mass_defect,
                dropped_mass_fraction: fine.dropped_mass_fraction,
            };
            if evidence.accepted_for_point_evaluation {
                return Ok((Some(fine.spectrum), evidence));
            }
            if taken >= budget {
                return Ok((None, evidence));
            }
            steps = steps.saturating_mul(2).min(budget);
        }
    }

    /// Lanczos steps the profiled-residual quadrature may grow to past the dense
    /// cap.
    ///
    /// Not an accuracy dial — a rule is admitted by its own convergence, not by
    /// reaching a step count — so this bounds how large a Krylov space we are
    /// willing to REORTHOGONALIZE before declining the point evaluation. The binding
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
        // The exact spectral form is taken whenever the dense eigendecomposition
        // fits its memory budget — which is wider than the Gram cache, so a
        // design past `DENSE_GRAM_MAX` still gets the certifiable profile and
        // pays for the Gram only for the duration of the decomposition.
        let (modes, residual) = if let Some(gram) = self.spectral_gram() {
            let (modes, spectrum) = self.dense_cascade_spectrum(&null_chol, &gram)?;
            drop(gram);
            (modes, CascadeResidualForm::Spectral(spectrum))
        } else {
            let modes = self.iterative_cascade_spectrum(&null_chol)?;
            // The residual's numerical evidence is charged over exactly the
            // interval the diagnostic point criterion may visit, so the
            // determinant modes — which define that interval — are built first.
            let domain = certified_log_lambda_domain_from_modes(&modes)?;
            let (spectrum, evidence) =
                self.iterative_residual_spectrum(&null_chol, domain)?;
            let spectrum = spectrum.ok_or_else(|| {
                format!(
                    "residual cascade: profiled-residual quadrature did not resolve inside its \
                     resource-derived budget (steps {}, coarse steps {}, rank {}, budget {}, \
                     invariant {}, tail estimate {:.3e} against target {:.3e}, relative tail {:.3e}, \
                     mass defect {:.3e}, dropped mass fraction {:.3e}); refusing the \
                     ill-conditioned per-lambda solve fallback",
                    evidence.steps,
                    evidence.coarse_steps,
                    evidence.rank,
                    evidence.budget,
                    evidence.invariant,
                    evidence.tail_estimate,
                    evidence.target,
                    evidence.relative_tail,
                    evidence.mass_defect,
                    evidence.dropped_mass_fraction,
                )
            })?;
            (modes, CascadeResidualForm::Quadrature(spectrum))
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
                buf.sort_unstable_by(|x, y| x.total_cmp(y));
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
                buf.sort_unstable_by(|x, y| x.total_cmp(y));
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

    /// `A = X'WX + λD` as canonical symmetric-UPPER CSC, assembled from the CSR
    /// design.
    ///
    /// A sparse accumulator, not a triplet list. `X` is transposed once
    /// (`nnz(X)` entries), then column `j` of `A` is gathered by walking the rows
    /// that touch column `j` and accumulating each such row's entries with
    /// column index `≤ j` into a dense scratch of length `m`. Peak extra memory
    /// is `O(m + nnz(X) + nnz(A))`. A triplet list would instead materialize
    /// `Σ_i k_i(k_i+1)/2` entries — the full upper outer product of every row,
    /// duplicates included — which scales with the ROW count rather than with
    /// the design and is an order of magnitude past `nnz(A)` on a wide cascade.
    ///
    /// CSR rows are built polynomial-block-first and then level by level with
    /// ascending `col_offset`, so a row's column indices ascend; that is what
    /// lets the inner scan stop at the first column past `j`.
    fn sparse_upper_system(&self, lambda: f64) -> Result<SparseColMat<usize, f64>, String> {
        let m = self.m;
        let n = self.w.len();
        let mut x_col_ptr = vec![0_usize; m + 1];
        for &c in &self.col_idx {
            x_col_ptr[c as usize + 1] += 1;
        }
        for j in 0..m {
            x_col_ptr[j + 1] += x_col_ptr[j];
        }
        let nnz_x = x_col_ptr[m];
        let mut x_rows = vec![0_u32; nnz_x];
        let mut x_vals = vec![0.0_f64; nnz_x];
        {
            let mut cursor = x_col_ptr.clone();
            for i in 0..n {
                for e in self.row_ptr[i]..self.row_ptr[i + 1] {
                    let c = self.col_idx[e] as usize;
                    let slot = cursor[c];
                    x_rows[slot] = i as u32;
                    x_vals[slot] = self.vals[e];
                    cursor[c] = slot + 1;
                }
            }
        }
        let mut acc = vec![0.0_f64; m];
        let mut marked = vec![false; m];
        let mut touched: Vec<usize> = Vec::new();
        let mut col_ptr: Vec<usize> = Vec::with_capacity(m + 1);
        col_ptr.push(0);
        let mut row_idx: Vec<usize> = Vec::new();
        let mut values: Vec<f64> = Vec::new();
        for j in 0..m {
            touched.clear();
            for e in x_col_ptr[j]..x_col_ptr[j + 1] {
                let row = x_rows[e] as usize;
                let weighted = self.w[row] * x_vals[e];
                for f in self.row_ptr[row]..self.row_ptr[row + 1] {
                    let c = self.col_idx[f] as usize;
                    if c > j {
                        break;
                    }
                    if !marked[c] {
                        marked[c] = true;
                        touched.push(c);
                    }
                    acc[c] += weighted * self.vals[f];
                }
            }
            // The prior precision is diagonal, so it only ever lands on (j, j).
            // The diagonal is stored unconditionally: a column the data never
            // touches still carries `λ d_j`, and an all-zero column would make
            // the symbolic factorization see a structurally singular matrix.
            if !marked[j] {
                marked[j] = true;
                touched.push(j);
            }
            acc[j] += lambda * self.pen_diag[j];
            touched.sort_unstable();
            for &c in &touched {
                let value = acc[c];
                acc[c] = 0.0;
                marked[c] = false;
                if !value.is_finite() {
                    return Err(format!(
                        "residual cascade: non-finite sparse normal-equation entry ({c}, {j}) = {value}"
                    ));
                }
                if value != 0.0 || c == j {
                    row_idx.push(c);
                    values.push(value);
                }
            }
            col_ptr.push(row_idx.len());
        }
        let symbolic = SymbolicSparseColMat::<usize>::new_checked(m, m, col_ptr, None, row_idx);
        Ok(SparseColMat::<usize, f64>::new(symbolic, values))
    }

    /// Exact sparse-direct factor of `A = X'WX + λD` at this λ, or `None` when
    /// the AMD ordering's MEASURED fill-in exceeds [`SPARSE_FACTOR_MAX_NNZ`].
    ///
    /// The symbolic phase is run twice — once here to price the fill before
    /// committing, once inside the factorization — because the decision has to
    /// be made from `nnz(L)` and only the symbolic phase can supply it. That is
    /// `O(nnz(A))` against the numeric phase's `O(Σ_j nnz(L_{:,j})²)`, so it is
    /// not the cost being controlled.
    fn sparse_exact_factor(&self, lambda: f64) -> Result<Option<SparseExactFactor>, String> {
        let a = self.sparse_upper_system(lambda)?;
        let nnz_a = a.compute_nnz();
        let nnz_l = sparse_spd_factor_nnz(&a).map_err(|error| {
            format!("residual cascade: sparse normal-equation symbolic analysis failed: {error}")
        })?;
        log::info!(
            "[2546-FILL] m={} nnz(A)={nnz_a} nnz(L)={nnz_l} dense_upper={} \
             fill_vs_A={:.2} fraction_of_dense={:.4}",
            self.m,
            self.m * (self.m + 1) / 2,
            nnz_l as f64 / nnz_a.max(1) as f64,
            2.0 * nnz_l as f64 / (self.m as f64 * (self.m as f64 + 1.0))
        );
        if nnz_l > SPARSE_FACTOR_MAX_NNZ {
            return Ok(None);
        }
        factorize_sparse_spd_strict(&a).map(Some).map_err(|error| {
            format!(
                "residual cascade: exact sparse factorization of X'WX + {lambda} D failed \
                 (m = {}, nnz(A) = {nnz_a}, nnz(L) = {nnz_l}): {error}",
                self.m
            )
        })
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

    /// Log-determinant through the most exact route available, with the route
    /// it took.
    ///
    /// `sparse` is the factor the caller has already built at this λ (see
    /// [`Self::sparse_exact_factor`]); one factorization serves both this
    /// determinant and every subsequent prediction solve, so it is threaded in
    /// rather than rebuilt here.
    fn logdet_with(
        &self,
        lambda: f64,
        sparse: Option<&SparseExactFactor>,
    ) -> Result<(f64, LogdetMethod), String> {
        if self.dense_gram.is_some() {
            return Ok((self.logdet_dense(lambda)?, LogdetMethod::DenseExact));
        }
        if let Some(factor) = sparse {
            // `2·Σ log L_jj` from an exact factorization of the very matrix
            // whose determinant is asked for. Not iterative, not stochastic.
            let logdet = logdet_from_factor(factor).map_err(|error| {
                format!("residual cascade: sparse log-determinant unavailable: {error}")
            })?;
            return Ok((logdet, LogdetMethod::SparseExact));
        }
        // The one route left that is not exact, and the only place a stochastic
        // determinant survives: the AMD ordering's fill-in on this design does
        // not fit `SPARSE_FACTOR_MAX_NNZ`, so no exact factorization exists to
        // read a diagonal off. It is REPORTED as `Slq` on the fit's certificate
        // and it underwrites nothing — `fit_reml` refuses at a far smaller
        // width, so no score sign, KKT root, or candidate ordering can ever
        // rest on this value.
        Ok((self.logdet_slq(lambda)?, LogdetMethod::Slq))
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
    /// iteration bound is trustworthy. When this returns `false`, automatic
    /// fitting must return a typed refusal rather than pay an iterative solve
    /// whose iteration count is no longer n-independent. The CG residual
    /// certificate would still *catch* a mis-solve at [`CG_MAX_ITERS`], but
    /// the guard prevents the silent O(n·iters) blow-up up front.
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

    /// The bounded `log λ` domain used by the exact dense REML search and by
    /// iterative-route diagnostic point evaluation — every determinant
    /// transition `λ ≈ θ`, padded by `ln(1/√ε)` past the extreme Schur modes.
    ///
    /// Exposed because the ENDPOINTS are where a criterion evaluation is hardest:
    /// `maximize_score_1d` evaluates the lower boundary before anything else, and
    /// on the iterative route that is the λ at which `X'WX + λD` is numerically
    /// singular (#2503). A gate on "the criterion is evaluable everywhere the
    /// profile may look" needs to know where that is, rather than hard-coding a
    /// λ read out of one failure's message. This does not authorize automatic
    /// iterative REML; [`Self::fit_reml`] returns a typed proof refusal there.
    ///
    /// Rebuilds the whole REML profile, exactly as [`Self::criterion`] does — both
    /// are single-shot oracles, not loop bodies. Past the dense cap that includes
    /// the determinant sweep and the residual quadrature, so calling either in a λ
    /// loop pays the profile per λ; [`Self::fit_reml`] builds it once.
    pub fn log_lambda_domain(&self) -> Result<(f64, f64), String> {
        self.core.reml_profile()?.log_lambda_domain()
    }

    /// Profiled-σ² REML criterion at `log λ` (differences across λ are
    /// exact-real certifiable on the dense route; one fixed numerical spectral
    /// quadrature is used for diagnostic evaluation past the cap).
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
        profile_normalized_logdet: Option<f64>,
    ) -> Result<ResidualCascadeFit, String> {
        let core = &self.core;
        let lambda = gam_problem::checked_exp_log_strength(log_lambda)
            .map_err(|error| format!("residual cascade: {error}"))?;
        // Exact sparse-direct factor at this λ, past the dense Gram cache. One
        // factorization serves the log-determinant AND every prediction solve
        // this fit will later perform, so it is built once here. A core rebuilt
        // from a persisted state has no CSR design to assemble it from and
        // carries its own dense factor instead.
        let sparse_factor = if core.dense_gram.is_none() && core.predict_chol.is_none() {
            core.sparse_exact_factor(lambda)?.map(Arc::new)
        } else {
            None
        };
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
        let (logdet, logdet_method) = match profile_normalized_logdet {
            // Supplied only by `fit_reml`, which refuses unless the exact
            // lambda-independent Schur spectrum was formed — so a normalized
            // logdet arriving here is exact dense linear algebra by
            // construction, at every width the certified route admits.
            Some(normalized) => (
                normalized + r * log_lambda + core.pen_logdet_const,
                LogdetMethod::DenseExact,
            ),
            None => core.logdet_with(lambda, sparse_factor.as_deref())?,
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
            training_sample_size: std::num::NonZeroUsize::new(core.y.len())
                .expect("ResidualCascadeDesign requires training rows"),
            predict_chol,
            predict_sparse: sparse_factor,
            coeff,
            log_lambda,
            sigma2,
            restricted_loglik,
            rss_pen,
            certificate: CascadeCertificate {
                solve_rel_residual: rel_res,
                solve_iters: iters,
                logdet_method,
            },
            refinement: None,
        })
    }

    /// Fit with `log λ` selected by the profiled REML criterion. Every
    /// stationary interval in the bounded domain is isolated from analytic
    /// derivative enclosures, refined by safeguarded Newton/bisection, and
    /// compared with both exact boundary candidates.
    ///
    /// Automatic selection is limited to designs whose λ-independent Schur
    /// spectrum can be formed, i.e. inside [`CERTIFIED_SPECTRUM_MAX`] — NOT to
    /// designs that carry a dense Gram cache. The two used to be the same gate,
    /// which meant a cascade whose refinement legitimately crossed
    /// [`DENSE_GRAM_MAX`] could be fitted but never certified, and so could not
    /// finish at all (#2546). The Gram is a cache; the spectrum is the proof.
    ///
    /// Past the spectrum budget there is no exact-real enclosure of the score,
    /// even when the separate β-seeded residual Krylov space closes: a pointwise
    /// solve, factorization, or quadrature value cannot certify a score sign, a
    /// stationary point, or a global ordering. Returning a typed refusal is the
    /// only sound result there; [`Self::fit_at`] remains available when the user
    /// explicitly fixes the smoothing parameter.
    pub fn fit_reml(&self) -> Result<ResidualCascadeFit, ResidualCascadeError> {
        if self.core.m > CERTIFIED_SPECTRUM_MAX {
            return Err(ResidualCascadeError::RemlScoreProofUnavailable {
                columns: self.core.m,
                certified_spectrum_max: CERTIFIED_SPECTRUM_MAX,
            });
        }
        let profile = self.core.reml_profile()?;
        let (log_lambda_lo, log_lambda_hi) = profile.log_lambda_domain()?;
        let resolution = f64::EPSILON.sqrt();
        let failed = |error: &dyn std::fmt::Display| {
            format!("residual cascade: REML stationary isolation failed: {error}")
        };
        let affine = profile.affine_view()?.ok_or(
            ResidualCascadeError::RemlScoreProofUnavailable {
                columns: self.core.m,
                certified_spectrum_max: CERTIFIED_SPECTRUM_MAX,
            },
        )?;
        // The budget refusal is re-typed here rather than formatted into the
        // generic computation failure: a nameless refusal at a derived depth
        // costs a day of triage, and the two numbers that explain it —
        // `rank` against `n - nullity` — are only available at this seam
        // (#2546).
        let search = match affine.maximize_value_ordered(
            log_lambda_lo,
            log_lambda_hi,
            resolution,
        ) {
            Ok(search) => search,
            Err(gam_math::score_opt::ScoreSearchError::SubdivisionBudget {
                subdivisions,
                budget,
                ..
            }) => {
                let nullity = self.core.nullity();
                return Err(ResidualCascadeError::RemlScoreSearchUndecomposable {
                    columns: self.core.m,
                    rank: self.core.m - nullity,
                    identifiable: self.core.y.len().saturating_sub(nullity),
                    subdivisions,
                    budget,
                    log_lambda_lo,
                    log_lambda_hi,
                });
            }
            Err(error) => return Err(ResidualCascadeError::Computation(failed(&error))),
        };
        if search.value_certificate.maximum_excess
            > search.value_certificate.comparison_resolution
        {
            return Err(ResidualCascadeError::RemlValueOrderingUnresolved {
                maximum_excess: search.value_certificate.maximum_excess,
                comparison_resolution: search.value_certificate.comparison_resolution,
            });
        }
        enum KktKind {
            LowerBoundary,
            UpperBoundary,
            Stationary,
        }
        let (bracket, kkt_kind) = match search.location {
            gam_math::score_opt::ScoreOptimumLocation::LowerBoundary => (
                gam_math::score_opt::ClosedInterval::point(search.lower_boundary.x),
                KktKind::LowerBoundary,
            ),
            gam_math::score_opt::ScoreOptimumLocation::UpperBoundary => (
                gam_math::score_opt::ClosedInterval::point(search.upper_boundary.x),
                KktKind::UpperBoundary,
            ),
            gam_math::score_opt::ScoreOptimumLocation::Stationary(index) => (
                search
                    .stationary_points
                    .get(index)
                    .ok_or_else(|| {
                        ResidualCascadeError::Computation(
                            "residual cascade: optimizer returned an invalid stationary index"
                                .to_string(),
                        )
                    })?
                    .bracket,
                KktKind::Stationary,
            ),
            gam_math::score_opt::ScoreOptimumLocation::ResolutionFlat(index) => {
                let flat = search.resolution_flat_regions.get(index).ok_or_else(|| {
                    ResidualCascadeError::Computation(
                        "residual cascade: optimizer returned an invalid resolution-flat index"
                            .to_string(),
                    )
                })?;
                return Err(ResidualCascadeError::RemlOptimumResolutionFlat {
                    lo: flat.bracket.lo,
                    hi: flat.bracket.hi,
                    max_score_gap: flat.max_score_gap,
                    score_resolution: flat.score_resolution,
                });
            }
        };
        let kkt = affine
            .enclose(bracket.lo, bracket.hi)
            .map_err(|error| ResidualCascadeError::Computation(failed(&error)))?;
        let kkt_holds = match kkt_kind {
            KktKind::LowerBoundary => kkt.derivative.hi <= 0.0,
            KktKind::UpperBoundary => kkt.derivative.lo >= 0.0,
            KktKind::Stationary => {
                kkt.derivative.contains_zero() && kkt.curvature.hi < 0.0
            }
        };
        if !kkt_holds {
            return Err(ResidualCascadeError::Computation(format!(
                "residual cascade: exact-real REML KKT certificate failed on \
                 {bracket:?}: {kkt:?}"
            )));
        }
        let selected_log_lambda = search.optimum.x;
        let selected = profile.evaluate(selected_log_lambda)?;
        Ok(self.fit_at_with_warm(
            selected_log_lambda,
            None,
            None,
            Some(selected.normalized_logdet),
        )?)
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

    /// Number of original training rows / experimental units.
    pub fn training_sample_size(&self) -> usize {
        self.training_sample_size.get()
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
        } else if let Some(factor) = &self.predict_sparse {
            // Exact, through the same factorization the fit's log-determinant
            // was read off — so the posterior variance carries no iterative
            // backward error at all past the dense Gram cache.
            solve_sparse_spd(factor, &Array1::from(dense_row.clone()))
                .map_err(|error| {
                    format!("residual cascade: sparse posterior-variance solve failed: {error}")
                })?
                .to_vec()
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
        let training_sample_size =
            u64::try_from(self.training_sample_size.get()).map_err(|_| {
                format!(
                    "residual cascade fit: training_sample_size {} exceeds the persistence format",
                    self.training_sample_size
                )
            })?;
        let training_sample_size =
            std::num::NonZeroU64::new(training_sample_size).ok_or_else(|| {
                "residual cascade fit: training_sample_size must be positive".to_string()
            })?;
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
            training_sample_size,
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
        let training_sample_size =
            usize::try_from(state.training_sample_size.get()).map_err(|_| {
                format!(
                    "residual cascade state: training_sample_size {} exceeds this platform's usize",
                    state.training_sample_size
                )
            })?;
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
            training_sample_size: std::num::NonZeroUsize::new(training_sample_size)
                .expect("nonzero wire count remains nonzero after conversion"),
            predict_chol: None,
            // The restored core carries the dense factor itself, so
            // `solve_coeff` replays through it; there is no CSR design left to
            // assemble a sparse system from.
            predict_sparse: None,
            coeff: state.coeff.clone(),
            log_lambda: state.log_lambda,
            sigma2: state.sigma2,
            restricted_loglik: state.restricted_loglik,
            rss_pen: state.rss_pen,
            certificate: CascadeCertificate {
                solve_rel_residual: 0.0,
                solve_iters: 0,
                logdet_method: LogdetMethod::DenseExact,
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
        // iterative solve up front with a typed signal before paying an
        // unbounded CG or grinding to CG_MAX_ITERS. (The guard is checked at
        // the root level only — refinement adds finer nets to the SAME scaled
        // cloud, so the aspect ratio is invariant under added levels.) The
        // typed computation refusal propagates through the selected cascade
        // route; callers must not silently replace this estimator with another
        // one.
        if levels == INITIAL_LEVELS && !design.quasi_uniformity_certified() {
            return Err(format!(
                "residual cascade: metric-scaled aspect ratio {:.3e} exceeds the \
                 quasi-uniformity ceiling {QUASI_UNIFORMITY_MAX_ASPECT:.0e}; the BPX \
                 iteration bound is not trustworthy on this (near-degenerate) metric",
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

    /// Column count of a `dense_fixture(side)` cascade at each level count, so
    /// the two width regimes the certified route now distinguishes are read off
    /// the design rather than guessed from the net arithmetic.
    #[test]
    fn zz_measure_cascade_width_by_level_count_2546() {
        for levels in 4..=8 {
            let (x1, x2, y) = dense_fixture(6);
            let weights = vec![1.0; y.len()];
            let axes: [&[f64]; 2] = [&x1, &x2];
            let design = ResidualCascadeDesign::build(&axes, &y, &weights, &[1.0, 1.0], 2.0, levels)
                .expect("cascade design");
            let m = design.core.m;
            println!(
                "#2546 levels={levels} m={m} gram_cached={} certified={}",
                design.core.dense_gram.is_some(),
                m <= CERTIFIED_SPECTRUM_MAX
            );
        }
        // The net is GEOMETRIC, not data-subsampled: `dense_fixture(6)` above is
        // 36 rows and still refines to 1725 columns at `levels = 6`. So the
        // identifiability the certified search needs -- every Schur mode carried
        // by the data -- is a race between a net set by `levels` and a sample set
        // by `side`, and neither the level table above nor the net arithmetic
        // says where it is won. Two guesses at it were wrong by 13 and by 8
        // columns respectively, so it is measured here instead.
        //
        // The band is swept DENSELY (every side from 45 to 64) rather than sampled,
        // because a coarse sample of this curve is what produced both wrong guesses
        // and then a third wrong claim drawn from the sample itself -- that no side
        // below 64 can be identified, inferred from 45, 50 and 60 all being short
        // by a small margin. The MARGIN is not monotone in `side` either, so
        // neighbouring sides disagree and only every-side settles it. Sides above
        // the band stay coarse: past the net discontinuity `m` falls away from `n`
        // and the outcome is no longer close.
        let mut sides: Vec<usize> = (45..=64).collect();
        sides.extend([70, 80, 90]);
        let mut identified_in_band: Vec<(usize, usize, usize)> = Vec::new();
        for side in sides {
            let (x1, x2, y) = dense_fixture(side);
            let weights = vec![1.0; y.len()];
            let axes: [&[f64]; 2] = [&x1, &x2];
            let design = ResidualCascadeDesign::build(&axes, &y, &weights, &[1.0, 1.0], 2.0, 6)
                .expect("cascade design");
            let m = design.core.m;
            let n = y.len();
            let nullity = design.core.nullity();
            let identified = m - nullity <= n - nullity;
            let past_cache = m > DENSE_GRAM_MAX && design.core.dense_gram.is_none();
            let certified = m <= CERTIFIED_SPECTRUM_MAX;
            println!(
                "#2546-IDENT side={side} n={n} m={m} nullity={nullity} margin={} \
                 identified={identified} past_cache={past_cache} certified={certified}",
                m as i64 - n as i64
            );
            if side <= 64 && identified && past_cache && certified {
                identified_in_band.push((side, m, n));
            }
        }
        println!(
            "#2546-IDENT sides in 45..=64 that are past the cache, inside the budget \
             and identified: {identified_in_band:?}"
        );
    }

    /// The width regime this issue existed to open: PAST the dense Gram cache,
    /// INSIDE the certified spectrum budget. Automatic REML must certify here.
    ///
    /// Before #2546 this design was fit-capable but never certifiable, because
    /// the proof was gated on the Gram CACHE rather than on the spectrum it is
    /// actually made of, so `fit_residual_cascade` could not finish at all once
    /// refinement crossed 1536 columns. The assertion is the certificate itself:
    /// a returned fit from `fit_reml` carries the KKT and ordering proofs, and
    /// its log-determinant route is exact.
    ///
    /// The fixture has MORE ROWS THAN COLUMNS on purpose. A cascade whose
    /// box-filling net outruns its sample — 36 rows against 1725 columns, say —
    /// does not certify at any width, including widths under `DENSE_GRAM_MAX`
    /// where the route was always open: `maximize_score_1d` subdivides for the
    /// unidentified end of the spectrum and does not terminate. That is a
    /// separate defect from this issue's, it predates the change here, and
    /// putting it in this gate's fixture would measure it instead of the
    /// capability.
    #[test]
    fn auto_reml_certifies_past_the_dense_gram_cache() {
        // The grid side is SEARCHED for rather than pinned, because the two
        // premises pull against each other and neither is a property of the code
        // under test. The width has to land strictly between the Gram cache and
        // the spectrum budget, and the sample has to be at least as large as the
        // width: a design with FEWER rows than columns is a separate,
        // pre-existing problem for the certified search (see
        // `the_spectral_residual_carries_no_null_modes`) and would be measured
        // here instead of the capability. Six levels of box-filling net set a
        // floor on the width, and a finer data grid adds centres of its own, so
        // the admissible sides are a band.
        //
        // The band is NOT where counting upward from 45 suggests, because `m` is
        // not monotone in `side`. Measured by
        // `zz_measure_cascade_width_by_level_count_2546` at `levels = 6`:
        //
        //   side=45  n=2025  m=2038   13 short
        //   side=47  n=2209  m=2159   IDENTIFIED, 50 to spare
        //   side=50  n=2500  m=2508    8 short
        //   side=60  n=3600  m=3628   28 short
        //   side=70  n=4900  m=1922   IDENTIFIED
        //   side=80  n=6400  m=2667   identified
        //   side=90  n=8100  m=3637   identified, past the spectrum budget
        //
        // `m` climbs to 3628 at side=60 and then FALLS to 1922 at side=70: once
        // the sample is finer than the level-6 net's own spacing the net stops
        // adding centres for it, so width and sample decouple and the data
        // overtake the spectrum. That discontinuity is why "64x64 gives 4117
        // columns, so the band ends below 64" was a trend argument and not a
        // measurement.
        //
        // Below the discontinuity, though, `m` does not track `n` at a fixed
        // offset, and the MARGIN `m - n` is not monotone in `side` either. Swept
        // exhaustively over 45..=64 by `zz_measure_cascade_width_by_level_count_2546`:
        //
        //   45:+13  46: +9  47:-50  48:-63  49:+27  50: +8  51:+22  52:+21
        //   53:+24  54:+19  55:+27  56:+14  57:+20  58:+18  59:+20  60:+28
        //   61:+20  62:+22  63:+24  64:+21
        //
        // Sides 47 and 48 are a two-point DIP of -50 and -63 in a band that is
        // otherwise +8 to +28, and both satisfy all three conditions. So a fine
        // search below 64 does succeed, and "45, 50 and 60 are all short, so no
        // side below 64 is identified" was a third extrapolation over the same
        // curve -- it samples straight across the only two sides that work.
        //
        // The reason to prefer the candidate list below is therefore COST, not
        // reachability: side=70 is identified by 2978 columns and hits on the first
        // design build, where stepping from 46 pays for two builds to reach 47 and
        // a reader cannot tell a deliberate choice from a lucky one.
        //
        // So the candidates are the measured ones, in cost order, with the band's
        // edges kept after them: the search still self-heals if DENSE_GRAM_MAX or
        // the spectrum budget moves, but it does not pay for two dozen design
        // builds to rediscover a curve that has already been sampled.
        let mut fixture = None;
        for side in [70_usize, 80, 75, 85, 90, 100, 60, 50] {
            let (x1, x2, y) = dense_fixture(side);
            let weights = vec![1.0; y.len()];
            let m = {
                let axes: [&[f64]; 2] = [&x1, &x2];
                ResidualCascadeDesign::build(&axes, &y, &weights, &[1.0, 1.0], 2.0, 6)
                    .expect("cascade design")
                    .core
                    .m
            };
            if m > DENSE_GRAM_MAX && m <= CERTIFIED_SPECTRUM_MAX && m <= y.len() {
                println!("#2546 certified-past-cache fixture: side={side} m={m} rows={}", y.len());
                fixture = Some((x1, x2, y, weights, m));
                break;
            }
        }
        let (x1, x2, y, weights, m) = fixture.expect(
            "no candidate grid side puts the width between DENSE_GRAM_MAX and              CERTIFIED_SPECTRUM_MAX with at least as many rows as columns -- if the caps              moved, re-run zz_measure_cascade_width_by_level_count_2546 and take the              candidates from its sweep rather than extrapolating a trend, because m is              not monotone in side",
        );
        let axes: [&[f64]; 2] = [&x1, &x2];
        let design = ResidualCascadeDesign::build(&axes, &y, &weights, &[1.0, 1.0], 2.0, 6)
            .expect("cascade design");
        assert_eq!(design.core.m, m);
        assert!(
            design.core.dense_gram.is_none(),
            "premise: the fixture must be past the dense Gram cache, got {m} columns"
        );
        let fit = design
            .fit_reml()
            .expect("a design past the Gram cache but inside the spectrum budget must certify");
        assert_eq!(fit.certificate.logdet_method, LogdetMethod::DenseExact);
        assert!(
            fit.log_lambda().is_finite(),
            "certified selection must return a finite log lambda, got {}",
            fit.log_lambda()
        );
    }

    /// The spectral residual handed to the interval extension carries only
    /// POSITIVE modes, and no more of them than the data can identify.
    ///
    /// `B = Z'WZ` for an `n × rank` whitened design, so `rank(B) ≤ n − nullity`;
    /// on a box-filling cascade over a small sample almost every column is a
    /// void-filling centre the data cannot pin, and the arithmetic returns
    /// roundoff for those directions. Carrying them as zeros costs exactness
    /// nothing on the scalar path and real enclosure width on the interval path,
    /// so the invariant is pinned here rather than left to the enclosure's
    /// behaviour.
    #[test]
    fn the_spectral_residual_carries_no_null_modes() {
        let (x1, x2, y) = dense_fixture(6);
        let weights = vec![1.0; y.len()];
        let axes: [&[f64]; 2] = [&x1, &x2];
        let design = ResidualCascadeDesign::build(&axes, &y, &weights, &[1.0, 1.0], 2.0, 6)
            .expect("cascade design");
        let core = &design.core;
        let identifiable = core.y.len() - core.nullity();
        assert!(
            core.m - core.nullity() > identifiable,
            "premise: the fixture must be rank-deficient (Schur rank {} against {identifiable} \
             identifiable directions)",
            core.m - core.nullity()
        );
        let profile = core.reml_profile().expect("spectral profile");
        let CascadeResidualForm::Spectral(spectrum) = &profile.residual else {
            panic!("this width must carry the spectral residual form");
        };
        assert!(
            spectrum.eigenvalue.iter().all(|&theta| theta > 0.0),
            "the spectral residual kept a non-positive mode"
        );
        assert_eq!(
            spectrum.eigenvalue.len(),
            spectrum.projected_square.len(),
            "mode and response-energy lists must stay aligned"
        );
        assert_eq!(spectrum.penalty.len(), spectrum.eigenvalue.len());
        assert!(
            spectrum.eigenvalue.len() <= identifiable,
            "kept {} modes against {identifiable} directions the data can identify",
            spectrum.eigenvalue.len()
        );
    }

    #[test]
    fn auto_reml_refuses_past_the_certified_spectrum_budget() {
        // One level finer than `auto_reml_certifies_past_the_dense_gram_cache`,
        // which quadruples the finest box net and takes the design past the
        // eigendecomposition's memory budget on the same tiny row set.
        let (x1, x2, y) = dense_fixture(6);
        let weights = vec![1.0; y.len()];
        let axes: [&[f64]; 2] = [&x1, &x2];
        let design = ResidualCascadeDesign::build(&axes, &y, &weights, &[1.0, 1.0], 2.0, 7)
            .expect("iterative-route cascade design");
        assert!(
            design.core.m > CERTIFIED_SPECTRUM_MAX,
            "fixture must exercise the uncertifiable route, got {} columns",
            design.core.m
        );

        let error = match design.fit_reml() {
            Ok(_) => panic!("auto-REML must not claim an enclosure it cannot form"),
            Err(error) => error,
        };
        assert!(matches!(
            error,
            ResidualCascadeError::RemlScoreProofUnavailable {
                columns,
                certified_spectrum_max: CERTIFIED_SPECTRUM_MAX,
            } if columns == design.core.m
        ));

        let fixed = design
            .fit_at(0.0, None)
            .expect("the same iterative design remains fit-capable at fixed lambda");
        assert_eq!(fixed.log_lambda(), 0.0);
        // Refusing the PROOF is not the same as accepting a stochastic number.
        // The fixed-λ fit's log-determinant is still exact, from a sparse direct
        // Cholesky of the same normal equations.
        assert_eq!(fixed.certificate.logdet_method, LogdetMethod::SparseExact);
    }

    /// Process high-water resident set size in bytes, or `None` where the
    /// kernel does not publish one.
    fn read_hwm_bytes() -> Option<f64> {
        let status = std::fs::read_to_string("/proc/self/status").ok()?;
        for line in status.lines() {
            if let Some(rest) = line.strip_prefix("VmHWM:") {
                let kb: f64 = rest.trim().trim_end_matches(" kB").trim().parse().ok()?;
                return Some(kb * 1024.0);
            }
        }
        None
    }

    /// Level counts whose `dense_fixture(45)` designs bracket the certified cap:
    /// `m = 891` and `m = 2038`, the step the block count is differenced over.
    const PEAK_MEMORY_LEVELS: [usize; 2] = [5, 6];

    /// Marker the per-width child prints its reading behind, and the gate finds
    /// it by. One definition so the writer and the reader cannot drift.
    const CHILD_READING_MARKER: &str = "#2546-child ";

    /// Build the certified spectral profile at one width and report the process
    /// high-water mark it reached, on stdout.
    ///
    /// Printed rather than returned because the gate reads it from a CHILD
    /// process, which is what makes the reading attributable.
    fn report_certified_spectrum_peak(levels: usize) {
        if read_hwm_bytes().is_none() {
            println!("{CHILD_READING_MARKER}levels={levels} vmhwm_unavailable");
            return;
        }
        let (x1, x2, y) = dense_fixture(45);
        let weights = vec![1.0; y.len()];
        let axes: [&[f64]; 2] = [&x1, &x2];
        let design = ResidualCascadeDesign::build(&axes, &y, &weights, &[1.0, 1.0], 2.0, levels)
            .expect("cascade design");
        let m = design.core.m;
        let profile = design.core.reml_profile().expect("spectral profile");
        let modes = profile.modes.len();
        drop(profile);
        let hwm = read_hwm_bytes().expect("VmHWM was readable a moment ago");
        println!("{CHILD_READING_MARKER}levels={levels} m={m} modes={modes} vmhwm_bytes={hwm}");
    }

    /// Narrow arm of the peak-memory measurement. Run on its own it certifies
    /// that the profile builds at this width; run as a child of the gate below
    /// it is one of the two readings the gate differences.
    #[test]
    fn zz_child_certified_spectrum_peak_memory_narrow_2546() {
        report_certified_spectrum_peak(PEAK_MEMORY_LEVELS[0]);
    }

    /// Wide arm of the peak-memory measurement; see the narrow arm.
    #[test]
    fn zz_child_certified_spectrum_peak_memory_wide_2546() {
        report_certified_spectrum_peak(PEAK_MEMORY_LEVELS[1]);
    }

    /// Peak resident memory of the certified spectral profile against the width
    /// it was built at, so [`CERTIFIED_SPECTRUM_BLOCKS`] is a measured number and
    /// not an inventory of the allocations this file can see.
    ///
    /// The block count is what converts a memory budget into a column cap, and it
    /// is NOT knowable from this file: `eigh` is `faer`'s self-adjoint EVD behind
    /// `FaerEigh`, and its tridiagonalization allocates workspace this crate never
    /// names.
    ///
    /// Each width is measured in its OWN CHILD PROCESS, and that is the subject
    /// of this comment rather than an implementation note. `VmHWM` is a
    /// PROCESS-WIDE high-water mark, so a difference of two readings taken in one
    /// process is this route's marginal growth only if nothing else allocated
    /// between them. Under `cargo test` the crate's ~1750 tests are threads in a
    /// SINGLE process and that condition does not hold. Measured at `b8745892a`:
    ///
    /// ```text
    /// exclusive process, host load 308-424, RAYON_NUM_THREADS 1/2/4/8
    ///     blocks = 6.41 / 6.84 / 6.63 / 6.79      -> passes
    /// shared process (`cargo test`), host load 1403
    ///     blocks = 15.00                          -> fails
    /// shared process (`cargo test`), host load 68
    ///     passes
    /// ```
    ///
    /// Load is not the variable and parallelism is not the variable: the
    /// exclusive arms ran at loads comparable to the failing shared arm and read
    /// 6.4-6.8 every time, flat across an 8x parallelism range. Process
    /// exclusivity is the variable. A shared-process reading over-attributes
    /// whatever else allocated between the two samples to this route, and the
    /// 15.00 it produced would have condemned a correct constant — raising
    /// [`CERTIFIED_SPECTRUM_BLOCKS`] to 16 cuts [`CERTIFIED_SPECTRUM_MAX`] from
    /// 2896 to 2048 and narrows the exact width regime this budget exists to
    /// open. A child process that builds one width and exits is exclusive by
    /// construction, under either harness.
    ///
    /// The gate is that the constant is not an UNDER-estimate: a block count
    /// smaller than the realized one would let the cap admit a width that
    /// overruns the budget it was derived from.
    #[test]
    fn zz_measure_certified_spectrum_peak_memory_2546() {
        if read_hwm_bytes().is_none() {
            println!("#2546 VmHWM unavailable on this platform; peak memory not measured");
            return;
        }
        let exe = match std::env::current_exe() {
            Ok(exe) => exe,
            Err(error) => {
                println!("#2546 test binary path unavailable ({error}); peak memory not measured");
                return;
            }
        };
        // Two widths, and the DIFFERENCE of their high-water marks over the
        // difference of their `m²`: the per-process baseline (test binary,
        // fixture, allocator arenas) is identical in the two children and
        // cancels, where a single absolute reading would attribute all of it to
        // the narrow width.
        let mut readings: Vec<(usize, f64)> = Vec::new();
        for child in [
            "zz_child_certified_spectrum_peak_memory_narrow_2546",
            "zz_child_certified_spectrum_peak_memory_wide_2546",
        ] {
            let path = format!("residual_cascade::refinement_decision_tests::{child}");
            let output = std::process::Command::new(&exe)
                .args(["--exact", path.as_str(), "--nocapture", "--test-threads=1"])
                .output()
                .unwrap_or_else(|error| panic!("spawn per-width child {child}: {error}"));
            let stdout = String::from_utf8_lossy(&output.stdout);
            let mut reading: Option<(usize, f64)> = None;
            for line in stdout.lines() {
                // The marker is searched for ANYWHERE in the line, not stripped
                // from its start. Under `--nocapture` libtest prints a test's
                // stdout INLINE after its own `test <name> ... ` prefix, so the
                // child's line arrives as
                //
                //     test residual_cascade::…::…_narrow_2546 ... #2546-child m=891 …
                //
                // and a prefix match reports every child as silent while the
                // reading is sitting in the middle of the line.
                let Some(offset) = line.find(CHILD_READING_MARKER) else {
                    continue;
                };
                let rest = &line[offset + CHILD_READING_MARKER.len()..];
                let mut width: Option<usize> = None;
                let mut hwm: Option<f64> = None;
                for field in rest.split_whitespace() {
                    if let Some(value) = field.strip_prefix("m=") {
                        width = value.parse().ok();
                    } else if let Some(value) = field.strip_prefix("vmhwm_bytes=") {
                        hwm = value.parse().ok();
                    }
                }
                if let (Some(width), Some(hwm)) = (width, hwm) {
                    reading = Some((width, hwm));
                }
            }
            let (m, hwm) = reading.unwrap_or_else(|| {
                panic!(
                    "child {child} produced no reading (status {:?}).\nstdout:\n{stdout}\nstderr:\n{}",
                    output.status,
                    String::from_utf8_lossy(&output.stderr)
                )
            });
            println!(
                "#2546 m={m} vmhwm={:.1}MiB (own process)",
                hwm / (1024.0 * 1024.0)
            );
            readings.push((m, hwm));
        }
        let (narrow, narrow_hwm) = readings[0];
        let (wide, wide_hwm) = readings[1];
        assert!(
            wide > narrow,
            "the widths must increase for a high-water difference to mean anything"
        );
        let square_growth =
            (wide as f64 * wide as f64 - narrow as f64 * narrow as f64) * size_of::<f64>() as f64;
        let blocks = (wide_hwm - narrow_hwm) / square_growth;
        println!(
            "#2546 marginal blocks={blocks:.2} declared={CERTIFIED_SPECTRUM_BLOCKS} \
             cap={CERTIFIED_SPECTRUM_MAX} budget_MiB={}",
            CERTIFIED_SPECTRUM_BYTES / (1024 * 1024)
        );
        assert!(
            blocks <= CERTIFIED_SPECTRUM_BLOCKS as f64,
            "the certified route grows by {blocks:.2} m-squared blocks against a declared \
             {CERTIFIED_SPECTRUM_BLOCKS} (m {narrow} -> {wide}); the column cap derived from \
             CERTIFIED_SPECTRUM_BYTES is therefore an under-estimate of the memory it admits"
        );
    }

    /// Fill-in of the sparse direct factor, against the dense triangle it
    /// replaces, on the past-cache widths this route exists to serve.
    ///
    /// The sparse route is worth taking only if the AMD ordering's realized
    /// `nnz(L)` is far below `m(m+1)/2` — the number a dense Cholesky would
    /// store. That is a property of the design's sparsity, not an assumption, so
    /// it is measured and asserted rather than argued: a multilevel Wendland row
    /// touches `O(1)` bumps per level, so `A` is sparse and its factor should be
    /// a small fraction of the dense triangle at every width here. If fill-in
    /// ever made the sparse factor comparable to dense storage, this gate fails
    /// and the route's premise is gone.
    #[test]
    fn zz_measure_sparse_factor_fill_in_2546() {
        for levels in [6usize, 7] {
            let (x1, x2, y) = dense_fixture(6);
            let weights = vec![1.0; y.len()];
            let axes: [&[f64]; 2] = [&x1, &x2];
            let design = ResidualCascadeDesign::build(&axes, &y, &weights, &[1.0, 1.0], 2.0, levels)
                .expect("cascade design");
            let core = &design.core;
            let system = core.sparse_upper_system(1.0).expect("sparse normal equations");
            let nnz_a = system.compute_nnz();
            let nnz_l = sparse_spd_factor_nnz(&system).expect("symbolic analysis");
            let dense_upper = core.m * (core.m + 1) / 2;
            println!(
                "#2546 levels={levels} m={} nnz(A)={nnz_a} nnz(L)={nnz_l} dense_upper={dense_upper} \
                 fraction_of_dense={:.5} budget={SPARSE_FACTOR_MAX_NNZ}",
                core.m,
                nnz_l as f64 / dense_upper as f64
            );
            assert!(
                nnz_l * 4 < dense_upper,
                "sparse factor is not sparse at m={}: nnz(L)={nnz_l} against a dense triangle of \
                 {dense_upper}; the sparse route's premise does not hold on this design",
                core.m
            );
            assert!(
                nnz_l <= SPARSE_FACTOR_MAX_NNZ,
                "fill-in {nnz_l} exceeds the factor budget {SPARSE_FACTOR_MAX_NNZ} at m={}",
                core.m
            );
        }
    }

    /// Fill-in and cost of the sparse direct factor at the width #2546 is about,
    /// on the geometry that produces it: ~6000 uniformly scattered 2-D rows, where
    /// `smoothness_ceiling_forces_refinement_and_certifies_residual_bias` refines
    /// to 2169 columns.
    ///
    /// The earlier fill measurement used a 36-row fixture, where almost every
    /// column is a void-filling centre carrying only a diagonal — so `nnz(A)` was
    /// unrepresentatively small and the absolute counts meant little. This one is
    /// data-rich, so `A` has the coupling a real cascade has.
    ///
    /// Timings are PRINTED, never asserted: a wall clock is not a contract (SPEC
    /// forbids wall-clock budgets outside tests, and even here it is evidence, not
    /// a gate). The assertion is on `nnz(L)` against the dense triangle, which is
    /// a property of the design and the ordering alone.
    #[test]
    fn zz_measure_sparse_factor_fill_in_at_the_2546_width() {
        for levels in [5usize, 6, 7] {
            let (x1, x2, y) = scattered_fixture(6000, 0x2546_0001);
            let weights = vec![1.0; y.len()];
            let axes: [&[f64]; 2] = [&x1, &x2];
            let design = ResidualCascadeDesign::build(&axes, &y, &weights, &[1.0, 1.0], 2.0, levels)
                .expect("cascade design");
            let core = &design.core;
            let m = core.m;
            let dense_upper = m * (m + 1) / 2;

            let assembled = std::time::Instant::now();
            let system = core.sparse_upper_system(1.0).expect("sparse normal equations");
            let assemble_ms = assembled.elapsed().as_secs_f64() * 1e3;
            let nnz_a = system.compute_nnz();
            let nnz_l = sparse_spd_factor_nnz(&system).expect("symbolic analysis");

            let factored = std::time::Instant::now();
            let sparse_logdet = core
                .sparse_exact_factor(1.0)
                .expect("sparse factorization")
                .map(|factor| logdet_from_factor(&factor).expect("sparse logdet"));
            let sparse_ms = factored.elapsed().as_secs_f64() * 1e3;

            // `logdet_dense` only exists under the Gram cache, so past it the
            // comparison would be an extrapolation of `m³/3`. It does not have to
            // be: `assemble_predict_factor` performs the SAME dense Cholesky of
            // the same `A` with no cap on width (it scatters the CSR rows into a
            // full `m × m` and factors in place), so the dense cost is MEASURED at
            // every width here rather than projected from the narrow one.
            let dense = if core.dense_gram.is_some() {
                let started = std::time::Instant::now();
                let value = core.logdet_dense(1.0).expect("dense logdet");
                Some((value, started.elapsed().as_secs_f64() * 1e3))
            } else {
                None
            };
            let uncapped_dense_ms = {
                let started = std::time::Instant::now();
                core.assemble_predict_factor(1.0)
                    .expect("uncapped dense factorization");
                started.elapsed().as_secs_f64() * 1e3
            };

            println!(
                "#2546-FILL levels={levels} m={m} rows={} nnz(A)={nnz_a} nnz(L)={nnz_l} \
                 dense_upper={dense_upper} nnz_L_over_dense={:.5} fill_over_A={:.3} \
                 factor_MiB={:.1} dense_MiB={:.1} assemble_ms={assemble_ms:.1} \
                 sparse_factor_ms={sparse_ms:.1} uncapped_dense_ms={uncapped_dense_ms:.1} \
                 sparse_speedup={:.2}x dense_chol={dense:?} sparse_logdet={sparse_logdet:?}",
                y.len(),
                nnz_l as f64 / dense_upper as f64,
                nnz_l as f64 / nnz_a.max(1) as f64,
                (nnz_l * (size_of::<f64>() + size_of::<usize>())) as f64 / (1024.0 * 1024.0),
                (dense_upper * size_of::<f64>()) as f64 / (1024.0 * 1024.0),
                uncapped_dense_ms / sparse_ms.max(f64::MIN_POSITIVE)
            );

            assert!(
                nnz_l * 4 < dense_upper,
                "the sparse route's premise fails at m={m}: nnz(L)={nnz_l} against a dense \
                 triangle of {dense_upper}"
            );
            if let Some((dense_value, _)) = dense {
                let sparse_value = sparse_logdet.expect("fill-in is inside the budget here");
                let resolution = f64::EPSILON * m as f64 * dense_value.abs().max(1.0);
                assert!(
                    (sparse_value - dense_value).abs() <= resolution,
                    "sparse and dense log-determinants disagree at m={m}: {sparse_value} versus \
                     {dense_value} (resolution {resolution})"
                );
            }
        }
    }

    /// The sparse direct log-determinant and the dense one are the same number.
    ///
    /// Both are exact factorizations of the same `X'WX + λD`, so they may differ
    /// only by floating-point summation order. The bound is the dense Cholesky's
    /// own forward error on a log-determinant — `O(m)·eps` per diagonal term over
    /// `m` terms — not a tuned tolerance, and it is charged on a design narrow
    /// enough to HAVE a dense route to compare against.
    #[test]
    fn sparse_and_dense_logdets_agree() {
        let (x1, x2, y) = dense_fixture(6);
        let weights = vec![1.0; y.len()];
        let axes: [&[f64]; 2] = [&x1, &x2];
        let design = ResidualCascadeDesign::build(&axes, &y, &weights, &[1.0, 1.0], 2.0, 4)
            .expect("cascade design");
        let core = &design.core;
        assert!(
            core.dense_gram.is_some(),
            "premise: the comparator needs the dense route, got m = {}",
            core.m
        );
        for log_lambda in [-4.0_f64, 0.0, 4.0] {
            let lambda = log_lambda.exp();
            let dense = core.logdet_dense(lambda).expect("dense logdet");
            let factor = core
                .sparse_exact_factor(lambda)
                .expect("sparse factorization")
                .expect("fill-in is inside the budget on this fixture");
            let sparse = logdet_from_factor(&factor).expect("sparse logdet");
            let resolution = f64::EPSILON * core.m as f64 * dense.abs().max(1.0);
            assert!(
                (sparse - dense).abs() <= resolution,
                "sparse and dense log-determinants disagree at log lambda {log_lambda}: \
                 {sparse} versus {dense} (resolution {resolution})"
            );
        }
    }

    /// The residual spectral sum and a direct factorization compute the same
    /// function of λ.
    ///
    /// The spectral expression reads the profiled residual and its three
    /// quadratic forms off the Schur decomposition; the comparator re-derives them from a
    /// factorization of `A = X'WX + λD`. If they ever disagree the criterion is
    /// representation-dependent, which is the defect the spectral form exists to remove —
    /// so the agreement is asserted directly rather than inferred from the
    /// scores that consume it.
    ///
    /// The bound is the textbook forward-error of the comparator, not a tuned
    /// number: the comparator's Cholesky solve carries `O(m)·eps·cond(A)`, and
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

        // `cond(A) = (θ_max + λ)/(θ_min + λ)` is taken over EVERY Schur mode,
        // including the certified-null ones the spectral residual now drops: a
        // null mode still sits at exactly λ in `A`'s spectrum, and it is what
        // sets the comparator's conditioning. Reading `spectrum.eigenvalue`
        // instead would silently tighten the comparator's own error budget.
        let smallest = profile
            .modes
            .iter()
            .map(|mode| mode.eigenvalue)
            .fold(f64::INFINITY, f64::min);
        let largest = profile
            .modes
            .iter()
            .map(|mode| mode.eigenvalue)
            .fold(0.0_f64, f64::max);

        for log_lambda in [-6.0_f64, -2.0, 0.0, 2.0, 6.0] {
            let lambda = log_lambda.exp();
            let (rss, penalty_energy, inverse_penalty_energy, third_energy) =
                spectrum.moments(lambda);

            let (coeff, _, _) = core
                .solve_coeff(lambda, &core.rhs, None)
                .expect("first solve");
            let dc: Vec<f64> = coeff
                .iter()
                .zip(core.pen_diag.iter())
                .map(|(&c, &d)| d * c)
                .collect();
            let (u, _, _) = core.solve_coeff(lambda, &dc, None).expect("second solve");
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
    /// two are held to agreement in value, slope and curvature here.
    ///
    /// The tolerance is the forward error of the arithmetic the two routes
    /// actually perform, read off THIS fixture's spectral moments at each
    /// evaluation point. It is not `rank·eps`, and the reason is measured:
    ///
    /// 1. THE TWO ROUTES DO NOT EVALUATE AT THE SAME LAMBDA. The cascade
    ///    exponentiates `rho` through `checked_exp_log_strength`, i.e. the
    ///    platform `exp` (sub-ulp); [`AffineRemlProfile::evaluate`] uses
    ///    `certified_exp_representative`, the midpoint of an outward-rounded
    ///    enclosure that is hundreds of ulps wide. Neither route may adopt the
    ///    other's: the cascade's criterion has to describe the lambda the fit
    ///    is actually solved at, and the affine profile's exponential has to be
    ///    the one its own enclosure is stated in. Over this domain the measured
    ///    `|lambda_affine - lambda_cascade| / lambda` runs to `2.59e-14`. A
    ///    RELATIVE shift `delta` in lambda IS an absolute shift `delta` in
    ///    `rho`, so every mode kernel moves by its own `d/drho` times `delta`,
    ///    and each accumulator below is charged exactly that. At
    ///    `rho = -2.6396` this term alone is `|curvature|·delta =
    ///    2.302 · 1.67e-14 = 3.8e-14`, which is the whole of the measured
    ///    `3.9e-14` slope disagreement that `rank·eps = 3.77e-15` was being
    ///    asked to cover.
    /// 2. The mode sums are sums of `rank` terms of SIZE, not of size one. The
    ///    determinant slope is `sum_i t_i` with `t_i = theta_i/(theta_i +
    ///    lambda)`, so its Wilkinson error is `rank·eps·sum_i t_i`; at that same
    ///    `rho`, `sum_i t_i = 10.71`, i.e. `4.0e-14` and not `3.8e-15`. The
    ///    profiled residual `R = anchor - S1` is the one subtraction of
    ///    near-equal positives, so its error carries the cancellation factor
    ///    `anchor/R` (up to `4.13` at the small-lambda end) — the same factor
    ///    [`spectral_and_solved_residual_forms_agree`] charges — and the score
    ///    multiplies it by `dof`.
    ///
    /// Every factor below is computed from the fixture at the evaluation point;
    /// nothing is a fitted constant, and the assertion is on ABSOLUTE
    /// disagreement so a normalization cannot quietly absorb a growing gap.
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
        let CascadeResidualForm::Spectral(spectrum) = &profile.residual else {
            panic!("the dense route must carry the spectral residual form");
        };
        let (lo, hi) = profile.log_lambda_domain().expect("domain");
        let rank = (design.core.m - design.core.nullity()) as f64;
        let dof = (design.core.y.len() - design.core.nullity()) as f64;
        let eps = f64::EPSILON;

        for step in 0..=8 {
            let log_lambda = lo + (hi - lo) * step as f64 / 8.0;
            let cascade = profile.evaluate(log_lambda).expect("cascade jet").jet;
            let spectral = affine.evaluate(log_lambda).expect("affine jet");

            // The two exponentials the two routes run, and the `rho` shift
            // between them.
            let lambda =
                gam_problem::checked_exp_log_strength(log_lambda).expect("cascade lambda");
            let affine_lambda = gam_math::score_opt::certified_exp_representative(log_lambda)
                .expect("affine lambda");
            let shift = (affine_lambda - lambda).abs() / lambda;

            let (rss, s2, s3, s4) = spectrum.moments(lambda);
            let anchor = spectrum.anchor_energy[0];
            // The magnitudes the three determinant accumulators run over, and
            // the `d/drho` of each summand: `d/drho log(1 + theta/lambda) = -t`
            // and `d/drho t = -t(1-t)`.
            let mut logdet_magnitude = 0.0_f64;
            let mut slope_magnitude = 0.0_f64;
            let mut curvature_magnitude = 0.0_f64;
            for &theta in &spectrum.eigenvalue {
                let t = theta / (theta + lambda);
                logdet_magnitude += (1.0 + theta / lambda).ln().abs();
                slope_magnitude += t;
                curvature_magnitude += t * (1.0 - t);
            }

            // Residual derivatives in `rho`, and the SUM OF MAGNITUDES of each
            // cancelling form — which is what a forward-error argument is
            // entitled to charge:
            //   first  = R'   = lambda S2
            //   second = R''  = R' - 2 lambda^2 S3
            //   third  = R''' = R' - 6 lambda^2 S3 + 6 lambda^3 S4.
            let lambda_squared = lambda * lambda;
            let lambda_cubed = lambda_squared * lambda;
            let first = lambda * s2;
            let second = first - 2.0 * lambda_squared * s3;
            let second_magnitude = first + 2.0 * lambda_squared * s3;
            let third_magnitude = first + 6.0 * lambda_squared * s3 + 6.0 * lambda_cubed * s4;

            // Each route accumulates `rank` terms sequentially and rounds a
            // handful of elementary operations per term, so each carries
            // `(rank + 4)·eps` relative on its own sum; the comparison is
            // charged both.
            let sum_eps = 2.0 * (rank + 4.0) * eps;
            let residual_error = sum_eps * anchor + first * shift;
            let first_error = sum_eps * first + second_magnitude * shift;
            let second_error = sum_eps * second_magnitude + third_magnitude * shift;
            let log_first = first / rss;
            let log_first_error = first_error / rss + log_first.abs() * residual_error / rss;
            let log_second_error = second_error / rss
                + (second / rss).abs() * residual_error / rss
                + 2.0 * log_first.abs() * log_first_error;

            let determinant_value_error = sum_eps * logdet_magnitude + slope_magnitude * shift;
            let determinant_slope_error = sum_eps * slope_magnitude + curvature_magnitude * shift;
            // The cascade forms the curvature summand as `t·(1-t)`, and `1-t`
            // cancels as `t -> 1`: one eps of `t` per mode. The affine route
            // carries the same complement as `lambda·s/h` and never subtracts.
            let determinant_curvature_error = sum_eps * curvature_magnitude
                + eps * slope_magnitude
                + curvature_magnitude * shift;

            let bounds = [
                0.5 * (determinant_value_error
                    + dof * (residual_error / rss + 2.0 * eps * (rss / dof).ln().abs())),
                0.5 * (determinant_slope_error + dof * log_first_error),
                0.5 * (determinant_curvature_error + dof * log_second_error),
            ];
            for ((name, a, b), bound) in [
                ("value", cascade.value, spectral.value),
                ("derivative", cascade.derivative, spectral.derivative),
                ("curvature", cascade.curvature, spectral.curvature),
            ]
            .into_iter()
            .zip(bounds)
            {
                let gap = (a - b).abs();
                assert!(
                    gap <= bound,
                    "{name} disagrees at log lambda {log_lambda}: cascade {a}, affine {b} \
                     (absolute {gap:e} exceeds the two routes' own forward error {bound:e}, \
                      whose terms are the lambda shift {shift:e}, the mode sums \
                      {slope_magnitude:e} / {curvature_magnitude:e} / {logdet_magnitude:e}, \
                      and the residual cancellation anchor/R {:e})",
                    anchor / rss
                );
            }
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
                .dense_cascade_spectrum(&null_chol, &core.spectral_gram().expect("fixture is inside the certified spectrum budget"))
                .expect("dense spectrum");
            let domain = certified_log_lambda_domain_from_modes(&modes).expect("domain");
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
                 coarse={} tail_estimate={:.3e} target={:.3e} accepted={} invariant={} tail={:.2e} \
                 mass_defect={:.2e} dropped={:.2e}",
                y.len(),
                core.m,
                certificate.budget,
                certificate.steps,
                certificate.coarse_steps,
                certificate.tail_estimate,
                certificate.target,
                certificate.accepted_for_point_evaluation,
                certificate.invariant,
                certificate.relative_tail,
                certificate.mass_defect,
                certificate.dropped_mass_fraction,
            );
            match worst {
                Some(worst) => println!(
                    "#2503   versus exact dense: dR/anchor={:.3e} S2={:.3e} S3={:.3e} S4={:.3e}",
                    worst[0], worst[1], worst[2], worst[3]
                ),
                None => println!("#2503   no rule admitted; point evaluation is refused"),
            }
        }
    }

    /// The admitted point quadrature matches the dense eigenbasis residual to the
    /// resolution its numerical evidence claims on this fixture.
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
            .dense_cascade_spectrum(&null_chol, &core.spectral_gram().expect("fixture is inside the certified spectrum budget"))
            .expect("dense spectrum");
        let domain = certified_log_lambda_domain_from_modes(&modes).expect("domain");
        let (spectrum, certificate) = core
            .iterative_residual_spectrum(&null_chol, domain)
            .expect("quadrature");
        let rank = core.m - core.nullity();
        assert!(
            certificate.accepted_for_point_evaluation,
            "the point quadrature must be admitted on this fixture: {certificate:?}"
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
        let spectrum = spectrum.expect("an admitted point rule is returned");

        // The comparator is the exact projection, and the bound is the resolution
        // the evidence claims — not a widened number. `R` is charged
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
        let domain = certified_log_lambda_domain_from_modes(&modes).expect("domain");
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
            let (coeff, _, _) = core
                .solve_coeff(lambda, &core.rhs, None)
                .expect("first certified solve");
            let dc: Vec<f64> = coeff
                .iter()
                .zip(core.pen_diag.iter())
                .map(|(&c, &d)| d * c)
                .collect();
            let (u, _, _) = core
                .solve_coeff(lambda, &dc, None)
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

    #[test]
    fn state_round_trip_requires_and_preserves_training_sample_size() {
        let (x1, x2, y) = dense_fixture(4);
        let weights = vec![1.0; y.len()];
        let axes: [&[f64]; 2] = [&x1, &x2];
        let design = ResidualCascadeDesign::build(&axes, &y, &weights, &[1.0, 1.0], 2.0, 2)
            .expect("cascade design");
        let fit = design.fit_at(0.0, None).expect("fixed-lambda fit");
        assert_eq!(fit.training_sample_size(), y.len());

        let state = fit.to_state().expect("persist cascade fit");
        assert_eq!(state.training_sample_size.get(), y.len() as u64);
        let restored = ResidualCascadeFit::from_state(&state).expect("restore cascade fit");
        assert_eq!(
            restored.training_sample_size(),
            y.len(),
            "prediction-only materialization must retain the original row count"
        );

        let mut encoded = serde_json::to_value(&state).expect("serialize cascade state");
        encoded
            .as_object_mut()
            .expect("cascade state serializes as an object")
            .remove("training_sample_size");
        assert!(
            serde_json::from_value::<ResidualCascadeState>(encoded).is_err(),
            "pre-training-size cascade state must not deserialize"
        );

        let mut zero = serde_json::to_value(&state).expect("serialize cascade state");
        zero.as_object_mut()
            .expect("cascade state serializes as an object")
            .insert("training_sample_size".to_string(), serde_json::json!(0));
        let error = serde_json::from_value::<ResidualCascadeState>(zero)
            .expect_err("zero training rows must not deserialize");
        assert!(
            error.to_string().contains("nonzero"),
            "zero-row rejection reported an unrelated error: {error}"
        );
    }

    /// Past the dense cap, a scattered design must EXHAUST its Krylov space inside
    /// the budget. On these designs that is the only exact residual admission,
    /// and without point admission the route refuses rather than reviving #2503's
    /// ill-conditioned per-lambda solve.
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
    /// work and then refuses anyway. The fixtures below are the three shapes the
    /// #2503 integration reds build, at the first level past the cap.
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
            let domain = certified_log_lambda_domain_from_modes(&modes).expect("domain");
            let rank = core.m - core.nullity();
            let ceiling = core.residual_krylov_ceiling();
            let budget = core.residual_quadrature_budget();
            assert_eq!(
                budget, ceiling,
                "n={n} levels={levels}: the budget must reach the Krylov ceiling (rank {rank}); \
                 stopping short of it pays the work and still refuses point evaluation"
            );
            let (spectrum, evidence) = core
                .iterative_residual_spectrum(&null_chol, domain)
                .expect("quadrature");
            println!(
                "#2503 n={n} levels={levels} m={} rank={rank} ceiling={ceiling} steps={} \
                 tail={:.2e} tail_estimate={:.3e} accepted={} invariant={} dropped={:.2e}",
                core.m,
                evidence.steps,
                evidence.relative_tail,
                evidence.tail_estimate,
                evidence.accepted_for_point_evaluation,
                evidence.invariant,
                evidence.dropped_mass_fraction,
            );
            assert!(
                evidence.invariant && evidence.accepted_for_point_evaluation && spectrum.is_some(),
                "n={n} levels={levels}: the past-cap residual quadrature must close its Krylov \
                 space and be admitted for point evaluation, else the route refuses (#2503): \
                 {evidence:?}"
            );
            assert!(
                evidence.relative_tail <= f64::EPSILON * evidence.steps as f64,
                "n={n} levels={levels}: admission here must come from EXHAUSTION — the Krylov \
                 residual against the operator scale must be at the arithmetic floor: \
                 {evidence:?}"
            );
            assert!(
                evidence.steps <= ceiling,
                "the run cannot exceed the reachable dimension: {evidence:?}"
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
            .dense_cascade_spectrum(&null_chol, &core.spectral_gram().expect("fixture is inside the certified spectrum budget"))
            .expect("dense spectrum");
        let (lo, hi) = certified_log_lambda_domain_from_modes(&modes).expect("domain");
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
            .dense_cascade_spectrum(&null_chol, &core.spectral_gram().expect("fixture is inside the certified spectrum budget"))
            .expect("dense spectrum");
        let (lo, hi) = certified_log_lambda_domain_from_modes(&modes).expect("domain");
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
    /// The admission evidence is a self-comparison — it never sees the truth — so the
    /// inference from "the ladder has contracted" to "the rule is right" is the
    /// thing that has to be measured. It rests on two facts and one model: every
    /// Gauss rule for a completely monotone kernel under-estimates its integral,
    /// so the ladder rises toward the truth; the gaps are therefore all of one
    /// sign and the remaining error is their tail; and the tail is extrapolated
    /// geometrically from the last two gaps, refusing outright when they do not
    /// contract. This charges that inference against the dense eigenbasis at every
    /// budget on the PRODUCTION ladder, over three designs — including the budgets
    /// the admission REFUSES, where it must be the refusal that is right.
    #[test]
    fn the_quadrature_tail_estimate_bounds_the_error_against_the_exact_spectrum_2503() {
        for (side, levels) in [(14usize, 4usize), (20, 5), (28, 5)] {
            let (x1, x2, y) = dense_fixture(side);
            let design = cascade_core((&x1, &x2, &y), levels);
            let core = &design.core;
            assert!(core.dense_gram.is_some());
            let (null_chol, _) = core.null_gram_factor().expect("null factor");
            let (modes, exact) = core
                .dense_cascade_spectrum(&null_chol, &core.spectral_gram().expect("fixture is inside the certified spectrum budget"))
                .expect("dense spectrum");
            let (lo, hi) = certified_log_lambda_domain_from_modes(&modes).expect("domain");
            let (beta, anchor) = core.whitened_residual_rhs(&null_chol);
            let mass = beta.iter().map(|value| value * value).sum::<f64>();
            let rank = core.m - core.nullity();
            let target = f64::EPSILON.sqrt();
            let ceiling = core.residual_krylov_ceiling();
            let budget = core.residual_quadrature_budget();
            let mut accepted_at_least_once = false;
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
                    accepted_at_least_once = true;
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
                accepted_at_least_once,
                "side={side} levels={levels}: the growth ladder must reach an admitted point \
                 rule inside the budget ({budget} of rank {rank}), or the diagnostic criterion \
                 remains unavailable"
            );
            assert!(
                refused_at_least_once,
                "side={side} levels={levels}: the ladder must also contain a REFUSED budget, or \
                 the criterion is not being exercised"
            );
        }
    }

}
