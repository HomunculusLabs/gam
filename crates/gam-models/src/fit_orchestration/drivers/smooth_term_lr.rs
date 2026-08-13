// The #1063 per-term smooth significance test: a genuine likelihood-ratio
// statistic from a constrained refit, its Lawley Bartlett correction, and the
// reference distribution it is scored against (#2672).
//
// Split out of `spatial_optimization.rs` under the #780 line-count gate. It is
// `include!`d into `drivers/mod.rs` alongside the driver it came from, so it
// keeps the same flat namespace and the same import surface — nothing here
// changed except which file it lives in.

/// Provenance tag for the smooth-term significance correction (#1063): which
/// statistic the reported p-value is built from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SmoothLrCorrection {
    /// A per-term LR statistic corrected by the full estimated-λ Lawley factor,
    /// including the ρ̂-sampling-variation contribution from the regularized
    /// inverse REML/LAML outer Hessian.
    LawleyLrEstimatedLambda,
    /// A per-term likelihood-ratio statistic `W = 2(ℓ_full − ℓ_null)` that has
    /// been Bartlett-corrected with the fixed-λ Lawley factor `c = E[W|λ]/d`
    /// (`W* = W/c`, referenced against `χ²_d`). This is used only when the
    /// estimated-λ handoff is unavailable.
    LawleyLrFixedLambda,
    /// No second-order correction was applied — either the family has no
    /// closed-form Lawley cumulant jets or the null refit did not converge — so
    /// the uncorrected `χ²_d` of the raw LR statistic stands.
    None,
}

impl SmoothLrCorrection {
    /// The serialized provenance label surfaced in the summary table.
    pub fn label(self) -> &'static str {
        match self {
            SmoothLrCorrection::LawleyLrEstimatedLambda => "lawley_lr_estimated_lambda",
            SmoothLrCorrection::LawleyLrFixedLambda => "lawley_lr_fixed_lambda",
            SmoothLrCorrection::None => "none",
        }
    }
}

/// Which lane supplied a [`SmoothLrReferenceDf`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SmoothLrReferenceSource {
    /// The statistic's own null spectrum `w`, in full, scored by Imhof
    /// inversion of its characteristic function. This is the exact lane: the
    /// reference IS the null law, not a distribution fitted to some of its
    /// moments.
    ///
    /// The spectrum is assembled from `[H⁻¹]_jj` and the term's own λ-weighted
    /// penalty block through the symmetric similarity
    /// `w_j = 1 − eig(B^{1/2} S_jj B^{1/2})²` — see
    /// [`lr_penalty_shares`] for why that is the same spectrum as
    /// `eig(2·F_jj − F_jj²)` and why it is the better-conditioned way to reach
    /// it.
    NullSpectrum,
    /// `[H⁻¹]_jj` or the penalty block was unavailable, but the
    /// coefficient-influence matrix was, so only the first two *moments* of the
    /// spectrum are recoverable (`tr A` and `tr A²` for `A = 2F_jj − F_jj²`,
    /// both traces of powers of one block). The reference is then the
    /// two-moment match `g·χ²_ν`.
    ///
    /// It is EXACT at both ends of the shrinkage range and wrong in between,
    /// which is worth stating precisely because the ends are where the intuition
    /// goes. An unpenalized term has `w ≡ 1` and the match is the textbook
    /// `χ²_q`; a term REML has shrunk to its null space has one weight of order
    /// one over a tail of dust — measured on a null-true `k = 12` fit, `w =
    /// (0.322, 5.9e-7, 7.1e-8, …)` — and a single distinct weight is a scaled
    /// chi-square exactly. The gap opens at moderate shrinkage, where several
    /// weights are comparable and unequal: on `f_j = 1/(1 + λγ_j)` for a
    /// second-difference penalty at `λ = 0.01`, `k = 20`, the size delivered at
    /// a nominal `α` is `1.02×` at `0.05`, `1.11×` at `0.01`, `1.31×` at `10⁻³`
    /// and `1.61×` at `10⁻⁴` — one-signed, anti-conservative, and worse the
    /// deeper the tail.
    ///
    /// It is a surrogate for the lane above, not a different claim about the
    /// statistic.
    SpectralMomentMatch,
    /// Neither the spectrum nor its moments were recoverable, so the reference
    /// falls back to the classical unit-weight shape
    /// `χ²_{max(edf, null_dim, 1)}` — every retained direction counted as if it
    /// were unpenalized. It is the only reference recoverable from a scalar
    /// EDF, and it is conservative for the same reason the whole pre-#2672
    /// assembly was: unit weights over-state the statistic's spread.
    UnitWeightFallback,
}

/// The null law of `W(λ̂)` when `λ̂` is CHOSEN by the outer criterion rather than
/// given — the reference the whole-term LR statistic actually needs (#2672).
///
/// # The defect this exists for
///
/// [`SmoothLrReferenceDf`]'s spectrum is the exact null law of `W` *at a fixed*
/// `λ`. `λ̂` is not fixed: REML picks it from a continuum, on the same data that
/// produced `W`. Measured on a Gaussian null with `σ` known — so no Lawley term
/// is in play and the reference is the only thing being tested — the two move
/// together (`corr(W, Σw) = 0.94–0.96`) but not by enough, and the conditional
/// reference over-rejects:
///
/// ```text
///                    α = .20    .10     .05     .01
///   conditional      .2060   .1320   .0840   .0180     n = 30,  k = 12
///                    .2160   .1100   .0580   .0140     n = 100, k = 12
///                    .1850   .1025   .0650   .0150     n = 200, k = 12
/// ```
///
/// It is not a mean problem, and it must not be fixed as one: on those same runs
/// `E[W]/E[Σw] ≈ 2.4–2.5`, and dividing `W` by that ratio takes the size at
/// `α = 0.05` from `.087` to `.0000`. The reference is not mean-matched to `W`
/// and is not supposed to be.
///
/// # The replay, and why it needs no refit
///
/// Diagonalize the term's fitted penalty `S_jj` against the Schur-complemented
/// information `Ĩ_jj` — the pair is symmetric-definite, so a single basis
/// diagonalizes both, with generalized eigenvalues `ν_k = p_k/(1 − p_k)` read
/// straight off the penalty shares [`lr_penalty_shares`] already computes. In
/// that basis the tested block is `q` independent standard normals `u_k`, and
/// BOTH the statistic and the criterion that selects `λ` are closed forms in
/// them and in the scale `t = λ/λ̂`:
///
/// ```text
/// W(t)  = Σ_k (2f_k − f_k²) u_k² ,          f_k = 1/(1 + t·ν_k)
/// V(t)  = ½ Σ_k u_k² ·t·ν_k/(1 + t·ν_k)
///       + ½ Σ_{k: ν_k > 0} log((1 + t·ν_k)/(t·ν_k))     (+ terms free of t)
/// ```
///
/// So the whole selection — draw data, choose `λ̂`, read `W` — is a function of
/// `q` numbers, and the null law of `W(λ̂)` can be generated exactly (within the
/// same quadratic expansion the conditional law already assumes) with no design,
/// no response and no refit. `t = 1` reproduces the conditional law, which is
/// what makes this a strict generalization rather than a different reference.
///
/// # What it buys, measured
///
/// Same runs, same replicates, `20 000` draws per fit:
///
/// ```text
///                    α = .20    .10     .05     .01
///   selection-aware  .1940   .1160   .0560   .0120     n = 30,  k = 12
///                    .2020   .0840   .0440   .0080     n = 100, k = 12
///                    .1775   .0925   .0425   .0075     n = 200, k = 12
/// ```
///
/// Closer to nominal at every level in every cell — twelve of twelve — and the
/// `α = 0.05` column goes from a mean of `.069` to `.047` against a per-cell
/// Monte-Carlo standard error of `.0097`.
///
/// # The Monte-Carlo error is removed where it would matter
///
/// The replay is a simulation, so its tail is an estimate. The conditional tail
/// is NOT — [`gam_math::probability::weighted_chi_square_sf`] evaluates it by
/// inversion. The two are strongly dependent (the same draws, differing only in
/// whether `t` is selected or held at one), so the replay reports the
/// DIFFERENCE and adds it to the exact conditional value:
///
/// ```text
/// p_selection = p_conditional + [ P̂(W_sel ≥ w) − P̂(W_cond ≥ w) ]
/// ```
///
/// a textbook control variate. The bracket is a difference of two indicators
/// that agree on most draws, so its variance is a fraction of either term's, and
/// the standard error of the pair is measured per query and published in the
/// report's own accuracy bound rather than assumed.
#[derive(Clone)]
pub struct SmoothLrSelectionReplay {
    /// `ν_k = eig(Ĩ_jj⁻¹ S_jj(λ̂))`, the term's generalized penalty spectrum at
    /// the fitted scale, ascending. `t = 1` is the fit.
    ///
    /// It is published on EVERY lane. The multi-scale lane used to leave it
    /// empty — its samples come from a grid whose basis moves with `t`, so it
    /// has no single diagonalizing spectrum to report *per grid point* — but the
    /// FITTED point always has one, it is the object the whole replay is built
    /// on, and a consumer asking what was replayed is asking about that. An
    /// empty vector there was an accident of which lane ran, not a statement
    /// about the term.
    pub generalized: Vec<f64>,
    /// `W(λ̂(u))` over the draws, IN DRAW ORDER.
    selection_sample: Vec<f64>,
    /// `W(1)` over the SAME draws, in the same order — the control variate.
    ///
    /// The order is the pairing, and the pairing is the whole point: sorting
    /// either sample would leave the two counts correct and destroy the paired
    /// difference whose variance is what makes this a control variate rather
    /// than two independent estimates.
    conditional_sample: Vec<f64>,
}

impl std::fmt::Debug for SmoothLrSelectionReplay {
    /// The two samples are thousands of draws each and are never what a reader
    /// of a failure message wants; the spectrum that generated them is.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SmoothLrSelectionReplay")
            .field("generalized", &self.generalized)
            .field("draws", &self.selection_sample.len())
            .finish()
    }
}

impl PartialEq for SmoothLrSelectionReplay {
    fn eq(&self, other: &Self) -> bool {
        self.generalized == other.generalized
            && self.selection_sample == other.selection_sample
            && self.conditional_sample == other.conditional_sample
    }
}

/// Draws used to generate the selection replay.
///
/// The replay's only error is the Monte-Carlo error of the control-variate
/// DIFFERENCE, not of a tail — the tail itself is inverted exactly. That
/// difference is a mean of indicators that agree on all but the draws whose
/// selected `t` moves them across the threshold, so its standard error is a
/// fraction of `√(p(1−p)/N)`. At this budget that raw bound is `3.4e-3` at
/// `p = 0.05`, and the measured control-variate standard error on the fixtures
/// is between one and two orders below it. It is measured and published per
/// query rather than assumed, so a caller never has to take this number on
/// trust — and the cost is one `N × G` reduction per term, `~40 ms`.
const SMOOTH_LR_SELECTION_DRAWS: usize = 4096;

/// Grid resolution of the replay's own `argmin` over `ln t`.
///
/// The criterion is smooth in `ln t` and the statistic is smooth in the selected
/// `t`, so the grid only has to resolve the criterion's minimizer to a fraction
/// of the scale on which `W` changes. `0.05` in `ln t` is a 5% change in `λ`,
/// which moves `w_k = 1 − (tν_k/(1+tν_k))²` by at most `0.025` in the worst
/// direction and much less in every other. The span is the solver's own `ρ` box
/// translated to the fitted point, so the replay never selects a `λ` the fit
/// could not have.
const SMOOTH_LR_SELECTION_LOG_STEP: f64 = 0.05;

/// Total grid points the multi-scale replay may spend, whatever `m` is.
///
/// The one-dimensional replay grids `ln t` at a fixed step because its per-point
/// cost is `O(q)`. The multi-scale one costs `O(q³)` to build a point and
/// `O(q²)` per draw at it, so the budget is on the TOTAL and each axis gets
/// `budget^(1/m)` points — the cost is then independent of `m` and the
/// resolution degrades with it, which is the trade the
/// [`SMOOTH_LR_SELECTION_MAX_SCALES`] cut-off exists to bound.
const SMOOTH_LR_SELECTION_GRID_BUDGET: usize = 441;

/// Draws for the multi-scale replay.
///
/// Fewer than the one-dimensional path's, and the reason is arithmetic rather
/// than a different accuracy target: a one-dimensional grid point costs `O(q)`
/// per draw because the whole problem is diagonal there, and a multi-scale one
/// costs `O(q²)` because it is not. At `q ≈ 11` and a `441`-point grid this is
/// about `0.1 s` per term, and the standard error it leaves is measured and
/// published per query — a coarser replay that says how coarse it is beats a
/// finer one nobody can afford to run.
const SMOOTH_LR_MULTISCALE_DRAWS: usize = 2048;

/// Scales past which the budget above would leave fewer than five points per
/// axis — a spacing of about `15` in `ln λ` over the solver's box, which is not
/// a selection, it is a coin toss. A term with more scales than this falls back
/// to the common-scale slice, and says so in its provenance rather than
/// pretending to a replay it did not do.
const SMOOTH_LR_SELECTION_MAX_SCALES: usize = 4;

/// The term's penalty geometry in the basis the replay's criterion lives in:
/// whitened by the Schur-complemented information and factored into ROOTS.
///
/// # Why roots, and why this is not an implementation detail
///
/// The replayed criterion is
///
/// ```text
/// V(t) = ½ Σ_j c_j² e_j/(1 + e_j)  +  ½ [ log|I + T(t)| − log|T(t)|₊ ],
/// T(t) = Σ_i t_i λ̂_i · Wᵀ S_i W,   e = eig T(t),
/// ```
///
/// and the bracket is its whole Occam half — the only term that stops the
/// selection running to `t → 0`. The first summand of the bracket is benign: a
/// mode `e` carries `log(1 + e)`, so an error of `ε‖T‖` in a mode near zero
/// costs `ε‖T‖`. The SECOND is not, and this repo has already written down why,
/// in `penalty_logdet.rs`'s `SpectrumScale` (#2644):
///
/// > `S_λ = Σ_k λ_k S_k` is a SUM OF SQUARES, so forming it squares the
/// > conditioning of the objects it is built from … Every backward-stable
/// > factorization of the ASSEMBLED matrix therefore prices `log|S_λ|₊` to
/// > `O(ε·κ(S_λ))`, while the same quantity taken from the stacked scaled ROOTS
/// > costs `O(ε·√κ(S_λ))`. The outer smoothing search routinely drives
/// > `κ(S_λ)` past `1e14` (one λ at its ceiling beside a null-space shrinkage λ
/// > near zero is enough) …
///
/// The parenthesis is this fixture. A default `s(z)` is a DOUBLE-penalty smooth
/// (wiggliness plus a null-space ridge), and a null-true smooth is exactly the
/// fit that rails the first λ up and the second down. Measured on a whitened
/// `q = 9` pair at the separations the box allows, the assembled route against
/// the root route:
///
/// ```text
/// ρ̂ = (0, 0)      offset  20.564 vs  20.564     error   0.000
/// ρ̂ = (12, −12)   offset  29.813 vs  29.813     error   0.000
/// ρ̂ = (18, −24)   offset  19.189 vs  53.811     error −34.623
/// ρ̂ = (29, −29)   offset   8.103 vs  63.811     error −55.709
/// ```
///
/// and the error is not a perturbation of the selection, it REPLACES it: the
/// dropped modes are the ones carrying `−ln t_i`, i.e. the coercivity that makes
/// the criterion blow up as `t_i → 0`, so their loss makes the criterion
/// monotone in `ln t_i` and the replay picks a wall. That is the same mechanism
/// `from_components` documents under #1237, reached from the replay's side.
///
/// So the geometry is carried as `R_i` with `R_iᵀ R_i = Wᵀ S_i W` for the
/// term's λ-FREE components, plus their `ρ̂_i` and the structural rank of their
/// SUM. Everything the replay needs at a grid point — the eigenbasis, the
/// shares, the statistic's weights and the criterion's log-determinant — is then
/// one thin SVD of the stacked scaled roots.
///
/// # Why the rank has to be structural
///
/// `log|T|₊` is a sum over `range(T)`, and `range(T)` does not depend on `t`
/// (every `t_i > 0`). Deciding membership by `e_j > 0` — which is what both
/// replay lanes did — asks a floating-point comparison to separate a structural
/// zero from a mode `1e18` below the largest one, and it answers at random:
/// a noise-negative mode silently drops a `log(1 + 1/e) ≈ 24` contribution, and
/// a noise-positive one invents `log(1 + 1e16) ≈ 37`. The rank is taken once,
/// from the λ-free stacked roots, where the spectrum is well scaled.
struct SelectionGeometry {
    /// `R_i`, `rank_i × q`, with `R_iᵀR_i = Wᵀ S_i W` for the term's λ-free
    /// penalty component `i`. Rows are `√σ · uᵀ` over that COMPONENT's own
    /// modes above its OWN relative floor, so the truncation never sees the λ
    /// dynamic range.
    roots: Vec<Array2<f64>>,
    /// `ρ̂_i = ln λ̂_i`, the fitted scale of component `i`. `ln t_i = 0` is the
    /// fit.
    log_lambda: Vec<f64>,
    /// `q`, the tested block's identified dimension after whitening.
    dimension: usize,
    /// `rank(Σ_i Wᵀ S_i W)`, `t`-independent — the index set `log|T|₊` runs
    /// over.
    rank: usize,
    /// Rows of the stacked root matrix, at least `dimension` so the thin SVD's
    /// right factor is a full orthonormal basis of the block (the padding rows
    /// are zero and change nothing else).
    stacked_rows: usize,
}

/// One grid point of the replay: the criterion's data operator, the statistic's
/// weights, the criterion's log-determinant offset, and the basis all three are
/// diagonal in.
struct SelectionPoint {
    /// `e_j = eig T(t)`, descending.
    eigenvalues: Vec<f64>,
    /// `f_j = e_j/(1 + e_j)`, the criterion's data operator.
    shares: Vec<f64>,
    /// `w_j = 2f̄_j − f̄_j²` with `f̄ = 1 − f`, the statistic's null weights.
    weights: Vec<f64>,
    /// `log|I + T| − log|T|₊`, the criterion's `t`-dependent Occam term.
    offset: f64,
    /// Columns are the eigenvectors of `T(t)`, in the same order.
    basis: Array2<f64>,
}

impl SelectionGeometry {
    /// Whiten the term's λ-free penalty components by the Schur-complemented
    /// information and factor each into its own root.
    ///
    /// Returns `None` when the information has no identified direction, a
    /// decomposition refuses, or the components leave nothing penalized —
    /// in every case the caller has nothing to replay.
    fn whiten(
        information: &Array2<f64>,
        unit_penalties: &[Array2<f64>],
        log_lambda: &[f64],
    ) -> Option<Self> {
        if unit_penalties.is_empty() || unit_penalties.len() != log_lambda.len() {
            return None;
        }
        let (values, vectors) =
            gam_linalg::faer_ndarray::strict_symmetric_eigh(information, faer::Side::Lower).ok()?;
        let largest = values.iter().copied().fold(0.0_f64, f64::max);
        if !(largest > 0.0) {
            return None;
        }
        let floor = largest * 1e-12;
        let kept: Vec<usize> = (0..values.len())
            .filter(|&index| values[index] > floor)
            .collect();
        if kept.is_empty() {
            return None;
        }
        let dimension = kept.len();
        let mut whitener = Array2::<f64>::zeros((information.nrows(), dimension));
        for (column, &index) in kept.iter().enumerate() {
            let scale = values[index].sqrt();
            for row in 0..information.nrows() {
                whitener[[row, column]] = vectors[[row, index]] / scale;
            }
        }

        let mut roots = Vec::with_capacity(unit_penalties.len());
        for penalty in unit_penalties {
            if penalty.nrows() != information.nrows() || penalty.ncols() != information.ncols() {
                return None;
            }
            let whitened = whitener.t().dot(penalty).dot(&whitener);
            if whitened.iter().any(|value| !value.is_finite()) {
                return None;
            }
            roots.push(psd_root(&whitened)?);
        }
        if roots.iter().all(|root| root.nrows() == 0) {
            return None;
        }
        let stacked_rows = roots.iter().map(|root| root.nrows()).sum::<usize>();
        // `range(Σ_i S̃_i)` is what `log|T|₊` runs over, and it is `t`-free. Taken
        // from the UNIT stacked roots, whose singular values span only the
        // components' own conditioning — the λ ratio that makes the assembled
        // sum unreadable is not present here at all.
        let unit = stack_roots(&roots, &vec![0.0; roots.len()], stacked_rows.max(dimension));
        let (_, unit_singular, _) =
            gam_linalg::faer_ndarray::FaerSvd::svd(&unit, false, false).ok()?;
        let unit_largest = unit_singular.iter().copied().fold(0.0_f64, f64::max);
        let rank = unit_singular
            .iter()
            .filter(|&&value| value > unit_largest * (dimension as f64) * f64::EPSILON * 100.0)
            .count();
        if rank == 0 {
            return None;
        }
        Some(Self {
            roots,
            log_lambda: log_lambda.to_vec(),
            dimension,
            rank,
            stacked_rows: stacked_rows.max(dimension),
        })
    }

    /// Evaluate the geometry at one grid point `ln t`.
    ///
    /// One thin SVD of the stacked scaled roots supplies all four quantities:
    /// `eig T = σ²`, the eigenbasis is the right singular factor, and the
    /// log-determinants are `Σ log(1 + σ²)` and `2 Σ_{j < rank} log σ` — the
    /// second over the structural rank rather than over a sign test.
    fn at(&self, log_t: &[f64]) -> Option<SelectionPoint> {
        if log_t.len() != self.roots.len() {
            return None;
        }
        let scaled: Vec<f64> = self
            .log_lambda
            .iter()
            .zip(log_t.iter())
            .map(|(rho, shift)| rho + shift)
            .collect();
        let stacked = stack_roots(&self.roots, &scaled, self.stacked_rows);
        let (_, singular, right) =
            gam_linalg::faer_ndarray::FaerSvd::svd(&stacked, false, true).ok()?;
        // Thin SVD on a matrix with at least `dimension` rows: the right factor
        // is `dimension × dimension`, so every direction of the block — including
        // the ones the penalty never reaches — has a basis vector.
        let right = right?;
        if right.nrows() != self.dimension || singular.len() < self.dimension {
            return None;
        }
        let mut eigenvalues = Vec::with_capacity(self.dimension);
        let mut shares = Vec::with_capacity(self.dimension);
        let mut weights = Vec::with_capacity(self.dimension);
        let mut log_det_hessian = 0.0_f64;
        let mut log_det_penalty = 0.0_f64;
        for index in 0..self.dimension {
            let sigma = singular[index];
            if !sigma.is_finite() || sigma < 0.0 {
                return None;
            }
            let eigenvalue = sigma * sigma;
            log_det_hessian += eigenvalue.ln_1p();
            if index < self.rank {
                if !(sigma > 0.0) {
                    return None;
                }
                log_det_penalty += 2.0 * sigma.ln();
            }
            let fraction = if eigenvalue.is_finite() {
                eigenvalue / (1.0 + eigenvalue)
            } else {
                1.0
            };
            let shrinkage = 1.0 - fraction;
            eigenvalues.push(eigenvalue);
            shares.push(fraction);
            weights.push(2.0 * shrinkage - shrinkage * shrinkage);
        }
        let offset = log_det_hessian - log_det_penalty;
        if !offset.is_finite() {
            return None;
        }
        // Columns are eigenvectors: `right` is `Vᵀ` from the thin SVD, so its
        // ROWS are the right singular vectors.
        let mut basis = Array2::<f64>::zeros((self.dimension, self.dimension));
        for column in 0..self.dimension {
            for row in 0..self.dimension {
                basis[[row, column]] = right[[column, row]];
            }
        }
        Some(SelectionPoint {
            eigenvalues,
            shares,
            weights,
            offset,
            basis,
        })
    }
}

/// `[√(e^{s_0}) R_0; √(e^{s_1}) R_1; …]`, zero-padded to `rows`.
///
/// The scale is applied as `exp(s/2)` rather than as `sqrt(exp(s))` so a `λ̂` at
/// the box wall never round-trips through an intermediate that overflows.
fn stack_roots(roots: &[Array2<f64>], log_scale: &[f64], rows: usize) -> Array2<f64> {
    let columns = roots
        .iter()
        .map(|root| root.ncols())
        .max()
        .unwrap_or_default();
    let mut stacked = Array2::<f64>::zeros((rows, columns));
    let mut offset = 0usize;
    for (root, &scale) in roots.iter().zip(log_scale.iter()) {
        let factor = (0.5 * scale).exp();
        for row in 0..root.nrows() {
            for column in 0..root.ncols() {
                stacked[[offset + row, column]] = factor * root[[row, column]];
            }
        }
        offset += root.nrows();
    }
    stacked
}

/// A root `R` of a symmetric PSD `S`, `RᵀR = S`, taken from `S`'s OWN
/// eigensystem and truncated at `S`'s own relative noise floor.
///
/// `S` here is always a λ-free whitened penalty component, so its spectrum is
/// well scaled and this is a benign `O(ε)` operation — the dynamic range that
/// makes the weighted sum hard lives in the λ's, not here. This mirrors
/// `penalty_logdet::psd_component_root`, which is private to `gam-solve`;
/// the contract is the same and so is the threshold.
fn psd_root(matrix: &Array2<f64>) -> Option<Array2<f64>> {
    let dimension = matrix.nrows();
    if dimension == 0 {
        return Some(Array2::zeros((0, 0)));
    }
    let (values, vectors) =
        gam_linalg::faer_ndarray::strict_symmetric_eigh(matrix, faer::Side::Lower).ok()?;
    let largest = values.iter().copied().fold(0.0_f64, |a, b| a.max(b.abs()));
    let threshold = 100.0 * (dimension as f64) * f64::EPSILON * largest;
    let kept: Vec<usize> = (0..dimension)
        .filter(|&index| values[index] > threshold)
        .collect();
    let mut root = Array2::<f64>::zeros((kept.len(), dimension));
    for (row, &index) in kept.iter().enumerate() {
        let scale = values[index].sqrt();
        for column in 0..dimension {
            root[[row, column]] = scale * vectors[[column, index]];
        }
    }
    Some(root)
}

/// The published order of a generalized spectrum: ascending, as
/// [`SmoothLrSelectionReplay::generalized`] documents.
fn ascending(mut values: Vec<f64>) -> Vec<f64> {
    values.sort_by(|a, b| a.partial_cmp(b).expect("finite generalized spectrum"));
    values
}

impl SmoothLrSelectionReplay {
    /// Generate the replay for one term from its whitened penalty geometry and
    /// the window of `ln t` the fit's own `ρ` box leaves open around the fitted
    /// point — ONE window per scale, because the outer search moved each `ρ_i`
    /// independently inside that box.
    ///
    /// Returns `None` when the term has no penalized direction (nothing to
    /// select), the geometry could not be whitened, or every window is empty
    /// (the fit is railed against both walls), in which case the conditional law
    /// IS the selection law and the caller should use it unmodified.
    fn generate(
        information: &Array2<f64>,
        unit_penalties: &[Array2<f64>],
        log_lambda: &[f64],
        log_scale_windows: &[(f64, f64)],
    ) -> Option<Self> {
        let geometry = SelectionGeometry::whiten(information, unit_penalties, log_lambda)?;
        Self::from_geometry(
            &geometry,
            log_scale_windows,
            SMOOTH_LR_SELECTION_DRAWS,
            SMOOTH_LR_MULTISCALE_DRAWS,
        )
    }

    /// Dispatch: a term selecting `m` scales inside the budget gets the
    /// `m`-dimensional replay; anything else gets the common-scale slice.
    fn from_geometry(
        geometry: &SelectionGeometry,
        log_scale_windows: &[(f64, f64)],
        diagonal_draws: usize,
        multiscale_draws: usize,
    ) -> Option<Self> {
        if log_scale_windows.len() != geometry.roots.len() {
            return None;
        }
        let scales = geometry.roots.len();
        if (2..=SMOOTH_LR_SELECTION_MAX_SCALES).contains(&scales)
            && let Some(replay) =
                Self::generate_multiscale(geometry, log_scale_windows, multiscale_draws)
        {
            return Some(replay);
        }
        Self::generate_common_scale(geometry, log_scale_windows, diagonal_draws)
    }

    /// The COMMON-SCALE replay: every scale moved together, `t_i ≡ t`.
    ///
    /// This is the whole selection when the term has one penalty, and it is the
    /// honest fallback when it has more scales than
    /// [`SMOOTH_LR_SELECTION_MAX_SCALES`] — where an `m`-dimensional grid inside
    /// the budget would space its axes about `15` apart in `ln λ`, which is not a
    /// selection, it is a coin toss.
    ///
    /// Under a common scale `T(t) = t·T(1)`, so the eigenBASIS does not move and
    /// the whole grid is diagonal in one decomposition: the per-point cost is
    /// `O(q)` rather than `O(q³)`, which is what pays for the finer `ln t` step.
    /// The log-determinant is exact in closed form for the same reason —
    /// `log|T(t)|₊ = rank·ln t + log|T(1)|₊` — so the only quantity that has to
    /// be priced carefully is the `t`-free constant, and it is, through the
    /// stacked roots at the fitted point.
    fn generate_common_scale(
        geometry: &SelectionGeometry,
        log_scale_windows: &[(f64, f64)],
        draws: usize,
    ) -> Option<Self> {
        // Moving every scale together, the reachable set is the INTERSECTION of
        // the per-scale windows: a common shift has to keep every `ρ̂_i + ln t`
        // inside the solver's box at once.
        let (mut low, mut high) = (f64::NEG_INFINITY, f64::INFINITY);
        for &(window_low, window_high) in log_scale_windows {
            if !(window_low.is_finite() && window_high.is_finite()) {
                return None;
            }
            low = low.max(window_low);
            high = high.min(window_high);
        }
        if !(low.is_finite() && high.is_finite()) || high <= low {
            return None;
        }
        let zero = vec![0.0_f64; geometry.roots.len()];
        let fitted = geometry.at(&zero)?;
        // `ν_j = eig T(1)`, descending, with the structural rank leading. Only
        // the leading `rank` of them are in `range(T)`; the rest are the term's
        // unpenalized directions and carry no log-determinant term at any `t`.
        let generalized = fitted.eigenvalues.clone();
        if !generalized.iter().take(geometry.rank).any(|&nu| nu > 0.0) {
            return None;
        }
        let constant: f64 = generalized
            .iter()
            .take(geometry.rank)
            .map(|nu| nu.ln())
            .sum();

        let steps = (((high - low) / SMOOTH_LR_SELECTION_LOG_STEP).ceil() as usize).max(1);
        // The fitted point is an explicit extra node. It has to be there twice
        // over: the SELECTION must be able to choose the scale the fit chose —
        // for the observed data it IS that scale, by construction — and the
        // control variate's conditional arm is read there, not at whichever node
        // happens to be nearest.
        let mut log_grid: Vec<f64> = (0..=steps)
            .map(|step| low + (high - low) * (step as f64) / (steps as f64))
            .collect();
        log_grid.push(0.0);
        let fitted_index = log_grid.len() - 1;

        let dimension = geometry.dimension;
        let mut penalty_share = Vec::<Vec<f64>>::with_capacity(log_grid.len());
        let mut null_weight = Vec::<Vec<f64>>::with_capacity(log_grid.len());
        let mut determinant = Vec::<f64>::with_capacity(log_grid.len());
        for &log_t in &log_grid {
            let t = log_t.exp();
            let mut share = Vec::with_capacity(dimension);
            let mut weight = Vec::with_capacity(dimension);
            // `log|I + tT(1)|`, over EVERY direction: an unpenalized one carries
            // `log(1 + 0) = 0` and is neither special-cased nor dropped.
            let mut log_det_hessian = 0.0_f64;
            for &nu in &generalized {
                let scaled = t * nu;
                let fraction = if scaled.is_finite() {
                    scaled / (1.0 + scaled)
                } else {
                    1.0
                };
                share.push(fraction);
                let shrinkage = 1.0 - fraction;
                weight.push(2.0 * shrinkage - shrinkage * shrinkage);
                log_det_hessian += scaled.ln_1p();
            }
            // `log|T(t)|₊ = rank·ln t + Σ_{j < rank} ln ν_j`, over the STRUCTURAL
            // rank. Deciding that index set by `ν_j > 0` is what put a spurious
            // `−ln t` per roundoff-positive null direction into the criterion.
            let log_det_penalty = geometry.rank as f64 * log_t + constant;
            penalty_share.push(share);
            null_weight.push(weight);
            determinant.push(log_det_hessian - log_det_penalty);
        }

        let mut selection_sample = Vec::with_capacity(draws);
        let mut conditional_sample = Vec::with_capacity(draws);
        let fitted_weight = null_weight[fitted_index].clone();
        let mut squares = vec![0.0_f64; dimension];
        let mut stream = SelectionDrawStream::new(dimension, draws);
        for _ in 0..draws {
            stream.fill_chi_square_ones(&mut squares);
            // argmin of the criterion over the window.
            let mut best = f64::INFINITY;
            let mut best_index = 0usize;
            for index in 0..log_grid.len() {
                let share = &penalty_share[index];
                let mut value = determinant[index];
                for (&square, &fraction) in squares.iter().zip(share.iter()) {
                    value += square * fraction;
                }
                if value < best {
                    best = value;
                    best_index = index;
                }
            }
            let selected = &null_weight[best_index];
            let mut chosen = 0.0_f64;
            let mut held = 0.0_f64;
            for ((&square, &weight), &held_weight) in
                squares.iter().zip(selected.iter()).zip(fitted_weight.iter())
            {
                chosen += square * weight;
                held += square * held_weight;
            }
            selection_sample.push(chosen);
            conditional_sample.push(held);
        }
        Some(Self {
            generalized: ascending(generalized),
            selection_sample,
            conditional_sample,
        })
    }

    /// Replay a term whose `λ̂` is a VECTOR, over each of its scales separately.
    ///
    /// A single-penalty term's selection is one-dimensional and diagonalizes
    /// (that is [`Self::generate_common_scale`]). A term with `m` penalties — a
    /// double-penalty smooth is `m = 2`, a tensor product more — selects `m`
    /// scales, and no single basis diagonalizes `m` penalties against the
    /// information at once. Scaling all of them together is a one-dimensional
    /// SLICE of that selection, and the slice is not enough: on a two-penalty
    /// Gaussian null,
    ///
    /// ```text
    ///                             α = .20    .10     .05     .01
    ///   conditional (no replay)     .4080   .3400   .2800   .1800
    ///   replay, common scale only   .2840   .1440   .0720   .0040
    ///   replay, both scales         .0440   .0120   .0080   .0040
    /// ```
    ///
    /// so the term's own `m` scales are gridded independently here. The absolute
    /// numbers in that table are from a harness whose own outer optimizer is a
    /// Nelder–Mead on a flat two-dimensional REML surface and are not to be read
    /// as calibration figures; the ORDERING is what it establishes, and the
    /// ordering is that the missing dimensions matter more than anything else
    /// measured on this issue.
    ///
    /// Each axis carries its OWN window. The reachable set for scale `i` is
    /// `ln t_i ∈ [−RHO_BOUND − ρ̂_i, RHO_BOUND − ρ̂_i]`, and those `m` intervals
    /// are only equal when the `m` fitted scales are. Handing this grid the
    /// COMMON-shift intersection — which is what it used to receive — truncates
    /// every axis to the narrowest one and empties the whole replay as soon as
    /// one `λ̂` rails, which for a null-true double-penalty smooth is the normal
    /// state and not a corner case.
    ///
    /// The grid is `SMOOTH_LR_SELECTION_GRID_BUDGET^(1/m)` points per axis, so
    /// the cost is bounded independently of `m`, and past
    /// [`SMOOTH_LR_SELECTION_MAX_SCALES`] scales the axes would be too coarse to
    /// be a selection at all and the common-scale slice is used instead.
    fn generate_multiscale(
        geometry: &SelectionGeometry,
        log_scale_windows: &[(f64, f64)],
        draws: usize,
    ) -> Option<Self> {
        let scales = geometry.roots.len();
        if scales < 2 || scales > SMOOTH_LR_SELECTION_MAX_SCALES {
            return None;
        }
        if log_scale_windows.len() != scales {
            return None;
        }
        // One axis per scale, budgeted so the total point count does not grow
        // with `m`. A scale whose own window is empty — its `λ̂` railed against
        // both walls at once — contributes a single node at the fitted point
        // rather than sinking the whole replay.
        let per_axis = ((SMOOTH_LR_SELECTION_GRID_BUDGET as f64)
            .powf(1.0 / scales as f64)
            .floor() as usize)
            .max(2);
        let mut axes = Vec::<Vec<f64>>::with_capacity(scales);
        let mut movable = 0usize;
        for &(low, high) in log_scale_windows {
            if !(low.is_finite() && high.is_finite()) {
                return None;
            }
            if high <= low {
                axes.push(vec![0.0]);
                continue;
            }
            movable += 1;
            axes.push(
                (0..per_axis)
                    .map(|step| low + (high - low) * (step as f64) / ((per_axis - 1) as f64))
                    .collect(),
            );
        }
        if movable == 0 {
            return None;
        }

        // The grid gets ONE extra point: the fitted `λ̂` itself, `ln t_i = 0` on
        // every axis. It has to be there twice over. The selection must be able
        // to choose the scale the fit chose — for the observed data it IS that
        // scale, by construction — and the control variate's conditional arm has
        // to be read AT it, not at whichever node happens to be nearest. With
        // `441^(1/2) = 21` points over a span of `35` the nearest node can be
        // `0.9` away in `ln λ`, a factor of 2.4, and the "conditional" sample
        // would then be a different law from the one whose tail the shift is
        // added to.
        let points = axes.iter().map(|axis| axis.len()).product::<usize>() + 1;
        let fitted_index = points - 1;

        let dimension = geometry.dimension;
        let mut bases = Vec::<Array2<f64>>::with_capacity(points);
        let mut shares = Vec::<Vec<f64>>::with_capacity(points);
        let mut weights_grid = Vec::<Vec<f64>>::with_capacity(points);
        let mut offsets = Vec::<f64>::with_capacity(points);
        let mut log_t = vec![0.0_f64; scales];
        for point in 0..points {
            let mut remainder = point;
            for (scale, axis) in axes.iter().enumerate() {
                log_t[scale] = if point == fitted_index {
                    0.0
                } else {
                    let step = remainder % axis.len();
                    remainder /= axis.len();
                    axis[step]
                };
            }
            let evaluated = geometry.at(&log_t)?;
            bases.push(evaluated.basis);
            shares.push(evaluated.shares);
            weights_grid.push(evaluated.weights);
            offsets.push(evaluated.offset);
        }
        let generalized = {
            let fitted = geometry.at(&vec![0.0_f64; scales])?;
            ascending(fitted.eigenvalues)
        };

        let mut selection_sample = Vec::with_capacity(draws);
        let mut conditional_sample = Vec::with_capacity(draws);
        let mut normals = vec![0.0_f64; dimension];
        let mut projected = vec![0.0_f64; dimension];
        let mut stream = SelectionDrawStream::new(dimension, draws);
        for _ in 0..draws {
            stream.fill_normals(&mut normals);
            let mut best = f64::INFINITY;
            let mut best_index = 0usize;
            for point in 0..points {
                let basis = &bases[point];
                let share = &shares[point];
                let mut value = offsets[point];
                for column in 0..dimension {
                    let mut coordinate = 0.0_f64;
                    for row in 0..dimension {
                        coordinate += basis[[row, column]] * normals[row];
                    }
                    value += coordinate * coordinate * share[column];
                }
                if value < best {
                    best = value;
                    best_index = point;
                }
            }
            let quadratic = |point: usize, out: &mut [f64]| -> f64 {
                let basis = &bases[point];
                let weight = &weights_grid[point];
                let mut total = 0.0_f64;
                for column in 0..dimension {
                    let mut coordinate = 0.0_f64;
                    for row in 0..dimension {
                        coordinate += basis[[row, column]] * normals[row];
                    }
                    out[column] = coordinate;
                    total += coordinate * coordinate * weight[column];
                }
                total
            };
            selection_sample.push(quadratic(best_index, &mut projected));
            conditional_sample.push(quadratic(fitted_index, &mut projected));
        }
        Some(Self {
            generalized,
            selection_sample,
            conditional_sample,
        })
    }
    /// `(shift, standard_error)`: how much the selection moves the tail at
    /// `statistic`, and the Monte-Carlo standard error of that shift.
    ///
    /// The shift is `P̂(W_sel ≥ x) − P̂(W_cond ≥ x)` on shared draws. Its variance
    /// is that of the paired indicator DIFFERENCE `d_i ∈ {−1, 0, +1}`, which is
    /// zero on every draw whose selected `t` did not move it across `x` — that
    /// is the control variate, and it is why the standard error is a fraction of
    /// the naive `√(p(1−p)/N)`.
    fn tail_shift(&self, statistic: f64) -> (f64, f64) {
        let draws = self.selection_sample.len();
        if draws == 0 {
            return (0.0, 0.0);
        }
        let mut sum = 0.0_f64;
        let mut sum_squares = 0.0_f64;
        for (&selected, &held) in self
            .selection_sample
            .iter()
            .zip(self.conditional_sample.iter())
        {
            let difference = f64::from(selected >= statistic) - f64::from(held >= statistic);
            sum += difference;
            sum_squares += difference * difference;
        }
        let count = draws as f64;
        let shift = sum / count;
        // `d_i ∈ {−1, 0, +1}` and is zero on every draw whose selected `t` left
        // it on the same side of `statistic` — which is most of them. That is
        // the control variate, and this is its own sample variance rather than
        // the `√(p(1−p)/N)` of either term alone.
        let variance = (sum_squares / count - shift * shift).max(0.0);
        (shift, (variance / count).sqrt())
    }
}

/// Deterministic `χ²_1` draws for the selection replay.
///
/// A p-value must not depend on a thread count, a machine or a run (#1017), so
/// the replay cannot take draws from a shared or seeded-at-startup generator. It
/// uses a counter-based stream instead: SplitMix64 on an index, mapped through
/// [`gam_math::probability::standard_normal_quantile`] and squared. Same
/// spectrum, same window, same numbers, everywhere, forever.
///
/// The stream is STRATIFIED per coordinate: draw `i` of coordinate `k` takes its
/// uniform from the `i`-th of `N` equal bins, in an order permuted per
/// coordinate. That is a Latin hypercube, and for a functional that is nearly a
/// sum over coordinates — which `W = Σ_k w_k u_k²` is exactly — it removes the
/// part of the Monte-Carlo error the bins already account for.
struct SelectionDrawStream {
    /// The `N` stratum midpoints, mapped through the normal quantile and
    /// squared. Every coordinate draws from THIS set — only the order differs —
    /// so the quantile is evaluated `N` times per term rather than `N × q`.
    values: Vec<f64>,
    /// The same strata as SIGNED normal quantiles.
    signed: Vec<f64>,
    /// One permutation of `0..N` per coordinate.
    permutations: Vec<Vec<u32>>,
    index: usize,
}

impl SelectionDrawStream {
    fn new(dimension: usize, draws: usize) -> Self {
        let signed: Vec<f64> = (0..draws)
            .map(|bin| {
                // Bin midpoint: never `0` or `1`, so the quantile is finite.
                let uniform = (bin as f64 + 0.5) / draws as f64;
                gam_math::probability::standard_normal_quantile(uniform)
                    .expect("a bin midpoint is strictly inside (0, 1)")
            })
            .collect();
        let values: Vec<f64> = signed.iter().map(|normal| normal * normal).collect();
        let mut permutations = Vec::with_capacity(dimension);
        for coordinate in 0..dimension {
            let mut order: Vec<u32> = (0..draws as u32).collect();
            // Fisher–Yates with a counter-based stream keyed by the coordinate.
            let mut state = 0x9E37_79B9_7F4A_7C15_u64
                ^ (coordinate as u64).wrapping_mul(0x94D0_49BB_1331_11EB);
            for position in (1..order.len()).rev() {
                state = split_mix64(state);
                let pick = (state % (position as u64 + 1)) as usize;
                order.swap(position, pick);
            }
            permutations.push(order);
        }
        Self {
            values,
            signed,
            permutations,
            index: 0,
        }
    }

    /// Fill one draw. The `zip` is the length contract: the stream writes one
    /// value per coordinate it was built for and nothing beyond, so a
    /// mis-sized buffer is a short write rather than an assertion.
    fn fill_chi_square_ones(&mut self, out: &mut [f64]) {
        for (slot, permutation) in out.iter_mut().zip(self.permutations.iter()) {
            *slot = self.values[permutation[self.index] as usize];
        }
        self.index += 1;
    }

    /// The same stratified draw as SIGNED normals, for the multi-scale replay:
    /// there the tested block is not diagonal at every grid point, so the
    /// quadratic form needs the vector and not its coordinatewise squares. The
    /// sign is taken from the stratum's own side of the median, which is what
    /// makes it the normal quantile rather than its absolute value.
    fn fill_normals(&mut self, out: &mut [f64]) {
        for (slot, permutation) in out.iter_mut().zip(self.permutations.iter()) {
            *slot = self.signed[permutation[self.index] as usize];
        }
        self.index += 1;
    }
}

/// SplitMix64, used only to permute the strata. Any full-period mixer would do;
/// what matters is that it is a pure function of an index.
#[inline]
fn split_mix64(state: u64) -> u64 {
    let mut z = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// The reference distribution [`SmoothTermLrInference`] scores its statistic
/// against, reported as the two spectral moments it is built from (#2672).
///
/// # What the statistic's null law actually is
///
/// Expand the log-likelihood quadratically about the unpenalized MLE `β̃` and
/// write `I = X'WX`, `S` for the penalty, `H = I + S`, `j` for the tested block
/// and `n` for the retained one. The penalized fit is `β̂ = Fβ̃` with
/// `F = H⁻¹I`, and the null fit is the retained block's own projection, so
///
/// ```text
/// W = β̃_j' (Ĩ_jj − N) β̃_j ,   β̃_j ~ N(0, Ĩ_jj⁻¹)
/// ```
///
/// with `Ĩ_jj = I_jj − I_jn I_nn⁻¹ I_nj` the Schur complement and
/// `N = [S H⁻¹ I H⁻¹ S]_jj`. Setting `H̃ = Ĩ_jj + S_jj` and `P = H̃⁻¹S_jj`, that
/// collapses to `(Ĩ_jj − N)Ĩ_jj⁻¹ = H̃(I − P²)H̃⁻¹`, and — because the block of
/// the GLOBAL influence matrix equals the Schur-complement influence,
/// `F_jj = H̃⁻¹Ĩ_jj = I − P` — the eigenvalues are exactly `2F_jj − F_jj²`.
/// So
///
/// ```text
/// W = Σ_j w_j χ²_1 ,   w = eig(2·F_jj − F_jj²) ∈ (0, 1]^q.
/// ```
///
/// # Consequences, and what this replaced
///
/// `Σ w_j = 2 tr(F_jj) − tr(F_jj²)` is Wood's `edf1`. So `edf1` is not a
/// citation here, it is the statistic's first-order null MEAN, derived. What it
/// is not is a chi-square degrees of freedom: `Var(W) = 2 Σ w_j²` against the
/// mean-matched `χ²_{Σw}`'s `2 Σ w_j`, and `w_j ≤ 1`, so a mean-matched chi-square
/// is over-dispersed for every penalized term and the test is conservative by
/// construction.
///
/// # Why the reference is the spectrum and not two of its moments
///
/// Matching the second moment as well — `W ≈ g·χ²_ν` with `ν = (Σw)²/Σw²`,
/// `g = Σw²/Σw` — fixes the *shape* with no free constant, and is EXACT whenever
/// the weights are equal (which includes the classical unpenalized case
/// `w ≡ 1 ⇒ ν = q, g = 1 ⇒ χ²_q`). It is not exact otherwise, and the error is
/// one-signed and grows with the depth of the tail, which is the half of the
/// p-value range that decides anything. Measured against the exact law on the
/// shrinkage spectrum `f_j = 1/(1 + λγ_j)` of a second-difference penalty, over
/// six decades of `λ` and `k ∈ {6, 12, 20}`, the size a two-moment reference
/// actually delivers at a nominal `α` is
///
/// ```text
/// α = 0.05   0.99 – 1.02 ×      α = 1e-3   1.01 – 1.31 ×
/// α = 0.01   1.00 – 1.11 ×      α = 1e-4   1.14 – 1.61 ×
/// ```
///
/// i.e. it is fine where the test is least discriminating and up to 61%
/// anti-conservative where it is most.
///
/// Where the gap lives matters as much as its size, and it is not where the
/// intuition puts it. The surrogate is exact at BOTH ends of the shrinkage
/// range — `w ≡ 1` unpenalized, and a single distinct weight once REML has
/// shrunk a term to its null space (measured on a null-true `k = 12` fit:
/// `w = (0.322, 5.9e-7, 7.1e-8, …)`, where the two references agree to eight
/// figures). It opens in the middle, at moderate shrinkage, which is exactly
/// where a smooth term carrying real signal sits. Nothing about the statistic requires that
/// trade: the weights are the parameters of an exactly invertible
/// characteristic function, and [`gam_math::probability::weighted_chi_square_sf`]
/// inverts it (Imhof) with a *returned* truncation bound of `1e-11` — eight
/// orders below the smallest tail any of the numbers above resolves. So the
/// reference is `P(Σ_j w_j χ²_1 > W)` itself, and the `(ν, g)` pair survives only
/// as a two-number summary of the spectrum's shape, published for continuity and
/// no longer consulted when the spectrum is known.
///
/// The one-moment reference this replaced sits at the far end of the same axis,
/// with the same sign: on a spectrum shaped like a shrunk smooth
/// (`0.08, 0.02, 0.005, 0.001, 2e-4`), at `x = 8·Σw` the exact tail is `1.4e-3`
/// and the mean-matched chi-square reports `3.6e-2` — 26× conservative.
///
/// # What went away with the mean-only reference
///
/// Three things, and none of them needed a replacement:
///
/// * `+ tr(X'WX · J Var(ρ̂) Jᵀ)/φ`, the Wood–Pya–Säfken smoothing-parameter
///   inflation added under #1872. That is a *coefficient-covariance* correction
///   for AIC; it is not a term in this statistic's null law, and it is largest
///   exactly where the outer criterion is flattest — i.e. where the term has the
///   LEAST effective d.f. Measured on the #2672 fixture: a replicate with
///   `edf = 0.070` was handed `rho_uncertainty = 1.79`, twenty-five times the
///   term's own effective d.f. It was holding the size up by an unrelated
///   mechanism. λ̂'s sampling variation enters `E[W]` through the estimated-λ
///   Lawley shift already applied as the Bartlett factor, at the `O(n⁻¹)` order
///   it belongs to.
/// * `.max(edf)` and `.max(null_dim)`. Both are automatic: `w_j = 1` exactly on
///   an unpenalized direction, so `Σ w_j ≥ null_dim` by construction, and `Σ w_j`
///   dominates `tr(F_jj) = edf` because `w_j = 2f_j − f_j² ≥ f_j` for `f_j ∈ [0,1]`.
/// * `.max(1.0)`, the #1766 degeneracy floor. It existed because `χ²_d` with
///   `d → 0` reports any positive `W` as maximally significant. The scaled
///   reference cannot degenerate that way: as REML shrinks a term the weights and
///   the statistic collapse *together*, `W/g` stays `O(1)`, and `ν → q`. The floor
///   was a patch on the wrong shape, not on a missing quantity.
#[derive(Clone, Debug, PartialEq)]
pub struct SmoothLrReferenceDf {
    /// The null spectrum itself, `w_j ∈ [0, 1]`, sorted descending — the whole
    /// reference on the [`SmoothLrReferenceSource::NullSpectrum`] lane. Empty on
    /// the two lanes that could not reach it, which is exactly the condition
    /// under which [`Self::tail_probability`] falls back to the `(ν, g)` pair.
    pub weights: Vec<f64>,
    /// First spectral moment `Σ_j w_j = 2·tr(F_jj) − tr(F_jj²)` — Wood's `edf1`,
    /// and exactly the statistic's first-order null mean `E[W|λ]`. This is the
    /// `d` the Lawley Bartlett factor `c = 1 + Δε/d` is denominated in.
    pub mean: f64,
    /// Second spectral moment `Σ_j w_j² = tr((2F_jj − F_jj²)²)`, i.e. `Var(W)/2`.
    pub second_moment: f64,
    /// Shape of the two-moment SUMMARY `ν = mean²/second_moment`. It is what the
    /// reference used to be, and it is still what the reference is on the
    /// [`SmoothLrReferenceSource::SpectralMomentMatch`] and
    /// [`SmoothLrReferenceSource::UnitWeightFallback`] lanes; on the exact lane
    /// it is a published descriptor of the spectrum's shape and is not consulted.
    pub chi_square_df: f64,
    /// Scale of that summary, `g = second_moment/mean`. Same status as
    /// [`Self::chi_square_df`].
    pub scale: f64,
    /// The agreement between the two independently-assembled routes to this
    /// spectrum, when the fit supplied the inputs for both: the larger of the
    /// two relative residuals between `(Σw, Σw²)` read off `[H⁻¹]_jj S_jj` and
    /// `(tr A, tr A²)` read off the influence block, `A = 2F_jj − F_jj²`.
    ///
    /// The two are the same object by an algebraic identity that depends on the
    /// penalty being block-diagonal by term AND on `Vb`, `F` and `S` being
    /// published in one coefficient basis. Neither is checkable by inspection,
    /// and both have been wrong here before (`#2672`'s similarity-map drop, its
    /// internal-basis first-order correction, and its block-local
    /// `coeff_range`). So the driver measures the identity on every fit that can
    /// support it and publishes the number rather than assuming it.
    ///
    /// `None` when only one route was available — which is a statement about the
    /// fit, not a failure.
    pub moment_residual: Option<f64>,
    /// The term's conditional effective degrees of freedom `tr(F_jj)`
    /// (`per_term_edf`), reported for continuity with the summary table and used
    /// as the fallback base when neither the spectrum nor its moments are
    /// available.
    pub edf: f64,
    /// The term's joint unpenalized null-space dimension `dim(∩_k null(S_k))`,
    /// reported because it is the analytic lower bound on `mean` and therefore
    /// the cheapest check that the spectrum was assembled on the right block.
    pub null_dim: usize,
    /// Which lane supplied the reference.
    pub source: SmoothLrReferenceSource,
    /// The λ̂-selection replay, when the term has a penalized direction and the
    /// fit left the outer criterion room to choose (#2672). `None` means the
    /// conditional law IS the selection law here — nothing was selected — and
    /// the tail is read from [`Self::weights`] alone.
    pub selection: Option<SmoothLrSelectionReplay>,
    /// The relative resolution of the statistic this reference will be asked
    /// about — the fit's own outer convergence tolerance (`FitOptions::tol`).
    ///
    /// `W = 2(ℓ_full − ℓ_null)` is a difference of two SEPARATELY converged
    /// optimizations, so it is not known better than that, and a p-value cannot
    /// be more accurate than the statistic it is read from. See
    /// [`Self::tail_probability_with_bound`] for what this is used for and why
    /// it is not a numerical-accuracy knob.
    pub statistic_resolution: f64,
}

/// Accumulated-roundoff floor on the requested tail accuracy.
///
/// The Imhof value is assembled as `0.5 + I/π` over `N` panels, so its own
/// arithmetic error is about `ε√N` — at the `10⁵`-panel scale this reference
/// reaches, `1e-13`. Asking the quadrature for a bound below that buys panels,
/// not digits.
const SMOOTH_LR_TAIL_ROUNDOFF_FLOOR: f64 = 1e-13;

/// Ceiling on the requested tail accuracy.
///
/// The derived request degenerates in one place: as `W → 0` the reference's
/// density diverges for `ν < 2`, so "how far does the p-value move when `W`
/// moves by its own resolution" becomes unbounded — while the p-value there is
/// within `1e-3` of one and nothing depends on it. This rail is the statement
/// that a probability is reported to at least three decimals whatever the
/// derivation says; it binds nowhere else, because `density · ΔW` falls below it
/// as soon as `W` leaves the origin.
const SMOOTH_LR_TAIL_COARSEST: f64 = 1e-3;

impl SmoothLrReferenceDf {
    /// `P(W > statistic)` under this reference.
    ///
    /// On the exact lane this is `P(Σ_j w_j χ²_1 > W)` by Imhof inversion; on the
    /// two surrogate lanes it is the two-moment `P(χ²_ν > W/g)`. Both are
    /// scale-equivariant in the same way, which is what lets the Bartlett
    /// correction be applied as `W/c` on either.
    ///
    /// A non-finite statistic propagates as `NaN` rather than being scored: the
    /// LR statistic is `NaN` exactly when the null refit did not produce a finite
    /// log-likelihood, and there is no p-value for a test that was not run.
    pub fn tail_probability(&self, statistic: f64) -> f64 {
        self.tail_probability_with_bound(statistic).0
    }

    /// The CONDITIONAL tail — the fixed-`λ` law alone, with the λ̂-selection
    /// replay held out. This is what [`Self::tail_probability`] returns when
    /// nothing was selected, and it is the reference the replay corrects.
    pub fn conditional_tail_probability(&self, statistic: f64) -> f64 {
        self.conditional_tail_with_bound(statistic).0
    }

    /// [`Self::tail_probability`] with the certified absolute bound the
    /// quadrature achieved on it.
    ///
    /// # How accurately the tail is resolved, and why that is derived
    ///
    /// Imhof's truncation point grows like `ε^{-2/(2+m)}` in the number `m` of
    /// weights active at it. A shrunk penalized smooth has ONE weight of order
    /// one over a tail of tiny ones, so `m = 1` across the whole useful range
    /// and the cost is `ε^{-2/3}`: at `gam-math`'s default `ε = 1e-11` a single
    /// p-value on a realistic spectrum measures **0.13 s to 3.3 s**. That is not
    /// an accuracy anyone asked for — it is the library's default standing in
    /// for a statement about what this particular answer is for.
    ///
    /// The statement is available. `W = 2(ℓ_full − ℓ_null)` is a difference of
    /// two separately-converged optimizations, so it is known to about
    /// `ΔW = tol · (W + E[W])` — the fit's own convergence tolerance on the
    /// natural scale of the statistic. A p-value is a deterministic function of
    /// `W`, so it is known to `|S(W) − S(W + ΔW)|` no matter how well the
    /// integral is done. **That** is what the quadrature is asked for, and it is
    /// evaluated through the two-moment summary — the distribution that used to
    /// BE the reference, which costs nothing and is within a factor of 1.6 of
    /// the exact tail everywhere it was measured, so it is an excellent scale
    /// for a derivative it is not being asked to be the value of.
    ///
    /// Resolving finer than this is arithmetic on the fit's own noise; resolving
    /// coarser would add some. The achieved bound is returned rather than
    /// assumed, so a consumer can see the accuracy instead of inheriting it.
    pub fn tail_probability_with_bound(&self, statistic: f64) -> (f64, f64) {
        let (conditional, bound) = self.conditional_tail_with_bound(statistic);
        let Some(replay) = self.selection.as_ref() else {
            return (conditional, bound);
        };
        if !conditional.is_finite() {
            return (conditional, bound);
        }
        let (shift, standard_error) = replay.tail_shift(statistic);
        (
            (conditional + shift).clamp(0.0, 1.0),
            bound + 2.0 * standard_error,
        )
    }

    fn conditional_tail_with_bound(&self, statistic: f64) -> (f64, f64) {
        if !statistic.is_finite() {
            return (f64::NAN, f64::NAN);
        }
        let summary =
            |w: f64| gam_math::probability::chi_square_sf(w / self.scale, self.chi_square_df);
        if self.weights.is_empty() {
            // The summary IS the reference on the two degraded lanes, and it is
            // a closed form: no truncation, so no bound to report.
            return (summary(statistic), 0.0);
        }
        let tolerance = if self.statistic_resolution.is_finite() && self.statistic_resolution > 0.0 {
            let delta = self.statistic_resolution * (statistic.abs() + self.mean.abs());
            (summary(statistic) - summary(statistic + delta))
                .abs()
                .clamp(SMOOTH_LR_TAIL_ROUNDOFF_FLOOR, SMOOTH_LR_TAIL_COARSEST)
        } else {
            // A reference built without a fit behind it (a unit test, a
            // hand-assembled spectrum) has no statistic resolution to derive
            // from, so it gets `gam-math`'s own default rather than a guess.
            gam_math::probability::WEIGHTED_CHI_SQUARE_TOLERANCE
        };
        gam_math::probability::weighted_chi_square_sf_to_tolerance(
            &self.weights,
            statistic,
            tolerance,
        )
    }
}

/// The Bartlett-corrected per-term significance report for one penalized smooth
/// term (#1063). Unlike the summary table's Wood rank-truncated **Wald**
/// statistic, this is a genuine **likelihood-ratio** statistic from a
/// constrained refit (the smooth dropped), so the exact Lawley LR Bartlett
/// factor corrects the right quantity.
#[derive(Clone, Debug)]
pub struct SmoothTermLrInference {
    /// Smooth-term name (matches the summary row).
    pub name: String,
    /// Smooth-term index within `resolvedspec.smooth_terms`.
    pub term_idx: usize,
    /// The uncorrected likelihood-ratio statistic `W = 2(ℓ_full − ℓ_null)`,
    /// floored at zero (a non-negative LR by construction).
    pub statistic_lr: f64,
    /// The statistic's first-order null mean `d = E[W|λ] = Σ_j w_j`, which is
    /// Wood's `edf1 = 2·tr(F_bb) − tr(F_bb²)` exactly (see
    /// [`SmoothLrReferenceDf`] for why that is a derivation and not a citation).
    /// This is the `d` the Lawley Bartlett factor `c = 1 + Δε/d` is denominated
    /// in. It is **not** a chi-square degrees of freedom — the reference the
    /// p-values are read from is [`Self::ref_df_provenance`]'s
    /// `chi_square_df`/`scale` pair, which coincides with `ref_df` only when the
    /// tested block is unpenalized.
    pub ref_df: f64,
    /// The reference distribution itself: both spectral moments of the null law,
    /// the `(ν, g)` pair resolved from them, and which lane supplied it (#2672).
    pub ref_df_provenance: SmoothLrReferenceDf,
    /// Lawley LR Bartlett factor `c = E[W]/d = 1 + Δε/d` when computable, else
    /// `1.0` (no correction).
    pub bartlett_factor: f64,
    /// Fixed-λ conditional factor `c_cond = 1 + Δε(ρ̂)/d` when the estimated-λ
    /// correction was applied. `None` means the applied factor was either the
    /// fixed-λ factor itself or no Lawley correction was available.
    pub bartlett_factor_conditional: Option<f64>,
    /// Increment in Lawley's LR mean shift due solely to ρ̂ sampling variation,
    /// `0.5 * tr(H_Δε Cov(ρ̂))`, when estimated-λ correction was applied.
    pub rho_variation_shift: Option<f64>,
    /// Bartlett-corrected statistic `W* = W / c`.
    pub statistic_corrected: f64,
    /// Uncorrected tail probability `P(χ²_ν > W/g)` under the null law's own
    /// two-moment reference.
    pub p_value_uncorrected: f64,
    /// Corrected tail probability `P(χ²_ν > W*/g)`; equals the uncorrected value
    /// when no correction was applied. Dividing the statistic by `c` and scaling
    /// every spectral weight by `c` are the same operation on this reference, so
    /// the Bartlett correction composes without a second convention.
    pub p_value_corrected: f64,
    /// Whether the second-order correction is **material** (#939 deliverable 4):
    /// the per-test diagnostic "is `n` too small for first-order inference
    /// *here*?". `true` when a correction was applied and it moves the result by
    /// more than [`SMOOTH_LR_MATERIAL_THRESHOLD`] — measured as the larger of the
    /// relative Bartlett-factor distance from one `|c − 1|` and the relative
    /// p-value change `|p* − p| / max(p, p*, ε)`. `false` when `correction` is
    /// [`SmoothLrCorrection::None`] (no correction was applied).
    pub material: bool,
    /// Which statistic the corrected p-value is built from.
    pub correction: SmoothLrCorrection,
    /// The CONDITIONAL tail of the corrected statistic — the p-value the
    /// fixed-`λ` law alone would report, before the λ̂-selection replay moves it
    /// (#2672).
    ///
    /// Published so the correction is visible rather than folded in:
    /// `p_value_corrected − p_value_conditional` is exactly what treating `λ̂` as
    /// chosen rather than given is worth on this fit, and it is the quantity a
    /// reader should be shown if they are going to be asked to accept it. Equal
    /// to `p_value_corrected` when no selection was possible.
    pub p_value_conditional: f64,
    /// Certified absolute accuracy of the two published p-values — the larger of
    /// the two truncation bounds the tail quadrature achieved (#2672).
    ///
    /// `0.0` on the closed-form lanes (a degraded reference, or a spectrum whose
    /// weights are all equal) because there is no truncation to bound. On the
    /// Imhof lane it is what the sweep reached against the accuracy
    /// [`SmoothLrReferenceDf::tail_probability_with_bound`] derived from the
    /// fit's own convergence tolerance, so a consumer reads the accuracy rather
    /// than inheriting it. A value large enough to matter means the quadrature
    /// hit its panel backstop, which is a statement about the spectrum's spread
    /// and not a defect in the p-value's derivation.
    pub p_value_bound: f64,
}

/// The materiality threshold for [`SmoothTermLrInference::material`] (#939
/// deliverable 4): a correction is flagged material when it changes the result
/// by more than 10%.
pub const SMOOTH_LR_MATERIAL_THRESHOLD: f64 = 0.10;

/// Build `S_b = lambda_b * S_b^unit` as global `p_total x p_total` matrices in
/// exactly the fitted rho/lambda ordering. This is the narrow handoff the
/// estimated-lambda Lawley correction needs: the same `design.penalties` order
/// already paired with `fit.lambdas`, without changing #740's outer-Hessian
/// algebra or the production penalty assembly.
fn fitted_rho_penalty_components(
    penalties: &[BlockwisePenalty],
    lambdas: &[f64],
    p_total: usize,
) -> Result<Vec<gam_terms::inference::lawley::RhoPenaltyComponent>, EstimationError> {
    if penalties.len() != lambdas.len() {
        return Err(EstimationError::InvalidInput(format!(
            "smooth_term_lr_inference: penalty/lambda count mismatch ({} penalties, {} lambdas)",
            penalties.len(),
            lambdas.len()
        )));
    }
    let mut components = Vec::with_capacity(penalties.len());
    for (idx, (penalty, &lambda)) in penalties.iter().zip(lambdas.iter()).enumerate() {
        if !(lambda.is_finite() && lambda >= 0.0) {
            return Err(EstimationError::InvalidInput(format!(
                "smooth_term_lr_inference: lambda[{idx}] is invalid: {lambda}"
            )));
        }
        let r = &penalty.col_range;
        if r.end > p_total {
            return Err(EstimationError::InvalidInput(format!(
                "smooth_term_lr_inference: penalty[{idx}] range {:?} exceeds coefficient dimension {p_total}",
                r
            )));
        }
        let mut s_component = Array2::<f64>::zeros((p_total, p_total));
        s_component
            .slice_mut(s![r.start..r.end, r.start..r.end])
            .scaled_add(lambda, &penalty.local);
        components.push(gam_terms::inference::lawley::RhoPenaltyComponent { s_component });
    }
    Ok(components)
}

/// The end-to-end per-term likelihood-ratio significance report for every
/// penalized (shape-unconstrained) smooth term in a fitted model, magically
/// Bartlett-corrected when the family carries closed-form Lawley cumulant jets
/// (#1063, follow-up to #939).
///
/// # Why an LR statistic (not the summary Wald)
///
/// The summary table's `wood_smooth_test` is Wood's rank-truncated **Wald**
/// statistic `T = β̂'Σ̂⁻β̂`. Lawley's ε corrects the **likelihood-ratio**
/// statistic, and under penalization the Wald form is already a weighted χ²
/// whose second-order mean is *not* `d + Δε` — dividing `T` by the LR factor
/// would correct the wrong statistic. The principled route (#1063 Option 1) is
/// to compute a real per-term LR statistic by a constrained refit and correct
/// *that*:
///
/// ```text
/// W = 2(ℓ_full − ℓ_null),   W* = W / c,   c = 1 + Δε/d,   p = P(χ²_d > W*).
/// ```
///
/// # Method
///
/// 1. Fit the full model and read `ℓ_full` and the per-term coefficient ranges /
///    EDF / influence block. The full design's column layout fixes the tested
///    block for the Lawley factor.
/// 2. For each penalized smooth term, refit a null model with that term dropped
///    from the spec; `W = max(2(ℓ_full − ℓ_null), 0)`.
/// 3. The reference d.f. `d` is the Wood truncation `tr(F)²/tr(F²)` on the
///    term's influence block (the same `ref_df` the summary Wald row reports),
///    floored at `max(edf, null_dim, 1)`: this LR test drops the whole term, so
///    `d` is at least the dimension the term spans when present (its null-space
///    dimension, never below 1). The non-symmetric `tr(F²)` can collapse toward
///    0 at a shrunk-to-null fit and violate that bound — see the inline note at
///    the `ref_df` binding.
/// 4. When the family has closed-form cumulant jets, evaluate Lawley's ε at the
///    **null** linear predictor (an expectation evaluated at the null fit), fold
///    the full λ-scaled penalty `S_λ` into the information, and Bartlett-correct
///    `W` with [`gam_terms::inference::lawley::lawley_lr_bartlett_factor`]. The
///    null annihilates the tested block's penalty (`S_λ β₀ = 0` on that block),
///    so the penalized Lawley expansion applies verbatim.
/// 5. Otherwise (no closed-form jets, or a null refit that did not converge) the
///    uncorrected `χ²_d` stands with provenance `none` — never weakened.
///
/// Random-effect smooths and shape-constrained smooths are skipped (their tests
/// are not a central-χ² LR), matching the summary table's policy.
pub fn smooth_term_lr_inference_forspec(
    data: ArrayView2<'_, f64>,
    y: ArrayView1<'_, f64>,
    weights: ArrayView1<'_, f64>,
    offset: ArrayView1<'_, f64>,
    resolvedspec: &TermCollectionSpec,
    family: LikelihoodSpec,
    options: &FitOptions,
) -> Result<Vec<SmoothTermLrInference>, EstimationError> {
    use gam_terms::inference::lawley::{
        LAWLEY_PAIR_MATRIX_MAX_ROWS, known_scale_expected_jets_with_dispersion,
        lawley_lr_bartlett_factor, lawley_lr_mean_shift_with_rho_variation,
    };

    let n = data.nrows();
    // Full fit: ℓ_full, the per-term coefficient ranges/EDF/influence, and the
    // full design whose column layout fixes each tested block for Lawley.
    let full = fit_term_collection_forspec(
        data,
        y,
        weights,
        offset,
        resolvedspec,
        family.clone(),
        options,
    )?;
    let ll_full = full.fit.log_likelihood;
    let p_total = full.design.design.ncols();
    let lambdas = full.fit.lambdas.as_slice().ok_or_else(|| {
        EstimationError::InvalidInput(
            "smooth_term_lr_inference: non-contiguous lambda vector".to_string(),
        )
    })?;
    let s_lambda = weighted_blockwise_penalty_sum(&full.design.penalties, lambdas, p_total);
    let rho_penalty_components =
        fitted_rho_penalty_components(&full.design.penalties, lambdas, p_total)?;
    let rho_covariance = full.fit.artifacts.rho_covariance.as_ref().filter(|cov| {
        cov.nrows() == rho_penalty_components.len() && cov.ncols() == rho_penalty_components.len()
    });
    // Full design as a dense n×p array for the Lawley pair-matrix reduction.
    let full_design_dense = full.design.design.to_dense();
    let influence = full.fit.coefficient_influence();
    // `H⁻¹`, unscaled: `beta_covariance()` publishes `Vb = H⁻¹·scale`, and the
    // scale is the family's own documented coefficient-covariance multiplier
    // (`σ̂²` for the profiled Gaussian, `1` for every family whose IRLS weight
    // already carries the dispersion). The null spectrum is `1 − eig(H⁻¹_jj
    // S_jj)²`, a product of two matrices in reciprocal units, so the multiplier
    // has to come off exactly here or every weight is wrong by that factor.
    // A family with no scalar multiplier (custom/GAMLSS) yields `None` and the
    // reference drops to the two-moment rung, which needs no scale at all.
    let hessian_inverse = full
        .fit
        .coefficient_covariance_scale()
        .ok()
        .filter(|scale| scale.is_finite() && *scale > 0.0)
        .zip(full.fit.beta_covariance())
        .map(|(scale, covariance)| covariance.mapv(|value| value / scale));
    // `SmoothTerm::coeff_range` is BLOCK-LOCAL — 0-based within the smooth block
    // — while the global coefficient layout is `[intercept | linear | random |
    // smooth]`. Every consumer that indexes a global object with it has to shift
    // by `smooth_start` first (`smooth_term_summary.rs`, the constraint audit and
    // the anisotropic provider all do). This driver did not, and it indexes FOUR
    // global objects with it: the influence matrix `F` (both the per-term EDF
    // trace and Wood's `edf1`), the weighted Gram and correction inside the WPS
    // trace, and — worst — the `tested` column set handed to Lawley, which
    // decides WHICH HYPOTHESIS the mean shift is computed for.
    //
    // This is the #1360 defect in a fourth place: the window slides one column
    // per preceding parametric column, folding the intercept and the linear
    // terms into the smooth's block and dropping as many real smooth columns off
    // the end. It is never zero — the intercept alone makes `smooth_start ≥ 1`.
    //
    // It was invisible because three of the four consumers were degraded to
    // index-free fallbacks: `coefficient_influence` was `None` on every model
    // with a conditioned parametric column (fixed alongside this, #2672), so
    // `per_term_edf` fell through to the penalty-block-trace channel — which is
    // indexed by PENALTY block, not by coefficient, and is therefore correct —
    // and `wood_reference_df` returned `None` outright. Restoring `F` is what
    // made the offset observable: on this issue's `y ~ x + s(z)` fixture the
    // per-term EDF of a null smooth jumped from `0.054` (penalty-trace channel,
    // correct) to `2.040` (influence trace over columns `0..9`), which is the
    // unpenalized intercept's `1` plus the parametric `x`'s `1` plus the smooth's
    // own `0.04` — the offset read off the arithmetic.
    let smooth_start = p_total.saturating_sub(full.design.smooth.total_smooth_cols());
    let fitted_likelihood = resolved_likelihood_for_fit(&full.fit)?;
    let family_disp = lawley_dispersion_for_family(&fitted_likelihood, &full.fit)?;

    let mut out = Vec::<SmoothTermLrInference>::new();
    for (term_idx, design_term) in full.design.smooth.terms.iter().enumerate() {
        let penalty_range = full
            .design
            .smooth_term_penalty_range(term_idx)
            .map_err(EstimationError::InvalidInput)?;
        let (block_start, k) = penalty_range
            .map(|range| (range.start, range.len()))
            .unwrap_or((0, 0));
        // Shape-constrained smooths get no central-χ² LR (cone-projected
        // boundary test); the summary table skips them too.
        if design_term.shape != ShapeConstraint::None {
            continue;
        }
        // Shifted into the GLOBAL coefficient layout — see `smooth_start` above.
        let coeff_range = (smooth_start + design_term.coeff_range.start)
            ..(smooth_start + design_term.coeff_range.end);
        if coeff_range.start >= coeff_range.end || coeff_range.end > p_total {
            continue;
        }
        // Per-term EDF for the χ² reference df FALLBACK (used only when the
        // influence matrix `F` is unavailable). Route through `per_term_edf`,
        // which uses the ADDITIVE per-block trace channel
        // (`|coeff_range| − Σ_{kk∈term} tr_kk`) and caps at the model total,
        // rather than the raw `edf_by_block` block-sum `Σ_{kk}(rank_kk − tr_kk)`.
        // For a multi-penalty term (te/ti/double-penalty) the penalties share one
        // coefficient range, so the rank-based block-sum OVER-COUNTS the term EDF
        // (Σ rank_kk > |coeff_range|) and would inflate the LR reference df,
        // biasing the smooth-term test conservative on large/sparse fits where `F`
        // is not materialised. (Same per-block over-count class as the multinomial
        // `edf_per_class` fix.)
        let edf = full.fit.per_term_edf(coeff_range.clone(), block_start, k);
        // The term's **joint** unpenalized null-space dimension: the coefficient
        // directions penalized by *no* active penalty — the polynomial part a
        // penalized smooth always carries when present, which no penalty can
        // shrink. This is `dim(∩_k null(S_k)) = p_local − rank(Σ_k S_k)`, the
        // INTERSECTION of the per-penalty null spaces, computed by
        // `wald_unpenalized_dim()` — the very same scalar the summary Wald test
        // (`wood_smooth_test`) floors its reference d.f. at, so the LR and Wald
        // tests reference a consistent d.f.
        //
        // It must NOT be `nullspace_dims.iter().sum()`: that *unions* the null
        // spaces (the #1360 defect — see `joint_unpenalized_dim`'s docs). A
        // double-penalty smooth carries a bending penalty (null space = its
        // polynomial part) plus a complementary null-space ridge (which penalizes
        // exactly that polynomial part), so the two null spaces are disjoint and
        // the joint null space is EMPTY (dim 0) — yet the per-penalty dims sum to
        // ~`p_local`. Flooring `ref_df` at that sum pins it to the full basis
        // dimension for every fit (e.g. 11 for a k=12 s(x)), making the LR test
        // badly conservative for genuine moderate signals while only accidentally
        // masking the collapse.
        let null_dim = design_term.wald_unpenalized_dim();
        // The reference the whole-term LR statistic is scored against: the first
        // two moments of its OWN null law, not a chi-square fitted to its mean.
        // See `lr_null_reference` for the derivation and for what this replaced.
        // The term's own penalties restricted to the tested coefficient block,
        // carried λ-FREE with their `ρ̂_i = ln λ̂_i` alongside. The separation is
        // not bookkeeping: the replay's criterion needs `log|Σ_i λ_i S_i|₊`, and
        // that quantity is only computable from the components and their scales
        // — an assembled sum has already lost it (#2644). See
        // [`SelectionGeometry`].
        let mut term_penalties = Vec::<Array2<f64>>::new();
        let mut term_log_lambda = Vec::<f64>::new();
        for (blockwise, &lambda) in full
            .design
            .penalties
            .get(block_start..block_start + k)
            .into_iter()
            .flatten()
            .zip(lambdas[block_start..(block_start + k).min(lambdas.len())].iter())
        {
            let range = &blockwise.col_range;
            if range.start < coeff_range.start || range.end > coeff_range.end {
                continue;
            }
            if !(lambda.is_finite() && lambda > 0.0) {
                continue;
            }
            let mut local = Array2::<f64>::zeros((coeff_range.len(), coeff_range.len()));
            let offset = range.start - coeff_range.start;
            let width = range.end - range.start;
            for row in 0..width {
                for column in 0..width {
                    local[[offset + row, offset + column]] = blockwise.local[[row, column]];
                }
            }
            term_penalties.push(local);
            term_log_lambda.push(lambda.ln());
        }
        // ONE window per scale. The outer search moved each `ρ_i` independently
        // inside its box, so scale `i` could reach `ln t_i ∈ [−B − ρ̂_i, B − ρ̂_i]`
        // and no further. The single COMMON-shift window this used to compute is
        // the intersection of those intervals: correct for a slice that moves
        // every scale together (which is what `generate_common_scale` still
        // derives from these), and wrong for a grid that moves them separately.
        // On a null-true double-penalty smooth at `ρ̂ = (18, −24)` each axis has
        // ~50 of room and the intersection leaves 18; when one `λ̂` rails the
        // intersection is EMPTY and the replay was declined outright.
        let reach = 2.0 * gam_solve::estimate::RHO_BOUND;
        let log_scale_windows: Vec<(f64, f64)> = term_log_lambda
            .iter()
            .map(|&rho| {
                // Clamped to the box's own full width: no `ρ` can move further
                // than from one wall to the other, and a `λ̂` that underflowed to
                // zero would otherwise put `ln t` at `±744` and `t` at infinity.
                (
                    (-gam_solve::estimate::RHO_BOUND - rho).clamp(-reach, reach),
                    (gam_solve::estimate::RHO_BOUND - rho).clamp(-reach, reach),
                )
            })
            .collect();
        let reference = lr_null_reference(
            influence,
            hessian_inverse.as_ref(),
            Some(&s_lambda),
            &coeff_range,
            edf,
            null_dim,
            options.tol,
            &log_scale_windows,
            &term_penalties,
            &term_log_lambda,
        );
        let ref_df = reference.mean;
        if !(ref_df.is_finite()
            && ref_df > 0.0
            && reference.chi_square_df.is_finite()
            && reference.chi_square_df > 0.0
            && reference.scale.is_finite()
            && reference.scale > 0.0)
        {
            continue;
        }
        let ref_df_provenance = reference.clone();

        // Null model: drop this smooth term from the spec and refit. The term's
        // name pins which spec entry to remove (design and spec share names).
        let mut null_spec = resolvedspec.clone();
        let Some(spec_pos) = null_spec
            .smooth_terms
            .iter()
            .position(|t| t.name == design_term.name)
        else {
            continue;
        };
        null_spec.smooth_terms.remove(spec_pos);
        let null_fit = fit_term_collection_forspec(
            data,
            y,
            weights,
            offset,
            &null_spec,
            family.clone(),
            options,
        );
        let (statistic_lr, eta_null) = match null_fit {
            Ok(null) if null.fit.log_likelihood.is_finite() => {
                let w = (2.0 * (ll_full - null.fit.log_likelihood)).max(0.0);
                // η at the null fit: X_null β_null + affine_offset + offset
                // (per-row linear predictor; design-layout independent — Lawley
                // reads it on the full design rows). `compose_offset` folds the
                // design's fixed affine channel (non-zero endpoint anchor,
                // #2297) into the user offset.
                let null_offset = null
                    .design
                    .compose_offset(offset, "smooth likelihood-ratio null model")
                    .map_err(|error| EstimationError::InvalidInput(error.to_string()))?;
                let mut eta = null.design.design.dot(&null.fit.beta);
                eta += &null_offset;
                (w, Some(eta))
            }
            _ => (f64::NAN, None),
        };

        let (p_uncorrected, mut p_bound) = reference.tail_probability_with_bound(statistic_lr);
        let mut p_conditional = reference.conditional_tail_probability(statistic_lr);

        // Magic Bartlett correction: only when the LR statistic is finite, the
        // family has closed-form jets, n is in the resolvable regime, and the
        // factor is computable. Otherwise the uncorrected χ² stands.
        let mut bartlett_factor = 1.0;
        let mut bartlett_factor_conditional = None;
        let mut rho_variation_shift = None;
        let mut statistic_corrected = statistic_lr;
        let mut p_corrected = p_uncorrected;
        let mut correction = SmoothLrCorrection::None;
        if let (Some(eta), true, true) = (
            eta_null.as_ref(),
            statistic_lr.is_finite(),
            n <= LAWLEY_PAIR_MATRIX_MAX_ROWS,
        ) {
            let kappas: Option<Vec<_>> = (0..n)
                .map(|i| {
                    known_scale_expected_jets_with_dispersion(
                        &fitted_likelihood.spec,
                        eta[i],
                        family_disp,
                    )
                    .and_then(|jets| jets.kappas().ok())
                })
                .collect();
            if let Some(kappas) = kappas {
                let fixed_factor = lawley_lr_bartlett_factor(
                    full_design_dense.view(),
                    &kappas,
                    Some(s_lambda.view()),
                    coeff_range.clone(),
                    ref_df,
                );
                if let Ok(c_cond) = fixed_factor
                    && c_cond.is_finite()
                    && c_cond > 0.0
                {
                    let mut c_applied = c_cond;
                    correction = SmoothLrCorrection::LawleyLrFixedLambda;
                    if let Some(cov) = rho_covariance
                        && let Ok(total_shift) = lawley_lr_mean_shift_with_rho_variation(
                            full_design_dense.view(),
                            &kappas,
                            s_lambda.view(),
                            coeff_range.clone(),
                            &rho_penalty_components,
                            cov.view(),
                        )
                    {
                        let mean_w = ref_df + total_shift;
                        if let Some(c_est) =
                            gam_terms::inference::higher_order::bartlett_factor_from_mean(
                                mean_w, ref_df,
                            )
                            && c_est.is_finite()
                            && c_est > 0.0
                        {
                            let conditional_shift = (c_cond - 1.0) * ref_df;
                            c_applied = c_est;
                            bartlett_factor_conditional = Some(c_cond);
                            rho_variation_shift = Some(total_shift - conditional_shift);
                            correction = SmoothLrCorrection::LawleyLrEstimatedLambda;
                        }
                    }
                    bartlett_factor = c_applied;
                    statistic_corrected = statistic_lr / c_applied;
                    // `W* = W/c` and "rescale every spectral weight by `c`" are
                    // the same operation on this reference — the law is exactly
                    // scale-equivariant — so the correction composes with the
                    // scaled reference without a second convention.
                    let (corrected, corrected_bound) =
                        reference.tail_probability_with_bound(statistic_corrected);
                    p_corrected = corrected;
                    p_conditional = reference.conditional_tail_probability(statistic_corrected);
                    p_bound = p_bound.max(corrected_bound);
                }
            }
        }

        // Materiality (#939 deliverable 4): only when a correction was actually
        // applied, flagged when it moves the result by more than the 10%
        // threshold — by the Bartlett factor's distance from one OR the relative
        // p-value shift, whichever is larger (a factor near one can still flip a
        // p-value sitting on the α boundary, and vice versa).
        let material = match correction {
            SmoothLrCorrection::LawleyLrEstimatedLambda
            | SmoothLrCorrection::LawleyLrFixedLambda => {
                let factor_move = (bartlett_factor - 1.0).abs();
                let p_denom = p_uncorrected.max(p_corrected).max(f64::MIN_POSITIVE);
                let p_move = if p_uncorrected.is_finite() && p_corrected.is_finite() {
                    (p_corrected - p_uncorrected).abs() / p_denom
                } else {
                    0.0
                };
                factor_move > SMOOTH_LR_MATERIAL_THRESHOLD || p_move > SMOOTH_LR_MATERIAL_THRESHOLD
            }
            SmoothLrCorrection::None => false,
        };

        out.push(SmoothTermLrInference {
            name: design_term.name.clone(),
            term_idx,
            statistic_lr,
            ref_df,
            ref_df_provenance,
            bartlett_factor,
            bartlett_factor_conditional,
            rho_variation_shift,
            statistic_corrected,
            p_value_uncorrected: p_uncorrected,
            p_value_corrected: p_corrected,
            material,
            correction,
            p_value_conditional: p_conditional,
            p_value_bound: p_bound,
        });
    }
    Ok(out)
}

fn resolved_likelihood_for_fit(
    fit: &UnifiedFitResult,
) -> Result<gam_spec::GlmLikelihoodSpec, EstimationError> {
    let spec = fit.likelihood_family.as_ref().ok_or_else(|| {
        EstimationError::InvalidInput(
            "smooth-term LR inference requires an engine-level GLM likelihood".to_string(),
        )
    })?;
    gam_spec::GlmLikelihoodSpec::try_new(spec.clone(), fit.likelihood_scale.clone())
        .map_err(|error| EstimationError::InvalidInput(error.to_string()))
}

/// The response dispersion `phi` Lawley needs for cumulant scaling. This is
/// deliberately distinct from the coefficient-covariance multiplier used by
/// the WPS trace below: Gamma Lawley uses `1 / shape`, while its PIRLS Hessian
/// already carries `shape` and therefore has covariance multiplier one.
fn lawley_dispersion_for_family(
    likelihood: &gam_spec::GlmLikelihoodSpec,
    fit: &UnifiedFitResult,
) -> Result<f64, EstimationError> {
    let profiled_standard_deviation = matches!(
        likelihood
            .resolved_scale()
            .map_err(|error| EstimationError::InvalidInput(error.to_string()))?,
        gam_spec::ResolvedLikelihoodScale::ProfiledGaussian
    )
    .then_some(fit.standard_deviation);
    gam_solve::estimate::dispersion_from_likelihood(likelihood, profiled_standard_deviation)
        .map(|dispersion| dispersion.phi())
}

/// The reference distribution for the whole-term LR statistic: its own null
/// spectrum `w` when that is recoverable, and the two-moment summary of it when
/// only the moments are.
///
/// The derivation, and why the spectrum rather than two of its moments, is on
/// [`SmoothLrReferenceDf`]. What is worth stating at the code is the ladder, and
/// that each rung is a strictly weaker instrument on the SAME quantity rather
/// than a different claim:
///
/// 1. **The spectrum** ([`lr_penalty_shares`]) — needs `[H⁻¹]_jj`
///    and the term's λ-weighted penalty block. Exact.
/// 2. **Its first two moments** ([`lr_null_spectral_moments`]) — needs only the
///    coefficient-influence block, because with `A = 2F − F²`
///
///    ```text
///    Σ w   = tr A  = 2·tr F − tr F²
///    Σ w²  = tr A² = 4·tr F² − 4·tr F³ + tr F⁴
///    ```
///
///    are traces of powers of one `q × q` block. Reading the weights THEMSELVES
///    off `F_jj` is what rung 1 avoids: `F_jj = H̃⁻¹Ĩ_jj` is not symmetric, so it
///    would need a general eigensolver, while rung 1 reaches the same spectrum
///    through a self-adjoint one.
/// 3. **A scalar EDF** — `χ²_{max(edf, null_dim, 1)}`, the unit-weight shape.
///
/// The lane taken is tagged in the returned provenance, so a consumer can tell
/// an exact reference from a summary of one instead of inferring it from the
/// numbers.
///
/// `term_penalties` are the term's λ-FREE penalty components on the tested
/// block and `term_log_lambda` their fitted `ρ̂_i`, carried separately rather
/// than pre-multiplied. That is not bookkeeping: the selection replay's
/// criterion needs `log|Σ_i λ_i S_i|₊`, and an assembled sum has already lost
/// it whenever the `λ_i` separate (#2644 — see [`SelectionGeometry`]).
/// `log_scale_windows` carries ONE window per component, because the outer
/// search moved each `ρ_i` independently inside its own box.
fn lr_null_reference(
    influence: Option<&Array2<f64>>,
    hessian_inverse: Option<&Array2<f64>>,
    penalty: Option<&Array2<f64>>,
    coeff_range: &Range<usize>,
    edf: f64,
    null_dim: usize,
    statistic_resolution: f64,
    log_scale_windows: &[(f64, f64)],
    term_penalties: &[Array2<f64>],
    term_log_lambda: &[f64],
) -> SmoothLrReferenceDf {
    let from_moments = |mean: f64, second_moment: f64, source| SmoothLrReferenceDf {
        weights: Vec::new(),
        mean,
        second_moment,
        chi_square_df: mean * mean / second_moment,
        scale: second_moment / mean,
        moment_residual: None,
        edf,
        null_dim,
        source,
        selection: None,
        statistic_resolution,
    };
    let unit_weight = || {
        let df = edf.max(null_dim as f64).max(1.0);
        from_moments(df, df, SmoothLrReferenceSource::UnitWeightFallback)
    };
    let influence_moments = lr_null_spectral_moments(influence, coeff_range);

    // Rung 1 — the spectrum itself.
    if let Some(shares) = lr_penalty_shares(hessian_inverse, penalty, coeff_range) {
        let mut weights: Vec<f64> = shares.iter().map(|&p| 1.0 - p * p).collect();
        weights.sort_by(|a, b| b.partial_cmp(a).expect("finite weights"));
        let mean: f64 = weights.iter().sum();
        let second_moment: f64 = weights.iter().map(|w| w * w).sum();
        if mean.is_finite() && mean > 0.0 && second_moment.is_finite() && second_moment > 0.0 {
            // The identity check, measured rather than assumed. Denominated
            // relatively and floored at one so a term shrunk to nothing does not
            // report a huge residual for a difference of `1e-16`.
            let moment_residual = influence_moments.map(|[trace_mean, trace_second]| {
                let first = (mean - trace_mean).abs() / mean.abs().max(1.0);
                let second = (second_moment - trace_second).abs() / second_moment.abs().max(1.0);
                first.max(second)
            });
            return SmoothLrReferenceDf {
                weights,
                mean,
                second_moment,
                chi_square_df: mean * mean / second_moment,
                scale: second_moment / mean,
                moment_residual,
                edf,
                null_dim,
                source: SmoothLrReferenceSource::NullSpectrum,
                // One entry point for both lanes. `generate` whitens the term's
                // λ-free components by the Schur-complemented information,
                // factors them into roots, and dispatches on how many scales the
                // term actually selects. The generalized spectrum it reports is
                // read off that geometry rather than reconstructed as
                // `p_k/(1 − p_k)` from the penalty shares — a share is a number
                // in `[0, 1]`, so a structural zero and a `1e-17` of roundoff are
                // one machine epsilon apart there, and the criterion's
                // log-determinant is the one place that difference is worth
                // `log(1 + 1e17)`.
                selection: lr_schur_information(hessian_inverse, penalty, coeff_range).and_then(
                    |information| {
                        SmoothLrSelectionReplay::generate(
                            &information,
                            term_penalties,
                            term_log_lambda,
                            log_scale_windows,
                        )
                    },
                ),
                statistic_resolution,
            };
        }
    }

    // Rung 2 — two moments of it, off the influence block.
    let Some([mean, second_moment]) = influence_moments else {
        return unit_weight();
    };
    if !(mean.is_finite() && mean > 0.0 && second_moment.is_finite() && second_moment > 0.0) {
        return unit_weight();
    }
    from_moments(
        mean,
        second_moment,
        SmoothLrReferenceSource::SpectralMomentMatch,
    )
}

/// The Schur-complemented information on the tested block,
/// `Ĩ_jj = ([H⁻¹]_jj)⁻¹ − S_jj`.
///
/// This is the object the whole derivation is stated against, and it is
/// available without ever forming a Schur complement: `[H⁻¹]_jj` IS
/// `(Ĩ_jj + S_jj)⁻¹` (the block of an inverse is the inverse of the Schur
/// complement of the OTHER block), so one inversion of a `q × q` symmetric
/// matrix and one subtraction recover it. The retained block's own penalties are
/// already inside it, which is correct: the null model keeps them.
fn lr_schur_information(
    hessian_inverse: Option<&Array2<f64>>,
    penalty: Option<&Array2<f64>>,
    coeff_range: &Range<usize>,
) -> Option<Array2<f64>> {
    let (h_inv, s_lambda) = (hessian_inverse?, penalty?);
    let (start, end) = (coeff_range.start, coeff_range.end);
    if start >= end
        || end > h_inv.nrows()
        || end > h_inv.ncols()
        || end > s_lambda.nrows()
        || end > s_lambda.ncols()
    {
        return None;
    }
    let block = h_inv.slice(s![start..end, start..end]).to_owned();
    let mut symmetric = block.clone();
    let q = symmetric.nrows();
    for row in 0..q {
        for column in 0..row {
            let mean = 0.5 * (symmetric[[row, column]] + symmetric[[column, row]]);
            symmetric[[row, column]] = mean;
            symmetric[[column, row]] = mean;
        }
    }
    let (values, vectors) =
        gam_linalg::faer_ndarray::strict_symmetric_eigh(&symmetric, faer::Side::Lower).ok()?;
    let largest = values.iter().copied().fold(0.0_f64, f64::max);
    if !(largest > 0.0) {
        return None;
    }
    // `[H⁻¹]_jj` is positive definite; a direction at the reciprocal-condition
    // floor carries no identified information and is dropped rather than
    // inverted into a huge eigenvalue that would dominate the whitening.
    let floor = largest * 1e-12;
    let mut inverse = Array2::<f64>::zeros((q, q));
    for (index, &value) in values.iter().enumerate() {
        if value <= floor {
            continue;
        }
        for row in 0..q {
            for column in 0..q {
                inverse[[row, column]] +=
                    vectors[[row, index]] * vectors[[column, index]] / value;
            }
        }
    }
    let mut information = inverse - s_lambda.slice(s![start..end, start..end]);
    for row in 0..q {
        for column in 0..row {
            let mean = 0.5 * (information[[row, column]] + information[[column, row]]);
            information[[row, column]] = mean;
            information[[column, row]] = mean;
        }
    }
    if information.iter().any(|value| !value.is_finite()) {
        return None;
    }
    Some(information)
}

/// The term's PENALTY SHARES `p = eig([H⁻¹]_jj · S_jj) ∈ [0, 1]`, sorted
/// ascending — the one object every reference on this path is a function of.
///
/// The null weights are `w_j = 1 − p_j²` (see below), and the generalized
/// eigenvalues that drive the selection replay are `ν_j = p_j/(1 − p_j)`, so a
/// single self-adjoint decomposition yields both.
///
/// # Why this is the same spectrum as `eig(2·F_jj − F_jj²)`
///
/// The penalty is block-diagonal by term, so `S_kj = 0` for `k ≠ j` and the
/// tested block of the GLOBAL shrinkage map factors exactly:
///
/// ```text
/// (I − F)_jj = [H⁻¹S]_jj = Σ_k [H⁻¹]_jk S_kj = [H⁻¹]_jj S_jj  =:  P.
/// ```
///
/// Therefore `F_jj = I − P` and `2F_jj − F_jj² = I − (I − F_jj)² = I − P²`, so
/// `w = 1 − eig(P)²` with no approximation anywhere — the same object the trace
/// identities in [`lr_null_spectral_moments`] summarise, arrived at without
/// forming a non-symmetric matrix.
///
/// # Why it is reachable with a self-adjoint eigensolver
///
/// `P = B S` with `B = [H⁻¹]_jj` symmetric PSD (a principal submatrix of the
/// inverse of a PD Hessian) and `S = S_jj` symmetric PSD. A product of two
/// symmetric PSD matrices is not symmetric, but it is similar to one:
///
/// ```text
/// B^{-1/2} (B S) B^{1/2} = B^{1/2} S B^{1/2},
/// ```
///
/// which is symmetric PSD and is what this computes — via `B = UΛUᵀ` and
/// `B^{1/2} = UΛ^{1/2}Uᵀ` rather than a Cholesky, so a `B` that is singular in
/// some direction (an exactly-unpenalized fit, a rank-deficient block) is a
/// zero eigenvalue rather than a factorization failure. The eigenvalues are real
/// and lie in `[0, 1]` because `F_jj = (Ĩ_jj + S_jj)⁻¹Ĩ_jj` has eigenvalues
/// `c/(c + s)`; they are clamped to that interval against roundoff, and the
/// clamp is the ONLY place a value is altered.
///
/// Returns `None` when either matrix is absent, the block does not fit inside
/// them, or the self-adjoint decomposition refuses — the caller then drops to
/// the two-moment rung rather than scoring against a spectrum it could not
/// compute.
fn lr_penalty_shares(
    hessian_inverse: Option<&Array2<f64>>,
    penalty: Option<&Array2<f64>>,
    coeff_range: &Range<usize>,
) -> Option<Vec<f64>> {
    let (h_inv, s_lambda) = (hessian_inverse?, penalty?);
    let (start, end) = (coeff_range.start, coeff_range.end);
    if start >= end
        || end > h_inv.nrows()
        || end > h_inv.ncols()
        || end > s_lambda.nrows()
        || end > s_lambda.ncols()
    {
        return None;
    }
    // Both blocks are symmetric as mathematical objects; the halves of an
    // assembled Gram/inverse differ only by summation order. Symmetrize
    // explicitly so the self-adjoint entry point receives the matrix it is being
    // asked about rather than one triangle's rounding of it.
    let symmetrize = |m: ndarray::ArrayView2<'_, f64>| -> Array2<f64> {
        let mut out = m.to_owned();
        let q = out.nrows();
        for row in 0..q {
            for col in 0..row {
                let mean = 0.5 * (out[[row, col]] + out[[col, row]]);
                out[[row, col]] = mean;
                out[[col, row]] = mean;
            }
        }
        out
    };
    let b = symmetrize(h_inv.slice(s![start..end, start..end]));
    let s = symmetrize(s_lambda.slice(s![start..end, start..end]));
    if b.iter().chain(s.iter()).any(|value| !value.is_finite()) {
        return None;
    }

    let (b_eigenvalues, b_vectors) =
        gam_linalg::faer_ndarray::strict_symmetric_eigh(&b, faer::Side::Lower).ok()?;
    // `B^{1/2} = U Λ^{1/2} Uᵀ`. A tiny negative eigenvalue is roundoff on a PSD
    // matrix, so its square root is zero rather than an error.
    let mut root_scaled = b_vectors.clone();
    for (mut column, &eigenvalue) in root_scaled.columns_mut().into_iter().zip(b_eigenvalues.iter())
    {
        let root = eigenvalue.max(0.0).sqrt();
        column.mapv_inplace(|value| value * root);
    }
    let b_root = root_scaled.dot(&b_vectors.t());
    let similar = symmetrize(b_root.dot(&s).dot(&b_root).view());
    let (shrinkage, _) =
        gam_linalg::faer_ndarray::strict_symmetric_eigh(&similar, faer::Side::Lower).ok()?;

    let mut spectrum: Vec<f64> = shrinkage.iter().map(|&p| p.clamp(0.0, 1.0)).collect();
    if spectrum.iter().any(|p| !p.is_finite()) {
        return None;
    }
    spectrum.sort_by(|a, b| a.partial_cmp(b).expect("finite shrinkage"));
    Some(spectrum)
}

/// `[tr A, tr A²]` for `A = 2·F_jj − F_jj²` on the tested coefficient block.
///
/// Returns `None` when the influence matrix is absent, the block is outside it,
/// or either trace is non-finite — the caller then falls back to the unit-weight
/// shape rather than scoring against a spectrum it could not compute.
fn lr_null_spectral_moments(
    influence: Option<&Array2<f64>>,
    coeff_range: &Range<usize>,
) -> Option<[f64; 2]> {
    let f = influence?;
    let (start, end) = (coeff_range.start, coeff_range.end);
    if start >= end || end > f.nrows() || end > f.ncols() {
        return None;
    }
    let block = f.slice(s![start..end, start..end]).to_owned();
    let squared = block.dot(&block);
    let cubed = squared.dot(&block);
    let quartic = squared.dot(&squared);
    let trace = |m: &Array2<f64>| (0..m.nrows()).map(|i| m[[i, i]]).sum::<f64>();
    let (t1, t2, t3, t4) = (
        trace(&block),
        trace(&squared),
        trace(&cubed),
        trace(&quartic),
    );
    let mean = 2.0 * t1 - t2;
    let second_moment = 4.0 * t2 - 4.0 * t3 + t4;
    (mean.is_finite() && second_moment.is_finite()).then_some([mean, second_moment])
}

#[cfg(test)]
mod lr_null_reference_tests {
    use super::{
        SmoothLrReferenceSource, lr_null_reference, lr_null_spectral_moments,
        lr_penalty_shares,
    };
    use ndarray::Array2;

    /// No selection window: these unit tests are about the CONDITIONAL law, so
    /// they hold `λ` fixed and the replay is inert. The replay's own behaviour
    /// is pinned separately.
    const WINDOW: &[(f64, f64)] = &[];

    /// `M⁻¹` for a symmetric PD `M`, through the same self-adjoint entry point
    /// the production path uses. The tests need an inverse only to BUILD the two
    /// inputs (`H⁻¹` and `F = H⁻¹(H − S)`) from one `H`; nothing under test reads
    /// it.
    fn symmetric_inverse(matrix: &Array2<f64>) -> Array2<f64> {
        let (values, vectors) =
            gam_linalg::faer_ndarray::strict_symmetric_eigh(matrix, faer::Side::Lower)
                .expect("symmetric PD inverse");
        let mut scaled = vectors.clone();
        for (mut column, &value) in scaled.columns_mut().into_iter().zip(values.iter()) {
            column.mapv_inplace(|entry| entry / value);
        }
        scaled.dot(&vectors.t())
    }

    /// A diagonal influence block has `F_jj` eigenvalues on the diagonal, so the
    /// spectrum is `2f − f²` term by term and both moments are hand-computable.
    /// This is the identity the whole reference rests on; it is checked against
    /// the definition rather than against another implementation of itself.
    #[test]
    fn the_spectral_moments_are_the_weights_of_the_null_law() {
        let f_diag = [0.9_f64, 0.5, 0.2, 0.05];
        let mut influence = Array2::<f64>::zeros((6, 6));
        // Deliberately offset: the block is columns 2..6, and rows/columns
        // outside it carry values that must not leak into either trace.
        influence[[0, 0]] = 7.0;
        influence[[1, 1]] = -3.0;
        influence[[0, 3]] = 11.0;
        influence[[5, 1]] = -2.0;
        for (i, &f) in f_diag.iter().enumerate() {
            influence[[2 + i, 2 + i]] = f;
        }
        let [mean, second] =
            lr_null_spectral_moments(Some(&influence), &(2..6)).expect("moments available");
        let weights: Vec<f64> = f_diag.iter().map(|f| 2.0 * f - f * f).collect();
        let want_mean: f64 = weights.iter().sum();
        let want_second: f64 = weights.iter().map(|w| w * w).sum();
        assert!(
            (mean - want_mean).abs() < 1e-12 && (second - want_second).abs() < 1e-12,
            "moments ({mean}, {second}) vs weights {weights:?} -> ({want_mean}, {want_second})"
        );
    }

    /// THE IDENTITY THE EXACT LANE RESTS ON, on a design where every block is
    /// coupled to every other: the spectrum read off `[H⁻¹]_jj` and `S_jj` has
    /// the same two moments as the spectrum read off the influence block, which
    /// are computed by completely different arithmetic (two self-adjoint
    /// decompositions versus four traces of powers of a non-symmetric matrix).
    ///
    /// `(I − F)_jj = [H⁻¹]_jj S_jj` is only true because the penalty is
    /// block-diagonal by term, so the fixture puts a SEPARATE penalty on the
    /// retained block as well: the identity must survive other terms being
    /// penalized (that is the difference between the Schur complement of the
    /// penalized retained block and of the unpenalized one), and it must fail if
    /// anyone ever lets a penalty couple two terms.
    #[test]
    fn the_penalty_spectrum_and_the_influence_moments_are_the_same_object() {
        let (retained, tested) = (3usize, 5usize);
        let p = retained + tested;
        // A dense SPD Gram with real cross-block coupling.
        let mut gram = Array2::<f64>::zeros((p, p));
        for row in 0..p {
            for col in 0..p {
                gram[[row, col]] = 1.0 / (1.0 + (row as f64 - col as f64).abs())
                    + if row == col { 0.75 } else { 0.0 };
            }
        }
        for lambda in [0.0_f64, 1e-3, 1.0, 25.0, 1e4, 1e7] {
            // Block-diagonal penalty: a second-difference block on the tested
            // term and an unrelated ridge on the retained one.
            let mut penalty = Array2::<f64>::zeros((p, p));
            for row in 0..retained {
                penalty[[row, row]] = 0.3;
            }
            for row in 0..tested.saturating_sub(2) {
                for (offset_a, coefficient_a) in [(0usize, 1.0_f64), (1, -2.0), (2, 1.0)] {
                    for (offset_b, coefficient_b) in [(0usize, 1.0_f64), (1, -2.0), (2, 1.0)] {
                        penalty[[retained + row + offset_a, retained + row + offset_b]] +=
                            lambda * coefficient_a * coefficient_b;
                    }
                }
            }
            let hessian = &gram + &penalty;
            let hessian_inverse = symmetric_inverse(&hessian);
            let influence = hessian_inverse.dot(&gram);

            let weights =
                lr_penalty_shares(Some(&hessian_inverse), Some(&penalty), &(retained..p))
                    .map(|shares| shares.iter().map(|&q| 1.0 - q * q).collect::<Vec<f64>>())
                    .expect("spectrum available");
            let [mean, second] = lr_null_spectral_moments(Some(&influence), &(retained..p))
                .expect("moments available");
            let spectrum_mean: f64 = weights.iter().sum();
            let spectrum_second: f64 = weights.iter().map(|w| w * w).sum();
            // `1e-7` relative, not roundoff: at `λ = 1e7` the INFLUENCE route
            // is what loses the digits — `2·trF − trF²` differences two nearly
            // equal quantities while `trF → 0` — and it comes in at `2.5e-9`
            // relative there against `<1e-15` at every smaller `λ`. That
            // asymmetry is one of the reasons the penalty route is the primary
            // one; the bar is set where the WEAKER of the two routes lives.
            assert!(
                (spectrum_mean - mean).abs() < 1e-7 * mean.abs().max(1.0)
                    && (spectrum_second - second).abs() < 1e-7 * second.abs().max(1.0),
                "lambda={lambda}: spectrum moments ({spectrum_mean}, {spectrum_second}) \
                 disagree with influence-trace moments ({mean}, {second})"
            );
            assert!(
                weights.iter().all(|w| (0.0..=1.0).contains(w)),
                "lambda={lambda}: weights escaped [0,1]: {weights:?}"
            );
            assert!(
                weights.windows(2).all(|pair| pair[0] >= pair[1]),
                "lambda={lambda}: weights are not sorted descending: {weights:?}"
            );
        }
    }

    /// The identity that makes this a strict generalization rather than a
    /// replacement: an UNPENALIZED tested block has `F_jj = I`, every weight is
    /// one, and the reference must be the textbook `χ²_q` — exactly, not
    /// approximately, on the EXACT lane as well as on the moment lane.
    #[test]
    fn an_unpenalized_block_is_exactly_the_classical_chi_square() {
        let q = 5;
        let identity = Array2::<f64>::eye(q);
        let zero_penalty = Array2::<f64>::zeros((q, q));
        let reference = lr_null_reference(
            Some(&identity),
            Some(&identity),
            Some(&zero_penalty),
            &(0..q),
            0.0,
            q,
            0.0,
            WINDOW,
            &[],
            &[],
        );
        assert_eq!(reference.source, SmoothLrReferenceSource::NullSpectrum);
        assert_eq!(reference.weights, vec![1.0; q]);
        assert_eq!(reference.chi_square_df, q as f64);
        assert_eq!(reference.scale, 1.0);
        assert_eq!(reference.mean, q as f64);
        // The exact lane resolves equal weights through its own closed form, so
        // this is the classical value bit for bit rather than to a tolerance.
        for statistic in [0.5_f64, 3.0, 11.07, 40.0] {
            assert_eq!(
                reference.tail_probability(statistic),
                gam_math::probability::chi_square_sf(statistic, q as f64)
            );
        }
    }

    /// Equal shrinkage is the other exact case: `p_j ≡ p` gives `w_j ≡ 1 − p²`,
    /// so the law is a SCALED `χ²_q` and BOTH lanes reproduce it exactly. This is
    /// what makes the scale a real parameter rather than a fudge, and it is the
    /// case in which the two lanes must not be distinguishable.
    #[test]
    fn equal_shrinkage_is_the_exact_scaled_chi_square() {
        let q = 6;
        let f = 0.4_f64;
        let w = 2.0 * f - f * f;
        let influence = Array2::<f64>::eye(q) * f;
        // `P = B·S = (1 − f)·I` on the block: B = I, S = (1 − f)·I.
        let hessian_inverse = Array2::<f64>::eye(q);
        let penalty = Array2::<f64>::eye(q) * (1.0 - f);
        for reference in [
            lr_null_reference(
                Some(&influence),
                Some(&hessian_inverse),
                Some(&penalty),
                &(0..q),
                f * q as f64,
                0,
                0.0,
                WINDOW,
                &[],
                &[],
            ),
            lr_null_reference(Some(&influence), None, None, &(0..q), f * q as f64, 0, 0.0, WINDOW, &[], &[]),
        ] {
            assert!((reference.chi_square_df - q as f64).abs() < 1e-12);
            assert!((reference.scale - w).abs() < 1e-12);
            for statistic in [0.2_f64, 2.0, 9.0] {
                let want = gam_math::probability::chi_square_sf(statistic / w, q as f64);
                assert!(
                    (reference.tail_probability(statistic) - want).abs() < 1e-12,
                    "{:?}: {} vs {want}",
                    reference.source,
                    reference.tail_probability(statistic)
                );
            }
        }
    }

    /// The reason the exact lane exists, pinned as a measurement rather than a
    /// preference: on a spread spectrum the two-moment summary is
    /// ANTI-conservative, one-signed, and worse the deeper the tail — so a fit
    /// that lands on the moment lane is not merely less precise, it rejects too
    /// often, and the gap grows exactly where a p-value is being used to claim
    /// something.
    ///
    /// The two references here are built from the SAME spectrum, so nothing but
    /// the shape of the reference differs between the arms.
    #[test]
    fn the_two_moment_summary_is_anti_conservative_in_the_tail() {
        // A shrunk smooth: one unpenalized direction and a geometric tail.
        let shrinkage = [0.0_f64, 0.55, 0.85, 0.96, 0.995];
        let q = shrinkage.len();
        let hessian_inverse = Array2::<f64>::eye(q);
        let mut penalty = Array2::<f64>::zeros((q, q));
        let mut influence = Array2::<f64>::zeros((q, q));
        for (index, &p) in shrinkage.iter().enumerate() {
            penalty[[index, index]] = p;
            influence[[index, index]] = 1.0 - p;
        }
        let exact = lr_null_reference(
            Some(&influence),
            Some(&hessian_inverse),
            Some(&penalty),
            &(0..q),
            0.0,
            1,
            0.0,
            WINDOW,
            &[],
            &[],
        );
        let summary = lr_null_reference(Some(&influence), None, None, &(0..q), 0.0, 1, 0.0, WINDOW, &[], &[]);
        assert_eq!(exact.source, SmoothLrReferenceSource::NullSpectrum);
        assert_eq!(summary.source, SmoothLrReferenceSource::SpectralMomentMatch);
        // Same spectrum, so the two moments agree to roundoff; only the shape
        // read off them differs.
        assert!((exact.mean - summary.mean).abs() < 1e-12);
        assert!((exact.second_moment - summary.second_moment).abs() < 1e-12);

        let mut previous_ratio = 0.99_f64;
        for &alpha in &[5e-2_f64, 1e-2, 1e-3, 1e-4] {
            // The statistic at which the SUMMARY reports exactly `alpha`, found
            // by bisecting its own (monotone) tail rather than by a quantile
            // routine, so the two arms are compared through one interface.
            let (mut low, mut high) = (0.0_f64, 1.0_f64);
            while summary.tail_probability(high) > alpha {
                high *= 2.0;
                assert!(high < 1e6, "alpha={alpha}: the summary tail never fell below it");
            }
            for _ in 0..200 {
                let middle = 0.5 * (low + high);
                if summary.tail_probability(middle) > alpha {
                    low = middle;
                } else {
                    high = middle;
                }
            }
            let statistic = 0.5 * (low + high);
            let exact_tail = exact.tail_probability(statistic);
            let ratio = exact_tail / alpha;
            // At `α = 0.05` the two are within a percent of each other — the
            // surrogate's error is a TAIL error, and this is where it is
            // smallest. The claim is the SHAPE of the error, so the bar here is
            // that it has not gone the other way, and the growth assertion
            // below is what carries it.
            assert!(
                ratio > 0.99,
                "alpha={alpha}: the summary reports a materially LARGER tail than the \
                 law (exact {exact_tail} at its own alpha); the error is supposed to be \
                 one-signed the other way"
            );
            assert!(
                ratio >= previous_ratio - 1e-9,
                "alpha={alpha}: the summary's error must not shrink as the tail deepens \
                 ({ratio} after {previous_ratio})"
            );
            previous_ratio = ratio;
        }
        assert!(
            previous_ratio > 1.3,
            "at alpha=1e-4 the summary should be materially anti-conservative on this \
             spectrum; measured {previous_ratio}x"
        );
    }

    /// The floors #1766 needed against a collapsing `χ²_d` are structural here:
    /// as the term shrinks, `W` and the reference scale collapse TOGETHER, so
    /// the tail probability of a statistic proportional to the weights stays
    /// put instead of running to zero. Asserted across six orders of shrinkage.
    #[test]
    fn a_collapsing_term_does_not_degenerate_the_reference() {
        let q = 5;
        let mut previous: Option<f64> = None;
        for exponent in 0..7 {
            let f = 10f64.powi(-exponent);
            let influence = Array2::<f64>::eye(q) * f;
            let hessian_inverse = Array2::<f64>::eye(q);
            let penalty = Array2::<f64>::eye(q) * (1.0 - f);
            let reference = lr_null_reference(
                Some(&influence),
                Some(&hessian_inverse),
                Some(&penalty),
                &(0..q),
                f * q as f64,
                0,
                0.0,
                WINDOW,
                &[],
                &[],
            );
            // A statistic drawn at the reference's own mean.
            let tail = reference.tail_probability(reference.mean);
            assert!(
                tail > 0.3 && tail < 0.6,
                "f=1e-{exponent}: tail at the mean is {tail}, mean={}",
                reference.mean
            );
            if let Some(prev) = previous {
                assert!(
                    (tail - prev).abs() < 1e-9,
                    "f=1e-{exponent}: the tail at the mean moved {prev} -> {tail} under pure rescaling"
                );
            }
            previous = Some(tail);
        }
    }

    /// Each rung of the ladder degrades to the next and SAYS so. A consumer that
    /// cannot tell an exact reference from a summary of one, or a summary from a
    /// scalar-EDF fallback, cannot reason about the number it was handed.
    #[test]
    fn each_missing_input_degrades_exactly_one_rung_and_visibly() {
        let q = 4;
        let influence = Array2::<f64>::eye(q) * 0.5;
        let hessian_inverse = Array2::<f64>::eye(q);
        let penalty = Array2::<f64>::eye(q) * 0.5;

        // Everything present: the exact lane, carrying weights.
        let exact = lr_null_reference(
            Some(&influence),
            Some(&hessian_inverse),
            Some(&penalty),
            &(0..q),
            2.0,
            1,
            0.0,
            WINDOW,
            &[],
            &[],
        );
        assert_eq!(exact.source, SmoothLrReferenceSource::NullSpectrum);
        assert_eq!(exact.weights.len(), q);

        // No `H⁻¹` (or no penalty): the moments off `F`, and NO weights — which
        // is exactly the condition `tail_probability` switches on.
        for degraded in [
            lr_null_reference(Some(&influence), None, Some(&penalty), &(0..q), 2.0, 1, 0.0, WINDOW, &[], &[]),
            lr_null_reference(
                Some(&influence),
                Some(&hessian_inverse),
                None,
                &(0..q),
                2.0,
                1,
                0.0,
                WINDOW,
                &[],
                &[],
            ),
        ] {
            assert_eq!(degraded.source, SmoothLrReferenceSource::SpectralMomentMatch);
            assert!(degraded.weights.is_empty());
            assert!((degraded.mean - exact.mean).abs() < 1e-12);
        }

        // Nothing at all: the unit-weight shape with its `max(edf, null_dim, 1)`.
        let fallback = lr_null_reference(None, None, None, &(0..q), 2.5, 1, 0.0, WINDOW, &[], &[]);
        assert_eq!(fallback.source, SmoothLrReferenceSource::UnitWeightFallback);
        assert!(fallback.weights.is_empty());
        assert_eq!(fallback.chi_square_df, 2.5);
        assert_eq!(fallback.scale, 1.0);
        // The `max(edf, null_dim, 1)` shape is retained only on this lane.
        assert_eq!(
            lr_null_reference(None, None, None, &(0..4), 0.01, 3, 0.0, WINDOW, &[], &[]).chi_square_df,
            3.0
        );
        assert_eq!(
            lr_null_reference(None, None, None, &(0..4), 0.01, 0, 0.0, WINDOW, &[], &[]).chi_square_df,
            1.0
        );
    }
}

#[cfg(test)]
mod selection_replay_tests {
    use super::{
        SMOOTH_LR_SELECTION_DRAWS, SMOOTH_LR_SELECTION_MAX_SCALES, SelectionGeometry,
        SmoothLrSelectionReplay,
    };
    use ndarray::Array2;

    /// A shrunk-smooth generalized spectrum: one direction the data can still
    /// see and a geometric tail the penalty has taken.
    fn spectrum() -> Vec<f64> {
        vec![0.3_f64, 1.0, 4.0, 20.0, 120.0, 900.0]
    }

    /// The geometry a bare generalized spectrum corresponds to: unit
    /// information, one penalty whose whitened form is `diag(ν)`, fitted at
    /// `λ̂ = 1`. `eig(Ĩ⁻¹S) = ν` exactly, so this is the identity map from the
    /// spectrum these tests are written in terms of onto the object the replay
    /// is built from.
    fn diagonal(spectrum: &[f64]) -> SelectionGeometry {
        let q = spectrum.len();
        let mut penalty = Array2::<f64>::zeros((q, q));
        for (index, &value) in spectrum.iter().enumerate() {
            penalty[[index, index]] = value;
        }
        SelectionGeometry::whiten(&Array2::eye(q), std::slice::from_ref(&penalty), &[0.0])
            .expect("diagonal geometry")
    }

    fn replay_from(spectrum: &[f64], window: (f64, f64), draws: usize) -> SmoothLrSelectionReplay {
        SmoothLrSelectionReplay::from_geometry(&diagonal(spectrum), &[window], draws, draws)
            .expect("replay")
    }

    /// The replay is a p-value input, so it must not depend on a thread, a
    /// machine or a run (#1017). It is a counter-based stratified stream, and
    /// this pins that: two independent generations are bit-identical.
    #[test]
    fn the_replay_is_bit_identical_across_generations() {
        let first = replay_from(&spectrum(), (-8.0, 8.0), SMOOTH_LR_SELECTION_DRAWS);
        let second = replay_from(&spectrum(), (-8.0, 8.0), SMOOTH_LR_SELECTION_DRAWS);
        assert_eq!(first, second);
        for statistic in [0.05_f64, 0.5, 1.5, 4.0] {
            assert_eq!(first.tail_shift(statistic), second.tail_shift(statistic));
        }
    }

    /// Every coordinate draws from the SAME stratum midpoints, in a different
    /// order — a Latin hypercube. Sorting one coordinate's draws must reproduce
    /// the strata exactly, or the stratification is not what the doc claims and
    /// the variance argument behind the draw budget does not hold.
    #[test]
    fn every_coordinate_is_a_permutation_of_the_same_strata() {
        let replay = replay_from(&[1.0], (-6.0, 6.0), SMOOTH_LR_SELECTION_DRAWS);
        let mut conditional = replay.conditional_sample.clone();
        conditional.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
        // The conditional weight at `t = 1` for `ν = 1` is `2·½ − ¼ = 0.75`.
        let mut expected: Vec<f64> = (0..SMOOTH_LR_SELECTION_DRAWS)
            .map(|bin| {
                let uniform = (bin as f64 + 0.5) / SMOOTH_LR_SELECTION_DRAWS as f64;
                let normal =
                    gam_math::probability::standard_normal_quantile(uniform).expect("interior");
                0.75 * normal * normal
            })
            .collect();
        expected.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
        for (got, want) in conditional.iter().zip(expected.iter()) {
            assert!(
                (got - want).abs() <= 1e-12 * want.abs().max(1.0),
                "the conditional sample is not the strata: {got} vs {want}"
            );
        }
    }

    /// The replay is not inert, and what it does is DISPERSE the statistic
    /// rather than shift it.
    ///
    /// I assumed twice over that selecting `λ` could only inflate `W`, and the
    /// measurement says otherwise both times: on this spectrum `E[W(λ̂)] = 1.13`
    /// against `E[W(1)] = 2.17`, and at `W = E[W(1)]` the replay *shrinks* the
    /// upper tail by `0.19`. Under a fresh null draw the criterion usually
    /// prefers MORE shrinkage than the fitted point, so the mean falls — while
    /// a draw that happens to look wiggly buys itself a smaller `λ` and a much
    /// larger `W`, so the spread rises. Dispersion is the invariant; the sign of
    /// the tail shift is a property of where the fitted `λ̂` sits relative to the
    /// null's typical choice, i.e. of the fit, not of the construction. It is
    /// asserted on real fits in the integration suite, not here.
    #[test]
    fn selection_disperses_the_statistic_and_is_not_inert() {
        let replay = replay_from(&spectrum(), (-10.0, 10.0), SMOOTH_LR_SELECTION_DRAWS);
        let draws = replay.selection_sample.len() as f64;
        let mean = |sample: &[f64]| sample.iter().sum::<f64>() / draws;
        let variance = |sample: &[f64]| {
            let m = mean(sample);
            sample.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / draws
        };
        assert!(
            variance(&replay.selection_sample) > variance(&replay.conditional_sample),
            "selection did not disperse the statistic: {} vs {}",
            variance(&replay.selection_sample),
            variance(&replay.conditional_sample)
        );
        let conditional_mean = mean(&replay.conditional_sample);
        let mut any_move = false;
        for multiple in [0.25_f64, 1.0, 4.0, 16.0] {
            let statistic = multiple * conditional_mean;
            let (shift, standard_error) = replay.tail_shift(statistic);
            assert!(
                shift.is_finite() && (-1.0..=1.0).contains(&shift) && standard_error >= 0.0,
                "at W={statistic} the shift {shift} is not a probability difference"
            );
            if shift.abs() > 4.0 * standard_error {
                any_move = true;
            }
        }
        assert!(
            any_move,
            "the replay never moved the tail by more than its own noise — it is inert"
        );
        // Dispersion at the far end, stated where `N` draws can still resolve
        // it: the most extreme value a selected scale reaches has to exceed the
        // most extreme the fitted scale reaches, on the SAME draws.
        let extreme = |sample: &[f64]| sample.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        assert!(
            extreme(&replay.selection_sample) > extreme(&replay.conditional_sample),
            "the selected law never reached past the conditional one's most extreme \
             draw ({} vs {}), so it is not dispersing the upper tail at all",
            extreme(&replay.selection_sample),
            extreme(&replay.conditional_sample)
        );
    }

    /// The published standard error has to be HONEST, because it is what the
    /// report's accuracy bound is built from. Quadrupling the draws must move
    /// the shift by no more than the two reported standard errors allow — a
    /// self-consistency check that fails if the error is understated, which is
    /// the failure mode that matters (an overstated one is merely pessimistic).
    #[test]
    fn the_published_standard_error_covers_a_four_fold_draw_increase() {
        let coarse = replay_from(&spectrum(), (-10.0, 10.0), 4096);
        let fine = replay_from(&spectrum(), (-10.0, 10.0), 16384);
        let conditional_mean = coarse.conditional_sample.iter().sum::<f64>()
            / coarse.conditional_sample.len() as f64;
        for multiple in [0.5_f64, 1.0, 2.0, 4.0, 8.0] {
            let statistic = multiple * conditional_mean;
            let (coarse_shift, coarse_error) = coarse.tail_shift(statistic);
            let (fine_shift, fine_error) = fine.tail_shift(statistic);
            let allowance = 3.0 * (coarse_error + fine_error) + 1e-12;
            assert!(
                (coarse_shift - fine_shift).abs() <= allowance,
                "at W={statistic} the shift moved {coarse_shift} -> {fine_shift} under a \
                 four-fold draw increase, outside the {allowance} the two published \
                 standard errors ({coarse_error:.3e}, {fine_error:.3e}) allow"
            );
        }
        // And the finer run's own error must actually be smaller — a standard
        // error that does not fall with the budget is not a standard error.
        let statistic = 2.0 * conditional_mean;
        assert!(
            fine.tail_shift(statistic).1 < coarse.tail_shift(statistic).1,
            "the reported standard error did not fall when the draws quadrupled"
        );
    }

    /// The multi-scale replay must agree with the one-dimensional one when
    /// there is only one scale to select — that is the seam between the two
    /// paths, and a seam nobody checks is a seam that drifts.
    ///
    /// `generate_multiscale` refuses a single penalty by construction (there is
    /// nothing it can do that the diagonal path cannot do faster), so the
    /// agreement is checked by handing it the SAME penalty split in two halves:
    /// `S = ½S + ½S` selects two scales whose sum is the one scale, so the
    /// two-dimensional grid contains the one-dimensional family along its
    /// diagonal and the two references must land on the same law.
    #[test]
    fn a_split_penalty_reproduces_the_single_scale_law() {
        let q = 4;
        let information = Array2::<f64>::eye(q);
        let mut penalty = Array2::<f64>::zeros((q, q));
        for index in 0..q {
            penalty[[index, index]] = 0.5 * (index as f64 + 1.0);
        }
        let half = penalty.clone() * 0.5;
        let split_geometry =
            SelectionGeometry::whiten(&information, &[half.clone(), half], &[0.0, 0.0])
                .expect("split geometry");
        let split = SmoothLrSelectionReplay::generate_multiscale(
            &split_geometry,
            &[(-6.0, 6.0), (-6.0, 6.0)],
            2048,
        )
        .expect("multiscale replay");
        // With `information = I` the generalized eigenvalues ARE the penalty's
        // diagonal, so the one-dimensional replay is directly constructible.
        let single = replay_from(
            &(0..q)
                .map(|index| 0.5 * (index as f64 + 1.0))
                .collect::<Vec<f64>>(),
            (-6.0, 6.0),
            2048,
        );
        let mean = |sample: &[f64]| sample.iter().sum::<f64>() / sample.len() as f64;
        let split_mean = mean(&split.conditional_sample);
        let single_mean = mean(&single.conditional_sample);
        assert!(
            (split_mean - single_mean).abs() <= 0.05 * single_mean.abs().max(1.0),
            "the two paths disagree on the CONDITIONAL law: {split_mean} vs {single_mean}"
        );
        let split_selected = mean(&split.selection_sample);
        let single_selected = mean(&single.selection_sample);
        assert!(
            (split_selected - single_selected).abs() <= 0.25 * single_selected.abs().max(1.0),
            "the two paths disagree on the SELECTED law by more than the coarser \
             grid can explain: {split_selected} vs {single_selected}"
        );
        // Both lanes publish the same generalized spectrum, because it is a
        // property of the term and not of which grid was affordable.
        for (from_split, from_single) in split
            .generalized
            .iter()
            .zip(single.generalized.iter())
        {
            assert!(
                (from_split - from_single).abs() <= 1e-9 * from_single.abs().max(1.0),
                "the two lanes publish different generalized spectra: {:?} vs {:?}",
                split.generalized,
                single.generalized
            );
        }
    }

    /// The multi-scale path is for terms that actually select several scales,
    /// and it says so rather than pretending on the ones it cannot serve.
    #[test]
    fn the_multiscale_path_declines_what_it_cannot_serve() {
        let information = Array2::<f64>::eye(3);
        let penalty = Array2::<f64>::eye(3);
        // One scale: the diagonal path is strictly better, so this declines.
        let single = SelectionGeometry::whiten(
            &information,
            std::slice::from_ref(&penalty),
            &[0.0],
        )
        .expect("single geometry");
        assert!(
            SmoothLrSelectionReplay::generate_multiscale(&single, &[(-6.0, 6.0)], 256).is_none()
        );
        // More scales than the grid budget can resolve: declines rather than
        // gridding five axes at four points each — and the dispatcher then hands
        // the term the common-scale slice rather than nothing.
        let many = vec![penalty.clone(); SMOOTH_LR_SELECTION_MAX_SCALES + 1];
        let windows = vec![(-6.0, 6.0); SMOOTH_LR_SELECTION_MAX_SCALES + 1];
        let crowded = SelectionGeometry::whiten(
            &information,
            &many,
            &vec![0.0; SMOOTH_LR_SELECTION_MAX_SCALES + 1],
        )
        .expect("crowded geometry");
        assert!(SmoothLrSelectionReplay::generate_multiscale(&crowded, &windows, 256).is_none());
        assert!(
            SmoothLrSelectionReplay::from_geometry(&crowded, &windows, 256, 256).is_some(),
            "a term with more scales than the grid budget still gets the common-scale slice"
        );
        // Every window closed: nothing to select on any axis.
        let pair = SelectionGeometry::whiten(
            &information,
            &[penalty.clone(), penalty.clone()],
            &[0.0, 0.0],
        )
        .expect("pair geometry");
        assert!(
            SmoothLrSelectionReplay::generate_multiscale(&pair, &[(1.0, 1.0), (2.0, 2.0)], 256)
                .is_none()
        );
        // But ONE open axis is still a selection, and used to be discarded with
        // the closed one — the intersection of the two windows is empty.
        assert!(
            SmoothLrSelectionReplay::generate_multiscale(&pair, &[(1.0, 1.0), (-6.0, 6.0)], 256)
                .is_some(),
            "a scale whose own window is open must still be replayed when a \
             SIBLING scale's window is closed"
        );
        // Information with no identified direction cannot be whitened at all.
        assert!(
            SelectionGeometry::whiten(
                &Array2::<f64>::zeros((3, 3)),
                &[penalty.clone(), penalty],
                &[0.0, 0.0],
            )
            .is_none()
        );
    }

    /// A term with nothing to select — no penalized direction, or a window the
    /// solver's box has closed — has no replay, and the conditional law is the
    /// selection law. This is the branch that keeps an unpenalized block exactly
    /// the textbook chi-square.
    #[test]
    fn nothing_to_select_means_no_replay() {
        assert!(SelectionGeometry::whiten(&Array2::eye(3), &[], &[]).is_none());
        assert!(
            SelectionGeometry::whiten(&Array2::eye(2), &[Array2::zeros((2, 2))], &[0.0]).is_none()
        );
        let geometry = diagonal(&spectrum());
        assert!(SmoothLrSelectionReplay::from_geometry(&geometry, &[(4.0, -4.0)], 256, 256).is_none());
        assert!(
            SmoothLrSelectionReplay::from_geometry(&geometry, &[(f64::NAN, 1.0)], 256, 256)
                .is_none()
        );
    }

    /// #2672: the criterion's log-determinant is priced from the stacked scaled
    /// ROOTS, so a term whose scales separate keeps the coercivity that decides
    /// the selection.
    ///
    /// The replay used to read `Σ_{e > 0} log(1 + 1/e)` off the eigenvalues of
    /// the ASSEMBLED whitened sum. That is the route `penalty_logdet.rs`'s
    /// `SpectrumScale` documents as `O(ε·κ(S_λ))` — and `κ` here is
    /// `exp(ρ̂₁ − ρ̂₂)`, which the box allows to reach `e⁶⁰`. Past `κ ≈ 1e16` the
    /// smaller scale's genuine modes are below the eigendecomposition's own
    /// noise floor, so their `log(1 + 1/e)` is dropped when the noise lands
    /// negative and invented when it lands positive.
    ///
    /// The identity that makes this checkable without a second implementation:
    /// under a COMMON shift the criterion's offset is exactly
    /// `Σ_j log(1 + t·ν_j) − rank·ln t − Σ_{j<rank} ln ν_j`, so scaling every
    /// scale by `t` must move the offset by exactly `−rank·ln t` once the
    /// `log(1 + tν)` part is subtracted. That is a statement about the ANSWER,
    /// not about the arithmetic, and the assembled route violates it by tens of
    /// nats at the separations this fixture uses.
    #[test]
    fn the_criterion_keeps_its_coercivity_when_the_scales_separate() {
        let q = 6;
        // A bending-style penalty on the first four directions and a
        // null-space ridge on the last two: the default double penalty's shape.
        let mut bending = Array2::<f64>::zeros((q, q));
        let mut ridge = Array2::<f64>::zeros((q, q));
        for index in 0..q {
            if index < 4 {
                bending[[index, index]] = 1.0 + index as f64;
            } else {
                ridge[[index, index]] = 1.0;
            }
        }
        for separation in [0.0_f64, 12.0, 24.0, 42.0, 58.0] {
            let geometry = SelectionGeometry::whiten(
                &Array2::eye(q),
                &[bending.clone(), ridge.clone()],
                &[0.5 * separation, -0.5 * separation],
            )
            .expect("geometry");
            assert_eq!(
                geometry.rank, q,
                "the two components span the block, so the structural rank is q \
                 whatever the separation"
            );
            let base = geometry.at(&[0.0, 0.0]).expect("fitted point");
            for shift in [-3.0_f64, 1.5] {
                let moved = geometry.at(&[shift, shift]).expect("shifted point");
                let predicted: f64 = base
                    .eigenvalues
                    .iter()
                    .map(|&nu| (nu * shift.exp()).ln_1p() - nu.ln_1p())
                    .sum::<f64>()
                    - geometry.rank as f64 * shift;
                assert!(
                    (moved.offset - base.offset - predicted).abs() <= 1e-8 * (1.0 + predicted.abs()),
                    "at separation {separation} a common shift of {shift} moved the \
                     criterion's offset by {} where the closed form says {predicted} \
                     — the log-determinant has lost the scales it cannot see",
                    moved.offset - base.offset
                );
            }
        }
    }

    /// The rank is STRUCTURAL, so an unpenalized direction contributes no
    /// log-determinant term at any scale — and a `1e-17` of roundoff on it
    /// cannot invent one.
    #[test]
    fn an_unpenalized_direction_carries_no_log_determinant_term() {
        let q = 4;
        let mut penalty = Array2::<f64>::zeros((q, q));
        for index in 0..q - 1 {
            penalty[[index, index]] = 1.0 + index as f64;
        }
        let geometry =
            SelectionGeometry::whiten(&Array2::eye(q), std::slice::from_ref(&penalty), &[0.0])
                .expect("geometry");
        assert_eq!(geometry.rank, q - 1);
        let base = geometry.at(&[0.0]).expect("fitted point");
        assert_eq!(base.eigenvalues.len(), q);
        // The last direction is unpenalized: weight one, share zero.
        let last = base.eigenvalues[q - 1];
        assert!(last.abs() < 1e-12, "expected a structural zero, got {last}");
        assert!((base.weights[q - 1] - 1.0).abs() < 1e-12);
        // And the offset moves by exactly `−rank·ln t` under a common shift, so
        // the unpenalized direction is not being counted.
        for shift in [-5.0_f64, 4.0] {
            let moved = geometry.at(&[shift]).expect("shifted point");
            let predicted: f64 = base
                .eigenvalues
                .iter()
                .map(|&nu| (nu * shift.exp()).ln_1p() - nu.ln_1p())
                .sum::<f64>()
                - geometry.rank as f64 * shift;
            assert!(
                (moved.offset - base.offset - predicted).abs() <= 1e-9 * (1.0 + predicted.abs()),
                "the unpenalized direction leaked a log-determinant term: {} vs {predicted}",
                moved.offset - base.offset
            );
        }
    }
}

#[cfg(test)]
mod lr_null_spectrum_moment_tests {
    use super::*;

    /// No penalty components and therefore no selection window: these tests are
    /// about the CONDITIONAL law, so the replay is inert.
    const WINDOW: &[(f64, f64)] = &[];

    // The whole-term LR reference (#1766, #1872, #2672). The first spectral
    // moment `tr(2F − F²)` IS Wood's `edf1`; what changed under #2672 is that it
    // is now the MEAN of a reference whose SHAPE comes from the second moment,
    // instead of being handed to a chi-square as a degrees of freedom.

    #[test]
    fn the_first_moment_is_wood_edf1() {
        // A symmetric smoother block with eigenvalues {0.9, 0.4}: a partially
        // shrunk penalized term. edf = tr = 1.3; tr(F²) = 0.81 + 0.16 = 0.97;
        // edf1 = 2·1.3 − 0.97 = 1.63. (Diagonal ⇒ block F² trace = Σ λ².)
        let f = ndarray::array![[0.9_f64, 0.0], [0.0, 0.4]];
        let [mean, second] = lr_null_spectral_moments(Some(&f), &(0..2)).unwrap();
        assert!(
            (mean - 1.63).abs() < 1e-12,
            "the first spectral moment is Wood's edf1 = 2*tr - tr(F^2) = 1.63, got {mean}"
        );
        // And it dominates the raw edf, analytically: w_j = 2f_j − f_j² ≥ f_j on
        // [0, 1], so no `.max(edf)` guard is needed to make it hold.
        assert!(mean >= 1.3 - 1e-12, "edf1 {mean} must be >= edf 1.3");
        // Second moment: w = {0.99, 0.64} ⇒ Σw² = 0.9801 + 0.4096.
        assert!((second - 1.3897).abs() < 1e-12, "second moment {second}");
    }

    #[test]
    fn a_corrupted_block_degrades_to_the_fallback_rather_than_being_floored() {
        // A real influence block has eigenvalues in [0, 1], so `tr(F²)` cannot
        // run away. Numerical corruption can still produce one that does, and
        // the pre-#2672 code floored `edf1` back at `tr` — silently returning a
        // reference derived from a block it had just decided was unusable. The
        // spectral reference does not paper over it: the first moment goes
        // negative and the assembly degrades to the unit-weight lane, VISIBLY.
        let f = ndarray::array![[0.5_f64, 40.0], [40.0, 0.5]];
        let [mean, _] = lr_null_spectral_moments(Some(&f), &(0..2)).unwrap();
        assert!(mean < 0.0, "the corrupted block's first moment is {mean}");
        let reference = lr_null_reference(Some(&f), None, None, &(0..2), 1.0, 1, 0.0, WINDOW, &[], &[]);
        assert_eq!(reference.source, SmoothLrReferenceSource::UnitWeightFallback);
        assert_eq!(reference.chi_square_df, 1.0);
        assert_eq!(reference.scale, 1.0);
    }

    #[test]
    fn returns_none_on_a_missing_or_out_of_range_block() {
        // No influence matrix at all → None (caller falls back to the
        // unit-weight `max(edf, null_dim, 1)` shape).
        assert!(lr_null_spectral_moments(None, &(0..2)).is_none());
        // An out-of-bounds range → None, never a panic.
        let f = ndarray::array![[0.5_f64, 0.0], [0.0, 0.5]];
        assert!(lr_null_spectral_moments(Some(&f), &(0..5)).is_none());
        // A fully-shrunk block has ZERO moments, which is not a usable
        // reference either — the caller must see the fallback, not a divide by
        // zero.
        let zero = ndarray::array![[0.0_f64, 0.0], [0.0, 0.0]];
        assert_eq!(
            lr_null_spectral_moments(Some(&zero), &(0..2)).unwrap(),
            [0.0, 0.0]
        );
        assert_eq!(
            lr_null_reference(Some(&zero), None, None, &(0..2), 0.0, 0, 0.0, WINDOW, &[], &[]).source,
            SmoothLrReferenceSource::UnitWeightFallback
        );
    }
}
