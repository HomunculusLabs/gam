//! Penalized score (Rao) lack-of-fit test for basis adequacy.
//!
//! # The question this answers
//!
//! A converged, certified GAM fit says nothing about whether the basis it was
//! given is rich enough to represent the function it was asked to model. The
//! two failure modes look identical from the optimizer's side: a smooth whose
//! basis spans the truth and a smooth whose basis cannot reach it both reach a
//! stationary REML point, both certify, and both report a per-term EDF that is
//! some fraction of the term's column count. What separates them is whether the
//! **residuals still carry structure in the term's own covariates**.
//!
//! This module tests exactly that. Given an *enrichment* design `Z` — a set of
//! higher-resolution directions over the term's covariates that the fitted
//! design `X` does not already span — it computes the penalized score statistic
//! for `H₀: γ = 0` in the augmented model `η = Xβ + Zγ`, evaluated at the fit's
//! own `β̂`. A significant statistic is a positive statement: *there is signal
//! in this smooth's covariates that its realized basis cannot represent.*
//!
//! # Why a score test and not an EDF-saturation rule
//!
//! The engine's `basis_is_saturated` predicate asks whether the term's
//! *penalized* EDF has reached its algebraic ceiling `realized_width −
//! nullspace_dim`. That fires only when λ has been driven to its floor and the
//! basis is exhausted. It cannot see a basis that is far too small while λ is
//! still binding, because basis size and λ both control smoothness and REML
//! trades them off against each other. On the #2774 fixture — a 16-D Duchon
//! smooth with `centers=24`, whose 17-column linear null space leaves a
//! penalized capacity of ~6 — the fit sits at penalized EDF 3.91, i.e. 65% of
//! capacity, so `basis_is_saturated` reports "certified" while the residual
//! confounding is large enough to produce a `6.2e-5` false association.
//!
//! Nor is local residual differencing (mgcv's `k.index`) a substitute. Measured
//! on that same fixture: the 16-D nearest-neighbour index reads `0.928` with a
//! randomization `p = 0.43`, and even an *oracle* ordering — sorting rows by the
//! true simulated confounder — only reaches `0.976`. Differencing throws away
//! the signal it is looking for whenever the missing component is a small
//! fraction of a Bernoulli residual variance. The score statistic below reads
//! `p = 9.5e-16` on the same fit, because its non-centrality grows like
//! `n × (explained variance fraction)` instead of being buried in local noise.
//!
//! # The statistic
//!
//! Write `s` for the working score, so that `∂ℓ/∂γ|_{γ=0} = Zᵀs`, and
//! `Var(s) = φ·W_F` (`W_F` the Fisher/score-side IRLS weights). Let
//! `G = XᵀW_H X` be the design's weighted Gram and
//!
//! ```text
//!     Z̃ = Z − X G⁻ XᵀW_H Z
//! ```
//!
//! the enrichment with the fitted design projected out **in the `W_H` metric**.
//! Then `Z̃ᵀW_H X = 0` exactly, and since the fit's error propagates into the
//! score only through `β̂ − β`,
//!
//! ```text
//!     U = Z̃ᵀ s(β̂) = Z̃ᵀ s(β) − Z̃ᵀW_H X(β̂ − β) = Z̃ᵀ s(β),
//!     Var(U) = φ · Z̃ᵀ W_F Z̃ =: φ · V,     T = Uᵀ V⁻ U / φ.
//! ```
//!
//! `T` is referred to `χ²_r` when the dispersion is known and to `F(r, ν)` when
//! it is estimated — the same `Known`/`Estimated` split, for the same reason, as
//! [`crate::inference::smooth_test`].
//!
//! # Why the UNPENALIZED Gram, and not `H⁻¹`
//!
//! The obvious construction projects with the fit's own penalized Hessian
//! `H⁻¹ = (XᵀW_H X + S_λ)⁻¹`, which is what the first-order expansion of a
//! penalized score test hands you. It is wrong here, for a reason that is about
//! the QUESTION and not about the algebra.
//!
//! A penalized fit is biased: `E[β̂] − β ≈ −H⁻¹S_λβ`, so the residuals carry a
//! systematic component `W_H X H⁻¹S_λβ` that lives inside `span(X)`. Under the
//! `H⁻¹` projection `Z̃ᵀW_H X = ZᵀW_H X H⁻¹S_λ ≠ 0`, and that component leaks
//! straight into `E[U]`, so the statistic is non-central under `H₀` by an amount set by
//! how hard λ is shrinking — it reports "λ is doing work", which is true of
//! every GAM ever fitted and grows with `n`. Projecting in the `W_H` metric
//! annihilates it exactly, because the entire shrinkage bias lies in `span(X)`.
//!
//! The semantic statement is the same one: with `G⁻`, `T` tests only directions
//! the realized design **cannot represent at all**. Shrinking a direction the
//! basis HAS is a smoothing-parameter question, not a basis-size one, and this
//! statistic deliberately declines to answer it. The invariance
//! `Z → Z + X·A ⟹ T unchanged` (pinned in
//! `statistic_is_invariant_to_shifting_the_enrichment_by_design_columns`) is the
//! executable form of that contract; the `H⁻¹` projection does not satisfy it.
//!
//! `V` is accumulated as the Gram `Z̃ᵀW_F Z̃` of the residualized enrichment
//! rather than as the algebraically equal Schur complement
//! `ZᵀW_F Z − ZᵀW_F X G⁻ XᵀW_F Z`. The Gram form is PSD to machine precision and
//! its small eigenvalues are genuinely small instead of being the residue of a
//! subtraction — which matters exactly when part of the enrichment is nearly
//! inside `span(X)`, i.e. always.
//!
//!
//! # It does not assume the fit solved an unmodified score equation
//!
//! `U = Z̃ᵀ s(β̂) = Z̃ᵀ s(β) − Z̃ᵀ W_H X (β̂ − β)`, and the second term vanishes
//! because `Z̃ᵀW_H X = 0` — for ANY `β̂`, not just the one a plain penalized
//! IRLS produces. A Firth/Jeffreys-adjusted fit, which solves
//! `Xᵀs − S_λβ̂ + ∂ log|I|/∂β = 0` rather than `Xᵀs = S_λβ̂`, is therefore
//! handled with no special case: the adjustment moves `β̂`, and the projection
//! removes whatever `β̂` does. The `H⁻¹`-projected variant has no such property,
//! since its correction term is a specific function of the score equation it
//! assumed.
//! # What it does not claim
//!
//! `λ̂` is held at its fitted value and the enrichment is a fixed alternative,
//! so `T` is conditional on both, exactly as the summary table's Wald statistic
//! is conditional on `λ̂`. A rejection says there is signal in this smooth's
//! covariates outside its realized column span; it does not say how much of the
//! *estimand* that signal moves. The caller pairs it with the term's
//! EDF-vs-capacity evidence and reports both.

use faer::Side;
use gam_linalg::faer_ndarray::strict_symmetric_eigh;
use gam_math::probability::{chi_square_sf, fisher_snedecor_sf};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

pub use crate::inference::smooth_test::SmoothTestScale;

/// Relative floor for calling an enrichment direction *estimable*.
///
/// `V = Z̃ᵀW_F Z̃` is a Gram matrix, so its eigenvalues are non-negative and an
/// enrichment direction that lies inside the fitted span produces a
/// numerically-zero one. The floor is taken relative to the enrichment's own
/// weighted energy scale (`max_j (ZᵀW_F Z)_jj`) rather than to `V`'s largest
/// eigenvalue, so a `Z` that is *mostly* absorbed by `X` does not have its floor
/// collapse along with its spectrum. `1e-9` sits two orders above the
/// `ε·cond(X)` roundoff of the projection for the conditioning this engine
/// admits, and directions below it contribute `(u'/λ)` ratios that are noise
/// over noise.
const ESTIMABLE_DIRECTION_FLOOR: f64 = 1.0e-9;

/// Inputs to [`basis_adequacy_score_test`].
///
/// Every matrix is in the fit's own coefficient/row layout. `design`,
/// `hessian_weights`, `score_weights` and `score` share the fit's row order;
/// `enrichment` must be evaluated at those same rows.
pub struct BasisAdequacyInput<'a> {
    /// `Z` — the enrichment design (`n × q`): higher-resolution directions over
    /// the tested term's covariates. Columns already inside `span(X)` are
    /// harmless; they leave the estimable rank rather than biasing it.
    pub enrichment: ArrayView2<'a, f64>,
    /// `X` — the fitted design (`n × p`) in the same coefficient frame as
    /// `design_gram`.
    pub design: ArrayView2<'a, f64>,
    /// `W_H` — the diagonal curvature weights the fit's penalized Hessian was
    /// assembled from (observed information where the fit tracked it). Used
    /// only to build the projection.
    pub hessian_weights: ArrayView1<'a, f64>,
    /// `W_F` — the Fisher/score-side IRLS weights, i.e. `Var(s) = φ·W_F`.
    /// Equal to `hessian_weights` for a canonical link.
    pub score_weights: ArrayView1<'a, f64>,
    /// `s` — the per-row working score, `sᵢ = wᵢ(yᵢ − μ̂ᵢ)(dμ/dη)ᵢ / V(μ̂ᵢ)`, so
    /// that `U = Zᵀ s` is the score for the enrichment coefficients.
    pub score: ArrayView1<'a, f64>,
    /// Factored `G = XᵀW_H X` — the design's weighted Gram, **without** the
    /// penalty and **without** dispersion scaling. Factored once by the caller
    /// and reused across the model's smooth terms; see [`DesignGramFactor`].
    /// The module header explains why this is the unpenalized Gram and not the
    /// penalized Hessian.
    pub design_gram: &'a DesignGramFactor,
    /// `φ̂` — the fitted dispersion. `1.0` for families that carry their
    /// dispersion inside the IRLS weight.
    pub dispersion: f64,
    /// Denominator d.f. for the `Estimated`-scale `F` reference. Ignored on the
    /// `Known` branch.
    pub residual_df: Option<f64>,
    pub scale: SmoothTestScale,
}

/// Outcome of the penalized score lack-of-fit test.
#[derive(Debug, Clone, PartialEq)]
pub struct BasisAdequacyResult {
    /// `T = Uᵀ V⁻ U / φ̂`.
    pub statistic: f64,
    /// Number of estimable enrichment directions actually summed — the
    /// reference d.f. This is the enrichment width MINUS whatever part of it the
    /// fitted design already spanned, so it reports how much genuinely new
    /// resolution the alternative carried.
    pub rank: usize,
    /// `P(χ²_rank > T)`, or the matching `F` tail when the scale is estimated.
    pub p_value: f64,
}

/// Penalized score (Rao) test of `H₀: γ = 0` in `η = Xβ + Zγ` at the fit's `β̂`.
///
/// Returns `None` — never a stand-in value — when the inputs cannot support the
/// test: mismatched shapes, a non-finite entry anywhere in the assembled
/// quadratic form, a non-positive dispersion, no estimable enrichment direction
/// left after projection, or an `Estimated` scale with no usable residual d.f.
/// An absent verdict is a caller-visible "not measured", which is the only
/// honest report when the geometry is missing.
pub fn basis_adequacy_score_test(
    input: BasisAdequacyInput<'_>,
) -> Option<BasisAdequacyResult> {
    let n = input.design.nrows();
    let p = input.design.ncols();
    let q = input.enrichment.ncols();
    if n == 0
        || p == 0
        || q == 0
        || input.enrichment.nrows() != n
        || input.hessian_weights.len() != n
        || input.score_weights.len() != n
        || input.score.len() != n
        || input.design_gram.dimension() != p
        || !(input.dispersion.is_finite() && input.dispersion > 0.0)
    {
        return None;
    }

    // Row-blocked first pass: `XᵀW_H Z` (the projection's right-hand side) and
    // the enrichment's own `W_F`-weighted energy scale (the estimability floor's
    // reference). Blocked for the same reason the second pass is — a second
    // `n × q` array is 154 MB at `n = 200_000, q = 96`, and a diagnostic may not
    // be the peak-memory term of the fit it is diagnosing.
    //
    // The energy scale is taken from the UNPROJECTED Gram so that an enrichment
    // largely absorbed by `X` keeps a meaningful reference scale instead of
    // having its floor collapse along with its spectrum.
    const ROW_BLOCK: usize = 4096;
    let mut cross = Array2::<f64>::zeros((p, q)); // XᵀW_H Z
    let mut energy = Array1::<f64>::zeros(q);
    let mut start = 0usize;
    while start < n {
        let stop = (start + ROW_BLOCK).min(n);
        let block = input.enrichment.slice(ndarray::s![start..stop, ..]);
        let mut hessian_weighted = block.to_owned();
        for local in 0..(stop - start) {
            let curvature = input.hessian_weights[start + local];
            let fisher = input.score_weights[start + local];
            if !curvature.is_finite() || !(fisher.is_finite() && fisher >= 0.0) {
                return None;
            }
            let source = block.row(local);
            let mut target = hessian_weighted.row_mut(local);
            for column in 0..q {
                let value = source[column];
                energy[column] += fisher * value * value;
                target[column] = curvature * value;
            }
        }
        cross += &input
            .design
            .slice(ndarray::s![start..stop, ..])
            .t()
            .dot(&hessian_weighted);
        start = stop;
    }
    let energy_scale = energy.iter().cloned().fold(0.0_f64, f64::max);
    if !(energy_scale > 0.0) || cross.iter().any(|value| !value.is_finite()) {
        return None;
    }

    // C = G⁻ (Xᵀ W_H Z): the `W_H`-orthogonal projection of the enrichment onto
    // the fitted column span. `Z̃ = Z − X·C` is the part of the enrichment the
    // realized design cannot represent, and it satisfies `Z̃ᵀW_H X = 0`.
    let coefficient_shift = input.design_gram.solve(&cross)?;
    if coefficient_shift.iter().any(|value| !value.is_finite()) {
        return None;
    }

    // `U = Z̃ᵀ s` and `V = Z̃ᵀ W_F Z̃`, accumulated together in row blocks so the
    // residualized enrichment never has to exist as a second `n × q` array.
    //
    // The score MUST be contracted against `Z̃`, not `Z`. `Zᵀs = Z̃ᵀs + (X·C)ᵀs`
    // and the fit solves the PENALIZED score equation `Xᵀs = S_λβ̂`, so the
    // second term is `Cᵀ S_λ β̂` — precisely the shrinkage this construction
    // exists to remove, re-entering through the numerator after the projection
    // took it out of the denominator. It also breaks the `Z → Z + X·A`
    // invariance, since `C → C + A`. Both failures are pinned as tests.
    let mut information = Array2::<f64>::zeros((q, q));
    let mut u = Array1::<f64>::zeros(q);
    let mut start = 0usize;
    while start < n {
        let stop = (start + ROW_BLOCK).min(n);
        let rows = stop - start;
        let mut residualized = input
            .enrichment
            .slice(ndarray::s![start..stop, ..])
            .to_owned();
        residualized -= &input
            .design
            .slice(ndarray::s![start..stop, ..])
            .dot(&coefficient_shift);
        u += &residualized
            .t()
            .dot(&input.score.slice(ndarray::s![start..stop]));
        let mut weighted = residualized.clone();
        for local in 0..rows {
            let weight = input.score_weights[start + local];
            if !(weight.is_finite() && weight >= 0.0) {
                return None;
            }
            weighted
                .row_mut(local)
                .iter_mut()
                .for_each(|value| *value *= weight);
        }
        information += &residualized.t().dot(&weighted);
        start = stop;
    }
    if information.iter().any(|value| !value.is_finite())
        || u.iter().any(|value| !value.is_finite())
    {
        return None;
    }
    // Symmetrize the accumulated Gram: the block sum is symmetric in exact
    // arithmetic, and `strict_symmetric_eigh` refuses anything that is not
    // symmetric on the nose rather than silently repairing it.
    let symmetric = 0.5 * (&information + &information.t());

    let (eigenvalues, eigenvectors) = strict_symmetric_eigh(&symmetric, Side::Lower).ok()?;
    let projected: Array1<f64> = eigenvectors.t().dot(&u);
    let floor = energy_scale * ESTIMABLE_DIRECTION_FLOOR;
    let mut statistic = 0.0_f64;
    let mut rank = 0usize;
    for (index, &eigenvalue) in eigenvalues.iter().enumerate() {
        if eigenvalue > floor {
            let component = projected[index];
            statistic += component * component / eigenvalue;
            rank += 1;
        }
    }
    if rank == 0 {
        return None;
    }
    let statistic = statistic / input.dispersion;
    if !statistic.is_finite() || statistic < 0.0 {
        return None;
    }

    let reference_df = rank as f64;
    let p_value = match input.scale {
        SmoothTestScale::Known => chi_square_sf(statistic, reference_df),
        SmoothTestScale::Estimated => {
            let residual_df = input
                .residual_df
                .filter(|value| value.is_finite() && *value > 0.0)?;
            fisher_snedecor_sf(statistic / reference_df, reference_df, residual_df)
        }
    };
    if !p_value.is_finite() {
        return None;
    }
    Some(BasisAdequacyResult {
        statistic,
        rank,
        p_value,
    })
}

/// A once-per-fit factorization of the weighted design Gram `G = XᵀW_H X`.
///
/// The projection `C = G⁻(XᵀW_H Z)` is applied once per SMOOTH TERM, but `G`
/// depends only on the design and the weights. Factoring it inside the test
/// would pay `O(p³)` per term — on a model with ten smooths that is ten extra
/// IRLS-iteration-equivalents on a fit that runs a few dozen, which is a
/// diagnostic charging a third of the fit. Building the factor is therefore the
/// caller's job and it is a type, not a convention: the input struct cannot be
/// constructed with a raw matrix that someone forgot to reuse.
pub struct DesignGramFactor {
    kind: DesignGramFactorKind,
    dimension: usize,
}

enum DesignGramFactorKind {
    /// The ordinary route. `O(p³)` once, then `O(p²q)` per solve.
    Cholesky(gam_linalg::faer_ndarray::FaerCholeskyFactor),
    /// Rank-deficient fallback: the spectral pseudo-inverse, held as
    /// `U diag(1/λ) Uᵀ` over the directions above the rank floor. It projects
    /// onto `range(G)`, which is the right answer for a design that is
    /// rank-deficient in the fit's own frame — directions the design cannot
    /// span in the `W_H` metric are not directions to project out. A dense
    /// symmetric eigendecomposition is the expensive route (it is the #2757
    /// cost complaint at `p = 4096`), so it is the exception rather than the
    /// default.
    SpectralPseudoInverse(Array2<f64>),
}

impl DesignGramFactor {
    /// Factor `G`. `None` when the matrix is empty, non-square, non-finite, or
    /// has no positive spectrum at all.
    pub fn new(gram: ArrayView2<'_, f64>) -> Option<Self> {
        use gam_linalg::faer_ndarray::FaerCholesky;
        let dimension = gram.nrows();
        if dimension == 0
            || gram.ncols() != dimension
            || gram.iter().any(|value| !value.is_finite())
        {
            return None;
        }
        let owned = gram.to_owned();
        if let Ok(factor) = owned.cholesky(Side::Lower) {
            return Some(Self {
                kind: DesignGramFactorKind::Cholesky(factor),
                dimension,
            });
        }
        let symmetric = 0.5 * (&owned + &owned.t());
        let (eigenvalues, eigenvectors) = strict_symmetric_eigh(&symmetric, Side::Lower).ok()?;
        let largest = eigenvalues.iter().cloned().fold(0.0_f64, f64::max);
        if !(largest > 0.0) {
            return None;
        }
        let floor = largest * GRAM_RANK_FLOOR;
        let mut scaled = eigenvectors.clone();
        for (index, &eigenvalue) in eigenvalues.iter().enumerate() {
            let factor = if eigenvalue > floor {
                1.0 / eigenvalue
            } else {
                0.0
            };
            scaled.column_mut(index).iter_mut().for_each(|v| *v *= factor);
        }
        let pseudo_inverse = scaled.dot(&eigenvectors.t());
        pseudo_inverse
            .iter()
            .all(|value| value.is_finite())
            .then_some(Self {
                kind: DesignGramFactorKind::SpectralPseudoInverse(pseudo_inverse),
                dimension,
            })
    }

    /// Side length of the factored Gram, i.e. the design's column count.
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    fn solve(&self, rhs: &Array2<f64>) -> Option<Array2<f64>> {
        let solved = match &self.kind {
            DesignGramFactorKind::Cholesky(factor) => factor.solve_mat(rhs),
            DesignGramFactorKind::SpectralPseudoInverse(inverse) => inverse.dot(rhs),
        };
        solved.iter().all(|value| value.is_finite()).then_some(solved)
    }
}

/// Relative eigenvalue floor for the rank-deficient-Gram fallback in
/// [`solve_symmetric_psd`]. Directions of the weighted design Gram below
/// `λ_max · 1e-12` are treated as outside the design's realized span.
const GRAM_RANK_FLOOR: f64 = 1.0e-12;

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2, array};

    /// Deterministic linear-congruential normal draws, so the size/power checks
    /// below are reproducible without pulling a sampler dependency into this
    /// crate's test surface.
    struct Lcg(u64);

    impl Lcg {
        fn next_uniform(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
        }

        fn next_normal(&mut self) -> f64 {
            // Box-Muller; the tail truncation from clamping u away from 0 is
            // far below anything these moment-level checks resolve.
            let u1 = self.next_uniform().max(1e-12);
            let u2 = self.next_uniform();
            (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
        }
    }

    /// Gaussian-identity harness: `W = 1`, `β̂` is the RIDGE-penalized least
    /// squares fit (`H = XᵀX + ridge·I`), `s = y − Xβ̂`, and the Gram handed to
    /// the test is the unpenalized `XᵀX`. The ridge is a knob so a test can vary
    /// how hard the fit is shrunk without touching anything else.
    struct GaussianHarness {
        design: Array2<f64>,
        enrichment: Array2<f64>,
        weights: Array1<f64>,
        score: Array1<f64>,
        design_gram: DesignGramFactor,
    }

    impl GaussianHarness {
        fn new(design: Array2<f64>, enrichment: Array2<f64>, y: Array1<f64>, ridge: f64) -> Self {
            let n = design.nrows();
            let p = design.ncols();
            let gram = design.t().dot(&design);
            let mut hessian = gram.clone();
            for index in 0..p {
                hessian[(index, index)] += ridge;
            }
            let beta = invert_symmetric(&hessian).dot(&design.t().dot(&y));
            let score = &y - &design.dot(&beta);
            Self {
                design,
                enrichment,
                weights: Array1::ones(n),
                score,
                design_gram: DesignGramFactor::new(gram.view())
                    .expect("test harness Gram is factorable"),
            }
        }

        fn input(&self) -> BasisAdequacyInput<'_> {
            BasisAdequacyInput {
                enrichment: self.enrichment.view(),
                design: self.design.view(),
                hessian_weights: self.weights.view(),
                score_weights: self.weights.view(),
                score: self.score.view(),
                design_gram: &self.design_gram,
                dispersion: 1.0,
                residual_df: None,
                scale: SmoothTestScale::Known,
            }
        }
    }

    fn invert_symmetric(matrix: &Array2<f64>) -> Array2<f64> {
        let (values, vectors) = strict_symmetric_eigh(matrix, Side::Lower)
            .expect("test harness matrix is symmetric positive definite");
        let mut inverse = Array2::<f64>::zeros(matrix.raw_dim());
        for (index, &value) in values.iter().enumerate() {
            let column = vectors.column(index);
            let scale = 1.0 / value;
            for row in 0..matrix.nrows() {
                for col in 0..matrix.ncols() {
                    inverse[(row, col)] += scale * column[row] * column[col];
                }
            }
        }
        inverse
    }

    /// An enrichment entirely inside `span(X)` leaves no estimable direction:
    /// the projection annihilates it and the test refuses rather than reporting
    /// a degenerate statistic against a zero-variance direction.
    #[test]
    fn enrichment_inside_the_fitted_span_has_no_estimable_direction() {
        let design = array![
            [1.0, 0.0],
            [1.0, 1.0],
            [1.0, 2.0],
            [1.0, 3.0],
            [1.0, 4.0],
            [1.0, 5.0]
        ];
        // Exact linear combinations of the two design columns.
        let enrichment = design.dot(&array![[2.0, -1.0], [0.5, 3.0]]);
        let y = array![0.3, -0.2, 0.7, 0.1, -0.5, 0.4];
        let harness = GaussianHarness::new(design, enrichment, y, 0.0);
        assert_eq!(basis_adequacy_score_test(harness.input()), None);
    }

    /// The rank reports the genuinely NEW resolution: a `q = 3` enrichment whose
    /// first column duplicates a design column is rank 2.
    #[test]
    fn rank_counts_only_directions_outside_the_fitted_span() {
        let mut rng = Lcg(20_260_823);
        let n = 200;
        let mut design = Array2::<f64>::zeros((n, 2));
        let mut enrichment = Array2::<f64>::zeros((n, 3));
        let mut y = Array1::<f64>::zeros(n);
        for row in 0..n {
            let x = row as f64 / n as f64;
            design[(row, 0)] = 1.0;
            design[(row, 1)] = x;
            enrichment[(row, 0)] = x; // already in span(X)
            enrichment[(row, 1)] = x * x;
            enrichment[(row, 2)] = x * x * x;
            y[row] = 0.5 + 2.0 * x + 0.1 * rng.next_normal();
        }
        let harness = GaussianHarness::new(design, enrichment, y, 0.0);
        let out = basis_adequacy_score_test(harness.input())
            .expect("two enrichment directions remain estimable");
        assert_eq!(out.rank, 2);
    }

    /// A correctly specified fit produces a p-value that is not concentrated at
    /// zero: the mean of the statistic sits near its reference d.f.
    ///
    /// This is the null-behaviour anchor. `y` is linear in `x` and the design
    /// spans that exactly, so the quadratic/cubic enrichment tests a true `H₀`;
    /// `E[T] = rank` is the moment identity a correctly scaled score statistic
    /// must satisfy.
    #[test]
    fn null_statistic_has_mean_near_its_reference_df() {
        let n = 400;
        let replicates = 200;
        let mut rng = Lcg(1_234_567);
        let mut total = 0.0;
        let mut rank_seen = 0usize;
        let mut rejections = 0usize;
        for _ in 0..replicates {
            let mut design = Array2::<f64>::zeros((n, 2));
            let mut enrichment = Array2::<f64>::zeros((n, 3));
            let mut y = Array1::<f64>::zeros(n);
            for row in 0..n {
                let x = (row as f64 + 0.5) / n as f64;
                design[(row, 0)] = 1.0;
                design[(row, 1)] = x;
                enrichment[(row, 0)] = x * x;
                enrichment[(row, 1)] = x * x * x;
                enrichment[(row, 2)] = (6.0 * x).sin();
                y[row] = 0.5 + 2.0 * x + rng.next_normal();
            }
            let harness = GaussianHarness::new(design, enrichment, y, 0.0);
            let out = basis_adequacy_score_test(harness.input()).expect("estimable enrichment");
            total += out.statistic;
            rank_seen = out.rank;
            if out.p_value < 0.05 {
                rejections += 1;
            }
        }
        let mean = total / replicates as f64;
        let expected = rank_seen as f64;
        // sd(χ²_r)/√reps = √(2r/reps) ≈ 0.17 for r = 3, reps = 200; 4σ ≈ 0.7.
        assert!(
            (mean - expected).abs() < 0.7,
            "null mean statistic {mean} should sit near rank {expected}"
        );
        // Nominal 5% over 200 draws: sd = √(0.05·0.95/200) ≈ 0.0154, so 0.12 is
        // a ~4.5σ band around 0.05 — loose enough to be stable, tight enough to
        // catch a statistic that is systematically inflated.
        let size = rejections as f64 / replicates as f64;
        assert!(size < 0.12, "null rejection rate {size} is inflated");
    }

    /// A basis that cannot reach the truth is detected: the same design/enrichment
    /// pair as the null check, but with a quadratic mean the linear design cannot
    /// represent, rejects overwhelmingly.
    #[test]
    fn missing_curvature_is_detected() {
        let n = 400;
        let mut rng = Lcg(7_654_321);
        let mut design = Array2::<f64>::zeros((n, 2));
        let mut enrichment = Array2::<f64>::zeros((n, 3));
        let mut y = Array1::<f64>::zeros(n);
        for row in 0..n {
            let x = (row as f64 + 0.5) / n as f64;
            design[(row, 0)] = 1.0;
            design[(row, 1)] = x;
            enrichment[(row, 0)] = x * x;
            enrichment[(row, 1)] = x * x * x;
            enrichment[(row, 2)] = (6.0 * x).sin();
            y[row] = 0.5 + 2.0 * x + 3.0 * x * x + rng.next_normal();
        }
        let harness = GaussianHarness::new(design, enrichment, y, 0.0);
        let out = basis_adequacy_score_test(harness.input()).expect("estimable enrichment");
        assert!(
            out.p_value < 1e-3,
            "quadratic lack of fit should be detected, got p={}",
            out.p_value
        );
    }

    /// **The defining contract.** Shifting the enrichment by ANY multiple of the
    /// design columns (`Z → Z + X·A`) leaves the statistic bit-comparably
    /// unchanged, because `Z̃` is the `W_H`-orthogonal complement of `span(X)`
    /// and `X·A` lies entirely inside it.
    ///
    /// This is what separates the shipped construction from the `H⁻¹`-projected
    /// penalized score test, which does NOT satisfy it: there `Z̃ = ... G H⁻¹S_λ`
    /// picks up whatever part of `X·A` the penalty shrinks, so the "same"
    /// alternative reparameterized differently gives a different answer, and the
    /// difference is the fit's shrinkage bias rather than any lack of fit.
    #[test]
    fn statistic_is_invariant_to_shifting_the_enrichment_by_design_columns() {
        let n = 300;
        let mut rng = Lcg(4_242);
        let mut design = Array2::<f64>::zeros((n, 3));
        let mut enrichment = Array2::<f64>::zeros((n, 2));
        let mut y = Array1::<f64>::zeros(n);
        for row in 0..n {
            let x = (row as f64 + 0.5) / n as f64;
            design[(row, 0)] = 1.0;
            design[(row, 1)] = x;
            design[(row, 2)] = (3.0 * x).cos();
            enrichment[(row, 0)] = x * x;
            enrichment[(row, 1)] = (5.0 * x).sin();
            y[row] = 1.0 + x + 0.8 * x * x + 0.4 * rng.next_normal();
        }
        // A heavy ridge, so the fit is visibly shrunk and any leak of the
        // shrinkage bias into the statistic would be large.
        let base = GaussianHarness::new(design.clone(), enrichment.clone(), y.clone(), 40.0);
        let shift = array![[7.0, -2.0], [0.5, 3.0], [-1.5, 4.0]];
        let shifted_enrichment = &enrichment + &design.dot(&shift);
        let shifted = GaussianHarness::new(design, shifted_enrichment, y, 40.0);
        let a = basis_adequacy_score_test(base.input()).expect("base result");
        let b = basis_adequacy_score_test(shifted.input()).expect("shifted result");
        assert_eq!(a.rank, b.rank);
        assert!(
            (a.statistic - b.statistic).abs() <= 1e-8 * a.statistic.max(1.0),
            "statistic must not move under Z -> Z + X·A; got {} vs {}",
            a.statistic,
            b.statistic
        );
    }

    /// The shrinkage bias of the penalized fit does not enter the statistic:
    /// varying the ridge over four orders of magnitude, with the DATA held
    /// fixed, leaves the null statistic in the same neighbourhood instead of
    /// growing with how hard the fit is shrunk.
    #[test]
    fn statistic_does_not_track_the_penalty_strength_under_the_null() {
        let n = 500;
        let mut rng = Lcg(31_337);
        let mut design = Array2::<f64>::zeros((n, 3));
        let mut enrichment = Array2::<f64>::zeros((n, 4));
        let mut y = Array1::<f64>::zeros(n);
        for row in 0..n {
            let x = (row as f64 + 0.5) / n as f64;
            design[(row, 0)] = 1.0;
            design[(row, 1)] = x;
            design[(row, 2)] = x * x;
            enrichment[(row, 0)] = x * x * x;
            enrichment[(row, 1)] = (7.0 * x).sin();
            enrichment[(row, 2)] = (7.0 * x).cos();
            enrichment[(row, 3)] = (11.0 * x).sin();
            // Truth is exactly in span(X): H₀ holds however hard the ridge bites.
            y[row] = 1.0 + 3.0 * x - 2.0 * x * x + rng.next_normal();
        }
        let mut statistics = Vec::new();
        for ridge in [0.0, 1.0, 1.0e2, 1.0e4] {
            let harness =
                GaussianHarness::new(design.clone(), enrichment.clone(), y.clone(), ridge);
            let out = basis_adequacy_score_test(harness.input()).expect("estimable enrichment");
            statistics.push(out.statistic);
        }
        let span = statistics
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max)
            - statistics.iter().cloned().fold(f64::INFINITY, f64::min);
        assert!(
            span < 4.0,
            "the ridge must not drive the null statistic; got {statistics:?}"
        );
    }

    /// Shape and finiteness guards refuse rather than returning a stand-in.
    #[test]
    fn degenerate_inputs_refuse() {
        let design = array![[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]];
        let enrichment = array![[0.0], [1.0], [4.0]];
        let y = array![0.1, 0.2, 0.3];
        let harness = GaussianHarness::new(design, enrichment, y, 1.0);

        let mut bad_dispersion = harness.input();
        bad_dispersion.dispersion = 0.0;
        assert_eq!(basis_adequacy_score_test(bad_dispersion), None);

        let mismatched = Array2::<f64>::zeros((2, 1));
        let mut bad_rows = harness.input();
        bad_rows.enrichment = mismatched.view();
        assert_eq!(basis_adequacy_score_test(bad_rows), None);

        let mut estimated_without_df = harness.input();
        estimated_without_df.scale = SmoothTestScale::Estimated;
        estimated_without_df.residual_df = None;
        assert_eq!(basis_adequacy_score_test(estimated_without_df), None);
    }
}
