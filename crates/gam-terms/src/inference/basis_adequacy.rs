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
//! Write `s` for the working score with `Uⱼ = (Zᵀs)ⱼ = ∂ℓ/∂γⱼ|_{γ=0}` and
//! `Var(s) = φ·W_F` (`W_F` the Fisher/score-side IRLS weights). The fit solves
//! the penalized score equation `Xᵀs − S_λ β̂ = 0`, so to first order
//!
//! ```text
//!     β̂ − β ≈ H⁻¹ Xᵀ s,        H = Xᵀ W_H X + S_λ
//! ```
//!
//! and therefore
//!
//! ```text
//!     U = Zᵀ s(β̂) ≈ [Z − X H⁻¹ Xᵀ W_H Z]ᵀ s =: Z̃ᵀ s,
//!     Var(U) = φ · Z̃ᵀ W_F Z̃ =: φ · V.
//! ```
//!
//! `Z̃` is the enrichment with the fitted span projected out *in the metric the
//! fit actually used* — `H⁻¹` carries the penalty, not `(XᵀW X)⁻¹`. That is what
//! makes this the score test of the **penalized** fit rather than of an
//! unpenalized one, and it is why `V` is formed from `Z̃` directly rather than
//! from the algebraically equal but cancellation-prone
//! `ZᵀWZ − ZᵀWX(2H⁻¹ − H⁻¹XᵀWXH⁻¹)XᵀWZ`: the `Z̃`-form is a Gram matrix, hence
//! positive semidefinite to machine precision, and its small eigenvalues are
//! genuinely small rather than the residue of a subtraction.
//!
//! The statistic is `T = Uᵀ V⁻ U / φ` over the estimable directions of `V`,
//! referred to `χ²_r` when the dispersion is known and to `F(r, ν)` when it is
//! estimated — the same `Known`/`Estimated` split, for the same reason, as
//! [`crate::inference::smooth_test`].
//!
//! # What it does not claim
//!
//! `λ̂` is held at its fitted value, so `T` is conditional on the selected
//! smoothing parameters, exactly as the summary table's Wald statistic is. The
//! penalty also biases `β̂`, so `E[U] ≠ 0` under `H₀` by an `O(S_λ β)` term that
//! this expansion drops; the same approximation underlies every penalized Wald
//! and score statistic in this crate. The consequence is worth stating plainly:
//! at very large `n` a *correctly sized* basis whose λ is deliberately holding
//! down real wiggle can also reject. That is not a false alarm about the model —
//! there genuinely is residual structure — but it IS a reason not to report
//! "increase k" from this statistic alone. The adequacy verdict a caller should
//! surface pairs it with the term's EDF-vs-capacity evidence.

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
#[derive(Debug, Clone)]
pub struct BasisAdequacyInput<'a> {
    /// `Z` — the enrichment design (`n × q`): higher-resolution directions over
    /// the tested term's covariates. Columns already inside `span(X)` are
    /// harmless; they leave the estimable rank rather than biasing it.
    pub enrichment: ArrayView2<'a, f64>,
    /// `X` — the fitted design (`n × p`) in the same coefficient frame as
    /// `hessian_inverse`.
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
    /// `H⁻¹` — the inverse penalized Hessian **without** dispersion scaling
    /// (`H = XᵀW_H X + S_λ`). This is `Vb/φ̂`.
    pub hessian_inverse: ArrayView2<'a, f64>,
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
        || input.hessian_inverse.nrows() != p
        || input.hessian_inverse.ncols() != p
        || !(input.dispersion.is_finite() && input.dispersion > 0.0)
    {
        return None;
    }

    // U = Zᵀ s — the score for the enrichment coefficients at γ = 0.
    let u = input.enrichment.t().dot(&input.score);
    if u.iter().any(|value| !value.is_finite()) {
        return None;
    }

    // The enrichment's own weighted energy scale, used for the estimability
    // floor. Taken from the UNprojected Gram so that an enrichment largely
    // absorbed by `X` keeps a meaningful reference scale.
    let mut energy_scale = 0.0_f64;
    for column in 0..q {
        let mut diagonal = 0.0;
        for row in 0..n {
            let value = input.enrichment[(row, column)];
            diagonal += input.score_weights[row] * value * value;
        }
        if !diagonal.is_finite() {
            return None;
        }
        energy_scale = energy_scale.max(diagonal);
    }
    if !(energy_scale > 0.0) {
        return None;
    }

    // C = H⁻¹ (Xᵀ W_H Z): the coefficient shift the fit would absorb if the
    // enrichment were switched on. `Z̃ = Z − X·C` is the enrichment with that
    // absorption removed.
    let mut weighted_enrichment = input.enrichment.to_owned();
    for row in 0..n {
        let weight = input.hessian_weights[row];
        if !weight.is_finite() {
            return None;
        }
        weighted_enrichment
            .row_mut(row)
            .iter_mut()
            .for_each(|value| *value *= weight);
    }
    let cross = input.design.t().dot(&weighted_enrichment); // p × q
    let coefficient_shift = input.hessian_inverse.dot(&cross); // p × q
    if coefficient_shift.iter().any(|value| !value.is_finite()) {
        return None;
    }

    // V = Z̃ᵀ W_F Z̃, accumulated in row blocks so the residualized enrichment
    // never has to exist as a second `n × q` array.
    const ROW_BLOCK: usize = 4096;
    let mut information = Array2::<f64>::zeros((q, q));
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
    if information.iter().any(|value| !value.is_finite()) {
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

    /// Gaussian-identity harness: `W = 1`, `s = y − Xβ̂`, `H = XᵀX + S`.
    /// Returns the assembled inputs' owned buffers so views can be taken.
    struct GaussianHarness {
        design: Array2<f64>,
        enrichment: Array2<f64>,
        weights: Array1<f64>,
        score: Array1<f64>,
        hessian_inverse: Array2<f64>,
    }

    impl GaussianHarness {
        fn new(design: Array2<f64>, enrichment: Array2<f64>, y: Array1<f64>, ridge: f64) -> Self {
            let n = design.nrows();
            let p = design.ncols();
            let mut hessian = design.t().dot(&design);
            for index in 0..p {
                hessian[(index, index)] += ridge;
            }
            let hessian_inverse = invert_symmetric(&hessian);
            let beta = hessian_inverse.dot(&design.t().dot(&y));
            let score = &y - &design.dot(&beta);
            Self {
                design,
                enrichment,
                weights: Array1::ones(n),
                score,
                hessian_inverse,
            }
        }

        fn input(&self) -> BasisAdequacyInput<'_> {
            BasisAdequacyInput {
                enrichment: self.enrichment.view(),
                design: self.design.view(),
                hessian_weights: self.weights.view(),
                score_weights: self.weights.view(),
                score: self.score.view(),
                hessian_inverse: self.hessian_inverse.view(),
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
            out.p_value < 1e-6,
            "quadratic lack of fit should be detected, got p={}",
            out.p_value
        );
    }

    /// The projection uses `H⁻¹`, not `(XᵀWX)⁻¹`: with a heavy ridge the two
    /// differ, and using the wrong one changes the statistic. Pinning the
    /// penalized form here is what keeps this a score test OF THE PENALIZED FIT.
    #[test]
    fn projection_uses_the_penalized_hessian() {
        let n = 120;
        let mut rng = Lcg(99);
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
            y[row] = 1.0 + x + 0.4 * rng.next_normal();
        }
        let penalized = GaussianHarness::new(design.clone(), enrichment.clone(), y.clone(), 50.0);
        let unpenalized = GaussianHarness::new(design, enrichment, y, 0.0);
        let a = basis_adequacy_score_test(penalized.input()).expect("penalized result");
        let b = basis_adequacy_score_test(unpenalized.input()).expect("unpenalized result");
        assert!(
            (a.statistic - b.statistic).abs() > 1e-8,
            "the ridge must move the score statistic; got {} vs {}",
            a.statistic,
            b.statistic
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
