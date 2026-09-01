//! #2690 — what a symmetric probe ladder can actually resolve about a curvature.
//!
//! Every gate in the criterion-resolution cluster (#2624, #2665, #2676, and the
//! `invert_identified_rho_hessian` refusal) compares a curvature against a bound
//! and none of them derives that bound from the criterion's own evaluation
//! error. This is that denominator, as an instrument rather than a procedure.
//!
//! # The law
//!
//! A central second difference along a unit direction `v` with step `h`
//!
//! ```text
//! D²f(h) = [f(x+hv) − 2f(x) + f(x−hv)] / h²
//! ```
//!
//! carries two errors moving oppositely in `h`: evaluation noise `4ε_f/h²` from
//! three evaluations each good to `ε_f`, and truncation `(h²/12)·M₄` with `M₄`
//! bounding the fourth directional derivative. Minimising their sum gives
//! `h* = (48·ε_f/M₄)^{1/4}`, where the two terms are equal, and
//!
//! ```text
//! δσ_min = (2/√3)·√(ε_f · M₄)
//! ```
//!
//! **`ε_f` enters under a square root.** That is why bounds built linearly from
//! `ε_f` are far too tight, and why a 100× reduction in evaluation error buys
//! only 10× in resolvable curvature.
//!
//! # Why both inputs come free from the ladder
//!
//! Given symmetric probes `Δ(α) = f(x ± αv) − f(x)`, the per-row estimate
//! `c(α) = 2Δ/α²` is not constant — it drifts as `c(α) ≈ vᵀHv + (M₄/12)·α²`.
//! Regressing `c` on `α²` therefore yields `M₄` as twelve times the slope and a
//! truncation-corrected curvature as the intercept.
//!
//! Differencing the ladder against `½·c·α²` leaves a residual that scales with α
//! where truncation dominates and **goes flat** where evaluation noise does. That
//! plateau is `ε_f`, because a term that does not scale with the step is not a
//! step effect. No extra evaluation, no frozen-θ re-entry, no knowledge of the
//! criterion's internals.
//!
//! # Scope
//!
//! This is the **finite-difference** law. An analytic Hessian has no `1/h²`
//! amplification and `ε_f` — a statement about the VALUE — does not bound a
//! separately coded second derivative; there the resolution is the analytic
//! Hessian's own reproducibility and Weyl gives `|σ_i(H+δH) − σ_i(H)| ≤ ‖δH‖₂`
//! with no propagation constant at all. **Applying the law below to a
//! `terminal-analytic` curvature is a category error**, and the two must not be
//! pooled: an `ε_f` cut of `r` buys `r` on an Armijo gate, `√r` here, and
//! nothing on a terminal-analytic one.
//!
//! `ε_f` is a property of *a criterion on a dataset at a ρ*, not of the
//! codebase. Two fixtures measured here differ by three orders (`1.5e-8` vs
//! `3.79e-5`), so a value must never be carried between them.

/// What a probe ladder resolves, with every field measured from that ladder.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CurvatureResolution {
    /// Criterion evaluation error, read off the residual plateau.
    pub eps_f: f64,
    /// Fourth-directional-derivative bound, twelve times the truncation slope.
    pub m4: f64,
    /// Truncation-corrected `vᵀHv` — the regression intercept, better than any
    /// single row.
    pub curvature: f64,
    /// Optimal step `h* = (48·ε_f/M₄)^{1/4}`.
    pub optimal_step: f64,
    /// `δσ_min = (2/√3)·√(ε_f·M₄)` — the smallest curvature this ladder can
    /// distinguish from zero. A gate comparing a curvature against anything
    /// tighter than this is asking for an accuracy the probes cannot deliver.
    pub delta_sigma_min: f64,
}

/// Why a ladder could not be metered. Refusing is the point: a fabricated
/// resolution is worse than none, because it is not legible as fabricated.
#[derive(Debug, Clone, PartialEq)]
pub enum CurvatureResolutionError {
    /// Fewer than four rows; the regression and the plateau need separate rows.
    TooFewRows { rows: usize },
    /// A non-finite or non-positive step, or a non-finite probe value.
    NonFiniteInput,
    /// Steps do not span enough decades to separate truncation from noise.
    InsufficientSpan { decades: f64 },
    /// The truncation slope is non-positive, so `M₄` cannot be read: the ladder
    /// is entirely noise-dominated, or `c(α)` drifts the wrong way.
    TruncationNotResolved { slope: f64 },
    /// Fewer than three large-α rows agree on `c = 2Δ/α²`, so there is no clean
    /// truncation regime to regress. Either the ladder starts already
    /// noise-dominated, or the steps are so coarse that truncation swamps the
    /// signal between adjacent rows.
    TruncationRegimeTooShort { agreeing_rows: usize },
    /// The small-α residuals are still scaling with the step, so no plateau has
    /// been reached and `ε_f` is not measurable from these rows. The ladder must
    /// be extended to smaller α — the rows usually discarded as "below the
    /// floor" ARE the measurement.
    PlateauNotReached { spread: f64 },
}

impl std::fmt::Display for CurvatureResolutionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TooFewRows { rows } => write!(
                f,
                "curvature resolution needs at least 4 ladder rows, got {rows}"
            ),
            Self::NonFiniteInput => {
                write!(f, "curvature resolution: non-finite or non-positive input")
            }
            Self::InsufficientSpan { decades } => write!(
                f,
                "curvature resolution needs the steps to span at least 2 decades \
                 to separate truncation from evaluation noise, got {decades:.2}"
            ),
            Self::TruncationNotResolved { slope } => write!(
                f,
                "curvature resolution: truncation slope {slope:.6e} is not positive, \
                 so M4 cannot be read from this ladder"
            ),
            Self::TruncationRegimeTooShort { agreeing_rows } => write!(
                f,
                "curvature resolution: only {agreeing_rows} large-step row(s) agree on \
                 2*delta/alpha^2, so there is no clean truncation regime to regress"
            ),
            Self::PlateauNotReached { spread } => write!(
                f,
                "curvature resolution: the small-step residuals still vary by \
                 {spread:.2e}x, so no evaluation-noise plateau has been reached; \
                 extend the ladder to smaller steps"
            ),
        }
    }
}

/// Meter a symmetric probe ladder for what it can resolve about a curvature.
///
/// `steps` are the `α`, `deltas` the symmetric differences `f(x±αv) − f(x)`
/// (averaged over the two signs when both were taken). Rows may be in any order.
///
/// Returns [`CurvatureResolution`], or refuses when the ladder cannot support
/// the measurement — see [`CurvatureResolutionError`].
pub fn curvature_resolution_from_ladder(
    steps: &[f64],
    deltas: &[f64],
) -> Result<CurvatureResolution, CurvatureResolutionError> {
    let rows = steps.len().min(deltas.len());
    if rows < 4 {
        return Err(CurvatureResolutionError::TooFewRows { rows });
    }
    if steps[..rows]
        .iter()
        .any(|alpha| !(alpha.is_finite() && *alpha > 0.0))
        || deltas[..rows].iter().any(|d| !d.is_finite())
    {
        return Err(CurvatureResolutionError::NonFiniteInput);
    }
    let lo = steps[..rows].iter().copied().fold(f64::INFINITY, f64::min);
    let hi = steps[..rows].iter().copied().fold(0.0_f64, f64::max);
    let decades = (hi / lo).log10();
    if !(decades >= 2.0) {
        return Err(CurvatureResolutionError::InsufficientSpan { decades });
    }

    // c(α) = 2Δ/α² ≈ vᵀHv + (M₄/12)·α².
    //
    // The regression MUST be fitted on the large-α rows only. `c` divides by
    // α², so the evaluation noise enters the per-row estimate as `2ε_f/α²` and
    // BLOWS UP as α shrinks: on a ladder with `ε_f = 1.5e-8` reaching
    // α = 3e-6, that term is ~3.3e3 against a true curvature of ~1.2e2 — the
    // small-α rows carry no curvature information at all, only amplified noise.
    // Including them does not merely add variance, it destroys the fit.
    //
    // The clean rows are the LEADING MONOTONE RUN from the largest α.
    //
    // The truncation term `(M₄/12)·α²` is monotone in α, so on truncation-
    // dominated rows `c(α)` decreases as α decreases. The noise term is
    // alternating in sign and growing as `α⁻²`, so the first row where `c`
    // stops decreasing is the first row where noise has overtaken drift. That
    // is a criterion with no tolerance to choose, and it encodes the physics
    // directly rather than approximating it with a percentage window.
    //
    // A relative window ("rows that agree to a few percent") does NOT work
    // here: the truncation drift can be a small fraction of `c` (7e-6 relative
    // on a realistic ladder) while the noise on a small-α row is thousands of
    // times larger and still inside a few percent. Such a window admits exactly
    // the rows that destroy the fit.
    let mut by_alpha: Vec<usize> = (0..rows).collect();
    by_alpha.sort_by(|a, b| steps[*b].partial_cmp(&steps[*a]).expect("finite steps"));
    let c_of = |index: usize| 2.0 * deltas[index] / (steps[index] * steps[index]);
    let anchor = c_of(by_alpha[0]);
    if !anchor.is_finite() {
        return Err(CurvatureResolutionError::NonFiniteInput);
    }
    let mut fit_rows: Vec<usize> = vec![by_alpha[0]];
    let mut previous = anchor;
    for &index in by_alpha.iter().skip(1) {
        let estimate = c_of(index);
        if !estimate.is_finite() || estimate >= previous {
            break;
        }
        previous = estimate;
        fit_rows.push(index);
    }
    if fit_rows.len() < 3 {
        return Err(CurvatureResolutionError::TruncationRegimeTooShort {
            agreeing_rows: fit_rows.len(),
        });
    }
    let mut xs = Vec::with_capacity(fit_rows.len());
    let mut ys = Vec::with_capacity(fit_rows.len());
    for &index in fit_rows.iter() {
        let alpha = steps[index];
        xs.push(alpha * alpha);
        ys.push(c_of(index));
    }
    let n = fit_rows.len() as f64;
    let mean_x = xs.iter().sum::<f64>() / n;
    let mean_y = ys.iter().sum::<f64>() / n;
    let mut sxx = 0.0_f64;
    let mut sxy = 0.0_f64;
    for index in 0..xs.len() {
        let dx = xs[index] - mean_x;
        sxx += dx * dx;
        sxy += dx * (ys[index] - mean_y);
    }
    if !(sxx.is_finite() && sxx > 0.0) {
        return Err(CurvatureResolutionError::NonFiniteInput);
    }
    let slope = sxy / sxx;
    let curvature = mean_y - slope * mean_x;
    // The truncation term is `+(M₄/12)α²`, so a non-positive slope means the
    // ladder never became truncation-dominated and M₄ is unavailable.
    if !(slope.is_finite() && slope > 0.0) {
        return Err(CurvatureResolutionError::TruncationNotResolved { slope });
    }
    let m4 = 12.0 * slope;

    // Residual against the fitted quadratic. Where this is flat in α it is the
    // evaluation noise; where it scales it is truncation.
    let mut order: Vec<usize> = (0..rows).collect();
    order.sort_by(|a, b| steps[*a].partial_cmp(&steps[*b]).expect("finite steps"));
    let plateau_rows = (rows / 2).max(2);
    let mut residuals = Vec::with_capacity(plateau_rows);
    for &index in order.iter().take(plateau_rows) {
        let alpha = steps[index];
        let fitted = 0.5 * curvature * alpha * alpha;
        residuals.push((deltas[index] - fitted).abs());
    }
    let plateau_lo = residuals.iter().copied().fold(f64::INFINITY, f64::min);
    let plateau_hi = residuals.iter().copied().fold(0.0_f64, f64::max);
    // A plateau is rows that agree; still-scaling rows do not. Two decades of
    // spread across the smallest steps means the noise floor is not yet visible.
    let spread = if plateau_lo > 0.0 {
        plateau_hi / plateau_lo
    } else if plateau_hi > 0.0 {
        f64::INFINITY
    } else {
        1.0
    };
    if !(spread <= 100.0) {
        return Err(CurvatureResolutionError::PlateauNotReached { spread });
    }
    // The plateau level. The median is used rather than the min so a single
    // fortuitously-cancelling row cannot report a noise floor below the truth.
    residuals.sort_by(|a, b| a.partial_cmp(b).expect("finite residuals"));
    let eps_f = residuals[residuals.len() / 2];

    let optimal_step = if eps_f > 0.0 {
        (48.0 * eps_f / m4).powf(0.25)
    } else {
        0.0
    };
    let delta_sigma_min = (2.0 / 3.0_f64.sqrt()) * (eps_f * m4).sqrt();
    Ok(CurvatureResolution {
        eps_f,
        m4,
        curvature,
        optimal_step,
        delta_sigma_min,
    })
}

#[cfg(test)]
mod curvature_resolution_tests {
    use super::{CurvatureResolutionError, curvature_resolution_from_ladder};

    /// Build a ladder from a KNOWN `(c, M₄, ε_f)` so recovery can be checked
    /// against truth rather than against another estimate.
    ///
    /// `Δ(α) = f(x±αv) − f(x) = ½·c·α² + (M₄/24)·α⁴ + O(α⁶)`, so
    /// `c(α) = 2Δ/α² = c + (M₄/12)·α²` — the drift the regression reads. The
    /// evaluation noise is a deterministic alternating `±ε_f` so the test is
    /// exact and cannot flake.
    fn ladder(c: f64, m4: f64, eps_f: f64, steps: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let deltas = steps
            .iter()
            .enumerate()
            .map(|(index, alpha)| {
                let sign = if index % 2 == 0 { 1.0 } else { -1.0 };
                0.5 * c * alpha * alpha + (m4 / 24.0) * alpha.powi(4) + sign * eps_f
            })
            .collect();
        (steps.to_vec(), deltas)
    }

    /// #2690: the ladder carries BOTH inputs to `δσ_min = (2/√3)·√(ε_f·M₄)`,
    /// and this recovers them from a construction where both are known.
    ///
    /// The `M₄` half is the part that has never been measured on any fixture:
    /// the per-row `c(α)` drift IS the truncation term, so regressing it on `α²`
    /// gives `M₄` as twelve times the slope at zero extra cost.
    #[test]
    fn recovers_known_eps_f_and_m4_from_a_synthetic_ladder_2690() {
        let (c, m4, eps_f) = (121.6_f64, 100.0_f64, 1.5e-8_f64);
        // The ladder must span BOTH regimes or the quantities are not present to
        // be recovered. Truncation beats noise only where `(M₄/12)α² > 2ε_f/α²`,
        // i.e. `α > (24ε_f/M₄)^{1/4} = 2.45e-3` for these inputs — so the four
        // large steps are the truncation regime and the four small ones the
        // plateau. A ladder that straddles that crossover badly is not a defect
        // in the meter; it is a ladder that cannot answer the question, and the
        // refusal arms below cover that case.
        let steps = [1.0e-1, 5.0e-2, 2.5e-2, 1.25e-2, 1.0e-3, 3.0e-4, 1.0e-4, 3.0e-5];
        let (steps, deltas) = ladder(c, m4, eps_f, &steps);
        let resolved = curvature_resolution_from_ladder(&steps, &deltas)
            .expect("a two-regime ladder must meter");

        eprintln!(
            "#2690 recovered: eps_f {:.6e} (true {eps_f:.6e}), M4 {:.6e} (true {m4:.6e}), \
             curvature {:.6e} (true {c:.6e}), h* {:.6e}, delta_sigma_min {:.6e}",
            resolved.eps_f, resolved.m4, resolved.curvature, resolved.optimal_step,
            resolved.delta_sigma_min
        );

        // Truncation-corrected curvature: the intercept beats any single row.
        assert!(
            (resolved.curvature - c).abs() <= 1.0e-3 * c,
            "#2690: curvature {:.6e} must recover {c:.6e}",
            resolved.curvature
        );
        // M4 from twelve times the truncation slope.
        assert!(
            (resolved.m4 - m4).abs() <= 0.05 * m4,
            "#2690: M4 {:.6e} must recover {m4:.6e}",
            resolved.m4
        );
        // The plateau IS the evaluation noise.
        assert!(
            resolved.eps_f >= 0.5 * eps_f && resolved.eps_f <= 2.0 * eps_f,
            "#2690: eps_f {:.6e} must recover {eps_f:.6e} within a factor of 2",
            resolved.eps_f
        );

        // The closed forms, checked against the recovered inputs rather than
        // re-derived: h* = (48 eps_f/M4)^(1/4) and dsigma = (2/sqrt3) sqrt(eps_f M4).
        let expected_step = (48.0 * resolved.eps_f / resolved.m4).powf(0.25);
        let expected_dsigma = (2.0 / 3.0_f64.sqrt()) * (resolved.eps_f * resolved.m4).sqrt();
        assert!((resolved.optimal_step - expected_step).abs() <= 1e-12 * expected_step);
        assert!((resolved.delta_sigma_min - expected_dsigma).abs() <= 1e-12 * expected_dsigma);

        // THE CONSEQUENCE, stated as a standing fact: `ε_f` enters under a
        // square root, so a 100x cut in evaluation error buys only 10x in
        // resolvable curvature. A gate that budgets linearly in `ε_f` is
        // miscalibrated by the square of its own expectation.
        let (_, finer) = ladder(c, m4, eps_f / 100.0, &steps);
        let finer = curvature_resolution_from_ladder(&steps, &finer)
            .expect("the finer ladder must also meter");
        let gain = resolved.delta_sigma_min / finer.delta_sigma_min;
        assert!(
            (gain - 10.0).abs() <= 1.0,
            "#2690: a 100x cut in eps_f must buy ~10x in resolvable curvature \
             (sqrt law), measured {gain:.3}x"
        );
    }

    /// Refusing beats fabricating: a resolution invented from a ladder that
    /// cannot support it is worse than none, because it is not legible as
    /// invented. Each arm names the ladder defect it detects.
    #[test]
    fn refuses_rather_than_fabricating_when_the_ladder_cannot_support_it_2690() {
        // Too few rows for a regression AND a plateau.
        let short = [1.0e-2, 1.0e-3, 1.0e-4];
        let (s, d) = ladder(121.6, 100.0, 1.5e-8, &short);
        assert!(matches!(
            curvature_resolution_from_ladder(&s, &d),
            Err(CurvatureResolutionError::TooFewRows { rows: 3 })
        ));

        // Enough rows, but under two decades of span: truncation and noise are
        // not separated, so neither term can be attributed.
        let narrow = [1.0e-3, 9.0e-4, 8.0e-4, 7.0e-4, 6.0e-4];
        let (s, d) = ladder(121.6, 100.0, 1.5e-8, &narrow);
        assert!(matches!(
            curvature_resolution_from_ladder(&s, &d),
            Err(CurvatureResolutionError::InsufficientSpan { .. })
        ));

        // A ladder that stops before reaching the noise floor. Every row is
        // truncation-dominated, so the small-step residuals are still SCALING
        // and no plateau exists to read `ε_f` from. This is the common real
        // case: the rows people discard as "below the floor" are the ones that
        // carry the measurement.
        let truncation_only = [1.0e-1, 5.0e-2, 2.5e-2, 1.25e-2, 6.25e-3, 3.125e-3];
        let (s, d) = ladder(121.6, 100.0, 0.0, &truncation_only);
        assert!(
            matches!(
                curvature_resolution_from_ladder(&s, &d),
                Err(CurvatureResolutionError::PlateauNotReached { .. })
            ),
            "#2690: a ladder with no noise floor in range must refuse, not \
             report the truncation residual as an evaluation error"
        );

        // Non-finite / non-positive steps.
        let (s, d) = ladder(121.6, 100.0, 1.5e-8, &[1.0e-2, 1.0e-3, 0.0, 1.0e-5]);
        assert!(matches!(
            curvature_resolution_from_ladder(&s, &d),
            Err(CurvatureResolutionError::NonFiniteInput)
        ));
    }
}
