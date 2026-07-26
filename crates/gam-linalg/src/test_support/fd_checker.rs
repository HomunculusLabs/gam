//! Central-difference finite-difference checking harness for tests.
//!
//! Test modules across the workspace repeatedly hand-roll the same central-
//! difference gradient check: clone the parameter vector, bump one coordinate by
//! `±eps`, evaluate a scalar objective, form `(f₊ − f₋) / (2·eps)`, and compare
//! against an analytic gradient component. This module captures the mechanical
//! shapes — a coordinate-wise scalar-objective gradient, a directional
//! derivative of a vector-valued map, a full central-difference Hessian, and the
//! matrix-level agreement assertions — behind named helpers so each call site
//! routes through one audited implementation instead of an open-coded loop.
//!
//! These helpers own no model-layer types: they are `ndarray` in, `ndarray` out.
//! That is exactly why they live in `gam-linalg` (the leaf that owns the dense
//! array seam) rather than in the model-level `gam-test-support` crate. Any
//! crate needing an FD cross-check gets it from a leaf dependency it already
//! has, instead of dragging the entire model layer into its test build.
//!
//! They are *only* for tests. They are not part of any production solver path;
//! the production outer-gradient FD audit is a different (criterion-level,
//! diagnostic-logging) facility that lives with the outer optimizer.

use ndarray::{Array1, Array2};

/// Configuration for the self-certifying [`ridders_derivative`] oracle.
///
/// The defaults are a geometric ladder `h₀ · r⁻ⁱ` with `h₀ = 1e-2`, `r = 2` and
/// 12 rungs, i.e. steps from `1e-2` down to `4.9e-6`. That span is what makes
/// the oracle scale-free: it brackets both a slowly-varying objective (whose
/// truncation only becomes negligible at the coarse end, where evaluator noise
/// is smallest) and a sharply-varying one (which needs the fine end before the
/// `O(h²)` law even starts to hold).
#[derive(Clone, Copy, Debug)]
pub struct RiddersConfig {
    /// Largest step in the ladder. Must be strictly positive.
    pub initial_step: f64,
    /// Ladder ratio: step `i` is `initial_step / shrink^i`. Must exceed 1.
    pub shrink: f64,
    /// Number of ladder rungs; the oracle costs `2 · rungs` evaluations.
    pub rungs: usize,
}

impl Default for RiddersConfig {
    fn default() -> Self {
        Self {
            initial_step: 1.0e-2,
            shrink: 2.0,
            rungs: 12,
        }
    }
}

/// A directional derivative measured together with a bound on the
/// measurement's OWN error.
///
/// A finite difference is an estimator, not a fact: its error is
/// `ν/h + h²·f‴/6`, and neither `ν` (the evaluator's absolute noise) nor `f‴`
/// is known a priori. A fixed-step oracle therefore reports a number whose
/// accuracy is unknown, and any disagreement with an analytic derivative is
/// unattributable — it may be the analytic side, or it may be the oracle. This
/// type makes the oracle's accuracy part of the answer, so a comparison can be
/// gated on `|analytic − value| > tol + uncertainty` and an oracle that cannot
/// resolve a component says so instead of manufacturing a violation.
#[derive(Clone, Debug)]
pub struct FdDerivative {
    /// Best estimate of the directional derivative.
    pub value: f64,
    /// Estimate of `|value − true derivative|`, taken from the disagreement
    /// between the two lower-order tableau entries that produced `value`.
    /// `f64::INFINITY` when no rung produced a usable entry.
    ///
    /// This is Ridders' estimate, not a proof: on an objective whose noise is
    /// *coherent* across neighbouring steps the tableau can agree better than
    /// it deserves. What it does reliably is stay LARGE when the ladder is
    /// incoherent, which is what [`FdDerivative::resolved`] keys on — so the
    /// safe reading is "small uncertainty ⇒ the ladder converged", and the
    /// unsafe one is "uncertainty is a certified error bar".
    pub uncertainty: f64,
    /// The ladder step whose column produced `value`.
    pub step: f64,
    /// Truncation order of the accepted extrapolant: `2` is a raw central
    /// difference, `4` one Richardson stage, `6` two, and so on.
    pub order: usize,
    /// The raw central differences `(h, D(h))`, coarsest first — kept so a
    /// diagnostic can print the law the gap follows without re-running.
    pub ladder: Vec<(f64, f64)>,
}

/// What a self-certifying oracle is entitled to conclude about one analytic
/// derivative component.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FdVerdict {
    /// The oracle resolved the component and the analytic value sits inside the
    /// tolerance band widened by the oracle's own uncertainty.
    Agree,
    /// The oracle resolved the component and the analytic value is outside that
    /// band. This is the only verdict that indicts the analytic derivative.
    Disagree,
    /// The oracle's own uncertainty is wider than the tolerance band, so it
    /// cannot tell agreement from disagreement. A statement about the
    /// OBJECTIVE at this point — too noisy, or too sharply curved to difference
    /// — not about the analytic gradient. Reporting it as a gradient violation
    /// is a category error.
    Unresolved,
}

impl FdDerivative {
    /// The tolerance band a component of this size is judged against:
    /// `rel_tol · max(|value|, |analytic|, abs_floor)`.
    pub fn band(&self, analytic: f64, rel_tol: f64, abs_floor: f64) -> f64 {
        rel_tol * self.value.abs().max(analytic.abs()).max(abs_floor)
    }

    /// Whether the oracle resolved the derivative sharply enough to judge the
    /// band it will judge against: `uncertainty ≤ band`.
    ///
    /// Deliberately relative to the BAND rather than to the value. An oracle
    /// that knows a `1.1e-3` derivative to `±4.8e-4` has measured something,
    /// but not to a precision that can decide a `5e-6` question, and pretending
    /// otherwise is how a conditioning limit becomes a "gradient defect".
    pub fn resolved(&self, analytic: f64, rel_tol: f64, abs_floor: f64) -> bool {
        self.uncertainty.is_finite() && self.uncertainty <= self.band(analytic, rel_tol, abs_floor)
    }

    /// The largest `|analytic − value|` this measurement is compatible with at
    /// the requested tolerance: the band widened by the oracle's own
    /// uncertainty.
    pub fn agreement_bound(&self, analytic: f64, rel_tol: f64, abs_floor: f64) -> f64 {
        self.band(analytic, rel_tol, abs_floor) + self.uncertainty
    }

    /// The single place that decides whether an analytic derivative component
    /// agrees with this measurement. Callers should route every comparison
    /// through it rather than re-deriving the three-way rule, which is easy to
    /// state as two-way and thereby convert every unmeasurable component into a
    /// false violation.
    pub fn judge(&self, analytic: f64, rel_tol: f64, abs_floor: f64) -> FdVerdict {
        if !self.value.is_finite() || !self.resolved(analytic, rel_tol, abs_floor) {
            return FdVerdict::Unresolved;
        }
        if (analytic - self.value).abs() > self.agreement_bound(analytic, rel_tol, abs_floor) {
            FdVerdict::Disagree
        } else {
            FdVerdict::Agree
        }
    }

    /// The ladder rendered for a diagnostic line: `h=… D=…` coarsest first.
    pub fn ladder_report(&self) -> String {
        self.ladder
            .iter()
            .map(|(h, d)| format!("h={h:.2e} D={d:+.10e}"))
            .collect::<Vec<_>>()
            .join("  ")
    }
}

/// Self-certifying numerical derivative of `f` at `t = 0` (Ridders' method).
///
/// `f` must evaluate the objective along the probe line, i.e. `f(t)` is the
/// objective at `base + t · direction`; the returned value estimates `f′(0)`.
///
/// The method evaluates central differences on a shrinking geometric ladder and
/// runs a Neville extrapolation across it, so column `j` of the tableau has
/// truncation order `2(j+1)`. It then accepts the tableau entry whose two
/// parents agree most closely, and reports that agreement as the uncertainty.
/// This is the standard cure for the fact that no single step is right for
/// every objective: shrinking `h` trades truncation (`h²·f‴/6`) for noise
/// (`ν/h`), and the crossover sits wherever the objective's third derivative
/// and the evaluator's noise floor happen to put it — which, for a criterion
/// evaluated through an inner solve, moves by many orders across a probe grid.
///
/// Cost is `2 · config.rungs` evaluations of `f`; the ladder is run to the end
/// rather than exited early, because a criterion that is not yet in its
/// asymptotic `O(h²)` regime at the coarse rungs produces a non-monotone error
/// sequence there, and an early exit on the first non-improvement would accept
/// a pre-asymptotic entry.
pub fn ridders_derivative<F>(mut f: F, config: RiddersConfig) -> FdDerivative
where
    F: FnMut(f64) -> f64,
{
    assert!(
        config.initial_step > 0.0 && config.initial_step.is_finite(),
        "ridders_derivative: initial_step must be finite and positive"
    );
    assert!(
        config.shrink > 1.0 && config.shrink.is_finite(),
        "ridders_derivative: shrink must exceed 1"
    );
    assert!(config.rungs >= 2, "ridders_derivative: need at least 2 rungs");

    // `tableau[i][j]` is the order-`2(j+1)` extrapolant built from rungs
    // `i-j ..= i`. Column 0 is the raw central difference at step `h_i`.
    let mut tableau: Vec<Vec<f64>> = Vec::with_capacity(config.rungs);
    let mut ladder: Vec<(f64, f64)> = Vec::with_capacity(config.rungs);
    let mut best = FdDerivative {
        value: f64::NAN,
        uncertainty: f64::INFINITY,
        step: f64::NAN,
        order: 0,
        ladder: Vec::new(),
    };
    let ratio_sq = config.shrink * config.shrink;

    let mut h = config.initial_step;
    for i in 0..config.rungs {
        let d = (f(h) - f(-h)) / (2.0 * h);
        ladder.push((h, d));
        let mut row = vec![d];
        if i > 0 {
            // Neville across the ladder: each stage removes the leading
            // even power of `h` still present in its two parents.
            let mut factor = ratio_sq;
            for j in 1..=i {
                let left = row[j - 1];
                let up = tableau[i - 1][j - 1];
                let extrapolant = (factor * left - up) / (factor - 1.0);
                row.push(extrapolant);
                // Ridders' error estimate: an extrapolant is only as
                // trustworthy as the agreement of the two entries it cancels.
                let error = (extrapolant - left).abs().max((extrapolant - up).abs());
                if extrapolant.is_finite() && error < best.uncertainty {
                    best.value = extrapolant;
                    best.uncertainty = error;
                    best.step = h;
                    best.order = 2 * (j + 1);
                }
                factor *= ratio_sq;
            }
        }
        tableau.push(row);
        h /= config.shrink;
    }
    best.ladder = ladder;
    best
}

/// [`ridders_derivative`] applied to coordinate `coord` of a scalar objective
/// at `x`.
pub fn ridders_partial_derivative<F>(
    mut objective: F,
    x: &Array1<f64>,
    coord: usize,
    config: RiddersConfig,
) -> FdDerivative
where
    F: FnMut(&Array1<f64>) -> f64,
{
    assert!(
        coord < x.len(),
        "ridders_partial_derivative: coordinate {coord} out of range for length {}",
        x.len()
    );
    ridders_derivative(
        |t| {
            let mut probe = x.clone();
            probe[coord] += t;
            objective(&probe)
        },
        config,
    )
}

/// Central finite-difference gradient of a scalar objective at `x`.
///
/// For each coordinate `i`, returns `(f(x + eps·eᵢ) − f(x − eps·eᵢ)) / (2·eps)`.
/// `f` is evaluated `2·len(x)` times. The input slice is never mutated (each
/// evaluation operates on a fresh clone), so `f` may borrow `x`'s surroundings
/// freely.
pub fn numerical_gradient_central_diff<F>(mut f: F, x: &Array1<f64>, eps: f64) -> Array1<f64>
where
    F: FnMut(&Array1<f64>) -> f64,
{
    let mut grad = Array1::zeros(x.len());
    for i in 0..x.len() {
        let mut xp = x.clone();
        let mut xm = x.clone();
        xp[i] += eps;
        xm[i] -= eps;
        grad[i] = (f(&xp) - f(&xm)) / (2.0 * eps);
    }
    grad
}

/// Directional central finite-difference of a vector-valued map `f` at `x` along
/// `direction`: `(f(x + eps·d) − f(x − eps·d)) / (2·eps)`.
///
/// This is the shape used to validate a Hessian-vector product or a directional
/// score derivative against an analytic operator action: pass the gradient/score
/// map as `f` and the probe vector as `direction`.
pub fn directional_central_diff<F>(
    mut f: F,
    x: &Array1<f64>,
    direction: &Array1<f64>,
    eps: f64,
) -> Array1<f64>
where
    F: FnMut(&Array1<f64>) -> Array1<f64>,
{
    assert_eq!(
        x.len(),
        direction.len(),
        "directional_central_diff: x and direction must have equal length"
    );
    let xp = x + &(direction * eps);
    let xm = x - &(direction * eps);
    (f(&xp) - f(&xm)) / (2.0 * eps)
}

/// Central finite-difference Hessian of a scalar objective at `x`.
///
/// Returns the dense `n×n` matrix whose `(i, j)` entry is the symmetric
/// four-point central difference
/// `(f(x + ε·eᵢ + ε·eⱼ) − f(x + ε·eᵢ − ε·eⱼ) − f(x − ε·eᵢ + ε·eⱼ) + f(x − ε·eᵢ − ε·eⱼ)) / (4·ε²)`.
/// For `i = j` this stencil degenerates to the `2ε`-spaced second difference
/// `(f(x + 2ε·eᵢ) − 2·f(x) + f(x − 2ε·eᵢ)) / (4·ε²)`, so the same expression
/// covers the diagonal without a special case. `f` is evaluated `4·n²` times
/// and the input is never mutated.
///
/// Every `(i, j)` and `(j, i)` entry is computed independently; the stencil is
/// symmetric in `i ↔ j` up to floating-point rounding, so callers that require
/// exact symmetry should average the result with its transpose.
pub fn numerical_hessian_central_diff<F>(mut f: F, x: &Array1<f64>, eps: f64) -> Array2<f64>
where
    F: FnMut(&Array1<f64>) -> f64,
{
    let n = x.len();
    let mut hess = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            let mut pp = x.clone();
            let mut pm = x.clone();
            let mut mp = x.clone();
            let mut mm = x.clone();
            pp[i] += eps;
            pp[j] += eps;
            pm[i] += eps;
            pm[j] -= eps;
            mp[i] -= eps;
            mp[j] += eps;
            mm[i] -= eps;
            mm[j] -= eps;
            hess[[i, j]] = (f(&pp) - f(&pm) - f(&mp) + f(&mm)) / (4.0 * eps * eps);
        }
    }
    hess
}

/// Verify an analytic gradient against the central finite-difference of the
/// objective, coordinate by coordinate.
///
/// Each component must agree to `tol·(1 + |fd|)` — a mixed absolute/relative
/// bound that stays meaningful both where the gradient is `O(1)` and where it is
/// near zero. Returns `Err` naming the first failing coordinate (with both
/// values and the realized gap) so the test panic message localizes the
/// disagreement; returns `Ok(())` when every coordinate agrees.
pub fn verify_gradient_vs_fd<F>(
    objective: F,
    analytic_grad: &Array1<f64>,
    x: &Array1<f64>,
    eps: f64,
    tol: f64,
) -> Result<(), String>
where
    F: FnMut(&Array1<f64>) -> f64,
{
    if analytic_grad.len() != x.len() {
        return Err(format!(
            "verify_gradient_vs_fd: analytic gradient length {} != x length {}",
            analytic_grad.len(),
            x.len()
        ));
    }
    let fd = numerical_gradient_central_diff(objective, x, eps);
    for i in 0..x.len() {
        let bound = tol * (1.0 + fd[i].abs());
        let gap = (analytic_grad[i] - fd[i]).abs();
        if gap > bound {
            return Err(format!(
                "verify_gradient_vs_fd: coordinate {i} disagrees: analytic={:.6e}, fd={:.6e}, gap={:.3e}, tol={:.3e} (bound {:.3e})",
                analytic_grad[i], fd[i], gap, tol, bound
            ));
        }
    }
    Ok(())
}

/// Asserts that a finite difference dense matrix closely matches an analytically
/// computed directional derivative matrix, both in tolerance and in
/// component-wise sign.
pub fn assert_matrix_derivativefd(fd: &Array2<f64>, analytic: &Array2<f64>, tol: f64, label: &str) {
    assert_eq!(analytic.dim(), fd.dim(), "{} dimensions must match", label);
    for i in 0..analytic.nrows() {
        for j in 0..analytic.ncols() {
            let analytic_ij = analytic[[i, j]];
            let fd_ij = fd[[i, j]];
            let diff = (analytic_ij - fd_ij).abs();

            if analytic_ij.abs() > tol && fd_ij.abs() > tol {
                assert_eq!(
                    analytic_ij.signum(),
                    fd_ij.signum(),
                    "{} sign mismatch at ({}, {}): analytic={}, fd={}",
                    label,
                    i,
                    j,
                    analytic_ij,
                    fd_ij
                );
            }
            assert!(
                diff <= tol,
                "{} value mismatch at ({}, {}): analytic={}, fd={}, abs_diff={}, tol={}",
                label,
                i,
                j,
                analytic_ij,
                fd_ij,
                diff,
                tol
            );
        }
    }
}

/// Asserts that a finite difference dense matrix matches an analytically
/// computed directional derivative matrix to a *relative* tolerance
/// `rel_tol·(1 + |analytic|)`, plus component-wise sign agreement.
///
/// Use this (rather than the absolute-tolerance [`assert_matrix_derivativefd`])
/// when the comparison's dominant components are O(0.1–1) and the finite
/// difference is contaminated by a small, non-smooth solver channel — e.g. an
/// adaptive PIRLS stabilization ridge whose magnitude shifts discontinuously
/// across the ± FD re-solves. There the exact analytic IFT derivative (which
/// correctly excludes that solver-only ridge) and the FD disagree by a fixed
/// *fraction* of the component magnitude, not a fixed absolute amount, so an
/// absolute bound tuned for the small components is spuriously tight on the
/// large ones. The two underlying derivative channels are validated separately
/// against their own FDs, so this asserts the composite to the achievable
/// relative precision rather than weakening the per-channel checks (gam#855).
pub fn assert_matrix_derivativefd_rel(
    fd: &Array2<f64>,
    analytic: &Array2<f64>,
    rel_tol: f64,
    label: &str,
) {
    assert_eq!(analytic.dim(), fd.dim(), "{} dimensions must match", label);
    for i in 0..analytic.nrows() {
        for j in 0..analytic.ncols() {
            let analytic_ij = analytic[[i, j]];
            let fd_ij = fd[[i, j]];
            let tol = rel_tol * (1.0 + analytic_ij.abs());
            if analytic_ij.abs() > tol && fd_ij.abs() > tol {
                assert_eq!(
                    analytic_ij.signum(),
                    fd_ij.signum(),
                    "{} sign mismatch at ({}, {}): analytic={}, fd={}",
                    label,
                    i,
                    j,
                    analytic_ij,
                    fd_ij
                );
            }
            let diff = (analytic_ij - fd_ij).abs();
            assert!(
                diff <= tol,
                "{} value mismatch at ({}, {}): analytic={}, fd={}, abs_diff={}, rel_tol={}, tol={}",
                label,
                i,
                j,
                analytic_ij,
                fd_ij,
                diff,
                rel_tol,
                tol
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    /// `f(x) = ½·xᵀA x + bᵀx` with symmetric `A`, whose exact gradient is
    /// `A x + b`. Exercises all three helpers against the closed form.
    #[test]
    fn quadratic_gradient_and_directional_match_closed_form() {
        let a = array![[3.0, 0.5, -0.2], [0.5, 2.0, 0.4], [-0.2, 0.4, 1.5]];
        let b = array![0.3, -1.1, 0.7];
        let x = array![0.9, -0.4, 1.3];

        let objective = |v: &Array1<f64>| 0.5 * v.dot(&a.dot(v)) + b.dot(v);
        let analytic_grad = a.dot(&x) + &b;

        let eps = 1e-6;
        let fd = numerical_gradient_central_diff(objective, &x, eps);
        for i in 0..x.len() {
            assert_abs_diff_eq!(fd[i], analytic_grad[i], epsilon = 1e-6);
        }

        verify_gradient_vs_fd(objective, &analytic_grad, &x, eps, 1e-5)
            .expect("analytic gradient matches FD of the quadratic");

        // Directional FD of the gradient map recovers the Hessian action A·d.
        let direction = array![0.6, -0.8, 0.2];
        let grad_map = |v: &Array1<f64>| a.dot(v) + &b;
        let hvp_fd = directional_central_diff(grad_map, &x, &direction, eps);
        let hvp_exact = a.dot(&direction);
        for i in 0..direction.len() {
            assert_abs_diff_eq!(hvp_fd[i], hvp_exact[i], epsilon = 1e-6);
        }

        // Full central-difference Hessian recovers the constant curvature A.
        let hess_fd = numerical_hessian_central_diff(objective, &x, 1e-4);
        for i in 0..x.len() {
            for j in 0..x.len() {
                assert_abs_diff_eq!(hess_fd[[i, j]], a[[i, j]], epsilon = 1e-6);
            }
        }

        // The matrix-level assertions accept the agreeing pair …
        assert_matrix_derivativefd(&hess_fd, &a, 1e-5, "quadratic curvature");
        assert_matrix_derivativefd_rel(&hess_fd, &a, 1e-5, "quadratic curvature (relative)");
    }

    /// A wrong analytic gradient must be rejected with the offending coordinate
    /// named.
    #[test]
    fn verify_rejects_wrong_gradient() {
        let x = array![1.0, 2.0];
        let objective = |v: &Array1<f64>| v[0] * v[0] + v[1] * v[1];
        let exact = array![2.0, 4.0];
        verify_gradient_vs_fd(objective, &exact, &x, 1e-6, 1e-5).expect("exact gradient passes");

        let wrong = array![2.0, 4.5];
        let err = verify_gradient_vs_fd(objective, &wrong, &x, 1e-6, 1e-5)
            .expect_err("perturbed gradient must be rejected");
        assert!(
            err.contains("coordinate 1"),
            "error should name coord 1: {err}"
        );
    }

    /// The Ridders oracle reproduces a closed-form derivative to near machine
    /// precision on a benign objective, and says so with a tiny uncertainty.
    #[test]
    fn ridders_matches_closed_form_on_a_benign_objective() {
        let f = |t: f64| (2.0 + t).exp() * (1.0 + 0.3 * t).ln();
        let exact = {
            let e2: f64 = 2.0_f64.exp();
            e2 * (1.0_f64).ln() + e2 * 0.3
        };
        let measured = ridders_derivative(f, RiddersConfig::default());
        assert!(
            (measured.value - exact).abs() <= 1e-10,
            "value {:.12e} vs exact {:.12e}",
            measured.value,
            exact
        );
        assert!(
            measured.resolved(exact, 1e-8, 1e-12),
            "uncertainty {:.3e} should certify a smooth objective",
            measured.uncertainty
        );
        assert_eq!(measured.judge(exact, 1e-8, 1e-12), FdVerdict::Agree);
        assert!(
            measured.order >= 4,
            "a smooth objective should accept an extrapolated entry, got order {}",
            measured.order
        );
    }

    /// #2461, in closed form. A fixed-step central difference is wrong by
    /// `(h/s)²/6` on an objective whose characteristic scale is `s`, and that
    /// error is CONSTANT in every other parameter — so it survives a sweep of
    /// anything except `h` and reads as a formula error.
    ///
    /// `s = 1.6647e-3` and `h = 3e-4` are the realized values on the
    /// `duchon_gaussian rho1@15` ψ row of the #2425 saturation ladder, where
    /// the criterion's third derivative is `≈ −9e7` against a gradient of
    /// `−249`. They reproduce the reported `5.4e-3` relative gap to three
    /// digits from `sinc` alone. The self-certifying oracle recovers the
    /// derivative anyway, which is the entire point of it.
    #[test]
    fn ridders_survives_the_scale_a_fixed_step_cannot() {
        const SCALE: f64 = 1.664_7e-3;
        const AMPLITUDE: f64 = -0.413_86;
        let f = |t: f64| AMPLITUDE * (t / SCALE).sin();
        let exact = AMPLITUDE / SCALE;

        // The shipped fixed step is off by the sinc defect — the reported
        // constant, to three digits, from nothing but the step and the scale.
        const LADDER_STEP: f64 = 3.0e-4;
        let fixed = (f(LADDER_STEP) - f(-LADDER_STEP)) / (2.0 * LADDER_STEP);
        let fixed_rel = (fixed - exact).abs() / exact.abs();
        assert!(
            (fixed_rel - 5.4e-3).abs() < 1.0e-4,
            "fixed-step defect should reproduce the reported 5.4e-3, got {fixed_rel:.4e}"
        );

        let measured = ridders_derivative(f, RiddersConfig::default());
        let rel = (measured.value - exact).abs() / exact.abs();
        assert!(
            rel < 1e-9,
            "Ridders value {:.10e} vs exact {:.10e} (rel {rel:.3e}), uncertainty {:.3e}",
            measured.value,
            exact,
            measured.uncertainty
        );
        // And it must certify itself: the reported uncertainty has to actually
        // bound the realized error, or `resolved` is decoration.
        assert!(
            measured.uncertainty >= (measured.value - exact).abs(),
            "uncertainty {:.3e} must bound the realized error {:.3e}",
            measured.uncertainty,
            (measured.value - exact).abs()
        );
        assert!(measured.resolved(exact, 1e-6, 1e-12));
        assert_eq!(measured.judge(exact, 1e-6, 1e-12), FdVerdict::Agree);
        // A wrong analytic value at the same rung must still be indicted: the
        // widened band must not have swallowed the whole comparison.
        assert_eq!(
            measured.judge(exact * 1.01, 1e-6, 1e-12),
            FdVerdict::Disagree
        );
    }

    /// An objective whose value carries a noise floor cannot be differentiated
    /// past that floor, and the oracle must SAY so rather than return a number
    /// with an unjustified precision.
    ///
    /// This is the failure mode the #2425 ladder hits at ρ ≈ 30, where the
    /// criterion's own evaluation noise swamps a gradient that has decayed to
    /// its λ=∞ face: a fixed-step oracle there reports a confident `-1.9e-1`
    /// against an analytic `+1.3e-7` and calls it a gradient defect.
    #[test]
    fn ridders_reports_an_unresolved_component_under_evaluator_noise() {
        // Deterministic but INCOHERENT jitter: a bit-mix of the probe point, so
        // neighbouring ladder steps see unrelated perturbations exactly as an
        // inner-solve stationarity floor does. A smooth surrogate (say
        // `sin(1e7·t)`) would be differentiable and the ladder would resolve
        // *it*, which is not the situation being modelled.
        fn jitter(t: f64) -> f64 {
            let mut z = t.to_bits().wrapping_mul(0x9E37_79B9_7F4A_7C15);
            z ^= z >> 31;
            z = z.wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z ^= z >> 27;
            (z >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        }
        const SLOPE: f64 = 1.0e-7;
        const NOISE: f64 = 1.0e-9;
        let noisy = |t: f64| SLOPE * t + NOISE * jitter(t);
        let measured = ridders_derivative(noisy, RiddersConfig::default());
        assert_eq!(
            measured.judge(SLOPE, 1e-3, 1e-9),
            FdVerdict::Unresolved,
            "a noise-dominated component must not be certified: value={:.3e} uncertainty={:.3e}",
            measured.value,
            measured.uncertainty
        );

        // The discrimination has to cut both ways: the SAME objective without
        // the noise channel, at the same tolerance, must resolve and be right.
        let clean = ridders_derivative(|t| SLOPE * t, RiddersConfig::default());
        assert_eq!(
            clean.judge(SLOPE, 1e-3, 1e-9),
            FdVerdict::Agree,
            "the noise-free objective must certify: uncertainty={:.3e}",
            clean.uncertainty
        );
        assert!(
            (clean.value - SLOPE).abs() <= 1e-16,
            "clean value {:.6e} vs slope {SLOPE:.6e}",
            clean.value
        );
    }

    /// The coordinate wrapper differentiates the requested axis and only that
    /// axis.
    #[test]
    fn ridders_partial_picks_the_requested_coordinate() {
        let x = array![0.7, -1.3, 2.1];
        let objective = |v: &Array1<f64>| v[0] * v[1] + (v[2] * v[2]).sin();
        for (coord, exact) in [
            (0usize, x[1]),
            (1, x[0]),
            (2, 2.0 * x[2] * f64::cos(x[2] * x[2])),
        ] {
            let measured =
                ridders_partial_derivative(objective, &x, coord, RiddersConfig::default());
            assert!(
                (measured.value - exact).abs() <= 1e-9,
                "coord {coord}: {:.10e} vs {exact:.10e}",
                measured.value
            );
        }
    }

    /// … and reject a matrix that disagrees beyond tolerance, naming the entry
    /// so the failure localizes instead of just reporting "matrices differ".
    #[test]
    #[should_panic(expected = "identity curvature value mismatch at (1, 1)")]
    fn matrix_assert_rejects_a_disagreeing_entry() {
        let analytic = array![[1.0, 0.0], [0.0, 1.0]];
        let fd = array![[1.0, 0.0], [0.0, 1.5]];
        assert_matrix_derivativefd(&fd, &analytic, 1e-6, "identity curvature");
    }
}
