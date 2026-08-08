//! Curvature resolution: the smallest curvature a gate may honestly call
//! nonzero — and the fact that it obeys **two different laws** (#2690).
//!
//! Every gate in the criterion-resolution cluster compares a **curvature**
//! against a bound. The bound is not a taste parameter and it is not one
//! formula: which law applies depends entirely on **how the curvature was
//! produced**, and the two laws differ by orders of magnitude on the same
//! problem. Picking the wrong one is not a small miscalibration — it is a
//! category error, and the reason this module exists rather than a constant.
//!
//! # Law 1 — a curvature formed by a FINITE DIFFERENCE of VALUES
//!
//! Central second difference along a unit direction `v` with step `h`:
//!
//! ```text
//!     D²f(h) = [f(x + hv) − 2f(x) + f(x − hv)] / h²
//! ```
//!
//! Two error sources, moving in opposite directions in `h`:
//!
//! * **evaluation noise** — three evaluations, each accurate only to `ε_f`
//!   (the criterion's own absolute evaluation error), so the numerator carries
//!   up to `4ε_f` and the quotient `4ε_f / h²`;
//! * **truncation** — the Taylor remainder `(h²/12)·M₄`, with `M₄` a bound on
//!   the fourth directional derivative.
//!
//! ```text
//!     |D²f(h) − vᵀHv|  ≤  4·ε_f/h²  +  (h²/12)·M₄        [`finite_difference_error_bound`]
//! ```
//!
//! Minimising over `h` — `−8ε_f·h⁻³ + M₄·h/6 = 0` — gives
//!
//! ```text
//!     h*      = (48·ε_f / M₄)^{1/4}
//!     δσ_min  = (2/√3)·√(ε_f · M₄)
//! ```
//!
//! where both terms come out equal at the optimum, as they always do at the
//! minimum of an `A/h² + B·h²` balance. **The achievable curvature resolution
//! goes as `√ε_f`, not as `ε_f`.** A bound built linearly from `ε_f` is far too
//! tight; and reducing `ε_f` by 100× buys only 10× in resolvable curvature.
//!
//! ## `ε_f` is a property of a fixture, never of the codebase
//!
//! `ε_f` is the evaluation error of *a criterion, on a dataset, at a `ρ`*.
//! Measured values on this project's own fixtures span three orders
//! (`1.5e-8` and `3.79e-5`). **It must be measured per fixture and must never
//! be carried between them** — the error that motivated #2690 was exactly a
//! borrowed `ε_f`. There is therefore deliberately no default here: this
//! module will not manufacture an `ε_f`, and a caller with no measurement has
//! no resolution.
//!
//! `ε_f` and `M₄` both come free from any symmetric probe ladder that has
//! already been run. Differencing the ladder against a large-`α` fit of
//! `½·c·α²`, the residual **scales** with `α` where it is truncation and goes
//! **flat** where it is evaluation noise — a term that does not scale with the
//! step is not a step effect, and that plateau is `ε_f`. The symmetric average
//! `c̄(α) = [f(x+αv) + f(x−αv) − 2f(x)]/α² = vᵀHv + (α²/12)·M₄` then carries
//! `M₄` as 12× the slope against `α²`. Only the **symmetric** average does:
//! a one-sided column carries an odd `2(g·v)/α` term that dominates and has
//! been misread as truncation.
//!
//! # Law 2 — a curvature formed ANALYTICALLY
//!
//! The `1/h²` amplification of Law 1 does not exist for a Hessian that was
//! coded and evaluated directly, and `ε_f` — a statement about the accuracy of
//! the *value* — bounds the error of a separately-coded second derivative not
//! at all. Assuming otherwise is the same category error as comparing a
//! gradient against an eigenvalue.
//!
//! What bounds an analytically-formed eigenvalue is Weyl's inequality: for
//! symmetric `H` and any symmetric perturbation `δH`,
//!
//! ```text
//!     |σ_i(H + δH) − σ_i(H)|  ≤  ‖δH‖₂
//! ```
//!
//! so the resolution of `σ_i` **is** `‖δH‖₂`, the Hessian's own error. There is
//! **no propagation constant at all** — [`CurvatureResolution::analytic_weyl`]
//! returns its argument unchanged, and that is the whole content of the law.
//! The work moves entirely into obtaining `‖δH‖₂`, which is measured, never
//! assumed: an eigensolver's computed residual `max_i ‖H v_i − σ_i v_i‖₂` is a
//! certified `‖δH‖₂` for the solver's own backward error, and a Hessian's
//! reproducibility across perturbations that must leave it invariant is a
//! measured `‖δH‖₂` for the assembly's error. Those two answer different
//! questions — *"given this matrix, how wrong is σ?"* versus *"how wrong is
//! this matrix?"* — and a site that needs the second must not be handed the
//! first.
//!
//! # Why this is a type and not two functions returning `f64`
//!
//! A bare `f64` resolution loses the one fact a caller most needs: which law
//! produced it. [`CurvatureResolution`] carries its [`CurvatureLaw`], so the
//! choice is made at the construction site, is visible at the comparison site,
//! and cannot be inherited by accident from a neighbouring call.
//!
//! # Every input, classified
//!
//! | input | class |
//! |---|---|
//! | `ε` (`f64::EPSILON`) | **machine** |
//! | `2/√3`, `48`, `4`, `1/12` | **derived** — the Taylor remainder and the `A/h² + B·h²` optimum; no freedom, and gated numerically by this module's tests against [`finite_difference_error_bound`] itself |
//! | `ε_f` | **measured, per fixture** — never defaulted, never borrowed |
//! | `M₄` | **measured, per fixture** — from the symmetric average of a probe ladder |
//! | `‖δH‖₂` | **measured, per site** — the analytic law supplies no value for it |
//!
//! # What this module deliberately does NOT do
//!
//! It does not widen anybody's floor. In particular a gradient term
//! `Σ_k |g_k| v_k²` appearing in a ρ-space curvature gate is **not** a
//! mis-derived resolution in need of this module: for `ρ = log λ` the identity
//! `H_ρ = diag(λ)·H_λ·diag(λ) + diag(g_ρ)` makes it one exact term of a chain
//! rule, and the resolution question belongs to the *other* term. Adding a
//! resolution to it would widen a bar that was never a resolution bar.

/// Which law produced a [`CurvatureResolution`].
///
/// Carried on the value so that a comparison site can see — and a reviewer can
/// grep — which of the two laws a floor came from. The two are not
/// interchangeable and are routinely orders apart on the same problem.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CurvatureLaw {
    /// The curvature was formed as a central second difference of criterion
    /// **values**, so its error is `4ε_f/h² + (h²/12)·M₄` and its best
    /// achievable resolution is `(2/√3)·√(ε_f·M₄)` at `h = (48ε_f/M₄)^{1/4}`.
    FiniteDifferenceOfValues,
    /// The curvature was formed **analytically**. There is no step and no
    /// `1/h²` amplification; by Weyl the resolution is `‖δH‖₂`, the Hessian's
    /// own error, and nothing else.
    AnalyticWeyl,
}

impl CurvatureLaw {
    /// Human-readable name, for refusal messages that must say which law they
    /// judged by.
    pub fn label(self) -> &'static str {
        match self {
            Self::FiniteDifferenceOfValues => "finite-difference (sqrt(eps_f*M4))",
            Self::AnalyticWeyl => "analytic (Weyl, ||dH||_2)",
        }
    }
}

/// A rejected attempt to state a curvature resolution.
///
/// Every variant is a refusal to manufacture a bound from an input that cannot
/// carry one, rather than a silently substituted default.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CurvatureResolutionError {
    /// A finite-difference resolution was requested with an evaluation error
    /// that is not a positive finite number. There is no default `ε_f`: it is
    /// fixture-specific and must be measured on the fixture in hand.
    EvaluationError(f64),
    /// A finite-difference resolution was requested with a fourth-derivative
    /// bound that is not a positive finite number.
    FourthDerivativeBound(f64),
    /// An analytic resolution was requested with a Hessian error that is
    /// negative or `NaN`. Zero is admissible — an exactly-formed Hessian has
    /// `‖δH‖₂ = 0` — and `+∞` is admissible, being the honest statement that
    /// no curvature on this matrix is resolvable.
    HessianError(f64),
}

impl std::fmt::Display for CurvatureResolutionError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EvaluationError(value) => write!(
                formatter,
                "a finite-difference curvature resolution needs a positive finite evaluation \
                 error eps_f measured ON THIS FIXTURE, got {value:.6e}"
            ),
            Self::FourthDerivativeBound(value) => write!(
                formatter,
                "a finite-difference curvature resolution needs a positive finite \
                 fourth-derivative bound M4, got {value:.6e}"
            ),
            Self::HessianError(value) => write!(
                formatter,
                "an analytic curvature resolution is Weyl's ||dH||_2 and needs a non-negative \
                 (possibly infinite) Hessian error, got {value:.6e}"
            ),
        }
    }
}

impl std::error::Error for CurvatureResolutionError {}

/// The error model of a central second difference at step `h`:
/// `4·ε_f/h² + (h²/12)·M₄`.
///
/// Exposed because it is the object [`CurvatureResolution::finite_difference`]
/// minimises, so the closed forms for `h*` and `δσ_min` can be gated against
/// it numerically rather than asserted.
///
/// Returns `f64::INFINITY` for a non-positive step, which is the correct limit
/// of the noise term rather than a sentinel.
pub fn finite_difference_error_bound(
    evaluation_error: f64,
    fourth_derivative: f64,
    step: f64,
) -> f64 {
    if !step.is_finite() || step <= 0.0 {
        return f64::INFINITY;
    }
    4.0 * evaluation_error / (step * step) + step * step * fourth_derivative / 12.0
}

/// One **measured** component of `‖δH‖₂`, carrying the name of the identity
/// that measured it.
///
/// Law 2 supplies no value for `‖δH‖₂`; a site obtains it by evaluating a
/// quantity that is exactly zero in exact arithmetic and reading what came
/// back. Different identities probe different parts of the error — an
/// eigensolver residual probes the *decomposition*, an invariance residual
/// probes the *assembly* — so a component without its provenance is not
/// interpretable, and this type refuses to carry one.
///
/// See [`CurvatureResolution::analytic_weyl_from_components`] for how several
/// are combined.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MeasuredHessianError {
    /// What was measured, in words a refusal message can print — e.g.
    /// `"eigensolver backward error"`.
    pub source: &'static str,
    /// The measured value, a certified lower bound on `‖δH‖₂`.
    pub value: f64,
}

impl MeasuredHessianError {
    /// A named measured component.
    pub fn new(source: &'static str, value: f64) -> Self {
        Self { source, value }
    }
}

impl std::fmt::Display for MeasuredHessianError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}={:.6e}", self.source, self.value)
    }
}

/// The smallest curvature a comparison may treat as nonzero, together with the
/// law that produced it.
///
/// Construct with [`CurvatureResolution::finite_difference`],
/// [`CurvatureResolution::analytic_weyl`] or
/// [`CurvatureResolution::analytic_weyl_from_components`]; there is
/// deliberately no constructor that takes a bare number without naming a law.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CurvatureResolution {
    law: CurvatureLaw,
    resolution: f64,
    optimal_step: Option<f64>,
    /// Which measured component set the resolution, when it came from a set of
    /// them. `None` for the single-component and finite-difference paths.
    dominant_source: Option<&'static str>,
}

impl CurvatureResolution {
    /// Law 1: the best resolution achievable by a central second difference of
    /// values, `(2/√3)·√(ε_f·M₄)`, attained at step `(48·ε_f/M₄)^{1/4}`.
    ///
    /// `evaluation_error` is `ε_f`, the criterion's absolute evaluation error
    /// **measured on the fixture in hand**; `fourth_derivative` is `M₄`, a
    /// bound on the fourth directional derivative along the probed direction.
    /// Both are measurements. Neither has a default and neither may be
    /// inherited from another fixture.
    pub fn finite_difference(
        evaluation_error: f64,
        fourth_derivative: f64,
    ) -> Result<Self, CurvatureResolutionError> {
        if !evaluation_error.is_finite() || evaluation_error <= 0.0 {
            return Err(CurvatureResolutionError::EvaluationError(evaluation_error));
        }
        if !fourth_derivative.is_finite() || fourth_derivative <= 0.0 {
            return Err(CurvatureResolutionError::FourthDerivativeBound(
                fourth_derivative,
            ));
        }
        // h* = (48 eps_f / M4)^{1/4}: the stationary point of
        // `4 eps_f/h^2 + h^2 M4/12`, where the two terms are equal.
        let optimal_step = (48.0 * evaluation_error / fourth_derivative).sqrt().sqrt();
        // Substituting h* back: each term becomes sqrt(eps_f*M4)/sqrt(3), so
        // the sum is (2/sqrt(3))*sqrt(eps_f*M4).
        let resolution =
            (2.0 / 3.0_f64.sqrt()) * (evaluation_error * fourth_derivative).sqrt();
        Ok(Self {
            law: CurvatureLaw::FiniteDifferenceOfValues,
            resolution,
            optimal_step: Some(optimal_step),
            dominant_source: None,
        })
    }

    /// Law 2: the resolution of an analytically-formed eigenvalue is Weyl's
    /// `‖δH‖₂` and nothing else.
    ///
    /// The returned resolution is `hessian_error_2norm` unchanged. That is not
    /// a stub: the content of the analytic law is precisely that there is no
    /// propagation constant, so all of the work is in the caller's measurement
    /// of `‖δH‖₂` — and routing it through here is what records, at the
    /// comparison site, that the caller chose this law rather than Law 1.
    pub fn analytic_weyl(hessian_error_2norm: f64) -> Result<Self, CurvatureResolutionError> {
        if hessian_error_2norm.is_nan() || hessian_error_2norm < 0.0 {
            return Err(CurvatureResolutionError::HessianError(hessian_error_2norm));
        }
        Ok(Self {
            law: CurvatureLaw::AnalyticWeyl,
            resolution: hessian_error_2norm,
            optimal_step: None,
            dominant_source: None,
        })
    }

    /// Law 2 from SEVERAL measured components of `‖δH‖₂`: the resolution is
    /// their **maximum**, and the component that set it is remembered.
    ///
    /// # Why the maximum, and not the sum
    ///
    /// Each component is obtained by evaluating an identity that is exactly
    /// zero in exact arithmetic, so each is a **certified lower bound** on the
    /// true `‖δH‖₂` — never an estimate of the whole of it, and never an
    /// independent additive contribution that could be summed. The largest
    /// lower bound is the strongest fact available, and using it is the
    /// least-conservative honest choice: it widens nothing beyond what one of
    /// the measurements has already demonstrated the assembly does.
    ///
    /// # Why several are needed
    ///
    /// The components answer different questions and are routinely orders
    /// apart. An eigensolver's residual answers *"given this matrix, how wrong
    /// is `σ`?"*; it says nothing whatsoever about how wrong the matrix is, and
    /// on an assembled criterion Hessian it under-reports the truth by many
    /// orders (measured: `7.4e-16` against an assembly inconsistency of
    /// `9.9e-8` on the same fixture, #2748). A site that has only the first has
    /// not measured `‖δH‖₂`; it has measured the eigensolver.
    ///
    /// Empty input is an error rather than a zero resolution: a caller with no
    /// measurement has no resolution, which is this module's standing rule.
    pub fn analytic_weyl_from_components(
        components: &[MeasuredHessianError],
    ) -> Result<Self, CurvatureResolutionError> {
        let mut dominant: Option<MeasuredHessianError> = None;
        for component in components {
            if component.value.is_nan() || component.value < 0.0 {
                return Err(CurvatureResolutionError::HessianError(component.value));
            }
            if dominant.is_none_or(|current| component.value > current.value) {
                dominant = Some(*component);
            }
        }
        let dominant = dominant.ok_or(CurvatureResolutionError::HessianError(f64::NAN))?;
        Ok(Self {
            law: CurvatureLaw::AnalyticWeyl,
            resolution: dominant.value,
            optimal_step: None,
            dominant_source: Some(dominant.source),
        })
    }

    /// The measured component that set this resolution, when it was built from
    /// a named set of them.
    pub fn dominant_source(&self) -> Option<&'static str> {
        self.dominant_source
    }

    /// Which law produced this resolution.
    pub fn law(&self) -> CurvatureLaw {
        self.law
    }

    /// The resolution itself: curvatures of smaller magnitude are not
    /// distinguishable from zero by the route that produced them.
    pub fn resolution(&self) -> f64 {
        self.resolution
    }

    /// The step at which a central second difference attains this resolution,
    /// `(48·ε_f/M₄)^{1/4}`.
    ///
    /// `None` under [`CurvatureLaw::AnalyticWeyl`], where there is no step —
    /// which is the type-level statement that the two laws are not
    /// interchangeable.
    pub fn optimal_step(&self) -> Option<f64> {
        self.optimal_step
    }

    /// Whether a measured curvature is resolved by the route that produced it,
    /// i.e. `|curvature| > resolution`.
    ///
    /// A non-finite curvature is never resolved: it carries no magnitude to
    /// compare.
    pub fn resolves(&self, curvature: f64) -> bool {
        curvature.is_finite() && curvature.abs() > self.resolution
    }
}

impl std::fmt::Display for CurvatureResolution {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.dominant_source {
            Some(source) => write!(
                formatter,
                "{:.6e} [{}; set by {source}]",
                self.resolution,
                self.law.label()
            ),
            None => write!(formatter, "{:.6e} [{}]", self.resolution, self.law.label()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A ladder of steps spanning the interesting decades, used to check the
    /// closed forms against the error model they claim to optimise.
    fn step_ladder() -> Vec<f64> {
        let mut steps = Vec::new();
        let mut step = 1.0e-8_f64;
        while step <= 1.0e1 {
            steps.push(step);
            step *= 10.0_f64.powf(0.05);
        }
        steps
    }

    /// The closed form `h* = (48 eps_f/M4)^{1/4}` is the MINIMISER of the error
    /// model, and `(2/sqrt(3))*sqrt(eps_f*M4)` is the value there.
    ///
    /// Gated against [`finite_difference_error_bound`] by scan rather than
    /// asserted, so the `48`, the `2/sqrt(3)`, the `4` and the `1/12` are
    /// checked rather than trusted. The scan is non-vacuous by construction:
    /// the ends of the ladder are asserted to be strictly worse.
    #[test]
    fn the_closed_forms_minimise_the_error_model_they_claim_to_optimise() {
        for &(eps_f, m4) in &[
            (1.0e-16, 1.0),
            (1.5e-8, 1.47e5),
            (3.79e-5, 1.0),
            (3.79e-5, 100.0),
            (1.0e-3, 1.0e8),
        ] {
            let resolved = CurvatureResolution::finite_difference(eps_f, m4)
                .expect("positive measured inputs must yield a resolution");
            let star = resolved.optimal_step().expect("law 1 carries a step");
            let at_star = finite_difference_error_bound(eps_f, m4, star);

            assert!(
                (at_star - resolved.resolution()).abs() <= 1.0e-12 * resolved.resolution(),
                "the closed-form resolution must equal the error model at h*: \
                 model={at_star:.12e} closed={:.12e} (eps_f={eps_f:.3e}, M4={m4:.3e})",
                resolved.resolution()
            );

            let mut strictly_worse = 0usize;
            for step in step_ladder() {
                let bound = finite_difference_error_bound(eps_f, m4, step);
                assert!(
                    bound >= at_star * (1.0 - 1.0e-12),
                    "h* must minimise the error model, but h={step:.6e} gives \
                     {bound:.6e} < {at_star:.6e} (eps_f={eps_f:.3e}, M4={m4:.3e})"
                );
                if bound > at_star * 1.5 {
                    strictly_worse += 1;
                }
            }
            assert!(
                strictly_worse > 0,
                "the scan must contain steps that are materially worse than h*, \
                 or it cannot have tested minimality (eps_f={eps_f:.3e}, M4={m4:.3e})"
            );
        }
    }

    /// At `h*` the evaluation-noise term and the truncation term are equal —
    /// the property that makes the `A/h² + B·h²` optimum what it is, and the
    /// reason the resolution is `2×` one term rather than a fitted constant.
    #[test]
    fn the_two_error_terms_balance_exactly_at_the_optimal_step() {
        let (eps_f, m4) = (1.5e-8, 1.47e5);
        let resolved = CurvatureResolution::finite_difference(eps_f, m4).expect("resolution");
        let star = resolved.optimal_step().expect("law 1 carries a step");
        let noise = 4.0 * eps_f / (star * star);
        let truncation = star * star * m4 / 12.0;
        assert!(
            (noise - truncation).abs() <= 1.0e-12 * noise,
            "the two terms must balance at h*: noise={noise:.12e} truncation={truncation:.12e}"
        );
        assert!(
            (noise + truncation - resolved.resolution()).abs() <= 1.0e-12 * resolved.resolution(),
            "the resolution is the sum of the two balanced terms"
        );
    }

    /// The measured witness this law was derived on (#2690/#2665): a fixture
    /// whose evaluation floor read `eps_f = 1.5e-8` from the flat residual
    /// plateau of its alpha-ladder, and whose symmetric average fitted
    /// `M4 = 1.47e5`.
    ///
    /// Both numbers are MEASUREMENTS on one fixture and are recorded here as
    /// the law's provenance, not as defaults — nothing in this module supplies
    /// them, and transferring them to another fixture is the error #2690 was
    /// opened to stop.
    #[test]
    fn the_2665_fixture_measurements_reproduce_the_recorded_step_and_resolution() {
        let resolved = CurvatureResolution::finite_difference(1.5e-8, 1.47e5).expect("resolution");
        let star = resolved.optimal_step().expect("law 1 carries a step");
        assert!(
            (star - 1.49e-3).abs() <= 0.01 * 1.49e-3,
            "recorded h* = 1.49e-3 on the #2665 fixture, got {star:.6e}"
        );
        assert!(
            (resolved.resolution() - 5.4e-2).abs() <= 0.01 * 5.4e-2,
            "recorded delta-sigma_min = 5.4e-2 on the #2665 fixture, got {:.6e}",
            resolved.resolution()
        );
    }

    /// The analytic law has NO propagation constant: the resolution is the
    /// supplied `‖δH‖₂`, bit-identically. If this ever stops holding, the law
    /// has grown a fudge factor.
    #[test]
    fn the_analytic_law_returns_its_input_unchanged() {
        for &error in &[0.0, 7.414e-16, 1.362e-18, 9.0e-8, 1.0, f64::INFINITY] {
            let resolved = CurvatureResolution::analytic_weyl(error).expect("resolution");
            assert_eq!(
                resolved.resolution().to_bits(),
                error.to_bits(),
                "Weyl supplies no constant; the resolution IS ||dH||_2"
            );
            assert_eq!(resolved.law(), CurvatureLaw::AnalyticWeyl);
            assert!(
                resolved.optimal_step().is_none(),
                "an analytic curvature has no step"
            );
        }
    }

    /// The two laws applied to the same numeric input disagree, and the type
    /// records which was chosen. This is the confusion the module exists to
    /// prevent: `eps_f = 1e-8` read as a Weyl bound gives `1e-8`, whereas the
    /// finite-difference law with any plausible `M4` gives orders more.
    #[test]
    fn the_two_laws_are_not_interchangeable_on_the_same_number() {
        let eps_f = 1.0e-8;
        let as_weyl = CurvatureResolution::analytic_weyl(eps_f).expect("resolution");
        let as_difference = CurvatureResolution::finite_difference(eps_f, 1.0).expect("resolution");
        assert_ne!(as_weyl.law(), as_difference.law());
        assert!(
            as_difference.resolution() > 1.0e4 * as_weyl.resolution(),
            "sqrt(eps_f) is not eps_f: fd={:.6e} weyl={:.6e}",
            as_difference.resolution(),
            as_weyl.resolution()
        );
    }

    /// `resolves` fires in both directions on the numbers that motivated the
    /// issue, so the predicate is a discriminator rather than a constant.
    #[test]
    fn resolves_has_a_witness_on_both_sides() {
        let resolved = CurvatureResolution::analytic_weyl(9.0e-8).expect("resolution");
        assert!(
            resolved.resolves(-3.199e-5),
            "an eigenvalue far above ||dH||_2 must be resolved"
        );
        assert!(
            !resolved.resolves(-8.0e-9),
            "an eigenvalue below ||dH||_2 must NOT be resolved"
        );
        assert!(
            !resolved.resolves(f64::NAN),
            "a non-finite curvature carries no magnitude and cannot be resolved"
        );
    }

    /// No default `eps_f`, no default `M4`, no negative `‖δH‖₂`: every refusal
    /// is a refusal rather than a substituted constant.
    #[test]
    fn missing_or_impossible_measurements_are_refused_not_defaulted() {
        assert_eq!(
            CurvatureResolution::finite_difference(0.0, 1.0),
            Err(CurvatureResolutionError::EvaluationError(0.0))
        );
        assert_eq!(
            CurvatureResolution::finite_difference(-1.0e-8, 1.0),
            Err(CurvatureResolutionError::EvaluationError(-1.0e-8))
        );
        assert!(matches!(
            CurvatureResolution::finite_difference(f64::INFINITY, 1.0),
            Err(CurvatureResolutionError::EvaluationError(_))
        ));
        assert_eq!(
            CurvatureResolution::finite_difference(1.0e-8, 0.0),
            Err(CurvatureResolutionError::FourthDerivativeBound(0.0))
        );
        assert_eq!(
            CurvatureResolution::analytic_weyl(-1.0),
            Err(CurvatureResolutionError::HessianError(-1.0))
        );
        assert!(matches!(
            CurvatureResolution::analytic_weyl(f64::NAN),
            Err(CurvatureResolutionError::HessianError(_))
        ));
    }

    /// A step that cannot be taken has an infinite error bound rather than a
    /// negative or `NaN` one.
    #[test]
    fn a_non_positive_step_has_an_infinite_error_bound() {
        assert_eq!(finite_difference_error_bound(1.0e-8, 1.0, 0.0), f64::INFINITY);
        assert_eq!(finite_difference_error_bound(1.0e-8, 1.0, -1.0e-3), f64::INFINITY);
    }
}
