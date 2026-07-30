#![cfg(test)]
//! #2273: the Firth-penalized inner solve must be a Newton solve on EVERY
//! binomial link, not only the canonical one.
//!
//! `WorkingState.hessian` is `XᵀWX + S` and deliberately omits the Jeffreys
//! coefficient Hessian `HΦ`, because the outer Laplace layer consumes `H₀` and
//! `HΦ` separately. Two consumers therefore have to fold it back in: the
//! quadratic model (`objective_hessian_quadratic_correction`, `−dᵀHΦd`) and the
//! linear system the direction is solved from
//! (`objective_hessian_matrix_correction`, subtracted from `H`).
//!
//! Only the first existed. The augmented-square-root direction solve folded
//! `HΦ` in by congruence, but that route is gated on
//! `hessian_curvature == Fisher`, which holds only for the CANONICAL logit
//! link; every non-canonical binomial link (probit, cloglog, …) resolves
//! `Observed` and fell through to a dense solve of `(XᵀWX + S + λD²) d = −g`,
//! with the Jeffreys score in `g` and no Jeffreys curvature in the matrix. That
//! is not a Newton step for any objective, and it contracts LINEARLY: measured
//! at rate 0.4937/iteration on the fixture below, reaching `‖g‖ = 4.3e-7` in 23
//! iterations, failing the exact-decrement certificate
//! (`decrement² = 3.4e-14` against a `2.7e-20` threshold) and returning
//! `StalledAtValidMinimum` — which the fit-assembly gate refuses. The refused β̂
//! was the RIGHT one, which is why the failure was so hard to read.
//!
//! These tests assert the invariant that makes the two consumers one fact: the
//! inner Firth solve converges to the same machine-precision stationarity on a
//! non-canonical link as on the canonical one, and lands on the answer an
//! independent reference computes.

#![cfg(test)]

use super::*;
use gam_problem::{
    InverseLink, LikelihoodSpec, LogSmoothingParamsView, ResponseFamily, StandardLink,
};
use gam_spec::GlmLikelihoodSpec;
use ndarray::{Array1, Array2};

/// #2273's exact-separation fixture at its smallest reported size: three rows of
/// class 0 at `x ∈ {1.0, 1.1, 1.2}` and three of class 1 at `x ∈ {10.0, 10.1,
/// 10.2}`, with the design centred and scaled the way the engine's parametric
/// column conditioning leaves it.
fn separated_probit_fixture() -> (Array2<f64>, Array1<f64>) {
    let x_raw = [1.0_f64, 1.1, 1.2, 10.0, 10.1, 10.2];
    let n = x_raw.len();
    let mean = x_raw.iter().sum::<f64>() / n as f64;
    let variance = x_raw.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n as f64;
    let sd = variance.sqrt();
    let mut x = Array2::<f64>::zeros((n, 2));
    for (row, &value) in x_raw.iter().enumerate() {
        x[[row, 0]] = 1.0;
        x[[row, 1]] = (value - mean) / sd;
    }
    let y = Array1::from(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
    (x, y)
}

/// Run one Firth-penalized, unpenalized-in-λ P-IRLS solve on the fixture and
/// report what the inner solve achieved.
fn firth_inner_solve(link: StandardLink) -> PirlsResult {
    let (x, y) = separated_probit_fixture();
    let p = x.ncols();
    let weights = Array1::<f64>::ones(y.len());
    let offset = Array1::<f64>::zeros(y.len());
    let (canonical, _) =
        gam_terms::construction::canonicalize_penalty_specs(&[], &[], p, "#2273 firth curvature")
            .expect("an empty penalty set canonicalizes");
    let config = PirlsConfig {
        likelihood: GlmLikelihoodSpec::canonical(LikelihoodSpec::new(
            ResponseFamily::Binomial,
            InverseLink::Standard(link),
        )),
        link_kind: InverseLink::Standard(link),
        max_iterations: 100,
        convergence_tolerance: 1e-12,
        firth_bias_reduction: true,
        initial_lm_lambda: None,
        arrow_schur: None,
    };
    let problem = PirlsProblem {
        x: x.clone(),
        offset: offset.view(),
        y: y.view(),
        priorweights: weights.view(),
        covariate_se: None,
        gaussian_fixed_cache: None,
        glm_first_step_gram: None,
    };
    let penalty = PenaltyConfig {
        canonical_penalties: &canonical,
        balanced_penalty_root: None,
        reparam_invariant: None,
        p,
        coefficient_lower_bounds: None,
        linear_constraints_original: None,
        penalty_shrinkage_floor: None,
        kronecker_factored: None,
    };
    let rho = Array1::<f64>::zeros(0);
    let (fit, _) = fit_model_for_fixed_rho(
        LogSmoothingParamsView::new(rho.view()).expect("an empty rho is in the strength domain"),
        problem,
        penalty,
        &config,
        None,
    )
    .expect("the Firth inner solve must not error on a 6-row separated fixture");
    fit
}

/// The cross-link invariant. A Firth solve on a NON-canonical binomial link must
/// certify inner convergence exactly as the canonical one does; the two differ
/// only in which direction-solve branch they take, and both branches have to
/// describe the Firth objective.
///
/// Before the fix: logit `Converged` in 5 iterations at `‖g‖ = 5.5e-14`, probit
/// and cloglog `StalledAtValidMinimum` after 23+ iterations of linear
/// contraction. After: all three certify, and the probit mode agrees with an
/// independent reference to 7 digits (the sibling test).
#[test]
fn firth_inner_solve_converges_on_every_binomial_link_2273() {
    let mut report = String::new();
    let mut failures = Vec::new();
    for link in [StandardLink::Logit, StandardLink::Probit, StandardLink::CLogLog] {
        let fit = firth_inner_solve(link);
        report.push_str(&format!(
            "\n  {link:?}: status={:?} iters={} |g|={:.3e} deviance={:.9e} max|eta|={:.4}",
            fit.status, fit.iteration, fit.lastgradient_norm, fit.deviance, fit.max_abs_eta,
        ));
        if !fit.status.is_converged() {
            failures.push(format!("{link:?} status {:?}", fit.status));
        }
        // A consistent Newton system on a 2-coefficient problem whose Fisher
        // information has condition number 1.0009 reaches the arithmetic floor
        // in single digits (logit, on the root route, takes 5). Contraction at
        // the pre-fix 0.4937/iteration needs ~50 steps to cover the same 8
        // orders, so this budget is what a return to a Hessian-free system would
        // trip.
        if fit.iteration > 30 {
            failures.push(format!("{link:?} took {} iterations", fit.iteration));
        }
        // Deliberately NOT a fixed gradient bound. The attainable `‖g‖` is set
        // by how precisely a minimum's LOCATION is determinable in f64 —
        // `~sqrt(eps)` relative, so `‖g‖ ~ sqrt(eps)·curvature ≈ 2e-8` here —
        // and it differs by route: the canonical link's augmented-root solve
        // never forms the cancelling coefficient-space gradient and reaches
        // 5.5e-14, while the dense route floors out near 2e-8. Asserting a
        // number below a route's own floor is the exact defect this issue is
        // about (see `exact_newton_decrement_threshold`), so the gradient is
        // reported and the CONTRACT — a certified inner mode — is asserted.
        if !fit.lastgradient_norm.is_finite() {
            failures.push(format!("{link:?} |g| is not finite"));
        }
    }
    assert!(
        failures.is_empty(),
        "#2273: the Firth inner solve does not converge alike on every binomial link \
         ({}):{report}",
        failures.join("; "),
    );
}

/// The value check, against a reference that shares no code with the engine.
///
/// `scipy.optimize.minimize(Nelder-Mead)` on `ℓ(β) + ½log|XᵀWX|` with the probit
/// Fisher weight `φ(η)²/(Φ(η)(1−Φ(η)))` gives, on this fixture,
///
/// ```text
///   beta = (−1.3e-9, 1.34683841)   max|eta| = 1.37652   deviance = 1.1201966
///   eig(I) = (1.91830, 1.92011)    cond = 1.00094
/// ```
///
/// The deviance and `max|η|` are invariant to how the design is scaled, so they
/// pin the engine's answer without assuming its coefficient frame. This is the
/// half of the claim the convergence test cannot make: that the mode P-IRLS now
/// certifies is the Firth mode and not merely a point it stopped moving from.
#[test]
fn firth_probit_inner_mode_matches_an_independent_reference_2273() {
    let fit = firth_inner_solve(StandardLink::Probit);
    // Nelder-Mead's own convergence limits the reference to ~1e-6 relative, so
    // the band is set by the reference, not by the engine.
    assert!(
        (fit.deviance - 1.1201966).abs() <= 5e-6,
        "#2273: Firth-probit deviance {:.9e} is not the independent reference 1.1201966 \
         (status {:?}, |g|={:.3e})",
        fit.deviance,
        fit.status,
        fit.lastgradient_norm,
    );
    assert!(
        (fit.max_abs_eta - 1.37652).abs() <= 1e-4,
        "#2273: Firth-probit max|eta| {:.6} is not the independent reference 1.37652",
        fit.max_abs_eta,
    );
}

/// The fold-in's sign convention, asserted where it is defined rather than
/// inferred from a fit. `objective_hessian_quadratic_correction` scores
/// `−dᵀHΦd`, so the matrix form must SUBTRACT `HΦ`; the two are one fact.
#[test]
fn objective_curvature_for_direction_subtracts_the_correction_2273() {
    use super::newton_solve::objective_curvature_for_direction;
    let hessian = ndarray::array![[4.0, 1.0], [1.0, 3.0]];
    let correction = ndarray::array![[1.0, 0.5], [0.5, 2.0]];

    let borrowed = objective_curvature_for_direction(&hessian, None)
        .expect("no correction is not an error");
    assert_eq!(borrowed.as_ref(), &hessian);
    assert!(
        matches!(borrowed, std::borrow::Cow::Borrowed(_)),
        "the uncorrected path must not allocate"
    );

    let folded = objective_curvature_for_direction(&hessian, Some(&correction))
        .expect("a conforming correction folds in");
    assert_eq!(folded.as_ref(), &ndarray::array![[3.0, 0.5], [0.5, 1.0]]);

    let mismatched = Array2::<f64>::zeros((3, 3));
    assert!(
        objective_curvature_for_direction(&hessian, Some(&mismatched)).is_err(),
        "a non-conforming correction must be refused, not broadcast"
    );
}
