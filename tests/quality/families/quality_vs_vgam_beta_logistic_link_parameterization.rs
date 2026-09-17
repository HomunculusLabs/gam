//! End-to-end OBJECTIVE quality: gam's *beta-logistic* inverse link — the exotic,
//! state-bearing variant of the sinh-arcsinh (SAS) link family that reuses the
//! same `SasLinkState` machine but parameterizes the mean through a regularized
//! incomplete-beta map instead of `asinh`/`tanh`.
//!
//! OBJECTIVE METRICS THIS TEST ASSERTS (none is "gam reproduces a peer tool's
//! fitted output"):
//!
//!   (A) TRUTH RECOVERY — primary quality claim. The binomial response is drawn
//!       from a *known* smooth probability surface `p_true(x1,x2) =
//!       I_{logistic(eta_true)}(a,b)`. gam fits `y ~ s(x_1d) + s(x_2d)` through the
//!       beta-logistic link and we assert the fitted success-probability surface
//!       recovers the truth: `RMSE(p_hat, p_true)` is below a principled bar set by
//!       the binomial sampling floor (the irreducible per-point noise SD of a
//!       single Bernoulli draw, averaged over the surface), not by any reference
//!       fit. The fit must also explain the data better than the constant model:
//!       its mean negative log-likelihood beats the intercept-only baseline.
//!
//!   (B) LINK-MATH CORRECTNESS vs MATHEMATICAL GROUND TRUTH — the beta-logistic
//!       inverse link is, by definition, the regularized incomplete beta function
//!       `mu(eta) = I_u(a,b)` at `u = logistic(x)`, `x = E Z + s*eta`, where
//!       `Z = logit(U)`, `U ~ Beta(a,b)`, has `E Z = digamma(a) - digamma(b)` and
//!       `s = sqrt((trigamma(a) + trigamma(b)) / (2*trigamma(1)))`, so `x` carries
//!       logit's location and scale (#2902 row 34). Its derivative is `mu'(eta) =
//!       s*dbeta(u,a,b)*u(1-u)`. Base-R `pbeta`/`dbeta`/`digamma`/`trigamma` are the
//!       *exact analytic definition* of those special functions (TOMS-708), so
//!       asserting gam's link code reproduces them is a correctness-vs-ground-truth
//!       claim, NOT a "same as a peer tool" claim. We KEEP this: gam's `mu` must
//!       equal `pbeta`, gam's analytic `d1` must equal the link-scale density
//!       `s*dbeta*u(1-u)` AND the finite difference of the CDF (catching a
//!       wrong-derivative link bug).
//!
//! VGAM's role: DEMOTED to a parameterization cross-check only. We confirm VGAM's
//! `betabinomial(size=1)` success mass equals the Beta mean `a/(a+b)` so the shape
//! map `a=exp(log_delta-eps)`, `b=exp(log_delta+eps)` is the one VGAM's family
//! uses — but gam's pass/fail never depends on matching a VGAM *fit*. The exact
//! beta special functions are base-R's; VGAM is not the source of truth here.
//! Here the spec's true `delta = 1.2`, `epsilon = 0.15` give `a = 1.2*exp(-0.15)`,
//! `b = 1.2*exp(+0.15)`.
//!
//! Requires R; a missing interpreter or package is a hard test failure, never a
//! silent skip (see `src/test_support/reference.rs`).

// True beta-logistic parameters (spec): natural-scale delta = 1.2, epsilon = 0.15.
// The link's additive shape term is `log_delta = ln(delta)`, so a = exp(log_delta
// - epsilon), b = exp(log_delta + epsilon) = 1.2*exp(-/+0.15) — the same Beta
// shapes VGAM's betabinomial mixes over.

use csv::StringRecord;
use gam::smooth::build_term_collection_design;
use gam::matrix::LinearOperator;
use gam::test_support::reference::{Column, rmse, run_r};
use gam::types::{InverseLink, LikelihoodSpec, ResponseFamily};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use gam_predict::{PredictUncertaintyOptions, predict_gamwith_uncertainty};
use gam_solve::mixture_link::beta_logistic_inverse_link_jet;
use gam_solve::model_types::FittedLinkState;
use ndarray::Array1;
use rand::{RngExt, SeedableRng, rngs::StdRng};
use statrs::distribution::{Beta, ContinuousCDF};

#[test]
fn gam_beta_logistic_link_matches_vgam_beta_parameterization() {
    init_parallelism();
    const N: usize = 400;
    let a = 1.2 * (-0.15_f64).exp();
    let b = 1.2 * 0.15_f64.exp();
    let distribution = Beta::new(a, b).expect("beta truth");
    let mut rng = StdRng::seed_from_u64(1082);
    let x: Vec<f64> = (0..N).map(|_| 10.0 * rng.random::<f64>()).collect();
    let z: Vec<f64> = (0..N).map(|_| 10.0 * rng.random::<f64>()).collect();
    let eta: Vec<f64> = x
        .iter()
        .zip(&z)
        .map(|(&x, &z)| {
            0.2 + 0.9 * ((x - 5.0) * std::f64::consts::PI / 10.0).sin()
                + 0.7 * ((z - 5.0) * std::f64::consts::PI / 12.0).cos()
        })
        .collect();
    // Generate independently of the production link kernel under test.
    let truth: Vec<f64> = eta
        .iter()
        .map(|&eta| distribution.cdf(1.0 / (1.0 + (-eta).exp())))
        .collect();
    let y: Vec<f64> = truth
        .iter()
        .map(|&p| if rng.random::<f64>() < p { 1.0 } else { 0.0 })
        .collect();
    let rows = (0..N)
        .map(|i| StringRecord::from(vec![x[i].to_string(), z[i].to_string(), y[i].to_string()]))
        .collect();
    let ds = encode_recordswith_inferred_schema(vec!["x".into(), "z".into(), "y".into()], rows)
        .expect("encode beta-logistic sample");
    let result = fit_from_formula(
        "y ~ s(x) + s(z)",
        &ds,
        &FitConfig {
            family: Some("binomial".into()),
            link: Some("beta-logistic".into()),
            ..FitConfig::default()
        },
    )
    .expect("converged beta-logistic fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected standard beta-logistic fit")
    };
    let FittedLinkState::BetaLogistic { state, .. } = &fit.fit.fitted_link else {
        panic!("formula fit must retain its estimated beta-logistic shape")
    };
    // Predictions use the ESTIMATED shape, never the generating shape applied
    // to the fitted predictor. This is the probability surface users receive.
    let fitted_family =
        LikelihoodSpec::new(ResponseFamily::Binomial, InverseLink::BetaLogistic(*state));
    let design = build_term_collection_design(ds.values.view(), &fit.resolvedspec)
        .expect("frozen prediction design");
    let offset = Array1::zeros(N);
    let prediction = predict_gamwith_uncertainty(
        design.design,
        fit.fit.beta.view(),
        offset.view(),
        fitted_family,
        &fit.fit,
        &PredictUncertaintyOptions::default(),
    )
    .expect("posterior-mean beta-logistic predictions");
    let probabilities = prediction.mean.to_vec();
    assert!(
        probabilities
            .iter()
            .all(|&p| p.is_finite() && p > 0.0 && p < 1.0)
    );
    let constant = vec![y.iter().sum::<f64>() / N as f64; N];
    let error = rmse(&probabilities, &truth);
    let constant_error = rmse(&constant, &truth);
    let noise = (truth.iter().map(|&p| p * (1.0 - p)).sum::<f64>() / N as f64).sqrt();
    let nll = |p: &[f64]| {
        y.iter()
            .zip(p)
            .map(|(&y, &p)| if y == 1.0 { -p.ln() } else { -(1.0 - p).ln() })
            .sum::<f64>()
            / N as f64
    };
    eprintln!(
        "#1082 beta-logistic: truth_rmse={error}, constant_rmse={constant_error}, nll={}, constant_nll={}",
        nll(&probabilities),
        nll(&constant)
    );
    assert!(
        error <= 0.5 * noise,
        "truth RMSE={error}, Bernoulli SD={noise}"
    );
    assert!(
        error <= 0.85 * constant_error,
        "truth RMSE={error}, constant RMSE={constant_error}"
    );
    assert!(nll(&probabilities) < nll(&constant));

    let reference = run_r(
        &[Column::new("eta", &eta)],
        r#"
        library(VGAM)
        a <- 1.2*exp(-0.15); b <- 1.2*exp(0.15)
        location <- digamma(a) - digamma(b)
        scale <- sqrt((trigamma(a) + trigamma(b)) / (2*trigamma(1)))
        mu_at <- function(eta) pbeta(plogis(location + scale*eta), a, b)
        u <- plogis(location + scale*df$eta)
        emit('mu', mu_at(df$eta))
        emit('d1', scale*dbeta(u, a, b)*u*(1-u))
        h <- 1e-5
        emit('d1_fd', (mu_at(df$eta+h)-mu_at(df$eta-h))/(2*h))
        emit('beta_mean', VGAM::dbetabinom.ab(1, size=1, shape1=a, shape2=b))
    "#,
    );
    for i in 0..N {
        let jet = beta_logistic_inverse_link_jet(eta[i], 1.2_f64.ln(), 0.15);
        assert!((jet.mu - reference.vector("mu")[i]).abs() < 1e-10);
        assert!((jet.d1 - reference.vector("d1")[i]).abs() < 1e-10);
        assert!((jet.d1 - reference.vector("d1_fd")[i]).abs() < 1e-8);
    }
    assert!((reference.vector("beta_mean")[0] - a / (a + b)).abs() < 1e-9);
}

/// #2902 row 34 (ii): the scale gauge is gone on the committed #2685 parametric
/// fixture.
///
/// Unstandardized, a flatter beta-logistic link let `β` grow while LAML's `½ log|H|`
/// fell. With the old shape bound lifted (job 1144648 arm B), the certified point had
/// max `|η|` 438.4, a worse deviance than the bounded fit (1199.699 against
/// 1198.880), and an assembly that refused. Standardized, `η` keeps logit's scale.
/// So the fitted linear predictor stays where `logistic(η)` is still resolvable in
/// f64: below `−ln(ε/2) ≈ 36.7`, past which `logistic(η)` rounds to exactly `1.0`.
/// For reference, the logit fit's max `|η|` on this fixture is 6.1, and the
/// standardized joint Bernoulli maximum is 5.59 (job 1147546).
#[test]
fn beta_logistic_parametric_fit_keeps_logit_scale_2902() {
    init_parallelism();
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("crates/gam-cli/tests/fixtures/bug_hunt_sas_link_cap_guard.csv");
    let mut reader = csv::Reader::from_path(&path).expect("read the #2685 fixture");
    let headers: Vec<String> = reader
        .headers()
        .expect("fixture headers")
        .iter()
        .map(|h| h.to_string())
        .collect();
    let rows: Vec<StringRecord> = reader.records().map(|r| r.expect("fixture row")).collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode the #2685 fixture");
    let result = fit_from_formula(
        "y ~ x",
        &ds,
        &FitConfig {
            family: Some("binomial".into()),
            link: Some("beta-logistic".into()),
            ..FitConfig::default()
        },
    )
    .expect("the standardized beta-logistic fit certifies on the #2685 fixture");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard beta-logistic fit")
    };
    assert!(
        matches!(fit.fit.fitted_link, FittedLinkState::BetaLogistic { .. }),
        "the fit must retain its estimated beta-logistic shape"
    );
    let design = build_term_collection_design(ds.values.view(), &fit.resolvedspec)
        .expect("frozen fixture design");
    let eta = design.design.apply(&fit.fit.beta);
    let max_eta = eta.iter().fold(0.0_f64, |largest, &value| largest.max(value.abs()));
    let resolvable = -(f64::EPSILON / 2.0).ln();
    assert!(
        max_eta.is_finite() && max_eta < resolvable,
        "the standardized fit's linear predictor left logit's scale: max|eta| = {max_eta} \
         against the logistic resolution edge {resolvable} (the scale gauge reached 438.4)"
    );
}
