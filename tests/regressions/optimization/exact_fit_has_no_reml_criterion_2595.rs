//! #2595 — an exactly-interpolating Gaussian fit has no REML/LAML criterion,
//! and every route must say so.
//!
//! The reported symptom was `Summary.raw_reml_score == 0.0` on a loaded model.
//! The cause was that `UnifiedFitResult` had no way to say "there is no
//! criterion here": `ensure_finite_scalar_estimation` required a finite number,
//! so the one route that KNEW the fit was exact
//! (`deterministic_gaussian_standard_fit`) wrote `0.0` to satisfy the contract.
//!
//! These tests come at that from the angles the parity test cannot reach: the
//! two in-process entry points must agree, the predicate must not fire on a fit
//! that merely fits well, the constructor must refuse the placeholder, and a
//! payload written before the absence was expressible must read back honestly.

use csv::StringRecord;
use gam::estimate::UnifiedFitResult;
use gam::inference::data::{EncodedDataset, encode_recordswith_inferred_schema};
use gam::solver::fit_orchestration::{
    FitConfig, FitResult, StandardFitResult, fit_from_formula, fit_model, materialize,
};

/// `y = 0.5 + 1.25·x` on `n` rows, plus a deterministic per-row perturbation of
/// size `noise`. At `noise == 0` the response is an EXACT affine function of the
/// design, so the unpenalized fit reproduces it to the last bit.
fn dataset(noise: f64) -> EncodedDataset {
    let n = 80usize;
    let mut rows: Vec<StringRecord> = Vec::with_capacity(n);
    for i in 0..n {
        let x = -1.0 + 2.0 * (i as f64) / ((n - 1) as f64);
        let y = 0.5 + 1.25 * x + noise * ((i % 7) as f64 - 3.0);
        rows.push(StringRecord::from(vec![x.to_string(), y.to_string()]));
    }
    encode_recordswith_inferred_schema(vec!["x".to_string(), "y".to_string()], rows)
        .expect("encode")
}

fn config() -> FitConfig {
    FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    }
}

fn standard(result: FitResult) -> StandardFitResult {
    match result {
        FitResult::Standard(fit) => fit,
        _ => panic!("expected a standard fit"),
    }
}

/// The formula entry point, which owns the deterministic-Gaussian fast-path
/// dispatch.
fn via_formula(formula: &str, noise: f64) -> StandardFitResult {
    let data = dataset(noise);
    standard(fit_from_formula(formula, &data, &config()).expect("formula fit"))
}

/// The estimator switch one level below that dispatch — the route the fast path
/// is a shortcut FOR.
fn via_fit_model(formula: &str, noise: f64) -> StandardFitResult {
    let data = dataset(noise);
    let materialized = materialize(formula, &data, &config()).expect("materialize");
    standard(fit_model(materialized.request).expect("direct fit"))
}

/// The core statement. Both routes must reach the same verdict about whether
/// the fit HAS a criterion, because they are the same estimator on the same
/// data. Before #2595 the formula route reported `0.0` and the direct route
/// `-641.1167425547742` — a value determined by the last ulp of β.
#[test]
fn both_entry_points_agree_that_an_exact_gaussian_fit_has_no_criterion() {
    for fit in [via_formula("y ~ x", 0.0), via_fit_model("y ~ x", 0.0)] {
        assert_eq!(
            fit.fit.reml_score(),
            None,
            "an exactly-interpolating Gaussian fit must report NO criterion"
        );
        assert_eq!(
            fit.fit.penalized_objective(),
            None,
            "the objective is absent together with the criterion"
        );
        assert!(
            fit.fit.at_zero_dispersion_boundary(),
            "the fit must recognize itself as sitting at the zero-dispersion boundary"
        );
        assert_eq!(
            fit.fit.standard_deviation, 0.0,
            "a residual at the arithmetic's own resolution is not a scale estimate"
        );
        // The record has to be internally consistent, not merely honest about
        // the criterion: `deviance / (n - edf)` is `sigma-hat^2`, so a deviance
        // of 1.1e-29 sitting next to a scale of exactly 0 is the same class of
        // defect one field over.
        assert_eq!(
            fit.fit.deviance, 0.0,
            "the Gaussian identity deviance IS the weighted RSS and must follow it"
        );
        assert_eq!(
            fit.fit.reported_log_likelihood(),
            None,
            "no normalized density exists at phi-hat = 0 either"
        );
        // The FIT is real and exact — this is not a refusal to fit.
        let beta = fit.fit.beta.to_vec();
        assert_eq!(beta.len(), 2);
        assert!((beta[0] - 0.5).abs() <= 1e-9, "intercept {}", beta[0]);
        assert!((beta[1] - 1.25).abs() <= 1e-9, "slope {}", beta[1]);
    }
}

/// The predicate must not fire on a fit that merely fits well. A model with
/// real residual variance keeps its criterion on both routes, and the two
/// routes agree on the value.
#[test]
fn a_fit_with_residual_variance_keeps_its_criterion_on_both_routes() {
    let formula_fit = via_formula("y ~ x", 0.01);
    let direct_fit = via_fit_model("y ~ x", 0.01);
    let formula_score = formula_fit
        .fit
        .reml_score()
        .expect("a fit with residual variance has a criterion");
    let direct_score = direct_fit
        .fit
        .reml_score()
        .expect("a fit with residual variance has a criterion");
    assert!(
        (formula_score - direct_score).abs() <= 1e-9,
        "the two entry points disagree on the criterion: formula={formula_score} \
         direct={direct_score}"
    );
    assert!(formula_fit.fit.standard_deviation > 0.0);
    assert!(!formula_fit.fit.at_zero_dispersion_boundary());
}

/// A penalized smooth on the SAME noiseless data does not interpolate — the
/// penalty keeps a real residual — so it must keep its criterion. This is the
/// over-firing guard: the boundary is about the arithmetic's resolution, not
/// about the fixture being synthetic.
#[test]
fn a_penalized_smooth_on_noiseless_data_still_has_a_criterion() {
    let fit = via_formula("y ~ s(x)", 0.0);
    let score = fit
        .fit
        .reml_score()
        .expect("a penalized smooth does not interpolate, so it has a criterion");
    assert!(score.is_finite(), "score={score}");
    assert!(
        fit.fit.standard_deviation > 0.0,
        "sigma={}",
        fit.fit.standard_deviation
    );
    assert!(!fit.fit.at_zero_dispersion_boundary());
}

/// A constant response reaches the boundary through the OTHER dispatch branch
/// (`gaussian_response_is_constant`, not `exact_unpenalized_gaussian_beta`).
/// Both branches must land on the same statement.
#[test]
fn a_constant_response_intercept_fit_also_has_no_criterion() {
    let n = 64usize;
    let rows: Vec<StringRecord> = (0..n)
        .map(|i| {
            let x = -1.0 + 2.0 * (i as f64) / ((n - 1) as f64);
            StringRecord::from(vec![x.to_string(), "3.0".to_string()])
        })
        .collect();
    let data = encode_recordswith_inferred_schema(vec!["x".to_string(), "y".to_string()], rows)
        .expect("encode");
    let fit = standard(fit_from_formula("y ~ 1", &data, &config()).expect("constant fit"));
    assert_eq!(fit.fit.reml_score(), None);
    assert!(fit.fit.at_zero_dispersion_boundary());
}

/// The invariant, stated at the one constructor every fit passes through: no
/// route may offer a criterion at the boundary. Exercised by taking a fit that
/// legitimately HAS one and re-assembling it with a zeroed scale.
#[test]
fn the_constructor_refuses_a_criterion_at_the_zero_dispersion_boundary() {
    let fit = via_formula("y ~ x", 0.01).fit;
    let serialized = serde_json::to_value(&fit).expect("serialize a fit with a criterion");
    assert!(
        serialized["reml_score"].as_f64().is_some(),
        "fixture must start with a criterion present"
    );

    // Rebuild the same fit at sigma-hat = 0 while keeping its criterion.
    let mut forged = serialized.clone();
    forged["standard_deviation"] = serde_json::json!(0.0);
    let forged: UnifiedFitResult =
        serde_json::from_value(forged).expect("the wire form itself is permissive");
    // The accessor refuses it even though the field still holds a number, so a
    // payload written before the absence was expressible cannot resurrect the
    // placeholder at any consumer.
    assert_eq!(
        forged.reml_score(),
        None,
        "a stored criterion at the boundary must not be readable as one"
    );
    assert_eq!(forged.penalized_objective(), None);
    assert_eq!(forged.reported_log_likelihood(), None);
}

/// Round-tripping a criterion-free fit through JSON preserves the absence — the
/// bug's original surface was a persisted model, so the wire form is the thing
/// that has to carry the statement.
#[test]
fn the_absent_criterion_survives_serialization() {
    let fit = via_formula("y ~ x", 0.0).fit;
    let serialized = serde_json::to_string(&fit).expect("serialize");
    assert!(
        serialized.contains("\"reml_score\":null"),
        "the wire form must carry an explicit null, not omit the key"
    );
    let restored: UnifiedFitResult = serde_json::from_str(&serialized).expect("deserialize");
    assert_eq!(restored.reml_score(), None);
    assert_eq!(restored.penalized_objective(), None);
    assert!(restored.at_zero_dispersion_boundary());
}
