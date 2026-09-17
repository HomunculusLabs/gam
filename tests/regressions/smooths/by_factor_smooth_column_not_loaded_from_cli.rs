//! Regression: a factor-`by` smooth `s(x, by=g)` is fittable from the `gam` CLI
//! with `g` named only through `by=`, and recovers each level's curve.
//!
//! The parser stores the `by=` grouping variable in `ParsedTerm::Smooth.options`
//! as `{"by": "g"}`, not in the smooth's `vars`, and the design builder consumes
//! it from there, so the column is genuinely required at fit time.
//!
//! The defect this test was written for: the CLI's required-column walk
//! (then gam-cli's `collect_term_column_names`) collected only a smooth's `vars` (here `["x"]`)
//! and ignored `by=`, so the CLI loaded the file with only `{x, y}` and the fit
//! aborted before any numerics:
//!
//! ```text
//!   $ gam fit data.csv 'y ~ s(x, by=g)' --out model.json
//!   error: column 'g' not found in data. Available columns: [x, y]
//!   $ gam fit data.csv 'y ~ s(x, by=g) + g' --out model.json   # workaround
//!   saved model: model.json
//! ```
//!
//! A numeric `by=z` varying-coefficient smooth had the same gap.
//! The fit's input contract is now gam-models `fit_orchestration::formula_columns`
//! / `fit_required_columns`, which walk the formula through
//! `parsed_term_column_names` (`crates/gam-terms/src/inference/formula_dsl.rs`),
//! including a smooth's `by` column.
//!
//! This test fits the by-smooth from the CLI, then predicts each level at a point
//! where the two true curves have opposite sign, and asserts the recovered
//! predictions have the correct, distinct signs.

use gam::test_support::cli_harness::{read_prediction_means, write_predict_csv_rows};
use std::path::Path;
use std::process::Command;

/// Per-level truth: `g="a"` follows `+sin(2πx)`, `g="b"` follows `-sin(2πx)`.
fn truth(x: f64, level: &str) -> f64 {
    let s = (2.0 * std::f64::consts::PI * x).sin();
    if level == "a" { s } else { -s }
}

/// Noisy training data on `x ∈ [0, 1]` for two factor levels.
fn write_training_csv(path: &Path) {
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use rand_distr::{Distribution, Normal};
    let mut rng = StdRng::seed_from_u64(7);
    let noise = Normal::new(0.0, 0.08).expect("normal");
    let mut writer = csv::Writer::from_path(path).expect("create training csv");
    writer.write_record(["x", "y", "g"]).expect("write header");
    let n = 400usize;
    for i in 0..n {
        let x = (i as f64) / ((n - 1) as f64);
        let level = if i % 2 == 0 { "a" } else { "b" };
        let y = truth(x, level) + noise.sample(&mut rng);
        writer
            .write_record([format!("{x:.10}"), format!("{y:.10}"), level.to_string()])
            .expect("write training row");
    }
    writer.flush().expect("flush training csv");
}

#[test]
fn by_factor_smooth_loads_its_by_column_and_recovers_per_level_curves() {
    let dir = tempfile::tempdir().expect("create tempdir");
    let train_path = dir.path().join("train.csv");
    let predict_path = dir.path().join("predict.csv");
    let model_path = dir.path().join("model.json");
    let out_path = dir.path().join("pred_out.csv");

    write_training_csv(&train_path);

    // Fit the by-factor smooth straight from the CLI, with `g` referenced ONLY
    // through `by=g` (no redundant `+ g`). This is the exact documented syntax
    // (`s(x, by=g)`) used throughout the test suite.
    let mut fit_cmd = Command::new(gam::gam_binary!());
    fit_cmd
        .arg("fit")
        .arg(&train_path)
        .arg("y ~ s(x, by=g)")
        .args(["--family", "gaussian"])
        .arg("--out")
        .arg(&model_path);
    let fit_out = fit_cmd.output().expect("spawn gam fit");
    assert!(
        fit_out.status.success(),
        "`gam fit 'y ~ s(x, by=g)'` failed; the required-column set must include \
         the by= column `g`.\n\
         --- stdout ---\n{}\n--- stderr ---\n{}",
        String::from_utf8_lossy(&fit_out.stdout),
        String::from_utf8_lossy(&fit_out.stderr),
    );
    assert!(model_path.is_file(), "gam fit did not write {model_path:?}");

    // x = 0.25: sin(2π·0.25) = +1, so level "a" ≈ +1 and level "b" ≈ −1.
    let probes: [(f64, &str); 2] = [(0.25, "a"), (0.25, "b")];
    write_predict_csv_rows(
        &predict_path,
        ["x", "y", "g"],
        probes
            .iter()
            .map(|&(x, g)| [format!("{x:.10}"), "0.0".to_string(), g.to_string()]),
    );

    let mut predict_cmd = Command::new(gam::gam_binary!());
    predict_cmd
        .arg("predict")
        .arg(&model_path)
        .arg(&predict_path)
        .arg("--out")
        .arg(&out_path);
    let pred_out = predict_cmd.output().expect("spawn gam predict");
    assert!(
        pred_out.status.success(),
        "`gam predict` failed for the by-factor smooth model.\n--- stderr ---\n{}",
        String::from_utf8_lossy(&pred_out.stderr),
    );

    let preds = read_prediction_means(&out_path);
    assert_eq!(preds.len(), 2, "expected one prediction per probe");
    let (pred_a, pred_b) = (preds[0], preds[1]);

    // The two levels' curves are mirror images, so at x = 0.25 they must have
    // opposite signs and both be near ±1. A loose 0.5 threshold makes this a
    // sign/identifiability check, not a precision test.
    assert!(
        pred_a > 0.5,
        "level a at x=0.25 should recover +sin(2π·0.25)=+1, got {pred_a:.4}"
    );
    assert!(
        pred_b < -0.5,
        "level b at x=0.25 should recover -sin(2π·0.25)=-1, got {pred_b:.4}"
    );
}
