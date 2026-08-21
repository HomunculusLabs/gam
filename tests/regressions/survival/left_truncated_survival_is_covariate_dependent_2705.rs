//! Regression for #2705's second face: a LEFT-TRUNCATED (delayed-entry)
//! Royston-Parmar fit must produce a non-degenerate, covariate-dependent
//! survival surface.
//!
//! # Why delayed entry landed in the tail
//!
//! `survival_time_knot_input` feeds the baseline's knot inference with the entry
//! times only when their range is non-degenerate; a cohort in which every row
//! shares ONE entry time contributes no range, so the knots come from the exit
//! times alone and the first knot sits at `log(min exit)`. Every entry row is
//! then strictly BELOW the first knot — in the basis's exterior — and an entry
//! time is by definition `≤` its own exit.
//!
//! Under the saturating convention the exterior of an I-spline value basis is
//! `I_k ≡ 0`, so `log Λ(entry) = intercept + xβ = log Λ(t_first_knot)`: the
//! model asserted that a subject had already accumulated the ENTIRE baseline
//! hazard up to the first observed exit time *before entering*. The likelihood
//! contribution is `exp(−(Λ(exit) − Λ(entry)))`, so that inflates `Λ(entry)`
//! to the level of the earliest exit for every row at once, and the fit answers
//! by inflating the baseline — which then swamps the covariate smooth.
//!
//! With Royston & Parmar's own linear tails (`ISplineBoundary::LinearTails`,
//! #2705) the exterior is the baseline's own affine continuation, so
//! `Λ(entry) < Λ(t_first_knot)` and the delayed-entry factor is the model's.
//!
//! # What is asserted
//!
//! The `entry == 0` arm is the control: it exercises the same data, formula and
//! predict path with no truncation, so a failure there means the harness broke
//! rather than the delayed-entry path. The `entry > 0` arm asserts the two
//! properties the degenerate surface violated — early survival for the
//! low-hazard covariate is not collapsed, and the two covariate values give
//! materially different curves.
//!
//! Mirrors `tests/bug_hunt_left_truncated_survival_predicts_degenerate_covariate_independent_survival_test.py`
//! (recorded red in `bench/gha_results/rust-test-suite/MASTER_FAILURES.md`) with
//! a deterministic generator so the fixture needs no RNG dependency.

use std::path::Path;
use std::process::Command;

use csv::StringRecord;
use gam::encode_recordswith_inferred_schema;
use gam::families::survival::predict::{
    SurvivalPredictRequest, SurvivalPredictionCovarianceMode, predict_survival,
};
use gam::inference::data::EncodedDataset;
use gam::inference::model::FittedModel;
use gam::test_support::cli_harness::run_or_panic;
use ndarray::Array1;

const N: usize = 1_200;
const X_LOW: f64 = -0.8;
const X_HIGH: f64 = 0.8;
const GRID: [f64; 3] = [0.5, 1.0, 2.0];

/// `hazard = 0.4·exp(0.9·x)` with exponential censoring at mean 5, i.e. a clear
/// covariate effect and ~65 % events. Deterministic LCG uniforms.
fn build_dataset(entry: f64, seed: u64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut state: u64 = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
    let mut next_u01 = || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (((state >> 11) as f64) / ((1u64 << 53) as f64)).clamp(1.0e-12, 1.0 - 1.0e-12)
    };
    let mut x = Vec::with_capacity(N);
    let mut exit = Vec::with_capacity(N);
    let mut event = Vec::with_capacity(N);
    for _ in 0..N {
        let xi = -1.0 + 2.0 * next_u01();
        let lam = 0.4 * (0.9 * xi).exp();
        let t_event = -next_u01().ln() / lam;
        let t_cens = -next_u01().ln() * 5.0;
        let observed = t_event.min(t_cens).max(entry + 0.11);
        x.push(xi);
        exit.push(observed);
        event.push(if t_event <= t_cens { 1.0 } else { 0.0 });
    }
    (x, exit, event)
}

fn write_training_csv(path: &Path, entry: f64, x: &[f64], exit: &[f64], event: &[f64]) {
    let mut writer = csv::Writer::from_path(path).expect("create training csv");
    writer
        .write_record(["entry", "exit", "event", "x"])
        .expect("write header");
    for i in 0..x.len() {
        writer
            .write_record([
                format!("{entry:.12}"),
                format!("{:.12}", exit[i]),
                format!("{}", event[i] as i64),
                format!("{:.12}", x[i]),
            ])
            .expect("write training row");
    }
    writer.flush().expect("flush training csv");
}

fn predict_rows() -> EncodedDataset {
    let headers = vec![
        "entry".to_string(),
        "exit".to_string(),
        "event".to_string(),
        "x".to_string(),
    ];
    let rows = vec![
        StringRecord::from(vec![
            "0.0".to_string(),
            "8.0".to_string(),
            "1".to_string(),
            format!("{X_LOW:.12}"),
        ]),
        StringRecord::from(vec![
            "0.0".to_string(),
            "8.0".to_string(),
            "1".to_string(),
            format!("{X_HIGH:.12}"),
        ]),
    ];
    encode_recordswith_inferred_schema(headers, rows).expect("encode predict rows")
}

/// `(S(grid) for x = X_LOW, S(grid) for x = X_HIGH)`.
fn fit_and_survival(entry: f64, seed: u64) -> (Vec<f64>, Vec<f64>) {
    let (x, exit, event) = build_dataset(entry, seed);
    let event_rate = event.iter().sum::<f64>() / event.len() as f64;
    assert!(
        event_rate > 0.3,
        "fixture must carry substantial mortality, got event rate {event_rate}"
    );

    let dir = tempfile::tempdir().expect("create tempdir");
    let train_path = dir.path().join("train.csv");
    let model_path = dir.path().join("model.json");
    write_training_csv(&train_path, entry, &x, &exit, &event);

    let mut fit_cmd = Command::new(gam::gam_binary!());
    fit_cmd
        .arg("fit")
        .arg(&train_path)
        .arg("Surv(entry, exit, event) ~ s(x)")
        .arg("--out")
        .arg(&model_path);
    run_or_panic(fit_cmd, "gam fit Surv(entry, exit, event) ~ s(x)");
    assert!(model_path.is_file(), "gam fit did not write {model_path:?}");

    let model = FittedModel::load_from_path(&model_path).expect("load saved survival model");
    let dataset = predict_rows();
    let col_map = dataset.column_map();
    let payload = model.payload();
    let training_headers = payload.training_headers.as_ref();
    let rows = dataset.values.nrows();
    let primary_offset = Array1::<f64>::zeros(rows);
    let noise_offset = Array1::<f64>::zeros(rows);
    let grid = GRID.to_vec();
    let request = SurvivalPredictRequest {
        model: &model,
        data: dataset.values.view(),
        col_map: &col_map,
        training_headers,
        primary_offset: &primary_offset,
        noise_offset: &noise_offset,
        time_grid: Some(&grid),
        with_uncertainty: false,
        estimand: gam::families::survival::predict::SurvivalPredictEstimand::Plugin,
    };
    let result = predict_survival(request, SurvivalPredictionCovarianceMode::Conditional)
        .expect("library survival predict");
    (
        result.survival.row(0).to_vec(),
        result.survival.row(1).to_vec(),
    )
}

#[test]
fn left_truncated_survival_is_nondegenerate_and_covariate_dependent_2705() {
    // Control: the same data, formula and predict path with no truncation, so a
    // failure here is the harness and not the delayed-entry path.
    let (control_low, control_high) = fit_and_survival(0.0, 11);
    assert!(
        control_low[0] > 0.5,
        "control (entry = 0): S_low(0.5) = {:.4} should be ~0.9; the harness is broken, \
         not the delayed-entry path. S_low = {control_low:?}",
        control_low[0]
    );
    let control_gap = control_low
        .iter()
        .zip(control_high.iter())
        .fold(0.0_f64, |worst, (lo, hi)| worst.max((lo - hi).abs()));
    assert!(
        control_gap > 0.05,
        "control (entry = 0): survival must depend on the covariate; max|Δ| = {control_gap:.4}, \
         S_low = {control_low:?}, S_high = {control_high:?}"
    );

    // The arm under test: every row shares one strictly positive entry time, so
    // every entry row sits below the first baseline knot.
    let (low, high) = fit_and_survival(0.05, 11);
    assert!(
        low[0] > 0.3,
        "left-truncated (entry = 0.05): S_low(0.5) = {:.4} collapsed (truth ~0.9). Under the \
         saturating baseline the exterior was I_k = 0, so Λ(entry) was Λ at the FIRST OBSERVED \
         EXIT and the fit answered by inflating the baseline (#2705). S_low = {low:?}",
        low[0]
    );
    let gap = low
        .iter()
        .zip(high.iter())
        .fold(0.0_f64, |worst, (lo, hi)| worst.max((lo - hi).abs()));
    assert!(
        gap > 0.05,
        "left-truncated (entry = 0.05): survival is identical across two covariate values that \
         differ in hazard by exp(0.9·1.6) = 4.2x; max|Δ| = {gap:.4}, S_low = {low:?}, \
         S_high = {high:?}"
    );
}
