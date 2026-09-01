//! Regression for #1564 (bug 2) and #2705: saved Royston-Parmar
//! (`transformation`) survival prediction must not fail at the top of its
//! default time grid, and the surface it returns must be ONE model.
//!
//! #1564: the saved predict path drove `royston_parmar_survival_hazard_components`,
//! whose guard required a STRICTLY positive `d(log Λ)/dt` and therefore aborted
//! with `eta_t=0`. The Python prediction surface always evaluates a grid whose
//! top node sits at `max_observed_exit * (1 + 1e-6)` (see
//! `default_survival_time_grid`), so **every** transformation-RP surface
//! prediction failed on its last grid node — the user saw no survival curves at
//! all. The fix relaxed the guard to `eta_derivative >= 0` (still rejecting NaN
//! and genuinely-negative slopes) and maps a zero boundary to a zero hazard.
//! That guard is pinned directly, at the function, by
//! `royston_parmar_hazard_accepts_zero_derivative_as_flat_boundary` and
//! `royston_parmar_hazard_zero_derivative_in_saturated_tail_is_zero_not_nan`
//! (`crates/gam-models/src/survival/predict.rs`), so this fixture does not have
//! to manufacture an exactly-zero hazard node to keep it covered.
//!
//! #2705: it could not manufacture one honestly anyway. The premise that
//! "beyond its last interior knot the baseline is flat, so the hazard there is
//! 0" was a defect and not a model. The value basis saturated while the
//! derivative basis — hand-rolled from a clamped B-spline first-derivative
//! basis — returned the boundary slope, so the published surface carried a FLAT
//! `Λ(t)` beside a NONZERO `h(t)`: measured on this very fixture as
//! `Λ ≡ 5.055558` with `t·h(t) ≡ 6.26088` from `t = 285` out to `t = 2.85e6`.
//! `h = dΛ/dt`, so those two cannot both describe one model. The baseline now
//! carries Royston & Parmar's own LINEAR TAILS on both sides
//! (`ISplineBoundary::LinearTails`), which is what makes `Λ` and `h` agree and
//! what gives the classical `Λ(t) ∝ t^c` extrapolation past the observed
//! follow-up.
//!
//! So what this fixture guards is no longer "some node has hazard exactly 0" —
//! that assertion pinned the defect — but the invariant the defect violated:
//! **the reported hazard is the derivative of the reported cumulative hazard**,
//! at grid nodes inside the support and outside it, plus `S(0) = 1` (a
//! saturating lower tail put a spurious atom of failures at time zero) and a
//! tail that is not a plateau.
//!
//! Data: the UCI Heart Failure Clinical Records dataset (Chicco & Jurman, 2020;
//! CC BY 4.0) with the exact multi-smooth formula from the #1564 report, fitted
//! through the real `gam fit` path and driven through the library predict
//! surface — the exact code path the Python `model.predict(...).survival_at(grid)`
//! FFI uses.

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

/// gam-format Royston-Parmar survival fixture derived from the UCI Heart Failure
/// Clinical Records dataset (n=299). Columns mirror the #1564 bug-2 formula.
const HEART_FAILURE_CSV: &str = include_str!("../../fixtures/survival/heart_failure_rp.csv");

const SURVIVAL_FORMULA: &str = "Surv(entry, exit, event) ~ s(age) \
    + s(log_creatinine_phosphokinase) + s(ejection_fraction) + s(log_platelets) \
    + s(log_serum_creatinine) + s(serum_sodium) + linear(anaemia) \
    + linear(diabetes) + linear(high_blood_pressure) + linear(sex) + linear(smoking)";

/// Parse the fixture into (header, rows-of-cells).
fn fixture_records() -> (Vec<String>, Vec<Vec<String>>) {
    let mut reader = csv::Reader::from_reader(HEART_FAILURE_CSV.as_bytes());
    let headers: Vec<String> = reader
        .headers()
        .expect("fixture header")
        .iter()
        .map(|s| s.to_string())
        .collect();
    let rows: Vec<Vec<String>> = reader
        .records()
        .map(|r| {
            r.expect("fixture row")
                .iter()
                .map(|s| s.to_string())
                .collect()
        })
        .collect();
    (headers, rows)
}

/// Build a small predict frame from the first `k` subjects with a large `exit`
/// placeholder so the surface frame is never the binding constraint on the grid.
fn predict_dataset(
    headers: &[String],
    rows: &[Vec<String>],
    k: usize,
    big_exit: f64,
) -> EncodedDataset {
    let exit_idx = headers
        .iter()
        .position(|h| h == "exit")
        .expect("exit column");
    let event_idx = headers
        .iter()
        .position(|h| h == "event")
        .expect("event column");
    let records: Vec<StringRecord> = rows
        .iter()
        .take(k)
        .map(|row| {
            let mut cells = row.clone();
            cells[exit_idx] = format!("{big_exit:.6}");
            cells[event_idx] = "1".to_string();
            StringRecord::from(cells)
        })
        .collect();
    encode_recordswith_inferred_schema(headers.to_vec(), records).expect("encode predict rows")
}

#[test]
fn royston_parmar_saved_predict_at_grid_top_does_not_fail() {
    let (headers, rows) = fixture_records();
    let exit_idx = headers
        .iter()
        .position(|h| h == "exit")
        .expect("exit column");
    let max_exit = rows
        .iter()
        .map(|row| row[exit_idx].parse::<f64>().expect("numeric exit"))
        .fold(f64::MIN, f64::max);
    assert!(max_exit > 0.0, "fixture must have positive exit times");

    let dir = tempfile::tempdir().expect("create tempdir");
    let train_path = dir.path().join("train.csv");
    let model_path = dir.path().join("model.json");
    std::fs::write(&train_path, HEART_FAILURE_CSV).expect("write training fixture");

    // Default survival likelihood is `transformation` (Royston-Parmar) with an
    // I-spline baseline log-cumulative-hazard — exactly the #1564 configuration.
    let mut fit_cmd = Command::new(gam::gam_binary!());
    fit_cmd
        .arg("fit")
        .arg(&train_path)
        .arg(SURVIVAL_FORMULA)
        .arg("--out")
        .arg(&model_path);
    run_or_panic(fit_cmd, "gam fit multi-smooth Royston-Parmar survival");
    assert!(model_path.is_file(), "gam fit did not write {model_path:?}");

    let model = FittedModel::load_from_path(Path::new(&model_path)).expect("load saved RP model");

    // The default prediction grid (`default_survival_time_grid`): 64 linear
    // nodes from 0 to `max_exit * (1 + 1e-6)`. The top node lands in the
    // saturated I-spline regime where `d(log Λ)/dt == 0`.
    let hi = max_exit * (1.0 + 1.0e-6);
    let step = hi / 63.0;
    let grid: Vec<f64> = (0..64).map(|i| step * (i as f64)).collect();

    let dataset = predict_dataset(&headers, &rows, 6, max_exit + 5.0);
    let col_map = dataset.column_map();
    let payload = model.payload();
    let training_headers = payload.training_headers.as_ref();
    let n = dataset.values.nrows();
    let primary_offset = Array1::<f64>::zeros(n);
    let noise_offset = Array1::<f64>::zeros(n);

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

    // The core regression: predict must NOT abort with `eta_t=0` at the grid top.
    let result = predict_survival(request, SurvivalPredictionCovarianceMode::Conditional)
        .expect("RP saved predict must succeed at the default grid top (#1564)");

    assert_eq!(
        result.survival.nrows(),
        n,
        "one survival row per predict row"
    );
    assert_eq!(
        result.survival.ncols(),
        grid.len(),
        "surface covers every grid time"
    );

    let mut smallest_haz_over_positive_cum = f64::INFINITY;
    let mut nodes_with_positive_cum = 0usize;
    for r in 0..n {
        let surv: Vec<f64> = result.survival.row(r).to_vec();
        let haz: Vec<f64> = result.hazard.row(r).to_vec();
        let cum: Vec<f64> = result.cumulative_hazard.row(r).to_vec();

        assert!(
            surv.iter()
                .all(|s| s.is_finite() && (0.0..=1.0).contains(s)),
            "survival must be finite and in [0,1]: {surv:?}"
        );
        assert!(
            haz.iter().all(|h| h.is_finite() && *h >= 0.0),
            "hazard must be finite and non-negative: {haz:?}"
        );
        assert!(
            cum.iter().all(|c| c.is_finite() && *c >= 0.0),
            "cumulative hazard must be finite and non-negative here: {cum:?}"
        );
        for w in surv.windows(2) {
            assert!(
                w[1] <= w[0] + 1e-9,
                "survival must be monotone non-increasing in t: {surv:?}"
            );
        }

        for j in 0..grid.len() {
            if cum[j] > 1e-9 {
                nodes_with_positive_cum += 1;
                if haz[j] < smallest_haz_over_positive_cum {
                    smallest_haz_over_positive_cum = haz[j];
                }
            }
        }
    }

    assert!(
        nodes_with_positive_cum > 0,
        "the surface must carry at least one node with a positive cumulative \
         hazard for the checks below to mean anything"
    );

    // ---------------------------------------------------------------------
    // #2705: the published hazard IS the derivative of the published
    // cumulative hazard — inside the fitted support and outside it.
    // ---------------------------------------------------------------------
    //
    // Taken on a dedicated three-point stencil per probe time rather than on
    // the 64-node display grid, whose ~4.5-unit spacing is far too coarse to
    // difference a curved `Lambda`. The stencil is `t·(1 ± 1e-4)`, which keeps
    // the truncation error at third order in `h` and still leaves ~13
    // significant digits after the cancellation in `Λ(t+h) − Λ(t−h)`.
    let probes = [
        0.25 * max_exit,
        0.75 * max_exit,
        max_exit,
        1.5 * max_exit,
        4.0 * max_exit,
    ];
    let mut stencil: Vec<f64> = Vec::with_capacity(3 * probes.len());
    for &t in probes.iter() {
        let h = 1.0e-4 * t;
        stencil.push(t - h);
        stencil.push(t);
        stencil.push(t + h);
    }
    let stencil_request = SurvivalPredictRequest {
        model: &model,
        data: dataset.values.view(),
        col_map: &col_map,
        training_headers,
        primary_offset: &primary_offset,
        noise_offset: &noise_offset,
        time_grid: Some(&stencil),
        with_uncertainty: false,
        estimand: gam::families::survival::predict::SurvivalPredictEstimand::Plugin,
    };
    let stencil_result = predict_survival(
        stencil_request,
        SurvivalPredictionCovarianceMode::Conditional,
    )
    .expect("RP saved predict must succeed on the derivative stencil");

    let mut exterior_probes_with_hazard = 0usize;
    for (probe_index, &t) in probes.iter().enumerate() {
        let h = 1.0e-4 * t;
        let lo = 3 * probe_index;
        let mid = lo + 1;
        let hi_node = lo + 2;
        for r in 0..n {
            let cum = stencil_result.cumulative_hazard.row(r);
            let haz = stencil_result.hazard.row(r);
            let difference = (cum[hi_node] - cum[lo]) / (2.0 * h);
            let analytic = haz[mid];
            let scale = analytic.abs().max(difference.abs());
            assert!(
                (difference - analytic).abs() <= 1.0e-4 * scale.max(1.0e-12),
                "row {r} at t={t}: reported hazard {analytic:.9e} is not the derivative of \
                 the reported cumulative hazard {difference:.9e} (cum(t-h)={:.9e}, \
                 cum(t+h)={:.9e}, h={h:.3e}). A flat cumulative hazard beside a nonzero \
                 hazard is gam#2705; a rising one beside a zero hazard is its mirror image.",
                cum[lo],
                cum[hi_node]
            );
            if t > max_exit && r == 0 && analytic > 0.0 {
                exterior_probes_with_hazard += 1;
            }
        }
    }

    // Non-vacuity, and the modelling claim itself: past the observed follow-up
    // the baseline must still have a TAIL. A saturating baseline satisfies the
    // derivative check above trivially — flat cumulative hazard, zero hazard —
    // which is exactly the state gam#2705 replaced, and it is the state that
    // says nobody fails after the last observed exit time.
    assert!(
        exterior_probes_with_hazard > 0,
        "every probe past max_exit={max_exit} reported a zero hazard, i.e. the baseline \
         saturates rather than continuing at its boundary slope (gam#2705). Smallest \
         hazard over {nodes_with_positive_cum} display node(s) with cum > 1e-9 was \
         {smallest_haz_over_positive_cum:.6e}"
    );

    // ---------------------------------------------------------------------
    // #2705, the lower tail: `S(0) = 1`.
    // ---------------------------------------------------------------------
    //
    // `default_survival_time_grid` starts at `t = 0`, which the time basis
    // floors to `SURVIVAL_TIME_FLOOR = 1e-9` before taking a log. A saturating
    // baseline returns `I_k = 0` for every node below the first knot, so
    // `Lambda(t) = exp(intercept + x·beta)` — a CONSTANT positive cumulative
    // hazard all the way down to the origin, i.e. an atom of failures at time
    // zero. With the tail it is `Lambda(t) -> 0`.
    for r in 0..n {
        let survival_at_origin = result.survival[[r, 0]];
        let cum_at_origin = result.cumulative_hazard[[r, 0]];
        assert!(
            cum_at_origin <= 1.0e-6,
            "row {r}: cumulative hazard at t=0 is {cum_at_origin:.6e}; a survival model \
             cannot have failures before its own time origin (gam#2705)"
        );
        assert!(
            survival_at_origin >= 1.0 - 1.0e-6,
            "row {r}: S(0) = {survival_at_origin:.9} must be 1"
        );
    }
}
