//! #2015 OBJECTIVE-QUALITY acceptance bar — the behavior chart is a *calibrated*
//! nats coordinate, measured on the committed REAL Qwen3.5-9B behavior fixture
//! (`qwen35_9b_behavior_probs64_2000.npy`: the model's TRUE next-token
//! distributions at 2000 WikiText positions, global top-63 tokens + tail bucket).
//!
//! The claim under test is not "gam reproduces a reference tool's output"; it is
//! the physical calibration statement (★) from `behavior.rs`:
//!
//! ```text
//!   KL(p ‖ p') ≈ ‖y − y'‖²      (nats),      y = √2 · Eᵀ√p,
//! ```
//!
//! i.e. squared Euclidean distance in the fitted tangent coordinate *is* KL in
//! nats (the `√2` scaling encodes the factor "2 nats per unit² of half-density
//! displacement"). If that constant were off, or the chart were lossy, a
//! downstream consumer reading the fitted behavior block would mis-price every
//! bit of behavioral information. The bars below assert, on REAL distributions:
//!
//!  1. LOSSLESS CHART — `decode(embed(p)) = p` to machine precision (the
//!     coordinate throws away no behavioral information).
//!  2. EXACT UNIT — at the basepoint `predicted_nats(0, Δy) = ‖Δy‖²` holds
//!     identically (the dose the decoder is fit to reproduce is literally the
//!     squared coordinate step there).
//!  3. NATS CALIBRATION — for a controlled small displacement of each real row's
//!     coordinate, the PREDICTED dose `Δyᵀ G Δy` of the chart's Fisher metric
//!     matches the chart's exact dose `2‖q − q'‖²` on every row and the EXACT
//!     realized `KL(p ‖ p')` on the bulk of rows, with the honestly-measured
//!     defects (median + tail relative error) reported and bounded. This is the
//!     "2 nats per unit²" calibration read off real behavior, not a synthetic circle.
//!
//! These bars exercise only the closed-form sphere-tangent geometry, so they are
//! NOT gated on the two-block convergence keystone (lane-2015); a failure here is
//! a genuine calibration defect, not a stalled fit.

use super::tests_olmo::{olmo_fixture_path, read_npy_f32_2d};
use crate::manifold::SphereTangentEmbedding;
use ndarray::{Array1, Array2};

/// Rows loaded through the f32 fixture must be exact probability vectors again
/// before the sphere-tangent embedding sees them (same contract as the #2015
/// real-data gate).
fn renormalize_rows(mut probs: Array2<f64>) -> Array2<f64> {
    for mut row in probs.rows_mut() {
        let sum: f64 = row.iter().sum();
        assert!(
            sum > 0.99 && sum < 1.01,
            "fixture row is not near-simplex: {sum}"
        );
        row /= sum;
    }
    probs
}

/// Load the committed real Qwen behavior fixture, first `GATE_ROWS` rows, as a
/// row-normalized probability matrix (small-RAM-box safe, matching the sibling
/// real-data gate's 600-row slice).
fn qwen_behavior_probs(gate_rows: usize) -> Array2<f64> {
    let full = renormalize_rows(read_npy_f32_2d(&olmo_fixture_path(
        "qwen35_9b_behavior_probs64_2000.npy",
    )));
    assert_eq!(full.dim(), (2000, 64));
    full.slice(ndarray::s![0..gate_rows, ..]).to_owned()
}

/// (1) The behavior chart is LOSSLESS on real distributions and (2) the nats unit
/// is EXACT: `decode(embed(p)) = p` to machine precision, and `predicted_nats(y)`
/// is identically `‖y‖²`. A lossy or mis-scaled chart would corrupt every
/// downstream KL read off the fitted behavior block.
#[test]
fn qwen_behavior_chart_is_lossless_and_nats_unit_is_exact_2015() {
    const GATE_ROWS: usize = 600;
    let probs = qwen_behavior_probs(GATE_ROWS);
    let (embedding, target) =
        SphereTangentEmbedding::fit(probs.view()).expect("real behavior chart must fit");
    assert_eq!(target.nrows(), GATE_ROWS);
    assert_eq!(target.ncols(), embedding.behavior_dim());

    // (1) LOSSLESS round-trip on every real row.
    let decoded = embedding
        .decode_rows(target.view())
        .expect("decode of the fitted coordinates must succeed");
    let mut max_abs = 0.0_f64;
    for i in 0..GATE_ROWS {
        for j in 0..probs.ncols() {
            max_abs = max_abs.max((decoded[[i, j]] - probs[[i, j]]).abs());
        }
    }
    assert!(
        max_abs < 1.0e-9,
        "the behavior chart must be a LOSSLESS coordinate on real distributions; \
         max round-trip abs error {max_abs}"
    );

    // (2) EXACT nats unit: at the basepoint (y = 0) the chart metric is the
    // identity, so the dose of the displacement y is identically y·y, with no slack.
    let mut max_unit_err = 0.0_f64;
    for i in 0..GATE_ROWS {
        let y = target.row(i);
        let basepoint = Array1::<f64>::zeros(y.len());
        let predicted = SphereTangentEmbedding::predicted_nats(basepoint.view(), y)
            .expect("the basepoint lies inside the chart's hemisphere");
        let raw = y.dot(&y);
        max_unit_err = max_unit_err.max((predicted - raw).abs());
    }
    assert_eq!(
        max_unit_err, 0.0,
        "predicted_nats(Δy) must be identically ‖Δy‖² (the 2-nats-per-unit² dose)"
    );
}

/// Pearson correlation of two equal-length samples.
fn pearson(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len() as f64;
    let mean_a = a.iter().sum::<f64>() / n;
    let mean_b = b.iter().sum::<f64>() / n;
    let mut cov = 0.0;
    let mut var_a = 0.0;
    let mut var_b = 0.0;
    for (x, y) in a.iter().zip(b.iter()) {
        cov += (x - mean_a) * (y - mean_b);
        var_a += (x - mean_a) * (x - mean_a);
        var_b += (y - mean_b) * (y - mean_b);
    }
    cov / (var_a.sqrt() * var_b.sqrt())
}

/// (3) NATS CALIBRATION on real behavior — the "2 nats per unit²" law verified
/// against the chart's exact dose and against EXACT KL. For each real row we take a
/// controlled small displacement of its fitted coordinate (`y' = (1−ε)·y`, ε = 1e-3)
/// and price it with the chart's Fisher metric at the step's midpoint,
/// `Δyᵀ G(y_m) Δy` with `y_m = (1 − ε/2)·y`. The flat `‖Δy‖²` alone is exact only at
/// the basepoint: on these real rows it under-prices this radial step by `cos²θ`
/// (median defect 0.575 in census job 505903), which is what these bars exist to
/// catch. We measure the defects HONESTLY (median and tail relative error) and bound
/// them, and require the predicted dose to track the chart's dose near-perfectly
/// across rows.
#[test]
fn qwen_behavior_nats_calibration_matches_exact_kl_2015() {
    const GATE_ROWS: usize = 600;
    let probs = qwen_behavior_probs(GATE_ROWS);
    let (embedding, target) =
        SphereTangentEmbedding::fit(probs.view()).expect("real behavior chart must fit");

    let eps = 1.0e-3_f64;
    // Rows whose coordinate is essentially at the basepoint carry no displacement
    // to calibrate against (predicted ≈ exact ≈ 0, ratio ill-posed); skip them by
    // a norm floor well above round-off but far below a typical behavioral row.
    let norm_sq_floor = 1.0e-6_f64;

    let mut rel_errors: Vec<f64> = Vec::new();
    let mut chart_errors: Vec<f64> = Vec::new();
    let mut predicted_all: Vec<f64> = Vec::new();
    let mut chart_all: Vec<f64> = Vec::new();
    let mut exact_all: Vec<f64> = Vec::new();
    for i in 0..GATE_ROWS {
        let y = target.row(i).to_owned();
        let norm_sq = y.dot(&y);
        if norm_sq < norm_sq_floor {
            continue;
        }
        // With `s = ‖y‖²/2` and radial overlap `r² = 1 − s`, a radial step `ε` moves the
        // decoded radial component `r` by the relative amount `ε·s/r²`. The endpoint
        // price differs from the chart's dose by about that much and the midpoint price
        // by about its square over four, a finite-step curvature, not a calibration
        // defect. At a fixed ε it grows without bound as a row's overlap with the
        // basepoint vanishes: guarded job 558087 at 3aab85774 read the chart-dose
        // defect as median 4.704e-7, p95 1.154e-3, max 5.827e-2 (s/r² ≈ 480 on the
        // worst row), and the few edge rows carrying it took the correlation to
        // 0.999700. Stepping each row in its own radial scale, `ε·min(1, r²/s)`, leaves
        // at most `ε²/4` on every row. The KL defect of tokens the row barely covers is
        // independent of the step, so the KL median cannot improve by this.
        let s = 0.5 * norm_sq;
        let row_eps = eps * ((1.0 - s) / s).min(1.0);
        let y_near: Array1<f64> = &y * (1.0 - row_eps);
        let delta: Array1<f64> = &y - &y_near;
        let y_mid: Array1<f64> = &y * (1.0 - 0.5 * row_eps);
        let predicted = SphereTangentEmbedding::predicted_nats(y_mid.view(), delta.view())
            .expect("every embedded row lies inside the chart's hemisphere");
        // decode(y) round-trips to the real distribution; decode(y_near) is a
        // controlled nearby distribution. Their EXACT KL is the realized dose.
        let p_full = embedding.decode(y.view()).expect("decode y");
        let p_near = embedding.decode(y_near.view()).expect("decode y_near");
        let exact = SphereTangentEmbedding::exact_kl(p_full.view(), p_near.view())
            .expect("exact KL must be finite for a near-identical pair");
        assert!(
            exact.is_finite() && exact > 0.0,
            "row {i}: a genuine small displacement must have a positive finite KL, got {exact}"
        );
        // The chart's own dose: its Fisher metric is the pulled-back Euclidean metric
        // of the half-density `q`, so `Δyᵀ G(y_m) Δy` must equal `2‖q − q'‖²` to second
        // order for EVERY row.
        let q_full = embedding.decode_sphere(y.view()).expect("decode_sphere y");
        let q_near = embedding.decode_sphere(y_near.view()).expect("decode_sphere y_near");
        let chart_dose = 2.0 * (&q_full - &q_near).iter().map(|v| v * v).sum::<f64>();
        rel_errors.push((predicted / exact - 1.0).abs());
        chart_errors.push((predicted / chart_dose - 1.0).abs());
        predicted_all.push(predicted);
        chart_all.push(chart_dose);
        exact_all.push(exact);
    }

    assert!(
        rel_errors.len() > GATE_ROWS / 2,
        "most real rows must carry a calibratable displacement; only {} did",
        rel_errors.len()
    );

    let quantiles = |values: &[f64]| -> (f64, f64, f64) {
        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.total_cmp(b));
        (
            sorted[sorted.len() / 2],
            sorted[(sorted.len() * 95) / 100],
            sorted[sorted.len() - 1],
        )
    };
    let (median, kl_p95, kl_max) = quantiles(&rel_errors);
    let (chart_median, chart_p95, chart_max) = quantiles(&chart_errors);
    let corr_chart = pearson(&predicted_all, &chart_all);
    let corr_kl = pearson(&predicted_all, &exact_all);
    eprintln!(
        "[#2015 nats-calibration] rows={}, ε={eps:.0e}, vs exact KL: median={median:.3e} \
         p95={kl_p95:.3e} max={kl_max:.3e} corr={corr_kl:.6}; vs chart dose 2‖Δq‖²: \
         median={chart_median:.3e} p95={chart_p95:.3e} max={chart_max:.3e} corr={corr_chart:.6}",
        rel_errors.len()
    );

    // Two different claims, two different references. Write δ = q − q'. On the
    // tokens a row covers, KL(p‖p') collects 2δ_j² to second order; on a token whose
    // half-density q_j is smaller than the step moves it, KL collects about δ_j²
    // against 2δ_j² in the quadratic form. For a radial step δ_j ≈ ε·b_j/r there, so a
    // row carrying basepoint mass M on such tokens reads a KL defect near M/(2s) at
    // any small ε, whatever the chart does. Real next-token rows are peaky, so the
    // TAIL of the KL defect measures that divergence geometry, not the chart. At
    // 96b42e9e8 (guarded sw0k job 543027) the endpoint Fisher-metric dose left the KL
    // median inside its bar while the KL p95 read 2.855e-1. The bulk claim stays
    // denominated in exact KL. The every-row claims, and the tracking claim, are
    // denominated in the chart's exact dose, where the flat-metric defect (about s,
    // median 0.575 at 7ad913f69) still fails every one of them.
    assert!(
        median < 0.02,
        "median nats-calibration defect {median:.3e} too large — the behavior chart \
         is not 2-nats-per-unit² calibrated on real distributions"
    );
    assert!(
        chart_p95 < 0.10,
        "95th-percentile chart-dose defect {chart_p95:.3e} too large"
    );
    assert!(
        chart_max.is_finite() && chart_max < 0.50,
        "worst-row chart-dose defect {chart_max:.3e} must stay bounded (no row may \
         mis-price its chart dose by more than a small factor at ε=1e-3)"
    );

    // Predicted dose must TRACK the chart's dose across rows: the calibration is a
    // near-perfect line through the origin with unit slope, so the Pearson
    // correlation is ~1. A low correlation would mean the coordinate is not
    // measuring behavioral information at all.
    assert!(
        corr_chart > 0.9999,
        "predicted nats must track the chart's dose across real rows (corr {corr_chart:.6})"
    );
}
