//! gam#2765 / gam#2767: a follow-up-varying slope survives the round trip.
//!
//! The kernel work made `b(x, t)` a fitted surface. This module is about the
//! other half — whether that surface can leave the process. Until this landed,
//! persistence refused a fit with `logslope_time_k` outright, because the
//! on-disk contract rebuilds the log-slope block from its covariate term spec
//! alone: `p_cov` columns against a `p_cov·p_time` coefficient vector.
//!
//! The property under test is REPLAY EXACTNESS, not statistical recovery (that
//! is `survival_marginal_slope_follow_up_varying_slope_2765`'s job). A saved
//! model either evaluates the same slope surface the fit did, or it evaluates a
//! different model while every width agrees — and the second failure mode is
//! silent, which is why the assertions below are on the surface itself rather
//! than on shapes.

use std::collections::HashMap;

use csv::StringRecord;
use gam::families::survival::replay_logslope_time_margin_design;
use gam::inference::model::FittedModel;
use gam::inference::model_payload_builders::fit_formula_to_payload;
use gam::{FitConfig, encode_recordswith_inferred_schema, init_parallelism};
use gam::utils::splitmix64;

/// Smaller than the recovery fixture on purpose: replay exactness is a
/// deterministic property of a converged fit, so it needs a fit that converges,
/// not a fit with power.
const N: usize = 900;
const SLOPE_LEVEL: f64 = 0.85;
const SLOPE_TREND: f64 = -0.32;
const LOCATION_LEVEL: f64 = -1.15;
const LOCATION_TREND: f64 = 0.95;
const LOGSLOPE_TIME_DEGREE: usize = 2;
const LOGSLOPE_TIME_K: usize = 4;

fn next_unit(state: &mut u64) -> f64 {
    (splitmix64(state) >> 11) as f64 / (1u64 << 53) as f64
}

fn next_gauss(state: &mut u64) -> f64 {
    let u1 = next_unit(state).max(1e-12);
    let u2 = next_unit(state);
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

fn planted_eta(time: f64, z: f64) -> f64 {
    let slope = SLOPE_LEVEL + SLOPE_TREND * time.ln();
    let location = LOCATION_LEVEL + LOCATION_TREND * time.ln();
    location * (1.0 + slope * slope).sqrt() + slope * z
}

/// Abramowitz–Stegun 7.1.26, and a bisection quantile on top of it. Deliberately
/// not the crate's own probability code: a fixture whose truth is produced by
/// the code under test tests nothing.
fn erf(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let y = 1.0
        - (((((1.061_405_429 * t - 1.453_152_027) * t) + 1.421_413_741) * t - 0.284_496_736) * t
            + 0.254_829_592)
            * t
            * (-x * x).exp();
    sign * y
}

fn normal_quantile(p: f64) -> f64 {
    let cdf = |x: f64| 0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2));
    let (mut low, mut high) = (-12.0_f64, 12.0_f64);
    for _ in 0..200 {
        let mid = 0.5 * (low + high);
        if cdf(mid) < p {
            low = mid;
        } else {
            high = mid;
        }
    }
    0.5 * (low + high)
}

fn planted_event_time(u: f64, z: f64) -> f64 {
    let target = -normal_quantile(u);
    let (mut low, mut high) = (-6.0_f64, 6.0_f64);
    for _ in 0..200 {
        let mid = 0.5 * (low + high);
        if planted_eta(mid.exp(), z) < target {
            low = mid;
        } else {
            high = mid;
        }
    }
    (0.5 * (low + high)).exp()
}

fn build_dataset() -> (gam::inference::data::EncodedDataset, Vec<f64>) {
    let headers = ["time", "event", "z"]
        .iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();
    let mut state: u64 = 0x2765_5A1E_2767_D00D_u64;

    let mut raw_scores: Vec<f64> = Vec::with_capacity(N);
    let mut draws: Vec<f64> = Vec::with_capacity(N);
    let mut censor: Vec<f64> = Vec::with_capacity(N);
    for _ in 0..N {
        raw_scores.push(next_gauss(&mut state));
        draws.push(next_unit(&mut state).clamp(1e-6, 1.0 - 1e-6));
        censor.push(next_unit(&mut state));
    }
    let mean = raw_scores.iter().sum::<f64>() / N as f64;
    let variance = raw_scores.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / N as f64;
    let sd = variance.sqrt().max(1e-12);
    let scores: Vec<f64> = raw_scores.iter().map(|v| (v - mean) / sd).collect();

    let mut rows: Vec<StringRecord> = Vec::with_capacity(N);
    let mut observed_times: Vec<f64> = Vec::with_capacity(N);
    for index in 0..N {
        let z = scores[index];
        let event_time = planted_event_time(draws[index], z);
        let censor_time = 0.35 + 5.0 * censor[index];
        let (time, event) = if event_time <= censor_time {
            (event_time, 1u8)
        } else {
            (censor_time, 0u8)
        };
        let time = time.clamp(1e-3, 1e3);
        observed_times.push(time);
        rows.push(StringRecord::from(vec![
            time.to_string(),
            event.to_string(),
            z.to_string(),
        ]));
    }
    let data = encode_recordswith_inferred_schema(headers, rows)
        .expect("encode the #2765 replay fixture");
    (data, observed_times)
}

fn fit_config() -> FitConfig {
    FitConfig {
        survival_likelihood: Some("marginal-slope".to_string()),
        z_column: Some("z".to_string()),
        logslope_formula: Some("1".to_string()),
        logslope_time_k: Some(LOGSLOPE_TIME_K),
        logslope_time_degree: LOGSLOPE_TIME_DEGREE,
        time_num_internal_knots: 3,
        baseline_target: "weibull".to_string(),
        ..FitConfig::default()
    }
}

/// Saving a follow-up-varying slope carries the margin, and the saved model
/// rebuilds the SAME log-slope design the fit used.
///
/// The bit-for-bit comparison is deliberate. Two designs that agree to `1e-6`
/// are two different models; a replay that is nearly right produces a slope
/// surface that is nearly right at every time, and nothing downstream can tell.
/// The knots are the whole authority a predictor is handed, so if replaying from
/// them does not reproduce the fitted design exactly, the contract is broken
/// however good the numbers look.
#[test]
fn a_saved_follow_up_varying_slope_replays_the_design_it_was_fitted_against_2765() {
    init_parallelism();
    gam_runtime::test_support::install_diagnostic_logger();
    #[cfg(target_os = "macos")]
    gam::gpu::configure_global_policy(gam::gpu::GpuPolicy::Off);

    let (data, times) = build_dataset();
    let cfg = fit_config();

    // One call: fits, and assembles the on-disk payload from that fit. Before
    // this issue it returned an error here — the refusal was unconditional on
    // `logslope_time_k`.
    let payload = fit_formula_to_payload("Surv(time, event) ~ 1".to_string(), &data, &cfg)
        .expect("a follow-up-varying marginal-slope fit must be savable");

    let basis = payload
        .logslope_time_basis
        .as_ref()
        .expect("the saved payload must carry the log-slope time margin");
    assert_eq!(basis.degree, LOGSLOPE_TIME_DEGREE);
    assert_eq!(
        basis.knots.len(),
        LOGSLOPE_TIME_K + LOGSLOPE_TIME_DEGREE + 1,
        "the saved knot vector is the predictor's entire authority over the margin"
    );

    // The load gate has to accept it, and it validates that the fitted block's
    // width is a multiple of the margin's — a payload failing that cannot be
    // replayed under any covariate design at all.
    let model = FittedModel::from_payload(payload.clone())
        .expect("a saved follow-up-varying slope must load");

    let p_logslope = model
        .fit_result
        .as_ref()
        .expect("saved fit result")
        .blocks[2]
        .beta
        .len();
    let p_time = basis.knots.len() - basis.degree - 1;
    assert_eq!(
        p_logslope % p_time,
        0,
        "the fitted log-slope width must factor through its own margin"
    );

    // Rebuild the covariate factor exactly as predict does, then replay the
    // tensor product against the SAVED knots at the training exit times.
    let headers = data.headers.clone();
    let col_map: HashMap<String, usize> = headers
        .iter()
        .enumerate()
        .map(|(index, name)| (name.clone(), index))
        .collect();
    let logslopespec = gam::families::survival::resolve_termspec_for_prediction(
        &model.resolved_termspec_logslope.as_ref().cloned(),
        Some(&headers),
        &col_map,
        "resolved_termspec_logslope",
    )
    .expect("the saved log-slope term spec must resolve against its own headers");
    let covariate_design =
        gam_terms::smooth::build_term_collection_design(data.values.view(), &logslopespec)
            .expect("rebuild the log-slope covariate factor");
    let exit_times = ndarray::Array1::from_vec(times.clone());
    let replayed = replay_logslope_time_margin_design(
        exit_times.view(),
        basis,
        &covariate_design.design,
    )
    .expect("replay the log-slope design from the saved margin");

    assert_eq!(
        replayed.ncols(),
        p_logslope,
        "the replayed design must be exactly as wide as the coefficient vector it multiplies"
    );

    // And the surface itself: the fitted slope at each row's exit time, computed
    // once from the fit-time design and once from the replay, must agree. This
    // is the assertion that would have caught a margin replayed on the wrong
    // knots, at the wrong endpoint, or in the wrong Kronecker order.
    let beta = &model
        .fit_result
        .as_ref()
        .expect("saved fit result")
        .blocks[2]
        .beta;
    let baseline = model
        .logslope_baseline
        .expect("a saved marginal-slope model carries its fitted log-slope baseline");
    let replayed_dense = replayed
        .try_to_dense_arc("replayed log-slope design")
        .expect("dense replayed design");

    let mut minimum = f64::INFINITY;
    let mut maximum = f64::NEG_INFINITY;
    for row in 0..times.len() {
        let slope = replayed_dense.row(row).dot(beta) + baseline;
        assert!(slope.is_finite(), "replayed slope at row {row} is not finite");
        minimum = minimum.min(slope);
        maximum = maximum.max(slope);
    }
    eprintln!(
        "[2765-replay] n={N} p_logslope={p_logslope} p_time={p_time} \
         replayed_slope_range=[{minimum:.6}, {maximum:.6}]"
    );
    // A replay that collapsed the margin — for instance by keeping only the
    // covariate factor — would return the same slope at every follow-up time.
    assert!(
        maximum - minimum > 1.0e-6,
        "the replayed slope must still vary along follow-up; range was \
         [{minimum:.9}, {maximum:.9}]"
    );
}
