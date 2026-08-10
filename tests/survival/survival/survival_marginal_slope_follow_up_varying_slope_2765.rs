//! gam#2765 / gam#2767 acceptance: the marginal slope can vary along the
//! follow-up axis, and the fit recovers a slope that *actually moves*.
//!
//! The fixture is simulated from the family's own model rather than from a
//! convenient approximation of it, because the point at issue is whether the
//! kernel gained the right terms:
//!
//! ```text
//!   S(t | x, z) = Φ(−η(t)),   η(t) = q(t)·c(t) + b(t)·z,
//!   c(t) = √(1 + b(t)²),      b(t) = β₀ + β₁·log t
//! ```
//!
//! with `q(t) = a₀ + a₁·log t` increasing. Event times are drawn by inverting
//! `Φ(−η(T)) = U` exactly, so the planted slope really is the conditional
//! effect of `z` at each follow-up time and not an artefact of a hazard-scale
//! shortcut.
//!
//! `β₁ < 0` is the attenuation case — a score whose effect fades with age —
//! which is the phenomenon #2767 asks about. The contract is that the fitted
//! per-row slope tracks the planted one: a model that cannot move `b` along
//! follow-up can only return a flat surface, whose correlation with a genuinely
//! varying truth is undefined, so this test cannot pass on the pre-#2765 kernel
//! by luck.

use csv::StringRecord;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};

const N: usize = 2_400;
/// Polynomial degree of the slope's follow-up margin. Bound to a constant so
/// the request and the assertion on the RESOLVED margin cannot drift: this test
/// previously asked for a quadratic margin and asserted a cubic one, which is
/// two different claims about the same object.
const LOGSLOPE_TIME_DEGREE: usize = 2;
/// Columns in that margin. `k >= degree + 1` is the admission rule; four columns
/// over a quadratic margin leaves one internal knot.
const LOGSLOPE_TIME_K: usize = 4;
/// Slope at `t = 1` (`log t = 0`).
const SLOPE_LEVEL: f64 = 0.85;
/// Slope drift per unit `log t`. Negative = the score's effect attenuates.
const SLOPE_TREND: f64 = -0.32;
/// Marginal probit index at `t = 1`.
const LOCATION_LEVEL: f64 = -1.15;
/// Marginal probit index drift per unit `log t`; positive so `q` is increasing
/// and the marginal survival curve is decreasing, as the family requires.
const LOCATION_TREND: f64 = 0.95;

use gam::utils::splitmix64;

fn next_unit(state: &mut u64) -> f64 {
    (splitmix64(state) >> 11) as f64 / (1u64 << 53) as f64
}

fn next_gauss(state: &mut u64) -> f64 {
    let u1 = next_unit(state).max(1e-12);
    let u2 = next_unit(state);
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

fn planted_slope(time: f64) -> f64 {
    SLOPE_LEVEL + SLOPE_TREND * time.ln()
}

/// `η(t)` under the planted model, for a subject with latent score `z`.
fn planted_eta(time: f64, z: f64) -> f64 {
    let slope = planted_slope(time);
    let location = LOCATION_LEVEL + LOCATION_TREND * time.ln();
    location * (1.0 + slope * slope).sqrt() + slope * z
}

/// Standard-normal quantile by bisection on `Φ`. Deliberately not imported from
/// the crate under test: the fixture's truth must not be produced by the same
/// code path the fit uses.
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

/// Abramowitz–Stegun 7.1.26; ~1.5e-7 absolute, far finer than the fixture needs.
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

/// Invert `Φ(−η(T)) = u` for `T` by bisection on `log T`. `η` is increasing in
/// `t` under the planted constants over the fixture's support, so the root is
/// unique there.
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

fn build_dataset() -> (gam::inference::data::EncodedDataset, Vec<f64>, Vec<f64>) {
    let headers = ["time", "event", "z"]
        .iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();
    let mut state: u64 = 0x2765_2767_5CA1_AB1E_u64;

    let mut raw_scores: Vec<f64> = Vec::with_capacity(N);
    let mut draws: Vec<f64> = Vec::with_capacity(N);
    let mut censor: Vec<f64> = Vec::with_capacity(N);
    for _ in 0..N {
        raw_scores.push(next_gauss(&mut state));
        draws.push(next_unit(&mut state).clamp(1e-6, 1.0 - 1e-6));
        censor.push(next_unit(&mut state));
    }
    // The latent score is standardized by construction, as the family expects.
    let mean = raw_scores.iter().sum::<f64>() / N as f64;
    let variance = raw_scores.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / N as f64;
    let sd = variance.sqrt().max(1e-12);
    let scores: Vec<f64> = raw_scores.iter().map(|v| (v - mean) / sd).collect();

    let mut rows: Vec<StringRecord> = Vec::with_capacity(N);
    let mut observed_times: Vec<f64> = Vec::with_capacity(N);
    for index in 0..N {
        let z = scores[index];
        let event_time = planted_event_time(draws[index], z);
        // Administrative censoring spread over the fixture's support, giving a
        // moderate event rate and a genuinely wide range of exit times — the
        // range is what identifies the time margin.
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
        .expect("encode the #2765 follow-up-varying slope fixture");
    (data, observed_times, scores)
}

fn pearson(left: &[f64], right: &[f64]) -> f64 {
    let n = left.len() as f64;
    let left_mean = left.iter().sum::<f64>() / n;
    let right_mean = right.iter().sum::<f64>() / n;
    let mut cov = 0.0;
    let mut left_var = 0.0;
    let mut right_var = 0.0;
    for (a, b) in left.iter().zip(right.iter()) {
        let da = a - left_mean;
        let db = b - right_mean;
        cov += da * db;
        left_var += da * da;
        right_var += db * db;
    }
    cov / (left_var.sqrt() * right_var.sqrt()).max(1e-300)
}

#[test]
fn survival_marginal_slope_recovers_a_follow_up_varying_slope_2765() {
    init_parallelism();
    gam_runtime::test_support::install_diagnostic_logger();
    #[cfg(target_os = "macos")]
    gam::gpu::configure_global_policy(gam::gpu::GpuPolicy::Off);

    let (data, times, _scores) = build_dataset();

    let cfg = FitConfig {
        survival_likelihood: Some("marginal-slope".to_string()),
        z_column: Some("z".to_string()),
        logslope_formula: Some("1".to_string()),
        // The slope's own follow-up margin — the whole point of #2765/#2767.
        // A quadratic margin with four columns spans the planted linear-in-log-t
        // slope with room to spare while staying well clear of the baseline
        // time surface, which is also a smooth function of `log t`: the two are
        // separated by the `z` interaction, not by their time shapes, so a very
        // rich margin buys nothing and costs conditioning.
        logslope_time_k: Some(LOGSLOPE_TIME_K),
        logslope_time_degree: LOGSLOPE_TIME_DEGREE,
        // Keep the baseline time surface only as flexible as the planted
        // `q(t) = a₀ + a₁·log t` needs.
        time_num_internal_knots: 3,
        // A Weibull baseline, not the linear default. Two reasons: the planted
        // `q(t) = a₀ + a₁·log t` IS a Weibull-shaped index, so the offset chart
        // is a faithful frame for it; and a linear baseline with no spatial term
        // leaves the joint outer problem with zero auxiliary and zero kappa
        // axes, which today takes a driver fast path that returns no outer
        // certificate and makes the survival marginal-slope entry refuse. That
        // refusal reproduces identically with `logslope_time_k` unset, so it is
        // a pre-existing property of that model shape and not of the follow-up
        // margin.
        baseline_target: "weibull".to_string(),
        ..FitConfig::default()
    };

    let result = fit_from_formula("Surv(time, event) ~ 1", &data, &cfg)
        .expect("survival marginal-slope fit with a follow-up-varying log-slope");
    let FitResult::SurvivalMarginalSlope(fit) = result else {
        panic!("expected a SurvivalMarginalSlope fit result");
    };

    // The resolved margin is carried on the fit result so a predictor replays it
    // against the same knots.
    let basis = fit
        .logslope_time_basis
        .as_ref()
        .expect("a fit with logslope_time_k must carry its resolved time margin");
    assert_eq!(
        basis.degree, LOGSLOPE_TIME_DEGREE,
        "the resolved margin must keep the requested degree"
    );
    assert_eq!(
        basis.knots.len(),
        LOGSLOPE_TIME_K + LOGSLOPE_TIME_DEGREE + 1,
        "a `k`-column B-spline of this degree owns `k + degree + 1` knots, and \
         those knots are the whole authority a predictor gets"
    );

    // With an intercept-only log-slope covariate formula the tensored design IS
    // the time margin at each row's exit time, so `design · β` is the fitted
    // slope deviation from the baseline at that row's follow-up time.
    let beta = &fit.fit.blocks[2].beta;
    assert_eq!(
        fit.logslope_design.design.ncols(),
        beta.len(),
        "the log-slope block's coefficients must match its tensored design width"
    );
    assert!(
        beta.len() > 1,
        "a follow-up-varying slope must own more than one coefficient; got {}",
        beta.len()
    );
    let design = fit.logslope_design.design.to_dense();
    let fitted: Vec<f64> = (0..times.len())
        .map(|row| design.row(row).dot(beta) + fit.baseline_slope)
        .collect();
    let truth: Vec<f64> = times.iter().map(|t| planted_slope(*t)).collect();

    for value in &fitted {
        assert!(value.is_finite(), "fitted slope must be finite; got {value}");
    }

    let correlation = pearson(&fitted, &truth);
    let fitted_min = fitted.iter().cloned().fold(f64::INFINITY, f64::min);
    let fitted_max = fitted.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let truth_min = truth.iter().cloned().fold(f64::INFINITY, f64::min);
    let truth_max = truth.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    eprintln!(
        "[2765] n={N} planted b(t)={SLOPE_LEVEL:+.3}{SLOPE_TREND:+.3}·log t \
         fitted_range=[{fitted_min:.4}, {fitted_max:.4}] \
         truth_range=[{truth_min:.4}, {truth_max:.4}] pearson={correlation:.4} \
         time_margin_cols={} outer_iters={}",
        beta.len(),
        fit.fit.outer_iterations,
    );

    // The fitted surface must MOVE. A slope that cannot vary along follow-up
    // returns a flat surface, whose spread is zero and whose correlation with a
    // varying truth is undefined — so this is the assertion the pre-#2765
    // kernel could not satisfy at all.
    assert!(
        fitted_max - fitted_min > 0.05,
        "the fitted slope must vary along follow-up; range was \
         [{fitted_min:.6}, {fitted_max:.6}]"
    );
    // And it must move in the planted direction, not merely move.
    assert!(
        correlation > 0.8,
        "the fitted slope surface must track the planted one along follow-up; \
         Pearson was {correlation:.4}"
    );
}
