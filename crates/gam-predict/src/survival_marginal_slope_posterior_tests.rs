#![cfg(test)]
//! Survival marginal-slope posterior-mean surface against Monte Carlo (gam#2927).
//!
//! The survival marginal-slope family is anchored the same way as the Bernoulli
//! one — `S(t | x, z) = Φ(−η(t))`, `η(t) = c(b)·q(t) + s·b·z` — and its
//! posterior-mean survival surface is produced by
//! `gam_models::survival::predict::predict_survival` under
//! `SurvivalPredictEstimand::PosteriorMean`, which re-solves the anchor at every
//! node of a `2·rank` symmetric sigma-point rule over the coefficient posterior.
//! That rule is exact for cubic functionals of `θ`, not for `Φ(−η(θ))`, so it is
//! measured here the same way the Bernoulli integrations are: against Monte
//! Carlo over the full coefficient posterior, every draw run through the
//! production plug-in prediction.
//!
//! Measured on this fixture (5 latent scores × 3 horizons, 20,000 draws): at
//! the fitted covariance the published surface is within 4.1e-4 of the Monte
//! Carlo integral where the plug-in is off by 1.4e-3; at 9× the covariance
//! (SE ×3) it is within 2.0e-3 where the plug-in is off by 1.3e-2. So the
//! anchor moves with the coefficients, and what remains is the cubature error
//! of a degree-3 rule — the same order as the linearised-anchor error the
//! Bernoulli pass used to carry, which an exact `(q(t), b(t))` bivariate
//! Gauss–Hermite integration (both affine in `θ`) would remove.

use crate::anchored_posterior_predictive_tests::{gaussian, psd_factor, uniform};
use crate::test_support::init_parallelism;
use gam_data::encode_recordswith_inferred_schema;
use gam_math::probability::normal_cdf;
use gam_models::fit_orchestration::FitConfig;
use gam_models::inference::model::FittedModel;
use gam_models::inference::model_payload_builders::fit_formula_to_payload;
use gam_models::survival::predict::{
    SurvivalPredictEstimand, SurvivalPredictRequest, SurvivalPredictionCovarianceMode,
    predict_survival,
};
use ndarray::{Array1, Array2};
use std::collections::HashMap;

/// Planted survival marginal-slope model, the #2765 acceptance fixture's: a
/// slope that attenuates along follow-up, `b(t) = 0.85 − 0.32·log t`, over the
/// Weibull-shaped marginal index `q(t) = −1.15 + 0.95·log t`.
const SURVIVAL_LOCATION_LEVEL: f64 = -1.15;
const SURVIVAL_LOCATION_TREND: f64 = 0.95;
const SURVIVAL_SLOPE_LEVEL: f64 = 0.85;
const SURVIVAL_SLOPE_TREND: f64 = -0.32;
const SURVIVAL_SLOPE_TIME_K: usize = 4;
const SURVIVAL_SLOPE_TIME_DEGREE: usize = 2;

fn survival_planted_eta(time: f64, z: f64) -> f64 {
    let slope = SURVIVAL_SLOPE_LEVEL + SURVIVAL_SLOPE_TREND * time.ln();
    let location = SURVIVAL_LOCATION_LEVEL + SURVIVAL_LOCATION_TREND * time.ln();
    location * (1.0 + slope * slope).sqrt() + slope * z
}

/// Standard-normal quantile by bisection, independent of the crate under test.
fn normal_quantile(p: f64) -> f64 {
    let (mut low, mut high) = (-12.0_f64, 12.0_f64);
    for _ in 0..200 {
        let mid = 0.5 * (low + high);
        if normal_cdf(mid) < p {
            low = mid;
        } else {
            high = mid;
        }
    }
    0.5 * (low + high)
}

/// Event times drawn by inverting `Φ(−η(T)) = u` exactly, by bisection on
/// `log T` (`η` is increasing in `t` over the fixture's support).
fn survival_planted_event_time(u: f64, z: f64) -> f64 {
    let target = -normal_quantile(u);
    let (mut low, mut high) = (-6.0_f64, 6.0_f64);
    for _ in 0..200 {
        let mid = 0.5 * (low + high);
        if survival_planted_eta(mid.exp(), z) < target {
            low = mid;
        } else {
            high = mid;
        }
    }
    (0.5 * (low + high)).exp()
}

fn simulate_survival_dataset(n: usize, seed: u64) -> gam_data::EncodedDataset {
    let mut state = seed;
    let headers = ["time", "event", "z"]
        .iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();
    let mut records = Vec::with_capacity(n);
    for _ in 0..n {
        let z = gaussian(&mut state);
        let u = uniform(&mut state).clamp(1e-6, 1.0 - 1e-6);
        let event_time = survival_planted_event_time(u, z);
        let censor_time = 0.35 + 5.0 * uniform(&mut state);
        let (time, event) = if event_time <= censor_time {
            (event_time, 1u8)
        } else {
            (censor_time, 0u8)
        };
        let time = time.clamp(1e-3, 1e3);
        records.push(csv::StringRecord::from(vec![
            format!("{time:.17e}"),
            event.to_string(),
            format!("{z:.17e}"),
        ]));
    }
    encode_recordswith_inferred_schema(headers, records).expect("encode survival dataset")
}

/// Fits the survival marginal-slope fixture with the #2765 acceptance test's
/// configuration (intercept-only surfaces, a quadratic follow-up margin on
/// the slope, Weibull baseline). A covariate-bearing variant of this fixture —
/// `Surv(time, event) ~ x` / `s(x, k=5)` with a constant slope — is refused at
/// seed validation on current `main` ("changed profile objective between value
/// screening and derivative assembly"), which is a fit-side defect outside this
/// test's scope; the contexts here therefore collapse and only the latent
/// score and the horizon vary.
fn fit_survival_marginal_slope_model(n: usize, seed: u64) -> FittedModel {
    let ds = simulate_survival_dataset(n, seed);
    let cfg = FitConfig {
        survival_likelihood: Some("marginal-slope".to_string()),
        slope_formula: Some("1".to_string()),
        z_column: Some("z".to_string()),
        frozen_score: true,
        slope_time_k: Some(SURVIVAL_SLOPE_TIME_K),
        slope_time_degree: SURVIVAL_SLOPE_TIME_DEGREE,
        baseline_target: "weibull".to_string(),
        time_num_internal_knots: 3,
        precompute_conformal: Some(false),
        ..FitConfig::default()
    };
    let payload = fit_formula_to_payload("Surv(time, event) ~ 1".to_string(), &ds, &cfg)
        .expect("survival marginal-slope fit");
    FittedModel::from_payload(payload)
}

/// Prediction rows for the survival fixture: the latent scores of the
/// Bernoulli grid (its contexts collapse, see
/// `fit_survival_marginal_slope_model`), with the exit-time column present (a
/// time grid overrides it) and no event.
fn survival_prediction_frame() -> (Array2<f64>, HashMap<String, usize>, Vec<(f64, f64)>) {
    let scores = [-2.0, -1.0, 0.0, 1.0, 2.0];
    let cells: Vec<(f64, f64)> = scores.iter().map(|&z| (0.0, z)).collect();
    let mut data = Array2::<f64>::zeros((cells.len(), 3));
    for (row, &(_, z)) in cells.iter().enumerate() {
        data[[row, 0]] = 1.0;
        data[[row, 2]] = z;
    }
    let col_map = HashMap::from([
        ("time".to_string(), 0usize),
        ("event".to_string(), 1usize),
        ("z".to_string(), 2usize),
    ]);
    (data, col_map, cells)
}

/// The survival surface `S(t)` over `times` for every row of `data`.
fn survival_surface(
    model: &FittedModel,
    data: &Array2<f64>,
    col_map: &HashMap<String, usize>,
    times: &[f64],
    estimand: SurvivalPredictEstimand,
) -> Array2<f64> {
    let zero = Array1::<f64>::zeros(data.nrows());
    predict_survival(
        SurvivalPredictRequest {
            model,
            data: data.view(),
            col_map,
            training_headers: model.training_headers.as_ref(),
            primary_offset: &zero,
            noise_offset: &zero,
            time_grid: Some(times),
            with_uncertainty: false,
            estimand,
        },
        SurvivalPredictionCovarianceMode::Conditional,
    )
    .unwrap_or_else(|e| panic!("survival {estimand:?} prediction: {e}"))
    .survival
}

/// Write a coefficient draw into a saved survival model, the way the
/// posterior quadrature does for each of its nodes.
fn assign_survival_coefficients(model: &mut FittedModel, draw: &Array1<f64>) {
    assert!(
        model.beta_baseline_timewiggle.is_none() && model.survival_beta_time.is_none(),
        "this fixture carries its coefficients in the fit result only"
    );
    let fit = model
        .fit_result
        .as_mut()
        .expect("survival model carries its fit result");
    fit.beta.assign(draw);
    let mut cursor = 0usize;
    for block in &mut fit.blocks {
        let end = cursor + block.beta.len();
        block.beta.assign(&draw.slice(ndarray::s![cursor..end]));
        cursor = end;
    }
    assert_eq!(cursor, draw.len());
}

/// `E_θ[S(t; θ)]` by Monte Carlo over the survival model's own posterior, every
/// draw run through the production plug-in prediction (anchor re-solved per
/// draw), with the standard error of each cell's estimate.
fn survival_monte_carlo_reference(
    model: &FittedModel,
    data: &Array2<f64>,
    col_map: &HashMap<String, usize>,
    times: &[f64],
    draws: usize,
    seed: u64,
) -> (Array2<f64>, Array2<f64>) {
    use rayon::iter::{IntoParallelIterator, ParallelIterator};
    let fit = model.fit_result.as_ref().expect("fit result");
    let covariance = fit
        .covariance_conditional
        .as_ref()
        .expect("survival fit carries a conditional covariance");
    let p = fit.beta.len();
    assert_eq!(covariance.dim(), (p, p));
    assert_eq!(
        model
            .saved_prediction_runtime()
            .expect("prediction runtime")
            .influence_absorber_width
            .unwrap_or(0),
        0,
        "a frozen score carries no influence absorber; every coefficient is active"
    );
    let theta_hat = fit.beta.clone();
    let factor = psd_factor(covariance);
    let n = data.nrows();
    let n_times = times.len();
    let chunks = 48usize;
    let per_chunk = draws.div_ceil(chunks);
    let (sum, sum_sq): (Array2<f64>, Array2<f64>) = (0..chunks)
        .into_par_iter()
        .map(|chunk| {
            let mut state = seed ^ (chunk as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            let mut draw_model = model.clone();
            let mut sum = Array2::<f64>::zeros((n, n_times));
            let mut sum_sq = Array2::<f64>::zeros((n, n_times));
            let mut standard = Array1::<f64>::zeros(p);
            let first = chunk * per_chunk;
            let last = ((chunk + 1) * per_chunk).min(draws);
            for _ in first..last {
                standard.mapv_inplace(|_| gaussian(&mut state));
                let theta = &theta_hat + &factor.dot(&standard);
                assign_survival_coefficients(&mut draw_model, &theta);
                let survival = survival_surface(
                    &draw_model,
                    data,
                    col_map,
                    times,
                    SurvivalPredictEstimand::Plugin,
                );
                sum += &survival;
                sum_sq += &survival.mapv(|s| s * s);
            }
            (sum, sum_sq)
        })
        .reduce(
            || (Array2::zeros((n, n_times)), Array2::zeros((n, n_times))),
            |(a, a2), (b, b2)| (a + b, a2 + b2),
        );
    let count = draws as f64;
    let mean = sum.mapv(|s| s / count);
    let standard_error = Array2::from_shape_fn((n, n_times), |(row, time)| {
        let variance = (sum_sq[[row, time]] / count - mean[[row, time]].powi(2)).max(0.0);
        (variance / count).sqrt()
    });
    (mean, standard_error)
}

/// Measures the plug-in and the published posterior-mean survival surfaces of
/// `model` against Monte Carlo; returns `(max|ΔS| plug-in, max|ΔS| posterior,
/// max MC se, worst posterior gap in MC standard errors)`.
fn measure_survival(
    label: &str,
    model: &FittedModel,
    data: &Array2<f64>,
    col_map: &HashMap<String, usize>,
    times: &[f64],
    cells: &[(f64, f64)],
    draws: usize,
    seed: u64,
) -> (f64, f64, f64, f64) {
    let plugin = survival_surface(model, data, col_map, times, SurvivalPredictEstimand::Plugin);
    let posterior = survival_surface(
        model,
        data,
        col_map,
        times,
        SurvivalPredictEstimand::PosteriorMean,
    );
    let (reference, standard_error) =
        survival_monte_carlo_reference(model, data, col_map, times, draws, seed);
    eprintln!(
        "[{label}] {:>6} {:>5} {:>5} | {:>9} {:>8} | {:>10} {:>10}",
        "x", "z", "t", "mc", "mc_se", "plug-in", "posterior"
    );
    let (mut max_plugin, mut max_posterior, mut max_se, mut worst_ratio) =
        (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
    for (row, &(x, z)) in cells.iter().enumerate() {
        for (column, &t) in times.iter().enumerate() {
            let delta_plugin = plugin[[row, column]] - reference[[row, column]];
            let delta_posterior = posterior[[row, column]] - reference[[row, column]];
            let se = standard_error[[row, column]];
            eprintln!(
                "[{label}] {x:>6.2} {z:>5.1} {t:>5.2} | {:>9.5} {:>8.1e} | {:>+10.2e} {:>+10.2e}",
                reference[[row, column]],
                se,
                delta_plugin,
                delta_posterior,
            );
            max_plugin = max_plugin.max(delta_plugin.abs());
            max_posterior = max_posterior.max(delta_posterior.abs());
            max_se = max_se.max(se);
            worst_ratio = worst_ratio.max(delta_posterior.abs() / (se + 1e-12));
        }
    }
    eprintln!(
        "[{label}] max|ΔS| vs MC ({draws} draws, max MC se {max_se:.1e}): plug-in {max_plugin:.2e}, \
         published posterior mean {max_posterior:.2e} (worst gap {worst_ratio:.1} MC se)"
    );
    (max_plugin, max_posterior, max_se, worst_ratio)
}

#[test]
fn survival_marginal_slope_posterior_mean_against_monte_carlo() {
    init_parallelism();
    let model = fit_survival_marginal_slope_model(800, 20260917);
    let (data, col_map, cells) = survival_prediction_frame();
    let times = [0.75, 1.5, 3.0];
    // Each Monte Carlo draw is a full survival prediction, so the reference
    // uses fewer draws than the Bernoulli test; the standard error is reported
    // and the comparison is stated in units of it.
    let draws = 20_000;

    let fitted = measure_survival(
        "survival/fitted-V",
        &model,
        &data,
        &col_map,
        &times,
        &cells,
        draws,
        11,
    );

    let mut wide_model = model.clone();
    {
        let fit = wide_model.fit_result.as_mut().expect("fit result");
        let wide = fit
            .covariance_conditional
            .as_ref()
            .expect("covariance")
            .mapv(|v| 9.0 * v);
        fit.covariance_conditional = Some(wide);
    }
    let wide = measure_survival(
        "survival/9×V",
        &wide_model,
        &data,
        &col_map,
        &times,
        &cells,
        draws,
        12,
    );

    // The published posterior mean must at least beat the plug-in it replaces,
    // and must stay within the cubature error measured above (with margin for
    // the Monte Carlo resolution): a rule that stopped moving the anchor, or a
    // coarser one, fails here; an exact integration only tightens it.
    for (label, (max_plugin, max_posterior, _, _), bound) in
        [("fitted-V", fitted, 1.0e-3), ("9×V", wide, 4.0e-3)]
    {
        assert!(
            max_posterior <= max_plugin + 1e-12,
            "[{label}] the posterior-mean survival surface (max|ΔS| {max_posterior:.2e}) is further \
             from Monte Carlo than the plug-in ({max_plugin:.2e})"
        );
        assert!(
            max_posterior <= bound,
            "[{label}] the posterior-mean survival surface is {max_posterior:.2e} from the Monte \
             Carlo integral; the sigma-point rule measured within {bound:.1e} here"
        );
    }
}
