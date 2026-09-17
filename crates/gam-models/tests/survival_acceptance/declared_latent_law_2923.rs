//! gam#2923 acceptance: the survival marginal-slope index is anchored on a
//! DECLARED latent law by the defining equation
//!
//! ```text
//!     Σ_k w_k Φ(−(α(t) + b·u_k)) = Φ(−q(t)),
//! ```
//!
//! not by the closed-form Gaussian lowering `α = q·√(1 + b²)`.
//!
//! Two claims, each against the closed-form fit of the same data:
//!
//! 1. **Gaussian is a special case.** On a Gauss–Hermite law the anchored fit
//!    reproduces the closed-form fit — coefficients, log-likelihood and the
//!    fitted survival index — to quadrature tolerance.
//! 2. **On a skewed law the closed form is miscalibrated and the anchored fit
//!    is not.** The data are simulated from the family's own model on a
//!    two-component law, so `S(t | z) = Φ(−(α(q(t), b) + b z))` and
//!    `Φ(−q(t))` IS the marginal survival. The closed form lowers the same
//!    identity with `c = √(1 + b²)`, which is the wrong `α` on this law; its
//!    fit lands on a biased slope, and the survival it predicts for a subject
//!    in context `(t, z)` — `E[p̂ | a]` for the context the score is observed
//!    in — is off from the truth by a measurable amount. The anchored fit
//!    recovers the slope and is calibrated.

use csv::StringRecord;
use gam_data::encode_recordswith_inferred_schema;
use gam_models::fit_orchestration::{DeclaredLatentLaw, FitConfig, FitResult, fit_from_formula};
use gam_models::inference::model::FittedModel;
use gam_models::inference::model_payload_builders::fit_formula_to_payload;
use gam_models::survival::{
    SurvivalPredictEstimand, SurvivalPredictRequest, SurvivalPredictionCovarianceMode,
    predict_survival,
};
use ndarray::Array1;
use std::collections::HashMap;

use gam_linalg::utils::splitmix64;

const N: usize = 2_400;
/// Planted slope of the latent score on the probit survival index. Large
/// enough that the closed form's error on a skewed law — which grows with the
/// slope — stands well clear of a 2 400-row fit's own estimation noise.
const SLOPE: f64 = 1.6;
/// Marginal probit index at `t = 1`.
const LOCATION_LEVEL: f64 = -1.15;
/// Marginal probit index drift per unit `log t`; positive so `q` increases.
const LOCATION_TREND: f64 = 0.95;

fn next_unit(state: &mut u64) -> f64 {
    (splitmix64(state) >> 11) as f64 / (1u64 << 53) as f64
}

fn normal_cdf(x: f64) -> f64 {
    gam_math::probability::normal_cdf(x)
}

/// Standard-normal quantile by bisection on `Φ`. Deliberately not imported from
/// the crate under test.
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

/// A finite law: ascending nodes and weights summing to one.
#[derive(Clone)]
struct Law {
    nodes: Vec<f64>,
    weights: Vec<f64>,
}

impl Law {
    /// Probabilists' Gauss–Hermite, built here by bisection on the Hermite
    /// recurrence so the fixture does not share code with the crate.
    fn gauss_hermite(m: usize) -> Self {
        let hermite = |x: f64| -> (f64, f64) {
            let mut prev = 1.0;
            let mut current = x;
            for n in 1..m {
                let next = x * current - n as f64 * prev;
                prev = current;
                current = next;
            }
            (current, prev)
        };
        // Roots by sign changes on a fine grid, polished by bisection. The
        // probabilists' polynomial's largest root is below `2√m`.
        let mut roots = Vec::with_capacity(m);
        let lo = -(2.0 * (m as f64).sqrt() + 2.0);
        let steps = 200_000;
        let width = -2.0 * lo / steps as f64;
        let mut previous = hermite(lo).0;
        for step in 1..=steps {
            let x = lo + width * step as f64;
            let value = hermite(x).0;
            if previous.signum() != value.signum() {
                let (mut a, mut b) = (x - width, x);
                for _ in 0..200 {
                    let mid = 0.5 * (a + b);
                    if hermite(a).0.signum() == hermite(mid).0.signum() {
                        a = mid;
                    } else {
                        b = mid;
                    }
                }
                roots.push(0.5 * (a + b));
            }
            previous = value;
        }
        assert_eq!(roots.len(), m, "found every Hermite root");
        // Probabilists' weights ∝ 1 / (m² He_{m−1}(x)²).
        let mut weights: Vec<f64> = roots
            .iter()
            .map(|&x| {
                let (_, prev) = hermite(x);
                1.0 / ((m * m) as f64 * prev * prev)
            })
            .collect();
        let total: f64 = weights.iter().sum();
        for w in weights.iter_mut() {
            *w /= total;
        }
        Self {
            nodes: roots,
            weights,
        }
    }

    /// A deliberately skewed two-component law on 41 nodes (skewness ≈ 1.5),
    /// standardised to zero mean and unit variance so that a fit that ASSUMED
    /// normality could not be rescued by its own normalisation. On this law
    /// at the planted slope the closed form's marginal index is off by
    /// `α(q, b)/√(1 + b²) − q ≈ 0.3` (rmse over the index range).
    fn skewed() -> Self {
        let raw_nodes: Vec<f64> = (0..41).map(|k| -2.5 + 0.15 * k as f64).collect();
        let raw: Vec<f64> = raw_nodes
            .iter()
            .map(|&u| {
                (-0.5 * ((u + 0.9) / 0.5).powi(2)).exp()
                    + 0.20 * (-0.5 * ((u - 2.8) / 0.5).powi(2)).exp()
            })
            .collect();
        let total: f64 = raw.iter().sum();
        let weights: Vec<f64> = raw.into_iter().map(|w| w / total).collect();
        let mean: f64 = raw_nodes.iter().zip(&weights).map(|(u, w)| u * w).sum();
        let var: f64 = raw_nodes
            .iter()
            .zip(&weights)
            .map(|(u, w)| (u - mean).powi(2) * w)
            .sum();
        let nodes = raw_nodes
            .iter()
            .map(|u| (u - mean) / var.sqrt())
            .collect();
        Self { nodes, weights }
    }

    fn marginal_survival(&self, alpha: f64, slope: f64) -> f64 {
        self.nodes
            .iter()
            .zip(&self.weights)
            .map(|(&u, &w)| w * normal_cdf(-(alpha + slope * u)))
            .sum()
    }

    /// The anchor `α(q, b)` on this law, by bisection: the left side is
    /// strictly decreasing in `α`.
    fn anchor(&self, q: f64, slope: f64) -> f64 {
        let target = normal_cdf(-q);
        let (mut low, mut high) = (-40.0_f64, 40.0_f64);
        for _ in 0..200 {
            let mid = 0.5 * (low + high);
            if self.marginal_survival(mid, slope) > target {
                low = mid;
            } else {
                high = mid;
            }
        }
        0.5 * (low + high)
    }

    /// Draw a node with its weight as probability.
    fn draw(&self, u: f64) -> f64 {
        let mut cumulative = 0.0;
        for (&node, &weight) in self.nodes.iter().zip(&self.weights) {
            cumulative += weight;
            if u < cumulative {
                return node;
            }
        }
        *self.nodes.last().expect("non-empty law")
    }

    fn declared(&self) -> DeclaredLatentLaw {
        DeclaredLatentLaw {
            nodes: self.nodes.clone(),
            weights: self.weights.clone(),
        }
    }
}

fn planted_index(time: f64) -> f64 {
    LOCATION_LEVEL + LOCATION_TREND * time.ln()
}

/// Simulate from the anchored model on `law`: `S(t | z) = Φ(−(α(q(t), b) + b·z))`,
/// so that `Φ(−q(t))` is exactly the marginal survival.
///
/// With `standardize` the drawn scores are rescaled to zero mean and unit
/// (weighted, population) variance. The closed form lowers the identity with
/// the SAMPLE covariance `Σ̂`, so a Gaussian law can only be its special case
/// when `Σ̂ = 1` exactly; a declared skewed law is left as drawn, because the
/// law then describes the sample as it is.
fn build_dataset(
    law: &Law,
    seed: u64,
    standardize: bool,
) -> (gam_data::EncodedDataset, Vec<f64>, Vec<f64>) {
    let headers = ["time", "event", "z"]
        .iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();
    let mut state = seed;
    let mut draws: Vec<f64> = (0..N).map(|_| law.draw(next_unit(&mut state))).collect();
    if standardize {
        let mean = draws.iter().sum::<f64>() / N as f64;
        let variance = draws.iter().map(|z| (z - mean).powi(2)).sum::<f64>() / N as f64;
        let sd = variance.sqrt();
        for z in draws.iter_mut() {
            *z = (*z - mean) / sd;
        }
    }
    let mut rows: Vec<StringRecord> = Vec::with_capacity(N);
    let mut times = Vec::with_capacity(N);
    let mut scores = Vec::with_capacity(N);
    for z in draws {
        let u = next_unit(&mut state).clamp(1e-6, 1.0 - 1e-6);
        let censor = 0.35 + 5.0 * next_unit(&mut state);
        // Invert `Φ(−(α(q(T), b) + b z)) = u` for `log T` by bisection; the
        // index is increasing in `t`.
        let target = -normal_quantile(u) - SLOPE * z;
        let (mut low, mut high) = (-6.0_f64, 6.0_f64);
        for _ in 0..200 {
            let mid = 0.5 * (low + high);
            if law.anchor(planted_index(mid.exp()), SLOPE) < target {
                low = mid;
            } else {
                high = mid;
            }
        }
        let event_time = (0.5 * (low + high)).exp();
        let (time, event) = if event_time <= censor {
            (event_time, 1u8)
        } else {
            (censor, 0u8)
        };
        let time = time.clamp(1e-3, 1e3);
        times.push(time);
        scores.push(z);
        rows.push(StringRecord::from(vec![
            time.to_string(),
            event.to_string(),
            z.to_string(),
        ]));
    }
    let data = encode_recordswith_inferred_schema(headers, rows)
        .expect("encode the #2923 declared-law fixture");
    (data, times, scores)
}

fn base_config() -> FitConfig {
    FitConfig {
        survival_likelihood: Some("marginal-slope".to_string()),
        z_column: Some("z".to_string()),
        slope_formula: Some("1".to_string()),
        // The planted `q(t) = a₀ + a₁·log t` lies in the time spline's own
        // linear null space, so the default linear baseline chart represents
        // the data-generating location curve exactly and the outer search has
        // only the two smoothing parameters to move.
        time_num_internal_knots: 3,
        ..FitConfig::default()
    }
}

struct Fitted {
    coefficients: Vec<f64>,
    log_likelihood: f64,
    /// The fitted survival index `q̂(t_i)` at every training row's exit time.
    exit_index: Vec<f64>,
    slope: f64,
    latent_measure_is_empirical: bool,
}

fn fit(data: &gam_data::EncodedDataset, config: &FitConfig) -> Fitted {
    let result = fit_from_formula("Surv(time, event) ~ 1", data, config)
        .expect("survival marginal-slope fit");
    let FitResult::SurvivalMarginalSlope(fit) = result else {
        panic!("expected a SurvivalMarginalSlope fit result");
    };
    let exit_index: Vec<f64> = fit.fitted_exit_index.to_vec();
    // An intercept-only slope surface: every row's slope is the same number.
    let slope_design = fit.slope_design.design.to_dense();
    let slope = slope_design.row(0).dot(&fit.fit.blocks[2].beta) + fit.baseline_slope;
    Fitted {
        coefficients: fit.fit.beta.to_vec(),
        log_likelihood: fit.fit.log_likelihood,
        exit_index,
        slope,
        latent_measure_is_empirical: matches!(
            fit.latent_measure,
            gam_models::bms::LatentMeasureKind::GlobalEmpirical { .. }
        ),
    }
}

fn rmse(left: &[f64], right: &[f64]) -> f64 {
    (left
        .iter()
        .zip(right)
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        / left.len() as f64)
        .sqrt()
}

#[test]
fn anchored_fit_on_a_gaussian_law_reproduces_the_closed_form_2923() {
    super::initialize_cpu_fitting();
    gam_runtime::test_support::install_diagnostic_logger();
    #[cfg(target_os = "macos")]
    gam_gpu::configure_global_policy(gam_gpu::GpuPolicy::Off);

    let law = Law::gauss_hermite(65);
    let (data, _, _) = build_dataset(&law, 0x2923_0000_0001, true);

    let closed_form = fit(
        &data,
        &FitConfig {
            latent_measure: Some("standard-normal".to_string()),
            ..base_config()
        },
    );
    let anchored = fit(
        &data,
        &FitConfig {
            declared_latent_law: Some(law.declared()),
            ..base_config()
        },
    );
    assert!(!closed_form.latent_measure_is_empirical);
    assert!(anchored.latent_measure_is_empirical, "the declared law must be the fit's measure");

    let coefficient_gap = closed_form
        .coefficients
        .iter()
        .zip(&anchored.coefficients)
        .map(|(a, b)| (a - b).abs() / (1.0 + a.abs()))
        .fold(0.0_f64, f64::max);
    let index_gap = rmse(&closed_form.exit_index, &anchored.exit_index);
    let ll_gap = (closed_form.log_likelihood - anchored.log_likelihood).abs();
    eprintln!(
        "[2923 gaussian] n={N} slope closed-form={:.6} anchored={:.6} | max rel coefficient gap={coefficient_gap:.3e} | index rmse={index_gap:.3e} | |Δ log-lik|={ll_gap:.3e}",
        closed_form.slope, anchored.slope,
    );
    // The row program agrees to quadrature tolerance (`anchored_frame_on_a_
    // gaussian_law_is_the_gaussian_frame`: 1e-7 relative on value, gradient
    // and Hessian). At the fit level the two searches stop inside the outer
    // optimizer's own tolerance band on criteria that differ by that much, so
    // what is pinned here is agreement at the level that band permits.
    assert!(
        coefficient_gap < 2e-3,
        "on a Gaussian law the anchored coefficients must be the closed form's; max relative gap {coefficient_gap:.3e}"
    );
    assert!(
        index_gap < 2e-3,
        "on a Gaussian law the anchored survival index must be the closed form's; rmse {index_gap:.3e}"
    );
    assert!(
        ll_gap < 1e-2,
        "on a Gaussian law the anchored log-likelihood must be the closed form's; gap {ll_gap:.3e}"
    );
    assert!(
        (closed_form.slope - anchored.slope).abs() < 1e-3,
        "on a Gaussian law the anchored slope must be the closed form's; {} vs {}",
        anchored.slope,
        closed_form.slope
    );
}

#[test]
fn closed_form_is_miscalibrated_on_a_skewed_law_and_the_anchored_fit_is_not_2923() {
    super::initialize_cpu_fitting();
    gam_runtime::test_support::install_diagnostic_logger();
    #[cfg(target_os = "macos")]
    gam_gpu::configure_global_policy(gam_gpu::GpuPolicy::Off);

    let law = Law::skewed();
    let (data, times, scores) = build_dataset(&law, 0x2923_0000_0002, false);
    let truth: Vec<f64> = times.iter().map(|t| planted_index(*t)).collect();
    // The true conditional survival of every subject at its own exit time.
    let true_survival: Vec<f64> = truth
        .iter()
        .zip(&scores)
        .map(|(&q, &z)| normal_cdf(-(law.anchor(q, SLOPE) + SLOPE * z)))
        .collect();

    let closed_form = fit(
        &data,
        &FitConfig {
            latent_measure: Some("standard-normal".to_string()),
            ..base_config()
        },
    );
    let anchored = fit(
        &data,
        &FitConfig {
            declared_latent_law: Some(law.declared()),
            ..base_config()
        },
    );

    // `E[p̂ | a]` in the context `(t, z)` the score is observed in: each fit's
    // predicted survival for a subject at its own exit time, against the true
    // conditional survival. The closed form predicts with its own lowering,
    // the anchored fit with the anchor on the declared law.
    let conditional_error = |fitted: &Fitted, anchored: bool| -> f64 {
        fitted
            .exit_index
            .iter()
            .zip(&scores)
            .zip(&true_survival)
            .map(|((&q_hat, &z), &truth)| {
                let location = if anchored {
                    law.anchor(q_hat, fitted.slope)
                } else {
                    q_hat * (1.0 + fitted.slope * fitted.slope).sqrt()
                };
                (normal_cdf(-(location + fitted.slope * z)) - truth).abs()
            })
            .sum::<f64>()
            / scores.len() as f64
    };
    // And the MARGINAL index itself: `Φ(−q̂(t))` against `Φ(−q(t))`.
    let marginal_error = |fitted: &Fitted| -> f64 {
        fitted
            .exit_index
            .iter()
            .zip(&truth)
            .map(|(&q_hat, &q)| (normal_cdf(-q_hat) - normal_cdf(-q)).abs())
            .sum::<f64>()
            / truth.len() as f64
    };
    let closed_form_conditional = conditional_error(&closed_form, false);
    let anchored_conditional = conditional_error(&anchored, true);
    let closed_form_marginal = marginal_error(&closed_form);
    let anchored_marginal = marginal_error(&anchored);
    let closed_form_index_rmse = rmse(&closed_form.exit_index, &truth);
    let anchored_index_rmse = rmse(&anchored.exit_index, &truth);
    eprintln!(
        "[2923 skewed] n={N} planted b={SLOPE} | slope closed-form={:.4} anchored={:.4} | \
         mean |Ŝ(t,z) − S(t,z)|: closed-form={closed_form_conditional:.4} anchored={anchored_conditional:.4} | \
         mean |Φ(−q̂)−Φ(−q)|: closed-form={closed_form_marginal:.4} anchored={anchored_marginal:.4} | \
         index rmse vs truth: closed-form={closed_form_index_rmse:.4} anchored={anchored_index_rmse:.4} | \
         log-lik closed-form={:.3} anchored={:.3}",
        closed_form.slope, anchored.slope, closed_form.log_likelihood, anchored.log_likelihood,
    );
    assert!(
        anchored_conditional < 0.02,
        "the anchored fit must be calibrated in context; mean |Ŝ − S| = {anchored_conditional:.4}"
    );
    assert!(
        closed_form_conditional > 2.0 * anchored_conditional && closed_form_conditional > 0.03,
        "the closed form must be measurably miscalibrated on a skewed law; \
         closed-form {closed_form_conditional:.4} vs anchored {anchored_conditional:.4}"
    );
    assert!(
        (anchored.slope - SLOPE).abs() < 0.15,
        "the anchored slope must recover the planted one; got {}",
        anchored.slope
    );
    assert!(
        (closed_form.slope - SLOPE).abs() > 2.0 * (anchored.slope - SLOPE).abs(),
        "the closed form's slope must be the biased one; closed-form {} vs anchored {} (planted {SLOPE})",
        closed_form.slope,
        anchored.slope
    );
    assert!(
        anchored.log_likelihood > closed_form.log_likelihood + 20.0,
        "the anchored model must fit the skewed data decisively better; log-lik anchored {} vs closed-form {}",
        anchored.log_likelihood,
        closed_form.log_likelihood
    );
    assert!(
        anchored_index_rmse < closed_form_index_rmse,
        "the anchored index must be closer to the planted marginal index; \
         anchored rmse {anchored_index_rmse:.4} vs closed-form {closed_form_index_rmse:.4}"
    );
}

/// The declared law is persisted as the model's latent measure and replayed
/// at prediction by the same anchoring equation: the saved model's index at
/// every training row is `α(q̂, b̂) + b̂·z` on the declared law, not the
/// closed form `q̂·√(1 + b̂²) + b̂·z`.
#[test]
fn declared_law_is_persisted_and_replayed_at_prediction_2923() {
    super::initialize_cpu_fitting();
    gam_runtime::test_support::install_diagnostic_logger();
    #[cfg(target_os = "macos")]
    gam_gpu::configure_global_policy(gam_gpu::GpuPolicy::Off);

    let law = Law::skewed();
    let (data, _, scores) = build_dataset(&law, 0x2923_0000_0003, false);
    let config = FitConfig {
        declared_latent_law: Some(law.declared()),
        ..base_config()
    };
    let fitted = fit(&data, &config);
    let payload = fit_formula_to_payload("Surv(time, event) ~ 1".to_string(), &data, &config)
        .expect("fit to a saved payload");
    match payload.latent_measure.as_ref() {
        Some(gam_models::bms::LatentMeasureKind::GlobalEmpirical { grid }) => {
            assert_eq!(grid.nodes, law.nodes, "the persisted law must be the declared nodes");
            assert_eq!(grid.weights, law.weights, "the persisted law must be the declared weights");
        }
        other => panic!(
            // SAFETY (test): the persisted measure is the property under test.
            "the saved model must carry the declared law as its latent measure; got {other:?}"
        ),
    }

    let model = FittedModel::from_payload(payload);
    let col_map: HashMap<String, usize> = data
        .headers
        .iter()
        .enumerate()
        .map(|(index, name)| (name.clone(), index))
        .collect();
    let zeros = Array1::<f64>::zeros(data.values.nrows());
    let prediction = predict_survival(
        SurvivalPredictRequest {
            model: &model,
            data: data.values.view(),
            col_map: &col_map,
            training_headers: Some(&data.headers),
            primary_offset: &zeros,
            noise_offset: &zeros,
            time_grid: None,
            with_uncertainty: false,
            estimand: SurvivalPredictEstimand::Plugin,
        },
        SurvivalPredictionCovarianceMode::Conditional,
    )
    .expect("saved survival marginal-slope prediction at the training rows");

    let mut worst_anchored = 0.0_f64;
    let mut worst_closed_form = 0.0_f64;
    for (row, (&q_hat, &z)) in fitted.exit_index.iter().zip(&scores).enumerate() {
        let expected = law.anchor(q_hat, fitted.slope) + fitted.slope * z;
        let closed_form = q_hat * (1.0 + fitted.slope * fitted.slope).sqrt() + fitted.slope * z;
        let replayed = prediction.linear_predictor[row];
        worst_anchored = worst_anchored.max((replayed - expected).abs());
        worst_closed_form = worst_closed_form.max((replayed - closed_form).abs());
        let survival = prediction.survival[[row, 0]];
        assert!(
            (survival - normal_cdf(-replayed)).abs() < 1e-10,
            "row {row}: survival {survival} must be Φ(−η̂) with η̂ = {replayed}"
        );
    }
    eprintln!(
        "[2923 replay] n={N} max |η̂_saved − (α(q̂, b̂) + b̂ z)| = {worst_anchored:.3e} | \
         max |η̂_saved − closed form| = {worst_closed_form:.3e}"
    );
    assert!(
        worst_anchored < 1e-6,
        "the saved model must replay the anchored index; worst gap {worst_anchored:.3e}"
    );
    assert!(
        worst_closed_form > 1e-2,
        "the replayed index must not be the closed form on a skewed law; worst gap {worst_closed_form:.3e}"
    );
}
