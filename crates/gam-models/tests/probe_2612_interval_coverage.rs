//! #2612 PROBE (prints only, never asserts a bound): what the multinomial
//! posterior-mean probability INTERVAL is made of, replication by replication.
//!
//! `tests/sbc_multinomial_prediction_interval_coverage.rs` reports one verdict
//! per nominal level and nothing else, so an anti-conservative verdict there
//! cannot say WHICH of the three ingredients is wrong:
//!
//!   * the CENTRE  `E[p_c]`  — a Tierney-Kadane ratio of normalising constants,
//!   * the SPREAD  `sd(p_c)` — conditional on `λ̂` being exact, or with the
//!     first-order ρ-uncertainty correction `gᵀCg` added (gam#2612), and
//!   * the SHAPE   — a symmetric Wald band `mean ± z·sd` clamped into `[0, 1]`,
//!     or the monotone `expit(logit(m) ± z·sd/(m(1−m)))` transform.
//!
//! Every arm is run on the SAME fits, so the differences are the ingredients
//! and not the fixture. The standardised residual `(p_true − mean)/sd` is the
//! discriminator: a spread that is too small shows up as an inflated residual
//! SD with a centred mean, a biased centre shows up as a shifted residual mean,
//! and a shape defect shows up as calibrated moments with asymmetric tails.

use csv::StringRecord;
use gam_data::encode_recordswith_inferred_schema;
use gam_models::fit_orchestration::FitConfig;
use gam_models::multinomial::{
    InferenceCovarianceMode, MultinomialFitRequest, fit_penalized_multinomial_formula,
    predict_multinomial_formula_with_intervals_in_mode,
    predict_multinomial_formula_with_se_in_mode,
};
use gam_test_support::calibration::{CalibrationRng, audit_coverage};

const N_TRAIN: usize = 240;
const N_REPLICATIONS: usize = 40;
const NOMINAL_LEVELS: [f64; 3] = [0.80, 0.90, 0.95];
const SEED: u64 = 0x1891_A17_1_C0DE;

const CLASS_LO: &str = "lo";
const CLASS_HI: &str = "hi";

fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

struct SmoothEta {
    center: f64,
    amplitude: f64,
    frequency: f64,
    phase: f64,
}

impl SmoothEta {
    fn draw(rng: &mut CalibrationRng) -> Self {
        Self {
            center: -0.3 + 0.6 * rng.uniform_open01(),
            amplitude: 0.5 + 0.5 * rng.uniform_open01(),
            frequency: 0.7 + 0.7 * rng.uniform_open01(),
            phase: rng.uniform_open01(),
        }
    }

    fn eta(&self, x: f64) -> f64 {
        self.center
            + self.amplitude * (std::f64::consts::TAU * (self.frequency * x + self.phase)).sin()
    }
}

fn training_grid(n: usize) -> Vec<f64> {
    (0..n).map(|i| i as f64 / (n - 1) as f64).collect()
}

/// One arm's running tallies: hits per nominal level plus the standardised
/// residuals its own spread produces.
#[derive(Default)]
struct Arm {
    hits: [usize; NOMINAL_LEVELS.len()],
    residuals: Vec<f64>,
    clipped_endpoints: usize,
}

impl Arm {
    fn report(&self, label: &str) {
        let n = self.residuals.len() as f64;
        let mean = self.residuals.iter().sum::<f64>() / n;
        let sd = (self
            .residuals
            .iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f64>()
            / (n - 1.0))
            .sqrt();
        let rms = (self.residuals.iter().map(|r| r * r).sum::<f64>() / n).sqrt();
        let below = self.residuals.iter().filter(|r| **r < -1.0).count();
        let above = self.residuals.iter().filter(|r| **r > 1.0).count();
        eprintln!(
            "#2612-INT [{label}] residual mean={mean:+.4} sd={sd:.4} rms={rms:.4} \
             tails(<-1)={below} (>+1)={above} clipped_endpoints={}",
            self.clipped_endpoints,
        );
        for (index, &level) in NOMINAL_LEVELS.iter().enumerate() {
            let verdict = audit_coverage(self.hits[index], N_REPLICATIONS, level);
            eprintln!(
                "#2612-INT [{label}] level={level} empirical={:.4} hits={}/{} \
                 wilson=[{:.4},{:.4}] class={:?}",
                verdict.empirical,
                verdict.hits,
                verdict.replications,
                verdict.ci_lo,
                verdict.ci_hi,
                verdict.class,
            );
        }
    }
}

#[test]
fn zz_probe_2612_multinomial_interval_ingredients() {
    let x = training_grid(N_TRAIN);
    let interior_lo = N_TRAIN / 10;
    let interior_hi = N_TRAIN - N_TRAIN / 10;
    let span = interior_hi - interior_lo;

    let mut rng = CalibrationRng::new(SEED);
    let mut conditional = Arm::default();
    let mut corrected = Arm::default();
    let mut corrections_retained = 0usize;

    for replication in 0..N_REPLICATIONS {
        let truth = SmoothEta::draw(&mut rng);

        let mut rows: Vec<StringRecord> = Vec::with_capacity(N_TRAIN);
        for &xi in &x {
            let p_hi = sigmoid(truth.eta(xi));
            let label = if rng.uniform_open01() < p_hi {
                CLASS_HI
            } else {
                CLASS_LO
            };
            rows.push(StringRecord::from(vec![xi.to_string(), label.to_string()]));
        }
        let headers = vec!["x".to_string(), "y".to_string()];
        let data =
            encode_recordswith_inferred_schema(headers, rows).expect("encode multinomial dataset");

        let config = FitConfig::default();
        let model = fit_penalized_multinomial_formula(&MultinomialFitRequest {
            init_lambda: 1.0,
            max_iter: 60,
            tol: 1e-8,
            ..MultinomialFitRequest::new(&data, "y ~ s(x, bs='tp', k=8)", &config)
        })
        .unwrap_or_else(|e| panic!("multinomial smooth fit failed: {e:?}"));
        let hi_col = model
            .class_levels
            .iter()
            .position(|c| c == CLASS_HI)
            .expect("hi class present among fitted class levels");
        let has_correction = model.smoothing_correction_flat.is_some();
        if has_correction {
            corrections_retained += 1;
        }

        let j = interior_lo + (rng.uniform_open01() * span as f64) as usize % span;
        let x_star = x[j];
        let p_true = sigmoid(truth.eta(x_star));

        let new_headers = vec!["x".to_string()];
        let new_rows = vec![StringRecord::from(vec![x_star.to_string()])];
        let newdata = encode_recordswith_inferred_schema(new_headers, new_rows)
            .expect("encode multinomial newdata");

        let mut arm_sd = [0.0_f64; 2];
        let mut arm_mean = 0.0_f64;
        for (index, mode) in [
            InferenceCovarianceMode::Conditional,
            InferenceCovarianceMode::SmoothingCorrected,
        ]
        .into_iter()
        .enumerate()
        {
            if mode == InferenceCovarianceMode::SmoothingCorrected && !has_correction {
                continue;
            }
            let arm = if index == 0 {
                &mut conditional
            } else {
                &mut corrected
            };
            let (mean, se) = predict_multinomial_formula_with_se_in_mode(&model, &newdata, mode)
                .expect("mean and se");
            let m = mean[[0, hi_col]];
            let s = se[[0, hi_col]];
            arm_sd[index] = s;
            arm_mean = m;
            arm.residuals.push((p_true - m) / s.max(f64::MIN_POSITIVE));
            for (level_index, &level) in NOMINAL_LEVELS.iter().enumerate() {
                let intervals = predict_multinomial_formula_with_intervals_in_mode(
                    &model, &newdata, level, mode,
                )
                .expect("intervals");
                let lower = intervals.mean_lower[[0, hi_col]];
                let upper = intervals.mean_upper[[0, hi_col]];
                if lower <= 0.0 || upper >= 1.0 {
                    arm.clipped_endpoints += 1;
                }
                if lower <= p_true && p_true <= upper {
                    arm.hits[level_index] += 1;
                }
            }
        }

        eprintln!(
            "#2612-INT rep={replication:3} x*={x_star:.4} p_true={p_true:.6} mean={arm_mean:.6} \
             sd_cond={:.6} sd_corr={:.6} inflation={:.4} correction={has_correction}",
            arm_sd[0],
            arm_sd[1],
            if arm_sd[0] > 0.0 {
                arm_sd[1] / arm_sd[0]
            } else {
                f64::NAN
            },
        );
    }

    eprintln!(
        "#2612-INT SUMMARY reps={N_REPLICATIONS} fits_retaining_a_correction={corrections_retained}"
    );
    conditional.report("conditional");
    if !corrected.residuals.is_empty() {
        corrected.report("smoothing-corrected");
    }
}
