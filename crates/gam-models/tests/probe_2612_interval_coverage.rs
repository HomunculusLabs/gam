//! #2612 PROBE (prints only, never asserts a bound): what the multinomial
//! posterior-mean probability INTERVAL is made of, replication by replication.
//!
//! `tests/sbc_multinomial_prediction_interval_coverage.rs` reports one verdict
//! per nominal level over 80 replications and nothing else, so an
//! anti-conservative verdict there cannot say WHICH of the three ingredients is
//! wrong:
//!
//!   * the CENTRE  `E[p_c]`  — a Tierney-Kadane ratio of normalising constants,
//!   * the SPREAD  `sd(p_c)` — the same ratio machinery one order up, taken as
//!     the difference `E[p_c²] − E[p_c]²` of two SEPARATELY normalised
//!     estimates, and
//!   * the SHAPE   — a symmetric Wald band `mean ± z·sd` on a quantity that
//!     lives on `[0, 1]`.
//!
//! This probe prints all three per replication, plus the standardised residual
//! `(p_true − mean)/sd` whose empirical distribution is the discriminator: a
//! spread that is too small shows up as an inflated residual SD with a centred
//! mean, a biased centre shows up as a shifted residual mean, and a
//! shape/skewness defect shows up as calibrated moments with asymmetric tail
//! counts.

use csv::StringRecord;
use gam_data::encode_recordswith_inferred_schema;
use gam_models::fit_orchestration::FitConfig;
use gam_models::multinomial::{
    MultinomialFitRequest, fit_penalized_multinomial_formula,
    predict_multinomial_formula_with_intervals, predict_multinomial_formula_with_se,
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

#[test]
fn zz_probe_2612_multinomial_interval_ingredients() {
    let x = training_grid(N_TRAIN);
    let interior_lo = N_TRAIN / 10;
    let interior_hi = N_TRAIN - N_TRAIN / 10;
    let span = interior_hi - interior_lo;

    let mut rng = CalibrationRng::new(SEED);
    let mut hits = [0usize; NOMINAL_LEVELS.len()];
    let mut residuals: Vec<f64> = Vec::new();

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

        let j = interior_lo + (rng.uniform_open01() * span as f64) as usize % span;
        let x_star = x[j];
        let p_true = sigmoid(truth.eta(x_star));

        let new_headers = vec!["x".to_string()];
        let new_rows = vec![StringRecord::from(vec![x_star.to_string()])];
        let newdata = encode_recordswith_inferred_schema(new_headers, new_rows)
            .expect("encode multinomial newdata");

        let (mean, se) =
            predict_multinomial_formula_with_se(&model, &newdata).expect("mean and se");
        let m = mean[[0, hi_col]];
        let s = se[[0, hi_col]];
        residuals.push((p_true - m) / s.max(f64::MIN_POSITIVE));

        for (level_idx, &level) in NOMINAL_LEVELS.iter().enumerate() {
            let intervals = predict_multinomial_formula_with_intervals(&model, &newdata, level)
                .expect("intervals");
            let lower = intervals.mean_lower[[0, hi_col]];
            let upper = intervals.mean_upper[[0, hi_col]];
            if lower <= p_true && p_true <= upper {
                hits[level_idx] += 1;
            }
        }

        eprintln!(
            "#2612-INT rep={replication:3} x*={x_star:.4} p_true={p_true:.6} \
             mean={m:.6} sd={s:.6} z=({:+.3}) edf_terms={}",
            (p_true - m) / s.max(f64::MIN_POSITIVE),
            model.lambdas.len(),
        );
    }

    let n = residuals.len() as f64;
    let mean_residual = residuals.iter().sum::<f64>() / n;
    let sd_residual = (residuals
        .iter()
        .map(|r| (r - mean_residual).powi(2))
        .sum::<f64>()
        / (n - 1.0))
        .sqrt();
    let rms_residual = (residuals.iter().map(|r| r * r).sum::<f64>() / n).sqrt();
    let below = residuals.iter().filter(|r| **r < -1.0).count();
    let above = residuals.iter().filter(|r| **r > 1.0).count();
    eprintln!(
        "#2612-INT SUMMARY reps={} residual mean={mean_residual:+.4} sd={sd_residual:.4} \
         rms={rms_residual:.4} tails(<-1)={below} (>+1)={above}",
        residuals.len()
    );
    for (level_idx, &level) in NOMINAL_LEVELS.iter().enumerate() {
        let verdict = audit_coverage(hits[level_idx], N_REPLICATIONS, level);
        eprintln!(
            "#2612-INT COVERAGE level={level} empirical={:.4} hits={}/{} \
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
