//! Temporary shape probe for the gam#2768 conditional-latent fixture: which
//! marginal / log-slope formula combinations actually converge on it.
//!
//! Prints only; it asserts nothing beyond "at least one shape converged", so a
//! stall is reported rather than hidden. Deleted once the fixture's shape is
//! locked in.

use csv::StringRecord;
use gam::{FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism};
use gam::utils::splitmix64;

const N: usize = 3_000;
const M_SHIFT: f64 = 0.5;
const TRUE_SLOPE: f64 = 0.6;
const TRUE_BETA_X: f64 = 0.5;
const TRUE_INTERCEPT: f64 = -1.0;
const TRUE_LOG_TIME: f64 = 0.8;

fn next_unit(state: &mut u64) -> f64 {
    (splitmix64(state) >> 11) as f64 / (1u64 << 53) as f64
}
fn next_gauss(state: &mut u64) -> f64 {
    let u1 = next_unit(state).max(1e-12);
    let u2 = next_unit(state);
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}
fn quantile(p: f64) -> f64 {
    let p = p.clamp(1e-9, 1.0 - 1e-9);
    let (mut lo, mut hi) = (-8.0_f64, 8.0_f64);
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if gam::probability::normal_cdf(mid) < p { lo = mid } else { hi = mid }
    }
    0.5 * (lo + hi)
}

fn build() -> (gam::inference::data::EncodedDataset, Vec<f64>) {
    let headers = ["time", "event", "z", "zeta", "x"].iter().map(|s| s.to_string()).collect::<Vec<_>>();
    let mut state: u64 = 0x2768_5EED_C0FF_EE01;
    let c = (1.0 + TRUE_SLOPE * TRUE_SLOPE).sqrt();
    let rsd = (1.0 - M_SHIFT * M_SHIFT).sqrt();
    let mut rows = Vec::with_capacity(N);
    let mut xs = Vec::with_capacity(N);
    let mut events = 0usize;
    for _ in 0..N {
        let x = next_gauss(&mut state);
        let zeta = next_gauss(&mut state);
        let z = M_SHIFT * x + rsd * zeta;
        let eta = quantile(next_unit(&mut state));
        let log_t = ((eta - TRUE_SLOPE * zeta) / c - TRUE_INTERCEPT - TRUE_BETA_X * x) / TRUE_LOG_TIME;
        let t_event = log_t.exp();
        let t_cens = 1.5 + 3.0 * next_unit(&mut state);
        let (time, ev) = if t_event <= t_cens { (t_event, 1u8) } else { (t_cens, 0u8) };
        events += usize::from(ev == 1);
        xs.push(x);
        rows.push(StringRecord::from(vec![
            time.max(1e-6).to_string(), ev.to_string(), z.to_string(), zeta.to_string(), x.to_string(),
        ]));
    }
    eprintln!("[2768-probe] n={N} events={events} ({:.1}%)", 100.0 * events as f64 / N as f64);
    (encode_recordswith_inferred_schema(headers, rows).expect("encode"), xs)
}

fn slope_on_x(values: &[f64], x: &[f64]) -> f64 {
    let n = values.len() as f64;
    let xm = x.iter().sum::<f64>() / n;
    let vm = values.iter().sum::<f64>() / n;
    let mut cov = 0.0;
    let mut var = 0.0;
    for i in 0..values.len() {
        let dx = x[i] - xm;
        cov += dx * (values[i] - vm);
        var += dx * dx;
    }
    cov / var
}

#[test]
fn probe_2768_which_shapes_converge() {
    init_parallelism();
    #[cfg(target_os = "macos")]
    gam::gpu::configure_global_policy(gam::gpu::GpuPolicy::Off);
    let (data, xs) = build();
    let shapes: [(&str, &str, &str); 4] = [
        ("linear/linear", "Surv(time, event) ~ x", "x"),
        ("linear/intercept", "Surv(time, event) ~ x", "1"),
        ("smooth/smooth", "Surv(time, event) ~ smooth(x)", "smooth(x)"),
        ("smooth/linear", "Surv(time, event) ~ smooth(x)", "x"),
    ];
    let mut any = false;
    for (label, formula, logslope) in shapes {
        for z_column in ["z", "zeta"] {
            let cfg = FitConfig {
                survival_likelihood: Some("marginal-slope".to_string()),
                z_column: Some(z_column.to_string()),
                logslope_formula: Some(logslope.to_string()),
                baseline_target: "linear".to_string(),
                ..FitConfig::default()
            };
            match fit_from_formula(formula, &data, &cfg) {
                Ok(FitResult::SurvivalMarginalSlope(fit)) => {
                    any = true;
                    let marginal_eta = fit.marginal_design.design.dot(&fit.fit.blocks[1].beta);
                    let logslope_eta = fit.logslope_design.design.dot(&fit.fit.blocks[2].beta);
                    let mean_slope = fit.baseline_slope
                        + logslope_eta.iter().sum::<f64>() / logslope_eta.len() as f64;
                    let cal = match fit.latent_z_calibrations.first() {
                        Some(gam::families::bms::LatentMeasureCalibration::None) => "none",
                        Some(gam::families::bms::LatentMeasureCalibration::RankInverseNormal(_)) => "rank-int",
                        Some(gam::families::bms::LatentMeasureCalibration::ConditionalLocationScale(_)) => "conditional",
                        None => "missing",
                    };
                    eprintln!(
                        "[2768-probe] {label:18} z={z_column:5} OK  cal={cal:11} beta_x={:.4} slope={:.4} declined={:?}",
                        slope_on_x(marginal_eta.as_slice().unwrap(), &xs),
                        mean_slope,
                        fit.fit.artifacts.covariance_declined.is_some(),
                    );
                }
                Ok(_) => eprintln!("[2768-probe] {label:18} z={z_column:5} WRONG-VARIANT"),
                Err(e) => {
                    let msg = e.to_string();
                    let head = msg.lines().next().unwrap_or("");
                    eprintln!("[2768-probe] {label:18} z={z_column:5} FAIL {head}");
                }
            }
        }
    }
    assert!(any, "no survival marginal-slope shape converged on this fixture at all");
}
