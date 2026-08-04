//! Temporary shape probe for the gam#2768 conditional-latent fixture.
//!
//! Round 1 (probit-transformation DGP, `q = γ₀ + γ₁·log t + β_x·x`, right
//! censoring, no truncation) stalled on EVERY shape tried — linear/linear,
//! linear/intercept, smooth/smooth, smooth/linear — and, decisively, on the
//! `zeta` arm too, which runs no calibration at all. So the stall is a property
//! of that DGP against this family's outer search, not of the gate.
//!
//! Round 2 takes the data-generating process from
//! `survival_marginal_slope_weibull_n3000_seed_startup_2627` — which exists
//! precisely because it is the small, well-conditioned frame for this family —
//! and injects the conditional shift into its latent score. Two score strengths,
//! because the leakage the gate removes is proportional to the fitted slope and a
//! score the outcome barely depends on would make a weak gate look like a
//! working one.
//!
//! Prints only. Deleted once the fixture's shape is locked in.

use csv::StringRecord;
use gam::utils::splitmix64;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};

const N: usize = 3_000;
const M_SHIFT: f64 = 0.6;

fn next_unit(state: &mut u64) -> f64 {
    (splitmix64(state) >> 11) as f64 / (1u64 << 53) as f64
}
fn next_gauss(state: &mut u64) -> f64 {
    let u1 = next_unit(state).max(f64::MIN_POSITIVE);
    let u2 = next_unit(state);
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}
fn next_weibull(state: &mut u64, shape: f64) -> f64 {
    let u = next_unit(state).max(f64::MIN_POSITIVE);
    (-u.ln()).powf(1.0 / shape)
}
fn standardize(v: &[f64]) -> Vec<f64> {
    let n = v.len() as f64;
    let mean = v.iter().sum::<f64>() / n;
    let var = v.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / n;
    let sd = var.sqrt().max(1e-12);
    v.iter().map(|x| (x - mean) / sd).collect()
}

/// The #2627 Weibull frame with a conditionally shifted latent score.
/// `score_strength` is the coefficient on the CLEAN score `ζ` in the log scale.
fn build(score_strength: f64, seed: u64) -> (gam::inference::data::EncodedDataset, Vec<f64>) {
    let headers = ["entry", "exit", "event", "bmi", "hba1c", "z", "zeta"]
        .iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();
    let mut state = seed;
    let mut bmi = Vec::with_capacity(N);
    let mut hba1c = Vec::with_capacity(N);
    let mut zeta = Vec::with_capacity(N);
    let mut entry = Vec::with_capacity(N);
    for _ in 0..N {
        bmi.push(27.0 + 4.5 * next_gauss(&mut state));
        hba1c.push(5.8 + 0.7 * next_gauss(&mut state));
        zeta.push(next_gauss(&mut state));
        entry.push(40.0 + 25.0 * next_unit(&mut state));
    }
    let bmi_z = standardize(&bmi);
    let hba1c_z = standardize(&hba1c);
    let zeta = standardize(&zeta);
    // `z = m·bmi_z + √(1−m²)·ζ`: exactly standard normal marginally (both parts
    // are, and they are independent), and conditionally shifted on `bmi` — which
    // is in the marginal formula, so the shift lands squarely in the influence
    // channel the gate protects.
    let residual_sd = (1.0 - M_SHIFT * M_SHIFT).sqrt();
    let z: Vec<f64> = (0..N)
        .map(|i| M_SHIFT * bmi_z[i] + residual_sd * zeta[i])
        .collect();

    const SHAPE: f64 = 1.45;
    let mut rows = Vec::with_capacity(N);
    let mut events = 0usize;
    for i in 0..N {
        let log_scale =
            18.0_f64.ln() - 0.18 * bmi_z[i] - 0.22 * hba1c_z[i] + score_strength * zeta[i];
        let event_gap = log_scale.exp() * next_weibull(&mut state, SHAPE);
        let censor_gap = 6.0 + 26.0 * next_unit(&mut state);
        let observed = event_gap.min(censor_gap);
        let event = if event_gap <= censor_gap { 1 } else { 0 };
        events += event;
        rows.push(StringRecord::from(vec![
            entry[i].to_string(),
            (entry[i] + observed).to_string(),
            event.to_string(),
            bmi[i].to_string(),
            hba1c[i].to_string(),
            z[i].to_string(),
            zeta[i].to_string(),
        ]));
    }
    eprintln!(
        "[2768-probe] strength={score_strength} n={N} events={events} ({:.1}%)",
        100.0 * events as f64 / N as f64
    );
    (
        encode_recordswith_inferred_schema(headers, rows).expect("encode"),
        bmi_z,
    )
}

fn slope_on(values: &[f64], x: &[f64]) -> f64 {
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
    let logslope = "smooth(bmi) + smooth(hba1c)";
    let formula = format!("Surv(entry, exit, event) ~ {logslope}");
    let mut any = false;
    for strength in [-0.10_f64, -0.45] {
        let (data, bmi_z) = build(strength, 0x2768_0BEE_F00D_0001 ^ strength.to_bits());
        for z_column in ["z", "zeta"] {
            let cfg = FitConfig {
                survival_likelihood: Some("marginal-slope".to_string()),
                z_column: Some(z_column.to_string()),
                logslope_formula: Some(logslope.to_string()),
                ..FitConfig::default()
            };
            match fit_from_formula(&formula, &data, &cfg) {
                Ok(FitResult::SurvivalMarginalSlope(fit)) => {
                    any = true;
                    let marginal_eta = fit.marginal_design.design.dot(&fit.fit.blocks[1].beta);
                    let logslope_eta = fit.logslope_design.design.dot(&fit.fit.blocks[2].beta);
                    let mean_slope = fit.baseline_slope
                        + logslope_eta.iter().sum::<f64>() / logslope_eta.len() as f64;
                    let cal = match fit.latent_z_calibrations.first() {
                        Some(gam::families::bms::LatentMeasureCalibration::None) => "none",
                        Some(gam::families::bms::LatentMeasureCalibration::RankInverseNormal(_)) => {
                            "rank-int"
                        }
                        Some(
                            gam::families::bms::LatentMeasureCalibration::ConditionalLocationScale(
                                _,
                            ),
                        ) => "conditional",
                        None => "missing",
                    };
                    eprintln!(
                        "[2768-probe] strength={strength:5} z={z_column:5} OK cal={cal:11} \
                         beta_bmi={:.4} slope={:.4} cov={} declined={:?}",
                        slope_on(marginal_eta.as_slice().unwrap(), &bmi_z),
                        mean_slope,
                        fit.fit.covariance_conditional.is_some(),
                        fit.fit.artifacts.covariance_declined.is_some(),
                    );
                }
                Ok(_) => eprintln!("[2768-probe] strength={strength} z={z_column} WRONG-VARIANT"),
                Err(e) => {
                    let msg = e.to_string();
                    let head: String = msg.chars().take(260).collect();
                    eprintln!("[2768-probe] strength={strength} z={z_column} FAIL {head}");
                }
            }
        }
    }
    assert!(any, "no survival marginal-slope shape converged at all");
}
