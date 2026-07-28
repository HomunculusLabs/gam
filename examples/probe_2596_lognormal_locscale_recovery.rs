//! Probe (NOT a permanent guard): replicate the #2596 synthetic arm of
//! `quality_vs_survival_location_scale_lognormal` WITHOUT the R reference, and
//! print the smoothing diagnostics the issue asks for first: the selected
//! lambda(s) on the location `s(z)` block, its EDF, the recovered log-sigma,
//! and the truth-recovery RMSE.
//!
//! Run with:
//!   cargo run --example probe_2596_lognormal_locscale_recovery

use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::rmse;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;

fn z_effect_truth(z: f64) -> f64 {
    use std::f64::consts::PI;
    (PI * z).sin() + 0.5 * (3.0 * PI * z).sin()
}

struct NumpyMt19937 {
    mt: [u32; 624],
    idx: usize,
    has_gauss: bool,
    gauss: f64,
}

impl NumpyMt19937 {
    fn new(seed: u32) -> Self {
        let mut mt = [0u32; 624];
        mt[0] = seed;
        for i in 1..624 {
            mt[i] = 1812433253u32
                .wrapping_mul(mt[i - 1] ^ (mt[i - 1] >> 30))
                .wrapping_add(i as u32);
        }
        Self {
            mt,
            idx: 624,
            has_gauss: false,
            gauss: 0.0,
        }
    }

    fn generate(&mut self) {
        const MATRIX_A: u32 = 0x9908b0df;
        const UPPER: u32 = 0x80000000;
        const LOWER: u32 = 0x7fffffff;
        for i in 0..624 {
            let y = (self.mt[i] & UPPER) | (self.mt[(i + 1) % 624] & LOWER);
            let mut next = self.mt[(i + 397) % 624] ^ (y >> 1);
            if y & 1 != 0 {
                next ^= MATRIX_A;
            }
            self.mt[i] = next;
        }
        self.idx = 0;
    }

    fn next_u32(&mut self) -> u32 {
        if self.idx >= 624 {
            self.generate();
        }
        let mut y = self.mt[self.idx];
        self.idx += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c5680;
        y ^= (y << 15) & 0xefc60000;
        y ^= y >> 18;
        y
    }

    fn next_f64(&mut self) -> f64 {
        let a = (self.next_u32() >> 5) as u64;
        let b = (self.next_u32() >> 6) as u64;
        (a as f64 * 67108864.0 + b as f64) / 9007199254740992.0
    }

    fn next_standard_normal(&mut self) -> f64 {
        if self.has_gauss {
            self.has_gauss = false;
            return self.gauss;
        }
        loop {
            let x1 = 2.0 * self.next_f64() - 1.0;
            let x2 = 2.0 * self.next_f64() - 1.0;
            let r2 = x1 * x1 + x2 * x2;
            if r2 < 1.0 && r2 != 0.0 {
                let f = (-2.0 * r2.ln() / r2).sqrt();
                self.gauss = f * x1;
                self.has_gauss = true;
                return f * x2;
            }
        }
    }
}

/// The exact #2596 fixture data (same seed, same draw order).
fn fixture() -> (usize, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, f64) {
    let n = 300usize;
    let mut rng = NumpyMt19937::new(2471);
    let g2 = -rng.next_f64().ln() - rng.next_f64().ln();
    let sigma_true = (1.0 / g2).sqrt();
    let x: Vec<f64> = (0..n).map(|_| 2.0 * rng.next_f64() - 1.0).collect();
    let z: Vec<f64> = (0..n).map(|_| 2.0 * rng.next_f64() - 1.0).collect();
    let eps: Vec<f64> = (0..n).map(|_| rng.next_standard_normal()).collect();
    let cens_u: Vec<f64> = (0..n).map(|_| rng.next_f64()).collect();
    let mut t = Vec::with_capacity(n);
    let mut event = Vec::with_capacity(n);
    for i in 0..n {
        let eta_loc = -0.5 + 0.8 * x[i] + z_effect_truth(z[i]);
        let t_event = (eta_loc + eps[i] * sigma_true).exp();
        let c = -cens_u[i].ln() * 0.9;
        if t_event <= c {
            t.push(t_event);
            event.push(1.0);
        } else {
            t.push(c.max(1e-6));
            event.push(0.0);
        }
    }
    (n, t, event, x, z, sigma_true)
}


/// Fit one survival location-scale arm and print the full smoothing readout.
fn run_survival_case(label: &str, formula: &str, cfg_mut: impl FnOnce(&mut FitConfig)) {
    let (n, t, event, x, z, sigma_true) = fixture();
    let headers: Vec<String> = ["t", "event", "x", "z"]
        .into_iter()
        .map(str::to_string)
        .collect();
    let rows: Vec<csv::StringRecord> = (0..n)
        .map(|i| {
            csv::StringRecord::from(vec![
                format!("{:.17e}", t[i]),
                format!("{:.17e}", event[i]),
                format!("{:.17e}", x[i]),
                format!("{:.17e}", z[i]),
            ])
        })
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode");
    let col = ds.column_map();
    let x_idx = col["x"];
    let z_idx = col["z"];
    let ncols = ds.headers.len();

    let mut cfg = FitConfig {
        survival_likelihood: Some("location-scale".to_string()),
        survival_distribution: "gaussian".to_string(),
        time_num_internal_knots: 2,
        outer_max_iter: Some(80),
        ..FitConfig::default()
    };
    cfg_mut(&mut cfg);

    eprintln!("\n================ CASE {label}: {formula} ================");
    let result = match fit_from_formula(formula, &ds, &cfg) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("[{label}] FIT ERROR: {e}");
            return;
        }
    };
    let FitResult::SurvivalLocationScale(fit) = result else {
        panic!("expected a survival location-scale fit result");
    };
    let unified = &fit.fit.fit;

    eprintln!("block roles: {:?}", unified.block_roles());
    for b in unified.blocks.iter() {
        eprintln!(
            "  role={:?} p={} lambdas={:?} edf={:.4}",
            b.role,
            b.beta.len(),
            b.lambdas.to_vec(),
            b.edf,
        );
        eprintln!("    beta={:?}", b.beta.to_vec());
    }
    eprintln!("edf_total = {:?}", unified.edf_total());

    let beta_location = unified.beta_threshold();
    let beta_log_sigma = unified.beta_log_sigma();

    let mut train_grid = Array2::<f64>::zeros((n, ncols));
    for i in 0..n {
        train_grid[[i, x_idx]] = x[i];
        train_grid[[i, z_idx]] = z[i];
    }
    let loc_design =
        build_term_collection_design(train_grid.view(), &fit.fit.resolved_thresholdspec)
            .expect("rebuild location design");
    let gam_mu_train: Vec<f64> = loc_design.design.apply(&beta_location).to_vec();

    // Per-column contribution magnitude: which location columns actually carry
    // signal, and which sit at (numerically) zero.
    let dense = loc_design.design.to_dense();
    eprintln!(
        "location design: {} rows x {} cols; per-column rms(X_j)*|beta_j| =",
        dense.nrows(),
        dense.ncols()
    );
    for j in 0..dense.ncols().min(beta_location.len()) {
        let colrms =
            (dense.column(j).iter().map(|v| v * v).sum::<f64>() / dense.nrows() as f64).sqrt();
        eprintln!(
            "   col {j:2}: rms(X)={colrms:.4e} beta={:+.6e} contrib_rms={:.4e}",
            beta_location[j],
            colrms * beta_location[j].abs()
        );
    }

    let ls_design =
        build_term_collection_design(train_grid.view(), &fit.fit.resolved_log_sigmaspec)
            .expect("rebuild log-sigma design");
    let gam_eta_ls: Vec<f64> = ls_design.design.apply(&beta_log_sigma).to_vec();
    let gam_log_sigma = gam_eta_ls[0];

    let truth: Vec<f64> = (0..n).map(|i| 0.8 * x[i] + z_effect_truth(z[i])).collect();
    let truth_mean = truth.iter().sum::<f64>() / n as f64;
    let truth_c: Vec<f64> = truth.iter().map(|&m| m - truth_mean).collect();
    let gam_mean = gam_mu_train.iter().sum::<f64>() / n as f64;
    let gam_mu_c: Vec<f64> = gam_mu_train.iter().map(|&m| m - gam_mean).collect();

    let gam_truth_rmse = rmse(&gam_mu_c, &truth_c);
    let signal_rms = (truth_c.iter().map(|v| v * v).sum::<f64>() / n as f64).sqrt();
    let fitted_rms = (gam_mu_c.iter().map(|v| v * v).sum::<f64>() / n as f64).sqrt();
    let num: f64 = gam_mu_c.iter().zip(&truth_c).map(|(&g, &tr)| g * tr).sum();
    let den: f64 = truth_c.iter().map(|v| v * v).sum::<f64>();

    eprintln!(
        "[{label}] signal_rms={signal_rms:.4} fitted_rms={fitted_rms:.4} \
         truth_rmse={gam_truth_rmse:.4} shrink_slope={:.4} \
         log_sigma gam={gam_log_sigma:.4} truth={:.4} err={:.4}",
        num / den,
        sigma_true.ln(),
        (gam_log_sigma - sigma_true.ln()).abs()
    );
}

fn probe_fixture() {
    run_survival_case(
        "A-fixture-k10",
        r#"Surv(t, event) ~ x + s(z, bs="tp", k=10)"#,
        |_| {},
    );
}

fn probe_2596_variants() {
    run_survival_case(
        "B-k5",
        r#"Surv(t, event) ~ x + s(z, bs="tp", k=5)"#,
        |_| {},
    );
    run_survival_case(
        "C-single-penalty",
        r#"Surv(t, event) ~ x + s(z, bs="tp", k=10, double_penalty=false)"#,
        |_| {},
    );
    run_survival_case(
        "D-pspline",
        r#"Surv(t, event) ~ x + s(z, bs="ps", k=10)"#,
        |_| {},
    );
    run_survival_case(
        "E-default-time-knots",
        r#"Surv(t, event) ~ x + s(z, bs="tp", k=10)"#,
        |c| c.time_num_internal_knots = 8,
    );
}

/// Control: the SAME z-effect and the SAME rows, but a plain uncensored Gaussian
/// GAM on the event rows only. If gam recovers `s(z)` here and not on the
/// survival route, the defect is on the survival route, not in the thin-plate
/// smooth or its REML selection.
fn probe_2596_gaussian_control() {
    let (n, t, event, x, z, sigma_true) = fixture();
    let keep: Vec<usize> = (0..n).filter(|&i| event[i] > 0.5).collect();
    let headers: Vec<String> = ["y", "x", "z"].into_iter().map(str::to_string).collect();
    let rows: Vec<csv::StringRecord> = keep
        .iter()
        .map(|&i| {
            csv::StringRecord::from(vec![
                format!("{:.17e}", t[i].ln()),
                format!("{:.17e}", x[i]),
                format!("{:.17e}", z[i]),
            ])
        })
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode");
    let col = ds.column_map();
    let x_idx = col["x"];
    let z_idx = col["z"];
    let ncols = ds.headers.len();
    let m = keep.len();

    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula(r#"y ~ x + s(z, bs="tp", k=10)"#, &ds, &cfg)
        .expect("gaussian control fit");
    let FitResult::Standard(std_fit) = result else {
        panic!("expected a standard gaussian fit");
    };
    eprintln!("\n================ CONTROL: uncensored gaussian on event rows ================");
    for b in std_fit.fit.blocks.iter() {
        eprintln!(
            "  role={:?} p={} lambdas={:?} edf={:.4}",
            b.role,
            b.beta.len(),
            b.lambdas.to_vec(),
            b.edf,
        );
    }
    eprintln!("edf_total={:?}", std_fit.fit.edf_total());

    let mut grid = Array2::<f64>::zeros((m, ncols));
    for (r, &i) in keep.iter().enumerate() {
        grid[[r, x_idx]] = x[i];
        grid[[r, z_idx]] = z[i];
    }
    let design = build_term_collection_design(grid.view(), &std_fit.resolvedspec)
        .expect("rebuild gaussian design");
    let mu: Vec<f64> = design.design.apply(&std_fit.fit.beta).to_vec();
    let truth: Vec<f64> = keep
        .iter()
        .map(|&i| 0.8 * x[i] + z_effect_truth(z[i]))
        .collect();
    let truth_mean = truth.iter().sum::<f64>() / m as f64;
    let truth_c: Vec<f64> = truth.iter().map(|&v| v - truth_mean).collect();
    let mu_mean = mu.iter().sum::<f64>() / m as f64;
    let mu_c: Vec<f64> = mu.iter().map(|&v| v - mu_mean).collect();
    let signal_rms = (truth_c.iter().map(|v| v * v).sum::<f64>() / m as f64).sqrt();
    eprintln!(
        "[CONTROL] m={m} signal_rms={signal_rms:.4} truth_rmse={:.4} (sigma_true={sigma_true:.4})",
        rmse(&mu_c, &truth_c),
    );
}

fn main() {
    init_parallelism();
    gam::test_support::install_diagnostic_logger();
    probe_fixture();
    probe_2596_gaussian_control();
    probe_2596_variants();
}
