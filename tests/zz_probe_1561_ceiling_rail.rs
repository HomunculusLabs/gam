//! #1561 ceiling-rail probe: on some seeds the by-group SCALE block selects a
//! lambda equal to exp(12) = exp(EFFECTIVE_DF_CEILING) bit-exactly, and the
//! widening experiment on `EXACT_JOINT_RHO_BOUND` (issue comment 07-26 04:09Z)
//! proved that constant is NOT the rail. The remaining candidate is the
//! custom-family outer rho box ceiling (`gam-custom-family/src/fit.rs`,
//! `EFFECTIVE_DF_CEILING`), which this fixture reaches because with no spatial
//! terms the exact-joint (rho, psi) optimizer is inactive and rho selection
//! happens inside `fit_custom_family`.
//!
//! Arms: rebuild with the ceiling at 12 (baseline) / 14 (decisive move) / 20
//! (absurd arm) and re-run THIS probe unchanged. If the railed coordinate
//! tracks exp(ceiling), the ceiling is the rail; the per-arm truth-RMSEs bound
//! what un-capping is worth to the #1561 quality gate. Seed 321 is the quality
//! test's exact fixture (same RNG, same draw order); 301/304 are the two
//! sigma-block-probe seeds that railed on 07-26.
//!
//! Also dumps each seed's data to `zz1561_ceiling_seed<seed>.csv` in the OS
//! temp dir so the gamlss reference arm can be fit on identical bytes from R
//! (temp dir, not cwd: a landed test must not dirty a shared tree).

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::rmse;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};

const N_PER_GROUP: usize = 100;
const GRID_POINTS: usize = 50;

fn mean_a(x: f64) -> f64 {
    (2.0 * std::f64::consts::PI * x).sin()
}
fn sigma_a(x: f64) -> f64 {
    0.10 + 0.10 * (std::f64::consts::PI * x).sin()
}
fn mean_b(x: f64) -> f64 {
    0.5 + 0.3 * (3.0 * std::f64::consts::PI * x).sin()
}
fn sigma_b(x: f64) -> f64 {
    0.12 + 0.08 * x
}

fn linspace(a: f64, b: f64) -> Vec<f64> {
    (0..GRID_POINTS)
        .map(|i| a + (b - a) * (i as f64) / ((GRID_POINTS - 1) as f64))
        .collect()
}

#[test]
fn zz_probe_1561_ceiling_rail() {
    init_parallelism();
    eprintln!("[zz1561rail] ceiling-rail probe start");

    for seed in [321u64, 301, 304] {
        // ---- data: EXACT replication of the quality test / sigma probe draw
        // order (x then y per row, group A rows first) --------------------
        let mut rng = StdRng::seed_from_u64(seed);
        let ux = Uniform::new(0.0_f64, 1.0_f64).expect("uniform x");
        let sn = Normal::new(0.0_f64, 1.0_f64).expect("standard normal");
        let headers = vec!["y".to_string(), "x".to_string(), "group".to_string()];
        let mut rows: Vec<StringRecord> = Vec::with_capacity(2 * N_PER_GROUP);
        let mut x_a = Vec::with_capacity(N_PER_GROUP);
        let mut x_b = Vec::with_capacity(N_PER_GROUP);
        let mut csv_lines = String::from("y,x,group\n");
        for _ in 0..N_PER_GROUP {
            let x = ux.sample(&mut rng);
            let y = mean_a(x) + sigma_a(x) * sn.sample(&mut rng);
            x_a.push(x);
            csv_lines.push_str(&format!("{y},{x},A\n"));
            rows.push(StringRecord::from(vec![
                y.to_string(),
                x.to_string(),
                "A".to_string(),
            ]));
        }
        for _ in 0..N_PER_GROUP {
            let x = ux.sample(&mut rng);
            let y = mean_b(x) + sigma_b(x) * sn.sample(&mut rng);
            x_b.push(x);
            csv_lines.push_str(&format!("{y},{x},B\n"));
            rows.push(StringRecord::from(vec![
                y.to_string(),
                x.to_string(),
                "B".to_string(),
            ]));
        }
        let dump = std::env::temp_dir().join(format!("zz1561_ceiling_seed{seed}.csv"));
        std::fs::write(&dump, &csv_lines).expect("dump csv");

        let data = encode_recordswith_inferred_schema(headers, rows).expect("encode data");
        let col = data.column_map();
        let x_idx = col["x"];
        let group_idx = col["group"];
        let ncols = data.headers.len();

        let cfg = FitConfig {
            family: Some("gaussian".to_string()),
            noise_formula: Some("s(x, bs='tp', by=group)".to_string()),
            ..FitConfig::default()
        };
        let result = match fit_from_formula("y ~ s(x, bs='tp', by=group)", &data, &cfg) {
            Ok(r) => r,
            Err(e) => {
                let m = e.to_string();
                eprintln!(
                    "[zz1561rail] seed={seed} REFUSED: {}",
                    &m[..m.len().min(160)]
                );
                continue;
            }
        };
        let FitResult::GaussianLocationScale(fit) = result else {
            eprintln!("[zz1561rail] seed={seed} unexpected fit kind");
            continue;
        };

        let loc = fit
            .fit
            .fit
            .block_by_role(gam::solver::estimate::BlockRole::Location)
            .expect("location block");
        let sca = fit
            .fit
            .fit
            .block_by_role(gam::solver::estimate::BlockRole::Scale)
            .expect("scale block");

        let fmt_rho = |lams: &ndarray::Array1<f64>| -> String {
            lams.iter()
                .map(|&l| format!("{:.6}", l.ln()))
                .collect::<Vec<_>>()
                .join(", ")
        };
        eprintln!(
            "[zz1561rail] seed={seed} SCALE p={} edf={:.4} lambdas={:?} rho=[{}]",
            sca.beta.len(),
            sca.edf,
            sca.lambdas.to_vec(),
            fmt_rho(&sca.lambdas)
        );
        eprintln!(
            "[zz1561rail] seed={seed} LOC   p={} edf={:.4} lambdas={:?} rho=[{}]",
            loc.beta.len(),
            loc.edf,
            loc.lambdas.to_vec(),
            fmt_rho(&loc.lambdas)
        );

        // ---- truth RMSE on the quality test's own grids ---------------------
        let bounds = |v: &[f64]| {
            v.iter()
                .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &t| {
                    (lo.min(t), hi.max(t))
                })
        };
        let (a_lo, a_hi) = bounds(&x_a);
        let (b_lo, b_hi) = bounds(&x_b);
        let grid_a = linspace(a_lo, a_hi);
        let grid_b = linspace(b_lo, b_hi);

        let predict_group = |grid: &[f64], code: f64| -> (Vec<f64>, Vec<f64>) {
            let mut design_pts = Array2::<f64>::zeros((grid.len(), ncols));
            for (i, &xv) in grid.iter().enumerate() {
                design_pts[[i, x_idx]] = xv;
                design_pts[[i, group_idx]] = code;
            }
            let mean_design =
                build_term_collection_design(design_pts.view(), &fit.fit.meanspec_resolved)
                    .expect("rebuild mean design at grid");
            let noise_design =
                build_term_collection_design(design_pts.view(), &fit.fit.noisespec_resolved)
                    .expect("rebuild noise design at grid");
            let mu = mean_design.design.apply(&loc.beta).to_vec();
            let log_sigma = noise_design.design.apply(&sca.beta).to_vec();
            (mu, log_sigma)
        };

        let (gam_mu_a, gam_ls_a) = predict_group(&grid_a, 0.0);
        let (gam_mu_b, gam_ls_b) = predict_group(&grid_b, 1.0);

        let true_mu_a: Vec<f64> = grid_a.iter().map(|&x| mean_a(x)).collect();
        let true_ls_a: Vec<f64> = grid_a.iter().map(|&x| sigma_a(x).ln()).collect();
        let true_mu_b: Vec<f64> = grid_b.iter().map(|&x| mean_b(x)).collect();
        let true_ls_b: Vec<f64> = grid_b.iter().map(|&x| sigma_b(x).ln()).collect();

        eprintln!(
            "[zz1561rail] seed={seed} RMSE mu_a={:.6} log_sigma_a={:.6} mu_b={:.6} log_sigma_b={:.6}",
            rmse(&gam_mu_a, &true_mu_a),
            rmse(&gam_ls_a, &true_ls_a),
            rmse(&gam_mu_b, &true_mu_b),
            rmse(&gam_ls_b, &true_ls_b)
        );
    }
    eprintln!("[zz1561rail] done");
}
