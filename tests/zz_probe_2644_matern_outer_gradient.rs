//! Scratch probes for #2644: the outer smoothing-parameter optimizer refusing
//! to certify a stationary optimum.
//!
//! Witness A (`prostate`): the three reference-quality `GAM_ERROR` rows
//! `families::quality_vs_sklearn_binomial_logit::*` and
//! `families::quality_vs_interpretml_ebm_binomial_logit::*` all report the SAME
//! numbers on run 30602192415 (`a20af61da`):
//!
//! ```text
//! standard REML: |g|=1.792e-3 |Pg|=1.792e-3 bound=1.133e-3
//!                (rung=curvature-resolvability)
//! ```
//!
//! `y ~ s(pc1,k=5) + s(pc2,k=5)`, binomial/logit, 490 training rows — the
//! cheapest witness of the refusal in the whole suite.
//!
//! Witness B (`matern`): the `tests/measure_jet` comparator fit named in the
//! issue thread (`|Pg|=2.751e1` vs `bound=6.441e-1`).
//!
//! Reporting probes, not gates.

use gam::data::EncodedDataset;
use gam::{FitConfig, encode_recordswith_inferred_schema, fit_from_formula, load_csvwith_inferred_schema};
use csv::StringRecord;
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};
use std::path::Path;

const PROSTATE_CSV: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/bench/datasets/prostate.csv");

fn subset_rows(ds: &EncodedDataset, rows: &[usize]) -> EncodedDataset {
    let mut sub = ds.clone();
    let ncol = ds.values.ncols();
    let mut values = Array2::<f64>::zeros((rows.len(), ncol));
    for (new_r, &old_r) in rows.iter().enumerate() {
        for c in 0..ncol {
            values[[new_r, c]] = ds.values[[old_r, c]];
        }
    }
    sub.values = values;
    sub
}

#[test]
fn zz_probe_2644_prostate_binomial_logit() {
    gam_solve::progress_log::init_logging_at(log::LevelFilter::Info);
    let ds = load_csvwith_inferred_schema(Path::new(PROSTATE_CSV)).expect("load prostate.csv");
    let n = ds.values.nrows();
    let train_rows: Vec<usize> = (0..n).filter(|i| i % 4 != 0).collect();
    let ds_train = subset_rows(&ds, &train_rows);
    let cfg = FitConfig {
        family: Some("binomial".to_string()),
        link: Some("logit".to_string()),
        ..FitConfig::default()
    };
    let started = std::time::Instant::now();
    let outcome = fit_from_formula("y ~ s(pc1, k=5) + s(pc2, k=5)", &ds_train, &cfg);
    println!(
        "[probe-2644-prostate] n_train={} elapsed={:.2}s",
        train_rows.len(),
        started.elapsed().as_secs_f64()
    );
    match outcome {
        Ok(_) => println!("[probe-2644-prostate] FIT OK"),
        Err(e) => println!("[probe-2644-prostate] FIT ERR: {e}"),
    }
}

const N_TRAIN: usize = 1_500;
const SIGMA: f64 = 0.10;
const TRAIN_SEED: u64 = 1_039;

fn clamp_unit_open(x: f64) -> f64 {
    x.max(1.0e-6).min(1.0 - 1.0e-6)
}

fn latent_to_coords(t: f64) -> [f64; 3] {
    [
        clamp_unit_open(t),
        clamp_unit_open(0.5 + 0.5 * (2.0 * std::f64::consts::PI * t).sin()),
        clamp_unit_open(t * t),
    ]
}

fn truth(t: f64) -> f64 {
    (2.0 * std::f64::consts::PI * t).sin() + 0.5 * (4.0 * std::f64::consts::PI * t).cos()
}

fn build_dataset(n: usize, sigma: f64, seed: u64) -> EncodedDataset {
    let mut rng = StdRng::seed_from_u64(seed);
    let latent = Uniform::new(0.0, 1.0).expect("uniform");
    let noise = Normal::new(0.0, sigma).expect("normal");
    let headers = ["x0", "x1", "x2", "y"]
        .iter()
        .cloned()
        .map(String::from)
        .collect();
    let rows: Vec<StringRecord> = (0..n)
        .map(|_| {
            let t = latent.sample(&mut rng);
            let coords = latent_to_coords(t);
            let y = truth(t) + noise.sample(&mut rng);
            StringRecord::from(vec![
                coords[0].to_string(),
                coords[1].to_string(),
                coords[2].to_string(),
                y.to_string(),
            ])
        })
        .collect();
    encode_recordswith_inferred_schema(headers, rows).expect("encode")
}

#[test]
fn zz_probe_2644_matern_outer_gradient() {
    gam_solve::progress_log::init_logging_at(log::LevelFilter::Info);
    let ds = build_dataset(N_TRAIN, SIGMA, TRAIN_SEED);
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let started = std::time::Instant::now();
    let outcome = fit_from_formula("y ~ matern(x0, x1, x2, k=16)", &ds, &cfg);
    println!("[probe-2644] elapsed={:.2}s", started.elapsed().as_secs_f64());
    match outcome {
        Ok(_) => println!("[probe-2644] FIT OK"),
        Err(e) => println!("[probe-2644] FIT ERR: {e}"),
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Witness C: `misc::mega_batch_k::te_with_disparate_scales`.
//
// `y ~ te(a,b,k=5)`, Gaussian, 300 rows, `a ∈ [0,1]` against `b ∈ [0,1000]`.
// On run 30619084852 it refuses with
//   |g| = |Pg| = 2.458e2  bound = 1.015e1  (rung=curvature-resolvability,
//   derived_standard=true, hessian_psd=yes)
// and on the PREVIOUS run (30602192415) the SAME test refused with the OTHER
// shape — the cost-only/analytic-sample value disagreement. One second.
// ─────────────────────────────────────────────────────────────────────────

fn mk_2d(
    n: usize,
    f: impl Fn(f64, f64) -> f64,
    ra: (f64, f64),
    rb: (f64, f64),
    sigma: f64,
    seed: u64,
) -> EncodedDataset {
    let mut rng = StdRng::seed_from_u64(seed);
    let ua = Uniform::new(ra.0, ra.1).expect("finite a range");
    let ub = Uniform::new(rb.0, rb.1).expect("finite b range");
    let noise = Normal::new(0.0, sigma).expect("finite sigma");
    let h = ["a", "b", "y"].into_iter().map(String::from).collect();
    let mut rows = Vec::with_capacity(n);
    for _ in 0..n {
        let a = ua.sample(&mut rng);
        let b = ub.sample(&mut rng);
        let y = f(a, b) + noise.sample(&mut rng);
        rows.push(StringRecord::from(vec![
            a.to_string(),
            b.to_string(),
            y.to_string(),
        ]));
    }
    encode_recordswith_inferred_schema(h, rows).expect("encode")
}

#[test]
fn zz_probe_2644_te_disparate_scales() {
    gam_solve::progress_log::init_logging_at(log::LevelFilter::Info);
    let ds = mk_2d(300, |a, b| a + b, (0.0, 1.0), (0.0, 1000.0), 0.05, 7);
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let started = std::time::Instant::now();
    let outcome = fit_from_formula("y~te(a,b,k=5)", &ds, &cfg);
    println!(
        "[probe-2644-te] elapsed={:.2}s",
        started.elapsed().as_secs_f64()
    );
    match outcome {
        Ok(_) => println!("[probe-2644-te] FIT OK"),
        Err(e) => println!("[probe-2644-te] FIT ERR: {e}"),
    }
}
