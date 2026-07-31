//! #2644: the outer smoothing-parameter optimizer refusing to certify a
//! stationary optimum at an interior PSD minimum.
//!
//! ONE GATE and two reporting probes.
//!
//! ## The gate — `te_with_disparate_scales_certifies`
//!
//! `y ~ te(a,b,k=5)` on 300 rows with `a ∈ [0,1]` against `b ∈ [0,1000]`. This
//! is `misc::mega_batch_k::te_with_disparate_scales`, a `GAM_ERROR` row on both
//! of the last two reference-quality nightlies, and it moved BETWEEN the two
//! shapes the refusal takes:
//!
//! ```text
//! run 30602192415:  cost-only value disagrees with analytic-sample value at
//!                   the same outer point (4.141e-5 vs a 2.842e-5 roundoff bound)
//! run 30619084852:  |g|=|Pg|=2.458e2  bound=1.015e1  hessian_psd=yes
//!                   after exhausting a 400-iteration outer budget
//! ```
//!
//! Both are the same criterion. Its penalty blocks reach `κ(S_λ) = 3.5e19`,
//! where the assembled-matrix Cholesky that used to price `log|S_λ|₊` disagrees
//! with the root-scale spectrum by **20 log units** — and, because that Cholesky
//! FAILS at some trial `rho` and succeeds at others, `V(rho)` was priced by two
//! different formulas with a step of that size between them. 196 of this fit's
//! 1015 penalty-logdet builds took one formula and 819 the other.
//!
//! Measured here: **refused after 400 outer iterations in 0.94 s** before
//! `75563f13e`, **certifies in 0.18 s** after it.
//!
//! The gate lives in `tests/` rather than only in the nightly quality suite
//! because the quality suite runs on a 6-hour schedule and this is a
//! sub-second fit.
//!
//! ## Reporting probes (print, never fail)
//!
//! * `prostate` — `y ~ s(pc1,k=5) + s(pc2,k=5)`, binomial/logit, 490 rows of
//!   `bench/datasets/prostate.csv`. Three reference-quality rows report the
//!   same `|g|=|Pg|=1.792e-3 bound=1.133e-3` from it. STILL REFUSES after the
//!   `log|S_λ|₊` fix, now at `|Pg|=3.396e-3` against `bound=3.015e-3`, because
//!   its residual criterion noise is `log|H|`, which is priced from the
//!   eigenvalues of the ASSEMBLED `H` (`κ = 3.8e12` ⇒ `6.6e-4` of scatter at
//!   fixed `rho`, against an outer cost floor of `1.8e-5`). Recovering that one
//!   needs a root of `H` that is not formed by summing `XᵀWX` and `S_λ` in f64,
//!   which is a change to what P-IRLS publishes; it is not attempted here.
//! * `matern` — the fit the issue thread recommends as a reproducer. Recorded
//!   because it does NOT reproduce any more (it certifies at `|Pg|=2.025e-5`
//!   against `bound=9.876e-5`).
//!
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

// The gate. See the module header.

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
fn te_with_disparate_scales_certifies() {
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
    let Err(error) = outcome else {
        println!("[probe-2644-te] FIT OK");
        return;
    };
    panic!(
        "#2644: `y~te(a,b,k=5)` on disparate scales must reach a certified outer \
         optimum. This fit refused for 400 outer iterations at |Pg|=2.458e2 while \
         `log|S_lambda|+` was priced from a factorization of the assembled penalty \
         sum, whose error is O(eps*kappa) and whose Cholesky FAILS at some trial \
         rho and succeeds at others -- so the objective carried a step of ~20 log \
         units. A refusal here means that pricing (or an equivalent one) is back: \
         {error}"
    );
}
