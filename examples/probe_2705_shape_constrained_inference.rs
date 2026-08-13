//! #2705 probe — the shape-constrained inference cluster, end to end, in one
//! binary that takes seconds instead of a 45-minute `gam::regressions` sweep.
//!
//! Two fixtures, four shapes each:
//!
//! * `2601` — 300 rows of `y = 2x + N(0, 0.1²)`, the fixture
//!   `smooths::shape_constrained_fit_survives_its_own_inference_2601` uses.
//! * `1191` — 400 rows of `y = sqrt(x) + N(0, 0.05²)`, the fixture
//!   `misc::shape_constrained_alo_seed_validation_aborts_1191` uses.
//!
//! For every (fixture, shape) it reports the terminal state verbatim, so the
//! group-A negative-diagonal attribution line and the group-B
//! `StalledAtValidMinimum` refusal are both readable without re-deriving which
//! test produces which.

use gam::{FitConfig, FitResult, fit_from_formula, init_parallelism, load_csvwith_inferred_schema};
use std::io::Write;

/// 300 rows of `y = 2x + N(0, 0.1)`; `x` sorted uniform on `[0, 1]`.
fn linear_fixture_2601() -> (Vec<f64>, Vec<f64>) {
    let n = 300usize;
    let mut state: u64 = 0x2601_0000_0000_0005;
    let mut next = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64) / ((1u64 << 53) as f64)
    };
    let mut x: Vec<f64> = (0..n).map(|_| next()).collect();
    x.sort_by(|a, b| a.partial_cmp(b).expect("finite covariate"));
    let y: Vec<f64> = x
        .iter()
        .map(|xi| {
            let u1 = next().max(1e-12);
            let u2 = next();
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            2.0 * xi + 0.1 * z
        })
        .collect();
    (x, y)
}

/// 400 rows of `y = sqrt(x) + N(0, 0.05²)`; the #1191 fixture's SplitMix64.
fn sqrt_fixture_1191() -> (Vec<f64>, Vec<f64>) {
    let n = 400usize;
    let mut state: u64 = 11;
    let mut next_unit = move || -> f64 {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 11) as f64 / (1u64 << 53) as f64
    };
    let mut x = vec![0.0f64; n];
    for xi in x.iter_mut() {
        *xi = next_unit();
    }
    x.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let y: Vec<f64> = x
        .iter()
        .map(|&xi| {
            let u1 = next_unit().max(1e-300);
            let u2 = next_unit();
            let noise = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            xi.sqrt() + 0.05 * noise
        })
        .collect();
    (x, y)
}

fn dataset_from(tag: &str, x: &[f64], y: &[f64]) -> gam::data::Dataset {
    let mut csv = String::from("x,y\n");
    for i in 0..x.len() {
        csv.push_str(&format!("{:.17e},{:.17e}\n", x[i], y[i]));
    }
    let mut tmp = std::env::temp_dir();
    tmp.push(format!("gam_probe_2705_{tag}_{}.csv", std::process::id()));
    {
        let mut file = std::fs::File::create(&tmp).expect("create fixture csv");
        file.write_all(csv.as_bytes()).expect("write fixture csv");
    }
    let dataset = load_csvwith_inferred_schema(&tmp).expect("load fixture");
    std::fs::remove_file(&tmp).expect("remove the fixture csv this probe created");
    dataset
}

fn run(fixture: &str, dataset: &gam::data::Dataset) {
    for kind in [
        "monotone_increasing",
        "monotone_decreasing",
        "convex",
        "concave",
    ] {
        let formula = format!("y ~ s(x, shape={kind})");
        let started = std::time::Instant::now();
        let outcome = fit_from_formula(&formula, dataset, &FitConfig::default());
        let elapsed = started.elapsed().as_secs_f64();
        match outcome {
            Ok(FitResult::Standard(fit)) => {
                let corrected_se_state = fit
                    .fit
                    .inference
                    .as_ref()
                    .map(|inference| {
                        match inference.beta_standard_errors_corrected.as_ref() {
                            Some(se) => {
                                let worst = se.iter().cloned().fold(0.0_f64, f64::max);
                                format!("corrected_se_present max={worst:.6e}")
                            }
                            None => "corrected_se_absent".to_string(),
                        }
                    })
                    .unwrap_or_else(|| "inference_absent".to_string());
                let edf = fit
                    .fit
                    .inference
                    .as_ref()
                    .map_or(f64::NAN, |inference| inference.edf_total);
                println!(
                    "[{fixture}] {kind:<20} OK   ({elapsed:.1}s) edf={edf:.4} {corrected_se_state}"
                );
            }
            Ok(_) => println!("[{fixture}] {kind:<20} OK   ({elapsed:.1}s) non-standard result"),
            Err(error) => println!("[{fixture}] {kind:<20} FAIL ({elapsed:.1}s) {error}"),
        }
    }
}

fn main() {
    init_parallelism();
    let (x, y) = linear_fixture_2601();
    let linear = dataset_from("2601", &x, &y);
    run("2601 linear", &linear);

    let (x, y) = sqrt_fixture_1191();
    let sqrt = dataset_from("1191", &x, &y);
    run("1191 sqrt  ", &sqrt);
}
