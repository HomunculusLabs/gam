//! #2761 probe: what does the profiled REML criterion actually look like along
//! the measure-jet representer range `ℓ`?
//!
//! `learn_length_scale` is on by default again (#2761, `6324d1c0b`), and the
//! `aniso-psi joint REML` search now refuses on five `measure_jet` fixtures with
//! `line_search=StepSizeTooSmall after 50 attempt(s)` — "the direction descended
//! but no step improved the objective". That message has exactly two causes:
//! the analytic gradient disagrees with the objective, or the objective is not
//! smooth in the coordinate being searched. This probe separates them from
//! outside the solver, using only the shipped pinned-`ℓ` path.
//!
//! An explicit `length_scale=` pins `ℓ` and short-circuits the ψ search
//! (`all_spatial_terms_kappa_fixed`), so fitting on a grid of `ℓ` gives the
//! criterion PROFILED over `ρ` at each `ℓ`. That profile is invariant to the
//! per-`ℓ` Frobenius normalization of the penalties (rescaling `S` by `c` is a
//! shift of `ρ` by `−ln c`, and `ρ` is optimized freely), so the values are
//! comparable across the sweep.
//!
//! Reading it:
//!   * a smooth profile with an interior optimum  -> the objective is fine and
//!     the refusal is the gradient or the solver;
//!   * a jagged profile at fine spacing           -> the objective itself is
//!     discontinuous in `ℓ` and no line search can work on it.
//!
//! Usage: `cargo run --release --example probe_2761_ell_profile -- [coarse|fine]`

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

// The `measure_jet_perf_parity` fixture, verbatim: a 1-D curve in 3-D.
const N_TRAIN: usize = 1_500;
const N_TEST: usize = 500;
const SIGMA: f64 = 0.10;
const TRAIN_SEED: u64 = 1_039;
const TEST_SEED: u64 = 2_039;

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

fn build_dataset(n: usize, sigma: f64, seed: u64) -> gam::data::EncodedDataset {
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

fn build_test_latents(n: usize, seed: u64) -> Vec<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let latent = Uniform::new(0.0, 1.0).expect("uniform");
    (0..n).map(|_| latent.sample(&mut rng)).collect()
}

fn held_out_rmse(
    fit: &gam::StandardFitResult,
    ds: &gam::data::EncodedDataset,
    test_latents: &[f64],
) -> f64 {
    let x0 = ds.column_map()["x0"];
    let x1 = ds.column_map()["x1"];
    let x2 = ds.column_map()["x2"];
    let mut grid = Array2::<f64>::zeros((test_latents.len(), ds.headers.len()));
    for (row, &t) in test_latents.iter().enumerate() {
        let coords = latent_to_coords(t);
        grid[[row, x0]] = coords[0];
        grid[[row, x1]] = coords[1];
        grid[[row, x2]] = coords[2];
    }
    let design =
        build_term_collection_design(grid.view(), &fit.resolvedspec).expect("rebuild design");
    let yhat: Vec<f64> = design.design.apply(&fit.fit.beta).to_vec();
    let truth_values: Vec<f64> = test_latents.iter().map(|&t| truth(t)).collect();
    rmse(&yhat, &truth_values)
}

fn main() {
    init_parallelism();
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(String::as_str).unwrap_or("coarse");

    let ds = build_dataset(N_TRAIN, SIGMA, TRAIN_SEED);
    let test_latents = build_test_latents(N_TEST, TEST_SEED);

    // The auto range this fixture resolves to is ~0.51 in the standardized
    // frame (#2761's own table), and the range the free search reported was
    // ~3.88, so the sweep has to cover both by a wide margin. `fine` walks a
    // narrow window in small multiplicative steps, which is where a
    // discontinuity would show up as a jump between adjacent points.
    let ells: Vec<f64> = match mode {
        "fine" => (0..61).map(|k| 2.0 * 1.02_f64.powi(k - 30)).collect(),
        _ => (0..25)
            .map(|k| 0.12 * 1.3_f64.powi(k as i32))
            .collect::<Vec<_>>(),
    };

    println!("[2761-profile] mode={mode} n={N_TRAIN} sigma={SIGMA}");
    println!(
        "[2761-profile] {:>10} {:>16} {:>10} {:>10} {:>28}",
        "ell", "reml", "edf", "rmse", "lambdas"
    );
    let mut best: Option<(f64, f64)> = None;
    for ell in ells {
        let body = format!("mjs(x0, x1, x2, centers=16, length_scale={ell})");
        let cfg = FitConfig {
            family: Some("gaussian".to_string()),
            ..FitConfig::default()
        };
        match fit_from_formula(&format!("y ~ {body}"), &ds, &cfg) {
            Ok(FitResult::Standard(fit)) => {
                let reml = fit.fit.reml_score().unwrap_or(f64::NAN);
                let edf = fit.fit.blocks[0].edf;
                let err = held_out_rmse(&fit, &ds, &test_latents);
                let lam: Vec<String> = fit.fit.blocks[0]
                    .lambdas
                    .iter()
                    .map(|l| format!("{l:.4e}"))
                    .collect();
                println!(
                    "[2761-profile] {ell:>10.5} {reml:>16.8} {edf:>10.4} {err:>10.6} {:>28}",
                    lam.join(",")
                );
                if reml.is_finite() && best.map(|(_, b)| reml < b).unwrap_or(true) {
                    best = Some((ell, reml));
                }
            }
            Ok(_) => println!("[2761-profile] {ell:>10.5}   non-standard variant"),
            Err(e) => println!(
                "[2761-profile] {ell:>10.5}   REFUSED: {}",
                e.to_string().chars().take(160).collect::<String>()
            ),
        }
    }
    if let Some((ell, reml)) = best {
        println!("[2761-profile] best profiled criterion: ell={ell:.5} reml={reml:.8}");
    }
}
