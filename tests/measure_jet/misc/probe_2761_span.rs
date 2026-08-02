//! TEMPORARY probe for #2761 — not an acceptance gate.
//!
//! Separates "the design cannot represent the target" from "the penalty is
//! misallocated" by measuring, for each comparator basis, the least-squares
//! projection of the NOISELESS truth onto the column span of the realized
//! design on the held-out grid. That number is the floor no choice of
//! smoothing parameter can beat, and it carries no fitting noise.
//!
//! Second arm: the same measurement for measure-jet as a function of the
//! representer range ℓ, which measure-jet freezes at the median
//! nearest-center spacing while Matérn REML-selects its own (κ).

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::terms::smooth::measure_jet_term_spec;
use gam::test_support::reference::rmse;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::{Array1, Array2};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};

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
    let col_names = ["x0", "x1", "x2", "y"];
    let headers = col_names.iter().cloned().map(String::from).collect();
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

/// Dense materialization of the term-collection design on `latents`.
fn dense_design(
    fit: &gam::StandardFitResult,
    ds: &gam::data::EncodedDataset,
    latents: &[f64],
) -> Array2<f64> {
    let x0_idx = ds.column_map()["x0"];
    let x1_idx = ds.column_map()["x1"];
    let x2_idx = ds.column_map()["x2"];
    let mut grid = Array2::<f64>::zeros((latents.len(), ds.headers.len()));
    for (row, &t) in latents.iter().enumerate() {
        let coords = latent_to_coords(t);
        grid[[row, x0_idx]] = coords[0];
        grid[[row, x1_idx]] = coords[1];
        grid[[row, x2_idx]] = coords[2];
    }
    let built = build_term_collection_design(grid.view(), &fit.resolvedspec).expect("rebuild");
    let op = &built.design;
    let (n, p) = (op.nrows(), op.ncols());
    let mut dense = Array2::<f64>::zeros((n, p));
    for j in 0..p {
        let mut e = Array1::<f64>::zeros(p);
        e[j] = 1.0;
        dense.column_mut(j).assign(&op.apply(&e));
    }
    dense
}

/// RMSE of the least-squares projection residual of `y` onto span(columns of
/// `x`), by modified Gram–Schmidt with a relative rank floor.
fn span_projection_rmse(x: &Array2<f64>, y: &[f64]) -> (f64, usize) {
    let n = x.nrows();
    let p = x.ncols();
    let mut basis: Vec<Array1<f64>> = Vec::new();
    let scale = (0..p)
        .map(|j| x.column(j).dot(&x.column(j)).sqrt())
        .fold(0.0_f64, f64::max);
    let floor = 1.0e-10 * scale.max(1.0);
    for j in 0..p {
        let mut v = x.column(j).to_owned();
        for _pass in 0..2 {
            for q in basis.iter() {
                let c = q.dot(&v);
                v.scaled_add(-c, q);
            }
        }
        let nrm = v.dot(&v).sqrt();
        if nrm > floor {
            v.mapv_inplace(|z| z / nrm);
            basis.push(v);
        }
    }
    let yv = Array1::from_vec(y.to_vec());
    let mut resid = yv.clone();
    for q in basis.iter() {
        let c = q.dot(&yv);
        resid.scaled_add(-c, q);
    }
    let zero = vec![0.0; n];
    (rmse(resid.as_slice().expect("contig"), &zero), basis.len())
}

fn arm(body: &str, ds: &gam::data::EncodedDataset, test_latents: &[f64]) -> Option<f64> {
    let formula = format!("y ~ {body}");
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = match fit_from_formula(&formula, ds, &cfg) {
        Ok(r) => r,
        Err(e) => {
            println!("[span-2761] {body}: FIT REFUSED: {e}");
            return None;
        }
    };
    let FitResult::Standard(fit) = result else {
        println!("[span-2761] {body}: non-standard fit");
        return None;
    };
    let truth_test: Vec<f64> = test_latents.iter().map(|&t| truth(t)).collect();
    let x_test = dense_design(&fit, ds, test_latents);
    let yhat: Vec<f64> = x_test.dot(&fit.fit.beta).to_vec();
    let fitted_rmse = rmse(&yhat, &truth_test);
    let (span_test, rank_test) = span_projection_rmse(&x_test, &truth_test);
    let edf: f64 = fit.fit.edf;
    let realized_ell = measure_jet_term_spec(&fit.resolvedspec, 0).map(|s| s.length_scale);
    println!(
        "[span-2761] {body}: p={} rank={rank_test} edf={edf:.3} fitted_rmse={fitted_rmse:.6} \
         span_floor={span_test:.6} realized_ell={realized_ell:?}",
        x_test.ncols()
    );
    realized_ell
}

#[test]
fn probe_2761_span_floor_of_each_basis() {
    init_parallelism();
    let ds = build_dataset(N_TRAIN, SIGMA, TRAIN_SEED);
    let test_latents = build_test_latents(N_TEST, TEST_SEED);

    let auto_ell = arm("mjs(x0, x1, x2, centers=16)", &ds, &test_latents);
    arm("matern(x0, x1, x2, k=16)", &ds, &test_latents);
    arm("duchon(x0, x1, x2, k=16)", &ds, &test_latents);
    arm(
        "mjs(x0, x1, x2, centers=16, learn_length_scale=true)",
        &ds,
        &test_latents,
    );

    let base = auto_ell.expect("auto mjs fit resolves a length scale");
    println!("[span-2761] auto ell = {base:.6}; sweeping multiples");
    for factor in [0.25_f64, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0] {
        let ell = base * factor;
        arm(
            &format!("mjs(x0, x1, x2, centers=16, length_scale={ell})"),
            &ds,
            &test_latents,
        );
        println!("[span-2761]   (above was factor {factor})");
    }
}
