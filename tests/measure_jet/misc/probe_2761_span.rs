//! TEMPORARY probe for #2761 — not an acceptance gate.
//!
//! The gam-terms span probe established that the measure-jet design span at the
//! frozen auto range already reaches `0.0065` on this fixture's truth — 24x
//! below the fitted `0.1556`. So the deficit is downstream of the basis. This
//! probe splits what is left into its three possible homes, on the shipped path:
//!
//!   * **the fit** — in-sample fitted values against the noiseless truth, using
//!     the FIT-TIME design the coefficients were estimated on;
//!   * **the replay** — the same coefficients through the predict-time REBUILT
//!     design on the same rows; a mismatch is a freeze/replay defect, not a
//!     smoothing one;
//!   * **the smoothing** — the held-out span floor and the unpenalized
//!     least-squares fit on the realized design, which bracket what any lambda
//!     could have achieved.

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

struct Fixture {
    ds: gam::data::EncodedDataset,
    train_latents: Vec<f64>,
    train_y: Vec<f64>,
}

fn build_fixture() -> Fixture {
    let mut rng = StdRng::seed_from_u64(TRAIN_SEED);
    let latent = Uniform::new(0.0, 1.0).expect("uniform");
    let noise = Normal::new(0.0, SIGMA).expect("normal");
    let headers = ["x0", "x1", "x2", "y"]
        .iter()
        .cloned()
        .map(String::from)
        .collect();
    let mut train_latents = Vec::with_capacity(N_TRAIN);
    let mut train_y = Vec::with_capacity(N_TRAIN);
    let rows: Vec<StringRecord> = (0..N_TRAIN)
        .map(|_| {
            let t = latent.sample(&mut rng);
            let coords = latent_to_coords(t);
            let y = truth(t) + noise.sample(&mut rng);
            train_latents.push(t);
            train_y.push(y);
            StringRecord::from(vec![
                coords[0].to_string(),
                coords[1].to_string(),
                coords[2].to_string(),
                y.to_string(),
            ])
        })
        .collect();
    Fixture {
        ds: encode_recordswith_inferred_schema(headers, rows).expect("encode"),
        train_latents,
        train_y,
    }
}

fn build_test_latents(n: usize, seed: u64) -> Vec<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let latent = Uniform::new(0.0, 1.0).expect("uniform");
    (0..n).map(|_| latent.sample(&mut rng)).collect()
}

fn dense_of(op: &dyn LinearOperator) -> Array2<f64> {
    let (n, p) = (op.nrows(), op.ncols());
    let mut dense = Array2::<f64>::zeros((n, p));
    for j in 0..p {
        let mut e = Array1::<f64>::zeros(p);
        e[j] = 1.0;
        dense.column_mut(j).assign(&op.apply(&e));
    }
    dense
}

fn rebuilt_dense(
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
    dense_of(&built.design)
}

/// (residual rmse, retained rank, ls coefficient vector) of the least-squares
/// projection of `y` onto span(columns of `x`).
fn span_projection(x: &Array2<f64>, y: &[f64]) -> (f64, usize) {
    let n = x.nrows();
    let p = x.ncols();
    let mut basis: Vec<Array1<f64>> = Vec::new();
    for j in 0..p {
        let mut v = x.column(j).to_owned();
        let raw = v.dot(&v).sqrt();
        for _ in 0..2 {
            for q in basis.iter() {
                let c = q.dot(&v);
                v.scaled_add(-c, q);
            }
        }
        let nrm = v.dot(&v).sqrt();
        if nrm > 1.0e-10 * raw.max(1.0) {
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
    ((resid.dot(&resid) / n as f64).sqrt(), basis.len())
}

/// Predicted values of the unpenalized least-squares fit of `y_obs` on
/// `x_train`, evaluated through the SAME orthonormal basis rebuilt on
/// `x_eval` is not possible column-wise, so instead: solve normal equations
/// with a symmetric eigen pseudo-inverse and apply the coefficients.
fn least_squares_beta(x: &Array2<f64>, y: &[f64]) -> Array1<f64> {
    let p = x.ncols();
    let xtx = x.t().dot(x);
    let xty = x.t().dot(&Array1::from_vec(y.to_vec()));
    // Tiny relative ridge only to make the solve well posed; p is small.
    let scale = (0..p).map(|j| xtx[[j, j]]).fold(0.0_f64, f64::max);
    let mut a = xtx;
    for j in 0..p {
        a[[j, j]] += 1.0e-12 * scale;
    }
    // Gaussian elimination with partial pivoting.
    let mut aug = Array2::<f64>::zeros((p, p + 1));
    aug.slice_mut(ndarray::s![.., ..p]).assign(&a);
    aug.column_mut(p).assign(&xty);
    for col in 0..p {
        let mut piv = col;
        for r in col + 1..p {
            if aug[[r, col]].abs() > aug[[piv, col]].abs() {
                piv = r;
            }
        }
        if piv != col {
            for c in 0..=p {
                let tmp = aug[[col, c]];
                aug[[col, c]] = aug[[piv, c]];
                aug[[piv, c]] = tmp;
            }
        }
        let d = aug[[col, col]];
        if d.abs() < 1.0e-300 {
            continue;
        }
        for r in 0..p {
            if r == col {
                continue;
            }
            let f = aug[[r, col]] / d;
            for c in col..=p {
                aug[[r, c]] -= f * aug[[col, c]];
            }
        }
    }
    Array1::from_iter((0..p).map(|j| {
        let d = aug[[j, j]];
        if d.abs() < 1.0e-300 { 0.0 } else { aug[[j, p]] / d }
    }))
}

fn arm(body: &str, fx: &Fixture, test_latents: &[f64]) {
    let formula = format!("y ~ {body}");
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = match fit_from_formula(&formula, &fx.ds, &cfg) {
        Ok(r) => r,
        Err(e) => {
            println!("[probe-2761] {body}: FIT REFUSED: {e}");
            return;
        }
    };
    let FitResult::Standard(fit) = result else {
        println!("[probe-2761] {body}: non-standard fit");
        return;
    };
    let truth_train: Vec<f64> = fx.train_latents.iter().map(|&t| truth(t)).collect();
    let truth_test: Vec<f64> = test_latents.iter().map(|&t| truth(t)).collect();

    // FIT-TIME design, exactly as estimated on.
    let x_fit = dense_of(&fit.design.design);
    let fitted_in_sample: Vec<f64> = x_fit.dot(&fit.fit.beta).to_vec();
    let in_sample_vs_truth = rmse(&fitted_in_sample, &truth_train);

    // PREDICT-TIME rebuild on the SAME rows: any difference is replay, not fit.
    let x_replay = rebuilt_dense(&fit, &fx.ds, &fx.train_latents);
    let replay_in_sample: Vec<f64> = x_replay.dot(&fit.fit.beta).to_vec();
    let replay_gap = rmse(&replay_in_sample, &fitted_in_sample);

    // HELD-OUT rebuild.
    let x_test = rebuilt_dense(&fit, &fx.ds, test_latents);
    let held_out: Vec<f64> = x_test.dot(&fit.fit.beta).to_vec();
    let held_out_rmse = rmse(&held_out, &truth_test);

    // What ANY lambda could have reached, both ends.
    let (floor_fit, rank_fit) = span_projection(&x_fit, &truth_train);
    let (floor_test, rank_test) = span_projection(&x_test, &truth_test);
    let ls_beta = least_squares_beta(&x_fit, &fx.train_y);
    let ls_held_out: Vec<f64> = x_test.dot(&ls_beta).to_vec();
    let ls_rmse = rmse(&ls_held_out, &truth_test);

    let realized_ell = measure_jet_term_spec(&fit.resolvedspec, 0).map(|s| s.length_scale);
    println!(
        "[probe-2761] {body}\n    \
         p_fit={} p_replay={} rank_fit={rank_fit} rank_test={rank_test} edf={:.3} lambdas={:?}\n    \
         in_sample_vs_truth={in_sample_vs_truth:.6}  replay_gap={replay_gap:.3e}  \
         held_out={held_out_rmse:.6}\n    \
         span_floor_fit={floor_fit:.6}  span_floor_test={floor_test:.6}  \
         unpenalized_ls_held_out={ls_rmse:.6}  ell={realized_ell:?}",
        x_fit.ncols(),
        x_replay.ncols(),
        fit.fit.edf_total().unwrap_or(f64::NAN),
        fit.fit.lambdas.as_slice(),
    );
}

#[test]
fn probe_2761_where_the_deficit_lives() {
    init_parallelism();
    let fx = build_fixture();
    let test_latents = build_test_latents(N_TEST, TEST_SEED);
    for body in [
        "mjs(x0, x1, x2, centers=16)",
        "matern(x0, x1, x2, k=16)",
        "duchon(x0, x1, x2, k=16)",
        "mjs(x0, x1, x2, centers=16, learn_length_scale=true)",
        "mjs(x0, x1, x2, centers=16, double_penalty=false)",
    ] {
        arm(body, &fx, &test_latents);
    }
}
