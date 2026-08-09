//! Regression gate for #1039: measure-jet single-scale mode must keep the same
//! outer footprint as the comparable kernel-representer method (Matern). With
//! 16 centers it uses one fused penalty (the nullspace ridge folded in, #1116)
//! and no psi dials, so this file checks speed and accuracy parity on a cheap
//! Gaussian low-dimensional-manifold problem.
//!
//! Comparator = MATERN, not Duchon (#1116). Both speed and accuracy are gated
//! against Matern — the same estimator CLASS (a finite kernel-representer basis
//! with a learned roughness penalty). Duchon is a different class: its penalty
//! is a CLOSED-FORM analytic polyharmonic operator (no empirical-measure
//! geometry — no centers/masses/band/per-cell affine projection), so it is both
//! exceptionally cheap and an exact interpolant. Demanding measure-jet's
//! empirical-geometry estimator stay within 2x Duchon's analytic penalty (or
//! 1.10x its exact-interpolant accuracy) is ill-posed by design.
//!
//! ## The 13.4x this file measured for a month was a FROZEN representer range (#2761)
//!
//! #2697 read the miss as a regression and #2761 measured it at both ends of the
//! suspected interval, each basis fitted and scored INDEPENDENTLY so one basis
//! refusing cannot abort the arm and suppress the others:
//!
//! ```text
//!                    mjs         matern      duchon
//!   7b0776e15^      0.155576     REFUSED    0.010521
//!   665dce521       0.155584    0.011639    0.010521
//! ```
//!
//! mjs agrees across the interval to 5.1e-5 relative and duchon to five
//! significant figures, so nothing about measure-jet's accuracy CHANGED there and
//! `7b0776e15` is exonerated. The deficit was older than the interval, and it was
//! not a basis-capacity limit either. Measured through `fit_from_formula` on this
//! fixture, with `span floor` = the least-squares projection residual of the
//! NOISELESS truth onto the realized design's column span -- the bound no `lambda`
//! can beat, since `lambda` shrinks inside a span and never moves one:
//!
//! ```text
//!   arm                       ell      edf   span floor  unpen. LS  held-out
//!   frozen (auto ell)      0.5144   14.684    0.152488   0.155484   0.155584
//!   REML-selected ell      3.8813   14.006    0.000014   0.008155   0.009642
//!   matern(k=16)                -   14.619    0.006077   0.011989   0.011639
//!   duchon(k=16)                -   15.016    0.002443   0.011308   0.010521
//! ```
//!
//! At the frozen range the fitted `0.1556` IS the span floor: unpenalized least
//! squares on the same design gives `0.1555`, dropping the null-component penalty
//! moves the fourth decimal, and `edf/p = 0.98` says the fit is already spending
//! everything it has. `mjs`'s representer range had been frozen at the median
//! nearest-center spacing since `b1d94d1a5` turned its REML coordinate off by
//! default; restoring it (#2761) puts mjs at `0.0096`, which beats matern by 1.21x
//! and duchon by 1.09x at LOWER edf. So the match-or-beat-matern bar below is
//! honest again, and it must still not be closed by widening the `1.10x` bound.
//!
//! The SPEED half is what pays for that: a design-moving outer coordinate rebuilds
//! the representer design per outer trial. The `2.0x`-vs-matern bound is where that
//! cost is measured (matern carries the same kind of coordinate in its `kappa`), and
//! it still guards against the prior 12x regression returning.

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
use std::time::Instant;

const N_TRAIN: usize = 1_500;
const N_TEST: usize = 500;
const SIGMA: f64 = 0.10;
const TRAIN_SEED: u64 = 1_039;
const TEST_SEED: u64 = 2_039;
const MJS_BODY: &str = "mjs(x0, x1, x2, centers=16)";
const MATERN_BODY: &str = "matern(x0, x1, x2, k=16)";
const DUCHON_BODY: &str = "duchon(x0, x1, x2, k=16)";

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

fn fit_and_time(
    formula_body: &str,
    ds: &gam::data::EncodedDataset,
) -> (std::time::Duration, gam::StandardFitResult) {
    let formula = format!("y ~ {formula_body}");
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let start = Instant::now();
    let result =
        fit_from_formula(&formula, ds, &cfg).unwrap_or_else(|e| panic!("gam fit '{formula}': {e}"));
    let elapsed = start.elapsed();
    let FitResult::Standard(fit) = result else {
        panic!("expected standard fit for '{formula}'");
    };
    (elapsed, fit)
}

fn held_out_rmse(
    fit: &gam::StandardFitResult,
    ds: &gam::data::EncodedDataset,
    formula_for_msg: &str,
    test_latents: &[f64],
) -> f64 {
    let x0_idx = ds.column_map()["x0"];
    let x1_idx = ds.column_map()["x1"];
    let x2_idx = ds.column_map()["x2"];
    let mut grid = Array2::<f64>::zeros((test_latents.len(), ds.headers.len()));
    for (row, &t) in test_latents.iter().enumerate() {
        let coords = latent_to_coords(t);
        grid[[row, x0_idx]] = coords[0];
        grid[[row, x1_idx]] = coords[1];
        grid[[row, x2_idx]] = coords[2];
    }
    let design = build_term_collection_design(grid.view(), &fit.resolvedspec)
        .unwrap_or_else(|e| panic!("rebuild '{formula_for_msg}': {e}"));
    let yhat: Vec<f64> = design.design.apply(&fit.fit.beta).to_vec();
    let truth_values: Vec<f64> = test_latents.iter().map(|&t| truth(t)).collect();
    rmse(&yhat, &truth_values)
}

/// Repetition count for the interleaved timing below.
///
/// This is a repetition count, not a threshold: it does not appear in the bar
/// and its only effect is how tight the minimum estimator is. Three is the
/// smallest count for which a single preempted sample cannot be the minimum of
/// its arm.
const TIMING_REPLICATES: usize = 3;

/// The outer work one arm spent, as the ENGINE counts it rather than as a clock
/// measures it: outer evaluations and design realizations.
///
/// These are printed, never asserted. They are exact integers and immune to host
/// load, but they are NOT comparable between the two arms: matern reaches the
/// joint solve with its range basin already selected and certified by a separate
/// scalar endpoint comparison (`spatial_optimization.rs`: *"The scalar Matern
/// endpoint comparison has already selected and certified the range basin. Give
/// its explicit theta0 the only joint start"*), so that search is inside the wall
/// clock and outside `kappa_timing`. A counted ratio would credit matern for work
/// it does off-counter. They are here because they are what a future regression
/// of this gate will need read first (gam#2750).
fn outer_work(fit: &gam::StandardFitResult) -> (usize, u64) {
    fit.kappa_timing
        .as_ref()
        .map_or((0, 0), |t| (t.eval_calls, t.design_revision_delta))
}

#[test]
fn measure_jet_single_scale_mode_is_speed_competitive() {
    init_parallelism();
    let ds = build_dataset(N_TRAIN, SIGMA, TRAIN_SEED);

    // The bar below is a statement about the ESTIMATORS, and a wall clock only
    // measures the estimators when the process has the cores to itself. This
    // target holds 20+ tests and the harness runs them concurrently by default,
    // so one sample measures the host's load as much as the fit. Measured on one
    // binary, one fixture, one seed, with only the harness concurrency changing
    // (gam#2750):
    //
    //     19 tests in parallel   mjs=3.367s  matern=1.513s   ratio 2.226   FAIL
    //     --test-threads=1       mjs=0.987s  matern=0.531s   ratio 1.859   pass
    //
    // The verdict flipped on a setting that has nothing to do with either
    // estimator, and the uncontended margin was 7%. So the ESTIMATOR is what
    // changes here, never the bar: repetitions are INTERLEAVED, so any drift in
    // host load lands on both arms in the same window, and each arm is summarized
    // by its MINIMUM — preemption can only ADD time, so the smallest of k samples
    // is the least contaminated one. Nothing about `2.0` moves; #2761's argument
    // for not widening the sibling accuracy bar applies here verbatim, and this
    // bound's own reason for existing ("guards the prior 12x regression") is
    // untouched by measuring it more carefully.
    let mut mjs_secs = f64::INFINITY;
    let mut matern_secs = f64::INFINITY;
    let mut duchon_secs = f64::INFINITY;
    let mut mjs_work = (0usize, 0u64);
    let mut matern_work = (0usize, 0u64);
    for replicate in 0..TIMING_REPLICATES {
        let (matern_elapsed, matern_fit) = fit_and_time(MATERN_BODY, &ds);
        matern_work = outer_work(&matern_fit);
        drop(matern_fit);
        let (duchon_elapsed, duchon_fit) = fit_and_time(DUCHON_BODY, &ds);
        drop(duchon_fit);
        let (mjs_elapsed, mjs_fit) = fit_and_time(MJS_BODY, &ds);
        mjs_work = outer_work(&mjs_fit);
        drop(mjs_fit);
        let (m, k, d) = (
            mjs_elapsed.as_secs_f64(),
            matern_elapsed.as_secs_f64(),
            duchon_elapsed.as_secs_f64(),
        );
        println!("[mjs-perf] replicate {replicate}: mjs={m:.3}s matern={k:.3}s duchon={d:.3}s");
        mjs_secs = mjs_secs.min(m);
        matern_secs = matern_secs.min(k);
        duchon_secs = duchon_secs.min(d);
    }
    println!(
        "[mjs-perf] min over {TIMING_REPLICATES}: mjs={mjs_secs:.3}s matern={matern_secs:.3}s \
         duchon={duchon_secs:.3}s  ratio={:.3}",
        mjs_secs / matern_secs
    );
    println!(
        "[mjs-perf] outer work (eval_calls, design_realizations): mjs={mjs_work:?} \
         matern={matern_work:?}"
    );
    // Speed parity is gated against MATERN, the comparable kernel-representer
    // method (#1116). Duchon's penalty is closed-form analytic (no
    // empirical-measure geometry), a different/cheaper class — measure-jet is
    // ~4x faster than matern but cannot match duchon's analytic-penalty cost,
    // just as it cannot match duchon's exact-interpolant accuracy. The 2.0x
    // bound guards the prior 12x regression; duchon's time is printed for
    // reference only.
    assert!(
        mjs_secs <= 2.0 * matern_secs,
        "measure-jet single-scale mode speed parity failed vs matern: mjs={mjs_secs:.3}s \
         matern={matern_secs:.3}s duchon={duchon_secs:.3}s (minima over {TIMING_REPLICATES} \
         interleaved replicates; outer work mjs={mjs_work:?} matern={matern_work:?})"
    );
}

#[test]
fn measure_jet_single_scale_mode_accuracy_parity() {
    init_parallelism();
    let ds = build_dataset(N_TRAIN, SIGMA, TRAIN_SEED);
    let test_latents = build_test_latents(N_TEST, TEST_SEED);

    let mjs_fit = fit_and_time(MJS_BODY, &ds).1;
    let matern_fit = fit_and_time(MATERN_BODY, &ds).1;
    let duchon_fit = fit_and_time(DUCHON_BODY, &ds).1;

    let mjs_formula = format!("y ~ {MJS_BODY}");
    let matern_formula = format!("y ~ {MATERN_BODY}");
    let duchon_formula = format!("y ~ {DUCHON_BODY}");
    let mjs_rmse = held_out_rmse(&mjs_fit, &ds, &mjs_formula, &test_latents);
    let matern_rmse = held_out_rmse(&matern_fit, &ds, &matern_formula, &test_latents);
    let duchon_rmse = held_out_rmse(&duchon_fit, &ds, &duchon_formula, &test_latents);
    println!("[mjs-accuracy] mjs={mjs_rmse:.5} matern={matern_rmse:.5} duchon={duchon_rmse:.5}");

    // Match-or-beat MATERN, the comparable kernel-representer method (#1116);
    // duchon's closed-form exact-interpolant accuracy is a different class and
    // its RMSE is printed for reference only.
    assert!(
        mjs_rmse <= 1.10 * matern_rmse,
        "measure-jet single-scale mode accuracy parity failed vs matern: mjs={mjs_rmse:.5} \
         matern={matern_rmse:.5} duchon={duchon_rmse:.5}"
    );
}

/// Dense materialization of a design operator, column by column.
fn dense_of(op: &dyn LinearOperator) -> Array2<f64> {
    let (n, p) = (op.nrows(), op.ncols());
    let mut dense = Array2::<f64>::zeros((n, p));
    for j in 0..p {
        let mut e = ndarray::Array1::<f64>::zeros(p);
        e[j] = 1.0;
        dense.column_mut(j).assign(&op.apply(&e));
    }
    dense
}

/// #2761 control, from the direction the accuracy gates cannot see.
///
/// Every accuracy number in the measure-jet cluster — this file's, the sweep's,
/// the BMS logslope correlation — is scored by applying the fitted coefficients
/// to a design **rebuilt from the frozen spec** on a fresh grid, not to the
/// design they were estimated on. So a freeze/replay defect and a smoothing
/// defect are indistinguishable from any of those numbers alone: both show up
/// only as a held-out miss. Diagnosing #2761 required ruling the replay out
/// first, by hand. This makes that a standing check instead.
///
/// Measure-jet has by far the largest replay surface of the kernel bases — the
/// realized quadrature (cell barycenters + masses), the scale band, the support
/// anchors, the penalty normalization scales, the composed identifiability
/// transform, the standardized representer range and the ambient-affine head
/// lift all have to come back identically from the frozen spec, and the head
/// lift is *recomputed* at predict time from the frozen centers and masses
/// rather than persisted. The tolerance here is machine-precision, not
/// statistical: the contract is that the rebuild reproduces the fit-time design,
/// not that it approximates it.
#[test]
fn measure_jet_predict_rebuild_replays_the_fit_time_design_2761() {
    init_parallelism();
    let ds = build_dataset(N_TRAIN, SIGMA, TRAIN_SEED);
    let fit = fit_and_time(MJS_BODY, &ds).1;

    let fit_time = dense_of(&fit.design.design);
    let x0 = ds.column_map()["x0"];
    let x1 = ds.column_map()["x1"];
    let x2 = ds.column_map()["x2"];
    let mut rows = Array2::<f64>::zeros((ds.values.nrows(), ds.headers.len()));
    for c in [x0, x1, x2] {
        rows.column_mut(c).assign(&ds.values.column(c));
    }
    let rebuilt = dense_of(
        &build_term_collection_design(rows.view(), &fit.resolvedspec)
            .expect("predict-time rebuild of the frozen measure-jet spec")
            .design,
    );

    assert_eq!(
        fit_time.dim(),
        rebuilt.dim(),
        "the predict-time rebuild must have the fit-time shape"
    );
    let scale = fit_time.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
    let worst = fit_time
        .iter()
        .zip(rebuilt.iter())
        .fold(0.0_f64, |a, (l, r)| a.max((l - r).abs()));
    assert!(
        worst <= 1.0e-11 * scale.max(1.0),
        "the predict-time measure-jet design must REPLAY the fit-time one, not \
         approximate it: worst column entry differs by {worst:.3e} against a design \
         scale of {scale:.3e}. Every accuracy bar in this cluster applies fitted \
         coefficients to this rebuild, so a drift here is scored as a smoothing miss \
         (#2761)"
    );

    // The same statement where it is actually consumed: the linear predictor.
    let eta_fit = fit_time.dot(&fit.fit.beta);
    let eta_replay = rebuilt.dot(&fit.fit.beta);
    let eta_scale = eta_fit.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
    let eta_worst = eta_fit
        .iter()
        .zip(eta_replay.iter())
        .fold(0.0_f64, |a, (l, r)| a.max((l - r).abs()));
    assert!(
        eta_worst <= 1.0e-11 * eta_scale.max(1.0),
        "fitted values through the rebuilt design must match the fit-time ones: \
         worst row differs by {eta_worst:.3e} against |eta|_inf = {eta_scale:.3e}"
    );
}
