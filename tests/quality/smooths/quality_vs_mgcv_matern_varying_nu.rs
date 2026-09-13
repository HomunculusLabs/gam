//! End-to-end **objective quality** of gam's Gaussian-process **Matérn family**
//! across the smoothness ladder ν ∈ {1.5, 2.5, 3.5}.
//!
//! The data are generated from a KNOWN smooth function
//!     f(x) = 0.5 + sin(3πx)·exp(-x²/2),   y = f(x) + N(0, 0.08²),
//! so the objective quality of any fit is how well it RECOVERS f — not whether it
//! reproduces some other tool's fitted curve. The primary assertion is therefore
//! **truth recovery**: for every supported Matérn order the gam fit must satisfy,
//! on the draw average,
//!     RMSE(gam_fit, f) ≤ 0.045  (well under the noise σ = 0.08 — a GP that has
//! learned the signal must beat the per-observation noise on the smooth grid),
//! and the fit must be highly correlated with the truth (Pearson > 0.99).
//!
//! mgcv's `bs="gp"` is kept ONLY as a *baseline to match-or-beat on accuracy*:
//! per `?gp.smooth` the first entry of `m` selects the Matérn correlation order
//! (m=3⇔ν=3/2, m=4⇔ν=5/2, m=5⇔ν=7/2), so we fit the SAME Matérn order on the
//! SAME data and require gam's truth-recovery error to be no worse than mgcv's by
//! more than 10% on the draw average, and not RESOLVED worse draw by draw
//! (`assert_paired_match_or_beat`). We never assert that gam reproduces mgcv's
//! curve, and we do not match EDF: matching a peer tool's noisy fit or its
//! complexity proves nothing about correctness.
//!
//! PAIRED over draws, not one draw (#2395). Either engine's RMSE against the
//! truth depends on which sample of x and noise it fit, so one draw's ratio
//! conflates the draw with the engine. The single draw this test used read gam
//! 0.0189 against mgcv 0.0155 at ν=2.5 (MSI job 578202 at 90c86056d),
//! and the case panicked there before ν=3.5 ran. Both engines now fit the SAME
//! `K_SEEDS` draws at every order, and every order's pair is emitted before any
//! assertion runs. The first draw is the seed the single-draw version used.
//!
//! A cross-order invariant additionally rules out the failure mode where `nu` is
//! silently ignored (every order collapsing onto one kernel). The Matérn-ν RKHS is
//! H^m with m = ν + d/2, and the term penalizes every derivative energy it controls
//! (`matern_for_smoothness`, plus the third-order block from ν ≥ 5/2), so the
//! smoothest order (ν=7/2) must carry strictly more penalty blocks than the
//! roughest (ν=3/2). A dropped `nu` would give both orders one topology. The mean
//! grid difference between those two fits is printed as context only: on two good
//! recoveries of the same smooth truth it is bounded by their own truth-recovery
//! errors, and job 602208 read 0.0069 there while the orders carried 3 and 4
//! blocks and different EDFs, so no fixed fraction measures "ν is not ignored".

use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{
    Column, PairedFoldComparison, QualityPair, assert_paired_match_or_beat, pearson, relative_l2,
    rmse, run_r,
};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};

/// Rows per draw.
const N_ROWS: usize = 160;
/// Observation noise SD.
const NOISE_SIGMA: f64 = 0.08;
/// Evaluation grid points.
const N_GRID: usize = 200;
/// Paired draws per Matérn order. See the "PAIRED over draws" note above.
const K_SEEDS: usize = 25;
/// Seed of the first draw: the single draw the unpaired version of this test used.
const FIRST_SEED: u64 = 20260529;

/// The KNOWN truth every fit is judged against.
fn truth(t: f64) -> f64 {
    0.5 + (3.0 * std::f64::consts::PI * t).sin() * (-t * t / 2.0).exp()
}

/// One draw, fed IDENTICALLY to both engines: x ~ U[0,1] (n=160), sorted, then
/// y = f(x) + N(0, 0.08²), both from the one stream seeded by `seed`.
fn matern_draw(seed: u64) -> (Vec<f64>, Vec<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let ux = Uniform::new(0.0, 1.0).expect("uniform [0,1]");
    let noise = Normal::new(0.0, NOISE_SIGMA).expect("gaussian noise");
    let mut x: Vec<f64> = std::iter::repeat_with(|| ux.sample(&mut rng))
        .take(N_ROWS)
        .collect();
    x.sort_by(|a, b| a.partial_cmp(b).expect("finite x"));
    let y: Vec<f64> = x
        .iter()
        .map(|&t| truth(t) + noise.sample(&mut rng))
        .collect();
    (x, y)
}

/// gam's `y ~ matern(x, nu=<ν>, k=18)` Gaussian REML fit on one draw: the fitted
/// function on `x_grid` (identity link, so design·beta is the mean), the fit's
/// total EDF and its number of penalty blocks. Prints the fit's selected
/// smoothing state.
fn gam_matern_on_grid(
    seed: u64,
    nu: f64,
    x: &[f64],
    y: &[f64],
    x_grid: &[f64],
) -> (Vec<f64>, f64, usize) {
    let headers = ["x", "y"].into_iter().map(String::from).collect();
    let rows: Vec<csv::StringRecord> = x
        .iter()
        .zip(y.iter())
        .map(|(a, b)| csv::StringRecord::from(vec![a.to_string(), b.to_string()]))
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode matern dataset");

    // ---- gam fit: y ~ matern(x, nu=<ν>, k=18), REML -----------------------
    let formula = format!("y ~ matern(x, nu={nu}, k=18)");
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula(&formula, &ds, &cfg)
        .unwrap_or_else(|e| panic!("gam matern fit (nu={nu}, seed={seed}) failed: {e:?}"));
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard Gaussian GAM fit for matern(nu={nu})");
    };
    let gam_edf = fit.fit.edf_total().expect("gam reports total edf");

    // gam fitted function on the grid (identity link ⇒ design·beta = mean).
    let mut g = Array2::<f64>::zeros((x_grid.len(), 2));
    for (i, &t) in x_grid.iter().enumerate() {
        g[[i, 0]] = t;
        g[[i, 1]] = 0.0;
    }
    let grid_design = build_term_collection_design(g.view(), &fit.resolvedspec)
        .expect("rebuild matern design at grid points");
    let gam_grid: Vec<f64> = grid_design.design.apply(&fit.fit.beta).to_vec();

    // #1561 λ-selection diagnostic (pure instrumentation; no pass criterion):
    // gam's REML argmin (log_lambdas + per-block EDF) on this draw, beside the
    // per-order mgcv line below.
    eprintln!(
        "lambda_diag test=matern_varying_nu nu={nu} seed={seed} \
         gam_reml={:.4} gam_edf_total={gam_edf:.4} \
         gam_lambdas={:?} gam_log_lambdas={:?} \
         gam_edf_by_block={:?} gam_block_trace={:?}",
        fit.fit.reml_score().expect("the fit reports a REML/LAML criterion"),
        fit.fit.lambdas.to_vec(),
        fit.fit.log_lambdas.to_vec(),
        fit.fit.edf_by_block().to_vec(),
        fit.fit.penalty_block_trace().to_vec(),
    );
    (gam_grid, gam_edf, fit.fit.lambdas.len())
}

#[test]
fn gam_matern_family_recovers_truth_across_nu() {
    init_parallelism();

    // shared dense interior evaluation grid (avoid GP-kernel edge dominance).
    let x_grid: Vec<f64> = (0..N_GRID)
        .map(|i| 0.005 + 0.99 * i as f64 / (N_GRID - 1) as f64)
        .collect();

    // KNOWN truth on the grid — the objective target every fit is judged against.
    let truth_grid: Vec<f64> = x_grid.iter().map(|&t| truth(t)).collect();
    // Signal range, used to sanity-bound the recovery error in absolute terms.
    let signal_range = {
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for &v in &truth_grid {
            lo = lo.min(v);
            hi = hi.max(v);
        }
        hi - lo
    };

    // ---- the draws, built once and reused for all three orders ------------
    let draws: Vec<(Vec<f64>, Vec<f64>)> = (0..K_SEEDS)
        .map(|k| matern_draw(FIRST_SEED + k as u64))
        .collect();
    let mut long_seed = Vec::with_capacity(K_SEEDS * N_ROWS);
    let mut long_x = Vec::with_capacity(K_SEEDS * N_ROWS);
    let mut long_y = Vec::with_capacity(K_SEEDS * N_ROWS);
    for (k, (x, y)) in draws.iter().enumerate() {
        long_seed.resize(long_seed.len() + N_ROWS, (FIRST_SEED + k as u64) as f64);
        long_x.extend_from_slice(x);
        long_y.extend_from_slice(y);
    }

    // ν ladder and the mgcv `m` (first entry) that selects the SAME Matérn
    // order: per `?gp.smooth`, m=3⇔ν=3/2, m=4⇔ν=5/2, m=5⇔ν=7/2.
    let orders: [(f64, i32); 3] = [(1.5, 3), (2.5, 4), (3.5, 5)];

    // Per-order panels (with gam's mean Pearson to the truth) and per-draw gam
    // grids, for the assertions and the cross-order kernel-distinctness check.
    let mut panels: Vec<(f64, PairedFoldComparison, f64)> = Vec::with_capacity(orders.len());
    let mut gam_grids_by_nu: Vec<Vec<Vec<f64>>> = Vec::with_capacity(orders.len());
    let mut penalty_blocks_by_nu: Vec<usize> = Vec::with_capacity(orders.len());

    for (nu, kappa) in orders {
        let mut gam_rmses = Vec::with_capacity(K_SEEDS);
        let mut gam_corr_total = 0.0;
        let mut gam_edf_total = 0.0;
        let mut gam_grids = Vec::with_capacity(K_SEEDS);
        let mut order_penalty_blocks: Option<usize> = None;
        for (k, (x, y)) in draws.iter().enumerate() {
            let (gam_grid, gam_edf, blocks) =
                gam_matern_on_grid(FIRST_SEED + k as u64, nu, x, y, &x_grid);
            if let Some(first) = order_penalty_blocks {
                assert_eq!(
                    blocks,
                    first,
                    "matern nu={nu}: the penalty topology must not depend on the draw (seed={})",
                    FIRST_SEED + k as u64
                );
            }
            order_penalty_blocks = Some(blocks);
            gam_rmses.push(rmse(&gam_grid, &truth_grid));
            gam_corr_total += pearson(&gam_grid, &truth_grid);
            gam_edf_total += gam_edf;
            gam_grids.push(gam_grid);
        }

        // ---- mgcv fit: same draws, s(x, bs="gp", k=18, m=κ), REML ---------
        // Kept ONLY as an accuracy baseline (match-or-beat on truth recovery),
        // never as the pass criterion. The scalar `m=κ` selects the Matérn
        // correlation order; the length-scale is left at mgcv's data-driven
        // default, matching gam's internal length-scale selection. All draws
        // run in ONE R session per order.
        let r = run_r(
            &[
                Column::new("seed", &long_seed),
                Column::new("x", &long_x),
                Column::new("y", &long_y),
                Column::new("xg", &x_grid),
                Column::new("kappa", &[f64::from(kappa)]),
            ],
            r#"
            suppressPackageStartupMessages(library(mgcv))
            kap <- as.integer(round(df$kappa[1]))
            grid_df <- data.frame(x = df$xg[is.finite(df$xg)])
            grid_all <- c()
            edf_all <- c()
            sp_all <- c()
            reml_all <- c()
            for (s in sort(unique(df$seed))) {
                keep <- which(df$seed == s)
                fit_df <- data.frame(x = df$x[keep], y = df$y[keep])
                m <- gam(y ~ s(x, bs = "gp", k = 18, m = kap),
                         data = fit_df, method = "REML")
                grid_all <- c(grid_all, as.numeric(predict(m, newdata = grid_df)))
                edf_all <- c(edf_all, sum(m$edf))
                sp_all <- c(sp_all, as.numeric(m$sp))
                reml_all <- c(reml_all, as.numeric(m$gcv.ubre))
            }
            emit("grid_fit", grid_all)
            emit("edf", edf_all)
            emit("sp", sp_all)
            emit("reml", reml_all)
            "#,
        );
        let mgcv_grid_flat = r.vector("grid_fit");
        let mgcv_edfs = r.vector("edf");
        assert_eq!(
            mgcv_grid_flat.len(),
            K_SEEDS * N_GRID,
            "mgcv grid prediction panel length mismatch (nu={nu})"
        );
        assert_eq!(
            mgcv_edfs.len(),
            K_SEEDS,
            "mgcv edf panel length mismatch (nu={nu})"
        );

        // ---- OBJECTIVE metric per draw: truth recovery on the grid ---------
        let mgcv_rmses: Vec<f64> = (0..K_SEEDS)
            .map(|k| rmse(&mgcv_grid_flat[k * N_GRID..(k + 1) * N_GRID], &truth_grid))
            .collect();
        // Context only (NOT a pass criterion): how close the two fitted curves are.
        let rel_to_mgcv = (0..K_SEEDS)
            .map(|k| relative_l2(&gam_grids[k], &mgcv_grid_flat[k * N_GRID..(k + 1) * N_GRID]))
            .sum::<f64>()
            / K_SEEDS as f64;
        let panel = PairedFoldComparison::new(&gam_rmses, &mgcv_rmses, true);
        let gam_corr = gam_corr_total / K_SEEDS as f64;

        eprintln!(
            "matern nu={nu} (mgcv m={kappa}) K={K_SEEDS}-draw paired: \
             fold-mean gam_rmse_vs_truth={:.4} mgcv_rmse_vs_truth={:.4} \
             mean gam_pearson_vs_truth={gam_corr:.5} mean gam_edf={:.3} \
             mean mgcv_edf={:.3} mean rel_l2(gam,mgcv)={rel_to_mgcv:.4} \
             noise_sigma={NOISE_SIGMA} signal_range={signal_range:.3}",
            panel.gam_mean,
            panel.reference_mean,
            gam_edf_total / K_SEEDS as f64,
            mgcv_edfs.iter().sum::<f64>() / K_SEEDS as f64,
        );
        // #1561 λ-selection diagnostic: mgcv's selected smoothing state per draw,
        // in draw order, beside gam's per-draw lambda_diag lines.
        eprintln!(
            "lambda_diag test=matern_varying_nu nu={nu} mgcv_m={kappa} \
             mgcv_sp={:?} mgcv_reml={:?} mgcv_edf={:?}",
            r.vector("sp"),
            r.vector("reml"),
            mgcv_edfs,
        );
        eprintln!("{}", panel.report(&format!("matern_varying_nu::nu{nu}")));
        // One metric per Matérn order: the aggregator keys an observation by
        // (case, metric, reference), so every ν in this loop needs its own
        // label or the second order's value conflicts with the first's.
        eprintln!(
            "{}",
            QualityPair::paired(
                "smooths",
                "quality_vs_mgcv_matern_varying_nu",
                format!("rmse_vs_truth_nu{nu}"),
                "mgcv",
                &panel,
            )
            .line()
        );
        panels.push((nu, panel, gam_corr));
        gam_grids_by_nu.push(gam_grids);
        penalty_blocks_by_nu.push(order_penalty_blocks.expect("every order fits K_SEEDS > 0 draws"));
    }

    // ---- kernel-distinctness invariant: `nu` must change the kernel order ---
    // Rules out the failure mode where gam silently collapses every `nu` onto ONE
    // effective kernel (e.g. a hard-wired ν=5/2). ν sets m = ν + d/2, and the term
    // penalizes every derivative energy of order j ≤ m (third order from ν ≥ 5/2),
    // so the smoothest order must carry strictly more penalty blocks than the
    // roughest. The grid difference between the two orders' fits is context: it is
    // bounded by the two fits' own truth-recovery errors, not by the kernel order.
    let nu_lo = orders[0].0; // ν = 3/2 (roughest)
    let nu_hi = orders[orders.len() - 1].0; // ν = 7/2 (smoothest)
    let grids_lo = &gam_grids_by_nu[0];
    let grids_hi = &gam_grids_by_nu[gam_grids_by_nu.len() - 1];
    let cross_order_rel = (0..K_SEEDS)
        .map(|k| relative_l2(&grids_lo[k], &grids_hi[k]))
        .sum::<f64>()
        / K_SEEDS as f64;
    let blocks_lo = penalty_blocks_by_nu[0];
    let blocks_hi = penalty_blocks_by_nu[penalty_blocks_by_nu.len() - 1];
    eprintln!(
        "kernel-distinctness: penalty blocks nu={nu_lo}: {blocks_lo}, nu={nu_hi}: {blocks_hi}; \
         mean rel_l2(nu={nu_lo}, nu={nu_hi}) = {cross_order_rel:.4} (context)"
    );

    for (nu, panel, gam_corr) in &panels {
        // PRIMARY claim: gam RECOVERS the known truth at every Matérn order.
        //  - RMSE ≤ 0.045: comfortably below the per-observation noise σ=0.08; a
        //    GP that has actually learned f must average the noise away on the
        //    smooth grid. This is ~7% of the signal range, an objective accuracy
        //    bar that no overfit-to-noise or wrong-kernel fit can clear.
        //  - Pearson > 0.99 vs TRUTH: the recovered shape must be the real signal.
        assert!(
            panel.gam_mean <= 0.045,
            "matern nu={nu}: gam fails to recover the known truth: \
             fold-mean RMSE_vs_truth={:.4} > 0.045 (noise σ={NOISE_SIGMA})",
            panel.gam_mean
        );
        assert!(
            *gam_corr > 0.99,
            "matern nu={nu}: gam recovered shape does not track the true signal: \
             mean pearson_vs_truth={gam_corr:.5}"
        );

        // BASELINE: match-or-beat mgcv on ACCURACY (its truth-recovery error),
        // paired across the K shared draws. This demotes mgcv from "we must
        // reproduce it" to "we must be at least as accurate as the mature tool
        // at recovering f": gam may not exceed mgcv's averaged error by more
        // than 10%, nor be resolved worse draw by draw.
        assert_paired_match_or_beat(&format!("matern nu={nu}"), panel, 1.10);
    }

    assert!(
        blocks_lo < blocks_hi,
        "gam Matérn fits for nu={nu_lo} and nu={nu_hi} carry {blocks_lo} and {blocks_hi} \
         penalty blocks; a smoother order must penalize more derivative energies, so `nu` \
         is not driving the kernel order (mean rel_l2 of the two fits {cross_order_rel:.4})"
    );
}
