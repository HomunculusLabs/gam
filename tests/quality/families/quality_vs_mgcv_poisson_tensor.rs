//! End-to-end **objective quality**: gam's **Poisson(log) × tensor-product**
//! smooth must *recover the true mean surface* it was generated from.
//!
//! OBJECTIVE METRIC ASSERTED (truth recovery, not tool-matching): the data is
//! simulated from a fully known log-mean surface
//!     eta_true(x, z) = 0.8 + 0.3*sin(x) + 0.2*z^2,  mu_true = exp(eta_true),
//! with noise entering *only* through the Poisson response draw. The smoother's
//! job is to estimate `mu_true` from the noisy counts. So the primary pass/fail
//! is the root-mean-square error of gam's fitted Poisson mean against the TRUE
//! mean surface, averaged over the draws below:
//!     mean over draws of RMSE(gam_mean, mu_true) <= 0.18 * range(mu_true).
//! On this 15×20 grid mu_true spans [1.662, 3.642] (range 1.98, MSI job 578202),
//! so the bar is an absolute RMSE of ~0.36 — comfortably below the per-cell
//! Poisson noise sd (sqrt(mu) ~ 1.29..1.91) yet tight enough that a broken PIRLS-row / tensor-
//! design integration (which would smear or bias the surface) fails it.
//!
//! This benchmarks the *critical cross-feature combination* that family-
//! agnostic basis code most often gets wrong: a non-Gaussian family (Poisson
//! with the canonical log link) layered on top of a multi-dimensional
//! tensor-product `te()` smooth. A 1-D Gaussian smooth can pass while the
//! tensor + IRLS-row interaction is subtly broken, so we test them together;
//! gam fits `y ~ te(x, z, k=[6,6])`, family = poisson, REML.
//!
//! mgcv is NOT the standard of correctness here — it is a peer smoother that is
//! itself only an *estimate* of the same truth, fit on the same noisy draws. We
//! therefore demote it to a MATCH-OR-BEAT ACCURACY BASELINE: gam's RMSE-to-truth
//! must be no worse than mgcv's RMSE-to-truth by more than 10% on the draw
//! average, and gam must not be RESOLVED worse draw by draw.
//! Both engines use six basis functions per margin. GAM now defaults to natural
//! cubic regression margins; the mgcv `bs="ps"` baseline uses cubic B-splines
//! with a second-order difference penalty. Equal dimension does not imply
//! identical function spaces or penalties. The natural-cubic mgcv diagnostic
//! separates this representation difference from differences between solvers.
//! GAM also shrinks the joint penalty null space by default; this baseline
//! does not. The standalone #1561 audit crosses both basis and shrinkage choices.
//!
//! PAIRED over draws, not one draw (#2395). Either engine's RMSE against the
//! truth depends on which Poisson draw it fit, so one draw's ratio conflates the
//! draw with the engine. The single draw this test used read gam 0.1845 against
//! mgcv `ps` 0.1565, and mgcv `cr` 0.1671 (MSI job 578202 at 90c86056d). Both engines
//! now fit the SAME `K_SEEDS` count draws on the same grid, and the decision is
//! `assert_paired_match_or_beat`. It compares them draw by draw so the common
//! noise cancels, keeps the 10% ceiling on the averaged metric, and additionally
//! refuses gam being RESOLVED worse across draws. The first draw is the seed the
//! single-draw version used. The pair is emitted before any assertion runs.
//!
//! The legacy rel_l2 / pearson "closeness to mgcv" numbers are still printed for
//! context but are NOT pass criteria — reproducing a peer tool's noisy fit is
//! not a quality claim; recovering the truth is.

use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{
    Column, PairedFoldComparison, QualityPair, assert_paired_match_or_beat, pad_to, pearson,
    relative_l2, rmse, run_r,
};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
    load_csvwith_inferred_schema,
};
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Poisson};
use std::path::Path;

// Real-data source: the `badhealth` count dataset shipped with the R package
// COUNT (Hilbe), redistributed here as bench/datasets/badhealth.csv. Each row is
// a survey respondent: `numvisit` = number of doctor visits (count response),
// `age` (years), `badh` = self-reported bad-health indicator (0/1).
const BADHEALTH_CSV: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/bench/datasets/badhealth.csv");

/// Paired Poisson draws per panel. See the "PAIRED over draws" note above.
const K_SEEDS: usize = 25;
/// Seed of the first draw: the single draw the unpaired version of this test used.
const FIRST_SEED: u64 = 345;

/// The deterministic 15×20 grid (n=300) and its noiseless mean surface: x on an
/// even grid over [0,2π], z on an even grid over [-1,1], log-mean
/// eta = 0.8 + 0.3*sin(x) + 0.2*z^2, and `mu_true[i] = exp(eta_true(x[i], z[i]))`
/// is the surface the smoother must recover.
fn poisson_tensor_grid() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let nx = 15usize;
    let nz = 20usize;
    let mut x = Vec::with_capacity(nx * nz);
    let mut z = Vec::with_capacity(nx * nz);
    let mut mu_true = Vec::with_capacity(nx * nz);
    for ix in 0..nx {
        // even grid endpoints included: x in [0, 2π], z in [-1, 1].
        let xi = (ix as f64) / ((nx - 1) as f64) * (2.0 * std::f64::consts::PI);
        for iz in 0..nz {
            let zi = -1.0 + 2.0 * (iz as f64) / ((nz - 1) as f64);
            let eta = 0.8 + 0.3 * xi.sin() + 0.2 * zi * zi;
            x.push(xi);
            z.push(zi);
            mu_true.push(eta.exp());
        }
    }
    (x, z, mu_true)
}

/// One draw of the counts, `y[i] ~ Poisson(mu_true[i])`, from the stream seeded
/// by `seed` in grid order. Only the response carries noise, so the identical
/// (x, z, y) triples reach both gam and mgcv via the shared CSV the harness
/// writes — there is no sampling difference between the two engines.
fn poisson_counts(mu_true: &[f64], seed: u64) -> Vec<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    mu_true
        .iter()
        .map(|&lambda| -> f64 {
            Poisson::new(lambda)
                .expect("poisson lambda > 0")
                .sample(&mut rng)
        })
        .collect()
}

/// Arithmetic mean of a panel column.
fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

/// gam's `y ~ te(x, z, k=[6,6])` Poisson(log) REML fit on one draw: the fitted
/// mean on the response scale at the training points, and the fit's total EDF.
fn gam_poisson_tensor_mean(x: &[f64], z: &[f64], y: &[f64]) -> (Vec<f64>, f64) {
    let n = x.len();

    // ---- build the encoded dataset for gam (columns x, z, y) --------------
    let headers = vec!["x".to_string(), "z".to_string(), "y".to_string()];
    let rows: Vec<csv::StringRecord> = (0..n)
        .map(|i| {
            csv::StringRecord::from(vec![x[i].to_string(), z[i].to_string(), y[i].to_string()])
        })
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode poisson dataset");
    let col = ds.column_map();
    let x_idx = col["x"];
    let z_idx = col["z"];

    // ---- fit with gam: y ~ te(x, z, k=[6,6]), poisson(log), REML ----------
    // Six natural-cubic basis functions per margin leave room for the
    // sin(x) / z^2 surface. The P-spline baseline has equal dimension.
    let cfg = FitConfig {
        family: Some("poisson".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("y ~ te(x, z, k=[6,6])", &ds, &cfg).expect("gam poisson te fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit for poisson(log) + te()");
    };
    let gam_edf = fit.fit.edf_total().expect("gam reports total edf");

    // gam fitted means on the response scale: rebuild the tensor design at the
    // observed (x, z), then mean = exp(design * beta) under the log link.
    let mut grid = Array2::<f64>::zeros((n, ds.headers.len()));
    for i in 0..n {
        grid[[i, x_idx]] = x[i];
        grid[[i, z_idx]] = z[i];
    }
    let design = build_term_collection_design(grid.view(), &fit.resolvedspec)
        .expect("rebuild tensor design at training points");
    let gam_eta = design.design.apply(&fit.fit.beta);
    (gam_eta.iter().map(|e| e.exp()).collect(), gam_edf)
}

#[test]
fn gam_poisson_tensor_recovers_true_mean_surface() {
    init_parallelism();

    // ---- the fixed grid and its truth, shared by every draw ---------------
    let (x, z, mu_true) = poisson_tensor_grid();
    let n = x.len();
    assert_eq!(n, 300, "grid 15x20 => n=300");
    let mu_min = mu_true.iter().copied().fold(f64::INFINITY, f64::min);
    let mu_max = mu_true.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mu_range = mu_max - mu_min;

    // ---- gam on every draw, and the long-format data mgcv replays ---------
    let mut gam_errs = Vec::with_capacity(K_SEEDS);
    let mut gam_means = Vec::with_capacity(K_SEEDS);
    let mut gam_edfs = Vec::with_capacity(K_SEEDS);
    let mut long_seed = Vec::with_capacity(K_SEEDS * n);
    let mut long_x = Vec::with_capacity(K_SEEDS * n);
    let mut long_z = Vec::with_capacity(K_SEEDS * n);
    let mut long_y = Vec::with_capacity(K_SEEDS * n);
    for k in 0..K_SEEDS {
        let seed = FIRST_SEED + k as u64;
        let y = poisson_counts(&mu_true, seed);
        let (gam_mean, gam_edf) = gam_poisson_tensor_mean(&x, &z, &y);
        let gam_err = rmse(&gam_mean, &mu_true);
        eprintln!(
            "poisson te(x,z) draw seed={seed}: gam_rmse_to_truth={gam_err:.6} gam_edf={gam_edf:.3}"
        );
        gam_errs.push(gam_err);
        gam_means.push(gam_mean);
        gam_edfs.push(gam_edf);
        long_seed.resize(long_seed.len() + n, seed as f64);
        long_x.extend_from_slice(&x);
        long_z.extend_from_slice(&z);
        long_y.extend_from_slice(&y);
    }

    // ---- the SAME draws through mgcv's P-spline baseline and natural-cubic
    // diagnostic, in ONE R session ------------------------------------------
    let r = run_r(
        &[
            Column::new("seed", &long_seed),
            Column::new("x", &long_x),
            Column::new("z", &long_z),
            Column::new("y", &long_y),
        ],
        r#"
        suppressPackageStartupMessages(library(mgcv))
        fitted_all <- c()
        edf_all <- c()
        natural_all <- c()
        natural_edf_all <- c()
        for (s in sort(unique(df$seed))) {
            d <- df[df$seed == s, c("x", "z", "y")]
            m <- gam(y ~ te(x, z, bs = "ps", k = c(6, 6)), data = d,
                     family = poisson(link = "log"), method = "REML")
            fitted_all <- c(fitted_all, as.numeric(fitted(m)))
            edf_all <- c(edf_all, sum(m$edf))
            natural <- gam(y ~ te(x, z, bs = "cr", k = c(6, 6)), data = d,
                           family = poisson(link = "log"), method = "REML")
            natural_all <- c(natural_all, as.numeric(fitted(natural)))
            natural_edf_all <- c(natural_edf_all, sum(natural$edf))
        }
        emit("fitted", fitted_all)
        emit("edf", edf_all)
        emit("natural_fitted", natural_all)
        emit("natural_edf", natural_edf_all)
        "#,
    );
    let mgcv_fitted = r.vector("fitted");
    let natural_fitted = r.vector("natural_fitted");
    let mgcv_edfs = r.vector("edf");
    let natural_edfs = r.vector("natural_edf");
    assert_eq!(
        mgcv_fitted.len(),
        K_SEEDS * n,
        "mgcv fitted panel length mismatch"
    );
    assert_eq!(
        natural_fitted.len(),
        K_SEEDS * n,
        "mgcv natural-cubic fitted panel length mismatch"
    );
    assert_eq!(mgcv_edfs.len(), K_SEEDS, "mgcv edf panel length mismatch");
    assert_eq!(
        natural_edfs.len(),
        K_SEEDS,
        "mgcv natural-cubic edf panel length mismatch"
    );

    // ---- OBJECTIVE METRIC per draw: recover the TRUE mean surface ---------
    // The pass/fail quantities are errors against `mu_true` (the noiseless
    // surface the data was generated from), NOT closeness to mgcv. We measure
    // each smoother's RMSE to the truth on the response scale.
    let mut mgcv_errs = Vec::with_capacity(K_SEEDS);
    let mut natural_errs = Vec::with_capacity(K_SEEDS);
    let mut rel_total = 0.0;
    let mut corr_total = 0.0;
    for k in 0..K_SEEDS {
        let mgcv_mean = &mgcv_fitted[k * n..(k + 1) * n];
        mgcv_errs.push(rmse(mgcv_mean, &mu_true));
        natural_errs.push(rmse(&natural_fitted[k * n..(k + 1) * n], &mu_true));
        // Context only (peer-tool agreement); explicitly NOT a pass criterion.
        rel_total += relative_l2(&gam_means[k], mgcv_mean);
        corr_total += pearson(&gam_means[k], mgcv_mean);
    }
    let panel = PairedFoldComparison::new(&gam_errs, &mgcv_errs, true);
    let natural_panel = PairedFoldComparison::new(&gam_errs, &natural_errs, true);
    eprintln!(
        "poisson te(x,z) K={K_SEEDS}-draw paired truth recovery: n={n} \
         mu_range=[{mu_min:.3},{mu_max:.3}] fold-mean rmse_to_truth gam={:.4} \
         mgcv_ps={:.4} mgcv_cr={:.4} mean_edf gam={:.3} mgcv_ps={:.3} mgcv_cr={:.3} \
         (context: mean rel_l2_vs_mgcv={:.4} mean pearson_vs_mgcv={:.5})",
        panel.gam_mean,
        panel.reference_mean,
        natural_panel.reference_mean,
        mean(&gam_edfs),
        mean(mgcv_edfs),
        mean(natural_edfs),
        rel_total / K_SEEDS as f64,
        corr_total / K_SEEDS as f64,
    );
    eprintln!("{}", panel.report("poisson_tensor::rmse_to_truth"));
    // The natural-cubic mgcv fit builds the margin kind gam builds by default.
    // Context only, never a pass criterion.
    eprintln!(
        "{}",
        natural_panel.report("poisson_tensor::rmse_to_truth_vs_mgcv_cr")
    );
    eprintln!(
        "{}",
        QualityPair::paired(
            "families",
            "quality_vs_mgcv_poisson_tensor",
            "rmse_to_truth",
            "mgcv",
            &panel,
        )
        .line()
    );

    // EDF sanity only (complexity in a signal-appropriate range), never a
    // match-to-reference: the surface has real 2-D structure (sin(x) + z^2), so
    // a sensible fit uses more than a flat plane yet far less than the full
    // 6*6-1 = 35-dim tensor basis. Every draw's fit must be sane.
    for (k, &gam_edf) in gam_edfs.iter().enumerate() {
        assert!(
            gam_edf > 1.0 && gam_edf < 35.0,
            "Poisson+te() effective degrees of freedom outside the sane range \
             (1, 35) on draw seed={}: gam_edf={gam_edf:.3}",
            FIRST_SEED + k as u64
        );
    }

    // PRIMARY claim: gam recovers the true mean surface. The absolute bar is a
    // small fraction of the signal range; well inside the per-cell Poisson
    // sampling sd, but tight enough that a biased/smeared tensor fit fails.
    let abs_bar = 0.18 * mu_range;
    assert!(
        panel.gam_mean <= abs_bar,
        "Poisson+te() failed to recover the true mean surface: \
         fold-mean RMSE(gam, truth)={:.4} > {abs_bar:.4} (= 0.18 * range {mu_range:.4})",
        panel.gam_mean
    );

    // SECONDARY claim (match-or-beat ACCURACY), paired across the K shared draws:
    // gam's error to the truth may not exceed the mature tensor smoother's by
    // more than 10% on the draw average, nor be resolved worse draw by draw.
    // mgcv is a baseline to match-or-beat on accuracy, not a correctness oracle.
    assert_paired_match_or_beat("poisson_tensor::rmse_to_truth", &panel, 1.10);
}

/// Mean Poisson deviance of a count predictor: the held-out goodness-of-fit
/// metric for a log-link count model.
///   dev_i = 2 * ( y_i * log(y_i / mu_i) - (y_i - mu_i) ),   y log y := 0 at y=0.
/// Lower is better; 0 is a perfect fit. We assert an ABSOLUTE bar on this and a
/// match-or-beat margin against mgcv on the SAME held-out rows.
fn poisson_deviance(pred_mean: &[f64], obs: &[f64]) -> f64 {
    assert_eq!(
        pred_mean.len(),
        obs.len(),
        "poisson_deviance length mismatch"
    );
    let n = obs.len() as f64;
    let mut s = 0.0;
    for (&mu, &y) in pred_mean.iter().zip(obs) {
        let mu = mu.max(1e-12);
        let ylogymu = if y > 0.0 { y * (y / mu).ln() } else { 0.0 };
        s += 2.0 * (ylogymu - (y - mu));
    }
    s / n.max(1.0)
}

/// REAL-DATA arm of the SAME capability (Poisson(log) × tensor-product te()),
/// exercised on the `badhealth` count survey instead of a known-truth surface.
///
/// On real data the truth is unknown, so quality is OBJECTIVE held-out
/// predictive fit, not truth recovery and not tool matching. We fit
///     numvisit ~ te(age, badh)
/// (Poisson log link, REML) on a deterministic train split, predict the held-out
/// rows, and assert two things on gam's OWN held-out predictions:
///
///   PRIMARY (absolute, tool-free): held-out mean Poisson deviance below a fixed
///     bar that a sensible count model clears but the intercept-only (predict the
///     train mean) model does not. The intercept-only deviance is computed in
///     plain Rust and the bar sits comfortably under it, so passing proves gam's
///     te(age, badh) surface genuinely improves held-out count fit.
///
///   BASELINE (match-or-beat): mgcv fits the SAME training rows and predicts the
///     SAME held-out rows; gam's held-out deviance must be no worse than
///     `mgcv_dev * 1.10`. mgcv is a peer baseline to match-or-beat, never a
///     target to reproduce.
#[test]
fn gam_poisson_tensor_recovers_true_mean_surface_on_real_data() {
    init_parallelism();

    // ---- load the real badhealth count dataset ----------------------------
    let ds = load_csvwith_inferred_schema(Path::new(BADHEALTH_CSV)).expect("load badhealth.csv");
    let col = ds.column_map();
    let age_idx = col["age"];
    let badh_idx = col["badh"];
    let numvisit_idx = col["numvisit"];
    let age: Vec<f64> = ds.values.column(age_idx).to_vec();
    let badh: Vec<f64> = ds.values.column(badh_idx).to_vec();
    let numvisit: Vec<f64> = ds.values.column(numvisit_idx).to_vec();
    let n = age.len();
    assert!(n > 1000, "badhealth should have ~1127 rows, got {n}");

    // ---- deterministic train/test split: every 4th row is held out --------
    let is_test = |i: usize| i % 4 == 0;
    let train_rows: Vec<usize> = (0..n).filter(|&i| !is_test(i)).collect();
    let test_rows: Vec<usize> = (0..n).filter(|&i| is_test(i)).collect();
    assert!(
        train_rows.len() > 700 && test_rows.len() > 250,
        "split sizes: train={} test={}",
        train_rows.len(),
        test_rows.len()
    );

    let train_age: Vec<f64> = train_rows.iter().map(|&i| age[i]).collect();
    let train_badh: Vec<f64> = train_rows.iter().map(|&i| badh[i]).collect();
    let train_numvisit: Vec<f64> = train_rows.iter().map(|&i| numvisit[i]).collect();
    let test_age: Vec<f64> = test_rows.iter().map(|&i| age[i]).collect();
    let test_badh: Vec<f64> = test_rows.iter().map(|&i| badh[i]).collect();
    let test_numvisit: Vec<f64> = test_rows.iter().map(|&i| numvisit[i]).collect();

    // Build a training-only dataset by sub-setting the encoded rows; headers,
    // schema and column kinds are unchanged, so the formula resolves identically.
    let p = ds.headers.len();
    let mut train_values = Array2::<f64>::zeros((train_rows.len(), p));
    for (out_row, &src_row) in train_rows.iter().enumerate() {
        for c in 0..p {
            train_values[[out_row, c]] = ds.values[[src_row, c]];
        }
    }
    let mut train_ds = ds.clone();
    train_ds.values = train_values;

    // ---- fit gam on TRAIN: numvisit ~ te(age, badh), poisson(log), REML ----
    let cfg = FitConfig {
        family: Some("poisson".to_string()),
        ..FitConfig::default()
    };
    let result =
        fit_from_formula("numvisit ~ te(age, badh)", &train_ds, &cfg).expect("gam poisson te fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit for poisson(log) + te() on real data");
    };
    let gam_edf = fit.fit.edf_total().expect("gam reports total edf");

    // gam held-out means: rebuild the tensor design at the test (age, badh),
    // mean = exp(design * beta) under the log link.
    let mut test_grid = Array2::<f64>::zeros((test_rows.len(), p));
    for (i, &row) in test_rows.iter().enumerate() {
        test_grid[[i, age_idx]] = age[row];
        test_grid[[i, badh_idx]] = badh[row];
    }
    let test_design = build_term_collection_design(test_grid.view(), &fit.resolvedspec)
        .expect("rebuild tensor design at held-out points");
    let gam_test_eta = test_design.design.apply(&fit.fit.beta);
    let gam_test_mean: Vec<f64> = gam_test_eta.iter().map(|e| e.exp()).collect();

    // ---- fit the SAME capability on TRAIN with mgcv, predict the SAME TEST ---
    // mgcv's baseline is the P-spline smooth-by-factor described in the R body.
    // The test columns ride along padded; only the first k are read.
    let r = run_r(
        &[
            Column::new("age", &train_age),
            Column::new("badh", &train_badh),
            Column::new("numvisit", &train_numvisit),
            Column::new("test_age", &pad_to(&test_age, train_age.len())),
            Column::new("test_badh", &pad_to(&test_badh, train_age.len())),
            Column::new("test_n", &vec![test_age.len() as f64; train_age.len()]),
        ],
        r#"
        suppressPackageStartupMessages(library(mgcv))
        # badh is binary {0,1}: a tensor te(age, badh) P-spline margin on badh is
        # larger than its 2 unique values, so mgcv's inner loop "can't correct
        # step size" / fails to converge. The mgcv-idiomatic encoding of the SAME
        # age x badh interaction that gam's te(age, badh) represents over a binary
        # margin is a smooth-by-factor: a separate s(age) curve per badh level
        # plus the badh main effect. The age curve is a ps basis, while gam's
        # default te() margin with k >= 3 is a natural cubic regression spline:
        # the same capability, not the same construction.
        df$badhf <- factor(df$badh)
        m <- gam(numvisit ~ s(age, bs = "ps", by = badhf) + badhf, data = df,
                 family = poisson(link = "log"), method = "REML")
        emit("edf", sum(m$edf))
        k <- df$test_n[1]
        newd <- data.frame(age = df$test_age[1:k],
                           badhf = factor(df$test_badh[1:k], levels = levels(df$badhf)))
        emit("test_pred", as.numeric(predict(m, newdata = newd, type = "response")))
        "#,
    );
    let mgcv_test_mean = r.vector("test_pred");
    let mgcv_edf = r.scalar("edf");
    assert_eq!(
        mgcv_test_mean.len(),
        test_rows.len(),
        "mgcv held-out prediction length mismatch"
    );

    // ---- OBJECTIVE held-out metric: mean Poisson deviance ------------------
    let gam_dev = poisson_deviance(&gam_test_mean, &test_numvisit);
    let mgcv_dev = poisson_deviance(mgcv_test_mean, &test_numvisit);

    // Intercept-only baseline: predict every held-out row with the TRAIN mean
    // count. Its held-out deviance is the "no covariate information" reference a
    // real model must beat; computed in plain Rust, no tool involved.
    let train_mean = train_numvisit.iter().sum::<f64>() / train_numvisit.len() as f64;
    let null_pred = vec![train_mean; test_rows.len()];
    let null_dev = poisson_deviance(&null_pred, &test_numvisit);

    eprintln!(
        "badhealth te(age,badh) held-out: n_train={} n_test={} gam_edf={gam_edf:.3} \
         mgcv_edf={mgcv_edf:.3} gam_dev={gam_dev:.4} mgcv_dev={mgcv_dev:.4} \
         null_dev={null_dev:.4}",
        train_rows.len(),
        test_rows.len(),
    );
    eprintln!(
        "{}",
        QualityPair::error(
            "families",
            "quality_vs_mgcv_poisson_tensor::real_data",
            "held_out_poisson_deviance",
            gam_dev,
            "mgcv",
            mgcv_dev,
        )
        .line()
    );

    // ---- PRIMARY objective assertion: gam beats the null model with margin --
    // The covariate surface must add real held-out predictive value over the
    // intercept-only model. The badhealth visit counts are intrinsically noisy:
    // the mature mgcv reference itself only reaches mean Poisson deviance ~3.26
    // here (vs the ~3.54 null), so an absolute floor must sit ABOVE the mature
    // tool's own deviance — a sub-reference fixed cutoff (the earlier 2.75)
    // demanded gam beat mgcv's 3.26 by ~15% in absolute terms, contradicting the
    // match-or-beat gate below. We instead require gam to clear the null by a
    // clear margin AND land at most 5% below the null (above mgcv's 3.26, below
    // the 3.54 null), catching a broken covariate surface (which would sit at
    // ~null) without penalizing gam for the DGP's irreducible Poisson noise.
    assert!(
        gam_dev < null_dev * 0.97,
        "gam te(age,badh) held-out deviance {gam_dev:.4} did not beat the \
         intercept-only null {null_dev:.4} by a 3% margin"
    );
    assert!(
        gam_dev <= null_dev * 0.95,
        "gam te(age,badh) held-out mean Poisson deviance {gam_dev:.4} too close to \
         null {null_dev:.4} (bar = 0.95*null = {:.4}; mgcv reaches {mgcv_dev:.4})",
        null_dev * 0.95
    );

    // ---- BASELINE (match-or-beat): no worse than mgcv on held-out deviance --
    assert!(
        gam_dev <= mgcv_dev * 1.10,
        "gam held-out deviance {gam_dev:.4} exceeds mgcv {mgcv_dev:.4} * 1.10"
    );

    // ---- complexity sanity: edf in a sane range (not matched to mgcv) -------
    assert!(
        gam_edf > 1.0 && gam_edf < 35.0,
        "gam effective dof out of sane range: {gam_edf:.3}"
    );
}
