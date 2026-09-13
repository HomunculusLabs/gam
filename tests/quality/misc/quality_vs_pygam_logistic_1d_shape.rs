//! End-to-end quality: gam's 1-D **binomial(logit) smooth** recovers a KNOWN
//! true logit-scale function, measured on the **logit (linear-predictor) scale**.
//!
//! OBJECTIVE METRIC ASSERTED (truth recovery): the data is generated from a known
//! ground-truth function `eta_truth = 1.2*sin(pi*x/50)` with Bernoulli responses,
//! so the test asserts `RMSE(eta_gam, eta_truth)` is small in absolute terms
//! (below a principled fraction of the signal's logit range). This is a direct
//! claim that gam recovers the true smooth — NOT that gam reproduces any other
//! tool's noisy fit.
//!
//! pyGAM's `LogisticGAM` is fit on the *identical* data and kept only as a
//! BASELINE TO MATCH-OR-BEAT on the same recovery metric: gam's RMSE-to-truth
//! must be no worse than pyGAM's by more than a 10% margin. Matching pyGAM's
//! fitted output is explicitly NOT a pass criterion — pyGAM could itself be
//! off, and two engines agreeing on a wrong answer proves nothing. The pearson
//! correlation and rel_l2 between the two fits are still computed and printed
//! with `eprintln!` purely for diagnostic context.
//!
//! We measure on the **linear-predictor (logit) scale** deliberately: the
//! probability scale compresses extreme eta through the squashing inverse-link
//! and would mask divergence in the tails. eta is where the smoother lives, and
//! truth is known exactly there.
//!
//! Data is a fixed-seed synthetic 1-D problem (n=400, x in [0,100]). The truth
//! amplitude (eta spanning roughly [-1.2, 1.2], probabilities ~0.23..0.77) is
//! deliberately well above the noise floor: binary data carries little
//! information per point, so a near-flat truth would let any engine shrink to an
//! essentially constant eta and make an RMSE bar trivially passable. A genuine,
//! identifiable curve keeps the recovery claim load-bearing.
//!
//! PAIRED over draws, not one draw (#2395). Either engine's RMSE against the
//! truth depends on which Bernoulli draw it fit, so one draw's ratio conflates
//! the draw with the engine. The single draw this test used read gam 0.3946
//! against pyGAM 0.3431 at fa0e33bb4 (CI run 34702231507), with pyGAM's λ fixed
//! at its default. Both engines now fit the SAME `K_SEEDS` draws, and the
//! decision is `assert_paired_match_or_beat`. It compares them draw by draw so
//! the common noise cancels, keeps the 10% ceiling on the averaged metric, and
//! additionally refuses gam being RESOLVED worse across draws. The first draw is
//! the seed the single-draw version used.
//!
//! Truth-recovery bar (principled, un-weakened): the true eta ranges over a span
//! of `2*1.2 = 2.4` logits. Bernoulli responses are extremely noisy (each carries
//! < 1 bit), so a penalized smooth at n=400 cannot pin eta tightly; we require
//! the draw-averaged `RMSE(eta_gam, eta_truth) < 0.45`, i.e. under ~19% of the
//! signal span — small enough that a flat or wrong-phase fit (RMSE near the
//! truth's own RMS of ~0.85) fails, yet honest about the binary noise floor. EDF
//! is reported for context only and not asserted against the reference.

use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{
    Column, PairedFoldComparison, QualityPair, assert_paired_match_or_beat, auc_no_skill_floor,
    pearson, relative_l2, rmse, run_python,
};
use gam::{FitConfig, FitResult, fit_from_formula, init_parallelism, load_csvwith_inferred_schema};
use ndarray::Array2;
use std::io::Write;
use std::path::Path;

/// Rows per draw.
const N_ROWS: usize = 400;
/// Evaluation grid points spanning [0, 100].
const N_GRID: usize = 100;
/// Paired Bernoulli draws per panel. See the "PAIRED over draws" note above.
const K_SEEDS: usize = 25;
/// Seed of the first draw: the single draw the unpaired version of this test used.
const FIRST_SEED: u64 = 42;

/// Known logit-scale truth: a smooth sinusoid completing one full period over
/// the domain.
fn truth_eta(xi: f64) -> f64 {
    1.2 * (std::f64::consts::PI * xi / 50.0).sin()
}

/// One draw's `(x, y)`: x uniform on [0,100] and y ~ Bernoulli(logistic(eta)).
/// A self-contained SplitMix64 stream (no external RNG crate) makes the x and y
/// vectors byte-identical for both engines.
fn logistic_draw(seed: u64) -> (Vec<f64>, Vec<f64>) {
    let mut state: u64 = seed;
    let mut next_unit = || -> f64 {
        // SplitMix64-style advance; map to [0,1).
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 11) as f64 / (1u64 << 53) as f64
    };
    let mut x = vec![0.0f64; N_ROWS];
    let mut y = vec![0.0f64; N_ROWS];
    for i in 0..N_ROWS {
        let xi = 100.0 * next_unit();
        let p = 1.0 / (1.0 + (-truth_eta(xi)).exp());
        let yi = if next_unit() < p { 1.0 } else { 0.0 };
        x[i] = xi;
        y[i] = yi;
    }
    (x, y)
}

/// A penalized smooth on a centered basis recovers eta only up to an additive
/// constant (the intercept), so curves are de-meaned before their shapes are
/// compared.
fn demean(v: &[f64]) -> Vec<f64> {
    let m = v.iter().sum::<f64>() / v.len() as f64;
    v.iter().map(|value| value - m).collect()
}

/// gam's `y ~ s(x, k=10)` binomial/logit REML fit on one draw: the logit-scale
/// linear predictor on the grid `xg`, and the fit's total EDF.
fn gam_logit_eta_on_grid(seed: u64, x: &[f64], y: &[f64], xg: &[f64]) -> (Vec<f64>, f64) {
    // (load_csvwith_inferred_schema is the same loader the canonical reference
    // tests use; a temp file keeps the data path identical to those tests.)
    let mut csv = String::from("x,y\n");
    for i in 0..x.len() {
        csv.push_str(&format!("{:.17e},{}\n", x[i], y[i] as i64));
    }
    let mut tmp = std::env::temp_dir();
    tmp.push(format!(
        "gam_pygam_logistic_1d_{}_{}_{seed}.csv",
        std::process::id(),
        x.len()
    ));
    {
        let mut f = std::fs::File::create(&tmp).expect("create synthetic csv");
        f.write_all(csv.as_bytes()).expect("write synthetic csv");
    }
    let ds = load_csvwith_inferred_schema(&tmp).expect("load synthetic logistic data");
    if let Err(err) = std::fs::remove_file(&tmp) {
        // The data is already loaded; a failed cleanup leaks one temp file and
        // must not fail the test, but it should not vanish either.
        eprintln!(
            "warning: could not remove the synthetic logistic csv {}: {err}",
            tmp.display()
        );
    }
    let col = ds.column_map();
    let x_idx = col["x"];

    let cfg = FitConfig {
        family: Some("binomial".to_string()),
        link: Some("logit".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("y ~ s(x, k=10)", &ds, &cfg).expect("gam binomial(logit) fit");
    let FitResult::Standard(fit) = result else {
        panic!("binomial(logit) 1-D smooth should be a Standard GAM fit");
    };
    let gam_edf = fit.fit.edf_total().expect("gam reports total edf");

    // The logit-scale eta is exactly design*beta (the link is applied
    // afterwards), rebuilt from the frozen resolved spec at the grid.
    let mut grid = Array2::<f64>::zeros((xg.len(), ds.headers.len()));
    for (j, &xj) in xg.iter().enumerate() {
        grid[[j, x_idx]] = xj;
    }
    let design = build_term_collection_design(grid.view(), &fit.resolvedspec)
        .expect("rebuild 1-D smooth design on evaluation grid");
    let gam_eta: Vec<f64> = design.design.apply(&fit.fit.beta).to_vec();
    assert_eq!(gam_eta.len(), xg.len(), "gam eta grid length mismatch");
    (gam_eta, gam_edf)
}

#[test]
fn gam_logistic_1d_shape_matches_pygam_on_logit_scale() {
    init_parallelism();

    // ---- evaluation grid and the known truth on it -------------------------
    // A dense, sorted grid spanning the domain, so the shape comparison is
    // independent of each draw's sampling density.
    let xg: Vec<f64> = (0..N_GRID)
        .map(|j| 100.0 * (j as f64) / (N_GRID as f64 - 1.0))
        .collect();
    let truth_grid: Vec<f64> = xg.iter().map(|&xj| truth_eta(xj)).collect();
    let truth_c = demean(&truth_grid);
    // RMS of the (de-meaned) truth itself — the error a degenerate flat fit would
    // incur, i.e. the scale the recovery bar must beat.
    let truth_rms = (truth_c.iter().map(|t| t * t).sum::<f64>() / truth_c.len() as f64).sqrt();

    // ---- gam on every draw, and the long-format data pyGAM replays ---------
    let mut gam_errs = Vec::with_capacity(K_SEEDS);
    let mut gam_etas = Vec::with_capacity(K_SEEDS);
    let mut gam_edf_total = 0.0;
    let mut long_seed = Vec::with_capacity(K_SEEDS * N_ROWS);
    let mut long_x = Vec::with_capacity(K_SEEDS * N_ROWS);
    let mut long_y = Vec::with_capacity(K_SEEDS * N_ROWS);
    for k in 0..K_SEEDS {
        let seed = FIRST_SEED + k as u64;
        let (x, y) = logistic_draw(seed);
        // Sanity: both classes present (required for a meaningful logistic fit).
        let n_pos = y.iter().filter(|&&v| v > 0.5).count();
        assert!(
            n_pos > 10 && n_pos < N_ROWS - 10,
            "draw {seed} should have both classes well represented, got {n_pos}/{N_ROWS} positives"
        );
        let (gam_eta, gam_edf) = gam_logit_eta_on_grid(seed, &x, &y, &xg);
        gam_errs.push(rmse(&demean(&gam_eta), &truth_c));
        gam_etas.push(gam_eta);
        gam_edf_total += gam_edf;
        for i in 0..N_ROWS {
            long_seed.push(seed as f64);
            long_x.push(x[i]);
            long_y.push(y[i]);
        }
    }

    // ---- the SAME draws through pyGAM's LogisticGAM, in ONE Python session ----
    // LogisticGAM(s(0, n_splines=10)) is a penalized binomial PIRLS fit over a
    // cubic B-spline smooth at pyGAM's default λ. The logit-scale predictor is
    // computed from the predicted probability, to avoid relying on a private
    // attribute. The grid travels as a (padded) column so both engines see the
    // exact same query points; only its first N_GRID entries are meaningful.
    let mut xg_column = vec![0.0f64; K_SEEDS * N_ROWS];
    xg_column[..N_GRID].copy_from_slice(&xg);
    let py = run_python(
        &[
            Column::new("seed", &long_seed),
            Column::new("x", &long_x),
            Column::new("y", &long_y),
            Column::new("xg", &xg_column),
        ],
        r#"
from pygam import LogisticGAM, s
n_grid = 100
Xg = np.asarray(df["xg"], dtype=float)[:n_grid].reshape(-1, 1)
seeds = np.asarray(df["seed"], dtype=float)
xs = np.asarray(df["x"], dtype=float)
ys = np.asarray(df["y"], dtype=float)
eta_all = []
edf_all = []
for seed in np.unique(seeds):
    mask = seeds == seed
    gam = LogisticGAM(s(0, n_splines=10)).fit(xs[mask].reshape(-1, 1), ys[mask])
    mu = np.asarray(gam.predict_mu(Xg), dtype=float)
    # clip mu off the open-interval boundary so the logit is finite even when a
    # grid point sits in a near-saturated region.
    mu = np.clip(mu, 1e-12, 1.0 - 1e-12)
    eta_all.extend(np.log(mu / (1.0 - mu)).tolist())
    edf_all.append(float(gam.statistics_["edof"]))
emit("eta", eta_all)
emit("edf", edf_all)
"#,
    );
    let pygam_eta_flat = py.vector("eta");
    let pygam_edfs = py.vector("edf");
    assert_eq!(
        pygam_eta_flat.len(),
        K_SEEDS * N_GRID,
        "pyGAM eta panel length mismatch"
    );
    assert_eq!(pygam_edfs.len(), K_SEEDS, "pyGAM edf panel length mismatch");

    // ---- OBJECTIVE METRIC per draw: recovery of the known true eta ----------
    // pyGAM is scored on the SAME truth, as a match-or-beat accuracy baseline and
    // never as the target itself.
    let mut pygam_errs = Vec::with_capacity(K_SEEDS);
    let mut corr_total = 0.0;
    let mut rel_total = 0.0;
    for k in 0..K_SEEDS {
        let pygam_eta = &pygam_eta_flat[k * N_GRID..(k + 1) * N_GRID];
        pygam_errs.push(rmse(&demean(pygam_eta), &truth_c));
        // Diagnostic context only (NOT assertion criteria): how close the two
        // fitted predictors are to each other.
        corr_total += pearson(&gam_etas[k], pygam_eta);
        rel_total += relative_l2(&gam_etas[k], pygam_eta);
    }
    let gam_edf_mean = gam_edf_total / K_SEEDS as f64;
    let pygam_edf_mean = pygam_edfs.iter().sum::<f64>() / K_SEEDS as f64;

    let panel = PairedFoldComparison::new(&gam_errs, &pygam_errs, true);
    eprintln!(
        "synthetic logistic s(x,k=10) K={K_SEEDS}-draw paired: n={N_ROWS} truth_rms={truth_rms:.3} \
         fold-mean err_to_truth gam={:.4} pygam={:.4} mean_edf gam={gam_edf_mean:.3} \
         pygam={pygam_edf_mean:.3} [diag only] mean eta-vs-pygam pearson={:.5} rel_l2={:.4}",
        panel.gam_mean,
        panel.reference_mean,
        corr_total / K_SEEDS as f64,
        rel_total / K_SEEDS as f64,
    );
    eprintln!("{}", panel.report("pygam_logistic_1d::err_to_truth"));
    eprintln!(
        "{}",
        QualityPair::paired(
            "misc",
            "quality_vs_pygam_logistic_1d_shape",
            "err_to_truth",
            "pygam",
            &panel,
        )
        .line()
    );

    // (1) PRIMARY truth-recovery claim, on the draw average: gam's fitted
    // logit-scale eta tracks the true sinusoid. The true (de-meaned) eta has RMS
    // ~0.85 over a 2.4-logit span; a flat or wrong-phase fit incurs RMSE of that
    // order. Requiring RMSE < 0.45 (under ~19% of the signal span) demands genuine
    // shape recovery while respecting that Bernoulli data carries < 1 bit/point
    // and cannot pin eta tightly at n=400. A wrong binomial reweight or misapplied
    // penalty pushes eta toward flat/wrong and blows past this bar.
    assert!(
        panel.gam_mean < 0.45 && panel.gam_mean < truth_rms,
        "gam fails to recover the true logit-scale smooth: \
         fold-mean RMSE(eta_gam, truth)={:.4} (bar 0.45, truth_rms={truth_rms:.3})",
        panel.gam_mean
    );
    // (2) MATCH-OR-BEAT the mature baseline on the SAME accuracy metric, paired
    // across the shared draws: gam's averaged recovery error may be no worse than
    // pyGAM's by more than 10%, nor resolved worse draw by draw.
    assert_paired_match_or_beat("pygam_logistic_1d::err_to_truth", &panel, 1.10);
}

/// Held-out AUC (rank statistic = P(score_pos > score_neg)) of `score` against
/// binary `label`, computed in plain Rust via the Mann-Whitney U identity with
/// average ranks for ties. 1.0 is perfect separation, 0.5 is chance.
fn auc(score: &[f64], label: &[f64]) -> f64 {
    assert_eq!(score.len(), label.len(), "auc length mismatch");
    let n = score.len();
    // Rank the scores (1-based), averaging ranks within tied groups.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| score[a].partial_cmp(&score[b]).expect("auc: NaN score"));
    let mut ranks = vec![0.0f64; n];
    let mut i = 0usize;
    while i < n {
        let mut j = i + 1;
        while j < n && score[order[j]] == score[order[i]] {
            j += 1;
        }
        // tied block order[i..j] gets the average of ranks (i+1)..=j
        let avg = ((i + 1 + j) as f64) / 2.0;
        for &o in &order[i..j] {
            ranks[o] = avg;
        }
        i = j;
    }
    let n_pos = label.iter().filter(|&&v| v > 0.5).count();
    let n_neg = n - n_pos;
    assert!(
        n_pos > 0 && n_neg > 0,
        "auc needs both classes present (pos={n_pos} neg={n_neg})"
    );
    let sum_ranks_pos: f64 = (0..n).filter(|&k| label[k] > 0.5).map(|k| ranks[k]).sum();
    let u_pos = sum_ranks_pos - (n_pos as f64) * (n_pos as f64 + 1.0) / 2.0;
    u_pos / (n_pos as f64 * n_neg as f64)
}

/// Mean binomial log-loss (negative log-likelihood per observation), with
/// probabilities clipped off the open-interval boundary so the log is finite.
fn log_loss(prob: &[f64], label: &[f64]) -> f64 {
    assert_eq!(prob.len(), label.len(), "log_loss length mismatch");
    let n = prob.len() as f64;
    let s: f64 = prob
        .iter()
        .zip(label)
        .map(|(&p, &y)| {
            let p = p.clamp(1e-12, 1.0 - 1e-12);
            -(y * p.ln() + (1.0 - y) * (1.0 - p).ln())
        })
        .sum();
    s / n.max(1.0)
}

/// End-to-end quality on REAL data: gam's 1-D binomial(logit) smooth must
/// PREDICT a held-out binary outcome well — measured by objective held-out AUC
/// and log-loss — and match-or-beat pyGAM's `LogisticGAM` on the SAME metrics.
///
/// Dataset SOURCE: `bench/datasets/prostate.csv` — a binary prostate-cancer
/// outcome `y` with continuous predictors `pc1`, `pc2` (principal-component
/// scores). This arm exercises the SAME capability as the synthetic test above
/// (a single 1-D penalized binomial/logit smooth, here `y ~ s(pc1)`), but on
/// real data with NO known ground truth, so the quality claim is out-of-sample
/// predictive accuracy rather than truth recovery.
#[test]
fn gam_logistic_1d_shape_matches_pygam_on_logit_scale_on_real_data() {
    init_parallelism();

    // ---- load the real prostate dataset (pc1 -> binary y) -----------------
    let ds = load_csvwith_inferred_schema(Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/bench/datasets/prostate.csv"
    )))
    .expect("load prostate.csv");
    let col = ds.column_map();
    let pc1_idx = col["pc1"];
    let y_idx = col["y"];
    let pc1: Vec<f64> = ds.values.column(pc1_idx).to_vec();
    let yv: Vec<f64> = ds.values.column(y_idx).to_vec();
    let n = pc1.len();
    assert!(n > 400, "prostate should have ~654 rows, got {n}");

    // ---- deterministic train/test split: every 4th row held out ----------
    let is_test = |i: usize| i % 4 == 0;
    let train_rows: Vec<usize> = (0..n).filter(|&i| !is_test(i)).collect();
    let test_rows: Vec<usize> = (0..n).filter(|&i| is_test(i)).collect();
    assert!(
        train_rows.len() > 300 && test_rows.len() > 100,
        "split sizes: train={} test={}",
        train_rows.len(),
        test_rows.len()
    );

    let train_pc1: Vec<f64> = train_rows.iter().map(|&i| pc1[i]).collect();
    let train_y: Vec<f64> = train_rows.iter().map(|&i| yv[i]).collect();
    let test_pc1: Vec<f64> = test_rows.iter().map(|&i| pc1[i]).collect();
    let test_y: Vec<f64> = test_rows.iter().map(|&i| yv[i]).collect();
    // Both classes must be present in both splits for AUC/log-loss to be sound.
    let train_pos = train_y.iter().filter(|&&v| v > 0.5).count();
    let test_pos = test_y.iter().filter(|&&v| v > 0.5).count();
    assert!(
        train_pos > 10
            && train_pos < train_y.len() - 10
            && test_pos > 10
            && test_pos < test_y.len() - 10,
        "both classes must appear in each split: train_pos={train_pos}/{} test_pos={test_pos}/{}",
        train_y.len(),
        test_y.len()
    );

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

    // ---- fit gam on TRAIN: y ~ s(pc1), binomial / logit / REML ------------
    let cfg = FitConfig {
        family: Some("binomial".to_string()),
        link: Some("logit".to_string()),
        ..FitConfig::default()
    };
    let result =
        fit_from_formula("y ~ s(pc1, k=10)", &train_ds, &cfg).expect("gam binomial(logit) fit");
    let FitResult::Standard(fit) = result else {
        panic!("binomial(logit) 1-D smooth should be a Standard GAM fit");
    };
    let gam_edf = fit.fit.edf_total().expect("gam reports total edf");

    // gam predictions at the held-out pc1 points: rebuild the design from the
    // frozen spec, apply beta to get the logit-scale eta, then squash to prob.
    let mut test_grid = Array2::<f64>::zeros((test_rows.len(), p));
    for (i, &v) in test_pc1.iter().enumerate() {
        test_grid[[i, pc1_idx]] = v;
    }
    let test_design = build_term_collection_design(test_grid.view(), &fit.resolvedspec)
        .expect("rebuild design at held-out points");
    let gam_test_eta: Vec<f64> = test_design.design.apply(&fit.fit.beta).to_vec();
    let gam_test_prob: Vec<f64> = gam_test_eta
        .iter()
        .map(|&e| 1.0 / (1.0 + (-e).exp()))
        .collect();
    assert_eq!(gam_test_prob.len(), test_rows.len());

    // ---- fit the SAME model on TRAIN with pyGAM, predict the SAME TEST -----
    // One run_python call exposes a single equal-length data.frame, so train and
    // test columns must share a length: we pass train pc1/y plus the test pc1
    // (and test count) padded into parallel train-length columns; pyGAM reads
    // only the first `test_n` entries of the test column back.
    let pad_to = |v: &[f64], len: usize| -> Vec<f64> {
        let fill = v.last().copied().unwrap_or(0.0);
        let mut out = v.to_vec();
        out.resize(len, fill);
        out
    };
    let py = run_python(
        &[
            Column::new("pc1", &train_pc1),
            Column::new("y", &train_y),
            Column::new("test_pc1", &pad_to(&test_pc1, train_pc1.len())),
            Column::new("test_n", &vec![test_pc1.len() as f64; train_pc1.len()]),
        ],
        r#"
from pygam import LogisticGAM, s
X = np.asarray(df["pc1"], dtype=float).reshape(-1, 1)
yv = np.asarray(df["y"], dtype=float)
k = int(np.asarray(df["test_n"], dtype=float)[0])
Xt = np.asarray(df["test_pc1"], dtype=float)[:k].reshape(-1, 1)
gam = LogisticGAM(s(0, n_splines=10)).fit(X, yv)
mu = np.asarray(gam.predict_mu(Xt), dtype=float)
emit("test_prob", mu)
emit("edf", [float(gam.statistics_["edof"])])
"#,
    );
    let pygam_test_prob = py.vector("test_prob");
    let pygam_edf = py.scalar("edf");
    assert_eq!(
        pygam_test_prob.len(),
        test_rows.len(),
        "pyGAM held-out prediction length mismatch"
    );

    // ---- OBJECTIVE held-out metrics on gam's OWN predictions --------------
    let gam_auc = auc(&gam_test_prob, &test_y);
    let gam_ll = log_loss(&gam_test_prob, &test_y);
    let pygam_auc = auc(pygam_test_prob, &test_y);
    let pygam_ll = log_loss(pygam_test_prob, &test_y);

    eprintln!(
        "prostate s(pc1,k=10) held-out: n_train={} n_test={} gam_edf={gam_edf:.3} \
         pygam_edf={pygam_edf:.3} gam_auc={gam_auc:.4} pygam_auc={pygam_auc:.4} \
         gam_logloss={gam_ll:.4} pygam_logloss={pygam_ll:.4}",
        train_rows.len(),
        test_rows.len(),
    );

    // ---- PRIMARY objective assertion: gam discriminates held-out classes ---
    // pc1 is a PC score predictive of the prostate outcome; a competent 1-D
    // binomial smooth must rank held-out positives above negatives beyond chance
    // (AUC = 0.5). The achievable AUC is bounded by how much a SINGLE PC carries
    // about the binary outcome on this split — for prostate that ceiling is only
    // ~0.64 (pyGAM's mature LogisticGAM tops out there too), so an absolute floor
    // like 0.70 is unreachable by ANY method and would assert signal the data does
    // not contain. The principled tool-free bar is therefore "significantly above
    // chance" sized to the held-out split: AUC at least 2 SE above 0.5 (one-sided
    // ~97.7%). A flat/wrong fit (AUC ≈ 0.5) fails it; the actual accuracy ceiling
    // is scored by the match-or-beat-pyGAM arm below.
    let no_skill = auc_no_skill_floor(test_pos, test_y.len() - test_pos, 2.0);
    assert!(
        gam_auc >= no_skill,
        "gam's held-out AUC not above chance: {gam_auc:.4} (< {no_skill:.4}, \
         2 SE above 0.5 for {test_pos}/{} positives)",
        test_y.len()
    );

    // ---- BASELINE (match-or-beat) on the SAME held-out metrics -------------
    // pyGAM's LogisticGAM is the mature baseline; gam must be no worse on AUC by
    // more than a 0.02 absolute margin, and no worse on log-loss by more than a
    // 10% margin. pyGAM is a competitor to beat on objective accuracy, never a
    // fitted target to reproduce.
    assert!(
        gam_auc >= pygam_auc - 0.02,
        "gam held-out AUC {gam_auc:.4} worse than pyGAM {pygam_auc:.4} by > 0.02"
    );
    assert!(
        gam_ll <= pygam_ll * 1.10,
        "gam held-out log-loss {gam_ll:.4} exceeds pyGAM {pygam_ll:.4} * 1.10"
    );

    // ---- complexity sanity: edf in a signal-appropriate range (not matched) -
    assert!(
        gam_edf > 1.0 && gam_edf < 30.0,
        "gam effective dof out of sane range: {gam_edf:.3}"
    );
}
