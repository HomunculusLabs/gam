//! End-to-end quality: gam's penalized Gaussian smooth must PREDICT well on
//! held-out data — not merely reproduce mgcv's in-sample fit.
//!
//! The lidar benchmark (`logratio ~ s(range)`) is real data with no known
//! ground-truth function, so the objective quality of a smoother is its
//! out-of-sample predictive accuracy.
//!
//! #2395 K-split averaging: the former single deterministic hold-out put the
//! gam-vs-mgcv margin on a knife-edge that flipped sign across splits (pure
//! single-split noise). We now score K random train/test partitions and average
//! the held-out metric. gam and mgcv are scored on the SAME K partitions
//! (identical 0/1 fold masks shipped into the R body), so the paired comparison
//! stays honest; only the split noise is averaged away. Objective bars on gam's
//! OWN averaged predictions:
//!
//!   PRIMARY (objective, tool-free): AVERAGED held-out `test_R2 >= 0.55` — gam's
//!     smooth genuinely explains held-out variance, well above the constant-mean
//!     predictor (R2 = 0).
//!
//!   BASELINE (match-or-beat): mgcv fits the SAME training rows and predicts the
//!     SAME held-out rows of each partition, so the two arms are PAIRED split by
//!     split. The verdict comes from `assert_paired_match_or_beat`: gam may not
//!     be behind by more than the paired split-to-split spread can explain, AND
//!     its averaged held-out RMSE must still clear the pre-existing
//!     `mgcv_rmse_avg * 1.10` ceiling. The first clause resolves gaps far inside
//!     that ceiling, so this is strictly harder than the former single split.

use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{
    Column, PairedFoldComparison, QualityPair, assert_paired_match_or_beat,
    paired_holdout_partition, r2, rmse, run_r,
};
use gam::{FitConfig, FitResult, fit_from_formula, init_parallelism, load_csvwith_inferred_schema};
use ndarray::Array2;
use std::path::Path;

const LIDAR_CSV: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/bench/datasets/lidar.csv");

/// #2395: K random train/test partitions per arm. The lidar smooth fit is
/// sub-second, so 2*K=20 fits stay far inside the fast envelope while cutting the
/// held-out metric's standard error ~sqrt(K)=3.2x.
const K_SPLITS: usize = 10;
/// Held-out fraction per partition (~80/20, matching the former split scale).
const HOLDOUT: f64 = 0.20;

/// Fit gam (`gam_formula`) + mgcv (`mgcv_formula`) on `K_SPLITS` random partitions
/// of the lidar data and return the PAIRED per-split RMSE panel, gam's averaged
/// held-out R^2, and a representative single-split edf (split 0). `seed_base`
/// offsets the partition stream so the two arms average over INDEPENDENT
/// partition families. mgcv scores the SAME K partitions: the per-row 0/1 masks
/// are shipped as `fold0..fold{K-1}` columns and R loops over them, one
/// subprocess. Split `k` is therefore the same split for both tools, and the
/// panel keeps that pairing instead of collapsing each arm to its own mean.
fn gs_kfold_lidar(
    gam_formula: &str,
    mgcv_formula: &str,
    seed_base: usize,
) -> (PairedFoldComparison, f64, f64) {
    let ds = load_csvwith_inferred_schema(Path::new(LIDAR_CSV)).expect("load lidar.csv");
    let col = ds.column_map();
    let range_idx = col["range"];
    let logratio_idx = col["logratio"];
    let range: Vec<f64> = ds.values.column(range_idx).to_vec();
    let logratio: Vec<f64> = ds.values.column(logratio_idx).to_vec();
    let n = range.len();
    assert!(n > 100, "lidar should have ~221 rows, got {n}");
    let p = ds.headers.len();

    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };

    let mut gam_rmses = Vec::with_capacity(K_SPLITS);
    let mut gam_r2s = Vec::with_capacity(K_SPLITS);
    let mut fold_data: Vec<Vec<f64>> = Vec::with_capacity(K_SPLITS);
    let mut fold_names: Vec<String> = Vec::with_capacity(K_SPLITS);
    let mut gam_edf_repr = f64::NAN;

    for k in 0..K_SPLITS {
        let split_key = seed_base + k;
        let partition = paired_holdout_partition(n, HOLDOUT, split_key as u64);
        let train_rows = &partition.train;
        let test_rows = &partition.test;
        let test_range: Vec<f64> = test_rows.iter().map(|&i| range[i]).collect();
        let test_logratio: Vec<f64> = test_rows.iter().map(|&i| logratio[i]).collect();

        let mut train_values = Array2::<f64>::zeros((train_rows.len(), p));
        for (out_row, &src_row) in train_rows.iter().enumerate() {
            for c in 0..p {
                train_values[[out_row, c]] = ds.values[[src_row, c]];
            }
        }
        let mut train_ds = ds.clone();
        train_ds.values = train_values;

        let result = fit_from_formula(gam_formula, &train_ds, &cfg).expect("gam fit");
        let FitResult::Standard(fit) = result else {
            panic!("expected a standard GAM fit");
        };
        if k == 0 {
            gam_edf_repr = fit.fit.edf_total().expect("gam reports total edf");
        }

        let mut test_grid = Array2::<f64>::zeros((test_rows.len(), p));
        for (i, &r) in test_range.iter().enumerate() {
            test_grid[[i, range_idx]] = r;
        }
        let test_design = build_term_collection_design(test_grid.view(), &fit.resolvedspec)
            .expect("rebuild design at held-out points");
        let gam_test_pred: Vec<f64> = test_design.design.apply(&fit.fit.beta).to_vec();
        gam_rmses.push(rmse(&gam_test_pred, &test_logratio));
        gam_r2s.push(r2(&gam_test_pred, &test_logratio));

        fold_data.push(partition.mask);
        fold_names.push(format!("fold{k}"));
    }

    let mut columns: Vec<Column> = vec![
        Column::new("range", &range),
        Column::new("logratio", &logratio),
    ];
    for (name, data) in fold_names.iter().zip(fold_data.iter()) {
        columns.push(Column::new(name, data));
    }
    let r = run_r(
        &columns,
        &format!(
            r#"
            suppressPackageStartupMessages(library(mgcv))
            K <- {K_SPLITS}
            rmses <- numeric(K)
            for (k in 0:(K - 1)) {{
              fold <- df[[paste0("fold", k)]]
              tr <- data.frame(range = df$range[fold < 0.5], logratio = df$logratio[fold < 0.5])
              te <- data.frame(range = df$range[fold >= 0.5])
              obs <- df$logratio[fold >= 0.5]
              m <- gam({mgcv_formula}, data = tr, method = "REML")
              p <- as.numeric(predict(m, newdata = te))
              rmses[k + 1] <- sqrt(mean((p - obs)^2))
            }}
            emit("mgcv_rmses", rmses)
            "#
        ),
    );
    let mgcv_rmses = r.vector("mgcv_rmses");
    assert_eq!(
        mgcv_rmses.len(),
        K_SPLITS,
        "mgcv per-split rmse count mismatch"
    );

    let gam_r2 = gam_r2s.iter().sum::<f64>() / gam_r2s.len() as f64;
    (
        PairedFoldComparison::new(&gam_rmses, mgcv_rmses, true),
        gam_r2,
        gam_edf_repr,
    )
}

#[test]
fn gam_smooth_predicts_lidar_better_than_baseline() {
    init_parallelism();

    let (panel, gam_r2, gam_edf) = gs_kfold_lidar("logratio ~ s(range)", "logratio ~ s(range)", 0);
    eprintln!(
        "lidar s(range) #2395 K={K_SPLITS}-split paired (seed base 0): \
         gam_edf(split0)={gam_edf:.3} gam_test_R2_avg={gam_r2:.4}"
    );
    eprintln!("{}", panel.report("gaussian_smooth::default_basis"));
    eprintln!(
        "{}",
        QualityPair::paired(
            "families",
            "quality_vs_mgcv_gaussian_smooth::default_basis",
            "test_rmse",
            "mgcv",
            &panel,
        )
        .line()
    );

    assert!(
        gam_r2 >= 0.55,
        "gam's averaged held-out predictive R2 too low: {gam_r2:.4} (< 0.55)"
    );
    assert_paired_match_or_beat("gaussian_smooth::default_basis", &panel, 1.10);
    assert!(
        gam_edf > 1.0 && gam_edf < 30.0,
        "gam effective dof out of sane range: edf(split0)={gam_edf:.3}"
    );
}

/// Real-data arm for the SAME Gaussian smooth capability using gam's explicit
/// P-spline basis `s(range, bs="ps")`, over an INDEPENDENT family of K random
/// partitions (offset partition stream) so the two arms corroborate rather than
/// duplicate.
///
/// Dataset SOURCE: `bench/datasets/lidar.csv` — the classic LIDAR scatterplot
/// distributed with R's `SemiPar` package (Ruppert, Wand & Carroll,
/// *Semiparametric Regression*, 2003). Real measurements; objective quality is
/// held-out predictive accuracy.
#[test]
fn gam_smooth_predicts_lidar_better_than_baseline_on_real_data() {
    init_parallelism();

    let (panel, gam_r2, gam_edf) = gs_kfold_lidar(
        "logratio ~ s(range, bs=\"ps\")",
        "logratio ~ s(range, bs = \"ps\")",
        1000,
    );
    eprintln!(
        "lidar s(range,bs=ps) #2395 K={K_SPLITS}-split paired (seed base 1000): \
         gam_edf(split0)={gam_edf:.3} gam_test_R2_avg={gam_r2:.4}"
    );
    eprintln!("{}", panel.report("gaussian_smooth::ps_basis"));
    eprintln!(
        "{}",
        QualityPair::paired(
            "families",
            "quality_vs_mgcv_gaussian_smooth::ps_basis",
            "test_rmse",
            "mgcv",
            &panel,
        )
        .line()
    );

    assert!(
        gam_r2 >= 0.55,
        "gam's averaged held-out predictive R2 too low: {gam_r2:.4} (< 0.55)"
    );
    assert_paired_match_or_beat("gaussian_smooth::ps_basis", &panel, 1.10);
    assert!(
        gam_edf > 1.0 && gam_edf < 30.0,
        "gam effective dof out of sane range: edf(split0)={gam_edf:.3}"
    );
}
