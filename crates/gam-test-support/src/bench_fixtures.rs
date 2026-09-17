//! Benchmark-harness preprocessing: cross-validation folds and per-fold z-scoring.
//!
//! The parity suite (`bench/run_suite.py`) splits every dataset into folds and
//! standardizes each fold's features before any contender sees them. Both are
//! harness choices, not model code, so they live here with the fixtures rather
//! than in the production extension. `bench/` reaches them through the
//! `bench_fixtures` binary, and a seed gives the fold layout the suite has always
//! produced.

use ndarray::{Array2, ArrayView2, Axis};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use std::collections::BTreeMap;

/// One split: ascending training rows, then ascending test rows.
pub type CvFold = (Vec<usize>, Vec<usize>);

/// The suite's cross-validation folds of `y`.
///
/// Rows are grouped into one bucket, or into one bucket per label value when
/// `stratified` (both IEEE zeros are one level; a non-finite label is refused).
/// Each bucket is shuffled with `StdRng::seed_from_u64(seed)`. With
/// `n_splits >= 2` a bucket is dealt round-robin, so every row lands in exactly
/// one test fold and strata are balanced across folds. With `n_splits == 1` it is
/// one holdout that sends `m / 5` rows of a bucket of size `m` to test, at least
/// one and at most `m - 1` once `m >= 2`: the share of one fold of the default
/// 5-fold split.
pub fn cv_folds(
    y: &[f64],
    n_splits: usize,
    seed: u64,
    stratified: bool,
) -> Result<Vec<CvFold>, String> {
    if n_splits == 0 {
        return Err("cv_folds: n_splits must be >= 1".to_string());
    }
    let n = y.len();
    if n == 0 {
        return Err("cv_folds: y must have at least one observation".to_string());
    }
    for (i, v) in y.iter().enumerate() {
        if !v.is_finite() {
            return Err(format!(
                "cv_folds: y[{i}] is not finite ({v}); CV labels must be finite"
            ));
        }
    }
    if n_splits >= 2 && n < n_splits {
        return Err(format!(
            "cv_folds: n_splits={n_splits} cannot exceed n_observations={n}"
        ));
    }

    let mut rng = StdRng::seed_from_u64(seed);

    let mut buckets: Vec<Vec<usize>> = if stratified {
        let label_key = |v: f64| {
            if v == 0.0 {
                0.0_f64.to_bits()
            } else {
                v.to_bits()
            }
        };
        let mut by_label: BTreeMap<u64, Vec<usize>> = BTreeMap::new();
        for (i, v) in y.iter().enumerate() {
            by_label.entry(label_key(*v)).or_default().push(i);
        }
        by_label.into_values().collect()
    } else {
        vec![(0..n).collect()]
    };

    if n_splits >= 2 {
        let mut fold_of_row = vec![usize::MAX; n];
        for bucket in buckets.iter_mut() {
            bucket.shuffle(&mut rng);
            for (k, &idx) in bucket.iter().enumerate() {
                fold_of_row[idx] = k % n_splits;
            }
        }
        let mut folds: Vec<CvFold> = (0..n_splits).map(|_| (Vec::new(), Vec::new())).collect();
        for (idx, &f) in fold_of_row.iter().enumerate() {
            for (k, fold) in folds.iter_mut().enumerate() {
                if k == f {
                    fold.1.push(idx);
                } else {
                    fold.0.push(idx);
                }
            }
        }
        for (k, (train, test)) in folds.iter().enumerate() {
            if test.is_empty() {
                return Err(format!(
                    "cv_folds: fold {k}/{n_splits} has empty test set \
                     (n={n}); reduce n_splits or supply more observations"
                ));
            }
            if train.is_empty() {
                return Err(format!(
                    "cv_folds: fold {k}/{n_splits} has empty train set \
                     (n={n}); reduce n_splits or supply more observations"
                ));
            }
        }
        Ok(folds)
    } else {
        const HOLDOUT_DENOMINATOR: usize = 5;
        let mut train: Vec<usize> = Vec::new();
        let mut test: Vec<usize> = Vec::new();
        for bucket in buckets.iter_mut() {
            bucket.shuffle(&mut rng);
            let m = bucket.len();
            let mut n_test = m / HOLDOUT_DENOMINATOR;
            if n_test == 0 && m >= 2 {
                n_test = 1;
            }
            if n_test >= m && m >= 2 {
                n_test = m - 1;
            }
            for (i, &idx) in bucket.iter().enumerate() {
                if i < n_test {
                    test.push(idx);
                } else {
                    train.push(idx);
                }
            }
        }
        if test.is_empty() {
            return Err(format!(
                "cv_folds: holdout split has empty test set (n={n}); \
                 supply at least 2 observations (and ≥2 per class when stratified)"
            ));
        }
        if train.is_empty() {
            return Err(format!(
                "cv_folds: holdout split has empty train set (n={n}); \
                 supply at least 2 observations (and ≥2 per class when stratified)"
            ));
        }
        train.sort_unstable();
        test.sort_unstable();
        Ok(vec![(train, test)])
    }
}

/// Standardize every feature column of `train` and `test` by `train`'s own mean
/// and population standard deviation (ddof 0, as scikit-learn's `StandardScaler`
/// and pandas). A column with no spread keeps divisor 1, so it is only centred.
///
/// The moments are one-pass Welford sums: the running mean stays inside the data's
/// range, so finite values near `f64::MAX` do not overflow the way `Σx / n` does.
/// Refuses a column-count mismatch, an empty `train`, a non-finite cell, and
/// moments that are not representable.
pub fn zscore_train_test(
    train: ArrayView2<'_, f64>,
    test: ArrayView2<'_, f64>,
) -> Result<(Array2<f64>, Array2<f64>), String> {
    let p = train.ncols();
    if test.ncols() != p {
        return Err(format!(
            "zscore_train_test: train has {p} columns but test has {}",
            test.ncols()
        ));
    }
    let n_train = train.nrows();
    if n_train == 0 {
        return Err(
            "zscore_train_test: train has zero rows; cannot estimate column statistics"
                .to_string(),
        );
    }
    for (label, frame) in [("train", train), ("test", test)] {
        for (j, col) in frame.axis_iter(Axis(1)).enumerate() {
            for (i, v) in col.iter().enumerate() {
                if !v.is_finite() {
                    return Err(format!(
                        "zscore_train_test: {label}[{i}, {j}] is not finite ({v})"
                    ));
                }
            }
        }
    }

    let mut train_out = Array2::<f64>::zeros(train.raw_dim());
    let mut test_out = Array2::<f64>::zeros(test.raw_dim());
    for j in 0..p {
        let mut mean = 0.0_f64;
        let mut m2 = 0.0_f64;
        for (k, v) in train.column(j).iter().enumerate() {
            let delta = v - mean;
            mean += delta / (k + 1) as f64;
            m2 += delta * (v - mean);
        }
        let std = (m2 / n_train as f64).sqrt();
        if !mean.is_finite() || !std.is_finite() {
            return Err(format!(
                "zscore_train_test: column {j} moments are not representable in f64 \
                 (mean={mean}, std={std}); rescale the inputs before standardizing"
            ));
        }
        let scale = if std > 0.0 { std } else { 1.0 };
        for i in 0..train.nrows() {
            train_out[[i, j]] = (train[[i, j]] - mean) / scale;
        }
        for i in 0..test.nrows() {
            test_out[[i, j]] = (test[[i, j]] - mean) / scale;
        }
    }
    Ok((train_out, test_out))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use std::collections::HashSet;

    #[test]
    fn kfold_puts_every_row_in_exactly_one_test_fold() {
        let n = 50usize;
        let y: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let folds = cv_folds(&y, 5, 7, false).expect("5-fold should succeed");
        assert_eq!(folds.len(), 5);
        let mut seen = vec![0usize; n];
        for (train, test) in &folds {
            assert!(!train.is_empty() && !test.is_empty());
            let train_set: HashSet<usize> = train.iter().copied().collect();
            for &i in test {
                assert!(!train_set.contains(&i), "row {i} is in train and test");
                seen[i] += 1;
            }
            assert_eq!(train.len() + test.len(), n);
        }
        assert!(seen.iter().all(|&count| count == 1), "seen={seen:?}");
    }

    #[test]
    fn stratified_kfold_balances_each_class_across_folds() {
        let mut y = vec![1.0; 30];
        y.extend(std::iter::repeat_n(0.0, 20));
        let folds = cv_folds(&y, 5, 11, true).expect("stratified 5-fold");
        assert_eq!(folds.len(), 5);
        for (_, test) in &folds {
            assert_eq!(test.iter().filter(|&&i| i < 30).count(), 6);
            assert_eq!(test.iter().filter(|&&i| i >= 30).count(), 4);
        }
    }

    #[test]
    fn holdout_sends_one_fifth_to_test() {
        let n = 25usize;
        let y: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let folds = cv_folds(&y, 1, 42, false).expect("holdout split");
        assert_eq!(folds.len(), 1);
        let (train, test) = &folds[0];
        assert_eq!(train.len() + test.len(), n);
        assert_eq!(test.len(), 5);
        let train_set: HashSet<usize> = train.iter().copied().collect();
        assert!(test.iter().all(|i| !train_set.contains(i)));
    }

    #[test]
    fn a_seed_reproduces_its_layout_and_another_seed_changes_it() {
        let y: Vec<f64> = (0..40).map(|i| i as f64).collect();
        let a = cv_folds(&y, 5, 17, false).expect("seed=17");
        let b = cv_folds(&y, 5, 17, false).expect("seed=17 (repeat)");
        let c = cv_folds(&y, 5, 18, false).expect("seed=18");
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn invalid_fold_requests_are_refused() {
        assert!(cv_folds(&[0.0, 1.0], 0, 0, false).is_err());
        assert!(cv_folds(&[], 5, 0, false).is_err());
        assert!(cv_folds(&[0.0, 1.0], 5, 0, false).is_err());
        assert!(cv_folds(&[0.0, f64::NAN, 1.0], 2, 0, false).is_err());
        assert!(cv_folds(&[0.0, f64::INFINITY, 1.0], 2, 0, false).is_err());
    }

    #[test]
    fn zscore_uses_the_training_moments_and_only_centres_a_constant_column() {
        let train = array![[1.0, 5.0], [3.0, 5.0]];
        let test = array![[5.0, 7.0]];
        let (train_out, test_out) =
            zscore_train_test(train.view(), test.view()).expect("standardize");
        assert_eq!(train_out, array![[-1.0, 0.0], [1.0, 0.0]]);
        assert_eq!(test_out, array![[3.0, 2.0]]);
        assert!(zscore_train_test(train.view(), array![[1.0]].view()).is_err());
        assert!(zscore_train_test(train.view(), array![[f64::NAN, 1.0]].view()).is_err());
    }
}
