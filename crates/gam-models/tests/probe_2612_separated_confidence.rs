//! #2612 fast harness: the under-confidence signature on a SYNTHETIC
//! quasi-separated multinomial, at a size that fits and predicts in seconds.
//!
//! The penguins acceptance arm is the issue's witness, but one fit plus its
//! posterior integration costs minutes there, which is the wrong instrument for
//! deciding between candidate repairs. This constructs the same geometry
//! deliberately and minimally: three classes stacked along one covariate with a
//! narrow overlap band, so the class boundaries are (quasi-)separating and the
//! softmax Fisher weight `W = diag(p) − ppᵀ` collapses on almost every row —
//! the exact regime in which no smoothing parameter can bound a penalty-null
//! direction (`S v = 0` ⇒ `(H + S_λ)v = Hv → 0`).
//!
//! Reported reference-free, on held-out rows drawn from the same law:
//!   * accuracy, under both published estimands;
//!   * mean argmax probability, whose gap to accuracy IS the calibration error;
//!   * the posterior standard deviation of the linear predictor, which is the
//!     quantity that converts into that gap through softmax's concavity.
//!
//! Prints only. The acceptance bar this measurement is meant to support lives
//! with the fix, not with the diagnostic.

use csv::StringRecord;
use gam_data::encode_recordswith_inferred_schema;
use gam_models::fit_orchestration::FitConfig;
use gam_models::multinomial::{
    MultinomialFitRequest, fit_penalized_multinomial_formula, predict_multinomial_formula,
    predict_multinomial_formula_plugin,
};

const K: usize = 3;

/// Overlap half-width of the two class boundaries, in covariate units. Zero is
/// COMPLETE separation (the likelihood has no finite maximiser at all); this
/// leaves a thin band in which both neighbouring classes appear, which is
/// quasi-separation — a finite mode with vanishing curvature around it, the
/// regime the penguins arm is in.
const OVERLAP: f64 = 0.06;

/// Deterministic low-discrepancy covariate: the van der Corput sequence in base
/// 2 mapped to `[-1, 1]`. No RNG, so the fixture is bit-reproducible and every
/// re-measurement is of the code and not of a draw.
fn covariate(index: usize) -> f64 {
    let mut numerator = 0.0_f64;
    let mut denominator = 1.0_f64;
    let mut n = index + 1;
    while n > 0 {
        denominator *= 2.0;
        numerator += ((n % 2) as f64) / denominator;
        n /= 2;
    }
    2.0 * numerator - 1.0
}

/// Class label: three bands along `x`, with the two boundaries perturbed inside
/// `OVERLAP` so a handful of rows fall on the wrong side of their own boundary.
fn label(index: usize, x: f64) -> usize {
    let wobble = if index % 7 == 0 { OVERLAP } else { -OVERLAP };
    let boundary_low = -1.0 / 3.0 + wobble;
    let boundary_high = 1.0 / 3.0 + wobble;
    if x < boundary_low {
        0
    } else if x < boundary_high {
        1
    } else {
        2
    }
}

fn mean_log_loss(probs: &[f64], labels: &[usize]) -> f64 {
    let mut acc = 0.0;
    for (i, &y) in labels.iter().enumerate() {
        acc -= probs[i * K + y].clamp(1e-15, 1.0).ln();
    }
    acc / labels.len() as f64
}

fn accuracy(probs: &[f64], labels: &[usize]) -> f64 {
    let mut correct = 0usize;
    for (i, &y) in labels.iter().enumerate() {
        let row = &probs[i * K..(i + 1) * K];
        let mut best = 0usize;
        for c in 1..K {
            if row[c] > row[best] {
                best = c;
            }
        }
        if best == y {
            correct += 1;
        }
    }
    correct as f64 / labels.len() as f64
}

fn mean_argmax_probability(probs: &[f64]) -> f64 {
    let rows = probs.len() / K;
    let mut acc = 0.0;
    for i in 0..rows {
        acc += probs[i * K..(i + 1) * K]
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
    }
    acc / rows as f64
}

fn dataset(indices: &[usize]) -> (gam_data::EncodedDataset, Vec<usize>) {
    let headers: Vec<String> = ["x", "cls"].iter().map(|s| s.to_string()).collect();
    let mut labels = Vec::with_capacity(indices.len());
    let records: Vec<StringRecord> = indices
        .iter()
        .map(|&i| {
            let x = covariate(i);
            let y = label(i, x);
            labels.push(y);
            StringRecord::from(vec![x.to_string(), format!("class{y}")])
        })
        .collect();
    let encoded = encode_recordswith_inferred_schema(headers, records)
        .expect("encode separated multinomial dataset");
    (encoded, labels)
}

#[test]
fn zz_probe_2612_separated_multinomial_is_under_confident() {
    faer::set_global_parallelism(faer::Par::rayon(0));

    let train_indices: Vec<usize> = (0..180).collect();
    let test_indices: Vec<usize> = (180..300).collect();
    let (train_ds, _) = dataset(&train_indices);
    let (test_ds, test_labels_raw) = dataset(&test_indices);

    let cfg = FitConfig::default();
    let model = fit_penalized_multinomial_formula(&MultinomialFitRequest {
        data: &train_ds,
        formula: "cls ~ s(x, k=8)",
        config: &cfg,
        init_lambda: 1.0,
        max_iter: 100,
        tol: 1e-8,
    })
    .expect("separated multinomial fit");

    // Realign the raw labels onto the fitted class order.
    let levels = model.class_levels.clone();
    let test_labels: Vec<usize> = test_labels_raw
        .iter()
        .map(|&y| {
            levels
                .iter()
                .position(|c| *c == format!("class{y}"))
                .expect("class level present")
        })
        .collect();

    let posterior = predict_multinomial_formula(&model, &test_ds).expect("posterior-mean predict");
    let plugin = predict_multinomial_formula_plugin(&model, &test_ds).expect("plug-in predict");
    let flat = |m: &ndarray::Array2<f64>| -> Vec<f64> {
        let mut v = Vec::with_capacity(m.nrows() * K);
        for i in 0..m.nrows() {
            for c in 0..K {
                v.push(m[[i, c]]);
            }
        }
        v
    };
    let post_flat = flat(&posterior);
    let plug_flat = flat(&plugin);

    eprintln!(
        "#2612 separated probe: n_train={} n_test={} overlap={OVERLAP}\n\
         #2612 separated probe: posterior-mean acc={:.4} logloss={:.5} mean_argmax_p={:.5} \
         calibration_gap={:.5}\n\
         #2612 separated probe: plug-in       acc={:.4} logloss={:.5} mean_argmax_p={:.5} \
         calibration_gap={:.5}",
        train_indices.len(),
        test_indices.len(),
        accuracy(&post_flat, &test_labels),
        mean_log_loss(&post_flat, &test_labels),
        mean_argmax_probability(&post_flat),
        accuracy(&post_flat, &test_labels) - mean_argmax_probability(&post_flat),
        accuracy(&plug_flat, &test_labels),
        mean_log_loss(&plug_flat, &test_labels),
        mean_argmax_probability(&plug_flat),
        accuracy(&plug_flat, &test_labels) - mean_argmax_probability(&plug_flat),
    );

    let covariance = model.coefficient_covariance().expect("joint covariance");
    let dim = covariance.nrows();
    let widest = (0..dim)
        .map(|i| covariance[[i, i]])
        .fold(f64::NEG_INFINITY, f64::max);
    let trace: f64 = (0..dim).map(|i| covariance[[i, i]]).sum();
    eprintln!(
        "#2612 separated probe: coefficient posterior dim={dim} widest_marginal_sd={:.4e} \
         trace={:.4e}",
        widest.max(0.0).sqrt(),
        trace,
    );
    let lambdas: Vec<String> = model.lambdas.iter().map(|v| format!("{v:.3e}")).collect();
    eprintln!(
        "#2612 separated probe: lambdas=[{}] edf_per_class={:?}",
        lambdas.join(", "),
        model.edf_per_class,
    );
}
