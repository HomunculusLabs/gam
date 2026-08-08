//! #2612 diagnostic probe (prints only, asserts nothing beyond "the fit exists").
//!
//! Fits the EXACT stride-3 penguins train split the failing quality arm uses and
//! reports the three quantities the issue's own analysis narrowed itself down to:
//!
//!   * the published held-out log-loss under BOTH estimands — the posterior mean
//!     `E[softmax(η)]` gam publishes and the plug-in `softmax(η̂)` at its own mode
//!     — so the posterior-width cost is a single subtraction;
//!   * the Laplace posterior's coefficient-space spectrum (widest standard
//!     deviations), which is where the width lives;
//!   * whether the fit armed the Jeffreys/Firth proper prior at all (via the
//!     `log::info` line the formula path emits at the criterion switch).
//!
//! Named `probe_*` so it is never mistaken for an acceptance bar.

use csv::StringRecord;
use gam_data::encode_recordswith_inferred_schema;
use gam_models::fit_orchestration::FitConfig;
use gam_models::multinomial::{
    MultinomialFitRequest, fit_penalized_multinomial_formula, predict_multinomial_formula,
    predict_multinomial_formula_plugin, predict_multinomial_formula_with_se,
};
use gam_linalg::faer_ndarray::FaerEigh;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

const PENGUINS_CSV: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../bench/datasets/penguins.csv"
);
const K: usize = 3;
const STRIDE: usize = 3;

struct StderrLogger;

impl log::Log for StderrLogger {
    fn enabled(&self, metadata: &log::Metadata<'_>) -> bool {
        metadata.level() <= log::Level::Info
    }
    fn log(&self, record: &log::Record<'_>) {
        if self.enabled(record.metadata()) {
            eprintln!("[{}] {}", record.level(), record.args());
        }
    }
    fn flush(&self) {}
}

static STDERR_LOGGER: StderrLogger = StderrLogger;

struct Penguin {
    bill_length: f64,
    bill_depth: f64,
    flipper: f64,
    body_mass: f64,
    species: String,
}

fn load_penguins() -> Vec<Penguin> {
    let file = File::open(Path::new(PENGUINS_CSV)).expect("open penguins.csv");
    let mut lines = BufReader::new(file).lines();
    let header = lines.next().expect("header").expect("read header");
    let cols: Vec<&str> = header.trim().split(',').collect();
    let idx = |name: &str| {
        cols.iter()
            .position(|c| *c == name)
            .unwrap_or_else(|| panic!("penguins.csv missing column {name}"))
    };
    let (i_species, i_bl, i_bd, i_fl, i_bm) = (
        idx("species"),
        idx("bill_length_mm"),
        idx("bill_depth_mm"),
        idx("flipper_length_mm"),
        idx("body_mass_g"),
    );
    let mut rows = Vec::new();
    for line in lines {
        let line = line.expect("read row");
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split(',').collect();
        if [i_bl, i_bd, i_fl, i_bm]
            .iter()
            .any(|&c| f[c] == "NA" || f[c].is_empty())
        {
            continue;
        }
        rows.push(Penguin {
            bill_length: f[i_bl].parse().expect("bill_length"),
            bill_depth: f[i_bd].parse().expect("bill_depth"),
            flipper: f[i_fl].parse().expect("flipper"),
            body_mass: f[i_bm].parse().expect("body_mass"),
            species: f[i_species].to_string(),
        });
    }
    rows
}

fn mean_log_loss(probs: &[f64], labels: &[usize]) -> f64 {
    let m = labels.len();
    let mut acc = 0.0;
    for (i, &y) in labels.iter().enumerate() {
        acc -= probs[i * K + y].clamp(1e-15, 1.0).ln();
    }
    acc / m as f64
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

/// Mean probability the estimand assigns to each row's OWN argmax class.
///
/// Against `accuracy` on the same rows this is a calibration statement that
/// needs no reference implementation: a calibrated classifier's mean argmax
/// probability equals its accuracy, an over-confident one exceeds it, and an
/// under-confident one falls short. It is the only tool-free way to say that
/// the published penguin probabilities are wrong rather than merely different
/// from `nnet`'s, so it belongs beside the log-loss columns.
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

#[test]
fn zz_probe_2612_penguins_posterior_width() {
    if log::set_logger(&STDERR_LOGGER).is_ok() {
        log::set_max_level(log::LevelFilter::Info);
    }
    faer::set_global_parallelism(faer::Par::rayon(0));

    let rows = load_penguins();
    let mut train_idx = Vec::new();
    let mut test_idx = Vec::new();
    for i in 0..rows.len() {
        if i % STRIDE == 0 {
            test_idx.push(i);
        } else {
            train_idx.push(i);
        }
    }

    let headers: Vec<String> = [
        "bill_length_mm",
        "bill_depth_mm",
        "flipper_length_mm",
        "body_mass_g",
        "species",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    let make_records = |idxs: &[usize]| -> Vec<StringRecord> {
        idxs.iter()
            .map(|&i| {
                let p = &rows[i];
                StringRecord::from(vec![
                    p.bill_length.to_string(),
                    p.bill_depth.to_string(),
                    p.flipper.to_string(),
                    p.body_mass.to_string(),
                    p.species.clone(),
                ])
            })
            .collect()
    };
    let train_ds = encode_recordswith_inferred_schema(headers.clone(), make_records(&train_idx))
        .expect("encode train");
    let test_ds =
        encode_recordswith_inferred_schema(headers.clone(), make_records(&test_idx)).expect("encode test");

    let cfg = FitConfig::default();
    let model = fit_penalized_multinomial_formula(&MultinomialFitRequest {
        data: &train_ds,
        formula: "species ~ s(bill_length_mm, k=10) + s(bill_depth_mm, k=10) + s(flipper_length_mm, k=10) + s(body_mass_g, k=10)",
        config: &cfg,
        init_lambda: 1.0,
        max_iter: 100,
        tol: 1e-8,
    })
    .expect("penguins fit");

    let class_levels = model.class_levels.clone();
    let labels: Vec<usize> = test_idx
        .iter()
        .map(|&i| {
            class_levels
                .iter()
                .position(|c| *c == rows[i].species)
                .expect("species in levels")
        })
        .collect();

    let post = predict_multinomial_formula(&model, &test_ds).expect("posterior predict");
    let plug = predict_multinomial_formula_plugin(&model, &test_ds).expect("plug-in predict");
    let flat = |m: &ndarray::Array2<f64>| -> Vec<f64> {
        let mut v = Vec::with_capacity(m.nrows() * K);
        for i in 0..m.nrows() {
            for c in 0..K {
                v.push(m[[i, c]]);
            }
        }
        v
    };
    let post_flat = flat(&post);
    let plug_flat = flat(&plug);

    eprintln!(
        "#2612 probe: n_train={} n_test={} K={K}\n\
         #2612 probe: posterior-mean  acc={:.4}  logloss={:.5}\n\
         #2612 probe: plug-in         acc={:.4}  logloss={:.5}\n\
         #2612 probe: width cost = {:.5} nats (nnet reference logloss = 0.09494)",
        train_idx.len(),
        test_idx.len(),
        accuracy(&post_flat, &labels),
        mean_log_loss(&post_flat, &labels),
        accuracy(&plug_flat, &labels),
        mean_log_loss(&plug_flat, &labels),
        mean_log_loss(&post_flat, &labels) - mean_log_loss(&plug_flat, &labels),
    );

    // The tool-free calibration statement: mean argmax probability against
    // accuracy on the SAME held-out rows. Zero is calibrated; positive is
    // under-confident (the model is right more often than it claims to be).
    eprintln!(
        "#2612 probe: calibration (accuracy - mean argmax probability): \
         posterior-mean={:.5} (mean_p={:.5})  plug-in={:.5} (mean_p={:.5})",
        accuracy(&post_flat, &labels) - mean_argmax_probability(&post_flat),
        mean_argmax_probability(&post_flat),
        accuracy(&plug_flat, &labels) - mean_argmax_probability(&plug_flat),
        mean_argmax_probability(&plug_flat),
    );

    let cov = model.coefficient_covariance().expect("covariance");
    let (evals, evecs) = cov.eigh(faer::Side::Lower).expect("covariance eigh");
    let mut order: Vec<usize> = (0..evals.len()).collect();
    order.sort_by(|a, b| evals[*b].partial_cmp(&evals[*a]).expect("finite"));
    let sds: Vec<f64> = order
        .iter()
        .map(|&i| evals[i].max(0.0).sqrt())
        .collect();
    eprintln!(
        "#2612 probe: posterior sd spectrum (top 8 of {}): {:?}",
        sds.len(),
        sds.iter()
            .take(8)
            .map(|v| format!("{v:.4e}"))
            .collect::<Vec<_>>()
    );
    eprintln!(
        "#2612 probe: lambdas={:?}\n#2612 probe: edf_per_class={:?}\n#2612 probe: edf_per_penalty={:?}\n#2612 probe: lambda_labels={:?}",
        model
            .lambdas
            .iter()
            .map(|v| format!("{v:.3e}"))
            .collect::<Vec<_>>(),
        model.edf_per_class,
        model
            .edf_per_penalty
            .as_ref()
            .map(|v| v.iter().map(|e| format!("{e:.4e}")).collect::<Vec<_>>()),
        model.lambda_labels,
    );

    // Where the width comes from, on the three flattest posterior directions:
    // `H v = (1/sigma^2) v` for an eigenpair of `Sigma = H^-1`, and the saved
    // influence matrix `F = H^-1 X'WX` splits that curvature exactly into the
    // data's share `v'Fv` and the penalty's `1 - v'Fv` (both nonnegative, both
    // measured, no reconstruction of `S` needed).
    if let Some(influence) = model.coefficient_influence() {
        for rank in 0..order.len().min(3) {
            let index = order[rank];
            let v = evecs.column(index).to_owned();
            let curvature = 1.0 / evals[index].max(f64::MIN_POSITIVE);
            let data_share = v.dot(&influence.dot(&v));
            eprintln!(
                "#2612 probe: posterior direction #{rank}: sd={:.4e} curvature={:.6e} \
                 data_share(v'Fv)={:.6} data={:.6e} penalty={:.6e}",
                evals[index].max(0.0).sqrt(),
                curvature,
                data_share,
                curvature * data_share,
                curvature * (1.0 - data_share),
            );
        }
    }

    // The quantity that actually flattens the published probabilities: the
    // integrated posterior spread of each held-out row's class probabilities.
    let (_, probability_se) =
        predict_multinomial_formula_with_se(&model, &test_ds).expect("posterior se");
    let mut spreads: Vec<f64> = probability_se.iter().copied().collect();
    spreads.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
    let quantile = |q: f64| spreads[((spreads.len() - 1) as f64 * q).round() as usize];
    eprintln!(
        "#2612 probe: held-out probability posterior sd: min={:.4} q25={:.4} median={:.4} q75={:.4} max={:.4}",
        spreads[0],
        quantile(0.25),
        quantile(0.50),
        quantile(0.75),
        spreads[spreads.len() - 1],
    );

    // ── Is the Gaussian the posterior actually IS a Gaussian? ───────────────
    //
    // The published width is `Sigma = H^-1` with `H` the curvature of the
    // penalized log-posterior AT THE MODE. That is only the posterior's width if
    // the log-posterior is quadratic over the mass the Gaussian assigns. Walk the
    // widest direction out to `t` standard deviations of the Gaussian's own
    // claim and read the TRAINING log-likelihood there. The Gaussian's quadratic
    // model of the likelihood's own share of the drop is `0.5 * t^2 * share`,
    // `share = v'(H - S_lambda)v / v'Hv` (the saved `F`), so the ratio of the
    // measured drop to that number is a direct, assumption-free statement about
    // whether the reported width is the posterior's width. Nothing here is a
    // refit: the coefficients are overwritten in a clone and scored by the
    // plug-in predictor on the TRAIN rows, so this is the exact likelihood the
    // fit maximized.
    let train_labels: Vec<usize> = train_idx
        .iter()
        .map(|&i| {
            class_levels
                .iter()
                .position(|c| *c == rows[i].species)
                .expect("species in levels")
        })
        .collect();
    let train_log_likelihood = |coefficients: &[f64]| -> f64 {
        let mut walked = model.clone();
        walked.coefficients_flat = coefficients.to_vec();
        let probabilities =
            predict_multinomial_formula_plugin(&walked, &train_ds).expect("plug-in on train");
        let mut total = 0.0;
        for (i, &y) in train_labels.iter().enumerate() {
            total += probabilities[[i, y]].max(1e-300).ln();
        }
        total
    };
    let at_mode = train_log_likelihood(&model.coefficients_flat);
    let p_per_class = model.p_per_class;
    let n_active = model.n_active_classes;
    for rank in 0..order.len().min(2) {
        let index = order[rank];
        let sd = evals[index].max(0.0).sqrt();
        let v = evecs.column(index).to_owned();
        let likelihood_share = model
            .coefficient_influence()
            .map(|f| v.dot(&f.dot(&v)))
            .unwrap_or(1.0);
        for t in [0.25_f64, 0.5, 1.0, 2.0] {
            let mut walked = model.coefficients_flat.clone();
            for i in 0..p_per_class {
                for a in 0..n_active {
                    walked[i * n_active + a] += t * sd * v[a * p_per_class + i];
                }
            }
            let drop = at_mode - train_log_likelihood(&walked);
            let quadratic = 0.5 * t * t * likelihood_share;
            eprintln!(
                "#2612 probe: quadratic check dir #{rank} t={t}: measured loglik drop={drop:.4e} \
                 Gaussian predicts={quadratic:.4e} ratio={:.4e}",
                drop / quadratic.max(f64::MIN_POSITIVE),
            );
        }
    }
}
