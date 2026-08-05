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
    predict_multinomial_formula_plugin,
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

    let cov = model.coefficient_covariance().expect("covariance");
    let (evals, _) = cov.eigh(faer::Side::Lower).expect("covariance eigh");
    let mut sds: Vec<f64> = evals.iter().map(|v| v.max(0.0).sqrt()).collect();
    sds.sort_by(|a, b| b.partial_cmp(a).expect("finite"));
    eprintln!(
        "#2612 probe: posterior sd spectrum (top 8 of {}): {:?}",
        sds.len(),
        sds.iter().take(8).map(|v| format!("{v:.4e}")).collect::<Vec<_>>()
    );
    eprintln!(
        "#2612 probe: lambdas={:?}\n#2612 probe: edf_per_class={:?}",
        model
            .lambdas
            .iter()
            .map(|v| format!("{v:.3e}"))
            .collect::<Vec<_>>(),
        model.edf_per_class,
    );
}
