//! #2615 probe: what the effective-df floor actually decides on penguins.
//!
//! Runs the SAME gam fit as
//! `tests/quality/misc/quality_vs_nnet_multinom_penguins_species.rs` (both
//! strides), with NO R, and prints the held-out scores plus every selected λ.
//! The nnet reference numbers are the ones recorded on #2612/#2615, so the
//! log-loss margin can be read without R installed.
//!
//!   cargo run --release --example probe_2615_edf_floor

use csv::StringRecord;
use gam::families::multinomial::{
    MultinomialFitRequest, fit_penalized_multinomial_formula, predict_multinomial_formula,
    predict_multinomial_formula_plugin,
};
use gam::{FitConfig, encode_recordswith_inferred_schema, init_parallelism};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::time::Instant;

const PENGUINS_CSV: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/bench/datasets/penguins.csv");
const K: usize = 3;

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
    let idx = |name: &str| cols.iter().position(|c| *c == name).expect("column");
    let (i_species, i_bl, i_bd, i_fl, i_bm) = (
        idx("species"),
        idx("bill_length_mm"),
        idx("bill_depth_mm"),
        idx("flipper_length_mm"),
        idx("body_mass_g"),
    );
    let mut rows = Vec::new();
    for line in lines {
        let line = line.expect("row");
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split(',').collect();
        let num = |i: usize| f.get(i).and_then(|v| v.trim().parse::<f64>().ok());
        let (Some(bl), Some(bd), Some(fl), Some(bm)) =
            (num(i_bl), num(i_bd), num(i_fl), num(i_bm))
        else {
            continue;
        };
        let species = f[i_species].trim().trim_matches('"').to_string();
        if species.is_empty() || species == "NA" {
            continue;
        }
        rows.push(Penguin {
            bill_length: bl,
            bill_depth: bd,
            flipper: fl,
            body_mass: bm,
            species,
        });
    }
    rows
}

fn accuracy(probs: &[f64], labels: &[usize]) -> f64 {
    let mut hit = 0usize;
    for (i, &y) in labels.iter().enumerate() {
        let row = &probs[i * K..(i + 1) * K];
        let mut best = 0usize;
        for c in 1..K {
            if row[c] > row[best] {
                best = c;
            }
        }
        if best == y {
            hit += 1;
        }
    }
    hit as f64 / labels.len() as f64
}

fn log_loss(probs: &[f64], labels: &[usize]) -> f64 {
    let mut s = 0.0;
    for (i, &y) in labels.iter().enumerate() {
        s -= probs[i * K + y].max(1e-15).ln();
    }
    s / labels.len() as f64
}

fn per_class_recall(probs: &[f64], labels: &[usize]) -> Vec<f64> {
    let mut hit = vec![0usize; K];
    let mut tot = vec![0usize; K];
    for (i, &y) in labels.iter().enumerate() {
        let row = &probs[i * K..(i + 1) * K];
        let mut best = 0usize;
        for c in 1..K {
            if row[c] > row[best] {
                best = c;
            }
        }
        tot[y] += 1;
        if best == y {
            hit[y] += 1;
        }
    }
    (0..K)
        .map(|c| if tot[c] == 0 { f64::NAN } else { hit[c] as f64 / tot[c] as f64 })
        .collect()
}

fn run_stride(rows: &[Penguin], stride: usize, nnet_logloss: f64, nnet_acc: f64) {
    let mut train_idx = Vec::new();
    let mut test_idx = Vec::new();
    for i in 0..rows.len() {
        if i % stride == 0 {
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
    let make = |idxs: &[usize]| -> Vec<StringRecord> {
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
    let train_ds =
        encode_recordswith_inferred_schema(headers.clone(), make(&train_idx)).expect("train ds");
    let test_ds =
        encode_recordswith_inferred_schema(headers.clone(), make(&test_idx)).expect("test ds");

    let cfg = FitConfig::default();
    let t0 = Instant::now();
    let model = fit_penalized_multinomial_formula(&MultinomialFitRequest {
        data: &train_ds,
        formula: "species ~ s(bill_length_mm, k=10) + s(bill_depth_mm, k=10) + s(flipper_length_mm, k=10) + s(body_mass_g, k=10)",
        config: &cfg,
        init_lambda: 1.0,
        max_iter: 100,
        tol: 1e-8,
    })
    .expect("gam penguin multinomial fit");
    let secs = t0.elapsed().as_secs_f64();

    let levels = model.class_levels.clone();
    let labels: Vec<usize> = test_idx
        .iter()
        .map(|&i| levels.iter().position(|c| *c == rows[i].species).expect("level"))
        .collect();

    let mat = predict_multinomial_formula(&model, &test_ds).expect("predict");
    let mut flat = Vec::with_capacity(labels.len() * K);
    for i in 0..labels.len() {
        for c in 0..K {
            flat.push(mat[[i, c]]);
        }
    }
    let plug = predict_multinomial_formula_plugin(&model, &test_ds).expect("plugin");
    let mut flat_plug = Vec::with_capacity(labels.len() * K);
    for i in 0..labels.len() {
        for c in 0..K {
            flat_plug.push(plug[[i, c]]);
        }
    }

    // In-sample replay: discriminates a dead FIT from a dead PREDICT path.
    let train_labels: Vec<usize> = train_idx
        .iter()
        .map(|&i| levels.iter().position(|c| *c == rows[i].species).expect("level"))
        .collect();
    let tr_mat = predict_multinomial_formula(&model, &train_ds).expect("predict train");
    let mut tr_flat = Vec::with_capacity(train_labels.len() * K);
    for i in 0..train_labels.len() {
        for c in 0..K {
            tr_flat.push(tr_mat[[i, c]]);
        }
    }
    let tr_acc = accuracy(&tr_flat, &train_labels);
    let tr_ll = log_loss(&tr_flat, &train_labels);
    let beta_absmax = model
        .coefficients_flat
        .iter()
        .fold(0.0_f64, |a, b| a.max(b.abs()));
    let beta_l2 = model
        .coefficients_flat
        .iter()
        .map(|b| b * b)
        .sum::<f64>()
        .sqrt();
    println!(
        "  TRAIN: acc={tr_acc:.4} logloss={tr_ll:.5} |beta|_inf={beta_absmax:.4e} \
         |beta|_2={beta_l2:.4e} deviance={:.4} iters={}",
        model.deviance, model.iterations
    );

    let acc = accuracy(&flat, &labels);
    let ll = log_loss(&flat, &labels);
    let acc_p = accuracy(&flat_plug, &labels);
    let ll_p = log_loss(&flat_plug, &labels);
    let recall = per_class_recall(&flat, &labels);
    let rho: Vec<f64> = model.lambdas.iter().map(|l| l.ln()).collect();

    println!(
        "[#2615] stride={stride} n_train={} n_test={} fit={secs:.1}s\n  \
         acc={acc:.4} (nnet {nnet_acc:.4})  logloss={ll:.5} (nnet {nnet_logloss:.5}, bar {:.5})  {}\n  \
         plugin: acc={acc_p:.4} logloss={ll_p:.5}\n  \
         recall={recall:?}\n  \
         edf_per_class={:?}\n  \
         rho=ln(lambda)={:?}",
        train_idx.len(),
        labels.len(),
        nnet_logloss + 0.05,
        if ll <= nnet_logloss + 0.05 { "PASS" } else { "FAIL" },
        model.edf_per_class,
        rho.iter().map(|r| (r * 1000.0).round() / 1000.0).collect::<Vec<_>>(),
    );
}

/// Minimal stderr logger: the crate depends on `log` but ships no logger, and
/// `env_logger` is not a dependency of this crate. Only the diagnostics this
/// probe is about are let through, so the output stays readable.
struct ProbeLogger;

impl log::Log for ProbeLogger {
    fn enabled(&self, meta: &log::Metadata<'_>) -> bool {
        meta.level() <= log::Level::Trace
    }
    fn log(&self, record: &log::Record<'_>) {
        let msg = format!("{}", record.args());
        if record.level() <= log::Level::Info
            || msg.starts_with("[EDF-FLOOR]")
            || msg.starts_with("[#2615]")
            || msg.starts_with("[WARM-KEY]")
            || msg.starts_with("[OUTER-EVAL]")
            || msg.starts_with("[LABELED-EVAL]")
            || msg.starts_with("[UNIFIED-GRAD]")
            || msg.starts_with("[RHO-GRAD]")
        {
            eprintln!("{}: {msg}", record.level());
        }
    }
    fn flush(&self) {}
}

static PROBE_LOGGER: ProbeLogger = ProbeLogger;

fn main() {
    if log::set_logger(&PROBE_LOGGER).is_ok() {
        log::set_max_level(log::LevelFilter::Trace);
    }
    init_parallelism();
    let rows = load_penguins();
    println!("[#2615] penguins rows={}", rows.len());
    // nnet reference numbers recorded on #2612 / #2615 (no R in this container).
    run_stride(&rows, 3, 0.09494, 0.9912);
    run_stride(&rows, 4, 0.76930, f64::NAN);
}
