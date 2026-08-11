//! #2612 focused instrument: the penguins stride-3 witness ONLY, at `Info`, so
//! the inner joint-Newton trail for ONE fit is readable without paying for the
//! five other fixtures `probe_2612_whole_picture` runs.
//!
//! This exists because the remaining blocker on #2612 is a property of exactly
//! this fit: with the Jeffreys/Firth span narrowed to `ker(S_λ)` (the directions
//! no smoothing parameter reaches — which is where the term belongs, and which
//! measurably fixes the calibration on the banded fixture) the inner joint
//! Newton no longer certifies the near-separable multinomial mode to the
//! accuracy the ρ-only LAML derivative lane demands, and the fit disappears.
//! Reading that requires the per-cycle trust-region trail of this one solve.
//!
//! Asserts nothing. `zz_` prefixed so it can never be mistaken for a bar.

use csv::StringRecord;
use gam_data::encode_recordswith_inferred_schema;
use gam_models::fit_orchestration::FitConfig;
use gam_models::multinomial::{
    MultinomialFitRequest, MultinomialSavedModel, fit_penalized_multinomial_formula,
    predict_multinomial_formula, predict_multinomial_formula_plugin,
};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::time::Instant;

const PENGUINS_CSV: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../bench/datasets/penguins.csv"
);

const FORMULA: &str = "species ~ s(bill_length_mm, k=10) + s(bill_depth_mm, k=10) \
     + s(flipper_length_mm, k=10) + s(body_mass_g, k=10)";

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

fn init() {
    if log::set_logger(&STDERR_LOGGER).is_ok() {
        log::set_max_level(log::LevelFilter::Info);
    }
    faer::set_global_parallelism(faer::Par::rayon(0));
}

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
        let fields: Vec<&str> = line.split(',').collect();
        let parse = |i: usize| fields.get(i).and_then(|v| v.parse::<f64>().ok());
        let (Some(bl), Some(bd), Some(fl), Some(bm)) =
            (parse(i_bl), parse(i_bd), parse(i_fl), parse(i_bm))
        else {
            continue;
        };
        let species = fields[i_species].trim().to_string();
        if species.is_empty() {
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

fn penguins_split(
    stride: usize,
) -> (
    gam_data::EncodedDataset,
    gam_data::EncodedDataset,
    Vec<String>,
) {
    let rows = load_penguins();
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
    let make = |keep_test: bool| -> Vec<StringRecord> {
        rows.iter()
            .enumerate()
            .filter(|(index, _)| (index % stride == 0) == keep_test)
            .map(|(_, p)| {
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
    let test_species: Vec<String> = rows
        .iter()
        .enumerate()
        .filter(|(index, _)| index % stride == 0)
        .map(|(_, p)| p.species.clone())
        .collect();
    (
        encode_recordswith_inferred_schema(headers.clone(), make(false))
            .expect("encode penguins train"),
        encode_recordswith_inferred_schema(headers, make(true)).expect("encode penguins test"),
        test_species,
    )
}

fn mean_log_loss(probs: &ndarray::Array2<f64>, labels: &[usize]) -> f64 {
    let mut acc = 0.0;
    for (row, &y) in labels.iter().enumerate() {
        acc -= probs[[row, y]].clamp(1e-15, 1.0).ln();
    }
    acc / labels.len() as f64
}

fn accuracy(probs: &ndarray::Array2<f64>, labels: &[usize]) -> f64 {
    let mut correct = 0usize;
    for (row, &y) in labels.iter().enumerate() {
        let mut best = 0usize;
        for class in 1..probs.ncols() {
            if probs[[row, class]] > probs[[row, best]] {
                best = class;
            }
        }
        if best == y {
            correct += 1;
        }
    }
    correct as f64 / labels.len() as f64
}

fn mean_argmax_probability(probs: &ndarray::Array2<f64>) -> f64 {
    let mut acc = 0.0;
    for row in 0..probs.nrows() {
        let mut best = f64::NEG_INFINITY;
        for class in 0..probs.ncols() {
            best = best.max(probs[[row, class]]);
        }
        acc += best;
    }
    acc / probs.nrows() as f64
}

fn report(model: &MultinomialSavedModel, test: &gam_data::EncodedDataset, labels: &[usize]) {
    eprintln!(
        "#2612 [penguins stride-3] separation_evidence = {:?}",
        model.separation_evidence
    );
    let lambdas: Vec<String> = model.lambdas.iter().map(|v| format!("{v:.4e}")).collect();
    eprintln!(
        "#2612 [penguins stride-3] classes={:?} P={} lambdas=[{}] edf_per_class={:?}",
        model.class_levels,
        model.p_per_class,
        lambdas.join(", "),
        model.edf_per_class,
    );
    for (estimand, predicted) in [
        ("posterior-mean", predict_multinomial_formula(model, test)),
        (
            "plug-in",
            predict_multinomial_formula_plugin(model, test),
        ),
    ] {
        match predicted {
            Ok(probs) => {
                let acc = accuracy(&probs, labels);
                let argmax = mean_argmax_probability(&probs);
                eprintln!(
                    "#2612 [penguins stride-3] {estimand}: acc={acc:.4} logloss={:.5} \
                     mean_argmax_p={argmax:.4} calib_gap={:+.5}",
                    mean_log_loss(&probs, labels),
                    argmax - acc,
                );
            }
            Err(error) => eprintln!("#2612 [penguins stride-3] {estimand}: UNAVAILABLE: {error}"),
        }
    }
}

#[test]
fn zz_probe_2612_penguins_stride3_inner_trail() {
    init();
    let (train, test, species) = penguins_split(3);
    let config = FitConfig::default();
    let started = Instant::now();
    let fitted = fit_penalized_multinomial_formula(&MultinomialFitRequest {
        data: &train,
        formula: FORMULA,
        config: &config,
        init_lambda: 1.0,
        max_iter: 100,
        tol: 1e-8,
    });
    let elapsed = started.elapsed().as_secs_f64();
    match fitted {
        Ok(model) => {
            eprintln!("#2612 [penguins stride-3] FIT OK in {elapsed:.1}s");
            let labels: Vec<usize> = species
                .iter()
                .map(|name| {
                    model
                        .class_levels
                        .iter()
                        .position(|level| level == name)
                        .expect("species level present")
                })
                .collect();
            report(&model, &test, &labels);
        }
        Err(error) => {
            eprintln!("#2612 [penguins stride-3] FIT FAILED after {elapsed:.1}s: {error}");
        }
    }
}
