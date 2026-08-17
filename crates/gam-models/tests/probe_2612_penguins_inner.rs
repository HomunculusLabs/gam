//! #2612 focused instrument: ONE fixture per test, at `Info`, so the inner
//! joint-Newton trail of a single solve is readable without paying for the five
//! other fixtures `probe_2612_whole_picture` runs.
//!
//! The remaining blocker on #2612 is a property of the inner solve, not of the
//! dataset: with the Jeffreys/Firth span narrowed to `ker(S_λ)` — where it
//! belongs, and which measurably fixes the calibration — the inner joint Newton
//! stops five to six orders above the ρ-only LAML derivative lane's `1e-11`
//! stationarity target, on a convex model, with `resolvable_negative_curvature
//! = false`. Both the banded fixture (`cls ~ s(x, k=8)`, ten seconds, sixteen
//! coefficients) and the penguins witness (seven minutes, seventy-four) plateau
//! at the same `~9e-7`. The banded arm is therefore the oracle and penguins is
//! the confirmation.
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

const PENGUINS_FORMULA: &str = "species ~ s(bill_length_mm, k=10) + s(bill_depth_mm, k=10) \
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

// ── the banded quasi-separated fixture (the ten-second oracle) ───────────────

const CLASS_NAMES: [&str; 3] = ["a", "b", "c"];

/// Byte-for-byte the fixture `multinomial_separation_arming_2612::banded_records`
/// builds, so this probe and that acceptance fit the SAME rows: the van der
/// Corput sequence in base 2 mapped onto `[-1, 1]`, three bands split at
/// `±1/3`, and a deterministic `±OVERLAP` wobble on every seventh row.
fn banded_records(indices: &[usize]) -> (Vec<StringRecord>, Vec<usize>) {
    const OVERLAP: f64 = 0.06;
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
    let mut labels = Vec::with_capacity(indices.len());
    let records = indices
        .iter()
        .map(|&index| {
            let x = covariate(index);
            let wobble = if index % 7 == 0 { OVERLAP } else { -OVERLAP };
            let class = if x < -1.0 / 3.0 + wobble {
                0
            } else if x < 1.0 / 3.0 + wobble {
                1
            } else {
                2
            };
            labels.push(class);
            StringRecord::from(vec![x.to_string(), CLASS_NAMES[class].to_string()])
        })
        .collect();
    (records, labels)
}

fn banded_frame(indices: &[usize]) -> (gam_data::EncodedDataset, Vec<usize>) {
    let (records, labels) = banded_records(indices);
    let headers: Vec<String> = ["x", "y"].iter().map(|s| s.to_string()).collect();
    (
        encode_recordswith_inferred_schema(headers, records).expect("encode banded frame"),
        labels,
    )
}

// ── penguins ────────────────────────────────────────────────────────────────

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

// ── scoring ─────────────────────────────────────────────────────────────────

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

fn report(
    label: &str,
    model: &MultinomialSavedModel,
    test: &gam_data::EncodedDataset,
    labels: &[usize],
) {
    eprintln!(
        "#2612 [{label}] separation_evidence = {:?}",
        model.separation_evidence
    );
    let lambdas: Vec<String> = model.lambdas.iter().map(|v| format!("{v:.4e}")).collect();
    eprintln!(
        "#2612 [{label}] classes={:?} P={} lambdas=[{}] edf_per_class={:?}",
        model.class_levels,
        model.p_per_class,
        lambdas.join(", "),
        model.edf_per_class,
    );
    for (estimand, predicted) in [
        ("posterior-mean", predict_multinomial_formula(model, test)),
        ("plug-in", predict_multinomial_formula_plugin(model, test)),
    ] {
        match predicted {
            Ok(probs) => {
                let acc = accuracy(&probs, labels);
                let argmax = mean_argmax_probability(&probs);
                eprintln!(
                    "#2612 [{label}] {estimand}: acc={acc:.4} logloss={:.5} \
                     mean_argmax_p={argmax:.4} calib_gap={:+.5}",
                    mean_log_loss(&probs, labels),
                    argmax - acc,
                );
            }
            Err(error) => eprintln!("#2612 [{label}] {estimand}: UNAVAILABLE: {error}"),
        }
    }
}

fn run(
    label: &str,
    train: &gam_data::EncodedDataset,
    formula: &str,
    test: &gam_data::EncodedDataset,
    label_of: impl Fn(&MultinomialSavedModel) -> Vec<usize>,
) {
    let config = FitConfig::default();
    let started = Instant::now();
    let fitted = fit_penalized_multinomial_formula(&MultinomialFitRequest {
        data: train,
        formula,
        config: &config,
        init_lambda: 1.0,
        max_iter: 100,
        tol: 1e-8,
    });
    let elapsed = started.elapsed().as_secs_f64();
    match fitted {
        Ok(model) => {
            eprintln!("#2612 [{label}] FIT OK in {elapsed:.1}s");
            let labels = label_of(&model);
            report(label, &model, test, &labels);
        }
        Err(error) => eprintln!("#2612 [{label}] FIT FAILED after {elapsed:.1}s: {error}"),
    }
}

#[test]
fn zz_probe_2612_banded_inner_trail() {
    init();
    let (train, _) = banded_frame(&(0..180).collect::<Vec<_>>());
    let (test, test_labels) = banded_frame(&(180..300).collect::<Vec<_>>());
    run(
        "B banded quasi-separated",
        &train,
        "y ~ s(x, k=8)",
        &test,
        |model| {
            test_labels
                .iter()
                .map(|&class| {
                    model
                        .class_levels
                        .iter()
                        .position(|level| level == CLASS_NAMES[class])
                        .expect("class level present")
                })
                .collect()
        },
    );
}

#[test]
fn zz_probe_2612_penguins_stride3_inner_trail() {
    init();
    let (train, test, species) = penguins_split(3);
    run(
        "C penguins stride-3",
        &train,
        PENGUINS_FORMULA,
        &test,
        |model| {
            species
                .iter()
                .map(|name| {
                    model
                        .class_levels
                        .iter()
                        .position(|level| level == name)
                        .expect("species level present")
                })
                .collect()
        },
    );
}

/// The stride-4 split, which is the one the FAILING acceptance arm
/// (`gam_multinomial_classifies_penguin_species_at_least_as_well_as_nnet`) uses.
/// Its sibling above is the stride-3 `_on_real_data` arm, which CI records as a
/// TIMEOUT rather than a FAIL — so the two arms need separate trails, because a
/// verdict that was never reached is not the same finding as one that was.
#[test]
fn zz_probe_2612_penguins_stride4_inner_trail() {
    init();
    let (train, test, species) = penguins_split(4);
    run(
        "D penguins stride-4",
        &train,
        PENGUINS_FORMULA,
        &test,
        |model| {
            species
                .iter()
                .map(|name| {
                    model
                        .class_levels
                        .iter()
                        .position(|level| level == name)
                        .expect("species level present")
                })
                .collect()
        },
    );
}

/// #2612: WHERE the posterior-mean estimand stops being publishable, and what
/// is different about those rows.
///
/// The stride-4 fit now completes, and the acceptance is red one step further
/// on: `predict_multinomial_formula` refuses because a held-out row's predictive
/// mass defect exceeds `PREDICTIVE_MASS_DEFECT_TOLERANCE`. That refusal is the
/// estimand's own exact identity (`Σ_c E[p_c] = 1`) measuring the Laplace
/// approximation's error at the row being published, so it is a real statement —
/// but "one row is bad" and "the fit's posterior is undescribable" are different
/// findings and the message could not tell them apart until the refusal was made
/// to report the block.
///
/// This prints, per held-out row, the plug-in `|η|∞` and the argmax probability
/// alongside the refusal's own block summary, so the offending rows can be
/// characterised rather than guessed at. Asserts nothing.
#[test]
fn zz_probe_2612_penguins_stride4_predictive_mass() {
    init();
    let (train, test, species) = penguins_split(4);
    let config = FitConfig::default();
    let started = Instant::now();
    let model = match fit_penalized_multinomial_formula(&MultinomialFitRequest {
        data: &train,
        formula: PENGUINS_FORMULA,
        config: &config,
        init_lambda: 1.0,
        max_iter: 100,
        tol: 1e-8,
    }) {
        Ok(model) => model,
        Err(error) => {
            eprintln!("#2612 [mass] FIT FAILED after {:.1}s: {error}", started.elapsed().as_secs_f64());
            return;
        }
    };
    eprintln!("#2612 [mass] FIT OK in {:.1}s", started.elapsed().as_secs_f64());

    // The refusal, now carrying the whole block rather than one witness.
    match predict_multinomial_formula(&model, &test) {
        Ok(_) => eprintln!("#2612 [mass] posterior-mean AVAILABLE for every row"),
        Err(error) => eprintln!("#2612 [mass] posterior-mean REFUSED: {error}"),
    }

    // What is different about the rows, in the coordinates the estimand is a
    // function of: the plug-in linear predictor's magnitude and its confidence.
    let plugin = predict_multinomial_formula_plugin(&model, &test).expect("plug-in predict");
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
    let mut summary: Vec<(usize, f64, bool)> = Vec::new();
    for row in 0..plugin.nrows() {
        let mut best = (0usize, f64::NEG_INFINITY);
        for class in 0..plugin.ncols() {
            if plugin[[row, class]] > best.1 {
                best = (class, plugin[[row, class]]);
            }
        }
        summary.push((row, best.1, best.0 == labels[row]));
    }
    summary.sort_by(|a, b| a.1.partial_cmp(&b.1).expect("finite"));
    eprintln!("#2612 [mass] ten LEAST confident held-out rows (plug-in):");
    for (row, p, correct) in summary.iter().take(10) {
        eprintln!("#2612 [mass]   row {row:3}  argmax_p={p:.6}  correct={correct}");
    }
    eprintln!("#2612 [mass] ten MOST confident held-out rows (plug-in):");
    for (row, p, correct) in summary.iter().rev().take(10) {
        eprintln!("#2612 [mass]   row {row:3}  argmax_p={p:.6}  correct={correct}");
    }
}
