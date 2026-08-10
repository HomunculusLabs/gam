//! #2612 single-binary instrument: every fixture this issue is decided on, in
//! one run, with nothing asserted.
//!
//! The four red targets this issue is currently carrying live in four different
//! test binaries, and each of them stops at its first panic. So a run that wants
//! to know *which* fixtures arm, *which* fixtures fit, and *what* the two
//! published estimands score costs four builds and still reports only the first
//! failure of each. This file asks all of those questions at once and prints
//! every answer, including the ones that are errors:
//!
//!   * `separation_evidence` — `None` for the unbiased penalized-REML mode, the
//!     certificate itself when the Jeffreys/Firth proper prior was armed;
//!   * whether the fit exists at all, and the full error text when it does not;
//!   * held-out log-loss and accuracy under BOTH estimands (`E[softmax(η)]`,
//!     which is what the acceptance bars score, and the plug-in `softmax(η̂)`),
//!     so the estimand split is a subtraction rather than an argument;
//!   * the selected λ against the formula path's own lower wall, because a fit
//!     railed on that wall is a boundary solution and not a stationary point.
//!
//! Named `probe_*`/`zz_*` so it can never be mistaken for an acceptance bar: it
//! asserts nothing. The bars stay where they are.

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

const PENGUINS_CSV: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../bench/datasets/penguins.csv"
);

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

// ── scoring ──────────────────────────────────────────────────────────────────

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

/// Everything the fit itself says about which estimand it published and where
/// its smoothing parameters landed.
fn report_fit(label: &str, model: &MultinomialSavedModel) {
    let lambdas: Vec<String> = model.lambdas.iter().map(|v| format!("{v:.4e}")).collect();
    eprintln!(
        "#2612 [{label}] separation_evidence = {:?}",
        model.separation_evidence
    );
    eprintln!(
        "#2612 [{label}] classes={:?} P={} lambdas=[{}] edf_per_class={:?}",
        model.class_levels,
        model.p_per_class,
        lambdas.join(", "),
        model.edf_per_class,
    );
}

/// Score a fitted model on held-out rows under BOTH published estimands.
fn report_scores(label: &str, model: &MultinomialSavedModel, test: &gam_data::EncodedDataset, labels: &[usize]) {
    match predict_multinomial_formula(model, test) {
        Ok(posterior) => eprintln!(
            "#2612 [{label}] posterior-mean: acc={:.4} logloss={:.5} mean_argmax_p={:.5} \
             calib_gap={:+.5}",
            accuracy(&posterior, labels),
            mean_log_loss(&posterior, labels),
            mean_argmax_probability(&posterior),
            mean_argmax_probability(&posterior) - accuracy(&posterior, labels),
        ),
        Err(error) => eprintln!("#2612 [{label}] posterior-mean predict FAILED: {error}"),
    }
    match predict_multinomial_formula_plugin(model, test) {
        Ok(plugin) => eprintln!(
            "#2612 [{label}] plug-in:        acc={:.4} logloss={:.5} mean_argmax_p={:.5} \
             calib_gap={:+.5}",
            accuracy(&plugin, labels),
            mean_log_loss(&plugin, labels),
            mean_argmax_probability(&plugin),
            mean_argmax_probability(&plugin) - accuracy(&plugin, labels),
        ),
        Err(error) => eprintln!("#2612 [{label}] plug-in predict FAILED: {error}"),
    }
}

fn fit_or_report(
    label: &str,
    data: &gam_data::EncodedDataset,
    formula: &str,
) -> Option<MultinomialSavedModel> {
    let config = FitConfig::default();
    let started = std::time::Instant::now();
    let outcome = fit_penalized_multinomial_formula(&MultinomialFitRequest {
        data,
        formula,
        config: &config,
        init_lambda: 1.0,
        max_iter: 100,
        tol: 1e-8,
    });
    let seconds = started.elapsed().as_secs_f64();
    match outcome {
        Ok(model) => {
            eprintln!("#2612 [{label}] FIT OK in {seconds:.1}s ({formula})");
            report_fit(label, &model);
            Some(model)
        }
        Err(error) => {
            eprintln!("#2612 [{label}] FIT FAILED in {seconds:.1}s ({formula}):\n    {error}");
            None
        }
    }
}

// ── fixture A: labels DRAWN from a smooth softmax truth (nothing separates) ───

const CLASS_NAMES: [&str; 3] = ["a", "b", "c"];

fn truth(x1: f64, x2: f64) -> [f64; 3] {
    let scores = [
        0.6 * x1 - 0.3 * x2,
        -0.4 * x1 + 0.5 * x2,
        0.2 * (x1 * x1 - x2 * x2) * 0.25,
    ];
    let shift = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let weights: Vec<f64> = scores.iter().map(|s| (s - shift).exp()).collect();
    let total: f64 = weights.iter().sum();
    [
        weights[0] / total,
        weights[1] / total,
        weights[2] / total,
    ]
}

fn covariates(index: usize) -> (f64, f64) {
    (
        -2.0 + 4.0 * (((index as f64) * 0.618_033_988_749_894_8) % 1.0),
        -2.0 + 4.0 * (((index as f64) * 0.414_213_562_373_095_1) % 1.0),
    )
}

fn drawn_records(n: usize) -> (Vec<StringRecord>, Vec<usize>) {
    let mut rows = Vec::with_capacity(n);
    let mut labels = Vec::with_capacity(n);
    let mut lcg: u64 = 0x2612_2612_2612_2612;
    let mut next_unit = || -> f64 {
        lcg = lcg
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((lcg >> 11) as f64) / ((1u64 << 53) as f64)
    };
    for index in 0..n {
        let (x1, x2) = covariates(index);
        let probabilities = truth(x1, x2);
        let mut draw = next_unit();
        let mut label = CLASS_NAMES.len() - 1;
        for (class, probability) in probabilities.iter().enumerate() {
            if draw < *probability {
                label = class;
                break;
            }
            draw -= probability;
        }
        labels.push(label);
        rows.push(StringRecord::from(vec![
            x1.to_string(),
            x2.to_string(),
            CLASS_NAMES[label].to_string(),
        ]));
    }
    (rows, labels)
}

fn two_covariate_frame(records: Vec<StringRecord>) -> gam_data::EncodedDataset {
    let headers = ["x1", "x2", "y"].into_iter().map(str::to_string).collect();
    encode_recordswith_inferred_schema(headers, records).expect("encode two-covariate frame")
}

// ── fixture B: a one-covariate QUASI-SEPARATED three-class stack ──────────────

const OVERLAP: f64 = 0.06;

fn van_der_corput(index: usize) -> f64 {
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

fn banded_label(index: usize, x: f64) -> usize {
    let wobble = if index % 7 == 0 { OVERLAP } else { -OVERLAP };
    if x < -1.0 / 3.0 + wobble {
        0
    } else if x < 1.0 / 3.0 + wobble {
        1
    } else {
        2
    }
}

fn banded_frame(indices: &[usize]) -> (gam_data::EncodedDataset, Vec<usize>) {
    let headers: Vec<String> = ["x", "cls"].iter().map(|s| s.to_string()).collect();
    let mut labels = Vec::with_capacity(indices.len());
    let records: Vec<StringRecord> = indices
        .iter()
        .map(|&index| {
            let x = van_der_corput(index);
            let y = banded_label(index, x);
            labels.push(y);
            StringRecord::from(vec![x.to_string(), format!("class{y}")])
        })
        .collect();
    (
        encode_recordswith_inferred_schema(headers, records).expect("encode banded frame"),
        labels,
    )
}

// ── fixture C: the penguins witness ──────────────────────────────────────────

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
        let (Some(bl), Some(bd), Some(fl), Some(bm)) = (
            parse(i_bl),
            parse(i_bd),
            parse(i_fl),
            parse(i_bm),
        ) else {
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

fn penguins_split(stride: usize) -> (gam_data::EncodedDataset, gam_data::EncodedDataset, Vec<String>) {
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

fn labels_in_model_order(model: &MultinomialSavedModel, species: &[String]) -> Vec<usize> {
    species
        .iter()
        .map(|name| {
            model
                .class_levels
                .iter()
                .position(|level| level == name)
                .unwrap_or_else(|| panic!("species {name:?} not in {:?}", model.class_levels))
        })
        .collect()
}

// ── the probe ────────────────────────────────────────────────────────────────

#[test]
fn zz_probe_2612_every_fixture_in_one_run() {
    init();

    // A. nothing separates: the prior must be DISARMED.
    let (train_records, _) = drawn_records(600);
    let train = two_covariate_frame(train_records);
    let mut holdout_records = Vec::with_capacity(600);
    let mut holdout_labels = Vec::with_capacity(600);
    let mut lcg: u64 = 0x0612_2612_2612_2612;
    let mut next_unit = || -> f64 {
        lcg = lcg
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((lcg >> 11) as f64) / ((1u64 << 53) as f64)
    };
    for index in 0..600 {
        let (x1, x2) = covariates(index + 100_000);
        let probabilities = truth(x1, x2);
        let mut draw = next_unit();
        let mut label = CLASS_NAMES.len() - 1;
        for (class, probability) in probabilities.iter().enumerate() {
            if draw < *probability {
                label = class;
                break;
            }
            draw -= probability;
        }
        holdout_labels.push(label);
        holdout_records.push(StringRecord::from(vec![
            x1.to_string(),
            x2.to_string(),
            CLASS_NAMES[label].to_string(),
        ]));
    }
    let holdout = two_covariate_frame(holdout_records);

    if let Some(model) = fit_or_report("A drawn-softmax smooth", &train, "y ~ s(x1, k=6) + s(x2, k=6)") {
        let labels: Vec<usize> = holdout_labels
            .iter()
            .map(|&class| {
                model
                    .class_levels
                    .iter()
                    .position(|level| level == CLASS_NAMES[class])
                    .expect("class level present")
            })
            .collect();
        report_scores("A drawn-softmax smooth", &model, &holdout, &labels);
        // The argmax-class gap `truth − fitted`, DECOMPOSED. A one-sided MAX over
        // 600 rows cannot tell a biased estimator from an unbiased noisy one:
        // the maximum of many mean-zero errors is positive by construction and
        // grows with the row count. So the same rows are reported four ways —
        // the signed MEAN (which is the bias, with its own standard error), the
        // worst gap in EACH direction, and the same statistics for the plug-in
        // estimand, so "the mode is shrunk" and "the predictive shrinks it" are
        // separable.
        let columns: Vec<usize> = CLASS_NAMES
            .iter()
            .map(|name| {
                model
                    .class_levels
                    .iter()
                    .position(|level| level == name)
                    .expect("class level")
            })
            .collect();
        let posterior = match predict_multinomial_formula(&model, &holdout) {
            Ok(prediction) => Some(prediction),
            Err(error) => {
                println!("[2612] posterior-mean prediction unavailable: {error}");
                None
            }
        };
        let plugin = match predict_multinomial_formula_plugin(&model, &holdout) {
            Ok(prediction) => Some(prediction),
            Err(error) => {
                println!("[2612] plug-in prediction unavailable: {error}");
                None
            }
        };
        for (estimand, predicted) in [("posterior-mean", &posterior), ("plug-in", &plugin)] {
            let Some(predicted) = predicted else { continue };
            let mut squared = 0.0_f64;
            let mut gaps: Vec<(f64, f64, f64)> = Vec::with_capacity(600);
            for index in 0..600 {
                let (x1, x2) = covariates(index + 100_000);
                let expected = truth(x1, x2);
                let argmax = expected
                    .iter()
                    .enumerate()
                    .fold((0usize, f64::NEG_INFINITY), |best, (c, v)| {
                        if *v > best.1 { (c, *v) } else { best }
                    })
                    .0;
                for class in 0..3 {
                    squared += (predicted[[index, columns[class]]] - expected[class]).powi(2);
                }
                gaps.push((
                    expected[argmax] - predicted[[index, columns[argmax]]],
                    x1,
                    x2,
                ));
            }
            let n = gaps.len() as f64;
            let mean = gaps.iter().map(|g| g.0).sum::<f64>() / n;
            let variance = gaps.iter().map(|g| (g.0 - mean).powi(2)).sum::<f64>() / (n - 1.0);
            let standard_error = (variance / n).sqrt();
            let worst_positive = gaps
                .iter()
                .copied()
                .fold((f64::NEG_INFINITY, 0.0, 0.0), |b, g| if g.0 > b.0 { g } else { b });
            let worst_negative = gaps
                .iter()
                .copied()
                .fold((f64::INFINITY, 0.0, 0.0), |b, g| if g.0 < b.0 { g } else { b });
            eprintln!(
                "#2612 [A drawn-softmax smooth] {estimand}: truth-RMSE={:.5}\n    \
                 argmax gap (truth-fitted): mean={:+.5} (s.e. {:.5}, t={:+.2}), sd={:.5}\n    \
                 worst SHRINK ={:+.5} at (x1={:+.3}, x2={:+.3})\n    \
                 worst INFLATE={:+.5} at (x1={:+.3}, x2={:+.3})",
                (squared / 1800.0).sqrt(),
                mean,
                standard_error,
                mean / standard_error,
                variance.sqrt(),
                worst_positive.0,
                worst_positive.1,
                worst_positive.2,
                worst_negative.0,
                worst_negative.1,
                worst_negative.2,
            );
        }
    }

    // A'. the same data with NO smooth: `S_lambda = 0`, so `H + S_lambda` is `H`.
    let (parametric_records, _) = drawn_records(600);
    let parametric = two_covariate_frame(parametric_records);
    if let Some(model) = fit_or_report("A' drawn-softmax parametric", &parametric, "y ~ x1 + x2") {
        report_fit("A' drawn-softmax parametric", &model);
    }

    // B. genuinely quasi-separated, one smooth.
    let (banded_train, _) = banded_frame(&(0..180).collect::<Vec<_>>());
    let (banded_test, banded_test_labels) = banded_frame(&(180..300).collect::<Vec<_>>());
    if let Some(model) = fit_or_report("B banded quasi-separated", &banded_train, "cls ~ s(x, k=8)") {
        let labels: Vec<usize> = banded_test_labels
            .iter()
            .map(|&class| {
                model
                    .class_levels
                    .iter()
                    .position(|level| *level == format!("class{class}"))
                    .expect("class level present")
            })
            .collect();
        report_scores("B banded quasi-separated", &model, &banded_test, &labels);
    }

    // B'. THE COUNTERFACTUAL. The same banded law at growing `n`, which is the
    // only knob that moves the arming decision without moving the geometry: the
    // certificate is taken on `ker(S_lambda)` — the class intercepts — and their
    // information is `sum_n w_n`, so it grows with `n` and eventually crosses
    // the gate's one-observation-equivalent knot. At n=180 it measured 0.984,
    // just under. Whatever `n` disarms it gives the ARMED and DISARMED fits on
    // ONE law, which is what says whether the calibration deficit belongs to the
    // proper prior or to the geometry.
    for n_train in [180usize, 360, 720, 1440] {
        let (train, _) = banded_frame(&(0..n_train).collect::<Vec<_>>());
        let (test, test_labels) = banded_frame(&(n_train..(n_train + 240)).collect::<Vec<_>>());
        let label = format!("B' banded n_train={n_train}");
        if let Some(model) = fit_or_report(&label, &train, "cls ~ s(x, k=8)") {
            let labels: Vec<usize> = test_labels
                .iter()
                .map(|&class| {
                    model
                        .class_levels
                        .iter()
                        .position(|level| *level == format!("class{class}"))
                        .expect("class level present")
                })
                .collect();
            report_scores(&label, &model, &test, &labels);
        }
    }

    // C. the penguins witness, at the stride the failing arm uses.
    for stride in [3usize, 4] {
        let (train, test, species) = penguins_split(stride);
        let label = format!("C penguins stride-{stride}");
        if let Some(model) = fit_or_report(
            &label,
            &train,
            "species ~ s(bill_length_mm, k=10) + s(bill_depth_mm, k=10) \
             + s(flipper_length_mm, k=10) + s(body_mass_g, k=10)",
        ) {
            let labels = labels_in_model_order(&model, &species);
            report_scores(&label, &model, &test, &labels);
        }
    }
}
