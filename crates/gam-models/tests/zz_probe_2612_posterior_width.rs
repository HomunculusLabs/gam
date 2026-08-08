//! #2612 measurement probe (diagnostic only, asserts nothing about the defect).
//!
//! Establishes, at the current tree, the quantities the issue thread argues
//! about: which estimand the published penguin probabilities are, how wide the
//! Laplace posterior is, where that width lives, whether the Jeffreys/Firth
//! proper prior armed at all, and which smoothing parameters sit on the
//! `MULTINOMIAL_FORMULA_PRIOR_PSEUDO_OBS` wall.
//!
//! Everything printed here is tool-free: no reference implementation is
//! consulted, so the numbers stand on their own on a host with no R.

use csv::StringRecord;
use gam_data::encode_recordswith_inferred_schema;
use gam_models::fit_orchestration::FitConfig;
use gam_models::multinomial::{
    MultinomialFitRequest, fit_penalized_multinomial_formula, predict_multinomial_formula,
    predict_multinomial_formula_plugin,
};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::sync::Mutex;

const PENGUINS_CSV: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../bench/datasets/penguins.csv"
);

const K: usize = 3;

static CAPTURED: Mutex<Vec<String>> = Mutex::new(Vec::new());

struct CapturingLogger;

impl log::Log for CapturingLogger {
    fn enabled(&self, metadata: &log::Metadata<'_>) -> bool {
        metadata.level() <= log::Level::Info
    }

    fn log(&self, record: &log::Record<'_>) {
        if self.enabled(record.metadata()) {
            let line = format!("{}", record.args());
            if line.contains("multinomial REML") {
                CAPTURED
                    .lock()
                    .expect("probe log buffer is not poisoned")
                    .push(line);
            }
        }
    }

    fn flush(&self) {}
}

static CAPTURING_LOGGER: CapturingLogger = CapturingLogger;

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
    let header = lines
        .next()
        .expect("penguins header line")
        .expect("read penguins header");
    let cols: Vec<&str> = header.trim().split(',').collect();
    let idx = |name: &str| {
        cols.iter()
            .position(|c| *c == name)
            .unwrap_or_else(|| panic!("penguins.csv missing column {name}"))
    };
    let i_species = idx("species");
    let i_bill_len = idx("bill_length_mm");
    let i_bill_dep = idx("bill_depth_mm");
    let i_flipper = idx("flipper_length_mm");
    let i_mass = idx("body_mass_g");

    let mut rows = Vec::new();
    for line in lines {
        let line = line.expect("read penguins row");
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split(',').collect();
        let any_na = [i_bill_len, i_bill_dep, i_flipper, i_mass]
            .iter()
            .any(|&c| f[c] == "NA" || f[c].is_empty());
        if any_na {
            continue;
        }
        rows.push(Penguin {
            bill_length: f[i_bill_len].parse().expect("parse bill_length_mm"),
            bill_depth: f[i_bill_dep].parse().expect("parse bill_depth_mm"),
            flipper: f[i_flipper].parse().expect("parse flipper_length_mm"),
            body_mass: f[i_mass].parse().expect("parse body_mass_g"),
            species: f[i_species].to_string(),
        });
    }
    rows
}

fn mean_log_loss(probs: &[f64], labels: &[usize], k: usize) -> f64 {
    let m = labels.len();
    let mut acc = 0.0;
    for (i, &y) in labels.iter().enumerate() {
        acc -= probs[i * k + y].clamp(1e-15, 1.0).ln();
    }
    acc / m as f64
}

fn accuracy(probs: &[f64], labels: &[usize], k: usize) -> f64 {
    let mut correct = 0usize;
    for (i, &y) in labels.iter().enumerate() {
        let row = &probs[i * k..(i + 1) * k];
        let mut best = 0usize;
        for c in 1..k {
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

/// Mean probability assigned to the row's own argmax class. Compared against
/// accuracy this is a reference-free calibration gap: a perfectly calibrated
/// classifier has `mean argmax probability == accuracy`.
fn mean_argmax_probability(probs: &[f64], n: usize, k: usize) -> f64 {
    let mut acc = 0.0;
    for i in 0..n {
        let row = &probs[i * k..(i + 1) * k];
        acc += row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    }
    acc / n as f64
}

#[test]
fn zz_probe_2612_penguins_posterior_width() {
    if log::set_logger(&CAPTURING_LOGGER).is_ok() {
        log::set_max_level(log::LevelFilter::Info);
    }
    faer::set_global_parallelism(faer::Par::rayon(0));

    const STRIDE: usize = 3;
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
    let n_test = test_idx.len();

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
        .expect("encode penguin train dataset");
    let test_ds = encode_recordswith_inferred_schema(headers.clone(), make_records(&test_idx))
        .expect("encode penguin test dataset");

    let cfg = FitConfig::default();
    let model = fit_penalized_multinomial_formula(&MultinomialFitRequest {
        data: &train_ds,
        formula: "species ~ s(bill_length_mm, k=10) + s(bill_depth_mm, k=10) + s(flipper_length_mm, k=10) + s(body_mass_g, k=10)",
        config: &cfg,
        init_lambda: 1.0,
        max_iter: 100,
        tol: 1e-8,
    })
    .expect("gam penguin multinomial fit");

    let class_levels = model.class_levels.clone();
    let label_to_col = |lvl: &str| -> usize {
        class_levels
            .iter()
            .position(|c| c == lvl)
            .expect("species in class_levels")
    };
    let test_labels: Vec<usize> = test_idx
        .iter()
        .map(|&i| label_to_col(&rows[i].species))
        .collect();

    let posterior = predict_multinomial_formula(&model, &test_ds).expect("posterior-mean predict");
    let plugin = predict_multinomial_formula_plugin(&model, &test_ds).expect("plug-in predict");
    let flat = |m: &ndarray::Array2<f64>| -> Vec<f64> {
        let mut v = Vec::with_capacity(n_test * K);
        for i in 0..n_test {
            for c in 0..K {
                v.push(m[[i, c]]);
            }
        }
        v
    };
    let post_flat = flat(&posterior);
    let plug_flat = flat(&plugin);

    let post_acc = accuracy(&post_flat, &test_labels, K);
    let plug_acc = accuracy(&plug_flat, &test_labels, K);
    let post_ll = mean_log_loss(&post_flat, &test_labels, K);
    let plug_ll = mean_log_loss(&plug_flat, &test_labels, K);
    let post_conf = mean_argmax_probability(&post_flat, n_test, K);
    let plug_conf = mean_argmax_probability(&plug_flat, n_test, K);

    eprintln!(
        "[2612] n_train={} n_test={n_test} K={K}\n\
         [2612] posterior-mean: acc={post_acc:.6} logloss={post_ll:.6} \
         mean_argmax_p={post_conf:.6} calib_gap={:.6}\n\
         [2612] plug-in      : acc={plug_acc:.6} logloss={plug_ll:.6} \
         mean_argmax_p={plug_conf:.6} calib_gap={:.6}",
        train_idx.len(),
        post_acc - post_conf,
        plug_acc - plug_conf,
    );

    // λ, the wall, and which of them sit on it.
    let wall = 8.0e-4 * 0.25 * (50.0f64 / 46.0).max(1.0);
    for (i, lam) in model.lambdas.iter().enumerate() {
        let label = model
            .lambda_labels
            .get(i)
            .map(|s| s.as_str())
            .unwrap_or("<unlabelled>");
        eprintln!(
            "[2612] lambda[{i}] = {lam:.9e}  ratio_to_wall={:.6}  label={label}",
            lam / wall
        );
    }
    for (i, edf) in model.edf_per_penalty.iter().enumerate() {
        eprintln!("[2612] edf_per_penalty[{i}] = {edf:.6}");
    }

    // Posterior width: the joint covariance spectrum and the per-row logit sd.
    let cov = model.coefficient_covariance().expect("joint covariance");
    let dim = cov.nrows();
    let sym = 0.5 * (&cov + &cov.t());
    let eig = gam_linalg::faer_ndarray::FaerEigh::eigh_vals(&sym, faer::Side::Lower)
        .expect("covariance eigenvalues");
    let lo = eig.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = eig.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let trace: f64 = (0..dim).map(|i| cov[[i, i]]).sum();
    eprintln!(
        "[2612] joint posterior covariance: dim={dim} lambda_min={lo:.6e} lambda_max={hi:.6e} \
         trace={trace:.6e}"
    );
    let mut top: Vec<f64> = eig.to_vec();
    top.sort_by(|a, b| b.partial_cmp(a).expect("finite eigenvalues"));
    let head: Vec<String> = top.iter().take(6).map(|v| format!("{v:.4e}")).collect();
    eprintln!("[2612] covariance top eigenvalues: {}", head.join(", "));

    for line in CAPTURED
        .lock()
        .expect("probe log buffer is not poisoned")
        .iter()
    {
        eprintln!("[2612][log] {line}");
    }
}
