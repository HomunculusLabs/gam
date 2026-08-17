//! #2612 PROBE (prints only): the two-class smooth multinomial predicts
//! `p = 0.500000` at every covariate value, in every replication.
//!
//! `probe_2612_interval_coverage` measured `mean = 0.500000` on all 40
//! replications of the #1891 coverage fixture while the truth ranged over
//! `[0.25, 0.72]`. That is not an interval-width defect and it is not
//! noise — a constant to six decimals is structural.
//!
//! There are exactly two places it can come from, and they need different
//! repairs, so this probe separates them by asking the SAME model the same
//! question two ways:
//!
//!   * through the SAVED TRAINING DESIGN (`predict_probabilities*`, no formula
//!     replay), which reads the coefficients and nothing else, and
//!   * through the PREDICT FRAME (`predict_multinomial_formula*`), which
//!     rebuilds the design from the saved termspec against fresh columns.
//!
//! If the first varies with `x` and the second does not, the design rebuild is
//! the defect. If both are flat, the fit collapsed. The coefficients, the
//! selected `λ`, and the per-class EDF are printed alongside so a collapse can
//! be told from a coefficient vector that is merely small.

use csv::StringRecord;
use gam_data::encode_recordswith_inferred_schema;
use gam_models::fit_orchestration::FitConfig;
use gam_models::multinomial::{
    MultinomialFitRequest, fit_penalized_multinomial_formula, predict_multinomial_formula,
    predict_multinomial_formula_plugin,
};

const N: usize = 240;

/// The solver narrates its own inner/outer lifecycle at `info`/`debug`. A probe
/// that has to guess which branch was taken is a probe that will guess wrong;
/// installing the sink is cheaper than another round of inference.
struct StderrLogger;

impl log::Log for StderrLogger {
    fn enabled(&self, metadata: &log::Metadata<'_>) -> bool {
        metadata.level() <= log::Level::Debug
    }

    fn log(&self, record: &log::Record<'_>) {
        if self.enabled(record.metadata()) {
            eprintln!("[{}] {}", record.target(), record.args());
        }
    }

    fn flush(&self) {}
}

static STDERR_LOGGER: StderrLogger = StderrLogger;

fn install_logger() {
    if log::set_logger(&STDERR_LOGGER).is_ok() {
        log::set_max_level(log::LevelFilter::Debug);
    }
}

struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1))
    }

    fn next_u01(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }
}

fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// The #1891 fixture's own truth: a low-frequency smooth log-odds on `x ∈ [0,1]`.
fn truth_eta(x: f64) -> f64 {
    0.1 + 1.4 * (std::f64::consts::TAU * (1.0 * x + 0.15)).sin()
}

#[test]
fn zz_probe_2612_two_class_smooth_predicts_a_constant_half() {
    install_logger();
    let x: Vec<f64> = (0..N).map(|i| i as f64 / (N - 1) as f64).collect();
    let mut rng = Lcg::new(0x2612_F1A7);
    let mut rows = Vec::with_capacity(N);
    let mut hi_count = 0usize;
    for &xi in &x {
        let p_hi = sigmoid(truth_eta(xi));
        let label = if rng.next_u01() < p_hi {
            hi_count += 1;
            "hi"
        } else {
            "lo"
        };
        rows.push(StringRecord::from(vec![
            format!("{xi:.8}"),
            label.to_string(),
        ]));
    }
    eprintln!(
        "#2612-FLAT fixture: n={N} hi={hi_count} ({:.4}) truth range p=[{:.4}, {:.4}]",
        hi_count as f64 / N as f64,
        x.iter()
            .map(|&v| sigmoid(truth_eta(v)))
            .fold(f64::INFINITY, f64::min),
        x.iter()
            .map(|&v| sigmoid(truth_eta(v)))
            .fold(f64::NEG_INFINITY, f64::max),
    );

    let data =
        encode_recordswith_inferred_schema(vec!["x".to_string(), "y".to_string()], rows.clone())
            .expect("encode training dataset");
    let config = FitConfig::default();
    let model = fit_penalized_multinomial_formula(&MultinomialFitRequest {
        init_lambda: 1.0,
        max_iter: 60,
        tol: 1e-8,
        ..MultinomialFitRequest::new(&data, "y ~ s(x, bs='tp', k=8)", &config)
    })
    .expect("two-class smooth multinomial fit");

    let coefficients = model.coefficients_active().expect("coefficients");
    eprintln!(
        "#2612-FLAT fit: class_levels={:?} p_per_class={} m={} lambdas={:?} \
         edf_per_class={:?} edf_per_penalty={:?}",
        model.class_levels,
        model.p_per_class,
        model.n_active_classes,
        model.lambdas,
        model.edf_per_class,
        model.edf_per_penalty,
    );
    eprintln!(
        "#2612-FLAT coefficients (max|beta|={:.6e}): {:?}",
        coefficients.iter().fold(0.0_f64, |a, v| a.max(v.abs())),
        coefficients.iter().copied().collect::<Vec<f64>>(),
    );
    // The likelihood the FIT reports against the likelihood the SAVED
    // coefficients imply. `deviance = −2·log L(β̂)` is computed inside the fit,
    // at the fit's own `η`; the intercept-only-at-zero model has
    // `−2·n·log(1/K)` exactly. If the two differ, the fit found a mode the
    // payload does not carry.
    let flat_deviance = 2.0 * (N as f64) * (2.0_f64).ln();
    eprintln!(
        "#2612-FLAT likelihood: deviance={:.6} penalized_neg_log_lik={:.6} \
         deviance_at_beta_zero={flat_deviance:.6} separation_evidence={:?}",
        model.deviance,
        model.penalized_neg_log_likelihood,
        model.separation_evidence.as_deref(),
    );

    // ---- route A: the SAVED TRAINING DESIGN, no formula replay -------------
    let design = model.training_design().expect("training design");
    eprintln!(
        "#2612-FLAT training design: {}x{} column sup-norms {:?}",
        design.nrows(),
        design.ncols(),
        (0..design.ncols())
            .map(|c| design.column(c).iter().fold(0.0_f64, |a, v| a.max(v.abs())))
            .collect::<Vec<f64>>(),
    );
    let eta_train = design.dot(&coefficients);
    eprintln!(
        "#2612-FLAT eta from the saved design: range [{:.6e}, {:.6e}]",
        eta_train.iter().copied().fold(f64::INFINITY, f64::min),
        eta_train.iter().copied().fold(f64::NEG_INFINITY, f64::max),
    );

    // ---- route B: the PREDICT FRAME, formula replayed ----------------------
    let probe_x = [0.05_f64, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95];
    let frame_without_response = encode_recordswith_inferred_schema(
        vec!["x".to_string()],
        probe_x
            .iter()
            .map(|v| StringRecord::from(vec![format!("{v:.8}")]))
            .collect(),
    )
    .expect("encode response-free predict frame");
    let frame_with_response = encode_recordswith_inferred_schema(
        vec!["x".to_string(), "y".to_string()],
        probe_x
            .iter()
            .map(|v| StringRecord::from(vec![format!("{v:.8}"), "hi".to_string()]))
            .collect(),
    )
    .expect("encode predict frame carrying the response column");

    for (label, frame) in [
        ("no-response", &frame_without_response),
        ("with-response", &frame_with_response),
    ] {
        let posterior = predict_multinomial_formula(&model, frame).expect("posterior mean");
        let plugin = predict_multinomial_formula_plugin(&model, frame).expect("plug-in");
        for (index, xi) in probe_x.iter().enumerate() {
            eprintln!(
                "#2612-FLAT [{label}] x={xi:.2} truth={:.6} posterior={:.6} plugin={:.6}",
                sigmoid(truth_eta(*xi)),
                posterior[[index, 0]],
                plugin[[index, 0]],
            );
        }
    }

    // ---- how far does the collapse reach? ---------------------------------
    // Same truth, same rows, one axis varied at a time: class count, basis, and
    // whether the term is smooth at all. A collapse that follows ONE of these is
    // a statement about that branch.
    for (label, formula, classes) in [
        ("K=2 tp k=8", "y ~ s(x, bs='tp', k=8)", 2usize),
        ("K=2 tp k=5", "y ~ s(x, bs='tp', k=5)", 2),
        ("K=2 default s()", "y ~ s(x)", 2),
        ("K=2 parametric", "y ~ x", 2),
        ("K=3 tp k=8", "y ~ s(x, bs='tp', k=8)", 3),
        ("K=3 parametric", "y ~ x", 3),
    ] {
        let mut rng = Lcg::new(0x2612_F1A7);
        let mut variant_rows = Vec::with_capacity(N);
        for &xi in &x {
            let p_hi = sigmoid(truth_eta(xi));
            let draw = rng.next_u01();
            let label = if classes == 2 {
                if draw < p_hi { "hi" } else { "lo" }
            } else if draw < p_hi * 0.7 {
                "hi"
            } else if draw < p_hi * 0.7 + 0.3 {
                "mid"
            } else {
                "lo"
            };
            variant_rows.push(StringRecord::from(vec![
                format!("{xi:.8}"),
                label.to_string(),
            ]));
        }
        let variant_data = encode_recordswith_inferred_schema(
            vec!["x".to_string(), "y".to_string()],
            variant_rows,
        )
        .expect("encode variant dataset");
        match fit_penalized_multinomial_formula(&MultinomialFitRequest {
            init_lambda: 1.0,
            max_iter: 60,
            tol: 1e-8,
            ..MultinomialFitRequest::new(&variant_data, formula, &config)
        }) {
            Ok(variant) => {
                let beta = variant.coefficients_active().expect("variant coefficients");
                let plugin = predict_multinomial_formula_plugin(&variant, &variant_data)
                    .expect("variant plug-in");
                let spread = (0..plugin.nrows())
                    .map(|r| plugin[[r, 0]])
                    .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), v| {
                        (lo.min(v), hi.max(v))
                    });
                eprintln!(
                    "#2612-FLAT variant [{label}] K={} m={} max|beta|={:.6e} \
                     plugin_p0_range=[{:.6}, {:.6}] edf_per_class={:?}",
                    variant.class_levels.len(),
                    variant.n_active_classes,
                    beta.iter().fold(0.0_f64, |a, v| a.max(v.abs())),
                    spread.0,
                    spread.1,
                    variant.edf_per_class,
                );
            }
            Err(error) => eprintln!("#2612-FLAT variant [{label}] REFUSED: {error}"),
        }
    }

    // ---- route A again, but through the model's own predict entry point ----
    // Same coefficients, same covariance, a design the formula replay never
    // touched: this is what the predict frame is being compared against.
    let (mean_direct, _) = model
        .predict_probabilities_with_se(design.view())
        .expect("posterior mean on the saved design");
    for &row in &[0usize, N / 6, N / 3, N / 2, 2 * N / 3, 5 * N / 6, N - 1] {
        eprintln!(
            "#2612-FLAT [saved-design] row={row} x={:.4} truth={:.6} posterior={:.6}",
            x[row],
            sigmoid(truth_eta(x[row])),
            mean_direct[[row, 0]],
        );
    }
}
