//! #2750 probe — WHERE the measure-jet fit spends the time that the
//! `mjs <= 2.0 x matern` speed gate is now measuring.
//!
//! The accuracy half of #2750 is delivered and the bill for it landed on
//! `measure_jet_perf_parity::measure_jet_single_scale_mode_is_speed_competitive`
//! (`mjs = 3.367 s`, `matern = 1.513 s`, bar `3.026 s`). `9b959b30e` already
//! records that gate as one of the fixtures the range screen MOVED, and the
//! parity header states the trade in advance ("a design-moving outer coordinate
//! rebuilds the representer design per outer trial"). What neither says is which
//! of the three candidate costs it is.
//!
//! This runs the A/B that separates them on the parity fixture's own rows,
//! without touching the estimator — each arm is a formula, not a code path:
//!
//! | arm | screen | ψ search on `ln ℓ` | reads |
//! |---|---|---|---|
//! | `mjs(...)`                              | on  | on  | as shipped |
//! | `mjs(..., length_scale=ℓ̂)`              | off | on  | shipped minus the screen |
//! | `mjs(..., learn_length_scale=false)`    | on  | off | shipped minus the search |
//! | `matern(...)`                           | -   | on  | the control: a same-size representer whose `log κ` is the same kind of coordinate |
//!
//! `kappa_timing` carries the outer accounting (`cost_calls`, `eval_calls`,
//! `*_total_s`), so the iteration count and the per-call cost are read off the
//! fit rather than inferred from wall clock.
//!
//! Diagnostic-only: it asserts that every arm produced a finite time and a
//! usable fit, and nothing about the level.
//!
//! # Why this is its own test binary
//!
//! It installs a process-global `log` sink, and `measure_jet_perf_parity`'s
//! speed gate lives in the `measure_jet` target. `log::set_max_level(Info)` is
//! process state, and `log::info!` evaluates its format ARGUMENTS eagerly — the
//! outer κ/ψ phase's per-callback records build a psi string and read the design
//! revision — so a logger installed by one test silently taxes every other test
//! in the same process. Under `cargo nextest` each test is its own process and
//! the hazard does not arise; under plain `cargo test` it does, and both routes
//! are supported here. A test that installs a global logger does not share a
//! process with a timing gate.

use csv::StringRecord;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};
use std::time::Instant;

use gam::smooth::SmoothBasisSpec;

/// Minimal `log` sink: the outer κ/ψ phase already emits one `[KAPPA-PHASE]`
/// record per cost / eval / EFS callback carrying the trial's `log_kappa_norm`,
/// and a `[KAPPA-PHASE-SUMMARY]` at the end. Those records are the ψ trajectory,
/// so the probe installs a sink for them rather than re-deriving the trajectory
/// from outside the optimizer. Only the two `[KAPPA-PHASE...]` prefixes are
/// forwarded; everything else the engine logs is dropped, because an unfiltered
/// sink at `info` buries the trajectory it exists to show.
struct KappaPhaseSink;

impl log::Log for KappaPhaseSink {
    fn enabled(&self, metadata: &log::Metadata<'_>) -> bool {
        metadata.level() <= log::Level::Info
    }

    fn log(&self, record: &log::Record<'_>) {
        let line = format!("{}", record.args());
        if line.contains("[KAPPA-PHASE") {
            println!("[2750-cost]   {line}");
        }
    }

    fn flush(&self) {}
}

static KAPPA_PHASE_SINK: KappaPhaseSink = KappaPhaseSink;

/// `measure_jet_perf_parity`'s fixture, verbatim: a 1-D curve in 3-D.
const N_TRAIN: usize = 1_500;
const SIGMA: f64 = 0.10;
const TRAIN_SEED: u64 = 1_039;

fn clamp_unit_open(x: f64) -> f64 {
    x.max(1.0e-6).min(1.0 - 1.0e-6)
}

fn latent_to_coords(t: f64) -> [f64; 3] {
    [
        clamp_unit_open(t),
        clamp_unit_open(0.5 + 0.5 * (2.0 * std::f64::consts::PI * t).sin()),
        clamp_unit_open(t * t),
    ]
}

fn truth(t: f64) -> f64 {
    (2.0 * std::f64::consts::PI * t).sin() + 0.5 * (4.0 * std::f64::consts::PI * t).cos()
}

fn build_dataset(n: usize, sigma: f64, seed: u64) -> gam::data::EncodedDataset {
    let mut rng = StdRng::seed_from_u64(seed);
    let latent = Uniform::new(0.0, 1.0).expect("uniform latent");
    let noise = Normal::new(0.0, sigma).expect("normal noise");
    let headers = ["x0", "x1", "x2", "y"]
        .iter()
        .cloned()
        .map(String::from)
        .collect();
    let rows: Vec<StringRecord> = (0..n)
        .map(|_| {
            let t = latent.sample(&mut rng);
            let coords = latent_to_coords(t);
            let y = truth(t) + noise.sample(&mut rng);
            StringRecord::from(vec![
                coords[0].to_string(),
                coords[1].to_string(),
                coords[2].to_string(),
                y.to_string(),
            ])
        })
        .collect();
    encode_recordswith_inferred_schema(headers, rows).expect("encode parity dataset")
}

fn fit_and_report(label: &str, body: &str, data: &gam::data::EncodedDataset) -> (f64, Option<f64>) {
    let config = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let start = Instant::now();
    let result = fit_from_formula(&format!("y ~ {body}"), data, &config)
        .unwrap_or_else(|error| panic!("{label}: {error}"));
    let elapsed = start.elapsed().as_secs_f64();
    let FitResult::Standard(fit) = result else {
        panic!("{label}: expected a standard Gaussian fit");
    };
    let realized = fit.resolvedspec.smooth_terms.iter().find_map(|term| {
        if let SmoothBasisSpec::MeasureJet { spec, .. } = &term.basis {
            Some(spec.length_scale)
        } else {
            None
        }
    });
    let timing = fit.kappa_timing.as_ref();
    println!(
        "[2750-cost] {label:<26} wall={elapsed:>7.3}s  psi_dim={:>2}  cost_calls={:>4} \
         cost_s={:>7.3}  eval_calls={:>4} eval_s={:>7.3}  efs_calls={:>4} efs_s={:>7.3}  \
         design_rebuilds={:>5}  ell={}",
        timing.map(|t| t.log_kappa_dim).unwrap_or(0),
        timing.map(|t| t.cost_calls).unwrap_or(0),
        timing.map(|t| t.cost_total_s).unwrap_or(0.0),
        timing.map(|t| t.eval_calls).unwrap_or(0),
        timing.map(|t| t.eval_total_s).unwrap_or(0.0),
        timing.map(|t| t.efs_calls).unwrap_or(0),
        timing.map(|t| t.efs_total_s).unwrap_or(0.0),
        timing.map(|t| t.design_revision_delta).unwrap_or(0),
        realized
            .map(|v| format!("{v:.6}"))
            .unwrap_or_else(|| "-".to_string()),
    );
    (elapsed, realized)
}

#[test]
fn measure_jet_range_cost_decomposition_2750() {
    init_parallelism();
    if log::set_logger(&KAPPA_PHASE_SINK).is_ok() {
        log::set_max_level(log::LevelFilter::Info);
    }
    let data = build_dataset(N_TRAIN, SIGMA, TRAIN_SEED);

    let (shipped_secs, screened_ell) = fit_and_report("mjs shipped", "mjs(x0, x1, x2, centers=16)", &data);
    let screened_ell = screened_ell.expect("the shipped mjs fit realizes a representer range");

    // The realized range travels in the FROZEN (standardized) frame, and a
    // fresh `length_scale=` request is in ORIGINAL units, so pinning it back
    // verbatim is not the same geometry. What this arm needs is only that the
    // screen does NOT run, which any positive explicit range achieves; the
    // frozen value is used so the arm sits at the range the shipped fit chose
    // up to that frame conversion.
    let (pinned_secs, _) = fit_and_report(
        "mjs explicit (no screen)",
        &format!("mjs(x0, x1, x2, centers=16, length_scale={screened_ell}, learn_length_scale=true)"),
        &data,
    );
    let (frozen_secs, _) = fit_and_report(
        "mjs frozen (no search)",
        "mjs(x0, x1, x2, centers=16, learn_length_scale=false)",
        &data,
    );
    let (matern_secs, _) = fit_and_report("matern control", "matern(x0, x1, x2, k=16)", &data);

    println!(
        "[2750-cost] shipped/matern={:.3}x   screen_share={:.3}s   search_share={:.3}s",
        shipped_secs / matern_secs,
        shipped_secs - pinned_secs,
        shipped_secs - frozen_secs,
    );
    assert!(
        [shipped_secs, pinned_secs, frozen_secs, matern_secs]
            .iter()
            .all(|s| s.is_finite() && *s > 0.0),
        "every arm must produce a positive finite wall time"
    );
}
