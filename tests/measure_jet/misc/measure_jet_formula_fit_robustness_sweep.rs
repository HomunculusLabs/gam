//! Robustness sweep (the *general-case* angle on #1126): the original report's
//! decisive evidence was statistical — "on a sweep of random seeds the Python
//! path fails on ~60% of datasets while the CLI fits every one." The single-repro
//! bug-hunt test and the single-dataset signal-recovery test each pin one input
//! geometry; neither rules out that the graceful κ-fallback only rescues *those*
//! layouts while other ordinary 1-D datasets still abort through the formula API.
//!
//! This test closes that gap directly: it fits a 1-D Gaussian measure-jet smooth
//! `s(x, bs="mjs")` through `fit_from_formula` (the exact path `gamfit.fit` and
//! the `fit_table` FFI take) across a spread of deterministic datasets that vary
//! the signal frequency, phase, additive-noise scale, sample size, and grid
//! jitter. Every one must return a usable result — finite coefficients, a finite
//! effective dof in a sane range, and a reconstruction that recovers its own
//! signal. A *single* abort fails the test, mirroring the CLI's "fit every one"
//! behaviour. Before #1126 a large fraction of these would have aborted with the
//! spatial-κ non-convergence escalated to a fatal `IntegrationFailed`.
//!
//! The datasets are bit-reproducible (a SplitMix64 finalizer, no RNG crate) so
//! the sweep is identical on every machine and the pass/fail verdict is stable.

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;

/// SplitMix64 finalizer mapped to [0, 1): deterministic, RNG-free per-index
/// pseudo-noise so every dataset in the sweep is bit-reproducible.
fn hashed_unit(index: u64) -> f64 {
    let mut z = index.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    (z >> 11) as f64 / (1u64 << 53) as f64
}

/// One sweep configuration: a single-cycle-family sine of variable frequency and
/// phase, sampled on a (optionally) jittered grid with additive noise. All knobs
/// are deterministic functions of `seed` so the whole sweep is reproducible.
struct SweepCase {
    seed: u64,
    n: usize,
    freq: f64,
    phase: f64,
    noise: f64,
    jitter: bool,
}

impl SweepCase {
    /// The noise-free signal this case's response is drawn around.
    fn signal(&self, x: f64) -> f64 {
        (std::f64::consts::TAU * self.freq * x + self.phase).sin()
    }

    fn dataset(&self) -> gam::data::EncodedDataset {
        let headers = ["x", "y"]
            .into_iter()
            .map(str::to_string)
            .collect::<Vec<_>>();
        // Build x values (sorted, in [0, 1]); jitter within one grid spacing when
        // requested so the sweep covers both regular and irregular layouts.
        let mut xs: Vec<f64> = (0..self.n)
            .map(|i| {
                let base = i as f64 / (self.n as f64 - 1.0);
                if self.jitter {
                    let j = (hashed_unit(i as u64 ^ self.seed.wrapping_mul(0x1234_5)) - 0.5)
                        / (self.n as f64 - 1.0);
                    (base + j).clamp(0.0, 1.0)
                } else {
                    base
                }
            })
            .collect();
        xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let rows = xs
            .iter()
            .enumerate()
            .map(|(i, &x)| {
                let noise =
                    2.0 * hashed_unit(
                        (i as u64)
                            .wrapping_mul(2_654_435_761)
                            .wrapping_add(self.seed.wrapping_mul(0x9E37_79B9)),
                    ) - 1.0;
                let y = self.signal(x) + self.noise * noise;
                StringRecord::from(vec![format!("{x:.17e}"), format!("{y:.17e}")])
            })
            .collect::<Vec<_>>();
        encode_recordswith_inferred_schema(headers, rows).expect("encode sweep dataset")
    }
}

/// One sweep case's measured outcome. Recorded for EVERY case, pass or fail, so
/// the table the caller prints is a measurement of the whole sweep rather than a
/// report of wherever the run happened to stop.
struct CaseReport {
    seed: u64,
    edf: f64,
    rmse: f64,
    budget: f64,
}

impl CaseReport {
    fn passed(&self) -> bool {
        self.rmse < self.budget
    }
}

/// Fit one case through the public formula API and MEASURE it. Any `Err` (the
/// pre-#1126 abort), a degenerate geometry, or a missed budget is RETURNED rather
/// than panicked.
///
/// #2750: this used to panic in place, so in any run where seed 1 failed, seeds
/// 2/3/4 were never measured at all -- the test's advertised coverage was
/// withdrawn by its own early exit exactly when it mattered. With the abort
/// deferred the remaining three were measured to pass with room (ratio 0.641 /
/// 0.228 / 0.568), which is a fact no run of the old shape could have produced.
///
/// No bar moves here: the budget expression and the edf sanity range below are
/// unchanged, and the caller still fails the test on any failing case.
fn check_case_fits(case: &SweepCase) -> Result<CaseReport, String> {
    let data = case.dataset();
    let config = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };

    let result = fit_from_formula("y ~ s(x, bs=\"mjs\")", &data, &config).map_err(|e| {
        format!(
            "measure-jet formula fit aborted on sweep case (seed={}, n={}, freq={}, \
             phase={:.2}, noise={}, jitter={}): {e}; \
             the gam CLI fits every such dataset, the formula path must too (#1126)",
            case.seed, case.n, case.freq, case.phase, case.noise, case.jitter
        )
    })?;
    let FitResult::Standard(fit) = result else {
        return Err(format!("seed={}: expected a standard Gaussian fit", case.seed));
    };

    if !fit.fit.beta.iter().all(|v| v.is_finite()) {
        return Err(format!(
            "seed={}: fitted coefficients must be finite",
            case.seed
        ));
    }
    let edf = fit
        .fit
        .edf_total()
        .ok_or_else(|| format!("seed={}: a fitted smooth must report a total effective dof", case.seed))?;
    if !(edf.is_finite() && edf > 1.0 && edf < case.n as f64) {
        return Err(format!(
            "seed={}: effective dof {edf} outside the sane range (1, {})",
            case.seed, case.n
        ));
    }

    // Signal recovery: reconstruct the fitted curve on a held-out fine grid and
    // compare to the noise-free truth. A flat/degenerate fallback would sit near
    // the mean (RMSE ≈ signal amplitude / √2 ≈ 0.71); a genuine smooth is well
    // below that. The budget scales loosely so higher-frequency cases (harder to
    // resolve at fixed n) and noisier cases still clear it without flaking.
    let grid: Vec<f64> = (0..300).map(|i| 0.003 + 0.994 * i as f64 / 299.0).collect();
    let mut m = Array2::<f64>::zeros((grid.len(), 2));
    for (i, &t) in grid.iter().enumerate() {
        m[[i, 0]] = t;
        m[[i, 1]] = 0.0;
    }
    let design = build_term_collection_design(m.view(), &fit.resolvedspec)
        .expect("rebuild measure-jet design on the held-out grid");
    let preds = design.design.apply(&fit.fit.beta).to_vec();
    let mut sse = 0.0;
    for (&t, &p) in grid.iter().zip(preds.iter()) {
        let e = p - case.signal(t);
        sse += e * e;
    }
    let rmse = (sse / grid.len() as f64).sqrt();
    let budget = 0.20 + 0.5 * case.noise;
    Ok(CaseReport {
        seed: case.seed,
        edf,
        rmse,
        budget,
    })
}

#[test]
fn measure_jet_formula_fit_succeeds_across_random_datasets() {
    init_parallelism();

    // A small spread of deterministic datasets covering the axes the original
    // report varied (the Python path failed on ordinary random draws). Keep the
    // normal CI gate representative rather than exhaustive: the test owns
    // formula-path non-abortion, not a full statistical simulation sweep.
    let cases = [
        SweepCase {
            seed: 1,
            n: 200,
            freq: 1.0,
            phase: 0.0,
            noise: 0.10,
            jitter: false,
        },
        SweepCase {
            seed: 2,
            n: 200,
            freq: 1.0,
            phase: 1.3,
            noise: 0.05,
            jitter: true,
        },
        SweepCase {
            seed: 3,
            n: 240,
            freq: 1.5,
            phase: 0.7,
            noise: 0.08,
            jitter: false,
        },
        SweepCase {
            seed: 4,
            n: 180,
            freq: 2.0,
            phase: 2.1,
            noise: 0.10,
            jitter: true,
        },
    ];

    // Measure every case BEFORE judging any of them (#2750).
    let outcomes: Vec<Result<CaseReport, String>> = cases.iter().map(check_case_fits).collect();

    // Printed unconditionally. A fixture whose diagnostics are conditional on its
    // own assertions cannot be read on the run that matters.
    println!("[2750-sweep] seed    edf    rmse  budget  ratio  verdict");
    for (case, outcome) in cases.iter().zip(outcomes.iter()) {
        match outcome {
            Ok(r) => println!(
                "[2750-sweep] {:>4}  {:5.2}  {:6.4}  {:6.4}  {:5.3}  {}",
                r.seed,
                r.edf,
                r.rmse,
                r.budget,
                r.rmse / r.budget,
                if r.passed() { "pass" } else { "FAIL" }
            ),
            Err(e) => println!("[2750-sweep] {:>4}      -       -       -      -  ERROR: {e}", case.seed),
        }
    }

    let failures: Vec<String> = outcomes
        .iter()
        .filter_map(|outcome| match outcome {
            Ok(r) if r.passed() => None,
            Ok(r) => Some(format!(
                "seed={}: measure-jet fit must recover its sine (RMSE={:.4}, budget={:.4}, \
                 edf={:.2}; a flat fit would be ~0.71)",
                r.seed, r.rmse, r.budget, r.edf
            )),
            Err(e) => Some(e.clone()),
        })
        .collect();
    assert!(
        failures.is_empty(),
        "{} of {} measure-jet sweep cases failed:\n{}",
        failures.len(),
        cases.len(),
        failures.join("\n")
    );
}
