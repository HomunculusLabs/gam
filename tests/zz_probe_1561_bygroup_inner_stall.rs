//! zz_probe DIAGNOSTIC (#1561): WHICH early exit makes the by-group location-scale
//! inner solve return `converged = false` after only 7 cycles, and which part of the
//! model triggers it?
//!
//! `quality_vs_gamlss_gaussian_location_scale_by_group` refuses at current main:
//!
//! ```text
//! custom-family inner solve did not converge after 7 cycle(s); refusing to expose
//! profile objective derivatives for theta_dim=8 (rho_dim=8, psi_dim=0)
//! rho_checkpoint=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
//! ```
//!
//! Two facts make "7" the interesting number rather than the refusal itself:
//!
//! 1. the refusal is raised at `psi_hyper.rs:1508` on `!inner.converged`, and the
//!    block immediately above it (`tighten_inner_for_deriv`) raises
//!    `inner_max_cycles` to `max(200)` for exactly this derivative-bearing,
//!    `psi_dim == 0` path. Stopping at 7 is therefore NOT budget exhaustion;
//! 2. `inner_blockwise_fit` has sixteen distinct `converged = false` exits, one of
//!    which (`inner_blockwise_fit.rs:2587`, the gam#1088 non-finite-curvature guard)
//!    exists *specifically* to bail early "instead of grinding to
//!    inner_max_cycles". Several others are trust-region / plateau refusals.
//!
//! Static reading cannot pick between sixteen candidates, and this seam has a
//! recorded history of exactly that going wrong: four successive static verdicts
//! were overturned by traces on the #2298/#2366 thread ("static forensics
//! identifies CANDIDATES only; every mechanism claim needs a trace before any
//! code"). So this probe does not reason — it turns the solver's own
//! `[PIRLS/joint-Newton convergence]` debug channel on and lets the solver say
//! which exit it took, at which cycle.
//!
//! The second question is `rho_checkpoint` being all zeros, which distinguishes
//!   (a) the stall is pre-first-iteration — the outer loop never took a step, so
//!       the checkpoint still holds the seed, from
//!   (b) something later reset it.
//! The trace answers that directly: if the FIRST ρ evaluation already refuses,
//! the outer loop never had a second point to check in.
//!
//! The 2x2 below then localises WHICH part of the model carries the defect. The
//! production model puts a by-group `tp` smooth on BOTH the mean and the log-scale
//! block (8 λ's). Turning `by=group` off in one block at a time separates "the
//! by-carrier construction is the trigger" from "the two-block coupling is", and
//! costs one fit each.
//!
//! The census probes remain measurements: they print every result without gating.
//! The deterministic #2524 seed-303 reproduction below is a regression bar now
//! that its premature optimizer stop is resolved.

use csv::StringRecord;
use gam::progress_log::{init_logging, set_log_level};
use gam::{FitConfig, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};

const N_PER_GROUP: usize = 100;

/// True mean for group A. Identical to the quality fixture.
fn mean_a(x: f64) -> f64 {
    (2.0 * std::f64::consts::PI * x).sin()
}
/// True sigma for group A (smooth heteroscedastic hump).
fn sigma_a(x: f64) -> f64 {
    0.10 + 0.10 * (std::f64::consts::PI * x).sin()
}
/// True mean for group B.
fn mean_b(x: f64) -> f64 {
    0.5 + 0.3 * (3.0 * std::f64::consts::PI * x).sin()
}
/// True sigma for group B (near-linear ramp).
fn sigma_b(x: f64) -> f64 {
    0.12 + 0.08 * x
}

#[test]
fn zz_probe_bygroup_inner_stall_trace() {
    init_parallelism();
    // The solver's per-cycle convergence channel is `debug`; the library default is
    // `Warn` (#1688 — the trace is measurable fit overhead, so verbosity is an
    // explicit API rather than a process-global env var). Install the backend, then
    // raise the level.
    init_logging();
    set_log_level("debug");

    // ---- rebuild the quality fixture's dataset byte-for-byte -----------------
    // seed 321, group A rows first then group B, `to_string()` formatting.
    let mut rng = StdRng::seed_from_u64(321);
    let ux = Uniform::new(0.0_f64, 1.0_f64).expect("uniform x");
    let std_normal = Normal::new(0.0_f64, 1.0_f64).expect("standard normal");

    let headers = vec!["y".to_string(), "x".to_string(), "group".to_string()];
    let mut rows: Vec<StringRecord> = Vec::with_capacity(2 * N_PER_GROUP);
    for _ in 0..N_PER_GROUP {
        let x = ux.sample(&mut rng);
        let y = mean_a(x) + sigma_a(x) * std_normal.sample(&mut rng);
        rows.push(StringRecord::from(vec![
            y.to_string(),
            x.to_string(),
            "A".to_string(),
        ]));
    }
    for _ in 0..N_PER_GROUP {
        let x = ux.sample(&mut rng);
        let y = mean_b(x) + sigma_b(x) * std_normal.sample(&mut rng);
        rows.push(StringRecord::from(vec![
            y.to_string(),
            x.to_string(),
            "B".to_string(),
        ]));
    }
    assert_eq!(rows.len(), 2 * N_PER_GROUP, "fixture row count");
    let data = encode_recordswith_inferred_schema(headers, rows).expect("encode by-group data");
    eprintln!("[zz1561bg] fixture rebuilt: {} rows", 2 * N_PER_GROUP);

    // ---- the 2x2: by=group on the mean block x by=group on the scale block ---
    // Arm 1 is the production call the quality test makes.
    let arms: [(&str, &str, &str); 4] = [
        (
            "1 PRODUCTION  mu=by-group  sigma=by-group",
            "y ~ s(x, bs='tp', by=group)",
            "s(x, bs='tp', by=group)",
        ),
        (
            "2            mu=by-group  sigma=pooled  ",
            "y ~ s(x, bs='tp', by=group)",
            "s(x, bs='tp')",
        ),
        (
            "3            mu=pooled    sigma=by-group",
            "y ~ s(x, bs='tp')",
            "s(x, bs='tp', by=group)",
        ),
        (
            "4            mu=pooled    sigma=pooled  ",
            "y ~ s(x, bs='tp')",
            "s(x, bs='tp')",
        ),
    ];

    let mut verdicts: Vec<(&str, bool, String)> = Vec::new();
    for (label, mean_formula, noise_formula) in arms {
        eprintln!("\n[zz1561bg] ================ ARM {label} ================");
        eprintln!("[zz1561bg]   mean  = {mean_formula}");
        eprintln!("[zz1561bg]   noise = {noise_formula}");
        let cfg = FitConfig {
            family: Some("gaussian".to_string()),
            noise_formula: Some(noise_formula.to_string()),
            ..FitConfig::default()
        };
        match fit_from_formula(mean_formula, &data, &cfg) {
            Ok(_) => {
                eprintln!("[zz1561bg] ARM {label}: FIT SUCCEEDED");
                verdicts.push((label, true, String::new()));
            }
            Err(e) => {
                let msg = e.to_string();
                eprintln!("[zz1561bg] ARM {label}: REFUSED");
                eprintln!("[zz1561bg-MSG-BEGIN arm={label}]");
                eprintln!("{msg}");
                eprintln!("[zz1561bg-MSG-END arm={label}]");
                verdicts.push((label, false, msg));
            }
        }
    }

    eprintln!("\n[zz1561bg] ================ SUMMARY ================");
    for (label, ok, head) in &verdicts {
        eprintln!(
            "[zz1561bg] {label} -> {}",
            if *ok {
                "FIT".to_string()
            } else {
                format!("REFUSE ({head})")
            }
        );
    }
    eprintln!("[zz1561bg] done");
}

// ---------------------------------------------------------------------------
// #1561 log-sigma lane: WHY does the by-group SCALE block lose when the pooled
// and cyclic scale blocks WIN? Reports the structure predictions (1) and (2)
// turn on -- the scale block's coefficient count, edf and selected lambdas,
// by-group vs pooled on identical data -- plus per-group log-sigma truth
// recovery over paired seeds.
// ---------------------------------------------------------------------------

fn sigma_a_t(x: f64) -> f64 {
    0.10 + 0.10 * (std::f64::consts::PI * x).sin()
}
fn sigma_b_t(x: f64) -> f64 {
    0.12 + 0.08 * x
}
fn mean_a_t(x: f64) -> f64 {
    (2.0 * std::f64::consts::PI * x).sin()
}
fn mean_b_t(x: f64) -> f64 {
    0.5 + 0.3 * (3.0 * std::f64::consts::PI * x).sin()
}

#[test]
fn zz_probe_bygroup_sigma_block_structure() {
    init_parallelism();
    let n_per = 100usize;
    eprintln!("[zz1561sig] by-group SCALE block structure + log-sigma recovery");

    for (label, noise_formula) in [
        ("sigma=BY-GROUP", "s(x, bs='tp', by=group)"),
        ("sigma=POOLED  ", "s(x, bs='tp')"),
    ] {
        for seed in 1..=8u64 {
            let mut rng = StdRng::seed_from_u64(300 + seed);
            let ux = Uniform::new(0.0_f64, 1.0_f64).expect("u");
            let sn = Normal::new(0.0_f64, 1.0_f64).expect("n");
            let headers = vec!["y".to_string(), "x".to_string(), "group".to_string()];
            let mut rows: Vec<StringRecord> = Vec::with_capacity(2 * n_per);
            let mut xs_a = Vec::new();
            let mut xs_b = Vec::new();
            for _ in 0..n_per {
                let x = ux.sample(&mut rng);
                let y = mean_a_t(x) + sigma_a_t(x) * sn.sample(&mut rng);
                xs_a.push(x);
                rows.push(StringRecord::from(vec![
                    y.to_string(),
                    x.to_string(),
                    "A".to_string(),
                ]));
            }
            for _ in 0..n_per {
                let x = ux.sample(&mut rng);
                let y = mean_b_t(x) + sigma_b_t(x) * sn.sample(&mut rng);
                xs_b.push(x);
                rows.push(StringRecord::from(vec![
                    y.to_string(),
                    x.to_string(),
                    "B".to_string(),
                ]));
            }
            let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode");
            let cfg = FitConfig {
                family: Some("gaussian".to_string()),
                noise_formula: Some(noise_formula.to_string()),
                ..FitConfig::default()
            };
            match fit_from_formula("y ~ s(x, bs='tp', by=group)", &ds, &cfg) {
                Err(e) => {
                    // The refusal is reported WHOLE and deliberately so. This
                    // site used to cut it at 120 bytes, and all 120 are wrapper
                    // text (`exact two-block spatial optimization failed:
                    // custom-family optimization error in fit_custom_family
                    // outer smoothing`) -- `fit.rs` puts the failed certificate,
                    // the rho and the last objective error after it. #2501 was
                    // filed, argued and nearly closed on that prefix alone; the
                    // cause was never in the record. The byte-index slice could
                    // also panic mid-UTF-8, and these messages carry non-ASCII.
                    eprintln!("[zz1561sig] {label} seed={seed} REFUSED");
                    eprintln!("[zz1561sig-MSG-BEGIN label={label} seed={seed}]");
                    eprintln!("{e}");
                    eprintln!("[zz1561sig-MSG-END label={label} seed={seed}]");
                }
                Ok(res) => {
                    let gam::FitResult::GaussianLocationScale(fit) = res else {
                        eprintln!("[zz1561sig] {label} seed={seed} unexpected fit kind");
                        continue;
                    };
                    let loc = fit
                        .fit
                        .fit
                        .block_by_role(gam::solver::estimate::BlockRole::Location);
                    let sca = fit
                        .fit
                        .fit
                        .block_by_role(gam::solver::estimate::BlockRole::Scale);
                    match (loc, sca) {
                        (Some(l), Some(s)) => eprintln!(
                            "[zz1561sig] {label} seed={seed} | SCALE p={} edf={:.4} lambdas={:?} \
                             | LOC p={} edf={:.4} nlam={}",
                            s.beta.len(),
                            s.edf,
                            s.lambdas.to_vec(),
                            l.beta.len(),
                            l.edf,
                            l.lambdas.len()
                        ),
                        _ => eprintln!("[zz1561sig] {label} seed={seed} missing block"),
                    }
                }
            }
        }
    }
    eprintln!("[zz1561sig] done");
}

// ---------------------------------------------------------------------------
// #2524 lane: the ONE fit in the 8x2 census that still refuses after #2485,
// end to end. Kept separate from the census above because it turns the solver's
// own per-cycle channel on -- the trace is megabytes, and the census must stay
// readable.
//
// Provenance: adopted from an untracked probe drafted in this lane
// (`zz_probe_2501_two_block_refusal_class.rs`, recovered 2026-07-26) and folded
// in here rather than landed beside it, so the fixture generator has ONE
// definition. #2524 cites this fit's refusal text and its restart trace, and a
// measurement cited from an unlanded instrument is not reproducible.
// ---------------------------------------------------------------------------

#[test]
fn zz_probe_2501_bygroup_seed3_refusal_traced() {
    init_parallelism();
    // The per-cycle convergence channel is `debug`; the library default is
    // `Warn` (#1688), so the backend must be installed and the level raised.
    init_logging();
    set_log_level("debug");

    let n_per = 100usize;
    let seed = 3u64;
    let mut rng = StdRng::seed_from_u64(300 + seed);
    let ux = Uniform::new(0.0_f64, 1.0_f64).expect("u");
    let sn = Normal::new(0.0_f64, 1.0_f64).expect("n");
    let headers = vec!["y".to_string(), "x".to_string(), "group".to_string()];
    let mut rows: Vec<StringRecord> = Vec::with_capacity(2 * n_per);
    for _ in 0..n_per {
        let x = ux.sample(&mut rng);
        let y = mean_a_t(x) + sigma_a_t(x) * sn.sample(&mut rng);
        rows.push(StringRecord::from(vec![
            y.to_string(),
            x.to_string(),
            "A".to_string(),
        ]));
    }
    for _ in 0..n_per {
        let x = ux.sample(&mut rng);
        let y = mean_b_t(x) + sigma_b_t(x) * sn.sample(&mut rng);
        rows.push(StringRecord::from(vec![
            y.to_string(),
            x.to_string(),
            "B".to_string(),
        ]));
    }
    assert_eq!(rows.len(), 2 * n_per, "fixture row count");
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode");
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        noise_formula: Some("s(x, bs='tp', by=group)".to_string()),
        ..FitConfig::default()
    };
    eprintln!("[zz2524] seed={} sigma=by-group, tracing", 300 + seed);
    fit_from_formula("y ~ s(x, bs='tp', by=group)", &ds, &cfg).unwrap_or_else(|error| {
        panic!(
            "#2524 regression: generic relative-stall termination pre-empted the \
             authoritative custom-family stationarity certificate:\n{error}"
        )
    });
    eprintln!("[zz2524] FIT");
    eprintln!("[zz2524] done");
}
