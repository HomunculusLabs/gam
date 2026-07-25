//! Warm-start invariance contract (#969): warm state may change wall-clock
//! only, never the fixed point.
//!
//! ## Why this harness exists
//!
//! Every warm/cold divergence in the tracker (#873's cache-state-dependent
//! concave fit, #869's topology-blind cache key) was found by a user-level
//! symptom and fixed point-wise; nothing prevented the CLASS. A fit whose
//! result depends on what you ran before is the worst bug genus a
//! statistics tool can have, because it is invisible to any single-run
//! test. This harness is the permanent regression net: a matrix of fits
//! (families × constraint shapes) each run through a cold fit, a cache-priming
//! fit, and a warm fit in one process, asserting criterion-value and
//! coefficient agreement.
//!
//! ## Mechanics
//!
//! The cold arm sets `persist_warm_start_disk = false`, which structurally
//! prevents any persistent read regardless of process history. A priming arm
//! then enables persistence and completes the same fit, and the warm arm repeats
//! it with persistence enabled. This remains correct even though the persistent
//! store is process-global and memoized; mutating `TMPDIR` cannot reset that
//! `OnceLock` and therefore cannot prove coldness.
//!
//! ## The contract
//!
//! For every configuration, cold and warm must agree BITWISE on the outer
//! criterion value (REML/LAML), on every coefficient, and on the certified
//! log-λ. Both arms converging is not a separate assertion — a fit object
//! only ever comes from a converged optimization, so an arm that fails to
//! certify fails the fit call and names itself.
//!
//! Bitwise, not "close". This gate previously allowed 1e-6 on the criterion
//! and 1e-4 on β, and under that bound two of the three families it caught in
//! #2363 agreed with their warm twin to three or four digits while actually
//! minimizing a *different* criterion (the frozen nuisance differed, so the
//! objective differed). A tolerance wide enough to absorb solver wobble is
//! therefore wide enough to absorb the next instance of the bug — and
//! numerically close state is not interchangeable provenance for a profiled
//! nonconvex objective anyway, which is the same argument
//! `bind_certified_custom_family_terminal_mode` makes about inner modes.
//!
//! What makes bitwise achievable — and what makes this a real contract rather
//! than a lucky coincidence — is that both the nuisance freeze (#2363) and the
//! inner mode (#2366) are now pinned to canonical anchors rather than to
//! whatever the search happened to be handed, so the two arms do not merely
//! converge near each other, they follow the same computation.
//!
//! Failures name the first disagreeing coordinate and its ulp distance, so a
//! regression reports how far it moved rather than only that it moved.

use gam::inference::data::load_csvwith_inferred_schema;
use gam::init_parallelism;
use gam::solver::fit_orchestration::{FitConfig, FitResult, fit_from_formula};
use std::io::Write;

struct InvarianceCase {
    name: &'static str,
    family: &'static str,
    formula: &'static str,
    /// (x, y) generator; deterministic so both arms see identical data.
    data: fn() -> (Vec<f64>, Vec<f64>),
}

/// Everything an arm has to reproduce exactly.
///
/// There is no `converged` field because there is nothing to compare: a fit
/// object only ever comes from a converged optimization, so an arm that
/// reaches here at all has certified, and one that did not would have failed
/// the `fit_from_formula` unwrap in `fit_once` naming which arm it was.
struct ArmOutcome {
    criterion: f64,
    beta: Vec<f64>,
    log_lambdas: Vec<f64>,
}

/// Signed ulp distance between two `f64`s, using the standard
/// monotone-ordering trick so the count stays meaningful across zero.
fn ulp_distance(left: f64, right: f64) -> i128 {
    let order = |value: f64| -> i128 {
        let bits = value.to_bits() as i64;
        if bits < 0 {
            (i64::MIN - bits) as i128
        } else {
            bits as i128
        }
    };
    order(left) - order(right)
}

/// Describe the first coordinate at which two arms disagree BITWISE, or
/// `None` when every coordinate is bit-identical.
fn first_bitwise_gap(cold: &[f64], warm: &[f64]) -> Option<String> {
    if cold.len() != warm.len() {
        return Some(format!(
            "layout differs: cold len={} warm len={}",
            cold.len(),
            warm.len()
        ));
    }
    cold.iter()
        .zip(warm.iter())
        .enumerate()
        .find(|(_, (a, b))| a.to_bits() != b.to_bits())
        .map(|(index, (a, b))| {
            format!(
                "coordinate {index}: cold={a:.17e} warm={b:.17e} ulps={} |Δ|={:.3e}",
                ulp_distance(*a, *b),
                (a - b).abs()
            )
        })
}

/// Deterministic 32-bit LCG (Numerical Recipes constants) for
/// reproducible jitter/thresholds without a rand dependency.
fn lcg_uniform(state: &mut u64) -> f64 {
    *state = state.wrapping_mul(1664525).wrapping_add(1013904223) & 0xffff_ffff;
    (*state as f64) / 4294967296.0
}

fn gaussian_smooth_data() -> (Vec<f64>, Vec<f64>) {
    let n = 500usize;
    let mut rng = 0x9e3779b9u64;
    (0..n)
        .map(|i| {
            let x = i as f64 / (n as f64 - 1.0);
            let noise = 0.1 * (lcg_uniform(&mut rng) - 0.5);
            (
                x,
                (2.0 * std::f64::consts::PI * x).sin() + 0.5 * (5.0 * x).cos() + noise,
            )
        })
        .unzip()
}

fn poisson_count_data() -> (Vec<f64>, Vec<f64>) {
    let n = 500usize;
    let mut rng = 0x6a09e667u64;
    (0..n)
        .map(|i| {
            let x = i as f64 / (n as f64 - 1.0);
            let lambda = (0.5 + (2.0 * std::f64::consts::PI * x).sin()).exp();
            // Deterministic count draw: invert a uniform through a crude
            // discretization (lambda + centered jitter, floored at 0).
            let y = (lambda + 2.0 * (lcg_uniform(&mut rng) - 0.5))
                .round()
                .max(0.0);
            (x, y)
        })
        .unzip()
}

fn binomial_data() -> (Vec<f64>, Vec<f64>) {
    let n = 600usize;
    let mut rng = 0xb7e15162u64;
    (0..n)
        .map(|i| {
            let x = i as f64 / (n as f64 - 1.0);
            let eta = 2.0 * (2.0 * std::f64::consts::PI * x).sin();
            let p = 1.0 / (1.0 + (-eta).exp());
            let y = if lcg_uniform(&mut rng) < p { 1.0 } else { 0.0 };
            (x, y)
        })
        .unzip()
}

fn negative_binomial_data() -> (Vec<f64>, Vec<f64>) {
    let n = 500usize;
    let mut rng = 0x243f6a88u64;
    (0..n)
        .map(|i| {
            let x = i as f64 / (n as f64 - 1.0);
            let mu = (0.2 + 1.5 * (2.0 * std::f64::consts::PI * x).sin()).exp();
            // A deterministic overdispersed count fixture. Its non-flat
            // conditional mean makes seed-η and converged-η profiling
            // materially different, directly exercising #2363.
            let multiplier = if lcg_uniform(&mut rng) < 0.3 {
                0.15
            } else {
                1.75
            };
            let y = (mu * multiplier + lcg_uniform(&mut rng)).round().max(0.0);
            (x, y)
        })
        .unzip()
}

fn gamma_data() -> (Vec<f64>, Vec<f64>) {
    let n = 500usize;
    let mut rng = 0x13198a2eu64;
    (0..n)
        .map(|i| {
            let x = i as f64 / (n as f64 - 1.0);
            let mu = (0.3 + 0.9 * (2.0 * std::f64::consts::PI * x).sin()).exp();
            let y = mu * (0.7 + 0.6 * lcg_uniform(&mut rng));
            (x, y)
        })
        .unzip()
}

fn tweedie_data() -> (Vec<f64>, Vec<f64>) {
    let n = 500usize;
    let mut rng = 0xa4093822u64;
    (0..n)
        .map(|i| {
            let x = i as f64 / (n as f64 - 1.0);
            let mu = (0.1 + 1.1 * (2.0 * std::f64::consts::PI * x).sin()).exp();
            let u = lcg_uniform(&mut rng);
            let y = if u < 0.2 {
                0.0
            } else {
                mu * (0.55 + 0.9 * lcg_uniform(&mut rng))
            };
            (x, y)
        })
        .unzip()
}

fn beta_data() -> (Vec<f64>, Vec<f64>) {
    let n = 500usize;
    let mut rng = 0x082efa98u64;
    (0..n)
        .map(|i| {
            let x = i as f64 / (n as f64 - 1.0);
            let eta = 1.8 * (2.0 * std::f64::consts::PI * x).sin();
            let mu = 1.0 / (1.0 + (-eta).exp());
            let y = (mu + 0.18 * (lcg_uniform(&mut rng) - 0.5)).clamp(0.01, 0.99);
            (x, y)
        })
        .unzip()
}

fn monotone_constrained_data() -> (Vec<f64>, Vec<f64>) {
    let n = 500usize;
    let mut rng = 0x3c6ef372u64;
    (0..n)
        .map(|i| {
            let x = i as f64 / (n as f64 - 1.0);
            // Monotone trend with wiggle the constraint must fight, plus
            // jitter — the binding-constraint shape whose cold seeds the
            // #509/#873 class rejected.
            let noise = 0.05 * (lcg_uniform(&mut rng) - 0.5);
            (x, x + 0.08 * (4.0 * std::f64::consts::PI * x).sin() + noise)
        })
        .unzip()
}

fn fit_once(
    case: &InvarianceCase,
    x: &[f64],
    y: &[f64],
    arm: &str,
    persist_warm_start_disk: bool,
) -> ArmOutcome {
    let mut csv = String::from("x,y\n");
    for i in 0..x.len() {
        csv.push_str(&format!("{:.17e},{:.17e}\n", x[i], y[i]));
    }
    let mut tmp = std::env::temp_dir();
    tmp.push(format!(
        "gam_wsi_{}_{}_{}.csv",
        case.name,
        arm,
        std::process::id()
    ));
    {
        let mut f = std::fs::File::create(&tmp).expect("create synthetic csv");
        f.write_all(csv.as_bytes()).expect("write synthetic csv");
    }
    let ds = load_csvwith_inferred_schema(&tmp).expect("load synthetic data");
    std::fs::remove_file(&tmp).ok();

    let cfg = FitConfig {
        family: Some(case.family.to_string()),
        persist_warm_start_disk,
        ..FitConfig::default()
    };
    let result = fit_from_formula(case.formula, &ds, &cfg).unwrap_or_else(|e| {
        panic!(
            "[{}/{}] fit aborted — a {} fit must succeed regardless of cache state: {e}",
            case.name, arm, arm
        )
    });
    let FitResult::Standard(fit) = result else {
        panic!("[{}/{}] expected a Standard GAM fit", case.name, arm);
    };
    assert!(
        fit.fit.reml_score.is_finite(),
        "[{}/{}] non-finite criterion value",
        case.name,
        arm
    );
    ArmOutcome {
        criterion: fit.fit.reml_score,
        beta: fit.fit.beta.to_vec(),
        log_lambdas: fit.fit.log_lambdas.to_vec(),
    }
}

#[test]
fn fits_are_invariant_to_warm_start_cache_state_across_families() {
    init_parallelism();

    let cases = [
        InvarianceCase {
            name: "gaussian_smooth",
            family: "gaussian",
            formula: "y ~ s(x, k=10)",
            data: gaussian_smooth_data,
        },
        InvarianceCase {
            name: "poisson_smooth",
            family: "poisson",
            formula: "y ~ s(x, k=8)",
            data: poisson_count_data,
        },
        InvarianceCase {
            name: "binomial_smooth",
            family: "binomial",
            formula: "y ~ s(x, k=8)",
            data: binomial_data,
        },
        InvarianceCase {
            name: "negative_binomial_smooth",
            family: "negative-binomial",
            formula: "y ~ s(x, k=8)",
            data: negative_binomial_data,
        },
        InvarianceCase {
            name: "gamma_smooth",
            family: "gamma",
            formula: "y ~ s(x, k=8)",
            data: gamma_data,
        },
        InvarianceCase {
            name: "tweedie_smooth",
            family: "tweedie",
            formula: "y ~ s(x, k=8)",
            data: tweedie_data,
        },
        InvarianceCase {
            name: "beta_smooth",
            family: "beta",
            formula: "y ~ s(x, k=8)",
            data: beta_data,
        },
        InvarianceCase {
            name: "gaussian_monotone",
            family: "gaussian",
            formula: "y ~ s(x, k=10, shape=monotone_increasing)",
            data: monotone_constrained_data,
        },
    ];

    let mut failures: Vec<String> = Vec::new();
    for case in &cases {
        let (x, y) = (case.data)();
        let cold = fit_once(case, &x, &y, "cold", false);
        // Prime both the exact-key and seed-prefix checkpoints. This arm need
        // not itself be cold; after it completes the next arm is guaranteed to
        // have a valid persistent entry.
        drop(fit_once(case, &x, &y, "prime", true));
        let warm = fit_once(case, &x, &y, "warm", true);

        if let Some(gap) = first_bitwise_gap(
            std::slice::from_ref(&cold.criterion),
            std::slice::from_ref(&warm.criterion),
        ) {
            failures.push(format!(
                "[{}] criterion depends on cache state: {gap}",
                case.name
            ));
        }
        if let Some(gap) = first_bitwise_gap(&cold.beta, &warm.beta) {
            failures.push(format!(
                "[{}] coefficients depend on cache state: {gap}",
                case.name
            ));
        }
        if let Some(gap) = first_bitwise_gap(&cold.log_lambdas, &warm.log_lambdas) {
            failures.push(format!(
                "[{}] certified log-λ depends on cache state: {gap}",
                case.name
            ));
        }
    }

    assert!(
        failures.is_empty(),
        "warm-start invariance contract violated — a cache must never change \
         the fitted result:\n{}",
        failures.join("\n")
    );
}
