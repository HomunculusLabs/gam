//! Measurement: is the ρ-only (penalty-search) REML outer loop n-independent?
//!
//! Mechanism (a) of #1033 — θ-invariant Gram caching. When the design provably
//! does NOT move across the hyperparameters being searched (a ρ-only model:
//! Gaussian + identity link, where only the penalty precision S(ρ) changes and
//! NOT a κ/ψ design-shape hyperparameter), the cross-product Gram XᵀWX is
//! θ-invariant. `RemlState::gaussian_fixed_cache_if_eligible`
//! (solver/reml/runtime.rs) assembles XᵀWX / XᵀW(y−offset) / yᵀWy ONCE per fit
//! under a double-checked write lock and every outer λ/ρ trial reuses it: the
//! dense PLS fast path and the sparse outer scatter consume the cached Gram
//! instead of re-streaming the n-row product. So the per-trial outer-eval cost
//! is k-dimensional (an O(p³) factorization of XᵀWX + S(ρ)), n-independent
//! after the single O(n·p²) Gram build.
//!
//! This harness isolates that ρ-phase directly. Two B-spline smooths give a real
//! 2-D ρ surface over a fixed design. For each n it builds one external REML
//! evaluator, primes its θ-invariant Gram once at ρ=0, then times a deterministic
//! sequence of distinct value-only ρ trials. No fit is minted: the harness
//! exercises the exact objective interface, so SPEC 20 convergence obligations
//! are neither bypassed nor abused as a timing primitive. With the Gram cached,
//! every measured trial is a k-space factorization and must not scale with n.
//!
//! Wall-clock on a shared cluster node is noisy, so this is a *measurement* read
//! from the printed table — the hard assertion is a catastrophe guard (the
//! ρ-phase must not blow up super-linearly by an order of magnitude across the
//! n-sweep), a real tripwire rather than a calibrated timing bound.

use gam::terms::basis::{BSplineBasisSpec, BSplineKnotSpec};
use gam::{
    estimate::ExternalOptimOptions,
    smooth::{
        ShapeConstraint, SmoothBasisSpec, SmoothTermSpec, TermCollectionSpec,
        build_term_collection_design,
    },
    types::{InverseLink, LikelihoodSpec, ResponseFamily, StandardLink},
};
use ndarray::{Array1, Array2};
use std::time::Instant;

/// Two-feature Gaussian-identity fixture: a smooth additive signal on each of
/// two columns, observed with light noise. Deterministic so this stays a
/// timing/geometry check, not a stochastic power test.
fn simulate_2d_gaussian(n: usize) -> (Array2<f64>, Array1<f64>) {
    let mut x = Array2::<f64>::zeros((n, 2));
    let mut y = Array1::<f64>::zeros(n);
    for i in 0..n {
        let u = (i as f64) / (n as f64 - 1.0); // [0,1]
        let v = ((i * 7 + 3) % n) as f64 / (n as f64 - 1.0); // decorrelated [0,1]
        x[[i, 0]] = u;
        x[[i, 1]] = v;
        // smooth additive truth; tiny deterministic wiggle as "noise"
        let noise = 0.01 * ((i as f64 * 0.37).sin());
        y[i] = (2.0 * std::f64::consts::PI * u).sin()
            + 0.5 * (2.0 * std::f64::consts::PI * 2.0 * v).cos()
            + noise;
    }
    (x, y)
}

fn bspline_smooth(name: &str, col: usize) -> SmoothTermSpec {
    SmoothTermSpec {
        name: name.to_string(),
        basis: SmoothBasisSpec::BSpline1D {
            feature_col: col,
            spec: BSplineBasisSpec {
                degree: 3,
                penalty_order: 2,
                knotspec: BSplineKnotSpec::Generate {
                    data_range: (0.0, 1.0),
                    num_internal_knots: 12,
                },
                double_penalty: false,
                identifiability: Default::default(),
                boundary: Default::default(),
                boundary_conditions: Default::default(),
            },
        },
        shape: ShapeConstraint::None,
        joint_null_rotation: None,
    }
}

fn spec_2d() -> TermCollectionSpec {
    TermCollectionSpec {
        linear_terms: vec![],
        random_effect_terms: vec![],
        // Two independent penalised smooths → a real 2-D ρ outer search over a
        // FIXED (θ-invariant) design: the cache-eligible regime.
        smooth_terms: vec![bspline_smooth("f_u", 0), bspline_smooth("f_v", 1)],
    }
}

fn external_options(
    family: LikelihoodSpec,
    design: &gam::smooth::TermCollectionDesign,
) -> ExternalOptimOptions {
    ExternalOptimOptions {
        family,
        latent_cloglog: None,
        mixture_link: None,
        optimize_mixture: false,
        sas_link: None,
        optimize_sas: false,
        compute_inference: false,
        skip_rho_posterior_inference: true,
        max_iter: 30,
        tol: 1e-6,
        nullspace_dims: design.nullspace_dims.clone(),
        linear_constraints: design.linear_constraints.clone(),
        firth_bias_reduction: Some(false),
        penalty_shrinkage_floor: None,
        rho_prior: Default::default(),
        kronecker_penalty_system: None,
        kronecker_factored: None,
        persist_warm_start_disk: false,
    }
}

#[derive(Clone, Copy, Debug)]
struct RhoTrialTiming {
    prime_s: f64,
    trial_s: f64,
    calls: usize,
    checksum: f64,
}

/// Build one fixed-design evaluator, pay its sole n-row Gram construction
/// outside the measured interval, then evaluate distinct ρ points. Passing one
/// stable `design_revision` makes the post-prime calls take the exact
/// design-revision fast path used by the production fixed-design outer search.
fn run_rho_trials(n: usize) -> Result<RhoTrialTiming, String> {
    let (x, y) = simulate_2d_gaussian(n);
    let weights = Array1::ones(n);
    let offset = Array1::zeros(n);
    let design =
        build_term_collection_design(x.view(), &spec_2d()).map_err(|error| error.to_string())?;
    let family = LikelihoodSpec::new(
        ResponseFamily::Gaussian,
        InverseLink::Standard(StandardLink::Identity),
    );
    let options = external_options(family, &design);
    let rho_dim = design.penalties.len();
    if rho_dim != 2 {
        return Err(format!(
            "expected a two-coordinate rho surface, found {rho_dim}"
        ));
    }

    let mut evaluator = gam_solve::estimate::ExternalJointHyperEvaluator::new(
        y.view(),
        weights.view(),
        &design.design,
        offset.view(),
        &design.penalties,
        &options,
        "rho-only n-scaling instrument",
    )
    .map_err(|error| error.to_string())?;

    let evaluate = |evaluator: &mut gam_solve::estimate::ExternalJointHyperEvaluator<'_>,
                    rho: &Array1<f64>|
     -> Result<f64, String> {
        evaluator
            .evaluate_cost_only(
                &design.design,
                &design.penalties,
                &design.nullspace_dims,
                design.linear_constraints.clone(),
                rho,
                rho_dim,
                None,
                "rho-only n-scaling value",
                Some(0),
            )
            .map_err(|error| error.to_string())
    };

    let prime_started = Instant::now();
    let prime = evaluate(&mut evaluator, &Array1::zeros(rho_dim))?;
    let prime_s = prime_started.elapsed().as_secs_f64();
    if !prime.is_finite() {
        return Err("non-finite priming criterion".to_string());
    }

    const TRIALS: usize = 24;
    let trial_started = Instant::now();
    let mut checksum = 0.0;
    for trial in 0..TRIALS {
        let t = trial as f64;
        let rho = Array1::from_vec(vec![-4.0 + 0.31 * t, 3.0 - 0.23 * t]);
        let value = evaluate(&mut evaluator, &rho)?;
        if !value.is_finite() {
            return Err(format!("non-finite criterion at trial {trial}"));
        }
        checksum += value;
    }
    let trial_s = trial_started.elapsed().as_secs_f64();
    Ok(RhoTrialTiming {
        prime_s,
        trial_s,
        calls: TRIALS,
        checksum,
    })
}

/// Diagnostic + catastrophe guard: the ρ-only outer phase must not scale
/// linearly with n. With the θ-invariant Gram cached once, every outer ρ trial
/// after the first is k-space; the directly measured post-prime cost per trial
/// should stay roughly flat across an n-sweep, not grow with n.
#[test]
fn rho_outer_loop_is_n_independent() {
    let ns = [20_000usize, 80_000, 320_000];

    eprintln!(
        "[rho-n-scaling] {:>9}  {:>11}  {:>11}  {:>12}  {:>14}",
        "n", "prime_s", "trials_s", "per_trial_ms", "checksum"
    );
    let mut per_trial = Vec::with_capacity(ns.len());
    for &n in &ns {
        let timing = run_rho_trials(n).unwrap_or_else(|reason| {
            panic!("[rho-n-scaling] n={n}: direct rho instrumentation failed — {reason}")
        });
        let seconds_per_trial = timing.trial_s / timing.calls as f64;
        per_trial.push(seconds_per_trial);
        eprintln!(
            "[rho-n-scaling] {n:>9}  {:>11.4}  {:>11.4}  {:>12.3}  {:>14.6e}",
            timing.prime_s,
            timing.trial_s,
            1e3 * seconds_per_trial,
            timing.checksum,
        );
    }

    let first = per_trial.first().copied().unwrap_or(0.0).max(1e-6);
    let last = per_trial.last().copied().unwrap_or(0.0).max(1e-6);
    let n_ratio = (*ns.last().unwrap() as f64) / (*ns.first().unwrap() as f64);
    let trial_ratio = last / first;
    eprintln!(
        "[rho-n-scaling] n grew {n_ratio:.0}× ; post-prime rho trial grew {trial_ratio:.2}× \
         (n-independent ⇒ ~1×, not ~{n_ratio:.0}×)"
    );

    // Catastrophe guard: a per-trial n-row Gram rebuild would make direct
    // post-prime evaluations scale ~linearly with n (≈16× across this sweep).
    // Generous slack absorbs shared-node wall-clock noise; the invariant and
    // threshold are unchanged from the former subtraction-based harness.
    assert!(
        trial_ratio < 0.25 * n_ratio,
        "post-prime rho trial grew {trial_ratio:.2}× across a {n_ratio:.0}× n-sweep \
         — expected n-INDEPENDENT (θ-invariant Gram cache); a near-linear growth \
         means the per-trial n-row Gram rebuild was re-introduced (#1033 mechanism a)"
    );
}
