//! #2953: a certified optimum that an evaluated state beats is declined (#2596, #2627), and
//! when nothing certifies in its place the refusal reports it instead of losing it.
//!
//! Every fixture is flat to roundoff at the neutral seed ρ = 0, so that seed certifies in zero
//! iterations at a flat top, while a search started below it stops lower without certifying.
//! The plan runner declines the flat top and continues from the state that beat it.
//! - A softplus knee over a gently curved slope: the continuation cannot certify, and the fit
//!   refuses with `DominanceUnresolved`.
//! - A Gaussian well: a budget one iteration short of the unbounded search lets the
//!   continuation certify the centre, which publishes.
//! - The well plus a concave ridge along a second coordinate: the continuation certifies a
//!   strict saddle whose escape cannot run, and the fit refuses with
//!   `IncumbentUnescapableSaddle`, or propagates the escape search's fatal failure.

use super::*;
use gam_problem::DominanceRefusalKind;
use ndarray::array;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Off the integer seed lattice, so no generated seed starts at the centre.
const CENTER: f64 = -3.5;
const WIDTH: f64 = 0.5;
const DEPTH: f64 = 10.0;
/// Inside the well's convex core and off its centre.
const WELL_START: f64 = -3.9;

fn well_value(x: f64) -> f64 {
    -DEPTH * (-(x - CENTER).powi(2) / (2.0 * WIDTH * WIDTH)).exp()
}

fn well_derivative(x: f64) -> f64 {
    -well_value(x) * (x - CENTER) / (WIDTH * WIDTH)
}

/// Search the well from `WELL_START` under `max_iter`, the neutral seed next in the cascade,
/// against a cache of its own so no run resumes another's checkpoint.
fn run_well(max_iter: usize, label: &str) -> Result<OuterResult, EstimationError> {
    let (_cache_dir, session) = tmp_cache_session(label);
    let problem = OuterProblem::new(1)
        .with_gradient(Derivative::Analytic)
        .with_hessian(DeclaredHessianForm::Unavailable)
        .with_bounds(array![-6.0], array![6.0])
        .with_initial_rho(array![WELL_START])
        .with_screen_initial_rho(false)
        .with_seed_config(gam_problem::SeedConfig {
            max_seeds: 1,
            seed_budget: 1,
            risk_profile: gam_problem::SeedRiskProfile::Gaussian,
            ..Default::default()
        })
        .with_max_iter(max_iter)
        .with_cache_session(session);
    let mut objective = problem.build_objective(
        (),
        |_: &mut (), rho: &Array1<f64>| Ok(well_value(rho[0])),
        |_: &mut (), rho: &Array1<f64>| {
            Ok(OuterEval {
                cost: well_value(rho[0]),
                gradient: array![well_derivative(rho[0])],
                hessian: HessianValue::Unavailable,
                inner_beta_hint: None,
            })
        },
        None::<fn(&mut ())>,
        None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
    );
    problem.run(&mut objective, label)
}

// The refusal fixture: a softplus knee with a gentle curvature below it,
// f = −S·q + ε·q²/2 with q = W·ln(1 + e^{−(x − K)/W}) ≈ max(0, K − x). It is flat to roundoff at
// the neutral seed, and below the knee it falls with a gradient S − ε·q that barely changes along a
// search. A one-iteration search there stops low without certifying. Its run-recorded gradient and
// the certificate-time gradient then agree to about ε per unit moved, so the certificate's
// gradient-reproducibility floor cannot widen the band past the checkpoint's own gradient, and the
// checkpoint is refused on that gradient. Below the knee the minimum sits beyond the declared
// lower bound, so no search reaches a certified point there.

const KNEE: f64 = -1.0;
const KNEE_WIDTH: f64 = 0.05;
const SLOPE: f64 = 1.0;
const SLOPE_CURVATURE: f64 = 0.01;
const SLOPE_START: f64 = -2.0;

fn softplus(z: f64) -> f64 {
    if z > 0.0 {
        z + (-z).exp().ln_1p()
    } else {
        z.exp().ln_1p()
    }
}

fn logistic(z: f64) -> f64 {
    if z >= 0.0 {
        1.0 / (1.0 + (-z).exp())
    } else {
        let e = z.exp();
        e / (1.0 + e)
    }
}

fn knee_depth(x: f64) -> f64 {
    KNEE_WIDTH * softplus(-(x - KNEE) / KNEE_WIDTH)
}

fn slope_value(x: f64) -> f64 {
    let q = knee_depth(x);
    -SLOPE * q + 0.5 * SLOPE_CURVATURE * q * q
}

fn slope_derivative(x: f64) -> f64 {
    logistic(-(x - KNEE) / KNEE_WIDTH) * (SLOPE - SLOPE_CURVATURE * knee_depth(x))
}

/// Search the slope from `SLOPE_START` under `max_iter`, the neutral seed next in the cascade,
/// against a cache of its own.
fn run_slope(max_iter: usize, label: &str) -> Result<OuterResult, EstimationError> {
    let (_cache_dir, session) = tmp_cache_session(label);
    let problem = OuterProblem::new(1)
        .with_gradient(Derivative::Analytic)
        .with_hessian(DeclaredHessianForm::Unavailable)
        .with_bounds(array![-50.0], array![6.0])
        .with_initial_rho(array![SLOPE_START])
        .with_screen_initial_rho(false)
        .with_seed_config(gam_problem::SeedConfig {
            max_seeds: 1,
            seed_budget: 1,
            risk_profile: gam_problem::SeedRiskProfile::Gaussian,
            ..Default::default()
        })
        .with_max_iter(max_iter)
        .with_cache_session(session);
    let mut objective = problem.build_objective(
        (),
        |_: &mut (), rho: &Array1<f64>| Ok(slope_value(rho[0])),
        |_: &mut (), rho: &Array1<f64>| {
            Ok(OuterEval {
                cost: slope_value(rho[0]),
                gradient: array![slope_derivative(rho[0])],
                hessian: HessianValue::Unavailable,
                inner_beta_hint: None,
            })
        },
        None::<fn(&mut ())>,
        None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
    );
    problem.run(&mut objective, label)
}

#[test]
fn a_declined_certified_optimum_is_reported_when_nothing_certifies_in_its_place_2953() {
    let error = run_slope(1, "dominated plateau refusal #2953").expect_err(
        "a one-iteration search cannot certify on the slope, and the flat top it beats must \
         not publish",
    );
    let EstimationError::DominatedCertifiedPlateau {
        kind,
        plateau_rho,
        plateau_value,
        incumbent_rho,
        incumbent_value,
        gap,
        band,
        continuation,
        terminal_refusal,
        ..
    } = error
    else {
        panic!("expected the typed dominated-plateau refusal, got {error}");
    };
    assert_eq!(kind, DominanceRefusalKind::DominanceUnresolved);
    // The declined optimum is the neutral seed, certified where the knee is flat to roundoff.
    assert_eq!(plateau_rho, vec![0.0]);
    assert_eq!(plateau_value.to_bits(), slope_value(0.0).to_bits());
    // The checkpoint the terminal certificate refused is on the slope, off the declared bound.
    assert!(
        incumbent_rho[0] < KNEE - 10.0 * KNEE_WIDTH && incumbent_rho[0] > -49.0,
        "the refused checkpoint must be the search on the slope, off the bound; \
         rho={incumbent_rho:?}"
    );
    assert!(
        incumbent_value < plateau_value - 0.5,
        "the refused checkpoint {incumbent_value:e} must sit well below the declined optimum \
         {plateau_value:e}"
    );
    // The decline was judged beyond the rounding envelope, by a gap of the slope's scale.
    assert!(
        band > 0.0 && gap > 0.5 && gap > band,
        "gap {gap:e} against band {band:e}"
    );
    assert!(
        continuation.starts_with("exhausted at objective")
            || continuation.starts_with("declined another certified optimum at objective"),
        "the continuation from the state that beat the optimum must have run without \
         certifying: {continuation}"
    );
    assert!(
        matches!(*terminal_refusal, EstimationError::RemlDidNotConverge { .. }),
        "the terminal certificate's own refusal must ride inside the report: {terminal_refusal}"
    );
}

#[test]
fn a_continuation_that_certifies_publishes_after_the_decline_2953() {
    // The same search without a binding budget certifies the centre, and its iteration count
    // sizes a budget that stops the first search one iteration short of it.
    let calibration = run_well(200, "dominated plateau calibration #2953")
        .expect("an unbounded search from inside the well certifies its centre");
    let unbounded_iterations = calibration.iterations;
    assert!(
        unbounded_iterations >= 2,
        "the well search must take at least two iterations for a budget to split it; took \
         {unbounded_iterations}"
    );
    let budget = unbounded_iterations - 1;
    let published = run_well(budget, "dominated plateau continuation #2953").unwrap_or_else(
        |error| {
            panic!(
                "the search continued from the state that beat the declined optimum must \
                 certify and publish: {error}"
            )
        },
    );
    assert!(
        (published.rho[0] - CENTER).abs() < 1.0e-3,
        "the published optimum must be the well's centre; rho={:?}",
        published.rho
    );
    // The first search stopped at `budget` iterations and the neutral seed took none, so a
    // larger ledger is the continuation's own work.
    assert!(
        published.iterations > budget,
        "the published fit must come from the continuation, not the budget-capped first search: \
         {} iteration(s) against a budget of {budget}",
        published.iterations
    );
}

// The strict-saddle fixture: the same well in x, and concave curvature along y. The search
// never leaves the ridge y = 0, because the gradient along y is exactly zero there, so the
// continuation certifies the well's centre under the first-order screening certificate and
// the terminal certificate refuses that point as a strict saddle. Its escape descends along
// y, and the objective refuses every evaluation once an off-ridge evaluation has been
// followed by a reset, which first happens when the certify loop starts the escape's search.

const RIDGE_CURVATURE: f64 = 4.0;
const ARMED_MARKER: &str = "the #2953 ridge fixture refuses evaluation after the saddle escape";

fn well_curvature(x: f64) -> f64 {
    let u = x - CENTER;
    -well_value(x) * (1.0 / (WIDTH * WIDTH) - u * u / WIDTH.powi(4))
}

fn ridge_value(rho: &Array1<f64>) -> f64 {
    well_value(rho[0]) - 0.5 * RIDGE_CURVATURE * rho[1] * rho[1]
}

/// What the ridge objective raises once armed.
#[derive(Clone, Copy)]
enum ArmedRefusal {
    /// A refusal of the trial point, which a search walks away from.
    TrialPoint,
    /// A failure of the evaluation itself, which stops the fit.
    Fatal,
}

struct RidgeState {
    refusal: Option<ArmedRefusal>,
    off_ridge: bool,
    armed: bool,
    armed_refusals: Arc<AtomicUsize>,
}

fn ridge_refusal(state: &mut RidgeState, rho: &Array1<f64>) -> Result<(), EstimationError> {
    if state.armed
        && let Some(refusal) = state.refusal
    {
        state.armed_refusals.fetch_add(1, Ordering::Relaxed);
        return Err(match refusal {
            ArmedRefusal::TrialPoint => EstimationError::TrialPointRefused {
                reason: ARMED_MARKER.to_string(),
            },
            ArmedRefusal::Fatal => EstimationError::fatal_outer_evaluation(
                "ridge fixture",
                EstimationError::InvalidInput(ARMED_MARKER.to_string()),
            ),
        });
    }
    if rho[1] != 0.0 {
        state.off_ridge = true;
    }
    Ok(())
}

/// The ridge problem from `WELL_START` on the ridge under `max_iter`: a gradient-only search
/// over a declared analytic Hessian, which the terminal certificate measures.
fn ridge_problem(max_iter: usize, label: &str) -> (tempfile::TempDir, OuterProblem) {
    let (cache_dir, session) = tmp_cache_session(label);
    let problem = OuterProblem::new(2)
        .with_gradient(Derivative::Analytic)
        .with_hessian(DeclaredHessianForm::Dense)
        .with_prefer_gradient_only(true)
        .with_fallback_policy(FallbackPolicy::Disabled)
        .with_bounds(array![-6.0, -6.0], array![6.0, 6.0])
        .with_initial_rho(array![WELL_START, 0.0])
        .with_screen_initial_rho(false)
        .with_seed_config(gam_problem::SeedConfig {
            max_seeds: 1,
            seed_budget: 1,
            risk_profile: gam_problem::SeedRiskProfile::Gaussian,
            ..Default::default()
        })
        .with_max_iter(max_iter)
        .with_cache_session(session);
    (cache_dir, problem)
}

macro_rules! ridge_objective {
    ($problem:expr, $refusal:expr, $armed_refusals:expr) => {
        $problem.build_objective(
            RidgeState {
                refusal: $refusal,
                off_ridge: false,
                armed: false,
                armed_refusals: $armed_refusals,
            },
            |state: &mut RidgeState, rho: &Array1<f64>| {
                ridge_refusal(state, rho)?;
                Ok(ridge_value(rho))
            },
            |state: &mut RidgeState, rho: &Array1<f64>| {
                ridge_refusal(state, rho)?;
                Ok(OuterEval {
                    cost: ridge_value(rho),
                    gradient: array![well_derivative(rho[0]), -RIDGE_CURVATURE * rho[1]],
                    hessian: HessianValue::Dense(array![
                        [well_curvature(rho[0]), 0.0],
                        [0.0, -RIDGE_CURVATURE]
                    ]),
                    inner_beta_hint: None,
                })
            },
            Some(|state: &mut RidgeState| {
                if state.off_ridge {
                    state.armed = true;
                }
            }),
            None::<fn(&mut RidgeState, &Array1<f64>) -> Result<EfsEval, EstimationError>>,
        )
    };
}

/// The ridge search from `WELL_START` without a binding budget, judged by the screening
/// certificate alone: the certified result, or `None` when it does not certify. The callers
/// check it, since a helper outside a `#[test]` does not panic.
fn ridge_screening_certified() -> Result<Option<OuterResult>, EstimationError> {
    let (_cache_dir, problem) = ridge_problem(200, "ridge calibration #2953");
    let mut objective = ridge_objective!(problem, None, Arc::new(AtomicUsize::new(0)));
    let config = problem.config();
    let capability = objective.capability();
    let the_plan = plan(&capability);
    Ok(
        match run_outer_with_plan(
            &mut objective,
            &config,
            "ridge calibration #2953",
            &capability,
            &the_plan,
            false,
        )? {
            PlanRunOutcome::Converged(result) => Some(result),
            _ => None,
        },
    )
}

#[test]
fn a_declined_optimum_beaten_by_an_unescapable_strict_saddle_is_reported_by_kind_2953() {
    let calibration = ridge_screening_certified()
        .expect("the ridge search from inside the well must run")
        .expect(
            "an unbounded ridge search from inside the well must certify its centre under the \
             screening certificate",
        );
    assert!(
        (calibration.rho[0] - CENTER).abs() < 1.0e-3 && calibration.rho[1] == 0.0,
        "the unbounded ridge search must certify the saddle at the well's centre; rho={:?}",
        calibration.rho
    );
    let unbounded_iterations = calibration.iterations;
    assert!(
        unbounded_iterations >= 2,
        "the ridge search must take at least two iterations for a budget to split it; took \
         {unbounded_iterations}"
    );
    let armed_refusals = Arc::new(AtomicUsize::new(0));
    let (_cache_dir, problem) =
        ridge_problem(unbounded_iterations - 1, "ridge saddle refusal #2953");
    let mut objective = ridge_objective!(
        problem,
        Some(ArmedRefusal::TrialPoint),
        Arc::clone(&armed_refusals)
    );
    let error = problem
        .run(&mut objective, "ridge saddle refusal #2953")
        .expect_err("a strict saddle whose escape cannot run must not publish");
    let EstimationError::DominatedCertifiedPlateau {
        kind,
        plateau_rho,
        incumbent_rho,
        gap,
        band,
        continuation,
        terminal_refusal,
        ..
    } = error
    else {
        panic!("expected the typed dominated-plateau refusal, got {error}");
    };
    assert_eq!(kind, DominanceRefusalKind::IncumbentUnescapableSaddle);
    assert_eq!(plateau_rho, vec![0.0, 0.0]);
    assert!(
        (incumbent_rho[0] - CENTER).abs() < 1.0e-3 && incumbent_rho[1] == 0.0,
        "the refused checkpoint must be the saddle at the well's centre on the ridge; \
         rho={incumbent_rho:?}"
    );
    assert!(
        band > 0.0 && gap > DEPTH / 2.0 && gap > band,
        "gap {gap:e} against band {band:e}"
    );
    assert!(
        continuation.starts_with("certified at objective"),
        "the continuation must have certified the saddle under the screening certificate: \
         {continuation}"
    );
    assert!(
        armed_refusals.load(Ordering::Relaxed) > 0,
        "the saddle escape must have armed the objective, so its search could not run"
    );
    assert!(
        terminal_refusal
            .to_string()
            .contains("SaddleEscape reseed could not run"),
        "the terminal refusal must carry why the escape was not taken: {terminal_refusal}"
    );
}

#[test]
fn a_fatal_failure_of_the_saddle_escape_search_propagates_as_it_is_2953() {
    let calibration = ridge_screening_certified()
        .expect("the ridge search from inside the well must run")
        .expect(
            "an unbounded ridge search from inside the well must certify its centre under the \
             screening certificate",
        );
    assert!(
        (calibration.rho[0] - CENTER).abs() < 1.0e-3 && calibration.rho[1] == 0.0,
        "the unbounded ridge search must certify the saddle at the well's centre; rho={:?}",
        calibration.rho
    );
    let unbounded_iterations = calibration.iterations;
    assert!(
        unbounded_iterations >= 2,
        "the ridge search must take at least two iterations for a budget to split it; took \
         {unbounded_iterations}"
    );
    let armed_refusals = Arc::new(AtomicUsize::new(0));
    let (_cache_dir, problem) =
        ridge_problem(unbounded_iterations - 1, "ridge fatal escape #2953");
    let mut objective =
        ridge_objective!(problem, Some(ArmedRefusal::Fatal), Arc::clone(&armed_refusals));
    let error = problem
        .run(&mut objective, "ridge fatal escape #2953")
        .expect_err("a fatal evaluation failure must stop the fit");
    assert!(
        error.is_fatal_outer_evaluation() && error.to_string().contains(ARMED_MARKER),
        "the escape search's fatal failure must be the refusal, not an older certification \
         refusal: {error}"
    );
    assert!(
        armed_refusals.load(Ordering::Relaxed) > 0,
        "the saddle escape must have armed the objective before the fatal failure"
    );
}
