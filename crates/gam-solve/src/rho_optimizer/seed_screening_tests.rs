//! Seed-screening cascade contract (gam#2943): a refused seed is rejected
//! while the others keep their ranking, a non-trial-point error stays fatal,
//! and screening never runs an uncapped inner solve.

use super::*;
use ndarray::array;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

/// What one screening evaluation at a seed returns.
#[derive(Clone, Copy)]
enum ProxyOutcome {
    Cost(f64),
    Refused,
    Fatal,
}

/// A one-coordinate objective whose screening proxy answers from a fixed
/// table per seed and records the inner cap in force at every call.
struct ScreeningTable {
    seeds: Vec<Array1<f64>>,
    outcomes: Vec<ProxyOutcome>,
    cap: Arc<AtomicUsize>,
    observed: Arc<Mutex<Vec<(usize, usize)>>>,
}

impl OuterObjective for ScreeningTable {
    fn capability(&self) -> OuterCapability {
        OuterCapability {
            gradient: Derivative::Analytic,
            hessian: DeclaredHessianForm::Dense,
            n_params: 1,
            psi_dim: 0,
            fixed_point_available: false,
            barrier_config: None,
            prefer_gradient_only: false,
            disable_fixed_point: false,
        }
    }

    fn eval_cost(&mut self, theta: &Array1<f64>) -> Result<f64, EstimationError> {
        Ok(theta.dot(theta))
    }

    fn eval_screening_proxy(&mut self, rho: &Array1<f64>) -> Result<f64, EstimationError> {
        let idx = self
            .seeds
            .iter()
            .position(|seed| seed == rho)
            .expect("screening evaluates only the seeds it was given");
        self.observed
            .lock()
            .expect("the observation log is not poisoned")
            .push((idx, self.cap.load(Ordering::Relaxed)));
        match self.outcomes[idx] {
            ProxyOutcome::Cost(cost) => Ok(cost),
            ProxyOutcome::Refused => Err(EstimationError::TrialPointRefused {
                reason: "no inner mode at this seed".to_string(),
            }),
            ProxyOutcome::Fatal => Err(EstimationError::InvalidInput(
                "malformed configuration".to_string(),
            )),
        }
    }

    fn eval(&mut self, theta: &Array1<f64>) -> Result<OuterEval, EstimationError> {
        Ok(OuterEval {
            cost: theta.dot(theta),
            gradient: 2.0 * theta,
            hessian: HessianValue::Dense(2.0 * Array2::<f64>::eye(theta.len())),
            inner_beta_hint: None,
        })
    }

    fn reset(&mut self) {}

    fn seed_inner_state(&mut self, beta: &Array1<f64>) -> Result<SeedOutcome, EstimationError> {
        Err(EstimationError::InvalidInput(format!(
            "seed screening must not seed inner state ({} coefficients)",
            beta.len()
        )))
    }
}

struct Screened {
    result: Result<Vec<Array1<f64>>, EstimationError>,
    seeds: Vec<Array1<f64>>,
    observed: Vec<(usize, usize)>,
    cap_after: usize,
}

/// Screens seeds ρ = 0, 1, 2, … inside the box [−10, 10], so no seed sits on
/// the over-smoothing boundary and the boundary demotion leaves the order alone.
fn screen(outcomes: &[ProxyOutcome]) -> Screened {
    let seeds: Vec<Array1<f64>> = (0..outcomes.len()).map(|i| array![i as f64]).collect();
    let cap = Arc::new(AtomicUsize::new(0));
    let observed = Arc::new(Mutex::new(Vec::new()));
    let mut obj = ScreeningTable {
        seeds: seeds.clone(),
        outcomes: outcomes.to_vec(),
        cap: Arc::clone(&cap),
        observed: Arc::clone(&observed),
    };
    let config = OuterConfig {
        screening_cap: Some(Arc::clone(&cap)),
        search_bounds_override: Some((array![-10.0], array![10.0])),
        ..OuterConfig::default()
    };
    let result = rank_seeds_with_screening(&mut obj, &config, "gam#2943 screening", &seeds);
    let observed = observed
        .lock()
        .expect("the observation log is not poisoned")
        .clone();
    Screened {
        result,
        seeds,
        observed,
        cap_after: cap.load(Ordering::Relaxed),
    }
}

#[test]
fn screening_rejects_a_refused_seed_and_ranks_the_others_2943() {
    let run = screen(&[
        ProxyOutcome::Cost(1.0),
        ProxyOutcome::Refused,
        ProxyOutcome::Cost(2.0),
    ]);
    let initial = gam_problem::SeedConfig::default()
        .screen_max_inner_iterations
        .max(1);
    assert_eq!(
        run.observed,
        vec![(0, initial), (1, initial), (2, initial)],
        "one stage at the initial cap: the finite costs end the cascade",
    );
    let ordered = run
        .result
        .expect("a trial-point refusal at one seed must not end screening");
    assert_eq!(
        ordered,
        vec![
            run.seeds[0].clone(),
            run.seeds[2].clone(),
            run.seeds[1].clone()
        ],
        "finite costs rank in cost order and the refused seed goes to the tail",
    );
}

#[test]
fn screening_keeps_a_non_trial_point_error_fatal_2943() {
    let run = screen(&[
        ProxyOutcome::Cost(1.0),
        ProxyOutcome::Fatal,
        ProxyOutcome::Cost(2.0),
    ]);
    let initial = gam_problem::SeedConfig::default()
        .screen_max_inner_iterations
        .max(1);
    assert_eq!(run.observed, vec![(0, initial), (1, initial)]);
    let error = run
        .result
        .expect_err("an error that is not about this seed must end screening");
    assert!(!error.is_trial_point_infeasible(), "{error}");
    assert_eq!(
        run.cap_after, 0,
        "screening restores the caller's inner cap"
    );
}

#[test]
fn screening_stops_after_the_capped_stages_2943() {
    let run = screen(&[
        ProxyOutcome::Cost(f64::NAN),
        ProxyOutcome::Refused,
        ProxyOutcome::Cost(f64::INFINITY),
    ]);
    let initial = gam_problem::SeedConfig::default()
        .screen_max_inner_iterations
        .max(1);
    let expected: Vec<(usize, usize)> = SEED_SCREENING_CASCADE_MULTIPLIERS
        .iter()
        .flat_map(|multiplier| (0..3).map(move |idx| (idx, initial * multiplier)))
        .collect();
    assert_eq!(
        run.observed, expected,
        "every seed is screened once per capped stage and never uncapped (cap 0)",
    );
    assert_eq!(
        run.result
            .expect("an unranked screen keeps the generated order"),
        run.seeds,
    );
    assert_eq!(
        run.cap_after, 0,
        "screening restores the caller's inner cap"
    );
}
