// Child module of `run_plan::run_plan_tests` (see the `#[path]` declaration
// there): #2334 terminal-coefficient-mode reset for cap-less mode owners, and
// the #2357/#2155 interior and railed strict-saddle escape. Scope comes from
// the parent via `use super::*`; the split is purely physical.

use super::*;
use ndarray::array;

// ─── #2334 terminal-coefficient-mode reset for cap-less mode owners ──
//
// The terminal certification sequence installs the selected point twice at
// `result.rho`: once via `finalize_outer_result` (where a mode-owning
// objective installs its owned coefficient mode) and once via the analytic
// re-eval in `certify_outer_optimality` (which sets `result.final_value`). On
// a nonconvex / bimodal inner solve those two evaluations can land in
// DIFFERENT coefficient basins unless each re-installs from the same clean
// baseline through `reset()`. That reset was gated solely on
// `config.outer_inner_cap.is_some()`, which the custom-family fit never sets
// (it holds its inner cap in a different field), so the reset never fired and
// the downstream bitwise bind `terminal_mode.objective == final_value` could
// spuriously fail. `owns_terminal_coefficient_mode()` now forces that reset
// independently of the cap.

/// A deterministic stand-in for a warm-start-sensitive bimodal inner solve.
///
/// Each derivative-bearing terminal evaluation (`eval` / `eval_with_order` /
/// `finalize`) reads a warm-parity flag: in the "bumped" parity the inner
/// solve lands in a basin whose objective is offset by `BASIN_BUMP`; every
/// such evaluation flips the parity, modeling an inner solve whose consecutive
/// warm-started solves alternate basins. `reset()` re-baselines the parity to
/// the un-bumped basin. The outer gradient is basin-independent (`= ρ`), so
/// the fit still certifies stationarity at ρ = 0 regardless of basin — exactly
/// the real defect's shape, where the two inner optima share the outer
/// gradient but differ in objective value.
///
/// Consequence: with NO terminal reset, `finalize` and the certifying re-eval
/// are one flip apart, so their objective values differ by `BASIN_BUMP`
/// (whole-basin disagreement, not roundoff). With the terminal reset, both
/// start from the un-bumped baseline and agree bitwise.
struct BimodalTerminalObjective {
    owns_terminal: bool,
    warm_bumped: Arc<Mutex<bool>>,
    finalize_installed: Arc<Mutex<Option<f64>>>,
}

const BASIN_BUMP: f64 = 2.6; // ~ the observed terminal(9.1931e2) − certified(9.1671e2) gap

impl BimodalTerminalObjective {
    /// The objective in the CURRENTLY installed basin, without changing it.
    ///
    /// Reading the installed state is not a solve, so it must not flip the
    /// parity — and, critically, every lane reading the same instant must see
    /// the SAME basin. A mock whose value lane and derivative lane disagree at
    /// one ρ is not a bimodal inner solve; it is a value/gradient desync, and
    /// the terminal certificate's same-ρ value-agreement audit refuses it on
    /// sight (measured: `value-only=0.0, analytic-sample=2.6,
    /// disagreement=2.600e0, roundoff bound=3.874e-8`). That refusal is the
    /// audit working correctly; it was the fixture that was modelling the wrong
    /// thing.
    fn basin_value(&self, rho: &Array1<f64>) -> f64 {
        let bump = if *self
            .warm_bumped
            .lock()
            .expect("the warm_bumped mutex is not poisoned")
        {
            BASIN_BUMP
        } else {
            0.0
        };
        0.5 * rho.dot(rho) + bump
    }

    /// A warm-started SOLVE: reads the current basin and lands the next one in
    /// the other basin. This is what makes two consecutive installations
    /// disagree unless a `reset()` re-baselines between them.
    fn basin_solve(&self, rho: &Array1<f64>) -> OuterEval {
        let cost = self.basin_value(rho);
        let mut warm = self
            .warm_bumped
            .lock()
            .expect("the warm_bumped mutex is not poisoned");
        *warm = !*warm; // consecutive warm-started solves alternate basins
        OuterEval {
            cost,
            gradient: rho.clone(),
            hessian: HessianValue::Unavailable,
            inner_beta_hint: None,
        }
    }
}

impl OuterObjective for BimodalTerminalObjective {
    fn capability(&self) -> OuterCapability {
        OuterCapability {
            gradient: Derivative::Analytic,
            hessian: DeclaredHessianForm::Unavailable,
            n_params: 1,
            psi_dim: 0,
            fixed_point_available: false,
            barrier_config: None,
            prefer_gradient_only: false,
            disable_fixed_point: true,
        }
    }
    fn eval_cost(&mut self, rho: &Array1<f64>) -> Result<f64, EstimationError> {
        // Reads the installed basin; does not solve, so it does not flip.
        Ok(self.basin_value(rho))
    }
    fn eval(&mut self, rho: &Array1<f64>) -> Result<OuterEval, EstimationError> {
        Ok(self.basin_solve(rho))
    }
    fn eval_with_order(
        &mut self,
        rho: &Array1<f64>,
        order: OuterEvalOrder,
    ) -> Result<OuterEval, EstimationError> {
        match order {
            // The value-only lane is the scalar authority sampling the state
            // that is already installed — the same read `eval_cost` performs.
            // It must agree with the derivative-bearing sample taken at the
            // same ρ, or the fixture is a desync rather than a bimodal solve.
            OuterEvalOrder::Value => Ok(OuterEval {
                cost: self.basin_value(rho),
                gradient: rho.clone(),
                hessian: HessianValue::Unavailable,
                inner_beta_hint: None,
            }),
            // Derivative-bearing orders are warm-started solves and DO advance
            // the basin, which is what leaves `finalize` and the certifying
            // re-eval one flip apart when no reset intervenes.
            OuterEvalOrder::ValueAndGradient | OuterEvalOrder::ValueGradientHessian => {
                Ok(self.basin_solve(rho))
            }
        }
    }
    fn finalize_outer_result(
        &mut self,
        rho: &Array1<f64>,
        plan: &OuterPlan,
    ) -> Result<(), EstimationError> {
        // This fixture's whole premise is the warm-started, DERIVATIVE-bearing
        // inner solve of `basin_solve` (see `eval_with_order`). A solver that
        // never requests derivatives would not exercise the parity flip these
        // tests measure, so refuse it rather than record an install whose
        // meaning the assertions do not cover.
        if matches!(plan.solver, Solver::Efs | Solver::HybridEfs) {
            return Err(EstimationError::RemlOptimizationFailed(format!(
                "bimodal terminal fixture reached finalize under solver {:?}, which never \
                 requests a derivative-bearing evaluation",
                plan.solver
            )));
        }
        // Install the owned coefficient mode: record the objective value the
        // mode carries, exactly as the custom-family evaluator does.
        let installed = self.basin_solve(rho).cost;
        *self
            .finalize_installed
            .lock()
            .expect("the finalize_installed mutex is not poisoned") = Some(installed);
        Ok(())
    }
    fn owns_terminal_coefficient_mode(&self) -> bool {
        self.owns_terminal
    }
    fn reset(&mut self) {
        *self
            .warm_bumped
            .lock()
            .expect("the warm_bumped mutex is not poisoned") = false;
    }
    fn seed_inner_state(&mut self, beta: &Array1<f64>) -> Result<SeedOutcome, EstimationError> {
        // The bimodal basin is carried by `warm_bumped`, not an inner-β slot —
        // but the offered seed must still be finite, or the driver handed this
        // fixture a state no warm start could legitimately resume from.
        if beta.iter().any(|value| !value.is_finite()) {
            return Err(EstimationError::RemlOptimizationFailed(format!(
                "bimodal terminal fixture was offered a non-finite inner seed of length {}",
                beta.len()
            )));
        }
        Ok(SeedOutcome::NoSlot)
    }
}

fn run_bimodal_terminal(owns_terminal: bool) -> (f64, f64) {
    let warm_bumped = Arc::new(Mutex::new(false));
    let finalize_installed = Arc::new(Mutex::new(None));
    let mut obj = BimodalTerminalObjective {
        owns_terminal,
        warm_bumped: Arc::clone(&warm_bumped),
        finalize_installed: Arc::clone(&finalize_installed),
    };
    // Seed AT the stationary point (∇ = ρ = 0) so the outer search certifies
    // at iteration 0 and control passes straight into the terminal sequence.
    let problem = OuterProblem::new(1)
        .with_gradient(Derivative::Analytic)
        .with_hessian(DeclaredHessianForm::Unavailable)
        .with_initial_rho(array![0.0])
        .with_screen_initial_rho(false)
        .with_seed_config(gam_problem::SeedConfig {
            max_seeds: 1,
            seed_budget: 1,
            ..Default::default()
        });
    let result = problem
        .run(&mut obj, "bimodal-terminal")
        .expect("stationary seed must certify");
    assert!(result.converged(), "stationary seed must converge");
    let installed = finalize_installed
        .lock()
        .expect("the finalize_installed mutex is not poisoned")
        .expect("finalize must have installed a terminal mode");
    (installed, result.final_value)
}

#[test]
fn terminal_reset_binds_bimodal_mode_owner_bitwise() {
    // WITH the ownership flag: the terminal reset fires before BOTH the
    // finalize install and the certifying re-eval, so the mode's objective and
    // the certified value come from the same un-bumped baseline — bitwise equal.
    let (installed, final_value) = run_bimodal_terminal(true);
    assert_eq!(
        installed.to_bits(),
        final_value.to_bits(),
        "terminal-coefficient-mode owner: finalize-installed objective ({installed:.17e}) \
         must bitwise-match the certified final_value ({final_value:.17e})",
    );
}

#[test]
fn without_ownership_flag_bimodal_terminal_bind_fails() {
    // WITHOUT the flag (and no outer_inner_cap wired, exactly the custom-family
    // situation): no terminal reset, so finalize and the certifying re-eval are
    // one warm-parity flip apart and settle in different basins — a whole-basin
    // bitwise mismatch, i.e. the spurious bind failure this fix removes.
    let (installed, final_value) = run_bimodal_terminal(false);
    assert_ne!(
        installed.to_bits(),
        final_value.to_bits(),
        "control: without the ownership flag the bimodal finalize/certify pair must disagree",
    );
    assert!(
        (installed - final_value).abs() >= BASIN_BUMP - 1e-9,
        "the disagreement must be a whole-basin gap (~{BASIN_BUMP}), not roundoff: \
         installed={installed:.17e} final_value={final_value:.17e}",
    );
}

// ─── #2357 interior strict-saddle escape ──────────────────────────

// A 2-D outer objective with a genuine interior saddle at ρ=(0,0) and a pair of
// PSD minima at ρ=(0,±1):
//
//   f(ρ) = ½·ρ₀²  +  ¼·ρ₁⁴ − ½·ρ₁²
//   ∇f    = (ρ₀,  ρ₁³ − ρ₁)
//   H     = [[1, 0], [0, 3ρ₁² − 1]]
//
// At ρ=(0,0): ∇=0 (first-order stationary) but H=diag(1,−1) is INDEFINITE — a
// strict saddle. The two minima ρ₁=±1 carry H=diag(1,2) (PSD) and f=−¼. A
// gradient-only convergence gate that arrives at the saddle stops there; only
// the certified negative-curvature escape reaches a minimum. This is the
// synthetic distillation of the periodic-te() cold-start refusal in #2357.
fn saddle_cost(rho: &Array1<f64>) -> f64 {
    let r0 = rho[0];
    let r1 = rho[1];
    0.5 * r0 * r0 + 0.25 * r1 * r1 * r1 * r1 - 0.5 * r1 * r1
}

fn saddle_eval(rho: &Array1<f64>) -> OuterEval {
    let r0 = rho[0];
    let r1 = rho[1];
    OuterEval {
        cost: saddle_cost(rho),
        gradient: array![r0, r1 * r1 * r1 - r1],
        hessian: HessianValue::Dense(array![[1.0, 0.0], [0.0, 3.0 * r1 * r1 - 1.0]]),
        inner_beta_hint: None,
    }
}

fn saddle_problem() -> OuterProblem {
    OuterProblem::new(2)
        .with_gradient(Derivative::Analytic)
        .with_hessian(DeclaredHessianForm::Dense)
        .with_initial_rho(array![0.0, 0.0])
        .with_screen_initial_rho(false)
        .with_seed_config(gam_problem::SeedConfig {
            max_seeds: 1,
            seed_budget: 1,
            ..Default::default()
        })
}

#[test]
fn certify_mints_saddle_escape_reseed_at_interior_saddle() {
    // Directly audit the saddle point: the certificate must refuse it for
    // indefinite interior curvature AND publish a negative-curvature escape
    // reseed strictly BELOW the saddle objective, moving along the indefinite
    // axis rather than the positive-curvature one (#2357).
    let problem = saddle_problem();
    let mut obj = problem.build_objective(
        (),
        |_: &mut (), rho: &Array1<f64>| Ok(saddle_cost(rho)),
        |_: &mut (), rho: &Array1<f64>| Ok(saddle_eval(rho)),
        None::<fn(&mut ())>,
        None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
    );
    let rejection = audit_stationary_point(&mut obj, array![0.0, 0.0], "saddle-escape #2357")
        .expect_err("an interior strict saddle must be refused, not certified");
    let result = &rejection.result;
    let cert = result
        .criterion_certificate
        .as_ref()
        .expect("refused certificate must be recorded");
    assert!(
        cert.is_stationary(),
        "the gradient must clear the stationarity band at the saddle"
    );
    assert_eq!(
        cert.hessian_psd(),
        Some(false),
        "the saddle Hessian must read indefinite"
    );
    assert!(
        cert.lambdas_railed.is_empty(),
        "the saddle is interior, not railed"
    );
    let reseed = result
        .saddle_escape_reseed
        .as_ref()
        .expect("a refused interior saddle must mint a negative-curvature escape reseed");
    let saddle_f = saddle_cost(&array![0.0, 0.0]);
    assert!(
        saddle_cost(reseed) < saddle_f - 1e-9,
        "escape reseed {reseed:?} (f={}) must strictly descend below the saddle f={saddle_f}",
        saddle_cost(reseed),
    );
    assert!(
        reseed[1].abs() > 1e-6,
        "escape must step along the indefinite ρ₁ axis, not the PSD ρ₀ axis: {reseed:?}"
    );
}

// #2155 — the SAME double-well saddle in (ρ₀, ρ₁), plus a third coordinate ρ₂
// pulled by a constant negative gradient onto the +rho_bound box rail. At the
// railed point the FULL Hessian is diag(1, −1, 0), but the load-bearing verdict
// is on the REDUCED (off-railed) sub-block diag(1, −1) over {ρ₀, ρ₁}, which is
// indefinite. This is the flexible-link binomial-wiggle failure genus (#2155):
// a genuine interior saddle in the free directions while a smoothing coordinate
// rails, where the #2357 escape used to be waived by the mere presence of a
// rail (`lambdas_railed.is_empty()`), so the fit was refused despite a real,
// box-feasible descent along ρ₁ that holds ρ₂ fixed on its rail.
fn railed_saddle_cost(rho: &Array1<f64>) -> f64 {
    let r0 = rho[0];
    let r1 = rho[1];
    let r2 = rho[2];
    0.5 * r0 * r0 + 0.25 * r1 * r1 * r1 * r1 - 0.5 * r1 * r1 - 0.3 * r2
}

fn railed_saddle_eval(rho: &Array1<f64>) -> OuterEval {
    let r0 = rho[0];
    let r1 = rho[1];
    OuterEval {
        cost: railed_saddle_cost(rho),
        // ∂/∂ρ₂ = −0.3: a constant outward pull that drives ρ₂ to the +rho_bound
        // rail and keeps it there. ρ₂ carries zero curvature, so the full Hessian
        // is only semidefinite; the indefinite direction lives entirely in the
        // un-railed {ρ₀, ρ₁} block.
        gradient: array![r0, r1 * r1 * r1 - r1, -0.3],
        hessian: HessianValue::Dense(array![
            [1.0, 0.0, 0.0],
            [0.0, 3.0 * r1 * r1 - 1.0, 0.0],
            [0.0, 0.0, 0.0],
        ]),
        inner_beta_hint: None,
    }
}

fn railed_saddle_problem() -> OuterProblem {
    // Default box is ±rho_bound = ±30, so ρ₂ = 30 sits exactly on the upper rail.
    OuterProblem::new(3)
        .with_gradient(Derivative::Analytic)
        .with_hessian(DeclaredHessianForm::Dense)
        .with_initial_rho(array![0.0, 0.0, 30.0])
        .with_screen_initial_rho(false)
        .with_seed_config(gam_problem::SeedConfig {
            max_seeds: 1,
            seed_budget: 1,
            ..Default::default()
        })
}

#[test]
fn certify_mints_saddle_escape_reseed_at_railed_saddle_2155() {
    // A saddle whose indefinite curvature is confined to the free (un-railed)
    // directions must STILL be refused and STILL publish a negative-curvature
    // escape reseed — one that steps along the interior indefinite axis (ρ₁)
    // while holding the railed coordinate (ρ₂) fixed on its bound (#2155).
    let problem = railed_saddle_problem();
    let mut obj = problem.build_objective(
        (),
        |_: &mut (), rho: &Array1<f64>| Ok(railed_saddle_cost(rho)),
        |_: &mut (), rho: &Array1<f64>| Ok(railed_saddle_eval(rho)),
        None::<fn(&mut ())>,
        None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
    );
    let rejection =
        audit_stationary_point(&mut obj, array![0.0, 0.0, 30.0], "railed-saddle-escape #2155")
            .expect_err("a railed strict saddle in the free directions must be refused");
    let result = &rejection.result;
    let cert = result
        .criterion_certificate
        .as_ref()
        .expect("refused certificate must be recorded");
    assert!(
        cert.is_stationary(),
        "the interior gradient must clear the stationarity band at the railed saddle"
    );
    assert_eq!(
        cert.hessian_psd(),
        Some(false),
        "the REDUCED (off-railed) Hessian must read indefinite"
    );
    assert!(
        cert.lambdas_railed.contains(&2),
        "ρ₂ must be recorded as railed: {:?}",
        cert.lambdas_railed
    );
    let reseed = result.saddle_escape_reseed.as_ref().expect(
        "a refused railed saddle with an indefinite reduced Hessian must mint a reseed (#2155)",
    );
    let saddle_f = railed_saddle_cost(&array![0.0, 0.0, 30.0]);
    assert!(
        railed_saddle_cost(reseed) < saddle_f - 1e-9,
        "escape reseed {reseed:?} (f={}) must strictly descend below the saddle f={saddle_f}",
        railed_saddle_cost(reseed),
    );
    assert!(
        reseed[1].abs() > 1e-6,
        "escape must step along the free indefinite ρ₁ axis: {reseed:?}"
    );
    assert!(
        (reseed[2] - 30.0).abs() < 1e-12,
        "escape must hold the railed ρ₂ fixed on its bound: {reseed:?}"
    );
}

#[test]
fn outer_search_escapes_railed_saddle_and_certifies_minimum_2155() {
    // Full pipeline: seeded at the railed saddle, the outer search must escape
    // via the one-shot reseed and certify a genuine reduced-PSD minimum at
    // ρ₁=±1 with ρ₂ still on its rail — not refuse at the saddle (#2155).
    let problem = railed_saddle_problem();
    let mut obj = problem.build_objective(
        (),
        |_: &mut (), rho: &Array1<f64>| Ok(railed_saddle_cost(rho)),
        |_: &mut (), rho: &Array1<f64>| Ok(railed_saddle_eval(rho)),
        None::<fn(&mut ())>,
        None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
    );
    let result = problem
        .run(&mut obj, "railed-saddle-escape pipeline #2155")
        .expect("the outer search must escape the railed saddle and certify a minimum");
    assert!(
        result.converged(),
        "must converge at a reduced-PSD minimum, not refuse at the railed saddle: rho={:?}",
        result.rho
    );
    assert!(
        (result.rho[1].abs() - 1.0).abs() < 1e-4,
        "must land on the free-direction minimum ρ₁=±1, got ρ={:?}",
        result.rho,
    );
    assert!(
        (result.rho[2] - 30.0).abs() < 1e-4,
        "ρ₂ must remain on its rail at the minimum: {:?}",
        result.rho,
    );
    assert!(
        result
            .criterion_certificate
            .as_ref()
            .is_some_and(|c| c.certifies() && c.hessian_psd() == Some(true)),
        "the escaped minimum must carry a reduced-PSD certificate",
    );
}

#[test]
fn outer_search_escapes_interior_saddle_and_certifies_minimum() {
    // Full pipeline: seeded AT the saddle, the outer search must escape via the
    // one-shot negative-curvature reseed and certify a genuine PSD minimum at
    // ρ₁=±1 — not refuse the whole fit at the saddle (#2357).
    let problem = saddle_problem();
    let mut obj = problem.build_objective(
        (),
        |_: &mut (), rho: &Array1<f64>| Ok(saddle_cost(rho)),
        |_: &mut (), rho: &Array1<f64>| Ok(saddle_eval(rho)),
        None::<fn(&mut ())>,
        None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
    );
    let result = problem
        .run(&mut obj, "saddle-escape pipeline #2357")
        .expect("the outer search must escape the saddle and certify a minimum");
    assert!(
        result.converged(),
        "must converge at a minimum, not refuse at the saddle"
    );
    assert!(
        (result.rho[1].abs() - 1.0).abs() < 1e-4,
        "must land on a PSD minimum ρ₁=±1, got ρ={:?}",
        result.rho,
    );
    assert!(
        result.rho[0].abs() < 1e-4,
        "ρ₀ must be 0 at the minimum: {:?}",
        result.rho,
    );
    assert!(
        result
            .criterion_certificate
            .as_ref()
            .is_some_and(|c| c.certifies() && c.hessian_psd() == Some(true)),
        "the escaped minimum must carry a PSD certificate",
    );
}
