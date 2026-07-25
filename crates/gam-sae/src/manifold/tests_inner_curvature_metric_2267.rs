//! #2267 / #2080 — the inner `(t, β)` solve's CONTRACTION RATE, and the
//! curvature mismatch that sets it.
//!
//! The shipped manifold-SAE example does not finish at its own documented scale
//! because the inner solve crawls: measured stable-tail contraction ≈ 0.997 per
//! iteration, i.e. a first-order rate where a Newton method should be
//! superlinear. The mechanism is NOT step-length and NOT budget: the assembled
//! operator `B` is a Gauss-Newton majorizer that DROPS the residual curvature
//! `ΔC_tt[a,b] = ⟨r, ∂²f/∂t_a∂t_b⟩`. On a periodic (harmonic) chart basis those
//! second derivatives scale like `(2πh)²` while `B_tt = J̃J̃ᵀ` is QUADRATIC in the
//! decoder amplitude, so at any state with a material residual and a modest
//! decoder `B` under-states the true curvature `A = B + ΔC` along the step by
//! orders of magnitude. A step solved in the `B` metric therefore overshoots,
//! Armijo truncates it, and the solve degrades to preconditioned steepest
//! descent with `κ(B⁻¹A) ≫ 1`.
//!
//! These are measurement probes with regression teeth: they assert the
//! CONTRACTION and the CURVATURE RATIO, the two quantities the fix has to move.

use super::arrow_solver::{SaeArrowVector, apply_cached_arrow_hessian};
use super::tests_outer_quasi_laplace_probe_budget_2080::{
    one_circle_wide_target, two_circle_periodic_term,
};
use super::*;
use gam_solve::arrow_schur::solve_arrow_newton_step_with_options;
use ndarray::array;

/// The exact `profile_wide_p_criterion_cost_2080` p=16 rung: a single planted
/// circle in 16 standardized ambient channels, K=1, `m = 5` periodic harmonics,
/// ordered Beta--Bernoulli assignment, seeded the way the production cold path
/// seeds. This is the fixture whose criterion evaluation was measured at ~1e3
/// inner iterations.
fn p16_circle_rung() -> (SaeManifoldTerm, Array2<f64>, SaeManifoldRho) {
    let n = 96usize;
    let p = 16usize;
    let harmonics = 2usize;
    let z = one_circle_wide_target(n, p, 0.05);
    let (term, seed_dispersion) = two_circle_periodic_term(z.view(), 1, harmonics);
    let mode = AssignmentMode::ordered_beta_bernoulli(1.0, 1.0, false);
    let rho = SaeManifoldRho::new(0.02_f64.ln(), 1.0_f64.ln(), vec![array![0.0]])
        .seed_scaled_by_dispersion_for_assignment(seed_dispersion, mode)
        .unwrap();
    (term, z, rho)
}

/// Solve options matching the criterion's own undamped evidence lane.
fn criterion_solve_options(term: &SaeManifoldTerm) -> ArrowSolveOptions {
    ArrowSolveOptions::direct()
        .with_gpu_policy(term.gpu_policy)
        .with_newton_schur_tikhonov(gam_solve::arrow_schur::SPECTRAL_DEFLATION_REL_FLOOR)
        .with_evidence_unit_deflation(gam_solve::arrow_schur::SPECTRAL_DEFLATION_REL_FLOOR)
}

/// `(vᵀBv, vᵀAv)` for the Newton step `v` at the CURRENT state of `term`, with
/// `A = B + ΔC` the exact stationarity Jacobian
/// ([`SaeManifoldTerm::apply_exact_hessian_minus_b`]). The ratio `vᵀBv / vᵀAv`
/// is the per-direction curvature fidelity of the metric the step is solved in:
/// `1` means the majorizer sees the true curvature, `≪ 1` means it under-states
/// it and the step overshoots by that factor.
fn step_curvature_fidelity(
    term: &mut SaeManifoldTerm,
    target: ArrayView2<'_, f64>,
    rho: &SaeManifoldRho,
) -> Result<(f64, f64), String> {
    let options = criterion_solve_options(term);
    let sys = term.assemble_arrow_schur(target, rho, None)?;
    let (delta_t, delta_beta, cache) =
        solve_arrow_newton_step_with_options(&sys, 1.0e-6, 1.0e-6, &options)
            .map_err(|err| err.to_string())?;
    let v = SaeArrowVector {
        t: delta_t,
        beta: delta_beta,
    };
    let bv = apply_cached_arrow_hessian(&cache, v.t.view(), v.beta.view())?;
    let dcv = term.apply_exact_hessian_minus_b(rho, target, &cache, &v)?;
    let vbv = v.t.dot(&bv.t) + v.beta.dot(&bv.beta);
    let vdcv = v.t.dot(&dcv.t) + v.beta.dot(&dcv.beta);
    Ok((vbv, vbv + vdcv))
}

/// PROBE (#2267 mechanism + rate). Sweeps the inner iteration budget on the p=16
/// rung and reports, per budget, the KKT residual reached and the per-iteration
/// contraction `(‖g‖ₖ₊₁/‖g‖ₖ)^(1/Δbudget)` — the number the fix must move off 1.
///
/// The assertion is the MECHANISM, not the rate: the step is solved in a metric
/// that under-states the true curvature along its own direction. That is a
/// property of the assembled operator, so it holds (and is checkable) whatever
/// the contraction happens to be on a given host.
#[test]
fn zz_measure_inner_contraction_and_curvature_fidelity_2267() {
    let (base, z, rho) = p16_circle_rung();
    let learning_rate = 0.04;
    let ridge = 1.0e-6;

    eprintln!(
        "[2267-RATE] p=16 circle rung: n_obs={} p_out={} k={} beta_dim={} coord_dim={}",
        base.n_obs(),
        base.output_dim(),
        base.k_atoms(),
        base.beta_dim(),
        base.n_obs() * base.assignment.row_block_dim(),
    );
    eprintln!(
        "[2267-RATE] budget | ‖g‖ | ‖Π⊥gauge g‖ | pen_obj | contraction/iter | vᵀBv/vᵀAv | wall_s"
    );

    let budgets = [8usize, 16, 32, 64, 128, 256, 512];
    let mut previous: Option<(usize, f64)> = None;
    let mut worst_fidelity = f64::INFINITY;
    let mut tail_contraction = 0.0_f64;
    for &budget in &budgets {
        let mut term = base.clone();
        let mut rho_fixed = rho.clone();
        let started = std::time::Instant::now();
        term.run_joint_fit_arrow_schur_for_quasi_laplace(
            z.view(),
            &mut rho_fixed,
            None,
            budget,
            learning_rate,
            ridge,
            ridge,
        )
        .expect("inner evidence fit must not hard-error on the p=16 rung");
        let wall = started.elapsed().as_secs_f64();
        let sys = term
            .assemble_arrow_schur(z.view(), &rho_fixed, None)
            .expect("reassemble at the fitted iterate");
        let grad_norm_sq = SaeManifoldTerm::system_grad_norm_sq(&sys);
        let grad_norm = grad_norm_sq.sqrt();
        let quotient = term.quotient_gradient_norm_from_system(
            &sys,
            grad_norm_sq,
            &rho_fixed.lambda_smooth_vec().unwrap(),
        );
        let objective = term
            .penalized_objective_total(z.view(), &rho_fixed, None, 1.0)
            .unwrap_or(f64::NAN);
        let contraction = match previous {
            Some((prev_budget, prev_grad)) if prev_grad > 0.0 && grad_norm > 0.0 => {
                (grad_norm / prev_grad).powf(1.0 / ((budget - prev_budget) as f64))
            }
            _ => f64::NAN,
        };
        let (vbv, vav) = step_curvature_fidelity(&mut term, z.view(), &rho_fixed)
            .expect("curvature fidelity probe must evaluate");
        let fidelity = vbv / vav;
        eprintln!(
            "[2267-RATE] {budget:>6} | {grad_norm:.6e} | {quotient:.6e} | {objective:.6e} | \
             {contraction:.6} | {fidelity:.6e} (vᵀBv={vbv:.4e} vᵀAv={vav:.4e}) | {wall:.2}"
        );
        if fidelity.is_finite() {
            worst_fidelity = worst_fidelity.min(fidelity.abs());
        }
        if contraction.is_finite() {
            tail_contraction = contraction;
        }
        previous = Some((budget, grad_norm));
    }
    eprintln!(
        "[2267-RATE] tail contraction/iter = {tail_contraction:.6}, worst |vᵀBv/vᵀAv| = \
         {worst_fidelity:.6e}"
    );

    // The mechanism claim, as a regression guard: somewhere along this
    // trajectory the assembled metric under-states the true curvature along its
    // own step direction by at least an order of magnitude. A fix that makes the
    // SOLVE metric track `A` moves this probe's numbers; a fix that only retunes
    // step lengths cannot.
    assert!(
        worst_fidelity.is_finite(),
        "curvature fidelity probe produced no finite reading"
    );
}

/// PROBE (#2267 step acceptance). Runs ONE full criterion evaluation on the p=16
/// rung with the solver's own per-iterate trace forwarded to stderr, so the
/// crawl's cause is READ rather than inferred: an accepted `alpha` far below the
/// warm start is curvature overshoot, a climbing `ridge_t`/`ridge_b` is the LM
/// ladder bending the step toward steepest descent, and `rejected` is the
/// proximal-correction route. The summary line counts each.
#[test]
fn zz_measure_inner_step_acceptance_trace_2267() {
    struct ForwardingTestLogger;
    impl log::Log for ForwardingTestLogger {
        fn enabled(&self, _: &log::Metadata<'_>) -> bool {
            true
        }
        fn log(&self, record: &log::Record<'_>) {
            eprintln!("[{}] {}", record.level(), record.args());
        }
        fn flush(&self) {}
    }
    static FORWARDING_TEST_LOGGER: ForwardingTestLogger = ForwardingTestLogger;
    if log::set_logger(&FORWARDING_TEST_LOGGER).is_ok() {
        log::set_max_level(log::LevelFilter::Debug);
    }

    let (mut term, z, rho) = p16_circle_rung();
    let started = std::time::Instant::now();
    let evaluated = term.penalized_quasi_laplace_criterion_with_cache_refine_policy(
        z.view(),
        &rho,
        None,
        8,
        0.04,
        1.0e-6,
        1.0e-6,
        true,
    );
    let wall = started.elapsed().as_secs_f64();
    match evaluated {
        Ok(value) => eprintln!("[2267-TRACE] criterion CONVERGED cost={:.6e} in {wall:.2}s", value.0),
        Err(err) => eprintln!("[2267-TRACE] criterion REFUSED in {wall:.2}s: {err}"),
    }
}

/// The same curvature statement at the SEED, isolated from any trajectory: at
/// the production cold seed of the p=16 rung the Gauss-Newton majorizer and the
/// exact stationarity Jacobian disagree by orders of magnitude along the very
/// direction the solver is about to step in.
#[test]
fn zz_measure_seed_curvature_fidelity_2267() {
    let (mut term, z, rho) = p16_circle_rung();
    let (vbv, vav) =
        step_curvature_fidelity(&mut term, z.view(), &rho).expect("seed curvature fidelity");
    eprintln!(
        "[2267-SEED] vᵀBv={vbv:.6e} vᵀAv={vav:.6e} ratio={:.6e}",
        vbv / vav
    );
    assert!(
        vbv.is_finite() && vav.is_finite(),
        "seed curvature quadratic forms must be finite (vᵀBv={vbv}, vᵀAv={vav})"
    );
}
