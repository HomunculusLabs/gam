//! #1575 measurement + regression: binomial/logit REML outer-loop cost.
//!
//! The issue reports a plain 3-smooth logistic GAM taking ~150 outer REML
//! cost/gradient/Hessian evaluations — an order of magnitude more outer work
//! than mgcv's REML Newton (~15) — with each outer eval paying a full n-sized
//! P-IRLS solve. The outer-eval count is n-independent, so this fixture uses a
//! deliberately small `n` (the outer overhead is fully visible there) to keep
//! the test cheap while still exercising the outer optimizer's convergence.

use super::*;
use gam_problem::{InverseLink, LikelihoodSpec, ResponseFamily, StandardLink};
use gam_terms::smooth::BlockwisePenalty;
use ndarray::{Array1, Array2};

const N: usize = 1000;
const K: usize = 10;
const N_SMOOTH: usize = 3;

struct Lcg {
    state: u64,
}
impl Lcg {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(0x9E37_79B9_7F4A_7C15),
        }
    }
    fn next_u64(&mut self) -> u64 {
        let mut z = self.state;
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Cox–de Boor cubic B-spline design on `[0,1]` with `n_basis` uniform-knot
/// bases (mgcv `bs="ps"` analogue).
fn cubic_bspline_design(xs: &[f64], n_basis: usize) -> Array2<f64> {
    let degree = 3usize;
    let n_internal = n_basis - (degree + 1);
    let n_knots = n_basis + degree + 1;
    let mut knots = vec![0.0f64; n_knots];
    for (k, slot) in knots.iter_mut().enumerate() {
        *slot = if k <= degree {
            0.0
        } else if k >= n_knots - degree - 1 {
            1.0
        } else {
            (k - degree) as f64 / (n_internal as f64 + 1.0)
        };
    }
    fn bspline(i: usize, p: usize, x: f64, knots: &[f64]) -> f64 {
        if p == 0 {
            return if (knots[i] <= x && x < knots[i + 1])
                || (x == 1.0 && knots[i + 1] == 1.0 && knots[i] < knots[i + 1])
            {
                1.0
            } else {
                0.0
            };
        }
        let mut left = 0.0;
        let d1 = knots[i + p] - knots[i];
        if d1 > 0.0 {
            left = (x - knots[i]) / d1 * bspline(i, p - 1, x, knots);
        }
        let mut right = 0.0;
        let d2 = knots[i + p + 1] - knots[i + 1];
        if d2 > 0.0 {
            right = (knots[i + p + 1] - x) / d2 * bspline(i + 1, p - 1, x, knots);
        }
        left + right
    }
    let mut b = Array2::<f64>::zeros((xs.len(), n_basis));
    for (r, &x) in xs.iter().enumerate() {
        for i in 0..n_basis {
            b[[r, i]] = bspline(i, degree, x, &knots);
        }
    }
    b
}

/// 2nd-order difference penalty `S = DᵀD` (nullspace dim 2).
fn second_difference_penalty(k: usize) -> Array2<f64> {
    let m = k - 2;
    let mut d = Array2::<f64>::zeros((m, k));
    for r in 0..m {
        d[[r, r]] = 1.0;
        d[[r, r + 1]] = -2.0;
        d[[r, r + 2]] = 1.0;
    }
    d.t().dot(&d)
}

fn build_fixture() -> (Array2<f64>, Array1<f64>, Vec<BlockwisePenalty>) {
    build_fixture_n(N, K)
}

/// Parameterised version of [`build_fixture`] so the outer-cost scaling harness
/// can measure the SAME 3-smooth logistic problem at several `n` in one process
/// (the outer-eval count is claimed n-independent in #1575; the harness checks
/// it empirically). `build_fixture()` is exactly `build_fixture_n(N, K)`.
fn build_fixture_n(n: usize, k: usize) -> (Array2<f64>, Array1<f64>, Vec<BlockwisePenalty>) {
    let p = 1 + N_SMOOTH * k;
    let mut rng = Lcg::new(100 + n as u64);

    let mut cov = vec![vec![0.0f64; n]; N_SMOOTH];
    for row in cov.iter_mut() {
        for v in row.iter_mut() {
            *v = rng.unit();
        }
    }

    let mut x = Array2::<f64>::zeros((n, p));
    for i in 0..n {
        x[[i, 0]] = 1.0;
    }
    let mut s_list = Vec::with_capacity(N_SMOOTH);
    for j in 0..N_SMOOTH {
        let block = cubic_bspline_design(&cov[j], k);
        for i in 0..n {
            for c in 0..k {
                x[[i, 1 + j * k + c]] = block[[i, c]];
            }
        }
        let start = 1 + j * k;
        s_list.push(BlockwisePenalty::new(
            start..(start + k),
            second_difference_penalty(k),
        ));
    }

    let mut y = Array1::<f64>::zeros(n);
    for i in 0..n {
        let (x1, x2, x3) = (cov[0][i], cov[1][i], cov[2][i]);
        let f = (2.0 * std::f64::consts::PI * x1).sin() * 1.5 + (x2 - 0.5).powi(2) * 6.0 - 1.0
            + (3.0 * std::f64::consts::PI * x3).cos();
        let prob = 1.0 / (1.0 + (-f).exp());
        y[i] = if rng.unit() < prob { 1.0 } else { 0.0 };
    }

    (x, y, s_list)
}

fn logit_options() -> FitOptions {
    FitOptions {
        compute_inference: true,
        max_iter: 300,
        tol: 1e-7,
        nullspace_dims: vec![2; N_SMOOTH],
        ..FitOptions::default()
    }
}

#[test]
fn binomial_logit_reml_outer_cost_is_bounded_1575() {
    let (x, y, s_list) = build_fixture();
    let weights = Array1::<f64>::ones(N);
    let offset = Array1::<f64>::zeros(N);

    let t0 = std::time::Instant::now();
    let fit = fit_gam(
        x.clone(),
        y.view(),
        weights.view(),
        offset.view(),
        &s_list,
        LikelihoodSpec::new(
            ResponseFamily::Binomial,
            InverseLink::Standard(StandardLink::Logit),
        ),
        &logit_options(),
    )
    .expect("binomial/logit P-spline REML fit should succeed");
    let dt = t0.elapsed();

    eprintln!(
        "#1575 binomial/logit REML: n={N} k={K} p={}  time={:.3}s  \
         outer_cost_evals={}  inner_pirls_solves={}  \
         grad_norm={:?}  reml={:?}  edf={:.4}  lambdas={:?}",
        1 + N_SMOOTH * K,
        dt.as_secs_f64(),
        fit.outer_cost_evals,
        fit.inner_pirls_solves,
        fit.outer_gradient_norm,
        fit.reml_score(),
        fit.edf_total().unwrap_or(f64::NAN),
        fit.lambdas
            .iter()
            .map(|l| format!("{l:.3e}"))
            .collect::<Vec<_>>(),
    );

    assert!(
        fit.convergence_evidence().outer_certificate().is_some(),
        "outer REML fit must carry its analytic convergence certificate"
    );
    let edf = fit
        .edf_total()
        .expect("the fitted three-smooth model must report total EDF");
    assert!(
        edf > 15.0,
        "the three smooths collapsed toward their affine penalty null spaces: \
         edf={edf:.4}, expected the nonlinear signal-recovery basin near 18.5 (#2519)",
    );
    let reml = fit
        .reml_score()
        .expect("a penalized binomial fit has a REML criterion");
    assert!(
        reml < 550.0,
        "the optimizer certified the wrong high-penalty basin: REML={reml:.6}, \
         expected the measured signal-recovery basin near 503.7 (#2519)",
    );
    // mgcv's REML Newton converges in well under ~15 outer iterations on this
    // family of problem. Pin a generous multiple so the test fails loudly if
    // the outer loop regresses back to the ~150-eval grind reported in #1575.
    assert!(
        fit.outer_cost_evals <= 60,
        "outer REML cost evals = {} — expected the bounded outer loop, not the \
         ~150-eval grind (#1575)",
        fit.outer_cost_evals
    );
    // The genuinely expensive work is the count of cache-missing full-n inner
    // P-IRLS solves (`outer_cost_evals` counts outer *requests*, cache hits
    // included, and under-counts the real solve budget). History: the
    // coordinate-descent seed-grid prepass once dominated this count (~74→~65
    // after memoizing its probes); that prepass was replaced by the analytic
    // seed in c9e1fba5e and deleted in ddd8af9cd, so it no longer exists to
    // dominate anything. Measured 2026-07-26 at 90ed57c6f the count is 119,
    // 86% of it BFGS/Wolfe line-search solves on a trajectory that walks to
    // the ρ ceiling and collapses every smooth onto its penalty null space
    // (edf 18.5 → 7.00, REML 503.7 → 605.0) — a fit regression, not a
    // counting artifact. This bound is deliberately NOT recalibrated to the
    // regressed trajectory: the red points at #2519 (the null-space collapse),
    // and relaxing it would launder that regression into a green.
    assert!(
        fit.inner_pirls_solves <= 90,
        "cache-missing inner P-IRLS solves = {} — expected the deduplicated \
         seed-grid + outer loop, not redundant re-solves of identical ρ (#1575)",
        fit.inner_pirls_solves
    );
}

/// n-independence gate for the binomial/logit REML outer loop (#1575's central
/// claim). Fits the SAME 3-smooth logistic problem at two sizes and reports the
/// outer-eval / inner-solve / wall-clock table, then asserts the outer cost-eval
/// count does NOT scale with `n` — each fit stays within the same bounded outer
/// budget the single-`n` sibling pins. The outer overhead is fully visible at
/// small `n`, so the sweep is capped there to keep the gate cheap; a data-scaling
/// regression (the ~150-eval grind #1575 reported) would blow the bound loudly.
#[test]
fn binomial_logit_reml_outer_cost_is_n_independent_1575() {
    eprintln!(
        "{:>7} {:>4} {:>4} {:>8} {:>8} {:>10} {:>10}",
        "n", "k", "p", "outer", "inner", "time_s", "time/inner"
    );
    let mut outer_by_n: Vec<(usize, usize)> = Vec::new();
    for &n in &[1000usize, 2000] {
        let k = K;
        let (x, y, s_list) = build_fixture_n(n, k);
        let weights = Array1::<f64>::ones(n);
        let offset = Array1::<f64>::zeros(n);
        let mut opts = logit_options();
        opts.nullspace_dims = vec![2; N_SMOOTH];
        let t0 = std::time::Instant::now();
        let fit = fit_gam(
            x,
            y.view(),
            weights.view(),
            offset.view(),
            &s_list,
            LikelihoodSpec::new(
                ResponseFamily::Binomial,
                InverseLink::Standard(StandardLink::Logit),
            ),
            &opts,
        )
        .expect("binomial/logit REML fit should succeed");
        let dt = t0.elapsed().as_secs_f64();
        eprintln!(
            "{:>7} {:>4} {:>4} {:>8} {:>8} {:>10.3} {:>10.4}",
            n,
            k,
            1 + N_SMOOTH * k,
            fit.outer_cost_evals,
            fit.inner_pirls_solves,
            dt,
            dt / (fit.inner_pirls_solves.max(1) as f64),
        );
        assert!(
            fit.convergence_evidence().outer_certificate().is_some(),
            "n={n}: outer REML fit must carry its analytic convergence certificate"
        );
        // Same bounded outer budget the single-`n` sibling gate pins: the count
        // must stay bounded as `n` grows (that IS n-independence). mgcv's Newton
        // does this in ~15; the #1575 grind was ~150.
        assert!(
            fit.outer_cost_evals <= 60,
            "n={n}: outer REML cost evals = {} — expected the bounded, \
             n-independent outer loop, not the ~150-eval grind (#1575)",
            fit.outer_cost_evals
        );
        outer_by_n.push((n, fit.outer_cost_evals));
    }
    // Direct n-independence check: doubling `n` must not materially grow the
    // outer cost-eval count (a data-scaling regression would roughly track `n`).
    let (n_small, outer_small) = outer_by_n[0];
    let (n_large, outer_large) = outer_by_n[1];
    assert!(
        outer_large <= outer_small + 15,
        "outer cost-eval count scaled with n (#1575): n={n_small}→{outer_small}, \
         n={n_large}→{outer_large}; the outer loop must be n-independent"
    );
}

/// #2519/#2614: WHY the first outer line search on this fixture fails.
///
/// The two gates above no longer reach their edf/REML assertions — the fit
/// itself errors, with `termination=line_search_failed(|g|=1.484825e0)` and
/// `line_search=MaxAttempts after 50 attempt(s)` after ONE outer iteration.
/// No step was ever accepted, so the reported objective and `|g|` are the
/// seed's.
///
/// If `g` really is the gradient of the function the line search evaluates,
/// Armijo backtracking cannot fail: for small enough α the decrease along
/// `d = −g/‖g‖` is `α‖g‖ + O(α²)`, which beats the Armijo target `c₁α‖g‖`
/// with `c₁ = 1e-4`. Something in that sentence is false. This probe walks a
/// decade α ladder along the SAME direction the optimizer takes at the SAME
/// seed and prints, per α, the outcome of the value-only outer evaluation —
/// including whether it is an error, an `OuterEval::infeasible` (+∞), or a
/// finite cost — next to the Armijo target. It also compares a central
/// finite difference of the directional derivative against the analytic
/// `gᵀd = −‖g‖`.
///
/// Direction and metric are the optimizer's, not invented here: on the
/// gradient-only BFGS path the initial metric is `InitialMetric::Scalar(1/‖g₀‖)`
/// (`rho_optimizer/run_plan.rs`), so iterate 0's direction is exactly
/// `−g/‖g‖`, unit norm. `c₁ = 1e-4` is `opt`'s `BfgsCore::c1` default; the
/// real acceptance test adds an `eps_f(f_k)` slack on top, so a step that
/// clears the target here would also clear it there.
#[test]
fn binomial_logit_first_outer_line_search_ladder_1575() {
    let (x, y, s_list) = build_fixture();
    let weights = Array1::<f64>::ones(N);
    let offset = Array1::<f64>::zeros(N);
    let p = x.ncols();

    let specs: Vec<PenaltySpec> = s_list.iter().map(PenaltySpec::from_blockwise_ref).collect();
    let ext = ExternalOptimOptions {
        family: LikelihoodSpec::new(
            ResponseFamily::Binomial,
            InverseLink::Standard(StandardLink::Logit),
        ),
        latent_cloglog: None,
        mixture_link: None,
        optimize_mixture: false,
        sas_link: None,
        optimize_sas: false,
        compute_inference: true,
        skip_rho_posterior_inference: false,
        max_iter: 300,
        tol: 1e-7,
        nullspace_dims: vec![2; N_SMOOTH],
        linear_constraints: None,
        firth_bias_reduction: Some(false),
        penalty_shrinkage_floor: None,
        rho_prior: Default::default(),
        kronecker_penalty_system: None,
        kronecker_factored: None,
        persist_warm_start_disk: false,
    };
    let cfg = super::external_options::resolved_external_config(&ext)
        .expect("the binomial/logit external config resolves")
        .0;
    let x_dm: DesignMatrix = x.clone().into();
    let (canonical, active_nullspace_dims) = gam_terms::construction::canonicalize_penalty_specs(
        &specs,
        &ext.nullspace_dims,
        p,
        "binomial_logit_first_outer_line_search_ladder_1575",
    )
    .expect("the three P-spline penalties canonicalize");
    let conditioning = ParametricColumnConditioning::infer_from_penalty_specs(&x_dm, &specs);
    let x_fit = conditioning.apply_to_design(&x_dm);
    let mut state = RemlState::newwith_offset(
        y.view(),
        x_fit,
        weights.view(),
        offset.view(),
        canonical,
        p,
        &cfg,
        Some(active_nullspace_dims),
        None,
        None,
    )
    .expect("the outer REML state builds on the #1575 fixture");
    state.set_penalty_shrinkage_floor(ext.penalty_shrinkage_floor);
    state.set_rho_prior(ext.rho_prior.clone());
    state.set_link_states(
        cfg.link_kind.mixture_state().cloned(),
        cfg.link_kind.sas_state().copied(),
    );

    // The ρ the failing fit reported as its checkpoint (run 30452220660). The
    // line search failed on outer iteration 1, so this IS the seed it started
    // from, not a point it walked to.
    let rho0 = Array1::from(vec![
        -0.7060116450326054,
        -0.11090597501554852,
        0.6994622375684085,
    ]);
    let seed = state
        .compute_outer_eval_with_order(&rho0, crate::rho_optimizer::OuterEvalOrder::ValueAndGradient)
        .expect("the seed evaluates: the failing fit reported a finite objective and |g| there");
    let f0 = seed.cost;
    let g = seed.gradient.clone();
    let gnorm = g.iter().map(|v| v * v).sum::<f64>().sqrt();
    assert!(
        f0.is_finite() && gnorm.is_finite() && gnorm > 0.0,
        "seed evaluation is degenerate: f0={f0}, |g|={gnorm}, g={g:.6e}",
    );
    // Unit-norm steepest descent in the optimizer's iterate-0 metric, so
    // gᵀd = −‖g‖ exactly.
    let d = g.mapv(|v| -v / gnorm);
    let c1 = 1.0e-4_f64;

    let value_at = |state: &RemlState<'_>, rho: &Array1<f64>| -> (Option<f64>, String) {
        match state.compute_outer_eval_with_order(rho, crate::rho_optimizer::OuterEvalOrder::Value) {
            Ok(eval) if eval.cost.is_finite() => (Some(eval.cost), format!("{:.12e}", eval.cost)),
            Ok(eval) if eval.cost == f64::INFINITY => {
                (None, "INFEASIBLE (+inf cost)".to_string())
            }
            Ok(eval) => (None, format!("NON-FINITE cost {}", eval.cost)),
            Err(err) => (None, format!("ERR {err}")),
        }
    };

    let mut ladder = String::new();
    let mut accepting_alpha: Option<f64> = None;
    let mut finite_probes = 0usize;
    for alpha in [
        1.0e0, 1.0e-1, 1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6, 1.0e-7, 1.0e-8, 1.0e-9, 1.0e-10,
        1.0e-11, 1.0e-12,
    ] {
        let trial = &rho0 + &d.mapv(|v| v * alpha);
        let (value, label) = value_at(&state, &trial);
        let target = -c1 * alpha * gnorm;
        let (delta_text, accepted) = match value {
            Some(f) => {
                finite_probes += 1;
                let delta = f - f0;
                (format!("{delta:+.6e}"), delta <= target)
            }
            None => ("n/a".to_string(), false),
        };
        if accepted && accepting_alpha.is_none() {
            accepting_alpha = Some(alpha);
        }
        ladder.push_str(&format!(
            "\n  alpha={alpha:.0e}  f={label}  f-f0={delta_text}  armijo_target={target:+.6e}  accept={accepted}"
        ));
    }

    // Central finite difference of the directional derivative. The analytic
    // value is gᵀd = −‖g‖ by construction of `d`. Step sizes span three
    // decades so a disagreement that is a truncation artefact (shrinks with h)
    // is distinguishable from a genuine gradient defect (does not).
    let mut fd_report = String::new();
    for h in [1.0e-3_f64, 1.0e-4, 1.0e-5] {
        let (plus, plus_label) = value_at(&state, &(&rho0 + &d.mapv(|v| v * h)));
        let (minus, minus_label) = value_at(&state, &(&rho0 - &d.mapv(|v| v * h)));
        let fd_text = match (plus, minus) {
            (Some(fp), Some(fm)) => {
                let fd = (fp - fm) / (2.0 * h);
                format!("fd={fd:+.9e}  fd/analytic={:+.6}", fd / (-gnorm))
            }
            _ => format!("fd=n/a (+: {plus_label}; -: {minus_label})"),
        };
        fd_report.push_str(&format!("\n  h={h:.0e}  {fd_text}"));
    }

    let summary = format!(
        "#1575/#2519 first-outer-line-search ladder at the reported checkpoint\n\
         rho0 = {rho0:.16e}\n\
         f0 = {f0:.12e}   |g| = {gnorm:.9e}   g = {g:.9e}\n\
         d = -g/|g| (unit norm; g'd = -|g| = {:.9e}), armijo c1 = {c1:.0e}\n\
         finite probes on the ladder: {finite_probes}/13\n\
         analytic directional derivative g'd = {:.9e}; central FD:{fd_report}\n\
         alpha ladder:{ladder}",
        -gnorm, -gnorm,
    );
    eprintln!("{summary}");

    assert!(
        accepting_alpha.is_some(),
        "no alpha in [1e0, 1e-12] along the optimizer's own first search direction \
         achieves the Armijo decrease the backtracking search demands, so the first \
         outer line search cannot succeed and the fit cannot leave its seed.\n{summary}"
    );
}

/// #2519/#2614: the outer objective must be a FUNCTION of rho.
///
/// The alpha ladder above (run 30453519953) measured the outer cost at rho
/// displacements from 1e0 down to 1e-12 along one direction and found it takes
/// three discrete values, not one continuous curve:
///
/// ```text
///   alpha=1e-8   f = 5.091491164067e2   (f0 - 2.4e-6)
///   alpha=1e-9   f = 4.999387787758e2   (f0 - 9.2103400803  = f0 - 0.5*ln(1e8))
///   alpha=1e-12  f = 5.275697976823e2   (f0 + 18.4206788262 = f0 + 0.5*ln(1e16))
/// ```
///
/// `exp(2*delta)` for those two offsets is `1.000001e-8` and `9.999962e+15` —
/// a determinant inside a `0.5*log|.|` term differing by exactly `1e8` on one
/// mode and by `1e8` on each of two modes. A 1e-12 change in rho cannot move a
/// spectrum by 1e8, and the analytic gradient is fine (the h=1e-4 central FD
/// gave -4.1497 against the analytic -4.1558, agreeing to 0.15%), so the
/// gradient is not the defect: the same rho is being evaluated against
/// different stabilizations. That is why no alpha satisfies Armijo — the
/// decrease being asked for is O(1e-4) and the jump between branches is 9.21.
///
/// This probe names the term. It evaluates the same rho at the start and again
/// at the end of a short ladder and reports, at every point, the cost beside
/// the stabilization ridge actually used and the inner P-IRLS state that
/// produced it (deviance, edf, penalty, iterations, inner gradient), so the
/// 9.21 shows up in a named column rather than only in the total.
#[test]
fn binomial_logit_outer_objective_is_a_function_of_rho_1575() {
    let (x, y, s_list) = build_fixture();
    let weights = Array1::<f64>::ones(N);
    let offset = Array1::<f64>::zeros(N);
    let p = x.ncols();

    let specs: Vec<PenaltySpec> = s_list.iter().map(PenaltySpec::from_blockwise_ref).collect();
    let ext = ExternalOptimOptions {
        family: LikelihoodSpec::new(
            ResponseFamily::Binomial,
            InverseLink::Standard(StandardLink::Logit),
        ),
        latent_cloglog: None,
        mixture_link: None,
        optimize_mixture: false,
        sas_link: None,
        optimize_sas: false,
        compute_inference: true,
        skip_rho_posterior_inference: false,
        max_iter: 300,
        tol: 1e-7,
        nullspace_dims: vec![2; N_SMOOTH],
        linear_constraints: None,
        firth_bias_reduction: Some(false),
        penalty_shrinkage_floor: None,
        rho_prior: Default::default(),
        kronecker_penalty_system: None,
        kronecker_factored: None,
        persist_warm_start_disk: false,
    };
    let cfg = super::external_options::resolved_external_config(&ext)
        .expect("the binomial/logit external config resolves")
        .0;
    let x_dm: DesignMatrix = x.clone().into();
    let (canonical, active_nullspace_dims) = gam_terms::construction::canonicalize_penalty_specs(
        &specs,
        &ext.nullspace_dims,
        p,
        "binomial_logit_outer_objective_is_a_function_of_rho_1575",
    )
    .expect("the three P-spline penalties canonicalize");
    let conditioning = ParametricColumnConditioning::infer_from_penalty_specs(&x_dm, &specs);
    let x_fit = conditioning.apply_to_design(&x_dm);
    let mut state = RemlState::newwith_offset(
        y.view(),
        x_fit,
        weights.view(),
        offset.view(),
        canonical,
        p,
        &cfg,
        Some(active_nullspace_dims),
        None,
        None,
    )
    .expect("the outer REML state builds on the #1575 fixture");
    state.set_penalty_shrinkage_floor(ext.penalty_shrinkage_floor);
    state.set_rho_prior(ext.rho_prior.clone());
    state.set_link_states(
        cfg.link_kind.mixture_state().cloned(),
        cfg.link_kind.sas_state().copied(),
    );

    let rho0 = Array1::from(vec![
        -0.7060116450326054,
        -0.11090597501554852,
        0.6994622375684085,
    ]);
    let seed = state
        .compute_outer_eval_with_order(&rho0, crate::rho_optimizer::OuterEvalOrder::ValueAndGradient)
        .expect("the seed evaluates");
    let f0 = seed.cost;
    let g = seed.gradient.clone();
    let gnorm = g.iter().map(|v| v * v).sum::<f64>().sqrt();
    let d = g.mapv(|v| -v / gnorm);

    let mut table = String::new();
    let mut record = |state: &RemlState<'_>, label: &str, rho: &Array1<f64>, cost_text: String| {
        let ridge = state.last_ridge_used();
        let inner = state.obtain_eval_bundle(rho).map(|bundle| {
            let pr = &bundle.pirls_result;
            format!(
                "dev={:.9e} edf={:.6} pen={:.9e} iters={} |g_inner|={:.3e} ridge_pirls={:.6e} ridge_bundle={:.6e} status={:?}",
                pr.deviance,
                pr.edf,
                pr.stable_penalty_term,
                pr.iteration,
                pr.lastgradient_norm,
                pr.ridge_passport.delta(),
                bundle.ridge_passport.delta(),
                pr.status,
            )
        });
        let inner_text = match inner {
            Ok(text) => text,
            Err(err) => format!("bundle unavailable: {err}"),
        };
        let ridge_text = match ridge {
            Some(value) => format!("{value:.9e}"),
            None => "none".to_string(),
        };
        table.push_str(&format!(
            "\n  {label:<12} cost={cost_text}  ridge={ridge_text}  {inner_text}"
        ));
    };

    record(&state, "rho0 (first)", &rho0, format!("{f0:.12e}"));
    for alpha in [1.0e-1_f64, 1.0e-4, 1.0e-7, 1.0e-9, 1.0e-12] {
        let trial = &rho0 + &d.mapv(|v| v * alpha);
        let cost_text = match state
            .compute_outer_eval_with_order(&trial, crate::rho_optimizer::OuterEvalOrder::Value)
        {
            Ok(eval) => format!("{:.12e}", eval.cost),
            Err(err) => format!("ERR {err}"),
        };
        record(&state, &format!("alpha={alpha:.0e}"), &trial, cost_text);
    }
    let repeat = state
        .compute_outer_eval_with_order(&rho0, crate::rho_optimizer::OuterEvalOrder::Value)
        .expect("re-evaluating the seed must still be feasible");
    record(&state, "rho0 (again)", &rho0, format!("{:.12e}", repeat.cost));

    let drift = repeat.cost - f0;
    let summary = format!(
        "#1575/#2519 outer objective at a FIXED rho, evaluated twice\n\
         f(rho0) first = {f0:.12e}\n\
         f(rho0) again = {:.12e}\n\
         drift         = {drift:+.12e}   (0.5*ln(1e8) = 9.210340372)\n\
         |g| at rho0   = {gnorm:.9e}{table}",
        repeat.cost,
    );
    eprintln!("{summary}");

    // Two evaluations at a BIT-IDENTICAL rho can differ only through where the
    // inner P-IRLS stopped. The inner mode is certified to a relative gradient
    // of `tol = 1e-7` (`logit_options`), and the outer cost is stationary in
    // beta at the mode, so that inner slack enters the cost at SECOND order —
    // below 1e-14 relative. A bound of 1e-6*(1+|f0|) is eight orders above that
    // floor (so inner-tolerance jitter can never trip it) and four orders below
    // the 9.21 branch jump the ladder measured, which is the thing this asserts
    // is absent.
    let bound = 1.0e-6 * (1.0 + f0.abs());
    assert!(
        drift.abs() <= bound,
        "the outer REML objective is not a function of rho: the SAME rho gives \
         two costs differing by {drift:+.6e} (bound {bound:.6e}). A line search \
         cannot descend on a criterion that moves under it.\n{summary}"
    );
}

/// #2519/#2614: is the trial point hard, or is the carried inner state wrong?
///
/// Run 30454491190 measured the branch jumps of the sibling probe down to their
/// source: at rho0 the inner P-IRLS converges in 5 iterations to an inner
/// gradient of 3.3e-14, and one displacement of 1e-7 away — warm-started from
/// that very solution — it stops after ONE iteration at 6.3e-4 and the outer
/// eval returns the infeasible `+inf`:
///
/// ```text
///   rho0        cost=5.091491188561e2  dev=9.912547084e2 edf=19.672597 iters=5 |g_inner|=3.344e-14
///   alpha=1e-7  cost=inf   "did not converge within 1 iterations. Last gradient norm was 6.314893e-4."
///   alpha=1e-9  cost=inf   "did not converge within 300 iterations. Last gradient norm was 4.117748e-4."
///   alpha=1e-12 cost=inf   "did not converge within 1 iterations. Last gradient norm was 6.321214e-4."
/// ```
///
/// (The count in that message is `pirls_result.iteration`, not the budget —
/// `PirlsDidNotConverge` is constructed with `max_iterations: iteration`. So
/// "within 1 iterations" is a solve that quit after one step, not a solve given
/// one step.)
///
/// A rho displacement of 1e-7 from a converged mode is not a hard inner
/// problem. What differs between the two calls is everything the inner solve
/// carries across calls: the warm-start beta, the recorded warm-start rho and
/// its IFT extrapolation, and the Levenberg-Marquardt damping hint seeded from
/// the previous solve's `final_lm_lambda`. This probe runs the SAME trial rho
/// twice — once through the ordinary stateful path, once through
/// `execute_pirls_stateless_for_cubature`, which by construction threads none
/// of that state — and prints both. If the stateless solve converges where the
/// stateful one refuses, the trial point is fine and the carried state is the
/// defect.
#[test]
fn binomial_logit_inner_solve_refusal_is_carried_state_1575() {
    let (x, y, s_list) = build_fixture();
    let weights = Array1::<f64>::ones(N);
    let offset = Array1::<f64>::zeros(N);
    let p = x.ncols();

    let specs: Vec<PenaltySpec> = s_list.iter().map(PenaltySpec::from_blockwise_ref).collect();
    let ext = ExternalOptimOptions {
        family: LikelihoodSpec::new(
            ResponseFamily::Binomial,
            InverseLink::Standard(StandardLink::Logit),
        ),
        latent_cloglog: None,
        mixture_link: None,
        optimize_mixture: false,
        sas_link: None,
        optimize_sas: false,
        compute_inference: true,
        skip_rho_posterior_inference: false,
        max_iter: 300,
        tol: 1e-7,
        nullspace_dims: vec![2; N_SMOOTH],
        linear_constraints: None,
        firth_bias_reduction: Some(false),
        penalty_shrinkage_floor: None,
        rho_prior: Default::default(),
        kronecker_penalty_system: None,
        kronecker_factored: None,
        persist_warm_start_disk: false,
    };
    let cfg = super::external_options::resolved_external_config(&ext)
        .expect("the binomial/logit external config resolves")
        .0;
    let x_dm: DesignMatrix = x.clone().into();
    let (canonical, active_nullspace_dims) = gam_terms::construction::canonicalize_penalty_specs(
        &specs,
        &ext.nullspace_dims,
        p,
        "binomial_logit_inner_solve_refusal_is_carried_state_1575",
    )
    .expect("the three P-spline penalties canonicalize");
    let conditioning = ParametricColumnConditioning::infer_from_penalty_specs(&x_dm, &specs);
    let x_fit = conditioning.apply_to_design(&x_dm);
    let mut state = RemlState::newwith_offset(
        y.view(),
        x_fit,
        weights.view(),
        offset.view(),
        canonical,
        p,
        &cfg,
        Some(active_nullspace_dims),
        None,
        None,
    )
    .expect("the outer REML state builds on the #1575 fixture");
    state.set_penalty_shrinkage_floor(ext.penalty_shrinkage_floor);
    state.set_rho_prior(ext.rho_prior.clone());
    state.set_link_states(
        cfg.link_kind.mixture_state().cloned(),
        cfg.link_kind.sas_state().copied(),
    );

    let rho0 = Array1::from(vec![
        -0.7060116450326054,
        -0.11090597501554852,
        0.6994622375684085,
    ]);
    let seed = state
        .compute_outer_eval_with_order(&rho0, crate::rho_optimizer::OuterEvalOrder::ValueAndGradient)
        .expect("the seed evaluates");
    let f0 = seed.cost;
    let g = seed.gradient.clone();
    let gnorm = g.iter().map(|v| v * v).sum::<f64>().sqrt();
    let d = g.mapv(|v| -v / gnorm);

    let describe = |result: Result<std::sync::Arc<crate::pirls::PirlsResult>, EstimationError>| {
        match result {
            Ok(pr) => format!(
                "OK iters={} |g_inner|={:.3e} lm_lambda={:.3e} dev={:.9e} edf={:.6} status={:?}",
                pr.iteration,
                pr.lastgradient_norm,
                pr.final_lm_lambda,
                pr.deviance,
                pr.edf,
                pr.status,
            ),
            Err(err) => format!("REFUSED {err}"),
        }
    };

    let mut table = String::new();
    let mut stateful_refusals = 0usize;
    let mut stateless_refusals = 0usize;
    for alpha in [1.0e-1_f64, 1.0e-4, 1.0e-7, 1.0e-9, 1.0e-12] {
        let trial = &rho0 + &d.mapv(|v| v * alpha);
        // Stateless FIRST, so the stateful arm sees exactly the carried state
        // the previous stateful call left behind (the stateless call writes
        // none of it) rather than a state this probe perturbed.
        let stateless = describe(state.execute_pirls_stateless_for_cubature(&trial, None));
        if stateless.starts_with("REFUSED") {
            stateless_refusals += 1;
        }
        let stateful_cost = match state
            .compute_outer_eval_with_order(&trial, crate::rho_optimizer::OuterEvalOrder::Value)
        {
            Ok(eval) => format!("{:.12e}", eval.cost),
            Err(err) => format!("ERR {err}"),
        };
        let stateful_inner = describe(state.obtain_eval_bundle(&trial).map(|b| b.pirls_result));
        if stateful_inner.starts_with("REFUSED") {
            stateful_refusals += 1;
        }
        table.push_str(&format!(
            "\n  alpha={alpha:.0e}\n    stateful  cost={stateful_cost}  {stateful_inner}\n    stateless {stateless}"
        ));
    }

    let summary = format!(
        "#1575/#2519 same trial rho, stateful vs stateless inner solve\n\
         f(rho0) = {f0:.12e}   |g| = {gnorm:.9e}\n\
         inner refusals: stateful {stateful_refusals}/5, stateless {stateless_refusals}/5{table}"
    );
    eprintln!("{summary}");

    // The claim under test: a trial rho within 1e-1 of a point whose inner mode
    // is certified to 3.3e-14 is a solvable inner problem. `execute_pirls_stateless_for_cubature`
    // is documented as bit-identical math to the ordinary non-screening branch
    // with every cross-call carry removed, so a refusal there WOULD mean the
    // point is genuinely hard and this probe would be the wrong lead.
    assert_eq!(
        stateless_refusals, 0,
        "the stateless inner solve also refuses, so these trial points are \
         genuinely hard and the carried warm-start/LM state is not the lead.\n{summary}"
    );
    assert_eq!(
        stateful_refusals, 0,
        "the same trial rho that the stateless inner solve fits is refused by \
         the stateful path, so the outer objective's +inf is produced by the \
         state the inner solve carries between calls, not by the point.\n{summary}"
    );
}
