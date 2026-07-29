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
