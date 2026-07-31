//! #2519: is the stabilization ridge constant along ρ on the paths the dense
//! GLM selector fix did not touch?
//!
//! `ensure_positive_definitewithridge` used to factor the bare penalized
//! Hessian first and return `ridge = 0.0` on success, adding
//! `FIXED_STABILIZATION_RIDGE` only on failure — so δ was a function of ρ
//! through a Cholesky-success predicate, while being carried into the outer
//! criterion through `0.5·log|H|`. On the #1575 binomial fixture that moved the
//! cost by exactly `0.5·ln(1e8) = 9.2103` between neighbouring ρ. That selector
//! now applies δ unconditionally.
//!
//! Two selectors USED TO keep the bare-first shape:
//!   * `pirls::pls_solver`'s Gaussian-identity PLS branch, and
//!   * `pirls::newton_solve::ensure_sparse_positive_definite_with_fixed_ridge`.
//!
//! Both are now unconditional as well, so all four ridge selectors agree.
//!
//! MEASURED RESULT FROM BEFORE THAT REPAIR, so the next reader does not repeat
//! the scan: across Vandermonde degrees 6–16 and ρ ∈ [−12, 12] — 208
//! evaluations — the Gaussian-identity PLS branch never changed its ridge. Its
//! bare attempt went through `StableSolver::factorize`, which succeeded at
//! every point, so δ ≡ 0 and the branch never bit. That was a NEGATIVE RESULT,
//! not a clearance: it said this fixture family could not make that selector
//! flip, not that no fixture could. Its bare-first shape was repaired on the
//! argument the dense selector's own comment makes — a δ chosen by a
//! Cholesky-success predicate is a function of ρ, whatever a given fixture
//! happens to exercise — and because the #1122 objection it was built for
//! (`a value/derivative desync under a nonzero δ`) is an objection to an
//! ADAPTIVE δ, not to a constant one.
//!
//! The sparse ladder was repaired at the same time for a second reason: it
//! reported the REQUESTED ridge rather than the APPLIED one. `pls_solver`'s
//! closure rewrote a requested `0.0` into `FIXED_STABILIZATION_RIDGE`, so the
//! ladder's first rung handed back a matrix carrying δ while reporting
//! `ridge_used = 0.0` — β̂ solved the ridged system while the criterion was
//! assembled as if unridged.
//!
//! What this file gates is the contract all four selectors now share: δ is
//! applied at every ρ, not only where a factorization fails. Note that
//! constancy alone does not gate it — with the fix reverted the ridge is a
//! constant 0.0 across the whole scan and a distinct-count check still passes.
//! Only pinning the VALUE catches the revert, and it was confirmed to do so
//! (it fires on 104 observations of 0.0). Both checks are armed below.

#![cfg(test)]

use super::*;
use gam_problem::{InverseLink, LikelihoodSpec, ResponseFamily, StandardLink};
use gam_terms::smooth::BlockwisePenalty;
use ndarray::{Array1, Array2};

/// Deterministic ill-conditioned fixture: a degree-11 Vandermonde design on
/// `[0,1]` plus an intercept, under a second-difference penalty on the
/// polynomial block. The Vandermonde condition number at this degree is past
/// 1e16, so `XᵀWX + S_λ` is at the numerical floor in its highest modes and the
/// bare Cholesky is MARGINAL — which is the regime where a ridge chosen by
/// "did the factorization succeed" can answer differently at different ρ. A
/// well-conditioned fixture cannot exhibit the defect at all, so it would
/// clear the selector without testing it.
fn vandermonde_fixture(
    n: usize,
    k: usize,
    count_response: bool,
) -> (Array2<f64>, Array1<f64>, Vec<BlockwisePenalty>) {

    let p = 1 + k;
    let mut x = Array2::<f64>::zeros((n, p));
    let mut y = Array1::<f64>::zeros(n);
    for i in 0..n {
        let t = i as f64 / (n as f64 - 1.0);
        x[[i, 0]] = 1.0;
        let mut power = 1.0;
        for c in 0..k {
            x[[i, 1 + c]] = power;
            power *= t;
        }
        // Smooth truth plus a deterministic alternating perturbation, so the
        // fit has genuine residual spread without any RNG.
        let wiggle = if i % 2 == 0 { 0.05 } else { -0.05 };
        let signal = (2.0 * std::f64::consts::PI * t).sin() + 0.5 * t + wiggle;
        // Poisson needs a non-negative integer count; Gaussian takes the raw
        // signal. Same design either way, so the ridge decision sees the same
        // conditioning.
        y[i] = if count_response {
            (3.0 + 2.0 * signal).round().max(0.0)
        } else {
            signal
        };
    }
    let m = k - 2;
    let mut d = Array2::<f64>::zeros((m, k));
    for r in 0..m {
        d[[r, r]] = 1.0;
        d[[r, r + 1]] = -2.0;
        d[[r, r + 2]] = 1.0;
    }
    let s = d.t().dot(&d);
    (x, y, vec![BlockwisePenalty::new(1..(1 + k), s)])
}

/// Sweep ρ across the box and report the stabilization ridge and the cost at
/// every point. The ridge must be the same at all of them: it enters the
/// criterion through `0.5·log|H|`, so a ridge that changes with ρ makes the
/// criterion discontinuous in ρ.
fn ridge_sweep(
    family: ResponseFamily,
    link: StandardLink,
    n: usize,
    k: usize,
    label: &str,
) -> (Vec<f64>, String) {
    let (x, y, s_list) = vandermonde_fixture(n, k, matches!(family, ResponseFamily::Poisson));
    let weights = Array1::<f64>::ones(n);
    let offset = Array1::<f64>::zeros(n);
    let p = x.ncols();
    let specs: Vec<PenaltySpec> = s_list.iter().map(PenaltySpec::from_blockwise_ref).collect();
    let ext = ExternalOptimOptions {
        family: LikelihoodSpec::new(family, InverseLink::Standard(link)),
        latent_cloglog: None,
        mixture_link: None,
        optimize_mixture: false,
        sas_link: None,
        optimize_sas: false,
        compute_inference: true,
        skip_rho_posterior_inference: false,
        max_iter: 300,
        tol: 1e-7,
        nullspace_dims: vec![2],
        linear_constraints: None,
        firth_bias_reduction: Some(false),
        rho_prior: Default::default(),
        kronecker_penalty_system: None,
        kronecker_factored: None,
        persistent_warm_start_store: None,
    };
    let cfg = super::external_options::resolved_external_config(&ext)
        .expect("external config resolves")
        .0;
    let x_dm: DesignMatrix = x.clone().into();
    let (canonical, dims) = gam_terms::construction::canonicalize_penalty_specs(
        &specs,
        &ext.nullspace_dims,
        p,
        "ridge_sweep",
    )
    .expect("penalty canonicalizes");
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
        Some(dims),
        None,
        None,
    )
    .expect("outer REML state builds");
    state.set_link_states(
        cfg.link_kind.mixture_state().cloned(),
        cfg.link_kind.sas_state().copied(),
    );

    // A wide sweep, not a local ladder. The question is whether δ takes more
    // than one value ANYWHERE over the ρ box the optimizer can reach; a local
    // ladder only sees a flip if it happens to straddle the crossing.
    let mut ridges: Vec<f64> = Vec::new();
    let mut table = format!("\n  {label}");
    for rho_value in [
        -12.0_f64, -9.0, -6.0, -3.0, -1.5, -0.5, 0.0, 0.5, 1.5, 3.0, 6.0, 9.0, 12.0,
    ] {
        let rho = Array1::from(vec![rho_value]);
        let cost_text = match state
            .compute_outer_eval_with_order(&rho, crate::rho_optimizer::OuterEvalOrder::Value)
        {
            Ok(eval) => format!("{:.12e}", eval.cost),
            Err(err) => format!("ERR {err}"),
        };
        let ridge_text = match state.last_ridge_used() {
            Some(r) => {
                ridges.push(r);
                format!("{r:.6e}")
            }
            None => "none".to_string(),
        };
        table.push_str(&format!(
            "\n    rho={rho_value:>6.1}  ridge={ridge_text}  cost={cost_text}"
        ));
    }
    (ridges, table)
}

/// Scan the design conditioning until the bare factorization the unfixed
/// selectors try FIRST actually changes its answer across the ρ box.
///
/// A fixture that is too well conditioned never fails the bare attempt (ridge ≡
/// 0 everywhere) and one that is too ill conditioned always fails it (ridge ≡
/// δ everywhere); both pass a constant-ridge check without ever exercising the
/// branch. Only a design whose smallest mode CROSSES the factorization
/// threshold somewhere inside the box can show whether δ is a function of ρ. So
/// scan the Vandermonde degree and report, per degree, how many distinct ridges
/// the ρ sweep produced.
#[test]
fn stabilization_ridge_is_constant_along_rho_2519() {
    let distinct = |ridges: &[f64]| -> usize {
        let mut seen: Vec<u64> = ridges.iter().map(|r| r.to_bits()).collect();
        seen.sort_unstable();
        seen.dedup();
        seen.len()
    };
    let mut report = String::new();
    let mut gaussian_flips: Vec<usize> = Vec::new();
    let mut poisson_flips: Vec<usize> = Vec::new();
    let mut off_value: Vec<String> = Vec::new();
    for k in [6usize, 8, 9, 10, 11, 12, 14, 16] {
        let (gaussian_ridges, gaussian_table) = ridge_sweep(
            ResponseFamily::Gaussian,
            StandardLink::Identity,
            200,
            k,
            &format!("k={k} gaussian/identity (pls_solver branch)"),
        );
        let (poisson_ridges, poisson_table) = ridge_sweep(
            ResponseFamily::Poisson,
            StandardLink::Log,
            200,
            k,
            &format!("k={k} poisson/log (dense GLM selector)"),
        );
        let gd = distinct(&gaussian_ridges);
        let pd = distinct(&poisson_ridges);
        if gd > 1 {
            gaussian_flips.push(k);
        }
        if pd > 1 {
            poisson_flips.push(k);
        }
        for (family, ridges) in [
            ("gaussian/identity", &gaussian_ridges),
            ("poisson/log", &poisson_ridges),
        ] {
            for &ridge in ridges {
                if ridge != crate::pirls::FIXED_STABILIZATION_RIDGE {
                    off_value.push(format!("k={k} {family} delta={ridge:.6e}"));
                }
            }
        }
        report.push_str(&format!(
            "\n  k={k}: gaussian distinct ridges={gd}, poisson distinct ridges={pd}{gaussian_table}{poisson_table}"
        ));
    }
    let summary = format!("#2519 stabilization ridge across a degree x rho scan{report}");
    eprintln!("{summary}");
    assert!(
        poisson_flips.is_empty(),
        "the dense GLM selector changes the ridge with rho at degrees {poisson_flips:?}, \
         so delta is not constant along rho\n{summary}"
    );
    assert!(
        gaussian_flips.is_empty(),
        "the pls_solver selector changes the ridge with rho at degrees {gaussian_flips:?}, \
         so delta is not constant along rho\n{summary}"
    );
    // The VALUE gate, now ARMED. Constancy alone verifies nothing: a bare-first
    // selector on a fixture that factors bare everywhere is also constant, at
    // zero — that is exactly what this file measured before the repair (104
    // observations of 0.0 on the Gaussian arm, distinct-count 1 at every
    // degree). Only pinning the VALUE distinguishes "δ is applied at every ρ"
    // from "δ is never applied here", and it was confirmed to bite in that
    // direction.
    //
    // It was previously held back because always-on δ made the companion forms
    // wrong (the rail-face λ→∞ certificate refused outright: "the limit fit
    // needed a stabilization ridge (1.000e-8), so its criterion is not the
    // plain LAML this form expands"). `LamlFaceParts::stabilization_ridge` now
    // carries δ into `laml_rail_face_limit`, so that prerequisite is met and
    // both selectors apply δ unconditionally.
    assert!(
        off_value.is_empty(),
        "every fit must carry exactly FIXED_STABILIZATION_RIDGE = {:.6e}; a selector that \
         answers with any other delta is choosing it from the data or from a \
         factorization-success predicate, which makes 0.5*log|H| a discontinuous function \
         of rho (#2519). Off-value observations: {:?}\n{}",
        crate::pirls::FIXED_STABILIZATION_RIDGE,
        off_value,
        summary,
    );
}
