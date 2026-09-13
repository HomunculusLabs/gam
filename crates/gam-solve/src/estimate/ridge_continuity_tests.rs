//! #2519 / #2901 V22: the penalized Hessian every PIRLS fit hands the outer
//! criterion is exactly `XᵀWX + S_λ`, at every ρ.
//!
//! The ridge selectors (the dense and sparse penalized-Hessian certificates in
//! `newton_solve`, and `pls_solver`'s Gaussian-identity branches) once added δ
//! to H only when a bare factorization failed. That made δ a function of ρ
//! through a Cholesky-success predicate and moved the #1575 cost by
//! `0.5·ln(1e8) = 9.2103` between neighbouring ρ (#2519). They then added a
//! fixed δ = 1e-8 at every ρ. That was continuous, but it put a coefficient-space
//! ridge and a magic constant into the REML criterion (SPEC rules 5 and 23). No
//! PIRLS path adds a ridge now, so the criterion is continuous in ρ because
//! nothing is added to it.
//!
//! No ridge value is left to report, so this file checks the identity itself.
//! Across a Vandermonde degree × ρ scan, on the Gaussian-identity PLS branch and
//! the dense GLM Newton branch, it rebuilds `XₜᵀWXₜ + S_t` from each fit's own
//! transformed design, final weights and transformed penalty. It then compares
//! that with the `penalized_hessian_transformed` the criterion reads, entry by
//! entry, against the assembly rounding band `n·p·ε·max|H|`. A shift above the
//! band is a stabilization ridge, fixed or adaptive.

#![cfg(test)]

use super::*;
use gam_problem::{InverseLink, LikelihoodSpec, ResponseFamily, StandardLink};
use gam_terms::smooth::BlockwisePenalty;
use ndarray::{Array1, Array2, Axis};

/// The magnitude of the retired `FIXED_STABILIZATION_RIDGE`. The scan must
/// resolve a diagonal shift this large somewhere, or it could not have seen
/// the ridge it guards against.
const RETIRED_RIDGE: f64 = 1.0e-8;

/// Deterministic ill-conditioned fixture: a degree-11 Vandermonde design on
/// `[0,1]` plus an intercept, under a second-difference penalty on the
/// polynomial block. The Vandermonde condition number at this degree is past
/// 1e16, so `XᵀWX + S_λ` is at the numerical floor in its highest modes and the
/// bare Cholesky is MARGINAL. That is the regime where a ridge chosen by "did
/// the factorization succeed" could answer differently at different ρ, and
/// where the rank-deficient minimum-norm solve is exercised.
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
        // signal. Same design either way, so both branches see the same
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

/// One scanned fit: the largest entrywise gap between the penalized Hessian
/// the criterion reads and `XₜᵀWXₜ + S_t` rebuilt from the same fit, the
/// largest diagonal gap, and the assembly rounding band they are judged against.
struct HessianIdentity {
    rho: f64,
    max_gap: f64,
    max_diagonal_gap: f64,
    band: f64,
}

/// Sweep ρ across the box and rebuild `XₜᵀWXₜ + S_t` at every fit that
/// produces an evaluation bundle.
fn hessian_identity_sweep(
    family: ResponseFamily,
    link: StandardLink,
    n: usize,
    k: usize,
    label: &str,
) -> (Vec<HessianIdentity>, String) {
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
        "hessian_identity_sweep",
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

    // A wide sweep, not a local ladder: the identity has to hold over the whole
    // ρ box the optimizer can reach.
    let mut observations: Vec<HessianIdentity> = Vec::new();
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
        let identity_text = match state.obtain_eval_bundle(&rho) {
            Ok(bundle) => {
                let pr = &bundle.pirls_result;
                let x_t = pr.x_transformed.to_dense();
                let weighted = &x_t * &pr.finalweights.view().insert_axis(Axis(1));
                let mut expected = x_t.t().dot(&weighted);
                expected += &pr.reparam_result.s_transformed;
                let h = pr.penalized_hessian_transformed.to_dense();
                assert_eq!(
                    h.dim(),
                    expected.dim(),
                    "{label} rho={rho_value}: the criterion's Hessian and the rebuilt XtWX + S \
                     must share one coefficient frame"
                );
                let scale = h
                    .iter()
                    .chain(expected.iter())
                    .fold(0.0_f64, |acc, value| acc.max(value.abs()));
                let band = (n * h.nrows()) as f64 * f64::EPSILON * scale;
                let max_gap = h
                    .iter()
                    .zip(expected.iter())
                    .fold(0.0_f64, |acc, (a, b)| acc.max((a - b).abs()));
                let max_diagonal_gap = (0..h.nrows())
                    .fold(0.0_f64, |acc, i| acc.max((h[[i, i]] - expected[[i, i]]).abs()));
                observations.push(HessianIdentity {
                    rho: rho_value,
                    max_gap,
                    max_diagonal_gap,
                    band,
                });
                format!(
                    "max|H-(XtWX+S)|={max_gap:.3e}  max diagonal gap={max_diagonal_gap:.3e}  \
                     band={band:.3e}"
                )
            }
            Err(err) => format!("bundle unavailable: {err}"),
        };
        table.push_str(&format!(
            "\n    rho={rho_value:>6.1}  {identity_text}  cost={cost_text}"
        ));
    }
    (observations, table)
}

#[test]
fn penalized_hessian_carries_no_stabilization_ridge_along_rho_2519() {
    let mut report = String::new();
    let mut violations: Vec<String> = Vec::new();
    let mut observed = 0usize;
    let mut resolving = 0usize;
    for k in [6usize, 8, 9, 10, 11, 12, 14, 16] {
        for (family, link, branch) in [
            (
                ResponseFamily::Gaussian,
                StandardLink::Identity,
                "gaussian/identity (pls_solver branch)",
            ),
            (
                ResponseFamily::Poisson,
                StandardLink::Log,
                "poisson/log (dense GLM Newton branch)",
            ),
        ] {
            let label = format!("k={k} {branch}");
            let (observations, table) = hessian_identity_sweep(family, link, 200, k, &label);
            report.push_str(&table);
            for observation in &observations {
                observed += 1;
                if observation.band < RETIRED_RIDGE {
                    resolving += 1;
                }
                if !(observation.max_gap <= observation.band) {
                    violations.push(format!(
                        "{label} rho={:.1}: max gap {:.3e} (diagonal {:.3e}) above band {:.3e}",
                        observation.rho,
                        observation.max_gap,
                        observation.max_diagonal_gap,
                        observation.band
                    ));
                }
            }
        }
    }
    let summary =
        format!("#2519/#2901 V22 penalized Hessian identity across a degree x rho scan{report}");
    eprintln!("{summary}");
    assert!(
        observed > 0,
        "no scanned fit produced an evaluation bundle, so the identity was never measured\n{summary}"
    );
    assert!(
        resolving > 0,
        "no scanned fit has an assembly band below {RETIRED_RIDGE:.0e}, so the scan could not \
         have seen the retired stabilization ridge\n{summary}"
    );
    assert!(
        violations.is_empty(),
        "the penalized Hessian the criterion reads must be exactly XtWX + S_lambda; a gap above \
         the assembly band is a stabilization ridge added to H (#2901 V22), and one chosen from \
         a factorization-success predicate makes 0.5*log|H| a discontinuous function of rho \
         (#2519). Violations: {violations:?}\n{summary}"
    );
}
