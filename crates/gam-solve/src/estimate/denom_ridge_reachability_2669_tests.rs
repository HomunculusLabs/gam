//! #2669 reachability probe: can `DENOM_RIDGE` bind, and what does the outer
//! REML criterion do when it does?
//!
//! `DENOM_RIDGE = 1e-8` (`reml_outer_engine/outer_entry_helpers.rs`) is applied
//! as `denom = (n - M_p).max(DENOM_RIDGE)` at four production sites, where
//! `M_p = p - rank(S_lambda)` is the UNPENALIZED coefficient count
//! (`reml/objective.rs`, `nullspace_dim = h_for_operator.ncols() - penalty_rank`).
//! Both operands are integer-valued, so the clamp cannot interpolate anything:
//! it is a hard branch that fires exactly when `n <= M_p` and replaces the
//! profiled Gaussian degrees of freedom by `1e-8`.
//!
//! The arithmetic consequence, if it fires, is that `phi = dp_c/1e-8` and the
//! data-fit term `dp_c/(2 phi) = denom/2 = 5e-9` INDEPENDENTLY OF THE DATA,
//! while `(denom/2) ln(2 pi phi)` collapses to the 1e-8 scale. For Gaussian
//! IDENTITY the weights are fixed, so `log|H|`, `log|S|` and the weight-log-sum
//! are all y-independent too: the whole REML criterion would go numerically
//! blind to the response.
//!
//! That is the measurable prediction, tested one variable at a time — ONE
//! design, ONE rho, TWO responses differing by ~50x, and a LADDER of penalty
//! layouts differing only in how many columns are penalized:
//!
//!   * `M_p = 23, 20, 16` -> `n - M_p = -15, -12, -8`, all deeply clamped;
//!   * `M_p = 6, 2`       -> `n - M_p = 2, 6`, healthy controls.
//!
//! A healthy REML criterion separates the two responses by O(n) nats. A clamped
//! one separates them at the 1e-8 scale.
//!
//! FIRST RUN (n=8, p=12, `M_p = 8 = n` vs `M_p = 6`) REFUTED the prediction:
//! the two arms separated by 3.294e1 and 3.130e1 nats respectively — i.e. the
//! `M_p = n` arm behaved exactly like the healthy control, so the clamp did NOT
//! bind. That is a negative result about THAT CONSTRUCTION, not about
//! reachability: the probe has no instrument on `nullspace_dim` or on which
//! `DispersionHandling` branch the evaluator took, so it cannot distinguish
//! "`M_p` never reached `n`" from "the ProfiledGaussian branch was never
//! entered". This ladder widens the margin to `n - M_p = -15` so that no
//! plausible off-by-a-few in `p - rank(S)` can explain a healthy result; if the
//! criterion still separates at O(n) nats across the whole ladder, the
//! conclusion is that this entry does not exercise the clamp at all and the
//! next instrument has to go at the clamp site itself.
//!
//! SECOND RUN (p=24 ladder, `n-M_p = -15,-12,-8,2,6`) ALSO came back healthy:
//! separations `3.299e1, 3.299e1, 3.298e1, 3.130e1, 3.130e1` — two tight
//! clusters, not five values, i.e. the criterion did NOT track the `from`
//! sweep at all. That localizes the error: `penalty_rank` is not the rank of
//! the block ridge I built, it follows the CALLER-DECLARED `nullspace_dims`,
//! which both earlier runs pinned at `vec![0]` — forcing `M_p ~ 0` and
//! `denom ~ n` on every rung. This run declares the block's TRUE nullity
//! within the p-dimensional space (`nullspace_dims = vec![from]`), so
//! `M_p = from` and `n - M_p` finally sweeps as intended. A REFUSED rung is
//! just as informative: it would mean upstream validation blocks the
//! declaration and the clamp is unreachable through this entry.

use super::*;
use gam_problem::{InverseLink, LikelihoodSpec, ResponseFamily, StandardLink};
use gam_terms::smooth::BlockwisePenalty;
use ndarray::{Array1, Array2};

const N: usize = 8;
const P: usize = 24;

/// 8 well-conditioned unpenalized columns (a DCT-II basis of R^8, so the
/// parametric block is exactly full rank and NOT rank-deficient) followed by 16
/// smooth-like columns that can carry the penalty.
fn build_design() -> Array2<f64> {
    let mut x = Array2::<f64>::zeros((N, P));
    for i in 0..N {
        let t = (i as f64 + 0.5) / N as f64;
        for j in 0..8 {
            x[[i, j]] = (std::f64::consts::PI * j as f64 * t).cos();
        }
        for j in 0..16 {
            x[[i, 8 + j]] = ((j as f64 + 1.0) * std::f64::consts::PI * t).sin()
                + 0.25 * (t - 0.5).powi((j % 4) as i32 + 1);
        }
    }
    x
}

fn cost_at(
    x: &Array2<f64>,
    y: &Array1<f64>,
    penalized_from: usize,
    declared_nullspace: usize,
    label: &str,
) -> Option<f64> {
    let weights = Array1::<f64>::ones(N);
    let offset = Array1::<f64>::zeros(N);
    let s_list = vec![BlockwisePenalty::ridge(penalized_from..P, 1.0)];
    let ext = ExternalOptimOptions {
        family: LikelihoodSpec::new(
            ResponseFamily::Gaussian,
            InverseLink::Standard(StandardLink::Identity),
        ),
        latent_cloglog: None,
        mixture_link: None,
        optimize_mixture: false,
        sas_link: None,
        optimize_sas: false,
        compute_inference: false,
        skip_rho_posterior_inference: true,
        max_iter: 100,
        tol: 1e-9,
        nullspace_dims: vec![declared_nullspace],
        linear_constraints: None,
        firth_bias_reduction: None,
        rho_prior: Default::default(),
        kronecker_penalty_system: None,
        kronecker_factored: None,
        persistent_warm_start_store: None,
    };
    let rho = Array1::from(vec![0.0_f64]);
    match evaluate_externalcost_andridge(
        y.view(),
        weights.view(),
        x.clone(),
        offset.view(),
        &s_list,
        &ext,
        &rho,
    ) {
        Ok((cost, ridge)) => {
            eprintln!("[#2669] {label}: cost={cost:.17e} ridge={ridge:.6e}");
            Some(cost)
        }
        Err(error) => {
            eprintln!("[#2669] {label}: REFUSED: {error}");
            None
        }
    }
}

#[test]
fn denom_ridge_reachability_probe_2669() {
    let x = build_design();
    let mut y1 = Array1::<f64>::zeros(N);
    let mut y2 = Array1::<f64>::zeros(N);
    for i in 0..N {
        let t = (i as f64 + 0.5) / N as f64;
        y1[i] = (3.0 * t).sin() + 0.1 * ((i * 7) % 5) as f64;
        y2[i] = 50.0 * y1[i] + 17.0;
    }
    eprintln!("[#2669] n={N} p={P}; penalty is a full-rank ridge on cols `from..{P}`");
    eprintln!("[#2669] so rank(S) = {P} - from  and  M_p = p - rank(S) = from");

    // (penalized_from, expected M_p) -- M_p equals `from` because the ridge is
    // full rank on the columns it covers.
    for &from in &[23usize, 20, 16, 6, 2] {
        let denom = N as f64 - from as f64;
        let clamped = denom <= 0.0;
        eprintln!(
            "[#2669] --- penalized {from}..{P}: M_p={from}, n-M_p={denom}, clamp predicted to bind = {clamped} ---"
        );
        let c1 = cost_at(&x, &y1, from, from, "y1");
        let c2 = cost_at(&x, &y2, from, from, "y2");
        match (c1, c2) {
            (Some(a), Some(b)) => eprintln!(
                "[#2669] M_p={from} n-M_p={denom} clamp_predicted={clamped} SEPARATION={:.6e}",
                (a - b).abs()
            ),
            _ => eprintln!("[#2669] M_p={from} n-M_p={denom} SEPARATION=unavailable (refused)"),
        }
    }
    eprintln!(
        "[#2669] READING: a SEPARATION at the 1e-8 scale means the criterion went blind to the"
    );
    eprintln!(
        "[#2669] response, i.e. the clamp bound. A separation of O(n) nats means it did not."
    );
}
