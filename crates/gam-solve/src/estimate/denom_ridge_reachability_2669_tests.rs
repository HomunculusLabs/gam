//! #2669 reachability probe: can `DENOM_RIDGE` bind, and what does the outer
//! REML criterion do when it does?
//!
//! `DENOM_RIDGE = 1e-8` (`reml_outer_engine/outer_entry_helpers.rs`) is applied
//! as `denom = (n - M_p).max(DENOM_RIDGE)` at four production sites, where
//! `M_p = p - rank(S_lambda)` is the UNPENALIZED coefficient count
//! (`reml/objective.rs`, `nullspace_dim = h.ncols() - penalty_rank`). Both
//! operands are integer-valued, so the clamp cannot interpolate anything: it is
//! a hard branch that fires exactly when `n <= M_p` and replaces the profiled
//! Gaussian degrees of freedom by `1e-8`.
//!
//! The arithmetic consequence, if it fires, is that
//! `phi = dp_c/1e-8` and the data-fit term `dp_c/(2 phi) = denom/2 = 5e-9`
//! INDEPENDENTLY OF THE DATA, while `(denom/2) ln(2 pi phi)` collapses to
//! ~1e-8 scale. For Gaussian identity every remaining term (`log|H|`,
//! `log|S|`, the weight-log-sum) is y-independent, so the whole REML criterion
//! becomes numerically blind to the response.
//!
//! That is the measurable prediction this probe tests, one variable at a time:
//! ONE design, ONE rho, TWO responses that differ by a factor of ~50, and two
//! penalty layouts that differ ONLY in how many columns are penalized:
//!
//!   * arm A: penalized cols `8..12` -> `M_p = 12 - 4 = 8 = n`, denom clamped.
//!   * arm B: penalized cols `6..12` -> `M_p = 12 - 6 = 6`, denom = 2, healthy.
//!
//! A healthy REML criterion separates the two responses by O(n) nats. If arm A
//! separates them by ~1e-8 while arm B separates them by ~1, the clamp is
//! reachable from the public `evaluate_externalcost_andridge` entry AND it
//! silently returns a criterion the outer optimizer then selects lambda
//! against. Printing rather than asserting a bound: this establishes
//! reachability first, which is the question the issue is blocked on.

use super::*;
use gam_problem::{InverseLink, LikelihoodSpec, ResponseFamily, StandardLink};
use gam_terms::smooth::BlockwisePenalty;
use ndarray::{Array1, Array2};

const N: usize = 8;
const P: usize = 12;

/// 8 well-conditioned unpenalized columns (a DCT-II basis of R^8, so the
/// parametric block is exactly full rank and NOT rank-deficient) followed by 4
/// polynomial columns that carry the penalty.
fn build_design() -> Array2<f64> {
    let mut x = Array2::<f64>::zeros((N, P));
    for i in 0..N {
        let t = (i as f64 + 0.5) / N as f64;
        for j in 0..8 {
            x[[i, j]] = (std::f64::consts::PI * j as f64 * t).cos();
        }
        for j in 0..4 {
            x[[i, 8 + j]] = (t - 0.5).powi(j as i32 + 1);
        }
    }
    x
}

fn cost_at(
    x: &Array2<f64>,
    y: &Array1<f64>,
    penalized_from: usize,
    label: &str,
) -> Result<(f64, f64), EstimationError> {
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
        nullspace_dims: vec![0],
        linear_constraints: None,
        firth_bias_reduction: None,
        rho_prior: Default::default(),
        kronecker_penalty_system: None,
        kronecker_factored: None,
        persistent_warm_start_store: None,
    };
    let rho = Array1::from(vec![0.0_f64]);
    let out = evaluate_externalcost_andridge(
        y.view(),
        weights.view(),
        x.clone(),
        offset.view(),
        &s_list,
        &ext,
        &rho,
    );
    match &out {
        Ok((cost, ridge)) => {
            println!("[#2669] {label}: cost={cost:.17e} ridge={ridge:.6e}");
        }
        Err(error) => {
            println!("[#2669] {label}: REFUSED: {error}");
        }
    }
    out
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
    println!("[#2669] n={N} p={P}");
    println!("[#2669] --- arm A: penalized cols 8..12, M_p = 12-4 = 8 = n (clamp predicted TO BIND) ---");
    let a1 = cost_at(&x, &y1, 8, "A/y1");
    let a2 = cost_at(&x, &y2, 8, "A/y2");
    println!("[#2669] --- arm B: penalized cols 6..12, M_p = 12-6 = 6, denom = 2 (control) ---");
    let b1 = cost_at(&x, &y1, 6, "B/y1");
    let b2 = cost_at(&x, &y2, 6, "B/y2");

    if let (Ok((a1c, _)), Ok((a2c, _))) = (&a1, &a2) {
        println!(
            "[#2669] ARM A response separation |cost(y1)-cost(y2)| = {:.6e}",
            (a1c - a2c).abs()
        );
    }
    if let (Ok((b1c, _)), Ok((b2c, _))) = (&b1, &b2) {
        println!(
            "[#2669] ARM B response separation |cost(y1)-cost(y2)| = {:.6e}",
            (b1c - b2c).abs()
        );
    }
    println!(
        "[#2669] VERDICT INPUT: arm A reached the evaluator = {}, arm B reached it = {}",
        a1.is_ok() && a2.is_ok(),
        b1.is_ok() && b2.is_ok()
    );
}
