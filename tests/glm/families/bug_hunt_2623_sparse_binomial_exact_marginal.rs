//! #2623 accuracy gate for the #784 higher-order LAML correction.
//!
//! Objective/gradient agreement is necessary but not sufficient: two channels
//! can agree perfectly while describing the wrong statistical objective.  This
//! fixture makes the exact marginal likelihood independently computable.  It is
//! a near-separated binomial cell with one Gaussian-penalized coefficient,
//! so the coefficient posterior is strongly skewed and the ordinary Laplace
//! approximation is precisely the approximation #784 exists to repair.
//!
//! For each smoothing strength we evaluate
//!
//!     M(lambda) = E_z [exp(y_sum beta) / (1 + exp(beta))^n],
//!     beta = z / sqrt(lambda), z ~ N(0,1),
//!
//! by high-resolution deterministic Simpson integration.  The normalizing
//! `sqrt(lambda)` term is already absorbed by the standard-normal change of
//! variables, making `-log M(lambda)` the exact normalized marginal criterion.
//! gam's LAML cost may differ by one rho-independent constant, so the test grades
//! the variation of `(cost - exact)` over rho.  It also reconstructs plain
//! Laplace as `corrected_cost + Delta_b` from the live audit and requires the
//! higher-order correction to reduce that surface error materially.

use gam::estimate::outer_eval_capture::{enable_rho_outer_audit, take_rho_outer_audit};
use gam::estimate::{ExternalOptimOptions, evaluate_externalcost_andridge};
use gam::smooth::BlockwisePenalty;
use gam::types::{InverseLink, LikelihoodSpec, ResponseFamily, StandardLink};
use ndarray::{Array1, Array2, array};

fn softplus(x: f64) -> f64 {
    if x > 0.0 {
        x + (-x).exp().ln_1p()
    } else {
        x.exp().ln_1p()
    }
}

/// Independent one-dimensional oracle after beta = z / sqrt(lambda).
fn exact_negative_log_marginal(rho: f64, observations: usize, successes: usize) -> f64 {
    const Z_LIMIT: f64 = 12.0;
    const PANELS: usize = 65_536;
    const INV_SQRT_TWO_PI: f64 = 0.398_942_280_401_432_7;

    let inv_sqrt_lambda = (-0.5 * rho).exp();
    let step = 2.0 * Z_LIMIT / PANELS as f64;
    let integrand = |z: f64| {
        let beta = z * inv_sqrt_lambda;
        (-0.5 * z * z + successes as f64 * beta
            - observations as f64 * softplus(beta))
        .exp()
            * INV_SQRT_TWO_PI
    };

    let mut sum = integrand(-Z_LIMIT) + integrand(Z_LIMIT);
    for index in 1..PANELS {
        let z = -Z_LIMIT + index as f64 * step;
        sum += if index % 2 == 0 { 2.0 } else { 4.0 } * integrand(z);
    }
    let marginal = sum * step / 3.0;
    assert!(
        marginal.is_finite() && marginal > 0.0,
        "exact marginal integral must be finite and positive, got {marginal}"
    );
    -marginal.ln()
}

fn options() -> ExternalOptimOptions {
    ExternalOptimOptions {
        latent_cloglog: None,
        mixture_link: None,
        optimize_mixture: false,
        sas_link: None,
        optimize_sas: false,
        family: LikelihoodSpec::new(
            ResponseFamily::Binomial,
            InverseLink::Standard(StandardLink::Logit),
        ),
        compute_inference: true,
        skip_rho_posterior_inference: false,
        max_iter: 300,
        tol: 1.0e-12,
        nullspace_dims: vec![0],
        linear_constraints: None,
        firth_bias_reduction: None,
        rho_prior: Default::default(),
        kronecker_penalty_system: None,
        kronecker_factored: None,
        persistent_warm_start_store: None,
    }
}

#[test]
fn higher_order_laml_tracks_exact_sparse_binomial_marginal_2623() {
    gam::init_parallelism();

    let observations = 12usize;
    let successes = 1usize;
    let mut y = Array1::<f64>::zeros(observations);
    y[0] = 1.0;
    let weights = Array1::<f64>::ones(observations);
    let design = Array2::<f64>::ones((observations, 1));
    let offset = Array1::<f64>::zeros(observations);
    let penalties = vec![BlockwisePenalty::new(0..1, array![[1.0]])];
    let opts = options();

    let mut rows = Vec::new();
    for half_step in -14..=-2 {
        let rho_scalar = half_step as f64 * 0.5;
        let rho = array![rho_scalar];
        enable_rho_outer_audit();
        let corrected_cost = evaluate_externalcost_andridge(
            y.view(),
            weights.view(),
            design.clone(),
            offset.view(),
            &penalties,
            &opts,
            &rho,
        )
        .expect("sparse-binomial LAML evaluation")
        .0;
        let audit = take_rho_outer_audit().expect("rho audit armed");
        if !audit.quadrature_marginal_engaged {
            continue;
        }
        let correction = audit
            .quadrature_marginal
            .expect("an engaged correction publishes its value and certificate");
        assert_eq!(
            correction.block_cols.len(),
            1,
            "the one-coefficient oracle must integrate exactly one direction"
        );
        let exact_cost = exact_negative_log_marginal(rho_scalar, observations, successes);
        let plain_laplace_cost = corrected_cost + correction.delta_b;
        eprintln!(
            "#2623 exact sparse binomial: rho={rho_scalar:+.2} exact={exact_cost:.10e} \
             corrected={corrected_cost:.10e} plain={plain_laplace_cost:.10e} \
             Delta_b={:.4e} quadrature_error={:.3e} nodes={}",
            correction.delta_b,
            correction.quadrature_error,
            correction.node_count,
        );
        rows.push((
            rho_scalar,
            exact_cost,
            corrected_cost,
            plain_laplace_cost,
            correction.quadrature_error,
        ));
    }

    assert!(
        rows.len() >= 3,
        "the exact sparse-binomial gate needs at least three engaged rho cells, got {}",
        rows.len()
    );

    let span = |values: &[f64]| {
        let minimum = values.iter().copied().fold(f64::INFINITY, f64::min);
        let maximum = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        maximum - minimum
    };
    let corrected_residuals: Vec<f64> = rows
        .iter()
        .map(|(_, exact, corrected, _, _)| corrected - exact)
        .collect();
    let plain_residuals: Vec<f64> = rows
        .iter()
        .map(|(_, exact, _, plain, _)| plain - exact)
        .collect();
    let corrected_surface_error = span(&corrected_residuals);
    let plain_surface_error = span(&plain_residuals);
    let largest_certificate = rows
        .iter()
        .map(|row| row.4)
        .fold(0.0_f64, f64::max);

    eprintln!(
        "#2623 exact sparse binomial summary: engaged={} corrected_surface_error={:.4e} \
         plain_surface_error={:.4e} improvement={:.2}x max_certificate={:.3e}",
        rows.len(),
        corrected_surface_error,
        plain_surface_error,
        plain_surface_error / corrected_surface_error.max(f64::MIN_POSITIVE),
        largest_certificate,
    );

    assert!(
        corrected_surface_error <= 4.0 * largest_certificate + 2.0e-6,
        "the correction's error certificate does not cover its exact marginal-surface error: \
         observed {corrected_surface_error:.4e}, max certificate {largest_certificate:.4e}"
    );
    assert!(
        corrected_surface_error < 0.25 * plain_surface_error,
        "the higher-order correction must materially improve on plain Laplace in the sparse, \
         separated regime: corrected surface error {corrected_surface_error:.4e}, plain \
         Laplace {plain_surface_error:.4e}"
    );
}
