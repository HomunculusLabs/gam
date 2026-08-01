//! Measurement probe for #2728: how wide is `beta_covariance_corrected()`
//! relative to `beta_covariance()`, and relative to the estimator's own
//! Monte-Carlo sampling spread, on the anisotropic-Duchon-on-PC fixture.
//!
//! The design matrix `X` is held FIXED across replicates and only the Gaussian
//! noise is redrawn, so `beta_hat` is comparable across replicates and the
//! Monte-Carlo spread of `x' beta_hat` is exactly the quantity the published
//! covariance claims to be.

use gam::basis::{
    CenterStrategy, DuchonBasisSpec, DuchonNullspaceOrder, DuchonOperatorPenaltySpec,
    duchon_max_active_operator_derivative_order, resolve_duchon_orders,
};
use gam::estimate::{FitOptions, UnifiedFitResult};
use gam::smooth::{
    ShapeConstraint, SmoothBasisSpec, SmoothTermSpec, TermCollectionSpec,
    build_term_collection_design, fit_term_collection_forspec, freeze_term_collection_from_design,
};
use gam::types::{InverseLink, LikelihoodSpec, ResponseFamily, StandardLink};
use ndarray::{Array1, Array2, ArrayView2};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};
use std::time::Instant;

const NOISE_SD: f64 = 0.30;
const HYBRID_LENGTH_SCALE: f64 = 1.0;

/// Route the library's `log` records to stdout so the probe sees the
/// `[smoothing-correction]` branch decision and the `[INDEF-HESS]` dump.
struct StdoutLogger;

impl log::Log for StdoutLogger {
    fn enabled(&self, _metadata: &log::Metadata<'_>) -> bool {
        true
    }

    fn log(&self, record: &log::Record<'_>) {
        println!("[{}] {}", record.level(), record.args());
    }

    fn flush(&self) {}
}

static STDOUT_LOGGER: StdoutLogger = StdoutLogger;

fn gaussian_identity_likelihood() -> LikelihoodSpec {
    LikelihoodSpec::new(
        ResponseFamily::Gaussian,
        InverseLink::Standard(StandardLink::Identity),
    )
}

fn truth(row: &[f64]) -> f64 {
    let mut linear = 0.0;
    let coefs = [0.55, -0.40, 0.30, 0.20, -0.15, 0.10];
    for (j, &xj) in row.iter().enumerate() {
        if j < coefs.len() {
            linear += coefs[j] * xj;
        }
    }
    let mut dist2 = 0.0;
    for (j, &xj) in row.iter().enumerate() {
        let cj = match j {
            0 => 0.30,
            1 => -0.20,
            2 => 0.10,
            _ => 0.0,
        };
        let d = xj - cj;
        dist2 += d * d;
    }
    let radial_bump = (-dist2 / (2.0 * 0.8 * 0.8)).exp();
    let sinusoid = 0.4 * (std::f64::consts::PI * row[0]).sin();
    linear + radial_bump + sinusoid
}

fn simulate_x(n: usize, pc_dim: usize, seed: u64) -> (Array2<f64>, Array1<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0, 1.0).expect("normal params must be valid");
    let mut x = Array2::<f64>::zeros((n, pc_dim));
    let mut y_true = Array1::<f64>::zeros(n);
    for i in 0..n {
        let mut row = vec![0.0_f64; pc_dim];
        for j in 0..pc_dim {
            let v = normal.sample(&mut rng);
            x[[i, j]] = v;
            row[j] = v;
        }
        y_true[i] = truth(&row);
    }
    (x, y_true)
}

fn noisy(y_true: &Array1<f64>, seed: u64) -> Array1<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let noise = Normal::new(0.0, NOISE_SD).expect("noise params must be valid");
    y_true.mapv(|v| v + noise.sample(&mut rng))
}

fn duchon_aniso_pc_spec(name: &str, pc_dim: usize, k_centers: usize) -> TermCollectionSpec {
    let operator_penalties = DuchonOperatorPenaltySpec::default();
    let (nullspace_order, power) = resolve_duchon_orders(
        pc_dim,
        DuchonNullspaceOrder::Linear,
        duchon_max_active_operator_derivative_order(&operator_penalties),
        Some(HYBRID_LENGTH_SCALE),
    );
    TermCollectionSpec {
        linear_terms: vec![],
        random_effect_terms: vec![],
        smooth_terms: vec![SmoothTermSpec {
            name: name.to_string(),
            basis: SmoothBasisSpec::Duchon {
                feature_cols: (0..pc_dim).collect(),
                spec: DuchonBasisSpec {
                    radial_reparam: None,
                    center_strategy: CenterStrategy::FarthestPoint {
                        num_centers: k_centers,
                    },
                    length_scale: Some(HYBRID_LENGTH_SCALE),
                    power: power as f64,
                    nullspace_order,
                    identifiability: gam::basis::SpatialIdentifiability::default(),
                    aniso_log_scales: Some(vec![0.0; pc_dim]),
                    operator_penalties,
                    periodic: None,
                    boundary: gam::basis::OneDimensionalBoundary::Open,
                },
                input_scale: None,
            },
            shape: ShapeConstraint::None,
            joint_null_rotation: None,
        }],
    }
}

fn fit_options(max_iter: usize) -> FitOptions {
    FitOptions {
        resource_policy: gam_runtime::resource::ResourcePolicy::default_library(),
        latent_cloglog: None,
        mixture_link: None,
        optimize_mixture: false,
        sas_link: None,
        optimize_sas: false,
        compute_inference: true,
        skip_rho_posterior_inference: false,
        max_iter,
        tol: 1e-5,
        nullspace_dims: vec![],
        linear_constraints: None,
        firth_bias_reduction: false,
        adaptive_regularization: None,
        rho_prior: Default::default(),
        kronecker_penalty_system: None,
        kronecker_factored: None,
        persistent_warm_start_store: None,
    }
}

fn row_se(design: ArrayView2<'_, f64>, covariance: &Array2<f64>) -> Array1<f64> {
    let mut se = Array1::<f64>::zeros(design.nrows());
    for (i, row) in design.outer_iter().enumerate() {
        let v = row.dot(&covariance.dot(&row));
        se[i] = v.max(0.0).sqrt();
    }
    se
}

fn rms(values: &Array1<f64>) -> f64 {
    (values.iter().map(|v| v * v).sum::<f64>() / values.len() as f64).sqrt()
}

fn report(tag: &str, fit: &UnifiedFitResult, design: ArrayView2<'_, f64>) {
    let vb = fit.beta_covariance().expect("conditional covariance").clone();
    let vp = fit
        .beta_covariance_corrected()
        .expect("corrected covariance")
        .clone();
    let se_b = row_se(design, &vb);
    let se_p = row_se(design, &vp);
    let (rb, rp) = (rms(&se_b), rms(&se_p));
    println!(
        "[{tag}] method={:?} first_order={:?} RMS(se_cond)={rb:.5} RMS(se_corr)={rp:.5} ratio={:.3}",
        fit.smoothing_correction_method(),
        fit.smoothing_correction_method_first_order(),
        rp / rb,
    );
    println!(
        "[{tag}] lambdas={:?}",
        fit.lambdas.iter().map(|v| format!("{v:.4e}")).collect::<Vec<_>>()
    );
    println!(
        "[{tag}] edf={:?} phi={:?} |g|={:?} outer_iters={}",
        fit.edf_total(),
        fit.dispersion_phi().ok(),
        fit.outer_gradient_norm,
        fit.outer_iterations,
    );
    if let Some(correction) = fit.smoothing_correction() {
        let trace_corr: f64 = correction.diag().iter().sum();
        let trace_vb: f64 = vb.diag().iter().sum();
        println!("[{tag}] tr(correction)={trace_corr:.6e} tr(Vb)={trace_vb:.6e}");
    }
    if let Some(first_order) = fit.smoothing_correction_first_order() {
        let trace_fo: f64 = first_order.diag().iter().sum();
        let mut vb_plus_fo = vb.clone();
        vb_plus_fo += first_order;
        let se_fo = row_se(design, &vb_plus_fo);
        println!(
            "[{tag}] tr(first_order)={trace_fo:.6e} RMS(se from Vb+first_order)={:.5}",
            rms(&se_fo)
        );
    }
    if let Some(v_rho) = fit.artifacts.rho_covariance.as_ref() {
        let diag: Vec<String> = v_rho.diag().iter().map(|v| format!("{v:.3e}")).collect();
        println!("[{tag}] diag(V_rho)={diag:?}");
    }
    assert!(rb.is_finite() && rp.is_finite(), "SE summaries must be finite");
}

#[test]
fn probe_2728_corrected_vs_conditional_width() {
    if log::set_logger(&STDOUT_LOGGER).is_ok() {
        log::set_max_level(log::LevelFilter::Trace);
    }
    let pc_dim = 4usize;
    let k_centers = 80usize;
    let n_train = 4_000usize;
    let n_eval = 400usize;
    // This was an `env::var` read, which `build.rs` bans outright and which
    // therefore made EVERY build of the workspace fail, not just this probe's.
    // The value is the one every unset-environment run already took, so the
    // observable behaviour of the probe is unchanged.
    //
    // NOTE FOR #2728: at 1 replicate the `n_replicates >= 3` Monte-Carlo arm
    // below cannot fire. It could not fire before this commit either — nothing
    // in the tree ever set the variable — so this makes an existing dead branch
    // visible rather than creating one. Raising this constant is how that arm
    // gets exercised; a re-added environment read is not.
    let n_replicates: usize = 1;
    let spec = duchon_aniso_pc_spec("duchon_pc_probe", pc_dim, k_centers);

    let (x_eval, _) = simulate_x(n_eval, pc_dim, 0x0EFA_1000_0000_0001);
    let (x_train, y_true) = simulate_x(n_train, pc_dim, 0xB10B_0001_0001_0001);
    let weights = Array1::ones(n_train);
    let offset = Array1::<f64>::zeros(n_train);

    let mut eta_draws: Vec<Array1<f64>> = Vec::new();
    let mut eval_dense: Option<Array2<f64>> = None;

    for replicate in 0..n_replicates {
        let y = noisy(&y_true, 0x51E5_0000_0000_0000 ^ replicate as u64);
        let start = Instant::now();
        let fitted = fit_term_collection_forspec(
            x_train.view(),
            y.view(),
            weights.view(),
            offset.view(),
            &spec,
            gaussian_identity_likelihood(),
            &fit_options(30),
        )
        .expect("fit");
        if eval_dense.is_none() {
            let frozen = freeze_term_collection_from_design(&spec, &fitted.design)
                .expect("freeze trained spec");
            let built = build_term_collection_design(x_eval.view(), &frozen)
                .expect("held-out design build");
            eval_dense = Some(built.design.to_dense());
        }
        let design = eval_dense.as_ref().expect("held-out design");
        println!("[replicate {replicate}] fit in {:.1?}", start.elapsed());
        if replicate == 0 {
            report("single", &fitted.fit, design.view());
        }
        eta_draws.push(design.dot(&fitted.fit.beta));
    }

    if n_replicates >= 3 {
        let design = eval_dense.as_ref().expect("held-out design");
        let m = n_replicates as f64;
        let mut mc_sd = Array1::<f64>::zeros(design.nrows());
        for i in 0..design.nrows() {
            let mean: f64 = eta_draws.iter().map(|d| d[i]).sum::<f64>() / m;
            let var: f64 = eta_draws
                .iter()
                .map(|d| (d[i] - mean) * (d[i] - mean))
                .sum::<f64>()
                / (m - 1.0);
            mc_sd[i] = var.sqrt();
        }
        println!("[mc] replicates={n_replicates} RMS(mc_sd)={:.5}", rms(&mc_sd));
    }
}
