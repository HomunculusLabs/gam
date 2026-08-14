//! #1122 / #901 iso-κ gradient class — the FULL joint-REML objective gradient
//! w.r.t. the Matérn isotropic length-scale coordinate `ψ = log κ` must match a
//! central finite difference of the COMPLETE outer criterion, not just the
//! penalty block.
//!
//! The penalty-only basis test
//! (`basis_matern_double_penalty_log_kappa_derivative_fd`) checks only
//! `∂(penalty matrices)/∂log_κ`. The full profiled REML objective
//!   V(κ) = data-fit(deviance) + ½log|H+Sλ| − ½log|Sλ|₊ + penalty-quad
//! also depends on κ through the Matérn DESIGN columns (and hence the deviance
//! and `H = XᵀWX`) and through the H-side of the `½log|H+Sλ|` term. A stall with
//! a large residual gradient at the iteration cap is the signature of an
//! objective↔gradient DESYNC: the optimizer follows a gradient inconsistent with
//! the objective it is minimizing.
//!
//! The original root cause was κ-dependent numerical null classification
//! combined with an identifiability chart `Z` that was not frozen across
//! per-trial rebuilds. Matérn topology is now structural (only an explicitly
//! appended intercept is a null direction), and `Z` is frozen at the first
//! rebuild and mirrored onto the spec read by the analytic ψ-gradient. Value
//! and gradient therefore share one fixed chart and topology at every trial.
//!
//! The generic outer runner exposes an opt-in structured finite-difference
//! record at its real bounded seed. This test fits an ordinary Gaussian 2-D
//! surface with a single `matern(x1, x2)` smooth (default double-penalty), then
//! asserts that the typed Matérn log-κ component matches the finite difference
//! of the full criterion.
//!
//! Reference-as-truth: every assertion is against the analytic FD of gam's own
//! profiled REML criterion — never another tool's output.

use gam::{
    FitConfig, FitRequest, FitResult, StandardFitRequest, encode_recordswith_inferred_schema,
    estimate::FitOptions,
    fit_model,
    smooth::{
        ShapeConstraint, SmoothBasisSpec, SmoothTermSpec, SpatialLengthScaleOptimizationOptions,
        TermCollectionSpec,
    },
    terms::basis::{CenterStrategy, MaternBasisSpec, MaternIdentifiability, MaternNu},
    types::{InverseLink, LikelihoodSpec, ResponseFamily, StandardLink},
};
use ndarray::{Array1, Array2};

fn init() {
    #[cfg(target_os = "macos")]
    gam::gpu::configure_global_policy(gam::gpu::GpuPolicy::Off);
    gam::init_parallelism();
}

use gam::utils::splitmix64;
fn next_unit(state: &mut u64) -> f64 {
    (splitmix64(state) >> 11) as f64 / (1u64 << 53) as f64
}
fn next_gauss(state: &mut u64) -> f64 {
    let u1 = next_unit(state).max(1.0e-12);
    let u2 = next_unit(state);
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

fn truth(a: f64, b: f64) -> f64 {
    (2.0 * std::f64::consts::PI * a).sin() * (2.0 * std::f64::consts::PI * b).sin()
}

fn build_dataset(n: usize, sigma: f64, seed: u64) -> gam::inference::data::EncodedDataset {
    let mut st = seed;
    let mut header = String::from("y,x1,x2\n");
    let mut body = String::new();
    for _ in 0..n {
        let a = next_unit(&mut st);
        let b = next_unit(&mut st);
        let y = truth(a, b) + sigma * next_gauss(&mut st);
        body.push_str(&format!("{y:.6},{a:.6},{b:.6}\n"));
    }
    header.push_str(&body);
    let mut rdr = csv::ReaderBuilder::new().from_reader(header.as_bytes());
    let records: Vec<csv::StringRecord> = rdr.records().map(|r| r.unwrap()).collect();
    let headers = vec!["y".to_string(), "x1".to_string(), "x2".to_string()];
    encode_recordswith_inferred_schema(headers, records).expect("encode dataset")
}

fn aniso_signal_dataset(n: usize) -> (Array2<f64>, Array1<f64>) {
    let mut x = Array2::<f64>::zeros((n, 2));
    let mut y = Array1::<f64>::zeros(n);
    for i in 0..n {
        let x1 = (i as f64) / (n as f64 - 1.0) * 6.0 - 3.0;
        let x2 = ((i as f64 * 0.618_033_988_749_894_9).fract()) * 6.0 - 3.0;
        x[[i, 0]] = x1;
        x[[i, 1]] = x2;
        y[i] = (2.0 * x1).sin();
    }
    (x, y)
}

/// MERGE GATE (#1122 / #901): the analytic outer REML gradient w.r.t. the
/// Matérn log-κ coordinate matches a central finite difference of the FULL
/// profiled REML criterion at θ₀ — data-fit + logdet (both Sλ-side AND H-side) +
/// penalty-quad — with the default double-penalty (nullspace shrinkage) active.
#[test]
fn matern_2d_iso_kappa_outer_gradient_matches_fd() {
    init();
    gam::estimate::enable_outer_gradient_fd_capture(1);
    let data = build_dataset(150, 0.05, 0x9A7E_7212_0001u64);
    let formula = "y ~ matern(x1, x2)";
    let config = FitConfig {
        family: Some("gaussian".to_string()),
        // Keep the prerequisite rho-only REML profile on its production
        // budget. Capping `outer_max_iter` here prevents that profile from
        // certifying, so the intended joint [rho, log-kappa] problem is never
        // constructed. Only the spatial loop may be capped: its structured
        // audit fires at the first bounded joint seed.
        spatial_optimization: SpatialLengthScaleOptimizationOptions {
            max_outer_iter: 2,
            ..SpatialLengthScaleOptimizationOptions::default()
        },
        gpu_policy: if cfg!(target_os = "macos") {
            gam::gpu::GpuPolicy::Off
        } else {
            gam::gpu::GpuPolicy::Auto
        },
        ..FitConfig::default()
    };
    match gam::fit_from_formula(formula, &data, &config) {
        Ok(_) => eprintln!("[FD-DIAG] matern(x1,x2) fit returned Ok"),
        Err(e) => eprintln!("[FD-DIAG] matern(x1,x2) fit returned Err (audit still ran): {e}"),
    }

    let audit = gam::estimate::take_outer_gradient_fd_capture()
        .expect("outer runner must return structured analytic-vs-FD evidence");
    assert!(
        audit.theta.len() >= 2,
        "matern(x1,x2) must enroll at least one rho and one log-kappa coordinate"
    );
    assert_eq!(audit.psi_dim, 1, "isotropic Matérn must own one psi axis");
    let analytic = audit.analytic_psi_gradient[0];
    let fd = audit.finite_difference_psi_gradient[0];
    let gap = (analytic - fd).abs();
    assert!(
        analytic.is_finite() && fd.is_finite(),
        "non-finite Matérn log-kappa gradient: analytic={analytic} fd={fd}"
    );
    let scale = analytic.abs().max(fd.abs()).max(1e-6);
    // 5e-3, not the 5e-2 this gate shipped with. The old number was not a
    // statement about the gradient: the audit differenced the criterion once at
    // `eps^0.25·(1+|ψ|)`, and at the production init `log κ ≈ 2.5` the operator
    // triplet scales like `κ^{2m}` with `m = ν + d/2 = 3.5`, so `V(ψ)` has a
    // third derivative large enough that the ORACLE's own truncation was ~1.6e-3
    // (the residual gap `iso_kappa_matern_2d_psi_fd_step_sweep_diagnostic` was
    // written to explain). The audit now Ridders-extrapolates and reports its own
    // uncertainty, so the tolerance can describe the gradient again. #2461.
    assert!(
        gap / scale < 5e-3,
        "Matérn iso-kappa outer-gradient analytic!=FD: analytic={analytic:.6e} \
         fd={fd:.6e} gap={gap:.3e} rel={:.3e} step={:.3e} oracle_unc={:.3e} order={}",
        gap / scale,
        audit.psi_steps[0],
        audit.psi_fd_uncertainty[0],
        audit.psi_fd_orders[0],
    );
    // The oracle must also have RESOLVED the component: an unresolved finite
    // difference agrees with everything, so a gate that only checks the gap can
    // pass on a measurement that measured nothing.
    assert!(
        audit.psi_fd_uncertainty[0] <= 5e-3 * fd.abs().max(1e-6),
        "outer-gradient FD oracle did not resolve the Matérn ψ component: \
         fd={fd:.6e} uncertainty={:.3e} at step {:.3e} (order {})",
        audit.psi_fd_uncertainty[0],
        audit.psi_steps[0],
        audit.psi_fd_orders[0],
    );
}

/// #1259: at the symmetric anisotropic Matérn init, the FULL outer REML
/// criterion must have a nonzero per-axis eta contrast in the direction that
/// increases the signal-axis eta. The audit is stronger than checking the final
/// fitted eta split: it verifies the value path itself sees trial eta
/// perturbations, so the optimizer has a real descent direction at theta0.
#[test]
fn aniso_matern_theta0_eta_contrast_gradient_is_fd_visible() {
    init();
    gam::estimate::enable_outer_gradient_fd_capture(2);

    let n = 180;
    let (x, y) = aniso_signal_dataset(n);
    let spec = TermCollectionSpec {
        linear_terms: vec![],
        random_effect_terms: vec![],
        smooth_terms: vec![SmoothTermSpec {
            frozen_parametric_residualization: None,
            name: "matern_2d_aniso".to_string(),
            basis: SmoothBasisSpec::Matern {
                feature_cols: vec![0, 1],
                spec: MaternBasisSpec {
                    center_strategy: CenterStrategy::FarthestPoint { num_centers: 12 },
                    periodic: None,
                    length_scale: gam::terms::basis::MaternLengthScale::fixed(1.0),
                    nu: MaternNu::FiveHalves,
                    include_intercept: false,
                    double_penalty: true,
                    identifiability: MaternIdentifiability::CenterSumToZero,
                    aniso_log_scales: Some(vec![0.0, 0.0]),
                },
                input_scale: None,
            },
            shape: ShapeConstraint::None,
            joint_null_rotation: None,
        }],
    };

    let outcome = fit_model(FitRequest::Standard(StandardFitRequest {
        data: gam::solver::fit_orchestration::StandardFitData::shared(x),
        y: std::sync::Arc::new(y),
        weights: std::sync::Arc::new(Array1::ones(n)),
        offset: std::sync::Arc::new(Array1::zeros(n)),
        spec,
        family: LikelihoodSpec::new(
            ResponseFamily::Gaussian,
            InverseLink::Standard(StandardLink::Identity),
        ),
        options: FitOptions {
            resource_policy: gam_runtime::resource::ResourcePolicy::default_library(),
            latent_cloglog: None,
            mixture_link: None,
            optimize_mixture: false,
            sas_link: None,
            optimize_sas: false,
            compute_inference: false,
            skip_rho_posterior_inference: false,
            // The baseline rho profile must certify before the joint
            // anisotropy problem exists. The separate kappa option below caps
            // only that joint loop after its seed audit.
            max_iter: 200,
            tol: 1e-6,
            nullspace_dims: vec![],
            linear_constraints: None,
            firth_bias_reduction: false,
            adaptive_regularization: None,
            rho_prior: Default::default(),
            kronecker_penalty_system: None,
            kronecker_factored: None,
            persistent_warm_start_store: None,
        },
        kappa_options: SpatialLengthScaleOptimizationOptions {
            enabled: true,
            max_outer_iter: 2,
            rel_tol: 1e-5,
            log_step: std::f64::consts::LN_2,
            min_length_scale: 1e-2,
            max_length_scale: 1e2,
            pilot_subsample_threshold: 0,
        },
        wiggle: None,
        coefficient_groups: Vec::new(),
        penalty_block_gamma_priors: Vec::new(),
        latent_coord: None,
        estimate_tweedie_p: false,
    }));
    match outcome {
        Ok(FitResult::Standard(_)) => eprintln!("[ANISO-ETA-GRAD] fit returned Ok"),
        Ok(_) => panic!("expected standard fit"),
        Err(e) => eprintln!("[ANISO-ETA-GRAD] fit returned Err after audit: {e}"),
    }

    let audit = gam::estimate::take_outer_gradient_fd_capture()
        .expect("outer runner must return structured analytic-vs-FD evidence");
    assert!(
        audit.theta.len() >= 3,
        "anisotropic Matérn must enroll rho plus both eta coordinates"
    );
    assert_eq!(
        audit.psi_dim, 2,
        "anisotropic Matérn must own exactly two psi axes"
    );
    let g_signal = audit.analytic_psi_gradient[0];
    let fd_signal = audit.finite_difference_psi_gradient[0];
    let g_noise = audit.analytic_psi_gradient[1];
    let fd_noise = audit.finite_difference_psi_gradient[1];
    let analytic_contrast = g_signal - g_noise;
    let fd_contrast = fd_signal - fd_noise;
    eprintln!(
        "[ANISO-ETA-GRAD] theta0 psi_grad=[{g_signal:.6e}, {g_noise:.6e}] \
         fd=[{fd_signal:.6e}, {fd_noise:.6e}] analytic_contrast={analytic_contrast:.6e} \
         fd_contrast={fd_contrast:.6e}"
    );

    for (psi_j, axis, a, fd) in [
        (0usize, "signal", g_signal, fd_signal),
        (1, "noise", g_noise, fd_noise),
    ] {
        assert!(
            a.is_finite() && fd.is_finite(),
            "non-finite anisotropic eta gradient component {axis}: analytic={a} fd={fd}"
        );
        let scale = a.abs().max(fd.abs()).max(1e-6);
        let gap = (a - fd).abs();
        // See the ψ-tolerance note on the isotropic gate above: 5e-2 was the
        // fixed-step oracle's error budget, not the gradient's. #2461.
        assert!(
            gap / scale < 5e-3,
            "anisotropic eta outer-gradient analytic!=FD on {axis}: analytic={a:.6e} \
             fd={fd:.6e} gap={gap:.3e} rel={:.3e} oracle_unc={:.3e} order={}",
            gap / scale,
            audit.psi_fd_uncertainty[psi_j],
            audit.psi_fd_orders[psi_j],
        );
        assert!(
            audit.psi_fd_uncertainty[psi_j] <= 5e-3 * fd.abs().max(1e-6),
            "outer-gradient FD oracle did not resolve the {axis} eta component: \
             fd={fd:.6e} uncertainty={:.3e} at step {:.3e} (order {})",
            audit.psi_fd_uncertainty[psi_j],
            audit.psi_steps[psi_j],
            audit.psi_fd_orders[psi_j],
        );
    }
    assert!(
        fd_contrast < -1e-3,
        "theta0 FD eta contrast must point toward increasing the signal-axis eta; \
         got fd_signal-fd_noise={fd_contrast:.6e}"
    );
    assert!(
        analytic_contrast < -1e-3,
        "theta0 analytic eta contrast must point toward increasing the signal-axis eta; \
         got g_signal-g_noise={analytic_contrast:.6e}"
    );
}

/// #1270 regression: a single `matern(x1, x2)` 2-D smooth must fit an ordinary
/// Gaussian surface to convergence over the FULL (uncapped) κ-optimizer loop,
/// exactly like the `duchon` control below.
///
/// Root cause: the single-spatial-term "n-free penalty re-key" fast path
/// declared itself supported for Matérn, so the design-revision skip path was
/// taken. But the realized Matérn design carries the collocation operator
/// triplet (mass/tension/stiffness, #1259) while the n-free re-key rebuilds the
/// projected-kernel double-penalty — a different block topology. The block-count
/// guard rejected the rebuild, cleared the staged surface, and the next
/// skip-path eval converted "no exact S(ψ) staged" into a HARD ERROR
/// (IntegrationError), aborting the fit. `duchon`/`thinplate`/`te` were
/// unaffected because their re-key reproduces the frozen topology exactly.
///
/// The fix drops `Matern` from `supports_nfree_penalty_rekey`, routing it
/// through the slow path that re-realizes the design every trial (re-deriving
/// the correct operator triplet). This test caps NOTHING on the outer loop, so
/// it reaches the skip-window evals that armed the bug; pre-fix it aborts with
/// IntegrationError, post-fix it converges.
#[test]
fn matern_2d_smooth_fits_ordinary_surface_full_outer_loop() {
    init();
    let data = build_dataset(160, 0.05, 0x1270_0001_2D5Eu64);
    let config = FitConfig {
        family: Some("gaussian".to_string()),
        // No outer_max_iter cap: run the FULL κ loop so the design-revision
        // skip path (the bug's trigger) is actually reached.
        gpu_policy: if cfg!(target_os = "macos") {
            gam::gpu::GpuPolicy::Off
        } else {
            gam::gpu::GpuPolicy::Auto
        },
        ..FitConfig::default()
    };

    // matern: must fit without the IntegrationError abort (#1270).
    let matern = gam::fit_from_formula("y ~ matern(x1, x2)", &data, &config);
    assert!(
        matern.is_ok(),
        "matern(x1,x2) 2-D smooth must fit an ordinary surface, but the fit \
         returned an error (#1270 regression): {:?}",
        matern.err()
    );

    // duchon control: the sibling spatial smooth that was always healthy.
    let duchon = gam::fit_from_formula("y ~ duchon(x1, x2)", &data, &config);
    assert!(
        duchon.is_ok(),
        "duchon(x1,x2) control must fit (it always did): {:?}",
        duchon.err()
    );
}
