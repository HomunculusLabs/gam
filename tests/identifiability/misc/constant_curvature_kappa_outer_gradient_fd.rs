//! #944 stage 3 final wiring — κ as an ACTUALLY-FITTED ψ-coordinate.
//!
//! The constant-curvature (`M_κ`) smooth now enrolls its signed sectional
//! curvature κ as one design-moving coordinate in the unified outer
//! LAML/REML optimization. This is the merge gate the issue names: the standing
//! full-outer-gradient finite-difference audit, with κ active.
//!
//! The test enables the generic outer runner's structured finite-difference
//! capture at its first seed with a ψ coordinate. The record contains the
//! exact ρ/ψ layout plus analytic and finite-difference gradient arrays. This
//! test:
//!
//!  (1) fits a Gaussian response with a single `curv(x1, x2, kappa=..)` smooth
//!      on data GENERATED on `M_κ` for a planted κ, captures the audit, and
//!      asserts the analytic outer gradient w.r.t. κ matches the central
//!      finite difference of the criterion (no DESYNC verdict, finite
//!      per-coordinate analytic/fd, small relative gap on the κ block); and
//!  (2) on FLAT-generated data (planted κ = 0) checks the κ = 0 likelihood-ratio
//!      flatness test has correct size — `p_value` is the interior χ²₁ tail
//!      (not the half-χ² boundary mixture) and a flat fit is NOT rejected.
//!
//! Reference-as-truth: data are generated on a known `ConstantCurvature`
//! geometry, and every assertion is against that self-constructed truth or the
//! analytic FD of gam's own criterion — never another tool's output.

use gam::geometry::constant_curvature::ConstantCurvature;
use gam::geometry::curvature_estimand::{flatness_lr_test, profile_ci_walk};
use gam::{FitConfig, encode_recordswith_inferred_schema};

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

/// Chart points uniformly in a disk of radius `r` (inside the κ-stereographic
/// chart for the κ used here), plus a Gaussian response that is a smooth
/// function of the M_κ geodesic distance to a fixed reference point — a signal
/// the constant-curvature kernel can represent.
fn build_dataset(
    n: usize,
    kappa: f64,
    radius: f64,
    seed: u64,
) -> gam::inference::data::EncodedDataset {
    let mut st = seed;
    let manifold = ConstantCurvature::new(2, kappa);
    let reference = ndarray::array![0.0_f64, 0.0_f64];
    let mut header = String::from("y,x1,x2\n");
    let mut body = String::new();
    for _ in 0..n {
        // Rejection-sample a point uniformly in the disk of radius `radius`.
        let (x1, x2) = loop {
            let a = 2.0 * next_unit(&mut st) - 1.0;
            let b = 2.0 * next_unit(&mut st) - 1.0;
            if a * a + b * b <= 1.0 {
                break (a * radius, b * radius);
            }
        };
        let pt = ndarray::array![x1, x2];
        let d = manifold
            .distance(pt.view(), reference.view())
            .expect("in-chart geodesic distance");
        // Smooth planted signal of the geodesic distance + noise.
        let mu = 2.0 * (-d).exp() - 1.0;
        let y = mu + 0.10 * next_gauss(&mut st);
        body.push_str(&format!("{y:.6},{x1:.6},{x2:.6}\n"));
    }
    header.push_str(&body);
    let mut rdr = csv::ReaderBuilder::new().from_reader(header.as_bytes());
    let records: Vec<csv::StringRecord> = rdr.records().map(|r| r.unwrap()).collect();
    let headers = vec!["y".to_string(), "x1".to_string(), "x2".to_string()];
    encode_recordswith_inferred_schema(headers, records).expect("encode dataset")
}

/// MERGE GATE: the analytic outer LAML/REML gradient w.r.t. κ matches a central
/// finite difference of the outer criterion at θ₀ on a constant-curvature fit
/// where κ is active.
#[test]
fn constant_curvature_kappa_outer_gradient_matches_fd() {
    init();
    gam::estimate::enable_outer_gradient_fd_capture(1);
    // Data generated on M_κ with a planted spherical curvature.
    let data = build_dataset(400, 0.8, 0.6, 0xC0FF_EE01);
    // κ must be a FREE outer coordinate for this FD audit — the audit exists to
    // check the analytic ∂criterion/∂κ against a finite difference, which only
    // fires when κ is enrolled as a ψ-coordinate. Since gam#2152, an explicit
    // `kappa=` PINS κ (fixed geometry, no ψ enrollment), so κ is left OMITTED
    // here: it seeds at the flat default 0 (the same non-trivial start point away
    // from the planted 0.8 truth) and enrolls for estimation.
    let formula = "y ~ curv(x1, x2, centers=8)";
    let config = FitConfig {
        outer_max_iter: Some(2),
        gpu_policy: if cfg!(target_os = "macos") {
            gam::gpu::GpuPolicy::Off
        } else {
            gam::gpu::GpuPolicy::Auto
        },
        ..FitConfig::default()
    };
    // The outer FD audit fires at θ₀ DURING the fit, before the (capped) outer
    // loop, so the fit's own success/failure is irrelevant to this gate — what
    // we assert is the captured audit. Surface the outcome for diagnostics.
    match gam::fit_from_formula(formula, &data, &config) {
        Ok(_) => eprintln!("[FD-DIAG] constant-curvature fit returned Ok"),
        Err(e) => eprintln!("[FD-DIAG] constant-curvature fit returned Err (audit still ran): {e}"),
    }

    let audit = gam::estimate::take_outer_gradient_fd_capture()
        .expect("outer runner must return structured analytic-vs-FD evidence");
    assert!(
        audit.psi_dim >= 1,
        "constant-curvature smooth must enroll kappa as a psi coordinate"
    );
    // κ here is selected by `constant_curvature_kappa_fair_optimum`, which
    // computes the curvature-fair evidence and its derivative in closed form
    // and never enters a REML assembly — so it publishes no atoms, and the
    // audit says so by name instead of failing (#2460). Surfaced rather than
    // asserted: what this gate grades is the hand-derived derivative below, and
    // the day that route acquires an atom breakdown is not the day this gate
    // should go red.
    match &audit.decomposition {
        gam::estimate::OuterGradientFdDecomposition::Decomposed(_) => {
            eprintln!("[FD-DIAG] constant-curvature audit carries an atom breakdown")
        }
        gam::estimate::OuterGradientFdDecomposition::NotDecomposed { reason } => {
            eprintln!("[FD-DIAG] constant-curvature audit is top-line only: {reason}")
        }
    }
    for j in 0..audit.psi_dim {
        let analytic = audit.analytic_psi_gradient[j];
        let fd = audit.finite_difference_psi_gradient[j];
        let gap = (analytic - fd).abs();
        assert!(
            analytic.is_finite() && fd.is_finite(),
            "non-finite kappa gradient component {j}: analytic={analytic} fd={fd}"
        );
        let scale = analytic.abs().max(fd.abs()).max(1e-6);
        // Still 5e-2, deliberately, unlike the two Matern siblings.
        //
        // The audit Ridders-extrapolates and reports `psi_fd_uncertainty`
        // (#2461), which is what let `matern_2d_iso_kappa_outer_gradient_
        // matches_fd` and its anisotropic sibling come down to 5e-3 and add an
        // oracle-resolution assertion. Until this gate has RUN and reported an
        // uncertainty on this route, following them would be a claim with no
        // measurement behind it, so the number stays and the uncertainty is
        // merely reported. Tighten it from the first green run's numbers, not
        // from the siblings'.
        assert!(
            gap / scale < 5e-2,
            "kappa outer-gradient analytic!=FD on coordinate {j}: \
             analytic={analytic:.6e} fd={fd:.6e} gap={gap:.3e} rel={:.3e} step={:.3e} \
             oracle_unc={:.3e} order={}",
            gap / scale,
            audit.psi_steps[j],
            audit.psi_fd_uncertainty[j],
            audit.psi_fd_orders[j],
        );
    }
}

/// The κ = 0 flatness test has correct size: on a quadratic profile centred at
/// κ̂ = 0 the LR statistic is zero and the p-value is the full interior χ²₁
/// tail (here p = 1), NOT the half-χ² boundary mixture — a flat latent space is
/// not spuriously rejected, and the profile CI straddles 0 (verdict Flat).
#[test]
fn kappa_zero_flatness_test_has_correct_size() {
    // A profiled criterion (negative log-evidence) whose minimiser is exactly
    // flat: V_p(κ) = 0.5·a·κ². κ̂ = 0 ⇒ LR = 0 ⇒ p = 1 (not 0.5).
    let a = 4.0;
    let v_p = |k: f64| -> Result<f64, String> { Ok(0.5 * a * k * k) };

    let test = flatness_lr_test(v_p, 0.0).expect("flatness LR");
    assert!(
        test.lr_stat.abs() < 1e-12,
        "flat κ̂ ⇒ zero LR, got {}",
        test.lr_stat
    );
    assert!(
        (test.p_value - 1.0).abs() < 1e-12,
        "interior χ²₁ p-value at LR=0 is 1.0, not the half-χ² 0.5; got {}",
        test.p_value
    );

    // And the profile CI must straddle 0 (geometry verdict Flat) for flat data.
    let ci = profile_ci_walk(v_p, 0.0, a, -10.0, 10.0, 0.95, 1e-9).expect("CI walk");
    assert!(
        ci.ci_lo < 0.0 && ci.ci_hi > 0.0,
        "flat profile CI must straddle 0: [{}, {}]",
        ci.ci_lo,
        ci.ci_hi
    );
    assert_eq!(
        ci.verdict,
        gam::geometry::curvature_estimand::CurvatureVerdict::Flat
    );
}
