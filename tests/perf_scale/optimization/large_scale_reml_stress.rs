//! End-to-end stress test for the closed-form Duchon pipeline at
//! large-scale-relevant scale.
//!
//! Runs in the default suite. The fit can take many minutes and use a
//! lot of memory — run under `--release` if iteration time matters:
//!
//! ```text
//! cargo test --release large_scale_reml_stress
//! ```
//!
//! It exercises the full Duchon-on-PC GAM pipeline end-to-end:
//!
//!   * Pure-Rust deterministic large-scale-style simulator producing
//!     `n` rows of `pc_dim` PC features sampled from N(0, I) and a
//!     continuous response `y = f_true(X) + ε`.
//!   * Hybrid anisotropic Duchon smooth (`length_scale = Some(...)`,
//!     `aniso_log_scales = Some(zeros)`) with `K` farthest-point centers.
//!   * REML/LAML outer loop must converge.
//!   * Held-out-grid relative L2 reconstruction error must be < 0.10.
//!   * Bias-corrected predictions must be available on `FitInference`
//!     and finite.
//!   * 95% prediction-interval coverage on held-out samples must
//!     exceed 0.85 across `N_COVERAGE_SIMS` independent simulations.
//!   * Each fit must terminate on convergence, strictly inside the outer
//!     iteration budget it was configured with.
//!
//! All randomness is seeded; failures are reproducible.

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
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};
use std::time::Instant;

// ─── Test scale knobs ───────────────────────────────────────────────────
//
// `N_TRAIN`, `K_CENTERS`, and `PC_DIM` are deliberately moderate so the
// test is feasible at all in a default `--release` run on a developer
// box. The team-lead spec calls out `K∈{500,1000}` (start small) and
// `n` in the 50K-300K range; `n=50_000` and `K=500` are the lower end
// of that range. Crank up by editing these constants.
const N_TRAIN: usize = 50_000;
const N_HOLDOUT: usize = 4_000;
const PC_DIM: usize = 6;
const K_CENTERS: usize = 500;
const NOISE_SD: f64 = 0.30;
const SEED_BASE: u64 = 0xB10B_0001_0001_0001;
// The coverage claim is a POOLED statistic: every per-row interval from every
// sim is aggregated into one `total_in / total_pts` fraction. With
// `N_COVERAGE_SIMS × N_COVERAGE_HOLDOUT` = 8 × 400 = 3200 pooled points, the
// empirical-coverage standard error is √(0.95·0.05/3200) ≈ 0.004, so the
// `coverage > 0.85` bound (true coverage ≈ 0.95) keeps a >20·SE margin. Reduced
// from 20 to 8 sims to keep this test under the 300s nextest SLOW budget; the
// pooled coverage estimate stays tight and the assertion is unchanged.
const N_COVERAGE_SIMS: usize = 8;
const N_COVERAGE_TRAIN: usize = 4_000;
const N_COVERAGE_HOLDOUT: usize = 400;
const K_COVERAGE: usize = 80;
const PC_DIM_COVERAGE: usize = 4;

// Work ceilings. These replace the wall-clock ceilings this file used to
// assert (1800s for the main fit, 120s per coverage fit). A wall-clock
// assertion on a shared CI runner measures the runner, not the solver, so it
// flakes in both directions; and the 1800s one could never fire at all,
// because the harness SIGKILLs the target long before half an hour elapses —
// dead code wearing the shape of a budget. What the fits are actually
// supposed to demonstrate is that the REML outer loop CONVERGES rather than
// grinding to its cap, and `max_iter` (the cap the fit is configured with)
// is the machine-independent statement of exactly that. No new magic
// constants: the ceiling is the fit's own configured budget.
const MAIN_MAX_ITER: usize = 40;
const COVERAGE_MAX_ITER: usize = 30;
const NORMAL_95_TWO_SIDED_Z: f64 = 1.959_963_984_540_054;

fn gaussian_identity_likelihood() -> LikelihoodSpec {
    LikelihoodSpec::new(
        ResponseFamily::Gaussian,
        InverseLink::Standard(StandardLink::Identity),
    )
}

// ─── Synthetic large-scale simulator ────────────────────────────────────────

/// Smooth ground-truth function on PC coordinates. Used both for
/// generating `y` and for evaluating reconstruction error.
///
/// The functional form mirrors the pipeline contract in
/// `production_pipeline_spec.md` and `large_scale_sim.py`: a sum of a
/// linear PC trend, a radial bump centered near the origin, and a
/// sinusoid on PC0. It is smooth, bounded, and not separable into
/// per-axis pieces — all properties an anisotropic Duchon smooth
/// should be able to track.
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
    let radial_bump = 1.0 * (-dist2 / (2.0 * 0.8 * 0.8)).exp();
    let sinusoid = 0.4 * (std::f64::consts::PI * row[0]).sin();
    linear + radial_bump + sinusoid
}

/// Generate `(X, y, y_true)` with PC coordinates sampled iid from
/// the standard normal and `y = truth(X) + N(0, NOISE_SD²)`.
fn simulate(n: usize, pc_dim: usize, seed: u64) -> (Array2<f64>, Array1<f64>, Array1<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0, 1.0).expect("normal params must be valid");
    let noise = Normal::new(0.0, NOISE_SD).expect("noise params must be valid");

    let mut x = Array2::<f64>::zeros((n, pc_dim));
    let mut y = Array1::<f64>::zeros(n);
    let mut y_true = Array1::<f64>::zeros(n);
    for i in 0..n {
        let mut row = vec![0.0_f64; pc_dim];
        for j in 0..pc_dim {
            let v = normal.sample(&mut rng);
            x[[i, j]] = v;
            row[j] = v;
        }
        let f = truth(&row);
        y_true[i] = f;
        y[i] = f + noise.sample(&mut rng);
    }
    (x, y, y_true)
}

/// Length scale of the hybrid (Matérn-blended) Duchon kernel. Bound to a
/// constant because the SAME value has to reach both the order resolution
/// below and the spec: `resolve_duchon_orders` branches on
/// `length_scale.is_none()` (the pure-mode CPD constraint `2s < d` applies
/// only there), so resolving for one mode and building the other would resolve
/// against constraints the built kernel does not have.
const HYBRID_LENGTH_SCALE: f64 = 1.0;

/// Build the anisotropic-hybrid Duchon term spec used throughout the
/// test.
fn duchon_aniso_pc_spec(name: &str, pc_dim: usize, k_centers: usize) -> TermCollectionSpec {
    let operator_penalties = DuchonOperatorPenaltySpec::default();
    // The Duchon orders are RESOLVED from `pc_dim`, not written down (#2709).
    //
    // The pointwise kernel is the inverse Fourier of `1/|ξ|^{2(p+s)}`, finite
    // at the origin iff `2(p+s) > d` — a condition on the dimension. This
    // fixture hardcoded `power: 1.0` with a `Linear` nullspace (`p = 2`), i.e.
    // `2(p+s) = 6`, which holds at the sibling coverage test's `PC_DIM_COVERAGE
    // = 4` and fails at `PC_DIM = 6` on the equality `6 > 6`. So the main test
    // died in 0.21 s inside basis construction and never ran a single one of
    // its large-scale assertions, while the coverage test using the same
    // builder ran fine — a constant that was correct for one caller and
    // inadmissible for the other.
    //
    // `resolve_duchon_orders` is the library's own answer to that question: the
    // smallest admissible `(nullspace, s)` at this dimension, also clearing the
    // D1 collocation margin `2(p+s) > d+1` that the active tension penalty
    // needs. Deriving it here means the fixture follows `PC_DIM` instead of
    // being re-broken by the next edit to it, and it is the SAME resolution the
    // production paths use rather than a fixture-local rule that could agree
    // with nothing.
    //
    // At the two dimensions this test uses, and with the default penalties
    // (mass + tension active, stiffness disabled ⇒ max operator order 1):
    //   pc_dim = 4 → (Linear, s = 1), 2(p+s) = 6 > 5 — what the coverage test
    //                already built, so its behaviour is unchanged.
    //   pc_dim = 6 → (Linear, s = 2), 2(p+s) = 8 > 7 — the main test's repair.
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
                    // Hybrid Duchon — required for aniso_log_scales.
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

/// L2 relative error: ||pred - truth||₂ / ||truth - mean(truth)||₂.
fn relative_l2(pred: &Array1<f64>, truth: &Array1<f64>) -> f64 {
    let mean_t = truth.mean().unwrap_or(0.0);
    let mut num = 0.0;
    let mut den = 0.0;
    for (p, t) in pred.iter().zip(truth.iter()) {
        let dp = p - t;
        let dt = t - mean_t;
        num += dp * dp;
        den += dt * dt;
    }
    (num / den.max(1e-30)).sqrt()
}

fn gaussian_identity_mean(
    design: ArrayView2<'_, f64>,
    beta: ArrayView1<'_, f64>,
    offset: ArrayView1<'_, f64>,
) -> Array1<f64> {
    let mut mean = design.dot(&beta);
    mean += &offset;
    mean
}

fn gaussian_identity_bias_corrected_mean(
    design: ArrayView2<'_, f64>,
    fit: &UnifiedFitResult,
    offset: ArrayView1<'_, f64>,
) -> Array1<f64> {
    let bias_correction = fit
        .inference
        .as_ref()
        .and_then(|inference| inference.bias_correction_beta.as_ref())
        .expect("FitInference must carry bias_correction_beta");
    let beta = &fit.beta + bias_correction;
    gaussian_identity_mean(design, beta.view(), offset)
}

fn gaussian_identity_bias_corrected_mean_interval(
    design: ArrayView2<'_, f64>,
    fit: &UnifiedFitResult,
    offset: ArrayView1<'_, f64>,
) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
    let mean = gaussian_identity_bias_corrected_mean(design, fit, offset);
    let covariance = fit
        .beta_covariance_corrected()
        .expect("Gaussian identity coverage requires smoothing-corrected covariance");
    assert_eq!(covariance.nrows(), fit.beta.len());
    assert_eq!(covariance.ncols(), fit.beta.len());

    let mut eta_se = Array1::<f64>::zeros(design.nrows());
    for (i, row) in design.outer_iter().enumerate() {
        let cov_row = covariance.dot(&row);
        eta_se[i] = row.dot(&cov_row).max(0.0).sqrt();
    }
    let z_se = eta_se.mapv(|se| NORMAL_95_TWO_SIDED_Z * se);
    let lower = &mean - &z_se;
    let upper = &mean + &z_se;
    (mean, lower, upper)
}

// ─── Main stress test ───────────────────────────────────────────────────

#[test]
fn large_scale_reml_stress_main() {
    let (x_train, y_train, _y_true_train) = simulate(N_TRAIN, PC_DIM, SEED_BASE);
    let (x_holdout, _y_holdout, y_true_holdout) =
        simulate(N_HOLDOUT, PC_DIM, SEED_BASE.wrapping_add(0xDEAD));

    let spec = duchon_aniso_pc_spec("duchon_pc_main", PC_DIM, K_CENTERS);
    let weights = Array1::ones(N_TRAIN);
    let offset = Array1::<f64>::zeros(N_TRAIN);

    let start = Instant::now();
    let fitted = fit_term_collection_forspec(
        x_train.view(),
        y_train.view(),
        weights.view(),
        offset.view(),
        &spec,
        gaussian_identity_likelihood(),
        &fit_options(MAIN_MAX_ITER),
    )
    .expect("large-scale Duchon-on-PC fit should succeed");
    let elapsed = start.elapsed();

    // (1) Fit existence is the sealed convergence proof (SPEC 20).
    assert!(
        fitted.fit.beta.iter().all(|v| v.is_finite()),
        "fitted coefficients must all be finite",
    );

    // (2) Held-out-grid reconstruction error: build the held-out design
    //     using the *fitted* term collection design (so centers, scaling,
    //     etc. match), then compute relative L2 against truth.
    let frozenspec = freeze_term_collection_from_design(&spec, &fitted.design)
        .expect("freezing trained spec must succeed");
    let holdout_design = build_term_collection_design(x_holdout.view(), &frozenspec)
        .expect("holdout design build must succeed");
    let holdout_dense = holdout_design.design.to_dense();
    let holdout_offset = Array1::<f64>::zeros(N_HOLDOUT);

    let pred_mean = gaussian_identity_mean(
        holdout_dense.view(),
        fitted.fit.beta.view(),
        holdout_offset.view(),
    );
    assert!(pred_mean.iter().all(|v| v.is_finite()));
    let rel_l2 = relative_l2(&pred_mean, &y_true_holdout);
    assert!(
        rel_l2 < 0.10,
        "held-out relative L2 reconstruction error too high: {rel_l2:.4} (>= 0.10)",
    );

    // (3) Bias-corrected predictions: FitInference must carry a finite
    //     bias-correction vector after a successful REML fit, and the
    //     bias-corrected Gaussian identity prediction must stay finite.
    let inference = fitted
        .fit
        .inference
        .as_ref()
        .expect("compute_inference=true must populate FitInference");
    let bc = inference
        .bias_correction_beta
        .as_ref()
        .expect("FitInference must carry bias_correction_beta");
    assert_eq!(bc.len(), fitted.fit.beta.len());
    assert!(
        bc.iter().all(|v| v.is_finite()),
        "bias_correction_beta must be entirely finite",
    );

    let pred_unc_mean = gaussian_identity_bias_corrected_mean(
        holdout_dense.view(),
        &fitted.fit,
        holdout_offset.view(),
    );
    assert!(pred_unc_mean.iter().all(|v| v.is_finite()));

    // (4) The outer loop converged inside its configured budget rather than
    //     stopping because it ran out of iterations.
    assert!(
        fitted.fit.outer_iterations < MAIN_MAX_ITER,
        "main large-scale stress fit ran {} outer iterations, exhausting its \
         configured {MAIN_MAX_ITER}-iteration REML budget (elapsed {:.1}s): the \
         outer loop is grinding to its cap instead of converging",
        fitted.fit.outer_iterations,
        elapsed.as_secs_f64(),
    );

    eprintln!(
        "[large_scale_reml_stress_main] n={N_TRAIN}, K={K_CENTERS}, pc_dim={PC_DIM} \
         | wall_clock={:.2}s, outer_iter={}, rel_l2_holdout={:.4}",
        elapsed.as_secs_f64(),
        fitted.fit.outer_iterations,
        rel_l2,
    );
}

/// #2708: report WHICH of the three candidate causes the coverage number is
/// consistent with. Report-only; it asserts nothing.
///
/// The aggregate SD of the standardised error `z = (truth − mean)/SE` splits two
/// of them — `SD ≈ 1` with an off-centre mean says bias carries it, `SD ≈ 2`
/// centred says the variance is understated — but it CANNOT separate a uniform
/// scale error from a missing variance component, because both inflate the same
/// aggregate. The binned columns do: a missing smoothing-parameter-uncertainty
/// term is heteroscedastic by construction, so it lands on the high-leverage /
/// boundary rows and leaves the interior near nominal, while a scale error is
/// flat in leverage.
///
/// `SD(z)` binned by SE quantile is the leverage axis (SE² = xᵀΣx is a monotone
/// leverage proxy); `SD(z)` binned by ‖x‖ is the boundary axis; and the mean
/// RESIDUAL binned by `x₀` is the approximation-error axis, because the truth's
/// out-of-model term is `0.4·sin(π·x₀)` and an approximation error is a
/// systematic function of position while sampling error is not.
///
/// Also printed: `|z| > 3` frequency against the 0.27% a standard normal gives.
/// A heavy tail with a near-nominal centre is the outcome the two-way split does
/// not cover, and it would mean an interaction rather than either single cause.
fn report_coverage_diagnostics(
    z: &[f64],
    resid: &[f64],
    se: &[f64],
    radius: &[f64],
    x0: &[f64],
) {
    if z.is_empty() {
        eprintln!("[cov-diag] no finite points");
        return;
    }
    let n = z.len() as f64;
    let mean_z = z.iter().sum::<f64>() / n;
    let sd_z = (z.iter().map(|v| (v - mean_z).powi(2)).sum::<f64>() / (n - 1.0).max(1.0)).sqrt();
    let tail = z.iter().filter(|v| v.abs() > 3.0).count() as f64 / n;
    let mean_abs_resid = resid.iter().map(|v| v.abs()).sum::<f64>() / n;
    let mean_se = se.iter().sum::<f64>() / n;
    eprintln!(
        "[cov-diag] n={} mean_z={mean_z:+.4} sd_z={sd_z:.4} |z|>3={:.4} (normal 0.0027) \
         mean|resid|={mean_abs_resid:.5} mean_se={mean_se:.5} ratio={:.3}",
        z.len(),
        tail,
        mean_abs_resid / mean_se.max(f64::MIN_POSITIVE),
    );

    // Binned SD(z): flat across bins => a scale error; ramping => a missing,
    // leverage-dependent variance component.
    let binned_sd = |key: &[f64], label: &str| {
        let mut idx: Vec<usize> = (0..key.len()).collect();
        idx.sort_by(|&a, &b| key[a].partial_cmp(&key[b]).unwrap_or(std::cmp::Ordering::Equal));
        const BINS: usize = 5;
        let per = idx.len() / BINS;
        if per == 0 {
            return;
        }
        let mut out = String::new();
        for b in 0..BINS {
            let lo = b * per;
            let hi = if b + 1 == BINS { idx.len() } else { (b + 1) * per };
            let slice = &idx[lo..hi];
            let m = slice.len() as f64;
            let mu = slice.iter().map(|&i| z[i]).sum::<f64>() / m;
            let sd = (slice
                .iter()
                .map(|&i| (z[i] - mu).powi(2))
                .sum::<f64>()
                / (m - 1.0).max(1.0))
            .sqrt();
            out.push_str(&format!(" [{:.3}..{:.3}] sd={sd:.3} mean={mu:+.3};", key[slice[0]], key[slice[slice.len() - 1]]));
        }
        eprintln!("[cov-diag] SD(z) by {label}:{out}");
    };
    binned_sd(se, "SE quantile (leverage proxy)");
    binned_sd(radius, "‖x‖ (distance from design centre)");

    // Mean RESIDUAL by x0: the truth's out-of-model term is `0.4·sin(π·x₀)`, so a
    // systematic sign pattern here is approximation error, not sampling error.
    {
        let mut idx: Vec<usize> = (0..x0.len()).collect();
        idx.sort_by(|&a, &b| x0[a].partial_cmp(&x0[b]).unwrap_or(std::cmp::Ordering::Equal));
        const BINS: usize = 8;
        let per = idx.len() / BINS;
        if per > 0 {
            let mut out = String::new();
            for b in 0..BINS {
                let lo = b * per;
                let hi = if b + 1 == BINS { idx.len() } else { (b + 1) * per };
                let slice = &idx[lo..hi];
                let mu = slice.iter().map(|&i| resid[i]).sum::<f64>() / slice.len() as f64;
                out.push_str(&format!(" [{:+.2}]={mu:+.4};", x0[slice[slice.len() / 2]]));
            }
            eprintln!("[cov-diag] mean(truth-mean) by x0:{out}");
        }
    }
}

// ─── Coverage simulation ────────────────────────────────────────────────

/// Repeatedly fit the same anisotropic Duchon model on freshly drawn
/// data, then check the empirical 95% coverage of the per-row mean
/// interval on held-out points. A correctly calibrated posterior
/// should produce at least 0.85 average coverage across simulations
/// (the slack accounts for finite-sample noise and the
/// well-known REML-conservativeness/anti-conservativeness drift at
/// this dimensionality).
#[test]
fn large_scale_reml_stress_coverage() {
    let mut total_in = 0usize;
    let mut total_pts = 0usize;
    // #2708 diagnostic accumulators. Report-only: nothing below changes the
    // assertion, the fixture, or the fit. The coverage number alone cannot say
    // WHICH of three things is wrong, and each has a different fix:
    //
    //   * the posterior variance is understated by a roughly uniform factor
    //     (a scale error or a missing term in `beta_covariance_corrected()`);
    //   * a variance COMPONENT is missing (smoothing-parameter uncertainty is
    //     the classic one), which is heteroscedastic by construction and so
    //     shows up at the boundary / high-leverage rows and not in the interior;
    //   * the truth is out of the fitted model's span, so a deterministic
    //     APPROXIMATION error sits at every point and the intervals may be
    //     perfectly calibrated for a mean function this test never asks about.
    //
    // The third is live here and is not hypothetical: `truth()` is a fixed
    // analytic function (linear + a Gaussian bump + `0.4·sin(π·x₀)`) while the
    // model is a hybrid-Duchon RBF over a finite center set, so the sinusoid in
    // particular is not in the span.
    let mut z_all: Vec<f64> = Vec::new();
    let mut resid_all: Vec<f64> = Vec::new();
    let mut se_all: Vec<f64> = Vec::new();
    let mut radius_all: Vec<f64> = Vec::new();
    let mut x0_all: Vec<f64> = Vec::new();

    for sim_idx in 0..N_COVERAGE_SIMS {
        let train_seed = SEED_BASE.wrapping_add(0xC0DE_0000 + sim_idx as u64);
        let test_seed = SEED_BASE.wrapping_add(0xFADE_0000 + sim_idx as u64);

        let (x_tr, y_tr, _) = simulate(N_COVERAGE_TRAIN, PC_DIM_COVERAGE, train_seed);
        let (x_te, _y_te, y_true_te) = simulate(N_COVERAGE_HOLDOUT, PC_DIM_COVERAGE, test_seed);

        let spec = duchon_aniso_pc_spec(
            &format!("duchon_pc_cov_{sim_idx}"),
            PC_DIM_COVERAGE,
            K_COVERAGE,
        );
        let weights = Array1::ones(N_COVERAGE_TRAIN);
        let offset_tr = Array1::<f64>::zeros(N_COVERAGE_TRAIN);

        let start = Instant::now();
        let fitted = fit_term_collection_forspec(
            x_tr.view(),
            y_tr.view(),
            weights.view(),
            offset_tr.view(),
            &spec,
            gaussian_identity_likelihood(),
            &fit_options(COVERAGE_MAX_ITER),
        )
        .expect("coverage-sim Duchon-on-PC fit should succeed");
        let elapsed = start.elapsed();
        assert!(
            fitted.fit.outer_iterations < COVERAGE_MAX_ITER,
            "coverage-sim fit {sim_idx} ran {} outer iterations, exhausting its \
             configured {COVERAGE_MAX_ITER}-iteration REML budget (elapsed \
             {:.1}s): the outer loop is grinding to its cap instead of converging",
            fitted.fit.outer_iterations,
            elapsed.as_secs_f64(),
        );
        // Fit existence is the sealed convergence proof (SPEC 20).

        let frozenspec = freeze_term_collection_from_design(&spec, &fitted.design)
            .expect("coverage-sim freeze spec must succeed");
        let holdout_design = build_term_collection_design(x_te.view(), &frozenspec)
            .expect("coverage-sim holdout design build must succeed");
        let holdout_dense = holdout_design.design.to_dense();
        let offset_te = Array1::<f64>::zeros(N_COVERAGE_HOLDOUT);

        let (pred_mean, pred_lower, pred_upper) = gaussian_identity_bias_corrected_mean_interval(
            holdout_dense.view(),
            &fitted.fit,
            offset_te.view(),
        );
        assert!(pred_mean.iter().all(|v| v.is_finite()));

        for i in 0..N_COVERAGE_HOLDOUT {
            let lo = pred_lower[i];
            let hi = pred_upper[i];
            let truth_i = y_true_te[i];
            if truth_i >= lo && truth_i <= hi {
                total_in += 1;
            }
            total_pts += 1;

            // The interval is `mean ± z·SE`, so SE is recoverable from its own
            // half-width without re-deriving it here — this stays a pure reader
            // of what the assertion already consumed.
            let se = (hi - lo) / (2.0 * NORMAL_95_TWO_SIDED_Z);
            let resid = truth_i - pred_mean[i];
            if se > 0.0 && resid.is_finite() {
                z_all.push(resid / se);
                resid_all.push(resid);
                se_all.push(se);
                let row = x_te.row(i);
                radius_all.push(row.dot(&row).sqrt());
                x0_all.push(row[0]);
            }
        }
    }

    report_coverage_diagnostics(&z_all, &resid_all, &se_all, &radius_all, &x0_all);

    let coverage = total_in as f64 / total_pts.max(1) as f64;
    assert!(
        coverage > 0.85,
        "empirical 95% coverage too low: {coverage:.4} (expected > 0.85, \
         {total_in}/{total_pts})",
    );
    eprintln!(
        "[large_scale_reml_stress_coverage] sims={N_COVERAGE_SIMS}, points={total_pts}, \
         coverage={coverage:.4}",
    );
}
