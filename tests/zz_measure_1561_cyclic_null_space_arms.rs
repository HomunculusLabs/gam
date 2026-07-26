//! zz_measure DIAGNOSTIC (#1561): is the cyclic location-scale μ "loss" a gam
//! defect, or an artifact of the reference's penalty NULL SPACE?
//!
//! Background
//! ----------
//! `quality_vs_gamlss_gaussian_location_scale_cyclic::mu` is the #2 loser on the
//! whole-suite meta-gate (log(gam/ref) = +0.892, gam 0.0501 vs gamlss 0.0206).
//! The reference is `gamlss::pbc()`, whose second-order circular difference
//! stencil is `c(-1, 2*cos(2*pi/n), -1)` — the HARMONIC recurrence, not the
//! ordinary `c(-1, 2, -1)`. Its penalty therefore annihilates the FUNDAMENTAL
//! sinusoid rather than the constant: driving pbc's λ to ∞ leaves a 2-df pure
//! first-harmonic fit (measured: df → 2.000, energy above mode 1 → 8e-7).
//!
//! The fixture's truth is `μ*(x) = sin(x)` and `σ*(x) = 0.15 + 0.1 cos(x)` —
//! BOTH pure fundamentals, i.e. both lie exactly inside that null space. On this
//! fixture the "mature reference" is an oracle-parametric model with no
//! approximation bias and ~2 parameters of variance, which no general-purpose
//! cyclic smoother can match.
//!
//! What this measurement does
//! --------------------------
//! Runs gam's cyclic location-scale fit over K seeds on TWO truths:
//!   ORIGINAL  μ* = sin x                          σ* = 0.15 + 0.1 cos x
//!   ENRICHED  μ* = sin x + 0.4 sin(2x + 0.7)      σ* = 0.15 + 0.07 cos x
//!                                                      + 0.04 cos(2x - 0.4)
//! The enriched truth is just as periodic and just as smooth; it differs only in
//! carrying content ABOVE the fundamental, so it is not inside pbc's null space.
//! If the null-space reading is right, gam's own accuracy should be essentially
//! UNCHANGED between the two arms (it is a general smoother; a second harmonic
//! is not harder than a first), while the reference's advantage should vanish.
//!
//! Every generated (x, y) row is echoed on a `[zz1561:data]` line so the R side
//! can be fed BIT-IDENTICAL data — common random numbers, paired per seed, which
//! is what this issue's own standard requires of a rule comparison.
//!
//! zz_measure discipline: numbers are eprintln'd, never gated. The only hard
//! asserts are finiteness and that the fit produced the expected block
//! structure, so this can never become a flaky bar.

use gam::estimate::BlockRole;
use gam::families::sigma_link::logb_sigma_from_eta_scalar;
use gam::gamlss::GaussianLocationScaleFitResult;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::rmse;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;
use std::f64::consts::PI;

/// Number of paired seeds. Each seed produces one gam fit per arm; the R side
/// replays the same seeds from the echoed data.
const N_SEEDS: u64 = 25;
/// Training points, matching the quality fixture exactly.
const N_TRAIN: usize = 150;
/// Evaluation grid points in [0, 2π), matching the quality fixture exactly.
const N_GRID: usize = 50;

/// Box–Muller standard normals from a 64-bit LCG. IDENTICAL to the stream in
/// tests/quality/families/quality_vs_gamlss_gaussian_location_scale_cyclic.rs,
/// so seed 123 reproduces that fixture byte for byte.
fn standard_normals(n: usize, seed: u64) -> Vec<f64> {
    let mut state = seed;
    let mut next_unit = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let bits = state >> 11;
        (bits as f64 + 0.5) / (1u64 << 53) as f64
    };
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        let u1 = next_unit();
        let u2 = next_unit();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * PI * u2;
        out.push(r * theta.cos());
        if out.len() < n {
            out.push(r * theta.sin());
        }
    }
    out.truncate(n);
    out
}

/// The two truths. `Original` is the shipped fixture; `Enriched` adds content
/// above the fundamental to both the mean and the scale.
#[derive(Clone, Copy)]
enum Arm {
    Original,
    Enriched,
}

impl Arm {
    fn label(self) -> &'static str {
        match self {
            Arm::Original => "original",
            Arm::Enriched => "enriched",
        }
    }

    fn true_mu(self, x: f64) -> f64 {
        match self {
            Arm::Original => x.sin(),
            Arm::Enriched => x.sin() + 0.4 * (2.0 * x + 0.7).sin(),
        }
    }

    fn true_sigma(self, x: f64) -> f64 {
        match self {
            Arm::Original => 0.15 + 0.1 * x.cos(),
            Arm::Enriched => 0.15 + 0.07 * x.cos() + 0.04 * (2.0 * x - 0.4).cos(),
        }
    }
}

/// One gam location-scale fit on the cyclic fixture; returns the truth-recovery
/// errors on the evaluation grid plus the selected complexity.
struct FitReadout {
    mu_rmse: f64,
    log_sigma_rmse: f64,
    edf_mu: f64,
    edf_sigma: f64,
    lambda_mu: Vec<f64>,
}

fn fit_one(arm: Arm, seed: u64, echo_data: bool) -> FitReadout {
    let period = 2.0 * PI;
    let xs: Vec<f64> = (0..N_TRAIN)
        .map(|i| period * (i as f64) / ((N_TRAIN - 1) as f64))
        .collect();
    let z = standard_normals(N_TRAIN, seed);
    let ys: Vec<f64> = xs
        .iter()
        .zip(z.iter())
        .map(|(&x, &zi)| arm.true_mu(x) + arm.true_sigma(x) * zi)
        .collect();

    if echo_data {
        for (i, (&x, &y)) in xs.iter().zip(ys.iter()).enumerate() {
            eprintln!(
                "[zz1561:data] arm={} seed={} i={} x={:.17e} y={:.17e}",
                arm.label(),
                seed,
                i,
                x,
                y
            );
        }
    }

    let headers = vec!["y".to_string(), "x".to_string()];
    let rows: Vec<csv::StringRecord> = xs
        .iter()
        .zip(ys.iter())
        .map(|(&x, &y)| csv::StringRecord::from(vec![format!("{y:.17e}"), format!("{x:.17e}")]))
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode cyclic dataset");
    let col = ds.column_map();
    let x_idx = col["x"];
    let ncols = ds.headers.len();

    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        noise_formula: Some(
            "1 + s(x, bs='cc', period_start=0, period_end=6.283185307179586)".to_string(),
        ),
        ..FitConfig::default()
    };
    let result = fit_from_formula(
        "y ~ s(x, bs='cc', period_start=0, period_end=6.283185307179586)",
        &ds,
        &cfg,
    )
    .expect("gam cyclic location-scale fit");
    let FitResult::GaussianLocationScale(GaussianLocationScaleFitResult { fit, .. }) = result
    else {
        panic!("expected a GaussianLocationScale fit");
    };

    let loc_block = fit
        .fit
        .block_by_role(BlockRole::Location)
        .expect("location block");
    let scale_block = fit
        .fit
        .block_by_role(BlockRole::Scale)
        .expect("scale block");
    let beta_mu = loc_block.beta.clone();
    let beta_sigma = scale_block.beta.clone();
    let edf_mu = loc_block.edf;
    let edf_sigma = scale_block.edf;
    let lambda_mu: Vec<f64> = loc_block.lambdas.to_vec();

    // Evaluate the fitted smooths on the same 50-point grid the quality test uses.
    let grid_x: Vec<f64> = (0..N_GRID)
        .map(|i| period * (i as f64) / (N_GRID as f64))
        .collect();
    let mut eval_grid = Array2::<f64>::zeros((N_GRID, ncols));
    for (i, &gx) in grid_x.iter().enumerate() {
        eval_grid[[i, x_idx]] = gx;
    }
    let mean_design = build_term_collection_design(eval_grid.view(), &fit.meanspec_resolved)
        .expect("rebuild mean design at eval grid");
    let noise_design = build_term_collection_design(eval_grid.view(), &fit.noisespec_resolved)
        .expect("rebuild noise design at eval grid");

    let gam_mu: Vec<f64> = mean_design.design.apply(&beta_mu).to_vec();
    let gam_log_sigma: Vec<f64> = noise_design
        .design
        .apply(&beta_sigma)
        .iter()
        .map(|&e| logb_sigma_from_eta_scalar(e).ln())
        .collect();

    let truth_mu: Vec<f64> = grid_x.iter().map(|&gx| arm.true_mu(gx)).collect();
    let truth_log_sigma: Vec<f64> = grid_x.iter().map(|&gx| arm.true_sigma(gx).ln()).collect();

    FitReadout {
        mu_rmse: rmse(&gam_mu, &truth_mu),
        log_sigma_rmse: rmse(&gam_log_sigma, &truth_log_sigma),
        edf_mu,
        edf_sigma,
        lambda_mu,
    }
}

#[test]
fn zz_measure_cyclic_null_space_arms() {
    init_parallelism();

    eprintln!(
        "[zz1561:hdr] #1561 cyclic location-scale: gam over {N_SEEDS} paired seeds \
         x 2 truths (n={N_TRAIN}, grid={N_GRID})"
    );
    eprintln!(
        "[zz1561:hdr] ORIGINAL mu*=sin x, sigma*=0.15+0.1 cos x  (both pure fundamentals \
         => inside gamlss pbc's penalty null space)"
    );
    eprintln!(
        "[zz1561:hdr] ENRICHED mu*=sin x+0.4 sin(2x+0.7), sigma*=0.15+0.07 cos x\
         +0.04 cos(2x-0.4)  (content above the fundamental)"
    );

    for arm in [Arm::Original, Arm::Enriched] {
        for seed in 1..=N_SEEDS {
            let r = fit_one(arm, seed, true);
            assert!(
                r.mu_rmse.is_finite() && r.log_sigma_rmse.is_finite(),
                "non-finite truth-recovery error on arm {} seed {seed}",
                arm.label()
            );
            eprintln!(
                "[zz1561:gam] arm={} seed={} mu_rmse={:.8e} log_sigma_rmse={:.8e} \
                 edf_mu={:.4} edf_sigma={:.4} lambda_mu={:?}",
                arm.label(),
                seed,
                r.mu_rmse,
                r.log_sigma_rmse,
                r.edf_mu,
                r.edf_sigma,
                r.lambda_mu
            );
        }
    }

    // Seed 123 is the shipped fixture's seed: report it explicitly so the log
    // carries the number the quality test reports, next to the seed sweep.
    for arm in [Arm::Original, Arm::Enriched] {
        let r = fit_one(arm, 123, true);
        eprintln!(
            "[zz1561:gam] arm={} seed=123 mu_rmse={:.8e} log_sigma_rmse={:.8e} \
             edf_mu={:.4} edf_sigma={:.4} lambda_mu={:?}",
            arm.label(),
            r.mu_rmse,
            r.log_sigma_rmse,
            r.edf_mu,
            r.edf_sigma,
            r.lambda_mu
        );
    }
    eprintln!("[zz1561:done]");
}
