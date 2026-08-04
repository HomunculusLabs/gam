//! #2676 repro harness: the `geo_disease_*_matern` curvature refusals.
//!
//! Mirrors the bench scenario `geo_disease_matern` — `n = 4000`, seed
//! `20260226`, the 16 noisy PCs of `geo_disease_columns` entering as ONE joint
//! Matern smooth with `k` centers, binomial-logit — and asks for inference so
//! the smoothing-correction branch (`invert_identified_rho_hessian`) runs too.
//!
//! NOT a test — examples skip dev-deps, so the generator is inlined from
//! `gam_test_support::synthetic::geo_disease_columns` verbatim.
//!
//! Run: `cargo run --release --example repro2676_geo_disease_matern -- [k] [n] [n_pcs] [log_level]`
//! The `[INDEF-HESS]` dumps are at `warn` (the default); pass `info` as the
//! fourth argument to see which objective certifies and what it deflated.

use gam::basis::{CenterStrategy, MaternBasisSpec, MaternIdentifiability, MaternNu};
use gam::estimate::FitOptions;
use gam::smooth::{
    ShapeConstraint, SmoothBasisSpec, SmoothTermSpec, SpatialLengthScaleOptimizationOptions,
    TermCollectionSpec,
};
use gam::types::{InverseLink, LikelihoodSpec, ResponseFamily, StandardLink};
use gam::{FitRequest, FitResult, StandardFitRequest};
use ndarray::{Array1, Array2};
use std::time::Instant;

/// SplitMix64 → uniform/normal, byte-identical in behaviour to
/// `gam_test_support::synthetic::SplitMixNormalRng` so the fixture matches the
/// bench's `synthetic_geo_disease_columns(4000, 20260226)`.
struct Rng {
    state: u64,
    spare: Option<f64>,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self {
            state: seed ^ 0x9E37_79B9_7F4A_7C15,
            spare: None,
        }
    }
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn uniform_range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.uniform()
    }
    fn standard_normal(&mut self) -> f64 {
        if let Some(value) = self.spare.take() {
            return value;
        }
        loop {
            let u = 2.0 * self.uniform() - 1.0;
            let v = 2.0 * self.uniform() - 1.0;
            let s = u * u + v * v;
            if s > 0.0 && s < 1.0 {
                let factor = (-2.0 * s.ln() / s).sqrt();
                self.spare = Some(v * factor);
                return u * factor;
            }
        }
    }
    fn normal(&mut self, mean: f64, sd: f64) -> f64 {
        mean + sd * self.standard_normal()
    }
    fn bernoulli(&mut self, p: f64) -> bool {
        self.uniform() < p
    }
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn geo_disease_columns(n: usize, seed: u64) -> (Array2<f64>, Array1<f64>) {
    let n = n.max(500);
    let mut rng = Rng::new(seed);
    let mut x = Array2::<f64>::zeros((n, 16));
    let mut y = Array1::<f64>::zeros(n);
    for i in 0..n {
        let lat = rng.uniform_range(-1.0, 1.0);
        let lon = rng.uniform_range(-1.0, 1.0);
        let equator = 1.0 - lat.abs();
        let geo_signal = -1.0
            + 2.20 * equator
            + 0.55 * (std::f64::consts::PI * lon).sin()
            + 0.35 * (2.25 * std::f64::consts::PI * lon).cos()
            + 0.30 * (2.0 * std::f64::consts::PI * equator * lon).sin();
        let southness = (-lat).clamp(0.0, 1.0);
        let eta = geo_signal + rng.normal(0.0, 0.20 + 0.85 * southness.powf(1.35));
        y[i] = if rng.bernoulli(sigmoid(eta)) { 1.0 } else { 0.0 };
        for j in 0..16 {
            let jf = j as f64;
            let a = 0.95 - 0.045 * jf;
            let b = 0.25 + 0.035 * jf;
            let c = if j % 2 == 0 { 1.0 } else { -1.0 } * (0.10 + 0.01 * jf);
            let noise_sd = 0.15 + 0.015 * jf;
            x[[i, j]] = a * lat + b * lon + c * lat * lon + rng.normal(0.0, noise_sd);
        }
    }
    (x, y)
}

fn smooth_term(n_pcs: usize, centers: usize) -> SmoothTermSpec {
    SmoothTermSpec {
        name: "geo".to_string(),
        basis: SmoothBasisSpec::Matern {
            feature_cols: (0..n_pcs).collect(),
            spec: MaternBasisSpec {
                // What `matern(pc1, …, pc16, centers=24)` resolves to for
                // `d = 16` with an EXPLICIT center count
                // (`spatial_center_strategy_for_dimension`).
                center_strategy: CenterStrategy::EqualMassCovarRepresentative {
                    num_centers: centers,
                },
                periodic: None,
                // An omitted `length_scale=` stays Auto, which is what turns on
                // the exact-joint [rho, psi] spatial route the #2676 refusals
                // live in. A fixed scale takes the rho-only path and never
                // reaches them.
                length_scale: gam::terms::basis::MaternLengthScale::auto(),
                nu: MaternNu::FiveHalves,
                include_intercept: false,
                double_penalty: false,
                identifiability: MaternIdentifiability::default(),
                aniso_log_scales: None,
            },
            input_scale: None,
        },
        shape: ShapeConstraint::None,
        joint_null_rotation: None,
    }
}

fn fit_options() -> FitOptions {
    FitOptions {
        resource_policy: gam_runtime::resource::ResourcePolicy::default_library(),
        latent_cloglog: None,
        mixture_link: None,
        optimize_mixture: false,
        sas_link: None,
        optimize_sas: false,
        // The smoothing-correction branch (#2676 path 2) only runs when
        // inference is requested; the outer certificate (path 1) fires either
        // way. Ask for both.
        compute_inference: true,
        skip_rho_posterior_inference: false,
        max_iter: 200,
        tol: 1e-7,
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

fn main() {
    // `RUST_LOG=info` is what makes the certificate's own route visible; the
    // `[INDEF-HESS]` dumps are already at `warn`.
    // Log level is a positional argument, not `RUST_LOG`: reading the
    // environment is banned repo-wide (`build.rs`'s scanner, `feedback_no_env_vars`).
    let args: Vec<String> = std::env::args().collect();
    let centers: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(24);
    let n: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(4000);
    let n_pcs: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(16);
    let level = args.get(4).map(String::as_str).unwrap_or("warn");
    gam_solve::progress_log::init_logging_at(log::LevelFilter::Warn);
    gam_solve::progress_log::set_log_level(level);

    eprintln!("[repro2676] centers={centers} n={n} n_pcs={n_pcs}");
    let (x_full, y) = geo_disease_columns(n, 20260226);
    let x = if n_pcs >= 16 {
        x_full
    } else {
        x_full.slice(ndarray::s![.., ..n_pcs]).to_owned()
    };
    let n = y.len();
    let pos = y.iter().filter(|&&v| v > 0.5).count();
    eprintln!("[repro2676] prevalence={:.4}", pos as f64 / n as f64);

    let spec = TermCollectionSpec {
        linear_terms: vec![],
        random_effect_terms: vec![],
        smooth_terms: vec![smooth_term(n_pcs, centers)],
    };

    let t0 = Instant::now();
    let result = gam::fit_model(FitRequest::Standard(StandardFitRequest {
        data: gam::solver::fit_orchestration::StandardFitData::shared(x),
        y: std::sync::Arc::new(y),
        weights: std::sync::Arc::new(Array1::ones(n)),
        offset: std::sync::Arc::new(Array1::zeros(n)),
        spec,
        family: LikelihoodSpec::new(
            ResponseFamily::Binomial,
            InverseLink::Standard(StandardLink::Logit),
        ),
        estimate_tweedie_p: false,
        options: fit_options(),
        kappa_options: SpatialLengthScaleOptimizationOptions::default(),
        wiggle: None,
        coefficient_groups: Vec::new(),
        penalty_block_gamma_priors: Vec::new(),
        latent_coord: None,
    }));
    let dt = t0.elapsed().as_secs_f64();

    match result {
        Ok(FitResult::Standard(s)) => {
            eprintln!(
                "[repro2676] OK in {dt:.2}s :: p={} finite={}",
                s.fit.beta.len(),
                s.fit.beta.iter().all(|v: &f64| v.is_finite()),
            );
        }
        Ok(_) => eprintln!("[repro2676] unexpected result kind in {dt:.2}s"),
        Err(e) => eprintln!("[repro2676] FAILED in {dt:.2}s :: {e}"),
    }
}
