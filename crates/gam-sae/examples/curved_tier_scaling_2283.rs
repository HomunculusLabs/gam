//! #2283 / #1017 — cheap reproducer for the CURVED-tier serial wall.
//!
//! The theorem-faithful hybrid row on #2283 is blocked by the curved tier, not
//! by the flat tier: the decoder-refresh wall was killed by #1017's block CG
//! (measured 69,173.60 s -> 127.18 s per epoch). What remains is a curved fit
//! that ran 4 h 41 m at ~107 % CPU on a 30-vCPU box at N=6400, P=2048, 16
//! charts, `d_atom=1`, `top_k=2` — effectively single-threaded.
//!
//! The mechanism this example is built to expose is the per-row packed row-jet
//! buffer. `SaeScheduledRowJets::zeros(q, p, n_beta)` allocates and zeroes
//!
//! ```text
//! q*p + q^2*p + n_beta*p + 2*q*n_beta*p   doubles
//! ```
//!
//! once per row, and the TopK / independent-logistic gate then writes only the
//! borders whose atom is *active* — `top_k` of `charts` of them. Every other
//! border channel is allocated and memset for the sole purpose of reading back
//! as zero. The predicted scaling is therefore **linear in `p` and linear in
//! the chart count** (through `n_beta`) at fixed `top_k`, on one core.
//!
//! Sweeping `GAM_2283_P` at fixed charts, and `GAM_2283_CHARTS` at fixed `p`,
//! separates that term from everything else in the fit: any cost that is not
//! the per-row jet buffer is either constant in one of those knobs or grows
//! with `top_k` rather than with the chart count.
//!
//! Run (release; the numbers are meaningless in debug):
//!
//! ```text
//! GAM_2283_N=256 GAM_2283_P=128 GAM_2283_CHARTS=8 \
//!   cargo run --release -p gam-sae --example curved_tier_scaling_2283
//! ```
//!
//! Knobs, all optional, with the defaults below:
//!
//! | env | default | meaning |
//! |---|---|---|
//! | `GAM_2283_N` | 256 | rows |
//! | `GAM_2283_P` | 128 | ambient output dimension |
//! | `GAM_2283_CHARTS` | 8 | circle charts (atoms) |
//! | `GAM_2283_TOPK` | 2 | active atoms per row |
//! | `GAM_2283_MAXITER` | 8 | inner Newton budget |
//! | `GAM_2283_RHO` | 1 | run the outer rho search (0 disables) |
//! | `GAM_2283_SEED` | 0 | data seed |
//!
//! One machine-readable `[2283-scaling]` line is printed on stdout so a sweep
//! script can parse it without re-deriving the shape.

use std::time::Instant;

use gam_sae::manifold::{
    SaeFitAssignmentKind, SaeFitConfig, SaeFitRequest, SaeFitSeedReport, SaeFitSeedRequest,
    SaeMinimalSeedReport, SaeMinimalSeedRequest, build_sae_fit_seed, build_sae_minimal_seed,
    run_sae_manifold_fit,
};
use gam_terms::analytic_penalties::AnalyticPenaltyRegistry;
use ndarray::Array2;

fn env_usize(key: &str, fallback: usize) -> usize {
    match std::env::var(key) {
        Ok(raw) => raw
            .trim()
            .parse::<usize>()
            .unwrap_or_else(|error| panic!("{key} must parse as usize: {error}")),
        Err(_) => fallback,
    }
}

/// Deterministic, dependency-free uniform stream so the fixture reproduces
/// bit-for-bit on any host without pulling an RNG crate into an example.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn centered(&mut self) -> f64 {
        self.unit() * 2.0 - 1.0
    }
}

/// `charts` circles embedded in `p` dimensions: row `i` sits on chart
/// `i % charts` at angle `theta_i`, decoded through that chart's own random
/// `(cos, sin)` frame, plus a small isotropic perturbation. This is the shape
/// the hybrid arm's curved tier fits — `d_atom = 1`, periodic basis — at a
/// scale where one fit is seconds rather than hours.
fn synthetic_charted_circles(n: usize, p: usize, charts: usize, seed: u64) -> Array2<f64> {
    let mut rng = SplitMix64(seed.wrapping_mul(0x2545_F491_4F6C_DD1D).wrapping_add(1));
    let frames: Vec<(Vec<f64>, Vec<f64>, Vec<f64>)> = (0..charts)
        .map(|_| {
            let center: Vec<f64> = (0..p).map(|_| 0.25 * rng.centered()).collect();
            let cos_dir: Vec<f64> = (0..p).map(|_| rng.centered()).collect();
            let sin_dir: Vec<f64> = (0..p).map(|_| rng.centered()).collect();
            let norm = |v: &Vec<f64>| -> f64 {
                v.iter().map(|value| value * value).sum::<f64>().sqrt().max(1e-12)
            };
            let cos_norm = norm(&cos_dir);
            let sin_norm = norm(&sin_dir);
            (
                center,
                cos_dir.into_iter().map(|value| value / cos_norm).collect(),
                sin_dir.into_iter().map(|value| value / sin_norm).collect(),
            )
        })
        .collect();

    let mut target = Array2::<f64>::zeros((n, p));
    for row in 0..n {
        let chart = row % charts;
        let theta = std::f64::consts::TAU * (row as f64) / (n as f64)
            + 0.37 * (chart as f64);
        let (center, cos_dir, sin_dir) = &frames[chart];
        for col in 0..p {
            target[[row, col]] = center[col]
                + theta.cos() * cos_dir[col]
                + theta.sin() * sin_dir[col]
                + 0.01 * rng.centered();
        }
    }
    target
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let n = env_usize("GAM_2283_N", 256);
    let p = env_usize("GAM_2283_P", 128);
    let charts = env_usize("GAM_2283_CHARTS", 8);
    let top_k = env_usize("GAM_2283_TOPK", 2);
    let max_iter = env_usize("GAM_2283_MAXITER", 8);
    let run_outer_rho_search = env_usize("GAM_2283_RHO", 1) != 0;
    let seed = env_usize("GAM_2283_SEED", 0) as u64;

    assert!(top_k >= 1 && top_k <= charts, "top_k must lie in 1..=charts");

    // The curved tier's own per-evaluation telemetry (`[SAE/outer] eval=…`,
    // `[SAE/inner] it=…`) is emitted through `log`, and the default backend
    // filter is `Warn` — which is why a curved fit that is looping has been
    // indistinguishable from one that is wedged on every prior attempt at this
    // measurement. Turn it on here, and start the process monitor so a stall
    // names its own frame instead of being reported as "no output".
    let level = match std::env::var("GAM_2283_LOG").as_deref() {
        Ok("debug") => log::LevelFilter::Debug,
        Ok("trace") => log::LevelFilter::Trace,
        Ok("warn") => log::LevelFilter::Warn,
        _ => log::LevelFilter::Info,
    };
    gam_solve::progress_log::init_logging_at(level);
    gam_runtime::process_monitor::start();

    let target = synthetic_charted_circles(n, p, charts, seed);

    let assignment_kind = SaeFitAssignmentKind::TopK;
    let build_started = Instant::now();
    let minimal = build_sae_minimal_seed(SaeMinimalSeedRequest {
        target: target.view(),
        atom_basis: vec!["periodic".to_string(); charts],
        atom_dim: vec![1; charts],
        assignment_kind,
        alpha: 1.0,
        tau: 1.0,
        threshold: 0.0,
        top_k: Some(top_k),
        random_state: seed,
        initial_logits: None,
        initial_coords: None,
    })?;
    let SaeMinimalSeedReport {
        geometry_plans,
        basis_values,
        basis_jacobian,
        decoder_coefficients,
        smooth_penalties,
        initial_logits,
        initial_coords,
        refine_routing,
    } = minimal;
    let border_scalars = decoder_coefficients.len();

    let registry = AnalyticPenaltyRegistry::new();
    let seed_report = build_sae_fit_seed(SaeFitSeedRequest {
        target: target.view(),
        geometry_plans: &geometry_plans,
        basis_values: basis_values.view(),
        basis_jacobian: basis_jacobian.view(),
        decoder_coefficients: decoder_coefficients.view(),
        smooth_penalties: smooth_penalties.view(),
        initial_logits: initial_logits.view(),
        initial_coords: initial_coords.view(),
        alpha: 1.0,
        tau: 1.0,
        learnable_alpha: false,
        assignment_kind,
        sparsity_strength: 1.0,
        smoothness: 1.0,
        max_iter,
        learning_rate: 1.0,
        ridge_ext_coord: 1.0e-6,
        ridge_beta: 1.0e-6,
        top_k: Some(top_k),
        threshold: 0.0,
        native_ard_enabled: true,
        seed_refine_routing: refine_routing,
        seed_refine_random_state: seed,
        data_row_reseed: false,
        fit_config: SaeFitConfig::default(),
        temperature_schedule: None,
        fisher_metric: None,
        row_loss_weights: None,
        registry: &registry,
    })?;
    let SaeFitSeedReport {
        base_term,
        initial_rho,
        isometry_pin_active,
        metric_provenance,
    } = seed_report;
    let seed_secs = build_started.elapsed().as_secs_f64();

    let fit_started = Instant::now();
    let report = run_sae_manifold_fit(SaeFitRequest {
        reconstruction_optimism_folds: None,
        base_term,
        target,
        registry,
        initial_rho,
        max_iter,
        learning_rate: 1.0,
        ridge_ext_coord: 1.0e-6,
        ridge_beta: 1.0e-6,
        alpha: 1.0,
        isometry_pin_active,
        metric_provenance,
        promote_from_residual: false,
        run_structure_search: false,
        run_outer_rho_search,
        structured_residual_passes: 0,
        cancel: None,
    })?
    .manifold_or_error()?;
    let fit_secs = fit_started.elapsed().as_secs_f64();

    println!(
        "[2283-scaling] n={n} p={p} charts={charts} top_k={top_k} max_iter={max_iter} \
         rho_search={run_outer_rho_search} border_scalars={border_scalars} \
         seed_s={seed_secs:.3} fit_s={fit_secs:.3} r2={:.6}",
        report.reconstruction_r2
    );
    Ok(())
}
