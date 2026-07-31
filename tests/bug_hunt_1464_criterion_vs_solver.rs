//! #1464 diagnostic experiment: evaluate the raw production fixed-κ
//! profiled-REML surface at +κ versus −κ for a genuinely hyperbolic dataset.
//!
//! The historical full fit rails κ̂ to the positive chart bound for both mirror
//! datasets. This scan records whether independently pinned complete fits have
//! the same preference. It does not settle the current curvature estimand's
//! optimizer-versus-objective question, because that point estimate and its
//! inference are driven by the distinct continuously differentiable curvature
//! likelihood profile.
//!
//! `constant_curvature_profiled_reml_scores` calls
//! `fixed_kappa_profiled_reml_score` on the data/spec/family/options materialised
//! exactly like the full fit. Each diagnostic score is therefore an independently
//! pinned complete production fit, not a basis-local re-derivation. Curvature
//! point estimation and inference use the distinct curvature likelihood
//! profile, so this experiment diagnoses the raw
//! production-fit surface rather than claiming to reproduce that estimand.
//!
//! Diagnostic only (plain `eprintln!`, no `{:?}`). It asserts nothing about which
//! way the answer falls — it PRINTS the scores so the maintainer reads the verdict.

use gam::geometry::constant_curvature::ConstantCurvature;
use gam::{
    FitConfig, constant_curvature_profiled_reml_scores, encode_recordswith_inferred_schema,
    init_parallelism,
};

use csv::StringRecord;

use gam::utils::splitmix64;
fn next_unit(state: &mut u64) -> f64 {
    (splitmix64(state) >> 11) as f64 / (1u64 << 53) as f64
}
fn next_gauss(state: &mut u64) -> f64 {
    let u1 = next_unit(state).max(1.0e-12);
    let u2 = next_unit(state);
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

/// Identical generator to the #1464 contract test, so the criterion is probed on
/// exactly the dataset the full fit rails on.
fn curved_dataset(kappa_star: f64, seed: u64) -> gam::data::EncodedDataset {
    let radius = 0.68_f64;
    let noise = 0.02_f64;
    let n = 600usize;
    let manifold = ConstantCurvature::new(2, kappa_star);
    let origin = ndarray::array![0.0_f64, 0.0_f64];
    let mut st = seed;
    let headers = vec!["y".to_string(), "x1".to_string(), "x2".to_string()];
    let mut records = Vec::with_capacity(n);
    let mut filled = 0usize;
    while filled < n {
        let a = 2.0 * next_unit(&mut st) - 1.0;
        let b = 2.0 * next_unit(&mut st) - 1.0;
        if a * a + b * b > 1.0 {
            continue;
        }
        let x1 = a * radius;
        let x2 = b * radius;
        let pt = ndarray::array![x1, x2];
        let d = manifold
            .distance(pt.view(), origin.view())
            .expect("in-chart geodesic distance");
        let y = 2.0 * (-d).exp() - 1.0 + noise * next_gauss(&mut st);
        records.push(StringRecord::from(vec![
            y.to_string(),
            x1.to_string(),
            x2.to_string(),
        ]));
        filled += 1;
    }
    encode_recordswith_inferred_schema(headers, records).expect("encode curved dataset")
}

fn print_scores(label: &str, kappa_star: f64, seed: u64) -> f64 {
    let data = curved_dataset(kappa_star, seed);
    let config = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    // Probe symmetric κ corners: ±chart-bound (~±1.08 at radius 0.68), the planted
    // truth ±2 (clamped into the chart by the production fit), an interior ±0.5,
    // and flat 0. Lower V_p = preferred (negative-log-evidence the outer loop
    // minimises).
    let kappas = [-2.0_f64, -1.08, -0.5, 0.0, 0.5, 1.08, 2.0];
    let scores = constant_curvature_profiled_reml_scores(
        "y ~ curv(x1, x2, centers=10)",
        &data,
        &config,
        &kappas,
    )
    .expect("fixed-κ profiled-REML scan should succeed");

    eprintln!("[#1464-crit] === {label} (planted kappa* = {kappa_star:+}) ===");
    let mut best_k = f64::NAN;
    let mut best_v = f64::INFINITY;
    for (k, v) in &scores {
        eprintln!("[#1464-crit]   V_p(kappa={k:+.4}) = {v}");
        if *v < best_v {
            best_v = *v;
            best_k = *k;
        }
    }
    eprintln!(
        "[#1464-crit]   --> argmin V_p over probed grid: kappa = {best_k:+.4} (V_p = {best_v})"
    );
    eprintln!(
        "[#1464-crit]   verdict: criterion prefers {} curvature for this {label} dataset",
        if best_k < 0.0 {
            "NEGATIVE (hyperbolic)"
        } else if best_k > 0.0 {
            "POSITIVE (spherical)"
        } else {
            "FLAT"
        }
    );
    best_k
}

#[test]
fn curv_raw_fixed_fit_surface_is_finite_for_both_mirror_datasets() {
    init_parallelism();
    // Diagnostic only: the raw pinned-fit surface is distinct from the
    // curvature likelihood estimand, so its argmin is payload rather than a sign gate.
    let hyp_k = print_scores("HYPERBOLIC", -2.0, 0x5151_0003);
    let sph_k = print_scores("SPHERICAL", 2.0, 0x5151_0001);
    assert!(
        hyp_k.is_finite() && sph_k.is_finite(),
        "raw fixed-fit diagnostics must return finite argmins: hyperbolic={hyp_k}, spherical={sph_k}"
    );
}
