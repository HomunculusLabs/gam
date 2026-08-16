#![cfg(test)]
//! #2757 probe — where the post-fit certification's wall-clock actually goes,
//! phase by phase, at the shape the issue was filed on.
//!
//! The issue measured `fit_diagnostics_report` at 3160.5 s / 45.97 GiB for
//! `p = 4096` and read a `dim ∝ p`, `time ∝ dim³`, `memory ∝ dim²` law off two
//! cells. The block-structured curvature (`2af28dddb`) removed the dense
//! `param_dim × param_dim` eigendecomposition on the branch where the metric
//! does not couple output coordinates. **This probe does not assume that
//! finished the job.** It times every phase of the report separately so the
//! surviving cost is measured rather than inferred, on both metric branches:
//!
//! * `metric.drives_gauge() == false` (Euclidean — what `diagnostic_metric`
//!   installs with no harvest, and what the #2731 cell ran) → the curvature is
//!   `p` blocks of `D × D`;
//! * `metric.drives_gauge() == true` (output-Fisher) → the curvature falls back
//!   to a root, or to a dense `(p·D)²` Gram once the root has more rows than
//!   columns, which is exactly the object #2757 is named for.
//!
//! Both are `#[ignore]`d: they are stopwatches, not gates. Drive them with
//!
//! ```sh
//! cargo test -p gam-sae --release --lib probe_2757 -- --ignored --nocapture
//! ```
//!
//! and set `GAM_2757_P_SWEEP` / `GAM_2757_CHARTS` / `GAM_2757_N` to move the
//! shape.

use super::tests_frame_curvature_2757::{planted_term_for_probe, unit_rho_for_probe};
use crate::identifiability::FrameColumnLayout;
use ndarray::Array2;
use std::time::Instant;

fn env_usize(key: &str, fallback: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(fallback)
}

fn env_sweep(key: &str, fallback: &[usize]) -> Vec<usize> {
    match std::env::var(key) {
        Ok(v) => v
            .split(',')
            .filter_map(|s| s.trim().parse::<usize>().ok())
            .collect(),
        Err(_) => fallback.to_vec(),
    }
}

/// Phase-by-phase wall-clock of the whole certification, on the branch the
/// #2731 cell actually ran.
#[test]
#[ignore = "stopwatch, not a gate: drive with --ignored --nocapture"]
fn probe_2757_report_phase_profile_euclidean() {
    let n = env_usize("GAM_2757_N", 256);
    let charts = env_usize("GAM_2757_CHARTS", 32);
    let sweep = env_sweep("GAM_2757_P_SWEEP", &[256, 512, 1024]);
    println!("\n#2757 phase profile — Euclidean metric (n={n}, charts={charts})");
    println!(
        "{:>6} {:>9} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "p", "param_dim", "two_lens", "curvature", "reduce+gen", "fidelity", "embed", "topology", "TOTAL"
    );
    for &p in &sweep {
        let term = planted_term_for_probe(n, p, charts, true);
        let rho = unit_rho_for_probe(charts);
        let fitted = term
            .try_fitted_target_aware(Array2::<f64>::zeros((n, p)).view(), Some(&rho))
            .expect("fitted");
        let metric = term.diagnostic_metric().expect("metric");
        assert!(!metric.drives_gauge(), "this arm is the Euclidean branch");
        let layout = FrameColumnLayout::new(p, &vec![1usize; charts]);
        let param_dim = layout.param_dim();

        let t0 = Instant::now();
        let _lens = crate::inference::atom_lens::atom_two_lens(&term, &metric, None)
            .expect("two-lens report");
        let two_lens = t0.elapsed().as_secs_f64();

        let t1 = Instant::now();
        let curvature = term
            .residual_gauge_streamed_data_curvature(
                &metric,
                &layout,
                Array2::<f64>::zeros((0, param_dim)),
            )
            .expect("streamed curvature");
        let curvature_build = t1.elapsed().as_secs_f64();
        let tag = curvature.structure_tag();
        let scalars = curvature.stored_scalars();

        let (model, streamed) = term
            .to_residual_gauge_model(metric.clone(), None, false)
            .expect("certificate model");
        let streamed = streamed.expect("unpinned path streams its curvature");
        let views = term.atom_parameter_views();
        let ops: Vec<Option<crate::identifiability::OrbitPenaltyOperator>> =
            (0..charts).map(|_| None).collect();
        let t2 = Instant::now();
        let gauge = crate::identifiability::residual_gauge_exact_from_curvature(
            &model, &views, &ops, streamed,
        )
        .expect("residual gauge");
        let reduce_and_generators = t2.elapsed().as_secs_f64();

        let t3 = Instant::now();
        let _fidelity: Vec<_> = (0..charts)
            .map(|k| super::coordinate_fidelity::atom_coordinate_fidelity(&term, k))
            .collect();
        let fidelity = t3.elapsed().as_secs_f64();

        let t4 = Instant::now();
        let _embed: Vec<_> = (0..charts)
            .map(|k| super::embeddedness::atom_decoder_embeddedness(&term, k))
            .collect();
        let embed = t4.elapsed().as_secs_f64();

        let t5 = Instant::now();
        let _topology: Vec<_> = (0..charts)
            .map(|k| super::persistence::atom_topology_persistence(&term, k))
            .collect();
        let topology = t5.elapsed().as_secs_f64();

        let t6 = Instant::now();
        let _report = term
            .fit_diagnostics_report(None, false, None, fitted.view(), None)
            .expect("diagnostics report");
        let total = t6.elapsed().as_secs_f64();

        println!(
            "{p:>6} {param_dim:>9} {two_lens:>10.3} {curvature_build:>10.3} \
             {reduce_and_generators:>10.3} {fidelity:>10.3} {embed:>10.3} {topology:>10.3} \
             {total:>10.3}"
        );
        println!(
            "        curvature={tag} stored_scalars={scalars} \
             (dense Gram would be {}) pinning_rank={} verdicts={}",
            param_dim * param_dim,
            gauge.pinning_rank,
            gauge.generators.len()
        );
    }
}

/// The other branch: a metric that genuinely couples output coordinates. This
/// is where the dense `param_dim × param_dim` Gram still lives.
#[test]
#[ignore = "stopwatch, not a gate: drive with --ignored --nocapture"]
fn probe_2757_report_phase_profile_gauge_driving() {
    let n = env_usize("GAM_2757_N", 256);
    let charts = env_usize("GAM_2757_CHARTS", 8);
    let rank = env_usize("GAM_2757_RANK", 4);
    let sweep = env_sweep("GAM_2757_P_SWEEP", &[32, 64, 128]);
    println!(
        "\n#2757 phase profile — output-Fisher (gauge-driving) metric \
         (n={n}, charts={charts}, metric rank={rank})"
    );
    println!(
        "{:>6} {:>9} {:>10} {:>12} {:>12} {:>12} {:>10}",
        "p", "param_dim", "root_rows", "structure", "curv build", "reduce+gen", "TOTAL"
    );
    let mut seed = 0x2757_0BE1_0000_0001u64;
    let mut lcg = move || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((seed >> 11) as f64) / ((1u64 << 53) as f64)
    };
    for &p in &sweep {
        let mut term = planted_term_for_probe(n, p, charts, true);
        let factors = Array2::<f64>::from_shape_fn((n, p * rank), |_| lcg() - 0.5);
        let metric =
            gam_problem::RowMetric::output_fisher(std::sync::Arc::new(factors), p, rank)
                .expect("output-Fisher metric");
        term.set_row_metric(metric.clone()).expect("conformable");
        assert!(metric.drives_gauge());
        let layout = FrameColumnLayout::new(p, &vec![1usize; charts]);
        let param_dim = layout.param_dim();

        let t1 = Instant::now();
        let curvature = term
            .residual_gauge_streamed_data_curvature(
                &metric,
                &layout,
                Array2::<f64>::zeros((0, param_dim)),
            )
            .expect("streamed curvature");
        let curvature_build = t1.elapsed().as_secs_f64();
        let tag = curvature.structure_tag();
        let root_rows = curvature.root_rows();

        let (model, streamed) = term
            .to_residual_gauge_model(metric.clone(), None, false)
            .expect("certificate model");
        let streamed = streamed.expect("unpinned path streams its curvature");
        let views = term.atom_parameter_views();
        let ops: Vec<Option<crate::identifiability::OrbitPenaltyOperator>> =
            (0..charts).map(|_| None).collect();
        let t2 = Instant::now();
        let gauge = crate::identifiability::residual_gauge_exact_from_curvature(
            &model, &views, &ops, streamed,
        )
        .expect("residual gauge");
        let reduce_and_generators = t2.elapsed().as_secs_f64();

        println!(
            "{p:>6} {param_dim:>9} {root_rows:>10} {tag:>12} {curvature_build:>12.3} \
             {reduce_and_generators:>12.3} {:>10.3}",
            curvature_build + reduce_and_generators
        );
        println!(
            "        stored_scalars={} (dense Gram would be {}) pinning_rank={} verdicts={}",
            curvature.stored_scalars(),
            param_dim * param_dim,
            gauge.pinning_rank,
            gauge.generators.len()
        );
    }
}
