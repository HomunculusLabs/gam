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
//! ## Why the shapes are constants and the runs are not `#[ignore]`d
//!
//! Both stopwatches arrived (`7917759c7`) as `#[ignore]`d tests reading their
//! sweep out of `GAM_2757_*` environment variables. Each of those is a build
//! ban in this workspace — `#[ignore]` because a test that never runs is not a
//! statement, `env::var` because a run whose shape comes from the environment
//! is not reproducible from the tree — so the scanner aborted **every** build
//! in the workspace and no lane could compile anything. See `0c9ed39c5` for the
//! same lesson on the #2714 probe.
//!
//! The instrument is unchanged in what it measures; only its entry conditions
//! are. The sweep is a `const` below (raise it in a working tree to reach the
//! production cell), and the committed shape is small enough that the phase
//! table is produced on every run rather than never. Read it with
//!
//! ```sh
//! cargo test -p gam-sae --release --lib probe_2757 -- --nocapture
//! ```

use super::tests_frame_curvature_2757::{
    planted_term_for_probe, source_root_rows, source_stored_scalars, source_structure_tag,
    unit_rho_for_probe,
};
use crate::identifiability::FrameColumnLayout;
use crate::manifold::construction::ResidualGaugeCurvatureSource;
use crate::manifold::streamed_frame_curvature::StreamedFrameCurvatureOperator;
use ndarray::Array2;
use std::time::Instant;

/// Rows in the committed sweep.
const PROBE_ROWS: usize = 64;
/// Charts in the committed Euclidean sweep.
const PROBE_EUCLIDEAN_CHARTS: usize = 8;
/// Output widths in the committed Euclidean sweep. The #2731 production cell is
/// `p = 2048, charts = 32, n = 256`; raise this to walk toward it.
const PROBE_EUCLIDEAN_WIDTHS: [usize; 3] = [16, 32, 64];
/// Charts in the committed gauge-driving sweep.
const PROBE_GAUGE_CHARTS: usize = 4;
/// Rank of the output-Fisher metric root in the committed gauge-driving sweep.
const PROBE_GAUGE_METRIC_RANK: usize = 2;
/// Output widths in the committed gauge-driving sweep.
const PROBE_GAUGE_WIDTHS: [usize; 3] = [8, 16, 32];

/// Phase-by-phase wall-clock of the whole certification, on the branch the
/// #2731 cell actually ran.
#[test]
fn probe_2757_report_phase_profile_euclidean() {
    let n = PROBE_ROWS;
    let charts = PROBE_EUCLIDEAN_CHARTS;
    println!("\n#2757 phase profile — Euclidean metric (n={n}, charts={charts})");
    println!(
        "{:>6} {:>9} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "p",
        "param_dim",
        "two_lens",
        "curvature",
        "reduce+gen",
        "fidelity",
        "embed",
        "topology",
        "TOTAL"
    );
    for &p in &PROBE_EUCLIDEAN_WIDTHS {
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
        crate::inference::atom_lens::atom_two_lens(&term, &metric, None).expect("two-lens report");
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
        let streamed = super::tests_frame_curvature_2757::expect_stored(streamed, "unpinned path streams its curvature");
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
        let fidelity_reports: Vec<_> = (0..charts)
            .map(|k| super::coordinate_fidelity::atom_coordinate_fidelity(&term, k))
            .collect();
        let fidelity = t3.elapsed().as_secs_f64();

        let t4 = Instant::now();
        let embed_reports: Vec<_> = (0..charts)
            .map(|k| super::embeddedness::atom_decoder_embeddedness(&term, k))
            .collect();
        let embed = t4.elapsed().as_secs_f64();

        let t5 = Instant::now();
        let topology_reports: Vec<_> = (0..charts)
            .map(|k| super::persistence::atom_topology_persistence(&term, k))
            .collect();
        let topology = t5.elapsed().as_secs_f64();

        let t6 = Instant::now();
        term.fit_diagnostics_report(None, false, None, fitted.view(), None)
            .expect("diagnostics report");
        let total = t6.elapsed().as_secs_f64();

        println!(
            "{p:>6} {param_dim:>9} {two_lens:>10.3} {curvature_build:>10.3} \
             {reduce_and_generators:>10.3} {fidelity:>10.3} {embed:>10.3} {topology:>10.3} \
             {total:>10.3}"
        );
        println!(
            "        curvature={tag} stored_scalars={scalars} \
             (dense Gram would be {}) pinning_rank={} verdicts={} \
             per-chart reports={}/{}/{}",
            param_dim * param_dim,
            gauge.pinning_rank,
            gauge.generators.len(),
            fidelity_reports.len(),
            embed_reports.len(),
            topology_reports.len()
        );
    }
}

/// The other branch: a metric that genuinely couples output coordinates. This
/// is where the dense `param_dim × param_dim` Gram still lives.
#[test]
fn probe_2757_report_phase_profile_gauge_driving() {
    let n = PROBE_ROWS;
    let charts = PROBE_GAUGE_CHARTS;
    let rank = PROBE_GAUGE_METRIC_RANK;
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
    for &p in &PROBE_GAUGE_WIDTHS {
        let mut term = planted_term_for_probe(n, p, charts, true);
        let factors = Array2::<f64>::from_shape_fn((n, p * rank), |_| lcg() - 0.5);
        let metric = gam_problem::RowMetric::output_fisher(std::sync::Arc::new(factors), p, rank)
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

        let (model, source) = term
            .to_residual_gauge_model(metric.clone(), None, false)
            .expect("certificate model");
        let views = term.atom_parameter_views();
        let ops: Vec<Option<crate::identifiability::OrbitPenaltyOperator>> =
            (0..charts).map(|_| None).collect();
        // The production route, whichever arm this shape lands on.
        let pin = Array2::<f64>::zeros((0, param_dim));
        let t2 = Instant::now();
        let gauge = match source {
            ResidualGaugeCurvatureSource::Stored(stored) => {
                crate::identifiability::residual_gauge_exact_from_curvature(
                    &model, &views, &ops, stored,
                )
                .expect("residual gauge")
            }
            ResidualGaugeCurvatureSource::Streamed { .. } => {
                let operator = StreamedFrameCurvatureOperator::new(
                    &term,
                    &metric,
                    &layout,
                    &pin,
                    n * rank,
                )
                .expect("streamed operator");
                crate::identifiability::residual_gauge_exact_from_streamed(
                    &model, &views, &ops, &operator,
                )
                .expect("residual gauge")
            }
        };
        let reduce_and_generators = t2.elapsed().as_secs_f64();

        println!(
            "{p:>6} {param_dim:>9} {root_rows:>10} {tag:>12} {curvature_build:>12.3} \
             {reduce_and_generators:>12.3} {:>10.3}",
            curvature_build + reduce_and_generators
        );
        println!(
            "        stored_scalars={} (dense Gram would be {}) pinning_rank={} ({}) verdicts={}",
            curvature.stored_scalars(),
            param_dim * param_dim,
            gauge.pinning_rank,
            gauge.pinning_rank_support.label(),
            gauge.generators.len()
        );
    }
}

/// Rows in the gauge-branch cost-law sweep.
const PROBE_LAW_ROWS: usize = 64;
/// Charts in the gauge-branch cost-law sweep.
const PROBE_LAW_CHARTS: usize = 8;
/// Metric root rank in the gauge-branch cost-law sweep. `root_rows = n · rank`
/// is held FIXED across the sweep so the only thing that moves is `param_dim`,
/// which is what makes the fitted exponents below statements about `param_dim`
/// and not about the row count. It is also the smallest rank that keeps every
/// cell on the branch under test (`root_rows > param_dim` at the widest cell).
const PROBE_LAW_METRIC_RANK: usize = 9;
/// Output widths in the gauge-branch cost-law sweep. `param_dim = 8·p` runs
/// `128 → 512`, a factor of 4, over which a cubic moves 64x and a linear pass
/// moves 4x. The #2731 production cell is `p = 2048, charts = 32` at
/// `param_dim = 65 536`; raise these to walk toward it.
const PROBE_LAW_WIDTHS: [usize; 4] = [16, 32, 48, 64];

/// The cost LAW of the gauge-driving branch, measured on both routes at once.
///
/// #2757 was filed on `9.35x time / 4.26x memory` for a `2x` in `p`, read off
/// two production cells, and attributed to a dense symmetric eigendecomposition.
/// That attribution is now correct only here: the Euclidean branch holds `p`
/// blocks of `D × D` and the topology audit that replaced it as the wall was
/// rewritten in `b7e148809`. What is left is this branch, where the per-row
/// metric couples output coordinates so `H` has no block structure at all.
///
/// The two routes are timed on identical data over the IDENTICAL phase — build
/// the curvature, reduce it, verdict every generator — so the ratio is a
/// statement about that phase and nothing else:
///
/// * **materialize** — fold every root row into a `param_dim`-square triangular
///   factor and take its singular values. `param_dim²` scalars,
///   `root_rows·param_dim²` to build and `param_dim³` to read. This is what
///   production did between `8adae9a67` and the streamed route, and it is
///   retained as the equivalence witness `tests_streamed_curvature_2757` judges
///   against.
/// * **stream** — never materialize anything: `λ_max` by a certified matrix-free
///   Krylov solve, `ξᵀHξ` exactly from one pass that folds `RΞ` into a `G × G`
///   factor. `0` curvature scalars, `O(param_dim)` working set.
///
/// Both fitted exponents are printed. At the committed shape the cubic is not
/// yet the dominant term — `param_dim ≤ 512` is small enough that enumerating
/// and embedding the generators is — which is exactly why the exponents are
/// printed rather than a wall-clock bar asserted: the number to watch as the
/// sweep is raised is the materialized exponent climbing toward 3 while the
/// streamed one stays flat.
#[test]
fn probe_2757_gauge_branch_cost_law() {
    let n = PROBE_LAW_ROWS;
    let charts = PROBE_LAW_CHARTS;
    let rank = PROBE_LAW_METRIC_RANK;
    println!(
        "\n#2757 gauge-branch cost law (n={n}, charts={charts}, metric rank={rank}, \
         root_rows={})",
        n * rank
    );
    println!(
        "{:>6} {:>10} {:>12} {:>12} {:>12} {:>10} {:>8} {:>6}",
        "p", "param_dim", "mat scalars", "mat gauge", "stream gauge", "speedup", "passes", "gens"
    );
    let mut seed = 0x2757_0C05_7A00_0001u64;
    let mut lcg = move || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((seed >> 11) as f64) / ((1u64 << 53) as f64)
    };
    let mut log_dim: Vec<f64> = Vec::new();
    let mut log_materialized: Vec<f64> = Vec::new();
    let mut log_streamed: Vec<f64> = Vec::new();
    for &p in &PROBE_LAW_WIDTHS {
        let mut term = planted_term_for_probe(n, p, charts, true);
        let factors = Array2::<f64>::from_shape_fn((n, p * rank), |_| lcg() - 0.5);
        let metric = gam_problem::RowMetric::output_fisher(std::sync::Arc::new(factors), p, rank)
            .expect("output-Fisher metric");
        term.set_row_metric(metric.clone()).expect("conformable");
        let layout = FrameColumnLayout::new(p, &vec![1usize; charts]);
        let param_dim = layout.param_dim();
        assert!(
            n * rank > param_dim,
            "cell p={p} must take the branch where the root is the LARGER object: \
             root_rows={} vs param_dim={param_dim}",
            n * rank
        );

        let (model, source) = term
            .to_residual_gauge_model(metric.clone(), None, false)
            .expect("certificate model");
        assert_eq!(source_structure_tag(&source), "streamed_operator");
        assert_eq!(source_stored_scalars(&source), 0);
        assert_eq!(source_root_rows(&source), n * rank);
        let views = term.atom_parameter_views();
        let ops: Vec<Option<crate::identifiability::OrbitPenaltyOperator>> =
            (0..charts).map(|_| None).collect();
        let pin = Array2::<f64>::zeros((0, param_dim));

        // Route 1 — materialize, exactly as the pre-streamed production path did.
        let t0 = Instant::now();
        let materialized = term
            .residual_gauge_streamed_data_curvature(&metric, &layout, pin.clone())
            .expect("materialized curvature");
        let mat_scalars = materialized.stored_scalars();
        let mat_report = crate::identifiability::residual_gauge_exact_from_curvature(
            &model,
            &views,
            &ops,
            materialized,
        )
        .expect("materialized residual gauge");
        let mat_total = t0.elapsed().as_secs_f64();

        // Route 2 — the production route at current main, same phase.
        let t1 = Instant::now();
        let operator =
            StreamedFrameCurvatureOperator::new(&term, &metric, &layout, &pin, n * rank)
                .expect("streamed operator");
        let stream_report = crate::identifiability::residual_gauge_exact_from_streamed(
            &model, &views, &ops, &operator,
        )
        .expect("streamed residual gauge");
        let stream_total = t1.elapsed().as_secs_f64();
        let passes = crate::identifiability::streamed_lambda_max(&operator)
            .expect("certified λ_max")
            .passes;

        log_dim.push((param_dim as f64).ln());
        log_materialized.push(mat_total.max(1e-9).ln());
        log_streamed.push(stream_total.max(1e-9).ln());

        println!(
            "{p:>6} {param_dim:>10} {mat_scalars:>12} {mat_total:>12.4} {stream_total:>12.4} \
             {:>10.2} {passes:>8} {:>6}",
            mat_total / stream_total.max(1e-9),
            stream_report.generators.len()
        );
        assert_eq!(
            mat_scalars,
            param_dim * param_dim,
            "the witness must be the param_dim-square object this route exists to avoid"
        );
        assert_eq!(
            mat_report.residual_gauge_dim, stream_report.residual_gauge_dim,
            "the two routes must certify the same group at every cell of the sweep"
        );
        assert_eq!(mat_report.group_signature(), stream_report.group_signature());
    }
    let slope = |ys: &[f64]| -> f64 {
        let m = log_dim.len() as f64;
        let mean_x = log_dim.iter().sum::<f64>() / m;
        let mean_y = ys.iter().sum::<f64>() / m;
        let cov: f64 = log_dim
            .iter()
            .zip(ys)
            .map(|(x, y)| (x - mean_x) * (y - mean_y))
            .sum();
        let var: f64 = log_dim.iter().map(|x| (x - mean_x) * (x - mean_x)).sum();
        cov / var
    };
    println!(
        "  fitted exponent d(log time)/d(log param_dim): materialized = {:.2}, streamed = {:.2}",
        slope(&log_materialized),
        slope(&log_streamed)
    );
}
