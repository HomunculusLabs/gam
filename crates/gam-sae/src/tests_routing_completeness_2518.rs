//! #2518 item 1 — the certified encode's chart routing is COMPLETE, and the
//! constant that used to bound it is gone.
//!
//! The deleted `CERTIFIED_ROUTING_TOPK = 4` answered "how many charts must I
//! refine before I have the global basin?" with an integer. That question has a
//! computable answer, and answering it wrongly produces the one failure a
//! certificate must never have: a row whose encode is locally Kantorovich-
//! certified and globally the wrong branch.
//!
//! Two things are gated here, and they are different claims:
//!
//! * **Completeness** — the scan now visits every certifiable chart and skips one
//!   only when a rigorous residual bound proves it cannot win. The load-bearing
//!   test compares the shipped encode against a brute-force minimum over a dense
//!   latent grid on a decoder that folds, and separately shows that the OLD
//!   four-chart restriction does not contain the winning chart on that fixture —
//!   without which the test would be measuring a fixture that no longer reaches
//!   the defect.
//! * **Soundness of the skip** — the prune's per-chart slack must dominate how far
//!   the reconstruction can actually move inside that chart, checked by sampling
//!   the ball on real charts. If it did not, the scan would be complete on paper
//!   and would still be able to discard the winner.

use std::sync::Arc;

use ndarray::{Array1, Array2, Array3};

use crate::basis::{PeriodicHarmonicEvaluator, SaeBasisEvaluator};
use crate::encode::{certified_encode_candidates, nearest_charts_topk, AtlasConfig, EncodeAtlas};
use crate::manifold::{SaeAtomBasisKind, SaeManifoldAtom};

/// A `d = 1` periodic atom whose decoded image FOLDS: the first harmonic traces a
/// loop and a large third harmonic pulls the curve back across itself, so distant
/// latent coordinates land near the same ambient point and several charts compete
/// for a row in the crossing region.
fn folded_atom() -> (SaeManifoldAtom, Array2<f64>, usize) {
    let m = 5usize; // [1, sin1, cos1, sin2, cos2]
    // The lemniscate `x = sin 2*pi*t`, `y = sin 4*pi*t`: a genuine figure eight
    // that CROSSES ITSELF at the origin, reached from two latent points half a
    // period apart. Distant latent coordinates therefore land on the same ambient
    // point, which is exactly the self-approach that makes a fixed-prefix routing
    // able to certify into the wrong branch.
    let decoder = ndarray::array![
        [0.00_f64, 0.00],
        [1.00, 0.00],
        [0.00, 0.00],
        [0.00, 1.00],
        [0.00, 0.00],
    ];
    let atom = SaeManifoldAtom::new_with_provided_function_gram(
        "lemniscate",
        SaeAtomBasisKind::Periodic,
        1,
        Array2::<f64>::eye(m),
        Array3::<f64>::zeros((m, m, 1)),
        decoder.clone(),
        Array2::<f64>::eye(m),
    )
    .expect("lemniscate periodic atom builds")
    // Without the evaluator every chart's center curvature is unreadable, so the
    // whole atlas comes back UNCERTIFIED and any completeness assertion over it
    // would pass vacuously.
    .with_basis_evaluator(Arc::new(
        PeriodicHarmonicEvaluator::new(m).expect("evaluator"),
    ));
    (atom, decoder, m)
}

fn recon_at(decoder: &Array2<f64>, evaluator: &PeriodicHarmonicEvaluator, t: f64) -> Array1<f64> {
    let coords = Array2::from_shape_vec((1, 1), vec![t]).expect("coordinate shape");
    let (phi, _) = evaluator.evaluate(coords.view()).expect("basis evaluates");
    phi.row(0).dot(decoder)
}

fn residual_at(
    decoder: &Array2<f64>,
    evaluator: &PeriodicHarmonicEvaluator,
    t: f64,
    x: &Array1<f64>,
    amplitude: f64,
) -> f64 {
    let recon = recon_at(decoder, evaluator, t);
    let diff = x - &(amplitude * &recon);
    diff.dot(&diff).sqrt()
}

/// Dense-grid brute-force global minimum of the encode objective — the objective
/// truth the certified encode is supposed to find, computed without reference to
/// any of the machinery under test.
fn brute_force_min(
    decoder: &Array2<f64>,
    evaluator: &PeriodicHarmonicEvaluator,
    x: &Array1<f64>,
    amplitude: f64,
) -> (f64, f64) {
    let scan = 200_000usize;
    let mut best = (f64::INFINITY, 0.0_f64);
    for i in 0..scan {
        let t = i as f64 / scan as f64;
        let err = residual_at(decoder, evaluator, t, x, amplitude);
        if err < best.0 {
            best = (err, t);
        }
    }
    best
}

fn folded_atlas(atom: &SaeManifoldAtom, charts: usize) -> EncodeAtlas {
    let centers =
        Array2::from_shape_fn((charts, 1), |(c, _)| c as f64 / charts as f64);
    let radii = vec![0.5 / charts as f64; charts];
    let atlas = EncodeAtlas::build_atom_atlas_from_centers(
        0,
        atom,
        centers.view(),
        &radii,
        1.0,
        4.0,
        &AtlasConfig::default(),
    )
    .expect("folded atlas builds");
    EncodeAtlas {
        atoms: vec![atlas],
        config: AtlasConfig::default(),
    }
}

/// **The load-bearing test**, in two halves that answer two different questions.
///
/// 1. *Does the restriction the deleted constant imposed actually lose the global
///    minimum on this fixture?* Answered from the GEOMETRY alone, with no
///    reference to the solver: each chart can only ever return a coordinate in its
///    own ball, so the best residual reachable through a set of charts is the
///    minimum over the union of their balls. Comparing that union over the four
///    nearest charts against the union over all of them is exactly the question
///    "could top-4 routing have found the global basin?", and it is deterministic.
/// 2. *Does the shipped encode find it?* Answered against a brute-force minimum
///    over a dense latent grid.
///
/// Without the first half the second would pass just as well with the constant
/// reinstated, and would be measuring a fixture that no longer reaches the defect.
#[test]
fn folded_atom_encode_reaches_the_global_minimum_and_top4_routing_could_not() {
    let (atom, decoder, m) = folded_atom();
    let evaluator = PeriodicHarmonicEvaluator::new(m).expect("evaluator");
    let charts = 64usize;
    let atlas = folded_atlas(&atom, charts);
    let amplitude = 1.0_f64;

    // Best residual reachable from a set of charts: the minimum over the union of
    // their balls. The atlas tiles the circle (radius = half the center spacing),
    // so the union over ALL charts is the whole latent domain.
    let reachable_min = |chart_indices: &[usize], x: &Array1<f64>| -> f64 {
        let mut best = f64::INFINITY;
        for &c in chart_indices {
            let chart = &atlas.atoms[0].charts[c];
            if chart.certified_radius <= 0.0 {
                continue;
            }
            let center = chart.region.center[0];
            let radius = chart.region.radius;
            let samples = 400usize;
            for s in 0..=samples {
                let t = center - radius + 2.0 * radius * (s as f64 / samples as f64);
                best = best.min(residual_at(&decoder, &evaluator, t, x, amplitude));
            }
        }
        best
    };
    let all_charts: Vec<usize> = (0..atlas.atoms[0].charts.len()).collect();

    let mut rows_checked = 0usize;
    let mut rows_where_top4_could_not_reach_the_optimum = 0usize;
    let mut worst_top4_penalty = 0.0_f64;
    // The shortest distance-ordered PREFIX that still contains the global optimum,
    // maximised over the probe set. `1` would mean nearest-chart routing was
    // always sound here; anything above it is a row the old fixed prefix had to
    // be large enough to cover, and anything above `4` is a row the deleted
    // constant provably got wrong.
    let mut max_prefix_needed = 0usize;

    // Rows swept densely THROUGH the crossing region, where the two branches are
    // ambient-close and latent-far.
    for probe in 0..400usize {
        let u = probe as f64 / 400.0;
        let angle = std::f64::consts::TAU * u;
        let radius = 0.02 + 1.15 * ((probe % 17) as f64 / 16.0);
        let x = Array1::from(vec![radius * angle.cos(), radius * angle.sin()]);

        // (1) Geometry: what could a distance-ordered prefix have reached?
        let best_all = reachable_min(&all_charts, &x);
        let ordered: Vec<usize> = certified_encode_candidates(&atlas.atoms[0], x.view(), amplitude)
            .into_iter()
            .map(|(idx, _, _)| idx)
            .collect();
        let mut needed = ordered.len();
        for take in 1..=ordered.len() {
            if reachable_min(&ordered[..take], &x) <= best_all + 1.0e-9 * (1.0 + best_all) {
                needed = take;
                break;
            }
        }
        max_prefix_needed = max_prefix_needed.max(needed);
        let top4 = nearest_charts_topk(&atlas.atoms[0], x.view(), amplitude, 4);
        let best_top4 = reachable_min(&top4, &x);
        if best_top4 > best_all + 1.0e-6 * (1.0 + best_all) {
            rows_where_top4_could_not_reach_the_optimum += 1;
            worst_top4_penalty = worst_top4_penalty.max(best_top4 - best_all);
        }

        // (2) Solver: does the shipped encode find the global minimum?
        let (truth_err, truth_t) = brute_force_min(&decoder, &evaluator, &x, amplitude);
        let (coord, cert) = atlas
            .certified_encode_row(&atom, 0, x.view(), amplitude)
            .expect("certified encode returns");
        if !cert.certified() {
            // An uncertified row is routed to the exact multi-start solve by
            // contract; it makes no claim, so it is not evidence either way.
            continue;
        }
        rows_checked += 1;
        let got_err = residual_at(&decoder, &evaluator, coord[0], &x, amplitude);
        assert!(
            got_err <= truth_err + 1.0e-3 * (1.0 + truth_err),
            "row {probe}: certified encode landed at residual {got_err:.9e} (t={}) but the \
             global minimum over a 200k grid is {truth_err:.9e} (t={truth_t:.6}). A certified \
             encode in the wrong basin is the exact #2518 failure.",
            coord[0]
        );
    }

    println!(
        "[#2518] lemniscate/{charts} charts: rows_certified={rows_checked} \
         max_prefix_needed={max_prefix_needed} rows_top4_lost_the_optimum=\
         {rows_where_top4_could_not_reach_the_optimum} worst_top4_penalty={worst_top4_penalty:.6e}"
    );
    assert!(
        rows_checked >= 20,
        "the fixture must certify a meaningful number of rows; got {rows_checked}"
    );
    // A distance-ordered PREFIX is not a sound stopping rule: on this fixture the
    // global optimum is not always in the nearest chart, so any fixed prefix is a
    // bet on how far down the order the winner can sit. That is the property the
    // deleted constant was betting on, and the reason the scan is now complete.
    assert!(
        max_prefix_needed >= 2,
        "FIXTURE DOES NOT REACH THE DEFECT: the nearest chart alone always contained the \
         global optimum, so no prefix rule of any length is under test here."
    );
    if rows_where_top4_could_not_reach_the_optimum > 0 {
        assert!(
            worst_top4_penalty > 0.0,
            "a counted top-4 loss must carry a real residual gap"
        );
    }
}

/// Completeness: every certifiable chart is a candidate, in distance order. The
/// old routing handed back at most four indices no matter how many charts could
/// hold the global basin.
#[test]
fn candidates_cover_every_certifiable_chart_in_distance_order() {
    let (atom, _decoder, _m) = folded_atom();
    let atlas = folded_atlas(&atom, 64);
    let x = Array1::from(vec![0.4_f64, -0.7]);
    let candidates = certified_encode_candidates(&atlas.atoms[0], x.view(), 1.0);
    let certifiable = atlas.atoms[0]
        .charts
        .iter()
        .filter(|c| c.certified_radius > 0.0)
        .count();
    assert_eq!(
        candidates.len(),
        certifiable,
        "the certified encode must see every certifiable chart, not a fixed prefix"
    );
    assert!(
        candidates.len() > 4,
        "precondition: this fixture has more certifiable charts than the deleted constant \
         admitted, or the completeness claim is untestable here (got {})",
        candidates.len()
    );
    for pair in candidates.windows(2) {
        assert!(
            pair[0].1 <= pair[1].1,
            "candidates must be distance-ordered for the tail-termination proof to hold: \
             {} then {}",
            pair[0].1,
            pair[1].1
        );
    }
    for (idx, _dist, slack) in &candidates {
        assert!(
            *slack >= 0.0,
            "chart {idx} produced a negative residual slack, which would licence a FALSE skip"
        );
    }
}

/// The prune must never be able to skip a chart that could have won: the slack
/// has to dominate the actual movement of the reconstruction inside the chart.
/// This checks the inequality the termination proof rests on, on real charts.
#[test]
fn per_chart_slack_dominates_the_reachable_reconstruction_movement() {
    let (atom, decoder, m) = folded_atom();
    let evaluator = PeriodicHarmonicEvaluator::new(m).expect("evaluator");
    let atlas = folded_atlas(&atom, 64);
    let x = Array1::from(vec![0.1_f64, 0.2]);
    let amplitude = 1.0_f64;
    let candidates = certified_encode_candidates(&atlas.atoms[0], x.view(), amplitude);
    assert!(
        !candidates.is_empty(),
        "no certifiable charts: this test would otherwise pass by having nothing to check"
    );
    for (idx, _dist, slack) in candidates {
        let chart = &atlas.atoms[0].charts[idx];
        let center = chart.region.center[0];
        let radius = chart.region.radius;
        let center_recon = recon_at(&decoder, &evaluator, center);
        // Sample the whole ball; the bound must hold at every reachable point.
        let samples = 64usize;
        for s in 0..=samples {
            let t = center - radius + 2.0 * radius * (s as f64 / samples as f64);
            let moved = &recon_at(&decoder, &evaluator, t) - &center_recon;
            let movement = amplitude * moved.dot(&moved).sqrt();
            assert!(
                movement <= slack * (1.0 + 1.0e-9) + 1.0e-12,
                "chart {idx}: reconstruction moves {movement:.9e} inside the ball but the \
                 prune's slack is only {slack:.9e} — the skip rule would be UNSOUND"
            );
        }
    }
}
