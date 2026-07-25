//! #2433 diagnostic: the collection design and the frozen single-term replay
//! must agree on how many penalty blocks a Duchon smooth owns.
//!
//! The κ (spatial length-scale) optimizer caches ONE authoritative
//! `TermCollectionDesign` and then, on every trial κ, re-realizes the single
//! spatial term through `build_single_local_smooth_term` driven from the spec
//! frozen out of that design. `replace_term_realization` hard-refuses if the
//! replay's penalty count differs from the cached range, which is the
//! `active_penalties=5, cached_penalties=4` refusal in the issue.
//!
//! This file is a probe, not (yet) a gate: it prints the two topologies so the
//! disagreement can be localised without running a full fit.

use gam::basis::{
    CenterStrategy, DuchonBasisSpec, DuchonNullspaceOrder, DuchonOperatorPenaltySpec,
    OneDimensionalBoundary, SpatialIdentifiability,
};
use gam::smooth::{
    ShapeConstraint, SmoothBasisSpec, SmoothTermSpec, TermCollectionSpec,
    build_term_collection_design, freeze_term_collection_from_design,
};
use ndarray::Array2;

fn simulate_1d(n: usize) -> Array2<f64> {
    let mut x = Array2::<f64>::zeros((n, 1));
    for i in 0..n {
        x[[i, 0]] = (i as f64) / (n as f64 - 1.0) * 6.0 - 3.0;
    }
    x
}

fn spec_1d() -> TermCollectionSpec {
    TermCollectionSpec {
        linear_terms: vec![],
        random_effect_terms: vec![],
        smooth_terms: vec![SmoothTermSpec {
            name: "duchon_1d".to_string(),
            basis: SmoothBasisSpec::Duchon {
                feature_cols: vec![0],
                spec: DuchonBasisSpec {
                    radial_reparam: None,
                    center_strategy: CenterStrategy::FarthestPoint { num_centers: 12 },
                    periodic: None,
                    length_scale: Some(1.0),
                    power: 1.0,
                    nullspace_order: DuchonNullspaceOrder::Linear,
                    identifiability: SpatialIdentifiability::default(),
                    aniso_log_scales: None,
                    operator_penalties: DuchonOperatorPenaltySpec::all_active(),
                    boundary: OneDimensionalBoundary::Open,
                },
                input_scale: None,
            },
            shape: ShapeConstraint::None,
            joint_null_rotation: None,
        }],
    }
}

#[test]
fn duchon_frozen_replay_matches_collection_penalty_topology_2433() {
    let data = simulate_1d(1000);
    let spec = spec_1d();

    let design = build_term_collection_design(data.view(), &spec).expect("collection design");
    let term = &design.smooth.terms[0];
    let collection_sources: Vec<String> = term
        .active_penalties
        .iter()
        .map(|p| format!("{:?}", p.info.source))
        .collect();
    let collection_dropped: Vec<String> = term
        .dropped_penalties
        .iter()
        .map(|p| format!("{:?}/{:?}", p.source, p.reason))
        .collect();
    eprintln!(
        "collection: p={} active={} {:?} dropped={:?}",
        term.coeff_range.len(),
        term.active_penalties.len(),
        collection_sources,
        collection_dropped
    );

    let frozen = freeze_term_collection_from_design(&spec, &design).expect("freeze");
    let mut workspace = gam_terms::basis::BasisWorkspace::default();
    let local = gam_terms::smooth::build_single_local_smooth_term(
        data.view(),
        &frozen.smooth_terms[0],
        &mut workspace,
    )
    .expect("frozen single-term replay");
    let replay_sources: Vec<String> = local
        .active_penalties
        .iter()
        .map(|p| format!("{:?}", p.info.source))
        .collect();
    let replay_dropped: Vec<String> = local
        .dropped_penalties
        .iter()
        .map(|p| format!("{:?}/{:?}", p.source, p.reason))
        .collect();
    eprintln!(
        "replay:     p={} active={} {:?} dropped={:?}",
        local.dim,
        local.active_penalties.len(),
        replay_sources,
        replay_dropped
    );

    for (label, penalties) in [
        ("collection", &term.active_penalties),
        ("replay", &local.active_penalties),
    ] {
        for p in penalties.iter() {
            eprintln!(
                "  {label} {:?} rank={} nullity={} norm_scale={:.6e} dim={}",
                p.info.source,
                p.info.effective_rank,
                p.nullity,
                p.info.normalization_scale,
                p.matrix.nrows()
            );
            if matches!(p.info.source, gam::basis::PenaltySource::Primary) {
                let (evals, _) = gam_linalg::faer_ndarray::FaerEigh::eigh(&p.matrix, faer::Side::Lower)
                    .expect("eigh");
                let lam_max = evals.iter().copied().fold(0.0_f64, |a, v| a.max(v.abs()));
                let rel: Vec<String> = evals
                    .iter()
                    .map(|v| format!("{:.3e}", v / lam_max))
                    .collect();
                eprintln!(
                    "    {label} primary rel-spectrum (tol={:.3e}): {}",
                    (p.matrix.nrows() as f64) * 1e-10,
                    rel.join(" ")
                );
            }
        }
    }

    assert_eq!(
        local.active_penalties.len(),
        term.active_penalties.len(),
        "frozen single-term replay must reproduce the collection's penalty topology \
         (collection={collection_sources:?}, replay={replay_sources:?})"
    );
    assert_eq!(
        local.dim,
        term.coeff_range.len(),
        "frozen single-term replay must reproduce the collection's coefficient width"
    );
}
