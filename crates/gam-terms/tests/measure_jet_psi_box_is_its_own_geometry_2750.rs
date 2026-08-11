//! gam#2750: the `ln ℓ` search window of a measure-jet term is a statement
//! about the node cloud, so it must MOVE WITH the chart.
//!
//! `ℓ` is a length in the frame the basis is realized in. The window it is
//! searched over was an absolute interval (`ln ℓ ∈ ln[1e-3, 1e2]`), chosen so
//! the bound producer would not need a data view. Two things follow from that
//! choice and both are defects, and this file pins the repair of each from a
//! different side:
//!
//! 1. **It is not equivariant.** Rescale the chart by `c` and every length in
//!    it — the node spacing, the node diameter, the fitted `ℓ`, the seed
//!    `ln ℓ` — moves by `ln c`, while an absolute window does not move at all.
//!    So the same configuration measured in metres and in millimetres gets two
//!    different search problems, and at `c = 10³` the seed sits `6.9` log units
//!    from where it sat before inside a window only `11.5` wide.
//! 2. **It is 2.8×–4.9× wider than the coordinate's whole physically distinct
//!    range.** A trust-region method scales its first step to the box it is
//!    handed; the measured first `ln ℓ` step on `measure_jet_perf_parity` was
//!    `−0.693`, landing at `ℓ = 0.488` against a node-spacing floor of
//!    `0.5145` — outside the term's own geometry — and it was rejected twelve
//!    times, every rejection a full design realization.
//!
//! The window is now the term's own two walls, both measured lengths and
//! neither chosen: the median nearest-node spacing (below it neighbouring
//! representers stop overlapping and the design is a bump-per-node indicator)
//! and `spacing/√(2√ε)` (above it the closest pair is not distinguishable from
//! a coincident one in `f64`, so no distinct model survives).
//!
//! The upper wall is deliberately NOT the node bounding-box diameter. That was
//! the screen's walk stop, and #2761 measured it cutting the walk off with the
//! criterion still descending — on a term whose `ℓ` dial is frozen there is no
//! second search to continue past a stopping rule, so the rule became the wall.
//! Both now read `measure_jet_range_feasibility_ceiling`, and
//! `the_screen_walk_and_the_search_window_stop_at_the_same_wall_2761` pins that
//! they agree exactly rather than approximately.

use gam_terms::basis::{
    CenterStrategy, MeasureJetBasisSpec, measure_jet_ln_range_window, measure_jet_range_bracket,
};
use ndarray::Array2;

/// A 1-D scatter with a deterministic irregular spacing, so the median nearest
/// node spacing is a real median rather than a constant grid step.
fn chart(scale: f64) -> Array2<f64> {
    Array2::from_shape_fn((240, 1), |(i, _)| {
        let t = i as f64 / 239.0;
        scale * (t + 0.04 * (7.0 * t).sin())
    })
}

fn spec(centers: usize) -> MeasureJetBasisSpec {
    MeasureJetBasisSpec {
        center_strategy: CenterStrategy::FarthestPoint {
            num_centers: centers,
        },
        ..MeasureJetBasisSpec::default()
    }
}

/// `ln(ceiling/floor) = -0.5*ln(2*sqrt(EPSILON))`, the window's width. It is a
/// pure function of `f64::EPSILON` — both ends are the same measured length, so
/// the geometry cancels out of their ratio — and it is restated here rather than
/// imported so a change to the derivation has to face a number.
const LN_WINDOW_WIDTH: f64 = 8.664_339_756_999_317;

#[test]
fn ln_range_window_floor_is_the_bracket_floor_and_its_width_is_pure_precision() {
    let data = chart(1.0);
    let spec = spec(40);
    let bracket = measure_jet_range_bracket(data.view(), &spec).expect("bracket realizes");
    let (lo, hi) = measure_jet_ln_range_window(data.view(), &spec).expect("window realizes");
    assert_eq!(
        lo,
        bracket.nodes[0].ln(),
        "the window floor must BE the bracket's floor node (the median nearest-node spacing), \
         not a second derivation of it"
    );
    assert!(
        (hi - lo - LN_WINDOW_WIDTH).abs() <= 1.0e-12,
        "the width is the ratio of two multiples of the SAME measured spacing, so it carries no \
         geometry at all: got {}, expected {LN_WINDOW_WIDTH}",
        hi - lo
    );
}

#[test]
fn the_screen_walk_and_the_search_window_stop_at_the_same_wall_2761() {
    // They did not, and that was a defect rather than a design (#2761). The
    // walk used to stop at `MeasureJetRangeBracket::node_diameter`, on the
    // argument that at a range that long every pair of representers overlaps at
    // `>= exp(-1/2)` so no distinct model survives. Two places in the tree
    // already recorded the opposite -- `measure_jet_ln_range_window`'s own docs
    // ("the profiled criterion genuinely prefers a range AT or ABOVE the node
    // diameter", measured on three fixtures) and the earlier version of this
    // test, which asserted the search window is strictly wider and called the
    // diameter "a stopping rule for the screen's walk over NODES, not a wall in
    // the model".
    //
    // That reconciliation holds only while something else keeps searching past
    // the stopping rule. On a term whose ell dial is FROZEN -- the BMS
    // marginal/log-slope pair, or any `learn_length_scale=false` -- nothing
    // does, and the stopping rule becomes the wall. So there is one wall now,
    // read from one definition, and this pins that they agree exactly.
    let data = chart(1.0);
    let spec = spec(40);
    let bracket = measure_jet_range_bracket(data.view(), &spec).expect("bracket realizes");
    let (lo, hi) = measure_jet_ln_range_window(data.view(), &spec).expect("window realizes");
    assert_eq!(
        bracket.feasibility_ceiling.ln(),
        hi,
        "the screen's walk stop and the outer search's window ceiling must be the SAME \
         number, not two derivations of the same idea: walk={} window={}",
        bracket.feasibility_ceiling,
        hi.exp()
    );
    // And the diameter, which is still reported as the geometric fact it is,
    // must sit strictly inside that wall -- otherwise the old stop was not a
    // tightening and this change is not the one described.
    assert!(
        bracket.node_diameter.ln() < hi && lo < bracket.node_diameter.ln(),
        "the node diameter must sit strictly inside the window it used to cap: diameter={} \
         window=[{}, {}]",
        bracket.node_diameter,
        lo.exp(),
        hi.exp()
    );
}

#[test]
fn ln_range_window_is_equivariant_under_an_isotropic_chart_rescale() {
    // Every length in the chart scales by `c`; the window is made of lengths,
    // so both ends must shift by exactly `ln c` and the WIDTH must not move.
    // An absolute window fails both halves by construction.
    let base = measure_jet_ln_range_window(chart(1.0).view(), &spec(40)).expect("base window");
    for scale in [1.0e-3, 0.25, 4.0, 1.0e3] {
        let moved =
            measure_jet_ln_range_window(chart(scale).view(), &spec(40)).expect("rescaled window");
        let shift = scale.ln();
        let tolerance = 1.0e-12 * (1.0 + shift.abs());
        assert!(
            (moved.0 - (base.0 + shift)).abs() <= tolerance,
            "floor must shift by ln(c): c={scale}, base={}, moved={}, expected={}",
            base.0,
            moved.0,
            base.0 + shift
        );
        assert!(
            (moved.1 - (base.1 + shift)).abs() <= tolerance,
            "ceiling must shift by ln(c): c={scale}, base={}, moved={}, expected={}",
            base.1,
            moved.1,
            base.1 + shift
        );
        assert!(
            ((moved.1 - moved.0) - (base.1 - base.0)).abs() <= 1.0e-12,
            "the window WIDTH is a dimensionless property of the node layout and may not \
             move with the chart's units: c={scale}, base width={}, moved width={}",
            base.1 - base.0,
            moved.1 - moved.0
        );
    }
}

#[test]
fn ln_range_window_brackets_the_auto_range_it_seeds() {
    // The auto range IS the band floor (`MEASURE_JET_AUTO_LENGTH_SCALE_FACTOR`
    // is 1 and both read the same median nearest-node spacing), so the seed a
    // fresh unscreened term starts at must be the window's own lower end — the
    // search may lengthen the range and has nothing distinct to reach below it.
    let data = chart(1.0);
    let spec = spec(40);
    let bracket = measure_jet_range_bracket(data.view(), &spec).expect("bracket realizes");
    let (lo, hi) = measure_jet_ln_range_window(data.view(), &spec).expect("window realizes");
    let nodes = gam_terms::basis::measure_jet_quadrature_nodes(
        data.view(),
        gam_terms::basis::select_centers_by_strategy(data.view(), &spec.center_strategy)
            .expect("centers")
            .view(),
    )
    .expect("quadrature nodes")
    .0;
    let auto =
        gam_terms::basis::realized_measure_jet_length_scale(nodes.view(), 0.0).expect("auto range");
    assert!(
        (auto.ln() - lo).abs() <= 1.0e-12,
        "the auto range {auto} must be the window floor {}, because both are the median \
         nearest-node spacing",
        lo.exp()
    );
    assert!(
        auto.ln() < hi,
        "the auto range must sit strictly inside the window: auto={auto}, ceiling={}",
        hi.exp()
    );
    assert!(
        bracket.feasibility_ceiling > bracket.nodes[bracket.nodes.len() - 1],
        "the walk ceiling is above the band's top node, so the screen may still walk past every \
         node it scored"
    );
}
