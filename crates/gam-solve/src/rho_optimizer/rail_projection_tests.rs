//! #2412: "railed" must mean the same box to the search loop and to the
//! terminal certificate.
//!
//! A smoothing coordinate whose variance component is genuinely zero drives
//! its λ toward the infinite-smoothing ceiling asymptotically: every outer step
//! shrinks the gap but none closes it, so the iterate lands strictly inside the
//! box. The rail detector calls that railed by margin; the exact `x >= upper`
//! test in `project_gradient_vector` calls it interior. While those two
//! disagreed, the cost-stall guard measured a stationarity residual made almost
//! entirely of an outward rail pull that the certificate would have discarded,
//! never reached a stationary verdict, and let the solver run to its iteration
//! cap on points the certificate goes on to accept.

use super::bridges::{
    projected_gradient_norm, rail_projected_gradient_norm, rail_relaxed_bounds,
};
use crate::model_types::CERTIFICATE_RAIL_MARGIN;
use ndarray::Array1;

/// The default outer search box, wide enough that the margin is not width-capped.
fn wide_box(n: usize) -> (Array1<f64>, Array1<f64>) {
    (
        Array1::from_elem(n, -30.0),
        Array1::from_elem(n, 30.0),
    )
}

#[test]
fn a_coordinate_creeping_onto_the_ceiling_is_railed_for_the_residual() {
    // The reported shape: ρ parked just short of the ceiling with a large
    // gradient pulling further out (λ wants to keep growing). At the upper
    // bound the outward direction is g < 0, since the step is -g.
    let bounds = wide_box(1);
    let x = Array1::from_vec(vec![29.9938]);
    let gradient = Array1::from_vec(vec![-158.8]);

    // The raw box calls it interior and keeps the whole outward pull, which is
    // what let a rail masquerade as a stationarity residual.
    let raw = projected_gradient_norm(&x, &gradient, Some(&bounds));
    assert!((raw - 158.8).abs() < 1e-9, "raw projection kept {raw}");

    // The rail-relaxed box recognizes it and drops the KKT-multiplier part.
    let railed = rail_projected_gradient_norm(&x, &gradient, Some(&bounds));
    assert_eq!(railed, 0.0, "rail-relaxed projection kept {railed}");
}

#[test]
fn the_lower_ceiling_behaves_the_same_way() {
    // Mirror case: ρ creeping onto the λ→0 floor, outward pull is g > 0.
    let bounds = wide_box(1);
    let x = Array1::from_vec(vec![-29.9938]);
    let gradient = Array1::from_vec(vec![42.5]);

    assert!((projected_gradient_norm(&x, &gradient, Some(&bounds)) - 42.5).abs() < 1e-9);
    assert_eq!(rail_projected_gradient_norm(&x, &gradient, Some(&bounds)), 0.0);
}

#[test]
fn feasible_descent_at_a_rail_is_never_discarded() {
    // The safety property that makes relaxing the box sound: only the OUTWARD
    // half is zeroed. A near-ceiling coordinate whose gradient still points
    // back INTO the box is not at an optimum, and must keep its residual under
    // both projections or the relaxation would manufacture certifications.
    let bounds = wide_box(1);
    let x = Array1::from_vec(vec![29.9938]);
    let gradient = Array1::from_vec(vec![7.25]);

    assert!((projected_gradient_norm(&x, &gradient, Some(&bounds)) - 7.25).abs() < 1e-9);
    assert!((rail_projected_gradient_norm(&x, &gradient, Some(&bounds)) - 7.25).abs() < 1e-9);
}

#[test]
fn interior_coordinates_are_untouched_by_the_relaxation() {
    // Well inside the box, the two projections must agree exactly — the
    // relaxation may not perturb ordinary interior optimization.
    let bounds = wide_box(3);
    let x = Array1::from_vec(vec![0.0, -4.5, 11.25]);
    let gradient = Array1::from_vec(vec![1.5, -2.5, 0.75]);

    let raw = projected_gradient_norm(&x, &gradient, Some(&bounds));
    let railed = rail_projected_gradient_norm(&x, &gradient, Some(&bounds));
    assert!((raw - railed).abs() < 1e-12, "raw {raw} vs railed {railed}");
}

#[test]
fn a_railed_coordinate_does_not_mask_an_interior_one() {
    // The mixed case the guard actually sees: one coordinate on the ceiling,
    // one genuinely far from stationary. The rail is removed; the interior
    // residual survives in full, so the guard still refuses to certify.
    let bounds = wide_box(2);
    let x = Array1::from_vec(vec![29.9938, 0.0]);
    let gradient = Array1::from_vec(vec![-158.8, 3.0]);

    assert!((rail_projected_gradient_norm(&x, &gradient, Some(&bounds)) - 3.0).abs() < 1e-9);
}

#[test]
fn unbounded_problems_are_unaffected() {
    let x = Array1::from_vec(vec![1.0, -2.0]);
    let gradient = Array1::from_vec(vec![3.0, -4.0]);
    let expected = 5.0_f64;
    assert!((rail_projected_gradient_norm(&x, &gradient, None) - expected).abs() < 1e-12);
}

#[test]
fn the_margin_is_the_certificate_margin_on_a_wide_box() {
    let (lower, upper) = rail_relaxed_bounds(&wide_box(1));
    assert!((lower[0] - (-30.0 + CERTIFICATE_RAIL_MARGIN)).abs() < 1e-12);
    assert!((upper[0] - (30.0 - CERTIFICATE_RAIL_MARGIN)).abs() < 1e-12);
}

#[test]
fn a_narrow_box_is_relaxed_without_inverting() {
    // A box narrower than twice the margin would invert under a flat
    // relaxation, making every point read as railed at BOTH ends and zeroing
    // the coordinate's residual outright — a silent false certification on
    // exactly the tightly-boxed coordinates that most need a real one. The
    // margin is capped at a quarter width, so the relaxed interval keeps at
    // least half the original and the endpoints stay ordered.
    let bounds = (
        Array1::from_vec(vec![-0.4]),
        Array1::from_vec(vec![0.4]),
    );
    let (lower, upper) = rail_relaxed_bounds(&bounds);
    assert!(lower[0] < upper[0], "relaxed box inverted: {lower:?} {upper:?}");
    assert!((upper[0] - lower[0]) >= 0.5 * 0.8 - 1e-12);

    // A point at the centre of a narrow box is not railed, so a real residual
    // there still counts.
    let x = Array1::from_vec(vec![0.0]);
    let gradient = Array1::from_vec(vec![2.0]);
    assert!((rail_projected_gradient_norm(&x, &gradient, Some(&bounds)) - 2.0).abs() < 1e-12);
}

#[test]
fn a_fixed_coordinate_gets_no_relaxation() {
    // `lower == upper` pins the coordinate; there is no interior to relax into,
    // and the exact bound test already handles it.
    let bounds = (
        Array1::from_vec(vec![2.0]),
        Array1::from_vec(vec![2.0]),
    );
    let (lower, upper) = rail_relaxed_bounds(&bounds);
    assert_eq!(lower[0], 2.0);
    assert_eq!(upper[0], 2.0);
}
