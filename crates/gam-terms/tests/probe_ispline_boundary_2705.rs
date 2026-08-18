//! PROBE (#2705): what happens EXACTLY AT the boundary knots of an I-spline
//! value/derivative pair, where the evaluator switches branch.
//!
//! Temporary instrument. Prints; the permanent gates live in
//! `basis::ispline_boundary::tests_ispline_boundary`.

use gam_terms::basis::{ISplineBoundary, ispline_value_and_first_derivative};
use ndarray::Array1;

fn clamped_knots(lo: f64, hi: f64, internal: usize) -> Array1<f64> {
    let mut knots: Vec<f64> = vec![lo; 5];
    for k in 1..=internal {
        knots.push(lo + (hi - lo) * (k as f64) / ((internal + 1) as f64));
    }
    knots.extend(std::iter::repeat_n(hi, 5));
    Array1::from_vec(knots)
}

#[test]
fn probe_ispline_pair_at_the_boundary_knots_2705() {
    let degree = 3usize;
    let (lo, hi) = (0.0_f64, 1.0_f64);
    let knots = clamped_knots(lo, hi, 3);
    let eps = 1.0e-9_f64;
    let points = Array1::from_vec(vec![hi - eps, hi, hi + eps, lo - eps, lo, lo + eps]);
    let labels = ["hi-eps", "hi", "hi+eps", "lo-eps", "lo", "lo+eps"];
    for boundary in [ISplineBoundary::Saturate, ISplineBoundary::LinearTails] {
        let (value, derivative) =
            ispline_value_and_first_derivative(points.view(), knots.view(), degree, boundary)
                .expect("pair");
        println!("\n=== {boundary:?}");
        for (row, label) in labels.iter().enumerate() {
            let v: Vec<String> = value.row(row).iter().map(|z| format!("{z:+.17e}")).collect();
            println!("{label:>8} value [{}]", v.join(", "));
        }
        for (row, label) in labels.iter().enumerate() {
            let d: Vec<String> = derivative
                .row(row)
                .iter()
                .map(|z| format!("{z:+.9e}"))
                .collect();
            println!("{label:>8} deriv [{}]", d.join(", "));
        }
        // One-sided jumps at the boundary, per column.
        println!(" column   value(hi)-value(hi-eps)   value(hi+eps)-value(hi)   deriv jump");
        for column in 0..value.ncols() {
            println!(
                "{column:>7}   {:+.6e}   {:+.6e}   {:+.6e} -> {:+.6e} -> {:+.6e}",
                value[[1, column]] - value[[0, column]],
                value[[2, column]] - value[[1, column]],
                derivative[[0, column]],
                derivative[[1, column]],
                derivative[[2, column]],
            );
        }
    }
}
