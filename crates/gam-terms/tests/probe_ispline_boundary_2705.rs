//! PROBE (#2705, group I / royston_parmar): does the I-spline VALUE basis agree
//! with the derivative basis the survival RP time block builds from it, outside
//! the knot range?
//!
//! Temporary instrument. Prints; asserts nothing that would gate CI.

use gam_terms::basis::{BasisOptions, Dense, KnotSource, create_basis};
use ndarray::{Array1, ArrayView1};

fn ispline_values(x: &Array1<f64>, knots: &Array1<f64>, degree: usize) -> ndarray::Array2<f64> {
    let (arc, _) = create_basis::<Dense>(
        x.view(),
        KnotSource::Provided(knots.view()),
        degree,
        BasisOptions::i_spline(),
    )
    .expect("ispline value basis");
    arc.as_ref().clone()
}

/// The exact cumulative-sum the survival RP builder performs by hand.
fn rp_style_derivative(
    x: &Array1<f64>,
    knots: &Array1<f64>,
    degree: usize,
    p_full: usize,
) -> ndarray::Array2<f64> {
    let bspline_degree = degree + 1;
    let (db_arc, _) = create_basis::<Dense>(
        x.view(),
        KnotSource::Provided(knots.view()),
        bspline_degree,
        BasisOptions::first_derivative(),
    )
    .expect("ispline derivative basis");
    let db = db_arc.as_ref();
    let mut out = ndarray::Array2::<f64>::zeros((x.len(), p_full));
    for i in 0..x.len() {
        let mut running = 0.0_f64;
        for j in (1..db.ncols()).rev() {
            let term = db[[i, j]];
            if term.is_finite() {
                running += term;
            }
            out[[i, j - 1]] = running;
        }
    }
    out
}

fn clamped_knots(lo: f64, hi: f64, internal: usize, bspline_degree: usize) -> Array1<f64> {
    let mut v: Vec<f64> = Vec::new();
    for _ in 0..=bspline_degree {
        v.push(lo);
    }
    for k in 1..=internal {
        v.push(lo + (hi - lo) * (k as f64) / ((internal + 1) as f64));
    }
    for _ in 0..=bspline_degree {
        v.push(hi);
    }
    Array1::from_vec(v)
}

#[test]
fn probe_ispline_value_and_rp_derivative_outside_the_knot_range_2705() {
    let degree = 3usize;
    let bspline_degree = degree + 1;
    let lo = 0.0_f64;
    let hi = 1.0_f64;
    let knots = clamped_knots(lo, hi, 3, bspline_degree);
    println!("knots = {:?}", knots.to_vec());

    let p_full = knots.len() - bspline_degree - 1 - 1;
    println!("p_full (ispline columns) = {p_full}");

    let xs = Array1::from_vec(vec![
        -0.30, -0.10, 0.0, 0.25, 0.5, 0.75, 1.0, 1.10, 1.30, 2.0,
    ]);
    let values = ispline_values(&xs, &knots, degree);
    let deriv = rp_style_derivative(&xs, &knots, degree, p_full);

    println!("\n x        value row                                   deriv row");
    for i in 0..xs.len() {
        let v: Vec<String> = values.row(i).iter().map(|z| format!("{z:+.5}")).collect();
        let d: Vec<String> = deriv.row(i).iter().map(|z| format!("{z:+.5}")).collect();
        println!("{:+.3}  [{}]  [{}]", xs[i], v.join(","), d.join(","));
    }

    // Finite difference of the VALUE basis against the analytic derivative.
    let h = 1e-6_f64;
    println!("\n x        max|fd(value) - analytic derivative|");
    for i in 0..xs.len() {
        let x = xs[i];
        let plus = ispline_values(&Array1::from_vec(vec![x + h]), &knots, degree);
        let minus = ispline_values(&Array1::from_vec(vec![x - h]), &knots, degree);
        let mut worst = 0.0_f64;
        for j in 0..p_full {
            let fd = (plus[[0, j]] - minus[[0, j]]) / (2.0 * h);
            worst = worst.max((fd - deriv[[i, j]]).abs());
        }
        println!("{:+.3}  {:.6e}", x, worst);
    }

    // What the tree's own shared I-spline derivative says (the #2695 route).
    let shared = gam_terms::basis::create_ispline_derivative_dense(
        xs.view() as ArrayView1<'_, f64>,
        &knots,
        degree,
        1,
    )
    .expect("shared ispline derivative");
    println!("\n x        shared create_ispline_derivative_dense row");
    for i in 0..xs.len() {
        let d: Vec<String> = shared.row(i).iter().map(|z| format!("{z:+.5}")).collect();
        println!("{:+.3}  [{}]", xs[i], d.join(","));
    }
}
