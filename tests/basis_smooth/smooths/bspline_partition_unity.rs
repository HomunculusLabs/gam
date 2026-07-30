//! B-spline partition of unity across degrees 1..=5.
//!
//! The five degrees previously lived in one file each
//! (`bspline_partition_unity_degree_{1..5}.rs`), which duplicated the two knot
//! builders and the check loop verbatim five times. The per-degree `#[test]`
//! functions are kept — the degrees are independent claims and each should fail
//! on its own — but they now share one implementation.
//!
//! Seeds are unchanged: every degree drew from `StdRng::seed_from_u64(0xBAD5_0000
//! + degree)`, so each case still evaluates the same knots at the same abscissae
//! it did as a standalone file.

use gam::terms::basis::{SplineScratch, evaluate_bspline_basis_scalar};
use ndarray::Array1;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

fn clamped_uniform_knots(degree: usize, num_basis: usize) -> Array1<f64> {
    let interior_count = num_basis.saturating_sub(degree + 1);
    let mut knots = Vec::with_capacity(num_basis + degree + 1);
    knots.extend(std::iter::repeat_n(0.0, degree + 1));
    for j in 1..=interior_count {
        knots.push(j as f64 / (interior_count as f64 + 1.0));
    }
    knots.extend(std::iter::repeat_n(1.0, degree + 1));
    Array1::from(knots)
}

fn clamped_random_knots(rng: &mut StdRng, degree: usize, num_basis: usize) -> Array1<f64> {
    let interior_count = num_basis.saturating_sub(degree + 1);
    let mut interior: Vec<f64> = (0..interior_count).map(|_| rng.random::<f64>()).collect();
    interior.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut knots = Vec::with_capacity(num_basis + degree + 1);
    knots.extend(std::iter::repeat_n(0.0, degree + 1));
    knots.extend(interior);
    knots.extend(std::iter::repeat_n(1.0, degree + 1));
    Array1::from(knots)
}

/// On a clamped knot vector the basis functions of any degree sum to exactly 1
/// everywhere on the interior span. Checked on both a uniform and a random
/// clamped knot vector, at 1000 random abscissae each.
fn assert_partition_of_unity(degree: usize) {
    let num_basis = degree + 8;
    let mut rng = StdRng::seed_from_u64(0xBAD5_0000 + degree as u64);

    for knots in [
        clamped_uniform_knots(degree, num_basis),
        clamped_random_knots(&mut rng, degree, num_basis),
    ] {
        let low = knots[degree];
        let high = knots[knots.len() - degree - 1];
        let mut out = vec![0.0; num_basis];
        let mut scratch = SplineScratch::new(degree);
        for _ in 0..1000 {
            let x = rng.random_range(low..high);
            evaluate_bspline_basis_scalar(x, knots.view(), degree, &mut out, &mut scratch)
                .expect("B-spline evaluation should succeed");
            let sum: f64 = out.iter().sum();
            assert!(
                (sum - 1.0).abs() <= 1e-12,
                "degree={degree}, x={x:.17e}, sum={sum:.17e}, knots={knots:?}"
            );
        }
    }
}

#[test]
fn bspline_partition_of_unity_degree_1_random_clamped_and_uniform_knots() {
    assert_partition_of_unity(1);
}

#[test]
fn bspline_partition_of_unity_degree_2_random_clamped_and_uniform_knots() {
    assert_partition_of_unity(2);
}

#[test]
fn bspline_partition_of_unity_degree_3_random_clamped_and_uniform_knots() {
    assert_partition_of_unity(3);
}

#[test]
fn bspline_partition_of_unity_degree_4_random_clamped_and_uniform_knots() {
    assert_partition_of_unity(4);
}

#[test]
fn bspline_partition_of_unity_degree_5_random_clamped_and_uniform_knots() {
    assert_partition_of_unity(5);
}
