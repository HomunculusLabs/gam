//! TEMPORARY gam#2695 probe — how smooth is the monotone warp basis, order by
//! order, at an INTERIOR knot and at the knot-hull edge, as a function of the
//! public degree?
//!
//! The composed warp `q = q₀ + Σ_j βw_j·I_j(q₀)` is evaluated at a β-dependent
//! index, and the inner objective differentiates it three times:
//!
//! ```text
//!   ℓ            reads  w′        (the event Jacobian q̇ = (1+w′)·r)
//!   ∇ℓ           reads  w″
//!   H            reads  w‴        (and Φ = ½Σg(λ(Z_JᵀHZ_J)) is IN the objective)
//!   ∇Φ = ∂H/∂β   reads  w⁗
//! ```
//!
//! so the first order at which `I_j` steps says which of those four objects is
//! the first to be discontinuous. This prints it; it asserts nothing.

use gam_models::wiggle::monotone_wiggle_basis_with_derivative_order;
use ndarray::Array1;

fn clamped_knots(degree: usize, left: f64, right: f64, internal: usize) -> Array1<f64> {
    let mut knots = Vec::with_capacity(2 * (degree + 1) + internal);
    for _ in 0..=degree {
        knots.push(left);
    }
    for i in 1..=internal {
        knots.push(left + (right - left) * (i as f64) / ((internal + 1) as f64));
    }
    for _ in 0..=degree {
        knots.push(right);
    }
    Array1::from_vec(knots)
}

fn row_at(x: f64, knots: &Array1<f64>, degree: usize, order: usize) -> Vec<f64> {
    let seed = Array1::from_elem(1, x);
    monotone_wiggle_basis_with_derivative_order(seed.view(), knots, degree, order)
        .expect("warp basis")
        .row(0)
        .to_vec()
}

/// Largest one-sided gap in `I^{(order)}` across `x`, measured at `±h`.
fn gap(x: f64, h: f64, knots: &Array1<f64>, degree: usize, order: usize) -> f64 {
    let lo = row_at(x - h, knots, degree, order);
    let hi = row_at(x + h, knots, degree, order);
    lo.iter()
        .zip(hi.iter())
        .fold(0.0_f64, |acc, (a, b)| acc.max((a - b).abs()))
}

#[test]
fn probe_2695_warp_basis_smoothness_ladder() {
    for degree in 2..=5usize {
        let left = -1.0;
        let right = 2.0;
        let internal = 2usize;
        let knots = clamped_knots(degree, left, right, internal);
        let interior = left + (right - left) / ((internal + 1) as f64);
        for (label, x) in [
            ("interior-knot", interior),
            ("hull-right", right),
            ("hull-left", left),
            ("smooth-point", 0.5 * (left + interior)),
        ] {
            let mut line = format!("[2695-SMOOTH] degree={degree} at={label} x={x:+.4}");
            for order in 0..=4usize {
                // The gap must be read against a SHRINKING h: a genuine step
                // holds its size, a kink halves with h, a continuous order
                // falls faster.
                let coarse = gap(x, 1.0e-3, &knots, degree, order);
                let fine = gap(x, 1.0e-6, &knots, degree, order);
                let ratio = if fine > 0.0 { coarse / fine } else { f64::INFINITY };
                line.push_str(&format!(
                    " | d{order}: {coarse:.3e}->{fine:.3e} (x{ratio:.1e})"
                ));
            }
            println!("{line}");
        }
    }
}
