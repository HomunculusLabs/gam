#![cfg(test)]
//! #2933 F24 — the ARD coordinate prior is normalized over each coordinate's
//! actual support.
//!
//! Before this repair every non-periodic axis carried the full-line Gaussian
//! normalizer `−½ log α`, including the three ambient axes of an embedded unit
//! sphere and a bounded interval axis. On the sphere `‖x‖ = 1`, so an isotropic
//! precision gives the uniform density and its score in `log α` is zero. The line
//! normalizers scored `−1` per row at `α = 1` and manufactured an optimum at
//! `α = 3`. On `[−1, 1]` at `α = 1` the truncated partition's log-precision
//! derivative is `−0.145563`, not `−½`.
//!
//! Every reference below is independent of the production kernels: exact
//! identities of the sphere, an independent product quadrature of the Bingham
//! law, or the closed form of the symmetric interval mass.

use super::*;
use gam_math::probability::normal_cdf;
use gam_math::special::{bessel_i0_centered_terms_from_log_abs, gauss_legendre};
use ndarray::array;
use std::f64::consts::TAU;

/// A one-atom term whose ARD energy reads only the coordinates and their
/// manifold; the decoder is inert.
fn ard_term(manifold: LatentManifold, coords: Array2<f64>) -> SaeManifoldTerm {
    let (n, d) = coords.dim();
    let (m, p) = (2_usize, 3_usize);
    let atom = SaeManifoldAtom::new_with_provided_function_gram(
        "ard",
        SaeAtomBasisKind::EuclideanPatch,
        d,
        Array2::<f64>::ones((n, m)),
        Array3::<f64>::zeros((n, m, d)),
        Array2::<f64>::zeros((m, p)),
        Array2::<f64>::eye(m),
    )
    .expect("atom fixture: basis, jet, decoder and Gram shapes agree by construction");
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        Array2::<f64>::zeros((n, 1)),
        vec![coords],
        vec![manifold],
        AssignmentMode::softmax(1.0),
    )
    .expect("assignment fixture: one coordinate block per manifold");
    SaeManifoldTerm::new(vec![atom], assignment).expect("term fixture: one atom, one block")
}

fn rho_with_log_ard(log_ard: Array1<f64>) -> SaeManifoldRho {
    SaeManifoldRho::new(0.0, 0.0, vec![log_ard])
}

fn sphere_rows() -> Array2<f64> {
    let raw = [
        [0.2, -0.9, 0.4],
        [0.7, 0.1, -0.7],
        [-0.3, 0.5, 0.8],
        [0.95, 0.2, 0.1],
        [-0.6, -0.6, 0.5],
    ];
    let mut out = Array2::<f64>::zeros((raw.len(), 3));
    for (i, row) in raw.iter().enumerate() {
        let norm = row.iter().map(|v| v * v).sum::<f64>().sqrt();
        for a in 0..3 {
            out[[i, a]] = row[a] / norm;
        }
    }
    out
}

/// `Σ_rows ½ α_a x_a²` per axis: the energy channel of `∂ ard_value/∂log α_a`.
fn energy_per_axis(coords: &Array2<f64>, alpha: &[f64]) -> Vec<f64> {
    (0..coords.ncols())
        .map(|a| coords.column(a).iter().map(|x| 0.5 * alpha[a] * x * x).sum())
        .collect()
}

/// `E[x_a²]` under the density `∝ exp(−Σ_a ½ α_a x_a²)` on `S²`, by Gauss–Legendre
/// in `x₃` and the periodic trapezoid in the azimuth.
fn bingham_second_moments_by_product_rule(alpha: [f64; 3]) -> [f64; 3] {
    let (nodes, weights) = gauss_legendre(160);
    let azimuths = 512;
    let mut mass = 0.0;
    let mut moments = [0.0_f64; 3];
    for (&z, &wz) in nodes.iter().zip(weights.iter()) {
        let r = (1.0 - z * z).sqrt();
        for j in 0..azimuths {
            let phi = TAU * j as f64 / azimuths as f64;
            let x = [r * phi.cos(), r * phi.sin(), z];
            let energy: f64 = (0..3).map(|a| 0.5 * alpha[a] * x[a] * x[a]).sum();
            let w = wz * (-energy).exp();
            mass += w;
            for a in 0..3 {
                moments[a] += w * x[a] * x[a];
            }
        }
    }
    moments.map(|m| m / mass)
}

#[test]
fn sphere_isotropic_ard_precision_carries_no_score_2933_f24() {
    let term = ard_term(LatentManifold::Sphere { dim: 3 }, sphere_rows());
    let n = sphere_rows().nrows() as f64;
    for log_alpha in [0.0, 3.0_f64.ln(), -2.0, 4.0] {
        let rho = rho_with_log_ard(array![log_alpha, log_alpha, log_alpha]);
        let derivatives = term
            .ard_log_precision_explicit_derivatives(&rho)
            .expect("ARD derivatives on a validated sphere fixture");
        let isotropic_score: f64 = derivatives[0].sum();
        let line_normalizer_score = 0.5 * log_alpha.exp() * n - 1.5 * n;
        assert!(
            isotropic_score.abs() < 1e-11 * n,
            "an isotropic precision on the unit sphere is the uniform law, so its log-precision \
             score must vanish: log α={log_alpha}, score {isotropic_score} (three line \
             normalizers give {line_normalizer_score})"
        );
    }
    let value_at_one = term
        .ard_value(&rho_with_log_ard(array![0.0, 0.0, 0.0]))
        .expect("ARD value at α = 1");
    let ln3 = 3.0_f64.ln();
    let value_at_three = term
        .ard_value(&rho_with_log_ard(array![ln3, ln3, ln3]))
        .expect("ARD value at α = 3");
    assert!(
        (value_at_three - value_at_one).abs() < 1e-10 * n,
        "the isotropic sphere prior must not depend on α: value {value_at_one} at α=1, \
         {value_at_three} at α=3"
    );
}

#[test]
fn sphere_ard_trace_shift_leaves_the_normalized_prior_unchanged_2933_f24() {
    let coords = sphere_rows();
    let term = ard_term(LatentManifold::Sphere { dim: 3 }, coords.clone());
    let n = coords.nrows() as f64;
    let alpha = [2.5, 0.7, 1.3];
    let log_ard = |shift: f64| {
        array![
            (alpha[0] + shift).ln(),
            (alpha[1] + shift).ln(),
            (alpha[2] + shift).ln()
        ]
    };
    let base = term
        .ard_value(&rho_with_log_ard(log_ard(0.0)))
        .expect("ARD value at the anisotropic precision");
    for shift in [1.9, -0.5, 40.0] {
        let shifted = term
            .ard_value(&rho_with_log_ard(log_ard(shift)))
            .expect("ARD value at the shifted precision");
        assert!(
            (shifted - base).abs() < 1e-10 * n.max(base.abs()),
            "on ‖x‖ = 1, A → A + cI must leave the normalized prior unchanged: shift {shift}, \
             value {shifted} vs {base}"
        );
    }
    // The normalizer channel of the log-precision derivative is `n·(−½ α_a E[x_a²])`
    // under the Bingham law, integrated here independently.
    let derivatives = term
        .ard_log_precision_explicit_derivatives(&rho_with_log_ard(log_ard(0.0)))
        .expect("ARD derivatives at the anisotropic precision");
    let energy = energy_per_axis(&coords, &alpha);
    let moments = bingham_second_moments_by_product_rule(alpha);
    for a in 0..3 {
        let normalizer = derivatives[0][a] - energy[a];
        let reference = -0.5 * alpha[a] * n * moments[a];
        assert!(
            (normalizer - reference).abs() < 1e-10,
            "axis {a}: normalizer derivative {normalizer}, integrated moment {reference}"
        );
    }
}

/// `α·∂ log Z/∂α` on `[−r, r]` for `X ~ N(0, 1/α)`: with `b = r√α`,
/// `−½ + b φ(b) / (Φ(b) − Φ(−b))`.
fn symmetric_interval_score(radius: f64, alpha: f64) -> f64 {
    let b = radius * alpha.sqrt();
    let density = (-0.5 * b * b).exp() / TAU.sqrt();
    -0.5 + b * density / (normal_cdf(b) - normal_cdf(-b))
}

#[test]
fn interval_ard_normalizer_is_the_truncated_normal_mass_2933_f24() {
    let coords = array![[0.3], [-0.8], [0.55], [0.0]];
    let n = coords.nrows() as f64;
    let sum_sq: f64 = coords.iter().map(|t| t * t).sum();
    let term = ard_term(LatentManifold::Interval { lo: -1.0, hi: 1.0 }, coords);
    let derivative = term
        .ard_log_precision_explicit_derivatives(&rho_with_log_ard(array![0.0]))
        .expect("ARD derivative at α = 1")[0][0];
    let reference = 0.5 * sum_sq + n * symmetric_interval_score(1.0, 1.0);
    assert!(
        (derivative - reference).abs() < 1e-12,
        "interval [-1, 1] at α=1: derivative {derivative}, truncated normal {reference} \
         (the full-line normalizer gives {})",
        0.5 * sum_sq - 0.5 * n
    );
    assert!((derivative - 0.5 * sum_sq - n * -0.145562547388).abs() < 1e-9);

    // α → 0: the partition tends to the interval length, so the per-row normalizer
    // tends to `ln 2 − ½ ln 2π` and the derivative to zero.
    let log_alpha = -40.0_f64;
    let value = term
        .ard_value(&rho_with_log_ard(array![log_alpha]))
        .expect("ARD value at vanishing precision");
    let expected = 0.5 * log_alpha.exp() * sum_sq + n * (2.0_f64.ln() - 0.5 * TAU.ln());
    assert!(
        (value - expected).abs() < 1e-9,
        "interval at log α={log_alpha}: value {value}, uniform limit {expected}"
    );
    let derivative = term
        .ard_log_precision_explicit_derivatives(&rho_with_log_ard(array![log_alpha]))
        .expect("ARD derivative at vanishing precision")[0][0];
    assert!(derivative.abs() < 1e-9, "uniform-limit derivative {derivative}");

    // α → ∞ around the mode: the mass factor is one and the line normalizer returns.
    let tight = array![[1.0e-9], [-2.0e-9], [5.0e-10]];
    let tight_sum_sq: f64 = tight.iter().map(|t| t * t).sum();
    let tight_term = ard_term(LatentManifold::Interval { lo: -1.0, hi: 1.0 }, tight);
    let log_alpha = 40.0_f64;
    let energy = 0.5 * log_alpha.exp() * tight_sum_sq;
    let value = tight_term
        .ard_value(&rho_with_log_ard(array![log_alpha]))
        .expect("ARD value at large precision");
    assert!(
        (value - (energy - 1.5 * log_alpha)).abs() < 1e-9,
        "interval at log α={log_alpha}: value {value}, line limit {}",
        energy - 1.5 * log_alpha
    );
    let derivative = tight_term
        .ard_log_precision_explicit_derivatives(&rho_with_log_ard(array![log_alpha]))
        .expect("ARD derivative at large precision")[0][0];
    assert!((derivative - (energy - 1.5)).abs() < 1e-9, "line-limit derivative {derivative}");
}

/// The Möbius width axis is the interval `[−1, 1]`, and its angle axis keeps the
/// von Mises partition.
#[test]
fn mobius_width_axis_takes_the_interval_normalizer_2933_f24() {
    let coords = array![[0.3, 0.2], [1.7, -0.6], [0.9, 0.95]];
    let n = coords.nrows() as f64;
    let term = ard_term(
        LatentManifold::Product(vec![
            LatentManifold::Circle { period: 2.0 },
            LatentManifold::Interval { lo: -1.0, hi: 1.0 },
        ]),
        coords.clone(),
    );
    let log_ard = [0.4_f64, 0.8];
    let derivatives = term
        .ard_log_precision_explicit_derivatives(&rho_with_log_ard(array![log_ard[0], log_ard[1]]))
        .expect("ARD derivatives on the Möbius cover");
    let width_alpha = log_ard[1].exp();
    let width_energy: f64 = coords.column(1).iter().map(|t| 0.5 * width_alpha * t * t).sum();
    let width_reference = width_energy + n * symmetric_interval_score(1.0, width_alpha);
    assert!(
        (derivatives[0][1] - width_reference).abs() < 1e-12,
        "Möbius width axis: derivative {}, truncated normal {width_reference}",
        derivatives[0][1]
    );
    let angle_alpha = log_ard[0].exp();
    let kappa = TAU / 2.0;
    let angle_energy: f64 = coords
        .column(0)
        .iter()
        .map(|t| (angle_alpha / (kappa * kappa)) * (1.0 - (kappa * t).cos()))
        .sum();
    let log_eta = log_ard[0] + 2.0 * (2.0_f64.ln() - TAU.ln());
    let angle_reference = angle_energy + n * bessel_i0_centered_terms_from_log_abs(log_eta).2;
    assert!(
        (derivatives[0][0] - angle_reference).abs() < 1e-12,
        "Möbius angle axis: derivative {}, von Mises {angle_reference}",
        derivatives[0][0]
    );
}

/// The criterion value and its analytic log-precision gradient describe one
/// function on the constrained supports.
#[test]
fn constrained_ard_value_and_gradient_agree_2933_f24() {
    let cases = [
        (
            ard_term(LatentManifold::Sphere { dim: 3 }, sphere_rows()),
            vec![0.9_f64, -0.4, 1.6],
        ),
        (
            ard_term(
                LatentManifold::Interval { lo: -0.5, hi: 2.0 },
                array![[0.3], [1.8], [-0.45]],
            ),
            vec![0.7_f64],
        ),
    ];
    let step = 1e-5;
    for (term, log_ard) in cases {
        let derivatives = term
            .ard_log_precision_explicit_derivatives(&rho_with_log_ard(Array1::from(log_ard.clone())))
            .expect("ARD derivatives on a constrained fixture");
        for axis in 0..log_ard.len() {
            let mut plus = log_ard.clone();
            let mut minus = log_ard.clone();
            plus[axis] += step;
            minus[axis] -= step;
            let value_plus = term
                .ard_value(&rho_with_log_ard(Array1::from(plus)))
                .expect("ARD value above");
            let value_minus = term
                .ard_value(&rho_with_log_ard(Array1::from(minus)))
                .expect("ARD value below");
            let slope = (value_plus - value_minus) / (2.0 * step);
            assert!(
                (slope - derivatives[0][axis]).abs() < 1e-7 * derivatives[0][axis].abs().max(1.0),
                "axis {axis} of {log_ard:?}: finite difference {slope}, analytic {}",
                derivatives[0][axis]
            );
        }
    }
}
