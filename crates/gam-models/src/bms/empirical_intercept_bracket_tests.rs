//! The empirical-law intercept solve and its analytic bracket.
//!
//! The calibrated intercept `a` of a row under a declared finite law is the root
//! of `Σ wᵢ Φ(a + β·zᵢ) = μ★` with `β = s·b`. Monotonicity in `a` and weights
//! summing to `W` put that root in `[Φ⁻¹(μ★/W) − max βzᵢ, Φ⁻¹(μ★/W) − min βzᵢ]`,
//! so the solve falls back to a bracketed refinement there where the
//! warm-started Newton–Halley refinement stops short. Every root is checked
//! against an independent linear-space bisection, never against the code under
//! test.

#![cfg(test)]

use super::gradient_paths::{
    empirical_intercept_from_marginal_within, empirical_intercept_in_analytic_bracket,
    empirical_intercept_tail_tolerance, empirical_rigid_calibration_eval,
    rigid_intercept_from_marginal,
};
use crate::probability::normal_cdf;

/// The node gam-cli's default posterior-mean predict refused (gam#2927
/// regression, `cli_bernoulli_marginal_slope_fit_saves_covariance_so_default_predict_succeeds`),
/// as captured at a35efd6588: the fixture's fitted 12-node law and one
/// Gauss–Hermite node with a steep slope. The warm-started solve evaluated
/// `F = −9.1e-6` at `a ≈ 10.176` in its warm-start probes, bracketed from the
/// seed instead, and refined for 48 iterations with only the probe side of
/// `[9.989, 12.737]` moving, returning log-residual `−4.75e-3` at `a ≈ 9.990`.
/// The production solve must return the root.
#[test]
fn empirical_intercept_solve_converges_where_warm_started_refinement_stalls() {
    let nodes = [
        -1.8255629993581926,
        -1.2313276572905227,
        -0.8029379970544664,
        -0.8029379970544664,
        -0.4368079942486818,
        -0.09471751209927097,
        0.2473729700501398,
        0.6135029728559244,
        0.6135029728559244,
        1.0418926330919809,
        1.0418926330919809,
        1.6361279751596505,
    ];
    let weights = [1.0 / 12.0; 12];
    let (slope, target_q, target_mu) = (
        1.06531872907183569e1,
        9.33588375376590118e-1,
        8.24741868009762014e-1,
    );
    let tol = empirical_intercept_tail_tolerance(target_mu);
    let eval =
        |a: f64| empirical_rigid_calibration_eval(a, target_mu.ln(), slope, 1.0, &nodes, &weights);
    let warm_started = crate::monotone_root::solve_monotone_root(
        eval,
        rigid_intercept_from_marginal(target_q, slope, 1.0),
        "warm-started witness",
        tol,
        64,
        48,
    );
    let root = empirical_intercept_from_marginal_within(
        target_mu, target_q, slope, 1.0, &nodes, &weights, None, tol,
    )
    .unwrap_or_else(|e| panic!("the captured node's intercept must solve: {e}"));
    let (residual, derivative, _) = eval(root).expect("the calibration evaluates at the root");
    let reference = bisection_root(target_mu, slope, &nodes, &weights);
    let bound = 1e3 * tol / derivative + 64.0 * f64::EPSILON * (1.0 + reference.abs());
    eprintln!(
        "[gam#2927 node] warm-started solve alone: {warm_started:?}; production root {root:+.12} \
         vs bisection {reference:+.12} (|Δa| {:.2e}, bound {bound:.2e}); log-residual \
         {residual:+.2e}",
        (root - reference).abs()
    );
    assert!(
        residual.abs() <= 1e3 * tol,
        "log-residual {residual:e} above the acceptance {:e}",
        1e3 * tol
    );
    assert!(
        (root - reference).abs() <= bound,
        "root {root} is {:e} from the bisection root {reference}, beyond the {bound:e} its \
         residual allows",
        (root - reference).abs()
    );
}

/// The latent scores of the gam-cli marginal-slope fixture, standardized, at
/// equal mass: an asymmetric 12-node law with tied scores.
fn fixture_law() -> (Vec<f64>, Vec<f64>) {
    let raw = [
        -1.2816_f64,
        -0.8416,
        -0.5244,
        -0.2533,
        0.0,
        0.2533,
        0.5244,
        0.8416,
        1.2816,
        -0.5244,
        0.5244,
        0.8416,
    ];
    let n = raw.len() as f64;
    let mean = raw.iter().sum::<f64>() / n;
    let sd = (raw.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n).sqrt();
    let mut nodes: Vec<f64> = raw.iter().map(|v| (v - mean) / sd).collect();
    nodes.sort_by(f64::total_cmp);
    (nodes, vec![1.0 / n; raw.len()])
}

/// The root of `Σ wᵢ Φ(a + β·zᵢ) = μ★` by bisection in linear space, on the tail
/// that is well conditioned: `Σ wᵢ Φ(−(a + β·zᵢ)) = 1 − μ★` above one half.
fn bisection_root(target_mu: f64, observed_slope: f64, nodes: &[f64], weights: &[f64]) -> f64 {
    let upper_tail = target_mu > 0.5;
    let target = if upper_tail {
        1.0 - target_mu
    } else {
        target_mu
    };
    let calibrated = |a: f64| -> f64 {
        nodes
            .iter()
            .zip(weights.iter())
            .map(|(&node, &weight)| {
                let eta = a + observed_slope * node;
                weight * normal_cdf(if upper_tail { -eta } else { eta })
            })
            .sum()
    };
    let (mut low, mut high) = (-200.0_f64, 200.0_f64);
    for _ in 0..300 {
        let mid = 0.5 * (low + high);
        let root_is_above = if upper_tail {
            calibrated(mid) > target
        } else {
            calibrated(mid) < target
        };
        if root_is_above {
            low = mid;
        } else {
            high = mid;
        }
    }
    0.5 * (low + high)
}

/// The bracket orders its ends through `min`/`max` of `β·zᵢ`, so a negative
/// slope is bracketed as correctly as a positive one; at `b = 0` it is the single
/// point `Φ⁻¹(μ★/W)`, returned exactly; and the solve inside it reaches the
/// calibration root at both link clamps. The accepted log-space residual bounds
/// how far a root may sit from the true one, through the calibration's slope
/// `F′` at the root; each root is held to that bound.
#[test]
fn empirical_intercept_analytic_bracket_solves_every_slope_sign_at_both_clamps() {
    let (nodes, weights) = fixture_law();
    let total_weight: f64 = weights.iter().sum();
    for &target_mu in &[1e-12, 0.3, 0.824771, 1.0 - 1e-12] {
        let tol = empirical_intercept_tail_tolerance(target_mu);
        for &slope in &[-12.0, -3.7, -0.6, 0.0, 0.6, 3.7, 12.0] {
            let root = empirical_intercept_in_analytic_bracket(
                target_mu, slope, 1.0, &nodes, &weights, tol,
            )
            .unwrap_or_else(|e| panic!("mu*={target_mu:e}, b={slope}: {e}"));
            let (residual, derivative, _) = empirical_rigid_calibration_eval(
                root,
                target_mu.ln(),
                slope,
                1.0,
                &nodes,
                &weights,
            )
            .expect("the calibration evaluates at the returned root");
            let reference = bisection_root(target_mu, slope, &nodes, &weights);
            let bound = 1e3 * tol / derivative + 64.0 * f64::EPSILON * (1.0 + reference.abs());
            eprintln!(
                "[bracket] mu*={target_mu:.3e} b={slope:+5.1}: a={root:+.12} vs bisection \
                 {reference:+.12} (|Δa| {:.2e}, bound {bound:.2e}); log-residual {residual:+.2e}",
                (root - reference).abs()
            );
            assert!(
                residual.abs() <= 1e3 * tol,
                "mu*={target_mu:e}, b={slope}: log-residual {residual:e} above the acceptance \
                 {:e}",
                1e3 * tol
            );
            assert!(
                (root - reference).abs() <= bound,
                "mu*={target_mu:e}, b={slope}: root {root} is {:e} from the bisection root \
                 {reference}, beyond the {bound:e} its residual allows",
                (root - reference).abs()
            );
            if slope == 0.0 {
                let quantile =
                    gam_math::probability::standard_normal_quantile(target_mu / total_weight)
                        .expect("quantile of an interior level");
                assert_eq!(
                    root.to_bits(),
                    quantile.to_bits(),
                    "mu*={target_mu:e}: at b = 0 the root is the quantile itself"
                );
            }
        }
    }
}
