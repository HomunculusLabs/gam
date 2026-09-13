//! #2080 — the gate prior's logit Jacobian `−ln[z(1 − z)/τ]`.
//!
//! The ordered Beta--Bernoulli and ThresholdGate priors are densities on the gate
//! probability, while the inner solve and the Laplace evidence integrate over its logit, so
//! the penalized objective carries the change of variables. These pins check every
//! derivative order against a central difference of the order below it, the loss value
//! against the gradient and curvature the assembly installs, and the finite mode the term
//! gives a gate whose data want it saturated.

use super::tests_outer_quasi_laplace_probe_budget_2080::{
    one_circle_wide_target, two_circle_periodic_term,
};
use super::*;
use crate::assignment::{
    GateLogitJacobian, assignment_prior_value_weighted, gate_logit_jacobian_grad_hdiag_weighted,
    gate_logit_jacobian_third_weighted, gate_logit_jacobian_value_weighted, sigmoid_gate_frame,
};
use ndarray::{Array2, array};

/// `(f(x + h) − f(x − h))/(2h)` and the bar it must meet: twice the mean-value truncation
/// `h²/6·sup_third`, where `sup_third` bounds `|f‴|` on the stencil (the slack keeps a
/// difference taken at `|f‴|`'s own extremum clear of rounding in the stencil), plus
/// round-off from the two values and from rounding the argument `x` (and the gate threshold
/// inside `f`), whose effect on `f` `sup_slope` bounds.
fn central_difference(
    f: impl Fn(f64) -> f64,
    x: f64,
    h: f64,
    argument_scale: f64,
    sup_slope: f64,
    sup_third: f64,
) -> (f64, f64) {
    let up = f(x + h);
    let down = f(x - h);
    let truncation = 2.0 * h * h / 6.0 * sup_third;
    let round_off =
        8.0 * f64::EPSILON * (up.abs() + down.abs() + sup_slope * (argument_scale + h)) / h;
    ((up - down) / (2.0 * h), truncation + round_off)
}

/// Suprema over every logit of `|J′|`, `|J″|`, `|J‴|`, `|J⁗|` and `|J⁽⁵⁾|` for a gate of
/// weight `w` and temperature `τ`: `w/τ`, `w/(2τ²)`, `w/(3√3·τ³)`, `w/(4τ⁴)` and at most `w/τ⁵`
/// (from `|2z − 1| ≤ 1`, `z(1 − z) ≤ ¼` and their products).
fn jacobian_derivative_suprema(weight: f64, temperature: f64) -> [f64; 5] {
    let inv_tau = temperature.recip();
    [
        weight * inv_tau,
        weight * inv_tau.powi(2) / 2.0,
        weight * inv_tau.powi(3) / (3.0 * 3.0_f64.sqrt()),
        weight * inv_tau.powi(4) / 4.0,
        weight * inv_tau.powi(5),
    ]
}

/// Each of `J′`, `J″`, `J‴` is the central difference of the order below it, from deep
/// saturation on either side through the centre, in the ordered Beta--Bernoulli frame and a
/// design-weighted ThresholdGate frame.
#[test]
fn gate_logit_jacobian_orders_match_central_differences_2080() {
    for (weight, threshold, temperature) in [(1.0_f64, 0.0_f64, 1.0_f64), (2.5, 0.7, 0.35)] {
        let h = temperature * f64::EPSILON.cbrt();
        let sup = jacobian_derivative_suprema(weight, temperature);
        let at = |logit: f64| GateLogitJacobian::eval(weight, logit, threshold, temperature);
        for offset in [-40.0_f64, -18.0, -6.0, -1.3, 0.0, 0.4, 2.2, 9.0, 25.0] {
            let logit = threshold + temperature * offset;
            let scale = logit.abs() + threshold.abs();
            let here = at(logit);
            let orders = [
                (
                    "gradient",
                    central_difference(|l| at(l).value(), logit, h, scale, sup[0], sup[2]),
                    here.gradient(),
                ),
                (
                    "curvature",
                    central_difference(|l| at(l).gradient(), logit, h, scale, sup[1], sup[3]),
                    here.curvature(),
                ),
                (
                    "third",
                    central_difference(|l| at(l).curvature(), logit, h, scale, sup[2], sup[4]),
                    here.third(),
                ),
            ];
            for (order, (difference, bar), analytic) in orders {
                assert!(
                    (difference - analytic).abs() <= bar + 4.0 * f64::EPSILON * analytic.abs(),
                    "gate logit Jacobian {order} at w={weight}, θ={threshold}, τ={temperature}, \
                     x={offset}: analytic {analytic:.12e}, central difference {difference:.12e}, \
                     bar {bar:.3e}"
                );
            }
        }
    }
}

/// The aggregate producers the loss, the assembly and the θ-adjoints read are one derivation:
/// under row weights, in both sigmoid modes, each order is the central difference of the
/// order below it along every logit.
#[test]
fn gate_logit_jacobian_producers_are_one_derivation_2080() {
    let logits = array![[6.0, -2.5], [-18.0, 0.3], [1.1, 25.0], [-0.4, -7.0]];
    let (n, k) = logits.dim();
    let row_weights = [0.5, 1.0, 2.0, 1.25];
    for mode in [
        AssignmentMode::ordered_beta_bernoulli(1.0, 1.0, false),
        AssignmentMode::threshold_gate(0.35, 0.7),
    ] {
        let mut coords = Vec::with_capacity(k);
        let mut manifolds = Vec::with_capacity(k);
        for atom in 0..k {
            coords.push(Array2::from_shape_fn((n, 1), |(row, axis)| {
                0.05 * (row + axis + atom) as f64
            }));
            manifolds.push(LatentManifold::Circle { period: 1.0 });
        }
        let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
            logits.clone(),
            coords,
            manifolds,
            mode,
        )
        .expect("one logit column, coordinate block and manifold per atom");
        let (threshold, temperature) = sigmoid_gate_frame(&assignment.mode)
            .expect("ordered Beta--Bernoulli and ThresholdGate gates are per-logit sigmoids");
        let h = temperature * f64::EPSILON.cbrt();
        let (gradient, curvature) =
            gate_logit_jacobian_grad_hdiag_weighted(&assignment, Some(&row_weights));
        for row in 0..n {
            let sup = jacobian_derivative_suprema(row_weights[row], temperature);
            for atom in 0..k {
                let index = row * k + atom;
                let logit = assignment.logits[[row, atom]];
                let scale = logit.abs() + threshold.abs();
                let moved = |value: f64| {
                    let mut shifted = assignment.clone();
                    shifted.logits[[row, atom]] = value;
                    shifted
                };
                let orders = [
                    (
                        "gradient",
                        central_difference(
                            |l| gate_logit_jacobian_value_weighted(&moved(l), Some(&row_weights)),
                            logit,
                            h,
                            scale,
                            sup[0],
                            sup[2],
                        ),
                        gradient[index],
                    ),
                    (
                        "curvature",
                        central_difference(
                            |l| {
                                gate_logit_jacobian_grad_hdiag_weighted(&moved(l), Some(&row_weights))
                                    .0[index]
                            },
                            logit,
                            h,
                            scale,
                            sup[1],
                            sup[3],
                        ),
                        curvature[index],
                    ),
                    (
                        "third",
                        central_difference(
                            |l| {
                                gate_logit_jacobian_grad_hdiag_weighted(&moved(l), Some(&row_weights))
                                    .1[index]
                            },
                            logit,
                            h,
                            scale,
                            sup[2],
                            sup[4],
                        ),
                        gate_logit_jacobian_third_weighted(&assignment, Some(&row_weights), row, atom),
                    ),
                ];
                for (order, (difference, bar), analytic) in orders {
                    assert!(
                        (difference - analytic).abs() <= bar + 4.0 * f64::EPSILON * analytic.abs(),
                        "{} gate Jacobian {order} at row {row}, atom {atom}, logit {logit}: \
                         analytic {analytic:.12e}, central difference {difference:.12e}, \
                         bar {bar:.3e}",
                        assignment.mode.family_label()
                    );
                }
            }
        }
    }
}

/// On a real term whose gates are seeded saturated, the loss field carries the Jacobian and
/// nothing else beyond the sparsity prior, its logit gradient is the producer's, and the
/// assembled `H_tt` logit diagonal holds at least the Jacobian's curvature (the data
/// Gauss--Newton term and the majorized prior term it sits beside are non-negative).
#[test]
fn loss_carries_the_jacobian_the_assembly_installs_2080() {
    let z = one_circle_wide_target(24, 8, 0.05);
    let term = two_circle_periodic_term(z.view(), 2, 1).0;
    let rho = SaeManifoldRho::new(0.02_f64.ln(), 1.0_f64.ln(), vec![array![0.0]; 2])
        .for_assignment(AssignmentMode::ordered_beta_bernoulli(1.0, 1.0, false));
    let (threshold, temperature) = sigmoid_gate_frame(&term.assignment.mode)
        .expect("the fixture's ordered Beta--Bernoulli gates are per-logit sigmoids");
    let h = temperature * f64::EPSILON.cbrt();
    let (n, k) = term.assignment.logits.dim();
    let weights = term.row_loss_weights.clone();
    let (gradient, curvature) =
        gate_logit_jacobian_grad_hdiag_weighted(&term.assignment, weights.as_deref());
    let jacobian_in_loss = |shifted: &SaeManifoldTerm| -> f64 {
        let loss = shifted.loss(z.view(), &rho).expect("loss at the shifted state");
        let prior =
            assignment_prior_value_weighted(&shifted.assignment, &rho, weights.as_deref())
                .expect("assignment prior value at the shifted state");
        loss.assignment_sparsity - prior
    };
    assert!(
        (jacobian_in_loss(&term)
            - gate_logit_jacobian_value_weighted(&term.assignment, weights.as_deref()))
        .abs()
            <= 64.0 * f64::EPSILON * (1.0 + jacobian_in_loss(&term).abs()),
        "the loss field must carry exactly the gate Jacobian beyond the sparsity prior"
    );
    for row in 0..n {
        let weight = weights.as_ref().map_or(1.0, |w| w[row]);
        let sup = jacobian_derivative_suprema(weight, temperature);
        for atom in 0..k {
            let index = row * k + atom;
            let logit = term.assignment.logits[[row, atom]];
            let moved = |value: f64| {
                let mut shifted = term.clone();
                shifted.assignment.logits[[row, atom]] = value;
                jacobian_in_loss(&shifted)
            };
            let (difference, bar) =
                central_difference(moved, logit, h, logit.abs() + threshold.abs(), sup[0], sup[2]);
            assert!(
                (difference - gradient[index]).abs() <= bar + 4.0 * f64::EPSILON * gradient[index].abs(),
                "loss-field Jacobian gradient at row {row}, atom {atom}, logit {logit}: producer \
                 {:.12e}, central difference {difference:.12e}, bar {bar:.3e}",
                gradient[index]
            );
        }
    }
    let mut assembled = term.clone();
    let system = assembled
        .assemble_arrow_schur(z.view(), &rho, None)
        .expect("assemble the fixture's arrow system");
    for (row, block) in system.rows.iter().enumerate() {
        let vars = assembled
            .row_vars_for_row_dim(row, block.htt.nrows())
            .expect("row layout of the assembled system");
        for (slot, var) in vars.iter().enumerate() {
            if let SaeLocalRowVar::Logit { atom } = *var {
                let installed = block.htt[[slot, slot]];
                let jacobian = curvature[row * k + atom];
                assert!(
                    installed >= jacobian,
                    "row {row}, atom {atom}: installed logit curvature {installed:.12e} is below \
                     the gate Jacobian's own {jacobian:.12e}"
                );
            }
        }
    }
}

/// A gate whose remaining objective gains `μ` per unit of gate probability wants the gate
/// fully on: `g(z) = −μ·z`, so along its logit `f(ℓ) = −μ·z + J(ℓ)` with `z = σ(ℓ/τ)`.
///
/// Without `J`, `f′ = −μ·z(1 − z)/τ < 0` at every finite logit, so there is no mode. With it
/// `f′ = 0` at `μ·z(1 − z) = w·(2z − 1)`, i.e. `z* = [(μ − 2w) + √(μ² + 4w²)]/(2μ)`, and there
/// `f″ = w·(2z*² − 2z* + 1)/τ²`. On `z ≥ ½`, `f″ = z(1 − z)·(2w − μ·(1 − 2z))/τ² > 0`, so `f′`
/// increases from `−μ/(4τ)` at `ℓ = 0` to `w/τ` in the limit and the mode is its one root there.
#[test]
fn a_saturating_gate_has_a_finite_mode_at_the_derived_curvature_2080() {
    for (mu, weight, temperature) in [(100.0_f64, 1.0_f64, 1.0_f64), (4.0e3, 0.5, 0.25), (12.0, 3.0, 2.0)]
    {
        let inv_tau = temperature.recip();
        let gate = |logit: f64| gam_linalg::utils::stable_logistic(logit * inv_tau);
        let slope = |logit: f64| {
            let z = gate(logit);
            -mu * z * (1.0 - z) * inv_tau
                + GateLogitJacobian::eval(weight, logit, 0.0, temperature).gradient()
        };
        let curvature = |logit: f64| {
            let z = gate(logit);
            -mu * z * (1.0 - z) * (1.0 - 2.0 * z) * inv_tau * inv_tau
                + GateLogitJacobian::eval(weight, logit, 0.0, temperature).curvature()
        };
        // Bracket the root of the increasing `f′` on `ℓ ≥ 0`, then bisect to float resolution.
        let mut low = 0.0_f64;
        let mut high = temperature;
        while slope(high) <= 0.0 {
            low = high;
            high *= 2.0;
            assert!(high.is_finite(), "the gate slope never turned positive");
        }
        loop {
            let middle = 0.5 * (low + high);
            if middle <= low || middle >= high {
                break;
            }
            if slope(middle) <= 0.0 {
                low = middle;
            } else {
                high = middle;
            }
        }
        let mode = 0.5 * (low + high);
        let z_mode = gate(mode);
        let z_derived = ((mu - 2.0 * weight) + (mu * mu + 4.0 * weight * weight).sqrt()) / (2.0 * mu);
        let x_mode = mode * inv_tau;
        assert!(
            mode.is_finite() && (z_mode - z_derived).abs() <= 16.0 * f64::EPSILON * (1.0 + x_mode.abs()),
            "μ={mu}, w={weight}, τ={temperature}: the mode's gate {z_mode:.15e} is not the derived \
             {z_derived:.15e}"
        );
        let derived_curvature =
            weight * (2.0 * z_derived * z_derived - 2.0 * z_derived + 1.0) * inv_tau * inv_tau;
        assert!(
            (curvature(mode) - derived_curvature).abs()
                <= 64.0 * f64::EPSILON * (mu + 4.0 * weight) * inv_tau * inv_tau * (1.0 + x_mode.abs()),
            "μ={mu}, w={weight}, τ={temperature}: curvature at the mode {:.15e} is not the derived \
             {derived_curvature:.15e}",
            curvature(mode)
        );
        // Negative control: without the Jacobian the same gate has no stationary logit past the
        // mode, where its slope stays strictly negative. The slope is taken as
        // `e^{−|x|}/(1 + e^{−|x|})²` rather than `z(1 − z)`, whose `1 − z` rounds to zero past
        // `x ≈ 37`.
        for step in 0..=6 {
            let logit = mode + 10.0 * step as f64 * temperature;
            let tail = (-(logit * inv_tau).abs()).exp();
            let bare_slope = -mu * tail / ((1.0 + tail) * (1.0 + tail)) * inv_tau;
            assert!(
                bare_slope < 0.0,
                "μ={mu}, τ={temperature}: without the Jacobian the slope at logit {logit} must be \
                 negative, got {bare_slope:.3e}"
            );
        }
    }
}
