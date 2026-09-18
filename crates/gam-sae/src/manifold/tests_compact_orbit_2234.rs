//! #2234 — the compact-orbit integral: the trapezoid quadrature against the von Mises closed form,
//! doubled-node agreement with the complement coupling on, and a positive control showing the
//! derived node count is not vacuously loose.

#![cfg(test)]

use super::compact_orbit::CircleOrbitIntegrand;
use gam_math::special::bessel_i0_centered_terms_from_log_abs;

/// `log ∫₀ᴾ exp(−c[cos φ̄ − cos(φ̄ + κs)]) ds = log P − c cos φ̄ + log I₀(c)`.
fn closed_form_log_integral(resultant_cos: f64, resultant_sin: f64, period: f64) -> f64 {
    let c = resultant_cos.hypot(resultant_sin);
    let centered = if c > 0.0 {
        bessel_i0_centered_terms_from_log_abs(c.ln()).0
    } else {
        0.0
    };
    period.ln() - resultant_cos + c + centered
}

/// The trapezoid sum at an explicit node count, the same arithmetic `integrate` runs.
fn trapezoid_log_integral(integrand: &CircleOrbitIntegrand, nodes: usize) -> f64 {
    let exponents: Vec<f64> = (0..nodes)
        .map(|j| integrand.exponent(std::f64::consts::TAU * j as f64 / nodes as f64))
        .collect();
    let normalizer = gam_math::categorical::log_sum_exp(&exponents).expect("finite exponents");
    (integrand.period / nodes as f64).ln() + normalizer
}

#[test]
fn circle_orbit_integral_matches_the_von_mises_closed_form_2234() {
    // Concentrations from the #2234 stall pin's railed ARD state (c ≈ 1.12e-6), an O(1) state and
    // E1's collapsed winner (c = 1689), at a phase off the minimum so `cos φ̄ ≠ 1`.
    for &(c, phase) in &[(1.12e-6, 0.3_f64), (1.0, -1.1), (1689.0, 0.02)] {
        let integrand = CircleOrbitIntegrand {
            resultant_cos: c * phase.cos(),
            resultant_sin: c * phase.sin(),
            coupling: [0.0, 0.0, 0.0],
            period: 1.0,
        };
        let integral = integrand.integrate().expect("orbit integral");
        let oracle = closed_form_log_integral(integrand.resultant_cos, integrand.resultant_sin, 1.0);
        let nodes = integral.angles.len();
        // The quadrature's own band: its node sum's accumulation growth plus the closed form's
        // log-magnitude rounding.
        let band = gam_linalg::roundoff::accumulation_growth(nodes)
            + 4.0 * f64::EPSILON * (oracle.abs() + c);
        eprintln!(
            "[#2234 orbit] c={c:e} nodes={nodes} quadrature={:.16e} closed_form={oracle:.16e} gap={:e} band={band:e}",
            integral.log_integral,
            (integral.log_integral - oracle).abs()
        );
        assert!(
            (integral.log_integral - oracle).abs() <= band,
            "c = {c}: trapezoid log integral {} misses the closed form {oracle} by {} against band {band}",
            integral.log_integral,
            (integral.log_integral - oracle).abs()
        );
        let total: f64 = integral.weights.iter().sum();
        assert!((total - 1.0).abs() <= gam_linalg::roundoff::accumulation_growth(nodes) * 2.0);
    }
}

#[test]
fn circle_orbit_node_count_is_tight_with_the_complement_coupling_on_2234() {
    // An O(1) concentration with a nonzero coupling, where no closed form exists: the derived count
    // must agree with twice as many nodes to the band, and a quarter of the count must miss it.
    let integrand = CircleOrbitIntegrand {
        resultant_cos: 2.5 * 0.4_f64.cos(),
        resultant_sin: 2.5 * 0.4_f64.sin(),
        coupling: [0.7, -0.2, 0.35],
        period: 1.0,
    };
    let nodes = integrand.node_count();
    let at_count = trapezoid_log_integral(&integrand, nodes);
    let doubled = trapezoid_log_integral(&integrand, 2 * nodes);
    let band = gam_linalg::roundoff::accumulation_growth(2 * nodes) + 4.0 * f64::EPSILON * at_count.abs();
    let quarter = (nodes / 4).max(1);
    let at_quarter = trapezoid_log_integral(&integrand, quarter);
    eprintln!(
        "[#2234 orbit coupling] nodes={nodes} at_count={at_count:.16e} doubled={doubled:.16e} \
         quarter_nodes={quarter} at_quarter={at_quarter:.16e} band={band:e}"
    );
    assert!(nodes >= 8, "the coupled integrand needs more than a handful of nodes, got {nodes}");
    assert!(
        (at_count - doubled).abs() <= band,
        "doubling the derived node count moved the integral by {} against band {band}",
        (at_count - doubled).abs()
    );
    assert!(
        (at_quarter - doubled).abs() > band,
        "a quarter of the derived node count ({quarter}) already agrees to the band, so the bound is loose"
    );
}

/// The derived node count terminates where the coupling growth outruns the concentration.
///
/// The demo's minted state carries log α = 6.012 (ad-sae2's defect 2, state 2). At the basis period
/// P = 1 that is η = αP²/(2π)² and a coupling scale ½η²κ², and the phase cloud is spread over the
/// demo's 160 rows (resultant η·√160). The complement forms `(a, b, d) = (0.05, 0.01, 0.05)` are
/// representative fixture inputs chosen to put Q above c, not measured values. That is the regime
/// where a strip width balancing only c·cosh σ against Nσ lets Q·cosh²σ outgrow the decay.
#[test]
fn circle_orbit_node_count_terminates_at_a_demo_like_coupling_2234() {
    let kappa = std::f64::consts::TAU;
    let eta = 6.012_f64.exp() / (kappa * kappa);
    let resultant = eta * 160.0_f64.sqrt();
    let scale = 0.5 * eta * eta * kappa * kappa;
    let forms = [0.05_f64, 0.01, 0.05];
    let integrand = CircleOrbitIntegrand {
        resultant_cos: resultant * 0.7_f64.cos(),
        resultant_sin: resultant * 0.7_f64.sin(),
        coupling: [forms[0] * scale, forms[1] * scale, forms[2] * scale],
        period: 1.0,
    };
    let c = integrand.concentration();
    let [qa, qb, qd] = integrand.coupling;
    let q = qa.abs() + 2.0 * qb.abs() + qd.abs();
    let nodes = integrand.node_count();
    let bound = integrand.relative_truncation_bound(nodes);
    let target = gam_linalg::roundoff::accumulation_growth(nodes);
    // Positive control: the superseded half-width σ = asinh(N/(c + 2Q)) on the same state. Its log
    // bound stays above every target and rises with N, so no node count meets it under that rule.
    let superseded: Vec<(usize, f64, f64)> = [1024usize, 4096, 16384, 65536]
        .iter()
        .map(|&count| {
            let sigma = (count as f64 / (c + 2.0 * q)).asinh();
            (
                count,
                integrand.log_relative_truncation_bound_at(count, sigma),
                gam_linalg::roundoff::accumulation_growth(count).ln(),
            )
        })
        .collect();
    eprintln!(
        "[#2234 orbit demo coupling] eta={eta:.6e} forms={forms:?} c={c:.6e} Q={q:.6e} nodes={nodes} bound={bound:e} \
         target={target:e} superseded (N, log bound, log target) {superseded:?}"
    );
    assert!(q > c, "the state must put the coupling growth above the concentration (Q = {q}, c = {c})");
    assert!(bound <= target, "the derived count {nodes} misses its own target: bound {bound} > {target}");
    assert!(
        superseded.iter().all(|&(_, log_bound, log_target)| log_bound > log_target)
            && superseded.windows(2).all(|pair| pair[1].1 > pair[0].1),
        "the superseded strip width must give a bound above its target that rises with N, got {superseded:?}"
    );
}
