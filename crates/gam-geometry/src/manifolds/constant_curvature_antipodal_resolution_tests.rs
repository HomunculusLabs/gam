//! gam#2687: what does the κ box's half-margin to the antipodal fold BUY?
//!
//! `CONSTANT_CURVATURE_KAPPA_CHART_FRACTION = 0.5` is documented as a
//! half-margin to the fold at `κ‖x‖‖c‖ = 1`, and until #2687 nothing in the tree
//! stated what the margin protects — so it could be re-asserted but not argued
//! with. This module measures it, and the measurement is what
//! `constant_curvature_kappa_bounds`' doc now cites.
//!
//! The instrument is an exact oracle. For the anti-aligned equal-radius pair
//! `(±R, 0)` the two colatitudes `θ = 2·arctan(√κ R)` simply add, so
//!
//! ```text
//!   d(κ) = (4/√κ)·arctan(√κ·R),   valid for t = κR² ∈ (0, 1]
//! ```
//!
//! and every expression in it and in its κ-derivatives is a sum of same-sign
//! terms in `1 + R²κ` — no cancellation at any `t`, right up to the fold. The
//! shipped route instead goes through `w = (−x) ⊕_κ y`, whose Möbius denominator
//! `D = (1 − t)²` collapses there. Differencing the two isolates exactly what
//! the margin is protecting, with no finite differences anywhere: the oracle's
//! derivatives are closed form.
//!
//! The measured law is `rel_err(∂ᵏd/∂κᵏ) ≈ ε·D^{−(k+1)/2}`, i.e. the value is
//! barely affected, the gradient goes as `ε/D`, and the κ-Hessian — the quantity
//! the outer route actually consumes (`Derivative::Analytic`, exact `d²V/dκ²`)
//! — goes as `ε/D^{3/2}`.

use super::constant_curvature::{ConstantCurvature, distance_kappa_jet};

/// `d(κ) = 4·arctan(R√κ)/√κ` for the anti-aligned pair `(±R, 0)`, and its exact
/// first and second κ-derivatives. The oracle: no subtraction of comparable
/// quantities anywhere, so its own error stays at a few ε for every `t < 1`.
fn antipodal_pair_reference(r: f64, kappa: f64) -> (f64, f64, f64) {
    let s = kappa.sqrt();
    let rs = r * s;
    let den = 1.0 + rs * rs;
    let at = rs.atan();
    // d(s) = 4·atan(Rs)/s
    let d = 4.0 * at / s;
    // N(s)  = Rs/(1+R²s²) − atan(Rs);      d'(s) = 4N/s²
    let n = rs / den - at;
    let d_s = 4.0 * n / (s * s);
    // N'(s) = −2R³s²/(1+R²s²)²
    let n_s = -2.0 * r * rs * rs / (den * den);
    let d_ss = 4.0 * (n_s / (s * s) - 2.0 * n / (s * s * s));
    // κ = s² ⇒ ∂/∂κ = (1/2s)∂/∂s, ∂²/∂κ² = (1/4s²)(∂²/∂s² − (1/s)∂/∂s)
    let d_k = d_s / (2.0 * s);
    let d_kk = (d_ss - d_s / s) / (4.0 * s * s);
    (d, d_k, d_kk)
}

/// The κ-Hessian's relative error obeys `ε/D^{3/2}` across four decades of `D`,
/// and the shipped `0.5` fraction sits SIX ORDERS inside the bar that law
/// implies. Both halves matter: the first makes the resolution claim in
/// `CONSTANT_CURVATURE_KAPPA_CHART_FRACTION`'s doc checkable, the second is the
/// evidence that the fraction is a MODELLING retreat and not a numerical one —
/// so anyone proposing to move it has to argue about the estimator.
#[test]
fn the_kappa_hessian_resolution_law_is_epsilon_over_d_to_the_three_halves_2687() {
    const R: f64 = 0.6;
    let x = ndarray::array![R, 0.0];
    let y = ndarray::array![-R, 0.0];
    let eps = f64::EPSILON;

    // Asymptotic regime: `D` small enough that the cancellation in the Möbius
    // quotient dominates the few-ε floor of everything else.
    let mut constants = Vec::new();
    for t in [1.0 - 1e-3_f64, 1.0 - 1e-4, 1.0 - 1e-5, 1.0 - 1e-6] {
        let kappa = t / (R * R);
        let manifold = ConstantCurvature::new(2, kappa);
        let (_, _, d_kk_ref) = antipodal_pair_reference(R, kappa);
        let (_, _, d_kk) = distance_kappa_jet(&manifold, x.view(), y.view())
            .expect("the sweep stays strictly inside the fold");
        let d = (1.0 - t) * (1.0 - t);
        let rel = ((d_kk - d_kk_ref) / d_kk_ref).abs();
        // rel ≈ c·ε·D^{−3/2}; recover `c` and require it to be O(1) — that is
        // the exponent claim, since a wrong exponent makes `c` drift by decades
        // across a four-decade sweep in `D`.
        let c = rel * d.powf(1.5) / eps;
        assert!(
            (0.05..=50.0).contains(&c),
            "κ-Hessian resolution at D = {d:.3e}: rel_err = {rel:.3e}, implying \
             c = rel·D^{{3/2}}/ε = {c:.3} — the law ε/D^{{3/2}} requires an O(1) c"
        );
        constants.push(c);
    }
    let (lo, hi) = constants
        .iter()
        .fold((f64::MAX, 0.0_f64), |(a, b), &c| (a.min(c), b.max(c)));
    assert!(
        hi / lo <= 20.0,
        "the recovered constant must be stable across four decades of D, not \
         absorb a wrong exponent: {constants:?}"
    );

    // What the shipped fraction buys. Half the mantissa on the Hessian — the bar
    // a Newton step needs — is `ε/D^{3/2} ≤ √ε`, i.e. `D ≥ ε^{1/3}`. The box
    // stops at `D = (1 − F)² = 0.25`, which is enormously inside it.
    let fraction = 0.5_f64;
    let shipped_d = (1.0 - fraction) * (1.0 - fraction);
    let resolution_limited_d = eps.powf(1.0 / 3.0);
    assert!(
        shipped_d > 1.0e4 * resolution_limited_d,
        "the shipped margin D = {shipped_d} must sit orders inside the arithmetic's \
         own limit D = ε^(1/3) = {resolution_limited_d:.3e}; if it ever does not, \
         the fraction has become a numerical constraint and its doc is wrong"
    );
    let kappa_at_margin = fraction / (R * R);
    let manifold = ConstantCurvature::new(2, kappa_at_margin);
    let (_, _, ref_kk) = antipodal_pair_reference(R, kappa_at_margin);
    let (_, _, got_kk) =
        distance_kappa_jet(&manifold, x.view(), y.view()).expect("inside the fold");
    let rel_at_margin = ((got_kk - ref_kk) / ref_kk).abs();
    assert!(
        rel_at_margin <= 1.0e-12,
        "at the box's own upper end the κ-Hessian must carry at least 12 digits; \
         measured relative error {rel_at_margin:.3e}"
    );
}

/// The other half of #2716's arithmetic, and the reason the box cannot be
/// derived pair-by-pair: `D = 1 + 2tμ + t² ≥ 1 − μ²` for EVERY κ, where `μ` is
/// the cosine between `x` and `c`. A pair that is not anti-aligned to within
/// `arcsin(√D)` can never reach that `D` at any curvature — so the fold is not a
/// wall for it at all, and a box that took the exact per-pair minimum would
/// depend on whether two of several hundred pairs happened to line up.
#[test]
fn the_moebius_denominator_is_floored_by_the_pair_angle_2716() {
    for mu in [-0.999_f64, -0.99, -0.95, -0.5, 0.0, 0.7] {
        // Minimize `1 + 2tμ + t²` over t ≥ 0 in closed form: t* = max(0, −μ), so
        // the floor is `1 − μ²` for an obtuse pair and a flat 1 for an acute one
        // — an acute pair never approaches the fold at ANY κ > 0.
        let t_star = (-mu).max(0.0);
        let expected = if mu < 0.0 { 1.0 - mu * mu } else { 1.0 };
        let d_min = 1.0 + 2.0 * t_star * mu + t_star * t_star;
        assert!(
            (d_min - expected).abs() <= 1e-12,
            "μ = {mu}: min over κ of D is {d_min}, expected {expected}"
        );
        // And the shipped geometry agrees: at the minimizing κ the pair is at
        // its closest approach to the cut locus and the distance still evaluates.
        if t_star > 0.0 {
            let r = 0.6_f64;
            let kappa = t_star / (r * r);
            let ang = mu.acos();
            let a = ndarray::array![r, 0.0];
            let b = ndarray::array![r * ang.cos(), r * ang.sin()];
            assert!(
                ConstantCurvature::new(2, kappa)
                    .distance(a.view(), b.view())
                    .is_ok(),
                "μ = {mu}: the closest approach to the fold must still evaluate"
            );
        }
    }
    // The consequence stated as the threshold #2716 derived: reaching the
    // resolution-limited D = ε^{1/3} needs |μ| within 3e-6 of exact
    // anti-alignment, which random data does not supply.
    let d = f64::EPSILON.powf(1.0 / 3.0);
    let required = (1.0_f64 - d).sqrt();
    assert!(
        required > 1.0 - 1.0e-5,
        "reaching D = ε^(1/3) requires |μ| ≥ {required}, i.e. anti-alignment to \
         within {:.3e} rad",
        (1.0 - required * required).sqrt()
    );
}
