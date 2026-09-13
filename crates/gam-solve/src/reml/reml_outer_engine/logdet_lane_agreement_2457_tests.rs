//! #2457 — the value-only and derivative-bearing lanes must price ONE `log|H|`.
//!
//! `certify_outer_optimality` evaluates the outer criterion twice at the same
//! ρ — once at `OuterEvalOrder::Value`, once at `ValueAndGradient` — and
//! `audit_outer_value_agreement` refuses the fit when the two disagree by more
//! than `outer_value_agreement_bound`. The two lanes can reach two different
//! `HessianFactorization`s: the value-only LLT fast path returns the EXACT
//! `Σ ln σ_j` over every eigenvalue, while every derivative-bearing lane prices
//! `log|H|₊` on H's identified subspace
//! (`DenseSpectralOperator::from_symmetric_on_identified_subspace`, #2901 V22)
//! and drops the eigenvalues inside H's rounding band. Wherever H carries such
//! an eigenvalue the two are different functions of ρ, so the LLT is admitted
//! only under a certificate that rules it out.
//!
//! Measured instance of the class — `gam::identifiability`
//! `smooths::constant_curvature_smooth::kappa_zero_fit_recovers_planted_flat_signal`:
//! value-only `−9.8977232379128066e2` against analytic-sample
//! `−9.8976254249264389e2`, a `9.781e-3` gap versus a `1.475e-5` envelope
//! (663×), while the derivative lanes still priced a smooth spectral floor.

use super::*;
use ndarray::{Array1, Array2};

const FIXTURE_DIM: usize = 32;

/// A deterministic, non-trivial orthogonal basis: one Householder reflector
/// `I − 2vvᵀ/vᵀv` with `v = (1, 2, …, n)`.
fn householder_basis(n: usize) -> Array2<f64> {
    let v: Array1<f64> = Array1::from_iter((0..n).map(|i| (i + 1) as f64));
    let vtv = v.dot(&v);
    let mut q = Array2::<f64>::eye(n);
    for i in 0..n {
        for j in 0..n {
            q[[i, j]] -= 2.0 * v[i] * v[j] / vtv;
        }
    }
    q
}

/// `Q diag(σ) Qᵀ`, symmetrized so the LLT sees an exactly symmetric input.
fn spd_with_spectrum(sigma: &[f64]) -> Array2<f64> {
    let n = sigma.len();
    let q = householder_basis(n);
    let mut scaled = q.clone();
    for j in 0..n {
        for i in 0..n {
            scaled[[i, j]] *= sigma[j];
        }
    }
    let h = scaled.dot(&q.t());
    let mut symmetric = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            symmetric[[i, j]] = 0.5 * (h[[i, j]] + h[[j, i]]);
        }
    }
    symmetric
}

/// Every eigenvalue orders of magnitude above H's rounding band.
fn well_conditioned_spectrum() -> Vec<f64> {
    (0..FIXTURE_DIM)
        .map(|i| 10f64.powf(-2.0 + 4.0 * i as f64 / (FIXTURE_DIM - 1) as f64))
        .collect()
}

/// The same spectrum with its smallest eigenvalue at `7·√ε·p`, where the retired
/// smooth floor made the lanes disagree (#2457): small, but far above the
/// rounding band, so it belongs to the identified subspace.
fn small_resolved_spectrum() -> Vec<f64> {
    let mut sigma: Vec<f64> = (0..FIXTURE_DIM - 1)
        .map(|i| 10f64.powf(-2.0 + 4.0 * i as f64 / (FIXTURE_DIM - 2) as f64))
        .collect();
    sigma.push(7.0 * f64::EPSILON.sqrt() * FIXTURE_DIM as f64);
    sigma
}

/// The well-conditioned spectrum with its smallest eigenvalue moved down to
/// `1e-14`, well inside H's rounding band `p·ε·‖H‖₂ ≈ 7e-13`: the identified
/// subspace drops that direction, while an exact LLT would log it.
fn band_touching_spectrum() -> Vec<f64> {
    let mut sigma = well_conditioned_spectrum();
    sigma[0] = 1.0e-14;
    sigma
}

/// Exactly the operator selection `build_dense_assembly` makes on a value-only
/// evaluation: try the LLT fast path, fall back to the identified-subspace
/// spectral operator on any decline.
fn value_lane_logdet(h: &Array2<f64>) -> f64 {
    match DenseCholeskyOperator::from_spd_with_logdet_agreement(h) {
        Ok(operator) => operator.logdet(),
        Err(_) => derivative_lane_logdet(h),
    }
}

fn derivative_lane_logdet(h: &Array2<f64>) -> f64 {
    DenseSpectralOperator::from_symmetric_on_identified_subspace(h, 0)
        .expect("fixture Hessian must decompose")
        .logdet()
}

/// THE CONTRACT. Whatever operator the value-only lane installs must price
/// `log|H|` within the SAME envelope the outer audit applies to the criterion
/// that log-determinant feeds — on a well-conditioned Hessian, on one with a
/// small resolved eigenvalue, and on one with an eigenvalue inside its rounding
/// band.
#[test]
fn value_and_derivative_lanes_price_one_logdet_2457() {
    for (label, sigma) in [
        ("well-conditioned", well_conditioned_spectrum()),
        ("small-resolved", small_resolved_spectrum()),
        ("band-touching", band_touching_spectrum()),
    ] {
        let h = spd_with_spectrum(&sigma);
        let value = value_lane_logdet(&h);
        let derivative = derivative_lane_logdet(&h);
        let gap = (value - derivative).abs();
        let envelope = crate::rho_optimizer::outer_value_agreement_bound(value, derivative);
        assert!(
            gap <= envelope,
            "{label}: the value lane and the derivative lane price different log|H| \
             (value {value:.17e}, derivative {derivative:.17e}, gap {gap:.3e} = {:.1}x the \
             {envelope:.3e} envelope) — the outer criterion is not a function of rho",
            gap / envelope
        );
    }
}

#[test]
fn strong_penalty_roundoff_uses_one_logdet_kernel_2834() {
    // Every eigenvalue is far above the rounding band. The competing source of
    // disagreement is the assembled matrix's condition number, as on the
    // shared/group smooth fixture when its deviation penalty tends to infinity.
    let rotation = ndarray::array![
        [0.5, 0.5, 0.5, 0.5],
        [0.5, -0.5, 0.5, -0.5],
        [0.5, 0.5, -0.5, -0.5],
        [0.5, -0.5, -0.5, 0.5]
    ];
    for strength in [1e10, 1e12] {
        let h = rotation
            .dot(&Array2::from_diag(&ndarray::array![1.0, 2.0, 3.0, strength]))
            .dot(&rotation.t());
        assert!(DenseCholeskyOperator::from_spd_with_logdet_agreement(&h).is_err());
        assert_eq!(value_lane_logdet(&h), derivative_lane_logdet(&h));
    }
    let benign = Array2::from_diag(&ndarray::array![1.0, 2.0, 3.0, 4.0]);
    assert!(DenseCholeskyOperator::from_spd_with_logdet_agreement(&benign).is_ok());
}

/// CONTROL — the fixture must be able to FAIL, or the contract above proves
/// nothing. On the band-touching Hessian the exact full-spectrum log-determinant
/// an admitted LLT would return really does disagree with the identified-subspace
/// criterion by more than the production audit's envelope, so the certificate
/// must decline it; on the well-conditioned Hessian the LLT is admitted.
#[test]
fn only_a_band_touching_hessian_separates_the_two_logdets_2457() {
    let benign = spd_with_spectrum(&well_conditioned_spectrum());
    assert!(
        DenseCholeskyOperator::from_spd_with_logdet_agreement(&benign).is_ok(),
        "a well-conditioned Hessian must keep the LLT fast path — the #2457 guard is \
         one-sided and must not cost the speedup where every eigenvalue is resolved"
    );

    let sigma = band_touching_spectrum();
    let hard = spd_with_spectrum(&sigma);
    let exact_full: f64 = sigma.iter().map(|s| s.ln()).sum();
    let identified = derivative_lane_logdet(&hard);
    let gap = (exact_full - identified).abs();
    let envelope = crate::rho_optimizer::outer_value_agreement_bound(exact_full, identified);
    assert!(
        gap > envelope,
        "the band-touching arm must be an input on which an admitted LLT would be refused \
         by the production audit, or it cannot witness #2457 (gap {gap:.3e}, envelope \
         {envelope:.3e})"
    );
    assert!(
        DenseCholeskyOperator::from_spd_with_logdet_agreement(&hard).is_err(),
        "the fast path must DECLINE a Hessian with an eigenvalue inside its rounding band, \
         rather than return a scalar no derivative lane agrees with"
    );
}
