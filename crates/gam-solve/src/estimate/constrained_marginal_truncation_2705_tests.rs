//! #2705 group A — the smoothing-corrected covariance of a constrained fit is
//! the TRUNCATION of `Vp`, not `Vp` minus a truncation built for `Vb`.
//!
//! The defect these pin: `beta_covariance_corrected` used to be assembled as
//!
//! ```text
//!     (Σ − GΔGᵀ)  +  (Vp − Σ)  =  Vp − GΔGᵀ,      G, Δ derived from Σ
//! ```
//!
//! where `Σ = Vb` is the ρ̂-conditional covariance and `Vp = Vb + J·V_ρ·Jᵀ` is
//! the ρ-marginal one. Along a coordinate the constraint pins, `(GΔGᵀ)_ii`
//! cancels `Σ_ii` to the last digit, so the residue `(Vp − Σ)_ii` — a
//! second-order, legitimately sign-indefinite increment — becomes the entire
//! published variance and can be negative.
//!
//! The feasible set constrains β and not ρ, so `1_C(β)` factors out of the
//! ρ-integral and the β-marginal of the truncated joint posterior IS the
//! truncation of the β-marginal of the untruncated one. The truncation
//! therefore belongs on `Vp`, with its own lift and its own orthant moments.

use crate::constrained_posterior::{
    constrained_posterior_correction_from_covariance, ConstrainedPosteriorGeometry,
};
use gam_problem::LinearInequalityConstraints;
use ndarray::{array, Array1, Array2};

/// A one-sided bound `β₀ ≥ 0` whose unconstrained centre sits far below it, so
/// the truncation removes essentially all of coordinate 0's variance and the
/// conditional subtraction cancels to the noise floor.
fn pinning_constraints() -> LinearInequalityConstraints {
    LinearInequalityConstraints::new(array![[1.0_f64, 0.0]], array![0.0_f64])
        .expect("a 1x2 inequality system with a matching bound is well formed")
}

/// `Vb`, `centre`, and a smoothing increment whose (0,0) entry is NEGATIVE —
/// the shape a cubature correction `φ̂·E_ρ[H(ρ)⁻¹] + Cov_ρ[β̂] − φ̂·H_opt⁻¹`
/// genuinely takes, and the shape that makes the old composition fail.
fn fixture() -> (Array2<f64>, Array1<f64>, Array2<f64>) {
    let conditional = array![[4.0e-2_f64, 6.0e-3], [6.0e-3, 9.0e-2]];
    // Twelve pre-truncation standard deviations below the bound: `Φ̄(−12)` is
    // below double-precision resolution, so the truncated coordinate keeps
    // essentially none of its variance and `Σ_00 − (GΔGᵀ)_00` is a cancellation.
    let centre = array![-12.0 * conditional[[0, 0]].sqrt(), 0.5];
    let smoothing_increment = array![[-3.0e-9_f64, 1.0e-9], [1.0e-9, 5.0e-3]];
    (conditional, centre, smoothing_increment)
}

fn truncate(
    covariance: &Array2<f64>,
    centre: &Array1<f64>,
    constraints: &LinearInequalityConstraints,
) -> Array2<f64> {
    let correction = constrained_posterior_correction_from_covariance(covariance, centre, constraints)
        .expect("the orthant moments of a well-conditioned 1-row face converge")
        .expect("a bound twelve standard deviations away is visible at f64 resolution");
    correction.apply_to_covariance(covariance)
}

/// The old composition is reproduced here explicitly so the regression cannot
/// pass by accident: on this fixture it publishes a NEGATIVE variance, which is
/// exactly what `se_from_covariance` refused on `y ~ s(x, shape=convex)`.
#[test]
fn conditional_lift_subtracted_from_the_marginal_covariance_goes_negative_2705() {
    let (conditional, centre, smoothing_increment) = fixture();
    let constraints = pinning_constraints();

    let conditional_truncated = truncate(&conditional, &centre, &constraints);
    assert!(
        conditional_truncated[[0, 0]] >= 0.0,
        "the CONDITIONAL truncation is well posed on its own covariance: {:.6e}",
        conditional_truncated[[0, 0]]
    );
    assert!(
        conditional_truncated[[0, 0]] < 1e-4 * conditional[[0, 0]],
        "the bound must pin coordinate 0 for this fixture to be the one #2705 measured: \
         {:.6e} against {:.6e}",
        conditional_truncated[[0, 0]],
        conditional[[0, 0]]
    );

    let old_composition = &conditional_truncated + &smoothing_increment;
    assert!(
        old_composition[[0, 0]] < 0.0,
        "the pre-#2705 order must produce the negative variance this issue reports, \
         otherwise this test is not measuring the defect: {:.6e}",
        old_composition[[0, 0]]
    );
}

/// The fixed composition: truncate `Vp`, and the published matrix is a genuine
/// truncated-Gaussian covariance — non-negative diagonal, and no larger than the
/// untruncated marginal it came from, because truncating a Gaussian to a convex
/// set cannot increase its covariance.
#[test]
fn marginal_covariance_truncated_at_itself_is_a_valid_covariance_2705() {
    let (conditional, centre, smoothing_increment) = fixture();
    let constraints = pinning_constraints();

    let marginal = &conditional + &smoothing_increment;
    let published = truncate(&marginal, &centre, &constraints);

    for index in 0..published.nrows() {
        assert!(
            published[[index, index]] >= 0.0,
            "truncating the marginal covariance at itself must leave a non-negative \
             variance at {index}: {:.6e}",
            published[[index, index]]
        );
        assert!(
            published[[index, index]] <= marginal[[index, index]] * (1.0 + 1e-9),
            "truncation cannot increase a variance: {:.6e} against {:.6e} at {index}",
            published[[index, index]],
            marginal[[index, index]]
        );
    }

    // Quadratic-form check on the whole matrix rather than the diagonal alone:
    // `Cov_π ⪯ Vp` in the PSD order, so `Vp − Cov_π` is PSD.
    let residual = &marginal - &published;
    for direction in [array![1.0_f64, 0.0], array![0.0, 1.0], array![1.0, 1.0], array![1.0, -1.0]] {
        let quadratic = direction.dot(&residual.dot(&direction));
        assert!(
            quadratic >= -1e-12,
            "the variance truncation removed must be non-negative in every direction: \
             {quadratic:.6e}"
        );
    }
}

/// A geometry whose moments were DECLINED never truncated the conditional
/// covariance, so the marginal one must be published untruncated too — the two
/// estimands stay consistent about whether the constraint was representable.
#[test]
fn declined_moments_leave_the_marginal_covariance_untouched_2705() {
    let (conditional, _, smoothing_increment) = fixture();
    let constraints = pinning_constraints();
    let geometry = ConstrainedPosteriorGeometry::with_decline(
        constraints,
        array![0.0_f64, 0.5],
        crate::constrained_posterior::ConePosteriorMomentDecline {
            ambient_precision_failure: "probe: the ambient route declined".to_string(),
            properness: crate::constrained_posterior::ConePropernessEvidence::CertificationFailed {
                reason: "probe: properness was not certified".to_string(),
            },
        },
    );

    let marginal = &conditional + &smoothing_increment;
    let mut published = marginal.clone();
    super::optimizer::apply_marginal_constraint_truncation(&geometry, &mut published)
        .expect("a declined geometry is not an error at this boundary");
    assert_eq!(
        published, marginal,
        "a declined constrained posterior must leave the marginal covariance bit-identical"
    );
}
