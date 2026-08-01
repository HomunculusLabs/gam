//! Root-cause regression for #2733.
//!
//! Two SAS fixtures died in the inner loop with
//!
//! ```text
//! PIRLS row geometry is not representable at row 0:
//! inverse-link value/derivative evaluated from eta=-59.45330695027052 produced 0.0
//! ```
//!
//! The `0.0` was the *mean*. The SAS link is `mu = Phi(z(eta))` — literally the
//! probit CDF composed with a smooth increasing reparameterization — and the
//! binomial deviance row needs `ln mu` and `ln(1 - mu)`. Every standard binomial
//! link already had a log-space route to that pair (`probit_binomial_geometry`
//! and friends); SAS had none, so it went through the mean, and the mean
//! underflows to exactly `0.0` roughly forty units of `z` before `ln mu` stops
//! being an ordinary finite number. `ln Phi(-59.45) = -1774.6` — nothing about
//! that row is unrepresentable.
//!
//! The gates below are all *reference* comparisons: SAS at its identity
//! parameters must reproduce the standard probit link bitwise, and away from
//! the identity the row's own value and score must agree with an independent
//! `ln Phi` evaluation and with a finite difference of the value it reports.
//! Each carries a positive control asserting that the mean route really does
//! collapse at the same point, so none of them can pass vacuously.

use super::*;
use approx::assert_relative_eq;
use gam_problem::{GlmLikelihoodSpec, InverseLink, LikelihoodSpec, ResponseFamily, StandardLink};

/// The exact eta reported by both #2733 fixtures, bit for bit.
const REPORTED_ETA: f64 = -59.45330695027052;

/// SAS `(epsilon, log_delta)` pairs whose latent argument at [`REPORTED_ETA`]
/// is deep enough in the left tail that `mu` underflows to exactly `0.0`.
/// `(0, 0)` is the identity (`z = eta = -59.45`); the other two compress and
/// expand the latent scale in opposite directions. The collapse is asserted,
/// not assumed, by `sas_mean_route_collapses_at_the_reported_eta_2733`.
const SATURATING_PARAMETERS: [(f64, f64); 3] = [(0.0, 0.0), (-0.55, 0.25), (0.9, 0.7)];

fn binomial(inverse_link: &InverseLink) -> GlmLikelihoodSpec {
    GlmLikelihoodSpec::canonical(LikelihoodSpec::new(
        ResponseFamily::Binomial,
        inverse_link.clone(),
    ))
}

fn row(inverse_link: &InverseLink, y: f64, eta: f64) -> Result<DevianceEtaRow, EstimationError> {
    deviance_eta_row_with_log_measure_scale(
        0,
        y,
        eta,
        &binomial(inverse_link),
        inverse_link,
        1.0,
        0.0,
    )
}

fn sas_link(epsilon: f64, log_delta: f64) -> InverseLink {
    InverseLink::Sas(
        crate::mixture_link::sas_link_state_from_raw(epsilon, log_delta).expect("finite SAS state"),
    )
}

/// The mean route's collapse, asserted directly. This is the positive control
/// for every gate below: it pins that `mu` (and its derivative) really are the
/// unrepresentable `0.0` the issue reported, so a row that nonetheless returns
/// a finite value can only have come from the log-space route.
#[test]
fn sas_mean_route_collapses_at_the_reported_eta_2733() {
    for (epsilon, log_delta) in SATURATING_PARAMETERS {
        let link = sas_link(epsilon, log_delta);
        let (mu, d1) = crate::mixture_link::inverse_link_mu_d1_for_inverse_link(&link, REPORTED_ETA)
            .expect("finite SAS eta");
        assert_eq!(
            mu, 0.0,
            "SAS mean must underflow to exactly zero at the reported eta \
             (eps={epsilon}, log_delta={log_delta}); if it no longer does, this \
             fixture has stopped testing the saturating regime"
        );
        assert_eq!(d1, 0.0, "and so must its derivative (eps={epsilon})");
    }
}

/// SAS at `(epsilon, log_delta) = (0, 0)` *is* the standard probit link: the
/// latent map is `z = sinh(asinh(eta)) = eta`. So the two must produce the same
/// deviance row bitwise. Before the fix they did not merely differ — probit
/// returned a row and SAS refused, which is the defect in its sharpest form:
/// the same mathematics, refused under one name and accepted under the other.
#[test]
fn sas_identity_parameters_reproduce_probit_deviance_row_bitwise_2733() {
    let sas = sas_link(0.0, 0.0);
    let probit = InverseLink::Standard(StandardLink::Probit);
    // A ladder from the well-conditioned interior out through the underflow of
    // `mu` (`eta < -38`), past the reported eta, to the far tail.
    let etas = [
        0.0,
        -1.25,
        -8.0,
        -37.0,
        -39.0,
        REPORTED_ETA,
        -150.0,
        -1.0e4,
        8.0,
        39.0,
        1.0e4,
    ];
    let mut saturated = 0usize;
    for eta in etas {
        for y in [0.0, 1.0, 0.25] {
            let expected = row(&probit, y, eta).expect("probit row");
            let actual = row(&sas, y, eta).expect("SAS row at identity parameters");
            assert_eq!(
                actual.half_deviance.to_bits(),
                expected.half_deviance.to_bits(),
                "SAS(0,0) half-deviance must equal probit's bitwise at eta={eta}, y={y}: \
                 got {}, want {}",
                actual.half_deviance,
                expected.half_deviance
            );
            assert_eq!(
                actual.eta_score.to_bits(),
                expected.eta_score.to_bits(),
                "SAS(0,0) eta score must equal probit's bitwise at eta={eta}, y={y}: \
                 got {}, want {}",
                actual.eta_score,
                expected.eta_score
            );
        }
        let (mu, _) = crate::mixture_link::inverse_link_mu_d1_for_inverse_link(&sas, eta)
            .expect("finite SAS eta");
        if mu == 0.0 || mu == 1.0 {
            saturated += 1;
        }
    }
    assert!(
        saturated >= 4,
        "non-vacuity: the ladder must contain rows whose mean has saturated to \
         exactly 0 or 1 — those are the rows the mean route cannot express — but \
         only {saturated} of {} did",
        etas.len()
    );
}

/// Away from the identity parameters the row is still exactly the probit
/// geometry at the latent argument `z`. Check the value against an independent
/// `ln Phi` evaluation of the Bernoulli cross-entropy — not against another
/// call into the same code path.
#[test]
fn sas_saturated_half_deviance_equals_log_phi_cross_entropy_2733() {
    for (epsilon, log_delta) in SATURATING_PARAMETERS {
        let link = sas_link(epsilon, log_delta);
        let (z, _) = crate::mixture_link::sas_latent_probit_argument(
            REPORTED_ETA,
            epsilon,
            log_delta,
        )
        .expect("finite SAS latent argument");
        let (log_mu, _) = gam_math::probability::signed_probit_logcdf_and_mills_ratio(z);
        let (log_one_minus_mu, _) =
            gam_math::probability::signed_probit_logcdf_and_mills_ratio(-z);
        assert!(
            log_mu.is_finite() && log_mu < -100.0,
            "fixture must sit in the tail the mean cannot express \
             (eps={epsilon}, log_delta={log_delta}): ln mu = {log_mu}"
        );
        // Bernoulli half-deviance at a 0/1 response is the cross-entropy: the
        // saturated log-likelihood is exactly zero there.
        let one = row(&link, 1.0, REPORTED_ETA).expect("SAS row, y=1");
        assert_relative_eq!(one.half_deviance, -log_mu, max_relative = 1.0e-14);
        let zero = row(&link, 0.0, REPORTED_ETA).expect("SAS row, y=0");
        assert_relative_eq!(
            zero.half_deviance,
            -log_one_minus_mu,
            max_relative = 1.0e-14
        );
    }
}

/// The score channel must be the derivative of the value channel the same row
/// reported — including inside the saturating band, where the mean route had
/// nothing at all to differentiate.
#[test]
fn sas_saturated_eta_score_matches_finite_difference_of_its_own_value_2733() {
    for (epsilon, log_delta) in [(0.0, 0.0), (0.38, -0.30), (-0.55, 0.25)] {
        let link = sas_link(epsilon, log_delta);
        for y in [0.0, 1.0, 0.25] {
            for eta in [REPORTED_ETA, -45.0, -12.0, -2.0, 3.0] {
                let h = 1.0e-6 * eta.abs();
                let centre = row(&link, y, eta).expect("SAS centre row");
                let plus = row(&link, y, eta + h).expect("SAS plus row").half_deviance;
                let minus = row(&link, y, eta - h).expect("SAS minus row").half_deviance;
                let finite_difference = (plus - minus) / (2.0 * h);
                assert!(
                    centre.half_deviance.is_finite() && centre.eta_score.is_finite(),
                    "SAS row must stay representable at eta={eta} \
                     (eps={epsilon}, log_delta={log_delta})"
                );
                assert_relative_eq!(
                    centre.eta_score,
                    finite_difference,
                    max_relative = 1.0e-6,
                    epsilon = 1.0e-9
                );
            }
        }
    }
}

/// `sas_latent_probit_argument` is the pair the log-space route is built on;
/// its slope must be the `dz/deta` the existing mean/derivative evaluator
/// already uses, or the score above would be the derivative of a different
/// surface than the value. Compare `d1 = phi(z) * dz` against the production
/// `sas_inverse_link_mu_d1` in the interior, where the mean is still
/// unsaturated and that comparison is meaningful.
#[test]
fn sas_latent_slope_reproduces_the_production_mean_derivative_2733() {
    for (epsilon, log_delta) in [(0.0, 0.0), (0.38, -0.30), (-0.55, 0.25), (0.9, 0.7)] {
        let link = sas_link(epsilon, log_delta);
        // Kept narrow on purpose. The mean route saturates *early* under an
        // expanded latent scale — at (0.9, 0.7) it is already exactly 1.0 by
        // eta = 2.5 — and this gate is the one place that needs the mean route
        // to still work, so it must stay inside the band where it does. The
        // saturating band is covered by the four gates above.
        for eta in [-1.0, -0.25, 0.0, 0.3, 1.0] {
            let (mu, d1) =
                crate::mixture_link::inverse_link_mu_d1_for_inverse_link(&link, eta)
                    .expect("finite SAS eta");
            assert!(
                mu > 0.0 && mu < 1.0 && d1 > 0.0,
                "interior control: the mean route must still work at eta={eta}"
            );
            let (z, dz) =
                crate::mixture_link::sas_latent_probit_argument(eta, epsilon, log_delta)
                    .expect("finite SAS latent argument");
            let reconstructed = gam_math::probability::normal_pdf(z) * dz;
            assert_relative_eq!(reconstructed, d1, max_relative = 1.0e-13);
        }
    }
}
