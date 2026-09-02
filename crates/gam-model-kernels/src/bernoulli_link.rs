//! Cancellation-free natural-coordinate derivatives for Bernoulli links.
//!
//! A bounded inverse-link jet `(mu, mu', mu'', mu''')` is not enough for a
//! numerically honest Bernoulli likelihood: either `mu` or `1 - mu` rounds to
//! an endpoint in the tails, precisely where the corresponding log probability
//! and score can still be finite and informative. This module carries the two
//! log-probability derivative towers directly. It is the single kernel used by
//! scalar and separable vector-response Bernoulli families.

use gam_problem::EstimationError;
use gam_solve::mixture_link::inverse_link_jet_for_inverse_link;
use gam_spec::{InverseLink, StandardLink};

/// Natural-coordinate derivative tower for a Bernoulli inverse link.
///
/// `log_mu[j]` and `log_one_minus_mu[j]` are the `j`th derivatives with
/// respect to the linear predictor for `j = 0, 1, 2, 3`. `log_fisher` is
/// `log((d mu / d eta)^2 / (mu (1 - mu)))`, evaluated without reconstructing
/// a rounded endpoint probability.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BernoulliNaturalJet {
    pub mu: f64,
    pub log_mu: [f64; 4],
    pub log_one_minus_mu: [f64; 4],
    pub log_fisher: f64,
}

#[inline]
fn row_geometry_error(
    row: usize,
    quantity: &'static str,
    eta: f64,
    value: f64,
) -> EstimationError {
    EstimationError::PirlsRowGeometryUnrepresentable {
        row,
        quantity,
        eta,
        value,
    }
}

#[inline]
fn logit_natural_jet(eta: f64) -> BernoulliNaturalJet {
    let tail = (-eta.abs()).exp();
    let (mu, one_minus_mu) = if eta >= 0.0 {
        let q = tail / (1.0 + tail);
        (1.0 - q, q)
    } else {
        let p = tail / (1.0 + tail);
        (p, 1.0 - p)
    };
    let curvature = mu * one_minus_mu;
    let third = curvature * (mu - one_minus_mu);
    BernoulliNaturalJet {
        mu,
        log_mu: [
            -gam_linalg::utils::stable_softplus(-eta),
            one_minus_mu,
            -curvature,
            third,
        ],
        log_one_minus_mu: [
            -gam_linalg::utils::stable_softplus(eta),
            -mu,
            -curvature,
            third,
        ],
        log_fisher: -gam_linalg::utils::stable_softplus(eta)
            - gam_linalg::utils::stable_softplus(-eta),
    }
}

#[inline]
fn probit_natural_jet(eta: f64) -> BernoulliNaturalJet {
    let left = gam_math::probability::normal_logcdf_derivatives(eta);
    let right_at_neg_eta = gam_math::probability::normal_logcdf_derivatives(-eta);
    let log_pdf = if eta.abs() <= f64::MAX.sqrt() {
        -0.5 * eta * eta - 0.5 * (2.0 * std::f64::consts::PI).ln()
    } else {
        f64::NEG_INFINITY
    };
    BernoulliNaturalJet {
        mu: left[0].exp(),
        log_mu: [left[0], left[1], left[2], left[3]],
        log_one_minus_mu: [
            right_at_neg_eta[0],
            -right_at_neg_eta[1],
            right_at_neg_eta[2],
            -right_at_neg_eta[3],
        ],
        log_fisher: 2.0 * log_pdf - left[0] - right_at_neg_eta[0],
    }
}

#[inline]
fn cloglog_natural_jet(eta: f64) -> BernoulliNaturalJet {
    let x = eta.exp();
    if x == f64::INFINITY {
        return BernoulliNaturalJet {
            mu: 1.0,
            log_mu: [0.0; 4],
            log_one_minus_mu: [f64::NEG_INFINITY; 4],
            log_fisher: f64::NEG_INFINITY,
        };
    }
    if x == 0.0 {
        return BernoulliNaturalJet {
            mu: 0.0,
            log_mu: [eta, 1.0, 0.0, 0.0],
            log_one_minus_mu: [0.0; 4],
            log_fisher: eta,
        };
    }
    let mu = -(-x).exp_m1();
    let log_mu = if x < 0.5 {
        eta + (mu / x).ln()
    } else {
        mu.ln()
    };
    let h = if x < 1.0 {
        x / x.exp_m1()
    } else {
        let exp_neg_x = (-x).exp();
        x * exp_neg_x / (1.0 - exp_neg_x)
    };
    let a = 1.0 - x - h;
    let d2_log_mu = h * a;
    let d3_log_mu = h * (a * a - x - h * a);
    BernoulliNaturalJet {
        mu,
        log_mu: [log_mu, h, d2_log_mu, d3_log_mu],
        log_one_minus_mu: [-x, -x, -x, -x],
        log_fisher: 2.0 * eta - x - log_mu,
    }
}

#[inline]
fn loglog_natural_jet(eta: f64) -> BernoulliNaturalJet {
    let mirrored = cloglog_natural_jet(-eta);
    BernoulliNaturalJet {
        mu: mirrored.log_one_minus_mu[0].exp(),
        log_mu: [
            mirrored.log_one_minus_mu[0],
            -mirrored.log_one_minus_mu[1],
            mirrored.log_one_minus_mu[2],
            -mirrored.log_one_minus_mu[3],
        ],
        log_one_minus_mu: [
            mirrored.log_mu[0],
            -mirrored.log_mu[1],
            mirrored.log_mu[2],
            -mirrored.log_mu[3],
        ],
        log_fisher: mirrored.log_fisher,
    }
}

#[inline]
fn cauchit_natural_jet(eta: f64) -> BernoulliNaturalJet {
    let (mu, one_minus_mu) = if eta > 0.0 {
        let q = (eta.recip()).atan() / std::f64::consts::PI;
        (1.0 - q, q)
    } else if eta < 0.0 {
        let p = (-eta.recip()).atan() / std::f64::consts::PI;
        (p, 1.0 - p)
    } else {
        (0.5, 0.5)
    };
    let abs_eta = eta.abs();
    let log_one_plus_eta_sq = if abs_eta <= f64::MAX.sqrt() {
        (eta * eta).ln_1p()
    } else {
        2.0 * abs_eta.ln() + eta.recip().powi(2).ln_1p()
    };
    let log_d1 = -std::f64::consts::PI.ln() - log_one_plus_eta_sq;
    let ratio = if abs_eta <= 1.0 {
        eta / (1.0 + eta * eta)
    } else {
        1.0 / (eta + eta.recip())
    };
    let d2_over_d1 = -2.0 * ratio;
    let inv_one_plus_sq = if abs_eta <= 1.0 {
        1.0 / (1.0 + eta * eta)
    } else {
        let inv = eta.recip();
        inv * inv / (1.0 + inv * inv)
    };
    let d3_over_d1 = inv_one_plus_sq * (6.0 * (eta * ratio) - 2.0 * inv_one_plus_sq);
    let d1_over_mu = (log_d1 - mu.ln()).exp();
    let d1_over_q = (log_d1 - one_minus_mu.ln()).exp();
    let left_d2_ratio = d2_over_d1 * d1_over_mu;
    let right_d2_ratio = d2_over_d1 * d1_over_q;
    BernoulliNaturalJet {
        mu,
        log_mu: [
            mu.ln(),
            d1_over_mu,
            left_d2_ratio - d1_over_mu * d1_over_mu,
            d3_over_d1 * d1_over_mu - 3.0 * d1_over_mu * left_d2_ratio
                + 2.0 * d1_over_mu.powi(3),
        ],
        log_one_minus_mu: [
            one_minus_mu.ln(),
            -d1_over_q,
            -right_d2_ratio - d1_over_q * d1_over_q,
            -d3_over_d1 * d1_over_q - 3.0 * d1_over_q * right_d2_ratio
                - 2.0 * d1_over_q.powi(3),
        ],
        log_fisher: 2.0 * log_d1 - mu.ln() - one_minus_mu.ln(),
    }
}

#[inline]
fn generic_natural_jet(
    row: usize,
    eta: f64,
    link: &InverseLink,
) -> Result<BernoulliNaturalJet, EstimationError> {
    let jet = inverse_link_jet_for_inverse_link(link, eta)?;
    if !(jet.mu.is_finite()
        && jet.mu > 0.0
        && jet.mu < 1.0
        && jet.d1.is_finite()
        && jet.d1 > 0.0
        && jet.d2.is_finite()
        && jet.d3.is_finite())
    {
        return Err(row_geometry_error(
            row,
            "bounded-family inverse-link jet",
            eta,
            jet.mu,
        ));
    }
    let mu = jet.mu;
    let q = 1.0 - mu;
    let r1 = jet.d1 / mu;
    let r2 = jet.d2 / mu;
    let r3 = jet.d3 / mu;
    let s1 = jet.d1 / q;
    let s2 = jet.d2 / q;
    let s3 = jet.d3 / q;
    Ok(BernoulliNaturalJet {
        mu,
        log_mu: [
            mu.ln(),
            r1,
            r2 - r1 * r1,
            r3 - 3.0 * r1 * r2 + 2.0 * r1.powi(3),
        ],
        log_one_minus_mu: [
            (-mu).ln_1p(),
            -s1,
            -s2 - s1 * s1,
            -s3 - 3.0 * s1 * s2 - 2.0 * s1.powi(3),
        ],
        log_fisher: 2.0 * jet.d1.ln() - mu.ln() - q.ln(),
    })
}

/// Evaluate a Bernoulli inverse link as two cancellation-free log-probability
/// derivative towers.
///
/// The standard bounded links have dedicated tail kernels. Parameterized
/// bounded links use the central inverse-link jet and are rejected if a trial
/// point cannot represent an interior probability with finite derivatives.
/// Identity and log links are not Bernoulli links and are rejected by the same
/// domain contract rather than silently clamped.
pub fn bernoulli_natural_jet(
    row: usize,
    eta: f64,
    link: &InverseLink,
) -> Result<BernoulliNaturalJet, EstimationError> {
    match link {
        InverseLink::Standard(StandardLink::Logit) => Ok(logit_natural_jet(eta)),
        InverseLink::Standard(StandardLink::Probit) => Ok(probit_natural_jet(eta)),
        InverseLink::Standard(StandardLink::CLogLog) => Ok(cloglog_natural_jet(eta)),
        InverseLink::Standard(StandardLink::LogLog) => Ok(loglog_natural_jet(eta)),
        InverseLink::Standard(StandardLink::Cauchit) => Ok(cauchit_natural_jet(eta)),
        _ => generic_natural_jet(row, eta, link),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn logit_and_cloglog_keep_informative_log_tails() {
        let logit = bernoulli_natural_jet(
            0,
            1_000.0,
            &InverseLink::Standard(StandardLink::Logit),
        )
        .expect("logit tail");
        assert_eq!(logit.mu, 1.0);
        assert_eq!(logit.log_one_minus_mu[0], -1_000.0);
        assert_eq!(logit.log_one_minus_mu[1], -1.0);

        let cloglog = bernoulli_natural_jet(
            0,
            -1_000.0,
            &InverseLink::Standard(StandardLink::CLogLog),
        )
        .expect("cloglog tail");
        assert_eq!(cloglog.mu, 0.0);
        assert_eq!(cloglog.log_mu[0], -1_000.0);
        assert_eq!(cloglog.log_mu[1], 1.0);
    }

    #[test]
    fn loglog_is_exact_cloglog_mirror() {
        for eta in [-12.0, -1.25, 0.0, 2.5, 12.0] {
            let left = bernoulli_natural_jet(
                0,
                eta,
                &InverseLink::Standard(StandardLink::LogLog),
            )
            .expect("loglog jet");
            let right = bernoulli_natural_jet(
                0,
                -eta,
                &InverseLink::Standard(StandardLink::CLogLog),
            )
            .expect("cloglog jet");
            assert_eq!(left.log_mu[0], right.log_one_minus_mu[0]);
            assert_eq!(left.log_one_minus_mu[0], right.log_mu[0]);
            assert_eq!(left.log_fisher, right.log_fisher);
        }
    }

    #[test]
    fn identity_is_not_accepted_as_a_bernoulli_link() {
        let error = bernoulli_natural_jet(
            7,
            0.5,
            &InverseLink::Standard(StandardLink::Identity),
        )
        .expect_err("identity must not enter a Bernoulli likelihood");
        assert!(matches!(
            error,
            EstimationError::PirlsRowGeometryUnrepresentable { row: 7, .. }
        ));
    }
}
