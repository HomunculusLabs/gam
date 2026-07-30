//! The #2566 `log sigma` curvature certificate: a production curvature value
//! paired with a measured bound on its own error, so a consumer that needs a
//! definite Hessian can refuse on a ratio rather than on a scale cutoff.
//!
//! Split out of `survival/mod.rs` so the #2566 diagnostic can carry its source
//! scanner exemption over ~150 lines of certificate machinery instead of the
//! 6,300-line latent-survival fit math. Allowlisting the parent would exempt the
//! entire fit path, which is precisely the hole the finite-difference ban exists
//! to close.

use super::{
    LATENT_SURVIVAL_PRIMARY_LOG_SIGMA, LatentSurvivalPrimaryPoint,
    latent_survival_row_primary_gradient_hessian,
};
use crate::quadrature::QuadratureContext;
use crate::survival::lognormal_kernel::LatentSurvivalRow;

// FD-OK: #2566 sanctioned diagnostic authority -- differences the GRADIENT to
// CERTIFY the analytic curvature on a separate entry point, never on the fit
// path. The production value comes from the analytic jet; the authority only
// measures the error on it.

/// A `log σ` curvature together with a measured bound on its own error (#2566).
///
/// The curvature is the production value; `estimated_absolute_error` is what an
/// independent authority says it could be wrong by. A consumer that needs a
/// definite Hessian refuses on the ratio rather than on a scale cutoff.
#[derive(Clone, Copy, Debug)]
pub struct CertifiedLogSigmaCurvature {
    /// The production negative-Hessian `∂²/∂(log σ)²` entry.
    pub curvature: f64,
    /// Measured bound on `|curvature − truth|`: the observed disagreement with an
    /// independent finite-difference authority, plus that authority's own
    /// numerical error. Not a proved bound — an estimate, from measurement.
    pub estimated_absolute_error: f64,
}

impl CertifiedLogSigmaCurvature {
    /// Relative error, or `f64::INFINITY` for a zero curvature carrying any
    /// error at all (a zero that could be anything is not a usable zero).
    pub fn relative_error(&self) -> f64 {
        if self.curvature == 0.0 {
            if self.estimated_absolute_error == 0.0 {
                0.0
            } else {
                f64::INFINITY
            }
        } else {
            self.estimated_absolute_error / self.curvature.abs()
        }
    }

    /// Whether the curvature is usable at the caller's relative tolerance.
    pub fn is_trustworthy(&self, relative_tolerance: f64) -> bool {
        self.curvature.is_finite() && self.relative_error() <= relative_tolerance
    }
}

/// `γ_n` roundoff accumulation for `n` floating-point operations.
fn latent_floating_point_gamma(operations: usize) -> f64 {
    let accumulated = operations as f64 * f64::EPSILON;
    accumulated / (1.0 - accumulated)
}

/// Fourth-order Richardson derivative and its local, measured error budget.
///
/// The first uncertainty term is the step-halving truncation estimate; the rest
/// propagates ordinary `γ_n` bounds through the two central differences and their
/// combination. No production derivative path is consulted, which is what makes
/// this an independent authority rather than a restatement.
fn latent_richardson_derivative(
    mut value_at: impl FnMut(f64) -> Result<f64, String>,
    coordinate: f64,
    step: f64,
) -> Result<(f64, f64), String> {
    let coarse_plus = value_at(coordinate + step)?;
    let coarse_minus = value_at(coordinate - step)?;
    let fine_step = 0.5 * step;
    let fine_plus = value_at(coordinate + fine_step)?;
    let fine_minus = value_at(coordinate - fine_step)?;
    let coarse = (coarse_plus - coarse_minus) / (2.0 * step);
    let fine = (fine_plus - fine_minus) / (2.0 * fine_step);
    let value = (4.0 * fine - coarse) / 3.0;

    let gamma = latent_floating_point_gamma(3);
    let coarse_roundoff = gamma * (coarse_plus.abs() + coarse_minus.abs()) / (2.0 * step);
    let fine_roundoff = gamma * (fine_plus.abs() + fine_minus.abs()) / (2.0 * fine_step);
    let combine_roundoff = gamma * (4.0 * fine.abs() + coarse.abs()) / 3.0;
    let uncertainty = (fine - coarse).abs() / 3.0
        + (4.0 * fine_roundoff + coarse_roundoff) / 3.0
        + combine_roundoff;
    Ok((value, uncertainty))
}

/// The `log σ` curvature with a measured error bound attached (#2566).
///
/// # Why this exists
///
/// The curvature is formed as a cancelling difference and becomes NOISE past
/// `log σ ≈ 5.4` — it changes sign between adjacent `0.05` samples there, while
/// the value and gradient channels stay smooth. Beyond that point there is no
/// correct value to return, so a consumer that needs a definite Hessian must be
/// able to refuse. Nothing already returned supports that: the reported
/// `IntegratedExpectationMode` is `ControlledAsymptotic` at both the healthy
/// `log σ = 4` and the sign-inverted `log σ = 7`, so it labels the two identically.
///
/// The discriminator that DOES separate them is a comparison against an
/// independent authority, and on the #2566 fixture it reads `0.853` at
/// `log σ = 4` against `109.765` at `log σ = 6` — two orders of magnitude on
/// exactly the axis where the mode gives no signal.
///
/// # What it costs, and what it is not
///
/// Four gradient samples, each paired with its own value-only Richardson
/// authority: roughly **eight extra row evaluations**. This is for a caller that
/// has decided it needs the guarantee, not for every row of every fit — which is
/// why it is a separate entry point and `latent_survival_row_primary_gradient_hessian`
/// is untouched.
///
/// The result is a MEASURED estimate, not a proved bound. A proved bound on the
/// `ControlledAsymptotic` branch is a much larger piece of work; the reformulation
/// that would remove the need for either is tracked separately.
pub fn latent_survival_log_sigma_curvature_certified(
    quadctx: &QuadratureContext,
    row: &LatentSurvivalRow,
    point: LatentSurvivalPrimaryPoint,
) -> Result<CertifiedLogSigmaCurvature, String> {
    let log_sigma = point.sigma.ln();
    if !log_sigma.is_finite() {
        return Err(format!(
            "certified log-sigma curvature needs a positive finite sigma, got {}",
            point.sigma
        ));
    }
    let point_at = |at: f64| LatentSurvivalPrimaryPoint {
        sigma: at.exp(),
        ..point
    };
    let evaluate = |at: f64| latent_survival_row_primary_gradient_hessian(quadctx, row, point_at(at), true);

    let (_, _, negative_hessian) = evaluate(log_sigma)?;
    let curvature = negative_hessian[[
        LATENT_SURVIVAL_PRIMARY_LOG_SIGMA,
        LATENT_SURVIVAL_PRIMARY_LOG_SIGMA,
    ]];

    // The authority differentiates the GRADIENT, so it never touches the
    // second-order algebra under test. Each gradient sample carries its own error,
    // measured against a value-only Richardson difference at the same point.
    let step = f64::EPSILON.cbrt() * (1.0 + log_sigma.abs());
    let mut sample_error = 0.0_f64;
    let mut gradient_at = |at: f64| -> Result<f64, String> {
        let (_, gradient, _) = evaluate(at)?;
        let analytic = gradient[LATENT_SURVIVAL_PRIMARY_LOG_SIGMA];
        let value_step = f64::EPSILON.cbrt() * (1.0 + at.abs());
        let (value_derivative, value_uncertainty) =
            latent_richardson_derivative(|inner| Ok(evaluate(inner)?.0), at, value_step)?;
        sample_error = sample_error.max((analytic - value_derivative).abs() + value_uncertainty);
        Ok(analytic)
    };
    let (authority, authority_uncertainty) =
        latent_richardson_derivative(&mut gradient_at, log_sigma, step)?;

    // The row API returns the NEGATIVE Hessian, hence the sign.
    let authority_curvature = -authority;
    // Triangle inequality against an authority known to within its own budget:
    // the input error enters the difference divided by the step.
    let estimated_absolute_error = (curvature - authority_curvature).abs()
        + authority_uncertainty
        + sample_error / step;
    Ok(CertifiedLogSigmaCurvature {
        curvature,
        estimated_absolute_error,
    })
}

// END-FD-OK
