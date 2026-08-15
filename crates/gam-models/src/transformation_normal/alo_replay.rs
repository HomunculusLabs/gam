use super::chart::{CtnRowBases, CtnRowFloors, ctn_component_sensitivity, ctn_row_geometry};
use crate::inference::model::TransformationNormalParameterization;
use ndarray::{Array1, Array2, ArrayView1};

/// Complete local state for one saved transformation-normal likelihood row.
pub struct TransformationNormalAloRowInput<'a> {
    pub response_value_basis: &'a [f64],
    pub response_derivative_basis: &'a [f64],
    pub response_lower_basis: &'a [f64],
    pub response_upper_basis: &'a [f64],
    pub alpha: &'a [f64],
    pub additive_offset: f64,
    pub response_floor_offset: f64,
    pub response_lower_floor_offset: f64,
    pub response_upper_floor_offset: f64,
    pub prior_weight: f64,
}

/// Exact negative-log-likelihood derivatives in the affine local coordinates
/// `alpha_k(x) = covariate_row(x) beta_k`.
#[derive(Clone, Debug, PartialEq)]
pub struct TransformationNormalAloRowGeometry {
    pub negative_log_likelihood: f64,
    pub nll_score: Array1<f64>,
    pub observed_hessian: Array2<f64>,
}

fn validate_row(input: &TransformationNormalAloRowInput<'_>) -> Result<usize, String> {
    let dimension = input.alpha.len();
    if dimension == 0
        || input.response_value_basis.len() != dimension
        || input.response_derivative_basis.len() != dimension
        || input.response_lower_basis.len() != dimension
        || input.response_upper_basis.len() != dimension
    {
        return Err(format!(
            "transformation-normal ALO row dimension mismatch: alpha={dimension}, value={}, derivative={}, lower={}, upper={}",
            input.response_value_basis.len(),
            input.response_derivative_basis.len(),
            input.response_lower_basis.len(),
            input.response_upper_basis.len(),
        ));
    }
    if !input.prior_weight.is_finite() || input.prior_weight < 0.0 {
        return Err(format!(
            "transformation-normal ALO prior weight must be finite and non-negative, got {}",
            input.prior_weight
        ));
    }
    if input
        .response_value_basis
        .iter()
        .chain(input.response_derivative_basis)
        .chain(input.response_lower_basis)
        .chain(input.response_upper_basis)
        .chain(input.alpha)
        .copied()
        .chain([
            input.additive_offset,
            input.response_floor_offset,
            input.response_lower_floor_offset,
            input.response_upper_floor_offset,
        ])
        .any(|value| !value.is_finite())
    {
        return Err("transformation-normal ALO row state must be finite".to_string());
    }
    Ok(dimension)
}

/// Replay one row of the fitted finite-support SCOP likelihood.
///
/// This is the row factorization of the same score and negative Hessian used by
/// `TransformationNormalFamily`: every component is affine in direct-alpha
/// coordinates, the monotonicity derivative floor is exact, and both
/// transformed support endpoints contribute through the normalized Gaussian
/// mass. Feasibility of the shape coordinates is owned by the fitted model's
/// Khatri-Rao cone before this row replay is called.
pub fn transformation_normal_alo_row_geometry(
    input: TransformationNormalAloRowInput<'_>,
) -> Result<TransformationNormalAloRowGeometry, String> {
    let dimension = validate_row(&input)?;
    if input.prior_weight == 0.0 {
        return Ok(TransformationNormalAloRowGeometry {
            negative_log_likelihood: 0.0,
            nll_score: Array1::zeros(dimension),
            observed_hessian: Array2::zeros((dimension, dimension)),
        });
    }

    // One chart, one evaluator (gam#2680): the saved-model ALO replay reads the
    // same coefficients as the fit and must read them the same way.
    let chart = TransformationNormalParameterization::DirectAlpha;
    let geometry = ctn_row_geometry(
        chart,
        ArrayView1::from(input.alpha),
        CtnRowBases {
            value: ArrayView1::from(input.response_value_basis),
            derivative: ArrayView1::from(input.response_derivative_basis),
            lower: ArrayView1::from(input.response_lower_basis),
            upper: ArrayView1::from(input.response_upper_basis),
        },
        CtnRowFloors {
            additive_offset: input.additive_offset,
            value_floor: input.response_floor_offset,
            lower_floor: input.response_lower_floor_offset,
            upper_floor: input.response_upper_floor_offset,
        },
    );
    let (h, h_prime, lower, upper) = (
        geometry.h,
        geometry.h_prime,
        geometry.lower,
        geometry.upper,
    );
    if !(h.is_finite() && h_prime.is_finite() && lower.is_finite() && upper.is_finite()) {
        return Err(format!(
            "transformation-normal ALO row transform is non-finite: h={h}, h_prime={h_prime}, lower={lower}, upper={upper}"
        ));
    }
    if h_prime <= 0.0 {
        return Err(format!(
            "transformation-normal ALO row derivative must be positive, got {h_prime}"
        ));
    }
    // gam#2600: the fitted likelihood is the untruncated MLT density, so the ALO
    // replay uses the same one — `f(y) = φ(h)·h'`, with no renormalization by
    // the mass between the saved support endpoints.
    let weight = input.prior_weight;
    let negative_log_likelihood =
        weight * (0.5 * h * h + 0.5 * (2.0 * std::f64::consts::PI).ln() - h_prime.ln());

    let mut dh = vec![0.0; dimension];
    let mut dh_prime = vec![0.0; dimension];
    let mut dlower = vec![0.0; dimension];
    let mut dupper = vec![0.0; dimension];
    for component in 0..dimension {
        dh[component] =
            ctn_component_sensitivity(chart, ArrayView1::from(input.response_value_basis), component);
        dh_prime[component] = ctn_component_sensitivity(
            chart,
            ArrayView1::from(input.response_derivative_basis),
            component,
        );
        dlower[component] =
            ctn_component_sensitivity(chart, ArrayView1::from(input.response_lower_basis), component);
        dupper[component] =
            ctn_component_sensitivity(chart, ArrayView1::from(input.response_upper_basis), component);
    }

    let inverse_h_prime = 1.0 / h_prime;
    let inverse_h_prime_squared = inverse_h_prime * inverse_h_prime;
    let mut nll_score = Array1::<f64>::zeros(dimension);
    let mut observed_hessian = Array2::<f64>::zeros((dimension, dimension));
    for left in 0..dimension {
        nll_score[left] = weight * (h * dh[left] - dh_prime[left] * inverse_h_prime);
        for right in 0..dimension {
            observed_hessian[[left, right]] = weight
                * (dh[left] * dh[right]
                    + dh_prime[left] * dh_prime[right] * inverse_h_prime_squared);
        }
    }
    if !negative_log_likelihood.is_finite()
        || nll_score.iter().any(|value| !value.is_finite())
        || observed_hessian.iter().any(|value| !value.is_finite())
    {
        return Err("transformation-normal ALO row geometry is non-finite".to_string());
    }
    Ok(TransformationNormalAloRowGeometry {
        negative_log_likelihood,
        nll_score,
        observed_hessian,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transformation_normal::TRANSFORMATION_MONOTONICITY_EPS;

    /// gam#2600: the replayed row density is the untruncated MLT density
    /// `log φ(h) + log h'`. The endpoint bases and floors the row input carries
    /// still define the certified support of the saved model, but they are no
    /// longer a term of the likelihood — so this independent reconstruction
    /// does not mention them, and the fixture below feeds deliberately
    /// asymmetric ones (`[1.0, 0.1]` / `[1.0, 0.9]`, floors `−0.04` / `0.06`)
    /// so that any endpoint contribution surviving in the production path shows
    /// up here as a mismatch rather than cancelling.
    fn scalar_nll(alpha: [f64; 2]) -> f64 {
        let value = [1.0, 0.4];
        let derivative = [0.0, 0.7];
        let offset = -0.15;
        let floor = 0.02;
        let weight = 1.3;
        let h = value[0] * alpha[0] + value[1] * alpha[1] + offset + floor;
        let h_prime = TRANSFORMATION_MONOTONICITY_EPS
            + derivative[0] * alpha[0]
            + derivative[1] * alpha[1];
        weight * (0.5 * h * h + 0.5 * (2.0 * std::f64::consts::PI).ln() - h_prime.ln())
    }

    #[test]
    fn saved_transformation_row_geometry_matches_independent_scalar_finite_difference() {
        let alpha: [f64; 2] = [0.25, 0.8];
        let geometry = transformation_normal_alo_row_geometry(TransformationNormalAloRowInput {
            response_value_basis: &[1.0, 0.4],
            response_derivative_basis: &[0.0, 0.7],
            response_lower_basis: &[1.0, 0.1],
            response_upper_basis: &[1.0, 0.9],
            alpha: &alpha,
            additive_offset: -0.15,
            response_floor_offset: 0.02,
            response_lower_floor_offset: -0.04,
            response_upper_floor_offset: 0.06,
            prior_weight: 1.3,
        })
        .expect("saved transformation-normal row must replay");
        let step = 2.0e-5;
        let base = scalar_nll(alpha);
        assert!((geometry.negative_log_likelihood - base).abs() <= 2.0e-13);
        for axis in 0..2 {
            let mut plus = alpha;
            let mut minus = alpha;
            plus[axis] += step;
            minus[axis] -= step;
            let gradient_fd = (scalar_nll(plus) - scalar_nll(minus)) / (2.0 * step);
            assert!(
                (geometry.nll_score[axis] - gradient_fd).abs() <= 2.0e-8,
                "score[{axis}] analytic={} fd={gradient_fd}",
                geometry.nll_score[axis]
            );
            for other in 0..2 {
                let mut pp = alpha;
                let mut pm = alpha;
                let mut mp = alpha;
                let mut mm = alpha;
                pp[axis] += step;
                pp[other] += step;
                pm[axis] += step;
                pm[other] -= step;
                mp[axis] -= step;
                mp[other] += step;
                mm[axis] -= step;
                mm[other] -= step;
                let hessian_fd = (scalar_nll(pp) - scalar_nll(pm) - scalar_nll(mp)
                    + scalar_nll(mm))
                    / (4.0 * step * step);
                assert!(
                    (geometry.observed_hessian[[axis, other]] - hessian_fd).abs() <= 3.0e-6,
                    "hessian[{axis},{other}] analytic={} fd={hessian_fd}",
                    geometry.observed_hessian[[axis, other]]
                );
            }
        }
        assert!(
            (geometry.observed_hessian[[0, 0]] - geometry.nll_score[0].powi(2)).abs() > 1.0e-3,
            "observed curvature must remain distinct from score covariance"
        );
    }
}
