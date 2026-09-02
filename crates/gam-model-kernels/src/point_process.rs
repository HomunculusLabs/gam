//! Natural-coordinate calculus for a piecewise-exponential point-process row.
//!
//! A row represents a count `y >= 0` and compensator exposure `e >= 0` at
//! log intensity `eta`:
//!
//! `log L = y * eta - e * exp(eta)`.
//!
//! Event rows may have zero exposure and quadrature/risk rows may have zero
//! count. Keeping those two quantities separate avoids the `log(0)` offset
//! required by a Poisson-count encoding of an exact event node. Constants that
//! do not depend on `eta` are intentionally absent: this is the point-process
//! density, not an interval-count probability mass function.

use gam_problem::EstimationError;

/// One unweighted point-process row in the natural log-intensity coordinate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PointProcessNaturalObservation {
    pub log_likelihood: f64,
    pub score: f64,
    pub negative_hessian: f64,
    pub negative_hessian_derivative: f64,
    pub negative_hessian_second_derivative: f64,
}

/// Evaluate a point-process row and the complete curvature tower required by
/// exact second-order LAML differentiation.
pub fn point_process_natural_observation(
    row: usize,
    count: f64,
    exposure: f64,
    eta: f64,
) -> Result<PointProcessNaturalObservation, EstimationError> {
    if !(count.is_finite() && count >= 0.0) {
        return Err(EstimationError::InvalidInput(format!(
            "point-process count at row {row} must be finite and non-negative, got {count}",
        )));
    }
    if !(exposure.is_finite() && exposure >= 0.0) {
        return Err(EstimationError::InvalidInput(format!(
            "point-process exposure at row {row} must be finite and non-negative, got {exposure}",
        )));
    }
    if !eta.is_finite() {
        return Err(EstimationError::PirlsRowGeometryUnrepresentable {
            row,
            quantity: "point-process log intensity",
            eta,
            value: eta,
        });
    }

    // An exact event node legitimately has exposure zero. Short-circuit it so
    // arbitrarily large finite eta never forms the indeterminate `0 * inf`.
    let integrated_intensity = if exposure == 0.0 {
        0.0
    } else {
        exposure * eta.exp()
    };
    if !integrated_intensity.is_finite() {
        return Err(EstimationError::PirlsRowGeometryUnrepresentable {
            row,
            quantity: "point-process integrated intensity",
            eta,
            value: integrated_intensity,
        });
    }
    let log_likelihood = count.mul_add(eta, -integrated_intensity);
    let score = count - integrated_intensity;
    if !(log_likelihood.is_finite() && score.is_finite()) {
        return Err(EstimationError::PirlsRowGeometryUnrepresentable {
            row,
            quantity: "point-process natural observation",
            eta,
            value: log_likelihood,
        });
    }
    Ok(PointProcessNaturalObservation {
        log_likelihood,
        score,
        negative_hessian: integrated_intensity,
        negative_hessian_derivative: integrated_intensity,
        negative_hessian_second_derivative: integrated_intensity,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn event_and_exposure_rows_add_to_the_exact_process_likelihood() {
        let eta = 0.7;
        let event = point_process_natural_observation(0, 1.0, 0.0, eta).expect("event");
        let risk = point_process_natural_observation(1, 0.0, 2.5, eta).expect("risk");
        let integrated = 2.5 * eta.exp();

        assert_eq!(event.log_likelihood + risk.log_likelihood, eta - integrated);
        assert_eq!(event.score + risk.score, 1.0 - integrated);
        assert_eq!(event.negative_hessian + risk.negative_hessian, integrated);
    }

    #[test]
    fn zero_exposure_remains_exact_in_the_finite_log_intensity_tail() {
        let observation =
            point_process_natural_observation(7, 1.0, 0.0, 1_000.0).expect("exact event row");
        assert_eq!(observation.log_likelihood, 1_000.0);
        assert_eq!(observation.score, 1.0);
        assert_eq!(observation.negative_hessian, 0.0);
        assert_eq!(observation.negative_hessian_derivative, 0.0);
        assert_eq!(observation.negative_hessian_second_derivative, 0.0);
    }

    #[test]
    fn curvature_tower_matches_finite_differences() {
        let count = 3.0;
        let exposure = 0.4;
        let eta = -0.2;
        let step = 1.0e-4;
        let minus = point_process_natural_observation(0, count, exposure, eta - step)
            .expect("minus");
        let centre =
            point_process_natural_observation(0, count, exposure, eta).expect("centre");
        let plus = point_process_natural_observation(0, count, exposure, eta + step)
            .expect("plus");

        let curvature_derivative =
            (plus.negative_hessian - minus.negative_hessian) / (2.0 * step);
        let curvature_second = (plus.negative_hessian
            - 2.0 * centre.negative_hessian
            + minus.negative_hessian)
            / step.powi(2);
        assert!(
            (centre.negative_hessian_derivative - curvature_derivative).abs() < 1.0e-9,
        );
        assert!(
            (centre.negative_hessian_second_derivative - curvature_second).abs() < 1.0e-8,
        );
    }

    #[test]
    fn invalid_measure_and_unrepresentable_intensity_are_typed_errors() {
        assert!(matches!(
            point_process_natural_observation(0, -1.0, 1.0, 0.0),
            Err(EstimationError::InvalidInput(_))
        ));
        assert!(matches!(
            point_process_natural_observation(0, 0.0, -1.0, 0.0),
            Err(EstimationError::InvalidInput(_))
        ));
        assert!(matches!(
            point_process_natural_observation(0, 0.0, 1.0, 1_000.0),
            Err(EstimationError::PirlsRowGeometryUnrepresentable { .. })
        ));
    }
}
