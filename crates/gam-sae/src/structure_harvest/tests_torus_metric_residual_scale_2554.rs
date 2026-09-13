use super::*;

/// The four `(coordinate, dS/dcoordinate)` samples the #2554 failure
/// reported: the two domain endpoints from its endpoint profile, and the
/// two ends of the bracket the refinement exhausted on.
const REPORTED: [(f64, f64); 4] = [
    (2.220446049250313e-16, -10.870223764685393),
    (0.00008987029644979831, -89016.42875485175),
    (0.00009002448793615778, 0.587801723293678),
    (1.0, 34.2821344402842),
];

/// The chain factor is what varies, not the stationarity.
///
/// Rescaling the reported residuals by `|dA/dq|` turns a sequence spanning
/// 1.5e5 into one whose sign structure is identical and whose magnitude
/// near the root is ordinary. This is the whole content of #2554: the
/// refinement was holding `dS/dA = -1.0e-6` — a good answer — and could not
/// report it, because the factor there was 5.9e5 and inflated it past a
/// tolerance calibrated where the factor is 0.5.
#[test]
fn rescaling_by_the_chain_factor_makes_the_2554_tolerance_reachable() {
    let mut rescaled = Vec::new();
    for (coordinate, coordinate_gradient) in REPORTED {
        let magnitude =
            torus_metric_aspect_derivative_magnitude(TorusMetricFamily::Flat, coordinate)
                .expect("every reported coordinate is inside the flat domain");
        let residual = coordinate_gradient / magnitude;
        assert_eq!(
            residual.signum(),
            coordinate_gradient.signum(),
            "dividing by a magnitude must not move a sign: {coordinate_gradient} at \
                 {coordinate} became {residual}"
        );
        rescaled.push(residual);
    }

    // The bracket the refinement exhausted on is a GENUINE sign change, not
    // a pole: the rescaled residual crosses zero between its two ends.
    assert!(
        rescaled[1] < 0.0 && rescaled[2] > 0.0,
        "the exhausted bracket must still enclose a sign change after rescaling, got \
             {} and {}",
        rescaled[1],
        rescaled[2]
    );

    let position_tolerance = f64::EPSILON.sqrt();

    // Before: the tolerance comes from the endpoint residuals in the
    // coordinate, and the bracket end is four orders above it.
    let coordinate_scale = REPORTED[0].1.abs().max(REPORTED[3].1.abs()).max(1.0);
    let coordinate_tolerance = position_tolerance * coordinate_scale;
    assert!(
        REPORTED[2].1.abs() > 1.0e3 * coordinate_tolerance,
        "the failure this gate encodes requires the coordinate residual to be far above \
             its own tolerance: {} vs {coordinate_tolerance}",
        REPORTED[2].1.abs()
    );

    // After: same construction in the rescaled coordinate is met.
    let aspect_scale = rescaled[0].abs().max(rescaled[3].abs()).max(1.0);
    let aspect_tolerance = position_tolerance * aspect_scale;
    assert!(
        rescaled[2].abs() <= aspect_tolerance,
        "the rescaled residual {} must satisfy the tolerance {aspect_tolerance} its own \
             endpoints imply",
        rescaled[2].abs()
    );
}

/// Why an absolute tolerance cannot serve this domain, stated as a
/// measurement rather than an assertion in a comment: the factor's span is
/// the size of the mismatch, and it is enormous for BOTH families. The
/// donut arm has not been driven into the failure yet; it is not immune.
#[test]
fn the_chain_factor_spans_orders_across_both_family_domains() {
    let flat_low =
        torus_metric_aspect_derivative_magnitude(TorusMetricFamily::Flat, f64::EPSILON)
            .expect("flat lower wall");
    let flat_high = torus_metric_aspect_derivative_magnitude(TorusMetricFamily::Flat, 1.0)
        .expect("flat upper wall");
    assert!(
        (flat_low / flat_high).log10() > 20.0,
        "flat chain factor span {} orders",
        (flat_low / flat_high).log10()
    );

    let resolution = f64::EPSILON.sqrt();
    let donut_low =
        torus_metric_aspect_derivative_magnitude(TorusMetricFamily::EmbeddedDonut, resolution)
            .expect("donut lower wall");
    let donut_high = torus_metric_aspect_derivative_magnitude(
        TorusMetricFamily::EmbeddedDonut,
        1.0 - resolution.sqrt(),
    )
    .expect("donut upper wall");
    assert!(
        (donut_low / donut_high).log10() > 15.0,
        "donut chain factor span {} orders",
        (donut_low / donut_high).log10()
    );
}

/// The factor is strictly signed on each open domain, which is what lets
/// the rescaling preserve the bracket certificate. A zero would move a
/// sign and a non-finite one would destroy the residual.
#[test]
fn the_chain_factor_is_finite_and_nonzero_across_each_domain() {
    for step in 1..64 {
        let flat = f64::from(step) / 64.0;
        let magnitude = torus_metric_aspect_derivative_magnitude(TorusMetricFamily::Flat, flat)
            .expect("interior flat coordinate");
        assert!(magnitude.is_finite() && magnitude > 0.0, "flat at {flat}");
        let donut =
            torus_metric_aspect_derivative_magnitude(TorusMetricFamily::EmbeddedDonut, flat)
                .expect("interior donut coordinate");
        assert!(donut.is_finite() && donut > 0.0, "donut at {flat}");
    }
    assert!(
        torus_metric_aspect_derivative_magnitude(TorusMetricFamily::Flat, 0.0).is_err(),
        "the flat domain is open at zero"
    );
    assert!(
        torus_metric_aspect_derivative_magnitude(TorusMetricFamily::EmbeddedDonut, 1.0)
            .is_err(),
        "the donut domain is open at one, where the factor vanishes"
    );
}
