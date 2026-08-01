use gam_sae::null_battery::{NullKind, Tail};
use gam_sae::saebench_metrics::{
    ChartInterpNullCalibration, ChartInterpNullDrawPolicy, ChartInterpNullProtocol,
    ChartInterpObservation, ChartInterpReadout, ChartInterpStatistic, ChartInterpVerdict,
    DoseResponseObservation, chart_interp_score, coordinate_posterior_from_precision,
    dose_response_calibration,
};

fn matched_spectrum_calibration(
    seed: u64,
    draws: Vec<Vec<ChartInterpObservation>>,
) -> ChartInterpNullCalibration {
    ChartInterpNullCalibration::new(
        ChartInterpNullProtocol::MatchedSpectrumGaussianV1,
        ChartInterpReadout::TokenMeanPcaPlaneV1,
        seed,
        draws.len(),
        draws,
    )
    .expect("complete matched-spectrum calibration")
}

/// The reversed-orientation observed ledger: the recovered coordinate runs
/// backwards relative to the cyclic label, so `label + recovered` is exactly one
/// turn on every row and the orientation-quotiented lock is exactly 1.
fn reversed_cyclic_observations() -> Vec<ChartInterpObservation> {
    vec![
        ChartInterpObservation {
            recovered_turns: 0.99,
            label_turns: 0.01,
            weight: 1.0,
        },
        ChartInterpObservation {
            recovered_turns: 0.24,
            label_turns: 0.76,
            weight: 1.0,
        },
        ChartInterpObservation {
            recovered_turns: 0.49,
            label_turns: 0.51,
            weight: 1.0,
        },
        ChartInterpObservation {
            recovered_turns: 0.74,
            label_turns: 0.26,
            weight: 1.0,
        },
    ]
}

/// A CONSTANT null ledger for `observed`: every recovered coordinate collapsed
/// to zero. Any number of copies of this draw has `sd == 0`, so it is the
/// negative control for the z-score refusal, never a calibration.
fn constant_null_draw(observed: &[ChartInterpObservation]) -> Vec<ChartInterpObservation> {
    observed
        .iter()
        .map(|row| ChartInterpObservation {
            recovered_turns: 0.0,
            label_turns: row.label_turns,
            weight: row.weight,
        })
        .collect()
}

/// SplitMix64 over `(seed, draw, row)` mapped onto the unit circle -- a
/// reproducible uniform phase with no RNG dependency in the test crate.
fn surrogate_turns(seed: u64, draw: usize, row: usize) -> f64 {
    let mut x = seed
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add((draw as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9))
        .wrapping_add((row as u64).wrapping_mul(0x94D0_49BB_1331_11EB));
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^= x >> 31;
    (x >> 11) as f64 / (1u64 << 53) as f64
}

/// `draws` matched-spectrum surrogate ledgers for `observed`, generated from the
/// calibration's own declared seed.
///
/// A surrogate keeps the labeled rows and their weights EXACTLY -- the scorer
/// refuses a draw that changes `label_turns`, and re-scoring the identical
/// labeled ledger is the whole content of the declared protocol -- and replaces
/// only the recovered coordinate with a phase-randomized one. Destroying the
/// phase relation to the labels while holding the labels fixed is what makes the
/// resulting statistics a null for the orientation-quotiented phase lock; a
/// per-draw constant offset would NOT, because the lock magnitude is invariant
/// under a rigid rotation of the recovered coordinate and every such draw would
/// score identically (`sd == 0` again, one defect swapped for another).
fn matched_spectrum_null_draws(
    observed: &[ChartInterpObservation],
    seed: u64,
    draws: usize,
) -> Vec<Vec<ChartInterpObservation>> {
    assert!(
        draws >= 2,
        "a null ensemble with fewer than two draws has no spread and cannot supply a z-score"
    );
    (0..draws)
        .map(|draw| {
            observed
                .iter()
                .enumerate()
                .map(|(row, observation)| ChartInterpObservation {
                    recovered_turns: surrogate_turns(seed, draw, row),
                    label_turns: observation.label_turns,
                    weight: observation.weight,
                })
                .collect()
        })
        .collect()
}

#[test]
fn chart_interp_wraps_cyclic_boundary_and_quotients_orientation() {
    let obs = reversed_cyclic_observations();
    // THIRTY-TWO surrogate draws, not one. Two constraints set the count and
    // neither is a tuning choice: a one-draw null has `sd == 0` by construction
    // (this fixture used to ask for a z-score from it and died at
    // `chart_interp_score`, #2699), and the plus-one calibration cannot report a
    // p-value below `1 / (draws + 1)`, so a 0.05 verdict is UNREACHABLE below 19
    // draws no matter what the data say. Under 19 draws the verdict assertion
    // below would be a non-measurement: `NullCompatible` for arithmetic reasons
    // rather than evidential ones.
    let calibration = matched_spectrum_calibration(7, matched_spectrum_null_draws(&obs, 7, 32));
    let report = chart_interp_score(&obs, &calibration, 0.05).unwrap();
    let null = &report.calibration.null_distribution;

    // The null was FREE TO DISAGREE before any of its verdict is read: it has 32
    // draws, a strictly positive spread (so the z-score is defined at all), and
    // its largest draw sits strictly below the observed lock -- a draw reaching
    // the observed statistic would have made `extreme_draws > 0` and flipped the
    // verdict.
    assert_eq!(null.n, 32, "the scorer must score every declared draw");
    assert!(
        null.sd > 0.0,
        "a null with sd == 0 cannot supply a z-score; the ensemble is degenerate: {null:?}"
    );
    assert!(
        null.max < report.observed.circular_correlation,
        "null max {} must sit strictly below the observed lock {}; the surrogates are not a null otherwise",
        null.max,
        report.observed.circular_correlation
    );

    assert!(
        report.observed.circular_correlation > 0.99,
        "orientation-quotiented cyclic chart score should recover the reversed coordinate: {report:?}"
    );
    assert!(report.observed.signed_circular_correlation < 0.0);
    // A perfect phase lock against a matched-spectrum null is evidence, and the
    // calibration now says so. The old `NullCompatible` assertion was written
    // against a null the scorer could never summarize, so it had never once been
    // evaluated.
    assert_eq!(null.extreme_draws, 0, "no surrogate reached the observed lock");
    assert!(
        null.p_value <= 0.05,
        "plus-one p-value {} at 32 draws with no extreme draw",
        null.p_value
    );
    assert_eq!(report.verdict, ChartInterpVerdict::Pass);
}

#[test]
fn chart_interp_refuses_a_null_that_cannot_supply_a_z_score_2699() {
    let obs = reversed_cyclic_observations();

    // (a) WRONG PROPERTY, refused at the declaration: one draw has no spread,
    // whatever it contains.
    let error = ChartInterpNullCalibration::new(
        ChartInterpNullProtocol::MatchedSpectrumGaussianV1,
        ChartInterpReadout::TokenMeanPcaPlaneV1,
        7,
        1,
        vec![constant_null_draw(&obs)],
    )
    .unwrap_err();
    assert!(
        error.contains("at least two draws"),
        "a one-draw calibration must be refused where the draw count is declared: {error}"
    );

    // (b) WRONG PROPERTY, refused at the statistic: two draws are necessary and
    // not sufficient. A CONSTANT null still has `sd == 0`, and the production
    // z-score refusal is the correct answer -- this fix must not weaken it.
    let calibration = matched_spectrum_calibration(
        7,
        vec![constant_null_draw(&obs), constant_null_draw(&obs)],
    );
    let error = chart_interp_score(&obs, &calibration, 0.05).unwrap_err();
    assert!(
        error.contains("z-score is undefined"),
        "a constant two-draw null must still be refused: {error}"
    );

    // (c) RIGHT PROPERTY: the surrogate ensemble does supply a defined z-score,
    // so (a) and (b) are refusals about the null's spread and not about the
    // scorer being unable to score anything at all.
    let calibration = matched_spectrum_calibration(7, matched_spectrum_null_draws(&obs, 7, 32));
    let report = chart_interp_score(&obs, &calibration, 0.05).unwrap();
    assert!(report.calibration.null_distribution.sd > 0.0);
    assert!(report.calibration.null_distribution.z.is_finite());
}

fn correlation_fixture(target: f64) -> Vec<ChartInterpObservation> {
    let phase_error = target.acos() / std::f64::consts::TAU;
    (0..12)
        .map(|index| {
            let label = index as f64 / 12.0;
            let sign = if index % 2 == 0 { 1.0 } else { -1.0 };
            ChartInterpObservation {
                recovered_turns: (label + sign * phase_error).rem_euclid(1.0),
                label_turns: label,
                weight: 1.0,
            }
        })
        .collect()
}

fn zero_null_draw() -> Vec<ChartInterpObservation> {
    (0..12)
        .map(|index| ChartInterpObservation {
            recovered_turns: 0.0,
            label_turns: index as f64 / 12.0,
            weight: 1.0,
        })
        .collect()
}

#[test]
fn chart_interp_report_carries_typed_provenance_and_recomputed_samples_2250() {
    let draws = vec![
        correlation_fixture(0.9),
        zero_null_draw(),
        correlation_fixture(0.6),
    ];
    let calibration = matched_spectrum_calibration(0x2250, draws);
    let report = chart_interp_score(&correlation_fixture(0.8), &calibration, 0.05).unwrap();

    assert_eq!(
        report.statistic,
        ChartInterpStatistic::OrientationQuotientedWeightedPhaseLock
    );
    assert_eq!(report.calibration.statistic, report.statistic);
    assert_eq!(
        report.calibration.protocol,
        ChartInterpNullProtocol::MatchedSpectrumGaussianV1
    );
    assert_eq!(
        report.calibration.readout,
        ChartInterpReadout::TokenMeanPcaPlaneV1
    );
    assert_eq!(
        report.calibration.draw_policy,
        ChartInterpNullDrawPolicy::RegenerateSurrogateAndRepeatReadout
    );
    assert_eq!(
        report.calibration.null_kind,
        NullKind::MatchedSpectrumGaussian
    );
    assert_eq!(report.calibration.seed, 0x2250);
    let null = &report.calibration.null_distribution;
    assert_eq!(null.kind, NullKind::MatchedSpectrumGaussian);
    assert_eq!(null.tail, Tail::Larger);
    assert_eq!(null.n, 3);
    assert_eq!(null.extreme_draws, 1);
    assert_eq!(null.samples.len(), 3);
    assert!((null.samples[0] - 0.9).abs() < 1.0e-12);
    assert!(null.samples[1].abs() < 1.0e-12);
    assert!((null.samples[2] - 0.6).abs() < 1.0e-12);
    assert!((null.p_value - 0.5).abs() < 1.0e-12);
    assert_eq!(report.verdict, ChartInterpVerdict::NullCompatible);
}

#[test]
fn chart_interp_rejects_scalar_only_evidence_2250() {
    assert!(
        ChartInterpNullProtocol::parse("matched_spectrum_gaussian_chart_refit_v1").is_err(),
        "the old protocol falsely asserted that PCA token-mean angles came from a chart refit"
    );
    assert!(ChartInterpReadout::parse("unspecified").is_err());

    let error = ChartInterpNullCalibration::new(
        ChartInterpNullProtocol::MatchedSpectrumGaussianV1,
        ChartInterpReadout::TokenMeanPcaPlaneV1,
        0x2250,
        0,
        Vec::new(),
    )
    .unwrap_err();
    assert!(error.contains("at least two draws"), "{error}");

    let error = ChartInterpNullCalibration::new(
        ChartInterpNullProtocol::MatchedSpectrumGaussianV1,
        ChartInterpReadout::TokenMeanPcaPlaneV1,
        0x2250,
        2,
        vec![zero_null_draw()],
    )
    .unwrap_err();
    assert!(error.contains("declares 2 draws"));
}

#[test]
fn chart_interp_rejects_a_null_ledger_for_different_labels_2250() {
    let observed = correlation_fixture(0.8);
    let mut wrong_labels = zero_null_draw();
    wrong_labels[0].label_turns = 0.25;
    // Two draws: the calibration must be constructible before the per-draw
    // label-alignment refusal is what gets measured.
    let calibration = matched_spectrum_calibration(0x2250, vec![wrong_labels, zero_null_draw()]);
    let error = chart_interp_score(&observed, &calibration, 0.05).unwrap_err();
    assert!(error.contains("changes label_turns"));
}

#[test]
fn dose_response_reports_fisher_calibration_slope_and_unit_speed_constancy() {
    // Endpoint dosing is quadratic in the applied chord: predicted = ½·arc²,
    // measured = arc², so the per-arc² rate is exactly constant (CV = 0) and
    // the calibration slope through the origin is exactly 2.
    let obs = [
        DoseResponseObservation {
            arc_length: 1.0,
            predicted_nats: 0.5,
            measured_nats: 1.0,
            weight: 1.0,
        },
        DoseResponseObservation {
            arc_length: 2.0,
            predicted_nats: 2.0,
            measured_nats: 4.0,
            weight: 1.0,
        },
        DoseResponseObservation {
            arc_length: 3.0,
            predicted_nats: 4.5,
            measured_nats: 9.0,
            weight: 1.0,
        },
    ];
    let report = dose_response_calibration(&obs).unwrap();
    assert!((report.slope_through_origin - 2.0).abs() < f64::EPSILON.sqrt());
    assert!((report.r2_through_origin - 1.0).abs() < f64::EPSILON.sqrt());
    assert!(report.cv_measured_nats_per_arc_squared < f64::EPSILON.sqrt());
}

#[test]
fn coordinate_posterior_inverts_row_hessian_precision_block() {
    let posterior =
        coordinate_posterior_from_precision(&[0.25, 0.75], &[4.0, 1.0, 1.0, 3.0]).unwrap();
    assert!((posterior.covariance_diag[0] - 3.0 / 11.0).abs() < f64::EPSILON.sqrt());
    assert!((posterior.covariance_diag[1] - 4.0 / 11.0).abs() < f64::EPSILON.sqrt());
    assert!((posterior.precision_weight - 11.0 / 7.0).abs() < f64::EPSILON.sqrt());
}
