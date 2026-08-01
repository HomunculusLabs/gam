//! gam#2718 / gam#2484 — a BMS fit whose latent-z adequacy gate rejects
//! StandardNormal must MINT its point estimates and DECLARE that it withheld
//! the coefficient covariance.
//!
//! Before this contract the seam returned `Err` from
//! `fit_bernoulli_marginal_slope_terms`, so a conditional location-scale
//! calibration whose calibrated residual failed the standard-normal adequacy
//! gate destroyed the whole fit — point estimates included — three stages after
//! the decision that caused it. The state is a legitimate point-estimation
//! result: the measure-selection warning has always said so. It was
//! unreachable in practice because both production marginal-slope entry points
//! set `compute_covariance = true` unconditionally.
//!
//! Two arms, because the positive one alone is satisfiable by a patch that
//! declares unconditionally:
//!
//! * **positive** — the PRS/PC-confounded fixture routes through the
//!   conditional calibration, fails the gate (measured: |skew| x16.5,
//!   |excess kurtosis| x10.1, KS x6.40), selects `global-empirical`, and must
//!   come back as a fit with finite coefficients, NO covariance, and a typed
//!   `CovarianceDeclined` on its artifacts.
//! * **negative** — the rank-reduced `centers=60` fixture reaches
//!   `StandardNormal` via rank-INT and must declare NOTHING. A declaration
//!   there would mean the gate fires on fits that have a valid covariance,
//!   which is the failure mode that makes the whole channel meaningless.
//!
//! Publishing the UNCORRECTED covariance is not the alternative being tested
//! and must never become one: it omits the first-stage generated-regressor
//! uncertainty, so the intervals come out too narrow and are indistinguishable
//! on the wire from corrected ones. The correction that would make them
//! publishable (`G_measure`) is gam#2484 and is not implemented.

use gam::data::EncodedDataset;
use gam::inference::model::{ColumnKindTag, DataSchema, SchemaColumn};
use gam::estimate::CovarianceDeclined;
use gam::{FitConfig, FitResult, fit_from_formula};
use ndarray::Array2;

fn binary_outcome_shape_dataset() -> EncodedDataset {
    let n = 96usize;
    let headers = vec![
        "event",
        "sex",
        "entry_age_z",
        "current_age_ns_1",
        "current_age_ns_2",
        "current_age_ns_3",
        "current_age_ns_4",
        "prs_z",
        "PC1",
        "PC2",
        "PC3",
    ]
    .into_iter()
    .map(str::to_string)
    .collect::<Vec<_>>();
    let mut values = Vec::<f64>::with_capacity(n * headers.len());
    for i in 0..n {
        let sex = if i % 2 == 0 { 0.0 } else { 1.0 };
        let entry_age_z = (i as f64 - 47.5) / 18.0;
        let t = ((i % 19) as f64 - 9.0) / 9.0;
        let current_age_ns_1 = 1.0;
        let current_age_ns_2 = t;
        let current_age_ns_3 = t * t;
        let current_age_ns_4 = t * t * t;
        let prs_z = (((i * 37) % 101) as f64 - 50.0) / 22.0;
        let pc1 = ((i as f64) * 0.17).sin();
        let pc2 = ((i as f64) * 0.23).cos();
        let pc3 = ((i as f64) * 0.31).sin() * ((i as f64) * 0.07).cos();
        let eta = -0.15 + 0.25 * sex + 0.18 * entry_age_z + 0.16 * t + 0.10 * prs_z + 0.08 * pc1;
        let deterministic_noise = (((i * 13) % 17) as f64 - 8.0) / 11.0;
        let event = if eta + deterministic_noise > 0.0 {
            1.0
        } else {
            0.0
        };
        values.extend_from_slice(&[
            event,
            sex,
            entry_age_z,
            current_age_ns_1,
            current_age_ns_2,
            current_age_ns_3,
            current_age_ns_4,
            prs_z,
            pc1,
            pc2,
            pc3,
        ]);
    }
    EncodedDataset {
        headers,
        values: Array2::from_shape_vec((n, 11), values)
            .expect("binary-outcome-shape BMS data shape"),
        schema: DataSchema {
            columns: vec![
                SchemaColumn {
                    name: "event".to_string(),
                    kind: ColumnKindTag::Binary,
                    levels: vec![],
                },
                SchemaColumn {
                    name: "sex".to_string(),
                    kind: ColumnKindTag::Binary,
                    levels: vec![],
                },
                SchemaColumn {
                    name: "entry_age_z".to_string(),
                    kind: ColumnKindTag::Continuous,
                    levels: vec![],
                },
                SchemaColumn {
                    name: "current_age_ns_1".to_string(),
                    kind: ColumnKindTag::Continuous,
                    levels: vec![],
                },
                SchemaColumn {
                    name: "current_age_ns_2".to_string(),
                    kind: ColumnKindTag::Continuous,
                    levels: vec![],
                },
                SchemaColumn {
                    name: "current_age_ns_3".to_string(),
                    kind: ColumnKindTag::Continuous,
                    levels: vec![],
                },
                SchemaColumn {
                    name: "current_age_ns_4".to_string(),
                    kind: ColumnKindTag::Continuous,
                    levels: vec![],
                },
                SchemaColumn {
                    name: "prs_z".to_string(),
                    kind: ColumnKindTag::Continuous,
                    levels: vec![],
                },
                SchemaColumn {
                    name: "PC1".to_string(),
                    kind: ColumnKindTag::Continuous,
                    levels: vec![],
                },
                SchemaColumn {
                    name: "PC2".to_string(),
                    kind: ColumnKindTag::Continuous,
                    levels: vec![],
                },
                SchemaColumn {
                    name: "PC3".to_string(),
                    kind: ColumnKindTag::Continuous,
                    levels: vec![],
                },
            ],
        },
        column_kinds: vec![
            ColumnKindTag::Binary,
            ColumnKindTag::Binary,
            ColumnKindTag::Continuous,
            ColumnKindTag::Continuous,
            ColumnKindTag::Continuous,
            ColumnKindTag::Continuous,
            ColumnKindTag::Continuous,
            ColumnKindTag::Continuous,
            ColumnKindTag::Continuous,
            ColumnKindTag::Continuous,
            ColumnKindTag::Continuous,
        ],
    }
}

fn duplicate_pc_binary_outcome_shape_dataset() -> EncodedDataset {
    let n = 160usize;
    let headers = vec![
        "event",
        "sex",
        "entry_age_z",
        "current_age_ns_1",
        "current_age_ns_2",
        "current_age_ns_3",
        "current_age_ns_4",
        "prs_z",
        "PC1",
        "PC2",
        "PC3",
    ]
    .into_iter()
    .map(str::to_string)
    .collect::<Vec<_>>();
    let pc_cloud = [
        [-1.0, -0.5, 0.25],
        [-0.25, 0.75, -0.5],
        [0.5, -0.75, 0.75],
        [1.0, 0.5, -0.25],
    ];
    let mut values = Vec::<f64>::with_capacity(n * headers.len());
    for i in 0..n {
        let sex = if i % 2 == 0 { 0.0 } else { 1.0 };
        let entry_age_z = (i as f64 - 79.5) / 30.0;
        let t = ((i % 29) as f64 - 14.0) / 14.0;
        let prs_z = (((i * 17) % 131) as f64 - 65.0) / 31.0;
        let pc = pc_cloud[i % pc_cloud.len()];
        let eta = -0.05 + 0.12 * sex + 0.08 * entry_age_z + 0.06 * t + 0.05 * prs_z + 0.04 * pc[0];
        let deterministic_noise = (((i * 11) % 23) as f64 - 11.0) / 10.0;
        let event = if eta + deterministic_noise > 0.0 {
            1.0
        } else {
            0.0
        };
        values.extend_from_slice(&[
            event,
            sex,
            entry_age_z,
            t,
            t * t,
            t * t * t,
            t * t * t * t,
            prs_z,
            pc[0],
            pc[1],
            pc[2],
        ]);
    }
    EncodedDataset {
        headers,
        values: Array2::from_shape_vec((n, 11), values)
            .expect("duplicate-PC binary-outcome-shape BMS data shape"),
        schema: DataSchema {
            columns: vec![
                SchemaColumn {
                    name: "event".to_string(),
                    kind: ColumnKindTag::Binary,
                    levels: vec![],
                },
                SchemaColumn {
                    name: "sex".to_string(),
                    kind: ColumnKindTag::Binary,
                    levels: vec![],
                },
                SchemaColumn {
                    name: "entry_age_z".to_string(),
                    kind: ColumnKindTag::Continuous,
                    levels: vec![],
                },
                SchemaColumn {
                    name: "current_age_ns_1".to_string(),
                    kind: ColumnKindTag::Continuous,
                    levels: vec![],
                },
                SchemaColumn {
                    name: "current_age_ns_2".to_string(),
                    kind: ColumnKindTag::Continuous,
                    levels: vec![],
                },
                SchemaColumn {
                    name: "current_age_ns_3".to_string(),
                    kind: ColumnKindTag::Continuous,
                    levels: vec![],
                },
                SchemaColumn {
                    name: "current_age_ns_4".to_string(),
                    kind: ColumnKindTag::Continuous,
                    levels: vec![],
                },
                SchemaColumn {
                    name: "prs_z".to_string(),
                    kind: ColumnKindTag::Continuous,
                    levels: vec![],
                },
                SchemaColumn {
                    name: "PC1".to_string(),
                    kind: ColumnKindTag::Continuous,
                    levels: vec![],
                },
                SchemaColumn {
                    name: "PC2".to_string(),
                    kind: ColumnKindTag::Continuous,
                    levels: vec![],
                },
                SchemaColumn {
                    name: "PC3".to_string(),
                    kind: ColumnKindTag::Continuous,
                    levels: vec![],
                },
            ],
        },
        column_kinds: vec![
            ColumnKindTag::Binary,
            ColumnKindTag::Binary,
            ColumnKindTag::Continuous,
            ColumnKindTag::Continuous,
            ColumnKindTag::Continuous,
            ColumnKindTag::Continuous,
            ColumnKindTag::Continuous,
            ColumnKindTag::Continuous,
            ColumnKindTag::Continuous,
            ColumnKindTag::Continuous,
            ColumnKindTag::Continuous,
        ],
    }
}


const MARGINAL_FORMULA_CENTERS6: &str = "event ~ matern(PC1, PC2, PC3, centers=6, length_scale=1.0) + sex + entry_age_z + current_age_ns_1 + current_age_ns_2 + current_age_ns_3 + current_age_ns_4";
const LOGSLOPE_FORMULA_CENTERS6: &str = "matern(PC1, PC2, PC3, centers=6, length_scale=1.0)";
const MARGINAL_FORMULA_CENTERS60: &str = "event ~ matern(PC1, PC2, PC3, centers=60, length_scale=1.0) + sex + entry_age_z + current_age_ns_1 + current_age_ns_2 + current_age_ns_3 + current_age_ns_4";
const LOGSLOPE_FORMULA_CENTERS60: &str = "matern(PC1, PC2, PC3, centers=60, length_scale=1.0)";

/// The witness fixture: `prs_z` is made an exact affine function of PC1/PC2, so
/// the conditional `E[z|C]`/`Var(z|C)` Rao gate fires and the calibrated
/// residual then fails the pooled standard-normal adequacy gate. Byte-identical
/// to the confound injection in
/// `tests/identifiability/misc/binary_outcome_bms_identifiability.rs`.
fn prs_pc_confounded_dataset() -> EncodedDataset {
    let mut data = binary_outcome_shape_dataset();
    let prs_idx = data
        .headers
        .iter()
        .position(|h| h == "prs_z")
        .expect("prs_z column");
    let pc1_idx = data
        .headers
        .iter()
        .position(|h| h == "PC1")
        .expect("PC1 column");
    let pc2_idx = data
        .headers
        .iter()
        .position(|h| h == "PC2")
        .expect("PC2 column");
    for row in 0..data.values.nrows() {
        data.values[[row, prs_idx]] =
            data.values[[row, pc1_idx]] + 0.1 * data.values[[row, pc2_idx]];
    }
    data
}

fn fit_bms(
    data: &EncodedDataset,
    marginal: &str,
    logslope: &str,
    label: &str,
) -> gam::families::bms::BernoulliMarginalSlopeFitResult {
    let cfg = FitConfig {
        logslope_formula: Some(logslope.to_string()),
        z_column: Some("prs_z".to_string()),
        ..FitConfig::default()
    };
    match fit_from_formula(marginal, data, &cfg) {
        Ok(FitResult::BernoulliMarginalSlope(out)) => out,
        Ok(_) => panic!("{label}: fit returned the wrong family variant"),
        Err(err) => panic!(
            "{label}: the fit must be MINTED. A latent measure the Murphy-Topel \
             correction cannot handle withholds the covariance; it does not \
             destroy the point estimates (gam#2718). Got: {err}"
        ),
    }
}

#[test]
fn bms_declines_covariance_and_still_mints_point_estimates_2718() {
    gam::init_parallelism();
    let data = prs_pc_confounded_dataset();
    let out = fit_bms(
        &data,
        MARGINAL_FORMULA_CENTERS6,
        LOGSLOPE_FORMULA_CENTERS6,
        "prs/pc-confounded BMS fit",
    );

    // 1. The point estimates are published. This is the whole reason the
    //    refusal moved: they were being destroyed to report a missing
    //    covariance.
    assert!(
        out.fit.beta.iter().all(|coef| coef.is_finite()),
        "gam#2718: a declined covariance must still publish finite coefficients, got {:?}",
        out.fit.beta
    );
    assert!(
        !out.fit.beta.is_empty(),
        "gam#2718: the published coefficient vector must be non-empty"
    );
    assert!(
        out.fit.log_lambdas.iter().all(|rho| rho.is_finite()),
        "gam#2718: a declined covariance must still publish finite smoothing parameters, got {:?}",
        out.fit.log_lambdas
    );

    // 2. The covariance is withheld — on EVERY surface that could carry it.
    //    Publishing the uncorrected matrix on any one of them would ship
    //    intervals that are too narrow and indistinguishable from corrected
    //    ones, which is the outcome this contract exists to prevent.
    assert!(
        out.fit.covariance_conditional.is_none(),
        "gam#2718: the conditional covariance must be withheld, not published uncorrected"
    );
    assert!(
        out.fit.covariance_corrected.is_none(),
        "gam#2718: the corrected covariance must be withheld, not published uncorrected"
    );
    if let Some(inference) = out.fit.inference.as_ref() {
        assert!(
            inference.beta_covariance.is_none(),
            "gam#2718: the inference-surface covariance must be withheld too"
        );
        assert!(
            inference.beta_standard_errors.is_none(),
            "gam#2718: standard errors derived from a withheld covariance must be withheld"
        );
        assert!(
            inference.beta_covariance_corrected.is_none(),
            "gam#2718: the corrected inference covariance must be withheld too"
        );
        assert!(
            inference.beta_standard_errors_corrected.is_none(),
            "gam#2718: corrected standard errors must be withheld too"
        );
    }

    // 3. The absence is DECLARED and typed. A bare `None` is three states in
    //    one costume (not requested / not computed / not valid to publish); the
    //    consumer has to be able to tell which one this is.
    match out.fit.artifacts.covariance_declined.as_ref() {
        Some(CovarianceDeclined::BmsGeneratedRegressorLatentMeasureNotStandardNormal {
            latent_measure,
        }) => {
            assert_eq!(
                latent_measure, "global-empirical",
                "gam#2718: the declaration must name the measure the adequacy gate selected"
            );
        }
        None => panic!(
            "gam#2718: the covariance was withheld with NO declaration. A silent absence is \
             indistinguishable from `never requested`, which is exactly the state a consumer \
             reading standard errors would misread."
        ),
    }

    let explanation = out
        .fit
        .artifacts
        .covariance_declined
        .as_ref()
        .expect("declaration asserted present above")
        .explain();
    assert!(
        explanation.contains("gam#2484"),
        "gam#2718: the explanation must point at the issue tracking the missing correction, got: {explanation}"
    );
}

#[test]
fn bms_standard_normal_latent_measure_declares_nothing_2718() {
    // Negative control. Without it, an implementation that sets the
    // declaration unconditionally passes the positive arm above — and a
    // declaration on a fit that HAS a valid covariance is worse than no
    // channel at all, because it teaches consumers to ignore the field.
    gam::init_parallelism();
    let data = duplicate_pc_binary_outcome_shape_dataset();
    let out = fit_bms(
        &data,
        MARGINAL_FORMULA_CENTERS60,
        LOGSLOPE_FORMULA_CENTERS60,
        "rank-reduced centers=60 BMS fit",
    );

    assert!(
        out.fit.beta.iter().all(|coef| coef.is_finite()),
        "control fixture must fit: got {:?}",
        out.fit.beta
    );
    assert!(
        out.fit.artifacts.covariance_declined.is_none(),
        "gam#2718: a fit that did NOT hit the non-StandardNormal seam must declare nothing; \
         got {:?}",
        out.fit.artifacts.covariance_declined
    );
}

#[test]
fn bms_declination_travels_with_the_curvature_it_is_about_2718() {
    // The declaration is only worth having if it crosses the persistence
    // boundary WITH the curvature it describes. A consumer that loads a saved
    // fit gets the penalized Hessian `H` and the dispersion `phi`, from which
    // `Vb_naive = phi * H^-1` is one Cholesky away; if the declaration did not
    // travel alongside, that consumer would not have ignored a warning, they
    // would never have received one.
    //
    // The BMS saved-model route stores the whole `UnifiedFitResult`
    // (`model_payload_builders.rs:802-803`), after a narrowing that carries
    // `artifacts` forward (`:606`), so the link this test pins is the wire
    // itself: `covariance_declined` is `#[serde(default)]`, NOT
    // `skip_serializing`, and must survive a round trip.
    //
    // Both halves are asserted together on purpose. A future change that keeps
    // the Hessian and drops the declaration is exactly the silent-consumption
    // case, and it would pass either assertion alone.
    gam::init_parallelism();
    let data = prs_pc_confounded_dataset();
    let out = fit_bms(
        &data,
        MARGINAL_FORMULA_CENTERS6,
        LOGSLOPE_FORMULA_CENTERS6,
        "prs/pc-confounded BMS fit (persistence arm)",
    );

    let declared = out.fit.artifacts.covariance_declined.clone();
    assert!(
        declared.is_some(),
        "gam#2718: fixture must reach the declining seam for this arm to mean anything"
    );

    // The curvature the declaration is about is present ...
    assert!(
        out.fit.penalized_hessian().is_some(),
        "gam#2718: a declined fit still publishes the penalized Hessian — EDF accounting, \
         posterior whitening and prediction all read it. This assertion is here so that the \
         NEXT one is load-bearing: shipping H is exactly why the declaration has to ship too."
    );

    // ... and the declaration survives the wire beside it.
    let encoded = serde_json::to_string(&out.fit.artifacts)
        .expect("gam#2718: fit artifacts must serialize");
    let decoded: gam::estimate::FitArtifacts = serde_json::from_str(&encoded)
        .expect("gam#2718: fit artifacts must deserialize");
    assert_eq!(
        decoded.covariance_declined, declared,
        "gam#2718: the declination must survive serialization. If this field ever becomes \
         `skip_serializing`, or a persistence route reassembles the fit without carrying \
         `artifacts`, a consumer loading the saved model gets H and phi with NO warning — \
         which is the silent consumption this channel exists to prevent."
    );
}
