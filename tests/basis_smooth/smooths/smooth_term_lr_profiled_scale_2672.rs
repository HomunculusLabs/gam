//! gam#2672 — the smooth-term LR statistic of a PROFILED-GAUSSIAN fit is a
//! ratio, and these are the identities the estimated-scale reference rests on.
//!
//! The size grids in `smooth_term_lr_size_calibration` measure whether the
//! reference is CALIBRATED. They cannot say why, and a size that lands can land
//! for the wrong reason. This module pins the three statements the reference is
//! assembled from, each against a quantity read off the shipped fit rather than
//! recomputed from the same code:
//!
//!   1. `W = n·ln(D_0/D_f) + B` exactly, with `B = n·ln(ν_f/ν_0) + (ν_0 − ν_f)`
//!      the published deterministic offset — the algebra of gam's own profiled
//!      Gaussian log-likelihood, and the only reason the tail can be inverted in
//!      closed form at all;
//!   2. the residual law's spectrum is the WHOLE MODEL's penalty shares, checked
//!      against the fit's own `edf_total` through `Σ_i p_i = p − tr F`;
//!   3. the channel is present exactly on the family that profiles a Gaussian
//!      scale out of a residual sum of squares, and absent on every other.
//!
//! Plus the sign: an estimated scale can only cost power, so the published
//! p-value must be at or above the one the known-scale reference would give on
//! the same statistic.

use gam::smooth::{SmoothTermLrInference, smooth_term_lr_inference_forspec};
use gam::{
    FitConfig, FitRequest, FitResult, encode_recordswith_inferred_schema, fit_from_formula,
    init_parallelism, materialize,
};

use csv::StringRecord;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Poisson};

const N: usize = 120;
const K: usize = 8;

/// `y ~ x + s(z)` with a genuine — not null — smooth, so the statistic is well
/// away from the degenerate corner and the identities are being checked
/// somewhere they could fail.
///
/// The `w` column carries per-row training weights. Every row is weight one
/// except a handful of exactly-ZERO rows, which is the one configuration where
/// the observation count the reference needs (`Σ[w > 0]`, the count the
/// optimizer's own `φ̂ = weighted_rss/(n − edf)` uses under #584) differs from
/// the raw row count. `weight_column` is only set on the arms that ask for it.
fn fixture(gaussian: bool) -> gam::data::EncodedDataset {
    let mut rng = StdRng::seed_from_u64(0x2672_5CA1_E000_0001);
    let headers = ["y", "x", "z", "w"].iter().map(|s| s.to_string()).collect();
    let mut rows = Vec::<StringRecord>::with_capacity(N);
    for i in 0..N {
        let x = i as f64 / (N as f64 - 1.0);
        let z: f64 = rng.random_range(0.0..1.0);
        let eta = 0.3 + 0.8 * x + 0.9 * (std::f64::consts::TAU * z).sin();
        let y: f64 = if gaussian {
            Normal::new(eta, 0.5).expect("normal").sample(&mut rng)
        } else {
            Poisson::new(eta.exp()).expect("rate").sample(&mut rng) as f64
        };
        let weight = if i % 17 == 3 { 0.0 } else { 1.0 };
        rows.push(StringRecord::from(vec![
            y.to_string(),
            x.to_string(),
            z.to_string(),
            weight.to_string(),
        ]));
    }
    encode_recordswith_inferred_schema(headers, rows).expect("encode the #2672 fixture")
}

/// How many rows the fixture leaves at positive weight.
fn positive_weight_rows() -> usize {
    (0..N).filter(|i| i % 17 != 3).count()
}

fn lr_report(family: &str, data: &gam::data::EncodedDataset) -> SmoothTermLrInference {
    lr_report_with(family, data, None)
}

fn lr_report_with(
    family: &str,
    data: &gam::data::EncodedDataset,
    weight_column: Option<&str>,
) -> SmoothTermLrInference {
    let cfg = FitConfig {
        family: Some(family.to_string()),
        weight_column: weight_column.map(str::to_string),
        ..FitConfig::default()
    };
    let formula = format!("y ~ x + s(z, k={K})");
    let mat = materialize(&formula, data, &cfg).expect("materialize");
    let FitRequest::Standard(req) = mat.request else {
        panic!("expected a standard fit request");
    };
    smooth_term_lr_inference_forspec(
        req.data.view(),
        req.y.view(),
        req.weights.view(),
        req.offset.view(),
        &req.spec,
        req.family,
        &req.options,
    )
    .expect("smooth-term LR inference")
    .into_iter()
    .find(|report| report.name.contains('z'))
    .expect("a report for the s(z) term")
}

/// `(deviance, ν = D/σ̂², edf_total, design columns)` for one formula, through
/// the ordinary fit entry point.
fn fit_summary(formula: &str, data: &gam::data::EncodedDataset) -> (f64, f64, f64, usize) {
    fit_summary_with(formula, data, None)
}

fn fit_summary_with(
    formula: &str,
    data: &gam::data::EncodedDataset,
    weight_column: Option<&str>,
) -> (f64, f64, f64, usize) {
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        weight_column: weight_column.map(str::to_string),
        ..FitConfig::default()
    };
    let FitResult::Standard(standard) = fit_from_formula(formula, data, &cfg).expect("fit") else {
        panic!("expected a standard fit");
    };
    let variance = standard.fit.standard_deviation * standard.fit.standard_deviation;
    let residual_df = standard.fit.deviance / variance;
    let edf_total = standard
        .fit
        .edf_total()
        .expect("an inference-carrying fit publishes its total edf");
    (
        standard.fit.deviance,
        residual_df,
        edf_total,
        standard.design.design.ncols(),
    )
}

/// THE ALGEBRA. gam's profiled Gaussian log-likelihood is
/// `ℓ = −½[n·ln 2π + n·ln(D/ν) − Σ ln w_i + ν]`, so
///
/// ```text
///   W = 2(ℓ_f − ℓ_0) = n·ln(D_0/D_f) + n·ln(ν_f/ν_0) + (ν_0 − ν_f),
/// ```
///
/// and the whole estimated-scale reference is the observation that the first
/// term is a monotone function of `Q/V`. If this identity does not hold on the
/// shipped fit then the inversion `c(w) = expm1((w − B)/n)` is inverting the
/// wrong map, and no amount of calibration measurement would say so.
#[test]
fn the_profiled_gaussian_lr_statistic_is_the_log_deviance_ratio_plus_its_offset() {
    init_parallelism();
    let data = fixture(true);
    // Both arms of the observation count: unweighted, where it is the row
    // count, and a fixture with exactly-zero-weight rows, where it is not. The
    // count that multiplies `ln σ̂²` and the count `φ̂` divides by have to be the
    // SAME number or this identity does not hold, and the zero-weight arm is
    // the only place they could differ.
    for (label, weight_column, observations) in [
        ("unweighted", None, N),
        ("zero-weight rows", Some("w"), positive_weight_rows()),
    ] {
        let report = lr_report_with("gaussian", &data, weight_column);
        let scale = report
            .ref_df_provenance
            .profiled_scale
            .as_ref()
            .expect("a profiled-Gaussian fit carries the estimated-scale channel");

        let (deviance_full, nu_full, _, _) =
            fit_summary_with(&format!("y ~ x + s(z, k={K})"), &data, weight_column);
        let (deviance_null, nu_null, _, _) = fit_summary_with("y ~ x", &data, weight_column);

        let observations = observations as f64;
        let offset = observations * (nu_full / nu_null).ln() + (nu_null - nu_full);
        let predicted = observations * (deviance_null / deviance_full).ln() + offset;

        eprintln!(
            "[2672-algebra:{label}] W={:.10} predicted={:.10} | D_f={deviance_full:.6} \
             D_0={deviance_null:.6} ν_f={nu_full:.6} ν_0={nu_null:.6} | n={observations} \
             B(published)={:.10} B(recomputed)={offset:.10}",
            report.statistic_lr, predicted, scale.deterministic_offset,
        );

        assert_eq!(
            scale.observations, observations,
            "{label}: the reference must count the rows the likelihood sums over"
        );
        assert!(
            (scale.deterministic_offset - offset).abs() <= 1e-9 * (1.0 + offset.abs()),
            "{label}: the published deterministic offset {} is not \
             n·ln(ν_f/ν_0) + (ν_0 − ν_f) = {offset}",
            scale.deterministic_offset
        );
        // The two log-likelihoods are separately converged optimizations, so the
        // identity is asserted at the resolution of the statistic rather than at
        // machine precision — but four orders inside it, not at it.
        assert!(
            (report.statistic_lr - predicted).abs() <= 1e-6 * (1.0 + predicted.abs()),
            "{label}: W = {} but n·ln(D_0/D_f) + B = {predicted}",
            report.statistic_lr
        );
        // And the identity is not vacuous: the offset is a real part of `W`.
        assert!(
            offset.abs() > 1e-3,
            "{label}: the fixture put the deterministic offset at {offset}, too small to be \
             testing anything"
        );
    }
}

/// THE RESIDUAL LAW. `V = ε'(I − A)²ε ~ Σ_i p_i²·χ²_1 + χ²_{n−p}` with `p_i` the
/// WHOLE model's penalty shares, and those shares satisfy `Σ_i p_i = p − tr F`
/// because `F = I − H⁻¹S_λ`. `tr F` is the fit's own `edf_total`, published
/// through a completely different route, so this pins the spectrum against a
/// number the reference never touches.
#[test]
fn the_residual_spectrum_is_the_whole_models_penalty_shares() {
    init_parallelism();
    let data = fixture(true);
    let report = lr_report("gaussian", &data);
    let scale = report
        .ref_df_provenance
        .profiled_scale
        .as_ref()
        .expect("a profiled-Gaussian fit carries the estimated-scale channel");
    let (_, _, edf_total, columns) = fit_summary(&format!("y ~ x + s(z, k={K})"), &data);

    // The unit-multiplicity term counts exactly the directions no column reaches.
    assert_eq!(
        scale.residual_unit_dimension,
        (N - columns) as f64,
        "the unit block must be n − p = {N} − {columns}"
    );
    assert_eq!(
        scale.residual_weights.len(),
        columns,
        "one share per design column"
    );
    let share_sum: f64 = scale
        .residual_weights
        .iter()
        .map(|weight| weight.sqrt())
        .sum();
    eprintln!(
        "[2672-residual] Σp_i={share_sum:.8} p−edf={:.8} (p={columns}, edf={edf_total:.6}) | \
         E[V]={:.4} against ν_f",
        columns as f64 - edf_total,
        scale.residual_weights.iter().sum::<f64>() + scale.residual_unit_dimension,
    );
    assert!(
        (share_sum - (columns as f64 - edf_total)).abs() <= 1e-6 * (1.0 + edf_total),
        "Σ_i p_i = {share_sum} but p − tr F = {}",
        columns as f64 - edf_total
    );
    for weight in &scale.residual_weights {
        assert!(
            (0.0..=1.0).contains(weight) && weight.is_finite(),
            "a residual weight is p_i² ∈ [0, 1]; got {weight}"
        );
    }
}

/// THE GATE. Only a family that profiles a Gaussian scale out of a residual sum
/// of squares gets this channel; a Poisson carries its dispersion in the IRLS
/// weight and its `W` is not a function of a second estimated scalar.
#[test]
fn only_the_profiled_gaussian_carries_the_estimated_scale_channel() {
    init_parallelism();
    assert!(
        lr_report("gaussian", &fixture(true))
            .ref_df_provenance
            .profiled_scale
            .is_some()
    );
    assert!(
        lr_report("poisson", &fixture(false))
            .ref_df_provenance
            .profiled_scale
            .is_none(),
        "a Poisson fit must not be scored against a residual-sum-of-squares ratio law"
    );
}

/// THE SIGN. An estimated scale adds spread and can only cost power, so the
/// published p-value must be at or above the one the SAME reference would give
/// with the channel switched off — and materially so at this `n`.
#[test]
fn the_estimated_scale_can_only_make_the_test_more_conservative() {
    init_parallelism();
    let data = fixture(true);
    let report = lr_report("gaussian", &data);
    let mut known_scale = report.ref_df_provenance.clone();
    assert!(known_scale.profiled_scale.is_some());
    known_scale.profiled_scale = None;

    let published = report.p_value_corrected;
    let bare = known_scale.tail_probability(report.statistic_corrected);
    eprintln!(
        "[2672-sign] W*={:.6} published p={published:.6e} known-scale p={bare:.6e}",
        report.statistic_corrected
    );
    assert!(
        published >= bare - 1e-12,
        "the estimated-scale p-value {published} is BELOW the known-scale {bare}"
    );
}
