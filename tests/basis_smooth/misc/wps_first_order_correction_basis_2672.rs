//! #2672: the RETAINED first-order smoothing-parameter correction must be
//! carried into the original coefficient basis, like every matrix it is
//! contracted against.
//!
//! `backtransform_external_result` maps the inference block out of the internal
//! (parametric-column-conditioned) basis. It back-transformed the PRIMARY
//! correction `smoothing_correction` as a covariance (`M·C·Mᵀ`) and the weighted
//! Gram `X'WX` as a congruence, but left `smoothing_correction_first_order`
//! untouched — so the WPS corrected-EDF trace `tr(X'WX · C)/φ` contracted an
//! original-basis Gram against an internal-basis correction on every model with
//! a conditioned parametric column. The two corrections are read by different
//! consumers (`model_comparison`'s corrected EDF/AIC, and since #2672 the
//! smooth-term LR reference d.f.), so nothing ever paired them and nothing ever
//! compared them.
//!
//! ## The discriminator
//!
//! The conditioning centers a parametric column by its own mean and scales it by
//! its own standard deviation, so replacing `x` with `1000·x` produces a
//! BIT-IDENTICAL internal basis: `(1000x − 1000m)/(1000s) = (x − m)/s`. The two
//! fits are therefore the same fit in two coordinate systems, and every
//! basis-invariant scalar must agree exactly — `tr(X'WX · C)` among them, since
//! the Gram transforms by `M⁻ᵀ(·)M⁻¹` and the correction by `M(·)Mᵀ`.
//!
//! With the correction left internal while the Gram is carried out, the trace
//! picks up the covariate rescaling directly: the `x` row/column of the
//! original-basis Gram moves by `10⁻⁶` between the arms while the internal-basis
//! correction does not move at all. So this comparison separates "both matrices
//! are in the same frame" from "one of them is not" without needing to know
//! which frame is right — a rescaling of an input column cannot change a
//! model's effective degrees of freedom.

use csv::StringRecord;
use gam::inference::model_comparison::model_comparison_from_unified;
use gam::{FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula};
use ndarray::Array1;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};

/// `y = 0.7·x + sin(2π z) + ε`, with the parametric column emitted at `x_scale`
/// times its natural size. `z` carries a genuine smooth signal so the fitted
/// term has real effective d.f. and a non-trivial ρ̂ covariance.
fn dataset(n: usize, seed: u64, x_scale: f64) -> (gam::data::EncodedDataset, Vec<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let noise = Normal::new(0.0, 0.3).expect("normal");
    let headers = ["y", "x", "z"].into_iter().map(String::from).collect();
    let mut ys = Vec::with_capacity(n);
    let rows: Vec<StringRecord> = (0..n)
        .map(|i| {
            let t = i as f64 / (n as f64 - 1.0);
            let z = ((i * 37) % n) as f64 / (n as f64 - 1.0);
            let y = 0.7 * t + (std::f64::consts::TAU * z).sin() + noise.sample(&mut rng);
            ys.push(y);
            StringRecord::from(vec![
                y.to_string(),
                (t * x_scale).to_string(),
                z.to_string(),
            ])
        })
        .collect();
    (
        encode_recordswith_inferred_schema(headers, rows).expect("encode"),
        ys,
    )
}

/// `(conditional edf, rho-uncertainty df)` for `y ~ x + s(z)` on one arm.
fn edf_pair(n: usize, seed: u64, x_scale: f64) -> (f64, f64) {
    let (data, ys) = dataset(n, seed, x_scale);
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let FitResult::Standard(std_fit) =
        fit_from_formula("y ~ x + s(z)", &data, &cfg).expect("Gaussian x + s(z) fit")
    else {
        panic!("expected a standard fit");
    };
    let fit = &std_fit.fit;
    let y = Array1::from(ys);
    let eta_hat = Array1::from(
        fit.artifacts
            .pirls
            .as_ref()
            .expect("converged PIRLS geometry retained")
            .final_eta
            .to_vec(),
    );
    let cmp = model_comparison_from_unified(
        fit,
        y.view(),
        eta_hat.view(),
        Array1::ones(n).view(),
        None,
    )
    .expect("model comparison from a converged Gaussian fit");
    let rho_df = cmp
        .edf
        .rho_uncertainty_df()
        .expect("the WPS corrected EDF must be retained for y ~ x + s(z)");
    (cmp.edf.conditional, rho_df)
}

#[test]
fn wps_rho_uncertainty_df_is_invariant_to_rescaling_a_parametric_column_2672() {
    let n = 200usize;
    for seed in 0..3u64 {
        let (edf_a, rho_a) = edf_pair(n, 31 + seed, 1.0);
        let (edf_b, rho_b) = edf_pair(n, 31 + seed, 1000.0);

        // Sanity: the two arms really are the same fit in two coordinate
        // systems. If the conditional EDF moved, the discriminator below would
        // be measuring a different model rather than a different basis.
        assert!(
            (edf_a - edf_b).abs() <= 1e-6 * edf_a.abs().max(1.0),
            "rescaling a parametric column must not change the conditional EDF: \
             {edf_a} vs {edf_b} (seed {seed})"
        );

        // The invariant under test.
        let tol = 1e-6 * rho_a.abs().max(1.0);
        assert!(
            (rho_a - rho_b).abs() <= tol,
            "the WPS ρ̂-uncertainty d.f. must not depend on the units of a \
             parametric covariate: {rho_a:.9e} at x-scale 1 vs {rho_b:.9e} at \
             x-scale 1000 (seed {seed}). A gap here means `tr(X'WX · C)` is \
             contracting matrices from two different coefficient bases — the \
             #2672 defect: `smoothing_correction_first_order` was left in the \
             internal basis while `weighted_gram` was carried to the original one."
        );

        // And it must remain a non-negative degrees-of-freedom increment on both
        // arms, which is the property the LR reference d.f. depends on.
        assert!(
            rho_a >= -tol && rho_b >= -tol,
            "the ρ̂-uncertainty d.f. must be non-negative on both arms: \
             {rho_a:.9e} / {rho_b:.9e} (seed {seed})"
        );
    }
}
