//! #2612: the joint penalty `S_λ` is a MEASUREMENT of the fit, not a by-product
//! of the influence-matrix reconstruction — and a model with no penalized term
//! has one.
//!
//! The posterior-mean predictive landed on #2612 needs the penalty operator,
//! because it evaluates the penalized log-posterior away from the mode. It read
//! that operator off the `joint_recon` chain, which returns `None` when there
//! are no penalty components at all *and* when the joint posterior covariance is
//! the wrong shape. Neither is a statement about the penalty:
//!
//! * a wholly parametric multinomial (`y ~ x1 + x2`) has no penalty component,
//!   so `S_λ` is the **zero operator** — a value, not an absence;
//! * `H⁻¹` is a different measurement of a different object.
//!
//! With `S_λ` derived from that chain, every parametric multinomial fit was
//! refused outright with "converged without the coupled joint penalty operator".
//! Both arms below are the shape that regressed
//! (`quality_vs_statsmodels_multinomial`), reduced to the smallest fixture that
//! reproduces it.
//!
//! The bars are two independent statements about the SAME operator, so a repair
//! that made the payload publishable without making it the fit's own penalty
//! could not pass both:
//!
//! 1. the unpenalized arm publishes `S_λ = 0` exactly, and its published
//!    influence matrix is exactly `I` — `F = I − H⁻¹S_λ` with `S_λ = 0`;
//! 2. the penalized arm publishes an `S_λ` that reproduces its own published
//!    influence matrix through `H⁻¹S_λ = I − F`, against the covariance and
//!    influence the same fit published. That is the "the payload and `F` cannot
//!    describe different penalties" property, checked from outside the crate.

use csv::StringRecord;
use gam_data::encode_recordswith_inferred_schema;
use gam_models::fit_orchestration::FitConfig;
use gam_models::multinomial::{
    MultinomialFitRequest, fit_penalized_multinomial_formula, predict_multinomial_formula,
};
use ndarray::Array2;

const N_TRAIN: usize = 600;
const CLASS_NAMES: [&str; 3] = ["a", "b", "c"];

/// Deterministic low-discrepancy covariates and a three-class softmax label with
/// a genuine overlap region, so the fit is an ordinary interior one and neither
/// arm is testing a separation path.
fn training_records() -> Vec<StringRecord> {
    let mut rows = Vec::with_capacity(N_TRAIN);
    for index in 0..N_TRAIN {
        // Two coprime golden-ratio-style strides: no RNG, no repeats.
        let x1 = -2.0 + 4.0 * (((index as f64) * 0.618_033_988_749_894_8) % 1.0);
        let x2 = -2.0 + 4.0 * (((index as f64) * 0.414_213_562_373_095_1) % 1.0);
        // A smooth, non-degenerate class assignment: the argmax of three linear
        // scores plus a deterministic wobble that keeps every class populated
        // and the boundaries genuinely overlapped.
        let wobble = (3.0 * x1 + 1.7 * x2).sin();
        let scores = [
            0.9 * x1 - 0.4 * x2 + 0.3 * wobble,
            -0.5 * x1 + 0.8 * x2 - 0.2 * wobble,
            0.1 * x1 + 0.1 * x2 + 0.5 * wobble,
        ];
        let mut best = 0usize;
        for class in 1..3 {
            if scores[class] > scores[best] {
                best = class;
            }
        }
        rows.push(StringRecord::from(vec![
            x1.to_string(),
            x2.to_string(),
            CLASS_NAMES[best].to_string(),
        ]));
    }
    rows
}

fn fit(formula: &str) -> gam_models::multinomial::MultinomialSavedModel {
    let headers = ["x1", "x2", "y"].into_iter().map(str::to_string).collect();
    let data = encode_recordswith_inferred_schema(headers, training_records())
        .expect("encode multinomial training dataset");
    let config = FitConfig::default();
    fit_penalized_multinomial_formula(&MultinomialFitRequest {
        data: &data,
        formula,
        config: &config,
        init_lambda: 1.0,
        max_iter: 100,
        tol: 1e-8,
    })
    .unwrap_or_else(|error| panic!("multinomial formula fit `{formula}`: {error}"))
}

fn square(flat: &[f64], d: usize, what: &str) -> Array2<f64> {
    Array2::from_shape_vec((d, d), flat.to_vec())
        .unwrap_or_else(|error| panic!("{what} is not {d}x{d}: {error}"))
}

/// A multinomial with no penalized term publishes the zero penalty it has, and
/// its influence matrix is the identity that zero penalty implies.
#[test]
fn parametric_multinomial_publishes_the_zero_penalty_it_has_2612() {
    let model = fit("y ~ x1 + x2");
    let d = model.p_per_class * model.n_active_classes;
    assert_eq!(
        (model.p_per_class, model.n_active_classes),
        (3, 2),
        "expected P=3 (intercept, x1, x2) and M=2 active classes"
    );
    assert!(
        model.smooth_term_spans.is_empty() && model.lambdas.is_empty(),
        "the parametric arm must carry no smooth term and no λ, got {} span(s) and {} λ",
        model.smooth_term_spans.len(),
        model.lambdas.len(),
    );

    let penalty = model.joint_penalty().expect("published joint penalty");
    assert_eq!(penalty.dim(), (d, d), "joint penalty shape");
    let worst = penalty.iter().fold(0.0_f64, |acc, v| acc.max(v.abs()));
    assert_eq!(
        worst, 0.0,
        "an unpenalized multinomial's S_λ must be exactly zero, worst entry {worst:.3e}"
    );

    // `F = I − H⁻¹S_λ`, so `S_λ = 0` forces `F = I` exactly. This is the second,
    // independent reading of the same operator: it is not merely publishable, it
    // is the penalty the fit's own influence matrix was built from.
    let influence = model
        .coefficient_influence_flat
        .as_ref()
        .expect("published influence matrix");
    let influence = square(influence, d, "influence matrix");
    let mut worst_off_identity = 0.0_f64;
    for i in 0..d {
        for j in 0..d {
            let target = if i == j { 1.0 } else { 0.0 };
            worst_off_identity = worst_off_identity.max((influence[[i, j]] - target).abs());
        }
    }
    assert!(
        worst_off_identity <= 1e-10,
        "an unpenalized fit's influence matrix must be I; worst deviation {worst_off_identity:.3e}"
    );

    // And the estimand the penalty exists for is computable: the published
    // probabilities are a simplex on held-out rows.
    let headers = ["x1", "x2", "y"].into_iter().map(str::to_string).collect();
    let holdout = encode_recordswith_inferred_schema(
        headers,
        training_records().into_iter().step_by(97).collect(),
    )
    .expect("encode held-out frame");
    let probabilities =
        predict_multinomial_formula(&model, &holdout).expect("parametric multinomial predict");
    for row in probabilities.rows() {
        let mass: f64 = row.iter().sum();
        assert!(
            (mass - 1.0).abs() <= 1e-6 && row.iter().all(|p| (0.0..=1.0).contains(p)),
            "published probabilities must be a simplex, got {row:?} summing to {mass}"
        );
    }
}

/// The other side of the same property: with penalized terms present, the
/// published `S_λ` is the one the published influence matrix was built from.
/// `F = I − H⁻¹S_λ` is checked against the fit's OWN covariance and influence,
/// so a payload assembled from different specs or different λ fails here.
#[test]
fn published_penalty_reproduces_the_published_influence_matrix_2612() {
    let model = fit("y ~ s(x1, k=6) + s(x2, k=6)");
    let d = model.p_per_class * model.n_active_classes;
    let penalty = model.joint_penalty().expect("published joint penalty");
    let covariance = square(&model.coefficient_covariance_flat, d, "covariance");
    let influence = square(
        model
            .coefficient_influence_flat
            .as_ref()
            .expect("published influence matrix"),
        d,
        "influence matrix",
    );

    let penalty_scale = penalty.iter().fold(0.0_f64, |acc, v| acc.max(v.abs()));
    assert!(
        penalty_scale > 0.0,
        "a penalized multinomial must publish a non-zero S_λ"
    );

    // `H⁻¹S_λ` against `I − F`.
    let reconstructed = covariance.dot(&penalty);
    let mut worst = 0.0_f64;
    for i in 0..d {
        for j in 0..d {
            let target = if i == j { 1.0 } else { 0.0 } - influence[[i, j]];
            worst = worst.max((reconstructed[[i, j]] - target).abs());
        }
    }
    // The two sides are assembled from the same specs and λ but through
    // different products, so the gap is arithmetic only; the influence matrix's
    // own entries are O(1).
    assert!(
        worst <= 1e-8,
        "the published S_λ does not reproduce the published influence matrix: \
         max |H⁻¹S_λ − (I − F)| = {worst:.3e} (‖S_λ‖_max = {penalty_scale:.3e})"
    );
}
