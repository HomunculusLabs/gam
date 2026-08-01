//! #2744: a multinomial fit whose shared per-class design is rank-deficient must
//! still fit.
//!
//! `s(x1) + s(x2) + te(x1, x2)` is a legal, common multinomial formula whose
//! tensor-product term re-spans its own marginals, so the assembled per-class
//! design `X` is column-rank-deficient. The pre-fit identifiability
//! canonicalisation in `fit_custom_family` reduces every parameter block to a
//! full-rank column subset before the inner solve runs — so the block specs the
//! solver hands back to the family are NARROWER than the design the family was
//! constructed with.
//!
//! `MultinomialFamily` derived its flat coefficient layout from its OWN stored
//! design (`(K−1) · P_raw`) while every exact-Newton joint quantity it was asked
//! for was denominated in the REDUCED block widths, and its own width guard
//! refused the mismatch:
//!
//! ```text
//! MultinomialFamily joint gradient: 2 block specs carry 30 coefficients but the
//! family's flat layout is 2 classes x 19 columns = 38
//! ```
//!
//! which surfaced as `inner solve refused this trial point` — a fit refusal, not
//! a wrong number. The layout is now defined once, by the block specs in force,
//! and both the assemblies and the guard derive from it.

use gam::data::EncodedDataset;
use gam::families::multinomial::{
    MultinomialFitRequest, fit_penalized_multinomial_formula, predict_multinomial_formula,
};
use gam::{FitConfig, encode_recordswith_inferred_schema, init_parallelism};

use csv::StringRecord;
use std::f64::consts::PI;

/// Deterministic (RNG-free) three-class dataset on `[0, 2π] × [-3, 3]`, labelled
/// by the hard `argmax` of the smooth logit field `[1.5·sin(x1),
/// −0.8·cos(x1)·x2, 0]`.
fn make_dataset(n: usize) -> EncodedDataset {
    let stride1 = (2.0_f64).sqrt().fract();
    let stride2 = (3.0_f64).sqrt().fract();
    let mut u1 = 0.12_f64;
    let mut u2 = 0.37_f64;

    let headers = ["x1", "x2", "y"].into_iter().map(String::from).collect();
    let mut rows: Vec<StringRecord> = Vec::with_capacity(n);
    for _ in 0..n {
        u1 = (u1 + stride1).fract();
        u2 = (u2 + stride2).fract();
        let a = 2.0 * PI * u1;
        let b = -3.0 + 6.0 * u2;
        let l0 = 1.5 * a.sin();
        let l1 = -0.8 * a.cos() * b;
        let label = if l0 >= l1 && l0 >= 0.0 {
            "A"
        } else if l1 >= l0 && l1 >= 0.0 {
            "B"
        } else {
            "C"
        };
        rows.push(StringRecord::from(vec![
            format!("{a:.17e}"),
            format!("{b:.17e}"),
            label.to_string(),
        ]));
    }
    encode_recordswith_inferred_schema(headers, rows).expect("encode multinomial dataset")
}

#[test]
fn multinomial_fits_when_identifiability_reduces_the_block_width_2744() {
    init_parallelism();

    let data = make_dataset(400);
    // `te(x1, x2)` re-spans the `s(x1)` / `s(x2)` marginals, so the assembled
    // per-class design is rank-deficient and the canonicalisation reduces it.
    let formula = "y ~ s(x1, bs='cc', k=8) + s(x2, bs='tp', k=5) + te(x1, x2, bs=c('cc','tp'))";
    let cfg = FitConfig::default();
    let model = fit_penalized_multinomial_formula(&MultinomialFitRequest {
        data: &data,
        formula,
        config: &cfg,
        init_lambda: 1.0,
        max_iter: 50,
        tol: 1e-7,
    })
    .expect("multinomial fit must survive an identifiability-reduced block width (#2744)");

    assert_eq!(model.n_active_classes, 2, "K-1 active classes for K=3");
    assert_eq!(model.class_levels.len(), 3, "three class levels");

    // The fit must also be USABLE: a refusal repaired into a fit that cannot
    // predict on the simplex would be the same defect wearing a different coat.
    let probs = predict_multinomial_formula(&model, &data).expect("multinomial predict");
    assert_eq!(
        probs.dim(),
        (data.values.nrows(), model.class_levels.len()),
        "predicted probability shape"
    );
    for i in 0..probs.nrows() {
        let mut row_sum = 0.0_f64;
        for k in 0..probs.ncols() {
            let p = probs[[i, k]];
            assert!(
                p.is_finite() && (-1e-12..=1.0 + 1e-9).contains(&p),
                "row {i} class {k}: probability {p} off the simplex"
            );
            row_sum += p;
        }
        assert!(
            (row_sum - 1.0).abs() < 1e-9,
            "row {i}: predicted probabilities sum to {row_sum}, not 1"
        );
    }
}
