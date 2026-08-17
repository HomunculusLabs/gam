//! gam#2600 probe: what the fitted CTN transformation does OUTSIDE the fitted
//! knot range, and what every replay path does with it.
//!
//! PROBE — prints, asserts only what it has measured.

use gam::inference::model::{FittedModel, PredictModelClass};
use gam::inference::model_payload_builders::fit_formula_to_payload;
use gam::predict::input::{
    build_transformation_normal_observed_scores, build_transformation_normal_quantile_grid,
};
use gam::probability::normal_cdf;
use gam::test_support::synthetic::SplitMixNormalRng;
use gam::{FitConfig, encode_recordswith_inferred_schema, init_parallelism};
use ndarray::{Array1, Array2};
use std::collections::HashMap;

const N: usize = 256;

#[test]
fn probe_ctn_transform_tails_2600() {
    init_parallelism();

    let mut rng = SplitMixNormalRng::new(0x2600_C7Du64);
    let z: Vec<f64> = (0..N).map(|_| rng.standard_normal()).collect();
    let y: Vec<f64> = z.iter().map(|value| value.exp()).collect();

    let headers = vec!["y".to_string()];
    let rows: Vec<csv::StringRecord> = y
        .iter()
        .map(|value| csv::StringRecord::from(vec![format!("{value:.17e}")]))
        .collect();
    let dataset = encode_recordswith_inferred_schema(headers, rows).expect("encode");
    let config = FitConfig {
        transformation_normal: true,
        ..FitConfig::default()
    };
    let payload = fit_formula_to_payload("y ~ 1".to_string(), &dataset, &config).expect("ctn fit");
    let model = FittedModel::from_payload(payload);
    assert_eq!(
        model.predict_model_class(),
        PredictModelClass::TransformationNormal
    );

    let col_map = dataset.column_map();
    let offset = Array1::<f64>::zeros(N);
    let grid = build_transformation_normal_quantile_grid(
        &model,
        dataset.values.view(),
        &col_map,
        model.training_headers.as_ref(),
        &offset,
    )
    .expect("quantile grid");

    let grid_y = grid.table.grid_y().to_owned();
    let h_grid = grid.table.latent().to_owned();
    let g = grid_y.len();
    let y_lo = grid_y[0];
    let y_hi = grid_y[g - 1];
    let lower = h_grid[[0, 0]];
    let upper = h_grid[[0, g - 1]];
    let y_min = y.iter().copied().fold(f64::INFINITY, f64::min);
    let y_max = y.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    eprintln!(
        "#2600 probe: support [{y_lo:.6e}, {y_hi:.6e}] (observed [{y_min:.6e}, {y_max:.6e}])\n\
         #2600 probe: L=h(y_lo)={lower:+.6} U=h(y_hi)={upper:+.6}  \
         Phi(L)={:.6e}  1-Phi(U)={:.6e}  (mass the clamp collapses onto the two endpoints)\n\
         #2600 probe: E[Y|x]={:.6e}  truth E[exp Z]=sqrt(e)={:.6e}",
        normal_cdf(lower),
        1.0 - normal_cdf(upper),
        grid.conditional_mean[0],
        0.5_f64.exp()
    );

    // How far does the tabulated grid say the fitted h moves per response unit
    // just inside the two boundaries?
    let slope_lo_inside = (h_grid[[0, 1]] - h_grid[[0, 0]]) / (grid_y[1] - y_lo);
    let slope_hi_inside = (h_grid[[0, g - 1]] - h_grid[[0, g - 2]]) / (y_hi - grid_y[g - 2]);
    let (tail_lo, tail_hi) = grid.table.tail_slopes(0);
    eprintln!(
        "#2600 probe: boundary secant slopes  h'(y_lo+)~{slope_lo_inside:.6e}  \
         h'(y_hi-)~{slope_hi_inside:.6e}   tail slopes carried: ({tail_lo:.6e}, {tail_hi:.6e})"
    );

    // The observed-score path, evaluated OUTSIDE the fitted support. The truth is
    // h(y) = ln y, so these should read ln(y).
    let probes: Vec<f64> = vec![
        y_lo * 0.5,
        y_lo * 0.9,
        y_hi * 1.1,
        y_hi * 2.0,
        y_hi * 10.0,
        y_hi * 1.0e6,
    ];
    let m = probes.len();
    let mut frame = Array2::<f64>::zeros((m, 1));
    for (i, &value) in probes.iter().enumerate() {
        frame[[i, 0]] = value;
    }
    let mut probe_map: HashMap<String, usize> = HashMap::new();
    probe_map.insert("y".to_string(), 0);
    let probe_response = Array1::from_vec(probes.clone());
    let scores = build_transformation_normal_observed_scores(
        &model,
        frame.view(),
        &probe_map,
        model.training_headers.as_ref(),
        &probe_response,
        &Array1::<f64>::zeros(m),
    )
    .expect("observed scores outside the support");
    for (i, &value) in probes.iter().enumerate() {
        eprintln!(
            "#2600 probe: y={value:.6e}  h_fit={:+.6}  ln y={:+.6}  (gap {:+.6})",
            scores[i],
            value.ln(),
            scores[i] - value.ln()
        );
    }

    // The predictive quantile ladder the CLI/py bands interpolate.
    for &target in &[-4.0_f64, -3.0, -2.0, 2.0, 3.0, 4.0] {
        let inverted = grid.table.invert(0, target);
        // Round trip through the model's OWN transform: the reported quantile
        // must be the response whose latent is `target`.
        let round = build_transformation_normal_observed_scores(
            &model,
            Array2::from_shape_vec((1, 1), vec![inverted]).expect("frame").view(),
            &probe_map,
            model.training_headers.as_ref(),
            &Array1::from_vec(vec![inverted]),
            &Array1::<f64>::zeros(1),
        )
        .expect("round-trip score")[0];
        eprintln!(
            "#2600 probe: z={target:+.1}  h^-1(z)={inverted:.6e}  truth exp(z)={:.6e}  \
             h(h^-1(z))={round:+.6}  clamped={}",
            target.exp(),
            inverted == grid_y[0] || inverted == grid_y[g - 1]
        );
    }
}
