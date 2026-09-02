use gam_linalg::matrix::DesignMatrix;
use gam_models::custom_family::{BlockwiseFitOptions, PenaltyMatrix};
use gam_models::indexed_point_process::{
    IndexedPointProcessFamily, fit_indexed_point_process_laml,
};
use gam_models::output_axis::{OutputAxisPenalty, OutputBlockAxis};
use gam_problem::{
    OwnedCellValues, OwnedLikelihoodWeights, OwnedSeparableCellMeasure, OwnedStructuralCells,
};
use ndarray::{Array2, array};

#[test]
fn indexed_process_outputs_share_laml_precision_and_recover_log_intensity() {
    const N: usize = 28;
    const K: usize = 2;

    let design = Array2::from_shape_fn((N, 2), |(row, column)| {
        if column == 0 {
            1.0
        } else {
            -1.0 + 2.0 * row as f64 / (N - 1) as f64
        }
    });
    let truth = [array![-0.4, 0.9], array![0.3, -0.7]];
    let exposures = Array2::<f64>::from_elem((N, K), 30.0);
    let mut counts = Array2::<f64>::zeros((N, K));
    for row in 0..N {
        for output in 0..K {
            let eta = design.row(row).dot(&truth[output]);
            counts[[row, output]] = exposures[[row, output]] * eta.exp();
        }
    }

    let mut active = Array2::<bool>::from_elem((N, K), true);
    let mut weights = Array2::<f64>::ones((N, K));
    active[[N - 1, 0]] = false;
    counts[[N - 1, 0]] = f64::NAN;
    weights[[0, 1]] = 0.0;
    let measure = OwnedSeparableCellMeasure::new(
        N,
        K,
        OwnedStructuralCells::Dense(active),
        OwnedLikelihoodWeights::ByCell(weights),
    )
    .expect("valid indexed process measure");
    let family = IndexedPointProcessFamily::new(
        OwnedCellValues::dense(counts),
        OwnedCellValues::dense(exposures),
        measure,
    )
    .expect("valid indexed point-process family");

    let slope_penalty = OutputAxisPenalty::shared(
        PenaltyMatrix::Diagonal(array![0.0, 1.0]),
        1,
        0.0,
        "shared_process_scale",
    )
    .expect("shared structured slope penalty");
    let specs = OutputBlockAxis::new(
        (0..K).map(|output| format!("mark_{output}")).collect(),
        DesignMatrix::from(design),
    )
    .expect("replicated output block axis")
    .with_initial_coefficients(Array2::zeros((2, K)))
    .expect("coefficient seed geometry")
    .with_penalty(slope_penalty)
    .expect("replicated penalty geometry")
    .parameter_blocks();
    let options = BlockwiseFitOptions {
        inner_tol: 1e-9,
        outer_tol: 1e-7,
        outer_max_iter: 30,
        compute_covariance: true,
        auto_outer_subsample: false,
        ..BlockwiseFitOptions::default()
    };

    let fit = fit_indexed_point_process_laml(&family, &specs, &options)
        .expect("joint indexed point-process LAML fit");

    assert_eq!(fit.blocks.len(), K);
    assert_eq!(fit.block_states.len(), K);
    assert_eq!(fit.log_lambdas.len(), K);
    assert_eq!(fit.log_lambdas[0], fit.log_lambdas[1]);
    assert!(fit.log_lambdas.iter().all(|value| value.is_finite()));
    assert!(fit.log_likelihood.is_finite());
    assert!(fit.covariance_conditional.is_some());
    for (output, state) in fit.block_states.iter().enumerate() {
        assert!(state.beta.iter().all(|value| value.is_finite()));
        assert!(
            (state.beta[0] - truth[output][0]).abs() < 0.03,
            "output {output} intercept mismatch: fitted={}, truth={}",
            state.beta[0],
            truth[output][0],
        );
        assert!(
            (state.beta[1] - truth[output][1]).abs() < 0.05,
            "output {output} slope mismatch: fitted={}, truth={}",
            state.beta[1],
            truth[output][1],
        );
    }
}
