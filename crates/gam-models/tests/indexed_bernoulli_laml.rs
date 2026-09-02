use gam_linalg::matrix::DesignMatrix;
use gam_models::custom_family::{BlockwiseFitOptions, ParameterBlockSpec, PenaltyMatrix};
use gam_models::indexed_bernoulli::{IndexedBernoulliFamily, fit_indexed_bernoulli_laml};
use gam_problem::{
    OwnedLikelihoodWeights, OwnedSeparableCellMeasure, OwnedStructuralCells,
};
use gam_spec::{InverseLink, StandardLink};
use ndarray::{Array1, Array2, array};

#[test]
fn indexed_outputs_share_one_selected_precision_without_losing_risk_geometry() {
    const N: usize = 24;
    const K: usize = 2;

    let design = Array2::from_shape_fn((N, 2), |(row, column)| {
        if column == 0 {
            1.0
        } else {
            -1.0 + 2.0 * row as f64 / (N - 1) as f64
        }
    });
    let mut response = Array2::<f64>::zeros((N, K));
    let mut active = Array2::<bool>::from_elem((N, K), true);
    let mut weights = Array2::<f64>::from_elem((N, K), 20.0);
    for row in 0..N {
        let x = design[[row, 1]];
        response[[row, 0]] = 1.0 / (1.0 + (-(-0.35 + 1.15 * x)).exp());
        response[[row, 1]] = 1.0 / (1.0 + (-(0.20 - 0.85 * x)).exp());
    }

    // These cells exercise the two independent layers of response geometry:
    // an inactive cell is absent and may carry NaN, while an active zero-mass
    // cell remains part of the declared risk set but contributes no likelihood.
    active[[N - 1, 0]] = false;
    response[[N - 1, 0]] = f64::NAN;
    weights[[0, 1]] = 0.0;
    let measure = OwnedSeparableCellMeasure::new(
        N,
        K,
        OwnedStructuralCells::Dense(active),
        OwnedLikelihoodWeights::ByCell(weights),
    )
    .expect("valid indexed response measure");
    let family = IndexedBernoulliFamily::new(
        response.view(),
        measure,
        InverseLink::Standard(StandardLink::Logit),
    )
    .expect("valid indexed Bernoulli family");

    let slope_penalty = array![[0.0, 0.0], [0.0, 1.0]];
    let specs: Vec<_> = (0..K)
        .map(|output| ParameterBlockSpec {
            name: format!("endpoint_{output}"),
            design: DesignMatrix::from(design.clone()),
            offset: Array1::zeros(N),
            penalties: vec![PenaltyMatrix::Dense(slope_penalty.clone())
                .with_precision_label("shared_history_scale")],
            nullspace_dims: vec![1],
            initial_log_lambdas: array![0.0],
            initial_beta: Some(Array1::zeros(2)),
            ..ParameterBlockSpec::defaults()
        })
        .collect();
    let options = BlockwiseFitOptions {
        inner_tol: 1e-9,
        outer_tol: 1e-7,
        outer_max_iter: 30,
        compute_covariance: true,
        auto_outer_subsample: false,
        ..BlockwiseFitOptions::default()
    };

    let fit = fit_indexed_bernoulli_laml(&family, &specs, &options)
        .expect("joint indexed Bernoulli LAML fit");

    assert_eq!(fit.blocks.len(), K);
    assert_eq!(fit.block_states.len(), K);
    // UnifiedFitResult retains physical penalty coordinates. Equal values here
    // prove that both physical penalties were driven by one labeled outer rho.
    assert_eq!(fit.log_lambdas.len(), K);
    assert_eq!(fit.log_lambdas[0], fit.log_lambdas[1]);
    assert!(fit.log_lambdas.iter().all(|value| value.is_finite()));
    assert!(fit.log_likelihood.is_finite());
    assert!(fit.covariance_conditional.is_some());

    let truth = [array![-0.35, 1.15], array![0.20, -0.85]];
    for (output, state) in fit.block_states.iter().enumerate() {
        assert!(state.beta.iter().all(|value| value.is_finite()));
        assert!(
            (state.beta[0] - truth[output][0]).abs() < 0.08,
            "output {output} intercept mismatch: fitted={}, truth={}",
            state.beta[0],
            truth[output][0],
        );
        assert!(
            (state.beta[1] - truth[output][1]).abs() < 0.12,
            "output {output} slope mismatch: fitted={}, truth={}",
            state.beta[1],
            truth[output][1],
        );
    }
}
