use approx::assert_abs_diff_eq;
use gam_models::marked_point_process::*;
use ndarray::{Array2, array, s};

fn control() -> LaplaceControl {
    LaplaceControl {
        max_iterations: 80,
        absolute_stationarity_tolerance: 1.0e-11,
        relative_stationarity_tolerance: 1.0e-11,
        armijo_fraction: 1.0e-4,
        step_shrink: 0.5,
        minimum_step: 1.0e-12,
    }
}

fn model(order: MaternMarkovOrder) -> MarkedPointProcessModel {
    let dimension = order.state_dimension();
    MarkedPointProcessModel {
        factors: vec![MaternFactor {
            order,
            marginal_variance: 1.7,
            length_scale: 2.3,
        }],
        mark_names: vec!["disease".to_string(), "encounter".to_string()],
        mark_roles: vec![MarkRole::Absorbing, MarkRole::Encounter],
        loadings: array![[0.8], [-0.35]],
        mark_impulses: Array2::zeros((dimension, 2)),
    }
}

fn history() -> SubjectHistory {
    SubjectHistory {
        subject: "patient-1".to_string(),
        intervals: vec![
            RiskInterval {
                entry: 0.0,
                exit: 0.5,
                counts: array![0, 1],
                fixed_log_intensity: array![-2.0, -0.2],
            },
            RiskInterval {
                entry: 0.5,
                exit: 1.0,
                counts: array![1, 0],
                fixed_log_intensity: array![-2.0, -0.2],
            },
            RiskInterval {
                entry: 1.0,
                exit: 1.8,
                counts: array![0, 1],
                fixed_log_intensity: array![-2.0, -0.2],
            },
        ],
    }
}

#[test]
fn ou_transition_preserves_stationary_variance() {
    let model = model(MaternMarkovOrder::Half);
    let stationary = model.stationary_covariance().unwrap();
    let transition = model.transition(0.73).unwrap();
    let propagated = transition
        .transition
        .dot(&stationary)
        .dot(&transition.transition.t())
        + transition.innovation_covariance;
    for (actual, expected) in propagated.iter().zip(stationary.iter()) {
        assert_abs_diff_eq!(actual, expected, epsilon = 2.0e-15);
    }
}

#[test]
fn matern_three_halves_transition_preserves_stationary_covariance() {
    let model = model(MaternMarkovOrder::ThreeHalves);
    let stationary = model.stationary_covariance().unwrap();
    let transition = model.transition(0.73).unwrap();
    let propagated = transition
        .transition
        .dot(&stationary)
        .dot(&transition.transition.t())
        + transition.innovation_covariance;
    for (actual, expected) in propagated.iter().zip(stationary.iter()) {
        assert_abs_diff_eq!(actual, expected, epsilon = 3.0e-15);
    }
}

#[test]
fn poisson_interval_derivatives_are_exact() {
    let evaluation = evaluate_poisson_interval(
        array![2_u32, 0].view(),
        array![0.3, -0.7].view(),
        0.4,
    )
    .unwrap();
    for mark in 0..2 {
        let mean = 0.4 * [0.3_f64, -0.7][mark].exp();
        assert_abs_diff_eq!(evaluation.gradient[mark], [2.0, 0.0][mark] - mean);
        assert_abs_diff_eq!(evaluation.negative_hessian[mark], mean);
    }
}

#[test]
fn global_laplace_mode_is_stationary_and_covariance_is_positive() {
    for order in [MaternMarkovOrder::Half, MaternMarkovOrder::ThreeHalves] {
        let model = model(order);
        let history = history();
        let fit = smooth_laplace(&model, &history, control()).unwrap();
        assert!(fit.stationarity < 1.0e-9);
        assert!(fit.laplace_log_marginal_likelihood.is_finite());
        assert_eq!(fit.mode.len(), history.intervals.len());
        for covariance in fit.marginal_covariances {
            assert!(covariance[[0, 0]] > 0.0);
            if covariance.nrows() == 2 {
                assert!(
                    covariance[[0, 0]] * covariance[[1, 1]]
                        - covariance[[0, 1]] * covariance[[1, 0]]
                        > 0.0
                );
            }
        }
    }
}

#[test]
fn online_filter_is_recursive_and_finite() {
    let model = model(MaternMarkovOrder::ThreeHalves);
    let filtered = filter_laplace(&model, &history(), control()).unwrap();
    assert_eq!(filtered.len(), 3);
    assert_eq!(filtered[2].mean.len(), 2);
    assert!(filtered
        .iter()
        .flat_map(|state| state.mean.iter())
        .all(|value| value.is_finite()));
}

#[test]
fn impulse_response_is_the_transition_times_impulse() {
    let mut model = model(MaternMarkovOrder::ThreeHalves);
    model.mark_impulses.slice_mut(s![.., 0]).assign(&array![1.2, -0.4]);
    let lag = 0.9;
    let expected = model
        .transition(lag)
        .unwrap()
        .transition
        .dot(&array![1.2, -0.4]);
    let actual = model.state_impulse_response(0, lag).unwrap();
    for (actual, expected) in actual.iter().zip(expected.iter()) {
        assert_abs_diff_eq!(actual, expected, epsilon = 1.0e-14);
    }
}

#[test]
fn gaussian_intensity_includes_jensen_correction() {
    let model = model(MaternMarkovOrder::Half);
    let mean = array![0.2];
    let covariance = array![[0.7]];
    let marginal = gaussian_mean_intensity(
        &model,
        array![-1.0, 0.1].view(),
        mean.view(),
        &covariance,
    )
    .unwrap();
    let plugin = (-1.0_f64 + 0.8 * 0.2).exp();
    assert!(marginal[0] > plugin);
    assert_abs_diff_eq!(
        marginal[0],
        (-1.0_f64 + 0.8 * 0.2 + 0.5 * 0.8 * 0.8 * 0.7).exp(),
        epsilon = 1.0e-14
    );
}

#[test]
fn loading_covariance_includes_factor_variance() {
    let model = model(MaternMarkovOrder::Half);
    let covariance = model.loading_covariance().unwrap();
    assert_abs_diff_eq!(covariance[[0, 0]], 1.7 * 0.8 * 0.8);
    assert_abs_diff_eq!(covariance[[0, 1]], 1.7 * 0.8 * -0.35);
    assert_abs_diff_eq!(covariance[[1, 1]], 1.7 * 0.35 * 0.35);
}

#[test]
fn forecast_integrates_paths_and_preserves_probability_mass() {
    let model = model(MaternMarkovOrder::Half);
    let landmark = filter_laplace(&model, &history(), control())
        .unwrap()
        .pop()
        .unwrap();
    let future = vec![
        ForecastInterval {
            duration: 0.25,
            fixed_log_intensity: array![-2.0, -0.2],
        },
        ForecastInterval {
            duration: 0.5,
            fixed_log_intensity: array![-2.0, -0.2],
        },
    ];
    let forecast = forecast_cumulative_incidence(
        &model,
        &landmark,
        &future,
        &[0],
        ForecastMonteCarlo {
            trajectories: 2048,
            seed: 7,
        },
    )
    .unwrap();
    for index in 0..future.len() {
        assert_abs_diff_eq!(
            forecast.survival[index] + forecast.cumulative_incidence[[index, 0]],
            1.0,
            epsilon = 2.0e-14
        );
    }
    assert!(forecast.cumulative_incidence[[1, 0]] > forecast.cumulative_incidence[[0, 0]]);
    assert!(forecast.survival[1] < forecast.survival[0]);
}
