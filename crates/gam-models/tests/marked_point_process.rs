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

fn assert_close(actual: f64, expected: f64, absolute: f64, relative: f64) {
    let tolerance = absolute + relative * actual.abs().max(expected.abs());
    assert!(
        (actual - expected).abs() <= tolerance,
        "actual={actual:.17e}, expected={expected:.17e}, tolerance={tolerance:.3e}"
    );
}

fn mixed_order_model() -> MarkedPointProcessModel {
    MarkedPointProcessModel {
        factors: vec![
            MaternFactor {
                order: MaternMarkovOrder::Half,
                marginal_variance: 0.7,
                length_scale: 0.8,
            },
            MaternFactor {
                order: MaternMarkovOrder::ThreeHalves,
                marginal_variance: 2.1,
                length_scale: 3.4,
            },
        ],
        mark_names: vec![
            "disease-a".to_string(),
            "disease-b".to_string(),
            "encounter".to_string(),
        ],
        mark_roles: vec![
            MarkRole::Absorbing,
            MarkRole::Recurrent,
            MarkRole::Encounter,
        ],
        loadings: array![[0.8, -0.1], [-0.4, 0.9], [0.2, 0.3]],
        mark_impulses: Array2::zeros((3, 3)),
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
fn tiny_time_transitions_have_positive_innovation_covariance() {
    let ou = model(MaternMarkovOrder::Half).transition(2.3e-12).unwrap();
    assert!(ou.innovation_covariance[[0, 0]] > 0.0);

    let matern = model(MaternMarkovOrder::ThreeHalves)
        .transition(2.3e-8)
        .unwrap();
    let innovation = &matern.innovation_covariance;
    assert!(innovation[[0, 0]] > 0.0);
    assert!(innovation[[1, 1]] > 0.0);
    assert!(innovation[[0, 0]] * innovation[[1, 1]] - innovation[[0, 1]].powi(2) > 0.0);
}

#[test]
fn mixed_order_transitions_obey_stationarity_and_the_semigroup_law() {
    let model = mixed_order_model();
    let stationary = model.stationary_covariance().unwrap();
    for &(first_time, second_time) in &[(1.0e-7, 2.0e-7), (0.03, 0.7), (1.2, 4.8), (12.0, 31.0)] {
        let first = model.transition(first_time).unwrap();
        let second = model.transition(second_time).unwrap();
        let total = model.transition(first_time + second_time).unwrap();
        let composed_transition = second.transition.dot(&first.transition);
        let composed_innovation = second
            .transition
            .dot(&first.innovation_covariance)
            .dot(&second.transition.t())
            + second.innovation_covariance;
        for (actual, expected) in total.transition.iter().zip(composed_transition.iter()) {
            assert_close(*actual, *expected, 2.0e-14, 3.0e-13);
        }
        for (actual, expected) in total
            .innovation_covariance
            .iter()
            .zip(composed_innovation.iter())
        {
            assert_close(*actual, *expected, 2.0e-14, 5.0e-12);
        }

        let propagated = total.transition.dot(&stationary).dot(&total.transition.t())
            + &total.innovation_covariance;
        for row in 0..stationary.nrows() {
            for column in 0..stationary.ncols() {
                let scale = (stationary[[row, row]] * stationary[[column, column]]).sqrt();
                assert_close(
                    propagated[[row, column]],
                    stationary[[row, column]],
                    8.0e-14 * scale,
                    8.0e-13,
                );
            }
        }
    }
}

#[test]
fn long_time_transition_saturates_without_zero_times_infinity() {
    let model = mixed_order_model();
    let stationary = model.stationary_covariance().unwrap();
    let transition = model.transition(1.0e300).unwrap();
    assert!(transition.transition.iter().all(|value| *value == 0.0));
    assert_eq!(transition.innovation_covariance, stationary);
}

#[test]
fn compensated_matern_scales_remain_representable() {
    let model = MarkedPointProcessModel {
        factors: vec![MaternFactor {
            order: MaternMarkovOrder::ThreeHalves,
            marginal_variance: 1.0e-200,
            length_scale: 1.0e-100,
        }],
        mark_names: vec!["event".to_string()],
        mark_roles: vec![MarkRole::Recurrent],
        loadings: array![[1.0]],
        mark_impulses: Array2::zeros((2, 1)),
    };
    let stationary = model.stationary_covariance().unwrap();
    assert_close(stationary[[1, 1]], 3.0, 1.0e-14, 1.0e-14);
    let transition = model.transition(1.0e-100).unwrap();
    let propagated = transition
        .transition
        .dot(&stationary)
        .dot(&transition.transition.t())
        + transition.innovation_covariance;
    for row in 0..2 {
        for column in 0..2 {
            let scale = (stationary[[row, row]] * stationary[[column, column]]).sqrt();
            assert_close(
                propagated[[row, column]],
                stationary[[row, column]],
                2.0e-13 * scale,
                2.0e-13,
            );
        }
    }
}

#[test]
fn unrepresentable_state_scales_and_time_steps_are_rejected() {
    let mut ou = model(MaternMarkovOrder::Half);
    ou.factors[0].length_scale = f64::MAX;
    assert!(ou.transition(f64::MIN_POSITIVE).is_err());

    let mut fast_matern = model(MaternMarkovOrder::ThreeHalves);
    fast_matern.factors[0].length_scale = f64::MIN_POSITIVE;
    assert!(fast_matern.validate().is_err());

    let mut flat_matern = model(MaternMarkovOrder::ThreeHalves);
    flat_matern.factors[0].marginal_variance = f64::MIN_POSITIVE;
    flat_matern.factors[0].length_scale = f64::MAX;
    assert!(flat_matern.validate().is_err());
}

#[test]
fn poisson_interval_derivatives_are_exact() {
    let evaluation =
        evaluate_poisson_interval(array![2_u32, 0].view(), array![0.3, -0.7].view(), 0.4).unwrap();
    for mark in 0..2 {
        let mean = 0.4 * [0.3_f64, -0.7][mark].exp();
        assert_abs_diff_eq!(evaluation.gradient[mark], [2.0, 0.0][mark] - mean);
        assert_abs_diff_eq!(evaluation.negative_hessian[mark], mean);
    }
    let first_mean = 0.4 * 0.3_f64.exp();
    let second_mean = 0.4 * (-0.7_f64).exp();
    let expected_log_likelihood =
        2.0 * (0.3 + 0.4_f64.ln()) - first_mean - 2.0_f64.ln() - second_mean;
    assert_abs_diff_eq!(evaluation.log_likelihood, expected_log_likelihood);
}

#[test]
fn poisson_mean_is_evaluated_in_the_log_domain() {
    let exposure = (-710.0_f64).exp();
    let evaluation =
        evaluate_poisson_interval(array![1_u32].view(), array![710.0].view(), exposure).unwrap();
    let expected_mean = (exposure.ln() + 710.0).exp();
    assert!(evaluation.log_likelihood.is_finite());
    assert_close(
        evaluation.expected_counts[0],
        expected_mean,
        2.0e-15,
        2.0e-15,
    );
    assert_close(
        evaluation.gradient[0],
        1.0 - expected_mean,
        2.0e-15,
        2.0e-15,
    );
}

#[test]
fn poisson_derivatives_match_central_differences_across_scales() {
    for &(count, eta, exposure) in &[
        (0_u32, -12.0, 1.0e-4),
        (1, -0.7, 0.3),
        (7, 2.1, 4.0),
        (100, 5.0, 0.2),
    ] {
        let step: f64 = 1.0e-5;
        let center =
            evaluate_poisson_interval(array![count].view(), array![eta].view(), exposure).unwrap();
        let upper =
            evaluate_poisson_interval(array![count].view(), array![eta + step].view(), exposure)
                .unwrap();
        let lower =
            evaluate_poisson_interval(array![count].view(), array![eta - step].view(), exposure)
                .unwrap();
        let numerical_gradient = (upper.log_likelihood - lower.log_likelihood) / (2.0 * step);
        let numerical_hessian = (upper.log_likelihood - 2.0 * center.log_likelihood
            + lower.log_likelihood)
            / step.powi(2);
        assert_close(center.gradient[0], numerical_gradient, 2.0e-6, 2.0e-7);
        assert_close(
            -center.negative_hessian[0],
            numerical_hessian,
            3.0e-4,
            3.0e-6,
        );
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
fn cohort_evidence_is_the_sum_of_independent_subject_chains() {
    let model = model(MaternMarkovOrder::Half);
    let first = history();
    let mut second = first.clone();
    second.subject = "patient-2".to_string();
    let individual = smooth_laplace(&model, &first, control()).unwrap();
    let cohort = smooth_laplace_cohort(&model, &[first, second], control()).unwrap();
    assert_eq!(cohort.subjects.len(), 2);
    assert_abs_diff_eq!(
        cohort.laplace_log_marginal_likelihood,
        2.0 * individual.laplace_log_marginal_likelihood,
        epsilon = 2.0e-14
    );
}

#[test]
fn one_interval_filter_and_smoother_match_the_analytic_laplace_result() {
    let model = MarkedPointProcessModel {
        factors: vec![MaternFactor {
            order: MaternMarkovOrder::Half,
            marginal_variance: 1.3,
            length_scale: 2.0,
        }],
        mark_names: vec!["event".to_string()],
        mark_roles: vec![MarkRole::Recurrent],
        loadings: array![[0.7]],
        mark_impulses: Array2::zeros((1, 1)),
    };
    let history = SubjectHistory {
        subject: "analytic".to_string(),
        intervals: vec![RiskInterval {
            entry: 0.0,
            exit: 0.4,
            counts: array![2],
            fixed_log_intensity: array![-1.0],
        }],
    };
    let smoother = smooth_laplace(&model, &history, control()).unwrap();
    let filter = filter_laplace(&model, &history, control()).unwrap();
    let mode = smoother.mode[0][0];
    let mean_count = 0.4 * (-1.0_f64 + 0.7 * mode).exp();
    let precision = 1.0 / 1.3 + 0.7 * 0.7 * mean_count;
    let log_likelihood = 2.0 * (0.4_f64.ln() - 1.0 + 0.7 * mode) - mean_count - 2.0_f64.ln();
    let log_joint =
        log_likelihood - 0.5 * (mode * mode / 1.3 + 1.3_f64.ln() + std::f64::consts::TAU.ln());
    let expected_evidence = log_joint + 0.5 * std::f64::consts::TAU.ln() - 0.5 * precision.ln();

    assert_close(filter[0].mean[0], mode, 2.0e-12, 2.0e-12);
    assert_close(
        filter[0].covariance[[0, 0]],
        1.0 / precision,
        2.0e-12,
        2.0e-12,
    );
    assert_close(
        smoother.marginal_covariances[0][[0, 0]],
        1.0 / precision,
        2.0e-12,
        2.0e-12,
    );
    assert_close(
        smoother.laplace_log_marginal_likelihood,
        expected_evidence,
        3.0e-12,
        3.0e-12,
    );
}

#[test]
fn invalid_histories_models_controls_and_cohorts_are_rejected() {
    let mut invalid_model = model(MaternMarkovOrder::Half);
    invalid_model.mark_names[1] = invalid_model.mark_names[0].clone();
    assert!(invalid_model.validate().is_err());

    let overflow_exposure = SubjectHistory {
        subject: "overflow".to_string(),
        intervals: vec![RiskInterval {
            entry: -f64::MAX,
            exit: f64::MAX,
            counts: array![0],
            fixed_log_intensity: array![0.0],
        }],
    };
    assert!(overflow_exposure.validate(1).is_err());

    let mut overlapping = history();
    overlapping.intervals[1].entry = 0.4;
    assert!(overlapping.validate(2).is_err());

    let duplicate = history();
    assert!(
        smooth_laplace_cohort(
            &model(MaternMarkovOrder::Half),
            &[duplicate.clone(), duplicate],
            control(),
        )
        .is_err()
    );

    let mut invalid_control = control();
    invalid_control.step_shrink = 1.0;
    assert!(smooth_laplace(&model(MaternMarkovOrder::Half), &history(), invalid_control,).is_err());
}

#[test]
fn online_filter_is_recursive_and_finite() {
    let model = model(MaternMarkovOrder::ThreeHalves);
    let filtered = filter_laplace(&model, &history(), control()).unwrap();
    assert_eq!(filtered.len(), 3);
    assert_eq!(filtered[2].mean.len(), 2);
    assert!(
        filtered
            .iter()
            .flat_map(|state| state.mean.iter())
            .all(|value| value.is_finite())
    );
}

#[test]
fn impulse_response_is_the_transition_times_impulse() {
    let mut model = model(MaternMarkovOrder::ThreeHalves);
    model
        .mark_impulses
        .slice_mut(s![.., 0])
        .assign(&array![1.2, -0.4]);
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
    let marginal =
        gaussian_mean_intensity(&model, array![-1.0, 0.1].view(), mean.view(), &covariance)
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
fn covariance_inputs_must_be_symmetric_positive_definite() {
    let model = mixed_order_model();
    let asymmetric = array![[1.0, 0.2, 0.0], [-0.1, 1.0, 0.0], [0.0, 0.0, 1.0]];
    assert!(
        gaussian_mean_intensity(
            &model,
            array![0.0, 0.0, 0.0].view(),
            array![0.0, 0.0, 0.0].view(),
            &asymmetric,
        )
        .is_err()
    );

    let indefinite = array![[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]];
    assert!(
        gaussian_mean_intensity(
            &model,
            array![0.0, 0.0, 0.0].view(),
            array![0.0, 0.0, 0.0].view(),
            &indefinite,
        )
        .is_err()
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

#[test]
fn forecast_survival_includes_absorbing_causes_omitted_from_cif_output() {
    let model = MarkedPointProcessModel {
        factors: vec![MaternFactor {
            order: MaternMarkovOrder::Half,
            marginal_variance: 1.0,
            length_scale: 2.0,
        }],
        mark_names: vec![
            "disease".to_string(),
            "encounter".to_string(),
            "death".to_string(),
        ],
        mark_roles: vec![
            MarkRole::Absorbing,
            MarkRole::Encounter,
            MarkRole::Absorbing,
        ],
        loadings: array![[0.3], [-0.2], [0.1]],
        mark_impulses: Array2::zeros((1, 3)),
    };
    let landmark = FilteredState {
        time: 4.0,
        mean: array![0.1],
        covariance: array![[0.2]],
    };
    let future = [ForecastInterval {
        duration: 0.75,
        fixed_log_intensity: array![-1.7, -0.1, -1.2],
    }];
    let monte_carlo = ForecastMonteCarlo {
        trajectories: 2048,
        seed: 91,
    };
    let subset =
        forecast_cumulative_incidence(&model, &landmark, &future, &[0], monte_carlo).unwrap();
    let all_causes =
        forecast_cumulative_incidence(&model, &landmark, &future, &[0, 2], monte_carlo).unwrap();

    assert_abs_diff_eq!(
        subset.survival[0],
        all_causes.survival[0],
        epsilon = 1.0e-15
    );
    assert!(subset.survival[0] + subset.cumulative_incidence[[0, 0]] < 1.0);
    assert_abs_diff_eq!(
        all_causes.survival[0]
            + all_causes.cumulative_incidence[[0, 0]]
            + all_causes.cumulative_incidence[[0, 1]],
        1.0,
        epsilon = 2.0e-14
    );
}

#[test]
fn forecast_handles_extreme_log_rates_deterministically() {
    let model = MarkedPointProcessModel {
        factors: vec![MaternFactor {
            order: MaternMarkovOrder::Half,
            marginal_variance: 1.0,
            length_scale: 1.0,
        }],
        mark_names: vec!["first".to_string(), "second".to_string()],
        mark_roles: vec![MarkRole::Absorbing, MarkRole::Absorbing],
        loadings: array![[0.0], [0.0]],
        mark_impulses: Array2::zeros((1, 2)),
    };
    let landmark = FilteredState {
        time: 0.0,
        mean: array![0.0],
        covariance: array![[1.0]],
    };
    let monte_carlo = ForecastMonteCarlo {
        trajectories: 8,
        seed: 17,
    };
    let certain = forecast_cumulative_incidence(
        &model,
        &landmark,
        &[ForecastInterval {
            duration: 1.0,
            fixed_log_intensity: array![1000.0, 999.0],
        }],
        &[0, 1],
        monte_carlo,
    )
    .unwrap();
    assert_eq!(certain.survival[0], 0.0);
    assert_close(
        certain.cumulative_incidence[[0, 0]],
        1.0 / (1.0 + (-1.0_f64).exp()),
        2.0e-15,
        2.0e-15,
    );
    assert_close(
        certain.cumulative_incidence[[0, 1]],
        1.0 / (1.0 + 1.0_f64.exp()),
        2.0e-15,
        2.0e-15,
    );

    let negligible = forecast_cumulative_incidence(
        &model,
        &landmark,
        &[ForecastInterval {
            duration: 1.0,
            fixed_log_intensity: array![-1000.0, -1001.0],
        }],
        &[0, 1],
        monte_carlo,
    )
    .unwrap();
    assert_eq!(negligible.survival[0], 1.0);
    assert_eq!(
        negligible.cumulative_incidence,
        Array2::<f64>::zeros((1, 2))
    );

    let repeated = forecast_cumulative_incidence(
        &model,
        &landmark,
        &[ForecastInterval {
            duration: 1.0,
            fixed_log_intensity: array![1000.0, 999.0],
        }],
        &[0, 1],
        monte_carlo,
    )
    .unwrap();
    assert_eq!(certain, repeated);
}

#[test]
fn forecast_rejects_nonfinite_cumulative_horizon() {
    let model = model(MaternMarkovOrder::Half);
    let landmark = FilteredState {
        time: 0.0,
        mean: array![0.0],
        covariance: array![[1.0]],
    };
    let future = [
        ForecastInterval {
            duration: f64::MAX,
            fixed_log_intensity: array![-2.0, -1.0],
        },
        ForecastInterval {
            duration: f64::MAX,
            fixed_log_intensity: array![-2.0, -1.0],
        },
    ];
    assert!(
        forecast_cumulative_incidence(
            &model,
            &landmark,
            &future,
            &[0],
            ForecastMonteCarlo {
                trajectories: 2,
                seed: 3,
            },
        )
        .is_err()
    );
}
