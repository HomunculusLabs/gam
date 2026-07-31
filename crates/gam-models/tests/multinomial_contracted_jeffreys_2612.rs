use gam_models::custom_family::{
    CustomFamily, ParameterBlockState, PenaltyMatrix,
};
use gam_models::MultinomialFamily;
use ndarray::{Array1, Array2};
use std::sync::Arc;

#[test]
fn contracted_jeffreys_trace_hessian_matches_every_pairwise_axis() {
    let n = 7;
    let p = 2;
    let k = 4;
    let m = k - 1;
    let dim = m * p;
    let mut response = Array2::<f64>::zeros((n, k));
    for row in 0..n {
        response[[row, row % k]] = 1.0;
    }
    let design = Arc::new(Array2::<f64>::from_shape_fn((n, p), |(row, column)| {
        ((row + column + 1) as f64).sin()
    }));
    let family = MultinomialFamily::new(
        response,
        Array1::<f64>::ones(n),
        k,
        Arc::clone(&design),
        Arc::new(vec![PenaltyMatrix::Dense(Array2::<f64>::eye(p))]),
    )
    .expect("the multinomial fixture is valid");
    let states: Vec<ParameterBlockState> = (0..m)
        .map(|class| {
            let beta = Array1::<f64>::from_shape_fn(p, |column| {
                0.19 * (class + 1) as f64 - 0.11 * column as f64
            });
            let eta = design.dot(&beta);
            ParameterBlockState { beta, eta }
        })
        .collect();
    let specs = family.build_block_specs();
    let weight = Array2::<f64>::from_shape_fn((dim, dim), |(row, column)| {
        ((row + column + 1) as f64).sin()
    });
    let contracted = family
        .joint_jeffreys_information_contracted_trace_hessian_with_specs(
            &states,
            &specs,
            &weight,
        )
        .expect("the fused contraction succeeds")
        .expect("multinomial exposes the fused contraction");
    assert!(
        family.joint_jeffreys_information_contracted_trace_hessian_available(),
        "the availability contract advertises the implemented fused path"
    );
    for left in 0..dim {
        let mut e_left = Array1::<f64>::zeros(dim);
        e_left[left] = 1.0;
        for right in 0..dim {
            let mut e_right = Array1::<f64>::zeros(dim);
            e_right[right] = 1.0;
            let pairwise = family
                .exact_newton_joint_hessiansecond_directional_derivative(
                    &states,
                    &e_left,
                    &e_right,
                )
                .expect("the pairwise second derivative succeeds")
                .expect("multinomial exposes exact second derivatives");
            let expected = weight
                .iter()
                .zip(pairwise.iter())
                .map(|(&trace, &second)| trace * second)
                .sum::<f64>();
            let actual = contracted[[left, right]];
            assert!(
                (actual - expected).abs() <= 2.0e-10 * (1.0 + expected.abs()),
                "fused contraction[{left},{right}]={actual:.16e} != pairwise \
                 trace={expected:.16e}"
            );
        }
    }
}

#[test]
fn batched_jeffreys_axis_derivatives_match_the_defining_directional_hooks() {
    let n = 7;
    let p = 3;
    let k = 4;
    let m = k - 1;
    let dim = m * p;
    let mut response = Array2::<f64>::zeros((n, k));
    for row in 0..n {
        response[[row, (2 * row + 1) % k]] = 1.0;
    }
    let design = Arc::new(Array2::<f64>::from_shape_fn((n, p), |(row, column)| {
        let phase = (row + 2 * column + 1) as f64;
        phase.sin() + 0.3 * (phase / 2.0).cos()
    }));
    let family = MultinomialFamily::new(
        response,
        Array1::<f64>::ones(n),
        k,
        Arc::clone(&design),
        Arc::new(vec![PenaltyMatrix::Dense(Array2::<f64>::eye(p))]),
    )
    .expect("the multinomial fixture is valid");
    let states: Vec<ParameterBlockState> = (0..m)
        .map(|class| {
            let beta = Array1::<f64>::from_shape_fn(p, |column| {
                0.13 * (class + 1) as f64 - 0.07 * (column + 1) as f64
            });
            let eta = design.dot(&beta);
            ParameterBlockState { beta, eta }
        })
        .collect();
    let specs = family.build_block_specs();

    let batched_first = family
        .joint_jeffreys_information_directional_derivative_all_axes_with_specs(
            &states,
            &specs,
        )
        .expect("the batched first derivative succeeds")
        .expect("multinomial exposes batched first derivatives");
    assert_eq!(batched_first.len(), dim);

    let fixed_direction =
        Array1::<f64>::from_shape_fn(dim, |axis| 0.17 * (axis + 1) as f64 - 0.31);
    let batched_second = family
        .joint_jeffreys_information_second_directional_all_axes_with_specs(
            &states,
            &specs,
            &fixed_direction,
        )
        .expect("the batched second derivative succeeds")
        .expect("multinomial exposes batched second derivatives");
    assert_eq!(batched_second.len(), dim);

    for axis in 0..dim {
        let mut unit_axis = Array1::<f64>::zeros(dim);
        unit_axis[axis] = 1.0;
        let expected_first = family
            .joint_jeffreys_information_directional_derivative_with_specs(
                &states,
                &specs,
                &unit_axis,
            )
            .expect("the defining first directional derivative succeeds")
            .expect("multinomial exposes first directional derivatives");
        let expected_second = family
            .joint_jeffreys_information_second_directional_derivative_with_specs(
                &states,
                &specs,
                &fixed_direction,
                &unit_axis,
            )
            .expect("the defining second directional derivative succeeds")
            .expect("multinomial exposes second directional derivatives");

        for (label, actual, expected) in [
            ("first", &batched_first[axis], &expected_first),
            ("second", &batched_second[axis], &expected_second),
        ] {
            for (entry, (&actual, &expected)) in
                actual.iter().zip(expected.iter()).enumerate()
            {
                assert!(
                    (actual - expected).abs() <= 2.0e-10 * (1.0 + expected.abs()),
                    "batched {label} derivative axis {axis}, entry {entry}: \
                     {actual:.16e} != defining hook {expected:.16e}"
                );
            }
        }
    }
}
