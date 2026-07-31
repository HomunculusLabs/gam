use gam_math::probability::normal_logcdf_derivatives;
use gam_row_macros::row_program;
use std::time::Instant;

fn supplied_link_stack(
    composition_point: f64,
    value: f64,
    first: f64,
    second: f64,
    third: f64,
    fourth: f64,
) -> [f64; 5] {
    if composition_point.is_nan() {
        [f64::NAN; 5]
    } else {
        [value, first, second, third, fourth]
    }
}

fn observed_scale_stack(observed_slope: f64) -> [f64; 5] {
    let scale = (1.0 + observed_slope * observed_slope).sqrt();
    let inverse = scale.recip();
    let inverse_squared = inverse * inverse;
    let inverse_cubed = inverse_squared * inverse;
    let inverse_fifth = inverse_cubed * inverse_squared;
    let inverse_seventh = inverse_fifth * inverse_squared;
    [
        scale,
        observed_slope * inverse,
        inverse_cubed,
        -3.0 * observed_slope * inverse_fifth,
        (12.0 * observed_slope * observed_slope - 3.0) * inverse_seventh,
    ]
}

fn signed_probit_stack(signed_margin: f64, weight: f64) -> [f64; 5] {
    if weight == 0.0 || signed_margin == f64::INFINITY {
        return [0.0; 5];
    }
    if signed_margin.is_nan() {
        return [f64::NAN; 5];
    }
    let derivatives = normal_logcdf_derivatives(signed_margin);
    derivatives.map(|derivative| -weight * derivative)
}

row_program! {
    fn generated_rigid_bms(
        marginal_eta,
        logslope;
        marginal_q,
        marginal_q1,
        marginal_q2,
        marginal_q3,
        marginal_q4,
        probit_scale,
        latent_score,
        outcome_sign,
        weight
    )
    emit [generic, order2, third, fourth, full];
    leaves {
        supplied_link => supplied_link_stack => supplied_link_stack_cuda,
        observed_scale => observed_scale_stack => observed_scale_stack_cuda,
        signed_probit => signed_probit_stack => signed_probit_stack_cuda,
    }
    witnesses [];
    {
        let q = compose(
            supplied_link,
            marginal_eta,
            marginal_q,
            marginal_q1,
            marginal_q2,
            marginal_q3,
            marginal_q4
        );
        let observed_logslope = scale(logslope, probit_scale);
        let observed_scale_value = compose(observed_scale, observed_logslope);
        let latent_index = add(
            mul(q, observed_scale_value),
            scale(observed_logslope, latent_score)
        );
        let signed_margin = scale(latent_index, outcome_sign);
        return compose(signed_probit, signed_margin, weight);
    }
}

type Channels = (f64, [f64; 2], [[f64; 2]; 2]);

#[derive(Clone, Copy)]
struct Row {
    marginal_eta: f64,
    logslope: f64,
    marginal: [f64; 5],
    probit_scale: f64,
    latent_score: f64,
    outcome_sign: f64,
    weight: f64,
    direction_u: [f64; 2],
    direction_v: [f64; 2],
}

#[inline(never)]
fn generated(row: Row) -> Channels {
    let (value, gradient, hessian, []) = generated_rigid_bms_order2(
        row.marginal_eta,
        row.logslope,
        row.marginal[0],
        row.marginal[1],
        row.marginal[2],
        row.marginal[3],
        row.marginal[4],
        row.probit_scale,
        row.latent_score,
        row.outcome_sign,
        row.weight,
    );
    (value, gradient, hessian)
}

#[inline(never)]
fn strongest_hand(row: Row) -> Channels {
    let observed_slope = row.probit_scale * row.logslope;
    let scale = (1.0 + observed_slope * observed_slope).sqrt();
    let inverse_scale = scale.recip();
    let scale_first = row.probit_scale * observed_slope * inverse_scale;
    let scale_second =
        row.probit_scale * row.probit_scale * inverse_scale * inverse_scale * inverse_scale;
    let latent_index = row.marginal[0] * scale + observed_slope * row.latent_score;
    let signed_margin = row.outcome_sign * latent_index;
    let outer = signed_probit_stack(signed_margin, row.weight);
    let outer_first = row.outcome_sign * outer[1];
    let outer_second = outer[2];
    let eta_marginal = row.marginal[1] * scale;
    let eta_logslope = row.marginal[0] * scale_first + row.probit_scale * row.latent_score;
    let gradient = [outer_first * eta_marginal, outer_first * eta_logslope];
    let cross =
        outer_second * eta_marginal * eta_logslope + outer_first * row.marginal[1] * scale_first;
    (
        outer[0],
        gradient,
        [
            [
                outer_second * eta_marginal * eta_marginal + outer_first * row.marginal[2] * scale,
                cross,
            ],
            [
                cross,
                outer_second * eta_logslope * eta_logslope
                    + outer_first * row.marginal[0] * scale_second,
            ],
        ],
    )
}

#[inline(always)]
fn generated_third(row: Row) -> [[f64; 2]; 2] {
    generated_rigid_bms_third_contracted(
        row.marginal_eta,
        row.logslope,
        row.marginal[0],
        row.marginal[1],
        row.marginal[2],
        row.marginal[3],
        row.marginal[4],
        row.probit_scale,
        row.latent_score,
        row.outcome_sign,
        row.weight,
        &row.direction_u,
    )
}

#[inline(always)]
fn generated_fourth(row: Row) -> [[f64; 2]; 2] {
    generated_rigid_bms_fourth_contracted(
        row.marginal_eta,
        row.logslope,
        row.marginal[0],
        row.marginal[1],
        row.marginal[2],
        row.marginal[3],
        row.marginal[4],
        row.probit_scale,
        row.latent_score,
        row.outcome_sign,
        row.weight,
        &row.direction_u,
        &row.direction_v,
    )
}

#[inline(always)]
fn generated_third_full(row: Row) -> [[[f64; 2]; 2]; 2] {
    generated_rigid_bms_third_full(
        row.marginal_eta,
        row.logslope,
        row.marginal[0],
        row.marginal[1],
        row.marginal[2],
        row.marginal[3],
        row.marginal[4],
        row.probit_scale,
        row.latent_score,
        row.outcome_sign,
        row.weight,
    )
}

#[inline(always)]
fn generated_fourth_full(row: Row) -> [[[[f64; 2]; 2]; 2]; 2] {
    generated_rigid_bms_fourth_full(
        row.marginal_eta,
        row.logslope,
        row.marginal[0],
        row.marginal[1],
        row.marginal[2],
        row.marginal[3],
        row.marginal[4],
        row.probit_scale,
        row.latent_score,
        row.outcome_sign,
        row.weight,
    )
}

#[inline(always)]
fn margin_chain(row: Row) -> ([f64; 5], [f64; 2], [f64; 3], [f64; 4], [f64; 5]) {
    let observed_slope = row.probit_scale * row.logslope;
    let observed_stack = observed_scale_stack(observed_slope);
    let mut scale_stack = [0.0; 5];
    let mut scale_power = 1.0;
    for order in 0..5 {
        scale_stack[order] = observed_stack[order] * scale_power;
        scale_power *= row.probit_scale;
    }
    let derivative = |order: usize, g_axes: usize| {
        let eta_axes = order - g_axes;
        let linear = if eta_axes == 0 && g_axes == 1 {
            row.probit_scale * row.latent_score
        } else {
            0.0
        };
        row.outcome_sign * (row.marginal[eta_axes] * scale_stack[g_axes] + linear)
    };
    let signed_margin = row.outcome_sign
        * (row.marginal[0] * observed_stack[0] + observed_slope * row.latent_score);
    (
        signed_probit_stack(signed_margin, row.weight),
        std::array::from_fn(|g_axes| derivative(1, g_axes)),
        std::array::from_fn(|g_axes| derivative(2, g_axes)),
        std::array::from_fn(|g_axes| derivative(3, g_axes)),
        std::array::from_fn(|g_axes| derivative(4, g_axes)),
    )
}

#[inline(always)]
fn dot2(left: [f64; 2], right: [f64; 2]) -> f64 {
    left[0] * right[0] + left[1] * right[1]
}

#[inline(always)]
fn contract2(derivative: &[f64; 3], left: [f64; 2], right: [f64; 2]) -> f64 {
    derivative[0] * left[0] * right[0]
        + derivative[1] * (left[0] * right[1] + left[1] * right[0])
        + derivative[2] * left[1] * right[1]
}

#[inline(never)]
fn strongest_hand_third(row: Row) -> [[f64; 2]; 2] {
    let (outer, d1, d2, d3, _) = margin_chain(row);
    let margin_u = dot2(d1, row.direction_u);
    std::array::from_fn(|axis_a| {
        std::array::from_fn(|axis_b| {
            let margin_a = d1[axis_a];
            let margin_b = d1[axis_b];
            let margin_ab = d2[axis_a + axis_b];
            let margin_au = d2[axis_a] * row.direction_u[0] + d2[axis_a + 1] * row.direction_u[1];
            let margin_bu = d2[axis_b] * row.direction_u[0] + d2[axis_b + 1] * row.direction_u[1];
            let margin_abu = d3[axis_a + axis_b] * row.direction_u[0]
                + d3[axis_a + axis_b + 1] * row.direction_u[1];
            outer[3] * margin_u * margin_a * margin_b
                + outer[2] * (margin_au * margin_b + margin_a * margin_bu + margin_u * margin_ab)
                + outer[1] * margin_abu
        })
    })
}

#[inline(never)]
fn strongest_hand_fourth(row: Row) -> [[f64; 2]; 2] {
    let (outer, d1, d2, d3, d4) = margin_chain(row);
    let margin_u = dot2(d1, row.direction_u);
    let margin_v = dot2(d1, row.direction_v);
    let margin_uv = contract2(&d2, row.direction_u, row.direction_v);
    std::array::from_fn(|axis_a| {
        std::array::from_fn(|axis_b| {
            let margin_a = d1[axis_a];
            let margin_b = d1[axis_b];
            let margin_ab = d2[axis_a + axis_b];
            let margin_au = d2[axis_a] * row.direction_u[0] + d2[axis_a + 1] * row.direction_u[1];
            let margin_av = d2[axis_a] * row.direction_v[0] + d2[axis_a + 1] * row.direction_v[1];
            let margin_bu = d2[axis_b] * row.direction_u[0] + d2[axis_b + 1] * row.direction_u[1];
            let margin_bv = d2[axis_b] * row.direction_v[0] + d2[axis_b + 1] * row.direction_v[1];
            let margin_abu = d3[axis_a + axis_b] * row.direction_u[0]
                + d3[axis_a + axis_b + 1] * row.direction_u[1];
            let margin_abv = d3[axis_a + axis_b] * row.direction_v[0]
                + d3[axis_a + axis_b + 1] * row.direction_v[1];
            let margin_auv = d3[axis_a] * row.direction_u[0] * row.direction_v[0]
                + d3[axis_a + 1]
                    * (row.direction_u[0] * row.direction_v[1]
                        + row.direction_u[1] * row.direction_v[0])
                + d3[axis_a + 2] * row.direction_u[1] * row.direction_v[1];
            let margin_buv = d3[axis_b] * row.direction_u[0] * row.direction_v[0]
                + d3[axis_b + 1]
                    * (row.direction_u[0] * row.direction_v[1]
                        + row.direction_u[1] * row.direction_v[0])
                + d3[axis_b + 2] * row.direction_u[1] * row.direction_v[1];
            let margin_abuv = d4[axis_a + axis_b] * row.direction_u[0] * row.direction_v[0]
                + d4[axis_a + axis_b + 1]
                    * (row.direction_u[0] * row.direction_v[1]
                        + row.direction_u[1] * row.direction_v[0])
                + d4[axis_a + axis_b + 2] * row.direction_u[1] * row.direction_v[1];
            let second_chain = margin_au * margin_b + margin_a * margin_bu + margin_u * margin_ab;
            let second_chain_v = margin_auv * margin_b
                + margin_au * margin_bv
                + margin_av * margin_bu
                + margin_a * margin_buv
                + margin_uv * margin_ab
                + margin_u * margin_abv;
            outer[4] * margin_v * margin_u * margin_a * margin_b
                + outer[3]
                    * (margin_uv * margin_a * margin_b
                        + margin_u * margin_av * margin_b
                        + margin_u * margin_a * margin_bv
                        + margin_v * second_chain)
                + outer[2] * (second_chain_v + margin_v * margin_abu)
                + outer[1] * margin_abuv
        })
    })
}

#[inline(never)]
fn strongest_hand_third_full(row: Row) -> [[[f64; 2]; 2]; 2] {
    let (outer, d1, d2, d3, _) = margin_chain(row);
    std::array::from_fn(|a| {
        std::array::from_fn(|b| {
            std::array::from_fn(|c| {
                outer[3] * d1[a] * d1[b] * d1[c]
                    + outer[2] * (d2[a + b] * d1[c] + d2[a + c] * d1[b] + d2[b + c] * d1[a])
                    + outer[1] * d3[a + b + c]
            })
        })
    })
}

#[inline(never)]
fn strongest_hand_fourth_full(row: Row) -> [[[[f64; 2]; 2]; 2]; 2] {
    let (outer, d1, d2, d3, d4) = margin_chain(row);
    std::array::from_fn(|a| {
        std::array::from_fn(|b| {
            std::array::from_fn(|c| {
                std::array::from_fn(|d| {
                    outer[4] * d1[a] * d1[b] * d1[c] * d1[d]
                        + outer[3]
                            * (d2[a + b] * d1[c] * d1[d]
                                + d2[a + c] * d1[b] * d1[d]
                                + d2[a + d] * d1[b] * d1[c]
                                + d2[b + c] * d1[a] * d1[d]
                                + d2[b + d] * d1[a] * d1[c]
                                + d2[c + d] * d1[a] * d1[b])
                        + outer[2]
                            * (d3[a + b + c] * d1[d]
                                + d3[a + b + d] * d1[c]
                                + d3[a + c + d] * d1[b]
                                + d3[b + c + d] * d1[a]
                                + d2[a + b] * d2[c + d]
                                + d2[a + c] * d2[b + d]
                                + d2[a + d] * d2[b + c])
                        + outer[1] * d4[a + b + c + d]
                })
            })
        })
    })
}

fn rows() -> Vec<Row> {
    (0..512)
        .map(|index| {
            let x = index as f64;
            let marginal_eta = 0.9 * (x * 0.17 + 0.3).sin();
            let q = 0.8 * marginal_eta + 0.12 * marginal_eta * marginal_eta;
            Row {
                marginal_eta,
                logslope: 0.7 * (x * 0.11 + 0.4).cos(),
                marginal: [q, 0.8 + 0.24 * marginal_eta, 0.24, 0.0, 0.0],
                probit_scale: 0.8,
                latent_score: 1.4 * (x * 0.13 + 0.2).sin(),
                outcome_sign: if index % 2 == 0 { 1.0 } else { -1.0 },
                weight: 0.55 + 0.45 * (x * 0.19 + 1.0).sin().abs(),
                direction_u: [0.7 * (x * 0.23 + 0.4).cos(), -0.6 * (x * 0.29 + 0.1).sin()],
                direction_v: [-0.5 * (x * 0.21 + 0.9).sin(), 0.8 * (x * 0.27 + 0.5).cos()],
            }
        })
        .collect()
}

fn close(got: f64, want: f64) {
    let tolerance = 3e-12 * got.abs().max(want.abs()).max(1.0);
    assert!(
        (got - want).abs() <= tolerance,
        "{got:+.16e} vs {want:+.16e}"
    );
}

fn assert_channels(got: Channels, want: Channels) {
    close(got.0, want.0);
    for axis in 0..2 {
        close(got.1[axis], want.1[axis]);
        for other in 0..2 {
            close(got.2[axis][other], want.2[axis][other]);
        }
    }
}

fn assert_matrix(got: [[f64; 2]; 2], want: [[f64; 2]; 2]) {
    for axis in 0..2 {
        for other in 0..2 {
            close(got[axis][other], want[axis][other]);
        }
    }
}

fn assert_third_full(got: [[[f64; 2]; 2]; 2], want: [[[f64; 2]; 2]; 2]) {
    for a in 0..2 {
        for b in 0..2 {
            for c in 0..2 {
                close(got[a][b][c], want[a][b][c]);
            }
        }
    }
}

fn assert_fourth_full(got: [[[[f64; 2]; 2]; 2]; 2], want: [[[[f64; 2]; 2]; 2]; 2]) {
    for a in 0..2 {
        for b in 0..2 {
            for c in 0..2 {
                for d in 0..2 {
                    close(got[a][b][c][d], want[a][b][c][d]);
                }
            }
        }
    }
}

fn sample(rows: &[Row], evaluate: impl Fn(Row) -> Channels) -> f64 {
    let started = Instant::now();
    let mut checksum = 0.0;
    for iteration in 0..4096 {
        for row in rows {
            let mut perturbed = *row;
            perturbed.logslope += checksum * 1e-24;
            let channels = std::hint::black_box(evaluate(perturbed));
            checksum += channels.0
                + channels.1.iter().sum::<f64>()
                + channels.2.iter().flat_map(|line| line.iter()).sum::<f64>();
        }
        checksum *= 1e-3 + iteration as f64 * 1e-12;
    }
    assert!(checksum.is_finite());
    started.elapsed().as_secs_f64() * 1e9 / (rows.len() * 4096) as f64
}

fn sample_matrix(rows: &[Row], evaluate: impl Fn(Row) -> [[f64; 2]; 2]) -> f64 {
    let started = Instant::now();
    let mut checksum = 0.0;
    for iteration in 0..4096 {
        for row in rows {
            let mut perturbed = *row;
            perturbed.logslope += checksum * 1e-24;
            let matrix = std::hint::black_box(evaluate(perturbed));
            checksum += matrix.iter().flat_map(|line| line.iter()).sum::<f64>();
        }
        checksum *= 1e-3 + iteration as f64 * 1e-12;
    }
    assert!(checksum.is_finite());
    started.elapsed().as_secs_f64() * 1e9 / (rows.len() * 4096) as f64
}

fn sample_third_full(rows: &[Row], evaluate: impl Fn(Row) -> [[[f64; 2]; 2]; 2]) -> f64 {
    let started = Instant::now();
    let mut checksum = 0.0;
    for iteration in 0..4096 {
        for row in rows {
            let mut perturbed = *row;
            perturbed.logslope += checksum * 1e-24;
            let tensor = std::hint::black_box(evaluate(perturbed));
            checksum += tensor
                .iter()
                .flat_map(|plane| plane.iter())
                .flat_map(|line| line.iter())
                .sum::<f64>();
        }
        checksum *= 1e-3 + iteration as f64 * 1e-12;
    }
    assert!(checksum.is_finite());
    started.elapsed().as_secs_f64() * 1e9 / (rows.len() * 4096) as f64
}

fn sample_fourth_full(rows: &[Row], evaluate: impl Fn(Row) -> [[[[f64; 2]; 2]; 2]; 2]) -> f64 {
    let started = Instant::now();
    let mut checksum = 0.0;
    for iteration in 0..4096 {
        for row in rows {
            let mut perturbed = *row;
            perturbed.logslope += checksum * 1e-24;
            let tensor = std::hint::black_box(evaluate(perturbed));
            checksum += tensor
                .iter()
                .flat_map(|cube| cube.iter())
                .flat_map(|plane| plane.iter())
                .flat_map(|line| line.iter())
                .sum::<f64>();
        }
        checksum *= 1e-3 + iteration as f64 * 1e-12;
    }
    assert!(checksum.is_finite());
    started.elapsed().as_secs_f64() * 1e9 / (rows.len() * 4096) as f64
}

fn median(mut values: [f64; 7]) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

#[test]
fn generated_rigid_bms_matches_strongest_hand_932() {
    let rows = rows();
    for row in &rows {
        assert_channels(generated(*row), strongest_hand(*row));
        assert_matrix(generated_third(*row), strongest_hand_third(*row));
        assert_matrix(generated_fourth(*row), strongest_hand_fourth(*row));
        assert_third_full(generated_third_full(*row), strongest_hand_third_full(*row));
        assert_fourth_full(
            generated_fourth_full(*row),
            strongest_hand_fourth_full(*row),
        );
    }

    let mut order2_samples = [(0.0, 0.0); 7];
    let mut third_samples = [(0.0, 0.0); 7];
    let mut fourth_samples = [(0.0, 0.0); 7];
    let mut third_full_samples = [(0.0, 0.0); 7];
    let mut fourth_full_samples = [(0.0, 0.0); 7];
    for round in 0..7 {
        if round % 2 == 0 {
            order2_samples[round] = (sample(&rows, generated), sample(&rows, strongest_hand));
            third_samples[round] = (
                sample_matrix(&rows, generated_third),
                sample_matrix(&rows, strongest_hand_third),
            );
            fourth_samples[round] = (
                sample_matrix(&rows, generated_fourth),
                sample_matrix(&rows, strongest_hand_fourth),
            );
            third_full_samples[round] = (
                sample_third_full(&rows, generated_third_full),
                sample_third_full(&rows, strongest_hand_third_full),
            );
            fourth_full_samples[round] = (
                sample_fourth_full(&rows, generated_fourth_full),
                sample_fourth_full(&rows, strongest_hand_fourth_full),
            );
        } else {
            let hand_ns = sample(&rows, strongest_hand);
            let generated_ns = sample(&rows, generated);
            order2_samples[round] = (generated_ns, hand_ns);
            let hand_ns = sample_matrix(&rows, strongest_hand_third);
            let generated_ns = sample_matrix(&rows, generated_third);
            third_samples[round] = (generated_ns, hand_ns);
            let hand_ns = sample_matrix(&rows, strongest_hand_fourth);
            let generated_ns = sample_matrix(&rows, generated_fourth);
            fourth_samples[round] = (generated_ns, hand_ns);
            let hand_ns = sample_third_full(&rows, strongest_hand_third_full);
            let generated_ns = sample_third_full(&rows, generated_third_full);
            third_full_samples[round] = (generated_ns, hand_ns);
            let hand_ns = sample_fourth_full(&rows, strongest_hand_fourth_full);
            let generated_ns = sample_fourth_full(&rows, generated_fourth_full);
            fourth_full_samples[round] = (generated_ns, hand_ns);
        }
    }
    for (channel, samples) in [
        ("order2", order2_samples),
        ("third", third_samples),
        ("fourth", fourth_samples),
        ("third_full", third_full_samples),
        ("fourth_full", fourth_full_samples),
    ] {
        let generated_ns = median(samples.map(|sample| sample.0));
        let hand_ns = median(samples.map(|sample| sample.1));
        let paired_ratio = median(samples.map(|sample| sample.1 / sample.0));
        eprintln!(
            "RIGID-BMS-HAND-932 channel={channel} generated={generated_ns:.3} ns/row \
             strongest_hand={hand_ns:.3} ns/row paired_hand_over_generated={paired_ratio:.6}",
        );
        assert!(
            paired_ratio > 1.0,
            "generated {channel} must beat the strongest direct analytic schedule by paired median: \
             generated={generated_ns:.3} ns/row hand={hand_ns:.3} ns/row ratio={paired_ratio:.6}",
        );
    }
}
