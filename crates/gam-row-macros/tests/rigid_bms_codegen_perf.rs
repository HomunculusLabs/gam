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
    emit [generic, order2];
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
        let nll = compose(signed_probit, signed_margin, weight);
        return nll;
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

fn sample(rows: &[Row], evaluate: fn(Row) -> Channels) -> f64 {
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

#[test]
fn generated_rigid_bms_matches_strongest_hand_932() {
    let rows = rows();
    for row in &rows {
        assert_channels(generated(*row), strongest_hand(*row));
    }

    let mut generated_ns = f64::INFINITY;
    let mut hand_ns = f64::INFINITY;
    for round in 0..7 {
        if round % 2 == 0 {
            generated_ns = generated_ns.min(sample(&rows, generated));
            hand_ns = hand_ns.min(sample(&rows, strongest_hand));
        } else {
            hand_ns = hand_ns.min(sample(&rows, strongest_hand));
            generated_ns = generated_ns.min(sample(&rows, generated));
        }
    }
    eprintln!(
        "RIGID-BMS-HAND-932 generated={generated_ns:.3} ns/row \
         strongest_hand={hand_ns:.3} ns/row hand_over_generated={:.6}",
        hand_ns / generated_ns,
    );
    assert!(
        generated_ns < hand_ns,
        "generated rigid BMS must beat the strongest hand kernel: \
         generated={generated_ns:.3} ns/row hand={hand_ns:.3} ns/row",
    );
}
