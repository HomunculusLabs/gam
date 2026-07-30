use gam_row_macros::row_atom;
use std::time::Instant;

row_atom! {
    fn generated_gaussian [order2_at_zero, third_at_zero, fourth_at_zero](
        delta_mu,
        delta_eta;
        obs_weight,
        standardized_residual,
        inv_sigma,
        kappa
    ) {
        obs_weight * ln((1.0 - kappa) + kappa * exp(delta_eta))
            + 0.5
                * obs_weight
                * (standardized_residual - delta_mu * inv_sigma)
                * (standardized_residual - delta_mu * inv_sigma)
                / ((1.0 - kappa) + kappa * exp(delta_eta))
                / ((1.0 - kappa) + kappa * exp(delta_eta))
    }
}

type Channels = (f64, [f64; 2], [[f64; 2]; 2]);

#[derive(Clone, Copy)]
struct Row {
    weight: f64,
    residual: f64,
    inv_sigma: f64,
    kappa: f64,
    direction_u: [f64; 2],
    direction_v: [f64; 2],
}

#[inline(always)]
fn generated_order2(row: Row) -> Channels {
    let atom =
        generated_gaussian_order2_at_zero(row.weight, row.residual, row.inv_sigma, row.kappa);
    let gradient = atom.gradient();
    (
        atom.value(),
        gradient,
        [
            [atom.hessian_at(0, 0), atom.hessian_at(0, 1)],
            [atom.hessian_at(1, 0), atom.hessian_at(1, 1)],
        ],
    )
}

#[inline(always)]
fn generated_third(row: Row) -> [[f64; 2]; 2] {
    generated_gaussian_third_contracted_at_zero(
        row.weight,
        row.residual,
        row.inv_sigma,
        row.kappa,
        &row.direction_u,
    )
}

#[inline(always)]
fn generated_fourth(row: Row) -> [[f64; 2]; 2] {
    generated_gaussian_fourth_contracted_at_zero(
        row.weight,
        row.residual,
        row.inv_sigma,
        row.kappa,
        &row.direction_u,
        &row.direction_v,
    )
}

#[inline(always)]
fn hand_order2(row: Row) -> Channels {
    let w = row.weight;
    let r = row.residual;
    let q = row.inv_sigma;
    let k = row.kappa;
    let k2 = k * k;
    let wr = w * r;
    let wq = w * q;
    let h00 = wq * q;
    let h01 = 2.0 * wr * q * k;
    let h11 = w * (k - k2 + r * r * (-k + 3.0 * k2));
    (
        0.5 * wr * r,
        [-wq * r, w * k * (1.0 - r * r)],
        [[h00, h01], [h01, h11]],
    )
}

#[inline(always)]
fn hand_third(row: Row) -> [[f64; 2]; 2] {
    let w = row.weight;
    let r = row.residual;
    let q = row.inv_sigma;
    let k = row.kappa;
    let k2 = k * k;
    let k3 = k2 * k;
    let c2 = -2.0 * k + 6.0 * k2;
    let c3 = -2.0 * k + 18.0 * k2 - 24.0 * k3;
    let l3 = k - 3.0 * k2 + 2.0 * k3;
    let xxe = -2.0 * w * q * q * k;
    let xee = -w * r * q * c2;
    let eee = w * (l3 + 0.5 * r * r * c3);
    let [u0, u1] = row.direction_u;
    [
        [xxe * u1, xxe * u0 + xee * u1],
        [xxe * u0 + xee * u1, xee * u0 + eee * u1],
    ]
}

#[inline(always)]
fn hand_fourth(row: Row) -> [[f64; 2]; 2] {
    let w = row.weight;
    let r = row.residual;
    let q = row.inv_sigma;
    let k = row.kappa;
    let k2 = k * k;
    let k3 = k2 * k;
    let k4 = k2 * k2;
    let c2 = -2.0 * k + 6.0 * k2;
    let c3 = -2.0 * k + 18.0 * k2 - 24.0 * k3;
    let c4 = -2.0 * k + 42.0 * k2 - 144.0 * k3 + 120.0 * k4;
    let l4 = k - 7.0 * k2 + 12.0 * k3 - 6.0 * k4;
    let xxee = w * q * q * c2;
    let xeee = -w * r * q * c3;
    let eeee = w * (l4 + 0.5 * r * r * c4);
    let [u0, u1] = row.direction_u;
    let [v0, v1] = row.direction_v;
    let uv = u0 * v1 + u1 * v0;
    [
        [xxee * u1 * v1, xxee * uv + xeee * u1 * v1],
        [
            xxee * uv + xeee * u1 * v1,
            xxee * u0 * v0 + xeee * uv + eeee * u1 * v1,
        ],
    ]
}

fn rows() -> Vec<Row> {
    (0..512)
        .map(|index| {
            let x = index as f64;
            Row {
                weight: 0.55 + 0.45 * (x * 0.19 + 1.0).sin().abs(),
                residual: 1.4 * (x * 0.17 + 0.3).sin() - 0.5 * (x * 0.09).cos(),
                inv_sigma: (0.8 * (x * 0.11 + 0.2).sin() - 0.25 * (x * 0.07).cos())
                    .exp()
                    .recip(),
                kappa: 0.05 + 0.9 * (x * 0.13 + 0.7).cos().abs(),
                direction_u: [
                    0.7 * (x * 0.23 + 0.4).cos() - 0.2 * (x * 0.03).sin(),
                    -0.6 * (x * 0.29 + 0.1).sin() + 0.25 * (x * 0.15).cos(),
                ],
                direction_v: [
                    -0.5 * (x * 0.21 + 0.9).sin() + 0.3 * (x * 0.06).cos(),
                    0.8 * (x * 0.27 + 0.5).cos() - 0.15 * (x * 0.04).sin(),
                ],
            }
        })
        .collect()
}

fn close(got: f64, want: f64) {
    let tolerance = 2e-12 * got.abs().max(want.abs()).max(1.0);
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

fn sample_channels(rows: &[Row], evaluate: impl Fn(Row) -> Channels) -> f64 {
    let started = Instant::now();
    let mut checksum = 0.0;
    for iteration in 0..4096 {
        for row in rows {
            let mut perturbed = *row;
            perturbed.residual += checksum * 1e-24;
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
            perturbed.residual += checksum * 1e-24;
            let matrix = std::hint::black_box(evaluate(perturbed));
            checksum += matrix.iter().flat_map(|line| line.iter()).sum::<f64>();
        }
        checksum *= 1e-3 + iteration as f64 * 1e-12;
    }
    assert!(checksum.is_finite());
    started.elapsed().as_secs_f64() * 1e9 / (rows.len() * 4096) as f64
}

#[test]
fn generated_gaussian_matches_and_beats_strongest_hand_932() {
    let rows = rows();
    for row in &rows {
        assert_channels(generated_order2(*row), hand_order2(*row));
        assert_matrix(generated_third(*row), hand_third(*row));
        assert_matrix(generated_fourth(*row), hand_fourth(*row));
    }

    let mut generated_order2_ns = f64::INFINITY;
    let mut hand_order2_ns = f64::INFINITY;
    let mut generated_third_ns = f64::INFINITY;
    let mut hand_third_ns = f64::INFINITY;
    let mut generated_fourth_ns = f64::INFINITY;
    let mut hand_fourth_ns = f64::INFINITY;
    for round in 0..7 {
        if round % 2 == 0 {
            generated_order2_ns = generated_order2_ns.min(sample_channels(&rows, generated_order2));
            hand_order2_ns = hand_order2_ns.min(sample_channels(&rows, hand_order2));
            generated_third_ns = generated_third_ns.min(sample_matrix(&rows, generated_third));
            hand_third_ns = hand_third_ns.min(sample_matrix(&rows, hand_third));
            generated_fourth_ns = generated_fourth_ns.min(sample_matrix(&rows, generated_fourth));
            hand_fourth_ns = hand_fourth_ns.min(sample_matrix(&rows, hand_fourth));
        } else {
            hand_order2_ns = hand_order2_ns.min(sample_channels(&rows, hand_order2));
            generated_order2_ns = generated_order2_ns.min(sample_channels(&rows, generated_order2));
            hand_third_ns = hand_third_ns.min(sample_matrix(&rows, hand_third));
            generated_third_ns = generated_third_ns.min(sample_matrix(&rows, generated_third));
            hand_fourth_ns = hand_fourth_ns.min(sample_matrix(&rows, hand_fourth));
            generated_fourth_ns = generated_fourth_ns.min(sample_matrix(&rows, generated_fourth));
        }
    }

    let results = [
        ("order2", generated_order2_ns, hand_order2_ns),
        ("third", generated_third_ns, hand_third_ns),
        ("fourth", generated_fourth_ns, hand_fourth_ns),
    ];
    for (channel, generated_ns, hand_ns) in results {
        eprintln!(
            "GAUSSIAN-JOINT-HAND-932 channel={channel} generated={generated_ns:.3} ns/row \
             strongest_hand={hand_ns:.3} ns/row hand_over_generated={:.6}",
            hand_ns / generated_ns,
        );
    }
    for (channel, generated_ns, hand_ns) in results {
        assert!(
            generated_ns < hand_ns,
            "generated {channel} {generated_ns:.3} ns/row must beat strongest hand \
             {hand_ns:.3} ns/row"
        );
    }
}
