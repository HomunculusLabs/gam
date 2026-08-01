use gam_row_macros::row_atom;
use gam_math::paired_timing::paired_interleaved;

row_atom! {
    fn generated_cause_specific [order2, third, fourth](
        eta_exit,
        eta_entry,
        derivative;
        weight: scale,
        entry_active: bool,
        event: bool
    ) {
        weight
            * (exp(eta_exit)
                - entry_active * exp(eta_entry)
                - event * (eta_exit + ln(derivative)))
    }
}

type Channels = (f64, [f64; 3], [[f64; 3]; 3]);

#[derive(Clone, Copy)]
pub struct Row {
    eta_exit: f64,
    eta_entry: f64,
    derivative: f64,
    weight: f64,
    entry_active: bool,
    event: bool,
    direction_u: [f64; 3],
    direction_v: [f64; 3],
}

#[inline(never)]
fn generated_order2(row: Row) -> Channels {
    let atom = generated_cause_specific_order2(
        row.eta_exit,
        row.eta_entry,
        row.derivative,
        row.weight,
        row.entry_active,
        row.event,
    );
    let gradient = atom.gradient();
    (
        atom.value(),
        gradient,
        std::array::from_fn(|left| std::array::from_fn(|right| atom.hessian_at(left, right))),
    )
}

#[inline(never)]
fn generated_third(row: Row) -> [[f64; 3]; 3] {
    generated_cause_specific_third_contracted(
        row.eta_exit,
        row.eta_entry,
        row.derivative,
        row.weight,
        row.entry_active,
        row.event,
        &row.direction_u,
    )
}

#[inline(never)]
fn generated_fourth(row: Row) -> [[f64; 3]; 3] {
    generated_cause_specific_fourth_contracted(
        row.eta_exit,
        row.eta_entry,
        row.derivative,
        row.weight,
        row.entry_active,
        row.event,
        &row.direction_u,
        &row.direction_v,
    )
}

#[inline(always)]
fn hand_coefficients(row: Row) -> (f64, f64, f64, f64) {
    let exit = row.weight * row.eta_exit.exp();
    let entry = if row.entry_active {
        -row.weight * row.eta_entry.exp()
    } else {
        0.0
    };
    let (weighted_event, inverse_derivative) = if row.event {
        (row.weight, row.derivative.recip())
    } else {
        (0.0, 0.0)
    };
    (exit, entry, weighted_event, inverse_derivative)
}

#[inline(never)]
fn hand_order2(row: Row) -> Channels {
    let (exit, entry, weighted_event, inverse_derivative) = hand_coefficients(row);
    let derivative_gradient = -weighted_event * inverse_derivative;
    let derivative_hessian = weighted_event * inverse_derivative * inverse_derivative;
    (
        exit + entry
            - if row.event {
                row.weight * (row.eta_exit + row.derivative.ln())
            } else {
                0.0
            },
        [exit - weighted_event, entry, derivative_gradient],
        [
            [exit, 0.0, 0.0],
            [0.0, entry, 0.0],
            [0.0, 0.0, derivative_hessian],
        ],
    )
}

#[inline(never)]
fn hand_third(row: Row) -> [[f64; 3]; 3] {
    let (exit, entry, weighted_event, inverse_derivative) = hand_coefficients(row);
    let inverse_squared = inverse_derivative * inverse_derivative;
    let derivative_third = -2.0 * weighted_event * inverse_squared * inverse_derivative;
    [
        [exit * row.direction_u[0], 0.0, 0.0],
        [0.0, entry * row.direction_u[1], 0.0],
        [0.0, 0.0, derivative_third * row.direction_u[2]],
    ]
}

#[inline(never)]
fn hand_fourth(row: Row) -> [[f64; 3]; 3] {
    let (exit, entry, weighted_event, inverse_derivative) = hand_coefficients(row);
    let inverse_squared = inverse_derivative * inverse_derivative;
    let derivative_fourth = 6.0 * weighted_event * inverse_squared * inverse_squared;
    [
        [exit * row.direction_u[0] * row.direction_v[0], 0.0, 0.0],
        [0.0, entry * row.direction_u[1] * row.direction_v[1], 0.0],
        [
            0.0,
            0.0,
            derivative_fourth * row.direction_u[2] * row.direction_v[2],
        ],
    ]
}

fn rows() -> Vec<Row> {
    (0..512)
        .map(|index| {
            let x = index as f64;
            Row {
                eta_exit: 1.1 * (x * 0.17 + 0.3).sin() - 0.4 * (x * 0.09).cos(),
                eta_entry: 0.7 * (x * 0.11 + 0.2).sin() - 0.25 * (x * 0.07).cos(),
                derivative: 0.2 + (0.8 * (x * 0.13 + 0.7).cos()).exp(),
                weight: 0.55 + 0.45 * (x * 0.19 + 1.0).sin().abs(),
                entry_active: index % 2 != 0,
                event: (index / 2) % 2 != 0,
                direction_u: [
                    0.7 * (x * 0.23 + 0.4).cos(),
                    -0.6 * (x * 0.29 + 0.1).sin(),
                    0.5 * (x * 0.31 + 0.8).cos(),
                ],
                direction_v: [
                    -0.5 * (x * 0.21 + 0.9).sin(),
                    0.8 * (x * 0.27 + 0.5).cos(),
                    -0.4 * (x * 0.25 + 0.6).sin(),
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
    for axis in 0..3 {
        close(got.1[axis], want.1[axis]);
        for other in 0..3 {
            close(got.2[axis][other], want.2[axis][other]);
        }
    }
}

fn assert_matrix(got: [[f64; 3]; 3], want: [[f64; 3]; 3]) {
    for axis in 0..3 {
        for other in 0..3 {
            close(got[axis][other], want[axis][other]);
        }
    }
}

/// One pass over every row, folded to a scalar the paired harness can chain.
///
/// The old `sample_channels`/`sample_matrix` pair timed a whole 4096-iteration
/// block per arm and the caller kept a running MINIMUM per arm across seven
/// rounds. Alternating the order between rounds -- which this test did, and did
/// correctly -- creates a pairing that taking two independent minima then throws
/// away. A minimum is the most favourable draw for each arm SEPARATELY, so the
/// arm whose timing is more dispersed is flattered, and the alternation cannot
/// undo that because it never reaches the statistic (#932).
fn channels_pass(rows: &[Row], nudge: f64, evaluate: impl Fn(Row) -> Channels) -> f64 {
    let mut fold = 0.0;
    for row in rows {
        let mut perturbed = *row;
        perturbed.eta_exit += nudge;
        let channels = evaluate(perturbed);
        fold += channels.0
            + channels.1.iter().sum::<f64>()
            + channels.2.iter().flat_map(|line| line.iter()).sum::<f64>();
    }
    fold
}

fn matrix_pass(rows: &[Row], nudge: f64, evaluate: impl Fn(Row) -> [[f64; 3]; 3]) -> f64 {
    let mut fold = 0.0;
    for row in rows {
        let mut perturbed = *row;
        perturbed.eta_exit += nudge;
        fold += evaluate(perturbed)
            .iter()
            .flat_map(|line| line.iter())
            .sum::<f64>();
    }
    fold
}

#[test]
fn generated_cause_specific_matches_strongest_hand_932() {
    let rows = rows();
    for row in &rows {
        assert_channels(generated_order2(*row), hand_order2(*row));
        assert_matrix(generated_third(*row), hand_third(*row));
        assert_matrix(generated_fourth(*row), hand_fourth(*row));
    }

    // One paired, interleaved, order-RANDOMISED measurement per channel. The
    // arms are timed adjacent within each repetition and the per-repetition
    // ratios are kept, so the pairing the old alternation created survives all
    // the way to the statistic.
    let reps = 15usize;
    let passes = 256usize;
    let mut losses: Vec<String> = Vec::new();
    for (channel, timing) in [
        (
            "order2",
            paired_interleaved(
                reps,
                passes,
                0x932_0_C502,
                |nudge| channels_pass(&rows, nudge, generated_order2),
                |nudge| channels_pass(&rows, nudge, hand_order2),
            ),
        ),
        (
            "third",
            paired_interleaved(
                reps,
                passes,
                0x932_0_C503,
                |nudge| matrix_pass(&rows, nudge, generated_third),
                |nudge| matrix_pass(&rows, nudge, hand_third),
            ),
        ),
        (
            "fourth",
            paired_interleaved(
                reps,
                passes,
                0x932_0_C504,
                |nudge| matrix_pass(&rows, nudge, generated_fourth),
                |nudge| matrix_pass(&rows, nudge, hand_fourth),
            ),
        ),
    ] {
        // `ns/iter` here is nanoseconds per PASS over `rows.len()` rows, not the
        // historical `ns/row`; the ratio the verdict rests on is unit-free
        // either way. `median_ratio` is hand / generated, so above 1 means the
        // generated kernel is faster -- the same orientation as the
        // `hand_over_generated` token this test has always printed.
        eprintln!(
            "CAUSE-SPECIFIC-HAND-932 channel={channel} rows={} {}",
            rows.len(),
            timing.summary("generated", "strongest_hand"),
        );
        // CONTRACT UNCHANGED: still a bare "generated must be faster", with no
        // margin. That bar is the open question on this test -- two semantically
        // identical kernels compared by `<` decide a coin flip when the true
        // difference is near zero -- and `ratio_resolution` in the line above is
        // the quantity a margin would have to be derived from. Deriving it in
        // the same commit that first measures it would fit the tolerance to the
        // numbers meant to validate it, so that is a separate change (#2469).
        if timing.median_ratio() <= 1.0 {
            losses.push(format!(
                "{channel}: {}",
                timing.summary("generated", "strongest_hand")
            ));
        }
    }

    // Collect every channel, THEN assert. Asserting inside the loop meant the
    // first failing channel aborted the test and the remaining two were never
    // measured -- so a report of "order2 is slower" could not say whether third
    // and fourth were fine, and a fix for one channel would surface the next as
    // a fresh surprise. A gate should say everything it objected to in one run.
    assert!(
        losses.is_empty(),
        "generated kernels must beat the strongest semantically identical hand kernel:\n{}",
        losses.join("\n"),
    );
}
