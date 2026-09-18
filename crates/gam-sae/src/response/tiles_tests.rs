//! #2946 pins for the chaos pair route: every row against the closed-form row within the kernel's own band plus the
//! reference's, for zero-bias and biased units, with a corrupted coefficient the pin must reject; the vectorized order
//! passes bit for bit against the portable body; and the speed contract of a row at LLM-like reader geometry.

use super::{
    CLOSED_FORM_OPERATIONS, PairChaosTable, PairRowScratch, chaos_operations, chaos_orders, chaos_orders_body,
};
use gam_linalg::roundoff::accumulation_growth;
use gam_math::gaussian_activation::{
    GaussianActivation, PairKernel, PreactivationPair, gaussian_hermite_coefficients, pair_kernel,
};
use gam_math::paired_timing::{SpeedGate, paired_interleaved};
use ndarray::Array2;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use std::f64::consts::PI;

const FIXTURE_ORDER: usize = 48;

/// The spread of the biased fixtures' `b_j`, of the size Qwen3-8B layer 18's biases take under its declared law
/// (`|b_j|` p50 0.39, p99 1.44; #2946 comment 5718398668).
const BIAS_SCALE: f64 = 0.75;

fn standard_normal(rng: &mut StdRng) -> f64 {
    // `1 − U[0, 1)` lies in `(0, 1]`, so the logarithm is finite.
    let u1: f64 = 1.0 - rng.random_range(0.0..1.0);
    let u2: f64 = rng.random_range(0.0..1.0);
    (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
}

/// A block with its covariance rows at `P = I` and at a random rank-two frame, and its metric rows.
struct Fixture {
    width: usize,
    biases: Vec<f64>,
    variances: Vec<f64>,
    /// `W Wᵀ`, row-major.
    identity_covariances: Vec<f64>,
    /// `(W Q)(W Q)ᵀ` for a random rank-two frame `Q`, row-major.
    frame_covariances: Vec<f64>,
    /// The first-order growth of either Gram: a `dim`-term inner product, and at the frame two projections of `dim`
    /// terms followed by a two-term inner product, so at most `2 dim + 2` operations on any path.
    covariance_growth: f64,
    /// `D = Uᵀ U`, row-major.
    metric_rows: Vec<f64>,
}

/// How the fixture's readers are drawn.
#[derive(Clone, Copy)]
enum ReaderLaw {
    /// Readers share one direction, so the correlations spread from near zero to near one, and the second reader
    /// duplicates the first, so one pair sits at correlation one at `P = I`.
    SharedWithDuplicate,
    /// Independent readers, whose correlations concentrate near `1/√dim`, as the Qwen3-8B readers do near
    /// `1/√4096` (#2946 comment 5716095908).
    Independent,
}

fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter().zip(right).map(|(a, b)| a * b).sum()
}

fn fixture(seed: u64, width: usize, dim: usize, outputs: usize, law: ReaderLaw, bias_scale: f64) -> Fixture {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut base = vec![0.0; dim];
    for entry in base.iter_mut() {
        *entry = standard_normal(&mut rng);
    }
    let mut readers = vec![0.0; width * dim];
    for unit in 0..width {
        let loading = match law {
            ReaderLaw::SharedWithDuplicate => standard_normal(&mut rng),
            ReaderLaw::Independent => 0.0,
        };
        for coordinate in 0..dim {
            readers[unit * dim + coordinate] = loading * base[coordinate] + 0.5 * standard_normal(&mut rng);
        }
    }
    if let ReaderLaw::SharedWithDuplicate = law {
        for coordinate in 0..dim {
            readers[dim + coordinate] = readers[coordinate];
        }
    }
    let reader = |unit: usize| &readers[unit * dim..(unit + 1) * dim];
    let mut first = vec![0.0; dim];
    let mut second = vec![0.0; dim];
    for coordinate in 0..dim {
        first[coordinate] = standard_normal(&mut rng);
        second[coordinate] = standard_normal(&mut rng);
    }
    let first_norm = dot(&first, &first).sqrt();
    first.iter_mut().for_each(|entry| *entry /= first_norm);
    let overlap = dot(&first, &second);
    for coordinate in 0..dim {
        second[coordinate] -= overlap * first[coordinate];
    }
    let second_norm = dot(&second, &second).sqrt();
    second.iter_mut().for_each(|entry| *entry /= second_norm);
    let mut writers = vec![0.0; outputs * width];
    for entry in writers.iter_mut() {
        *entry = standard_normal(&mut rng);
    }
    let mut biases = vec![0.0; width];
    for bias in biases.iter_mut() {
        *bias = bias_scale * standard_normal(&mut rng);
    }
    let mut variances = vec![0.0; width];
    let mut identity_covariances = vec![0.0; width * width];
    let mut frame_covariances = vec![0.0; width * width];
    let mut metric_rows = vec![0.0; width * width];
    for unit in 0..width {
        variances[unit] = dot(reader(unit), reader(unit));
        let unit_frame = [dot(reader(unit), &first), dot(reader(unit), &second)];
        for other in 0..width {
            let other_frame = [dot(reader(other), &first), dot(reader(other), &second)];
            identity_covariances[unit * width + other] = dot(reader(unit), reader(other));
            frame_covariances[unit * width + other] = dot(&unit_frame, &other_frame);
            let mut metric_product = 0.0;
            for output in 0..outputs {
                metric_product += writers[output * width + unit] * writers[output * width + other];
            }
            metric_rows[unit * width + other] = metric_product;
        }
    }
    Fixture {
        width,
        biases,
        variances,
        identity_covariances,
        frame_covariances,
        covariance_growth: accumulation_growth(2 * dim + 2),
        metric_rows,
    }
}

/// Columns for the loop under test: the coefficient owner's normalized coefficients `a_{j,n}` with their derived
/// bounds, and tails by banded subtraction against the diagonal kernel `K(b, b; v, v, v) = E σ(X)²` and its slope
/// energy `v ∂_r K = v E σ'(X)²`. Production tails come from the block's column owner; this producer only feeds the
/// loop under test.
struct Columns {
    coefficients: Array2<f64>,
    band: Array2<f64>,
    value_tails: Array2<f64>,
    slope_tails: Array2<f64>,
}

fn banded_root(raw: f64, absolute: f64, operations: usize) -> f64 {
    (raw.max(0.0) + accumulation_growth(operations) * absolute).sqrt()
}

fn columns(activation: GaussianActivation, biases: &[f64], variances: &[f64], order: usize) -> Columns {
    let width = variances.len();
    let mut coefficients = Array2::<f64>::zeros((width, order + 1));
    let mut band = Array2::<f64>::zeros((width, order + 1));
    let mut value_tails = Array2::<f64>::zeros((width, order + 1));
    let mut slope_tails = Array2::<f64>::zeros((width, order + 1));
    let mut unit_coefficients = vec![0.0; order + 1];
    let mut unit_bounds = vec![0.0; order + 1];
    for unit in 0..width {
        let (bias, variance) = (biases[unit], variances[unit]);
        gaussian_hermite_coefficients(activation, bias, variance.sqrt(), &mut unit_coefficients, &mut unit_bounds)
            .expect("finite fixture units admit every order");
        for degree in 0..=order {
            coefficients[[unit, degree]] = unit_coefficients[degree];
            band[[unit, degree]] = unit_bounds[degree];
        }
        // The diagonal pair's covariance is the variance itself, so the law is exact.
        let diagonal = pair_kernel(
            activation,
            PreactivationPair {
                mean_x: bias,
                mean_y: bias,
                variance_x: variance,
                variance_y: variance,
                covariance: variance,
                covariance_rounding: 0.0,
            },
        )
        .expect("a finite diagonal is admitted");
        let second_moment = diagonal.value;
        let slope_energy = variance * diagonal.covariance_derivative;
        let mean = coefficients[[unit, 0]];
        let mut captured = mean * mean;
        let mut captured_slope = 0.0;
        let mut propagated = 2.0 * mean.abs() * band[[unit, 0]];
        let mut propagated_slope = 0.0;
        for degree in 0..=order {
            if degree > 0 {
                let coefficient = coefficients[[unit, degree]];
                let error = 2.0 * coefficient.abs() * band[[unit, degree]];
                captured += coefficient * coefficient;
                captured_slope += degree as f64 * coefficient * coefficient;
                propagated += error;
                propagated_slope += degree as f64 * error;
            }
            // Two closed forms, the squares and the sums; the coefficients' own errors propagate through their squares.
            let operations = 2 * CLOSED_FORM_OPERATIONS + 2 * degree + 3;
            value_tails[[unit, degree]] =
                banded_root(second_moment - captured + propagated, second_moment + captured, operations);
            slope_tails[[unit, degree]] = banded_root(
                slope_energy - captured_slope + propagated_slope,
                slope_energy.abs() + captured_slope,
                operations,
            );
        }
    }
    Columns {
        coefficients,
        band,
        value_tails,
        slope_tails,
    }
}

fn table(activation: GaussianActivation, fixture: &Fixture, columns: &Columns) -> PairChaosTable {
    PairChaosTable::from_columns(
        activation,
        &fixture.biases,
        &fixture.variances,
        columns.coefficients.view(),
        columns.band.view(),
        columns.value_tails.view(),
        columns.slope_tails.view(),
    )
    .expect("finite, nonnegative fixture columns build a table")
}

/// What one sweep of a table's rows found against the closed-form rows.
struct Tally {
    violations: usize,
    chaos_pairs: usize,
    direct_off_diagonal: usize,
}

/// The rounding the fixture's Gram states for the pair `(unit, other)`: `γ` of its operations times the Cauchy–Schwarz
/// bound `s_j s_k` on the product sum.
fn fixture_rounding(table: &PairChaosTable, covariance_growth: f64, unit: usize, other: usize) -> f64 {
    covariance_growth * (table.variances[unit] * table.variances[other]).sqrt()
}

/// The closed form of one pair, with the rounding its covariance projection may absorb, as `pair_row` passes it.
fn closed_form_pair(
    table: &PairChaosTable,
    unit: usize,
    other: usize,
    covariance: f64,
    covariance_growth: f64,
) -> PairKernel {
    pair_kernel(
        table.activation,
        PreactivationPair {
            mean_x: table.biases[unit],
            mean_y: table.biases[other],
            variance_x: table.variances[unit],
            variance_y: table.variances[other],
            covariance,
            covariance_rounding: fixture_rounding(table, covariance_growth, unit, other),
        },
    )
    .expect("a finite pair is admitted")
}

/// The closed-form row the retained-response operator runs: `Σ_k D_jk [K_σ − m_j m_k]`, leaving `B_jk` in `weights`.
#[inline(never)]
fn closed_form_row(
    table: &PairChaosTable,
    unit: usize,
    covariance_row: &[f64],
    covariance_growth: f64,
    weights: &mut [f64],
) -> f64 {
    let mut sum = 0.0;
    for other in 0..table.width {
        let kernel = closed_form_pair(table, unit, other, covariance_row[other], covariance_growth);
        let metric_product = weights[other];
        sum += metric_product * (kernel.value - table.means[unit] * table.means[other]);
        weights[other] = metric_product * kernel.covariance_derivative;
    }
    sum
}

#[inline(never)]
fn chaos_row(
    table: &PairChaosTable,
    unit: usize,
    covariance_row: &[f64],
    covariance_growth: f64,
    weights: &mut [f64],
    scratch: &mut PairRowScratch,
) -> f64 {
    table
        .pair_row(
            unit,
            covariance_row,
            |other| fixture_rounding(table, covariance_growth, unit, other),
            weights,
            true,
            scratch,
        )
        .expect("finite fixture pairs are admitted")
        .value
}

/// Compare every row of `table` against the closed-form row. A row value may differ by at most the kernel's reported
/// band plus the closed-form reference's own: `γ` of its operations times its magnitudes per pair, and `γ_width` times
/// the fold's absolute sum. A slope entry may differ by at most its truncation bound and Horner's rounding, each at
/// most `γ` of the chaos operations times `d_j(0) d_k(0)/(s_j s_k)`, the columns' errors weighted by `√n`, and the
/// reference's rounding.
fn sweep(table: &PairChaosTable, columns: &Columns, fixture: &Fixture, covariances: &[f64]) -> Tally {
    let width = table.width;
    let growth = fixture.covariance_growth;
    let chaos_band = accumulation_growth(chaos_operations(table.order));
    let closed_band = accumulation_growth(CLOSED_FORM_OPERATIONS);
    let fold_band = accumulation_growth(width);
    let mut slope_errors = vec![0.0; width];
    for unit in 0..width {
        let mut weighted = 0.0;
        for degree in 1..=table.order {
            let error = columns.band[[unit, degree]];
            weighted += degree as f64 * error * error;
        }
        slope_errors[unit] = weighted.sqrt();
    }
    let mut scratch = table.row_scratch();
    let mut tally = Tally {
        violations: 0,
        chaos_pairs: 0,
        direct_off_diagonal: 0,
    };
    for unit in 0..width {
        let covariance_row = &covariances[unit * width..(unit + 1) * width];
        let metric_row = &fixture.metric_rows[unit * width..(unit + 1) * width];
        let mut weights = metric_row.to_vec();
        let chaos = table
            .pair_row(
                unit,
                covariance_row,
                |other| fixture_rounding(table, growth, unit, other),
                &mut weights,
                true,
                &mut scratch,
            )
            .expect("finite fixture pairs are admitted");
        tally.direct_off_diagonal += scratch.direct.len() - 1;
        tally.chaos_pairs += width - scratch.direct.len();
        let mut closed_weights = metric_row.to_vec();
        let closed_sum = closed_form_row(table, unit, covariance_row, growth, &mut closed_weights);
        let mut reference_band = 0.0;
        let mut absolute = 0.0;
        let unit_slope_scale = columns.slope_tails[[unit, 0]];
        for other in 0..width {
            let kernel = closed_form_pair(table, unit, other, covariance_row[other], growth);
            let mean_product = table.means[unit] * table.means[other];
            let metric_product = metric_row[other];
            absolute += (metric_product * (kernel.value - mean_product)).abs();
            reference_band += metric_product.abs() * closed_band * (kernel.value.abs() + mean_product.abs());
            let other_slope_scale = columns.slope_tails[[other, 0]];
            let series = 2.0 * chaos_band * unit_slope_scale * other_slope_scale
                + slope_errors[unit] * other_slope_scale
                + unit_slope_scale * slope_errors[other]
                + slope_errors[unit] * slope_errors[other];
            let slope_allowed = metric_product.abs()
                * (series * table.inverse_scales[unit] * table.inverse_scales[other]
                    + closed_band * kernel.covariance_derivative.abs());
            if (weights[other] - closed_weights[other]).abs() > slope_allowed {
                tally.violations += 1;
            }
        }
        let allowed = chaos.band + reference_band + fold_band * absolute;
        if (chaos.value - closed_sum).abs() > allowed {
            tally.violations += 1;
        }
    }
    tally
}

#[test]
fn chaos_rows_match_the_closed_form_rows_within_the_derived_band() {
    for bias_scale in [0.0, BIAS_SCALE] {
        let fixture = fixture(0x2946_7117, 24, 6, 3, ReaderLaw::SharedWithDuplicate, bias_scale);
        for activation in [GaussianActivation::Relu, GaussianActivation::ExactGelu] {
            let columns = columns(activation, &fixture.biases, &fixture.variances, FIXTURE_ORDER);
            let table = table(activation, &fixture, &columns);
            for (label, covariances) in [
                ("P = I", &fixture.identity_covariances),
                ("rank-two frame", &fixture.frame_covariances),
            ] {
                let tally = sweep(&table, &columns, &fixture, covariances);
                assert_eq!(
                    tally.violations, 0,
                    "{activation:?} at {label}, bias scale {bias_scale}: {} row or slope entries left the derived band",
                    tally.violations
                );
                assert!(
                    tally.chaos_pairs > 0,
                    "{activation:?} at {label}, bias scale {bias_scale}: no pair took the chaos route, so the pin \
                     checked nothing of it"
                );
            }
            let identity = sweep(&table, &columns, &fixture, &fixture.identity_covariances);
            assert!(
                identity.direct_off_diagonal > 0,
                "{activation:?}, bias scale {bias_scale}: the duplicated reader pair sits at correlation one at P = I \
                 and must take the closed form"
            );
            // Positive control: a chaos coefficient of order two scaled by 5/4 moves every chaos pair with a nonzero
            // correlation by `(25/16 − 1) ρ² a₂ a₂`, far outside the band, and the pin must say so.
            let mut corrupted = table.clone();
            for unit in 0..corrupted.width {
                corrupted.coefficients[corrupted.width + unit] *= 1.25;
            }
            let corrupted_tally = sweep(&corrupted, &columns, &fixture, &fixture.identity_covariances);
            assert!(
                corrupted_tally.violations > 0,
                "{activation:?}, bias scale {bias_scale}: a corrupted order-two coefficient passed the pin, so the \
                 pin cannot fail"
            );
        }
    }
}

#[test]
fn vectorized_order_passes_write_the_portable_words() {
    let mut rng = StdRng::seed_from_u64(0x2946_0A5E);
    // An odd width leaves a remainder after every SIMD lane width.
    let width = 37;
    let order = 9;
    let mut unit_coefficients = vec![0.0; order];
    for entry in unit_coefficients.iter_mut() {
        *entry = standard_normal(&mut rng);
    }
    let mut columns = vec![0.0; order * width];
    for entry in columns.iter_mut() {
        *entry = standard_normal(&mut rng);
    }
    let mut correlations = vec![0.0; width];
    for entry in correlations.iter_mut() {
        *entry = rng.random_range(-1.0..1.0);
    }
    for gradient in [false, true] {
        let mut dispatched = [vec![0.0; width], vec![0.0; width]];
        let mut portable = [vec![0.0; width], vec![0.0; width]];
        let [dispatched_values, dispatched_slopes] = &mut dispatched;
        chaos_orders(
            &unit_coefficients,
            &columns,
            &correlations,
            dispatched_values,
            dispatched_slopes,
            gradient,
        );
        let [portable_values, portable_slopes] = &mut portable;
        chaos_orders_body(
            &unit_coefficients,
            &columns,
            &correlations,
            portable_values,
            portable_slopes,
            gradient,
        );
        for k in 0..width {
            assert_eq!(
                dispatched[0][k].to_bits(),
                portable[0][k].to_bits(),
                "value {k} differs between the dispatched and portable passes (gradient {gradient})"
            );
            assert_eq!(
                dispatched[1][k].to_bits(),
                portable[1][k].to_bits(),
                "slope {k} differs between the dispatched and portable passes (gradient {gradient})"
            );
        }
        assert!(
            dispatched[0].iter().any(|value| *value != 0.0),
            "the fixture wrote no values, so the comparison checked nothing"
        );
    }
}

/// The chaos row must beat the closed-form row it replaces, on a row at LLM-like reader geometry, for zero-bias units
/// and for biased units whose closed form needs the bivariate normal CDF. Parity is pinned in every build first; the
/// timing runs only where the codegen is the shipped one.
#[test]
fn chaos_row_is_faster_than_the_closed_form_row() {
    let mut blocks = Vec::new();
    for bias_scale in [0.0, BIAS_SCALE] {
        let fixture = fixture(0x2946_5EED, 1024, 256, 16, ReaderLaw::Independent, bias_scale);
        let mut tables = Vec::new();
        for activation in [GaussianActivation::Relu, GaussianActivation::ExactGelu] {
            let columns = columns(activation, &fixture.biases, &fixture.variances, FIXTURE_ORDER);
            let table = table(activation, &fixture, &columns);
            let tally = sweep(&table, &columns, &fixture, &fixture.identity_covariances);
            assert_eq!(
                tally.violations, 0,
                "{activation:?}, bias scale {bias_scale}: the timed rows left the derived band, so their timing \
                 compares different results"
            );
            assert!(
                tally.chaos_pairs > tally.direct_off_diagonal,
                "{activation:?}, bias scale {bias_scale}: at independent-reader geometry most pairs must take the \
                 chaos route"
            );
            tables.push((activation, table));
        }
        blocks.push((bias_scale, fixture, tables));
    }
    if cfg!(debug_assertions) {
        return; // dev lane: the codegen is not the shipped one
    }
    let mut gate = SpeedGate::open("FR-PERF-CHAOS-ROW-2946");
    for (bias_scale, fixture, tables) in &blocks {
        let width = fixture.width;
        let unit = width / 2;
        let growth = fixture.covariance_growth;
        let covariance_row = &fixture.identity_covariances[unit * width..(unit + 1) * width];
        let metric_row = &fixture.metric_rows[unit * width..(unit + 1) * width];
        for (activation, table) in tables {
            let mut chaos_weights = metric_row.to_vec();
            let mut scratch = table.row_scratch();
            let mut closed_weights = metric_row.to_vec();
            let timing = paired_interleaved(
                15,
                200,
                0x2946_71AE,
                |nudge| {
                    chaos_weights.copy_from_slice(metric_row);
                    chaos_weights[0] += nudge;
                    chaos_row(table, unit, covariance_row, growth, &mut chaos_weights, &mut scratch)
                },
                |nudge| {
                    closed_weights.copy_from_slice(metric_row);
                    closed_weights[0] += nudge;
                    closed_form_row(table, unit, covariance_row, growth, &mut closed_weights)
                },
            );
            let cell = format!("{activation:?} bias scale {bias_scale} row 1024");
            gate.faster(&cell, &timing, "chaos", "closed_form");
        }
    }
    gate.finish();
}
