#![cfg(test)]
//! #2946 A2 and A3 pins for the retained-response operator.
//!
//! The exact identities (A2 and the planted subspace) run on dyadic data: small-integer readers, frames and rotations
//! built from `H = I − ½·11ᵀ`, whose entries are all `±½`. Every covariance `w_jᵀ P w_k` the operator forms is then
//! computed exactly, both routes of an identity feed the kernel bit-identical arguments, and the identity holds
//! bit for bit with no tolerance. The Monte Carlo pins run the executed block and accept within a standard-error
//! multiple derived from a declared false-alarm rate, each with a positive control that the test shows it rejects.
//! Every pin prints its numbers whether it passes or fails, so a green run is a receipt.

use super::{KnownBlock, ResponseError};
use gam_math::gaussian_activation::{
    GaussianActivation, PreactivationPair, gaussian_smoothing_derivatives, pair_kernel,
};
use gam_math::probability::standard_normal_quantile;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis, array, s};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use std::f64::consts::PI;

/// The declared probability that a correct operator fails one Monte Carlo test for a fresh seed. It is split evenly
/// over the test's two-sided agreement arms (Bonferroni), which fixes each arm's standard-error multiple.
const FAMILY_FALSE_ALARM: f64 = 1.0e-6;

/// The acceptance multiple of one two-sided arm among `arms`.
fn standard_error_multiple(arms: usize) -> f64 {
    standard_normal_quantile(1.0 - FAMILY_FALSE_ALARM / (2.0 * arms as f64))
        .expect("the false-alarm split lies inside (0, 1)")
}

/// `H = I − ½·11ᵀ` in four dimensions: symmetric, orthogonal, every entry `±½`.
fn half_reflector() -> Array2<f64> {
    Array2::from_shape_fn((4, 4), |(row, column)| if row == column { 0.5 } else { -0.5 })
}

fn identity(dim: usize) -> Array2<f64> {
    Array2::from_shape_fn((dim, dim), |(row, column)| if row == column { 1.0 } else { 0.0 })
}

/// The first two columns of `H` span `{(1, −1, 0, 0), (0, 0, 1, 1)}`, its last two `{(1, 1, 0, 0), (0, 0, 1, −1)}`.
fn retained_frame() -> Array2<f64> {
    half_reflector().slice(s![.., ..2]).to_owned()
}

/// Readers whose discarded parts all overlap: `(2,0,0,0) = (1,−1,0,0) + (1,1,0,0)`,
/// `(1,1,1,1) = (0,0,1,1) + (1,1,0,0)`, `(2,0,1,1) = (1,−1,0,0) + (0,0,1,1) + (1,1,0,0)` and
/// `(1,1,1,−1) = (0,0,1,−1) + (1,1,0,0)`. Every pair has positive covariance both inside and outside the frame, so the
/// cross terms carry a large share of `V` and `E`.
fn overlapping_readers() -> Array2<f64> {
    array![
        [2.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0],
        [2.0, 0.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, -1.0],
    ]
}

/// Readers inside the span of [`retained_frame`].
fn planted_readers() -> Array2<f64> {
    array![
        [1.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 1.0],
        [1.0, -1.0, 1.0, 1.0],
        [2.0, -2.0, -1.0, -1.0],
    ]
}

/// Writer columns `(1,0,1)`, `(1,1,0)`, `(0,1,1)`, `(1,1,1)`: nonnegative, so every `D_jk = u_jᵀ M u_k` is positive.
fn writers() -> Array2<f64> {
    array![
        [1.0, 1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0, 1.0],
        [1.0, 0.0, 1.0, 1.0],
    ]
}

/// A dyadic, symmetric, positive definite output metric with a coupled pair of outputs.
fn metric() -> Array2<f64> {
    array![[2.0, 0.5, 0.0], [0.5, 1.0, 0.0], [0.0, 0.0, 1.0]]
}

fn relu_block_with_output_bias(readers: Array2<f64>, output_bias: Array1<f64>) -> KnownBlock {
    let width = readers.nrows();
    KnownBlock::new(
        readers,
        Array1::zeros(width),
        writers(),
        output_bias,
        metric().view(),
        GaussianActivation::Relu,
    )
    .expect("a finite block with a symmetric positive definite metric")
}

fn relu_block(readers: Array2<f64>) -> KnownBlock {
    relu_block_with_output_bias(readers, Array1::zeros(3))
}

fn standard_normal(rng: &mut StdRng) -> f64 {
    // `1 − U[0, 1)` lies in `(0, 1]`, so the logarithm is finite.
    let u1: f64 = 1.0 - rng.random_range(0.0..1.0);
    let u2: f64 = rng.random_range(0.0..1.0);
    (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
}

fn standard_normal_vector(rng: &mut StdRng, dim: usize) -> Array1<f64> {
    let mut draw = Array1::<f64>::zeros(dim);
    for entry in draw.iter_mut() {
        *entry = standard_normal(rng);
    }
    draw
}

/// The executed zero-bias ReLU block `U relu(W z)`.
fn execute_relu_block(readers: &Array2<f64>, z: ArrayView1<'_, f64>) -> Array1<f64> {
    writers().dot(&readers.dot(&z).mapv(|activation| activation.max(0.0)))
}

/// `Pz = Q Qᵀ z`.
fn project(frame: ArrayView2<'_, f64>, z: ArrayView1<'_, f64>) -> Array1<f64> {
    frame.dot(&frame.t().dot(&z))
}

/// The mean of i.i.d. terms and its standard error.
fn mean_and_standard_error(terms: &[f64]) -> (f64, f64) {
    let count = terms.len() as f64;
    let mean = terms.iter().sum::<f64>() / count;
    let variance = terms.iter().map(|term| (term - mean).powi(2)).sum::<f64>() / (count - 1.0);
    (mean, (variance / count).sqrt())
}

/// `Σ_j D_jj [K_σ(0, 0; v_j, v_j, ‖Qᵀ w_j‖²) − m_j²]`: `V(P)` with every cross term `j ≠ k` dropped, the positive
/// control the Monte Carlo pins must reject.
fn diagonal_only_relu_variance(readers: &Array2<f64>, frame: ArrayView2<'_, f64>) -> f64 {
    let metric_writers = metric().dot(&writers());
    let coordinates = readers.dot(&frame);
    let mut total = 0.0;
    for unit in 0..readers.nrows() {
        let variance = readers.row(unit).dot(&readers.row(unit));
        let covariance = coordinates.row(unit).dot(&coordinates.row(unit));
        // Integer readers and the dyadic frames of these tests form every variance and covariance exactly, so the
        // law is exact and states no rounding.
        let second_moment = pair_kernel(
            GaussianActivation::Relu,
            PreactivationPair {
                mean_x: 0.0,
                mean_y: 0.0,
                variance_x: variance,
                variance_y: variance,
                covariance,
                covariance_rounding: 0.0,
            },
        )
        .expect("ReLU pair kernel at zero mean")
        .value;
        let mut mean = [0.0];
        gaussian_smoothing_derivatives(GaussianActivation::Relu, 0.0, variance, &mut mean)
            .expect("ReLU smoothing at a finite variance");
        let metric_diagonal = writers().column(unit).dot(&metric_writers.column(unit));
        total += metric_diagonal * (second_moment - mean[0] * mean[0]);
    }
    total
}

#[test]
fn discarded_error_vanishes_on_the_full_frame_and_shrinks_strictly_as_the_frame_grows() {
    let block = relu_block(overlapping_readers());
    let reflector = half_reflector();
    let errors: Vec<f64> = (1..=4)
        .map(|rank| {
            block
                .discarded_error(reflector.slice(s![.., ..rank]))
                .expect("an orthonormal frame inside the input")
        })
        .collect();
    eprintln!(
        "#2946 A2 V(I) = {}; E at frame rank 1..4 = {}, {}, {}, {}",
        block.total_variance(),
        errors[0],
        errors[1],
        errors[2],
        errors[3],
    );
    // `W H` has half-integer entries and `(W H)(W H)ᵀ = W Wᵀ` holds exactly, so the full frame feeds the kernel the
    // same covariances as `V(I)` and the discarded error is exactly zero.
    assert_eq!(errors[3], 0.0, "E(I) must be exactly 0, got {}", errors[3]);
    assert_eq!(
        block
            .explained_variance(reflector.view())
            .expect("the full frame"),
        block.total_variance(),
    );
    assert!(block.total_variance() > 0.0);
    // Conditioning on a larger subspace can only reduce the discarded error, and these readers overlap every
    // column of `H`, so each added column removes a positive share.
    for rank in 0..3 {
        assert!(
            errors[rank] > errors[rank + 1],
            "E must shrink as the frame grows: E(rank {}) = {} vs E(rank {}) = {}",
            rank + 1,
            errors[rank],
            rank + 2,
            errors[rank + 1],
        );
    }
}

#[test]
fn explained_variance_and_retained_response_are_invariant_under_an_orthogonal_input_rotation() {
    // W → W H and Q → Hᵀ Q = H Q, with H orthogonal and not a signed permutation. Every reader coordinate
    // `(W H)(H Q) = W Q`, every reader variance and every discarded residual is formed exactly, so the rotated block
    // must reproduce the unrotated one bit for bit.
    let reflector = half_reflector();
    let readers = overlapping_readers();
    let rotated_readers = readers.dot(&reflector);
    assert_ne!(rotated_readers, readers, "the rotation must move the readers");
    let block = relu_block(readers);
    let rotated_block = relu_block(rotated_readers);
    let frame = identity(4).slice(s![.., ..2]).to_owned();
    let rotated_frame = reflector.dot(&frame);

    assert_eq!(rotated_block.total_variance(), block.total_variance());
    assert_eq!(
        rotated_block
            .explained_variance(rotated_frame.view())
            .expect("rotated frame"),
        block.explained_variance(frame.view()).expect("frame"),
    );
    assert_eq!(
        rotated_block
            .discarded_error(rotated_frame.view())
            .expect("rotated frame"),
        block.discarded_error(frame.view()).expect("frame"),
    );

    let points = array![[0.0, 0.0, 0.0, 0.0], [1.0, -2.0, 0.5, 3.0], [-2.0, 0.5, 1.0, -1.5]];
    let rotated_points = points.dot(&reflector);
    let response = block
        .retained_response(frame.view(), points.view())
        .expect("finite points");
    let rotated_response = rotated_block
        .retained_response(rotated_frame.view(), rotated_points.view())
        .expect("finite rotated points");
    assert_eq!(rotated_response, response);
}

#[test]
fn a_coordinate_frame_reads_the_selected_reader_columns() {
    // Integer readers and a 0/1 frame form every covariance exactly, so the coordinate route and the frame route
    // feed the kernel the same bits.
    let block = relu_block(overlapping_readers());
    let unit_frame = identity(4);
    for retained in [vec![], vec![0], vec![1, 3], vec![0, 2, 3], vec![0, 1, 2, 3]] {
        let frame = unit_frame.select(Axis(1), &retained);
        assert_eq!(
            block
                .explained_variance_of_coordinates(&retained)
                .expect("strictly increasing coordinates"),
            block.explained_variance(frame.view()).expect("a coordinate frame"),
            "coordinates {retained:?}",
        );
    }
    assert_eq!(
        block
            .explained_variance_of_coordinates(&[0, 1, 2, 3])
            .expect("every coordinate"),
        block.total_variance(),
    );
    let refusal = block.explained_variance_of_coordinates(&[2, 1]);
    assert!(
        matches!(refusal, Err(ResponseError::InvalidRetainedCoordinates { position: 1 })),
        "decreasing coordinates must be refused, got {refusal:?}",
    );
    let refusal = block.explained_variance_of_coordinates(&[0, 4]);
    assert!(
        matches!(refusal, Err(ResponseError::InvalidRetainedCoordinates { position: 1 })),
        "an out-of-range coordinate must be refused, got {refusal:?}",
    );
}

#[test]
fn a_planted_exact_subspace_discards_nothing_and_its_complement_discards_everything() {
    let block = relu_block(planted_readers());
    let reflector = half_reflector();
    let retained = block
        .discarded_error(reflector.slice(s![.., ..2]))
        .expect("the planted frame");
    // The planted readers are exactly orthogonal to the complement, so the complement frame feeds the kernel the
    // zero covariances of the empty frame: both discard the same, positive, error.
    let complement = block
        .discarded_error(reflector.slice(s![.., 2..]))
        .expect("the complement frame");
    let empty = block
        .discarded_error(reflector.slice(s![.., ..0]))
        .expect("the empty frame");
    eprintln!("#2946 A3 planted: E(planted) = {retained}, E(complement) = {complement}, E(empty) = {empty}");
    assert_eq!(retained, 0.0, "E on the planted subspace must be exactly 0, got {retained}");
    assert_eq!(complement, empty);
    assert!(complement > 0.0, "the complement must discard a positive error, got {complement}");
}

#[test]
fn monte_carlo_of_the_executed_block_reproduces_explained_variance_and_discarded_error() {
    // Couple Z' = P Z + (I − P) Z̃. For the executed block, E⟨F(Z) − μ, F(Z') − μ⟩_M = V(P), ½ E‖F(Z) − F(Z')‖²_M =
    // E(P), E‖F(Z) − μ‖²_M = V(I), and by R2 E‖F(Z) − F̄_P(PZ)‖²_M = E(P). None of these estimators touches the
    // kernels, so they check V, E and F̄_P against the block itself.
    let readers = overlapping_readers();
    let block = relu_block(readers.clone());
    let frame = retained_frame();
    let output_metric = metric();
    let draws = 1 << 18;
    let mut rng = StdRng::seed_from_u64(0x2946_a3);
    let mut outputs = Array2::<f64>::zeros((draws, 3));
    let mut coupled_outputs = Array2::<f64>::zeros((draws, 3));
    let mut inputs = Array2::<f64>::zeros((draws, 4));
    for draw in 0..draws {
        let z = standard_normal_vector(&mut rng, 4);
        let independent = standard_normal_vector(&mut rng, 4);
        let coupled = &independent + &project(frame.view(), (&z - &independent).view());
        outputs.row_mut(draw).assign(&execute_relu_block(&readers, z.view()));
        coupled_outputs
            .row_mut(draw)
            .assign(&execute_relu_block(&readers, coupled.view()));
        inputs.row_mut(draw).assign(&z);
    }
    let mean = outputs.mean_axis(Axis(0)).expect("draws");
    let coupled_mean = coupled_outputs.mean_axis(Axis(0)).expect("draws");
    let retained = block
        .retained_response(frame.view(), inputs.view())
        .expect("finite draws");
    let mut total_terms = Vec::with_capacity(draws);
    let mut explained_terms = Vec::with_capacity(draws);
    let mut coupled_error_terms = Vec::with_capacity(draws);
    let mut residual_error_terms = Vec::with_capacity(draws);
    for draw in 0..draws {
        let centered = &outputs.row(draw) - &mean;
        let coupled_centered = &coupled_outputs.row(draw) - &coupled_mean;
        let difference = &outputs.row(draw) - &coupled_outputs.row(draw);
        let residual = &outputs.row(draw) - &retained.row(draw);
        total_terms.push(centered.dot(&output_metric.dot(&centered)));
        explained_terms.push(centered.dot(&output_metric.dot(&coupled_centered)));
        coupled_error_terms.push(0.5 * difference.dot(&output_metric.dot(&difference)));
        residual_error_terms.push(residual.dot(&output_metric.dot(&residual)));
    }
    let multiple = standard_error_multiple(4);
    // Centring on the sample mean scales each covariance estimate by (n − 1)/n; undo it so the estimate is unbiased.
    let unbias = draws as f64 / (draws as f64 - 1.0);
    let (total_estimate, total_se) = mean_and_standard_error(&total_terms);
    let (explained_estimate, explained_se) = mean_and_standard_error(&explained_terms);
    let (coupled_error_estimate, coupled_error_se) = mean_and_standard_error(&coupled_error_terms);
    let (residual_error_estimate, residual_error_se) = mean_and_standard_error(&residual_error_terms);

    let explained = block.explained_variance(frame.view()).expect("frame");
    let discarded = block.discarded_error(frame.view()).expect("frame");
    let arms = [
        ("V(I)", block.total_variance(), total_estimate * unbias, total_se),
        ("V(P)", explained, explained_estimate * unbias, explained_se),
        ("E(P) coupled", discarded, coupled_error_estimate, coupled_error_se),
        ("E(P) residual of F-bar", discarded, residual_error_estimate, residual_error_se),
    ];
    for (label, analytic, estimate, standard_error) in arms {
        eprintln!(
            "#2946 A3 {label}: analytic {analytic} executed MC {estimate} se {standard_error} z {:.3} multiple {multiple:.3} draws {draws}",
            (analytic - estimate) / standard_error,
        );
        assert!(
            (analytic - estimate).abs() <= multiple * standard_error,
            "{label}: analytic {analytic} vs executed Monte Carlo {estimate} ± {standard_error} (multiple {multiple})",
        );
    }

    // Positive control: the same operator without its cross terms is rejected by the same arms.
    let diagonal_total = diagonal_only_relu_variance(&readers, identity(4).view());
    let diagonal_explained = diagonal_only_relu_variance(&readers, frame.view());
    let controls = [
        ("V(P) without cross terms", diagonal_explained, explained_estimate * unbias, explained_se),
        (
            "E(P) without cross terms",
            diagonal_total - diagonal_explained,
            coupled_error_estimate,
            coupled_error_se,
        ),
    ];
    for (label, control, estimate, standard_error) in controls {
        eprintln!(
            "#2946 A3 control {label}: {control} executed MC {estimate} se {standard_error} z {:.3}",
            (control - estimate) / standard_error,
        );
        assert!(
            (control - estimate).abs() > multiple * standard_error,
            "{label} must be rejected: control {control} vs executed Monte Carlo {estimate} ± {standard_error}",
        );
    }
}

#[test]
fn retained_response_is_the_executed_block_averaged_over_the_discarded_input() {
    // R1 at fixed points: F̄_P(Pz) = E F(Pz + (I − P) Z̃), estimated by executing the block.
    let readers = overlapping_readers();
    let block = relu_block(readers.clone());
    let frame = retained_frame();
    let points = array![[0.0, 0.0, 0.0, 0.0], [1.0, -1.0, 0.5, 2.0], [-2.0, 0.5, 1.0, -1.0]];
    let response = block
        .retained_response(frame.view(), points.view())
        .expect("finite points");
    let draws = 1 << 16;
    let mut rng = StdRng::seed_from_u64(0x2946_a1);
    let multiple = standard_error_multiple(points.nrows() * 3);
    for point in 0..points.nrows() {
        let retained = project(frame.view(), points.row(point));
        let mut samples = Array2::<f64>::zeros((draws, 3));
        for draw in 0..draws {
            let independent = standard_normal_vector(&mut rng, 4);
            let input = &retained + &independent - &project(frame.view(), independent.view());
            samples.row_mut(draw).assign(&execute_relu_block(&readers, input.view()));
        }
        let plug_in = execute_relu_block(&readers, retained.view());
        for output in 0..3 {
            let column: Vec<f64> = samples.column(output).to_vec();
            let (estimate, standard_error) = mean_and_standard_error(&column);
            let analytic = response[[point, output]];
            eprintln!(
                "#2946 R1 point {point} output {output}: analytic {analytic} executed MC {estimate} se {standard_error} plug-in {}",
                plug_in[output],
            );
            assert!(
                (analytic - estimate).abs() <= multiple * standard_error,
                "point {point} output {output}: analytic {analytic} vs executed Monte Carlo {estimate} ± {standard_error}",
            );
            // Positive control at the origin, where the smoothing gap is largest: plugging the projected point into
            // F, which ignores the discarded variance, is rejected.
            if point == 0 {
                assert!(
                    (plug_in[output] - estimate).abs() > multiple * standard_error,
                    "the plug-in F(Pz) must be rejected at the origin: {} vs {estimate} ± {standard_error}",
                    plug_in[output],
                );
            }
        }
    }
}

#[test]
fn the_output_bias_moves_every_response_and_no_variance() {
    let readers = overlapping_readers();
    let output_bias = array![0.5, -1.25, 2.0];
    let unbiased = relu_block(readers.clone());
    let biased = relu_block_with_output_bias(readers, output_bias.clone());
    let frame = retained_frame();
    let points = array![[0.0, 0.0, 0.0, 0.0], [1.0, -1.0, 0.5, 2.0], [-2.0, 0.5, 1.0, -1.0]];
    let response = unbiased
        .retained_response(frame.view(), points.view())
        .expect("finite points");
    assert_eq!(
        biased
            .retained_response(frame.view(), points.view())
            .expect("finite points"),
        &response + &output_bias,
    );
    assert_eq!(biased.total_variance(), unbiased.total_variance());
    assert_eq!(
        biased.explained_variance(frame.view()).expect("frame"),
        unbiased.explained_variance(frame.view()).expect("frame"),
    );
    assert_eq!(
        biased
            .explained_variance_gradient(frame.view())
            .expect("frame")
            .horizontal_gradient,
        unbiased
            .explained_variance_gradient(frame.view())
            .expect("frame")
            .horizontal_gradient,
    );
    let refusal = KnownBlock::new(
        overlapping_readers(),
        Array1::zeros(4),
        writers(),
        Array1::zeros(2),
        metric().view(),
        GaussianActivation::Relu,
    );
    assert!(
        matches!(
            refusal,
            Err(ResponseError::DimensionMismatch {
                context: "block output bias",
                expected: 3,
                got: 2,
            })
        ),
        "an output bias of the wrong length must be refused, got {refusal:?}",
    );
}

#[test]
fn the_operator_refuses_an_asymmetric_or_indefinite_metric_and_an_over_wide_frame() {
    let mut asymmetric = metric();
    asymmetric[[0, 1]] = 0.25;
    let refusal = KnownBlock::new(
        overlapping_readers(),
        Array1::zeros(4),
        writers(),
        Array1::zeros(3),
        asymmetric.view(),
        GaussianActivation::Relu,
    );
    assert!(
        matches!(refusal, Err(ResponseError::MetricNotSymmetric { row: 0, column: 1 })),
        "an asymmetric metric must be refused, got {refusal:?}",
    );
    let indefinite = array![[1.0, 2.0, 0.0], [2.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let refusal = KnownBlock::new(
        overlapping_readers(),
        Array1::zeros(4),
        writers(),
        Array1::zeros(3),
        indefinite.view(),
        GaussianActivation::Relu,
    );
    assert!(
        matches!(refusal, Err(ResponseError::MetricNotPositiveDefinite { .. })),
        "an indefinite metric must be refused, got {refusal:?}",
    );
    let block = relu_block(overlapping_readers());
    let wide = Array2::<f64>::zeros((4, 5));
    let refusal = block.explained_variance(wide.view());
    assert!(
        matches!(refusal, Err(ResponseError::FrameWiderThanInput { rank: 5, input_dim: 4 })),
        "a frame wider than the input must be refused, got {refusal:?}",
    );
}

#[test]
fn a_rounding_scale_frame_defect_is_projected_and_a_stretched_frame_is_refused() {
    // (1 + 1e-12)·H[:, :2]: its Gram overshoots I by about 2e-12, so each planted reader's in-frame covariance exceeds
    // its variance by that share. The measured defect enters the stated covariance rounding, so the kernel projects
    // the law onto the Cauchy–Schwarz boundary instead of refusing it.
    let readers = planted_readers();
    let block = relu_block(readers.clone());
    let frame = retained_frame() * (1.0 + 1.0e-12);
    let explained = block.explained_variance(frame.view());
    assert!(
        explained.is_ok(),
        "a rounding-scale defect must be projected, got {explained:?}",
    );
    // Positive control: the same diagonal law stating no rounding is refused, so the band is what projects it.
    let coordinates = readers.dot(&frame);
    let variance = readers.row(0).dot(&readers.row(0));
    let covariance = coordinates.row(0).dot(&coordinates.row(0));
    assert!(covariance > variance, "the stretched frame must overshoot: {covariance} vs {variance}");
    let refusal = pair_kernel(
        GaussianActivation::Relu,
        PreactivationPair {
            mean_x: 0.0,
            mean_y: 0.0,
            variance_x: variance,
            variance_y: variance,
            covariance,
            covariance_rounding: 0.0,
        },
    );
    assert!(
        matches!(
            refusal,
            Err(gam_math::gaussian_activation::GaussianActivationError::CovarianceOutsideCauchySchwarz { .. })
        ),
        "without a stated rounding the overshooting law must be refused, got {refusal:?}",
    );
    // A frame stretched far past rounding has no nearest orthonormal frame the operator could be exact for.
    let stretched = array![[2.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]];
    let refusal = block.explained_variance(stretched.view());
    assert!(
        matches!(refusal, Err(ResponseError::FrameNotOrthonormal { .. })),
        "a stretched frame must be refused, got {refusal:?}",
    );
}
