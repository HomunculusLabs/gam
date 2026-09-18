#![cfg(test)]
//! #2946 Stage E pins for the compiled retained response.
//!
//! The blocks are zero-bias ReLU and exact-GELU blocks on dyadic data, as in `subspace_tests.rs`: readers with small
//! half-integer entries and frames built from `H = I − ½·11ᵀ`, so `E(P)` of a frame containing every reader is exactly
//! zero. Monte Carlo agreement is accepted within a standard-error multiple derived from a declared false-alarm rate,
//! and each agreement arm has a positive control the test shows it rejects. Executed draws come from the same draw owner
//! the compile uses, on seeds distinct from the compile's.

use super::{
    CompileAction, CompileDesign, CompiledResponse, FunctionRepresentation, compile_retained_response,
    enrichment_step, frame_parameter_count,
};
use crate::response::subspace::KnownBlock;
use gam_linalg::roundoff::accumulation_growth;
use gam_linalg::utils::splitmix64;
use gam_math::gaussian_activation::GaussianActivation;
use gam_math::probability::{normal_cdf, standard_normal_from_uniform_bits, standard_normal_quantile};
use ndarray::{Array1, Array2, ArrayView1, array, s};

/// The declared probability that a correct compile fails one Monte Carlo test for a fresh seed, split evenly over the
/// test's two-sided agreement arms (Bonferroni).
const FAMILY_FALSE_ALARM: f64 = 1.0e-6;

const TRAINING_DRAWS: usize = 2_000;
const HOLDOUT_DRAWS: usize = 2_000;
const EXECUTED_DRAWS: usize = 20_000;

/// The acceptance multiple of one two-sided arm among `arms`.
fn standard_error_multiple(arms: usize) -> f64 {
    standard_normal_quantile(1.0 - FAMILY_FALSE_ALARM / (2.0 * arms as f64))
        .expect("the false-alarm split lies inside (0, 1)")
}

/// `H = I − ½·11ᵀ` in four dimensions: symmetric, orthogonal, every entry `±½`.
fn half_reflector() -> Array2<f64> {
    Array2::from_shape_fn((4, 4), |(row, column)| if row == column { 0.5 } else { -0.5 })
}

/// The first two columns of `H` span `{(1, −1, 0, 0), (0, 0, 1, 1)}`.
fn retained_frame() -> Array2<f64> {
    half_reflector().slice(s![.., ..2]).to_owned()
}

/// The last two columns of `H` span `{(1, 1, 0, 0), (0, 0, 1, −1)}`, the complement of [`retained_frame`].
fn complement_frame() -> Array2<f64> {
    half_reflector().slice(s![.., 2..]).to_owned()
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

/// A zero-bias block with no output bias, the executed form [`execute_block`] runs.
fn block_with(readers: Array2<f64>, block_writers: Array2<f64>, activation: GaussianActivation) -> KnownBlock {
    let width = readers.nrows();
    let output_dim = block_writers.nrows();
    KnownBlock::new(
        readers,
        Array1::zeros(width),
        block_writers,
        Array1::zeros(output_dim),
        metric().view(),
        activation,
    )
    .expect("a finite block with a symmetric positive definite metric")
}

fn known_block(readers: Array2<f64>, activation: GaussianActivation) -> KnownBlock {
    block_with(readers, writers(), activation)
}

/// The executed zero-bias block `U σ(W z)`.
fn execute_block(
    readers: &Array2<f64>,
    block_writers: &Array2<f64>,
    activation: GaussianActivation,
    z: ArrayView1<'_, f64>,
) -> Array1<f64> {
    let preactivations = readers.dot(&z);
    let activations = match activation {
        GaussianActivation::Relu => preactivations.mapv(|t| t.max(0.0)),
        GaussianActivation::ExactGelu => preactivations.mapv(|t| t * normal_cdf(t)),
        GaussianActivation::Silu => {
            panic!("the compile pins execute the closed-form ReLU and exact-GELU blocks only")
        }
    };
    block_writers.dot(&activations)
}

fn metric_norm_squared(v: ArrayView1<'_, f64>) -> f64 {
    v.dot(&metric().dot(&v))
}

/// Executed draws `Z ~ N(0, I_4)` from the draw owner.
fn executed_draws(seed: u64) -> Array2<f64> {
    let mut state = seed;
    let mut draws = Array2::<f64>::zeros((EXECUTED_DRAWS, 4));
    for slot in draws.iter_mut() {
        *slot = standard_normal_from_uniform_bits(splitmix64(&mut state)).expect("inversion never refuses its words");
    }
    draws
}

/// The mean of i.i.d. terms and its standard error.
fn mean_and_standard_error(terms: &[f64]) -> (f64, f64) {
    let count = terms.len() as f64;
    let mean = terms.iter().sum::<f64>() / count;
    let variance = terms.iter().map(|term| (term - mean).powi(2)).sum::<f64>() / (count - 1.0);
    (mean, (variance / count).sqrt())
}

/// `‖F(z) − g(Qᵀz)‖²_M` over executed draws of the block, with its standard error.
fn executed_total_error(
    compiled: &CompiledResponse,
    readers: &Array2<f64>,
    block_writers: &Array2<f64>,
    activation: GaussianActivation,
    points: &Array2<f64>,
) -> (f64, f64) {
    let compiled_values = compiled.evaluate_input(points.view()).expect("a replayable compiled response");
    let terms: Vec<f64> = (0..points.nrows())
        .map(|row| {
            let executed = execute_block(readers, block_writers, activation, points.row(row));
            metric_norm_squared((&executed - &compiled_values.row(row)).view())
        })
        .collect();
    mean_and_standard_error(&terms)
}

#[test]
fn a_planted_smooth_function_of_a_planted_subspace_is_recovered() {
    // Exact-GELU readers of unit scale inside the span of the retained frame: `F = F̄_P` is a smooth function of `PZ`.
    let readers = array![
        [0.5, -0.5, 0.0, 0.0],
        [0.0, 0.0, 0.5, 0.5],
        [0.5, -0.5, 0.5, 0.5],
        [1.0, -1.0, -0.5, -0.5],
    ];
    let activation = GaussianActivation::ExactGelu;
    let block = known_block(readers.clone(), activation);
    let total_variance = block.total_variance();
    assert!(total_variance > 0.0, "the planted block has output variance {total_variance}");
    let design = CompileDesign::pilot(TRAINING_DRAWS, HOLDOUT_DRAWS, 2);
    let compiled = compile_retained_response(&block, retained_frame().view(), design, 2_946_051)
        .expect("a planted block compiles on its own frame");
    let split = compiled.split();
    assert_eq!(split.discarded_error, 0.0, "the planted frame holds every reader: {split:?}");
    assert!(
        matches!(compiled.representation(), FunctionRepresentation::Reml { .. }),
        "a GELU response is fitted, not the mean: {:?}",
        compiled.representation()
    );

    // With `E(P) = 0`, R2 says the executed error of `g` is the function error alone.
    let multiple = standard_error_multiple(2);
    let points = executed_draws(2_946_151);
    let (executed, executed_se) = executed_total_error(&compiled, &readers, &writers(), activation, &points);
    let combined_se = executed_se.hypot(split.function_error_standard_error);
    assert!(
        (executed - split.function_error).abs() <= multiple * combined_se,
        "executed error {executed} ± {executed_se} disagrees with the function error {split:?} at multiple {multiple}"
    );

    // Recovered: the compiled function leaves under one percent of the block's output variance, at the Monte Carlo
    // upper bound. The complement frame, which sees none of the readers, must fail the same bar.
    let recovery_bar = 1.0e-2 * total_variance;
    assert!(
        executed + multiple * executed_se <= recovery_bar,
        "the compiled response leaves {executed} ± {executed_se} of variance {total_variance}"
    );
    let blind = compile_retained_response(&block, complement_frame().view(), design, 2_946_052)
        .expect("a frame orthogonal to every reader compiles to the mean");
    assert_eq!(blind.representation(), FunctionRepresentation::Constant, "a constant response is the mean");
    assert_eq!(blind.fitted_directions(), 0, "a constant response has no resolvable direction");
    let (blind_executed, blind_se) = executed_total_error(&blind, &readers, &writers(), activation, &points);
    assert!(
        blind_executed - multiple * blind_se > recovery_bar,
        "the positive control passed the recovery bar: {blind_executed} ± {blind_se} against {recovery_bar}"
    );
}

/// The rounding bar of a compile of `block`: its variance scaled by the growth factor of an accumulation as deep as the
/// training design (`n·p` products). A REML fit that rails a penalty at the lower edge of its resolvability domain
/// shrinks each direction it penalizes by at most `√ε` of itself, a squared error of `ε` per penalty, under this bar.
fn rounding_bar(block: &KnownBlock, compiled: &CompiledResponse) -> f64 {
    block.total_variance() * accumulation_growth(TRAINING_DRAWS * compiled.price().function_coefficients)
}

#[test]
fn a_response_the_design_reproduces_compiles_to_rounding() {
    // `relu(t) − relu(−t) = t`: two opposite ReLU units with opposite writers make `F = u wᵀz` linear, inside the
    // Duchon design's affine block. Only the intercept is unpenalized, so this is no interpolation without a finite
    // optimum: REML rails the slope penalties at the lower edge of its domain and reproduces `F` to rounding.
    let readers = array![[0.5, -0.5, 0.0, 0.0], [-0.5, 0.5, 0.0, 0.0]];
    let linear_writers = array![[1.0, -1.0], [0.0, 0.0], [1.0, -1.0]];
    let block = block_with(readers.clone(), linear_writers.clone(), GaussianActivation::Relu);
    let design = CompileDesign::pilot(TRAINING_DRAWS, HOLDOUT_DRAWS, 2);
    let compiled = compile_retained_response(&block, retained_frame().view(), design, 2_946_055)
        .expect("a linear response compiles, not a refusal");
    let split = compiled.split();
    assert_eq!(split.discarded_error, 0.0, "both readers lie in the frame: {split:?}");
    assert_eq!(compiled.fitted_directions(), 1, "a rank-one response resolves one output direction");
    let bar = rounding_bar(&block, &compiled);
    let points = executed_draws(2_946_155);
    let (executed, executed_se) =
        executed_total_error(&compiled, &readers, &linear_writers, GaussianActivation::Relu, &points);
    assert!(
        split.function_error <= bar && executed <= bar,
        "the linear response leaves {split:?} held out and {executed} ± {executed_se} executed against {bar}: {:?}",
        compiled.representation()
    );

    // Positive control: without the opposite unit `F = u relu(wᵀz)` keeps its kink, which no Duchon design reproduces,
    // and the same bar must reject it.
    let kinked_readers = readers.slice(s![..1, ..]).to_owned();
    let kinked_writers = linear_writers.slice(s![.., ..1]).to_owned();
    let kinked = block_with(kinked_readers, kinked_writers, GaussianActivation::Relu);
    let kinked_compiled = compile_retained_response(&kinked, retained_frame().view(), design, 2_946_060)
        .expect("a single ReLU unit compiles");
    let kinked_bar = rounding_bar(&kinked, &kinked_compiled);
    assert!(
        kinked_compiled.split().function_error > kinked_bar,
        "the positive control passed the rounding bar: {:?} against {kinked_bar}",
        kinked_compiled.split()
    );
}

#[test]
fn the_error_split_sums_to_the_executed_total() {
    // Readers whose discarded parts all overlap, so `E(P)` carries cross terms.
    let readers = array![
        [2.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0],
        [2.0, 0.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, -1.0],
    ];
    let activation = GaussianActivation::Relu;
    let block = known_block(readers.clone(), activation);
    let frame = retained_frame();
    let design = CompileDesign::pilot(TRAINING_DRAWS, HOLDOUT_DRAWS, 2);
    let compiled =
        compile_retained_response(&block, frame.view(), design, 2_946_053).expect("the overlapping block compiles");
    let split = compiled.split();
    assert!(split.discarded_error > 0.0, "the frame discards part of every reader: {split:?}");

    let multiple = standard_error_multiple(2);
    let points = executed_draws(2_946_153);
    let compiled_values = compiled.evaluate_input(points.view()).expect("a replayable compiled response");
    let retained = block
        .retained_response(frame.view(), points.view())
        .expect("the retained response at the executed draws");

    // Arm 1, unpaired: the executed total against `E(P) + Â`, from independent draws.
    let (executed, executed_se) = executed_total_error(&compiled, &readers, &writers(), activation, &points);
    assert!(
        (executed - split.total()).abs() <= multiple * executed_se.hypot(split.function_error_standard_error),
        "executed total {executed} ± {executed_se} disagrees with the split {split:?} at multiple {multiple}"
    );

    // Arm 2, paired at the same draws: `‖F − g‖²_M − ‖F̄_P − g‖²_M` has mean `E(P)` exactly when the cross term
    // `E⟨F − F̄_P, F̄_P − g⟩_M` vanishes, which R2 guarantees for every `g` of `PZ`.
    let discarded_coordinate = complement_frame().column(0).to_owned();
    let paired_terms = |leak: &Array1<f64>| -> Vec<f64> {
        (0..points.nrows())
            .map(|row| {
                let z = points.row(row);
                let executed = execute_block(&readers, &writers(), activation, z);
                let leaked = &compiled_values.row(row) + &(leak * discarded_coordinate.dot(&z));
                metric_norm_squared((&executed - &leaked).view())
                    - metric_norm_squared((&retained.row(row) - &leaked).view())
            })
            .collect()
    };
    let (paired, paired_se) = mean_and_standard_error(&paired_terms(&Array1::zeros(3)));
    assert!(
        (paired - split.discarded_error).abs() <= multiple * paired_se,
        "paired difference {paired} ± {paired_se} disagrees with E(P) = {} at multiple {multiple}",
        split.discarded_error
    );

    // Positive control: add a function of the discarded coordinate `q⊥ᵀz` to `g`. The cross term no longer vanishes (by
    // Stein, `E[relu(wᵀz) q⊥ᵀz] = ½ wᵀq⊥ ≠ 0` for three of the readers), and the check must reject it.
    let (leaked, leaked_se) = mean_and_standard_error(&paired_terms(&array![1.0, 1.0, 1.0]));
    assert!(
        (leaked - split.discarded_error).abs() > multiple * leaked_se,
        "the leaked evaluator passed the orthogonality check: {leaked} ± {leaked_se} against E(P) = {}",
        split.discarded_error
    );
}

#[test]
fn the_dominance_call_adds_a_missing_direction_and_enriches_an_under_resolved_function() {
    let multiple = standard_error_multiple(2);
    let frame = retained_frame();

    // A missing direction: one reader inside the frame and one of norm √8 along the complement's `(1, 1, 0, 0)`.
    // The two are orthogonal, so the discarded unit's whole variance is `E(P)`, while `F̄_P` is a single smoothed kink.
    let missing = array![[1.0, -1.0, 0.0, 0.0], [2.0, 2.0, 0.0, 0.0]];
    let block = block_with(missing, writers().slice(s![.., ..2]).to_owned(), GaussianActivation::Relu);
    let design = CompileDesign::pilot(TRAINING_DRAWS, HOLDOUT_DRAWS, 2);
    let compiled =
        compile_retained_response(&block, frame.view(), design, 2_946_054).expect("the missing-direction block compiles");
    let split = compiled.split();
    let call = compiled.dominance_call();
    assert_eq!(call.action, CompileAction::AddDirection, "{split:?} {call:?}");
    assert!(
        split.discarded_error - split.function_error > multiple * split.function_error_standard_error,
        "the missing direction does not dominate at multiple {multiple}: {split:?}"
    );
    // The measured frame step agrees with the call: extending along the frame gradient removes more error than the
    // whole function error.
    let step = compiled.frame_step().expect("a two-dimensional frame in four dimensions has a frame step");
    assert!(
        step.gain > split.function_error + multiple * split.function_error_standard_error,
        "the frame step gains {step:?} against the function error {split:?}"
    );

    // An under-resolved function: sharp kinks of norm up to `8√3` inside the frame, a quarter-unit discarded part on
    // one reader so `E(P) > 0`, and a basis of ten centers.
    let sharp = array![
        [8.25, -7.75, 0.0, 0.0],
        [0.0, 0.0, 8.0, 8.0],
        [8.0, -8.0, 8.0, 8.0],
        [-8.0, 8.0, 8.0, 8.0],
    ];
    let block = known_block(sharp, GaussianActivation::Relu);
    let coarse = CompileDesign {
        training_draws: TRAINING_DRAWS,
        holdout_draws: HOLDOUT_DRAWS,
        centers: 10,
    };
    let compiled =
        compile_retained_response(&block, frame.view(), coarse, 2_946_056).expect("the sharp block compiles coarsely");
    let split = compiled.split();
    let call = compiled.dominance_call();
    assert!(split.discarded_error > 0.0, "the sharp block discards a quarter-unit part: {split:?}");
    assert_eq!(call.action, CompileAction::EnrichFunction, "{split:?} {call:?}");
    assert!(
        split.function_error - split.discarded_error > multiple * split.function_error_standard_error,
        "the under-resolved function does not dominate at multiple {multiple}: {split:?}"
    );

    // The measured enrichment step agrees with the call: the next resolution removes function error.
    let step = enrichment_step(&block, &compiled, 2_946_057)
        .expect("the sharp block compiles at the enriched resolution")
        .expect("ten centers lie below the production ceiling");
    assert_eq!(step.enriched.split().discarded_error, split.discarded_error, "enrichment never moves E(P)");
    assert!(
        step.gain > multiple * step.gain_standard_error,
        "enrichment did not reduce the function error: {split:?} then {:?}",
        step.enriched.split()
    );
}

#[test]
fn the_price_reports_gauge_correct_frames_the_function_and_the_whole_block_for_an_executed_residual() {
    // Stiefel minus `dim O(r)` is Grassmann.
    assert_eq!(frame_parameter_count(2, 4, false), 2 * 4 - 3);
    assert_eq!(frame_parameter_count(2, 4, true), 2 * (4 - 2));
    assert_eq!(frame_parameter_count(1, 4, true), frame_parameter_count(1, 4, false), "O(1) is discrete");

    let readers = array![
        [2.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0],
        [2.0, 0.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, -1.0],
    ];
    let block = known_block(readers, GaussianActivation::Relu);
    let design = CompileDesign::pilot(TRAINING_DRAWS, HOLDOUT_DRAWS, 2);
    let compiled =
        compile_retained_response(&block, retained_frame().view(), design, 2_946_058).expect("the block compiles");
    let price = compiled.price();
    let fitted = compiled.fitted_directions();
    assert_eq!(fitted, 3, "three writer directions of a generic response are resolvable");
    assert_eq!(price.reader_frame, frame_parameter_count(2, 4, true), "isotropic Duchon has the rotation gauge");
    assert_eq!(price.writer_frame, frame_parameter_count(fitted, 3, true));
    assert_eq!(price.output_mean, 3);
    assert_eq!(price.connections, 2 * fitted);
    assert_eq!(price.function_coefficients % fitted, 0, "one coefficient vector per direction: {price:?}");
    let FunctionRepresentation::Reml { smoother_edf, .. } = compiled.representation() else {
        panic!("a ReLU response with discarded parts is fitted, not the mean: {:?}", compiled.representation());
    };
    assert_eq!(price.function_edf, smoother_edf, "the ledger reports the one shared-smoother figure");
    assert!(
        smoother_edf > 0.0 && smoother_edf <= (price.function_coefficients / fitted) as f64,
        "the shared smoother's effective degrees of freedom lie within its design width: {price:?}"
    );
    assert_eq!(
        price.residual.executed_parameters,
        4 * (4 + 1 + 3) + 3,
        "readers, biases, writers and output bias of the executed residual"
    );
    assert_eq!(price.residual.unexplained_error, compiled.split().total());
    assert_eq!(compiled.seed(), 2_946_058);
}

#[test]
fn the_frozen_basis_replays_the_fit_time_design_at_the_training_draws() {
    // Every held-out and executed evaluation of `g` goes through the replay spec, so it must rebuild the fit-time design
    // at the fit's own rows, whatever other rows it is handed with them: the replay runs on the training rows stacked
    // above fresh rows, and its training block must match. A replay that re-derived its radial chart, its centers or any
    // state from the rows it is handed would differ at `O(scale)`; the bound is the growth factor of an accumulation as
    // deep as the design's `p × p` transforms applied across its `p` columns, at the design's own scale.
    use gam_terms::basis::{
        CenterStrategy, DuchonBasisSpec, DuchonOperatorPenaltySpec, OneDimensionalBoundary, SpatialIdentifiability,
        build_duchon_basis, duchon_cubic_default, starting_num_centers,
    };
    let mut state = 2_946_059_u64;
    let draws = Array2::from_shape_simple_fn((TRAINING_DRAWS, 2), || {
        standard_normal_from_uniform_bits(splitmix64(&mut state)).expect("inversion never refuses its words")
    });
    let (nullspace_order, power) = duchon_cubic_default(2);
    let spec = DuchonBasisSpec {
        center_strategy: CenterStrategy::EqualMass {
            num_centers: starting_num_centers(TRAINING_DRAWS, 2),
        },
        periodic: None,
        length_scale: None,
        power,
        nullspace_order,
        identifiability: SpatialIdentifiability::None,
        aniso_log_scales: None,
        operator_penalties: DuchonOperatorPenaltySpec::default(),
        boundary: OneDimensionalBoundary::Open,
        radial_reparam: None,
    };
    let built = build_duchon_basis(draws.view(), &spec).expect("a Duchon basis on Gaussian draws");
    let fit_design = super::dense_design(&built).expect("a dense fit-time design");
    let replay = super::replay_spec(&built.metadata).expect("the fit carries a replayable state");
    let fresh = Array2::from_shape_simple_fn((HOLDOUT_DRAWS, 2), || {
        standard_normal_from_uniform_bits(splitmix64(&mut state)).expect("inversion never refuses its words")
    });
    let mut stacked = Array2::<f64>::zeros((TRAINING_DRAWS + HOLDOUT_DRAWS, 2));
    stacked.slice_mut(s![..TRAINING_DRAWS, ..]).assign(&draws);
    stacked.slice_mut(s![TRAINING_DRAWS.., ..]).assign(&fresh);
    let replayed_stack = super::replayed_design(&replay, stacked.view()).expect("the replay builds at stacked rows");
    assert_eq!(replayed_stack.nrows(), TRAINING_DRAWS + HOLDOUT_DRAWS, "one replayed row per stacked row");
    let replayed = replayed_stack.slice(s![..TRAINING_DRAWS, ..]).to_owned();
    assert_eq!(fit_design.dim(), replayed.dim(), "the replay realizes the fit-time width");
    let width = fit_design.ncols();
    let scale = fit_design.iter().fold(0.0_f64, |acc, value| acc.max(value.abs()));
    let difference = fit_design
        .iter()
        .zip(replayed.iter())
        .fold(0.0_f64, |acc, (fit, replay)| acc.max((fit - replay).abs()));
    let bound = accumulation_growth(width * width) * width as f64 * scale;
    assert!(
        difference <= bound,
        "the replayed design differs from the fit-time design by {difference} against {bound} at scale {scale}"
    );
}
