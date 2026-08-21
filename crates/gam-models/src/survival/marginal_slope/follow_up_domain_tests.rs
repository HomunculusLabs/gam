#![cfg(test)]
//! Gates for the follow-up-varying likelihood domain (#2765 / #2767).
//!
//! The family is a transformation model, so its row log-density carries
//! `log η′₁` and its DOMAIN is `η′₁ > 0` at every event row, with
//!
//! ```text
//!   η′₁(t) = q′(t)·c(t) + q(t)·c′(t) + b′(t)ᵀz,    c = √(1 + bᵀΣb).
//! ```
//!
//! With a time-CONSTANT slope the last two terms vanish and `η′₁ = q′·c ≥ q′`,
//! so the time block's linear guard implies the condition and there is nothing
//! for a separate rule to do. A follow-up-varying slope breaks that implication
//! — `q·c′` and `b′ᵀz` carry no sign and read three blocks — and the condition
//! became a refusal at the row evaluator instead of a feasible set the solver
//! could steer by.
//!
//! These gates pin the two halves of putting it back where a solver can use it,
//! and pin the property that makes them trustworthy: the limiter and the row
//! program's own admission are the SAME arithmetic, so they cannot disagree
//! about which side of the boundary a coefficient is on.

use super::*;
use crate::custom_family::CustomFamily;
use gam_linalg::matrix::DenseDesignMatrix;
use ndarray::{Array1, Array2, Axis};
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;

const N_ROWS: usize = 16;

fn unit_score_covariance() -> ScoreCovarianceField {
    ScoreCovarianceField::pooled(
        MarginalSlopeCovariance::diagonal(ndarray::array![1.0])
            .expect("a 1x1 unit latent-score covariance"),
    )
}

/// The log-slope exit design: an intercept and one covariate, which is the
/// tensored shape the acceptance fixture's `logslope_formula = "1"` produces
/// against a two-column time margin.
fn logslope_exit_design() -> Array2<f64> {
    Array2::from_shape_fn((N_ROWS, 2), |(row, col)| {
        let t = (row as f64 + 0.5) / (N_ROWS as f64);
        match col {
            0 => 1.0,
            _ => t,
        }
    })
}

/// The entry channel: the same covariate chart read at the row's ENTRY time, so
/// it differs from the exit channel without being unrelated to it.
fn logslope_entry_design() -> Array2<f64> {
    Array2::from_shape_fn((N_ROWS, 2), |(row, col)| {
        let t = (row as f64 + 0.5) / (N_ROWS as f64);
        match col {
            0 => 1.0,
            _ => 0.6 * t,
        }
    })
}

/// The exit-RATE channel `∂/∂t` of the exit design. It carries no offset — the
/// property that makes `β_g = 0` a provably interior point.
fn logslope_rate_design() -> Array2<f64> {
    Array2::from_shape_fn((N_ROWS, 2), |(row, col)| {
        let t = (row as f64 + 0.5) / (N_ROWS as f64);
        match col {
            0 => 0.0,
            _ => 0.8 + 0.3 * t,
        }
    })
}

fn marginal_design() -> Array2<f64> {
    Array2::from_shape_fn((N_ROWS, 1), |(row, _)| {
        0.20 + 0.35 * (row as f64) / (N_ROWS as f64)
    })
}

/// Every row is an event, because a censored row's density has no `log η′₁`
/// factor and therefore imposes no domain condition at all — a fixture of
/// censored rows would gate nothing.
fn events() -> Array1<f64> {
    Array1::from_elem(N_ROWS, 1.0)
}

fn latent_scores() -> Array1<f64> {
    Array1::from_shape_fn(N_ROWS, |row| {
        -1.4 + 2.8 * (row as f64) / ((N_ROWS - 1) as f64)
    })
}

/// A family with a positive baseline time derivative and no time or baseline
/// chart of its own, so `q′` is exactly the offset below and the ONLY thing
/// that can drive `η′₁` negative is the slope's follow-up motion — which is
/// what these gates are about.
fn family(frame_is_follow_up_varying: bool) -> SurvivalMarginalSlopeFamily {
    let layout: LogslopeLayout = DesignMatrix::from(logslope_exit_design()).into();
    let logslope_layout = if frame_is_follow_up_varying {
        layout
            .with_follow_up(
                DesignMatrix::from(logslope_entry_design()),
                DesignMatrix::from(logslope_rate_design()),
            )
            .expect("a shared log-slope layout accepts a follow-up margin")
    } else {
        layout
    };
    SurvivalMarginalSlopeFamily {
        n: N_ROWS,
        event: Arc::new(events()),
        weights: Arc::new(Array1::from_elem(N_ROWS, 1.0)),
        z: Arc::new(latent_scores().insert_axis(Axis(1))),
        score_covariance: unit_score_covariance(),
        gaussian_frailty_sd: None,
        family_hyper: SurvivalMarginalSlopeFamilyHyperState::new(None, None)
            .expect("a family with no baseline chart and no learned sigma"),
        derivative_guard: 1e-8,
        design_entry: DesignMatrix::from(Array2::zeros((N_ROWS, 0))),
        design_exit: DesignMatrix::from(Array2::zeros((N_ROWS, 0))),
        design_derivative_exit: DesignMatrix::from(Array2::zeros((N_ROWS, 0))),
        offset_entry: Arc::new(Array1::from_elem(N_ROWS, -0.9)),
        offset_exit: Arc::new(Array1::from_elem(N_ROWS, -0.3)),
        derivative_offset_exit: Arc::new(Array1::from_elem(N_ROWS, 1.0)),
        marginal_design: DesignMatrix::from(marginal_design()),
        logslope_layout,
        score_warp: None,
        link_dev: None,
        influence_absorber: None,
        time_linear_constraints: None,
        time_wiggle_knots: None,
        time_wiggle_degree: None,
        time_wiggle_ncols: 0,
        intercept_warm_starts: None,
        auto_subsample_phase_counter: Arc::new(AtomicUsize::new(0)),
        auto_subsample_last_rho: Arc::new(std::sync::Mutex::new(None)),
    }
}

fn states(family: &SurvivalMarginalSlopeFamily, logslope_beta: Array1<f64>) -> Vec<ParameterBlockState> {
    let marginal_beta = ndarray::array![0.30];
    let marginal = family.marginal_design.to_dense().to_owned();
    let logslope = family
        .logslope_layout
        .coefficient_design()
        .to_dense()
        .to_owned();
    vec![
        ParameterBlockState {
            beta: Array1::zeros(0),
            eta: Array1::zeros(family.n),
        },
        ParameterBlockState {
            eta: marginal.dot(&marginal_beta),
            beta: marginal_beta,
        },
        ParameterBlockState {
            eta: logslope.dot(&logslope_beta),
            beta: logslope_beta,
        },
    ]
}

/// An interior log-slope coefficient: a moderate level and a mild follow-up
/// trend.
fn interior_logslope_beta() -> Array1<f64> {
    ndarray::array![0.20, -0.05]
}

/// The step direction that leaves the domain: it drives the slope's follow-up
/// RATE hard, which is the channel a time-constant slope does not have.
fn exiting_direction() -> Array1<f64> {
    // (time: 0 coefficients | marginal: 1 | log-slope: 2)
    ndarray::array![0.0, 0.0, -6.0]
}

/// Evaluate the row program's OWN admission at a coefficient, the way the
/// likelihood does. This is the oracle the limiter is graded against: the point
/// of the design is that these two never disagree.
fn row_program_admits(
    family: &SurvivalMarginalSlopeFamily,
    states: &[ParameterBlockState],
) -> Result<(), String> {
    for row in 0..family.n {
        let inputs = rigid_row_inputs(family, states, row, "follow-up domain gate")?;
        let primaries = rigid_row_kernel_primaries::<DYNAMIC_SLOPE_PRIMARIES, DynamicSlopeGeometry>(
            family, states, row,
        )?;
        let [neg_eta0, neg_eta1, adjusted_derivative] =
            rigid_row_admission_witnesses::<DYNAMIC_SLOPE_PRIMARIES, DynamicSlopeGeometry>(
                &primaries, &inputs,
            );
        validate_rigid_row_admission::<DYNAMIC_SLOPE_PRIMARIES, DynamicSlopeGeometry>(
            primaries[PRIMARY_QD1],
            &inputs,
            neg_eta0,
            neg_eta1,
            adjusted_derivative,
        )?;
    }
    Ok(())
}

fn margin_along(
    family: &SurvivalMarginalSlopeFamily,
    base: &[ParameterBlockState],
    direction: &Array1<f64>,
    alpha: f64,
) -> f64 {
    let moved = family
        .displaced_block_states(base, direction, alpha)
        .expect("the fixture's blocks and direction agree in width");
    family
        .follow_up_domain_margin(&moved)
        .expect("the fixture is on the follow-up-varying frame")
        .expect("the follow-up frame reports a margin")
}

/// The fixture has to actually exercise the thing: the base point is interior
/// and the whole step is not. Asserted separately so a later edit that makes
/// the fixture trivial fails HERE, naming the fixture, rather than making the
/// gates below vacuously true.
#[test]
fn the_fixture_step_really_leaves_the_domain_2765() {
    let family = family(true);
    let base = states(&family, interior_logslope_beta());
    let direction = exiting_direction();
    let base_margin = margin_along(&family, &base, &direction, 0.0);
    let full_margin = margin_along(&family, &base, &direction, 1.0);
    assert!(
        base_margin > 0.0,
        "the base coefficient must be interior; min η′₁ = {base_margin:.6e}"
    );
    assert!(
        full_margin < 0.0,
        "the whole step must leave the domain; min η′₁ = {full_margin:.6e}"
    );
    row_program_admits(&family, &base).expect("the row program admits the interior base");
    let outside = family
        .displaced_block_states(&base, &direction, 1.0)
        .expect("width agreement");
    let error = row_program_admits(&family, &outside)
        .expect_err("the row program must refuse the exterior endpoint");
    assert!(
        error.contains("transformed time derivative must be positive"),
        "the refusal must be the follow-up domain condition: {error}"
    );
}

/// The limiter returns a point the ROW PROGRAM admits, and it goes to the
/// boundary rather than stopping short of it.
///
/// Both halves matter. Returning an admitted point is the whole purpose — a
/// limiter whose endpoint the likelihood then refuses has moved the problem
/// rather than solved it. Going to the boundary is what keeps it a
/// globalization and not a step-crusher: an `α` far inside would shorten every
/// step in the fit for nothing.
#[test]
fn the_joint_step_limit_lands_on_a_point_the_row_program_admits_2765() {
    let family = family(true);
    let base = states(&family, interior_logslope_beta());
    let direction = exiting_direction();

    let alpha = family
        .max_feasible_follow_up_joint_step(&base, &direction)
        .expect("the follow-up frame answers")
        .expect("a step that leaves the domain must be limited");
    assert!(
        (0.0..1.0).contains(&alpha),
        "a limited step is a fraction strictly below one; got {alpha:.6e}"
    );
    assert!(alpha > 0.0, "the base is interior, so some step is feasible");

    let landed = family
        .displaced_block_states(&base, &direction, alpha)
        .expect("width agreement");
    row_program_admits(&family, &landed).unwrap_or_else(|error| {
        panic!("the row program refused the limiter's own endpoint α={alpha:.9e}: {error}")
    });

    // Tightness, measured against an independent scan rather than asserted from
    // the limiter's internals: the first grid point at which the margin turns
    // non-positive brackets the true crossing, and the returned α must be
    // inside that bracket.
    const GRID: usize = 4096;
    let mut crossing = 1.0_f64;
    for step in 1..=GRID {
        let probe = step as f64 / GRID as f64;
        if !(margin_along(&family, &base, &direction, probe) > 0.0) {
            crossing = probe;
            break;
        }
    }
    let grid_step = 1.0 / GRID as f64;
    assert!(
        alpha >= crossing - 2.0 * grid_step,
        "the limiter stopped {:.6e} short of the boundary the scan puts at {crossing:.6e} \
         (grid step {grid_step:.3e})",
        crossing - alpha,
    );
    assert!(
        alpha < crossing,
        "the limiter must stay strictly inside the scanned crossing: α={alpha:.9e}, \
         crossing={crossing:.9e}"
    );
}

/// A time-constant slope answers `None` on every direction: the rule declines
/// where the time block's own linear guard already implies the domain, so the
/// static path keeps the exact feasible set it always had.
#[test]
fn a_time_constant_slope_declines_the_joint_step_limit_2765() {
    let family = family(false);
    let base = states(&family, interior_logslope_beta());
    for scale in [-9.0_f64, -1.0, 0.0, 1.0, 9.0] {
        let direction = &exiting_direction() * scale;
        assert!(
            family
                .max_feasible_follow_up_joint_step(&base, &direction)
                .expect("the static frame answers")
                .is_none(),
            "the static frame must not limit a step (scale {scale})"
        );
        assert!(
            family
                .max_feasible_joint_step_size(&base, &direction)
                .expect("the trait hook answers")
                .is_none(),
            "the trait hook must agree with the rule it delegates to (scale {scale})"
        );
    }
}

/// A step that stays inside is not limited at all. A limiter that answered
/// `Some(1.0)` here would be indistinguishable from one that answered `None` in
/// its effect, but it would route every cycle through the `Scaled` arm and the
/// log line that goes with it; `None` is the honest answer and the gate says so.
#[test]
fn an_interior_step_is_not_limited_2765() {
    let family = family(true);
    let base = states(&family, interior_logslope_beta());
    let inward = &exiting_direction() * -0.05;
    assert!(
        margin_along(&family, &base, &inward, 1.0) > 0.0,
        "the fixture's inward step must stay interior"
    );
    assert!(
        family
            .max_feasible_follow_up_joint_step(&base, &inward)
            .expect("the follow-up frame answers")
            .is_none(),
        "an interior step must not be limited"
    );
}

/// The warm-start restoration: a seed outside the domain is retreated toward the
/// log-slope block's origin until the row program admits it, and a seed already
/// inside is left bit-identical.
#[test]
fn an_exterior_warm_start_is_retreated_into_the_domain_2765() {
    let family = family(true);
    // The seed the outer step left behind: the interior base plus the whole
    // exiting step, i.e. exactly the exterior point the previous gate pinned.
    let exterior_beta = &interior_logslope_beta() + &ndarray::array![0.0, -6.0];
    let mut blocks = blocks_for(&family, exterior_beta.clone());
    let fraction = family
        .retreat_seed_into_follow_up_domain(&mut blocks)
        .expect("the retreat answers on the follow-up frame");
    assert!(
        fraction > 0.0 && fraction <= 1.0,
        "an exterior seed must be retreated by a real fraction; got {fraction:.6e}"
    );
    let restored = blocks[2]
        .initial_beta
        .as_ref()
        .expect("the retreat writes the seed it restored")
        .clone();
    assert!(
        restored
            .iter()
            .zip(exterior_beta.iter())
            .any(|(a, b)| a != b),
        "the retreat must move the seed it declared exterior"
    );
    row_program_admits(&family, &states(&family, restored.clone())).unwrap_or_else(|error| {
        panic!("the retreated seed is still outside the domain: {error}")
    });
    // The retreat is toward the ORIGIN of the log-slope block, so every
    // coordinate shrinks by the same factor and none changes sign.
    for (restored_value, seed_value) in restored.iter().zip(exterior_beta.iter()) {
        assert!(
            restored_value.abs() <= seed_value.abs() + f64::EPSILON,
            "a retreat toward the origin cannot grow a coordinate: {restored_value} vs {seed_value}"
        );
        assert!(
            restored_value * seed_value >= 0.0,
            "a retreat toward the origin cannot cross zero: {restored_value} vs {seed_value}"
        );
    }
}

#[test]
fn an_interior_warm_start_is_left_alone_2765() {
    let family = family(true);
    let seed = interior_logslope_beta();
    let mut blocks = blocks_for(&family, seed.clone());
    let fraction = family
        .retreat_seed_into_follow_up_domain(&mut blocks)
        .expect("the retreat answers on the follow-up frame");
    assert_eq!(
        fraction, 0.0,
        "an interior seed is not retreated at all"
    );
    let untouched = blocks[2]
        .initial_beta
        .as_ref()
        .expect("the fixture seeded this block");
    assert_eq!(
        untouched.to_vec(),
        seed.to_vec(),
        "an interior seed must be left bit-identical"
    );
}

/// The static frame never retreats, whatever the seed.
#[test]
fn a_time_constant_warm_start_is_never_retreated_2765() {
    let family = family(false);
    let seed = &interior_logslope_beta() + &ndarray::array![0.0, -6.0];
    let mut blocks = blocks_for(&family, seed.clone());
    let fraction = family
        .retreat_seed_into_follow_up_domain(&mut blocks)
        .expect("the static frame answers");
    assert_eq!(fraction, 0.0, "the static frame declines the retreat");
    assert_eq!(
        blocks[2]
            .initial_beta
            .as_ref()
            .expect("the fixture seeded this block")
            .to_vec(),
        seed.to_vec(),
        "the static frame must leave the seed bit-identical"
    );
}

fn blocks_for(
    family: &SurvivalMarginalSlopeFamily,
    logslope_beta: Array1<f64>,
) -> Vec<ParameterBlockSpec> {
    let state = states(family, logslope_beta);
    let designs: [DesignMatrix; 3] = [
        family.design_exit.clone(),
        family.marginal_design.clone(),
        family.logslope_layout.coefficient_design().clone(),
    ];
    state
        .into_iter()
        .zip(designs)
        .enumerate()
        .map(|(index, (block_state, design))| ParameterBlockSpec {
            name: format!("follow_up_domain_{index}"),
            design: DesignMatrix::Dense(DenseDesignMatrix::from(
                design.to_dense().to_owned(),
            )),
            offset: Array1::zeros(N_ROWS),
            penalties: Vec::new(),
            nullspace_dims: Vec::new(),
            initial_log_lambdas: Array1::zeros(0),
            initial_beta: Some(block_state.beta),
            gauge_priority: 100,
            jacobian_callback: None,
            stacked_design: None,
            stacked_offset: None,
        })
        .collect()
}
