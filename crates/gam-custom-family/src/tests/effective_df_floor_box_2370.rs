//! Regression (#2370): the per-term effective-df-floor ρ *upper* bound must
//! never fall below the ρ box's own *lower* bound.
//!
//! `fit_custom_family_with_rho_prior` boxes the outer ρ = log λ with
//!
//! ```text
//! lower = options.rho_lower_bound                        (default -10.0)
//! upper = effective_df_floor_rho_upper_bounds(.., RhoBox { lower, ceiling })
//! ```
//!
//! The upper-bound derivation used to bisect the `edf(ρ) = 1` crossing on
//! `[-ceiling, ceiling] = [-12, 12]` and reject a crossing only at or below
//! `-ceiling`. That lower reference was a MIRROR of the ceiling, decoupled from
//! the box's actual floor. While the ceiling was 10.0 the two coincided and the
//! defect was latent; #2356 raised the ceiling to 12.0 and opened the window
//! `(-12, -10)`. A term whose structural `edf = 1` crossing lands in that
//! window yielded an upper bound below the real floor -10, so the optimizer
//! received the inverted box `[-10, -11.855…]` and `f64::clamp(min, max)`
//! panicked with `min > max` across the FFI boundary.
//!
//! The fix anchors the pre-check, the bisection endpoint, and the acceptance
//! guard on the caller's true lower wall, so the below-box case routes to the
//! "floor not enforceable inside the box" arm (`edf(lower) ≤ 1` ⇒ since edf
//! decreases in ρ, `edf ≤ 1` everywhere in the box) and the term keeps the
//! uniform ceiling. The emitted bound is then strictly above the floor by
//! construction.

use super::*;

/// A 2-column term with `X = c·I₂` and penalty `S = I₂` has design Gram
/// `G = XᵀX = c²I₂` and structural generalized eigenvalues `γ = [c², c²]`, so
///
/// ```text
/// edf(ρ) = Σ_j γ_j/(γ_j + e^ρ) = 2c²/(c² + e^ρ),
/// ```
///
/// which equals the `EFFECTIVE_DF_FLOOR` of 1 exactly when `e^ρ = c²`. Setting
/// `c = exp(rho_star/2)` therefore places the `edf = 1` crossing at `rho_star`.
fn two_dir_term(rho_star: f64) -> (Vec<ParameterBlockSpec>, PenaltyLabelLayout) {
    let c = (0.5 * rho_star).exp(); // c² = e^{rho_star}
    let design = DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(array![
        [c, 0.0],
        [0.0, c]
    ]));
    let spec = ParameterBlockSpec {
        name: "wiggle".to_string(),
        design,
        offset: array![0.0, 0.0],
        penalties: vec![PenaltyMatrix::Dense(array![[1.0, 0.0], [0.0, 1.0]])],
        nullspace_dims: vec![0],
        initial_log_lambdas: array![0.0],
        initial_beta: Some(array![0.0, 0.0]),
        gauge_priority: 100,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    };
    let layout = PenaltyLabelLayout {
        penalty_counts: vec![1],
        physical_to_outer: vec![Some(0)],
        fixed_log_lambdas: vec![None],
        initial_rho: array![0.0],
        joint_specs: Vec::new(),
        joint_to_outer: Vec::new(),
    };
    (vec![spec], layout)
}

/// The production floor, so this tracks `rho_lower_bound` if it ever moves
/// rather than pinning a private copy of the literal.
fn production_lower() -> f64 {
    BlockwiseFitOptions::default().rho_lower_bound
}

/// Single call site for the bound derivation.
///
/// The two walls used to be adjacent bare `f64` parameters, so transposing
/// them compiled silently and surfaced only as a wrong answer — a real earlier
/// revision of this very file did exactly that. They now travel inside
/// [`RhoBox`], whose per-wall newtypes make the transposition a type error and
/// whose constructor validates the ordering once.
fn upper_bounds_for(
    specs: &[ParameterBlockSpec],
    layout: &PenaltyLabelLayout,
    ceiling: f64,
    lower: f64,
) -> Result<Array1<f64>, CustomFamilyError> {
    let rho_box = RhoBox::new(RhoLowerWall(lower), RhoCeiling(ceiling))?;
    effective_df_floor_rho_upper_bounds(specs, layout, 1, rho_box)
}

#[test]
fn crossing_between_neg_ceiling_and_the_box_floor_keeps_the_ceiling_2370() {
    let lower = production_lower();
    let ceiling = EFFECTIVE_DF_CEILING;
    // The decoupled window opened by #2356: strictly below the box floor, but
    // strictly above the old `-ceiling` reference. This is the exact geometry
    // that inverted the box.
    assert!(
        -ceiling < -11.0 && -11.0 < lower,
        "fixture must sit in the (-ceiling, lower) window: ceiling={ceiling}, lower={lower}"
    );
    let (specs, layout) = two_dir_term(-11.0);
    let upper =
        upper_bounds_for(&specs, &layout, ceiling, lower).expect("bounds derivation must succeed");

    // The invariant #2370 forbids violating. Pre-fix this returned -11.0.
    assert!(
        upper[0] >= lower,
        "effective-df-floor upper {} fell below the rho box lower bound {lower} \
         (an inverted box reaches f64::clamp and panics)",
        upper[0],
    );
    // And the correct disposition is NOT a pin at the floor: the edf floor is
    // unenforceable anywhere in this box, so the term keeps the uniform ceiling.
    assert_eq!(
        upper[0], ceiling,
        "a term whose edf=1 crossing lies below the box must keep the uniform ceiling",
    );
}

#[test]
fn interior_crossing_still_tightens_the_upper_bound_2370() {
    // Positive control: the fix must not disable legitimate tightening. A
    // crossing at ρ = 0 is well inside [-10, 12], so the bound tracks it.
    let lower = production_lower();
    let ceiling = EFFECTIVE_DF_CEILING;
    let (specs, layout) = two_dir_term(0.0);
    let upper =
        upper_bounds_for(&specs, &layout, ceiling, lower).expect("bounds derivation must succeed");
    assert!(
        upper[0] > lower && upper[0] < ceiling,
        "an interior crossing must yield an interior upper bound, got {}",
        upper[0],
    );
    assert!(
        upper[0].abs() < 1e-3,
        "upper bound must track the edf=1 crossing at rho=0, got {}",
        upper[0],
    );
}

#[test]
fn derived_upper_bound_never_inverts_the_box_across_the_crossing_range_2370() {
    // The invariant itself, swept across crossings from far below the box to
    // far above it: the derivation may only ever return a bound the box can
    // actually contain.
    let lower = production_lower();
    let ceiling = EFFECTIVE_DF_CEILING;
    for step in 0..=40 {
        let rho_star = -20.0 + f64::from(step);
        let (specs, layout) = two_dir_term(rho_star);
        let upper = upper_bounds_for(&specs, &layout, ceiling, lower)
            .expect("bounds derivation must succeed");
        assert!(
            upper[0] > lower && upper[0] <= ceiling,
            "crossing at rho*={rho_star} produced upper={} outside the box ({lower}, {ceiling}]",
            upper[0],
        );
    }
}

#[test]
fn a_pinned_box_yields_a_well_ordered_single_point_box_2370() {
    // The derivation needs no special case for a pinned box: no tightening is
    // possible (the term keeps the uniform ceiling), so the emitted upper bound
    // equals the floor and the box the optimizer receives is the single point
    // the caller pinned — never inverted.
    let pinned_at = EFFECTIVE_DF_CEILING;
    let (specs, layout) = two_dir_term(0.0);
    let upper = upper_bounds_for(&specs, &layout, pinned_at, pinned_at)
        .expect("a pinned rho box must be accepted");
    assert_eq!(
        upper[0], pinned_at,
        "a pinned box must emit its own wall as the upper bound, got {}",
        upper[0],
    );
}

#[test]
fn the_rho_box_constructor_rejects_an_inverted_pair_but_accepts_a_pinned_one_2370() {
    let ceiling = EFFECTIVE_DF_CEILING;
    // Inverted: floor above the ceiling.
    let inverted = RhoBox::new(RhoLowerWall(13.0), RhoCeiling(ceiling))
        .expect_err("an inverted pair must be rejected at construction");
    let message = inverted.to_string();
    assert!(
        message.contains("13") && message.contains("12"),
        "the refusal must name both offending walls, got: {message}"
    );
    // A PINNED box (lower == ceiling) is legal, not degenerate: the caller has
    // fixed λ. This mirrors the outer optimizer's
    // `pinned_equal_rho_bounds_are_accepted_2370`; the two layers must agree on
    // what the admissible set is, or they can drift apart exactly as the two
    // #2370 constants did.
    let pinned = RhoBox::new(RhoLowerWall(ceiling), RhoCeiling(ceiling))
        .expect("a pinned rho box must be accepted, matching the outer optimizer");
    assert_eq!(pinned.lower(), pinned.ceiling());
    // Non-finite walls are rejected as admissible log-strengths.
    assert!(
        RhoBox::new(RhoLowerWall(f64::NAN), RhoCeiling(ceiling)).is_err(),
        "a non-finite floor must be rejected"
    );
    // And the production pair is accepted, with the walls readable back in the
    // order they were supplied.
    let ok = RhoBox::new(RhoLowerWall(production_lower()), RhoCeiling(ceiling))
        .expect("the production rho box must be valid");
    assert_eq!(ok.lower(), production_lower());
    assert_eq!(ok.ceiling(), ceiling);
}

#[test]
fn a_caller_box_that_is_already_inverted_is_a_typed_error_2370() {
    // If the CALLER hands in a lower wall at or above the ceiling the box is
    // empty or degenerate before any per-term tightening; that must be a typed
    // refusal carrying both bounds, not a panic.
    let (specs, layout) = two_dir_term(0.0);
    let error = upper_bounds_for(&specs, &layout, EFFECTIVE_DF_CEILING, 13.0)
        .expect_err("an inverted caller box must be rejected");
    let message = error.to_string();
    assert!(
        message.contains("13") && message.contains("12"),
        "typed error must name both offending bounds, got: {message}"
    );
}

/// #2608 (zz_measure): how far would a RELATIVE floor move the wall for the
/// rank-1 term the absolute floor cannot touch?
///
/// `effective_df_floor_bound` opens with `if !(edf_max > EFFECTIVE_DF_FLOOR)`,
/// and the floor is exactly `1.0`. A rank-1 penalty has a one-dimensional range
/// space, so `edf(ρ) = γ/(γ + e^ρ)` ranges over `(0, 1]` and reaches 1 only as
/// `λ → 0`: the test is false for every rank-1 term and the term is skipped. The
/// null-space half of a Marra–Wood double penalty IS rank-1, so the linear
/// direction of every smooth is unprotected — measured on penguins at
/// `edf ≈ 4.6e-5`, i.e. dead.
///
/// The proposed repair is a floor RELATIVE to what the term can attain: require
/// `edf ≥ f · edf_max` for some `f < 1`. The objection to it is that on
/// near-separable data the Fisher weights `p(1−p)` collapse and the REML surface
/// flattens regardless, so a relative floor might only move a wall rather than
/// restore data-driven selection. That objection deserves a number.
///
/// For rank 1 the crossing is CLOSED FORM, so no bisection or fitting is needed
/// to get it. From `γ/(γ + e^ρ) = f`:
///
/// ```text
/// ρ*(f) = ln γ + ln((1 − f)/f)
/// ```
///
/// which is finite for every `f ∈ (0, 1)` — so the relative floor is well posed
/// exactly where the absolute floor is not. This prints `ρ*(f)` against the
/// uniform ceiling the term currently keeps, so the size of the tightening is a
/// number rather than an intuition. It does NOT settle whether the tightening
/// helps the fit; that needs penguins. It settles that there is something real
/// to tighten TO.
///
/// zz_measure discipline: prints only, asserts nothing.
#[test]
fn zz_measure_2608_relative_floor_wall_movement() {
    let ceiling = EFFECTIVE_DF_CEILING;
    eprintln!(
        "#2608: rank-1 term, absolute floor {EFFECTIVE_DF_FLOOR} is UNREACHABLE \
         (edf_max = 1.0 exactly, attained only as lambda -> 0) => term exempt, \
         keeps uniform ceiling {ceiling}"
    );
    eprintln!("#2608: gamma   f      rho*(f)=ln(gamma)+ln((1-f)/f)   tightening vs ceiling");
    for gamma in [1.0e-2_f64, 1.0, 1.0e2, 1.0e4] {
        for f in [0.25_f64, 0.5, 0.75, 0.9] {
            let rho_star = gamma.ln() + ((1.0 - f) / f).ln();
            // edf actually retained at that rho, as a check on the closed form.
            let retained = gamma / (gamma + rho_star.exp());
            eprintln!(
                "#2608: {gamma:>7.1e}  {f:>4.2}   {rho_star:>12.6}                  \
                 {:>8.4} nats   (edf there = {retained:.6})",
                ceiling - rho_star,
            );
        }
    }
    eprintln!(
        "#2608: reading -- rho*(f) is FINITE for every f in (0,1), so the relative \
         floor is well posed exactly where the absolute one is not. Whether the \
         tightening improves the FIT is a penguins question, not this one."
    );
}
