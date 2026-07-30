//! Unit gates for the #2629 floor classifier.
//!
//! Every case here is a SYNTHETIC ladder built from a known law, so the
//! classifier is tested against ground truth rather than against a fixture's
//! behaviour. The two families it must separate are:
//!
//! * `g(ρ) = w·a·tanh(a·ρ̃) + c·e^{−ρ}` — the criterion carries the barrier;
//! * `g(ρ) = c·e^{−ρ}` — it does not.
//!
//! The measured constants from the shipped fixtures are used as the `c` values
//! (`+87.5` from #2450's Matérn/Gaussian ladder, `−22.8` from #2629's
//! SAS/binomial one) so the synthetic ladders sit exactly where the real ones do.

use super::*;

/// #2450's measured face constant, on the Matérn/Gaussian fixture.
const FACE_C_POSITIVE: f64 = 87.512;
/// #2629's measured face constant, on the SAS/binomial fixture. Opposite sign:
/// the face may be approached from either side, and a classifier fed `|g|`
/// could not tell that from a floor.
const FACE_C_NEGATIVE: f64 = -22.82;

fn ladder_from(law: impl Fn(f64) -> f64) -> Vec<GuardLadderRung> {
    SATURATED_RHO_LADDER
        .iter()
        .map(|&rho| GuardLadderRung {
            rho,
            rho_gradient: law(rho),
        })
        .collect()
}

fn carrying_ladder(c: f64, anchor: f64) -> Vec<GuardLadderRung> {
    ladder_from(|rho| soft_rho_guard_emission_at(rho, anchor) + c * (-rho).exp())
}

fn bare_face_ladder(c: f64) -> Vec<GuardLadderRung> {
    ladder_from(|rho| c * (-rho).exp())
}

#[test]
fn a_criterion_that_adds_the_barrier_is_classified_as_carrying_it() {
    for c in [FACE_C_POSITIVE, FACE_C_NEGATIVE] {
        let verdict = classify_soft_rho_guard_floor(&carrying_ladder(c, 0.0), 0.0);
        let SoftRhoGuardFloor::Carried { face, guard } = &verdict else {
            panic!(
                "a ladder built as guard(rho) + {c}*e^-rho must classify as CARRIED, got {}",
                verdict.summary()
            );
        };
        assert!(
            (face.constant - c).abs() <= 1.0e-6 * c.abs(),
            "the recovered face constant must be the one the ladder was built \
             from: expected {c}, got {}",
            face.constant
        );
        assert!(
            face.spread <= 1.0e-12,
            "an exactly-generated ladder has an exactly constant pencil; got \
             spread {:.3e}",
            face.spread
        );
        // The floor really is the dominant term at these rungs — that is what
        // makes the discrimination four orders wide rather than marginal.
        let deepest_guard = *guard.last().expect("four rungs");
        let deepest_tail = (c * (-SATURATED_RHO_LADDER[3]).exp()).abs();
        assert!(
            deepest_tail <= 1.0e-3 * deepest_guard,
            "the fixture constants must put the face tail ORDERS below the floor \
             at rho=30 (tail {deepest_tail:.3e} vs guard {deepest_guard:.3e})"
        );
    }
}

#[test]
fn a_criterion_with_no_barrier_is_classified_as_a_bare_decaying_face() {
    for c in [FACE_C_POSITIVE, FACE_C_NEGATIVE] {
        let verdict = classify_soft_rho_guard_floor(&bare_face_ladder(c), 0.0);
        let SoftRhoGuardFloor::AbsentDecayingFace { face, .. } = &verdict else {
            panic!(
                "a ladder built as {c}*e^-rho with no floor must classify as \
                 ABSENT, got {}",
                verdict.summary()
            );
        };
        assert!(
            (face.constant - c).abs() <= 1.0e-9 * c.abs(),
            "expected face constant {c}, got {}",
            face.constant
        );
    }
}

/// The case the magnitude branch exists for: a family whose face has decayed
/// into roundoff. Its raw pencil is `noise * e^30` — pure garbage — so the
/// shape test cannot answer, and only "the gradient is smaller than the floor"
/// remains. That statement is still a complete answer to the question asked.
#[test]
fn a_face_decayed_into_roundoff_is_absent_by_magnitude_not_indeterminate() {
    // Alternating signs at the 1e-14 level: exactly what a converged inner
    // solve's leftover looks like, and exactly what defeats a pencil test.
    let noise = [1.7e-14, -9.2e-15, 4.1e-14, -2.3e-14];
    let ladder: Vec<GuardLadderRung> = SATURATED_RHO_LADDER
        .iter()
        .zip(noise)
        .map(|(&rho, g)| GuardLadderRung {
            rho,
            rho_gradient: g,
        })
        .collect();
    let verdict = classify_soft_rho_guard_floor(&ladder, 0.0);
    let SoftRhoGuardFloor::AbsentBelowTheFloor {
        max_abs_gradient,
        min_guard,
    } = &verdict
    else {
        panic!(
            "a roundoff-level gradient is DEMONSTRABLY floor-free — the floor is \
             larger than the whole gradient. Got {}",
            verdict.summary()
        );
    };
    assert!(*max_abs_gradient < min_guard * ABSENCE_MAGNITUDE_FRACTION);
    assert!(verdict.is_absent());
}

/// The refusal that keeps the instrument honest: a criterion carrying a term
/// that is neither a floor nor a decaying face (here a configured `ρ/sd²`
/// Normal ρ-prior, the pre-#2450 default) must not be rounded into either
/// verdict.
#[test]
fn a_criterion_with_a_third_non_decaying_term_is_refused_not_rounded() {
    // `Normal { mean: 0, sd: 3 }` contributes `rho/9`: 2.333 at rho=21 up to
    // 3.333 at rho=30. Neither constant nor exponentially decaying.
    let ladder = ladder_from(|rho| rho / 9.0);
    let verdict = classify_soft_rho_guard_floor(&ladder, 0.0);
    let SoftRhoGuardFloor::Indeterminate { reason, .. } = &verdict else {
        panic!(
            "a linear-in-rho prior term is neither hypothesis and must be \
             REFUSED, got {}",
            verdict.summary()
        );
    };
    assert!(
        reason.contains("neither pencil"),
        "the refusal must name what failed, got: {reason}"
    );
    assert!(!verdict.is_absent() && !verdict.is_carried());
}

/// The weight anchor is a required argument because getting it wrong is
/// silent (#877/#2545). On a weighted state the barrier is evaluated at
/// `ρ̃ = ρ − log g(w)`, so a ladder built with an anchor and classified without
/// one must NOT come back as a clean `Carried` — the subtracted floor is a
/// different function and the residual is not a face.
#[test]
fn classifying_a_weighted_ladder_at_the_wrong_anchor_does_not_mint_a_face() {
    const ANCHOR: f64 = 4.0;
    let ladder = carrying_ladder(FACE_C_POSITIVE, ANCHOR);

    // With the right anchor: the face comes back exactly.
    let right = classify_soft_rho_guard_floor(&ladder, ANCHOR);
    let SoftRhoGuardFloor::Carried { face, .. } = &right else {
        panic!("at the correct anchor this is CARRIED, got {}", right.summary());
    };
    assert!((face.constant - FACE_C_POSITIVE).abs() <= 1.0e-6 * FACE_C_POSITIVE);

    // With the anchor dropped: the residual is `guard(ρ−4) − guard(ρ) + c·e^-ρ`,
    // a nonzero near-constant offset, so its pencil GROWS like `e^ρ` and the
    // hypothesis must fail rather than report a wrong face constant.
    let wrong = classify_soft_rho_guard_floor(&ladder, 0.0);
    assert!(
        !wrong.is_carried(),
        "dropping the weight anchor measures a different function; the \
         classifier must not report a clean face from it. Got {}",
        wrong.summary()
    );
}

/// Two rungs cannot show a pencil to be constant — any two points define a
/// constant through them. The instrument refuses rather than accepting a
/// tautology.
#[test]
fn a_ladder_too_short_to_show_constancy_is_refused() {
    let short = vec![
        GuardLadderRung {
            rho: 27.0,
            rho_gradient: soft_rho_guard_emission_at(27.0, 0.0),
        },
        GuardLadderRung {
            rho: 30.0,
            rho_gradient: soft_rho_guard_emission_at(30.0, 0.0),
        },
    ];
    let verdict = classify_soft_rho_guard_floor(&short, 0.0);
    assert!(
        matches!(verdict, SoftRhoGuardFloor::Indeterminate { .. }),
        "got {}",
        verdict.summary()
    );
}

/// A non-finite rung (a diverged probe) is refused, not silently propagated
/// into a NaN pencil that then compares false against every bound.
#[test]
fn a_non_finite_rung_is_refused() {
    let mut ladder = bare_face_ladder(FACE_C_POSITIVE);
    ladder[1].rho_gradient = f64::NAN;
    let verdict = classify_soft_rho_guard_floor(&ladder, 0.0);
    let SoftRhoGuardFloor::Indeterminate { reason, .. } = &verdict else {
        panic!("got {}", verdict.summary());
    };
    assert!(reason.contains("non-finite"), "got: {reason}");
}

/// The emission this module subtracts must be the criterion's own atom, not a
/// second closed form. Pin it against `RemlState::build_prior`'s route: the
/// same atom, the same constants, the same anchor.
#[test]
fn the_subtracted_emission_is_the_criterions_own_atom() {
    for anchor in [0.0, -2.5, 4.0] {
        for &rho in SATURATED_RHO_LADDER.iter() {
            let from_atom = SoftRhoGuardPriorAtom::evaluate_anchored(
                &Array1::from_elem(1, rho),
                RHO_SOFT_PRIOR_WEIGHT,
                RHO_SOFT_PRIOR_SHARPNESS,
                RHO_BOUND,
                anchor,
            )
            .gradient()[0];
            assert_eq!(
                soft_rho_guard_emission_at(rho, anchor),
                from_atom,
                "the classifier's subtrahend must be BIT-identical to the atom \
                 `build_prior` adds (rho={rho}, anchor={anchor})"
            );
        }
    }
    // And the saturation is real at this ladder: the deepest rung is within a
    // hair of `w*a`, which is why the floor cannot be mistaken for a tail.
    let saturated = RHO_SOFT_PRIOR_WEIGHT * (RHO_SOFT_PRIOR_SHARPNESS / RHO_BOUND);
    let deepest = soft_rho_guard_emission_at(RHO_BOUND, 0.0);
    assert!(
        (deepest - saturated).abs() <= 3.0e-3 * saturated,
        "at rho=RHO_BOUND the barrier must be at its saturation w*a={saturated:.6e}, \
         got {deepest:.6e}"
    );
}
