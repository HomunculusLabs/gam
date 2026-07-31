//! #2568 caller-side outer stationarity requirement.
//!
//! Split out of `run_plan_tests.rs` when that file reached 10,063 lines and
//! tripped the 10,000-line ban gate (#780), which aborts the ROOT crate's
//! build and therefore every root-crate target. Declared with `#[path]` from
//! `run_plan.rs` exactly as its sibling is, so `use super::*` resolves
//! identically; no test changed subject, assertion or name.

use super::*;

// ─── #2568 caller-side outer stationarity requirement ──────────────

/// The saturating shape this issue was filed against: an objective whose
/// declared scale widens the engine's band by six orders, so a point nowhere
/// near stationary clears it.
fn wide_band_config_2568(required: Option<f64>) -> OuterConfig {
    OuterConfig {
        tolerance: 1e-5,
        objective_scale: Some(1.0e6),
        required_projected_gradient_norm: required,
        ..Default::default()
    }
}

#[test]
fn a_caller_requirement_tightens_the_solver_band_2568() {
    // The load-bearing half. If the requirement did not reach the SOLVER's
    // band, the outer loop would still stop at the sealed bound and the
    // requirement could only ever be a post-hoc rejection.
    let engine = outer_gradient_tolerance(&wide_band_config_2568(None)).abs;
    assert!(
        engine > 1.0e-3,
        "fixture must reproduce a saturated engine band, got {engine:.6e}"
    );
    let tightened = outer_gradient_tolerance(&wide_band_config_2568(Some(1.0e-3))).abs;
    assert_eq!(
        tightened, 1.0e-3,
        "the caller's requirement must become the band the optimizer is told to \
         reach (engine band was {engine:.6e})"
    );
}

#[test]
fn a_caller_requirement_survives_the_score_relative_widening_2568() {
    // The widening is what produced `bound = 1.000e0`, so a requirement applied
    // before it would be defeated by precisely the case it exists for.
    let cost = 1.0e6;
    let engine = outer_stationarity_band_at(&wide_band_config_2568(None), cost);
    assert!(
        engine > 1.0e-3,
        "fixture must reproduce a widened certificate band, got {engine:.6e}"
    );
    let capped = outer_stationarity_band_at(&wide_band_config_2568(Some(1.0e-3)), cost);
    assert_eq!(
        capped, 1.0e-3,
        "the certificate band must be capped at the caller's requirement, not \
         widened past it (engine band was {engine:.6e})"
    );
}

#[test]
fn a_looser_caller_requirement_never_weakens_the_engine_band_2568() {
    // A caller asking for less accuracy than the engine already guarantees is
    // not asking for anything. If this inverted, the knob would be a way to
    // launder a fit past a standard the engine derived from the criterion.
    let cfg_tight_engine = OuterConfig {
        tolerance: 1.0e-9,
        ..Default::default()
    };
    let engine = outer_gradient_tolerance(&cfg_tight_engine).abs;
    let with_loose = outer_gradient_tolerance(&OuterConfig {
        required_projected_gradient_norm: Some(1.0),
        ..cfg_tight_engine.clone()
    })
    .abs;
    assert_eq!(
        with_loose, engine,
        "a requirement of 1.0 must leave a {engine:.6e} engine band in force"
    );
    let cost = 1.0e3;
    let engine_band = outer_stationarity_band_at(&cfg_tight_engine, cost);
    let loose_band = outer_stationarity_band_at(
        &OuterConfig {
            required_projected_gradient_norm: Some(1.0e9),
            ..cfg_tight_engine
        },
        cost,
    );
    assert_eq!(
        loose_band, engine_band,
        "nor may it widen the certificate band"
    );
}

#[test]
fn an_absent_requirement_reproduces_the_engine_exactly_2568() {
    // The compatibility claim, stated as a test rather than asserted in a doc
    // comment: `None` must be byte-for-byte today's behaviour on both bands.
    for scale in [None, Some(1.0), Some(1.0e6)] {
        for cost in [0.0, 1.0, -3.7e2, 1.0e6, f64::INFINITY] {
            let cfg = OuterConfig {
                tolerance: 1e-5,
                objective_scale: scale,
                required_projected_gradient_norm: None,
                ..Default::default()
            };
            let bare = OuterConfig {
                tolerance: 1e-5,
                objective_scale: scale,
                ..Default::default()
            };
            assert_eq!(
                outer_gradient_tolerance(&cfg).abs,
                outer_gradient_tolerance(&bare).abs,
                "solver band moved with scale={scale:?}"
            );
            assert_eq!(
                outer_stationarity_band_at(&cfg, cost),
                outer_stationarity_band_at(&bare, cost),
                "certificate band moved with scale={scale:?} cost={cost}"
            );
        }
    }
}

#[test]
fn an_unsatisfiable_requirement_is_rejected_at_the_builder_2568() {
    // `0.0` and `NaN` are not stricter standards, they are unsatisfiable ones.
    // Honouring them would grind the outer loop to `max_iter` and then refuse
    // every fit -- a performance cliff wearing an accuracy requirement's clothes.
    for bad in [0.0, -1.0e-3, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let cfg = OuterProblem::new(1)
            .with_required_projected_gradient_norm(Some(bad))
            .config();
        assert_eq!(
            cfg.required_projected_gradient_norm, None,
            "requirement {bad} must be rejected, not clamped"
        );
    }
    let cfg = OuterProblem::new(1)
        .with_required_projected_gradient_norm(Some(1.0e-7))
        .config();
    assert_eq!(cfg.required_projected_gradient_norm, Some(1.0e-7));
}

#[test]
fn the_caller_requirement_rung_names_itself_2568() {
    // A refusal that reported `solver-band` while the caller's number decided it
    // would be unauditable in exactly the way #2465 is about.
    assert_eq!(
        StationarityBoundSource::CallerRequirement.label(),
        "caller-requirement"
    );
    assert!(
        !StationarityBoundSource::CallerRequirement.is_derived_standard(),
        "the caller's requirement is not the engine's derived resolvability \
         standard and must not claim to be"
    );
}
