// Auto-derived `ResourcePolicy::for_problem` selection.
//
// The policy no longer carries a row/column cliff: shape alone never flips the
// mode (that anti-pattern was replaced by the process-wide `MemoryGovernor`
// byte ledger, which decides dense-vs-chunked-vs-matrix-free from the checked
// predicted live bytes of each operation). Strict mode is reached only through
// a *structural* signal — the marginal-slope large-scale path, which is
// operator-only by construction — or the explicit `analytic_operator_required`
// preset. These integration tests pin that contract.

use gam::resource::{DerivativeStorageMode, ProblemHints, ResourcePolicy};

#[test]
fn small_problem_selects_default_library() {
    let p = ResourcePolicy::for_problem(ProblemHints::default());
    assert!(matches!(
        p.derivative_storage_mode,
        DerivativeStorageMode::MaterializeIfSmall
    ));
}

#[test]
fn shape_never_flips_the_mode() {
    // The old `STRICT_POLICY_NROWS_THRESHOLD` cliff is gone: a nominally
    // large-scale problem with no structural hint stays permissive, and the
    // per-operation governor — not the row count — makes the dense decision.
    let default_hints = ResourcePolicy::for_problem(ProblemHints::default());
    let large_shape_hints = ResourcePolicy::for_problem(ProblemHints {
        marginal_slope_large_scale_active: false,
    });
    assert!(matches!(
        default_hints.derivative_storage_mode,
        DerivativeStorageMode::MaterializeIfSmall
    ));
    assert!(matches!(
        large_shape_hints.derivative_storage_mode,
        DerivativeStorageMode::MaterializeIfSmall
    ));
}

#[test]
fn marginal_slope_hint_forces_strict() {
    let p = ResourcePolicy::for_problem(ProblemHints {
        marginal_slope_large_scale_active: true,
    });
    assert!(matches!(
        p.derivative_storage_mode,
        DerivativeStorageMode::AnalyticOperatorRequired
    ));
}
