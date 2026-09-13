//! Regression test for https://github.com/SauersML/gam/issues/163 and #175.
//!
//! `ManifoldSAE.predict(X_subset)` (#163) and `ManifoldSAE.reconstruct(X_oos)`
//! (#175) both raise `ValueError: SaeManifoldTerm::run_joint_fit_arrow_schur:
//! arrow-Schur: per-row H_tt^(i) Cholesky failed: row i H_tt was non-PD at
//! ridge_t=0.000001; cholesky error: non-PD pivot ... at index 0` when the
//! per-row Newton block ends up rank-deficient at the caller's nominal
//! `ridge_t = 1e-6`. The PCA-seeded latent coordinates on subset / new data
//! routinely hit this degeneracy: a row whose assignment mass is near zero
//! has effectively no data contribution to `H_tt`, leaving the per-row block
//! at the floor set by the ridge alone, and a tiny rounding-noise pivot
//! breaks the Cholesky.
//!
//! The principled fix routes every Newton-step solve through
//! `arrow_schur::newton_step::solve_with_lm_escalation_inner`, which geometrically grows a
//! proximal ridge on top of `ridge_ext_coord` / `ridge_beta` until the
//! per-row / Schur factor succeeds — the same Ceres-style LM damping the
//! training-time `run_joint_fit_arrow_schur` driver already used inline.
//! Both `run_joint_fit_arrow_schur` (the multi-iteration training/predict
//! driver) and the single-shot Newton entry that the basis-refresh refinement
//! uses now self-heal, so the OOS predict /
//! reconstruct call path can no longer surface the per-row factor failure.
//!
//! No `let _`, no `#[allow]`, no `#[ignore]`, no env vars.

use ndarray::{Array1, Array2, array};
use std::sync::Arc;

use gam::terms::{
    latent::LatentManifold, sae::manifold::AssignmentMode,
    sae::manifold::PeriodicHarmonicEvaluator, sae::manifold::SaeAssignment,
    sae::manifold::SaeAtomBasisKind, sae::manifold::SaeBasisEvaluator,
    sae::manifold::SaeManifoldAtom, sae::manifold::SaeManifoldRho, sae::manifold::SaeManifoldTerm,
};

fn degenerate_oos_term() -> (SaeManifoldTerm, SaeManifoldRho, Array2<f64>) {
    // Mirrors the failing reproducer shape: a single periodic atom whose
    // per-row data Hessian is identically zero (assignments are zero),
    // so `H_tt + ridge_t·I` at ridge_t=1e-6 is the only thing keeping the
    // Cholesky finite. With ridge_t this small the standard Cholesky finds
    // a tiny negative pivot from rounding — exactly the regime issues
    // #163 and #175 describe.
    //
    // The atom is built as the OOS entry builds one: the first-harmonic periodic
    // evaluator supplies `Φ` and its Jacobian at the coordinates and stays
    // installed, because the inner Newton step reads its analytic second jets.
    let coords = array![[0.1], [0.4], [0.7]];
    let evaluator = PeriodicHarmonicEvaluator::new(3).expect("first-harmonic periodic evaluator");
    let (phi, jet) = evaluator
        .evaluate(coords.view())
        .expect("evaluate the periodic basis");
    let atom = SaeManifoldAtom::new_with_provided_function_gram(
        "periodic",
        SaeAtomBasisKind::Periodic,
        1,
        phi,
        jet,
        array![[0.05], [-0.05], [0.05]],
        Array2::<f64>::zeros((3, 3)),
    )
    .expect("build periodic atom")
    .with_basis_evaluator(Arc::new(evaluator));
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        Array2::<f64>::zeros((3, 1)),
        vec![coords],
        vec![LatentManifold::Circle { period: 1.0 }],
        AssignmentMode::softmax(0.7),
    )
    .expect("build assignment");
    let term = SaeManifoldTerm::new(vec![atom], assignment).expect("build term");
    let target = array![[0.20], [-0.10], [0.45]];
    let rho = SaeManifoldRho::new(0.0, -20.0, vec![Array1::<f64>::zeros(1)]);
    (term, rho, target)
}

#[test]
fn run_joint_fit_arrow_schur_recovers_on_oos_predict_path() {
    // This is the path `ManifoldSAE.predict` / `.reconstruct` reaches via
    // `sae_manifold_predict_oos` → `sae_manifold_fit_inner` →
    // `run_joint_fit_arrow_schur`. Before the LM-escalation refactor the
    // driver surfaced the per-row Cholesky failure to the Python caller as
    // a `ValueError`. After the fix the same call must Ok at the nominal
    // `ridge_ext_coord = ridge_beta = 1e-6` that the FFI defaults to.
    let (mut term, mut rho, target) = degenerate_oos_term();
    let result =
        term.run_joint_fit_arrow_schur(target.view(), &mut rho, None, 1, 1.0, 1.0e-6, 1.0e-6);
    assert!(
        result.is_ok(),
        "OOS predict driver must recover from degenerate H_tt via LM ridge escalation; got: {result:?}",
    );
}

