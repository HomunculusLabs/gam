//! Verify that the per-atom decoder identifiability audit fires inside
//! `SaeManifoldTerm::run_joint_fit_arrow_schur`.
//!
//! # What is tested
//!
//! Two scenarios:
//!
//! 1. **Well-posed atoms** — two decoder atoms whose per-atom weighted designs
//!    `D_k = diag(a_·k)·Φ_k` are each full column rank.  The audit passes
//!    cleanly and `run_joint_fit_arrow_schur` returns `Ok`.
//!
//! 2. **Rank-0 atom** — one atom whose basis evaluations are identically zero,
//!    so its weighted design `D_k` is rank 0.  The Arrow-Schur Newton system
//!    for that decoder block is singular, and the pre-fit audit surfaces this
//!    as an identifiability error before any Newton step.
//!
//! # Why this is the correct check
//!
//! The SAE decoder Hessian for atom `k` is `H_data = G_k ⊗ I_p` with
//! `G_k = D_kᵀ D_k`, so decoder identifiability is fully determined by the
//! per-atom `(n, M_k)` design `D_k` — the `p`-fold output replication carries
//! no extra structural information.  The audit therefore runs the pivoted-QR
//! rank check directly on each `D_k`, never materialising the (mis-specified)
//! `(n·p, M_k·p)` channel-replicated block that previously broadcast-panicked
//! when routed through the cross-block flat audit.
//!
//! # How the atoms are built
//!
//! Every atom is built the way production builds one: `Φ_k` and its Jacobian
//! are the basis evaluator's output at the atom's coordinates, and the evaluator
//! stays installed. The inner Newton step majorizes the data term's residual
//! curvature with the basis second jets, so the joint fit refuses an atom that
//! carries hand-supplied arrays and no evaluator. The live atoms carry the
//! degree-2 monomial patch `{1, t, t²}`; a rank-0 atom carries the zero
//! function, whose jets are exactly zero.

use gam::terms::latent::LatentManifold;
use gam::terms::sae::manifold::{
    AssignmentMode, EuclideanPatchEvaluator, SaeAssignment, SaeAtomBasisKind, SaeBasisEvaluator,
    SaeManifoldAtom, SaeManifoldRho, SaeManifoldTerm,
};
use ndarray::{Array1, Array2, Array3, Array4, Array5, ArrayView2};
use std::sync::Arc;

const N: usize = 20;
const M: usize = 3;
const P: usize = 2;
const LATENT_DIM: usize = 1;

/// The zero function on the latent: `M` identically zero basis columns with
/// exactly zero first, second and third jets. An atom on this basis has a rank-0
/// weighted design at every coordinate.
#[derive(Debug)]
struct ZeroBasis;

impl SaeBasisEvaluator for ZeroBasis {
    fn evaluate(&self, coords: ArrayView2<'_, f64>) -> Result<(Array2<f64>, Array3<f64>), String> {
        let (n, d) = coords.dim();
        Ok((Array2::<f64>::zeros((n, M)), Array3::<f64>::zeros((n, M, d))))
    }

    fn second_jet_dyn(&self, coords: ArrayView2<'_, f64>) -> Option<Result<Array4<f64>, String>> {
        let (n, d) = coords.dim();
        Some(Ok(Array4::<f64>::zeros((n, M, d, d))))
    }

    fn third_jet_dyn(
        &self,
        coords: ArrayView2<'_, f64>,
    ) -> Result<gam::terms::sae::manifold::SaeBasisThirdJetCapability, String> {
        let (n, d) = coords.dim();
        Ok(gam::terms::sae::manifold::SaeBasisThirdJetCapability::Analytic(
            Array5::<f64>::zeros((n, M, d, d, d)),
        ))
    }
}

/// Build an atom the way production does: `Φ` and its Jacobian are the
/// evaluator's output at the atom's coordinates, and the evaluator stays
/// installed for the second jets the inner Newton step reads.
fn make_atom(
    name: &str,
    evaluator: Arc<dyn SaeBasisEvaluator>,
    coords: &Array2<f64>,
) -> SaeManifoldAtom {
    let (phi, jet) = evaluator.evaluate(coords.view()).unwrap();
    let m = phi.ncols();
    let mut b = Array2::<f64>::zeros((m, P));
    // Give each atom a distinct, non-zero decoder so the Jacobian columns
    // are non-trivial. The atoms differ only in their basis evaluations.
    for mm in 0..m {
        for pp in 0..P {
            b[[mm, pp]] = (mm as f64 + 1.0) + (pp as f64) * 0.1;
        }
    }
    let penalty = Array2::<f64>::eye(m);
    SaeManifoldAtom::new_with_provided_function_gram(
        name,
        SaeAtomBasisKind::EuclideanPatch,
        LATENT_DIM,
        phi,
        jet,
        b,
        penalty,
    )
    .unwrap()
    .with_basis_evaluator(evaluator)
}

/// Build uniform-weight assignment (softmax logits = 0) over the atoms' coordinates.
fn make_assignment(coords: Vec<Array2<f64>>) -> SaeAssignment {
    let k_atoms = coords.len();
    let logits = Array2::<f64>::zeros((N, k_atoms));
    SaeAssignment::from_blocks_with_mode_and_manifolds(
        logits,
        coords,
        vec![LatentManifold::Euclidean; k_atoms],
        AssignmentMode::softmax(1.0),
    )
    .unwrap()
}

fn make_rho(k_atoms: usize) -> SaeManifoldRho {
    let log_ard = (0..k_atoms)
        .map(|_| Array1::<f64>::zeros(LATENT_DIM))
        .collect();
    SaeManifoldRho::new(-2.0_f64.ln(), -2.0_f64.ln(), log_ard)
}

/// Distinct coordinates for the degree-2 monomial patch: atom 0 sits at
/// `t = (i + 1)/N`, atom 1 at `t = 1 − (i + 1)/N`. `{1, t, t²}` is full column
/// rank over these N = 20 distinct points, so the per-atom audit passes for both.
fn distinct_coords(atom_idx: usize) -> Array2<f64> {
    Array2::from_shape_fn((N, LATENT_DIM), |(i, _)| {
        let t = (i as f64 + 1.0) / (N as f64);
        if atom_idx == 0 { t } else { 1.0 - t }
    })
}

/// The degree-2 monomial patch `{1, t, t²}` (`M = 3` columns).
fn quadratic_patch() -> Arc<dyn SaeBasisEvaluator> {
    Arc::new(EuclideanPatchEvaluator::new(LATENT_DIM, 2).unwrap())
}

/// A trivial target matrix (all zeros). The audit runs before any Newton step,
/// so the target only affects the loss, not the audit outcome.
fn zero_target() -> Array2<f64> {
    Array2::<f64>::zeros((N, P))
}

#[test]
fn run_joint_fit_passes_with_full_rank_atoms() {
    // Two atoms with distinct, full-column-rank weighted designs — the per-atom
    // audit passes cleanly and the fit returns Ok.
    let coords_a = distinct_coords(0);
    let coords_b = distinct_coords(1);
    let atom0 = make_atom("atom_a", quadratic_patch(), &coords_a);
    let atom1 = make_atom("atom_b", quadratic_patch(), &coords_b);
    let assignment = make_assignment(vec![coords_a, coords_b]);
    let mut term = SaeManifoldTerm::new(vec![atom0, atom1], assignment).unwrap();
    let mut rho = make_rho(2);
    let target = zero_target();

    let result = term.run_joint_fit_arrow_schur(
        target.view(),
        &mut rho,
        None,
        1,   // max_iter = 1 — we only need to confirm the audit fires
        1.0, // step_size
        1.0e-3,
        1.0e-3,
    );
    assert!(
        result.is_ok(),
        "run_joint_fit_arrow_schur must succeed with full-rank atoms; got: {:?}",
        result,
    );
}

#[test]
fn run_joint_fit_parks_single_rank_zero_atom_among_live() {
    // #1026/#1522 — one atom whose basis evaluations are identically zero (its
    // weighted design D_k is rank 0) ALONGSIDE a full-rank atom. In an
    // over-complete dictionary a surplus atom whose assignment weights all
    // vanish SHOULD die gracefully: the Arrow-Schur ridge parks its singular
    // block (β_k → 0) exactly as it regularises any rank-deficient block. The
    // pre-fit audit must therefore NOT reject the fit — a single dead atom among
    // identifiable ones is the intended over-complete outcome, not a fatal error.
    let coords_ok = distinct_coords(0);
    let coords_degenerate = distinct_coords(1);
    let atom0 = make_atom("atom_ok", quadratic_patch(), &coords_ok);
    let atom1 = make_atom("atom_degenerate", Arc::new(ZeroBasis), &coords_degenerate);
    let assignment = make_assignment(vec![coords_ok, coords_degenerate]);
    let mut term = SaeManifoldTerm::new(vec![atom0, atom1], assignment).unwrap();
    let mut rho = make_rho(2);
    let target = zero_target();

    let result =
        term.run_joint_fit_arrow_schur(target.view(), &mut rho, None, 1, 1.0, 1.0e-3, 1.0e-3);
    assert!(
        result.is_ok(),
        "run_joint_fit_arrow_schur must PARK a single rank-0 atom among live atoms (graceful \
         death in an over-complete dictionary), not reject the fit; got: {result:?}",
    );
}

#[test]
fn run_joint_fit_fails_when_all_atoms_rank_zero() {
    // The genuine pathology the audit must still catch loudly: EVERY atom has a
    // rank-0 weighted design, so the whole dictionary is unidentifiable and the
    // joint Newton system has no ridge-recoverable signal anywhere. This must
    // error before any Newton step, mentioning identifiability.
    let coords_a = distinct_coords(0);
    let coords_b = distinct_coords(1);
    let atom0 = make_atom("atom_dead_a", Arc::new(ZeroBasis), &coords_a);
    let atom1 = make_atom("atom_dead_b", Arc::new(ZeroBasis), &coords_b);
    let assignment = make_assignment(vec![coords_a, coords_b]);
    let mut term = SaeManifoldTerm::new(vec![atom0, atom1], assignment).unwrap();
    let mut rho = make_rho(2);
    let target = zero_target();

    let result =
        term.run_joint_fit_arrow_schur(target.view(), &mut rho, None, 1, 1.0, 1.0e-3, 1.0e-3);
    assert!(
        result.is_err(),
        "run_joint_fit_arrow_schur must fail when ALL atoms have rank-0 weighted design; got Ok(…)",
    );
    let msg = result.unwrap_err();
    assert!(
        msg.contains("identifiability"),
        "error message must mention identifiability; got: {msg}",
    );
}
