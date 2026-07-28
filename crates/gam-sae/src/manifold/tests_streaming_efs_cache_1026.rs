//! Streaming EFS-cache capability regression, split out of `tests.rs` to keep
//! that tracked file under the #780 10k-line gate.

use super::*;
use super::tests::small_two_atom_periodic_term;
use crate::assignment::{AssignmentMode, SaeAssignment};
use ndarray::{Array2, Array3};

/// #2509 causal certificate pin. `ΔC` has no ββ channel, so a β-only model
/// (K=1 Softmax and zero latent axes) is structurally exact for every state.
/// Any row-local logit or coordinate support must refuse categorically; no
/// fitted-state tolerance or sampled correction is evidence for exactness.
#[test]
fn streaming_exact_a_certificate_is_structural_2509() {
    let (local_t, _, _) = small_two_atom_periodic_term();
    assert!(matches!(
        local_t.certify_streaming_majorizer_is_exact_observed_information(),
        Err(SaeCriterionError::ExactObservedInformationUnavailable {
            route: "streaming"
        })
    ));

    let n = 3usize;
    let atom = SaeManifoldAtom::new_with_provided_function_gram(
        "beta-only",
        SaeAtomBasisKind::Linear,
        0,
        Array2::<f64>::ones((n, 1)),
        Array3::<f64>::zeros((n, 1, 0)),
        Array2::<f64>::ones((1, 1)),
        Array2::<f64>::eye(1),
    )
    .expect("zero-axis beta-only atom");
    let assignment = SaeAssignment::from_blocks_with_mode(
        Array2::<f64>::zeros((n, 1)),
        vec![Array2::<f64>::zeros((n, 0))],
        AssignmentMode::softmax(1.0),
    )
    .expect("K=1 Softmax has no free logit");
    let beta_only =
        SaeManifoldTerm::new(vec![atom], assignment).expect("beta-only term is well formed");
    assert_eq!(beta_only.assignment.row_block_dim(), 0);
    beta_only
        .certify_streaming_majorizer_is_exact_observed_information()
        .expect("an empty local block proves A == B algebraically");
}

/// #2509: a B-factor cache is not an exact-A EFS drop-in merely because both
/// operators could be evaluated at the same converged inner state. This fixture
/// has a non-empty local `t` block, so `ΔC` can be live and the cache-returning
/// streaming entry must issue the same dedicated capability refusal as the
/// scalar entry, before fitting or constructing a misleading cache.
#[test]
fn streaming_cache_refuses_unrepresented_exact_observed_information_2509() {
    let (mut streaming, target, rho) = small_two_atom_periodic_term();
    let error = streaming
        .penalized_quasi_laplace_criterion_streaming_exact_with_cache(
            target.view(),
            &rho,
            None,
            2,
            0.25,
            1.0e-4,
            1.0e-4,
        )
        .expect_err("an unrepresented exact-A cache must never escape as a B cache");
    assert!(matches!(
        error,
        SaeCriterionError::ExactObservedInformationUnavailable {
            route: "streaming"
        }
    ));
}
