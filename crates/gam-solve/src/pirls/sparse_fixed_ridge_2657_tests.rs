use super::{FIXED_STABILIZATION_RIDGE, ensure_sparse_positive_definite_with_fixed_ridge};
use crate::estimate::EstimationError;
use faer::sparse::{SparseColMat, Triplet};

#[test]
fn sparse_indefiniteness_refuses_without_selecting_a_rho_dependent_ridge_2657() {
    let mut requested_ridges = Vec::new();
    let result = ensure_sparse_positive_definite_with_fixed_ridge(|ridge| {
        requested_ridges.push(ridge);
        Ok(SparseColMat::try_new_from_triplets(
            2usize,
            2usize,
            &[
                Triplet::new(0usize, 0usize, 1.0 + ridge),
                Triplet::new(0usize, 1usize, 2.0),
                Triplet::new(1usize, 1usize, 1.0 + ridge),
            ],
        )
        .expect("construct the 2x2 sparse indefiniteness witness"))
    });

    assert!(
        matches!(
            result,
            Err(EstimationError::HessianNotPositiveDefinite { .. })
        ),
        "the fixed-ridged matrix has eigenvalue 1 + δ - 2 < 0 and must be refused"
    );
    assert_eq!(
        requested_ridges,
        [FIXED_STABILIZATION_RIDGE],
        "the sparse selector must never derive a second ridge from H(rho)"
    );
}
