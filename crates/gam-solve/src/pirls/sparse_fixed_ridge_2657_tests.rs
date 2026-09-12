use super::certify_sparse_penalized_hessian;
use crate::estimate::EstimationError;
use faer::sparse::{SparseColMat, Triplet};

#[test]
fn sparse_indefiniteness_refuses_without_selecting_a_rho_dependent_ridge_2657() {
    let indefinite = SparseColMat::try_new_from_triplets(
        2usize,
        2usize,
        &[
            Triplet::new(0usize, 0usize, 1.0),
            Triplet::new(0usize, 1usize, 2.0),
            Triplet::new(1usize, 1usize, 1.0),
        ],
    )
    .expect("construct the 2x2 sparse indefiniteness witness");

    assert!(
        matches!(
            certify_sparse_penalized_hessian(indefinite),
            Err(EstimationError::HessianNotPositiveDefinite { .. })
        ),
        "the assembled matrix has eigenvalue 1 - 2 < 0 and must be refused, not shifted by a \
         ridge chosen from H(rho) (#2657, #2901 V22)"
    );
}
