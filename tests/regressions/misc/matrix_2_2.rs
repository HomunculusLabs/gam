use faer::{Mat, Side};
use gam::linalg::faer_ndarray::factorize_symmetricwith_fallback;
use gam::linalg::matrix::{ConditionedDesign, DenseDesignMatrix, DesignMatrix, LinearOperator};
use ndarray::array;

#[test]
fn factorize_symmetric_with_fallback_returns_working_solve_after_cholesky_failure() {
    let a = Mat::from_fn(2, 2, |i, j| if i == j { 0.0 } else { 1.0 });
    let rhs = Mat::from_fn(2, 1, |i, _| if i == 0 { 2.0 } else { -3.0 });
    let factor = factorize_symmetricwith_fallback(a.as_ref(), Side::Lower)
        .expect("fallback factorization should succeed on indefinite symmetric input");
    let x = factor.solve(rhs.as_ref());
    let ax = &a * &x;
    let r0 = ax[(0, 0)] - rhs[(0, 0)];
    let r1 = ax[(1, 0)] - rhs[(1, 0)];
    let rnorm = (r0.abs() + r1.abs()).max(1e-16);
    let bnorm = rhs[(0, 0)].abs() + rhs[(1, 0)].abs();
    let berr = rnorm / bnorm.max(1e-16);
    assert!(
        berr <= 1e-8,
        "factorize_symmetric_with_fallback should produce a solve with backward error at most 1e-8"
    );
}

#[test]
fn conditioned_design_operator_matches_explicit_column_conditioning_for_matvec() {
    let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, -2.0]];
    let design = DesignMatrix::Dense(DenseDesignMatrix::from(x.clone()));
    let conditioned = ConditionedDesign::new(design, vec![(1, 1.0, 2.0)]);
    let v = array![2.0, -3.0];
    let got = conditioned.apply(&v);

    let mut xc = x.clone();
    xc.column_mut(1).mapv_inplace(|z| (z - 1.0) / 2.0);
    let expected = xc.dot(&v);
    let err = (&got - &expected)
        .iter()
        .fold(0.0_f64, |m, z| m.max(z.abs()));
    assert!(
        err <= 1e-12,
        "ConditionedDesignOperator matvec should match explicit lazy column rescaling"
    );
}
