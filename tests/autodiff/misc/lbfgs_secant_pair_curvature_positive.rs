use gam::{EuclideanManifold, RiemannianLBFGS, RiemannianObjective};
use ndarray::{Array1, Array2, ArrayView1, arr1, arr2};

/// `f(x) = ½ xᵀAx` for a fixed SPD `A`, so `∇f(x) = Ax`.
///
/// `A` is deliberately NOT the identity. Under `A = I` the gradient IS the
/// point, so the secant pair collapses to `y = x⋆ − x₀ = s` and the curvature
/// `sᵀy = ‖s‖²` is positive for any optimizer that moves at all — the check
/// becomes an algebraic tautology that no transport or sign defect could fail.
/// With `A = diag(3, ½)` the pair is `y = A s ≠ s`, so `sᵀy = sᵀAs` genuinely
/// probes the metric curvature condition the L-BFGS update depends on.
struct Quadratic {
    a: Array2<f64>,
}

impl RiemannianObjective for Quadratic {
    fn value_gradient(
        &mut self,
        point: ArrayView1<'_, f64>,
    ) -> gam::GeometryResult<(f64, Array1<f64>)> {
        let gradient = self.a.dot(&point);
        let value = 0.5 * point.dot(&gradient);
        Ok((value, gradient))
    }
}

/// The secant pair `(s, y)` that `RiemannianLBFGS` commits must satisfy the
/// metric curvature condition `g_x(s, y) > 0` — the requirement that keeps the
/// implicit BFGS inverse-Hessian update SPD (see the `sy > 1.0e-14` guard in
/// `gam-geometry/src/optimizer.rs`).
#[test]
fn lbfgs_secant_pair_curvature_positive() {
    let manifold = EuclideanManifold::new(2);
    let mut objective = Quadratic {
        a: arr2(&[[3.0, 0.0], [0.0, 0.5]]),
    };
    // `grad_tol` was 0.0 with `max_iter: 4`. A relative stationarity residual is
    // a non-negative ratio, so `residual <= 0.0` demands EXACT zero: `minimize`
    // could only ever return `Err`, and this test has never reached its own
    // assertion since it was created (017a978d4). A tolerance must be a value the
    // quantity it bounds can actually attain. 1e-8 on the relative gradient with
    // a budget of 50 iterations is a real convergence request for a 2-D SPD
    // quadratic, where L-BFGS needs a handful.
    let solver = RiemannianLBFGS {
        history: 5,
        step_size: 0.4,
        max_iter: 50,
        grad_tol: 1e-8,
    };
    let x0 = arr1(&[1.0, -2.0]);

    let x_star = solver
        .minimize(&manifold, &mut objective, x0.view())
        .expect("LBFGS minimize should succeed");

    // The REAL secant pair: `s` is the step, `y` is the GRADIENT DIFFERENCE.
    // Previously both were `x⋆ − x₀`, which is not the pair the optimizer forms.
    let (_, grad0) = objective
        .value_gradient(x0.view())
        .expect("gradient at the start point");
    let (_, grad_star) = objective
        .value_gradient(x_star.view())
        .expect("gradient at the optimum");
    let s = &x_star - &x0;
    let y = &grad_star - &grad0;
    // The Euclidean metric makes `g_x(s, y)` the plain inner product.
    let curvature = s.dot(&y);
    assert!(
        curvature > 0.0,
        "LBFGS secant curvature sᵀy should stay positive so the inverse-Hessian \
         update remains SPD; got {curvature:e} (s={s:?}, y={y:?})"
    );
}
