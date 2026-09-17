//! A latent-coordinate design derivative at a center collision `t = c`.
//!
//! The joint REML search moves latent rows freely, and data-derived centers
//! start on latent rows, so `t = c` is reachable. The design row's gradient
//! `∂φ(‖t − c‖)/∂t = φ'(r)·(t − c)/r` has norm `|φ'(r)|`:
//! - A kernel that is C¹ at the origin (`φ'(0⁺) = 0`) has gradient 0 there, and
//!   the operator must return that limit, even when `q = φ'/r` diverges. That
//!   is the case for `r² log r`, the two-dimensional order-2 Duchon block.
//!   Before, that block was classified by its opposite-signed ψ-scaling
//!   exponent and required a finite Hessian scalar, so the operator returned
//!   `DegenerateAtCollision` and `LatentCoordDerivativeOp::materialize_local`
//!   turned it into a panic ("fit_table panicked inside Rust boundary" in
//!   examples/standard_fit_latent_demo.py and mechanism_sparsity_demo.py).
//! - A kernel with a cone point (`φ'(0⁺) ≠ 0`, Matérn ν = 1/2) has no gradient
//!   there, and the operator must refuse when it is built, not panic inside an
//!   outer evaluation.

use gam_terms::basis::{
    BasisError, DuchonNullspaceOrder, LatentCoordDesignDerivative, LocalDesignJacobianProvider,
    MaternNu,
};
use gam_terms::latent::{LatentCoordValues, LatentIdMode, LatentManifold};
use gam_terms::{IsotropicScale, OriginalUnits};
use ndarray::{Array1, Array2, array};
use std::sync::Arc;

fn centers() -> Array2<f64> {
    array![
        [-1.20, -0.40],
        [-0.65, 0.85],
        [-0.10, -0.95],
        [0.35, 0.30],
        [0.90, 1.15],
        [1.45, -0.20],
        [0.15, 1.50],
        [1.25, 0.55],
    ]
}

/// Center 3, the one latent row 0 collides with.
fn collided_center() -> [f64; 2] {
    let c = centers();
    [c[[3, 0]], c[[3, 1]]]
}

/// Latent rows: row 0 at `point`, the others off every center.
fn latent_with_row_at(point: [f64; 2]) -> Arc<LatentCoordValues> {
    let data = array![[point[0], point[1]], [-0.85, 0.10], [0.60, -1.30], [1.00, 0.00]];
    Arc::new(LatentCoordValues::from_matrix_with_manifold(
        data.view(),
        LatentIdMode::None,
        LatentManifold::Euclidean,
    ))
}

fn matern(nu: MaternNu, point: [f64; 2]) -> Result<LatentCoordDesignDerivative, BasisError> {
    LatentCoordDesignDerivative::new_matern(
        latent_with_row_at(point),
        Arc::new(centers()),
        IsotropicScale::ONE,
        OriginalUnits::new(1.3),
        nu,
        false,
        None,
    )
}

/// Pure Duchon with a linear null space (p = 2), s = 0, in d = 2: `φ = c·r² log r`.
fn duchon_order_two(point: [f64; 2]) -> LatentCoordDesignDerivative {
    LatentCoordDesignDerivative::new_duchon(
        latent_with_row_at(point),
        Arc::new(centers()),
        IsotropicScale::ONE,
        None,
        0.0,
        DuchonNullspaceOrder::Linear,
        None,
        None,
    )
    .expect("pure Duchon order-2 latent derivative")
}

fn row0_axis0(op: &LatentCoordDesignDerivative) -> Array1<f64> {
    op.local_design_jacobian_row(0, 0)
        .expect("design-row gradient of latent row 0 along axis 0")
}

#[test]
fn a_cone_point_kernel_is_refused_when_the_latent_operator_is_built() {
    let refused = matern(MaternNu::Half, collided_center());
    assert!(
        matches!(refused, Err(BasisError::DegenerateAtCollision { .. })),
        "Matérn ν = 1/2 has φ'(0⁺) ≠ 0 and must be refused: {:?}",
        refused.as_ref().err()
    );
    // Positive control: the same inputs with a kernel that is C¹ at the origin build.
    assert!(matern(MaternNu::ThreeHalves, collided_center()).is_ok());
}

#[test]
fn a_c1_matern_kernel_returns_the_zero_limit_at_a_center_collision() {
    let at_center = collided_center();
    let collided = row0_axis0(&matern(MaternNu::ThreeHalves, at_center).expect("Matérn 3/2"));
    assert_eq!(collided[3], 0.0, "{collided:?}");
    // The collision value is the limit: the collided column's gradient shrinks
    // like r as the row approaches the center (φ'(r) ≈ −3r/ℓ²).
    let near: Vec<f64> = [1e-2, 1e-4, 1e-6]
        .iter()
        .map(|eps| {
            let op = matern(MaternNu::ThreeHalves, [at_center[0] + eps, at_center[1]])
                .expect("Matérn 3/2");
            row0_axis0(&op)[3].abs()
        })
        .collect();
    assert!(
        near[1] < 0.05 * near[0] && near[2] < 0.05 * near[1],
        "the collided column's gradient must vanish linearly toward the center: {near:?}"
    );
}

#[test]
fn the_log_polyharmonic_duchon_block_returns_its_zero_limit_instead_of_refusing() {
    let at_center = collided_center();
    let collided = row0_axis0(&duchon_order_two(at_center));
    assert!(collided.iter().all(|value| value.is_finite()), "{collided:?}");
    // q = φ'/r = c(2 log r + 1) diverges, but φ'(r) → 0, so the projected
    // gradient row converges to its collision value like ε·|log ε|.
    let gaps: Vec<f64> = [1e-2, 1e-4, 1e-6]
        .iter()
        .map(|eps| {
            let near = row0_axis0(&duchon_order_two([at_center[0] + eps, at_center[1]]));
            (&near - &collided).iter().map(|value| value * value).sum::<f64>().sqrt()
        })
        .collect();
    assert!(
        gaps[1] < 0.05 * gaps[0] && gaps[2] < 0.05 * gaps[1],
        "the projected gradient row must converge to its collision value: {gaps:?}"
    );
}
