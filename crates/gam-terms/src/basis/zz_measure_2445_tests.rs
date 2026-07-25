//! Measurement probe for #2445 / #2444 — is the `DoublePenaltyNullspace` FD
//! residue exactly the motion of the rebuild's rank-test frame?
//!
//! Not a gate. It prints, and it asserts only the ONE quantitative prediction
//! the root-cause model makes, so a wrong model fails here instead of surviving
//! into the fix.
//!
//! The model: `rebuild_metric_consistent_ridge` builds `R = N (NᵀR_cN) Nᵀ` with
//! `N = null(S_c)` from a numerical rank test. In this fixture `dim N = 1`, so
//! after Frobenius normalization the shipped block is exactly `u uᵀ` — the
//! scalar metric `NᵀR_cN` cancels. Hence
//!
//!   `‖d/dψ (u uᵀ)‖_F = √2 ‖u′‖`   (u′ ⊥ u since ‖u‖ = 1)
//!
//! and the observed `fd_norm = 2.3142e-6` predicts `‖u′‖ = 1.6364e-6`, i.e. the
//! angle between `u(+ε)` and `u(−ε)` at `ε = 1e-4` must be `2ε‖u′‖ = 3.27e-10`.
//! That number is not tunable: it is fixed by the failing test's own output.

use ndarray::{Array2, ArrayView2, s};

use super::*;

/// The `_frozen` / `_linear` fixture from
/// `test_duchon_log_kappa_derivative_matchesfd_dim1_power1_*`, chart frozen.
fn frozen_fixture() -> (Array2<f64>, DuchonBasisSpec) {
    let n = 80usize;
    let mut data = Array2::<f64>::zeros((n, 1));
    for i in 0..n {
        data[[i, 0]] = i as f64 / (n as f64 - 1.0);
    }
    let mut spec = DuchonBasisSpec {
        radial_reparam: None,
        periodic: None,
        center_strategy: CenterStrategy::FarthestPoint { num_centers: 8 },
        length_scale: Some(1.0),
        power: 1.0,
        nullspace_order: DuchonNullspaceOrder::Linear,
        identifiability: SpatialIdentifiability::default(),
        aniso_log_scales: None,
        operator_penalties: DuchonOperatorPenaltySpec::default(),
        boundary: OneDimensionalBoundary::Open,
    };
    let base = build_duchon_basis(data.view(), &spec).expect("cold base build");
    if let BasisMetadata::Duchon {
        centers,
        identifiability_transform,
        radial_reparam,
        ..
    } = &base.metadata
    {
        spec.center_strategy = CenterStrategy::UserProvided(centers.clone());
        spec.radial_reparam = radial_reparam.clone();
        spec.identifiability = match identifiability_transform {
            Some(t) => SpatialIdentifiability::FrozenTransform {
                transform: t.clone(),
            },
            None => SpatialIdentifiability::None,
        };
    } else {
        panic!("expected Duchon metadata");
    }
    (data, spec)
}

/// The `(kernel_transform, outer_identifiability)` pair the forward penalty
/// builder consumes, replicated exactly from
/// `build_duchon_native_penalty_psi_derivatives`.
fn frozen_chart_transforms(
    centers: ArrayView2<'_, f64>,
    spec: &DuchonBasisSpec,
) -> (Array2<f64>, Option<Array2<f64>>, usize, usize) {
    let mut workspace = BasisWorkspace::default();
    let order = duchon_effective_nullspace_order(centers, spec.nullspace_order);
    let mut z = kernel_constraint_nullspace(centers, order, &mut workspace.cache)
        .expect("kernel constraint nullspace");
    if let Some(v) = spec.radial_reparam.as_ref() {
        z = z.dot(v);
    }
    let kernel_cols = z.ncols();
    let poly_cols = polynomial_block_from_order(centers, order).ncols();
    let transform = match &spec.identifiability {
        SpatialIdentifiability::FrozenTransform { transform } => Some(transform.clone()),
        _ => None,
    };
    (z, transform, kernel_cols, poly_cols)
}

/// `sin θ_max` between two orthonormal frames, computed as
/// `‖(I − BBᵀ)A‖₂` rather than `acos(σ_min(AᵀB))`.
///
/// The cosine form cannot resolve this measurement at all: the angle the model
/// predicts is `3.3e-10`, whose cosine is `1 − 5.4e-20`, which rounds to
/// exactly `1.0` in double precision and reports an angle of zero. The
/// projection residual loses no digits.
fn frame_sin_angle(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    assert_eq!(a.ncols(), b.ncols(), "frames must have equal dimension");
    let residual = a - &b.dot(&b.t().dot(a));
    let (_, singular, _) = gam_linalg::faer_ndarray::FaerSvd::svd(&residual, false, false)
        .expect("svd of projection residual");
    singular.iter().copied().fold(0.0_f64, f64::max)
}

#[test]
fn zz_measure_2445_rank_test_frame_moves_with_psi() {
    let (_, spec) = frozen_fixture();
    let centers = match &spec.center_strategy {
        CenterStrategy::UserProvided(c) => c.clone(),
        _ => unreachable!("fixture freezes the centers"),
    };
    let (z, transform, kernel_cols, poly_cols) = frozen_chart_transforms(centers.view(), &spec);
    eprintln!(
        "[2445] chart: kernel_cols={kernel_cols} poly_cols={poly_cols} \
         transform={:?}",
        transform.as_ref().map(|t| t.dim())
    );

    let eps = 1e-4_f64;
    let mut frames: Vec<Array2<f64>> = Vec::new();
    let mut ridges: Vec<Array2<f64>> = Vec::new();
    for (label, psi) in [("minus", -eps), ("zero", 0.0), ("plus", eps)] {
        let length_scale = 1.0 / psi.exp();
        let candidates = duchon_native_penalty_candidates(
            centers.view(),
            Some(length_scale),
            spec.power,
            duchon_effective_nullspace_order(centers.view(), spec.nullspace_order),
            None,
            &z,
            transform.as_ref(),
        )
        .expect("native penalty candidates");
        let primary = candidates
            .iter()
            .find(|c| matches!(c.source, PenaltySource::Primary))
            .expect("primary");
        let ridge = candidates
            .iter()
            .find(|c| matches!(c.source, PenaltySource::DoublePenaltyNullspace))
            .expect("trend ridge");
        let physical_primary = primary
            .matrix
            .scaled(primary.normalization_scale, "physical primary")
            .expect("scale primary");
        let null = constructive_nullspace_basis(&physical_primary)
            .expect("nullspace basis")
            .unwrap_or_else(|| Array2::<f64>::zeros((physical_primary.nrows(), 0)));
        eprintln!(
            "[2445] psi={psi:+.1e} ({label}): rank-test null dim={} ridge_norm={:.6e}",
            null.ncols(),
            ridge.matrix.dense().iter().map(|v| v * v).sum::<f64>().sqrt(),
        );
        frames.push(null);
        ridges.push(ridge.matrix.dense().to_owned());
    }

    let angle = frame_sin_angle(&frames[0], &frames[2]);
    let fd = (&ridges[2] - &ridges[0]) / (2.0 * eps);
    let fd_norm = fd.iter().map(|v| v * v).sum::<f64>().sqrt();
    eprintln!(
        "[2445] principal angle between null(S_c)(-eps) and null(S_c)(+eps) = {angle:.6e}\n\
         [2445] fd_norm(rebuilt trend ridge) = {fd_norm:.6e}\n\
         [2445] predicted angle from fd_norm (dim-1 frame) = {:.6e}",
        2.0 * eps * fd_norm / std::f64::consts::SQRT_2
    );

    assert_eq!(
        frames[1].ncols(),
        1,
        "fixture is expected to have a 1-D rank-test null space"
    );
    // With `dim N = 1` the normalization makes the shipped block exactly `uuᵀ`.
    // Confirm that before reading anything into the frame motion.
    for (index, (frame, ridge)) in frames.iter().zip(ridges.iter()).enumerate() {
        let outer = frame.dot(&frame.t());
        let gap = (&outer - ridge).iter().map(|v| v * v).sum::<f64>().sqrt();
        eprintln!("[2445] ‖ridge - u uᵀ‖_F at sample {index} = {gap:.6e}");
    }
    let predicted = 2.0 * eps * fd_norm / std::f64::consts::SQRT_2;
    assert!(
        (angle - predicted).abs() <= 0.05 * predicted.max(1e-14),
        "the FD residue must BE the frame motion: angle={angle:.6e} predicted={predicted:.6e}"
    );

    // The structural transport: `N_c = {γ : Tγ ∈ span(polynomial block)}`,
    // i.e. the null space of the kernel-block rows of the chart. Frozen chart
    // and a κ-free polynomial block, so this cannot move with ψ.
    if let Some(t) = transform.as_ref() {
        let kernel_rows_t = t.slice(s![..kernel_cols, ..]).t().to_owned();
        let (structural, rank) = gam_linalg::faer_ndarray::rrqr_nullspace_basis(&kernel_rows_t, 1.0)
            .expect("structural transport nullspace");
        eprintln!(
            "[2445] structural transported frame: rank(T_kernel)={rank} dim={} (expected {})",
            structural.ncols(),
            poly_cols - 1
        );
        let overlap = frame_sin_angle(&structural, &frames[1]);
        eprintln!(
            "[2445] angle(structural frame, rank-test frame at psi=0) = {overlap:.6e}"
        );
    }
}
