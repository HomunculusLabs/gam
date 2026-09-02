//! gam#979 — the hybrid Duchon design ψ-derivative under the kernel chart.
//!
//! In high dimension at a large spectral power the raw hybrid Duchon–Matérn
//! kernel underflows (its spectral normalization is ~1e-15 at `d = 16`,
//! `s = 9`), and the forward basis ships the kernel block multiplied by the
//! chart amplitude `α(ψ) = 1/max|K|` (`duchon_kernel_chart`). The design the
//! REML criterion is built on is therefore `α(ψ)·K(ψ)`, and its ψ-derivative
//! is `α(K_ψ + (ln α)_ψ K)`, not `K_ψ`. Before this gate existed, the
//! derivative operator formed `K_ψ` alone: at the large-scale benchmark's
//! `duchon(pc1..pc16, order=0, power=9, length_scale=1)` that is ~1e-15 of
//! the true derivative, the analytic outer gradient silently dropped every
//! κ-dependence that enters through the design, and the κ line search walked
//! uphill on every trial (the `gam fit --transformation-normal` timeout).
//!
//! The gate differences the FORWARD design the basis actually ships, through
//! its own frozen chart, against the operator's materialized first and second
//! ψ-derivatives. The low-dimensional control has `α = 1` and pins that the
//! chart is inert there; the 16-D fixture asserts `α ≠ 1` so it cannot pass
//! vacuously.

#![cfg(test)]

use ndarray::{Array2, ArrayView2};

use super::*;

/// A deterministic, non-degenerate cloud in `d` dimensions on the ±2 range of
/// a standardized coordinate (distinct irrational multipliers per axis so no
/// two axes alias).
fn standardized_cloud(n: usize, d: usize) -> Array2<f64> {
    let mut data = Array2::<f64>::zeros((n, d));
    for i in 0..n {
        for a in 0..d {
            let multiplier = ((a + 2) as f64 * 2.0 + 1.0).sqrt().fract();
            data[[i, a]] = 4.0 * ((i as f64 * multiplier + 0.37 * a as f64).fract() - 0.5);
        }
    }
    data
}

/// A hybrid Duchon spec at `(order, power)` with the chart frozen off one cold
/// build, so nothing but the length scale moves when ψ does — the same
/// discipline `zz_duchon_axis_psi_2735_tests` uses.
fn frozen_hybrid_fixture(
    d: usize,
    n: usize,
    centers: usize,
    order: DuchonNullspaceOrder,
    power: f64,
) -> (Array2<f64>, DuchonBasisSpec) {
    let data = standardized_cloud(n, d);
    let mut spec = DuchonBasisSpec {
        radial_reparam: None,
        periodic: None,
        center_strategy: CenterStrategy::FarthestPoint {
            num_centers: centers,
        },
        length_scale: Some(1.0),
        power,
        nullspace_order: order,
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

fn fixture_centers(spec: &DuchonBasisSpec) -> Array2<f64> {
    match &spec.center_strategy {
        CenterStrategy::UserProvided(c) => c.clone(),
        _ => unreachable!("fixture freezes the centers"),
    }
}

/// The frozen spec with the isotropic coordinate moved to `ψ`: `ℓ = e^{−ψ}`.
fn spec_at_psi(spec: &DuchonBasisSpec, psi: f64) -> DuchonBasisSpec {
    let mut out = spec.clone();
    out.length_scale = Some((-psi).exp());
    out
}

fn forward_design(data: ArrayView2<'_, f64>, spec: &DuchonBasisSpec) -> Array2<f64> {
    build_duchon_basis(data, spec)
        .expect("forward design at ψ")
        .design
        .to_dense()
}

fn analytic_first(data: ArrayView2<'_, f64>, spec: &DuchonBasisSpec) -> Array2<f64> {
    let bundle = build_duchon_basis_log_kappa_derivatives(data, spec).expect("ψ-derivative bundle");
    bundle
        .implicit_operator
        .as_ref()
        .expect("hybrid Duchon design derivatives are operator-backed")
        .materialize_first(0)
        .expect("materialize ∂X/∂ψ")
}

fn analytic_second(data: ArrayView2<'_, f64>, spec: &DuchonBasisSpec) -> Array2<f64> {
    let bundle = build_duchon_basis_log_kappa_derivatives(data, spec).expect("ψ-derivative bundle");
    bundle
        .implicit_operator
        .as_ref()
        .expect("hybrid Duchon design derivatives are operator-backed")
        .materialize_second_diag(0)
        .expect("materialize ∂²X/∂ψ²")
}

fn frobenius(m: &Array2<f64>) -> f64 {
    m.iter().map(|v| v * v).sum::<f64>().sqrt()
}

fn chart_amplification(data: ArrayView2<'_, f64>, spec: &DuchonBasisSpec) -> f64 {
    let centers = fixture_centers(spec);
    let order = duchon_effective_nullspace_order(centers.view(), spec.nullspace_order);
    let p_order = duchon_p_from_nullspace_order(order);
    let s_order = spec.power_as_usize();
    let length_scale = spec.length_scale.expect("hybrid fixture");
    let coeffs = duchon_partial_fraction_coeffs(p_order, s_order, 1.0 / length_scale);
    duchon_kernel_chart(
        centers.view(),
        Some(length_scale),
        p_order,
        s_order,
        data.ncols(),
        None,
        Some(&coeffs),
        None,
    )
    .amplification
}

/// Central differences of the SHIPPED design and of the analytic first
/// derivative along ψ, at two steps (ratio 2). The returned oracle is their
/// Richardson combination: both central differences have `O(h²)` truncation,
/// so `D(h/2) + (D(h/2) - D(h))/3` removes that leading error without weakening
/// the steps or acceptance bars. Raw gaps remain printed so a true formula
/// defect (which does not contract by four) stays visibly distinct from an
/// otherwise-correct derivative whose two-point oracle is truncation-limited.
fn chart_gaps(data: ArrayView2<'_, f64>, spec: &DuchonBasisSpec, label: &str) -> (f64, f64) {
    let first_an = analytic_first(data, spec);
    let second_an = analytic_second(data, spec);
    let mut first_differences = Vec::with_capacity(2);
    let mut second_differences = Vec::with_capacity(2);
    for &h in &[2.0e-3_f64, 1.0e-3] {
        let plus = spec_at_psi(spec, h);
        let minus = spec_at_psi(spec, -h);
        let first_fd = (forward_design(data, &plus) - forward_design(data, &minus)) / (2.0 * h);
        let first_gap = frobenius(&(&first_an - &first_fd)) / frobenius(&first_fd).max(1e-300);
        let second_fd = (analytic_first(data, &plus) - analytic_first(data, &minus)) / (2.0 * h);
        let second_gap =
            frobenius(&(&second_an - &second_fd)) / frobenius(&second_fd).max(1e-300);
        eprintln!(
            "[{label}] h={h:.1e} first |an|={:.6e} |fd|={:.6e} gap={first_gap:.3e}; \
             second |an|={:.6e} |fd|={:.6e} gap={second_gap:.3e}",
            frobenius(&first_an),
            frobenius(&first_fd),
            frobenius(&second_an),
            frobenius(&second_fd),
        );
        first_differences.push(first_fd);
        second_differences.push(second_fd);
    }
    let first_richardson =
        &first_differences[1] + (&first_differences[1] - &first_differences[0]) / 3.0;
    let second_richardson =
        &second_differences[1] + (&second_differences[1] - &second_differences[0]) / 3.0;
    let first_gap =
        frobenius(&(&first_an - &first_richardson)) / frobenius(&first_richardson).max(1e-300);
    let second_gap = frobenius(&(&second_an - &second_richardson))
        / frobenius(&second_richardson).max(1e-300);
    eprintln!(
        "[{label}] Richardson first gap={first_gap:.3e}; second gap={second_gap:.3e}"
    );
    (first_gap, second_gap)
}

/// The benchmark's chart: sixteen axes, constant-only null space, spectral
/// power 9. The chart MUST be amplified here (the kernel underflows), and the
/// operator's first and second ψ-derivatives must be those of the shipped,
/// amplified design.
#[test]
fn duchon_chart_design_psi_derivatives_match_the_shipped_design_16d_order0_power9() {
    let (data, spec) = frozen_hybrid_fixture(16, 120, 12, DuchonNullspaceOrder::Zero, 9.0);
    let amplification = chart_amplification(data.view(), &spec);
    assert!(
        amplification != 1.0,
        "the 16-D power-9 fixture must underflow so the chart is exercised; got α = {amplification}"
    );
    let (first_gap, second_gap) = chart_gaps(data.view(), &spec, "chart_16d_order0_power9");
    assert!(
        first_gap < 1e-5,
        "charted ∂X/∂ψ differs from the shipped design's central difference by {first_gap:.3e}"
    );
    assert!(
        second_gap < 1e-4,
        "charted ∂²X/∂ψ² differs from the central difference of ∂X/∂ψ by {second_gap:.3e}"
    );
}

/// The `Linear` null-space sibling at the same shape (a different `p`, the
/// same underflow).
#[test]
fn duchon_chart_design_psi_derivatives_match_the_shipped_design_16d_linear_power9() {
    // 24 centers: the `Linear` null space is d + 1 = 17 columns, and with fewer
    // centers `duchon_effective_nullspace_order` degrades the order to `Zero`
    // (the 12-center fixture above would silently re-run the order-0 case).
    let (data, spec) = frozen_hybrid_fixture(16, 120, 24, DuchonNullspaceOrder::Linear, 9.0);
    assert_eq!(
        duchon_effective_nullspace_order(fixture_centers(&spec).view(), spec.nullspace_order),
        DuchonNullspaceOrder::Linear,
        "the fixture must realize the Linear null space, not a degraded one"
    );
    let amplification = chart_amplification(data.view(), &spec);
    assert!(
        amplification != 1.0,
        "the 16-D power-9 fixture must underflow so the chart is exercised; got α = {amplification}"
    );
    let (first_gap, second_gap) = chart_gaps(data.view(), &spec, "chart_16d_linear_power9");
    assert!(first_gap < 1e-5, "charted ∂X/∂ψ gap {first_gap:.3e}");
    assert!(second_gap < 1e-4, "charted ∂²X/∂ψ² gap {second_gap:.3e}");
}

/// The three-axis sibling of the benchmark chart (the shape the CTN 3-D gate
/// runs). Whether it is amplified is a property of `(p, s, d, ℓ)` alone, so
/// the printed `α` here is the same one every 3-D order-0 power-9 fixture
/// sees; the gate does not assume either value, it only pins the derivative.
#[test]
fn duchon_chart_design_psi_derivatives_match_the_shipped_design_3d_order0_power9() {
    let (data, spec) = frozen_hybrid_fixture(3, 160, 10, DuchonNullspaceOrder::Zero, 9.0);
    let amplification = chart_amplification(data.view(), &spec);
    eprintln!("[chart_3d_order0_power9] alpha={amplification:.6e}");
    let (first_gap, second_gap) = chart_gaps(data.view(), &spec, "chart_3d_order0_power9");
    assert!(first_gap < 1e-5, "charted ∂X/∂ψ gap {first_gap:.3e}");
    assert!(second_gap < 1e-4, "charted ∂²X/∂ψ² gap {second_gap:.3e}");
}

/// The low-dimensional control: no underflow, identity chart, and the
/// derivative surface must be exactly what it was before the chart existed.
/// The shape is the one `zz_duchon_axis_psi_2735_tests` certifies (2-D,
/// `Linear`, power 2).
#[test]
fn duchon_chart_is_inert_where_the_kernel_does_not_underflow() {
    let (data, spec) = frozen_hybrid_fixture(2, 90, 10, DuchonNullspaceOrder::Linear, 2.0);
    let amplification = chart_amplification(data.view(), &spec);
    assert_eq!(amplification, 1.0, "a 2-D power-2 hybrid must not be amplified");
    let (first_gap, second_gap) = chart_gaps(data.view(), &spec, "chart_2d_linear_power2");
    assert!(first_gap < 1e-5, "∂X/∂ψ gap {first_gap:.3e}");
    assert!(second_gap < 1e-4, "∂²X/∂ψ² gap {second_gap:.3e}");
}

// ---------------------------------------------------------------------------
// The latent-coordinate Jacobian under the same chart.
//
// `LatentCoordDesignDerivative::new_duchon` supplies `∂X/∂t` for the joint
// `[rho, latent]` driver. The shipped design is `α·φ(||t/σ − c||)`, and `α`
// depends on the centers and the range only, so the coordinate Jacobian is
// the raw one times `α`. The ground truth is the production rebuild
// (`build_term_collection_design` through the frozen spec), central-differenced
// in one latent coordinate — the same discipline as the #2643 frame gate.
// ---------------------------------------------------------------------------

use crate::latent::{LatentCoordValues, LatentIdMode};
use crate::smooth::{
    ShapeConstraint, SmoothBasisSpec, SmoothTermSpec, TermCollectionSpec,
    build_term_collection_design, freeze_term_collection_from_design,
};
use ndarray::{Array1, s};

fn latent_duchon_collection(
    d: usize,
    centers: usize,
    order: DuchonNullspaceOrder,
    power: f64,
) -> TermCollectionSpec {
    TermCollectionSpec {
        linear_terms: vec![],
        random_effect_terms: vec![],
        smooth_terms: vec![SmoothTermSpec {
            frozen_parametric_residualization: None,
            name: "latent_duchon".to_string(),
            basis: SmoothBasisSpec::Duchon {
                feature_cols: (0..d).collect(),
                spec: DuchonBasisSpec {
                    radial_reparam: None,
                    periodic: None,
                    center_strategy: CenterStrategy::FarthestPoint {
                        num_centers: centers,
                    },
                    length_scale: Some(1.0),
                    power,
                    nullspace_order: order,
                    // Global parametric centering would couple every row's
                    // design to the differenced coordinate; the operator's
                    // contract is the term-local design.
                    identifiability: SpatialIdentifiability::None,
                    aniso_log_scales: None,
                    operator_penalties: DuchonOperatorPenaltySpec::default(),
                    boundary: OneDimensionalBoundary::Open,
                },
                input_scale: None,
            },
            shape: ShapeConstraint::None,
            joint_null_rotation: None,
        }],
    }
}

/// The term's design row at latent configuration `data`, through the FROZEN
/// spec (σ and centers held; only the coordinate moves).
fn latent_design_row(data: &Array2<f64>, spec: &TermCollectionSpec, row: usize) -> Array1<f64> {
    let built = build_term_collection_design(data.view(), spec).expect("design rebuild");
    let p_total = built.design.ncols();
    let smooth_start = p_total.saturating_sub(built.smooth.total_smooth_cols());
    let range = &built.smooth.terms[0].coeff_range;
    built
        .design
        .to_dense()
        .row(row)
        .slice(s![smooth_start + range.start..smooth_start + range.end])
        .to_owned()
}

/// Worst relative disagreement between `local_design_jacobian_row` and the
/// central difference of the production rebuild, over three rows and every
/// axis. Returns `(chart amplification, worst relative error)`.
fn latent_jacobian_worst_gap(
    d: usize,
    order: DuchonNullspaceOrder,
    power: f64,
    label: &str,
) -> (f64, f64) {
    // A raw cloud with σ ≠ 1 so the standardized and original frames differ
    // (the #2643 trap) and the chart is exercised at the benchmark's shape.
    let data = standardized_cloud(60, d) * 2.5;
    let fresh = latent_duchon_collection(d, 12, order, power);
    let built = build_term_collection_design(data.view(), &fresh).expect("fresh design");
    let frozen = freeze_term_collection_from_design(&fresh, &built).expect("freeze");
    let BasisMetadata::Duchon {
        centers,
        length_scale,
        power: meta_power,
        nullspace_order,
        identifiability_transform,
        input_scale,
        radial_reparam,
        ..
    } = &built.smooth.terms[0].metadata
    else {
        panic!("fixture must produce Duchon metadata");
    };
    let sigma = input_scale.reciprocal().recip();
    let coeffs = {
        let p_order = duchon_p_from_nullspace_order(duchon_effective_nullspace_order(
            centers.view(),
            *nullspace_order,
        ));
        let ell = input_scale
            .to_standardized_units(length_scale.expect("hybrid fixture"))
            .standardized_value();
        (p_order, ell, duchon_partial_fraction_coeffs(p_order, power as usize, 1.0 / ell))
    };
    let amplification = duchon_kernel_chart(
        centers.view(),
        Some(coeffs.1),
        coeffs.0,
        power as usize,
        d,
        None,
        Some(&coeffs.2),
        None,
    )
    .amplification;

    let latent = std::sync::Arc::new(LatentCoordValues::from_flat(
        Array1::from_iter(data.iter().copied()),
        data.nrows(),
        d,
        LatentIdMode::None,
    ));
    let derivative = LatentCoordDesignDerivative::new_duchon(
        latent,
        std::sync::Arc::new(centers.clone()),
        *input_scale,
        *length_scale,
        *meta_power,
        *nullspace_order,
        radial_reparam.as_ref(),
        identifiability_transform.clone(),
    )
    .expect("latent Duchon design derivative");

    let step = f64::EPSILON.cbrt() * sigma;
    let mut worst = 0.0_f64;
    let mut largest_analytic = 0.0_f64;
    for row in [0usize, 17, 41] {
        for axis in 0..d {
            let analytic = derivative
                .local_design_jacobian_row(row, axis)
                .expect("analytic local design jacobian");
            let mut plus = data.clone();
            plus[[row, axis]] += step;
            let mut minus = data.clone();
            minus[[row, axis]] -= step;
            let numeric = (latent_design_row(&plus, &frozen, row)
                - latent_design_row(&minus, &frozen, row))
                / (2.0 * step);
            assert_eq!(analytic.len(), numeric.len(), "Jacobian rows must span the same columns");
            let magnitude = analytic
                .iter()
                .chain(numeric.iter())
                .fold(0.0_f64, |acc, value| acc.max(value.abs()));
            largest_analytic = largest_analytic.max(
                analytic
                    .iter()
                    .fold(0.0_f64, |acc, value| acc.max(value.abs())),
            );
            let denominator = magnitude.max(1e-8);
            for (a, n) in analytic.iter().zip(numeric.iter()) {
                worst = worst.max((a - n).abs() / denominator);
            }
        }
    }
    eprintln!(
        "[{label}] sigma={sigma:.4} alpha={amplification:.6e} step={step:.3e} \
         max|analytic|={largest_analytic:.3e} worst_relative_error={worst:.3e}"
    );
    (amplification, worst)
}

/// The benchmark's chart again, now for the coordinate Jacobian: the chart
/// MUST be amplified, and `∂X/∂t` must be the derivative of the shipped
/// (amplified) design. Before the chart reached this operator the analytic
/// rows were ~`1/α` of the numeric ones.
#[test]
fn latent_jacobian_matches_the_shipped_design_16d_order0_power9() {
    let (amplification, worst) =
        latent_jacobian_worst_gap(16, DuchonNullspaceOrder::Zero, 9.0, "latent_16d_order0_power9");
    assert!(
        amplification != 1.0,
        "the 16-D power-9 fixture must underflow so the chart is exercised; got α = {amplification}"
    );
    assert!(worst < 1e-5, "latent Jacobian differs from the rebuild by {worst:.3e} (relative)");
}

/// The low-dimensional control: identity chart, the Jacobian unchanged.
#[test]
fn latent_jacobian_chart_is_inert_where_the_kernel_does_not_underflow() {
    let (amplification, worst) =
        latent_jacobian_worst_gap(2, DuchonNullspaceOrder::Linear, 2.0, "latent_2d_linear_power2");
    assert_eq!(amplification, 1.0, "a 2-D power-2 hybrid must not be amplified");
    assert!(worst < 1e-5, "latent Jacobian differs from the rebuild by {worst:.3e} (relative)");
}

// ---------------------------------------------------------------------------
// The operator penalties (mass, tension) under the same chart.
//
// `duchon_operator_penalty_candidates` is the forward; its collocation
// quadratures now carry the chart amplitude `α` like the design does, and
// `build_duchon_operator_penalty_psi_derivatives` mirrors it. The gate
// differences the forward's NORMALIZED penalties along ψ against the analytic
// normalized first jets, per penalty source, at the benchmark shape (α ≫ 1)
// and at the 3-D sibling (α = 1).
// ---------------------------------------------------------------------------

fn fixture_collocation_points(data: ArrayView2<'_, f64>, spec: &DuchonBasisSpec) -> Array2<f64> {
    match build_duchon_basis(data, spec).expect("frozen build").metadata {
        BasisMetadata::Duchon {
            operator_collocation_points: Some(points),
            ..
        } => points,
        _ => panic!("hybrid Duchon with operator penalties must realize collocation points"),
    }
}

fn forward_operator_penalties(
    collocation: &Array2<f64>,
    centers: &Array2<f64>,
    spec: &DuchonBasisSpec,
) -> Vec<(String, Array2<f64>)> {
    duchon_operator_penalty_candidates(
        collocation.view(),
        centers.view(),
        &spec.operator_penalties,
        spec.length_scale,
        spec.power,
        spec.nullspace_order,
        false,
        None,
        spec.radial_reparam.as_ref(),
        &mut BasisWorkspace::default(),
    )
    .expect("forward operator penalties")
    .into_iter()
    .map(|candidate| (format!("{:?}", candidate.source), candidate.matrix.dense().clone()))
    .collect()
}

/// The mass penalty rebuilt from first principles beside the worker: amplified
/// kernel values at (collocation, center) pairs through the frozen chart
/// `Z·V`, the polynomial block, column centering, the Gram, and the
/// normalization — with `∂φ/∂ψ = δ φ + r φ_r` from the same radial jets.
/// Returns `(S̃, ∂S̃/∂ψ)`.
fn mass_reconstruction(
    collocation: &Array2<f64>,
    centers: &Array2<f64>,
    spec: &DuchonBasisSpec,
) -> (Array2<f64>, Array2<f64>) {
    let order = duchon_effective_nullspace_order(centers.view(), spec.nullspace_order);
    let p_order = duchon_p_from_nullspace_order(order);
    let s_order = spec.power_as_usize();
    let ell = spec.length_scale.expect("hybrid fixture");
    let d = centers.ncols();
    let coeffs = duchon_partial_fraction_coeffs(p_order, s_order, 1.0 / ell);
    let amp = duchon_kernel_amplification(
        centers.view(),
        Some(ell),
        p_order,
        s_order,
        d,
        None,
        Some(&coeffs),
        None,
    );
    let mut workspace = BasisWorkspace::default();
    let z = duchon_frozen_radial_chart(
        kernel_constraint_nullspace(centers.view(), order, &mut workspace.cache)
            .expect("side-condition null space"),
        spec,
        "mass reconstruction",
    )
    .expect("frozen radial chart");
    let delta = duchon_scaling_exponent(p_order, s_order, d);
    let (m, k) = (collocation.nrows(), centers.nrows());
    let mut raw = Array2::<f64>::zeros((m, k));
    let mut raw_psi = Array2::<f64>::zeros((m, k));
    for i in 0..m {
        for j in 0..k {
            let r = (0..d)
                .map(|a| (collocation[[i, a]] - centers[[j, a]]).powi(2))
                .sum::<f64>()
                .sqrt();
            let jets = duchon_radial_jets(r, ell, p_order, s_order, d, &coeffs).expect("jets");
            raw[[i, j]] = amp * jets.phi;
            raw_psi[[i, j]] = amp * (delta * jets.phi + r * jets.phi_r);
        }
    }
    let poly = polynomial_block_from_order(collocation.view(), order);
    let kernel_cols = z.ncols();
    let total = kernel_cols + poly.ncols();
    let mut d0 = Array2::<f64>::zeros((m, total));
    let mut d0_psi = Array2::<f64>::zeros((m, total));
    d0.slice_mut(s![.., ..kernel_cols]).assign(&raw.dot(&z));
    d0.slice_mut(s![.., kernel_cols..]).assign(&poly);
    d0_psi.slice_mut(s![.., ..kernel_cols]).assign(&raw_psi.dot(&z));
    let zeros = Array2::<f64>::zeros((m, total));
    let (s0, s0_psi, _) = centered_operator_gram_and_psi_derivatives(&d0, &d0_psi, &zeros);
    let (s_norm, s_norm_psi, _, _) =
        normalize_penaltywith_psi_derivatives(&s0, &s0_psi, &Array2::<f64>::zeros(s0.raw_dim()));
    (s_norm, s_norm_psi)
}

/// Per source: `(source, |fd|, best relative gap over two steps)`.
fn operator_penalty_gaps(
    data: ArrayView2<'_, f64>,
    spec: &DuchonBasisSpec,
    label: &str,
) -> Vec<(String, f64, f64)> {
    let collocation = fixture_collocation_points(data, spec);
    let centers = fixture_centers(spec);
    let (sources, firsts, _) = build_duchon_operator_penalty_psi_derivatives(
        collocation.view(),
        centers.view(),
        spec,
        None,
        &mut BasisWorkspace::default(),
    )
    .expect("operator penalty ψ-jets");
    assert!(!sources.is_empty(), "{label}: the fixture must emit operator penalties");
    // Diagnostic (printed, not asserted): where does the mass jet disagree —
    // in the value the two sides build, or in the jet they assemble from it?
    {
        let forward_base = forward_operator_penalties(&collocation, &centers, spec);
        if let (Some((_, s_fwd)), Some(worker_idx)) = (
            forward_base.iter().find(|(n, _)| n == "OperatorMass"),
            sources.iter().position(|s| format!("{s:?}") == "OperatorMass"),
        ) {
            let (s_mine, s_mine_psi) = mass_reconstruction(&collocation, &centers, spec);
            let value_gap = frobenius(&(s_fwd - &s_mine)) / frobenius(s_fwd).max(1e-300);
            // Split the value gap: the builder's own D0 through the same centered
            // Gram + normalization, against the candidate and against the rebuild.
            let ops = build_duchon_collocation_operator_matriceswithworkspace(
                centers.view(),
                collocation.view(),
                None,
                spec.length_scale,
                spec.power,
                spec.nullspace_order,
                None,
                None,
                1,
                spec.radial_reparam.as_ref().map(|v| v.view()),
                &mut BasisWorkspace::default(),
            )
            .expect("forward collocation blocks");
            let (s_builder, _) = normalize_penalty(&symmetrize_penalty(&centered_design_gram(&ops.d0)));
            eprintln!(
                "[{label}] MASS-RECON builder D0 {}x{} amp={:.3e}; gap(candidate vs builder-gram)={:.3e} \
                 gap(rebuilt vs builder-gram)={:.3e} |cand|={:.6e} |builder|={:.6e} |rebuilt|={:.6e}",
                ops.d0.nrows(),
                ops.d0.ncols(),
                ops.kernel_amplification,
                frobenius(&(s_fwd - &s_builder)) / frobenius(s_fwd).max(1e-300),
                frobenius(&(&s_mine - &s_builder)) / frobenius(&s_builder).max(1e-300),
                frobenius(s_fwd),
                frobenius(&s_builder),
                frobenius(&s_mine)
            );
            let jet_gap = frobenius(&(&firsts[worker_idx] - &s_mine_psi))
                / frobenius(&s_mine_psi).max(1e-300);
            eprintln!(
                "[{label}] MASS-RECON value gap(forward vs rebuilt)={value_gap:.3e} \
                 jet gap(worker vs rebuilt)={jet_gap:.3e} |rebuilt jet|={:.6e}",
                frobenius(&s_mine_psi)
            );
        }
    }
    let mut out = Vec::new();
    for (source, analytic) in sources.iter().zip(firsts.iter()) {
        let name = format!("{source:?}");
        let mut best_gap = f64::INFINITY;
        let mut fd_norm = 0.0;
        for &h in &[2.0e-3_f64, 1.0e-3] {
            let plus = forward_operator_penalties(&collocation, &centers, &spec_at_psi(spec, h));
            let minus = forward_operator_penalties(&collocation, &centers, &spec_at_psi(spec, -h));
            let find = |list: &[(String, Array2<f64>)]| {
                list.iter()
                    .find(|(candidate, _)| *candidate == name)
                    .map(|(_, matrix)| matrix.clone())
                    .unwrap_or_else(|| panic!("{label}: forward emits no {name} penalty"))
            };
            let fd = (find(&plus) - find(&minus)) / (2.0 * h);
            let gap = frobenius(&(analytic - &fd)) / frobenius(&fd).max(1e-300);
            eprintln!(
                "[{label}] {name} h={h:.1e} |an|={:.6e} |fd|={:.6e} gap={gap:.3e}",
                frobenius(analytic),
                frobenius(&fd)
            );
            if gap < best_gap {
                best_gap = gap;
                fd_norm = frobenius(&fd);
            }
        }
        out.push((name, fd_norm, best_gap));
    }
    out
}

fn assert_operator_penalty_gaps(gaps: &[(String, f64, f64)], label: &str) {
    for (name, fd_norm, gap) in gaps {
        assert!(
            *fd_norm > 1e-6,
            "{label}: {name} does not move with ψ in this fixture (|fd| = {fd_norm:.3e}), so the gate is vacuous"
        );
        assert!(
            *gap < 1e-4,
            "{label}: analytic ∂S̃/∂ψ of {name} differs from the forward's central difference by {gap:.3e}"
        );
    }
}

/// The benchmark's chart: mass and tension jets must be those of the shipped
/// (amplified, normalized) penalties. Before this gate they were exactly zero.
#[test]
fn duchon_operator_penalty_psi_jets_match_the_forward_16d_order0_power9() {
    let (data, spec) = frozen_hybrid_fixture(16, 120, 24, DuchonNullspaceOrder::Zero, 9.0);
    assert!(chart_amplification(data.view(), &spec) != 1.0, "the fixture must be amplified");
    let gaps = operator_penalty_gaps(data.view(), &spec, "opers_16d_order0_power9");
    assert_operator_penalty_gaps(&gaps, "opers_16d_order0_power9");
}

#[test]
fn duchon_operator_penalty_psi_jets_match_the_forward_16d_linear_power9() {
    let (data, spec) = frozen_hybrid_fixture(16, 120, 24, DuchonNullspaceOrder::Linear, 9.0);
    assert!(chart_amplification(data.view(), &spec) != 1.0, "the fixture must be amplified");
    let gaps = operator_penalty_gaps(data.view(), &spec, "opers_16d_linear_power9");
    assert_operator_penalty_gaps(&gaps, "opers_16d_linear_power9");
}

/// The un-amplified sibling: the same jets with `α = 1`, so a gap here is a
/// formula gap and not a scale one.
#[test]
fn duchon_operator_penalty_psi_jets_match_the_forward_3d_order0_power9() {
    let (data, spec) = frozen_hybrid_fixture(3, 160, 10, DuchonNullspaceOrder::Zero, 9.0);
    assert_eq!(chart_amplification(data.view(), &spec), 1.0, "3-D order-0 power-9 is not amplified");
    let gaps = operator_penalty_gaps(data.view(), &spec, "opers_3d_order0_power9");
    assert_operator_penalty_gaps(&gaps, "opers_3d_order0_power9");
}
