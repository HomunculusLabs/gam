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
/// derivative along ψ, at two steps (ratio 2) so a truncation-limited
/// estimate is told apart from a formula defect. Returns the best relative
/// Frobenius gaps `(first, second)`.
fn chart_gaps(data: ArrayView2<'_, f64>, spec: &DuchonBasisSpec, label: &str) -> (f64, f64) {
    let first_an = analytic_first(data, spec);
    let second_an = analytic_second(data, spec);
    let mut best_first = f64::INFINITY;
    let mut best_second = f64::INFINITY;
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
        best_first = best_first.min(first_gap);
        best_second = best_second.min(second_gap);
    }
    (best_first, best_second)
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
#[test]
fn duchon_chart_is_inert_where_the_kernel_does_not_underflow() {
    let (data, spec) = frozen_hybrid_fixture(2, 90, 10, DuchonNullspaceOrder::Linear, 1.0);
    let amplification = chart_amplification(data.view(), &spec);
    assert_eq!(amplification, 1.0, "a 2-D power-1 hybrid must not be amplified");
    let (first_gap, second_gap) = chart_gaps(data.view(), &spec, "chart_2d_linear_power1");
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
        latent_jacobian_worst_gap(2, DuchonNullspaceOrder::Linear, 1.0, "latent_2d_linear_power1");
    assert_eq!(amplification, 1.0, "a 2-D power-1 hybrid must not be amplified");
    assert!(worst < 1e-5, "latent Jacobian differs from the rebuild by {worst:.3e} (relative)");
}
