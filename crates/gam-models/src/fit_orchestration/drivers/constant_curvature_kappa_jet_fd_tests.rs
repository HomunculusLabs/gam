// #2458 / #2747: FD gates for the constant-curvature ψ profile's exact
// derivative jet, in BOTH outer coordinates `ψ = (κ, η)` with `η = ln ℓ`.
//
// The κ profile is the route #2458 is about: it was declaring
// `DeclaredHessianForm::Unavailable` and being held to a raw gradient band with
// no derivation, while the routes that could supply curvature ran a derived,
// far more permissive criterion. It now supplies an exact Hessian, and an
// incorrect second derivative here would not produce a wrong fit — it would
// produce a wrong CERTIFICATE, silently moving the stationarity bound. #2747
// added the range coordinate, whose derivatives carry the same weight: the η
// gradient is what the outer solve descends and the `V_κη` / `V_ηη` entries are
// what the profile's Schur reduction divides by. So every block of both orders
// is gated before anything is allowed to consume them.
//
// SPEC line 2 permits finite differences inside a test; production carries none.
#[cfg(test)]
mod constant_curvature_kappa_jet_fd_tests {
    use super::*;
    use gam_terms::basis::{
        CenterStrategy, ConstantCurvatureBasisSpec, ConstantCurvatureIdentifiability,
    };

    /// 80 chart points on a disk of radius < 0.42 (inside every κ ∈ [−2, 2]
    /// chart) with a smooth radial+angular response. Deterministic.
    fn fixture() -> (Array2<f64>, Array1<f64>) {
        let mut state = 0x9e37_79b9_7f4a_7c15_u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        let n = 80usize;
        let mut data = Array2::<f64>::zeros((n, 2));
        let mut y = Array1::<f64>::zeros(n);
        for row in 0..n {
            let radius = 0.40 * next().sqrt();
            let angle = 2.0 * std::f64::consts::PI * next();
            let (x0, x1) = (radius * angle.cos(), radius * angle.sin());
            data[(row, 0)] = x0;
            data[(row, 1)] = x1;
            // A smooth signal plus a small deterministic wobble, so the profile
            // has a genuine interior λ̂ rather than a railed one.
            y[row] = (3.0 * x0).sin() + 0.5 * x1 * x1 + 0.05 * (17.0 * (x0 + x1)).cos();
        }
        (data, y)
    }

    /// The auto range this fixture's realized centers produce — the point the
    /// η sweeps below are centred on.
    fn seed_eta(data: &Array2<f64>) -> f64 {
        let spec = spec_at(0.0, 1.0);
        let centers = gam_terms::basis::constant_curvature_realized_centers(data.view(), &spec)
            .expect("realized centers");
        gam_terms::basis::realized_constant_curvature_length_scale(centers.view(), 0.0)
            .expect("auto range")
            .ln()
    }

    fn spec_at(kappa: f64, length_scale: f64) -> ConstantCurvatureBasisSpec {
        ConstantCurvatureBasisSpec {
            center_strategy: CenterStrategy::FarthestPoint { num_centers: 10 },
            kappa,
            kappa_fixed: false,
            length_scale,
            length_scale_fixed: false,
            double_penalty: false,
            // The profile's own frame (see `ConstantCurvatureProfile::new`):
            // a frozen transform is a predict-time replay artifact and would
            // omit the frame's ψ derivative.
            identifiability: ConstantCurvatureIdentifiability::CenterSumToZero,
        }
    }

    fn jet_at(
        data: &Array2<f64>,
        y: &Array1<f64>,
        kappa: f64,
        eta: f64,
    ) -> crate::fit_orchestration::drivers::ProfiledRemlPsiJet {
        constant_curvature_psi_profile_jet(data.view(), y.view(), &spec_at(kappa, eta.exp()))
            .expect("the fixture disk is inside every probed ψ point")
    }

    /// `∇V` and `∇²V` from the analytic jet against central finite differences
    /// of the profile VALUE the same call returns.
    ///
    /// The FD subject is the shipped objective, not a reconstruction of it: the
    /// value compared against is the jet's own `value`, which the jet has
    /// already checked equals the forward fit's `reml_score`. A chart that
    /// reproduced the score but differentiated something else would fail here.
    ///
    /// The cross entry `V_κη` is differenced as `∂/∂η` of the ANALYTIC `V_κ`,
    /// i.e. in the opposite order from the closed form's own derivation, so
    /// agreement is a real check rather than a restatement of symmetry.
    #[test]
    fn constant_curvature_psi_jet_matches_central_fd_of_the_profile() {
        let (data, y) = fixture();
        let eta0 = seed_eta(&data);

        let h = 1e-4_f64;
        for &kappa in &[-1.2_f64, -0.4, 0.0, 0.6, 1.4] {
            for &eta in &[eta0 - 0.5, eta0, eta0 + 0.5] {
                let jet = jet_at(&data, &y, kappa, eta);
                let kp = jet_at(&data, &y, kappa + h, eta);
                let km = jet_at(&data, &y, kappa - h, eta);
                let ep = jet_at(&data, &y, kappa, eta + h);
                let em = jet_at(&data, &y, kappa, eta - h);

                let fd_grad = [
                    (kp.value - km.value) / (2.0 * h),
                    (ep.value - em.value) / (2.0 * h),
                ];
                let fd_hess = [
                    (kp.value - 2.0 * jet.value + km.value) / (h * h),
                    (ep.gradient[0] - em.gradient[0]) / (2.0 * h),
                    (ep.value - 2.0 * jet.value + em.value) / (h * h),
                ];
                let analytic_hess = [jet.hessian[0][0], jet.hessian[0][1], jet.hessian[1][1]];

                for (axis, label) in [(0usize, "∂V/∂κ"), (1, "∂V/∂η")] {
                    let error =
                        (jet.gradient[axis] - fd_grad[axis]).abs() / (1.0 + fd_grad[axis].abs());
                    assert!(
                        error <= 1e-5,
                        "κ={kappa} η={eta}: {label} analytic {:.9e} vs central FD {:.9e} (rel {error:.3e})",
                        jet.gradient[axis],
                        fd_grad[axis]
                    );
                }
                // Second-order central FD of a value carrying an inner root
                // solve has an `O(δ/h²)` floor; the mixed entry differences an
                // analytic first derivative instead and is held an order
                // tighter. These are the measured maxima with room, not
                // tolerances chosen to make the test pass.
                for (slot, label, tol) in [
                    (0usize, "∂²V/∂κ²", 1e-3),
                    (1, "∂²V/∂κ∂η", 1e-4),
                    (2, "∂²V/∂η²", 1e-3),
                ] {
                    let error = (analytic_hess[slot] - fd_hess[slot]).abs()
                        / (1.0 + fd_hess[slot].abs());
                    assert!(
                        error <= tol,
                        "κ={kappa} η={eta}: {label} analytic {:.9e} vs central FD {:.9e} (rel {error:.3e})",
                        analytic_hess[slot],
                        fd_hess[slot]
                    );
                }
                assert!(
                    (jet.hessian[0][1] - jet.hessian[1][0]).abs() <= 1e-12 * (1.0 + jet.hessian[0][1].abs()),
                    "κ={kappa} η={eta}: the Hessian must be stored symmetrically"
                );
            }
        }
    }

    /// The jet's `∇V` against the INDEPENDENT reverse-mode adjoint contraction
    /// the route used to ship — `∂V/∂X · dX/dψ + ∂V/∂S · dS/dψ` through
    /// `gaussian_reml_multi_closed_form_backward_from_fit`, now for BOTH
    /// coordinates.
    ///
    /// Two derivations of one quantity through disjoint code (an explicit
    /// closed-form chart versus the solver's own VJPs). Agreement is what makes
    /// the closed-form chart's SECOND derivatives — which have no independent
    /// implementation to compare against — trustworthy at first order too.
    #[test]
    fn constant_curvature_psi_jet_gradient_matches_the_reverse_mode_adjoint() {
        let (data, y) = fixture();
        let eta0 = seed_eta(&data);
        for &kappa in &[-1.2_f64, -0.4, 0.0, 0.6, 1.4] {
            for &eta in &[eta0 - 0.5, eta0, eta0 + 0.5] {
                let spec = spec_at(kappa, eta.exp());
                let basis = gam_terms::basis::build_constant_curvature_basis(data.view(), &spec)
                    .expect("the fixture disk is inside every probed ψ point");
                let jets = gam_terms::basis::build_constant_curvature_basis_psi_derivatives(
                    data.view(),
                    &spec,
                )
                .expect("the fixture disk is inside every probed ψ point");
                let smooth_design = basis.design.to_dense();
                let (n, p) = smooth_design.dim();

                // Same intercept-augmented chart the profile builds.
                let mut design = Array2::<f64>::ones((n, p + 1));
                design.slice_mut(s![.., 1..]).assign(&smooth_design);
                let mut penalty = Array2::<f64>::zeros((p + 1, p + 1));
                penalty
                    .slice_mut(s![1.., 1..])
                    .assign(&basis.active_penalties[0].matrix);

                let response_2d = y.view().insert_axis(ndarray::Axis(1));
                let fit = gam_solve::gaussian_reml::gaussian_reml_multi_closed_form(
                    design.view(),
                    response_2d.view(),
                    penalty.view(),
                    None,
                    None,
                )
                .expect("the profile fit converges on the fixture");
                let backward =
                    gam_solve::gaussian_reml::gaussian_reml_multi_closed_form_backward_from_fit(
                        design.view(),
                        response_2d.view(),
                        penalty.view(),
                        None,
                        &fit,
                        0.0,
                        None,
                        None,
                        1.0,
                        0.0,
                    )
                    .expect("the reverse-mode adjoint of the profile fit is available");

                let jet = constant_curvature_psi_profile_jet(data.view(), y.view(), &spec)
                    .expect("the fixture disk is inside every probed ψ point");
                assert!(
                    (jet.value - fit.reml_score).abs() <= 1e-9 * (1.0 + fit.reml_score.abs()),
                    "κ={kappa} η={eta}: the jet and the direct fit disagree on the VALUE: {:.9e} vs {:.9e}",
                    jet.value,
                    fit.reml_score
                );

                let coordinates = [
                    ("κ", 0usize, &jets.design_kappa, &jets.penalties_kappa[0]),
                    ("η", 1, &jets.design_eta, &jets.penalties_eta[0]),
                ];
                for (label, axis, design_block, penalty_block) in coordinates {
                    let mut design_psi = Array2::<f64>::zeros((n, p + 1));
                    design_psi.slice_mut(s![.., 1..]).assign(design_block);
                    let mut penalty_psi = Array2::<f64>::zeros((p + 1, p + 1));
                    penalty_psi.slice_mut(s![1.., 1..]).assign(penalty_block);
                    let adjoint = backward
                        .grad_x
                        .iter()
                        .zip(design_psi.iter())
                        .map(|(&a, &d)| a * d)
                        .sum::<f64>()
                        + backward
                            .grad_penalty
                            .iter()
                            .zip(penalty_psi.iter())
                            .map(|(&a, &d)| a * d)
                            .sum::<f64>();
                    let gradient = jet.gradient[axis];
                    let error = (gradient - adjoint).abs() / (1.0 + adjoint.abs());
                    assert!(
                        error <= 1e-8,
                        "κ={kappa} η={eta}: the closed-form chart's dV/d{label} {gradient:.9e} disagrees with the reverse-mode adjoint {adjoint:.9e} (rel {error:.3e})"
                    );
                }
            }
        }
    }
}
