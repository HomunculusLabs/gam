// #2458: FD gates for the constant-curvature κ profile's exact derivative jet.
//
// The κ profile is the route this issue is about: it was declaring
// `DeclaredHessianForm::Unavailable` and being held to a raw gradient band with
// no derivation, while the routes that could supply curvature ran a derived,
// far more permissive criterion. It now supplies an exact `d²V/dκ²`, and an
// incorrect second derivative here would not produce a wrong fit — it would
// produce a wrong CERTIFICATE, silently moving the stationarity bound. So both
// orders are gated before anything is allowed to consume them.
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

    fn spec_at(kappa: f64) -> ConstantCurvatureBasisSpec {
        ConstantCurvatureBasisSpec {
            center_strategy: CenterStrategy::FarthestPoint { num_centers: 10 },
            kappa,
            kappa_fixed: false,
            length_scale: 0.0,
            double_penalty: false,
            // The profile's own frame (see `ConstantCurvatureProfile::new`):
            // a frozen transform is a predict-time replay artifact and would
            // omit the frame's κ derivative.
            identifiability: ConstantCurvatureIdentifiability::CenterSumToZero,
        }
    }

    /// `dV/dκ` and `d²V/dκ²` from the analytic jet against central finite
    /// differences of the profile VALUE the same call returns.
    ///
    /// The FD subject is the shipped objective, not a reconstruction of it: the
    /// value compared against is `jet(κ).0`, which the jet itself has already
    /// checked equals the forward fit's `reml_score`. A chart that reproduced
    /// the score but differentiated something else would fail here.
    #[test]
    fn constant_curvature_kappa_jet_matches_central_fd_of_the_profile() {
        let (data, y) = fixture();
        let value_at = |kappa: f64| -> f64 {
            constant_curvature_kappa_profile_value_jet(data.view(), y.view(), &spec_at(kappa))
                .expect("the fixture disk is inside every probed κ chart")
                .0
        };

        let h = 1e-4_f64;
        for &kappa in &[-1.2_f64, -0.4, 0.0, 0.6, 1.4] {
            let (value, gradient, curvature) =
                constant_curvature_kappa_profile_value_jet(data.view(), y.view(), &spec_at(kappa))
                    .expect("the fixture disk is inside every probed κ chart");
            let up = value_at(kappa + h);
            let down = value_at(kappa - h);
            let fd_first = (up - down) / (2.0 * h);
            let fd_second = (up - 2.0 * value + down) / (h * h);
            let scale = 1.0 + value.abs();
            let first_error = (gradient - fd_first).abs() / (1.0 + fd_first.abs());
            let second_error = (curvature - fd_second).abs() / (1.0 + fd_second.abs());
            eprintln!(
                "[#2458 κ-jet] κ={kappa}: V={value:.9e} (scale {scale:.3e}) \
                 dV/dκ analytic {gradient:.9e} vs FD {fd_first:.9e} (rel {first_error:.3e}) | \
                 d²V/dκ² analytic {curvature:.9e} vs FD {fd_second:.9e} (rel {second_error:.3e})"
            );
            assert!(
                first_error <= 1e-5,
                "κ={kappa}: dV/dκ analytic {gradient:.9e} vs central FD {fd_first:.9e} (rel {first_error:.3e})"
            );
            // Second-order central FD of a value carrying an inner root solve
            // has an `O(δ/h²)` floor; this is the measured maximum with room,
            // not a tolerance chosen to make the test pass.
            assert!(
                second_error <= 1e-3,
                "κ={kappa}: d²V/dκ² analytic {curvature:.9e} vs central FD {fd_second:.9e} (rel {second_error:.3e})"
            );
        }
    }

    /// The jet's `dV/dκ` against the INDEPENDENT reverse-mode adjoint
    /// contraction the route used to ship — `∂V/∂X · dX/dκ + ∂V/∂S · dS/dκ`
    /// through `gaussian_reml_multi_closed_form_backward_from_fit`.
    ///
    /// Two derivations of one quantity through disjoint code (an explicit
    /// closed-form chart versus the solver's own VJPs). Agreement is what makes
    /// the closed-form chart's SECOND derivative — which has no independent
    /// implementation to compare against — trustworthy at first order too.
    #[test]
    fn constant_curvature_kappa_jet_gradient_matches_the_reverse_mode_adjoint() {
        let (data, y) = fixture();
        for &kappa in &[-1.2_f64, -0.4, 0.0, 0.6, 1.4] {
            let spec = spec_at(kappa);
            let basis = gam_terms::basis::build_constant_curvature_basis(data.view(), &spec)
                .expect("the fixture disk is inside every probed κ chart");
            let derivatives = gam_terms::basis::build_constant_curvature_basis_kappa_derivatives(
                data.view(),
                &spec,
            )
            .expect("the fixture disk is inside every probed κ chart");
            let smooth_design = basis.design.to_dense();
            let (n, p) = smooth_design.dim();

            // Same intercept-augmented chart the profile builds.
            let mut design = Array2::<f64>::ones((n, p + 1));
            design.slice_mut(s![.., 1..]).assign(&smooth_design);
            let mut design_kappa = Array2::<f64>::zeros((n, p + 1));
            design_kappa
                .slice_mut(s![.., 1..])
                .assign(&derivatives.first.design_derivative);
            let mut penalty = Array2::<f64>::zeros((p + 1, p + 1));
            penalty
                .slice_mut(s![1.., 1..])
                .assign(&basis.active_penalties[0].matrix);
            let mut penalty_kappa = Array2::<f64>::zeros((p + 1, p + 1));
            penalty_kappa
                .slice_mut(s![1.., 1..])
                .assign(&derivatives.first.penalties_derivative[0]);

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
            let adjoint = backward
                .grad_x
                .iter()
                .zip(design_kappa.iter())
                .map(|(&a, &d)| a * d)
                .sum::<f64>()
                + backward
                    .grad_penalty
                    .iter()
                    .zip(penalty_kappa.iter())
                    .map(|(&a, &d)| a * d)
                    .sum::<f64>();

            let (value, gradient, _) =
                constant_curvature_kappa_profile_value_jet(data.view(), y.view(), &spec)
                    .expect("the fixture disk is inside every probed κ chart");
            assert!(
                (value - fit.reml_score).abs() <= 1e-9 * (1.0 + fit.reml_score.abs()),
                "κ={kappa}: the jet and the direct fit disagree on the VALUE: {value:.9e} vs {:.9e}",
                fit.reml_score
            );
            let error = (gradient - adjoint).abs() / (1.0 + adjoint.abs());
            eprintln!(
                "[#2458 κ-jet vs adjoint] κ={kappa}: closed-form chart {gradient:.9e} vs reverse-mode adjoint {adjoint:.9e} (rel {error:.3e})"
            );
            assert!(
                error <= 1e-8,
                "κ={kappa}: the closed-form chart's dV/dκ {gradient:.9e} disagrees with the reverse-mode adjoint {adjoint:.9e} (rel {error:.3e})"
            );
        }
    }
}
