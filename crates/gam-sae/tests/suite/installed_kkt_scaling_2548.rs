use approx::assert_abs_diff_eq;
use gam_sae::manifold::{
    SaeInnerKktScaleBlock, SaeInnerKktScaleError, SaeInstalledInnerKktAudit,
    SaeManifoldTerm, SaeParameterSpaceKktAudit,
};
use gam_solve::arrow_schur::ArrowSchurSystem;

/// Row replication changes gradient-space L2 norms but must not change the
/// componentwise parameter displacement. Decoder gradient and curvature both
/// grow with the number of rows; coordinate blocks remain row-local.
#[test]
fn parameter_scale_is_intensive_under_row_replication() {
    fn fixture(rows: usize) -> ArrowSchurSystem {
        let mut system = ArrowSchurSystem::new(rows, 1, 2);
        for row in &mut system.rows {
            row.htt[[0, 0]] = 4.0;
            row.gt[0] = 0.08;
        }
        system.hbb[[0, 0]] = 10.0 * rows as f64;
        system.hbb[[1, 1]] = 20.0 * rows as f64;
        system.gb[0] = 0.3 * rows as f64;
        system.gb[1] = -0.2 * rows as f64;
        system
    }

    let one_row = fixture(1);
    let many_rows = fixture(64);
    let raw_norm_sq = |system: &ArrowSchurSystem| {
        system
            .rows
            .iter()
            .flat_map(|row| row.gt.iter())
            .chain(system.gb.iter())
            .map(|gradient| gradient * gradient)
            .sum::<f64>()
    };
    assert!(raw_norm_sq(&many_rows) > 1_000.0 * raw_norm_sq(&one_row));

    let one_scaled = SaeManifoldTerm::system_scaled_grad_max(&one_row)
        .expect("positive diagonal curvature");
    let many_scaled = SaeManifoldTerm::system_scaled_grad_max(&many_rows)
        .expect("positive diagonal curvature");
    assert_abs_diff_eq!(one_scaled, 0.03, epsilon = 1.0e-15);
    assert_abs_diff_eq!(many_scaled, one_scaled, epsilon = 1.0e-15);
}

/// A missing curvature scale may not be skipped: zero curvature with nonzero
/// gradient is a typed non-certificate, not a zero contribution to the max.
#[test]
fn parameter_scale_refuses_unscaled_gradient() {
    let mut system = ArrowSchurSystem::new(0, 0, 1);
    system.gb[0] = 1.0;
    let error = SaeManifoldTerm::system_scaled_grad_max(&system)
        .expect_err("nonzero gradient without curvature must be unresolved");
    assert!(matches!(
        error,
        SaeInnerKktScaleError::InvalidCurvature {
            block: SaeInnerKktScaleBlock::SharedDecoder,
            component: 0,
            gradient: 1.0,
            curvature: 0.0,
        }
    ));
}

#[test]
fn audit_accepts_either_valid_stationarity_currency() {
    let parameter_certified = SaeInstalledInnerKktAudit {
        raw_gradient_norm: 1.0,
        quotient_gradient_norm: 1.0,
        stationarity_bound: 1.0e-5,
        parameter_space: SaeParameterSpaceKktAudit::Resolved {
            scaled_gradient_max: 9.0e-6,
            stationarity_bound: 1.0e-5,
        },
    };
    assert!(parameter_certified.certifies());

    let unresolved = SaeInstalledInnerKktAudit {
        parameter_space: SaeParameterSpaceKktAudit::Unresolved(
            SaeInnerKktScaleError::InvalidCurvature {
                block: SaeInnerKktScaleBlock::SharedDecoder,
                component: 0,
                gradient: 1.0,
                curvature: 0.0,
            },
        ),
        ..parameter_certified
    };
    assert!(!unresolved.certifies());
}
