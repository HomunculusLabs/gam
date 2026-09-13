#![cfg(test)]
//! #979 Jeffreys ruling (b), pin 1: a clean fit through the arming lifecycle IS the
//! unarmed objective's fit. The lifecycle hands back the unarmed member's certified fit
//! unchanged and publishes no arming evidence. The negative control shows that the armed
//! member prices a measurably different objective at the same smoothing strength, so the
//! equality can fail.

use super::*;
use ndarray::array;

/// The one-coefficient quartic likelihood the #2898 mint pins certify, with an arming switch.
#[derive(Clone)]
struct ArmableQuarticFamily {
    linear: f64,
    curvature: f64,
    armed: bool,
}

impl JeffreysArming for ArmableQuarticFamily {
    fn with_jeffreys_armed(
        &self,
        evidence: Option<&gam_problem::jeffreys_arming::JeffreysArmingEvidence>,
    ) -> Self {
        Self {
            armed: evidence.is_some(),
            ..self.clone()
        }
    }
}

impl CustomFamily for ArmableQuarticFamily {
    fn joint_jeffreys_term_required(&self) -> bool {
        self.armed
    }

    fn exact_newton_joint_hessian_beta_dependent(&self) -> bool {
        true
    }

    fn has_explicit_joint_hessian(&self) -> bool {
        true
    }

    fn evaluate(&self, block_states: &[ParameterBlockState]) -> Result<FamilyEvaluation, String> {
        let beta = block_states[0].beta[0];
        Ok(FamilyEvaluation {
            log_likelihood: self.linear * beta
                - 0.5 * beta * beta
                - self.curvature * beta.powi(4) / 12.0,
            blockworking_sets: vec![BlockWorkingSet::ExactNewton {
                gradient: array![self.linear - beta - self.curvature * beta.powi(3) / 3.0],
                hessian: SymmetricMatrix::Dense(array![[1.0 + self.curvature * beta * beta]]),
            }],
        })
    }

    fn exact_newton_joint_hessian(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<Option<Array2<f64>>, String> {
        let beta = block_states[0].beta[0];
        Ok(Some(array![[1.0 + self.curvature * beta * beta]]))
    }

    fn exact_newton_joint_hessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        let beta = block_states[0].beta[0];
        Ok(Some(array![[2.0 * self.curvature * beta * d_beta_flat[0]]]))
    }

    fn exact_newton_hessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        block_idx: usize,
        direction: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        assert_eq!(block_idx, 0, "the quartic family has one block");
        let beta = block_states[0].beta[0];
        Ok(Some(array![[2.0 * self.curvature * beta * direction[0]]]))
    }

    fn exact_newton_hessian_second_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        block_idx: usize,
        u: &Array1<f64>,
        v: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        assert_eq!(block_idx, 0, "the quartic family has one block");
        assert!(
            block_states[0].beta.iter().all(|value| value.is_finite()),
            "the quartic second directional derivative owes a finite mode"
        );
        Ok(Some(array![[2.0 * self.curvature * u[0] * v[0]]]))
    }
}

#[test]
fn a_clean_fit_through_the_arming_lifecycle_is_the_unarmed_fit_979() {
    let family = ArmableQuarticFamily {
        linear: 3.0,
        curvature: 0.5,
        armed: true,
    };
    let specs = vec![ParameterBlockSpec {
        name: "quartic".to_string(),
        design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(array![[1.0]])),
        offset: array![0.0],
        penalties: vec![PenaltyMatrix::Dense(array![[1.0]])],
        nullspace_dims: vec![],
        initial_log_lambdas: array![0.0],
        initial_beta: Some(array![0.75]),
        gauge_priority: 100,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    }];
    let options = BlockwiseFitOptions {
        inner_tol: 1e-11,
        use_remlobjective: true,
        use_outer_hessian: true,
        compute_covariance: false,
        ..BlockwiseFitOptions::default()
    };

    let lifecycle = fit_custom_family_arming_on_evidence(&family, &specs, &options)
        .expect("the clean quartic fit must certify through the arming lifecycle");
    let plain = fit_custom_family(&family.with_jeffreys_armed(None), &specs, &options)
        .expect("the unarmed quartic fit must certify");

    let penalty_counts = validate_blockspecs(&specs).expect("valid quartic spec");
    let layout = penalty_label_layout_with_joint(&specs, penalty_counts, Vec::new())
        .expect("valid label layout");
    let value_at = |member: &ArmableQuarticFamily| {
        outerobjectivegradienthessian_labeled(
            member,
            &specs,
            &options,
            &layout,
            &array![0.0],
            None,
            &gam_problem::RhoPrior::Flat,
            EvalMode::ValueOnly,
        )
        .expect("the quartic outer value must evaluate")
        .objective
    };
    let unarmed_value = value_at(&family.with_jeffreys_armed(None));
    let armed_value = value_at(&family.with_jeffreys_armed(Some(
        &gam_problem::jeffreys_arming::JeffreysArmingEvidence::StrictSaddle {
            stationarity_residual: 0.0,
        },
    )));
    println!(
        "[#979 pin 1] lifecycle beta={:?} plain beta={:?} evidence={:?} \
         lifecycle objective={:?} plain objective={:?} unarmed_value={unarmed_value:.12e} \
         armed_value={armed_value:.12e}",
        lifecycle.blocks[0].beta,
        plain.blocks[0].beta,
        lifecycle.artifacts.jeffreys_arming_evidence,
        lifecycle.penalized_objective(),
        plain.penalized_objective(),
    );

    assert!(
        lifecycle.artifacts.jeffreys_arming_evidence.is_none(),
        "a clean fit must publish no arming evidence"
    );
    assert_eq!(
        lifecycle.blocks[0].beta[0].to_bits(),
        plain.blocks[0].beta[0].to_bits(),
        "a clean fit through the lifecycle must be the unarmed member's fit, bit for bit"
    );
    assert_eq!(
        lifecycle.penalized_objective().map(f64::to_bits),
        plain.penalized_objective().map(f64::to_bits),
        "a clean fit through the lifecycle must price the unarmed objective, bit for bit"
    );
    // Negative control: the armed member prices a different objective at the same rho, so an
    // armed fit could not pass the equalities above by accident.
    assert!(
        (armed_value - unarmed_value).abs() > 1e-6 * (1.0 + unarmed_value.abs()),
        "the armed member must price a measurably different objective: armed {armed_value:e}, \
         unarmed {unarmed_value:e}"
    );
}
