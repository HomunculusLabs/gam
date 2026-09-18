#![cfg(test)]
//! #2976 — a returned mode settles on the accumulation band of its row sum.
//!
//! One coefficient over `ROWS` rows with `ℓ = −½ Σᵢ (yᵢ − β)²`. The responses
//! alternate `±1e6` about a small pattern, so near the mode every row's term is
//! `O(1e6)` while their sum cancels. The computed residual then floors on that
//! sum's rounding: far above an absolute target of `1e-11`, and far below the band
//! the summands accumulate. The workspace serves the gradient, as the survival
//! marginal-slope workspace does, and reports the absolute summands when
//! `measures_summands` is set.
use super::*;

const ROWS: usize = 20_000;
const INNER_TOL: f64 = 1.0e-11;

#[derive(Clone)]
struct CancellingRowSumFamily {
    responses: Arc<Vec<f64>>,
    /// The Newton model's curvature as a multiple of the objective's own.
    model_scale: f64,
    /// Whether the workspace reports the absolute row summands.
    measures_summands: bool,
}

struct RowSum {
    log_likelihood: f64,
    gradient: f64,
    absolute_sum: f64,
}

impl CancellingRowSumFamily {
    fn new(model_scale: f64, measures_summands: bool) -> Self {
        let responses = (0..ROWS)
            .map(|row| {
                let sign = if row % 2 == 0 { 1.0 } else { -1.0 };
                sign * 1.0e6 + 0.1 * (row % 7) as f64
            })
            .collect();
        Self {
            responses: Arc::new(responses),
            model_scale,
            measures_summands,
        }
    }

    fn mode(&self) -> f64 {
        self.responses.iter().sum::<f64>() / ROWS as f64
    }

    fn row_sum(&self, beta: f64) -> RowSum {
        let mut sum = RowSum {
            log_likelihood: 0.0,
            gradient: 0.0,
            absolute_sum: 0.0,
        };
        for &response in self.responses.iter() {
            let term = response - beta;
            sum.log_likelihood -= 0.5 * term * term;
            sum.gradient += term;
            sum.absolute_sum += term.abs();
        }
        sum
    }

    fn model_curvature(&self) -> f64 {
        self.model_scale * ROWS as f64
    }

    /// The accumulation band of the row sum at `beta`: `γ_ROWS · Σᵢ |yᵢ − β|`.
    fn summand_band(&self, beta: f64) -> f64 {
        gam_linalg::roundoff::accumulation_growth(ROWS) * self.row_sum(beta).absolute_sum
    }
}

struct CancellingRowSumWorkspace {
    sum: RowSum,
    curvature: f64,
    measures_summands: bool,
}

impl ExactNewtonJointHessianWorkspace for CancellingRowSumWorkspace {
    fn warm_up_outer_caches_for_mode(&self, eval_mode: EvalMode) -> Result<(), String> {
        // No directional cache to prime, in any mode.
        match eval_mode {
            EvalMode::ValueOnly | EvalMode::ValueAndGradient | EvalMode::ValueGradientHessian => {
                Ok(())
            }
        }
    }

    fn hessian_dense(&self) -> Result<Option<Array2<f64>>, String> {
        Ok(Some(array![[self.curvature]]))
    }

    fn directional_derivative(
        &self,
        direction: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        assert_direction_finite(direction, "cancelling row-sum workspace directional derivative");
        Ok(None)
    }

    fn joint_gradient_evaluation(
        &self,
    ) -> Result<Option<ExactNewtonJointGradientEvaluation>, String> {
        Ok(Some(ExactNewtonJointGradientEvaluation {
            log_likelihood: self.sum.log_likelihood,
            gradient: array![self.sum.gradient],
        }))
    }

    fn joint_gradient_accumulation(&self) -> Result<Option<GradientAccumulation>, String> {
        // The row loop is one sequential sum of `ROWS` terms.
        Ok(self.measures_summands.then(|| GradientAccumulation {
            accumulation_depth: ROWS,
            absolute_sums: array![self.sum.absolute_sum],
        }))
    }
}

impl CustomFamily for CancellingRowSumFamily {
    fn evaluate(&self, block_states: &[ParameterBlockState]) -> Result<FamilyEvaluation, String> {
        assert_states_finite(block_states, "cancelling row-sum evaluation");
        let sum = self.row_sum(block_states[0].beta[0]);
        Ok(FamilyEvaluation {
            log_likelihood: sum.log_likelihood,
            blockworking_sets: vec![BlockWorkingSet::ExactNewton {
                gradient: array![sum.gradient],
                hessian: SymmetricMatrix::Dense(array![[self.model_curvature()]]),
            }],
        })
    }

    fn inner_coefficient_hessian_hvp_available(&self, specs: &[ParameterBlockSpec]) -> bool {
        assert_specs_consistent(specs, "cancelling row-sum HVP availability");
        true
    }

    fn inner_joint_workspace_gradient_available(&self, specs: &[ParameterBlockSpec]) -> bool {
        assert_specs_consistent(specs, "cancelling row-sum workspace gradient availability");
        true
    }

    fn exact_newton_joint_hessian_workspace(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
    ) -> Result<Option<Arc<dyn ExactNewtonJointHessianWorkspace>>, String> {
        assert_specs_consistent(specs, "cancelling row-sum workspace");
        assert_states_finite(block_states, "cancelling row-sum workspace");
        Ok(Some(Arc::new(CancellingRowSumWorkspace {
            sum: self.row_sum(block_states[0].beta[0]),
            curvature: self.model_curvature(),
            measures_summands: self.measures_summands,
        })))
    }
}

/// Fit from `start` under a cycle budget, returning `(converged, β̂)`.
fn fit(family: &CancellingRowSumFamily, start: f64, inner_max_cycles: usize) -> (bool, f64) {
    let spec = ParameterBlockSpec {
        name: "location".to_string(),
        design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(array![[1.0]])),
        offset: array![0.0],
        penalties: Vec::new(),
        nullspace_dims: Vec::new(),
        initial_log_lambdas: Array1::zeros(0),
        initial_beta: Some(array![start]),
        gauge_priority: 100,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    };
    let options = BlockwiseFitOptions {
        inner_max_cycles,
        inner_tol: INNER_TOL,
        use_remlobjective: false,
        compute_covariance: false,
        ..BlockwiseFitOptions::default()
    };
    let result = inner_blockwise_fit(family, &[spec], &[Array1::zeros(0)], &options, None)
        .expect("a cancelling row-sum solve ends with an inner result, not a refusal");
    (result.converged, result.block_states[0].beta[0])
}

/// (a) The stalled mode settles. Its residual is above the caller's target and
/// within the band its summands accumulate, and the workspace measures them.
#[test]
fn a_returned_mode_at_its_row_sum_floor_settles_on_the_summand_band_2976() {
    let family = CancellingRowSumFamily::new(1.0, true);
    let (converged, beta) = fit(&family, family.mode() + 0.5, 40);
    let residual = family.row_sum(beta).gradient.abs();
    let caller_target = INNER_TOL * (1.0 + residual);
    let band = family.summand_band(beta);
    println!(
        "[2976] measured summands: converged={converged} residual={residual:.3e} \
         caller_target={caller_target:.3e} summand_band={band:.3e}"
    );
    assert!(
        residual > caller_target,
        "the fixture must floor above the caller's target: residual {residual:.3e} against \
         {caller_target:.3e}"
    );
    assert!(
        converged && residual <= band,
        "a decrement-certified mode within its summand band must settle: converged={converged}, \
         residual {residual:.3e}, band {band:.3e}"
    );
}

/// (b) Negative control: a mark whose residual is well above the summand band is
/// still revoked. The model carries four times the objective's curvature, so each
/// cycle removes a quarter of the error and twelve cycles leave the residual far
/// above the band.
#[test]
fn a_mark_above_the_summand_band_is_still_revoked_2976() {
    let family = CancellingRowSumFamily::new(4.0, true);
    let (converged, beta) = fit(&family, family.mode() + 0.05, 12);
    let residual = family.row_sum(beta).gradient.abs();
    let band = family.summand_band(beta);
    println!(
        "[2976] above the band: converged={converged} residual={residual:.3e} \
         summand_band={band:.3e}"
    );
    assert!(
        residual > band,
        "the control must end above the summand band: residual {residual:.3e} against {band:.3e}"
    );
    assert!(
        !converged,
        "a residual above the summand band must not settle: residual {residual:.3e}, band \
         {band:.3e}"
    );
}

/// (c) Positive control: the same stalled mode from a workspace that measures no
/// summands keeps the assembled-gradient band, and every mark is revoked until the
/// budget ends the solve.
#[test]
fn a_workspace_that_measures_no_summands_keeps_revoking_the_stalled_mode_2976() {
    let family = CancellingRowSumFamily::new(1.0, false);
    let (converged, beta) = fit(&family, family.mode() + 0.5, 40);
    let residual = family.row_sum(beta).gradient.abs();
    let caller_target = INNER_TOL * (1.0 + residual);
    println!(
        "[2976] no summands measured: converged={converged} residual={residual:.3e} \
         caller_target={caller_target:.3e}"
    );
    assert!(
        residual > caller_target,
        "the control must floor above the caller's target: residual {residual:.3e} against \
         {caller_target:.3e}"
    );
    assert!(
        !converged,
        "without a summand measurement the stalled mode must stay revoked: residual \
         {residual:.3e} against {caller_target:.3e}"
    );
}
