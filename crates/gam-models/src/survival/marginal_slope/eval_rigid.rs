//! Rigid (no-flex) per-row evaluation: the rigid closed-form primary kernel
//! wrapper, scalar-flex score-geometry guard, and the rigid vector NLL value.

use super::*;

impl SurvivalMarginalSlopeFamily {
    pub(crate) fn ensure_scalar_flex_exact_score_geometry(
        &self,
        context: &str,
    ) -> Result<(), String> {
        if self.score_dim() == 1 {
            return Ok(());
        }
        Err(SurvivalMarginalSlopeError::UnsupportedConfiguration {
            reason: format!(
                "{context}: survival marginal-slope exact flexible row calculus is scalar-z only; K={} must use the rigid shared-slope vector kernel or a widened per-z primary kernel",
                self.score_dim()
            ),
        }
        .into())
    }

    pub(crate) fn row_neglog_rigid_vector_value(
        &self,
        row: usize,
        q_geom: SurvivalMarginalSlopeDynamicRowValues,
        block_states: &[ParameterBlockState],
        probit_scale: f64,
        slope_workspace: &mut SlopeRowWorkspace,
        value_workspace: &RigidVectorValueWorkspace<'_>,
    ) -> Result<f64, String> {
        self.fill_slope_values_for_row(row, block_states, slope_workspace)?;
        let z_row = self.z.row(row);
        let z = z_row.as_slice().ok_or_else(|| {
            "survival marginal-slope vector value score row must be contiguous".to_string()
        })?;
        survival_marginal_slope_vector_neglog(
            row,
            q_geom.q0,
            q_geom.q1,
            q_geom.qd1,
            slope_workspace.values(),
            z,
            value_workspace,
            self.weights[row],
            self.event[row],
            self.derivative_guard,
            probit_scale,
        )
    }
}
