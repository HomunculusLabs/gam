use crate::custom_family::{CustomFamilyWarmStart, EvalMode};
use ndarray::Array1;

/// Deterministic coefficient-mode branch for a nonconvex profiled objective.
///
/// A coefficient warm start is not merely a performance cache when the inner
/// problem has multiple local minima: letting rejected outer trials replace it
/// makes the next objective value depend on line-search history. Before the
/// outer search requests derivatives, value-only probes may carry the best
/// solved mode forward. The first derivative-bearing evaluation freezes that
/// input mode as an immutable anchor. Every subsequent trial compares a cold
/// solve with that same anchor, so the profiled objective is a function of the
/// requested hyperparameters rather than the order in which they were visited.
#[derive(Default)]
pub(crate) struct ExactCoefficientModeBranch {
    carried_mode: Option<CustomFamilyWarmStart>,
    anchor: Option<CustomFamilyWarmStart>,
    frozen: bool,
}

impl ExactCoefficientModeBranch {
    pub(crate) fn candidates(
        &mut self,
        eval_mode: EvalMode,
        rho: &Array1<f64>,
    ) -> (bool, Vec<Option<CustomFamilyWarmStart>>) {
        let froze = self.prepare(eval_mode);
        let carried = self
            .frozen
            .then_some(&self.anchor)
            .unwrap_or(&self.carried_mode)
            .as_ref()
            .filter(|warm| warm.compatible_with_rho(rho))
            .cloned();
        match carried {
            Some(warm) => (froze, vec![None, Some(warm)]),
            None => (froze, vec![None]),
        }
    }

    /// Install an external coefficient seed only while value-only exploration
    /// may still advance the branch. A seed arriving after derivative geometry
    /// has been requested cannot replace the immutable anchor.
    pub(crate) fn install_seed(&mut self, warm_start: CustomFamilyWarmStart) -> bool {
        if self.frozen {
            false
        } else {
            self.carried_mode = Some(warm_start);
            true
        }
    }

    /// Freeze the input mode before the first derivative-bearing solve.
    pub(crate) fn prepare(&mut self, eval_mode: EvalMode) -> bool {
        if self.frozen || matches!(eval_mode, EvalMode::ValueOnly) {
            return false;
        }
        self.anchor = self.carried_mode.take();
        self.frozen = true;
        true
    }

    /// Only pre-freeze value probes may advance the carried mode.
    pub(crate) fn record_value(
        &mut self,
        eval_mode: EvalMode,
        warm_start: CustomFamilyWarmStart,
    ) {
        if !self.frozen && matches!(eval_mode, EvalMode::ValueOnly) {
            self.carried_mode = Some(warm_start);
        }
    }
}
