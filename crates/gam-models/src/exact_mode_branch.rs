use crate::custom_family::{CustomFamilyWarmStart, EvalMode};
use ndarray::Array1;

/// The coefficient mode an exact profiled objective is solved from.
///
/// The profiled objective `V(θ) = min_β J(β; θ)` is a function of `θ` only
/// when the inner minimizer is reached from a start that does not depend on
/// which outer trials happened to be evaluated before it. A warm start taken
/// from the last *trial* breaks that: a rejected line-search probe would then
/// choose the basin of the next probe, and the value at one `θ` would depend on
/// the order in which the line search visited its candidates.
///
/// The previous answer was a cold solve at every trial after the first
/// derivative-bearing evaluation. That keeps the value history-free but throws
/// away the one thing the outer walk knows — the certified mode of the iterate
/// it is stepping from — and far from the seed the cold solve is the fragile
/// one (gam#2765: at `ρ = 12` with the Jeffreys term armed, every cold start
/// took one trust-region step into a region where the reduced information is
/// singular and stalled there, so every line-search probe was refused as
/// infeasible and the outer search halted with `|g| = 7.5` while the accepted
/// iterates had been solving fine).
///
/// This branch therefore carries ONE anchor: the certified mode of the current
/// accepted outer iterate. Every evaluation warm-starts from it, and only a
/// derivative-bearing evaluation — which the outer solver requests at an
/// iterate it has accepted, never at a probe — may replace it. Value-only
/// probes read the anchor and never write it, so within one line search every
/// candidate step is solved from the same start and the objective is a
/// function of `θ` and of the accepted iterate, not of the probe order. Before
/// the first derivative-bearing evaluation there is no accepted iterate, so the
/// value-only seed probes carry their converged mode forward.
///
/// An evaluation at a `θ` the walk has already accepted is solved from that
/// iterate's own certified mode, not from the anchor. After a multi-start the
/// anchor belongs to whichever walk accepted an iterate last, while the terminal
/// certification re-evaluates the winning walk's iterate. Solving that
/// evaluation from another walk's mode is a different inner problem at the
/// certified `θ` (gam#2765: seed 0 certified, a later seed moved the anchor, and
/// the terminal `ValueGradientHessian` evaluation at seed 0's `θ` started from
/// `|β|∞ = 9.46` and diverged onto the singular Jeffreys face). The certified
/// modes are keyed by the bits of the full `θ`, so a ψ move is a different
/// iterate even where `ρ` did not move.
#[derive(Default)]
pub(crate) struct ExactCoefficientModeBranch {
    /// The mode every evaluation is solved from.
    anchor: Option<CustomFamilyWarmStart>,
    /// Whether `anchor` is the certified mode of an accepted outer iterate
    /// (written by a derivative-bearing evaluation) rather than a seed-time
    /// carry. Once true, seeds and value-only probes can no longer write it.
    anchored_at_iterate: bool,
    /// The certified mode of every accepted outer iterate, keyed by its `θ` bits.
    iterate_modes: Vec<(Vec<u64>, CustomFamilyWarmStart)>,
}

fn theta_bits(theta: &Array1<f64>) -> Vec<u64> {
    theta.iter().map(|value| value.to_bits()).collect()
}

impl ExactCoefficientModeBranch {
    /// The warm-start candidates for one evaluation at `theta`: the certified
    /// mode of the accepted iterate at `theta` when there is one, otherwise the
    /// anchor, when it is dimensionally compatible with `rho`; otherwise a cold
    /// solve.
    ///
    /// The returned flag is true exactly at the first derivative-bearing
    /// evaluation, the moment the anchor becomes iterate-owned.
    pub(crate) fn candidates(
        &mut self,
        eval_mode: EvalMode,
        theta: &Array1<f64>,
        rho: &Array1<f64>,
    ) -> (bool, Vec<Option<CustomFamilyWarmStart>>) {
        let first_iterate_evaluation =
            !self.anchored_at_iterate && !matches!(eval_mode, EvalMode::ValueOnly);
        let key = theta_bits(theta);
        let warm = self
            .iterate_modes
            .iter()
            .find(|(bits, _)| *bits == key)
            .map(|(_, warm)| warm)
            .or(self.anchor.as_ref())
            .filter(|warm| warm.compatible_with_rho(rho))
            .cloned();
        (first_iterate_evaluation, vec![warm])
    }

    /// Install a seed mode from outside the walk (an outer ρ-cache coefficient
    /// seed). Refused once an accepted iterate owns the anchor: a cached seed
    /// from another walk must not displace the mode this walk certified.
    pub(crate) fn install_seed(&mut self, warm_start: CustomFamilyWarmStart) -> bool {
        if self.anchored_at_iterate {
            false
        } else {
            self.anchor = Some(warm_start);
            true
        }
    }

    /// Record the mode an evaluation at `theta` converged to. A
    /// derivative-bearing evaluation is an accepted iterate: it replaces the
    /// anchor and becomes the certified mode at `theta`. A value-only probe
    /// writes only while no iterate has been accepted yet. A mode that did not
    /// converge is never recorded — the evaluation it came from is refused by
    /// the caller, and a refused trial must leave no trace.
    pub(crate) fn record_value(
        &mut self,
        eval_mode: EvalMode,
        theta: &Array1<f64>,
        warm_start: CustomFamilyWarmStart,
        converged: bool,
    ) {
        if !converged {
            return;
        }
        if !matches!(eval_mode, EvalMode::ValueOnly) {
            let key = theta_bits(theta);
            self.iterate_modes.retain(|(bits, _)| *bits != key);
            self.iterate_modes.push((key, warm_start.clone()));
            self.anchor = Some(warm_start);
            self.anchored_at_iterate = true;
        } else if !self.anchored_at_iterate {
            self.anchor = Some(warm_start);
        }
    }
}
