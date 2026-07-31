//! Custom-family error type and its String conversions.

use thiserror::Error;

use crate::{IdentifiabilityAudit, MapUniquenessError};


#[derive(Debug, Clone, Copy, PartialEq)]
pub enum JointNewtonTerminalReason {
    CycleBudget,
    FullyRejectedExactFixedPoint {
        consecutive_cycles: usize,
        joint_trust_radius: f64,
        rejection_counts: [usize; 4],
    },
    FullyRejectedAtTrustRegionFloor {
        consecutive_cycles: usize,
        joint_trust_radius: f64,
        rejection_counts: [usize; 4],
    },
}

impl std::fmt::Display for JointNewtonTerminalReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CycleBudget => write!(f, "cycle budget"),
            Self::FullyRejectedExactFixedPoint {
                consecutive_cycles,
                joint_trust_radius,
                rejection_counts,
            } => write!(
                f,
                "complete rejected-cycle state repeated {consecutive_cycles} times at \
                 trust radius {joint_trust_radius:.6e}; rejects \
                 [model,likelihood,objective,feasibility]={rejection_counts:?}"
            ),
            Self::FullyRejectedAtTrustRegionFloor {
                consecutive_cycles,
                joint_trust_radius,
                rejection_counts,
            } => write!(
                f,
                "all attempts rejected for {consecutive_cycles} cycles at the absolute \
                 trust-region floor {joint_trust_radius:.6e}; rejects \
                 [model,likelihood,objective,feasibility]={rejection_counts:?}"
            ),
        }
    }
}

/// The blockwise inner loop's terminal decision variables — the quantities its
/// convergence verdict is actually taken on.
///
/// The loop certifies with
/// `max_accepted_step <= step_tol && objective_change <= objective_tol`, and then
/// `joint_stationarity_ok || max_proposed_step <= step_tol`. Reporting only the
/// cycle count cannot say which of those four conjuncts failed, and they have
/// different causes: steps still large means the solve needs more cycles, steps
/// tiny with `joint_stationarity_ok == false` means the exact joint gate is the
/// blocker rather than the budget, and an `objective_change` above tolerance
/// means the iterate is still moving. This is deliberately NOT a KKT residual:
/// `BlockwiseInnerResult::kkt_residual` is `None` off a converged iterate on
/// purpose, because no caller may trust an IFT correction there, so the honest
/// diagnostic is the decision variables themselves rather than a residual
/// recomputed at a non-KKT point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum InnerConvergenceTerminalState {
    /// The blockwise Gauss-Seidel route's terminal cycle.
    Blockwise {
        cycle: usize,
        max_accepted_step: f64,
        max_proposed_step: f64,
        step_tol: f64,
        objective_change: f64,
        objective_tol: f64,
        joint_stationarity_ok: bool,
    },
    /// The exact joint-Newton route's terminal cycle. This route DOES have a
    /// genuine stationarity residual (the blockwise one does not, off a
    /// converged iterate), and it has a third outcome the other lacks:
    /// `resolvable_negative_curvature` marks a first-order stationary STRICT
    /// SADDLE, where the score and the Newton proposal both vanish but the exact
    /// penalized Hessian has resolvable negative curvature. That refuses
    /// convergence deliberately, and it is nothing like exhausting a budget.
    JointNewton {
        cycle: usize,
        stationarity_residual: f64,
        residual_tol: f64,
        step_inf: f64,
        step_tol: f64,
        resolvable_negative_curvature: bool,
        /// The smallest stationarity residual this solve actually computed, and
        /// how many cycles have passed since it last improved.
        ///
        /// The terminal residual alone cannot separate a solve that never got
        /// close from one that reached a near-tolerance point and then walked
        /// away from it, and those are different defects with different fixes.
        /// Measured on the transformation-normal wine arm (#2600): the terminal
        /// residual is `1.906e0` while the smallest this same solve computed is
        /// `1.578e-3` — 1200x better, within 1.9x of `residual_tol`, and reached
        /// 27 cycles earlier, after which every accepted step raised the
        /// residual again. Read from the terminal value alone that solve looks
        /// like it never approached stationarity; read with the best value it
        /// is a solve that drifted off a point it had essentially reached.
        best_stationarity_residual: f64,
        cycles_since_best_residual: usize,
        termination_reason: JointNewtonTerminalReason,
    },
}

impl std::fmt::Display for InnerConvergenceTerminalState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Blockwise {
                cycle,
                max_accepted_step,
                max_proposed_step,
                step_tol,
                objective_change,
                objective_tol,
                joint_stationarity_ok,
            } => write!(
                f,
                "blockwise terminal cycle {cycle}: max_accepted_step={max_accepted_step:.6e} \
                 (tol={step_tol:.6e}), max_proposed_step={max_proposed_step:.6e}, \
                 objective_change={objective_change:.6e} (tol={objective_tol:.6e}), \
                 joint_stationarity_ok={joint_stationarity_ok}"
            ),
            Self::JointNewton {
                cycle,
                stationarity_residual,
                residual_tol,
                step_inf,
                step_tol,
                resolvable_negative_curvature,
                best_stationarity_residual,
                cycles_since_best_residual,
                termination_reason,
            } => write!(
                f,
                "joint-Newton terminal cycle {cycle}: \
                 stationarity_residual={stationarity_residual:.6e} (tol={residual_tol:.6e}), \
                 step_inf={step_inf:.6e} (tol={step_tol:.6e}), \
                 resolvable_negative_curvature={resolvable_negative_curvature}, \
                 best_stationarity_residual={best_stationarity_residual:.6e} \
                 (last improved {cycles_since_best_residual} cycle(s) before this one), \
                 termination={termination_reason}"
            ),
        }
    }
}

/// Render the projected-KKT comparison in the inner-refusal message.
///
/// The pair used to be printed as `|r|_inf={:?} against tol={:?}`, which on the
/// common path renders `|r|_inf=None against tol=None` — **two absences laid out
/// as a comparison**. That reads as a measurement that was taken and came out
/// unfavourable, and it is the opposite: nothing was measured. It cost real time
/// on gam#2600, where the phrase sat in every refusal while the actual decision
/// variables (which the `[{terminal}]` block does carry) said something quite
/// different. A missing value has to say it is missing, and which side is
/// missing, because "the solver emitted no KKT diagnostic on this path" and "the
/// residual is 4e5x its tolerance" call for different next steps.
fn render_projected_kkt_comparison(residual: Option<f64>, tol: Option<f64>) -> String {
    match (residual, tol) {
        (Some(residual), Some(tol)) => format!(
            "projected KKT residual |r|_inf={residual:.6e} against tol={tol:.6e}"
        ),
        (Some(residual), None) => format!(
            "projected KKT residual |r|_inf={residual:.6e}; \
             no stationarity tolerance was recorded to compare it against"
        ),
        (None, Some(tol)) => format!(
            "no projected KKT residual was recorded; the stationarity tolerance \
             on this path was {tol:.6e}"
        ),
        (None, None) => "this solver path emits no typed projected-KKT diagnostic, so \
                         neither a residual nor a tolerance was recorded — read the \
                         terminal decision variables above instead"
            .to_string(),
    }
}

#[derive(Debug, Clone, Error)]
pub enum CustomFamilyError {
    #[error("custom-family invalid input in {context}: {reason}")]
    InvalidInput {
        context: &'static str,
        reason: String,
    },
    #[error("custom-family optimization error in {context}: {reason}")]
    Optimization {
        context: &'static str,
        reason: String,
    },
    #[error("{reason}")]
    DimensionMismatch { reason: String },
    #[error("{reason}")]
    NumericalFailure { reason: String },
    #[error("{reason}")]
    ConstraintViolation { reason: String },
    #[error("{reason}")]
    UnsupportedConfiguration { reason: String },
    /// The inner solve did not reach its KKT condition at THIS trial
    /// point, so the analytic outer gradient/Hessian cannot be exposed
    /// (they require `F_beta(beta, theta) = 0`).
    ///
    /// This is a statement about one `theta`, not about the problem: the
    /// outer search should treat the trial as infeasible, back off, and
    /// continue. It previously travelled as
    /// [`UnsupportedConfiguration`](Self::UnsupportedConfiguration) — a
    /// variant that *means* the configuration is structurally
    /// unsupported, i.e. fatal — with the real distinction encoded only
    /// in the message text. Downstream then had to recover it by
    /// substring-matching that text, and two call sites reached opposite
    /// verdicts on the same error (#2553). Choosing the variant that says
    /// what happened removes the need to guess.
    #[error(
        "custom-family inner solve did not converge after {cycles} cycle(s) [{}] \
         ({}); \
         refusing to expose profile objective derivatives for theta_dim={theta_dim} \
         (rho_dim={rho_dim}, psi_dim={psi_dim}). The analytic outer gradient/Hessian \
         require the inner KKT equation F_beta(beta, theta)=0; returning a value with \
         zero or shape-only derivatives is mathematically inconsistent. This trial \
         point is infeasible; the outer search may step away from it.",
        match terminal {
            Some(state) => state.to_string(),
            None => "no terminal convergence state was recorded".to_string(),
        },
        render_projected_kkt_comparison(*kkt_residual, *kkt_tol)
    )]
    InnerSolveNotConverged {
        cycles: usize,
        /// The decision variables the inner loop's verdict was taken on. See
        /// [`InnerConvergenceTerminalState`] — a cycle count alone cannot say
        /// which conjunct of the convergence test failed.
        terminal: Option<InnerConvergenceTerminalState>,
        /// Sup-norm of the projected KKT residual at the terminal inner iterate,
        /// i.e. the quantity this refusal was decided against. A cycle count
        /// alone cannot distinguish a solve that ran out of budget one order
        /// from its tolerance — where the budget is the thing to look at — from
        /// one sitting many orders away, which is a stalled or diverging solve
        /// and a different defect entirely. `None` when the producing solver
        /// path emits no typed KKT diagnostic (blockwise NR fallback,
        /// eager-stop), which is itself worth seeing in the refusal.
        kkt_residual: Option<f64>,
        /// The stationarity tolerance `kkt_residual` was compared against.
        kkt_tol: Option<f64>,
        theta_dim: usize,
        rho_dim: usize,
        psi_dim: usize,
    },
    #[error("{reason}")]
    BasisDecompositionFailed { reason: String },
    /// Pre-fit cross-block identifiability audit refused the fit. The
    /// joint design across `ParameterBlockSpec`s carries a rank
    /// deficiency that the post-`joint_null_rotation` absorption did
    /// not resolve: two or more blocks contribute the same direction,
    /// or a structural >2-way alias was detected without per-pair
    /// attribution. The full `IdentifiabilityAudit` is held so
    /// consumers (logs, structured-error sinks, the seed driver's
    /// classifier) can extract the alias pairs and the summary string
    /// without reparsing.
    #[error("identifiability audit refused the fit: {}", audit.summary)]
    IdentifiabilityFailure { audit: IdentifiabilityAudit },
    /// MAP estimate uniqueness condition `ker(J^T W J) ∩ ker(S) = {0}` is
    /// violated.  A null direction of `J^T W J` carries zero penalty
    /// curvature, so the posterior is flat along that direction and the
    /// MAP is non-unique.  The structured [`MapUniquenessError`] names the
    /// dominant block so the caller can add the missing penalty or remove
    /// the unpenalised direction.
    #[error("MAP estimate non-unique: {}", error)]
    MapUniquenessFailure { error: MapUniquenessError },
    /// A numerical verdict the inner solve reached AT ONE TRIAL POINT: no
    /// Laplace mode here, this active face's curvature refuses certification
    /// here, this quadratic subproblem is degenerate here.
    ///
    /// Like [`Self::InnerSolveNotConverged`] this is a statement about one
    /// `theta`, not about the problem — an indefinite coefficient point at one
    /// rho is an ordinary Laplace mode at another — so the outer search should
    /// reject the trial and step away, which is what the inner solver's own
    /// logs say should happen. It is a separate variant because
    /// `InnerSolveNotConverged` carries a fixed cycles/theta_dim/rho_dim/psi_dim
    /// shape and a message specifically about refusing to expose profile
    /// derivatives; reusing it for a curvature refusal would state something
    /// untrue.
    #[error("inner solve refused this trial point: {reason}")]
    TrialPointRefused { reason: String },
}

impl CustomFamilyError {
    /// A numerical refusal raised while evaluating at one trial point.
    ///
    /// The named constructor exists so a boundary that *knows* it is reporting
    /// a rho-local failure can say so, rather than leaning on the blanket
    /// `From<String>` below and hoping its default is right.
    pub fn trial_point(reason: impl Into<String>) -> Self {
        Self::TrialPointRefused {
            reason: reason.into(),
        }
    }
}

impl From<String> for CustomFamilyError {
    /// # Why this lands on `TrialPointRefused` and not `InvalidInput`
    ///
    /// A `String` cannot carry the one bit the outer smoothing search needs —
    /// is this failure a property of the trial point, or of the problem? — so
    /// any conversion from it must answer by default. This one used to answer
    /// `InvalidInput`, the variant [`Self::is_trial_point_infeasible`] returns
    /// `false` for, and gam-custom-family's inner solver reports *every*
    /// refusal as `Err(String)`. So "there is no Laplace mode at this rho", a
    /// verdict about one rho, was graded fatal and killed the whole fit at the
    /// first probe, at an optimizer whose seed loop has the correct branch one
    /// line above the one it took (gam#2590).
    ///
    /// The default is not a coin flip, because the two mistakes are not
    /// comparable:
    ///
    /// * A structural failure graded rho-local recurs at every probed rho. The
    ///   seed loop exhausts, the run still fails, and it fails quoting this
    ///   same reason — after a bounded number of cheap, identical inner
    ///   failures.
    /// * A rho-local refusal graded structural aborts a fit that was
    ///   perfectly fittable one rho away. Measured twice: #2553, #2590.
    ///
    /// So where the type system forces a guess, the guess must be
    /// "trial point". Where a caller knows better in either direction, it
    /// should construct the variant it means — [`Self::trial_point`] or the
    /// structural variant — instead of routing through here.
    fn from(value: String) -> Self {
        Self::TrialPointRefused { reason: value }
    }
}

impl From<CustomFamilyError> for String {
    fn from(value: CustomFamilyError) -> Self {
        value.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_absences_are_not_reported_as_a_comparison_2600() {
        // `|r|_inf=None against tol=None` reads as a measurement that came out
        // badly. Nothing was measured, and the message has to say which side is
        // missing: "no diagnostic on this path" and "the residual is 4e5x tol"
        // call for different next steps.
        let absent = CustomFamilyError::InnerSolveNotConverged {
            cycles: 53,
            terminal: None,
            kkt_residual: None,
            kkt_tol: None,
            theta_dim: 3,
            rho_dim: 3,
            psi_dim: 0,
        };
        let msg = absent.to_string();
        assert!(
            !msg.contains("None against"),
            "two absences must not be laid out as a comparison: {msg}"
        );
        assert!(
            msg.contains("emits no typed projected-KKT diagnostic"),
            "the message must name the absence as an absence: {msg}"
        );

        // With both present it still reads as the comparison it is.
        let measured = CustomFamilyError::InnerSolveNotConverged {
            cycles: 53,
            terminal: None,
            kkt_residual: Some(1.906428e0),
            kkt_tol: Some(8.307952e-4),
            theta_dim: 3,
            rho_dim: 3,
            psi_dim: 0,
        };
        let msg = measured.to_string();
        assert!(
            msg.contains("|r|_inf=1.906428e0 against tol=8.307952e-4"),
            "a real comparison must still render as one: {msg}"
        );

        // A half-present pair names WHICH half is missing rather than printing
        // `Some(..)`/`None` and leaving the reader to work it out.
        let half = CustomFamilyError::InnerSolveNotConverged {
            cycles: 7,
            terminal: None,
            kkt_residual: Some(4.069e3),
            kkt_tol: None,
            theta_dim: 1,
            rho_dim: 1,
            psi_dim: 0,
        };
        let msg = half.to_string();
        assert!(
            msg.contains("no stationarity tolerance was recorded"),
            "a half-present pair must name the missing half: {msg}"
        );
    }

    #[test]
    fn joint_newton_terminal_state_reports_the_best_residual_not_only_the_last_2600() {
        // The #2600 shape: a solve that reached 1.578e-3 (within 1.9x of tol)
        // and then drifted for 27 cycles to a terminal 1.906e0. A reader given
        // only the terminal value concludes the solve never approached
        // stationarity; the correct reading is that it did and left. Both
        // numbers and the distance back to the best one must be in the message.
        let state = InnerConvergenceTerminalState::JointNewton {
            cycle: 52,
            stationarity_residual: 1.906428e0,
            residual_tol: 8.307952e-4,
            step_inf: 4.958893e0,
            step_tol: 8.493315e-5,
            resolvable_negative_curvature: true,
            best_stationarity_residual: 1.578e-3,
            cycles_since_best_residual: 27,
        };
        let msg = state.to_string();
        assert!(
            msg.contains("stationarity_residual=1.906428e0"),
            "message: {msg}"
        );
        assert!(
            msg.contains("best_stationarity_residual=1.578000e-3"),
            "message: {msg}"
        );
        assert!(
            msg.contains("27 cycle(s) before this one"),
            "message: {msg}"
        );
    }

    #[test]
    fn invalid_input_display_contains_context_and_reason() {
        let err = CustomFamilyError::InvalidInput {
            context: "my_context",
            reason: "something broke".to_string(),
        };
        let msg = err.to_string();
        assert!(msg.contains("my_context"), "message: {msg}");
        assert!(msg.contains("something broke"), "message: {msg}");
    }

    #[test]
    fn optimization_display_contains_context_and_reason() {
        let err = CustomFamilyError::Optimization {
            context: "outer_loop",
            reason: "diverged".to_string(),
        };
        let msg = err.to_string();
        assert!(
            msg.contains("outer_loop") && msg.contains("diverged"),
            "message: {msg}"
        );
    }

    #[test]
    fn dimension_mismatch_displays_reason() {
        let err = CustomFamilyError::DimensionMismatch {
            reason: "3 vs 4".to_string(),
        };
        assert_eq!(err.to_string(), "3 vs 4");
    }

    #[test]
    fn numerical_failure_displays_reason() {
        let err = CustomFamilyError::NumericalFailure {
            reason: "NaN detected".to_string(),
        };
        assert_eq!(err.to_string(), "NaN detected");
    }

    #[test]
    fn a_string_boundary_refusal_is_recoverable_not_invalid_input() {
        // The regression this exists for (gam#2590): the refusal used to
        // arrive as `InvalidInput`, which classifies fatal, so an outer
        // optimizer explicitly built to step away from an infeasible trial
        // point aborted the whole fit at the first one it met.
        let err = CustomFamilyError::from("no Laplace mode at this rho".to_string());
        assert!(matches!(err, CustomFamilyError::TrialPointRefused { .. }));
        assert!(err.is_trial_point_infeasible());
        assert!(err.to_string().contains("no Laplace mode at this rho"));
        assert_eq!(
            CustomFamilyError::trial_point("x").to_string(),
            CustomFamilyError::from("x".to_string()).to_string(),
            "the named constructor and the blanket conversion must agree"
        );
        assert!(
            !CustomFamilyError::InvalidInput {
                context: "c",
                reason: "r".to_string(),
            }
            .is_trial_point_infeasible(),
            "`InvalidInput` must keep meaning what it says"
        );
    }

    #[test]
    fn from_custom_family_error_for_string_uses_display() {
        let err = CustomFamilyError::NumericalFailure {
            reason: "singular".to_string(),
        };
        let s = String::from(err);
        assert_eq!(s, "singular");
    }
}

impl CustomFamilyError {
    /// Whether a failure of this kind invalidates the whole outer run or
    /// only the trial point it was produced at.
    ///
    /// The producer's judgement, made once against the variant. It
    /// replaces a downstream substring match on the rendered message that
    /// classified one variant two different ways depending on which call
    /// site it crossed (#2553).
    ///
    /// The match is deliberately exhaustive with no wildcard arm: a new
    /// variant must be classified when it is added, rather than
    /// defaulting to whichever answer happens to be listed last.
    #[must_use]
    pub fn is_trial_point_infeasible(&self) -> bool {
        match self {
            // The inner solve missed its KKT condition at THIS theta. The
            // outer search can step away; the problem is fine.
            Self::InnerSolveNotConverged { .. } => true,
            // Likewise rho-local: a numerical refusal evaluated at one trial
            // point, which becomes true or false by moving theta (gam#2590).
            Self::TrialPointRefused { .. } => true,
            // Everything else is a property of the configuration, the
            // data, or the numerics, and does not become true or false by
            // moving theta.
            Self::InvalidInput { .. }
            | Self::Optimization { .. }
            | Self::DimensionMismatch { .. }
            | Self::NumericalFailure { .. }
            | Self::ConstraintViolation { .. }
            | Self::UnsupportedConfiguration { .. }
            | Self::BasisDecompositionFailed { .. }
            | Self::IdentifiabilityFailure { .. }
            | Self::MapUniquenessFailure { .. } => false,
        }
    }
}
