//! The automatic latent-score measure gate for the survival marginal-slope
//! family (gam#2768).
//!
//! The Bernoulli marginal-slope family has run an automatic gate on its latent
//! score since #905: a Rao score test on `E[z|C]` and `Var(z|C)` over the
//! marginal-index span, escalating to the conditional location-scale correction
//! `ζ = (z − m(C))/√v(C)` when it fires, with rank inverse-normal and empirical
//! fallbacks below it. The survival marginal-slope family ran *none* of it. It
//! called `standardize_latent_z_with_policy` and nothing else, and under the
//! default policy — `Frozen { mean: 0, sd: 1 }` — that transform is the
//! identity: it checked, it warned, and it passed z through unchanged.
//!
//! That is not a cosmetic gap. The survival row index is
//! `η = q·c(g) + s(g)·z`, so a conditional shift `E[z|C] = m(C) ≠ 0` puts
//! `s(g(C))·m(C)` into the *influence* channel `q` — the same `b(C)·m(C)`
//! leakage the Bernoulli gate exists to remove, in a model whose whole point is
//! that `q` is the marginal index. The pooled marginal gate cannot see it (the
//! marginal law of z can be exactly N(0,1) while every conditional law is
//! shifted), and rank-INT provably cannot fix it (no transform depending only on
//! the marginal `F_Z` can enforce `E[T(Z)|C] ≡ const`).
//!
//! This module is the survival caller of the *shared* gate
//! ([`build_latent_measure_decision`]), not a second copy of it. The one thing
//! it declares that BMS does not is the family's kernel capability: the survival
//! row program is the closed-form standard-normal probit lowering and owns no
//! empirical-grid branch, so it asks for
//! [`EmpiricalLatentMeasureSupport::StandardNormalOnly`] and routes the gate's
//! residual verdict through the spec's own [`LatentZCheckMode`].

use super::*;

use crate::bms::{
    EmpiricalLatentMeasureSupport, LatentMeasureCalibration, LatentMeasureKind, LatentZCheckMode,
    LatentZConditionalCalibration, build_latent_measure_decision,
};

/// Everything the fit and its persistence need from the automatic gate: the
/// per-coordinate decisions, the score the gate saw *before* calibrating it, and
/// the conditioning block it conditioned on.
///
/// The raw score and the conditioning block are not diagnostics. The
/// Murphy-Topel generated-regressor correction needs both — it differentiates
/// the first stage, which regressed the RAW score on that block — and the
/// conditioning block is also what the persistence gate compares against the
/// finally-resolved marginal design to decide whether prediction can reproduce
/// the map at all.
pub(crate) struct SurvivalLatentScoreCalibration {
    /// One decision per latent-score column, in column order.
    pub(crate) per_score: Vec<LatentMeasureCalibration>,
    /// The (normalised, pre-calibration) latent scores the gate was handed.
    pub(crate) raw_scores: Array2<f64>,
    /// The conditioning span `a(C)` the conditional branch used, when it was
    /// built. `None` when the CTN Stage-1 absorber suppressed it.
    pub(crate) conditioning: Option<std::sync::Arc<Array2<f64>>>,
}

impl SurvivalLatentScoreCalibration {
    /// The conditional location-scale calibration on the PRIMARY score, if the
    /// Rao gate escalated to one. This is the only branch with a generated
    /// regressor: rank-INT is a fixed monotone map of the marginal ECDF and the
    /// identity is not estimated at all.
    pub(crate) fn primary_conditional(&self) -> Option<&LatentZConditionalCalibration> {
        match self.per_score.first() {
            Some(LatentMeasureCalibration::ConditionalLocationScale(cal)) => Some(cal),
            _ => None,
        }
    }
}

/// Run the automatic latent-measure gate over every latent-score coordinate and
/// replace `spec.z` by the calibrated score in place.
///
/// Returns one [`LatentMeasureCalibration`] per z column, in column order, for
/// persistence: prediction MUST apply the identical map, so the fit's decision
/// travels with the model rather than being re-derived from the prediction
/// sample.
///
/// # Why per coordinate
///
/// With `K > 1` latent scores the row index is `η = q·c + Σ_k s(g_k)·z_k`, so
/// the leakage is `Σ_k s(g_k(C))·m_k(C)` — a sum of per-coordinate conditional
/// shifts. Under the current single global score covariance `Σ`, the
/// K-generalisation of the correction is therefore the per-coordinate
/// conditional standardisation on the same basis `a(C)`, after which `Σ` is
/// recomputed from the calibrated scores by the caller. (A conditional `Σ(C)` is
/// a different and larger question; it is gam#2766.)
pub(crate) fn resolve_survival_latent_score_calibration(
    spec: &mut SurvivalMarginalSlopeTermSpec,
    marginal_design: &TermCollectionDesign,
) -> Result<SurvivalLatentScoreCalibration, String> {
    let k = spec.z.ncols();
    let raw_scores = spec.z.clone();
    // #461 seam, mirrored from BMS: when a CTN Stage-1 influence absorber is
    // active the conditional leakage is already absorbed by the absorber's own
    // orthogonalisation, and replacing z here would perturb the widened-marginal
    // predict seam. The conditional gate is then not engaged; the pooled gates
    // below it still are, because they are about the marginal law of z and the
    // absorber says nothing about that.
    let absorber_active = spec
        .score_influence_jacobian
        .as_ref()
        .is_some_and(|jacobian| jacobian.ncols() > 0);
    let conditioning = if absorber_active {
        None
    } else {
        Some(
            marginal_design
                .design
                .try_to_dense_arc("survival marginal-slope conditional latent-z gate")?,
        )
    };

    let mut calibrations = Vec::with_capacity(k);
    let mut calibrated_scores = spec.z.clone();
    for col in 0..k {
        let raw = spec.z.column(col).to_owned();
        let decision = build_latent_measure_decision(
            &raw,
            &spec.weights,
            &spec.latent_z_policy,
            conditioning.as_ref().map(|design| design.view()),
            EmpiricalLatentMeasureSupport::StandardNormalOnly,
            "survival-marginal-slope",
        )?;
        if !matches!(decision.kind, LatentMeasureKind::StandardNormal) {
            // Unreachable by construction — `StandardNormalOnly` never returns
            // another kind — but this family's whole kernel rests on it, so the
            // invariant is checked rather than assumed.
            return Err(
                "survival marginal-slope latent-measure gate returned a non-standard-normal \
                 measure for a standard-normal-only kernel"
                    .to_string(),
            );
        }
        if let Some(adequacy) = decision.unmodelled_residual.as_ref() {
            let message = format!(
                "survival-marginal-slope latent score column {col} still fails the \
                 standard-normal adequacy gate after the automatic {} calibration, and this \
                 family's row kernel has no empirical latent measure to carry the residual law: \
                 the closed-form standard-normal probit kernel is being applied to a sample the \
                 gate rejects. Point estimation still uses the calibrated axis (it is the closest \
                 available to the kernel's own assumption); what is unmodelled is the residual \
                 SHAPE. Adequacy ledger (x = statistic / bound, x<=1 passed): {}",
                calibration_label(&decision.calibration),
                adequacy.ledger(),
            );
            match spec.latent_z_policy.check_mode {
                LatentZCheckMode::Strict => return Err(message),
                LatentZCheckMode::WarnOnly => log::warn!("{message}"),
                LatentZCheckMode::Off => {}
            }
        }
        let calibrated = match &decision.calibration {
            LatentMeasureCalibration::None => raw,
            LatentMeasureCalibration::RankInverseNormal(cal) => cal.apply_to_training(&raw)?,
            LatentMeasureCalibration::ConditionalLocationScale(cal) => {
                // The conditional branch is only reachable when the gate had a
                // conditioning block to fire on, so it is present here.
                let a_block = conditioning.as_ref().ok_or_else(|| {
                    "survival marginal-slope conditional latent calibration requires the \
                     marginal conditioning block"
                        .to_string()
                })?;
                cal.apply(raw.view(), a_block.view())?
            }
        };
        if !matches!(decision.calibration, LatentMeasureCalibration::None) {
            log::info!(
                "[survival-marginal-slope latent-z] score column {col}: applied the {} \
                 calibration before any downstream consumer saw the score",
                calibration_label(&decision.calibration),
            );
        }
        calibrated_scores.column_mut(col).assign(&calibrated);
        calibrations.push(decision.calibration);
    }
    spec.z = calibrated_scores;
    Ok(SurvivalLatentScoreCalibration {
        per_score: calibrations,
        raw_scores,
        conditioning,
    })
}

fn calibration_label(calibration: &LatentMeasureCalibration) -> &'static str {
    match calibration {
        LatentMeasureCalibration::None => "identity",
        LatentMeasureCalibration::RankInverseNormal(_) => "rank inverse-normal",
        LatentMeasureCalibration::ConditionalLocationScale(_) => "conditional location-scale",
    }
}
