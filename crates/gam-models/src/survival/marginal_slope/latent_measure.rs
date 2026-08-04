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
    let resolved = resolve_latent_score_calibration_from_parts(
        &spec.z,
        &spec.weights,
        &spec.latent_z_policy,
        absorber_active,
        &marginal_design.design,
    )?;
    spec.z = resolved.calibrated_scores;
    Ok(SurvivalLatentScoreCalibration {
        per_score: resolved.per_score,
        raw_scores: resolved.raw_scores,
        conditioning: resolved.conditioning,
    })
}

/// The gate itself, over exactly the five things it reads.
///
/// Split out from the spec-shaped entry point above so it is directly
/// exercisable: the decision is about `z`, the weights, the policy, the
/// absorber flag and the marginal design, and nothing else about a survival term
/// spec bears on it.
pub(crate) struct ResolvedLatentScoreCalibration {
    pub(crate) per_score: Vec<LatentMeasureCalibration>,
    pub(crate) raw_scores: Array2<f64>,
    pub(crate) calibrated_scores: Array2<f64>,
    pub(crate) conditioning: Option<std::sync::Arc<Array2<f64>>>,
}

pub(crate) fn resolve_latent_score_calibration_from_parts(
    scores: &Array2<f64>,
    weights: &Array1<f64>,
    policy: &LatentZPolicy,
    absorber_active: bool,
    marginal_design: &DesignMatrix,
) -> Result<ResolvedLatentScoreCalibration, String> {
    let k = scores.ncols();
    let raw_scores = scores.clone();
    let conditioning = if absorber_active {
        None
    } else {
        Some(marginal_design.try_to_dense_arc("survival marginal-slope conditional latent-z gate")?)
    };

    let mut calibrations = Vec::with_capacity(k);
    let mut calibrated_scores = scores.clone();
    for col in 0..k {
        let raw = scores.column(col).to_owned();
        let decision = build_latent_measure_decision(
            &raw,
            weights,
            policy,
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
            match policy.check_mode {
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
    Ok(ResolvedLatentScoreCalibration {
        per_score: calibrations,
        raw_scores,
        calibrated_scores,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bms::{LatentZCheckMode, LatentZNormalizationMode};
    use gam_linalg::matrix::DenseDesignMatrix;

    /// Deterministic standard normals (Box–Muller over splitmix64).
    fn gaussians(n: usize, seed: u64) -> Vec<f64> {
        let mut state = seed;
        let mut unit = || {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            ((z >> 11) as f64 + 0.5) / (1u64 << 53) as f64
        };
        let mut out = Vec::with_capacity(n + 1);
        while out.len() < n {
            let u1 = unit().max(1e-12);
            let u2 = unit();
            let r = (-2.0 * u1.ln()).sqrt();
            out.push(r * (std::f64::consts::TAU * u2).cos());
            out.push(r * (std::f64::consts::TAU * u2).sin());
        }
        out.truncate(n);
        out
    }

    fn standardized(mut v: Vec<f64>) -> Vec<f64> {
        let n = v.len() as f64;
        let mean = v.iter().sum::<f64>() / n;
        let sd = (v.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / n)
            .sqrt()
            .max(1e-12);
        for value in v.iter_mut() {
            *value = (*value - mean) / sd;
        }
        v
    }

    /// `z = m·x + √(1−m²)·ζ`, exactly standard normal marginally and
    /// conditionally shifted on the marginal design's `x` column.
    fn shifted_fixture(n: usize, m: f64) -> (Array2<f64>, Array1<f64>, DesignMatrix, Vec<f64>) {
        let x = standardized(gaussians(n, 0x2768_A1));
        let zeta = standardized(gaussians(n, 0x2768_B2));
        let residual_sd = (1.0 - m * m).sqrt();
        let mut z = Array2::<f64>::zeros((n, 1));
        for row in 0..n {
            z[[row, 0]] = m * x[row] + residual_sd * zeta[row];
        }
        let mut design = Array2::<f64>::ones((n, 2));
        for row in 0..n {
            design[[row, 1]] = x[row];
        }
        (
            z,
            Array1::<f64>::ones(n),
            DesignMatrix::Dense(DenseDesignMatrix::from(design)),
            zeta,
        )
    }

    fn auto_policy(check_mode: LatentZCheckMode) -> LatentZPolicy {
        LatentZPolicy {
            check_mode,
            normalization: LatentZNormalizationMode::Frozen { mean: 0.0, sd: 1.0 },
            ..LatentZPolicy::frozen_transformation_normal()
        }
    }

    /// The gate must fire on a conditionally shifted score, and the score it
    /// hands the kernel must be the CLEAN one — not merely centred (gam#2768).
    ///
    /// Recovering `ζ` up to sign and a common scale is the whole claim: the fit's
    /// coefficients are then the ones the outcome was generated with, and the
    /// influence channel `q` no longer carries `b(C)·m(C)`.
    #[test]
    fn survival_gate_recovers_the_conditionally_standardized_score() {
        let n = 4000;
        let m = 0.6;
        let (z, weights, design, zeta_truth) = shifted_fixture(n, m);
        let resolved = resolve_latent_score_calibration_from_parts(
            &z,
            &weights,
            &auto_policy(LatentZCheckMode::WarnOnly),
            false,
            &design,
        )
        .expect("gate");

        assert_eq!(resolved.per_score.len(), 1, "one decision per score column");
        assert!(
            matches!(
                resolved.per_score[0],
                LatentMeasureCalibration::ConditionalLocationScale(_)
            ),
            "the E[z|C] Rao gate must escalate to the conditional location-scale \
             correction at Corr(z, x) = {m} and n = {n}; got {}",
            calibration_label(&resolved.per_score[0])
        );

        // The calibrated column must be ζ itself. Correlation, because the
        // calibration is fit rather than known and carries estimation error in
        // m̂ and in the residual scale.
        let calibrated = resolved.calibrated_scores.column(0);
        let dot: f64 = calibrated
            .iter()
            .zip(zeta_truth.iter())
            .map(|(a, b)| a * b)
            .sum();
        let norm_cal = calibrated.iter().map(|v| v * v).sum::<f64>().sqrt();
        let norm_truth = zeta_truth.iter().map(|v| v * v).sum::<f64>().sqrt();
        let correlation = dot / (norm_cal * norm_truth);
        assert!(
            correlation > 0.9995,
            "the calibrated score must BE the clean score; correlation {correlation:.6}"
        );

        // Unit variance, which is what keeps `q` the marginal index.
        let mean = calibrated.iter().sum::<f64>() / n as f64;
        let sd = (calibrated.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n as f64).sqrt();
        assert!(
            mean.abs() < 0.02 && (sd - 1.0).abs() < 0.05,
            "calibrated score must be standardised; mean={mean:.4} sd={sd:.4}"
        );

        // The raw score is kept for the generated-regressor correction, which
        // differentiates the FIRST stage and therefore needs the axis that stage
        // regressed — not the one the fit consumed.
        assert_eq!(resolved.raw_scores, z, "the raw score must survive the gate");
        assert!(
            resolved.conditioning.is_some(),
            "the conditional branch must retain the block it conditioned on"
        );
    }

    /// The gate must NOT fire on a score that is already conditionally standard
    /// normal. A trigger-happy gate would redefine the latent axis of every
    /// clean fit, which is a worse failure than the one it exists to prevent.
    #[test]
    fn survival_gate_stays_quiet_on_an_unshifted_score() {
        let n = 4000;
        let (_, weights, design, zeta) = shifted_fixture(n, 0.6);
        let mut clean = Array2::<f64>::zeros((n, 1));
        for row in 0..n {
            clean[[row, 0]] = zeta[row];
        }
        let resolved = resolve_latent_score_calibration_from_parts(
            &clean,
            &weights,
            &auto_policy(LatentZCheckMode::WarnOnly),
            false,
            &design,
        )
        .expect("gate");
        assert!(
            matches!(resolved.per_score[0], LatentMeasureCalibration::None),
            "got {}",
            calibration_label(&resolved.per_score[0])
        );
        assert_eq!(
            resolved.calibrated_scores, clean,
            "an unfired gate must leave the score untouched, byte for byte"
        );
    }

    /// The #461 absorber seam: with a CTN Stage-1 influence absorber active the
    /// conditional leakage is already absorbed, and replacing z would perturb the
    /// widened-marginal predict seam. The conditional branch must then be
    /// unreachable even on a score that would otherwise fire it.
    #[test]
    fn survival_gate_does_not_condition_behind_an_active_influence_absorber() {
        let n = 4000;
        let (z, weights, design, _) = shifted_fixture(n, 0.6);
        let resolved = resolve_latent_score_calibration_from_parts(
            &z,
            &weights,
            &auto_policy(LatentZCheckMode::WarnOnly),
            true,
            &design,
        )
        .expect("gate");
        assert!(
            !matches!(
                resolved.per_score[0],
                LatentMeasureCalibration::ConditionalLocationScale(_)
            ),
            "the conditional branch must be suppressed behind an active absorber"
        );
        assert!(
            resolved.conditioning.is_none(),
            "no conditioning block may be built when the branch is suppressed"
        );
    }

    /// Every latent-score column is gated, not just the primary one: with `K > 1`
    /// the leakage is a SUM of per-coordinate conditional shifts, so a gate that
    /// only looked at column 0 would leave the rest of it in `q`.
    #[test]
    fn survival_gate_covers_every_score_column() {
        let n = 4000;
        let m = 0.6;
        let (first, weights, design, zeta) = shifted_fixture(n, m);
        let mut two = Array2::<f64>::zeros((n, 2));
        two.column_mut(0).assign(&first.column(0));
        // A second column shifted the OTHER way, so a gate that reused column
        // 0's calibration would leave a visible residual correlation.
        let residual_sd = (1.0 - m * m).sqrt();
        let x: Vec<f64> = (0..n)
            .map(|row| (first[[row, 0]] - residual_sd * zeta[row]) / m)
            .collect();
        let other = standardized(gaussians(n, 0x2768_C3));
        for row in 0..n {
            two[[row, 1]] = -m * x[row] + residual_sd * other[row];
        }
        let resolved = resolve_latent_score_calibration_from_parts(
            &two,
            &weights,
            &auto_policy(LatentZCheckMode::WarnOnly),
            false,
            &design,
        )
        .expect("gate");
        assert_eq!(resolved.per_score.len(), 2);
        for (column, calibration) in resolved.per_score.iter().enumerate() {
            assert!(
                matches!(
                    calibration,
                    LatentMeasureCalibration::ConditionalLocationScale(_)
                ),
                "column {column} must be gated on its own conditional moments; got {}",
                calibration_label(calibration)
            );
        }
        // Both calibrated columns must be conditionally uncorrelated with x.
        for column in 0..2 {
            let calibrated = resolved.calibrated_scores.column(column);
            let cov: f64 = calibrated
                .iter()
                .zip(x.iter())
                .map(|(z, x)| z * x)
                .sum::<f64>()
                / n as f64;
            assert!(
                cov.abs() < 0.03,
                "column {column} still carries a conditional shift: Cov(ζ, x) = {cov:.4}"
            );
        }
    }
}
