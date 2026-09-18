//! The joint event model's production path (#2961, #2966): fit a cohort,
//! condition on a new history, forecast, save and reload. gam-cli, gam-pyffi
//! and gamfit call these functions.
//!
//! A saved model is the shared event-model envelope of kind `joint`, holding
//! the mark vocabulary and the posterior that forecasting integrates, never
//! the training records. Save → reload → forecast reproduces the in-memory
//! forecast bit for bit.

use super::law::JointLikelihood;
use super::constant_rate_inference::ConstantRatePosterior;
use super::law::{JointSpecification, invalid};
use gam_model_api::saved_model::{
    SavedModelError, read_saved_model_file, read_saved_model_text, saved_model_text,
    write_saved_model,
};
use crate::{EventHistoryError, MarkKind, SubjectHistory};
use ndarray::Array2;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Version of the saved joint event model. Version 2 saves the full law
/// specification, so version 1 payloads are refused.
const JOINT_EVENT_MODEL_VERSION: u64 = 2;

/// Kind of the saved joint event model in the shared saved-model envelope.
const JOINT_EVENT_MODEL_KIND: &str = "joint";

/// The posterior a model integrates when it forecasts.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
enum JointModelLaw {
    /// No latent signatures: independent exact Gamma rate posteriors.
    RankZero(ConstantRatePosterior),
}

/// A fitted joint event model.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct JointEventModel {
    mark_names: Vec<String>,
    specification: JointSpecification,
    law: JointModelLaw,
}

/// A model conditioned on one history: the posterior given the training cohort
/// and that history, and the risk set the history leaves open at its exit.
pub struct ConditionedJointModel<'m> {
    model: &'m JointEventModel,
    law: JointModelLaw,
    at_risk: Vec<bool>,
}

/// Posterior-predictive absolute risks after a history's exit `s`.
#[derive(Clone, Debug, PartialEq)]
pub struct JointForecast {
    /// Offsets `u` after the exit.
    pub horizons: Vec<f64>,
    /// `P(T_dagger > s+u | H_s)`: no terminal mark fires by each horizon.
    pub survival: Vec<f64>,
    /// `P(s < T_d <= s+u, T_d < T_dagger | H_s)` for every mark's next
    /// occurrence (horizons × marks); zero for a once-only mark that has fired.
    pub incidence: Array2<f64>,
    /// A bound on each incidence's numerical error; zero where it is closed form.
    pub incidence_error: Array2<f64>,
}

fn validate_mark_names(
    names: &[String],
    specification: &JointSpecification,
) -> Result<(), EventHistoryError> {
    if names.len() != specification.marks.len() {
        return Err(invalid(format!(
            "{} mark names for {} mark kinds",
            names.len(),
            specification.marks.len()
        )));
    }
    let mut sorted: Vec<&String> = names.iter().collect();
    sorted.sort();
    if let Some(pair) = sorted.windows(2).find(|pair| pair[0] == pair[1]) {
        return Err(invalid(format!("duplicate mark name {:?}", pair[0])));
    }
    Ok(())
}

/// Fit the joint event model to a cohort of follow-up records.
pub fn fit_joint_event_model(
    mark_names: Vec<String>,
    mark_kinds: Vec<MarkKind>,
    subjects: &[SubjectHistory],
) -> Result<JointEventModel, EventHistoryError> {
    let specification = JointSpecification::new(mark_kinds)?;
    validate_mark_names(&mark_names, &specification)?;
    if subjects.is_empty() {
        return Err(invalid("the joint event model needs at least one subject"));
    }
    let validated = JointLikelihood::new(specification.clone())?;
    let posterior = ConstantRatePosterior::infer(
        &specification.marks,
        subjects.iter().map(|subject| {
            specification
                .history(subject)
                .and_then(|history| validated.validate_history(&history).map(|()| history))
        }),
    )?;
    Ok(JointEventModel {
        mark_names,
        specification,
        law: JointModelLaw::RankZero(posterior),
    })
}

impl JointEventModel {
    pub fn mark_names(&self) -> &[String] {
        &self.mark_names
    }

    pub fn mark_kinds(&self) -> &[MarkKind] {
        &self.specification.marks
    }

    /// Condition on one history, which ends at its exit `s`.
    pub fn condition(
        &self,
        history: &SubjectHistory,
    ) -> Result<ConditionedJointModel<'_>, EventHistoryError> {
        let marks = &self.specification.marks;
        let nodes = self.specification.history(history)?;
        JointLikelihood::new(self.specification.clone())?.validate_history(&nodes)?;
        let at_risk = nodes.open_risk_set(marks).ok_or_else(|| {
            invalid(format!(
                "subject {:?}: a terminal event ended this history, so there is nothing to forecast",
                history.id
            ))
        })?;
        let law = match &self.law {
            JointModelLaw::RankZero(posterior) => {
                JointModelLaw::RankZero(posterior.condition(marks, &nodes)?)
            }
        };
        Ok(ConditionedJointModel {
            model: self,
            law,
            at_risk,
        })
    }

    /// Save the model to `path`, atomically.
    pub fn save(&self, path: &Path) -> Result<(), SavedModelError> {
        write_saved_model(path, &self.saved_text()?)
    }

    /// Load a saved model, refusing another kind or version, or a state its
    /// law cannot hold.
    pub fn load(path: &Path) -> Result<Self, SavedModelError> {
        Self::from_saved_text(&read_saved_model_file(path)?)
    }

    fn saved_text(&self) -> Result<String, SavedModelError> {
        saved_model_text(JOINT_EVENT_MODEL_KIND, JOINT_EVENT_MODEL_VERSION, self)
    }

    fn from_saved_text(text: &str) -> Result<Self, SavedModelError> {
        let model: Self =
            read_saved_model_text(text, JOINT_EVENT_MODEL_KIND, JOINT_EVENT_MODEL_VERSION)?;
        let inconsistent =
            |reason: EventHistoryError| SavedModelError::Inconsistent { reason: Box::new(reason) };
        let specification =
            JointSpecification::new(model.specification.marks.clone()).map_err(inconsistent)?;
        // A rank-zero law holds exactly the rank-zero specification of its marks.
        if model.specification != specification {
            return Err(inconsistent(invalid(
                "a rank-zero joint model holds only the rank-zero specification of its marks",
            )));
        }
        validate_mark_names(&model.mark_names, &specification).map_err(inconsistent)?;
        match &model.law {
            JointModelLaw::RankZero(posterior) => {
                posterior.validate(&specification.marks).map_err(inconsistent)?
            }
        }
        Ok(model)
    }
}

impl ConditionedJointModel<'_> {
    /// Forecast `horizons` after the history's exit.
    pub fn forecast(&self, horizons: &[f64]) -> Result<JointForecast, EventHistoryError> {
        match &self.law {
            JointModelLaw::RankZero(posterior) => {
                posterior.forecast(&self.model.specification.marks, &self.at_risk, horizons)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Event;

    fn subject(id: &str, entry: f64, exit: f64, events: &[(f64, usize)]) -> SubjectHistory {
        SubjectHistory {
            id: id.to_string(),
            entry,
            exit,
            events: events
                .iter()
                .map(|&(time, mark)| Event { time, mark })
                .collect(),
            segments: Vec::new(),
        }
    }

    /// A once-only diagnosis, two terminal causes of death and a recurrent visit.
    fn competing_model() -> JointEventModel {
        fit_joint_event_model(
            vec![
                "diagnosis".to_string(),
                "cvd_death".to_string(),
                "other_death".to_string(),
                "visit".to_string(),
            ],
            vec![
                MarkKind::Once,
                MarkKind::Terminal,
                MarkKind::Terminal,
                MarkKind::Recurrent,
            ],
            &[
                subject("a", 0.0, 4.0, &[(4.0, 1)]),
                subject("b", 0.0, 6.0, &[(0.0, 0), (1.0, 3), (5.0, 3)]),
            ],
        )
        .unwrap()
    }

    fn bits(forecast: &JointForecast) -> Vec<u64> {
        forecast
            .horizons
            .iter()
            .chain(&forecast.survival)
            .chain(forecast.incidence.iter())
            .chain(forecast.incidence_error.iter())
            .map(|v| v.to_bits())
            .collect()
    }

    #[test]
    fn save_reload_forecast_is_bit_identical_and_other_versions_are_refused() {
        let model = competing_model();
        let history = subject("new", 0.5, 2.5, &[(1.0, 3)]);
        let horizons = [0.25, 3.0, 40.0];
        let before = model.condition(&history).unwrap().forecast(&horizons).unwrap();
        assert!(before.incidence_error.iter().any(|&e| e > 0.0));
        let path = std::env::temp_dir().join(format!("gam-joint-model-{}.json", std::process::id()));
        model.save(&path).unwrap();
        let reloaded = JointEventModel::load(&path).unwrap();
        std::fs::remove_file(&path).unwrap();
        assert_eq!(reloaded, model);
        let after = reloaded.condition(&history).unwrap().forecast(&horizons).unwrap();
        assert_eq!(bits(&after), bits(&before));

        let text = model.saved_text().unwrap();
        assert_eq!(reloaded.saved_text().unwrap(), text);
        assert!(!text.contains("\"a\"") && !text.contains("\"b\""));
        let version = format!("\"version\": {JOINT_EVENT_MODEL_VERSION}");
        assert!(text.contains(&version) && text.contains("\"kind\": \"joint\""));
        assert!(matches!(
            JointEventModel::from_saved_text(&text.replace(&version, "\"version\": 0")),
            Err(SavedModelError::Version { found: Some(0), expected: JOINT_EVENT_MODEL_VERSION, .. })
        ));
        // A version 1 payload, which held only slice 0's rank-zero specification, is refused.
        let previous = format!("\"version\": {}", JOINT_EVENT_MODEL_VERSION - 1);
        assert!(matches!(
            JointEventModel::from_saved_text(&text.replace(&version, &previous)),
            Err(SavedModelError::Version { found: Some(1), expected: JOINT_EVENT_MODEL_VERSION, .. })
        ));
        assert!(matches!(
            JointEventModel::from_saved_text(&text.replace(&format!("{version},"), "")),
            Err(SavedModelError::Version { found: None, .. })
        ));
        assert!(matches!(
            JointEventModel::from_saved_text(&text.replace("\"kind\": \"joint\"", "\"kind\": \"event_history\"")),
            Err(SavedModelError::Kind { .. })
        ));
        assert!(text.contains("\"shape\": 1.0"));
        assert!(matches!(
            JointEventModel::from_saved_text(&text.replacen("\"shape\": 1.0", "\"shape\": 0.5", 1)),
            Err(SavedModelError::Inconsistent { .. })
        ));
        // A rank-zero law refuses any specification beyond its marks' rank-zero one.
        assert!(text.contains("\"population_columns\": 1"));
        assert!(matches!(
            JointEventModel::from_saved_text(
                &text.replacen("\"population_columns\": 1", "\"population_columns\": 2", 1)
            ),
            Err(SavedModelError::Inconsistent { .. })
        ));
    }

    #[test]
    fn conditioning_follows_the_history_risk_set() {
        let model = competing_model();
        // A fired diagnosis has no next occurrence to forecast; death still does.
        let diagnosed = model
            .condition(&subject("dx", 0.0, 2.0, &[(1.0, 0)]))
            .unwrap()
            .forecast(&[3.0])
            .unwrap();
        assert_eq!(diagnosed.incidence[[0, 0]], 0.0);
        assert!(diagnosed.incidence[[0, 1]] > 0.0 && diagnosed.incidence[[0, 3]] > 0.0);
        let error = model
            .condition(&subject("dead", 0.0, 2.0, &[(2.0, 2)]))
            .err()
            .unwrap()
            .to_string();
        assert!(error.contains("terminal event ended"), "{error}");
        let error = model
            .condition(&subject("late", 0.0, 2.0, &[]))
            .unwrap()
            .forecast(&[-1.0])
            .err()
            .unwrap()
            .to_string();
        assert!(error.contains("nonnegative"), "{error}");
    }

    #[test]
    fn fits_refuse_an_inconsistent_vocabulary() {
        let cohort = [subject("a", 0.0, 1.0, &[])];
        for (names, kinds, reason) in [
            (vec!["x".to_string()], vec![MarkKind::Once, MarkKind::Terminal], "mark names for"),
            (
                vec!["x".to_string(), "x".to_string()],
                vec![MarkKind::Once, MarkKind::Terminal],
                "duplicate mark name",
            ),
        ] {
            let error = fit_joint_event_model(names, kinds, &cohort).err().unwrap().to_string();
            assert!(error.contains(reason), "{error}");
        }
        let error = fit_joint_event_model(vec!["x".to_string()], vec![MarkKind::Terminal], &[])
            .err()
            .unwrap()
            .to_string();
        assert!(error.contains("at least one subject"), "{error}");
    }

    #[test]
    fn a_shared_horizon_forecast_is_bit_identical_whatever_other_horizons_are_requested() {
        // #2963 (A8): horizons are evaluation points, never accuracy controls.
        let model = competing_model();
        let conditioned = model.condition(&subject("new", 0.0, 2.0, &[])).unwrap();
        let single = conditioned.forecast(&[7.0]).unwrap();
        let several = conditioned.forecast(&[0.5, 7.0, 40.0, 400.0]).unwrap();
        assert_eq!(single.survival[0].to_bits(), several.survival[1].to_bits());
        for d in 0..model.mark_kinds().len() {
            assert_eq!(single.incidence[[0, d]].to_bits(), several.incidence[[1, d]].to_bits());
            assert_eq!(
                single.incidence_error[[0, d]].to_bits(),
                several.incidence_error[[1, d]].to_bits()
            );
        }
        // The diagnosis goes through the bracket, so its error term is live in the comparison.
        assert!(single.incidence_error[[0, 0]] > 0.0);
        // Positive control: a different horizon changes the values this test compares.
        assert_ne!(single.survival[0].to_bits(), several.survival[2].to_bits());
        assert_ne!(single.incidence[[0, 0]].to_bits(), several.incidence[[2, 0]].to_bits());
    }

    #[test]
    fn extending_a_history_after_the_assessment_time_never_changes_its_forecast() {
        // #2962 (A7): a forecast made at s sees exactly what was known at s.
        let model = competing_model();
        let kinds = model.mark_kinds().to_vec();
        let horizons = [0.5, 4.0, 30.0];
        // Alive, undiagnosed, with one visit, at the assessment time 3.
        let at_cutoff = model
            .condition(&subject("p", 0.0, 3.0, &[(1.5, 3)]))
            .unwrap()
            .forecast(&horizons)
            .unwrap();
        for later in [
            subject("p", 0.0, 5.0, &[(1.5, 3)]),
            subject("p", 0.0, 6.0, &[(1.5, 3), (4.0, 0), (4.5, 3)]),
            subject("p", 0.0, 9.0, &[(1.5, 3), (4.0, 0), (9.0, 2)]),
        ] {
            let cut = later.prefix(3.0, &kinds).unwrap();
            let forecast = model.condition(&cut).unwrap().forecast(&horizons).unwrap();
            assert_eq!(bits(&forecast), bits(&at_cutoff));
        }
        // Positive controls. A diagnosis AT the cutoff is part of H_s and closes that mark's risk.
        let diagnosed_at_cutoff = subject("p", 0.0, 6.0, &[(1.5, 3), (3.0, 0)]).prefix(3.0, &kinds).unwrap();
        let closed = model
            .condition(&diagnosed_at_cutoff)
            .unwrap()
            .forecast(&horizons)
            .unwrap();
        assert!(closed.incidence.column(0).iter().all(|&p| p == 0.0));
        assert_ne!(bits(&closed), bits(&at_cutoff));
        // A later cutoff adds exposure, so the comparison does see conditioning.
        let later_cut = subject("p", 0.0, 6.0, &[(1.5, 3), (4.0, 0)]).prefix(3.5, &kinds).unwrap();
        let moved = model.condition(&later_cut).unwrap().forecast(&horizons).unwrap();
        assert_ne!(bits(&moved), bits(&at_cutoff));
    }
}
