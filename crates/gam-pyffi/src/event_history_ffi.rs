//! Python bindings for the event-history family: fit from arrays, then
//! predict, score and read the latent state of subjects from the in-memory
//! fit.

use crate::ffi::ffi_errors::{detach_py_result, py_value_error};
use gam::families::custom_family::BlockwiseFitOptions;
use gam::families::event_history::{
    CovariateSegment, Event, EventHistoryCohort, EventHistoryFit, MarkKind, SubjectHistory,
    fit_event_history_formula, forecast, kolmogorov_smirnov_uniform, latent_exposure,
    latent_state, population_forecast, predictive_pit,
};
use ndarray::{Array1, Array2, Array3};
use numpy::{PyArray1, PyArray2, PyArray3, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyModule};
use std::sync::Arc;

/// A fitted event-history model held in memory.
#[pyclass(name = "_EventHistoryModel", frozen)]
pub(crate) struct PyEventHistoryModel {
    fit: Arc<EventHistoryFit>,
    cohort: Arc<EventHistoryCohort>,
}

impl PyEventHistoryModel {
    fn risk_dict<'py>(
        &self,
        py: Python<'py>,
        result: gam::families::event_history::RiskForecast,
    ) -> PyResult<Bound<'py, PyDict>> {
        let out = PyDict::new(py);
        out.set_item("horizons", result.horizons)?;
        out.set_item("marks", self.cohort.mark_names.clone())?;
        out.set_item("risk", PyArray2::from_owned_array(py, result.risk))?;
        out.set_item("survival", result.survival)?;
        out.set_item(
            "expected_counts",
            PyArray2::from_owned_array(py, result.expected_counts),
        )?;
        Ok(out)
    }

    fn subject(&self, index: usize) -> PyResult<SubjectHistory> {
        self.cohort
            .subjects
            .get(index)
            .cloned()
            .ok_or_else(|| py_value_error(format!("subject index {index} is out of range")))
    }
}

#[pymethods]
impl PyEventHistoryModel {
    fn mark_names(&self) -> Vec<String> {
        self.cohort.mark_names.clone()
    }

    fn mark_kinds(&self) -> Vec<String> {
        self.cohort
            .mark_kinds
            .iter()
            .map(|k| k.as_str().to_string())
            .collect()
    }

    fn covariate_names(&self) -> Vec<String> {
        self.cohort.covariate_names.clone()
    }

    fn subject_ids(&self) -> Vec<String> {
        self.cohort.subjects.iter().map(|s| s.id.clone()).collect()
    }

    fn rank(&self) -> usize {
        self.fit.rank()
    }

    fn loadings<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        PyArray2::from_owned_array(py, self.fit.loadings.clone())
    }

    fn rates(&self) -> Vec<f64> {
        self.fit.rates.clone()
    }

    fn log_rates(&self) -> Vec<f64> {
        self.fit.log_rates.clone()
    }

    fn atom_log_lambdas(&self) -> Vec<f64> {
        self.fit.atom_log_lambdas.clone()
    }

    fn atom_evidence(&self) -> Vec<f64> {
        self.fit.atom_evidence.clone()
    }

    fn rank_path<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let out = PyList::empty(py);
        for step in &self.fit.rank_path {
            let item = PyDict::new(py);
            item.set_item("rank", step.rank)?;
            item.set_item("score_eigenvalue", step.score_eigenvalue)?;
            item.set_item("proposed_log_rate", step.proposed_log_rate)?;
            item.set_item("at_resolution_limit", step.at_resolution_limit)?;
            item.set_item("evidence_gain", step.evidence_gain)?;
            item.set_item("accepted", step.accepted)?;
            out.append(item)?;
        }
        Ok(out)
    }

    fn time_scale(&self) -> f64 {
        self.fit.time_scale
    }

    fn log_likelihood(&self) -> f64 {
        self.fit.fit.log_likelihood
    }

    fn reml_score(&self) -> Option<f64> {
        self.fit.fit.reml_score()
    }

    fn outer_iterations(&self) -> usize {
        self.fit.fit.outer_iterations
    }

    fn coefficients(&self, mark: usize) -> PyResult<Vec<f64>> {
        if mark >= self.fit.marks() {
            return Err(py_value_error(format!(
                "mark index {mark} is outside the {} marks of the fit",
                self.fit.marks()
            )));
        }
        Ok(self.fit.mark_coefficients(mark).to_vec())
    }

    fn disease_covariance<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        PyArray2::from_owned_array(py, self.fit.disease_covariance())
    }

    fn temporal_covariance<'py>(&self, py: Python<'py>, lag: f64) -> Bound<'py, PyArray2<f64>> {
        PyArray2::from_owned_array(py, self.fit.temporal_covariance(lag))
    }

    fn eigenmodes<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray2<f64>>)> {
        let (values, vectors) = self.fit.eigenmodes().map_err(|e| py_value_error(e.to_string()))?;
        Ok((
            PyArray1::from_owned_array(py, values),
            PyArray2::from_owned_array(py, vectors),
        ))
    }

    /// Per-mark first-occurrence risks of one training subject beyond its exit.
    #[pyo3(signature = (subject, horizons, future_row=None))]
    fn forecast<'py>(
        &self,
        py: Python<'py>,
        subject: usize,
        horizons: Vec<f64>,
        future_row: Option<usize>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let history = self.subject(subject)?;
        let row = future_row.unwrap_or_else(|| history.segments.last().map(|s| s.row).unwrap_or(0));
        let fit = Arc::clone(&self.fit);
        let cohort = Arc::clone(&self.cohort);
        let result = detach_py_result(py, "event-history forecast", move || {
            forecast(&fit, &cohort, &history, &horizons, row).map_err(|e| e.to_string())
        })?;
        self.risk_dict(py, result)
    }

    /// Per-mark risks of a subject with no observed history from covariate
    /// values alone: the latent state starts at its stationary prior.
    fn population_forecast<'py>(
        &self,
        py: Python<'py>,
        covariates: Vec<f64>,
        start: f64,
        horizons: Vec<f64>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let fit = Arc::clone(&self.fit);
        let cohort = Arc::clone(&self.cohort);
        let result = detach_py_result(py, "event-history population forecast", move || {
            population_forecast(&fit, &cohort, &covariates, start, &horizons).map_err(|e| e.to_string())
        })?;
        self.risk_dict(py, result)
    }

    /// The smoothed latent state of one training subject on its own nodes.
    fn latent_state<'py>(&self, py: Python<'py>, subject: usize) -> PyResult<Bound<'py, PyDict>> {
        let history = self.subject(subject)?;
        let fit = Arc::clone(&self.fit);
        let cohort = Arc::clone(&self.cohort);
        let path = detach_py_result(py, "event-history latent state", move || {
            latent_state(&fit, &cohort, &history).map_err(|e| e.to_string())
        })?;
        let atoms = self.fit.rank();
        let n = path.times.len();
        let covariances = Array3::from_shape_vec((n, atoms, atoms), path.covariances)
            .map_err(|e| py_value_error(e.to_string()))?;
        let out = PyDict::new(py);
        out.set_item("times", PyArray1::from_owned_array(py, Array1::from(path.times)))?;
        out.set_item("means", PyArray2::from_owned_array(py, path.means))?;
        out.set_item("covariances", PyArray3::from_owned_array(py, covariances))?;
        Ok(out)
    }

    /// The follow-up average of one training subject's latent state as a
    /// posterior Gaussian.
    fn latent_exposure<'py>(&self, py: Python<'py>, subject: usize) -> PyResult<Bound<'py, PyDict>> {
        let history = self.subject(subject)?;
        let fit = Arc::clone(&self.fit);
        let cohort = Arc::clone(&self.cohort);
        let (mean, cov) = detach_py_result(py, "event-history latent exposure", move || {
            latent_exposure(&fit, &cohort, &history).map_err(|e| e.to_string())
        })?;
        let atoms = self.fit.rank();
        let covariance = Array2::from_shape_vec((atoms, atoms), cov)
            .map_err(|e| py_value_error(e.to_string()))?;
        let out = PyDict::new(py);
        out.set_item("mean", PyArray1::from_owned_array(py, Array1::from(mean)))?;
        out.set_item("covariance", PyArray2::from_owned_array(py, covariance))?;
        Ok(out)
    }

    /// Predictive PIT of every observed event of one training subject.
    fn pit(&self, py: Python<'_>, subject: usize) -> PyResult<Vec<f64>> {
        let history = self.subject(subject)?;
        let fit = Arc::clone(&self.fit);
        let cohort = Arc::clone(&self.cohort);
        detach_py_result(py, "event-history pit", move || {
            predictive_pit(&fit, &cohort, &history).map_err(|e| e.to_string())
        })
    }

    /// Kolmogorov–Smirnov distance of the cohort's predictive PITs from uniform.
    fn pit_ks(&self, py: Python<'_>) -> PyResult<f64> {
        let fit = Arc::clone(&self.fit);
        let cohort = Arc::clone(&self.cohort);
        detach_py_result(py, "event-history pit", move || {
            let mut pits = Vec::new();
            for subject in &cohort.subjects {
                pits.extend(predictive_pit(&fit, &cohort, subject).map_err(|e| e.to_string())?);
            }
            Ok(kolmogorov_smirnov_uniform(&pits))
        })
    }
}

fn parse_kind(kind: &str) -> PyResult<MarkKind> {
    match kind {
        "recurrent" => Ok(MarkKind::Recurrent),
        "once" => Ok(MarkKind::Once),
        "terminal" => Ok(MarkKind::Terminal),
        other => Err(py_value_error(format!(
            "mark kind must be \"recurrent\", \"once\" or \"terminal\", got {other:?}"
        ))),
    }
}

/// Fit an event-history model from flat arrays.
#[pyfunction]
#[pyo3(signature = (mark_names, mark_kinds, covariate_names, covariates, subject_ids, entry, exit, event_subject, event_time, event_mark, segment_subject, segment_start, segment_row, formula))]
fn fit_event_history(
    py: Python<'_>,
    mark_names: Vec<String>,
    mark_kinds: Vec<String>,
    covariate_names: Vec<String>,
    covariates: PyReadonlyArray2<'_, f64>,
    subject_ids: Vec<String>,
    entry: Vec<f64>,
    exit: Vec<f64>,
    event_subject: Vec<usize>,
    event_time: Vec<f64>,
    event_mark: Vec<usize>,
    segment_subject: Vec<usize>,
    segment_start: Vec<f64>,
    segment_row: Vec<usize>,
    formula: String,
) -> PyResult<PyEventHistoryModel> {
    let n = subject_ids.len();
    if entry.len() != n || exit.len() != n {
        return Err(py_value_error(format!(
            "subject_ids, entry and exit must have one entry per subject ({n})"
        )));
    }
    if mark_kinds.len() != mark_names.len() {
        return Err(py_value_error(format!(
            "{} mark kinds for {} marks",
            mark_kinds.len(),
            mark_names.len()
        )));
    }
    if event_subject.len() != event_time.len() || event_subject.len() != event_mark.len() {
        return Err(py_value_error(
            "event_subject, event_time and event_mark must have equal length".to_string(),
        ));
    }
    if segment_subject.len() != segment_start.len() || segment_subject.len() != segment_row.len() {
        return Err(py_value_error(
            "segment_subject, segment_start and segment_row must have equal length".to_string(),
        ));
    }
    let mark_kinds = mark_kinds
        .iter()
        .map(|k| parse_kind(k))
        .collect::<PyResult<Vec<MarkKind>>>()?;
    let covariates: Array2<f64> = covariates.as_array().to_owned();
    let mut subjects: Vec<SubjectHistory> = subject_ids
        .iter()
        .zip(entry.iter().zip(exit.iter()))
        .map(|(id, (&entry, &exit))| SubjectHistory {
            id: id.clone(),
            entry,
            exit,
            events: Vec::new(),
            segments: Vec::new(),
        })
        .collect();
    for ((&s, &t), &m) in event_subject.iter().zip(event_time.iter()).zip(event_mark.iter()) {
        let subject = subjects
            .get_mut(s)
            .ok_or_else(|| py_value_error(format!("event subject index {s} is out of range")))?;
        subject.events.push(Event { time: t, mark: m });
    }
    for ((&s, &start), &row) in segment_subject
        .iter()
        .zip(segment_start.iter())
        .zip(segment_row.iter())
    {
        let subject = subjects
            .get_mut(s)
            .ok_or_else(|| py_value_error(format!("segment subject index {s} is out of range")))?;
        subject.segments.push(CovariateSegment { start, row });
    }
    let mut cohort = EventHistoryCohort {
        mark_names,
        mark_kinds,
        covariate_names,
        covariates,
        subjects,
    };
    let fit = detach_py_result(py, "event-history fit", move || {
        let fit = fit_event_history_formula(&mut cohort, &formula, BlockwiseFitOptions::default())
            .map_err(|e| e.to_string())?;
        Ok((fit, cohort))
    })?;
    let (fit, cohort) = fit;
    Ok(PyEventHistoryModel {
        fit: Arc::new(fit),
        cohort: Arc::new(cohort),
    })
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyEventHistoryModel>()?;
    module.add_function(wrap_pyfunction!(fit_event_history, module)?)?;
    Ok(())
}
