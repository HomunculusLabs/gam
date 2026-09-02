//! Python bindings for the event-history family: fit from arrays, then
//! forecast and score subjects from the in-memory fit.

use crate::ffi::ffi_errors::{detach_py_result, py_value_error};
use gam::families::event_history::{
    CovariateSegment, Event, EventHistoryCohort, EventHistoryFit, ForecastRequest,
    SubjectHistory, fit_event_history_formula, forecast, kolmogorov_smirnov_uniform,
    predictive_pit,
};
use gam::families::custom_family::BlockwiseFitOptions;
use ndarray::Array2;
use numpy::{PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};
use std::sync::Arc;

/// A fitted event-history model held in memory.
#[pyclass(name = "_EventHistoryModel", frozen)]
pub(crate) struct PyEventHistoryModel {
    fit: Arc<EventHistoryFit>,
    cohort: Arc<EventHistoryCohort>,
}

#[pymethods]
impl PyEventHistoryModel {
    fn mark_names(&self) -> Vec<String> {
        self.cohort.mark_names.clone()
    }

    fn covariate_names(&self) -> Vec<String> {
        self.cohort.covariate_names.clone()
    }

    fn subject_ids(&self) -> Vec<String> {
        self.cohort.subjects.iter().map(|s| s.id.clone()).collect()
    }

    fn atoms(&self) -> usize {
        self.fit.atoms()
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

    fn quadrature<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let out = PyDict::new(py);
        out.set_item("order", self.fit.quadrature.order)?;
        out.set_item("checked_order", self.fit.quadrature.checked_order)?;
        out.set_item("log_likelihood", self.fit.quadrature.log_likelihood)?;
        out.set_item(
            "log_likelihood_at_checked_order",
            self.fit.quadrature.log_likelihood_at_checked_order,
        )?;
        Ok(out)
    }

    /// Forecast one training subject beyond its exit.
    #[pyo3(signature = (subject, horizons, absorbing, future_row=None))]
    fn forecast<'py>(
        &self,
        py: Python<'py>,
        subject: usize,
        horizons: Vec<f64>,
        absorbing: Vec<bool>,
        future_row: Option<usize>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let history = self
            .cohort
            .subjects
            .get(subject)
            .ok_or_else(|| py_value_error(format!("subject index {subject} is out of range")))?;
        let row = future_row.unwrap_or_else(|| {
            history
                .segments
                .last()
                .map(|s| s.row)
                .unwrap_or(0)
        });
        let fit = Arc::clone(&self.fit);
        let cohort = Arc::clone(&self.cohort);
        let history = history.clone();
        let result = detach_py_result(py, "event-history forecast", move || {
            forecast(
                &fit,
                &cohort,
                &ForecastRequest {
                    history: &history,
                    horizons: &horizons,
                    absorbing: &absorbing,
                    future_row: row,
                },
            )
            .map_err(|e| e.to_string())
        })?;
        let out = PyDict::new(py);
        out.set_item("horizons", result.horizons)?;
        out.set_item("survival", result.survival)?;
        out.set_item(
            "expected_counts",
            PyArray2::from_owned_array(py, result.expected_counts),
        )?;
        Ok(out)
    }

    /// Predictive PIT of every event of one training subject.
    fn pit(&self, py: Python<'_>, subject: usize) -> PyResult<Vec<f64>> {
        let history = self
            .cohort
            .subjects
            .get(subject)
            .ok_or_else(|| py_value_error(format!("subject index {subject} is out of range")))?
            .clone();
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

/// Fit an event-history model from flat arrays.
#[pyfunction]
#[pyo3(signature = (mark_names, covariate_names, covariates, subject_ids, entry, exit, event_subject, event_time, event_mark, segment_subject, segment_start, segment_row, formula, atoms))]
fn fit_event_history(
    py: Python<'_>,
    mark_names: Vec<String>,
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
    atoms: usize,
) -> PyResult<PyEventHistoryModel> {
    let n = subject_ids.len();
    if entry.len() != n || exit.len() != n {
        return Err(py_value_error(format!(
            "subject_ids, entry and exit must have one entry per subject ({n})"
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
        covariate_names,
        covariates,
        subjects,
    };
    let fit = detach_py_result(py, "event-history fit", move || {
        let fit = fit_event_history_formula(&mut cohort, &formula, atoms, BlockwiseFitOptions::default())
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
