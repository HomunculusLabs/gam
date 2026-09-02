//! Python bindings for the event-history family: fit from arrays, then
//! forecast and score subjects from the in-memory fit. Every rule about the
//! data (mark kinds, categorical levels, follow-up) lives in the Rust cohort
//! validation; this layer only moves arrays.

use crate::ffi::ffi_errors::{detach_py_result, py_value_error};
use gam::families::custom_family::BlockwiseFitOptions;
use gam::families::event_history::{
    CovariateSegment, Event, EventHistoryCohort, EventHistoryFit, ForecastRequest,
    FutureSegment, MarkKind, PopulationForecastRequest, SubjectHistory,
    fit_event_history_formula, forecast, kolmogorov_smirnov_uniform, population_forecast,
    predictive_pit,
};
use ndarray::Array2;
use numpy::{PyArray1, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyModule};
use std::sync::Arc;

/// A fitted event-history model held in memory.
#[pyclass(name = "_EventHistoryModel", frozen)]
pub(crate) struct PyEventHistoryModel {
    fit: Arc<EventHistoryFit>,
    cohort: Arc<EventHistoryCohort>,
}

fn future_segments(future: Vec<(f64, Vec<f64>)>) -> Vec<FutureSegment> {
    future
        .into_iter()
        .map(|(start, covariates)| FutureSegment { start, covariates })
        .collect()
}

fn forecast_dict<'py>(
    py: Python<'py>,
    result: gam::families::event_history::Forecast,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("horizons", result.horizons)?;
    out.set_item("survival", result.survival)?;
    out.set_item(
        "expected_counts",
        PyArray2::from_owned_array(py, result.expected_counts),
    )?;
    Ok(out)
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
            .map(|k| k.name().to_string())
            .collect()
    }

    fn covariate_names(&self) -> Vec<String> {
        self.cohort.covariate_names.clone()
    }

    fn covariate_levels(&self) -> Vec<Vec<String>> {
        self.cohort.covariate_levels.clone()
    }

    fn subject_ids(&self) -> Vec<String> {
        self.cohort.subjects.iter().map(|s| s.id.clone()).collect()
    }

    fn subject_exits(&self) -> Vec<f64> {
        self.cohort.subjects.iter().map(|s| s.exit).collect()
    }

    fn rank(&self) -> usize {
        self.fit.rank()
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
            item.set_item("converged", step.converged)?;
            item.set_item("evidence_gain", step.evidence_gain)?;
            item.set_item("accepted", step.accepted)?;
            out.append(item)?;
        }
        Ok(out)
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
        let q = &self.fit.quadrature;
        let out = PyDict::new(py);
        out.set_item("gauss_hermite_order", q.gauss_hermite_order)?;
        out.set_item("mesh_refinement", q.mesh_refinement)?;
        out.set_item("log_likelihood", q.log_likelihood)?;
        let gh = PyDict::new(py);
        gh.set_item("order", q.gauss_hermite.candidate)?;
        gh.set_item("coefficient_shift", q.gauss_hermite.coefficient_shift)?;
        gh.set_item("log_likelihood", q.gauss_hermite.log_likelihood)?;
        out.set_item("gauss_hermite_check", gh)?;
        let mesh = PyDict::new(py);
        mesh.set_item("refinement", q.mesh.candidate)?;
        mesh.set_item("coefficient_shift", q.mesh.coefficient_shift)?;
        mesh.set_item("log_likelihood", q.mesh.log_likelihood)?;
        out.set_item("mesh_check", mesh)?;
        Ok(out)
    }

    /// Forecast one training subject beyond its exit over a covariate path.
    #[pyo3(signature = (subject, horizons, future))]
    fn forecast<'py>(
        &self,
        py: Python<'py>,
        subject: usize,
        horizons: Vec<f64>,
        future: Vec<(f64, Vec<f64>)>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let history = self
            .cohort
            .subjects
            .get(subject)
            .ok_or_else(|| py_value_error(format!("subject index {subject} is out of range")))?
            .clone();
        let fit = Arc::clone(&self.fit);
        let cohort = Arc::clone(&self.cohort);
        let future = future_segments(future);
        let result = detach_py_result(py, "event-history forecast", move || {
            forecast(
                &fit,
                &cohort,
                &ForecastRequest {
                    history: &history,
                    horizons: &horizons,
                    future: &future,
                },
            )
            .map_err(|e| e.to_string())
        })?;
        forecast_dict(py, result)
    }

    /// Forecast a subject with no history from a covariate path alone.
    #[pyo3(signature = (start, horizons, future))]
    fn population_forecast<'py>(
        &self,
        py: Python<'py>,
        start: f64,
        horizons: Vec<f64>,
        future: Vec<(f64, Vec<f64>)>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let fit = Arc::clone(&self.fit);
        let cohort = Arc::clone(&self.cohort);
        let future = future_segments(future);
        let result = detach_py_result(py, "event-history population forecast", move || {
            population_forecast(
                &fit,
                &cohort,
                &PopulationForecastRequest {
                    start,
                    horizons: &horizons,
                    future: &future,
                },
            )
            .map_err(|e| e.to_string())
        })?;
        forecast_dict(py, result)
    }

    /// Predictive PIT of every event of one training subject: times, marks,
    /// PIT values and the predictive mark probabilities at each event.
    fn pit<'py>(&self, py: Python<'py>, subject: usize) -> PyResult<Bound<'py, PyDict>> {
        let history = self
            .cohort
            .subjects
            .get(subject)
            .ok_or_else(|| py_value_error(format!("subject index {subject} is out of range")))?
            .clone();
        let fit = Arc::clone(&self.fit);
        let cohort = Arc::clone(&self.cohort);
        let pits = detach_py_result(py, "event-history pit", move || {
            predictive_pit(&fit, &cohort, &history).map_err(|e| e.to_string())
        })?;
        let marks = self.fit.marks();
        let mut probabilities = Array2::<f64>::zeros((pits.len(), marks));
        for (i, pit) in pits.iter().enumerate() {
            for d in 0..marks {
                probabilities[[i, d]] = pit.mark_probabilities[d];
            }
        }
        let out = PyDict::new(py);
        out.set_item("time", pits.iter().map(|p| p.time).collect::<Vec<_>>())?;
        out.set_item("mark", pits.iter().map(|p| p.mark).collect::<Vec<_>>())?;
        out.set_item("pit", pits.iter().map(|p| p.pit).collect::<Vec<_>>())?;
        out.set_item(
            "mark_probabilities",
            PyArray2::from_owned_array(py, probabilities),
        )?;
        Ok(out)
    }

    /// Kolmogorov–Smirnov distance of the cohort's predictive PITs from
    /// uniform, or `None` when the cohort has no events.
    fn pit_ks(&self, py: Python<'_>) -> PyResult<Option<f64>> {
        let fit = Arc::clone(&self.fit);
        let cohort = Arc::clone(&self.cohort);
        detach_py_result(py, "event-history pit", move || {
            let mut pits = Vec::new();
            for subject in &cohort.subjects {
                pits.extend(
                    predictive_pit(&fit, &cohort, subject)
                        .map_err(|e| e.to_string())?
                        .into_iter()
                        .map(|p| p.pit),
                );
            }
            Ok(kolmogorov_smirnov_uniform(&pits))
        })
    }
}

/// Fit an event-history model from flat arrays.
#[pyfunction]
#[pyo3(signature = (mark_names, mark_kinds, covariate_names, covariate_levels, covariates, subject_ids, entry, exit, event_subject, event_time, event_mark, segment_subject, segment_start, segment_row, formula))]
fn fit_event_history(
    py: Python<'_>,
    mark_names: Vec<String>,
    mark_kinds: Vec<String>,
    covariate_names: Vec<String>,
    covariate_levels: Vec<Vec<String>>,
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
        .map(|k| MarkKind::parse(k).map_err(|e| py_value_error(e.to_string())))
        .collect::<PyResult<Vec<_>>>()?;
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
        covariate_levels,
        covariates,
        subjects,
    };
    let (fit, cohort) = detach_py_result(py, "event-history fit", move || {
        let fit = fit_event_history_formula(&mut cohort, &formula, BlockwiseFitOptions::default())
            .map_err(|e| e.to_string())?;
        Ok((fit, cohort))
    })?;
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
