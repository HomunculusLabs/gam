//! Python bindings for the joint event model: fit, save, load, condition and
//! forecast through the one Rust model path the CLI also calls. This layer only
//! moves arrays.

use crate::ffi::ffi_errors::{detach_py_result, py_value_error};
use gam::event_history::joint::{self, JointEventModel};
use gam::event_history::{Event, MarkKind, SubjectHistory, mark_index_of, resolve_mark_vocabulary};
use numpy::PyArray2;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};
use std::path::Path;
use std::sync::Arc;

/// A fitted or loaded joint event model.
#[pyclass(name = "_JointEventModel", frozen)]
pub(crate) struct PyJointEventModel {
    model: Arc<JointEventModel>,
}

#[pymethods]
impl PyJointEventModel {
    fn mark_names(&self) -> Vec<String> {
        self.model.mark_names().to_vec()
    }

    fn mark_kinds(&self) -> Vec<String> {
        self.model
            .mark_kinds()
            .iter()
            .map(|k| k.name().to_string())
            .collect()
    }

    /// Save the model to `path`: the mark vocabulary and the posterior, never
    /// the training records.
    fn save(&self, path: &str) -> PyResult<()> {
        self.model
            .save(Path::new(path))
            .map_err(|e| py_value_error(e.to_string()))
    }

    /// Condition on one history (entry, exit, and its events as parallel time
    /// and mark-label vectors) and forecast `horizons` after its exit.
    #[pyo3(signature = (entry, exit, event_time, event_marks, horizons))]
    fn forecast<'py>(
        &self,
        py: Python<'py>,
        entry: f64,
        exit: f64,
        event_time: Vec<f64>,
        event_marks: Vec<String>,
        horizons: Vec<f64>,
    ) -> PyResult<Bound<'py, PyDict>> {
        if event_time.len() != event_marks.len() {
            return Err(py_value_error(
                "event_time and event_marks must have equal length".to_string(),
            ));
        }
        let events = event_time
            .iter()
            .zip(&event_marks)
            .map(|(&time, label)| {
                mark_index_of(self.model.mark_names(), label)
                    .map(|mark| Event { time, mark })
                    .map_err(|e| py_value_error(e.to_string()))
            })
            .collect::<PyResult<Vec<Event>>>()?;
        let history = SubjectHistory {
            id: "history".to_string(),
            entry,
            exit,
            events,
            segments: Vec::new(),
        };
        let model = Arc::clone(&self.model);
        let result = detach_py_result(py, "joint event forecast", move || {
            model
                .condition(&history)
                .and_then(|conditioned| conditioned.forecast(&horizons))
                .map_err(|e| e.to_string())
        })?;
        let out = PyDict::new(py);
        out.set_item("horizons", result.horizons)?;
        out.set_item("survival", result.survival)?;
        out.set_item("incidence", PyArray2::from_owned_array(py, result.incidence))?;
        out.set_item(
            "incidence_error",
            PyArray2::from_owned_array(py, result.incidence_error),
        )?;
        Ok(out)
    }
}

/// Fit the joint event model from flat arrays.
#[pyfunction]
#[pyo3(signature = (declared_marks, subject_ids, entry, exit, event_subject, event_time, event_marks))]
fn fit_joint_event_model(
    py: Python<'_>,
    declared_marks: Option<Vec<(String, String)>>,
    subject_ids: Vec<String>,
    entry: Vec<f64>,
    exit: Vec<f64>,
    event_subject: Vec<usize>,
    event_time: Vec<f64>,
    event_marks: Vec<String>,
) -> PyResult<PyJointEventModel> {
    let n = subject_ids.len();
    if entry.len() != n || exit.len() != n {
        return Err(py_value_error(format!(
            "subject_ids, entry and exit must have one entry per subject ({n})"
        )));
    }
    if event_subject.len() != event_time.len() || event_subject.len() != event_marks.len() {
        return Err(py_value_error(
            "event_subject, event_time and event_marks must have equal length".to_string(),
        ));
    }
    let declared = declared_marks
        .map(|pairs| {
            pairs
                .into_iter()
                .map(|(name, kind)| {
                    MarkKind::parse(&kind)
                        .map(|kind| (name, kind))
                        .map_err(|e| py_value_error(e.to_string()))
                })
                .collect::<PyResult<Vec<(String, MarkKind)>>>()
        })
        .transpose()?;
    let labels: Vec<&str> = event_marks.iter().map(String::as_str).collect();
    let (mark_names, mark_kinds, event_mark) = resolve_mark_vocabulary(declared, &labels)
        .map_err(|e| py_value_error(e.to_string()))?;
    let mut subjects: Vec<SubjectHistory> = subject_ids
        .into_iter()
        .zip(entry.into_iter().zip(exit))
        .map(|(id, (entry, exit))| SubjectHistory {
            id,
            entry,
            exit,
            events: Vec::new(),
            segments: Vec::new(),
        })
        .collect();
    for ((&s, &time), &mark) in event_subject.iter().zip(&event_time).zip(&event_mark) {
        subjects
            .get_mut(s)
            .ok_or_else(|| py_value_error(format!("event subject index {s} is out of range")))?
            .events
            .push(Event { time, mark });
    }
    let model = detach_py_result(py, "joint event fit", move || {
        joint::fit_joint_event_model(mark_names, mark_kinds, &subjects).map_err(|e| e.to_string())
    })?;
    Ok(PyJointEventModel {
        model: Arc::new(model),
    })
}

/// Load a saved joint event model.
#[pyfunction]
fn load_joint_event_model(path: &str) -> PyResult<PyJointEventModel> {
    let model =
        JointEventModel::load(Path::new(path)).map_err(|e| py_value_error(e.to_string()))?;
    Ok(PyJointEventModel {
        model: Arc::new(model),
    })
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyJointEventModel>()?;
    module.add_function(wrap_pyfunction!(fit_joint_event_model, module)?)?;
    module.add_function(wrap_pyfunction!(load_joint_event_model, module)?)?;
    Ok(())
}
