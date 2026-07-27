//! Public fitted-model surface for overcomplete hard-TopK manifold SAEs.
//!
//! This is a distinct representation of the same public model family: routing
//! is canonical support state (`N×s` indices plus heterogeneous coordinates),
//! never dense logits or gates. The dense `K<=P` artifact remains its
//! full-support specialization; crossing `K>P` changes representation before
//! any seed allocation.

use gam::terms::sae::front_door::{SaeFitLane, admit_topk_manifold};
use gam::terms::sae::manifold::{
    SAE_SUPPORT_INNER_FIXED_POINT_MAX_ITER, SaeSupportFixedPointReport, SaeSupportOuterRequest,
    SaeSupportRehydrateRequest, SaeSupportSparseTerm, SaeSupportTermSeedRequest,
    build_sae_support_seed, build_sae_support_term_seed, rehydrate_sae_support_term,
    run_sae_support_outer, sae_support_effective_atom_dims,
};
use gam::terms::sae::manifold::{SaeSupportSeedRequest, SaeSupportStationarity};
use ndarray::{Array1, Array2, ArrayView2, Axis};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::{PyObject, json_value_to_py, py_value_error};

pub(crate) struct SupportSparseFitRequest<'a> {
    pub target: ArrayView2<'a, f64>,
    pub atom_basis: Vec<String>,
    pub atom_dim: Vec<usize>,
    pub support_k: usize,
    pub initial_smoothness: f64,
    pub max_iter: usize,
    pub trust_radius: f64,
    pub tolerance: f64,
    pub random_state: u64,
}

/// The on-disk tag for this representation. Checked exactly, never sniffed:
/// the dense artifact's tag also begins `gamfit.ManifoldSAE`, so a substring
/// test routes dense payloads here and support payloads there (#2567).
pub(crate) const SUPPORT_SCHEMA_TAG: &str = "gamfit.ManifoldSAE/support-v2";

fn required_field<'py>(
    payload: &Bound<'py, PyDict>,
    key: &str,
) -> PyResult<Bound<'py, PyAny>> {
    payload.get_item(key)?.ok_or_else(|| {
        py_value_error(format!(
            "ManifoldSAESupport.from_dict: payload is missing required field {key:?}"
        ))
    })
}

#[pyclass(module = "gamfit._rust", name = "ManifoldSAESupport", frozen)]
pub(crate) struct SupportSparseManifoldSaeCore {
    term: SaeSupportSparseTerm,
    requested_k: usize,
    retained_atom_indices: Vec<usize>,
    atom_basis: Vec<String>,
    atom_dim: Vec<usize>,
    atom_topologies: Vec<String>,
    support_k: usize,
    training_mean: Vec<f64>,
    fitted: Array2<f64>,
    reconstruction_r2: f64,
    log_lambda_smooth: Vec<f64>,
    ard_precisions: Vec<Vec<f64>>,
    criterion: f64,
    certificates: serde_json::Value,
    termination: serde_json::Value,
    max_iter: usize,
    trust_radius: f64,
    tolerance: f64,
    random_state: u64,
}

fn add_mean(mut fitted: Array2<f64>, mean: &[f64]) -> Array2<f64> {
    for mut row in fitted.rows_mut() {
        for (value, location) in row.iter_mut().zip(mean) {
            *value += location;
        }
    }
    fitted
}

fn centered(target: ArrayView2<'_, f64>, mean: &[f64]) -> Array2<f64> {
    Array2::from_shape_fn(target.dim(), |(row, column)| {
        target[[row, column]] - mean[column]
    })
}

fn support_indices(term: &SaeSupportSparseTerm) -> Result<Array2<u32>, String> {
    let rows = term.n_obs();
    let support = term.assignment.support_indices(0).len();
    let values = (0..rows)
        .flat_map(|row| term.assignment.support_indices(row).iter().copied())
        .collect::<Vec<_>>();
    Array2::from_shape_vec((rows, support), values)
        .map_err(|error| format!("support index shape: {error}"))
}

fn support_values(term: &SaeSupportSparseTerm) -> Array2<f64> {
    Array2::ones((term.n_obs(), term.assignment.support_indices(0).len()))
}

fn coords_rows<'py>(py: Python<'py>, term: &SaeSupportSparseTerm) -> PyResult<Bound<'py, PyList>> {
    let rows = PyList::empty(py);
    for row in 0..term.n_obs() {
        rows.append(Array1::from(term.assignment.coords_row(row).to_vec()).into_pyarray(py))?;
    }
    Ok(rows)
}

fn fixed_point_json(report: &SaeSupportFixedPointReport) -> serde_json::Value {
    let SaeSupportStationarity {
        decoder_l2,
        decoder_max_abs,
        coordinate_l2,
        coordinate_max_abs,
    } = report.stationarity;
    serde_json::json!({
        "iterations": report.iterations,
        "objective": report.objective,
        "decoder_l2": decoder_l2,
        "decoder_max_abs": decoder_max_abs,
        "coordinate_l2": coordinate_l2,
        "coordinate_max_abs": coordinate_max_abs,
        "max_recurrence_change": report.max_recurrence_change,
        "recurred": report.recurred,
    })
}

impl SupportSparseManifoldSaeCore {
    fn infer(
        &self,
        target: ArrayView2<'_, f64>,
    ) -> Result<(SaeSupportSparseTerm, Array2<f64>, serde_json::Value), String> {
        if target.ncols() != self.training_mean.len() || target.nrows() == 0 {
            return Err(format!(
                "ManifoldSAE.converged_latents requires positive rows and P={}; got {:?}",
                self.training_mean.len(),
                target.dim()
            ));
        }
        if target.iter().any(|value| !value.is_finite()) {
            return Err("ManifoldSAE.converged_latents target contains a non-finite value".into());
        }
        let centered_target = centered(target, &self.training_mean);
        let mut term = self.term.reroute_fixed_decoder(
            centered_target.view(),
            self.support_k,
            self.random_state,
        )?;
        let report = term.solve_coordinates_fixed_decoder(
            centered_target.view(),
            &self.ard_precisions,
            self.max_iter,
            self.tolerance,
            self.trust_radius,
        )?;
        let fitted = add_mean(term.reconstruct()?, &self.training_mean);
        let certificate = serde_json::json!({
            "iterations": report.iterations,
            "objective": report.objective,
            "coordinate_l2": report.coordinate_l2,
            "coordinate_max_abs": report.coordinate_max_abs,
            "max_recurrence_change": report.max_recurrence_change,
            "recurred": report.recurred,
        });
        Ok((term, fitted, certificate))
    }

    fn latents_dict<'py>(
        &self,
        py: Python<'py>,
        term: &SaeSupportSparseTerm,
        fitted: Array2<f64>,
        certificate: serde_json::Value,
    ) -> PyResult<Py<PyDict>> {
        let out = PyDict::new(py);
        out.set_item("fitted", fitted.into_pyarray(py))?;
        out.set_item(
            "support_indices",
            support_indices(term)
                .map_err(py_value_error)?
                .into_pyarray(py),
        )?;
        out.set_item("support_values", support_values(term).into_pyarray(py))?;
        out.set_item("coords", coords_rows(py, term)?)?;
        out.set_item("certificate", json_value_to_py(py, certificate)?)?;
        Ok(out.unbind())
    }
}

#[pymethods]
impl SupportSparseManifoldSaeCore {
    #[pyo3(signature = (x_new=None))]
    fn converged_latents<'py>(
        &self,
        py: Python<'py>,
        x_new: Option<PyReadonlyArray2<'py, f64>>,
    ) -> PyResult<Py<PyDict>> {
        match x_new {
            Some(values) => {
                let (term, fitted, certificate) =
                    self.infer(values.as_array()).map_err(py_value_error)?;
                self.latents_dict(py, &term, fitted, certificate)
            }
            None => self.latents_dict(
                py,
                &self.term,
                self.fitted.clone(),
                self.certificates["inner_fixed_point"].clone(),
            ),
        }
    }

    fn reconstruct<'py>(
        &self,
        py: Python<'py>,
        x_new: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let (_, fitted, _) = self.infer(x_new.as_array()).map_err(py_value_error)?;
        Ok(fitted.into_pyarray(py))
    }

    fn predict<'py>(
        &self,
        py: Python<'py>,
        x_new: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.reconstruct(py, x_new)
    }

    fn encode<'py>(
        &self,
        py: Python<'py>,
        x_new: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Py<PyDict>> {
        let (term, _, certificate) = self.infer(x_new.as_array()).map_err(py_value_error)?;
        let out = PyDict::new(py);
        out.set_item(
            "indices",
            support_indices(&term)
                .map_err(py_value_error)?
                .into_pyarray(py),
        )?;
        out.set_item("values", support_values(&term).into_pyarray(py))?;
        out.set_item("coords", coords_rows(py, &term)?)?;
        out.set_item("certificate", json_value_to_py(py, certificate)?)?;
        Ok(out.unbind())
    }

    /// Sample one atom's decoded curve at coordinates `(n, d)` → `(n, P)`.
    fn atom_curve<'py>(
        &self,
        py: Python<'py>,
        atom_k: usize,
        coords: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let curve = self
            .term
            .decode_atom_at(atom_k, coords.as_array())
            .map_err(py_value_error)?;
        Ok(curve.into_pyarray(py))
    }

    /// On-manifold steering delta along one atom:
    /// `amplitude · (γ_k(t_to) − γ_k(t_from))`, decoded by the atom's own
    /// evaluator. Returns the ambient-chart delta plus both decoded images.
    fn steer<'py>(
        &self,
        py: Python<'py>,
        atom_k: usize,
        amplitude: f64,
        t_from: PyReadonlyArray1<'py, f64>,
        t_to: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Py<PyDict>> {
        if !amplitude.is_finite() {
            return Err(py_value_error(format!(
                "ManifoldSAE.steer: amplitude must be finite; got {amplitude}"
            )));
        }
        let from = t_from.as_array();
        let to = t_to.as_array();
        let width = from.len();
        if to.len() != width {
            return Err(py_value_error(format!(
                "ManifoldSAE.steer: t_from width {} != t_to width {}",
                width,
                to.len()
            )));
        }
        let stack = |values: ndarray::ArrayView1<'_, f64>| {
            ndarray::Array2::from_shape_vec((1, width), values.to_vec())
                .map_err(|error| py_value_error(format!("ManifoldSAE.steer: {error}")))
        };
        let g_from = self
            .term
            .decode_atom_at(atom_k, stack(from)?.view())
            .map_err(py_value_error)?;
        let g_to = self
            .term
            .decode_atom_at(atom_k, stack(to)?.view())
            .map_err(py_value_error)?;
        let delta = (&g_to - &g_from).row(0).mapv(|value| amplitude * value);
        let out = PyDict::new(py);
        out.set_item("delta", delta.into_pyarray(py))?;
        out.set_item("decoded_from", g_from.row(0).to_owned().into_pyarray(py))?;
        out.set_item("decoded_to", g_to.row(0).to_owned().into_pyarray(py))?;
        Ok(out.unbind())
    }

    fn to_dict(&self, py: Python<'_>) -> PyResult<PyObject> {
        let out = PyDict::new(py);
        out.set_item("schema", SUPPORT_SCHEMA_TAG)?;
        out.set_item("requested_k", self.requested_k)?;
        out.set_item("retained_atom_indices", self.retained_atom_indices.clone())?;
        out.set_item("atom_basis", self.atom_basis.clone())?;
        out.set_item("atom_dim", self.atom_dim.clone())?;
        out.set_item("top_k", self.support_k)?;
        out.set_item("training_mean", self.training_mean.clone())?;
        out.set_item("fitted", self.fitted.clone().into_pyarray(py))?;
        out.set_item(
            "support_indices",
            support_indices(&self.term)
                .map_err(py_value_error)?
                .into_pyarray(py),
        )?;
        out.set_item(
            "support_values",
            support_values(&self.term).into_pyarray(py),
        )?;
        out.set_item("coords", coords_rows(py, &self.term)?)?;
        let decoders = PyList::empty(py);
        for atom in &self.term.atoms {
            decoders.append(atom.decoder_coefficients().clone().into_pyarray(py))?;
        }
        out.set_item("decoder_blocks", decoders)?;
        out.set_item("log_lambda_smooth", self.log_lambda_smooth.clone())?;
        out.set_item("ard_precisions", self.ard_precisions.clone())?;
        out.set_item("criterion", self.criterion)?;
        out.set_item(
            "certificates",
            json_value_to_py(py, self.certificates.clone())?,
        )?;
        out.set_item(
            "termination",
            json_value_to_py(py, self.termination.clone())?,
        )?;
        // Without these the payload cannot rebuild the model it came from:
        // `reconstruction_r2` is not recoverable from `fitted` alone (the
        // training target is not stored), and the four fit knobs are model
        // state that `from_dict` must restore rather than invent.
        out.set_item("reconstruction_r2", self.reconstruction_r2)?;
        out.set_item("max_iter", self.max_iter)?;
        out.set_item("trust_radius", self.trust_radius)?;
        out.set_item("tolerance", self.tolerance)?;
        out.set_item("random_state", self.random_state)?;
        Ok(out.unbind().into_any())
    }

    /// Rebuild a fitted overcomplete model from [`Self::to_dict`] (#2567).
    ///
    /// `atom_topologies` is re-derived from `atom_basis` rather than stored,
    /// because it is a function of the bases and a stored copy could disagree
    /// with them.
    #[staticmethod]
    fn from_dict(payload: &Bound<'_, PyDict>) -> PyResult<Self> {
        let schema: String = required_field(payload, "schema")?.extract()?;
        if schema != SUPPORT_SCHEMA_TAG {
            return Err(py_value_error(format!(
                "ManifoldSAESupport.from_dict: schema {schema:?} is not {SUPPORT_SCHEMA_TAG:?}; \
                 dense payloads tagged gamfit.ManifoldSAE/v6 load with ManifoldSAE.from_dict"
            )));
        }
        let requested_k: usize = required_field(payload, "requested_k")?.extract()?;
        let retained_atom_indices: Vec<usize> =
            required_field(payload, "retained_atom_indices")?.extract()?;
        let atom_basis: Vec<String> = required_field(payload, "atom_basis")?.extract()?;
        let atom_dim: Vec<usize> = required_field(payload, "atom_dim")?.extract()?;
        let support_k: usize = required_field(payload, "top_k")?.extract()?;
        let training_mean: Vec<f64> = required_field(payload, "training_mean")?.extract()?;
        let fitted = required_field(payload, "fitted")?
            .extract::<PyReadonlyArray2<'_, f64>>()?
            .as_array()
            .to_owned();
        let support_indices = required_field(payload, "support_indices")?
            .extract::<PyReadonlyArray2<'_, u32>>()?
            .as_array()
            .rows()
            .into_iter()
            .map(|row| row.to_vec())
            .collect::<Vec<_>>();
        let support_values = required_field(payload, "support_values")?
            .extract::<PyReadonlyArray2<'_, f64>>()?
            .as_array()
            .rows()
            .into_iter()
            .map(|row| row.to_vec())
            .collect::<Vec<_>>();
        let coord_items: Vec<Bound<'_, PyAny>> =
            required_field(payload, "coords")?.extract()?;
        let mut coords = Vec::with_capacity(coord_items.len());
        for item in &coord_items {
            coords.push(
                item.extract::<PyReadonlyArray1<'_, f64>>()?
                    .as_array()
                    .to_vec(),
            );
        }
        let decoder_items: Vec<Bound<'_, PyAny>> =
            required_field(payload, "decoder_blocks")?.extract()?;
        let mut decoder_blocks = Vec::with_capacity(decoder_items.len());
        for item in &decoder_items {
            decoder_blocks.push(
                item.extract::<PyReadonlyArray2<'_, f64>>()?
                    .as_array()
                    .to_owned(),
            );
        }
        let log_lambda_smooth: Vec<f64> =
            required_field(payload, "log_lambda_smooth")?.extract()?;
        let ard_precisions: Vec<Vec<f64>> =
            required_field(payload, "ard_precisions")?.extract()?;
        let criterion: f64 = required_field(payload, "criterion")?.extract()?;
        let reconstruction_r2: f64 = required_field(payload, "reconstruction_r2")?.extract()?;
        let max_iter: usize = required_field(payload, "max_iter")?.extract()?;
        let trust_radius: f64 = required_field(payload, "trust_radius")?.extract()?;
        let tolerance: f64 = required_field(payload, "tolerance")?.extract()?;
        let random_state: u64 = required_field(payload, "random_state")?.extract()?;
        let certificates = crate::manifold::manifold_sae_coercion::py_any_to_json_value(
            &required_field(payload, "certificates")?,
        )?;
        let termination = crate::manifold::manifold_sae_coercion::py_any_to_json_value(
            &required_field(payload, "termination")?,
        )?;

        let output_dim = training_mean.len();
        let atom_topologies = gam::terms::sae::atom_schema::topologies_for_bases(&atom_basis)
            .map_err(py_value_error)?;
        let term = rehydrate_sae_support_term(SaeSupportRehydrateRequest {
            atom_basis: atom_basis.clone(),
            atom_dim: atom_dim.clone(),
            output_dim,
            support_k,
            random_state,
            support_indices,
            support_values,
            coords,
            decoder_blocks,
        })
        .map_err(py_value_error)?;
        Ok(Self {
            term,
            requested_k,
            retained_atom_indices,
            atom_basis,
            atom_dim,
            atom_topologies,
            support_k,
            training_mean,
            fitted,
            reconstruction_r2,
            log_lambda_smooth,
            ard_precisions,
            criterion,
            certificates,
            termination,
            max_iter,
            trust_radius,
            tolerance,
            random_state,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "ManifoldSAESupport(K_requested={}, K_retained={}, n={}, p={}, assignment=\"topk\", support={})",
            self.requested_k,
            self.term.k_atoms(),
            self.term.n_obs(),
            self.term.output_dim(),
            self.support_k,
        )
    }

    #[getter]
    fn chosen_k(&self) -> usize {
        self.term.k_atoms()
    }
    #[getter]
    fn requested_k(&self) -> usize {
        self.requested_k
    }
    #[getter]
    fn assignment(&self) -> &'static str {
        "topk"
    }
    #[getter]
    fn top_k(&self) -> usize {
        self.support_k
    }
    #[getter]
    fn atom_topologies(&self) -> Vec<String> {
        self.atom_topologies.clone()
    }
    #[getter]
    fn hybrid_split(&self, py: Python<'_>) -> PyObject {
        py.None()
    }
    #[getter]
    fn certificates(&self, py: Python<'_>) -> PyResult<PyObject> {
        json_value_to_py(py, self.certificates.clone())
    }
    #[getter]
    fn termination(&self, py: Python<'_>) -> PyResult<PyObject> {
        json_value_to_py(py, self.termination.clone())
    }
    #[getter]
    fn fitted<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.fitted.clone().into_pyarray(py)
    }
    #[getter]
    fn training_mean<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        Array1::from(self.training_mean.clone()).into_pyarray(py)
    }
    #[getter]
    fn reconstruction_r2(&self) -> f64 {
        self.reconstruction_r2
    }
    #[getter]
    fn penalized_quasi_laplace_criterion(&self) -> f64 {
        self.criterion
    }
    #[getter]
    fn support_indices<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<u32>>> {
        Ok(support_indices(&self.term)
            .map_err(py_value_error)?
            .into_pyarray(py))
    }
    #[getter]
    fn support_values<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        support_values(&self.term).into_pyarray(py)
    }
    #[getter]
    fn decoder_blocks<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let blocks = PyList::empty(py);
        for atom in &self.term.atoms {
            blocks.append(atom.decoder_coefficients().clone().into_pyarray(py))?;
        }
        Ok(blocks)
    }
}

pub(crate) fn fit_support_sparse_manifold_sae(
    py: Python<'_>,
    request: SupportSparseFitRequest<'_>,
) -> PyResult<PyObject> {
    let (n_obs, output_dim) = request.target.dim();
    let requested_k = request.atom_basis.len();
    let mut resolved_basis = request.atom_basis.clone();
    gam::terms::sae::manifold::resolve_support_auto_atoms(&mut resolved_basis);
    let effective_dims = sae_support_effective_atom_dims(&resolved_basis, &request.atom_dim)
        .map_err(py_value_error)?;
    let d_max = effective_dims.iter().copied().max().unwrap_or(1);
    let admission = admit_topk_manifold(n_obs, output_dim, requested_k, d_max, request.support_k)
        .map_err(py_value_error)?;
    if admission.lane != SaeFitLane::CurvedStreaming {
        return Err(py_value_error(format!(
            "support-sparse fit requires CurvedStreaming admission; got {:?}",
            admission.lane
        )));
    }
    let training_mean = request
        .target
        .mean_axis(Axis(0))
        .ok_or_else(|| py_value_error("support-sparse fit requires positive rows".to_string()))?
        .to_vec();
    let centered_target = centered(request.target, &training_mean);
    let seed = build_sae_support_seed(SaeSupportSeedRequest {
        target: centered_target.view(),
        atom_basis: &resolved_basis,
        atom_dim: &request.atom_dim,
        support_k: request.support_k,
        random_state: request.random_state,
        admission,
    })
    .map_err(py_value_error)?;
    let retained_atom_indices = seed.retained_atom_indices;
    let atom_basis = retained_atom_indices
        .iter()
        .map(|&atom| resolved_basis[atom].clone())
        .collect::<Vec<_>>();
    let atom_dim = retained_atom_indices
        .iter()
        .map(|&atom| request.atom_dim[atom])
        .collect::<Vec<_>>();
    let term_seed = build_sae_support_term_seed(SaeSupportTermSeedRequest {
        assignment: seed.assignment,
        atom_basis: atom_basis.clone(),
        atom_dim: atom_dim.clone(),
        output_dim,
        random_state: request.random_state,
    })
    .map_err(py_value_error)?;
    let ard_precisions = (0..term_seed.term.k_atoms())
        .map(|atom| vec![1.0; term_seed.term.assignment.atom_coord_dim(atom)])
        .collect::<Vec<_>>();
    let outer = run_sae_support_outer(SaeSupportOuterRequest {
        term: term_seed.term,
        target: centered_target.clone(),
        initial_smoothness: request.initial_smoothness,
        ard_precisions: ard_precisions.clone(),
        max_outer_iter: request.max_iter,
        // `max_iter` is the caller's OUTER smoothing-search budget; the inner
        // fixed point gets the engine's own, which the tiered driver already
        // reads from the same declaration.
        max_inner_iter: SAE_SUPPORT_INNER_FIXED_POINT_MAX_ITER,
        inner_tolerance: request.tolerance,
        trust_radius: request.trust_radius,
        random_state: request.random_state,
    })
    .map_err(|error| py_value_error(error.to_string()))?;
    let centered_fitted = outer.term.reconstruct().map_err(py_value_error)?;
    let fitted = add_mean(centered_fitted, &training_mean);
    let residual_ss = request
        .target
        .iter()
        .zip(fitted.iter())
        .map(|(truth, fit)| (truth - fit).powi(2))
        .sum::<f64>();
    let mut total_ss = 0.0;
    for row in request.target.rows() {
        for (column, value) in row.iter().enumerate() {
            total_ss += (value - training_mean[column]).powi(2);
        }
    }
    let reconstruction_r2 = if total_ss > 0.0 {
        1.0 - residual_ss / total_ss
    } else {
        1.0
    };
    let fixed = fixed_point_json(&outer.fixed_point);
    let outer_certificate = serde_json::to_value(&outer.outer_certificate)
        .map_err(|error| py_value_error(error.to_string()))?;
    let certificates = serde_json::json!({
        "representation": "support_sparse",
        "inner_fixed_point": fixed,
        "outer_stationarity": outer_certificate,
    });
    let termination = serde_json::json!({
        "verdict": "converged",
        "inner_iterations": outer.fixed_point.iterations,
        "outer_iterations": outer.outer_iterations,
        "recurred": outer.fixed_point.recurred,
    });
    let atom_topologies =
        gam::terms::sae::atom_schema::topologies_for_bases(&atom_basis).map_err(py_value_error)?;
    let log_lambda_smooth = outer.lambda_smooth.iter().map(|value| value.ln()).collect();
    let model = SupportSparseManifoldSaeCore {
        term: outer.term,
        requested_k,
        retained_atom_indices,
        atom_basis,
        atom_dim,
        atom_topologies,
        support_k: request.support_k,
        training_mean,
        fitted,
        reconstruction_r2,
        log_lambda_smooth,
        ard_precisions,
        criterion: outer.criterion,
        certificates,
        termination,
        max_iter: request.max_iter,
        trust_radius: request.trust_radius,
        tolerance: request.tolerance,
        random_state: request.random_state,
    };
    Ok(Py::new(py, model)?.into_any())
}
