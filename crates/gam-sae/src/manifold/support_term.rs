//! Canonical support-sparse curved term and fixed-point inner solve.
//!
//! Hard-TopK gates are read-only binary support. Consequently a row's only
//! live local parameters are the heterogeneous coordinates
//! `concat_{k in S_i} t_ik`; no gate/logit coordinate exists. This term owns
//! that representation directly and evaluates basis values and analytic jets
//! only for active `(row, atom)` pairs.

use crate::assignment::AssignmentMode;
use crate::assignment_state::{SaeAssignmentAtomSpec, SaeAssignmentState};
use gam_linalg::utils::KahanSum;
use ndarray::{Array1, Array2, ArrayView2};
use rayon::prelude::*;
use std::ops::Range;
use std::sync::Arc;

use super::*;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SaeSupportStationarity {
    pub decoder_l2: f64,
    pub decoder_max_abs: f64,
    pub coordinate_l2: f64,
    pub coordinate_max_abs: f64,
}

impl SaeSupportStationarity {
    pub fn max_abs(self) -> f64 {
        self.decoder_max_abs.max(self.coordinate_max_abs)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SaeSupportFixedPointReport {
    pub iterations: usize,
    pub objective: f64,
    pub stationarity: SaeSupportStationarity,
    pub max_recurrence_change: f64,
    /// True only after a second complete decoder/coordinate cycle recurs within
    /// the same tolerance at the raw (undamped) stationarity point.
    pub recurred: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SaeSupportCoordinateFixedPointReport {
    pub iterations: usize,
    pub objective: f64,
    pub coordinate_l2: f64,
    pub coordinate_max_abs: f64,
    pub max_recurrence_change: f64,
    /// True only after two complete frozen-decoder coordinate cycles recur at
    /// the raw coordinate stationarity point.
    pub recurred: bool,
}

struct ActiveAtomEval {
    phi: Array1<f64>,
    decoded: Array1<f64>,
    /// Coordinate-major decoded jet, `(d_k, P)`.
    jacobian: Array2<f64>,
}

#[derive(Clone)]
struct SupportBasisBlock {
    beta_offset: usize,
    phi: Array1<f64>,
}

#[derive(Clone)]
struct SupportLinearizedRow {
    blocks: Vec<SupportBasisBlock>,
    jacobian: Array2<f64>,
}

#[derive(Clone)]
struct SupportBetaOperator {
    rows: Vec<SupportLinearizedRow>,
    beta_offsets: Vec<usize>,
    basis_sizes: Vec<usize>,
    penalties: Vec<Array2<f64>>,
    lambda_smooth: Vec<f64>,
    output_dim: usize,
    beta_dim: usize,
}

impl SupportBetaOperator {
    fn apply(&self, vector: ndarray::ArrayView1<'_, f64>, out: &mut Array1<f64>) {
        assert_eq!(
            vector.len(),
            self.beta_dim,
            "SupportBetaOperator input width must equal its declared beta dimension"
        );
        assert_eq!(
            out.len(),
            self.beta_dim,
            "SupportBetaOperator output width must equal its declared beta dimension"
        );
        out.fill(0.0);
        let mut output = vec![0.0; self.output_dim];
        for row in &self.rows {
            output.fill(0.0);
            for block in &row.blocks {
                for basis in 0..block.phi.len() {
                    let base = block.beta_offset + basis * self.output_dim;
                    for channel in 0..self.output_dim {
                        output[channel] += block.phi[basis] * vector[base + channel];
                    }
                }
            }
            for block in &row.blocks {
                for basis in 0..block.phi.len() {
                    let base = block.beta_offset + basis * self.output_dim;
                    for channel in 0..self.output_dim {
                        out[base + channel] += block.phi[basis] * output[channel];
                    }
                }
            }
        }
        for atom in 0..self.penalties.len() {
            let lambda = self.lambda_smooth[atom];
            let m = self.basis_sizes[atom];
            let offset = self.beta_offsets[atom];
            for left in 0..m {
                for right in 0..m {
                    let weight = lambda * self.penalties[atom][[left, right]];
                    for channel in 0..self.output_dim {
                        out[offset + left * self.output_dim + channel] +=
                            weight * vector[offset + right * self.output_dim + channel];
                    }
                }
            }
        }
    }

    fn htbeta_forward(
        &self,
        row: usize,
        vector: ndarray::ArrayView1<'_, f64>,
        out: &mut Array1<f64>,
    ) {
        let linearized = &self.rows[row];
        let mut output = vec![0.0; self.output_dim];
        for block in &linearized.blocks {
            for basis in 0..block.phi.len() {
                let base = block.beta_offset + basis * self.output_dim;
                for channel in 0..self.output_dim {
                    output[channel] += block.phi[basis] * vector[base + channel];
                }
            }
        }
        out.fill(0.0);
        for axis in 0..linearized.jacobian.nrows() {
            for channel in 0..self.output_dim {
                out[axis] += linearized.jacobian[[axis, channel]] * output[channel];
            }
        }
    }

    fn htbeta_transpose(
        &self,
        row: usize,
        vector: ndarray::ArrayView1<'_, f64>,
        out: &mut Array1<f64>,
    ) {
        let linearized = &self.rows[row];
        let mut output = vec![0.0; self.output_dim];
        for axis in 0..linearized.jacobian.nrows() {
            for channel in 0..self.output_dim {
                output[channel] += linearized.jacobian[[axis, channel]] * vector[axis];
            }
        }
        for block in &linearized.blocks {
            for basis in 0..block.phi.len() {
                let base = block.beta_offset + basis * self.output_dim;
                for channel in 0..self.output_dim {
                    out[base + channel] += block.phi[basis] * output[channel];
                }
            }
        }
    }
}

/// One hard-TopK curved model with no dense assignment specialization.
#[derive(Debug, Clone)]
pub struct SaeSupportSparseTerm {
    pub atoms: Vec<SaeManifoldAtom>,
    pub assignment: SaeAssignmentState,
    output_dim: usize,
    /// Inverted support index. Total entries are exactly `N·support_k`.
    atom_rows: Vec<Vec<(usize, usize)>>,
}

impl SaeSupportSparseTerm {
    #[must_use = "term construction error must be handled"]
    pub fn new(
        atoms: Vec<SaeManifoldAtom>,
        assignment: SaeAssignmentState,
    ) -> Result<Self, String> {
        let k_atoms = atoms.len();
        if k_atoms == 0 || assignment.k_atoms() != k_atoms {
            return Err(format!(
                "SaeSupportSparseTerm::new: atom count {k_atoms} != assignment K={}",
                assignment.k_atoms()
            ));
        }
        let support_k = match assignment.mode() {
            AssignmentMode::TopK { k } => k,
            other => {
                return Err(format!(
                    "SaeSupportSparseTerm::new requires hard TopK assignment state; got {other:?}"
                ));
            }
        };
        let output_dim = atoms[0].output_dim();
        if output_dim == 0 {
            return Err(
                "SaeSupportSparseTerm::new: decoder output dimension must be positive".into(),
            );
        }
        for (atom, template) in atoms.iter().enumerate() {
            if template.output_dim() != output_dim {
                return Err(format!(
                    "SaeSupportSparseTerm::new: atom {atom} output dimension {} != {output_dim}",
                    template.output_dim()
                ));
            }
            if template.latent_dim() != assignment.atom_coord_dim(atom) {
                return Err(format!(
                    "SaeSupportSparseTerm::new: atom {atom} latent dim {} != assignment dim {}",
                    template.latent_dim(),
                    assignment.atom_coord_dim(atom)
                ));
            }
            if template.basis_evaluator.is_none() {
                return Err(format!(
                    "SaeSupportSparseTerm::new: atom {atom} has no analytic basis evaluator"
                ));
            }
        }
        let mut atom_rows = vec![Vec::new(); k_atoms];
        for row in 0..assignment.n_obs() {
            let support = assignment.support_indices(row);
            if support.len() != support_k {
                return Err(format!(
                    "SaeSupportSparseTerm::new: row {row} support width {} != top_k={support_k}",
                    support.len()
                ));
            }
            for (slot, &atom) in support.iter().enumerate() {
                atom_rows[atom as usize].push((row, slot));
            }
        }
        Ok(Self {
            atoms,
            assignment,
            output_dim,
            atom_rows,
        })
    }

    pub fn n_obs(&self) -> usize {
        self.assignment.n_obs()
    }

    pub fn k_atoms(&self) -> usize {
        self.atoms.len()
    }

    pub fn output_dim(&self) -> usize {
        self.output_dim
    }

    pub fn active_pair_count(&self) -> usize {
        self.atom_rows.iter().map(Vec::len).sum()
    }

    /// Route new rows against this fitted decoder without constructing a
    /// `rows × K` score matrix. Candidate reconstruction improvements are
    /// streamed one atom at a time and only the best `support_k` candidates,
    /// including their heterogeneous coordinates, survive for each row.
    pub fn reroute_fixed_decoder(
        &self,
        target: ArrayView2<'_, f64>,
        support_k: usize,
        random_state: u64,
    ) -> Result<Self, String> {
        if target.ncols() != self.output_dim || target.nrows() == 0 {
            return Err(format!(
                "SaeSupportSparseTerm::reroute_fixed_decoder: target {:?} must have positive rows and P={}",
                target.dim(),
                self.output_dim
            ));
        }
        if support_k == 0 || support_k > self.k_atoms() {
            return Err(format!(
                "SaeSupportSparseTerm::reroute_fixed_decoder requires 1 <= support_k <= K={}; got {support_k}",
                self.k_atoms()
            ));
        }
        if target.iter().any(|value| !value.is_finite()) {
            return Err(
                "SaeSupportSparseTerm::reroute_fixed_decoder: target contains a non-finite value"
                    .into(),
            );
        }

        struct Candidate {
            atom: usize,
            score: f64,
            coords: Vec<f64>,
        }
        let better = |left: &Candidate, right: &Candidate| {
            left.score > right.score || (left.score == right.score && left.atom < right.atom)
        };
        let mut indices = Vec::with_capacity(target.nrows());
        let mut gate_params = Vec::with_capacity(target.nrows());
        let mut coords = Vec::with_capacity(target.nrows());
        for row in target.rows() {
            let row_values = row.as_slice().ok_or_else(|| {
                "SaeSupportSparseTerm::reroute_fixed_decoder: target row is not contiguous"
                    .to_string()
            })?;
            let mut selected = Vec::<Candidate>::with_capacity(support_k);
            for (atom_index, atom) in self.atoms.iter().enumerate() {
                let candidate_coords = (0..atom.latent_dim())
                    .map(|axis| {
                        let raw = super::support_seed::projection(
                            row_values,
                            atom_index,
                            axis + 1,
                            random_state,
                        );
                        super::support_seed::chart_coordinate(atom.basis_kind(), axis, raw)
                    })
                    .collect::<Vec<_>>();
                let coordinate =
                    Array2::from_shape_vec((1, atom.latent_dim()), candidate_coords.clone())
                        .map_err(|error| {
                            format!("SaeSupportSparseTerm::reroute_fixed_decoder: {error}")
                        })?;
                let evaluator = atom.basis_evaluator.as_ref().ok_or_else(|| {
                    format!(
                        "SaeSupportSparseTerm::reroute_fixed_decoder: atom {atom_index} has no evaluator"
                    )
                })?;
                let (phi, _) = evaluator.evaluate(coordinate.view())?;
                let decoded = phi.row(0).dot(&atom.decoder_coefficients);
                let score = row
                    .iter()
                    .zip(decoded.iter())
                    .map(|(truth, fit)| 2.0 * truth * fit - fit * fit)
                    .sum::<f64>();
                let candidate = Candidate {
                    atom: atom_index,
                    score,
                    coords: candidate_coords,
                };
                if selected.len() < support_k {
                    selected.push(candidate);
                } else {
                    let mut worst = 0usize;
                    for slot in 1..selected.len() {
                        if better(&selected[worst], &selected[slot]) {
                            worst = slot;
                        }
                    }
                    if better(&candidate, &selected[worst]) {
                        selected[worst] = candidate;
                    }
                }
            }
            selected.sort_by_key(|candidate| candidate.atom);
            indices.push(
                selected
                    .iter()
                    .map(|candidate| candidate.atom as u32)
                    .collect(),
            );
            gate_params.push(selected.iter().map(|candidate| candidate.score).collect());
            coords.push(
                selected
                    .into_iter()
                    .flat_map(|candidate| candidate.coords)
                    .collect(),
            );
        }
        let atom_specs = self
            .atoms
            .iter()
            .enumerate()
            .map(|(atom, template)| SaeAssignmentAtomSpec {
                latent_dim: template.latent_dim(),
                id_mode: gam_terms::latent::LatentIdMode::None,
                manifold: template.basis_kind().latent_manifold(template.latent_dim()),
                retraction: gam_problem::LatentRetractionRegistry::all_euclidean(),
                latent_id: super::support_seed::splitmix64(atom as u64),
            })
            .collect();
        let assignment = SaeAssignmentState::from_topk_support_heterogeneous(
            target.nrows(),
            self.k_atoms(),
            support_k,
            atom_specs,
            indices,
            gate_params,
            coords,
        )?;
        Self::new(self.atoms.clone(), assignment)
    }

    pub(crate) fn beta_layout(&self) -> Result<(Vec<usize>, usize), String> {
        let mut offsets = Vec::with_capacity(self.k_atoms());
        let mut cursor = 0usize;
        for atom in &self.atoms {
            offsets.push(cursor);
            cursor =
                cursor
                    .checked_add(atom.basis_size().checked_mul(self.output_dim).ok_or_else(
                        || "SaeSupportSparseTerm: beta block width overflow".to_string(),
                    )?)
                    .ok_or_else(|| "SaeSupportSparseTerm: beta dimension overflow".to_string())?;
        }
        Ok((offsets, cursor))
    }

    /// Assemble the exact support-row Gauss-Newton Arrow system. `H_bb` and
    /// every `H_tb` row are installed as sparse matvec/adjoint operators; the
    /// only resident row matrices are `q_i×q_i`, with
    /// `q_i = sum_{k in S_i} d_k`.
    pub fn assemble_arrow_schur(
        &self,
        target: ArrayView2<'_, f64>,
        lambda_smooth: &[f64],
        ard_precisions: &[Vec<f64>],
    ) -> Result<ArrowSchurSystem, String> {
        if target.dim() != (self.n_obs(), self.output_dim) {
            return Err(format!(
                "SaeSupportSparseTerm::assemble_arrow_schur: target {:?} != ({}, {})",
                target.dim(),
                self.n_obs(),
                self.output_dim
            ));
        }
        self.validate_smoothing(lambda_smooth)?;
        if ard_precisions.len() != self.k_atoms() {
            return Err(format!(
                "SaeSupportSparseTerm::assemble_arrow_schur: ARD blocks {} != K={}",
                ard_precisions.len(),
                self.k_atoms()
            ));
        }
        for (atom, values) in ard_precisions.iter().enumerate() {
            if values.len() != self.assignment.atom_coord_dim(atom)
                || values
                    .iter()
                    .any(|value| !value.is_finite() || *value <= 0.0)
            {
                return Err(format!(
                    "SaeSupportSparseTerm::assemble_arrow_schur: atom {atom} ARD must contain {} finite positive precisions",
                    self.assignment.atom_coord_dim(atom)
                ));
            }
        }
        let (beta_offsets, beta_dim) = self.beta_layout()?;
        let row_layout = SaeRowLayout::from_assignment_state(&self.assignment)?;
        let per_row_dims = (0..self.n_obs())
            .map(|row| row_layout.row_q_active(row))
            .collect::<Vec<_>>();
        let mut system = ArrowSchurSystem::new_with_per_row_dims_empty_hbb_and_htbeta_cols(
            per_row_dims,
            beta_dim,
            0,
        );
        let mut linearized_rows = Vec::with_capacity(self.n_obs());
        let mut hbb_diag = Array1::<f64>::zeros(beta_dim);
        for row in 0..self.n_obs() {
            let q = row_layout.row_q_active(row);
            let mut fitted = Array1::<f64>::zeros(self.output_dim);
            let mut jacobian = Array2::<f64>::zeros((q, self.output_dim));
            let mut blocks = Vec::with_capacity(self.assignment.support_indices(row).len());
            for slot in 0..self.assignment.support_indices(row).len() {
                let atom_idx = self.assignment.support_indices(row)[slot] as usize;
                let active = self.evaluate_active(row, slot)?;
                fitted += &active.decoded;
                let cursor = row_layout.coord_starts[row][slot];
                for axis in 0..active.jacobian.nrows() {
                    jacobian
                        .row_mut(cursor + axis)
                        .assign(&active.jacobian.row(axis));
                }
                for basis in 0..active.phi.len() {
                    let base = beta_offsets[atom_idx] + basis * self.output_dim;
                    for channel in 0..self.output_dim {
                        hbb_diag[base + channel] += active.phi[basis] * active.phi[basis];
                    }
                }
                blocks.push(SupportBasisBlock {
                    beta_offset: beta_offsets[atom_idx],
                    phi: active.phi,
                });
            }
            let residual = &target.row(row) - &fitted;
            system.rows[row].htt.assign(&jacobian.dot(&jacobian.t()));
            system.rows[row].gt.assign(&(-jacobian.dot(&residual)));
            let periods = self
                .assignment
                .support_indices(row)
                .iter()
                .flat_map(|&atom| self.assignment.atom_axis_periods(atom as usize))
                .collect::<Vec<_>>();
            let mut coord_cursor = 0usize;
            for (slot, &atom) in self.assignment.support_indices(row).iter().enumerate() {
                let atom = atom as usize;
                for axis in 0..self.assignment.atom_coord_dim(atom) {
                    let coordinate = self.assignment.coords_for_slot(row, slot)[axis];
                    let prior = ArdAxisPrior::eval(
                        ard_precisions[atom][axis],
                        coordinate,
                        periods[coord_cursor],
                    );
                    system.rows[row].gt[coord_cursor] += prior.grad;
                    system.rows[row].htt[[coord_cursor, coord_cursor]] +=
                        prior.psd_majorizer_hess();
                    coord_cursor += 1;
                }
            }
            for block in &blocks {
                for basis in 0..block.phi.len() {
                    let base = block.beta_offset + basis * self.output_dim;
                    for channel in 0..self.output_dim {
                        system.gb[base + channel] -= block.phi[basis] * residual[channel];
                    }
                }
            }
            linearized_rows.push(SupportLinearizedRow { blocks, jacobian });
        }
        for atom in 0..self.k_atoms() {
            let m = self.atoms[atom].basis_size();
            let lambda = lambda_smooth[atom];
            let sb = self.atoms[atom]
                .smooth_penalty()
                .dot(&self.atoms[atom].decoder_coefficients);
            for basis in 0..m {
                let base = beta_offsets[atom] + basis * self.output_dim;
                for channel in 0..self.output_dim {
                    system.gb[base + channel] += lambda * sb[[basis, channel]];
                    hbb_diag[base + channel] +=
                        lambda * self.atoms[atom].smooth_penalty()[[basis, basis]];
                }
            }
        }
        let operator = Arc::new(SupportBetaOperator {
            rows: linearized_rows,
            beta_offsets: beta_offsets.clone(),
            basis_sizes: self.atoms.iter().map(SaeManifoldAtom::basis_size).collect(),
            penalties: self
                .atoms
                .iter()
                .map(|atom| atom.smooth_penalty().clone())
                .collect(),
            lambda_smooth: lambda_smooth.to_vec(),
            output_dim: self.output_dim,
            beta_dim,
        });
        let shared = Arc::clone(&operator);
        system.set_shared_beta_operator(move |vector, out| shared.apply(vector, out), hbb_diag);
        let forward = Arc::clone(&operator);
        let transpose = Arc::clone(&operator);
        system.set_row_htbeta_operator(
            move |row, vector, out| forward.htbeta_forward(row, vector, out),
            move |row, vector, out| transpose.htbeta_transpose(row, vector, out),
        );
        let block_offsets: Arc<[Range<usize>]> = self
            .atoms
            .iter()
            .enumerate()
            .map(|(atom, template)| {
                beta_offsets[atom]..beta_offsets[atom] + template.basis_size() * self.output_dim
            })
            .collect::<Vec<_>>()
            .into();
        system.set_block_offsets(block_offsets);
        system.refresh_row_hessian_fingerprint();
        Ok(system)
    }

    fn evaluate_active(&self, row: usize, slot: usize) -> Result<ActiveAtomEval, String> {
        let atom_idx = self.assignment.support_indices(row)[slot] as usize;
        let atom = &self.atoms[atom_idx];
        let d = atom.latent_dim();
        let coords =
            Array2::from_shape_vec((1, d), self.assignment.coords_for_slot(row, slot).to_vec())
                .map_err(|error| format!("SaeSupportSparseTerm::evaluate_active: {error}"))?;
        let evaluator = atom.basis_evaluator.as_ref().ok_or_else(|| {
            format!("SaeSupportSparseTerm::evaluate_active: atom {atom_idx} has no evaluator")
        })?;
        let (phi, jet) = evaluator.evaluate(coords.view())?;
        let m = atom.basis_size();
        if phi.dim() != (1, m) || jet.dim() != (1, m, d) {
            return Err(format!(
                "SaeSupportSparseTerm::evaluate_active: atom {atom_idx} evaluator shapes Phi={:?}, jet={:?}, expected (1,{m}) and (1,{m},{d})",
                phi.dim(),
                jet.dim()
            ));
        }
        let phi = phi.row(0).to_owned();
        let decoded = phi.dot(&atom.decoder_coefficients);
        let mut jacobian = Array2::<f64>::zeros((d, self.output_dim));
        for axis in 0..d {
            for basis in 0..m {
                let weight = jet[[0, basis, axis]];
                for output in 0..self.output_dim {
                    jacobian[[axis, output]] += weight * atom.decoder_coefficients[[basis, output]];
                }
            }
        }
        Ok(ActiveAtomEval {
            phi,
            decoded,
            jacobian,
        })
    }

    /// Decode one atom's image at caller coordinates: `Φ(t)·B_k`, shape
    /// `(n, P)`. The atom's own evaluator is the single source of truth for
    /// the chart convention — callers never re-derive the basis.
    pub fn decode_atom_at(
        &self,
        atom_idx: usize,
        coords: ArrayView2<'_, f64>,
    ) -> Result<Array2<f64>, String> {
        if atom_idx >= self.k_atoms() {
            return Err(format!(
                "SaeSupportSparseTerm::decode_atom_at: atom {atom_idx} out of range K={}",
                self.k_atoms()
            ));
        }
        let atom = &self.atoms[atom_idx];
        if coords.ncols() != atom.latent_dim() {
            return Err(format!(
                "SaeSupportSparseTerm::decode_atom_at: coords width {} != atom latent dim {}",
                coords.ncols(),
                atom.latent_dim()
            ));
        }
        let evaluator = atom.basis_evaluator.as_ref().ok_or_else(|| {
            format!("SaeSupportSparseTerm::decode_atom_at: atom {atom_idx} has no evaluator")
        })?;
        let (phi, _jet) = evaluator.evaluate(coords)?;
        Ok(phi.dot(&atom.decoder_coefficients))
    }

    fn reconstruct_row(&self, row: usize) -> Result<Array1<f64>, String> {
        let mut fitted = Array1::<f64>::zeros(self.output_dim);
        for slot in 0..self.assignment.support_indices(row).len() {
            let active = self.evaluate_active(row, slot)?;
            fitted += &active.decoded;
        }
        Ok(fitted)
    }

    /// Direct active-row reconstruction. No K-wide gate or basis row exists.
    /// Rows are independent reads of shared state, so they decode in parallel.
    pub fn reconstruct(&self) -> Result<Array2<f64>, String> {
        let rows = (0..self.n_obs())
            .into_par_iter()
            .map(|row| self.reconstruct_row(row))
            .collect::<Result<Vec<_>, String>>()?;
        let mut fitted = Array2::<f64>::zeros((self.n_obs(), self.output_dim));
        for (row, decoded) in rows.into_iter().enumerate() {
            fitted.row_mut(row).assign(&decoded);
        }
        Ok(fitted)
    }

    /// Raw response residual `target - fitted`, deliberately before any
    /// smoothing or coordinate-prior transformation.
    pub fn raw_residual(&self, target: ArrayView2<'_, f64>) -> Result<Array2<f64>, String> {
        if target.dim() != (self.n_obs(), self.output_dim) {
            return Err(format!(
                "SaeSupportSparseTerm::raw_residual: target {:?} != ({}, {})",
                target.dim(),
                self.n_obs(),
                self.output_dim
            ));
        }
        Ok(&target - &self.reconstruct()?)
    }

    fn validate_smoothing(&self, lambda_smooth: &[f64]) -> Result<(), String> {
        if lambda_smooth.len() != self.k_atoms() {
            return Err(format!(
                "SaeSupportSparseTerm: smoothing length {} != K={}",
                lambda_smooth.len(),
                self.k_atoms()
            ));
        }
        if lambda_smooth
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(
                "SaeSupportSparseTerm: smoothing strengths must be finite and non-negative".into(),
            );
        }
        Ok(())
    }

    fn validate_ard(&self, ard_precisions: &[Vec<f64>]) -> Result<(), String> {
        if ard_precisions.len() != self.k_atoms() {
            return Err(format!(
                "SaeSupportSparseTerm: ARD blocks {} != K={}",
                ard_precisions.len(),
                self.k_atoms()
            ));
        }
        for (atom, values) in ard_precisions.iter().enumerate() {
            if values.len() != self.assignment.atom_coord_dim(atom)
                || values
                    .iter()
                    .any(|value| !value.is_finite() || *value <= 0.0)
            {
                return Err(format!(
                    "SaeSupportSparseTerm: atom {atom} ARD must contain {} finite positive precisions",
                    self.assignment.atom_coord_dim(atom)
                ));
            }
        }
        Ok(())
    }

    /// Gaussian loss plus the declared final-function seminorm
    /// `0.5 λ_k tr(B_k' S_ref,k B_k)`.
    pub fn penalized_objective(
        &self,
        target: ArrayView2<'_, f64>,
        lambda_smooth: &[f64],
        ard_precisions: &[Vec<f64>],
    ) -> Result<f64, String> {
        self.validate_smoothing(lambda_smooth)?;
        self.validate_ard(ard_precisions)?;
        let residual = self.raw_residual(target)?;
        self.penalized_objective_with_residual(&residual, lambda_smooth, ard_precisions)
    }

    /// [`Self::penalized_objective`] against a caller-supplied residual.
    pub fn penalized_objective_with_residual(
        &self,
        residual: &Array2<f64>,
        lambda_smooth: &[f64],
        ard_precisions: &[Vec<f64>],
    ) -> Result<f64, String> {
        self.validate_smoothing(lambda_smooth)?;
        self.validate_ard(ard_precisions)?;
        let mut value = 0.5 * residual.iter().map(|entry| entry * entry).sum::<f64>();
        for (atom, &lambda) in self.atoms.iter().zip(lambda_smooth) {
            let sb = atom.smooth_penalty().dot(&atom.decoder_coefficients);
            value += 0.5
                * lambda
                * atom
                    .decoder_coefficients
                    .iter()
                    .zip(sb.iter())
                    .map(|(left, right)| left * right)
                    .sum::<f64>();
        }
        value += (0..self.n_obs())
            .into_par_iter()
            .map(|row| {
                let mut row_value = 0.0_f64;
                for (slot, &atom) in self.assignment.support_indices(row).iter().enumerate() {
                    let atom = atom as usize;
                    let periods = self.assignment.atom_axis_periods(atom);
                    for axis in 0..self.assignment.atom_coord_dim(atom) {
                        row_value += ArdAxisPrior::eval(
                            ard_precisions[atom][axis],
                            self.assignment.coords_for_slot(row, slot)[axis],
                            periods[axis],
                        )
                        .value;
                    }
                }
                row_value
            })
            .sum::<f64>();
        if value.is_finite() {
            Ok(value)
        } else {
            Err("SaeSupportSparseTerm::penalized_objective is non-finite".into())
        }
    }

    /// Canonical Moore-Penrose solution of a symmetric PSD normal equation.
    /// Null directions are set to zero; an RHS component in the numerical null
    /// space is a malformed normal equation and is refused.
    fn solve_psd_minimum_norm(
        gram: &Array2<f64>,
        rhs: &Array2<f64>,
        context: &str,
    ) -> Result<Array2<f64>, String> {
        let m = gram.nrows();
        if gram.dim() != (m, m) || rhs.nrows() != m {
            return Err(format!(
                "{context}: normal-equation shape mismatch gram={:?}, rhs={:?}",
                gram.dim(),
                rhs.dim()
            ));
        }
        let symmetric = (gram + &gram.t()) * 0.5;
        let (eigenvalues, eigenvectors) = symmetric
            .eigh(Side::Lower)
            .map_err(|error| format!("{context}: eigendecomposition failed: {error}"))?;
        let scale = eigenvalues
            .iter()
            .map(|value| value.abs())
            .fold(0.0_f64, f64::max);
        let tolerance = f64::EPSILON * scale * m.max(1) as f64;
        if eigenvalues.iter().any(|value| *value < -tolerance) {
            return Err(format!(
                "{context}: normal equation is not positive semidefinite"
            ));
        }
        let projected = eigenvectors.t().dot(rhs);
        let rhs_scale = rhs.iter().map(|value| value.abs()).fold(0.0_f64, f64::max);
        let rhs_tolerance = f64::EPSILON * rhs_scale * m.max(1) as f64;
        let mut scaled = Array2::<f64>::zeros(projected.dim());
        for mode in 0..m {
            if eigenvalues[mode] > tolerance {
                for column in 0..rhs.ncols() {
                    scaled[[mode, column]] = projected[[mode, column]] / eigenvalues[mode];
                }
            } else if projected
                .row(mode)
                .iter()
                .any(|value| value.abs() > rhs_tolerance)
            {
                return Err(format!(
                    "{context}: RHS has a component in the normal-equation null space"
                ));
            }
        }
        Ok(eigenvectors.dot(&scaled))
    }

    /// One deterministic Gauss-Seidel decoder sweep. Each block update is the
    /// exact minimum-norm minimizer of the current final-function-penalized
    /// quadratic, not a coefficient-ridge surrogate.
    /// Greedy conflict coloring of atoms by shared rows: atoms in one color
    /// class touch pairwise-disjoint row sets, so their Gauss-Seidel updates
    /// commute exactly.
    fn decoder_conflict_colors(&self) -> Vec<Vec<usize>> {
        let mut row_atoms: Vec<Vec<u32>> = vec![Vec::new(); self.n_obs()];
        for (atom_idx, rows) in self.atom_rows.iter().enumerate() {
            for &(row, _slot) in rows {
                row_atoms[row].push(atom_idx as u32);
            }
        }
        let mut color_of: Vec<u32> = vec![u32::MAX; self.k_atoms()];
        let mut classes: Vec<Vec<usize>> = Vec::new();
        let mut used: Vec<u32> = Vec::new();
        for atom_idx in 0..self.k_atoms() {
            used.clear();
            for &(row, _slot) in &self.atom_rows[atom_idx] {
                for &other in &row_atoms[row] {
                    let color = color_of[other as usize];
                    if color != u32::MAX {
                        used.push(color);
                    }
                }
            }
            used.sort_unstable();
            used.dedup();
            let mut color = 0u32;
            for &taken in &used {
                if taken == color {
                    color += 1;
                } else if taken > color {
                    break;
                }
            }
            color_of[atom_idx] = color;
            if classes.len() <= color as usize {
                classes.resize(color as usize + 1, Vec::new());
            }
            classes[color as usize].push(atom_idx);
        }
        classes
    }

    fn decoder_sweep(
        &mut self,
        target: ArrayView2<'_, f64>,
        lambda_smooth: &[f64],
    ) -> Result<f64, String> {
        self.validate_smoothing(lambda_smooth)?;
        let mut fitted = self.reconstruct()?;
        let mut max_change = 0.0_f64;
        let classes = self.decoder_conflict_colors();
        for class in &classes {
            // Atoms in one class are row-disjoint: solve in parallel against
            // the shared `fitted` snapshot (each atom reads only its own rows),
            // then apply the disjoint updates.
            let solved: Vec<(usize, Array2<f64>, Vec<(usize, Array1<f64>)>, f64)> = class
                .par_iter()
                .map(|&atom_idx| -> Result<_, String> {
                    let m = self.atoms[atom_idx].basis_size();
                    let old_decoder = &self.atoms[atom_idx].decoder_coefficients;
                    let mut gram =
                        self.atoms[atom_idx].smooth_penalty().clone() * lambda_smooth[atom_idx];
                    let mut rhs = Array2::<f64>::zeros((m, self.output_dim));
                    let mut rows = Vec::with_capacity(self.atom_rows[atom_idx].len());
                    for &(row, slot) in &self.atom_rows[atom_idx] {
                        let active = self.evaluate_active(row, slot)?;
                        for left in 0..m {
                            for right in 0..m {
                                gram[[left, right]] += active.phi[left] * active.phi[right];
                            }
                            for output in 0..self.output_dim {
                                let residual_without = target[[row, output]]
                                    - fitted[[row, output]]
                                    + active.decoded[output];
                                rhs[[left, output]] += active.phi[left] * residual_without;
                            }
                        }
                        rows.push((row, active.phi, active.decoded));
                    }
                    let decoder = Self::solve_psd_minimum_norm(
                        &gram,
                        &rhs,
                        "SaeSupportSparseTerm::decoder_sweep",
                    )?;
                    let mut atom_change = 0.0_f64;
                    for (new, old) in decoder.iter().zip(old_decoder.iter()) {
                        atom_change = atom_change.max((new - old).abs());
                    }
                    let mut row_updates = Vec::with_capacity(rows.len());
                    for (row, phi, old_decoded) in rows {
                        let new_decoded = phi.dot(&decoder);
                        let mut delta = new_decoded;
                        delta -= &old_decoded;
                        row_updates.push((row, delta));
                    }
                    Ok((atom_idx, decoder, row_updates, atom_change))
                })
                .collect::<Result<Vec<_>, String>>()?;
            for (atom_idx, decoder, row_updates, atom_change) in solved {
                max_change = max_change.max(atom_change);
                self.atoms[atom_idx].decoder_coefficients = decoder;
                for (row, delta) in row_updates {
                    for output in 0..self.output_dim {
                        fitted[[row, output]] += delta[output];
                    }
                }
            }
        }
        Ok(max_change)
    }

    /// One direct active-row Gauss-Newton coordinate sweep with manifold-aware
    /// backtracking. Exact row snapshots provide rollback; inverse retractions
    /// are never assumed.
    fn coordinate_sweep(
        &mut self,
        target: ArrayView2<'_, f64>,
        ard_precisions: &[Vec<f64>],
        trust_radius: f64,
        stationarity_tolerance: f64,
    ) -> Result<f64, String> {
        self.validate_ard(ard_precisions)?;
        if !(trust_radius.is_finite() && trust_radius > 0.0) {
            return Err(format!(
                "SaeSupportSparseTerm::coordinate_sweep: trust_radius must be finite and positive; got {trust_radius}"
            ));
        }
        if !(stationarity_tolerance.is_finite() && stationarity_tolerance > 0.0) {
            return Err(format!(
                "SaeSupportSparseTerm::coordinate_sweep: stationarity tolerance must be finite and positive; got {stationarity_tolerance}"
            ));
        }
        // Rows are independent given the frozen decoder: each owns a disjoint
        // coordinate block. Take the storage so rows solve in parallel with
        // `self` shared-read, then put it back (also on a row error).
        let mut coords_rows = self.assignment.take_coords();
        let row_results: Vec<Result<f64, String>> = coords_rows
            .par_iter_mut()
            .enumerate()
            .map(|(row, coords_row)| {
                self.row_coordinate_solve(
                    row,
                    coords_row,
                    target,
                    ard_precisions,
                    trust_radius,
                    stationarity_tolerance,
                )
            })
            .collect();
        self.assignment.restore_coords(coords_rows)?;
        let mut max_change = 0.0_f64;
        for row_result in row_results {
            max_change = max_change.max(row_result?);
        }
        Ok(max_change)
    }

    /// Per-slot offset ranges into a row's compact coordinate block.
    fn slot_offsets(&self, row: usize) -> Vec<Range<usize>> {
        let mut out = Vec::with_capacity(self.assignment.support_indices(row).len());
        let mut cursor = 0usize;
        for &atom in self.assignment.support_indices(row) {
            let d = self.assignment.atom_coord_dim(atom as usize);
            out.push(cursor..cursor + d);
            cursor += d;
        }
        out
    }

    /// Fill one active slot's basis row, jet, decoded image, and coordinate
    /// Jacobian into caller-owned buffers — the allocation-free counterpart of
    /// [`Self::evaluate_active`] for the parallel row solve. The profiled
    /// inner-cycle cost (98.6% of every core in `__memset`) was these buffers
    /// being freshly zero-allocated for every slot of every line-search trial
    /// of every row; the basis itself goes through the trait's
    /// [`SaeBasisEvaluator::evaluate_into`].
    fn fill_active_eval(
        &self,
        row: usize,
        slot: usize,
        slot_coords: &[f64],
        phi: &mut Array2<f64>,
        jet: &mut ndarray::Array3<f64>,
        decoded: &mut Array1<f64>,
        jacobian: &mut Array2<f64>,
    ) -> Result<(), String> {
        let atom_idx = self.assignment.support_indices(row)[slot] as usize;
        let atom = &self.atoms[atom_idx];
        let d = atom.latent_dim();
        let m = atom.basis_size();
        if slot_coords.len() != d
            || phi.dim() != (1, m)
            || jet.dim() != (1, m, d)
            || decoded.len() != self.output_dim
            || jacobian.dim() != (d, self.output_dim)
        {
            return Err(format!(
                "SaeSupportSparseTerm::fill_active_eval: atom {atom_idx} buffer shapes \
                 coords={}, phi={:?}, jet={:?}, decoded={}, jacobian={:?} do not match \
                 (m={m}, d={d}, p={})",
                slot_coords.len(),
                phi.dim(),
                jet.dim(),
                decoded.len(),
                jacobian.dim(),
                self.output_dim
            ));
        }
        let coords = ndarray::ArrayView2::from_shape((1, d), slot_coords)
            .map_err(|error| format!("SaeSupportSparseTerm::fill_active_eval: {error}"))?;
        let evaluator = atom.basis_evaluator.as_ref().ok_or_else(|| {
            format!("SaeSupportSparseTerm::fill_active_eval: atom {atom_idx} has no evaluator")
        })?;
        evaluator.evaluate_into(phi, jet, coords)?;
        decoded.fill(0.0);
        for basis in 0..m {
            let weight = phi[[0, basis]];
            for output in 0..self.output_dim {
                decoded[output] += weight * atom.decoder_coefficients[[basis, output]];
            }
        }
        jacobian.fill(0.0);
        for axis in 0..d {
            for basis in 0..m {
                let weight = jet[[0, basis, axis]];
                for output in 0..self.output_dim {
                    jacobian[[axis, output]] +=
                        weight * atom.decoder_coefficients[[basis, output]];
                }
            }
        }
        Ok(())
    }

    /// One row's exact Gauss-Newton coordinate step with manifold-aware
    /// backtracking, on the row's caller-held coordinate block. Semantically
    /// the serial sweep's row iteration; storage-wise a single per-row scratch
    /// filled in place, so the line-search halvings allocate nothing.
    fn row_coordinate_solve(
        &self,
        row: usize,
        coords_row: &mut Vec<f64>,
        target: ArrayView2<'_, f64>,
        ard_precisions: &[Vec<f64>],
        trust_radius: f64,
        stationarity_tolerance: f64,
    ) -> Result<f64, String> {
        let mut max_change = 0.0_f64;
        let offsets = self.slot_offsets(row);
        let n_slots = offsets.len();
        let q = coords_row.len();
        let p = self.output_dim;

        // ---- per-row scratch, allocated once ----
        let support = self.assignment.support_indices(row).to_vec();
        let dims: Vec<(usize, usize)> = support
            .iter()
            .map(|&atom| {
                let atom = atom as usize;
                (self.atoms[atom].basis_size(), self.atoms[atom].latent_dim())
            })
            .collect();
        let mut phi_cur: Vec<Array2<f64>> =
            dims.iter().map(|&(m, _)| Array2::zeros((1, m))).collect();
        let mut jet_cur: Vec<ndarray::Array3<f64>> = dims
            .iter()
            .map(|&(m, d)| ndarray::Array3::zeros((1, m, d)))
            .collect();
        let mut dec_cur: Vec<Array1<f64>> = dims.iter().map(|_| Array1::zeros(p)).collect();
        let mut jac_cur: Vec<Array2<f64>> =
            dims.iter().map(|&(_, d)| Array2::zeros((d, p))).collect();
        let mut phi_trial = phi_cur.clone();
        let mut jet_trial = jet_cur.clone();
        let mut dec_trial = dec_cur.clone();
        let mut jac_trial = jac_cur.clone();
        let mut fitted = Array1::<f64>::zeros(p);
        let mut jacobian = Array2::<f64>::zeros((q, p));
        let mut trial_fitted = Array1::<f64>::zeros(p);
        let mut trial_residual = Array1::<f64>::zeros(p);
        let mut trial_delta = vec![0.0_f64; q];
        let mut fitted_delta: Vec<KahanSum> = (0..p).map(|_| KahanSum::default()).collect();

        for slot in 0..n_slots {
            self.fill_active_eval(
                row,
                slot,
                &coords_row[offsets[slot].clone()],
                &mut phi_cur[slot],
                &mut jet_cur[slot],
                &mut dec_cur[slot],
                &mut jac_cur[slot],
            )?;
            fitted += &dec_cur[slot];
            for axis in 0..dims[slot].1 {
                jacobian
                    .row_mut(offsets[slot].start + axis)
                    .assign(&jac_cur[slot].row(axis));
            }
        }
        let residual = &target.row(row) - &fitted;
        let mut row_objective_scale =
            1.0 + 0.5 * residual.iter().map(|value| value * value).sum::<f64>();
        let mut rhs_vector = jacobian.dot(&residual);
        let mut gram = jacobian.dot(&jacobian.t());
        let mut prior_cursor = 0usize;
        for (slot, &atom) in support.iter().enumerate() {
            let atom = atom as usize;
            let periods = self.assignment.atom_axis_periods(atom);
            for axis in 0..self.assignment.atom_coord_dim(atom) {
                let prior = ArdAxisPrior::eval(
                    ard_precisions[atom][axis],
                    coords_row[offsets[slot].start + axis],
                    periods[axis],
                );
                row_objective_scale += prior.value.abs();
                rhs_vector[prior_cursor] -= prior.grad;
                gram[[prior_cursor, prior_cursor]] += prior.psd_majorizer_hess();
                prior_cursor += 1;
            }
        }
        let raw_gradient_max = rhs_vector
            .iter()
            .map(|value| value.abs())
            .fold(0.0_f64, f64::max);
        // A row already satisfying the caller's KKT request is a certified
        // fixed point of this coordinate block. The row gradient scales
        // with the row's own residual energy, so the skip threshold is
        // relative to the row objective (mirroring the solve-level
        // certificate); an absolute threshold left every near-converged
        // row re-solving its trust region on every cycle.
        if raw_gradient_max <= stationarity_tolerance * row_objective_scale {
            return Ok(0.0);
        }
        let delta = gam_linalg::psd_trust_region::solve_psd_trust_region(
            gram.view(),
            rhs_vector.view(),
            trust_radius,
        )
        .map_err(|error| format!("SaeSupportSparseTerm::coordinate_sweep: {error}"))?;
        let directional = rhs_vector.dot(&delta);
        let delta_max = delta
            .iter()
            .map(|value| value.abs())
            .fold(0.0_f64, f64::max);
        if !directional.is_finite() || directional < 0.0 {
            return Err(format!(
                "SaeSupportSparseTerm::coordinate_sweep: trust-region step is not a finite descent direction (rhs_dot_delta={directional})"
            ));
        }
        // `rhsᵀ delta` is quadratic in the gradient near a stationary point.
        // Comparing it with an absolute machine epsilon therefore invents a
        // sqrt(EPSILON) gradient floor (~1.5e-8 for f64), preventing tighter
        // KKT tolerances from ever being reached. Exact zero is the only
        // no-direction case; any positive value remains a valid descent
        // certificate regardless of magnitude.
        if directional == 0.0 {
            return Ok(0.0);
        }
        let old_coords = coords_row.clone();
        let mut accepted = None;
        let mut best_gap = f64::INFINITY;
        let mut best_step = 0.0_f64;
        let mut best_objective_delta = f64::NAN;
        let mut best_armijo_bound = f64::NAN;
        let evaluation_ops = 1usize
            + p
            + q
            + dims.iter().map(|&(m, _)| m * p).sum::<usize>();
        let gamma =
            evaluation_ops as f64 * f64::EPSILON / (1.0 - evaluation_ops as f64 * f64::EPSILON);
        let objective_resolution = gamma * row_objective_scale;
        for halving in 0..=24 {
            self.assignment.project_row_coords(row, &old_coords, coords_row)?;
            let step = 2.0_f64.powi(-(halving as i32));
            for (target_slot, value) in trial_delta.iter_mut().zip(delta.iter()) {
                *target_slot = step * value;
            }
            self.assignment.retract_row_coords(row, coords_row, &trial_delta)?;
            // Evaluate f(trial) - f(old) directly. Near stationarity the
            // decrease is O(||g||^2), so subtracting two O(1) objective
            // values loses the Armijo signal at exactly sqrt(EPSILON).
            // For r = y-f and prediction change d, the data-loss increment
            // is -r'd + 1/2 d'd; the prior authority supplies equally stable
            // per-axis energy increments. Kahan accumulation preserves their
            // first-order cancellation in a wide output/coordinate block.
            let mut objective_delta = KahanSum::default();
            for accumulator in fitted_delta.iter_mut() {
                *accumulator = KahanSum::default();
            }
            for slot in 0..n_slots {
                let atom = support[slot] as usize;
                self.fill_active_eval(
                    row,
                    slot,
                    &coords_row[offsets[slot].clone()],
                    &mut phi_trial[slot],
                    &mut jet_trial[slot],
                    &mut dec_trial[slot],
                    &mut jac_trial[slot],
                )?;
                for basis in 0..dims[slot].0 {
                    // Subtract basis values before multiplying by decoder
                    // coefficients. This cancels shared constant/intercept
                    // components before rounding, instead of subtracting two
                    // already-decoded O(1) predictions to recover an O(step)
                    // difference.
                    let phi_delta = phi_trial[slot][[0, basis]] - phi_cur[slot][[0, basis]];
                    for output in 0..p {
                        fitted_delta[output].add(
                            phi_delta * self.atoms[atom].decoder_coefficients[[basis, output]],
                        );
                    }
                }
            }
            for (output, delta_sum) in fitted_delta.iter().enumerate() {
                let fitted_delta = delta_sum.sum();
                objective_delta
                    .add(fitted_delta.mul_add(0.5 * fitted_delta - residual[output], 0.0));
            }
            let mut coord_cursor = 0usize;
            for (slot, &atom) in support.iter().enumerate() {
                let atom = atom as usize;
                let periods = self.assignment.atom_axis_periods(atom);
                for axis in 0..self.assignment.atom_coord_dim(atom) {
                    objective_delta.add(ArdAxisPrior::value_delta(
                        ard_precisions[atom][axis],
                        old_coords[coord_cursor],
                        coords_row[offsets[slot].start + axis],
                        periods[axis],
                    ));
                    coord_cursor += 1;
                }
            }
            let objective_delta = objective_delta.sum();
            let armijo_bound = -1.0e-4 * step * directional;
            let gap = objective_delta - armijo_bound;
            if gap.is_finite() && gap < best_gap {
                best_gap = gap;
                best_step = step;
                best_objective_delta = objective_delta;
                best_armijo_bound = armijo_bound;
            }
            trial_fitted.fill(0.0);
            for decoded in &dec_trial {
                trial_fitted += decoded;
            }
            trial_residual.assign(&target.row(row));
            trial_residual -= &trial_fitted;
            let mut trial_gradient_max = 0.0_f64;
            for (slot, &atom) in support.iter().enumerate() {
                let atom = atom as usize;
                let periods = self.assignment.atom_axis_periods(atom);
                for axis in 0..dims[slot].1 {
                    let likelihood_gradient =
                        -jac_trial[slot].row(axis).dot(&trial_residual);
                    let gradient = likelihood_gradient
                        + ArdAxisPrior::eval(
                            ard_precisions[atom][axis],
                            coords_row[offsets[slot].start + axis],
                            periods[axis],
                        )
                        .grad;
                    trial_gradient_max = trial_gradient_max.max(gradient.abs());
                }
            }
            let armijo_accept = objective_delta.is_finite() && objective_delta <= armijo_bound;
            let roundoff_tie_accept = objective_delta.is_finite()
                && objective_delta.abs() <= objective_resolution
                && trial_gradient_max < raw_gradient_max;
            if armijo_accept || roundoff_tie_accept {
                accepted = Some(step);
                break;
            }
        }
        match accepted {
            Some(step) => {
                for value in delta.iter() {
                    max_change = max_change.max((step * value).abs());
                }
            }
            None => {
                self.assignment.project_row_coords(row, &old_coords, coords_row)?;
                return Err(format!(
                    "SaeSupportSparseTerm::coordinate_sweep: row {row} has a raw descent direction but manifold line search found no decreasing step \
                     (raw KKT max={raw_gradient_max:.17e}, rhs_dot_delta={directional:.17e}, \
                     delta_max={delta_max:.17e}, best_step={best_step:.17e}, \
                     best_objective_delta={best_objective_delta:.17e}, \
                     best_armijo_bound={best_armijo_bound:.17e}, gap={best_gap:.17e}, \
                     objective_resolution={objective_resolution:.17e})"
                ));
            }
        }
        Ok(max_change)
    }

    /// Raw (undamped) KKT residual of the exact objective.
    pub fn raw_stationarity(
        &self,
        target: ArrayView2<'_, f64>,
        lambda_smooth: &[f64],
        ard_precisions: &[Vec<f64>],
    ) -> Result<SaeSupportStationarity, String> {
        let residual = self.raw_residual(target)?;
        self.raw_stationarity_with_residual(&residual, lambda_smooth, ard_precisions)
    }

    /// [`Self::raw_stationarity`] against a caller-supplied residual, so one
    /// residual pass per fixed-point cycle serves both the certificate and the
    /// objective. Atoms and rows are independent reads of shared state; both
    /// reductions run in parallel.
    pub fn raw_stationarity_with_residual(
        &self,
        residual: &Array2<f64>,
        lambda_smooth: &[f64],
        ard_precisions: &[Vec<f64>],
    ) -> Result<SaeSupportStationarity, String> {
        self.validate_smoothing(lambda_smooth)?;
        self.validate_ard(ard_precisions)?;
        if residual.dim() != (self.n_obs(), self.output_dim) {
            return Err(format!(
                "SaeSupportSparseTerm::raw_stationarity_with_residual: residual {:?} != ({}, {})",
                residual.dim(),
                self.n_obs(),
                self.output_dim
            ));
        }
        let (decoder_sq, decoder_max) = (0..self.k_atoms())
            .into_par_iter()
            .map(|atom_idx| -> Result<(f64, f64), String> {
                let atom = &self.atoms[atom_idx];
                let mut gradient = atom.smooth_penalty().dot(&atom.decoder_coefficients)
                    * lambda_smooth[atom_idx];
                for &(row, slot) in &self.atom_rows[atom_idx] {
                    let active = self.evaluate_active(row, slot)?;
                    for basis in 0..atom.basis_size() {
                        for output in 0..self.output_dim {
                            gradient[[basis, output]] -=
                                active.phi[basis] * residual[[row, output]];
                        }
                    }
                }
                let mut sq = 0.0_f64;
                let mut max = 0.0_f64;
                for value in gradient {
                    sq += value * value;
                    max = max.max(value.abs());
                }
                Ok((sq, max))
            })
            .try_reduce(|| (0.0, 0.0), |a, b| Ok((a.0 + b.0, a.1.max(b.1))))?;
        let (coordinate_sq, coordinate_max) = (0..self.n_obs())
            .into_par_iter()
            .map(|row| -> Result<(f64, f64), String> {
                let mut sq = 0.0_f64;
                let mut max = 0.0_f64;
                for slot in 0..self.assignment.support_indices(row).len() {
                    let atom = self.assignment.support_indices(row)[slot] as usize;
                    let active = self.evaluate_active(row, slot)?;
                    let periods = self.assignment.atom_axis_periods(atom);
                    for axis in 0..active.jacobian.nrows() {
                        let mut gradient = 0.0;
                        for output in 0..self.output_dim {
                            gradient -= active.jacobian[[axis, output]] * residual[[row, output]];
                        }
                        gradient += ArdAxisPrior::eval(
                            ard_precisions[atom][axis],
                            self.assignment.coords_for_slot(row, slot)[axis],
                            periods[axis],
                        )
                        .grad;
                        sq += gradient * gradient;
                        max = max.max(gradient.abs());
                    }
                }
                Ok((sq, max))
            })
            .try_reduce(|| (0.0, 0.0), |a, b| Ok((a.0 + b.0, a.1.max(b.1))))?;
        Ok(SaeSupportStationarity {
            decoder_l2: decoder_sq.sqrt(),
            decoder_max_abs: decoder_max,
            coordinate_l2: coordinate_sq.sqrt(),
            coordinate_max_abs: coordinate_max,
        })
    }

    /// Raw coordinate KKT residual with decoder coefficients held fixed.
    pub fn raw_coordinate_stationarity(
        &self,
        target: ArrayView2<'_, f64>,
        ard_precisions: &[Vec<f64>],
    ) -> Result<(f64, f64), String> {
        self.validate_ard(ard_precisions)?;
        let residual = self.raw_residual(target)?;
        let mut coordinate_sq = 0.0;
        let mut coordinate_max = 0.0_f64;
        for row in 0..self.n_obs() {
            for slot in 0..self.assignment.support_indices(row).len() {
                let atom = self.assignment.support_indices(row)[slot] as usize;
                let active = self.evaluate_active(row, slot)?;
                let periods = self.assignment.atom_axis_periods(atom);
                for axis in 0..active.jacobian.nrows() {
                    let likelihood_gradient = active
                        .jacobian
                        .row(axis)
                        .iter()
                        .zip(residual.row(row).iter())
                        .map(|(jet, error)| -jet * error)
                        .sum::<f64>();
                    let gradient = likelihood_gradient
                        + ArdAxisPrior::eval(
                            ard_precisions[atom][axis],
                            self.assignment.coords_for_slot(row, slot)[axis],
                            periods[axis],
                        )
                        .grad;
                    coordinate_sq += gradient * gradient;
                    coordinate_max = coordinate_max.max(gradient.abs());
                }
            }
        }
        Ok((coordinate_sq.sqrt(), coordinate_max))
    }

    fn frozen_decoder_coordinate_objective(
        &self,
        target: ArrayView2<'_, f64>,
        ard_precisions: &[Vec<f64>],
    ) -> Result<f64, String> {
        let residual = self.raw_residual(target)?;
        let mut objective = 0.5 * residual.iter().map(|value| value * value).sum::<f64>();
        for row in 0..self.n_obs() {
            for (slot, &atom) in self.assignment.support_indices(row).iter().enumerate() {
                let atom = atom as usize;
                let periods = self.assignment.atom_axis_periods(atom);
                for axis in 0..self.assignment.atom_coord_dim(atom) {
                    objective += ArdAxisPrior::eval(
                        ard_precisions[atom][axis],
                        self.assignment.coords_for_slot(row, slot)[axis],
                        periods[axis],
                    )
                    .value;
                }
            }
        }
        if objective.is_finite() {
            Ok(objective)
        } else {
            Err("SaeSupportSparseTerm::frozen_decoder_coordinate_objective is non-finite".into())
        }
    }

    /// Frozen-decoder OOS coordinate solve over active supports only. A
    /// budget-exhausted or merely damped point is rejected; the returned state
    /// has recurred for two full raw-stationary coordinate cycles.
    pub fn solve_coordinates_fixed_decoder(
        &mut self,
        target: ArrayView2<'_, f64>,
        ard_precisions: &[Vec<f64>],
        max_iter: usize,
        tolerance: f64,
        trust_radius: f64,
    ) -> Result<SaeSupportCoordinateFixedPointReport, String> {
        if target.dim() != (self.n_obs(), self.output_dim) {
            return Err(format!(
                "SaeSupportSparseTerm::solve_coordinates_fixed_decoder: target {:?} != ({}, {})",
                target.dim(),
                self.n_obs(),
                self.output_dim
            ));
        }
        if max_iter == 0 || !(tolerance.is_finite() && tolerance > 0.0) {
            return Err("SaeSupportSparseTerm::solve_coordinates_fixed_decoder requires positive max_iter and finite positive tolerance".into());
        }
        let mut previous_candidate = false;
        let mut last_objective: Option<f64> = None;
        for iteration in 1..=max_iter {
            let max_change =
                self.coordinate_sweep(target, ard_precisions, trust_radius, tolerance)?;
            let (coordinate_l2, coordinate_max_abs) =
                self.raw_coordinate_stationarity(target, ard_precisions)?;
            // Same scale-invariant certificate as solve_fixed_point: the raw
            // coordinate KKT sums data gradients over the full output width,
            // so it is certified relative to max(1, |objective|).
            let objective = self.frozen_decoder_coordinate_objective(target, ard_precisions)?;
            let kkt_scale = objective.abs().max(1.0);
            let objective_recurred = last_objective
                .map(|previous: f64| (objective - previous).abs() <= tolerance * kkt_scale)
                .unwrap_or(false);
            last_objective = Some(objective);
            let candidate =
                objective_recurred && coordinate_max_abs <= tolerance * kkt_scale;
            if candidate && previous_candidate {
                return Ok(SaeSupportCoordinateFixedPointReport {
                    iterations: iteration,
                    objective,
                    coordinate_l2,
                    coordinate_max_abs,
                    max_recurrence_change: max_change,
                    recurred: true,
                });
            }
            previous_candidate = candidate;
        }
        let (_, coordinate_max_abs) = self.raw_coordinate_stationarity(target, ard_precisions)?;
        let objective = self.frozen_decoder_coordinate_objective(target, ard_precisions)?;
        Err(format!(
            "SaeSupportSparseTerm::solve_coordinates_fixed_decoder did not recur within {max_iter} cycles (raw coordinate KKT max={coordinate_max_abs:.6e}, relative to objective {objective:.6e}: {:.6e})",
            coordinate_max_abs / objective.abs().max(1.0)
        ))
    }

    /// Alternate exact decoder blocks and direct active-row coordinate Newton
    /// steps until the raw KKT residual AND a full-cycle recurrence agree. A
    /// budget-exhausted iterate is an error; only converged fits are returned.
    pub fn solve_fixed_point(
        &mut self,
        target: ArrayView2<'_, f64>,
        lambda_smooth: &[f64],
        ard_precisions: &[Vec<f64>],
        max_iter: usize,
        tolerance: f64,
        trust_radius: f64,
    ) -> Result<SaeSupportFixedPointReport, String> {
        if target.dim() != (self.n_obs(), self.output_dim) {
            return Err(format!(
                "SaeSupportSparseTerm::solve_fixed_point: target {:?} != ({}, {})",
                target.dim(),
                self.n_obs(),
                self.output_dim
            ));
        }
        if max_iter == 0 || !(tolerance.is_finite() && tolerance > 0.0) {
            return Err("SaeSupportSparseTerm::solve_fixed_point requires positive max_iter and finite positive tolerance".into());
        }
        let mut previous_candidate = false;
        let mut last_max_change = f64::NAN;
        let mut last_objective: Option<f64> = None;
        for iteration in 1..=max_iter {
            let decoder_change = self.decoder_sweep(target, lambda_smooth)?;
            let coordinate_change =
                self.coordinate_sweep(target, ard_precisions, trust_radius, tolerance)?;
            let max_change = decoder_change.max(coordinate_change);
            last_max_change = max_change;
            let residual = self.raw_residual(target)?;
            let stationarity =
                self.raw_stationarity_with_residual(&residual, lambda_smooth, ard_precisions)?;
            // The raw KKT is EXTENSIVE: each decoder entry sums per-row data
            // gradients over every row on the atom's support, so its natural
            // scale grows with rows-per-atom x residual scale. Certify the
            // scale-invariant first-order condition |g|_inf <= tol * max(1, |f|)
            // instead of an absolute bound an irreducible-residual problem can
            // never meet at any cycle budget.
            let objective =
                self.penalized_objective_with_residual(&residual, lambda_smooth, ard_precisions)?;
            let kkt_scale = objective.abs().max(1.0);
            // Both certificate limbs are relative AND gauge-invariant: the KKT
            // against the objective scale, and the OBJECTIVE's own recurrence
            // instead of a parameter step. A parameter-recurrence limb can
            // never certify here: the alternating solve slides along exactly
            // flat gauge orbits (e.g. a periodic atom's phase origin — rotate
            // its coordinates and counter-rotate its Fourier block and f is
            // unchanged), so parameters keep moving at zero gradient. Measured
            // on real activations: relative KKT 6.9e-5 with per-cycle
            // parameter moves of 1.4e-1.
            let objective_recurred = last_objective
                .map(|previous: f64| (objective - previous).abs() <= tolerance * kkt_scale)
                .unwrap_or(false);
            last_objective = Some(objective);
            let candidate =
                objective_recurred && stationarity.max_abs() <= tolerance * kkt_scale;
            log::info!(
                "support fixed-point cycle {iteration}: raw KKT max={:.3e} rel={:.3e} max_change={:.3e} objective={:.6e}",
                stationarity.max_abs(),
                stationarity.max_abs() / kkt_scale,
                max_change,
                objective
            );
            if candidate && previous_candidate {
                return Ok(SaeSupportFixedPointReport {
                    iterations: iteration,
                    objective,
                    stationarity,
                    max_recurrence_change: max_change,
                    recurred: true,
                });
            }
            previous_candidate = candidate;
        }
        let stationarity = self.raw_stationarity(target, lambda_smooth, ard_precisions)?;
        let objective = self.penalized_objective(target, lambda_smooth, ard_precisions)?;
        Err(format!(
            "SaeSupportSparseTerm::solve_fixed_point did not recur within {max_iter} cycles (raw KKT max={:.6e}, relative to objective {:.6e}: {:.6e}; last parameter max_change={last_max_change:.6e}, gauge-invariant limbs required)",
            stationarity.max_abs(),
            objective,
            stationarity.max_abs() / objective.abs().max(1.0)
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assignment_state::SaeAssignmentAtomSpec;
    use ndarray::array;
    use std::sync::Arc;

    fn atom(
        name: &str,
        kind: SaeAtomBasisKind,
        d: usize,
        evaluator: Arc<dyn SaeBasisSecondJet>,
        coords: &[f64],
        decoder: Array2<f64>,
    ) -> SaeManifoldAtom {
        let coord = Array2::from_shape_vec((1, d), coords.to_vec()).expect("coords");
        let (phi, jet) = evaluator.evaluate(coord.view()).expect("evaluate");
        let m = phi.ncols();
        SaeManifoldAtom::new_with_provided_function_gram(
            name,
            kind,
            d,
            phi,
            jet,
            decoder,
            Array2::eye(m),
        )
        .expect("atom")
        .with_basis_second_jet(evaluator)
    }

    #[test]
    fn direct_reconstruction_uses_only_heterogeneous_support() {
        let periodic_eval: Arc<dyn SaeBasisSecondJet> =
            Arc::new(PeriodicHarmonicEvaluator::new(3).expect("periodic"));
        let patch_eval: Arc<dyn SaeBasisSecondJet> =
            Arc::new(EuclideanPatchEvaluator::new(2, 1).expect("patch"));
        let atoms = vec![
            atom(
                "circle",
                SaeAtomBasisKind::Periodic,
                1,
                periodic_eval,
                &[0.0],
                array![[0.0], [1.0], [0.0]],
            ),
            atom(
                "plane",
                SaeAtomBasisKind::Linear,
                2,
                patch_eval,
                &[0.0, 0.0],
                array![[0.0], [2.0], [-1.0]],
            ),
        ];
        let specs = vec![
            SaeAssignmentAtomSpec {
                latent_dim: 1,
                id_mode: LatentIdMode::None,
                manifold: SaeAtomBasisKind::Periodic.latent_manifold(1),
                retraction: gam_problem::LatentRetractionRegistry::all_euclidean(),
                latent_id: 1,
            },
            SaeAssignmentAtomSpec::euclidean(2),
        ];
        let state = SaeAssignmentState::from_topk_support_heterogeneous(
            2,
            2,
            1,
            specs,
            vec![vec![0], vec![1]],
            vec![vec![9.0], vec![-4.0]],
            vec![vec![0.25], vec![3.0, 1.0]],
        )
        .expect("state");
        let term = SaeSupportSparseTerm::new(atoms, state).expect("term");
        let fitted = term.reconstruct().expect("reconstruct");
        assert!((fitted[[0, 0]] - 1.0).abs() < 1.0e-12);
        assert!((fitted[[1, 0]] - 5.0).abs() < 1.0e-12);
        assert_eq!(term.active_pair_count(), 2);
    }

    #[test]
    fn decoder_sweep_decreases_final_function_objective() {
        let evaluator: Arc<dyn SaeBasisSecondJet> =
            Arc::new(EuclideanPatchEvaluator::new(1, 1).expect("patch"));
        let atoms = vec![atom(
            "line",
            SaeAtomBasisKind::Linear,
            1,
            evaluator,
            &[0.0],
            Array2::zeros((2, 1)),
        )];
        let state = SaeAssignmentState::from_topk_support(
            3,
            1,
            1,
            1,
            vec![vec![0]; 3],
            vec![vec![1.0]; 3],
            vec![vec![-1.0], vec![0.0], vec![1.0]],
        )
        .expect("state");
        let mut term = SaeSupportSparseTerm::new(atoms, state).expect("term");
        let target = array![[-1.0], [0.0], [1.0]];
        let ard = vec![vec![1.0]];
        let before = term
            .penalized_objective(target.view(), &[0.1], &ard)
            .expect("before");
        term.decoder_sweep(target.view(), &[0.1]).expect("sweep");
        let after = term
            .penalized_objective(target.view(), &[0.1], &ard)
            .expect("after");
        assert!(after < before);
        assert!(
            term.raw_stationarity(target.view(), &[0.1], &ard)
                .expect("kkt")
                .decoder_max_abs
                < 1.0e-10
        );
    }
}
