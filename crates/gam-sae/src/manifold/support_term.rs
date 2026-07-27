//! Canonical support-sparse curved term and fixed-point inner solve.
//!
//! Hard-TopK gates are read-only binary support. Consequently a row's only
//! live local parameters are the heterogeneous coordinates
//! `concat_{k in S_i} t_ik`; no gate/logit coordinate exists. This term owns
//! that representation directly and evaluates basis values and analytic jets
//! only for active `(row, atom)` pairs.

use crate::assignment::AssignmentMode;
use crate::assignment_state::{SaeAssignmentAtomSpec, SaeAssignmentState};
use gam_linalg::anderson::AndersonAccelerator;
use gam_linalg::utils::KahanSum;
use ndarray::{Array1, Array2, ArrayView2};
use rayon::prelude::*;
use std::ops::Range;
use std::sync::Arc;

use super::*;

/// Rows per rayon task in the read-only active-set passes (#2575).
///
/// The unit of work is a row, but the unit of ALLOCATION should not be: each
/// task builds its evaluation scratch once and reuses it across the rows it
/// takes, so the chunk width sets how many rows amortise one scratch. Wide
/// enough that the per-task setup is negligible against the `support_k · M · P`
/// work per row, narrow enough that a 4-core host still gets even load at the
/// smallest shapes the lane admits.
const RECONSTRUCT_ROW_CHUNK: usize = 64;

/// Order of the Anderson multisecant model on the support fixed point (#2575).
///
/// This is a COST bound, not a tuning knob. The accelerator drops every
/// difference column whose contribution is below its own roundoff floor, so a
/// history longer than the map's informative secant subspace costs memory and
/// buys nothing rather than mispricing anything — which is why the depth can be
/// declared here instead of derived from the problem. What it bounds:
/// `2·depth·(N·support_k)` doubles of history, and an `order × order`
/// eigendecomposition per cycle, both negligible against one sweep's
/// `N·support_k·M·P` work.
///
/// Eight is the upper end of the range the literature reports gains over
/// (Walker & Ni, *SINUM* 2011, §4; Fang & Saad, *NLAA* 2009): past it, the
/// stored differences on a slowly-contracting map are numerically dependent and
/// the extra columns are exactly the ones the roundoff floor discards.
const SUPPORT_ANDERSON_DEPTH: usize = 8;

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

/// Reusable storage for ONE active `(row, slot)` evaluation (#2575).
///
/// Every read-only pass over the active set — reconstruct, the raw KKT
/// reductions, the penalized objective, the decoder sweep's normal equations,
/// the arrow assembly — needs the same four arrays for each `(row, slot)` pair:
/// the basis row `Φ(t)`, its jet, the decoded image `Φ·B`, and the coordinate
/// Jacobian `∂(Φ·B)/∂t`. The producer used to ALLOCATE all four (plus a coords
/// array and a `dot` result) on every call, and it is called `n·support_k` times
/// per sweep — 358,544 times per sweep at the #2502 flagship shape, which is
/// where the profiled 12.4% of self time in `malloc`/`free`/`memmove` went.
///
/// Held by the caller and reused across rows (per rayon worker, via
/// `map_init`), it is sized once for the first `(m, d, p)` it sees and only
/// resized when a later atom needs a different shape — so a homogeneous atom
/// portfolio allocates once per worker per sweep instead of once per pair.
#[derive(Debug, Default, Clone)]
struct ActiveAtomScratch {
    /// `(1, m)` — the evaluator's own buffer shape.
    phi: Array2<f64>,
    /// `(1, m, d)`.
    jet: ndarray::Array3<f64>,
    /// `(P,)`.
    decoded: Array1<f64>,
    /// Coordinate-major decoded jet, `(d, P)`.
    jacobian: Array2<f64>,
}

impl ActiveAtomScratch {
    /// Resize to hold one `(m, d)` atom's evaluation against a `p`-wide
    /// response. A no-op when the shapes already match, which is the common
    /// case: the shapes are a property of the atom, not of the row.
    fn fit(&mut self, m: usize, d: usize, p: usize) {
        if self.phi.dim() != (1, m) {
            self.phi = Array2::zeros((1, m));
        }
        if self.jet.dim() != (1, m, d) {
            self.jet = ndarray::Array3::zeros((1, m, d));
        }
        if self.decoded.len() != p {
            self.decoded = Array1::zeros(p);
        }
        if self.jacobian.dim() != (d, p) {
            self.jacobian = Array2::zeros((d, p));
        }
    }

    /// The basis row as a flat `m`-vector view — every consumer reads it as
    /// `phi[basis]`, and the evaluator writes it as `(1, m)`.
    fn phi_row(&self) -> ndarray::ArrayView1<'_, f64> {
        self.phi.row(0)
    }
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

/// Reusable storage for ONE row's coordinate solve (#2575).
///
/// Held per rayon worker and reused across every row that worker takes. The
/// row solve's working set is a function of the row's SUPPORT SHAPE — the
/// number of active slots, each slot's `(m, d)`, the compact coordinate width
/// `q` — and on this lane those are the same for almost every row (one support
/// width, one atom portfolio), so [`Self::fit`] resizes on the first row and is
/// a no-op thereafter.
#[derive(Debug, Default, Clone)]
struct RowSolveScratch {
    /// Per-slot offsets into the row's compact coordinate block.
    offsets: Vec<Range<usize>>,
    /// The row's support, in slot order.
    support: Vec<u32>,
    /// Per-slot `(basis width, latent dim)`.
    dims: Vec<(usize, usize)>,
    /// Per-slot evaluation at the CURRENT coordinates.
    current: Vec<ActiveAtomScratch>,
    /// Per-slot evaluation at the line search's trial coordinates.
    trial: Vec<ActiveAtomScratch>,
    fitted: Array1<f64>,
    /// `(q, P)` coordinate-major row Jacobian.
    jacobian: Array2<f64>,
    trial_fitted: Array1<f64>,
    trial_residual: Array1<f64>,
    trial_delta: Vec<f64>,
    fitted_delta: Vec<KahanSum>,
    old_coords: Vec<f64>,
}

impl RowSolveScratch {
    fn fit(&mut self, term: &SaeSupportSparseTerm, row: usize, q: usize, p: usize) {
        term.slot_offsets_into(row, &mut self.offsets);
        self.support.clear();
        self.support
            .extend_from_slice(term.assignment.support_indices(row));
        self.dims.clear();
        self.dims.extend(self.support.iter().map(|&atom| {
            let atom = atom as usize;
            (
                term.atoms[atom].basis_size(),
                term.atoms[atom].latent_dim(),
            )
        }));
        let slots = self.dims.len();
        self.current.resize_with(slots, ActiveAtomScratch::default);
        self.trial.resize_with(slots, ActiveAtomScratch::default);
        for (slot, &(m, d)) in self.dims.iter().enumerate() {
            self.current[slot].fit(m, d, p);
            self.trial[slot].fit(m, d, p);
        }
        if self.fitted.len() != p {
            self.fitted = Array1::zeros(p);
            self.trial_fitted = Array1::zeros(p);
            self.trial_residual = Array1::zeros(p);
            self.fitted_delta = vec![KahanSum::default(); p];
        }
        if self.jacobian.dim() != (q, p) {
            self.jacobian = Array2::zeros((q, p));
        }
        self.trial_delta.clear();
        self.trial_delta.resize(q, 0.0);
        self.old_coords.clear();
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
    /// Per-atom axis periodicity, resolved ONCE at construction (#2575).
    ///
    /// `SaeAssignmentState::atom_axis_periods` builds a fresh `Vec` on every
    /// call, and the ARD prior needs it at every `(row, slot, axis)` — including
    /// inside the coordinate line search, so up to 25 times per row per sweep.
    /// It is a property of the atom's declared manifold and retraction, both of
    /// which are fixed when the assignment state is built and are never
    /// mutated after, so resolving it per call was re-deriving a constant.
    atom_axis_periods: Vec<Vec<Option<f64>>>,
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
            // The kernels below subscript FOUR quantities per atom: the
            // decoder's basis rows, the decoder's output columns, the reference
            // Gram's width, and the coordinate block's latent width. This door
            // used to validate the last two only, so an atom whose decoder did
            // not span its own basis was ADMITTED and aborted later, inside a
            // rayon worker, as a bare `ndarray: index out of bounds` naming no
            // row and no atom (#2572). The atom states its own contract; check
            // it here, where the two shapes can still be named.
            template.validate_shape_contract().map_err(|error| {
                format!("SaeSupportSparseTerm::new: atom {atom}: {error}")
            })?;
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
        let atom_axis_periods = (0..k_atoms)
            .map(|atom| assignment.atom_axis_periods(atom))
            .collect();
        Ok(Self {
            atoms,
            assignment,
            output_dim,
            atom_rows,
            atom_axis_periods,
        })
    }

    /// Axis periodicity of one atom's coordinate block: `None` on a Euclidean
    /// axis, `Some(period)` on a circular one.
    fn atom_axis_periods(&self, atom: usize) -> &[Option<f64>] {
        &self.atom_axis_periods[atom]
    }

    /// Total width of the compact coordinate state `T` — the concatenation of
    /// every row's active coordinate block.
    pub fn coordinate_state_len(&self) -> usize {
        (0..self.n_obs())
            .map(|row| self.assignment.coords_row(row).len())
            .sum()
    }

    /// Copy `T` into caller storage, row-major over rows and slot-major within
    /// a row — the same order `install_coordinates` and
    /// `wrapped_coordinate_residual` read.
    fn snapshot_coordinates(&self, out: &mut Vec<f64>) {
        out.clear();
        for row in 0..self.n_obs() {
            out.extend_from_slice(self.assignment.coords_row(row));
        }
    }

    /// Apply one compact step to `T`, retracting each row onto its atoms'
    /// manifolds — the same retraction the coordinate sweep's line search uses,
    /// so an extrapolated step lands on the manifold by construction rather
    /// than by being projected back afterwards.
    fn retract_coordinates(&mut self, step: &[f64]) -> Result<(), String> {
        let mut coords_rows = self.assignment.take_coords();
        let mut cursor = 0usize;
        let mut outcome = Ok(());
        for (row, coords_row) in coords_rows.iter_mut().enumerate() {
            let end = cursor + coords_row.len();
            if end > step.len() {
                outcome = Err(format!(
                    "SaeSupportSparseTerm::retract_coordinates: step width {} is short of \
                     row {row}'s block end {end}",
                    step.len()
                ));
                break;
            }
            if let Err(error) = self
                .assignment
                .retract_row_coords(row, coords_row, &step[cursor..end])
            {
                outcome = Err(error);
                break;
            }
            cursor = end;
        }
        self.assignment.restore_coords(coords_rows)?;
        outcome?;
        if cursor != step.len() {
            return Err(format!(
                "SaeSupportSparseTerm::retract_coordinates: step width {} != compact \
                 coordinate width {cursor}",
                step.len()
            ));
        }
        Ok(())
    }

    /// Install a whole `T`, projecting each row onto its atoms' manifolds.
    /// Used to restore a rejected extrapolation, where the target state is an
    /// absolute snapshot rather than a step.
    fn install_coordinates(&mut self, values: &[f64]) -> Result<(), String> {
        let mut cursor = 0usize;
        for row in 0..self.n_obs() {
            let width = self.assignment.coords_row(row).len();
            let end = cursor + width;
            if end > values.len() {
                return Err(format!(
                    "SaeSupportSparseTerm::install_coordinates: state width {} is short of                      row {row}'s block end {end}",
                    values.len()
                ));
            }
            self.assignment.set_row_coords(row, &values[cursor..end])?;
            cursor = end;
        }
        if cursor != values.len() {
            return Err(format!(
                "SaeSupportSparseTerm::install_coordinates: state width {} != compact                  coordinate width {cursor}",
                values.len()
            ));
        }
        Ok(())
    }

    /// `after - before` on each coordinate axis, taken on the axis's own
    /// manifold.
    ///
    /// On a periodic axis the sweep's projection returns the image to a
    /// principal branch, so a literal difference across the branch cut reads as
    /// a whole period where the step was infinitesimal. Wrapping to the
    /// principal branch is what makes the residual the honest step — and what
    /// lets the accelerator treat `before + residual` as a lifted image whose
    /// differences are consistent across cycles.
    fn wrapped_coordinate_residual(&self, before: &[f64], after: &[f64], out: &mut Vec<f64>) {
        out.clear();
        let mut cursor = 0usize;
        for row in 0..self.n_obs() {
            for &atom in self.assignment.support_indices(row) {
                for &period in self.atom_axis_periods(atom as usize) {
                    let delta = after[cursor] - before[cursor];
                    out.push(match period {
                        Some(period) if period.is_finite() && period > 0.0 => {
                            delta - period * (delta / period).round()
                        }
                        _ => delta,
                    });
                    cursor += 1;
                }
            }
        }
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
        let zero_prior: Vec<Vec<f64>> = self
            .atoms
            .iter()
            .map(|atom| vec![0.0_f64; atom.latent_dim()])
            .collect();
        self.reroute_fixed_decoder_ard(target, support_k, random_state, &zero_prior)
    }

    /// [`Self::reroute_fixed_decoder`] scoring the coordinate prior too, so the
    /// greedy step and the caller's acceptance test agree on the objective.
    pub fn reroute_fixed_decoder_ard(
        &self,
        target: ArrayView2<'_, f64>,
        support_k: usize,
        random_state: u64,
        ard_precisions: &[Vec<f64>],
    ) -> Result<Self, String> {
        if ard_precisions.len() != self.k_atoms() {
            return Err(format!(
                "reroute_fixed_decoder_ard: ard_precisions length {} must equal K={}",
                ard_precisions.len(),
                self.k_atoms()
            ));
        }
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
        // Each row's routing reads only that row and the frozen decoders, so the
        // sweep is parallel by construction; an indexed `collect` restores row
        // order, making the result identical to the serial sweep it replaces.
        // This is the dominant cost of an out-of-sample reconstruct -- it scores
        // every atom against every row -- and it was leaving a 30-core box at
        // load 10.
        // ---- residual-greedy (OMP) routing --------------------------------
        // Marginal top-s is not the right selection rule for a K > P dictionary:
        // the atoms are necessarily coherent (Welch), so the s best-individually
        // atoms are near-duplicates and span far less than the s best jointly.
        // Greedy against the running residual fixes that; the chart argmax is
        // taken on a grid so the score is a property of the atom's image rather
        // than of its index.
        // One trial coordinate per basis coefficient. A basis carrying `m`
        // coefficients cannot resolve more than about `m` independent features
        // along its chart, so `m` is the basis's own resolution rather than a
        // tuning constant. Atoms may carry different widths, so slots are
        // addressed through a prefix offset instead of a uniform stride.
        //
        // Multi-axis atoms fall through to the marginal path below: a product
        // grid is exponential in the latent dimension, and the overcomplete
        // lane this serves admits 1-D charts.
        if self.atoms.iter().all(|atom| atom.latent_dim() == 1) {
            let k_atoms = self.k_atoms();
            let mut grid_offset = Vec::with_capacity(k_atoms + 1);
            let mut slot_atom = Vec::new();
            let mut slots = 0usize;
            for (atom_index, atom) in self.atoms.iter().enumerate() {
                grid_offset.push(slots);
                let width = atom.basis_size().max(2);
                slot_atom.extend(std::iter::repeat(atom_index).take(width));
                slots += width;
            }
            grid_offset.push(slots);
            let mut gamma = Array2::<f64>::zeros((slots, self.output_dim));
            let mut theta = vec![0.0_f64; slots];
            for (atom_index, atom) in self.atoms.iter().enumerate() {
                let evaluator = atom.basis_evaluator.as_ref().ok_or_else(|| {
                    format!("reroute omp: atom {atom_index} has no evaluator")
                })?;
                let width = grid_offset[atom_index + 1] - grid_offset[atom_index];
                for g in 0..width {
                    let raw = -1.0 + 2.0 * (g as f64 / width as f64);
                    let t = super::support_seed::chart_coordinate(atom.basis_kind(), 0, raw);
                    let coordinate = Array2::from_shape_vec((1, 1), vec![t])
                        .map_err(|error| format!("reroute grid: {error}"))?;
                    let (phi, _) = evaluator.evaluate(coordinate.view())?;
                    let decoded = phi.row(0).dot(atom.decoder_coefficients());
                    let slot = grid_offset[atom_index] + g;
                    for channel in 0..self.output_dim {
                        gamma[[slot, channel]] = decoded[channel];
                    }
                    theta[slot] = t;
                }
            }
            let self_term: Vec<f64> = (0..slots)
                .map(|slot| {
                    (0..self.output_dim).map(|c| gamma[[slot, c]] * gamma[[slot, c]]).sum::<f64>()
                })
                .collect();
            // `2 * V(alpha, t)` -- the prior the objective charges for placing a
            // row at this chart coordinate, on the same scale as `gain`.
            let prior_term: Vec<f64> = (0..slots)
                .map(|slot| {
                    let atom_index = slot_atom[slot];
                    let period = self.atom_axis_periods(atom_index)[0];
                    2.0 * ArdAxisPrior::eval(
                        ard_precisions[atom_index][0],
                        theta[slot],
                        period,
                    )
                    .value
                })
                .collect();

            let routed: Vec<(Vec<u32>, Vec<f64>, Vec<f64>)> = (0..target.nrows())
                .into_par_iter()
                .map(|row| {
                    let mut residual: Vec<f64> =
                        (0..self.output_dim).map(|c| target[[row, c]]).collect();
                    let mut taken = vec![false; k_atoms];
                    let mut picked: Vec<(usize, f64, f64)> = Vec::with_capacity(support_k);
                    for _ in 0..support_k {
                        let mut best_gain = f64::NEG_INFINITY;
                        let mut best_atom = usize::MAX;
                        let mut best_theta = 0.0;
                        let mut best_slot = 0usize;
                        for atom_index in 0..k_atoms {
                            if taken[atom_index] {
                                continue;
                            }
                            for slot in grid_offset[atom_index]..grid_offset[atom_index + 1] {
                                let mut cross = 0.0;
                                for c in 0..self.output_dim {
                                    cross += residual[c] * gamma[[slot, c]];
                                }
                                let gain = 2.0 * cross - self_term[slot] - prior_term[slot];
                                if gain > best_gain {
                                    best_gain = gain;
                                    best_atom = atom_index;
                                    best_theta = theta[slot];
                                    best_slot = slot;
                                }
                            }
                        }
                        if best_atom == usize::MAX {
                            break;
                        }
                        taken[best_atom] = true;
                        for c in 0..self.output_dim {
                            residual[c] -= gamma[[best_slot, c]];
                        }
                        picked.push((best_atom, best_gain, best_theta));
                    }
                    picked.sort_by_key(|entry| entry.0);
                    (
                        picked.iter().map(|e| e.0 as u32).collect::<Vec<u32>>(),
                        picked.iter().map(|e| e.1).collect::<Vec<f64>>(),
                        picked.iter().map(|e| e.2).collect::<Vec<f64>>(),
                    )
                })
                .collect();

            let mut indices = Vec::with_capacity(target.nrows());
            let mut gate_params = Vec::with_capacity(target.nrows());
            let mut coords = Vec::with_capacity(target.nrows());
            for (row_indices, row_gates, row_coords) in routed {
                indices.push(row_indices);
                gate_params.push(row_gates);
                coords.push(row_coords);
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
                k_atoms,
                support_k,
                atom_specs,
                indices,
                gate_params,
                coords,
            )?;
            return Self::new(self.atoms.clone(), assignment);
        }
        type RowRoute = (Vec<u32>, Vec<f64>, Vec<f64>);
        let per_row: Vec<RowRoute> = target
            .axis_iter(ndarray::Axis(0))
            .into_par_iter()
            .map(|row| -> Result<RowRoute, String> {
            let row_values = row.as_slice().ok_or_else(|| {
                "SaeSupportSparseTerm::reroute_fixed_decoder: target row is not contiguous"
                    .to_string()
            })?;
            let mut selected = Vec::<Candidate>::with_capacity(support_k);
            for (atom_index, atom) in self.atoms.iter().enumerate() {
                // The hashed coordinate is only a stand-in for "where on this atom's
                // curve does this row sit". Scoring an atom at an arbitrary point
                // makes selection near-uncorrelated with which atoms can actually
                // represent the row, so a 1-D atom searches its own curve at the
                // basis's resolution before being scored. Only a multi-axis atom,
                // whose product grid is exponential, still falls back to the hash.
                let route_grid = atom.basis_size().max(2);
                let candidate_coords = if atom.latent_dim() == 1 {
                    let periodic = matches!(
                        atom.basis_kind(),
                        super::SaeAtomBasisKind::Periodic
                    );
                    let mut best_t = 0.0_f64;
                    let mut best_s = f64::NEG_INFINITY;
                    for g in 0..route_grid {
                        let frac = g as f64 / route_grid as f64;
                        let t_try = if periodic { frac } else { -1.0 + 2.0 * frac };
                        let c_try = Array2::from_shape_vec((1, 1), vec![t_try])
                            .map_err(|error| format!("reroute grid: {error}"))?;
                        if let Some(ev) = atom.basis_evaluator.as_ref() {
                            let (phi_try, _) = ev.evaluate(c_try.view())?;
                            let dec = phi_try.row(0).dot(atom.decoder_coefficients());
                            let s_try: f64 = row
                                .iter()
                                .zip(dec.iter())
                                .map(|(truth, fit)| 2.0 * truth * fit - fit * fit)
                                .sum();
                            if s_try > best_s {
                                best_s = s_try;
                                best_t = t_try;
                            }
                        }
                    }
                    vec![best_t]
                } else {
                    (0..atom.latent_dim())
                        .map(|axis| {
                            let raw = super::support_seed::projection(
                                row_values,
                                atom_index,
                                axis + 1,
                                random_state,
                            );
                            super::support_seed::chart_coordinate(atom.basis_kind(), axis, raw)
                        })
                        .collect::<Vec<_>>()
                };
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
                let decoded = phi.row(0).dot(atom.decoder_coefficients());
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
            let row_indices: Vec<u32> =
                selected.iter().map(|candidate| candidate.atom as u32).collect();
            let row_gates: Vec<f64> =
                selected.iter().map(|candidate| candidate.score).collect();
            let row_coords: Vec<f64> = selected
                .into_iter()
                .flat_map(|candidate| candidate.coords)
                .collect();
            Ok((row_indices, row_gates, row_coords))
            })
            .collect::<Result<Vec<_>, String>>()?;
        let mut indices = Vec::with_capacity(target.nrows());
        let mut gate_params = Vec::with_capacity(target.nrows());
        let mut coords = Vec::with_capacity(target.nrows());
        for (row_indices, row_gates, row_coords) in per_row {
            indices.push(row_indices);
            gate_params.push(row_gates);
            coords.push(row_coords);
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
        // One evaluation scratch for the whole assembly (#2575); `blocks` still
        // owns a copy of each active basis row because the linearized operator
        // outlives this loop.
        let mut scratch = ActiveAtomScratch::default();
        for row in 0..self.n_obs() {
            let q = row_layout.row_q_active(row);
            let mut fitted = Array1::<f64>::zeros(self.output_dim);
            let mut jacobian = Array2::<f64>::zeros((q, self.output_dim));
            let mut blocks = Vec::with_capacity(self.assignment.support_indices(row).len());
            for slot in 0..self.assignment.support_indices(row).len() {
                let atom_idx = self.assignment.support_indices(row)[slot] as usize;
                self.fill_active(row, slot, &mut scratch)?;
                fitted += &scratch.decoded;
                let cursor = row_layout.coord_starts[row][slot];
                for axis in 0..scratch.jacobian.nrows() {
                    jacobian
                        .row_mut(cursor + axis)
                        .assign(&scratch.jacobian.row(axis));
                }
                let phi = scratch.phi_row();
                for basis in 0..phi.len() {
                    let base = beta_offsets[atom_idx] + basis * self.output_dim;
                    for channel in 0..self.output_dim {
                        hbb_diag[base + channel] += phi[basis] * phi[basis];
                    }
                }
                blocks.push(SupportBasisBlock {
                    beta_offset: beta_offsets[atom_idx],
                    phi: phi.to_owned(),
                });
            }
            let residual = &target.row(row) - &fitted;
            system.rows[row].htt.assign(&jacobian.dot(&jacobian.t()));
            system.rows[row].gt.assign(&(-jacobian.dot(&residual)));
            let periods = self
                .assignment
                .support_indices(row)
                .iter()
                .flat_map(|&atom| self.atom_axis_periods(atom as usize).iter().copied())
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
                .dot(self.atoms[atom].decoder_coefficients());
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

    /// Evaluate one active `(row, slot)` pair into caller-owned storage.
    ///
    /// The allocating counterpart this replaces (`evaluate_active`) built six
    /// fresh arrays per call and was called once per active pair per pass —
    /// `n·support_k` times per sweep (#2575). The evaluation itself is
    /// unchanged: it delegates to [`Self::fill_active_eval`], which is the one
    /// place that reads the evaluator and folds the decoder, so the row solve
    /// and every read-only pass now share a single producer.
    fn fill_active(
        &self,
        row: usize,
        slot: usize,
        scratch: &mut ActiveAtomScratch,
    ) -> Result<(), String> {
        let atom_idx = self.assignment.support_indices(row)[slot] as usize;
        let atom = &self.atoms[atom_idx];
        scratch.fit(atom.basis_size(), atom.latent_dim(), self.output_dim);
        let ActiveAtomScratch {
            phi,
            jet,
            decoded,
            jacobian,
        } = scratch;
        self.fill_active_eval(
            row,
            slot,
            self.assignment.coords_for_slot(row, slot),
            phi,
            jet,
            decoded,
            jacobian,
        )
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
        Ok(phi.dot(atom.decoder_coefficients()))
    }

    fn reconstruct_row_into(
        &self,
        row: usize,
        scratch: &mut ActiveAtomScratch,
        fitted: &mut Array1<f64>,
    ) -> Result<(), String> {
        fitted.fill(0.0);
        for slot in 0..self.assignment.support_indices(row).len() {
            self.fill_active(row, slot, scratch)?;
            *fitted += &scratch.decoded;
        }
        Ok(())
    }

    /// Direct active-row reconstruction. No K-wide gate or basis row exists.
    /// Rows are independent reads of shared state, so they decode in parallel.
    ///
    /// #2575: the per-row decode used to allocate a fresh `(P,)` row and six
    /// arrays per active pair, and the whole `(N, P)` result was collected as a
    /// `Vec` of owned rows before being copied into the output. Each rayon
    /// worker now carries ONE scratch and ONE row accumulator across all the
    /// rows it takes, and writes into its own disjoint slice of the output.
    pub fn reconstruct(&self) -> Result<Array2<f64>, String> {
        let mut fitted = Array2::<f64>::zeros((self.n_obs(), self.output_dim));
        let output_dim = self.output_dim;
        fitted
            .axis_chunks_iter_mut(ndarray::Axis(0), RECONSTRUCT_ROW_CHUNK)
            .into_par_iter()
            .enumerate()
            .try_for_each(|(chunk, mut block)| -> Result<(), String> {
                let mut scratch = ActiveAtomScratch::default();
                let mut row_fitted = Array1::<f64>::zeros(output_dim);
                let base = chunk * RECONSTRUCT_ROW_CHUNK;
                for local in 0..block.nrows() {
                    self.reconstruct_row_into(base + local, &mut scratch, &mut row_fitted)?;
                    block.row_mut(local).assign(&row_fitted);
                }
                Ok(())
            })?;
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
            let sb = atom.smooth_penalty().dot(atom.decoder_coefficients());
            value += 0.5
                * lambda
                * atom
                    .decoder_coefficients()
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
                    let periods = self.atom_axis_periods(atom);
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
        // Parallel width of this sweep is the SIZE of a colour class, not the atom
        // count: atoms in one class are row-disjoint and solved together, but the
        // classes run in sequence. With top_k = s every row forces its s atoms into
        // s distinct classes, so a dense conflict graph collapses the width.
        if !classes.is_empty() {
            let widest = classes.iter().map(|c| c.len()).max().unwrap_or(0);
            let narrowest = classes.iter().map(|c| c.len()).min().unwrap_or(0);
            let mean = self.k_atoms() as f64 / classes.len() as f64;
            log::info!(
                "decoder sweep colouring: {} classes over {} atoms (widest {}, narrowest {}, mean {:.1} atoms/class)",
                classes.len(), self.k_atoms(), widest, narrowest, mean
            );
        }
        for class in &classes {
            // Atoms in one class are row-disjoint: solve in parallel against
            // the shared `fitted` snapshot (each atom reads only its own rows),
            // then apply the disjoint updates.
            let solved: Vec<(usize, Array2<f64>, Array2<f64>, f64)> = class
                .par_iter()
                .map(|&atom_idx| -> Result<_, String> {
                    let m = self.atoms[atom_idx].basis_size();
                    let old_decoder = self.atoms[atom_idx].decoder_coefficients();
                    let mut gram =
                        self.atoms[atom_idx].smooth_penalty().clone() * lambda_smooth[atom_idx];
                    let mut rhs = Array2::<f64>::zeros((m, self.output_dim));
                    // #2575: the atom's basis rows and decoded images used to be
                    // two fresh `Array1`s PER ROW on the atom's support, plus
                    // six more inside the allocating evaluator, and a third per
                    // row for the delta. They are one `(rows, m)` and one
                    // `(rows, P)` block now — which also turns the decoded
                    // refresh below into a single GEMM instead of a GEMV per row.
                    let atom_rows = &self.atom_rows[atom_idx];
                    let row_count = atom_rows.len();
                    let mut phi_rows = Array2::<f64>::zeros((row_count, m));
                    let mut decoded_rows = Array2::<f64>::zeros((row_count, self.output_dim));
                    let mut scratch = ActiveAtomScratch::default();
                    for (index, &(row, slot)) in atom_rows.iter().enumerate() {
                        self.fill_active(row, slot, &mut scratch)?;
                        let phi = scratch.phi_row();
                        for left in 0..m {
                            for right in 0..m {
                                gram[[left, right]] += phi[left] * phi[right];
                            }
                            for output in 0..self.output_dim {
                                let residual_without = target[[row, output]]
                                    - fitted[[row, output]]
                                    + scratch.decoded[output];
                                rhs[[left, output]] += phi[left] * residual_without;
                            }
                        }
                        phi_rows.row_mut(index).assign(&phi);
                        decoded_rows.row_mut(index).assign(&scratch.decoded);
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
                    let mut deltas = phi_rows.dot(&decoder);
                    deltas -= &decoded_rows;
                    Ok((atom_idx, decoder, deltas, atom_change))
                })
                .collect::<Result<Vec<_>, String>>()?;
            for (atom_idx, decoder, deltas, atom_change) in solved {
                max_change = max_change.max(atom_change);
                self.atoms[atom_idx].set_decoder_coefficients(decoder)?;
                for (index, &(row, _slot)) in self.atom_rows[atom_idx].iter().enumerate() {
                    for output in 0..self.output_dim {
                        fitted[[row, output]] += deltas[[index, output]];
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
        // #2575: one scratch per rayon worker, not one per row. The row solve's
        // working set is ~18 allocations sized by the row's support shape, which
        // is identical for almost every row on this lane, so a worker allocates
        // once and reuses across every row it takes.
        let row_results: Vec<Result<f64, String>> = coords_rows
            .par_iter_mut()
            .enumerate()
            .map_init(RowSolveScratch::default, |scratch, (row, coords_row)| {
                self.row_coordinate_solve(
                    row,
                    coords_row,
                    scratch,
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
    fn slot_offsets_into(&self, row: usize, out: &mut Vec<Range<usize>>) {
        out.clear();
        let mut cursor = 0usize;
        for &atom in self.assignment.support_indices(row) {
            let d = self.assignment.atom_coord_dim(atom as usize);
            out.push(cursor..cursor + d);
            cursor += d;
        }
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
                decoded[output] += weight * atom.decoder_coefficients()[[basis, output]];
            }
        }
        jacobian.fill(0.0);
        for axis in 0..d {
            for basis in 0..m {
                let weight = jet[[0, basis, axis]];
                for output in 0..self.output_dim {
                    jacobian[[axis, output]] +=
                        weight * atom.decoder_coefficients()[[basis, output]];
                }
            }
        }
        Ok(())
    }

    /// One row's exact Gauss-Newton coordinate step with manifold-aware
    /// backtracking, on the row's caller-held coordinate block. Semantically
    /// the serial sweep's row iteration.
    ///
    /// Storage-wise: nothing here allocates. The scratch is the CALLER's, held
    /// per rayon worker and reused across every row that worker takes (#2575).
    /// It used to be per-row — eighteen allocations per row, `N` rows per
    /// sweep, hundreds of sweeps per fit — and the doc comment's claim that
    /// "the line-search halvings allocate nothing" was true within a row and
    /// misleading across them; the profiled 12.4% of self time in
    /// `malloc`/`free`/`memmove` is what that cost.
    fn row_coordinate_solve(
        &self,
        row: usize,
        coords_row: &mut Vec<f64>,
        scratch: &mut RowSolveScratch,
        target: ArrayView2<'_, f64>,
        ard_precisions: &[Vec<f64>],
        trust_radius: f64,
        stationarity_tolerance: f64,
    ) -> Result<f64, String> {
        let mut max_change = 0.0_f64;
        let q = coords_row.len();
        let p = self.output_dim;
        scratch.fit(self, row, q, p);
        let RowSolveScratch {
            offsets,
            support,
            dims,
            current,
            trial,
            fitted,
            jacobian,
            trial_fitted,
            trial_residual,
            trial_delta,
            fitted_delta,
            old_coords,
        } = scratch;
        let n_slots = offsets.len();
        fitted.fill(0.0);
        jacobian.fill(0.0);

        for slot in 0..n_slots {
            let slot_scratch = &mut current[slot];
            self.fill_active_eval(
                row,
                slot,
                &coords_row[offsets[slot].clone()],
                &mut slot_scratch.phi,
                &mut slot_scratch.jet,
                &mut slot_scratch.decoded,
                &mut slot_scratch.jacobian,
            )?;
            *fitted += &slot_scratch.decoded;
            for axis in 0..dims[slot].1 {
                jacobian
                    .row_mut(offsets[slot].start + axis)
                    .assign(&slot_scratch.jacobian.row(axis));
            }
        }
        let residual = &target.row(row) - &*fitted;
        let mut row_objective_scale =
            1.0 + 0.5 * residual.iter().map(|value| value * value).sum::<f64>();
        let mut rhs_vector = jacobian.dot(&residual);
        let mut gram = jacobian.dot(&jacobian.t());
        let mut prior_cursor = 0usize;
        for (slot, &atom) in support.iter().enumerate() {
            let atom = atom as usize;
            let periods = self.atom_axis_periods(atom);
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
        // SPEC-22: the exact PSD trust-region subproblem is general outer
        // optimizer machinery and lives in `opt`. gam kept a private copy
        // until #2574.
        let delta = opt::solve_psd_trust_region(gram.view(), rhs_vector.view(), trust_radius)
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
        old_coords.clear();
        old_coords.extend_from_slice(coords_row);
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
            self.assignment.project_row_coords(row, old_coords, coords_row)?;
            let step = 2.0_f64.powi(-(halving as i32));
            for (target_slot, value) in trial_delta.iter_mut().zip(delta.iter()) {
                *target_slot = step * value;
            }
            self.assignment.retract_row_coords(row, coords_row, trial_delta)?;
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
                let slot_trial = &mut trial[slot];
                self.fill_active_eval(
                    row,
                    slot,
                    &coords_row[offsets[slot].clone()],
                    &mut slot_trial.phi,
                    &mut slot_trial.jet,
                    &mut slot_trial.decoded,
                    &mut slot_trial.jacobian,
                )?;
                for basis in 0..dims[slot].0 {
                    // Subtract basis values before multiplying by decoder
                    // coefficients. This cancels shared constant/intercept
                    // components before rounding, instead of subtracting two
                    // already-decoded O(1) predictions to recover an O(step)
                    // difference.
                    let phi_delta = trial[slot].phi[[0, basis]] - current[slot].phi[[0, basis]];
                    for output in 0..p {
                        fitted_delta[output].add(
                            phi_delta * self.atoms[atom].decoder_coefficients()[[basis, output]],
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
                let periods = self.atom_axis_periods(atom);
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
            for slot_trial in trial.iter() {
                *trial_fitted += &slot_trial.decoded;
            }
            trial_residual.assign(&target.row(row));
            *trial_residual -= &*trial_fitted;
            let mut trial_gradient_max = 0.0_f64;
            for (slot, &atom) in support.iter().enumerate() {
                let atom = atom as usize;
                let periods = self.atom_axis_periods(atom);
                for axis in 0..dims[slot].1 {
                    let likelihood_gradient =
                        -trial[slot].jacobian.row(axis).dot(&*trial_residual);
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
                self.assignment.project_row_coords(row, old_coords, coords_row)?;
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
            .map_init(ActiveAtomScratch::default, |scratch, atom_idx| -> Result<(f64, f64), String> {
                let atom = &self.atoms[atom_idx];
                let mut gradient = atom.smooth_penalty().dot(atom.decoder_coefficients())
                    * lambda_smooth[atom_idx];
                for &(row, slot) in &self.atom_rows[atom_idx] {
                    self.fill_active(row, slot, scratch)?;
                    let phi = scratch.phi_row();
                    for basis in 0..atom.basis_size() {
                        for output in 0..self.output_dim {
                            gradient[[basis, output]] -= phi[basis] * residual[[row, output]];
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
            .map_init(ActiveAtomScratch::default, |scratch, row| -> Result<(f64, f64), String> {
                let mut sq = 0.0_f64;
                let mut max = 0.0_f64;
                for slot in 0..self.assignment.support_indices(row).len() {
                    let atom = self.assignment.support_indices(row)[slot] as usize;
                    self.fill_active(row, slot, scratch)?;
                    let periods = self.atom_axis_periods(atom);
                    for axis in 0..scratch.jacobian.nrows() {
                        let mut gradient = 0.0;
                        for output in 0..self.output_dim {
                            gradient -= scratch.jacobian[[axis, output]] * residual[[row, output]];
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
        let mut scratch = ActiveAtomScratch::default();
        for row in 0..self.n_obs() {
            for slot in 0..self.assignment.support_indices(row).len() {
                let atom = self.assignment.support_indices(row)[slot] as usize;
                self.fill_active(row, slot, &mut scratch)?;
                let periods = self.atom_axis_periods(atom);
                for axis in 0..scratch.jacobian.nrows() {
                    let likelihood_gradient = scratch
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
                let periods = self.atom_axis_periods(atom);
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
        // #2575: the alternating map's contraction is linear at ρ ≈ 0.975 on
        // real activations, so most cycles are spent crawling the tail rather
        // than resolving a nonlinearity. Anderson extrapolates over the
        // COORDINATE block alone, which is the whole state of the map: the
        // decoder sweep is the EXACT block minimiser of `B` given `T`, so the
        // fixed point is `T ↦ C(D(T))` and carrying `B` in the history would
        // store `Σ_k M_k·P` redundant numbers per column.
        let mut accelerator = AndersonAccelerator::new(SUPPORT_ANDERSON_DEPTH)
            .map_err(|error| format!("SaeSupportSparseTerm::solve_fixed_point: {error}"))?;
        let mut cycle_start = Vec::with_capacity(self.coordinate_state_len());
        let mut cycle_end = Vec::with_capacity(self.coordinate_state_len());
        let mut cycle_residual = Vec::with_capacity(self.coordinate_state_len());
        // `x_k − x_{k-1}` in the accelerator's difference-only contract; zero
        // before the first cycle, where it is ignored.
        let mut taken_step = vec![0.0_f64; self.coordinate_state_len()];
        let mut accepted_extrapolations = 0usize;
        for iteration in 1..=max_iter {
            self.snapshot_coordinates(&mut cycle_start);
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
            if candidate && previous_candidate {
                // The alternating sweeps hold the SUPPORT fixed, so a point that
                // is stationary in the coordinates and the decoders can still be
                // improved by re-routing rows onto atoms that now explain them
                // better -- a TopK SAE re-selects its latents on every forward
                // pass, and this loop never did. Proposing the move HERE, at the
                // inner fixed point, is the dictionary-learning alternation the
                // scheme was missing, and it needs no cadence constant because
                // convergence is itself the trigger.
                //
                // The move is guarded on the certificate's own objective. A
                // re-route changes the objective discontinuously, so accepting
                // it unconditionally would destroy the monotonicity the
                // certificate rests on; accepting only a strict decrease keeps
                // the scheme monotone and makes the returned point locally
                // optimal against a support move as well as stationary within
                // one, which is strictly stronger than certifying a frozen
                // support.
                let support_k = match self.assignment.mode() {
                    AssignmentMode::TopK { k } => k,
                    _ => 0,
                };
                if support_k > 0 {
                    let moved =
                        self.reroute_fixed_decoder_ard(target, support_k, 0, ard_precisions)?;
                    let after =
                        moved.penalized_objective(target, lambda_smooth, ard_precisions)?;
                    if after < objective {
                        log::info!(
                            "support move accepted at cycle {iteration}: objective \
                             {objective:.6e} -> {after:.6e}"
                        );
                        *self = moved;
                        // The map itself changed, so every difference the
                        // accelerator holds describes a map that no longer
                        // exists, and the two-cycle recurrence has to be
                        // re-established against the new support.
                        accelerator.reset();
                        taken_step.clear();
                        taken_step.resize(self.coordinate_state_len(), 0.0);
                        last_objective = None;
                        previous_candidate = false;
                        continue;
                    }
                    log::info!(
                        "support move rejected at cycle {iteration}: objective \
                         {objective:.6e} -> {after:.6e}"
                    );
                }
                log::info!(
                    "support fixed-point cycle {iteration}: raw KKT max={:.3e} rel={:.3e} \
                     max_change={:.3e} objective={:.6e} anderson_accepted={accepted_extrapolations}",
                    stationarity.max_abs(),
                    stationarity.max_abs() / kkt_scale,
                    max_change,
                    objective
                );
                return Ok(SaeSupportFixedPointReport {
                    iterations: iteration,
                    objective,
                    stationarity,
                    max_recurrence_change: max_change,
                    recurred: true,
                });
            }
            previous_candidate = candidate;

            // The certified point is ALWAYS a plain post-sweep iterate: the
            // certificate above has already been evaluated and either returned
            // or not, and what follows only chooses where the NEXT cycle starts.
            //
            // Anderson has no descent guarantee, so the proposal is safeguarded
            // on the objective the certificate itself uses, at the SAME decoder
            // this cycle solved — a like-for-like comparison, and a conservative
            // one, because the next cycle's exact decoder solve can only lower
            // it further. On rejection the plain iterate is restored and the
            // history is dropped: differences taken across a rejected candidate
            // would fit a secant model to a trajectory that never happened.
            self.snapshot_coordinates(&mut cycle_end);
            self.wrapped_coordinate_residual(&cycle_start, &cycle_end, &mut cycle_residual);
            let proposal = accelerator
                .propose(&cycle_residual, &taken_step)
                .map_err(|error| format!("SaeSupportSparseTerm::solve_fixed_point: {error}"))?;
            // The step that reached the NEXT cycle's iterate, whichever arm is
            // taken. The accelerator only ever sees differences, so this is the
            // one piece of state the caller owes it.
            taken_step.clear();
            match proposal {
                None => taken_step.extend_from_slice(&cycle_residual),
                Some(proposal) => {
                    // The extrapolated step is applied through the SAME
                    // retraction the line search uses, from the cycle's own
                    // starting iterate — so `x_start + step` is on the manifold
                    // by construction, and the step the accelerator is told
                    // about is exactly the one that was taken.
                    self.install_coordinates(&cycle_start)?;
                    self.retract_coordinates(&proposal)?;
                    let extrapolated =
                        self.penalized_objective(target, lambda_smooth, ard_precisions)?;
                    if extrapolated < objective {
                        accepted_extrapolations += 1;
                        taken_step.extend_from_slice(&proposal);
                    } else {
                        self.install_coordinates(&cycle_end)?;
                        accelerator.reset();
                        taken_step.extend_from_slice(&cycle_residual);
                    }
                }
            }
            log::info!(
                "support fixed-point cycle {iteration}: raw KKT max={:.3e} rel={:.3e} \
                 max_change={:.3e} objective={:.6e} anderson={}/{} order={}",
                stationarity.max_abs(),
                stationarity.max_abs() / kkt_scale,
                max_change,
                objective,
                accepted_extrapolations,
                iteration,
                accelerator.history_len()
            );
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

    /// One-atom, one-row support term whose decoder can be tampered with.
    fn single_linear_atom_term() -> (SaeSupportSparseTerm, Array2<f64>) {
        let evaluator: Arc<dyn SaeBasisSecondJet> =
            Arc::new(EuclideanPatchEvaluator::new(1, 1).expect("patch"));
        let atoms = vec![atom(
            "line",
            SaeAtomBasisKind::Linear,
            1,
            evaluator,
            &[0.0],
            Array2::zeros((2, 2)),
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
        let term = SaeSupportSparseTerm::new(atoms, state).expect("term");
        let target = array![[-1.0, 0.5], [0.0, 0.0], [1.0, -0.5]];
        (term, target)
    }

    /// #2572 — a decoder that does not span its own basis is not indexable, and
    /// the lane must say so instead of aborting a rayon worker.
    ///
    /// Measured before the fix, with `decoder_coefficients` a `pub` field: the
    /// row-short atom was ACCEPTED by `SaeSupportSparseTerm::new` (which
    /// validated `output_dim` and `latent_dim` but not the basis coupling) and
    /// then aborted in `reconstruct`, `raw_stationarity`, `solve_fixed_point`
    /// and `assemble_arrow_schur`; the column-short atom aborted the same four
    /// with the reported `ndarray: index out of bounds`. Both are now typed
    /// refusals AT THE MUTATION, so no kernel can be reached with either.
    #[test]
    fn a_decoder_that_cannot_be_indexed_is_refused_not_aborted() {
        let (term, _) = single_linear_atom_term();
        let full = term.atoms[0].decoder_coefficients().clone();
        assert_eq!(full.dim(), (2, 2));

        for (label, broken) in [
            ("one row short", full.slice(s![..1, ..]).to_owned()),
            ("one column short", full.slice(s![.., ..1]).to_owned()),
            ("one row too many", Array2::<f64>::zeros((3, 2))),
        ] {
            let mut atom = term.atoms[0].clone();
            let error = atom
                .set_decoder_coefficients(broken.clone())
                .expect_err(label);
            assert!(
                error.contains("set_decoder_coefficients") && error.contains("(2, 2)"),
                "{label}: {error}"
            );
            // The refusal is total: the atom keeps the decoder it had, so a
            // caller that ignores the error still cannot reach a kernel with an
            // unindexable atom.
            assert_eq!(atom.decoder_coefficients(), &full, "{label}");
        }
    }

    /// #2572 — the lane's door states the WHOLE contract its kernels subscript
    /// under, not the half it used to.
    ///
    /// `basis_values` stays a public field (its column count IS
    /// [`SaeManifoldAtom::basis_size`], so it cannot disagree with itself), which
    /// leaves exactly one way to break the coupling from outside: narrow the
    /// basis and leave the decoder wide. Before the fix this term ACCEPTED such
    /// an atom and every kernel that touched it aborted; now the door refuses it
    /// with the atom index and both shapes.
    #[test]
    fn the_support_term_door_refuses_an_atom_whose_decoder_misses_its_basis() {
        let (term, target) = single_linear_atom_term();
        let mut atoms = term.atoms.clone();
        atoms[0].basis_values = atoms[0].basis_values.slice(s![.., ..1]).to_owned();
        atoms[0].basis_jacobian = atoms[0].basis_jacobian.slice(s![.., ..1, ..]).to_owned();

        let error = SaeSupportSparseTerm::new(atoms, term.assignment.clone())
            .expect_err("an atom whose decoder overruns its basis is not indexable");
        assert!(error.contains("atom 0"), "{error}");
        assert!(error.contains("basis width is 1"), "{error}");

        // The untampered term is untouched by the new check.
        let ard = vec![vec![1.0]];
        term.raw_stationarity(target.view(), &[0.1], &ard)
            .expect("a well-formed term still evaluates");
    }

    /// #2572 — the assembled support-sparse system must be usable by the PCG
    /// preconditioner ladder, whose escalated tiers all build the β-coupling
    /// graph first.
    ///
    /// This lane assembles with `htbeta_cols = 0` and carries `H_tβ` in a matvec
    /// pair, so every row's dense cross block is `(d_i, 0)`; it also registers
    /// per-atom `block_offsets`. `BetaCouplingGraph::build` read that slab
    /// directly and aborted with `ndarray: index out of bounds` on the first
    /// subscript. Measured on the seeded `K = 24 > P = 8`, `top_k = 4` term in
    /// `examples/issue_2572_precond_probe.rs`: `ClusterJacobi` and
    /// `AdditiveSchwarz{overlap: 1}` both aborted; both now build.
    #[test]
    fn the_assembled_system_can_build_the_escalated_preconditioner_tiers() {
        use gam_solve::arrow_schur::{
            AdditiveSchwarzPreconditioner, BatchedBlockSolver, ClusterJacobiPreconditioner,
            CpuBatchedBlockSolver,
        };

        // Two linear atoms on disjoint rows, so the coupling graph has real
        // block structure to partition and both atoms carry a PD `H_tt`.
        let evaluator: Arc<dyn SaeBasisSecondJet> =
            Arc::new(EuclideanPatchEvaluator::new(1, 1).expect("patch"));
        let atoms = vec![
            atom(
                "left",
                SaeAtomBasisKind::Linear,
                1,
                Arc::clone(&evaluator),
                &[0.0],
                array![[0.5, -0.25], [1.0, 0.75]],
            ),
            atom(
                "right",
                SaeAtomBasisKind::Linear,
                1,
                evaluator,
                &[0.0],
                array![[-0.5, 0.25], [0.75, -1.0]],
            ),
        ];
        let state = SaeAssignmentState::from_topk_support(
            4,
            2,
            1,
            1,
            vec![vec![0], vec![1], vec![0], vec![1]],
            vec![vec![1.0]; 4],
            vec![vec![-1.0], vec![0.5], vec![1.0], vec![-0.5]],
        )
        .expect("state");
        let term = SaeSupportSparseTerm::new(atoms, state).expect("term");
        let target = array![[-1.0, 0.5], [0.25, 0.0], [1.0, -0.5], [-0.25, 0.75]];
        let ard = vec![vec![1.0], vec![1.0]];
        let lambda = vec![0.1, 0.1];
        let system = term
            .assemble_arrow_schur(target.view(), &lambda, &ard)
            .expect("arrow system");
        // The shape that used to abort: a zero-column cross-block slab with
        // registered block offsets.
        assert_eq!(system.rows[0].htbeta.dim().1, 0);
        assert!(!system.block_offsets.is_empty());

        let backend = CpuBatchedBlockSolver;
        let htt = backend
            .factor_blocks(&system.rows, 0.0, system.d, true)
            .expect("per-row blocks factor");
        ClusterJacobiPreconditioner::from_arrow_schur(&system, &htt, 0.0, &backend)
            .expect("cluster-Jacobi tier builds on a matrix-free system");
        AdditiveSchwarzPreconditioner::from_arrow_schur(&system, &htt, 0.0, &backend, 1)
            .expect("additive-Schwarz tier builds on a matrix-free system");
    }

    /// Two atoms active on EVERY row, so the alternating map is genuinely
    /// coupled: each atom's exact decoder block is solved against a residual the
    /// other atom is about to move, which is the structure that produces a
    /// linear rate near one (#2575).
    fn coupled_two_atom_fixture() -> (SaeSupportSparseTerm, Array2<f64>) {
        let evaluator: Arc<dyn SaeBasisSecondJet> =
            Arc::new(EuclideanPatchEvaluator::new(1, 1).expect("patch"));
        let rows = 24usize;
        let atoms = vec![
            atom(
                "left",
                SaeAtomBasisKind::Linear,
                1,
                Arc::clone(&evaluator),
                &[0.0],
                array![[0.10, -0.05], [0.90, 0.20]],
            ),
            atom(
                "right",
                SaeAtomBasisKind::Linear,
                1,
                evaluator,
                &[0.0],
                array![[-0.05, 0.10], [0.20, 0.85]],
            ),
        ];
        let state = SaeAssignmentState::from_topk_support(
            rows,
            2,
            2,
            2,
            vec![vec![0, 1]; rows],
            vec![vec![1.0, 1.0]; rows],
            (0..rows)
                .map(|row| {
                    let t = row as f64 / rows as f64;
                    vec![t - 0.5, 0.5 - t]
                })
                .collect(),
        )
        .expect("state");
        let term = SaeSupportSparseTerm::new(atoms, state).expect("term");
        let mut target = Array2::<f64>::zeros((rows, 2));
        for row in 0..rows {
            let t = row as f64 / rows as f64;
            // Not in the dictionary's span: a curved response two straight
            // decoders must trade against each other to fit.
            target[[row, 0]] = (3.0 * t).sin() + 0.20 * t;
            target[[row, 1]] = (2.0 * t).cos() - 0.15 * t * t;
        }
        (term, target)
    }

    /// Drive the PLAIN alternating map with the same certificate
    /// `solve_fixed_point` applies, and report the cycle it recurs on.
    ///
    /// The certificate is restated here on purpose: production no longer runs
    /// the un-accelerated map, and an A/B needs both arms measured under one
    /// standard.
    fn plain_cycles_to_recur(
        term: &mut SaeSupportSparseTerm,
        target: ArrayView2<'_, f64>,
        lambda_smooth: &[f64],
        ard_precisions: &[Vec<f64>],
        max_iter: usize,
        tolerance: f64,
        trust_radius: f64,
    ) -> Option<usize> {
        let mut previous_candidate = false;
        let mut last_objective: Option<f64> = None;
        for iteration in 1..=max_iter {
            term.decoder_sweep(target, lambda_smooth).expect("decoder");
            term.coordinate_sweep(target, ard_precisions, trust_radius, tolerance)
                .expect("coordinates");
            let residual = term.raw_residual(target).expect("residual");
            let stationarity = term
                .raw_stationarity_with_residual(&residual, lambda_smooth, ard_precisions)
                .expect("kkt");
            let objective = term
                .penalized_objective_with_residual(&residual, lambda_smooth, ard_precisions)
                .expect("objective");
            let scale = objective.abs().max(1.0);
            let recurred = last_objective
                .map(|previous: f64| (objective - previous).abs() <= tolerance * scale)
                .unwrap_or(false);
            last_objective = Some(objective);
            let candidate = recurred && stationarity.max_abs() <= tolerance * scale;
            if candidate && previous_candidate {
                return Some(iteration);
            }
            previous_candidate = candidate;
        }
        None
    }

    /// #2575 — acceleration must reach the certificate in no more cycles than
    /// the plain map, and the point it certifies must genuinely satisfy the
    /// certificate.
    ///
    /// The second half is the one that matters: Anderson has no descent
    /// guarantee, so an unsafeguarded extrapolation can land on a point that
    /// merely LOOKS recurred because two consecutive objectives happen to agree.
    /// The safeguard is what forbids that, and this re-checks the returned state
    /// against the same bar from scratch.
    #[test]
    fn acceleration_certifies_the_same_point_in_no_more_cycles() {
        let tolerance = 1.0e-6;
        let trust_radius = 1.0;
        let lambda = vec![1.0e-3, 1.0e-3];
        let ard = vec![vec![1.0e-4], vec![1.0e-4]];
        let budget = 4_000usize;

        let (mut plain, target) = coupled_two_atom_fixture();
        let plain_cycles = plain_cycles_to_recur(
            &mut plain,
            target.view(),
            &lambda,
            &ard,
            budget,
            tolerance,
            trust_radius,
        );

        let (mut accelerated, target) = coupled_two_atom_fixture();
        let report = accelerated
            .solve_fixed_point(
                target.view(),
                &lambda,
                &ard,
                budget,
                tolerance,
                trust_radius,
            )
            .expect("the accelerated fixed point recurs");
        assert!(report.recurred);

        // The certificate, re-derived at the returned state.
        let stationarity = accelerated
            .raw_stationarity(target.view(), &lambda, &ard)
            .expect("kkt");
        let objective = accelerated
            .penalized_objective(target.view(), &lambda, &ard)
            .expect("objective");
        let scale = objective.abs().max(1.0);
        assert!(
            stationarity.max_abs() <= tolerance * scale,
            "the certified point must be stationary: {:.3e} > {:.3e}",
            stationarity.max_abs(),
            tolerance * scale
        );

        // Both arms must find the same optimum, not merely stop.
        let plain_objective = plain
            .penalized_objective(target.view(), &lambda, &ard)
            .expect("plain objective");
        if let Some(plain_cycles) = plain_cycles {
            assert!(
                (objective - plain_objective).abs() <= 1.0e-6 * scale,
                "accelerated {objective:.9e} vs plain {plain_objective:.9e}"
            );
            assert!(
                report.iterations <= plain_cycles,
                "acceleration must not cost cycles: {} vs plain {plain_cycles}",
                report.iterations
            );
        }
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
