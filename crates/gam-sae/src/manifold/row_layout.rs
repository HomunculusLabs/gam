use super::*;

/// Per-row layout for the explicit hard-TopK assignment model.
///
/// TopK gates are exactly zero or one and carry no optimizable logit
/// coordinates. Only the selected atoms' manifold coordinates enter a row's
/// Arrow-Schur block, whose dimension is exactly
/// `Σ_{k ∈ support_i} d_k`. Dense Softmax, ordered independent
/// Beta--Bernoulli, and threshold gates never use this type: each has nonzero
/// derivatives on its full support and is assembled exactly or refused before
/// allocation if the exact system does not fit the declared memory budget.
#[derive(Debug, Clone)]
pub struct SaeRowLayout {
    /// `active_atoms[row]` — sorted indices of active atoms for that row. Every
    /// active atom carries a coordinate block.
    pub active_atoms: Vec<Vec<usize>>,
    /// For row `i`, active atom `active_atoms[i][j]` has its coord block
    /// starting at compressed position `coord_starts[i][j]`.
    pub coord_starts: Vec<Vec<usize>>,
    /// Full-q coordinate offset for atom `k` (length `k_atoms`).
    pub coord_offsets_full: Vec<usize>,
    /// Per-atom coordinate dimensions, indexed by atom index.
    pub coord_dims: Vec<usize>,
}

impl SaeRowLayout {
    /// Build directly from the canonical support-sparse state. This is the
    /// production TopK path: it never constructs K-wide gates merely to recover
    /// the support indices that are already the fundamental state.
    pub(crate) fn from_assignment_state(
        state: &crate::assignment_state::SaeAssignmentState,
    ) -> Result<Self, String> {
        let mut coord_offsets_full = Vec::with_capacity(state.k_atoms());
        let mut cursor = 0usize;
        let mut coord_dims = Vec::with_capacity(state.k_atoms());
        for atom in 0..state.k_atoms() {
            coord_offsets_full.push(cursor);
            let d = state.atom_coord_dim(atom);
            coord_dims.push(d);
            cursor += d;
        }
        let active_atoms = (0..state.n_obs())
            .map(|row| {
                state
                    .support_indices(row)
                    .iter()
                    .map(|&atom| atom as usize)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let mut coord_starts = Vec::with_capacity(state.n_obs());
        for active in &active_atoms {
            let mut row_cursor = 0usize;
            let mut starts = Vec::with_capacity(active.len());
            for &atom in active {
                starts.push(row_cursor);
                row_cursor += coord_dims[atom];
            }
            coord_starts.push(starts);
        }
        Ok(Self {
            active_atoms,
            coord_starts,
            coord_offsets_full,
            coord_dims,
        })
    }

    /// Build the exact compact layout from hard TopK gates. Every row must have
    /// exactly `support_size` entries equal to one and every other entry equal
    /// to zero; accepting approximate weights here would silently change the
    /// model by dropping nonzero derivatives.
    pub(crate) fn from_topk_gates(
        assignments: &[Array1<f64>],
        support_size: usize,
        coord_dims: Vec<usize>,
        coord_offsets_full: Vec<usize>,
    ) -> Result<Self, String> {
        if support_size == 0 {
            return Err("SaeRowLayout::from_topk_gates requires positive support_size".to_string());
        }
        let mut per_row = Vec::with_capacity(assignments.len());
        for (row, gates) in assignments.iter().enumerate() {
            let mut active = Vec::with_capacity(support_size);
            for (atom, &gate) in gates.iter().enumerate() {
                if gate == 1.0 {
                    active.push(atom);
                } else if gate != 0.0 {
                    return Err(format!(
                        "SaeRowLayout::from_topk_gates: row {row}, atom {atom} has non-binary gate {gate}"
                    ));
                }
            }
            if active.len() != support_size.min(gates.len()) {
                return Err(format!(
                    "SaeRowLayout::from_topk_gates: row {row} has {} active atoms; expected {}",
                    active.len(),
                    support_size.min(gates.len())
                ));
            }
            per_row.push(active);
        }
        let mut coord_starts = Vec::with_capacity(per_row.len());
        for active in &per_row {
            let mut cursor = 0usize;
            let mut starts = Vec::with_capacity(active.len());
            for &atom in active {
                starts.push(cursor);
                cursor += coord_dims[atom];
            }
            coord_starts.push(starts);
        }
        Ok(Self {
            active_atoms: per_row,
            coord_starts,
            coord_offsets_full,
            coord_dims,
        })
    }

    /// Per-row compressed dimension: coordinate blocks for active atoms.
    pub fn row_q_active(&self, row: usize) -> usize {
        let active = &self.active_atoms[row];
        let coord_sum: usize = active.iter().map(|&k| self.coord_dims[k]).sum();
        coord_sum
    }

    /// Expand a compact TopK coordinate step back into the full coordinate row,
    /// writing zeros for inactive atoms.
    pub fn expand_row(&self, row: usize, delta_t_row: &[f64], out: &mut [f64]) {
        for v in out.iter_mut() {
            *v = 0.0;
        }
        let active = &self.active_atoms[row];
        let starts = &self.coord_starts[row];
        for (pos, &k) in active.iter().enumerate() {
            let d = self.coord_dims[k];
            let full_off = self.coord_offsets_full[k];
            for axis in 0..d {
                out[full_off + axis] = delta_t_row[starts[pos] + axis];
            }
        }
    }

    /// Restrict one dense joint `(n * q_full, beta)` vector to this layout's
    /// exact hard-TopK arrow chart.  The decoder border is copied unchanged;
    /// only inactive per-row coordinate blocks are removed.
    ///
    /// This is the single full-to-compact coordinate authority used for joint
    /// chart gauges.  A gauge is generated naturally in the dense product
    /// chart because its decoder compensation is global, while the TopK arrow
    /// operator contains only active coordinate blocks.  Treating a length
    /// mismatch as "no gauge" silently puts an analytic chart null back into
    /// the physical spectrum.  Every shape relation is therefore checked here
    /// and a malformed/stale layout is a typed error, never a skipped vector.
    pub(crate) fn restrict_dense_joint_vector(
        &self,
        dense: ArrayView1<'_, f64>,
        dense_row_width: usize,
        compact_row_offsets: &[usize],
        border_dim: usize,
        owner: &str,
    ) -> Result<Array1<f64>, String> {
        let n = self.active_atoms.len();
        if self.coord_starts.len() != n {
            return Err(format!(
                "{owner}: compact layout has {} active-atom rows but {} coordinate-start rows",
                n,
                self.coord_starts.len()
            ));
        }
        if compact_row_offsets.len() != n + 1 || compact_row_offsets.first() != Some(&0) {
            return Err(format!(
                "{owner}: compact row offsets must have length {} and start at zero, got {:?}",
                n + 1,
                compact_row_offsets
            ));
        }
        if self.coord_offsets_full.len() != self.coord_dims.len() {
            return Err(format!(
                "{owner}: full coordinate offsets ({}) and dimensions ({}) disagree",
                self.coord_offsets_full.len(),
                self.coord_dims.len()
            ));
        }
        let full_width = self
            .coord_offsets_full
            .iter()
            .zip(self.coord_dims.iter())
            .try_fold(0usize, |width, (&offset, &dimension)| {
                offset
                    .checked_add(dimension)
                    .map(|end| width.max(end))
                    .ok_or_else(|| format!("{owner}: full coordinate width overflows usize"))
            })?;
        if full_width != dense_row_width {
            return Err(format!(
                "{owner}: layout covers full row width {full_width}, but the dense joint chart has width {dense_row_width}"
            ));
        }
        let dense_t_len = n
            .checked_mul(dense_row_width)
            .ok_or_else(|| format!("{owner}: dense coordinate length overflows usize"))?;
        let expected_dense_len = dense_t_len
            .checked_add(border_dim)
            .ok_or_else(|| format!("{owner}: dense joint length overflows usize"))?;
        if dense.len() != expected_dense_len {
            return Err(format!(
                "{owner}: dense joint vector has length {}, expected {expected_dense_len} ({dense_t_len} coordinates + {border_dim} border)",
                dense.len()
            ));
        }

        for row in 0..n {
            let active = &self.active_atoms[row];
            let starts = &self.coord_starts[row];
            if active.len() != starts.len() {
                return Err(format!(
                    "{owner}: compact row {row} has {} active atoms but {} coordinate starts",
                    active.len(),
                    starts.len()
                ));
            }
            if active.windows(2).any(|pair| pair[0] >= pair[1]) {
                return Err(format!(
                    "{owner}: compact row {row} active atoms are not strictly increasing"
                ));
            }
            let mut expected_start = 0usize;
            for (&atom, &start) in active.iter().zip(starts.iter()) {
                let Some(&dimension) = self.coord_dims.get(atom) else {
                    return Err(format!(
                        "{owner}: compact row {row} names atom {atom}, outside {} coordinate blocks",
                        self.coord_dims.len()
                    ));
                };
                if start != expected_start {
                    return Err(format!(
                        "{owner}: compact row {row}, atom {atom} starts at {start}, expected contiguous offset {expected_start}"
                    ));
                }
                expected_start = expected_start.checked_add(dimension).ok_or_else(|| {
                    format!("{owner}: compact row {row} width overflows usize")
                })?;
            }
            let row_span = compact_row_offsets[row + 1]
                .checked_sub(compact_row_offsets[row])
                .ok_or_else(|| format!("{owner}: compact row offsets decrease at row {row}"))?;
            if row_span != expected_start {
                return Err(format!(
                    "{owner}: compact row {row} offset span is {row_span}, but its active coordinate width is {expected_start}"
                ));
            }
        }

        let compact_t_len = compact_row_offsets[n];
        let compact_len = compact_t_len
            .checked_add(border_dim)
            .ok_or_else(|| format!("{owner}: compact joint length overflows usize"))?;
        let mut compact = Array1::<f64>::zeros(compact_len);
        for row in 0..n {
            let dense_row = row * dense_row_width;
            let compact_row = compact_row_offsets[row];
            for (position, &atom) in self.active_atoms[row].iter().enumerate() {
                let dimension = self.coord_dims[atom];
                let dense_start = dense_row + self.coord_offsets_full[atom];
                let compact_start = compact_row + self.coord_starts[row][position];
                compact
                    .slice_mut(s![compact_start..compact_start + dimension])
                    .assign(&dense.slice(s![dense_start..dense_start + dimension]));
            }
        }
        compact
            .slice_mut(s![compact_t_len..])
            .assign(&dense.slice(s![dense_t_len..]));
        Ok(compact)
    }
}

#[cfg(test)]
mod support_state_tests {
    use super::*;
    use crate::assignment_state::{SaeAssignmentAtomSpec, SaeAssignmentState};

    #[test]
    fn direct_layout_preserves_heterogeneous_active_offsets() {
        let state = SaeAssignmentState::from_topk_support_heterogeneous(
            2,
            4,
            2,
            vec![
                SaeAssignmentAtomSpec::euclidean(1),
                SaeAssignmentAtomSpec::euclidean(3),
                SaeAssignmentAtomSpec::euclidean(2),
                SaeAssignmentAtomSpec::euclidean(1),
            ],
            vec![vec![2, 0], vec![3, 1]],
            vec![vec![1.0; 2]; 2],
            vec![vec![0.0; 3], vec![0.0; 4]],
        )
        .expect("state builds");
        let layout = SaeRowLayout::from_assignment_state(&state).expect("layout builds");
        assert_eq!(layout.active_atoms, vec![vec![0, 2], vec![1, 3]]);
        assert_eq!(layout.coord_starts, vec![vec![0, 1], vec![0, 3]]);
        assert_eq!(layout.coord_dims, vec![1, 3, 2, 1]);
        assert_eq!(layout.coord_offsets_full, vec![0, 1, 4, 6]);
        assert_eq!(layout.row_q_active(0), 3);
        assert_eq!(layout.row_q_active(1), 4);
    }
}
