//! The chart families' declared gauge directions: the dense step gauge menu, its
//! image in the arrow layout, the closed-form border quotient the Newton step pins,
//! and the per-row deflation candidates the evidence factor qualifies.

use super::*;
use super::fit_drivers::ambient_sphere_killing_directions;

impl SaeManifoldTerm {
    pub(crate) fn dense_step_gauge_vectors(&self) -> Result<Vec<Array1<f64>>, String> {
        let n = self.n_obs();
        let q = self.assignment.row_block_dim();
        let p = self.output_dim();
        let coord_offsets = self.assignment.coord_offsets();
        let beta_offsets = self.factored_border_offsets();
        let total_len = n * q + self.factored_border_dim();
        let mut out = Vec::new();
        for atom_idx in 0..self.k_atoms() {
            out.extend(self.dense_step_gauge_vectors_of_atom(
                atom_idx,
                &coord_offsets,
                &beta_offsets,
                total_len,
            )?);
        }
        if p == 0 {
            return Ok(Vec::new());
        }
        Ok(out)
    }

    /// One atom's declared step gauge generators: the per-kind menu that
    /// [`Self::dense_step_gauge_vectors`] concatenates over atoms, in atom order.
    fn dense_step_gauge_vectors_of_atom(
        &self,
        atom_idx: usize,
        coord_offsets: &[usize],
        beta_offsets: &[usize],
        total_len: usize,
    ) -> Result<Vec<Array1<f64>>, String> {
        let n = self.n_obs();
        let mut out = Vec::new();
        let d = self.assignment.coords[atom_idx].latent_dim();
        let coords = self.assignment.coords[atom_idx].as_matrix();
        match self.atoms[atom_idx].basis_kind() {
            // The Poincaré tangent patch shares the Euclidean patch's
            // translation + scale gauge orbit on the tangent coordinate
            // (the hyperbolic structure lives in the penalty, not the
            // gauge), so it deflates the same step-gauge vectors.
            // The genuinely-linear (affine) atom shares the Euclidean patch's
            // translation + scale gauge orbit on its tangent coordinate (its
            // constant column carries the translation gauge, its `t` column
            // the scale gauge), so it deflates the same step-gauge vectors.
            SaeAtomBasisKind::Linear
            | SaeAtomBasisKind::EuclideanPatch
            | SaeAtomBasisKind::Poincare => {
                for axis in 0..d {
                    let mut field = Array2::<f64>::zeros((n, d));
                    field.column_mut(axis).fill(1.0);
                    if let Some(g) = self.dense_step_gauge_vector_from_field(
                        atom_idx,
                        field.view(),
                        &coord_offsets,
                        &beta_offsets,
                        total_len,
                    )? {
                        out.push(g);
                    }
                }
                for axis in 0..d {
                    let mut field = Array2::<f64>::zeros((n, d));
                    for row in 0..n {
                        field[[row, axis]] = coords[[row, axis]];
                    }
                    if let Some(g) = self.dense_step_gauge_vector_from_field(
                        atom_idx,
                        field.view(),
                        &coord_offsets,
                        &beta_offsets,
                        total_len,
                    )? {
                        out.push(g);
                    }
                }
            }
            SaeAtomBasisKind::Duchon => {
                for axis in 0..d {
                    let mut field = Array2::<f64>::zeros((n, d));
                    field.column_mut(axis).fill(1.0);
                    if let Some(g) = self.dense_step_gauge_vector_from_field(
                        atom_idx,
                        field.view(),
                        &coord_offsets,
                        &beta_offsets,
                        total_len,
                    )? {
                        out.push(g);
                    }
                }
                for axis in 0..d {
                    let mut field = Array2::<f64>::zeros((n, d));
                    for row in 0..n {
                        field[[row, axis]] = coords[[row, axis]];
                    }
                    if let Some(g) = self.dense_step_gauge_vector_from_field(
                        atom_idx,
                        field.view(),
                        &coord_offsets,
                        &beta_offsets,
                        total_len,
                    )? {
                        out.push(g);
                    }
                }
            }
            SaeAtomBasisKind::Periodic | SaeAtomBasisKind::Torus => {
                for axis in 0..d {
                    let mut field = Array2::<f64>::zeros((n, d));
                    field.column_mut(axis).fill(1.0);
                    if let Some(g) = self.dense_step_gauge_vector_from_field(
                        atom_idx,
                        field.view(),
                        &coord_offsets,
                        &beta_offsets,
                        total_len,
                    )? {
                        out.push(g);
                    }
                }
            }
            SaeAtomBasisKind::KleinBottle => {
                if d != 2 {
                    return Err(format!(
                        "dense_step_gauge_vectors: Klein atom {atom_idx} requires latent dimension 2, got {d}"
                    ));
                }
                let mut field = Array2::<f64>::zeros((n, d));
                field.column_mut(0).fill(1.0);
                if let Some(g) = self.dense_step_gauge_vector_from_field(
                    atom_idx,
                    field.view(),
                    &coord_offsets,
                    &beta_offsets,
                    total_len,
                )? {
                    out.push(g);
                }
            }
            SaeAtomBasisKind::Sphere | SaeAtomBasisKind::ProjectivePlane => {
                if d != 3 {
                    return Err(format!(
                        "dense_step_gauge_vectors: spherical atom {atom_idx} rides the ambient cover and requires latent dimension 3, got {d}"
                    ));
                }
                let mut fields = [
                    Array2::<f64>::zeros((n, d)),
                    Array2::<f64>::zeros((n, d)),
                    Array2::<f64>::zeros((n, d)),
                ];
                for row in 0..n {
                    let directions = ambient_sphere_killing_directions([
                        coords[[row, 0]],
                        coords[[row, 1]],
                        coords[[row, 2]],
                    ]);
                    for generator in 0..3 {
                        for axis in 0..3 {
                            fields[generator][[row, axis]] = directions[generator][axis];
                        }
                    }
                }
                for field in fields {
                    if let Some(g) = self.dense_step_gauge_vector_from_field(
                        atom_idx,
                        field.view(),
                        &coord_offsets,
                        &beta_offsets,
                        total_len,
                    )? {
                        out.push(g);
                    }
                }
            }
            // `Cylinder` (`S¹ × ℝ`) carries exactly one continuous gauge: the
            // shift (rotation) of the periodic axis 0. The line axis 1 has no
            // rotational gauge and its translation is pinned by the constant
            // column, so we deflate only the axis-0 constant-shift field —
            // matching the `AtomTopology::Circle` identifiability choice.
            SaeAtomBasisKind::Cylinder => {
                let mut field = Array2::<f64>::zeros((n, d));
                if d > 0 {
                    field.column_mut(0).fill(1.0);
                }
                if let Some(g) = self.dense_step_gauge_vector_from_field(
                    atom_idx,
                    field.view(),
                    &coord_offsets,
                    &beta_offsets,
                    total_len,
                )? {
                    out.push(g);
                }
            }
            // The remaining kinds deflate no dense step gauge, each for its
            // own reason — spelled out as named arms so a new topology has
            // to state its gauge orbit here instead of silently inheriting
            // "none" from a wildcard.
            //
            // * `Mobius`: the half-twist lives in the parity culling of the
            //   basis (`trig(πks)·wᵐ`, `k + m` even), so a constant shift of
            //   the double-cover phase is NOT a symmetry of the culled span
            //   the way it is for `Periodic`/`Torus`; there is no enumerated
            //   continuous orbit to project out.
            // * `FiniteSet`: the latent is CATEGORICAL (anchor indicators),
            //   so there is no continuous coordinate to move along at all.
            // * `Precomputed`: the basis is supplied from outside this
            //   module and carries no declared gauge structure, so nothing
            //   may be assumed deflatable.
            SaeAtomBasisKind::Mobius
            | SaeAtomBasisKind::FiniteSet
            | SaeAtomBasisKind::Precomputed(_) => {}
        }
        Ok(out)
    }

    /// Put a dense-product-chart joint vector into the exact variable-stride
    /// arrow layout used by one assembled system/cache.
    ///
    /// Hard TopK removes inactive coordinate blocks row by row but retains the
    /// global decoder border.  All joint gauge/null producers naturally emit
    /// the full `n * q + beta` chart, so every consumer must cross this one seam
    /// before applying an arrow operator.  A stale layout or any shape mismatch
    /// is an invariant error; silently dropping the direction changes the
    /// quotient and can turn a chart null into a reported physical saddle.
    pub(crate) fn dense_joint_vector_in_arrow_layout(
        &self,
        dense: ArrayView1<'_, f64>,
        row_offsets: &[usize],
        border_dim: usize,
        owner: &str,
    ) -> Result<Array1<f64>, String> {
        let n = self.n_obs();
        let q = self.assignment.row_block_dim();
        let declared_border = self.factored_border_dim();
        if border_dim != declared_border {
            return Err(format!(
                "{owner}: arrow border dimension {border_dim} != term border dimension {declared_border}"
            ));
        }
        if row_offsets.len() != n + 1 || row_offsets.first() != Some(&0) {
            return Err(format!(
                "{owner}: arrow row offsets must have length {} and start at zero, got {:?}",
                n + 1,
                row_offsets
            ));
        }
        if let Some(layout) = self.last_row_layout.as_ref() {
            return layout.restrict_dense_joint_vector(
                dense,
                q,
                row_offsets,
                border_dim,
                owner,
            );
        }

        for row in 0..n {
            let span = row_offsets[row + 1]
                .checked_sub(row_offsets[row])
                .ok_or_else(|| format!("{owner}: arrow row offsets decrease at row {row}"))?;
            if span != q {
                return Err(format!(
                    "{owner}: dense arrow row {row} has width {span}, expected full chart width {q}"
                ));
            }
        }
        let expected = row_offsets[n]
            .checked_add(border_dim)
            .ok_or_else(|| format!("{owner}: arrow joint length overflows usize"))?;
        if dense.len() != expected {
            return Err(format!(
                "{owner}: dense joint vector has length {}, but the dense arrow layout has length {expected}",
                dense.len()
            ));
        }
        Ok(dense.to_owned())
    }

    /// Orthonormal analytic chart-gauge basis in one assembled arrow layout.
    /// Both dense exact-A quotient geometry and matrix-free arrow consumers use
    /// this basis, so the physical subspace cannot depend on representation.
    pub(crate) fn joint_chart_gauge_basis_for_arrow_layout(
        &self,
        row_offsets: &[usize],
        border_dim: usize,
        owner: &str,
    ) -> Result<Vec<Array1<f64>>, String> {
        let mut basis = Vec::<Array1<f64>>::new();
        // #2653 — a BORDER-dimension disagreement is not a stale layout, it is a
        // different chart. Hard TopK compaction drops inactive per-row coordinate
        // BLOCKS and retains the decoder border unchanged, so a genuine
        // full-to-compact map always agrees on the border; only the row widths
        // move (the filed 132 -> 84 case). When the border itself differs, this
        // operator is not a compaction of the joint chart at all and the
        // closed-form chart gauges simply do not live in its space. That is the
        // "no matching gauge" condition the caller already diagnoses as
        // `NonIdentifiable` — reporting it as an internal invariant error instead
        // converts a legitimate, more specific refusal into a bug report
        // (regression: `outer_gradient_solver_rejects_near_singular_cache_without_matching_gauge`
        // saw `arrow border dimension 1 != term border dimension 3`).
        // Row-layout disagreement stays a typed invariant error below, because
        // there the gauge IS mappable and silently skipping it would put an
        // analytic chart null back into the physical spectrum.
        if border_dim != self.factored_border_dim() {
            return Ok(Vec::new());
        }
        let mut orthogonality_defect = 0.0_f64;
        for dense in self.dense_step_gauge_vectors()? {
            let mut gauge = self.dense_joint_vector_in_arrow_layout(
                dense.view(),
                row_offsets,
                border_dim,
                owner,
            )?;
            let original_norm = gauge.dot(&gauge).max(0.0).sqrt();
            if !(original_norm.is_finite() && original_norm > 0.0) {
                continue;
            }
            // Two-pass MGS gives the same stable quotient basis to the dense and
            // matrix-free paths. A dependent candidate leaves only rounding: two
            // passes over `k` bases stay inside `2k·(γ_{N+4} + Σω)·‖g₀‖` (as in 10347d95e).
            for _ in 0..2 {
                for kept in &basis {
                    let coefficient = gauge.dot(kept);
                    gauge.scaled_add(-coefficient, kept);
                }
            }
            let residual_norm = gauge.dot(&gauge).max(0.0).sqrt();
            let growth = gam_linalg::roundoff::accumulation_growth(gauge.len() + 4);
            let band = 2.0 * basis.len() as f64 * (growth + orthogonality_defect);
            if !(residual_norm.is_finite()
                && residual_norm > band * original_norm)
            {
                continue;
            }
            orthogonality_defect += band * original_norm / residual_norm;
            gauge.mapv_inplace(|value| value / residual_norm);
            basis.push(gauge);
        }
        Ok(basis)
    }

    /// Closed-form chart-gauge directions restricted to the reduced β border, for
    /// the generators along which every term of the penalized objective is
    /// structurally invariant.
    ///
    /// Each returned vector is the β (decoder-border) component of a
    /// [`Self::dense_step_gauge_vectors`] orbit — the exact decoder compensation
    /// `δβ` that, paired with the coordinate motion `δt`, leaves the atom
    /// reconstruction invariant (`Φ(t+δt)·(β+δβ) = Φ(t)·β` to first order).
    ///
    /// Reconstruction invariance alone does not make a generator a null of the
    /// reduced β-Schur `S_β`. A term the orbit moves gives it slope and curvature,
    /// and installing that direction as an [`ArrowBetaGaugeQuotient`] projects live
    /// gradient out of the Newton step. On #2267's k2 fit the declared chart gauge
    /// carried |g_β| = 440.8 where the projected step saw |P g_β| = 0.714, and the fit
    /// died on "adaptive proximal correction failed" (sae2267's lane-only probe, job
    /// 617095). So a generator is declared only where invariance holds by
    /// construction, term by term:
    ///
    /// * Reconstruction: the compensation `δβ` holds it by construction.
    /// * Gate priors, including the logit Jacobians: they read only logits, which no
    ///   chart generator moves.
    /// * Decoder smoothing, the amplitude and separation barriers, and decoder
    ///   repulsion: invariant when `δβ` rotates the basis coefficients orthogonally.
    ///   That is exactly the phase and Killing generators of the harmonic families
    ///   (periodic, torus, sphere, projective plane, Klein bottle, cylinder axis 0),
    ///   whose roughness is Laplacian and so diagonal by frequency. The translation
    ///   and scale generators of the linear, Euclidean, Poincaré and Duchon patches
    ///   compensate non-orthogonally, so smoothing breaks them and they are never
    ///   declared.
    /// * Coordinate ARD: its centered prior, `½αt²` or `(α/κ²)(1 − cos κt)`, moves
    ///   under every generator on its axes, so an atom with ARD enabled declares
    ///   nothing.
    /// * Analytic registry penalties can target any slice, so any present penalty
    ///   withholds every declaration.
    /// * The rotation argument needs `δβ` to BE the coefficient rotation. The solve
    ///   returns it only when the atom's compensation design has full column rank
    ///   ([`Self::atom_compensation_has_full_column_rank`]); otherwise it returns
    ///   the minimum-norm compensation, which the penalties see move, so a
    ///   rank-deficient atom declares nothing. A two-atom torus on 42 rows with
    ///   M = 49 carried smoothing slope at 2148x to 8840x the inner tolerance along
    ///   its declared generators (sae2267, job 1102636).
    ///
    /// Where nothing is declared, the step solves the direction with its real
    /// curvature. Installing the declared set as the Faddeev–Popov quotient
    /// `P S_β P + Q Qᵀ` (`P = I − Q Qᵀ`) removes only a true null. The count follows
    /// from the chart menu and the model's terms, so it does not flicker across the
    /// ρ-walk (#2253).
    pub(crate) fn closed_form_beta_gauge_directions(
        &self,
        rho: &SaeManifoldRho,
        registry: Option<&AnalyticPenaltyRegistry>,
    ) -> Result<Vec<Array1<f64>>, String> {
        let border = self.factored_border_dim();
        if border == 0 {
            return Ok(Vec::new());
        }
        if registry.is_some_and(|registry| !registry.penalties.is_empty()) {
            return Ok(Vec::new());
        }
        let n = self.n_obs();
        let q = self.assignment.row_block_dim();
        let coord_offsets = self.assignment.coord_offsets();
        let beta_offsets = self.factored_border_offsets();
        let coord_len = n * q;
        let total_len = coord_len + border;
        let mut out = Vec::new();
        // Independence probe for the emitted set. `ArrowBetaGaugeQuotient::new`
        // treats a linearly dependent direction as a MALFORMED DECLARATION and
        // refuses the whole solve — the right contract, since a caller that
        // declares a redundant generator has mis-specified its gauge. Honoring
        // it is the producer's job: the closed-form chart menu below is a
        // GENERATING SET, and a generating set can become dependent once
        // decoder frames compress the border. Two chart symmetries whose
        // decoder compensations are distinct in the full-`B` border can have
        // the SAME image in the factored `M_k × r_k` border (an atom whose
        // decoder is rank-deficient inside its own frame collapses its
        // per-axis translation/scale compensations onto each other), and the
        // fit then dies on `direction j is zero or linearly dependent` rather
        // than solving.
        //
        // Emit a maximal independent subset. This is exactly equivalent, not a
        // relaxation: the Faddeev–Popov pin `P S_β P + Q Qᵀ` with
        // `P = I − Q Qᵀ` is a function of the gauge SPAN alone, so dropping a
        // generator that adds nothing to the span leaves `P`, the pin, and the
        // projected step unchanged. The surviving vectors are pushed in their
        // ORIGINAL form (the probe basis is internal), so a set that was
        // already independent — every full-`B` fit — reaches the constructor
        // byte-for-byte as before and its quotient is bit-identical.
        //
        // Drop predicate: EXACTLY the constructor's own, `‖residual‖² > 0`
        // after modified Gram–Schmidt in the same order. Not a tolerance of
        // ours — deliberately not, because any stricter floor would be an
        // invented threshold that silently re-shapes the quotient on fits that
        // were solving perfectly well.
        //
        // Matching the predicate makes the change provably behavior-preserving
        // wherever it was not fatal. A dropped candidate contributed no vector
        // to the probe basis (its residual was zero), so the probe basis
        // evolves identically with or without it, and the constructor re-running
        // MGS over the survivors reproduces its previous arithmetic step for
        // step. The ONLY fits that move are the ones that previously died.
        let mut probe: Vec<Array1<f64>> = Vec::new();
        for atom_idx in 0..self.k_atoms() {
            if !self.atom_step_gauge_is_structural_null(atom_idx, rho)? {
                continue;
            }
            for gauge in self.dense_step_gauge_vectors_of_atom(
                atom_idx,
                &coord_offsets,
                &beta_offsets,
                total_len,
            )? {
                if gauge.len() != coord_len + border {
                    continue;
                }
                let beta_part = gauge.slice(s![coord_len..]).to_owned();
                let norm_sq = beta_part.iter().map(|&v| v * v).sum::<f64>();
                // A gauge whose reconstruction motion is entirely absorbed by the
                // coordinate block (no decoder compensation) contributes no β-Schur
                // null direction; skip it so the quotient stays exactly the reduced
                // border nullspace.
                if !(norm_sq.is_finite() && norm_sq > 1.0e-24) {
                    continue;
                }
                let mut residual = beta_part.clone();
                for basis in &probe {
                    let coefficient = residual.dot(basis);
                    residual.scaled_add(-coefficient, basis);
                }
                let residual_norm_sq = residual.dot(&residual);
                if !(residual_norm_sq.is_finite() && residual_norm_sq > 0.0) {
                    continue;
                }
                residual *= residual_norm_sq.sqrt().recip();
                probe.push(residual);
                out.push(beta_part);
            }
        }
        Ok(out)
    }

    /// Whether every term present is structurally invariant along atom
    /// `atom_idx`'s declared step gauge generators; see
    /// [`Self::closed_form_beta_gauge_directions`] for the term-by-term derivation.
    fn atom_step_gauge_is_structural_null(
        &self,
        atom_idx: usize,
        rho: &SaeManifoldRho,
    ) -> Result<bool, String> {
        let isometry = matches!(
            self.atoms[atom_idx].basis_kind(),
            SaeAtomBasisKind::Periodic
                | SaeAtomBasisKind::Torus
                | SaeAtomBasisKind::Sphere
                | SaeAtomBasisKind::ProjectivePlane
                | SaeAtomBasisKind::KleinBottle
                | SaeAtomBasisKind::Cylinder
        );
        if !(isometry && rho.log_ard.get(atom_idx).is_some_and(|axes| axes.is_empty())) {
            return Ok(false);
        }
        self.atom_compensation_has_full_column_rank(atom_idx)
    }

    /// The design `a·Φ` that atom `atom_idx`'s decoder compensation `δβ` is solved
    /// against: one row per observation, zero where the atom is inactive.
    fn atom_compensation_design(&self, atom_idx: usize) -> Result<Array2<f64>, String> {
        let n = self.n_obs();
        let atom = &self.atoms[atom_idx];
        let m = atom.basis_size();
        let mut design = Array2::<f64>::zeros((n, m));
        for row in 0..n {
            let a = self.assignment.try_assignments_row(row)?[atom_idx];
            if a == 0.0 {
                continue;
            }
            for col in 0..m {
                design[[row, col]] = a * atom.basis_values[[row, col]];
            }
        }
        Ok(design)
    }

    /// Whether atom `atom_idx`'s compensation design keeps every column above
    /// [`design_rank_cutoff`], so that [`solve_design_least_squares`] returns the
    /// unique `δβ` rather than the minimum-norm one.
    ///
    /// A harmonic generator moves the atom inside its own span, `∂Φ·ξ = Φ·G`, so the
    /// exact compensation is the coefficient rotation `δβ = −G β`, and that rotation
    /// is what leaves the smoothing, the barriers and the repulsion invariant. On a
    /// rank-deficient design the solve drops the component of `G β` in the design's
    /// null space: the reconstruction on the active rows is still invariant, but the
    /// penalties read the dropped component and move.
    pub(crate) fn atom_compensation_has_full_column_rank(
        &self,
        atom_idx: usize,
    ) -> Result<bool, String> {
        let design = self.atom_compensation_design(atom_idx)?;
        let (rows, cols) = design.dim();
        if rows < cols {
            return Ok(false);
        }
        let decomposition = design.svd(false, false).map_err(|err| {
            format!("atom_compensation_has_full_column_rank: SVD failed: {err}")
        })?;
        let sigma = decomposition.1;
        let sigma_max = sigma.iter().fold(0.0_f64, |acc, &v| acc.max(v));
        if !(sigma_max.is_finite() && sigma_max > 0.0) {
            return Ok(false);
        }
        let cutoff = design_rank_cutoff(sigma_max, rows, cols);
        Ok(sigma.iter().filter(|&&value| value > cutoff).count() == cols)
    }

    pub(crate) fn row_gauge_deflation_for_layout(
        &self,
        row_layout: Option<&SaeRowLayout>,
    ) -> Result<Option<ArrowRowGaugeDeflation>, String> {
        let n = self.n_obs();
        let mut rows: Vec<Vec<Array1<f64>>> = Vec::with_capacity(n);
        for row in 0..n {
            let q_row = match row_layout {
                Some(layout) => layout.row_q_active(row),
                None => self.assignment.row_block_dim(),
            };
            rows.push(Vec::with_capacity(self.k_atoms().min(4)));
            match row_layout {
                Some(layout) => {
                    for (active_pos, &atom_idx) in layout.active_atoms[row].iter().enumerate() {
                        let start = layout.coord_starts[row][active_pos];
                        self.push_atom_row_gauge_deflations(
                            &mut rows[row],
                            row,
                            atom_idx,
                            start,
                            q_row,
                        )?;
                    }
                }
                None => {
                    let coord_offsets = self.assignment.coord_offsets();
                    for atom_idx in 0..self.k_atoms() {
                        self.push_atom_row_gauge_deflations(
                            &mut rows[row],
                            row,
                            atom_idx,
                            coord_offsets[atom_idx],
                            q_row,
                        )?;
                    }
                }
            }
        }
        if rows.iter().all(Vec::is_empty) {
            Ok(None)
        } else {
            Ok(Some(ArrowRowGaugeDeflation::new(rows)))
        }
    }

    pub(crate) fn push_atom_row_gauge_deflations(
        &self,
        row_dirs: &mut Vec<Array1<f64>>,
        row: usize,
        atom_idx: usize,
        coord_start: usize,
        q_row: usize,
    ) -> Result<(), String> {
        let d = self.assignment.coords[atom_idx].latent_dim();
        let atom = &self.atoms[atom_idx];
        let mut motion = vec![0.0_f64; self.output_dim()];
        let mut absolute = vec![0.0_f64; self.output_dim()];
        match self.atoms[atom_idx].basis_kind() {
            SaeAtomBasisKind::Linear
            | SaeAtomBasisKind::EuclideanPatch
            | SaeAtomBasisKind::Duchon
            | SaeAtomBasisKind::Poincare => {
                for axis in 0..d {
                    if atom.decoded_motion_is_rounding_zero(
                        row,
                        [(axis, 1.0)],
                        &mut motion,
                        &mut absolute,
                    ) {
                        continue;
                    }
                    let mut translation = Array1::<f64>::zeros(q_row);
                    translation[coord_start + axis] = 1.0;
                    row_dirs.push(translation);

                    let coord_value = self.assignment.coords[atom_idx].as_matrix()[[row, axis]];
                    let mut scale = Array1::<f64>::zeros(q_row);
                    scale[coord_start + axis] = coord_value;
                    row_dirs.push(scale);
                }
            }
            SaeAtomBasisKind::Periodic | SaeAtomBasisKind::Torus => {
                for axis in 0..d {
                    if atom.decoded_motion_is_rounding_zero(
                        row,
                        [(axis, 1.0)],
                        &mut motion,
                        &mut absolute,
                    ) {
                        continue;
                    }
                    let mut phase = Array1::<f64>::zeros(q_row);
                    phase[coord_start + axis] = 1.0;
                    row_dirs.push(phase);
                }
            }
            SaeAtomBasisKind::KleinBottle => {
                if d != 2 {
                    return Err(format!(
                        "push_atom_row_gauge_deflations: Klein atom {atom_idx} requires latent dimension 2, got {d}"
                    ));
                }
                if !atom.decoded_motion_is_rounding_zero(
                    row,
                    [(0, 1.0)],
                    &mut motion,
                    &mut absolute,
                ) {
                    let mut phase = Array1::<f64>::zeros(q_row);
                    phase[coord_start] = 1.0;
                    row_dirs.push(phase);
                }
            }
            SaeAtomBasisKind::Sphere | SaeAtomBasisKind::ProjectivePlane => {
                if d != 3 {
                    return Err(format!(
                        "push_atom_row_gauge_deflations: spherical atom {atom_idx} rides the ambient cover and requires latent dimension 3, got {d}"
                    ));
                }
                let coords = self.assignment.coords[atom_idx].as_matrix();
                let directions = ambient_sphere_killing_directions([
                    coords[[row, 0]],
                    coords[[row, 1]],
                    coords[[row, 2]],
                ]);
                for direction in directions {
                    if atom.decoded_motion_is_rounding_zero(
                        row,
                        direction.iter().copied().enumerate(),
                        &mut motion,
                        &mut absolute,
                    ) {
                        continue;
                    }
                    let mut rotation = Array1::<f64>::zeros(q_row);
                    for axis in 0..2 {
                        rotation[coord_start + axis] = direction[axis];
                    }
                    row_dirs.push(rotation);
                }
            }
            // `Cylinder` (`S¹ × ℝ`): only the periodic axis 0 carries a phase
            // (rotation) gauge; the line axis 1 has none (matching the
            // `AtomTopology::Circle` choice). Deflate the axis-0 phase only.
            SaeAtomBasisKind::Cylinder => {
                if d > 0 {
                    if !atom.decoded_motion_is_rounding_zero(
                        row,
                        [(0, 1.0)],
                        &mut motion,
                        &mut absolute,
                    ) {
                        let mut phase = Array1::<f64>::zeros(q_row);
                        phase[coord_start] = 1.0;
                        row_dirs.push(phase);
                    }
                }
            }
            // No row gauge direction, for the same per-kind reasons spelled out
            // in `dense_step_gauge_vectors`: `Mobius`'s parity-culled basis has
            // no enumerated continuous phase orbit, `FiniteSet`'s latent is
            // categorical rather than continuous, and `Precomputed` carries no
            // declared gauge structure. Named so a new topology cannot inherit
            // "none" silently.
            SaeAtomBasisKind::Mobius
            | SaeAtomBasisKind::FiniteSet
            | SaeAtomBasisKind::Precomputed(_) => {}
        }
        Ok(())
    }

    pub(crate) fn dense_step_gauge_vector_from_field(
        &self,
        atom_idx: usize,
        field: ArrayView2<'_, f64>,
        coord_offsets: &[usize],
        beta_offsets: &[usize],
        total_len: usize,
    ) -> Result<Option<Array1<f64>>, String> {
        let n = self.n_obs();
        let q = self.assignment.row_block_dim();
        let p = self.output_dim();
        let atom = &self.atoms[atom_idx];
        let m = atom.basis_size();
        let d = self.assignment.coords[atom_idx].latent_dim();
        if field.dim() != (n, d) {
            return Err(format!(
                "dense_step_gauge_vector_from_field: field shape {:?} != ({n}, {d})",
                field.dim()
            ));
        }
        let design = self.atom_compensation_design(atom_idx)?;
        let mut motion = Array2::<f64>::zeros((n, p));
        for row in 0..n {
            let assignments = self.assignment.try_assignments_row(row)?;
            let a = assignments[atom_idx];
            if a == 0.0 {
                continue;
            }
            for axis in 0..d {
                let dt = field[[row, axis]];
                if dt == 0.0 {
                    continue;
                }
                for col in 0..m {
                    let w = a * dt * atom.basis_jacobian[[row, col, axis]];
                    if w == 0.0 {
                        continue;
                    }
                    for out_col in 0..p {
                        motion[[row, out_col]] += w * atom.decoder_coefficients()[[col, out_col]];
                    }
                }
            }
        }
        let raw = motion.iter().map(|v| v * v).sum::<f64>();
        if raw == 0.0 || !raw.is_finite() {
            return Ok(None);
        }
        motion.mapv_inplace(|v| -v);
        let delta_b = solve_design_least_squares(design.view(), motion.view())?;
        let mut gauge = Array1::<f64>::zeros(total_len);
        for row in 0..n {
            let row_base = row * q + coord_offsets[atom_idx];
            for axis in 0..d {
                gauge[row_base + axis] = field[[row, axis]];
            }
        }
        let beta_base = n * q + beta_offsets[atom_idx];
        let delta_border = match atom.decoder_frame.as_ref() {
            Some(frame) => delta_b.dot(&frame.frame()),
            None => delta_b,
        };
        let border_rank = delta_border.ncols();
        for col in 0..m {
            for channel in 0..border_rank {
                gauge[beta_base + col * border_rank + channel] = delta_border[[col, channel]];
            }
        }
        Ok(Some(gauge))
    }
}
