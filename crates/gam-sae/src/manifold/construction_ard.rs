//! ARD (automatic relevance determination) coordinate-precision + latent-block
//! helpers for `SaeManifoldTerm`, split out of `construction.rs` to keep that
//! file under the 10k-line ban gate.
use super::*;
use gam_math::constrained_partition::{
    bingham_sphere_log_partition_and_second_moments,
    normal_interval_log_mass_and_log_precision_score,
};
use gam_math::special::bessel_i0_centered_terms_from_log_abs;
use gam_terms::latent::CoordinatePriorSupport;

/// Per-row log partition of one atom's ARD coordinate prior and its
/// log-precision derivatives (see [`SaeManifoldTerm::ard_log_partition`]).
pub(crate) struct ArdLogPartition {
    /// One log partition per support factor, in axis order.
    pub(crate) per_factor: Vec<f64>,
    /// `∂/∂log α_axis` of the summed log partition, one entry per axis.
    pub(crate) log_precision_gradient: Array1<f64>,
}

impl SaeManifoldTerm {
    /// Per-row log partition of an atom's ARD coordinate prior over the
    /// coordinate's actual support, with its log-precision derivatives.
    ///
    /// Each factor enters paired with the Laplace integration constant
    /// `−½·log 2π` per integrated dimension, which `½·log|A|` leaves out
    /// (#2933 F26):
    ///
    /// * a line axis: `−½ log α`, from the Gaussian `√(2π/α)`;
    /// * a periodic axis: the von Mises partition `log P − η + log I0(η) − ½ log 2π`
    ///   with `η = αP²/(2π)²`, where `P` is the PRIOR period `prior_periods` gives
    ///   (a quotient atom's half-turned axis uses the half period, #2933 F25);
    /// * an interval `[lo, hi]`: `−½ log α + log[Φ(hi√α) − Φ(lo√α)]`. The mass
    ///   factor reaches one only as the interval covers the line. On `[−1, 1]` at
    ///   `α = 1` the log-precision derivative is `−0.1456`, not the line's `−½`
    ///   (#2933 F24).
    /// * an embedded unit sphere `S^(d−1)` holding `d` axes: the Bingham partition
    ///   of `Σ_a ½α_a x_a²` against the surface measure, less `((d−1)/2)·log 2π`,
    ///   with `∂/∂log α_a = −½α_a·E[x_a²]`. Since `‖x‖ = 1`, adding one constant to
    ///   every `α_a` leaves the normalized prior unchanged, and an isotropic
    ///   precision carries no score. The three independent line normalizers this
    ///   replaces scored `−1` per row at `α = 1` (#2933 F24).
    pub(crate) fn ard_log_partition(
        supports: &[CoordinatePriorSupport],
        prior_periods: &[Option<f64>],
        log_alpha: ndarray::ArrayView1<'_, f64>,
        alpha: ndarray::ArrayView1<'_, f64>,
    ) -> Result<ArdLogPartition, String> {
        if prior_periods.len() != log_alpha.len() {
            return Err(format!(
                "ARD log partition: {} prior periods for {} ARD axes",
                prior_periods.len(),
                log_alpha.len()
            ));
        }
        let mut per_factor = Vec::with_capacity(supports.len());
        let mut log_precision_gradient = Array1::<f64>::zeros(log_alpha.len());
        let mut axis = 0;
        for support in supports {
            let width = support.ambient_axes();
            if axis + width > log_alpha.len() {
                return Err(format!(
                    "ARD log partition: support factors span more than the {} ARD axes",
                    log_alpha.len()
                ));
            }
            let periodic = matches!(support, CoordinatePriorSupport::Circle { .. });
            if prior_periods[axis..axis + width].iter().any(|period| period.is_some() != periodic) {
                return Err(format!(
                    "ARD log partition: support {support:?} at axis {axis} disagrees with the \
                     prior periods {:?}",
                    &prior_periods[axis..axis + width]
                ));
            }
            let log_partition = match *support {
                CoordinatePriorSupport::Line => {
                    log_precision_gradient[axis] = -0.5;
                    -0.5 * log_alpha[axis]
                }
                CoordinatePriorSupport::Circle { .. } => {
                    let Some(period) = prior_periods[axis] else {
                        return Err(format!("ARD log partition: periodic axis {axis} has no prior period"));
                    };
                    // Evaluate η = αP²/(2π)² in log space: both η and the
                    // intermediate κ² can leave the float range even when the
                    // centered log-partition remains representable. The partition
                    // over one period is `Z(α) = ∫₀ᴾ exp[-V] dt = P·e^{-η}·I0(η)`
                    // (sub `u=κt`, `dt = P/(2π) du`), so `log Z = log P − η + log I0(η)`,
                    // which tends to `½·log(2π/α)` as η → ∞ and is paired the way the
                    // line's `½·log 2π` is. The constants are ρ-independent.
                    let log_eta =
                        log_alpha[axis] + 2.0 * (period.ln() - std::f64::consts::TAU.ln());
                    let (centered_log_i0, _, scaled_derivative) =
                        bessel_i0_centered_terms_from_log_abs(log_eta);
                    // d/d(log α) of `-η + log I0(η)` is `η·(I1/I0−1)`. The centered
                    // primitive evaluates the complete product, so its `−½` large-η
                    // limit survives after the ordinary ratio has rounded to one and
                    // even when η itself is not representable.
                    log_precision_gradient[axis] = scaled_derivative;
                    period.ln() + centered_log_i0 - 0.5 * std::f64::consts::TAU.ln()
                }
                CoordinatePriorSupport::Interval { lo, hi } => {
                    let root = alpha[axis].sqrt();
                    let (log_mass, score) =
                        normal_interval_log_mass_and_log_precision_score(lo * root, hi * root)
                            .map_err(|error| {
                                format!(
                                    "ARD interval log partition on [{lo}, {hi}] at precision {}: {error}",
                                    alpha[axis]
                                )
                            })?;
                    log_precision_gradient[axis] = -0.5 + score;
                    -0.5 * log_alpha[axis] + log_mass
                }
                CoordinatePriorSupport::Sphere { dim } => {
                    let lambda: Vec<f64> = (axis..axis + width).map(|a| 0.5 * alpha[a]).collect();
                    let (log_z, second_moments) =
                        bingham_sphere_log_partition_and_second_moments(&lambda)
                            .map_err(|error| format!("ARD sphere log partition: {error}"))?;
                    for (offset, (&energy_coefficient, &moment)) in
                        lambda.iter().zip(second_moments.iter()).enumerate()
                    {
                        log_precision_gradient[axis + offset] = -energy_coefficient * moment;
                    }
                    log_z - 0.5 * (dim as f64 - 1.0) * std::f64::consts::TAU.ln()
                }
            };
            per_factor.push(log_partition);
            axis += width;
        }
        if axis != log_alpha.len() {
            return Err(format!(
                "ARD log partition: support factors span {axis} axes but the atom has {} ARD axes",
                log_alpha.len()
            ));
        }
        Ok(ArdLogPartition {
            per_factor,
            log_precision_gradient,
        })
    }

    /// Per-axis period of atom `atom`'s ARD coordinate prior: the kind's
    /// [`SaeAtomBasisKind::ard_axis_periods`] of the coordinate block's wrap
    /// periods, so a quotient atom's half-turned axis carries its deck-invariant
    /// half period (#2933 F25). Every prior consumer (value, gradient, curvature,
    /// traces, IFT channels) reads this seam; `effective_axis_periods` stays the
    /// retraction's wrap period.
    pub(crate) fn ard_axis_periods(&self, atom: usize) -> Vec<Option<f64>> {
        self.atoms[atom]
            .basis_kind()
            .ard_axis_periods(&self.assignment.coords[atom].effective_axis_periods())
    }

    /// [`Self::ard_axis_periods`] for every atom, hoisted out of row loops.
    pub(crate) fn all_ard_axis_periods(&self) -> Vec<Vec<Option<f64>>> {
        (0..self.assignment.coords.len())
            .map(|atom| self.ard_axis_periods(atom))
            .collect()
    }

    /// Validate the ARD table against this term's atom geometry and materialize
    /// each physical precision exactly once. This is the structural choke point
    /// shared by assembly, value, traces, exact-Hessian, and IFT channels.
    pub(crate) fn validated_ard_precisions(
        &self,
        rho: &SaeManifoldRho,
    ) -> Result<Vec<Array1<f64>>, String> {
        if rho.log_ard.len() != self.k_atoms() {
            return Err(format!(
                "ARD rho has {} atom blocks but term has {} atoms",
                rho.log_ard.len(),
                self.k_atoms()
            ));
        }
        for (atom, coordinate) in self.assignment.coords.iter().enumerate() {
            let stored = rho.log_ard[atom].len();
            let dimension = coordinate.latent_dim();
            if stored != 0 && stored != dimension {
                return Err(format!(
                    "ARD rho atom {atom} has {stored} axes; expected 0 (disabled) or \
                     latent dimension {dimension}"
                ));
            }
        }
        rho.ard_precisions()
    }

    /// Per-atom, per-axis coordinate sum-of-squares `‖t_kj‖² = Σ_i t_{i,k,j}²`.
    ///
    /// This is the data-fit sufficient statistic for the ARD precision update
    /// (the numerator-side `‖t‖²` of the deleted `α = n/‖t‖²` rule). Returned
    /// per atom as an `Array1` of length `d_k`.
    ///
    /// On a *periodic* (Circle) axis the relevant statistic is the von-Mises
    /// energy-equivalent `Σ_i 2/α·V(t_i) = Σ_i (2/κ²)(1−cos κ t_i)` (independent
    /// of α), so that `½·α·sumsq == Σ_i V(t_i)` matches `ard_value`. This keeps
    /// the Mackay/Fellner–Schall fixed point `α ← n / (sumsq + tr H⁻¹)`
    /// consistent with the actual periodic prior energy rather than the
    /// origin-dependent raw `t²`.
    pub(crate) fn ard_coord_sumsq(&self) -> Vec<Array1<f64>> {
        // Horvitz–Thompson row weighting: the `‖t‖²` sufficient statistic is the
        // numerator of the same `α ← n_eff / (Σ sq_equiv + tr H⁻¹)` fixed point the
        // (now weight-aware) `ard_value` energy defines, so it MUST carry the SAME
        // per-row inclusion weight `wᵢ` — else the subsampled MacKay/Fellner–Schall
        // step ranks a different precision than the criterion's energy. `None` ⇒
        // `w_row = 1`, bit-for-bit the historical sum.
        let row_w = self.row_loss_weights.as_deref();
        let mut out = Vec::with_capacity(self.k_atoms());
        for (atom_idx, coord) in self.assignment.coords.iter().enumerate() {
            let d = coord.latent_dim();
            let periods = self.ard_axis_periods(atom_idx);
            let mut sq = Array1::<f64>::zeros(d);
            for row in 0..coord.n_obs() {
                let w_row = row_w.map_or(1.0, |w| w[row]);
                let t = coord.row(row);
                for axis in 0..d {
                    // `sq_equiv` is independent of `alpha`; pass 1.0.
                    sq[axis] += w_row * ArdAxisPrior::eval(1.0, t[axis], periods[axis]).sq_equiv;
                }
            }
            out.push(sq);
        }
        out
    }

    /// Per-atom, per-axis posterior-variance trace `tr_kj(H⁻¹) =
    /// Σ_i [(H⁻¹)_tt]_{(i,k,j),(i,k,j)}` from the converged factor cache.
    ///
    /// `cache.latent_block_inverse_diagonal()` returns the diagonal of the
    /// latent block `(H⁻¹)_tt` in the cache's compact per-row `delta_t`
    /// layout (length `row_offsets[N]`). A compact hard-TopK row contains only
    /// the selected atoms' coordinate axes; a dense row contains assignment
    /// coordinates followed by all atom-coordinate axes. This routine
    /// sums those diagonal entries over the coord positions belonging to each
    /// `(atom k, axis j)` across all observation rows where atom `k` is active.
    ///
    /// `self.last_row_layout` must be the layout from the *same* assemble that
    /// produced `cache`:
    /// - `Some(layout)`: compact exact hard-TopK support. For row `i`, atom `k`'s position in the
    ///   active list gives its compact coord-block start `coord_starts[i][pos]`;
    ///   inactive atoms contribute 0 (the prior dominates there anyway).
    /// - `None`: dense full-support layout, uniform row dim
    ///   `q = assignment_dim + Σ d_k`; atom `k`'s coord block sits at the
    ///   fixed full-row offset `coord_offsets[k]` after the assignment chart.
    ///
    /// This `tr_kj(H⁻¹)` is exactly the posterior-variance term the deleted
    /// `α = n/‖t‖²` rule dropped; the corrected Mackay/Fellner-Schall fixed
    /// point is `α_new = n / (‖t_kj‖² + tr_kj(H⁻¹))`.
    ///
    /// At `K ≥ ARD_TRACE_HUTCHINSON_MIN_ATOMS` the exact selected-inverse diagonal
    /// (one dense `K×K` Schur solve per latent coordinate — `O(total_t·K²) ≈
    /// O(K³)` at massive `K`) is replaced by the matrix-free Hutchinson estimate
    /// [`Self::latent_block_inverse_diagonal_hutchinson`]; below it the exact
    /// diagonal is used unchanged (bit-for-bit tests preserved).
    pub(crate) fn ard_inverse_traces(
        &self,
        cache: &ArrowFactorCache,
    ) -> Result<Vec<Array1<f64>>, ArrowSchurError> {
        let inv_diag = if self.k_atoms() >= Self::ARD_TRACE_HUTCHINSON_MIN_ATOMS {
            // Massive-K: `total_t` dense Schur solves is infeasible — estimate the
            // whole latent inverse diagonal matrix-free with one full-arrow solve
            // per Hutchinson probe (the grouped sums below tolerate the stochastic
            // error, as this feeds a Fellner–Schall / dispersion denominator).
            Self::latent_block_inverse_diagonal_hutchinson(
                cache,
                Self::ARD_TRACE_HUTCHINSON_PROBES,
                Self::ARD_TRACE_HUTCHINSON_SEED,
            )?
        } else {
            cache.latent_block_inverse_diagonal()?
        };
        Ok(self.accumulate_latent_inverse_diagonal(cache, &inv_diag, |_, _, _| 1.0))
    }

    /// Sum the latent inverse diagonal into per-`(atom, axis)` groups, one
    /// dimensionless `slot_factor(row, atom, axis)` per slot.
    ///
    /// The single source of truth for walking the arrow's latent layout — both
    /// the compact hard-TopK support and the dense full-support layout — so the
    /// two quantities the ARD machinery needs off the same diagonal
    /// ([`Self::ard_inverse_traces`] with `factor ≡ 1`, and
    /// [`Self::ard_shrinkage_traces`] with the per-row prior-curvature factor)
    /// cannot drift apart in which slots they visit.
    ///
    /// Horvitz–Thompson row weight, IDENTICAL to `ard_coord_sumsq`'s numerator
    /// weighting: the posterior-variance trace `tr H⁻¹` is the OTHER half of
    /// the `α ← n_eff / (Σ wᵢ t̂ᵢ² + Σ wᵢ (H⁻¹)ᵢᵢ)` MacKay/Fellner–Schall
    /// fixed point, so a retained row standing in for `wᵢ` rows must
    /// contribute its posterior variance `wᵢ` times too — else the α-step's
    /// denominator uses a different inclusion measure than its numerator and
    /// `n_eff`. Commit 4862e8355 weighted value/sumsq/gradient/curvature
    /// "together" but missed this trace channel. `None` ⇒ `wᵢ = 1`,
    /// bit-for-bit the historical unweighted sum.
    fn accumulate_latent_inverse_diagonal<F>(
        &self,
        cache: &ArrowFactorCache,
        inv_diag: &Array1<f64>,
        mut slot_factor: F,
    ) -> Vec<Array1<f64>>
    where
        F: FnMut(usize, usize, usize) -> f64,
    {
        let n = self.n_obs();
        let coord_offsets = self.assignment.coord_offsets();
        let row_w = self.row_loss_weights.as_deref();
        let mut traces: Vec<Array1<f64>> = self
            .assignment
            .coords
            .iter()
            .map(|c| Array1::<f64>::zeros(c.latent_dim()))
            .collect();
        for row in 0..n {
            let row_base = cache.row_offsets[row];
            let w_row = row_w.map_or(1.0, |w| w[row]);
            match self.last_row_layout {
                Some(ref layout) => {
                    let active = &layout.active_atoms[row];
                    let starts = &layout.coord_starts[row];
                    for (pos, &k) in active.iter().enumerate() {
                        let d = self.assignment.coords[k].latent_dim();
                        let block_start = starts[pos];
                        for axis in 0..d {
                            traces[k][axis] += slot_factor(row, k, axis)
                                * w_row
                                * inv_diag[row_base + block_start + axis];
                        }
                    }
                }
                None => {
                    for k in 0..self.k_atoms() {
                        let d = self.assignment.coords[k].latent_dim();
                        let block_start = coord_offsets[k];
                        for axis in 0..d {
                            traces[k][axis] += slot_factor(row, k, axis)
                                * w_row
                                * inv_diag[row_base + block_start + axis];
                        }
                    }
                }
            }
        }
        traces
    }

    /// Per-`(atom, axis)` ARD SHRINKAGE trace
    /// `τ_kj = Σ_i f(t_ikj) · wᵢ · [H⁻¹]_{ss}`, the dimensionless companion of
    /// [`Self::ard_inverse_traces`] that the effective-degrees-of-freedom
    /// identity — not the Fellner–Schall α-step — is a function of.
    ///
    /// # Why this is a different quantity from `tr(H⁻¹)` (#2499)
    ///
    /// The ARD EDF identity is `edf_kj = n_active − Σ_i P_i·[H⁻¹]_{ss}` where
    /// `P_i` is the prior curvature the arrow was ACTUALLY assembled with at
    /// row `i`. Writing that shrinkage as `α·tr(H⁻¹)` silently assumes
    /// `P_i ≡ α`, which holds on a Euclidean axis and FAILS on a periodic one:
    /// the von-Mises coordinate prior has exact curvature `V'' = α·cos κt`, and
    /// what the factorization declares is its PSD majorizer
    /// `P_i = α·softplus_{τ₀}(cos κt_i) ∈ [0, α]`
    /// ([`ArdAxisPrior::psd_majorizer_hess`], written into `htt` by both
    /// assembly paths). With `P_i < α` the slot has `[H⁻¹]_{ss} > 1/α`, so
    /// `α·tr(H⁻¹)` is bounded by nothing and the "EDF" leaves `[0, n_active]`
    /// from below — measured at `−72.56` against an interval of width `24` on a
    /// `Circle` fixture, ten orders past any roundoff tolerance.
    ///
    /// With the true `P_i` restored the interval is a theorem again. Write
    /// `H = C + P` with Gauss-Newton data curvature `C ⪰ 0` and diagonal
    /// `P ⪰ 0`. For any slot `s` with `P_s > 0`, `H ⪰ P` gives
    /// `[H⁻¹]_{ss} ≤ 1/P_s`, so `P_s·[H⁻¹]_{ss} ∈ [0, 1]`; where `P_s = 0` the
    /// term is exactly `0`. Summing over the axis's `n_active` slots puts the
    /// shrinkage in `[0, n_active]` and the EDF in `[0, n_active]` — with no
    /// appeal to `λ_min(H) ≥ α`, which a periodic axis does not satisfy.
    ///
    /// Factored as `α·τ_kj` with `τ` carrying only the DIMENSIONLESS
    /// `f = softplus_{τ₀}(cos κt)` ([`ArdAxisPrior::curvature_shrinkage_factor`])
    /// so that a Euclidean axis has `f ≡ 1` and this returns
    /// [`Self::ard_inverse_traces`] bit-for-bit.
    /// Needs no `rho`: `α` is factored out, and the only ρ-dependence left in
    /// `f` is through the fitted coordinates themselves.
    pub(crate) fn ard_shrinkage_traces(
        &self,
        cache: &ArrowFactorCache,
    ) -> Result<Vec<Array1<f64>>, String> {
        let inv_diag = if self.k_atoms() >= Self::ARD_TRACE_HUTCHINSON_MIN_ATOMS {
            Self::latent_block_inverse_diagonal_hutchinson(
                cache,
                Self::ARD_TRACE_HUTCHINSON_PROBES,
                Self::ARD_TRACE_HUTCHINSON_SEED,
            )
            .map_err(|e| format!("ard_shrinkage_traces: {e}"))?
        } else {
            cache
                .latent_block_inverse_diagonal()
                .map_err(|e| format!("ard_shrinkage_traces: {e}"))?
        };
        let periods: Vec<Vec<Option<f64>>> = self.all_ard_axis_periods();
        // Hoisting the per-axis coordinate columns is not possible (the factor is
        // per row AND per axis), but the period lookup and the coordinate read are
        // both O(1), so this stays one pass over the same slots.
        let coords = &self.assignment.coords;
        Ok(
            self.accumulate_latent_inverse_diagonal(cache, &inv_diag, |row, k, axis| {
                ArdAxisPrior::curvature_shrinkage_factor(coords[k].row(row)[axis], periods[k][axis])
            }),
        )
    }

    /// Per-atom, per-axis posterior-variance trace `tr_kj(H⁻¹)` — the SAME
    /// quantity [`Self::ard_inverse_traces`] returns — from the #2080 SHARED
    /// selected-inverse bundle instead of the dense Schur factor
    /// (`full_inverse_apply` / `latent_block_inverse_diagonal`). The massive-`K`
    /// surrogate-lane replacement that removes the last dense `S⁻¹` from the ARD
    /// Fellner–Schall denominator.
    ///
    /// # Reformulation (arrow selected-inverse, no dense `S⁻¹`)
    ///
    /// For an arrow `H = [[A (⊕_i A_i), B], [Bᵀ, C]]` with reduced Schur
    /// complement `S`, the per-row latent block is exactly
    /// `(H⁻¹)_{t_i t_i} = A_i⁻¹ + G_i S⁻¹ G_iᵀ`, `A_i = H_tt^(i)` (the per-row
    /// `undamped_factor`), `B_i = H_tβ^(i)` (via `apply_htbeta_row`),
    /// `G_i = A_i⁻¹ B_i`. So the per-slot diagonal the ARD denominator sums splits
    /// into a ROW-LOCAL exact term `(A_i⁻¹)[s,s]` and a border term
    /// `(G_i S⁻¹ G_iᵀ)[s,s] = g_sᵀ S⁻¹ g_s`, `g_s = G_iᵀ e_s`. Summed over rows
    /// for a FIXED `(atom k, axis a)` — the group the ARD α-fixed-point actually
    /// needs — the border piece is a trace `Σ_i g_{s_i}ᵀ S⁻¹ g_{s_i} =
    /// tr(S⁻¹ M_{ka})`, `M_{ka} = Σ_i g_{s_i} g_{s_i}ᵀ`, estimated off the shared
    /// bundle `(z_j, S⁻¹ z_j)` by
    ///   `tr(S⁻¹ M_{ka}) = (1/m) Σ_j Σ_i (g_{s_i}ᵀ S⁻¹ z_j)(g_{s_i}ᵀ z_j)
    ///                   = (1/m) Σ_j Σ_i s_ij[s_i]·w_ij[s_i]`,
    /// with `w_ij = G_i z_j = A_i⁻¹ B_i z_j` and `s_ij = G_i S⁻¹ z_j =
    /// A_i⁻¹ B_i (S⁻¹ z_j)` — both per-row `t`-space vectors from
    /// `apply_htbeta_row` + a per-row Cholesky solve. The final `H_βt` never
    /// appears: it is absorbed by contracting against `S⁻¹ z_j`. Everything is
    /// sourced from the cache (`undamped_factor` + `apply_htbeta_row`) plus the
    /// bundle — no `ArrowSchurSystem`, no dense `S⁻¹`.
    ///
    /// `probes` and `sinv_probes` are the surrogate lane's frozen `(z_j, S⁻¹ z_j)`
    /// pairs (each length `cache.k`, the reduced-Schur border dim); with
    /// full-basis probes `√k·e_j` the Hutchinson average is exact, which the FD
    /// gate exploits to assert equality with [`Self::ard_inverse_traces`]. HT
    /// row-weighting matches `ard_inverse_traces` bit-for-bit (both diagonal parts
    /// carry `w_row`; `None` ⇒ `w_row = 1`).
    pub(crate) fn ard_inverse_traces_from_probes(
        &self,
        cache: &ArrowFactorCache,
        probes: &[Array1<f64>],
        sinv_probes: &[Array1<f64>],
    ) -> Result<Vec<Array1<f64>>, String> {
        let m = probes.len();
        if m == 0 || sinv_probes.len() != m {
            return Err(format!(
                "ard_inverse_traces_from_probes: need matching non-empty probe/solve \
                 bundles, got {m} probes and {} solves",
                sinv_probes.len()
            ));
        }
        let k_border = cache.k;
        for (label, set) in [("probe", probes), ("solve", sinv_probes)] {
            for (j, v) in set.iter().enumerate() {
                if v.len() != k_border {
                    return Err(format!(
                        "ard_inverse_traces_from_probes: {label} {j} has length {} != border \
                         dim {k_border}",
                        v.len()
                    ));
                }
            }
        }
        let n = self.n_obs();
        let coord_offsets = self.assignment.coord_offsets();
        let row_w = self.row_loss_weights.as_deref();
        let inv_m = 1.0 / (m as f64);
        let mut traces: Vec<Array1<f64>> = self
            .assignment
            .coords
            .iter()
            .map(|c| Array1::<f64>::zeros(c.latent_dim()))
            .collect();
        for row in 0..n {
            let q = cache.row_dims[row];
            let w_row = row_w.map_or(1.0, |w| w[row]);
            let factor = cache.undamped_factor(row);
            // A_i⁻¹ diagonal (row-local, exact): solve A_i e_s = e_s per local slot.
            let mut a_inv_diag = Array1::<f64>::zeros(q);
            let mut e_s = Array1::<f64>::zeros(q);
            for s in 0..q {
                e_s.fill(0.0);
                e_s[s] = 1.0;
                let col = cholesky_solve_vector(factor, e_s.view());
                a_inv_diag[s] = col[s];
            }
            // Per-probe border vectors w_ij = A_i⁻¹ B_i z_j and s_ij = A_i⁻¹ B_i
            // (S⁻¹ z_j), both row-local `t`-space (length q).
            let mut w_probes: Vec<Array1<f64>> = Vec::with_capacity(m);
            let mut s_probes: Vec<Array1<f64>> = Vec::with_capacity(m);
            let mut b_tmp = Array1::<f64>::zeros(q);
            for j in 0..m {
                b_tmp.fill(0.0);
                if !cache.apply_htbeta_row(row, probes[j].view(), &mut b_tmp) {
                    return Err(format!(
                        "ard_inverse_traces_from_probes: H_tβ^({row}) probe apply failed"
                    ));
                }
                w_probes.push(cholesky_solve_vector(factor, b_tmp.view()));
                b_tmp.fill(0.0);
                if !cache.apply_htbeta_row(row, sinv_probes[j].view(), &mut b_tmp) {
                    return Err(format!(
                        "ard_inverse_traces_from_probes: H_tβ^({row}) solve apply failed"
                    ));
                }
                s_probes.push(cholesky_solve_vector(factor, b_tmp.view()));
            }
            // Per-slot diagonal = (A_i⁻¹)[s,s] + (1/m) Σ_j w_ij[s]·s_ij[s]; sum into
            // the owning (atom, axis) trace exactly as `ard_inverse_traces` does.
            let accumulate = |k: usize, block_start: usize, traces: &mut Vec<Array1<f64>>| {
                let d = self.assignment.coords[k].latent_dim();
                for axis in 0..d {
                    let s = block_start + axis;
                    let mut border = 0.0_f64;
                    for j in 0..m {
                        border += w_probes[j][s] * s_probes[j][s];
                    }
                    traces[k][axis] += w_row * (a_inv_diag[s] + inv_m * border);
                }
            };
            match self.last_row_layout {
                Some(ref layout) => {
                    let active = &layout.active_atoms[row];
                    let starts = &layout.coord_starts[row];
                    for (pos, &k) in active.iter().enumerate() {
                        accumulate(k, starts[pos], &mut traces);
                    }
                }
                None => {
                    for k in 0..self.k_atoms() {
                        accumulate(k, coord_offsets[k], &mut traces);
                    }
                }
            }
        }
        Ok(traces)
    }

    /// Atom-count threshold at/above which [`Self::ard_inverse_traces`] switches
    /// from the exact selected-inverse latent diagonal (one dense `K×K` Schur
    /// solve per latent coordinate — the `O(total_t·K²) ≈ O(K³)` massive-`K`
    /// wall) to the matrix-free Hutchinson stochastic-diagonal estimator
    /// [`Self::latent_block_inverse_diagonal_hutchinson`]. Set to match the
    /// smoothness-dof Hutchinson gate ([`Self::SMOOTHNESS_DOF_HUTCHINSON_MIN_ATOMS`]),
    /// well above every exact-path test fixture so ordinary-`K` behaviour — and
    /// its bit-for-bit tests — is unchanged; the estimator engages only in the
    /// massive dictionary regime (`K` up to 32k).
    pub(crate) const ARD_TRACE_HUTCHINSON_MIN_ATOMS: usize = 2048;
    /// Rademacher probe count for the Hutchinson latent-inverse-diagonal
    /// estimator. One [`ArrowFactorCache::full_inverse_apply`] per probe yields
    /// the WHOLE diagonal at once, so this is the total full-arrow solve count
    /// that replaces the exact `total_t` per-coordinate Schur solves.
    pub(crate) const ARD_TRACE_HUTCHINSON_PROBES: usize = 64;
    /// Fixed base seed so the ARD-trace estimate is bit-reproducible across REML
    /// outer iterations (cf. the SLQ log-det and smoothness-dof seeds).
    pub(crate) const ARD_TRACE_HUTCHINSON_SEED: u64 = 0x5AED_A3D0_1ACE_9C01;

    /// Matrix-free Hutchinson estimate of `diag((H⁻¹)_tt)` — the SAME quantity
    /// [`ArrowFactorCache::latent_block_inverse_diagonal`] returns EXACTLY, but at
    /// `O(num_probes · matvec)` instead of the exact `O(total_t · K²)`.
    ///
    /// The exact selected-inverse builds the latent inverse diagonal one
    /// coordinate at a time, each coordinate paying a dense `K×K` Schur solve;
    /// over all `total_t = Σ_i d_i` latent coordinates that is `O(total_t·K²) ≈
    /// O(K³)` at massive `K` (32k). This estimator replaces the per-coordinate
    /// loop with `num_probes` full-arrow solves: for a Rademacher probe `z` over
    /// the `t`-block (`E[z zᵀ] = I`), `u_t = (H⁻¹)_tt z` — the `t`-block of
    /// `H⁻¹·[z; 0]`; the trailing `w_β = 0` drops the border coupling out of the
    /// `t`-block — so the Hadamard product `z ⊙ u_t` has expectation exactly
    /// `diag((H⁻¹)_tt)` (off-diagonal `i≠j` terms are mean-zero under
    /// `E[z_i z_j] = 0`). Averaging over probes gives the unbiased diagonal. Each
    /// probe is one [`ArrowFactorCache::full_inverse_apply`] (per-row solves, one
    /// Schur solve, and any gauge-deflation correction), so it applies the same
    /// assembled curvature operator as the exact path.
    ///
    /// Probes run serially and accumulate in a fixed order, so for a fixed
    /// `(seed, num_probes)` the estimate is bit-reproducible (the REML determinism
    /// contract, matching the SLQ log-det and smoothness-dof Hutchinson paths).
    pub(crate) fn latent_block_inverse_diagonal_hutchinson(
        cache: &ArrowFactorCache,
        num_probes: usize,
        seed: u64,
    ) -> Result<Array1<f64>, ArrowSchurError> {
        let total_len = cache.delta_t_len();
        let k = cache.k;
        let probes = num_probes.max(1);
        let mut out = Array1::<f64>::zeros(total_len);
        let mut z = Array1::<f64>::zeros(total_len);
        let w_beta_zero = Array1::<f64>::zeros(k);
        for probe in 0..probes {
            // Deterministic Rademacher probe (±1) over the t-block, seeded by
            // `seed + probe` so the whole estimate is reproducible.
            let mut state = seed.wrapping_add(probe as u64);
            let mut bits = 0u64;
            let mut remaining = 0u32;
            for zi in z.iter_mut() {
                if remaining == 0 {
                    bits = gam_linalg::utils::splitmix64(&mut state);
                    remaining = 64;
                }
                *zi = if bits & 1 == 1 { 1.0 } else { -1.0 };
                bits >>= 1;
                remaining -= 1;
            }
            // u_t = (H⁻¹)_tt z (w_β = 0 ⇒ the border coupling drops from the
            // t-block); this applies the full assembled inverse.
            let (u_t, _u_beta) = cache.full_inverse_apply(z.view(), w_beta_zero.view())?;
            for i in 0..total_len {
                out[i] += z[i] * u_t[i];
            }
        }
        let inv_p = 1.0 / (probes as f64);
        for v in out.iter_mut() {
            *v *= inv_p;
        }
        Ok(out)
    }

    pub(crate) fn ard_log_precision_explicit_derivatives(
        &self,
        rho: &SaeManifoldRho,
    ) -> Result<Vec<Array1<f64>>, String> {
        self.assignment.validate_rho_domain(rho)?;
        let ard_precisions = self.validated_ard_precisions(rho)?;
        let n = self.n_obs() as f64;
        // HT row weighting: this is the ρ-derivative of `ard_value` (the `explicit`
        // outer-gradient channel), so it carries the identical per-row inclusion
        // weight and effective row count `n_eff = Σᵢ wᵢ` as the energy — otherwise
        // the analytic gradient desyncs from the (now weight-aware) criterion value
        // on the subsample. `None` ⇒ `w_row = 1`, `n_eff = n`, historical exactly.
        let row_w = self.row_loss_weights.as_deref();
        let n_eff = row_w.map_or(n, |w| w.iter().sum::<f64>());
        let mut out = Vec::with_capacity(self.k_atoms());
        for (atom_idx, coord) in self.assignment.coords.iter().enumerate() {
            let d = coord.latent_dim();
            let mut atom_out = Array1::<f64>::zeros(rho.log_ard[atom_idx].len());
            if rho.log_ard[atom_idx].is_empty() {
                out.push(atom_out);
                continue;
            }
            let periods = self.ard_axis_periods(atom_idx);
            // The log partition over the coordinate's support; its derivative is
            // the normalizer channel of `∂ ard_value/∂log α`.
            let partition = Self::ard_log_partition(
                &coord.effective_prior_supports(),
                &periods,
                rho.log_ard[atom_idx].view(),
                ard_precisions[atom_idx].view(),
            )?;
            for axis in 0..d {
                let alpha = ard_precisions[atom_idx][axis];
                let period = periods[axis];
                let mut energy_deriv = 0.0_f64;
                for row in 0..coord.n_obs() {
                    let w_row = row_w.map_or(1.0, |w| w[row]);
                    let t = coord.row(row)[axis];
                    energy_deriv += w_row * ArdAxisPrior::eval(alpha, t, period).value;
                }
                atom_out[axis] = energy_deriv + n_eff * partition.log_precision_gradient[axis];
            }
            out.push(atom_out);
        }
        Ok(out)
    }

    /// `operator` (#2515) names WHICH curvature the `∂H/∂log α` operand is:
    /// `Majorizer` differentiates `B`'s PSD-clamped `α·s_{τ₀}(cos κt)`,
    /// `ExactObservedInformation` differentiates `A`'s unmajorized `α·cos κt`.
    /// It must agree with the operator whose inverse `solver` factors, which is
    /// why every trace entry point takes it explicitly rather than defaulting.
    pub(crate) fn ard_log_precision_hessian_trace(
        &self,
        rho: &SaeManifoldRho,
        cache: &ArrowFactorCache,
        solver: &DeflatedArrowSolver<'_>,
        operator: EvidenceOperator,
    ) -> Result<Vec<Array1<f64>>, ArrowSchurError> {
        self.assignment
            .validate_rho_domain(rho)
            .map_err(|reason| ArrowSchurError::SchurFactorFailed { reason })?;
        let ard_precisions = self
            .validated_ard_precisions(rho)
            .map_err(|reason| ArrowSchurError::SchurFactorFailed { reason })?;
        // RAW selected-inverse diagonal: the per-axis diagonal contraction uses
        // the DEFLATED inverse; the full kept-subspace + rotation deflation
        // correction `tr(inv_vv·(D − DΦ[D]))` is subtracted per (row, axis)
        // afterwards via the Daleckii–Krein helper. Each ARD ρ-component
        // `(atom k, axis)` differentiates a SINGLE coordinate-slot diagonal entry,
        // so its `D` is the rank-one `hess·e_s e_sᵀ` at that local slot `s`.
        let inv_diag = solver
            .latent_inverse_diagonal()
            .map_err(|err| ArrowSchurError::SchurFactorFailed { reason: err })?;
        // HT row weighting: the assembled per-row ARD curvature `∂H/∂logα` is scaled
        // by the inclusion weight `wᵢ` (see the assembly seam), and `inv_diag` = the
        // diagonal of `H⁻¹` already reflects that w-scaled `H`. So each row's trace
        // contribution `½·(H⁻¹)_ss·(wᵢ·hess)` must carry the SAME `wᵢ` here, or the
        // ½log|H| gradient desyncs from the assembled Hessian on the subsample.
        // `None` ⇒ `w_row = 1`, bit-for-bit the historical trace.
        let row_w = self.row_loss_weights.as_deref();
        let n = self.n_obs();
        let total_t = cache.delta_t_len();
        let coord_offsets = self.assignment.coord_offsets();
        let ard_axis_periods: Vec<Vec<Option<f64>>> = self.all_ard_axis_periods();
        let mut traces: Vec<Array1<f64>> = self
            .assignment
            .coords
            .iter()
            .enumerate()
            .map(|(k, c)| {
                if rho.log_ard[k].is_empty() {
                    Array1::<f64>::zeros(0)
                } else {
                    Array1::<f64>::zeros(c.latent_dim())
                }
            })
            .collect();
        // Hoisted RHS scratch reused across every (row, col) solve. Setting and
        // clearing a SINGLE entry per column is O(1); a fresh
        // `Array1::zeros(total_t)` memsets total_t≈n·q slots per inner iteration
        // (O(n) per col ⇒ O(n²) redundant zeroing across the block build).
        let mut rhs_t_scratch = Array1::<f64>::zeros(total_t);
        let rhs_beta_zero = Array1::<f64>::zeros(cache.k);
        for row in 0..n {
            let w_row = row_w.map_or(1.0, |w| w[row]);
            let row_base = cache.row_offsets[row];
            let q = cache.row_dims[row];
            let dirs = cache
                .deflated_row_directions
                .get(row)
                .map(Vec::as_slice)
                .unwrap_or(&[]);
            let spectrum = cache
                .deflation_row_spectra
                .get(row)
                .and_then(Option::as_ref);
            // Per-row selected-inverse t-block, built once (only when deflated).
            let inv_vv = if !Self::row_deflation_is_live(dirs, spectrum) {
                None
            } else {
                let mut m = Array2::<f64>::zeros((q, q));
                for col in 0..q {
                    rhs_t_scratch[row_base + col] = 1.0;
                    let solved = solver
                        .solve(rhs_t_scratch.view(), rhs_beta_zero.view())
                        .map_err(|err| ArrowSchurError::SchurFactorFailed { reason: err })?;
                    rhs_t_scratch[row_base + col] = 0.0;
                    for r in 0..q {
                        m[[r, col]] = solved.t[row_base + r];
                    }
                }
                Some(m)
            };
            // Correction for one local coordinate slot `s` with curvature `hess`.
            let slot_correction = |s: usize, hess: f64| -> f64 {
                let Some(iv) = inv_vv.as_ref() else {
                    return 0.0;
                };
                if s >= q || hess == 0.0 {
                    return 0.0;
                }
                let mut d = Array2::<f64>::zeros((q, q));
                d[[s, s]] = hess;
                Self::deflation_block_correction(iv, &d, dirs, spectrum)
            };
            match self.last_row_layout {
                Some(ref layout) => {
                    let active = &layout.active_atoms[row];
                    let starts = &layout.coord_starts[row];
                    for (pos, &k) in active.iter().enumerate() {
                        if rho.log_ard[k].is_empty() {
                            continue;
                        }
                        let coord = &self.assignment.coords[k];
                        let d = coord.latent_dim();
                        let block_start = starts[pos];
                        for axis in 0..d {
                            let alpha = ard_precisions[k][axis];
                            let t = coord.row(row)[axis];
                            let prior = ArdAxisPrior::eval(alpha, t, ard_axis_periods[k][axis]);
                            let hess = w_row * prior.log_precision_curvature(operator);
                            let s = block_start + axis;
                            traces[k][axis] += 0.5 * inv_diag[row_base + s] * hess;
                            traces[k][axis] -= 0.5 * slot_correction(s, hess);
                        }
                    }
                }
                None => {
                    for k in 0..self.k_atoms() {
                        if rho.log_ard[k].is_empty() {
                            continue;
                        }
                        let coord = &self.assignment.coords[k];
                        let d = coord.latent_dim();
                        let block_start = coord_offsets[k];
                        for axis in 0..d {
                            let alpha = ard_precisions[k][axis];
                            let t = coord.row(row)[axis];
                            let prior = ArdAxisPrior::eval(alpha, t, ard_axis_periods[k][axis]);
                            let hess = w_row * prior.log_precision_curvature(operator);
                            let s = block_start + axis;
                            traces[k][axis] += 0.5 * inv_diag[row_base + s] * hess;
                            traces[k][axis] -= 0.5 * slot_correction(s, hess);
                        }
                    }
                }
            }
        }
        Ok(traces)
    }

    /// Per-atom, per-axis `½ tr(H⁻¹ ∂H/∂logα_{kj})` — the ARD ½log|H| ρ-gradient
    /// channel [`Self::ard_log_precision_hessian_trace`] computes — from the #2080
    /// SHARED selected-inverse bundle instead of the dense `DeflatedArrowSolver`
    /// (`latent_inverse_diagonal` + per-column `solve`). The massive-lane /
    /// eventual-dense-cache-retirement replacement for the last dense `S⁻¹` in the
    /// analytic outer ρ-gradient's ARD block.
    ///
    /// Each ARD component differentiates ONE coordinate-slot diagonal entry, so the
    /// trace is `Σ_{row, slot s(k,j)} ½·(H⁻¹)_tt[s,s]·(w_row·hess)`. The diagonal is
    /// the SAME per-row arrow selected-inverse the ARD posterior-variance trace uses
    /// (`(A_i⁻¹)[s,s]` row-local + border `(1/m)Σ_j w_ij[s]·s_ij[s]` off the bundle,
    /// see [`Self::ard_inverse_traces_from_probes`] for the derivation), matching
    /// `solver.latent_inverse_diagonal()` on the PLAIN (undeflated) selected inverse
    /// to solve precision.
    ///
    /// # Deflation is priced, not refused (#2712)
    ///
    /// The dense path's Daleckii–Krein correction `−½ tr(inv_vv·(D − DΦ[D]))`
    /// needs the DEFLATED per-row inverse block, and this lane has it: `A_i` is
    /// `cache.undamped_factor(i)`, the Cholesky of the spectrally CONDITIONED
    /// `Φ(H_tt^(i))`, and the reduced Schur behind the bundle is that same
    /// conditioned arrow's — so `A_i⁻¹ + G_i S⁻¹ G_iᵀ` IS the deflated block
    /// ([`row_selected_inverse_from_probes`]). This lane used to hard-refuse
    /// deflated rows on the stated grounds that the reconstruction was the
    /// UNdeflated block; that was a misreading of `undamped_factor`, and the
    /// correction's remaining operands (`deflated_row_directions`,
    /// `deflation_row_spectra`, and the per-slot diagonal `D` built below) never
    /// involved `S⁻¹` at all. Deflated and undeflated rows now take the same
    /// route, and the correction is identically zero on a PD row.
    // #2080 analytic-gradient cluster channel: wired into
    // `analytic_outer_rho_gradient_components_with_bundle`'s `Some`-bundle branch
    // (the all-or-nothing selected-inverse cluster, alongside the from-probes
    // smoothness EDF), dormant until the analytic-gradient routing flips (every
    // caller passes `None` today). That real call site is the non-test consumer,
    // so this matches its `pub(crate)` sibling `ard_inverse_traces_from_probes`.
    pub(crate) fn ard_log_precision_hessian_trace_from_probes(
        &self,
        rho: &SaeManifoldRho,
        cache: &ArrowFactorCache,
        probes: &[Array1<f64>],
        sinv_probes: &[Array1<f64>],
        operator: EvidenceOperator,
    ) -> Result<Vec<Array1<f64>>, ArrowSchurError> {
        self.assignment
            .validate_rho_domain(rho)
            .map_err(|reason| ArrowSchurError::SchurFactorFailed { reason })?;
        let ard_precisions = self
            .validated_ard_precisions(rho)
            .map_err(|reason| ArrowSchurError::SchurFactorFailed { reason })?;
        let m = probes.len();
        if m == 0 || sinv_probes.len() != m {
            return Err(ArrowSchurError::SchurFactorFailed {
                reason: format!(
                    "ard_log_precision_hessian_trace_from_probes: need matching non-empty \
                     probe/solve bundles, got {m} probes and {} solves",
                    sinv_probes.len()
                ),
            });
        }
        let k_border = cache.k;
        for (label, set) in [("probe", probes), ("solve", sinv_probes)] {
            for (j, v) in set.iter().enumerate() {
                if v.len() != k_border {
                    return Err(ArrowSchurError::SchurFactorFailed {
                        reason: format!(
                            "ard_log_precision_hessian_trace_from_probes: {label} {j} has length \
                             {} != border dim {k_border}",
                            v.len()
                        ),
                    });
                }
            }
        }
        let row_w = self.row_loss_weights.as_deref();
        let n = self.n_obs();
        let coord_offsets = self.assignment.coord_offsets();
        let ard_axis_periods: Vec<Vec<Option<f64>>> = self.all_ard_axis_periods();
        let mut traces: Vec<Array1<f64>> = self
            .assignment
            .coords
            .iter()
            .enumerate()
            .map(|(k, c)| {
                if rho.log_ard[k].is_empty() {
                    Array1::<f64>::zeros(0)
                } else {
                    Array1::<f64>::zeros(c.latent_dim())
                }
            })
            .collect();
        // #2915 — a clamp-basin price moves with the clamp itself, which the exact
        // operator's `α·cos κt` does not carry.
        let clamp = if operator.is_exact_a() {
            Some(
                self.materialize_ard_concave_clamp_diagonal_for_rows(rho, &cache.row_dims)
                    .map_err(|reason| ArrowSchurError::SchurFactorFailed { reason })?,
            )
        } else {
            None
        };
        // #2915 — so does a reduced-Schur clamp-basin price.
        // The border clamp is the decoder priors' remainder at the unit penalty scale of
        // the full evidence system this lane factors.
        let border_remainder = if clamp.is_some() && cache.beta_schur_conditioning.is_some() {
            self.decoder_prior_border_remainder_op(cache.k, 1.0)
                .map_err(|reason| ArrowSchurError::SchurFactorFailed { reason })?
        } else {
            None
        };
        let beta_price = match clamp.as_ref() {
            Some(clamp) => Self::beta_schur_clamp_basin_price_weights(
                cache,
                clamp.view(),
                border_remainder.as_ref().map(|op| op as &dyn BetaPenaltyOp),
            )
            .map_err(|reason| ArrowSchurError::SchurFactorFailed { reason })?,
            None => None,
        };
        for row in 0..n {
            let w_row = row_w.map_or(1.0, |w| w[row]);
            let q = cache.row_dims[row];
            // The DEFLATED row-block selected inverse from the shared bundle
            // (#2712). The `t–β` block is not contracted here, so it is not built.
            let (mut inv_vv, _) = row_selected_inverse_from_probes(
                cache,
                row,
                probes,
                sinv_probes,
                false,
                "ard_log_precision_hessian_trace_from_probes",
            )
            .map_err(|reason| ArrowSchurError::SchurFactorFailed { reason })?;
            if let Some(weights) = beta_price.as_ref() {
                inv_vv += &weights[row].0;
            }
            let inv_diag_local = inv_vv.diag().to_owned();
            let dirs = cache
                .deflated_row_directions
                .get(row)
                .map(Vec::as_slice)
                .unwrap_or(&[]);
            let spectrum = cache
                .deflation_row_spectra
                .get(row)
                .and_then(Option::as_ref);
            let row_base = cache.row_offsets[row];
            let price = clamp.as_ref().and_then(|clamp| {
                Self::clamp_basin_price_weights(
                    &inv_vv,
                    clamp.slice(ndarray::s![row_base..row_base + q]),
                    spectrum,
                )
            });
            // Correction for one local coordinate slot `s` with curvature `hess`,
            // identical to the dense sibling's `slot_correction`.
            let slot_correction = |s: usize, hess: f64| -> f64 {
                if !Self::row_deflation_is_live(dirs, spectrum) || s >= q || hess == 0.0 {
                    return 0.0;
                }
                let mut d = Array2::<f64>::zeros((q, q));
                d[[s, s]] = hess;
                Self::deflation_block_correction(&inv_vv, &d, dirs, spectrum)
            };
            let accumulate = |k: usize, block_start: usize, traces: &mut Vec<Array1<f64>>| {
                if rho.log_ard[k].is_empty() {
                    return;
                }
                let coord = &self.assignment.coords[k];
                let d = coord.latent_dim();
                for axis in 0..d {
                    let alpha = ard_precisions[k][axis];
                    let t = coord.row(row)[axis];
                    let prior = ArdAxisPrior::eval(alpha, t, ard_axis_periods[k][axis]);
                    let hess = w_row * prior.log_precision_curvature(operator);
                    let s = block_start + axis;
                    traces[k][axis] += 0.5 * inv_diag_local[s] * hess;
                    traces[k][axis] -= 0.5 * slot_correction(s, hess);
                    if let (Some((explicit, response)), Some(clamp)) = (price.as_ref(), clamp.as_ref())
                    {
                        if s < q {
                            // `∂E/∂ρ_ard` at slot `s` is the ARD clamp itself: degree one in `α`.
                            traces[k][axis] += 0.5
                                * (explicit[s] * clamp[row_base + s] + response[[s, s]] * hess);
                        }
                    }
                    if let (Some(weights), Some(clamp)) = (beta_price.as_ref(), clamp.as_ref()) {
                        if s < q {
                            // The reduced-Schur basin prices read the same ARD clamp.
                            traces[k][axis] += 0.5 * weights[row].1[s] * clamp[row_base + s];
                        }
                    }
                }
            };
            match self.last_row_layout {
                Some(ref layout) => {
                    let active = &layout.active_atoms[row];
                    let starts = &layout.coord_starts[row];
                    for (pos, &k) in active.iter().enumerate() {
                        accumulate(k, starts[pos], &mut traces);
                    }
                }
                None => {
                    for k in 0..self.k_atoms() {
                        accumulate(k, coord_offsets[k], &mut traces);
                    }
                }
            }
        }
        Ok(traces)
    }
}
