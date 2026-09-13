// [#780] Exact stationarity-Jacobian correction (`apply_exact_hessian_minus_b`),
// the exact inner-fit Hessian apply (`apply_exact_hessian`), and the exact
// stationarity solve (`solve_exact_stationarity`) were extracted verbatim from
// `construction.rs` into this sibling file to keep that file under the #780
// per-file line-count gate. It is `include!`d back into the parent module in
// `construction.rs`, so these methods share that module's scope exactly as
// before (same `impl SaeManifoldTerm`, same `use super::*` imports).

/// ONE identifiability floor for directions of the exact observed information
/// `A = B + ΔC`, in ONE metric, for the value path and the gradient path alike
/// (#2673, #2080 defect 4, #2253 x #2330).
///
/// `B` is the positive-definite arrow factorization: the scale the inner Newton
/// solve, the IFT solve and the evidence factor are all expressed in. The
/// generalized Rayleigh quotient
///
/// ```text
///   μ(v) = vᵀAv / vᵀBv
/// ```
///
/// therefore measures a direction's exact curvature against the problem's own
/// scale. The floor is `√ε_machine`, the standard boundary below which a
/// double-precision curvature ratio is not numerically identifiable; it is
/// derived from the scalar type, not tuned to a fixture. A direction below it
/// (a saturated ordered Beta--Bernoulli gate logit has data curvature
/// `∝ σ'(ℓ)² → 0`) is numerically curvature-free: the inner optimizer cannot
/// resolve the iterate's position along it, so the IFT response
/// `θ̂_ρ = −A⁻¹g_ρ` there is an unidentifiable `1/μ` amplification rather than a
/// derivative. That amplification is what flipped the analytic λ-gradient's sign
/// against the criterion it differentiates (the #931 objective↔gradient desync).
///
/// # Why `μ` and not an absolute eigenvalue of `A` (#2673)
///
/// Until this became the single floor, the VALUE path classified the same
/// directions of the same `A` by `|λ| ≤ 1e-9 · max(λ_max(A), 1)` while the
/// GRADIENT path used `|μ| < √ε`. Both floors ran inside ONE evaluation on the
/// streaming route — the value path's terminal Newton polish reaches
/// `solve_exact_stationarity`, which is not route-gated, while the gradient
/// adjoint takes the matrix-free sibling — and which of them the value path used
/// was decided by `direct_logdet_admitted`, i.e. by ambient free memory. Three
/// things were wrong with the pair, and none of them needed a fixture to
/// witness:
///
/// 1. **They are not one rule in two spellings.** Written as thresholds on the
///    same `|λ|`, the value rule is ONE number for every direction and the
///    gradient rule is `√ε·vᵀBv`, which varies across directions of one
///    operator by the spread of the `B`-Rayleigh quotient (measured 24x on the
///    #2515 route-invariance state, unbounded in general). No choice of the two
///    constants makes a constant threshold equal a varying one, and on that same
///    state the ratio STRADDLES 1 — so neither rule was even a conservative
///    version of the other.
/// 2. **Only `μ` is a curvature.** `μ` is invariant under a reparametrization
///    `θ → Lθ` (`A → LᵀAL`, `B → LᵀBL` transform congruently); `λ/λ_max(A)` is
///    not. `θ = (t, β)` mixes chart coordinates with border coefficients whose
///    scale is set by the data's units, so `λ_max(A)` is a maximum over
///    incommensurable coordinates and the band it defined moved when the units
///    did.
/// 3. **`1e-9` was tuned; `√ε` is derived.** SPEC rule 21.
///
/// Consistency then FORCES the direction of the unification: the value must pin
/// exactly what the gradient cannot differentiate, or the criterion depends on
/// `ρ` through a direction whose response the adjoint has projected out — the
/// #931/#2253 desync in a new place. So the larger, derived, invariant floor
/// wins at both sites and the absolute one is gone.
///
/// Classification is per ordinary eigendirection of A. Dense and matrix-free
/// solves both use Euclidean spectral projections and the same direction floor;
/// the matrix-free solve lifts Ritz vectors to measure their B quadratic forms.
/// An aggregate Rayleigh quotient of a solution is not a null-space policy.
pub(crate) fn sae_exact_a_identifiability_floor() -> f64 {
    f64::EPSILON.sqrt()
}

/// The null-band half-width, in `λ` units, for ONE direction of a dense
/// exact-`A` block (#2673).
///
/// ```text
///   floor = max( spectral_dim·ε·‖A‖₂ ,  √ε · vᵀBv )
/// ```
///
/// This is the whole rule, as a scalar function of three measured numbers, so
/// that every consumer applies the SAME rule while supplying its OWN operands.
/// Production reaches it through [`ExactHessianSpectralBlock::rank_floor`]; the
/// independent oracles that re-derive the classification from a separately built
/// dense `A` call it directly, which keeps them oracles (their inputs are their
/// own) without letting a second copy of the predicate drift from this one — the
/// failure mode #2740 names and the one this issue is.
///
/// See [`sae_exact_a_identifiability_floor`] for why the metric is `B` and why
/// the second term is the one that classifies.
pub(crate) fn sae_exact_a_direction_floor(
    spectral_dim: usize,
    spectral_norm: f64,
    b_quadratic_form: f64,
) -> f64 {
    gam_solve::arrow_schur::exact_a_direction_floor(spectral_dim, spectral_norm, b_quadratic_form)
}

/// The null-band edge on the side of one direction's curvature, for the dense and
/// the matrix-free exact-`A` routes (#2267).
///
/// ```text
///   λ ≤ 0:  floor
///   λ > 0:  max( floor ,  s ),     s = vᵀ(Φ(B_raw) − B_raw)v
/// ```
///
/// `floor` is [`sae_exact_a_direction_floor`], denominated in the CONDITIONED
/// evidence metric `Φ(B_raw)`. Where the majorizer has no resolved curvature the
/// evidence factor pins a direction at unit stiffness (`log 1 = 0`), so along it
/// `Φ(B_raw)` carries a stiffness `s` that no term of the objective supplies, and
/// `√ε·vᵀΦ(B_raw)v` compares `λ` against `√ε` in the pin's own units. A positive
/// direction whose exact curvature does not exceed `s` is resolved by the
/// substitution alone, so the value prices it the way the factor prices the pin.
/// At a unit pin the edge is `λ ≈ 1`, where `½·ln λ ≈ 0`: the price is continuous
/// across it, where the bare floor put a step of `½·ln √ε ≈ −9` per direction.
///
/// Pool job 598561 (`sae_manifold_euclidean_k2_fit_terminates`) read that step. At
/// one ρ, two evaluations whose loss differed by 1.8e-6 priced `½log|A|` at
/// −3.630e3 (all 996 directions retained, the smallest at 1.457× its floor) and at
/// +5.859e2 (480 in band, the largest at 0.614×), and the outer search's halvings
/// compared values from both strata.
///
/// The negative side keeps the bare floor. A resolved negative direction is a basin
/// or saddle verdict (#2330/#2336) that the #2080 descent reads, and a pin does not
/// turn it into a null.
pub(crate) fn sae_exact_a_band_edge(
    curvature: f64,
    spectral_dim: usize,
    spectral_norm: f64,
    b_quadratic_form: f64,
    substituted_stiffness: f64,
) -> f64 {
    let floor = sae_exact_a_direction_floor(spectral_dim, spectral_norm, b_quadratic_form);
    if curvature > 0.0 {
        floor.max(substituted_stiffness)
    } else {
        floor
    }
}

/// One row's assembled `ΔC = A − B` blocks, in the arrow layout the streaming
/// evidence system already uses (`ArrowRowBlock::{htt, htbeta}`).
#[derive(Debug, Clone)]
pub(crate) struct ExactHessianDeltaRow {
    /// `ΔC_tt^(i)`, shape `(q_i, q_i)`.
    pub(crate) tt: Array2<f64>,
    /// `ΔC_tβ^(i)`, shape `(q_i, border_dim)`.
    pub(crate) tbeta: Array2<f64>,
}

/// Rank-revealing spectral representation of one dense exact-stationarity
/// block.  The materialized operator and its eigensystem remain together so a
/// pseudo-inverse response can be certified against the physical operator that
/// produced it, rather than against a projected Krylov surrogate.
struct ExactHessianSpectralBlock {
    operator: Array2<f64>,
    eigenvalues: Array1<f64>,
    /// Ambient eigenvectors, SQUARE: every direction of the materialized
    /// operator is classified by `rank_floor` and nothing is deleted ahead of
    /// it.  #2674 — the analytic chart orbit used to be quotiented out here
    /// before diagonalization, which deleted directions the penalized operator
    /// has genuine curvature and genuine slope in.
    eigenvectors: Array2<f64>,
    /// `vᵢᵀBvᵢ` for every eigendirection — the scale the classification is
    /// relative to (#2673). `B` is the arrow factorization's own operator
    /// restricted to this block's coordinates, so every entry is strictly
    /// positive and the ratio `λᵢ / metric_scale[i]` is the pencil curvature the
    /// gradient path classifies the same directions by.
    metric_scale: Array1<f64>,
    /// `vᵢᵀ(Φ(B_raw) − B_raw)vᵢ` for every eigendirection: the part of
    /// `metric_scale` the evidence factor substituted rather than measured. It sets
    /// the positive band edge; see [`sae_exact_a_band_edge`] (#2267).
    substituted_stiffness: Array1<f64>,
    /// `‖A‖₂ = maxᵢ|λᵢ|`, the scale the eigendecomposition's own backward error
    /// is proportional to.
    spectral_norm: f64,
}

/// The `B` metric one spectral block is classified in (#2673).
///
/// `B` is the arrow factorization that the inner Newton solve, the IFT solve and
/// the evidence factor are all expressed in, and the ONE thing a direction's
/// curvature is measured against at both the value and the gradient site. Which
/// restriction of it applies is decided by which block of `A` is being
/// classified, so the two cannot be paired up wrongly:
///
/// * the JOINT block is `A` itself, so its metric is the whole arrow operator,
///   border and all.
///
/// Both are applies through the cached factors, never a materialized `B`: the
/// dense route already carries one `dim × dim` block and #2724/#2757 price that
/// memory, so a second one would be paid for a scalar per direction.
#[derive(Clone, Copy)]
pub(crate) enum ArrowMetric<'a> {
    /// `B` on the joint `(t, β)` coordinates.
    Joint(&'a ArrowFactorCache),
}

impl ArrowMetric<'_> {
    pub(crate) fn quadratic_form(&self, v: ArrayView1<'_, f64>) -> Result<f64, String> {
        Ok(v.dot(&self.apply(v)?))
    }

    /// `vᵀ(Φ(B_raw) − B_raw)v`: the part of [`Self::quadratic_form`] the evidence
    /// factor substituted rather than measured, read off the row spectra
    /// `add_raw_row_deflation_correction` restores `B_raw` from (#2267). Zero on
    /// every row the factor kept raw; the border block is raw in both applies, so
    /// only `t` enters. A direction whose pins lower curvature more than they raise
    /// it carries no substituted stiffness, so the total is clamped at zero.
    pub(crate) fn substituted_stiffness(&self, v: ArrayView1<'_, f64>) -> Result<f64, String> {
        match self {
            Self::Joint(cache) => {
                let total_t = cache.delta_t_len();
                if v.len() != total_t + cache.k {
                    return Err(format!(
                        "ArrowMetric::Joint: direction length {} != joint dimension {}",
                        v.len(),
                        total_t + cache.k
                    ));
                }
                let v_t = v.slice(s![..total_t]);
                let mut raw_minus_conditioned = Array1::<f64>::zeros(total_t);
                add_raw_row_deflation_correction(
                    cache,
                    v_t,
                    raw_minus_conditioned.view_mut(),
                    "ArrowMetric::substituted_stiffness",
                )?;
                Ok((-v_t.dot(&raw_minus_conditioned)).max(0.0))
            }
        }
    }

    fn apply(&self, v: ArrayView1<'_, f64>) -> Result<Array1<f64>, String> {
        match self {
            Self::Joint(cache) => {
                let total_t = cache.delta_t_len();
                if v.len() != total_t + cache.k {
                    return Err(format!(
                        "ArrowMetric::Joint: direction length {} != joint dimension {}",
                        v.len(),
                        total_t + cache.k
                    ));
                }
                let b_v = apply_cached_arrow_hessian(
                    cache,
                    v.slice(s![..total_t]),
                    v.slice(s![total_t..]),
                )?;
                Ok(Array1::from_iter(b_v.t.iter().chain(b_v.beta.iter()).copied()))
            }
        }
    }
}

/// One point of the Levenberg--Marquardt path of the LINEAR residual model,
/// read off the eigensystem that is already materialized.
///
/// For damping `ν ≥ 0` the step
///
/// ```text
///   Δ(ν) = Σ_i u_i λ_i (u_iᵀ rhs) / (λ_i² + ν)
/// ```
///
/// is the exact minimizer of `‖rhs − AΔ‖² + ν‖Δ‖²`, and `ν = 0` reproduces the
/// pseudoinverse step of [`ExactHessianSpectralBlock::solve_stationarity`]
/// (same `rank_floor`, same retained band). Because the eigensystem is already
/// in hand, the WHOLE path costs one diagonal pass per point — no
/// refactorization, no second operator apply.
///
/// The caller prices the trial point in the currency its convergence gate
/// owns (#2080: the penalized objective); this lower-level spectral object
/// deliberately does not attach an ambient or quotient scalar merit to it.
pub(crate) struct DampedResidualStep {
    /// `Δ(ν)`.
    pub(crate) step: SaeArrowVector,
    /// `‖Δ(ν)‖²`.
    pub(crate) step_norm_sq: f64,
    /// Directions whose damped denominator cleared the null band.
    pub(crate) retained_rank: usize,
    /// `‖g‖²` carried by the directions inside the null band. The step moves
    /// nothing along them, so this is the part of the stationarity residual that
    /// no step of this operator can reduce, at any damping.
    pub(crate) excluded_gradient_norm_sq: f64,
}

impl ExactHessianSpectralBlock {
    /// The null-band half-width for eigendirection `index`, in `λ` units
    /// (#2673).
    ///
    /// ```text
    ///   floor(i) = max( dim·ε·‖A‖₂ ,  √ε · vᵢᵀBvᵢ )
    /// ```
    ///
    /// The second term is [`sae_exact_a_identifiability_floor`] — the SAME
    /// predicate the matrix-free gradient path applies to its own solution
    /// direction, written in `λ` units so this site can compare it against an
    /// eigenvalue. It is the term that decides the classification: a direction
    /// under it is one whose `A⁻¹` response is an unidentifiable `1/μ`
    /// amplification, so the value must not price a `ρ`-dependence there that
    /// the adjoint has projected out.
    ///
    /// The first term is not a second classification. It is the standard
    /// backward-error bound for a symmetric eigendecomposition — the computed
    /// spectrum of a perturbed `A + E` with `‖E‖₂ ≲ p(dim)·ε·‖A‖₂` — so an
    /// eigenvalue below it carries no significant digits and `ln λ` is not a
    /// quantity. The matrix-free Ritz solve uses the same arithmetic floor.
    /// It is a floor under the identifiability term, never a ceiling, so it can
    /// only pin directions, never resurrect one the gradient has deflated.
    ///
    /// #2267 — on the positive side the edge rises to the stiffness the evidence
    /// factor substituted along the direction, where that exceeds this floor; see
    /// [`sae_exact_a_band_edge`]. The value, the differential, the polish and the
    /// solves all read the band through this one function, so they move together.
    fn rank_floor(&self, index: usize) -> f64 {
        sae_exact_a_band_edge(
            self.eigenvalues[index],
            self.eigenvalues.len(),
            self.spectral_norm,
            self.metric_scale[index],
            self.substituted_stiffness[index],
        )
    }

    /// Directions discarded by arithmetic resolution despite a resolved
    /// identifiability ratio. Both adjoint routes discard these directions;
    /// report them because they indicate that arithmetic sets the rank.
    fn arithmetic_band_crossings(&self) -> usize {
        let arithmetic = (self.eigenvalues.len() as f64) * f64::EPSILON * self.spectral_norm;
        let identifiability_floor = sae_exact_a_identifiability_floor();
        (0..self.eigenvalues.len())
            .filter(|&index| {
                let lambda = self.eigenvalues[index];
                lambda.abs() <= arithmetic
                    && lambda.abs() >= identifiability_floor * self.metric_scale[index]
            })
            .count()
    }

    /// Smallest and largest `|λ|` the null band retained, or `None` when the
    /// whole spectrum is inside it. The two set the DERIVED damping ladder the
    /// polish walks: below `λ_min²` a damping cannot change the flattest
    /// resolved direction, and above `λ_max²` it has already flattened every
    /// direction there is, so no ladder needs to leave `[λ_min², λ_max²]`.
    fn retained_curvature_extremes(&self) -> Option<(f64, f64)> {
        let mut smallest = f64::INFINITY;
        let mut largest = 0.0_f64;
        for (index, &lambda) in self.eigenvalues.iter().enumerate() {
            let magnitude = lambda.abs();
            if magnitude > self.rank_floor(index) {
                smallest = smallest.min(magnitude);
                largest = largest.max(magnitude);
            }
        }
        (largest > 0.0 && smallest.is_finite()).then_some((smallest, largest))
    }

    /// A spectrally scaled descent step for the scalar objective whose gradient
    /// is `residual`.  This uses `|A|` rather than `A`: every retained
    /// component therefore has negative directional derivative even when the
    /// stationarity operator is indefinite.  `nu` has the same
    /// squared-curvature units as the damping ladder.
    fn damped_objective_step(
        &self,
        residual: &SaeArrowVector,
        nu: f64,
    ) -> Result<DampedResidualStep, String> {
        let total_t = residual.t.len();
        let dim = total_t + residual.beta.len();
        if self.eigenvectors.dim() != (dim, dim) || self.eigenvalues.len() != dim {
            return Err("damped objective step: geometry and residual dimensions differ".into());
        }
        if !(nu.is_finite() && nu >= 0.0) {
            return Err(format!(
                "damped objective step: damping must be finite and >= 0; got {nu}"
            ));
        }
        let mut flat = Array1::<f64>::zeros(dim);
        flat.slice_mut(s![..total_t]).assign(&residual.t);
        flat.slice_mut(s![total_t..]).assign(&residual.beta);
        if !flat.iter().all(|value| value.is_finite()) {
            return Err("damped objective step: residual contains a non-finite value".into());
        }
        let coefficients = self.eigenvectors.t().dot(&flat);
        let mut step_coefficients = Array1::<f64>::zeros(dim);
        let mut retained_rank = 0;
        let mut excluded_gradient_norm_sq = 0.0_f64;
        for index in 0..dim {
            let magnitude = self.eigenvalues[index].abs();
            if magnitude > self.rank_floor(index) {
                step_coefficients[index] = -coefficients[index] / (magnitude + nu.sqrt());
                retained_rank += 1;
            } else {
                excluded_gradient_norm_sq += coefficients[index] * coefficients[index];
            }
        }
        let solution = self.eigenvectors.dot(&step_coefficients);
        Ok(DampedResidualStep {
            step: SaeArrowVector {
                t: solution.slice(s![..total_t]).to_owned(),
                beta: solution.slice(s![total_t..]).to_owned(),
            },
            step_norm_sq: solution.dot(&solution),
            retained_rank,
            excluded_gradient_norm_sq,
        })
    }

    /// Apply the symmetric Moore--Penrose inverse.  Resolved positive and
    /// negative modes are both retained; only the spectral null band
    /// `|λ| ≤ rank_floor` is removed, and that band is the ONLY null predicate
    /// on this route (#2674).  Three independent certificates guard the result:
    /// physical backward residual on the retained range, least-squares
    /// stationarity for the retained operator, and minimum-norm membership in the
    /// retained range.
    fn solve_stationarity(&self, rhs: &SaeArrowVector) -> Result<SaeArrowVector, String> {
        let total_t = rhs.t.len();
        let dim = total_t + rhs.beta.len();
        let spectral_dim = self.eigenvalues.len();
        if self.operator.dim() != (dim, dim)
            || self.eigenvectors.dim() != (dim, spectral_dim)
            || spectral_dim != dim
        {
            return Err(format!(
                "dense exact-stationarity pseudoinverse: geometry dimension {:?}, spectrum {}, \
                 eigenvectors {:?}, but RHS dimension is {dim}",
                self.operator.dim(),
                spectral_dim,
                self.eigenvectors.dim(),
            ));
        }
        let mut flat_rhs = Array1::<f64>::zeros(dim);
        flat_rhs.slice_mut(s![..total_t]).assign(&rhs.t);
        flat_rhs.slice_mut(s![total_t..]).assign(&rhs.beta);
        if !flat_rhs.iter().all(|value| value.is_finite()) {
            return Err(
                "dense exact-stationarity pseudoinverse: RHS contains a non-finite value"
                    .to_string(),
            );
        }

        let coefficients = self.eigenvectors.t().dot(&flat_rhs);
        let mut projected_coefficients = coefficients.clone();
        let mut inverse_coefficients = Array1::<f64>::zeros(spectral_dim);
        let mut retained_rank = 0usize;
        for index in 0..spectral_dim {
            let lambda = self.eigenvalues[index];
            if lambda.abs() > self.rank_floor(index) {
                inverse_coefficients[index] = coefficients[index] / lambda;
                retained_rank += 1;
            } else {
                projected_coefficients[index] = 0.0;
            }
        }
        let solution = self.eigenvectors.dot(&inverse_coefficients);
        let projected_rhs = self.eigenvectors.dot(&projected_coefficients);
        let applied = self.operator.dot(&solution);
        // Range stationarity is `P_range A x = P_range rhs`, with `P_range` the
        // projector onto the eigendirections the SPECTRAL floor retained.  The
        // eigenbasis is complete, so this reprojection removes exactly the
        // measured null band and nothing that was declared null in advance.
        let applied_coefficients = self.eigenvectors.t().dot(&applied);
        let projected_applied = self.eigenvectors.dot(&applied_coefficients);
        let physical_residual = &projected_applied - &projected_rhs;
        // The normal equations belong to the truncated operator too. Using
        // the discarded RHS coefficients here rejects a pure numerical-null
        // RHS even though its declared pseudoinverse response is exactly zero.
        let residual_coefficients = &applied_coefficients - &projected_coefficients;
        let normal_coefficients = &self.eigenvalues * &residual_coefficients;
        let normal_residual = self.eigenvectors.dot(&normal_coefficients);

        let norm = |vector: &Array1<f64>| vector.dot(vector).max(0.0).sqrt();
        let operator_norm = self
            .eigenvalues
            .iter()
            .map(|value| value.abs())
            .fold(0.0_f64, f64::max);
        let solution_norm = norm(&solution);
        let projected_rhs_norm = norm(&projected_rhs);
        let physical_norm = norm(&physical_residual);
        let normal_norm = norm(&normal_residual);
        let physical_scale = operator_norm * solution_norm + projected_rhs_norm;
        let normal_scale = operator_norm * (operator_norm * solution_norm + projected_rhs_norm);

        // A Moore--Penrose solution must be orthogonal to the declared null
        // space. Reproject the computed physical vector (not the coefficients
        // used to construct it) so this gate also detects loss of orthogonality.
        let solved_coefficients = self.eigenvectors.t().dot(&solution);
        let spectral_null_solution_norm_sq = solved_coefficients
            .iter()
            .enumerate()
            .filter_map(|(index, coefficient)| {
                (self.eigenvalues[index].abs() <= self.rank_floor(index))
                    .then_some(coefficient * coefficient)
            })
            .sum::<f64>();
        let discarded_solution_norm = spectral_null_solution_norm_sq.max(0.0).sqrt();
        let tolerance = f64::EPSILON.sqrt();
        let within = |residual: f64, scale: f64| {
            residual == 0.0 || (scale > 0.0 && residual <= tolerance * scale)
        };
        if !solution.iter().all(|value| value.is_finite())
            || !within(physical_norm, physical_scale)
            || !within(normal_norm, normal_scale)
            || !within(discarded_solution_norm, solution_norm)
        {
            return Err(format!(
                "dense exact-stationarity pseudoinverse failed certification: \
                 physical residual {physical_norm:.6e} / backward scale {physical_scale:.6e}, \
                 normal-equation stationarity {normal_norm:.6e} / scale {normal_scale:.6e}, \
                 null-space solution mass {discarded_solution_norm:.6e} / solution norm \
                 {solution_norm:.6e}, tolerance {tolerance:.6e}, rank {retained_rank}/{spectral_dim} \
                 on ambient dimension {dim}, identifiability floor {:.6e}·vᵀBv with \
                 vᵀBv ∈ [{:.6e}, {:.6e}], arithmetic floor {:.6e}",
                sae_exact_a_identifiability_floor(),
                self.metric_scale
                    .iter()
                    .copied()
                    .fold(f64::INFINITY, f64::min),
                self.metric_scale
                    .iter()
                    .copied()
                    .fold(f64::NEG_INFINITY, f64::max),
                (spectral_dim as f64) * f64::EPSILON * self.spectral_norm,
            ));
        }

        Ok(SaeArrowVector {
            t: solution.slice(s![..total_t]).to_owned(),
            beta: solution.slice(s![total_t..]).to_owned(),
        })
    }
}

/// One coherent dense exact-A quotient geometry.  The joint eigensystem owns
/// the stationarity pseudoinverse; the priced joint inverse owns the exact-A
/// log-determinant derivative.  Both are derived from the same materialized
/// block and classification floor.
struct ExactHessianQuotientGeometry {
    joint: ExactHessianSpectralBlock,
    joint_pricing: ExactHessianPricing,
}

/// Value and classified basin spectrum, without realizing a dense differential.
struct ExactHessianBasin {
    log_det: f64,
    negative: Vec<usize>,
    complement: Vec<usize>,
    basis: Array2<f64>,
    rotation: Array2<f64>,
    vectors: Array2<f64>,
    inverse_values: Array1<f64>,
}

/// Differential of one coherently priced spectral block.
/// E has a coordinate diagonal and a dense decoder-prior border block.
/// `a_derivative` includes the negative-subspace projector response.
struct ExactHessianPricing {
    a_derivative: Array2<f64>,
    clamp_diagonal_derivative: Array1<f64>,
    clamp_border_derivative: Array2<f64>,
}

/// Complete dense exact-A derivative cluster.  The stationarity adjoint is
/// solved before the geometry is discarded, making this the sole owner of the
/// dense exact-A inverse action.
pub(crate) struct DenseExactALogdetChannels {
    pub(crate) logdet_trace: Array1<f64>,
    pub(crate) theta_adjoint: SaeArrowVector,
    pub(crate) stationarity_adjoint: SaeArrowVector,
}

/// #2515 — WHICH curvature operator a derivative channel differentiates and
/// contracts.
///
/// `A = B + ΔC = ∇²_θθ L` is the exact observed information; it is the operator
/// whose log-determinant the Laplace criterion **is**. `B` is the Gauss--Newton /
/// PSD-majorizer arrow system: the positive-definite scale the Newton and IFT
/// solves factor, and a preconditioner for `A`. A preconditioner is not the
/// operator it preconditions.
///
/// This exists as one value resolved ONCE per gradient assembly, rather than as
/// a per-channel argument, because the failure mode it guards is a channel
/// contracting one operator's inverse while differentiating the other's — a
/// state that is neither `A` nor `B` and that no single channel can detect
/// locally.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum EvidenceOperator {
    /// The Gauss--Newton / PSD-majorizer arrow system `B`.
    Majorizer,
    /// The exact observed information `A = B + ΔC = ∇²_θθ L`.
    ExactObservedInformation,
}

impl EvidenceOperator {
    /// The historical boolean spelling, for channels whose exact-`A` port landed
    /// before this type did (`logdet_theta_adjoint_from_probes`, `ac499b513`).
    #[must_use]
    pub(crate) fn is_exact_a(self) -> bool {
        matches!(self, Self::ExactObservedInformation)
    }
}

/// #2515 — the selected-inverse evidence a bundle-routed outer ρ-gradient
/// contracts, and there is exactly ONE: the exact observed information
/// `A = B + ΔC = ∇²_θθ L`.
///
/// The from-probes channels reconstruct the arrow inverse blocks as
/// `(H⁻¹)_tt = A_i⁻¹ + G_i S⁻¹ G_iᵀ` and `(H⁻¹)_tβ = −G_i S⁻¹`: the row factors
/// `A_i` and cross blocks come from a factor CACHE, and `S⁻¹` comes from the
/// probe bundle. Those two must factor the SAME operator. Before this type
/// existed the streaming lane paired a bundle built on `exact_a_evidence_system`'s
/// reduced Schur with the `B` row-factor cache from
/// `converge_inner_for_undamped_logdet`, reconstructing an inverse belonging to
/// neither.
///
/// Carrying the cache HERE, rather than reading it off the assembler's `cache`
/// argument, is what makes that mixture unrepresentable. `cache` stays the `B`
/// stationarity geometry that
/// [`SaeManifoldTerm::solve_exact_stationarity_matrix_free`] reassembles
/// `A = B + ΔC` on top of; promoting it to `A` would double-count `ΔC`.
///
/// PRODUCTION ALWAYS MINTS `ExactObservedInformation`, from one place: the
/// `StreamingEvidenceArtifacts` of a single gradient-bearing evidence
/// evaluation, where the cache and the bundle were produced by one
/// factorization of one system and cannot be paired across evaluations. The
/// criterion that route feeds ranks `½log|S_A|` —
/// `rank_adjusted_quasi_laplace_complexity` takes `½log_det` (coordinate block included) and
/// both come off `exact_a_evidence_system`, so the per-row t-block
/// log-determinants cancel and the reduced Schur of `A` is the whole operator
/// exposure — so a `B`-rooted derivative here would differentiate an operator
/// the value never ranked. That was #2515.
///
/// `Majorizer` is kept reachable for the regression gates that pin the
/// from-probes RECONSTRUCTION against its dense sibling (#2712, #2080): that
/// contract is about the reconstruction machinery and holds for either operator,
/// so forcing those fixtures onto an exact-`A` geometry their states may not even
/// admit would test less, not more.
pub(crate) struct BundleEvidenceGeometry<'a> {
    /// Which operator `cache` and `sinv` BOTH factor.
    pub(crate) operator: EvidenceOperator,
    /// Factor cache of the arrow system whose reduced Schur produced `sinv`.
    /// Distinct from the assembler's `cache`, which stays `B` on every route.
    pub(crate) cache: &'a ArrowFactorCache,
    /// `(probes, S_A⁻¹·probes)`. For the rational lane these are the identical
    /// weighted vectors emitted by
    /// `RationalLogdetPlan::into_directional_derivative_bundle`, so every
    /// contraction is the derivative of the SAME shifted rational value rather
    /// than a separately sampled `S⁻¹`.
    pub(crate) probes: &'a [Array1<f64>],
    pub(crate) sinv: &'a [Array1<f64>],
}

include!("exact_stationarity_krylov.rs");

/// #2330 Patch D — shared per-row context for the residual-curvature
/// third-derivative legs: the whitened `√w·M·r` error metric, its `√w` twin,
/// the frozen assignments/jets, and the ordered-Beta–Bernoulli gate mode. One
/// borrow per row replaces the per-call argument tower of the two leg helpers.
#[derive(Clone, Copy)]
struct PatchDResidualCtx<'a> {
    row: usize,
    error_metric: &'a [f64],
    sqrt_w: f64,
    assignments: &'a Array1<f64>,
    second_jets: &'a [Array4<f64>],
    third_jets: Option<&'a [Option<ndarray::Array5<f64>>]>,
    is_obb: bool,
    inv_tau: f64,
}

/// #2500 — what the assignment prior's sparse log-strength curvature operator
/// `∂H_tt/∂ρ_sparse` IS for the family in play, produced by the single authority
/// [`SaeManifoldTerm::sparse_logit_curvature_rho_derivative`].
///
/// Before this type existed, three channels each re-derived that operator by
/// matching on [`AssignmentMode`] independently — the arrow assembly (which
/// INSTALLS it into `block.htt`), ch4's Daleckii–Krein Hessian, and ch5's
/// forward-sensitivity operator map — and the two derivative channels shared a
/// catch-all `_ =>` refusal. The catch-all is what made `ThresholdGate` unfittable
/// on the dense route: its operator was never hard, it was simply absent from an
/// enumeration, while `assignment_prior_log_strength_hdiag_weighted` had been
/// computing it exactly the whole time for the gradient's trace channel. One
/// quantity with two implementations, and the matrix-valued one declined the case
/// the trace-valued one already computed.
///
/// The three outcomes are exhaustive over `AssignmentMode`, so a family added
/// later cannot fall through to a silently-zero operator: it has to name itself
/// here, and each consumer then decides what to do with the answer.
enum SparseLogitCurvature {
    /// There is no operator to install and a zero row is CORRECT: no sparse outer
    /// coordinate at all (`TopK` is `FixedSupport`), no free logit (`K ≤ 1`
    /// softmax), or a frozen/ungated routing whose prior is inert.
    Inert,
    /// The operator is DIAGONAL on the cache's global logit `t`-slots, as
    /// `(global slot, ∂H_{slot,slot}/∂ρ_sparse)`. Every installed assignment-prior
    /// curvature is diagonal in the free-logit chart — softmax writes the
    /// Gershgorin majorizer `D = diag(Σ_j|H_kj|)` (`row_psd_majorizer` is a
    /// diagonal matrix), threshold-gate writes the exact per-logit
    /// `w·λ·s·(1−2a)/τ²` — and both are degree-one in `λ_sparse = e^{ρ_sparse}`,
    /// so the derivative equals the installed entry itself.
    Diagonal(Vec<(usize, f64)>),
    /// The operator is NOT diagonal and is owned elsewhere: the ordered
    /// Beta--Bernoulli integrated marginal couples every row in an atom column, so
    /// `∂A/∂ρ_sparse` is a cross-row Hessian supplied by
    /// [`SaeManifoldTerm::dense_exact_a_ordered_bb_sparse_trace`]. A consumer that
    /// has that channel emits nothing here; a consumer that does NOT must refuse,
    /// because a diagonal-only stand-in would be a wrong operator rather than a
    /// missing one.
    CrossRowOwnedElsewhere,
}

/// #2731 — the residual-curvature half of `ΔC` at ONE state, contracted once
/// instead of once per apply.
///
/// Legs (1a) and (1b) of [`SaeManifoldTerm::apply_exact_hessian_minus_b_prepared`]
/// are `⟨r_n, ∂²f_ab⟩` and `⟨r_n, ∂²f_aβ⟩`: the metric-applied row residual
/// against the row's second and mixed jets. Both factors belong to the state;
/// only the direction they multiply changes from one apply to the next. The
/// per-apply form rebuilt every row's packed jet buffer
/// (`q·p + q²·p + n_β·p + 2·q·n_β·p` doubles, allocated zeroed) and contracted
/// it again on every probe. On #2731's `p = 2048, charts = 32` cell that buffer
/// is ~24 MB per row, for each of 256 rows, for each of the 290 probes of one
/// dense materialization; job 391502 read 700–827 ms per apply on one core.
///
/// Held per row: `q × q` and `q × n_β` doubles, the order of the `H_tβ` block
/// the factor cache already holds for that row. Empty under a softmax gate, whose
/// resident contracted kernel never materializes the packed channels; that gate
/// carries its row-jet inputs and residual probe rows in `softmax` instead. A
/// plan is valid for exactly the state it was built from; the callers hold
/// `&self` across the applies they share it with, which is the proof.
pub(crate) struct PreparedResidualCurvatureRows {
    rows: Vec<PreparedResidualCurvatureRow>,
    /// `border_channels_for_cache(cache)[β].index`, in border order.
    border_indices: Vec<usize>,
    /// #2822 — the softmax gate's resident row-jet plan; `None` for every other gate.
    softmax: Option<PreparedSoftmaxRowJets>,
}

/// #2822 — the state-only inputs of the softmax residual-curvature HVP
/// (`SaeManifoldTerm::prepare_softmax_row_jets`): the border channels, and per tile of
/// same-shape rows the planner's path, the row-program inputs and the residual probe rows.
pub(crate) struct PreparedSoftmaxRowJets {
    border: Vec<SaeBorderChannel>,
    tiles: Vec<PreparedSoftmaxRowJetTile>,
}

struct PreparedSoftmaxRowJetTile {
    start: usize,
    q: usize,
    path: crate::gpu_kernels::sae_rowjet::SaeRowJetPath,
    inputs: Vec<crate::gpu_kernels::sae_rowjet::SaeSoftmaxRowJetInput>,
    probe: Vec<f64>,
}

struct PreparedResidualCurvatureRow {
    vars: Vec<SaeLocalRowVar>,
    /// `⟨r, ∂²f_ab⟩`, row-major `q × q`.
    residual_tt: Vec<f64>,
    /// `⟨r, ∂²f_aβ⟩`, row-major `q × n_β`.
    residual_tbeta: Vec<f64>,
}

impl SaeManifoldTerm {
    /// Contract the residual-curvature legs of `ΔC` at this state. The row
    /// residual and the row program's jets are read exactly as the per-apply
    /// form read them — the same one-row refill, the same `sae_dot` — so an
    /// apply against the plan is bit-identical to one that re-derived them.
    pub(crate) fn prepare_residual_curvature_rows(
        &self,
        target: ArrayView2<'_, f64>,
        cache: &ArrowFactorCache,
    ) -> Result<PreparedResidualCurvatureRows, String> {
        if matches!(self.assignment.mode, AssignmentMode::Softmax { .. }) {
            return Ok(PreparedResidualCurvatureRows {
                rows: Vec::new(),
                border_indices: Vec::new(),
                softmax: Some(self.prepare_softmax_row_jets(target, cache)?),
            });
        }
        let p = self.output_dim();
        let n = self.n_obs();
        let k_atoms = self.k_atoms();
        let second_jets = self.atom_second_jets()?;
        let border = self.border_channels_for_cache(cache)?;
        let n_border = border.len();
        let row_loss_w = self.row_loss_weights.as_deref();
        let whitens = self
            .row_metric
            .as_ref()
            .is_some_and(|metric| metric.whitens_likelihood());
        // #2731 — a row's plan reads only that row: its assignments, its row
        // program's jets (a non-softmax refill builds exactly the row it is asked
        // for), and its residual. The rows run on the rayon pool and are gathered in
        // row order, so the plan is bit-identical to the serial loop and the first
        // failing row's error is the one returned. A pool thread holds one row's
        // jets at a time. The dense exact-A build and the materialization forecast
        // each prepare this plan once per polish step.
        use rayon::prelude::*;
        let planned: Vec<Result<PreparedResidualCurvatureRow, String>> = (0..n)
            .into_par_iter()
            .map(|row| -> Result<PreparedResidualCurvatureRow, String> {
                let q = cache.row_dims[row];
                let mut assignments = Array1::<f64>::zeros(k_atoms);
                let a_scratch = assignments.as_slice_mut().ok_or_else(|| {
                    "prepare_residual_curvature_rows: assignment scratch is not contiguous"
                        .to_string()
                })?;
                self.assignment.try_assignments_row_into(row, a_scratch)?;
                // #932 complete schedule: non-softmax gates use their distinct
                // dynamic row program, one row per refill.
                let mut jet_window: std::collections::VecDeque<SaeRowJets> =
                    std::collections::VecDeque::new();
                self.refill_jet_window(row, cache, &second_jets, &border, &mut jet_window)?;
                let jets = jet_window.pop_front().ok_or_else(|| {
                    format!("prepare_residual_curvature_rows: the jet refill built no row {row}")
                })?;
                let sqrt_row_w = row_loss_w.map_or(1.0, |w| w[row].sqrt());

                // √w-scaled metric-applied per-row residual `error_metric = √w·M_n r_n`
                // (the SAME object the assembly's β-tier gradient contracts). The
                // data-fit `½ r_nᵀ M_n r_n` has residual curvature `Σ (M_n r_n)·∂²f`,
                // so this is exactly the residual contracted against the raw `∂²f`
                // jets. `M_n = I` on the isotropic path ⇒ `error_metric = √w·r`.
                let mut decoded = vec![0.0_f64; p];
                let mut fitted = Array1::<f64>::zeros(p);
                let mut error = Array1::<f64>::zeros(p);
                let active_atoms = self
                    .last_row_layout
                    .as_ref()
                    .map(|layout| layout.active_atoms[row].as_slice());
                for k in 0..k_atoms {
                    if active_atoms.is_some_and(|active| active.binary_search(&k).is_err()) {
                        continue;
                    }
                    self.atoms[k].fill_decoded_row(row, &mut decoded);
                    let a_k = assignments[k];
                    for out_col in 0..p {
                        fitted[out_col] += a_k * decoded[out_col];
                    }
                }
                for out_col in 0..p {
                    error[out_col] = sqrt_row_w * (fitted[out_col] - target[[row, out_col]]);
                }
                let error_metric: Vec<f64> = match self.row_metric.as_ref() {
                    Some(metric) if whitens => metric.apply_metric_row(row, error.view()),
                    _ => error.to_vec(),
                };

                let mut residual_tt = vec![0.0_f64; q * q];
                for a in 0..q {
                    for b in 0..q {
                        residual_tt[a * q + b] = sae_dot(&error_metric, jets.second(a, b));
                    }
                }
                let mut residual_tbeta = vec![0.0_f64; q * n_border];
                for a in 0..q {
                    for beta_pos in 0..n_border {
                        residual_tbeta[a * n_border + beta_pos] =
                            sae_dot(&error_metric, jets.beta_deriv(a, beta_pos));
                    }
                }
                Ok(PreparedResidualCurvatureRow {
                    vars: jets.vars,
                    residual_tt,
                    residual_tbeta,
                })
            })
            .collect();
        let rows = planned
            .into_iter()
            .collect::<Result<Vec<PreparedResidualCurvatureRow>, String>>()?;
        Ok(PreparedResidualCurvatureRows {
            rows,
            border_indices: border.iter().map(|channel| channel.index).collect(),
            softmax: None,
        })
    }
}

impl SaeManifoldTerm {
    /// #2500 — the ONE authority for `∂H_tt/∂ρ_sparse` on the free-logit slots.
    /// See [`SparseLogitCurvature`] for why this exists and what each outcome
    /// obliges a consumer to do.
    ///
    /// Both diagonal families are degree-one in `λ_sparse = e^{ρ_sparse}` at a
    /// frozen inner state, so the derivative IS the installed entry:
    ///
    /// * softmax — `D_k = Σ_j soft|scale·H_kj|` with `scale = λ_sparse·s/τ²`; the
    ///   soft-abs seam is positively homogeneous of degree one in `scale`
    ///   (`ε_k² = ε₀²Σ_l H_kl²` scales with it), and its `sign(H_kj)` kink lives in
    ///   the LOGITS, which a ρ perturbation never moves;
    /// * threshold gate — `w·λ_sparse·s·(1−2a)/τ²` with `a = σ((ℓ−θ)/τ)`,
    ///   `s = a(1−a)`, read from `assignment_prior_log_strength_hdiag_weighted`,
    ///   which is the SAME builder `assignment_prior_grad_hdiag_weighted` supplies
    ///   to the arrow assembly and already carries the `#991` row weights, the
    ///   `#Bug4` fixed-logit mask, and the frozen-routing zeroing.
    ///
    /// Note the threshold-gate operator is SIGNED (`1−2a` flips at the threshold):
    /// unlike softmax's Gershgorin radius and the ARD `max(·,0)` majorizer, no
    /// clamp is interposed on this family — the assembly installs the exact prior
    /// curvature (`construction_arrow_schur_assembly.rs`, the `raw` branch), so the
    /// exact-minus-majorizer delta `ΔC` has no threshold-gate part and
    /// `∂A/∂ρ_sparse = ∂B/∂ρ_sparse` here.
    fn sparse_logit_curvature_rho_derivative(
        &self,
        rho: &SaeManifoldRho,
        cache: &ArrowFactorCache,
    ) -> Result<SparseLogitCurvature, String> {
        if rho.sparse_flat_index().is_none() {
            return Ok(SparseLogitCurvature::Inert);
        }
        let k_atoms = self.k_atoms();
        let row_w = self.row_loss_weights.as_deref();
        let assignment_dim = self.assignment.assignment_coord_dim();
        // Only hard-TopK mints a compact row layout, and TopK carries no sparse
        // coordinate — but a FORCED layout can still reach here, and the compact
        // slot map is not the dense `base + atom` chart these operators are written
        // in. Refuse for any family rather than write into the wrong slots.
        let compact_layout_refusal = || {
            format!(
                "sparse_logit_curvature_rho_derivative: the compact top-k row layout is not \
                 covered by the sparse log-strength operator ({}); refusing to assemble a \
                 curvature operator with an unmodelled sparse row",
                self.assignment.mode.family_label()
            )
        };
        match self.assignment.mode {
            AssignmentMode::TopK { .. } => Ok(SparseLogitCurvature::Inert),
            AssignmentMode::OrderedBetaBernoulli { .. } => {
                Ok(SparseLogitCurvature::CrossRowOwnedElsewhere)
            }
            // K ≤ 1 softmax has no free logit: the gradient's sparse logdet trace
            // is identically zero, so a zero operator is the CORRECT curvature.
            AssignmentMode::Softmax { .. } if k_atoms <= 1 => Ok(SparseLogitCurvature::Inert),
            AssignmentMode::Softmax {
                temperature,
                sparsity,
            } => {
                if self.last_row_layout.is_some() {
                    return Err(compact_layout_refusal());
                }
                let inv_tau = 1.0 / temperature;
                let scale = rho.lambda_sparse()? * sparsity * inv_tau * inv_tau;
                let penalty = gam_terms::analytic_penalties::SoftmaxAssignmentSparsityPenalty::new(
                    k_atoms,
                    temperature,
                );
                let mut entries = Vec::new();
                for row in 0..self.n_obs() {
                    let w_row = row_w.map_or(1.0, |w| w[row]);
                    let base = cache.row_offsets[row];
                    let logit_dim = assignment_dim.min(cache.row_dims[row]);
                    let row_logits: Vec<f64> = (0..k_atoms)
                        .map(|atom| self.assignment.logits[[row, atom]])
                        .collect();
                    let d = penalty.psd_majorizer_abs_row_sums(&row_logits, scale);
                    for atom in 0..logit_dim {
                        entries.push((base + atom, w_row * d[atom]));
                    }
                }
                Ok(SparseLogitCurvature::Diagonal(entries))
            }
            AssignmentMode::ThresholdGate { .. } => {
                if self.last_row_layout.is_some() {
                    return Err(compact_layout_refusal());
                }
                let hdiag = crate::assignment::assignment_prior_log_strength_hdiag_weighted(
                    &self.assignment,
                    rho,
                    row_w,
                )?;
                if hdiag.is_empty() {
                    return Ok(SparseLogitCurvature::Inert);
                }
                let mut entries = Vec::new();
                for row in 0..self.n_obs() {
                    let base = cache.row_offsets[row];
                    let logit_dim = assignment_dim.min(cache.row_dims[row]);
                    for atom in 0..logit_dim {
                        entries.push((base + atom, hdiag[row * k_atoms + atom]));
                    }
                }
                Ok(SparseLogitCurvature::Diagonal(entries))
            }
        }
    }

    /// Legs (1)–(4) of `Self::apply_exact_hessian_minus_b`. Leg (5) is
    /// `Self::decoder_prior_gap_border_leg`, which
    /// [`Self::apply_exact_hessian_minus_b_prepared`] folds in after these four.
    ///
    /// #2828 — the β leg's plan is a property of the DECODER STATE, not of the
    /// direction, so a caller that applies `ΔC` many times at one state (a dense
    /// materialization's `slots + k` probes, a Krylov solve's iterations) builds
    /// it once and hands it to leg (5) instead of rebuilding it per apply.
    /// Measured on a 10-atom, `p = 16`, `n = 60` fixture with every pair
    /// near-collinear: 2.79 ms of a 15.19 ms apply. None of legs (1)–(4) reads it.
    ///
    /// #2731 — the residual-curvature legs are the same kind of object and take
    /// the same treatment: `residual` is
    /// [`Self::prepare_residual_curvature_rows`] at this state.
    fn apply_exact_hessian_minus_b_prepared_before_beta_prior_leg(
        &self,
        rho: &SaeManifoldRho,
        cache: &ArrowFactorCache,
        v: &SaeArrowVector,
        residual: &PreparedResidualCurvatureRows,
    ) -> Result<SaeArrowVector, String> {
        self.assignment.validate_rho_domain(rho)?;
        let n = self.n_obs();
        let k_atoms = self.k_atoms();
        let total_t = cache.delta_t_len();
        let row_loss_w = self.row_loss_weights.as_deref();
        let ard_axis_periods: Vec<Vec<Option<f64>>> = self
            .assignment
            .coords
            .iter()
            .map(|coord| coord.effective_axis_periods())
            .collect();
        let ard_precisions = self.validated_ard_precisions(rho)?;

        // Optional softmax exact-entropy-minus-majorizer delta operator (#1419).
        let softmax_delta: Option<(
            gam_terms::analytic_penalties::SoftmaxAssignmentSparsityPenalty,
            f64,
        )> = match self.assignment.mode {
            AssignmentMode::Softmax {
                temperature,
                sparsity,
            } if k_atoms > 1 => {
                let inv_tau = 1.0 / temperature;
                let scale = rho.lambda_sparse()? * sparsity * inv_tau * inv_tau;
                Some((
                    gam_terms::analytic_penalties::SoftmaxAssignmentSparsityPenalty::new(
                        k_atoms,
                        temperature,
                    ),
                    scale,
                ))
            }
            _ => None,
        };

        let mut out = SaeArrowVector {
            t: Array1::<f64>::zeros(total_t),
            beta: Array1::<f64>::zeros(cache.k),
        };
        // #1557 — reuse one K-sized scratch row across all N rows (alias-free).
        let mut assignments = Array1::<f64>::zeros(self.k_atoms());
        // Ordered Beta--Bernoulli's exact prior Hessian couples all rows within
        // each atom column. Gather the logit slice of `v` while visiting the
        // row-local cache layout, then apply the analytic column reductions once
        // after the row loop. This remains O(NK) memory/time and constructs no
        // dense cross-row matrix or persistent carrier.
        let mut ordered_logit_direction = matches!(
            self.assignment.mode,
            AssignmentMode::OrderedBetaBernoulli { .. }
        )
        .then(|| Array1::<f64>::zeros(n * k_atoms));
        // #2520 — the ThresholdGate prior's concave half, dropped by the PSD
        // majorizer `B` now declares and restored here so `A = B + ΔC` is still
        // the exact signed curvature. Diagonal in the logit slots, so unlike
        // ordered Beta--Bernoulli it needs no direction gather.
        let threshold_gate_remainder = match self.assignment.mode {
            AssignmentMode::ThresholdGate { .. } => Some(
                crate::assignment::threshold_gate_negative_hessian_remainder_weighted(
                    &self.assignment,
                    rho,
                    row_loss_w,
                )?,
            ),
            _ => None,
        };
        if matches!(self.assignment.mode, AssignmentMode::Softmax { .. }) {
            // #2822 — the resident contracted kernel below never materializes the
            // packed channels. Its row-program inputs, residual probe rows and border
            // channels are state-only, so `residual` carries them for a softmax gate and
            // an apply gathers only the direction.
            let prepared = residual.softmax.as_ref().ok_or_else(|| {
                "apply_exact_hessian_minus_b: a softmax gate's residual plan carries no row-jet plan"
                    .to_string()
            })?;
            let border = &prepared.border;
            // #2304 resident path for the residual-curvature blocks (1a)+(1b):
            // the raw second/mixed jets are contracted on device (when the plan
            // admits it) against the metric-applied √w-scaled residual and the
            // direction's (t, β) coefficients — the packed channel tensors are
            // never materialized. Blocks (2)-(3) below are logit/coord-space
            // prior curvatures with no channel tensors involved and stay on
            // the host.
            {
                let v_t_for_row = |row: usize, q: usize| -> Result<Vec<f64>, String> {
                    let base = cache.row_offsets[row];
                    Ok((0..q).map(|c| v.t[base + c]).collect())
                };
                let v_beta_row: Vec<f64> =
                    border.iter().map(|channel| v.beta[channel.index]).collect();
                let out_ref = &mut out;
                self.contracted_softmax_bilinear_hvp(
                    prepared,
                    v_t_for_row,
                    &v_beta_row,
                    |row, row_vars, t_row, beta_row| {
                        // The callback's per-row var count must agree with the
                        // row slice it is handed; a mismatch would silently
                        // scatter a short row into the wrong output offsets.
                        assert_eq!(t_row.len(), row_vars);
                        let base = cache.row_offsets[row];
                        for (a, &value) in t_row.iter().enumerate() {
                            out_ref.t[base + a] += value;
                        }
                        for (channel, &value) in border.iter().zip(beta_row) {
                            out_ref.beta[channel.index] += value;
                        }
                        Ok(())
                    },
                )?;
            }
            // (2) softmax entropy-minus-majorizer and (3) periodic-ARD deltas,
            // per row with the layout rebuilt from the cache (no jets needed).
            for row in 0..n {
                let q = cache.row_dims[row];
                let base = cache.row_offsets[row];
                self.assignment.try_assignments_row_into(
                    row,
                    assignments.as_slice_mut().ok_or_else(|| {
                        "apply_exact_hessian_minus_b: assignment scratch is not contiguous"
                            .to_string()
                    })?,
                )?;
                let vars = self.row_vars_for_cache_row(row, cache)?;
                let v_t: Vec<f64> = (0..q).map(|c| v.t[base + c]).collect();
                let w_row = row_loss_w.map_or(1.0, |w| w[row]);
                if let Some((_penalty, scale)) = softmax_delta.as_ref() {
                    let assignment_dim = self.assignment.assignment_coord_dim();
                    let a_soft = assignments
                        .as_slice()
                        .expect("softmax assignments row must be contiguous");
                    let m = softmax_majorizer_log_mean(a_soft);
                    for (a, va) in vars.iter().enumerate() {
                        let SaeLocalRowVar::Logit { atom: ka } = *va else {
                            continue;
                        };
                        if ka >= assignment_dim {
                            continue;
                        }
                        let mut acc = 0.0_f64;
                        for (b, vb) in vars.iter().enumerate() {
                            let SaeLocalRowVar::Logit { atom: kb } = *vb else {
                                continue;
                            };
                            if kb >= assignment_dim {
                                continue;
                            }
                            let h_entropy =
                                softmax_dense_entropy_hessian_entry(a_soft, ka, kb, m, *scale);
                            let delta = if ka == kb {
                                h_entropy
                                    - active_softmax_gershgorin_majorizer_entry(
                                        a_soft, ka, m, *scale,
                                    )
                            } else {
                                h_entropy
                            };
                            acc += w_row * delta * v_t[b];
                        }
                        out.t[base + a] += acc;
                    }
                }
                for (a, va) in vars.iter().enumerate() {
                    let SaeLocalRowVar::Coord { atom, axis } = *va else {
                        continue;
                    };
                    if rho.log_ard[atom].is_empty() {
                        continue;
                    }
                    let alpha = ard_precisions[atom][axis];
                    let t_val = self.assignment.coords[atom].row(row)[axis];
                    let prior = ArdAxisPrior::eval(alpha, t_val, ard_axis_periods[atom][axis]);
                    let neg = prior.negative_hessian_remainder();
                    if neg != 0.0 {
                        out.t[base + a] += w_row * neg * v_t[a];
                    }
                }
            }
            return Ok(out);
        }
        // #932 complete schedule, #2731 contracted once per state: the row
        // program's jets and the row residual were read by
        // `Self::prepare_residual_curvature_rows`; only the direction is read here.
        if residual.rows.len() != n || residual.border_indices.len() != cache.k {
            return Err(format!(
                "apply_exact_hessian_minus_b: residual-curvature plan has {} rows and {} border \
                 channels, but this state has {n} rows and border width {}; a plan is valid only \
                 for the state it was prepared from",
                residual.rows.len(),
                residual.border_indices.len(),
                cache.k,
            ));
        }
        let border_indices = residual.border_indices.as_slice();
        let n_border = border_indices.len();
        for row in 0..n {
            let q = cache.row_dims[row];
            let base = cache.row_offsets[row];
            let row_plan = &residual.rows[row];
            if row_plan.vars.len() != q {
                return Err(format!(
                    "apply_exact_hessian_minus_b: row {row} was prepared with {} primaries, but \
                     the cache row has {q}",
                    row_plan.vars.len(),
                ));
            }

            // Local t-slice of `v` for this row.
            let v_t: Vec<f64> = (0..q).map(|c| v.t[base + c]).collect();
            if let Some(direction) = ordered_logit_direction.as_mut() {
                for (local, var) in row_plan.vars.iter().enumerate() {
                    if let SaeLocalRowVar::Logit { atom } = *var {
                        direction[row * k_atoms + atom] = v_t[local];
                    }
                }
            }

            // (1a) residual curvature, t–t: ΔC_tt[a,b] = ⟨r, ∂²f_ab⟩.
            for a in 0..q {
                let mut acc = 0.0_f64;
                for b in 0..q {
                    let r_ab = row_plan.residual_tt[a * q + b];
                    acc += r_ab * v_t[b];
                }
                out.t[base + a] += acc;
            }
            // (1b) residual curvature, t–β and β–t: ΔC_tβ[a,β] = ⟨r, ∂²f_aβ⟩.
            //      `jets.beta_deriv[a][β]` = ∂(∂f/∂β_β)/∂θ_a (the mixed second jet).
            for a in 0..q {
                for (beta_pos, &index) in border_indices.iter().enumerate() {
                    let r_ab = row_plan.residual_tbeta[a * n_border + beta_pos];
                    // t row picks up β leg of v; β row picks up t leg of v.
                    out.t[base + a] += r_ab * v.beta[index];
                    out.beta[index] += r_ab * v_t[a];
                }
            }

            // (2) softmax entropy-minus-majorizer: softmax gates return through
            // the resident contracted branch above (#1419 algebra preserved
            // there verbatim, including the #1410 active-slot contraction and
            // the #991 `w_row` convention), so no softmax delta arises here.

            // (3) periodic ARD: ΔC_coord = V'' − psd_majorizer_hess =
            // negative_hessian_remainder, diagonal (#2339: the smooth
            // homogeneity-preserving clamp, non-positive). The assembly writes the
            // mean-one design-weighted majorizer `w_row·psd_majorizer_hess`, so the
            // dropped-curvature correction must carry that same `w_row`: `A = B + ΔC`
            // then recovers `w_row·V''` exactly (the seam guarantees
            // `psd_majorizer_hess + negative_hessian_remainder == V''` bit-for-bit).
            // The prior is weighted directly, not through the √w data-jet seam.
            let w_row = row_loss_w.map_or(1.0, |w| w[row]);
            for (a, va) in row_plan.vars.iter().enumerate() {
                let SaeLocalRowVar::Coord { atom, axis } = *va else {
                    continue;
                };
                if rho.log_ard[atom].is_empty() {
                    continue;
                }
                let alpha = ard_precisions[atom][axis];
                let t_val = self.assignment.coords[atom].row(row)[axis];
                let prior = ArdAxisPrior::eval(alpha, t_val, ard_axis_periods[atom][axis]);
                let neg = prior.negative_hessian_remainder();
                if neg != 0.0 {
                    out.t[base + a] += w_row * neg * v_t[a];
                }
            }

            // (3b) #2520 threshold gate: the same shape as (3), on logit slots
            // rather than coordinate slots. `B` now carries the PSD majorizer
            // `smooth_psd_clamp(w·λ·s/τ², 1 − 2a)`, so `ΔC` must carry the
            // non-positive remainder or `A` would no longer be the exact signed
            // curvature and every exact-Hessian consumer (the IFT response, the
            // terminal Newton polish, the #2336 attributability test) would be
            // differentiating a different operator than it declares. The
            // remainder is already design-weighted and fixed-logit masked by the
            // producer, exactly as the majorizer is.
            if let Some(remainder) = threshold_gate_remainder.as_ref() {
                for (a, va) in row_plan.vars.iter().enumerate() {
                    let SaeLocalRowVar::Logit { atom } = *va else {
                        continue;
                    };
                    let neg = remainder[row * k_atoms + atom];
                    if neg != 0.0 {
                        out.t[base + a] += neg * v_t[a];
                    }
                }
            }
        }

        // (4) ordered Beta--Bernoulli: exact integrated-marginal Hessian minus
        // the diagonal PSD majorizer written into B. The helper evaluates the
        // negative within-column rank-one action by column reductions and the
        // row-local diagonal remainder directly, then we scatter its flat logit
        // result back into the cache's row-local coordinates.
        if let Some(direction) = ordered_logit_direction {
            let delta = crate::assignment::ordered_beta_bernoulli_exact_hessian_minus_majorizer_hvp_weighted(
                &self.assignment,
                rho,
                row_loss_w,
                direction.view(),
            )?;
            for row in 0..n {
                let base = cache.row_offsets[row];
                let vars = self.row_vars_for_cache_row(row, cache)?;
                for (local, var) in vars.iter().enumerate() {
                    if let SaeLocalRowVar::Logit { atom } = *var {
                        out.t[base + local] += delta[row * k_atoms + atom];
                    }
                }
            }
        }

        Ok(out)
    }

    /// `ΔC·v` against plans prepared once for this state: legs (1)–(4) from
    /// `Self::apply_exact_hessian_minus_b_prepared_before_beta_prior_leg`, then
    /// leg (5) from `Self::decoder_prior_gap_border_leg`, added to `out.β` in index
    /// order.
    pub(crate) fn apply_exact_hessian_minus_b_prepared(
        &self,
        rho: &SaeManifoldRho,
        cache: &ArrowFactorCache,
        v: &SaeArrowVector,
        prepared: &PreparedDecoderPriorBetaCurvature,
        residual: &PreparedResidualCurvatureRows,
    ) -> Result<SaeArrowVector, String> {
        let mut out = self.apply_exact_hessian_minus_b_prepared_before_beta_prior_leg(
            rho, cache, v, residual,
        )?;
        if cache.k > 0 {
            let projection = crate::frames::FrameProjection::new(self);
            let leg =
                self.decoder_prior_gap_border_leg(cache, prepared, &projection, v.beta.view())?;
            for (index, &value) in leg.iter().enumerate() {
                out.beta[index] += value;
            }
        }
        Ok(out)
    }

    /// Leg (5) of `ΔC·v` (#2828): the β-tier decoder priors' exact-minus-majorizer
    /// curvature along `v_beta`, in the cache's own border coordinates.
    ///
    /// Until this leg existed `ΔC` had NO β block at all, so `A_ββ` was
    /// whatever PSD majorizer the assembly installed (the repulsion's
    /// Gauss-Newton block, the amplitude barrier's isotropic ridge, the
    /// separation barrier's `|M|` coupling and `lev` ridge) rather than the
    /// second derivative of the objective `assemble_arrow_schur` gradients —
    /// the whole of the #2330 disagreement, and one-directional: a majorizer
    /// only ever OVER-claims curvature, which is exactly how an
    /// `IndefiniteObservedInformation` refusal can fire on a mode that is not
    /// a saddle.
    ///
    /// The remainder is derived in the full-`B` decoder layout because that is
    /// where the priors live and where the assembly writes them (BEFORE the
    /// frame transform). Under an engaged frame the border coordinate is the
    /// factored `C`, and `B = ΦC` with `Φ = blkdiag(I_M ⊗ U_k)`, `U_kᵀU_k = I`,
    /// so the correct factored operator is the congruence `Φᵀ ΔC_ββ Φ` — lift
    /// the direction, apply, project back. That is the same sandwich
    /// `add_factored_repulsion_curvature` applies to the majorizer this
    /// subtracts, so the two stay in one coordinate system.
    ///
    /// `E_ββ = B − A` on the border is this leg's negation, so a caller that keeps
    /// the columns of `k` border probes has the gap border without a second pass.
    fn decoder_prior_gap_border_leg(
        &self,
        cache: &ArrowFactorCache,
        prepared: &PreparedDecoderPriorBetaCurvature,
        projection: &crate::frames::FrameProjection,
        v_beta: ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, String> {
        let beta_dim = self.beta_dim();
        let framed = self.last_frames_active && cache.k == self.factored_border_dim();
        if framed {
            let lifted = projection.lift_border_vec(v_beta);
            let delta = self.decoder_prior_exact_minus_majorizer_beta_hvp_prepared(
                prepared,
                lifted.view(),
            )?;
            Ok(projection.project_border_vec(delta.view()))
        } else if cache.k == beta_dim {
            self.decoder_prior_exact_minus_majorizer_beta_hvp_prepared(prepared, v_beta)
        } else {
            Err(format!(
                "apply_exact_hessian_minus_b: border width {} is neither the full-B \
                 beta_dim {beta_dim} nor the factored border dim {}, so the beta-tier \
                 decoder-prior curvature correction has no coordinate system to be \
                 expressed in",
                cache.k,
                self.factored_border_dim(),
            ))
        }
    }
    /// #2828 — every entry `Σ_{r,c} e_beta[r,c]·left[total_t+r, a]·right[total_t+c, b]`,
    /// the border block's contribution to the quadratic forms between the columns of two
    /// bases in the `(t, β)` layout. All zero when the block is absent or the vectors
    /// carry no border rows (the coordinate-only spectral block).
    ///
    /// The block is applied once to each right column, `k²` work per column, and each
    /// entry contracts one left column against that image, `k` work per entry. A scalar
    /// form per entry paid `k²` for each of `left × right` entries: on the high-p
    /// gauge-deflated K=1 circle at p = 512, a repeated eigenvalue run of ~1530 border
    /// directions against k = 1536 put every eu-stack sample after the operator build in
    /// this form (pool job 614715). The image keeps the scalar form's per-row inner sum
    /// and the contraction keeps its row order and its skip of zero left rows, so every
    /// entry is bit-identical to the scalar form.
    fn dropped_curvature_border_forms(
        e_beta: Option<&Array2<f64>>,
        total_t: usize,
        left: ndarray::ArrayView2<'_, f64>,
        right: ndarray::ArrayView2<'_, f64>,
    ) -> Array2<f64> {
        let mut forms = Array2::<f64>::zeros((left.ncols(), right.ncols()));
        let Some(block) = e_beta else {
            return forms;
        };
        let k = block.nrows();
        if left.nrows() < total_t + k || right.nrows() < total_t + k {
            return forms;
        }
        let mut image = Array2::<f64>::zeros((k, right.ncols()));
        for b in 0..right.ncols() {
            for row in 0..k {
                let mut inner = 0.0_f64;
                for col in 0..k {
                    inner += block[[row, col]] * right[[total_t + col, b]];
                }
                image[[row, b]] = inner;
            }
        }
        for a in 0..left.ncols() {
            for b in 0..right.ncols() {
                let mut acc = 0.0_f64;
                for row in 0..k {
                    let scale = left[[total_t + row, a]];
                    if scale == 0.0 {
                        continue;
                    }
                    acc += scale * image[[row, b]];
                }
                forms[[a, b]] = acc;
            }
        }
        forms
    }

    /// #2336 — the diagonal of `E = B − A` restricted to the ARD periodic
    /// prior's concave-half clamp (block (3) of
    /// `Self::apply_exact_hessian_minus_b`), over the coordinate (t) block; zero
    /// on the β border and on logit rows.
    ///
    /// `E ⪰ 0` is diagonal in the t-block with entries `w_row·|min(V'',0)|`, the
    /// negative curvature of the periodic ARD prior that the Newton/Schur majorizer
    /// DROPS: the assembly writes only `w_row·max(V'',0)` into `B`
    /// ([`SaeManifoldAtom::psd_majorizer_hess`]), so `A = B + ΔC` with the ARD
    /// channel of `ΔC` equal to `w_row·min(V'',0) ≤ 0`. This is the EXACTLY-known,
    /// bounded amplitude of the prior micro-wrinkle that turns a `B`-converged mode
    /// into an `A`-saddle. Collected here as a diagonal so the criterion can test,
    /// per negative exact-`A` eigendirection `v`, whether `vᵀEv ≥ |λ|` — i.e. whether
    /// the indefiniteness is fully attributable to the clamp (#2336 value-side
    /// E-attributability). Reuses the identical per-row term block (3) applies, so
    /// the two cannot drift.
    pub(crate) fn materialize_ard_concave_clamp_diagonal(
        &self,
        rho: &SaeManifoldRho,
        cache: &ArrowFactorCache,
    ) -> Result<Array1<f64>, String> {
        self.materialize_ard_concave_clamp_diagonal_for_rows(rho, &cache.row_dims)
    }

    /// Factorization-free sibling used while `exact_a_evidence_system` still
    /// owns the raw arrow layout.  Classification is part of assembling the
    /// exact-A evidence operator, so requiring a factor cache here would force
    /// the wrong order (factor first, then decide what was factored).
    pub(crate) fn materialize_ard_concave_clamp_diagonal_for_rows(
        &self,
        rho: &SaeManifoldRho,
        row_dims: &[usize],
    ) -> Result<Array1<f64>, String> {
        self.assignment.validate_rho_domain(rho)?;
        if row_dims.len() != self.n_obs() {
            return Err(format!(
                "materialize_ard_concave_clamp_diagonal_for_rows: {} row dimensions for {} observations",
                row_dims.len(),
                self.n_obs(),
            ));
        }
        let total_t: usize = row_dims.iter().sum();
        let mut e_diag = Array1::<f64>::zeros(total_t);
        if self.k_atoms() == 0 {
            return Ok(e_diag);
        }
        let ard_axis_periods: Vec<Vec<Option<f64>>> = self
            .assignment
            .coords
            .iter()
            .map(|coord| coord.effective_axis_periods())
            .collect();
        let ard_precisions = self.validated_ard_precisions(rho)?;
        let row_loss_w = self.row_loss_weights.as_deref();
        // #2520 — the ThresholdGate's own concave half is the SAME kind of
        // exactly-known, bounded `E ⪰ 0` as the periodic-ARD clamp: `B` declares
        // `smooth_psd_clamp(w·λ·s/τ², 1 − 2a)` and drops the negative part, so a
        // mode whose only indefiniteness IS that dropped part is attributable
        // and must be PRICED under #2336 rather than refused. Reads the same
        // producer as the ΔC channel, so E and ΔC cannot disagree.
        let threshold_gate_remainder =
            crate::assignment::threshold_gate_negative_hessian_remainder_weighted(
                &self.assignment,
                rho,
                row_loss_w,
            )?;
        let k_atoms = self.k_atoms();
        let mut base = 0usize;
        for row in 0..self.n_obs() {
            let vars = self.row_vars_for_row_dim(row, row_dims[row])?;
            let w_row = row_loss_w.map_or(1.0, |w| w[row]);
            for (a, va) in vars.iter().enumerate() {
                if let SaeLocalRowVar::Logit { atom } = *va {
                    // E = B − A, so E = −ΔC ≥ 0 here. The remainder already
                    // carries `w_row` (its producer applies the #991 weight).
                    let neg = threshold_gate_remainder[row * k_atoms + atom];
                    if neg != 0.0 {
                        e_diag[base + a] += -neg;
                    }
                    continue;
                }
                let SaeLocalRowVar::Coord { atom, axis } = *va else {
                    continue;
                };
                if rho.log_ard[atom].is_empty() {
                    continue;
                }
                let alpha = ard_precisions[atom][axis];
                let t_val = self.assignment.coords[atom].row(row)[axis];
                let prior = ArdAxisPrior::eval(alpha, t_val, ard_axis_periods[atom][axis]);
                let neg = prior.negative_hessian_remainder();
                if neg != 0.0 {
                    // E = B − A, so on this diagonal E = −(w_row·neg) = w_row·|neg| ≥ 0.
                    e_diag[base + a] += -w_row * neg;
                }
            }
            base += row_dims[row];
        }
        Ok(e_diag)
    }

    /// #1418: matrix-free apply of the EXACT stationarity Jacobian `A = ∇²_θθ L`:
    /// `A v = B_raw v + ΔC v`, the raw objective-majorizer apply
    /// ([`apply_raw_cached_arrow_hessian`]) plus the matrix-free dropped-curvature
    /// correction `ΔC = A − B` (`Self::apply_exact_hessian_minus_b_prepared`),
    /// against the β-tier plan and the residual-curvature rows prepared once for
    /// this state (#2828, #2731).
    fn apply_exact_hessian_prepared(
        &self,
        rho: &SaeManifoldRho,
        cache: &ArrowFactorCache,
        v: &SaeArrowVector,
        prepared: &PreparedDecoderPriorBetaCurvature,
        residual: &PreparedResidualCurvatureRows,
    ) -> Result<SaeArrowVector, String> {
        // #2515 — the cache factors the conditioned evidence majorizer
        // `Phi(B_raw)`.  That conditioning is a solve/log-determinant policy, not
        // part of the objective Hessian.  Adding ΔC to it would build
        // `Phi(B_raw) + ΔC` on the dense route while the streaming arrow route
        // builds `Phi(B_raw + ΔC)`.  Recover B_raw first so both routes classify
        // the one statistical operator `A_raw = B_raw + ΔC`.
        let b_v = apply_raw_cached_arrow_hessian(cache, v.t.view(), v.beta.view())?;
        let dc_v = self.apply_exact_hessian_minus_b_prepared(rho, cache, v, prepared, residual)?;
        Ok(SaeArrowVector {
            t: &b_v.t + &dc_v.t,
            beta: &b_v.beta + &dc_v.beta,
        })
    }

    /// #1418/#2653: solve `A x = rhs` for the materializable EXACT stationarity
    /// Jacobian `A = ∇²_θθ L` with its symmetric rank-revealing
    /// pseudoinverse.  The same materialized and symmetrized `A` used by the
    /// exact observed-information route declares the quotient null band; both
    /// resolved positive and negative modes are inverted.  This avoids the
    /// ill-conditioned Krylov-basis coefficient cancellation that can satisfy a
    /// projected residual while failing `A x = P_range rhs` on reapplication.
    /// The IFT step `θ̂_ρ = −A⁺ g_ρ` (the sign lives in the caller's
    /// `-0.5` contraction) therefore has one dense owner. Ritz vectors are used in
    /// [`Self::solve_exact_stationarity_matrix_free`], where `A` cannot be
    /// materialized.
    pub(crate) fn solve_exact_stationarity(
        &self,
        rho: &SaeManifoldRho,
        target: ArrayView2<'_, f64>,
        cache: &ArrowFactorCache,
        rhs: &SaeArrowVector,
    ) -> Result<SaeArrowVector, String> {
        self.materialize_exact_stationarity_geometry(rho, target, cache)?
            .solve_stationarity(rhs)
    }

    /// `Self::apply_exact_hessian_matrix_free` against a β-tier plan prepared
    /// once — the form the Krylov solve installs, so the plan is built once per
    /// solve rather than once per iteration.
    pub(crate) fn apply_exact_hessian_matrix_free_prepared(
        &self,
        rho: &SaeManifoldRho,
        cache: &ArrowFactorCache,
        system: &ArrowSchurSystem,
        vector: &SaeArrowVector,
        prepared: &PreparedDecoderPriorBetaCurvature,
        residual: &PreparedResidualCurvatureRows,
    ) -> Result<SaeArrowVector, String> {
        let (base_t, base_beta) =
            matrix_free_arrow_operator_apply(system, cache, vector.t.view(), vector.beta.view())
                .map_err(|error| format!("matrix-free evidence operator: {error}"))?;
        let correction = self.apply_exact_hessian_minus_b_prepared(
            rho, cache, vector, prepared, residual,
        )?;
        let mut out = SaeArrowVector {
            t: &base_t + &correction.t,
            beta: &base_beta + &correction.beta,
        };
        add_raw_row_deflation_correction(
            cache,
            vector.t.view(),
            out.t.view_mut(),
            "apply_exact_hessian_matrix_free",
        )?;
        Ok(out)
    }

    /// Matrix-free exact-stationarity sibling used by the wide-border penalized quasi-Laplace
    /// assignment-strength residual. `system` is the reassembled undamped
    /// bordered operator at the converged inner state; `cache` supplies the same
    /// row factors and H_tbeta operator whose rational log-determinant and shared
    /// inverse-probe bundle were consumed by the value/trace lanes.
    ///
    /// The adjoint uses ordinary-A Ritz directions and Euclidean projections,
    /// with the same per-direction null floor as the dense pseudoinverse.
    fn solve_exact_stationarity_matrix_free(
        &self,
        rho: &SaeManifoldRho,
        target: ArrayView2<'_, f64>,
        cache: &ArrowFactorCache,
        system: &ArrowSchurSystem,
        rhs: &SaeArrowVector,
    ) -> Result<SaeArrowVector, String> {
        // `B` — the CONDITIONED evidence majorizer `Phi(B_raw)`. This is the
        // metric, and it is the right one: it is what `ArrowMetric::Joint` uses
        // on the dense route, and what the criterion's `½log|B|` and the `mu`
        // deflation predicate are denominated in.
        let apply_b = |vector: &SaeArrowVector| -> Result<SaeArrowVector, String> {
            let (t, beta) = matrix_free_arrow_operator_apply(
                system,
                cache,
                vector.t.view(),
                vector.beta.view(),
            )
            .map_err(|error| format!("matrix-free evidence operator: {error}"))?;
            Ok(SaeArrowVector { t, beta })
        };
        // #2828/#2731 — one plan of each kind for the whole Krylov solve, not one
        // per iteration.
        let prepared = self.prepare_decoder_prior_beta_curvature(1.0);
        let residual = self.prepare_residual_curvature_rows(target, cache)?;
        let apply_a = |vector: &SaeArrowVector| -> Result<SaeArrowVector, String> {
            self.apply_exact_hessian_matrix_free_prepared(
                rho, cache, system, vector, &prepared, &residual,
            )
        };
        // #2267 — `B_raw`, the physical majorizer `Phi` conditions. Where they differ
        // the Ritz band edge rises to the stiffness `Phi` substituted, as on the
        // dense route.
        let apply_b_raw = |vector: &SaeArrowVector| -> Result<SaeArrowVector, String> {
            let mut raw = apply_b(vector)?;
            add_raw_row_deflation_correction(
                cache,
                vector.t.view(),
                raw.t.view_mut(),
                "matrix-free raw evidence majorizer",
            )?;
            Ok(raw)
        };
        // Classify ordinary A Ritz directions with the dense route's floor
        // and use Euclidean projections. B is only the classification metric;
        // its inverse and generalized eigenvectors do not define A's inverse.
        solve_exact_stationarity_krylov(rhs, &apply_a, &apply_b, &apply_b_raw)
    }

    /// The raw per-flat-coordinate penalty curvature operators
    /// `M_i = ∂H_raw/∂ρ_i` at a frozen inner state, keyed by flat outer coordinate.
    /// Its consumer is [`Self::exact_stationarity_penalty_derivatives_by_flat`],
    /// the raw exact-`A` derivative map. Each `M_i` is degree-one in
    /// `exp(ρ_i)`: `λ_k·½(S_k+S_kᵀ)⊗I`
    /// on atom `k`'s β-block for smoothing; `w_row·max(α cos κt,0)` on the active
    /// row-local t-slots for periodic ARD (`w_row·α` Euclidean); the softmax
    /// Gershgorin majorizer `w_row·diag(Σ_j|H_kj|)` on the logit slots for the
    /// sparse coordinate, which refuses a compact top-k layout and a non-softmax prior.
    fn raw_penalty_curvature_operators_by_flat(
        &self,
        rho: &SaeManifoldRho,
        cache: &ArrowFactorCache,
    ) -> Result<std::collections::BTreeMap<usize, Array2<f64>>, String> {
        let total_t = cache.delta_t_len();
        let k = cache.k;
        let dim = total_t + k;
        let mut c_by_flat: std::collections::BTreeMap<usize, Array2<f64>> =
            std::collections::BTreeMap::new();

        // Smoothing: Cₐ = (λ_a·½(Sₐ+Sₐᵀ)) ⊗ I on atom a's β-block.
        let lambda_smooth = rho.lambda_smooth_vec()?;
        let p = self.output_dim();
        let frames_active = self.frames_active();
        let (beta_offsets, beta_out_dim): (Vec<usize>, Box<dyn Fn(usize) -> usize>) =
            if frames_active {
                let ranks: Vec<usize> = self.atoms.iter().map(|a| a.border_frame_rank()).collect();
                (
                    self.factored_beta_offsets(),
                    Box::new(move |kk: usize| ranks[kk]),
                )
            } else {
                (self.beta_offsets(), Box::new(move |_: usize| p))
            };
        // #2604 — sectional curvature enters the criterion ONLY through the
        // penalty, because a constant-curvature atom's basis is a monomial patch
        // in the tangent coordinate and carries no κ. So `∂H/∂κ_a = λ_a·∂S_a/∂κ`,
        // the same shape as the smoothness coordinate's `∂H/∂log λ_a = λ_a·S_a`
        // with the Gram replaced by its derivative — which is why it slots into
        // this assembly rather than needing a channel of its own. Every trace,
        // the penalty energy and the rank-aware log-determinant term are then
        // derived by the same machinery that already consumes this map.
        for &a in &rho.kappa_atoms {
            let flat = rho
                .kappa_flat_index(a)
                .ok_or_else(|| format!("curvature atom {a} has no flat outer coordinate"))?;
            let atom = self.atoms.get(a).ok_or_else(|| {
                format!(
                    "curvature coordinate names atom {a}, outside term K={}",
                    self.atoms.len()
                )
            })?;
            let Some(ds) = atom.smooth_penalty_kappa_derivative() else {
                // An atom whose roughness is not curvature-parameterised has no
                // κ to move; leaving the coordinate un-assembled is what makes
                // its gradient entry exactly zero rather than silently wrong.
                continue;
            };
            let m = atom.basis_size();
            let off = beta_offsets[a];
            let r = beta_out_dim(a);
            let lambda = lambda_smooth[a];
            let c = c_by_flat
                .entry(flat)
                .or_insert_with(|| Array2::<f64>::zeros((dim, dim)));
            for mu in 0..m {
                for nu in 0..m {
                    let ds_sym = 0.5 * (ds[[nu, mu]] + ds[[mu, nu]]);
                    let val = lambda * ds_sym;
                    if val == 0.0 {
                        continue;
                    }
                    for oc in 0..r {
                        c[[total_t + off + nu * r + oc, total_t + off + mu * r + oc]] += val;
                    }
                }
            }
        }
        for a in 0..rho.log_lambda_smooth.len() {
            let atom = &self.atoms[a];
            let s = atom.smooth_penalty();
            let m = atom.basis_size();
            let off = beta_offsets[a];
            let r = beta_out_dim(a);
            let lambda = lambda_smooth[a];
            let flat = rho.smooth_flat_index(a);
            let c = c_by_flat
                .entry(flat)
                .or_insert_with(|| Array2::<f64>::zeros((dim, dim)));
            for mu in 0..m {
                for nu in 0..m {
                    let s_sym = 0.5 * (s[[nu, mu]] + s[[mu, nu]]);
                    let val = lambda * s_sym;
                    if val == 0.0 {
                        continue;
                    }
                    for oc in 0..r {
                        c[[total_t + off + nu * r + oc, total_t + off + mu * r + oc]] += val;
                    }
                }
            }
        }

        // ARD: C_{k,axis} = w_row·max(α cos κt, 0) (periodic) / w_row·α (Euclidean)
        // on the row-local t-slot for (atom k, axis).
        let ard_precisions = self.validated_ard_precisions(rho)?;
        let row_w = self.row_loss_weights.as_deref();
        let coord_offsets = self.assignment.coord_offsets();
        let periods: Vec<Vec<Option<f64>>> = self
            .assignment
            .coords
            .iter()
            .map(LatentCoordValues::effective_axis_periods)
            .collect();
        for row in 0..self.n_obs() {
            let w_row = row_w.map_or(1.0, |w| w[row]);
            let base = cache.row_offsets[row];
            match self.last_row_layout {
                Some(ref layout) => {
                    for (pos, &kk) in layout.active_atoms[row].iter().enumerate() {
                        if rho.log_ard[kk].is_empty() {
                            continue;
                        }
                        let start = layout.coord_starts[row][pos];
                        let coord = &self.assignment.coords[kk];
                        for axis in 0..coord.latent_dim() {
                            let alpha = ard_precisions[kk][axis];
                            let t = coord.row(row)[axis];
                            let hess = w_row
                                * ArdAxisPrior::eval(alpha, t, periods[kk][axis])
                                    .psd_majorizer_hess();
                            if hess == 0.0 {
                                continue;
                            }
                            let flat = rho.ard_flat_index(kk, axis);
                            let c = c_by_flat
                                .entry(flat)
                                .or_insert_with(|| Array2::<f64>::zeros((dim, dim)));
                            let g_idx = base + start + axis;
                            c[[g_idx, g_idx]] += hess;
                        }
                    }
                }
                None => {
                    for kk in 0..self.k_atoms() {
                        if rho.log_ard[kk].is_empty() {
                            continue;
                        }
                        let coord = &self.assignment.coords[kk];
                        for axis in 0..coord.latent_dim() {
                            let alpha = ard_precisions[kk][axis];
                            let t = coord.row(row)[axis];
                            let hess = w_row
                                * ArdAxisPrior::eval(alpha, t, periods[kk][axis])
                                    .psd_majorizer_hess();
                            if hess == 0.0 {
                                continue;
                            }
                            let flat = rho.ard_flat_index(kk, axis);
                            let c = c_by_flat
                                .entry(flat)
                                .or_insert_with(|| Array2::<f64>::zeros((dim, dim)));
                            let g_idx = base + coord_offsets[kk] + axis;
                            c[[g_idx, g_idx]] += hess;
                        }
                    }
                }
            }
        }

        // Sparse (assignment log-strength): whatever the single authority says the
        // installed logit-slot curvature's ρ-derivative is (#2500). Softmax reads
        // its Gershgorin majorizer `w_row·diag(Σ_j|H_kj|)`, the threshold gate its
        // exact `w_row·λ·s·(1−2a)/τ²` — both degree-one in `λ_sparse = e^ρ` exactly
        // like smoothing/ARD, so the derivative is the installed entry itself.
        if let Some(sparse_flat) = rho.sparse_flat_index() {
            match self.sparse_logit_curvature_rho_derivative(rho, cache)? {
                SparseLogitCurvature::Inert => {}
                SparseLogitCurvature::Diagonal(entries) => {
                    let c = c_by_flat
                        .entry(sparse_flat)
                        .or_insert_with(|| Array2::<f64>::zeros((dim, dim)));
                    for (slot, value) in entries {
                        c[[slot, slot]] += value;
                    }
                }
                SparseLogitCurvature::CrossRowOwnedElsewhere => {
                    // #2330: the ordered-Beta–Bernoulli sparse ∂A/∂ρ_sparse is the
                    // EXACT integrated-marginal logit Hessian (cross-row), supplied
                    // by `dense_exact_a_ordered_bb_sparse_trace`, NOT a diagonal
                    // majorizer operator this map can assemble. Emit nothing here
                    // (the dense-A gradient adds that coordinate's trace directly)
                    // rather than a wrong diagonal-only operator.
                }
            }
        }

        Ok(c_by_flat)
    }

    /// The ρ-derivative of the EXACT-minus-majorizer
    /// stationarity correction, `∂(ΔC)/∂ρ_i` where `ΔC = A − B`
    /// (`Self::apply_exact_hessian_minus_b`), keyed by flat coordinate. The IFT
    /// sensitivity `∂a/∂ρ_i = A⁺(∂Γ/∂ρ_i − (∂A/∂ρ_i)a)` differentiates the EXACT
    /// stationarity Hessian `A = B + ΔC`, not the majorized solver operator `B = H`
    /// (`raw_penalty_curvature_operators_by_flat` = `∂B/∂ρ`). So the `M_i·a` term must
    /// use `∂A/∂ρ_i = ∂B/∂ρ_i + ∂(ΔC)/∂ρ_i` — this map supplies the second piece.
    ///
    /// Both deltas are degree-one in their ρ (so `∂(ΔC)/∂ρ_i` is the delta itself)
    /// and mirror `apply_exact_hessian_minus_b`'s deltas exactly:
    /// * periodic ARD: `w_row·min(α cos κt, 0)` (the negative-part remainder the
    ///   `max(·,0)` majorizer drops) on the coord slot, ALL rows — nonzero only on
    ///   the inactive half `cos κt < 0`. This is the term the ARD-perturbed
    ///   `H3[ard,·]` rows need (the transposed smooth-perturbed rows, where
    ///   `∂A = ∂B`, are already exact).
    /// * softmax sparse: the exact entropy Hessian minus the Gershgorin majorizer
    ///   on the row's logit block (dense, off-diagonal + diagonal), `∝ λ_sparse`.
    /// * ThresholdGate sparse: the raw signed logit curvature minus its PSD
    ///   clamp (diagonal), degree-one in `lambda_sparse` — nonzero on exactly
    ///   the logits the gate has switched ON, where the clamp `B` installs is
    ///   a hard zero (#2520).
    /// Smooth is unmajorized (`ΔC` has no smooth part), so its delta is zero and it
    /// is absent from the map. Covered config only (softmax, dense row layout).
    pub(crate) fn exact_stationarity_penalty_derivative_delta_by_flat(
        &self,
        rho: &SaeManifoldRho,
        cache: &ArrowFactorCache,
    ) -> Result<std::collections::BTreeMap<usize, Array2<f64>>, String> {
        let total_t = cache.delta_t_len();
        let dim = total_t + cache.k;
        let k_atoms = self.k_atoms();
        let ard_precisions = self.validated_ard_precisions(rho)?;
        let row_w = self.row_loss_weights.as_deref();
        let ard_axis_periods: Vec<Vec<Option<f64>>> = self
            .assignment
            .coords
            .iter()
            .map(|coord| coord.effective_axis_periods())
            .collect();
        let softmax_delta: Option<(usize, f64)> = match self.assignment.mode {
            AssignmentMode::Softmax {
                temperature,
                sparsity,
            } if k_atoms > 1 => {
                let inv_tau = 1.0 / temperature;
                match rho.sparse_flat_index() {
                    Some(sparse_flat) => Some((
                        sparse_flat,
                        rho.lambda_sparse()? * sparsity * inv_tau * inv_tau,
                    )),
                    None => None,
                }
            }
            _ => None,
        };
        // #2520 - the ThresholdGate's exact-minus-majorizer remainder, the third
        // member of this map. `B` installs `smooth_psd_clamp(magnitude, 1-2a)`
        // and `dC` restores the raw signed `magnitude*(1-2a)`, so the remainder
        // is nonzero on exactly the logits the gate has switched ON. It is
        // degree-one in `lambda_sparse = e^rho` for the same reason the majorizer
        // is (the clamp is homogeneous in its prefactor and the prefactor carries
        // `lambda_sparse`), so `d(dC)/drho_sparse` IS the remainder itself -- the
        // identical argument the smooth and ARD channels use.
        //
        // Its absence is why the dense exact-A sparse log-det trace disagreed
        // with the finite difference of the value it differentiates IN SIGN on a
        // straddling gate: `dB/drho_sparse` is the clamp, which is a hard `0`
        // there, so the whole of `dA/drho_sparse` on those slots lives here.
        let threshold_gate_remainder: Option<(usize, Array1<f64>)> =
            match (self.assignment.mode, rho.sparse_flat_index()) {
                (AssignmentMode::ThresholdGate { .. }, Some(sparse_flat)) => Some((
                    sparse_flat,
                    crate::assignment::threshold_gate_negative_hessian_remainder_weighted(
                        &self.assignment,
                        rho,
                        row_w,
                    )?,
                )),
                _ => None,
            };
        let mut deltas: std::collections::BTreeMap<usize, Array2<f64>> =
            std::collections::BTreeMap::new();
        let mut assignments = Array1::<f64>::zeros(k_atoms);
        for row in 0..self.n_obs() {
            let base = cache.row_offsets[row];
            self.assignment.try_assignments_row_into(
                row,
                assignments
                    .as_slice_mut()
                    .expect("assignment scratch is contiguous"),
            )?;
            let vars = self.row_vars_for_cache_row(row, cache)?;
            let w_row = row_w.map_or(1.0, |w| w[row]);
            // Softmax entropy-minus-majorizer delta on the logit block.
            if let Some((sparse_flat, scale)) = softmax_delta {
                let assignment_dim = self.assignment.assignment_coord_dim();
                let a_soft = assignments
                    .as_slice()
                    .expect("softmax assignments row must be contiguous");
                let m = softmax_majorizer_log_mean(a_soft);
                let c = deltas
                    .entry(sparse_flat)
                    .or_insert_with(|| Array2::<f64>::zeros((dim, dim)));
                for (a, va) in vars.iter().enumerate() {
                    let SaeLocalRowVar::Logit { atom: ka } = *va else {
                        continue;
                    };
                    if ka >= assignment_dim {
                        continue;
                    }
                    for (b, vb) in vars.iter().enumerate() {
                        let SaeLocalRowVar::Logit { atom: kb } = *vb else {
                            continue;
                        };
                        if kb >= assignment_dim {
                            continue;
                        }
                        let h_entropy =
                            softmax_dense_entropy_hessian_entry(a_soft, ka, kb, m, scale);
                        let delta = if ka == kb {
                            h_entropy
                                - active_softmax_gershgorin_majorizer_entry(a_soft, ka, m, scale)
                        } else {
                            h_entropy
                        };
                        c[[base + a, base + b]] += w_row * delta;
                    }
                }
            }
            // ThresholdGate exact-minus-majorizer remainder on the logit slots.
            if let Some((sparse_flat, remainder)) = threshold_gate_remainder.as_ref() {
                for (a, va) in vars.iter().enumerate() {
                    let SaeLocalRowVar::Logit { atom } = *va else {
                        continue;
                    };
                    if atom >= k_atoms {
                        continue;
                    }
                    // Already `w_row`-weighted and fixed-logit-masked by the
                    // shared seam, so no second weighting here.
                    let neg = remainder[row * k_atoms + atom];
                    if neg != 0.0 {
                        let c = deltas
                            .entry(*sparse_flat)
                            .or_insert_with(|| Array2::<f64>::zeros((dim, dim)));
                        c[[base + a, base + a]] += neg;
                    }
                }
            }
            // Periodic-ARD negative-part remainder on the coord slots.
            for (a, va) in vars.iter().enumerate() {
                let SaeLocalRowVar::Coord { atom, axis } = *va else {
                    continue;
                };
                if rho.log_ard[atom].is_empty() {
                    continue;
                }
                let alpha = ard_precisions[atom][axis];
                let t_val = self.assignment.coords[atom].row(row)[axis];
                let neg = ArdAxisPrior::eval(alpha, t_val, ard_axis_periods[atom][axis])
                    .negative_hessian_remainder();
                if neg != 0.0 {
                    let flat = rho.ard_flat_index(atom, axis);
                    let c = deltas
                        .entry(flat)
                        .or_insert_with(|| Array2::<f64>::zeros((dim, dim)));
                    c[[base + a, base + a]] += w_row * neg;
                }
            }
        }
        Ok(deltas)
    }

    /// The complete frozen-state derivative of the raw exact stationarity
    /// Hessian, `A_raw = B_raw + ΔC`, keyed by flat outer coordinate.
    ///
    /// Keeping this sum behind one named owner makes it possible to compare the
    /// operator derivative directly with finite differences of
    /// [`Self::materialize_exact_hessian_dense_with_gap_border`], instead of validating only a
    /// downstream trace where spectral classification can obscure which operand
    /// drifted (#2515).
    pub(crate) fn exact_stationarity_penalty_derivatives_by_flat(
        &self,
        rho: &SaeManifoldRho,
        cache: &ArrowFactorCache,
    ) -> Result<std::collections::BTreeMap<usize, Array2<f64>>, String> {
        let mut derivatives = self.raw_penalty_curvature_operators_by_flat(rho, cache)?;
        for (flat, delta) in self.exact_stationarity_penalty_derivative_delta_by_flat(rho, cache)? {
            match derivatives.entry(flat) {
                std::collections::btree_map::Entry::Occupied(mut entry) => {
                    *entry.get_mut() += &delta;
                }
                std::collections::btree_map::Entry::Vacant(entry) => {
                    entry.insert(delta);
                }
            }
        }
        Ok(derivatives)
    }

    /// #2330 Patch D — the t--β residual-curvature second-derivative leg
    /// `⟨error_metric, ∂²(gate_kβ·φ_mβ)/∂θ_a∂θ_w⟩` (term-2 of `∂ΔC_tβ[a,β]/∂θ_w`;
    /// the term-1 `⟨jets.first(w), jets.beta_deriv(a,β)⟩` is added inline). The
    /// border channel `β = (atom kβ, basis mβ, output-vector)` gives
    /// `∂f_out/∂β = gate_kβ·φ_mβ·output_out`, so this leg is
    /// `eo · g_kβ^{(l)} · ∂^{2−l}φ_mβ` with `eo = Σ_out error_metric[out]·output[out]`,
    /// `l` the number of LOGIT derivatives among `{a,w}`, on the coord axes of the
    /// rest; nonzero only when `a,w` both touch `kβ`. `l≥1` uses the ordered-BB
    /// logistic-gate derivatives; skipped for other modes (softmax follow-on).
    /// #2330 Patch D — one row's `error_metric = √w·M·r` in OUTPUT space, the
    /// object `apply_exact_hessian_minus_b` contracts `ΔC` against.
    ///
    /// Built exactly as that assembler builds it: the fitted row is the
    /// assignment-weighted decode over this row's ACTIVE atoms, the residual is
    /// scaled by `√w` before the metric, and the whitening metric is applied only
    /// where the row jets are whitened — so a plain dot of this against a jet
    /// reconstitutes the same `w`-weighted `M`-inner product the assembly uses.
    ///
    /// #2515 — extracted so the dense θ-adjoint, its from-probes sibling, and the
    /// coordinate-block leg all build it ONCE rather than three times. A route
    /// that reconstructed the residual with a different weighting would produce a
    /// Patch-D leg that silently disagreed with the operator it is supposed to
    /// differentiate, which is the failure this whole front is about.
    pub(crate) fn patchd_row_error_metric(
        &self,
        row: usize,
        w_row: f64,
        target: ArrayView2<'_, f64>,
        assignments: &Array1<f64>,
        whiten_row_jets: bool,
    ) -> Vec<f64> {
        let p_out = self.output_dim();
        let sqrt_w = w_row.sqrt();
        let active_atoms = self
            .last_row_layout
            .as_ref()
            .map(|layout| layout.active_atoms[row].as_slice());
        let mut fitted = vec![0.0_f64; p_out];
        let mut decoded = vec![0.0_f64; p_out];
        for k in 0..self.k_atoms() {
            if active_atoms.is_some_and(|active| active.binary_search(&k).is_err()) {
                continue;
            }
            self.atoms[k].fill_decoded_row(row, &mut decoded);
            let a_k = assignments[k];
            for out in 0..p_out {
                fitted[out] += a_k * decoded[out];
            }
        }
        let mut err = Array1::<f64>::zeros(p_out);
        for out in 0..p_out {
            err[out] = sqrt_w * (fitted[out] - target[[row, out]]);
        }
        match self.row_metric.as_ref() {
            Some(metric) if whiten_row_jets => metric.apply_metric_row(row, err.view()),
            _ => err.to_vec(),
        }
    }

    fn patchd_residual_third_leg_beta(
        &self,
        ctx: &PatchDResidualCtx<'_>,
        a_var: SaeLocalRowVar,
        w_var: SaeLocalRowVar,
        ch: &SaeBorderChannel,
    ) -> f64 {
        let PatchDResidualCtx {
            row,
            error_metric,
            sqrt_w,
            assignments,
            second_jets,
            ..
        } = *ctx;
        let classify = |v: SaeLocalRowVar| -> (usize, Option<usize>) {
            match v {
                SaeLocalRowVar::Coord { atom, axis } => (atom, Some(axis)),
                SaeLocalRowVar::Logit { atom } => (atom, None),
            }
        };
        let (ka, aa) = classify(a_var);
        let (kw, aw) = classify(w_var);
        if (aa.is_some() && ka != ch.atom) || (aw.is_some() && kw != ch.atom) {
            return 0.0;
        }
        let atom_idx = ch.atom;
        let m = ch.basis_col;
        let mut coord_axes: Vec<usize> = Vec::with_capacity(2);
        let mut logit_count = 0usize;
        for opt in [aa, aw] {
            match opt {
                Some(axis) => coord_axes.push(axis),
                None => logit_count += 1,
            }
        }
        let atom = &self.atoms[atom_idx];
        // ∂^{2−l}φ_m over the coord axes.
        let phi = match coord_axes.len() {
            2 => second_jets[atom_idx][[row, m, coord_axes[0], coord_axes[1]]],
            1 => atom.basis_jacobian[[row, m, coord_axes[0]]],
            _ => atom.basis_values[[row, m]],
        };
        let s = assignments[atom_idx];
        let gate_factor = match self.assignment.mode {
            AssignmentMode::Softmax { temperature, .. } => {
                let delta = |i, j| if i == j { 1.0 } else { 0.0 };
                match logit_count {
                    0 => s,
                    1 => {
                        let j = if aa.is_none() { ka } else { kw };
                        s * (delta(atom_idx, j) - assignments[j]) / temperature
                    }
                    _ => s * ((delta(atom_idx, ka) - assignments[ka])
                        * (delta(atom_idx, kw) - assignments[kw])
                        - assignments[ka] * (delta(ka, kw) - assignments[kw]))
                        / (temperature * temperature),
                }
            }
            AssignmentMode::OrderedBetaBernoulli { temperature, .. }
            | AssignmentMode::ThresholdGate { temperature, .. } => {
                if ka != atom_idx || kw != atom_idx { return 0.0; }
                match logit_count {
                    0 => s,
                    1 => s * (1.0 - s) / temperature,
                    _ => s * (1.0 - s) * (1.0 - 2.0 * s) / (temperature * temperature),
                }
            }
            AssignmentMode::TopK { .. } => if logit_count == 0 { s } else { 0.0 },
        };
        // eo = Σ_out error_metric[out]·output[out] (the channel's output weighting).
        let p = error_metric.len().min(ch.output.len());
        let mut eo = 0.0_f64;
        for out in 0..p {
            eo += error_metric[out] * ch.output[out];
        }
        sqrt_w * gate_factor * phi * eo
    }

    /// #2330 Patch D — the exact-A residual-curvature THIRD-derivative leg
    /// `⟨error_metric, ∂³f_{a,b,w}⟩`, the second half of `∂ΔC_tt[a,b]/∂θ_w`
    /// (the first half `⟨∂error_metric/∂θ_w, ∂²f⟩ = ⟨jets.first(w), jets.second(a,b)⟩`
    /// is added inline as term 1a). The data fit is `½rᵀMr` so its residual
    /// curvature is `⟨M r, ∂²f⟩`; differentiating the SECOND-jet factor gives this
    /// leg. For the per-atom gated decoder `f_out = Σ_k g_k(ℓ_k)·Σ_m B_k[m,out]·φ_m(x_k)`,
    /// `∂³f` is nonzero only when `a,b,w` all touch ONE atom `k` (each summand
    /// depends only on that atom's `(x_k, ℓ_k)` — exact for ordered-Beta–Bernoulli
    /// where `g_k` depends on `ℓ_k` alone). It then factors as `g_k^{(l)} · Σ_m
    /// B_k[m,out]·∂^{c}φ_m` where `l` = number of LOGIT derivatives among `{a,b,w}`
    /// and `c = 3−l` = number of COORD derivatives (over their axes). The `l=0`
    /// coord³ leg uses the plain gate value and holds for ANY mode; the `l≥1` legs
    /// use the ordered-Beta–Bernoulli logistic-gate derivatives and are skipped
    /// (returns 0) for other modes — softmax's cross-atom gate third-order is a
    /// separate follow-on. `error_metric` already carries one `√w·M`; this leg
    /// carries the other `√w`, matching the `⟨error_metric, jets.second⟩`
    /// convention exactly.
    fn patchd_residual_third_leg(
        &self,
        ctx: &PatchDResidualCtx<'_>,
        a_var: SaeLocalRowVar,
        b_var: SaeLocalRowVar,
        w_var: SaeLocalRowVar,
    ) -> f64 {
        let PatchDResidualCtx {
            row,
            error_metric,
            sqrt_w,
            assignments,
            second_jets,
            third_jets,
            is_obb,
            inv_tau,
        } = *ctx;
        // Classify each var as (atom, Some(axis)) for a coordinate or
        // (atom, None) for a logit; all three must share ONE atom.
        let classify = |v: SaeLocalRowVar| -> (usize, Option<usize>) {
            match v {
                SaeLocalRowVar::Coord { atom, axis } => (atom, Some(axis)),
                SaeLocalRowVar::Logit { atom } => (atom, None),
            }
        };
        let (ka, aa) = classify(a_var);
        let (kb, ab) = classify(b_var);
        let (kw, aw) = classify(w_var);
        if ka != kb || ka != kw {
            return 0.0;
        }
        let atom_idx = ka;
        // Collect coord axes; count logit derivatives.
        let mut coord_axes: Vec<usize> = Vec::with_capacity(3);
        let mut logit_count = 0usize;
        for opt in [aa, ab, aw] {
            match opt {
                Some(axis) => coord_axes.push(axis),
                None => logit_count += 1,
            }
        }
        if logit_count > 0 && !is_obb {
            // Non-OBB gate third-order (softmax cross-atom) is a follow-on;
            // the l==0 basis third jet still applies to any mode.
            return 0.0;
        }
        let atom = &self.atoms[atom_idx];
        let basis = atom.basis_size();
        let decoder = atom.decoder_coefficients(); // (basis, out)
        let p = error_metric.len();
        // D_c[out] = Σ_m B[m,out] · ∂^c φ_m over the collected coord axes.
        let mut d_c = vec![0.0_f64; p];
        match coord_axes.len() {
            3 => {
                let Some(tj) = third_jets.and_then(|t| t[atom_idx].as_ref()) else {
                    return 0.0; // no analytic third jet for this atom
                };
                let (a0, a1, a2) = (coord_axes[0], coord_axes[1], coord_axes[2]);
                for m in 0..basis {
                    let phi3 = tj[[row, m, a0, a1, a2]];
                    for out in 0..p {
                        d_c[out] += decoder[[m, out]] * phi3;
                    }
                }
            }
            2 => {
                let sj = &second_jets[atom_idx];
                let (a0, a1) = (coord_axes[0], coord_axes[1]);
                for m in 0..basis {
                    let phi2 = sj[[row, m, a0, a1]];
                    for out in 0..p {
                        d_c[out] += decoder[[m, out]] * phi2;
                    }
                }
            }
            1 => {
                let a0 = coord_axes[0];
                for m in 0..basis {
                    let phi1 = atom.basis_jacobian[[row, m, a0]];
                    for out in 0..p {
                        d_c[out] += decoder[[m, out]] * phi1;
                    }
                }
            }
            _ => {
                for m in 0..basis {
                    let phi0 = atom.basis_values[[row, m]];
                    for out in 0..p {
                        d_c[out] += decoder[[m, out]] * phi0;
                    }
                }
            }
        }
        // Gate factor g^{(l)}: l logit derivatives of the atom's gate. For OBB
        // g = σ(ℓ/τ): g0=s, g1=s(1−s)/τ, g2=s(1−s)(1−2s)/τ², g3=s(1−s)(1−6s+6s²)/τ³.
        let s = assignments[atom_idx];
        let gate_factor = match logit_count {
            0 => s,
            1 => s * (1.0 - s) * inv_tau,
            2 => s * (1.0 - s) * (1.0 - 2.0 * s) * inv_tau * inv_tau,
            _ => s * (1.0 - s) * (1.0 - 6.0 * s + 6.0 * s * s) * inv_tau * inv_tau * inv_tau,
        };
        let mut acc = 0.0_f64;
        for out in 0..p {
            acc += error_metric[out] * d_c[out];
        }
        sqrt_w * gate_factor * acc
    }

    /// Dense reconstruction of the θ-adjoint `Γ_w = tr(inv · ∂H/∂θ_w)` against an
    /// arbitrary dense joint inverse `inv` (`dim×dim` over the `(t, β)` blocks).
    pub(crate) fn logdet_theta_adjoint_dense(
        &self,
        rho: &SaeManifoldRho,
        cache: &ArrowFactorCache,
        inv: &Array2<f64>,
        skip_deflation_dk: bool,
        exact_a: bool,
        // #2330 Patch D — the data target, required ONLY for the exact-A
        // residual-curvature third-derivative leg `⟨error_metric, ∂³f⟩`. `None`
        // reproduces the pre-Patch-D behaviour exactly (the leg is skipped), so
        // every non-exact-A caller passes `None`.
        residual_target: Option<ArrayView2<'_, f64>>,
    ) -> Result<SaeArrowVector, String> {
        // #2330 — `skip_deflation_dk` drops the Daleckii–Krein deflation
        // correction, leaving the raw trace contraction (a deflation-blind
        // counterfactual for tests). Production callers pass `false`.
        let ard_precisions = self.validated_ard_precisions(rho)?;
        let total_t = cache.delta_t_len();
        let k = cache.k;
        let k_atoms = self.k_atoms();
        let n = self.n_obs();
        let mut gamma_t = Array1::<f64>::zeros(total_t);
        let mut gamma_beta = Array1::<f64>::zeros(k);
        let second_jets = self.atom_second_jets()?;
        let border = self.border_channels_for_cache(cache)?;
        let whiten_row_jets = self.whiten_logdet_row_jets();
        // `1/τ` (always, for the softmax data-weight logit factor) and the
        // entropy Gershgorin majorizer scale `λ_sparse·s/τ²` (only a live free
        // logit, i.e. `k_atoms > 1`, carries the sparsity penalty).
        let (entropy_scale, inv_tau) = match self.assignment.mode {
            AssignmentMode::Softmax {
                temperature,
                sparsity,
            } => {
                let inv_tau = 1.0 / temperature;
                let scale = if k_atoms > 1 {
                    rho.lambda_sparse()? * sparsity * inv_tau * inv_tau
                } else {
                    0.0
                };
                (scale, inv_tau)
            }
            _ => (0.0, 0.0),
        };
        // #2330 Patch D residual-curvature leg setup. Active only on the exact-A
        // route with a target: builds `∂³f` from raw basis jets + gate
        // derivatives (see `patchd_residual_third_leg`).
        let patchd_residual = exact_a.then_some(residual_target).flatten();
        let patchd_third_jets = if patchd_residual.is_some() {
            Some(self.atom_third_jets()?)
        } else {
            None
        };
        let patchd_is_obb = matches!(
            self.assignment.mode,
            AssignmentMode::OrderedBetaBernoulli { .. }
        );
        // Full ordered-BB prior curvature: the row loop below contributes only
        // softmax entropy, so this owner must include both the positive local
        // diagonal and the exact-minus-majorizer remainder.
        let patchd_obb_adjoint = if patchd_residual.is_some() && patchd_is_obb {
            crate::assignment::ordered_beta_bernoulli_logit_adjoint_data_weighted(
                &self.assignment,
                rho,
                self.row_loss_weights.as_deref(),
            )?
        } else {
            None
        };
        let patchd_obb_inv_tau = match self.assignment.mode {
            AssignmentMode::OrderedBetaBernoulli { temperature, .. } => 1.0 / temperature,
            _ => 0.0,
        };
        let mut jet_window: std::collections::VecDeque<SaeRowJets> =
            std::collections::VecDeque::new();
        let mut jet_window_next = 0usize;
        let mut assignments = Array1::<f64>::zeros(k_atoms);
        for row in 0..n {
            let q = cache.row_dims[row];
            let base = cache.row_offsets[row];
            let a_scratch = assignments.as_slice_mut().expect("contiguous scratch");
            self.assignment.try_assignments_row_into(row, a_scratch)?;
            if jet_window.is_empty() {
                jet_window_next = self.refill_jet_window(
                    jet_window_next,
                    cache,
                    &second_jets,
                    &border,
                    &mut jet_window,
                )?;
            }
            let mut jets = jet_window
                .pop_front()
                .ok_or_else(|| "logdet_theta_adjoint_dense: empty jet window".to_string())?;
            if whiten_row_jets {
                self.apply_whiten_to_logdet_row_jets(row, &mut jets)?;
            }
            let a_soft = assignments
                .as_slice()
                .expect("softmax assignments row must be contiguous");
            let m_log_mean = softmax_majorizer_log_mean(a_soft);
            let simplex_count = crate::assignment::simplex_gate_free_count(&self.assignment);
            let w_row = self.row_loss_weights.as_deref().map_or(1.0, |w| w[row]);
            // #2330 Patch D — per-row `error_metric = √w·M·r` in output space,
            // built EXACTLY as `apply_exact_hessian_minus_b` builds the object it
            // contracts ΔC against (√w residual, then whitening metric applied).
            let patchd_error_metric: Option<Vec<f64>> = patchd_residual.map(|tgt| {
                self.patchd_row_error_metric(row, w_row, tgt, &assignments, whiten_row_jets)
            });
            let patchd_sqrt_w = w_row.sqrt();
            let patchd_ctx: Option<PatchDResidualCtx<'_>> =
                patchd_error_metric.as_deref().map(|em| PatchDResidualCtx {
                    row,
                    error_metric: em,
                    sqrt_w: patchd_sqrt_w,
                    assignments: &assignments,
                    second_jets: &second_jets,
                    third_jets: patchd_third_jets.as_deref(),
                    is_obb: patchd_is_obb,
                    inv_tau: patchd_obb_inv_tau,
                });
            // #2308 — per-row spectral/gauge deflation the criterion factor applied.
            // It is FROZEN at the fixed stratum (the radial-gauge / ARD-inactive-half
            // null is ρ-invariant), so contracting the DEFLATED inverse `inv` and
            // subtracting the SAME Daleckii–Krein correction the production θ-adjoint
            // subtracts makes `Γ(inv)` — and its twist `Γ(−G Mᵢ G)` — match the
            // gradient on the deflated circle route (where deflation is the norm, not
            // an error). `deflation_block_correction` is linear in `inv`, so the twist
            // rides through it exactly.
            let defl_dirs = cache
                .deflated_row_directions
                .get(row)
                .map(Vec::as_slice)
                .unwrap_or(&[]);
            let defl_spectrum = cache
                .deflation_row_spectra
                .get(row)
                .and_then(Option::as_ref);
            let defl_live = Self::row_deflation_is_live(defl_dirs, defl_spectrum);
            let inv_vv_block = if !defl_live {
                Array2::<f64>::zeros((0, 0))
            } else {
                inv.slice(s![base..base + q, base..base + q]).to_owned()
            };
            for w in 0..q {
                let logit_w = match jets.vars[w] {
                    SaeLocalRowVar::Logit { atom } => Some(atom),
                    SaeLocalRowVar::Coord { .. } => None,
                };
                // The exact operator's entropy block moves with the logit through the
                // dense entropy third derivative; one O(K) setup per logit variable.
                let softmax_entropy_derivative = match logit_w {
                    Some(atom_w) if exact_a && entropy_scale != 0.0 => Some(
                        SoftmaxEntropyDerivative::new(
                            a_soft,
                            atom_w,
                            m_log_mean,
                            entropy_scale,
                            inv_tau,
                        ),
                    ),
                    _ => None,
                };
                let mut gamma = 0.0_f64;
                let mut dh_mat = if !defl_live {
                    Array2::<f64>::zeros((0, 0))
                } else {
                    Array2::<f64>::zeros((q, q))
                };
                for a in 0..q {
                    for b in 0..q {
                        let mut dh = 0.0_f64;
                        dh += match (logit_w, jets.vars[a], jets.vars[b]) {
                            (
                                Some(atom_w),
                                SaeLocalRowVar::Coord { atom: atom_a, .. },
                                SaeLocalRowVar::Coord { atom: atom_b, .. },
                            ) => {
                                sae_dot(jets.first(a), jets.first(b))
                                    * (Self::softmax_data_weight_product_logit_factor(
                                        a_soft, atom_a, atom_b, atom_w, inv_tau,
                                    ) + if patchd_is_obb {
                                        // #2330 / #2371 -- ordered-Beta--Bernoulli gate
                                        // gradient of the GN curvature. `B[a,b] = <J_a, J_b>`
                                        // and each leg `J_k` carries its INDEPENDENT gate
                                        // `g_k = sigma(l_k/tau)` linearly, so
                                        // `dB/dl_w = [1(w==a) + 1(w==b)] * (1-g_w)/tau * B`.
                                        // The matching leg gate is `g_w`, so a single
                                        // `(1 - a_soft[atom_w])` is correct per side:
                                        // same-atom-both gives sided=2 (bitwise the prior
                                        // landed value), one-sided cross-atom gives sided=1
                                        // (the #2371 term wrongly dropped as exactly zero).
                                        // The softmax factor above is 0 here (`inv_tau` is
                                        // 0 for non-softmax modes), so softmax is unchanged.
                                        let sided = (atom_w == atom_a) as u32
                                            + (atom_w == atom_b) as u32;
                                        sided as f64
                                            * (1.0 - a_soft[atom_w])
                                            * patchd_obb_inv_tau
                                    } else {
                                        0.0
                                    })
                            }
                            _ => {
                                sae_dot(jets.second(a, w), jets.first(b))
                                    + sae_dot(jets.first(a), jets.second(b, w))
                            }
                        };
                        if let Some(ctx) = patchd_ctx.as_ref() {
                            dh += self.patchd_residual_third_leg(
                                ctx,
                                jets.vars[a],
                                jets.vars[b],
                                jets.vars[w],
                            );
                        }
                        if exact_a {
                            // #2330 Patch D (1a) — `A = B + ΔC` carries the residual
                            // curvature `ΔC_tt[a,b] = ⟨error_metric, ∂²f_ab⟩` that the
                            // Gauss-Newton assembly drops, and that block moves with
                            // `θ_w` too:
                            //   `∂ΔC_tt[a,b]/∂θ_w = ⟨∂error_metric/∂θ_w, ∂²f_ab⟩`
                            //                      `+ ⟨error_metric, ∂³f_abw⟩`.
                            // `∂error_metric/∂θ_w = √w·M·∂f/∂θ_w`, which in THIS
                            // function's jet convention is exactly `jets.first(w)`:
                            // every jet carries one `√w` and (under whitening) one
                            // metric factor `L`, so a plain dot of two jets
                            // reconstitutes the `w`-weighted `M`-inner product the
                            // assembly uses. Only the FIRST leg lands here; the
                            // third-jet leg `⟨error_metric, ∂³f_abw⟩` needs a jet
                            // channel `SaeRowJets` does not expose.
                            dh += sae_dot(jets.first(w), jets.second(a, b));
                        }
                        if let (
                            Some(atom_w),
                            SaeLocalRowVar::Logit { atom: atom_a },
                            SaeLocalRowVar::Logit { atom: atom_b },
                        ) = (logit_w, jets.vars[a], jets.vars[b])
                        {
                            if exact_a {
                                // #2333 — `A` carries the exact dense entropy Hessian on
                                // the logit block: `B`'s Gershgorin majorizer `D̃` plus the
                                // ΔC remainder `h_entropy − D̃` on the diagonal and
                                // `h_entropy` off it. Its logit derivative is the dense
                                // entropy third derivative, off-diagonal pairs included.
                                if let Some(derivative) = softmax_entropy_derivative.as_ref() {
                                    dh += w_row * derivative.entry(atom_a, atom_b).1;
                                }
                            } else if atom_a == atom_b {
                                dh += w_row
                                    * active_softmax_majorizer_logit_derivative_entry(
                                        a_soft,
                                        atom_a,
                                        atom_w,
                                        m_log_mean,
                                        entropy_scale,
                                        inv_tau,
                                    );
                            }
                            // #2080 — the softmax row's logit Jacobian has the exact dense
                            // curvature `c·(diag z − zzᵀ)/τ²` in both `B` and `A`.
                            if let Some(count) = simplex_count {
                                if !self.assignment.logit_is_fixed(atom_a)
                                    && !self.assignment.logit_is_fixed(atom_b)
                                    && !self.assignment.logit_is_fixed(atom_w)
                                {
                                    dh += w_row
                                        * crate::assignment::simplex_gate_logit_jacobian_third(
                                            a_soft, atom_a, atom_b, atom_w, count, inv_tau,
                                        );
                                }
                            }
                        }
                        if a == b && a == w {
                            // #2080 — the gate prior's logit Jacobian has the exact curvature
                            // `2z(1 − z)/τ²` in both `B` and `A`.
                            if let SaeLocalRowVar::Logit { atom } = jets.vars[a] {
                                dh += crate::assignment::gate_logit_jacobian_third_weighted(
                                    &self.assignment,
                                    self.row_loss_weights.as_deref(),
                                    row,
                                    atom,
                                );
                            }
                            if let SaeLocalRowVar::Coord { atom, axis } = jets.vars[a] {
                                if !ard_precisions[atom].is_empty() {
                                    dh += if exact_a {
                                        self.ard_exact_hessian_derivative(
                                            ard_precisions[atom][axis],
                                            row,
                                            atom,
                                            axis,
                                        )
                                    } else {
                                        self.ard_majorized_hessian_derivative(
                                            ard_precisions[atom][axis],
                                            row,
                                            atom,
                                            axis,
                                        )
                                    };
                                }
                            }
                        }
                        if defl_live {
                            dh_mat[[a, b]] = dh;
                        }
                        gamma += inv[[base + b, base + a]] * dh;
                    }
                }
                if defl_live && !skip_deflation_dk {
                    gamma -= Self::deflation_block_correction(
                        &inv_vv_block,
                        &dh_mat,
                        defl_dirs,
                        defl_spectrum,
                    );
                }
                for a in 0..q {
                    for (beta_pos, ch) in border.iter().enumerate() {
                        // #2330 Patch D (1a), t--beta leg: `ΔC_tβ[a,β] =
                        // ⟨error_metric, ∂²f_aβ⟩` moves with `θ_w` through the
                        // residual exactly as the t--t block does.
                        let mut dh = sae_dot(jets.second(a, w), jets.beta(beta_pos))
                            + sae_dot(jets.first(a), jets.beta_deriv(w, beta_pos))
                            + if exact_a {
                                sae_dot(jets.first(w), jets.beta_deriv(a, beta_pos))
                            } else {
                                0.0
                            };
                        if let Some(ctx) = patchd_ctx.as_ref() {
                            dh += self.patchd_residual_third_leg_beta(
                                ctx,
                                jets.vars[a],
                                jets.vars[w],
                                ch,
                            );
                        }
                        gamma += 2.0 * inv[[base + a, total_t + ch.index]] * dh;
                    }
                }
                for (beta_i, ch_i) in border.iter().enumerate() {
                    for (beta_j, ch_j) in border.iter().enumerate() {
                        let dh = sae_dot(jets.beta_deriv(w, beta_i), jets.beta(beta_j))
                            + sae_dot(jets.beta(beta_i), jets.beta_deriv(w, beta_j));
                        gamma += inv[[total_t + ch_i.index, total_t + ch_j.index]] * dh;
                    }
                }
                gamma_t[base + w] = gamma;
            }
            for (w_beta_pos, w_channel) in border.iter().enumerate() {
                let mut gamma = 0.0_f64;
                let mut dh_mat = if !defl_live {
                    Array2::<f64>::zeros((0, 0))
                } else {
                    Array2::<f64>::zeros((q, q))
                };
                for a in 0..q {
                    for b in 0..q {
                        let mut dh = sae_dot(jets.beta_l_deriv(a, w_beta_pos), jets.first(b))
                            + sae_dot(jets.first(a), jets.beta_l_deriv(b, w_beta_pos));
                        if exact_a {
                            dh += sae_dot(jets.beta(w_beta_pos), jets.second(a, b));
                        }
                        if let Some(ctx) = patchd_ctx.as_ref() {
                            dh += self.patchd_residual_third_leg_beta(
                                ctx, jets.vars[a], jets.vars[b], w_channel,
                            );
                        }
                        if defl_live {
                            dh_mat[[a, b]] = dh;
                        }
                        gamma += inv[[base + b, base + a]] * dh;
                    }
                }
                if defl_live && !skip_deflation_dk {
                    gamma -= Self::deflation_block_correction(
                        &inv_vv_block,
                        &dh_mat,
                        defl_dirs,
                        defl_spectrum,
                    );
                }
                for a in 0..q {
                    for (beta_pos, ch) in border.iter().enumerate() {
                        let mut dh = sae_dot(jets.beta_l_deriv(a, w_beta_pos), jets.beta(beta_pos));
                        if exact_a {
                            dh += sae_dot(jets.beta(w_beta_pos), jets.beta_deriv(a, beta_pos));
                        }
                        gamma += 2.0 * inv[[base + a, total_t + ch.index]] * dh;
                    }
                }
                gamma_beta[w_channel.index] += gamma;
            }
        }
        if exact_a {
            gamma_beta += &self.exact_decoder_prior_theta_trace(
                cache, inv.slice(s![total_t.., total_t..]),
            )?;
        }
        // Fold the entire ordered-BB prior derivative into the logit slots.
        if let Some(data) = patchd_obb_adjoint.as_ref() {
            let obb = self.dense_exact_a_ordered_bb_logit_theta_adjoint(cache, inv, data)?;
            gamma_t += &obb;
        }
        Ok(SaeArrowVector {
            t: gamma_t,
            beta: gamma_beta,
        })
    }

    /// #2080 forward plumbing — the analytic outer-ρ gradient with an OPTIONAL
    /// low-rank representation of the reduced-logdet derivative.
    ///
    /// #2515 — `evidence` is a [`BundleEvidenceGeometry`], not a bare probe pair:
    /// its variant NAMES the operator whose selected inverse the from-probes
    /// channels contract, and the exact-`A` variant carries that operator's own
    /// factor cache. `cache` stays the `B` stationarity geometry on every route.
    ///
    /// When `evidence` is `Some`, the THREE reduced-logdet channels
    /// that have matrix-free siblings — the per-atom decoder smoothness EDF
    /// `tr(H⁻¹ M_k)`, the per-(atom,axis) ARD log-precision Hessian trace
    /// `½tr(H⁻¹ ∂H/∂logα)`, and the #1006 envelope Γ = tr(H⁻¹ ∂H/∂θ) — are evaluated
    /// off that bundle (`decoder_smoothness_effective_dof_per_atom_from_probes` /
    /// `ard_log_precision_hessian_trace_from_probes` / `logdet_theta_adjoint_from_probes`)
    /// instead of the dense `DeflatedArrowSolver` selected inverse. For the
    /// rational route the two slices are the identical weighted vectors emitted
    /// by `RationalLogdetPlan::into_directional_derivative_bundle`, so every
    /// contraction is the derivative of the SAME shifted rational value, not a
    /// separately sampled `S^-1`. They convert
    /// together as ONE all-or-nothing cluster on the single `Some` (invariant #1):
    /// never a partial mix within a single eval. Each from-probes channel PRICES
    /// deflated rows (#2712): `A_i⁻¹ + G_i S⁻¹ G_iᵀ` built on the conditioned row
    /// Cholesky IS the deflated block, so each applies the same Daleckii–Krein
    /// correction its dense sibling applies, and no fit is routed to the dense
    /// channel for carrying deflation.
    ///
    /// The complete all-coordinate assembler is single-adjoint (#2080-A): the IFT
    /// correction `−½·⟨Γ, A⁺ g_ρ_l⟩` over every outer coordinate collapses to ONE
    /// exact-stationarity solve `a = A⁺Γ` plus O(K) cheap `⟨a, g_ρ_l⟩`
    /// contractions (self-adjointness of `A⁺`; see the collapse below). That
    /// single adjoint solve is the ONLY solver-bound step, so the whole assembler
    /// runs matrix-free at massive K: pass `matrix_free_system = Some(system)` to
    /// route it through [`Self::solve_exact_stationarity_matrix_free`] (the
    /// reduced-Schur CG on the reassembled undamped operator) with
    /// `solver = DeflatedArrowSolver::plain(cache)` for the cheap per-row
    /// `coordinate_block_*` subtractions — the K≥4096, direct-logdet-not-admitted
    /// route, mirroring the matrix-free branch of this complete assembler.
    /// Pass `matrix_free_system = None` to use the dense [`DeflatedArrowSolver`]
    /// adjoint (the direct-logdet-admitted route). Both produce the same complete
    /// derivative; the from-probes trace channels and the matrix-free adjoint
    /// convert together as one all-or-nothing matrix-free cluster (invariant #1).
    pub(crate) fn analytic_outer_rho_gradient_components_with_bundle(
        &self,
        target: ArrayView2<'_, f64>,
        rho: &SaeManifoldRho,
        loss: &SaeManifoldLoss,
        cache: &ArrowFactorCache,
        solver: &DeflatedArrowSolver<'_>,
        evidence: Option<BundleEvidenceGeometry<'_>>,
        matrix_free_system: Option<&ArrowSchurSystem>,
    ) -> Result<SaeOuterRhoGradientComponents, OuterGradientError> {
        self.assignment
            .validate_rho_domain(rho)
            .map_err(OuterGradientError::internal)?;
        // #2515 — resolve the evidence geometry ONCE. `logdet_derivative_bundle`
        // is the probe pair every from-probes channel contracts; `evidence_cache`
        // is the factor cache whose row blocks those channels reconstruct the
        // arrow inverse from; `evidence_operator` is which operator's ρ/θ
        // derivative the curvature channels differentiate. All three come from
        // one value, so they cannot name different operators.
        //
        // No bundle ⇒ the dense `DeflatedArrowSolver` selected inverse on the
        // caller's `cache`, which is `B`. A bundle ⇒ the exact observed
        // information, carried with its own cache.
        let logdet_derivative_bundle = evidence
            .as_ref()
            .map(|geometry| (geometry.probes, geometry.sinv));
        let evidence_cache = evidence.as_ref().map_or(cache, |geometry| geometry.cache);
        let evidence_operator = evidence
            .as_ref()
            .map_or(EvidenceOperator::Majorizer, |geometry| geometry.operator);
        let n_params = rho.to_flat().len();
        let mut explicit = Array1::<f64>::zeros(n_params);
        let mut logdet_trace = Array1::<f64>::zeros(n_params);
        let mut occam = Array1::<f64>::zeros(n_params);
        let mut third_order_correction = Array1::<f64>::zeros(n_params);
        let rank_charge = self
            .production_rank_charge_derivative(target, rho, loss, cache)
            .map_err(OuterGradientError::internal)?;
        // #2330 Phase-2 / #2333 — which majorizer the logdet channels belong to
        // is a property of the ROUTE, and it is known here, before any of them is
        // produced. On the dense direct-logdet route the ranked term is ½log|A|,
        // so `logdet_trace` and `Γ` come from `dense_exact_a_logdet_channels`
        // and every B-majorizer producer below would be discarded unread: a
        // selected-inverse pass for the smoothness EDF, two solver-bound
        // assignment log-strength traces, the ARD Hessian traces, and two
        // θ-adjoint towers. Deciding once, up front, is what lets them be
        // skipped rather than computed and overwritten. The `explicit`, `occam`
        // and rank-charge-direct channels are majorizer-independent and are
        // produced on every route.
        // #2500 — and only where the dense θ-adjoint reconstruction MODELS this
        // fit's assignment family. Taking the exact-A arm with a `Γ` that is not
        // the θ-derivative of `A` would trade the B-route's staged value↔gradient
        // gap for an outright wrong gradient; deciding it HERE, with the rest of
        // the route, is also what keeps the B-majorizer producers alive for a
        // family that needs them.
        let exact_a_logdet_route = logdet_derivative_bundle.is_none()
            && matrix_free_system.is_none()
            && self.dense_exact_a_theta_adjoint_is_modelled();

        // #2087/#2330 ROUTE-COHERENCE GUARD. The VALUE's log-determinant route and
        // THIS gradient's are selected by two unrelated predicates:
        //
        //   value    `streaming_plan().admitted_or_error(..).direct_logdet_admitted()`
        //            — a working-set/memory admission (construction_quasi_laplace.rs).
        //   gradient `exact_a_logdet_route` above — the bundle / matrix-free /
        //            assignment-family triple. Nothing consults the value's admission.
        //
        // ⚠ WHAT EACH VALUE ROUTE PRICES CHANGED UNDER #2509 PHASE-2b (`5563a2a18`),
        // AND THIS GUARD'S PREMISE DID NOT. Until then, "not admitted ⇒ delegates to
        // the streaming implementation, which ranks the majorizer `½log|B|`" — which
        // is what the rest of this comment was written against. Phase-2b moved BOTH
        // branches of `streaming_exact_arrow_log_det_with_lane_and_system` onto the
        // exact observed information via `exact_a_evidence_system`, so TODAY both
        // value routes price `½log|A|` and the sentence is false. It is corrected
        // rather than deleted because the guard below still reads the predicate it
        // named, and a reader comparing the two needs to know which era each
        // describes.
        //
        // Nothing tied the two predicates, so a fit whose value was priced by the
        // streaming lane could still be handed an exact-A derivative here. The desync
        // was `½·d/dρ log|I + B⁻¹ΔC|` — unbounded on a near-singular `A`, and invisible
        // to a value-side identity check, which re-derives the VALUE predicate and
        // so cannot observe that the gradient took the other route. Refuse rather
        // than return the derivative of an operator the value never ranked.
        //
        // SCOPE — this refuses ONLY the cell (value = B, gradient = A). Post-Phase-2b
        // no production value route prices `B`, so the cell this fires on is now
        // reachable only by a caller that hand-picks the streaming entry on a shape
        // the plan would have admitted; it is retained because that caller exists
        // (see `tests_streaming_outer_gradient_2026`) and because a future route that
        // reintroduces a `B`-priced value must not silently acquire an `A` gradient.
        //
        // THE MIRROR CELL IS CLOSED. It used to be the live one: `exact_a_logdet_route`
        // is false whenever a derivative bundle or a matrix-free system is present, so
        // every streaming/bundle evaluation paired an `A`-priced VALUE with
        // `B`-differentiated trace and θ-adjoint channels. That was #2515's open half,
        // and on its fixture it cost `logdet_trace` gaps of `1.231477e-1` (smooth atom 0)
        // and `5.052577e-2` (ARD). The bundle route now carries a
        // `BundleEvidenceGeometry` naming the exact observed information and carrying its
        // OWN factor cache, so `exact_a_logdet_route` being false no longer means the
        // gradient prices `B` — it means the exact-`A` channels are assembled from the
        // bundle rather than from the dense pseudo-inverse. Measured route-parity at a
        // fixed state, complete gradient: `1.57e-14`
        // (`laplace_value_and_gradient_are_route_invariant_2515`).
        //
        // So this guard is now the ONLY value/gradient pairing that can still go wrong,
        // and it is checked. Its scope note below is unchanged.
        //
        // LIMIT — this guard reads the PLAN, so it catches a predicate mismatch, not
        // a caller that hand-picks `penalized_quasi_laplace_criterion_streaming_exact_with_cache`
        // on a shape the plan would have admitted (see
        // `tests_streaming_outer_gradient_2026`, which does exactly that on purpose).
        if exact_a_logdet_route {
            let value_route_is_exact_a = self
                .streaming_plan()
                .map_err(OuterGradientError::internal)?
                .admitted_or_error(self.n_obs(), self.output_dim(), self.k_atoms())
                .map_err(OuterGradientError::internal)?
                .direct_logdet_admitted();
            if !value_route_is_exact_a {
                return Err(OuterGradientError::internal(format!(
                    "analytic_outer_rho_gradient_components_with_bundle: log-determinant route \
                     incoherence — this gradient would differentiate the exact ½log|A|, \
                     but at shape n={}, p={}, K={} the criterion VALUE is priced by the \
                     streaming ½log|B| implementation (direct_logdet_admitted = false). \
                     Returning it would desync value and gradient by ½·d/dρ log|I + B⁻¹ΔC|.",
                    self.n_obs(),
                    self.output_dim(),
                    self.k_atoms()
                )));
            }
        }

        if let Some(sparse_index) = rho.sparse_flat_index() {
            explicit[sparse_index] =
                crate::assignment::assignment_prior_log_strength_derivative_weighted(
                    &self.assignment,
                    rho,
                    self.row_loss_weights.as_deref(),
                )
                .map_err(OuterGradientError::internal)?;
            // ordered Beta--Bernoulli concentration controls only the Beta--Bernoulli prior. The
            // final reconstruction gate is `sigmoid(logit/tau)`, so the data
            // likelihood and its Gauss--Newton blocks have no direct alpha
            // derivative. Structurally fixed assignments have no sparse index
            // and skip this channel entirely.
            if !exact_a_logdet_route {
                let joint_trace = match logdet_derivative_bundle {
                    Some((probes, sinv)) => self
                        .assignment_log_strength_hessian_trace_from_probes(
                            rho,
                            evidence_cache,
                            probes,
                            sinv,
                            evidence_operator,
                        )
                        .map_err(OuterGradientError::internal)?,
                    None => self
                        .assignment_log_strength_hessian_trace(rho, cache, solver)
                        .map_err(OuterGradientError::internal)?,
                };
                logdet_trace[sparse_index] = joint_trace;
            }
        }

        // #1556: λ_smooth is per-atom, so the smoothness gradient block occupies
        // the K layout-derived smooth indices (one per atom). Each atom
        // `k` carries its own explicit penalty-energy derivative, log|H| trace,
        // and Occam-normalizer derivative.
        let k_smooth = rho.log_lambda_smooth.len();
        let lambda_smooth_vec = rho
            .lambda_smooth_vec()
            .map_err(OuterGradientError::internal)?;
        // Explicit `∂loss.smoothness/∂log λ_k = 0.5·λ_k·<B_k, S_k B_k>` (the
        // per-atom split). Its sum is the λ-scaled penalty energy; renormalize to
        // `loss.smoothness` so the total matches the criterion's reported energy
        // bit-for-bit (folding in any minibatch `penalty_scale` baked into it).
        let mut smooth_explicit = self
            .decoder_smoothness_value_per_atom(&lambda_smooth_vec)
            .map_err(OuterGradientError::internal)?;
        let smooth_explicit_sum: f64 = smooth_explicit.iter().sum();
        if smooth_explicit_sum.abs() > 0.0 {
            let renorm = loss.smoothness / smooth_explicit_sum;
            for v in smooth_explicit.iter_mut() {
                *v *= renorm;
            }
        }
        // #2080: the per-atom smoothness logdet derivative off the shared
        // low-rank derivative representation when the rational lane supplied it;
        // the dense `DeflatedArrowSolver` selected inverse otherwise.
        let smooth_logdet = if exact_a_logdet_route {
            None
        } else {
            Some(match logdet_derivative_bundle {
                Some((probes, sinv)) => self
                    .decoder_smoothness_effective_dof_per_atom_from_probes(
                        probes,
                        sinv,
                        &lambda_smooth_vec,
                    )
                    .map_err(|err| OuterGradientError::InternalInvariant {
                        reason: format!(
                            "analytic_outer_rho_gradient_components_with_bundle: smooth dof (matrix-free): {err}"
                        ),
                    })?,
                None => self
                    .decoder_smoothness_effective_dof_with_solver_per_atom(
                        cache,
                        solver,
                        &lambda_smooth_vec,
                    )
                    .map_err(|err| OuterGradientError::InternalInvariant {
                        reason: format!("analytic_outer_rho_gradient_components_with_bundle: {err}"),
                    })?,
            })
        };
        let smooth_occam = self
            .reml_occam_log_lambda_smooth_derivative(rho)
            .map_err(OuterGradientError::internal)?;
        for atom_idx in 0..k_smooth {
            let index = rho.smooth_flat_index(atom_idx);
            explicit[index] = smooth_explicit[atom_idx];
            if let Some(smooth_logdet) = smooth_logdet.as_ref() {
                logdet_trace[index] = 0.5 * smooth_logdet[atom_idx];
            }
            occam[index] = -smooth_occam[atom_idx];
        }

        let ard_explicit = self
            .ard_log_precision_explicit_derivatives(rho)
            .map_err(OuterGradientError::internal)?;
        // #2080: the per-(atom,axis) ARD log-precision Hessian derivative off the
        // SAME shared low-rank representation (the all-or-nothing cluster's
        // second channel) when present; the dense
        // deflated selected inverse otherwise. The from-probes channel HARD-REFUSES
        // any row carrying gauge/rotation deflation (the plain-S⁻¹ bundle cannot
        // reconstruct the Daleckii–Krein correction), routing that fit to the dense
        // channel rather than silently dropping the correction.
        let ard_logdet_traces = if exact_a_logdet_route {
            None
        } else {
            let joint = match logdet_derivative_bundle {
                Some((probes, sinv)) => self
                    .ard_log_precision_hessian_trace_from_probes(
                        rho,
                        evidence_cache,
                        probes,
                        sinv,
                        evidence_operator,
                    )
                    .map_err(|err| OuterGradientError::InternalInvariant {
                        reason: format!(
                            "analytic_outer_rho_gradient_components_with_bundle: ARD logdet trace \
                             (matrix-free): {err}"
                        ),
                    })?,
                None => self
                    .ard_log_precision_hessian_trace(rho, cache, solver, evidence_operator)
                    .map_err(|err| OuterGradientError::InternalInvariant {
                        reason: format!("analytic_outer_rho_gradient_components_with_bundle: {err}"),
                    })?,
            };
            Some(joint)
        };
        // #1026 shared-ARD: `ard_flat_index` maps `(k, axis)` onto the flat outer
        // coordinate for BOTH parameterizations. In `Shared` mode several atoms
        // alias one axis coordinate `1+K+axis`, and the outer derivative there is
        // `∂/∂log α_axis = Σ_{k owns axis} ∂/∂log α_{k,axis}` (chain rule through
        // the broadcast), so we ACCUMULATE. In `PerAtom` mode each `(k, axis)` has
        // a unique coordinate, so `+=` is identical to the historical `=`. Walking
        // a raw per-atom cursor in `Shared` mode would index past the flat length
        // `1+K+max_d` (OOB) and split one shared strength across phantom slots.
        for k in 0..rho.log_ard.len() {
            for axis in 0..rho.log_ard[k].len() {
                let idx = rho.ard_flat_index(k, axis);
                explicit[idx] += ard_explicit[k][axis];
                if let Some(joint) = ard_logdet_traces.as_ref() {
                    logdet_trace[idx] += joint[k][axis];
                }
            }
        }

        // The scalar criterion adds the realised-rank charge to `½ log|H|`.
        // Its direct rho differential belongs alongside the explicit
        // penalty channels and is present on every layout (dense or probes).
        explicit += &rank_charge.direct_rho;

        // #2080: the envelope Γ off the SAME shared low-rank logdet derivative
        // representation (the all-or-nothing cluster's third channel) when
        // present; the dense selected inverse otherwise. #2712: the border-only
        // bundle reconstructs the row block on the DEFLATED chart too — `A_i` is
        // the conditioned row Cholesky, so `A_i⁻¹ + G_i S⁻¹ G_iᵀ` is the deflated
        // `(H⁻¹)_tt` — and `logdet_theta_adjoint_from_probes` subtracts the same
        // Daleckii–Krein correction the dense route subtracts instead of routing
        // the fit away. Ordered Beta--Bernoulli uses its row-local PSD majorizer
        // and shared-mass derivative directly.
        // This completes the matrix-free selected-inverse cluster (smoothness EDF + ARD
        // Hessian trace + θ-adjoint); assignment log-strength traces remain
        // solver-bound
        // — the last gaps before the routing flip (see the docstring).
        let majorizer_gamma = if exact_a_logdet_route {
            None
        } else {
            let gamma = match logdet_derivative_bundle {
                Some((probes, sinv)) => self
                    .logdet_theta_adjoint_from_probes(
                        rho,
                        evidence_cache,
                        probes,
                        sinv,
                        evidence_operator,
                        Some(target),
                    )
                    .map_err(OuterGradientError::internal)?,
                None => self
                    .logdet_theta_adjoint(rho, cache, solver)
                    .map_err(OuterGradientError::internal)?,
            };
            Some(gamma)
        };
        // `½ Γ_joint·theta_hat + ∇R·theta_hat` is represented by one effective
        // logdet adjoint `Γ_eff = Γ_joint + 2∇R`, preserving the existing
        // `-½ <Γ_eff, A^-1 g_rho>` contraction convention below.
        let majorizer_gamma = majorizer_gamma.map(|mut gamma| {
            gamma.t.scaled_add(2.0, &rank_charge.theta.t);
            gamma.beta.scaled_add(2.0, &rank_charge.theta.beta);
            gamma
        });
        // #1418: the implicit-function correction is `−½·Γᵀ·θ̂_ρ` with
        // `θ̂_ρ = −A⁻¹ g_ρ` (the code contracts `−½·⟨Γ, A⁻¹ g_ρ⟩` with rhs `= +∂g/∂ρ`, i.e. `+½·Γᵀθ̂_ρ` of the response — the sign lives in the −0.5 factor), where `A = ∇²_θθ L` is the EXACT stationarity
        // Jacobian of the inner fit — data residual curvature, exact softmax
        // entropy Hessian, exact ordered Beta--Bernoulli marginal curvature, and
        // exact periodic ARD curvature. The matrix the `solver`
        // factors is `B` (Gauss-Newton data curvature, the softmax Gershgorin
        // majorizer, the ordered Beta--Bernoulli row-local PSD majorizer, and
        // `max(V'',0)` ARD curvature): the `½log|B|` Laplace term is consistent
        // with `Γ = ½tr(B⁻¹ ∂B/∂θ)`, but the implicit step is governed by `A`.
        // `solve_exact_stationarity` applies the spectral pseudoinverse of
        // `A = B + ΔC`, where
        // `ΔC = apply_exact_hessian_minus_b`, so the correction is no longer
        // biased by `(B⁻¹ − A⁻¹)` and does not assume `A` is SPD.
        //
        // A numerical stopping tolerance does not change the mathematical
        // objective.  At the exact inner optimum the envelope theorem cancels
        // the penalized-loss response, but the Laplace term still contributes
        // `-1/2 Gamma' theta_hat_rho`.  Dropping this term differentiates a
        // fictitious criterion in which the fitted state is held fixed.  The
        // exact stationarity solve above supplies the required implicit response.
        // #2231 — the trailing `L−1` flat coordinates are the crosscoder block
        // relevances `log λ_ℓ` (`SaeManifoldRho::to_flat` appends them last).
        // Their inner-gradient dependence enters through the λ-scaled target, so
        // their RHS is `−½·Jᵀ_M Z̃^{(ℓ)}` (`crosscoder_block_ift_rhs`), NOT the
        // penalty/prior channels `outer_rho_gradient_ift_rhs` owns. The adjoint
        // contraction below then completes the block gradient with the same
        // `−½·Γᵀθ̂_ρ` channel every other coordinate carries; the explicit data
        // + Jacobian parts stay with the eval lane's `block_log_lambda_gradient`.
        // #2080(A): collapse the per-coordinate IFT solves into ONE adjoint solve.
        // The implicit correction is `−½·⟨Γ, A⁺ g_ρ_l⟩` for every outer coordinate
        // `l`. The exact θθ-Hessian `A = ∇²_θθ L` is symmetric and its near-null
        // deflation uses Euclidean spectral projections, so `A⁺` is
        // self-adjoint and `⟨Γ, A⁺ g_ρ_l⟩ = ⟨A⁺Γ, g_ρ_l⟩ = ⟨a, g_ρ_l⟩` with the
        // adjoint `a = A⁺Γ` solved ONCE. A near-null pencil direction contributes
        // `g_i r_i / μ_i` only when BOTH Γ and `g_ρ_l` excite it, in which case the
        // forward (per-coordinate) and this adjoint solve deflate it identically —
        // so the collapse is EXACT, not an approximation, while dropping the outer
        // IFT cost from `O(P_ρ)` solves to one. `solve_exact_stationarity_is_self_adjoint_2080`
        // pins the self-adjointness this identity rests on.
        // The single adjoint solve `a = A⁺Γ` — the only solver-bound step. At
        // #2330 Phase-2: on the dense direct-logdet route (no probe bundle, no
        // matrix-free system) the ranked value is ½log|A|, so the logdet channels
        // must be A-based; `exact_a_logdet_route` suppressed the B-majorizer
        // producers above precisely so this is the only assembly that ran.
        // Explicit / occam / rank-charge-direct channels are majorizer-
        // independent and were produced on both routes.
        //
        // BOTH ROUTES DIFFERENTIATE ½log|A| (#2515). #2509 Phase-2b (`5563a2a18`)
        // moved every production VALUE route onto the exact observed information;
        // #2515 then moved the bundle route's DERIVATIVE there too, by giving it a
        // `BundleEvidenceGeometry` that names the operator and carries `A`'s own
        // factor cache. `exact_a_logdet_route` still selects which ASSEMBLY runs —
        // the dense priced pseudo-inverse below, or the from-probes channels above
        // — but no longer which operator is priced. #2333 (routing this θ-adjoint
        // through the Trace row-jet seam on `A`'s selected inverse) is a
        // representation change downstream of that, not a missing operator.
        //
        // Exactly one arm produces Γ, so the two assemblies cannot both be paid
        // for on one gradient.
        let (gamma, dense_stationarity_adjoint) = match majorizer_gamma {
            Some(gamma) => (gamma, None),
            None => {
                let DenseExactALogdetChannels {
                    logdet_trace: exact_logdet_trace,
                    theta_adjoint: exact_gamma,
                    stationarity_adjoint,
                } = self
                    .dense_exact_a_logdet_channels(target, rho, loss, cache)
                    .map_err(OuterGradientError::internal)?;
                logdet_trace = exact_logdet_trace;
                (exact_gamma, Some(stationarity_adjoint))
            }
        };

        // At massive K (`matrix_free_system = Some`) the materialized operator is
        // unavailable, so the adjoint uses the certified Ritz pseudoinverse
        // route. Every dense arm is owned by the rank-revealing spectral
        // pseudoinverse. On the exact-A logdet arm it reuses the eigensystem that
        // produced Γ; on a B-majorizer arm it materializes that same physical A
        // once here. Both representations use the same spectral null policy.
        let adjoint = match (matrix_free_system, dense_stationarity_adjoint) {
            (Some(system), None) => {
                self.solve_exact_stationarity_matrix_free(rho, target, cache, system, &gamma)
            }
            (None, Some(adjoint)) => Ok(adjoint),
            (None, None) => self.solve_exact_stationarity(rho, target, cache, &gamma),
            (Some(_), Some(_)) => Err(
                "analytic_outer_rho_gradient_components_with_bundle: dense exact-A adjoint was assembled \
                 for a matrix-free operator route"
                    .to_string(),
            ),
        }
        .map_err(|err| {
            OuterGradientError::classify_arrow_solver_error(
                &err,
                OuterGradientError::NonIdentifiable {
                    reason: err.clone(),
                },
            )
        })?;
        let block_tail_start = n_params - rho.log_lambda_block.len();
        for coord in 0..n_params {
            let rhs = if coord >= block_tail_start && !rho.log_lambda_block.is_empty() {
                let &(p_x, ref block_dims) =
                    self.crosscoder_pricing_spans.as_ref().ok_or_else(|| {
                        OuterGradientError::internal(
                            "analytic_outer_rho_gradient_components_with_bundle: rho carries block \
                             coordinates but no crosscoder pricing spans are installed"
                                .to_string(),
                        )
                    })?;
                let block = coord - block_tail_start;
                let start = p_x + block_dims[..block].iter().sum::<usize>();
                self.crosscoder_block_ift_rhs(cache, target, start..start + block_dims[block])
                    .map_err(OuterGradientError::internal)?
            } else {
                self.outer_rho_gradient_ift_rhs(rho, coord, cache)
                    .map_err(OuterGradientError::internal)?
            };
            let mut dot = 0.0_f64;
            for idx in 0..adjoint.t.len() {
                dot += adjoint.t[idx] * rhs.t[idx];
            }
            for idx in 0..adjoint.beta.len() {
                dot += adjoint.beta[idx] * rhs.beta[idx];
            }
            third_order_correction[coord] = -0.5 * dot;
        }

        Ok(SaeOuterRhoGradientComponents {
            explicit,
            logdet_trace,
            occam,
            third_order_correction,
        })
    }

    /// Classification of the exact observed information `A = B + ΔC` for the
    /// value path (#2330 / #2336 / #2673).
    ///
    /// A converged inner mode is a genuine exact-Laplace maximum iff every
    /// direction of `A` is `≥ −floor` in its own band, and that band is
    /// [`ExactHessianSpectralBlock::rank_floor`] — ONE predicate, in the ONE
    /// metric the gradient path classifies the same directions by. #2330 ACCEPTS
    /// above `−floor`; #2336's clamp attribution TRIGGERS below it; both read
    /// that one function, so they cannot disagree in the band.
    ///
    /// **The absolute floor this used to be is gone (#2673).** It was
    /// `1e-9 · max(λ_max(A), 1)`, a single number applied to every direction,
    /// while the gradient path used `|μ| = |vᵀAv/vᵀBv| < √ε` — and both ran on
    /// the same `A` inside one evaluation, with WHICH one the value path used
    /// decided by `direct_logdet_admitted`, i.e. by ambient free memory. Written
    /// as thresholds on the same `|λ|`, one was constant across directions and
    /// the other varied by the spread of the `B`-Rayleigh quotient (24.13x
    /// measured on the #2515 state, ratio straddling 1, so neither was even the
    /// conservative one), `λ/λ_max(A)` was not a curvature (it moves under a
    /// reparametrization `θ → Lθ` that leaves `μ` fixed, and `θ = (t, β)` mixes
    /// chart coordinates with data-scaled border coefficients), and `1e-9` was
    /// tuned where `√ε` is derived. See
    /// [`sae_exact_a_identifiability_floor`] for the argument and
    /// `tests::the_two_floors_are_incommensurable_thresholds_on_one_operator_2673`
    /// for the measurement.
    ///
    /// #2674 — the band is the ONLY null predicate on the exact-A route.
    /// `exact_hessian_spectral_block` used to delete the analytic chart-gauge
    /// orbit structurally before the spectrum was ever classified, which made two
    /// null predicates own one eigenvalue array. The measurement that settled
    /// which to keep: on the #2336/#2330 stall the declared orbit carried
    /// 86.6%–93.2% of the KKT gradient and per-direction slopes of the PENALIZED
    /// objective at 8x, 10x and 210x the convergence tolerance, so it was not a
    /// null of this operator at all — the priors are not invariant along a
    /// symmetry of the reconstruction.
    ///
    /// Exact-A evidence value, with one coherent basin matrix on each resolved
    /// negative spectral subspace. The same owner supplies the differential
    /// consumed by the outer gradient.
    pub(crate) fn exact_observed_information_log_dets_with_saddle_directions(
        &self,
        rho: &SaeManifoldRho,
        target: ArrayView2<'_, f64>,
        cache: &ArrowFactorCache,
        saddle_directions: &mut Vec<(Array1<f64>, f64)>,
    ) -> Result<f64, SaeCriterionError> {
        let total_t = cache.delta_t_len();
        // #2828 — the border half of `E = B − A`, read off the same border probes
        // that build `A` (#2731).
        let (a, e_beta) =
            self.materialize_exact_hessian_dense_with_gap_border(rho, target, cache)?;
        let e_diag = self.materialize_ard_concave_clamp_diagonal(rho, cache)?;
        let joint = Self::exact_hessian_spectral_block(
            a,
            &e_diag,
            e_beta.as_ref(),
            total_t,
            ArrowMetric::Joint(cache),
        )?;
        // #2080/#2267 — a refused basin pushes its refused directions here: unit
        // vectors in the joint `(t, β)` cache layout, each with its basin curvature,
        // most negative first. The evidence root descends them before it concludes
        // the state has no Laplace normaliser, off this same eigensystem, so a
        // refusal pays one dense materialization and eigendecomposition, not two.
        let joint_pricing = Self::classify_exact_hessian_basin(
            &joint,
            &e_diag,
            e_beta.as_ref(),
            total_t,
            |v| ArrowMetric::Joint(cache).quadratic_form(v),
            "joint",
            Some(saddle_directions),
        )?;
        // #2267 — the priced rank stratum, once per evaluation. Trace job 578389 read
        // sae_manifold_euclidean_k2_terminates price its incumbent at −3.591e3 while trial
        // points 0.125, 0.0625 and 0.03125 away priced +5.468e2, +5.536e2 and +5.570e2,
        // and the inner objective moved continuously (−45.10, −38.46, −35.14 → −31.81).
        // A jump that size over a continuous state is a change in which directions
        // ½log|A| prices: an in-band direction adds nothing, a retained one adds ½·ln λ.
        // `substituted` counts the in-band directions the bare floor would have kept,
        // the ones only the evidence factor's substituted stiffness puts in the band.
        let mut retained = 0usize;
        let mut in_band = 0usize;
        let mut substituted = 0usize;
        let mut min_retained_over_floor = f64::INFINITY;
        let mut max_band_over_floor = 0.0_f64;
        for index in 0..joint.eigenvalues.len() {
            let magnitude = joint.eigenvalues[index].abs();
            let floor = joint.rank_floor(index);
            if magnitude <= floor {
                in_band += 1;
                max_band_over_floor = max_band_over_floor.max(magnitude / floor);
                let bare = sae_exact_a_direction_floor(
                    joint.eigenvalues.len(),
                    joint.spectral_norm,
                    joint.metric_scale[index],
                );
                if magnitude > bare {
                    substituted += 1;
                }
            } else if joint.eigenvalues[index] > 0.0 {
                retained += 1;
                min_retained_over_floor = min_retained_over_floor.min(magnitude / floor);
            }
        }
        log::info!(
            "[SAE-EXACT-DENSE] priced: dim={} retained={retained} in_band={in_band} \
             substituted={substituted} negative={} ½log|A|={:.6e} \
             min retained |λ|/floor={:.3e} max in-band |λ|/floor={:.3e}",
            joint.eigenvalues.len(),
            joint_pricing.negative.len(),
            0.5 * joint_pricing.log_det,
            min_retained_over_floor,
            max_band_over_floor,
        );
        Ok(joint_pricing.log_det)
    }

    /// Build a cluster-stable eigensystem and the shared absolute null floor for
    /// one already-materialized exact-Hessian block.
    ///
    /// #2674 — this used to take an analytic chart-gauge basis and diagonalize
    /// `Zᵀ A Z` on its orthogonal complement, deleting the declared orbit
    /// STRUCTURALLY so a spectral floor never had to rediscover it. That made
    /// two null predicates own one eigenvalue array, and the declaration-based
    /// one is the one that is wrong here: the chart orbit is a symmetry of the
    /// RECONSTRUCTION (measured reconstruction-invariant to `rel_ls ~1e-16`),
    /// not of the PENALIZED objective this operator is the Hessian of — the ARD
    /// and smoothing priors are not invariant along it. At the #2336/#2330 stall
    /// the two declared directions carried 86.6%–93.2% of the KKT gradient's
    /// norm and per-direction slopes of 8x, 10x and 210x the convergence
    /// tolerance, so quotienting them out handed the Newton step a right-hand
    /// side it was structurally unable to reduce and the inner solve stalled.
    ///
    /// One predicate owns the classification: [`Self::rank_floor`], which is
    /// also the predicate the matrix-free gradient path applies to its own
    /// solution direction (#2673). Where the orbit really is flat for the
    /// penalized operator its eigenvalue lands in `[−floor(i), floor(i)]` and the
    /// pseudoinverse discards it, which is what the independent oracle in
    /// `exact_observed_information_log_det_matches_eigendecomposition_2330`
    /// (full-spectrum `eigh` of the dense `A`, keeping `λ > floor(i)`) reads.
    fn exact_hessian_spectral_block(
        operator: Array2<f64>,
        e_diag: &Array1<f64>,
        e_beta: Option<&Array2<f64>>,
        total_t: usize,
        metric: ArrowMetric<'_>,
    ) -> Result<ExactHessianSpectralBlock, String> {
        let dimension = operator.nrows();
        if operator.ncols() != dimension || total_t > dimension || e_diag.len() < total_t {
            return Err(format!(
                "exact_hessian_spectral_block: operator {:?}, t dimension {total_t}, E diagonal length {}",
                operator.dim(),
                e_diag.len()
            ));
        }
        // #2267 — the other half of the split; see `materialize_exact_hessian_dense`.
        let eigh_started = std::time::Instant::now();
        let (mut eigenvalues, mut eigenvectors) =
            Self::cluster_stable_eigh(&operator, e_diag, e_beta, total_t)?;
        let eigh_elapsed = eigh_started.elapsed();
        log::info!(
            "[SAE-EXACT-DENSE] eigendecomposition DONE: dim={dimension}, {:.3} s",
            eigh_elapsed.as_secs_f64(),
        );
        // #2673 — the scale every direction is classified relative to. `dim`
        // applies of the arrow factorization's own operator, which is `O(dim²)`
        // against the `O(dim³)` decomposition above it and needs no second
        // dense block.
        let spectral_norm = eigenvalues
            .iter()
            .map(|value| value.abs())
            .fold(0.0_f64, f64::max);
        // Rank depends on v'Bv, so a repeated A eigenspace must resolve its
        // B directions too. The matrix-free route uses this same convention.
        canonicalize_exact_a_rank_clusters(
            &mut eigenvalues,
            &mut eigenvectors,
            spectral_norm,
            &|v| metric.apply(v.view()),
        )?;
        let mut metric_scale = Array1::<f64>::zeros(eigenvalues.len());
        let mut substituted_stiffness = Array1::<f64>::zeros(eigenvalues.len());
        for index in 0..eigenvalues.len() {
            let value = metric.quadratic_form(eigenvectors.column(index))?;
            if !(value.is_finite() && value > 0.0) {
                return Err(format!(
                    "exact_hessian_spectral_block: the arrow factorization must be positive \
                     definite along every direction it classifies, but direction {index} of the \
                     {dimension}-dimensional block has vᵀBv={value:.6e} (#2673)"
                ));
            }
            metric_scale[index] = value;
            substituted_stiffness[index] =
                metric.substituted_stiffness(eigenvectors.column(index))?;
        }
        let block = ExactHessianSpectralBlock {
            operator,
            eigenvalues,
            eigenvectors,
            metric_scale,
            substituted_stiffness,
            spectral_norm,
        };
        let crossings = block.arithmetic_band_crossings();
        if crossings > 0 {
            // Report when arithmetic, rather than the B-relative scale,
            // determines the shared spectral rank.
            log::warn!(
                "[SAE-EXACT-DENSE] arithmetic rank limit: {crossings} of {dimension} \
                 directions have |λ| inside the eigendecomposition's backward error \
                 ({:.6e}) while their B-relative curvature is identifiable; \
                 value and adjoint discard these directions",
                (dimension as f64) * f64::EPSILON * block.spectral_norm,
            );
        }
        Ok(block)
    }

    /// Price the full basin on A's resolved negative spectral subspace.
    /// If V spans that subspace, C = V' A V + V' E V. Its determinant and
    /// definiteness are invariant under V -> V R, including repeated negative
    /// eigenvalues. Independent diagonal prices discard E's coupling and do
    /// not even define a continuous value at a repeated eigenvalue.
    ///
    /// On a fixed rank stratum, Q = V C^+ V' is the E derivative. The A
    /// derivative adds the positive inverse and the response of V. Only
    /// negative/complement spectral gaps enter that response; rotations inside
    /// the negative subspace cancel in the determinant.
    ///
    /// A basin direction with `μ < −floor` refuses the block. With
    /// `refused_directions` present, every such direction is pushed with its
    /// `μ` (most negative first) before the refusal is returned, so a caller can
    /// descend the saddle rather than only learn that one exists (#2080).
    fn classify_exact_hessian_basin(
        block: &ExactHessianSpectralBlock,
        e_diag: &Array1<f64>,
        e_beta: Option<&Array2<f64>>,
        total_t: usize,
        metric: impl Fn(ArrayView1<'_, f64>) -> Result<f64, String>,
        label: &'static str,
        mut refused_directions: Option<&mut Vec<(Array1<f64>, f64)>>,
    ) -> Result<ExactHessianBasin, SaeCriterionError> {
        let dim = block.eigenvalues.len();
        if block.eigenvectors.dim() != (dim, dim) || total_t > dim || e_diag.len() < total_t {
            return Err(SaeCriterionError::Numerical(
                "exact-A pricing dimensions disagree".to_string(),
            ));
        }
        let negative: Vec<usize> = (0..dim)
            .filter(|&i| block.eigenvalues[i] < -block.rank_floor(i))
            .collect();
        let complement: Vec<usize> = (0..dim)
            .filter(|&i| block.eigenvalues[i] >= -block.rank_floor(i))
            .collect();
        let mut log_det = 0.0;
        for &i in &complement {
            let lambda = block.eigenvalues[i];
            if lambda <= block.rank_floor(i) {
                continue;
            }
            log_det += lambda.ln();
        }
        if negative.is_empty() {
            return Ok(ExactHessianBasin {
                log_det,
                negative,
                complement,
                basis: Array2::zeros((dim, 0)),
                rotation: Array2::zeros((0, 0)),
                vectors: Array2::zeros((dim, 0)),
                inverse_values: Array1::zeros(0),
            });
        }
        let q = negative.len();
        let basis = Array2::from_shape_fn((dim, q), |(row, col)| {
            block.eigenvectors[[row, negative[col]]]
        });
        let mut basin = Array2::<f64>::zeros((q, q));
        for row in 0..total_t {
            for i in 0..q {
                let left = e_diag[row] * basis[[row, i]];
                for j in 0..q {
                    basin[[i, j]] += left * basis[[row, j]];
                }
            }
        }
        if e_beta.is_some() {
            // #2828 — the β-tier decoder priors' majorization gap. Dense on the
            // border block, so unlike the coordinate clamps it cannot be folded
            // in as a diagonal weight.
            let border = Self::dropped_curvature_border_forms(
                e_beta,
                total_t,
                basis.view(),
                basis.view(),
            );
            for i in 0..q {
                for j in 0..q {
                    basin[[i, j]] += border[[i, j]];
                }
            }
        }
        for (i, &index) in negative.iter().enumerate() {
            basin[[i, i]] += block.eigenvalues[index];
        }
        let (basin_values, basin_rotation) = basin
            .eigh(Side::Lower)
            .map_err(|error| format!("exact-A basin eigendecomposition: {error:?}"))?;
        let basin_vectors = basis.dot(&basin_rotation);
        // Assembly of C can cancel A against E. Its arithmetic scale must
        // include both operands, while the statistical band uses the actual
        // B quadratic form of each rotated basin direction.
        let assembly_scale = block.spectral_norm
            + e_diag
                .iter()
                .take(total_t)
                .map(|x| x.abs())
                .fold(0.0_f64, f64::max)
            + e_beta.map_or(0.0, |gap| {
                // The border block is dense, so its arithmetic scale is a row sum
                // (the induced ∞-norm), not a single entry: a cancellation in `C`
                // can be as large as the whole row.
                (0..gap.nrows())
                    .map(|row| {
                        (0..gap.ncols())
                            .map(|col| gap[[row, col]].abs())
                            .sum::<f64>()
                    })
                    .fold(0.0_f64, f64::max)
            });
        let mut inverse_values = Array1::<f64>::zeros(q);
        let mut refused = false;
        for (i, &mu) in basin_values.iter().enumerate() {
            let vector = basin_vectors.column(i);
            let metric_scale = metric(vector)?;
            if !(mu.is_finite() && metric_scale.is_finite() && metric_scale > 0.0) {
                return Err(SaeCriterionError::Numerical(
                    "exact-A basin needs finite curvature and a positive metric".to_string(),
                ));
            }
            let floor = sae_exact_a_direction_floor(dim, assembly_scale, metric_scale);
            if mu < -floor {
                log::warn!(
                    "SAE exact-A basin refusal: block={label}, mode={i}, curvature={mu:e}, floor={floor:e}"
                );
                let Some(directions) = refused_directions.as_mut() else {
                    return Err(SaeCriterionError::IndefiniteObservedInformation { block: label });
                };
                directions.push((vector.to_owned(), mu));
                refused = true;
                continue;
            }
            if mu <= floor {
                continue;
            }
            log_det += mu.ln();
            inverse_values[i] = 1.0 / mu;
        }
        if refused {
            return Err(SaeCriterionError::IndefiniteObservedInformation { block: label });
        }
        Ok(ExactHessianBasin {
            log_det,
            negative,
            complement,
            basis,
            rotation: basin_rotation,
            vectors: basin_vectors,
            inverse_values,
        })
    }

    /// Realize the differential only for callers that consume it. Scalar
    /// evidence evaluation stops at the classified basin spectrum above.
    fn price_exact_hessian_block(
        block: &ExactHessianSpectralBlock,
        e_diag: &Array1<f64>,
        e_beta: Option<&Array2<f64>>,
        total_t: usize,
        metric: impl Fn(ArrayView1<'_, f64>) -> Result<f64, String>,
        label: &'static str,
    ) -> Result<ExactHessianPricing, SaeCriterionError> {
        let basin = Self::classify_exact_hessian_basin(
            block, e_diag, e_beta, total_t, metric, label, None,
        )?;
        Self::exact_hessian_basin_differential(block, e_diag, e_beta, total_t, &basin)
    }

    fn exact_hessian_basin_differential(
        block: &ExactHessianSpectralBlock,
        e_diag: &Array1<f64>,
        e_beta: Option<&Array2<f64>>,
        total_t: usize,
        basin: &ExactHessianBasin,
    ) -> Result<ExactHessianPricing, SaeCriterionError> {
        let dim = block.eigenvalues.len();
        let negative = &basin.negative;
        let complement = &basin.complement;
        let basis = &basin.basis;
        let q = negative.len();
        // The positive-inverse part `Σ vᵢvᵢᵀ/λᵢ` over the retained complement, as the
        // Gram `S·Sᵀ` of the columns `vᵢ/√λᵢ` (#2267). The rank-1 loop this replaces
        // paid `retained·dim²` scalar updates on every gradient evaluation.
        let retained: Vec<usize> = complement
            .iter()
            .copied()
            .filter(|&i| block.eigenvalues[i] > block.rank_floor(i))
            .collect();
        let scaled = Array2::from_shape_fn((dim, retained.len()), |(row, col)| {
            let i = retained[col];
            block.eigenvectors[[row, i]] / block.eigenvalues[i].sqrt()
        });
        let mut a_derivative = scaled.dot(&scaled.t());
        let mut clamp_diagonal_derivative = Array1::<f64>::zeros(total_t);
        let mut clamp_border_derivative = Array2::<f64>::zeros((dim - total_t, dim - total_t));
        let mut basin_inverse = Array2::<f64>::zeros((q, q));
        for (i, &inverse) in basin.inverse_values.iter().enumerate() {
            if inverse == 0.0 {
                continue;
            }
            let vector = basin.vectors.column(i);
            for row in total_t..dim {
                for col in total_t..dim {
                    clamp_border_derivative[[row - total_t, col - total_t]] += inverse * vector[row] * vector[col];
                }
            }
            for row in 0..dim {
                for col in 0..dim {
                    a_derivative[[row, col]] += inverse * vector[row] * vector[col];
                }
                if row < total_t {
                    clamp_diagonal_derivative[row] += inverse * vector[row] * vector[row];
                }
            }
            for row in 0..q {
                for col in 0..q {
                    basin_inverse[[row, col]] +=
                        inverse * basin.rotation[[row, i]] * basin.rotation[[col, i]];
                }
            }
        }
        if !negative.is_empty() && !complement.is_empty() {
            let other = Array2::from_shape_fn((dim, complement.len()), |(row, col)| {
                block.eigenvectors[[row, complement[col]]]
            });
            let mut e_cross = Array2::<f64>::zeros((q, complement.len()));
            for row in 0..total_t {
                for i in 0..q {
                    let left = e_diag[row] * basis[[row, i]];
                    for j in 0..complement.len() {
                        e_cross[[i, j]] += left * other[[row, j]];
                    }
                }
            }
            if e_beta.is_some() {
                // #2828 — `E` now has a border block, so the negative
                // projector's first-order response to it is part of `∂value/∂A`
                // too. Omitting it here would leave the value consistent and the
                // gradient not.
                let border = Self::dropped_curvature_border_forms(
                    e_beta,
                    total_t,
                    basis.view(),
                    other.view(),
                );
                for i in 0..q {
                    for j in 0..complement.len() {
                        e_cross[[i, j]] += border[[i, j]];
                    }
                }
            }
            let mut response = basin_inverse.dot(&e_cross);
            for (i, &negative_index) in negative.iter().enumerate() {
                for (j, &other_index) in complement.iter().enumerate() {
                    let gap = block.eigenvalues[negative_index] - block.eigenvalues[other_index];
                    if gap == 0.0 {
                        return Err(SaeCriterionError::Numerical(
                            "exact-A rank classification splits a repeated eigenspace; its negative projector is not differentiable"
                                .to_string(),
                        ));
                    }
                    response[[i, j]] /= gap;
                }
            }
            let cross = basis.dot(&response).dot(&other.t());
            for row in 0..dim {
                for col in 0..dim {
                    a_derivative[[row, col]] += cross[[row, col]] + cross[[col, row]];
                }
            }
        }
        Ok(ExactHessianPricing {
            a_derivative,
            clamp_diagonal_derivative,
            clamp_border_derivative,
        })
    }

    /// Materialize only the joint spectral geometry required by a dense
    /// exact-stationarity solve.  No log-determinant pricing or coordinate-block
    /// decomposition is paid on routes that rank the B majorizer.
    fn materialize_exact_stationarity_geometry(
        &self,
        rho: &SaeManifoldRho,
        target: ArrayView2<'_, f64>,
        cache: &ArrowFactorCache,
    ) -> Result<ExactHessianSpectralBlock, String> {
        let total_t = cache.delta_t_len();
        let (a, e_beta) =
            self.materialize_exact_hessian_dense_with_gap_border(rho, target, cache)?;
        let e_diag = self.materialize_ard_concave_clamp_diagonal(rho, cache)?;
        Self::exact_hessian_spectral_block(
            a,
            &e_diag,
            e_beta.as_ref(),
            total_t,
            ArrowMetric::Joint(cache),
        )
    }

    /// #2330 Phase-2/#2653 — one coherent quotient geometry for the exact-A
    /// outer-ρ derivative.  The priced pseudo-inverse `A⁺` and the raw signed
    /// stationarity pseudoinverse are derived from the SAME joint eigensystem
    /// and floor.  A genuine saddle still refuses
    /// the log-determinant derivative; resolved negative directions remain live
    /// in the stationarity solve when the value's clamp pricing admits them.
    fn materialize_exact_hessian_quotient_geometry(
        &self,
        rho: &SaeManifoldRho,
        target: ArrayView2<'_, f64>,
        cache: &ArrowFactorCache,
    ) -> Result<ExactHessianQuotientGeometry, String> {
        let total_t = cache.delta_t_len();
        let (a, e_beta) =
            self.materialize_exact_hessian_dense_with_gap_border(rho, target, cache)?;
        let e_diag = self.materialize_ard_concave_clamp_diagonal(rho, cache)?;
        let joint = Self::exact_hessian_spectral_block(
            a,
            &e_diag,
            e_beta.as_ref(),
            total_t,
            ArrowMetric::Joint(cache),
        )?;
        let joint_pricing = Self::price_exact_hessian_block(
            &joint,
            &e_diag,
            e_beta.as_ref(),
            total_t,
            |v| ArrowMetric::Joint(cache).quadratic_form(v),
            "joint",
        )
        .map_err(|error| error.to_string())?;
        Ok(ExactHessianQuotientGeometry {
            joint,
            joint_pricing,
        })
    }

    /// #2330 — dense symmetric materialization of the EXACT stationarity
    /// Hessian `A = ∇²_θθ L = B + ΔC` (`dim×dim`, `dim = total_t + k`), built
    /// column by column via [`Self::apply_exact_hessian`] and symmetrized, at the
    /// small-dense (circle-mint) scale; shared by the observed-information
    /// log-determinant (VALUE) and its `A⁻¹` selected inverse (GRADIENT) so both
    /// factor one identical operator
    /// ([`Self::exact_observed_information_log_dets_with_saddle_directions`]).
    /// The exact stationarity Hessian as a dense `dim × dim` matrix, assembled
    /// from `slots + k` Hessian-vector applies instead of `dim` (gam#2267).
    ///
    /// The operator is an arrow plus the ordered-Beta--Bernoulli prior's
    /// cross-row mass rank-one blocks. Subtract those known low-rank actions
    /// from each batched probe and add their complete matrix once afterward.
    /// In the remaining arrow, each row couples only to itself and the border.
    /// Probing coordinate slot `c` of EVERY row at once
    /// therefore returns column `c` of every row's diagonal block in one apply
    /// (the cross-row `t–t` entries that sum would otherwise mix are zero), and
    /// probing border column `j` returns `A[·, β_j]` whole, its `t` entries
    /// included, so the `t–β` block comes from the `k` border probes by
    /// symmetry. On the #2267 K=8 arm (508 rows × 2 coordinates + 72) that is
    /// 74 applies where the column loop took 1088, 3.8 ms each, 129 times in
    /// the run's first 14 minutes.
    ///
    /// [`Self::materialize_exact_hessian_dense_by_columns`] is that column
    /// loop, kept as the oracle the equality pin measures this against.
    ///
    /// The border probes also yield `E_ββ`, the β-tier decoder-prior majorizer gap
    /// on the border (#2828): it is the negation of the leg-(5) columns those probes
    /// compute (#2731), symmetrized, and `None` when no β-tier prior is live and
    /// the block is identically zero.
    pub(crate) fn materialize_exact_hessian_dense_with_gap_border(
        &self,
        rho: &SaeManifoldRho,
        target: ArrayView2<'_, f64>,
        cache: &ArrowFactorCache,
    ) -> Result<(Array2<f64>, Option<Array2<f64>>), String> {
        let total_t = cache.delta_t_len();
        let k = cache.k;
        let dim = sae_exact_stationarity_dim(total_t, k);
        let n_rows = cache.n_rows();
        let offsets = &cache.row_offsets;
        let mass_channels = ordered_beta_bernoulli_psd_majorizer_third_channels_weighted(
            &self.assignment,
            rho,
            self.row_loss_weights.as_deref(),
        )?;
        let mut mass_carriers: Vec<(f64, Vec<(usize, f64)>)> = Vec::new();
        if let Some(channels) = mass_channels.as_ref() {
            mass_carriers = channels
                .mass_hessian_coefficient
                .iter()
                .map(|&coefficient| (coefficient, Vec::new()))
                .collect();
            for row in 0..n_rows {
                for (local, variable) in self.row_vars_for_cache_row(row, cache)?.iter().enumerate()
                {
                    if let SaeLocalRowVar::Logit { atom } = *variable {
                        let value = channels.z_jac[row * channels.k_max + atom];
                        if value != 0.0 {
                            mass_carriers[atom].1.push((offsets[row] + local, value));
                        }
                    }
                }
            }
        }
        let slots = (0..n_rows)
            .map(|row| offsets[row + 1] - offsets[row])
            .max()
            .unwrap_or(0);
        log::info!(
            "[SAE-EXACT-DENSE] materializing the exact stationarity Hessian: dim={dim} \
             (coords={total_t} + border={k}) from {} arrow probes ({slots} coordinate \
             slots + {k} border columns), {:.1} MiB per dim x dim f64 block",
            slots + k,
            sae_exact_stationarity_block_bytes(dim) as f64 / (1024.0 * 1024.0),
        );
        let build_started = std::time::Instant::now();
        // #2828 — one β-tier decoder-prior plan for all `slots + k` probes.
        let prepared = self.prepare_decoder_prior_beta_curvature(1.0);
        // #2731 — and one residual-curvature plan: the row jets and residual are
        // contracted here once, where every probe used to rebuild them.
        let residual = self.prepare_residual_curvature_rows(target, cache)?;
        let mut a = Array2::<f64>::zeros((dim, dim));
        // #2731 — every probe is an independent apply of one fixed operator against
        // plans prepared once for this state, so the probes run on the rayon pool.
        // Columns are written serially in probe order: `a` is bit-identical to the
        // one-probe-at-a-time loop, and the first failing probe's error is the one
        // returned. Probes run in batches of the pool width, so at most one batch of
        // `dim`-length columns is held beside `a`. Job 391502 read 290 probes of
        // 699–827 ms each per polish step at `p = 2048, charts = 32`, on one core.
        use rayon::prelude::*;
        let pool_threads = rayon::current_num_threads().max(1);
        for batch_start in (0..slots).step_by(pool_threads) {
            let batch_end = (batch_start + pool_threads).min(slots);
            let columns: Vec<Result<SaeArrowVector, String>> = (batch_start..batch_end)
                .into_par_iter()
                .map(|slot| -> Result<SaeArrowVector, String> {
                    let mut unit = SaeArrowVector {
                        t: Array1::<f64>::zeros(total_t),
                        beta: Array1::<f64>::zeros(k),
                    };
                    for row in 0..n_rows {
                        let (start, end) = (offsets[row], offsets[row + 1]);
                        if start + slot < end {
                            unit.t[start + slot] = 1.0;
                        }
                    }
                    let mut av = self.apply_exact_hessian_prepared(
                        rho, cache, &unit, &prepared, &residual,
                    )?;
                    for (coefficient, carrier) in &mass_carriers {
                        let projection = carrier
                            .iter()
                            .map(|&(index, value)| value * unit.t[index])
                            .sum::<f64>();
                        for &(index, value) in carrier {
                            av.t[index] -= coefficient * value * projection;
                        }
                    }
                    Ok(av)
                })
                .collect();
            for (offset, av) in columns.into_iter().enumerate() {
                let av = av?;
                let slot = batch_start + offset;
                for row in 0..n_rows {
                    let (start, end) = (offsets[row], offsets[row + 1]);
                    if start + slot < end {
                        let col = start + slot;
                        for i in start..end {
                            a[[i, col]] = av.t[i];
                        }
                    }
                }
            }
        }
        for (coefficient, carrier) in &mass_carriers {
            for &(row, left) in carrier {
                for &(col, right) in carrier {
                    a[[row, col]] += coefficient * left * right;
                }
            }
        }
        // #2731 — each border probe is `apply_exact_hessian_prepared` with leg (5)
        // of `ΔC` computed here and folded into `ΔC·e_j` exactly as
        // `apply_exact_hessian_minus_b_prepared` folds it, so the column is
        // bit-identical, and the leg column is kept: `E_ββ` is its negation, so the
        // gap border needs no second pass of `k` β-tier applies. Job 539190 read that
        // second pass at 11.109–13.432 s of each 32.00–40.34 s polish step
        // (`p = 2048, charts = 32, k = 288`).
        let projection = crate::frames::FrameProjection::new(self);
        let mut leg_columns = Array2::<f64>::zeros((k, k));
        for batch_start in (0..k).step_by(pool_threads) {
            let batch_end = (batch_start + pool_threads).min(k);
            let columns: Vec<Result<(SaeArrowVector, Array1<f64>), String>> =
                (batch_start..batch_end)
                    .into_par_iter()
                    .map(|j| -> Result<(SaeArrowVector, Array1<f64>), String> {
                        let mut unit = SaeArrowVector {
                            t: Array1::<f64>::zeros(total_t),
                            beta: Array1::<f64>::zeros(k),
                        };
                        unit.beta[j] = 1.0;
                        let b_v =
                            apply_raw_cached_arrow_hessian(cache, unit.t.view(), unit.beta.view())?;
                        let mut dc_v = self
                            .apply_exact_hessian_minus_b_prepared_before_beta_prior_leg(
                                rho, cache, &unit, &residual,
                            )?;
                        let leg = self.decoder_prior_gap_border_leg(
                            cache,
                            &prepared,
                            &projection,
                            unit.beta.view(),
                        )?;
                        for (index, &value) in leg.iter().enumerate() {
                            dc_v.beta[index] += value;
                        }
                        let av = SaeArrowVector {
                            t: &b_v.t + &dc_v.t,
                            beta: &b_v.beta + &dc_v.beta,
                        };
                        Ok((av, leg))
                    })
                    .collect();
            for (offset, column) in columns.into_iter().enumerate() {
                let (av, leg) = column?;
                let j = batch_start + offset;
                let col = total_t + j;
                for i in 0..total_t {
                    a[[i, col]] = av.t[i];
                    a[[col, i]] = av.t[i];
                }
                for i in 0..k {
                    a[[total_t + i, col]] = av.beta[i];
                    leg_columns[[i, j]] = leg[i];
                }
            }
        }
        for r in 0..dim {
            for c in (r + 1)..dim {
                let avg = 0.5 * (a[[r, c]] + a[[c, r]]);
                a[[r, c]] = avg;
                a[[c, r]] = avg;
            }
        }
        // `E = B − A` and leg (5) is the β-tier part of `A − B` on the border, so
        // `E_ββ` negates the kept columns. The remainder is a difference of two
        // symmetric operators; symmetrize the probe assembly so the basin quadratic
        // forms cannot pick up an asymmetric round-off residue.
        let gap_border = if k == 0 {
            None
        } else {
            let mut gap = leg_columns.mapv(|value| -value);
            if gap.iter().all(|&value| value == 0.0) {
                None
            } else {
                for row in 0..k {
                    for col in (row + 1)..k {
                        let average = 0.5 * (gap[[row, col]] + gap[[col, row]]);
                        gap[[row, col]] = average;
                        gap[[col, row]] = average;
                    }
                }
                Some(gap)
            }
        };
        let build_elapsed = build_started.elapsed();
        log::info!(
            "[SAE-EXACT-DENSE] operator BUILT: dim={dim}, {} arrow probes on {pool_threads} pool \
             threads + symmetrization in {:.3} s ({:.3} ms wall per probe), with the \
             decoder-prior majorizer gap border from the same {k} border probes; the \
             O(dim^3) symmetric eigendecomposition has NOT started yet",
            slots + k,
            build_elapsed.as_secs_f64(),
            build_elapsed.as_secs_f64() * 1.0e3 / ((slots + k).max(1) as f64),
        );
        Ok((a, gap_border))
    }

    /// #2330 Phase-2 — the A-based logdet gradient channels on the dense direct
    /// route: the direct trace vector `logdet_trace_i = ½tr(A⁺ ∂A/∂ρ_i)
    /// − ½tr(A_tt⁺ ∂A/∂ρ_i)` and the effective θ-adjoint
    /// `Γ_eff = tr(A⁺ ∂A/∂θ) − tr(A_tt⁺ ∂A_tt/∂θ) + 2∇R` (fed to the unchanged
    /// single-adjoint IFT collapse `a = A⁺Γ_eff`, `−½⟨a, g_ρ⟩`). `∂A/∂ρ_i =
    /// ∂B/∂ρ_i (raw_penalty_curvature_operators_by_flat) + ∂ΔC/∂ρ_i
    /// (exact_stationarity_penalty_derivative_delta_by_flat)`, already exact. The
    /// θ-adjoint rides `exact_a = true` (ARD clamp-free) with `skip_deflation_dk
    /// = true` (the exact A carries only the ρ-invariant gauge null, handled by
    /// the quotient pseudo-inverse — no B-style Daleckii–Krein correction).
    ///
    /// EXACT-MINUS-PATCH-D: the two `logdet_theta_adjoint_dense` calls emit
    /// `∂B/∂θ + ∂ΔC_ard/∂θ` but NOT the residual-curvature / softmax-entropy legs
    /// of `∂ΔC/∂θ` (Patch D). Until D lands, Γ_eff — hence the IFT correction — is
    /// missing that term and the conservation bisection stays red by exactly it.
    /// #2336 flag-1 — eigendecomposition with an `E`-canonical basis inside
    /// exactly repeated eigenspaces.
    ///
    /// A basis rotation preserves the eigenpair equation `A V = V diag(λ)` only
    /// when every rotated eigenvalue is identical. The previous implementation
    /// also rotated merely near-equal eigenvalues (gap at most `√ε·‖A‖₂`) while
    /// leaving their distinct eigenvalues attached to the rotated columns. That
    /// produced a matrix called a priced inverse which was not the inverse of the
    /// operator whose eigenvalues the value summed; #2515 measured direct trace
    /// derivatives of `5.340922` and `5.146446` for two scalar values whose
    /// derivatives were both `4.974886`.
    ///
    /// Repeated eigenvalues genuinely have an arbitrary eigenspace basis, so
    /// those and only those runs are resolved against the restriction of `E`.
    /// Distinct eigenvalues retain the actual `eigh` columns, however small their
    /// gap: numerical uncertainty cannot authorize returning false eigenpairs.
    /// `e_diag` is the t-block diagonal of `E` (zero on the β border), so the
    /// repeated-space restriction reads only the first `total_t` rows.
    pub(crate) fn cluster_stable_eigh(
        m: &Array2<f64>,
        e_diag: &Array1<f64>,
        e_beta: Option<&Array2<f64>>,
        total_t: usize,
    ) -> Result<(Array1<f64>, Array2<f64>), String> {
        if m.nrows() != m.ncols() || total_t > m.nrows() || e_diag.len() < total_t {
            return Err(format!(
                "cluster_stable_eigh: operator {:?}, t dimension {total_t}, E diagonal length {}",
                m.dim(),
                e_diag.len()
            ));
        }
        let (eigs, mut vecs) = m
            .eigh(Side::Lower)
            .map_err(|e| format!("cluster_stable_eigh: eigh failed: {e:?}"))?;
        let dim = eigs.len();
        let mut i = 0usize;
        while i < dim {
            let mut j = i + 1;
            while j < dim && eigs[j] == eigs[i] {
                j += 1;
            }
            let width = j - i;
            if width > 1 {
                // `E` is diagonal on the first `total_t` coordinate rows.  Its
                // restriction to this cluster is therefore
                //
                //     E_c[a,b] = sum_r e_diag[r] V[r,i+a] V[r,i+b]
                //
                // plus, since #2828, the border block's own dense contribution
                // (the β-tier decoder priors' majorization gap).  Accumulate one
                // weighted row outer product at a time.  Besides reading the
                // actual representation rather than manufacturing a dense matrix
                // of zeros, this changes the coordinate half of the work from
                // O(width^2 * dim^2) to O(width^2 * total_t).  Keeping `b` as the
                // inner loop walks both the cluster row and `ec` contiguously.
                let mut ec = Array2::<f64>::zeros((width, width));
                for row in 0..total_t {
                    let weight = e_diag[row];
                    if weight == 0.0 {
                        continue;
                    }
                    let cluster_row = vecs.slice(s![row, i..j]);
                    for a in 0..width {
                        let weighted_a = weight * cluster_row[a];
                        for b in a..width {
                            ec[[a, b]] += weighted_a * cluster_row[b];
                        }
                    }
                }
                // The upper triangle was accumulated once in the same order as
                // the former scalar quadratic form.  Copy it exactly rather than
                // averaging two independently-rounded contractions.
                for a in 0..width {
                    for b in (a + 1)..width {
                        ec[[b, a]] = ec[[a, b]];
                    }
                }
                if e_beta.is_some() {
                    let run = vecs.slice(s![.., i..j]);
                    let border = Self::dropped_curvature_border_forms(e_beta, total_t, run, run);
                    for a in 0..width {
                        for b in 0..width {
                            ec[[a, b]] += border[[a, b]];
                        }
                    }
                }
                let (_ec_eigs, rot) = ec
                    .eigh(Side::Lower)
                    .map_err(|e| format!("cluster_stable_eigh: cluster eigh failed: {e:?}"))?;
                drop(ec);
                let cluster = vecs.slice(s![.., i..j]).to_owned();
                let rotated = cluster.dot(&rot);
                vecs.slice_mut(s![.., i..j]).assign(&rotated);
            }
            i = j;
        }
        Ok((eigs, vecs))
    }

    /// #2336 — the coordinate-block (t-index → (atom, axis)) map for a cache, so
    /// the ARD-clamp E-attributability channels can attribute each priced
    /// direction's `e_v` mass back to the ρ_ard slot that scales it. `None` on
    /// logit / β rows (E is zero there).
    pub(crate) fn coord_axis_map_for_cache(
        &self,
        cache: &ArrowFactorCache,
    ) -> Result<Vec<Option<(usize, usize)>>, String> {
        let total_t = cache.delta_t_len();
        let mut map = vec![None; total_t];
        for row in 0..self.n_obs() {
            let base = cache.row_offsets[row];
            let vars = self.row_vars_for_cache_row(row, cache)?;
            for (a, va) in vars.iter().enumerate() {
                if let SaeLocalRowVar::Coord { atom, axis } = *va {
                    map[base + a] = Some((atom, axis));
                }
            }
        }
        Ok(map)
    }

    /// #2336 — the t-derivative diagonal of the ARD concave-clamp remainder E,
    /// `∂E_rr/∂t_r = w_row·κ²·grad·[hess<0]` (companion to
    /// `materialize_ard_concave_clamp_diagonal`; `grad = (α/κ)·sin κt`,
    /// `hess = α·cos κt`, so `κ²·grad = α·κ·sin κt = ∂(−min(hess,0))/∂t` on the
    /// concave half, 0 elsewhere). ThresholdGate logits carry the derivative of
    /// their own `majorized - exact` remainder from the shared curvature seam.
    pub(crate) fn ard_concave_clamp_dt_diagonal(
        &self,
        rho: &SaeManifoldRho,
        cache: &ArrowFactorCache,
    ) -> Result<Array1<f64>, String> {
        let total_t = cache.delta_t_len();
        let mut dt = Array1::<f64>::zeros(total_t);
        if self.k_atoms() == 0 {
            return Ok(dt);
        }
        let ard_axis_periods: Vec<Vec<Option<f64>>> = self
            .assignment
            .coords
            .iter()
            .map(|coord| coord.effective_axis_periods())
            .collect();
        let ard_precisions = self.validated_ard_precisions(rho)?;
        let row_loss_w = self.row_loss_weights.as_deref();
        for row in 0..self.n_obs() {
            let base = cache.row_offsets[row];
            let vars = self.row_vars_for_cache_row(row, cache)?;
            let w_row = row_loss_w.map_or(1.0, |w| w[row]);
            for (a, va) in vars.iter().enumerate() {
                if let SaeLocalRowVar::Logit { atom } = *va {
                    if self.assignment.logit_is_fixed(atom) {
                        continue;
                    }
                    if let AssignmentMode::ThresholdGate {
                        temperature,
                        threshold,
                    } = self.assignment.mode
                    {
                        let curvature = crate::assignment::ThresholdGateLogitCurvature::eval(
                            w_row * rho.lambda_sparse()?,
                            self.assignment.logits[[row, atom]],
                            threshold,
                            1.0 / temperature,
                        );
                        dt[base + a] = curvature.majorized_hess_logit_derivative()
                            - curvature.exact_hess_logit_derivative();
                    }
                    continue;
                }
                let SaeLocalRowVar::Coord { atom, axis } = *va else {
                    continue;
                };
                if rho.log_ard[atom].is_empty() {
                    continue;
                }
                let Some(period) = ard_axis_periods[atom][axis] else {
                    continue; // non-periodic axis: hess = α > 0, clamp never bites.
                };
                let alpha = ard_precisions[atom][axis];
                let t_val = self.assignment.coords[atom].row(row)[axis];
                let prior = ArdAxisPrior::eval(alpha, t_val, Some(period));
                let kappa = std::f64::consts::TAU / period;
                // #2339 smooth clamp: E = hess_majorized − hess = α·softplus_τ(−cos κt),
                // so ∂E/∂t = κ²·grad·(1 − clamp_slope(cos κt)) (clamp_slope = logistic(cos/τ);
                // τ→0 recovers the hard-clamp κ²·grad·[cos<0]).
                let cos = prior.hess / alpha;
                let contrib = kappa * kappa * prior.grad * (1.0 - ArdAxisPrior::clamp_slope(cos));
                if contrib != 0.0 {
                    dt[base + a] += w_row * contrib;
                }
            }
        }
        Ok(dt)
    }

    /// Chain the diagonal E derivative of one priced block to the model's
    /// strength and theta coordinates. Both results use full logdet units.
    fn priced_clamp_adjoint_extras(
        &self,
        rho: &SaeManifoldRho,
        cache: &ArrowFactorCache,
        pricing: &ExactHessianPricing,
    ) -> Result<(Array1<f64>, Array1<f64>), String> {
        let total_t = cache.delta_t_len();
        if pricing.clamp_diagonal_derivative.len() != total_t {
            return Err("priced clamp derivative has the wrong coordinate dimension".to_string());
        }
        let e_diag = self.materialize_ard_concave_clamp_diagonal(rho, cache)?;
        let de_dt = self.ard_concave_clamp_dt_diagonal(rho, cache)?;
        let coord_axis = self.coord_axis_map_for_cache(cache)?;
        let mut strength_flat: Vec<Option<usize>> = coord_axis
            .iter()
            .map(|axis| axis.map(|(atom, axis)| rho.ard_flat_index(atom, axis)))
            .collect();
        if matches!(self.assignment.mode, AssignmentMode::ThresholdGate { .. }) {
            for row in 0..self.n_obs() {
                let variables = self.row_vars_for_cache_row(row, cache)?;
                for (local, variable) in variables.iter().enumerate() {
                    let slot = cache.row_offsets[row] + local;
                    if matches!(variable, SaeLocalRowVar::Logit { .. }) && e_diag[slot] != 0.0 {
                        strength_flat[slot] = Some(rho.sparse_flat_index().ok_or_else(|| {
                            "nonzero threshold-gate clamp has no sparse strength coordinate"
                                .to_string()
                        })?);
                    }
                }
            }
        }
        let mut delta_trace = Array1::<f64>::zeros(rho.to_flat().len());
        let mut delta_gamma_t = Array1::<f64>::zeros(total_t);
        for r in 0..total_t {
            let weight = pricing.clamp_diagonal_derivative[r];
            if let Some(flat) = strength_flat[r] {
                let contribution = weight * e_diag[r];
                if contribution != 0.0 {
                    delta_trace[flat] += contribution;
                }
            }
            delta_gamma_t[r] = weight * de_dt[r];
        }
        Ok((delta_trace, delta_gamma_t))
    }

    /// #2500 — does [`Self::logdet_theta_adjoint_dense`] model this fit's
    /// assignment family? The exact-A logdet channels are taken only where it
    /// does; elsewhere the B-majorizer channels stand, because those are modelled
    /// for EVERY family.
    ///
    /// The dense reconstruction carries three prior legs: the softmax entropy
    /// Gershgorin majorizer (`want_entropy`), the ordered-Beta–Bernoulli Patch-D
    /// cross-row adjoint, and the periodic-ARD majorizer diagonal. It carries no
    /// per-atom-logistic GATE legs, which is what a `ThresholdGate` row needs.
    /// `TopK` mints no free logit at all, so there is
    /// nothing for a gate leg to model and the reconstruction is complete there by
    /// construction.
    ///
    /// MEASURED on `threshold_gate_tiny_fixture`, dense vs the production
    /// `logdet_theta_adjoint` on the SAME inverse:
    ///
    /// ```text
    ///   deflation-free arm   worst |production − dense| = 2.53e0  (on a 8.91e-1 entry)
    ///   deflating arm        worst |production − dense| = 3.31e0  (on a −2.50e-1 entry)
    /// ```
    ///
    /// and against a central finite difference of `½(log|A| − log|A_tt|)` (the
    /// criterion's complexity when this was measured) in a logit, the dense Γ read `1.373e-1` where the FD read `3.521e-1`, with a
    /// SIGN FLIP on the next logit. So this is not a tolerance question.
    ///
    /// Before the sparse operator was modelled, a ThresholdGate fit could not
    /// reach this code at all — the sparse curvature operator map refused
    /// first — so the gap was unreachable rather than absent. Gating here keeps
    /// that family on the fully-modelled `½log|B|` channels (`logdet_theta_adjoint`
    /// and `assignment_log_strength_hessian_trace` both carry it), which is the
    /// same staged position the matrix-free and bundle routes already hold until
    /// Phase-2b.
    fn dense_exact_a_theta_adjoint_is_modelled(&self) -> bool {
        match self.assignment.mode {
            AssignmentMode::Softmax { .. }
            | AssignmentMode::OrderedBetaBernoulli { .. }
            | AssignmentMode::TopK { .. } => true,
            AssignmentMode::ThresholdGate { .. } => false,
        }
    }

    pub(crate) fn dense_exact_a_logdet_channels(
        &self,
        target: ArrayView2<'_, f64>,
        rho: &SaeManifoldRho,
        loss: &SaeManifoldLoss,
        cache: &ArrowFactorCache,
    ) -> Result<DenseExactALogdetChannels, String> {
        let n_params = rho.to_flat().len();
        let geometry = self.materialize_exact_hessian_quotient_geometry(rho, target, cache)?;
        // The common basin owner includes the negative-subspace response in
        // dA. Chain its remaining explicit dE term to rho and theta here.
        let (priced_joint_trace, priced_joint_gamma) =
            self.priced_clamp_adjoint_extras(rho, cache, &geometry.joint_pricing)?;
        let a_pinv = &geometry.joint_pricing.a_derivative;
        // This value diagonalizes `A_raw = B_raw + ΔC`; differentiate that raw
        // operator, not the row-conditioned operator carried by arrow factors.
        let da_by_flat = self.exact_stationarity_penalty_derivatives_by_flat(rho, cache)?;
        let frob = |x: &Array2<f64>, y: &Array2<f64>| -> f64 { (x * y).sum() };
        let mut logdet_trace = Array1::<f64>::zeros(n_params);
        for (&i, da) in da_by_flat.iter() {
            logdet_trace[i] = 0.5 * frob(&a_pinv, da);
        }
        // Ordered-Beta–Bernoulli sparse coordinate: its ∂A/∂ρ_sparse is the exact
        // integrated-marginal logit Hessian (cross-row), absent from the operator
        // map above (softmax-only). Add its ½log|A| trace directly.
        if let Some(sparse) = rho.sparse_flat_index() {
            if matches!(
                self.assignment.mode,
                AssignmentMode::OrderedBetaBernoulli { .. }
            ) {
                logdet_trace[sparse] =
                    self.dense_exact_a_ordered_bb_sparse_trace(rho, cache, &a_pinv)?;
            }
        }
        let mut gamma = self.logdet_theta_adjoint_dense(
            rho,
            cache,
            &a_pinv,
            true,
            true,
            Some(target),
        )?;
        // The scalar criterion carries a leading half; Gamma is the full
        // log-determinant theta derivative used in the caller's -1/2 IFT fold.
        logdet_trace.scaled_add(0.5, &priced_joint_trace);
        gamma.t += &priced_joint_gamma;
        gamma.beta += &self.decoder_prior_gap_theta_trace(
            cache, geometry.joint_pricing.clamp_border_derivative.view(),
        )?;
        let rank_charge = self.production_rank_charge_derivative(target, rho, loss, cache)?;
        gamma.t.scaled_add(2.0, &rank_charge.theta.t);
        gamma.beta.scaled_add(2.0, &rank_charge.theta.beta);
        let stationarity_adjoint = geometry.joint.solve_stationarity(&gamma)?;
        Ok(DenseExactALogdetChannels {
            logdet_trace,
            theta_adjoint: gamma,
            stationarity_adjoint,
        })
    }

    /// #2330 — the ordered-Beta–Bernoulli (non-softmax) sparse-coordinate ½log|A|
    /// trace `½[tr(A⁺ ∂A/∂ρ_sparse) − tr(A_tt⁺ ∂A/∂ρ_sparse)]`. For the
    /// non-learnable prior `∂A/∂ρ_sparse` is the EXACT integrated-marginal logit
    /// Hessian `H_obb` (linear-in-`weight` proof on the parent issue): its column
    /// `H_obb·e_j = ΔC_obb·e_j (cross-row HVP) + hdiag[j]·e_j (majorizer diagonal)`.
    /// The operator lives on logit t-slots only (no β border), so the coordinate
    /// block reuses the same columns against `A_tt⁺`. Learnable α (nonlinear
    /// concentration derivative) is refused, not silently mispriced.
    /// #2330 Patch D — the ordered-Beta--Bernoulli prior curvature θ-adjoint
    /// `Σ_{i,j} inv[i,j]·∂H_obb[i,j]/∂ℓ_w`, the full prior contribution the
    /// reconstruction and softmax-only row loops cannot carry. Per column `c`,
    /// `H_obb = weight·S'_c·uuᵀ + diag(D_i)` with `u_i = w_i·z_i(1−z_i)/τ`,
    /// `curv_i = z_i(1−z_i)(1−2z_i)/τ²`, `D_i = weight·S_c·w_i·curv_i`. Its logit
    /// derivative contracts to (with `P = uᵀ inv_cc u`, `(inv·u)_r`,
    /// `G = Σ_i inv[i,i]·w_i·curv_i`, `curv'_i = z_i(1−z_i)(1−6z_i+6z_i²)/τ³`):
    ///   `Γ[w=(r,c)] = weight·{ S''_c·u_r·P + 2·S'_c·w_r·curv_r·(inv·u)_r
    ///                          + S'_c·u_r·G + S_c·inv[r,r]·w_r·curv'_r }`.
    /// Contracts the matrix the caller passes as `inv` (production passes `A⁺`
    /// of the joint operator), on the logit t-slots.
    fn dense_exact_a_ordered_bb_logit_theta_adjoint(
        &self,
        cache: &ArrowFactorCache,
        inv: &Array2<f64>,
        data: &gam_terms::analytic_penalties::OrderedBetaBernoulliLogitAdjointData,
    ) -> Result<Array1<f64>, String> {
        let n = data.n;
        let k = data.k_max;
        let weight = data.weight;
        let inv_tau = 1.0 / data.tau;
        let inv_tau2 = inv_tau * inv_tau;
        let inv_tau3 = inv_tau2 * inv_tau;
        let total_t = cache.delta_t_len();
        let mut out = Array1::<f64>::zeros(total_t);
        // Global t-slot of each (row, column) logit in the cache layout.
        let mut gindex: Vec<Vec<Option<usize>>> = vec![vec![None; k]; n];
        for row in 0..n {
            let base = cache.row_offsets[row];
            let vars = self.row_vars_for_cache_row(row, cache)?;
            for (local, var) in vars.iter().enumerate() {
                if let SaeLocalRowVar::Logit { atom } = *var {
                    if atom < k {
                        gindex[row][atom] = Some(base + local);
                    }
                }
            }
        }
        // Structural quantities per (row, column): (u, curv, curv', w).
        let uval = |row: usize, col: usize| -> (f64, f64, f64, f64) {
            let z = data.z[row * k + col];
            let w = data.row_weight[row];
            let zc = z * (1.0 - z);
            let u = w * zc * inv_tau;
            let curv = zc * (1.0 - 2.0 * z) * inv_tau2;
            let curvp = zc * (1.0 - 6.0 * z + 6.0 * z * z) * inv_tau3;
            (u, curv, curvp, w)
        };
        for col in 0..k {
            if data.column_fixed[col] {
                continue;
            }
            let s = data.score[col];
            let sp = data.score_derivative[col];
            let spp = data.score_second[col];
            let rows: Vec<usize> = (0..n).filter(|&r| gindex[r][col].is_some()).collect();
            let mut au = vec![0.0_f64; n];
            let mut p = 0.0_f64;
            let mut g = 0.0_f64;
            for &ri in &rows {
                let gi = gindex[ri][col].expect("row filtered to Some");
                let (ui, curvi, _curvpi, wi) = uval(ri, col);
                let mut au_ri = 0.0_f64;
                for &rj in &rows {
                    let gj = gindex[rj][col].expect("row filtered to Some");
                    let (uj, _, _, _) = uval(rj, col);
                    au_ri += inv[[gi, gj]] * uj;
                }
                au[ri] = au_ri;
                p += ui * au_ri;
                g += inv[[gi, gi]] * wi * curvi;
            }
            for &ri in &rows {
                let gi = gindex[ri][col].expect("row filtered to Some");
                let (ui, curvi, curvpi, wi) = uval(ri, col);
                let val = spp * ui * p
                    + 2.0 * sp * wi * curvi * au[ri]
                    + sp * ui * g
                    + s * inv[[gi, gi]] * wi * curvpi;
                out[gi] += weight * val;
            }
        }
        Ok(out)
    }

    pub(crate) fn dense_exact_a_ordered_bb_sparse_trace(
        &self,
        rho: &SaeManifoldRho,
        cache: &ArrowFactorCache,
        a_pinv: &Array2<f64>,
    ) -> Result<f64, String> {
        let k_atoms = self.k_atoms();
        let n = self.n_obs();
        let row_weights = self.row_loss_weights.as_deref();
        // Global t-index of each (row, atom) logit slot in the cache layout.
        let mut logit_gindex: Vec<Vec<Option<usize>>> = vec![vec![None; k_atoms]; n];
        for row in 0..n {
            let base = cache.row_offsets[row];
            let vars = self.row_vars_for_cache_row(row, cache)?;
            for (local, var) in vars.iter().enumerate() {
                if let SaeLocalRowVar::Logit { atom } = *var {
                    if atom < k_atoms {
                        logit_gindex[row][atom] = Some(base + local);
                    }
                }
            }
        }
        // ∂B/∂ρ_sparse: the majorizer's diagonal log-strength derivative on the
        // logit slots — the SAME builder the B-majorizer trace uses.
        let mut hdiag = crate::assignment::assignment_prior_log_strength_hdiag_weighted(
            &self.assignment,
            rho,
            row_weights,
        )?;
        if hdiag.is_empty() {
            // Inert / frozen prior: ∂B and ΔC are both zero.
            return Ok(0.0);
        }
        let channels = ordered_beta_bernoulli_psd_majorizer_third_channels_weighted(
            &self.assignment,
            rho,
            row_weights,
        )?;
        if self.assignment.effective_alpha_is_learnable() {
            // A learnable concentration makes `ρ_sparse = log(α/α_base)` move only the
            // Beta shapes `a_k` (the prior weight stays one), so on the logit block
            // `∂A/∂ρ_sparse` is the concentration derivative of the EXACT prior Hessian:
            // `B` and `ΔC` split one operator and their majorizer parts cancel. Per atom
            // column that derivative is `∂S'_k/∂ρ·u uᵀ + diag(∂S_k/∂ρ·w_i·curv_i)` with
            // `u = w·J` (`z_jac`), and `hdiag` already holds its full diagonal
            // `∂S'_k/∂ρ·u_i² + ∂S_k/∂ρ·w_i·curv_i`. The trace is one rank-one quadratic
            // form per column plus the row-local part of the diagonal.
            let ch = channels.as_ref().ok_or_else(|| {
                "dense_exact_a_ordered_bb_sparse_trace: a learnable concentration needs the \
                 ordered Beta--Bernoulli prior channels"
                    .to_string()
            })?;
            let mut tr_joint = 0.0_f64;
            for atom in 0..k_atoms {
                let mass_curvature = ch.mass_hessian_log_alpha_derivative[atom];
                let mut quadratic = 0.0_f64;
                for irow in 0..n {
                    let Some(gi) = logit_gindex[irow][atom] else {
                        continue;
                    };
                    let islot = irow * k_atoms + atom;
                    let ui = ch.z_jac[islot];
                    tr_joint += a_pinv[[gi, gi]] * (hdiag[islot] - mass_curvature * ui * ui);
                    for jrow in 0..n {
                        let Some(gj) = logit_gindex[jrow][atom] else {
                            continue;
                        };
                        quadratic += a_pinv[[gi, gj]] * ui * ch.z_jac[jrow * k_atoms + atom];
                    }
                }
                tr_joint += mass_curvature * quadratic;
            }
            return Ok(0.5 * tr_joint);
        }
        if let Some(ch) = channels.as_ref() {
            for row in 0..n {
                for atom in 0..k_atoms {
                    let slot = row * k_atoms + atom;
                    hdiag[slot] =
                        super::construction_arrow_schur_assembly::ordered_beta_bernoulli_psd_majorized_hdiag(
                            ch, row, k_atoms, atom, hdiag[slot],
                        );
                }
            }
        }
        // ½tr(A⁺ ∂A/∂ρ_sparse), column by column over
        // the flat logit basis: ∂A/∂ρ_sparse·e_j = ΔC_obb·e_j + hdiag[j]·e_j.
        let n_logits = n * k_atoms;
        let mut e = Array1::<f64>::zeros(n_logits);
        let mut tr_joint = 0.0_f64;
        for jrow in 0..n {
            for jatom in 0..k_atoms {
                let Some(gj) = logit_gindex[jrow][jatom] else {
                    continue;
                };
                let jflat = jrow * k_atoms + jatom;
                e[jflat] = 1.0;
                let dc = crate::assignment::ordered_beta_bernoulli_exact_hessian_minus_majorizer_hvp_weighted(
                    &self.assignment,
                    rho,
                    row_weights,
                    e.view(),
                )?;
                e[jflat] = 0.0;
                for irow in 0..n {
                    for iatom in 0..k_atoms {
                        let val = dc[irow * k_atoms + iatom];
                        if val == 0.0 {
                            continue;
                        }
                        if let Some(gi) = logit_gindex[irow][iatom] {
                            tr_joint += a_pinv[[gi, gj]] * val;
                        }
                    }
                }
                tr_joint += a_pinv[[gj, gj]] * hdiag[jflat];
            }
        }
        Ok(0.5 * tr_joint)
    }

    /// Assemble `ΔC = A − B` per row, so the arrow evidence system can carry the
    /// EXACT observed information instead of the Newton/Schur majorizer.
    ///
    /// `Self::apply_exact_hessian_minus_b` contracts these blocks against a
    /// direction without ever forming them, which is all a matvec consumer needs.
    /// The streaming log-determinant is not a matvec consumer: it takes
    /// `log|H_tt^(i)|` off assembled per-row factors and reduces an assembled
    /// border, so it can only price `A` if the blocks exist (#2509).
    ///
    /// Every channel is row-local — (1a)/(1b) residual curvature, (2) the softmax
    /// entropy-minus-Gershgorin delta, (3) the periodic ARD concave clamp, (3b) the
    /// ThresholdGate concave remainder on logit slots — except
    /// ordered Beta–Bernoulli, whose integrated-marginal prior couples every row
    /// within an atom column. That mode has no arrow-structured `ΔC` and is
    /// REFUSED here rather than silently dropped: pricing `B` while claiming `A`
    /// is the defect this function exists to remove.
    ///
    /// These are the row and cross-block legs only. Leg (5), `ΔC_ββ`, lives on the
    /// border, and `Self::exact_a_evidence_system` composes it into the shared
    /// block (#2828).
    pub(crate) fn assemble_exact_hessian_minus_b_rows(
        &self,
        rho: &SaeManifoldRho,
        target: ArrayView2<'_, f64>,
        row_dims: &[usize],
        border_dim: usize,
    ) -> Result<Vec<ExactHessianDeltaRow>, String> {
        self.assignment.validate_rho_domain(rho)?;
        if matches!(
            self.assignment.mode,
            AssignmentMode::OrderedBetaBernoulli { .. }
        ) {
            return Err(
                "assemble_exact_hessian_minus_b_rows: the ordered Beta-Bernoulli prior couples \
                 every row within an atom column, so A - B has no per-row arrow block; this \
                 route must refuse rather than assemble a majorizer and call it exact (#2509)"
                    .to_string(),
            );
        }
        let p = self.output_dim();
        let n = self.n_obs();
        let k_atoms = self.k_atoms();
        let second_jets = self.atom_second_jets()?;
        let border = self.border_channels_for_border_dim(border_dim)?;
        let row_loss_w = self.row_loss_weights.as_deref();
        let ard_axis_periods: Vec<Vec<Option<f64>>> = self
            .assignment
            .coords
            .iter()
            .map(|coord| coord.effective_axis_periods())
            .collect();
        let ard_precisions = self.validated_ard_precisions(rho)?;

        // Softmax entropy-minus-majorizer scale (#1419); `None` off softmax.
        let softmax_scale: Option<f64> = match self.assignment.mode {
            AssignmentMode::Softmax {
                temperature,
                sparsity,
            } if k_atoms > 1 => {
                let inv_tau = 1.0 / temperature;
                Some(rho.lambda_sparse()? * sparsity * inv_tau * inv_tau)
            }
            _ => None,
        };
        // (3b) #2520 — the ThresholdGate's concave remainder, from the producer the
        // applier and the clamp diagonal read; `None` off the threshold gate.
        let threshold_gate_remainder = match self.assignment.mode {
            AssignmentMode::ThresholdGate { .. } => Some(
                crate::assignment::threshold_gate_negative_hessian_remainder_weighted(
                    &self.assignment,
                    rho,
                    row_loss_w,
                )?,
            ),
            _ => None,
        };

        let whitens = self
            .row_metric
            .as_ref()
            .is_some_and(|metric| metric.whitens_likelihood());
        let mut decoded = vec![0.0_f64; p];
        let mut fitted = Array1::<f64>::zeros(p);
        let mut error = Array1::<f64>::zeros(p);
        let mut assignments = Array1::<f64>::zeros(k_atoms);

        let mut rows_out: Vec<ExactHessianDeltaRow> = Vec::with_capacity(n);
        let mut jet_window: std::collections::VecDeque<SaeRowJets> =
            std::collections::VecDeque::new();
        let mut jet_window_next = 0usize;
        for row in 0..n {
            let q = row_dims[row];
            let a_scratch = assignments.as_slice_mut().ok_or_else(|| {
                "assemble_exact_hessian_minus_b_rows: assignment scratch is not contiguous"
                    .to_string()
            })?;
            self.assignment.try_assignments_row_into(row, a_scratch)?;
            if jet_window.is_empty() {
                jet_window_next = self.refill_jet_window_with_row_dims(
                    jet_window_next,
                    row_dims,
                    &second_jets,
                    &border,
                    &mut jet_window,
                )?;
            }
            let jets = jet_window
                .pop_front()
                .expect("jet window must be non-empty");
            let sqrt_row_w = row_loss_w.map_or(1.0, |w| w[row].sqrt());
            let w_row = row_loss_w.map_or(1.0, |w| w[row]);

            // The same sqrt(w)-scaled metric-applied residual the applier contracts.
            fitted.fill(0.0);
            let active_atoms = self
                .last_row_layout
                .as_ref()
                .map(|layout| layout.active_atoms[row].as_slice());
            for k in 0..k_atoms {
                if active_atoms.is_some_and(|active| active.binary_search(&k).is_err()) {
                    continue;
                }
                self.atoms[k].fill_decoded_row(row, &mut decoded);
                let a_k = assignments[k];
                for out_col in 0..p {
                    fitted[out_col] += a_k * decoded[out_col];
                }
            }
            for out_col in 0..p {
                error[out_col] = sqrt_row_w * (fitted[out_col] - target[[row, out_col]]);
            }
            let error_metric: Vec<f64> = match self.row_metric.as_ref() {
                Some(metric) if whitens => metric.apply_metric_row(row, error.view()),
                _ => error.to_vec(),
            };

            let mut tt = Array2::<f64>::zeros((q, q));
            let mut tbeta = Array2::<f64>::zeros((q, border.len()));

            // (1a) residual curvature, t-t.
            for a in 0..q {
                for b in 0..q {
                    tt[[a, b]] = sae_dot(&error_metric, jets.second(a, b));
                }
            }
            // (1b) residual curvature, t-beta. The beta-t block is its transpose;
            // the arrow system stores only this orientation.
            for a in 0..q {
                for beta_pos in 0..border.len() {
                    tbeta[[a, beta_pos]] = sae_dot(&error_metric, jets.beta_deriv(a, beta_pos));
                }
            }
            // (2) softmax exact entropy minus the Gershgorin majorizer written into B.
            if let Some(scale) = softmax_scale {
                let assignment_dim = self.assignment.assignment_coord_dim();
                let a_soft = assignments
                    .as_slice()
                    .expect("softmax assignments row must be contiguous");
                let m = softmax_majorizer_log_mean(a_soft);
                for (a, va) in jets.vars.iter().enumerate() {
                    let SaeLocalRowVar::Logit { atom: ka } = *va else {
                        continue;
                    };
                    if ka >= assignment_dim {
                        continue;
                    }
                    for (b, vb) in jets.vars.iter().enumerate() {
                        let SaeLocalRowVar::Logit { atom: kb } = *vb else {
                            continue;
                        };
                        if kb >= assignment_dim {
                            continue;
                        }
                        let h_entropy =
                            softmax_dense_entropy_hessian_entry(a_soft, ka, kb, m, scale);
                        let delta = if ka == kb {
                            h_entropy
                                - active_softmax_gershgorin_majorizer_entry(a_soft, ka, m, scale)
                        } else {
                            h_entropy
                        };
                        tt[[a, b]] += w_row * delta;
                    }
                }
            }
            // (3) periodic ARD concave clamp, diagonal on coordinate vars.
            for (a, va) in jets.vars.iter().enumerate() {
                let SaeLocalRowVar::Coord { atom, axis } = *va else {
                    continue;
                };
                if rho.log_ard[atom].is_empty() {
                    continue;
                }
                let alpha = ard_precisions[atom][axis];
                let t_val = self.assignment.coords[atom].row(row)[axis];
                let prior = ArdAxisPrior::eval(alpha, t_val, ard_axis_periods[atom][axis]);
                let neg = prior.negative_hessian_remainder();
                if neg != 0.0 {
                    tt[[a, a]] += w_row * neg;
                }
            }
            // (3b) #2520 threshold gate: the applier's channel (3b), on logit slots.
            // `B` carries the PSD clamp of the gate's curvature. Without the
            // non-positive remainder the arrow system prices `B` on every switched-on
            // logit, while the classification's clamp diagonal restores a concave
            // half the operator never subtracted (#2915). The producer already
            // applies `w_row` and the fixed-logit mask.
            if let Some(remainder) = threshold_gate_remainder.as_ref() {
                for (a, va) in jets.vars.iter().enumerate() {
                    let SaeLocalRowVar::Logit { atom } = *va else {
                        continue;
                    };
                    let neg = remainder[row * k_atoms + atom];
                    if neg != 0.0 {
                        tt[[a, a]] += neg;
                    }
                }
            }

            rows_out.push(ExactHessianDeltaRow { tt, tbeta });
        }
        Ok(rows_out)
    }
}

#[cfg(test)]
mod test_support {
    use super::Side;
    use super::{
        ArrowFactorCache, DeflatedArrowSolver, SaeArrowVector, SaeManifoldRho,
    };
    use gam_linalg::faer_ndarray::FaerEigh;
    use ndarray::{Array1, Array2};

    fn spectral_fixture(a: &ndarray::Array2<f64>) -> super::ExactHessianSpectralBlock {
        let (eigenvalues, eigenvectors) = a.eigh(Side::Lower).expect("symmetric fixture");
        let spectral_norm = eigenvalues.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
        super::ExactHessianSpectralBlock {
            operator: a.clone(),
            eigenvalues,
            eigenvectors,
            metric_scale: Array1::ones(a.nrows()),
            substituted_stiffness: Array1::zeros(a.nrows()),
            spectral_norm,
        }
    }

    struct PricedFixture {
        log_det: f64,
        a_derivative: ndarray::Array2<f64>,
        clamp_diagonal_derivative: Array1<f64>,
    }

    #[test]
    fn ordered_bb_batched_exact_hessian_preserves_cross_row_mass_curvature_2820() {
        for weighted in [false, true] {
            let (mut term, target, rho) =
                crate::manifold::tests_logdet_adjoint_780::obb_patchd_fixture(0.01, -1.0);
            if weighted {
                term.set_row_loss_weights(
                    (0..term.n_obs())
                        .map(|row| 0.5 + (row % 4) as f64 * 0.25)
                        .collect(),
                )
                .expect("positive design weights");
            }
            // Operator equality applies at indefinite states too. Build the
            // positive Newton metric directly, without asking the evidence
            // criterion to admit this deliberately nonstationary state.
            let system = term
                .assemble_arrow_schur(target.view(), &rho, None)
                .expect("fixed-state OBB assembly");
            let (_, _, cache) = super::solve_arrow_newton_step_with_options(
                &system,
                1.0e-6,
                1.0e-6,
                &super::ArrowSolveOptions::direct(),
            )
            .expect("positive Newton metric");
            let oracle = term
                .materialize_exact_hessian_dense_by_columns(&rho, target.view(), &cache)
                .expect("independent column-probe oracle");
            let batched = term
                .materialize_exact_hessian_dense(&rho, target.view(), &cache)
                .expect("arrow plus mass assembly");
            let mut cross_row_signal = 0.0_f64;
            for row in 0..cache.n_rows() {
                for i in cache.row_offsets[row]..cache.row_offsets[row + 1] {
                    for j in cache.row_offsets[row + 1]..cache.delta_t_len() {
                        cross_row_signal = cross_row_signal.max(oracle[[i, j]].abs());
                    }
                }
            }
            assert!(
                cross_row_signal > 1.0e-4,
                "cross-row mass block must be live"
            );
            let scale = oracle
                .iter()
                .map(|value| value.abs())
                .fold(1.0_f64, f64::max);
            let error = (&batched - &oracle)
                .iter()
                .map(|value| value.abs())
                .fold(0.0_f64, f64::max);
            assert!(
                error <= 1.0e-12 * scale,
                "weighted={weighted}: batched/column error={error:e}, scale={scale:e}"
            );
        }
    }

    #[test]
    fn ordered_bb_full_prior_theta_adjoint_matches_existing_hvp_difference_2820() {
        for weighted in [false, true] {
            let (mut term, target, rho) =
                crate::manifold::tests_logdet_adjoint_780::obb_patchd_fixture(0.01, -1.0);
            if weighted {
                term.set_row_loss_weights(
                    (0..term.n_obs())
                        .map(|row| 0.5 + (row % 4) as f64 * 0.25)
                        .collect(),
                )
                .expect("positive design weights");
            }
            let system = term
                .assemble_arrow_schur(target.view(), &rho, None)
                .expect("OBB assembly");
            let (_, _, cache) = super::solve_arrow_newton_step_with_options(
                &system,
                1.0e-6,
                1.0e-6,
                &super::ArrowSolveOptions::direct(),
            )
            .expect("Newton metric");
            let mut sites = Vec::new();
            for row in 0..term.n_obs() {
                for (local, var) in term
                    .row_vars_for_cache_row(row, &cache)
                    .expect("cache row layout")
                    .iter()
                    .enumerate()
                {
                    if let super::SaeLocalRowVar::Logit { atom } = *var {
                        sites.push((row * term.k_atoms() + atom, cache.row_offsets[row] + local));
                    }
                }
            }
            let dim = cache.delta_t_len() + cache.k;
            let weight = ndarray::Array2::from_shape_fn((dim, dim), |(i, j)| {
                ((i + j + 1) as f64 * 0.17).cos() + if i == j { 2.0 } else { 0.0 }
            });
            let data = crate::assignment::ordered_beta_bernoulli_logit_adjoint_data_weighted(
                &term.assignment,
                &rho,
                term.row_loss_weights.as_deref(),
            )
            .expect("prior data")
            .expect("OBB family");
            let analytic = term
                .dense_exact_a_ordered_bb_logit_theta_adjoint(&cache, &weight, &data)
                .expect("full prior adjoint");
            let channels = super::ordered_beta_bernoulli_psd_majorizer_third_channels_weighted(
                &term.assignment,
                &rho,
                term.row_loss_weights.as_deref(),
            )
            .expect("prior channels")
            .expect("OBB family");
            assert!(channels.diagonal_term.iter().any(|&x| x > 0.0));
            assert!(channels.diagonal_term.iter().any(|&x| x < 0.0));
            let contraction = |moved: &super::SaeManifoldTerm| {
                let channels = super::ordered_beta_bernoulli_psd_majorizer_third_channels_weighted(
                    &moved.assignment,
                    &rho,
                    moved.row_loss_weights.as_deref(),
                )
                .expect("perturbed channels")
                .expect("OBB family");
                let mut unit = Array1::zeros(moved.assignment.logits.len());
                let mut trace = 0.0;
                for &(flat_col, global_col) in &sites {
                    unit[flat_col] = 1.0;
                    let mut column = crate::assignment::ordered_beta_bernoulli_exact_hessian_minus_majorizer_hvp_weighted(
                        &moved.assignment, &rho, moved.row_loss_weights.as_deref(), unit.view(),
                    ).expect("existing exact HVP remainder");
                    unit[flat_col] = 0.0;
                    column[flat_col] += channels.diagonal_term[flat_col].max(0.0);
                    for &(flat_row, global_row) in &sites {
                        trace += weight[[global_row, global_col]] * column[flat_row];
                    }
                }
                trace
            };
            let h = 1.0e-5;
            let mut plus = term.clone();
            let mut minus = term.clone();
            let mut expected = 0.0;
            for &(flat, global) in &sites {
                let direction = ((flat + 1) as f64 * 0.31).sin();
                let row = flat / term.k_atoms();
                let atom = flat % term.k_atoms();
                plus.assignment.logits[[row, atom]] += h * direction;
                minus.assignment.logits[[row, atom]] -= h * direction;
                expected += analytic[global] * direction;
            }
            let fd = (contraction(&plus) - contraction(&minus)) / (2.0 * h);
            assert!(
                (fd - expected).abs() <= 1.0e-7 * (1.0 + expected.abs()),
                "weighted={weighted}: analytic={expected:e}, fd={fd:e}"
            );
        }
    }

    /// #2822 — with a learnable concentration `ρ_sparse = log(α/α_base)`, the sparse
    /// ½log|A| trace contracts the concentration derivative of the exact prior Hessian.
    /// The reference reads the exact Hessian's columns off the existing HVP remainder
    /// plus its majorizer diagonal, differentiates them centrally in `ρ_sparse` at a
    /// fixed state, and contracts against an arbitrary symmetric matrix: the trace is
    /// linear in `A⁺`, so this isolates the channel algebra from the pseudo-inverse.
    #[test]
    fn ordered_bb_learnable_alpha_sparse_trace_matches_exact_prior_hessian_difference_2822() {
        for weighted in [false, true] {
            let (mut term, target, rho) =
                crate::manifold::tests_logdet_adjoint_780::obb_patchd_fixture(0.01, -1.0);
            let super::AssignmentMode::OrderedBetaBernoulli {
                temperature, alpha, ..
            } = term.assignment.mode
            else {
                panic!("the patch-D fixture is an ordered Beta--Bernoulli term");
            };
            term.assignment.mode =
                super::AssignmentMode::ordered_beta_bernoulli(temperature, alpha, true);
            assert!(term.assignment.effective_alpha_is_learnable());
            if weighted {
                term.set_row_loss_weights(
                    (0..term.n_obs())
                        .map(|row| 0.5 + (row % 4) as f64 * 0.25)
                        .collect(),
                )
                .expect("positive design weights");
            }
            let system = term
                .assemble_arrow_schur(target.view(), &rho, None)
                .expect("learnable-concentration OBB assembly");
            let (_, _, cache) = super::solve_arrow_newton_step_with_options(
                &system,
                1.0e-6,
                1.0e-6,
                &super::ArrowSolveOptions::direct(),
            )
            .expect("Newton metric");
            let mut sites = Vec::new();
            for row in 0..term.n_obs() {
                for (local, var) in term
                    .row_vars_for_cache_row(row, &cache)
                    .expect("cache row layout")
                    .iter()
                    .enumerate()
                {
                    if let super::SaeLocalRowVar::Logit { atom } = *var {
                        sites.push((row * term.k_atoms() + atom, cache.row_offsets[row] + local));
                    }
                }
            }
            let dim = cache.delta_t_len() + cache.k;
            let weight = ndarray::Array2::from_shape_fn((dim, dim), |(i, j)| {
                ((i + j + 1) as f64 * 0.17).cos() + if i == j { 2.0 } else { 0.0 }
            });
            let analytic = term
                .dense_exact_a_ordered_bb_sparse_trace(&rho, &cache, &weight)
                .expect("learnable-concentration sparse trace");
            let contraction = |moved: &super::SaeManifoldRho| {
                let channels = super::ordered_beta_bernoulli_psd_majorizer_third_channels_weighted(
                    &term.assignment,
                    moved,
                    term.row_loss_weights.as_deref(),
                )
                .expect("perturbed channels")
                .expect("OBB family");
                let mut unit = Array1::zeros(term.assignment.logits.len());
                let mut trace = 0.0;
                for &(flat_col, global_col) in &sites {
                    unit[flat_col] = 1.0;
                    let mut column = crate::assignment::ordered_beta_bernoulli_exact_hessian_minus_majorizer_hvp_weighted(
                        &term.assignment, moved, term.row_loss_weights.as_deref(), unit.view(),
                    ).expect("existing exact HVP remainder");
                    unit[flat_col] = 0.0;
                    column[flat_col] += channels.diagonal_term[flat_col].max(0.0);
                    for &(flat_row, global_row) in &sites {
                        trace += weight[[global_row, global_col]] * column[flat_row];
                    }
                }
                0.5 * trace
            };
            let h = 1.0e-5;
            let mut plus = rho.clone();
            let mut minus = rho.clone();
            plus.log_lambda_sparse += h;
            minus.log_lambda_sparse -= h;
            let fd = (contraction(&plus) - contraction(&minus)) / (2.0 * h);
            assert!(
                analytic.is_finite() && analytic != 0.0,
                "weighted={weighted}: the concentration channel must carry signal; analytic={analytic:e}"
            );
            assert!(
                (fd - analytic).abs() <= 1.0e-7 * (1.0 + analytic.abs()),
                "weighted={weighted}: analytic={analytic:e}, fd={fd:e}"
            );
        }
    }

    fn price_fixture(
        a: &ndarray::Array2<f64>,
        e: &Array1<f64>,
    ) -> Result<PricedFixture, super::SaeCriterionError> {
        let block = spectral_fixture(a);
        let basin = super::SaeManifoldTerm::classify_exact_hessian_basin(
            &block,
            e,
            None,
            e.len(),
            |v| Ok(v.dot(&v)),
            "fixture",
            None,
        )?;
        let differential = super::SaeManifoldTerm::exact_hessian_basin_differential(
            &block,
            e,
            None,
            e.len(),
            &basin,
        )?;
        Ok(PricedFixture {
            log_det: basin.log_det,
            a_derivative: differential.a_derivative,
            clamp_diagonal_derivative: differential.clamp_diagonal_derivative,
        })
    }

    #[test]
    fn priced_basin_is_continuous_and_basis_invariant_at_a_repeated_negative_eigenvalue_2820() {
        let e = Array1::from_vec(vec![2.0, 4.0]);
        for epsilon in [0.0, 1.0e-8, 1.0e-4] {
            let a = ndarray::arr2(&[[-1.0, epsilon], [epsilon, -1.0]]);
            let priced = price_fixture(&a, &e).expect("positive basin");
            let expected = (3.0 - epsilon * epsilon).ln();
            assert!(
                (priced.log_det - expected).abs() <= 1.0e-12,
                "epsilon={epsilon:e}: price={}, expected={expected}",
                priced.log_det
            );
        }
        let a = -ndarray::Array2::<f64>::eye(2);
        for angle in [0.0_f64, 0.31, std::f64::consts::FRAC_PI_4] {
            let mut block = spectral_fixture(&a);
            let rotation =
                ndarray::arr2(&[[angle.cos(), -angle.sin()], [angle.sin(), angle.cos()]]);
            block.eigenvectors = block.eigenvectors.dot(&rotation);
            let basin = super::SaeManifoldTerm::classify_exact_hessian_basin(
                &block,
                &e,
                None,
                2,
                |v| Ok(v.dot(&v)),
                "rotated fixture",
                None,
            )
            .expect("same negative subspace");
            let priced = super::SaeManifoldTerm::exact_hessian_basin_differential(
                &block, &e, None, 2, &basin,
            )
            .expect("same negative-subspace differential");
            assert!((basin.log_det - 3.0_f64.ln()).abs() <= 1.0e-12);
            let inverse = ndarray::arr2(&[[1.0, 0.0], [0.0, 1.0 / 3.0]]);
            assert!(
                (&priced.a_derivative - &inverse)
                    .iter()
                    .all(|x| x.abs() <= 1.0e-12)
            );
        }
    }

    #[test]
    fn priced_basin_refuses_an_indefinite_block_with_positive_diagonal_prices_2820() {
        let a = ndarray::arr2(&[[-1.0, 0.01], [0.01, -1.0]]);
        let e = Array1::from_vec(vec![0.5, 3.0]);
        let block = spectral_fixture(&a);
        for i in 0..2 {
            let vector = block.eigenvectors.column(i);
            let diagonal_price =
                block.eigenvalues[i] + (0..2).map(|r| e[r] * vector[r] * vector[r]).sum::<f64>();
            assert!(
                diagonal_price > 0.7,
                "the old diagonal-only rule must admit this fixture"
            );
        }
        assert!(matches!(
            price_fixture(&a, &e),
            Err(super::SaeCriterionError::IndefiniteObservedInformation { .. })
        ));
    }

    #[test]
    fn priced_basin_differential_covers_a_rotated_negative_cluster_and_positive_complement_2820() {
        // Eigenvalues (-1,-1,2); the positive eigenvector has equal components.
        // Nonuniform E couples the repeated negative cluster to its complement.
        let a = ndarray::arr2(&[[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]);
        let e = Array1::from_vec(vec![2.0, 4.0, 3.0]);
        let priced = price_fixture(&a, &e).expect("positive restricted basin");
        let positive = Array1::from_elem(3, 1.0 / 3.0_f64.sqrt());
        let response = priced.a_derivative.dot(&positive) - positive.mapv(|x| 0.5 * x);
        assert!(
            response.dot(&response).sqrt() > 1.0e-3,
            "the projector derivative must contribute beyond the block inverse"
        );
        let h = 1.0e-5;
        for i in 0..3 {
            for j in i..3 {
                let mut plus = a.clone();
                let mut minus = a.clone();
                plus[[i, j]] += h;
                minus[[i, j]] -= h;
                if i != j {
                    plus[[j, i]] += h;
                    minus[[j, i]] -= h;
                }
                let fd = (price_fixture(&plus, &e)
                    .expect("positive A endpoint")
                    .log_det
                    - price_fixture(&minus, &e)
                        .expect("negative A endpoint")
                        .log_det)
                    / (2.0 * h);
                let analytic = priced.a_derivative[[i, j]] * if i == j { 1.0 } else { 2.0 };
                assert!(
                    (fd - analytic).abs() <= 1.0e-7,
                    "A[{i},{j}]: analytic={analytic:e}, fd={fd:e}"
                );
            }
            let mut plus = e.clone();
            let mut minus = e.clone();
            plus[i] += h;
            minus[i] -= h;
            let fd = (price_fixture(&a, &plus)
                .expect("positive E endpoint")
                .log_det
                - price_fixture(&a, &minus)
                    .expect("negative E endpoint")
                    .log_det)
                / (2.0 * h);
            assert!(
                (fd - priced.clamp_diagonal_derivative[i]).abs() <= 1.0e-7,
                "E[{i}]: analytic={}, fd={fd:e}",
                priced.clamp_diagonal_derivative[i]
            );
        }
    }

    #[test]
    fn priced_clamp_explicit_theta_derivative_has_full_logdet_scale_2820() {
        let (mut term, target, rho) =
            crate::manifold::tests_sparse_curvature_operator_2500::threshold_gate_tiny_fixture(
                true,
            );
        let (_, _, cache) = term
            .penalized_quasi_laplace_criterion_with_cache(
                target.view(),
                &rho,
                None,
                0,
                0.4,
                1.0e-6,
                1.0e-6,
            )
            .expect("fixed-state clamp fixture");
        let geometry = term
            .materialize_exact_hessian_quotient_geometry(&rho, target.view(), &cache)
            .expect("priced spectral geometry");
        let h = 1.0e-6;
        let mut live = 0;
        {
            let (block, pricing) = (&geometry.joint, &geometry.joint_pricing);
            let (_, analytic) = term
                .priced_clamp_adjoint_extras(&rho, &cache, pricing)
                .expect("explicit clamp derivative");
            // Hold A's eigensystem fixed to isolate dE. The companion K
            // contraction owns its eigenvector response, tested end to end by
            // the sparse logdet trace gate. The reference is the classifier's
            // own basin log-determinant at the moved E, so it prices exactly the
            // modes the criterion prices: log(mu) above the direction floor and
            // nothing at or inside it, where Gamma carries no weight either. It
            // prices log(mu) at full scale, so an accidental leading half in
            // Gamma cannot pass.
            assert!(
                (0..block.eigenvalues.len())
                    .any(|index| block.eigenvalues[index] < -block.rank_floor(index)),
                "fixture must price a negative direction"
            );
            let value = |moved: &super::SaeManifoldTerm| {
                let e = moved
                    .materialize_ard_concave_clamp_diagonal(&rho, &cache)
                    .expect("perturbed clamp");
                // #2828 gave E a border block, the decoder priors' majorization
                // gap, and the priced basin carries it, so this reference does.
                let e_beta = moved
                    .decoder_prior_majorizer_gap_border(&cache)
                    .expect("perturbed border gap");
                super::SaeManifoldTerm::classify_exact_hessian_basin(
                    block,
                    &e,
                    e_beta.as_ref(),
                    e.len(),
                    |direction| super::ArrowMetric::Joint(&cache).quadratic_form(direction),
                    "joint",
                    None,
                )
                .expect("reference basin")
                .log_det
            };
            for row in 0..term.n_obs() {
                for (local, variable) in term
                    .row_vars_for_cache_row(row, &cache)
                    .expect("row variables")
                    .iter()
                    .enumerate()
                {
                    // A clone drops the three frozen gates, and
                    // `barrier_coactivation_pairs` then recomputes the barrier
                    // coactivation from the moved logits, so the border gap would
                    // move with theta. Production holds the gates fixed across a step.
                    let mut plus = term.clone();
                    let mut minus = term.clone();
                    for endpoint in [&mut plus, &mut minus] {
                        endpoint.decoder_repulsion_gate = term.decoder_repulsion_gate.clone();
                        endpoint.barrier_coactivation_gate = term.barrier_coactivation_gate.clone();
                        endpoint.amplitude_barrier_gate = term.amplitude_barrier_gate;
                        endpoint.streaming_gates_frozen = true;
                    }
                    match *variable {
                        super::SaeLocalRowVar::Logit { atom } => {
                            plus.assignment.logits[[row, atom]] += h;
                            minus.assignment.logits[[row, atom]] -= h;
                        }
                        super::SaeLocalRowVar::Coord { atom, axis } => {
                            let index = row * term.assignment.coords[atom].latent_dim() + axis;
                            let mut p = plus.assignment.coords[atom].as_flat().clone();
                            let mut m = minus.assignment.coords[atom].as_flat().clone();
                            p[index] += h;
                            m[index] -= h;
                            plus.assignment.coords[atom].set_flat(p.view());
                            minus.assignment.coords[atom].set_flat(m.view());
                        }
                    }
                    let fd = (value(&plus) - value(&minus)) / (2.0 * h);
                    let expected = analytic[cache.row_offsets[row] + local];
                    assert!(
                        (fd - expected).abs() <= 1.0e-6 + 1.0e-5 * expected.abs(),
                        "row={row}, variable={variable:?}: analytic={expected:e}, fd={fd:e}"
                    );
                    live += usize::from(expected.abs() > 1.0e-3);
                }
            }
        }
        assert!(live > 0, "the explicit theta remainder must carry signal");
    }

    impl super::SaeManifoldTerm {
        /// The dense joint arrow inverse `G = H⁻¹` (`dim×dim`), materialized
        /// column by column against each unit arrow basis vector and symmetrized:
        /// the dense reference the θ-adjoint parity tests contract against.
        /// `solver` must be `DeflatedArrowSolver::plain`.
        pub(crate) fn materialize_joint_inverse(
            &self,
            cache: &ArrowFactorCache,
            solver: &DeflatedArrowSolver<'_>,
        ) -> Result<Array2<f64>, String> {
            let total_t = cache.delta_t_len();
            let k = cache.k;
            let dim = total_t + k;
            let mut g = Array2::<f64>::zeros((dim, dim));
            let mut rhs_t = Array1::<f64>::zeros(total_t);
            let rhs_beta_zero = Array1::<f64>::zeros(k);
            for col in 0..total_t {
                rhs_t[col] = 1.0;
                let sol = solver.solve(rhs_t.view(), rhs_beta_zero.view())?;
                rhs_t[col] = 0.0;
                for r in 0..total_t {
                    g[[r, col]] = sol.t[r];
                }
                for r in 0..k {
                    g[[total_t + r, col]] = sol.beta[r];
                }
            }
            let rhs_t_zero = Array1::<f64>::zeros(total_t);
            let mut rhs_beta = Array1::<f64>::zeros(k);
            for col in 0..k {
                rhs_beta[col] = 1.0;
                let sol = solver.solve(rhs_t_zero.view(), rhs_beta.view())?;
                rhs_beta[col] = 0.0;
                for r in 0..total_t {
                    g[[r, total_t + col]] = sol.t[r];
                }
                for r in 0..k {
                    g[[total_t + r, total_t + col]] = sol.beta[r];
                }
            }
            for a in 0..dim {
                for b in (a + 1)..dim {
                    let avg = 0.5 * (g[[a, b]] + g[[b, a]]);
                    g[[a, b]] = avg;
                    g[[b, a]] = avg;
                }
            }
            Ok(g)
        }

        /// #2330 Patch D arbiter support — spectrum summary of the EXACT `A` at a
        /// built cache: `(min_eig, max_eig, n_below_neg_floor, ‖ΔC‖_F, ‖A‖_F)`.
        /// The PD-window scan uses it to pick an arbiter fixture whose exact `A`
        /// is positive definite (so the criterion does not refuse) while the
        /// residual-curvature block `ΔC` — the very object Patch D
        /// differentiates — stays large enough for a finite difference to
        /// resolve. A fixture with `‖ΔC‖ ≈ 0` would false-green the arbiter.
        pub(crate) fn exact_a_spectrum_summary(
            &self,
            rho: &SaeManifoldRho,
            target: ndarray::ArrayView2<'_, f64>,
            cache: &ArrowFactorCache,
        ) -> Result<(f64, f64, usize, f64, f64), String> {
            let a = self.materialize_exact_hessian_dense(rho, target, cache)?;
            let (eigs, _vecs) = a
                .eigh(Side::Lower)
                .map_err(|e| format!("exact_a_spectrum_summary: eigh failed: {e:?}"))?;
            let max_eig = eigs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let min_eig = eigs.iter().copied().fold(f64::INFINITY, f64::min);
            // #2673 — the arbiter only needs a scale-aware "is this decisively
            // negative" cut, and the classification floor it used to borrow is
            // per-direction now. `√ε·‖A‖₂` is the coarsest band any direction of
            // this operator can have, so counting under it is an upper bound on
            // the refusing population and cannot under-report.
            let spectral_norm = eigs.iter().map(|value| value.abs()).fold(0.0_f64, f64::max);
            let floor = super::sae_exact_a_identifiability_floor() * spectral_norm;
            let n_neg = eigs.iter().filter(|&&lambda| lambda < -floor).count();
            let mut sorted: Vec<f64> = eigs.to_vec();
            sorted.sort_by(|x, y| x.partial_cmp(y).expect("finite eigenvalues"));
            let tail: Vec<String> = sorted.iter().take(6).map(|l| format!("{l:.6e}")).collect();
            eprintln!(
                "PATCHD_SPECTRUM floor={floor:.6e} smallest6=[{}]",
                tail.join(", ")
            );
            let total_t = cache.delta_t_len();
            let dim = total_t + cache.k;
            let mut dc_sq = 0.0_f64;
            let mut unit = SaeArrowVector {
                t: Array1::<f64>::zeros(total_t),
                beta: Array1::<f64>::zeros(cache.k),
            };
            for col in 0..dim {
                if col < total_t {
                    unit.t[col] = 1.0;
                } else {
                    unit.beta[col - total_t] = 1.0;
                }
                let dcv = self.apply_exact_hessian_minus_b(rho, target, cache, &unit)?;
                if col < total_t {
                    unit.t[col] = 0.0;
                } else {
                    unit.beta[col - total_t] = 0.0;
                }
                dc_sq += dcv.t.iter().map(|x| x * x).sum::<f64>()
                    + dcv.beta.iter().map(|x| x * x).sum::<f64>();
            }
            let a_frob = a.iter().map(|x| x * x).sum::<f64>().sqrt();
            Ok((min_eig, max_eig, n_neg, dc_sq.sqrt(), a_frob))
        }

        /// #2330 Patch D arbiter support — the EXACT-A joint θ-adjoint
        /// `Γ_A = tr(A⁺ ∂A/∂θ) = ∂(log|A|)/∂θ`, built from the quotient
        /// pseudo-inverse and the `exact_a = true` dh (`∂B/∂θ + ∂ΔC/∂θ`).
        /// Comparing this against a central difference of
        /// `exact_observed_information_log_dets(...).0` over frozen θ̂ measures
        /// exactly the residual-curvature/ordered-BB/entropy legs of `∂ΔC/∂θ`
        /// still missing, coordinate by coordinate.
        pub(crate) fn exact_a_theta_adjoint_joint(
            &self,
            rho: &SaeManifoldRho,
            target: ndarray::ArrayView2<'_, f64>,
            cache: &ArrowFactorCache,
        ) -> Result<SaeArrowVector, String> {
            let geometry = self.materialize_exact_hessian_quotient_geometry(rho, target, cache)?;
            let (_, clamp_gamma) =
                self.priced_clamp_adjoint_extras(rho, cache, &geometry.joint_pricing)?;
            let mut gamma = self.logdet_theta_adjoint_dense(
                rho,
                cache,
                &geometry.joint_pricing.a_derivative,
                true,
                true,
                Some(target),
            )?;
            gamma.t += &clamp_gamma;
            gamma.beta += &self.decoder_prior_gap_theta_trace(
                cache, geometry.joint_pricing.clamp_border_derivative.view(),
            )?;
            Ok(gamma)
        }
    }

    /// #2915 — channel (3b), the ThresholdGate concave remainder on logit slots,
    /// through both readers of `ΔC`.
    ///
    /// The #2509 pin below runs a softmax fixture and never reaches (3b). The
    /// applier carried (3b) from #2520 on while the assembler did not, so the arrow
    /// exact-A system priced the majorizer `B` on every switched-on logit: in job
    /// 580711 the central difference of the assembled row block in
    /// `log_lambda_sparse` read `0` on each switched-on logit, where the remainder
    /// is `−2.74e-2` or `−3.19e-2`. The straddling fixture switches a logit on in
    /// every row, so the remainder is live on every row. The positive control
    /// removes (3b) from the assembled contraction and requires the comparison to
    /// see it.
    #[test]
    fn assembled_exact_hessian_delta_carries_the_threshold_gate_remainder_2915() {
        let (mut term, target, rho) =
            crate::manifold::tests_sparse_curvature_operator_2500::threshold_gate_tiny_fixture(
                true,
            );
        let system = term
            .assemble_arrow_schur(target.view(), &rho, None)
            .expect("#2915 threshold-gate arrow assembly");
        let cache = gam_solve::arrow_schur::solve_arrow_newton_step_with_options(
            &system,
            1.0e-6,
            1.0e-6,
            &gam_solve::arrow_schur::ArrowSolveOptions::direct(),
        )
        .expect("#2915 positive Newton metric")
        .2;
        let remainder = crate::assignment::threshold_gate_negative_hessian_remainder_weighted(
            &term.assignment,
            &rho,
            term.row_loss_weights.as_deref(),
        )
        .expect("#2915 threshold-gate remainder");
        let n = term.n_obs();
        let k_atoms = term.k_atoms();
        let live_rows = (0..n)
            .filter(|&row| (0..k_atoms).any(|atom| remainder[row * k_atoms + atom] < 0.0))
            .count();
        assert_eq!(
            live_rows, n,
            "#2915 premise: the straddling gate must switch a logit on in every row"
        );
        let blocks = term
            .assemble_exact_hessian_minus_b_rows(&rho, target.view(), &cache.row_dims, cache.k)
            .expect("#2915 assembled delta rows");
        assert_eq!(blocks.len(), n);
        let row_vars: Vec<Vec<super::SaeLocalRowVar>> = (0..n)
            .map(|row| {
                term.row_vars_for_cache_row(row, &cache)
                    .expect("#2915 row variables")
            })
            .collect();
        let total_t = cache.delta_t_len();
        let mut control_separated = false;
        for probe in 0..=total_t {
            let mut v = SaeArrowVector {
                t: Array1::<f64>::zeros(total_t),
                beta: Array1::<f64>::zeros(cache.k),
            };
            if probe < total_t {
                v.t[probe] = 1.0;
            } else {
                for (idx, value) in v.t.iter_mut().enumerate() {
                    *value = 1.0 + 0.25 * (idx as f64);
                }
            }
            let applied = term
                .apply_exact_hessian_minus_b(&rho, target.view(), &cache, &v)
                .expect("#2915 matrix-free delta apply");
            // `v.beta = 0`, so the t components see only the assembled t–t blocks.
            let scale = applied
                .t
                .iter()
                .fold(1.0_f64, |acc, value| acc.max(value.abs()));
            let tolerance = 4096.0 * f64::EPSILON * scale;
            for (row, block) in blocks.iter().enumerate() {
                let base = cache.row_offsets[row];
                let q = cache.row_dims[row];
                assert_eq!(row_vars[row].len(), q);
                for a in 0..q {
                    let mut assembled = 0.0_f64;
                    for b in 0..q {
                        assembled += block.tt[[a, b]] * v.t[base + b];
                    }
                    assert!(
                        (applied.t[base + a] - assembled).abs() <= tolerance,
                        "probe {probe}: assembled ΔC t[{}] = {assembled} but the applier says {} \
                         (tolerance {tolerance:.3e})",
                        base + a,
                        applied.t[base + a]
                    );
                    if let super::SaeLocalRowVar::Logit { atom } = row_vars[row][a] {
                        let without_gate =
                            assembled - remainder[row * k_atoms + atom] * v.t[base + a];
                        control_separated |=
                            (applied.t[base + a] - without_gate).abs() > tolerance;
                    }
                }
            }
        }
        assert!(
            control_separated,
            "#2915 positive control: removing (3b) from the assembled contraction must break \
             the agreement, else this comparison cannot see the channel"
        );
    }

    /// #2509 — the assembled `ΔC = A − B` row blocks and the matrix-free applier
    /// are ONE derivation with two readers, and this is the executable link.
    ///
    /// Assembling `ΔC` necessarily writes the four channels' arithmetic a second
    /// time (the applier contracts a block against a direction; the assembler
    /// stores the block), and one quantity with two standards is a failure mode
    /// this repository keeps paying for. So the blocks are contracted here and
    /// required to reproduce `apply_exact_hessian_minus_b` to round-off.
    ///
    /// It is a real cross-check rather than a tautology: on softmax rows the
    /// applier reaches the residual-curvature channels (1a)/(1b) through the
    /// device-contracted `contracted_softmax_bilinear_hvp`, while the assembler
    /// reads the shared row-jet window. Agreement is agreement between two
    /// different execution paths over the same jets. The fixture is softmax with
    /// periodic (Circle) manifolds and non-empty `log_ard`, so channels (1a),
    /// (1b), (2) and (3) are all live — asserted below rather than assumed.
    ///
    /// The applier also adds leg (5), the decoder priors' exact-minus-majorizer
    /// border block (#2828), which no row block holds. The assembled side contracts
    /// the operator the exact-A arrow system carries for it, so the gate covers
    /// both halves of the `ΔC` that system prices.
    #[test]
    fn assembled_exact_hessian_delta_contracts_like_the_applier_2509() {
        use ndarray::Array1;
        // #2681 — the assembled/applied parity below is a statement about two
        // readers of ONE derivation, evaluated at whatever `(t, β)` and cache
        // they are handed; it has no stake in the inner solve reaching a KKT
        // point. This fixture's inner solve does not reach it (#2681), so
        // demanding convergence here only prevented the parity from ever being
        // checked. Take the pinned shared state and factor once at it through
        // the production `FROZEN_INNER_STATE` freeze lane: the criterion's own
        // refresh and freeze-lane converge, without the criterion's ½log|A|
        // pricing. The pinned state is not a KKT point and its exact A is
        // indefinite, so that pricing refuses (census job 532879 read
        // `IndefiniteObservedInformation { block: "joint" }` here). That verdict is
        // about the Laplace normaliser, not about the parity pinned below.
        let (term0, target, rho) =
            crate::manifold::tests::small_two_atom_periodic_term_at_shared_inner_state();
        let mut term = term0;
        let mut rho_fixed = rho.clone();
        let frozen_refresh = term
            .run_joint_fit_arrow_schur_for_quasi_laplace(
                target.view(),
                &mut rho_fixed,
                None,
                crate::manifold::tests::FROZEN_INNER_STATE,
                0.25,
                1.0e-4,
                1.0e-4,
            )
            .expect("freeze-lane refresh at the pinned #2509 witness state");
        let mut frozen_loss = frozen_refresh.loss;
        let mut frozen_fixed_point = frozen_refresh.fixed_point;
        let options = gam_solve::arrow_schur::ArrowSolveOptions::direct()
            .with_gpu_policy(term.gpu_policy)
            .with_newton_schur_tikhonov(gam_solve::arrow_schur::SPECTRAL_DEFLATION_REL_FLOOR)
            .with_evidence_unit_deflation(gam_solve::arrow_schur::SPECTRAL_DEFLATION_REL_FLOOR);
        let cache = term
            .converge_inner_for_undamped_logdet(
                target.view(),
                &rho,
                &mut rho_fixed,
                None,
                crate::manifold::tests::FROZEN_INNER_STATE,
                0.25,
                1.0e-4,
                1.0e-4,
                &mut frozen_loss,
                &mut frozen_fixed_point,
                &options,
                true,
            )
            .expect("freeze-lane factorization at the pinned #2509 witness state");

        let blocks = term
            .assemble_exact_hessian_minus_b_rows(&rho, target.view(), &cache.row_dims, cache.k)
            .expect("assembled delta rows");
        let border = term
            .border_channels_for_cache(&cache)
            .expect("border channels");
        let total_t = cache.delta_t_len();
        let dim = total_t + cache.k;
        assert_eq!(blocks.len(), term.n_obs());

        // NON-VACUITY: the correction must be non-zero on this fixture, or the
        // agreement below says nothing at all.
        let max_block = blocks
            .iter()
            .flat_map(|row| row.tt.iter().chain(row.tbeta.iter()))
            .fold(0.0_f64, |acc, v| acc.max(v.abs()));
        assert!(
            max_block > 0.0,
            "ΔC is identically zero on this fixture, so the gate cannot discriminate"
        );
        let border_remainder = term
            .decoder_prior_border_remainder_op(cache.k, 1.0)
            .expect("the pinned state's border is the full-B or the factored layout");

        // Deterministic probes: every (t, β) unit direction, plus one dense mix so
        // every stored entry contributes to at least one compared component.
        for probe in 0..=dim {
            let mut v = SaeArrowVector {
                t: Array1::<f64>::zeros(total_t),
                beta: Array1::<f64>::zeros(cache.k),
            };
            if probe < dim {
                if probe < total_t {
                    v.t[probe] = 1.0;
                } else {
                    v.beta[probe - total_t] = 1.0;
                }
            } else {
                for (idx, value) in v.t.iter_mut().enumerate() {
                    *value = 1.0 + 0.25 * (idx as f64);
                }
                for (idx, value) in v.beta.iter_mut().enumerate() {
                    *value = -0.5 - 0.125 * (idx as f64);
                }
            }

            let applied = term
                .apply_exact_hessian_minus_b(&rho, target.view(), &cache, &v)
                .expect("matrix-free delta apply");

            let mut assembled = SaeArrowVector {
                t: Array1::<f64>::zeros(total_t),
                beta: Array1::<f64>::zeros(cache.k),
            };
            for (row, block) in blocks.iter().enumerate() {
                let base = cache.row_offsets[row];
                let q = cache.row_dims[row];
                for a in 0..q {
                    let mut acc = 0.0_f64;
                    for b in 0..q {
                        acc += block.tt[[a, b]] * v.t[base + b];
                    }
                    for (beta_pos, channel) in border.iter().enumerate() {
                        acc += block.tbeta[[a, beta_pos]] * v.beta[channel.index];
                        assembled.beta[channel.index] += block.tbeta[[a, beta_pos]] * v.t[base + a];
                    }
                    assembled.t[base + a] += acc;
                }
            }
            // Leg (5), `ΔC_ββ`, is a border object the rows do not hold: the
            // exact-A arrow system carries it as the decoder-prior remainder
            // operator (#2828), so the assembled side contracts that operator.
            if let Some(remainder) = border_remainder.as_ref() {
                gam_solve::arrow_schur::BetaPenaltyOp::matvec(
                    remainder,
                    v.beta.as_slice().expect("an owned probe is contiguous"),
                    assembled
                        .beta
                        .as_slice_mut()
                        .expect("an owned accumulator is contiguous"),
                );
            }

            // Both sides sum the SAME products in a different association, so the
            // admissible gap is f64 round-off at the largest magnitude involved.
            let scale = applied
                .t
                .iter()
                .chain(applied.beta.iter())
                .fold(1.0_f64, |acc, v| acc.max(v.abs()));
            let tolerance = 4096.0 * f64::EPSILON * scale;
            for idx in 0..total_t {
                assert!(
                    (applied.t[idx] - assembled.t[idx]).abs() <= tolerance,
                    "probe {probe}: assembled ΔC t[{idx}] = {} but the applier says {} \
                     (tolerance {tolerance:.3e})",
                    assembled.t[idx],
                    applied.t[idx]
                );
            }
            for idx in 0..cache.k {
                assert!(
                    (applied.beta[idx] - assembled.beta[idx]).abs() <= tolerance,
                    "probe {probe}: assembled ΔC beta[{idx}] = {} but the applier says {} \
                     (tolerance {tolerance:.3e})",
                    assembled.beta[idx],
                    applied.beta[idx]
                );
            }
        }
    }
}

#[cfg(test)]
mod tests_inverse_power_deflation_cost_2627 {
    use super::*;

    /// A gapless near-null cluster must be removed without disturbing any
    /// resolved coordinate. The former inverse-power loop exhausted its work
    /// bound on this fixture; a spectral action classifies the cluster together.
    #[test]
    fn gapless_near_null_cluster_deflates_instead_of_exhausting_the_krylov_bound_2627() {
        const NEAR_NULL: f64 = 1.0e-10;
        const RESOLVED: usize = 14;
        let mut curvature: Vec<f64> = (1..=RESOLVED).map(|c| c as f64).collect();
        curvature.push(NEAR_NULL);
        curvature.push(0.9 * NEAR_NULL);
        let dim = curvature.len();

        let apply_a = |v: &SaeArrowVector| -> Result<SaeArrowVector, String> {
            let mut out = v.clone();
            for (slot, value) in out.t.iter_mut().enumerate() {
                *value *= curvature[slot];
            }
            Ok(out)
        };
        let apply_b = |v: &SaeArrowVector| -> Result<SaeArrowVector, String> { Ok(v.clone()) };
        let rhs = SaeArrowVector {
            t: Array1::from_elem(dim, 1.0),
            beta: Array1::zeros(0),
        };

        let solved =
            solve_exact_stationarity_krylov(&rhs, &apply_a, &apply_b, &apply_b)
                .expect("a gapless near-null cluster must deflate, not exhaust the Krylov bound");

        for slot in 0..RESOLVED {
            let expected = 1.0 / curvature[slot];
            assert!(
                (solved.t[slot] - expected).abs() <= 1.0e-6 * expected,
                "resolved slot {slot} was disturbed by the deflation: {:.6e} vs {expected:.6e}",
                solved.t[slot],
            );
        }
        // Undeflated, each near-null slot carries the full `1/μ` amplification
        // `1/NEAR_NULL = 1e10`. The pseudoinverse must instead return zero
        // in these directions, to the same accuracy as its resolved entries.
        for slot in RESOLVED..dim {
            assert!(
                solved.t[slot].abs() < 1.0e-6,
                "near-null slot {slot} still carries a 1/μ amplification: {:.3e}",
                solved.t[slot],
            );
        }
    }
}

#[cfg(test)]
mod tests_route_forced_classification_2673 {
    use super::super::tests::{TestPeriodicEvaluator, periodic_basis};
    use super::*;
    use gam_solve::arrow_schur::{ArrowSolveOptions, solve_arrow_newton_step_with_options};
    use ndarray::{Array1, Array2, array};
    use std::sync::Arc;

    /// #2828 item 2 — the matrix-free exact-`A` apply IS the dense one, and the
    /// two exact-stationarity solves agree on a right-hand side aimed straight
    /// at the classification band.
    ///
    /// `route_forced_stationarity_classification_agrees_2673` below reports the
    /// same comparison but does not assert it, and says why: its fixture "sits
    /// `2.8e7` bands away from any classification boundary, so it exercises the
    /// routes, not the predicate". #2828 item 2 is about a state that is IN the
    /// band, so this gate anchors on the #2330 Patch-D converged mode, whose
    /// exact `A` carries a cluster of eigenvalues at `2.7e-8` — a factor of 1.8
    /// above their own `√ε·vᵀBv` floor, i.e. inside the band by any reading —
    /// against a spectral norm of `3.0e1`. The band membership is ASSERTED, so
    /// the gate cannot quietly become the far-from-the-boundary one it replaces.
    ///
    /// What it caught: the matrix-free route reaches its majorizer through
    /// `matrix_free_arrow_operator_apply`, which applies the CONDITIONED row
    /// factor, so it was building `Φ(B_raw) + ΔC` rather than `B_raw + ΔC`. On
    /// the pinned columns the two operators differed by 1.0 in absolute terms,
    /// and on a right-hand side aligned with the smallest eigendirection the
    /// matrix-free "solution" had `‖Ax − rhs‖ = ‖rhs‖` — no residual reduction —
    /// while the `μ` deflation detector read 0.99 and accepted it, because it
    /// inspects the solution and that solution had no near-null component to
    /// detect. See [`SaeManifoldTerm::apply_exact_hessian_matrix_free`].
    #[test]
    fn matrix_free_exact_a_matches_the_dense_operator_and_solve_in_the_band_2828() {
        use crate::manifold::tests_logdet_adjoint_780::obb_patchd_fixture;
        let (mut term, target, rho) = obb_patchd_fixture(0.0, -6.0);
        term.penalized_quasi_laplace_criterion_with_cache(
            target.view(),
            &rho,
            None,
            200,
            0.4,
            1.0e-6,
            1.0e-6,
        )
        .expect("the Patch-D fixture must converge to its own mode");
        // Reassemble the undamped system at the converged state and factor it,
        // exactly as the matrix-free route's own caller does.
        let system = term
            .assemble_arrow_schur(target.view(), &rho, None)
            .expect("undamped arrow-Schur assembly at the converged mode");
        let options = ArrowSolveOptions::direct().with_positive_definite_evidence();
        let (_delta_t, _delta_beta, cache) =
            solve_arrow_newton_step_with_options(&system, 0.0, 0.0, &options)
                .expect("undamped factor cache");
        let total_t = cache.delta_t_len();
        let k = cache.k;
        let dim = total_t + k;
        let dense = term
            .materialize_exact_hessian_dense(&rho, target.view(), &cache)
            .expect("dense exact A at the converged mode");
        let (eigenvalues, eigenvectors) = dense.eigh(Side::Lower).expect("dense exact-A spectrum");
        let spectral_norm = eigenvalues
            .iter()
            .map(|value| value.abs())
            .fold(0.0_f64, f64::max);

        // The gate is only about the classification band, so prove the fixture
        // is in it: the smallest direction must sit within a decade of its own
        // floor. Far above and this is the #2673 test; below and the dense route
        // would deflate it and there would be nothing to compare.
        let smallest = eigenvectors.column(0);
        let metric = ArrowMetric::Joint(&cache)
            .quadratic_form(smallest)
            .expect("B quadratic form of the smallest direction");
        let floor = sae_exact_a_direction_floor(dim, spectral_norm, metric);
        let ratio = eigenvalues[0].abs() / floor;
        assert!(
            (1.0..10.0).contains(&ratio),
            "#2828 item 2: this gate is stated ON the classification band; the smallest \
             exact-A direction is {:.6e} against a floor of {floor:.6e} (ratio {ratio:.4}, \
             vBv={metric:.6e}, ||A||={spectral_norm:.6e})",
            eigenvalues[0]
        );

        // (1) the OPERATORS, column by column.
        let mut worst_column = 0.0_f64;
        let mut worst_at = 0usize;
        for column in 0..dim {
            let mut unit = Array1::<f64>::zeros(dim);
            unit[column] = 1.0;
            let probe = SaeArrowVector {
                t: unit.slice(s![..total_t]).to_owned(),
                beta: unit.slice(s![total_t..]).to_owned(),
            };
            let applied = term
                .apply_exact_hessian_matrix_free(&rho, target.view(), &cache, &system, &probe)
                .expect("matrix-free exact-A apply");
            for row in 0..dim {
                let value = if row < total_t {
                    applied.t[row]
                } else {
                    applied.beta[row - total_t]
                };
                let error = (value - dense[[row, column]]).abs();
                if error > worst_column {
                    worst_column = error;
                    worst_at = column;
                }
            }
        }
        assert!(
            worst_column <= 1.0e-10 * spectral_norm,
            "#2828 item 2: the matrix-free exact-A apply is not the dense operator. Worst \
             column error {worst_column:.6e} at column {worst_at} against a spectral norm of \
             {spectral_norm:.6e}. Adding `ΔC` to the CONDITIONED `Φ(B_raw)` pins every \
             spectrally deflated direction to unit curvature, so the two routes' `A⁻¹` \
             responses differ by the whole `1/λ` of that direction."
        );

        // (2) the SOLVES, on a right-hand side aimed at the band. A rhs that
        // missed the near-null directions would leave the two routes nothing to
        // classify differently — which is exactly why #2673's report is not
        // evidence about the predicate.
        let resolved = eigenvectors.column(dim - 1).to_owned();
        for (label, flat) in [
            ("null-only", smallest.to_owned()),
            ("null+resolved", &smallest.to_owned() + &resolved),
        ] {
            let rhs = SaeArrowVector {
                t: flat.slice(s![..total_t]).to_owned(),
                beta: flat.slice(s![total_t..]).to_owned(),
            };
            let dense_solution = term
                .solve_exact_stationarity(&rho, target.view(), &cache, &rhs)
                .unwrap_or_else(|error| panic!("{label}: dense stationarity solve: {error}"));
            let free_solution = term
                .solve_exact_stationarity_matrix_free(&rho, target.view(), &cache, &system, &rhs)
                .unwrap_or_else(|error| panic!("{label}: matrix-free stationarity solve: {error}"));
            let flatten = |x: &SaeArrowVector| -> Array1<f64> {
                let mut out = Array1::<f64>::zeros(dim);
                out.slice_mut(s![..total_t]).assign(&x.t);
                out.slice_mut(s![total_t..]).assign(&x.beta);
                out
            };
            let norm = |x: &Array1<f64>| x.iter().map(|v| v * v).sum::<f64>().sqrt();
            let dense_flat = flatten(&dense_solution);
            let free_flat = flatten(&free_solution);
            let rhs_norm = norm(&flat);
            // Both must actually SOLVE. The defect this gate was written for was
            // silent precisely because the matrix-free route returned a finite
            // vector that reduced no residual at all.
            for (who, solution) in [("dense", &dense_flat), ("matrix-free", &free_flat)] {
                let residual = norm(&(&dense.dot(solution) - &flat));
                assert!(
                    residual <= 1.0e-6 * rhs_norm,
                    "#2828 item 2 ({label}): the {who} route returned a vector with \
                     ||A x − rhs|| = {residual:.6e} against ||rhs|| = {rhs_norm:.6e}. A \
                     residual at the scale of the right-hand side is not a solution."
                );
            }
            let scale = norm(&dense_flat).max(norm(&free_flat));
            let difference = norm(&(&dense_flat - &free_flat));
            assert!(
                difference <= 1.0e-6 * scale,
                "#2828 item 2 ({label}): the dense and matrix-free exact-stationarity solves \
                 disagree by {difference:.6e} against a solution scale of {scale:.6e}. The \
                 IFT adjoint `a = A⁺Γ` would then depend on which route the working-set \
                 predicate picked — i.e. on ambient free memory."
            );
        }
    }

    /// #2673 — FORCE both classification routes on ONE state and compare.
    ///
    /// ## Why this test exists, and what it now guards
    ///
    /// Two floors used to classify directions of the same `A = B + ΔC`:
    /// `SAE_EXACT_A_PD_FLOOR_REL` (`1e-9·max(λ_max(A), 1)`, absolute) on the
    /// dense spectral path, and `√ε` on the pencil curvature
    /// `μ = xᵀAx/xᵀBx` in the former inverse-power solver. They
    /// are now ONE predicate in ONE metric — see
    /// [`sae_exact_a_identifiability_floor`] and
    /// [`ExactHessianSpectralBlock::rank_floor`] — so the two routes below
    /// cannot classify a direction differently by construction, and this test is
    /// the executable statement of that.
    ///
    /// ## The call chain that made it a live hazard rather than a tidiness one
    ///
    /// AN EARLIER VERSION OF THIS COMMENT CLAIMED "exactly one runs per
    /// evaluation ... they never coexist at a fixed state", and concluded the
    /// hazard was route-dependence (#2509/#2515) rather than the
    /// value↔gradient contradiction #2673 was filed about. **That claim was
    /// false, and it made a live hazard look structurally impossible.** It was
    /// true only within one GRADIENT assembly; an evaluation is value AND
    /// gradient, and on the streaming route both floors ran on the same `A`:
    ///
    /// * VALUE, when `direct_logdet_admitted == false`:
    ///   `penalized_quasi_laplace_criterion_streaming_exact_with_cache`
    ///   (`construction_quasi_laplace.rs:263`) →
    ///   `..._lane_and_system` (`:2971`) →
    ///   `converge_inner_for_undamped_logdet` (`:3039`) →
    ///   `..._gate_frozen` (`:698`) →
    ///   `terminal_exact_newton_polish` (`:1317`, `:1748`) →
    ///   `solve_exact_stationarity` (`:2346`, **with no route gate**) →
    ///   `materialize_exact_stationarity_geometry` →
    ///   `exact_hessian_spectral_block`.
    /// * GRADIENT, same evaluation: `matrix_free_system = Some(..)` →
    ///   `solve_exact_stationarity_matrix_free` →
    ///   `solve_exact_stationarity_krylov`.
    ///
    /// And there is no "massive-`K` threshold" to cross. The route predicate is
    /// a working-set comparison in `streaming_plan.rs:151` —
    /// `direct_peak_bytes <= in_core_budget_bytes || direct_fits_tiny` — so which
    /// floor classified the value path was a function of AMBIENT FREE MEMORY, not
    /// of `K`. `K` enters only through `row_block_dim`. The same data on the same
    /// build could be classified two ways on two differently-loaded machines,
    /// which is why unification did not wait for a fixture that populates the
    /// gauge band.
    ///
    /// ## What is compared
    ///
    /// The SAME state through the two PRODUCTION solves —
    /// `solve_exact_stationarity` (dense rank-revealing pseudoinverse) and
    /// `solve_exact_stationarity_matrix_free` (the Ritz pseudoinverse)
    /// — with the same rhs. If the two rules classify the same
    /// directions the same way, the solutions agree.
    ///
    /// Reported, not asserted equal. Agreement here is necessary but not
    /// sufficient: this fixture's spectrum sits `2.8e7` bands away from any
    /// classification boundary, so it exercises the routes, not the predicate.
    /// The predicate's own arms are
    /// `tests::the_two_floors_are_incommensurable_thresholds_on_one_operator_2673`
    /// (what the old pair did) and
    /// `tests::the_classification_is_invariant_under_a_reparametrization_2673`
    /// (what the new one does). What IS asserted here is that the comparison is
    /// well posed — both routes reached a typed outcome on one state, so the
    /// numbers below compare two answers rather than an answer and a refusal.
    #[test]
    fn route_forced_stationarity_classification_agrees_2673() {
        let n = 24usize;
        let coords = Array2::from_shape_fn((n, 1), |(row, _)| (row as f64 + 0.25) / n as f64);
        let (phi, jet) = periodic_basis(&coords);
        let decoder = array![[0.30, -0.10], [1.20, 0.20], [0.10, 1.10]];
        let mut target = phi.dot(&decoder);
        for row in 0..n {
            target[[row, 0]] += 1.0e-3 * (0.37 * row as f64).sin();
            target[[row, 1]] += 1.0e-3 * (0.29 * row as f64).cos();
        }
        let atom = SaeManifoldAtom::new_with_provided_function_gram(
            "periodic",
            SaeAtomBasisKind::Periodic,
            1,
            phi,
            jet,
            decoder,
            Array2::<f64>::eye(3),
        )
        .unwrap()
        .with_basis_evaluator(Arc::new(TestPeriodicEvaluator));
        let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
            Array2::<f64>::zeros((n, 1)),
            vec![coords],
            vec![LatentManifold::Circle { period: 1.0 }],
            AssignmentMode::softmax(1.0),
        )
        .unwrap();
        let mut term = SaeManifoldTerm::new(vec![atom], assignment).unwrap();
        let rho = SaeManifoldRho::new(0.0, 0.8_f64.ln(), vec![array![250.0_f64.ln()]]);
        let sys = term
            .assemble_arrow_schur(target.view(), &rho, None)
            .expect("arrow-Schur assembly");
        let options = ArrowSolveOptions::direct().with_positive_definite_evidence();
        let (_dt, _db, cache) = solve_arrow_newton_step_with_options(&sys, 0.0, 0.0, &options)
            .expect("undamped factor cache");

        // One rhs, deterministic and dense enough to excite every direction —
        // a rhs that misses the near-null directions would leave both routes
        // nothing to classify differently.
        let total_t = cache.delta_t_len();
        let rhs = SaeArrowVector {
            t: Array1::from_shape_fn(total_t, |i| 0.5 + 0.25 * ((i as f64) * 0.7).sin()),
            beta: Array1::from_shape_fn(cache.k, |i| -0.3 + 0.2 * ((i as f64) * 1.1).cos()),
        };

        let dense = term.solve_exact_stationarity(&rho, target.view(), &cache, &rhs);
        let matrix_free =
            term.solve_exact_stationarity_matrix_free(&rho, target.view(), &cache, &sys, &rhs);

        match (&dense, &matrix_free) {
            (Ok(d), Ok(m)) => {
                let dt =
                    d.t.iter()
                        .zip(m.t.iter())
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0_f64, f64::max);
                let db = d
                    .beta
                    .iter()
                    .zip(m.beta.iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0_f64, f64::max);
                let scale =
                    d.t.iter()
                        .chain(d.beta.iter())
                        .map(|v| v.abs())
                        .fold(0.0_f64, f64::max)
                        .max(1.0);
                println!(
                    "[#2673 ROUTE-FORCED] both routes solved. max|Δt|={dt:.6e} \
                     max|Δbeta|={db:.6e} scale={scale:.6e} relative={:.6e}",
                    dt.max(db) / scale
                );
            }
            (Err(d), Ok(_)) => println!(
                "[#2673 ROUTE-FORCED] DENSE refused, matrix-free solved — the routes \
                 disagree on whether this state is solvable at all: {d}"
            ),
            (Ok(_), Err(m)) => println!(
                "[#2673 ROUTE-FORCED] MATRIX-FREE refused, dense solved — the routes \
                 disagree on whether this state is solvable at all: {m}"
            ),
            (Err(d), Err(m)) => println!(
                "[#2673 ROUTE-FORCED] both routes refused.\n  dense: {d}\n  matrix-free: {m}"
            ),
        }

        // Well-posedness: both routes must reach a TYPED outcome on this state, and
        // a solution that is returned must be finite. Without this the print above
        // could be comparing an answer against a panic.
        for (label, solved) in [("dense", &dense), ("matrix-free", &matrix_free)] {
            if let Ok(x) = solved {
                assert!(
                    x.t.iter().chain(x.beta.iter()).all(|v| v.is_finite()),
                    "#2673: the {label} route returned a non-finite stationarity solution"
                );
                assert_eq!(
                    x.t.len(),
                    total_t,
                    "#2673: the {label} route must return the declared t layout"
                );
                assert_eq!(
                    x.beta.len(),
                    cache.k,
                    "#2673: the {label} route must return the declared beta layout"
                );
            }
        }
    }
}

/// The column-loop oracle for [`SaeManifoldTerm::materialize_exact_hessian_dense`]:
/// one Hessian-vector apply per column of the `dim × dim` matrix, exact for ANY
/// operator, arrow or not. Test-only: production never pays for it, and the
/// equality pin in `tests_sparse_curvature_operator_2500` measures the probe
/// assembly against it.
#[cfg(test)]
mod column_loop_oracle_tests {
    use super::*;

    impl SaeManifoldTerm {
        pub(crate) fn materialize_exact_hessian_dense_by_columns(
            &self,
            rho: &SaeManifoldRho,
            target: ArrayView2<'_, f64>,
            cache: &ArrowFactorCache,
        ) -> Result<Array2<f64>, String> {
            let total_t = cache.delta_t_len();
            let k = cache.k;
            // #2724 - ONE size expression, shared with the admission that decides
            // whether this route may run at all (streaming_plan.rs). The planner
            // evaluates the same function on a shape-derived bound for total_t, so
            // the ledger and the allocation cannot describe two different matrices.
            let dim = sae_exact_stationarity_dim(total_t, k);
            // #2267 — the same reason the Krylov sibling logs `[SAE-DEFLATE]`: this
            // route is silent for as long as it takes, and on #2267 that silence has
            // been read as a hang, as a hardware ceiling, and as an example
            // misconfiguration across five months of comments. It is none of those.
            // The route is DENSE: `dim` Hessian-vector applies to build the operator,
            // then a symmetric eigendecomposition of it — `O(dim^2)` memory and
            // `O(dim^3)` time. `dim` is the JOINT dimension, so it grows with
            // `rows x atoms`, not with the atom count: a 160-row K=1 chart is a few
            // hundred and finishes in ~1.6 s, while a 508-row K=8 dense-softmax rung
            // is ~5.3e3 and one step measured >=25 min at 4.7 GiB peak RSS. One line,
            // once per materialization, states the size of the bill before it is paid.
            log::info!(
                "[SAE-EXACT-DENSE] materializing the exact stationarity Hessian: dim={dim} \
                 (coords={total_t} + border={k}), {:.1} MiB per dim x dim f64 block, \
                 {:.1} MiB resident across {} live blocks, \
                 O(dim^3) symmetric eigendecomposition to follow",
                sae_exact_stationarity_block_bytes(dim) as f64 / (1024.0 * 1024.0),
                sae_exact_stationarity_resident_bytes(dim) as f64 / (1024.0 * 1024.0),
                SAE_EXACT_STATIONARITY_LIVE_DIM_BLOCKS,
            );
            // #2267 — the `[SAE-EXACT-DENSE]` line above states the SIZE of the bill;
            // these two stopwatches state which HALF of it is being paid. Measured at
            // `55c56d6f4`, the K=8 rung of the shipped ladder spends >=37 minutes
            // between that line and this routine's return, and nothing distinguishes
            // the O(dim) column loop from the O(dim^3) eigendecomposition that follows
            // it. Any size predicate that would refuse this route BEFORE paying has to
            // be denominated in whichever half dominates, so the split is the
            // prerequisite for the guard, not decoration.
            let build_started = std::time::Instant::now();
            // #2828 — one β-tier decoder-prior plan for all `dim` columns, so
            // this oracle and the probe assembly it checks pay the same per-apply
            // cost and their timings stay comparable.
            let prepared = self.prepare_decoder_prior_beta_curvature(1.0);
            let residual = self.prepare_residual_curvature_rows(target, cache)?;
            let mut a = Array2::<f64>::zeros((dim, dim));
            let mut unit = SaeArrowVector {
                t: Array1::<f64>::zeros(total_t),
                beta: Array1::<f64>::zeros(k),
            };
            for col in 0..dim {
                if col < total_t {
                    unit.t[col] = 1.0;
                } else {
                    unit.beta[col - total_t] = 1.0;
                }
                let av = self.apply_exact_hessian_prepared(
                    rho, cache, &unit, &prepared, &residual,
                )?;
                if col < total_t {
                    unit.t[col] = 0.0;
                } else {
                    unit.beta[col - total_t] = 0.0;
                }
                for r in 0..total_t {
                    a[[r, col]] = av.t[r];
                }
                for r in 0..k {
                    a[[total_t + r, col]] = av.beta[r];
                }
            }
            // The matrix-free apply is symmetric only up to round-off; symmetrize
            // so downstream Cholesky / selected-inverse factors see an exactly
            // symmetric operand.
            for r in 0..dim {
                for c in (r + 1)..dim {
                    let avg = 0.5 * (a[[r, c]] + a[[c, r]]);
                    a[[r, c]] = avg;
                    a[[c, r]] = avg;
                }
            }
            let build_elapsed = build_started.elapsed();
            log::info!(
                "[SAE-EXACT-DENSE] operator BUILT: dim={dim}, {dim} Hessian-vector applies \
                 + symmetrization in {:.3} s ({:.3} ms per apply); \
                 the O(dim^3) symmetric eigendecomposition has NOT started yet",
                build_elapsed.as_secs_f64(),
                build_elapsed.as_secs_f64() * 1.0e3 / (dim.max(1) as f64),
            );
            Ok(a)
        }
    }
}

// The un-prepared wrappers below build a decoder-prior plan per call; production
// callers all go through the `_prepared` forms with a plan built once per state
// (#2828), so the wrappers are test-only conveniences and live here to keep the
// workspace `warnings = "deny"` gate green (dead_code otherwise).
#[cfg(test)]
mod tests_exact_hessian_apply_wrappers {
    use super::*;

    impl SaeManifoldTerm {
        pub(crate) fn apply_exact_hessian(
            &self,
            rho: &SaeManifoldRho,
            target: ArrayView2<'_, f64>,
            cache: &ArrowFactorCache,
            v: &SaeArrowVector,
        ) -> Result<SaeArrowVector, String> {
            let prepared = self.prepare_decoder_prior_beta_curvature(1.0);
            let residual = self.prepare_residual_curvature_rows(target, cache)?;
            self.apply_exact_hessian_prepared(rho, cache, v, &prepared, &residual)
        }
        pub(crate) fn apply_exact_hessian_minus_b(
            &self,
            rho: &SaeManifoldRho,
            target: ArrayView2<'_, f64>,
            cache: &ArrowFactorCache,
            v: &SaeArrowVector,
        ) -> Result<SaeArrowVector, String> {
            let prepared = self.prepare_decoder_prior_beta_curvature(1.0);
            let residual = self.prepare_residual_curvature_rows(target, cache)?;
            self.apply_exact_hessian_minus_b_prepared(rho, cache, v, &prepared, &residual)
        }
        pub(crate) fn apply_exact_hessian_matrix_free(
            &self,
            rho: &SaeManifoldRho,
            target: ArrayView2<'_, f64>,
            cache: &ArrowFactorCache,
            system: &ArrowSchurSystem,
            vector: &SaeArrowVector,
        ) -> Result<SaeArrowVector, String> {
            let prepared = self.prepare_decoder_prior_beta_curvature(1.0);
            let residual = self.prepare_residual_curvature_rows(target, cache)?;
            self.apply_exact_hessian_matrix_free_prepared(
                rho, cache, system, vector, &prepared, &residual,
            )
        }
    }
}

// The residual-currency damped path the terminal polish globalized on before
// #2080 moved acceptance to the penalized objective. It stays as a TEST ORACLE:
// the #2762 pins hold it to the pseudoinverse at ν = 0 and to the exact
// closed-form model residual `g + AΔ(ν) = Σ_i u_i c_i ν/(λ_i² + ν)`, and the
// #2080 pin uses it to show the objective step descends where this one ascends.
#[cfg(test)]
mod tests_damped_residual_path {
    use super::*;

    /// One point of the damped residual path.
    pub(crate) struct DampedResidualPathPoint {
        /// `Δ(ν)`.
        pub(crate) step: SaeArrowVector,
        /// `g + AΔ(ν)`, in the same arrow layout as the residual handed in.
        pub(crate) model_residual: SaeArrowVector,
        /// `‖Δ(ν)‖²`.
        pub(crate) step_norm_sq: f64,
        /// Directions whose damped denominator cleared the null band.
        pub(crate) retained_rank: usize,
    }

    impl ExactHessianSpectralBlock {
        /// `residual` is the stationarity residual `g`; the step returned solves
        /// the damped system against `−g`, and the modeled residual is reported
        /// for `g` itself. A direction whose damped denominator `λ² + ν` is inside
        /// the null band (`≤ rank_floor²`) contributes nothing to the step and its
        /// whole coefficient to the model residual: at `ν = 0` that is exactly the
        /// pseudoinverse's own classification.
        pub(crate) fn damped_residual_step(
            &self,
            residual: &SaeArrowVector,
            nu: f64,
        ) -> Result<DampedResidualPathPoint, String> {
            let total_t = residual.t.len();
            let dim = total_t + residual.beta.len();
            let spectral_dim = self.eigenvalues.len();
            if self.eigenvectors.dim() != (dim, spectral_dim) || spectral_dim != dim {
                return Err(format!(
                    "damped residual step: eigenvectors {:?} and spectrum {spectral_dim} do not \
                     match residual dimension {dim}",
                    self.eigenvectors.dim(),
                ));
            }
            if !(nu.is_finite() && nu >= 0.0) {
                return Err(format!(
                    "damped residual step: damping must be finite and ≥ 0; got {nu}"
                ));
            }
            let mut flat = Array1::<f64>::zeros(dim);
            flat.slice_mut(s![..total_t]).assign(&residual.t);
            flat.slice_mut(s![total_t..]).assign(&residual.beta);
            if !flat.iter().all(|value| value.is_finite()) {
                return Err("damped residual step: residual contains a non-finite value".to_string());
            }
            let coefficients = self.eigenvectors.t().dot(&flat);
            let mut step_coefficients = Array1::<f64>::zeros(spectral_dim);
            let mut model_coefficients = Array1::<f64>::zeros(spectral_dim);
            let mut retained_rank = 0usize;
            for index in 0..spectral_dim {
                let lambda = self.eigenvalues[index];
                let floor = self.rank_floor(index);
                let null_band = floor * floor;
                let denominator = lambda * lambda + nu;
                let coefficient = coefficients[index];
                let surviving = if denominator > null_band {
                    // Δ solves `(A² + ν) Δ = −A g` in this direction.
                    step_coefficients[index] = -lambda * coefficient / denominator;
                    retained_rank += 1;
                    coefficient * nu / denominator
                } else {
                    coefficient
                };
                model_coefficients[index] = surviving;
            }
            let solution = self.eigenvectors.dot(&step_coefficients);
            let model = self.eigenvectors.dot(&model_coefficients);
            Ok(DampedResidualPathPoint {
                step: SaeArrowVector {
                    t: solution.slice(s![..total_t]).to_owned(),
                    beta: solution.slice(s![total_t..]).to_owned(),
                },
                model_residual: SaeArrowVector {
                    t: model.slice(s![..total_t]).to_owned(),
                    beta: model.slice(s![total_t..]).to_owned(),
                },
                step_norm_sq: solution.dot(&solution),
                retained_rank,
            })
        }
    }
}

#[cfg(test)]
#[path = "tests_exact_a_probes_2828.rs"]
mod tests_exact_a_probes_2828;

#[cfg(test)]
#[path = "tests_clamp_basin_deflation_2333.rs"]
mod tests_clamp_basin_deflation_2333;

#[cfg(test)]
#[path = "tests_residual_curvature_rows_2731.rs"]
mod tests_residual_curvature_rows_2731;

/// #2731 — test-side names for the two halves production reads together from
/// `SaeManifoldTerm::materialize_exact_hessian_dense_with_gap_border`.
#[cfg(test)]
mod tests_dense_exact_a_names_2731 {
    use super::*;

    impl SaeManifoldTerm {
        /// The dense exact `A` alone.
        pub(crate) fn materialize_exact_hessian_dense(
            &self,
            rho: &SaeManifoldRho,
            target: ArrayView2<'_, f64>,
            cache: &ArrowFactorCache,
        ) -> Result<Array2<f64>, String> {
            self.materialize_exact_hessian_dense_with_gap_border(rho, target, cache)
                .map(|(a, _)| a)
        }

        /// `E_ββ` alone, from its own `k` applies of leg (5): an independent arm
        /// against the columns the dense build keeps.
        pub(crate) fn decoder_prior_majorizer_gap_border(
            &self,
            cache: &ArrowFactorCache,
        ) -> Result<Option<Array2<f64>>, String> {
            let k = cache.k;
            if k == 0 {
                return Ok(None);
            }
            let prepared = self.prepare_decoder_prior_beta_curvature(1.0);
            let projection = crate::frames::FrameProjection::new(self);
            let mut gap = Array2::<f64>::zeros((k, k));
            let mut unit = Array1::<f64>::zeros(k);
            for col in 0..k {
                unit.fill(0.0);
                unit[col] = 1.0;
                let column =
                    self.decoder_prior_gap_border_leg(cache, &prepared, &projection, unit.view())?;
                for row in 0..k {
                    gap[[row, col]] = -column[row];
                }
            }
            if gap.iter().all(|&value| value == 0.0) {
                return Ok(None);
            }
            for row in 0..k {
                for col in (row + 1)..k {
                    let average = 0.5 * (gap[[row, col]] + gap[[col, row]]);
                    gap[[row, col]] = average;
                    gap[[col, row]] = average;
                }
            }
            Ok(Some(gap))
        }
    }
}

/// #2267 — the priced `log|A|` alone, for the tests that read the value or the
/// refusal without the refused directions production descends.
#[cfg(test)]
mod tests_exact_observed_information_names_2267 {
    use super::*;

    impl SaeManifoldTerm {
        pub(crate) fn exact_observed_information_log_dets(
            &self,
            rho: &SaeManifoldRho,
            target: ArrayView2<'_, f64>,
            cache: &ArrowFactorCache,
        ) -> Result<f64, SaeCriterionError> {
            let mut saddle_directions = Vec::new();
            self.exact_observed_information_log_dets_with_saddle_directions(
                rho,
                target,
                cache,
                &mut saddle_directions,
            )
        }
    }
}
