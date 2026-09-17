// #2228/#2822 — the evidence root refinement: carry an admitted inner state to the root of its
// stationarity residual (`refine_evidence_root`, `refine_accepted_root`), the exact step it takes
// (`evidence_root_step`), and the removal of the evidence factor's unit-stiffened row directions from
// that step. Included from `construction_quasi_laplace.rs`.

impl SaeManifoldTerm {
    /// #2228 — carry a state the KKT band admitted to the numerical root of its own
    /// stationarity residual before the criterion prices it.
    ///
    /// The band admits any state with `‖g‖ ≤ tol`, so it admits a state error `A⁺g`
    /// as large as `‖A⁺‖·tol`. The loss is second order in that error, but `½log|A|`
    /// is first order in it, so where the band stopped a trajectory moved the value.
    /// Pool job 642871 at `55e561270` read
    /// `value_probe_refine_policy_ranks_same_criterion_as_full_policy`'s two lanes
    /// accept at `‖g‖` 3.8e-6 and 5.0e-6 (tol 3.0e-5, `‖Δ‖ ≈ 2e-6`) with losses
    /// 3.2e-11 apart and `½log|A|` 1.4e-5 apart.
    ///
    /// Inside the band the objective cannot verify a step: the predicted decrease
    /// `½gᵀA⁺g` is under the Armijo round-off floor where the polish's ladder ends.
    /// The residual still resolves the root. Each step is the exact Newton step on
    /// the route the polish takes (the dense geometry's pseudoinverse, on the
    /// operator and null band the value prices, for ordered Beta–Bernoulli; the arrow
    /// exact-A solve with no ridge escalation otherwise). A step commits only if it
    /// strictly contracts the gate norm and does not raise the penalized objective
    /// past its round-off cushion. On a nonsingular `A` the contraction is quadratic
    /// and the phase ends at the first step round-off cannot contract, so no step
    /// count is chosen. Returns whether the state moved.
    ///
    /// #2822 — the residual the step solves, the step, and the gate that judges it all
    /// live on the complement of the directions the evidence factor unit-stiffened
    /// (`ArrowFactorCache::deflated_row_directions`). Along such a direction the factor,
    /// and with it `A`, carries the substituted stiffness `κ = 1` where the objective's own
    /// curvature sits under the gauge qualification bar, so `A⁺g` moves the state along it
    /// with unit gain while the residual barely responds. The inner step already fixes these
    /// directions by projection, not stiffness (`row_sub_floor_null_directions`). This phase
    /// did not, and its gate kept their residual: job 1179477 at `b7525ec4bd` (the small-N
    /// circle under the production seed) read crawl steps whose Newton step was all
    /// coordinate, along pencil directions at `μ = 1.000`, with the linear model predicting
    /// `‖g + AΔ‖ ≈ 1e-20` while the trial left `‖r‖ = 4.057776e-9` from `4.057830e-9`
    /// (`‖JΔ‖/‖AΔ‖ ≈ 1.46e-5`), so every step contracted strictly and the phase committed
    /// thousands of them.
    fn refine_evidence_root(
        &mut self,
        target: ArrayView2<'_, f64>,
        rho_fixed: &SaeManifoldRho,
        registry: Option<&AnalyticPenaltyRegistry>,
        lambda_smooth: &[f64],
        options: &ArrowSolveOptions,
    ) -> Result<bool, String> {
        let mut moved = false;
        loop {
            let Some(EvidenceRootStep { gate, step, cache }) =
                self.evidence_root_step(target, rho_fixed, registry, lambda_smooth, options)?
            else {
                return Ok(moved);
            };
            let pre_objective = self.penalized_objective_total(target, rho_fixed, registry, 1.0)?;
            let snapshot = self.snapshot_mutable_state();
            let trial = match self.apply_newton_step(step.t.view(), step.beta.view(), 1.0) {
                Ok(()) => match self.assemble_arrow_schur(target, rho_fixed, registry) {
                    // The trial residual is judged on the same complement as the step: the
                    // directions this state's factor stiffened, not the trial state's.
                    Ok(mut trial_sys) => match Self::remove_unit_stiffened_directions_from_rows(
                        &mut trial_sys,
                        &cache.deflated_row_directions,
                    ) {
                        Ok(()) => {
                            let trial_sq = Self::system_grad_norm_sq(&trial_sys);
                            let trial_gate = self.quotient_gradient_norm_from_system(
                                &trial_sys,
                                trial_sq,
                                lambda_smooth,
                            );
                            let trial_objective = self
                                .penalized_objective_total(target, rho_fixed, registry, 1.0)
                                .unwrap_or(f64::INFINITY);
                            Some((trial_gate, trial_objective))
                        }
                        Err(err) => {
                            log::debug!("[SAE-ROOT] root step trial residual: {err}");
                            None
                        }
                    },
                    Err(err) => {
                        log::debug!("[SAE-ROOT] root step trial assembly: {err}");
                        None
                    }
                },
                Err(err) => {
                    log::debug!("[SAE-ROOT] root step application: {err}");
                    None
                }
            };
            match trial {
                Some((trial_gate, trial_objective))
                    if trial_gate < gate
                        && trial_objective.is_finite()
                        && trial_objective
                            <= pre_objective + opt::armijo_roundoff_cushion(pre_objective) =>
                {
                    log::info!(
                        "[SAE-ROOT] committed: gate ‖g‖ {gate:.6e} → {trial_gate:.6e}, penalized \
                         objective {pre_objective:.16e} → {trial_objective:.16e}"
                    );
                    moved = true;
                }
                other => {
                    self.restore_mutable_state(&snapshot)?;
                    log::info!(
                        "[SAE-ROOT] root at gate ‖g‖ {gate:.6e}: the exact Newton trial left \
                         (gate, objective) = {other:?} against objective {pre_objective:.16e}"
                    );
                    return Ok(moved);
                }
            }
        }
    }

    /// #2228/#2822 — one exact root step from the installed state, for
    /// [`Self::refine_evidence_root`]: assemble, take the deflated evidence factor, remove the
    /// directions it unit-stiffened from the residual, read the gate, solve on the route the
    /// polish takes, and remove the same directions from the step. `None` is where the phase
    /// ends without a step: a factor or route that yields no Newton step, a gate that is not a
    /// finite positive number, or a non-finite step. The state is not moved.
    fn evidence_root_step(
        &mut self,
        target: ArrayView2<'_, f64>,
        rho_fixed: &SaeManifoldRho,
        registry: Option<&AnalyticPenaltyRegistry>,
        lambda_smooth: &[f64],
        options: &ArrowSolveOptions,
    ) -> Result<Option<EvidenceRootStep>, String> {
        let mut sys = self
            .assemble_arrow_schur(target, rho_fixed, registry)
            .map_err(|err| format!("SaeManifoldTerm::refine_evidence_root: {err}"))?;
        let factor =
            match self.factor_deflated_evidence_with_grad_norms(&mut sys, lambda_smooth, options) {
                Ok(factor) => factor,
                Err(err) => {
                    log::debug!("[SAE-ROOT] no root step: deflated evidence factor: {err}");
                    return Ok(None);
                }
            };
        let cache = factor.cache;
        Self::remove_unit_stiffened_directions_from_rows(&mut sys, &cache.deflated_row_directions)?;
        let grad_norm_sq = Self::system_grad_norm_sq(&sys);
        let gate = self.quotient_gradient_norm_from_system(&sys, grad_norm_sq, lambda_smooth);
        if !(gate.is_finite() && gate > 0.0) {
            return Ok(None);
        }
        let exact_dim = sae_exact_stationarity_dim(cache.delta_t_len(), cache.k);
        let dense_geometry_route = matches!(
            self.assignment.mode,
            AssignmentMode::OrderedBetaBernoulli { .. }
        ) && sae_exact_stationarity_admitted(exact_dim, self.host_available_bytes);
        let mut step = if dense_geometry_route {
            let mut residual_t = Array1::<f64>::zeros(cache.delta_t_len());
            let mut offset = 0usize;
            for row in &sys.rows {
                for (axis, &g) in row.gt.iter().enumerate() {
                    residual_t[offset + axis] = g;
                }
                offset += row.gt.len();
            }
            let residual = SaeArrowVector {
                t: residual_t,
                beta: sys.gb.clone(),
            };
            let solve = self
                .materialize_exact_stationarity_geometry(rho_fixed, target, &cache)
                .and_then(|geometry| geometry.solve_stationarity(&residual));
            match self.evidence_root_step_from_pencil(solve) {
                Some(step) => step,
                None => return Ok(None),
            }
        } else {
            let exact = match self.exact_a_evidence_system(target, rho_fixed, &sys, 1.0) {
                Ok(exact) => exact,
                Err(err) => {
                    log::debug!("[SAE-ROOT] no root step: arrow exact-A system: {err}");
                    return Ok(None);
                }
            };
            let mut exact_options = options.clone();
            exact_options.sae_resident_frame = None;
            match gam_solve::arrow_schur::solve_with_lm_escalation_inner(
                &exact,
                0.0,
                0.0,
                &exact_options,
            ) {
                Ok((delta_t, delta_beta, diagnostics)) if diagnostics.ridge_escalations == 0 => {
                    SaeArrowVector {
                        t: delta_t,
                        beta: delta_beta,
                    }
                }
                Ok((_, _, diagnostics)) => {
                    log::debug!(
                        "[SAE-ROOT] no root step: the arrow exact-A solve escalated its ridge {} \
                         time(s), so its step is not the Newton step",
                        diagnostics.ridge_escalations
                    );
                    return Ok(None);
                }
                Err(err) => {
                    log::debug!("[SAE-ROOT] no root step: arrow exact-A solve: {err}");
                    return Ok(None);
                }
            }
        };
        if !(step.t.iter().all(|v| v.is_finite()) && step.beta.iter().all(|v| v.is_finite())) {
            log::debug!("[SAE-ROOT] no root step: the exact Newton step is not finite");
            return Ok(None);
        }
        Self::remove_unit_stiffened_directions_from_flat(
            &mut step.t,
            &cache.row_offsets,
            &cache.deflated_row_directions,
        )?;
        Ok(Some(EvidenceRootStep { gate, step, cache }))
    }

    /// #2822 — remove from every row's coordinate gradient each direction the evidence
    /// factor unit-stiffened in that row ([`Self::refine_evidence_root`]). An empty ledger,
    /// from a factorization that stiffened nothing, removes nothing.
    fn remove_unit_stiffened_directions_from_rows(
        sys: &mut ArrowSchurSystem,
        directions: &[Vec<Array1<f64>>],
    ) -> Result<(), String> {
        if directions.is_empty() {
            return Ok(());
        }
        if directions.len() != sys.rows.len() {
            return Err(format!(
                "SaeManifoldTerm::refine_evidence_root: the evidence factor recorded unit-stiffened \
                 directions for {} rows, but the system has {} rows",
                directions.len(),
                sys.rows.len()
            ));
        }
        for (row_index, (row, row_directions)) in
            sys.rows.iter_mut().zip(directions.iter()).enumerate()
        {
            let Some(values) = row.gt.as_slice_mut() else {
                return Err(format!(
                    "SaeManifoldTerm::refine_evidence_root: row {row_index}'s coordinate gradient \
                     is not contiguous"
                ));
            };
            Self::remove_unit_stiffened_directions(values, row_directions, row_index)?;
        }
        Ok(())
    }

    /// #2822 — the same removal on a flat coordinate vector laid out by `row_offsets`.
    fn remove_unit_stiffened_directions_from_flat(
        values: &mut Array1<f64>,
        row_offsets: &[usize],
        directions: &[Vec<Array1<f64>>],
    ) -> Result<(), String> {
        if directions.is_empty() {
            return Ok(());
        }
        if row_offsets.len() != directions.len() + 1 || row_offsets[directions.len()] != values.len()
        {
            return Err(format!(
                "SaeManifoldTerm::refine_evidence_root: {} rows of unit-stiffened directions do not \
                 lay out a coordinate step of length {} ({} row offsets)",
                directions.len(),
                values.len(),
                row_offsets.len()
            ));
        }
        let Some(flat) = values.as_slice_mut() else {
            return Err(
                "SaeManifoldTerm::refine_evidence_root: the root step's coordinate block is not \
                 contiguous"
                    .to_string(),
            );
        };
        for (row_index, row_directions) in directions.iter().enumerate() {
            Self::remove_unit_stiffened_directions(
                &mut flat[row_offsets[row_index]..row_offsets[row_index + 1]],
                row_directions,
                row_index,
            )?;
        }
        Ok(())
    }

    /// `x ← x − Σᵥ v (vᵀx)` over one row's unit-stiffened directions, which the factor
    /// returns orthonormal, so the sweep is the orthogonal projection onto their complement.
    fn remove_unit_stiffened_directions(
        values: &mut [f64],
        directions: &[Array1<f64>],
        row_index: usize,
    ) -> Result<(), String> {
        for direction in directions {
            if direction.len() != values.len() {
                return Err(format!(
                    "SaeManifoldTerm::refine_evidence_root: row {row_index} carries a unit-stiffened \
                     direction of length {} against a {}-dimensional coordinate block",
                    direction.len(),
                    values.len()
                ));
            }
            let coefficient = direction
                .iter()
                .zip(values.iter())
                .map(|(component, value)| component * value)
                .sum::<f64>();
            for (value, component) in values.iter_mut().zip(direction.iter()) {
                *value -= coefficient * component;
            }
        }
        Ok(())
    }

    /// #2228 — the state an acceptance certificate admitted, carried to its root before
    /// the criterion prices it ([`Self::refine_evidence_root`]). A moved state takes one
    /// evidence re-entry that must recur exactly, and the undamped deflated evidence
    /// factor is then taken at the refined root, assembled under the caller's own
    /// assembly ρ (`None` for `rho_fixed`). `None` means nothing moved, or the accepted
    /// state and its loss have been restored, so the caller returns the factor it
    /// already holds: a refinement never turns an acceptance into a refusal.
    fn refine_accepted_root(
        &mut self,
        target: ArrayView2<'_, f64>,
        assembly_rho: Option<&SaeManifoldRho>,
        rho_fixed: &mut SaeManifoldRho,
        registry: Option<&AnalyticPenaltyRegistry>,
        lambda_smooth: &[f64],
        options: &ArrowSolveOptions,
        inner_max_iter: usize,
        learning_rate: f64,
        ridge_ext_coord: f64,
        ridge_beta: f64,
        loss: &mut SaeManifoldLoss,
        criterion_fixed_point: &mut bool,
        total_inner_iter: &mut usize,
    ) -> Result<Option<DeflatedEvidenceFactor>, String> {
        let accepted_state = self.snapshot_mutable_state();
        let accepted_loss = *loss;
        if !self.refine_evidence_root(target, rho_fixed, registry, lambda_smooth, options)? {
            return Ok(None);
        }
        let refine_iter = inner_max_iter.max(1);
        let refine = self.run_joint_fit_arrow_schur_for_quasi_laplace(
            target,
            rho_fixed,
            registry,
            refine_iter,
            learning_rate,
            ridge_ext_coord,
            ridge_beta,
        )?;
        *total_inner_iter += refine_iter;
        if refine.fixed_point {
            let assembled = match assembly_rho {
                Some(rho) => self.assemble_arrow_schur(target, rho, registry),
                None => self.assemble_arrow_schur(target, rho_fixed, registry),
            };
            let mut system =
                assembled.map_err(|err| format!("SaeManifoldTerm::refine_accepted_root: {err}"))?;
            match self.factor_deflated_evidence_with_grad_norms(&mut system, lambda_smooth, options)
            {
                Ok(factor) => {
                    log::info!(
                        "[SAE-ROOT] accepted at the refined root: ‖g‖={:.6e} ‖Π⊥null g‖={:.6e} \
                         after {} inner iterations",
                        factor.grad_norm,
                        factor.quotient_grad_norm,
                        *total_inner_iter,
                    );
                    *loss = refine.loss;
                    *criterion_fixed_point = true;
                    return Ok(Some(factor));
                }
                Err(err) => log::info!(
                    "[SAE-ROOT] the deflated evidence factor at the refined root failed ({err}); \
                     pricing the accepted state"
                ),
            }
        } else {
            log::info!(
                "[SAE-ROOT] the evidence re-entry at the refined state did not recur; pricing \
                 the accepted state"
            );
        }
        self.restore_mutable_state(&accepted_state)?;
        *loss = accepted_loss;
        Ok(None)
    }
}

/// #2228/#2822 — one exact root step ([`SaeManifoldTerm::refine_evidence_root`]): the gate the
/// installed state reads, the step, and the deflated evidence factor whose unit-stiffened
/// directions both were removed from.
struct EvidenceRootStep {
    gate: f64,
    step: SaeArrowVector,
    cache: ArrowFactorCache,
}

#[cfg(test)]
mod evidence_root_gauge_projection_2822_tests {
    use super::*;
    use crate::manifold::{
        SaeFitAssignmentKind, SaeFitConfig, SaeFitSeedReport, SaeFitSeedRequest,
        SaeMinimalSeedRequest, build_sae_fit_seed, build_sae_minimal_seed,
    };

    const N: usize = 180;
    const P: usize = 24;

    fn idx_uniform(seed: u64) -> f64 {
        let mut state = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64) * f64::from_bits(0x3CA0000000000000)
    }

    fn idx_normal(seed: u64) -> f64 {
        let u1 = idx_uniform(seed).max(1.0e-12);
        let u2 = idx_uniform(seed.wrapping_add(0x9E3779B97F4A7C15));
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }

    /// `sae_manifold_small_n_circle_accepts_a_seed_and_fits`'s bank: a circle planted in a
    /// 2-plane of `R^24` with a second harmonic and small ambient noise, N = 180.
    fn planted_small_circle() -> Array2<f64> {
        let mut z = Array2::<f64>::zeros((N, P));
        for i in 0..N {
            let theta = std::f64::consts::TAU * ((i as f64) * 0.061_803 + 0.13).rem_euclid(1.0);
            z[[i, 0]] = theta.cos();
            z[[i, 1]] = theta.sin();
            z[[i, 2]] = 0.4 * (2.0 * theta).cos();
            for col in 0..P {
                z[[i, col]] += 0.03 * idx_normal((i as u64) * 31 + col as u64);
            }
        }
        z
    }

    /// The small-N circle's production seed, fitted at `θ = log λ_smooth = 15`: a decoder
    /// smoothed nearly flat, so every row's phase curvature (≈ 7e-9, job 1186336) sits under
    /// `factor_gauge_deflated_evidence_row`'s qualification bar and all 180 phase gauges are
    /// unit-pinned with no raw spectrum. Those pins stay in `A`, which is the configuration
    /// the crawl read. At `θ = 14` the rows are pinned spectrally instead, and `A` restores
    /// their raw curvature. The joint fit is the evidence policy's own inner walk, so no root
    /// phase runs before the step under test.
    fn pinned_circle_state() -> (SaeManifoldTerm, Array2<f64>, SaeManifoldRho) {
        let z = planted_small_circle();
        let assignment_kind = SaeFitAssignmentKind::OrderedBetaBernoulli;
        let minimal = build_sae_minimal_seed(SaeMinimalSeedRequest {
            target: z.view(),
            atom_basis: vec!["periodic".to_string()],
            atom_dim: vec![1],
            assignment_kind,
            alpha: 1.0,
            tau: 0.5,
            threshold: 0.0,
            top_k: None,
            random_state: 0,
            initial_logits: None,
            initial_coords: None,
        })
        .expect("the production minimal seed builds on the small-N bank");
        let registry = AnalyticPenaltyRegistry::new();
        let SaeFitSeedReport {
            base_term,
            initial_rho,
            ..
        } = build_sae_fit_seed(SaeFitSeedRequest {
            target: z.view(),
            geometry_plans: &minimal.geometry_plans,
            basis_values: minimal.basis_values.view(),
            basis_jacobian: minimal.basis_jacobian.view(),
            decoder_coefficients: minimal.decoder_coefficients.view(),
            smooth_penalties: minimal.smooth_penalties.view(),
            initial_logits: minimal.initial_logits.view(),
            initial_coords: minimal.initial_coords.view(),
            alpha: 1.0,
            tau: 0.5,
            learnable_alpha: false,
            assignment_kind,
            sparsity_strength: 1.0,
            smoothness: 1.0,
            max_iter: 50,
            learning_rate: 1.0,
            ridge_ext_coord: 1.0e-6,
            ridge_beta: 1.0e-6,
            top_k: None,
            threshold: 0.0,
            native_ard_enabled: false,
            seed_refine_routing: minimal.refine_routing,
            seed_refine_random_state: 0,
            fit_config: SaeFitConfig::default(),
            temperature_schedule: None,
            fisher_metric: None,
            row_loss_weights: None,
            registry: &registry,
        })
        .expect("the production fit seed builds on the small-N bank");
        let mut term = base_term;
        let flat = initial_rho
            .to_flat(&term.assignment)
            .expect("the seed rho is bound to the term's assignment");
        assert_eq!(flat.len(), 1, "the small-N circle's outer layout is the one smoothing coordinate");
        let mut rho = initial_rho
            .from_flat(Array1::from_vec(vec![15.0]).view())
            .expect("the pinned coordinate rebuilds a rho on the seed's layout");
        let fitted = term
            .run_joint_fit_arrow_schur_for_quasi_laplace(z.view(), &mut rho, None, 200, 1.0, 1.0e-6, 1.0e-6)
            .expect("the evidence joint fit runs at the pinned coordinate");
        assert!(
            fitted.loss.total().is_finite(),
            "the evidence joint fit must reach a finite penalized objective at the pinned coordinate"
        );
        (term, z, rho)
    }

    /// The root step moves nothing along a direction the evidence factor unit-stiffened
    /// (#2822). Along such a direction the factor carries the substituted `κ = 1`, so an
    /// unprojected `A⁺g` moves it with unit gain while the objective barely responds, and the
    /// root phase committed thousands of strict contractions of a residual it could not remove.
    ///
    /// Bar, derived from the removal's arithmetic. A row of dimension `d` carries `m`
    /// orthonormal directions. Removing `v` computes `vᵀx` with rounding at most `d·ε·‖x‖`
    /// and subtracts it along a `v` whose `vᵀv` is one to `d·ε`, leaving at most `2·d·ε·‖x‖`
    /// along `v`. Each of the other `m − 1` removals, along a direction orthogonal to `v` to
    /// `d·ε`, reintroduces at most `d·ε·‖x‖`. Reading `vᵀΔ` here rounds by another `d·ε·‖Δ‖`.
    /// So `|vᵀΔ_row| ≤ (m + 2)·d·ε·‖Δ_row‖`.
    ///
    /// Premise: the factor's own discarded Newton step, which is not projected, moves along
    /// some unit-stiffened direction by more than that bar, so an unprojected root step would
    /// too and the assertion is not vacuous.
    #[test]
    fn root_step_leaves_every_unit_stiffened_row_direction_unmoved_2822() {
        let (mut term, z, rho) = pinned_circle_state();
        let options = term.evidence_factor_options();
        let lambda_smooth = rho
            .lambda_smooth_vec()
            .expect("the crawl rho carries its smoothing coordinates");
        let along = |values: &[f64], direction: &Array1<f64>| {
            direction
                .iter()
                .zip(values.iter())
                .map(|(component, value)| component * value)
                .sum::<f64>()
        };
        let norm = |values: &[f64]| values.iter().map(|value| value * value).sum::<f64>().sqrt();
        let bar = |dimension: usize, directions: usize, magnitude: f64| {
            (directions as f64 + 2.0) * dimension as f64 * f64::EPSILON * magnitude
        };

        let mut sys = term
            .assemble_arrow_schur(z.view(), &rho, None)
            .expect("the pinned state assembles");
        let reference = term
            .factor_deflated_evidence_with_grad_norms(&mut sys, &lambda_smooth, &options)
            .expect("the pinned state's deflated evidence factor");
        let reference_t = reference
            .delta_t
            .as_slice()
            .expect("the factor's coordinate step is contiguous");
        let offsets = &reference.cache.row_offsets;
        let mut resolvable_premise = 0usize;
        let mut stiffened = 0usize;
        for (row, directions) in reference.cache.deflated_row_directions.iter().enumerate() {
            let segment = &reference_t[offsets[row]..offsets[row + 1]];
            for direction in directions {
                stiffened += 1;
                if along(segment, direction).abs() > bar(segment.len(), directions.len(), norm(segment)) {
                    resolvable_premise += 1;
                }
            }
        }
        eprintln!(
            "[#2822 pin] unit-stiffened directions={stiffened} premise-resolvable={resolvable_premise}"
        );
        assert!(
            resolvable_premise > 0,
            "#2822 premise: at the pinned coordinate the deflated factor's own Newton step must move \
             along some unit-stiffened direction by more than the projection bar \
             ({stiffened} stiffened directions, none resolvable)"
        );

        let root = term
            .evidence_root_step(z.view(), &rho, None, &lambda_smooth, &options)
            .expect("the root step evaluates at the pinned state")
            .expect("the pinned state has a finite positive gate and a finite Newton step");
        let root_t = root
            .step
            .t
            .as_slice()
            .expect("the root step's coordinate block is contiguous");
        assert!(norm(root_t) > 0.0, "the root step must move the coordinates at the pinned state");
        let offsets = &root.cache.row_offsets;
        for (row, directions) in root.cache.deflated_row_directions.iter().enumerate() {
            let segment = &root_t[offsets[row]..offsets[row + 1]];
            for direction in directions {
                let moved = along(segment, direction).abs();
                let allowed = bar(segment.len(), directions.len(), norm(segment));
                assert!(
                    moved <= allowed,
                    "#2822: the root step moves row {row} along a unit-stiffened direction by \
                     {moved:.6e} > {allowed:.6e}; such a direction carries the factor's substituted \
                     unit stiffness, not the objective's curvature"
                );
            }
        }
    }
}
