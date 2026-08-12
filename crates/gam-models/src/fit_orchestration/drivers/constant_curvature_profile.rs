// #2747 — the constant-curvature smooth's outer objective, in its OWN two
// coordinates `ψ = (κ, η = ln ℓ)`.
//
// Extracted from `spatial_optimization.rs` for the same reason
// `constant_curvature_kappa_jet.rs` was: that file sits at the 10,000-line ban
// and this machinery grew when the range became an estimated coordinate rather
// than a heuristic. `include!`d into `drivers/mod.rs` exactly like the sibling
// files, so the flat module namespace and every private-item reference are
// unchanged.
//
// Everything the curvature estimand is built on lives here and nowhere else:
// the value-only criterion the inner bracket screens with, the full ψ jet the
// Newton refines with, the profile object that owns both, and the bounded outer
// solve that mints κ̂. One owner, because the point estimate, the profile CI and
// the flatness LR have to be extrema of the same object — this subsystem
// already carries the scar from the last time one coordinate had two.

/// The profile's VALUE alone at one `(κ, η)`, with no derivative blocks built.
///
/// The bracketing scan calls this and the Newton refinement calls the full jet.
/// The split is worth its own function because the two costs are not close: the
/// value needs one kernel pass (`distance`), the jet needs the Tower2 κ-jet of
/// every pair plus five more `n×p` blocks. A thirteen-point deterministic
/// bracket at jet cost would multiply a production `curv(...)` fit's outer work
/// by an order of magnitude for information the bracket does not use.
fn constant_curvature_psi_profile_value(
    data: ArrayView2<'_, f64>,
    y: ArrayView1<'_, f64>,
    spec: &gam_terms::basis::ConstantCurvatureBasisSpec,
) -> Result<(f64, bool), EstimationError> {
    let mut profile_spec = spec.clone();
    profile_spec.double_penalty = false;
    let basis = gam_terms::basis::build_constant_curvature_basis(data, &profile_spec)
        .map_err(EstimationError::from)?;
    if basis.active_penalties.len() != 1 {
        crate::bail_invalid_estim!(
            "constant-curvature profile expected exactly one primary penalty; got {}",
            basis.active_penalties.len()
        );
    }
    let smooth_design = basis.design.to_dense();
    let (n, p) = smooth_design.dim();
    let mut design = Array2::<f64>::ones((n, p + 1));
    design.slice_mut(s![.., 1..]).assign(&smooth_design);
    let mut penalty = Array2::<f64>::zeros((p + 1, p + 1));
    penalty
        .slice_mut(s![1.., 1..])
        .assign(&basis.active_penalties[0].matrix);
    let response_2d = y.insert_axis(ndarray::Axis(1));
    let fit = gam_solve::gaussian_reml::gaussian_reml_multi_closed_form(
        design.view(),
        response_2d.view(),
        penalty.view(),
        None,
        None,
    )?;
    let rho_at_bound = (fit.rho - gam_solve::gaussian_reml::RHO_LOWER).abs() <= 1.0e-9
        || (fit.rho - gam_solve::gaussian_reml::RHO_UPPER).abs() <= 1.0e-9;
    Ok((fit.reml_score, rho_at_bound))
}

/// Value, exact gradient and exact Hessian of the continuously
/// smoothing-profiled Gaussian REML negative log evidence used for curvature
/// inference, in the smooth's TWO outer coordinates `ψ = (κ, η)`, `η = ln ℓ`.
///
/// The likelihood-ratio statistic must compare values of this one likelihood.
/// Subtracting a second REML fit to a response-dependent radial smoother would
/// produce neither a likelihood nor a calibrated likelihood ratio: the
/// subtraction can manufacture curvature signal even when the response is
/// constant plus noise.
///
/// The range enters as a coordinate rather than as a heuristic because it is
/// confounded with the curvature (#2747): pinning ℓ makes κ absorb the range
/// error, and the criterion then rails, inverts the reported sign, or invents
/// curvature from flat data. The exact second derivatives are what let this
/// route run the SAME stationarity certificate every other route runs (#2458).
fn constant_curvature_psi_profile_jet(
    data: ArrayView2<'_, f64>,
    y: ArrayView1<'_, f64>,
    spec: &gam_terms::basis::ConstantCurvatureBasisSpec,
) -> Result<ProfiledRemlPsiJet, EstimationError> {
    if y.len() != data.nrows() || y.is_empty() {
        crate::bail_invalid_estim!(
            "constant-curvature profile needs one non-empty response per row: data={}, response={}",
            data.nrows(),
            y.len(),
        );
    }

    let mut profile_spec = spec.clone();
    profile_spec.double_penalty = false;
    let basis = gam_terms::basis::build_constant_curvature_basis(data, &profile_spec)
        .map_err(EstimationError::from)?;
    let jets =
        gam_terms::basis::build_constant_curvature_basis_psi_derivatives(data, &profile_spec)
            .map_err(EstimationError::from)?;
    let penalty_block_counts = [
        basis.active_penalties.len(),
        jets.penalties_kappa.len(),
        jets.penalties_eta.len(),
        jets.penalties_kappa2.len(),
        jets.penalties_kappa_eta.len(),
        jets.penalties_eta2.len(),
    ];
    if penalty_block_counts.iter().any(|&count| count != 1) {
        crate::bail_invalid_estim!(
            "constant-curvature profile expected exactly one primary penalty in every block; got {penalty_block_counts:?}"
        );
    }

    let smooth_design = basis.design.to_dense();
    let n = smooth_design.nrows();
    let p = smooth_design.ncols();
    let smooth_penalty = &basis.active_penalties[0].matrix;
    let smooth_design_blocks = [
        &jets.design_kappa,
        &jets.design_eta,
        &jets.design_kappa2,
        &jets.design_kappa_eta,
        &jets.design_eta2,
    ];
    let smooth_penalty_blocks = [
        &jets.penalties_kappa[0],
        &jets.penalties_eta[0],
        &jets.penalties_kappa2[0],
        &jets.penalties_kappa_eta[0],
        &jets.penalties_eta2[0],
    ];
    if smooth_penalty.dim() != (p, p)
        || smooth_design_blocks.iter().any(|m| m.dim() != (n, p))
        || smooth_penalty_blocks.iter().any(|m| m.dim() != (p, p))
    {
        crate::bail_invalid_estim!(
            "constant-curvature ψ derivative bundle does not match its value basis"
        );
    }

    // The unpenalized intercept column is ψ-independent, so it contributes zero
    // to every ψ-derivative and its coordinate stays in the penalty null space
    // at all ψ — the ψ-fixed-null-space premise the jet verifies.
    let mut design = Array2::<f64>::ones((n, p + 1));
    design.slice_mut(s![.., 1..]).assign(&smooth_design);
    let bordered_design = |block: &Array2<f64>| -> Array2<f64> {
        let mut out = Array2::<f64>::zeros((n, p + 1));
        out.slice_mut(s![.., 1..]).assign(block);
        out
    };
    let bordered_penalty = |block: &Array2<f64>| -> Array2<f64> {
        let mut out = Array2::<f64>::zeros((p + 1, p + 1));
        out.slice_mut(s![1.., 1..]).assign(block);
        out
    };
    let penalty = bordered_penalty(smooth_penalty);
    let design_blocks: Vec<Array2<f64>> = smooth_design_blocks
        .iter()
        .map(|block| bordered_design(block))
        .collect();
    let penalty_blocks: Vec<Array2<f64>> = smooth_penalty_blocks
        .iter()
        .map(|block| bordered_penalty(block))
        .collect();

    profiled_gaussian_reml_psi_jet(
        &design,
        &penalty,
        &PsiCoordinateBlocks {
            design_first: [&design_blocks[0], &design_blocks[1]],
            design_second: [&design_blocks[2], &design_blocks[3], &design_blocks[4]],
            penalty_first: [&penalty_blocks[0], &penalty_blocks[1]],
            penalty_second: [&penalty_blocks[2], &penalty_blocks[3], &penalty_blocks[4]],
        },
        y,
    )
}

/// The constant-curvature smooth's outer objective in its own two coordinates.
///
/// `ψ = (κ, η)` with `η = ln ℓ`: the signed sectional curvature and the log
/// kernel range. Both move the design and the penalty, both are estimated, and
/// the reason the second one exists is that it is confounded with the first
/// (#2747) — a κ optimized at a pinned ℓ measures the range error, not the
/// curvature.
///
/// This type is the SINGLE owner of the criterion. The point estimate, the
/// profile CI and the flatness LR all read [`Self::evaluate`], the
/// range-profiled κ jet, so they cannot be extrema of different objects.
struct ConstantCurvatureProfile<'a> {
    data: ArrayView2<'a, f64>,
    response: ArrayView1<'a, f64>,
    spec: gam_terms::basis::ConstantCurvatureBasisSpec,
    /// Derived `[ln ℓ_lo, ln ℓ_hi]` evaluability box; `None` when the user
    /// pinned the range, in which case η is not a coordinate at all.
    eta_bounds: Option<(f64, f64)>,
    /// `[ln d_min⁺, ln d_max]` over the pairs the kernel evaluates — where the
    /// inner search BRACKETS, as opposed to where it is walled.
    eta_bracket: (f64, f64),
    /// `η` seed — the auto rule's realized `ℓ_ref`, in logs.
    eta_seed: f64,
    cache: std::cell::RefCell<std::collections::HashMap<(u64, u64), ProfiledRemlPsiJet>>,
    /// Value-only cache for the bracketing scan: `(V, ρ̂ railed)`; see
    /// [`Self::evaluate_value`].
    value_cache: std::cell::RefCell<std::collections::HashMap<(u64, u64), (f64, bool)>>,
}

/// How the inner range solve at one κ terminated.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RangeSolveOutcome {
    /// `V_η = 0` and `V_ηη > 0` at an η strictly inside the box: the envelope
    /// and Schur reductions of the profile are both valid.
    InteriorMinimum,
    /// The range could not be certified as an interior minimizer — it is pinned
    /// by the user, parked at an evaluability wall, or the inner solve stopped
    /// at a point whose η-curvature does not identify the reduction. The value
    /// is still the best one found, but `η̂` is locally CONSTANT in κ there, so
    /// the profile's derivatives are the plain κ slice.
    LocallyFixed,
    /// `η̂` reached the TOP of the range chart, where the kernel has become the
    /// geodesic-distance kernel to within `√ε` in every design entry
    /// (`constant_curvature_length_scale_bounds`). The reduction is unavailable
    /// for the same reason it is at any wall — but the two are not the same
    /// statement, and conflating them is the whole of gam#2747 on this
    /// coordinate.
    ///
    /// A wall says the estimator was stopped. This says it ARRIVED: `k → −d_κ`
    /// as `ℓ → ∞`, and `−d_κ` is conditionally positive definite on all three
    /// space forms, so the far face of the range is an ordinary non-degenerate
    /// model rather than a degeneracy. `V(ℓ)` converging monotonically to it is
    /// therefore an answer — "the range is at or beyond the point where the
    /// kernel IS the geodesic distance" — and not the "readout of the box"
    /// `20bde053f` reverted the free-range enrollment over. Nothing past the top
    /// is a different model, so nothing past it is worth searching, and the
    /// stopping rule that comment asked for is a consequence of the chart rather
    /// than a rule.
    ///
    /// Declared rather than inferable, exactly as `146f9232d` made
    /// `KappaEstimateSupport` for the curvature coordinate: a consumer that
    /// reads `ℓ̂` alone cannot tell an arrival from a truncation, and the two
    /// support very different claims about the magnitude.
    DistanceKernelLimit,
}

impl<'a> ConstantCurvatureProfile<'a> {
    /// Construct the curvature-estimation profile in its fit-time constraint
    /// frame.
    ///
    /// A frozen transform is a predict-time replay artifact: it is the global
    /// identifiability frame realized at one particular fitted ψ. Reusing that
    /// fixed frame while this profile varies ψ changes the objective and omits
    /// the frame's ψ derivative. Inference must instead use the same local
    /// center-sum-to-zero quotient that produced the point estimate. Realized
    /// centers remain a valid frozen representation of a deterministic fit-time
    /// choice, so only the ψ-anchored transform is removed.
    fn new(
        data: ArrayView2<'a, f64>,
        response: ArrayView1<'a, f64>,
        mut spec: gam_terms::basis::ConstantCurvatureBasisSpec,
    ) -> Result<Self, EstimationError> {
        if response.len() != data.nrows() || response.is_empty() {
            crate::bail_invalid_estim!(
                "constant-curvature profile needs one non-empty response per row: data={}, response={}",
                data.nrows(),
                response.len(),
            );
        }
        spec.identifiability = gam_terms::basis::ConstantCurvatureIdentifiability::CenterSumToZero;
        // Box, bracket and seed are all read from the REALIZED center set — the
        // one the basis builder itself will use — and in the κ = 0 chart gauge,
        // so all three are κ-FIXED and none of them moves while the optimizer
        // walks κ.
        //
        // They are DERIVED, not configured, and deliberately do not consult
        // `SpatialLengthScaleOptimizationOptions`: the κ box beside them is
        // derived the same way (the half-margin to the antipodal fold), and the
        // curvature-inference entry point has no access to those options at
        // all. A box visible to the fit but not to the profile CI would put the
        // point estimate and its interval on two different parameter spaces.
        let centers = gam_terms::basis::constant_curvature_realized_centers(data, &spec)
            .map_err(EstimationError::from)?;
        let ell_seed = gam_terms::basis::realized_constant_curvature_length_scale(
            centers.view(),
            spec.length_scale,
        )
        .map_err(EstimationError::from)?;
        let (span_lo, span_hi) =
            gam_terms::basis::constant_curvature_evaluated_scale_span(data, centers.view())
                .map_err(EstimationError::from)?;
        let eta_bounds = if spec.length_scale_fixed {
            None
        } else {
            let (lo, hi) =
                gam_terms::basis::constant_curvature_length_scale_bounds(data, centers.view())
                    .map_err(EstimationError::from)?;
            Some((lo.ln(), hi.ln()))
        };
        let eta_seed = match eta_bounds {
            Some((lo, hi)) => ell_seed.ln().clamp(lo, hi),
            None => ell_seed.ln(),
        };
        Ok(Self {
            data,
            response,
            spec,
            eta_bounds,
            eta_bracket: (span_lo.ln(), span_hi.ln()),
            eta_seed,
            cache: std::cell::RefCell::new(std::collections::HashMap::new()),
            value_cache: std::cell::RefCell::new(std::collections::HashMap::new()),
        })
    }

    /// The profile VALUE at one point of the plane, without derivative blocks.
    ///
    /// Shares the jet cache: a point already evaluated at full order answers
    /// from there, so the bracket never re-pays for a point the Newton has
    /// visited and vice versa.
    fn evaluate_value(&self, kappa: f64, eta: f64) -> Result<f64, EstimationError> {
        if !(kappa.is_finite() && eta.is_finite()) {
            crate::bail_invalid_estim!(
                "constant-curvature profile probed a non-finite ψ = ({kappa}, {eta})"
            );
        }
        let key = (kappa.to_bits(), eta.to_bits());
        if let Some(cached) = self.cache.borrow().get(&key) {
            return Self::comparable_value(kappa, eta, cached.value, cached.rho_at_bound);
        }
        if let Some(&cached) = self.value_cache.borrow().get(&key) {
            return Self::comparable_value(kappa, eta, cached.0, cached.1);
        }
        let mut probe_spec = self.spec.clone();
        probe_spec.kappa = kappa;
        probe_spec.length_scale = eta.exp();
        let sample = constant_curvature_psi_profile_value(self.data, self.response, &probe_spec)?;
        self.value_cache.borrow_mut().insert(key, sample);
        Self::comparable_value(kappa, eta, sample.0, sample.1)
    }

    /// A criterion value the range search is allowed to COMPARE, or a refusal
    /// naming why not.
    ///
    /// `V` is a λ-profile only where `ρ̂` is interior. At a rail it is a
    /// constrained minimum over a truncated λ range, and a constrained minimum
    /// is not comparable to an unconstrained one — picking the smaller of the
    /// two is picking whichever happened to be truncated harder.
    ///
    /// **The reason this was written is gone, and the check is kept anyway.**
    /// It was added because the range coordinate DROVE `ρ̂`: the realized design
    /// scaled like `1/ℓ`, so λ had to follow it and `ρ̂ ≈ const − ln ℓ`
    /// (measured: each ×100 in `ℓ` cost 4.6 in `ρ̂`, which is `ln 100`), and a
    /// range box eight orders wide was therefore always wide enough to walk `ρ̂`
    /// into `RHO_LOWER` for no statistical reason whatever. That was the
    /// `exp(−d/ℓ)` gauge's `1/ℓ` collapse, and gam#2747 removed it at the
    /// source: in the contrast gauge `ℓ·(e^{−d/ℓ} − 1)` the design does not
    /// collapse and `ρ̂` is flat in the range (measured: `−5.0978 ± 1e-4` across
    /// eleven decades on the κ=1 sphere fixture). So this refusal should now
    /// almost never fire from the range coordinate.
    ///
    /// It stays because the ARGUMENT was never about the range. A constrained
    /// minimum is not comparable to an unconstrained one whatever drove it
    /// there, and a dataset whose λ̂ genuinely wants to leave the ρ box still
    /// exists. What changed is its status: it was a systematic artefact of a
    /// gauge and is now a rare, real event.
    ///
    /// Refusing rather than clamping is deliberate: the point is not infeasible
    /// for the MODEL, only unusable as a comparison, and the search treats a
    /// refusal exactly as it treats an unbuildable design — it moves on.
    fn comparable_value(
        kappa: f64,
        eta: f64,
        value: f64,
        rho_at_bound: bool,
    ) -> Result<f64, EstimationError> {
        if rho_at_bound {
            crate::bail_invalid_estim!(
                "constant-curvature profile at ψ = ({kappa}, ln ℓ = {eta}) railed ρ̂ at its bound,                  so its value is a truncated minimum and not comparable across the range"
            );
        }
        Ok(value)
    }

    /// The full `(κ, η)` jet at one point of the plane.
    fn evaluate_psi(&self, kappa: f64, eta: f64) -> Result<ProfiledRemlPsiJet, EstimationError> {
        if !(kappa.is_finite() && eta.is_finite()) {
            crate::bail_invalid_estim!(
                "constant-curvature profile probed a non-finite ψ = ({kappa}, {eta})"
            );
        }
        let key = (kappa.to_bits(), eta.to_bits());
        if let Some(cached) = self.cache.borrow().get(&key) {
            return Ok(cached.clone());
        }
        let mut probe_spec = self.spec.clone();
        probe_spec.kappa = kappa;
        probe_spec.length_scale = eta.exp();
        let sample = constant_curvature_psi_profile_jet(self.data, self.response, &probe_spec)?;
        self.cache.borrow_mut().insert(key, sample.clone());
        Ok(sample)
    }

    /// `η̂(κ) = argmin_η V(κ, η)` on the evaluability box, and the jet there.
    ///
    /// A deterministic scan across the geometry's own scale span brackets the
    /// basin before a safeguarded Newton refines it. The scan is an
    /// INITIALIZATION, not a constraint: the measured criterion puts its minimum
    /// above the largest evaluated separation on a third of the planted
    /// fixtures, and the Newton is free to walk there — only the evaluability
    /// box stops it. Making the bracket deterministic (rather than warm-starting
    /// from the previous κ) is what makes `V_p` a function of κ alone, which the
    /// CI walk and the LR test both require.
    ///
    /// # What this costs, and why it is paid
    ///
    /// Roughly seventeen criterion evaluations per κ where the pinned-range
    /// criterion needed one — thirteen bracket points and a handful of Newton
    /// steps — of which only the accepted iterates pay for derivative blocks
    /// (see [`Self::evaluate_value`]). That is the intrinsic price of profiling
    /// a nuisance coordinate rather than guessing it, and the alternative is the
    /// defect: a κ estimated against a heuristic range is an estimate of the
    /// range error. Warm-starting from the previous κ would cut the bracket, but
    /// it makes `V_p` depend on the ORDER the CI walk visits κ, and a profile
    /// likelihood that is not a function of its own argument cannot support an
    /// interval.
    fn minimize_over_eta(
        &self,
        kappa: f64,
    ) -> Result<(f64, ProfiledRemlPsiJet, RangeSolveOutcome), EstimationError> {
        let Some((lo, hi)) = self.eta_bounds else {
            let jet = self.evaluate_psi(kappa, self.eta_seed)?;
            return Ok((self.eta_seed, jet, RangeSolveOutcome::LocallyFixed));
        };
        // Bracket over the evaluated scale span (clamped into the box), plus the
        // seed, so a user-supplied or previously fitted range is always among
        // the candidates.
        const SCAN_POINTS: usize = 13;
        let scan_lo = self.eta_bracket.0.clamp(lo, hi);
        let scan_hi = self.eta_bracket.1.clamp(lo, hi);
        // Two lists: points whose value the search may COMPARE (interior ρ̂),
        // and every point that evaluated at all. The second exists only so a
        // dataset whose ρ̂ rails everywhere still gets an answer — it is a
        // fallback, and the outcome it produces is reported as `LocallyFixed`
        // because a comparison across truncated minima is not a minimization.
        let mut comparable: Vec<(f64, f64)> = Vec::with_capacity(SCAN_POINTS + 1);
        let mut any: Vec<(f64, f64)> = Vec::with_capacity(SCAN_POINTS + 1);
        let consider = |eta: f64, ok: &mut Vec<(f64, f64)>, all: &mut Vec<(f64, f64)>| {
            match self.evaluate_value(kappa, eta) {
                Ok(value) => {
                    ok.push((value, eta));
                    all.push((value, eta));
                }
                Err(_) => {
                    if let Some(&(value, _)) = self
                        .value_cache
                        .borrow()
                        .get(&(kappa.to_bits(), eta.to_bits()))
                    {
                        all.push((value, eta));
                    }
                }
            }
        };
        for i in 0..SCAN_POINTS {
            consider(
                scan_lo + (scan_hi - scan_lo) * (i as f64) / ((SCAN_POINTS - 1) as f64),
                &mut comparable,
                &mut any,
            );
        }
        consider(self.eta_seed, &mut comparable, &mut any);
        let fell_back = comparable.is_empty();
        let mut candidates = if fell_back { any } else { comparable };
        candidates.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        // The jet carries checks the value path does not (the ψ-fixed null-space
        // premise, and the chart's reproduction of the forward score), so a point
        // the value accepts can still be one the jet refuses. Walk the scan in
        // value order and start from the best point that yields a jet, rather
        // than failing the whole profile at that κ.
        let started = candidates.iter().find_map(|&(_, eta)| {
            self.evaluate_psi(kappa, eta).ok().map(|jet| (eta, jet))
        });
        let Some((mut eta, mut jet)) = started else {
            crate::bail_invalid_estim!(
                "constant-curvature profile could not evaluate the range box at κ = {kappa}"
            );
        };
        if fell_back {
            // Every probed range railed ρ̂. There is no comparison to make, so
            // the best raw value stands and the range is declared locally fixed
            // rather than pretending to be a certified minimizer.
            return Ok((eta, jet, RangeSolveOutcome::LocallyFixed));
        }
        // Safeguarded Newton on η at fixed κ. The trust step is capped at the
        // BRACKET width rather than the (deliberately enormous) box width, so a
        // flat or non-convex stretch cannot throw the iterate fifteen orders of
        // magnitude away from the geometry.
        const MAX_NEWTON: usize = 60;
        let width = (scan_hi - scan_lo).max(1.0);
        // The criterion's own forward resolution in η. A value-comparing line
        // search cannot separate two η closer than this — the values differ
        // below the rounding of `V` — so an iterate that stops moving at this
        // scale HAS converged, and demanding a smaller gradient than the value
        // can resolve would classify every successful solve as a stall.
        let eta_resolution = |eta: f64| f64::EPSILON.sqrt() * (1.0 + eta.abs());
        let mut converged = false;
        for _ in 0..MAX_NEWTON {
            let g = jet.gradient[1];
            let h = jet.hessian[1][1];
            let at_lo = eta <= lo + 1.0e-12 * (1.0 + lo.abs());
            let at_hi = eta >= hi - 1.0e-12 * (1.0 + hi.abs());
            if at_hi && g <= 0.0 {
                return Ok((eta, jet, RangeSolveOutcome::DistanceKernelLimit));
            }
            if at_lo && g >= 0.0 {
                return Ok((eta, jet, RangeSolveOutcome::LocallyFixed));
            }
            if g.abs() <= 1.0e-9 * (1.0 + jet.value.abs()) {
                converged = true;
                break;
            }
            if h.is_finite() && h > 0.0 && (g / h).abs() <= eta_resolution(eta) {
                // The exact Newton step is already inside the resolution: the
                // remaining gradient is below what the value can express.
                converged = true;
                break;
            }
            let raw = if h.is_finite() && h > 0.0 {
                -g / h
            } else {
                -g.signum() * 0.25 * width
            };
            let mut step = raw.clamp(-width, width);
            let mut accepted = None;
            for _ in 0..30 {
                let trial = (eta + step).clamp(lo, hi);
                if (trial - eta).abs() <= 1.0e-14 * (1.0 + eta.abs()) {
                    break;
                }
                // Screen the trial with the cheap value; only the ACCEPTED point
                // pays for a jet. A trial the value accepts but the jet refuses
                // is treated as a rejected trial and the step keeps shrinking.
                if let Ok(value) = self.evaluate_value(kappa, trial)
                    && value <= jet.value
                    && let Ok(next) = self.evaluate_psi(kappa, trial)
                {
                    accepted = Some((trial, next));
                    break;
                }
                step *= 0.5;
            }
            match accepted {
                Some((next_eta, next_jet)) => {
                    let moved = (next_eta - eta).abs();
                    eta = next_eta;
                    jet = next_jet;
                    if moved <= eta_resolution(eta) {
                        converged = true;
                        break;
                    }
                }
                None => break,
            }
        }
        let at_top = eta >= hi - 1.0e-9 * (1.0 + hi.abs());
        let interior = converged
            && jet.hessian[1][1].is_finite()
            && jet.hessian[1][1] > 0.0
            && eta > lo + 1.0e-9 * (1.0 + lo.abs())
            && !at_top;
        // The chart's TOP is the geodesic-distance face and its bottom is an
        // evaluability wall, so an iterate that stopped at one is not the same
        // finding as an iterate that stopped at the other — see
        // `RangeSolveOutcome::DistanceKernelLimit`. Both give the plain κ slice;
        // only one of them is an answer about the model.
        let outcome = match (interior, at_top) {
            (true, _) => RangeSolveOutcome::InteriorMinimum,
            (false, true) => RangeSolveOutcome::DistanceKernelLimit,
            (false, false) => RangeSolveOutcome::LocallyFixed,
        };
        Ok((eta, jet, outcome))
    }

    /// `(V_p(κ), V_p′(κ), V_p″(κ))` with the range PROFILED out — the
    /// one-dimensional likelihood the point estimate, the CI and the flatness
    /// test all consume.
    ///
    /// At a certified interior η̂ the envelope theorem gives `V_p′ = V_κ` and the
    /// Schur complement gives `V_p″ = V_κκ − V_κη²/V_ηη`. Otherwise η̂ is
    /// locally constant in κ, the reduction is absent, and the derivatives are
    /// the plain κ slice at the η actually used — exactly the premise this file
    /// already applies to a railed ρ̂. The VALUE is the best one found either
    /// way, so the criterion the CI compares is the same object in both cases.
    fn evaluate(&self, kappa: f64) -> Result<(f64, f64, f64), EstimationError> {
        let (_, jet, outcome) = self.minimize_over_eta(kappa)?;
        match outcome {
            RangeSolveOutcome::InteriorMinimum => jet.eta_profiled_kappa_jet(),
            RangeSolveOutcome::LocallyFixed | RangeSolveOutcome::DistanceKernelLimit => {
                Ok(jet.kappa_slice())
            }
        }
    }
}

/// `ℓ̂` at a PINNED κ — the range half of the profile, run on its own.
///
/// A pinned `kappa=` fixes the geometry (gam#2152) and takes the term out of the
/// curvature search. It does not fix the RANGE, and the two were coupled here
/// only because one function owned both: `20bde053f` reverted the pinned-κ /
/// free-range enrollment because the range criterion "is monotone in ell all
/// the way to its asymptote … a readout of the box rather than of the data".
/// That reading was correct about the symptom and wrong about the cause — past
/// `ℓ ≈ 10⁶` the old kernel gauge's criterion was fabricated, descending ~100
/// nats per decade into its own cancellation — and both halves are fixed:
/// the criterion is now a function of the data across the whole chart, and the
/// chart's top is the geodesic-distance face, an arrival the solve DECLARES
/// (see [`RangeSolveOutcome::DistanceKernelLimit`]).
///
/// So the range is estimated whenever the user did not pin it, at whatever κ the
/// term carries. This runs the same inner solve
/// [`ConstantCurvatureProfile::minimize_over_eta`] the full profile runs at each
/// trial κ — one owner, one objective — rather than a second range search with
/// its own bracket.
fn constant_curvature_range_only_optimum(
    data: ArrayView2<'_, f64>,
    y: ArrayView1<'_, f64>,
    resolvedspec: &TermCollectionSpec,
    term_idx: usize,
) -> Result<f64, EstimationError> {
    let (feature_cols, base_spec) = match resolvedspec
        .smooth_terms
        .get(term_idx)
        .map(|term| &term.basis)
    {
        Some(SmoothBasisSpec::ConstantCurvature {
            feature_cols, spec, ..
        }) => (feature_cols, spec.clone()),
        _ => {
            crate::bail_invalid_estim!(
                "constant-curvature range optimum requested for non-curvature term {term_idx}"
            )
        }
    };
    let pinned_kappa = base_spec.kappa;
    let x_term = select_columns(data, feature_cols).map_err(EstimationError::from)?;
    let profile = ConstantCurvatureProfile::new(x_term.view(), y, base_spec)?;
    let (eta_hat, _, outcome) = profile.minimize_over_eta(pinned_kappa)?;
    let length_scale_hat = eta_hat.exp();
    log::info!(
        "[spatial-kappa] pinned kappa={pinned_kappa:.6}: range profiled to \
         length_scale_hat={length_scale_hat:.6} ({outcome:?}) for term {term_idx}",
    );
    Ok(length_scale_hat)
}

fn validate_constant_curvature_profile_inputs(
    weights: ArrayView1<'_, f64>,
    offset: ArrayView1<'_, f64>,
    family: &LikelihoodSpec,
) -> Result<(), EstimationError> {
    if *family != LikelihoodSpec::gaussian_identity() {
        crate::bail_invalid_estim!(
            "curvature-as-an-estimand profile currently requires Gaussian identity likelihood"
        );
    }
    let input_tolerance = f64::EPSILON.sqrt();
    if weights
        .iter()
        .any(|&weight| (weight - 1.0).abs() > input_tolerance)
        || offset.iter().any(|&value| value.abs() > input_tolerance)
    {
        crate::bail_invalid_estim!(
            "curvature-as-an-estimand profile requires unit weights and zero offset"
        );
    }
    Ok(())
}

/// The constant-curvature smooth's fitted outer coordinates.
#[derive(Clone, Copy, Debug)]
struct ConstantCurvatureOptimum {
    /// Signed sectional curvature κ̂.
    kappa: f64,
    /// Kernel range ℓ̂ = exp(η̂(κ̂)) — the range the criterion profiles to at the
    /// fitted curvature. Equals the pinned value when the user set
    /// `length_scale=`.
    length_scale: f64,
}

/// Minimize the RANGE-PROFILED, continuously smoothing-profiled Gaussian REML
/// evidence `V_p(κ) = min_{η,ρ} V(κ, η, ρ)` on the chart-valid κ interval, with
/// the shared bounded analytic outer solver — so every accepted result has
/// passed the solver's final box-KKT projected-gradient certificate. No sampled
/// point is ever returned as the estimate: samples are only line-search probes
/// for the continuous solve.
///
/// # Why the range is profiled rather than searched jointly
///
/// The range has to be estimated at all because it is confounded with the
/// curvature (#2747): the two enter `exp(−d_κ/ℓ)` through one exponent, so a κ
/// optimized against a pinned ℓ reports the range error rather than the
/// curvature — measured, it rails, inverts the sign, or invents curvature from
/// flat data.
///
/// But it must be profiled, not co-searched, because **the point estimate and
/// the interval have to be extrema of the SAME object**. A joint search over
/// `(κ, η)` returns a local stationary point of `V(κ, η)`, while the profile CI
/// and the flatness LR compare values of `V_p(κ) = min_η V(κ, η)`; where the
/// two disagree the reported κ̂ is not the argmin of its own interval's
/// criterion. This file already carries the scar from the last time one
/// coordinate had two objective owners — see the `spatial_terms` filter, which
/// exists because that "made the scalar and joint routes disagree at the
/// identical seed on flat data". So there is one owner: `ConstantCurvatureProfile`,
/// whose inner range solve is deterministic and globally bracketed, and whose
/// κ jet is the exact envelope/Schur reduction of it.
///
/// A user who pins `length_scale=` gets the same one-dimensional κ search at
/// that range, exactly as a user who pins `kappa=` gets fixed geometry.
fn constant_curvature_kappa_profile_optimum(
    data: ArrayView2<'_, f64>,
    y: ArrayView1<'_, f64>,
    resolvedspec: &TermCollectionSpec,
    term_idx: usize,
    options: &FitOptions,
) -> Result<ConstantCurvatureOptimum, EstimationError> {
    let (kappa_min, kappa_max) = constant_curvature_kappa_bounds(data, resolvedspec, term_idx);
    if !(kappa_min.is_finite() && kappa_max.is_finite() && kappa_max > kappa_min) {
        crate::bail_invalid_estim!(
            "constant-curvature term {term_idx} has invalid kappa bounds [{kappa_min}, {kappa_max}]"
        );
    }
    let (feature_cols, base_spec) = match resolvedspec
        .smooth_terms
        .get(term_idx)
        .map(|term| &term.basis)
    {
        Some(SmoothBasisSpec::ConstantCurvature {
            feature_cols, spec, ..
        }) => (feature_cols, spec.clone()),
        _ => {
            crate::bail_invalid_estim!(
                "constant-curvature optimum requested for non-curvature term {term_idx}"
            )
        }
    };
    let x_term = select_columns(data, feature_cols).map_err(EstimationError::from)?;
    let profile = ConstantCurvatureProfile::new(x_term.view(), y, base_spec)?;
    let mut seed_config = gam_problem::SeedConfig::default();
    seed_config.bounds = (kappa_min, kappa_max);
    seed_config.max_seeds = 1;
    seed_config.seed_budget = 1;
    seed_config.risk_profile = gam_problem::SeedRiskProfile::Gaussian;
    seed_config.num_auxiliary_trailing = 1;
    seed_config.over_smoothing_probe_rho = None;
    let initial_kappa = profile.spec.kappa.clamp(kappa_min, kappa_max);
    let problem = gam_solve::rho_optimizer::OuterProblem::new(1)
        .with_gradient(gam_problem::Derivative::Analytic)
        // #2458: the κ profile supplies an EXACT d²V_p/dκ², so this route runs
        // the same curvature-denominated stationarity certificate every other
        // route runs. It previously declared `Unavailable` — not because the
        // curvature was unavailable, but because this call site never asked the
        // basis bundle for the seconds it already ships.
        .with_hessian(gam_problem::DeclaredHessianForm::Dense)
        // Gradient-only SEARCH is retained deliberately: the change this makes
        // is the terminal certification, not the trajectory. Declaring the
        // Hessian while preferring gradient-only routes the planner through the
        // `(Analytic, Analytic) if prefer_gradient_only` arm to the same BFGS it
        // used before, so kappa-hat is selected by the same solve -- but the
        // terminal mint can now MEASURE curvature and run the derived criterion
        // instead of the un-derived gradient band.
        .with_prefer_gradient_only(true)
        .with_disable_fixed_point(true)
        .with_fallback_policy(gam_solve::rho_optimizer::FallbackPolicy::Disabled)
        .with_psi_dim(1)
        .with_tolerance(options.tol.max(f64::EPSILON.sqrt()))
        .with_max_iter(options.max_iter.max(1))
        .with_bounds(
            Array1::from_vec(vec![kappa_min]),
            Array1::from_vec(vec![kappa_max]),
        )
        .with_initial_rho(Array1::from_vec(vec![initial_kappa]))
        .with_seed_config(seed_config);
    let mut objective = problem.build_objective(
        profile,
        |profile: &mut ConstantCurvatureProfile<'_>, theta: &Array1<f64>| {
            profile.evaluate(theta[0]).map(|(value, _, _)| value)
        },
        |profile: &mut ConstantCurvatureProfile<'_>, theta: &Array1<f64>| {
            let (cost, derivative, curvature) = profile.evaluate(theta[0])?;
            Ok(gam_problem::OuterEval {
                cost,
                gradient: Array1::from_vec(vec![derivative]),
                hessian: gam_problem::HessianValue::Dense(
                    Array2::from_shape_vec((1, 1), vec![curvature]).expect("1x1 from one element"),
                ),
                inner_beta_hint: None,
            })
        },
        None::<fn(&mut ConstantCurvatureProfile<'_>)>,
        None::<
            fn(
                &mut ConstantCurvatureProfile<'_>,
                &Array1<f64>,
            ) -> Result<gam_problem::EfsEval, EstimationError>,
        >,
    );
    let result = problem.run(
        &mut objective,
        &format!("constant-curvature likelihood profile term {term_idx}"),
    )?;
    if !result.converged() {
        crate::bail_invalid_estim!(
            "constant-curvature likelihood-profile κ optimization did not converge for term {} after {} iterations (negative_log_evidence={:.6e}, final_grad_norm={})",
            term_idx,
            result.iterations,
            result.final_value,
            result.final_grad_norm_report(),
        );
    }
    let kappa_hat = result.rho[0];
    // Read ℓ̂ off the SAME profile object the solve just used, so the reported
    // range is the one the accepted κ̂ was profiled against (and replays from its
    // cache rather than re-solving).
    let (eta_hat, _, range_outcome) = objective.state.minimize_over_eta(kappa_hat)?;
    let length_scale_hat = eta_hat.exp();
    // The range's support, said rather than left to be read off the magnitude
    // (gam#2747). `DistanceKernelLimit` is not a rail: the kernel has become
    // `−d_κ`, which is the model, so `ℓ̂` there is a lower bound with a meaning
    // and not a readout of a box.
    let range_support = match range_outcome {
        RangeSolveOutcome::InteriorMinimum => "interior",
        RangeSolveOutcome::DistanceKernelLimit => "at the geodesic-distance limit",
        RangeSolveOutcome::LocallyFixed => "locally fixed",
    };
    log::info!(
        "[spatial-kappa] continuous likelihood-profile optimum kappa_hat={:.6} \
         length_scale_hat={:.6} ({range_support}) \
         (negative_log_evidence={:.6e}, projected_gradient={}) for term {term_idx}",
        kappa_hat,
        length_scale_hat,
        result.final_value,
        result.final_grad_norm_report(),
    );
    Ok(ConstantCurvatureOptimum {
        kappa: kappa_hat,
        length_scale: length_scale_hat,
    })
}
