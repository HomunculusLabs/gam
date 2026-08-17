//! #2720 — WHICH TERM of the penalized objective is not flat along the chart
//! orbit, measured per direction, per atom kind, by finite difference of the
//! objective's own value functions.
//!
//! ## What this adds to the thread, and why the existing numbers are not enough
//!
//! Every measurement on #2720 so far reports ONE scalar per direction: `|gᵀvᵢ|`
//! against the convergence tolerance (`8.24x` / `10.45x`, later `2.33x` /
//! `3.63x` per geometry). That scalar says the penalized objective is not flat.
//! It does not say WHICH penalty is not flat, and the modelling fix is a
//! different change depending on the answer:
//!
//! * if it is the **ARD prior on the coordinates**, the prior's fixed origin
//!   (Gaussian mean 0 on a Euclidean axis, von Mises mean 0 on a periodic one)
//!   is an arbitrary gauge choice, and the fix is to stop making it;
//! * if it is the **smoothness prior on the decoder**, the penalty matrix `S` is
//!   attached to a chart whose coordinates moved, and the fix is in the basis;
//! * if it is a **barrier**, the direction is not a symmetry of the admissible
//!   set at all and must simply leave the quotient.
//!
//! `|gᵀvᵢ|` cannot distinguish those three. A per-term central difference of the
//! value functions can, and the value functions are the right instrument
//! precisely because they are what the line search descends — an analytic
//! gradient agreeing with itself proves nothing (#2714).
//!
//! ## The second question this answers: is the direction likelihood-null for a
//! REASON, or only because the least-squares solve had room?
//!
//! [`SaeManifoldTerm::dense_step_gauge_vector_from_field`] certifies nothing. It
//! solves `design·δB = −motion` in the least-squares sense and returns the
//! solution whatever the residual is. When the design `n × M` has full ROW rank
//! (`M ≥ rank`), that solve is exact for EVERY field, so a `1e-16` reconstruction
//! residual is a statement about the shape of the fixture and not about the
//! field being a symmetry of anything. This module records `n_active` against
//! `M` next to every residual so the two readings cannot be confused.

// `manifold/mod.rs` declares this module as
// `#[cfg(test)] mod tests_gauge_posterior_flatness_2720;` — its single
// declaration.
#![cfg(test)]
use super::*;

/// The #2253/#2234 planted circle, verbatim from
/// `tests_gauge_frame_roundtrip_2720` (same LCG, same seed, same shape) so every
/// number here sits on the geometry the `1.29e-16` unframed measurement and the
/// `2.33x` geometry sweep were both taken on.
pub(crate) fn planted_circle_cloud() -> Array2<f64> {
    let n = 42usize;
    let p = 48usize;
    let mut state = 0x2468_ace0_1357_9bdfu64;
    let mut unit = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64) / ((1u64 << 53) as f64)
    };
    let two_pi = std::f64::consts::TAU;
    let b0: Vec<f64> = (0..p).map(|_| 2.0 * unit() - 1.0).collect();
    let b1: Vec<f64> = (0..p).map(|_| 2.0 * unit() - 1.0).collect();
    let mut z = Array2::<f64>::zeros((n, p));
    for i in 0..n {
        let theta = two_pi * unit();
        for j in 0..p {
            let noise = 0.01 * (2.0 * unit() - 1.0);
            z[[i, j]] = theta.cos() * b0[j] + theta.sin() * b1[j] + noise;
        }
    }
    z
}

/// A seeded single-atom term on `basis`, at latent dimension `latent_dim`.
pub(crate) fn seeded_term_of_kind(
    target: ArrayView2<'_, f64>,
    basis: &str,
    latent_dim: usize,
) -> SaeManifoldTerm {
    let minimal = build_sae_minimal_seed(SaeMinimalSeedRequest {
        target,
        atom_basis: vec![basis.to_string()],
        atom_dim: vec![latent_dim],
        assignment_kind: SaeFitAssignmentKind::Softmax,
        alpha: 1.0,
        tau: 1.0,
        threshold: 0.0,
        top_k: None,
        random_state: 45,
        initial_logits: None,
        initial_coords: None,
    })
    .expect("minimal seed");
    let registry = AnalyticPenaltyRegistry::new();
    let seed = build_sae_fit_seed(SaeFitSeedRequest {
        target,
        geometry_plans: &minimal.geometry_plans,
        basis_values: minimal.basis_values.view(),
        basis_jacobian: minimal.basis_jacobian.view(),
        decoder_coefficients: minimal.decoder_coefficients.view(),
        smooth_penalties: minimal.smooth_penalties.view(),
        initial_logits: minimal.initial_logits.view(),
        initial_coords: minimal.initial_coords.view(),
        alpha: 1.0,
        tau: 1.0,
        learnable_alpha: false,
        assignment_kind: SaeFitAssignmentKind::Softmax,
        sparsity_strength: 1.0,
        smoothness: 1.0,
        max_iter: 40,
        learning_rate: 0.05,
        ridge_ext_coord: 1.0e-6,
        ridge_beta: 1.0e-6,
        top_k: None,
        threshold: 0.0,
        native_ard_enabled: true,
        seed_refine_routing: minimal.refine_routing,
        seed_refine_random_state: 45,
        data_row_reseed: false,
        fit_config: SaeFitConfig::default(),
        temperature_schedule: None,
        fisher_metric: None,
        row_loss_weights: None,
        registry: &registry,
    })
    .expect("fit seed");
    seed.base_term
}

/// The penalized objective, split into the pieces `penalized_objective_total`
/// sums. Every field is a VALUE, not a gradient — this struct is differenced,
/// never differentiated.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct ObjectiveTerms {
    pub data_fit: f64,
    pub assignment_sparsity: f64,
    pub smoothness: f64,
    pub ard: f64,
    pub analytic: f64,
    pub repulsion: f64,
    pub amplitude_barrier: f64,
    pub separation_barrier: f64,
}

impl ObjectiveTerms {
    pub(crate) fn total(&self) -> f64 {
        self.data_fit
            + self.assignment_sparsity
            + self.smoothness
            + self.ard
            + self.analytic
            + self.repulsion
            + self.amplitude_barrier
            + self.separation_barrier
    }

    fn combine(&self, other: &Self, scale: f64) -> Self {
        Self {
            data_fit: (self.data_fit - other.data_fit) * scale,
            assignment_sparsity: (self.assignment_sparsity - other.assignment_sparsity) * scale,
            smoothness: (self.smoothness - other.smoothness) * scale,
            ard: (self.ard - other.ard) * scale,
            analytic: (self.analytic - other.analytic) * scale,
            repulsion: (self.repulsion - other.repulsion) * scale,
            amplitude_barrier: (self.amplitude_barrier - other.amplitude_barrier) * scale,
            separation_barrier: (self.separation_barrier - other.separation_barrier) * scale,
        }
    }
}

pub(crate) fn objective_terms(
    term: &SaeManifoldTerm,
    target: ArrayView2<'_, f64>,
    rho: &SaeManifoldRho,
    registry: &AnalyticPenaltyRegistry,
) -> Result<ObjectiveTerms, String> {
    let loss = term.loss_scaled(target, rho, 1.0)?;
    Ok(ObjectiveTerms {
        data_fit: loss.data_fit,
        assignment_sparsity: loss.assignment_sparsity,
        smoothness: loss.smoothness,
        ard: loss.ard,
        analytic: term
            .analytic_penalty_value_total(registry, 1.0)
            .map_err(|err| err.to_string())?,
        repulsion: term.decoder_repulsion_value(1.0),
        amplitude_barrier: term.amplitude_barrier_value(1.0),
        separation_barrier: term.separation_barrier_value(1.0),
    })
}

/// Central difference of every objective term along the unit direction
/// `(δt, δβ)`, at step `h`, through the SAME `apply_newton_step` the inner solve
/// moves with (so the measured surface is the one the solver walks, wrapping and
/// basis rebuild included).
pub(crate) fn directional_derivative_terms(
    term: &mut SaeManifoldTerm,
    target: ArrayView2<'_, f64>,
    rho: &SaeManifoldRho,
    registry: &AnalyticPenaltyRegistry,
    direction: &Array1<f64>,
    h: f64,
) -> Result<ObjectiveTerms, String> {
    let dense_len = term.n_obs() * term.assignment.row_block_dim();
    let snapshot = term.snapshot_mutable_state();
    // `apply_newton_step` refuses a non-positive step, so the backward arm walks
    // the NEGATED direction at the same positive length rather than a negative
    // length along the same one. Same point, same code path.
    let evaluate = |term: &mut SaeManifoldTerm,
                    walk: &Array1<f64>|
     -> Result<ObjectiveTerms, String> {
        term.apply_newton_step(walk.slice(s![..dense_len]), walk.slice(s![dense_len..]), h)?;
        let out = objective_terms(term, target, rho, registry);
        term.restore_mutable_state(&snapshot)?;
        out
    };
    let backward = direction.mapv(|value| -value);
    let plus = evaluate(term, direction)?;
    let minus = evaluate(term, &backward)?;
    Ok(plus.combine(&minus, 1.0 / (2.0 * h)))
}

/// How many rows the atom is actually active on, against the atom's basis size.
/// `n_active <= m` is the regime in which the gauge construction's least-squares
/// compensation is exact for ANY field, symmetry or not.
pub(crate) fn active_rows_and_basis_size(
    term: &SaeManifoldTerm,
    atom_idx: usize,
) -> Result<(usize, usize), String> {
    let mut active = 0usize;
    for row in 0..term.n_obs() {
        let assignments = term.assignment.try_assignments_row(row)?;
        if assignments[atom_idx] != 0.0 {
            active += 1;
        }
    }
    Ok((active, term.atoms[atom_idx].basis_size()))
}

fn unit_norm(mut v: Array1<f64>) -> Option<Array1<f64>> {
    let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if !(norm.is_finite() && norm > 0.0) {
        return None;
    }
    for x in v.iter_mut() {
        *x /= norm;
    }
    Some(v)
}

/// #2720 — the per-term attribution the thread has been missing.
///
/// This test PRINTS a table and asserts only what it can prove without deciding
/// the modelling question: that the data fit is flat along every constructed
/// direction (the likelihood-symmetry claim, re-measured here through the VALUE
/// function rather than through the construction's own least-squares residual),
/// and that at least one direction moves at least one penalty term by more than
/// the data fit (the demonstrandum: the orbit is not a posterior symmetry).
#[test]
fn chart_orbit_directional_derivative_splits_by_objective_term_2720() {
    let z = planted_circle_cloud();
    let registry = AnalyticPenaltyRegistry::new();
    let kinds: [(&str, usize); 4] = [
        ("periodic", 1),
        ("duchon", 1),
        ("linear", 1),
        ("euclidean", 1),
    ];
    let mut any_penalty_dominates = false;
    let mut worst_data_fit_slope = 0.0_f64;
    for (kind, latent_dim) in kinds {
        let mut term = seeded_term_of_kind(z.view(), kind, latent_dim);
        let rho = SaeManifoldRho::new(0.0, 0.0, vec![Array1::<f64>::zeros(latent_dim)]);
        let base = objective_terms(&term, z.view(), &rho, &registry).expect("base objective");
        let gauges = term.dense_step_gauge_vectors().expect("gauge vectors");
        let (active, basis_size) = active_rows_and_basis_size(&term, 0).expect("active rows");
        println!(
            "\n[2720-split] kind={kind} d={latent_dim} n={} p={} M={basis_size} n_active={active} \
             gauge_dirs={} objective={:.9e}",
            term.n_obs(),
            term.output_dim(),
            gauges.len(),
            base.total(),
        );
        println!(
            "[2720-split]   {:>4} {:>12} {:>12} {:>12} {:>12} {:>12} {:>12}",
            "dir", "data_fit", "sparsity", "smoothness", "ard", "barriers", "total"
        );
        for (index, gauge) in gauges.into_iter().enumerate() {
            let Some(direction) = unit_norm(gauge) else {
                continue;
            };
            // `h` is a relative step on a unit direction: small enough that the
            // second-order term is below the difference's own truncation error,
            // large enough that the difference is not cancellation noise.
            let h = 1.0e-5;
            let slope = directional_derivative_terms(
                &mut term,
                z.view(),
                &rho,
                &registry,
                &direction,
                h,
            )
            .expect("directional derivative");
            let barriers =
                slope.analytic + slope.repulsion + slope.amplitude_barrier + slope.separation_barrier;
            println!(
                "[2720-split]   {index:>4} {:>12.4e} {:>12.4e} {:>12.4e} {:>12.4e} {:>12.4e} {:>12.4e}",
                slope.data_fit,
                slope.assignment_sparsity,
                slope.smoothness,
                slope.ard,
                barriers,
                slope.total(),
            );
            worst_data_fit_slope = worst_data_fit_slope.max(slope.data_fit.abs());
            let penalty = slope.total() - slope.data_fit;
            if penalty.abs() > slope.data_fit.abs().max(1.0e-12) {
                any_penalty_dominates = true;
            }
        }
    }
    assert!(
        any_penalty_dominates,
        "no constructed chart-gauge direction moved the penalty block more than the data fit; \
         #2720's central claim (the orbit is a likelihood symmetry that the priors break) would \
         then be unreproducible on this fixture and the modelling question would be moot"
    );
    println!("[2720-split] worst |d data_fit| over all directions = {worst_data_fit_slope:.6e}");
}

/// The three families [`SaeManifoldTerm::gauge_quotient_basis`] concatenates,
/// counted, so a per-direction verdict can be attributed to the family that
/// produced it without re-deriving the concatenation order.
pub(crate) fn quotient_family_sizes(
    term: &SaeManifoldTerm,
    lambda_smooth: &[f64],
) -> Result<(usize, usize, usize), String> {
    Ok((
        term.dense_step_gauge_vectors()?.len(),
        term.joint_decoder_beta_null_directions(lambda_smooth)?.len(),
        term.decoder_channel_null_directions()?.len(),
    ))
}

/// #2720's ACCEPTANCE CRITERION, as an executable gate.
///
/// > the directional derivative of the **penalized** objective along every
/// > constructed orbit direction is at or below the convergence tolerance
///
/// denominated in the tolerance the accept path itself uses
/// (`SAE_MANIFOLD_INNER_GRAD_REL_TOL · inner_iterate_scale()`, the single source
/// of truth at `construction_quasi_laplace.rs:937`), and measured by central
/// difference of the value function rather than by the analytic gradient, so it
/// cannot pass by the gradient and the objective agreeing with each other while
/// both are wrong (#2714).
///
/// This is a property of the SPAN THE GATES REMOVE, not of the chart-gauge
/// construction: it reads `posterior_null_quotient_basis`, the single source of
/// truth for that span, so it fails whichever family contributes a live
/// direction.
#[test]
fn quotient_span_is_flat_for_the_penalized_objective_2720() {
    let z = planted_circle_cloud();
    let registry = AnalyticPenaltyRegistry::new();
    let kinds: [(&str, usize); 4] = [
        ("periodic", 1),
        ("duchon", 1),
        ("linear", 1),
        ("euclidean", 1),
    ];
    let mut violations: Vec<String> = Vec::new();
    let mut measured = 0usize;
    for (kind, latent_dim) in kinds {
        let mut term = seeded_term_of_kind(z.view(), kind, latent_dim);
        let rho = SaeManifoldRho::new(0.0, 0.0, vec![Array1::<f64>::zeros(latent_dim)]);
        let lambda_smooth = rho
            .lambda_smooth_vec()
            .expect("the fixture rho carries one smoothing block per atom");
        let tolerance = SAE_MANIFOLD_INNER_GRAD_REL_TOL * term.inner_iterate_scale();
        let (chart, beta_null, channel_null) =
            quotient_family_sizes(&term, &lambda_smooth).expect("family sizes");
        let basis = term
            .posterior_null_quotient_basis(&lambda_smooth)
            .expect("the seeded fixture has a well-defined quotient span");
        println!(
            "\n[2720-gate] kind={kind} tol={tolerance:.6e} span={} \
             (chart={chart}, beta_null={beta_null}, channel_null={channel_null})",
            basis.len(),
        );
        for (index, direction) in basis.into_iter().enumerate() {
            let slope = directional_derivative_terms(
                &mut term,
                z.view(),
                &rho,
                &registry,
                &direction,
                1.0e-5,
            )
            .expect("directional derivative");
            measured += 1;
            let total = slope.total().abs();
            println!(
                "[2720-gate]   dir {index:>3}  |d f| = {total:.6e}  ({:.3}x tol)  \
                 data_fit={:.3e} smooth={:.3e} ard={:.3e}",
                total / tolerance,
                slope.data_fit,
                slope.smoothness,
                slope.ard,
            );
            if total > tolerance {
                violations.push(format!(
                    "{kind} direction {index}: |d f| = {total:.6e} = {:.3}x the convergence \
                     tolerance {tolerance:.6e} (data_fit {:.3e}, smoothness {:.3e}, ard {:.3e})",
                    total / tolerance,
                    slope.data_fit,
                    slope.smoothness,
                    slope.ard,
                ));
            }
        }
    }
    assert!(
        measured > 0,
        "no quotient direction was measured on any fixture, so this gate certified nothing"
    );
    assert!(
        violations.is_empty(),
        "the span both inner convergence gates project out of the KKT residual carries LIVE \
         descent of the penalized objective. Every consumer that reads a small quotient as \
         `stationary` — the inner accept gate, the terminal polish's stationarity return, and \
         `SaeInstalledInnerKktAudit::certifies()` via `parameter_space.certifies()` — is reading \
         a point that is not stationary.\n  {}",
        violations.join("\n  "),
    );
}
