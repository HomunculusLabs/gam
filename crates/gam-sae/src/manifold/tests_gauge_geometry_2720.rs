//! #2720 geometry extension: is the chart-gauge orbit-derivative violation
//! Periodic-specific or geometry-general — and does a decoder frame change it?
//!
//! ## Background
//!
//! The issue's "Not established" section records that every #2720 derivative
//! measurement — the original two fixtures, the #2770 re-measurement, and the
//! `b8f0c4d97` Fixture-B round trip — ran on `Periodic` atoms only, and (except
//! for Fixture B's reconstruction-side check) with `decoder_frame = None`.
//! Whether the orbit-tolerance violation survives on flat (`Duchon`) patches,
//! hyperbolic (`Poincare`) patches, or framed atoms is unmeasured.
//!
//! This diagnostic extends the #2770 measurement to those configurations so the
//! modelling fix is designed against the right generality: if the violation is
//! Periodic-specific the fix can live in the periodic gauge construction; if it
//! is geometry-general it must live at the gauge/penalty interface.
//!
//! ## Measured (at `9b9973f`, ARD-saddle rho, 40-outer-iteration budget)
//!
//! | cell | max \|gᵀv\|/tol | worst-dir near-null share (e⁻²⁰) | verdict |
//! |---|---|---|---|
//! | `periodic`/unframed | 2.33× | 3.4e-9 | above |
//! | `periodic`/framed | 2.33× | 3.4e-9 | above (verdict unchanged) |
//! | `duchon`/unframed | 3.63× | 4.0e-7 | **above — flat patches violate too** |
//! | `duchon`/framed | 3.63× | 4.0e-7 | above (verdict unchanged) |
//! | `poincare`/unframed | — | — | refused at budget 40, still refused at 400 |
//! | `poincare`/framed | — | — | refused at budget 40 (not retried at 400) |
//! | `linear`/unframed | 0.13× | 2.0e-9 | below this tolerance |
//! | `linear`/framed | 0.13× | 2.0e-9 | below this tolerance |
//!
//! Conclusions the modelling fix can rely on (scoped to this instrument):
//!
//! * On the shared circle-cloud fixture, the instrument detects a
//!   gauge-direction posterior gradient above solver tolerance for `periodic`
//!   and `duchon`, below tolerance for `linear`, and obtains no `poincare`
//!   measurement because the solve refused (robustly — re-attempted at a
//!   10× budget, same refusal). The detection is **not confined to the
//!   periodic kind** — a flat patch exhibits it too — so a fix inside the
//!   periodic gauge construction cannot be sufficient.
//! * The near-null-penalty control shows the violating projection is carried
//!   by the penalty block, not the data term: at the same state, with
//!   penalties collapsed to `exp(-20)`, the worst direction's projection
//!   drops to `3.4e-9` (periodic) / `4.0e-7` (duchon) of its full-penalty
//!   value, and EVERY direction's near-null projection sits below
//!   `1.2e-5× tol` (asserted). The deep floor `exp(-26)` discriminates the
//!   two readings: periodic's share scales with the floor (→ residual
//!   penalty), duchon/linear's are floor-independent roundoff (→ noise, not
//!   likelihood). This is the likelihood-vs-posterior split #2720 claims,
//!   verified independently on each measured geometry.
//! * A decoder frame has no detectable effect on the verdicts on this fixture
//!   (identical max ratios; projections agree to ~3e-4 relative on duchon,
//!   machine-exact on periodic/linear), consistent with Fixture B's exact
//!   reconstruction-side round trip (`b8f0c4d97`).
//!
//! ## Scope and caveats
//!
//! * All cells share one target (the planted circle cloud), which is native
//!   geometry for the periodic atom only: the duchon/linear fits sit at
//!   `data_fit ≈ 130` against periodic's `0.056`. The claim supported is
//!   "a non-periodic kind can violate on this fixture family", not a
//!   magnitude ranking across geometries — a target native to each geometry
//!   would be needed for that.
//! * `poincare`'s refusal is **budget-conditioned, not a geometry
//!   measurement**: the solver reports `gauge_share = 0.9456` at the refusing
//!   iterate (94.6% of the KKT gradient inside the chart-gauge span), which
//!   is refusal telemetry from a nonstationary state, not an orbit
//!   derivative. An escalated-budget arm re-attempts it; if it still
//!   refuses, that is reported, not interpreted.
//! * The measurement is a derivative **conditional on fixed ρ** (the
//!   ARD-saddle penalty state), not a statement about the joint posterior
//!   over smoothing parameters: ρ is overridden before the solve, and both
//!   the criterion and the gradient are assembled at that same ρ.
//! * "Below tolerance" for `linear` is a statement about this tolerance
//!   (`SAE_MANIFOLD_INNER_GRAD_REL_TOL · iterate_scale`), not a symmetry
//!   proof.
//!
//! ## Refusal policy
//!
//! Unlike the #2770 baseline (a single known-solvable cell, which `.expect`s
//! its solve), this sweep crosses geometries whose solves may legitimately
//! refuse. Convergence refusal is therefore a REPORTED outcome
//! ([`CellOutcome::Refused`]), while harness failures (seed construction,
//! assembly, frame activation, non-finite values) panic with cell context —
//! a refusal is a measurement about the geometry, a broken instrument is a
//! bug in this file.
//!
//! Marked `#[ignore]` like the #2770 baseline: it runs inner solves and
//! reports measurements rather than asserting a regression invariant.
//! Run with: `cargo test -p gam-sae gauge_geometry -- --ignored --nocapture`
//!
//! ## Method
//!
//! For each atom kind in {`periodic` (control), `duchon`, `poincare`, `linear`}
//! and each frame state in {inactive, active}:
//!
//! 1. Build a seeded term via the minimal-seed path (the same constructor
//!    Fixture B uses), with the atom basis replaced by the kind under test.
//! 2. Override ρ to the ARD-saddle settings BEFORE the solve (the same
//!    settings as the #2770 baseline), so priors are demonstrably active and
//!    the criterion and gradient see one penalty state.
//! 3. Run `penalized_quasi_laplace_criterion_with_cache` to the inner solve
//!    state (40 outer iterations; `poincare` additionally re-attempts at 400).
//! 4. Assemble the Arrow-Schur system, extract the joint KKT gradient.
//! 5. Project onto every chart-gauge direction; report `|gᵀv|/tol` where
//!    `tol = SAE_MANIFOLD_INNER_GRAD_REL_TOL · iterate_scale` — the same gate
//!    as the baseline and the issue's acceptance criteria. Gauge vectors are
//!    unit-normalized by the construction; this test asserts it.
//! 6. Null arm: re-assemble at the same state with penalties collapsed to
//!    `exp(-20)` and project the same directions, so the likelihood-only
//!    share of each violation is visible.

#![cfg(test)]
use super::*;
use crate::manifold::tests_gauge_frame_roundtrip_2720::planted_circle_cloud;

/// One measured cell: the full-penalty sweep and the near-null-penalty
/// controls at the same state.
#[derive(Debug, Clone, Copy)]
struct CellMeasurement {
    max_ratio: f64,
    directions: usize,
    tolerance: f64,
    /// For the direction with the worst total ratio: `|g_nullᵀv| / |gᵀv|` at
    /// penalty floor `exp(-20)` — the near-null share of THAT direction.
    null_share_worst_dir: f64,
    /// Same, at floor `exp(-26)`. If the near-null projection is residual
    /// penalty floor (not likelihood leakage) it shrinks by ≈ e⁻⁶ here.
    null_share_worst_dir_deep: f64,
    /// Worst `|g_nullᵀv| / tol` over ALL directions at floor `exp(-20)` —
    /// asserted small per direction, so contamination cannot hide in a weak
    /// direction.
    max_null_over_tol: f64,
}

/// The outcome of one cell: either a measurement, or a convergence refusal
/// (reported, never interpreted as a symmetry statement). Harness failures
/// panic; they are not outcomes.
#[derive(Debug, Clone)]
enum CellOutcome {
    Measured(CellMeasurement),
    Refused { budget: usize, note: String },
}

/// Build a seeded one-atom term of the requested basis kind on the planted
/// circle cloud — the identical seed path Fixture B (`b8f0c4d97`) uses, with
/// only the `atom_basis` token changed. Returns the term and its seed rho.
fn seeded_term_of_kind(
    kind: &str,
    target: ArrayView2<'_, f64>,
) -> (SaeManifoldTerm, SaeManifoldRho) {
    let minimal = build_sae_minimal_seed(SaeMinimalSeedRequest {
        target,
        atom_basis: vec![kind.to_string()],
        atom_dim: vec![1],
        assignment_kind: SaeFitAssignmentKind::Softmax,
        alpha: 1.0,
        tau: 1.0,
        threshold: 0.0,
        top_k: None,
        random_state: 45,
        initial_logits: None,
        initial_coords: None,
    })
    .unwrap_or_else(|e| panic!("[2720-geom] minimal seed failed for {kind}: {e}"));
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
    .unwrap_or_else(|e| panic!("[2720-geom] fit seed failed for {kind}: {e}"));
    (seed.base_term, seed.initial_rho)
}

/// Apply the #2770 baseline's ard-saddle penalty settings to the seed rho.
/// Applied BEFORE the solve; the criterion and every gradient in this file
/// are assembled at this same rho (fixed-ρ derivative — see module docs).
fn ard_saddle_rho(mut rho: SaeManifoldRho) -> SaeManifoldRho {
    rho.log_lambda_sparse = -0.5;
    for value in rho.log_lambda_smooth.iter_mut() {
        *value = -1.0;
    }
    for axis in rho.log_ard.iter_mut() {
        for value in axis.iter_mut() {
            *value = -0.5;
        }
    }
    rho
}

/// Collapse every penalty channel to `exp(log_floor)` (≈ 2e-9 at −20) while
/// preserving the rho LAYOUT (per-atom smooth vector, ARD axis shapes,
/// block/kappa entries), so the near-null arm assembles a system of identical
/// shape at the same state. Two floors (`−20`, `−26`) distinguish residual
/// penalty floor from likelihood leakage: a residual-floor projection scales
/// with `exp(floor)`, leakage does not.
fn near_null_penalty_rho(rho: &SaeManifoldRho, log_floor: f64) -> SaeManifoldRho {
    let mut null = rho.clone();
    null.log_lambda_sparse = log_floor;
    for value in null.log_lambda_smooth.iter_mut() {
        *value = log_floor;
    }
    for axis in null.log_ard.iter_mut() {
        for value in axis.iter_mut() {
            *value = log_floor;
        }
    }
    for value in null.log_lambda_block.iter_mut() {
        *value = log_floor;
    }
    null
}

/// Joint KKT gradient of the Arrow-Schur system at the given penalty state,
/// with the row-offset layout the gauge basis expects.
fn joint_gradient(
    term: &mut SaeManifoldTerm,
    target: ArrayView2<'_, f64>,
    rho: &SaeManifoldRho,
    label: &str,
) -> (Array1<f64>, usize, usize, Vec<usize>) {
    let sys = term
        .assemble_arrow_schur(target, rho, None)
        .unwrap_or_else(|e| panic!("[2720-geom] {label}: assemble_arrow_schur failed: {e}"));
    let n_rows = sys.rows.len();
    let border_dim = sys.gb.len();
    let full_len: usize = sys.rows.iter().map(|r| r.gt.len()).sum::<usize>() + border_dim;
    let mut grad = Array1::<f64>::zeros(full_len);
    let mut row_offsets = Vec::with_capacity(n_rows + 1);
    row_offsets.push(0usize);
    let mut offset = 0usize;
    for row in &sys.rows {
        for (i, &v) in row.gt.iter().enumerate() {
            grad[offset + i] = v;
        }
        offset += row.gt.len();
        row_offsets.push(offset);
    }
    for (i, &v) in sys.gb.iter().enumerate() {
        grad[offset + i] = v;
    }
    assert!(
        grad.iter().all(|v| v.is_finite()),
        "[2720-geom] {label}: KKT gradient contains non-finite entries"
    );
    (grad, n_rows, border_dim, row_offsets)
}

/// The #2770 measurement core with the null-penalty control: inner solve →
/// Arrow-Schur → joint gauge basis → `|gᵀv|/tol` per direction, plus the
/// likelihood-only share via a penalties-collapsed re-assembly at the same
/// state. Returns [`CellOutcome::Refused`] only for convergence refusal.
fn measure_orbit_projection_2720(
    term: &mut SaeManifoldTerm,
    target: ArrayView2<'_, f64>,
    rho: &SaeManifoldRho,
    budget: usize,
    label: &str,
) -> CellOutcome {
    let result = term.penalized_quasi_laplace_criterion_with_cache(
        target, rho, None, budget, 0.4, 1.0e-6, 1.0e-6,
    );
    let (criterion_value, loss, _cache) = match result {
        Ok(ok) => ok,
        Err(e) => {
            // Only the RECOGNIZED convergence-refusal is a Refused outcome;
            // anything else (NaN, malformed state, gate bugs) is an
            // instrument failure and must fail loudly with context.
            let note = e.to_string();
            assert!(
                note.contains("inner solve did not converge"),
                "[2720-geom] {label}: inner solve failed for an unrecognized reason \
                 (not the known convergence refusal): {note}"
            );
            // Validate the refusal telemetry rather than trusting it: a
            // gauge_share outside [0,1] or non-finite refusal norms mean the
            // refusal message itself is corrupt.
            if let Some(idx) = note.find("gauge_share=") {
                let tail = &note[idx + "gauge_share=".len()..];
                let token: String = tail
                    .chars()
                    .take_while(|c| c.is_ascii_digit() || *c == '.' || *c == '-' || *c == 'e')
                    .collect();
                if let Ok(share) = token.parse::<f64>() {
                    assert!(
                        share.is_finite() && (0.0..=1.0).contains(&share),
                        "[2720-geom] {label}: refusal telemetry corrupt — \
                         gauge_share={share} outside [0,1]"
                    );
                    eprintln!(
                        "[2720-geom] {label}: refusal telemetry: {note} \
                         (from a NONSTATIONARY iterate; not an orbit measurement)"
                    );
                }
            }
            return CellOutcome::Refused { budget, note };
        }
    };
    eprintln!(
        "[2720-geom] {label}: criterion={criterion_value:.6e} \
         (data_fit={:.3e}, sparsity={:.3e}, smoothness={:.3e}, ard={:.3e})",
        loss.data_fit, loss.assignment_sparsity, loss.smoothness, loss.ard
    );

    let (grad, n_rows, border_dim, row_offsets) = joint_gradient(term, target, rho, label);
    let grad_norm = grad.dot(&grad).sqrt();
    eprintln!("[2720-geom] {label}: n_rows={n_rows} border_dim={border_dim} ‖g‖={grad_norm:.6e}");

    let gauge_basis = term
        .joint_chart_gauge_basis_for_arrow_layout(
            &row_offsets,
            border_dim,
            &format!("2720-geom {label}"),
        )
        .unwrap_or_else(|e| panic!("[2720-geom] {label}: joint_chart_gauge_basis failed: {e}"));
    if gauge_basis.is_empty() {
        panic!(
            "[2720-geom] {label}: NO gauge directions at this state — the measurement \
             instrument is inapplicable here, which for these fixtures is a harness bug \
             (the baseline measured ≥1 direction on the same fixture family)"
        );
    }

    // The construction unit-normalizes (fit_drivers MGS); assert what the
    // ratio below relies on rather than trusting it silently.
    for (i, v) in gauge_basis.iter().enumerate() {
        let norm = v.dot(v).sqrt();
        assert!(
            (norm - 1.0).abs() < 1.0e-12,
            "[2720-geom] {label}: gauge vector v_{i} has norm {norm:.6e}, expected 1"
        );
    }

    let iterate_scale = term.inner_iterate_scale();
    let tolerance = SAE_MANIFOLD_INNER_GRAD_REL_TOL * iterate_scale;
    assert!(
        tolerance.is_finite() && tolerance > 0.0,
        "[2720-geom] {label}: tolerance is non-finite or non-positive"
    );
    // NOTE on the gate's norm: the solver's own stationarity gate is L2
    // (‖g‖ ≤ tol, construction_quasi_laplace.rs:931) with tol defined exactly
    // as above. With unit v, |gᵀv| ≤ ‖g‖₂, so any direction whose projection
    // exceeds tolerance provably violates the solver's L2 gate — no
    // componentwise ‖v‖₁ factor applies. Reported ratios are therefore
    // conservative: the full gradient norm is at least as large.

    // Per-direction bookkeeping: total ratio, and near-null projections at
    // two penalty floors so contamination cannot hide in a weak direction and
    // residual floor can be distinguished from likelihood leakage.
    let null_grad = {
        let null_rho = near_null_penalty_rho(rho, -20.0);
        let (g, _, _, _) = joint_gradient(term, target, &null_rho, label);
        g
    };
    let deep_grad = {
        let deep_rho = near_null_penalty_rho(rho, -26.0);
        let (g, _, _, _) = joint_gradient(term, target, &deep_rho, label);
        g
    };

    let mut max_ratio = 0.0f64;
    let mut worst_abs = 0.0f64;
    let mut worst_dir = 0usize;
    let mut max_null_over_tol = 0.0f64;
    for (i, v) in gauge_basis.iter().enumerate() {
        let proj = grad.dot(v);
        assert!(
            proj.is_finite(),
            "[2720-geom] {label}: v_{i} projection is non-finite"
        );
        let ratio = proj.abs() / tolerance;
        let null_proj = null_grad.dot(v).abs();
        let null_over_tol = null_proj / tolerance;
        max_null_over_tol = max_null_over_tol.max(null_over_tol);
        if proj.abs() > worst_abs {
            worst_abs = proj.abs();
            worst_dir = i;
        }
        max_ratio = max_ratio.max(ratio);
        eprintln!(
            "[2720-geom] {label}: v_{i}: |gᵀv| = {:.6e}  ({:.2}× tolerance)  \
             near-null e⁻²⁰: {:.3e}  ({:.2e}× tol)",
            proj.abs(),
            ratio,
            null_proj,
            null_over_tol
        );
    }

    // Per-direction near-null gate: EVERY direction's near-null projection
    // must sit far below tolerance, else the likelihood term itself carries
    // gauge content on this fixture (which would falsify #2720's premise).
    assert!(
        max_null_over_tol < 1.0e-3,
        "[2720-geom] {label}: near-null projection hit {:.3e}× tol on some direction — \
         the data term carries gauge content, so the violation is NOT purely prior-side",
        max_null_over_tol
    );

    let null_share_worst_dir = null_grad.dot(&gauge_basis[worst_dir]).abs() / worst_abs;
    let null_share_worst_dir_deep = deep_grad.dot(&gauge_basis[worst_dir]).abs() / worst_abs;
    eprintln!(
        "[2720-geom] {label}: max |gᵀv|/tol = {max_ratio:.2}×  over {} directions  \
         (tol={tolerance:.3e}; worst-dir near-null share: {:.2e} at e⁻²⁰, {:.2e} at e⁻²⁶)",
        gauge_basis.len(),
        null_share_worst_dir,
        null_share_worst_dir_deep
    );

    CellOutcome::Measured(CellMeasurement {
        max_ratio,
        directions: gauge_basis.len(),
        tolerance,
        null_share_worst_dir,
        null_share_worst_dir_deep,
        max_null_over_tol,
    })
}

/// One (kind, frame) cell of the geometry sweep.
fn run_cell_2720(kind: &str, framed: bool) -> CellOutcome {
    let label = format!("{kind}/{}", if framed { "framed" } else { "unframed" });
    let z = planted_circle_cloud();
    let (mut term, rho) = seeded_term_of_kind(kind, z.view());
    let rho = ard_saddle_rho(rho);

    if framed {
        let output_dim = term.output_dim();
        let mut activated = Vec::new();
        for (atom_idx, atom) in term.atoms.iter_mut().enumerate() {
            match atom.maybe_activate_decoder_frame() {
                Ok(Some(rank)) => activated.push((atom_idx, rank)),
                Ok(None) => {}
                Err(e) => panic!("[2720-geom] {label}: frame activation errored: {e}"),
            }
        }
        eprintln!("[2720-geom] {label}: activated_frames={activated:?} output_dim={output_dim}");
        assert!(
            !activated.is_empty(),
            "[2720-geom] {label}: no decoder frame activated, so the framed cell would \
             silently duplicate the unframed cell"
        );
        for (_atom_idx, rank) in activated {
            assert!(
                rank < term.output_dim(),
                "[2720-geom] {label}: rank-{rank} frame at output_dim {} is full rank; \
                 the round trip is trivially exact and measures nothing",
                term.output_dim()
            );
        }
    } else {
        for atom in term.atoms.iter_mut() {
            atom.deactivate_decoder_frame();
        }
    }

    measure_orbit_projection_2720(&mut term, z.view(), &rho, 40, &label)
}

/// #2720 geometry sweep: the orbit-derivative measurement across atom kinds
/// and frame states. See module docs. The periodic/unframed cell is the
/// instrument control.
#[test]
#[ignore = "manual diagnostic reproducer for #2720; run with --ignored --nocapture"]
fn chart_gauge_orbit_violation_across_geometries_2720() {
    let kinds = ["periodic", "duchon", "poincare", "linear"];

    eprintln!("[2720-geom] ════════ geometry × frame sweep ════════");
    let mut summary: Vec<(String, CellOutcome)> = Vec::new();
    for kind in kinds {
        for framed in [false, true] {
            let label = format!("{kind}/{}", if framed { "framed" } else { "unframed" });
            let cell = run_cell_2720(kind, framed);
            summary.push((label, cell));
        }
    }

    eprintln!("[2720-geom] ════════ summary ════════");
    eprintln!("[2720-geom] (tol gate = SAE_MANIFOLD_INNER_GRAD_REL_TOL · iterate_scale)");
    for (label, cell) in &summary {
        match cell {
            CellOutcome::Measured(m) => {
                let verdict = if m.max_ratio <= 1.0 {
                    "AT/BELOW tol"
                } else {
                    "ABOVE tol"
                };
                eprintln!(
                    "[2720-geom] {label:<20} max|gᵀv|/tol = {:6.2}×  ({} dirs, tol={:.3e}, \
                     near-null: {:.1e}/e⁻²⁰ {:.1e}/e⁻²⁶ worst-dir, {:.1e}× tol max)  {verdict}",
                    m.max_ratio,
                    m.directions,
                    m.tolerance,
                    m.null_share_worst_dir,
                    m.null_share_worst_dir_deep,
                    m.max_null_over_tol
                );
            }
            CellOutcome::Refused { budget, note } => eprintln!(
                "[2720-geom] {label:<20} REFUSED under budget={budget}: {}",
                note.chars().take(160).collect::<String>()
            ),
        }
    }

    // Instrument health: every cell must have been attempted, and every
    // non-poincare kind must have produced a MEASUREMENT in both frames — a
    // refusal there means the instrument broke, not the geometry refused.
    assert_eq!(
        summary.len(),
        8,
        "the sweep must attempt all 8 cells (4 kinds × 2 frame states)"
    );
    for (label, cell) in &summary {
        if !label.starts_with("poincare") {
            assert!(
                matches!(cell, CellOutcome::Measured(_)),
                "[2720-geom] {label}: expected a measurement — a refusal outside poincare \
                 is an instrument failure, not a geometry result"
            );
        }
    }
    let control = &summary[0];
    assert!(
        matches!(control.1, CellOutcome::Measured(_)),
        "[2720-geom] the periodic/unframed control produced no measurement; every \
         other cell is uninterpretable until the instrument control works"
    );

    // Poincare budget escalation: if the 40-budget arms refused, re-attempt at
    // 400 so the refusal is established as robust (or overturned).
    let poincare_refused = summary
        .iter()
        .filter(|(l, c)| l.starts_with("poincare") && matches!(c, CellOutcome::Refused { .. }))
        .count();
    if poincare_refused > 0 {
        eprintln!("[2720-geom] ════════ poincare escalated-budget retry (400) ════════");
        let z = planted_circle_cloud();
        let (mut term, rho) = seeded_term_of_kind("poincare", z.view());
        let rho = ard_saddle_rho(rho);
        for atom in term.atoms.iter_mut() {
            atom.deactivate_decoder_frame();
        }
        match measure_orbit_projection_2720(&mut term, z.view(), &rho, 400, "poincare/retry400") {
            CellOutcome::Measured(m) => eprintln!(
                "[2720-geom] poincare/retry400      max|gᵀv|/tol = {:6.2}× over {} dirs — \
                 the 40-budget refusal was budget-conditioned, measurement obtained",
                m.max_ratio, m.directions
            ),
            CellOutcome::Refused { budget, note } => eprintln!(
                "[2720-geom] poincare/retry400      STILL REFUSED under budget={budget}: {}",
                note.chars().take(160).collect::<String>()
            ),
        }
    }
}
