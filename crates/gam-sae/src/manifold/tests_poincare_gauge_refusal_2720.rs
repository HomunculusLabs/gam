//! Poincaré gauge-orbit diagnosis: WHY does the inner solve refuse where
//! periodic/duchon/linear solve?
//!
//! ## Background
//!
//! The #2720 geometry sweep (PR #2772) found the poincaré atom's inner solve
//! REFUSES on the planted-circle fixture: after 2680 inner iterations,
//! `gauge_share = 0.9456` — 94.6% of the KKT gradient lies INSIDE the
//! chart-gauge span, and the solver declines to rank an off-optimum
//! criterion. Refusal is robust to a 10× budget.
//!
//! The gauge construction treats the poincaré tangent patch as sharing the
//! Euclidean patch's translation + scale orbit (`dense_step_gauge_vectors`,
//! fit_drivers.rs): "the hyperbolic structure lives in the penalty, not the
//! gauge." Two stories explain the refusal:
//!
//! * **Story 1 (construction defect):** the claimed gauge vectors are NOT
//!   likelihood-null on a poincaré atom — the quotient projection discards
//!   gradient the data term genuinely cares about, and the solver chases it.
//! * **Story 2 (penalty magnitude):** the vectors ARE null, but the
//!   conformal-Dirichlet penalty's gradient along the orbit is enormous, so
//!   the posterior gradient never meets tolerance.
//!
//! ## Probes
//!
//! **A — seed-state nullness (no solve).** At the SEEDED state, project the
//! near-null-penalty gradient onto the gauge basis. If |g_nullᵀv| is at
//! numerical noise relative to ‖g‖ for periodic but NOT poincaré, the vectors
//! themselves are non-null on curved charts — Story 1. Comparing kinds at the
//! seed isolates the geometry: same fixture, same seed path, same instrument,
//! zero iterations of solver drift.
//!
//! **B — native target.** Re-run the measurement with a target that is a
//! poincaré atom's own reconstruction at its own seed (data_fit ≈ 0 by
//! construction). If the solve STILL refuses on a self-consistent target,
//! the phenomenon is not fixture stress — it is native to the atom kind.
//!
//! Marked `#[ignore]` — a manual diagnostic reproducer, same convention as
//! the #2770 baseline and the #2772 sweep.
//! Run: `cargo test -p gam-sae poincare_gauge -- --ignored --nocapture`

#![cfg(test)]
use super::*;

/// One seeded term of the requested kind on the Fixture-B circle cloud
/// (`tests_gauge_frame_roundtrip_2720`): n=42, p=48, same LCG.
fn seeded_circle_term(kind: &str) -> (SaeManifoldTerm, SaeManifoldRho, Array2<f64>) {
    let n = 42usize;
    let p = 48usize;
    let mut state = 0x2468_ace0_1357_9bdfu64;
    let mut next_unit = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / (1u64 << 53) as f64
    };
    let b0: Vec<f64> = (0..p).map(|j| (j as f64 * 0.7).sin() * 0.9).collect();
    let b1: Vec<f64> = (0..p).map(|j| (j as f64 * 1.3 + 0.9).cos() * 0.9).collect();
    let two_pi = std::f64::consts::TAU;
    let mut z = Array2::<f64>::zeros((n, p));
    for i in 0..n {
        let theta = two_pi * next_unit();
        for j in 0..p {
            let noise = 0.01 * (2.0 * next_unit() - 1.0);
            z[[i, j]] = theta.cos() * b0[j] + theta.sin() * b1[j] + noise;
        }
    }
    let minimal = build_sae_minimal_seed(SaeMinimalSeedRequest {
        target: z.view(),
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
    .unwrap_or_else(|e| panic!("[poincare-gauge] minimal seed failed for {kind}: {e}"));
    let registry = AnalyticPenaltyRegistry::new();
    let seed = build_sae_fit_seed(SaeFitSeedRequest {
        target: z.view(),
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
    .unwrap_or_else(|e| panic!("[poincare-gauge] fit seed failed for {kind}: {e}"));
    let mut rho = seed.initial_rho;
    // ARD-saddle settings, matching the sweep (#2772).
    rho.log_lambda_sparse = -0.5;
    for value in rho.log_lambda_smooth.iter_mut() {
        *value = -1.0;
    }
    for axis in rho.log_ard.iter_mut() {
        for value in axis.iter_mut() {
            *value = -0.5;
        }
    }
    (seed.base_term, rho, z)
}

/// Near-null rho: every penalty channel collapsed to `exp(-20)`, layout kept.
fn near_null_rho(rho: &SaeManifoldRho) -> SaeManifoldRho {
    let mut null = rho.clone();
    null.log_lambda_sparse = -20.0;
    for value in null.log_lambda_smooth.iter_mut() {
        *value = -20.0;
    }
    for axis in null.log_ard.iter_mut() {
        for value in axis.iter_mut() {
            *value = -20.0;
        }
    }
    for value in null.log_lambda_block.iter_mut() {
        *value = -20.0;
    }
    null
}

/// Joint gradient + gauge basis at the CURRENT term state, no solve.
fn state_snapshot(
    term: &mut SaeManifoldTerm,
    target: ArrayView2<'_, f64>,
    rho: &SaeManifoldRho,
    label: &str,
) -> (Array1<f64>, Array1<f64>, Vec<Array1<f64>>, f64) {
    let sys = term
        .assemble_arrow_schur(target, rho, None)
        .unwrap_or_else(|e| panic!("[poincare-gauge] {label}: assemble failed: {e}"));
    let border_dim = sys.gb.len();
    let full_len: usize = sys.rows.iter().map(|r| r.gt.len()).sum::<usize>() + border_dim;
    let mut grad = Array1::<f64>::zeros(full_len);
    let mut row_offsets = vec![0usize];
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
    let null_sys = term
        .assemble_arrow_schur(target, &near_null_rho(rho), None)
        .unwrap_or_else(|e| panic!("[poincare-gauge] {label}: null assemble failed: {e}"));
    let mut null_grad = Array1::<f64>::zeros(full_len);
    let mut off2 = 0usize;
    for row in &null_sys.rows {
        for (i, &v) in row.gt.iter().enumerate() {
            null_grad[off2 + i] = v;
        }
        off2 += row.gt.len();
    }
    for (i, &v) in null_sys.gb.iter().enumerate() {
        null_grad[off2 + i] = v;
    }
    let basis = term
        .joint_chart_gauge_basis_for_arrow_layout(
            &row_offsets,
            border_dim,
            &format!("poincare-gauge {label}"),
        )
        .unwrap_or_else(|e| panic!("[poincare-gauge] {label}: gauge basis failed: {e}"));
    let grad_norm = grad.dot(&grad).sqrt();
    (grad, null_grad, basis, grad_norm)
}

/// Probe A: at the SEEDED state, is the gauge basis likelihood-null for each
/// kind? The instrument: worst |g_nullᵀv| / ‖g_null‖ over basis directions,
/// plus the same for the FULL gradient (penalty included) as context.
#[test]
#[ignore = "manual diagnostic; run with --ignored --nocapture"]
fn poincare_gauge_nullness_at_seed_2720() {
    for kind in ["periodic", "duchon", "poincare", "linear"] {
        let (mut term, rho, z) = seeded_circle_term(kind);
        for atom in term.atoms.iter_mut() {
            atom.deactivate_decoder_frame();
        }
        let (grad, null_grad, basis, grad_norm) = state_snapshot(&mut term, z.view(), &rho, kind);
        let null_norm = null_grad.dot(&null_grad).sqrt();
        let coords = term.assignment.coords[0].as_matrix();
        let max_abs_coord = coords.iter().fold(0.0f64, |m, &v| m.max(v.abs()));
        if basis.is_empty() {
            eprintln!(
                "[poincare-gauge] {kind:<9} seed: NO gauge directions (‖g‖={grad_norm:.3e}, \
                 ‖g_null‖={null_norm:.3e}, max|t|={max_abs_coord:.3})"
            );
            continue;
        }
        let mut worst_null_share = 0.0f64;
        let mut worst_full_share = 0.0f64;
        for v in basis.iter() {
            let n = null_grad.dot(v).abs() / null_norm.max(f64::MIN_POSITIVE);
            let f = grad.dot(v).abs() / grad_norm.max(f64::MIN_POSITIVE);
            worst_null_share = worst_null_share.max(n);
            worst_full_share = worst_full_share.max(f);
        }
        eprintln!(
            "[poincare-gauge] {kind:<9} seed: {} dirs | ‖g‖={:.3e} ‖g_null‖={:.3e} | \
             worst |g_nullᵀv|/‖g_null‖={:.3e}  |gᵀv|/‖g‖={:.3e} | max|t|={max_abs_coord:.3}",
            basis.len(),
            grad_norm,
            null_norm,
            worst_null_share,
            worst_full_share
        );
    }
}

/// Probe B: a target that is BY CONSTRUCTION this poincaré atom's own
/// reconstruction at its own seed state. data_fit ≈ 0 at t=0; if the solve
/// still refuses, the refusal is native to the kind, not fixture stress.
#[test]
#[ignore = "manual diagnostic; run with --ignored --nocapture"]
fn poincare_self_consistent_target_2720() {
    let (mut term, rho, _z) = seeded_circle_term("poincare");
    for atom in term.atoms.iter_mut() {
        atom.deactivate_decoder_frame();
    }
    // Reconstruct: atom basis at current coords through the decoder.
    let phi = term.atoms[0].basis_values.view();
    let decoder = term.atoms[0].decoder_coefficients();
    let native = phi.dot(decoder);
    let data_fit = ((&native - &_z).mapv(|v| v * v)).sum() / 2.0;
    eprintln!(
        "[poincare-gauge] native target: shape {:?}, seed-state data_fit vs circle={data_fit:.3e}",
        native.dim()
    );

    // First: measure at the seed against the NATIVE target (no solve).
    let (_grad, null_grad, basis, grad_norm) =
        state_snapshot(&mut term, native.view(), &rho, "poincare/native@seed");
    if !basis.is_empty() {
        let null_norm = null_grad.dot(&null_grad).sqrt();
        let worst = basis
            .iter()
            .map(|v| null_grad.dot(v).abs() / null_norm.max(f64::MIN_POSITIVE))
            .fold(0.0f64, f64::max);
        eprintln!(
            "[poincare-gauge] poincare  native@seed: {} dirs | ‖g‖={grad_norm:.3e} \
             ‖g_null‖={null_norm:.3e} | worst |g_nullᵀv|/‖g_null‖={worst:.3e}",
            basis.len()
        );
    } else {
        eprintln!("[poincare-gauge] poincare  native@seed: NO gauge directions");
    }

    // Then: try the solve on the native target.
    let budget = 40usize;
    match term.penalized_quasi_laplace_criterion_with_cache(
        native.view(),
        &rho,
        None,
        budget,
        0.4,
        1.0e-6,
        1.0e-6,
    ) {
        Ok((value, loss, _)) => eprintln!(
            "[poincare-gauge] poincare  native SOLVED: criterion={value:.6e} \
             (data_fit={:.3e}, smoothness={:.3e}, ard={:.3e}) — the refusal WAS fixture stress",
            loss.data_fit, loss.smoothness, loss.ard
        ),
        Err(e) => {
            let note = e.to_string();
            eprintln!(
                "[poincare-gauge] poincare  native REFUSED (budget={budget}): {}",
                note.chars().take(400).collect::<String>()
            );
            eprintln!("[poincare-gauge] → refusal is NATIVE to the poincare kind, not the fixture");
        }
    }

    // Control: same native-target experiment on periodic.
    let (mut pterm, prho, _pz) = seeded_circle_term("periodic");
    for atom in pterm.atoms.iter_mut() {
        atom.deactivate_decoder_frame();
    }
    let pphi = pterm.atoms[0].basis_values.view();
    let pdec = pterm.atoms[0].decoder_coefficients();
    let pnative = pphi.dot(pdec);
    match pterm.penalized_quasi_laplace_criterion_with_cache(
        pnative.view(),
        &prho,
        None,
        budget,
        0.4,
        1.0e-6,
        1.0e-6,
    ) {
        Ok((value, loss, _)) => eprintln!(
            "[poincare-gauge] periodic  native SOLVED: criterion={value:.6e} \
             (data_fit={:.3e})",
            loss.data_fit
        ),
        Err(e) => eprintln!(
            "[poincare-gauge] periodic  native REFUSED: {}",
            e.to_string().chars().take(300).collect::<String>()
        ),
    }
}
