//! #2712 — the from-probes selected-inverse cluster on a SECOND deflating
//! fixture, and on the two trace channels the θ-adjoint gates do not touch.
//!
//! The gates in `tests_logdet_adjoint_780` run the θ-adjoint on the ordered
//! Beta–Bernoulli fixture. This module runs the whole cluster — the
//! reconstruction identity, the θ-adjoint, the ARD log-precision trace and the
//! assignment-strength trace — on the SOFTMAX two-atom fixture at its cold,
//! genuinely indefinite seed, where `factor_spectral_deflated_criterion_row`
//! (#1117) records a real `RowDeflationSpectrum`.
//!
//! The spectral branch is the one worth pinning: the correction there is
//! `Σ_{a,b} W[a,b]·M[a,b]·(1 − F[a,b])` with `W = Uᵀ inv_vv U`, so it reads every
//! OFF-DIAGONAL entry of the row's selected-inverse block through the
//! Daleckii–Krein rotation coefficients `(λₘ − 1)/(λₘ − λᵢ)` that couple the kept
//! and deflated subspaces. A reconstruction that recovered only the diagonal
//! would pass a diagonal comparison and fail here. The gauge-only fallback
//! (`spectrum = None`, the two-sided projection `Σᵢ vᵢᵀ D vᵢ`) cannot make that
//! distinction, which is why the premise below asserts a recorded spectrum
//! rather than merely a deflated direction.

use super::tests::small_two_atom_periodic_term;
use super::tests_logdet_adjoint_780::deflation_blind_cache;
use super::*;

/// The cold, genuinely indefinite two-atom state, with the fixture premise
/// asserted rather than assumed: at least one row must carry a recorded
/// SPECTRAL deflation, not merely a gauge direction.
fn spectrally_deflated_cold_state() -> (SaeManifoldTerm, SaeManifoldRho, ArrowFactorCache) {
    let (mut term, target, rho) = small_two_atom_periodic_term();
    let options = ArrowSolveOptions::direct().with_positive_definite_evidence();
    let system = term
        .assemble_arrow_schur(target.view(), &rho, None)
        .expect("cold arrow assembly");
    let (_delta_t, _delta_beta, cache) =
        solve_arrow_newton_step_with_options(&system, 0.0, 0.0, &options)
            .expect("the cold undamped factor is spectrally conditioned (#1117), not refused");
    let spectral_rows = cache
        .deflation_row_spectra
        .iter()
        .filter(|spectrum| spectrum.is_some())
        .count();
    assert!(
        spectral_rows > 0,
        "#2712 premise: this gate needs a row whose deflation carries a RECORDED \
         SPECTRUM (the Daleckii–Krein branch that reads the off-diagonal block). \
         Got {spectral_rows} spectral row(s) and {} gauge direction(s) — the fixture \
         no longer reaches the indefinite cold seed and the gate would be testing \
         the projection fallback instead.",
        cache.gauge_deflated_directions
    );
    assert!(
        cache.k > 0,
        "#2712 premise: the fixture must carry a border, or `S⁻¹` is not in play at all"
    );
    (term, rho, cache)
}

/// The exact `(z_j, S⁻¹ z_j)` bundle at full-basis probes `√k·e_j`, where the
/// Hutchinson outer products are algebraically exact.
fn full_basis_bundle(cache: &ArrowFactorCache) -> (Vec<Array1<f64>>, Vec<Array1<f64>>) {
    let k = cache.k;
    let sqrt_k = (k as f64).sqrt();
    let probes: Vec<Array1<f64>> = (0..k)
        .map(|j| {
            let mut v = Array1::<f64>::zeros(k);
            v[j] = sqrt_k;
            v
        })
        .collect();
    let sinv: Vec<Array1<f64>> = probes
        .iter()
        .map(|v| {
            cache
                .schur_inverse_apply(v.view())
                .expect("schur_inverse_apply")
        })
        .collect();
    (probes, sinv)
}

/// The reconstruction identity on a SPECTRALLY deflated row, including the
/// off-diagonal entries the Daleckii–Krein rotation term reads.
#[test]
fn row_selected_inverse_from_probes_matches_dense_on_spectrally_deflated_rows_2712() {
    let (_term, _rho, cache) = spectrally_deflated_cold_state();
    let (probes, sinv) = full_basis_bundle(&cache);
    let solver = DeflatedArrowSolver::plain(&cache);
    let beta_inv = solver.beta_inv().expect("beta_inv");

    let mut rows = 0usize;
    let mut worst_diagonal = 0.0_f64;
    let mut worst_off_diagonal = 0.0_f64;
    let mut worst_border = 0.0_f64;
    let mut off_diagonal_mass = 0.0_f64;
    let mut block_scale = 0.0_f64;
    for row in 0..cache.row_dims.len() {
        if cache
            .deflation_row_spectra
            .get(row)
            .and_then(Option::as_ref)
            .is_none()
        {
            continue;
        }
        rows += 1;
        let q = cache.row_dims[row];
        let (dense_vv, dense_vbeta) = solver
            .selected_inverse_row_blocks(row, &beta_inv)
            .expect("dense selected inverse row blocks");
        let (probe_vv, probe_vbeta) = row_selected_inverse_from_probes(
            &cache,
            row,
            &probes,
            &sinv,
            true,
            "#2712 spectral reconstruction gate",
        )
        .expect("from-probes selected inverse row blocks");
        for a in 0..q {
            for b in 0..q {
                let err = (dense_vv[[a, b]] - probe_vv[[a, b]]).abs();
                if a == b {
                    worst_diagonal = worst_diagonal.max(err);
                } else {
                    worst_off_diagonal = worst_off_diagonal.max(err);
                    off_diagonal_mass = off_diagonal_mass.max(dense_vv[[a, b]].abs());
                }
                block_scale = block_scale.max(dense_vv[[a, b]].abs());
            }
        }
        for (d, p) in dense_vbeta.iter().zip(probe_vbeta.iter()) {
            worst_border = worst_border.max((d - p).abs());
            block_scale = block_scale.max(d.abs());
        }
    }
    eprintln!(
        "#2712 spectral reconstruction: {rows} spectrally deflated row(s); \
         worst diagonal error {worst_diagonal:.3e}, worst off-diagonal error \
         {worst_off_diagonal:.3e} (off-diagonal magnitude {off_diagonal_mass:.3e}), \
         worst t–β error {worst_border:.3e}, block magnitude {block_scale:.3e}"
    );
    // A reconstruction that only got the DIAGONAL right would pass a
    // diagonal-only comparison; the off-diagonal mass is what makes the
    // off-diagonal assertion non-vacuous.
    assert!(
        rows > 0,
        "the premise promised a spectrally deflated row and the loop found none"
    );
    assert!(
        off_diagonal_mass > 1.0e-6 * (1.0 + block_scale),
        "the deflated selected-inverse block must carry real off-diagonal mass for \
         the Daleckii–Krein rotation term to be under test; got \
         {off_diagonal_mass:.3e} against block magnitude {block_scale:.3e}"
    );
    // RELATIVE: a kept near-null eigendirection legitimately inflates `inv_vv`.
    let tol = 1.0e-11 * (1.0 + block_scale);
    assert!(
        worst_diagonal <= tol && worst_off_diagonal <= tol && worst_border <= tol,
        "from-probes reconstruction must equal the dense selected inverse on a \
         spectrally deflated row: diag {worst_diagonal:.3e}, off-diag \
         {worst_off_diagonal:.3e}, t–β {worst_border:.3e} against tolerance {tol:.3e}"
    );
}

/// Separation + parity for the θ-adjoint on the SPECTRAL branch.
#[test]
fn logdet_theta_adjoint_from_probes_matches_dense_on_spectrally_deflated_rows_2712() {
    let (term, rho, cache) = spectrally_deflated_cold_state();
    let (probes, sinv) = full_basis_bundle(&cache);
    let solver = DeflatedArrowSolver::plain(&cache);
    let dense = term
        .logdet_theta_adjoint(&rho, &cache, &solver)
        .expect("dense theta-adjoint");

    let blind_cache = deflation_blind_cache(&cache);
    let blind_solver = DeflatedArrowSolver::plain(&blind_cache);
    let blind = term
        .logdet_theta_adjoint(&rho, &blind_cache, &blind_solver)
        .expect("deflation-blind dense theta-adjoint");
    let separation = dense
        .t
        .iter()
        .zip(blind.t.iter())
        .chain(dense.beta.iter().zip(blind.beta.iter()))
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);

    let from_probes = term
        .logdet_theta_adjoint_from_probes(&rho, &cache, &probes, &sinv, false)
        .expect("the from-probes theta-adjoint must PRICE a deflated cache, not refuse it");
    let parity = dense
        .t
        .iter()
        .zip(from_probes.t.iter())
        .chain(dense.beta.iter().zip(from_probes.beta.iter()))
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    eprintln!(
        "#2712 spectral θ-adjoint: separation from the deflation-blind operator \
         {separation:.6e}, from-probes parity error {parity:.6e}"
    );
    assert!(
        separation > 1.0e-6 && separation > 1.0e4 * parity,
        "the deflated and deflation-blind θ-adjoints must SEPARATE before parity is \
         evidence: separation {separation:.6e} against parity error {parity:.6e}"
    );
    for (i, (d, m)) in dense.t.iter().zip(from_probes.t.iter()).enumerate() {
        assert!(
            (d - m).abs() <= 1.0e-8 * (1.0 + d.abs()),
            "gamma_t[{i}]: dense={d:.10e}, from_probes={m:.10e}"
        );
    }
    for (i, (d, m)) in dense.beta.iter().zip(from_probes.beta.iter()).enumerate() {
        assert!(
            (d - m).abs() <= 1.0e-8 * (1.0 + d.abs()),
            "gamma_beta[{i}]: dense={d:.10e}, from_probes={m:.10e}"
        );
    }
}

/// Separation + parity for the ARD log-precision Hessian trace on the SPECTRAL
/// branch. The dense sibling builds its `inv_vv` with per-column full-system
/// solves; the from-probes route reconstructs it from the border bundle.
#[test]
fn ard_log_precision_hessian_trace_from_probes_matches_dense_on_deflated_rows_2712() {
    let (term, rho, cache) = spectrally_deflated_cold_state();
    let (probes, sinv) = full_basis_bundle(&cache);
    let solver = DeflatedArrowSolver::plain(&cache);
    let dense = term
        .ard_log_precision_hessian_trace(&rho, &cache, &solver)
        .expect("dense ARD trace");

    let blind_cache = deflation_blind_cache(&cache);
    let blind_solver = DeflatedArrowSolver::plain(&blind_cache);
    let blind = term
        .ard_log_precision_hessian_trace(&rho, &blind_cache, &blind_solver)
        .expect("deflation-blind dense ARD trace");

    let from_probes = term
        .ard_log_precision_hessian_trace_from_probes(&rho, &cache, &probes, &sinv)
        .expect("the from-probes ARD trace must PRICE a deflated cache, not refuse it");

    let mut separation = 0.0_f64;
    let mut parity = 0.0_f64;
    let mut dense_scale = 0.0_f64;
    let mut entries = 0usize;
    for ((d, b), m) in dense.iter().zip(blind.iter()).zip(from_probes.iter()) {
        assert_eq!(d.len(), m.len());
        assert_eq!(d.len(), b.len());
        for ((dv, bv), mv) in d.iter().zip(b.iter()).zip(m.iter()) {
            separation = separation.max((dv - bv).abs());
            parity = parity.max((dv - mv).abs());
            dense_scale = dense_scale.max(dv.abs());
            entries += 1;
        }
    }
    eprintln!(
        "#2712 spectral ARD trace over {entries} (atom, axis) entries: magnitude \
         {dense_scale:.6e}, separation {separation:.6e}, from-probes parity error \
         {parity:.6e}"
    );
    assert!(
        entries > 0,
        "the fixture must carry at least one live ARD axis for this gate to mean anything"
    );
    assert!(
        separation > 1.0e4 * parity && separation > 0.0,
        "the deflated and deflation-blind ARD traces must SEPARATE before parity is \
         evidence: separation {separation:.6e} against parity error {parity:.6e}"
    );
    assert!(
        parity <= 1.0e-11 * (1.0 + dense_scale),
        "from-probes ARD trace must equal the dense trace on a deflated cache: \
         {parity:.6e} against trace magnitude {dense_scale:.6e}"
    );
}

/// Separation + parity for the assignment-strength Hessian trace on the SPECTRAL
/// branch.
#[test]
fn assignment_log_strength_hessian_trace_from_probes_matches_dense_on_deflated_rows_2712() {
    let (term, rho, cache) = spectrally_deflated_cold_state();
    let (probes, sinv) = full_basis_bundle(&cache);
    let solver = DeflatedArrowSolver::plain(&cache);
    let dense = term
        .assignment_log_strength_hessian_trace(&rho, &cache, &solver)
        .expect("dense assignment-strength trace");

    let blind_cache = deflation_blind_cache(&cache);
    let blind_solver = DeflatedArrowSolver::plain(&blind_cache);
    let blind = term
        .assignment_log_strength_hessian_trace(&rho, &blind_cache, &blind_solver)
        .expect("deflation-blind dense assignment-strength trace");

    let from_probes = term
        .assignment_log_strength_hessian_trace_from_probes(&rho, &cache, &probes, &sinv)
        .expect("the from-probes assignment trace must PRICE a deflated cache, not refuse it");

    let separation = (dense - blind).abs();
    let parity = (dense - from_probes).abs();
    eprintln!(
        "#2712 spectral assignment trace: dense {dense:.10e}, deflation-blind \
         {blind:.10e} (separation {separation:.6e}), from-probes {from_probes:.10e} \
         (parity error {parity:.6e})"
    );
    assert!(
        separation > 1.0e4 * parity && separation > 0.0,
        "the deflated and deflation-blind assignment traces must SEPARATE before \
         parity is evidence: separation {separation:.6e} against parity error \
         {parity:.6e}"
    );
    assert!(
        parity <= 1.0e-11 * (1.0 + dense.abs()),
        "from-probes assignment trace must equal the dense trace on a deflated \
         cache: {parity:.6e}"
    );
}

/// #2712 END-TO-END: the COMPLETE analytic outer ρ-gradient must be the same
/// vector whether its selected-inverse channels go through the dense solver or
/// through the probe bundle — on a DEFLATED fit.
///
/// This is the statement the routing actually needs, and the reason fixing only
/// the θ-adjoint would not have been enough.
/// `analytic_outer_rho_gradient_components_with_bundle` converts the smoothness
/// EDF, the ARD Hessian trace and the θ-adjoint together as ONE all-or-nothing
/// cluster (invariant #1); before this change every one of them refused a
/// deflated cache, so the whole wide-`p` lane routed any deflated fit to a dense
/// channel it cannot afford at massive `K`.
///
/// `matrix_free_system = None` keeps the single-adjoint IFT solve dense on BOTH
/// sides, so the only difference between the two gradients is the from-probes
/// channels — the object under test — rather than the CG adjoint.
#[test]
fn complete_outer_gradient_from_probes_matches_dense_on_deflated_rows_2712() {
    let (mut term, target, rho) = super::tests::gamma_fd_tiny_fixture();
    term.assignment.mode = AssignmentMode::ordered_beta_bernoulli(0.7, 0.9, true);
    let anchor = super::tests_recovery_split_780::certified_fd_anchor(
        "#2712 complete-gradient deflated parity",
        &target,
        super::tests_recovery_split_780::FdAnchorRegime::deflated(),
        super::tests_recovery_split_780::rho_ladder_family(
            &term,
            super::tests_recovery_split_780::sparse_lift_ladder(
                &rho,
                &[2.4, 1.8, 1.3, 0.9, 0.5, 0.2, 0.0, -0.3, -0.6, -1.0],
            ),
            5,
        ),
    );
    let term = anchor.term;
    let rho = anchor.rho;
    let cache = anchor.cache;
    let deflated_rows = cache
        .deflated_row_directions
        .iter()
        .filter(|d| !d.is_empty())
        .count();
    assert!(deflated_rows > 0, "the certified anchor promised deflation");

    let loss = term
        .loss(target.view(), &rho)
        .expect("loss at the frozen anchor");
    let solver = DeflatedArrowSolver::plain(&cache);
    let dense = term
        .analytic_outer_rho_gradient_components(target.view(), &rho, &loss, &cache, &solver)
        .expect("dense complete outer gradient")
        .gradient();

    let (probes, sinv) = full_basis_bundle(&cache);
    let bundled = term
        .analytic_outer_rho_gradient_components_with_bundle(
            target.view(),
            &rho,
            &loss,
            &cache,
            &solver,
            Some((&probes, &sinv)),
            None,
        )
        .expect(
            "the from-probes cluster must PRICE a deflated fit; before #2712 every \
             channel in it refused here",
        )
        .gradient();

    let blind_cache = deflation_blind_cache(&cache);
    let blind_solver = DeflatedArrowSolver::plain(&blind_cache);
    let blind = term
        .analytic_outer_rho_gradient_components(
            target.view(),
            &rho,
            &loss,
            &blind_cache,
            &blind_solver,
        )
        .expect("deflation-blind dense complete outer gradient")
        .gradient();

    assert_eq!(dense.len(), bundled.len());
    assert_eq!(dense.len(), blind.len());
    let mut scale = 0.0_f64;
    let mut parity = 0.0_f64;
    let mut separation = 0.0_f64;
    for i in 0..dense.len() {
        assert!(
            dense[i].is_finite() && bundled[i].is_finite() && blind[i].is_finite(),
            "gradient coordinate {i} must be finite (dense={}, bundled={}, blind={})",
            dense[i],
            bundled[i],
            blind[i]
        );
        scale = scale.max(dense[i].abs());
        parity = parity.max((dense[i] - bundled[i]).abs());
        separation = separation.max((dense[i] - blind[i]).abs());
    }
    eprintln!(
        "#2712 complete outer gradient over {} coordinate(s), {deflated_rows} deflated \
         row(s): ‖g‖∞ = {scale:.6e}, dense↔from-probes {parity:.6e}, \
         dense↔deflation-blind {separation:.6e}",
        dense.len()
    );
    assert!(
        scale > 1.0e-10 && scale.is_finite(),
        "a zero gradient would make the parity check vacuous; ‖g‖∞ = {scale:.6e}"
    );
    assert!(
        separation > 1.0e4 * parity && separation > 1.0e-9,
        "the deflated and deflation-blind complete gradients must SEPARATE before \
         parity is evidence: separation {separation:.6e} against parity error \
         {parity:.6e}"
    );
    assert!(
        parity <= 1.0e-8 * (1.0 + scale),
        "the complete outer ρ-gradient must not depend on whether its \
         selected-inverse channels went dense or from-probes, on a DEFLATED fit: \
         {parity:.6e} against ‖g‖∞ = {scale:.6e}"
    );
}
