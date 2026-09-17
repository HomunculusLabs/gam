//! #2933 F07 lane — the reduced exact-A lane's pencil operands are the Ritz forms of the
//! dense route's evidence metric on the `A`-lift.
//!
//! `gam_solve::arrow_schur::exact_a_reduced_pencil_operands` rebuilds the evidence factor's
//! row pins inside gam-solve, off the majorizer blocks `B_tt = A_tt − ΔC_tt` of the exact-A
//! system. The dense route reads the same `Φ` through [`PreparedArrowMetric`], off the
//! majorizer's own evidence factor. On the lift `z(v) = (−T⁻¹A_tβ v, v)` through a row factor
//! slab `T`, the lane's `Y` must be `zᵀΦz` and its `s` must be `zᵀ(Φ − B_raw)z`, to
//! `dim·ε·‖·‖`, on a state whose evidence factor carries spectral row pins the lift reaches.
//! Gauge pins enter `Y` only, as `substituted_image` leaves them installed. The border carries
//! no spectral pin on either route here; a state with border pins is refused as a premise
//! until the border pins reach `substituted_image`, when this test extends to them.

use super::*;
use gam_solve::arrow_schur::{
    ArrowFactorSlab, ArrowUndampedFactors, BetaSchurSpectralConditioning,
    SPECTRAL_DEFLATION_REL_FLOOR, SurrogateLaneConfig, SurrogateLaneState,
    exact_a_reduced_pencil_operands, matrix_free_arrow_evidence_evaluation,
};
use ndarray::{Array1, Array2};

/// The #2712 weak-phase periodic state: flat decoders, and a first atom at the phase where its
/// periodic curvature sits under the spectral floor, so the majorizer's row blocks deflate.
fn weak_phase_state() -> (SaeManifoldTerm, Array2<f64>, SaeManifoldRho) {
    let (mut term, target, rho) = crate::manifold::tests::small_two_atom_periodic_term();
    let cosine = SPECTRAL_DEFLATION_REL_FLOOR / std::f64::consts::LN_2
        * SPECTRAL_DEFLATION_REL_FLOOR.sqrt().ln();
    let weak_phase = cosine.acos() / std::f64::consts::TAU;
    let n = term.n_obs();
    for atom in &mut term.atoms {
        atom.decoder_coefficients_mut().fill(0.0);
    }
    for (atom, coords) in term.assignment.coords.iter_mut().enumerate() {
        let phase = if atom == 0 { weak_phase } else { 0.05 };
        coords.set_flat(Array1::from_elem(n, phase).view());
    }
    term.refresh_basis_from_current_coords()
        .expect("the weak-phase basis refreshes");
    (term, target, rho)
}

/// `A_tβ` of one row: the dense slab and the installed row operator, as the reduced lane
/// materializes it.
fn cross_block(system: &ArrowSchurSystem, row: usize) -> Array2<f64> {
    let q = system.row_dims[row];
    let k = system.k;
    let dense = system.htbeta_dense_supplement || system.htbeta_matvec.is_none();
    let mut block = if dense && system.rows[row].htbeta.dim() == (q, k) {
        system.rows[row].htbeta.clone()
    } else {
        Array2::<f64>::zeros((q, k))
    };
    if let Some(operator) = system.htbeta_matvec.as_ref() {
        let mut unit = Array1::<f64>::zeros(k);
        let mut column = Array1::<f64>::zeros(q);
        for axis in 0..k {
            unit.fill(0.0);
            unit[axis] = 1.0;
            column.fill(0.0);
            operator(row, unit.view(), &mut column);
            block.column_mut(axis).scaled_add(1.0, &column);
        }
    }
    block
}

/// The lift `Z = [−T⁻¹A_tβ; I]`, `(total_t + k) × k`, through the row factor slab `T`.
fn lift(system: &ArrowSchurSystem, factors: &ArrowFactorSlab) -> Array2<f64> {
    let k = system.k;
    let total_t = system.row_offsets[system.rows.len()];
    let mut lift = Array2::<f64>::zeros((total_t + k, k));
    for row in 0..system.rows.len() {
        let q = system.row_dims[row];
        let base = system.row_offsets[row];
        let cross = cross_block(system, row);
        for axis in 0..k {
            let solved =
                gam_linalg::triangular::cholesky_solve_vector(factors.factor(row), cross.column(axis));
            lift.slice_mut(s![base..base + q, axis])
                .assign(&solved.mapv(|value| -value));
        }
    }
    for axis in 0..k {
        lift[[total_t + axis, axis]] = 1.0;
    }
    lift
}

fn max_abs(matrix: &Array2<f64>) -> f64 {
    matrix.iter().fold(0.0_f64, |acc, value| acc.max(value.abs()))
}

#[test]
fn the_reduced_lane_pencil_operands_are_the_dense_evidence_metric_on_the_lift_2933_f07() {
    let refusing = ArrowSolveOptions::direct()
        .with_newton_schur_tikhonov(SPECTRAL_DEFLATION_REL_FLOOR)
        .with_indefinite_refusing_evidence_unit_deflation(SPECTRAL_DEFLATION_REL_FLOOR);
    let mut checked = 0usize;
    for (label, (mut term, target, rho)) in [
        ("weak phase", weak_phase_state()),
        ("cold", crate::manifold::tests::small_two_atom_periodic_term()),
    ] {
        let mut majorizer = term
            .assemble_arrow_schur(target.view(), &rho, None)
            .expect("majorizer assembly");
        SaeManifoldTerm::ensure_row_gauge_deflation_for_quasi_laplace(&mut majorizer);
        let (_, _, cache) = solve_arrow_newton_step_with_options(
            &majorizer,
            0.0,
            0.0,
            &term.evidence_factor_options(),
        )
        .expect("the majorizer's evidence factor");
        let spectral_rows = cache
            .deflation_row_spectra
            .iter()
            .filter(|spectrum| spectrum.is_some())
            .count();
        let border_pins = cache
            .beta_schur_conditioning
            .as_ref()
            .map_or(0, |spectrum| {
                spectrum
                    .conditioning
                    .iter()
                    .filter(|branch| matches!(branch, BetaSchurSpectralConditioning::UnitDeflated))
                    .count()
            });
        let exact = term
            .exact_a_evidence_system(target.view(), &rho, &majorizer, 1.0)
            .expect("the exact-A evidence system");
        let mut lane = SurrogateLaneState::new(SurrogateLaneConfig {
            num_probes: 4,
            seed: 0x2933,
            rel_tol: 1.0e-10,
            cg_rel_tol: 1.0e-12,
            deflation_subspace_iters: 1,
            deflation_target_std_err_rel: 1.0,
        });
        let evaluated = matrix_free_arrow_evidence_evaluation(
            &exact, 0.0, 0.0, &refusing, 4, 1, 0x2933, &mut lane, true,
        );
        println!(
            "[pencil lane cross-crate] {label}: spectral rows {spectral_rows}, gauge directions \
             {}, border pins {border_pins}, quotient {}, border {}, exact-A evaluation {}",
            cache.gauge_deflated_directions,
            majorizer.beta_gauge_quotient.is_some(),
            cache.k,
            evaluated
                .as_ref()
                .map_or_else(|error| format!("refused: {error}"), |_| "priced".to_string()),
        );
        if spectral_rows == 0 || border_pins > 0 || majorizer.beta_gauge_quotient.is_some() {
            continue;
        }
        let Ok(evaluation) = evaluated else {
            continue;
        };
        let prepared = ArrowMetric::Joint(&cache)
            .prepare()
            .expect("the prepared evidence metric");
        let majorizer_factors = match &cache.htt_factors_undamped {
            ArrowUndampedFactors::SameAsDamped => &cache.htt_factors,
            ArrowUndampedFactors::Owned(factors) => factors,
        };
        for (factors_label, factors) in [
            ("exact-A evidence factors", &evaluation.factor_cache.htt_factors),
            ("majorizer evidence factors", majorizer_factors),
        ] {
            let operands = exact_a_reduced_pencil_operands(&exact, factors)
                .expect("the lane's pencil operands");
            let lift = lift(&exact, factors);
            let k = exact.k;
            let mut metric_image = Array2::<f64>::zeros(lift.dim());
            let mut substituted_image = Array2::<f64>::zeros(lift.dim());
            for axis in 0..k {
                metric_image
                    .column_mut(axis)
                    .assign(&prepared.apply(lift.column(axis)).expect("Φ apply"));
                substituted_image.column_mut(axis).assign(
                    &prepared
                        .substituted_image(lift.column(axis))
                        .expect("substituted image"),
                );
            }
            let dense_metric = lift.t().dot(&metric_image);
            let dense_substituted = lift.t().dot(&substituted_image);
            let lift_norm_sq = (0..k)
                .map(|axis| operands.lift_gram[[axis, axis]])
                .fold(0.0_f64, f64::max);
            // Both routes form a quadratic form of the lift in `Φ`: its backward error is
            // `dim·ε·‖Φ‖·‖z‖²`, and the lane's `B_tt = A_tt − ΔC_tt` adds `ε‖A‖`.
            let bar = operands.joint_dimension as f64
                * f64::EPSILON
                * (operands.operator_frobenius + operands.metric_frobenius)
                * lift_norm_sq;
            let metric_gap = max_abs(&(&operands.metric - &dense_metric));
            let substituted_gap = max_abs(&(&operands.substituted_metric - &dense_substituted));
            let reached = max_abs(&dense_substituted);
            println!(
                "[pencil lane cross-crate] {label}, {factors_label}: max|Y − zᵀΦz| {metric_gap:.3e}, \
                 max|s − zᵀ(Φ − B_raw)z| {substituted_gap:.3e}, bar {bar:.3e}; max|zᵀ(Φ − B_raw)z| \
                 {reached:.3e} ({:.3e} bars), max|Y| {:.3e}",
                reached / bar,
                max_abs(&dense_metric),
            );
            assert!(
                reached > bar,
                "{label}, {factors_label}: premise: the lift must reach the row pins above the \
                 comparison bar"
            );
            assert!(
                metric_gap <= bar,
                "{label}, {factors_label}: the lane's Y departs from zᵀΦz by {metric_gap:e}"
            );
            assert!(
                substituted_gap <= bar,
                "{label}, {factors_label}: the lane's s departs from zᵀ(Φ − B_raw)z by \
                 {substituted_gap:e}"
            );
            checked += 1;
        }
    }
    assert!(
        checked > 0,
        "no candidate state carries spectral row pins, no border pin or quotient, and a priced \
         exact-A evaluation; see the census above"
    );
}
