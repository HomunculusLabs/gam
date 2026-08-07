//! #2515 measurement scaffold — what a bundle route handed the EXACT-`A`
//! geometry actually produces, channel by channel, against the dense exact-`A`
//! authority.
//!
//! The open half of #2515 is that route selection changes the ranked criterion:
//! the dense direct-logdet route prices `½log|A|` for `A = ∇²_θθ L`, while the
//! bundle / matrix-free route differentiates the Gauss--Newton majorizer `B`.
//! The repair has two independent halves, and this module exists to measure
//! which of them each ρ-coordinate needs before either is written:
//!
//! * the WRONG INVERSE — the from-probes channels reconstruct `(H⁻¹)_tt` as
//!   `A_i⁻¹ + G_i S⁻¹ G_iᵀ` off the factor cache they are handed, so handing
//!   them `B`'s cache contracts `B⁻¹` no matter which operator's probes arrive;
//! * the WRONG OPERATOR — `∂A/∂ρ = ∂B/∂ρ + ∂ΔC/∂ρ`, and `∂ΔC/∂ρ` is nonzero on
//!   exactly the coordinates `exact_stationarity_penalty_derivative_delta_by_flat`
//!   keys (ARD log-precision, and the softmax sparse log-strength at `K ≥ 2`).
//!
//! The probe below fixes the state, builds the exact-`A` evidence system, factors
//! it, forms the FULL-BASIS probe set `√k·e_j` with exact `S_A⁻¹ e_j` off that
//! factorization, and reports every channel both ways. Full-basis probes remove
//! stochastic approximation, so a residual is an authority defect and not probe
//! noise.

use super::tests::{TestPeriodicEvaluator, periodic_basis};
use super::construction::ThetaAdjointDhChannel;
use super::outer_objective::sae_surrogate_lane_config;
use super::*;
use ndarray::array;
use std::sync::Arc;

/// The shared #2515 witness state: one periodic atom on a unit-period circle at
/// `α = 250`, which puts `cos κt < 0` over a third of the rows so the periodic
/// ARD concave clamp is genuinely ACTIVE — that is what makes `ΔC = A − B`
/// nonzero and the two routes distinguishable at all.
pub(crate) struct ExactAWitness2515 {
    pub(crate) term: SaeManifoldTerm,
    pub(crate) target: Array2<f64>,
    pub(crate) rho: SaeManifoldRho,
}

pub(crate) fn exact_a_witness_2515() -> ExactAWitness2515 {
    exact_a_witness_2515_at_alpha(250.0)
}

/// The same state at a caller-chosen ARD precision. `α ≤ 10` keeps `A = B + ΔC`
/// and its reduced Schur positive definite, so BOTH evidence routes are admitted
/// and route parity is statable; the historical `α = 250` witness is far past
/// that boundary (see `zz_scan_exact_a_admitted_alpha_2515`).
pub(crate) fn exact_a_witness_2515_at_alpha(alpha: f64) -> ExactAWitness2515 {
    let n = 24usize;
    let p = 2usize;
    let coords = Array2::from_shape_fn((n, 1), |(row, _)| (row as f64 + 0.25) / n as f64);
    let (phi, jet) = periodic_basis(&coords);
    let decoder = array![[0.30, -0.10], [1.20, 0.20], [0.10, 1.10]];
    assert_eq!(decoder.ncols(), p);
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
    .expect("the #2515 witness atom is built from a well-formed periodic chart")
    .with_basis_evaluator(Arc::new(TestPeriodicEvaluator));
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        Array2::<f64>::zeros((n, 1)),
        vec![coords],
        vec![LatentManifold::Circle { period: 1.0 }],
        AssignmentMode::softmax(1.0),
    )
    .expect("the #2515 witness assignment has one block and one manifold");
    let term = SaeManifoldTerm::new(vec![atom], assignment)
        .expect("one atom against a one-block assignment is a valid term");
    let rho = SaeManifoldRho::new(0.0, 0.8_f64.ln(), vec![array![alpha.ln()]]);
    ExactAWitness2515 { term, target, rho }
}

/// SCAN — find an `α` where the exact observed information `A = B + ΔC` is still
/// per-row PD (so the streaming/bundle evidence route is ADMITTED) while the
/// periodic ARD concave clamp is genuinely ACTIVE (so `ΔC ≠ 0` and the two routes
/// are distinguishable at all). The named #2515 witness sits at `α = 250`, where
/// `A`'s worst per-row eigenvalue is `−1.93e2` and the streaming route refuses —
/// so route PARITY cannot be stated there, only the operator gap.
#[test]
fn zz_scan_exact_a_admitted_alpha_2515() {
    for log10_alpha in [-0.5_f64, 0.0, 0.5, 0.7, 1.0, 1.3, 1.6, 2.0, 2.4] {
        let alpha = 10.0_f64.powf(log10_alpha);
        let ExactAWitness2515 {
            mut term, target, ..
        } = exact_a_witness_2515();
        let rho = SaeManifoldRho::new(0.0, 0.8_f64.ln(), vec![array![alpha.ln()]]);
        let sys = match term.assemble_arrow_schur(target.view(), &rho, None) {
            Ok(sys) => sys,
            Err(err) => {
                println!("[#2515 SCAN] alpha={alpha:.4e}: assembly refused: {err}");
                continue;
            }
        };
        let a_sys = match term.exact_a_evidence_system(target.view(), &rho, &sys) {
            Ok(sys) => sys,
            Err(err) => {
                println!("[#2515 SCAN] alpha={alpha:.4e}: exact-A assembly refused: {err}");
                continue;
            }
        };
        let mut worst_b = f64::INFINITY;
        let mut worst_a = f64::INFINITY;
        for (s, worst) in [(&sys, &mut worst_b), (&a_sys, &mut worst_a)] {
            for row in &s.rows {
                let (eigs, _) =
                    gam_linalg::faer_ndarray::FaerEigh::eigh(&row.htt, faer::Side::Lower).unwrap();
                *worst = worst.min(eigs.iter().cloned().fold(f64::INFINITY, f64::min));
            }
        }
        // How many rows carry a live `ΔC` on the ARD coordinate?
        let options = ArrowSolveOptions::direct().with_positive_definite_evidence();
        let clamped = match solve_arrow_newton_step_with_options(&sys, 0.0, 0.0, &options) {
            Ok((_, _, cache)) => term
                .exact_stationarity_penalty_derivative_delta_by_flat(&rho, &cache)
                .map(|d| {
                    d.get(&rho.ard_flat_index(0, 0))
                        .map(|m| m.diag().iter().filter(|v| **v != 0.0).count())
                        .unwrap_or(0)
                })
                .unwrap_or(0),
            Err(_) => usize::MAX,
        };
        let mut lane = SurrogateLaneState::new(sae_surrogate_lane_config());
        lane.request_logdet_derivative_bundle();
        let evidence_options = ArrowSolveOptions::direct()
            .with_newton_schur_tikhonov(gam_solve::arrow_schur::SPECTRAL_DEFLATION_REL_FLOOR)
            .with_evidence_unit_deflation(gam_solve::arrow_schur::SPECTRAL_DEFLATION_REL_FLOOR);
        let streaming = gam_solve::arrow_schur::matrix_free_arrow_evidence_evaluation(
            &a_sys,
            0.0,
            0.0,
            &evidence_options,
            8,
            16,
            0xA51_5u64,
            &mut lane,
        );
        let dense_a = solve_arrow_newton_step_with_options(&a_sys, 0.0, 0.0, &options).is_ok();
        println!(
            "[#2515 SCAN] alpha={alpha:.4e}  min_eig(B)={worst_b:.4e}  min_eig(A)={worst_a:.4e}  \
             clamped_rows={clamped}  dense_A_factors={dense_a}  streaming_A={}",
            match &streaming {
                Ok(e) => format!("OK logdet={:.6e}", e.log_det()),
                Err(err) => format!("REFUSED {err:?}"),
            }
        );
    }
}

/// Full-basis probe set `√k·e_j` with exact `S⁻¹ e_j` off `cache`. At this probe
/// set the Hutchinson umbrella `(1/m)Σ_j (S⁻¹z_j)ᵀ M z_j` is EXACTLY
/// `tr(S⁻¹ M)`, so every from-probes channel is deterministic.
pub(crate) fn full_basis_probe_bundle(
    cache: &ArrowFactorCache,
) -> (Vec<Array1<f64>>, Vec<Array1<f64>>) {
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
                .expect("the #2515 probe directions are in the cached Schur complement's domain")
        })
        .collect();
    (probes, sinv)
}

/// MEASUREMENT — hand the bundle route the exact-`A` factor cache and the
/// exact-`A` probe bundle, and report every logdet-trace coordinate against the
/// dense exact-`A` authority.
///
/// This is a probe, not a contract: it asserts only that both routes are
/// constructible and finite, and prints the residual per coordinate. The
/// contract it is scouting for lives on
/// `laplace_value_and_gradient_are_route_invariant_2515`.
#[test]
fn zz_measure_exact_a_geometry_bundle_channels_2515() {
    let ExactAWitness2515 {
        mut term,
        target,
        rho,
    } = exact_a_witness_2515_at_alpha(10.0);
    let sys = term
        .assemble_arrow_schur(target.view(), &rho, None)
        .unwrap();
    let options = ArrowSolveOptions::direct().with_positive_definite_evidence();
    let (_delta_t, _delta_beta, b_cache) =
        solve_arrow_newton_step_with_options(&sys, 0.0, 0.0, &options).unwrap();
    let loss = term.loss(target.view(), &rho).unwrap();

    // The exact-A evidence operator and ITS factorization.
    let a_sys = term
        .exact_a_evidence_system(target.view(), &rho, &sys)
        .expect("the exact-A evidence system must be constructible on this witness");
    println!(
        "[#2515 A-GEOMETRY] row_gauge_deflation installed: B={} A={}",
        sys.row_gauge_deflation.is_some(),
        a_sys.row_gauge_deflation.is_some()
    );
    for (label, s) in [("B", &sys), ("A", &a_sys)] {
        let mut worst = f64::INFINITY;
        let mut worst_row = 0usize;
        for (i, row) in s.rows.iter().enumerate() {
            let eig = gam_linalg::faer_ndarray::FaerEigh::eigh(&row.htt, faer::Side::Lower).unwrap();
            let min = eig.0.iter().cloned().fold(f64::INFINITY, f64::min);
            if min < worst {
                worst = min;
                worst_row = i;
            }
        }
        println!("[#2515 A-GEOMETRY] {label} min per-row H_tt eigenvalue = {worst:.6e} (row {worst_row})");
    }
    let mut lane = SurrogateLaneState::new(sae_surrogate_lane_config());
    lane.request_logdet_derivative_bundle();
    let evidence_options = ArrowSolveOptions::direct()
        .with_newton_schur_tikhonov(gam_solve::arrow_schur::SPECTRAL_DEFLATION_REL_FLOOR)
        .with_evidence_unit_deflation(gam_solve::arrow_schur::SPECTRAL_DEFLATION_REL_FLOOR);
    match gam_solve::arrow_schur::matrix_free_arrow_evidence_evaluation(
        &a_sys,
        0.0,
        0.0,
        &evidence_options,
        8,
        16,
        0xA51_5u64,
        &mut lane,
    ) {
        Ok(evaluated) => println!(
            "[#2515 A-GEOMETRY] streaming exact-A evidence OK: log_det_tt={:.17e} \
             log_det_schur={:.17e} bundle={:?}",
            evaluated.log_det_tt,
            evaluated.log_det_schur,
            lane.take_logdet_derivative_bundle().map(|b| b.vectors.len()),
        ),
        Err(err) => println!("[#2515 A-GEOMETRY] streaming exact-A evidence REFUSED: {err:?}"),
    }
    let a_factored = solve_arrow_newton_step_with_options(&a_sys, 0.0, 0.0, &options);
    let a_cache = match a_factored {
        Ok((_, _, cache)) => cache,
        Err(err) => {
            println!("[#2515 A-GEOMETRY] dense exact-A arrow factorization REFUSED: {err:?}");
            return;
        }
    };
    println!(
        "[#2515 A-GEOMETRY] B log|H|={:?} A log|H|={:?}  A rows deflated={}",
        b_cache.arrow_log_det(),
        a_cache.arrow_log_det(),
        a_cache
            .deflated_row_directions
            .iter()
            .filter(|d| !d.is_empty())
            .count()
    );

    let dense = term
        .dense_exact_a_logdet_channels(target.view(), &rho, &loss, &b_cache)
        .expect("dense exact-A channels must assemble");

    let (a_probes, a_sinv) = full_basis_probe_bundle(&a_cache);
    let (b_probes, b_sinv) = full_basis_probe_bundle(&b_cache);
    let lambda_smooth = rho.lambda_smooth_vec().unwrap();

    // Channel 1 — decoder smoothness EDF. Pure `tr(S⁻¹ M_k)`: no factor cache at
    // all, so its ONLY route dependence is which operator's `S⁻¹` arrives.
    let smooth_on_a = term
        .decoder_smoothness_effective_dof_per_atom_from_probes(&a_probes, &a_sinv, &lambda_smooth)
        .unwrap();
    let smooth_on_b = term
        .decoder_smoothness_effective_dof_per_atom_from_probes(&b_probes, &b_sinv, &lambda_smooth)
        .unwrap();
    for atom in 0..rho.k_atoms() {
        let flat = rho.smooth_flat_index(atom);
        println!(
            "[#2515 A-GEOMETRY] smooth atom {atom} (flat {flat}): dense_A={:.17e} \
             from_probes_on_A={:.17e} from_probes_on_B={:.17e}",
            dense.logdet_trace[flat],
            0.5 * smooth_on_a[atom],
            0.5 * smooth_on_b[atom],
        );
    }

    // Channel 2 — ARD log-precision Hessian trace. Reads the factor cache for the
    // row-local block AND differentiates the majorized curvature, so it can be
    // wrong in both ways at once.
    for (label, cache, probes, sinv, operator) in [
        ("A/exact-A-operand", &a_cache, &a_probes, &a_sinv, EvidenceOperator::ExactObservedInformation),
        ("A/B-operand", &a_cache, &a_probes, &a_sinv, EvidenceOperator::Majorizer),
        ("B/B-operand", &b_cache, &b_probes, &b_sinv, EvidenceOperator::Majorizer),
    ] {
        let joint = term
            .ard_log_precision_hessian_trace_from_probes(&rho, cache, probes, sinv, operator)
            .unwrap();
        let coordinate = term
            .coordinate_block_ard_log_precision_hessian_trace(&rho, cache, operator)
            .unwrap();
        for k in 0..rho.log_ard.len() {
            for axis in 0..rho.log_ard[k].len() {
                let flat = rho.ard_flat_index(k, axis);
                println!(
                    "[#2515 A-GEOMETRY] ard ({k},{axis}) (flat {flat}) on {label}: \
                     dense_A={:.17e} from_probes={:.17e} (joint={:.17e} coord={:.17e})",
                    dense.logdet_trace[flat],
                    joint[k][axis] - coordinate[k][axis],
                    joint[k][axis],
                    coordinate[k][axis],
                );
            }
        }
    }

    // Channel 3 — the #1006 envelope θ-adjoint Γ, with the exact-A operand switch
    // that `ac499b513` built and nothing wires.
    let dense_gamma = &dense.theta_adjoint;
    for (label, cache, probes, sinv, exact_a) in [
        ("A-cache/exact-A-operands", &a_cache, &a_probes, &a_sinv, true),
        (
            "A-cache/B-operands",
            &a_cache,
            &a_probes,
            &a_sinv,
            false,
        ),
        ("B-cache/B-operands", &b_cache, &b_probes, &b_sinv, false),
    ] {
        match term.logdet_theta_adjoint_from_probes(&rho, cache, probes, sinv, exact_a) {
            Ok(mut gamma) => {
                let coordinate_gamma = term
                    .coordinate_block_logdet_theta_adjoint(&rho, cache)
                    .unwrap();
                gamma.t -= &coordinate_gamma.t;
                gamma.beta -= &coordinate_gamma.beta;
                let rank_charge = term
                    .production_rank_charge_derivative(target.view(), &rho, &loss, cache)
                    .unwrap();
                gamma.t.scaled_add(2.0, &rank_charge.theta.t);
                gamma.beta.scaled_add(2.0, &rank_charge.theta.beta);
                let dt = gamma
                    .t
                    .iter()
                    .zip(dense_gamma.t.iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0_f64, f64::max);
                let db = gamma
                    .beta
                    .iter()
                    .zip(dense_gamma.beta.iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0_f64, f64::max);
                println!(
                    "[#2515 A-GEOMETRY] gamma {label}: max|Δt|={dt:.6e} max|Δbeta|={db:.6e}"
                );
            }
            Err(err) => println!("[#2515 A-GEOMETRY] gamma {label}: REFUSED: {err}"),
        }
    }

    // ATTRIBUTION for the γ residual: the from-probes θ-adjoint documents that it
    // does NOT carry the #2330 Patch-D residual THIRD-derivative legs. Those are
    // exactly the difference between `logdet_theta_adjoint_dense` called with
    // `residual_target = Some(..)` and with `None`, so measure that difference on
    // the SAME pseudo-inverse and see whether it accounts for the gap.
    let a_dense = term
        .materialize_exact_hessian_dense(&rho, target.view(), &b_cache)
        .unwrap();
    let (eigs, vecs) =
        gam_linalg::faer_ndarray::FaerEigh::eigh(&a_dense, faer::Side::Lower).unwrap();
    let max_eig = eigs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let floor = SaeManifoldTerm::SAE_EXACT_A_PD_FLOOR_REL * max_eig.max(1.0);
    let weights = Array1::from_iter(
        eigs.iter()
            .map(|&l| if l > floor { 1.0 / l } else { 0.0 }),
    );
    let a_pinv = vecs.dot(&Array2::from_diag(&weights)).dot(&vecs.t());
    let min_eig = eigs.iter().cloned().fold(f64::INFINITY, f64::min);
    println!(
        "[#2515 A-GEOMETRY] joint A spectrum: min={min_eig:.6e} max={max_eig:.6e} floor={floor:.6e}"
    );
    let without = term
        .logdet_theta_adjoint_dense(
            &rho,
            &b_cache,
            &a_pinv,
            ThetaAdjointDhChannel::All,
            true,
            true,
            None,
        )
        .unwrap();
    let with = term
        .logdet_theta_adjoint_dense(
            &rho,
            &b_cache,
            &a_pinv,
            ThetaAdjointDhChannel::All,
            true,
            true,
            Some(target.view()),
        )
        .unwrap();
    let patchd = with
        .t
        .iter()
        .zip(without.t.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    println!("[#2515 A-GEOMETRY] Patch-D residual third-derivative leg: max|Δt|={patchd:.6e}");
    let probes_exact = term
        .logdet_theta_adjoint_from_probes(&rho, &a_cache, &a_probes, &a_sinv, true)
        .unwrap();
    let against_none = probes_exact
        .t
        .iter()
        .zip(without.t.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    let against_some = probes_exact
        .t
        .iter()
        .zip(with.t.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    println!(
        "[#2515 A-GEOMETRY] from_probes(exact_a) vs dense(exact_a): \
         residual_target=None -> max|Δt|={against_none:.6e}; Some -> max|Δt|={against_some:.6e}"
    );

    // THE ASSEMBLED GRADIENT, both geometries, through the one production entry
    // point. `Majorizer` reproduces the historical bundle route byte for byte;
    // `ExactObservedInformation` routes every from-probes and coordinate-block
    // channel onto the exact-A factor cache.
    let b_solver = DeflatedArrowSolver::plain(&b_cache);
    let dense_components = term
        .analytic_outer_rho_gradient_components(target.view(), &rho, &loss, &b_cache, &b_solver)
        .unwrap();
    for (label, geometry) in [
        (
            "Majorizer",
            BundleEvidenceGeometry {
                operator: EvidenceOperator::Majorizer,
                cache: &b_cache,
                probes: &b_probes,
                sinv: &b_sinv,
            },
        ),
        (
            "ExactObservedInformation",
            BundleEvidenceGeometry {
                operator: EvidenceOperator::ExactObservedInformation,
                cache: &a_cache,
                probes: &a_probes,
                sinv: &a_sinv,
            },
        ),
    ] {
        match term.analytic_outer_rho_gradient_components_with_bundle(
            target.view(),
            &rho,
            &loss,
            &b_cache,
            &b_solver,
            Some(geometry),
            None,
        ) {
            Ok(components) => {
                for i in 0..components.logdet_trace.len() {
                    println!(
                        "[#2515 A-GEOMETRY] assembled {label} logdet_trace[{i}]={:.17e} \
                         dense_exact_A={:.17e} |Δ|={:.6e}",
                        components.logdet_trace[i],
                        dense_components.logdet_trace[i],
                        (components.logdet_trace[i] - dense_components.logdet_trace[i]).abs(),
                    );
                }
                let g = components.gradient();
                let d = dense_components.gradient();
                let worst = g
                    .iter()
                    .zip(d.iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0_f64, f64::max);
                println!("[#2515 A-GEOMETRY] assembled {label} complete gradient max|Δ|={worst:.6e}");
            }
            Err(err) => println!("[#2515 A-GEOMETRY] assembled {label}: REFUSED: {err}"),
        }
    }

    assert!(
        dense.logdet_trace.iter().all(|v| v.is_finite()),
        "the dense exact-A logdet trace must be finite"
    );
}
