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
use approx::assert_abs_diff_eq;
use super::construction::{ArrowMetric, ThetaAdjointDhChannel, sae_exact_a_direction_floor};
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

    // VALUE side. `log|A| - log|A_tt| = log|S_A|` is an algebraic identity, and the
    // two routes reach it by completely different means: the dense route
    // eigendecomposes the materialized `A` and its t-block under one spectral
    // floor, the streaming route factors `exact_a_evidence_system` row by row and
    // eliminates. If the two disagree the routes are not ranking one operator, and
    // no gradient parity would mean anything.
    let (dense_log_a, dense_log_a_tt) = term
        .exact_observed_information_log_dets(&rho, target.view(), &b_cache)
        .expect("the dense exact-A log-determinants must be available");
    let arrow_log_a = a_cache
        .arrow_log_det()
        .expect("the exact-A arrow factorization must publish its joint log-det");
    let arrow_log_a_tt = super::construction::coordinate_block_log_det(&a_cache)
        .expect("the exact-A coordinate block log-det must be available");
    println!(
        "[#2515 A-GEOMETRY] value: dense 1/2(log|A|-log|A_tt|)={:.17e}          arrow 1/2 log|S_A|={:.17e} |Δ|={:.6e}",
        0.5 * (dense_log_a - dense_log_a_tt),
        0.5 * (arrow_log_a - arrow_log_a_tt),
        (0.5 * (dense_log_a - dense_log_a_tt) - 0.5 * (arrow_log_a - arrow_log_a_tt)).abs(),
    );
    let b_log_b = b_cache
        .arrow_log_det()
        .expect("the majorizer arrow factorization must publish its joint log-det");
    let b_log_b_tt = super::construction::coordinate_block_log_det(&b_cache)
        .expect("the majorizer coordinate block log-det must be available");
    println!(
        "[#2515 A-GEOMETRY] value: majorizer 1/2 log|S_B|={:.17e} (A-vs-B separation {:.6e})",
        0.5 * (b_log_b - b_log_b_tt),
        (0.5 * (arrow_log_a - arrow_log_a_tt) - 0.5 * (b_log_b - b_log_b_tt)).abs(),
    );
    println!(
        "[#2515 A-GEOMETRY] row-Hessian fingerprints: B={} A={} equal={}",
        b_cache.row_hessian_fingerprint,
        a_cache.row_hessian_fingerprint,
        b_cache.row_hessian_fingerprint == a_cache.row_hessian_fingerprint,
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
    for (label, cache, probes, sinv, gamma_operator) in [
        (
            "A-cache/exact-A-operands",
            &a_cache,
            &a_probes,
            &a_sinv,
            EvidenceOperator::ExactObservedInformation,
        ),
        (
            "A-cache/B-operands",
            &a_cache,
            &a_probes,
            &a_sinv,
            EvidenceOperator::Majorizer,
        ),
        (
            "B-cache/B-operands",
            &b_cache,
            &b_probes,
            &b_sinv,
            EvidenceOperator::Majorizer,
        ),
    ] {
        // The residual target rides WITH the exact-A operand, exactly as the
        // assembler pairs them: `patchd_residual = exact_a.then_some(target)`.
        let gamma_target = gamma_operator.is_exact_a().then(|| target.view());
        match term.logdet_theta_adjoint_from_probes(
            &rho,
            cache,
            probes,
            sinv,
            gamma_operator,
            gamma_target,
        ) {
            Ok(mut gamma) => {
                let coordinate_gamma = term
                    .coordinate_block_logdet_theta_adjoint(&rho, cache, gamma_operator, gamma_target)
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
    // #2673 — the retained band is per direction, in the `B` metric.
    let spectral_norm = eigs.iter().map(|value| value.abs()).fold(0.0_f64, f64::max);
    let joint_metric = ArrowMetric::Joint(&b_cache);
    let floors: Vec<f64> = (0..eigs.len())
        .map(|idx| {
            let vbv = joint_metric
                .quadratic_form(vecs.column(idx))
                .expect("B quadratic form");
            sae_exact_a_direction_floor(eigs.len(), spectral_norm, vbv)
        })
        .collect();
    let weights = Array1::from_iter(
        eigs.iter()
            .enumerate()
            .map(|(idx, &l)| if l > floors[idx] { 1.0 / l } else { 0.0 }),
    );
    let a_pinv = vecs.dot(&Array2::from_diag(&weights)).dot(&vecs.t());
    let min_eig = eigs.iter().cloned().fold(f64::INFINITY, f64::min);
    println!(
        "[#2515 A-GEOMETRY] joint A spectrum: min={min_eig:.6e} max={max_eig:.6e} \
         widest floor={:.6e}",
        floors.iter().copied().fold(0.0_f64, f64::max)
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
        .logdet_theta_adjoint_from_probes(
            &rho,
            &a_cache,
            &a_probes,
            &a_sinv,
            EvidenceOperator::ExactObservedInformation,
            None,
        )
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

/// #2515 THE CONTRACT — storage and host capability must not choose the
/// statistical criterion.
///
/// The Laplace normalizer is `½log|∇²_θθ L|`, and `A = B + ΔC` IS that Hessian.
/// `B` is the Gauss--Newton / PSD-majorizer arrow system: the positive-definite
/// scale the Newton and IFT solves factor, and a preconditioner for `A`. A
/// preconditioner is not the operator it preconditions, so a route that ranks
/// `½log|A|` and differentiates `½log|B|` is differentiating a criterion nothing
/// evaluates. This fixture holds `θ`, `ρ`, the assembled system and every
/// non-logdet channel fixed, then requires BOTH the value component and the
/// COMPLETE analytic outer gradient to agree across the dense and bundle routes.
///
/// # Why `α = 10` and not the historical `α = 250`
///
/// Route parity is only a statement about states where BOTH routes are admitted.
/// At `α = 250` the exact observed information is badly indefinite — worst per-row
/// `H_tt` eigenvalue `−1.93e2`, reduced Schur indefinite — and the streaming
/// evidence route refuses outright while the dense arrow factorization refuses
/// too. Only the globally-priced dense route survives there, through #2336
/// clamp-attributable pricing (`1/(λ+e_v)`), for which the per-row spectral
/// conditioning of the streaming lane has no analogue. Asserting parity there
/// would be asserting a property of one route.
///
/// `zz_scan_exact_a_admitted_alpha_2515` walks four decades and locates the
/// boundary: at `α ≤ 10` both routes are admitted, `A` stays PD, and the periodic
/// ARD concave clamp is still ACTIVE on 12 of 24 rows — so `ΔC ≠ 0` and the two
/// operators remain distinguishable. That is where the contract can be written,
/// and the non-vacuity block below refuses to claim it unless the state still has
/// all three properties.
///
/// # Why full-basis probes
///
/// `√k·e_j` with exact `S⁻¹ e_j` makes the Hutchinson umbrella
/// `(1/m)Σ_j (S⁻¹z_j)ᵀ M z_j` EXACTLY `tr(S⁻¹M)`, so any residual is an
/// operator-authority defect rather than probe noise. The production lane
/// contracts the rational surrogate's own weighted derivative representation
/// instead, whose value/gradient consistency is #2080's contract, not this one's.
#[test]
fn laplace_value_and_gradient_are_route_invariant_2515() {
    let ExactAWitness2515 {
        mut term,
        target,
        rho,
    } = exact_a_witness_2515_at_alpha(10.0);
    let sys = term
        .assemble_arrow_schur(target.view(), &rho, None)
        .expect("the witness arrow system must assemble");
    let options = ArrowSolveOptions::direct().with_positive_definite_evidence();
    let (_delta_t, _delta_beta, b_cache) =
        solve_arrow_newton_step_with_options(&sys, 0.0, 0.0, &options)
            .expect("the majorizer arrow factorization must succeed");
    let loss = term
        .loss(target.view(), &rho)
        .expect("the loss at the frozen state must be available");

    // (0) ADMISSION. Both routes must exist at this state, or parity is a
    // statement about one of them.
    let a_sys = term
        .exact_a_evidence_system(target.view(), &rho, &sys)
        .expect("#2515: the exact-A evidence system must be constructible at this state");
    let (_, _, a_cache) = solve_arrow_newton_step_with_options(&a_sys, 0.0, 0.0, &options)
        .expect(
            "#2515: this witness must sit in the regime where the exact-A arrow \
             factorization is ADMITTED — see zz_scan_exact_a_admitted_alpha_2515 for the \
             α boundary; at α = 250 it refuses and no parity statement is available",
        );

    // (1) NON-VACUITY, before any parity claim.
    //
    // (1a) `ΔC ≠ 0` on the ARD coordinate. Without a live clamp `A == B` and
    // every assertion below passes for the wrong reason.
    let delta_by_flat = term
        .exact_stationarity_penalty_derivative_delta_by_flat(&rho, &b_cache)
        .expect("the exact-A penalty derivative delta must be assemblable");
    let ard_flat = rho.ard_flat_index(0, 0);
    let clamped_rows = delta_by_flat
        .get(&ard_flat)
        .map(|block| block.diag().iter().filter(|v| **v != 0.0).count())
        .unwrap_or(0);
    assert!(
        clamped_rows > 0,
        "#2515: this witness must carry a live ∂ΔC/∂log α on ARD coordinate {ard_flat} \
         (cos κt < 0 on some rows), or A == B and route invariance is vacuous. \
         Keys present: {:?}",
        delta_by_flat.keys().collect::<Vec<_>>()
    );

    // (1b) The two caches must factor DIFFERENT operators. The row-Hessian
    // fingerprint is content-derived (#2515 `2a740061c` / `60feddc2e`), so equal
    // fingerprints would mean the exact-A system is the majorizer and the
    // geometry swap below is a no-op.
    assert_ne!(
        a_cache.row_hessian_fingerprint, b_cache.row_hessian_fingerprint,
        "#2515: the exact-A and majorizer caches must factor different operators; \
         identical row-Hessian fingerprints mean ΔC never reached the arrow blocks"
    );

    // (2) VALUE parity. `log|A| − log|A_tt| = log|S_A|` is an identity, and the
    // two routes reach it by unrelated means: the dense route eigendecomposes the
    // materialized `A` and its t-block under one spectral floor; the arrow route
    // factors `exact_a_evidence_system` row by row and eliminates the border.
    let (dense_log_a, dense_log_a_tt) = term
        .exact_observed_information_log_dets(&rho, target.view(), &b_cache)
        .expect("the dense exact-A log-determinants must be available");
    let arrow_log_a = a_cache
        .arrow_log_det()
        .expect("the exact-A arrow factorization must publish its joint log-det");
    let arrow_log_a_tt = super::construction::coordinate_block_log_det(&a_cache)
        .expect("the exact-A coordinate block log-det must be available");
    let dense_value = 0.5 * (dense_log_a - dense_log_a_tt);
    let arrow_value = 0.5 * (arrow_log_a - arrow_log_a_tt);
    let majorizer_value = 0.5
        * (b_cache
            .arrow_log_det()
            .expect("the majorizer arrow factorization must publish its joint log-det")
            - super::construction::coordinate_block_log_det(&b_cache)
                .expect("the majorizer coordinate block log-det must be available"));
    eprintln!(
        "[#2515 ROUTE-PARITY] dense ½(log|A|−log|A_tt|)={dense_value:.17e} \
         arrow ½log|S_A|={arrow_value:.17e} majorizer ½log|S_B|={majorizer_value:.17e}"
    );
    // (1c) …and they must be far apart, or "both routes price A" is unfalsifiable
    // here: a bundle still rooted in `B` would pass a parity test whose two
    // operators happen to coincide.
    assert!(
        (arrow_value - majorizer_value).abs() > 1.0e-3,
        "#2515: the exact-A and majorizer Laplace values must SEPARATE on this \
         witness, else nothing distinguishes the two routes: A={arrow_value:.17e} \
         B={majorizer_value:.17e}"
    );
    assert_abs_diff_eq!(dense_value, arrow_value, epsilon = 1.0e-9);

    // (3) GRADIENT parity. The dense route takes the exact-A arm (no bundle, no
    // matrix-free system); the bundle route is handed the exact-A geometry, which
    // is exactly what production now mints.
    let b_solver = DeflatedArrowSolver::plain(&b_cache);
    let dense = term
        .analytic_outer_rho_gradient_components(target.view(), &rho, &loss, &b_cache, &b_solver)
        .expect("the dense exact-A gradient must assemble");
    let (a_probes, a_sinv) = full_basis_probe_bundle(&a_cache);
    let bundled = term
        .analytic_outer_rho_gradient_components_with_bundle(
            target.view(),
            &rho,
            &loss,
            &b_cache,
            &b_solver,
            Some(BundleEvidenceGeometry {
                operator: EvidenceOperator::ExactObservedInformation,
                cache: &a_cache,
                probes: &a_probes,
                sinv: &a_sinv,
            }),
            None,
        )
        .expect("the exact-A bundle gradient must assemble");

    // (1d) THE CARRIER CONTROL. A geometry that silently passed the MAJORIZER
    // cache in the exact-A slot would reconstruct `B⁻¹` while claiming `A`, and
    // every assertion above would still pass. Require that mis-paired carrier to
    // land far from the dense route before believing the correct one lands on it.
    let (b_probes, b_sinv) = full_basis_probe_bundle(&b_cache);
    let miswired = term
        .analytic_outer_rho_gradient_components_with_bundle(
            target.view(),
            &rho,
            &loss,
            &b_cache,
            &b_solver,
            Some(BundleEvidenceGeometry {
                operator: EvidenceOperator::Majorizer,
                cache: &b_cache,
                probes: &b_probes,
                sinv: &b_sinv,
            }),
            None,
        )
        .expect("the majorizer bundle gradient must assemble")
        .gradient();
    let dense_gradient = dense.gradient();
    let miswired_gap = dense_gradient
        .iter()
        .zip(miswired.iter())
        .map(|(d, m)| (d - m).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        miswired_gap > 1.0e-3,
        "#2515: the MAJORIZER-rooted bundle must still disagree with the dense \
         exact-A gradient (measured 8.46e-1 when this was written). A gap of \
         {miswired_gap:.6e} means the two operators have collapsed on this state and \
         the parity assertion below proves nothing"
    );

    // Per-CHANNEL first, so a divergence names the channel that produced it
    // rather than only the summed number (#2465).
    assert_eq!(dense.logdet_trace.len(), bundled.logdet_trace.len());
    for (i, (d, b)) in dense
        .logdet_trace
        .iter()
        .zip(bundled.logdet_trace.iter())
        .enumerate()
    {
        assert!(
            d.is_finite() && b.is_finite(),
            "logdet-trace coordinate {i} must be finite (dense={d}, bundled={b})"
        );
        // Name the coordinate. `sparse_flat_index` / `smooth_flat_start` /
        // `ard_flat_index` are the layout's own accessors, so this cannot drift
        // from `to_flat` — and a bare "0.159 vs 0.282" costs the next reader the
        // whole layout walk to find out which channel desynced.
        let role = if Some(i) == rho.sparse_flat_index() {
            "assignment log-strength".to_string()
        } else if i >= rho.smooth_flat_start() && i < rho.smooth_flat_start() + rho.k_atoms() {
            format!("smooth atom {}", i - rho.smooth_flat_start())
        } else {
            format!("ard flat {i}")
        };
        assert!(
            (d - b).abs() <= 1.0e-9,
            "logdet-trace coordinate {i} ({role}) desynced between the dense exact-A \
             route and the exact-A bundle route: dense={d}, bundled={b}, |Δ|={:.6e}",
            (d - b).abs()
        );
    }
    for (label, d, b) in [
        ("explicit", &dense.explicit, &bundled.explicit),
        ("occam", &dense.occam, &bundled.occam),
        (
            "third_order_correction",
            &dense.third_order_correction,
            &bundled.third_order_correction,
        ),
    ] {
        assert_eq!(d.len(), b.len(), "{label} length");
        for (i, (dv, bv)) in d.iter().zip(b.iter()).enumerate() {
            assert!(
                dv.is_finite() && bv.is_finite(),
                "{label} coordinate {i} must be finite (dense={dv}, bundled={bv})"
            );
            assert_abs_diff_eq!(dv, bv, epsilon = 1.0e-9);
        }
    }
    let bundled_gradient = bundled.gradient();
    assert_eq!(dense_gradient.len(), bundled_gradient.len());
    let worst = dense_gradient
        .iter()
        .zip(bundled_gradient.iter())
        .map(|(d, b)| (d - b).abs())
        .fold(0.0_f64, f64::max);
    eprintln!(
        "[#2515 ROUTE-PARITY] complete gradient max|Δ|={worst:.6e}  (majorizer-rooted \
         carrier control: {miswired_gap:.6e}, clamped rows: {clamped_rows})"
    );
    for (d, b) in dense_gradient.iter().zip(bundled_gradient.iter()) {
        assert_abs_diff_eq!(d, b, epsilon = 1.0e-9);
    }

    // Non-triviality of the channel under test: the routed trace must be a
    // genuine, nonzero contribution so the parity assertion is not vacuous.
    let trace_sq: f64 = dense.logdet_trace.iter().map(|v| v * v).sum();
    assert!(
        trace_sq > 0.0 && trace_sq.is_finite(),
        "the routed log|H|-trace channel must be non-trivial; ‖logdet_trace‖²={trace_sq}"
    );
}

/// #2515/#2712 — THE MEASUREMENT THAT KEEPS THE STREAMING DEFLATION REFUSAL
/// ALIVE, and the one that will tell the next person when to lift it.
///
/// `penalized_quasi_laplace_streaming_outer_evaluation` refuses any evaluation
/// whose cache carries a per-row spectral deflation direction. Its stated reason
/// was #2515 — "the complete matrix-free outer gradient still desyncs from the
/// dense one through the #2515 beta-Schur smoothness channel, whose gap is
/// deflation-independent and unfixed" — with the note that lifting it needs
/// "#2515 first, then a measured parity gate on a streaming-lane evaluation whose
/// cache actually deflates. Argument alone is what put the wrong reason on it in
/// the first place."
///
/// #2515 is now first, and this is that measurement, on #2712's own certified
/// deflated anchor. The result says the refusal must STAY, for a reason that is
/// no longer #2515's:
///
/// ```text
/// majorizer cache deflated rows = 10, exact-A cache deflated rows = 10
/// exact-A arrow factorization REFUSED under positive-definite evidence
///                             OK under unit deflation (the production policy)
/// complete gradient max|Δ| = 9.131537e0 against ‖g‖∞ = 5.004339e0
/// ```
///
/// On a NON-deflating state the same comparison is `1.57e-14`
/// ([`laplace_value_and_gradient_are_route_invariant_2515`]), so what is left is
/// a DEFLATION-PRICING disagreement, not an operator one: the dense route floors
/// the spectrum of the materialized `A` globally (with #2336 clamp-attributable
/// pricing on negative directions), while the arrow route conditions each row
/// block and pseudo-inverts the reduced Schur. Two floors, two metrics, one `A` —
/// the #2673 genus.
///
/// THIS TEST ASSERTS THE GAP IS STILL LARGE. That is deliberate. A refusal whose
/// justification is a number needs that number under test, or it decays into
/// folklore exactly as the #2515 reason did. When someone reconciles the two
/// deflation prices this test goes RED, and its message says to lift the
/// production gate rather than to loosen the bar.
#[test]
fn exact_a_route_parity_still_fails_on_a_deflated_cache_2515() {
    let (mut term, rho, target, b_cache) =
        super::tests_deflated_from_probes_2712::residual_excited_deflated_anchor(
            "#2515 exact-A parity on the #2712 deflated anchor",
        );
    let deflated_rows = b_cache
        .deflated_row_directions
        .iter()
        .filter(|d| !d.is_empty())
        .count();
    println!("[#2515 DEFLATED] majorizer cache deflated rows = {deflated_rows}");
    let loss = match term.loss(target.view(), &rho) {
        Ok(loss) => loss,
        Err(err) => {
            println!("[#2515 DEFLATED] loss unavailable: {err}");
            return;
        }
    };
    let sys = match term.assemble_arrow_schur(target.view(), &rho, None) {
        Ok(sys) => sys,
        Err(err) => {
            println!("[#2515 DEFLATED] arrow assembly refused: {err}");
            return;
        }
    };
    let a_sys = match term.exact_a_evidence_system(target.view(), &rho, &sys) {
        Ok(sys) => sys,
        Err(err) => {
            println!("[#2515 DEFLATED] exact-A assembly refused: {err}");
            return;
        }
    };
    // The PD-evidence policy refuses an indefinite reduced Schur outright. The
    // PRODUCTION evidence policy is unit deflation at the canonical spectral
    // floor, which pseudo-inverts across the null/negative directions instead —
    // that is the policy the streaming lane runs and therefore the one a parity
    // statement about it has to use.
    let mut a_cache = None;
    for (label, options) in [
        (
            "positive-definite",
            ArrowSolveOptions::direct().with_positive_definite_evidence(),
        ),
        (
            "unit-deflation (production evidence policy)",
            ArrowSolveOptions::direct()
                .with_newton_schur_tikhonov(gam_solve::arrow_schur::SPECTRAL_DEFLATION_REL_FLOOR)
                .with_evidence_unit_deflation(
                    gam_solve::arrow_schur::SPECTRAL_DEFLATION_REL_FLOOR,
                ),
        ),
    ] {
        match solve_arrow_newton_step_with_options(&a_sys, 0.0, 0.0, &options) {
            Ok((_, _, cache)) => {
                println!("[#2515 DEFLATED] exact-A arrow factorization OK under {label}");
                a_cache = Some(cache);
                break;
            }
            Err(err) => println!(
                "[#2515 DEFLATED] exact-A arrow factorization refused under {label}: {err:?}"
            ),
        }
    }
    let Some(a_cache) = a_cache else {
        return;
    };
    println!(
        "[#2515 DEFLATED] exact-A cache deflated rows = {}  fingerprints differ = {}",
        a_cache
            .deflated_row_directions
            .iter()
            .filter(|d| !d.is_empty())
            .count(),
        a_cache.row_hessian_fingerprint != b_cache.row_hessian_fingerprint,
    );
    let b_solver = DeflatedArrowSolver::plain(&b_cache);
    let dense = match term.analytic_outer_rho_gradient_components(
        target.view(),
        &rho,
        &loss,
        &b_cache,
        &b_solver,
    ) {
        Ok(components) => components.gradient(),
        Err(err) => {
            println!("[#2515 DEFLATED] dense exact-A gradient refused: {err}");
            return;
        }
    };
    let (a_probes, a_sinv) = full_basis_probe_bundle(&a_cache);
    let bundled = match term.analytic_outer_rho_gradient_components_with_bundle(
        target.view(),
        &rho,
        &loss,
        &b_cache,
        &b_solver,
        Some(BundleEvidenceGeometry {
            operator: EvidenceOperator::ExactObservedInformation,
            cache: &a_cache,
            probes: &a_probes,
            sinv: &a_sinv,
        }),
        None,
    ) {
        Ok(components) => components.gradient(),
        Err(err) => {
            println!("[#2515 DEFLATED] exact-A bundle gradient refused: {err}");
            return;
        }
    };
    let mut worst = 0.0_f64;
    let mut scale = 0.0_f64;
    for i in 0..dense.len() {
        worst = worst.max((dense[i] - bundled[i]).abs());
        scale = scale.max(dense[i].abs());
        println!(
            "[#2515 DEFLATED] coord {i}: dense={:+.10e} bundle={:+.10e} |Δ|={:.3e}",
            dense[i],
            bundled[i],
            (dense[i] - bundled[i]).abs()
        );
    }
    println!(
        "[#2515 DEFLATED] complete gradient max|Δ|={worst:.6e} against ‖g‖∞={scale:.6e} \
         (relative {:.6e})",
        worst / scale.max(f64::MIN_POSITIVE)
    );
    assert!(
        deflated_rows > 0,
        "#2515/#2712: this anchor must actually deflate, or the measurement is about \
         nothing; certified regime promised SomeRowDeflates"
    );
    assert!(
        scale.is_finite() && scale > 1.0e-6,
        "#2515/#2712: the gradient must be non-trivial for a relative gap to mean \
         anything; ‖g‖∞={scale:.6e}"
    );
    assert!(
        worst > 1.0e-2 * scale,
        "#2515/#2712: the exact-A bundle route now AGREES with the dense exact-A route \
         on a deflating cache (max|Δ|={worst:.6e} against ‖g‖∞={scale:.6e}; it was \
         9.131537e0 against 5.004339e0 when this was written). That is the condition \
         `penalized_quasi_laplace_streaming_outer_evaluation`'s spectral-deflation \
         refusal is waiting on. LIFT THAT GATE and delete this assertion — do not \
         loosen this bar."
    );
}

/// MEASUREMENT — attribute the deflating-anchor route gap to the
/// CLASSIFICATION, not to a channel.
///
/// `exact_a_route_parity_still_fails_on_a_deflated_cache_2515` establishes that
/// the two routes disagree by `9.13` against `‖g‖∞ = 5.00` once a cache
/// deflates, and the production streaming gate refuses on that number. It does
/// not say WHICH directions the two routes classify differently, and the repair
/// is a different one for each answer. This probe prints both classifications
/// direction by direction.
///
/// The two rules under comparison, both on the SAME `A`:
///
/// * DENSE (`ExactHessianSpectralBlock::rank_floor`, #2673) — the null band of
///   an eigendirection `v` is `max(dim·ε·‖A‖₂, √ε·vᵀBv)`, i.e. the pencil
///   curvature against the majorizer metric the gradient path also uses; a
///   negative direction is priced at its ARD-clamp basin `λ+vᵀEv` (#2336) and
///   only a basin below `−floor` refuses.
/// * ARROW (`factor_spectral_deflated_criterion_row`,
///   `factor_evidence_unit_deflated_schur`) — the band is
///   `SPECTRAL_DEFLATION_REL_FLOOR·max|λ|` of the block ALONE, which sees
///   neither `B` nor the rest of the operator, and EVERY non-positive
///   eigenvalue is unit-pinned regardless of what the clamp explains.
///
/// So there are two independent disagreements — the BAND and the SIGN — and
/// which of them carries the `9.13` decides whether the repair is a metric or a
/// pricing rule.
#[test]
fn zz_attribute_deflated_route_classification_2515() {
    let (mut term, rho, target, b_cache) =
        super::tests_deflated_from_probes_2712::residual_excited_deflated_anchor(
            "#2515 deflation-pricing attribution",
        );
    let total_t = b_cache.delta_t_len();
    let k = b_cache.k;
    println!("[#2515 CLASSIFY] total_t={total_t} k={k} rows={}", b_cache.n_rows());

    let a = term
        .materialize_exact_hessian_dense(&rho, target.view(), &b_cache)
        .expect("the anchor's exact Hessian materializes");
    let e_diag = term
        .materialize_ard_concave_clamp_diagonal(&rho, &b_cache)
        .expect("the anchor's ARD concave clamp diagonal is available");

    for (label, block, metric) in [
        ("joint", a.clone(), ArrowMetric::Joint(&b_cache)),
        (
            "coordinate",
            a.slice(s![..total_t, ..total_t]).to_owned(),
            ArrowMetric::Coordinate(&b_cache),
        ),
    ] {
        let (evals, evecs) =
            gam_linalg::faer_ndarray::FaerEigh::eigh(&block, faer::Side::Lower)
                .expect("a symmetric block diagonalizes");
        let spectral_norm = evals.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()));
        let mut pinned = 0usize;
        let mut priced_negative = 0usize;
        let mut log_det = 0.0_f64;
        for idx in 0..evals.len() {
            let lambda = evals[idx];
            let v = evecs.column(idx);
            let bvv = metric
                .quadratic_form(v)
                .expect("the majorizer metric is defined on every direction");
            let floor = sae_exact_a_direction_floor(evals.len(), spectral_norm, bvv);
            let e_v: f64 = (0..total_t.min(v.len()))
                .map(|j| e_diag[j] * v[j] * v[j])
                .sum();
            let priced = if lambda < -floor {
                priced_negative += 1;
                lambda + e_v
            } else {
                lambda
            };
            if priced > floor {
                log_det += priced.ln();
            } else {
                pinned += 1;
                println!(
                    "[#2515 CLASSIFY] dense {label} dir {idx}: lambda={lambda:+.6e} \
                     floor={floor:.6e} v'Bv={bvv:.6e} v'Ev={e_v:.6e} priced={priced:+.6e} PINNED"
                );
            }
        }
        println!(
            "[#2515 CLASSIFY] dense {label}: dim={} ||A||2={spectral_norm:.6e} pinned={pinned} \
             clamp-priced-negative={priced_negative} log_det={log_det:.10e}",
            evals.len()
        );
    }

    let sys = term
        .assemble_arrow_schur(target.view(), &rho, None)
        .expect("the anchor assembles");
    let a_sys = term
        .exact_a_evidence_system(target.view(), &rho, &sys)
        .expect("the anchor's exact-A evidence system builds");
    let options = ArrowSolveOptions::direct()
        .with_newton_schur_tikhonov(gam_solve::arrow_schur::SPECTRAL_DEFLATION_REL_FLOOR)
        .with_evidence_unit_deflation(gam_solve::arrow_schur::SPECTRAL_DEFLATION_REL_FLOOR);
    let (_, _, a_cache) = solve_arrow_newton_step_with_options(&a_sys, 0.0, 0.0, &options)
        .expect("the production evidence policy factors the anchor's exact-A system");

    let mut arrow_row_log_det = 0.0_f64;
    let mut arrow_pinned = 0usize;
    for row in 0..a_cache.n_rows() {
        let Some(spectrum) = a_cache.deflation_row_spectra.get(row).and_then(Option::as_ref)
        else {
            continue;
        };
        let a_row = &a_sys.rows[row].htt;
        let b_row = &sys.rows[row].htt;
        let norm = spectrum
            .raw_evals
            .iter()
            .fold(0.0_f64, |acc, &v| acc.max(v.abs()));
        for idx in 0..spectrum.raw_evals.len() {
            let lambda = spectrum.raw_evals[idx];
            let v = spectrum.evecs.column(idx);
            let bvv = v.dot(&b_row.dot(&v));
            let avv = v.dot(&a_row.dot(&v));
            let pencil_floor = sae_exact_a_direction_floor(spectrum.raw_evals.len(), norm, bvv);
            let base = a_cache.row_offsets[row];
            let e_v: f64 = (0..v.len()).map(|j| e_diag[base + j] * v[j] * v[j]).sum();
            let arrow_floor = gam_solve::arrow_schur::SPECTRAL_DEFLATION_REL_FLOOR * norm;
            let conditioned = spectrum.cond_evals[idx];
            if conditioned > 0.0 {
                arrow_row_log_det += conditioned.ln();
            }
            let deflated = matches!(
                spectrum.conditioning[idx],
                gam_solve::arrow_schur::RowSpectralConditioning::UnitDeflated
            );
            if deflated {
                arrow_pinned += 1;
            }
            println!(
                "[#2515 CLASSIFY] arrow row {row} dir {idx}: lambda={lambda:+.6e} \
                 cond={conditioned:+.6e} arrow_floor={arrow_floor:.6e} \
                 pencil_floor={pencil_floor:.6e} v'Bv={bvv:.6e} v'Av={avv:+.6e} \
                 v'Ev={e_v:.6e} basin={:+.6e} deflated={deflated} \
                 pencil_would_pin={}",
                lambda + e_v,
                (if lambda < -pencil_floor { lambda + e_v } else { lambda }) <= pencil_floor
            );
        }
    }
    println!(
        "[#2515 CLASSIFY] arrow rows: pinned={arrow_pinned} \
         sum_log_cond_row={arrow_row_log_det:.10e}"
    );
    match a_cache.beta_schur_deflation.as_ref() {
        Some(spectrum) => {
            for idx in 0..spectrum.raw_evals.len() {
                println!(
                    "[#2515 CLASSIFY] arrow schur dir {idx}: lambda={:+.6e} cond={:+.6e} \
                     deflated={}",
                    spectrum.raw_evals[idx], spectrum.cond_evals[idx], spectrum.deflated[idx]
                );
            }
        }
        None => println!("[#2515 CLASSIFY] arrow schur: no deflation recorded"),
    }
}
