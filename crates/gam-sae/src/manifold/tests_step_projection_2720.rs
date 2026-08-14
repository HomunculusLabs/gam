//! The decisive probe for the #2720 root-cause claim: is the solver's step
//! actually gauge-projected at a stalling state?
//!
//! ## Background
//!
//! Posted on #2720: "the quotient solver projects in-orbit gradient out of
//! the step, then its raw-KKT acceptance gate demands that same gradient
//! vanish." If TRUE, the modelling fix (make the orbit a true posterior
//! symmetry) unblocks the solver. If FALSE — if `pin_reduced_schur`'s unit-
//! stiffness damping lets the step move in-orbit — a correction is owed.
//!
//! ## Method
//!
//! Drive a periodic term to its native-target stall (the probe-B refusal:
//! ‖g‖ = 9.43e-3 vs tol 7.85e-5, orthogonal-to-orbit component 9.428e-3 of
//! 9.428e-3 — 99.996% of the gradient lies along the gauge orbit). At that
//! state:
//!
//! 1. Assemble the Arrow-Schur system, take the gauge basis `Q`.
//! 2. Solve the Newton step via the SAME entry the solver uses
//!    (`solve_arrow_newton_step_with_options`).
//! 3. Project: ‖QᵀΔ‖ (in-orbit) vs ‖Δ‖. Also the gradient split
//!    ‖Qᵀg‖ / ‖g‖ and the projected gradient-step alignment gᵀΔ.
//! 4. If ‖QᵀΔ‖/‖Δ‖ ≈ 0 the step is projected — claim holds. Report
//!    honestly either way.
//!
//! #[ignore]d manual diagnostic, same convention.

#![cfg(test)]
use super::*;
use crate::manifold::tests_gauge_frame_roundtrip_2720::planted_circle_cloud;

/// Seed a one-atom term of the given kind (Fixture-B path), ARD-saddle rho.
fn seeded_native(kind: &str) -> (SaeManifoldTerm, SaeManifoldRho, Array2<f64>) {
    let z = planted_circle_cloud();
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
    .unwrap_or_else(|e| panic!("[step-probe] minimal seed failed for {kind}: {e}"));
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
    .unwrap_or_else(|e| panic!("[step-probe] fit seed failed for {kind}: {e}"));
    let mut rho = seed.initial_rho;
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

/// The native target: the atom's own reconstruction at its seed state.
fn native_target(term: &SaeManifoldTerm) -> Array2<f64> {
    let phi = term.atoms[0].basis_values.view();
    let decoder = term.atoms[0].decoder_coefficients();
    phi.dot(decoder)
}

/// H·v under the joint [t (per-row); β] packing, from the arrow blocks:
/// row i: (H_tt δt)_i + H_tβ δβ ; border: Σ_i H_βt δt_i + H_ββ δβ.
/// If this and the solver agree (‖HΔ+g‖ small), the packing is consistent.
fn joint_h_matvec(sys: &ArrowSchurSystem, v: ArrayView1<'_, f64>) -> Array1<f64> {
    let border_dim = sys.gb.len();
    let coord_dims: Vec<usize> = sys.rows.iter().map(|r| r.gt.len()).collect();
    let total_t: usize = coord_dims.iter().sum();
    assert_eq!(v.len(), total_t + border_dim, "joint vector length");
    let mut out = Array1::<f64>::zeros(total_t + border_dim);
    let mut off = 0usize;
    for (row_idx, row) in sys.rows.iter().enumerate() {
        let d = coord_dims[row_idx];
        let dt = v.slice(s![off..off + d]);
        // (H_tt δt)_i
        let mut htt_dt = row.htt.dot(&dt);
        // + H_tβ δβ
        if !row.htbeta.is_empty() && border_dim > 0 {
            let db = v.slice(s![total_t..]);
            htt_dt.scaled_add(1.0, &row.htbeta.dot(&db));
        }
        for (i, &val) in htt_dt.iter().enumerate() {
            out[off + i] = val;
        }
        off += d;
    }
    // border: Σ_i H_βt δt_i + H_ββ δβ
    if border_dim > 0 {
        let db = v.slice(s![total_t..]);
        let mut hbb_db = if sys.hbb.is_empty() {
            Array1::zeros(border_dim)
        } else {
            sys.hbb.dot(&db)
        };
        let mut off2 = 0usize;
        for (row_idx, row) in sys.rows.iter().enumerate() {
            let d = coord_dims[row_idx];
            if !row.htbeta.is_empty() {
                let dt = v.slice(s![off2..off2 + d]);
                hbb_db.scaled_add(1.0, &(row.htbeta.t().dot(&dt)));
            }
            off2 += d;
        }
        for (i, &val) in hbb_db.iter().enumerate() {
            out[total_t + i] = val;
        }
    }
    out
}

#[test]
#[ignore = "manual diagnostic; run with --ignored --nocapture"]
fn step_projection_at_poincare_circle_stall_2720() {
    // The REAL stall: poincare atom against the circle cloud (its refusal is
    // robust — reproduced across every run and a 10x budget in #2772).
    // Measure the step AT the stalled state, then, to see what the solver
    // COULD do there, also after a no-op refinement that keeps the state.
    use crate::manifold::tests_gauge_frame_roundtrip_2720::planted_circle_cloud;
    let z = planted_circle_cloud();
    let minimal = build_sae_minimal_seed(SaeMinimalSeedRequest {
        target: z.view(),
        atom_basis: vec!["poincare".to_string()],
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
    .expect("[step-probe] poincare minimal seed");
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
    .expect("[step-probe] poincare fit seed");
    let mut term = seed.base_term;
    let mut rho = seed.initial_rho;
    rho.log_lambda_sparse = -0.5;
    for value in rho.log_lambda_smooth.iter_mut() {
        *value = -1.0;
    }
    for axis in rho.log_ard.iter_mut() {
        for value in axis.iter_mut() {
            *value = -0.5;
        }
    }
    for atom in term.atoms.iter_mut() {
        atom.deactivate_decoder_frame();
    }

    let outcome = term.penalized_quasi_laplace_criterion_with_cache(
        z.view(),
        &rho,
        None,
        40,
        0.4,
        1.0e-6,
        1.0e-6,
    );
    match &outcome {
        Ok(_) => eprintln!(
            "[step-probe] UNEXPECTED: poincare/circle converged this run — the robust refusal did not reproduce"
        ),
        Err(e) => eprintln!(
            "[step-probe] stall reproduced: {}",
            e.to_string().chars().take(220).collect::<String>()
        ),
    }

    // Measure at whatever state that attempt left behind.
    let sys = term
        .assemble_arrow_schur(z.view(), &rho, None)
        .expect("[step-probe] assemble");
    let n_rows = sys.rows.len();
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
    let basis = term
        .joint_chart_gauge_basis_for_arrow_layout(
            &row_offsets,
            border_dim,
            "step-probe poincare/circle",
        )
        .expect("[step-probe] gauge basis");
    let g_norm = grad.dot(&grad).sqrt();
    let mut g_in_sq = 0.0f64;
    for v in basis.iter() {
        g_in_sq += grad.dot(v).powi(2);
    }
    let g_in_norm = g_in_sq.sqrt();
    eprintln!(
        "[step-probe] poincare/circle state: ‖g‖={g_norm:.6e}, in-orbit ‖Qᵀg‖={g_in_norm:.6e} \
         ({:.4} of ‖g‖), {} gauge dirs",
        if g_norm > 0.0 {
            g_in_norm / g_norm
        } else {
            f64::NAN
        },
        basis.len()
    );

    let options = ArrowSolveOptions::automatic(n_rows);
    match solve_arrow_newton_step_with_options(&sys, 0.0, 0.0, &options) {
        Ok((delta_t, delta_beta, _cache)) => {
            let coord_dim = sys.rows.first().map(|r| r.gt.len()).unwrap_or(0);
            assert_eq!(
                delta_t.len(),
                coord_dim * n_rows,
                "[step-probe] delta_t layout"
            );
            let mut delta = Array1::<f64>::zeros(full_len);
            for (row_idx, chunk) in delta_t.exact_chunks(coord_dim).into_iter().enumerate() {
                for (i, &v) in chunk.iter().enumerate() {
                    delta[row_idx * coord_dim + i] = v;
                }
            }
            for (i, &v) in delta_beta.iter().enumerate() {
                delta[coord_dim * n_rows + i] = v;
            }

            // ==== ORACLE P1 checks: joint-packed-layout consistency ====
            // (a) Newton residual ‖HΔ+g‖/‖g‖ ≈ 0 validates that H, g, and Δ
            //     share one ordering — layout, gradient, and step in one eq.
            let h_delta = joint_h_matvec(&sys, delta.view());
            let residual = &h_delta + &grad;
            let res_rel = residual.dot(&residual).sqrt() / grad.dot(&grad).sqrt();
            eprintln!(
                "[step-probe] poincare/circle Newton residual ‖HΔ+g‖/‖g‖ = {res_rel:.3e} \
                 (validates joint packing of H, g, Δ)"
            );
            // (b) Gauge-basis orthonormality QᵀQ ≈ I; without it ‖Qᵀg‖ is
            //     not a projection norm.
            for (i, vi) in basis.iter().enumerate() {
                for (j, vj) in basis.iter().enumerate().skip(i) {
                    let gram = vi.dot(vj);
                    let want = if i == j { 1.0 } else { 0.0 };
                    assert!(
                        (gram - want).abs() < 1.0e-10,
                        "[step-probe] gauge Gram QᵀQ[{i}][{j}]={gram:.3e} ≠ {want}"
                    );
                }
            }
            eprintln!(
                "[step-probe] gauge Gram QᵀQ ≈ I verified ({} dirs)",
                basis.len()
            );

            let d_norm = delta.dot(&delta).sqrt();
            let mut d_in_sq = 0.0f64;
            for v in basis.iter() {
                d_in_sq += delta.dot(v).powi(2);
            }
            let d_in_norm = d_in_sq.sqrt();
            let gd = grad.dot(&delta);
            let cos_theta = if d_norm * g_norm > 0.0 {
                gd / (d_norm * g_norm)
            } else {
                f64::NAN
            };
            let d_in_share = if d_norm > 0.0 {
                d_in_norm / d_norm
            } else {
                f64::NAN
            };
            eprintln!(
                "[step-probe] poincare/circle step: ‖Δ‖={d_norm:.6e}, in-orbit \
                 ‖QᵀΔ‖={d_in_norm:.6e} ({d_in_share:.6} of ‖Δ‖), gᵀΔ={gd:.6e}, \
                 cos θ={cos_theta:.6}"
            );
            if d_in_share < 1.0e-8 {
                eprintln!(
                    "[step-probe] VERDICT: step is gauge-PROJECTED at the stall — the original mechanism claim holds HERE"
                );
            } else {
                eprintln!(
                    "[step-probe] VERDICT: step carries in-orbit motion ({d_in_share:.4}) at the stall; projection claim does not hold here either"
                );
            }
        }
        Err(e) => eprintln!(
            "[step-probe] Newton step FAILED at the stalled state: {e:?} — the refusal path may fail before a step exists"
        ),
    }
}

#[test]
#[ignore = "manual diagnostic; run with --ignored --nocapture"]
fn step_projection_at_periodic_native_stall_2720() {
    let (mut term, rho, _z) = seeded_native("periodic");
    for atom in term.atoms.iter_mut() {
        atom.deactivate_decoder_frame();
    }
    let native = native_target(&term);

    // Drive toward the stall: run the solver until it refuses (the refusal IS
    // the stall state), then measure AT the state the refusal left behind.
    let outcome = term.penalized_quasi_laplace_criterion_with_cache(
        native.view(),
        &rho,
        None,
        40,
        0.4,
        1.0e-6,
        1.0e-6,
    );
    match &outcome {
        Ok(_) => eprintln!("[step-probe] NOTE: solve converged — stall not reproduced this run"),
        Err(e) => eprintln!(
            "[step-probe] stall reproduced: {}",
            e.to_string().chars().take(200).collect::<String>()
        ),
    }

    // Assemble at the (stalled) current state.
    let sys = term
        .assemble_arrow_schur(native.view(), &rho, None)
        .expect("[step-probe] assemble");
    let n_rows = sys.rows.len();
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

    let basis = term
        .joint_chart_gauge_basis_for_arrow_layout(
            &row_offsets,
            border_dim,
            "step-probe periodic/native",
        )
        .expect("[step-probe] gauge basis");
    assert!(
        !basis.is_empty(),
        "[step-probe] no gauge directions — instrument inapplicable"
    );
    let g_norm = grad.dot(&grad).sqrt();
    let mut g_in = 0.0f64;
    for v in basis.iter() {
        g_in += grad.dot(v).powi(2);
    }
    let g_in_norm = g_in.sqrt();
    eprintln!(
        "[step-probe] state: ‖g‖={g_norm:.6e}, in-orbit ‖Qᵀg‖={g_in_norm:.6e} \
         ({:.4} of ‖g‖), {} gauge dirs",
        g_in_norm / g_norm,
        basis.len()
    );

    // The solver's own entry, same options family the criterion uses.
    let options = ArrowSolveOptions::automatic(n_rows);
    let solve = solve_arrow_newton_step_with_options(&sys, 0.0, 0.0, &options);
    let (delta_t, delta_beta, _cache) = match solve {
        Ok(triple) => triple,
        Err(e) => {
            eprintln!(
                "[step-probe] Newton step FAILED at the stalled state: {e:?} — \
                 the refusal may fire before any step is available; measuring \
                 the system projection structure instead"
            );
            return;
        }
    };
    let mut delta = Array1::<f64>::zeros(full_len);
    // delta_t layout: per-row chunks of coord width; rebuild joint layout.
    let coord_dim = sys.rows.first().map(|r| r.gt.len()).unwrap_or(0);
    assert_eq!(
        delta_t.len(),
        coord_dim * n_rows,
        "[step-probe] delta_t length {} does not match {} rows × {} coords",
        delta_t.len(),
        n_rows,
        coord_dim
    );
    for (row_idx, chunk) in delta_t.exact_chunks(coord_dim).into_iter().enumerate() {
        for (i, &v) in chunk.iter().enumerate() {
            delta[row_idx * coord_dim + i] = v;
        }
    }
    for (i, &v) in delta_beta.iter().enumerate() {
        delta[coord_dim * n_rows + i] = v;
    }

    let d_norm = delta.dot(&delta).sqrt();
    let mut d_in = 0.0f64;
    for v in basis.iter() {
        d_in += delta.dot(v).powi(2);
    }
    let d_in_norm = d_in.sqrt();
    let gd = grad.dot(&delta);
    eprintln!(
        "[step-probe] step: ‖Δ‖={d_norm:.6e}, in-orbit ‖QᵀΔ‖={d_in_norm:.6e} \
         ({:.6} of ‖Δ‖), gᵀΔ={gd:.6e}",
        d_in_norm / d_norm
    );
    if d_in_norm / d_norm < 1.0e-8 {
        eprintln!(
            "[step-probe] VERDICT: step is gauge-PROJECTED (in-orbit share < 1e-8) — \
             the posted mechanism holds: the solver cannot follow the in-orbit gradient"
        );
    } else if d_in_norm / d_norm > 1.0e-2 {
        eprintln!(
            "[step-probe] VERDICT: step carries MATERIAL in-orbit motion ({:.4}) — \
             the posted mechanism is WRONG in this regime; pin_reduced_schur's unit-stiffness \
             damping permits in-orbit motion; correction owed",
            d_in_norm / d_norm
        );
    } else {
        eprintln!(
            "[step-probe] VERDICT: in-orbit step share {:.3e} — small but not projected; \
             interpret with the alignment gᵀΔ",
            d_in_norm / d_norm
        );
    }
}
