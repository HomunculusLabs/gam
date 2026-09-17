//! #2267 — a chart gauge the joint fit installs as a β quotient must be a null of
//! the whole penalized objective, not only of the reconstruction.
//!
//! `run_joint_fit_arrow_schur` installs
//! [`SaeManifoldTerm::closed_form_beta_gauge_directions`] as an
//! `ArrowBetaGaugeQuotient`: the step solves `P S_β P + Q Qᵀ` and projects the
//! right-hand side off `Q`. A declared direction along which some term still has
//! slope is gradient no step can reduce. On #2267's k2 fit the declared gauge
//! carried |g_β| = 440.8 against |P g_β| = 0.714, and the fit died on "adaptive
//! proximal correction failed" (job 617095).
//!
//! The sweep covers every seedable kind that emits a chart orbit, with coordinate
//! ARD off and on, on a two-atom term so the separation barrier and the decoder
//! repulsion are live. The planted circle gives the torus fewer active rows than
//! coefficients, so a planted two-angle torus with more rows than coefficients is
//! swept as well. Each generator's slope is measured through the value functions
//! the line search descends (`directional_derivative_terms`, a central difference),
//! so it cannot pass by an analytic gradient agreeing with itself. The test asserts:
//!
//! * every generator whose β part lies in the declared span is flat for the
//!   penalized objective, at or below the inner convergence tolerance;
//! * the harmonic families with ARD off and a full-rank compensation design still
//!   declare their gauge, so the quotient is narrowed to the true nulls rather than
//!   switched off, and the planted torus is one of them;
//! * a harmonic family with ARD off on a rank-deficient compensation design has a
//!   live generator. The least-squares compensation there is the minimum-norm one,
//!   not the coefficient rotation, so the rank condition is load-bearing (job
//!   1102636 measured the undeclared-by-rank torus, when it was still declared, at
//!   2148x to 8840x the tolerance through the smoothing);
//! * the same instrument reads a slope above tolerance on some undeclared
//!   generator, so the silence on the declared ones is a measurement.

// `manifold/mod.rs` declares this module as
// `#[cfg(test)] mod tests_chart_gauge_structural_null_2267;`.
#![cfg(test)]
use super::*;
use crate::manifold::tests_gauge_posterior_flatness_2720::{
    GAUGE_SWEEP_KINDS, active_rows_and_basis_size, directional_derivative_terms,
    planted_circle_cloud, unit_norm,
};

/// Rows of the planted torus: twice the 7×7 tensor harmonic basis (M = 49) the
/// torus seed builds on the planted circle, where 42 rows leave its compensation
/// design rank-deficient.
const PLANTED_TORUS_ROWS: usize = 98;

/// A planted two-angle torus in the planted circle's ambient dimension, from the
/// same LCG and noise level: `z = cos θ₁ b₀ + sin θ₁ b₁ + cos θ₂ b₂ + sin θ₂ b₃`.
fn planted_torus_cloud(n: usize) -> Array2<f64> {
    let p = 48usize;
    let mut state = 0x2468_ace0_1357_9bdfu64;
    let mut unit = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64) / ((1u64 << 53) as f64)
    };
    let two_pi = std::f64::consts::TAU;
    let loadings: Vec<Vec<f64>> = (0..4)
        .map(|_| (0..p).map(|_| 2.0 * unit() - 1.0).collect())
        .collect();
    let mut z = Array2::<f64>::zeros((n, p));
    for i in 0..n {
        let first = two_pi * unit();
        let second = two_pi * unit();
        let harmonics = [first.cos(), first.sin(), second.cos(), second.sin()];
        for j in 0..p {
            let noise = 0.01 * (2.0 * unit() - 1.0);
            z[[i, j]] = (0..4)
                .map(|column| harmonics[column] * loadings[column][j])
                .sum::<f64>()
                + noise;
        }
    }
    z
}

/// A seeded two-atom term on `basis`, at latent dimension `latent_dim`. Two atoms
/// make the separation barrier and the decoder repulsion part of the objective.
fn seeded_pair_of_kind(
    target: ArrayView2<'_, f64>,
    basis: &str,
    latent_dim: usize,
) -> SaeManifoldTerm {
    let minimal = build_sae_minimal_seed(SaeMinimalSeedRequest {
        target,
        atom_basis: vec![basis.to_string(); 2],
        atom_dim: vec![latent_dim; 2],
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
        fit_config: SaeFitConfig::default(),
        temperature_schedule: None,
        fisher_metric: None,
        row_loss_weights: None,
        registry: &registry,
    })
    .expect("fit seed");
    seed.base_term
}

/// Modified Gram–Schmidt over the declared β directions.
fn orthonormal_span(vectors: Vec<Array1<f64>>) -> Vec<Array1<f64>> {
    let mut basis: Vec<Array1<f64>> = Vec::new();
    for mut vector in vectors {
        for existing in &basis {
            let coefficient = vector.dot(existing);
            vector.scaled_add(-coefficient, existing);
        }
        if let Some(unit) = unit_norm(vector) {
            basis.push(unit);
        }
    }
    basis
}

#[test]
fn declared_chart_gauges_are_nulls_of_the_penalized_objective_2267() {
    let circle = planted_circle_cloud();
    let torus = planted_torus_cloud(PLANTED_TORUS_ROWS);
    let cells = GAUGE_SWEEP_KINDS
        .iter()
        .map(|&(kind, latent_dim)| (kind, latent_dim, "circle", &circle))
        .chain(std::iter::once(("torus", 2usize, "torus", &torus)));
    let registry = AnalyticPenaltyRegistry::new();
    let mut violations: Vec<String> = Vec::new();
    let mut declared_measured = 0usize;
    let mut undeclared_live = 0usize;
    let mut rank_deficient_harmonic_live = 0usize;
    let mut planted_torus_declared = 0usize;
    let mut harmonic_without_declaration: Vec<String> = Vec::new();
    for (kind, latent_dim, cloud, z) in cells {
        for ard in [false, true] {
            let mut term = seeded_pair_of_kind(z.view(), kind, latent_dim);
            // The assembly chokepoint freezes these gates before every value the
            // line search reads; the sweep reads the same frozen gates.
            term.refresh_decoder_repulsion_gate();
            term.refresh_barrier_coactivation_gate();
            term.refresh_amplitude_barrier_gate();
            let log_ard: Vec<Array1<f64>> = (0..term.k_atoms())
                .map(|atom| {
                    let axes = if ard {
                        term.assignment.coords[atom].latent_dim()
                    } else {
                        0
                    };
                    Array1::<f64>::zeros(axes)
                })
                .collect();
            let rho = SaeManifoldRho::new(0.0, 0.0, log_ard);
            let tolerance = SAE_MANIFOLD_INNER_GRAD_REL_TOL * term.inner_iterate_scale();
            let declared = orthonormal_span(
                term.closed_form_beta_gauge_directions(&rho, Some(&registry))
                    .expect("closed-form gauge directions"),
            );
            let harmonic = matches!(
                term.atoms[0].basis_kind(),
                SaeAtomBasisKind::Periodic
                    | SaeAtomBasisKind::Torus
                    | SaeAtomBasisKind::Sphere
                    | SaeAtomBasisKind::ProjectivePlane
                    | SaeAtomBasisKind::KleinBottle
                    | SaeAtomBasisKind::Cylinder
            );
            let mut full_rank = Vec::with_capacity(term.k_atoms());
            for atom in 0..term.k_atoms() {
                full_rank.push(
                    term.atom_compensation_has_full_column_rank(atom)
                        .expect("compensation design rank"),
                );
            }
            let all_full_rank = full_rank.iter().all(|&exact| exact);
            if harmonic && !ard && all_full_rank && declared.is_empty() {
                harmonic_without_declaration.push(format!("{kind} on the planted {cloud}"));
            }
            if cloud == "torus" && !ard && all_full_rank && !declared.is_empty() {
                planted_torus_declared += 1;
            }
            let (active, basis_size) =
                active_rows_and_basis_size(&term, 0).expect("active rows");
            let coord_len = term.n_obs() * term.assignment.row_block_dim();
            let gauges = term.dense_step_gauge_vectors().expect("gauge vectors");
            println!(
                "\n[2267-null] kind={kind} cloud={cloud} ard={ard} n_active={active} \
                 M={basis_size} full_rank={full_rank:?} generators={} declared_span={} \
                 tol={tolerance:.6e} separation={:.6e} repulsion={:.6e} amplitude={:.6e}",
                gauges.len(),
                declared.len(),
                term.separation_barrier_value(1.0),
                term.decoder_repulsion_value(1.0),
                term.amplitude_barrier_value(1.0),
            );
            for (index, gauge) in gauges.into_iter().enumerate() {
                let beta = gauge.slice(s![coord_len..]).to_owned();
                let beta_norm = beta.dot(&beta).sqrt();
                let mut residual = beta;
                for basis in &declared {
                    let coefficient = residual.dot(basis);
                    residual.scaled_add(-coefficient, basis);
                }
                // Each atom's generators live on that atom's β block, so a β part
                // is either inside the declared span (residual at roundoff) or on a
                // block the span never touches (residual = ‖β‖). Any bar strictly
                // between the two separates them.
                let is_declared =
                    beta_norm > 0.0 && residual.dot(&residual).sqrt() <= 0.5 * beta_norm;
                let Some(direction) = unit_norm(gauge) else {
                    continue;
                };
                let slope = directional_derivative_terms(
                    &mut term,
                    z.view(),
                    &rho,
                    &registry,
                    &direction,
                    1.0e-5,
                )
                .expect("directional derivative");
                let total = slope.total().abs();
                let barriers = slope.analytic
                    + slope.repulsion
                    + slope.amplitude_barrier
                    + slope.separation_barrier;
                println!(
                    "[2267-null]   gen {index:>2} declared={is_declared} |d f|={total:.4e} \
                     ({:.3}x tol) data_fit={:.3e} sparsity={:.3e} smooth={:.3e} ard={:.3e} \
                     barriers={barriers:.3e}",
                    total / tolerance,
                    slope.data_fit,
                    slope.assignment_sparsity,
                    slope.smoothness,
                    slope.ard,
                );
                if is_declared {
                    declared_measured += 1;
                    if total > tolerance {
                        violations.push(format!(
                            "{kind} on the planted {cloud} ard={ard} generator {index}: \
                             |d f| = {total:.6e} = {:.3}x the convergence tolerance \
                             {tolerance:.6e} (data_fit {:.3e}, smoothness {:.3e}, ard {:.3e}, \
                             barriers {barriers:.3e})",
                            total / tolerance,
                            slope.data_fit,
                            slope.smoothness,
                            slope.ard,
                        ));
                    }
                } else if total > tolerance {
                    undeclared_live += 1;
                    if harmonic && !ard && !all_full_rank {
                        rank_deficient_harmonic_live += 1;
                    }
                }
            }
        }
    }
    assert!(
        declared_measured > 0,
        "no declared generator was measured on any fixture, so the null claim went untested"
    );
    assert!(
        undeclared_live > 0,
        "no undeclared generator carried slope above tolerance, so the instrument never showed \
         it can see a live orbit and its silence on the declared set proves nothing"
    );
    assert!(
        rank_deficient_harmonic_live > 0,
        "no harmonic family with ARD off on a rank-deficient compensation design carried a live \
         generator, so the sweep never showed the full-column-rank condition is load-bearing"
    );
    assert!(
        planted_torus_declared > 0,
        "the planted torus never reached a full-rank compensation design with its gauge declared \
         at ARD off, so the torus's null claim went untested"
    );
    assert!(
        harmonic_without_declaration.is_empty(),
        "harmonic families with coordinate ARD off and a full-rank compensation design declared \
         no gauge: {harmonic_without_declaration:?}. Their phase and Killing generators are nulls \
         of every term there, so the quotient must still pin them"
    );
    assert!(
        violations.is_empty(),
        "the joint fit's β quotient declares generators the penalized objective is not flat \
         along; the step projects that gradient out and can never reduce it:\n  {}",
        violations.join("\n  "),
    );
}
