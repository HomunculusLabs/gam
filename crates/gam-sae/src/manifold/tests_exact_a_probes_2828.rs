//! Exact-A beta derivatives (#2828): independent differences of the exact prior
//! curvature and its majorization gap, and differences of the priced joint log
//! determinant with all routing gates frozen.
#![cfg(test)]
use super::*;
use crate::manifold::tests_sparse_curvature_operator_2500::threshold_gate_tiny_fixture;
use ndarray::Array2;

fn frozen_anchor_and_cache(
    term: &SaeManifoldTerm,
    target: &Array2<f64>,
    rho: &SaeManifoldRho,
) -> (SaeManifoldTerm, ArrowFactorCache) {
    let mut anchor = term.clone();
    let (_value, _loss, cache) = anchor
        .penalized_quasi_laplace_criterion_with_cache(
            target.view(),
            rho,
            None,
            0,
            0.4,
            1.0e-6,
            1.0e-6,
        )
        .expect("frozen-gate cache at the fixture state");
    anchor.streaming_gates_frozen = true;
    (anchor, cache)
}

/// `log|A|` at a state, with the gates pinned to the anchor's (the objective
/// whose Hessian `A` is differentiated holds them fixed).
fn frozen_exact_a_logdet(
    anchor: &SaeManifoldTerm,
    mut endpoint: SaeManifoldTerm,
    target: &Array2<f64>,
    rho: &SaeManifoldRho,
) -> Option<f64> {
    endpoint.decoder_repulsion_gate = anchor.decoder_repulsion_gate.clone();
    endpoint.barrier_coactivation_gate = anchor.barrier_coactivation_gate.clone();
    endpoint.amplitude_barrier_gate = anchor.amplitude_barrier_gate;
    endpoint.streaming_gates_frozen = true;
    let (_v, _l, cache) = endpoint
        .penalized_quasi_laplace_criterion_with_cache(
            target.view(),
            rho,
            None,
            0,
            0.4,
            1.0e-6,
            1.0e-6,
        )
        .ok()?;
    endpoint
        .exact_observed_information_log_dets(rho, target.view(), &cache)
        .ok()
}

#[test]
fn exact_decoder_prior_theta_trace_matches_curvature_differences_2828() {
    let (term, target, rho) = threshold_gate_tiny_fixture(false);
    let (mut anchor, cache) = frozen_anchor_and_cache(&term, &target, &rho);
    // Make the radial barrier measurable as well as the two overlap priors.
    anchor.amplitude_barrier_gate = Some(0.02);
    let width = anchor.beta_dim();
    assert_eq!(cache.k, width);
    let inverse = Array2::from_shape_fn((width, width), |(i, j)| {
        if i == j {
            1.0
        } else {
            0.1 * ((i + j) as f64).cos()
        }
    });
    for exact in [true, false] {
        let analytic = if exact {
            anchor.exact_decoder_prior_theta_trace(&cache, inverse.view())
        } else {
            anchor.decoder_prior_gap_theta_trace(&cache, inverse.view())
        }
        .expect("exact prior trace adjoint");
        let offsets = anchor.beta_offsets();
        let p = anchor.output_dim();
        let price = |endpoint: &SaeManifoldTerm| {
            let plan = endpoint.prepare_decoder_prior_beta_curvature(1.0);
            let mut trace = 0.0;
            for col in 0..width {
                let (image, majorizer) = endpoint
                    .decoder_prior_beta_hvp_pair_prepared(&plan, inverse.column(col))
                    .expect("exact prior Hessian action");
                trace += if exact {
                    image[col]
                } else {
                    majorizer[col] - image[col]
                };
            }
            trace
        };
        for atom in 0..anchor.atoms.len() {
            for mu in 0..anchor.atoms[atom].basis_size() {
                for out in 0..p {
                    let mut estimates = Vec::new();
                    for h in [2.0e-5, 1.0e-5] {
                        let mut endpoints = Vec::new();
                        for sign in [-1.0, 1.0] {
                            let mut endpoint = anchor.clone();
                            endpoint.decoder_repulsion_gate = anchor.decoder_repulsion_gate.clone();
                            endpoint.barrier_coactivation_gate =
                                anchor.barrier_coactivation_gate.clone();
                            endpoint.amplitude_barrier_gate = anchor.amplitude_barrier_gate;
                            endpoint.streaming_gates_frozen = true;
                            let mut beta = endpoint.atoms[atom].decoder_coefficients().clone();
                            beta[[mu, out]] += sign * h;
                            endpoint.atoms[atom]
                                .set_decoder_coefficients(beta)
                                .expect("same decoder shape");
                            endpoints.push(price(&endpoint));
                        }
                        estimates.push((endpoints[1] - endpoints[0]) / (2.0 * h));
                    }
                    let fd = (4.0 * estimates[1] - estimates[0]) / 3.0;
                    let actual = analytic[offsets[atom] + mu * p + out];
                    assert!(
                        (fd - actual).abs() < 1.0e-5 * (1.0 + fd.abs()),
                        "prior third derivative exact={exact} ({atom},{mu},{out}): {actual:e} versus {fd:e}"
                    );
                }
            }
        }
    }
}

#[test]
fn exact_a_joint_theta_adjoint_beta_block_matches_finite_difference_2828() {
    let (term, target, rho) = threshold_gate_tiny_fixture(false);
    let (anchor, cache) = frozen_anchor_and_cache(&term, &target, &rho);
    let gamma = anchor
        .exact_a_theta_adjoint_joint(&rho, target.view(), &cache)
        .expect("analytic exact-A joint theta adjoint");
    let offsets = anchor.beta_offsets();
    let p = anchor.output_dim();
    let mut worst = 0.0_f64;
    let mut rows = Vec::new();
    for atom in 0..anchor.atoms.len().min(2) {
        let basis = anchor.atoms[atom].basis_size();
        let probes: Vec<(usize, usize)> = (0..basis)
            .flat_map(|mu| (0..p).map(move |out| (mu, out)))
            .take(6)
            .collect();
        for (mu, out) in probes {
            let index = offsets[atom] + mu * p + out;
            let analytic = gamma.beta[index];
            let mut estimates = [0.0_f64; 3];
            let coarse_step = 1.0e-4;
            for (step_index, divisor) in [1.0_f64, 2.0, 4.0].into_iter().enumerate() {
                let h = coarse_step / divisor;
                let mut plus = anchor.clone();
                let mut minus = anchor.clone();
                for (endpoint, sign) in [(&mut plus, 1.0_f64), (&mut minus, -1.0)] {
                    let mut decoder = endpoint.atoms[atom].decoder_coefficients().clone();
                    decoder[[mu, out]] += sign * h;
                    endpoint.atoms[atom]
                        .set_decoder_coefficients(decoder)
                        .expect("same-shape decoder");
                }
                let a = frozen_exact_a_logdet(&anchor, plus, &target, &rho)
                    .expect("positive endpoint admitted");
                let b = frozen_exact_a_logdet(&anchor, minus, &target, &rho)
                    .expect("negative endpoint admitted");
                estimates[step_index] = (a - b) / (2.0 * h);
            }
            let coarse = (4.0 * estimates[1] - estimates[0]) / 3.0;
            let fine = (4.0 * estimates[2] - estimates[1]) / 3.0;
            let oracle_error = (fine - coarse).abs() / (1.0 + fine.abs().max(coarse.abs()));
            let rel = (fine - analytic).abs() / (1.0 + fine.abs().max(analytic.abs()));
            eprintln!(
                "EXACT_A_PROBE_2 atom={atom} mu={mu} out={out} fd={fine:.9e} analytic={analytic:.9e} rel={rel:.3e} oracle_error={oracle_error:.3e}"
            );
            rows.push((atom, mu, out, fine, analytic, rel));
            worst = worst.max(rel + oracle_error);
        }
    }
    eprintln!(
        "EXACT_A_PROBE_2 worst={worst:.3e} over {} β coordinates",
        rows.len()
    );
    assert!(
        worst < 1.0e-3,
        "exact-A joint θ-adjoint β block must match FD: worst={worst:.3e}; rows={rows:?}"
    );
}
