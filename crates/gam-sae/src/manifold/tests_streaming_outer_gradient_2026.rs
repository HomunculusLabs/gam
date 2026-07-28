//! Streaming/matrix-free evidence route — operator primitives, memory
//! admission, and exact-observed-information capability refusal.
//!
//! #2509 established that the Arrow-Schur streaming functional represents the
//! positive majorizer `B`, while the canonical criterion prices exact observed
//! information `A=B+ΔC`. The low-level matrix-free `B` applies and inverse-probe
//! contractions remain useful and are tested here under honest names, but an
//! end-to-end streaming criterion may return only when a structural certificate
//! proves `ΔC≡0`; otherwise it must return the dedicated capability refusal.
//!
//! The tests separately pin (a) the planner's storage decision, (b) propagation
//! of capability refusal through the production outer route, and (c) the
//! low-level matrix-free algebra that a future full-A functional can reuse.

use super::*;
use crate::assignment::{AssignmentMode, SaeAssignment};
use approx::assert_abs_diff_eq;
use gam_solve::inference::residual_factor::{ResidualFactorInput, StructuredResidualModel};
use gam_solve::rho_optimizer::{FixedPointCoordinateCertificate, OuterObjective};
use gam_terms::latent::LatentManifold;
use ndarray::{Array1, Array2};

use super::tests::{
    PlantedCircleAssignmentMode, TestPeriodicEvaluator, periodic_basis, planted_circle_embedded,
    planted_circle_seed_term, small_two_atom_periodic_term,
};
use std::sync::Arc;

// ---- Large-K / wide-border whitened completion ------------------------------

/// Deterministic standard-normal draws (Box–Muller over an LCG) so the whitening
/// factor fitted below is reproducible bit-for-bit.
fn lcg_uniform(s: &mut u64) -> f64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*s >> 11) as f64) / ((1u64 << 53) as f64)
}
fn lcg_normal(s: &mut u64) -> f64 {
    let u1 = lcg_uniform(s).max(1e-12);
    let u2 = lcg_uniform(s);
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

/// A K-atom periodic term over `(n, p)` with a softmax assignment (non-ordered Beta--Bernoulli, so the
/// streaming reduced-Schur log-det has a matrix-free route). Each atom carries the
/// `TestPeriodicEvaluator` — REQUIRED by the streaming path, which re-evaluates
/// Φ(t) per chunk via `materialize_chunk` — and a distinct nonzero decoder so the
/// reconstruction (and hence the residual the row metric whitens) is genuinely
/// nonzero. Mirrors the `small_two_atom_periodic_term` fixture the parity test
/// above uses, generalized to K atoms and a `p`-channel decoder.
fn build_softmax_term(n: usize, p: usize, k: usize) -> SaeManifoldTerm {
    let coord_cols: Vec<Array2<f64>> = (0..k)
        .map(|i| {
            Array2::<f64>::from_shape_fn((n, 1), |(r, _)| {
                (0.03 + 0.11 * i as f64 + 0.017 * (i + 1) as f64 * r as f64).rem_euclid(1.0)
            })
        })
        .collect();
    let atoms: Vec<SaeManifoldAtom> = (0..k)
        .map(|i| {
            let (phi, jet) = periodic_basis(&coord_cols[i]);
            let f = (i as f64) + 1.0;
            // Periodic basis width is 3 ([1, sin, cos]); decoder is (3, p).
            let decoder = Array2::<f64>::from_shape_fn((3, p), |(m, c)| {
                0.1 * f * ((m + 1) as f64) - 0.05 * (c as f64) + 0.02 * f
            });
            SaeManifoldAtom::new_with_provided_function_gram(
                format!("atom{i}"),
                SaeAtomBasisKind::Periodic,
                1,
                phi,
                jet,
                decoder,
                Array2::<f64>::eye(3),
            )
            .unwrap()
            .with_basis_evaluator(Arc::new(TestPeriodicEvaluator))
        })
        .collect();
    let manifolds = vec![LatentManifold::Circle { period: 1.0 }; k];
    let logits =
        Array2::<f64>::from_shape_fn((n, k), |(r, c)| 0.3 * (c as f64) - 0.1 * (r as f64) + 0.2);
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        logits,
        coord_cols,
        manifolds,
        AssignmentMode::softmax(0.8),
    )
    .unwrap();
    SaeManifoldTerm::new(atoms, assignment).unwrap()
}

/// A `WhitenedStructured` per-row precision fitted over `(n, p)` correlated,
/// heteroscedastic residuals (mirrors the #2021 fixture).
fn fit_structured_metric(n: usize, p: usize) -> gam_problem::RowMetric {
    let lam = [1.0_f64, -0.7, 0.4, 0.9, -0.5];
    let dscale = [0.10_f64, 0.55, 0.95, 0.30, 0.70];
    let mut seed = 0x2026_00D5_1234_ABCDu64;
    let mut residuals = Array2::<f64>::zeros((n, p));
    let mut activity = Array1::<f64>::zeros(n);
    for row in 0..n {
        let common = lcg_normal(&mut seed);
        activity[row] = 0.25 + (row as f64) / (n as f64);
        let amp = activity[row].sqrt();
        for i in 0..p {
            residuals[[row, i]] = amp * lam[i % lam.len()] * common
                + dscale[i % dscale.len()] * lcg_normal(&mut seed);
        }
    }
    let model = StructuredResidualModel::fit(ResidualFactorInput {
        residuals: residuals.view(),
        activity: activity.view(),
        max_factor_rank: 2,
    })
    .expect("StructuredResidualModel::fit");
    model.row_metric(n).expect("row_metric")
}

/// At K=32, p=128 the width-2 euclidean border is `border_dim = Σ_k M_k·p =
/// 64·128 = 8192`, so the dense direct evidence peak (`N·q·border_dim`,
/// q=K(1+d)=64) is ≈2.6 GB and exceeds a representative 2 GiB in-core budget,
/// while the matrix-free plan's peak (chunk window + sparse row-cross + border
/// vector workspace) stays in the tens of MB. The storage planner must therefore
/// refuse the dense direct plan while admitting the matrix-free working set.
/// This is memory admission only; #2509's exact-observed-information capability
/// check remains a separate value-side authority.
#[test]
fn wide_border_storage_plan_admits_matrix_free_working_set() {
    let (n, p, k, d_max) = (500usize, 128usize, 32usize, 1usize);
    let total_basis = 2 * k; // width-2 euclidean basis per atom.
    let border_dim = total_basis * p;
    let budget = 2 * 1024 * 1024 * 1024usize; // 2 GiB representative in-core budget.
    let host_available = 8 * 1024 * 1024 * 1024usize;
    let chunk_window = SAE_CPU_L2_CACHE_BYTES * SAE_CHUNK_CACHE_MULTIPLE;
    let plan = sae_streaming_plan_from_budget(
        n,
        total_basis,
        k,
        d_max,
        border_dim,
        budget,
        chunk_window,
        host_available,
    );
    assert!(
        !plan.direct_admitted,
        "the dense direct evidence peak ({} bytes) must exceed the 2 GiB budget so the \
         criterion routes to streaming",
        plan.estimated_direct_peak_bytes
    );
    assert!(
        plan.matrix_free_admitted,
        "the matrix-free working set ({} bytes) must fit the memory budget",
        plan.estimated_matrix_free_peak_bytes
    );
    assert!(
        plan.streaming,
        "a non-direct-admitted plan must select streaming"
    );
    let dense_plan = sae_streaming_plan_from_budget(
        n,
        total_basis,
        k,
        d_max,
        border_dim,
        usize::MAX,
        chunk_window,
        usize::MAX,
    );
    assert!(dense_plan.direct_admitted);
    assert_eq!(
        sae_outer_gradient_capability(),
        Derivative::Analytic,
        "dense SAE retains its exact joint-Hessian IFT gradient"
    );
    let (_representative_term, _, representative_rho) = small_two_atom_periodic_term();
    assert_eq!(
        assignment_strength_gradient_coordinate(&representative_rho),
        representative_rho.sparse_flat_index(),
        "every active assignment strength must enter Hybrid-EFS's \
         exact-gradient block; the outer-plan crossover decides whether that block \
         is consumed, not whether the coordinate has an analytic root"
    );
    // The memory admission gate must accept the working set. It does not mint
    // the distinct exact-A criterion certificate.
    plan.admitted_or_error(n, border_dim, k)
        .expect("matrix-free-admitted plan must not hard-error at the admission gate");
}

/// #2509 production routing pin. Force a planted-circle objective through the
/// streaming route. Its non-empty coordinate block admits potentially live
/// residual / ARD curvature `ΔC`, so the route must propagate the dedicated
/// exact-observed-information capability refusal before constructing a
/// value/gradient artifact. This must never become a dense retry or a synthetic
/// zero gradient.
#[test]
fn production_outer_route_propagates_streaming_exact_a_capability_refusal_2509() {
    let target = planted_circle_embedded(32, 4, 0.02);
    let mut term = planted_circle_seed_term(target.view(), PlantedCircleAssignmentMode::Softmax).0;
    term.atoms[0].basis_second_jet = Some(Arc::new(
        PeriodicHarmonicEvaluator::new(3).expect("periodic evaluator"),
    ));
    let seed_rho = SaeManifoldRho::new(0.0, 0.05_f64.ln(), vec![Array1::<f64>::zeros(1)]);
    let mut streaming =
        SaeManifoldOuterObjective::new(term, target, None, seed_rho, 40, 1.0, 1.0e-6, 1.0e-6);

    // Construction binds the outer-coordinate layout to the assignment family.
    // In particular K=1 Softmax has no entropy-strength coordinate, so the
    // unbound constructor seed has three coordinates while each objective owns
    // the correct two-coordinate layout.  Drive each route from that owned
    // authority; retaining the pre-construction seed here would test a phantom
    // parameter that the production objective correctly refuses.
    let rho_flat = streaming.baseline_rho.to_flat();
    let rho = streaming
        .baseline_rho
        .from_flat(rho_flat.view())
        .expect("streaming objective must own its typed rho layout");
    assert_eq!(
        rho_flat.len(),
        2,
        "K=1 Softmax has no assignment-strength coordinate"
    );

    let error = match streaming.evaluate_outer_criterion_route(&rho, false, false) {
        Err(error) => error,
        Ok(_) => panic!("forced streaming route must not return a B-priced artifact"),
    };
    assert!(matches!(
        error,
        SaeCriterionError::ExactObservedInformationUnavailable {
            route: "streaming"
        }
    ));
}

/// Hybrid-EFS must replace the former held-zero non-ordered Beta--Bernoulli assignment coordinate
/// with the exact penalized quasi-Laplace derivative and expose that same root-equivalent update to
/// the final fixed-point proof hook. This dense fixture exercises the admitted
/// exact-A route; the unrepresented streaming sibling is covered by the typed
/// refusal tests above.
#[test]
fn fixed_point_certificate_covers_non_ordered_beta_bernoulli_exact_gradient() {
    let make_objective = || {
        let (term, target, rho) = small_two_atom_periodic_term();
        let rho_flat = rho.to_flat();
        (
            SaeManifoldOuterObjective::new(term, target, None, rho, 2, 0.25, 1.0e-4, 1.0e-4),
            rho_flat,
        )
    };

    let (mut iteration_objective, rho) = make_objective();
    let iteration = iteration_objective
        .eval_efs(&rho)
        .expect("non-ordered Beta--Bernoulli EFS startup evaluation");
    let gradient = iteration
        .psi_gradient
        .as_ref()
        .expect("assignment strength must be the Hybrid-EFS gradient block")[0];
    assert_eq!(
        iteration.psi_indices.as_deref(),
        Some(&[0][..]),
        "the Hybrid-EFS gradient must map back to log_lambda_sparse"
    );
    assert!(gradient.is_finite(), "assignment gradient must be finite");
    assert_abs_diff_eq!(
        iteration.steps[0],
        -gradient / gradient.abs().max(1.0),
        epsilon = 1.0e-12
    );

    let (mut proof_objective, proof_rho) = make_objective();
    let proof = proof_objective
        .eval_fixed_point_certificate(&proof_rho)
        .expect("fixed-point proof hook must evaluate");
    let (mut exact_objective, exact_rho) = make_objective();
    let exact = exact_objective
        .eval(&exact_rho)
        .expect("authoritative analytic gradient");
    assert_eq!(proof.coordinates.len(), proof_rho.len());
    match &proof.coordinates[0] {
        FixedPointCoordinateCertificate::Covered { update, scale } => {
            assert_abs_diff_eq!(*update, -exact.gradient[0], epsilon = 1.0e-12);
            assert_eq!(*scale, 1.0);
        }
        FixedPointCoordinateCertificate::Uncovered { reason } => panic!(
            "the exact assignment-strength derivative must certify this coordinate: {reason}"
        ),
    }
}

/// Learnable ordered Beta--Bernoulli concentration uses the same complete criterion
/// derivative as every other assignment-strength coordinate. This guards
/// against reintroducing the removed occupancy-only alpha fixed point, whose
/// stationarity equation omitted the inner response and log-determinant terms.
#[test]
fn fixed_point_certificate_covers_ordered_beta_bernoulli_complete_gradient() {
    let make_objective = || {
        let (mut term, target, mut rho) = small_two_atom_periodic_term();
        term.assignment.mode = AssignmentMode::ordered_beta_bernoulli(0.8, 1.0, true);
        rho.log_lambda_sparse = 0.7_f64.ln();
        let rho_flat = rho.to_flat();
        (
            SaeManifoldOuterObjective::new(term, target, None, rho, 2, 0.25, 1.0e-4, 1.0e-4),
            rho_flat,
        )
    };

    let (mut iteration_objective, rho) = make_objective();
    let iteration = iteration_objective
        .eval_efs(&rho)
        .expect("ordered Beta--Bernoulli EFS startup evaluation");
    let gradient = iteration
        .psi_gradient
        .as_ref()
        .expect("learnable concentration must use the complete gradient block")[0];
    assert!(gradient.is_finite());
    assert_eq!(iteration.psi_indices.as_deref(), Some(&[0][..]));
    assert_abs_diff_eq!(
        iteration.steps[0],
        -gradient / gradient.abs().max(1.0),
        epsilon = 1.0e-12
    );

    let (mut proof_objective, proof_rho) = make_objective();
    let proof = proof_objective
        .eval_fixed_point_certificate(&proof_rho)
        .expect("ordered Beta--Bernoulli fixed-point proof hook must evaluate");
    let (mut exact_objective, exact_rho) = make_objective();
    let exact = exact_objective
        .eval(&exact_rho)
        .expect("authoritative analytic gradient");
    match &proof.coordinates[0] {
        FixedPointCoordinateCertificate::Covered { update, scale } => {
            assert_abs_diff_eq!(*update, -exact.gradient[0], epsilon = 1.0e-12);
            assert_eq!(*scale, 1.0);
        }
        FixedPointCoordinateCertificate::Uncovered { reason } => panic!(
            "the complete ordered Beta--Bernoulli concentration derivative must certify this coordinate: {reason}"
        ),
    }
}

/// The B-majorizer assignment-strength
/// `0.5 tr(B^-1 dB/dlog_lambda_sparse)` primitive must be reconstructible from
/// its reduced-Schur inverse-probe bundle. Full-basis probes with exact dense
/// `S^-1` make this low-level identity exact, isolating the contraction from
/// stochastic-CG error without claiming that B is the canonical A criterion.
#[test]
fn assignment_strength_trace_from_probes_matches_dense_softmax() {
    let (n, p, k) = (24usize, 2usize, 2usize);
    let term = build_softmax_term(n, p, k);
    let rho = SaeManifoldRho::new(
        0.7_f64.ln(),
        0.8_f64.ln(),
        vec![Array1::from_elem(1, 1.2_f64.ln()); k],
    );
    // Keep the fixture on the same positive-rank Laplace branch that the
    // production criterion admits.  The old unrelated synthetic target made
    // both decoders fall below the hard MP edge, so the canonical complete
    // gradient correctly refused the rank-zero branch before this test could
    // reach its dense-vs-probe identity.  A deterministic residual around
    // this term's own nonzero reconstruction exercises the identical trace and
    // IFT seams without relying on a value-invalid atom.
    let fitted = term
        .try_fitted_for_rho(&rho)
        .expect("softmax positive-rank fixture reconstruction");
    let target = Array2::<f64>::from_shape_fn((n, p), |(row, col)| {
        fitted[[row, col]] + 1.0e-3 * ((row + 2 * col) as f64 * 0.17).sin()
    });
    let system = term
        .assemble_full_matrix_free_evidence_system(target.view(), &rho, None, None)
        .expect("softmax matrix-free evidence system");
    let options = ArrowSolveOptions::direct().with_positive_definite_evidence();
    let (_, _, cache) = solve_arrow_newton_step_with_options(&system, 0.0, 0.0, &options)
        .expect("direct factorization");
    assert!(
        cache.deflated_row_directions.iter().all(Vec::is_empty),
        "the probe identity is defined on the plain undeflated fixture"
    );

    let solver = DeflatedArrowSolver::plain(&cache);
    let dense = term
        .assignment_log_strength_hessian_trace(&rho, &cache, &solver)
        .expect("dense assignment-strength trace");

    let border_dim = cache.k;
    let sqrt_dim = (border_dim as f64).sqrt();
    let probes = (0..border_dim)
        .map(|column| {
            let mut probe = Array1::<f64>::zeros(border_dim);
            probe[column] = sqrt_dim;
            probe
        })
        .collect::<Vec<_>>();
    let inverse_probes = probes
        .iter()
        .map(|probe| {
            cache
                .schur_inverse_apply(probe.view())
                .expect("exact reduced-Schur inverse probe")
        })
        .collect::<Vec<_>>();
    let matrix_free = term
        .assignment_log_strength_hessian_trace_from_probes(&rho, &cache, &probes, &inverse_probes)
        .expect("matrix-free assignment-strength trace");

    assert!(
        dense.abs() > 1.0e-12,
        "fixture must excite a nonzero assignment-strength trace"
    );
    assert_abs_diff_eq!(matrix_free, dense, epsilon = 1.0e-9);
}

/// End-to-end #2509 refusal on a whitened multi-atom fit. Whitening changes the
/// residual-curvature contraction but does not remove `ΔC`; the non-empty local
/// block therefore cannot be priced through the B-only streaming operator.
/// The exact entry must return its capability error, not a finite approximate
/// criterion.
#[test]
fn whitened_streaming_criterion_refuses_unrepresented_exact_a_2509() {
    let (n, p, k) = (128usize, 16usize, 8usize);
    let mut term = build_softmax_term(n, p, k);
    let metric = fit_structured_metric(n, p);
    assert!(
        metric.whitens_likelihood(),
        "the fitted structured-residual metric must whiten the likelihood"
    );
    term.set_row_metric(metric).unwrap();

    let target = Array2::<f64>::from_shape_fn((n, p), |(r, c)| {
        0.4 - 0.15 * (r as f64 / n as f64)
            + 0.25 * (c as f64 / p as f64)
            + 0.05 * (((r + c) % 7) as f64)
    });
    let rho = SaeManifoldRho::new(
        -1.0_f64,
        0.7_f64.ln(),
        vec![Array1::<f64>::from_elem(1, 0.0); k],
    );

    let error = term
        .penalized_quasi_laplace_criterion_streaming_exact(
            target.view(),
            &rho,
            None,
            2,
            0.25,
            1.0e-4,
            1.0e-4,
        )
        .expect_err("whitened streaming criterion must not price B as exact A");
    assert!(matches!(
        error,
        SaeCriterionError::ExactObservedInformationUnavailable {
            route: "streaming"
        }
    ));
}
