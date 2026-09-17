#![cfg(test)]
//! #2933 F39 — a learned Grassmann decoder frame prices its integrated response.
//!
//! A framed decoder `B = C·Uᵀ` is fitted by alternating the border Newton solve in
//! `C` with the closed-form polar refresh of `U`, so perturbing the data moves the
//! frame as well as the coordinates. The dispersion used to hold every frame at
//! its fitted orientation and charge its `r·(p − r)` tangent dimensions as fully
//! determined directions, an upper bound rather than the response. The oracle here
//! re-solves each perturbed fit to the joint fixed point of both blocks and
//! differences every fitted scalar, so it measures the frame-integrated response
//! without reading the operator.
use super::tests_fitted_response_edf_2933::{
    ROOT_GRADIENT_CEILING, polish_to_root, root_norm, trace_and_residual_dof,
};
use super::*;
use gam_terms::latent::LatentManifold;
use ndarray::{Array1, Array2};

/// Central-difference step on a target entry of an order-one target.
const FD_STEP: f64 = 1.0e-4;

/// Relative agreement required between a priced quantity and its re-solved value.
const RELATIVE_TOLERANCE: f64 = 1.0e-4;

/// Largest number of polish-then-refresh alternations one re-solve may take.
const FRAME_ALTERNATIONS: usize = 60;

/// A frame refresh is at its fixed point once it moves no decoder coefficient by
/// more than this against order-one decoders, far below the `FD_STEP ×
/// RELATIVE_TOLERANCE` resolution the response comparison is held to.
const FRAME_FIXED_POINT_TOLERANCE: f64 = 1.0e-12;

/// One periodic atom `m(t) = B·[1, sin 2πt, cos 2πt]` in `p = 12` outputs whose
/// decoder lies in the span of the first `rank` output axes, so the fit activates
/// a rank-`rank` frame on that span (the #2933 F35 fixture). `off_span_constant`
/// adds a constant along output axis 2 at rank 2, a component outside the frame
/// span meant to bind the rank constraint. Whether it does at the re-solved root,
/// so that `∇_B L·U⊥ ≠ 0` and the bilinear cross curvature `E` is material, is
/// measured there and printed, not assumed.
fn framed_circle(
    rank: usize,
    off_span_constant: f64,
) -> (SaeManifoldTerm, Array2<f64>, SaeManifoldRho) {
    let (n, p, m) = (24usize, 12usize, 3usize);
    let evaluator = Arc::new(PeriodicHarmonicEvaluator::new(m).expect("periodic basis"));
    let coords = Array2::from_shape_fn((n, 1), |(row, _)| (row as f64 + 0.25) / n as f64);
    let (phi, jet) = evaluator.evaluate(coords.view()).expect("periodic jets");
    let mut decoder = Array2::<f64>::zeros((m, p));
    decoder[[1, 0]] = 0.9;
    decoder[[1, 1]] = 0.2;
    decoder[[2, 0]] = -0.1;
    decoder[[2, 1]] = 0.8;
    if rank == 3 {
        decoder[[0, 2]] = 0.35;
    }
    let mut target = phi.dot(&decoder);
    for row in 0..n {
        let x = row as f64;
        target[[row, 0]] += 0.02 * (1.7 * x).sin();
        target[[row, 1]] += 0.02 * (1.3 * x).cos();
        if rank == 3 {
            target[[row, 2]] += 0.02 * (0.9 * x).sin();
        } else {
            target[[row, 2]] += off_span_constant;
        }
    }
    let atom = SaeManifoldAtom::new_with_provided_function_gram(
        "framed_circle".to_string(),
        SaeAtomBasisKind::Periodic,
        1,
        phi,
        jet,
        decoder,
        Array2::<f64>::eye(m),
    )
    .expect("atom shapes agree")
    .with_basis_second_jet(evaluator);
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        Array2::<f64>::zeros((n, 1)),
        vec![coords],
        vec![LatentManifold::Circle { period: 1.0 }],
        AssignmentMode::softmax(1.0),
    )
    .expect("assignment shapes agree");
    let term = SaeManifoldTerm::new(vec![atom], assignment).expect("term");
    let rho = SaeManifoldRho::new(
        0.0,
        0.8_f64.ln(),
        vec![Array1::from_vec(vec![250.0_f64.ln()])],
    );
    (term, target, rho)
}

/// Alternate the exact-A polish of the border and coordinates with the polar
/// frame refresh until the refresh is at its fixed point and the polish is at its
/// root. Returns the evidence factorization at that root, the root's gradient
/// norm and the last refresh's largest decoder change.
fn framed_root(
    term: &mut SaeManifoldTerm,
    target: &Array2<f64>,
    rho: &SaeManifoldRho,
) -> (ArrowFactorCache, f64, f64) {
    let mut change = f64::INFINITY;
    for _ in 0..FRAME_ALTERNATIONS {
        let (cache, trajectory) = polish_to_root(term, target.view(), rho);
        let norm = root_norm(&trajectory);
        if change <= FRAME_FIXED_POINT_TOLERANCE {
            return (cache, norm, change);
        }
        let before: Vec<Array2<f64>> = term
            .atoms
            .iter()
            .map(|atom| atom.decoder_coefficients().to_owned())
            .collect();
        term.refresh_active_frames_from_data(target.view())
            .expect("the polar frame refresh runs at every alternation");
        change = term
            .atoms
            .iter()
            .zip(&before)
            .flat_map(|(atom, previous)| {
                atom.decoder_coefficients()
                    .iter()
                    .zip(previous.iter())
                    .map(|(now, then)| (now - then).abs())
                    .collect::<Vec<f64>>()
            })
            .fold(0.0_f64, f64::max);
    }
    let (cache, trajectory) = polish_to_root(term, target.view(), rho);
    (cache, root_norm(&trajectory), change)
}

/// `R = ∂f̂/∂y` over the `n·p` scalars by central differences of framed fits
/// re-solved to their joint fixed points under the base term's declared gates.
fn resolved_framed_response(
    base: &SaeManifoldTerm,
    target: &Array2<f64>,
    rho: &SaeManifoldRho,
) -> Array2<f64> {
    let (n, p) = target.dim();
    let gates = base.collapse_prevention_gates();
    let mut response = Array2::<f64>::zeros((n * p, n * p));
    for row in 0..n {
        for col in 0..p {
            let mut fitted = Vec::with_capacity(2);
            for sign in [1.0_f64, -1.0] {
                let mut term = base.clone();
                term.declare_collapse_prevention_gates(&gates);
                let mut perturbed = target.clone();
                perturbed[[row, col]] += sign * FD_STEP;
                let (_, norm, change) = framed_root(&mut term, &perturbed, rho);
                assert!(
                    norm <= ROOT_GRADIENT_CEILING && change <= FRAME_FIXED_POINT_TOLERANCE,
                    "the framed re-solve at entry ({row}, {col}), side {sign} stopped at \
                     ‖g‖={norm:.3e}, frame change {change:.3e}"
                );
                fitted.push(
                    term.try_fitted_for_rho(rho)
                        .expect("the re-solved framed fit reconstructs"),
                );
            }
            for (index, (plus, minus)) in fitted[0].iter().zip(fitted[1].iter()).enumerate() {
                response[[index, row * p + col]] = (plus - minus) / (2.0 * FD_STEP);
            }
        }
    }
    response
}

fn assert_agrees(label: &str, value: f64, resolved: f64) {
    let gap = (value - resolved).abs();
    assert!(
        gap <= RELATIVE_TOLERANCE * resolved.abs().max(1.0),
        "{label}: {value:.9e} against the re-solved response {resolved:.9e} (gap {gap:.3e})"
    );
}

/// #2933 F39 — at rank 2 on a 3-column basis the frame orientation carries a
/// response the fixed-frame divergence omits: with the target in the frame span,
/// and with an off-span constant intended to bind the rank constraint. At rank 3
/// the frame is a pure factorization gauge. In each, the divergence, its residual
/// dof and the residual dof the dispersion prices must be the re-solved response's,
/// priced integrated over the frame, with no count charged for it. Each arm prints
/// `‖∇_B L·U⊥‖_F` at the root, the normal gradient the cross curvature `E` is built
/// from, so a record can say whether `E` was material in that arm, and the elapsed
/// time of the oracle, of the frame-integrated divergence and of the fixed-frame
/// geometry the criterion already forms, so the integrated route's extra cost is
/// measured.
#[test]
fn learned_frames_price_their_integrated_response_2933_f39() {
    for (label, rank, off_span_constant) in [
        ("rank 2 in span", 2usize, 0.0_f64),
        ("rank 2 binding off-span constant", 2, 0.3),
        ("rank 3 gauge", 3, 0.0),
    ] {
        let (mut term, target, rho) = framed_circle(rank, off_span_constant);
        term.recompute_joint_shape_uncertainty(target.view(), &rho, None, 40, 0.4, 1.0e-6, 1.0e-6)
            .unwrap_or_else(|error| panic!("{label}: the framed fixture fits: {error}"));
        let frame_rank = term.atoms[0]
            .decoder_frame
            .as_ref()
            .map(|frame| frame.rank());
        assert_eq!(
            frame_rank,
            Some(rank),
            "{label}: the fit must activate a frame carrying the decoder's rank"
        );
        let gates = term.collapse_prevention_gates();
        term.declare_collapse_prevention_gates(&gates);
        let (cache, norm, change) = framed_root(&mut term, &target, &rho);
        assert!(
            norm <= ROOT_GRADIENT_CEILING && change <= FRAME_FIXED_POINT_TOLERANCE,
            "{label}: the framed fixture stopped at ‖g‖={norm:.3e}, frame change {change:.3e}"
        );
        let oracle_started = std::time::Instant::now();
        let (trace, residual_dof) =
            trace_and_residual_dof(&resolved_framed_response(&term, &target, &rho));
        let oracle_seconds = oracle_started.elapsed().as_secs_f64();
        let integrated_started = std::time::Instant::now();
        let priced = term
            .fitted_response_divergence(target.view(), &rho, &cache)
            .expect("the framed state admits a fitted-response divergence");
        let integrated_seconds = integrated_started.elapsed().as_secs_f64();
        let fixed_frame_started = std::time::Instant::now();
        term.materialize_exact_stationarity_geometry(&rho, target.view(), &cache)
            .expect("the framed state materializes its fixed-frame geometry");
        let fixed_frame_seconds = fixed_frame_started.elapsed().as_secs_f64();
        // The data gradient of the single ungated atom is `∇_B L = Φᵀ·r` (the softmax
        // gate of one atom is identically one and no row carries a weight). Its
        // smoothness gradient `λ·S·B` has no normal part, because `B·U⊥ = 0`.
        let frame = term.atoms[0]
            .decoder_frame
            .as_ref()
            .expect("the framed state carries its frame")
            .frame()
            .to_owned();
        let residual_at_root = term
            .reconstruction_residual(target.view(), &rho)
            .expect("the framed state has a residual");
        let data_gradient = term.atoms[0].basis_values.t().dot(&residual_at_root);
        let normal_gradient = &data_gradient - &data_gradient.dot(&frame).dot(&frame.t());
        let frobenius = |matrix: &Array2<f64>| matrix.iter().map(|v| v * v).sum::<f64>().sqrt();
        eprintln!(
            "[#2933 F39 frames {label}] ‖∇_B L·U⊥‖_F={:.3e} of ‖∇_B L‖_F={:.3e}; oracle \
             {oracle_seconds:.2} s, integrated divergence {integrated_seconds:.3} s, fixed-frame \
             geometry {fixed_frame_seconds:.3} s",
            frobenius(&normal_gradient),
            frobenius(&data_gradient)
        );
        let loss = term.loss(target.view(), &rho).expect("the framed state has a loss");
        let residual = term
            .reconstruction_residual(target.view(), &rho)
            .expect("the framed state has a residual");
        let dispersion = term
            .reconstruction_dispersion(&loss, &cache, &rho, residual.view())
            .expect("the framed state prices a dispersion");
        let priced_residual_dof = 2.0 * loss.data_fit / dispersion.raw_output_noise_variance;
        eprintln!(
            "[#2933 F39 frames {label}] N={} resolved tr R={trace:.9e} ‖I−R‖²={residual_dof:.9e}; \
             divergence {:.9e} ({:?}, {:?}), residual dof {:.9e}, dispersion RSS/φ \
             {priced_residual_dof:.9e}, frame dimension {}",
            target.len(),
            priced.divergence,
            priced.estimator,
            priced.frame_conditioning,
            priced.likelihood_residual_dof,
            term.grassmann_evidence_dimension()
        );
        assert!(
            trace > 1.0 && residual_dof > 1.0,
            "{label}: the re-solved response (tr R {trace}, ‖I−R‖² {residual_dof}) must be \
             material and leave residual dof"
        );
        assert_agrees(&format!("{label} divergence"), priced.divergence, trace);
        assert_agrees(
            &format!("{label} residual dof"),
            priced.likelihood_residual_dof,
            residual_dof,
        );
        assert_agrees(
            &format!("{label} residual dof the dispersion prices"),
            priced_residual_dof,
            residual_dof,
        );
        assert_eq!(
            priced.frame_conditioning,
            SaeFrameConditioning::MarginalOverLearnedFrames,
            "{label}: a small framed fit must integrate its frames"
        );
    }
}
