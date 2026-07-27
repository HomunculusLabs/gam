//! #2576 measurement harness for the support-sparse grouped-LAML evidence
//! criterion: what `log|S|` actually costs and how accurate it actually is.
//!
//! The criterion's whole wall-clock is the reduced-Schur log-determinant and its
//! ρ-derivatives. Before #2576 that number could not be produced at all on a
//! CPU-only host, and the lane reported nothing about what it spent. This
//! harness assembles the SAME arrow system the criterion assembles and then
//! prints, for a ladder of probe counts and deflation-rank targets, the frozen
//! rational surrogate's estimate, its Hutchinson error bar, and its wall-clock —
//! next to the exact dense `log|S|` whenever the border is small enough to
//! afford one.
//!
//! ```text
//! cargo run -p gam-sae --release --example support_laml_trace_probe -- \
//!     chart.bin <rows> <cols> <k_atoms> <top_k> <inner_cycles>
//! ```

use gam_sae::front_door::{SaeFitLane, admit_topk_manifold};
use gam_sae::manifold::{
    SaeSupportSeedRequest, SaeSupportTermSeedRequest, build_sae_support_seed,
    build_sae_support_term_seed, resolve_support_auto_atoms, sae_support_effective_atom_dims,
};
use gam_solve::arrow_schur::{
    ArrowSolveOptions, BatchedBlockSolver, CpuBatchedBlockSolver, SurrogateLaneConfig,
    SurrogateLaneState,
    matrix_free_arrow_evidence_log_det_surrogate, rational_reduced_schur_log_det,
};
use ndarray::{Array2, Axis};
use std::time::Instant;

/// The exact dense reduced-Schur log-determinant is `O(n·d·k²)` to assemble and
/// `O(k³/3)` to factor. Above this border width the oracle is skipped rather
/// than run: the harness would spend hours producing a number the surrogate
/// exists precisely to avoid.
const DENSE_ORACLE_MAX_BORDER: usize = 1_200;

fn main() -> Result<(), String> {
    env_logger::init();
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 7 {
        return Err("usage: support_laml_trace_probe <f64-le.bin> <rows> <cols> <k_atoms> \
                    <top_k> <inner_cycles>"
            .to_string());
    }
    let rows: usize = args[2].parse().map_err(|e| format!("rows: {e}"))?;
    let cols: usize = args[3].parse().map_err(|e| format!("cols: {e}"))?;
    let k_atoms: usize = args[4].parse().map_err(|e| format!("k_atoms: {e}"))?;
    let top_k: usize = args[5].parse().map_err(|e| format!("top_k: {e}"))?;
    let cycles: usize = args[6].parse().map_err(|e| format!("inner_cycles: {e}"))?;
    let bytes = std::fs::read(&args[1]).map_err(|e| format!("{}: {e}", args[1]))?;
    if bytes.len() != rows * cols * 8 {
        return Err(format!(
            "{} holds {} bytes; rows*cols*8 = {}",
            args[1],
            bytes.len(),
            rows * cols * 8
        ));
    }
    let data: Vec<f64> = bytes
        .chunks_exact(8)
        .map(|chunk| f64::from_le_bytes(chunk.try_into().expect("8-byte chunk")))
        .collect();
    let target = Array2::from_shape_vec((rows, cols), data).map_err(|e| format!("{e}"))?;

    let mut atom_basis = vec!["auto".to_string(); k_atoms];
    resolve_support_auto_atoms(&mut atom_basis);
    let atom_dim = vec![1usize; k_atoms];
    let effective = sae_support_effective_atom_dims(&atom_basis, &atom_dim)?;
    let d_max = effective.iter().copied().max().unwrap_or(1);
    let admission = admit_topk_manifold(rows, cols, k_atoms, d_max, top_k)?;
    if admission.lane != SaeFitLane::CurvedStreaming {
        return Err(format!("expected CurvedStreaming; got {:?}", admission.lane));
    }
    let mean = target.mean_axis(Axis(0)).ok_or("empty target")?;
    let centered = &target - &mean;

    let seed = build_sae_support_seed(SaeSupportSeedRequest {
        target: centered.view(),
        atom_basis: &atom_basis,
        atom_dim: &atom_dim,
        support_k: top_k,
        random_state: 0,
        admission,
    })?;
    let retained = seed.retained_atom_indices.clone();
    let term_seed = build_sae_support_term_seed(SaeSupportTermSeedRequest {
        assignment: seed.assignment,
        atom_basis: retained
            .iter()
            .map(|&atom| atom_basis[atom].clone())
            .collect(),
        atom_dim: retained.iter().map(|&atom| atom_dim[atom]).collect(),
        output_dim: cols,
        random_state: 0,
    })?;
    let mut term = term_seed.term;
    let ard_precisions = (0..term.k_atoms())
        .map(|atom| vec![1.0; term.assignment.atom_coord_dim(atom)])
        .collect::<Vec<_>>();
    let lambda_smooth = vec![1.0; term.k_atoms()];

    // Drive the inner fixed point toward its optimum: the evidence is only the
    // Laplace normalizer AT a stationary inner state, and a random seed's arrow
    // system is not the one production factors.
    let t_inner = Instant::now();
    let _ = term.solve_fixed_point(
        centered.view(),
        &lambda_smooth,
        &ard_precisions,
        cycles,
        1.0e-4,
        1.0,
    );
    println!(
        "inner fixed point: {cycles} cycles, {:.1}s",
        t_inner.elapsed().as_secs_f64()
    );

    let system = term.assemble_arrow_schur(centered.view(), &lambda_smooth, &ard_precisions)?;
    let border = system.k;
    println!(
        "assembled: border {border}, rows {}, atoms {}",
        term.n_obs(),
        term.k_atoms()
    );
    if let Some(diag) = system.hbb_diag.as_ref() {
        let lo = diag.iter().copied().fold(f64::INFINITY, f64::min);
        let hi = diag.iter().copied().fold(0.0_f64, f64::max);
        println!("shared-block diagonal: min {lo:.4e} max {hi:.4e} spread {:.1e}", hi / lo);
    }

    let backend = CpuBatchedBlockSolver;
    let options = ArrowSolveOptions::inexact_pcg().with_positive_definite_evidence();
    let htt = backend
        .factor_blocks(&system.rows, 0.0, system.d, false)
        .map_err(|e| format!("row factorization: {e}"))?;

    // A/B on ONE operator.
    //
    // The preconditioner build reads the shared block's diagonal through
    // `penalty_diagonal_add`, which prefers `penalty_op` and falls back to
    // `hbb_diag`. `set_shared_beta_operator` installs BOTH (the penalty op is a
    // `MatvecDiagPenaltyOp` wrapping the very same matvec `Arc` plus the same
    // diagonal), so clearing only `hbb_diag` leaves the preconditioner fully
    // supplied — the first version of this harness did exactly that and
    // measured the preconditioned iteration twice.
    //
    // Clearing both leaves the operator itself unchanged: every hot path falls
    // back to `hbb_matvec`, which is the identical closure. The `log|S|`
    // equality printed by the two arms is the check that this is so — the value
    // is a property of the operator, not of the preconditioner.
    let mut bare = system.clone();
    bare.hbb_diag = None;
    bare.penalty_op = None;

    for (label, sys) in [
        ("identity (pre-#2576)", &bare),
        ("shared-block diagonal", &system),
    ] {
        for probes in [8usize] {
            let start = Instant::now();
            match rational_reduced_schur_log_det(
                sys, &htt, 0.0, &backend, None, None, probes, 0xC0FFEE, 1.0e-8, 40, 1.0e-8, 20_000,
            ) {
                Some((_plan, eval)) => println!(
                    "{label:<22} probes={probes:3}: log|S| = {:+.6e}  std_err {:.3e} \
                     (rel {:.2e})  cg_iters {}  {:.1}s",
                    eval.estimate,
                    eval.std_err,
                    eval.std_err / (eval.estimate.abs() + 1.0),
                    eval.cg_iterations,
                    start.elapsed().as_secs_f64(),
                ),
                None => println!(
                    "{label:<22} probes={probes:3}: REFUSED after {:.1}s",
                    start.elapsed().as_secs_f64()
                ),
            }
        }
    }

    // Derived-rank lane at a ladder of relative error-bar targets: how much
    // deflation each target actually costs, and which ones are reachable at all.
    for target_rel in [1.0e-2_f64, 1.0e-3, 1.0e-4, 1.0e-6, 1.0e-9] {
        let mut lane = SurrogateLaneState::new(SurrogateLaneConfig {
            num_probes: 32,
            seed: 0xC0FFEE,
            rel_tol: 1.0e-8,
            power_iters: 40,
            cg_rel_tol: 1.0e-8,
            cg_max_iters: 20_000,
            deflation_max_rank: 128,
            deflation_subspace_iters: 4,
            deflation_target_std_err_rel: target_rel,
        });
        let start = Instant::now();
        match matrix_free_arrow_evidence_log_det_surrogate(
            &system,
            0.0,
            0.0,
            &options,
            32,
            64,
            0xC0FFEE,
            Some(&mut lane),
        ) {
            Ok((row, schur)) => println!(
                "derived lane target_rel={target_rel:.0e}: row log|H_tt| {:+.6e}  \
                 log|S| {:+.6e}  rank {}  {:.1}s",
                row,
                schur,
                lane.plan().and_then(|p| p.deflation.as_ref()).map_or(0, |d| d.basis.len()),
                start.elapsed().as_secs_f64(),
            ),
            Err(error) => println!(
                "derived lane target_rel={target_rel:.0e}: REFUSED after {:.1}s — {error}",
                start.elapsed().as_secs_f64()
            ),
        }
    }

    if border <= DENSE_ORACLE_MAX_BORDER {
        println!("(dense oracle would be affordable at border {border}; see the crate tests)");
    } else {
        println!(
            "(dense oracle skipped: border {border} exceeds {DENSE_ORACLE_MAX_BORDER}, where the \
             O(n*d*k^2) assembly the surrogate exists to avoid would dominate this harness)"
        );
    }
    Ok(())
}
