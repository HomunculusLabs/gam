//! #2576 measurement harness: what one LAML trace probe actually costs on the
//! overcomplete support-sparse border, preconditioned versus not.
//!
//! The grouped-LAML criterion's Hutchinson trace estimator solves the bordered
//! arrow system once per probe. That solve is the lane's whole wall-clock, and
//! before #2576 it reported nothing — not its iteration count, not its residual
//! — so a CG that stagnated to its cap was indistinguishable from one that
//! converged. This harness runs the SAME solve the estimator runs and prints
//! the certificate, with the shared-block diagonal present (the production
//! preconditioner) and stripped (the previous unpreconditioned iteration), so
//! the two are measured against one assembled system rather than two builds.
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
    ArrowSolveOptions, matrix_free_arrow_inverse_apply, solve_arrow_newton_step_with_options,
};
use ndarray::{Array1, Array2, Axis};
use std::time::Instant;

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

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

    // Drive the inner fixed point toward its optimum: the trace is only the
    // Laplace normalizer's derivative AT a stationary inner state, and a random
    // seed's arrow system is not the one production solves.
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
        "inner fixed point: {} cycles, {:.1}s",
        cycles,
        t_inner.elapsed().as_secs_f64()
    );

    let system = term.assemble_arrow_schur(centered.view(), &lambda_smooth, &ard_precisions)?;
    let beta_dim = system.k;
    let options = ArrowSolveOptions::inexact_pcg().with_positive_definite_evidence();
    let t_factor = Instant::now();
    let (_, _, cache) = solve_arrow_newton_step_with_options(&system, 0.0, 0.0, &options)
        .map_err(|e| format!("factorization: {e}"))?;
    println!(
        "arrow factorization: {:.1}s (border dim {beta_dim}, rows {}, atoms {})",
        t_factor.elapsed().as_secs_f64(),
        term.n_obs(),
        term.k_atoms(),
    );
    if let Some(diag) = system.hbb_diag.as_ref() {
        let lo = diag.iter().copied().fold(f64::INFINITY, f64::min);
        let hi = diag.iter().copied().fold(0.0_f64, f64::max);
        println!("shared-block diagonal: min {lo:.4e}, max {hi:.4e}, spread {:.1e}", hi / lo);
    }

    let mut probe = Array1::<f64>::zeros(beta_dim);
    for index in 0..beta_dim {
        probe[index] = if splitmix64(index as u64) >> 63 == 0 {
            -1.0
        } else {
            1.0
        };
    }
    let rhs_t = Array1::<f64>::zeros(cache.delta_t_len());
    let max_iters = beta_dim.saturating_mul(2).clamp(128, 4096);

    // The unpreconditioned comparison strips the assembled shared-block
    // diagonal from a CLONE. `penalty_diagonal_add` then finds neither a
    // penalty operator nor a diagonal and reads the (empty, matrix-free) dense
    // `hbb`, so the preconditioner falls back to the identity — i.e. exactly
    // the iteration this lane ran before #2576, on the very same operator.
    let mut bare = system.clone();
    bare.hbb_diag = None;

    for (label, sys) in [("identity (pre-#2576)", &bare), ("shared-block diagonal", &system)] {
        for &tol in &[1.0e-8_f64, 2.2e-2_f64] {
            let start = Instant::now();
            match matrix_free_arrow_inverse_apply(
                sys,
                &cache,
                rhs_t.view(),
                probe.view(),
                tol,
                max_iters,
            ) {
                Ok((_, _, report)) => println!(
                    "{label:<22} tol {tol:8.1e}: {:5} iters (cap {}), residual {:.3e}, \
                     converged {}, {:.1}s",
                    report.iterations,
                    report.max_iterations,
                    report.relative_residual,
                    report.converged(),
                    start.elapsed().as_secs_f64(),
                ),
                Err(error) => println!("{label:<22} tol {tol:8.1e}: FAILED {error}"),
            }
        }
    }
    Ok(())
}
