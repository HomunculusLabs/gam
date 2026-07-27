//! Fit the overcomplete support lane on a real chart to an inner-certified
//! fixed point at fixed smoothing, then dump REAL fitted-atom artifacts for
//! plotting: per-atom topology + usage, Rust-decoded curve samples along each
//! selected atom, and the coordinates/rows of the real tokens on it.
//!
//! ```text
//! cargo run -p gam-sae --release --example support_fit_dump -- \
//!     chart.bin <rows> <cols> <k_atoms> <top_k> <max_cycles> <out_dir>
//! ```

use gam_sae::front_door::{SaeFitLane, admit_topk_manifold};
use gam_sae::manifold::{
    SaeSupportSeedRequest, SaeSupportTermSeedRequest, build_sae_support_seed,
    build_sae_support_term_seed, resolve_support_auto_atoms, sae_support_effective_atom_dims,
};
use ndarray::{Array2, Axis};
use std::io::Write;
use std::time::Instant;

fn write_f64s(path: &str, values: &[f64]) -> Result<(), String> {
    let mut bytes = Vec::with_capacity(values.len() * 8);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    std::fs::write(path, bytes).map_err(|error| format!("{path}: {error}"))
}

fn main() -> Result<(), String> {
    env_logger::init();
    let args: Vec<String> = std::env::args().collect();
    if !matches!(args.len(), 8 | 10 | 11) {
        return Err("usage: support_fit_dump <f64-le.bin> <rows> <cols> <k> <top_k> <max_cycles> <out_dir>".into());
    }
    let rows: usize = args[2].parse().map_err(|e| format!("rows: {e}"))?;
    let cols: usize = args[3].parse().map_err(|e| format!("cols: {e}"))?;
    let k_atoms: usize = args[4].parse().map_err(|e| format!("k: {e}"))?;
    let top_k: usize = args[5].parse().map_err(|e| format!("top_k: {e}"))?;
    let max_cycles: usize = args[6].parse().map_err(|e| format!("max_cycles: {e}"))?;
    let out_dir = &args[7];
    std::fs::create_dir_all(out_dir).map_err(|e| format!("{out_dir}: {e}"))?;

    let bytes = std::fs::read(&args[1]).map_err(|e| format!("{}: {e}", args[1]))?;
    if bytes.len() != rows * cols * 8 {
        return Err(format!("chart holds {} bytes != rows*cols*8", bytes.len()));
    }
    let data: Vec<f64> = bytes
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().expect("8-byte chunk")))
        .collect();
    let target = Array2::from_shape_vec((rows, cols), data).map_err(|e| e.to_string())?;

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
    let retained_basis: Vec<String> =
        retained.iter().map(|&atom| atom_basis[atom].clone()).collect();
    let mut term_seed = build_sae_support_term_seed(SaeSupportTermSeedRequest {
        assignment: seed.assignment,
        atom_basis: retained_basis.clone(),
        atom_dim: retained.iter().map(|&atom| atom_dim[atom]).collect(),
        output_dim: cols,
        random_state: 0,
    })?;
    let k_ret = term_seed.term.k_atoms();
    let ard: Vec<Vec<f64>> = (0..k_ret)
        .map(|atom| vec![1.0; term_seed.term.assignment.atom_coord_dim(atom)])
        .collect();
    let lambda = vec![1.0_f64; k_ret];
    println!("seeded: retained {k_ret} of {k_atoms}");

    let t0 = Instant::now();
    let report = term_seed
        .term
        .solve_fixed_point(centered.view(), &lambda, &ard, max_cycles, 1.0e-4, 1.0)?;
    println!(
        "inner CERTIFIED in {} cycles, {:.0}s, objective {:.4e}",
        report.iterations,
        t0.elapsed().as_secs_f64(),
        report.objective
    );
    let term = &term_seed.term;

    // reconstruction EV at the certified point
    let fitted = term.reconstruct()?;
    let res: f64 = centered
        .iter()
        .zip(fitted.iter())
        .map(|(t, f)| (t - f) * (t - f))
        .sum();
    let tot: f64 = centered.iter().map(|v| v * v).sum();
    println!("centered train EV = {:.4}", 1.0 - res / tot);

    if args.len() >= 10 {
        let te_rows: usize = args[9].parse().map_err(|e| format!("test rows: {e}"))?;
        let te_bytes = std::fs::read(&args[8]).map_err(|e| format!("{}: {e}", args[8]))?;
        if te_bytes.len() != te_rows * cols * 8 {
            println!("HELDOUT skipped: bad size");
        } else {
            let te: Vec<f64> = te_bytes.chunks_exact(8)
                .map(|c| f64::from_le_bytes(c.try_into().expect("8"))).collect();
            let x_test = Array2::from_shape_vec((te_rows, cols), te).map_err(|e| format!("{e}"))?;
            let centered_test = &x_test - &mean;
            let mut te_term = term_seed.term.reroute_fixed_decoder_ard(centered_test.view(), top_k, 0, &ard)?;
            match te_term.solve_coordinates_fixed_decoder(centered_test.view(), &ard, 400, 1.0e-4, 1.0) {
                Ok(rep) => {
                    let recon = te_term.reconstruct()?;
                    let sse: f64 = centered_test.iter().zip(recon.iter()).map(|(x, r)| (x - r).powi(2)).sum();
                    let ss: f64 = centered_test.iter().map(|x| x * x).sum();
                    println!("HELDOUT rows={te_rows} recurred={} EV={:.4}", rep.recurred, 1.0 - sse / ss);
                }
                Err(e) => println!("HELDOUT refused: {e}"),
            }
        }
    }

    // usage census + per-atom rows
    let mut usage = vec![0usize; k_ret];
    let mut atom_tokens: Vec<Vec<(usize, f64, f64)>> = vec![Vec::new(); k_ret]; // (row, t, value)
    for row in 0..rows {
        let support = term.assignment.support_indices(row);
        let values = term.assignment.gate_params(row);
        for (slot, (&atom, &value)) in support.iter().zip(values.iter()).enumerate() {
            if value != 0.0 {
                let atom = atom as usize;
                usage[atom] += 1;
                let t = term.assignment.coords_for_slot(row, slot)[0];
                atom_tokens[atom].push((row, t, value));
            }
        }
    }

    // pick the most-used atoms per topology kind (up to 4 each)
    let mut order: Vec<usize> = (0..k_ret).collect();
    order.sort_by_key(|&a| std::cmp::Reverse(usage[a]));
    let mut picked: Vec<usize> = Vec::new();
    for kind in ["linear", "euclidean", "periodic"] {
        let mut count = 0;
        for &a in &order {
            if retained_basis[a] == kind && usage[a] >= 12 {
                picked.push(a);
                count += 1;
                if count == 4 {
                    break;
                }
            }
        }
    }
    println!("picked atoms: {picked:?}");

    // dump: per picked atom — kind, usage, curve samples over the coordinate
    // range actually used (Rust-decoded), token coords + chart rows projected later
    let mut manifest = String::from("[");
    for (idx, &a) in picked.iter().enumerate() {
        let toks = &atom_tokens[a];
        let kind = &retained_basis[a];
        let (lo, hi) = if kind == "periodic" {
            (0.0, 1.0)
        } else {
            let mut lo = f64::INFINITY;
            let mut hi = f64::NEG_INFINITY;
            for &(_, t, _) in toks {
                lo = lo.min(t);
                hi = hi.max(t);
            }
            let pad = 0.06 * (hi - lo).max(1e-9);
            (lo - pad, hi + pad)
        };
        let grid: Vec<f64> = (0..161)
            .map(|j| lo + (hi - lo) * j as f64 / 160.0)
            .collect();
        let coords = Array2::from_shape_vec((161, 1), grid.clone()).map_err(|e| e.to_string())?;
        let curve = term.decode_atom_at(a, coords.view())?;
        write_f64s(
            &format!("{out_dir}/curve_{idx}.bin"),
            curve.as_slice().ok_or("curve not contiguous")?,
        )?;
        let mut tok_flat: Vec<f64> = Vec::with_capacity(toks.len() * 3);
        for &(row, t, value) in toks {
            tok_flat.push(row as f64);
            tok_flat.push(t);
            tok_flat.push(value);
        }
        write_f64s(&format!("{out_dir}/tokens_{idx}.bin"), &tok_flat)?;
        manifest.push_str(&format!(
            "{}{{\"idx\":{idx},\"atom\":{a},\"kind\":\"{kind}\",\"usage\":{},\"n_tokens\":{},\"grid_lo\":{lo},\"grid_hi\":{hi}}}",
            if idx == 0 { "" } else { "," },
            usage[a],
            toks.len()
        ));
    }
    manifest.push(']');
    let mut file = std::fs::File::create(format!("{out_dir}/manifest.json"))
        .map_err(|e| e.to_string())?;
    file.write_all(manifest.as_bytes()).map_err(|e| e.to_string())?;
    write_f64s(&format!("{out_dir}/mean.bin"), mean.as_slice().ok_or("mean")?)?;
    println!("DUMPED to {out_dir}");
    Ok(())
}
