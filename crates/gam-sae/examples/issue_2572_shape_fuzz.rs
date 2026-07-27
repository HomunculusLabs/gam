//! #2572 shape fuzz: drive every stage of the overcomplete support-sparse lane
//! over a grid of shapes and atom portfolios, and report which cells ABORT
//! rather than returning a typed error.
//!
//! The reported panic is shape-dependent (`top_k = 4` aborts, `top_k = 8` does
//! not, at three values of `K` and two of `N`), so the discriminator is a shape,
//! not a datum. This walks the shape space directly instead of trying to
//! reproduce one activation matrix: every stage the fit runs — seed, term build,
//! fixed-point cycles, arrow assembly, the read-only reductions — is called under
//! `catch_unwind`, and a cell that aborts prints `PANIC` with the stage.
//!
//! ```text
//! cargo run -p gam-sae --release --example issue_2572_shape_fuzz
//! ```

use gam_sae::front_door::{SaeFitLane, admit_topk_manifold};
use gam_sae::manifold::{
    SaeSupportSeedRequest, SaeSupportTermSeedRequest, build_sae_support_seed,
    build_sae_support_term_seed, sae_support_effective_atom_dims,
};
use ndarray::{Array2, Axis};

fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

fn unit(seed: u64) -> f64 {
    ((splitmix64(seed) >> 11) as f64) * (1.0 / ((1_u64 << 53) as f64))
}

/// A structured chart: a sparse mixture of curved 1-D features in `p` outputs,
/// which is the shape of an activation chart and not merely noise (a pure-noise
/// target makes every atom's normal equation trivially well posed).
fn structured_chart(n_obs: usize, p_out: usize, seed: u64) -> Array2<f64> {
    let features = 8usize;
    let mut target = Array2::<f64>::zeros((n_obs, p_out));
    for feature in 0..features {
        let mut decoder = vec![0.0_f64; 3 * p_out];
        for (index, value) in decoder.iter_mut().enumerate() {
            *value = 2.0 * unit(seed ^ (feature as u64) << 32 ^ index as u64) - 1.0;
        }
        for row in 0..n_obs {
            let gate = unit(seed ^ 0x51 ^ (feature as u64) << 40 ^ (row as u64) << 8);
            if gate > 0.3 {
                continue;
            }
            let t = unit(seed ^ 0xa7 ^ (feature as u64) << 44 ^ (row as u64) << 4);
            let phi = [
                1.0,
                (std::f64::consts::TAU * t).sin(),
                (std::f64::consts::TAU * t).cos(),
            ];
            for out in 0..p_out {
                let mut value = 0.0;
                for (basis, weight) in phi.iter().enumerate() {
                    value += weight * decoder[basis * p_out + out];
                }
                target[[row, out]] += gate * value;
            }
        }
    }
    for row in 0..n_obs {
        for out in 0..p_out {
            target[[row, out]] += 0.05 * (2.0 * unit(seed ^ 0xf0 ^ (row * p_out + out) as u64) - 1.0);
        }
    }
    target
}

struct Cell {
    n_obs: usize,
    p_out: usize,
    bases: &'static [&'static str],
    dims: &'static [usize],
    k_atoms: usize,
    support_k: usize,
}

fn run_cell(cell: &Cell) -> Result<String, String> {
    let atom_basis: Vec<String> = (0..cell.k_atoms)
        .map(|atom| cell.bases[atom % cell.bases.len()].to_string())
        .collect();
    let atom_dim: Vec<usize> = (0..cell.k_atoms)
        .map(|atom| cell.dims[atom % cell.dims.len()])
        .collect();
    let effective = sae_support_effective_atom_dims(&atom_basis, &atom_dim)?;
    let d_max = effective.iter().copied().max().unwrap_or(1);
    let admission =
        admit_topk_manifold(cell.n_obs, cell.p_out, cell.k_atoms, d_max, cell.support_k)?;
    if admission.lane != SaeFitLane::CurvedStreaming {
        return Ok(format!("skipped: lane {:?}", admission.lane));
    }
    let target = structured_chart(cell.n_obs, cell.p_out, 0x2572);
    let mean = target.mean_axis(Axis(0)).ok_or("empty target")?;
    let centered = &target - &mean;

    let seed = build_sae_support_seed(SaeSupportSeedRequest {
        target: centered.view(),
        atom_basis: &atom_basis,
        atom_dim: &atom_dim,
        support_k: cell.support_k,
        random_state: 0,
        admission,
    })?;
    let retained = seed.retained_atom_indices.clone();
    let mut term = build_sae_support_term_seed(SaeSupportTermSeedRequest {
        assignment: seed.assignment,
        atom_basis: retained.iter().map(|&a| atom_basis[a].clone()).collect(),
        atom_dim: retained.iter().map(|&a| atom_dim[a]).collect(),
        output_dim: cell.p_out,
        random_state: 0,
    })?
    .term;
    let k_ret = term.k_atoms();
    let ard: Vec<Vec<f64>> = (0..k_ret)
        .map(|atom| vec![1.0; term.assignment.atom_coord_dim(atom)])
        .collect();
    let lambda = vec![1.0_f64; k_ret];

    // The fixed point is allowed to run out of cycles: this is a shape probe,
    // not a convergence probe. Only an ABORT is a defect here.
    let inner = term
        .solve_fixed_point(centered.view(), &lambda, &ard, 3, 1.0e-4, 1.0)
        .map(|report| format!("inner recurred in {}", report.iterations))
        .unwrap_or_else(|error| format!("inner Err({})", &error[..error.len().min(60)]));
    term.reconstruct()?;
    term.raw_stationarity(centered.view(), &lambda, &ard)?;
    term.penalized_objective(centered.view(), &lambda, &ard)?;
    let system = term.assemble_arrow_schur(centered.view(), &lambda, &ard)?;
    Ok(format!(
        "retained {k_ret}, border {}, row dims {:?}..{:?}, {inner}",
        system.k,
        system.rows.first().map(|row| row.htt.dim()),
        system.rows.last().map(|row| row.htt.dim())
    ))
}

fn main() {
    const PORTFOLIOS: &[(&str, &[&str], &[usize])] = &[
        ("auto-trio", &["linear", "euclidean", "periodic"], &[1]),
        ("periodic-only", &["periodic"], &[1]),
        ("linear-only", &["linear"], &[1]),
        ("euclidean-only", &["euclidean"], &[1]),
        ("duchon", &["duchon"], &[2]),
        ("poincare", &["poincare"], &[1]),
        ("hetero-d", &["linear", "euclidean", "periodic"], &[1, 2, 3]),
        ("hetero-topology", &["sphere", "torus", "mobius", "periodic"], &[2]),
        (
            "hetero-mixed",
            &["linear", "sphere", "periodic", "torus", "euclidean"],
            &[1, 2, 1, 2, 3],
        ),
    ];
    let mut aborts = 0usize;
    for (name, bases, dims) in PORTFOLIOS {
        for &(n_obs, p_out) in &[(120usize, 8usize), (400, 16), (900, 12)] {
            for &k_atoms in &[9usize, 17, 24, 48, 96] {
                for &support_k in &[1usize, 2, 3, 4, 5, 6, 8, 9] {
                    if support_k > k_atoms {
                        continue;
                    }
                    let cell = Cell {
                        n_obs,
                        p_out,
                        bases,
                        dims,
                        k_atoms,
                        support_k,
                    };
                    let label =
                        format!("{name} N={n_obs} P={p_out} K={k_atoms} s={support_k}");
                    match std::panic::catch_unwind(|| run_cell(&cell)) {
                        Ok(Ok(text)) => println!("ok    {label}: {text}"),
                        Ok(Err(error)) => {
                            println!("err   {label}: {}", &error[..error.len().min(120)])
                        }
                        Err(_) => {
                            aborts += 1;
                            println!("PANIC {label}");
                        }
                    }
                }
            }
        }
    }
    println!("aborting cells: {aborts}");
}
