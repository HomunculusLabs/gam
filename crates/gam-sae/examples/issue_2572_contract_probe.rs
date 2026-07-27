//! #2572 mechanism probe: what a violated atom shape contract does on the
//! overcomplete support-sparse lane, and what refuses it now.
//!
//! Every `[[i, j]]` subscript in that lane's kernels is in bounds *iff* one
//! cross-field contract on `SaeManifoldAtom` holds — decoder rows equal the
//! basis width, decoder columns equal the term's output dim, the reference Gram
//! is that width square, and the latent dim agrees with the assignment.
//!
//! Measured BEFORE the fix, with `decoder_coefficients` a `pub` field that any
//! site could replace (`SaeSupportSparseTerm` seeded at N=240, P=8, K=24, s=4,
//! atom 0's decoder sliced by one):
//!
//! ```text
//! == decoder one row short ==
//! SaeSupportSparseTerm::new          Ok(accepted the atom)
//! reconstruct                        PANIC(ndarray: inputs 8 x 2 and 3 x 1 are not
//!                                          compatible for matrix multiplication)
//! raw_stationarity                   PANIC(same)
//! solve_fixed_point (1 cycle)        PANIC(same)
//! assemble_arrow_schur               PANIC(same)
//!
//! == decoder one column short ==
//! SaeSupportSparseTerm::new          Err(atom 1 output dimension 8 != 7)
//! reconstruct                        PANIC(ndarray: index out of bounds)
//! raw_stationarity                   PANIC(ndarray: index out of bounds)
//! solve_fixed_point (1 cycle)        PANIC(ndarray: index out of bounds)
//! assemble_arrow_schur               PANIC(ndarray: index out of bounds)
//! ```
//!
//! Two things to read off it. The column-short half reproduces the reported
//! abort message EXACTLY, from four different kernels, each on an unnamed rayon
//! worker — the reported signature. And the row-short half was accepted by the
//! lane's own constructor: `SaeSupportSparseTerm::new` validated the two
//! quantities its kernels subscript with (`output_dim`, `latent_dim`) and not
//! the two that produce this abort (decoder rows vs basis width, Gram width).
//!
//! After the fix the mutation itself is refused, so no kernel ever sees the
//! broken atom: this harness now prints the refusal.
//!
//! ```text
//! cargo run -p gam-sae --release --example issue_2572_contract_probe
//! ```

use gam_sae::front_door::admit_topk_manifold;
use gam_sae::manifold::{
    SaeSupportSeedRequest, SaeSupportSparseTerm, SaeSupportTermSeedRequest, build_sae_support_seed,
    build_sae_support_term_seed, resolve_support_auto_atoms, sae_support_effective_atom_dims,
};
use ndarray::{Array2, Axis};

fn seeded_term(
    n_obs: usize,
    p_out: usize,
    k_atoms: usize,
    support_k: usize,
) -> Result<(SaeSupportSparseTerm, Array2<f64>), String> {
    let mut target = Array2::<f64>::zeros((n_obs, p_out));
    for row in 0..n_obs {
        for col in 0..p_out {
            let t = (row * 7 + col * 13) as f64;
            target[[row, col]] = (0.1 * t).sin() + 0.3 * (0.03 * t).cos();
        }
    }
    let mut atom_basis = vec!["auto".to_string(); k_atoms];
    resolve_support_auto_atoms(&mut atom_basis);
    let atom_dim = vec![1usize; k_atoms];
    let effective = sae_support_effective_atom_dims(&atom_basis, &atom_dim)?;
    let d_max = effective.iter().copied().max().unwrap_or(1);
    let admission = admit_topk_manifold(n_obs, p_out, k_atoms, d_max, support_k)?;
    let mean = target.mean_axis(Axis(0)).ok_or("empty target")?;
    let centered = &target - &mean;
    let seed = build_sae_support_seed(SaeSupportSeedRequest {
        target: centered.view(),
        atom_basis: &atom_basis,
        atom_dim: &atom_dim,
        support_k,
        random_state: 0,
        admission,
    })?;
    let retained = seed.retained_atom_indices.clone();
    let term = build_sae_support_term_seed(SaeSupportTermSeedRequest {
        assignment: seed.assignment,
        atom_basis: retained.iter().map(|&a| atom_basis[a].clone()).collect(),
        atom_dim: retained.iter().map(|&a| atom_dim[a]).collect(),
        output_dim: p_out,
        random_state: 0,
    })?
    .term;
    Ok((term, centered))
}

fn report(label: &str, run: impl FnOnce() -> Result<String, String>) {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(run)) {
        Ok(Ok(text)) => println!("{label:<34} Ok({text})"),
        Ok(Err(error)) => println!("{label:<34} Err({})", &error[..error.len().min(140)]),
        Err(payload) => {
            let message = payload
                .downcast_ref::<&str>()
                .map(|s| (*s).to_string())
                .or_else(|| payload.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "<non-string panic payload>".to_string());
            println!("{label:<34} PANIC({message})");
        }
    }
}

fn main() -> Result<(), String> {
    let (base, centered) = seeded_term(240, 8, 24, 4)?;
    let k_ret = base.k_atoms();
    let ard: Vec<Vec<f64>> = (0..k_ret)
        .map(|atom| vec![1.0; base.assignment.atom_coord_dim(atom)])
        .collect();
    let lambda = vec![1.0_f64; k_ret];
    println!(
        "seeded: K={k_ret}, output_dim={}, atom 0 basis width {}",
        base.output_dim(),
        base.atoms[0].basis_size()
    );

    for (name, shrink) in [
        (
            "decoder one ROW short (basis width)",
            Box::new(|full: &Array2<f64>| {
                full.slice(ndarray::s![..full.nrows() - 1, ..]).to_owned()
            }) as Box<dyn Fn(&Array2<f64>) -> Array2<f64>>,
        ),
        (
            "decoder one COLUMN short (output dim)",
            Box::new(|full: &Array2<f64>| {
                full.slice(ndarray::s![.., ..full.ncols() - 1]).to_owned()
            }),
        ),
    ] {
        println!("\n== {name} ==");
        let mut term = base.clone();
        let broken = shrink(term.atoms[0].decoder_coefficients());
        report("set_decoder_coefficients", || {
            term.atoms[0]
                .set_decoder_coefficients(broken)
                .map(|()| "INSTALLED a decoder that cannot be indexed".to_string())
        });
        // The atom is unchanged, so the kernels below run on a well-formed term.
        // They are exercised anyway: this harness's claim is that no stage of
        // the lane can be reached with a broken atom, not merely that the
        // setter refuses.
        report("reconstruct", || {
            term.reconstruct()
                .map(|fitted| format!("{:?}", fitted.dim()))
        });
        report("raw_stationarity", || {
            term.raw_stationarity(centered.view(), &lambda, &ard)
                .map(|s| format!("{:.3e}", s.max_abs()))
        });
        report("assemble_arrow_schur", || {
            term.assemble_arrow_schur(centered.view(), &lambda, &ard)
                .map(|system| format!("border {}", system.k))
        });
    }
    Ok(())
}
