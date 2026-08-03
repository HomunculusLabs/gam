//! #2712 PROBE (diagnostic, not an acceptance gate).
//!
//! The issue states the from-probes θ-adjoint refuses per-row deflation because
//! "the plain-S⁻¹ bundle does not carry the DEFLATED block". That is a claim
//! about `cache.undamped_factor(row)`: if that Cholesky factorizes the RAW
//! `H_tt`, the claim holds and a genuine derivation is needed; if it factorizes
//! the spectrally CONDITIONED `Φ(H_tt)` (the one that pinned `λ̃ = 1` on each
//! deflated direction `vᵢ`), then `A_i⁻¹ + G_i S⁻¹ G_iᵀ` — which is literally
//! what BOTH routes build — already IS the deflated block, and the only thing
//! missing from the probe route is the Daleckii–Krein correction TERM, whose
//! every operand (`dirs`, `spectrum`, the locally assembled raw `D`) is already
//! in hand.
//!
//! This module measures which of the two it is.

use super::tests::gamma_fd_tiny_fixture;
use super::tests_recovery_split_780::{
    FdAnchorRegime, certified_fd_anchor, rho_ladder_family, sparse_lift_ladder,
};
use super::*;

fn full_basis_bundle(cache: &ArrowFactorCache) -> (Vec<Array1<f64>>, Vec<Array1<f64>>) {
    let k = cache.k;
    let sqrt_k = (k as f64).sqrt();
    let probes: Vec<Array1<f64>> = (0..k)
        .map(|j| {
            let mut v = Array1::<f64>::zeros(k);
            v[j] = sqrt_k;
            v
        })
        .collect();
    let sinv: Vec<Array1<f64>> = probes
        .iter()
        .map(|v| {
            cache
                .schur_inverse_apply(v.view())
                .expect("schur_inverse_apply")
        })
        .collect();
    (probes, sinv)
}

/// Rebuild `A_i = L Lᵀ` from the cached lower Cholesky factor.
fn undamped_block(cache: &ArrowFactorCache, row: usize) -> Array2<f64> {
    let l = cache.undamped_factor(row);
    let q = cache.row_dims[row];
    let mut a = Array2::<f64>::zeros((q, q));
    for i in 0..q {
        for j in 0..q {
            let mut acc = 0.0;
            for t in 0..=i.min(j) {
                acc += l[[i, t]] * l[[j, t]];
            }
            a[[i, j]] = acc;
        }
    }
    a
}

#[test]
fn probe_2712_is_the_cached_row_factor_already_the_deflated_block() {
    let (mut term, target, rho) = gamma_fd_tiny_fixture();
    term.assignment.mode = AssignmentMode::ordered_beta_bernoulli(0.7, 0.9, true);
    let anchor = certified_fd_anchor(
        "#2712 probe: deflated anchor",
        &target,
        FdAnchorRegime::deflated(),
        rho_ladder_family(
            &term,
            sparse_lift_ladder(&rho, &[2.4, 1.8, 1.3, 0.9, 0.5, 0.2, 0.0, -0.3, -0.6, -1.0]),
            5,
        ),
    );
    let cache = anchor.cache;
    let deflated_rows: Vec<usize> = (0..cache.row_dims.len())
        .filter(|&r| {
            cache
                .deflated_row_directions
                .get(r)
                .is_some_and(|d| !d.is_empty())
        })
        .collect();
    eprintln!("#2712 probe: deflated rows = {deflated_rows:?}, k = {}", cache.k);
    assert!(!deflated_rows.is_empty(), "probe premise: some row deflates");

    for &row in &deflated_rows {
        let a = undamped_block(&cache, row);
        let dirs = &cache.deflated_row_directions[row];
        let spectrum = cache.deflation_row_spectra.get(row).and_then(Option::as_ref);
        eprintln!(
            "  row {row}: q={} dirs={} spectrum={}",
            cache.row_dims[row],
            dirs.len(),
            spectrum.is_some()
        );
        for (i, v) in dirs.iter().enumerate() {
            let av = a.dot(v);
            let resid = (&av - v).iter().map(|x| x * x).sum::<f64>().sqrt();
            let rayleigh = v.dot(&av) / v.dot(v);
            eprintln!(
                "    v[{i}]: ||A v - v|| = {resid:.6e}   vᵀAv/vᵀv = {rayleigh:.6e}  (unit ⇒ CONDITIONED)"
            );
        }
        if let Some(spec) = spectrum {
            eprintln!(
                "    raw_evals  = {:?}",
                spec.raw_evals.iter().map(|x| format!("{x:.4e}")).collect::<Vec<_>>()
            );
            eprintln!(
                "    cond_evals = {:?}",
                spec.cond_evals.iter().map(|x| format!("{x:.4e}")).collect::<Vec<_>>()
            );
            let recon = spec
                .evecs
                .dot(&Array2::from_diag(&spec.cond_evals))
                .dot(&spec.evecs.t());
            let err = (&recon - &a).iter().map(|x| x * x).sum::<f64>().sqrt();
            let raw = spec
                .evecs
                .dot(&Array2::from_diag(&spec.raw_evals))
                .dot(&spec.evecs.t());
            let raw_err = (&raw - &a).iter().map(|x| x * x).sum::<f64>().sqrt();
            eprintln!(
                "    ||A - U diag(cond) Uᵀ|| = {err:.6e}   ||A - U diag(raw) Uᵀ|| = {raw_err:.6e}"
            );
        }
    }

    // Now: does the probe reconstruction of `inv_vv` equal the dense selected
    // inverse the deflation correction is contracted against?
    let solver = DeflatedArrowSolver::plain(&cache);
    let beta_inv = solver.beta_inv().expect("beta_inv");
    let (probes, sinv) = full_basis_bundle(&cache);
    let m = probes.len();
    let inv_m = 1.0 / m as f64;
    for &row in &deflated_rows {
        let q = cache.row_dims[row];
        let (dense_vv, dense_vbeta) = solver
            .selected_inverse_row_blocks(row, &beta_inv)
            .expect("dense selected inverse row blocks");
        let factor = cache.undamped_factor(row);
        let mut a_inv = Array2::<f64>::zeros((q, q));
        let mut e = Array1::<f64>::zeros(q);
        for j in 0..q {
            e.fill(0.0);
            e[j] = 1.0;
            let col = cholesky_solve_vector(factor, e.view());
            for r in 0..q {
                a_inv[[r, j]] = col[r];
            }
        }
        let mut probe_vv = a_inv.clone();
        let mut probe_vbeta = Array2::<f64>::zeros((q, cache.k));
        let mut b_tmp = Array1::<f64>::zeros(q);
        for l in 0..m {
            b_tmp.fill(0.0);
            assert!(cache.apply_htbeta_row(row, probes[l].view(), &mut b_tmp));
            let w = cholesky_solve_vector(factor, b_tmp.view());
            b_tmp.fill(0.0);
            assert!(cache.apply_htbeta_row(row, sinv[l].view(), &mut b_tmp));
            let s = cholesky_solve_vector(factor, b_tmp.view());
            for a in 0..q {
                for b in 0..q {
                    probe_vv[[a, b]] += 0.5 * inv_m * (w[a] * s[b] + s[a] * w[b]);
                }
            }
            for a in 0..q {
                probe_vbeta.row_mut(a).scaled_add(-inv_m * w[a], &sinv[l]);
            }
        }
        let vv_err = (&probe_vv - &dense_vv)
            .iter()
            .map(|x| x * x)
            .sum::<f64>()
            .sqrt();
        let vbeta_err = (&probe_vbeta - &dense_vbeta)
            .iter()
            .map(|x| x * x)
            .sum::<f64>()
            .sqrt();
        let vv_scale = dense_vv.iter().map(|x| x * x).sum::<f64>().sqrt();
        eprintln!(
            "  row {row}: ||inv_vv(probes) - inv_vv(dense)|| = {vv_err:.6e} (scale {vv_scale:.6e}), \
             ||inv_vbeta diff|| = {vbeta_err:.6e}"
        );
    }
}
