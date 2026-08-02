//! #2757 probe — the residual-gauge curvature Gram's shape, cost, and
//! (hypothesised) exact block structure.
//!
//! Not a contract. This module exists to CONFIRM or REFUTE, empirically:
//!
//!   H1. `param_dim = p · Σ_k d_k`, so the Gram is `(p·D)²` and the reported
//!       45.97 GiB at `p = 4096` is one `f64` copy of it.
//!   H2. Under a metric that does not drive the gauge (`Euclidean` — the
//!       `diagnostic_metric()` fallback), `H = RᵀR` is EXACTLY block diagonal
//!       when its columns are grouped by output coordinate `i`, with `p`
//!       blocks of size `D × D`.
//!   H3. The wall really is `dim³` / `dim²`: time and memory of the shipped
//!       path against `p`, and the eigendecomposition's share of the whole
//!       `fit_diagnostics_report`.

use crate::manifold::{
    AssignmentMode, PeriodicHarmonicEvaluator, SaeAssignment, SaeAtomBasisKind, SaeBasisEvaluator,
    SaeManifoldAtom, SaeManifoldRho, SaeManifoldTerm,
};
use gam_terms::latent::LatentManifold;
use ndarray::{Array1, Array2};
use std::sync::Arc;
use std::time::Instant;

fn lcg(s: &mut u64) -> f64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*s >> 11) as f64) / ((1u64 << 53) as f64)
}

/// `k_atoms` planted circles in `p`-dimensional output space, assembled (not
/// fitted) so the probe measures the certificate rather than the optimizer.
fn planted_term(n: usize, p: usize, k_atoms: usize) -> (SaeManifoldTerm, SaeManifoldRho) {
    let mut s = 0x2757_0000_0000_0001u64;
    let evaluator = Arc::new(PeriodicHarmonicEvaluator::new(3).expect("harmonic order 3"));
    let mut atoms = Vec::with_capacity(k_atoms);
    let mut coord_blocks = Vec::with_capacity(k_atoms);
    let mut manifolds = Vec::with_capacity(k_atoms);
    for k in 0..k_atoms {
        let theta: Vec<f64> = (0..n).map(|_| lcg(&mut s)).collect();
        let coords = Array2::<f64>::from_shape_fn((n, 1), |(r, _)| theta[r]);
        let (phi, jet) = evaluator.evaluate(coords.view()).expect("periodic evaluate");
        let mut decoder = Array2::<f64>::zeros((3, p));
        decoder[[1, (2 * k) % p]] = 1.0;
        decoder[[2, (2 * k + 1) % p]] = 1.0;
        // A dense tail so the frame is not axis-sparse (the block-diagonality
        // claim must not be an artifact of a sparse planted decoder).
        for c in 0..p {
            decoder[[0, c]] = 0.05 * (lcg(&mut s) - 0.5);
            decoder[[1, c]] += 0.05 * (lcg(&mut s) - 0.5);
            decoder[[2, c]] += 0.05 * (lcg(&mut s) - 0.5);
        }
        atoms.push(
            SaeManifoldAtom::new_with_provided_function_gram(
                format!("circle{k}"),
                SaeAtomBasisKind::Periodic,
                1,
                phi,
                jet,
                decoder,
                Array2::<f64>::eye(3),
            )
            .expect("atom blocks agree")
            .with_basis_second_jet(evaluator.clone()),
        );
        coord_blocks.push(coords);
        manifolds.push(LatentManifold::Circle { period: 1.0 });
    }
    let logits = Array2::<f64>::from_elem((n, k_atoms), 3.0);
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        logits,
        coord_blocks,
        manifolds,
        AssignmentMode::ordered_beta_bernoulli(0.7, 1.0, false),
    )
    .expect("assignment blocks agree");
    let mut term = SaeManifoldTerm::new(atoms, assignment).expect("term");
    term.set_guards_enabled(false);
    let rho = SaeManifoldRho::new(0.0, 0.0, vec![Array1::<f64>::zeros(1); k_atoms]);
    (term, rho)
}

#[test]
fn probe_2757_curvature_gram_shape_structure_and_cost() {
    let n = 64;
    let k_atoms = 4;
    println!("\n#2757 probe: residual-gauge curvature Gram (n={n}, K={k_atoms}, d=1 each)");
    println!(
        "{:>6} {:>10} {:>14} {:>12} {:>12} {:>12}",
        "p", "param_dim", "gram_bytes", "build_s", "eigh_s", "offblock_max"
    );
    for &p in &[16usize, 32, 64, 128] {
        let (term, _rho) = planted_term(n, p, k_atoms);
        let metric = term.diagnostic_metric().expect("metric");
        assert!(
            !metric.drives_gauge(),
            "the diagnostic fallback must be the Euclidean (non-gauge-driving) metric"
        );
        let d_total = k_atoms; // every atom is d = 1 here
        let param_dim = p * d_total;
        let offsets: Vec<usize> = (0..k_atoms).map(|k| k * p).collect();
        let axis_dim = vec![1usize; k_atoms];

        let t0 = Instant::now();
        let (gram, root_rows) = term
            .residual_gauge_streamed_data_curvature(&metric, &offsets, &axis_dim, param_dim)
            .expect("streamed curvature");
        let build_s = t0.elapsed().as_secs_f64();
        assert_eq!(gram.dim(), (param_dim, param_dim), "H1: Gram is param_dim²");
        assert_eq!(root_rows, n * metric.metric_rank());

        // H2: group columns by output coordinate. Column `c` of atom `k` sits at
        // `offsets[k] + i*d_k + axis`, so its output coordinate is
        // `(c - offsets[k]) / d_k`.
        let out_of = |c: usize| -> usize {
            let k = c / p; // d = 1, contiguous per atom
            (c - offsets[k]) / axis_dim[k]
        };
        let mut offblock_max = 0.0_f64;
        let mut diag_max = 0.0_f64;
        for a in 0..param_dim {
            for b in 0..param_dim {
                let v = gram[[a, b]].abs();
                if out_of(a) == out_of(b) {
                    diag_max = diag_max.max(v);
                } else {
                    offblock_max = offblock_max.max(v);
                }
            }
        }

        let t1 = Instant::now();
        let g2 = gram.clone();
        let evals = {
            use gam_linalg::faer_ndarray::FaerEigh;
            let (evals, _vecs) = g2.eigh(faer::Side::Lower).expect("eigh");
            evals
        };
        let eigh_s = t1.elapsed().as_secs_f64();
        let smax = evals.iter().cloned().fold(0.0_f64, f64::max);

        println!(
            "{p:>6} {param_dim:>10} {:>14} {build_s:>12.4} {eigh_s:>12.4} {:>12.3e}",
            param_dim * param_dim * 8,
            offblock_max
        );
        println!(
            "        in-block max {diag_max:.6e}   sigma_max {smax:.6e}   \
             off/in ratio {:.3e}",
            offblock_max / diag_max.max(f64::MIN_POSITIVE)
        );
    }
}

/// H3 — where the wall is inside `fit_diagnostics_report`, and how it scales.
/// Splits the whole report against the streamed-Gram build plus the
/// eigendecomposition alone, doubling `p` each row.
#[test]
#[ignore = "scaling probe: minutes of wall-clock, run explicitly for #2757"]
fn probe_2757_diagnostics_report_wall_scaling() {
    let n = 48;
    let k_atoms = 4;
    println!("\n#2757 probe: fit_diagnostics_report wall split (n={n}, K={k_atoms}, d=1)");
    println!(
        "{:>6} {:>10} {:>12} {:>12} {:>12} {:>12} {:>8}",
        "p", "param_dim", "gram_GiB", "build_s", "eigh_s", "report_s", "eigh_%"
    );
    for &p in &[64usize, 128, 256, 512, 1024] {
        let (term, rho) = planted_term(n, p, k_atoms);
        let fitted = term
            .try_fitted_target_aware(
                Array2::<f64>::zeros((n, p)).view(),
                Some(&rho),
            )
            .expect("fitted");
        let metric = term.diagnostic_metric().expect("metric");
        let param_dim = p * k_atoms;
        let offsets: Vec<usize> = (0..k_atoms).map(|k| k * p).collect();
        let axis_dim = vec![1usize; k_atoms];

        let t0 = Instant::now();
        let (gram, _rows) = term
            .residual_gauge_streamed_data_curvature(&metric, &offsets, &axis_dim, param_dim)
            .expect("streamed curvature");
        let build_s = t0.elapsed().as_secs_f64();
        let t1 = Instant::now();
        {
            use gam_linalg::faer_ndarray::FaerEigh;
            let _ = gram.eigh(faer::Side::Lower).expect("eigh");
        }
        let eigh_s = t1.elapsed().as_secs_f64();
        drop(gram);

        let t2 = Instant::now();
        let _report = term
            .fit_diagnostics_report(None, false, None, fitted.view(), None)
            .expect("diagnostics report");
        let report_s = t2.elapsed().as_secs_f64();

        println!(
            "{p:>6} {param_dim:>10} {:>12.4} {build_s:>12.3} {eigh_s:>12.3} {report_s:>12.3} \
             {:>7.1}%",
            (param_dim * param_dim * 8) as f64 / (1024.0 * 1024.0 * 1024.0),
            100.0 * eigh_s / report_s.max(1e-12)
        );
    }
}
