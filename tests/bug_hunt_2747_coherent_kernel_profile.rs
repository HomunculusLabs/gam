//! #2747 decisive experiment: the COHERENT constant-curvature RKHS model, with
//! its kernel range `ℓ` PROFILED rather than pinned by the fill heuristic.
//!
//! Two models are compared on identical data, identical centers, identical
//! response, identical REML solver:
//!
//! * **F (shipped)** — `build_constant_curvature_basis`: design at the
//!   data→center fill-invariant length `L(κ)`, penalty at the center→center
//!   fill-invariant length `L_S(κ)`, both derived from a heuristic `ℓ_ref` that
//!   is never estimated.
//! * **C (coherent)** — one kernel, one range: `X = K_{κ,ℓ}(data,C)·z`,
//!   `S = zᵀK_{κ,ℓ}(C,C)z`. This is the subset-of-regressors GP with kernel
//!   `exp(−d_κ/ℓ)`, so `S` really is the RKHS roughness of the fitted design.
//!
//! and for each, `V(κ)` is reported twice: at the heuristic `ℓ_ref`, and
//! PROFILED, `min over ℓ`.
//!
//! The REML criterion is invariant to the choice of basis for the constrained
//! coefficient subspace (`X ↦ XT`, `S ↦ TᵀST` leaves `log|H| − log|S|` and the
//! fit invariant), so model C uses an explicit orthonormal sum-to-zero frame
//! rather than reaching for the crate-private one.
//!
//! Diagnostic only: it PRINTS.

use gam::basis::{
    CenterStrategy, ConstantCurvatureBasisSpec, ConstantCurvatureIdentifiability,
    build_constant_curvature_basis, constant_curvature_kernel_matrix,
};
use gam::gaussian_reml::gaussian_reml_multi_closed_form;

use gam::utils::splitmix64;
use ndarray::{Array1, Array2, ArrayView2};

fn next_unit(state: &mut u64) -> f64 {
    (splitmix64(state) >> 11) as f64 / (1u64 << 53) as f64
}
fn next_gauss(state: &mut u64) -> f64 {
    let u1 = next_unit(state).max(1.0e-12);
    let u2 = next_unit(state);
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

/// Orthonormal `m × (m−1)` frame for `{α : Σα = 0}` (Helmert contrasts).
fn sum_to_zero_frame(m: usize) -> Array2<f64> {
    let mut z = Array2::<f64>::zeros((m, m - 1));
    for k in 1..m {
        let norm = ((k * (k + 1)) as f64).sqrt();
        for i in 0..k {
            z[(i, k - 1)] = 1.0 / norm;
        }
        z[(k, k - 1)] = -(k as f64) / norm;
    }
    z
}

/// `(V, edf, lambda)` of the Gaussian REML fit of `[1 | X]` with penalty
/// `blockdiag(0, S)`.
fn reml_of(design_block: &Array2<f64>, penalty_block: &Array2<f64>, y: &Array1<f64>) -> Option<(f64, f64, f64)> {
    let (n, p) = design_block.dim();
    let mut design = Array2::<f64>::ones((n, p + 1));
    design.slice_mut(ndarray::s![.., 1..]).assign(design_block);
    let mut penalty = Array2::<f64>::zeros((p + 1, p + 1));
    penalty
        .slice_mut(ndarray::s![1.., 1..])
        .assign(penalty_block);
    let y2 = y.view().insert_axis(ndarray::Axis(1));
    let fit =
        gaussian_reml_multi_closed_form(design.view(), y2, penalty.view(), None, None).ok()?;
    Some((fit.reml_score, fit.edf, fit.lambda))
}

/// Model C: the coherent single-range RKHS design/penalty pair.
fn coherent_reml(
    feats: &Array2<f64>,
    centers: ArrayView2<'_, f64>,
    y: &Array1<f64>,
    kappa: f64,
    ell: f64,
) -> Option<(f64, f64, f64)> {
    let z = sum_to_zero_frame(centers.nrows());
    let k_dc = constant_curvature_kernel_matrix(feats.view(), centers, kappa, ell).ok()?;
    let k_cc = constant_curvature_kernel_matrix(centers, centers, kappa, ell).ok()?;
    let x = k_dc.dot(&z);
    let s_raw = z.t().dot(&k_cc).dot(&z);
    let s = (&s_raw + &s_raw.t()) * 0.5;
    reml_of(&x, &s, y)
}

/// Model F: the shipped builder (fill-invariant `L(κ)` / `L_S(κ)`).
fn shipped_reml(
    feats: &Array2<f64>,
    centers: &Array2<f64>,
    y: &Array1<f64>,
    kappa: f64,
    ell_ref: f64,
) -> Option<(f64, f64, f64)> {
    let spec = ConstantCurvatureBasisSpec {
        center_strategy: CenterStrategy::UserProvided(centers.clone()),
        kappa,
        kappa_fixed: false,
        length_scale: ell_ref,
        double_penalty: false,
        identifiability: ConstantCurvatureIdentifiability::CenterSumToZero,
    };
    let basis = build_constant_curvature_basis(feats.view(), &spec).ok()?;
    let x = basis.design.to_dense();
    let s = basis.active_penalties[0].matrix.clone();
    reml_of(&x, &s, y)
}

struct Fixture {
    feats: Array2<f64>,
    centers: Array2<f64>,
    y: Array1<f64>,
    ell_ref: f64,
    cap: f64,
}

/// `n` chart points in a radius-`radius` disk; `m` centers on a ring plus the
/// origin (κ-independent, deterministic, and off-origin so the plant's pairwise
/// pattern carries curvature); response = a smooth superposition of κ⋆ kernel
/// sections at `truth_ell`, standardized, plus noise.
fn fixture(
    n: usize,
    m: usize,
    kappa_star: f64,
    radius: f64,
    truth_ell_mult: f64,
    noise_sd: f64,
    seed: u64,
) -> Fixture {
    let mut st = seed;
    let mut feats = Array2::<f64>::zeros((n, 2));
    let mut noise = Array1::<f64>::zeros(n);
    for i in 0..n {
        let (x1, x2) = loop {
            let a = 2.0 * next_unit(&mut st) - 1.0;
            let b = 2.0 * next_unit(&mut st) - 1.0;
            if a * a + b * b <= 1.0 {
                break (a * radius, b * radius);
            }
        };
        feats[(i, 0)] = x1;
        feats[(i, 1)] = x2;
        noise[i] = next_gauss(&mut st);
    }
    // Centers: origin + a ring at 2/3 the data radius.
    let mut centers = Array2::<f64>::zeros((m, 2));
    for k in 1..m {
        let th = std::f64::consts::TAU * ((k - 1) as f64) / ((m - 1) as f64);
        centers[(k, 0)] = 0.667 * radius * th.cos();
        centers[(k, 1)] = 0.667 * radius * th.sin();
    }
    // ℓ_ref: the builder's own auto rule (median pairwise chart distance among
    // centers, doubled to the κ = 0 gauge).
    let mut dists = Vec::new();
    for i in 0..m {
        for j in (i + 1)..m {
            let dx = centers[(i, 0)] - centers[(j, 0)];
            let dy = centers[(i, 1)] - centers[(j, 1)];
            dists.push(2.0 * (dx * dx + dy * dy).sqrt());
        }
    }
    dists.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let ell_ref = dists[dists.len() / 2];

    // The truth must be a member of the SPAN the fit searches, not merely of
    // the raw kernel's column space: the realized design is `K·z` with `z` the
    // sum-to-zero frame, so a plant `K·w` with `Σw ≠ 0` leaves a component
    // along `K·1` that lies in NO κ-span and adds a second misspecification on
    // top of the range one this probe is about. Planting `X·v` in the model-C
    // design at (κ⋆, truth_ell) makes the truth exactly reachable at κ⋆ and at
    // no other curvature.
    let truth_ell = ell_ref * truth_ell_mult;
    let z = sum_to_zero_frame(m);
    let k_truth =
        constant_curvature_kernel_matrix(feats.view(), centers.view(), kappa_star, truth_ell)
            .expect("truth kernel");
    let x_truth = k_truth.dot(&z);
    let mut y = Array1::<f64>::zeros(n);
    for j in 0..x_truth.ncols() {
        let w = 1.0 / (1.0 + j as f64);
        for i in 0..n {
            y[i] += w * x_truth[(i, j)];
        }
    }
    let mean = y.iter().sum::<f64>() / n as f64;
    let sd = (y.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n as f64).sqrt();
    for i in 0..n {
        y[i] = (y[i] - mean) / sd + noise_sd * noise[i];
    }
    let mut max_r2 = 0.0_f64;
    for row in feats.outer_iter() {
        max_r2 = max_r2.max(row.dot(&row));
    }
    for row in centers.outer_iter() {
        max_r2 = max_r2.max(row.dot(&row));
    }
    Fixture {
        feats,
        centers,
        y,
        ell_ref,
        cap: 0.5 / max_r2,
    }
}

#[test]
fn probe_coherent_range_profiled_criterion_recovers_kappa_star() {
    gam::init_parallelism();
    let n = 200usize;
    let m = 7usize;
    let radius = 0.6_f64;
    let noise_sd = 0.05_f64;
    let seed = 0x5EED_2747_0000_0000_u64;

    for kappa_star in [-1.0_f64, 0.0, 1.0] {
        for truth_ell_mult in [0.5_f64, 1.0, 2.0] {
            let f = fixture(n, m, kappa_star, radius, truth_ell_mult, noise_sd, seed);
            eprintln!(
                "\n=== κ⋆={kappa_star:+.2}  truth_ℓ={:.5} ({truth_ell_mult}×ℓ_ref={:.5})  cap=±{:.4} ===",
                f.ell_ref * truth_ell_mult,
                f.ell_ref,
                f.cap
            );
            eprintln!(
                "  kappa      | F@ℓref        F-prof        ℓ̂_F      | C@ℓref        C-prof        ℓ̂_C"
            );
            let mut arg = [(f64::INFINITY, f64::NAN); 4];
            let mut ellhat = [f64::NAN; 2];
            for i in 0..=24 {
                let kappa = -f.cap + 2.0 * f.cap * (i as f64) / 24.0;
                let vf = shipped_reml(&f.feats, &f.centers, &f.y, kappa, f.ell_ref);
                let vc = coherent_reml(&f.feats, f.centers.view(), &f.y, kappa, f.ell_ref);
                let mut bf = (f64::INFINITY, f64::NAN);
                let mut bc = (f64::INFINITY, f64::NAN);
                for j in 0..=48 {
                    let ell = f.ell_ref * (-1.8_f64 + 3.6 * (j as f64) / 48.0).exp();
                    if let Some((v, _, _)) = shipped_reml(&f.feats, &f.centers, &f.y, kappa, ell)
                        && v < bf.0
                    {
                        bf = (v, ell);
                    }
                    if let Some((v, _, _)) = coherent_reml(&f.feats, f.centers.view(), &f.y, kappa, ell)
                        && v < bc.0
                    {
                        bc = (v, ell);
                    }
                }
                let vf0 = vf.map(|t| t.0).unwrap_or(f64::NAN);
                let vc0 = vc.map(|t| t.0).unwrap_or(f64::NAN);
                for (slot, value) in [(0, vf0), (1, bf.0), (2, vc0), (3, bc.0)] {
                    if value < arg[slot].0 {
                        arg[slot] = (value, kappa);
                        if slot == 1 {
                            ellhat[0] = bf.1;
                        }
                        if slot == 3 {
                            ellhat[1] = bc.1;
                        }
                    }
                }
                eprintln!(
                    "  k={kappa:<9.4} | {vf0:<13.6} {:<13.6} {:<8.4} | {vc0:<13.6} {:<13.6} {:<8.4}",
                    bf.0, bf.1, bc.0, bc.1
                );
            }
            let interior = |k: f64| k.abs() < f.cap * 0.999;
            eprintln!(
                "  -> F@ℓref argmin {:+.4} (int={})  F-prof argmin {:+.4} (int={}, ℓ̂={:.4})",
                arg[0].1,
                interior(arg[0].1),
                arg[1].1,
                interior(arg[1].1),
                ellhat[0]
            );
            eprintln!(
                "  -> C@ℓref argmin {:+.4} (int={})  C-prof argmin {:+.4} (int={}, ℓ̂={:.4})   [truth {kappa_star:+.2}]",
                arg[2].1,
                interior(arg[2].1),
                arg[3].1,
                interior(arg[3].1),
                ellhat[1]
            );
        }
    }
}
