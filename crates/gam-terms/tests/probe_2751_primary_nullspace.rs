//! #2751 probe, second layer: the energy form annihilates the affine span to
//! 1e-17 (`probe_2751_energy_affine_null`), so if a shipped measure-jet Primary
//! penalty fails to annihilate an ambient-linear direction, the loss happens
//! between the center-value form and the emitted coefficient penalty.
//!
//! This builds the basis exactly as the fit does (`build_measure_jet_basis`)
//! on a 2-D uniform sample, then asks the EMITTED Primary what it charges each
//! ambient-linear direction. Reported as: the spectrum of the emitted matrix,
//! and the least-squares reconstruction of a planted plane restricted to the
//! Primary's numerical null space at a sweep of ridge weights — the same
//! instrument the end-to-end probe used, with the fit and the collection
//! removed.
//!
//! Printed, not asserted.

use gam_terms::basis::{
    CenterStrategy, MeasureJetBasisSpec, MeasureJetIdentifiability, build_measure_jet_basis,
};
use ndarray::{Array1, Array2};

fn splitmix(state: &mut u64) -> f64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
}

/// Cyclic Jacobi eigenvalue iteration for small symmetric matrices; returns
/// (eigenvalues ascending, eigenvectors as columns).
fn jacobi_eigh(input: &Array2<f64>) -> (Vec<f64>, Array2<f64>) {
    let n = input.nrows();
    let mut a = input.clone();
    let mut v = Array2::<f64>::eye(n);
    for _sweep in 0..100 {
        let off = (0..n)
            .flat_map(|i| ((i + 1)..n).map(move |j| (i, j)))
            .map(|(i, j)| a[[i, j]] * a[[i, j]])
            .sum::<f64>();
        if off <= 1e-30 * a.iter().map(|x| x * x).sum::<f64>().max(f64::MIN_POSITIVE) {
            break;
        }
        for p in 0..n {
            for q in (p + 1)..n {
                if a[[p, q]].abs() < 1e-300 {
                    continue;
                }
                let theta = (a[[q, q]] - a[[p, p]]) / (2.0 * a[[p, q]]);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..n {
                    let akp = a[[k, p]];
                    let akq = a[[k, q]];
                    a[[k, p]] = c * akp - s * akq;
                    a[[k, q]] = s * akp + c * akq;
                }
                for k in 0..n {
                    let apk = a[[p, k]];
                    let aqk = a[[q, k]];
                    a[[p, k]] = c * apk - s * aqk;
                    a[[q, k]] = s * apk + c * aqk;
                }
                for k in 0..n {
                    let vkp = v[[k, p]];
                    let vkq = v[[k, q]];
                    v[[k, p]] = c * vkp - s * vkq;
                    v[[k, q]] = s * vkp + c * vkq;
                }
            }
        }
    }
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&i, &j| a[[i, i]].partial_cmp(&a[[j, j]]).unwrap());
    let evals: Vec<f64> = idx.iter().map(|&i| a[[i, i]]).collect();
    let mut evecs = Array2::<f64>::zeros((n, n));
    for (c, &i) in idx.iter().enumerate() {
        evecs.column_mut(c).assign(&v.column(i));
    }
    (evals, evecs)
}

fn spec(centers: usize, multiscale: bool, scales: usize) -> MeasureJetBasisSpec {
    MeasureJetBasisSpec {
        center_strategy: CenterStrategy::FarthestPoint {
            num_centers: centers,
        },
        order_s: 0.0,
        alpha: 1.0,
        tau0: 1e-3,
        num_scales: scales,
        length_scale: 0.0,
        double_penalty: true,
        learn_length_scale: false,
        multiscale,
        identifiability: MeasureJetIdentifiability::CenterSumToZero,
        frozen_quadrature: None,
    }
}

#[test]
fn probe_2751_primary_annihilates_every_ambient_linear_direction() {
    let n = 1500;
    let mut state = 0x2751_2026_0803_0002u64;
    let mut data = Array2::<f64>::zeros((n, 2));
    for i in 0..n {
        // Standardized-looking uniform square, the frame the engine fits in.
        data[[i, 0]] = 3.46 * (splitmix(&mut state) - 0.5);
        data[[i, 1]] = 3.46 * (splitmix(&mut state) - 0.5);
    }
    let built = build_measure_jet_basis(data.view(), &spec(16, false, 3)).expect("build basis");
    let design = built.design.to_dense();
    let (rows, p) = design.dim();
    println!("[2751-primary] design {rows}x{p} penalties={}", built.active_penalties.len());
    for (k, pen) in built.active_penalties.iter().enumerate() {
        let (evals, evecs) = jacobi_eigh(&pen.matrix);
        let top = evals.last().copied().unwrap_or(0.0).max(f64::MIN_POSITIVE);
        let rel: Vec<String> = evals.iter().map(|e| format!("{:.2e}", e / top)).collect();
        println!(
            "[2751-primary] penalty#{k} source={:?} nullity={} declared_frame={:?} \
             relative spectrum=[{}]",
            pen.info.source,
            pen.nullity,
            pen.info.structural_null_frame.as_ref().map(|f| f.ncols()),
            rel.join(", ")
        );
        // For the four cheapest directions, what function do they make? Report
        // the ambient-linear tilt of each eigenvector's realized surface.
        for c in 0..4.min(p) {
            let coef = evecs.column(c).to_owned();
            let f = design.dot(&coef);
            let (b1, b2, nl) = linear_tilt(&data, &f);
            println!(
                "[2751-primary]   penalty#{k} eigen{c} rel_eval={:.3e} tilt=({b1:.4}, {b2:.4}) \
                 nonlinear_frac={nl:.4}",
                evals[c] / top
            );
        }
    }
}

/// Least-squares tilt of `f` against `{1, x1, x2}` on the sample, plus the
/// fraction of `f`'s variance that the plane does not explain.
fn linear_tilt(data: &Array2<f64>, f: &Array1<f64>) -> (f64, f64, f64) {
    let n = f.len() as f64;
    let mean = |v: &[f64]| v.iter().sum::<f64>() / n;
    let x1: Vec<f64> = data.column(0).to_vec();
    let x2: Vec<f64> = data.column(1).to_vec();
    let fv: Vec<f64> = f.to_vec();
    let (m1, m2, mf) = (mean(&x1), mean(&x2), mean(&fv));
    let dot = |a: &[f64], am: f64, b: &[f64], bm: f64| {
        a.iter()
            .zip(b)
            .map(|(u, v)| (u - am) * (v - bm))
            .sum::<f64>()
    };
    // Solve the 2x2 normal equations rather than assuming orthogonality: a
    // random sample's coordinate columns are only approximately uncorrelated.
    let (s11, s12, s22) = (
        dot(&x1, m1, &x1, m1),
        dot(&x1, m1, &x2, m2),
        dot(&x2, m2, &x2, m2),
    );
    let (r1, r2) = (dot(&x1, m1, &fv, mf), dot(&x2, m2, &fv, mf));
    let det = s11 * s22 - s12 * s12;
    let (b1, b2) = ((s22 * r1 - s12 * r2) / det, (s11 * r2 - s12 * r1) / det);
    let mut resid = 0.0;
    for i in 0..fv.len() {
        let r = (fv[i] - mf) - b1 * (x1[i] - m1) - b2 * (x2[i] - m2);
        resid += r * r;
    }
    let total = dot(&fv, mf, &fv, mf).max(f64::MIN_POSITIVE);
    (b1, b2, resid / total)
}
