//! #2751 probe, basis layer: does the measure-jet jet-energy annihilate the
//! ambient-affine span EXACTLY, as its own module header claims?
//!
//! The end-to-end measurement (gam#2751) found the BMS log-slope surface
//! collapsing onto a single 45-degree linear direction when REML pushes the
//! energy's lambda up. The ridge-limit projection of a noiseless plane onto the
//! shipped design reproduced it with no estimator in the loop, which points the
//! finger at the energy's null space rather than at the fit. This probe removes
//! the design, the gauge and the collection too: it asks the energy form
//! itself, in center-value space, what it does to `{1, c_1, ..., c_d}`.
//!
//! Printed, not asserted — the assertion belongs with the fix.

use gam_terms::basis::{measure_jet_band, measure_jet_energy_form};
use ndarray::{Array1, Array2};

/// Energy of every affine center-value direction, reported as the 3x3 (or
/// (d+1)x(d+1)) Gram `A' Q A` relative to `||Q||_F`. Exact annihilation means
/// every entry is at roundoff.
fn affine_energy_report(label: &str, centers: &Array2<f64>, masses: &Array1<f64>, scales: usize) {
    let band = measure_jet_band(centers.view(), scales).expect("band");
    let q = measure_jet_energy_form(centers.view(), masses.view(), &band, 1.0, 1.0, 1e-3)
        .expect("energy form");
    let qnorm = q.iter().map(|v| v * v).sum::<f64>().sqrt();
    let m = centers.nrows();
    let d = centers.ncols();
    let mut affine = Array2::<f64>::ones((m, d + 1));
    affine.slice_mut(ndarray::s![.., 1..]).assign(centers);
    let gram = affine.t().dot(&q).dot(&affine);
    let scale: Vec<f64> = (0..=d)
        .map(|k| affine.column(k).mapv(|v| v * v).sum().sqrt())
        .collect();
    println!("[2751-basis {label}] m={m} d={d} ||Q||_F={qnorm:.6e} band={:?}", band.eps);
    for a in 0..=d {
        let row: Vec<String> = (0..=d)
            .map(|b| {
                format!(
                    "{:.3e}",
                    gram[[a, b]] / (qnorm * scale[a] * scale[b]).max(f64::MIN_POSITIVE)
                )
            })
            .collect();
        println!("[2751-basis {label}]   relative A'QA row {a}: [{}]", row.join(", "));
    }
    // The worst normalized Rayleigh quotient over the affine span is the number
    // that decides whether a large lambda can delete an affine direction.
    let mut worst = 0.0_f64;
    let mut worst_dir = vec![0.0; d + 1];
    // Sweep the linear part of the affine span on a fine angular grid (d = 2)
    // plus each coordinate axis; enough to expose a direction the energy fails
    // to annihilate.
    if d == 2 {
        for step in 0..3600 {
            let theta = std::f64::consts::PI * (step as f64) / 1800.0;
            let v = Array1::from(vec![0.0, theta.cos(), theta.sin()]);
            let f = affine.dot(&v);
            let num = f.dot(&q.dot(&f));
            let den = f.dot(&f);
            let r = num / den.max(f64::MIN_POSITIVE) / qnorm;
            if r > worst {
                worst = r;
                worst_dir = v.to_vec();
            }
        }
        println!(
            "[2751-basis {label}]   worst affine Rayleigh/||Q||_F = {worst:.3e} at direction \
             ({:.4}, {:.4})",
            worst_dir[1], worst_dir[2]
        );
    }
}

fn uniform_masses(m: usize) -> Array1<f64> {
    Array1::from_elem(m, 1.0 / m as f64)
}

#[test]
fn probe_2751_energy_affine_null_space() {
    // 1. A regular 4x4 grid on the standardized square: the cleanest possible
    //    geometry, no degeneracy anywhere.
    let mut grid = Array2::<f64>::zeros((16, 2));
    for i in 0..4 {
        for j in 0..4 {
            grid[[i * 4 + j, 0]] = -1.5 + i as f64;
            grid[[i * 4 + j, 1]] = -1.5 + j as f64;
        }
    }
    affine_energy_report("grid4x4", &grid, &uniform_masses(16), 3);

    // 2. Scattered centers on the same square (SplitMix64), the shape the
    //    farthest-point strategy actually produces.
    let mut state = 0x2751_2026_0803_0001u64;
    let mut next = || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
    };
    let mut scattered = Array2::<f64>::zeros((16, 2));
    for i in 0..16 {
        scattered[[i, 0]] = 3.4 * (next() - 0.5);
        scattered[[i, 1]] = 3.4 * (next() - 0.5);
    }
    affine_energy_report("scattered16", &scattered, &uniform_masses(16), 3);

    // 3. One dimension compressed by 10x: the anisotropic case where a local
    //    affine projection could plausibly lose the weak direction.
    let mut squashed = scattered.clone();
    for i in 0..16 {
        squashed[[i, 1]] *= 0.1;
    }
    affine_energy_report("squashed16", &squashed, &uniform_masses(16), 3);

    // 4. Single-scale band, the mode the fixture runs in when `scales` is
    //    small — isolates whether the leak is a per-scale or a band effect.
    affine_energy_report("scattered16-1scale", &scattered, &uniform_masses(16), 1);
}
