//! #2601 mechanism 3 capture: run the `shape=monotone_decreasing` fit that
//! refuses with "truncated moments for an 11-dimensional constraint face did
//! not converge" and print the face the cubature was actually handed.
//!
//! The synthetic sweep in `constrained_posterior.rs` established that
//! CORRELATION between constraint normals, not tail depth, is what breaks the
//! rule — but a synthetic AR(1) face is a stand-in. This lifts the real one out
//! so the remedy is chosen against it.

use gam::{FitConfig, fit_from_formula, init_parallelism, load_csvwith_inferred_schema};
use std::io::Write;

#[test]
fn probe_2601_capture_the_face_that_refuses() {
    init_parallelism();
    gam::progress_log::init_logging_at(log::LevelFilter::Debug);

    // Same shape as `tests/test_python_api.py::
    // test_gaussian_reml_fit_all_shape_constraints_do_not_panic`: 300 rows of
    // clean increasing linear data. A `monotone_decreasing` constraint binds on
    // every coefficient, which is what produces the fully-active 11-row face.
    let n = 300usize;
    let mut state: u64 = 0x2601_0000_0000_0005;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64) / ((1u64 << 53) as f64)
    };
    let mut x: Vec<f64> = (0..n).map(|_| next()).collect();
    x.sort_by(|a, b| a.partial_cmp(b).expect("finite covariate"));
    let y: Vec<f64> = x
        .iter()
        .map(|xi| {
            let u1 = next().max(1e-12);
            let u2 = next();
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            2.0 * xi + 0.1 * z
        })
        .collect();

    let mut csv = String::from("x,y\n");
    for i in 0..n {
        csv.push_str(&format!("{:.17e},{:.17e}\n", x[i], y[i]));
    }
    let mut tmp = std::env::temp_dir();
    tmp.push(format!("gam_2601_face_{}.csv", std::process::id()));
    {
        let mut f = std::fs::File::create(&tmp).expect("create csv");
        f.write_all(csv.as_bytes()).expect("write csv");
    }
    let ds = load_csvwith_inferred_schema(&tmp).expect("load csv");
    if let Err(err) = std::fs::remove_file(&tmp) {
        eprintln!("temporary csv {} was not removed: {err}", tmp.display());
    }

    for kind in [
        "monotone_decreasing",
        "convex",
        "concave",
        "monotone_increasing",
    ] {
        let formula = format!("y ~ s(x, shape={kind})");
        match fit_from_formula(&formula, &ds, &FitConfig::default()) {
            Ok(_) => println!("PROBE2601FACE {kind}: OK"),
            Err(e) => println!("PROBE2601FACE {kind}: ERR={e}"),
        }
    }
}
