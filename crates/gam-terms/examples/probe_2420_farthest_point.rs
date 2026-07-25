//! #2420 probe: what does farthest-point center selection cost, and how much of
//! that is the invariant tie-break rather than the maximin recursion itself?
//!
//! Times the two sibling selectors at the same `n` and the same center budget:
//!
//! * `select_spherical_farthest_point_centers` — geodesic maximin on S²;
//! * `select_thin_plate_knots` — its Euclidean twin, which drives every
//!   `FarthestPoint` center strategy (thin-plate, Duchon, Matérn) and seeds
//!   k-means.
//!
//! Each is run on two clouds that differ only in layout: a regular grid, where a
//! whole symmetry class ties every `O(1)` invariant key exactly, and an
//! irregular cloud, where nothing ties and the `O(n log n)` sorted-profile
//! tie-break can decide nothing. A selector that charges the tie-break correctly
//! is fast on the second and no worse than the symmetry warrants on the first.
//!
//! Run: `cargo run --release -p gam-terms --example probe_2420_farthest_point -- [n] [m]`

use gam_terms::basis::{select_spherical_farthest_point_centers, select_thin_plate_knots};
use ndarray::Array2;
use std::time::Instant;

/// Deterministic unit draws: SplitMix64's finalizer, no RNG state, no seed
/// coupling between rows.
fn hashed_unit(index: u64) -> f64 {
    let mut z = index.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
}

/// A regular lat/lon grid: the canonical gridded-geospatial layout.
fn latlon_grid(n: usize) -> Array2<f64> {
    let n_lon = (n as f64).sqrt().round().max(2.0) as usize;
    Array2::from_shape_fn((n, 2), |(row, col)| {
        let (i, j) = (row / n_lon, row % n_lon);
        if col == 0 {
            -85.0 + 170.0 * (i as f64 + 0.5) / (n.div_ceil(n_lon) as f64)
        } else {
            -180.0 + (360.0 * j as f64) / n_lon as f64
        }
    })
}

/// An area-uniform cloud on the same support, with no exact spherical symmetry.
fn latlon_cloud(n: usize) -> Array2<f64> {
    Array2::from_shape_fn((n, 2), |(row, col)| {
        if col == 0 {
            (1.0 - 2.0 * hashed_unit(2 * row as u64)).asin().to_degrees()
        } else {
            360.0 * hashed_unit(2 * row as u64 + 1) - 180.0
        }
    })
}

fn planar_grid(n: usize) -> Array2<f64> {
    let side = (n as f64).sqrt().round().max(2.0) as usize;
    Array2::from_shape_fn((n, 2), |(row, col)| {
        let (i, j) = (row / side, row % side);
        if col == 0 { i as f64 } else { j as f64 }
    })
}

fn planar_cloud(n: usize, d: usize) -> Array2<f64> {
    Array2::from_shape_fn((n, d), |(row, col)| {
        hashed_unit((row * d + col) as u64 ^ 0x5DEE_CE66_D000_0000)
    })
}

fn report(label: &str, n: usize, m: usize, elapsed: f64, rows: Result<usize, String>) {
    match rows {
        Ok(rows) => println!("  {label:<26} n={n:<8} m={m:<5} out={rows:<5} {elapsed:>8.3} s"),
        Err(e) => println!("  {label:<26} n={n:<8} m={m:<5} ERROR after {elapsed:>8.3} s: {e}"),
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let n: usize = args.next().and_then(|a| a.parse().ok()).unwrap_or(50_000);
    let m: usize = args.next().and_then(|a| a.parse().ok()).unwrap_or(200);

    println!("rayon threads: {}", rayon::current_num_threads());

    println!("sphere: select_spherical_farthest_point_centers");
    for (label, data) in [
        ("regular lat/lon grid", latlon_grid(n)),
        ("irregular cloud", latlon_cloud(n)),
    ] {
        let start = Instant::now();
        let outcome = select_spherical_farthest_point_centers(data.view(), m, false);
        let elapsed = start.elapsed().as_secs_f64();
        report(
            label,
            n,
            m,
            elapsed,
            outcome.map(|c| c.nrows()).map_err(|e| e.to_string()),
        );
    }

    println!("plane: select_thin_plate_knots");
    for (label, data) in [
        ("regular grid d=2", planar_grid(n)),
        ("irregular cloud d=2", planar_cloud(n, 2)),
        ("irregular cloud d=8", planar_cloud(n, 8)),
    ] {
        let start = Instant::now();
        let outcome = select_thin_plate_knots(data.view(), m);
        let elapsed = start.elapsed().as_secs_f64();
        report(
            label,
            n,
            m,
            elapsed,
            outcome.map(|k| k.nrows()).map_err(|e| e.to_string()),
        );
    }
}
