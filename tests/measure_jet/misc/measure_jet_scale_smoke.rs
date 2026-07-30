//! Measure-jet frame acceptance gate 5 (docs/measure_jet_frame.md §7.5): scale smoke —
//! "fast" as a CI gate, not a vibe. Gates the measure-jet BUILD path (no
//! REML fit): masses O(n·m·d), design O(n·m·d), energy O(m²·d·L). The point
//! is catching an O(n²) regression anywhere in that path, not benchmarking.
//!
//! Row-count note. FarthestPoint center selection is NOT the bottleneck:
//! `select_thin_plate_knots` keeps a maintained min-distance array, so each
//! added center costs one rayon-parallel O(n·d) sweep — O(n·m·d) total,
//! linear in n. That claim only became true with the #2420 tie-break
//! accounting: until then the selector ALSO built two serial length-n sorted
//! support-distance profiles per added center — an unconditional
//! `2·m · O(n·d + n log n)` on top of the maintained sweeps, and ~3e9
//! operations at this fixture's `n = 200_000, d = 8, m = 300` — because its
//! invariant tie-break was written as a two-argument comparator that rebuilt
//! both profiles on every call, including the call that compares the sole
//! surviving candidate against itself. What forces the drop from the charter's 10⁶ rows to
//! n = 200_000 is the constraint-transform GEMM inside the build
//! (`raw_design · z`, an (n×m)·(m×(m−1)) product): it is O(n·m²) — already
//! the asymptotically dominant term over the documented O(n·m·d) passes —
//! and CI executes tests at opt-level 0 (no [profile.*] opt override in
//! Cargo.toml; test.yml runs `cargo test --config profile.dev.debug=0`),
//! where the ≈1.8e11-flop product at n = 10⁶ alone takes minutes. At
//! n = 200_000 the whole build sits well inside the bound while an O(n²)
//! row-pairwise regression (≥ 4e10 ops) would still blow it.
//!
//! Memory: the data matrix is built once (200_000 × 8 f64 ≈ 13 MB); the
//! transient peak is the raw n×m representer design plus its constrained
//! copy (≈ 0.5 GB each) — sane for CI.

use std::time::Instant;

use gam::basis::{BasisMetadata, CenterStrategy, MeasureJetBasisSpec, build_measure_jet_basis};
use ndarray::Array2;

const N_ROWS: usize = 200_000;
/// Quarter-size reference arm. The gate is the ratio `t(N_ROWS)/t(N_SMALL)`,
/// not either time on its own.
const N_SMALL: usize = 50_000;
const N_DIMS: usize = 8;
const N_CENTERS: usize = 300;
/// Ceiling on the measured 4x-rows time ratio.
///
/// This replaces an absolute `elapsed < 120s` bound. An absolute wall-clock
/// bound on a shared CI runner measures the RUNNER: it fails on a loaded box
/// with the build path healthy, and it passes on a fast box with a quadratic
/// term already back, so it never actually gated the property the test is
/// named for. The ratio of two builds of the SAME code on the SAME machine,
/// seconds apart, divides the machine speed out and measures the exponent
/// directly, which is the only thing this smoke claims to check.
///
/// Calibration. Every documented term of the build is linear in n at fixed
/// m and d - the O(n*m*d) mass/design passes, the maintained-min-distance
/// FarthestPoint sweeps, and the dominant O(n*m^2) constraint-transform GEMM
/// - while the O(m^2*d*L) energy assembly does not grow with n at all. So a
/// healthy 4x in rows costs AT MOST 4x in time, and in practice slightly
/// less. An O(n^2) row-pairwise regression costs 16x. The bound sits between
/// them in log space: 2x headroom over the healthy ceiling, 2x below the
/// regression it exists to catch.
const SCALING_RATIO_BOUND: f64 = 8.0;
/// Deterministic pseudo-noise amplitude off the filament backbone.
const JITTER: f64 = 1e-3;

/// SplitMix64 finalizer mapped to [0, 1): deterministic per-index jitter
/// with no RNG state and no seed dependence between rows.
fn hashed_unit(index: u64) -> f64 {
    let mut z = index.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    (z >> 11) as f64 / (1u64 << 53) as f64
}

/// Three deterministic 1-D strands (trig curves of the row index) embedded
/// in 8-D, separated by per-strand offset vectors, with hashed sub-resolution
/// jitter so the filament has honest thickness.
fn filament_coordinate(row: usize, dim: usize, n_rows: usize) -> f64 {
    let strand = (row % 3) as f64;
    let t = (row / 3) as f64 / (n_rows / 3) as f64;
    let k = dim as f64;
    let freq = 1.0 + 0.45 * k + 0.6 * strand;
    let phase = 0.8 * strand + 0.37 * k;
    let amp = 1.0 / (1.0 + 0.25 * k);
    let drift = (0.6 - 0.15 * k) * (strand - 1.0);
    let backbone = amp * (std::f64::consts::TAU * 0.35 * freq * t + phase).sin()
        + drift * t
        + 1.7 * strand * (0.9 * k).cos();
    let jitter = JITTER * (2.0 * hashed_unit((row * N_DIMS + dim) as u64) - 1.0);
    backbone + jitter
}

/// Build the basis once at `n` rows and return the elapsed seconds alongside
/// the built basis.
fn timed_build(n: usize) -> (f64, gam::basis::BasisBuildResult) {
    let data = Array2::<f64>::from_shape_fn((n, N_DIMS), |(i, k)| filament_coordinate(i, k, n));

    // Multiscale (per-scale spectral split) is an explicit opt-in (#1116);
    // this smoke exercises the spectral build path, so it opts in.
    let spec = MeasureJetBasisSpec {
        center_strategy: CenterStrategy::FarthestPoint {
            num_centers: N_CENTERS,
        },
        multiscale: true,
        ..Default::default()
    };

    let started = Instant::now();
    // (a) the build must succeed at filament scale.
    let built = build_measure_jet_basis(data.view(), &spec)
        .unwrap_or_else(|e| panic!("measure-jet build must succeed on an {n}-row 8-D filament: {e}"));
    let elapsed = started.elapsed().as_secs_f64();
    println!(
        "measure-jet scale smoke: n={n} d={N_DIMS} m={N_CENTERS} \
         build={elapsed:.2}s rate={:.0} rows/s",
        n as f64 / elapsed.max(f64::MIN_POSITIVE)
    );
    (elapsed, built)
}

#[test]
fn measure_jet_build_scale_smoke_200k_rows() {
    // Reference arm first, so a machine that is warming up penalises the
    // SMALL time (raising the ratio) rather than flattering it.
    let (t_small, _small) = timed_build(N_SMALL);
    let (t_big, built) = timed_build(N_ROWS);

    // (b) scaling gate: 4x the rows must not cost more than SCALING_RATIO_BOUND
    //     times the seconds. An O(n²) pass over the rows costs 16x.
    let ratio = t_big / t_small;
    println!(
        "measure-jet scale smoke: rows {N_SMALL}->{N_ROWS} (4.0x) cost \
         {t_small:.2}s->{t_big:.2}s (ratio {ratio:.2}x, bound {SCALING_RATIO_BOUND:.1}x)"
    );
    assert!(
        ratio < SCALING_RATIO_BOUND,
        "measure-jet build time grew {ratio:.2}x for a 4x row increase \
         ({N_SMALL} rows: {t_small:.2}s -> {N_ROWS} rows: {t_big:.2}s), over the \
         {SCALING_RATIO_BOUND:.1}x bound — every documented term of the build is \
         linear in n at fixed m and d, so this is an O(n²) regression in the \
         build path"
    );

    // (c) per-level (spectral) mode under the multiscale opt-in (#1116): one
    // penalty candidate per band scale plus the double-penalty ridge.
    let BasisMetadata::MeasureJet {
        eps_band, order_s, ..
    } = &built.metadata
    else {
        panic!("measure-jet build must carry MeasureJet metadata");
    };
    assert_eq!(
        *order_s, 0.0,
        "default order keeps the auto (spectral) sentinel"
    );
    assert!(
        !eps_band.is_empty(),
        "realized scale band must be non-empty"
    );
    assert_eq!(
        built.active_penalties.len(),
        eps_band.len() + 1,
        "per-level candidate count must be band length ({}) + 1 ridge",
        eps_band.len()
    );

    // Shape sanity: every row designed against the m−1 sum-to-zero columns.
    assert_eq!(built.design.nrows(), N_ROWS);
    assert_eq!(built.design.ncols(), N_CENTERS - 1);
}
