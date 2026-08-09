//! #2750 probe — the outer ψ box the `ln ℓ` search is given, against the box
//! the term's own geometry defines.
//!
//! `measure_jet_psi_bound_values` hands every measure-jet ψ coordinate the same
//! kind of box: a chosen absolute interval. For the two PENALTY dials (`α`,
//! `ln τ`) that is right — they are dimensionless and no geometry bounds them.
//! For `ln ℓ` it is not: `ℓ` is a LENGTH in the chart the basis is realized in,
//! and the term already derives both of its walls,
//!
//! * FLOOR — the median nearest-node spacing (`MeasureJetBand::eps[0]`, which is
//!   also the auto range): below it neighbouring representers stop overlapping
//!   and the design degenerates from a partition of unity into a bump-per-node
//!   indicator;
//! * CEILING — the node bounding-box diagonal
//!   (`MeasureJetRangeBracket::ceiling`): at that range every PAIR of
//!   representers overlaps at `≥ exp(−1/2)`, so the block is numerically one
//!   function plus the affine head and there is no distinct model past it.
//!
//! This prints both, per fixture, next to the shipped absolute window, so the
//! ratio of the two boxes is a measured number. It matters because a
//! trust-region method scales its first step to the box it is given: the
//! measured first `ln ℓ` step on the parity fixture is `0.69`, rejected twelve
//! times, and every rejection is a full design realization.
//!
//! Diagnostic-only; asserts only that each fixture produced a finite window.

use csv::StringRecord;
use gam::smooth::SmoothBasisSpec;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};

/// The absolute `ln ℓ` window `MEASURE_JET_PSI_LN_LENGTH_SCALE_BOUNDS` ships,
/// restated here so the probe compares against the shipped numbers rather than
/// against a description of them.
const SHIPPED_LN_ELL_BOX: (f64, f64) = (-6.907_755_278_982_137, 4.605_170_185_988_092);

fn clamp_unit_open(x: f64) -> f64 {
    x.max(1.0e-6).min(1.0 - 1.0e-6)
}

/// The `measure_jet_perf_parity` fixture: a 1-D curve in 3-D.
fn parity_dataset(n: usize, sigma: f64, seed: u64) -> gam::data::EncodedDataset {
    let mut rng = StdRng::seed_from_u64(seed);
    let latent = Uniform::new(0.0, 1.0).expect("uniform latent");
    let noise = Normal::new(0.0, sigma).expect("normal noise");
    let headers = ["x0", "x1", "x2", "y"]
        .iter()
        .cloned()
        .map(String::from)
        .collect();
    let rows: Vec<StringRecord> = (0..n)
        .map(|_| {
            let t = latent.sample(&mut rng);
            let coords = [
                clamp_unit_open(t),
                clamp_unit_open(0.5 + 0.5 * (2.0 * std::f64::consts::PI * t).sin()),
                clamp_unit_open(t * t),
            ];
            let y = (2.0 * std::f64::consts::PI * t).sin()
                + 0.5 * (4.0 * std::f64::consts::PI * t).cos()
                + noise.sample(&mut rng);
            StringRecord::from(vec![
                coords[0].to_string(),
                coords[1].to_string(),
                coords[2].to_string(),
                y.to_string(),
            ])
        })
        .collect();
    encode_recordswith_inferred_schema(headers, rows).expect("encode parity dataset")
}

/// The `measure_jet_formula_fit_robustness_sweep` seed-1 rows: a single-cycle
/// sine on a regular 1-D grid.
fn sweep_dataset() -> gam::data::EncodedDataset {
    fn hashed_unit(index: u64) -> f64 {
        let mut z = index.wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 11) as f64 / (1u64 << 53) as f64
    }
    let headers = ["x", "y"]
        .into_iter()
        .map(str::to_string)
        .collect::<Vec<_>>();
    let rows = (0..200usize)
        .map(|i| {
            let x = i as f64 / 199.0;
            let draw = 2.0
                * hashed_unit(
                    (i as u64)
                        .wrapping_mul(2_654_435_761)
                        .wrapping_add(1u64.wrapping_mul(0x9E37_79B9)),
                )
                - 1.0;
            let y = (std::f64::consts::TAU * x).sin() + 0.10 * draw;
            StringRecord::from(vec![format!("{x:.17e}"), format!("{y:.17e}")])
        })
        .collect::<Vec<_>>();
    encode_recordswith_inferred_schema(headers, rows).expect("encode sweep dataset")
}

/// Axis-aligned bounding-box diagonal — the same deterministic diameter proxy
/// `measure_jet_range_bracket` measures its ceiling with.
fn bounding_box_diagonal(points: &ndarray::Array2<f64>) -> f64 {
    let mut squared = 0.0_f64;
    for k in 0..points.ncols() {
        let column = points.column(k);
        let lo = column.iter().copied().fold(f64::INFINITY, f64::min);
        let hi = column.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        if lo.is_finite() && hi.is_finite() {
            squared += (hi - lo) * (hi - lo);
        }
    }
    squared.sqrt()
}

fn report(label: &str, body: &str, data: &gam::data::EncodedDataset) {
    let config = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula(&format!("y ~ {body}"), data, &config)
        .unwrap_or_else(|error| panic!("{label}: {error}"));
    let FitResult::Standard(fit) = result else {
        panic!("{label}: expected a standard Gaussian fit");
    };
    let term = fit
        .resolvedspec
        .smooth_terms
        .iter()
        .find_map(|term| match &term.basis {
            SmoothBasisSpec::MeasureJet { spec, .. } => Some(spec.clone()),
            _ => None,
        })
        .expect("the fitted collection carries a measure-jet term");
    let gam::basis::CenterStrategy::UserProvided(centers) = &term.center_strategy else {
        panic!("{label}: a frozen measure-jet term pins its quadrature nodes");
    };
    let frozen = term
        .frozen_quadrature
        .as_ref()
        .expect("a frozen measure-jet term carries its fit-time quadrature");
    let floor = frozen.eps_band[0];
    let ceiling = bounding_box_diagonal(centers);
    let geometry_width = ceiling.ln() - floor.ln();
    let shipped_width = SHIPPED_LN_ELL_BOX.1 - SHIPPED_LN_ELL_BOX.0;
    println!(
        "[2750-box] {label:<12} m={:<4} ell={:.6} ln_ell={:+.4}  \
         geometry_box=[{:+.4}, {:+.4}] width={:.4}  shipped_box=[{:+.4}, {:+.4}] width={:.4}  \
         shipped/geometry={:.2}x  band_top={:.6}",
        centers.nrows(),
        term.length_scale,
        term.length_scale.ln(),
        floor.ln(),
        ceiling.ln(),
        geometry_width,
        SHIPPED_LN_ELL_BOX.0,
        SHIPPED_LN_ELL_BOX.1,
        shipped_width,
        shipped_width / geometry_width,
        frozen.eps_band[frozen.eps_band.len() - 1],
    );
    assert!(
        floor.is_finite() && floor > 0.0 && ceiling.is_finite() && ceiling > floor,
        "{label}: the geometry window must be a nondegenerate positive interval"
    );
}

#[test]
fn measure_jet_psi_box_against_its_own_geometry_2750() {
    init_parallelism();
    report(
        "parity/3d",
        "mjs(x0, x1, x2, centers=16)",
        &parity_dataset(1_500, 0.10, 1_039),
    );
    report("sweep/1d", "s(x, bs=\"mjs\")", &sweep_dataset());
}
