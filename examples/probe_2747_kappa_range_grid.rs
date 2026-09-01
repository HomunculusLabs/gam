//! gam#2747: the 3 curvatures x 3 ranges grid, measured through the SHIPPED
//! estimator end to end.
//!
//! The issue's own acceptance is *"an interior optimum of `V_p(kappa)` near the
//! planted `kappa*` on signal-carrying data at matched edf, **on both curvature
//! signs**"*, and the table that motivated the whole redesign is a 3x3 sweep of
//! planted curvature against planted RANGE. Nothing in the tree measures that
//! grid: `constant_curvature_kappa_coverage_sims` cycles the three range
//! multipliers but only at `kappa* = +1` and `kappa* = 0`, and
//! `constant_curvature_kappa_inference_e2e` carries the hyperbolic arm but only
//! at the auto range (`1x`) -- the single configuration a range-blind criterion
//! was already able to handle. So the cell the pre-fix estimator got most
//! wrong -- hyperbolic truth at a range the heuristic does not supply -- is
//! measured nowhere.
//!
//! This probe closes that gap as a PRINTOUT first. It plants inside the
//! `kappa*` span at `ell = m * ell_ref` exactly as the coverage fixture does,
//! runs the real pipeline (`fit_term_collectionwith_spatial_length_scale_
//! optimization` + `curvature_inference_forspec`), and prints, per cell:
//! `kappa_hat`, its support, the CI, `ell_hat` against the planted range, and
//! the range support.
//!
//! Run: `cargo run --release --example probe_2747_kappa_range_grid`
//!
//! The shape is the coverage fixture's, so the two are comparable cell by cell:
//! `n = 120`, six centers, noise 0.10, chart radius 0.6.

use gam::basis::{
    build_constant_curvature_basis, constant_curvature_realized_centers,
    realized_constant_curvature_length_scale,
};
use gam::estimate::FitOptions;
use gam::inference::data::EncodedDataset;
use gam::inference::formula_dsl::parse_formula;
use gam::inference::model::{ColumnKindTag, DataSchema, SchemaColumn};
use gam::smooth::{
    CurvatureInference, SmoothBasisSpec, SpatialLengthScaleOptimizationOptions,
    TermCollectionSpec, curvature_inference_forspec,
    fit_term_collectionwith_spatial_length_scale_optimization,
};
use gam::terms::term_builder::build_termspec;
use gam::types::LikelihoodSpec;
use gam::utils::splitmix64;
use ndarray::{Array1, Array2};

const KAPPA_STARS: [f64; 3] = [-1.0, 0.0, 1.0];
const RANGE_MULTIPLIERS: [f64; 3] = [0.5, 1.0, 2.0];

/// The coverage fixture's shape, so a cell here and a replicate there are the
/// same experiment.
const N: usize = 120;
const CENTERS: usize = 6;
const REPS: usize = 1;
const NOISE_SD: f64 = 0.10;
const CHART_RADIUS: f64 = 0.6;

fn next_unit(state: &mut u64) -> f64 {
    (splitmix64(state) >> 11) as f64 / (1u64 << 53) as f64
}

fn next_gauss(state: &mut u64) -> f64 {
    let u1 = next_unit(state).max(1.0e-12);
    let u2 = next_unit(state);
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

fn termspec_for(formula: &str, frame: &Array2<f64>) -> TermCollectionSpec {
    let parsed = parse_formula(formula).expect("formula parses");
    let headers = vec!["y".to_string(), "x1".to_string(), "x2".to_string()];
    let ds = EncodedDataset {
        headers: headers.clone(),
        values: frame.clone(),
        schema: DataSchema {
            columns: headers
                .iter()
                .map(|name| SchemaColumn {
                    name: name.clone(),
                    kind: ColumnKindTag::Continuous,
                    levels: vec![],
                })
                .collect(),
        },
        column_kinds: vec![ColumnKindTag::Continuous; 3],
    };
    let col_map = ds.column_map();
    let mut notes = Vec::new();
    build_termspec(
        &parsed.terms,
        &ds,
        &col_map,
        &mut notes,
        &gam::ResourcePolicy::default_library(),
    )
    .expect("term spec")
}

/// The coverage fixture's generator, verbatim in its arithmetic: chart points
/// in a disk, a response drawn as a decaying combination of the `kappa*` basis's
/// own columns at `range_multiplier * ell_ref`, standardized, plus noise.
/// Returns the features, the response, and the PLANTED range.
fn dataset_on_m_kappa(
    formula: &str,
    n: usize,
    kappa_star: f64,
    radius: f64,
    noise_sd: f64,
    range_multiplier: f64,
    seed: u64,
) -> (Array2<f64>, Array1<f64>, f64) {
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
    let mut y = Array1::<f64>::zeros(n);
    let mut frame = Array2::<f64>::zeros((n, 3));
    for i in 0..n {
        frame[(i, 1)] = feats[(i, 0)];
        frame[(i, 2)] = feats[(i, 1)];
    }
    let fitspec = termspec_for(formula, &frame);
    let SmoothBasisSpec::ConstantCurvature { spec: cc, .. } = &fitspec.smooth_terms[0].basis else {
        panic!("the fixture formula must resolve to a constant-curvature term");
    };
    let mut truth_spec = cc.clone();
    truth_spec.kappa = kappa_star;
    truth_spec.kappa_fixed = true;
    truth_spec.double_penalty = false;
    let centers = constant_curvature_realized_centers(feats.view(), &truth_spec)
        .expect("the fixture cloud yields a realized center set");
    let ell_ref = realized_constant_curvature_length_scale(centers.view(), 0.0)
        .expect("the realized centers span a positive pairwise distance");
    let planted_ell = ell_ref * range_multiplier;
    truth_spec.length_scale = planted_ell;
    truth_spec.length_scale_fixed = true;
    let basis = build_constant_curvature_basis(feats.view(), &truth_spec)
        .expect("the planted kappa* geometry must be inside its own chart");
    let design = basis.design.to_dense();
    for j in 0..design.ncols() {
        let w = 1.0 / (1.0 + j as f64);
        for i in 0..n {
            y[i] += w * design[(i, j)];
        }
    }
    let mean = y.iter().sum::<f64>() / n as f64;
    let sd = (y.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n as f64).sqrt();
    assert!(sd > 0.0, "the planted kappa* = {kappa_star} signal collapsed");
    for i in 0..n {
        y[i] = (y[i] - mean) / sd + noise_sd * noise[i];
    }
    (feats, y, planted_ell)
}

fn fit_and_infer(formula: &str, feats: &Array2<f64>, y: &Array1<f64>) -> CurvatureInference {
    let n = y.len();
    let mut frame = Array2::<f64>::zeros((n, 3));
    for i in 0..n {
        frame[(i, 0)] = y[i];
        frame[(i, 1)] = feats[(i, 0)];
        frame[(i, 2)] = feats[(i, 1)];
    }
    let spec = termspec_for(formula, &frame);
    let weights = Array1::<f64>::ones(n);
    let offset = Array1::<f64>::zeros(n);
    let options = FitOptions::default();
    let kappa_options = SpatialLengthScaleOptimizationOptions {
        max_outer_iter: 8,
        rel_tol: 1e-4,
        pilot_subsample_threshold: 0,
        ..SpatialLengthScaleOptimizationOptions::default()
    };
    let fitted = fit_term_collectionwith_spatial_length_scale_optimization(
        frame.view(),
        y.clone(),
        weights.clone(),
        offset.clone(),
        &spec,
        LikelihoodSpec::gaussian_identity(),
        &options,
        &kappa_options,
    )
    .expect("constant-curvature fit with kappa optimization");
    curvature_inference_forspec(
        frame.view(),
        y.view(),
        weights.view(),
        offset.view(),
        &fitted.resolvedspec,
        0,
        LikelihoodSpec::gaussian_identity(),
        &options,
        0.95,
    )
    .expect("curvature inference")
}

fn main() {
    gam::init_parallelism();
    let n = N;
    let centers = CENTERS;
    let reps = REPS;
    let noise = NOISE_SD;
    let radius = CHART_RADIUS;
    let formula = format!("y ~ curv(x1, x2, centers={centers})");
    println!(
        "#2747 curvature x range grid -- n={n} centers={centers} noise={noise} \
         radius={radius} reps={reps}  formula: {formula}"
    );
    println!(
        "{:>7} {:>6} {:>4} {:>9} {:>22} {:>19} {:>11} {:>11} {:>22}",
        "kappa*", "range", "rep", "kappa_hat", "kappa support", "CI", "ell_plant", "ell_hat",
        "range support"
    );
    let mut cells = 0usize;
    let mut interior = 0usize;
    let mut sign_ok = 0usize;
    let mut abs_err_sum = 0.0_f64;
    for (ki, &kappa_star) in KAPPA_STARS.iter().enumerate() {
        for (mi, &multiplier) in RANGE_MULTIPLIERS.iter().enumerate() {
            for rep in 0..reps {
                let seed = 0x2747_0944_0000_0000
                    ^ ((ki as u64) << 40)
                    ^ ((mi as u64) << 24)
                    ^ ((rep as u64) << 8);
                let (feats, y, planted_ell) = dataset_on_m_kappa(
                    &formula, n, kappa_star, radius, noise, multiplier, seed,
                );
                let inf = fit_and_infer(&formula, &feats, &y);
                cells += 1;
                if !inf.ci.kappa_hat_support.is_railed() {
                    interior += 1;
                }
                if kappa_star == 0.0 || inf.kappa_hat * kappa_star > 0.0 {
                    sign_ok += 1;
                }
                abs_err_sum += (inf.kappa_hat - kappa_star).abs();
                println!(
                    "{:>7.2} {:>5.1}x {:>4} {:>+9.4} {:>22} [{:>+8.3},{:>+8.3}] {:>11.4} {:>11.4e} {:>22}",
                    kappa_star,
                    multiplier,
                    rep,
                    inf.kappa_hat,
                    inf.ci.kappa_hat_support.label(),
                    inf.ci.ci_lo,
                    inf.ci.ci_hi,
                    planted_ell,
                    inf.length_scale_hat,
                    inf.length_scale_support.label(),
                );
            }
        }
    }
    println!(
        "\nsummary: interior kappa_hat {interior}/{cells}  sign correct {sign_ok}/{cells}  \
         mean |kappa_hat - kappa*| = {:.4}",
        abs_err_sum / cells as f64
    );
}
