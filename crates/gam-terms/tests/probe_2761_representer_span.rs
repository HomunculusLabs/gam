//! TEMPORARY probe for #2761 — not an acceptance gate.
//!
//! #2761 reports measure-jet at 13.4x Matérn's held-out RMSE on a 1-D curve in
//! 3-D, at matched `p` and at `edf/p = 0.98`. A fit that spends its whole basis
//! and still misses by an order of magnitude is bounded by its APPROXIMATION
//! SPACE, not by its smoothing parameter. This probe measures that bound
//! directly: the least-squares projection residual of the NOISELESS truth onto
//! the realized design's column span (plus an intercept), which no choice of λ
//! can beat, as a function of the representer range ℓ.
//!
//! Matérn's kernel range is a REML-selected coordinate (`MaternLengthScale::Auto`
//! → the κ optimizer). Measure-jet freezes ℓ at
//! `MEASURE_JET_AUTO_LENGTH_SCALE_FACTOR × median nearest-center spacing`.

use gam_terms::basis::{
    CenterStrategy, MaternBasisSpec, MaternIdentifiability, MaternLengthScale, MaternNu,
    MeasureJetBasisSpec, MeasureJetIdentifiability, build_matern_basis, build_measure_jet_basis,
    measure_jet_quadrature_nodes, realized_measure_jet_length_scale, select_centers_by_strategy,
};
use ndarray::{Array1, Array2};

const N: usize = 1_500;
const CENTERS: usize = 16;

fn clamp_unit_open(x: f64) -> f64 {
    x.max(1.0e-6).min(1.0 - 1.0e-6)
}

fn latent_to_coords(t: f64) -> [f64; 3] {
    [
        clamp_unit_open(t),
        clamp_unit_open(0.5 + 0.5 * (2.0 * std::f64::consts::PI * t).sin()),
        clamp_unit_open(t * t),
    ]
}

fn truth(t: f64) -> f64 {
    (2.0 * std::f64::consts::PI * t).sin() + 0.5 * (4.0 * std::f64::consts::PI * t).cos()
}

/// Deterministic low-discrepancy latents (golden-ratio additive recurrence) —
/// no rng dependency, and reproducible across runs.
fn latents(n: usize) -> Vec<f64> {
    let phi = 0.618_033_988_749_894_9_f64;
    (0..n).map(|i| ((i as f64 + 0.5) * phi).fract()).collect()
}

fn data_matrix(ts: &[f64]) -> Array2<f64> {
    let mut x = Array2::<f64>::zeros((ts.len(), 3));
    for (r, &t) in ts.iter().enumerate() {
        let c = latent_to_coords(t);
        for k in 0..3 {
            x[[r, k]] = c[k];
        }
    }
    x
}

/// Residual RMSE of the least-squares projection of `y` onto span(intercept,
/// columns of `x`), by twice-reorthogonalized modified Gram–Schmidt.
fn span_floor(x: &Array2<f64>, y: &[f64]) -> (f64, usize) {
    let n = x.nrows();
    let mut basis: Vec<Array1<f64>> = Vec::new();
    let push = |v: Array1<f64>, basis: &mut Vec<Array1<f64>>| {
        let mut v = v;
        let raw = v.dot(&v).sqrt();
        for _ in 0..2 {
            for q in basis.iter() {
                let c = q.dot(&v);
                v.scaled_add(-c, q);
            }
        }
        let nrm = v.dot(&v).sqrt();
        if nrm > 1.0e-10 * raw.max(1.0) {
            v.mapv_inplace(|z| z / nrm);
            basis.push(v);
        }
    };
    push(Array1::<f64>::ones(n), &mut basis);
    for j in 0..x.ncols() {
        push(x.column(j).to_owned(), &mut basis);
    }
    let yv = Array1::from_vec(y.to_vec());
    let mut resid = yv.clone();
    for q in basis.iter() {
        let c = q.dot(&yv);
        resid.scaled_add(-c, q);
    }
    ((resid.dot(&resid) / n as f64).sqrt(), basis.len())
}

fn mjs_spec(length_scale: f64) -> MeasureJetBasisSpec {
    MeasureJetBasisSpec {
        center_strategy: CenterStrategy::FarthestPoint {
            num_centers: CENTERS,
        },
        length_scale,
        identifiability: MeasureJetIdentifiability::CenterSumToZero,
        ..MeasureJetBasisSpec::default()
    }
}

#[test]
fn probe_2761_span_floor_versus_representer_range() {
    let ts = latents(N);
    let data = data_matrix(&ts);
    let y: Vec<f64> = ts.iter().map(|&t| truth(t)).collect();
    let y_rms = (y.iter().map(|v| v * v).sum::<f64>() / y.len() as f64).sqrt();
    let y_mean = y.iter().sum::<f64>() / y.len() as f64;
    let y_sd = (y.iter().map(|v| (v - y_mean).powi(2)).sum::<f64>() / y.len() as f64).sqrt();
    println!("[probe-2761] truth rms={y_rms:.6} sd(flat-fit floor)={y_sd:.6}");

    // Geometry the auto sentinel realizes.
    let seeds = select_centers_by_strategy(
        data.view(),
        &CenterStrategy::FarthestPoint {
            num_centers: CENTERS,
        },
    )
    .expect("seeds");
    let (nodes, masses) = measure_jet_quadrature_nodes(data.view(), seeds.view()).expect("nodes");
    let auto_ell = realized_measure_jet_length_scale(nodes.view(), 0.0).expect("auto ell");
    // Center separation and fill distance.
    let mut nn = Vec::new();
    for i in 0..nodes.nrows() {
        let mut best = f64::INFINITY;
        for j in 0..nodes.nrows() {
            if i != j {
                let d: f64 = (0..3).map(|k| (nodes[[i, k]] - nodes[[j, k]]).powi(2)).sum();
                best = best.min(d);
            }
        }
        nn.push(best.sqrt());
    }
    nn.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut fill: f64 = 0.0;
    for r in 0..data.nrows() {
        let mut best = f64::INFINITY;
        for i in 0..nodes.nrows() {
            let d: f64 = (0..3).map(|k| (data[[r, k]] - nodes[[i, k]]).powi(2)).sum();
            best = best.min(d);
        }
        fill = fill.max(best.sqrt());
    }
    println!(
        "[probe-2761] auto_ell={auto_ell:.6} nn_min={:.6} nn_med={:.6} nn_max={:.6} fill={fill:.6} \
         mass_min={:.1} mass_max={:.1}",
        nn[0],
        nn[nn.len() / 2],
        nn[nn.len() - 1],
        masses.iter().cloned().fold(f64::INFINITY, f64::min),
        masses.iter().cloned().fold(0.0_f64, f64::max),
    );

    // The ambient-affine head alone. On this fixture x1 = 0.5 + 0.5·sin(2πt),
    // so the head already contains the sin(2πt) half of the truth exactly; what
    // the representer block has to supply is the 0.5·cos(4πt) half.
    let (head_floor, head_rank) = span_floor(&data, &y);
    println!("[probe-2761] ambient-affine head only: rank={head_rank} span_floor={head_floor:.6}");

    for factor in [
        0.25_f64, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0,
    ] {
        let ell = auto_ell * factor;
        match build_measure_jet_basis(data.view(), &mjs_spec(ell)) {
            Ok(built) => {
                let dense = built.design.to_dense();
                let (floor, rank) = span_floor(&dense, &y);
                println!(
                    "[probe-2761] mjs  factor={factor:<5} ell={ell:.6} p={} rank={rank} span_floor={floor:.6}",
                    dense.ncols()
                );
            }
            Err(e) => println!("[probe-2761] mjs  factor={factor:<5} ell={ell:.6} BUILD FAILED: {e}"),
        }
    }

    for ell in [0.02_f64, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2] {
        let spec = MaternBasisSpec {
            center_strategy: CenterStrategy::FarthestPoint {
                num_centers: CENTERS,
            },
            periodic: None,
            length_scale: MaternLengthScale::fixed(ell),
            nu: MaternNu::FiveHalves,
            include_intercept: false,
            double_penalty: true,
            identifiability: MaternIdentifiability::CenterSumToZero,
            aniso_log_scales: None,
        };
        match build_matern_basis(data.view(), &spec) {
            Ok(built) => {
                let dense = built.design.to_dense();
                let (floor, rank) = span_floor(&dense, &y);
                println!(
                    "[probe-2761] matern ell={ell:.4} p={} rank={rank} span_floor={floor:.6}",
                    dense.ncols()
                );
            }
            Err(e) => println!("[probe-2761] matern ell={ell:.4} BUILD FAILED: {e}"),
        }
    }
}
