//! #2680 probe — the per-row `(y, h, L, U, z)` geometry the issue thread names
//! as its one missing measurement, plus the discriminator the thread has not
//! run: the fit's OWN score (`block_states[0].eta`, written by
//! `calibrate_transformation_scores` from `row_quantities`) against the score
//! the predict path reconstructs.
//!
//! The thread has established, and this probe does NOT re-derive: the moment
//! gate is faithful, the PIT kernel computes the correct finite-support ratio,
//! and no single location shift reconciles `u_mean = 0.857` with `51/400`
//! saturated rows.
//!
//! Prints only; asserts nothing about the defect. Reads the numpy fixture cells
//! from `/tmp/p2680/ctn_n{n}_s{seed}.csv` when they are present and silently
//! skips the ones that are not, so the probe is a no-op on a machine that has
//! not staged them.

use gam::smooth::build_term_collection_design;
use gam::terms::basis::{
    BasisOptions, Dense, KnotSource, create_basis, create_ispline_derivative_dense,
};
use gam::transformation_normal::{
    TRANSFORMATION_MONOTONICITY_EPS, TransformationNormalFitResult,
    transformation_normal_pit_score,
};
use gam::{
    FitConfig, FitRequest, FitResult, encode_recordswith_inferred_schema, fit_model,
    init_parallelism, materialize,
};
use ndarray::{Array1, Array2};

/// The clip the production score path applies (`TRANSFORMATION_SCORE_PIT_CLIP_EPS`).
const CLIP_EPS: f64 = 1.0e-12;

const CELLS: [(&str, usize, usize); 12] = [
    ("/tmp/p2680/ctn_n128_s1.csv", 128, 1),
    ("/tmp/p2680/ctn_n128_s2.csv", 128, 2),
    ("/tmp/p2680/ctn_n128_s3.csv", 128, 3),
    ("/tmp/p2680/ctn_n200_s1.csv", 200, 1),
    ("/tmp/p2680/ctn_n200_s2.csv", 200, 2),
    ("/tmp/p2680/ctn_n200_s3.csv", 200, 3),
    ("/tmp/p2680/ctn_n400_s1.csv", 400, 1),
    ("/tmp/p2680/ctn_n400_s2.csv", 400, 2),
    ("/tmp/p2680/ctn_n400_s3.csv", 400, 3),
    ("/tmp/p2680/ctn_n800_s1.csv", 800, 1),
    ("/tmp/p2680/ctn_n800_s2.csv", 800, 2),
    ("/tmp/p2680/ctn_n800_s3.csv", 800, 3),
];

struct Rows {
    headers: Vec<String>,
    values: Vec<Vec<f64>>,
}

fn read_csv(path: &str) -> Option<Rows> {
    let text = std::fs::read_to_string(path).ok()?;
    let mut lines = text.lines();
    let headers: Vec<String> = lines
        .next()
        .expect("csv header")
        .split(',')
        .map(|s| s.trim().to_string())
        .collect();
    let values: Vec<Vec<f64>> = lines
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            line.split(',')
                .map(|cell| cell.trim().parse::<f64>().expect("numeric cell"))
                .collect()
        })
        .collect();
    Some(Rows { headers, values })
}

struct Transform {
    h: Vec<f64>,
    h_prime: Vec<f64>,
    lower: Vec<f64>,
    upper: Vec<f64>,
}

/// Rebuild `(h, h', L, U)`. `square_shape = true` is the chart the PREDICT and
/// generated-regressor paths apply (`h = γ₀ + Σ I_k(y)·γ_k(x)²`); `false` is the
/// direct-α chart the FIT evaluates (`h = α₀ + Σ I_k(y)·α_k(x)`, `α_k ≥ 0` by
/// the Khatri-Rao monotonicity cone).
fn reconstruct(
    tn: &TransformationNormalFitResult,
    cov_rows: &Array2<f64>,
    y: &[f64],
    square_shape: bool,
) -> Transform {
    let family = &tn.family;
    let resp_knots = family.response_knots().clone();
    let resp_transform = family.response_transform();
    let degree = family.response_degree();
    let median = family.response_median();
    let eps = TRANSFORMATION_MONOTONICITY_EPS;

    let n = y.len();
    let p_cov = cov_rows.ncols();
    assert_eq!(cov_rows.nrows(), n);

    let beta = &tn.fit.blocks[0].beta;
    let p_shape = resp_transform.ncols();
    let p_resp = 1 + p_shape;
    assert_eq!(beta.len(), p_resp * p_cov, "beta / design shape mismatch");
    let gamma = beta
        .view()
        .into_shape_with_order((p_resp, p_cov))
        .expect("reshape beta");

    let y_arr = Array1::from_vec(y.to_vec());
    let (raw_val, _) = create_basis::<Dense>(
        y_arr.view(),
        KnotSource::Provided(resp_knots.view()),
        degree,
        BasisOptions::i_spline(),
    )
    .expect("I-spline value basis");
    let shape_val = raw_val.as_ref().dot(resp_transform);
    let raw_deriv = create_ispline_derivative_dense(y_arr.view(), &resp_knots, degree, 1)
        .expect("M-spline basis");
    let shape_deriv = raw_deriv.dot(resp_transform);

    let mut upper_shape = vec![0.0; p_shape];
    for c in 0..p_shape {
        upper_shape[c] = resp_transform.column(c).sum();
    }
    let lower_floor = eps * (resp_knots[0] - median);
    let upper_floor = eps * (resp_knots[resp_knots.len() - 1] - median);

    let mut out = Transform {
        h: vec![0.0; n],
        h_prime: vec![0.0; n],
        lower: vec![0.0; n],
        upper: vec![0.0; n],
    };
    for i in 0..n {
        let cov_row = cov_rows.row(i);
        let gamma0 = gamma.row(0).dot(&cov_row);
        let mut val = gamma0;
        let mut up = gamma0;
        let mut hp = 0.0;
        for r in 1..p_resp {
            let g = gamma.row(r).dot(&cov_row);
            let coefficient = if square_shape { g * g } else { g };
            val += shape_val[[i, r - 1]] * coefficient;
            up += upper_shape[r - 1] * coefficient;
            hp += shape_deriv[[i, r - 1]] * coefficient;
        }
        out.h[i] = val + eps * (y[i] - median);
        out.h_prime[i] = hp + eps;
        out.lower[i] = gamma0 + lower_floor;
        out.upper[i] = up + upper_floor;
    }
    out
}

fn quantile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let idx = ((sorted.len() - 1) as f64 * q).round() as usize;
    sorted[idx]
}

fn mean_sd(v: &[f64]) -> (f64, f64) {
    let n = v.len() as f64;
    let mean = v.iter().sum::<f64>() / n;
    let var = v.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / (n - 1.0);
    (mean, var.sqrt())
}

fn score_of(t: &Transform, n: usize) -> (Vec<f64>, usize) {
    let mut z = Vec::with_capacity(n);
    let mut saturated = 0usize;
    for i in 0..n {
        let zi = match transformation_normal_pit_score(t.h[i], t.lower[i], t.upper[i], CLIP_EPS) {
            Ok(value) => value,
            Err(err) => {
                eprintln!("#2680 probe: PIT refused at row {i}: {err}");
                f64::NAN
            }
        };
        if zi.abs() > 7.0 {
            saturated += 1;
        }
        z.push(zi);
    }
    (z, saturated)
}

fn run_cell(path: &str, n_expected: usize, seed: usize) {
    let Some(rows) = read_csv(path) else {
        eprintln!("#2680 probe: {path} absent, skipping");
        return;
    };
    let n = rows.values.len();
    assert_eq!(n, n_expected, "{path} row count");
    let headers = rows.headers.clone();
    let records: Vec<csv::StringRecord> = rows
        .values
        .iter()
        .map(|row| {
            csv::StringRecord::from(row.iter().map(|v| format!("{v:.17e}")).collect::<Vec<_>>())
        })
        .collect();
    let ds = encode_recordswith_inferred_schema(headers.clone(), records).expect("encode dataset");

    let cfg = FitConfig {
        transformation_normal: true,
        scale_dimensions: true,
        ..FitConfig::default()
    };
    let formula = "PGS ~ duchon(pc1, pc2, pc3, pc4, centers=6, order=0, power=3, length_scale=1)";
    let materialized = materialize(formula, &ds, &cfg).expect("materialize CTN request");
    let FitRequest::TransformationNormal(_) = materialized.request else {
        panic!("expected a TransformationNormal fit request");
    };
    let started = std::time::Instant::now();
    let result = match fit_model(materialized.request) {
        Ok(result) => result,
        Err(err) => {
            eprintln!("#2680 probe: n={n} seed={seed} fit REFUSED: {err}");
            return;
        }
    };
    let FitResult::TransformationNormal(tn) = result else {
        panic!("expected a TransformationNormal fit result");
    };
    eprintln!(
        "\n#2680 probe: ===== n={n} seed={seed} fitted in {:.1}s =====",
        started.elapsed().as_secs_f64()
    );

    let response_index = headers
        .iter()
        .position(|h| h == "PGS")
        .expect("PGS column present");
    let mut data = Array2::<f64>::zeros((n, headers.len()));
    for (i, row) in rows.values.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            data[[i, j]] = v;
        }
    }
    let design = build_term_collection_design(data.view(), &tn.covariate_spec_resolved)
        .expect("rebuild training covariate design");
    let cov_rows = design.design.to_dense();
    let y: Vec<f64> = (0..n).map(|i| data[[i, response_index]]).collect();

    let knots = tn.family.response_knots();
    let y_lo = knots[0];
    let y_hi = knots[knots.len() - 1];
    eprintln!(
        "#2680 probe: support y in [{y_lo:.5}, {y_hi:.5}] median={:.5} p_resp={} p_cov={} edf={:.3} \
         loglik={:.3}",
        tn.family.response_median(),
        1 + tn.family.response_transform().ncols(),
        cov_rows.ncols(),
        tn.fit.edf_total().unwrap_or(f64::NAN),
        tn.fit.log_likelihood,
    );

    // (A) The fit's OWN score. `calibrate_transformation_scores` overwrites
    // `block_states[0].eta` with `transformation_normal_pit_score` applied to
    // `row_quantities`, so this is the PIT the fitted model actually carries.
    let fit_eta = tn
        .fit
        .block_states
        .first()
        .map(|state| state.eta.to_vec())
        .unwrap_or_default();

    // (B) The direct-α reconstruction, and (C) the squared-chart reconstruction
    // the predict / generated-regressor paths use.
    let linear = reconstruct(&tn, &cov_rows, &y, false);
    let squared = reconstruct(&tn, &cov_rows, &y, true);
    let (z_linear, sat_linear) = score_of(&linear, n);
    let (z_squared, sat_squared) = score_of(&squared, n);

    if fit_eta.len() == n {
        let (m, s) = mean_sd(&fit_eta);
        let max_gap = (0..n)
            .map(|i| (fit_eta[i] - z_linear[i]).abs())
            .fold(0.0_f64, f64::max);
        eprintln!(
            "#2680 probe: FIT eta        mean={m:+.4} sd={s:.4}   max|eta - direct-alpha recon|={max_gap:.3e}"
        );
    }
    let (ml, sl) = mean_sd(&z_linear);
    let (ms, ss) = mean_sd(&z_squared);
    eprintln!("#2680 probe: direct-alpha  mean={ml:+.4} sd={sl:.4} saturated={sat_linear}/{n}");
    eprintln!("#2680 probe: SQUARED chart mean={ms:+.4} sd={ss:.4} saturated={sat_squared}/{n}");

    let mut lin_width: Vec<f64> = (0..n).map(|i| linear.upper[i] - linear.lower[i]).collect();
    let mut sq_width: Vec<f64> = (0..n).map(|i| squared.upper[i] - squared.lower[i]).collect();
    lin_width.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
    sq_width.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
    let mut lowers = linear.lower.clone();
    lowers.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
    eprintln!(
        "#2680 probe: support width  direct-alpha med={:.4} [{:.4},{:.4}]  squared med={:.4} [{:.4},{:.4}]  L med={:.4}",
        quantile(&lin_width, 0.5),
        lin_width[0],
        lin_width[n - 1],
        quantile(&sq_width, 0.5),
        sq_width[0],
        sq_width[n - 1],
        quantile(&lowers, 0.5),
    );
    eprintln!("#2680 probe: per-row head (i, y, L, h_alpha, U_alpha, h_sq, U_sq, z_alpha, z_sq):");
    for i in 0..n.min(12) {
        eprintln!(
            "#2680 probe:   {i:4} y={:+.4} L={:+.4} h_a={:+.4} U_a={:+.4} h_s={:+.4} U_s={:+.4} \
             z_a={:+.4} z_s={:+.4}",
            y[i],
            linear.lower[i],
            linear.h[i],
            linear.upper[i],
            squared.h[i],
            squared.upper[i],
            z_linear[i],
            z_squared[i]
        );
    }
}

#[test]
fn zz_probe_2680_ctn_pit_per_row_geometry() {
    init_parallelism();
    for (path, n, seed) in CELLS {
        run_cell(path, n, seed);
    }
}
