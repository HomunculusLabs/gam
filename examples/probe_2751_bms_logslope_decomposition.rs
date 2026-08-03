//! #2751 probe: decompose the BMS log-slope surface against its planted truth,
//! with the comparator bases fitted on byte-identical data.
//!
//! The issue thread narrowed the row to a confirmation print (which ρ belongs
//! to which block, and whether the pair is (energy, null-space)) plus two live
//! hypotheses for the truth-orthogonal component that is as large as the
//! recovered signal:
//!
//!   (1) leakage — the marginal block's `x2` structure reproduced in the
//!       log-slope surface;
//!   (2) variance — basis directions with no support in the truth fitted to
//!       noise.
//!
//! Those make opposite predictions under two interventions this probe runs:
//! deleting the marginal surface's `x2` term (kills (1), not (2)) and growing
//! `n` (shrinks (2) like `1/√n`, leaves (1) where it is).
//!
//! Every arm prints a number even when another arm refuses, so one refusal
//! cannot suppress the comparison (the failure mode that cost #2761 a probe).
//!
//! Usage: `cargo run --release --example probe_2751_bms_logslope_decomposition
//! -- <arm> [n]` where `<arm>` is one of `bases`, `nox2`, `scaling`.

use gam::families::bms::BernoulliMarginalSlopeFitResult;
use gam::terms::smooth::{SmoothBasisSpec, TermCollectionSpec, build_term_collection_design};
use gam::test_support::reference::pearson;
use gam::utils::splitmix64;
use gam::{FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula};
use ndarray::Array2;

struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    fn next_u64(&mut self) -> u64 {
        splitmix64(&mut self.state)
    }
    fn next_unit(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
    fn next_normal(&mut self) -> f64 {
        let u1 = self.next_unit().max(1.0e-300);
        let u2 = self.next_unit();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + statrs::function::erf::erf(x / std::f64::consts::SQRT_2))
}

fn beta_true(x1: f64) -> f64 {
    0.2 + 0.9 * x1
}

fn alpha_true(x1: f64, x2: f64, x2_weight: f64) -> f64 {
    -0.2 + 0.7 * (std::f64::consts::PI * x1).sin()
        + x2_weight * 0.3 * (std::f64::consts::PI * x2).cos()
}

/// The fixture's four columns plus `f`, the NOISELESS planted log-slope truth
/// carried as a response so the span arm can fit the identical basis
/// realization with no estimation noise in it.
fn build_dataset(x1: &[f64], x2: &[f64], y: &[f64], z: &[f64]) -> gam::data::EncodedDataset {
    let n = x1.len();
    let headers = vec![
        "x1".to_string(),
        "x2".to_string(),
        "y".to_string(),
        "z".to_string(),
        "f".to_string(),
    ];
    let records: Vec<csv::StringRecord> = (0..n)
        .map(|i| {
            csv::StringRecord::from(vec![
                format!("{:.17e}", x1[i]),
                format!("{:.17e}", x2[i]),
                format!("{:.17e}", y[i]),
                format!("{:.17e}", z[i]),
                format!("{:.17e}", beta_true(x1[i])),
            ])
        })
        .collect();
    encode_recordswith_inferred_schema(headers, records).expect("encode probe dataset")
}

/// The fixture's generative law, with a dial on the marginal surface's `x2`
/// content so the leakage hypothesis can be switched off at the source.
fn draw_with_covariates(
    n: usize,
    seed_off: u64,
    x2_weight: f64,
) -> (gam::data::EncodedDataset, Vec<f64>) {
    let mut rng = SplitMix64::new(0x315A_2026_0612_0001u64.wrapping_add(seed_off));
    let mut x1 = vec![0.0; n];
    let mut x2 = vec![0.0; n];
    let mut z = vec![0.0; n];
    for i in 0..n {
        x1[i] = rng.next_unit();
        x2[i] = rng.next_unit();
        z[i] = rng.next_normal();
    }
    let mut rng_y = SplitMix64::new(0x315A_2026_0612_0002u64.wrapping_add(seed_off));
    let mut y = vec![0.0; n];
    for i in 0..n {
        let eta = alpha_true(x1[i], x2[i], x2_weight) + beta_true(x1[i]) * z[i];
        let p = normal_cdf(eta).clamp(1e-9, 1.0 - 1e-9);
        y[i] = if rng_y.next_unit() < p { 1.0 } else { 0.0 };
    }
    let ds = build_dataset(&x1, &x2, &y, &z);
    (ds, x1)
}

fn draw(n: usize, seed_off: u64, x2_weight: f64) -> gam::data::EncodedDataset {
    draw_with_covariates(n, seed_off, x2_weight).0
}

fn fit_bms(
    body: &str,
    ds: &gam::data::EncodedDataset,
) -> Result<BernoulliMarginalSlopeFitResult, String> {
    let config = FitConfig {
        family: Some("bernoulli-marginal-slope".to_string()),
        logslope_formula: Some(body.to_string()),
        z_column: Some("z".to_string()),
        ..FitConfig::default()
    };
    match fit_from_formula(&format!("y ~ {body}"), ds, &config) {
        Ok(FitResult::BernoulliMarginalSlope(fit)) => Ok(fit),
        Ok(_) => Err("wrong family variant".to_string()),
        Err(e) => Err(format!("{e}")),
    }
}

fn grid_7x7() -> Vec<(f64, f64)> {
    let mut grid = Vec::with_capacity(49);
    for k1 in 0..7 {
        for k2 in 0..7 {
            grid.push((k1 as f64 / 6.0, k2 as f64 / 6.0));
        }
    }
    grid
}

fn surface(
    spec: &TermCollectionSpec,
    beta: &ndarray::Array1<f64>,
    baseline: f64,
    grid: &[(f64, f64)],
) -> Vec<f64> {
    let n = grid.len();
    let mut data = Array2::<f64>::zeros((n, 2));
    for (i, &(g1, g2)) in grid.iter().enumerate() {
        data[[i, 0]] = g1;
        data[[i, 1]] = g2;
    }
    let design = build_term_collection_design(data.view(), spec).expect("rebuild design");
    let dense = design.design.to_dense();
    assert_eq!(dense.ncols(), beta.len(), "design/beta width mismatch");
    (0..n)
        .map(|i| baseline + (0..dense.ncols()).map(|j| dense[[i, j]] * beta[j]).sum::<f64>())
        .collect()
}

/// Least-squares decomposition of `f` on the grid against the three-dimensional
/// frame {1, x1, x2}: this separates the planted direction (`x1`), the leakage
/// direction the truth is flat in (`x2`), and everything nonlinear (residual).
fn decompose(f: &[f64], grid: &[(f64, f64)]) -> (f64, f64, f64, f64) {
    let n = f.len() as f64;
    let mean = |v: &[f64]| v.iter().sum::<f64>() / n;
    let fm = mean(f);
    let x1: Vec<f64> = grid.iter().map(|g| g.0).collect();
    let x2: Vec<f64> = grid.iter().map(|g| g.1).collect();
    let (m1, m2) = (mean(&x1), mean(&x2));
    // The 7x7 product grid makes {x1 - m1, x2 - m2} exactly orthogonal, so the
    // two projections are independent one-dimensional least squares.
    let dot = |a: &[f64], am: f64, b: &[f64], bm: f64| {
        a.iter()
            .zip(b)
            .map(|(u, v)| (u - am) * (v - bm))
            .sum::<f64>()
    };
    let s11 = dot(&x1, m1, &x1, m1);
    let s22 = dot(&x2, m2, &x2, m2);
    let b1 = dot(&x1, m1, f, fm) / s11;
    let b2 = dot(&x2, m2, f, fm) / s22;
    let mut resid_sq = 0.0;
    for i in 0..f.len() {
        let r = (f[i] - fm) - b1 * (x1[i] - m1) - b2 * (x2[i] - m2);
        resid_sq += r * r;
    }
    // Component energies as RMS contributions on the grid.
    let e1 = b1 * (s11 / n).sqrt();
    let e2 = b2 * (s22 / n).sqrt();
    (b1, e1, e2, (resid_sq / n).sqrt())
}

fn report(tag: &str, fit: &BernoulliMarginalSlopeFitResult) {
    let grid = grid_7x7();
    let beta_hat = surface(
        &fit.logslopespec_resolved,
        &fit.fit.blocks[1].beta,
        fit.baseline_logslope,
        &grid,
    );
    let truth: Vec<f64> = grid.iter().map(|&(g1, _)| beta_true(g1)).collect();
    let corr = pearson(&beta_hat, &truth);
    let (slope_on_x1, e_x1, e_x2, e_resid) = decompose(&beta_hat, &grid);
    let (t_slope, t_x1, t_x2, t_resid) = decompose(&truth, &grid);
    println!(
        "[2751 {tag}] pearson={corr:.4} beta_hat: d/dx1={slope_on_x1:.4} (truth {t_slope:.4}) \
         rms[x1]={e_x1:.5} rms[x2]={e_x2:.5} rms[nonlinear]={e_resid:.5} \
         | truth rms[x1]={t_x1:.5} rms[x2]={t_x2:.5} rms[nl]={t_resid:.5} baseline={:.5}",
        fit.baseline_logslope
    );
    let rho = &fit.fit.log_lambdas;
    println!(
        "[2751 {tag}] rho={:?} marginal_penalties={} logslope_penalties={} \
         edf=[{:.3}, {:.3}] blocks={}",
        rho.iter().map(|r| (r * 1e4).round() / 1e4).collect::<Vec<_>>(),
        fit.marginal_design.penalties.len(),
        fit.logslope_design.penalties.len(),
        fit.fit.blocks[0].edf,
        fit.fit.blocks[1].edf,
        fit.fit.blocks.len(),
    );
    for (b, block) in fit.fit.blocks.iter().enumerate() {
        println!(
            "[2751 {tag}] block{b} role={:?} p={} edf={:.4} lambdas={:?}",
            block.role,
            block.beta.len(),
            block.edf,
            block
                .lambdas
                .iter()
                .map(|l| (l * 1e6).round() / 1e6)
                .collect::<Vec<_>>()
        );
    }
    for (what, spec) in [
        ("marginal", &fit.marginalspec_resolved),
        ("logslope", &fit.logslopespec_resolved),
    ] {
        for term in &spec.smooth_terms {
            if let SmoothBasisSpec::MeasureJet { spec: mj, .. } = &term.basis {
                let band = mj
                    .frozen_quadrature
                    .as_ref()
                    .map(|q| q.eps_band.clone())
                    .unwrap_or_default();
                println!(
                    "[2751 {tag}] {what} mjs: learn_ell={} ell={:?} double_penalty={} band={:?}",
                    mj.learn_length_scale, mj.length_scale, mj.double_penalty, band
                );
            }
        }
    }
}

/// Modified Gram-Schmidt (re-orthogonalized) least squares: returns the
/// coefficient vector and the residual RMS. `p` is small (≈17) so this is both
/// exact enough and dependency-free.
fn least_squares(x: &Array2<f64>, y: &[f64]) -> (Vec<f64>, f64) {
    let (n, p) = x.dim();
    let mut q: Vec<Vec<f64>> = Vec::with_capacity(p);
    let mut r = vec![vec![0.0f64; p]; p];
    let mut keep: Vec<usize> = Vec::new();
    for j in 0..p {
        let mut v: Vec<f64> = (0..n).map(|i| x[[i, j]]).collect();
        for _pass in 0..2 {
            for (k, &col) in keep.iter().enumerate() {
                let d: f64 = (0..n).map(|i| q[k][i] * v[i]).sum();
                r[col][j] += d;
                for i in 0..n {
                    v[i] -= d * q[k][i];
                }
            }
        }
        let nrm = v.iter().map(|a| a * a).sum::<f64>().sqrt();
        let col_nrm = (0..n).map(|i| x[[i, j]] * x[[i, j]]).sum::<f64>().sqrt();
        if nrm > 1e-10 * col_nrm.max(1e-300) {
            r[j][j] = nrm;
            for a in v.iter_mut() {
                *a /= nrm;
            }
            q.push(v);
            keep.push(j);
        }
    }
    // Coefficients on the kept columns by back substitution on R.
    let mut qty = vec![0.0f64; keep.len()];
    for (k, _) in keep.iter().enumerate() {
        qty[k] = (0..n).map(|i| q[k][i] * y[i]).sum();
    }
    let mut coef = vec![0.0f64; p];
    for kk in (0..keep.len()).rev() {
        let col = keep[kk];
        let mut acc = qty[kk];
        for ll in (kk + 1)..keep.len() {
            acc -= r[col][keep[ll]] * coef[keep[ll]];
        }
        coef[col] = acc / r[col][col];
    }
    let mut sse = 0.0;
    for i in 0..n {
        let f: f64 = (0..p).map(|j| x[[i, j]] * coef[j]).sum();
        sse += (y[i] - f) * (y[i] - f);
    }
    (coef, (sse / n as f64).sqrt())
}

/// Symmetric positive-definite solve by Cholesky with a relative jitter
/// fallback — `p` is ~17, so a textbook factorization is exact enough and
/// keeps the probe dependency-free.
fn spd_solve(a: &Array2<f64>, b: &[f64]) -> Vec<f64> {
    let p = b.len();
    let scale = (0..p).map(|i| a[[i, i]].abs()).fold(0.0, f64::max).max(1.0);
    let mut jitter = 0.0;
    loop {
        let mut l = vec![vec![0.0f64; p]; p];
        let mut ok = true;
        for i in 0..p {
            for j in 0..=i {
                let mut s = a[[i, j]] + if i == j { jitter * scale } else { 0.0 };
                for k in 0..j {
                    s -= l[i][k] * l[j][k];
                }
                if i == j {
                    if s <= 0.0 {
                        ok = false;
                        break;
                    }
                    l[i][i] = s.sqrt();
                } else {
                    l[i][j] = s / l[j][j];
                }
            }
            if !ok {
                break;
            }
        }
        if !ok {
            jitter = if jitter == 0.0 { 1e-14 } else { jitter * 100.0 };
            assert!(jitter < 1.0, "penalized normal equations are not solvable");
            continue;
        }
        let mut y = vec![0.0f64; p];
        for i in 0..p {
            let mut s = b[i];
            for k in 0..i {
                s -= l[i][k] * y[k];
            }
            y[i] = s / l[i][i];
        }
        let mut x = vec![0.0f64; p];
        for i in (0..p).rev() {
            let mut s = y[i];
            for k in (i + 1)..p {
                s -= l[k][i] * x[k];
            }
            x[i] = s / l[i][i];
        }
        return x;
    }
}

/// Ridge-limit least squares against ONE of the design's own penalties: as
/// `lambda -> inf` the solution is the least-squares fit restricted to that
/// penalty's null space. Applied to the measure-jet Primary this reads out
/// exactly the subspace the energy leaves free — the object the end-to-end fit
/// collapses onto when REML puts the energy at `lambda = 85`.
fn penalized_fit(
    x: &Array2<f64>,
    y: &[f64],
    penalty: Option<(&Array2<f64>, std::ops::Range<usize>)>,
    lambda: f64,
) -> Vec<f64> {
    let (n, p) = x.dim();
    let mut xtx = Array2::<f64>::zeros((p, p));
    for i in 0..n {
        for j in 0..p {
            let xij = x[[i, j]];
            if xij == 0.0 {
                continue;
            }
            for k in j..p {
                xtx[[j, k]] += xij * x[[i, k]];
            }
        }
    }
    for j in 0..p {
        for k in 0..j {
            xtx[[j, k]] = xtx[[k, j]];
        }
    }
    if let Some((s, range)) = penalty {
        // `x` carries an appended intercept in column 0, so the design column
        // `c` of the penalty's range sits at `c + 1`.
        for (a, ca) in range.clone().enumerate() {
            for (b, cb) in range.clone().enumerate() {
                xtx[[ca + 1, cb + 1]] += lambda * s[[a, b]];
            }
        }
    }
    let xty: Vec<f64> = (0..p)
        .map(|j| (0..n).map(|i| x[[i, j]] * y[i]).sum())
        .collect();
    spd_solve(&xtx, &xty)
}

/// Span floor of a surface basis against the planted affine truth, with no fit
/// and no penalty in it: least-squares projection of the NOISELESS
/// `beta_true(x1)` onto the fit-time design's column span (intercept
/// appended), then the same coefficients replayed on the scoring grid through
/// `build_term_collection_design`. Separates three failure modes that the
/// end-to-end Pearson cannot:
///   * the span cannot represent the plane  -> `train_floor` is large;
///   * the span can, but the grid rebuild disagrees with the fit-time design
///     -> `train_floor` tiny and `grid_floor` large;
///   * both tiny -> the basis is exonerated and the defect is the penalty or
///     the estimator.
fn span_arm(n: usize, bodies: &[&str]) {
    let (ds, x1) = draw_with_covariates(n, 0, 1.0);
    let grid = grid_7x7();
    for body in bodies {
        // Response is the planted log-slope truth itself, noiseless, so the
        // Gaussian fit's resolved spec is the same basis realization the BMS
        // log-slope block gets on this data.
        let fitted = match fit_from_formula(&format!("f ~ {body}"), &ds, &FitConfig::default()) {
            Ok(FitResult::Standard(fit)) => fit,
            Ok(_) => {
                println!("[2751 span {body}] non-standard fit variant");
                continue;
            }
            Err(e) => {
                println!("[2751 span {body}] REFUSED: {e}");
                continue;
            }
        };
        let dense = fitted.design.design.to_dense();
        let (rows, cols) = dense.dim();
        let mut x = Array2::<f64>::ones((rows, cols + 1));
        x.slice_mut(ndarray::s![.., 1..]).assign(&dense);
        assert_eq!(rows, x1.len(), "design rows must match the drawn sample");
        let truth_rows: Vec<f64> = x1.iter().map(|&v| beta_true(v)).collect();
        let (coef, train_floor) = least_squares(&x, &truth_rows);
        // Replay the identical coefficients on the scoring grid.
        let mut gdata = Array2::<f64>::zeros((grid.len(), 2));
        for (i, &(g1, g2)) in grid.iter().enumerate() {
            gdata[[i, 0]] = g1;
            gdata[[i, 1]] = g2;
        }
        let gdesign = build_term_collection_design(gdata.view(), &fitted.resolvedspec)
            .expect("rebuild scoring design");
        let gdense = gdesign.design.to_dense();
        let on_grid: Vec<f64> = (0..grid.len())
            .map(|i| coef[0] + (0..cols).map(|j| gdense[[i, j]] * coef[j + 1]).sum::<f64>())
            .collect();
        let truth_grid: Vec<f64> = grid.iter().map(|&(g1, _)| beta_true(g1)).collect();
        let grid_rmse = (on_grid
            .iter()
            .zip(&truth_grid)
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>()
            / grid.len() as f64)
            .sqrt();
        let (b1, e1, e2, enl) = decompose(&on_grid, &grid);
        println!(
            "[2751 span {body}] p={cols} train_floor={train_floor:.6} grid_rmse={grid_rmse:.6} \
             d/dx1={b1:.4} rms[x1]={e1:.5} rms[x2]={e2:.5} rms[nl]={enl:.5} \
             fit_edf={:.3} fit_pearson_on_grid={:.4}",
            fitted.fit.blocks[0].edf,
            pearson(&on_grid, &truth_grid),
        );
        // The shipped penalized Gaussian fit of the same noiseless response:
        // if the span is fine and this is not, the penalty is the defect.
        let beta = &fitted.fit.blocks[0].beta;
        let pen_grid: Vec<f64> = (0..grid.len())
            .map(|i| (0..gdense.ncols()).map(|j| gdense[[i, j]] * beta[j]).sum())
            .collect();
        let mean_shift = truth_grid
            .iter()
            .zip(&pen_grid)
            .map(|(t, p)| t - p)
            .sum::<f64>()
            / grid.len() as f64;
        let shifted: Vec<f64> = pen_grid.iter().map(|v| v + mean_shift).collect();
        let (pb1, pe1, pe2, penl) = decompose(&shifted, &grid);
        println!(
            "[2751 span {body}] PENALIZED gaussian fit: d/dx1={pb1:.4} rms[x1]={pe1:.5} \
             rms[x2]={pe2:.5} rms[nl]={penl:.5} pearson={:.4} lambdas={:?}",
            pearson(&shifted, &truth_grid),
            fitted.fit.blocks[0]
                .lambdas
                .iter()
                .map(|l| (l * 1e6).round() / 1e6)
                .collect::<Vec<_>>()
        );
        // The ridge limit against each of the design's own penalties: at
        // lambda -> inf this is the least-squares fit restricted to that
        // penalty's null space, with no estimator and no smoothing search in
        // it. For the measure-jet Primary (the jet energy) that null space is
        // the affine head, which is where the end-to-end fit lands when REML
        // puts the energy at lambda = 85.
        for (pi, pen) in fitted.design.penalties.iter().enumerate() {
            for lambda in [1.0e2_f64, 1.0e6, 1.0e10] {
                let b = penalized_fit(
                    &x,
                    &truth_rows,
                    Some((&pen.local, pen.col_range.clone())),
                    lambda,
                );
                let g: Vec<f64> = (0..grid.len())
                    .map(|i| b[0] + (0..cols).map(|j| gdense[[i, j]] * b[j + 1]).sum::<f64>())
                    .collect();
                let (rb1, re1, re2, renl) = decompose(&g, &grid);
                println!(
                    "[2751 span {body}] ridge-limit penalty#{pi} range={:?} lambda={lambda:.0e} \
                     d/dx1={rb1:.4} rms[x1]={re1:.5} rms[x2]={re2:.5} rms[nl]={renl:.5} \
                     pearson={:.4}",
                    pen.col_range,
                    pearson(&g, &truth_grid)
                );
            }
        }
        for term in &fitted.resolvedspec.smooth_terms {
            if let SmoothBasisSpec::MeasureJet { spec: mj, .. } = &term.basis {
                let masses = mj
                    .frozen_quadrature
                    .as_ref()
                    .map(|q| q.masses.clone())
                    .expect("frozen quadrature");
                let centers = match &mj.center_strategy {
                    gam::terms::basis::CenterStrategy::UserProvided(c) => c.clone(),
                    other => {
                        println!("[2751 span {body}] centers not frozen: {other:?}");
                        continue;
                    }
                };
                let head = gam::terms::basis::measure_jet_affine_head_transform(
                    centers.view(),
                    masses.view(),
                );
                println!(
                    "[2751 span {body}] centers={:?} head_rank={} head_T={:?} ell={:?}",
                    centers.dim(),
                    head.ncols(),
                    head,
                    mj.length_scale
                );
            }
        }
    }
}

fn main() {
    gam::init_parallelism();
    let args: Vec<String> = std::env::args().collect();
    let arm = args.get(1).map(String::as_str).unwrap_or("bases");
    let n: usize = args
        .get(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1500usize);

    let bodies = [
        "mjs(x1, x2, centers=16, scales=3)",
        "matern(x1, x2, k=16)",
        "duchon(x1, x2, k=16)",
        "s(x1, k=8) + s(x2, k=8)",
    ];

    match arm {
        // Which bases recover the plane, on byte-identical data. If the
        // comparators recover it and mjs does not, the row is measuring the
        // basis; if none of them do, it is measuring the BMS estimator.
        "bases" => {
            let ds = draw(n, 0, 1.0);
            for body in bodies {
                match fit_bms(body, &ds) {
                    Ok(fit) => report(body, &fit),
                    Err(e) => println!("[2751 {body}] REFUSED: {e}"),
                }
            }
        }
        // Hypothesis (1): delete the marginal surface's x2 structure. If the
        // truth-orthogonal x2 energy in the log-slope surface follows it down,
        // the component is leakage from the marginal block.
        "nox2" => {
            for weight in [1.0_f64, 0.0] {
                let ds = draw(n, 0, weight);
                for body in bodies {
                    match fit_bms(body, &ds) {
                        Ok(fit) => report(&format!("{body} x2w={weight}"), &fit),
                        Err(e) => println!("[2751 {body} x2w={weight}] REFUSED: {e}"),
                    }
                }
            }
        }
        // Hypothesis (2): grow n. Estimation variance falls like 1/sqrt(n);
        // leakage and approximation bias do not.
        "scaling" => {
            for rows in [750usize, 1500, 3000, 6000] {
                let ds = draw(rows, 0, 1.0);
                for body in [bodies[0], bodies[1]] {
                    match fit_bms(body, &ds) {
                        Ok(fit) => report(&format!("{body} n={rows}"), &fit),
                        Err(e) => println!("[2751 {body} n={rows}] REFUSED: {e}"),
                    }
                }
            }
        }
        // No BMS, no penalty: what the basis's own column span can do with the
        // planted plane, and whether the scoring-grid rebuild agrees with it.
        "span" => span_arm(n, &bodies),
        other => panic!("unknown arm {other}"),
    }
}
