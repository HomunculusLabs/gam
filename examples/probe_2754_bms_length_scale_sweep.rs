//! #2754 / #2761 probe: is the measure-jet BMS accuracy gap a TUNING miss on the
//! representer range ℓ, or a basis-CAPACITY limit?
//!
//! `measure_jet_bms_accuracy_parity_1041.rs` answers "capacity" and cites a
//! length-scale sweep (`zz_mjs_lengthscale_sweep_1041`) as decisive evidence.
//! **That test does not exist in the tree** (`grep` finds only the citation), so
//! the claim currently rests on a measurement nobody can re-take. This probe is
//! that measurement, rebuilt, on the parity fixture's own data law and its own
//! held-out score.
//!
//! Two independent readouts per arm, because they separate the two candidate
//! causes and disagree under exactly one of them:
//!
//!   * **span floor** — the least-squares projection residual of the NOISELESS
//!     truth onto the realized design's column span. No penalty, no estimator,
//!     no smoothing search: a bound no `λ` can beat, because `λ` shrinks inside
//!     a span and never moves one. If the span floor is flat in ℓ, the basis
//!     genuinely cannot do better and the gap is capacity.
//!   * **held-out RMSE** through the shipped `fit_from_formula` BMS path — the
//!     number the parity bar actually reads.
//!
//! Usage:
//!   cargo run --release --example probe_2754_bms_length_scale_sweep -- [arm]
//! where `arm` is `sweep` (default, the BMS marginal accuracy vs ℓ) or `span`
//! (the response-free span floor vs ℓ, Gaussian path, both surfaces' truths).
//!
//! # The open question this instrument left behind (#2754/#2761 follow-up)
//!
//! After the range resolver reached this branch and the screen's walk stop was
//! moved to the feasibility wall, the screen picks `ℓ = 3.105` here — a genuine
//! INTERIOR optimum of its criterion, not an edge: the walk scores `4.47708`,
//! that node does not improve, and the parabolic refinement then lands at
//! `3.105`. Held-out RMSE does not turn around there. It keeps falling out to
//! `ℓ = 68.5` (`0.04179 → 0.03788`), a factor of 22 further, with the block
//! still carrying `edf = 7.47` and not degenerate.
//!
//! `ℓ` decides WHICH span the representers occupy and `λ` cannot move a span,
//! so a criterion that stops 22× short of the held-out optimum is choosing the
//! model rather than tuning it. Three candidate causes, each separated by one
//! run of an arm this file already has the shape for:
//!
//! 1. **The Gaussian-REML-on-a-binary-response approximation.** The screen ranks
//!    spans by a Gaussian REML of `y ∈ {0,1}`; the fit is a probit
//!    marginal-slope. Re-run the same grid with the continuous response and ask
//!    whether the criterion's argmin then tracks the held-out argmin.
//! 2. **Term-alone screening vs the full collection.** The screen scores
//!    `[1 | X(ℓ)]` with the single jet-energy penalty and `double_penalty =
//!    false`; the shipped fit carries the null component and a second coupled
//!    block. Re-score the grid with the double penalty on.
//! 3. **The held-out functional.** This file scores the marginal PROBABILITY
//!    surface at `z = 0`; the criterion scores a fit to `y`. Score held-out
//!    log-likelihood instead.
//!
//! Not claimed: that the criterion is wrong. gam#2750 measured it tracking
//! held-out truth at every node on its own fixture, collapse past the diameter
//! included. Whatever this is, it is fixture-dependent, and the three arms above
//! are the way in.

use gam::families::bms::BernoulliMarginalSlopeFitResult;
use gam::smooth::build_term_collection_design;
use gam::terms::smooth::SmoothBasisSpec;
use gam::utils::splitmix64;
use gam::{FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula};
use gam::matrix::LinearOperator;
use ndarray::Array2;

const N_TRAIN: usize = 1_500;
const N_TEST: usize = 600;
const CENTERS: usize = 10;

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
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn next_normal(&mut self) -> f64 {
        let u1 = self.next_unit().max(1e-12);
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

fn alpha_true(x1: f64, x2: f64) -> f64 {
    -0.2 + 0.7 * (std::f64::consts::PI * x1).sin() + 0.3 * (std::f64::consts::PI * x2).cos()
}

/// The parity fixture's four columns plus `a`, the NOISELESS planted marginal
/// surface, so the span arm can fit the identical basis realization with no
/// estimation noise in it.
fn build_dataset(x1: &[f64], x2: &[f64], y: &[f64], z: &[f64]) -> gam::data::EncodedDataset {
    let n = x1.len();
    let headers = vec![
        "x1".to_string(),
        "x2".to_string(),
        "y".to_string(),
        "z".to_string(),
        "a".to_string(),
    ];
    let records: Vec<csv::StringRecord> = (0..n)
        .map(|i| {
            csv::StringRecord::from(vec![
                format!("{:.17e}", x1[i]),
                format!("{:.17e}", x2[i]),
                format!("{:.17e}", y[i]),
                format!("{:.17e}", z[i]),
                format!("{:.17e}", alpha_true(x1[i], x2[i])),
            ])
        })
        .collect();
    encode_recordswith_inferred_schema(headers, records).expect("encode parity dataset")
}

fn draw() -> (gam::data::EncodedDataset, Vec<(f64, f64)>) {
    let mut rng = SplitMix64::new(0x1041_2026_0613_0001);
    let mut x1 = vec![0.0; N_TRAIN];
    let mut x2 = vec![0.0; N_TRAIN];
    let mut z = vec![0.0; N_TRAIN];
    for i in 0..N_TRAIN {
        x1[i] = rng.next_unit();
        x2[i] = rng.next_unit();
        z[i] = rng.next_normal();
    }
    let mut rng_y = SplitMix64::new(0x1041_2026_0613_0002);
    let mut y = vec![0.0; N_TRAIN];
    for i in 0..N_TRAIN {
        let eta = alpha_true(x1[i], x2[i]) + beta_true(x1[i]) * z[i];
        let p = normal_cdf(eta).clamp(1e-9, 1.0 - 1e-9);
        y[i] = if rng_y.next_unit() < p { 1.0 } else { 0.0 };
    }
    let mut rng_g = SplitMix64::new(0x1041_2026_0613_0003);
    let grid: Vec<(f64, f64)> = (0..N_TEST)
        .map(|_| (rng_g.next_unit(), rng_g.next_unit()))
        .collect();
    (build_dataset(&x1, &x2, &y, &z), grid)
}

fn grid_matrix(grid: &[(f64, f64)]) -> Array2<f64> {
    let mut data = Array2::<f64>::zeros((grid.len(), 2));
    for (i, &(g1, g2)) in grid.iter().enumerate() {
        data[[i, 0]] = g1;
        data[[i, 1]] = g2;
    }
    data
}

/// The parity test's own score: held-out RMSE of the fitted marginal
/// probability `Phi(alpha_hat)` at `z = 0` against `Phi(alpha_true)`.
fn marginal_prob_rmse(fit: &BernoulliMarginalSlopeFitResult, grid: &[(f64, f64)]) -> f64 {
    let data = grid_matrix(grid);
    let design = build_term_collection_design(data.view(), &fit.marginalspec_resolved)
        .expect("rebuild marginal design");
    let yhat = design.design.apply(&fit.fit.blocks[0].beta);
    let mut sse = 0.0;
    for (i, &(g1, g2)) in grid.iter().enumerate() {
        let d = normal_cdf(fit.baseline_marginal + yhat[i]) - normal_cdf(alpha_true(g1, g2));
        sse += d * d;
    }
    (sse / grid.len() as f64).sqrt()
}

/// Realized measure-jet range of the (single) mjs term in a resolved spec.
fn realized_ell(spec: &gam::terms::smooth::TermCollectionSpec) -> Option<f64> {
    spec.smooth_terms.iter().find_map(|term| match &term.basis {
        SmoothBasisSpec::MeasureJet { spec: mj, .. } => Some(mj.length_scale),
        _ => None,
    })
}

/// The realized GEOMETRY of the (single) mjs term: center count, the frozen
/// band, and the centers' own coordinate extent. Two fit paths that hand the
/// basis the same rows must realize the same numbers here; if they do not, the
/// range they derive from that geometry cannot be compared.
fn realized_geometry(spec: &gam::terms::smooth::TermCollectionSpec) -> String {
    spec.smooth_terms
        .iter()
        .find_map(|term| match &term.basis {
            SmoothBasisSpec::MeasureJet { spec: mj, .. } => {
                let centers = match &mj.center_strategy {
                    gam::terms::basis::CenterStrategy::UserProvided(c) => Some(c.clone()),
                    _ => None,
                };
                let extent = centers.as_ref().map(|c| {
                    (0..c.ncols())
                        .map(|k| {
                            let col = c.column(k);
                            let lo = col.iter().cloned().fold(f64::INFINITY, f64::min);
                            let hi = col.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                            hi - lo
                        })
                        .collect::<Vec<_>>()
                });
                let band = mj
                    .frozen_quadrature
                    .as_ref()
                    .map(|q| q.eps_band.clone())
                    .unwrap_or_default();
                Some(format!(
                    "m={:?} extent={:?} band0={:?} learn_ell={}",
                    centers.as_ref().map(|c| c.dim()),
                    extent.map(|e| e.iter().map(|v| (v * 1e3).round() / 1e3).collect::<Vec<_>>()),
                    band.first().map(|v| (v * 1e4).round() / 1e4),
                    mj.learn_length_scale,
                ))
            }
            _ => None,
        })
        .unwrap_or_else(|| "no mjs term".to_string())
}

fn fit_bms(body: &str, ds: &gam::data::EncodedDataset) -> Result<BernoulliMarginalSlopeFitResult, String> {
    let config = FitConfig {
        family: Some("bernoulli-marginal-slope".to_string()),
        link: Some("probit".to_string()),
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

/// Modified Gram-Schmidt (re-orthogonalized) least squares; returns the residual
/// RMS. Widths here are tens of columns, so a textbook factorization is exact
/// enough and pulls in no dependency.
fn ls_residual_rms(x: &Array2<f64>, y: &[f64]) -> f64 {
    let (n, p) = x.dim();
    let mut q: Vec<Vec<f64>> = Vec::with_capacity(p);
    let mut resid: Vec<f64> = y.to_vec();
    for j in 0..p {
        let mut v: Vec<f64> = (0..n).map(|i| x[[i, j]]).collect();
        let col_nrm = v.iter().map(|a| a * a).sum::<f64>().sqrt();
        for _pass in 0..2 {
            for qk in q.iter() {
                let d: f64 = (0..n).map(|i| qk[i] * v[i]).sum();
                for i in 0..n {
                    v[i] -= d * qk[i];
                }
            }
        }
        let nrm = v.iter().map(|a| a * a).sum::<f64>().sqrt();
        if nrm > 1e-10 * col_nrm.max(1e-300) {
            for a in v.iter_mut() {
                *a /= nrm;
            }
            q.push(v);
        }
    }
    for qk in q.iter() {
        let d: f64 = (0..n).map(|i| qk[i] * resid[i]).sum();
        for i in 0..n {
            resid[i] -= d * qk[i];
        }
    }
    (resid.iter().map(|r| r * r).sum::<f64>() / n as f64).sqrt()
}

fn bodies_for(ell_multiples: &[f64], auto_ell: f64) -> Vec<(String, String)> {
    let mut out = vec![(
        "mjs auto".to_string(),
        format!("mjs(x1, x2, centers={CENTERS})"),
    )];
    for &mult in ell_multiples {
        let ell = mult * auto_ell;
        out.push((
            format!("mjs ell={mult:.2}x"),
            format!("mjs(x1, x2, centers={CENTERS}, length_scale={ell:.6})"),
        ));
    }
    out.push((
        "matern".to_string(),
        format!("matern(x1, x2, k={CENTERS})"),
    ));
    out.push((
        "duchon".to_string(),
        format!("duchon(x1, x2, k={CENTERS})"),
    ));
    out
}

/// Response-free capacity readout: least-squares projection residual of the
/// noiseless planted marginal surface onto each realized design's column span,
/// on the TRAINING rows and replayed on the held-out grid.
fn span_arm(auto_ell: f64, ell_multiples: &[f64], ds: &gam::data::EncodedDataset, grid: &[(f64, f64)]) {
    let truth_grid: Vec<f64> = grid.iter().map(|&(g1, g2)| alpha_true(g1, g2)).collect();
    for (tag, body) in bodies_for(ell_multiples, auto_ell) {
        // A Gaussian fit of the NOISELESS marginal truth realizes the identical
        // basis the BMS marginal block would get on this data.
        let fitted = match fit_from_formula(&format!("a ~ {body}"), ds, &FitConfig::default()) {
            Ok(FitResult::Standard(fit)) => fit,
            Ok(_) => {
                println!("[2754 span {tag}] non-standard fit variant");
                continue;
            }
            Err(e) => {
                println!("[2754 span {tag}] REFUSED: {e}");
                continue;
            }
        };
        let dense = fitted.design.design.to_dense();
        let (rows, cols) = dense.dim();
        let mut x = Array2::<f64>::ones((rows, cols + 1));
        x.slice_mut(ndarray::s![.., 1..]).assign(&dense);
        // The training-row truth in the same row order as the design.
        let a_idx = ds.column_map()["a"];
        let truth_rows: Vec<f64> = (0..rows).map(|i| ds.values[[i, a_idx]]).collect();
        let floor = ls_residual_rms(&x, &truth_rows);
        let gdesign = build_term_collection_design(grid_matrix(grid).view(), &fitted.resolvedspec)
            .expect("rebuild scoring design");
        let gdense = gdesign.design.to_dense();
        let mut gx = Array2::<f64>::ones((grid.len(), cols + 1));
        gx.slice_mut(ndarray::s![.., 1..]).assign(&gdense);
        let grid_floor = ls_residual_rms(&gx, &truth_grid);
        println!(
            "[2754 span {tag:16}] p={cols:3} ell={:?} train_span_floor={floor:.6} \
             grid_span_floor={grid_floor:.6} fit_edf={:.3}",
            realized_ell(&fitted.resolvedspec).map(|v| (v * 1e4).round() / 1e4),
            fitted.fit.blocks[0].edf,
        );
    }
}

fn main() {
    gam::init_parallelism();
    let args: Vec<String> = std::env::args().collect();
    let arm = args.get(1).map(String::as_str).unwrap_or("sweep");
    let (ds, grid) = draw();

    // Realize the auto range once so the sweep is stated in multiples of it.
    let auto_fit = fit_from_formula(
        &format!("a ~ mjs(x1, x2, centers={CENTERS})"),
        &ds,
        &FitConfig::default(),
    );
    let auto_ell = match auto_fit {
        Ok(FitResult::Standard(fit)) => realized_ell(&fit.resolvedspec).unwrap_or(f64::NAN),
        _ => f64::NAN,
    };
    // The Gaussian path LEARNS ℓ, so the realized value above is the REML one.
    // The auto SEED is what BMS freezes at; recover it by pinning the dial off.
    let seed_fit = fit_from_formula(
        &format!("a ~ mjs(x1, x2, centers={CENTERS}, learn_length_scale=false)"),
        &ds,
        &FitConfig::default(),
    );
    let seed_ell = match seed_fit {
        Ok(FitResult::Standard(fit)) => realized_ell(&fit.resolvedspec).unwrap_or(f64::NAN),
        _ => f64::NAN,
    };
    println!("[2754] auto-seed ell={seed_ell:.6}  gaussian-REML ell={auto_ell:.6}");
    // The same declaration through the BMS entry point, so the two paths'
    // realized geometry can be compared side by side rather than inferred.
    if let Ok(FitResult::Standard(fit)) = fit_from_formula(
        &format!("a ~ mjs(x1, x2, centers={CENTERS}, learn_length_scale=false)"),
        &ds,
        &FitConfig::default(),
    ) {
        println!(
            "[2754 geometry gaussian-seed] ell={:?} {}",
            realized_ell(&fit.resolvedspec),
            realized_geometry(&fit.resolvedspec)
        );
    }
    if let Ok(fit) = fit_bms(&format!("mjs(x1, x2, centers={CENTERS})"), &ds) {
        println!(
            "[2754 geometry bms-marginal ] ell={:?} {}",
            realized_ell(&fit.marginalspec_resolved),
            realized_geometry(&fit.marginalspec_resolved)
        );
        println!(
            "[2754 geometry bms-logslope ] ell={:?} {}",
            realized_ell(&fit.logslopespec_resolved),
            realized_geometry(&fit.logslopespec_resolved)
        );
    }

    let base = if seed_ell.is_finite() { seed_ell } else { auto_ell };
    let multiples = [0.25_f64, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0];

    match arm {
        "span" => span_arm(base, &multiples, &ds, &grid),
        _ => {
            for (tag, body) in bodies_for(&multiples, base) {
                match fit_bms(&body, &ds) {
                    Ok(fit) => {
                        let rmse = marginal_prob_rmse(&fit, &grid);
                        println!(
                            "[2754 sweep {tag:16}] rmse={rmse:.5} ell={:?} edf=[{:.3}, {:.3}] \
                             |beta|_max={:.4e}",
                            realized_ell(&fit.marginalspec_resolved)
                                .map(|v| (v * 1e4).round() / 1e4),
                            fit.fit.blocks[0].edf,
                            fit.fit.blocks[1].edf,
                            fit.fit.beta.iter().fold(0.0_f64, |a, b| a.max(b.abs())),
                        );
                    }
                    Err(e) => println!("[2754 sweep {tag:16}] REFUSED: {e}"),
                }
            }
        }
    }
}
