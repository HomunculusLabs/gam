//! #2750 probe: is the measure-jet 1-D over-smoothing a SMOOTHING-PARAMETER
//! defect, or a SPAN defect that λ is only reporting?
//!
//! The issue measured `edf/p = 0.17-0.19` at a converged, interior `ρ ≈ +5.5`
//! on a single-cycle sine, RMSE `0.545` against a budget of `0.25` — with the
//! passing seeds of the same generator sitting at `ρ = 1.8-3.7`, `edf/p =
//! 0.35-0.66`. Both fits converged, so REML genuinely chose that ρ.
//!
//! Since then #2761 restored `learn_length_scale`, so the shipped default now
//! lets REML move the representer range ℓ. That changes what this probe has to
//! separate, and it is now a THREE-way question at each fixed ℓ:
//!
//!   1. **span** — the least-squares projection of the noiseless truth onto the
//!      realized design's own column span. The floor no λ can beat.
//!   2. **λ-path** — the profiled Gaussian REML criterion and the held-out RMSE
//!      along `λ_k(δ) = λ̂_k · e^δ`, i.e. the fitted allocation scaled up and
//!      down as one. If RMSE improves sharply at `δ < 0` while the REML score
//!      gets worse, the criterion prefers an over-smoothed fit and the defect
//!      is in the criterion (or in the objects it is handed). If REML also
//!      improves at `δ < 0`, the outer search left a good optimum.
//!   3. **ℓ** — the same pair swept over pinned ranges, so the state at the
//!      auto ℓ (which is what every FROZEN-range consumer still gets: the BMS
//!      marginal/log-slope pair, and any explicit `length_scale=`) is visible
//!      next to the state the free search reaches.
//!
//! Usage: `cargo run --release --example probe_2750_lambda_path`

use csv::StringRecord;
use gam::gaussian_reml::gaussian_reml_point_eval_at_rho;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::{Array1, Array2};

/// SplitMix64 unit draws for the 3-D perf-parity replica used by `logdets`.
fn next_unit(state: &mut u64) -> f64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    (z >> 11) as f64 / (1u64 << 53) as f64
}

/// The sweep fixture's SplitMix64 finalizer, verbatim.
fn hashed_unit(index: u64) -> f64 {
    let mut z = index.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    (z >> 11) as f64 / (1u64 << 53) as f64
}

/// `measure_jet_formula_fit_robustness_sweep` seed 1: n = 200, freq = 1,
/// phase = 0, noise = 0.10, no jitter — the row this issue measured at
/// RMSE 0.5452.
const N: usize = 200;
const FREQ: f64 = 1.0;
const PHASE: f64 = 0.0;
const NOISE: f64 = 0.10;
const SEED: u64 = 1;

fn signal(x: f64) -> f64 {
    (std::f64::consts::TAU * FREQ * x + PHASE).sin()
}

fn xs() -> Vec<f64> {
    (0..N).map(|i| i as f64 / (N as f64 - 1.0)).collect()
}

fn dataset() -> gam::data::EncodedDataset {
    let headers = ["x", "y"]
        .into_iter()
        .map(str::to_string)
        .collect::<Vec<_>>();
    let rows = xs()
        .iter()
        .enumerate()
        .map(|(i, &x)| {
            let noise =
                2.0 * hashed_unit(
                    (i as u64)
                        .wrapping_mul(2_654_435_761)
                        .wrapping_add(SEED.wrapping_mul(0x9E37_79B9)),
                ) - 1.0;
            let y = signal(x) + NOISE * noise;
            StringRecord::from(vec![format!("{x:.17e}"), format!("{y:.17e}")])
        })
        .collect::<Vec<_>>();
    encode_recordswith_inferred_schema(headers, rows).expect("encode")
}

/// The fixture's held-out readout grid.
fn grid() -> Vec<f64> {
    (0..300).map(|i| 0.003 + 0.994 * i as f64 / 299.0).collect()
}

fn frame(points: &[f64]) -> Array2<f64> {
    let mut m = Array2::<f64>::zeros((points.len(), 2));
    for (i, &t) in points.iter().enumerate() {
        m[[i, 0]] = t;
    }
    m
}

fn dense_design(fit: &gam::StandardFitResult, points: &[f64]) -> Array2<f64> {
    let built = build_term_collection_design(frame(points).view(), &fit.resolvedspec)
        .expect("rebuild design");
    let op = &built.design;
    let (n, p) = (op.nrows(), op.ncols());
    let mut dense = Array2::<f64>::zeros((n, p));
    for j in 0..p {
        let mut e = Array1::<f64>::zeros(p);
        e[j] = 1.0;
        dense.column_mut(j).assign(&op.apply(&e));
    }
    dense
}

fn rmse(a: &[f64], b: &[f64]) -> f64 {
    let sse: f64 = a.iter().zip(b).map(|(u, v)| (u - v) * (u - v)).sum();
    (sse / a.len() as f64).sqrt()
}

/// RMSE of the least-squares projection residual of `y` onto span(cols of `x`),
/// by modified Gram-Schmidt with a relative rank floor. Returns the residual
/// RMSE and the realized numerical rank.
fn span_projection_rmse(x: &Array2<f64>, y: &[f64]) -> (f64, usize) {
    let p = x.ncols();
    let mut basis: Vec<Array1<f64>> = Vec::new();
    let scale = (0..p)
        .map(|j| x.column(j).dot(&x.column(j)).sqrt())
        .fold(0.0_f64, f64::max);
    let floor = 1.0e-10 * scale.max(1.0);
    for j in 0..p {
        let mut v = x.column(j).to_owned();
        for _ in 0..2 {
            for q in basis.iter() {
                let c = q.dot(&v);
                v.scaled_add(-c, q);
            }
        }
        let nrm = v.dot(&v).sqrt();
        if nrm > floor {
            v.mapv_inplace(|z| z / nrm);
            basis.push(v);
        }
    }
    let yv = Array1::from_vec(y.to_vec());
    let mut resid = yv.clone();
    for q in basis.iter() {
        let c = q.dot(&yv);
        resid.scaled_add(-c, q);
    }
    let zero = vec![0.0; resid.len()];
    (rmse(resid.as_slice().expect("contig"), &zero), basis.len())
}

/// One arm: fit the formula, then walk the fitted penalty allocation up and
/// down as a single scalar and report the criterion against the truth.
///
/// `walk = false` reports only the fitted row, which is what the ell sweep
/// needs: the SHIPPED profiled criterion at each pinned range, comparable
/// across the sweep because rescaling a penalty by `c` shifts rho by `-ln c`
/// and rho is optimized freely.
fn arm(body: &str, ds: &gam::data::EncodedDataset, walk: bool) {
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let fit = match fit_from_formula(&format!("y ~ {body}"), ds, &cfg) {
        Ok(FitResult::Standard(fit)) => fit,
        Ok(_) => {
            println!("[2750] {body}: non-standard fit variant");
            return;
        }
        Err(e) => {
            println!(
                "[2750] {body}: REFUSED: {}",
                e.to_string().chars().take(200).collect::<String>()
            );
            return;
        }
    };

    let train_x = xs();
    let gridx = grid();
    let truth_grid: Vec<f64> = gridx.iter().map(|&t| signal(t)).collect();
    let truth_train: Vec<f64> = train_x.iter().map(|&t| signal(t)).collect();
    let y: Vec<f64> = {
        let col = ds.column_map()["y"];
        ds.values.column(col).to_vec()
    };

    let x_grid = dense_design(&fit, &gridx);
    let x_train = dense_design(&fit, &train_x);
    let fitted_grid: Vec<f64> = x_grid.dot(&fit.fit.beta).to_vec();
    let fitted_rmse = rmse(&fitted_grid, &truth_grid);
    let (span_grid, rank_grid) = span_projection_rmse(&x_grid, &truth_grid);
    let (span_train, _) = span_projection_rmse(&x_train, &truth_train);

    let block = &fit.fit.blocks[0];
    let lambdas: Vec<f64> = block.lambdas.to_vec();
    let ell = fit
        .resolvedspec
        .smooth_terms
        .first()
        .map(|t| format!("{:?}", t.basis))
        .unwrap_or_default();
    let ell = ell
        .split("length_scale:")
        .nth(1)
        .and_then(|s| s.split(',').next())
        .unwrap_or("?")
        .trim()
        .to_string();

    let reml = fit.fit.reml_score().unwrap_or(f64::NAN);
    println!(
        "[2750] {body}\n       p={p_cols} rank={rank_grid} edf={edf:.3} n_lam={n_lam} \
         lambdas={lambdas:?} ell={ell}\n       reml={reml:.6}  fitted_rmse={fitted_rmse:.6}  \
         span_floor(grid)={span_grid:.6}  span_floor(train)={span_train:.6}",
        p_cols = x_grid.ncols(),
        edf = block.edf,
        n_lam = lambdas.len(),
    );
    if !walk {
        return;
    }

    // The fitted allocation as one dense matrix, then walked as a family.
    let built = build_term_collection_design(frame(&train_x).view(), &fit.resolvedspec)
        .expect("rebuild train design");
    let p = x_train.ncols();
    let mut s_total = Array2::<f64>::zeros((p, p));
    for (k, pen) in built.penalties.iter().enumerate() {
        let lam = lambdas.get(k).copied().unwrap_or(0.0);
        let r = pen.col_range.clone();
        let mut view = s_total.slice_mut(ndarray::s![r.clone(), r.clone()]);
        view.scaled_add(lam, &pen.local);
    }
    if built.penalties.len() != lambdas.len() {
        println!(
            "       WARNING: {} penalty blocks against {} lambdas; the walk uses the shared prefix",
            built.penalties.len(),
            lambdas.len()
        );
    }

    println!(
        "       {:>7} {:>16} {:>9} {:>10} {:>10}",
        "delta", "reml", "edf", "rmse", "sigma2"
    );
    let yv = Array1::from_vec(y.clone());
    for step in -14..=8 {
        let delta = step as f64;
        match gaussian_reml_point_eval_at_rho(
            x_train.view(),
            yv.view(),
            s_total.view(),
            None,
            None,
            delta,
        ) {
            Ok(ev) => {
                let pred: Vec<f64> = x_grid.dot(&ev.coefficients).to_vec();
                let e = rmse(&pred, &truth_grid);
                println!(
                    "       {delta:>7.1} {:>16.6} {:>9.3} {:>10.6} {:>10.6}",
                    ev.reml_score, ev.edf, e, ev.sigma2
                );
            }
            Err(err) => println!(
                "       {delta:>7.1}   REFUSED: {}",
                err.to_string().chars().take(120).collect::<String>()
            ),
        }
    }
}

/// Is the SHIPPED measure-jet penalty positive semidefinite along the ell axis?
///
/// The energy is assembled as a constructive factor form (`B^T B`, PSD by
/// construction, #2761/`03d5318d5`) and then transported by the term
/// collection's gauge. This prints the realized dense block's spectrum so a
/// violation is a measured number rather than a refusal message.
fn psd_scan(ds: &gam::data::EncodedDataset) {
    use gam::smooth::SmoothBasisSpec;
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let FitResult::Standard(seed_fit) =
        fit_from_formula("y ~ s(x, bs=\"mjs\", double_penalty=false)", ds, &cfg)
            .expect("seed mjs fit")
    else {
        panic!("standard fit");
    };
    let train_x = xs();
    println!(
        "[2750-psd] {:>10} {:>6} {:>5} {:>13} {:>13} {:>11} {:>11}",
        "ell", "arm", "p", "lambda_max", "lambda_min", "min/max", "asym"
    );
    for k in 0..30 {
        let ell = 0.006 * 1.3_f64.powi(k);
        for arm_name in ["cold", "frozen"] {
            let mut spec = seed_fit.resolvedspec.clone();
            match &mut spec.smooth_terms[0].basis {
                SmoothBasisSpec::MeasureJet { spec, .. } => {
                    spec.length_scale = ell;
                    spec.learn_length_scale = false;
                    if arm_name == "cold" {
                        spec.identifiability =
                            gam::basis::MeasureJetIdentifiability::CenterSumToZero;
                        spec.frozen_quadrature = None;
                    }
                }
                _ => panic!("mjs term"),
            }
            let built = match build_term_collection_design(frame(&train_x).view(), &spec) {
                Ok(b) => b,
                Err(e) => {
                    println!("[2750-psd] {ell:>10.5} {arm_name:>6}   build refused: {e}");
                    continue;
                }
            };
            let Some(pen) = built.penalties.first() else {
                println!("[2750-psd] {ell:>10.5} {arm_name:>6}   no penalty block");
                continue;
            };
            let m = &pen.local;
            let asym = m
                .indexed_iter()
                .map(|((i, j), v)| (v - m[[j, i]]).abs())
                .fold(0.0_f64, f64::max);
            let sym = (m + &m.t()).mapv(|v| 0.5 * v);
            let (vals, _) =
                gam::linalg::faer_ndarray::strict_symmetric_eigh(&sym, faer::Side::Lower)
                    .expect("eigh");
            let lo = vals.iter().copied().fold(f64::INFINITY, f64::min);
            let hi = vals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            println!(
                "[2750-psd] {ell:>10.5} {arm_name:>6} {:>5} {hi:>13.5e} {lo:>13.5e} {:>11.3e} {asym:>11.3e}",
                m.nrows(),
                lo / hi.abs().max(1e-300)
            );
        }
    }
}

/// The two ell profiles, side by side, on ONE penalty.
///
/// * `cold` re-realizes the whole basis at each pinned range — centers, the
///   rank-revealed identifiability section, the quadrature. That is what a user
///   gets from `length_scale=`.
/// * `frozen` is what the OUTER SEARCH walks: `build_measure_jet_basis_psi_
///   derivatives` runs on the FROZEN spec ("the driver runs post-freeze, so
///   per-trial rebuilds move only the dials"), so the section, the centers, the
///   masses and the band are pinned at the cold realization at the SEED range
///   and only ell moves.
///
/// `double_penalty=false` throughout so the criterion is the shipped
/// single-penalty closed form at every node and the two columns are the same
/// object evaluated on two design families.
fn ell_profiles(ds: &gam::data::EncodedDataset) {
    use gam::smooth::SmoothBasisSpec;
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let seed_formula = "y ~ s(x, bs=\"mjs\", double_penalty=false)";
    let FitResult::Standard(seed_fit) =
        fit_from_formula(seed_formula, ds, &cfg).expect("seed mjs fit")
    else {
        panic!("standard fit");
    };
    let seed_ell = match &seed_fit.resolvedspec.smooth_terms[0].basis {
        SmoothBasisSpec::MeasureJet { spec, .. } => spec.length_scale,
        _ => panic!("mjs term"),
    };
    println!(
        "[2750-profiles] seed fit: ell_hat={seed_ell:.6} reml={:.6} edf={:.3}",
        seed_fit.fit.reml_score().unwrap_or(f64::NAN),
        seed_fit.fit.blocks[0].edf
    );
    println!(
        "                {:>10} | {:>4} {:>14} {:>8} {:>9} | {:>4} {:>14} {:>8} {:>9}",
        "ell", "p", "reml", "edf", "rmse", "p", "reml", "edf", "rmse"
    );
    println!(
        "                {:>10} | {:^38} | {:^38}",
        "", "COLD (re-realized)", "FROZEN (what the search walks)"
    );

    let train_x = xs();
    let gridx = grid();
    let truth_grid: Vec<f64> = gridx.iter().map(|&t| signal(t)).collect();
    let y: Array1<f64> = {
        let col = ds.column_map()["y"];
        Array1::from_vec(ds.values.column(col).to_vec())
    };

    for k in 0..30 {
        let ell = 0.006 * 1.3_f64.powi(k);
        let cold = {
            // A cold spec at this range: same formula, ell pinned, nothing
            // frozen. `learn_length_scale=false` follows from an explicit
            // `length_scale=`, so this is one basis realization, not a search.
            let mut spec = seed_fit.resolvedspec.clone();
            match &mut spec.smooth_terms[0].basis {
                SmoothBasisSpec::MeasureJet { spec, .. } => {
                    spec.length_scale = ell;
                    spec.learn_length_scale = false;
                    spec.identifiability = gam::basis::MeasureJetIdentifiability::CenterSumToZero;
                    spec.frozen_quadrature = None;
                }
                _ => panic!("mjs term"),
            }
            profile_one(&spec, &train_x, &gridx, y.view(), &truth_grid)
        };
        let frozen = {
            let mut spec = seed_fit.resolvedspec.clone();
            match &mut spec.smooth_terms[0].basis {
                SmoothBasisSpec::MeasureJet { spec, .. } => {
                    spec.length_scale = ell;
                    spec.learn_length_scale = false;
                }
                _ => panic!("mjs term"),
            }
            profile_one(&spec, &train_x, &gridx, y.view(), &truth_grid)
        };
        let fmt = |r: &Option<(usize, f64, f64, f64)>| match r {
            Some((p, reml, edf, err)) => {
                format!("{p:>4} {reml:>14.6} {edf:>8.3} {err:>9.6}")
            }
            None => format!("{:>4} {:>14} {:>8} {:>9}", "-", "REFUSED", "-", "-"),
        };
        println!(
            "                {ell:>10.5} | {} | {}",
            fmt(&cold),
            fmt(&frozen)
        );
    }
}

/// `(p, profiled reml, edf, held-out rmse)` for one realized spec.
fn profile_one(
    spec: &gam::smooth::TermCollectionSpec,
    train_x: &[f64],
    gridx: &[f64],
    y: ndarray::ArrayView1<'_, f64>,
    truth_grid: &[f64],
) -> Option<(usize, f64, f64, f64)> {
    let built = match build_term_collection_design(frame(train_x).view(), spec) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("        build refused: {e}");
            return None;
        }
    };
    let op = &built.design;
    let (n, p) = (op.nrows(), op.ncols());
    let mut x_train = Array2::<f64>::zeros((n, p));
    for j in 0..p {
        let mut e = Array1::<f64>::zeros(p);
        e[j] = 1.0;
        x_train.column_mut(j).assign(&op.apply(&e));
    }
    if built.penalties.len() != 1 {
        eprintln!("        {} penalty blocks, not 1", built.penalties.len());
        return None;
    }
    let mut penalty = Array2::<f64>::zeros((p, p));
    let r = built.penalties[0].col_range.clone();
    penalty
        .slice_mut(ndarray::s![r.clone(), r.clone()])
        .assign(&built.penalties[0].local);
    let y2 = y.insert_axis(ndarray::Axis(1));
    let fit = match gam::gaussian_reml::gaussian_reml_multi_closed_form(
        x_train.view(),
        y2,
        penalty.view(),
        None,
        None,
    ) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("        reml refused: {e}");
            return None;
        }
    };
    let built_grid = build_term_collection_design(frame(gridx).view(), spec).ok()?;
    let opg = &built_grid.design;
    let mut x_grid = Array2::<f64>::zeros((opg.nrows(), p));
    for j in 0..p {
        let mut e = Array1::<f64>::zeros(p);
        e[j] = 1.0;
        x_grid.column_mut(j).assign(&opg.apply(&e));
    }
    let beta = fit.coefficients.column(0).to_owned();
    let pred: Vec<f64> = x_grid.dot(&beta).to_vec();
    Some((p, fit.reml_score, fit.edf, rmse(&pred, truth_grid)))
}

/// Sweep case 3 (n = 240, freq 1.5, noise 0.08), the row whose outer search
/// refuses. Fit it at a grid of PINNED ranges — which short-circuits the psi
/// search — and separately with the range free, so a refusal can be attributed
/// to the range region or to the search.
fn case3() {
    let n = 240usize;
    let freq = 1.5_f64;
    let phase = 0.7_f64;
    let noise = 0.08_f64;
    let seed = 3u64;
    let sig = |x: f64| (std::f64::consts::TAU * freq * x + phase).sin();
    let xs: Vec<f64> = (0..n).map(|i| i as f64 / (n as f64 - 1.0)).collect();
    let headers = ["x", "y"].into_iter().map(str::to_string).collect::<Vec<_>>();
    let rows = xs
        .iter()
        .enumerate()
        .map(|(i, &x)| {
            let e = 2.0
                * hashed_unit(
                    (i as u64)
                        .wrapping_mul(2_654_435_761)
                        .wrapping_add(seed.wrapping_mul(0x9E37_79B9)),
                )
                - 1.0;
            StringRecord::from(vec![
                format!("{x:.17e}"),
                format!("{:.17e}", sig(x) + noise * e),
            ])
        })
        .collect::<Vec<_>>();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode");
    let gridx: Vec<f64> = (0..300).map(|i| 0.003 + 0.994 * i as f64 / 299.0).collect();
    let truth: Vec<f64> = gridx.iter().map(|&t| sig(t)).collect();
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let mut bodies = vec!["s(x, bs=\"mjs\")".to_string()];
    for k in 0..18 {
        let ell = 0.004 * 1.4_f64.powi(k);
        bodies.push(format!("s(x, bs=\"mjs\", length_scale={ell})"));
    }
    for body in bodies {
        match fit_from_formula(&format!("y ~ {body}"), &ds, &cfg) {
            Ok(FitResult::Standard(fit)) => {
                let x_grid = dense_design(&fit, &gridx);
                let pred: Vec<f64> = x_grid.dot(&fit.fit.beta).to_vec();
                println!(
                    "[2750-case3] {body}: p={} edf={:.3} reml={:.4} rmse={:.6}",
                    x_grid.ncols(),
                    fit.fit.blocks[0].edf,
                    fit.fit.reml_score().unwrap_or(f64::NAN),
                    rmse(&pred, &truth)
                );
            }
            Ok(_) => println!("[2750-case3] {body}: non-standard"),
            Err(e) => println!(
                "[2750-case3] {body}: REFUSED {}",
                e.to_string().chars().take(150).collect::<String>()
            ),
        }
    }
}

/// Is the analytic `d log|S|+ / d ln ell` right at a LONG representer range?
///
/// The #2761 FD gate decomposes the ln-ell outer gradient into REML atoms and
/// reports `logdet_S` analytic `50.66` against FD `42.76` once the range screen
/// moves the seed out to `ell = 1.54` on the perf-parity fixture. This isolates
/// that atom from the outer engine entirely: the shipped basis at one range,
/// the shipped psi producer's `dS/dln ell`, `tr(S+ Sdot)` by hand, and a central
/// difference of `log|S|+` on the same objects.
fn logdet_s_scan() {
    use gam::basis::{
        MeasureJetBasisSpec, build_measure_jet_basis, build_measure_jet_basis_psi_derivatives,
    };
    // The perf-parity fixture: a 1-D curve in 3-D, n = 1500, 16 centers.
    let n = 1_500usize;
    let mut raw = Array2::<f64>::zeros((n, 3));
    let mut state = 1_039u64;
    for row in 0..n {
        let t = next_unit(&mut state);
        raw[[row, 0]] = t.clamp(1e-6, 1.0 - 1e-6);
        raw[[row, 1]] = (0.5 + 0.5 * (std::f64::consts::TAU * t).sin()).clamp(1e-6, 1.0 - 1e-6);
        raw[[row, 2]] = (t * t).clamp(1e-6, 1.0 - 1e-6);
    }
    let scale = gam::smooth::input_standardization::estimate_isotropic_scale(raw.view())
        .expect("scale");
    let mut x = raw;
    scale.standardize(&mut x);
    let base = MeasureJetBasisSpec {
        center_strategy: gam::basis::CenterStrategy::FarthestPoint { num_centers: 16 },
        double_penalty: false,
        learn_length_scale: true,
        ..MeasureJetBasisSpec::default()
    };
    println!(
        "[2750-logdetS] {:>9} {:>4} {:>4} {:>15} {:>15} {:>10} {:>12}",
        "ell", "p", "rank", "analytic", "fd", "gap", "cond(S)"
    );
    let logdet_positive = |m: &Array2<f64>, cut: f64| -> (f64, usize, f64) {
        let (vals, _) = gam::linalg::faer_ndarray::strict_symmetric_eigh(m, faer::Side::Lower)
            .expect("eigh");
        let hi = vals.iter().copied().fold(0.0_f64, f64::max);
        let mut sum = 0.0;
        let mut rank = 0usize;
        let mut lo = f64::INFINITY;
        for &v in vals.iter() {
            if v > cut * hi {
                sum += v.ln();
                rank += 1;
                lo = lo.min(v);
            }
        }
        (sum, rank, hi / lo)
    };
    // FREEZE the chart at each base range before differencing. `Z` is realized
    // from `K_cc(ell)` in a cold build, so a cold FD differences a coefficient
    // chart that MOVED, while the analytic jet holds the chart fixed by
    // contract ("rank/gauge realization happens once at fit time, then every
    // psi trial differentiates the same coefficient chart"). Comparing those
    // two measures the chart's motion, not the jet.
    let freeze = |basis: &gam::basis::BasisBuildResult, spec: &MeasureJetBasisSpec| {
        let gam::basis::BasisMetadata::MeasureJet {
            centers,
            eps_band,
            masses,
            support_means,
            penalty_normalization_scales,
            raw_penalty_normalization_scales,
            fused_penalty_normalization_scale,
            constraint_transform,
            sigma_coord,
            ..
        } = &basis.metadata
        else {
            panic!("measure-jet metadata")
        };
        let mut frozen = spec.clone();
        frozen.center_strategy = gam::basis::CenterStrategy::UserProvided(centers.clone());
        frozen.num_scales = eps_band.len();
        frozen.frozen_quadrature = Some(gam::basis::MeasureJetFrozenQuadrature {
            masses: masses.clone(),
            eps_band: eps_band.clone(),
            support_means: support_means.clone(),
            penalty_normalization_scales: penalty_normalization_scales.clone(),
            raw_penalty_normalization_scales: raw_penalty_normalization_scales.clone(),
            fused_penalty_normalization_scale: *fused_penalty_normalization_scale,
            sigma_coord: *sigma_coord,
        });
        frozen.identifiability = gam::basis::MeasureJetIdentifiability::FrozenTransform {
            transform: constraint_transform.clone().expect("constraint transform"),
        };
        frozen
    };
    for k in 0..14 {
        let ell = 0.25 * 1.3_f64.powi(k);
        let mut cold = base.clone();
        cold.length_scale = ell;
        let Ok(cold_basis) = build_measure_jet_basis(x.view(), &cold) else {
            println!("[2750-logdetS] {ell:>9.4}   cold build refused");
            continue;
        };
        let spec = freeze(&cold_basis, &cold);
        let Ok(basis) = build_measure_jet_basis(x.view(), &spec) else {
            println!("[2750-logdetS] {ell:>9.4}   build refused");
            continue;
        };
        if basis.active_penalties.len() != 1 {
            println!("[2750-logdetS] {ell:>9.4}   {} penalties", basis.active_penalties.len());
            continue;
        }
        let s = basis.active_penalties[0].matrix.clone();
        let jets = build_measure_jet_basis_psi_derivatives(x.view(), &spec).expect("jets");
        let sdot = jets.penalties_first[0][0].clone();
        // tr(S+ Sdot) over the numerically positive range, the same object the
        // outer engine differentiates.
        let (vals, vecs) =
            gam::linalg::faer_ndarray::strict_symmetric_eigh(&s, faer::Side::Lower).expect("eigh");
        let hi = vals.iter().copied().fold(0.0_f64, f64::max);
        let cut = f64::EPSILON * 64.0;
        let mut analytic = 0.0;
        for (i, &v) in vals.iter().enumerate() {
            if v > cut * hi {
                let u = vecs.column(i);
                analytic += u.dot(&sdot.dot(&u)) / v;
            }
        }
        let h = 1e-4_f64;
        let mut up = spec.clone();
        up.length_scale = ell * (h.exp());
        let mut down = spec.clone();
        down.length_scale = ell * ((-h).exp());
        let fd = match (
            build_measure_jet_basis(x.view(), &up),
            build_measure_jet_basis(x.view(), &down),
        ) {
            (Ok(a), Ok(b)) if a.active_penalties.len() == 1 && b.active_penalties.len() == 1 => {
                let (la, _, _) = logdet_positive(&a.active_penalties[0].matrix, cut);
                let (lb, _, _) = logdet_positive(&b.active_penalties[0].matrix, cut);
                (la - lb) / (2.0 * h)
            }
            _ => f64::NAN,
        };
        let (_, rank, cond) = logdet_positive(&s, cut);
        // Is the PRODUCER's dS/dln(ell) itself right? Compare it entrywise to a
        // central difference of the shipped penalty. If this is clean, the
        // logdet gap above is in the trace assembly; if it is not, the jet is.
        let jet_gap = match (
            build_measure_jet_basis(x.view(), &up),
            build_measure_jet_basis(x.view(), &down),
        ) {
            (Ok(a), Ok(b)) if a.active_penalties.len() == 1 && b.active_penalties.len() == 1 => {
                let fd_mat = (&a.active_penalties[0].matrix - &b.active_penalties[0].matrix)
                    .mapv(|v| v / (2.0 * h));
                let delta = &sdot - &fd_mat;
                let num = delta.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
                let den = sdot.iter().fold(0.0_f64, |m, v| m.max(v.abs())).max(1e-300);
                // Is the discrepancy PROPORTIONAL TO S? That is the shape a
                // missing/incorrect `-S * dln(c)/dpsi` normalization term has.
                let ss = s.iter().map(|v| v * v).sum::<f64>().max(1e-300);
                let alpha = delta.iter().zip(s.iter()).map(|(d, v)| d * v).sum::<f64>() / ss;
                let residual = delta
                    .iter()
                    .zip(s.iter())
                    .fold(0.0_f64, |m, (d, v)| m.max((d - alpha * v).abs()));
                eprintln!(
                    "                 alpha={alpha:+.6} residual_rel={:.3e}",
                    residual / den
                );
                num / den
            }
            _ => f64::NAN,
        };
        println!(
            "[2750-logdetS] {ell:>9.4} {:>4} {rank:>4} {analytic:>15.6} {fd:>15.6} {:>10.4} {cond:>12.3e} jet_rel={jet_gap:.3e}",
            s.nrows(),
            analytic - fd
        );
    }
}

fn main() {
    init_parallelism();
    let ds = dataset();
    println!(
        "[2750] measure_jet_formula_fit_robustness_sweep seed=1: n={N} freq={FREQ} noise={NOISE}"
    );
    println!("[2750] flat fit scores ~0.707; the fixture budget is 0.25");

    // The auto range is one median nearest-center spacing; with 50 centers on
    // [0, 1] that is ~1/49. Bracket it by two decades in each direction so the
    // frozen-range state and the free-search state are both on one table.
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(String::as_str).unwrap_or("path");

    if mode == "logdets" {
        logdet_s_scan();
        return;
    }

    if mode == "case3" {
        case3();
        return;
    }

    if mode == "psd" {
        psd_scan(&ds);
        return;
    }

    if mode == "frozen" {
        ell_profiles(&ds);
        return;
    }

    if mode == "ell" {
        // The SHIPPED profiled criterion along a pinned-range sweep, against
        // the range the free search actually lands on. `length_scale=` is in
        // the ORIGINAL units; the printed `ell` is the realized standardized
        // one (x has sd ~0.29 here, so the two differ by ~3.4x).
        arm("s(x, bs=\"mjs\")", &ds, false);
        for k in 0..26 {
            let ell = 0.0025 * 1.3_f64.powi(k);
            arm(&format!("s(x, bs=\"mjs\", length_scale={ell})"), &ds, false);
        }
        arm("s(x, bs=\"tp\")", &ds, false);
        return;
    }

    if mode == "ell1" {
        // Same sweep with ONE penalty everywhere, so every node's criterion is
        // the same object and the comparison carries no topology change.
        arm("s(x, bs=\"mjs\", double_penalty=false)", &ds, false);
        for k in 0..26 {
            let ell = 0.0025 * 1.3_f64.powi(k);
            arm(
                &format!("s(x, bs=\"mjs\", length_scale={ell}, double_penalty=false)"),
                &ds,
                false,
            );
        }
        arm("s(x, bs=\"tp\", double_penalty=false)", &ds, false);
        return;
    }

    let mut arms: Vec<String> = vec!["s(x, bs=\"mjs\")".to_string()];
    for ell in [0.005, 0.0102, 0.0204, 0.04, 0.08, 0.16, 0.32, 0.64] {
        arms.push(format!("s(x, bs=\"mjs\", length_scale={ell})"));
    }
    arms.push("s(x, bs=\"tp\")".to_string());
    for body in arms {
        arm(&body, &ds, true);
    }
}
