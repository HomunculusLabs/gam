//! #2691 — the chart-collapse boundary sweep.
//!
//! `tests_steering_e4.rs` recovers a planted circle at R² ≈ 1 with
//! `n=240, p=6`, noise-free. The #2691 reproducer collapses to a CONSTANT
//! `d_atom=1` coordinate (std 1.06e-14, one distinct value over 70 rows) at
//! `n=70, p=8` with isotropic noise — and certifies it. This module locates the
//! boundary between those two regimes in the pure-Rust harness (no model, no
//! wheel, no GPU) so any hypothesis about the cause has a seconds-scale
//! reproducer.
//!
//! The measured quantity is the FITTED CHART COORDINATE itself, never `fit_ev`:
//! the issue measures the collapsed arm's EV (0.0581) BELOW a partially
//! collapsed arm's (0.0883), so an EV-denominated diagnostic provably cannot
//! see this defect.

#![cfg(test)]

use super::tests::{deterministic_circle_noise, global_ev};
use super::tests_startup_validation_1782::{Topo, build_term};
use super::*;
use ndarray::{Array1, Array2, array};

/// A planted circle of radius `radius` in a generic 2-plane of `R^p`, plus
/// isotropic noise of RMS `sigma`, `n` rows. This is the #2691 reproducer's
/// generator expressed in the Rust harness.
pub(crate) struct PlantedCircle {
    pub(crate) z: Array2<f64>,
    pub(crate) theta: Array1<f64>,
}

pub(crate) fn planted_circle_cloud(n: usize, p: usize, radius: f64, sigma: f64) -> PlantedCircle {
    // Deterministic orthonormal 2-frame in R^p (Gram-Schmidt), so the planted
    // ring is an exact circle and the sweep needs no RNG.
    let mut u = Array1::<f64>::zeros(p);
    let mut v = Array1::<f64>::zeros(p);
    for j in 0..p {
        u[j] = ((j as f64 + 1.0) * 0.7).sin();
        v[j] = ((j as f64 + 1.0) * 0.7).cos();
    }
    let un = u.dot(&u).sqrt();
    u.mapv_inplace(|x| x / un);
    let uv = u.dot(&v);
    for j in 0..p {
        v[j] -= uv * u[j];
    }
    let vn = v.dot(&v).sqrt();
    v.mapv_inplace(|x| x / vn);

    let theta =
        Array1::<f64>::from_shape_fn(n, |i| std::f64::consts::TAU * (i as f64 + 0.5) / n as f64);
    let mut z = Array2::<f64>::zeros((n, p));
    for i in 0..n {
        let (c, s) = (theta[i].cos(), theta[i].sin());
        for j in 0..p {
            z[[i, j]] =
                radius * (c * u[j] + s * v[j]) + sigma * deterministic_circle_noise(i, j);
        }
    }
    PlantedCircle { z, theta }
}

/// Squared circular correlation between a fitted period-1 chart coordinate and
/// the known generating phase, maximised over both orientations — the #2691
/// "recovery R²". It cannot be low for a units or a sign reason.
pub(crate) fn circular_recovery_r2(coord: &Array1<f64>, theta: &Array1<f64>) -> f64 {
    let n = coord.len();
    assert_eq!(n, theta.len());
    let mut best = 0.0_f64;
    for orientation in [1.0_f64, -1.0_f64] {
        // Regress (cos θ, sin θ) on (cos 2πt, sin 2πt) — equivalently, the
        // squared modulus of the mean of exp(i(θ ∓ 2πt)) is the circular
        // correlation of the two phases up to an unknown offset φ.
        let (mut re, mut im) = (0.0_f64, 0.0_f64);
        for i in 0..n {
            let phase = theta[i] - orientation * std::f64::consts::TAU * coord[i];
            re += phase.cos();
            im += phase.sin();
        }
        re /= n as f64;
        im /= n as f64;
        best = best.max(re * re + im * im);
    }
    best
}

pub(crate) struct ChartOutcome {
    pub(crate) ev: f64,
    /// Raw f64 standard deviation of the coordinate — what the #2691 ledger
    /// reports. On a period-1 circle this is NOT the chart's dispersion: `0.0`
    /// and `1.0` are the SAME point, so a fully collapsed chart can read
    /// `std ≈ 0.5` here.
    pub(crate) coord_std: f64,
    /// Circular variance `1 − |mean exp(2πi t)|` — the chart's dispersion IN
    /// ITS OWN MANIFOLD. `0` ⇒ every row sits on one point of the circle.
    pub(crate) circular_variance: f64,
    pub(crate) distinct: usize,
    /// Distinct coordinate values AFTER wrapping to `[0, 1)` and rounding to
    /// 1e-9 — the number of genuinely distinct chart points.
    pub(crate) distinct_wrapped: usize,
    pub(crate) recovery_r2: f64,
    /// Final per-atom ARD log-precision on the chart axis (the von-Mises
    /// coordinate prior's precision, `atom.rs::ArdAxisPrior`).
    pub(crate) log_ard: f64,
    pub(crate) refusal: Option<String>,
}

/// Fit the #2691 call shape (`K=1`, `d_atom=1`, circle, softmax) in the Rust
/// harness and report the CHART's own health, never the reconstruction EV.
pub(crate) fn fit_and_measure_chart(cloud: &PlantedCircle, n_iter: usize) -> ChartOutcome {
    let z = &cloud.z;
    let (mut term, _disp) = build_term(z.view(), 1, Topo::Circle, AssignmentMode::softmax(1.0));
    let mut rho = SaeManifoldRho::new(
        1.0e-3_f64.ln(),
        1.0e-3_f64.ln(),
        vec![array![1.0e-3_f64.ln()]; 1],
    );
    if let Err(error) =
        term.run_joint_fit_arrow_schur(z.view(), &mut rho, None, n_iter, 1.0, 1.0e-6, 1.0e-6)
    {
        return ChartOutcome {
            ev: f64::NAN,
            coord_std: f64::NAN,
            circular_variance: f64::NAN,
            distinct: 0,
            distinct_wrapped: 0,
            recovery_r2: f64::NAN,
            log_ard: rho.log_ard[0][0],
            refusal: Some(format!("{error}")),
        };
    }
    let ev = term
        .try_fitted()
        .map(|fitted| global_ev(z.view(), fitted.view()))
        .unwrap_or(f64::NAN);
    let coords = term.assignment.coords[0].as_matrix();
    let coord: Array1<f64> = coords.column(0).to_owned();
    let n = coord.len() as f64;
    let mean = coord.iter().sum::<f64>() / n;
    let coord_std = (coord.iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>() / n).sqrt();
    // Circular dispersion in the chart's OWN manifold: `t` and `t + 1` are the
    // same point of a period-1 circle, so a raw std cannot decide whether the
    // chart is degenerate.
    let (mut re, mut im) = (0.0_f64, 0.0_f64);
    for &t in coord.iter() {
        let phase = std::f64::consts::TAU * t;
        re += phase.cos();
        im += phase.sin();
    }
    re /= n;
    im /= n;
    let circular_variance = 1.0 - (re * re + im * im).sqrt();
    let mut sorted: Vec<f64> = coord.iter().copied().collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    sorted.dedup();
    let distinct = sorted.len();
    let mut wrapped: Vec<i64> = coord
        .iter()
        .map(|&t| (t.rem_euclid(1.0) * 1.0e9).round() as i64 % 1_000_000_000)
        .collect();
    wrapped.sort_unstable();
    wrapped.dedup();
    let distinct_wrapped = wrapped.len();
    let recovery_r2 = circular_recovery_r2(&coord, &cloud.theta);
    ChartOutcome {
        ev,
        coord_std,
        circular_variance,
        distinct,
        distinct_wrapped,
        recovery_r2,
        log_ard: rho.log_ard[0][0],
        refusal: None,
    }
}

/// The 2-D (n × p) × noise boundary sweep. Diagnostic — it prints a table and
/// asserts nothing about the interior, because the interior is the thing under
/// investigation. Run with `--nocapture` to read the table.
#[test]
fn zz_2691_chart_collapse_boundary_sweep() {
    // The whole grid, fixed: 48 fits in 2.4 s. `radius` and `sigma` are the
    // #2691 reproducer's own generator constants (circle of radius 2.086,
    // isotropic noise RMS 0.352).
    let ns: Vec<usize> = vec![60, 70, 90, 120, 180, 240];
    let ps: Vec<usize> = vec![4, 6, 8, 12];
    let sigmas: Vec<f64> = vec![0.0, 0.352];
    let radius: f64 = 2.086;

    eprintln!(
        "[2691-sweep] n\tp\tsigma\tEV\tcoord_std\tcirc_var\tdistinct\tdistinct_wrapped\t\
         recovery_R2\tlog_ard\tsecs\trefusal"
    );
    let mut worst: Option<(usize, usize, f64, f64)> = None;
    for &n in &ns {
        for &p in &ps {
            for &sigma in &sigmas {
                let cloud = planted_circle_cloud(n, p, radius, sigma);
                let start = std::time::Instant::now();
                let outcome = fit_and_measure_chart(&cloud, 40);
                let secs = start.elapsed().as_secs_f64();
                match &outcome.refusal {
                    Some(message) => {
                        let first = message.lines().next().unwrap_or("").to_string();
                        eprintln!(
                            "[2691-sweep] {n}\t{p}\t{sigma:.3}\t-\t-\t-\t-\t-\t-\t{:.3}\t\
                             {secs:.1}\tREFUSED: {first}",
                            outcome.log_ard
                        );
                    }
                    None => eprintln!(
                        "[2691-sweep] {n}\t{p}\t{sigma:.3}\t{:.4}\t{:.3e}\t{:.3e}\t{}/{n}\t{}/{n}\t\
                         {:.4}\t{:.3}\t{secs:.1}\t-",
                        outcome.ev,
                        outcome.coord_std,
                        outcome.circular_variance,
                        outcome.distinct,
                        outcome.distinct_wrapped,
                        outcome.recovery_r2,
                        outcome.log_ard
                    ),
                }
                if outcome.refusal.is_none()
                    && worst.is_none_or(|(_, _, _, r2)| outcome.recovery_r2 < r2)
                {
                    worst = Some((n, p, sigma, outcome.recovery_r2));
                }
            }
        }
    }
    // The bar the sweep earns: at a FIXED, seeded ρ the inner joint fit
    // recovers the planted circle EVERYWHERE on this grid — measured
    // 0.9793..1.0000 across n ∈ [60, 240] × p ∈ [4, 12] × σ ∈ {0, 0.352}. There
    // is no n × p boundary between E4's regime and #2691's, so a future
    // regression that reintroduces one fails here.
    let (n, p, sigma, r2) = worst.expect("the sweep must fit at least one cell");
    assert!(
        r2 > 0.9,
        "#2691: the inner joint fit must recover the planted circle at a fixed ρ;          worst cell n={n} p={p} sigma={sigma} recovery R²={r2:.4}"
    );
}

/// A planted ordinal LINE in a generic direction of `R^p` plus isotropic noise
/// — the companion #2691 fixture for `atom_topology="euclidean"`, `d_atom=1`,
/// which the issue reports refusing at every chart dimension with
/// `dense exact-stationarity pseudoinverse failed certification`.
pub(crate) fn planted_line_cloud(n: usize, p: usize, span: f64, sigma: f64) -> PlantedCircle {
    let mut u = Array1::<f64>::zeros(p);
    for j in 0..p {
        u[j] = ((j as f64 + 1.0) * 0.7).sin();
    }
    let un = u.dot(&u).sqrt();
    u.mapv_inplace(|x| x / un);
    // `theta` here carries the planted ordinal position, not an angle.
    let theta = Array1::<f64>::from_shape_fn(n, |i| (i as f64 + 0.5) / n as f64 - 0.5);
    let mut z = Array2::<f64>::zeros((n, p));
    for i in 0..n {
        for j in 0..p {
            z[[i, j]] = span * theta[i] * u[j] + sigma * deterministic_circle_noise(i, j);
        }
    }
    PlantedCircle { z, theta }
}

/// The euclidean/ordinal companion sweep. Reports whether the fit refuses and
/// with what, plus the linear recovery R² of the fitted chart coordinate
/// against the planted ordinal position.
#[test]
fn zz_2691_euclidean_line_refusal_sweep() {
    // Every chart dimension the issue reports the euclidean arm refusing at.
    let ns: Vec<usize> = vec![70];
    let ps: Vec<usize> = vec![3, 4, 6, 8, 12, 16];
    let sigmas: Vec<f64> = vec![0.0, 0.352];

    eprintln!("[2691-line] n\tp\tsigma\tEV\tcoord_std\trecovery_R2\tsecs\trefusal");
    for &n in &ns {
        for &p in &ps {
            for &sigma in &sigmas {
                let cloud = planted_line_cloud(n, p, 4.0, sigma);
                let z = &cloud.z;
                let (mut term, _disp) =
                    build_term(z.view(), 1, Topo::Euclidean, AssignmentMode::softmax(1.0));
                let mut rho = SaeManifoldRho::new(
                    1.0e-3_f64.ln(),
                    1.0e-3_f64.ln(),
                    vec![array![1.0e-3_f64.ln()]; 1],
                );
                let start = std::time::Instant::now();
                let result = term.run_joint_fit_arrow_schur(
                    z.view(),
                    &mut rho,
                    None,
                    40,
                    1.0,
                    1.0e-6,
                    1.0e-6,
                );
                let secs = start.elapsed().as_secs_f64();
                match result {
                    Err(error) => {
                        let text = format!("{error}").replace('\n', " ");
                        eprintln!("[2691-line] {n}\t{p}\t{sigma:.3}\t-\t-\t-\t{secs:.1}\tREFUSED: {text}");
                    }
                    Ok(_) => {
                        let ev = term
                            .try_fitted()
                            .map(|fitted| global_ev(z.view(), fitted.view()))
                            .unwrap_or(f64::NAN);
                        let coords = term.assignment.coords[0].as_matrix();
                        let coord: Array1<f64> = coords.column(0).to_owned();
                        let nn = coord.len() as f64;
                        let cm = coord.iter().sum::<f64>() / nn;
                        let tm = cloud.theta.iter().sum::<f64>() / nn;
                        let (mut sxy, mut sxx, mut syy) = (0.0_f64, 0.0_f64, 0.0_f64);
                        for i in 0..coord.len() {
                            let a = coord[i] - cm;
                            let b = cloud.theta[i] - tm;
                            sxy += a * b;
                            sxx += a * a;
                            syy += b * b;
                        }
                        let r2 = if sxx > 0.0 && syy > 0.0 {
                            sxy * sxy / (sxx * syy)
                        } else {
                            0.0
                        };
                        let std = (sxx / nn).sqrt();
                        eprintln!(
                            "[2691-line] {n}\t{p}\t{sigma:.3}\t{ev:.4}\t{std:.3e}\t{r2:.4}\t{secs:.1}\t-"
                        );
                    }
                }
            }
        }
    }
}

/// The FULL production entry, the way `gamfit.sae_manifold_fit` reaches it:
/// minimal seed → fit seed → `run_sae_manifold_fit` with
/// `run_outer_rho_search: true`. This is the only structural difference from
/// [`fit_and_measure_chart`], which drives the inner joint fit alone at a FIXED
/// ρ — and the inner-only sweep recovers the planted circle at R² ≥ 0.98
/// everywhere on `n ∈ [60, 240] × p ∈ [4, 12] × σ ∈ {0, 0.352}`. So whatever
/// #2691 is, it lives in the outer ρ search or the production seed, not in the
/// joint Newton.
#[test]
fn zz_2691_outer_path_chart_collapse_sweep() {
    use crate::manifold::{
        SaeFitAssignmentKind, SaeFitConfig, SaeFitRequest, SaeFitSeedReport, SaeFitSeedRequest,
        SaeMinimalSeedReport, SaeMinimalSeedRequest, build_sae_fit_seed, build_sae_minimal_seed,
        run_sae_manifold_fit,
    };
    use gam_terms::analytic_penalties::AnalyticPenaltyRegistry;

    // ONE cell: the outer path costs tens of seconds a fit, and the point this
    // test carries is that ρ moves `log_ard` at all (measured +1.341 here, and
    // -11.729 one chart dimension away) — not a grid. The σ ≥ 1.5 cells that
    // rail it to the outer box are on the issue with their wall times; they are
    // 300-500 s apiece and do not belong in a routine suite.
    let ns: Vec<usize> = vec![70];
    let ps: Vec<usize> = vec![8];
    let sigmas: Vec<f64> = vec![0.352];
    let radius: f64 = 2.086;

    eprintln!(
        "[2691-outer] n\tp\tsigma\tR2rec\tcoord_std\tcirc_var\tdistinct_wrapped\trecovery_R2\t\
         log_ard\tsecs\trefusal"
    );
    for &n in &ns {
        for &p in &ps {
            for &sigma in &sigmas {
                let cloud = planted_circle_cloud(n, p, radius, sigma);
                let target = cloud.z.clone();
                let start = std::time::Instant::now();
                let minimal = build_sae_minimal_seed(SaeMinimalSeedRequest {
                    target: target.view(),
                    atom_basis: vec!["periodic".to_string()],
                    atom_dim: vec![1],
                    assignment_kind: SaeFitAssignmentKind::Softmax,
                    alpha: 1.0,
                    tau: 1.0,
                    threshold: 0.0,
                    top_k: None,
                    random_state: 20260731,
                    initial_logits: None,
                    initial_coords: None,
                })
                .expect("minimal seed");
                let SaeMinimalSeedReport {
                    geometry_plans,
                    basis_values,
                    basis_jacobian,
                    decoder_coefficients,
                    smooth_penalties,
                    initial_logits,
                    initial_coords,
                    refine_routing,
                } = minimal;
                let registry = AnalyticPenaltyRegistry::new();
                let seed = build_sae_fit_seed(SaeFitSeedRequest {
                    target: target.view(),
                    geometry_plans: &geometry_plans,
                    basis_values: basis_values.view(),
                    basis_jacobian: basis_jacobian.view(),
                    decoder_coefficients: decoder_coefficients.view(),
                    smooth_penalties: smooth_penalties.view(),
                    initial_logits: initial_logits.view(),
                    initial_coords: initial_coords.view(),
                    alpha: 1.0,
                    tau: 1.0,
                    learnable_alpha: false,
                    assignment_kind: SaeFitAssignmentKind::Softmax,
                    sparsity_strength: 1.0,
                    smoothness: 1.0,
                    max_iter: 40,
                    learning_rate: 1.0,
                    ridge_ext_coord: 1.0e-6,
                    ridge_beta: 1.0e-6,
                    top_k: None,
                    threshold: 0.0,
                    native_ard_enabled: true,
                    seed_refine_routing: refine_routing,
                    seed_refine_random_state: 20260731,
                    data_row_reseed: false,
                    fit_config: SaeFitConfig::default(),
                    temperature_schedule: None,
                    fisher_metric: None,
                    row_loss_weights: None,
                    registry: &registry,
                })
                .expect("fit seed");
                let SaeFitSeedReport {
                    base_term,
                    initial_rho,
                    isometry_pin_active,
                    metric_provenance,
                } = seed;
                let outcome = run_sae_manifold_fit(SaeFitRequest {
                    reconstruction_optimism_folds: None,
                    base_term,
                    target: target.clone(),
                    registry,
                    initial_rho,
                    max_iter: 40,
                    learning_rate: 1.0,
                    ridge_ext_coord: 1.0e-6,
                    ridge_beta: 1.0e-6,
                    alpha: 1.0,
                    isometry_pin_active,
                    metric_provenance,
                    promote_from_residual: false,
                    run_structure_search: false,
                    run_outer_rho_search: true,
                    structured_residual_passes: 0,
                    cancel: None,
                });
                let secs = start.elapsed().as_secs_f64();
                let report = match outcome
                    .map_err(|error| format!("{error}"))
                    .and_then(|o| o.manifold_or_error())
                {
                    Ok(report) => report,
                    Err(error) => {
                        let text = error.replace('\n', " ");
                        eprintln!(
                            "[2691-outer] {n}\t{p}\t{sigma:.3}\t-\t-\t-\t-\t-\t-\t{secs:.1}\t\
                             REFUSED: {text}"
                        );
                        continue;
                    }
                };
                let coords = report.term.assignment.coords[0].as_matrix();
                let coord: Array1<f64> = coords.column(0).to_owned();
                let nn = coord.len() as f64;
                let mean = coord.iter().sum::<f64>() / nn;
                let coord_std =
                    (coord.iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>() / nn).sqrt();
                let (mut re, mut im) = (0.0_f64, 0.0_f64);
                for &t in coord.iter() {
                    let phase = std::f64::consts::TAU * t;
                    re += phase.cos();
                    im += phase.sin();
                }
                re /= nn;
                im /= nn;
                let circular_variance = 1.0 - (re * re + im * im).sqrt();
                let mut wrapped: Vec<i64> = coord
                    .iter()
                    .map(|&t| (t.rem_euclid(1.0) * 1.0e9).round() as i64 % 1_000_000_000)
                    .collect();
                wrapped.sort_unstable();
                wrapped.dedup();
                let recovery_r2 = circular_recovery_r2(&coord, &cloud.theta);
                eprintln!(
                    "[2691-outer] {n}\t{p}\t{sigma:.3}\t{:.4}\t{coord_std:.3e}\t\
                     {circular_variance:.3e}\t{}/{n}\t{recovery_r2:.4}\t{:.3}\t{secs:.1}\t-",
                    report.reconstruction_r2,
                    wrapped.len(),
                    report.rho.log_ard[0][0]
                );
            }
        }
    }
}

/// TEST A CONSTANT BY WHETHER IT VARIES. The inner sweep above holds the ARD
/// log-precision fixed at `ln 1e-3` and never collapses. The von-Mises
/// coordinate prior (`atom.rs::ArdAxisPrior`) is `V(t) = (α/κ²)(1 − cos κt)`,
/// whose UNIQUE minimum is the chart origin `t = 0`; its precision `α` is an
/// outer coordinate the ρ search is free to raise. This ladder asks the only
/// question that matters: does raising `α` alone, at a fixture the inner solver
/// recovers perfectly, drive the chart to a constant?
#[test]
fn zz_2691_ard_precision_ladder_collapses_the_chart() {
    let n: usize = 70;
    let p: usize = 8;
    let cloud = planted_circle_cloud(n, p, 2.086, 0.352);
    let z = &cloud.z;
    let ladder: Vec<f64> = vec![
        1.0e-3_f64.ln(),
        1.0_f64.ln(),
        1.0e1_f64.ln(),
        1.0e2_f64.ln(),
        1.0e3_f64.ln(),
        1.0e4_f64.ln(),
        1.0e6_f64.ln(),
        1.0e9_f64.ln(),
    ];
    eprintln!("[2691-ard] log_ard\talpha\tEV\tcoord_std\tcirc_var\tdistinct_wrapped\trecovery_R2");
    let mut rungs: Vec<(f64, f64, usize)> = Vec::new();
    for &log_ard in &ladder {
        let (mut term, _disp) = build_term(z.view(), 1, Topo::Circle, AssignmentMode::softmax(1.0));
        let mut rho =
            SaeManifoldRho::new(1.0e-3_f64.ln(), 1.0e-3_f64.ln(), vec![array![log_ard]; 1]);
        let fit =
            term.run_joint_fit_arrow_schur(z.view(), &mut rho, None, 40, 1.0, 1.0e-6, 1.0e-6);
        if let Err(error) = fit {
            let text = format!("{error}").replace('\n', " ");
            eprintln!("[2691-ard] {log_ard:.3}\t{:.3e}\tREFUSED: {text}", log_ard.exp());
            continue;
        }
        let ev = term
            .try_fitted()
            .map(|fitted| global_ev(z.view(), fitted.view()))
            .unwrap_or(f64::NAN);
        let coords = term.assignment.coords[0].as_matrix();
        let coord: Array1<f64> = coords.column(0).to_owned();
        let nn = coord.len() as f64;
        let mean = coord.iter().sum::<f64>() / nn;
        let coord_std = (coord.iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>() / nn).sqrt();
        let (mut re, mut im) = (0.0_f64, 0.0_f64);
        for &t in coord.iter() {
            let phase = std::f64::consts::TAU * t;
            re += phase.cos();
            im += phase.sin();
        }
        re /= nn;
        im /= nn;
        let circular_variance = 1.0 - (re * re + im * im).sqrt();
        let mut wrapped: Vec<i64> = coord
            .iter()
            .map(|&t| (t.rem_euclid(1.0) * 1.0e9).round() as i64 % 1_000_000_000)
            .collect();
        wrapped.sort_unstable();
        wrapped.dedup();
        let recovery_r2 = circular_recovery_r2(&coord, &cloud.theta);
        eprintln!(
            "[2691-ard] {log_ard:.3}\t{:.3e}\t{ev:.4}\t{coord_std:.3e}\t{circular_variance:.3e}\t\
             {}/{n}\t{recovery_r2:.4}",
            log_ard.exp(),
            wrapped.len()
        );
        rungs.push((log_ard.exp(), circular_variance, wrapped.len()));
    }
    // The mechanism, pinned. Nothing but α moved between these two rungs.
    if let (Some(low), Some(high)) = (
        rungs.iter().find(|(alpha, _, _)| *alpha <= 1.0e-3),
        rungs.iter().find(|(alpha, _, _)| *alpha >= 1.0e9),
    ) {
        assert!(
            low.1 > 0.5,
            "#2691: at α=1e-3 the chart must still be spread over the circle              (circular variance {:.3e})",
            low.1
        );
        assert!(
            high.1 < 1.0e-6 && high.2 <= 1,
            "#2691: raising ONLY the von-Mises coordinate precision to α=1e9 must collapse the              chart to one point — this is the mechanism the guard exists for              (circular variance {:.3e}, {} resolved chart points)",
            high.1,
            high.2
        );
    }
}

/// #2691 REGRESSION — the production entry must REFUSE a chart that has
/// collapsed to a single point of its own manifold, instead of returning a
/// certified fit whose coordinate is a constant.
///
/// The fixture is the ARD ladder's terminal rung driven through the REAL entry
/// (`run_sae_manifold_fit`, the same function `gamfit.sae_manifold_fit` calls),
/// at a FIXED ρ so the collapse is placed by construction rather than waited
/// for: `α = 1e9` on the periodic chart axis. At the parent commit this call
/// returns `Ok` with `coord_std = 5.000e-1`, one distinct chart point over 70
/// rows, and a healthy-looking report — the exact #2691 `ideal` arm. Here it
/// must be a typed `SaeFitError::DegenerateChart`.
///
/// The bar is deliberately NOT denominated in explained variance: #2691
/// measured the fully collapsed arm's EV (0.0581) BELOW a partially collapsed
/// arm's (0.0883), so an EV gate passes every one of these states.
#[test]
fn zz_2691_collapsed_chart_is_refused_by_the_production_entry() {
    use crate::manifold::{
        SaeFitAssignmentKind, SaeFitConfig, SaeFitError, SaeFitRequest, SaeFitSeedReport,
        SaeFitSeedRequest, SaeMinimalSeedReport, SaeMinimalSeedRequest, build_sae_fit_seed,
        build_sae_minimal_seed, run_sae_manifold_fit,
    };
    use gam_terms::analytic_penalties::AnalyticPenaltyRegistry;

    let cloud = planted_circle_cloud(70, 8, 2.086, 0.352);
    let target = cloud.z.clone();
    let minimal = build_sae_minimal_seed(SaeMinimalSeedRequest {
        target: target.view(),
        atom_basis: vec!["periodic".to_string()],
        atom_dim: vec![1],
        assignment_kind: SaeFitAssignmentKind::Softmax,
        alpha: 1.0,
        tau: 1.0,
        threshold: 0.0,
        top_k: None,
        random_state: 20260731,
        initial_logits: None,
        initial_coords: None,
    })
    .expect("minimal seed");
    let SaeMinimalSeedReport {
        geometry_plans,
        basis_values,
        basis_jacobian,
        decoder_coefficients,
        smooth_penalties,
        initial_logits,
        initial_coords,
        refine_routing,
    } = minimal;
    let registry = AnalyticPenaltyRegistry::new();
    let seed = build_sae_fit_seed(SaeFitSeedRequest {
        target: target.view(),
        geometry_plans: &geometry_plans,
        basis_values: basis_values.view(),
        basis_jacobian: basis_jacobian.view(),
        decoder_coefficients: decoder_coefficients.view(),
        smooth_penalties: smooth_penalties.view(),
        initial_logits: initial_logits.view(),
        initial_coords: initial_coords.view(),
        alpha: 1.0,
        tau: 1.0,
        learnable_alpha: false,
        assignment_kind: SaeFitAssignmentKind::Softmax,
        sparsity_strength: 1.0,
        smoothness: 1.0,
        max_iter: 40,
        learning_rate: 1.0,
        ridge_ext_coord: 1.0e-6,
        ridge_beta: 1.0e-6,
        top_k: None,
        threshold: 0.0,
        native_ard_enabled: true,
        seed_refine_routing: refine_routing,
        seed_refine_random_state: 20260731,
        data_row_reseed: false,
        fit_config: SaeFitConfig::default(),
        temperature_schedule: None,
        fisher_metric: None,
        row_loss_weights: None,
        registry: &registry,
    })
    .expect("fit seed");
    let SaeFitSeedReport {
        base_term,
        mut initial_rho,
        isometry_pin_active,
        metric_provenance,
    } = seed;
    // The collapse lever, and the ONLY thing changed from a healthy call: the
    // origin-anchored von-Mises coordinate prior's precision on the chart axis.
    for axis in initial_rho.log_ard.iter_mut() {
        axis.fill(1.0e9_f64.ln());
    }

    let outcome = run_sae_manifold_fit(SaeFitRequest {
        reconstruction_optimism_folds: None,
        base_term,
        target,
        registry,
        initial_rho,
        max_iter: 40,
        learning_rate: 1.0,
        ridge_ext_coord: 1.0e-6,
        ridge_beta: 1.0e-6,
        alpha: 1.0,
        isometry_pin_active,
        metric_provenance,
        promote_from_residual: false,
        run_structure_search: false,
        // Fixed ρ: the collapse is PLACED, not waited for, so this reproducer
        // runs in seconds and cannot be a flaky search outcome.
        run_outer_rho_search: false,
        structured_residual_passes: 0,
        cancel: None,
    });

    match outcome {
        Err(SaeFitError::DegenerateChart {
            atoms,
            evidence,
            report,
        }) => {
            eprintln!("[2691-regression] refused as required: atoms={atoms:?} {evidence}");
            assert_eq!(
                atoms,
                vec![0],
                "the single K=1 atom must be the one named as chart-less"
            );
            assert!(
                report.axes.iter().all(|axis| axis.resolved_points <= 1),
                "the collapsed chart axis must resolve at most one chart point; got {:?}",
                report
                    .axes
                    .iter()
                    .map(|axis| axis.resolved_points)
                    .collect::<Vec<_>>()
            );
        }
        Err(other) => panic!(
            "#2691: a chart collapsed to one point must refuse as DegenerateChart, got: {other}"
        ),
        Ok(outcome) => {
            let report = outcome
                .manifold_or_error()
                .expect("outcome carried a manifold");
            let coords = report.term.assignment.coords[0].as_matrix();
            let coord: Array1<f64> = coords.column(0).to_owned();
            let chart = report.term.chart_degeneracy_report();
            panic!(
                "#2691 REGRESSION: the production entry certified a collapsed chart. \
                 reconstruction_r2={:.4}, chart dispersion={:?}, resolved chart points={:?}, \
                 first coordinates={:?}",
                report.reconstruction_r2,
                chart
                    .axes
                    .iter()
                    .map(|axis| axis.dispersion)
                    .collect::<Vec<_>>(),
                chart
                    .axes
                    .iter()
                    .map(|axis| axis.resolved_points)
                    .collect::<Vec<_>>(),
                coord.iter().take(5).collect::<Vec<_>>()
            );
        }
    }
}


// #2691 — the euclidean/ordinal arm through the FULL production entry is NOT a
// test here: one point of it (n=70, p=3, σ=0.352, `run_outer_rho_search: true`)
// costs 833 s, which does not belong in a routine suite. It was measured once
// and the result is on the issue: the fitted chart came back with standard
// deviation EXACTLY 0.000000e0 and ONE resolved point over 70 rows — the same
// collapse as the circle, via the Euclidean coordinate prior `½αt²`, whose
// minimum is also the chart origin. The cheap inner-path arm above
// (`zz_2691_euclidean_line_refusal_sweep`, 0.1 s a cell) stands as the standing
// witness that the inner solver is not the one that loses the chart.

// #2691 — the CAUSE-side fix (an evidence rail on the outer search's ARD
// chart-coordinate box, so ρ cannot drive the von-Mises precision past the
// point at which the prior owns the coordinate) is NOT in this commit. A first
// attempt was written and measured: the rail it computes was never invoked on
// the σ=1.5 witness (`ard_evidence_domain_upper` emitted zero entries, so its
// own debug line never fired), which means the hook it was placed on is not the
// one the SAE outer search consults. A constraint whose only evidence is that
// something ELSE caught the case is not a fix, so it is held rather than landed.
// What ships here is the reproducer, the mechanism, and the refusal that makes
// the collapse impossible to return silently.
//
// FOLLOW-UP: the cause-side face now exists —
// `outer_objective::periodic_ard_chart_resolution_upper`, installed on BOTH
// exits of `outer_domain_upper_bound`, which IS the hook the SAE outer search
// consults (`fit_outer_stage_to_boundary` → `OuterProblem::run` →
// `rho_optimizer::run::install_objective_domain`). Three tests below gate it:
// the MECHANISM (the face's value moves with the data geometry it is derived
// from), the PROPERTY (the face separates precisions this fixture measurably
// survives from ones it does not), and a BOUNDED σ witness that returns an
// answer at every σ instead of nothing at the σ that matters.

/// Build the #2691 K=1 periodic objective for a planted circle and return
/// `(installed ARD face, seeded ARD entry, chart period)`. The face is READ BACK
/// OUT of `outer_domain_upper_bound()`, so it is what the search is handed, not
/// a number recomputed alongside it.
fn ard_face_for(n: usize, p: usize, radius: f64, sigma: f64) -> (f64, f64, f64) {
    use gam_solve::rho_optimizer::OuterObjective;
    let cloud = planted_circle_cloud(n, p, radius, sigma);
    let z = &cloud.z;
    let (term, _disp) = build_term(z.view(), 1, Topo::Circle, AssignmentMode::softmax(1.0));
    let period = term.assignment.coords[0].effective_axis_periods()[0]
        .expect("the circle chart axis must carry a period");
    // Canonicalize the flat layout against the term's assignment family FIRST —
    // exactly as `fit_outer_stage_to_boundary` does — so the flat index read here
    // is the one the objective uses (a K=1 softmax dictionary has no
    // sparse/router coordinate, which shifts the layout).
    let rho = SaeManifoldRho::new(
        1.0e-3_f64.ln(),
        1.0e-3_f64.ln(),
        vec![array![1.0e-3_f64.ln()]; 1],
    )
    .for_assignment(term.assignment.mode);
    let ard_index = rho.ard_flat_index(0, 0);
    let seed = rho.to_flat()[ard_index];
    let objective =
        SaeManifoldOuterObjective::new(term, z.clone(), None, rho, 40, 1.0, 1.0e-6, 1.0e-6);
    let upper = objective
        .outer_domain_upper_bound()
        .expect("the SAE outer domain face must be constructible")
        .expect("the SAE outer objective declares a typed upper face");
    (upper[ard_index], seed, period)
}

/// #2691 MECHANISM — the ARD chart coordinate's domain face must be denominated
/// in this fixture's own geometry, not in binary64's normal range.
///
/// At the parent of `669d59532` the face was
/// `gam_problem::LOG_STRENGTH_MAX = 700`, a floating-point representability
/// policy by its own module doc, and the generic `RHO_BOUND = 30` was the only
/// thing standing between the ρ search and `α = e^700`. The installed face is
/// the MINIMUM of two derived quantities: the chart-resolution limit
/// `2·(ln 2n − ln P)` and the data-curvature-matching precision
/// `ln(max_row w·‖∂(gated decode)/∂t‖²)`.
///
/// The half of this that cannot be faked is the INVARIANCE. The binding face on
/// this fixture is the curvature one, and the observed latent curvature scales
/// as `‖∂z/∂t‖² ∝ scale²` — so **doubling the planted cloud must raise the face
/// by exactly `2·ln 2`**, and nothing else about the call changes. A hand-picked constant cannot move; a constant "derived" from
/// another constant cannot move with the DATA. That assertion is the one that
/// would fail first if anyone replaced this with a literal.
///
/// Both one-sided halves are pinned too: the face must bind (strictly below
/// `RHO_BOUND` and `LOG_STRENGTH_MAX`) and must still admit the seeded entry.
#[test]
fn zz_2691_the_ard_domain_face_moves_with_the_data_not_with_binary64() {
    let n: usize = 70;
    let (face_r1, seed, period) = ard_face_for(n, 8, 2.086, 0.352);
    // The WHOLE cloud is scaled, signal and noise together: the decoder is a
    // ridge-LSQ fit of `z`, linear in `z` at fixed basis, and the PCA-seeded
    // chart coordinate is scale-invariant — so doubling `z` doubles the decoder
    // exactly. Scaling the radius alone would not double `z` and the invariance
    // below would be approximate for a reason that has nothing to do with the
    // face.
    let (face_r2, _, period_2) = ard_face_for(n, 8, 2.0 * 2.086, 2.0 * 0.352);
    assert_eq!(period, period_2, "the chart period must not depend on scale");

    let resolution_face = 2.0 * ((2.0 * n as f64) / period).ln();
    eprintln!(
        "[2691-face] period={period} seed_log_ard={seed:.4} face(r=2.086)={face_r1:.6} \
         face(r=4.172)={face_r2:.6} resolution_face={resolution_face:.6} RHO_BOUND={} \
         LOG_STRENGTH_MAX={}",
        gam_solve::estimate::RHO_BOUND,
        gam_problem::LOG_STRENGTH_MAX,
    );

    // REJECT half — it binds, and the representability constant is gone.
    assert!(
        face_r1 < gam_solve::estimate::RHO_BOUND,
        "#2691: a face at or above the generic ρ box ({}) constrains nothing; got {face_r1:.6}",
        gam_solve::estimate::RHO_BOUND
    );
    assert!(
        face_r1 < gam_problem::LOG_STRENGTH_MAX,
        "#2691: the ARD face must not be the binary64 representability policy ({}); got {face_r1:.6}",
        gam_problem::LOG_STRENGTH_MAX
    );
    // The face is the MINIMUM of the two derived faces, so it can never exceed
    // the resolution one.
    assert!(
        face_r1 <= resolution_face + 1.0e-12,
        "#2691: the installed face {face_r1:.6} must not exceed the chart-resolution face {resolution_face:.6}"
    );

    // ACCEPT half — the box is not degenerate at the seeded entry.
    assert!(
        seed < face_r1,
        "#2691: the face must still admit the seeded ARD entry {seed:.6}; got {face_r1:.6}"
    );

    // THE INVARIANCE. Observed latent curvature ∝ radius², so log-face moves by
    // exactly 2·ln 2 when the planted radius doubles.
    let delta = face_r2 - face_r1;
    let expected = 2.0 * std::f64::consts::LN_2;
    assert!(
        (delta - expected).abs() <= 1.0e-6 * expected,
        "#2691: doubling the planted cloud must raise the data-curvature face by 2·ln 2 = {expected:.6}; got {delta:.6}. A face that does not move with the data is a constant, not a derivation (and a face still pinned to the chart-resolution limit would not move either — resolution_face={resolution_face:.6})"
    );
}

/// #2691 PROPERTY — every precision the face EXCLUDES must be one this
/// fixture's chart measurably cannot carry, and the healthiest precision must be
/// ADMITTED. Plus the residual gap, located rather than asserted away.
///
/// This is deliberately the bar a NECESSARY face can carry, and no more. The
/// first version of this fix used only the chart-resolution face and the
/// fixture's ladder refuted it — circular recovery R² fell `0.969 → 0.037`
/// between `α = 1e1` and `α = 1e2`, two orders below that face. The
/// data-curvature face added since is strictly tighter, and this test pins the
/// two directions that are actually claimable:
///
/// * REJECT — no rung OUTSIDE the face may be healthy (`R² > 0.9`). A face that
///   excludes a working precision is too tight and this catches it.
/// * ACCEPT — the healthiest rung on the ladder must be INSIDE the face.
///
/// It does NOT assert that everything inside is healthy, because measured, it is
/// not: see the printed `[2691-gap]` line, which locates the transition on a
/// fine ladder and reports how far the installed face sits above it. That gap is
/// recorded as a number in the test output rather than hidden behind a bar
/// chosen to pass, and closing it needs a quantity the system owns and that
/// nobody has identified yet — inventing one is the laundered-literal failure
/// this issue already documented.
///
/// Note on instrument choice, which #2691 is itself about: `distinct_wrapped`
/// stays `70/70` out to `α = 1e4` while circular variance is `6.4e-4`, so a
/// "more than one resolved point" bar passes a dead chart. A COUNT is not a
/// resolution measure. Recovery R² is.
#[test]
fn zz_2691_the_face_excludes_every_precision_this_chart_cannot_carry() {
    let n: usize = 70;
    let (face, seed, period) = ard_face_for(n, 8, 2.086, 0.352);
    let resolution_face = 2.0 * ((2.0 * n as f64) / period).ln();
    let cloud = planted_circle_cloud(n, 8, 2.086, 0.352);
    let z = &cloud.z;

    let recovery_at = |log_ard: f64| -> Option<f64> {
        let (mut term, _disp) = build_term(z.view(), 1, Topo::Circle, AssignmentMode::softmax(1.0));
        let mut rho =
            SaeManifoldRho::new(1.0e-3_f64.ln(), 1.0e-3_f64.ln(), vec![array![log_ard]; 1]);
        term.run_joint_fit_arrow_schur(z.view(), &mut rho, None, 40, 1.0, 1.0e-6, 1.0e-6)
            .ok()?;
        let coords = term.assignment.coords[0].as_matrix();
        let coord: Array1<f64> = coords.column(0).to_owned();
        Some(circular_recovery_r2(&coord, &cloud.theta))
    };

    // A FINE ladder through the measured transition, so the gap is located
    // rather than bracketed by two decades.
    let ladder: Vec<f64> = vec![
        1.0e-3, 1.0e0, 1.0e1, 2.0e1, 4.0e1, 8.0e1, 1.6e2, 3.2e2, 1.0e3, 1.0e4, 1.0e6, 1.0e9,
    ];
    let mut healthy_outside: Vec<f64> = Vec::new();
    let mut best: Option<(f64, f64)> = None;
    let mut last_healthy: Option<f64> = None;
    let mut first_dead: Option<f64> = None;
    eprintln!(
        "[2691-sep] face={face:.4} resolution_face={resolution_face:.4} seed={seed:.4}"
    );
    for alpha in ladder {
        let log_ard = alpha.ln();
        let Some(r2) = recovery_at(log_ard) else {
            eprintln!("[2691-sep] alpha={alpha:.3e} log_ard={log_ard:.3} REFUSED");
            continue;
        };
        let inside = log_ard <= face;
        eprintln!(
            "[2691-sep] alpha={alpha:.3e} log_ard={log_ard:.3} inside={inside} recovery_R2={r2:.4}"
        );
        if best.is_none_or(|(_, b)| r2 > b) {
            best = Some((log_ard, r2));
        }
        if r2 > 0.9 {
            last_healthy = Some(log_ard);
            if !inside {
                healthy_outside.push(log_ard);
            }
        } else if r2 < 0.5 && first_dead.is_none() {
            first_dead = Some(log_ard);
        }
    }

    let (best_log_ard, best_r2) = best.expect("the ladder must fit at least one rung");
    let transition = first_dead.expect(
        "the ladder must reach a dead rung (R² < 0.5) or it cannot say where the chart fails",
    );
    eprintln!(
        "[2691-gap] last_healthy_log_ard={:?} first_dead_log_ard={transition:.4} \
         installed_face={face:.4} face_minus_first_dead={:.4} \
         resolution_face_minus_first_dead={:.4}",
        last_healthy,
        face - transition,
        resolution_face - transition
    );

    // ACCEPT — the healthiest precision must be reachable.
    assert!(
        best_log_ard <= face,
        "#2691 ACCEPT half: the face {face:.4} excludes the healthiest precision on the ladder \
         (log_ard {best_log_ard:.4}, recovery R² {best_r2:.4}) — the face is too tight"
    );
    assert!(
        seed < face,
        "#2691: the face must still admit the seeded ARD entry {seed:.6}; got {face:.6}"
    );

    // REJECT — nothing the face throws away may be a working chart.
    assert!(
        healthy_outside.is_empty(),
        "#2691 REJECT half: the face {face:.4} excludes precisions this chart SURVIVES \
         (recovery R² > 0.9) at log_ard {healthy_outside:?} — the face is too tight"
    );

    // The installed face must be strictly tighter than the chart-resolution face
    // alone, which the ladder measured admitting three dead rungs. This is the
    // part of the gap the data-curvature face actually closes; the rest is
    // reported on the `[2691-gap]` line above and is NOT claimed here.
    assert!(
        face < resolution_face,
        "#2691: the data-curvature face must bind on this fixture — installed {face:.4} is not \
         below the chart-resolution face {resolution_face:.4}, so the two-order gap the ladder \
         measured is untouched"
    );
}

/// #2691 — the σ witness, BOUNDED BY ITERATIONS rather than by the clock.
///
/// The first attempt at this drove the full production entry at σ = 1.5 with an
/// unbounded outer budget. Measured: base refused `DegenerateChart` at 659 s;
/// with the box corrected it was still running at 3600 s and was killed with no
/// `test result:` line. **A timeout measured nothing**, and leaving it there
/// would mean the regime with the worst behaviour is the one with no instrument
/// — a fix and a hang produce the same absence.
///
/// So the budget is `with_max_iter`, not a wall clock: every σ returns a
/// terminal ρ whether or not the search converged, and the row is printed either
/// way. `OuterProblem::run` is the same seam that installs the objective domain
/// (`rho_optimizer::run::install_objective_domain`), so the face under test is
/// the one in force.
///
/// The assertion is the one this issue was filed on and it is now checkable at
/// every σ: the terminal ARD log-precision must lie inside the installed face,
/// and must NOT sit on the generic `RHO_BOUND = 30` the issue measured it
/// railing to. The printed `moved` column is the mute-fixture guard: if the
/// search never left the seed at any σ, the run proved nothing and the test says
/// so rather than passing quietly.
#[test]
fn zz_2691_bounded_sigma_witness_returns_an_answer_at_every_sigma() {
    let n: usize = 70;
    let p: usize = 8;
    let radius: f64 = 2.086;
    let sigmas: Vec<f64> = vec![0.352, 1.0, 1.5];
    let mut any_moved = false;
    let mut on_generic_box: Vec<f64> = Vec::new();
    let mut outside_face: Vec<(f64, f64, f64)> = Vec::new();

    eprintln!("[2691-sigma] sigma\tface\tseed\tterminal_log_ard\tmoved\tconverged\tsecs");
    for &sigma in &sigmas {
        let (face, seed, _period) = ard_face_for(n, p, radius, sigma);
        let cloud = planted_circle_cloud(n, p, radius, sigma);
        let z = &cloud.z;
        let (term, _disp) = build_term(z.view(), 1, Topo::Circle, AssignmentMode::softmax(1.0));
        let rho = SaeManifoldRho::new(
            1.0e-3_f64.ln(),
            1.0e-3_f64.ln(),
            vec![array![1.0e-3_f64.ln()]; 1],
        )
        .for_assignment(term.assignment.mode);
        let ard_index = rho.ard_flat_index(0, 0);
        let rho_flat = rho.to_flat();
        let n_params = rho_flat.len();
        let mut objective =
            SaeManifoldOuterObjective::new(term, z.clone(), None, rho, 40, 1.0, 1.0e-6, 1.0e-6);
        let start = std::time::Instant::now();
        // The BUDGET, and the whole point of this test: an iteration cap returns
        // a terminal ρ at every σ. A clock cap returns nothing at the σ that
        // matters.
        let result = gam_solve::rho_optimizer::OuterProblem::new(n_params)
            .with_initial_rho(rho_flat.clone())
            .with_max_iter(8)
            .run(&mut objective, "SAE #2691 bounded σ witness");
        let secs = start.elapsed().as_secs_f64();
        let (terminal, converged) = match &result {
            Ok(outcome) => (outcome.rho[ard_index], outcome.converged()),
            Err(error) => {
                eprintln!(
                    "[2691-sigma] {sigma:.3}\t{face:.4}\t{seed:.4}\t-\t-\t-\t{secs:.1}\tREFUSED: {}",
                    format!("{error}").replace('\n', " ")
                );
                continue;
            }
        };
        let moved = (terminal - seed).abs() > 1.0e-9;
        any_moved |= moved;
        eprintln!(
            "[2691-sigma] {sigma:.3}\t{face:.4}\t{seed:.4}\t{terminal:.4}\t{moved}\t{converged}\t{secs:.1}"
        );
        if (terminal - gam_solve::estimate::RHO_BOUND).abs() <= 1.0e-6 {
            on_generic_box.push(sigma);
        }
        if terminal > face + 1.0e-9 {
            outside_face.push((sigma, terminal, face));
        }
    }

    assert!(
        any_moved,
        "#2691: the bounded σ witness never moved the ARD coordinate off its seed at any σ, so \
         it cannot say anything about railing — raise the iteration budget or the fixture is mute"
    );
    assert!(
        on_generic_box.is_empty(),
        "#2691: the terminal ARD log-precision sat on the generic ρ box ({}) at σ {on_generic_box:?} \
         — that is the rail this issue was filed on",
        gam_solve::estimate::RHO_BOUND
    );
    assert!(
        outside_face.is_empty(),
        "#2691: the terminal ARD log-precision escaped its installed face at (σ, terminal, face) \
         {outside_face:?}"
    );
}
