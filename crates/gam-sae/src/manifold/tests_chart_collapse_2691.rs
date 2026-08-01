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
// `rho_optimizer::run::install_objective_domain`). The two tests below gate it:
// the first on the MECHANISM (the face's value and where its inputs come from),
// the second on the PROPERTY, measured: every precision the face excludes has a
// measurably unresolvable chart, and the seeded entry's does not.

/// #2691 MECHANISM — the ARD chart coordinate's domain face must be the CHART's
/// own resolution limit, not binary64's normal range.
///
/// At the parent commit `SaeManifoldRho::flat_domain_upper_bound` handed
/// `gam_problem::LOG_STRENGTH_MAX = 700` to every outer
/// coordinate — a floating-point representability policy by that module's own
/// doc — and the generic `RHO_BOUND = 30` box was the only thing standing
/// between the ρ search and `α = e^700` on a quantity that deletes the chart
/// nine orders lower.
///
/// The face installed instead is `log α ≤ 2·(ln 2n − ln P)`, i.e. the precision
/// whose own concentration scale `σ_u = 1/(P√α)` equals the occupancy
/// adjudicator's resolution floor `σ_floor = 1/(2n)`
/// (`coordinate_fidelity::classify_occupancy`). This test pins BOTH halves:
///
/// * REJECT — the face is finite, strictly below `RHO_BOUND` and below
///   `LOG_STRENGTH_MAX`, so it actually binds and the nine orders are gone;
/// * ACCEPT — the box is not degenerate (it still admits the seeded entry), and
///   the face MOVES WITH `n` by exactly the `σ_floor` law: doubling the row
///   count raises it by `2·ln 2`. A hand-picked constant cannot do that, so
///   this half is what distinguishes a derived face from a laundered literal.
#[test]
fn zz_2691_the_ard_domain_face_is_the_chart_resolution_not_binary64() {
    use gam_solve::rho_optimizer::OuterObjective;

    // The period of the canonical `d=1` circle chart, read off the term rather
    // than assumed, so the assertion below is the derivation and not a copy of
    // one number into two places.
    let face_at = |n: usize| -> (f64, f64, f64) {
        let cloud = planted_circle_cloud(n, 8, 2.086, 1.5);
        let z = &cloud.z;
        let (term, _disp) = build_term(z.view(), 1, Topo::Circle, AssignmentMode::softmax(1.0));
        let period = term.assignment.coords[0].effective_axis_periods()[0]
            .expect("the circle chart axis must carry a period");
        // Canonicalize the flat layout against the term's assignment family
        // FIRST — exactly as `fit_outer_stage_to_boundary` does — so the flat
        // index read here is the one the objective will use (a K=1 softmax
        // dictionary has no sparse/router coordinate, which shifts the layout).
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
        (upper[ard_index], period, seed)
    };

    let (face_70, period, seed) = face_at(70);
    let (face_140, period_140, _) = face_at(140);
    assert_eq!(period, period_140, "the chart period must not depend on n");

    // The derivation, evaluated here from `n` and `P` alone.
    let expected_70 = 2.0 * ((2.0 * 70.0) / period).ln();
    eprintln!(
        "[2691-face] period={period} seed_log_ard={seed:.4} face(n=70)={face_70:.6} \
         expected={expected_70:.6} face(n=140)={face_140:.6} RHO_BOUND={} \
         LOG_STRENGTH_MAX={}",
        gam_solve::estimate::RHO_BOUND,
        gam_problem::LOG_STRENGTH_MAX,
    );
    assert!(
        (face_70 - expected_70).abs() <= 1.0e-12 * expected_70.abs(),
        "#2691: the ARD face must be the chart-resolution limit 2·(ln 2n − ln P) = {expected_70:.6}; got {face_70:.6}"
    );

    // REJECT half — it binds, and the representability constant is gone.
    assert!(
        face_70 < gam_solve::estimate::RHO_BOUND,
        "#2691: a face at or above the generic ρ box ({}) constrains nothing; got {face_70:.6}",
        gam_solve::estimate::RHO_BOUND
    );
    assert!(
        face_70 < gam_problem::LOG_STRENGTH_MAX,
        "#2691: the ARD face must not be the binary64 representability policy ({}); got {face_70:.6}",
        gam_problem::LOG_STRENGTH_MAX
    );

    // ACCEPT half — the box is not degenerate at the seeded entry.
    assert!(
        seed < face_70,
        "#2691: the face must still admit the seeded ARD entry {seed:.6}; got {face_70:.6}"
    );

    // ACCEPT half — the face is denominated in the data resolution. Doubling
    // `n` halves `σ_floor` and must raise the face by exactly `2·ln 2`; a
    // literal (however derived-sounding) cannot move at all.
    let delta = face_140 - face_70;
    let expected_delta = 2.0 * std::f64::consts::LN_2;
    assert!(
        (delta - expected_delta).abs() <= 1.0e-12 * expected_delta,
        "#2691: doubling n must raise the chart-resolution face by 2·ln 2 = {expected_delta:.6}; got {delta:.6} (a face that does not move with n is a constant, not a derivation)"
    );
}


/// #2691 PROPERTY — every ARD precision the face EXCLUDES must be one this
/// fixture's own chart measurably cannot survive, and the face must still admit
/// the precisions it survives.
///
/// This is the half of the claim that can be checked against data rather than
/// against the algebra. The face says: past `alpha = (2n/P)^2` the von-Mises
/// prior's own concentration scale is finer than the occupancy adjudicator's
/// resolution floor `sigma_floor = 1/(2n)`, so `OccupancyLaw::Collapsed` is
/// FORCED whatever the likelihood wants. That is a claim about the fitted
/// chart's occupied extent, and the extent is measurable: drive the inner joint
/// fit at fixed `alpha` and measure the smallest containing arc of the folded
/// coordinate — `1 - largest cyclic gap`, exactly `coordinate_fidelity::
/// occupied_extent`.
///
/// Two-sided, and both halves would break if the face went back to the generic
/// box:
///
/// * REJECT — every rung ABOVE the face has measured extent BELOW `sigma_floor`.
///   At `RHO_BOUND = 30` or `LOG_STRENGTH_MAX = 700` those rungs are inside the
///   domain, which is the #2691 defect.
/// * ACCEPT — the seeded entry's own precision has extent far above
///   `sigma_floor`, so the face has not swallowed the working regime.
///
/// What this test deliberately does NOT claim: that everything inside the face
/// is a good chart. Measured on this fixture (see
/// `zz_2691_ard_precision_ladder_collapses_the_chart`), circular recovery R²
/// falls from 0.969 at `alpha = 1e1` to 0.037 at `alpha = 1e2` — two orders
/// BELOW the face. `Collapsed` is a weaker condition than "recovers the planted
/// structure", the face is denominated in `Collapsed` because that is the
/// threshold the system owns, and so this face is NECESSARY, not sufficient.
/// Stating it as sufficient would be the overclaim.
#[test]
fn zz_2691_every_precision_outside_the_face_measurably_collapses_the_chart() {
    let n: usize = 70;
    let cloud = planted_circle_cloud(n, 8, 2.086, 0.352);
    let z = &cloud.z;
    let period = 1.0_f64;
    let face = 2.0 * ((2.0 * n as f64) / period).ln();
    let sigma_floor = 1.0 / (2.0 * n as f64);

    // `coordinate_fidelity::occupied_extent` on the circle: the smallest arc
    // containing every row, i.e. one minus the largest cyclic gap. A raw std
    // cannot stand in for this — a chart piled onto the wrap point reads 0.5.
    let extent_at = |log_ard: f64| -> f64 {
        let (mut term, _disp) = build_term(z.view(), 1, Topo::Circle, AssignmentMode::softmax(1.0));
        let mut rho =
            SaeManifoldRho::new(1.0e-3_f64.ln(), 1.0e-3_f64.ln(), vec![array![log_ard]; 1]);
        term.run_joint_fit_arrow_schur(z.view(), &mut rho, None, 40, 1.0, 1.0e-6, 1.0e-6)
            .unwrap_or_else(|error| {
                panic!("#2691: the fixed-α inner fit must succeed at log_ard={log_ard:.3}: {error}")
            });
        let coords = term.assignment.coords[0].as_matrix();
        let mut pts: Vec<f64> = coords
            .column(0)
            .iter()
            .map(|&t| {
                let f = t.rem_euclid(period) / period;
                if f >= 1.0 { 0.0 } else { f }
            })
            .collect();
        pts.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mut largest_gap = (pts[0] + 1.0) - pts[pts.len() - 1];
        for pair in pts.windows(2) {
            largest_gap = largest_gap.max(pair[1] - pair[0]);
        }
        (1.0 - largest_gap).max(0.0)
    };

    // The seeded entry, and two decade rungs strictly ABOVE the face. Decades of
    // the fixture's own ladder, not values chosen to make the bar pass.
    let seed_log_ard = 1.0e-3_f64.ln();
    let outside: Vec<f64> = vec![1.0e6_f64.ln(), 1.0e9_f64.ln()];
    let seed_extent = extent_at(seed_log_ard);
    eprintln!(
        "[2691-extent] face={face:.4} sigma_floor={sigma_floor:.6e} \
         seed log_ard={seed_log_ard:.3} extent={seed_extent:.6e}"
    );
    assert!(
        seed_log_ard < face,
        "#2691: the face must admit the seeded entry {seed_log_ard:.4}; face={face:.4}"
    );
    assert!(
        seed_extent > sigma_floor,
        "#2691 ACCEPT half: the seeded entry's chart must be resolvable — extent \
         {seed_extent:.6e} must exceed sigma_floor {sigma_floor:.6e}"
    );

    for log_ard in outside {
        assert!(
            log_ard > face,
            "#2691: rung {log_ard:.4} was meant to be OUTSIDE the face {face:.4}"
        );
        let extent = extent_at(log_ard);
        eprintln!("[2691-extent] outside log_ard={log_ard:.3} extent={extent:.6e}");
        assert!(
            extent < sigma_floor,
            "#2691 REJECT half: log_ard={log_ard:.4} lies outside the chart-resolution face \
             {face:.4}, so the chart there must be unresolvable — measured extent \
             {extent:.6e} is not below sigma_floor {sigma_floor:.6e}. A face that excludes \
             a precision the chart survives is too tight; a face that admits one it does not \
             is the #2691 defect."
        );
    }
}

// #2691 — the σ=1.5 outer witness (n=70, p=8, full production entry,
// `run_outer_rho_search: true`) is NOT a test here, and the reason is a
// measurement, not a preference. Both arms were run at the pinned base
// `2b3029dc6` on one host, one variable (this fix reverted / applied), same
// test source:
//
//   base    659.0 s  -> refused `DegenerateChart` (circular variance 0.000e0,
//                       the landed `6235c387b` guard firing correctly)
//   patched 3600   s  -> TIMEOUT, no `test result:` line, no verdict
//
// A timeout is not a pass and not a fail; it measured nothing about the chart,
// and a fix and a hang produce the same absence. It is recorded here rather
// than shipped so nobody reads its silence as green. The property this commit
// DOES gate is the extent test above, which terminates in seconds.
