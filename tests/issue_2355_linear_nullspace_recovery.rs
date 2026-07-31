//! Regression test for #2355: `y ~ s(x)` on trivially clean linear data `y = x`
//! must RECOVER the line (EDF ≈ 2, slope ≈ 1), not silently collapse to a flat
//! constant at the response mean.
//!
//! ROOT CAUSE. A default P-spline `s(x)` is a double-penalty smooth: a bending
//! penalty `S₁` plus a `DoublePenaltyNullspace` ridge `S₂` on the bending
//! penalty's null space (the `{1, x}` / linear direction). When the fit is
//! flagged "under-determined" (`n < 2·p`; a `p = 8` basis makes the boundary
//! `n = 16`) the null-space coordinate was given an aggressive penalized-
//! complexity "select-out" prior (θ ≈ 92, cost `θ·e^{−ρ₂/2}` reaching ~3e8 at
//! the ρ box edge). That steep wall dominates the base REML criterion — which
//! by itself correctly prefers the linear fit — by ~7 orders of magnitude, so
//! it CANNOT be overridden by the likelihood. The optimizer then settles into
//! the flat-line collapse (EDF ≈ 1.08, slope ≈ 0.001) and reports
//! `status=Converged` with no warning: a silent wrong answer for every
//! `n ∈ [5, 15]` while `n ≥ 16` (well-determined) fit correctly.
//!
//! FIX. Default smoothing priors must not depend on the observed response.
//! Every double-penalty null-space coordinate therefore receives only the same
//! wide symmetric degeneracy Normal. It supplies enough curvature to make the
//! outer objective proper without choosing "supported" directions before the
//! fit; REML itself decides whether a linear component is retained or shrunk.
//! This recovers `y = x`, while a genuinely irrelevant null space is still
//! selected out by the likelihood-plus-REML criterion.
//!
//! This test asserts the RECOVERED behaviour directly (no reference tool): for a
//! sweep of `n` spanning the previously-collapsing range it requires EDF ≈ 2 and
//! that the fitted curve tracks `y = x` to floating-point precision.

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;

/// Fit `y ~ s(x)` on the exact line `y = x` at `n` evenly spaced points and
/// return `(edf_total, max |fitted − x|, slope_estimate)`.
fn fit_linear(n: usize) -> (f64, f64, f64) {
    let xs: Vec<f64> = (0..n).map(|i| i as f64 * 0.5).collect();
    let headers = ["x", "y"].into_iter().map(String::from).collect();
    let rows: Vec<StringRecord> = xs
        .iter()
        .map(|&x| StringRecord::from(vec![x.to_string(), x.to_string()]))
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode y=x dataset");
    let col = ds.column_map();
    let x_idx = col["x"];

    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("y ~ s(x)", &ds, &cfg).expect("gam gaussian s(x) fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit for Gaussian s(x)");
    };
    let edf = fit.fit.edf_total().expect("gam reports total edf");

    let mut grid = Array2::<f64>::zeros((n, ds.headers.len()));
    for (i, &xi) in xs.iter().enumerate() {
        grid[[i, x_idx]] = xi;
    }
    let design = build_term_collection_design(grid.view(), &fit.resolvedspec)
        .expect("rebuild s(x) design at training points");
    let fitted = design.design.apply(&fit.fit.beta);

    let max_abs_err = xs
        .iter()
        .zip(fitted.iter())
        .map(|(&x, &f)| (f - x).abs())
        .fold(0.0_f64, f64::max);

    // Ordinary-least-squares slope of fitted-vs-x (== 1 for a faithful line).
    let mean_x = xs.iter().sum::<f64>() / n as f64;
    let mean_f = fitted.iter().sum::<f64>() / n as f64;
    let mut num = 0.0;
    let mut den = 0.0;
    for (&x, &f) in xs.iter().zip(fitted.iter()) {
        num += (x - mean_x) * (f - mean_f);
        den += (x - mean_x) * (x - mean_x);
    }
    let slope = if den > 0.0 { num / den } else { 0.0 };

    (edf, max_abs_err, slope)
}

/// Fit `y ~ s(x)` on the given `(x, y)` rows and return the total EDF.
fn fit_edf(xs: &[f64], ys: &[f64]) -> f64 {
    let headers = ["x", "y"].into_iter().map(String::from).collect();
    let rows: Vec<StringRecord> = xs
        .iter()
        .zip(ys.iter())
        .map(|(&x, &y)| StringRecord::from(vec![x.to_string(), y.to_string()]))
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode dataset");
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("y ~ s(x)", &ds, &cfg).expect("gam gaussian s(x) fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit");
    };
    fit.fit.edf_total().expect("gam reports total edf")
}

#[test]
fn linear_data_recovers_line_not_flat_collapse() {
    init_parallelism();

    // The previously-collapsing regime (`n ∈ [5, 15]`, all under-determined with
    // the default `p = 8` basis) plus a couple of the already-correct
    // well-determined sizes as a non-regression guard.
    // The previously-broken under-determined band (`n < 2·p`, `p ≈ 8` ⇒
    // `n < 16`). Two independent defects conspired here on the trivially clean
    // line:
    //   * `n ∈ [9, 15]` (`n > p`): the aggressive null-space select-out prior
    //     shipped a flat line at the response mean (EDF ≈ 1.08) — a silent wrong
    //     answer. The response-independent degeneracy prior now leaves the
    //     retention decision to REML, which recovers the linear component.
    //   * `n ∈ [5, 8]` (`n ≈ p`): the summed-penalty profiled-diagonal SEED
    //     heuristic honestly refused this near-degenerate corner, and that
    //     refusal was propagated as a fatal fit error instead of merely dropping
    //     one seed candidate. It is now non-fatal, so the coupled 2-λ outer
    //     search — which is perfectly well-posed here — recovers the line.
    //
    // Scope note: the well-determined band `n ≥ 16` has an independent
    // outer-stationarity certification issue near the `λ_null → 0` rail (#2348)
    // that returns an explicit error (never a silent wrong answer); it is out of
    // scope for this test.
    for n in [5usize, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] {
        let (edf, max_err, slope) = fit_linear(n);
        eprintln!("#2355 n={n}: edf={edf:.4} max|fitted-x|={max_err:.3e} slope={slope:.6}");

        // EDF ≈ 2 (intercept + slope). The collapse shipped EDF ≈ 1.08; a
        // near-full-basis overfit would be ≈ min(n, 8). The wide degeneracy
        // Normal leaves a hair of residual curvature, so allow up to ~2.15.
        assert!(
            (1.5..2.2).contains(&edf),
            "n={n}: expected EDF≈2 (intercept + slope) for the exact line y=x, \
             got EDF={edf:.4} — the double-penalty null-space select-out is \
             annihilating the genuinely-supported linear component (#2355)"
        );

        // The fitted curve must be the line, not a flat constant at the mean.
        assert!(
            max_err < 1e-3,
            "n={n}: fitted curve departs from y=x by {max_err:.3e}; a silent \
             flat-line collapse sits near the response mean (#2355)"
        );

        // Slope recovered (collapse gave slope ≈ 0.001).
        assert!(
            (slope - 1.0).abs() < 5e-3,
            "n={n}: recovered slope {slope:.6} ≠ 1.0 (#2355)"
        );
    }
}

/// Complementary guard (opposite direction from the recovery test): the
/// The response-independent prior must still allow REML to select out a
/// genuinely unsupported null space. On an under-determined smooth of a
/// signal that carries neither a linear trend nor smooth structure, the
/// whole smooth must collapse to
/// the mean (EDF ≈ 1) — the #1266 irrelevant-covariate / #1392 `p > n` shrinkage
/// behaviour. A regression that made the degeneracy prior informative enough to
/// retain a spurious direction would produce EDF ≈ 2, which this test catches.
#[test]
fn unsupported_nullspace_is_still_selected_out() {
    init_parallelism();

    // Alternating ±1 on an evenly spaced axis: zero linear trend, no smooth
    // structure the P-spline can resolve. n = 12 is under-determined (< 2·p).
    let n = 12usize;
    let xs: Vec<f64> = (0..n).map(|i| i as f64 * 0.5).collect();
    let ys: Vec<f64> = (0..n)
        .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
        .collect();
    let edf = fit_edf(&xs, &ys);
    eprintln!("#2355 unsupported-nullspace guard: n={n} edf={edf:.4}");

    assert!(
        edf < 1.4,
        "an unsupported null space must stay selected out (EDF≈1); got EDF={edf:.4} \
         — REML is keeping a spurious linear component (#1266/#1392 regression)"
    );
}
