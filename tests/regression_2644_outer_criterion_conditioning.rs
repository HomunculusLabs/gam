//! #2644: the outer smoothing-parameter optimizer refusing to certify a
//! stationary optimum at an interior PSD minimum.
//!
//! ONE GATE and two reporting probes.
//!
//! ## The gate — `te_with_disparate_scales_certifies`
//!
//! `y ~ te(a,b,k=5)` on 300 rows with `a ∈ [0,1]` against `b ∈ [0,1000]`. This
//! is `misc::mega_batch_k::te_with_disparate_scales`, a `GAM_ERROR` row on both
//! of the last two reference-quality nightlies, and it moved BETWEEN the two
//! shapes the refusal takes:
//!
//! ```text
//! run 30602192415:  cost-only value disagrees with analytic-sample value at
//!                   the same outer point (4.141e-5 vs a 2.842e-5 roundoff bound)
//! run 30619084852:  |g|=|Pg|=2.458e2  bound=1.015e1  hessian_psd=yes
//!                   after exhausting a 400-iteration outer budget
//! ```
//!
//! Both are the same criterion. Its penalty blocks reach `κ(S_λ) = 3.5e19`,
//! where the assembled-matrix Cholesky that used to price `log|S_λ|₊` disagrees
//! with the root-scale spectrum by **20 log units** — and, because that Cholesky
//! FAILS at some trial `rho` and succeeds at others, `V(rho)` was priced by two
//! different formulas with a step of that size between them. 196 of this fit's
//! 1015 penalty-logdet builds took one formula and 819 the other.
//!
//! Measured here: **refused after 400 outer iterations in 0.94 s** before
//! `75563f13e`, **certifies in 0.18 s** after it.
//!
//! The gate lives in `tests/` rather than only in the nightly quality suite
//! because the quality suite runs on a 6-hour schedule and this is a
//! sub-second fit.
//!
//! ## Reporting probes (print, never fail)
//!
//! * `prostate` — `y ~ s(pc1,k=5) + s(pc2,k=5)`, binomial/logit, 490 rows of
//!   `bench/datasets/prostate.csv`. Three reference-quality rows report the
//!   same `|g|=|Pg|=1.792e-3 bound=1.133e-3` from it. STILL REFUSES after the
//!   `log|S_λ|₊` fix, now at `|Pg|=3.396e-3` against `bound=3.015e-3`, because
//!   its residual criterion noise is `log|H|`, which is priced from the
//!   eigenvalues of the ASSEMBLED `H` (`κ = 3.8e12` ⇒ `6.6e-4` of scatter at
//!   fixed `rho`, against an outer cost floor of `1.8e-5`). Recovering that one
//!   needs a root of `H` that is not formed by summing `XᵀWX` and `S_λ` in f64,
//!   which is a change to what P-IRLS publishes; it is not attempted here.
//! * `matern` — the fit the issue thread recommends as a reproducer. Recorded
//!   because it does NOT reproduce any more (it certifies at `|Pg|=2.025e-5`
//!   against `bound=9.876e-5`).
//!
use gam::data::EncodedDataset;
use gam::{FitConfig, encode_recordswith_inferred_schema, fit_from_formula, load_csvwith_inferred_schema};
use csv::StringRecord;
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};
use std::path::Path;

const PROSTATE_CSV: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/bench/datasets/prostate.csv");

fn subset_rows(ds: &EncodedDataset, rows: &[usize]) -> EncodedDataset {
    let mut sub = ds.clone();
    let ncol = ds.values.ncols();
    let mut values = Array2::<f64>::zeros((rows.len(), ncol));
    for (new_r, &old_r) in rows.iter().enumerate() {
        for c in 0..ncol {
            values[[new_r, c]] = ds.values[[old_r, c]];
        }
    }
    sub.values = values;
    sub
}

#[test]
fn zz_probe_2644_prostate_binomial_logit() {
    gam_solve::progress_log::init_logging_at(log::LevelFilter::Info);
    let ds = load_csvwith_inferred_schema(Path::new(PROSTATE_CSV)).expect("load prostate.csv");
    let n = ds.values.nrows();
    let train_rows: Vec<usize> = (0..n).filter(|i| i % 4 != 0).collect();
    let ds_train = subset_rows(&ds, &train_rows);
    let cfg = FitConfig {
        family: Some("binomial".to_string()),
        link: Some("logit".to_string()),
        ..FitConfig::default()
    };
    let started = std::time::Instant::now();
    let outcome = fit_from_formula("y ~ s(pc1, k=5) + s(pc2, k=5)", &ds_train, &cfg);
    println!(
        "[probe-2644-prostate] n_train={} elapsed={:.2}s",
        train_rows.len(),
        started.elapsed().as_secs_f64()
    );
    match outcome {
        Ok(_) => println!("[probe-2644-prostate] FIT OK"),
        Err(e) => println!("[probe-2644-prostate] FIT ERR: {e}"),
    }
}

const N_TRAIN: usize = 1_500;
const SIGMA: f64 = 0.10;
const TRAIN_SEED: u64 = 1_039;

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

fn build_dataset(n: usize, sigma: f64, seed: u64) -> EncodedDataset {
    let mut rng = StdRng::seed_from_u64(seed);
    let latent = Uniform::new(0.0, 1.0).expect("uniform");
    let noise = Normal::new(0.0, sigma).expect("normal");
    let headers = ["x0", "x1", "x2", "y"]
        .iter()
        .cloned()
        .map(String::from)
        .collect();
    let rows: Vec<StringRecord> = (0..n)
        .map(|_| {
            let t = latent.sample(&mut rng);
            let coords = latent_to_coords(t);
            let y = truth(t) + noise.sample(&mut rng);
            StringRecord::from(vec![
                coords[0].to_string(),
                coords[1].to_string(),
                coords[2].to_string(),
                y.to_string(),
            ])
        })
        .collect();
    encode_recordswith_inferred_schema(headers, rows).expect("encode")
}

#[test]
fn zz_probe_2644_matern_outer_gradient() {
    gam_solve::progress_log::init_logging_at(log::LevelFilter::Info);
    let ds = build_dataset(N_TRAIN, SIGMA, TRAIN_SEED);
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let started = std::time::Instant::now();
    let outcome = fit_from_formula("y ~ matern(x0, x1, x2, k=16)", &ds, &cfg);
    println!("[probe-2644] elapsed={:.2}s", started.elapsed().as_secs_f64());
    match outcome {
        Ok(_) => println!("[probe-2644] FIT OK"),
        Err(e) => println!("[probe-2644] FIT ERR: {e}"),
    }
}

// The gate. See the module header.

fn mk_2d(
    n: usize,
    f: impl Fn(f64, f64) -> f64,
    ra: (f64, f64),
    rb: (f64, f64),
    sigma: f64,
    seed: u64,
) -> EncodedDataset {
    let mut rng = StdRng::seed_from_u64(seed);
    let ua = Uniform::new(ra.0, ra.1).expect("finite a range");
    let ub = Uniform::new(rb.0, rb.1).expect("finite b range");
    let noise = Normal::new(0.0, sigma).expect("finite sigma");
    let h = ["a", "b", "y"].into_iter().map(String::from).collect();
    let mut rows = Vec::with_capacity(n);
    for _ in 0..n {
        let a = ua.sample(&mut rng);
        let b = ub.sample(&mut rng);
        let y = f(a, b) + noise.sample(&mut rng);
        rows.push(StringRecord::from(vec![
            a.to_string(),
            b.to_string(),
            y.to_string(),
        ]));
    }
    encode_recordswith_inferred_schema(h, rows).expect("encode")
}

#[test]
fn te_with_disparate_scales_certifies() {
    gam_solve::progress_log::init_logging_at(log::LevelFilter::Info);
    let ds = mk_2d(300, |a, b| a + b, (0.0, 1.0), (0.0, 1000.0), 0.05, 7);
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let started = std::time::Instant::now();
    let outcome = fit_from_formula("y~te(a,b,k=5)", &ds, &cfg);
    println!(
        "[probe-2644-te] elapsed={:.2}s",
        started.elapsed().as_secs_f64()
    );
    let Err(error) = outcome else {
        println!("[probe-2644-te] FIT OK");
        return;
    };
    panic!(
        "#2644: `y~te(a,b,k=5)` on disparate scales must reach a certified outer \
         optimum. This fit refused for 400 outer iterations at |Pg|=2.458e2 while \
         `log|S_lambda|+` was priced from a factorization of the assembled penalty \
         sum, whose error is O(eps*kappa) and whose Cholesky FAILS at some trial \
         rho and succeeds at others -- so the objective carried a step of ~20 log \
         units. A refusal here means that pricing (or an equivalent one) is back: \
         {error}"
    );
}

// ─────────────────────────────────────────────────────────────────────────
// Reporting probe: `misc::broad_sweep_batch_h::matern_low_n_does_not_crash`.
//
// `y ~ matern(x, nu=5/2)`, Gaussian, **15 rows**, one coordinate — the
// smallest witness of an outer-criterion refusal in the suite, and the one
// that fires on the joint-spatial route-AGREEMENT half of the certificate
// rather than its descent half. It refuses in 0.6 s with
//
//   joint_seed = 2.787290435812e0   baseline = 2.787290399076e0
//   gap = 3.674e-8 (1.318e-8 relative)   agreement_tolerance = 2.787e-8
//
// bit-identically across two nightlies on different runners and two local
// runs. This is the residue `9e8ca98da` left open when it closed the
// O(eps*kappa) log-determinant family, and the root-scale work could not have
// touched it.
//
// MEASURED (2026-07-31, at `04671c5b3`), by widening the in-tree
// `[#1271-diag]` from `{:.6}` to `{:.17e}` and adding `beta` and
// `h_total_original` to it. Both routes evaluate at a BIT-IDENTICAL `rho`:
//
//   logS, logH                          bit-identical to 17 digits
//   h_total_original (all 49 entries)   bit-identical (max|H1-H2| = 0.0)
//   pirls_edf                           identical
//   beta                                differs in COORDINATE 0 ALONE, by
//                                       2.13012903753208305e-1; coordinates
//                                       1..6 agree to ~10 digits
//   H[0][0] = 1.50000000100000008e1 = n + 1e-8, so coordinate 0 is the
//   intercept and it carries the absolute `FIXED_STABILIZATION_RIDGE`, which
//   that constant's own doc says `penalty_term` carries as `ridge*||beta||^2`.
//
// The gap is then fully accounted for, with no free parameter:
//
//   ridge * (b0'^2 - b0^2) = 7.7024058e-10   vs measured dDp = 7.7024051e-10
//                                            (ratio 1.00000009)
//   (n/2) * dDp / Dp       = 3.6736410122e-8 vs measured gap = 3.673641e-8
//
// So 100% of the disagreement is `(n/2)*d log(rss+pen)`, 0% is either
// log-determinant, and the ONLY term that notices the intercept moving is the
// 1e-8 ridge. The refusal is the certificate correctly reporting that the
// criterion assigns two values to one predictor.
//
// REFUTED along the way, so nobody re-runs them: the offset seam (instrumented
// `compose_offset` itself; both routes compose an identically ZERO offset from
// an identically zero affine channel), and the Gaussian sufficient-statistic
// cache (`[gaussian-fixed-cache]` is built 11 times, on both routes).
//
// STILL OPEN: what makes the two routes solve different systems at all. Both
// report the same `H` with `lambda_min = 1.386e-1` and a strictly convex
// quadratic has one minimizer, so the designs must differ while
// `X'X + S_lambda + delta I` stays bit-identical. Their own logs disagree at
// the same `rho` on `inner pirls solve ... max_eta` (0.6 vs 0.8) and on
// `reml_laml cost_only_done ... ext_dim` (0 vs 1).
//
// This probe PRINTS and never fails: the defect is real but unfixed, and a
// gate here would be a second red on a known-open cause.
// ─────────────────────────────────────────────────────────────────────────

fn mk_1d(n: usize, f: impl Fn(f64) -> f64, sigma: f64, seed: u64) -> EncodedDataset {
    let mut rng = StdRng::seed_from_u64(seed);
    let u = Uniform::new(0.0_f64, 1.0).expect("uniform");
    let noise = Normal::new(0.0, sigma).expect("normal");
    let mut x: Vec<f64> = (0..n).map(|_| u.sample(&mut rng)).collect();
    x.sort_by(|a, b| a.partial_cmp(b).expect("finite draws"));
    let y: Vec<f64> = x.iter().map(|&t| f(t) + noise.sample(&mut rng)).collect();
    let headers = ["x", "y"].into_iter().map(String::from).collect();
    let rows: Vec<StringRecord> = x
        .iter()
        .zip(y.iter())
        .map(|(a, b)| StringRecord::from(vec![a.to_string(), b.to_string()]))
        .collect();
    encode_recordswith_inferred_schema(headers, rows).expect("encode")
}

#[test]
fn zz_probe_2644_matern_low_n_route_agreement() {
    gam_solve::progress_log::init_logging_at(log::LevelFilter::Info);
    let ds = mk_1d(15, |t| t.powi(2), 0.05, 7);
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let started = std::time::Instant::now();
    let outcome = fit_from_formula("y ~ matern(x, nu=5/2)", &ds, &cfg);
    println!(
        "[probe-2644-lown] elapsed={:.2}s",
        started.elapsed().as_secs_f64()
    );
    match outcome {
        Ok(_) => println!("[probe-2644-lown] FIT OK"),
        Err(e) => println!("[probe-2644-lown] FIT ERR: {e}"),
    }
}

// ─────────────────────────────────────────────────────────────────────────
// A REFUTED hypothesis, kept as a passing guard because the property it
// checks is real and worth protecting.
//
// #2644's decomposition showed that the outer criterion notices `beta[0]`
// (the intercept) moving ONLY through `FIXED_STABILIZATION_RIDGE = 1e-8`:
//
//     ridge*(b0'^2 - b0^2) = 7.7024058e-10  vs measured dDp = 7.7024051e-10
//                                           (ratio 1.00000009)
//
// I read that as a SPEC.md line 12 violation — "an intercept generally should
// not have a penalty" — and predicted the observable harm: the criterion should
// not be invariant under a shift in the ORIGIN of `y`. Adding a constant to the
// response shifts the intercept by that constant and leaves the residuals,
// `S_lambda`, `X'WX` and `log|H|` untouched, so if the ridge were charged
// against a FIXED target the criterion would have to move by roughly
// `(n/2)*delta*((b0+c)^2 - b0^2)/(rss+pen)`.
//
// MEASURED (2026-07-31), and the prediction is WRONG by nine orders:
//
//     v(y)    = -9.06920633442079271e0
//     v(y+10) = -9.06920633442075186e0
//     gap     =  4.085621e-14   relative = 4.504937e-15   (~20 machine eps)
//     predicted, had the ridge broken this:  5.092357e-05
//
// The reason: the ridge is **Tikhonov centered on a moving `prior_mean_target`,
// not on zero**. `pls_solver` augments the RHS with `r + delta*mu`, so
// `penalty_term` carries `delta*||beta - mu||^2`; under an origin shift `mu`
// moves with `beta` and the term is unchanged. The `delta*||beta||^2` shorthand
// in `FIXED_STABILIZATION_RIDGE`'s own doc is what misled me — check the RHS
// augmentation in `pls_solver`, not the doc comment.
//
// CONSEQUENCE, and it is the useful part: **the ridge is not the defect, it is
// the messenger.** In #2671 the two routes move `beta[0]` by 0.213 relative to a
// FIXED `mu`, so the ridge does charge it — which is how the disagreement
// becomes visible at all. Exempting the intercept would make that witness go
// green by DELETING the only term that can see two routes computing different
// coefficients, while they went on producing different fitted models. That is
// strictly worse than the current loud refusal, and it would have looked like a
// successful fix. Do not do it on #2644's evidence.
//
// This test PASSES on main and is a guard, not a probe: a future change that
// makes the criterion depend on the origin of `y` is a real defect and this
// catches it directly. The bound is deliberately loose (1e-10 relative, ~5e4x
// the measured 4.5e-15) so it fires on a mechanism, never on roundoff.
// ─────────────────────────────────────────────────────────────────────────

#[test]
fn outer_criterion_is_invariant_to_the_origin_of_y() {
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let base = mk_1d(15, |t| t.powi(2), 0.05, 7);
    let mut shifted = base.clone();
    // Column 1 is `y` (headers are ["x", "y"] in `mk_1d`).
    const SHIFT: f64 = 10.0;
    for r in 0..shifted.values.nrows() {
        shifted.values[[r, 1]] += SHIFT;
    }

    let score = |ds: &EncodedDataset, label: &str| -> f64 {
        match fit_from_formula("y ~ s(x, k=5)", ds, &cfg) {
            Ok(gam::FitResult::Standard(fit)) => fit
                .fit
                .reml_score()
                .unwrap_or_else(|| panic!("{label}: fit reports no REML criterion")),
            Ok(_) => panic!("{label}: expected a standard fit"),
            Err(e) => panic!("{label}: fit failed: {e}"),
        }
    };

    let v0 = score(&base, "y");
    let v1 = score(&shifted, "y + 10");
    let relative = (v1 - v0).abs() / v0.abs().max(f64::MIN_POSITIVE);
    println!(
        "[2644-origin] v(y)={v0:.17e} v(y+{SHIFT})={v1:.17e} gap={:.6e} relative={relative:.6e}",
        v1 - v0,
    );
    assert!(
        relative < 1.0e-10,
        "the outer REML criterion must be a function of the MODEL, and adding a constant to \
         `y` changes no part of the model: the intercept absorbs it while residuals, S_lambda, \
         X'WX and log|H| are untouched. Measured relative change {relative:.6e} against a \
         1e-10 bar (main measures 4.5e-15, ~20 machine epsilons). A failure here means some \
         term is charging the intercept against a FIXED target -- see #2644, where exactly \
         that term, `FIXED_STABILIZATION_RIDGE`, accounts for 100% of a cross-route criterion \
         disagreement (ratio 1.00000009)."
    );
}
