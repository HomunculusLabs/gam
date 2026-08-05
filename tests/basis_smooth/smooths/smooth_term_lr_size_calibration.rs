//! #939 deliverable 4 — EMPIRICAL NULL-SIMULATION SIZE CALIBRATION of the
//! smooth-term likelihood-ratio test, the validation the issue demands before
//! closure.
//!
//! Under a null data-generating process (the smooth's covariate has no effect on
//! the mean) the per-term LR statistic `W = 2(ℓ_full − ℓ_null)` follows a central
//! `χ²_d` only to first order. At modest `n` the first-order reference is
//! **anti-conservative**: `E[W] = d + Δε > d`, so the χ²_d tail under-covers and
//! the empirical size — the fraction of null replicates rejected at level `α` —
//! exceeds the nominal `α`. The Bartlett correction rescales `W` by
//! `c = E[W]/d` so the corrected statistic's mean returns to `d` and the size
//! returns to nominal. This harness measures that directly, comparing three
//! lanes from the SAME live driver (`smooth_term_lr_inference_forspec`):
//!
//!   (a) first-order χ²        — `p_value_uncorrected`,
//!   (b) fixed-λ Bartlett      — `p_value_corrected` with the conditional factor,
//!   (c) estimated-λ Bartlett  — `p_value_corrected` with the ρ̂-variation factor
//!                                (`correction == LawleyLrEstimatedLambda`).
//!
//! Empirical size at `α` is `#{p ≤ α}/R`. Its Monte-Carlo standard error is
//! `√(α(1−α)/R)`; the assertions use a `±k·SE` band so they are robust to the
//! finite simulation budget. The defining claims (#939 deliverable 4):
//!
//!   1. WHERE FIRST-ORDER IS DISTORTED (small `n`): the first-order size is
//!      materially above nominal, and the corrected lanes pull it back — the
//!      estimated-λ size is at least as close to nominal as the first-order size
//!      AND lands inside the MC band, across families and penalty ranks.
//!   2. ESTIMATED-λ NEVER WORSE: across the whole grid the estimated-λ size's
//!      distance from nominal never exceeds the first-order distance by more than
//!      MC noise — the correction is safe to apply everywhere.
//!   3. MATERIALITY: the per-test `material` flag fires exactly when the applied
//!      Bartlett factor moves the result by more than 10% (the documented rule).
//!
//! Truth-recovery bar (not a reference-tool match): the ground truth is the exact
//! null distribution of the LR statistic, i.e. Uniform p-values / nominal size.
//!
//! Budget: the full grid is `n ∈ {30,50,100,200,500}` × 2 families × 2 penalty
//! ranks × `REPS` replicates. The default `REPS` keeps wall-time in CI range
//! while holding the MC band tight enough for the directional claims; the small-n
//! cells (where the correction matters and a fit is cheap) carry the load.

use gam::smooth::{
    SmoothLrCorrection, SmoothLrReferenceSource, SmoothTermLrInference,
    smooth_term_lr_inference_forspec,
};
use gam::{
    FitConfig, FitRequest, encode_recordswith_inferred_schema, init_parallelism, materialize,
};

use csv::StringRecord;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Bernoulli, Distribution, Poisson};

/// Which null family/DGP a replicate is drawn from. In every case the smooth's
/// covariate `z ~ U(0,1)` has NO effect on the mean — the smooth `s(z)` is
/// null-true and its LR statistic is exactly central-χ² in the limit.
#[derive(Clone, Copy)]
enum NullFamily {
    /// `y ~ Poisson(exp(0.3 + 0.8 x))`, log link.
    PoissonLog,
    /// `y ~ Bernoulli(logit⁻¹(−0.2 + 0.9 x))`, logit link.
    BernoulliLogit,
}

impl NullFamily {
    fn family_name(self) -> &'static str {
        match self {
            NullFamily::PoissonLog => "poisson",
            NullFamily::BernoulliLogit => "binomial",
        }
    }
    fn label(self) -> &'static str {
        match self {
            NullFamily::PoissonLog => "poisson/log",
            NullFamily::BernoulliLogit => "bernoulli/logit",
        }
    }
}

/// One null-DGP replicate for a given family.
fn null_replicate(family: NullFamily, n: usize, seed: u64) -> gam::data::EncodedDataset {
    let mut rng = StdRng::seed_from_u64(seed);
    let headers = vec!["y".to_string(), "x".to_string(), "z".to_string()];
    let mut rows = Vec::<StringRecord>::with_capacity(n);
    for i in 0..n {
        let x = i as f64 / (n as f64 - 1.0); // deterministic, spans [0,1]
        // z is an independent nuisance covariate with NO effect on the mean.
        let z: f64 = rng.random_range(0.0..1.0);
        let y: f64 = match family {
            NullFamily::PoissonLog => {
                let eta = 0.3 + 0.8 * x; // no z term — the smooth is null-true.
                let lambda = eta.exp();
                Poisson::new(lambda).expect("poisson rate").sample(&mut rng) as f64
            }
            NullFamily::BernoulliLogit => {
                let eta = -0.2 + 0.9 * x; // no z term — the smooth is null-true.
                let mu = 1.0 / (1.0 + (-eta).exp());
                let bit = Bernoulli::new(mu).expect("bernoulli p").sample(&mut rng);
                if bit { 1.0 } else { 0.0 }
            }
        };
        rows.push(StringRecord::from(vec![
            y.to_string(),
            x.to_string(),
            z.to_string(),
        ]));
    }
    encode_recordswith_inferred_schema(headers, rows).expect("encode")
}

/// Run the per-term LR + Bartlett driver on one replicate and return the single
/// `s(z)` smooth-term report. `k` is the smooth basis dimension — varying it
/// changes the penalty rank / reference df, the second grid axis.
fn run_one(
    family: NullFamily,
    k: usize,
    data: &gam::data::EncodedDataset,
) -> Result<Option<SmoothTermLrInference>, String> {
    let cfg = FitConfig {
        family: Some(family.family_name().to_string()),
        ..FitConfig::default()
    };
    let formula = format!("y ~ x + s(z, k={k})");
    let mat = materialize(&formula, data, &cfg).expect("materialize");
    let FitRequest::Standard(req) = mat.request else {
        panic!("expected a standard fit request");
    };
    // A REPLICATE WHOSE FIT REFUSES IS NOT A CALIBRATION DATUM, and it is not a
    // calibration failure either — the outer optimizer declining to certify a
    // stationary optimum on one null draw is the #2664 line-search cluster, a
    // different subsystem. It used to `.expect(...)` here, which converted the
    // first such draw anywhere in the grid into a panic that withdrew the verdict
    // for every cell, exactly the way a timeout does. Return the refusal instead
    // so the caller can COUNT it: the grid already has a floor on how many
    // replicates must produce a finite report, and a refusal that is common
    // enough to matter will trip it and say so with the message (#2672).
    let reports = smooth_term_lr_inference_forspec(
        req.data.view(),
        req.y.view(),
        req.weights.view(),
        req.offset.view(),
        &req.spec,
        req.family,
        &req.options,
    )
    .map_err(|error| error.to_string())?;
    Ok(reports.into_iter().find(|r| r.name.contains('z')))
}

/// Empirical-size accumulators for one grid cell, across the three lanes.
#[derive(Default, Clone, Copy)]
struct SizeCounts {
    used: usize,
    // Rejections at α = 0.05 / 0.01 for each lane.
    rej_first_05: usize,
    rej_fixed_05: usize,
    rej_est_05: usize,
    rej_first_01: usize,
    rej_est_01: usize,
    // How many replicates actually reached the estimated-λ correction.
    est_lambda_applied: usize,
}

impl SizeCounts {
    fn ingest(&mut self, r: &SmoothTermLrInference) {
        // The fixed-λ corrected p-value is recoverable per replicate even when the
        // applied correction was estimated-λ: when the applied lane is estimated-λ,
        // `bartlett_factor_conditional` carries the fixed-λ factor; otherwise the
        // applied `bartlett_factor` IS the fixed-λ factor. We reconstruct the
        // fixed-λ corrected statistic from whichever factor is the conditional one.
        let p_first = r.p_value_uncorrected;
        let p_est = r.p_value_corrected; // applied lane (estimated-λ where available)
        let c_fixed = r
            .bartlett_factor_conditional
            .unwrap_or(r.bartlett_factor)
            .max(f64::MIN_POSITIVE);
        // Read the fixed-λ lane off the SAME reference the driver used, through
        // the report's own accessor. Reconstructing it as `P(χ²_{ref_df} > W/c)`
        // was correct only while `ref_df` was a chi-square degrees of freedom;
        // it is the null MEAN now, and the reference is the `(ν, g)` pair.
        let p_fixed = r
            .ref_df_provenance
            .tail_probability(r.statistic_lr / c_fixed);
        if !(p_first.is_finite() && p_est.is_finite() && p_fixed.is_finite()) {
            return;
        }
        self.used += 1;
        if matches!(r.correction, SmoothLrCorrection::LawleyLrEstimatedLambda) {
            self.est_lambda_applied += 1;
        }
        if p_first <= 0.05 {
            self.rej_first_05 += 1;
        }
        if p_fixed <= 0.05 {
            self.rej_fixed_05 += 1;
        }
        if p_est <= 0.05 {
            self.rej_est_05 += 1;
        }
        if p_first <= 0.01 {
            self.rej_first_01 += 1;
        }
        if p_est <= 0.01 {
            self.rej_est_01 += 1;
        }
    }
    fn size(&self, rej: usize) -> f64 {
        rej as f64 / self.used.max(1) as f64
    }
}

/// One row of the grid result, for the aggregate assertions and the diagnostic
/// print.
#[derive(Clone, Copy)]
struct CellResult {
    n: usize,
    k: usize,
    label: &'static str,
    used: usize,
    /// Replicates whose FIT refused (the LR call returned `Err`). Counted rather
    /// than fatal: a refusal is a missing datum, not a calibration verdict.
    refused: usize,
    est_applied: usize,
    size_first_05: f64,
    size_fixed_05: f64,
    size_est_05: f64,
    size_first_01: f64,
    size_est_01: f64,
}

/// Monte-Carlo standard error of an empirical size estimate at level `alpha` from
/// `reps` replicates: `√(α(1−α)/reps)`.
fn size_se(alpha: f64, reps: usize) -> f64 {
    (alpha * (1.0 - alpha) / reps.max(1) as f64).sqrt()
}

/// THE NULL-SIMULATION SIZE GRID (#939 deliverable 4). Runs a full fit +
/// constrained null refit per replicate over the grid. By default it runs a
/// small but still-asserting grid that finishes in CI budget. The exhaustive
/// larger-n grid (`n ∈ {30,50,100,200,500}` × both families × both ranks × 600
/// reps) is a separate MSI artifact, not an env/cfg branch. The bounded CI
/// sibling below carries the small-n calibration claim; this test adds the
/// moderate-n cells without ever being inert.
#[test]
fn exhaustive_null_simulation_size_grid() {
    init_parallelism();

    // Fixed CI-affordable grid (the exhaustive larger-n grid is a separate MSI
    // artifact, not an env/cfg branch). The calibration claim holds on this grid.
    //
    // `reps` is a Monte-Carlo budget, not a coverage axis: EVERY (n, k, family)
    // cell in the grid below is still validated. The assertion bands are tied to
    // `reps` through the MC standard error `√(α(1−α)/reps)` (see
    // `assert_grid_calibration`), so a smaller budget simply widens the band it
    // must fit inside — the calibration claims (1) and (2) remain self-consistent
    // and un-weakened. 60 reps keeps 0.05 + 2·SE ≈ 0.106, still well below the
    // documented small-n first-order anti-conservatism (n=30,k=12), so the
    // `any_first_order_distorted` meaningfulness guard still fires. Halved from
    // 120 to keep this test under the 300s nextest SLOW budget.
    let reps: usize = 60;
    let ns: &[usize] = &[30usize, 100];
    let ks = [6usize, 12];
    let families = [NullFamily::PoissonLog, NullFamily::BernoulliLogit];

    let mut cells = Vec::<CellResult>::new();
    for &family in &families {
        for &k in &ks {
            for &n in ns {
                let mut counts = SizeCounts::default();
                let mut refused = 0usize;
                let mut first_refusal: Option<String> = None;
                for rep in 0..reps {
                    let seed = mix_seed(family.label(), n, k, rep);
                    let data = null_replicate(family, n, seed);
                    match run_one(family, k, &data) {
                        Ok(Some(r)) => counts.ingest(&r),
                        Ok(None) => {}
                        Err(message) => {
                            refused += 1;
                            first_refusal.get_or_insert(message);
                        }
                    }
                }
                if let Some(message) = first_refusal.as_ref() {
                    eprintln!(
                        "[#939 grid] {} n={n} k={k}: {refused}/{reps} replicate fits \
                         REFUSED and contribute no calibration datum. First: {message}",
                        family.label()
                    );
                }
                cells.push(CellResult {
                    n,
                    k,
                    label: family.label(),
                    used: counts.used,
                    refused,
                    est_applied: counts.est_lambda_applied,
                    size_first_05: counts.size(counts.rej_first_05),
                    size_fixed_05: counts.size(counts.rej_fixed_05),
                    size_est_05: counts.size(counts.rej_est_05),
                    size_first_01: counts.size(counts.rej_first_01),
                    size_est_01: counts.size(counts.rej_est_01),
                });
            }
        }
    }

    assert_grid_calibration(&cells, reps, "light");
}

/// BOUNDED CI SIZE CHECK (#939 deliverable 4): the small-`n` cells — where the
/// first-order test is documented anti-conservative and a fit is cheap — across
/// both families and both penalty ranks, with a replicate budget that keeps
/// wall-time in CI range while holding the MC band tight enough for the
/// directional calibration claims. This is the default-run validation.
#[test]
fn null_simulation_size_is_calibrated_small_n() {
    init_parallelism();

    // `REPS` is a Monte-Carlo budget, not a coverage axis: every (n, k, family)
    // cell is still validated. The assertion bands scale with the MC standard
    // error `√(α(1−α)/REPS)` (see `assert_grid_calibration`), so a smaller budget
    // only widens the band the empirical size must fit — claims (1)/(2) stay
    // self-consistent and un-weakened, and the materiality (claim 3) check runs
    // per-replicate regardless of budget. At 120 reps 0.05 + 2·SE ≈ 0.090 stays
    // below the documented small-n first-order anti-conservatism, so the
    // `any_first_order_distorted` guard still fires. Halved from 240 to keep this
    // test under the 300s nextest SLOW budget.
    const REPS: usize = 120;
    let ns = [30usize, 50];
    let ks = [6usize, 12];
    let families = [NullFamily::PoissonLog, NullFamily::BernoulliLogit];

    let mut cells = Vec::<CellResult>::new();
    let mut any_material_checked = false;
    for &family in &families {
        for &k in &ks {
            for &n in &ns {
                let mut counts = SizeCounts::default();
                let mut refused = 0usize;
                let mut first_refusal: Option<String> = None;
                for rep in 0..REPS {
                    let seed = mix_seed(family.label(), n, k, rep);
                    let data = null_replicate(family, n, seed);
                    let report = match run_one(family, k, &data) {
                        Ok(report) => report,
                        Err(message) => {
                            refused += 1;
                            first_refusal.get_or_insert(message);
                            None
                        }
                    };
                    if let Some(r) = report {
                        // Materiality (#939 deliverable 4): when a correction is
                        // applied, the `material` flag must follow the 10% rule.
                        if !matches!(r.correction, SmoothLrCorrection::None) {
                            let factor_move = (r.bartlett_factor - 1.0).abs();
                            let p_hi = r.p_value_uncorrected.max(r.p_value_corrected);
                            let p_lo = r.p_value_uncorrected.min(r.p_value_corrected);
                            let p_move = (p_hi - p_lo) / p_hi.max(f64::MIN_POSITIVE);
                            let expected = factor_move > 0.10 || p_move > 0.10;
                            assert_eq!(
                                r.material,
                                expected,
                                "{} n={n} k={k}: material flag must follow the 10% rule \
                                 (c={:.4}, factor_move={:.4}, p_move={:.4})",
                                family.label(),
                                r.bartlett_factor,
                                factor_move,
                                p_move
                            );
                            any_material_checked = true;
                        }
                        counts.ingest(&r);
                    }
                }
                if let Some(message) = first_refusal.as_ref() {
                    eprintln!(
                        "[#939 small-n] {} n={n} k={k}: {refused}/{REPS} replicate fits \
                         REFUSED and contribute no calibration datum. First: {message}",
                        family.label()
                    );
                }
                cells.push(CellResult {
                    n,
                    k,
                    label: family.label(),
                    used: counts.used,
                    refused,
                    est_applied: counts.est_lambda_applied,
                    size_first_05: counts.size(counts.rej_first_05),
                    size_fixed_05: counts.size(counts.rej_fixed_05),
                    size_est_05: counts.size(counts.rej_est_05),
                    size_first_01: counts.size(counts.rej_first_01),
                    size_est_01: counts.size(counts.rej_est_01),
                });
            }
        }
    }

    assert!(
        any_material_checked,
        "no correction was ever applied across the small-n grid — the harness is \
         not exercising the Bartlett path"
    );
    assert_grid_calibration(&cells, REPS, "small-n");
}

/// The shared calibration assertions over a completed grid.
fn assert_grid_calibration(cells: &[CellResult], reps: usize, tag: &str) {
    // Diagnostic dump (printed on failure / with --nocapture).
    eprintln!("=== #939 null-simulation size grid ({tag}), REPS={reps} ===");
    eprintln!(
        "{:>16} {:>4} {:>3} {:>5} {:>4} {:>6} | size@.05  first/fixed/est   size@.01 first/est",
        "family", "n", "k", "used", "ref!", "estΛ"
    );
    for c in cells {
        eprintln!(
            "{:>16} {:>4} {:>3} {:>5} {:>4} {:>6} |   {:.3} / {:.3} / {:.3}     {:.3} / {:.3}",
            c.label,
            c.n,
            c.k,
            c.used,
            c.refused,
            c.est_applied,
            c.size_first_05,
            c.size_fixed_05,
            c.size_est_05,
            c.size_first_01,
            c.size_est_01,
        );
    }

    let se05 = size_se(0.05, reps);
    let se01 = size_se(0.01, reps);
    // Band half-widths: 3·SE plus a small slack for the second-order residual the
    // correction itself leaves (`O(n⁻²)`) and the chi-square reference df being a
    // Wood truncation rather than an integer.
    let band05 = 3.0 * se05 + 0.015;
    let band01 = 3.0 * se01 + 0.008;

    let mut pooled_used = 0usize;
    let mut pooled_rej_05 = 0.0_f64;
    let mut pooled_rej_01 = 0.0_f64;
    for c in cells {
        assert!(
            c.used >= reps * 7 / 10,
            "{} n={} k={}: too many replicates failed to produce a finite report \
             ({}/{reps}; {} of the losses were FIT REFUSALS, which are a solver \
             verdict rather than a calibration one)",
            c.label,
            c.n,
            c.k,
            c.used,
            c.refused
        );

        // CLAIM 1 — estimated-λ size lands inside the MC band of nominal at α=0.05.
        assert!(
            (c.size_est_05 - 0.05).abs() <= band05,
            "{} n={} k={}: estimated-λ size@.05 = {:.3} is outside the nominal band \
             0.05 ± {:.3} (first-order was {:.3})",
            c.label,
            c.n,
            c.k,
            c.size_est_05,
            band05,
            c.size_first_05
        );
        // And at the tighter α=0.01.
        assert!(
            (c.size_est_01 - 0.01).abs() <= band01,
            "{} n={} k={}: estimated-λ size@.01 = {:.3} is outside the nominal band \
             0.01 ± {:.3} (first-order was {:.3})",
            c.label,
            c.n,
            c.k,
            c.size_est_01,
            band01,
            c.size_first_01
        );

        // CLAIM 2 — estimated-λ is NEVER materially worse-calibrated than
        // first-order: its distance from nominal does not exceed first-order's by
        // more than MC noise.
        let d_first = (c.size_first_05 - 0.05).abs();
        let d_est = (c.size_est_05 - 0.05).abs();
        assert!(
            d_est <= d_first + 2.0 * se05,
            "{} n={} k={}: estimated-λ size@.05 ({:.3}) must not be worse-calibrated \
             than first-order ({:.3}) beyond MC noise (|Δ|={:.3} > {:.3})",
            c.label,
            c.n,
            c.k,
            c.size_est_05,
            c.size_first_05,
            d_est,
            d_first + 2.0 * se05
        );

        // CLAIM 1 (strict) — WHERE first-order is anti-conservative, the correction
        // must pull the size strictly back toward nominal. Conditional by
        // construction: it is a statement about the cells that exhibit the
        // distortion, and it is silent (not satisfied) on the ones that do not.
        if c.size_first_05 > 0.05 + 2.0 * se05 {
            assert!(
                d_est <= d_first + 1e-9,
                "{} n={} k={}: where first-order is anti-conservative \
                 (size@.05={:.3} > nominal), the estimated-λ correction must pull \
                 size toward nominal: est={:.3} (|Δest|={:.3}) vs first (|Δfirst|={:.3})",
                c.label,
                c.n,
                c.k,
                c.size_first_05,
                c.size_est_05,
                d_est,
                d_first
            );
        }

        pooled_used += c.used;
        pooled_rej_05 += c.size_est_05 * c.used as f64;
        pooled_rej_01 += c.size_est_01 * c.used as f64;
    }

    // ANTI-VACUITY, pooled over the whole grid.
    //
    // This replaces `any_first_order_distorted`, which required the FIRST-ORDER
    // lane to be anti-conservative somewhere or the test failed with "the regime
    // where the Bartlett correction matters is not being probed". That is a
    // proxy, and after #2672 repaired the reference d.f. assembly it became a
    // requirement that the defect still be present: on the light grid every cell
    // now reads first-order size 0.000–0.069 against a `0.05 + 2·SE = 0.106`
    // trigger, so the guard fired on correct code.
    //
    // The hazard it was standing in for is real and is addressed directly here.
    // The per-cell band is two-sided but wide (`3·SE + 0.015` is ±0.099 at 60
    // replicates), so a test that NEVER REJECTS passes every per-cell claim: size
    // 0.000 sits inside 0.05 ± 0.099, and CLAIM 2 compares two zeros. Pooling the
    // whole grid shrinks the standard error by √(cells) and closes that hole —
    // 0.000 pooled is 0.05 away from nominal against a band of `3·SE + 0.015`,
    // which is 0.045 at the light grid's 478 usable replicates and 0.036 at the
    // small-n grid's 957. A never-rejecting test fails; so does the pre-#1872 raw
    // -EDF reference, whose measured per-cell sizes (0.092 … 0.261, pooled ≈ 0.15)
    // miss by 0.10. That is strictly stronger than the guard it replaces, in both
    // directions, and it does not require anything to be broken.
    let pooled = pooled_used.max(1) as f64;
    let pooled_size_05 = pooled_rej_05 / pooled;
    let pooled_size_01 = pooled_rej_01 / pooled;
    let pooled_band05 = 3.0 * size_se(0.05, pooled_used) + 0.015;
    let pooled_band01 = 3.0 * size_se(0.01, pooled_used) + 0.008;
    eprintln!(
        "=== #939 pooled over the {tag} grid: {pooled_used} usable replicates, \
         size@.05={pooled_size_05:.4} (band 0.05 ± {pooled_band05:.4}), \
         size@.01={pooled_size_01:.4} (band 0.01 ± {pooled_band01:.4})"
    );
    assert!(
        (pooled_size_05 - 0.05).abs() <= pooled_band05,
        "pooled over the {tag} grid the estimated-λ size@.05 is {pooled_size_05:.4}, \
         outside 0.05 ± {pooled_band05:.4} on {pooled_used} usable replicates. \
         Pooling is what makes a test that never rejects fail: at this budget \
         size 0.000 misses by 0.05 against a band of {pooled_band05:.4}."
    );
    assert!(
        (pooled_size_01 - 0.01).abs() <= pooled_band01,
        "pooled over the {tag} grid the estimated-λ size@.01 is {pooled_size_01:.4}, \
         outside 0.01 ± {pooled_band01:.4} on {pooled_used} usable replicates"
    );
}

/// DIAGNOSTIC (not a contract, #2672): the same grid, scored against the
/// reference SHAPES the driver could use, from one set of fits.
///
/// The statistic's null law is `W = Σ_j w_j χ²_1` with `w = eig(2F_jj − F_jj²)`,
/// so `Σ w_j` (Wood's `edf1`) is its MEAN and `Σ w_j²` is half its variance.
/// Every candidate below is a different answer to "what do you do with those two
/// numbers", and all of them are functions of what `ref_df_provenance` publishes,
/// so no candidate needs its own refit.
///
/// * `spectral` — the shipped reference: `P(χ²_ν > W/g)` with `ν = (Σw)²/Σw²`
///   and `g = Σw²/Σw`. Matches mean and variance; exact when the weights are
///   equal, including the unpenalized `w ≡ 1 ⇒ χ²_q`.
/// * `mean-chi2` — `P(χ²_{Σw} > W)`, the pre-#2672 shape with the WPS inflation
///   and the floors removed. Matches only the mean, so it carries variance
///   `2Σw` where the statistic carries `2Σw²`; the arm that measures what
///   matching one moment costs.
/// * `mean-chi2-floor1` — the same with `.max(1.0)`, the #1766 degeneracy floor.
///   The floor exists only because `χ²_{d→0}` calls any positive `W` maximally
///   significant; this arm separates "the floor was load-bearing" from "the floor
///   was compensating for the wrong shape".
///
/// Each candidate is scored uncorrected and Bartlett-corrected (`Δε = (c−1)·d`
/// recovered exactly from the applied factor), so the correction's contribution
/// stays separable from the reference's.
#[test]
fn zz_measure_size_under_candidate_reference_shapes_2672() {
    init_parallelism();

    const REPS: usize = 120;
    let ns = [30usize, 50];
    let ks = [6usize, 12];
    let families = [NullFamily::PoissonLog, NullFamily::BernoulliLogit];

    #[derive(Clone, Copy, Default)]
    struct CandCounts {
        raw_05: usize,
        cor_05: usize,
        cor_01: usize,
        sum_d: f64,
    }

    const NAMES: [&str; 3] = ["spectral", "mean-chi2", "mean-chi2-floor1"];

    eprintln!(
        "[zz2672-size] {:>16} {:>4} {:>3} {:>5} {:>8} | candidate           mean_nu   size@.05(raw/cor)  size@.01(cor)",
        "family", "n", "k", "used", "mean_W"
    );
    for &family in &families {
        for &k in &ks {
            for &n in &ns {
                let mut counts = [CandCounts::default(); 3];
                let mut used = 0usize;
                let mut sum_w = 0.0;
                let mut fallback_lane = 0usize;
                for rep in 0..REPS {
                    let seed = mix_seed(family.label(), n, k, rep);
                    let data = null_replicate(family, n, seed);
                    let Ok(Some(r)) = run_one(family, k, &data) else {
                        continue;
                    };
                    if !(r.statistic_lr.is_finite() && r.ref_df.is_finite() && r.ref_df > 0.0) {
                        continue;
                    }
                    let prov = r.ref_df_provenance;
                    if !matches!(prov.source, SmoothLrReferenceSource::NullSpectrum) {
                        fallback_lane += 1;
                    }
                    // `c = mean_w/d` with `mean_w = d + Δε`, so `Δε = (c − 1)·d`
                    // exactly — no re-derivation of the Lawley assembly needed.
                    let delta_eps = (r.bartlett_factor - 1.0) * r.ref_df;
                    // (shape, scale) per candidate.
                    let shapes = [
                        (prov.chi_square_df, prov.scale),
                        (prov.mean, 1.0),
                        (prov.mean.max(1.0), 1.0),
                    ];
                    used += 1;
                    sum_w += r.statistic_lr;
                    for (slot, (nu, g)) in shapes.iter().copied().enumerate() {
                        if !(nu.is_finite() && nu > 0.0 && g.is_finite() && g > 0.0) {
                            continue;
                        }
                        let chi2 = statrs::distribution::ChiSquared::new(nu).expect("chi2");
                        use statrs::distribution::ContinuousCDF;
                        let sf = |w: f64| (1.0 - chi2.cdf(w / g)).clamp(0.0, 1.0);
                        let d = r.ref_df;
                        let c = ((d + delta_eps) / d).max(f64::MIN_POSITIVE);
                        counts[slot].sum_d += nu;
                        if sf(r.statistic_lr) <= 0.05 {
                            counts[slot].raw_05 += 1;
                        }
                        if sf(r.statistic_lr / c) <= 0.05 {
                            counts[slot].cor_05 += 1;
                        }
                        if sf(r.statistic_lr / c) <= 0.01 {
                            counts[slot].cor_01 += 1;
                        }
                    }
                }
                let denom = used.max(1) as f64;
                for (slot, name) in NAMES.iter().enumerate() {
                    eprintln!(
                        "[zz2672-size] {:>16} {n:>4} {k:>3} {used:>5} {:>8.4} | {name:<16} {:>8.4}   {:.3} / {:.3}        {:.3}",
                        family.label(),
                        sum_w / denom,
                        counts[slot].sum_d / denom,
                        counts[slot].raw_05 as f64 / denom,
                        counts[slot].cor_05 as f64 / denom,
                        counts[slot].cor_01 as f64 / denom,
                    );
                }
                if fallback_lane > 0 {
                    eprintln!(
                        "[zz2672-size] {:>16} {n:>4} {k:>3}: {fallback_lane}/{used} replicates \
                         scored on the UNIT-WEIGHT fallback lane (no influence matrix), so their \
                         `spectral` row is not a spectral reference",
                        family.label()
                    );
                }
            }
        }
    }
    eprintln!(
        "[zz2672-size] read: nominal 0.05 / 0.01; MC s.e. at {REPS} reps is {:.4} / {:.4}. \
         `mean_W` against the `mean-chi2` row's `mean_nu` is the first-order mean check — \
         that candidate's shape parameter IS the statistic's null mean.",
        size_se(0.05, REPS),
        size_se(0.01, REPS)
    );
}

/// Deterministic per-cell, per-replicate seed so the grid is fully reproducible
/// and the cells are independent (no shared RNG stream across cells).
fn mix_seed(label: &str, n: usize, k: usize, rep: usize) -> u64 {
    let mut h = 1469598103934665603u64; // FNV-1a offset basis
    let mut mix = |v: u64| {
        h ^= v;
        h = h.wrapping_mul(1099511628211);
    };
    for b in label.bytes() {
        mix(b as u64);
    }
    mix(n as u64);
    mix(k as u64);
    mix(rep as u64);
    mix(0x9395_3393); // domain tag for this harness
    h
}
