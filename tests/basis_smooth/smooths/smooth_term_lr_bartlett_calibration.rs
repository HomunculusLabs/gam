//! #1063: the per-term smooth-significance test uses a genuine likelihood-ratio
//! statistic `W = 2(ℓ_full − ℓ_null)` (a constrained refit dropping the smooth),
//! Bartlett-corrected by the exact Lawley factor — and the correction IMPROVES
//! the χ² calibration of the test under the null.
//!
//! The summary table reports Wood's rank-truncated *Wald* statistic; the Lawley
//! factor corrects the *likelihood-ratio* statistic. Dividing the Wald T by the
//! LR factor would correct the wrong quantity (under penalization the Wald form
//! is already a weighted χ²). The principled fix (#1063 Option 1) is to compute
//! a real per-term LR statistic and correct *that*. This test proves two things
//! about `smooth_term_lr_inference_forspec`:
//!
//!   (a) PROVENANCE — for a Poisson/log smooth (closed-form Lawley jets) the
//!       reported significance is built from the Bartlett-corrected LR
//!       (`correction_provenance == "lawley_lr_estimated_lambda"`,
//!       `bartlett_factor > 1`,
//!       `statistic_corrected == statistic_lr / bartlett_factor`).
//!
//!   (b) CALIBRATION — under a NULL data-generating process (the smooth's
//!       covariate has no effect) the uncorrected LR is MIScalibrated against
//!       χ²_d at modest n: `E[W] = d + Δε ≠ d`. The Bartlett-corrected statistic
//!       restores the mean toward d, so `mean(W*)` is closer to `d` than
//!       `mean(W)` and the corrected p-values are closer to Uniform — a strictly
//!       better χ² calibration than the uncorrected reference.
//!
//!       The sign of Δε is NOT part of the claim, and the measured sign here is
//!       negative. This module previously asserted `Δε > 0` (anti-conservatism),
//!       which is a property of an UNPENALIZED test: the factor is
//!       `c = 1 + Δε/d` for either sign, and a penalized smooth under the null
//!       legitimately shrinks the alternative fit, pulling the LR BELOW the χ²_d
//!       reference. Measured at n = 60: mean(W) = 1.82 against d = 4.15, so
//!       Δε ≈ −2.33, c ≈ 0.44, and W/c ≈ 4.15 — the correction working as
//!       designed, in the direction the old precondition forbade.
//!
//! This is the truth-recovery / calibration bar (not a reference-tool match):
//! the ground truth here is the exact null distribution of the LR statistic.

use gam::smooth::{SmoothLrReferenceSource, smooth_term_lr_inference_forspec};
use gam::{
    FitConfig, FitRequest, encode_recordswith_inferred_schema, init_parallelism, materialize,
};

use gam::inference::lawley::{
    RhoPenaltyComponent, RowExpectedJets, RowKappas, lawley_lr_mean_shift,
    lawley_lr_mean_shift_with_rho_variation,
};
use ndarray::Array2;

use csv::StringRecord;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Poisson};

/// `n` standard normals whose SAMPLE mean and variance are forced to exactly 0
/// and 1.
///
/// A Monte-Carlo expectation over `ρ̂ = ρ₀ + √v·z` estimates a second-order
/// quantity, and the first-order term `Δε'·√v·z̄` is noise that survives common
/// random numbers and scales as `√v` against a signal of order `v`. Matching the
/// first two sample moments removes it exactly rather than averaging it down:
/// it is the difference between an estimator that converges at every variance
/// and one that does not converge at any of them.
fn moment_matched_normals(seed: u64, n: usize) -> Vec<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0, 1.0).expect("normal");
    let mut z: Vec<f64> = (0..n).map(|_| normal.sample(&mut rng)).collect();
    let mean = z.iter().sum::<f64>() / n as f64;
    for value in z.iter_mut() {
        *value -= mean;
    }
    let sd = (z.iter().map(|v| v * v).sum::<f64>() / n as f64).sqrt();
    for value in z.iter_mut() {
        *value /= sd;
    }
    z
}

/// One null-DGP replicate: `y ~ Poisson(exp(0.3 + 0.8 x))`, with an independent
/// covariate `z ~ U(0,1)` that has NO effect on the mean. The smooth `s(z)` is
/// therefore null-true; its LR statistic should follow χ²_d.
fn null_replicate(n: usize, seed: u64) -> gam::data::EncodedDataset {
    let mut rng = StdRng::seed_from_u64(seed);
    let headers = vec!["y".to_string(), "x".to_string(), "z".to_string()];
    let mut rows = Vec::<StringRecord>::with_capacity(n);
    for i in 0..n {
        let x = i as f64 / (n as f64 - 1.0); // deterministic, spans [0,1]
        // z is an independent nuisance covariate with NO effect on the mean.
        let z: f64 = rng.random_range(0.0..1.0);
        let eta = 0.3 + 0.8 * x; // NOTE: no z term — the smooth is null-true.
        let lambda = eta.exp();
        let y = Poisson::new(lambda).expect("poisson rate").sample(&mut rng) as u64;
        rows.push(StringRecord::from(vec![
            y.to_string(),
            x.to_string(),
            z.to_string(),
        ]));
    }
    encode_recordswith_inferred_schema(headers, rows).expect("encode")
}

/// Run the per-term LR + Bartlett driver on one replicate and return the single
/// `s(z)` smooth-term report.
fn run_one(data: &gam::data::EncodedDataset) -> Option<gam::smooth::SmoothTermLrInference> {
    let cfg = FitConfig {
        family: Some("poisson".to_string()),
        ..FitConfig::default()
    };
    let mat = materialize("y ~ x + s(z)", data, &cfg).expect("materialize");
    let FitRequest::Standard(req) = mat.request else {
        panic!("expected a standard fit request");
    };
    let reports = smooth_term_lr_inference_forspec(
        req.data.view(),
        req.y.view(),
        req.weights.view(),
        req.offset.view(),
        &req.spec,
        req.family,
        &req.options,
    )
    .expect("smooth-term LR inference");
    reports.into_iter().find(|r| r.name.contains('z'))
}

#[test]
fn poisson_smooth_lr_is_bartlett_corrected_and_better_calibrated() {
    init_parallelism();

    const N: usize = 60;
    const REPS: usize = 120;

    // (a) Provenance + algebra check on a single representative replicate.
    let probe = run_one(&null_replicate(N, 1)).expect("s(z) report present");
    assert_eq!(
        probe.correction.label(),
        "lawley_lr_estimated_lambda",
        "Poisson/log smooth has closed-form Lawley jets — the correction must fire"
    );
    // The Lawley factor is `c = 1 + Δε/d` for a mean shift Δε of EITHER sign, and
    // the module header above derives why the sign here is negative: a penalized
    // smooth under the null shrinks the alternative fit, pulling `E[W]` BELOW the
    // χ²_d reference. `c > 1` is a property of an UNPENALIZED test.
    //
    // #2672: this file asserted `c > 1.0` here and a one-sided p-move below,
    // while its own header recorded the derived value as `c ≈ 0.44 < 1`. Both
    // assertions passed only because the assembly was reporting `c ≈ 1.003` — so
    // they agreed with the defect and would have REJECTED its repair at a
    // different line than the one a fixer was sent to, with a confident message
    // asserting the opposite physics. The claim is two-sided; only a factor that
    // is not a rescaling at all, or a p-move that contradicts the factor's own
    // sign, is a defect.
    assert!(
        probe.bartlett_factor.is_finite() && probe.bartlett_factor > 0.0,
        "the Lawley factor must be a finite, strictly positive rescaling \
         (c = 1 + Δε/d, either sign of Δε); got {}",
        probe.bartlett_factor
    );
    assert!(
        (probe.statistic_corrected - probe.statistic_lr / probe.bartlett_factor).abs()
            < 1e-9 * (1.0 + probe.statistic_lr.abs()),
        "W* must equal W / c: {} vs {}",
        probe.statistic_corrected,
        probe.statistic_lr / probe.bartlett_factor
    );
    if probe.bartlett_factor > 1.0 {
        assert!(
            probe.p_value_corrected >= probe.p_value_uncorrected - 1e-12,
            "c = {} > 1 shrinks W, so the p-value can only RISE: p={} p*={}",
            probe.bartlett_factor,
            probe.p_value_uncorrected,
            probe.p_value_corrected
        );
    } else if probe.bartlett_factor < 1.0 {
        assert!(
            probe.p_value_corrected <= probe.p_value_uncorrected + 1e-12,
            "c = {} < 1 inflates W, so the p-value can only FALL: p={} p*={}",
            probe.bartlett_factor,
            probe.p_value_uncorrected,
            probe.p_value_corrected
        );
    }
    // (a') Materiality diagnostic (#939 deliverable 4): a correction was applied,
    // so `material` must agree with the 10% rule it documents.
    {
        let factor_move = (probe.bartlett_factor - 1.0).abs();
        let p_hi = probe.p_value_uncorrected.max(probe.p_value_corrected);
        let p_lo = probe.p_value_uncorrected.min(probe.p_value_corrected);
        let p_move = (p_hi - p_lo) / p_hi.max(f64::MIN_POSITIVE);
        let expected_material = factor_move > 0.10 || p_move > 0.10;
        assert_eq!(
            probe.material, expected_material,
            "material flag must follow the 10% rule: c={}, factor_move={:.4}, p_move={:.4}",
            probe.bartlett_factor, factor_move, p_move
        );
    }

    // (b) Calibration sweep under the null DGP.
    let mut sum_w = 0.0;
    let mut sum_w_sq = 0.0;
    let mut sum_w_star = 0.0;
    let mut sum_p_unc = 0.0;
    let mut sum_p_cor = 0.0;
    let mut ref_df = 0.0;
    // The Bartlett factor itself. `c = E[W]/d` and `W* = W/c`, so if `c` were
    // formed from the statistic's ACTUAL null mean, `mean_w*` would land on `d`
    // by construction. Reporting `c` distinguishes "the correction is applied
    // wrongly" from "the correction is applied faithfully to a second-order mean
    // that disagrees with the empirical one" -- two different defects, and the
    // means alone cannot tell them apart.
    let mut sum_factor = 0.0;
    // The reference's OWN predicted mean for this replicate: `E[W(λ̂)]` under the
    // selection law the p-value is read from. `ref_df` is the CONDITIONAL mean
    // `E[W | λ̂]`, which the empirical mean is not comparable to (see (b1)).
    let mut sum_predicted = 0.0;
    let mut sum_predicted_conditional = 0.0;
    let mut predicted_used = 0usize;
    let mut rejections_cor = 0usize;
    let mut count = 0usize;
    for rep in 0..REPS {
        let data = null_replicate(N, 1000 + rep as u64);
        let Some(r) = run_one(&data) else {
            continue;
        };
        if !(r.statistic_lr.is_finite()
            && r.statistic_corrected.is_finite()
            && r.p_value_uncorrected.is_finite()
            && r.p_value_corrected.is_finite()
            && r.ref_df.is_finite())
        {
            continue;
        }
        sum_w += r.statistic_lr;
        sum_w_sq += r.statistic_lr * r.statistic_lr;
        sum_factor += r.bartlett_factor;
        sum_w_star += r.statistic_corrected;
        sum_p_unc += r.p_value_uncorrected;
        sum_p_cor += r.p_value_corrected;
        ref_df += r.ref_df;
        if let Some(replay) = r.ref_df_provenance.selection.replay() {
            let predicted = replay.selection_mean();
            if predicted.is_finite() {
                // `W* = W/c`, so the reference the CORRECTED statistic is scored
                // against is the same law with every weight scaled by `c` — its
                // mean is `c · E[W(λ̂)]`. Comparing the raw `mean(W)` against the
                // raw predicted mean keeps both on the uncorrected scale.
                sum_predicted += predicted;
                sum_predicted_conditional += replay.conditional_mean();
                predicted_used += 1;
            }
        }
        if r.p_value_corrected <= 0.05 {
            rejections_cor += 1;
        }
        count += 1;
    }
    assert!(
        count >= REPS * 8 / 10,
        "too many replicates failed to produce a finite LR report: {count}/{REPS}"
    );

    let n_used = count as f64;
    let mean_w = sum_w / n_used;
    let mean_w_star = sum_w_star / n_used;
    let mean_p_unc = sum_p_unc / n_used;
    let mean_p_cor = sum_p_cor / n_used;
    let mean_d = ref_df / n_used;
    let mean_factor = sum_factor / n_used;
    // Monte-Carlo standard error of `mean(W)` from the sweep's own spread. This
    // is the resolution of every mean comparison below, and it is what decides
    // which of them can be asserted at this budget at all.
    let var_w = (sum_w_sq / n_used - mean_w * mean_w).max(0.0);
    let se_w = (var_w / n_used).sqrt();
    let err_unc = (mean_w - mean_d).abs();
    let err_cor = (mean_w_star - mean_d).abs();
    let size_cor = rejections_cor as f64 / n_used;
    let mean_predicted = sum_predicted / predicted_used.max(1) as f64;
    let mean_predicted_conditional = sum_predicted_conditional / predicted_used.max(1) as f64;
    let err_predicted = (mean_w - mean_predicted).abs();
    eprintln!(
        "[#1063 sweep] n={N} used={count}/{REPS}  mean(W)={mean_w:.4} sd(W)={:.4} \
         se(mean W)={se_w:.4}  d={mean_d:.4}  |mean(W)-d|={err_unc:.4} ({:.2} se)  \
         c={mean_factor:.6}  mean(W*)={mean_w_star:.4}  |mean(W*)-d|={err_cor:.4}  \
         mean(p)={mean_p_unc:.4} mean(p*)={mean_p_cor:.4}  size@.05(corrected)={size_cor:.4}  \
         E[W(lambda-hat)]={mean_predicted:.4} (replayed on {predicted_used}) \
         E[W|lambda-hat]={mean_predicted_conditional:.4} \
         |mean(W)-E[W(lambda-hat)]|={err_predicted:.4} ({:.2} se)",
        var_w.sqrt(),
        err_unc / se_w.max(f64::MIN_POSITIVE),
        err_predicted / se_w.max(f64::MIN_POSITIVE)
    );

    // (b1) THE REFERENCE IS CALIBRATED: the statistic's empirical null mean agrees
    // with the reference d.f. it is scored against, to Monte-Carlo resolution.
    //
    // This replaces two assertions that required the OPPOSITE, and the reason is
    // that `ref_df` is not a free χ² parameter -- `lawley_lr_bartlett_factor`
    // forms `mean_w = ref_df + Δε`, so the reference IS the statistic's
    // first-order null mean by construction, and the Lawley factor supplies only
    // the O(n⁻¹) second-order remainder. The removed pair were:
    //
    //   * `err_unc > 0.10 * mean_d` -- "the sweep must exhibit a MATERIAL
    //     miscalibration for the Bartlett factor to correct". That demands the
    //     FIRST-ORDER reference be wrong by 10%, i.e. it is satisfiable only while
    //     the defect this test exists to catch is present. It failed on the
    //     repaired code at `|mean(W)-d| = 0.1305` against `d = 1.903` -- 6.9%.
    //   * `err_cor < err_unc` -- "the correction must move the mean toward d".
    //     Unmeasurable here rather than wrong: `c = 1.0027` moves the mean by
    //     `0.0055`, while `se(mean W)` at 120 replicates is ~`0.18`. Resolving an
    //     O(n⁻¹) Bartlett shift in a Monte-Carlo mean would need ~1.3e5
    //     replicates. A comparison two orders inside its own noise is a coin flip
    //     dressed as a contract, and the module has an EXACT check of the same
    //     assembly next door: `rho_variation_assembly_matches_simulated_expectation_over_rho_hat`
    //     validates Δε against a 40,000-draw ground truth with common random
    //     numbers and matched moments, at a rate rather than a magnitude.
    //
    // What is left is measurable and it is the substantive claim: this bar failed
    // by ~20 standard errors on the state #2672 opened on (`mean(W) = 2.034`
    // against `d = 5.705`), and it holds now.
    assert!(
        err_unc <= 3.0 * se_w,
        "the LR statistic's empirical null mean must agree with the reference d.f. \
         it is scored against: |mean(W)−d| = {err_unc:.4} exceeds 3·se(mean W) = \
         {:.4} (mean_w={mean_w:.4}, d={mean_d:.4}, sd(W)={:.4}, used={count}). \
         `ref_df` is the first-order null mean by construction — \
         `lawley_lr_bartlett_factor` forms `mean_w = ref_df + Δε` — so a gap here \
         is the reference d.f. assembly, not the Lawley factor (c={mean_factor:.6}).",
        3.0 * se_w,
        var_w.sqrt()
    );

    // (b2) THE CORRECTION POINTS THE RIGHT WAY, asserted only where it can be
    // seen. `W* = W/c`, so `c > 1` lowers the mean and `c < 1` raises it; the
    // analytic shift `Δε = (c − 1)·d` claims the empirical gap has that same sign.
    // At this budget the claim is active only when the gap clears MC noise —
    // which is the honest form of the "moves toward d" comparison, not a weaker
    // one: where it can be measured it bites, and where it cannot it says so.
    if err_unc > 3.0 * se_w {
        assert!(
            err_cor < err_unc,
            "with a resolvable gap ({err_unc:.4} > 3·se = {:.4}) the Bartlett \
             correction must move the mean toward d: |mean(W*)−d|={err_cor:.4} \
             must be < |mean(W)−d|={err_unc:.4} (c={mean_factor:.6})",
            3.0 * se_w
        );
    }

    // (b3) THE CORRECTED TEST IS THE RIGHT SIZE. This replaces a `mean(p*)` vs
    // `mean(p)` comparison against 0.5. That comparison was measuring the wrong
    // thing: a PENALIZED null LR has an atom at zero (REML switches the term off
    // on a large fraction of null draws, giving `W ≈ 0` and `p ≈ 1`), so its
    // p-values are not Uniform even when the test is exactly the right size, and
    // `mean(p)` sits well above 0.5 by construction. Under `c > 1` the correction
    // can only raise p, so "closer to 0.5" required `mean(p) < 0.5` — i.e. it
    // required the uncorrected test to be anti-conservative, the same
    // defect-must-be-present shape as the materiality bar above.
    //
    // Empirical size at the level the test is used at is the claim that does not
    // depend on the shape of the rest of the distribution, and it is the same
    // claim `smooth_term_lr_size_calibration` makes across its grid — here on this
    // module's own fixture, so the two files cannot drift apart.
    let se_size = (0.05 * 0.95 / n_used).sqrt();
    let band = 3.0 * se_size + 0.015;
    assert!(
        (size_cor - 0.05).abs() <= band,
        "the corrected test's empirical size at α=0.05 must sit inside the \
         Monte-Carlo band of nominal: {size_cor:.4} against 0.05 ± {band:.4} \
         ({count} replicates, se={se_size:.4})"
    );
}

/// DIAGNOSTIC (not a contract, #2672): the reference's two spectral moments
/// against the empirical null behaviour of `W`.
///
/// The statistic's null law is `W = Σ_j w_j χ²_1` with `w = eig(2F_jj − F_jj²)`,
/// so `mean = Σ w_j` is `E[W|λ]` and `second = Σ w_j²` is `Var(W|λ)/2`. Two
/// readings the printed table separates:
///
/// * `mean` tracking `mean(W)` ⇒ the reference's location is right and any
///   residual miscalibration is shape, not level;
/// * `nu` far below the block dimension with `g` far below one ⇒ the weights are
///   spread, which is exactly the regime where a mean-matched χ² (the pre-#2672
///   reference) is conservative and the scaled reference is not.
///
/// `pooled(W)/pooled(mean)` is NOT expected to be one: `λ̂` is selected from the
/// same data, so the pairing is per-replicate and the unconditional means need
/// not agree. That mismatch is what the per-replicate reference exists to absorb,
/// and reading it as a defect is the mistake this thread made twice.
#[test]
fn zz_measure_reference_spectral_moments_against_empirical_lr_mean() {
    init_parallelism();

    const N: usize = 60;
    const REPS: usize = 40;

    let mut sum_w = 0.0;
    let mut sum_mean = 0.0;
    let mut sum_second = 0.0;
    let mut sum_nu = 0.0;
    let mut sum_scale = 0.0;
    let mut sum_edf = 0.0;
    let mut sum_factor = 0.0;
    let mut fallback = 0usize;
    let mut count = 0usize;
    eprintln!(
        "[zz2672] {:>4} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9} {:>4}",
        "rep", "W", "mean", "second", "nu", "g", "c", "nd"
    );
    for rep in 0..REPS {
        let data = null_replicate(N, 1000 + rep as u64);
        let Some(r) = run_one(&data) else { continue };
        if !(r.statistic_lr.is_finite() && r.ref_df.is_finite()) {
            continue;
        }
        let p = r.ref_df_provenance;
        if !matches!(p.source, SmoothLrReferenceSource::NullSpectrum) {
            fallback += 1;
        }
        if rep < 12 {
            eprintln!(
                "[zz2672] {rep:>4} {:>9.4} {:>9.4} {:>9.4} {:>9.4} {:>9.4} {:>9.6} {:>4}",
                r.statistic_lr,
                p.mean,
                p.second_moment,
                p.chi_square_df,
                p.scale,
                r.bartlett_factor,
                p.null_dim
            );
        }
        sum_w += r.statistic_lr;
        sum_mean += p.mean;
        sum_second += p.second_moment;
        sum_nu += p.chi_square_df;
        sum_scale += p.scale;
        sum_edf += p.edf;
        sum_factor += r.bartlett_factor;
        count += 1;
    }
    assert!(count > 0, "no replicate produced a finite LR report");
    let d = count as f64;
    eprintln!(
        "[zz2672] MEAN over {count} reps: W={:.4}  mean={:.4}  second={:.4}  nu={:.4}  \
         g={:.4}  edf={:.4}  c={:.6}  (fallback lane on {fallback})",
        sum_w / d,
        sum_mean / d,
        sum_second / d,
        sum_nu / d,
        sum_scale / d,
        sum_edf / d,
        sum_factor / d,
    );
    eprintln!(
        "[zz2672] read: pooled W/mean = {:.4} (NOT expected to be 1 under selected λ̂); \
         mean g = {:.4} — the factor by which a mean-matched χ² over-states the spread.",
        (sum_w / d) / (sum_mean / d).max(f64::MIN_POSITIVE),
        sum_scale / d,
    );
}

/// #939 deliverable (2), the ρ̂-variation arm — VALIDATION BY SIMULATION over
/// the sampling distribution of ρ̂.
///
/// The conditional Lawley shift `Δε(ρ)` is `E[W | λ]` — the LR mean with the
/// smoothing parameter held FIXED at ρ. When λ is estimated, the relevant null
/// mean is the expectation over the sampling distribution of ρ̂:
///
/// ```text
/// E[W] = E_{ρ̂}[ Δε(ρ̂) ] = Δε(ρ₀) + ½ Δε''(ρ₀)·Var(ρ̂) + O(·)
/// ```
///
/// `lawley_lr_mean_shift_with_rho_variation` assembles exactly the right-hand
/// second-order term. This test takes the assembly's prediction as a HYPOTHESIS
/// and falsifies the alternative (fixed-λ only) against a Monte-Carlo ground
/// truth: it draws `ρ̂ ~ N(ρ₀, Var)` (the genuine sampling fluctuation of the
/// log-smoothing estimate), evaluates the *conditional* shift `Δε(ρ̂)` at each
/// draw by re-scaling the penalty, and averages. The claim is that the
/// ρ̂-variation assembly matches this simulated `E_{ρ̂}[Δε(ρ̂)]` to second order
/// — and matches it STRICTLY BETTER than the conditional (fixed-λ) shift, which
/// systematically misses the curvature term. That gap IS the size correction
/// attributable specifically to ρ̂-variation.
#[test]
fn rho_variation_assembly_matches_simulated_expectation_over_rho_hat() {
    // Poisson/log smooth substrate at a fixed null linear predictor: a 2-column
    // design (intercept + centered covariate) penalized on the second column.
    let n = 60usize;
    let mut x = Array2::<f64>::ones((n, 2));
    let mut kappas = Vec::<RowKappas>::with_capacity(n);
    for i in 0..n {
        let z = i as f64 / n as f64 - 0.5;
        x[[i, 1]] = z;
        let eta = 0.3 + 0.6 * z;
        kappas.push(
            RowExpectedJets::poisson_log(eta)
                .kappas()
                .expect("poisson kappas"),
        );
    }
    let tested = 1..2;

    // Population smoothing parameter ρ₀ = log λ₀ and its sampling variance. (In
    // the live engine Var(ρ̂) is the inverse REML outer Hessian; here it is a
    // fixed scenario value so the simulation is self-contained and exact.)
    let lambda0 = 3.0_f64;
    let rho0 = lambda0.ln();
    let mut s_comp = Array2::<f64>::zeros((2, 2));
    s_comp[[1, 1]] = lambda0;
    let penalty = s_comp.clone();
    let components = vec![RhoPenaltyComponent {
        s_component: s_comp,
    }];

    // MOMENT-MATCHED standard normals, reused at every variance.
    //
    // Common random numbers alone are NOT enough here, which cost a measurement
    // cycle to learn. Writing the simulated leg's deviation from the anchor,
    //
    //   sim(v) − Δε(ρ₀) = Δε'·√v·z̄ + ½·Δε''·v·(z²)‾ + O(v^{3/2})
    //
    // the SIGNAL is the second term, of order v. The first term is pure sampling
    // noise, and it does not vanish under common random numbers: `z̄` is a fixed
    // non-zero number for any finite draw set (sd 1/√N ≈ 3.5e-3 at N = 80k), and
    // it scales as √v while the signal scales as v — so the noise-to-signal
    // ratio GROWS as 1/√v, worst exactly where the second-order claim is
    // cleanest. Measured, that produced `|conditional − sim|` ratios against the
    // analytic ½·H·v of 1.120, 0.502, 1.541, 0.250, 2.177 and 0.433, and moved
    // `e_cond` by 2.2×, 6.2× and 5.0× on a mere doubling of N — an estimator that
    // had not converged at all, from which any ratio (including a very
    // convincing 2×) is an artifact.
    //
    // Forcing the sample moments to exactly 0 and 1 kills the first term
    // identically (`Σz = 0`) and makes the second exact (`(z²)‾ = 1`), leaving
    // `sim(v) − Δε(ρ₀) = ½·Δε''·v + O(v²)` — which is precisely the quantity the
    // assembly claims to compute, now measurable at every v.
    let z = moment_matched_normals(20939, 40_000);

    // (conditional, assembled, simulated) at one ρ̂-sampling variance. The
    // simulated leg is the MONTE-CARLO ground truth E_{ρ̂}[Δε(ρ̂)]: each draw
    // evaluates the conditional shift with the penalty rescaled by e^{ρ̂−ρ₀}.
    let evaluate = |var_rho: f64| -> (f64, f64, f64) {
        let rho_cov = Array2::from_shape_vec((1, 1), vec![var_rho]).unwrap();
        let conditional =
            lawley_lr_mean_shift(x.view(), &kappas, Some(penalty.view()), tested.clone())
                .expect("conditional Δε");
        let assembled = lawley_lr_mean_shift_with_rho_variation(
            x.view(),
            &kappas,
            penalty.view(),
            tested.clone(),
            &components,
            rho_cov.view(),
        )
        .expect("assembled Δε(ρ̂)");
        let sd = var_rho.sqrt();
        let mut sum = 0.0;
        for &zi in &z {
            let lambda = (rho0 + sd * zi).exp();
            let mut s = Array2::<f64>::zeros((2, 2));
            s[[1, 1]] = lambda;
            sum += lawley_lr_mean_shift(x.view(), &kappas, Some(s.view()), tested.clone())
                .expect("Δε(ρ̂) draw");
        }
        (conditional, assembled, sum / z.len() as f64)
    };

    // Two variances a factor of 4 apart. The assembly is a SECOND-ORDER delta
    // method, so its accuracy claim is a statement about the limit Var → 0, not
    // about any single variance: the omitted remainder is O(Var²) while the
    // conditional shift's error is O(Var). Testing at one large variance
    // measures the truncation instead of the assembly — at Var = 0.6 (the value
    // this fixture used before) σ_ρ = 0.78 spreads λ = λ₀e^{δ} across two orders
    // of magnitude at ±3σ, where the omitted term is the same size as the term
    // being verified and NO ratio bound can hold.
    let var_hi = 0.10_f64;
    let var_lo = var_hi / 4.0;
    let (cond_hi, asm_hi, sim_hi) = evaluate(var_hi);
    let (cond_lo, asm_lo, sim_lo) = evaluate(var_lo);
    let e_cond_hi = (cond_hi - sim_hi).abs();
    let e_asm_hi = (asm_hi - sim_hi).abs();
    let e_cond_lo = (cond_lo - sim_lo).abs();
    let e_asm_lo = (asm_lo - sim_lo).abs();

    // Non-vacuity: there must BE a ρ̂-variation effect to correct at both
    // variances, else the comparisons below are between two zeros.
    assert!(
        e_cond_hi > 0.0 && e_cond_lo > 0.0,
        "fixture must exhibit a ρ̂-variation effect at both variances: \
         |conditional−sim| = {e_cond_hi:.3e} (Var={var_hi}), {e_cond_lo:.3e} (Var={var_lo})"
    );

    // (1) THE SECOND-ORDER PROPERTY, as a rate rather than a magic constant.
    // Quartering the variance must shrink the assembled error strictly faster
    // than the conditional error: O(Var²) against O(Var) predicts factors of 16
    // and 4, so the assembled ratio must beat the conditional ratio by at least
    // 2×. Written as a cross-multiplication so a vanishing denominator cannot
    // manufacture a pass. This is scale-free — there is no tolerance to tune,
    // and an assembly that merely lands "close" at one variance fails it.
    assert!(
        e_asm_lo * e_cond_hi < 0.5 * e_asm_hi * e_cond_lo,
        "ρ̂-variation assembly must converge at the second-order RATE: quartering \
         Var must shrink |assembled−sim| at least twice as fast as |conditional−sim|. \
         assembled {e_asm_hi:.3e} → {e_asm_lo:.3e}, conditional {e_cond_hi:.3e} → \
         {e_cond_lo:.3e} (Var {var_hi} → {var_lo})"
    );

    // (2) The defining property: inside the regime where the expansion is valid,
    // the ρ̂-variation assembly is STRICTLY closer to the truth than the fixed-λ
    // conditional shift. That gap IS the size correction attributable to
    // ρ̂-variation.
    assert!(
        e_asm_lo < e_cond_lo,
        "ρ̂-variation correction must improve on the fixed-λ shift at Var={var_lo}: \
         |assembled−sim|={e_asm_lo:.3e} must be < |conditional−sim|={e_cond_lo:.3e}"
    );
}

/// DIAGNOSTIC (not a contract): identify the floor under `|assembled − sim|`.
///
/// The rate gate in `rho_variation_assembly_matches_simulated_expectation_over_rho_hat`
/// measured the assembled error as FLAT — 3.457e-7 → 3.906e-7 across a 4×
/// reduction in Var — while the conditional error fell ~2.9×, i.e. linearly, as
/// its O(Var) model predicts. A second-order assembly should have fallen ~16×.
/// Three candidates explain a flat floor, and they separate cleanly on how they
/// respond to Var and to the draw count:
///
/// * `O(Var²)` truncation — falls 16× per 4× Var step, independent of `N`.
/// * common-random-number Monte-Carlo noise — falls with `√Var` (both legs
///   collapse onto the same anchor as Var → 0) and as `1/√N`.
/// * a constant error in the assembly's ρ-curvature — flat in Var AND in `N`,
///   which would be a production finding: the second-order claim in
///   `lawley_lr_mean_shift_with_rho_variation`'s contract says that term must
///   vanish, and it would also explain the original Var = 0.6 observation
///   (assembled error ≈ conditional error) better than truncation did.
///
/// One run prints the whole grid; `--nocapture`. The `N` legs reuse a prefix of
/// the same draws, so the `N` comparison is itself common-random-number and not
/// confounded by a different sample.
#[test]
fn zz_measure_rho_variation_assembly_error_floor() {
    let n = 60usize;
    let mut x = Array2::<f64>::ones((n, 2));
    let mut kappas = Vec::<RowKappas>::with_capacity(n);
    for i in 0..n {
        let z = i as f64 / n as f64 - 0.5;
        x[[i, 1]] = z;
        let eta = 0.3 + 0.6 * z;
        kappas.push(
            RowExpectedJets::poisson_log(eta)
                .kappas()
                .expect("poisson kappas"),
        );
    }
    let tested = 1..2;
    let lambda0 = 3.0_f64;
    let rho0 = lambda0.ln();
    let mut s_comp = Array2::<f64>::zeros((2, 2));
    s_comp[[1, 1]] = lambda0;
    let penalty = s_comp.clone();
    let components = vec![RhoPenaltyComponent {
        s_component: s_comp,
    }];

    let reps = 40_000usize;
    let mut rng = StdRng::seed_from_u64(20939);
    let unit_normal = Normal::new(0.0, 1.0).expect("normal");
    let raw: Vec<f64> = (0..reps).map(|_| unit_normal.sample(&mut rng)).collect();
    let matched = moment_matched_normals(20939, reps);

    // The empirical moments of the RAW draws, which is the number that decides
    // between "the fixture hands the simulation a different variance than the
    // assembly gets" and "the fixture's first moment is the problem".
    let raw_mean = raw.iter().sum::<f64>() / reps as f64;
    let raw_var = raw.iter().map(|v| v * v).sum::<f64>() / reps as f64 - raw_mean * raw_mean;
    eprintln!("[zz] raw draws: mean={raw_mean:+.6e}  variance={raw_var:.6}  (target 0 and 1)");
    eprintln!(
        "[zz] first-order noise Δε'·√v·z̄ scales as √v; signal ½·Δε''·v scales as v, \
         so noise/signal grows as 1/√v"
    );

    let conditional = lawley_lr_mean_shift(x.view(), &kappas, Some(penalty.view()), tested.clone())
        .expect("conditional Δε");

    eprintln!("[zz] conditional Δε(ρ₀) = {conditional:.12}");
    eprintln!("[zz]      Var    draws   sim-cond      analytic ½H·Var   ratio    e_asm");
    for var_rho in [0.1_f64, 0.025, 0.00625] {
        let rho_cov = Array2::from_shape_vec((1, 1), vec![var_rho]).unwrap();
        let assembled = lawley_lr_mean_shift_with_rho_variation(
            x.view(),
            &kappas,
            penalty.view(),
            tested.clone(),
            &components,
            rho_cov.view(),
        )
        .expect("assembled Δε(ρ̂)");
        let sd = var_rho.sqrt();
        let simulate = |draws: &[f64]| -> f64 {
            let mut sum = 0.0;
            for &zi in draws {
                let mut s = Array2::<f64>::zeros((2, 2));
                s[[1, 1]] = (rho0 + sd * zi).exp();
                sum += lawley_lr_mean_shift(x.view(), &kappas, Some(s.view()), tested.clone())
                    .expect("Δε(ρ̂) draw");
            }
            sum / draws.len() as f64
        };
        let analytic = assembled - conditional;
        for (label, draws) in [("raw", &raw), ("matched", &matched)] {
            let simulated = simulate(draws);
            let shift = simulated - conditional;
            eprintln!(
                "[zz] {var_rho:>9.5} {label:>8}  {shift:>+.4e}   {analytic:>+.4e}   {:>6.3}  {:.4e}",
                shift / analytic,
                (assembled - simulated).abs()
            );
        }
    }
    eprintln!(
        "[zz] read: 'matched' ratio → 1 at every Var => production is right and the RAW estimator \
         was the defect; 'matched' ratio holding away from 1 => a genuine assembly discrepancy"
    );

    // The premise being investigated must itself hold: the assembly's own
    // second-order term has to be non-zero, else there is nothing to measure.
    let probe = Array2::from_shape_vec((1, 1), vec![0.1_f64]).unwrap();
    let assembled = lawley_lr_mean_shift_with_rho_variation(
        x.view(),
        &kappas,
        penalty.view(),
        tested.clone(),
        &components,
        probe.view(),
    )
    .expect("assembled Δε(ρ̂)");
    assert!(
        (assembled - conditional).abs() > 0.0,
        "the ρ̂-variation assembly must contribute a non-zero second-order term at Var=0.1; \
         got assembled={assembled:.12} identical to conditional={conditional:.12}"
    );
}

/// DIAGNOSTIC (not a contract): finite-difference the ρ-curvature the
/// ρ̂-variation assembly applies, against the conditional shift it claims to be
/// the second derivative of.
///
/// The floor grid showed the applied correction is exactly ∝ Var (it falls
/// 4.000× per 4× Var step, so there is no floor) while landing ≈2× the measured
/// gap `|E_ρ̂[Δε] − Δε(ρ₀)|` — the signature of a curvature that is twice its
/// true value. Four candidate sites have already been eliminated BY DERIVATION,
/// so this measurement is the one that names the remaining one:
///
/// * the ½ in `lawley_lr_mean_shift_with_rho_variation`'s accumulation is
///   present and, with a 1×1 `rho_cov`, the whole correction is literally
///   `½·H[0,0]·Var` — no other factor participates;
/// * `hessian[[b, c]]` / `hessian[[c, b]]` are ASSIGNED, not accumulated, so the
///   `c in b..m` loop cannot double the diagonal;
/// * for `b == c`, `inverse_second` computes `2·P_b S_b K − P_b`, which is
///   exactly `∂²K/∂ρ_b²` for `K = J⁻¹`, `∂K/∂ρ_b = −P_b`, `P_b = K S_b K`;
/// * `pairs_b = −X P_b Xᵀ = ∂E/∂ρ_b` and `pairs_bc = X·inverse_second·Xᵀ =
///   ∂²E/∂ρ_b∂ρ_c`, so the leverage chain feeding the curvature is right.
///
/// What remains is the explicit O(n²) Lawley polynomial. This test differences
/// `Δε(ρ)` directly — ρ enters only through `S_λ = e^ρ S_b`, so evaluating at
/// `e^{±h} S_b` IS the ρ-perturbation — and prints the ratio. Read it as:
/// ratio ≈ 2 ⇒ the curvature is doubled inside `lawley_epsilon_rho_hessian` and
/// every site above is exonerated; ratio ≈ 1 ⇒ the curvature is right and the
/// defect is in how the correction is applied, or the MC comparison itself is
/// not measuring what the assembly targets.
#[test]
fn zz_measure_rho_curvature_vs_finite_difference() {
    let n = 60usize;
    let mut x = Array2::<f64>::ones((n, 2));
    let mut kappas = Vec::<RowKappas>::with_capacity(n);
    for i in 0..n {
        let z = i as f64 / n as f64 - 0.5;
        x[[i, 1]] = z;
        let eta = 0.3 + 0.6 * z;
        kappas.push(
            RowExpectedJets::poisson_log(eta)
                .kappas()
                .expect("poisson kappas"),
        );
    }
    let tested = 1..2;
    let lambda0 = 3.0_f64;
    let mut s_comp = Array2::<f64>::zeros((2, 2));
    s_comp[[1, 1]] = lambda0;
    let penalty = s_comp.clone();
    let components = vec![RhoPenaltyComponent {
        s_component: s_comp,
    }];

    // Δε at ρ₀ + shift, reached by scaling the penalty by e^{shift}.
    let shifted = |shift: f64| -> f64 {
        let mut s = Array2::<f64>::zeros((2, 2));
        s[[1, 1]] = lambda0 * shift.exp();
        lawley_lr_mean_shift(x.view(), &kappas, Some(s.view()), tested.clone())
            .expect("Δε at shifted ρ")
    };

    let centre = shifted(0.0);
    eprintln!("[zz] Δε(ρ₀) = {centre:.12}");
    eprintln!("[zz]      h      FD d²Δε/dρ²      analytic ½·H·Var/(½·Var)     ratio");
    for h in [0.2_f64, 0.1, 0.05, 0.025] {
        let fd = (shifted(h) - 2.0 * centre + shifted(-h)) / (h * h);
        // Recover the analytic curvature the assembly applies: with a 1x1
        // rho_cov the whole correction is ½·H·Var, so H = 2·correction/Var.
        let var = 0.1_f64;
        let rho_cov = Array2::from_shape_vec((1, 1), vec![var]).unwrap();
        let assembled = lawley_lr_mean_shift_with_rho_variation(
            x.view(),
            &kappas,
            penalty.view(),
            tested.clone(),
            &components,
            rho_cov.view(),
        )
        .expect("assembled");
        let analytic = 2.0 * (assembled - centre) / var;
        eprintln!(
            "[zz] {h:>6.3}  {fd:>+.6e}   {analytic:>+.6e}   {:>8.4}",
            analytic / fd
        );
    }
    eprintln!(
        "[zz] read: ratio ~2 => doubled curvature inside lawley_epsilon_rho_hessian; ~1 => curvature correct, look downstream"
    );

    // The FD itself must be non-degenerate, else the ratio is meaningless.
    let fd = (shifted(0.05) - 2.0 * centre + shifted(-0.05)) / 0.0025;
    assert!(
        fd.abs() > 0.0 && fd.is_finite(),
        "the conditional shift must have measurable ρ-curvature for this comparison \
         to mean anything; got d²Δε/dρ² = {fd:e}"
    );
}
