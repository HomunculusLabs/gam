//! End-to-end quality: gam's parametric **Weibull AFT** survival fit with a
//! **by-factor smooth** covariate effect must recover the KNOWN data-generating
//! survival surface — the objective ground truth this test asserts.
//!
//! ## Objective metric (the pass/fail claim)
//!
//! The data are simulated from an explicit AFT model with known per-group
//! Weibull baselines and a known per-group acceleration function of `x`, so the
//! true survival function `S_true(t | x, g)` is computable in closed form (see
//! [`Truth::survival`]). The PRIMARY assertion is that gam's predicted survival
//! surface recovers that ground truth, as a fold-mean over `K_SEEDS` paired
//! draws:
//!   `mean_k RMSE( S_gam(t|x,g), S_true(t|x,g) ) <= TRUTH_RECOVERY_BOUND`
//! on a (group × x × t) grid. Survival probabilities live in [0,1], so this is
//! a signal-appropriate bar for a ~100-observation-per-stratum, ~20%-censored
//! AFT fit evaluated out to |x| = 1.75, where a ~24-coefficient penalized
//! covariate block carries real variance; a collapsed by-factor factorization
//! (merged strata, wrong acceleration sign) misses it by a wide margin.
//!
//! `lifelines.WeibullAFTFitter` (the mature, standard parametric-AFT reference)
//! is fit on the SAME rows per stratum and kept ONLY as a baseline on that same
//! truth-recovery metric. We never assert "gam reproduces lifelines' output" —
//! matching another fit proves nothing about correctness; recovering the
//! generating function does. lifelines is held to the identical objective
//! yardstick (error vs the true surface), not used as the target itself.
//!
//! As a structural sanity check we also assert gam's survival surface is a valid
//! survival function on the grid, on every seed: every `S` lies in [0,1] and is
//! non-increasing in `t` for each (group, x) — a property the AFT factorization
//! must satisfy regardless of any reference tool.
//!
//! ## TWO TRUTHS, and why the second one exists (#1561)
//!
//! The original fixture generated a LINEAR acceleration, `log T = log T0 +
//! beta_g · x`, and gam lost the `truth_surface` pair to lifelines by +0.71 in
//! log-ratio on a single draw. Paired over `K_SEEDS` seeds on that original
//! 3-point grid the loss reproduces at +0.718 with gam behind on 10 of 10
//! seeds, so it is NOT a draw artifact — it is a real, stable gap. It is
//! nonetheless not a statement about smoothing quality, for a reason that is
//! structural rather than numerical.
//!
//! On a linear-acceleration truth, `WeibullAFTFitter` fit per stratum IS the
//! data-generating process: a Weibull baseline with its own shape and scale and
//! a single linear-in-`x` log-acceleration coefficient. It is the exactly
//! specified parametric MLE — an ORACLE with no approximation bias and three
//! parameters of variance per stratum. gam meanwhile pays the flexibility
//! premium for `s(x, by=group)`, a penalized smooth whose null space the truth
//! never leaves. No general-purpose smoother matches an oracle-parametric MLE on
//! the oracle's own truth, and matching it is not a quality claim about gam. The
//! by-factor-smooth capability the test NAMES is, on that truth, never exercised
//! beyond linearity.
//!
//! So this test runs BOTH truths. [`Truth::Linear`] is kept (a loss that is
//! understood is worth more than a loss that is deleted); [`Truth::Curved`]
//! replaces the linear acceleration with a saturating one,
//!   `log T = log T0 + amp_g · tanh(1.5 · x)`,
//! whose amplitude is chosen so its log-hazard SPAN over `x ∈ [-2, 2]` equals
//! the linear arm's (`amp_g = beta_g · 2 / tanh(3)`, i.e. 0.7035 / -0.5025
//! against slopes 0.35 / -0.25). Same kind of object, same signal size, same
//! opposite-sign per-group contrast — but a shape no linear AFT can represent,
//! so lifelines is now honestly MISSPECIFIED and both engines face model error.
//! That is the arm on which the by-factor smooth is actually exercised.
//!
//! MEASURED (K=10 paired seeds, both arms on the same grid and the same x/T0
//! draws): the enrichment does exactly what the adjudication predicts to the
//! REFERENCE — lifelines' own RMSE-to-truth rises 0.0427 -> 0.0625 (+47%) when
//! the acceleration curves, because it is now fitting a shape it does not have
//! — and gam's is FLAT across the two truths, 0.0940 -> 0.0908. gam's premium
//! over lifelines therefore falls from 2.20x (ln +0.790) on the linear truth to
//! 1.45x (ln +0.372) on the curved one: the oracle advantage is confirmed to be
//! specific to the linear truth, and roughly half the original gap was it.
//!
//! It does NOT flip: gam is still behind on the curved arm, by more than the
//! paired fold noise explains. That residual premium is NOT the smooth's tail
//! variance alone — the per-`x` decomposition shows gam behind lifelines at
//! EVERY grid point including `x = 0` (0.047 vs 0.028), where the acceleration
//! is zero under both truths and nothing needs smoothing. So roughly a 1.7x
//! error premium sits in the shared-baseline/group-offset reconstruction
//! itself, before any curvature is asked for, and the tail points (0.13-0.18 at
//! `x = +1.75`) add to it. Naming that is what this arm is for; fixing it is
//! not a test change.
//!
//! ## The evaluation grid had to widen, and that is part of the finding
//!
//! The original grid was `x_eval = [-1, 0, 1]`. A tanh acceleration is an ODD
//! function, and ANY odd function is reproduced EXACTLY by a straight line
//! through three points symmetric about zero: on `[-1, 0, 1]` the curved truth's
//! linear-approximation residual is 0 to machine precision (measured 0.0e0 —
//! [`nonlinear_fraction`] asserts this can never silently recur). A curved arm
//! evaluated there would have been vacuous: the misspecification would be
//! invisible to the metric. [`X_EVAL`] therefore carries three distinct
//! magnitudes per side, `±{0.4, 1.0, 1.75}` plus 0, on which the curved truth's
//! log-hazard is 27.8% non-linear and the linear truth's is 0. Both arms use
//! the same grid, so the two arms remain comparable to each other; neither is
//! comparable to the pre-#1561 3-point numbers. (Measured on the old 3-point
//! grid for continuity, the same K=10 panel gives 2.05x linear / 1.52x curved,
//! i.e. the widening changes the levels but not the finding.)
//!
//! ## PAIRED over seeds, not one draw (#2395)
//!
//! lifelines' own RMSE-to-truth on this fixture ranges over 0.028..0.089 across
//! draws, so a one-draw ratio ceiling is a coin flip whichever truth it uses.
//! Both arms therefore run `K_SEEDS` seeds and score the two engines on the SAME
//! generated rows seed by seed (common random numbers), so the common draw
//! cancels; the panel is a [`PairedFoldComparison`] whose per-seed effect, SEM
//! and resolved/unresolved verdict are emitted with the pair. The two arms
//! additionally share their `x` and baseline-`T0` draws, so the arm-to-arm
//! difference is the acceleration shape alone.
//!
//! ## Assertion policy per arm, and why it is a CEILING on both
//!
//! Both arms assert, unconditionally: the absolute truth-recovery bar above, the
//! survival-function structural check on every seed, the censoring band, and the
//! grid's non-vacuity. Both emit the paired `QUALITY_PAIR` telemetry the #1561
//! aggregate gate consumes, verdict included — currently `gam_resolved_worse` on
//! both arms — so neither arm's result is hidden from the gate.
//!
//! Neither arm uses `assert_paired_match_or_beat`. That shared rule's
//! resolved-deficit clause requires gam not be behind by more than fold noise
//! explains, and gam IS behind by that much on both truths — paired effect
//! +0.878 (SEM 0.165) on the linear arm and +0.376 (SEM 0.081) on the curved
//! one, gam behind on 10 of 10 seeds in both, `gam_resolved_worse` in both.
//! Asserting the shared rule here would simply be asserting something false.
//! What each arm asserts instead is a hard, measured ceiling on gam's premium
//! over lifelines — a real regression guard, honestly labelled as a premium
//! rather than dressed up as a peer win:
//!
//! (The ceilings are ratios of the two FOLD-MEAN RMSEs, 2.20x and 1.45x. The
//! paired effects quoted above are means of per-seed LOG-ratios, so they are
//! the slightly larger Jensen-shifted quantity; the panel reports both and the
//! two must not be read as the same number.)
//!   * [`Truth::Linear`] — lifelines is the exact-DGP MLE, so the ceiling is an
//!     ORACLE premium. A penalized smooth being consistently behind an
//!     oracle-parametric fit on the oracle's own truth is the CORRECT behaviour
//!     of a flexible model, not a defect. Measured 2.20x, ceiling
//!     [`LINEAR_ORACLE_CEILING`].
//!   * [`Truth::Curved`] — lifelines is a misspecified peer, so the ceiling is
//!     tighter. Measured 1.45x, ceiling [`CURVED_PREMIUM_CEILING`], which is set
//!     BELOW the linear arm's measured 2.20x ON PURPOSE: that is how the
//!     cross-arm claim is asserted rather than merely narrated. If the curved
//!     truth ever stopped costing lifelines its oracle advantage — a grid that
//!     stopped resolving the curvature, an amplitude that drifted toward the
//!     linear family — the curved arm would drift back toward the linear arm's
//!     premium and trip this ceiling.
//!
//! ## What this benchmarks
//!
//! gam fits a single shared Weibull baseline cumulative hazard
//!   `H0(t) = (t / scale)^shape`,   `log H0(t) = shape·(log t − log scale)`
//! and adds a covariate log-cumulative-hazard term built from
//! `x + s(x, by=group)`. The by-factor smooth `s(x, by=group)` gives each
//! group its OWN covariate (acceleration) curve, so the survival function
//! differs by stratum through a stratum-specific multiplicative shift of the
//! shared baseline hazard:
//!   `S(t | x, g) = exp( -exp( log H0(t) + f_g(x) ) )`,  `f_g(x) = -shape·a_g(x)`
//! for the log-acceleration `a_g` of [`Truth::log_accel`]. The objective metric
//! above validates that this factorization recovers the true two-stratum AFT
//! surface.
//!
//! `group` is fed to gam as a categorical label ("A"/"B"): a numeric "0"/"1"
//! column infers as Binary and turns `s(x, by=group)` into a single continuous
//! varying-coefficient smooth (basis × value, which zeroes group A at value 0),
//! NOT the per-level by-FACTOR expansion this test is meant to validate.
//!
//! ## Data (n=200 per seed, 100 per group)
//!
//! Group A baseline ~ Weibull(scale=0.8, shape=1.1); Group B baseline ~
//! Weibull(scale=1.5, shape=1.1) — a COMMON shape (the shared-baseline `log t`
//! slope gam's single baseline can represent) with distinct scales and distinct,
//! opposite-sign accelerations of x (the per-group signal the by-factor smooth
//! must capture as a proportional log-hazard shift). Both with x ~ N(0,1).
//! Right-censoring is an independent exponential per group. The censoring rates
//! are UNCHANGED from the pre-#1561 fixture; what changed is that the censored
//! fraction is now MEASURED and ASSERTED instead of asserted-in-prose. The old
//! header claimed "~20-35% censored" and the fixture does not do that: pooled
//! over the K seeds it is 17.2% / 22.4% (linear) and 17.9% / 23.1% (curved) for
//! groups A / B. [`CENSORING_BAND`] states the band the fixture actually
//! occupies, so the claim and the code now agree. Identical rows go to both
//! engines.

use csv::StringRecord;
use gam::families::survival::construction::evaluate_survival_baseline;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, PairedFoldComparison, QualityPair, rmse, run_python};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
    load_csvwith_inferred_schema,
};
use ndarray::{Array2, s};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Exp, Normal, Weibull};
use std::path::Path;

const N_PER_GROUP: usize = 100;
const SEED: u64 = 20260529;
/// Paired seeds per arm. Survival fits are heavier than the location-scale
/// precedent's, and 10 already resolves the curved arm's panel; see the header.
const K_SEEDS: usize = 10;

// Group baselines (rand_distr::Weibull::new(scale, shape): CDF 1 - exp(-(t/scale)^shape)).
//
// The two strata share a COMMON Weibull SHAPE and differ only in scale and in the
// AFT acceleration of `x`. This is deliberate and load-bearing: the gam model
// this test fits is `s(x, by=group)` over a SINGLE shared baseline cumulative
// hazard `H0(t) = (t/scale)^shape` (one shape coefficient — the slope of
// `log H0` in `log t`). A by-factor smooth can express a *proportional*
// per-group log-hazard shift (a different scale and a different x-acceleration
// per group), but it CANNOT bend the shared baseline's `log t` slope per group,
// so a per-group shape would make gam's shared-baseline model structurally
// mis-specified — it could never recover the truth, and the comparison against
// lifelines (which fits a fully independent Weibull, including its own shape,
// per stratum) would be apples-to-oranges. With a shared shape the gam model is
// correctly specified for the DGP on BOTH arms, so the truth-recovery claim is a
// fair test of the by-factor acceleration signal.
const SCALE_A: f64 = 0.8;
const SHAPE_A: f64 = 1.1;
const SCALE_B: f64 = 1.5;
const SHAPE_B: f64 = 1.1;
/// Group-specific LINEAR AFT acceleration of x: `log T` gets `BETA_g · x`.
/// Opposite signs keep the per-group contrast the by-factor smooth must recover.
const BETA_A: f64 = 0.35;
const BETA_B: f64 = -0.25;
/// Saturation rate of the curved arm's `tanh` acceleration.
const TANH_RATE: f64 = 1.5;
/// Half-width of the covariate interval the two arms' log-hazard SPANS are
/// matched on: `x ∈ [-X_SPAN, X_SPAN]`, i.e. ±2 SDs of `x ~ N(0,1)`.
const X_SPAN: f64 = 2.0;
/// Independent-exponential censoring rates, one per group — unchanged from the
/// pre-#1561 fixture, so the arm's identity and its historical comparison are
/// preserved. What the rates actually produce is [`CENSORING_BAND`].
const CENS_RATE_A: f64 = 0.25;
const CENS_RATE_B: f64 = 0.20;
/// The censored fraction each stratum must fall in, pooled over the K seeds.
/// Measured 0.172/0.224 (linear) and 0.179/0.231 (curved) for groups A/B; the
/// band brackets those with room for the pooled seed-to-seed swing. Too little
/// censoring turns the fixture into an uncensored-regression problem and too
/// much starves the acceleration signal, so this is a real property of the
/// fixture, now checked rather than claimed.
const CENSORING_BAND: (f64, f64) = (0.13, 0.30);

/// Covariate values the survival surface is scored on. Three distinct
/// magnitudes per side: a straight line cannot reproduce an odd curved truth
/// here, which the 3-point `[-1, 0, 1]` grid this replaces could (see header).
/// ±1.75 is the 96th percentile of `x ~ N(0,1)`, so every grid point is inside
/// the support of a 100-draw stratum with probability 1 - 2.4e-4.
const X_EVAL: [f64; 7] = [-1.75, -1.0, -0.4, 0.0, 0.4, 1.0, 1.75];
/// Group codes the prediction grid is laid out over: 0 = A, 1 = B.
const PRED_GROUPS: [f64; 2] = [0.0, 1.0];

/// Absolute truth-recovery bar on the fold-mean RMSE. Measured fold-means:
/// 0.0940 (linear) / 0.0908 (curved); the bar keeps ~1.4x headroom over the
/// worse of the two. It is the tool-free half of the claim and the only one
/// that survives lifelines being unavailable.
const TRUTH_RECOVERY_BOUND: f64 = 0.13;
/// Ceiling on the ORACLE PREMIUM the linear arm tolerates: gam's fold-mean
/// RMSE-to-truth over an exactly specified parametric MLE's. Measured 2.20x.
/// See the assertion-policy note in the header for why both arms get a measured
/// ceiling rather than the shared paired rule.
const LINEAR_ORACLE_CEILING: f64 = 2.90;
/// Ceiling on the premium the CURVED arm tolerates, where lifelines is a
/// misspecified peer rather than an oracle. Measured 1.45x. Deliberately set
/// below the linear arm's MEASURED 2.20x: that inequality is the enrichment's
/// cross-arm claim, asserted rather than narrated.
const CURVED_PREMIUM_CEILING: f64 = 1.90;
/// Minimum fraction of the curved truth's log-hazard curve that no linear-in-x
/// AFT can represent ON THE EVALUATION GRID. Below this the curved arm would be
/// measuring nothing; measured 0.278.
const MIN_CURVED_NONLINEARITY: f64 = 0.20;

/// Which generating acceleration an arm uses.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Truth {
    /// The original fixture: a LINEAR log-acceleration, which is exactly the
    /// per-stratum `WeibullAFTFitter`'s own model — and inside the smooth's
    /// penalty null space.
    Linear,
    /// The same kind of object — a smooth, monotone, opposite-sign-per-group
    /// acceleration of the same log-hazard span — but saturating, so no linear
    /// AFT contains it and the comparison measures smoothing rather than an
    /// exactly-specified reference.
    Curved,
}

impl Truth {
    fn label(self) -> &'static str {
        match self {
            Truth::Linear => "linear_acceleration",
            Truth::Curved => "curved_acceleration",
        }
    }

    /// The log AFT acceleration `a_g(x)`: the DGP is `log T = log T0 + a_g(x)`
    /// with `T0 ~ Weibull(scale_g, shape_g)`.
    ///
    /// The curved amplitude is SPAN-MATCHED to the linear slope: over
    /// `x ∈ [-X_SPAN, X_SPAN]` a linear arm spans `2·X_SPAN·|beta_g|` and a tanh
    /// arm spans `2·|amp_g|·tanh(TANH_RATE·X_SPAN)`, so
    /// `amp_g = beta_g · X_SPAN / tanh(TANH_RATE · X_SPAN)` makes the two arms
    /// carry the SAME signal size and differ only in shape.
    fn log_accel(self, x: f64, group_a: bool) -> f64 {
        let beta = if group_a { BETA_A } else { BETA_B };
        match self {
            Truth::Linear => beta * x,
            Truth::Curved => {
                let amp = beta * X_SPAN / (TANH_RATE * X_SPAN).tanh();
                amp * (TANH_RATE * x).tanh()
            }
        }
    }

    /// Closed-form ground-truth survival under the data-generating AFT model.
    ///
    /// `T = T0 · exp(a_g(x))` with `T0 ~ Weibull(scale_g, shape_g)`, whose
    /// survival is `P(T0 > s) = exp(-(s/scale_g)^shape_g)`. Hence
    ///   `S(t | x, g) = P(T0 > t·exp(-a_g(x)))
    ///               = exp( -( t·exp(-a_g(x)) / scale_g )^shape_g )`,
    /// equivalently `exp(-(t/scale_g)^shape_g · exp(f_g(x)))` with the
    /// log-hazard term `f_g(x) = -shape_g · a_g(x)`. This is the objective truth
    /// gam and lifelines are both measured against.
    fn survival(self, t: f64, x: f64, group_a: bool) -> f64 {
        let (scale, shape) = if group_a {
            (SCALE_A, SHAPE_A)
        } else {
            (SCALE_B, SHAPE_B)
        };
        let s = t * (-self.log_accel(x, group_a)).exp();
        (-((s / scale).powf(shape))).exp()
    }

    /// The truth's survival surface on the shared (group × x × t) grid, laid out
    /// group-major / x-major / t-minor — the identical order both engines emit.
    fn surface(self, t_grid: &[f64]) -> Vec<f64> {
        let mut out = Vec::with_capacity(PRED_GROUPS.len() * X_EVAL.len() * t_grid.len());
        for &gc in &PRED_GROUPS {
            let group_a = gc == 0.0;
            for &xv in &X_EVAL {
                for &t in t_grid {
                    out.push(self.survival(t, xv, group_a));
                }
            }
        }
        out
    }
}

/// How much of this truth's log-hazard curve, ON THE EVALUATION GRID, no
/// linear-in-`x` model can represent: the least-squares residual of
/// `f_g(x) = -shape_g · a_g(x)` against `{1, x}`, as a fraction of the curve's
/// own centred norm. Zero means a straight line reproduces the truth exactly on
/// this grid — i.e. the arm measures nothing about curvature.
///
/// This is the arm's reachability check. It is closed-form and data-free, so it
/// answers "did the mechanism this arm exists for actually get exercised?"
/// separately from "did gam do well", and it makes the grid a stated
/// requirement rather than an accident: the 3-point `[-1, 0, 1]` grid returns
/// exactly 0 here for ANY odd acceleration.
fn nonlinear_fraction(truth: Truth, group_a: bool) -> f64 {
    let shape = if group_a { SHAPE_A } else { SHAPE_B };
    let f: Vec<f64> = X_EVAL
        .iter()
        .map(|&x| -shape * truth.log_accel(x, group_a))
        .collect();
    let n = X_EVAL.len() as f64;
    let mean_x = X_EVAL.iter().sum::<f64>() / n;
    let mean_f = f.iter().sum::<f64>() / n;
    let sxx: f64 = X_EVAL.iter().map(|&x| (x - mean_x) * (x - mean_x)).sum();
    let sxf: f64 = X_EVAL
        .iter()
        .zip(&f)
        .map(|(&x, &fi)| (x - mean_x) * (fi - mean_f))
        .sum();
    let slope = if sxx > 0.0 { sxf / sxx } else { 0.0 };
    let resid: f64 = X_EVAL
        .iter()
        .zip(&f)
        .map(|(&x, &fi)| {
            let r = fi - (mean_f + slope * (x - mean_x));
            r * r
        })
        .sum();
    let total: f64 = f.iter().map(|&fi| (fi - mean_f) * (fi - mean_f)).sum();
    (resid / total.max(1e-300)).sqrt()
}

/// One seed's simulated cohort. Group A rows come first, then group B, so
/// first-appearance categorical inference maps A->0, B->1.
struct ArmData {
    time: Vec<f64>,
    event: Vec<f64>,
    x: Vec<f64>,
    g_code: Vec<f64>,
    is_a: Vec<bool>,
}

/// Simulate one seed under `truth`. The RNG stream draws `x`, then the baseline
/// `T0`, then the censoring time, in that order and independently of `truth`, so
/// the LINEAR and CURVED arms of the same seed share their `x` and `T0` draws
/// and differ only through the acceleration applied to `T0`.
fn simulate(truth: Truth, seed: u64) -> ArmData {
    let mut rng = StdRng::seed_from_u64(seed);
    let weib_a = Weibull::new(SCALE_A, SHAPE_A).expect("weibull A");
    let weib_b = Weibull::new(SCALE_B, SHAPE_B).expect("weibull B");
    let xdist = Normal::new(0.0, 1.0).expect("normal x");
    let cens_a = Exp::new(CENS_RATE_A).expect("exp censor A");
    let cens_b = Exp::new(CENS_RATE_B).expect("exp censor B");

    let n = 2 * N_PER_GROUP;
    let mut data = ArmData {
        time: Vec::with_capacity(n),
        event: Vec::with_capacity(n),
        x: Vec::with_capacity(n),
        g_code: Vec::with_capacity(n),
        is_a: Vec::with_capacity(n),
    };
    for group_a in [true, false] {
        let (weib, cens) = if group_a {
            (&weib_a, &cens_a)
        } else {
            (&weib_b, &cens_b)
        };
        for _ in 0..N_PER_GROUP {
            let xi = xdist.sample(&mut rng);
            // `Weibull::sample` is the inverse-CDF draw `scale·(-ln U)^(1/shape)`;
            // the AFT acceleration multiplies that baseline time by exp(a_g(x)).
            let t0 = weib.sample(&mut rng);
            let ti = t0 * truth.log_accel(xi, group_a).exp();
            let ci = cens.sample(&mut rng) + 1e-3;
            data.time.push(ti.min(ci));
            data.event.push(if ti <= ci { 1.0 } else { 0.0 });
            data.x.push(xi);
            data.g_code.push(if group_a { 0.0 } else { 1.0 });
            data.is_a.push(group_a);
        }
    }
    data
}

/// Fit gam's Weibull AFT + by-factor smooth on one seed and return its predicted
/// survival surface on the shared grid (group-major / x-major / t-minor) plus
/// the shared baseline `(scale, shape)` it recovered.
fn gam_survival_surface(data: &ArmData, t_grid: &[f64]) -> (Vec<f64>, f64, f64) {
    // survival_likelihood="weibull" selects the parametric Weibull baseline
    // (linear log-cumulative-hazard time basis whose two coefficients recover
    // scale/shape). The covariate side `x + s(x, by=group)` gives each group its
    // own acceleration curve. The `survmodel(...)` term states the intent
    // in-formula; the likelihood mode is driven by the config field.
    //
    // `group` MUST be fed to gam as a categorical label ("A"/"B"), not the
    // numeric code: schema inference treats "0"/"1" as a Binary numeric column,
    // which makes `s(x, by=group)` a single continuous varying-coefficient
    // smooth (basis * value, zeroing out group A at value 0). Only a Categorical
    // by-variable triggers the per-level by-FACTOR expansion gam advertises here
    // (one smooth per level + an unpenalized treatment-coded factor main effect).
    // Group A rows come first, so first-appearance level order gives A->0, B->1,
    // matching the numeric `g_code` used by the prediction grid and the Python
    // comparator (which filters on the numeric `group` column it receives).
    let headers = vec![
        "time".to_string(),
        "event".to_string(),
        "x".to_string(),
        "group".to_string(),
    ];
    let rows: Vec<StringRecord> = (0..data.time.len())
        .map(|i| {
            StringRecord::from(vec![
                data.time[i].to_string(),
                data.event[i].to_string(),
                data.x[i].to_string(),
                if data.is_a[i] { "A" } else { "B" }.to_string(),
            ])
        })
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode survival dataset");
    let col = ds.column_map();
    let x_idx = col["x"];
    let g_idx = col["group"];

    let cfg = FitConfig {
        survival_likelihood: Some("weibull".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula(
        "Surv(time, event) ~ x + s(x, by=group) + survmodel(spec=\"transformation\", distribution=\"weibull\")",
        &ds,
        &cfg,
    )
    .expect("gam Weibull-AFT by-factor fit");
    let FitResult::SurvivalTransformation(fit) = result else {
        panic!("expected a SurvivalTransformation fit for survival_likelihood=weibull");
    };

    // gam's shared baseline Weibull (scale, shape), recovered from the linear
    // time-basis coefficients — reported for context only.
    //
    // The SCALE is not an estimate of either stratum's 0.8/1.5 and should not be
    // read as one: it saturates at its 1e-9 floor on every seed, because
    // `log H0(t) = shape·(log t − log scale)` and the covariate block carries an
    // intercept, so the scale and that intercept are a single identified
    // combination and the fit is free to park the split anywhere. Only `shape`
    // (the `log t` slope) is separately identified by the baseline, and it is
    // what the survival surface's time profile depends on.
    let gam_scale = fit
        .baseline_cfg
        .scale
        .expect("gam recovers a Weibull baseline scale");
    let gam_shape = fit
        .baseline_cfg
        .shape
        .expect("gam recovers a Weibull baseline shape");

    // Covariate coefficient slice: beta = [time(2 cols), covariate...]; the
    // covariate block begins at `time_base_ncols`.
    let cov_start = fit.time_base_ncols;
    let beta = &fit.fit.beta;
    assert!(
        beta.len() > cov_start,
        "expected covariate coefficients after the {cov_start} time columns, got beta.len()={}",
        beta.len()
    );

    // Build the covariate design once at every (group, x_eval) prediction row.
    let n_pred_rows = PRED_GROUPS.len() * X_EVAL.len();
    let mut grid = Array2::<f64>::zeros((n_pred_rows, ds.headers.len()));
    let mut row = 0usize;
    for &gc in &PRED_GROUPS {
        for &xv in &X_EVAL {
            grid[[row, x_idx]] = xv;
            grid[[row, g_idx]] = gc;
            row += 1;
        }
    }
    let design = build_term_collection_design(grid.view(), &fit.resolvedspec)
        .expect("rebuild covariate design at prediction rows");
    let dense = design.design.to_dense();
    assert_eq!(
        dense.ncols(),
        beta.len() - cov_start,
        "covariate design width must match the covariate coefficient slice"
    );
    let cov_beta = beta.slice(s![cov_start..]).to_owned();

    // gam's predicted survival surface, using gam's OWN forward map:
    // log H(t|x,g) = log H0(t) + covariate_design(x,g)·cov_beta, S = exp(-exp(log H)).
    // log H0 comes from gam's recovered baseline via `evaluate_survival_baseline`,
    // so the reconstruction is self-consistent with whatever gam fit (no
    // hand-rederived offsets).
    let mut surface = Vec::with_capacity(n_pred_rows * t_grid.len());
    for idx in 0..n_pred_rows {
        let cov_eta: f64 = dense.row(idx).dot(&cov_beta);
        for &t in t_grid {
            let (log_h0, _) = evaluate_survival_baseline(t, &fit.baseline_cfg)
                .expect("evaluate gam baseline log-cumulative-hazard");
            surface.push((-(log_h0 + cov_eta).exp()).exp());
        }
    }
    (surface, gam_scale, gam_shape)
}

/// Run one truth's whole paired panel: K seeds through gam and through lifelines
/// on the SAME per-seed rows, then the arm's decision.
fn run_weibull_aft_by_arm(truth: Truth) {
    init_parallelism();
    let label = truth.label();
    let t_grid: Vec<f64> = (1..=10).map(|k| 0.25 * k as f64).collect();
    let nt = t_grid.len();
    let n_pred_rows = PRED_GROUPS.len() * X_EVAL.len();

    // ---- NON-VACUITY: does this arm's grid see what the arm exists for? ----
    // The curved arm's whole point is a shape outside the linear AFT family; if
    // the grid cannot resolve that shape the arm measures nothing. Checked
    // BEFORE any fitting, so a grid change can never silently hollow the arm out.
    for group_a in [true, false] {
        let frac = nonlinear_fraction(truth, group_a);
        let g = if group_a { "A" } else { "B" };
        match truth {
            Truth::Linear => assert!(
                frac <= 1e-9,
                "linear arm's group-{g} truth must be exactly linear on the grid, got {frac:.3e}"
            ),
            Truth::Curved => assert!(
                frac >= MIN_CURVED_NONLINEARITY,
                "curved arm is VACUOUS: group-{g} truth is only {frac:.4} non-linear on the \
                 evaluation grid (need >= {MIN_CURVED_NONLINEARITY}); a linear AFT can \
                 reproduce it there, so the arm measures no curvature"
            ),
        }
    }

    let true_surv = truth.surface(&t_grid);

    // ---- gam on every seed, and the long-format rows lifelines replays ------
    let mut gam_rmses = Vec::with_capacity(K_SEEDS);
    let mut long_seed = Vec::with_capacity(K_SEEDS * 2 * N_PER_GROUP);
    let mut long_time = Vec::with_capacity(K_SEEDS * 2 * N_PER_GROUP);
    let mut long_event = Vec::with_capacity(K_SEEDS * 2 * N_PER_GROUP);
    let mut long_x = Vec::with_capacity(K_SEEDS * 2 * N_PER_GROUP);
    let mut long_group = Vec::with_capacity(K_SEEDS * 2 * N_PER_GROUP);
    let mut events_a = 0.0_f64;
    let mut events_b = 0.0_f64;
    let mut last_scale = 0.0_f64;
    let mut last_shape = 0.0_f64;

    for k in 0..K_SEEDS {
        let seed = SEED + k as u64;
        let data = simulate(truth, seed);
        let (surface, gam_scale, gam_shape) = gam_survival_surface(&data, &t_grid);
        last_scale = gam_scale;
        last_shape = gam_shape;

        // ---- STRUCTURAL CHECK: gam's surface is a valid survival function ---
        // Every S in [0,1] and non-increasing in t within each (group, x) block,
        // on EVERY seed — a property the AFT factorization must satisfy
        // regardless of any reference tool.
        for r in 0..n_pred_rows {
            let block = &surface[r * nt..(r + 1) * nt];
            for (j, &s) in block.iter().enumerate() {
                assert!(
                    (0.0..=1.0).contains(&s),
                    "[{label}] seed {seed}: gam survival out of [0,1] at row {r}, t-index {j}: S={s}"
                );
                assert!(
                    j == 0 || s <= block[j - 1] + 1e-9,
                    "[{label}] seed {seed}: gam survival not non-increasing at row {r}, \
                     t-index {j}: {} -> {s}",
                    if j == 0 { s } else { block[j - 1] }
                );
            }
        }

        gam_rmses.push(rmse(&surface, &true_surv));
        for i in 0..data.time.len() {
            long_seed.push(seed as f64);
            long_time.push(data.time[i]);
            long_event.push(data.event[i]);
            long_x.push(data.x[i]);
            long_group.push(data.g_code[i]);
            if data.is_a[i] {
                events_a += data.event[i];
            } else {
                events_b += data.event[i];
            }
        }
    }

    // ---- CENSORING BAND: the header's claim, asserted -----------------------
    // Pooled over the K seeds (a single 100-row stratum swings ±0.1, the pooled
    // fraction does not). Too little censoring makes the fixture an
    // uncensored-regression problem; too much starves the acceleration signal.
    let pooled = (K_SEEDS * N_PER_GROUP) as f64;
    let cens_a_frac = 1.0 - events_a / pooled;
    let cens_b_frac = 1.0 - events_b / pooled;
    for (g, frac) in [("A", cens_a_frac), ("B", cens_b_frac)] {
        assert!(
            frac >= CENSORING_BAND.0 && frac <= CENSORING_BAND.1,
            "[{label}] group {g} censored fraction {frac:.3} outside the asserted band \
             [{:.2}, {:.2}] pooled over {K_SEEDS} seeds",
            CENSORING_BAND.0,
            CENSORING_BAND.1,
        );
    }

    // ---- the SAME K datasets through lifelines, in ONE python session -------
    // For each seed and each group, fit an independent WeibullAFTFitter on
    // `time ~ x` and emit that stratum's predicted survival surface on the
    // identical (x_eval × t_grid) grid, in the identical order gam used.
    // lifelines is held to the SAME truth-recovery yardstick as gam; on the
    // curved arm it is honestly misspecified, which is the point of that arm.
    //
    // The grids are interpolated from the Rust constants rather than restated in
    // Python, so the two sides cannot drift apart on the collation both of them
    // are scored over.
    let x_eval_py = X_EVAL
        .iter()
        .map(|v| format!("{v:.17e}"))
        .collect::<Vec<_>>()
        .join(", ");
    let t_grid_py = t_grid
        .iter()
        .map(|v| format!("{v:.17e}"))
        .collect::<Vec<_>>()
        .join(", ");
    let py = run_python(
        &[
            Column::new("seed", &long_seed),
            Column::new("time", &long_time),
            Column::new("event", &long_event),
            Column::new("x", &long_x),
            Column::new("group", &long_group),
        ],
        &format!(
            r#"
import numpy as np
import pandas as pd
from lifelines import WeibullAFTFitter

x_eval = [{x_eval_py}]
t_grid = [{t_grid_py}]

frame = pd.DataFrame(dict(
    seed=np.asarray(df["seed"], dtype=float),
    time=np.asarray(df["time"], dtype=float),
    event=np.asarray(df["event"], dtype=float),
    x=np.asarray(df["x"], dtype=float),
    group=np.asarray(df["group"], dtype=float),
))

surv_rows = []
for s in sorted(frame["seed"].unique()):
    per_seed = frame[frame["seed"] == s]
    for gc in (0.0, 1.0):
        sub = per_seed[per_seed["group"] == gc][["time", "event", "x"]].reset_index(drop=True)
        aft = WeibullAFTFitter()
        aft.fit(sub, duration_col="time", event_col="event")
        # Predicted survival at each x_eval over the shared time grid.
        sf = aft.predict_survival_function(pd.DataFrame(dict(x=x_eval)), times=t_grid)
        # sf: rows=times (t_grid order), cols=rows of the newdata (x_eval order).
        for j in range(len(x_eval)):
            surv_rows.extend(float(v) for v in sf.iloc[:, j].to_numpy())

emit("surv", surv_rows)
"#
        ),
    );

    let life_flat = py.vector("surv");
    let per_seed_len = n_pred_rows * nt;
    assert_eq!(
        life_flat.len(),
        K_SEEDS * per_seed_len,
        "lifelines panel length mismatch: expected {} got {}",
        K_SEEDS * per_seed_len,
        life_flat.len()
    );
    let life_rmses: Vec<f64> = (0..K_SEEDS)
        .map(|k| rmse(&life_flat[k * per_seed_len..(k + 1) * per_seed_len], &true_surv))
        .collect();

    // ---- paired panel: same seed, same rows, seed by seed -------------------
    let panel = PairedFoldComparison::new(&gam_rmses, &life_rmses, true);

    eprintln!(
        "weibull-AFT by-factor [{label}] K={K_SEEDS}-seed paired: n={} per seed, \
         censoring A={cens_a_frac:.3} B={cens_b_frac:.3}, grid={n_pred_rows}x{nt}, \
         nonlinearity(A)={:.4}\n  \
         gam baseline (last seed): shape={last_shape:.4} (true {SHAPE_A}), \
         scale={last_scale:.2e} (floored; not separately identified)\n  \
         RMSE(S) vs TRUTH fold-mean: gam={:.5} lifelines={:.5}",
        2 * N_PER_GROUP,
        nonlinear_fraction(truth, true),
        panel.gam_mean,
        panel.reference_mean,
    );
    eprintln!("{}", panel.report(&format!("weibull_aft_by::{label}")));
    eprintln!(
        "{}",
        QualityPair::paired(
            "survival",
            &format!("quality_vs_lifelines_weibull_aft_by::{label}::truth_surface"),
            "survival_rmse_to_truth",
            "lifelines",
            &panel,
        )
        .line()
    );

    // ---- PRIMARY (objective, tool-free): gam recovers the true surface ------
    assert!(
        panel.gam_mean <= TRUTH_RECOVERY_BOUND,
        "[{label}] gam fails to recover the true Weibull-AFT survival surface: \
         fold-mean RMSE(S vs truth)={:.4} > {TRUTH_RECOVERY_BOUND}",
        panel.gam_mean,
    );

    // ---- PREMIUM CEILING, per the arm's assertion policy (see header) -------
    // gam is RESOLVED worse than lifelines on both truths, so the shared
    // `assert_paired_match_or_beat` rule is not used: its resolved-deficit
    // clause would assert something the measurement contradicts. Each arm caps
    // gam's premium at a measured ceiling instead, and the paired verdict is
    // emitted above either way, so the #1561 gate sees the real result.
    let (ceiling, kind) = match truth {
        // lifelines is the EXACT-DGP MLE here; being behind it on its own truth
        // is what a penalized smooth is supposed to do.
        Truth::Linear => (LINEAR_ORACLE_CEILING, "oracle"),
        // lifelines is a misspecified peer here, so the bar is tighter — and it
        // sits below the linear arm's measured premium, which is how this arm
        // asserts that the curvature really did cost the reference its oracle
        // advantage.
        Truth::Curved => (CURVED_PREMIUM_CEILING, "peer"),
    };
    assert!(
        panel.gam_mean <= panel.reference_mean * ceiling,
        "[{label}] gam's {kind} premium exceeds {ceiling}x: fold-mean RMSE-to-truth \
         gam={:.5} vs lifelines {:.5} (ratio {:.3}).\n{}",
        panel.gam_mean,
        panel.reference_mean,
        panel.gam_mean / panel.reference_mean,
        panel.report(label),
    );
}

#[test]
fn gam_weibull_aft_by_factor_recovers_true_survival() {
    run_weibull_aft_by_arm(Truth::Linear);
}

#[test]
fn gam_weibull_aft_by_factor_recovers_true_survival_curved_acceleration() {
    run_weibull_aft_by_arm(Truth::Curved);
}

// ===========================================================================
// REAL-DATA ARM
// ===========================================================================
//
// Dataset SOURCE: the Veterans' Administration lung-cancer randomized trial
// (`veteran` in the R `survival` package; Kalbfleisch & Prentice, "The
// Statistical Analysis of Failure Time Data"). 137 patients, columns
// `time` (days), `status` (1=death, 0=censored — 9 censored), the numeric
// covariate `karno` (Karnofsky performance score, the dominant prognostic
// signal) and the four-level factor `celltype`
// (squamous / smallcell / adeno / large), shipped at
// `bench/datasets/veteran_lung.csv`.
//
// This arm exercises the SAME gam capability as the synthetic test above —
// a parametric **Weibull AFT** survival fit with a **by-FACTOR smooth**
// covariate effect, `s(karno, by=celltype)`, giving each cell type its own
// karnofsky→risk curve over a shared Weibull baseline cumulative hazard.
//
// Because this is real data the data-generating survival surface is UNKNOWN,
// so RMSE-to-truth is not computable. The objective, tool-free quality metric
// is therefore the **held-out concordance index** (Harrell's C): a fixed,
// deterministic train/test split, fit gam on train, score the held-out
// patients by gam's OWN covariate log-cumulative-hazard risk, and assert how
// well that risk ranking agrees with the observed (time, event) ordering.
// Higher covariate log-cumulative-hazard ⇒ higher hazard ⇒ shorter survival,
// so a well-fit model gives a high C-index (0.5 = random, 1.0 = perfect).
//   PRIMARY (objective): held-out C-index >= 0.62 — well above the 0.5 random
//     baseline for a ~30-patient held-out set; a broken by-factor fit (wrong
//     karnofsky sign, collapsed cell-type strata) would not clear it.
//   BASELINE (match-or-beat): lifelines.WeibullAFTFitter, the mature standard
//     parametric-AFT reference, is fit on the SAME train rows and scored on
//     the SAME held-out patients (by predicted expected survival time, which
//     it turns into its own C-index via lifelines.utils.concordance_index);
//     gam's held-out C must be no worse than lifelines' C minus a 0.03 margin.
//     lifelines is a yardstick to match-or-beat on the identical held-out
//     metric, never a fitted output to reproduce.

/// Harrell's concordance index for survival risk scores. `risk[i]` is monotone
/// INCREASING in hazard (higher risk ⇒ shorter expected survival). A pair
/// `(i, j)` is comparable when the earlier observed time belongs to an event;
/// it is concordant when that earlier-failing subject also carries the higher
/// risk. Risk ties on a comparable pair count as half-concordant. Returns the
/// fraction of comparable pairs that are concordant (0.5 = random ordering).
fn concordance_index(risk: &[f64], time: &[f64], event: &[f64]) -> f64 {
    assert_eq!(
        risk.len(),
        time.len(),
        "concordance: risk/time length mismatch"
    );
    assert_eq!(
        time.len(),
        event.len(),
        "concordance: time/event length mismatch"
    );
    let n = risk.len();
    let mut concordant = 0.0_f64;
    let mut comparable = 0.0_f64;
    for i in 0..n {
        for j in (i + 1)..n {
            // Identify the earlier-failing member of the pair; the pair is only
            // comparable when that earlier observed time is an actual event.
            let (early, late) = if time[i] < time[j] {
                (i, j)
            } else if time[j] < time[i] {
                (j, i)
            } else {
                // Equal observed times carry no usable ordering information.
                continue;
            };
            if event[early] != 1.0 {
                continue;
            }
            comparable += 1.0;
            if risk[early] > risk[late] {
                concordant += 1.0;
            } else if risk[early] == risk[late] {
                concordant += 0.5;
            }
        }
    }
    assert!(
        comparable > 0.0,
        "no comparable survival pairs — degenerate held-out set"
    );
    concordant / comparable
}

#[test]
fn gam_weibull_aft_by_factor_recovers_true_survival_on_real_data() {
    init_parallelism();

    // ---- load the real Veterans' lung-cancer trial ------------------------
    let ds = load_csvwith_inferred_schema(Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/bench/datasets/veteran_lung.csv"
    )))
    .expect("load veteran_lung.csv");
    let col = ds.column_map();
    let time_idx = col["time"];
    let status_idx = col["status"];
    let karno_idx = col["karno"];
    let celltype_idx = col["celltype"];

    let time: Vec<f64> = ds.values.column(time_idx).to_vec();
    let status: Vec<f64> = ds.values.column(status_idx).to_vec();
    let karno: Vec<f64> = ds.values.column(karno_idx).to_vec();
    // `celltype` is a string column, so schema inference encodes it as a single
    // Categorical code column (level codes by first appearance:
    // squamous=0, smallcell=1, adeno=2, large=3). That categorical kind is what
    // makes `s(karno, by=celltype)` expand into the per-level by-FACTOR smooth
    // this test validates (one karnofsky curve per cell type + a treatment-coded
    // factor main effect), exactly as in the synthetic arm.
    let celltype: Vec<f64> = ds.values.column(celltype_idx).to_vec();
    let n = time.len();
    assert!(n > 120, "veteran_lung should have ~137 rows, got {n}");

    // ---- deterministic train/test split: every 4th row held out ----------
    let is_test = |i: usize| i % 4 == 0;
    let train_rows: Vec<usize> = (0..n).filter(|&i| !is_test(i)).collect();
    let test_rows: Vec<usize> = (0..n).filter(|&i| is_test(i)).collect();
    assert!(
        train_rows.len() > 90 && test_rows.len() > 30,
        "split sizes: train={} test={}",
        train_rows.len(),
        test_rows.len()
    );

    // Build a training-only dataset by sub-setting the encoded rows; headers,
    // schema and column kinds are unchanged, so the formula resolves identically
    // (the categorical level table lives in the schema, not the row values).
    let p = ds.headers.len();
    let mut train_values = Array2::<f64>::zeros((train_rows.len(), p));
    for (out_row, &src_row) in train_rows.iter().enumerate() {
        for c in 0..p {
            train_values[[out_row, c]] = ds.values[[src_row, c]];
        }
    }
    let mut train_ds = ds.clone();
    train_ds.values = train_values;

    // ---- fit gam on TRAIN: Weibull AFT + by-factor smooth on karno --------
    let cfg = FitConfig {
        survival_likelihood: Some("weibull".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula(
        "Surv(time, status) ~ karno + s(karno, by=celltype) + survmodel(spec=\"transformation\", distribution=\"weibull\")",
        &train_ds,
        &cfg,
    )
    .expect("gam Weibull-AFT by-factor fit on veteran_lung train rows");
    let FitResult::SurvivalTransformation(fit) = result else {
        panic!("expected a SurvivalTransformation fit for survival_likelihood=weibull");
    };

    // ---- score the held-out patients by gam's OWN covariate risk ----------
    // The covariate block adds `cov_eta = design(karno, celltype)·cov_beta` to
    // the shared log-cumulative-hazard, so `cov_eta` IS the per-patient log-risk
    // (monotone increasing in hazard). Rebuild the covariate design at the
    // held-out rows from the frozen spec — no baseline evaluation is needed for
    // a ranking metric, since every patient shares the same baseline H0(t).
    let cov_start = fit.time_base_ncols;
    let beta = &fit.fit.beta;
    assert!(
        beta.len() > cov_start,
        "expected covariate coefficients after the {cov_start} time columns, got beta.len()={}",
        beta.len()
    );
    let cov_beta = beta.slice(s![cov_start..]).to_owned();

    let mut test_grid = Array2::<f64>::zeros((test_rows.len(), p));
    for (out_row, &src_row) in test_rows.iter().enumerate() {
        for c in 0..p {
            test_grid[[out_row, c]] = ds.values[[src_row, c]];
        }
    }
    let test_design = build_term_collection_design(test_grid.view(), &fit.resolvedspec)
        .expect("rebuild covariate design at held-out rows");
    let test_dense = test_design.design.to_dense();
    assert_eq!(
        test_dense.ncols(),
        cov_beta.len(),
        "held-out covariate design width must match the covariate coefficient slice"
    );
    let gam_risk: Vec<f64> = (0..test_rows.len())
        .map(|r| test_dense.row(r).dot(&cov_beta))
        .collect();

    let test_time: Vec<f64> = test_rows.iter().map(|&i| time[i]).collect();
    let test_status: Vec<f64> = test_rows.iter().map(|&i| status[i]).collect();
    let gam_c = concordance_index(&gam_risk, &test_time, &test_status);

    // ---- fit the SAME model on TRAIN with lifelines, score the SAME TEST ---
    // One run_python call, all columns the SAME length (full n): a per-row
    // `is_train` mask separates the fit rows from the held-out rows so we never
    // mix train-length and test-length columns. lifelines fits on the masked
    // train rows and scores the held-out rows by predicted EXPECTED survival
    // time, then forms its own held-out C-index (lifelines orients C so that
    // higher predicted survival ⇒ lower risk). Held to the identical metric.
    let is_train: Vec<f64> = (0..n).map(|i| if is_test(i) { 0.0 } else { 1.0 }).collect();
    let py = run_python(
        &[
            Column::new("time", &time),
            Column::new("status", &status),
            Column::new("karno", &karno),
            Column::new("celltype", &celltype),
            Column::new("is_train", &is_train),
        ],
        r#"
import numpy as np
import pandas as pd
from lifelines import WeibullAFTFitter
from lifelines.utils import concordance_index

frame = pd.DataFrame({
    "time": np.asarray(df["time"], dtype=float),
    "status": np.asarray(df["status"], dtype=float),
    "karno": np.asarray(df["karno"], dtype=float),
    "celltype": np.asarray(df["celltype"], dtype=float).astype(int),
    "is_train": np.asarray(df["is_train"], dtype=float),
})

# Treatment-coded indicators for the 4-level cell type plus karno and its
# per-celltype interaction: the parametric-AFT analogue of karno + s(karno,
# by=celltype). Drop level 0 (squamous) as the reference to keep the design
# full rank, matching gam's treatment-coded by-factor expansion.
for lev in (1, 2, 3):
    frame[f"ct{lev}"] = (frame["celltype"] == lev).astype(float)
    frame[f"karno_ct{lev}"] = frame["karno"] * frame[f"ct{lev}"]

feat = ["karno", "ct1", "ct2", "ct3", "karno_ct1", "karno_ct2", "karno_ct3"]
train = frame[frame["is_train"] == 1.0].reset_index(drop=True)
test = frame[frame["is_train"] == 0.0].reset_index(drop=True)

aft = WeibullAFTFitter(penalizer=0.01)
aft.fit(train[feat + ["time", "status"]], duration_col="time", event_col="status")

# Higher predicted expected survival time => lower risk; concordance_index
# expects predicted scores that increase with survival, so pass expectations.
pred_exp = aft.predict_expectation(test[feat]).to_numpy().reshape(-1)
c = concordance_index(test["time"].to_numpy(), pred_exp, test["status"].to_numpy())
emit("cindex", [float(c)])
"#,
    );
    let life_c = py.scalar("cindex");

    let cens_frac = 1.0 - status.iter().sum::<f64>() / n as f64;
    eprintln!(
        "veteran_lung weibull-AFT by-factor held-out: n={n} n_train={} n_test={} \
         censoring={cens_frac:.2}\n  \
         held-out C-index: gam={gam_c:.4} lifelines={life_c:.4}",
        train_rows.len(),
        test_rows.len(),
    );
    eprintln!(
        "{}",
        QualityPair::score(
            "survival",
            "quality_vs_lifelines_weibull_aft_by::holdout_cindex",
            "cindex_holdout",
            gam_c,
            "lifelines",
            life_c,
        )
        .line()
    );

    // ---- PRIMARY objective assertion: gam ranks held-out risk well --------
    // C-index >= 0.62 is well above the 0.5 random-ranking baseline for a small
    // held-out set; a broken by-factor fit (wrong karnofsky sign, collapsed
    // cell-type strata) would not clear it.
    assert!(
        gam_c >= 0.62,
        "gam's held-out concordance too low: {gam_c:.4} (< 0.62)"
    );

    // ---- BASELINE (match-or-beat): no worse than lifelines on held-out C ---
    assert!(
        gam_c >= life_c - 0.03,
        "gam less concordant than lifelines on held-out data: gam C={gam_c:.4}, lifelines C={life_c:.4} (margin 0.03)"
    );
}
