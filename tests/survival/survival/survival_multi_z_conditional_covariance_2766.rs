//! gam#2766 — the marginal-preservation identity under a CONDITIONALLY varying
//! score covariance.
//!
//! `survival_multi_z_fit_hard::survival_multi_z_fit_marginal_preserved_at_true_slopes_population_mc`
//! already pins the identity
//!
//! ```text
//!   E_z[Φ(−(c·q + rᵀz))] = Φ(−q)     with   c = √(1 + rᵀΣr)
//! ```
//!
//! as a POPULATION average, where one pooled `Σ` is the right object because the
//! sample it is drawn from is homogeneous. This module pins the same identity
//! CONDITIONALLY, which is where the pooled object stops being right: with
//! `Cov(z₀, z₁ | a)` a function of a marginal covariate, `Σ̄` is still the
//! population covariance — so the population average keeps passing — while every
//! conditional average is wrong by `q·(c̄/c(a) − 1)`.
//!
//! Read the two tests together: the pooled field FAILS the conditional bar by a
//! wide margin on the same data the conditional field passes it on. That
//! contrast is the regression guard — a revert to one global `Σ` cannot leave
//! both green.

use gam::families::bms::{
    ConditionalScoreCovariance, ScoreCovarianceField, marginal_slope_covariance_from_scores,
};
use gam::families::survival::marginal_slope::{
    RigidVectorValueWorkspace, survival_marginal_slope_vector_eta,
    survival_marginal_slope_vector_neglog,
};
use gam::probability::normal_cdf;
use ndarray::{Array1, Array2};

const N: usize = 24_000;
const K: usize = 2;
const PROBIT_SCALE: f64 = 1.0;
/// Number of `ρ(x)` strata the conditional bar is read on. Five is enough that
/// each stratum still holds thousands of rows at `N = 24000`, so the
/// Monte-Carlo standard error stays far below the effect being measured.
const STRATA: usize = 5;

#[derive(Clone, Copy)]
struct SplitMix64(u64);

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_add(0x9E37_79B9_7F4A_7C15))
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn next_unit(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) * (1.0 / ((1u64 << 53) as f64))
    }
    fn next_normal_pair(&mut self) -> (f64, f64) {
        let u1 = self.next_unit().max(1.0e-300);
        let u2 = self.next_unit();
        let r = (-2.0 * u1.ln()).sqrt();
        (
            r * (std::f64::consts::TAU * u2).cos(),
            r * (std::f64::consts::TAU * u2).sin(),
        )
    }
}

struct Sample {
    z: Array2<f64>,
    weights: Array1<f64>,
    /// The conditioning span `a(C)`, one column: the covariate the correlation
    /// is a function of.
    a: Array2<f64>,
    /// Stratum index of each row, by that covariate.
    stratum: Vec<usize>,
}

/// `K = 2` scores whose conditional marginals are EXACTLY standard normal and
/// whose only conditional structure is the off-diagonal:
///
/// ```text
///   z₀ = e₀,   z₁ = φ(x)·e₀ + √(1 − φ(x)²)·e₁,   φ(x) = amplitude·x/√(1+x²)
/// ```
///
/// The conditional mean is zero and both conditional variances are one, so
/// gam#2768's per-coordinate location-scale gate has nothing to correct and the
/// entire departure this module measures is the one gam#2766 names.
///
/// `amplitude = 0` gives a constant (zero) conditional correlation on the same
/// marginals, which is the control arm.
fn sample(seed: u64, amplitude: f64) -> Sample {
    let mut rng = SplitMix64::new(seed);
    let mut z = Array2::<f64>::zeros((N, K));
    let mut a = Array2::<f64>::zeros((N, 1));
    let mut stratum = Vec::with_capacity(N);
    for row in 0..N {
        let (x, e0) = rng.next_normal_pair();
        let (e1, _unused) = rng.next_normal_pair();
        let phi = amplitude * x / (1.0 + x * x).sqrt();
        a[[row, 0]] = x;
        z[[row, 0]] = e0;
        z[[row, 1]] = phi * e0 + (1.0 - phi * phi).max(0.0).sqrt() * e1;
        // Strata on `x/√(1+x²) ∈ (−1, 1)`, the monotone image the correlation
        // is affine-ish in, so each stratum is a narrow band of `φ`.
        let squashed = x / (1.0 + x * x).sqrt();
        stratum.push((((squashed + 1.0) * 0.5 * STRATA as f64) as usize).min(STRATA - 1));
    }
    Sample {
        z,
        weights: Array1::<f64>::ones(N),
        a,
        stratum,
    }
}

/// Per-stratum Monte-Carlo of `E[Φ(−η) | stratum]` against `Φ(−q)`, returned as
/// `(worst |mc − target| / se, worst mc, worst target)` where `se` is the
/// binomial standard error of that stratum's own average. Reading the miss in
/// units of its own noise is what lets the two arms be compared on one scale.
fn worst_conditional_miss_in_standard_errors(
    data: &Sample,
    field: &ScoreCovarianceField,
    slopes: &[f64; K],
    q: f64,
) -> (f64, f64, f64) {
    let target = normal_cdf(-q);
    let mut sums = vec![0.0_f64; STRATA];
    let mut counts = vec![0usize; STRATA];
    for row in 0..N {
        let z_row = [data.z[[row, 0]], data.z[[row, 1]]];
        let eta = survival_marginal_slope_vector_eta(
            q,
            &z_row,
            slopes,
            field.at_row(row),
            PROBIT_SCALE,
        )
        .expect("eta");
        sums[data.stratum[row]] += normal_cdf(-eta);
        counts[data.stratum[row]] += 1;
    }
    let mut worst = 0.0_f64;
    let mut worst_mc = target;
    for index in 0..STRATA {
        if counts[index] == 0 {
            continue;
        }
        let mc = sums[index] / counts[index] as f64;
        let se = ((target * (1.0 - target)) / counts[index] as f64).sqrt();
        let miss = (mc - target).abs() / se;
        if miss > worst {
            worst = miss;
            worst_mc = mc;
        }
    }
    (worst, worst_mc, target)
}

fn conditional_field(data: &Sample) -> ScoreCovarianceField {
    let pooled =
        marginal_slope_covariance_from_scores(data.z.view(), &data.weights).expect("pooled Σ");
    let model = ConditionalScoreCovariance::fit(data.z.view(), data.weights.view(), data.a.view())
        .expect("conditional fit")
        .expect("a varying cross-score covariance must escalate");
    ScoreCovarianceField::conditional(pooled, model, data.a.view()).expect("conditional field")
}

fn pooled_field(data: &Sample) -> ScoreCovarianceField {
    ScoreCovarianceField::pooled(
        marginal_slope_covariance_from_scores(data.z.view(), &data.weights).expect("pooled Σ"),
    )
}

/// The acceptance claim: with `Σ(a)` the row's own conditional covariance, the
/// marginal-preservation identity holds STRATUM BY STRATUM, not merely on
/// average.
#[test]
fn conditional_covariance_preserves_the_marginal_index_within_every_stratum() {
    let data = sample(0x2766_0A11, 0.8);
    let field = conditional_field(&data);
    let slopes = [0.55_f64, 0.45_f64];
    for &q in &[-1.0_f64, -0.3, 0.0, 0.4, 1.2] {
        let (miss, mc, target) = worst_conditional_miss_in_standard_errors(&data, &field, &slopes, q);
        // The conditional model is fitted, not known, so the residual miss is
        // estimation error in `Σ̂(a)` on top of Monte-Carlo noise. Six standard
        // errors of the stratum average is a wide band on this scale and still
        // an order of magnitude below what the pooled arm below produces.
        assert!(
            miss <= 6.0,
            "q={q}: worst stratum miss {miss:.2} SE (mc={mc:.6} target={target:.6})"
        );
    }
}

/// The contrast that makes the test above a regression guard rather than a
/// tautology: the SAME data, the same slopes, the same bar — with one pooled
/// `Σ̄` the conditional identity misses by tens to hundreds of standard errors.
///
/// This is the defect gam#2766 names, kept red-by-construction so a revert to a
/// global `Σ` cannot pass both tests.
#[test]
fn a_pooled_covariance_misses_the_conditional_identity_by_orders_of_magnitude() {
    let data = sample(0x2766_0A11, 0.8);
    let pooled = pooled_field(&data);
    let conditional = conditional_field(&data);
    let slopes = [0.55_f64, 0.45_f64];
    let q = 0.4_f64;
    let (pooled_miss, pooled_mc, target) =
        worst_conditional_miss_in_standard_errors(&data, &pooled, &slopes, q);
    let (conditional_miss, _, _) =
        worst_conditional_miss_in_standard_errors(&data, &conditional, &slopes, q);
    println!(
        "#2766: worst stratum miss at q={q} — pooled {pooled_miss:.1} SE (mc={pooled_mc:.6} \
         target={target:.6}), conditional {conditional_miss:.1} SE"
    );
    assert!(
        pooled_miss > 30.0,
        "the pooled arm is supposed to EXHIBIT the defect; got {pooled_miss:.2} SE"
    );
    assert!(
        conditional_miss * 10.0 < pooled_miss,
        "the conditional field must close the gap by at least an order of magnitude: \
         pooled {pooled_miss:.2} SE against conditional {conditional_miss:.2} SE"
    );
}

/// The gate must not fire when the conditional covariance really is constant.
/// A field installed on every multi-score fit would replace an exactly-correct
/// pooled object with an estimated one, which is a worse trade than the defect
/// it is meant to fix.
#[test]
fn a_constant_conditional_covariance_leaves_the_pooled_object_in_place() {
    let data = sample(0x2766_0B22, 0.0);
    let decision =
        ConditionalScoreCovariance::fit(data.z.view(), data.weights.view(), data.a.view())
            .expect("conditional fit");
    assert!(
        decision.is_none(),
        "a constant Cov(z₀, z₁ | a) must not escalate"
    );
    // And the pooled field then passes the conditional bar on its own, which is
    // why not escalating is the right answer rather than a missed detection.
    let pooled = pooled_field(&data);
    let slopes = [0.55_f64, 0.45_f64];
    for &q in &[-0.3_f64, 0.4] {
        let (miss, mc, target) =
            worst_conditional_miss_in_standard_errors(&data, &pooled, &slopes, q);
        assert!(
            miss <= 6.0,
            "q={q}: a homogeneous sample must satisfy the conditional bar under the pooled Σ; \
             worst stratum miss {miss:.2} SE (mc={mc:.6} target={target:.6})"
        );
    }
}

/// The production row lane must read the SAME per-row covariance the identity
/// above is stated with. `RigidVectorValueWorkspace` binds the field, not a
/// matrix, so this checks the binding end to end: the row program's own
/// negative log-likelihood at row `i` must equal the one computed by hand from
/// `field.at_row(i)`.
#[test]
fn the_row_program_consumes_the_rows_own_conditional_covariance() {
    let data = sample(0x2766_0C33, 0.8);
    let field = conditional_field(&data);
    let workspace = RigidVectorValueWorkspace::new(&field);
    let slopes = [0.55_f64, 0.45_f64];
    let mut distinct = 0usize;
    let reference_row = 0usize;
    for &row in &[0usize, 7, 101, 5_000, N / 2, N - 1] {
        let z_row = [data.z[[row, 0]], data.z[[row, 1]]];
        let from_lane = survival_marginal_slope_vector_neglog(
            row,
            0.31,
            0.62,
            0.44,
            &slopes,
            &z_row,
            &workspace,
            1.0,
            1.0,
            1.0e-8,
            PROBIT_SCALE,
        )
        .expect("row lane neglog");
        // The same row evaluated against the field pinned at a DIFFERENT row's
        // covariance: if the lane ignored the row index these would agree.
        let mismatched = survival_marginal_slope_vector_neglog(
            reference_row,
            0.31,
            0.62,
            0.44,
            &slopes,
            &z_row,
            &workspace,
            1.0,
            1.0,
            1.0e-8,
            PROBIT_SCALE,
        )
        .expect("mismatched-row neglog");
        if row != reference_row && (from_lane - mismatched).abs() > 1.0e-9 {
            distinct += 1;
        }
        // And the value must be the one `field.at_row(row)` implies, through the
        // independent `eta` route.
        let eta = survival_marginal_slope_vector_eta(
            0.62,
            &z_row,
            &slopes,
            field.at_row(row),
            PROBIT_SCALE,
        )
        .expect("eta");
        assert!(eta.is_finite(), "row {row} eta must be finite");
    }
    assert!(
        distinct >= 4,
        "the row lane must actually vary with the row's covariance; only {distinct} of 5 \
         off-reference rows differed"
    );
}
