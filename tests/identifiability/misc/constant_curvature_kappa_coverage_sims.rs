//! #944 stage-4 validation sims — the deferred quantitative half of "curvature
//! as an estimand": across REPLICATES of data generated on a known `M_κ`, the
//! profile-likelihood machinery must (1) RECOVER the planted curvature with low
//! bias, (2) COVER the true κ⋆ with its 95% profile CI at ≈ the nominal rate,
//! and (3) hold SIZE on the interior κ=0 flatness test (flat data is not
//! spuriously rejected) while having POWER (curved data is rejected). The
//! single-dataset e2e test (`constant_curvature_kappa_inference_e2e`) asserts
//! sign-recovery and flatness DIRECTION; this test adds the replicate-level
//! calibration the issue charter names ("recovery of κ̂, CI coverage, size of
//! the κ=0 test") — the claims that make "κ̂ = … (95% CI …)" a statistically
//! honest sentence rather than a point estimate.
//!
//! Reference-as-truth: every dataset is generated on a known `ConstantCurvature`
//! geometry and every assertion is against that self-constructed truth or the
//! exact χ² calibration of gam's own profiled REML criterion — never another
//! tool's output. Bars are sized to the small replicate count `R` so they catch
//! a genuinely miscalibrated estimator/CI/test without flaking on binomial noise
//! (kept CI-cheap: small n, few centers, a handful of replicates).

use gam::estimate::FitOptions;
use gam::geometry::constant_curvature::ConstantCurvature;
use gam::inference::data::EncodedDataset;
use gam::inference::formula_dsl::parse_formula;
use gam::inference::model::{ColumnKindTag, DataSchema, SchemaColumn};
use gam::smooth::{
    CurvatureInference, SpatialLengthScaleOptimizationOptions, curvature_inference_forspec,
    fit_term_collectionwith_spatial_length_scale_optimization,
};
use gam::terms::term_builder::build_termspec;
use gam::types::LikelihoodSpec;
use ndarray::{Array1, Array2};

// --- deterministic RNG (splitmix64 → unit / gaussian), no external deps ------

use gam::utils::splitmix64;
fn next_unit(state: &mut u64) -> f64 {
    (splitmix64(state) >> 11) as f64 / (1u64 << 53) as f64
}
fn next_gauss(state: &mut u64) -> f64 {
    let u1 = next_unit(state).max(1.0e-12);
    let u2 = next_unit(state);
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

/// Build a `TermCollectionSpec` for a `curv(...)` formula. Mirrors the e2e
/// inference test's builder: a 3-column `[y, x1, x2]` continuous schema so the
/// `curv(x1, x2)` term resolves and is not rejected as a constant-column smooth.
fn termspec_for(formula: &str, frame: &Array2<f64>) -> gam::smooth::TermCollectionSpec {
    let parsed = parse_formula(formula).expect("formula parses");
    let headers = vec!["y".to_string(), "x1".to_string(), "x2".to_string()];
    let ds = EncodedDataset {
        headers: headers.clone(),
        values: frame.clone(),
        schema: DataSchema {
            columns: headers
                .iter()
                .map(|name| SchemaColumn {
                    name: name.clone(),
                    kind: ColumnKindTag::Continuous,
                    levels: vec![],
                })
                .collect(),
        },
        column_kinds: vec![ColumnKindTag::Continuous; 3],
    };
    let col_map = ds.column_map();
    let mut notes = Vec::new();
    build_termspec(
        &parsed.terms,
        &ds,
        &col_map,
        &mut notes,
        &gam::ResourcePolicy::default_library(),
    )
    .expect("term spec")
}

/// `n` chart points uniformly in a disk of radius `radius`, with a Gaussian
/// response that is a smooth function of the `M_κ` geodesic distance to the
/// origin — a κ⋆-dependent signal the constant-curvature kernel can represent,
/// so curvature is identified.
fn dataset_on_m_kappa(
    n: usize,
    kappa_star: f64,
    radius: f64,
    noise_sd: f64,
    seed: u64,
) -> (Array2<f64>, Array1<f64>) {
    let mut st = seed;
    let manifold = ConstantCurvature::new(2, kappa_star);
    let reference = ndarray::array![0.0_f64, 0.0_f64];
    let mut feats = Array2::<f64>::zeros((n, 2));
    let mut y = Array1::<f64>::zeros(n);
    for i in 0..n {
        let (x1, x2) = loop {
            let a = 2.0 * next_unit(&mut st) - 1.0;
            let b = 2.0 * next_unit(&mut st) - 1.0;
            if a * a + b * b <= 1.0 {
                break (a * radius, b * radius);
            }
        };
        let pt = ndarray::array![x1, x2];
        let d = manifold
            .distance(pt.view(), reference.view())
            .expect("in-chart geodesic distance");
        let mu = 2.0 * (-d).exp() - 1.0;
        feats[(i, 0)] = x1;
        feats[(i, 1)] = x2;
        y[i] = mu + noise_sd * next_gauss(&mut st);
    }
    (feats, y)
}

/// Fit `curv(x1, x2)` with κ optimized as an outer ψ-coordinate, then run the
/// full curvature inference (κ̂ + profile CI + κ=0 LR test) off the REAL
/// profiled REML criterion. CI-cheap: small `centers`, capped outer iters.
fn fit_and_infer(feats: &Array2<f64>, y: &Array1<f64>) -> CurvatureInference {
    let n = y.len();
    let mut frame = Array2::<f64>::zeros((n, 3));
    for i in 0..n {
        frame[(i, 0)] = y[i];
        frame[(i, 1)] = feats[(i, 0)];
        frame[(i, 2)] = feats[(i, 1)];
    }
    let spec = termspec_for("y ~ curv(x1, x2, centers=6)", &frame);

    let weights = Array1::<f64>::ones(n);
    let offset = Array1::<f64>::zeros(n);
    let options = FitOptions::default();
    let kappa_options = SpatialLengthScaleOptimizationOptions {
        max_outer_iter: 8,
        rel_tol: 1e-4,
        pilot_subsample_threshold: 0,
        ..SpatialLengthScaleOptimizationOptions::default()
    };

    let fitted = fit_term_collectionwith_spatial_length_scale_optimization(
        frame.view(),
        y.clone(),
        weights.clone(),
        offset.clone(),
        &spec,
        LikelihoodSpec::gaussian_identity(),
        &options,
        &kappa_options,
    )
    .expect("constant-curvature fit with κ optimization");

    curvature_inference_forspec(
        frame.view(),
        y.view(),
        weights.view(),
        offset.view(),
        &fitted.resolvedspec,
        0,
        LikelihoodSpec::gaussian_identity(),
        &options,
        0.95,
    )
    .expect("curvature inference")
}

/// Number of replicate datasets per arm, and the miss bar that goes with it.
///
/// The count is DERIVED, not chosen (gam#2687). A "at most one miss out of `n`"
/// bar is unfalsifiable at `n = 3`: exact binomial arithmetic gives
/// `P(pass | true coverage 0.50) = 0.5000` and `P(pass | 0.83) = 0.9231`, so an
/// estimator with half the nominal coverage passes as often as it fails, and
/// tightening the bar does not help — at `n = 3` with ZERO misses allowed a
/// 0.83-coverage estimator still passes 0.5718 of the time while a correct one
/// already fails 0.1426 of the time. It is the COUNT, not the bar, that makes
/// the claim unresolvable.
///
/// Taking the bar as the smallest `k` holding false alarm ≤ 1% at true coverage
/// 0.95:
///
/// | n | bar `k` | P(pass \| 0.95) | power vs 0.50 | power vs 0.83 |
/// |---|---|---|---|---|
/// | 3 | 1 | 0.9928 | 0.5000 | 0.0769 |
/// | **9** | **2** | 0.9916 | **0.9102** | 0.1861 |
/// | 25 | 4 | 0.9928 | 0.9995 | 0.4241 |
/// | 60 | 7 | 0.9902 | 1.0000 | 0.8222 |
///
/// `n = 9, k = 2` is the smallest count that catches a GROSSLY broken estimator
/// (coverage ≈ 0.5) at ≥ 90% power while false-alarming under 1% on a correct
/// one. Catching a mildly miscalibrated one (0.83) needs `n ≈ 60`, which is a
/// cluster-scale sweep and is deliberately out of scope here — so this gate's
/// stated job is "grossly broken", and the table says so rather than leaving the
/// reader to assume more.
///
/// The bar must move with the count: at `n = 50`, "at most one miss" would
/// reject a perfectly calibrated estimator 72% of the time. Any change to
/// `REPLICATE_COUNT` has to re-derive `MAX_MISSES` from the same binomial.
const REPLICATE_COUNT: usize = 9;

/// Maximum number of missed replicates the coverage/size bars tolerate. Derived
/// with [`REPLICATE_COUNT`]; see its table.
const MAX_MISSES: usize = 2;

fn replicate_count() -> usize {
    REPLICATE_COUNT
}

/// How one replicate's profile CI resolved against a target κ.
///
/// The middle arm is the one this file was missing (gam#2687). `KappaProfileCi`
/// carries `lo_at_bound` / `hi_at_bound` — documented as *"CI is left/right-open
/// at the bound"* — and `kappa_hat_support`, which says whether κ̂ is itself a
/// box endpoint. A railed κ̂ is not a profile minimiser, so the Wilks region
/// `2[V_p(κ) − V_p(κ̂)] ≤ χ²₁` anchored at it is not a 95% interval at all; and a
/// bound-open interval does not EXCLUDE a κ beyond it, so counting such a
/// replicate as a miss reports an exclusion the data never made. Both are
/// UNRESOLVED, which is neither coverage nor a miss, and a gate that silently
/// folds them into either number is measuring something other than coverage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Resolution {
    Covered,
    Missed,
    /// The interval carries no coverage claim about this target: κ̂ is railed, or
    /// the interval is open on the side the target lies.
    Unresolved,
}

fn resolve(inf: &CurvatureInference, target: f64) -> Resolution {
    if inf.ci.kappa_hat_support.is_railed() {
        return Resolution::Unresolved;
    }
    if target < inf.ci.ci_lo {
        // Below the interval: only a genuine (closed) lower endpoint excludes it.
        return if inf.ci.lo_at_bound {
            Resolution::Unresolved
        } else {
            Resolution::Missed
        };
    }
    if target > inf.ci.ci_hi {
        return if inf.ci.hi_at_bound {
            Resolution::Unresolved
        } else {
            Resolution::Missed
        };
    }
    Resolution::Covered
}

/// CI COVERAGE + κ̂ RECOVERY on CURVED truth. Across `R` independent M_κ
/// datasets at a planted spherical κ⋆, the 95% profile CI must cover κ⋆ at close
/// to the nominal rate and κ̂ must recover κ⋆ with low bias and correct sign.
#[test]
fn profile_ci_covers_planted_curvature_across_replicates() {
    gam::init_parallelism();
    let reps = replicate_count();
    let kappa_star = 1.5_f64;
    let mut covered = 0usize;
    let mut missed = 0usize;
    let mut railed = 0usize;
    let mut sign_correct = 0usize;
    let mut sum_khat = 0.0_f64;
    let mut khats = Vec::with_capacity(reps);
    for r in 0..reps {
        let seed = 0x5EED_0944_0000_0000 ^ ((r as u64) << 8);
        let (feats, y) = dataset_on_m_kappa(120, kappa_star, 0.6, 0.10, seed);
        let inf = fit_and_infer(&feats, &y);
        match resolve(&inf, kappa_star) {
            Resolution::Covered => covered += 1,
            Resolution::Missed => missed += 1,
            Resolution::Unresolved => {}
        }
        if inf.ci.kappa_hat_support.is_railed() {
            railed += 1;
        }
        if inf.kappa_hat > 0.0 {
            sign_correct += 1;
        }
        sum_khat += inf.kappa_hat;
        khats.push(inf.kappa_hat);
        eprintln!(
            "[cov κ⋆=+{kappa_star}] r={r} κ̂={:+.3} support={} CI=[{:+.3},{:+.3}] \
             open=[{},{}] -> {:?}",
            inf.kappa_hat,
            inf.ci.kappa_hat_support.label(),
            inf.ci.ci_lo,
            inf.ci.ci_hi,
            inf.ci.lo_at_bound,
            inf.ci.hi_at_bound,
            resolve(&inf, kappa_star)
        );
    }
    let mean_khat = sum_khat / reps as f64;
    let resolved = covered + missed;
    eprintln!(
        "[cov κ⋆=+{kappa_star}] covered {covered} / missed {missed} / unresolved {} of {reps}  \
         railed κ̂ {railed}/{reps}  sign_correct {sign_correct}/{reps}  \
         mean κ̂={mean_khat:+.3}  κ̂={khats:?}",
        reps - resolved
    );

    // (0) THE ESTIMATE MUST BE AN ESTIMATE. `κ̂` is the argmin of `V_p` over the
    // chart-feasible box; when the box constraint is active, `κ̂` is a readout of
    // the BOX and moves with it, and the Wilks region anchored at it is not a
    // 95% interval. Coverage is then unmeasured — not 0%, not 100% — so this has
    // to be asserted BEFORE any coverage number is read, or the gate reports a
    // rate for a quantity that does not exist. Before #2687 this file computed
    // `covers` as a closed containment and reported `covers=false`, which read
    // as a mis-covering interval when the true state was "no interval".
    assert!(
        railed == 0,
        "κ̂ was RAILED at a κ-box endpoint in {railed}/{reps} replicates ({khats:?}); \
         the profile criterion has no interior optimum there, so the profile CI is \
         not a 95% interval and coverage is UNMEASURED. Widening the box does not \
         fix this — it moves the rail (#2687 measured κ̂ = 2.78 against κ⋆ = 1.5 at \
         the resolution-derived box end). The defect is in the criterion or in \
         this fixture's identifiability, not in the bound."
    );
    // (1) COVERAGE, among the replicates that carry a coverage claim. The bar is
    // derived with `REPLICATE_COUNT`; see its table.
    assert!(
        resolved == reps && covered + MAX_MISSES >= reps,
        "profile CI covered the planted κ⋆=+{kappa_star} in {covered}/{resolved} resolved \
         replicates ({} unresolved of {reps}); the derived bar at n={reps} is at most \
         {MAX_MISSES} misses",
        reps - resolved
    );
    // (2) SIGN RECOVERY: spherical truth ⇒ κ̂ > 0 in all but the derived bar.
    assert!(
        sign_correct + MAX_MISSES >= reps,
        "κ̂ sign recovered (>0) in only {sign_correct}/{reps} replicates for κ⋆=+{kappa_star}"
    );
    // (3) LOW BIAS: the mean estimate tracks the truth within a tolerance honest
    // about the noisy Gaussian signal at n=120 — not railed to a chart bound, not
    // collapsed toward 0.
    assert!(
        (mean_khat - kappa_star).abs() < 1.0,
        "mean κ̂={mean_khat:+.3} too far from planted κ⋆=+{kappa_star} (bias bar 1.0)"
    );
}

/// SIZE of the interior κ=0 flatness test on FLAT truth. Across `R` flat
/// datasets the LR test must NOT spuriously reject (a badly-sized test would
/// reject most), and the profile CI must cover κ=0 (verdict Flat) in a large
/// majority — the controlled-size "is my latent space flat?" claim.
#[test]
fn flatness_test_holds_size_across_flat_replicates() {
    gam::init_parallelism();
    let reps = replicate_count();
    let alpha = 0.05_f64;
    let mut rejections = 0usize;
    let mut ci_covers_zero = 0usize;
    let mut unresolved = 0usize;
    let mut pvals = Vec::with_capacity(reps);
    for r in 0..reps {
        let seed = 0x71A7_0944_0000_0000 ^ ((r as u64) << 8);
        let (feats, y) = dataset_on_m_kappa(120, 0.0, 0.6, 0.10, seed);
        let inf = fit_and_infer(&feats, &y);
        if inf.flatness.p_value < alpha {
            rejections += 1;
        }
        match resolve(&inf, 0.0) {
            Resolution::Covered => ci_covers_zero += 1,
            Resolution::Missed => {}
            Resolution::Unresolved => unresolved += 1,
        }
        pvals.push(inf.flatness.p_value);
        eprintln!(
            "[size κ⋆=0] r={r} κ̂={:+.3} support={} p={:.4} CI=[{:+.3},{:+.3}] -> {:?}",
            inf.kappa_hat,
            inf.ci.kappa_hat_support.label(),
            inf.flatness.p_value,
            inf.ci.ci_lo,
            inf.ci.ci_hi,
            resolve(&inf, 0.0)
        );
    }
    eprintln!(
        "[size κ⋆=0] rejected {rejections}/{reps} at α={alpha}  CI⊇0 in {ci_covers_zero}/{reps}  \
         unresolved {unresolved}/{reps}  p-values={pvals:?}"
    );

    // SIZE CONTROL: a level-α interior χ²₁ test on truly flat data rejects ~α of
    // the time (expected ≈0.05·reps). A test that over-rejects (wrong reference,
    // e.g. a phantom curvature from the basis, or a mis-scaled LR) rejects many.
    // Allow a strict minority (≤ reps/2) to absorb the small-R binomial tail while
    // still failing a test that rejects flat data routinely, at either count.
    assert!(
        rejections <= reps / 2,
        "κ=0 flatness test rejected truly-flat data in {rejections}/{reps} replicates at α={alpha} \
         (size-inflated): p-values {pvals:?}"
    );
    // The profile CI must straddle 0 (verdict Flat) for flat data in all but the
    // derived bar — the CI-side mirror of the size claim. Unresolved replicates
    // are named separately: a bound-open interval or a railed κ̂ carries no claim
    // about κ=0 either way, and folding them into the covered count would let a
    // fit that never resolved report perfect coverage.
    assert!(
        unresolved == 0,
        "the profile CI carried no coverage claim about κ=0 in {unresolved}/{reps} flat \
         replicates (railed κ̂ or a bound-open interval on the side of 0); size is \
         UNMEASURED there, not passing"
    );
    assert!(
        ci_covers_zero + MAX_MISSES >= reps,
        "profile CI failed to cover κ=0 on flat data in {}/{reps} replicates",
        reps - ci_covers_zero
    );
}
