//! gam#2768 — the conditional latent-measure gate must return the MARGINAL
//! index, end to end, on a score that is exactly `N(0, 1)` and conditionally
//! shifted.
//!
//! # What is at stake
//!
//! The marginal-slope parameterisation `η = q·c(b) + b·z`, `c(b) = √(1+b²)`,
//! earns its name from
//!
//! ```text
//!     E_z[Φ(q·√(1+b²) + b·z)] = Φ(q)     for   z | C ~ N(0, 1),
//! ```
//!
//! so `q` is the marginal index — **conditional on `C`**. A score can be exactly
//! standard normal overall while every conditional law `z | C` is shifted, and
//! the pooled adequacy gate cannot see the difference. This fixture builds
//! exactly that:
//!
//! ```text
//!     x, ζ ~ N(0,1) independent,     z = m·x + √(1−m²)·ζ
//! ```
//!
//! `z ~ N(0,1)` exactly; `E[z | x] = m·x`. The outcome is generated from the
//! marginal-slope model **on ζ**, so the truth is `q = β₀ + β_x·x` with a
//! constant slope `b` — and `P(Y=1 | x) = Φ(β₀ + β_x·x)` exactly.
//!
//! A fit handed the raw `z` axis is a different model. Substituting
//! `ζ = (z − m x)/√(1−m²)`:
//!
//! ```text
//!     η = q·c(b) + b·ζ
//!       = [q·c(b) − (b·m/√(1−m²))·x] + (b/√(1−m²))·z
//! ```
//!
//! so it estimates slope `b′ = b/√(1−m²)` and marginal x-coefficient
//! `β_x·c(b)/c(b′) − (b·m/√(1−m²))/c(b′)` — the `b(C)·m(C)` leakage, landing in
//! the influence channel. With the constants below that is `0.107` against a
//! truth of `0.500`: the sign of the effect survives and almost nothing else
//! does.
//!
//! # Why this fixture is Bernoulli
//!
//! The gate, the conditional location-scale correction, and the unit-variance
//! fix are one shared object serving both marginal-slope families, so an
//! end-to-end estimand gate on either one gates the shared arithmetic. The
//! Bernoulli family is the one whose outer search is well conditioned at this
//! size; the survival family's own wiring is gated at its seams (its
//! latent-measure decision, its row-level `∂(score)/∂ζ` channel against a finite
//! difference, its predict conditioning span, and its two persistence refusals).

use csv::StringRecord;
use gam::utils::splitmix64;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};

const N: usize = 6_000;
/// `Corr(z, x)`, and the conditional-mean slope the gate has to find.
const M_SHIFT: f64 = 0.6;
/// True conditional slope on the standardised latent score.
const TRUE_SLOPE: f64 = 0.6;
/// True marginal-index coefficients.
const TRUE_BETA_X: f64 = 0.5;
const TRUE_INTERCEPT: f64 = -0.2;

fn next_unit(state: &mut u64) -> f64 {
    (splitmix64(state) >> 11) as f64 / (1u64 << 53) as f64
}

fn next_gauss(state: &mut u64) -> f64 {
    let u1 = next_unit(state).max(f64::MIN_POSITIVE);
    let u2 = next_unit(state);
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

fn standardized(mut v: Vec<f64>) -> Vec<f64> {
    let n = v.len() as f64;
    let mean = v.iter().sum::<f64>() / n;
    let sd = (v.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / n)
        .sqrt()
        .max(1e-12);
    for value in v.iter_mut() {
        *value = (*value - mean) / sd;
    }
    v
}

struct Fixture {
    dataset: gam::inference::data::EncodedDataset,
    x: Vec<f64>,
}

fn build_fixture() -> Fixture {
    let headers = ["y", "z", "zeta", "x"]
        .iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();
    let mut state: u64 = 0x2768_BEEF_5EED_0001;
    let x = standardized((0..N).map(|_| next_gauss(&mut state)).collect());
    let zeta = standardized((0..N).map(|_| next_gauss(&mut state)).collect());
    let residual_sd = (1.0 - M_SHIFT * M_SHIFT).sqrt();
    let c_true = (1.0 + TRUE_SLOPE * TRUE_SLOPE).sqrt();

    let mut rows: Vec<StringRecord> = Vec::with_capacity(N);
    let mut positives = 0usize;
    for row in 0..N {
        let z = M_SHIFT * x[row] + residual_sd * zeta[row];
        let q = TRUE_INTERCEPT + TRUE_BETA_X * x[row];
        let eta = q * c_true + TRUE_SLOPE * zeta[row];
        let y = u8::from(next_unit(&mut state) < gam::probability::normal_cdf(eta));
        positives += usize::from(y == 1);
        rows.push(StringRecord::from(vec![
            y.to_string(),
            z.to_string(),
            zeta[row].to_string(),
            x[row].to_string(),
        ]));
    }
    eprintln!(
        "[2768] n={N} positives={positives} ({:.1}%)",
        100.0 * positives as f64 / N as f64
    );
    Fixture {
        dataset: encode_recordswith_inferred_schema(headers, rows).expect("encode #2768 fixture"),
        x,
    }
}

struct Arm {
    /// `∂(X_m β_m)/∂x`, read by projection rather than from a coefficient index:
    /// the marginal block carries whatever identifiability chart the frozen
    /// joint build chose, so the index is not a stable contract and the
    /// projection is.
    marginal_x_slope: f64,
    mean_slope: f64,
    calibration: &'static str,
}

fn slope_on(values: &[f64], x: &[f64]) -> f64 {
    let n = values.len() as f64;
    let x_mean = x.iter().sum::<f64>() / n;
    let v_mean = values.iter().sum::<f64>() / n;
    let mut cov = 0.0;
    let mut var = 0.0;
    for i in 0..values.len() {
        let dx = x[i] - x_mean;
        cov += dx * (values[i] - v_mean);
        var += dx * dx;
    }
    cov / var
}

fn fit_arm(fixture: &Fixture, z_column: &str) -> Arm {
    let cfg = FitConfig {
        family: Some("bernoulli-marginal-slope".to_string()),
        z_column: Some(z_column.to_string()),
        logslope_formula: Some("1".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("y ~ x", &fixture.dataset, &cfg)
        .unwrap_or_else(|e| panic!("bernoulli marginal-slope fit on z_column={z_column}: {e}"));
    let FitResult::BernoulliMarginalSlope(fit) = result else {
        panic!("expected a BernoulliMarginalSlope fit for z_column={z_column}");
    };
    let marginal_eta = fit.marginal_design.design.dot(&fit.fit.blocks[0].beta);
    let logslope_eta = fit.logslope_design.design.dot(&fit.fit.blocks[1].beta);
    let mean_slope =
        fit.baseline_logslope + logslope_eta.iter().sum::<f64>() / logslope_eta.len() as f64;
    let calibration = match (
        fit.latent_z_conditional_calibration.as_ref(),
        fit.latent_z_rank_int_calibration.as_ref(),
    ) {
        (Some(_), _) => "conditional",
        (None, Some(_)) => "rank-int",
        (None, None) => "none",
    };
    Arm {
        marginal_x_slope: slope_on(
            marginal_eta
                .as_slice()
                .expect("marginal eta is standard layout"),
            &fixture.x,
        ),
        mean_slope,
        calibration,
    }
}

/// The uncalibrated marginal x-coefficient this fixture produces when the raw
/// `z` axis reaches the kernel. Derived in the module header, not measured from
/// an old build, so the gate is a statement about the model.
fn uncalibrated_marginal_x_slope() -> f64 {
    let c_true = (1.0 + TRUE_SLOPE * TRUE_SLOPE).sqrt();
    let residual_sd = (1.0 - M_SHIFT * M_SHIFT).sqrt();
    let slope_raw = TRUE_SLOPE / residual_sd;
    let c_raw = (1.0 + slope_raw * slope_raw).sqrt();
    TRUE_BETA_X * c_true / c_raw - (TRUE_SLOPE * M_SHIFT / residual_sd) / c_raw
}

#[test]
fn conditional_latent_gate_returns_the_marginal_index() {
    init_parallelism();
    #[cfg(target_os = "macos")]
    gam::gpu::configure_global_policy(gam::gpu::GpuPolicy::Off);

    let fixture = build_fixture();
    let shifted = fit_arm(&fixture, "z");
    let clean = fit_arm(&fixture, "zeta");
    let uncalibrated = uncalibrated_marginal_x_slope();

    eprintln!(
        "[2768] shifted: cal={} beta_x={:.4} slope={:.4} | clean: cal={} beta_x={:.4} \
         slope={:.4} | truth beta_x={TRUE_BETA_X:.4} slope={TRUE_SLOPE:.4} | uncalibrated \
         beta_x={uncalibrated:.4}",
        shifted.calibration,
        shifted.marginal_x_slope,
        shifted.mean_slope,
        clean.calibration,
        clean.marginal_x_slope,
        clean.mean_slope,
    );

    // 1. The gate fires where the shift is and stays quiet where it is not.
    assert_eq!(
        shifted.calibration, "conditional",
        "the E[z|C] Rao gate must escalate at Corr(z, x) = {M_SHIFT}, n = {N}"
    );
    assert_eq!(
        clean.calibration, "none",
        "the gate must not fire on an already conditionally-standard score — a \
         trigger-happy gate redefines the latent axis of every clean fit"
    );

    // 2. Both arms return the MARGINAL index and the true slope.
    for (label, arm) in [("shifted", &shifted), ("clean", &clean)] {
        assert!(
            (arm.marginal_x_slope - TRUE_BETA_X).abs() < 0.07,
            "{label} arm must return the marginal x-coefficient {TRUE_BETA_X}; got {:.4}",
            arm.marginal_x_slope
        );
        assert!(
            (arm.mean_slope - TRUE_SLOPE).abs() < 0.12,
            "{label} arm must return the conditional slope {TRUE_SLOPE}; got {:.4}",
            arm.mean_slope
        );
    }

    // 3. The estimate must not depend on whether the shift was removed
    //    upstream, and must be nowhere near what the uncalibrated axis gives.
    //    This is the clause that goes red if the gate is ever bypassed.
    let arm_gap = (shifted.marginal_x_slope - clean.marginal_x_slope).abs();
    let to_uncalibrated = (shifted.marginal_x_slope - uncalibrated).abs();
    assert!(
        arm_gap < 0.06,
        "the fitted marginal index must not depend on which axis it was handed; arms \
         differ by {arm_gap:.4}"
    );
    assert!(
        to_uncalibrated > 0.25,
        "fixture invariant: the shifted arm must be far from the uncalibrated value \
         {uncalibrated:.4}, or this test cannot tell a working gate from an absent one; \
         distance {to_uncalibrated:.4}"
    );
}
