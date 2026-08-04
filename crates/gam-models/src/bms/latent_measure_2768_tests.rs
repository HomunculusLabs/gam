//! Unit gates on the shared latent-measure decision (gam#2768) and on the
//! conditional location-scale calibration it escalates to.

use super::*;
use ndarray::{Array1, Array2};

/// Deterministic standard-normal draws (Box–Muller over a splitmix64 stream) so
/// every gate here is reproducible without a test-only RNG dependency.
fn gaussian_stream(n: usize, seed: u64) -> Vec<f64> {
    let mut state = seed;
    let mut next_unit = || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        ((z >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    };
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        let u1 = next_unit().max(1e-12);
        let u2 = next_unit();
        let r = (-2.0 * u1.ln()).sqrt();
        out.push(r * (std::f64::consts::TAU * u2).cos());
        if out.len() < n {
            out.push(r * (std::f64::consts::TAU * u2).sin());
        }
    }
    out
}

fn weighted_moments(values: &Array1<f64>, weights: &Array1<f64>) -> (f64, f64) {
    let total: f64 = weights.iter().sum();
    let mean: f64 = values
        .iter()
        .zip(weights.iter())
        .map(|(&v, &w)| w * v)
        .sum::<f64>()
        / total;
    let var: f64 = values
        .iter()
        .zip(weights.iter())
        .map(|(&v, &w)| w * (v - mean) * (v - mean))
        .sum::<f64>()
        / total;
    (mean, var.sqrt())
}

/// `z = m·x + √(1−m²)·ζ` with `x, ζ` independent standard normals: the latent
/// score is EXACTLY standard normal marginally while every conditional law
/// `z | x` is shifted. This is the fixture the whole conditional gate exists
/// for — the pooled marginal gate cannot see the shift, and no transform of the
/// marginal law can remove it.
fn conditionally_shifted_score(n: usize, m: f64, seed: u64) -> (Array1<f64>, Array2<f64>) {
    let x = gaussian_stream(n, seed);
    let zeta = gaussian_stream(n, seed ^ 0xA5A5_A5A5_A5A5_A5A5);
    let residual_sd = (1.0 - m * m).sqrt();
    let z = Array1::from_iter(
        (0..n).map(|i| m * x[i] + residual_sd * zeta[i]),
    );
    // The conditioning span is the marginal design `[1 | x]`.
    let mut a_block = Array2::<f64>::ones((n, 2));
    for i in 0..n {
        a_block[[i, 1]] = x[i];
    }
    (z, a_block)
}

/// The conditionally-calibrated score must have unit variance.
///
/// This is not a cosmetic moment check. The closed-form probit kernel is the
/// standard-normal one, and the marginal-slope identity that makes `q` the
/// MARGINAL index — `E_ζ[Φ(q·√(1+b²) + b·ζ)] = Φ(q)` — holds only when
/// `Var(ζ | C) = 1`. Any residual scale `v` turns it into
/// `Φ(q·√(1+b²)/√(1+b²v))`, i.e. a multiplicative distortion of every marginal
/// coefficient.
///
/// The trap is that a conditional-MEAN shift necessarily eats variance: with z
/// standardised to unit marginal variance, `1 = Var(m(C)) + E[Var(z|C)]`, so the
/// residual variance is `1 − R²` and is strictly below the marginal variance
/// whenever the gate fires at all.
#[test]
fn conditional_calibration_leaves_the_score_at_unit_variance() {
    let n = 4000;
    let m = 0.5_f64;
    let (z, a_block) = conditionally_shifted_score(n, m, 0x2768_0001);
    let weights = Array1::<f64>::ones(n);

    let (pooled_mean, pooled_sd) = weighted_moments(&z, &weights);
    assert!(
        pooled_mean.abs() < 0.05 && (pooled_sd - 1.0).abs() < 0.05,
        "fixture invariant: the raw score is marginally standard normal, so the \
         pooled gate cannot see the conditional shift; got mean={pooled_mean:.4} sd={pooled_sd:.4}"
    );

    let calibration = fit_conditional_latent_calibration_if_needed(&z, &weights, a_block.view())
        .expect("conditional gate")
        .expect("the E[z|C] Rao gate must fire on a 0.5 conditional correlation at n=4000");

    let zeta = calibration
        .apply(z.view(), a_block.view())
        .expect("apply the conditional calibration");
    let (zeta_mean, zeta_sd) = weighted_moments(&zeta, &weights);

    assert!(
        zeta_mean.abs() < 0.02,
        "the calibrated score must be centred; got {zeta_mean:.4}"
    );
    assert!(
        (zeta_sd - 1.0).abs() < 0.05,
        "the calibrated score must have unit variance so `q` keeps its marginal-index \
         meaning; got sd={zeta_sd:.4} (a residual scale v distorts every marginal \
         coefficient by √(1+b²)/√(1+b²v))"
    );
    assert!(
        (calibration.post_sd - 1.0).abs() < 0.05,
        "the calibration's own recorded post_sd must agree with the sample it produced; \
         got {:.4}",
        calibration.post_sd
    );
}
