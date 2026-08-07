//! gam#2766 probe — how far does the pooled `Σ` push `E[Φ(−η)|a]` off `Φ(−q)`?
//!
//! Temporary. This module exists to answer ONE question before anything is
//! changed: with `K ≥ 2` latent scores whose CONDITIONAL correlation varies with
//! the marginal-index span `a`, and whose conditional marginals are exactly
//! `N(0,1)` (so #2768's per-coordinate location-scale gate has nothing left to
//! do), how large is the departure from the identity the whole marginal-slope
//! family is defined by?
//!
//! The identity, transcribed from `bms/gradient_paths.rs`:
//!
//! ```text
//!   z | a ~ N(0, Σ(a)),   η = c(a)·q(t,a) + r(a)ᵀz
//!   E_z[Φ(−η) | a] = Φ(−q(t,a))   ⟺   c(a) = √(1 + r(a)ᵀ Σ(a) r(a))
//! ```
//!
//! The shipped `c` uses ONE pooled `Σ̄` (`marginal_slope_covariance_from_scores`
//! over every row). Substituting a constant `c̄ = √(1 + r'Σ̄r)` into the exact
//! integral gives
//!
//! ```text
//!   E_z[Φ(−η) | a] = Φ(−q · c̄ / √(1 + r'Σ(a)r))
//! ```
//!
//! so the realized marginal index is `q · c̄/c(a)` rather than `q`: a
//! MULTIPLICATIVE, covariate-dependent distortion of the estimand this family
//! exists to deliver. The probe measures that ratio directly, and also confirms
//! the closed form against a Monte-Carlo integral so the algebra is not taken on
//! trust.

use super::*;

/// Deterministic standard normals (Box–Muller over splitmix64).
fn gaussians(n: usize, seed: u64) -> Vec<f64> {
    let mut state = seed;
    let mut unit = || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        ((z >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    };
    let mut out = Vec::with_capacity(n + 1);
    while out.len() < n {
        let u1 = unit().max(1e-12);
        let u2 = unit();
        let r = (-2.0 * u1.ln()).sqrt();
        out.push(r * (std::f64::consts::TAU * u2).cos());
        out.push(r * (std::f64::consts::TAU * u2).sin());
    }
    out.truncate(n);
    out
}

use gam_math::probability::{normal_cdf as standard_normal_cdf, standard_normal_quantile};

/// `K = 2` scores, conditionally standard normal in each coordinate, with a
/// conditional correlation that is a deterministic function of the single
/// marginal covariate `x`:
///
/// ```text
///   z1 = e1,   z2 = ρ(x)·e1 + √(1−ρ(x)²)·e2,   e ~ N(0, I2)
///   ρ(x) = ρ_amplitude · tanh(x)
/// ```
///
/// Every conditional marginal is exactly `N(0,1)`, every conditional mean is
/// exactly `0`, and the ONLY conditional structure is the off-diagonal. That is
/// precisely the state #2768 leaves behind and this issue names.
fn varying_correlation_scores(n: usize, amplitude: f64) -> (Array2<f64>, Vec<f64>, Vec<f64>) {
    let x_raw = gaussians(n, 0x2766_A1);
    let e1 = gaussians(n, 0x2766_B2);
    let e2 = gaussians(n, 0x2766_C3);
    let mut z = Array2::<f64>::zeros((n, 2));
    let mut rho = Vec::with_capacity(n);
    for row in 0..n {
        let r = amplitude * x_raw[row].tanh();
        rho.push(r);
        z[[row, 0]] = e1[row];
        z[[row, 1]] = r * e1[row] + (1.0 - r * r).max(0.0).sqrt() * e2[row];
    }
    (z, x_raw, rho)
}

/// The measurement. Prints the realized marginal index against the intended one
/// across the range of `ρ(x)`, and reports the worst multiplicative distortion.
#[test]
fn probe_2766_pooled_sigma_distorts_the_marginal_index() {
    let n = 20_000;
    let amplitude = 0.8;
    let (z, x, rho) = varying_correlation_scores(n, amplitude);
    let weights = Array1::<f64>::ones(n);

    // What the shipped fit installs.
    let pooled = marginal_slope_covariance_from_scores(z.view(), &weights).expect("pooled Σ");
    let pooled_dense = pooled.to_dense();
    println!(
        "#2766 probe: pooled Σ = [[{:.6}, {:.6}], [{:.6}, {:.6}]]  (mean ρ(x) = {:.6})",
        pooled_dense[[0, 0]],
        pooled_dense[[0, 1]],
        pooled_dense[[1, 0]],
        pooled_dense[[1, 1]],
        rho.iter().sum::<f64>() / n as f64,
    );

    // A shared log-slope surface: r = (g, g). This is the SHARED lane, whose
    // whole covariance dependence is the cached scalar `1ᵀΣ1`.
    let probit_scale = 1.0;
    for &g in &[0.3_f64, 0.6, 1.0] {
        let slopes = [g, g];
        let c_pooled =
            marginal_slope_preserving_scale(&slopes, &pooled, probit_scale).expect("c̄");
        let mut worst = 1.0_f64;
        let mut worst_rho = 0.0_f64;
        for &r in &[-amplitude, -0.4, 0.0, 0.4, amplitude] {
            // The TRUE conditional covariance at that ρ.
            let sigma_a = MarginalSlopeCovariance::full(ndarray::array![[1.0, r], [r, 1.0]])
                .expect("Σ(a)");
            let c_true =
                marginal_slope_preserving_scale(&slopes, &sigma_a, probit_scale).expect("c(a)");
            let ratio = c_pooled / c_true;
            if (ratio - 1.0).abs() > (worst - 1.0).abs() {
                worst = ratio;
                worst_rho = r;
            }
            println!(
                "  g={g:.2}  ρ(a)={r:+.2}  c̄={c_pooled:.6}  c(a)={c_true:.6}  \
                 realized index = q × {ratio:.6}"
            );
        }
        println!("  g={g:.2}  WORST distortion {worst:.6} at ρ(a)={worst_rho:+.2}\n");
    }

    // Confirm the closed form against a Monte-Carlo integral of the actual
    // identity, on the rows themselves: bucket rows by ρ(x) and compare the
    // sample mean of Φ(−η) to Φ(−q) with q fixed.
    let g = 0.6_f64;
    let slopes = [g, g];
    let c_pooled = marginal_slope_preserving_scale(&slopes, &pooled, probit_scale).expect("c̄");
    let q = 0.5_f64;
    let mut buckets: Vec<(f64, f64, usize)> = vec![(0.0, 0.0, 0); 5];
    for row in 0..n {
        let bucket = (((x[row].tanh() + 1.0) * 0.5 * 5.0) as usize).min(4);
        let eta = c_pooled * q + probit_scale * (slopes[0] * z[[row, 0]] + slopes[1] * z[[row, 1]]);
        buckets[bucket].0 += standard_normal_cdf(-eta);
        buckets[bucket].1 += rho[row];
        buckets[bucket].2 += 1;
    }
    println!("#2766 probe: bucketed Monte-Carlo of E[Φ(−η)|a] against the intended Φ(−q) = {:.6}", standard_normal_cdf(-q));
    let mut max_index_error = 0.0_f64;
    for (index, &(sum, rho_sum, count)) in buckets.iter().enumerate() {
        if count == 0 {
            continue;
        }
        let realized = sum / count as f64;
        let rho_bar = rho_sum / count as f64;
        // Invert Φ to read the realized marginal index.
        let realized_q = -standard_normal_quantile(realized).expect("realized survival is interior");
        let sigma_a =
            MarginalSlopeCovariance::full(ndarray::array![[1.0, rho_bar], [rho_bar, 1.0]]).unwrap();
        let predicted_q =
            q * c_pooled / marginal_slope_preserving_scale(&slopes, &sigma_a, probit_scale).unwrap();
        println!(
            "  bucket {index}: n={count:5}  ρ̄={rho_bar:+.4}  E[Φ(−η)]={realized:.6}  \
             realized q={realized_q:.6}  closed-form q={predicted_q:.6}"
        );
        max_index_error = max_index_error.max((realized_q - q).abs() / q);
    }
    println!("#2766 probe: worst relative marginal-index error = {:.4}", max_index_error);
    assert!(
        max_index_error > 0.05,
        "the probe is supposed to EXHIBIT the defect; got only {max_index_error:.4}"
    );
}

