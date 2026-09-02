//! Dependence-corrected summaries of a serially correlated sample.
//!
//! Both statistics read a sequence `x_1..x_n` whose terms may be autocorrelated
//! (held-out per-row losses in row order, chain draws) and correct the naive
//! i.i.d. summary for that dependence with the lag window `L = ⌊√n⌋`, which
//! grows with `n` while its share `L/n` vanishes — the standard consistent
//! bandwidth for a sample of unknown correlation length.

/// Effective sample size `n / (1 + 2 Σ_{k≥1} ρ_k)` from the initial positive
/// sequence of sample autocorrelations `ρ_k`, truncated at the first
/// non-positive lag (Geyer's rule) and at the lag window `⌊√n⌋`. Returns `n`
/// itself for a degenerate or constant sample and never less than one.
pub fn autocorr_ess(x: &[f64]) -> f64 {
    let n = x.len();
    if n <= 1 {
        return n as f64;
    }
    let mean = x.iter().sum::<f64>() / n as f64;
    let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n as f64;
    if var <= 0.0 {
        return n as f64;
    }
    let lag_cap = (n as f64).sqrt() as usize;
    let mut rho_sum = 0.0;
    for lag in 1..=lag_cap.max(1).min(n - 1) {
        let mut cov = 0.0;
        for i in lag..n {
            cov += (x[i] - mean) * (x[i - lag] - mean);
        }
        cov /= (n - lag) as f64;
        let rho = cov / var;
        if rho <= 0.0 || !rho.is_finite() {
            break;
        }
        rho_sum += rho;
    }
    (n as f64 / (1.0 + 2.0 * rho_sum)).max(1.0)
}

/// Newey–West (Bartlett-kernel) standard error of the sample mean with lag
/// window `⌊√n⌋`: `√(γ_0 + 2 Σ_k w_k γ_k) / √n`, `w_k = 1 − k/(L+1)`. Infinite
/// for a sample of fewer than two terms, where no dispersion is measurable.
pub fn newey_west_se(x: &[f64]) -> f64 {
    let n = x.len();
    if n <= 1 {
        return f64::INFINITY;
    }
    let mean = x.iter().sum::<f64>() / n as f64;
    let lag_cap = (n as f64).sqrt() as usize;
    let mut gamma0 = 0.0;
    for v in x {
        gamma0 += (v - mean) * (v - mean);
    }
    gamma0 /= n as f64;
    let mut var = gamma0;
    for lag in 1..=lag_cap.max(1).min(n - 1) {
        let mut gamma = 0.0;
        for i in lag..n {
            gamma += (x[i] - mean) * (x[i - lag] - mean);
        }
        gamma /= n as f64;
        let w = 1.0 - lag as f64 / (lag_cap as f64 + 1.0);
        var += 2.0 * w * gamma;
    }
    (var.max(0.0) / n as f64).sqrt()
}
