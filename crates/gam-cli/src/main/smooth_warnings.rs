use super::*;

/// Implied continuous order from a measure-jet raw-form per-scale λ spectrum:
/// ŝ = −½ · (least-squares slope of ln λ̂_ℓ on ln ε_ℓ). `None` unless at
/// least two scales carry finite positive (ε_ℓ, λ̂_ℓ) and the band has
/// nonzero log-spread.
pub(crate) fn measure_jet_implied_order(per_scale: &[(f64, f64)]) -> Option<f64> {
    let pts: Vec<(f64, f64)> = per_scale
        .iter()
        .filter(|&&(eps, lam)| eps.is_finite() && eps > 0.0 && lam.is_finite() && lam > 0.0)
        .map(|&(eps, lam)| (eps.ln(), lam.ln()))
        .collect();
    if pts.len() < 2 {
        return None;
    }
    let n = pts.len() as f64;
    let xbar = pts.iter().map(|p| p.0).sum::<f64>() / n;
    let ybar = pts.iter().map(|p| p.1).sum::<f64>() / n;
    let sxx = pts.iter().map(|p| (p.0 - xbar).powi(2)).sum::<f64>();
    if sxx <= 0.0 {
        return None;
    }
    let sxy = pts.iter().map(|p| (p.0 - xbar) * (p.1 - ybar)).sum::<f64>();
    let s_hat = -0.5 * (sxy / sxx);
    s_hat.is_finite().then_some(s_hat)
}

/// Print learned per-axis spatial anisotropy for spatial terms to stdout.
pub(crate) fn print_spatial_aniso_scales(spec: &TermCollectionSpec) {
    use gam::smooth::get_spatial_aniso_log_scales;
    use gam::terms::smooth::get_spatial_length_scale;
    for (term_idx, term) in spec.smooth_terms.iter().enumerate() {
        let Some(eta) = get_spatial_aniso_log_scales(spec, term_idx) else {
            continue;
        };
        if eta.is_empty() {
            continue;
        }
        let ls = get_spatial_length_scale(spec, term_idx);
        match ls {
            Some(ls) => cli_out!(
                "[spatial-kappa] term {} (\"{}\"): anisotropic length scales (global length_scale={:.4})",
                term_idx,
                term.name,
                ls
            ),
            None => cli_out!(
                "[spatial-kappa] term {} (\"{}\"): pure Duchon shape anisotropy",
                term_idx,
                term.name
            ),
        }
        for (a, &eta_a) in eta.iter().enumerate() {
            if let Some(ls) = ls {
                let length_a = ls * (-eta_a).exp();
                let kappa_a = (1.0 / ls) * eta_a.exp();
                cli_out!(
                    "  axis {}: eta={:+.4}, length={:.4}, kappa={:.4}",
                    a,
                    eta_a,
                    length_a,
                    kappa_a
                );
            } else {
                cli_out!("  axis {}: eta={:+.4}", a, eta_a);
            }
        }
    }
}
