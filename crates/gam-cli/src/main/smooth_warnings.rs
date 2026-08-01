use super::*;

// The three smooth-structure advisories — separate 1D spatial smooths, a
// smooth/linear feature overlap, and nested-smooth hierarchical ownership — are
// library-owned now (`gam_terms::smooth::structure_warnings`, #2470). They were
// ~215 lines of pure `TermCollectionSpec -> Vec<String>` sitting in the CLI with
// no counterpart anywhere else, so a Python user fitting
// `s(x1,type=tps) + s(x2,type=tps)` got two unrelated 1-D smooths and no word
// about it, while the identical CLI invocation told them to write
// `thinplate(x1,x2)`. SPEC line 10 requires the three surfaces to be unified.
//
// What stays here is the only CLI-specific half: rendering. The library decides
// WHAT to say; `emit_smooth_structure_warnings` below decides that it goes to
// stderr with a stage prefix.
pub(crate) use gam::smooth::collect_smooth_structure_warnings;

pub(crate) fn emit_smooth_structure_warnings(stage: &str, warnings: &[String]) {
    for warning in warnings {
        cli_err!("WARNING [{stage}]: {warning}");
    }
}

/// Build anisotropic spatial-geometry report rows from an optional resolved spec.
pub(crate) fn build_anisotropic_scales_rows(
    spec: Option<&TermCollectionSpec>,
) -> Vec<report::AnisotropicScalesRow> {
    use gam::smooth::get_spatial_aniso_log_scales;
    use gam::terms::smooth::get_spatial_length_scale;
    let Some(spec) = spec else {
        return Vec::new();
    };
    let mut rows = Vec::new();
    for (term_idx, term) in spec.smooth_terms.iter().enumerate() {
        let Some(eta) = get_spatial_aniso_log_scales(spec, term_idx) else {
            continue;
        };
        if eta.is_empty() {
            continue;
        }
        let ls = get_spatial_length_scale(spec, term_idx);
        let axes = eta
            .iter()
            .enumerate()
            .map(|(a, &eta_a)| {
                let (length_a, kappa_a) = if let Some(ls) = ls {
                    (Some(ls * (-eta_a).exp()), Some((1.0 / ls) * eta_a.exp()))
                } else {
                    (None, None)
                };
                (a, eta_a, length_a, kappa_a)
            })
            .collect();
        rows.push(report::AnisotropicScalesRow {
            term_name: term.name.clone(),
            global_length_scale: ls,
            axes,
        });
    }
    rows
}

/// Build measure-jet spectrum report rows from a saved (frozen) spec alone:
/// realized band + spec order, no per-scale λ̂s (those need the rebuilt
/// design's penalty layout). Used when the report runs without a dataset.
pub(crate) fn measure_jet_spectrum_rows_from_spec(
    spec: Option<&TermCollectionSpec>,
) -> Vec<report::MeasureJetSpectrumRow> {
    let Some(spec) = spec else {
        return Vec::new();
    };
    let mut rows = Vec::new();
    for term in &spec.smooth_terms {
        let SmoothBasisSpec::MeasureJet { spec: mj, .. } = &term.basis else {
            continue;
        };
        let Some(frozen) = mj.frozen_quadrature.as_ref() else {
            continue;
        };
        let (Some(&eps_min), Some(&eps_max)) = (frozen.eps_band.first(), frozen.eps_band.last())
        else {
            continue;
        };
        rows.push(report::MeasureJetSpectrumRow {
            term_name: term.name.clone(),
            eps_min,
            eps_max,
            n_scales: frozen.eps_band.len(),
            length_scale: mj.length_scale,
            spec_order_s: mj.order_s,
            per_scale: Vec::new(),
            implied_order: None,
        });
    }
    rows
}

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
