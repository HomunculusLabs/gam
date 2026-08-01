// PROBE (temporary, #2687/#2716): map the κ profile criterion across the WHOLE
// interval the geometry admits, on the exact fixture whose κ̂ is railed at the
// shipped box.
//
// The box's upper end is `0.5/max‖x‖²`; `constant_curvature_kappa_coverage_sims`
// plants `κ⋆ = 1.5` on a radius-0.6 disk (`κ⋆·R² = 0.54`) and reports κ̂ pinned
// at the cap in 3/3 replicates. Widening the box is only defensible if the
// criterion actually HAS an interior minimum near the truth out there — and if
// the region between the shipped cap and the antipodal fold is well behaved
// rather than full of spurious structure. Neither has ever been looked at.
#[cfg(test)]
mod constant_curvature_kappa_box_probe_tests {
    use super::*;
    use gam_geometry::manifolds::constant_curvature::ConstantCurvature;
    use gam_terms::basis::{
        CenterStrategy, ConstantCurvatureBasisSpec, ConstantCurvatureIdentifiability,
    };

    fn next_unit(state: &mut u64) -> f64 {
        (gam_linalg::utils::splitmix64(state) >> 11) as f64 / (1u64 << 53) as f64
    }
    fn next_gauss(state: &mut u64) -> f64 {
        let u1 = next_unit(state).max(1.0e-12);
        let u2 = next_unit(state);
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }

    /// Byte-for-byte the coverage sims' `dataset_on_m_kappa`.
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

    fn spec_at(kappa: f64, centers: usize) -> ConstantCurvatureBasisSpec {
        ConstantCurvatureBasisSpec {
            center_strategy: CenterStrategy::FarthestPoint {
                num_centers: centers,
            },
            kappa,
            kappa_fixed: false,
            length_scale: 0.0,
            double_penalty: false,
            identifiability: ConstantCurvatureIdentifiability::CenterSumToZero,
        }
    }

    /// Is the monotone descent a property of the BOX's endpoint, or of the
    /// criterion? Sweep far past the fold (where the geometry is still
    /// evaluable, just non-injective) and across three planted truths.
    /// WHICH TERM of the REML criterion is monotone in kappa? The value is
    ///
    ///   V = 0.5*(log|H| - log|S|+ - rank*rho) + 0.5*nu*(1 + ln(2*pi*Dp/nu))
    ///
    /// with H = X'X + lambda*S profiled over rho. If the fill-invariant
    /// effective length L(kappa) is doing its job — holding the basis'
    /// flexibility fixed so only the distance-matrix SHAPE moves — then the
    /// determinant block should be roughly kappa-flat and any descent should be
    /// the deviance actually fitting better.
    #[test]
    fn probe_which_reml_term_is_monotone_in_kappa() {
        for (label, kappa_star, centers) in [
            ("spherical k*=+1.5", 1.5_f64, 6usize),
            ("flat      k*= 0.0", 0.0, 6),
        ] {
            let (feats, y) = dataset_on_m_kappa(120, kappa_star, 0.6, 0.10, 0x5EED_0944_0000_0000);
            eprintln!("\n### {label}");
            eprintln!(
                "kappa      L(kappa)   lambda      rho       edf      Dp         \
                 logdetH    logdetS+   det_block   dev_block   V"
            );
            for &kappa in &[
                -1.35_f64, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.2, 1.35, 1.40,
            ] {
                let mut spec = spec_at(kappa, centers);
                spec.double_penalty = false;
                let Ok(basis) = gam_terms::basis::build_constant_curvature_basis(feats.view(), &spec)
                else {
                    eprintln!("  k={kappa:<9} basis refused");
                    continue;
                };
                let ell = match &basis.metadata {
                    gam_terms::basis::BasisMetadata::ConstantCurvature { length_scale, .. } => {
                        *length_scale
                    }
                    _ => f64::NAN,
                };
                let xs = basis.design.to_dense();
                let (n, p) = xs.dim();
                let mut design = Array2::<f64>::ones((n, p + 1));
                design.slice_mut(ndarray::s![.., 1..]).assign(&xs);
                let mut penalty = Array2::<f64>::zeros((p + 1, p + 1));
                penalty
                    .slice_mut(ndarray::s![1.., 1..])
                    .assign(&basis.active_penalties[0].matrix);
                let y2 = y.view().insert_axis(ndarray::Axis(1));
                let Ok(fit) = gam_solve::gaussian_reml::gaussian_reml_multi_closed_form(
                    design.view(),
                    y2,
                    penalty.view(),
                    None,
                    None,
                ) else {
                    eprintln!("  k={kappa:<9} reml refused");
                    continue;
                };
                let (v, _, _) = constant_curvature_kappa_profile_value_jet(
                    feats.view(),
                    y.view(),
                    &spec_at(kappa, centers),
                )
                .expect("profile value");
                // The residual sum of squares the criterion's deviance block is
                // built from, and the log|H| / log|S|+ determinant block, both
                // recomputed here so the descent can be attributed to a term.
                let resid: f64 = y
                    .iter()
                    .zip(fit.fitted.column(0).iter())
                    .map(|(a, b)| (a - b) * (a - b))
                    .sum();
                eprintln!(
                    "  k={kappa:<8.3} L={ell:<9.5} lam={:<11.4e} rho={:<9.4} edf={:<8.4} \
                     rss={resid:<10.6} sig2={:<10.6} V={v:<12.6}",
                    fit.lambda,
                    fit.rho,
                    fit.edf,
                    fit.sigma2[0],
                );
            }
        }
    }

    #[test]
    fn probe_criterion_far_past_the_fold_and_across_planted_truths() {
        for (label, kappa_star, centers) in [
            ("spherical k*=+1.5", 1.5_f64, 6usize),
            ("flat      k*= 0.0", 0.0, 6),
            ("hyperbolic k*=-1.5", -1.5, 6),
            ("spherical k*=+1.5 (30 centers)", 1.5, 30),
        ] {
            let (feats, y) = dataset_on_m_kappa(120, kappa_star, 0.6, 0.10, 0x5EED_0944_0000_0000);
            let mut max_r2 = 0.0_f64;
            for row in feats.outer_iter() {
                max_r2 = max_r2.max(row.dot(&row));
            }
            eprintln!(
                "\n### {label}: max_r2={max_r2:.6} cap={:.4} fold={:.4}",
                0.5 / max_r2,
                1.0 / max_r2
            );
            let mut best = (f64::INFINITY, f64::NAN);
            for &kappa in &[
                -2.7_f64, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.39, 1.5, 2.0, 2.78, 3.0, 4.0,
                6.0, 10.0, 20.0, 50.0, 100.0, 300.0, 1000.0, 1.0e4, 1.0e6,
            ] {
                match constant_curvature_kappa_profile_value_jet(
                    feats.view(),
                    y.view(),
                    &spec_at(kappa, centers),
                ) {
                    Ok((v, g, _)) => {
                        if v < best.0 {
                            best = (v, kappa);
                        }
                        eprintln!("  k={kappa:<10} t={:<9.4} V={v:<20.10} dV={g:<14.5e}", kappa * max_r2);
                    }
                    Err(e) => eprintln!("  k={kappa:<10} REFUSED {e}"),
                }
            }
            eprintln!("  -> argmin {:.5} (truth {kappa_star})", best.1);
        }
    }

    #[test]
    fn probe_criterion_across_the_whole_admissible_kappa_interval() {
        for (rep, radius, kappa_star) in [(0usize, 0.6_f64, 1.5_f64), (1, 0.6, 1.5)] {
            let seed = 0x5EED_0944_0000_0000_u64 ^ ((rep as u64) << 8);
            let (feats, y) = dataset_on_m_kappa(120, kappa_star, radius, 0.10, seed);
            let mut max_r2 = 0.0_f64;
            for row in feats.outer_iter() {
                max_r2 = max_r2.max(row.dot(&row));
            }
            let fold = 1.0 / max_r2;
            eprintln!(
                "\n=== rep {rep}: max_r2={max_r2:.6}  shipped cap={:.4}  fold={fold:.4}  \
                 kappa*={kappa_star} ===",
                0.5 / max_r2
            );
            eprintln!("kappa        t=k*R^2   V(kappa)              dV/dk            d2V/dk2");
            let mut best = (f64::INFINITY, f64::NAN);
            for i in 0..=60 {
                // Sweep from the hyperbolic wall to just short of the fold.
                let t = -0.98 + (0.98 + 0.998) * (i as f64) / 60.0;
                let kappa = t / max_r2;
                match constant_curvature_kappa_profile_value_jet(
                    feats.view(),
                    y.view(),
                    &spec_at(kappa, 6),
                ) {
                    Ok((v, g, h)) => {
                        if v < best.0 {
                            best = (v, kappa);
                        }
                        eprintln!("{kappa:<12.5} {t:<9.4} {v:<21.12} {g:<16.6e} {h:<16.6e}");
                    }
                    Err(e) => eprintln!("{kappa:<12.5} {t:<9.4} REFUSED {e}"),
                }
            }
            eprintln!(
                "rep {rep}: argmin over the swept grid = {:.5} (t={:.4}), V={:.10}; truth {kappa_star}",
                best.1,
                best.1 * max_r2,
                best.0
            );
        }
    }
}
