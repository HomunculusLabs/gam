// #2747: the curvature criterion must identify κ⋆ at a range it was NOT handed.
//
// This file replaces the #2687/#2716 probe module that mapped `V_p(κ)` across
// the admissible interval and reported it monotone. That question is answered:
// the descent was the RANGE coordinate leaking into `dκ` through the fill
// rule's slice, and both the map and the rail move with `ℓ`, not with the box.
// A probe whose question has an answer is a regression test or it is nothing,
// so the sweep it ran is now an assertion.
//
// The bar is deliberately a 3 × 3 grid rather than a single fixture. The
// pre-#2747 criterion is CORRECT on the diagonal cell where the truth's own
// radial length scale happens to equal the auto `ℓ_ref` — that is the cell the
// shipped acceptance fixture uses — so any gate built on one range cannot see
// the defect at all.
#[cfg(test)]
mod constant_curvature_kappa_range_identification_tests {
    use super::*;
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

    fn spec_at(kappa: f64, centers: usize, length_scale: f64) -> ConstantCurvatureBasisSpec {
        ConstantCurvatureBasisSpec {
            center_strategy: CenterStrategy::FarthestPoint {
                num_centers: centers,
            },
            kappa,
            kappa_fixed: false,
            length_scale,
            length_scale_fixed: false,
            double_penalty: false,
            identifiability: ConstantCurvatureIdentifiability::CenterSumToZero,
        }
    }

    /// `n` chart points in a radius-`radius` disk, and a response that is an
    /// exact member of the κ⋆ span AT `truth_ell` — so the truth is reachable at
    /// the planted curvature and at no other, and any failure is an estimator
    /// defect rather than misspecification.
    fn dataset_in_span(
        n: usize,
        kappa_star: f64,
        radius: f64,
        truth_ell: f64,
        centers: usize,
        noise_sd: f64,
        seed: u64,
    ) -> (Array2<f64>, Array1<f64>) {
        let mut state = seed;
        let mut feats = Array2::<f64>::zeros((n, 2));
        let mut noise = Array1::<f64>::zeros(n);
        for i in 0..n {
            let (x1, x2) = loop {
                let a = 2.0 * next_unit(&mut state) - 1.0;
                let b = 2.0 * next_unit(&mut state) - 1.0;
                if a * a + b * b <= 1.0 {
                    break (a * radius, b * radius);
                }
            };
            feats[(i, 0)] = x1;
            feats[(i, 1)] = x2;
            noise[i] = next_gauss(&mut state);
        }
        let truth = gam_terms::basis::build_constant_curvature_basis(
            feats.view(),
            &spec_at(kappa_star, centers, truth_ell),
        )
        .expect("the planted κ⋆ geometry is inside its own chart");
        let design = truth.design.to_dense();
        let mut y = Array1::<f64>::zeros(n);
        for j in 0..design.ncols() {
            let w = 1.0 / (1.0 + j as f64);
            for i in 0..n {
                y[i] += w * design[(i, j)];
            }
        }
        let mean = y.iter().sum::<f64>() / n as f64;
        let sd = (y.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n as f64).sqrt();
        assert!(sd > 0.0, "the planted κ⋆ = {kappa_star} signal collapsed");
        for i in 0..n {
            y[i] = (y[i] - mean) / sd + noise_sd * noise[i];
        }
        (feats, y)
    }

    /// The auto range the builder picks for this cloud, and the κ box.
    fn seed_range_and_box(feats: &Array2<f64>, centers: usize) -> (f64, f64) {
        let spec = spec_at(0.0, centers, 0.0);
        let realized = gam_terms::basis::constant_curvature_realized_centers(feats.view(), &spec)
            .expect("realized centers");
        let ell = gam_terms::basis::realized_constant_curvature_length_scale(realized.view(), 0.0)
            .expect("auto range");
        let mut max_r2 = 0.0_f64;
        for row in feats.outer_iter().chain(realized.outer_iter()) {
            max_r2 = max_r2.max(row.dot(&row));
        }
        (ell, 0.5 / max_r2)
    }

    /// THE GATE. Three planted curvatures × three planted ranges. The
    /// range-profiled criterion must put its argmin near κ⋆ in every cell —
    /// interior, correct sign on the curved arms, and not curved at all on the
    /// flat one.
    ///
    /// Measured before #2747 on exactly this grid: the criterion was right in
    /// the `1×` column and nowhere else — railed at a box endpoint, sign
    /// inverted, or reporting a confident interior `κ̂ = ∓0.94` on genuinely
    /// FLAT data. Every one of those failures is a range error wearing a
    /// curvature's clothes.
    /// The bias bar, and where it comes from.
    ///
    /// `κ̂` at this fixture size is an estimate from `n = 240` rows at SNR 33
    /// through a 6-center basis, so it is not exact. Measured across the nine
    /// cells of this grid, the largest `|κ̂ − κ⋆|` is **0.19** and the median is
    /// 0.07; the realized `ℓ̂` recovers the planted range to within 3%.
    ///
    /// `0.45` is that maximum plus room for one grid step (0.116) of
    /// platform-to-platform floating-point drift in the argmin, and a little
    /// more. It is not a tolerance chosen to make the test pass — it is more
    /// than twice the observed error, because the failures this gate exists to
    /// catch are not fractions of a unit: the pre-#2747 criterion railed at
    /// `±1.41`, inverted the sign (`κ̂ = −0.35` against a planted `+1.0`), and
    /// read `∓0.94` on flat truth. The two sharp claims below — zero rails,
    /// correct sign in every cell — carry the load and admit no tolerance at
    /// all.
    const KAPPA_BIAS_BAR: f64 = 0.45;

    #[test]
    fn range_profiled_criterion_identifies_kappa_star_at_every_planted_range() {
        // Powered rather than cheap: `κ̂` at n = 120 / noise 0.10 has a sampling
        // spread of ±0.5, which would force a bar too loose to separate a fixed
        // estimator from a broken one. n = 240 at SNR 33 costs a fraction of a
        // second and brings the spread inside 0.31.
        let n = 240usize;
        let centers = 6usize;
        let radius = 0.6_f64;
        let seed = 0x5EED_2747_0000_0000_u64;
        const GRID: usize = 24;

        for &kappa_star in &[-1.0_f64, 0.0, 1.0] {
            for &range_mult in &[0.5_f64, 1.0, 2.0] {
                // The auto range is a property of the cloud, so read it from a
                // throwaway cloud drawn with the same stream before planting.
                let (probe_feats, _) = dataset_in_span(n, 0.0, radius, 1.0, centers, 0.0, seed);
                let (ell_ref, cap) = seed_range_and_box(&probe_feats, centers);
                let (feats, y) = dataset_in_span(
                    n,
                    kappa_star,
                    radius,
                    ell_ref * range_mult,
                    centers,
                    0.03,
                    seed,
                );

                let profile = ConstantCurvatureProfile::new(
                    feats.view(),
                    y.view(),
                    spec_at(0.0, centers, 0.0),
                )
                .expect("profile is constructible on the fixture");
                let mut best = (f64::INFINITY, f64::NAN);
                for i in 0..=GRID {
                    let kappa = -cap + 2.0 * cap * (i as f64) / (GRID as f64);
                    if let Ok((value, _, _)) = profile.evaluate(kappa)
                        && value < best.0
                    {
                        best = (value, kappa);
                    }
                }
                let step = 2.0 * cap / (GRID as f64);
                let (eta_hat, _, outcome) = profile
                    .minimize_over_eta(best.1)
                    .expect("the range box is searchable at the argmin");
                eprintln!(
                    "[#2747] κ⋆={kappa_star:+.2} range={range_mult}×ℓ_ref({ell_ref:.4}): \
                     κ̂={:+.4} (box ±{cap:.4}, step {step:.4}), ℓ̂={:.4} [{outcome:?}]",
                    best.1,
                    eta_hat.exp()
                );
                assert!(
                    best.1.abs() < cap * 0.999,
                    "κ⋆={kappa_star} at {range_mult}×ℓ_ref: κ̂={} is RAILED at the box endpoint \
                     ±{cap}; a railed κ̂ is a readout of the box, not of the data",
                    best.1
                );
                assert!(
                    (best.1 - kappa_star).abs() <= KAPPA_BIAS_BAR,
                    "κ⋆={kappa_star} at {range_mult}×ℓ_ref: κ̂={} is more than the derived bias \
                     bar {KAPPA_BIAS_BAR} from the planted curvature",
                    best.1
                );
                if kappa_star != 0.0 {
                    assert!(
                        best.1.signum() == kappa_star.signum(),
                        "κ⋆={kappa_star} at {range_mult}×ℓ_ref: κ̂={} has the WRONG SIGN — the \
                         geometry verdict is inverted",
                        best.1
                    );
                }
            }
        }
    }

    /// The range coordinate must actually be estimated, and estimated WELL:
    /// `ℓ̂` at the criterion's own argmin must track the planted range across a
    /// factor of four, not sit at the heuristic seed.
    ///
    /// This is the half that separates "the optimizer moved η" from "η is
    /// identified". If `ℓ̂` were pinned at `ℓ_ref` the κ gate above would still
    /// be satisfiable by luck on one cell; it is not satisfiable on three.
    #[test]
    fn the_fitted_range_tracks_the_planted_range() {
        let n = 240usize;
        let centers = 6usize;
        let radius = 0.6_f64;
        let seed = 0x5EED_2747_0000_0001_u64;
        let (probe_feats, _) = dataset_in_span(n, 0.0, radius, 1.0, centers, 0.0, seed);
        let (ell_ref, _) = seed_range_and_box(&probe_feats, centers);

        let mut fitted = Vec::new();
        for &range_mult in &[0.5_f64, 1.0, 2.0] {
            let (feats, y) =
                dataset_in_span(n, 1.0, radius, ell_ref * range_mult, centers, 0.03, seed);
            let profile =
                ConstantCurvatureProfile::new(feats.view(), y.view(), spec_at(0.0, centers, 0.0))
                    .expect("profile is constructible on the fixture");
            let (eta_hat, _, _) = profile
                .minimize_over_eta(1.0)
                .expect("the range box is searchable at the planted κ");
            eprintln!(
                "[#2747 range] planted {:.4} ({range_mult}×ℓ_ref) -> ℓ̂ = {:.4}",
                ell_ref * range_mult,
                eta_hat.exp()
            );
            fitted.push(eta_hat.exp());
        }
        assert!(
            fitted[0] < fitted[1] && fitted[1] < fitted[2],
            "ℓ̂ must be strictly increasing in the planted range; got {fitted:?}"
        );
        // The realized ratio must recover the planted factor of two to within a
        // factor of two itself — a real identification claim, loose enough for
        // the n = 120 sampling error and tight enough to fail a pinned ℓ̂ (which
        // would give ratio 1.0 exactly).
        for (lo, hi) in [(0usize, 1usize), (1, 2)] {
            let ratio = fitted[hi] / fitted[lo];
            assert!(
                ratio > 1.25 && ratio < 4.0,
                "ℓ̂ ratio {ratio} across a planted factor of 2 ({:?}) does not identify the range",
                fitted
            );
        }
    }
}
