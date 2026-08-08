//! #2672: the LR statistic's own null spectrum must reach the smooth-term
//! reference on models that carry a PARAMETRIC term, not only on the
//! `y ~ s(x)` shape every other fixture uses.
//!
//! The reference is built from `w = eig(2·F_jj − F_jj²)` on the tested block of
//! the coefficient influence `F = H⁻¹X'WX`. `F` transforms by SIMILARITY under
//! the parametric column conditioning — `F_orig = M·F_int·M⁻¹` — and the
//! back-transform used to drop it rather than apply that map, on the stated
//! ground that the primitive was not carried. It was: `left_multiply_by_m` and
//! `right_multiply_by_m_inv` are both defined on the same type. The drop was
//! invisible because `backtransform_external_result` returns early when the
//! conditioning is inactive, and the conditioning is active exactly when the
//! model has a non-intercept parametric column — which no fixture reading `F`
//! back had.
//!
//! With `F` gone the whole spectrum is unavailable and the reference falls back
//! to the unit-weight shape `χ²_{max(edf, null_dim, 1)}` — every retained
//! direction counted as if it were unpenalized, which is the anti-conservative
//! reference #1766 replaced, measured there at a 5%-level FPR of ~0.15. The two
//! `smooth_term_lr_size_calibration` fixtures are exactly this shape
//! (`y ~ x + s(z)`).
//!
//! The guard is stated as a CONTRAST rather than as a bare "must be the spectral
//! lane", so it cannot pass by the spectrum becoming unconditionally available
//! for some unrelated reason: the parametric-term model and the pure-smooth
//! control must BOTH report `NullSpectrum`, both must satisfy Wood's analytic
//! band `edf ≤ Σw ≤ 2·edf`, and both must publish p-values that are the ones
//! their own reference produces.

use gam::smooth::{SmoothLrReferenceSource, smooth_term_lr_inference_forspec};
use gam::{
    FitConfig, FitRequest, FitResult, encode_recordswith_inferred_schema, fit_from_formula,
    init_parallelism, materialize,
};

use csv::StringRecord;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Poisson};

/// `y ~ Poisson(exp(0.3 + 0.8·x + amplitude·sin(2π z)))`. `amplitude > 0` gives
/// the smooth a genuine signal so the fitted term has healthy effective d.f.;
/// `amplitude = 0` makes `z` a pure nuisance covariate and the smooth null-true.
fn dataset(n: usize, seed: u64, amplitude: f64) -> gam::data::EncodedDataset {
    let mut rng = StdRng::seed_from_u64(seed);
    let headers = ["y", "x", "w", "v", "z"]
        .into_iter()
        .map(String::from)
        .collect();
    let mut rows = Vec::<StringRecord>::with_capacity(n);
    for i in 0..n {
        let x = i as f64 / (n as f64 - 1.0);
        // Two further parametric covariates with NO effect on the mean, present
        // only to widen the unpenalized block so the offset's overshoot is large.
        let w: f64 = rng.random_range(-1.0..1.0);
        let v: f64 = rng.random_range(-1.0..1.0);
        let z: f64 = rng.random_range(0.0..1.0);
        let eta = 0.3 + 0.8 * x + amplitude * (std::f64::consts::TAU * z).sin();
        let lambda: f64 = eta.exp();
        let y = Poisson::new(lambda).expect("poisson rate").sample(&mut rng) as f64;
        rows.push(StringRecord::from(vec![
            y.to_string(),
            x.to_string(),
            w.to_string(),
            v.to_string(),
            z.to_string(),
        ]));
    }
    encode_recordswith_inferred_schema(headers, rows).expect("encode")
}

/// `edf_total = tr(F)` for one formula on one dataset, from the ordinary fit
/// entry point. The LR driver refits internally from the same spec and options,
/// and the fit is deterministic, so this is the same fit seen from outside.
fn fit_edf_total(formula: &str, data: &gam::data::EncodedDataset) -> f64 {
    let cfg = FitConfig {
        family: Some("poisson".to_string()),
        ..FitConfig::default()
    };
    let FitResult::Standard(std_fit) =
        fit_from_formula(formula, data, &cfg).unwrap_or_else(|e| panic!("fit {formula}: {e}"))
    else {
        panic!("expected a standard fit for {formula}");
    };
    std_fit
        .fit
        .edf_total()
        .unwrap_or_else(|| panic!("edf_total retained for {formula}"))
}

/// The `s(z)` LR report for one formula on one dataset.
fn report(formula: &str, data: &gam::data::EncodedDataset) -> gam::smooth::SmoothTermLrInference {
    let cfg = FitConfig {
        family: Some("poisson".to_string()),
        ..FitConfig::default()
    };
    let mat = materialize(formula, data, &cfg).expect("materialize");
    let FitRequest::Standard(req) = mat.request else {
        panic!("expected a standard fit request for {formula}");
    };
    let reports = smooth_term_lr_inference_forspec(
        req.data.view(),
        req.y.view(),
        req.weights.view(),
        req.offset.view(),
        &req.spec,
        req.family,
        &req.options,
    )
    .expect("smooth-term LR inference");
    reports
        .into_iter()
        .find(|r| r.name.contains('z'))
        .unwrap_or_else(|| panic!("no s(z) report for {formula}"))
}

/// The block-offset guard, stated as an exact accounting identity so no seed can
/// make it pass or fail by luck.
///
/// `SmoothTerm::coeff_range` is BLOCK-LOCAL while the global coefficient layout
/// is `[intercept | linear | random | smooth]`, so every consumer that indexes a
/// global object with it must shift by `smooth_start` first. The LR driver did
/// not, and it indexes four of them — the influence matrix (per-term EDF and
/// Wood's `edf1`), the weighted Gram and correction inside the WPS trace, and
/// the `tested` column set handed to Lawley, which decides which hypothesis the
/// mean shift is computed for. `smooth_start` is never zero: the intercept alone
/// makes it at least one.
///
/// The identity: an UNPENALIZED column spends exactly one effective degree of
/// freedom, and `edf_total = tr(F)` sums every column's share. So for a model
/// with `u` unpenalized columns (intercept plus the parametric block) and a
/// single smooth,
///
/// ```text
///     edf(smooth) + u  ==  edf_total
/// ```
///
/// exactly. An unshifted window makes the smooth's own EDF *include* those `u`
/// columns, so the left side overshoots by `u` — a margin set by the fixture's
/// own design rather than by a tolerance. Measured on the n=60 fixture before the
/// repair, `y ~ x + s(z)` reported `edf = 2.0404` for a null smooth whose
/// penalty-block-trace value was `0.0537`: `1 + 1 + 0.04`.
///
/// Run with a THREE-column parametric block so the overshoot is 4, not 1, and
/// with the smooth both null and signal-bearing so the identity is exercised at
/// both ends of the shrinkage range.
#[test]
fn per_term_edf_plus_unpenalized_columns_equals_edf_total_2672() {
    init_parallelism();
    // (formula, unpenalized column count = intercept + parametric columns)
    let shapes = [("y ~ s(z)", 1usize), ("y ~ x + w + v + s(z)", 4usize)];
    for amplitude in [0.0_f64, 0.9] {
        for seed in 0..2u64 {
            let data = dataset(180, 4400 + seed, amplitude);
            for (formula, unpenalized) in shapes {
                let r = report(formula, &data);
                let edf_total = fit_edf_total(formula, &data);
                let p = r.ref_df_provenance;
                let lhs = p.edf + unpenalized as f64;
                let slack = 1e-3 * edf_total.abs().max(1.0);
                assert!(
                    (lhs - edf_total).abs() <= slack,
                    "{formula} (amplitude {amplitude}, seed {seed}): the smooth's own \
                     effective d.f. plus its {unpenalized} unpenalized column(s) must \
                     equal the model total exactly — an unpenalized column spends one \
                     full degree of freedom and `edf_total = tr(F)` sums them all. \
                     Got edf(s(z)) = {} and edf_total = {edf_total}, so the left side \
                     is {lhs}. An overshoot of about {unpenalized} means the term's \
                     coefficient window is unshifted and has folded the unpenalized \
                     block into the smooth. Provenance: {p:?}",
                    p.edf
                );
            }
        }
    }
}

#[test]
fn the_null_spectrum_reaches_the_reference_with_a_parametric_term_2672() {
    init_parallelism();
    let data = dataset(150, 20672, 0.9);

    // The model shape the size-calibration fixtures use: a parametric `x`
    // alongside the smooth, so parametric column conditioning is ACTIVE.
    let conditioned = report("y ~ x + s(z)", &data);
    // The control: no non-intercept parametric column, so the conditioning is
    // inactive and `F` was never dropped even before the repair.
    let unconditioned = report("y ~ s(z)", &data);

    for (label, r) in [("y ~ x + s(z)", &conditioned), ("y ~ s(z)", &unconditioned)] {
        let p = &r.ref_df_provenance;
        assert_eq!(
            p.source,
            SmoothLrReferenceSource::NullSpectrum,
            "{label}: the reference must come from the statistic's own null \
             spectrum. The unit-weight fallback means the coefficient influence \
             F = H⁻¹X'WX was unavailable — the #2672 similarity-map drop. \
             Provenance: {p:?}"
        );
        // Wood's analytic band, which is now a statement about the spectrum's
        // first moment: `Σ w_j = Σ_i λ_i(2 − λ_i)` with the block's influence
        // eigenvalues `λ_i ∈ [0, 1]`, so it can neither fall below
        // `tr(F) = edf` nor exceed `2·edf`. A violation means the block being
        // differenced is not the block whose trace produced `edf`.
        let slack = 1e-6 * p.edf.abs().max(1.0);
        assert!(
            p.mean >= p.edf - slack && p.mean <= 2.0 * p.edf + slack,
            "{label}: the spectral mean {} must lie in Wood's band [edf, 2·edf] = \
             [{}, {}]; provenance {p:?}",
            p.mean,
            p.edf,
            2.0 * p.edf
        );
        // The fixture must actually exercise a term with fitted complexity —
        // otherwise the band above is a statement about zero.
        assert!(
            p.edf > 1.5,
            "{label}: the planted sin(2πz) signal must give s(z) real effective \
             d.f. for this guard to constrain anything; got edf = {}",
            p.edf
        );
        // Cauchy–Schwarz on the same spectrum: `(Σw)² ≤ q·Σw²` and
        // `Σw² ≤ (max w)·Σw ≤ Σw`, so the resolved shape and scale must satisfy
        // `ν ≤ q` and `g ≤ 1`, with equality exactly when every weight is one
        // (an unpenalized block). Both bounds are analytic, so a violation is an
        // assembly error rather than a fixture accident.
        let block = &r.ref_df_provenance;
        assert!(
            block.scale > 0.0 && block.scale <= 1.0 + 1e-12,
            "{label}: the reference scale g = Σw²/Σw must lie in (0, 1]; got {}",
            block.scale
        );
        assert!(
            block.chi_square_df >= 1.0 - 1e-9,
            "{label}: ν = (Σw)²/Σw² is at least one whenever any weight is \
             positive; got {}",
            block.chi_square_df
        );
        // `ref_df` is the mean, so it is `ν·g` by construction. Pinning the
        // identity catches a future edit that lets the three drift apart.
        assert!(
            (r.ref_df - block.chi_square_df * block.scale).abs() <= 1e-9 * r.ref_df.abs().max(1.0),
            "{label}: ref_df {} must equal ν·g = {}·{}",
            r.ref_df,
            block.chi_square_df,
            block.scale
        );
        // The published p-values must be the ones this reference produces —
        // the report cannot carry a tail probability from a different law than
        // the one it publishes.
        assert!(
            (r.p_value_uncorrected - block.tail_probability(r.statistic_lr)).abs() <= 1e-12,
            "{label}: p_value_uncorrected {} is not P(χ²_ν > W/g) = {}",
            r.p_value_uncorrected,
            block.tail_probability(r.statistic_lr)
        );
        assert!(
            (r.p_value_corrected - block.tail_probability(r.statistic_corrected)).abs() <= 1e-12,
            "{label}: p_value_corrected {} is not P(χ²_ν > W*/g) = {}",
            r.p_value_corrected,
            block.tail_probability(r.statistic_corrected)
        );
    }
}

/// #2672: the two independent routes to the null spectrum must agree on REAL
/// fits, and the published p-value must be the exact weighted-chi-square tail of
/// the spectrum the report carries.
///
/// The spectrum is assembled from `[H⁻¹]_jj S_jj`; the moments the previous
/// reference used are assembled from traces of powers of the influence block.
/// `(I − F)_jj = [H⁻¹]_jj S_jj` is an identity — but only if the penalty is
/// block-diagonal by term AND `Vb`, `F` and `S(λ)` are published in ONE
/// coefficient basis. Both halves of that have been wrong in this exact code
/// path before: the influence matrix was dropped rather than mapped through the
/// conditioning similarity, the retained first-order correction was left in the
/// internal basis while its Gram was carried out, and the tested window was
/// indexed block-locally into a global object. None of those is visible by
/// reading; all three move this residual off roundoff.
///
/// So the driver measures it on every fit and publishes it, and this pins the
/// published number. The fixtures span both shrinkage regimes (null-true and
/// signal-bearing), both conditioning states (with and without a parametric
/// block), and both scale conventions (`poisson`, whose IRLS weight already
/// carries the dispersion, and `gaussian`, whose coefficient covariance carries
/// `σ̂²` — the multiplier that has to be divided back out of `Vb` before it can
/// be multiplied by a penalty).
#[test]
fn the_two_routes_to_the_null_spectrum_agree_on_real_fits_2672() {
    init_parallelism();
    let mut checked = 0usize;
    let mut largest_shift = 0.0_f64;
    for amplitude in [0.0_f64, 0.9] {
        for &(formula, family) in &[
            ("y ~ s(z)", "poisson"),
            ("y ~ x + s(z)", "poisson"),
            ("y ~ x + w + v + s(z)", "poisson"),
            ("y ~ x + s(z)", "gaussian"),
            ("y ~ x + s(z, k=12)", "poisson"),
        ] {
            let data = dataset(200, 90_672 + (amplitude * 10.0) as u64, amplitude);
            let r = report_for(formula, family, &data);
            let p = &r.ref_df_provenance;
            assert_eq!(
                p.source,
                SmoothLrReferenceSource::NullSpectrum,
                "{formula} [{family}] amplitude={amplitude}: the exact lane must be \
                 reachable on an ordinary fit. Falling back means `beta_covariance()`, \
                 its scale, or the penalty block was unavailable. Provenance: {p:?}"
            );
            let residual = p.moment_residual.unwrap_or_else(|| {
                panic!(
                    "{formula} [{family}] amplitude={amplitude}: both routes were \
                     available on this fit, so the identity between them must have \
                     been MEASURED. Provenance: {p:?}"
                )
            });
            assert!(
                residual < 1e-8,
                "{formula} [{family}] amplitude={amplitude}: the spectrum from \
                 `[H⁻¹]_jj S_jj` and the moments from the influence block disagree by \
                 a relative {residual:.3e}. That is a basis or block-structure \
                 defect, not a tolerance: the two are the same object. \
                 Provenance: {p:?}"
            );

            // The spectrum is a spectrum: `q` weights in `[0,1]`, descending,
            // summing to the published mean.
            assert!(!p.weights.is_empty());
            assert!(
                p.weights.iter().all(|w| (-1e-12..=1.0 + 1e-12).contains(w)),
                "{formula} [{family}]: weights outside [0,1]: {:?}",
                p.weights
            );
            assert!(
                p.weights.windows(2).all(|pair| pair[0] >= pair[1] - 1e-15),
                "{formula} [{family}]: weights not descending: {:?}",
                p.weights
            );
            let sum: f64 = p.weights.iter().sum();
            assert!(
                (sum - p.mean).abs() <= 1e-9 * p.mean.abs().max(1.0),
                "{formula} [{family}]: Σw = {sum} but the published mean is {}",
                p.mean
            );

            // The λ̂-selection replay must be REACHABLE on an ordinary fit —
            // it is the difference between pricing `λ̂` as chosen and pricing it
            // as given, and a model shape that cannot reach it silently gets
            // the reference this issue measured at `size@.05 = 0.0962`.
            let replay = p.selection.as_ref().unwrap_or_else(|| {
                panic!(
                    "{formula} [{family}] amplitude={amplitude}: no selection replay on a \
                     penalized smooth whose λ̂ the outer search chose. Provenance: {p:?}"
                )
            });
            assert!(
                !replay.generalized.is_empty()
                    && replay.generalized.iter().all(|v| v.is_finite() && *v >= 0.0),
                "{formula} [{family}]: the replay's generalized spectrum is not a \
                 spectrum: {:?}",
                replay.generalized
            );

            // The CONDITIONAL half of the published p-value is the exact tail of
            // the spectrum, to the accuracy the report itself certifies. The
            // driver resolves that tail only as finely as `W` is known (see
            // `tail_probability_with_bound`), so the comparison is against
            // `gam-math`'s strict default THROUGH the published bound: the bound
            // has to be honest, not decorative.
            let strict = gam_math::probability::weighted_chi_square_sf(
                &p.weights,
                r.statistic_corrected,
            );
            assert!(
                (r.p_value_conditional - strict).abs() <= r.p_value_bound + 1e-12,
                "{formula} [{family}]: the published conditional p {} differs from the \
                 strictly integrated P(Σ w_j χ²₁ > W*) = {strict} by more than the \
                 accuracy the report certifies ({}). W* = {}",
                r.p_value_conditional,
                r.p_value_bound,
                r.statistic_corrected
            );
            // And the selection is what separates the two published numbers.
            let selection_shift = r.p_value_corrected - r.p_value_conditional;
            assert!(
                selection_shift.is_finite() && selection_shift.abs() <= 1.0,
                "{formula} [{family}]: the selection shift {selection_shift} is not a \
                 probability difference"
            );
            largest_shift = largest_shift.max(selection_shift.abs());
            checked += 1;
        }
    }
    assert!(checked >= 10, "the sweep must actually run: {checked} cells");
    // The replay is not decoration: somewhere in this sweep it has to have moved
    // a published p-value by more than its own Monte-Carlo noise. A replay that
    // never moves anything is the conditional reference with extra steps.
    assert!(
        largest_shift > 1e-3,
        "the λ̂-selection replay never moved a published p-value by more than \
         {largest_shift} across {checked} fits — it is inert"
    );
}

/// #2672: where the two-moment summary is exact, the exact lane must agree with
/// it — and where it is not, the disagreement must have one sign.
///
/// This is the pair of claims that says the change is neither cosmetic nor
/// reckless, and the first half is the one I had backwards before measuring.
///
/// * A term REML has shrunk to its null space collapses to ONE weight of order
///   one over a tail of dust (measured: `w = (0.322, 5.9e-7, 7.1e-8, …)` on a
///   null-true `k = 12` fit). A single distinct weight IS a scaled chi-square,
///   so the surrogate is exact there and the exact lane must reproduce it. The
///   shrunk regime — the one a null-simulation grid spends all its time in — is
///   therefore not where the two references can differ, which is why the
///   `size@.05` grid could never have detected this.
/// * At moderate shrinkage several weights are comparable and unequal, and the
///   surrogate reports a SMALLER tail than the law — one-signed,
///   anti-conservative, growing with the depth of the tail.
#[test]
fn the_two_moment_summary_is_exact_when_shrunk_and_one_signed_otherwise_2672() {
    init_parallelism();

    // Shrunk arm: null-true smooth, spectrum collapses to one weight.
    let shrunk = report("y ~ x + s(z, k=12)", &dataset(200, 20_672, 0.0));
    let p = &shrunk.ref_df_provenance;
    assert_eq!(p.source, SmoothLrReferenceSource::NullSpectrum);
    let dust: f64 = p.weights.iter().skip(1).sum();
    assert!(
        dust < 1e-4 * p.weights[0],
        "the null-true arm is supposed to be the collapsed one: {:?}",
        p.weights
    );
    for multiple in [1.0_f64, 4.0, 16.0] {
        let statistic = multiple * p.mean;
        let exact = gam_math::probability::weighted_chi_square_sf(&p.weights, statistic);
        let summary =
            gam_math::probability::chi_square_sf(statistic / p.scale, p.chi_square_df);
        // The bar is the dust itself, not a tolerance. The only thing separating
        // this spectrum from a single weight — where the two references are
        // IDENTICAL — is `dust`, and the two laws place that mass differently
        // over an interval of length `dust`, so no tail can move by more than
        // the density times it. `10·dust` is that statement with room for a
        // density above one (the scale `g` is 0.32 here, so it is about 1.5).
        assert!(
            (exact - summary).abs() <= 10.0 * dust + 1e-12,
            "on a collapsed spectrum the two references must agree to the dust that \
             separates it from a single weight (dust = {dust:.3e}): exact {exact} vs \
             summary {summary} at W = {multiple}·Σw. Spectrum: {:?}",
            p.weights
        );
    }

    // Signal arm: a fitted smooth with real effective d.f., so several weights
    // are comparable and unequal.
    let signal = report("y ~ x + s(z, k=12)", &dataset(300, 20_673, 0.9));
    let q = &signal.ref_df_provenance;
    assert_eq!(q.source, SmoothLrReferenceSource::NullSpectrum);
    assert!(
        q.chi_square_df > 1.5,
        "the signal arm must carry a spread spectrum for the sign claim to say \
         anything; ν = {} on {:?}",
        q.chi_square_df,
        q.weights
    );
    for multiple in [2.0_f64, 4.0, 8.0] {
        let statistic = multiple * q.mean;
        let exact = gam_math::probability::weighted_chi_square_sf(&q.weights, statistic);
        let summary =
            gam_math::probability::chi_square_sf(statistic / q.scale, q.chi_square_df);
        assert!(
            summary <= exact + 1e-12,
            "at W = {multiple}·Σw the two-moment summary ({summary}) reports a LARGER \
             tail than the law ({exact}); its error is one-signed the other way. \
             Spectrum: {:?}",
            q.weights
        );
    }
}

/// The `s(z)` LR report for one formula/family on one dataset.
fn report_for(
    formula: &str,
    family: &str,
    data: &gam::data::EncodedDataset,
) -> gam::smooth::SmoothTermLrInference {
    let cfg = FitConfig {
        family: Some(family.to_string()),
        ..FitConfig::default()
    };
    let mat = materialize(formula, data, &cfg).expect("materialize");
    let FitRequest::Standard(req) = mat.request else {
        panic!("expected a standard fit request for {formula}");
    };
    let reports = smooth_term_lr_inference_forspec(
        req.data.view(),
        req.y.view(),
        req.weights.view(),
        req.offset.view(),
        &req.spec,
        req.family,
        &req.options,
    )
    .expect("smooth-term LR inference");
    reports
        .into_iter()
        .find(|r| r.name.contains('z'))
        .unwrap_or_else(|| panic!("no s(z) report for {formula} [{family}]"))
}
