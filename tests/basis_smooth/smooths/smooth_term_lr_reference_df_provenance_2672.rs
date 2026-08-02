//! #2672: Wood's smoothing-selection-corrected `edf1` must reach the smooth-term
//! LR reference d.f. on models that carry a PARAMETRIC term, not only on the
//! `y ~ s(x)` shape every other fixture uses.
//!
//! `edf1 = 2·tr(F_jj) − tr(F_jj²)` is computed from the coefficient influence
//! `F = H⁻¹X'WX`. `F` transforms by SIMILARITY under the parametric column
//! conditioning — `F_orig = M·F_int·M⁻¹` — and the back-transform used to drop it
//! rather than apply that map, on the stated ground that the primitive was not
//! carried. It was: `left_multiply_by_m` and `right_multiply_by_m_inv` are both
//! defined on the same type. The drop was invisible because
//! `backtransform_external_result` returns early when the conditioning is
//! inactive, and the conditioning is active exactly when the model has a
//! non-intercept parametric column — which no fixture reading `F` back had.
//!
//! With `F` gone, `wood_reference_df` returned `None` for every such model and
//! the whole-term LR test silently fell back to the raw conditional EDF: the
//! anti-conservative reference #1766 replaced, measured there at a 5%-level FPR
//! of ~0.15. The two `smooth_term_lr_size_calibration` fixtures are exactly this
//! shape (`y ~ x + s(z)`).
//!
//! The guard is stated as a CONTRAST rather than as a bare "must be Some", so it
//! cannot pass by the reference d.f. becoming unconditionally available for some
//! unrelated reason: the parametric-term model and the pure-smooth control must
//! BOTH publish `wood_edf1`, and both must satisfy Wood's analytic band
//! `edf ≤ edf1 ≤ 2·edf`.

use gam::smooth::smooth_term_lr_inference_forspec;
use gam::{
    FitConfig, FitRequest, encode_recordswith_inferred_schema, init_parallelism, materialize,
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
    let headers = vec!["y".to_string(), "x".to_string(), "z".to_string()];
    let mut rows = Vec::<StringRecord>::with_capacity(n);
    for i in 0..n {
        let x = i as f64 / (n as f64 - 1.0);
        let z: f64 = rng.random_range(0.0..1.0);
        let eta = 0.3 + 0.8 * x + amplitude * (std::f64::consts::TAU * z).sin();
        let lambda: f64 = eta.exp();
        let y = Poisson::new(lambda).expect("poisson rate").sample(&mut rng) as f64;
        rows.push(StringRecord::from(vec![
            y.to_string(),
            x.to_string(),
            z.to_string(),
        ]));
    }
    encode_recordswith_inferred_schema(headers, rows).expect("encode")
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

/// The block-offset guard, stated so it cannot be satisfied by a coincidence.
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
/// A NULL smooth is the cleanest probe. REML shrinks `s(z)` onto ~nothing, so
/// its effective d.f. must be well under one — while the columns the unshifted
/// window would fold in are the *unpenalized* intercept and parametric `x`,
/// which carry exactly one degree of freedom each. The measured contrast on this
/// fixture was `edf = 0.054` (the penalty-block-trace channel, which is indexed
/// by penalty and therefore immune) against `edf = 2.040` (the influence trace
/// over the unshifted window) — 1 + 1 + 0.04, the offset read off the arithmetic.
#[test]
fn a_null_smooth_does_not_absorb_the_parametric_columns_edf_2672() {
    init_parallelism();
    for seed in 0..3u64 {
        let data = dataset(150, 990 + seed, 0.0);
        let with_parametric = report("y ~ x + s(z)", &data);
        let without = report("y ~ s(z)", &data);
        for (label, r, folded) in [
            ("y ~ x + s(z)", &with_parametric, "the intercept and `x`"),
            ("y ~ s(z)", &without, "the intercept"),
        ] {
            let p = r.ref_df_provenance;
            assert!(
                p.edf < 1.0,
                "{label} (seed {seed}): a null-true s(z) must be shrunk below one \
                 effective degree of freedom; got edf = {} with wood_edf1 = {:?}. \
                 An edf at or above the number of unpenalized parametric columns \
                 means the term's coefficient window is unshifted and has folded \
                 in {folded}, each of which carries exactly 1 d.f.",
                p.edf,
                p.wood_edf1
            );
            // Wood's band must hold on the same block, which it cannot if the
            // window straddles unpenalized columns whose influence eigenvalue is
            // pinned at 1.
            if let Some(edf1) = p.wood_edf1 {
                let slack = 1e-6 * p.edf.abs().max(1.0);
                assert!(
                    edf1 >= p.edf - slack && edf1 <= 2.0 * p.edf + slack,
                    "{label} (seed {seed}): edf1 = {edf1} outside Wood's band \
                     [edf, 2·edf] = [{}, {}]",
                    p.edf,
                    2.0 * p.edf
                );
            }
        }
    }
}

#[test]
fn wood_edf1_reaches_the_reference_df_with_a_parametric_term_2672() {
    init_parallelism();
    let data = dataset(150, 20672, 0.9);

    // The model shape the size-calibration fixtures use: a parametric `x`
    // alongside the smooth, so parametric column conditioning is ACTIVE.
    let conditioned = report("y ~ x + s(z)", &data);
    // The control: no non-intercept parametric column, so the conditioning is
    // inactive and `F` was never dropped even before the repair.
    let unconditioned = report("y ~ s(z)", &data);

    for (label, r) in [
        ("y ~ x + s(z)", &conditioned),
        ("y ~ s(z)", &unconditioned),
    ] {
        let p = r.ref_df_provenance;
        let edf1 = p.wood_edf1.unwrap_or_else(|| {
            panic!(
                "{label}: Wood's edf1 must reach the LR reference d.f.; \
                 `wood_reference_df` returned None, which means the coefficient \
                 influence F = H⁻¹X'WX was unavailable — the #2672 similarity-map \
                 drop. Provenance: {p:?}"
            )
        });
        // Wood's analytic band. `edf1 = 2·tr(F) − tr(F²) = Σ_i λ_i(2 − λ_i)` with
        // the block's influence eigenvalues `λ_i ∈ [0, 1]`, so it can neither
        // fall below `tr(F) = edf` nor exceed `2·edf`. A violation means the
        // block being differenced is not the block whose trace produced `edf`.
        let slack = 1e-6 * p.edf.abs().max(1.0);
        assert!(
            edf1 >= p.edf - slack && edf1 <= 2.0 * p.edf + slack,
            "{label}: edf1 = {edf1} must lie in Wood's band [edf, 2·edf] = \
             [{}, {}]; provenance {p:?}",
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
        // And the reference the test is scored against must be at least that
        // band's lower end: the assembly floors, it never truncates.
        assert!(
            r.ref_df >= edf1 - slack,
            "{label}: ref_df = {} must not fall below edf1 = {edf1}; provenance {p:?}",
            r.ref_df
        );
    }
}
