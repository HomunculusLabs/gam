//! #2754/#2761 regression gate: the realized measure-jet representer range is a
//! property of `(rows, declaration, screening response)` — **not** of the family
//! entry point the fit happened to take.
//!
//! ## The defect this pins
//!
//! `length_scale == 0.0` is an UNRESOLVED request, and the tree carries two
//! resolvers for it: a pure-geometry rule inside the basis builder (the median
//! nearest-node spacing) and the gam#2750 response screen. `fit_standard_model`
//! runs the screen so every standard-fit branch gets the same one. The Bernoulli
//! marginal-slope family has its own entry point and did not pass through it, so
//! the identical declaration on byte-identical rows realized two different spans:
//!
//! ```text
//!   gaussian entry, screened   ell = 2.5197
//!   BMS marginal, geometry     ell = 1.0807     (2.33x apart)
//!   BMS logslope, geometry     ell = 1.0807
//!   ...on the SAME 10 centers, the same extent, the same band floor 1.0807.
//! ```
//!
//! `ell` decides WHICH span the representers occupy and `lambda` cannot move a
//! span, so that is not a tuning difference between entry points; it is a
//! different model reached by typing a different family name. gam#2750 measured
//! the geometry heuristic landing 21.7 nats away from the criterion's global
//! optimum and gam#2761 measured its span floor four orders above the chosen
//! range, so "different" is not "equivalent".
//!
//! ## Why the assertion is exact equality rather than a tolerance
//!
//! The screen is a deterministic function of `(feature columns, response,
//! weights, spec)`. Handed the same four, the two entry points must return the
//! same `f64`. A tolerance here would hide exactly the failure mode being
//! pinned: a second resolver that happens to land nearby on this fixture.
//!
//! The comparison arm pins `learn_length_scale=false` because the standard entry
//! LEARNS the range after seeding it, and this test is about the seed. BMS
//! freezes the dial itself (its coupled marginal/log-slope pair cannot carry a
//! design-moving coordinate), so its realized range IS the seed.

use gam::families::bms::BernoulliMarginalSlopeFitResult;
use gam::terms::smooth::{SmoothBasisSpec, TermCollectionSpec};
use gam::utils::splitmix64;
use gam::{FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula};

const N: usize = 800;
const CENTERS: usize = 10;

struct SplitMix64 {
    state: u64,
}
impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    fn next_unit(&mut self) -> f64 {
        ((splitmix64(&mut self.state) >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
    fn next_normal(&mut self) -> f64 {
        let u1 = self.next_unit().max(1.0e-300);
        let u2 = self.next_unit();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + statrs::function::erf::erf(x / std::f64::consts::SQRT_2))
}

fn beta_true(x1: f64) -> f64 {
    0.2 + 0.9 * x1
}

fn alpha_true(x1: f64, x2: f64) -> f64 {
    -0.2 + 0.7 * (std::f64::consts::PI * x1).sin() + 0.3 * (std::f64::consts::PI * x2).cos()
}

fn dataset() -> gam::data::EncodedDataset {
    let mut rng = SplitMix64::new(0x2754_2026_0811_0007);
    let headers = vec![
        "x1".to_string(),
        "x2".to_string(),
        "y".to_string(),
        "z".to_string(),
    ];
    let mut rng_y = SplitMix64::new(0x2754_2026_0811_0008);
    let records: Vec<csv::StringRecord> = (0..N)
        .map(|_| {
            let x1 = rng.next_unit();
            let x2 = rng.next_unit();
            let z = rng.next_normal();
            let p = normal_cdf(alpha_true(x1, x2) + beta_true(x1) * z).clamp(1e-9, 1.0 - 1e-9);
            let y = f64::from(rng_y.next_unit() < p);
            csv::StringRecord::from(vec![
                format!("{x1:.17e}"),
                format!("{x2:.17e}"),
                format!("{y:.17e}"),
                format!("{z:.17e}"),
            ])
        })
        .collect();
    encode_recordswith_inferred_schema(headers, records).expect("encode entry-point fixture")
}

fn realized_ell(spec: &TermCollectionSpec, what: &str) -> f64 {
    spec.smooth_terms
        .iter()
        .find_map(|term| match &term.basis {
            SmoothBasisSpec::MeasureJet { spec: mj, .. } => Some(mj.length_scale),
            _ => None,
        })
        .unwrap_or_else(|| panic!("{what} resolved spec must carry the mjs term"))
}

/// The frozen center layout, so a range difference cannot be explained away as
/// the two entry points having realized different geometry.
fn realized_geometry(spec: &TermCollectionSpec, what: &str) -> (usize, Vec<f64>) {
    spec.smooth_terms
        .iter()
        .find_map(|term| match &term.basis {
            SmoothBasisSpec::MeasureJet { spec: mj, .. } => {
                let band = mj
                    .frozen_quadrature
                    .as_ref()
                    .map(|q| q.eps_band.clone())
                    .unwrap_or_default();
                let m = match &mj.center_strategy {
                    gam::terms::basis::CenterStrategy::UserProvided(c) => c.nrows(),
                    _ => 0,
                };
                Some((m, band))
            }
            _ => None,
        })
        .unwrap_or_else(|| panic!("{what} resolved spec must carry the mjs term"))
}

fn fit_bms(body: &str, ds: &gam::data::EncodedDataset) -> BernoulliMarginalSlopeFitResult {
    let config = FitConfig {
        family: Some("bernoulli-marginal-slope".to_string()),
        link: Some("probit".to_string()),
        logslope_formula: Some(body.to_string()),
        z_column: Some("z".to_string()),
        ..FitConfig::default()
    };
    match fit_from_formula(&format!("y ~ {body}"), ds, &config) {
        Ok(FitResult::BernoulliMarginalSlope(fit)) => fit,
        Ok(_) => panic!("expected a BernoulliMarginalSlope fit for '{body}'"),
        Err(e) => panic!("bms fit '{body}': {e}"),
    }
}

#[test]
fn measure_jet_auto_range_is_the_same_through_every_family_entry_point_2754() {
    gam::init_parallelism();
    let ds = dataset();

    // Arm 1 — the standard entry, screened against `y` and with the learning
    // dial pinned off so what is read back is the SEED.
    let standard = match fit_from_formula(
        &format!("y ~ mjs(x1, x2, centers={CENTERS}, learn_length_scale=false)"),
        &ds,
        &FitConfig::default(),
    ) {
        Ok(FitResult::Standard(fit)) => fit,
        Ok(_) => panic!("expected a standard fit"),
        Err(e) => panic!("standard fit: {e}"),
    };
    let standard_ell = realized_ell(&standard.resolvedspec, "standard");
    let standard_geometry = realized_geometry(&standard.resolvedspec, "standard");

    // Arm 2 — the BMS entry, same rows, same declaration. Its marginal block is
    // screened against the same `y`, so its seed must be the same number.
    let bms = fit_bms(&format!("mjs(x1, x2, centers={CENTERS})"), &ds);
    let marginal_ell = realized_ell(&bms.marginalspec_resolved, "bms marginal");
    let marginal_geometry = realized_geometry(&bms.marginalspec_resolved, "bms marginal");
    let logslope_ell = realized_ell(&bms.logslopespec_resolved, "bms logslope");

    println!(
        "[#2754 entry-point] standard ell={standard_ell:.6} bms-marginal ell={marginal_ell:.6} \
         bms-logslope ell={logslope_ell:.6} | geometry standard={standard_geometry:?} \
         bms={marginal_geometry:?}"
    );

    // The controls first: a range difference is only a resolver difference if
    // the two entry points realized the same geometry to begin with.
    assert_eq!(
        standard_geometry, marginal_geometry,
        "the two entry points must realize the same measure-jet geometry (centers, band) on the \
         same rows; they did not, so the range comparison below would not be about the resolver"
    );

    assert_eq!(
        standard_ell, marginal_ell,
        "the same mjs declaration on the same rows realized two different representer ranges \
         through two family entry points (standard {standard_ell:.6} vs BMS marginal \
         {marginal_ell:.6}). `length_scale == 0.0` has ONE resolver — the #2750 response screen — \
         and every entry point must reach it; lambda cannot move a span, so this is a different \
         model, not a different tuning."
    );

    // Arm 3 — the transformation-normal entry, which builds its bootstrap
    // covariate design straight from the caller's spec and took the same bypass.
    // Its covariate surface enters the linear predictor of the transformed
    // response, so it screens against the same `y`, so it must land on the same
    // number. A refusal here is reported rather than asserted away: CTN can
    // decline a fixture for reasons that have nothing to do with the range, and
    // a silent `continue` would let this arm rot into a no-op.
    let ctn_config = FitConfig {
        family: Some("transformation-normal".to_string()),
        ..FitConfig::default()
    };
    match fit_from_formula(
        &format!("y ~ mjs(x1, x2, centers={CENTERS}, learn_length_scale=false)"),
        &ds,
        &ctn_config,
    ) {
        Ok(FitResult::TransformationNormal(fit)) => {
            let ctn_ell = realized_ell(&fit.covariate_spec_resolved, "transformation-normal");
            println!("[#2754 entry-point] transformation-normal ell={ctn_ell:.6}");
            assert_eq!(
                standard_ell, ctn_ell,
                "the same mjs declaration on the same rows realized two different representer \
                 ranges through the standard entry ({standard_ell:.6}) and the \
                 transformation-normal entry ({ctn_ell:.6}), both of which screen against `y`"
            );
        }
        Ok(other) => panic!(
            "family=\"transformation-normal\" returned the wrong variant: {:?}",
            std::mem::discriminant(&other)
        ),
        Err(e) => panic!(
            "the transformation-normal arm of this gate must fit — it is the arm that pins the \
             CTN half of the resolver hole: {e}"
        ),
    }

    // The log-slope block is screened against its own target (the first-order
    // score surrogate), so it is NOT required to equal the marginal's range.
    // What it must not be is the unscreened geometry heuristic, which is the
    // band floor by construction (`ell_auto = 1.0 x median nearest-node spacing
    // = eps_band[0]`). That equality is the fingerprint of a term that never
    // reached a resolver.
    let band_floor = marginal_geometry.1.first().copied().unwrap_or(f64::NAN);
    assert!(
        band_floor.is_finite() && band_floor > 0.0,
        "the frozen quadrature must carry a positive band floor"
    );
    assert_ne!(
        logslope_ell, band_floor,
        "the BMS log-slope surface's range is still exactly the geometry heuristic \
         (ell == eps_band[0] == {band_floor:.6}), i.e. it never reached the response screen"
    );
}
