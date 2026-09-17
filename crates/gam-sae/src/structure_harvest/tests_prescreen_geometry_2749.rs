use super::*;
use ndarray::Array2;

/// A seed wide enough that the `d_k = 2` menu offers the sphere: the menu
/// gates it on `d_seed >= 3`, because `S²` needs three independent seed
/// directions to be identifiable from a great circle at all.
fn wide_seed() -> Array2<f64> {
    Array2::<f64>::from_shape_fn((64, 3), |(r, c)| {
        let t = r as f64 * 0.1 + c as f64;
        t.sin() + 0.5 * (t * 2.0).cos() + c as f64 * 0.25
    })
}

/// Every span band's pre-screen atom is a plan the birth race BUILDS, matched
/// on its full geometry (kind, latent dim, resolution, reference metric) —
/// not on width, which two different atoms can share.
///
/// This is the guard #2749 was missing: the map used to be a table of
/// literals, so when `1dfa70140` deleted the `(lat, lon)` sphere chart the
/// pre-screen went on charging its width 7 and nothing anywhere disagreed.
#[test]
fn curved_prescreen_matches_birth_race_2749() {
    let seed = wide_seed();
    // (span representative, the menu dimension `d_k` its atom is offered at)
    for &(span, d_k) in &[(1.0_f64, 1usize), (2.0, 1), (3.0, 2), (4.0, 2), (9.0, 2)] {
        let plan = SaeAtomGeometryPlan::curved_prescreen_atom_for_span(span)
            .unwrap_or_else(|e| panic!("span {span} must price a buildable atom: {e}"));
        let menu = topology_candidates_for_dim(
            CandidateBases {
                seed: seed.view(),
                ambient: None,
            },
            d_k,
        )
        .unwrap_or_else(|e| panic!("d_k={d_k} menu must build: {e}"));
        let offered: Vec<String> = menu
            .iter()
            .map(|spec| {
                format!(
                    "{:?}/latent{}/{:?}",
                    spec.geometry.kind(),
                    spec.geometry.latent_dim(),
                    spec.geometry.resolution()
                )
            })
            .collect();
        assert!(
            menu.iter().any(|spec| spec.geometry == plan),
            "span {span}: the pre-screen prices {:?}/latent{}/{:?}, which the \
                 d_k={d_k} birth menu does not offer: {offered:?}",
            plan.kind(),
            plan.latent_dim(),
            plan.resolution(),
        );
    }
}

/// The two numbers the pre-screen consumes are theorems of that plan, and the
/// sphere's are `(d = 2, m = (degree+1)² = 9)` — NOT the deleted chart's
/// `(2, 7)`. The sphere is the one atom whose coordinate is wider than the
/// manifold it parameterises, so this also pins that the price uses
/// `intrinsic_dim` (2) and never `latent_dim` (3).
#[test]
fn sphere_is_priced_at_its_realizable_ambient_width_2749() {
    let (d, m) = curved_topology_for_span(3.0).expect("the sphere band must price");
    let degree = SAE_AMBIENT_SPHERE_DEFAULT_DEGREE;
    assert_eq!(d, 2, "S² is intrinsically 2-D whatever its coordinate width");
    assert_eq!(
        m,
        (degree + 1) * (degree + 1),
        "the ambient sphere carries every harmonic through degree {degree}"
    );
    assert_ne!(m, 7, "7 was the width of the chart deleted in 1dfa70140");

    // The neighbouring bands are untouched by #2749 — the reprice moves
    // exactly one band, which is what bounds the acceptance-boundary move.
    assert_eq!(curved_topology_for_span(2.0).expect("circle band"), (1, 3));
    assert_eq!(curved_topology_for_span(4.0).expect("torus band"), (2, 25));
}

/// POSITIVE CONTROL for the whole mechanism: the reason a literal `7` could
/// survive is that nothing ever tried to BUILD the atom it named. Building it
/// is now the only way to obtain a width, and the chart form is refused — so
/// the #2749 defect can no longer be expressed here.
#[test]
fn the_deleted_sphere_chart_form_is_unbuildable_2749() {
    let refused = SaeAtomGeometryPlan::new(
        SaeAtomBasisKind::Sphere,
        2,
        SaeBasisResolution::AmbientSphereHarmonics {
            degree: SAE_AMBIENT_SPHERE_DEFAULT_DEGREE,
        },
        SaeReferenceMetricPlan::RoundSphere,
    );
    assert!(
        refused.is_err(),
        "a 2-coordinate sphere is the deleted chart; the constructor must refuse it"
    );
    // ... while the form the pre-screen actually asks for does build, or the
    // control above would pass for the wrong reason.
    assert!(
        SaeAtomGeometryPlan::curved_prescreen_atom_for_span(3.0).is_ok(),
        "the ambient sphere must remain buildable, or this control is vacuous"
    );
}

/// The reprice is MONOTONE and its size is closed-form: widening `m` by `Δm`
/// lowers the predicted birth saving by exactly `Δm·P·½log₂N` bits and
/// changes nothing else, so a span-3 birth can only be DEFERRED by #2749,
/// never newly admitted. The priority only orders proposals (#2933 F22); the
/// e-process gate is the sole arbiter.
#[test]
fn repricing_the_sphere_only_defers_2749() {
    let (d, m) = curved_topology_for_span(3.0).expect("the sphere band must price");
    let base = BirthMdlPrescreen {
        rho: 0.05,
        span: 3.0,
        intrinsic_dim: d,
        basis_size: m,
        signal_var: 12.0,
        noise_floor: 1.0,
        n_tokens: 2000.0,
        p_out: 8,
        g_dict: 1024,
        l0: 32.0,
    };
    let deleted_chart_width = 7usize;
    let at_chart_width = birth_proposal_priority(&BirthMdlPrescreen {
        basis_size: deleted_chart_width,
        ..base
    })
    .bits()
    .expect("a firing candidate with a positive noise floor has a finite priority");
    let at_realizable_width = birth_proposal_priority(&base)
        .bits()
        .expect("a firing candidate with a positive noise floor has a finite priority");
    assert!(
        at_realizable_width < at_chart_width,
        "pricing the realizable atom must be the more conservative of the two \
             (chart {at_chart_width}, realizable {at_realizable_width})"
    );
    let expected_drop = (m as f64 - deleted_chart_width as f64)
        * base.p_out as f64
        * 0.5
        * base.n_tokens.log2();
    let observed_drop = at_chart_width - at_realizable_width;
    assert!(
        (observed_drop - expected_drop).abs()
            <= expected_drop.abs() * 8.0 * f64::EPSILON + f64::EPSILON,
        "the reprice must move the pre-screen by exactly the BIC decoder-column \
             delta: expected {expected_drop}, observed {observed_drop}"
    );
}
