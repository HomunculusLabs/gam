//! Regression (#2680, out-of-fold half): every fold of a cross-fit CTN Stage-1
//! must score against the **same** certified response support.
//!
//! ## Why the support has to be shared
//!
//! A CTN's latent score is the probability-integral transform of `h(y | x)`
//! against a standard normal **truncated to `[y_lo, y_hi]`**, the span of the
//! fitted response knot vector:
//!
//! ```text
//! z = Φ⁻¹( (Φ(h) − Φ(L)) / (Φ(U) − Φ(L)) ),   L = h(y_lo | x),  U = h(y_hi | x).
//! ```
//!
//! The cross-fit Stage-1 refits the CTN on each fold complement and evaluates
//! the score on that fold's held-out rows. It already freezes the covariate spec
//! once on full data and pins the response-basis knot *count* once at the
//! smallest complement (#859), both so `p₁ = p_resp · p_cov` is fold-invariant.
//! It did **not** pin the knot *positions*, and those are what set the support:
//! `build_response_basis` seeded them from whichever rows the fold happened to
//! train on, with boundary repeats at `[min − guard, max + guard]` and a guard of
//! `0.1 %` of that fold's span.
//!
//! Two independent things break on a fold-local support:
//!
//! 1. **Refusal.** Whichever fold holds out the global response minimum or
//!    maximum scores it against a support built from a complement that does not
//!    contain it. `transformation_normal_pit_score` then correctly reports it as
//!    outside the certified domain and the whole out-of-fold assembly dies. With
//!    `K` folds this is close to guaranteed; it is the observed failure of
//!    `marginal_slope_neyman_orthogonal_reference::sim_a`/`sim_b`, where the
//!    excursion is `ε·(y_i − y_lo)` to the digit (`9.414625e-10 / 1e-8 =
//!    0.0941…` response units below the fold's own `y_lo`).
//! 2. **Incommensurable scores.** Even with no refusal, `z_oof` would be
//!    stitched from `K` PITs taken against `K` different truncations. Stage 2
//!    requires `z ~ N(0, 1)` and `bms::gradient_paths` refuses a score whose
//!    first two moments miss — but a mixture of differently-truncated PITs has no
//!    single latent scale for that gate to be about. This is #2680's headline
//!    claim ("the stage-1 latent score is not `N(0,1)`") arriving by a second,
//!    independent route from the coefficient-chart one.
//!
//! ## What this pins
//!
//! `ctn_response_knots` resolved on the FULL response, then used verbatim by
//! every fold, gives one support that contains every row by construction. The
//! test asserts the property directly and cheaply, with no fit: that the knot
//! vector a fold complement would produce on its own does **not** contain the
//! held-out extremes, while the pinned full-response vector does, and that
//! `ctn_resolved_response_knots` — the one decision point for whose response
//! defines the support, and the function `build_response_basis` routes through —
//! honours the pin verbatim on a subsample whose own knots would differ.
//!
//! Asserting the geometry rather than a fitted `z_oof` is deliberate: the
//! cross-fit path needs `K` CTN fits, and the CTN inner solve refuses on
//! unrelated grounds often enough (see #2600 and the 96-dimensional
//! constraint-face refusal) that a fit-based gate here would report someone
//! else's defect. The property above is the whole mechanism and it is exact.

use gam::transformation_normal::{
    CTN_RESPONSE_SUPPORT_GUARD_FRACTION, ctn_response_knot_count, ctn_resolved_response_knots,
    ctn_response_knots,
};
use ndarray::Array1;

/// Deterministic response with well-separated extremes: the smallest and
/// largest values sit far outside the bulk, so a complement that omits them
/// produces a visibly narrower support.
fn response_fixture() -> Array1<f64> {
    let mut values: Vec<f64> = (0..60).map(|i| -1.0 + 2.0 * (i as f64) / 59.0).collect();
    values[0] = -4.0;
    values[59] = 5.0;
    Array1::from_vec(values)
}

#[test]
fn ctn_fold_local_response_support_excludes_held_out_extremes_2680() {
    let full = response_fixture();
    let degree = 3usize;
    let internal = 5usize;

    let full_knots = ctn_response_knots(full.view(), degree, internal).expect("full-response knots");
    assert_eq!(
        full_knots.len(),
        ctn_response_knot_count(degree, internal).expect("knot count"),
        "the declared knot count must match what the builder produces"
    );
    let (full_lo, full_hi) = (full_knots[0], full_knots[full_knots.len() - 1]);

    // Every observation is strictly inside the full support: that is what the
    // guard fraction exists for.
    let observed_min = full.iter().copied().fold(f64::INFINITY, f64::min);
    let observed_max = full.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    assert!(
        full_lo < observed_min && observed_max < full_hi,
        "full-response support [{full_lo}, {full_hi}] must strictly contain \
         [{observed_min}, {observed_max}]"
    );
    let span = observed_max - observed_min;
    let guard = span.abs().max(1.0) * CTN_RESPONSE_SUPPORT_GUARD_FRACTION;
    assert!(
        (observed_min - full_lo - guard).abs() < 1e-12
            && (full_hi - observed_max - guard).abs() < 1e-12,
        "the boundary repeats are the observed extremes widened by the guard fraction"
    );

    // A fold complement that holds out both extremes. This is the ordinary case,
    // not a contrived one: with K folds, some fold holds out the minimum and
    // some fold holds out the maximum.
    let complement: Array1<f64> = Array1::from_iter(full.iter().copied().skip(1).take(58));
    let fold_knots =
        ctn_response_knots(complement.view(), degree, internal).expect("fold-complement knots");
    let (fold_lo, fold_hi) = (fold_knots[0], fold_knots[fold_knots.len() - 1]);

    // THE DEFECT, stated as a property: the held-out extremes are outside the
    // support the fold would have built for itself. Scoring them against it is
    // what `transformation_normal_pit_score` refuses.
    assert!(
        full[0] < fold_lo,
        "held-out minimum {} must fall below the fold-local support lower endpoint {fold_lo} — \
         if it does not, the fixture is not exercising the defect",
        full[0]
    );
    assert!(
        full[59] > fold_hi,
        "held-out maximum {} must fall above the fold-local support upper endpoint {fold_hi}",
        full[59]
    );

    // THE FIX: the pinned full-response support contains them.
    assert!(
        full_lo < full[0] && full[59] < full_hi,
        "the pinned full-response support [{full_lo}, {full_hi}] must contain every row"
    );
}

#[test]
fn ctn_response_basis_honours_the_pinned_knot_vector_2680() {
    let full = response_fixture();
    let degree = 3usize;
    let internal = 5usize;
    let full_knots = ctn_response_knots(full.view(), degree, internal).expect("full-response knots");

    // A subsample whose own knots would differ from the full-response ones.
    let complement: Array1<f64> = Array1::from_iter(full.iter().copied().skip(1).take(58));

    // Unpinned: the fold resolves its own support, and it is a different one.
    let unpinned = ctn_resolved_response_knots(complement.view(), degree, internal, None)
        .expect("unpinned fold knots");
    assert!(
        (unpinned[0] - full_knots[0]).abs() > 1e-6,
        "the fixture must be one where the fold's own knots differ from the full-response knots"
    );

    // Pinned: used verbatim, whatever this fold's own response would have said.
    let pinned =
        ctn_resolved_response_knots(complement.view(), degree, internal, Some(&full_knots))
            .expect("pinned fold knots");
    assert_eq!(
        pinned.len(),
        full_knots.len(),
        "the pinned knot vector must be used verbatim"
    );
    for k in 0..full_knots.len() {
        assert_eq!(
            pinned[k], full_knots[k],
            "pinned knot {k} was regenerated instead of used"
        );
    }

    // A pinned support that does not match the declared internal-knot count is a
    // typed refusal, not a silent reinterpretation: the width of the coefficient
    // block depends on it, so a mismatch would produce a `p_resp` that disagrees
    // with the one the cross-fit pinned and the fold-alignment check downstream
    // would fail with a much less informative message.
    let short = Array1::from_iter(full_knots.iter().copied().skip(1));
    let err = ctn_resolved_response_knots(complement.view(), degree, internal, Some(&short))
        .expect_err("a mismatched pinned knot vector must be rejected");
    assert!(
        err.contains("pinned response knot vector"),
        "the rejection must name the pinned vector: {err}"
    );
}
