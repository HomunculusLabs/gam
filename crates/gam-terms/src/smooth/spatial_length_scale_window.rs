//! The caller's spatial length-scale window, and the ONE expression that
//! projects onto it.
//!
//! gam#2726: this projection used to be spelled out inline inside each ψ seed
//! constructor in `term_specs.rs`, so the joint `[ρ, ψ]` route seeded ψ from the
//! PROJECTED `length_scale` while the scalar-ρ incumbent it is graded against
//! was fit from the spec's RAW one. On the `length_scale = 1e-3` /
//! `min_length_scale = 1e-2` arm the two routes were therefore evaluating one
//! criterion at two points exactly `ln 10` apart, and the joint route's
//! monotonicity certificate refused with `gap = 98.857` against
//! `accept_tol = 3.873e-5` while asserting `AT THE SAME POINT theta0` in its own
//! message. Sharing θ₀ collapses that gap to `7.185e-11`.
//!
//! Two consequences shape this module:
//!
//! * **One expression.** Both seed constructors and the upstream spec
//!   projection derive ψ through [`spatial_term_seed_psi`] →
//!   [`project_spatial_length_scale_into_window`], so the routes cannot drift
//!   apart again by editing one of them.
//! * **Project once, upstream.** [`project_spatial_length_scales_in_spec`]
//!   writes the projected value into the spec BEFORE the baseline fit, so every
//!   later application is idempotent rather than a second, independent
//!   projection. That matters because there are two projection sites onto the
//!   same face (here and the seed's `clamp_to_bounds` against the data-derived
//!   search box): removing either one alone changes nothing at runtime, because
//!   the other re-projects by exactly `−ln 10`.

use super::*;

/// The ONE expression that projects a spatial `length_scale` onto the caller's
/// `[min_length_scale, max_length_scale]` window.
pub fn project_spatial_length_scale_into_window(
    length_scale: f64,
    options: &SpatialLengthScaleOptimizationOptions,
) -> f64 {
    length_scale.clamp(options.min_length_scale, options.max_length_scale)
}

/// The ψ̄ = −ln(length_scale) seed for one spatial term, derived through
/// [`project_spatial_length_scale_into_window`]. Shared by
/// [`SpatialLogKappaCoords::from_length_scales`] and
/// [`SpatialLogKappaCoords::from_length_scales_aniso`] so the isotropic and
/// anisotropic routes cannot drift apart (gam#2726).
pub fn spatial_term_seed_psi(
    spec: &TermCollectionSpec,
    term_idx: usize,
    options: &SpatialLengthScaleOptimizationOptions,
) -> f64 {
    let length_scale =
        get_spatial_length_scale(spec, term_idx).unwrap_or(options.min_length_scale);
    -project_spatial_length_scale_into_window(length_scale, options).ln()
}

/// Project every listed spatial term's `length_scale` onto the caller's window
/// IN THE SPEC, once, before anything is fit from that spec.
///
/// gam#2726 repair candidate (b). The alternative — widening the joint ψ window
/// to contain a raw incumbent — was measured (it is the arm that proves the
/// diagnosis) and is deliberately not what ships: it would admit a
/// `length_scale` BELOW the caller's own `min_length_scale`, i.e. overrule an
/// explicit caller bound to make an internal comparison agree. The ρ analogue
/// (#1464/#2454) widens only as far as the engine's own `RHO_BOUND` and never
/// past a caller constraint. Projecting upstream instead keeps the caller's
/// window authoritative and makes the incumbent and the joint seed the same
/// point by construction — which is what the certificate's
/// `AT THE SAME POINT theta0` premise had been asserting in prose.
///
/// Constant-curvature terms carry a signed-κ chart rather than a log-ℓ chart
/// and are skipped; terms with no explicit `length_scale` are left alone so
/// `reseed_from_data` keeps ownership of their seed.
///
/// Returns `(term_idx, raw, projected)` for every term that actually moved.
pub fn project_spatial_length_scales_in_spec(
    spec: &mut TermCollectionSpec,
    term_indices: &[usize],
    options: &SpatialLengthScaleOptimizationOptions,
) -> Result<Vec<(usize, f64, f64)>, EstimationError> {
    let mut moved = Vec::new();
    for &term_idx in term_indices {
        if constant_curvature_term_spec(spec, term_idx).is_some() {
            continue;
        }
        let Some(raw) = get_spatial_length_scale(spec, term_idx) else {
            continue;
        };
        if !(raw.is_finite() && raw > 0.0) {
            continue;
        }
        let projected = project_spatial_length_scale_into_window(raw, options);
        if projected == raw {
            continue;
        }
        set_spatial_length_scale(spec, term_idx, projected)?;
        moved.push((term_idx, raw, projected));
    }
    Ok(moved)
}
