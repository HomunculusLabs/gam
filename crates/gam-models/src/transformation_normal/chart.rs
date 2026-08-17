//! The CTN coefficient chart — the single definition of what `β` means.
//!
//! Everything that turns fitted CTN coefficients into a transformed response
//! goes through this module. That is not a stylistic preference: the family, the
//! predictor, the observed-score path, the generated-regressor Jacobian and the
//! ALO row replay all read the *same* `blocks[0].beta`, so if any one of them
//! reads it through a different chart the model silently means two different
//! things at fit time and at predict time. gam#2680 is exactly that failure —
//! `#2306` moved the likelihood onto the direct-α chart and left three consumers
//! evaluating `Σ_k I_k(y)·γ_k(x)²`, which reproduces a fitted score of
//! `c·z + (1−c)·L` instead of `z` and is *identically correct only at `c = 1`*
//! (hence: passes small fixtures, fails at production `n`).
//!
//! # The chart
//!
//! `β` is `vec(A)` for a `p_resp × p_cov` coefficient matrix `A`. The
//! covariate-side coordinates are
//!
//! ```text
//! α_k(x) = ψ(x)ᵀ A[k, :],      k = 0 .. p_resp−1
//! ```
//!
//! and the transform is **affine in `α`**, with the shape coordinates kept
//! non-negative by the factored Khatri-Rao monotonicity cone
//! (`TransformationNormalFamily::block_linear_constraints`) rather than by a
//! squared latent reparameterization:
//!
//! ```text
//! h(y, x)  = Σ_k value_k(y)      · α_k(x) + offset(x) + ε·(y − median)
//! h'(y, x) = Σ_k derivative_k(y) · α_k(x) + ε
//! L(x)     = Σ_k lower_k         · α_k(x) + offset(x) + ε·(y_lo − median)
//! U(x)     = Σ_k upper_k         · α_k(x) + offset(x) + ε·(y_hi − median)
//! ```
//!
//! with `value = [1, I_1(y), …]`, `derivative = [0, M_1(y), …]`,
//! `lower = [1, 0, …, 0]` and `upper = [1, 1ᵀT_{·1}, …]` — the same I-splines
//! evaluated at the two boundary knots, where every anchored `I_k` is exactly
//! `0` and exactly `1` respectively.
//!
//! Because the chart is affine, the derivative of every one of those four
//! quantities with respect to `A[k, j]` is just `basis_k · ψ_j(x)` — no chart
//! factor. [`ctn_row_geometry`] and [`CtnRowBases`] are the only place that
//! statement is written down.

use super::{
    BasisOptions, Dense, KnotSource, TRANSFORMATION_MONOTONICITY_EPS, create_basis,
    create_ispline_derivative_dense, initializewiggle_knots_from_seed,
};
use crate::inference::model::TransformationNormalParameterization;
use ndarray::{Array1, Array2, ArrayView1};

/// Number of leading response-basis columns that carry the unconstrained
/// location field `b(x)` rather than a monotone shape coordinate. The location
/// column is the constant `1` in the value basis and `0` in the derivative
/// basis, and it is the one coordinate the monotonicity cone does not
/// constrain.
pub const CTN_LOCATION_COLUMNS: usize = 1;

/// Fraction of the response span by which the certified support is widened past
/// the observed extremes, so every observation the knots were built from sits
/// STRICTLY inside `[y_lo, y_hi]` rather than on its boundary. A response
/// exactly at an endpoint would make its PIT exactly `0` or `1` and clip, which
/// is a real score for a genuinely extreme observation but a fabricated one for
/// the sample maximum of any finite sample.
pub const CTN_RESPONSE_SUPPORT_GUARD_FRACTION: f64 = 1.0e-3;

/// Response-direction basis rows for one observation, in the chart's own order.
///
/// All four slices are `p_resp` long and are indexed by the same `k` as `alpha`.
#[derive(Clone, Copy, Debug)]
pub struct CtnRowBases<'a> {
    /// `[1, I_1(y_i), …]` — the value basis at this row's response.
    pub value: ArrayView1<'a, f64>,
    /// `[0, M_1(y_i), …]` — the derivative basis at this row's response.
    pub derivative: ArrayView1<'a, f64>,
    /// `[1, 0, …, 0]` — the value basis at the lower support knot.
    pub lower: ArrayView1<'a, f64>,
    /// `[1, 1ᵀT_{·1}, …]` — the value basis at the upper support knot.
    pub upper: ArrayView1<'a, f64>,
}

/// The additive scalars that do not depend on `α`: the composed linear-predictor
/// offset (which enters `h`, `L` and `U` identically) and the three
/// monotonicity-floor terms `ε·(y − median)`.
#[derive(Clone, Copy, Debug)]
pub struct CtnRowFloors {
    /// The composed additive offset for this row.
    pub additive_offset: f64,
    /// `ε·(y_i − median)`.
    pub value_floor: f64,
    /// `ε·(y_lo − median)`.
    pub lower_floor: f64,
    /// `ε·(y_hi − median)`.
    pub upper_floor: f64,
}

/// The transformed response and its support at one row.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CtnRowGeometry {
    /// `h(y_i, x_i)`.
    pub h: f64,
    /// `h'(y_i, x_i)`, always `≥ ε` by construction on a feasible `α`.
    pub h_prime: f64,
    /// `L(x_i) = h(y_lo, x_i)`.
    pub lower: f64,
    /// `U(x_i) = h(y_hi, x_i)`.
    pub upper: f64,
}

/// One affine chart component: `floor + Σ_k basis_k · α_k`.
///
/// The accumulation starts from `basis[0]·α[0] + floor` and then adds the shape
/// terms in index order — the order the family's row build has always used, so
/// routing that build through here is bit-identical rather than merely
/// mathematically equal.
///
/// `basis` is read at every index of `alpha`; a caller that passes mismatched
/// widths gets a bounds panic, which is the correct outcome for a corrupted
/// coefficient layout (the surrounding paths all validate `p_resp` first).
///
/// The arguments are strided views rather than slices deliberately. `α` is a row
/// of `Ψ · Aᵀ`, whose memory layout is the linear-algebra backend's business —
/// requiring contiguity here would make a correct chart evaluation depend on
/// whether a matrix product happened to come back row-major, which is a latent
/// panic waiting for a shape that flips it.
#[inline]
pub fn ctn_chart_component(
    alpha: ArrayView1<'_, f64>,
    basis: ArrayView1<'_, f64>,
    floor: f64,
) -> f64 {
    let mut acc = basis[0] * alpha[0] + floor;
    for k in CTN_LOCATION_COLUMNS..alpha.len() {
        acc += basis[k] * alpha[k];
    }
    acc
}

/// Evaluate the CTN transform geometry at one row from its covariate-side
/// coordinates `α_k(x_i)`.
///
/// This is *the* definition of the chart. Every consumer — the likelihood, the
/// PIT score, the `E[Y|x]` inversion grid, the generated-regressor Jacobian, the
/// ALO row replay — calls it, so none of them can drift onto a different
/// parameterization of the same `β`.
///
/// `chart` is not decoration. A persisted CTN model carries
/// [`TransformationNormalParameterization`] precisely so *"a reader can reject
/// coefficients written under any other chart as a typed mismatch instead of
/// silently reinterpreting them"* — and before gam#2680 every replay path
/// validated that marker and then reinterpreted the coefficients anyway.
/// Requiring it here makes the marker load-bearing: a replay path must name the
/// chart it believes it is evaluating, and adding a second variant is a compile
/// error in exactly one function instead of a silent divergence in five.
#[inline]
pub fn ctn_row_geometry(
    chart: TransformationNormalParameterization,
    alpha: ArrayView1<'_, f64>,
    bases: CtnRowBases<'_>,
    floors: CtnRowFloors,
) -> CtnRowGeometry {
    match chart {
        TransformationNormalParameterization::DirectAlpha => CtnRowGeometry {
            h: ctn_chart_component(
                alpha,
                bases.value,
                floors.additive_offset + floors.value_floor,
            ),
            h_prime: ctn_chart_component(alpha, bases.derivative, TRANSFORMATION_MONOTONICITY_EPS),
            lower: ctn_chart_component(
                alpha,
                bases.lower,
                floors.additive_offset + floors.lower_floor,
            ),
            upper: ctn_chart_component(
                alpha,
                bases.upper,
                floors.additive_offset + floors.upper_floor,
            ),
        },
    }
}

/// The derivative of every component of [`ctn_row_geometry`] with respect to the
/// coordinate `α_k` — which, the chart being affine, is just the basis entry.
///
/// Stated as a function so a consumer that differentiates the transform cannot
/// invent a chart factor the evaluator does not have. The pre-gam#2680
/// generated-regressor Jacobian carried `2·γ_k` here, the derivative of the
/// squared chart, against a value path that had already moved.
#[inline]
pub fn ctn_component_sensitivity(
    chart: TransformationNormalParameterization,
    basis: ArrayView1<'_, f64>,
    k: usize,
) -> f64 {
    match chart {
        TransformationNormalParameterization::DirectAlpha => basis[k],
    }
}

/// Number of knots the I-spline response basis carries for a given degree and
/// internal-knot count.
///
/// The builder integrates a degree-`(response_degree + 1)` B-spline basis, so
/// the seed produces `k_prime = K − 2` interior knots inside a clamped vector
/// with `response_degree + 2` boundary repeats at each end.
pub fn ctn_response_knot_count(
    response_degree: usize,
    response_num_internal_knots: usize,
) -> Result<usize, String> {
    let k_prime = response_num_internal_knots.checked_sub(2).ok_or_else(|| {
        format!(
            "response_num_internal_knots = {response_num_internal_knots}; I-spline contract \
             requires K' = K − 2 ≥ 0, so need K ≥ 2"
        )
    })?;
    Ok(k_prime + 2 * (response_degree + 2))
}

/// The clamped I-spline knot vector for a response column, and with it the
/// **certified response support** `[knots.first, knots.last]` that every PIT
/// normalizes against.
///
/// Interior knots come from the wiggle seed; the boundary repeats are pinned to
/// `[min − guard, max + guard]` with a guard of `0.1 %` of the response span, so
/// every observation used to build them sits strictly inside the support.
///
/// This is a function rather than an inline block in
/// [`super::build_response_basis`] because the support it defines is a *shared*
/// object whenever more than one CTN fit has to produce comparable scores. The
/// cross-fit Stage-1 calibration is exactly that case: it refits the CTN on each
/// fold complement and evaluates the score on the held-out rows, so a
/// fold-local support both (a) fails outright on whichever fold holds out a
/// response extreme — the held-out row is then outside its own fold's certified
/// domain and the PIT refuses it — and (b) when it does not fail, assembles the
/// out-of-fold score from `K` PITs taken against `K` *different* truncations,
/// which is not one latent scale. Resolving it once on the full response and
/// pinning it (`TransformationNormalConfig::response_knots_pinned`) removes both.
pub fn ctn_response_knots(
    response: ArrayView1<'_, f64>,
    response_degree: usize,
    response_num_internal_knots: usize,
) -> Result<Array1<f64>, String> {
    let k_prime = response_num_internal_knots.checked_sub(2).ok_or_else(|| {
        format!(
            "response_num_internal_knots = {response_num_internal_knots}; I-spline contract \
             requires K' = K − 2 ≥ 0, so need K ≥ 2"
        )
    })?;
    // The I-spline builder integrates a degree-`(response_degree + 1)` B-spline
    // basis into a degree-`response_degree` value basis, so the seed-time degree
    // is `response_degree + 1`.
    let mut knots = initializewiggle_knots_from_seed(response, response_degree + 1, k_prime)?;
    let response_min = response.iter().copied().fold(f64::INFINITY, f64::min);
    let response_max = response.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let response_span = (response_max - response_min).abs().max(1.0);
    let support_guard = response_span * CTN_RESPONSE_SUPPORT_GUARD_FRACTION;
    let boundary_repeats = response_degree + 2;
    if knots.len() >= 2 * boundary_repeats {
        for idx in 0..boundary_repeats {
            knots[idx] = response_min - support_guard;
            let right_idx = knots.len() - 1 - idx;
            knots[right_idx] = response_max + support_guard;
        }
    }
    Ok(knots)
}

/// The knot vector a CTN fit will actually use: the pinned one when the config
/// carries it, otherwise one resolved from this fit's own response.
///
/// This is the single decision point for "whose response defines the certified
/// support", which is why it is a named function rather than a branch inside
/// [`super::build_response_basis`] — the cross-fit needs to make that decision
/// and needs to be able to check it (gam#2680).
pub fn ctn_resolved_response_knots(
    response: ArrayView1<'_, f64>,
    response_degree: usize,
    response_num_internal_knots: usize,
    pinned: Option<&Array1<f64>>,
) -> Result<Array1<f64>, String> {
    let expected = ctn_response_knot_count(response_degree, response_num_internal_knots)?;
    match pinned {
        Some(knots) => {
            if knots.len() != expected {
                return Err(format!(
                    "pinned response knot vector has {} entries but degree {response_degree} with \
                     {response_num_internal_knots} internal knots requires {expected}",
                    knots.len()
                ));
            }
            Ok(knots.clone())
        }
        None => ctn_response_knots(response, response_degree, response_num_internal_knots),
    }
}

/// The endpoint value bases `(lower, upper)` for a response-shape coefficient
/// transform `T`.
///
/// An anchored I-spline satisfies `I_k(y_lo) = 0` and `I_k(y_hi) = 1` exactly,
/// so the lower endpoint reads only the location column and the upper endpoint
/// reads the column sums of `T`. Stating the endpoints structurally rather than
/// re-evaluating the basis at the boundary knots is what makes `U − L` exactly
/// the represented support width instead of that width plus two evaluations of
/// round-off.
pub fn ctn_endpoint_bases(transform: &Array2<f64>) -> (Array1<f64>, Array1<f64>) {
    let p_shape = transform.ncols();
    let mut lower = Array1::<f64>::zeros(p_shape + CTN_LOCATION_COLUMNS);
    let mut upper = Array1::<f64>::zeros(p_shape + CTN_LOCATION_COLUMNS);
    lower[0] = 1.0;
    upper[0] = 1.0;
    for col in 0..p_shape {
        upper[col + CTN_LOCATION_COLUMNS] = transform.column(col).sum();
    }
    (lower, upper)
}

/// The three floor scalars for a whole column of responses.
///
/// Returns `(per-row ε·(y_i − median), ε·(y_lo − median), ε·(y_hi − median))`
/// with the support endpoints taken from the fitted knot vector.
pub fn ctn_floor_offsets(
    response: ArrayView1<'_, f64>,
    knots: ArrayView1<'_, f64>,
    response_median: f64,
) -> Result<(Array1<f64>, f64, f64), String> {
    let (Some(&lower_y), Some(&upper_y)) = (knots.first(), knots.last()) else {
        return Err("CTN floor offsets require a non-empty response knot vector".to_string());
    };
    let row_offsets = response.mapv(|y| TRANSFORMATION_MONOTONICITY_EPS * (y - response_median));
    Ok((
        row_offsets,
        TRANSFORMATION_MONOTONICITY_EPS * (lower_y - response_median),
        TRANSFORMATION_MONOTONICITY_EPS * (upper_y - response_median),
    ))
}

/// The modelling interval `[t_q, t_{n_B}]` of the CTN response I-spline basis,
/// where `q = degree + 1` is the degree of the B-splines the value basis
/// integrates and `n_B = len(knots) − q − 1` is how many of them there are.
///
/// Read off the same index arithmetic the evaluator uses
/// (`evaluate_ispline_scalarwith_scratch`) rather than as `knots.first()` /
/// `knots.last()`: on the clamped vectors [`ctn_response_knots`] builds the two
/// agree, but it is the evaluator's interval — not the knot vector's extent —
/// that decides where the basis stops being a spline and starts being an
/// extension, and the extension has to be anchored exactly where the spline ends.
fn ctn_ispline_modelling_interval(
    knots: ArrayView1<'_, f64>,
    degree: usize,
) -> Result<(f64, f64), String> {
    let bs_degree = degree
        .checked_add(1)
        .ok_or_else(|| "CTN response I-spline degree overflow".to_string())?;
    let num_bspline = knots
        .len()
        .checked_sub(bs_degree + 1)
        .filter(|count| *count > bs_degree)
        .ok_or_else(|| {
            format!(
                "CTN response knot vector needs more than {} knots for degree {degree}, got {}",
                2 * bs_degree + 1,
                knots.len()
            )
        })?;
    let (left, right) = (knots[bs_degree], knots[num_bspline]);
    if !(left.is_finite() && right.is_finite() && left < right) {
        return Err(format!(
            "CTN response I-spline modelling interval [{left}, {right}] is degenerate"
        ));
    }
    Ok((left, right))
}

/// Continue the raw I-spline value/derivative bases **affinely** past the two
/// boundary knots, at the basis's own one-sided boundary derivative.
///
/// # Why the CTN transformation cannot saturate outside its knots
///
/// A conditional transformation-normal model *is* the statement
/// `F(y | x) = Φ(h(y | x))`, and gam#2600 removed the endpoint renormalizer that
/// used to truncate it — so the fitted density `φ(h)·h'` is a density on the
/// whole real line and `h` is its quantile map. The shared I-spline evaluator,
/// however, holds `I_k` *constant* outside `[t_q, t_{n_B}]` (its own comment
/// justifies that: a linear continuation would make an I-spline entry negative
/// below the support and greater than one above it, which the `[0, 1]` basis
/// contract forbids), and gam#2695 correctly made the M-spline derivative agree
/// by zeroing the exterior. So outside the knots the CTN transform is
///
/// ```text
/// h(y) = h(y_b) + ε·(y − y_b),   ε = TRANSFORMATION_MONOTONICITY_EPS = 1e-8,
/// ```
///
/// i.e. the model's entire tail behaviour is a readout of a numerical floor.
/// Measured on an intercept-only fit to `Y = exp(N(0,1))` at `n = 256`
/// (gam#2600): `Φ(h(y_lo)) = 2.4e-2` of the model's own predictive mass sits
/// below the tabulated support, the transform needs `Δy ≈ 1.4e8` to carry `Φ(h)`
/// from `Φ(U)` to `0.9999`, and two responses a factor `1.8` apart on the far
/// side of the boundary receive PIT scores identical to seven digits.
///
/// # What this does instead, and why it is not a new modelling choice
///
/// The classical transformation-model answer — Royston-Parmar's linear tails,
/// `mlt`'s `extrapolate` — is to continue the transformation at its own boundary
/// derivative, which is exactly what `apply_linear_extension_from_first_derivative`
/// already does for every *clamped B-spline* value basis in this tree:
///
/// ```text
/// y > right:  I_k(y) = I_k(right) + (y − right)·M_k(right⁻),  M_k(y) = M_k(right⁻)
/// y < left :  I_k(y) = I_k(left)  + (y − left) ·M_k(left⁺),   M_k(y) = M_k(left⁺)
/// ```
///
/// Consequences, all structural rather than tuned:
/// * `h` is `C¹` and strictly increasing on all of `ℝ`, with `h' ≥ ε` outside
///   preserved because `M_k(y_b) ≥ 0` and the monotonicity cone keeps `α ≥ 0`;
/// * `Φ(h)` is a proper CDF whose tails are those of the fitted transform rather
///   than of the floor, so a predictive quantile past the support is a finite
///   arithmetic fact instead of a `1/ε` runaway a consumer must clamp away;
/// * the value basis is **unchanged at and inside** `[left, right]`, so every
///   fitted quantity is bit-identical — `ctn_response_knots` guards the support
///   by `0.1 %` of the response span precisely so every training row is strictly
///   inside, and the two branches agree at the endpoints by construction.
///
/// The exterior I-spline entries do leave `[0, 1]`, which is why this is done
/// here and not in the shared basis: for the CTN the entries are coefficients of
/// a transformation that must keep increasing, not weights that must stay in a
/// simplex, and the survival link warp that shares the evaluator (gam#2695)
/// genuinely wants the saturating convention.
///
/// This does mean the fitted CTN puts mass on the whole line, so a strictly
/// positive response can be extrapolated below zero. That is a property of a
/// Gaussian transformation model on the raw response scale — the same property
/// the likelihood is already maximised under — and it is honest where the clamp
/// was not; a model that must respect a bound belongs on the transformed scale
/// (fit `log y`), exactly as `mlt`'s `log_first` does.
fn ctn_extend_bases_affinely_past_the_knots(
    response: ArrayView1<'_, f64>,
    knots: ArrayView1<'_, f64>,
    degree: usize,
    value: &mut Array2<f64>,
    derivative: &mut Array2<f64>,
) -> Result<(), String> {
    let (left, right) = ctn_ispline_modelling_interval(knots, degree)?;
    // Strict inequalities: an evaluation AT a boundary knot is already the
    // spline's own one-sided value/derivative there (the I-spline value
    // saturates to `I_k(right)` and `create_ispline_derivative_dense`
    // deliberately keeps the interior one-sided slope at the endpoints), which
    // is what the extension is anchored at. `!(y >= left && y <= right)` is not
    // used so that a non-finite `y` — refused upstream by every caller — falls
    // through to the raw evaluation rather than silently acquiring a tail.
    if !response.iter().any(|&y| y < left || y > right) {
        return Ok(());
    }
    let boundary = Array1::from_vec(vec![left, right]);
    let (boundary_value, _) = create_basis::<Dense>(
        boundary.view(),
        KnotSource::Provided(knots),
        degree,
        BasisOptions::i_spline(),
    )
    .map_err(|error| format!("CTN boundary I-spline value basis failed: {error}"))?;
    let boundary_value = boundary_value.as_ref();
    let boundary_derivative =
        create_ispline_derivative_dense(boundary.view(), &knots.to_owned(), degree, 1)
            .map_err(|error| format!("CTN boundary M-spline derivative basis failed: {error}"))?;
    let columns = value.ncols();
    if boundary_value.ncols() != columns || boundary_derivative.ncols() != columns {
        return Err(format!(
            "CTN boundary bases are {}/{} columns wide but the response basis is {columns}",
            boundary_value.ncols(),
            boundary_derivative.ncols()
        ));
    }
    for (row, &y) in response.iter().enumerate() {
        let (end, anchor) = if y < left {
            (0usize, left)
        } else if y > right {
            (1usize, right)
        } else {
            continue;
        };
        let step = y - anchor;
        for column in 0..columns {
            let slope = boundary_derivative[[end, column]];
            value[[row, column]] = boundary_value[[end, column]] + step * slope;
            derivative[[row, column]] = slope;
        }
    }
    Ok(())
}

/// Build the response-direction value and derivative bases `[1, I(y)·T]` and
/// `[0, M(y)·T]` at arbitrary response values, on the frozen knots/degree.
///
/// `transform = None` means the identity chart (`T = I`), which is what
/// [`super::build_response_basis`] constructs and therefore what the fit uses;
/// a persisted model passes its saved `T` so a prediction reproduces the fitted
/// basis exactly. The two callers differing on how the location column is
/// prepended is precisely the class of bug this function exists to remove.
///
/// Outside the fitted knot range the bases are continued affinely at the
/// boundary derivative — see [`ctn_extend_bases_affinely_past_the_knots`] for
/// why a transformation model cannot let its transform saturate there. The
/// continuation is applied to the RAW I-spline frame, before `T`; the chart is
/// linear in the basis, so extending-then-transforming and
/// transforming-then-extending are the same matrix and the raw frame is where
/// the boundary derivative is defined.
pub fn ctn_response_bases_at(
    response: ArrayView1<'_, f64>,
    knots: ArrayView1<'_, f64>,
    degree: usize,
    transform: Option<&Array2<f64>>,
) -> Result<(Array2<f64>, Array2<f64>), String> {
    let response_owned = response.to_owned();
    let (raw_value, _) = create_basis::<Dense>(
        response_owned.view(),
        KnotSource::Provided(knots),
        degree,
        BasisOptions::i_spline(),
    )
    .map_err(|error| format!("CTN response I-spline value basis failed: {error}"))?;
    let mut raw_value = raw_value.as_ref().clone();
    let mut raw_derivative =
        create_ispline_derivative_dense(response_owned.view(), &knots.to_owned(), degree, 1)
            .map_err(|error| format!("CTN response M-spline derivative basis failed: {error}"))?;
    if raw_derivative.ncols() != raw_value.ncols() {
        return Err(format!(
            "CTN response derivative basis has {} columns but the value basis has {}",
            raw_derivative.ncols(),
            raw_value.ncols()
        ));
    }
    ctn_extend_bases_affinely_past_the_knots(
        response_owned.view(),
        knots,
        degree,
        &mut raw_value,
        &mut raw_derivative,
    )?;

    let (shape_value, shape_derivative) = match transform {
        Some(t) => {
            if raw_value.ncols() != t.nrows() {
                return Err(format!(
                    "CTN response transform has {} rows but the I-spline basis has {} columns",
                    t.nrows(),
                    raw_value.ncols()
                ));
            }
            (raw_value.dot(t), raw_derivative.dot(t))
        }
        None => (raw_value, raw_derivative),
    };
    if shape_derivative.ncols() != shape_value.ncols() {
        return Err(format!(
            "CTN response derivative basis has {} columns but the value basis has {}",
            shape_derivative.ncols(),
            shape_value.ncols()
        ));
    }

    let n = shape_value.nrows();
    let p_resp = shape_value.ncols() + CTN_LOCATION_COLUMNS;
    let mut value = Array2::<f64>::zeros((n, p_resp));
    let mut derivative = Array2::<f64>::zeros((n, p_resp));
    value.column_mut(0).fill(1.0);
    value
        .slice_mut(ndarray::s![.., CTN_LOCATION_COLUMNS..])
        .assign(&shape_value);
    derivative
        .slice_mut(ndarray::s![.., CTN_LOCATION_COLUMNS..])
        .assign(&shape_derivative);
    Ok((value, derivative))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bases<'a>(
        value: &'a [f64],
        derivative: &'a [f64],
        lower: &'a [f64],
        upper: &'a [f64],
    ) -> CtnRowBases<'a> {
        CtnRowBases {
            value: ArrayView1::from(value),
            derivative: ArrayView1::from(derivative),
            lower: ArrayView1::from(lower),
            upper: ArrayView1::from(upper),
        }
    }

    #[test]
    fn chart_component_is_affine_in_alpha() {
        // The defining property: doubling alpha doubles the response of every
        // component about its floor. A squared chart fails this by 4x on the
        // shape block, which is exactly gam#2680.
        let basis = [1.0, 0.4, 0.9];
        let alpha = [0.5, 1.5, 2.5];
        let doubled: Vec<f64> = alpha.iter().map(|a| 2.0 * a).collect();
        let floor = -0.25;
        let base = ctn_chart_component(ArrayView1::from(&alpha[..]), ArrayView1::from(&basis[..]), floor);
        let twice = ctn_chart_component(
            ArrayView1::from(&doubled[..]),
            ArrayView1::from(&basis[..]),
            floor,
        );
        assert!(
            ((twice - floor) - 2.0 * (base - floor)).abs() < 1e-14,
            "chart is not affine: base={base} twice={twice}"
        );
    }

    #[test]
    fn row_geometry_matches_the_hand_written_chart() {
        let value = [1.0, 0.25, 0.75];
        let derivative = [0.0, 0.6, 1.1];
        let lower = [1.0, 0.0, 0.0];
        let upper = [1.0, 1.0, 1.0];
        let alpha = [-1.5, 0.8, 1.2];
        let floors = CtnRowFloors {
            additive_offset: 0.1,
            value_floor: 1.0e-9,
            lower_floor: -2.0e-9,
            upper_floor: 3.0e-9,
        };
        let g = ctn_row_geometry(
            TransformationNormalParameterization::DirectAlpha,
            ArrayView1::from(&alpha[..]),
            bases(&value, &derivative, &lower, &upper),
            floors,
        );
        let expect_h = 0.1 + 1.0e-9 + 1.0 * -1.5 + 0.25 * 0.8 + 0.75 * 1.2;
        let expect_hp = TRANSFORMATION_MONOTONICITY_EPS + 0.6 * 0.8 + 1.1 * 1.2;
        let expect_lo = 0.1 - 2.0e-9 + -1.5;
        let expect_hi = 0.1 + 3.0e-9 + -1.5 + 0.8 + 1.2;
        assert!((g.h - expect_h).abs() < 1e-12, "h {} vs {expect_h}", g.h);
        assert!(
            (g.h_prime - expect_hp).abs() < 1e-12,
            "h' {} vs {expect_hp}",
            g.h_prime
        );
        assert!((g.lower - expect_lo).abs() < 1e-12);
        assert!((g.upper - expect_hi).abs() < 1e-12);
    }

    #[test]
    fn endpoint_bases_read_the_location_column_and_the_transform_column_sums() {
        let transform = ndarray::array![[1.0, 0.0], [0.5, 2.0], [0.25, -1.0]];
        let (lower, upper) = ctn_endpoint_bases(&transform);
        assert_eq!(lower.to_vec(), vec![1.0, 0.0, 0.0]);
        assert_eq!(upper.to_vec(), vec![1.0, 1.75, 1.0]);
    }

    #[test]
    fn support_width_is_the_shape_coordinates_own_scale() {
        // U − L on the identity chart is exactly Σ_k α_k: the represented span
        // of the transformation. Under a squared chart it would be Σ_k α_k²,
        // which is the quantity gam#2680's over-dispersion is a readout of.
        let transform = Array2::<f64>::eye(3);
        let (lower, upper) = ctn_endpoint_bases(&transform);
        let alpha = [-2.0, 0.7, 1.3, 0.5];
        let floors = CtnRowFloors {
            additive_offset: 0.0,
            value_floor: 0.0,
            lower_floor: 0.0,
            upper_floor: 0.0,
        };
        let value = [1.0, 0.0, 0.0, 0.0];
        let derivative = [0.0, 1.0, 1.0, 1.0];
        let g = ctn_row_geometry(
            TransformationNormalParameterization::DirectAlpha,
            ArrayView1::from(&alpha[..]),
            bases(
                &value,
                &derivative,
                lower.as_slice().expect("contiguous"),
                upper.as_slice().expect("contiguous"),
            ),
            floors,
        );
        assert!((g.upper - g.lower - (0.7 + 1.3 + 0.5)).abs() < 1e-12);
    }

    #[test]
    fn response_bases_carry_the_location_column_and_a_zero_derivative_column() {
        let knots = Array1::from_vec(vec![
            -1.2, -1.2, -1.2, -1.2, -1.2, 0.0, 1.2, 1.2, 1.2, 1.2, 1.2,
        ]);
        let y = Array1::from_vec(vec![-1.2, -0.3, 0.4, 1.2]);
        let (value, derivative) =
            ctn_response_bases_at(y.view(), knots.view(), 3, None).expect("bases");
        assert_eq!(value.nrows(), 4);
        assert_eq!(value.ncols(), derivative.ncols());
        for i in 0..4 {
            assert_eq!(value[[i, 0]], 1.0, "location column must be 1");
            assert_eq!(derivative[[i, 0]], 0.0, "location column carries no slope");
        }
        // Anchored I-splines: every shape column is 0 at the left boundary knot
        // and 1 at the right one — the structural fact `ctn_endpoint_bases`
        // encodes.
        for k in CTN_LOCATION_COLUMNS..value.ncols() {
            assert!(
                value[[0, k]].abs() < 1e-12,
                "I_{k}(y_lo) = {} is not 0",
                value[[0, k]]
            );
            assert!(
                (value[[3, k]] - 1.0).abs() < 1e-12,
                "I_{k}(y_hi) = {} is not 1",
                value[[3, k]]
            );
        }
    }

    /// A clamped I-spline knot vector for `degree = 3`: `degree + 2 = 5`
    /// boundary repeats at each end, one interior knot, support `[-1.2, 1.2]`.
    fn tail_knots() -> Array1<f64> {
        Array1::from_vec(vec![
            -1.2, -1.2, -1.2, -1.2, -1.2, 0.0, 1.2, 1.2, 1.2, 1.2, 1.2,
        ])
    }

    #[test]
    fn response_bases_continue_affinely_past_the_knots_2600() {
        // The defect gam#2600 left behind: `I_k` saturated outside the knot
        // range and `M_k` was zero there, so the CTN transform's whole exterior
        // was `h(y_b) + ε·(y − y_b)` with `ε = 1e-8`. The basis must instead
        // continue at its own boundary derivative.
        let knots = tail_knots();
        let (y_lo, y_hi) = (-1.2_f64, 1.2_f64);
        let steps = [1.0e-6_f64, 0.25, 3.0, 250.0];
        let mut points = vec![y_lo, y_hi];
        for step in steps {
            points.push(y_lo - step);
            points.push(y_hi + step);
        }
        let y = Array1::from_vec(points);
        let (value, derivative) =
            ctn_response_bases_at(y.view(), knots.view(), 3, None).expect("bases with tails");

        // Row 0 / row 1 are the two anchors; every exterior row is the anchor
        // plus its own distance times the anchor's slope, and carries exactly
        // the anchor's slope.
        for (index, &point) in y.iter().enumerate().skip(2) {
            let (anchor_row, anchor) = if point < y_lo { (0, y_lo) } else { (1, y_hi) };
            let step = point - anchor;
            for k in CTN_LOCATION_COLUMNS..value.ncols() {
                let slope = derivative[[anchor_row, k]];
                let expected = value[[anchor_row, k]] + step * slope;
                assert!(
                    (value[[index, k]] - expected).abs() <= 1.0e-12 * expected.abs().max(1.0),
                    "column {k} at y={point} is {} not the affine continuation {expected}",
                    value[[index, k]]
                );
                assert!(
                    (derivative[[index, k]] - slope).abs() <= 1.0e-15,
                    "column {k} at y={point} has slope {} not the boundary slope {slope}",
                    derivative[[index, k]]
                );
            }
        }

        // Non-degeneracy: the continuation would be vacuous if the boundary
        // derivative were zero, which is exactly the state this replaces.
        let lower_slope: f64 = derivative
            .row(0)
            .iter()
            .skip(CTN_LOCATION_COLUMNS)
            .sum::<f64>();
        let upper_slope: f64 = derivative
            .row(1)
            .iter()
            .skip(CTN_LOCATION_COLUMNS)
            .sum::<f64>();
        assert!(
            lower_slope > 1.0e-3 && upper_slope > 1.0e-3,
            "the boundary derivatives the continuation is anchored at are degenerate: \
             lower={lower_slope:.6e} upper={upper_slope:.6e}"
        );
    }

    #[test]
    fn response_bases_are_c1_across_both_boundary_knots_2600() {
        // The invariant that separates "extrapolates" from "saturates" with no
        // tolerance to argue about: `M_k` has no jump at the boundary. Before
        // this, the jump was the whole boundary derivative (O(1) inside, exactly
        // 0 outside).
        let knots = tail_knots();
        let delta = 1.0e-9_f64;
        let y = Array1::from_vec(vec![-1.2 - delta, -1.2 + delta, 1.2 - delta, 1.2 + delta]);
        let (value, derivative) =
            ctn_response_bases_at(y.view(), knots.view(), 3, None).expect("bases across the knots");
        for (outside, inside) in [(0usize, 1usize), (3usize, 2usize)] {
            // Scale every comparison by the largest slope the interior row
            // carries: the boundary derivative lives in one column, and the
            // question is whether that column survives the crossing at all.
            let scale = derivative
                .row(inside)
                .iter()
                .fold(0.0_f64, |acc, value| acc.max(value.abs()));
            assert!(
                scale > 1.0e-3,
                "the interior boundary derivative is already degenerate ({scale:.6e}); \
                 there is nothing for the continuation to match"
            );
            for k in CTN_LOCATION_COLUMNS..value.ncols() {
                let jump = (derivative[[outside, k]] - derivative[[inside, k]]).abs();
                assert!(
                    jump <= 1.0e-4 * scale,
                    "M_{k} jumps by {jump:.6e} across a boundary knot (inside {:.6e}, \
                     outside {:.6e}); before gam#2600's continuation this jump WAS the whole \
                     boundary derivative",
                    derivative[[inside, k]],
                    derivative[[outside, k]]
                );
                let value_jump = (value[[outside, k]] - value[[inside, k]]).abs();
                assert!(
                    value_jump <= 1.0e-4 * scale.max(1.0),
                    "I_{k} jumps by {value_jump:.6e} across a boundary knot"
                );
            }
        }
    }

    #[test]
    fn the_transform_is_strictly_increasing_and_unbounded_on_the_whole_line_2600() {
        // The modelling consequence, on a feasible coefficient vector: with
        // `α ≥ 0` (the Khatri-Rao monotonicity cone) the transform must run from
        // −∞ to +∞ so that `F = Φ(h)` is a proper CDF. Before this it ran
        // between two finite plateaux joined by the `1e-8` floor, so the model
        // needed `Δy ~ 1e8` to spend its tail mass.
        let knots = tail_knots();
        let probe = Array1::from_vec(vec![-1.2, 1.2]);
        let (probe_value, _) =
            ctn_response_bases_at(probe.view(), knots.view(), 3, None).expect("endpoint bases");
        let p_resp = probe_value.ncols();
        // `α₀ = 0` and every shape coordinate 1: a feasible, strictly monotone
        // transformation on the Khatri-Rao cone.
        let mut alpha = Array1::from_elem(p_resp, 1.0);
        alpha[0] = 0.0;
        let (lower_basis, upper_basis) =
            ctn_endpoint_bases(&Array2::<f64>::eye(p_resp - CTN_LOCATION_COLUMNS));
        let floors = CtnRowFloors {
            additive_offset: 0.0,
            value_floor: 0.0,
            lower_floor: 0.0,
            upper_floor: 0.0,
        };
        let far = Array1::from_vec(vec![-1.0e3, -1.2, 0.0, 1.2, 1.0e3]);
        let (value, derivative) =
            ctn_response_bases_at(far.view(), knots.view(), 3, None).expect("far bases");
        let mut previous = f64::NEG_INFINITY;
        let mut extremes = Vec::new();
        for row in 0..far.len() {
            let value_row = value.row(row);
            let derivative_row = derivative.row(row);
            let geometry = ctn_row_geometry(
                TransformationNormalParameterization::DirectAlpha,
                alpha.view(),
                bases(
                    value_row.as_slice().expect("contiguous value row"),
                    derivative_row.as_slice().expect("contiguous slope row"),
                    lower_basis.as_slice().expect("contiguous"),
                    upper_basis.as_slice().expect("contiguous"),
                ),
                floors,
            );
            assert!(
                geometry.h > previous,
                "h is not strictly increasing at y={}: {} <= {previous}",
                far[row],
                geometry.h
            );
            previous = geometry.h;
            assert!(
                geometry.h_prime > 1.0e-3,
                "h' at y={} collapsed to {:.6e}; the exterior slope must be the boundary \
                 derivative, not the monotonicity floor",
                far[row],
                geometry.h_prime
            );
            if row == 0 || row + 1 == far.len() {
                extremes.push(geometry.h);
            }
        }
        // `h` really does reach the far tails of the standard normal: at
        // `|y| = 1e3` the latent is hundreds of sigma, not the `±U` plateau.
        assert!(
            extremes[0] < -1.0e2 && extremes[1] > 1.0e2,
            "the transform is still bounded outside its knots: h(-1e3)={:.6e}, h(1e3)={:.6e}",
            extremes[0],
            extremes[1]
        );
    }

    #[test]
    fn response_bases_inside_the_knots_are_untouched_by_the_continuation_2600() {
        // The fit only ever evaluates strictly inside the certified support
        // (`ctn_response_knots` guards it by 0.1 % of the response span), so the
        // continuation must be bit-identical there or it would silently move
        // every fitted model.
        let knots = tail_knots();
        let interior = Array1::from_vec(vec![-1.2, -0.9, -0.3, 0.0, 0.4, 1.1, 1.2]);
        let (interior_value, interior_derivative) =
            ctn_response_bases_at(interior.view(), knots.view(), 3, None).expect("interior");
        // Same points, evaluated in a batch that also contains exterior rows, so
        // the continuation branch is definitely taken.
        let mut mixed = interior.to_vec();
        mixed.push(-9.0);
        mixed.push(9.0);
        let mixed = Array1::from_vec(mixed);
        let (mixed_value, mixed_derivative) =
            ctn_response_bases_at(mixed.view(), knots.view(), 3, None).expect("mixed");
        for row in 0..interior.len() {
            for k in 0..interior_value.ncols() {
                assert_eq!(
                    interior_value[[row, k]].to_bits(),
                    mixed_value[[row, k]].to_bits(),
                    "interior value moved at (row {row}, column {k})"
                );
                assert_eq!(
                    interior_derivative[[row, k]].to_bits(),
                    mixed_derivative[[row, k]].to_bits(),
                    "interior derivative moved at (row {row}, column {k})"
                );
            }
        }
    }

    #[test]
    fn response_bases_apply_a_non_identity_transform() {
        let knots = Array1::from_vec(vec![
            -1.2, -1.2, -1.2, -1.2, -1.2, 0.0, 1.2, 1.2, 1.2, 1.2, 1.2,
        ]);
        let y = Array1::from_vec(vec![-0.3, 0.4]);
        let (raw_value, _) = ctn_response_bases_at(y.view(), knots.view(), 3, None).expect("raw");
        let p_shape = raw_value.ncols() - CTN_LOCATION_COLUMNS;
        let transform = Array2::<f64>::eye(p_shape) * 2.0;
        let (scaled, _) =
            ctn_response_bases_at(y.view(), knots.view(), 3, Some(&transform)).expect("scaled");
        for i in 0..2 {
            assert_eq!(scaled[[i, 0]], 1.0);
            for k in CTN_LOCATION_COLUMNS..scaled.ncols() {
                assert!((scaled[[i, k]] - 2.0 * raw_value[[i, k]]).abs() < 1e-12);
            }
        }
    }
}
