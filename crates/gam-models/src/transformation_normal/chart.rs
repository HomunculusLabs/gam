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
    create_ispline_derivative_dense,
};
use crate::inference::model::TransformationNormalParameterization;
use ndarray::{Array1, Array2, ArrayView1};

/// Number of leading response-basis columns that carry the unconstrained
/// location field `b(x)` rather than a monotone shape coordinate. The location
/// column is the constant `1` in the value basis and `0` in the derivative
/// basis, and it is the one coordinate the monotonicity cone does not
/// constrain.
pub const CTN_LOCATION_COLUMNS: usize = 1;

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

/// Build the response-direction value and derivative bases `[1, I(y)·T]` and
/// `[0, M(y)·T]` at arbitrary response values, on the frozen knots/degree.
///
/// `transform = None` means the identity chart (`T = I`), which is what
/// [`super::build_response_basis`] constructs and therefore what the fit uses;
/// a persisted model passes its saved `T` so a prediction reproduces the fitted
/// basis exactly. The two callers differing on how the location column is
/// prepended is precisely the class of bug this function exists to remove.
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
    let raw_derivative =
        create_ispline_derivative_dense(response_owned.view(), &knots.to_owned(), degree, 1)
            .map_err(|error| format!("CTN response M-spline derivative basis failed: {error}"))?;

    let (shape_value, shape_derivative) = match transform {
        Some(t) => {
            if raw_value.as_ref().ncols() != t.nrows() {
                return Err(format!(
                    "CTN response transform has {} rows but the I-spline basis has {} columns",
                    t.nrows(),
                    raw_value.as_ref().ncols()
                ));
            }
            (raw_value.as_ref().dot(t), raw_derivative.dot(t))
        }
        None => (raw_value.as_ref().clone(), raw_derivative),
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
