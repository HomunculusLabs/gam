use super::*;

// ---------------------------------------------------------------------------
// Response-direction basis construction
// ---------------------------------------------------------------------------

/// Build the response-direction basis: an unconstrained location column plus
/// I-spline values `I_k(y)` with derivatives `M_k(y) = I'_k(y)`.
///
/// Returns (value_basis = `[1, I_k]`, derivative_basis = `[0, M_k]`,
/// penalties embedded with an unpenalized location row/column, regenerated
/// I-spline knots, identity coef_transform for the I-spline shape block).
pub(crate) fn build_response_basis(
    response: &Array1<f64>,
    config: &TransformationNormalConfig,
) -> Result<
    (
        Array2<f64>,
        Array2<f64>,
        Vec<Array2<f64>>,
        Array1<f64>,
        Array2<f64>,
    ),
    String,
> {
    let n = response.len();
    if n < 4 {
        return Err(TransformationNormalError::InvalidInput {
            reason: format!("need at least 4 observations, got {n}"),
        }
        .into());
    }
    for (i, &v) in response.iter().enumerate() {
        if !v.is_finite() {
            return Err(TransformationNormalError::NonFinite {
                reason: format!("response[{i}] is not finite: {v}"),
            }
            .into());
        }
    }

    let response_degree = config.response_degree;
    if response_degree < 1 {
        return Err(TransformationNormalError::InvalidInput {
            reason: format!(
                "response_degree must be >= 1 for the I-spline basis, got {response_degree}"
            ),
        }
        .into());
    }
    // The knot vector, and with it the certified response support `[y_lo, y_hi]`
    // every PIT is normalized against. A caller that has already resolved it
    // supplies it verbatim; see `ctn_response_knots` for why the cross-fit must.
    let knots = ctn_resolved_response_knots(
        response.view(),
        response_degree,
        config.response_num_internal_knots,
        config.response_knots_pinned.as_ref(),
    )?;

    // Response-direction value / derivative bases `[1, I_k(y)]` / `[0, M_k(y)]`,
    // assembled by the chart module so the fit and every replay path (predict,
    // observed score, generated-regressor Jacobian) prepend the location column
    // the same way. Passing `None` for the coefficient transform IS the identity
    // chart this builder then returns below — the fit is the definition of the
    // chart, so it evaluates the untransformed frame. See gam#2680.
    let (resp_val, resp_deriv) =
        ctn_response_bases_at(response.view(), knots.view(), response_degree, None)?;
    let p_resp = resp_val.ncols();
    let p_shape = p_resp - CTN_LOCATION_COLUMNS;
    if resp_val.nrows() != n || resp_deriv.nrows() != n {
        return Err(TransformationNormalError::InvalidInput {
            reason: format!(
                "response basis row counts ({}, {}) do not match n = {n}",
                resp_val.nrows(),
                resp_deriv.nrows()
            ),
        }
        .into());
    }

    // SCOP-CTN coef-transform is identity: the direct-α chart (gam#2306) keeps
    // the I-spline shape coordinates as-is (no square, no reparameterization),
    // and I-splines carry no constant in their span, so no column folding is
    // needed.
    let transform = Array2::<f64>::eye(p_shape);

    // SPEC-5: the response-direction penalty is the EXACT function-space
    // roughness of the represented I-spline value function, not a
    // coefficient-difference operator. For derivative order `m` the shape
    // block carries
    //
    //     S_{y,m} = Cᵀ (∫ B_q^{(m)}(y) B_q^{(m)}(y)ᵀ dy) C,
    //
    // assembled span by span by Gauss–Legendre with the I-spline cumulative
    // frame `C` (see `ispline_function_penalties`). This is a quadratic
    // functional of the represented function itself, so it is scale- and
    // knot-width-aware (a difference operator is not) while remaining exactly
    // quadratic in the shape coefficients, i.e. compatible with the fixed
    // Gaussian-quadratic REML normalizer.
    //
    // In the direct-alpha chart (#2306), this Gram enters
    // `1/2 vec(A)^T (S_{y,m} kron G_x) vec(A)`, so it penalizes the final
    // represented transformation itself rather than a latent square root.
    //
    // The represented value functions have per-span polynomial degree
    // `value_degree = response_degree + 1`; the `m`-th derivative of a
    // degree-`value_degree` piecewise polynomial vanishes identically for
    // `m > value_degree`, so such an order carries no function roughness and
    // is a hard configuration error rather than a silently skipped no-op.
    // Embed each Gram into the full response block with an unpenalized
    // location row/column.
    let value_degree = response_degree + 1;
    let mut resp_penalties = Vec::new();
    let add_penalty = |order: usize, penalties: &mut Vec<Array2<f64>>| -> Result<(), String> {
        if order == 0 {
            return Err(TransformationNormalError::InvalidInput {
                reason: "response penalty derivative order must be >= 1; order 0 is the value \
                         function, not a roughness penalty"
                    .to_string(),
            }
            .into());
        }
        if order > value_degree {
            return Err(TransformationNormalError::InvalidInput {
                reason: format!(
                    "response penalty derivative order {order} exceeds the I-spline value degree \
                     {value_degree}; the {order}-th derivative of the response basis is \
                     identically zero, so this order carries no function-space roughness"
                ),
            }
            .into());
        }
        let function_penalty = ispline_function_penalties(knots.view(), response_degree, order, false)
            .map_err(|e| e.to_string())?;
        let mut shape_pen = function_penalty.roughness;
        if function_penalty.roughness_nullspace_dim == 0 {
            // SPEC null-recovery ("the default should allow a configuration
            // which recovers the null"). An anchored I-spline loses the constant
            // from its order-`m` polynomial null, so the structural nullity is
            // `m − 1`: for `m ≥ 2` the null contains the AFFINE transformation
            // (`h' ≡ const > 0`), which is the map this shape block exists to
            // bend away from and is a perfectly good transformation. For `m = 1`
            // the nullity is 0, so `∫(f')²` is minimised uniquely at `α = 0`,
            // i.e. `h' ≡ 0` — a CONSTANT transformation that sends every
            // response to one score and at which the likelihood is not defined.
            // Such a penalty always has a λ at which the penalized objective
            // prefers the degenerate map to the data (gam#2600 measured
            // `Δobj = +9.987e5` for `≈50` of likelihood at `λ ≈ 1370`).
            //
            // Restore the affine null by penalizing the VARIATION of the slope
            // about its mean instead of the slope itself — the same order of
            // derivative, still an exact function-space functional, and PSD by
            // construction because it is the integral of a square.
            subtract_constant_slope_component(&mut shape_pen, knots.view(), value_degree)?;
        }
        if shape_pen.dim() != (p_shape, p_shape) {
            return Err(TransformationNormalError::InvalidInput {
                reason: format!(
                    "order-{order} I-spline function roughness is {}x{} but the response shape \
                     block has {p_shape} columns",
                    shape_pen.nrows(),
                    shape_pen.ncols(),
                ),
            }
            .into());
        }
        let mut full_pen = Array2::<f64>::zeros((p_resp, p_resp));
        full_pen.slice_mut(s![1.., 1..]).assign(&shape_pen);
        penalties.push(full_pen);
        Ok(())
    };
    add_penalty(config.response_penalty_order, &mut resp_penalties)?;
    for &order in &config.response_extra_penalty_orders {
        if order == config.response_penalty_order {
            continue;
        }
        add_penalty(order, &mut resp_penalties)?;
    }

    Ok((resp_val, resp_deriv, resp_penalties, knots, transform))
}

/// Length of the interval the I-spline roughness Grams integrate over.
///
/// [`bspline_derivative_penalty_matrix`] assembles `∫ B^(m) B^(m)ᵀ` over the
/// spans `[t_k, t_{k+1}]` for `k = value_degree .. num_bspline`, so the modeling
/// interval is `[t_{value_degree}, t_{num_bspline}]`. Reading the endpoints off
/// that same index arithmetic keeps this length in step with the assembler
/// rather than assuming the knot vector is clamped.
fn ispline_domain_length(knots: ArrayView1<'_, f64>, value_degree: usize) -> Result<f64, String> {
    let num_bspline = knots
        .len()
        .checked_sub(value_degree + 1)
        .filter(|count| *count > value_degree)
        .ok_or_else(|| {
            format!(
                "I-spline domain length needs at least {} knots for value degree {value_degree}, \
                 got {}",
                2 * value_degree + 2,
                knots.len()
            )
        })?;
    let length = knots[num_bspline] - knots[value_degree];
    if !(length > 0.0) || !length.is_finite() {
        return Err(format!(
            "I-spline modeling interval [{}, {}] has non-positive or non-finite length {length}",
            knots[value_degree], knots[num_bspline]
        ));
    }
    Ok(length)
}

/// Replace an order-1 I-spline roughness `∫_D (f')² dy` by the affine-invariant
/// functional of the same order, the variation of the slope about its mean:
///
/// ```text
/// ∫_D (f' − c*)² dy,   c* = (1/|D|) ∫_D f' dy = (f(right) − f(left)) / |D|.
/// ```
///
/// Every anchored I-spline satisfies `I_k(left) = 0` and `I_k(right) = 1`, so
/// `∫_D f' dy = 𝟙ᵀα` exactly and the expansion is closed-form:
///
/// ```text
/// ∫_D (f' − c*)² = ∫_D (f')² − (𝟙ᵀα)² / |D|   ⟹   S₁ ← S₁ − 𝟙𝟙ᵀ / |D|.
/// ```
///
/// The result is PSD (it is the integral of a square), remains an exact
/// function-space functional of the represented transformation rather than a
/// coefficient operator (SPEC 5), and its null space is exactly
/// `{f : f' ≡ const}` — the affine transformations, i.e. the same structural
/// null the order-2 roughness already carries.
fn subtract_constant_slope_component(
    penalty: &mut Array2<f64>,
    knots: ArrayView1<'_, f64>,
    value_degree: usize,
) -> Result<(), String> {
    let domain_length = ispline_domain_length(knots, value_degree)?;
    let scale = 1.0 / domain_length;
    for mut row in penalty.rows_mut() {
        for entry in row.iter_mut() {
            *entry -= scale;
        }
    }
    Ok(())
}

/// Shape-block coefficient direction whose represented I-spline value function
/// is exactly affine in `y`: the AFFINE transformation.
///
/// The anchored I-spline frame is `I_k = Σ_{r ≥ k+1} B_r − I_k(left)`, so
/// `Σ_k α_k I_k = Σ_r c_r B_r + const` with `c_r = Σ_{k < r} α_k`. A B-spline
/// expansion is affine exactly when its coefficients are affine in the Greville
/// abscissae, `c_r = a + b·ξ_r` with `ξ_r = (1/q) Σ_{j=1..q} t_{r+j}`, hence
/// `α_k = b (ξ_{k+1} − ξ_k)`. Returned at `b = 1`, i.e. `Σ_k α_k M_k(y) ≡ 1`.
///
/// This direction is the transformation model's own null: the response-shape
/// block exists to bend `h` away from the affine `(y − μ)/σ` map, and `h' ≡
/// const > 0` is a non-degenerate transformation. Every SHRINKAGE penalty on the
/// shape block must vanish on it — a penalty that does not has its unique
/// minimiser at `α = 0`, where `h' ≡ 0` maps every response to a single score
/// and the likelihood is not defined (gam#2600).
pub(crate) fn affine_shape_direction(
    knots: ArrayView1<'_, f64>,
    response_degree: usize,
    p_shape: usize,
) -> Result<Array1<f64>, String> {
    let value_degree = response_degree + 1;
    let num_bspline = knots
        .len()
        .checked_sub(value_degree + 1)
        .ok_or_else(|| {
            format!(
                "affine shape direction needs more than {} knots for value degree {value_degree}, \
                 got {}",
                value_degree + 1,
                knots.len()
            )
        })?;
    if num_bspline != p_shape + 1 {
        return Err(format!(
            "affine shape direction: knot vector carries {num_bspline} value B-splines, which \
             represents {} I-spline columns, but the shape block has {p_shape}",
            num_bspline.saturating_sub(1)
        ));
    }
    let greville: Vec<f64> = (0..num_bspline)
        .map(|r| {
            (1..=value_degree).map(|j| knots[r + j]).sum::<f64>() / value_degree as f64
        })
        .collect();
    let direction = Array1::from_iter((0..p_shape).map(|k| greville[k + 1] - greville[k]));
    if let Some((k, value)) = direction
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite() || *value <= 0.0)
    {
        return Err(format!(
            "affine shape direction component {k} is {value}; Greville abscissae must be strictly \
             increasing for a valid knot vector"
        ));
    }
    Ok(direction)
}

/// Data-driven cap on the response-shape internal-knot budget keyed on how far
/// the marginal response distribution is from a location-scale Gaussian.
///
/// The CTN response-direction I-spline block exists solely to bend the
/// transformation `h(y)` away from the affine `(y − μ)/σ` map that already makes
/// a homoskedastic-Gaussian response standard normal. When the marginal
/// response is itself close to Gaussian (after centering/scaling), the
/// data carry essentially no information to identify those bend directions:
/// every shape×covariate tensor coordinate beyond "constant scale × location
/// shift" is weakly identified, so a degree-3, 10-internal-knot block (~13
/// response columns, ~100+ tensor coefficients) makes the custom-family
/// optimizer re-factorize a dense exact SCOP Hessian for directions that
/// contribute nothing to the likelihood — the #720 timeout.
///
/// The complexity score is `|skewness| + ½·|excess kurtosis|`, both classic
/// departures from normality that the response-shape basis is there to absorb.
/// For a clean location-scale Gaussian transformation the score is ≈ 0 and the
/// budget collapses to a handful of knots; for genuinely nonlinear / skewed /
/// heteroskedastic transformations (heavy-tailed survival times, censored or
/// log-normal responses, multimodal mixtures) the score is large and the budget
/// relaxes back to the configured count, preserving CTN's expressiveness on
/// real transformations. This adapts the *effective* basis size rather than
/// shrinking the default, so nonlinear accuracy is untouched.
pub(crate) fn transformation_complexity_knot_budget(
    response: ArrayView1<'_, f64>,
    min_internal: usize,
) -> usize {
    let n = response.len();
    if n < 8 {
        // Too few rows to estimate higher moments reliably; do not let a noisy
        // moment estimate gate the basis — fall back to the structural caps.
        return usize::MAX;
    }
    let n_f = n as f64;
    let mean = response.iter().copied().sum::<f64>() / n_f;
    let mut m2 = 0.0;
    let mut m3 = 0.0;
    let mut m4 = 0.0;
    for &y in response.iter() {
        let d = y - mean;
        let d2 = d * d;
        m2 += d2;
        m3 += d2 * d;
        m4 += d2 * d2;
    }
    m2 /= n_f;
    m3 /= n_f;
    m4 /= n_f;
    if m2 <= 0.0 || !m2.is_finite() {
        // Degenerate (constant) response: no shape information at all.
        return min_internal;
    }
    let sd = m2.sqrt();
    let skewness = (m3 / (sd * sd * sd)).abs();
    // Excess kurtosis (Gaussian reference subtracts 3).
    let excess_kurtosis = (m4 / (m2 * m2) - 3.0).abs();
    let complexity = skewness + 0.5 * excess_kurtosis;
    // Each unit of non-normality unlocks a few extra interior knots. A clean
    // Gaussian (complexity ≈ 0) keeps just `min_internal`; moderate departures
    // (complexity ≳ 1) already unlock a rich block, and heavy departures
    // saturate the structural caps below. The slope is deliberately generous so
    // mild nonlinearity is not under-resolved.
    let extra = (complexity * 6.0).round() as usize;
    min_internal.saturating_add(extra)
}

pub fn effective_response_num_internal_knots(
    config: &TransformationNormalConfig,
    n_obs: usize,
    p_cov: usize,
    response: ArrayView1<'_, f64>,
) -> usize {
    // I-spline contract requires K' = K − 2 ≥ 0, i.e. K ≥ 2 internal knots.
    let min_internal = 2usize;
    let sample_cap = (n_obs / 10).max(min_internal);
    let tensor_width_cap = (BASE_TRANSFORMATION_TENSOR_WIDTH + n_obs / 25)
        .min(LARGE_SAMPLE_TRANSFORMATION_TENSOR_WIDTH);
    let max_resp_cols_from_tensor =
        (tensor_width_cap / p_cov.max(1)).max(config.response_degree + 2);
    // One response column is the unconstrained location; the remaining columns
    // are the I-spline shape block controlled by response_num_internal_knots.
    let max_shape_cols_from_tensor = max_resp_cols_from_tensor.saturating_sub(1);
    let tensor_cap = max_shape_cols_from_tensor
        .saturating_sub(config.response_degree + 1)
        .max(min_internal);
    // Data-driven cap: a near-Gaussian transformation does not need (and cannot
    // identify) a heavy shape block. This trims the dense SCOP Hessian /
    // tensor-coefficient cost on easy signals while leaving genuinely nonlinear
    // transformations at the full structural budget.
    let complexity_cap = transformation_complexity_knot_budget(response, min_internal);
    config
        .response_num_internal_knots
        .min(sample_cap)
        .min(tensor_cap)
        .min(complexity_cap)
        .max(min_internal)
}

// ---------------------------------------------------------------------------
// Tensor product construction
// ---------------------------------------------------------------------------

pub(crate) fn assert_rowwise_kronecker_dimensions(
    n: usize,
    p_resp: usize,
    p_cov: usize,
    context: &str,
) -> Result<(), String> {
    if p_resp == 0 || p_cov == 0 {
        return Err(TransformationNormalError::InvalidInput {
            reason: format!(
                "{context} rowwise Kronecker dimensions must be non-empty: n={n}, p_resp={p_resp}, p_cov={p_cov}"
            ),
        }
        .into());
    }
    Ok(())
}
