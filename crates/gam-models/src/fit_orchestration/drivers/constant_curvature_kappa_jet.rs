// #2458 — the constant-curvature κ profile's exact derivative jet.
//
// Extracted from `spatial_optimization.rs` rather than added to it: that file
// is 9231 lines at the point this landed, and a file that crosses 10,000 goes on
// permanent probation requiring ≤7000, which is a 30% cut rather than a trim.
// `include!`d into `drivers/mod.rs` exactly like the sibling files, so the flat
// module namespace and every private-item reference are unchanged.

/// Value, exact first derivative AND exact second derivative of the λ-profiled
/// Gaussian REML negative log evidence along one signed-curvature direction.
///
/// # Why this exists (#2458)
///
/// The outer stationarity certificate accepts a point when no computation could
/// distinguish it from a stationary one — `|Pg| ≤ √(2·h·τ)`, with `h` the
/// curvature along the projected gradient and `τ` the criterion's own forward
/// resolution. Every route that cannot supply `h` was instead held to a raw
/// gradient band with no derivation, which is how one subsystem ended up with
/// stationarity bounds spanning five orders **tiered by which derivative
/// machinery each route happens to implement**. This function supplies `h` for
/// the constant-curvature κ profile, so that route stops being judged by its
/// implementation.
///
/// # The expression
///
/// Write `F(κ, ρ)` for the REML score at FIXED `ρ = log λ`, with the design
/// `X(κ)` and penalty `S(κ)`; the profile is `V(κ) = F(κ, ρ̂(κ))`. The envelope
/// theorem gives `dV/dκ = F_κ` (which is what the adjoint contraction in
/// the adjoint contraction this route used to run) but it does NOT
/// carry to second order — differentiating again picks up the ρ̂ response:
///
/// ```text
///   d²V/dκ² = F_κκ − F_κρ² / F_ρρ
/// ```
///
/// the Schur complement of the λ block. `F_ρρ` is already computed by the
/// forward fit (`reml_hess_rho`); `F_κκ` and `F_κρ` are computed here.
///
/// # The chart
///
/// The score is fully explicit, so no second-order adjoint of the REML solver is
/// needed — only exact matrix calculus in the reduced `p×p` chart, where `p` is
/// the coefficient count (a handful of centers), not `n`:
///
/// ```text
///   A = XᵀX      b = Xᵀy      H = A + λS      β = H⁻¹b
///   dp = yᵀy − bᵀβ            (the PENALIZED deviance: RSS + λβᵀSβ)
///   F  = ½[log|H| − log|S|₊ − rank·ρ] + ½ν[1 + log(2π·dp/ν)]
/// ```
///
/// Every κ-derivative below is the exact derivative of that expression, driven
/// by `A′,A″,b′,b″,S′,S″` — which is why this needs the basis bundle's
/// `.second`. **No finite differences and no autodiff participate** (SPEC 1/2);
/// the FD gates that validate this live in tests.
///
/// The recomputed value is checked against the forward fit's own `reml_score`
/// before any derivative is returned: a chart that does not reproduce the
/// shipped objective cannot be differentiating it.
fn profiled_gaussian_reml_value_kappa_jet(
    design: &Array2<f64>,
    design_kappa: &Array2<f64>,
    design_kappa2: &Array2<f64>,
    penalty: &Array2<f64>,
    penalty_kappa: &Array2<f64>,
    penalty_kappa2: &Array2<f64>,
    response: ArrayView1<'_, f64>,
) -> Result<(f64, f64, f64), EstimationError> {
    use faer::Side;
    use gam_linalg::faer_ndarray::{FaerCholesky, strict_symmetric_eigh};

    let (n, p) = design.dim();
    if design_kappa.dim() != (n, p)
        || design_kappa2.dim() != (n, p)
        || penalty.dim() != (p, p)
        || penalty_kappa.dim() != (p, p)
        || penalty_kappa2.dim() != (p, p)
        || response.len() != n
    {
        crate::bail_invalid_estim!("constant-curvature profile κ-jet shape mismatch");
    }

    let response_2d = response.insert_axis(ndarray::Axis(1));
    let fit = gam_solve::gaussian_reml::gaussian_reml_multi_closed_form(
        design.view(),
        response_2d.view(),
        penalty.view(),
        None,
        None,
    )?;
    let lambda = fit.lambda;
    let nullity = fit.cache.nullity;
    let rank = p.saturating_sub(nullity);
    if rank == 0 {
        crate::bail_invalid_estim!(
            "constant-curvature profile κ-jet needs a penalty of positive rank; got nullity {nullity} of {p}"
        );
    }
    let nu = n as f64 - nullity as f64;
    if nu <= 0.0 {
        crate::bail_invalid_estim!(
            "constant-curvature profile κ-jet needs positive residual degrees of freedom"
        );
    }

    // Reduced κ-jets. `A = XᵀX` and `b = Xᵀy` are the only places `n` appears;
    // everything after this is p×p.
    let sym = |m: Array2<f64>| -> Array2<f64> { (&m + &m.t()) * 0.5 };
    let a0 = sym(fast_atb(&design.view(), &design.view()));
    let a1 = sym(fast_atb(&design_kappa.view(), &design.view()) * 2.0);
    let a2 = sym(
        fast_atb(&design_kappa2.view(), &design.view()) * 2.0
            + fast_atb(&design_kappa.view(), &design_kappa.view()) * 2.0,
    );
    let b0 = fast_atv(&design.view(), &response);
    let b1 = fast_atv(&design_kappa.view(), &response);
    let b2 = fast_atv(&design_kappa2.view(), &response);
    let yty = response.dot(&response);

    let s0 = sym(penalty.clone());
    let s1 = sym(penalty_kappa.clone());
    let s2 = sym(penalty_kappa2.clone());

    // H = A + λS and its κ-jets at FIXED ρ.
    let h0 = &a0 + &(&s0 * lambda);
    let h1 = &a1 + &(&s1 * lambda);
    let h2 = &a2 + &(&s2 * lambda);
    let chol = h0
        .cholesky(Side::Lower)
        .map_err(|error| EstimationError::InvalidInput(error.to_string()))?;
    let beta0 = chol.solvevec(&b0);
    let beta1 = chol.solvevec(&(&b1 - &h1.dot(&beta0)));
    let beta2 = chol.solvevec(&(&b2 - &(&h1.dot(&beta1) * 2.0) - &h2.dot(&beta0)));

    // log|H| and its κ-jets. `g = H⁻¹H′` is formed once and reused.
    let logdet_h = {
        let diag = chol.diag();
        2.0 * diag.iter().map(|value| value.ln()).sum::<f64>()
    };
    let g1 = chol.solve_mat(&h1);
    let g2 = chol.solve_mat(&h2);
    let trace = |m: &Array2<f64>| -> f64 { (0..m.nrows()).map(|i| m[(i, i)]).sum::<f64>() };
    let trace_product = |left: &Array2<f64>, right: &Array2<f64>| -> f64 {
        left.iter()
            .zip(right.t().iter())
            .map(|(&a, &b)| a * b)
            .sum::<f64>()
    };
    let logdet_h_k = trace(&g1);
    let logdet_h_kk = trace(&g2) - trace_product(&g1, &g1);

    // log|S|₊ and its κ-jets on the penalty's POSITIVE subspace. The subspace is
    // κ-fixed (the smooth's null directions are structural — the unpenalized
    // parametric coordinates — and carry no κ dependence), which is what makes
    // the pseudo-determinant differentiable at all; it is verified rather than
    // assumed: S′ and S″ must annihilate the null frame.
    let (s_eigenvalues, s_eigenvectors) = strict_symmetric_eigh(&s0, Side::Lower)
        .map_err(|error| EstimationError::InvalidInput(error.to_string()))?;
    let mut order: Vec<usize> = (0..p).collect();
    order.sort_by(|&i, &j| {
        s_eigenvalues[j]
            .partial_cmp(&s_eigenvalues[i])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut positive_frame = Array2::<f64>::zeros((p, rank));
    for (column, &index) in order.iter().take(rank).enumerate() {
        positive_frame
            .column_mut(column)
            .assign(&s_eigenvectors.column(index));
    }
    if nullity > 0 {
        let mut null_frame = Array2::<f64>::zeros((p, nullity));
        for (column, &index) in order.iter().skip(rank).enumerate() {
            null_frame
                .column_mut(column)
                .assign(&s_eigenvectors.column(index));
        }
        let s0_scale = s0.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        for (label, block) in [("∂S/∂κ", &s1), ("∂²S/∂κ²", &s2)] {
            let leak = block.dot(&null_frame);
            let leak_norm = leak.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
            if leak_norm > 1.0e-8 * (1.0 + s0_scale) {
                crate::bail_invalid_estim!(
                    "constant-curvature profile κ-jet requires a κ-fixed penalty null space, but {label} moves it by {leak_norm:.3e}"
                );
            }
        }
    }
    let restrict = |m: &Array2<f64>| -> Array2<f64> {
        sym(positive_frame.t().dot(&m.dot(&positive_frame)))
    };
    let r0 = restrict(&s0);
    let r1 = restrict(&s1);
    let r2 = restrict(&s2);
    let r_chol = r0
        .cholesky(Side::Lower)
        .map_err(|error| EstimationError::InvalidInput(error.to_string()))?;
    let logdet_s_positive = {
        let diag = r_chol.diag();
        2.0 * diag.iter().map(|value| value.ln()).sum::<f64>()
    };
    let m1 = r_chol.solve_mat(&r1);
    let m2 = r_chol.solve_mat(&r2);
    let logdet_s_k = trace(&m1);
    let logdet_s_kk = trace(&m2) - trace_product(&m1, &m1);

    // Penalized deviance `dp = yᵀy − bᵀβ` and its κ-jets.
    let dp = yty - b0.dot(&beta0);
    if !(dp > 0.0) {
        crate::bail_invalid_estim!(
            "constant-curvature profile κ-jet found a non-positive profiled deviance {dp:.6e}"
        );
    }
    let dp_k = -b1.dot(&beta0) - b0.dot(&beta1);
    let dp_kk = -b2.dot(&beta0) - 2.0 * b1.dot(&beta1) - b0.dot(&beta2);

    // The value, recomputed in this chart, must be the shipped objective.
    let rho = fit.rho;
    let value = 0.5 * (logdet_h - (logdet_s_positive + rank as f64 * rho))
        + 0.5 * nu * (1.0 + (2.0 * std::f64::consts::PI * dp / nu).ln());
    if !value.is_finite()
        || (value - fit.reml_score).abs() > 1.0e-7 * (1.0 + fit.reml_score.abs())
    {
        crate::bail_invalid_estim!(
            "constant-curvature profile κ-jet chart does not reproduce the REML score it differentiates: chart {value:.9e} vs forward {:.9e}",
            fit.reml_score
        );
    }

    let f_k = 0.5 * (logdet_h_k - logdet_s_k) + 0.5 * nu * dp_k / dp;
    let f_kk =
        0.5 * (logdet_h_kk - logdet_s_kk) + 0.5 * nu * (dp_kk / dp - (dp_k / dp) * (dp_k / dp));

    // Mixed ∂²F/∂κ∂ρ. Only H and β carry ρ: ∂H/∂ρ = λS, ∂H′/∂ρ = λS′.
    let lambda_s0 = &s0 * lambda;
    let lambda_s1 = &s1 * lambda;
    let beta0_rho = chol.solvevec(&lambda_s0.dot(&beta0)) * -1.0;
    let beta1_rho = {
        let inner = lambda_s1.dot(&beta0) * -1.0 + h1.dot(&chol.solvevec(&lambda_s0.dot(&beta0)));
        chol.solvevec(&(&inner - &lambda_s0.dot(&beta1)))
    };
    let logdet_h_k_rho = trace(&chol.solve_mat(&lambda_s1))
        - trace_product(&chol.solve_mat(&lambda_s0), &g1);
    let dp_rho = -b0.dot(&beta0_rho);
    let dp_k_rho = -b1.dot(&beta0_rho) - b0.dot(&beta1_rho);
    let f_krho =
        0.5 * logdet_h_k_rho + 0.5 * nu * (dp_k_rho / dp - dp_k * dp_rho / (dp * dp));

    // The λ̂ response is the derivative of an INTERIOR stationary root only. At a
    // railed ρ̂ the selection is locally constant, so dλ̂/dκ = 0 and the Schur
    // term is absent — the same premise the backward VJP already gates on. When
    // the ρ curvature is unusable the profile's second derivative is not
    // identified and this refuses rather than substituting a number.
    let hess_rho = fit.reml_hess_rho;
    let rho_at_bound = (rho - gam_solve::gaussian_reml::RHO_LOWER).abs() <= 1.0e-9
        || (rho - gam_solve::gaussian_reml::RHO_UPPER).abs() <= 1.0e-9;
    let second = if rho_at_bound {
        f_kk
    } else if hess_rho.is_finite() && hess_rho.abs() > 1.0e-14 {
        f_kk - f_krho * f_krho / hess_rho
    } else {
        crate::bail_invalid_estim!(
            "constant-curvature profile κ-jet cannot identify d²V/dκ²: the ρ curvature is {hess_rho:.3e} at an interior ρ̂"
        );
    };
    if !(f_k.is_finite() && second.is_finite()) {
        crate::bail_invalid_estim!(
            "constant-curvature profile κ-jet produced a non-finite derivative"
        );
    }
    Ok((fit.reml_score, f_k, second))
}
