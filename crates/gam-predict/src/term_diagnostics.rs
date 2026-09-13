//! Per-term diagnostics of a standard GAM's additive predictor, evaluated on a
//! design built at caller-supplied rows.

use ndarray::{ArrayView1, ArrayView2};
use std::ops::Range;

/// Partial dependence of one term block: `f_t(x) = X_t(x) β_t` and the matching
/// delta-method standard error `sqrt(diag(X_t V_t X_tᵀ))`, where `V_t` is the
/// term block of the coefficient covariance.
pub fn term_partial_dependence(
    design: ArrayView2<'_, f64>,
    beta: ArrayView1<'_, f64>,
    covariance: ArrayView2<'_, f64>,
    block: Range<usize>,
) -> Result<(Vec<f64>, Vec<f64>), String> {
    let p = beta.len();
    if block.start > block.end
        || block.end > p
        || block.end > design.ncols()
        || block.end > covariance.nrows()
        || block.end > covariance.ncols()
    {
        return Err(format!(
            "term partial dependence block {block:?} must lie inside the {p} coefficients, the \
             {}-column design and the {:?} covariance",
            design.ncols(),
            covariance.dim()
        ));
    }
    let n = design.nrows();
    let mut predicted = vec![0.0_f64; n];
    let mut se = vec![0.0_f64; n];
    for i in 0..n {
        let xi = design.row(i);
        let mut f = 0.0_f64;
        for c in block.clone() {
            f += xi[c] * beta[c];
        }
        predicted[i] = f;
        let mut var = 0.0_f64;
        for a in block.clone() {
            let xa = xi[a];
            for b in block.clone() {
                var += xa * covariance[[a, b]] * xi[b];
            }
        }
        se[i] = var.max(0.0).sqrt();
    }
    Ok((predicted, se))
}

/// Per-term variance share `cov(X_t β_t, X β) / var(X β)` for each named block.
///
/// This is a genuine variance decomposition: `var(η) = Σ_t cov(f_t, η)`, so the
/// shares over every term sum to exactly 1 (an intercept contributes a constant
/// with zero covariance). Each term's cross-covariance with every other term is
/// split symmetrically — half to each side — which is the Shapley allocation
/// for a sum of terms. The naive `var(f_t) / var(η)` drops all cross terms: for
/// `f_1 = x`, `f_2 = -0.9x` it reports shares 100 and 81 against a total of
/// 0.01·var(x). A share can exceed 1 or be negative only when terms genuinely
/// anticorrelate, which is honest rather than a bug.
pub fn term_variance_shares(
    design: ArrayView2<'_, f64>,
    beta: ArrayView1<'_, f64>,
    blocks: &[(String, Range<usize>)],
) -> Result<Vec<(String, f64)>, String> {
    let p = beta.len();
    if design.ncols() < p {
        return Err(format!(
            "term variance shares need a design with at least the {p} coefficient columns, got {}",
            design.ncols()
        ));
    }
    if let Some((name, block)) = blocks
        .iter()
        .find(|(_, block)| block.start > block.end || block.end > p)
    {
        return Err(format!(
            "term variance shares: block {block:?} of term {name:?} lies outside the {p} coefficients"
        ));
    }
    let n = design.nrows();
    let mut eta = vec![0.0_f64; n];
    for i in 0..n {
        let xi = design.row(i);
        let mut s = 0.0_f64;
        for c in 0..p {
            s += xi[c] * beta[c];
        }
        eta[i] = s;
    }
    let total_var = population_variance(&eta);
    let mut out: Vec<(String, f64)> = Vec::with_capacity(blocks.len());
    for (name, block) in blocks {
        let mut contrib = vec![0.0_f64; n];
        for i in 0..n {
            let xi = design.row(i);
            let mut s = 0.0_f64;
            for c in block.clone() {
                s += xi[c] * beta[c];
            }
            contrib[i] = s;
        }
        let share = if total_var > 0.0 {
            population_covariance(&contrib, &eta) / total_var
        } else {
            0.0
        };
        out.push((name.clone(), share));
    }
    Ok(out)
}

/// Population variance (divide by `n`, matching numpy `np.var`'s default).
fn population_variance(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    values
        .iter()
        .map(|value| (value - mean) * (value - mean))
        .sum::<f64>()
        / values.len() as f64
}

/// Population covariance of two equal-length slices (divide by `n`, matching
/// `population_variance`).
fn population_covariance(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len();
    if n == 0 {
        return 0.0;
    }
    let mean_a = a.iter().sum::<f64>() / n as f64;
    let mean_b = b.iter().sum::<f64>() / n as f64;
    a.iter()
        .zip(b.iter())
        .map(|(&va, &vb)| (va - mean_a) * (vb - mean_b))
        .sum::<f64>()
        / n as f64
}
