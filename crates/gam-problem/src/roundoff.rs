//! One definition of "this residual is indistinguishable from zero".
//!
//! Two places in the engine have to decide whether a fitted mean reproduces its
//! response *exactly*: the formula path's deterministic-Gaussian dispatch, which
//! PREDICTS the state from the shape of the request, and the solver, which
//! MEASURES it after the dispersion has been estimated. They must decide it the
//! same way — a fit that is exact on one route and merely near-exact on the
//! other reports a different scale, a different covariance, and a different
//! criterion for the same data, which is how #2595 stayed invisible for a week.
//!
//! The shared quantity is Wilkinson's accumulated-roundoff growth factor. A
//! floating-point sum of `k` operations carries a relative error bounded by
//!
//! ```text
//!     γ_k = k·ε / (1 − k·ε),      ε = f64::EPSILON
//! ```
//!
//! so a linear predictor `η_i = Σ_j x_ij β_j (+ offset)` formed from `p` terms
//! cannot be trusted below `γ_{p+1} · (Σ_j |x_ij β_j| + |offset_i|)` — and a
//! residual `y_i − η_i` smaller than that is not evidence of misfit, it is the
//! arithmetic. This is a derived bound, not a tuned threshold: it moves with the
//! model width and the data scale and has no free parameter.

/// Wilkinson's growth factor `γ_k = k·ε/(1 − k·ε)` for a sum of `operations`
/// floating-point operations.
///
/// `None` when `k·ε ≥ 1` — a model so wide that the accumulated bound exceeds
/// the operands themselves, where no residual can be certified as roundoff and
/// the caller must not treat any fit as exact.
pub fn roundoff_growth_factor(operations: usize) -> Option<f64> {
    let relative = (operations as f64) * f64::EPSILON;
    (relative < 1.0).then(|| relative / (1.0 - relative))
}

/// Is a weighted residual sum of squares indistinguishable from zero?
///
/// `operand_scales[i]` is the sum of the magnitudes of the terms that formed row
/// `i`'s residual — the caller supplies it, because only the caller knows which
/// operands it summed. `terms` is the number of those operands, which fixes the
/// growth factor.
///
/// Returns `true` when `Σ_i w_i r_i² ≤ Σ_i w_i (γ·scale_i)²`: every row's
/// residual is, in aggregate, within the arithmetic's own resolution. Callers
/// that can only supply a LOWER bound on the operand scale (`|y_i| + |η_i|`
/// rather than `|y_i| + Σ_j |x_ij β_j|`) get a conservative answer — the
/// predicate then fires less often, never more.
pub fn weighted_residual_is_at_roundoff_floor(
    weighted_rss: f64,
    weights: impl IntoIterator<Item = f64>,
    operand_scales: impl IntoIterator<Item = f64>,
    terms: usize,
) -> bool {
    if !weighted_rss.is_finite() || weighted_rss < 0.0 {
        return false;
    }
    let Some(gamma) = roundoff_growth_factor(terms) else {
        return false;
    };
    let mut budget = 0.0_f64;
    for (weight, scale) in weights.into_iter().zip(operand_scales) {
        if !(weight.is_finite() && weight >= 0.0) || !scale.is_finite() {
            return false;
        }
        let bound = gamma * scale.abs();
        budget += weight * bound * bound;
    }
    budget.is_finite() && weighted_rss <= budget
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn growth_factor_is_monotone_and_matches_the_closed_form() {
        let one = roundoff_growth_factor(1).expect("k=1 is representable");
        let ten = roundoff_growth_factor(10).expect("k=10 is representable");
        assert!(one < ten);
        assert!((one - f64::EPSILON / (1.0 - f64::EPSILON)).abs() <= f64::EPSILON * 1e-3);
    }

    #[test]
    fn growth_factor_refuses_a_width_that_saturates_the_bound() {
        // k·ε ≥ 1 means the accumulated bound is no smaller than the operands.
        assert_eq!(roundoff_growth_factor(usize::MAX), None);
    }

    #[test]
    fn an_exactly_zero_residual_is_at_the_floor() {
        assert!(weighted_residual_is_at_roundoff_floor(
            0.0,
            vec![1.0; 4],
            vec![1.0; 4],
            3
        ));
    }

    #[test]
    fn a_one_ulp_residual_on_unit_data_is_at_the_floor() {
        // Four rows, each off by one ulp of a unit-scale operand pair.
        let residual = f64::EPSILON;
        let rss = 4.0 * residual * residual;
        assert!(weighted_residual_is_at_roundoff_floor(
            rss,
            vec![1.0; 4],
            vec![2.0; 4],
            3
        ));
    }

    #[test]
    fn ordinary_noise_is_not_at_the_floor() {
        let rss = 4.0 * 1.0e-3 * 1.0e-3;
        assert!(!weighted_residual_is_at_roundoff_floor(
            rss,
            vec![1.0; 4],
            vec![2.0; 4],
            3
        ));
    }

    #[test]
    fn the_bound_scales_with_the_data() {
        // The same RELATIVE misfit is at the floor on large data and not on
        // small: the predicate is scale-covariant, not an absolute epsilon.
        let residual = 1.0e6 * f64::EPSILON;
        let rss = 4.0 * residual * residual;
        assert!(weighted_residual_is_at_roundoff_floor(
            rss,
            vec![1.0; 4],
            vec![2.0e6; 4],
            3
        ));
        assert!(!weighted_residual_is_at_roundoff_floor(
            rss,
            vec![1.0; 4],
            vec![2.0; 4],
            3
        ));
    }

    #[test]
    fn zero_weight_rows_contribute_no_budget() {
        // A zero-weight row is equivalent to an absent row on both sides of the
        // comparison, so it can neither create nor consume slack.
        assert!(!weighted_residual_is_at_roundoff_floor(
            1.0,
            vec![0.0; 4],
            vec![1.0e300; 4],
            3
        ));
    }
}
