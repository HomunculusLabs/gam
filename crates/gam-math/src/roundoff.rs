//! The unit roundoff and Wilkinson's accumulation factor, owned in the lowest crate so every rounding bound in the
//! workspace reads one definition. `gam_linalg::roundoff` re-exports both at its own paths, beside the bands built on
//! them.
//!
//! The factor is Wilkinson's, in the form given by Higham (*Accuracy and Stability of Numerical Algorithms*, 2nd ed.,
//! SIAM 2002, Lemma 3.1): a product of `k` factors `(1 + δᵢ)^{±1}` with `|δᵢ| ≤ u` is `1 + θ` with
//! `|θ| ≤ γ_k = k·u/(1 − k·u)`. A first-order count `k·u` sits below `γ_k` and agrees with it to `O(u²)`.

/// Unit roundoff `u = EPSILON/2`.
///
/// `EPSILON` is the gap between `1.0` and the next representable `f64`; the
/// error of a single correctly-rounded operation is at most half that gap
/// relative to the result, which is the quantity every backward-error bound is
/// stated in. The factor of two between the two is the single most common
/// source of "the same tolerance, twice, 2× apart".
pub const UNIT_ROUNDOFF: f64 = f64::EPSILON / 2.0;

/// Wilkinson's growth factor `γ_n = n·u / (1 − n·u)` for an `n`-operation
/// accumulation.
///
/// Returns infinity once `n·u ≥ 1`, where the bound carries no information —
/// an accumulation that long has no useful error bound, and reporting an
/// infinite band is the honest answer rather than a negative or wrapped one.
pub const fn accumulation_growth(operations: usize) -> f64 {
    let scaled = operations as f64 * UNIT_ROUNDOFF;
    if !(scaled < 1.0) {
        return f64::INFINITY;
    }
    scaled / (1.0 - scaled)
}

/// An upper bound on a positive expression whose computed value is `value` after `operations` rounded operations:
/// `exact ≤ value·(1 − u)^−k ≤ value·(1 + γ_k)`. The three extra operations cover forming the factor and the product.
pub fn inflated(value: f64, operations: usize) -> f64 {
    value * (1.0 + accumulation_growth(operations + 3))
}
