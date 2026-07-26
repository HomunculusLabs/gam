//! Derived floating-point roundoff bands.
//!
//! Numerical code is full of the question "is this quantity indistinguishable
//! from zero?" — a negative eigenvalue that should be a zero one, a residual
//! that should be a converged one, a directional derivative that should be a
//! flat one. The answer is not a taste parameter: it is the backward-error band
//! of the arithmetic that produced the quantity, and that band is a function of
//! how many operations were accumulated and how large the accumulated terms
//! were. Both are known at the call site.
//!
//! The bound is Wilkinson's, in the form given by Higham (*Accuracy and
//! Stability of Numerical Algorithms*, 2nd ed., SIAM 2002, Lemma 3.1): the
//! floating-point evaluation of a sum or inner product of `n` terms satisfies
//!
//! ```text
//! |fl(s) − s|  ≤  γ_n · Σ|terms|,      γ_n = n·u / (1 − n·u),
//! ```
//!
//! with `u` the unit roundoff. Writing a bare multiple of `EPSILON` instead
//! substitutes a constant for `γ_n`, which is wrong in both directions as `n`
//! moves — too tight for long accumulations, so exact-arithmetic-zero
//! quantities get rejected as materially nonzero, and needlessly loose for
//! short ones.

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
pub fn accumulation_growth(operations: usize) -> f64 {
    let scaled = operations as f64 * UNIT_ROUNDOFF;
    if !(scaled < 1.0) {
        return f64::INFINITY;
    }
    scaled / (1.0 - scaled)
}

/// Backward-error band of a `terms`-term accumulation whose summands have
/// absolute sum `absolute_sum`.
///
/// This is the magnitude below which the computed result is indistinguishable
/// from zero: a sum whose exact value is zero can be computed as anything
/// within `±accumulation_band(n, Σ|terms|)`, and nothing outside it.
pub fn accumulation_band(terms: usize, absolute_sum: f64) -> f64 {
    accumulation_growth(terms) * absolute_sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unit_roundoff_is_half_an_epsilon_gap() {
        assert_eq!(UNIT_ROUNDOFF * 2.0, f64::EPSILON);
        // 1.0 + u rounds to 1.0 (ties-to-even), 1.0 + 2u does not.
        assert_eq!(1.0_f64 + UNIT_ROUNDOFF, 1.0);
        assert!(1.0_f64 + 2.0 * UNIT_ROUNDOFF > 1.0);
    }

    #[test]
    fn growth_is_zero_for_exact_arithmetic_and_grows_linearly() {
        assert_eq!(accumulation_growth(0), 0.0);
        let one = accumulation_growth(1);
        assert!((one - UNIT_ROUNDOFF).abs() <= UNIT_ROUNDOFF * UNIT_ROUNDOFF * 4.0);
        // gamma_n / (n u) -> 1 from above, and is monotone in n.
        let mut previous = 0.0_f64;
        for n in [1_usize, 8, 64, 1024, 1 << 20] {
            let gamma = accumulation_growth(n);
            assert!(gamma > previous, "gamma must be monotone in n");
            assert!(gamma >= n as f64 * UNIT_ROUNDOFF);
            assert!(gamma <= n as f64 * UNIT_ROUNDOFF * 1.000_001);
            previous = gamma;
        }
    }

    #[test]
    fn growth_saturates_rather_than_going_negative() {
        // n u >= 1 makes the textbook quotient negative; the band must not be.
        let vacuous = (1.0 / UNIT_ROUNDOFF).ceil() as usize;
        assert_eq!(accumulation_growth(vacuous), f64::INFINITY);
        assert_eq!(accumulation_growth(usize::MAX), f64::INFINITY);
    }

    #[test]
    fn band_bounds_a_cancelling_sum_that_is_exactly_zero() {
        // Every magnitude appears once positive and once negative, so the sum
        // is exactly zero in real arithmetic. The two signs are kept apart so
        // the cancellation is not absorbed by adjacent-pair exactness, which
        // would make the computed sum trivially 0.0.
        let magnitudes: Vec<f64> = (1..=256)
            .map(|k| (k as f64) * 0.1_f64.powi(k % 7))
            .collect();
        let terms: Vec<f64> = magnitudes
            .iter()
            .copied()
            .chain(magnitudes.iter().map(|value| -value))
            .collect();
        let absolute_sum: f64 = terms.iter().map(|value| value.abs()).sum();
        let computed: f64 = terms.iter().sum();
        let band = accumulation_band(terms.len(), absolute_sum);
        assert!(band > 0.0);
        assert!(
            computed.abs() <= band,
            "computed {computed:e} escaped its band {band:e}"
        );
    }
}
