//! Categorical distributions named by logits: log-sum-exp, log-softmax, and the Kullback–Leibler divergence between two
//! categorical distributions from their logits (#2951).
//!
//! A logit vector `z ∈ [−∞, ∞)^K` with at least one finite entry names the distribution `p = softmax z`. A `−∞` logit is
//! a category of probability zero. `NaN` and `+∞` name no distribution and are refused.
//!
//! # Log-sum-exp and log-softmax
//!
//! `lse(z) = log Σ_i e^{z_i}` is [`signed_log_sum_exp`] with every sign positive: one common max-shift and a compensated
//! sum, so gam-math keeps one owner of the reduction. `log softmax(z)_i = z_i − lse(z)`.
//!
//! # KL from logits
//!
//! Let `S = {i : z_i > −∞}` be the support of `p`, `q` the distribution `p′ = softmax z′` renormalized on `S`, and on `S`
//! the gaps `δ_i = z′_i − z_i`, `μ = E_p δ` and `d_i = δ_i − μ`. Then
//!
//! ```text
//! KL(p ‖ p′) = KL(p ‖ q) + log(Σ_all e^{z′_i} / Σ_S e^{z′_i}),
//! KL(p ‖ q)  = log E_p e^δ − E_p δ = log E_p e^{d} = log1p(Σ_{i∈S} p_i φ(d_i)),     φ(d) = e^d − 1 − d ≥ 0,
//! ```
//!
//! because `log Σ_S e^{z′_i} − lse(z) = log Σ_S p_i e^{δ_i}` and `Σ_S p_i d_i = 0`. The second term is the mass `p′`
//! places outside `S`, formed as `log1p(e^{lse(z′ outside S) − lse(z′ on S)})`. The divergence is `+∞` when `p′`
//! vanishes at a category of `S`.
//!
//! Every summand of `Σ_S p_i φ(d_i)` is non-negative, so the sum never cancels. The subtraction form `lse(z′) − lse(z) −
//! μ` instead subtracts two numbers of size `|lse z|` to leave one of size `KL`, and its rounding, of order `u·|lse z|`,
//! swamps a divergence below that scale (the tests pin such a case). The sum is formed in log space,
//! `log Σ_S p_i φ(d_i) = lse_i(log p_i + log φ(d_i))`, so neither an underflowing probability nor an overflowing `e^{d_i}`
//! loses a term, and `log1p(e^L)` is evaluated without overflow.
//!
//! `log φ(d)` has three forms:
//! - `|d| ≤ 1`: the series `Σ_{j≥2} d^j/j!`, summed until a term `t_j` is at most `u` times the partial sum. Consecutive
//!   terms shrink by `|d|/(j + 1) ≤ 1/3`, and every later ratio is at most `1/4`, so the remainder is at most
//!   `(1/3)·(4/3)·|t_j| < |t_j|`. The series avoids `expm1(d) − d`, which cancels to relative error `2u/|d|`.
//! - `d > 1`: `d + log1p(−(1 + d)·e^{−d})`, whose argument stays above `−2/e > −1`.
//! - `d < −1`: `log(expm1(d) − d)`, a difference that keeps at least the fraction `e^{−1}` of its larger operand `−d`.
//!
//! # Accuracy
//!
//! The divergence carries the rounding of its terms' inputs, not that of a cancelling difference:
//! - each `log p_i` carries a few ulps of `|z_i| + |lse z|`, and each gap one ulp of `|δ_i|`, which moves its term by at
//!   most `u·|δ_i|·|expm1 d_i|` against a term of size `φ(d_i)`;
//! - `μ`'s rounding moves the sum only at relative first order `|Δμ|`, because
//!   `∂_μ Σ_S p_i φ(δ_i − μ) = −Σ_S p_i expm1(d_i) = −Σ_S p_i φ(d_i)`;
//! - the log-space round trip adds relative error `O(u·|log KL|)`.

use std::fmt;

use crate::probability::signed_log_sum_exp;

/// A logit vector that names no categorical distribution, or two logit vectors that cannot be compared.
#[derive(Clone, Debug, PartialEq)]
pub enum CategoricalError {
    /// A logit vector with no entries.
    Empty { name: &'static str },
    /// A logit that is `NaN` or `+∞`.
    InvalidLogit {
        name: &'static str,
        index: usize,
        value: f64,
    },
    /// Every logit is `−∞`, so no category has positive probability.
    NoSupport { name: &'static str },
    /// Two logit vectors of different lengths.
    LengthMismatch { reference: usize, perturbed: usize },
    /// The gap `z′_i − z_i` at a supported category, or its offset from the mean gap, overflowed.
    GapOverflow { index: usize },
}

impl fmt::Display for CategoricalError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty { name } => write!(formatter, "categorical {name} are empty"),
            Self::InvalidLogit { name, index, value } => write!(
                formatter,
                "categorical {name}[{index}] = {value}; a logit must be finite or -inf"
            ),
            Self::NoSupport { name } => write!(
                formatter,
                "every categorical {name} entry is -inf, so no category has positive probability"
            ),
            Self::LengthMismatch {
                reference,
                perturbed,
            } => write!(
                formatter,
                "categorical logits have {reference} categories but the perturbed logits have {perturbed}"
            ),
            Self::GapOverflow { index } => write!(
                formatter,
                "the logit gap at category {index} overflowed"
            ),
        }
    }
}

impl std::error::Error for CategoricalError {}

fn validate(name: &'static str, logits: &[f64]) -> Result<(), CategoricalError> {
    if logits.is_empty() {
        return Err(CategoricalError::Empty { name });
    }
    for (index, &value) in logits.iter().enumerate() {
        if value.is_nan() || value == f64::INFINITY {
            return Err(CategoricalError::InvalidLogit { name, index, value });
        }
    }
    if logits.iter().all(|value| *value == f64::NEG_INFINITY) {
        return Err(CategoricalError::NoSupport { name });
    }
    Ok(())
}

/// `log Σ_i e^{x_i}` over terms that may be `−∞`; an empty or all-`−∞` sum is `−∞`.
fn positive_log_sum_exp(log_terms: &[f64]) -> f64 {
    signed_log_sum_exp(log_terms, &vec![1.0; log_terms.len()]).0
}

/// `log(1 + e^x)` without overflow.
fn log1p_exp(value: f64) -> f64 {
    if value > 0.0 {
        value + (-value).exp().ln_1p()
    } else {
        value.exp().ln_1p()
    }
}

/// `log φ(d)` with `φ(d) = e^d − 1 − d`, in the three forms of the module documentation. `log φ(0) = −∞`.
fn log_excess_exponential(d: f64) -> f64 {
    if d > 1.0 {
        d + (-(1.0 + d) * (-d).exp()).ln_1p()
    } else if d < -1.0 {
        (d.exp_m1() - d).ln()
    } else {
        let unit_roundoff = f64::EPSILON / 2.0;
        let mut term = d * d / 2.0;
        let mut sum = term;
        let mut order = 2.0_f64;
        while term.abs() > unit_roundoff * sum {
            order += 1.0;
            term *= d / order;
            sum += term;
        }
        sum.ln()
    }
}

/// `lse(z) = log Σ_i e^{z_i}`, through [`signed_log_sum_exp`]'s max-shifted compensated sum.
pub fn log_sum_exp(logits: &[f64]) -> Result<f64, CategoricalError> {
    validate("logits", logits)?;
    Ok(positive_log_sum_exp(logits))
}

/// `log softmax(z)_i = z_i − lse(z)`. A `−∞` logit gives `−∞`.
pub fn log_softmax(logits: &[f64]) -> Result<Vec<f64>, CategoricalError> {
    let normalizer = log_sum_exp(logits)?;
    Ok(logits.iter().map(|logit| logit - normalizer).collect())
}

/// `KL(softmax z ‖ softmax z′)` from the logits `z = logits` and `z′ = perturbed`, by the non-cancelling form of the
/// module documentation. `+∞` when `p′` vanishes where `p` does not.
pub fn categorical_kl_from_logits(logits: &[f64], perturbed: &[f64]) -> Result<f64, CategoricalError> {
    if logits.len() != perturbed.len() {
        return Err(CategoricalError::LengthMismatch {
            reference: logits.len(),
            perturbed: perturbed.len(),
        });
    }
    let log_probabilities = log_softmax(logits)?;
    validate("perturbed logits", perturbed)?;

    let mut supported = Vec::with_capacity(logits.len());
    let mut perturbed_on_support = Vec::with_capacity(logits.len());
    let mut perturbed_off_support = Vec::new();
    for (index, ((&log_probability, &reference), &moved)) in log_probabilities
        .iter()
        .zip(logits)
        .zip(perturbed)
        .enumerate()
    {
        if reference == f64::NEG_INFINITY {
            perturbed_off_support.push(moved);
            continue;
        }
        if moved == f64::NEG_INFINITY {
            return Ok(f64::INFINITY);
        }
        let gap = moved - reference;
        if !gap.is_finite() {
            return Err(CategoricalError::GapOverflow { index });
        }
        supported.push((index, log_probability, gap));
        perturbed_on_support.push(moved);
    }

    let mean_gap: f64 = supported
        .iter()
        .map(|entry| entry.1.exp() * entry.2)
        .sum();
    let mut log_terms = Vec::with_capacity(supported.len());
    for &(index, log_probability, gap) in &supported {
        let centered = gap - mean_gap;
        if !centered.is_finite() {
            return Err(CategoricalError::GapOverflow { index });
        }
        log_terms.push(log_probability + log_excess_exponential(centered));
    }
    let divergence_on_support = log1p_exp(positive_log_sum_exp(&log_terms));
    let mass_off_support = log1p_exp(
        positive_log_sum_exp(&perturbed_off_support) - positive_log_sum_exp(&perturbed_on_support),
    );
    Ok(divergence_on_support + mass_off_support)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::LN_2;

    const UNIT_ROUNDOFF: f64 = f64::EPSILON / 2.0;

    /// Relative rounding band of the divergence at value `divergence`: the series retains at most 20 terms for
    /// `|d| ≤ 1` (`19! > 3/u`), each a product and an addition on operands summing to at most `φ(|d|) ≤ 1.95·φ(d)`,
    /// which is below `80u`; the probabilities, the two log-sum-exps and `log1p` add at most `48u`; and the log-space
    /// round trip adds `4u·|log KL|`.
    fn divergence_band(divergence: f64) -> f64 {
        (128.0 + 4.0 * divergence.ln().abs()) * UNIT_ROUNDOFF * divergence
    }

    #[test]
    fn symmetric_pair_divergence_is_log_cosh_and_the_subtraction_form_loses_it() {
        // p = (½, ½), δ = (c, −c): KL = log cosh c.
        for c in [1e-6_f64, 1e-3] {
            // Series truncated after c⁶: the remainder is below 17c⁸/2520, relatively below c⁶ ≤ 10⁻¹⁸ of the value.
            let reference = c * c / 2.0 - c.powi(4) / 12.0 + c.powi(6) / 45.0;
            let divergence = categorical_kl_from_logits(&[0.0, 0.0], &[c, -c]).expect("valid logits");
            let band = divergence_band(reference) + reference * c.powi(6);
            assert!((divergence - reference).abs() <= band, "c = {c}: KL {divergence}, reference {reference}");
        }
        for c in [1.0_f64, 30.0] {
            // log cosh c = c + log1p(e^{−2c}) − log 2, three operands of size at most c + 1.
            let reference = c + (-2.0 * c).exp().ln_1p() - LN_2;
            let divergence = categorical_kl_from_logits(&[0.0, 0.0], &[c, -c]).expect("valid logits");
            let band = divergence_band(reference) + 8.0 * UNIT_ROUNDOFF * (c + 1.0);
            assert!((divergence - reference).abs() <= band, "c = {c}: KL {divergence}, reference {reference}");
        }
        // Positive control: the subtraction form lse(z′) − lse(z) − E_p δ at c = 10⁻⁶ is a difference of two numbers in
        // [½, 1), so it lies on the grid of 2⁻⁵³, which cannot hold 5·10⁻¹³ to the band's accuracy.
        let c = 1e-6_f64;
        let reference = c * c / 2.0 - c.powi(4) / 12.0;
        let subtraction = log_sum_exp(&[c, -c]).expect("valid logits") - log_sum_exp(&[0.0, 0.0]).expect("valid logits");
        assert!(
            (subtraction - reference).abs() > divergence_band(reference),
            "subtraction form {subtraction}, reference {reference}"
        );
    }

    #[test]
    fn identical_logits_and_a_constant_shift_give_zero_divergence() {
        let logits = [0.3, -1.2, 2.0, 0.7];
        assert_eq!(categorical_kl_from_logits(&logits, &logits), Ok(0.0));
        let shifted: Vec<f64> = logits.iter().map(|logit| logit + 7.0).collect();
        let divergence = categorical_kl_from_logits(&logits, &shifted).expect("valid logits");
        // Each computed gap is 7 up to two roundings of operands no larger than 9, so every centered gap is at most
        // 8u·9 and KL ≈ ½·E_p d² stays below (8u·9)².
        let floor = 8.0 * UNIT_ROUNDOFF * 9.0;
        assert!((0.0..=floor * floor).contains(&divergence), "KL {divergence}");
    }

    #[test]
    fn zero_probability_categories_follow_the_divergence_convention() {
        // p = (1, 0) and p′ = softmax(1, 5): KL = −log p′₀ = log(1 + e⁴), all of it mass off the support.
        let divergence =
            categorical_kl_from_logits(&[0.0, f64::NEG_INFINITY], &[1.0, 5.0]).expect("valid logits");
        let reference = 4.0_f64.exp().ln_1p();
        assert!(
            (divergence - reference).abs() <= 16.0 * UNIT_ROUNDOFF * reference,
            "KL {divergence}, reference {reference}"
        );
        // Positive control: dropping the off-support mass leaves KL(p ‖ q) = 0 for the one supported category.
        assert!(divergence > 4.0);
        // p′ vanishes where p does not.
        assert_eq!(
            categorical_kl_from_logits(&[0.0, 0.0], &[0.0, f64::NEG_INFINITY]),
            Ok(f64::INFINITY)
        );
        // p vanishes where p′ does not: KL((1, 0) ‖ (½, ½)) = log 2.
        let finite = categorical_kl_from_logits(&[0.0, f64::NEG_INFINITY], &[0.0, 0.0]).expect("valid logits");
        assert!((finite - LN_2).abs() <= 16.0 * UNIT_ROUNDOFF, "KL {finite}");
    }

    #[test]
    fn divergence_matches_the_log_softmax_route_on_generic_and_huge_logits() {
        let cases = [
            (
                vec![0.4, -1.3, 2.1, 0.0, -0.7, 1.5],
                vec![-0.2, 0.9, 1.7, 0.6, -2.0, 1.1],
            ),
            (vec![1000.0, 999.0, -1000.0], vec![1000.5, 998.0, -1000.0]),
        ];
        for (logits, perturbed) in cases {
            let divergence = categorical_kl_from_logits(&logits, &perturbed).expect("valid logits");
            let reference_log = log_softmax(&logits).expect("valid logits");
            let perturbed_log = log_softmax(&perturbed).expect("valid logits");
            let route: f64 = reference_log
                .iter()
                .zip(&perturbed_log)
                .map(|(a, b)| a.exp() * (a - b))
                .sum();
            // Each route term rounds a few ulps of |z_i| + |lse z| + |z′_i| + |lse z′| and the sum adds K of them.
            let normalizers = log_sum_exp(&logits).expect("valid logits").abs()
                + log_sum_exp(&perturbed).expect("valid logits").abs();
            let scale: f64 = logits
                .iter()
                .zip(&perturbed)
                .map(|(z, w)| z.abs() + w.abs() + normalizers)
                .sum();
            let band = 8.0 * UNIT_ROUNDOFF * scale;
            assert!(divergence > band, "the fixture must move the distribution; KL {divergence}");
            assert!((divergence - route).abs() <= band, "KL {divergence}, route {route}, band {band}");
        }
        // Positive control: the unshifted exponential sum overflows at logits of size 1000.
        assert!((1000.0_f64.exp() + 999.0_f64.exp()).is_infinite());
    }

    #[test]
    fn log_sum_exp_is_max_shifted_and_logits_naming_no_distribution_are_refused() {
        let shifted = log_sum_exp(&[1000.0, 1000.0, f64::NEG_INFINITY]).expect("valid logits");
        assert!((shifted - (1000.0 + LN_2)).abs() <= 8.0 * UNIT_ROUNDOFF * 1001.0, "lse {shifted}");
        let softmax = log_softmax(&[2.0, f64::NEG_INFINITY, -1.0]).expect("valid logits");
        assert_eq!(softmax[1], f64::NEG_INFINITY);
        let total: f64 = softmax.iter().map(|value| value.exp()).sum();
        assert!((total - 1.0).abs() <= 8.0 * UNIT_ROUNDOFF * 3.0, "softmax total {total}");

        assert_eq!(log_sum_exp(&[]), Err(CategoricalError::Empty { name: "logits" }));
        assert!(matches!(
            log_sum_exp(&[0.0, f64::NAN]),
            Err(CategoricalError::InvalidLogit { index: 1, .. })
        ));
        assert!(matches!(
            log_softmax(&[f64::INFINITY]),
            Err(CategoricalError::InvalidLogit { index: 0, .. })
        ));
        assert_eq!(
            log_sum_exp(&[f64::NEG_INFINITY; 3]),
            Err(CategoricalError::NoSupport { name: "logits" })
        );
        assert_eq!(
            categorical_kl_from_logits(&[0.0], &[0.0, 1.0]),
            Err(CategoricalError::LengthMismatch {
                reference: 1,
                perturbed: 2
            })
        );
        assert!(matches!(
            categorical_kl_from_logits(&[0.0, 0.0], &[f64::NEG_INFINITY, f64::NEG_INFINITY]),
            Err(CategoricalError::NoSupport { .. })
        ));
    }
}
