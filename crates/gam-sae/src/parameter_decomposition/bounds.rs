//! Fidelity bounds with explicit evidence status (#2951 P14, P15).
//!
//! Every function here returns an [`EvidenceStatus`] and never a stronger one than it proved. A certificate is
//! reported with the region it holds over, and a claim whose hypotheses fail is reported as unresolved rather than as a
//! number (the #2946 fr-census overclaim audit, issue comment 5716123817).
//!
//! # P15, the KL oscillation bound
//!
//! For logits `z, z′ ∈ ℝ^K` with gap `δ = z′ − z` and `p = softmax z`,
//!
//! ```text
//! KL(softmax z ‖ softmax z′) = log E_p e^δ − E_p δ ≤ osc(δ)²/8,      osc(δ) = max_i δ_i − min_i δ_i.
//! ```
//!
//! The identity: `log p_i − log p′_i = −δ_i + lse(z′) − lse(z)`, and `lse(z′) − lse(z) = log Σ_i p_i e^{δ_i}`. The
//! inequality is Hoeffding's lemma for `δ` under `p`, a variable with values in `[min δ, max δ]`. The same argument
//! under `p′` with gap `−δ` bounds the reverse divergence by the same number. A constant shift of the logits changes no
//! probability and has `osc = 0`.
//!
//! The bound reads only the oscillation, so a bound on the norm of the gap suffices: `osc(δ) ≤ 2‖δ‖_∞`, and
//! `osc(δ) ≤ √2‖δ‖₂` because `(a − b)² ≤ 2(a² + b²)`. Hence `KL ≤ ‖δ‖_∞²/2` and `KL ≤ ‖δ‖₂²/4`. Both are attained to
//! second order at `z = 0 ∈ ℝ²`: `δ = (ε, −ε)` and `δ = (ε/√2, −ε/√2)` give `KL = log cosh a = a²/2 − O(a⁴)` with
//! `a = ε` and `a = ε/√2`.
//!
//! # P14, a KL certificate over a ball of inputs
//!
//! A [`Contract`] chain ending at the logits bounds `‖F_n(x) − G_n(x)‖` by its `total_defect`, but only at inputs whose
//! trajectories stay inside every stage's certificate. [`certify_kl_over_input_ball`] asks [`whole_set_containment`]
//! for that over a declared ball and converts the gap bound through P15. When containment fails no upper bound is
//! derived, and the result is [`EvidenceStatus::Unresolved`] with `KL ≥ 0` as its only side.

use std::fmt;

use gam_linalg::roundoff::accumulation_growth;
use ndarray::ArrayView1;

use super::supports::{EvidenceStatus, EvidenceStatusError, Extremum};
use crate::inference::contracts::{Contract, whole_set_containment};

/// The norm a logit-gap bound is stated in, declared by the chart metric of the stage that outputs the logits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LogitGapNorm {
    /// `‖δ‖_∞`, with `osc(δ) ≤ 2‖δ‖_∞`.
    Supremum,
    /// `‖δ‖₂`, with `osc(δ) ≤ √2‖δ‖₂`.
    Euclidean,
}

impl LogitGapNorm {
    /// `c` in `KL ≤ ‖δ‖²/c`: `osc²/8` with `osc ≤ 2‖δ‖_∞` gives 2, and with `osc ≤ √2‖δ‖₂` gives 4.
    fn divisor(self) -> f64 {
        match self {
            Self::Supremum => 2.0,
            Self::Euclidean => 4.0,
        }
    }
}

/// The region a KL bound holds over. The bounded quantity is `KL(softmax z ‖ softmax z′)`, and equally the reverse
/// divergence.
#[derive(Clone, Debug, PartialEq)]
pub enum KlBoundRegion {
    /// One pair of logit vectors.
    LogitPair,
    /// Every perturbed logit vector whose gap from the reference has norm at most `radius`.
    LogitGapBall { radius: f64, norm: LogitGapNorm },
    /// Every chain input within `initial_radius` of the nominal input, whose native logits `G_n(x)` and realized logits
    /// `F_n(x)` a contained contract chain keeps within `total_defect` of each other in `norm`.
    ContractChainInputBall {
        initial_radius: f64,
        total_defect: f64,
        norm: LogitGapNorm,
    },
}

/// Why a bound could not be evaluated.
#[derive(Clone, Debug, PartialEq)]
pub enum BoundError {
    /// A shape, finiteness or sign requirement failed.
    InvalidInput(String),
    /// The evidence constructor refused the computed values.
    Evidence(EvidenceStatusError),
}

impl fmt::Display for BoundError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput(message) => f.write_str(message),
            Self::Evidence(error) => write!(f, "fidelity bound: {error}"),
        }
    }
}

impl std::error::Error for BoundError {}

impl From<EvidenceStatusError> for BoundError {
    fn from(error: EvidenceStatusError) -> Self {
        Self::Evidence(error)
    }
}

/// P15 for one pair of logit vectors: `KL ≤ osc(z′ − z)²/8`, a [`EvidenceStatus::UniformBound`] over
/// [`KlBoundRegion::LogitPair`].
///
/// Rounding: each computed gap `δ̂_i` is one correctly rounded subtraction, so `|δ̂_i − δ_i| ≤ u·|δ_i|`, and the exact
/// oscillation exceeds `max δ̂ − min δ̂` by at most `2u′·max|δ̂|` (`u′ = u/(1 − u)`). Forming that difference rounds by
/// `u` of its value. `γ_3·(osc + 2·max|δ̂|)` covers both and the rounding of the band itself, one `next_up` covers
/// adding the band, and one more covers the square and the division by 8, which is exact above the subnormal range and
/// adds at most half a unit in the last place below it.
pub fn softmax_kl_oscillation_bound(
    logits: ArrayView1<'_, f64>,
    perturbed: ArrayView1<'_, f64>,
) -> Result<EvidenceStatus<(), KlBoundRegion>, BoundError> {
    if logits.is_empty() || logits.len() != perturbed.len() {
        return Err(BoundError::InvalidInput(format!(
            "the KL oscillation bound needs two nonempty logit vectors of one length; got {} and {}",
            logits.len(),
            perturbed.len()
        )));
    }
    let mut largest = f64::NEG_INFINITY;
    let mut smallest = f64::INFINITY;
    for (index, (&reference, &moved)) in logits.iter().zip(perturbed.iter()).enumerate() {
        if !(reference.is_finite() && moved.is_finite()) {
            return Err(BoundError::InvalidInput(format!(
                "logit {index} must be finite in both vectors; got {reference} and {moved}"
            )));
        }
        let gap = moved - reference;
        largest = largest.max(gap);
        smallest = smallest.min(gap);
    }
    let oscillation = largest - smallest;
    let band = accumulation_growth(3) * (oscillation + 2.0 * largest.abs().max(smallest.abs()));
    let oscillation_upper = (oscillation + band).next_up();
    let upper = (oscillation_upper * oscillation_upper / 8.0).next_up();
    let nominal = oscillation * oscillation / 8.0;
    Ok(EvidenceStatus::uniform_bound(
        upper,
        upper - nominal,
        KlBoundRegion::LogitPair,
    )?)
}

/// P15 from a bound on the gap's norm alone: `KL ≤ ‖δ‖_∞²/2` or `KL ≤ ‖δ‖₂²/4`, a [`EvidenceStatus::UniformBound`]
/// over [`KlBoundRegion::LogitGapBall`].
pub fn kl_bound_from_logit_gap(
    gap: f64,
    norm: LogitGapNorm,
) -> Result<EvidenceStatus<(), KlBoundRegion>, BoundError> {
    if !(gap.is_finite() && gap >= 0.0) {
        return Err(BoundError::InvalidInput(format!(
            "a logit-gap KL bound needs a finite non-negative gap; got {gap}"
        )));
    }
    let upper = kl_upper_from_gap(gap, norm);
    Ok(EvidenceStatus::uniform_bound(
        upper,
        upper - gap * gap / norm.divisor(),
        KlBoundRegion::LogitGapBall { radius: gap, norm },
    )?)
}

/// P14 then P15: a KL bound at every input of a declared ball, from a [`Contract`] chain whose last stage outputs
/// logits measured in `norm`.
///
/// When [`whole_set_containment`] proves every stage contained, `‖F_n(x) − G_n(x)‖ ≤ total_defect` on the whole ball
/// and the result is a [`EvidenceStatus::UniformBound`] of `total_defect²/c`. `total_defect` is formed by at most `2n`
/// rounded operations on non-negative operands for `n` stages (`n − j` products and `n − j` additions carry stage `j`'s
/// term), so `γ_{2n+3}·total_defect` bounds its rounding together with the band's own, as in
/// [`StageContainment::rounding_band`](crate::inference::contracts::StageContainment::rounding_band). When containment
/// fails the result is [`EvidenceStatus::Unresolved`] with lower side `0` and no upper side.
pub fn certify_kl_over_input_ball(
    chain: &[Contract],
    initial_radius: f64,
    nominal_offsets: &[f64],
    norm: LogitGapNorm,
) -> Result<EvidenceStatus<(), KlBoundRegion>, BoundError> {
    let containment = whole_set_containment(chain, initial_radius, nominal_offsets)
        .map_err(BoundError::InvalidInput)?;
    let total_defect = containment.composed.total_defect;
    let region = KlBoundRegion::ContractChainInputBall {
        initial_radius,
        total_defect,
        norm,
    };
    if !containment.contained {
        return Ok(EvidenceStatus::unresolved(
            0.0,
            f64::INFINITY,
            Extremum::Supremum,
            None,
            region,
        )?);
    }
    let gap = total_defect + accumulation_growth(2 * chain.len() + 3) * total_defect;
    let upper = kl_upper_from_gap(gap, norm);
    Ok(EvidenceStatus::uniform_bound(
        upper,
        upper - total_defect * total_defect / norm.divisor(),
        region,
    )?)
}

/// `gap²/c` rounded up. Dividing by a power of two is exact above the subnormal range, so one `next_up` covers the
/// rounded square there, and below it the two roundings add at most one unit in the last place. A zero gap is exact.
fn kl_upper_from_gap(gap: f64, norm: LogitGapNorm) -> f64 {
    if gap == 0.0 {
        0.0
    } else {
        (gap * gap / norm.divisor()).next_up()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gam_linalg::roundoff::{UNIT_ROUNDOFF, accumulation_band};
    use gam_math::categorical::categorical_kl_from_logits;
    use ndarray::array;

    fn contract(name: &str, domain_radius: f64, defect: f64, lipschitz: f64) -> Contract {
        Contract {
            name: name.to_string(),
            domain_radius,
            defect,
            lipschitz,
        }
    }

    /// `KL(softmax z ‖ softmax z′)` from gam-math's categorical owner, with a first-order relative band per the owner's
    /// accuracy section: the series below `80u`, the probabilities, normalizers and `log1p` below `48u` plus `4u` per
    /// unit of the largest logit or gap, and the log-space round trip `4u·|log KL|`.
    fn divergence_with_band(logits: &[f64], perturbed: &[f64]) -> (f64, f64) {
        let divergence = categorical_kl_from_logits(logits, perturbed).expect("valid logits");
        let magnitude = logits
            .iter()
            .chain(perturbed)
            .fold(0.0_f64, |largest, value| largest.max(value.abs()));
        let relative = 128.0 + 4.0 * magnitude + 4.0 * divergence.ln().abs();
        (divergence, relative * UNIT_ROUNDOFF * divergence)
    }

    #[test]
    fn oscillation_bound_is_tight_on_the_symmetric_pair_and_its_constant_cannot_be_halved() {
        // z = (0, 0), z′ = (c, −c): p = (½, ½), KL = log cosh c ≥ c²/2 − c⁴/12, osc = 2c, bound c²/2.
        for c in [1e-3, 1e-1, 1.0, 3.0] {
            let status =
                softmax_kl_oscillation_bound(array![0.0, 0.0].view(), array![c, -c].view()).expect("finite logits");
            assert!(matches!(status, EvidenceStatus::UniformBound { .. }));
            let upper = status.upper_bound().expect("a uniform bound has an upper side");
            let (divergence, band) = divergence_with_band(&[0.0, 0.0], &[c, -c]);
            assert!(divergence - band <= upper, "c = {c}: KL {divergence} above bound {upper}");
            if c < 1.0 {
                // The Taylor remainder puts KL within c²/6 of the bound, relatively.
                assert!(divergence + band >= upper * (1.0 - c * c), "c = {c}: KL {divergence}, bound {upper}");
            }
        }
        // Positive control: osc²/16 is violated at c = 10⁻³, where KL ≈ c²/2 = 2·(osc²/16).
        let c = 1e-3;
        let (divergence, band) = divergence_with_band(&[0.0, 0.0], &[c, -c]);
        assert!(divergence - band > (2.0 * c) * (2.0 * c) / 16.0, "KL {divergence}");
    }

    #[test]
    fn oscillation_bound_dominates_the_divergence_on_generic_logits() {
        let logits = [0.4, -1.3, 2.1, 0.0, -0.7, 1.5];
        let perturbed = [-0.2, 0.9, 1.7, 0.6, -2.0, 1.1];
        let status = softmax_kl_oscillation_bound(
            ndarray::ArrayView1::from(&logits[..]),
            ndarray::ArrayView1::from(&perturbed[..]),
        )
        .expect("finite logits");
        let upper = status.upper_bound().expect("a uniform bound has an upper side");
        // osc = 2.2 − (−1.3) = 3.5, so the bound is 3.5²/8. Its relative excess is twice the oscillation's (the band
        // γ_3·7.9/3.5, three gap roundings of |δ| ≤ 2.2 and a last-place unit, together below γ_6) plus one last-place
        // unit, below γ_24.
        assert!(upper >= 3.5 * 3.5 / 8.0);
        assert!(upper <= 3.5 * 3.5 / 8.0 * (1.0 + accumulation_growth(24)), "upper {upper}");
        let (divergence, band) = divergence_with_band(&logits, &perturbed);
        assert!(divergence > band, "the fixture must move the distribution; KL {divergence}");
        assert!(divergence - band <= upper, "KL {divergence} above bound {upper}");
    }

    #[test]
    fn a_constant_logit_shift_has_a_rounding_floor_bound() {
        let logits = array![0.3, -1.2, 2.0, 0.7];
        let shifted = logits.mapv(|value| value + 5.0);
        let status = softmax_kl_oscillation_bound(logits.view(), shifted.view()).expect("finite logits");
        let upper = status.upper_bound().expect("a uniform bound has an upper side");
        // Every computed gap is 5 up to two roundings of operands no larger than 7, so the computed oscillation is at
        // most γ_2·14, the band adds γ_3·(osc + 10), and (osc + band)²/8 stays below (γ_8·7)².
        let floor = accumulation_band(8, 7.0);
        assert!(upper <= floor * floor, "upper {upper}");
        // Positive control: tilting one logit by 10⁻³ lifts the bound to (10⁻³)²/8, above the floor.
        let mut tilted = shifted;
        tilted[0] += 1e-3;
        let raised = softmax_kl_oscillation_bound(logits.view(), tilted.view())
            .expect("finite logits")
            .upper_bound()
            .expect("a uniform bound has an upper side");
        assert!(raised >= (1e-3_f64 * (1.0 - 1e-6)).powi(2) / 8.0, "raised {raised}");
        assert!(raised > floor * floor, "raised {raised}");
    }

    #[test]
    fn logit_gap_bounds_are_attained_to_second_order_and_the_norm_declaration_is_load_bearing() {
        let gap = 1e-3;
        let supremum = kl_bound_from_logit_gap(gap, LogitGapNorm::Supremum)
            .expect("finite gap")
            .upper_bound()
            .expect("a uniform bound has an upper side");
        let euclidean = kl_bound_from_logit_gap(gap, LogitGapNorm::Euclidean)
            .expect("finite gap")
            .upper_bound()
            .expect("a uniform bound has an upper side");
        // `ln cosh a` at a ≤ 10⁻³: cosh rounds within one unit in the last place of 1, and the logarithm near 1 adds
        // at most its own rounding.
        let band = accumulation_band(4, 1.0);
        // ‖(ε, −ε)‖_∞ = ε.
        let supremum_divergence = gap.cosh().ln();
        assert!(supremum_divergence - band <= supremum);
        assert!(supremum_divergence + band >= supremum * (1.0 - gap * gap));
        // ‖(ε/√2, −ε/√2)‖₂ = ε.
        let euclidean_divergence = (gap / 2.0_f64.sqrt()).cosh().ln();
        assert!(euclidean_divergence - band <= euclidean);
        assert!(euclidean_divergence + band >= euclidean * (1.0 - gap * gap));
        // Positive control: the Euclidean constant applied to a supremum-norm gap is violated by (ε, −ε).
        assert!(supremum_divergence - band > euclidean, "KL {supremum_divergence}, bound {euclidean}");
        // A zero gap is exact.
        let zero = kl_bound_from_logit_gap(0.0, LogitGapNorm::Supremum).expect("finite gap");
        assert_eq!(zero.upper_bound(), Some(0.0));
        let refused = kl_bound_from_logit_gap(f64::NAN, LogitGapNorm::Euclidean).expect_err("NaN gap");
        assert!(matches!(refused, BoundError::InvalidInput(..)));
    }

    #[test]
    fn input_ball_certificate_is_withheld_when_containment_fails() {
        // Stage 1 expands by 3 with defect 0.01; stage 2 has defect 0.01. Both balls have radius 1 about the nominal
        // trajectory, so a ball of radius 0.3 is contained (0.9 + 0.01 ≤ 1) and one of radius 0.5 is not.
        let chain = [
            contract("expand", 1.0, 0.01, 3.0),
            contract("logits", 1.0, 0.01, 1.0),
        ];
        let contained = certify_kl_over_input_ball(&chain, 0.3, &[0.0, 0.0], LogitGapNorm::Supremum)
            .expect("valid chain");
        assert!(matches!(contained, EvidenceStatus::UniformBound { .. }));
        let upper = contained.upper_bound().expect("a uniform bound has an upper side");
        // total_defect = 0.02, so the bound is 0.02²/2 widened by γ_7 and one next_up.
        assert!(upper >= 0.02 * 0.02 / 2.0, "upper {upper}");
        assert!(upper <= 0.02 * 0.02 / 2.0 * (1.0 + accumulation_growth(20)), "upper {upper}");

        let escaped = certify_kl_over_input_ball(&chain, 0.5, &[0.0, 0.0], LogitGapNorm::Supremum)
            .expect("valid chain");
        assert!(matches!(escaped, EvidenceStatus::Unresolved { .. }));
        assert_eq!(escaped.upper_bound(), None);
        assert_eq!(escaped.lower_bound(), Some(0.0));
        assert!(!escaped.certifies_at_most(f64::MAX));

        let refused = certify_kl_over_input_ball(&chain, 0.3, &[0.0], LogitGapNorm::Supremum)
            .expect_err("offset count");
        assert!(matches!(refused, BoundError::InvalidInput(..)));
    }
}
