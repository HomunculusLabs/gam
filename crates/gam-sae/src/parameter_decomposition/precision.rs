//! Declared-precision real codes, quotient representatives and decode-then-evaluate
//! distortion (#2951, P11 and P18).
//!
//! A mechanism program's code carries no free real-valued coefficient. Every real
//! an artifact holds is sent as integers under a precision the experiment declares,
//! and the decoder rebuilds it from those integers and the declaration alone.
//! Fidelity is then measured on what the decoder rebuilt.
//!
//! # Dyadic lattice codes
//!
//! A [`DeclaredPrecision`] is the step `2^-p` for a declared integer `p`, so the
//! precision is itself an integer and costs integer bits wherever an artifact
//! transmits it. A real `x` is sent as `k = round(x·2^p)` and decoded as
//! `x̂ = k·2^-p`. Scaling by a power of two is exact in binary floating point and
//! rounding to an integer is exact, so for `|k| ≤ 2^53` the index and the decoded
//! value are both exact and
//!
//! ```text
//!   |x − x̂| ≤ 2^-(p+1)
//! ```
//!
//! holds with no rounding term. A step that is not a power of two would put a
//! rounding error into every decoded value.
//!
//! # Quotient representatives
//!
//! A coordinate of `R/TZ`, such as the projective line `RP1` as `t ~ t + π` (P11) or
//! a turn angle with period `2π`, names one class. A code that sends `t` and `t + T`
//! differently spends bits on the gauge. [`QuotientCode`] sends the class once: the
//! canonical representative in `[0, T)` is a convention the encoder and decoder
//! share, and the transmitted index names the nearest of `N = 2^b` cells of the
//! period, at `b` bits per coordinate.
//!
//! With `u` the unit roundoff, the encoder computes:
//! - `r = t mod T`. The floating-point remainder is exact; only the correction
//!   `r + T` of a negative remainder rounds, by at most `uT`.
//! - The fraction `s = r/T`, one division, rounding by at most `u` since `r ≤ T`.
//! - `j = round(s·N) mod N`, which is exact.
//!
//! The decoder returns `j·(T/N)`. The cell width is an exact scaling, and the
//! product rounds by at most `uT`. In period units the decoded representative is
//! within `1/(2N) + 3u` of the input's class, so
//!
//! ```text
//!   d_{R/TZ}(t, t̂) < T·(2^-(b+1) + 3u).
//! ```
//!
//! [`QuotientCode::worst_case_error`] reports `T·2^-(b+1) + 4uT`. Both terms are exact
//! power-of-two scalings of `T`, and the extra `uT` absorbs the one rounded addition,
//! so the figure never falls below the bound. A cell whose half-width `2^-(b+1)` does
//! not exceed the `3u` rounding floor is not resolved and is refused.
//!
//! # Decode, then evaluate
//!
//! [`decode_then_evaluate`] decodes an artifact, executes the decoded artifact, and
//! measures the declared distortion of its outputs against the native reference.
//! The quantization error of the parameters never stands in for the error of the
//! outputs: the computation is nonlinear, and a parameter error bounds an output
//! error only through a derived Lipschitz constant. The measurement covers the inputs
//! the evaluation executes and certifies nothing beyond them (#2946 fr-census
//! overclaim audit, comment 5716123817).

use gam_linalg::roundoff::UNIT_ROUNDOFF;

/// Largest `|p|` for which both `2^p` and `2^-p` are normal `f64` powers of two: the
/// normal exponents span `[f64::MIN_EXP - 1, f64::MAX_EXP - 1] = [-1022, 1023]`.
const EXPONENT_LIMIT: i32 = -(f64::MIN_EXP - 1);

/// Every integer of magnitude at most `2^53` is an `f64`, so an index in this range
/// decodes exactly.
const INDEX_LIMIT: u64 = 1 << f64::MANTISSA_DIGITS;

/// `2^exponent` for a normal exponent in `[-1022, 1023]`, built from its bits: an
/// all-zero fraction under the biased exponent `exponent + 1023`.
fn power_of_two(exponent: i32) -> f64 {
    f64::from_bits(((exponent + f64::MAX_EXP - 1) as u64) << (f64::MANTISSA_DIGITS - 1))
}

/// A declared precision for real coordinates: the dyadic step `2^-fraction_bits`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DeclaredPrecision {
    fraction_bits: i32,
}

impl DeclaredPrecision {
    /// The step `2^-fraction_bits`: binary digits after the point, negative for a
    /// step above one. Refused when `2^fraction_bits` or `2^-fraction_bits` is not a
    /// normal `f64`, because the encoder and decoder then stop being exact scalings.
    pub fn new(fraction_bits: i32) -> Result<Self, String> {
        if fraction_bits.unsigned_abs() > EXPONENT_LIMIT.unsigned_abs() {
            return Err(format!(
                "DeclaredPrecision: fraction bits {fraction_bits} put the step 2^-{fraction_bits} \
                 or its inverse outside the normal f64 exponents (|p| <= {EXPONENT_LIMIT})"
            ));
        }
        Ok(Self { fraction_bits })
    }

    /// The declared binary digits after the point.
    pub fn fraction_bits(self) -> i32 {
        self.fraction_bits
    }

    /// The lattice step `2^-fraction_bits`.
    pub fn step(self) -> f64 {
        power_of_two(-self.fraction_bits)
    }

    /// The exact worst-case decoding error `2^-(fraction_bits+1)`, with no rounding
    /// term (see the module note).
    pub fn worst_case_error(self) -> f64 {
        0.5 * self.step()
    }
}

/// An artifact that a decoder rebuilds from integers and declarations alone.
pub trait DecodableArtifact {
    /// What the decoder rebuilds.
    type Decoded;

    /// Rebuild the artifact from its transmitted integers.
    fn decode(&self) -> Result<Self::Decoded, String>;
}

/// Reals sent as indices on the dyadic lattice of one [`DeclaredPrecision`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LatticeCode {
    precision: DeclaredPrecision,
    indices: Vec<i64>,
}

impl LatticeCode {
    /// Encode each value as its nearest lattice index `round(x·2^p)`, with ties rounded
    /// away from zero. Refused for a non-finite value, for an index beyond `2^53`, or
    /// when the decoded value would overflow.
    pub fn encode(values: &[f64], precision: DeclaredPrecision) -> Result<Self, String> {
        let scale = power_of_two(precision.fraction_bits);
        let mut indices = Vec::with_capacity(values.len());
        for (position, &value) in values.iter().enumerate() {
            if !value.is_finite() {
                return Err(format!(
                    "LatticeCode::encode: value {value} at position {position} is not finite"
                ));
            }
            let index = (value * scale).round();
            if !(index.abs() <= INDEX_LIMIT as f64) {
                return Err(format!(
                    "LatticeCode::encode: value {value} at position {position} needs index {index} \
                     at step 2^-{}, beyond the exactly decodable 2^53",
                    precision.fraction_bits
                ));
            }
            indices.push(index as i64);
        }
        Self::from_indices(precision, indices)
    }

    /// Accept transmitted indices. Refused for an index beyond `2^53` or one whose
    /// decoded value overflows.
    pub fn from_indices(precision: DeclaredPrecision, indices: Vec<i64>) -> Result<Self, String> {
        let step = precision.step();
        for (position, &index) in indices.iter().enumerate() {
            if index.unsigned_abs() > INDEX_LIMIT {
                return Err(format!(
                    "LatticeCode: index {index} at position {position} is beyond the exactly \
                     decodable 2^53"
                ));
            }
            if !(index as f64 * step).is_finite() {
                return Err(format!(
                    "LatticeCode: index {index} at position {position} overflows at step 2^-{}",
                    precision.fraction_bits
                ));
            }
        }
        Ok(Self { precision, indices })
    }

    /// The declared precision.
    pub fn precision(&self) -> DeclaredPrecision {
        self.precision
    }

    /// The transmitted lattice indices.
    pub fn indices(&self) -> &[i64] {
        &self.indices
    }
}

impl DecodableArtifact for LatticeCode {
    type Decoded = Vec<f64>;

    fn decode(&self) -> Result<Vec<f64>, String> {
        let step = self.precision.step();
        Ok(self.indices.iter().map(|&index| index as f64 * step).collect())
    }
}

/// The quotient `R/TZ` of the real line by a period `T`, such as `RP1` with `T = π`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PeriodicQuotient {
    period: f64,
}

impl PeriodicQuotient {
    /// The quotient by `period`. Refused unless the period is finite and positive, and
    /// its scaling by `2^-53` is still normal. That condition makes every power-of-two
    /// scaling of the period used here (at most `2^-51`) exact.
    pub fn new(period: f64) -> Result<Self, String> {
        let admissible = period.is_finite()
            && period > 0.0
            && (period * power_of_two(-(f64::MANTISSA_DIGITS as i32))).is_normal();
        if !admissible {
            return Err(format!(
                "PeriodicQuotient: period {period} must be finite and positive, with its 2^-53 \
                 scaling still a normal f64"
            ));
        }
        Ok(Self { period })
    }

    /// The period `T`.
    pub fn period(self) -> f64 {
        self.period
    }
}

/// Refuse a resolution whose cell half-width `2^-(b+1)`, in period units, does not
/// exceed the `3u` rounding of the representative (see the module note).
fn check_resolution(resolution_bits: u32) -> Result<(), String> {
    let resolved = resolution_bits < f64::MANTISSA_DIGITS
        && power_of_two(-(resolution_bits as i32) - 1) > 3.0 * UNIT_ROUNDOFF;
    if !resolved {
        return Err(format!(
            "QuotientCode: {resolution_bits} resolution bits make cells no wider than the \
             rounding of their representative"
        ));
    }
    Ok(())
}

/// Coordinates of a [`PeriodicQuotient`] sent once per class, as cell indices of the
/// canonical representative in `[0, T)`.
#[derive(Clone, Debug, PartialEq)]
pub struct QuotientCode {
    quotient: PeriodicQuotient,
    resolution_bits: u32,
    indices: Vec<u64>,
}

impl QuotientCode {
    /// Encode each value's class as the nearest of `2^resolution_bits` cells of the
    /// period. Refused for a non-finite value or an unresolved resolution.
    pub fn encode(
        values: &[f64],
        quotient: PeriodicQuotient,
        resolution_bits: u32,
    ) -> Result<Self, String> {
        check_resolution(resolution_bits)?;
        let cells = power_of_two(resolution_bits as i32);
        let modulus = 1_u64 << resolution_bits;
        let period = quotient.period;
        let mut indices = Vec::with_capacity(values.len());
        for (position, &value) in values.iter().enumerate() {
            if !value.is_finite() {
                return Err(format!(
                    "QuotientCode::encode: value {value} at position {position} is not finite"
                ));
            }
            let fraction = value.rem_euclid(period) / period;
            indices.push((fraction * cells).round() as u64 % modulus);
        }
        Ok(Self {
            quotient,
            resolution_bits,
            indices,
        })
    }

    /// Accept transmitted cell indices. Refused for an unresolved resolution or an
    /// index outside `[0, 2^resolution_bits)`.
    pub fn from_indices(
        quotient: PeriodicQuotient,
        resolution_bits: u32,
        indices: Vec<u64>,
    ) -> Result<Self, String> {
        check_resolution(resolution_bits)?;
        let modulus = 1_u64 << resolution_bits;
        if let Some(position) = indices.iter().position(|&index| index >= modulus) {
            return Err(format!(
                "QuotientCode: index {} at position {position} is outside [0, 2^{resolution_bits})",
                indices[position]
            ));
        }
        Ok(Self {
            quotient,
            resolution_bits,
            indices,
        })
    }

    /// The quotient the coordinates live in.
    pub fn quotient(&self) -> PeriodicQuotient {
        self.quotient
    }

    /// Bits per coordinate.
    pub fn resolution_bits(&self) -> u32 {
        self.resolution_bits
    }

    /// The transmitted cell indices.
    pub fn indices(&self) -> &[u64] {
        &self.indices
    }

    /// The exact length of the fixed-rate index code: `resolution_bits` per coordinate.
    pub fn index_bits(&self) -> u64 {
        u64::from(self.resolution_bits) * self.indices.len() as u64
    }

    /// An upper bound on the quotient distance between each input's class and its
    /// decoded representative, rounding included: `T·2^-(b+1) + 4uT` (see the module
    /// note).
    pub fn worst_case_error(&self) -> f64 {
        let period = self.quotient.period;
        period * power_of_two(-(self.resolution_bits as i32) - 1) + 4.0 * UNIT_ROUNDOFF * period
    }
}

impl DecodableArtifact for QuotientCode {
    type Decoded = Vec<f64>;

    fn decode(&self) -> Result<Vec<f64>, String> {
        let cell_width = self.quotient.period * power_of_two(-(self.resolution_bits as i32));
        Ok(self
            .indices
            .iter()
            .map(|&index| index as f64 * cell_width)
            .collect())
    }
}

/// The declared distortion of a decoded artifact's outputs against the native
/// reference, under the declared fidelity tolerance.
///
/// This is a measurement over the inputs the evaluation executed and bounds nothing
/// beyond them (#2946 fr-census overclaim audit, comment 5716123817).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DecodedFidelity {
    /// The declared distortion of the decoded artifact's outputs against the native
    /// reference.
    pub distortion: f64,
    /// The declared fidelity tolerance.
    pub tolerance: f64,
}

impl DecodedFidelity {
    /// Whether the measured distortion is within the declared tolerance.
    pub fn meets_tolerance(&self) -> bool {
        self.distortion <= self.tolerance
    }
}

/// Decode `artifact`, execute the decoded artifact with `evaluate`, and measure
/// `distortion(outputs, native_reference)` under the declared `tolerance`.
///
/// The evaluation receives only what the decoder rebuilt. The tolerance is a required
/// experiment declaration and is refused unless finite and non-negative. A distortion
/// that is not finite and non-negative is refused rather than compared.
pub fn decode_then_evaluate<A, O, E, D>(
    artifact: &A,
    evaluate: E,
    native_reference: &O,
    distortion: D,
    tolerance: f64,
) -> Result<DecodedFidelity, String>
where
    A: DecodableArtifact,
    E: FnOnce(&A::Decoded) -> Result<O, String>,
    D: FnOnce(&O, &O) -> f64,
{
    if !(tolerance.is_finite() && tolerance >= 0.0) {
        return Err(format!(
            "decode_then_evaluate: the declared fidelity tolerance {tolerance} must be finite \
             and non-negative"
        ));
    }
    let decoded = artifact.decode()?;
    let outputs = evaluate(&decoded)?;
    let measured = distortion(&outputs, native_reference);
    if !(measured.is_finite() && measured >= 0.0) {
        return Err(format!(
            "decode_then_evaluate: the declared distortion returned {measured}, not a finite \
             non-negative figure"
        ));
    }
    Ok(DecodedFidelity {
        distortion: measured,
        tolerance,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;
    use std::f64::consts::PI;

    /// The distance between the classes of `value` and `representative` in `R/TZ`,
    /// measured to within `4uT`. The remainder `%` is exact. The subtraction rounds by
    /// at most `2uT`, since the offset is below `2T`; the Euclidean correction by at
    /// most `uT`; and the complement by at most `uT`.
    fn class_distance(value: f64, representative: f64, period: f64) -> f64 {
        let wrapped = (value % period - representative).rem_euclid(period);
        wrapped.min(period - wrapped)
    }

    /// `sum_c (2/C) v(t_c) v(t_c)^T` with `v(t) = (cos t, sin t)`, row-major.
    fn projector_sum(angles: &[f64]) -> [f64; 4] {
        let weight = 2.0 / angles.len() as f64;
        let mut sum = [0.0; 4];
        for &angle in angles {
            let (sine, cosine) = angle.sin_cos();
            sum[0] += weight * cosine * cosine;
            sum[1] += weight * cosine * sine;
            sum[2] += weight * sine * cosine;
            sum[3] += weight * sine * sine;
        }
        sum
    }

    fn frobenius_distance(left: &[f64; 4], right: &[f64; 4]) -> f64 {
        left.iter()
            .zip(right)
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>()
            .sqrt()
    }

    #[test]
    fn lattice_decoding_error_is_at_most_half_a_step_with_no_rounding_term() {
        for fraction_bits in [3, 0, -4] {
            let precision = DeclaredPrecision::new(fraction_bits).expect("precision in range");
            let step = precision.step();
            let values = [
                0.0,
                0.5 * step,
                -0.5 * step,
                0.3 * step,
                1.0 / 3.0,
                -7.0 / 3.0,
                123.456,
                1.0e6 + 0.3,
            ];
            let code = LatticeCode::encode(&values, precision).expect("finite values encode");
            let decoded = code.decode().expect("a lattice code decodes");
            for ((&value, &index), &estimate) in values.iter().zip(code.indices()).zip(&decoded) {
                assert_eq!(estimate.to_bits(), (index as f64 * step).to_bits());
                // Either the estimate is zero or |estimate| >= step while the error is at
                // most step/2, so estimate/2 <= value <= 2·estimate (Sterbenz) and the
                // measured difference is exact: the bound is asserted with no slack.
                assert!(
                    (value - estimate).abs() <= precision.worst_case_error(),
                    "value {value} decoded to {estimate} at step {step}"
                );
            }
            // A value half a step off the lattice attains the bound, so it is tight.
            assert_eq!((values[1] - decoded[1]).abs(), precision.worst_case_error());
            // Decoded values are codewords: re-encoding them is a fixed point.
            let reencoded = LatticeCode::encode(&decoded, precision).expect("decoded values encode");
            assert_eq!(reencoded.indices(), code.indices());
        }
    }

    #[test]
    fn lattice_code_refuses_what_it_cannot_decode_exactly() {
        assert!(DeclaredPrecision::new(EXPONENT_LIMIT).is_ok());
        assert!(DeclaredPrecision::new(-EXPONENT_LIMIT).is_ok());
        assert!(DeclaredPrecision::new(EXPONENT_LIMIT + 1).is_err());
        assert!(DeclaredPrecision::new(i32::MIN).is_err());

        let unit = DeclaredPrecision::new(0).expect("unit step");
        assert!(LatticeCode::encode(&[f64::NAN], unit).is_err());
        assert!(LatticeCode::encode(&[f64::NEG_INFINITY], unit).is_err());
        // Index guard: 2^53 is accepted, 2^54 is refused.
        assert!(LatticeCode::encode(&[INDEX_LIMIT as f64], unit).is_ok());
        assert!(LatticeCode::encode(&[2.0 * INDEX_LIMIT as f64], unit).is_err());
        assert!(LatticeCode::from_indices(unit, vec![INDEX_LIMIT as i64]).is_ok());
        assert!(LatticeCode::from_indices(unit, vec![-(INDEX_LIMIT as i64) - 1]).is_err());
        // Overflow guard at the coarsest step 2^1022: 2^1023 decodes, while f64::MAX
        // rounds to index 4, whose decoded 2^1024 overflows.
        let coarsest = DeclaredPrecision::new(-EXPONENT_LIMIT).expect("coarsest step");
        assert!(LatticeCode::encode(&[power_of_two(f64::MAX_EXP - 1)], coarsest).is_ok());
        assert!(LatticeCode::encode(&[f64::MAX], coarsest).is_err());
    }

    #[test]
    fn quotient_code_decodes_within_its_bound_of_each_class() {
        let quotient = PeriodicQuotient::new(PI).expect("the RP1 period is admissible");
        let values = [
            0.0,
            1.0,
            -1.0,
            0.5 * PI,
            PI,
            -PI,
            3.0,
            1.0e6 + 0.1,
            -12345.678,
            1.0e-300,
        ];
        let measurement_band = 4.0 * UNIT_ROUNDOFF * PI;
        for resolution_bits in [0, 1, 7, 30, 50] {
            let code = QuotientCode::encode(&values, quotient, resolution_bits)
                .expect("finite values encode");
            let decoded = code.decode().expect("a quotient code decodes");
            let bound = code.worst_case_error();
            for (&value, &representative) in values.iter().zip(&decoded) {
                assert!(
                    (0.0..PI).contains(&representative),
                    "representative {representative} of {value} is outside [0, pi)"
                );
                let distance = class_distance(value, representative, PI);
                assert!(
                    distance <= (bound + measurement_band).next_up(),
                    "value {value} decoded to {representative} at {resolution_bits} bits: \
                     distance {distance} beyond bound {bound}"
                );
            }
            let reencoded = QuotientCode::encode(&decoded, quotient, resolution_bits)
                .expect("decoded representatives encode");
            assert_eq!(reencoded.indices(), code.indices());
            assert_eq!(
                code.index_bits(),
                u64::from(resolution_bits) * values.len() as u64
            );
        }
    }

    #[test]
    fn translates_of_a_class_share_one_quotient_index_but_not_one_lattice_index() {
        let quotient = PeriodicQuotient::new(PI).expect("the RP1 period is admissible");
        let resolution_bits = 10;
        let cell_width = PI * power_of_two(-10);
        let precision = DeclaredPrecision::new(10).expect("precision in range");
        for cell_index in [0_u64, 1, 511, 1023] {
            // A quarter cell off a codeword. Translating by up to three periods rounds by
            // at most 7u in period units: u·3pi for the product and u·4pi for the sum.
            // Encoding adds 2u. Both are far inside the quarter-cell margin of 2^-12.
            let base = (cell_index as f64 + 0.25) * cell_width;
            let translates: Vec<f64> = (-3..=3).map(|turns| base + f64::from(turns) * PI).collect();
            let code = QuotientCode::encode(&translates, quotient, resolution_bits)
                .expect("translates encode");
            assert!(
                code.indices().iter().all(|&index| index == cell_index),
                "translates of cell {cell_index} encoded to {:?}",
                code.indices()
            );
            // Negative control: the plain lattice code sends the translates as seven reals.
            let lattice = LatticeCode::encode(&translates, precision).expect("translates encode");
            let distinct: BTreeSet<i64> = lattice.indices().iter().copied().collect();
            assert_eq!(distinct.len(), translates.len());
        }
    }

    #[test]
    fn quotient_code_refuses_cells_below_the_rounding_of_their_representative() {
        let quotient = PeriodicQuotient::new(PI).expect("the RP1 period is admissible");
        // Half-width 2^-51 = 4u exceeds the 3u floor; 2^-52 = 2u does not.
        assert!(QuotientCode::encode(&[1.0], quotient, 50).is_ok());
        assert!(QuotientCode::encode(&[1.0], quotient, 51).is_err());
        assert!(QuotientCode::encode(&[1.0], quotient, u32::MAX).is_err());
        assert!(QuotientCode::encode(&[f64::NAN], quotient, 3).is_err());
        assert!(QuotientCode::from_indices(quotient, 3, vec![7]).is_ok());
        assert!(QuotientCode::from_indices(quotient, 3, vec![8]).is_err());
        assert!(PeriodicQuotient::new(1.0e-200).is_ok());
        for period in [0.0, -PI, f64::NAN, f64::INFINITY, 1.0e-300] {
            assert!(
                PeriodicQuotient::new(period).is_err(),
                "period {period} was admitted"
            );
        }
    }

    #[test]
    fn fidelity_is_measured_on_the_decoded_artifact_not_on_the_parameters_it_came_from() {
        // P11: (2/C) v(t_c) v(t_c)^T at t_c = c·pi/C sums to I_2 on RP1.
        let angles: Vec<f64> = (0..3).map(|component| f64::from(component) * PI / 3.0).collect();
        let identity = [1.0, 0.0, 0.0, 1.0];
        let tolerance = 0.1;
        let quotient = PeriodicQuotient::new(PI).expect("the RP1 period is admissible");
        let evaluate = |decoded: &Vec<f64>| Ok::<[f64; 4], String>(projector_sum(decoded));
        let distortion =
            |outputs: &[f64; 4], native: &[f64; 4]| frobenius_distance(outputs, native);

        // Positive control: the parameters before coding reproduce the identity, so an
        // evaluation that read them would pass.
        assert!(frobenius_distance(&projector_sum(&angles), &identity) <= tolerance);

        // At 2 bits the decoded angles are 0, pi/4 and 3pi/4, whose projector sum
        // misses the identity by sqrt(2)/3.
        let coarse = QuotientCode::encode(&angles, quotient, 2).expect("angles encode");
        assert_eq!(coarse.indices(), &[0_u64, 1, 3]);
        let coarse_fidelity =
            decode_then_evaluate(&coarse, evaluate, &identity, distortion, tolerance)
                .expect("the coarse artifact evaluates");
        assert!(
            !coarse_fidelity.meets_tolerance(),
            "the decoded coarse artifact passed with distortion {}",
            coarse_fidelity.distortion
        );

        let fine = QuotientCode::encode(&angles, quotient, 20).expect("angles encode");
        let fine_fidelity = decode_then_evaluate(&fine, evaluate, &identity, distortion, tolerance)
            .expect("the fine artifact evaluates");
        assert!(
            fine_fidelity.meets_tolerance(),
            "the decoded fine artifact failed with distortion {}",
            fine_fidelity.distortion
        );
    }

    #[test]
    fn decode_then_evaluate_refuses_an_undeclared_tolerance_or_an_invalid_distortion() {
        let precision = DeclaredPrecision::new(2).expect("precision in range");
        let code = LatticeCode::encode(&[0.25, -1.5], precision).expect("values encode");
        let reference = vec![0.25, -1.5];
        let evaluate = |decoded: &Vec<f64>| Ok::<Vec<f64>, String>(decoded.clone());
        let squared_error = |outputs: &Vec<f64>, native: &Vec<f64>| {
            outputs
                .iter()
                .zip(native)
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f64>()
        };

        // Positive control: both values lie on the lattice, so the decoded artifact is
        // exact and meets a zero tolerance.
        let exact = decode_then_evaluate(&code, evaluate, &reference, squared_error, 0.0)
            .expect("a declared tolerance evaluates");
        assert_eq!(exact.distortion, 0.0);
        assert!(exact.meets_tolerance());

        for tolerance in [f64::NAN, -1.0, f64::INFINITY] {
            assert!(
                decode_then_evaluate(&code, evaluate, &reference, squared_error, tolerance).is_err(),
                "tolerance {tolerance} was admitted"
            );
        }
        let not_a_number =
            |outputs: &Vec<f64>, native: &Vec<f64>| squared_error(outputs, native) + f64::NAN;
        assert!(decode_then_evaluate(&code, evaluate, &reference, not_a_number, 0.0).is_err());
        let negative =
            |outputs: &Vec<f64>, native: &Vec<f64>| squared_error(outputs, native) - 1.0;
        assert!(decode_then_evaluate(&code, evaluate, &reference, negative, 0.0).is_err());
        let refused = |decoded: &Vec<f64>| {
            Err::<Vec<f64>, String>(format!("native execution refused {} values", decoded.len()))
        };
        assert!(decode_then_evaluate(&code, refused, &reference, squared_error, 0.0).is_err());
    }
}
