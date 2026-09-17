#![cfg(test)]
//! A13 (#2951): planted-rotation teacher controls, scored at bounded code.
//!
//! Take `p` states `x_i ∈ ℝ^d` and the shift `i → i + 1 mod p`. When `p ≤ d` and the
//! states are linearly independent, as generic states are, `X₊ X⁺` realizes the shift
//! exactly whether or not a rotation was planted. So held-out fidelity alone cannot reject
//! a random connection (mpd-modadd, #2951 comment 5716898972).
//!
//! What separates a planted rotation is code length at equal decoded fidelity (P11, P18). A
//! plane rotation sends one plane and one angle on `ℝ/2πℤ`; a generic operator sends `d²`
//! reals. Lengths are compared only between artifacts whose decoded distortion meets the
//! declared tolerance ([`code_saving_at_declared_fidelity`]).
//!
//! The declared inputs are the lattice precision, the angle resolution and the fidelity
//! tolerance. The regime `p > d`, where random states admit no linear realizer, needs a
//! derived lower bound on the least-squares residual and lands separately.

use super::codec::{BitString, DecodedArtifactScore, code_saving_at_declared_fidelity};
use super::precision::{
    DecodableArtifact, DeclaredPrecision, LatticeCode, PeriodicQuotient, QuotientCode,
    decode_then_evaluate,
};
use gam_linalg::faer_ndarray::{FaerArrayView, col_piv_qr_solve_lstsq};
use gam_linalg::roundoff::UNIT_ROUNDOFF;
use ndarray::Array2;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use std::f64::consts::TAU;

const WIDTH: usize = 8;
const STATES: usize = 6;

/// A plane rotation sent as its plane basis (`d × 2` reals on the declared lattice, row by
/// row) and its angle on `ℝ/2πℤ`.
struct PlaneRotationArtifact {
    basis: LatticeCode,
    angle: QuotientCode,
}

impl PlaneRotationArtifact {
    /// The written length of both messages.
    fn code_bits(&self) -> u64 {
        let mut message = BitString::new();
        self.basis.write(&mut message).expect("the plane basis writes");
        self.angle.write(&mut message).expect("the angle writes");
        message.len_bits()
    }
}

impl DecodableArtifact for PlaneRotationArtifact {
    type Decoded = Array2<f64>;

    /// `W = I + U (R(α) − I) Uᵀ`, rebuilt from the decoded plane and angle only.
    fn decode(&self) -> Result<Array2<f64>, String> {
        let basis = self.basis.decode()?;
        let angle = self.angle.decode()?;
        if basis.len() != 2 * WIDTH || angle.len() != 1 {
            return Err(format!(
                "a plane rotation decodes {} basis reals and {} angles, expected {} and 1",
                basis.len(),
                angle.len(),
                2 * WIDTH
            ));
        }
        let (sine, cosine) = angle[0].sin_cos();
        let block = [[cosine - 1.0, -sine], [sine, cosine - 1.0]];
        Ok(Array2::from_shape_fn((WIDTH, WIDTH), |(row, column)| {
            let mut entry = if row == column { 1.0 } else { 0.0 };
            for (a, block_row) in block.iter().enumerate() {
                for (b, value) in block_row.iter().enumerate() {
                    entry += basis[2 * row + a] * value * basis[2 * column + b];
                }
            }
            entry
        }))
    }
}

fn lattice_bits(code: &LatticeCode) -> u64 {
    let mut message = BitString::new();
    code.write(&mut message).expect("the lattice code writes");
    message.len_bits()
}

fn shifted_rows(states: &Array2<f64>) -> Array2<f64> {
    Array2::from_shape_fn(states.dim(), |(row, column)| {
        states[[(row + 1) % states.nrows(), column]]
    })
}

/// `x_i = (cos θ_i, sin θ_i, ½, …, ½)` with `θ_i = 2πi/p`, and the shifted rows. The shift is
/// a row permutation of the same floats, so the reference is exact.
fn planted_cycle_states() -> (Array2<f64>, Array2<f64>) {
    let states = Array2::from_shape_fn((STATES, WIDTH), |(row, column)| {
        let (sine, cosine) = (TAU * row as f64 / STATES as f64).sin_cos();
        match column {
            0 => cosine,
            1 => sine,
            _ => 0.5,
        }
    });
    let shifted = shifted_rows(&states);
    (states, shifted)
}

fn random_cycle_states(seed: u64) -> (Array2<f64>, Array2<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let states = Array2::from_shape_simple_fn((STATES, WIDTH), || rng.random_range(-1.0..1.0));
    let shifted = shifted_rows(&states);
    (states, shifted)
}

/// A realizer `T` of the shift with `X Tᵀ = Y`. It solves the square system
/// `(X Xᵀ) Z = Y` by the rank-aware QR owner, so `T = Zᵀ X` and `X Tᵀ = X Xᵀ Z = Y`.
fn least_squares_realizer(states: &Array2<f64>, shifted: &Array2<f64>) -> Array2<f64> {
    let gram = states.dot(&states.t());
    let solution = col_piv_qr_solve_lstsq(
        FaerArrayView::new(&gram).as_ref(),
        FaerArrayView::new(shifted).as_ref(),
    );
    let coefficients = Array2::from_shape_fn((STATES, WIDTH), |(row, column)| {
        solution[(row, column)]
    });
    coefficients.t().dot(states)
}

/// Decodes the artifact, applies the decoded operator to every state, and measures the largest
/// `|image − reference|` entry. Each entry is one rounded subtraction of two floats, so the
/// exact distortion exceeds the measured one by at most `u·distortion`, rounded up.
fn score<A: DecodableArtifact>(
    artifact: &A,
    code_bits: u64,
    operator: impl FnOnce(&A::Decoded) -> Array2<f64>,
    states: &Array2<f64>,
    reference: &Array2<f64>,
    tolerance: f64,
) -> DecodedArtifactScore {
    let fidelity = decode_then_evaluate(
        artifact,
        |decoded| Ok(states.dot(&operator(decoded).t())),
        reference,
        |images, native| {
            images
                .iter()
                .zip(native.iter())
                .fold(0.0_f64, |largest, (image, target)| largest.max((image - target).abs()))
        },
        tolerance,
    )
    .expect("the artifact decodes and evaluates");
    DecodedArtifactScore {
        code_bits,
        decoded_distortion: fidelity.distortion,
        distortion_roundoff: (UNIT_ROUNDOFF * fidelity.distortion).next_up(),
    }
}

/// On planted cycle states, a plane rotation and its dense generic realizer both meet the
/// declared tolerance, and the rotation's code is shorter. On random states the generic
/// realizer also meets the tolerance, so fidelity alone accepts a random connection, which is
/// the premise of correction C3. The planted rotation's artifact does not realize that
/// connection. Positive control: the code comparison refuses that pair, because lengths at
/// different fidelities are not a model comparison.
#[test]
fn a_planted_rotation_wins_on_code_where_fidelity_alone_accepts_a_random_connection() {
    let precision = DeclaredPrecision::new(20).expect("a declared dyadic precision");
    let angle_resolution = 20;
    let tolerance = 2.0_f64.powi(-10);
    let quotient = PeriodicQuotient::new(TAU).expect("the angle quotient");
    let as_operator = |values: &Vec<f64>| {
        Array2::from_shape_vec((WIDTH, WIDTH), values.clone()).expect("d × d reals")
    };

    let mut plane = vec![0.0; 2 * WIDTH];
    plane[0] = 1.0;
    plane[3] = 1.0;
    let rotation = PlaneRotationArtifact {
        basis: LatticeCode::encode(&plane, precision).expect("the plane basis encodes"),
        angle: QuotientCode::encode(&[TAU / STATES as f64], quotient, angle_resolution)
            .expect("the angle encodes"),
    };
    let rotation_bits = rotation.code_bits();
    let (sine, cosine) = (TAU / STATES as f64).sin_cos();
    let mut native = Array2::<f64>::eye(WIDTH);
    native[[0, 0]] = cosine;
    native[[0, 1]] = -sine;
    native[[1, 0]] = sine;
    native[[1, 1]] = cosine;
    let generic = LatticeCode::encode(native.as_slice().expect("standard layout"), precision)
        .expect("the dense operator encodes");

    let (planted, planted_shifted) = planted_cycle_states();
    let planted_rotation = score(
        &rotation,
        rotation_bits,
        |operator| operator.clone(),
        &planted,
        &planted_shifted,
        tolerance,
    );
    let planted_generic = score(
        &generic,
        lattice_bits(&generic),
        as_operator,
        &planted,
        &planted_shifted,
        tolerance,
    );
    let saving = code_saving_at_declared_fidelity(tolerance, &planted_generic, &planted_rotation)
        .expect("both planted artifacts meet the declared tolerance");
    assert!(
        saving > 0,
        "the plane rotation must be shorter: {planted_rotation:?} against {planted_generic:?}"
    );

    let (random, random_shifted) = random_cycle_states(2951);
    let realizer = least_squares_realizer(&random, &random_shifted);
    let random_code = LatticeCode::encode(realizer.as_slice().expect("standard layout"), precision)
        .expect("the realizer encodes");
    let random_generic = score(
        &random_code,
        lattice_bits(&random_code),
        as_operator,
        &random,
        &random_shifted,
        tolerance,
    );
    assert!(
        random_generic.decoded_distortion + random_generic.distortion_roundoff <= tolerance,
        "a random connection's generic realizer must meet the tolerance: {random_generic:?}"
    );

    let random_rotation = score(
        &rotation,
        rotation_bits,
        |operator| operator.clone(),
        &random,
        &random_shifted,
        tolerance,
    );
    assert!(
        code_saving_at_declared_fidelity(tolerance, &random_generic, &random_rotation).is_err(),
        "the planted rotation does not realize a random connection, so no code comparison stands: \
         {random_rotation:?}"
    );
}
