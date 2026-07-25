//! Analytic infinite-smoothing rail FACE certificate (#2348 Inc 5).
//!
//! The [`asymptote_certificate`](super::asymptote_certificate) module confirms a
//! rail by *measuring* the exponential tail: probe `∂V/∂ρ_j` a few e-folds back
//! from the bound and check that the pencil constant `ĉ = −e^{ρ}∂V/∂ρ` holds
//! still. That is evidence about ONE coordinate along ONE ray, gathered at
//! finite `λ` where the criterion's logdet pair is already cancelling. This
//! module proves the same statement *analytically, at the face itself*, for
//! EVERY direction out of it at once.
//!
//! # The limit is finite, and it is the null-space-restricted fit
//!
//! Write the (profiled-Gaussian) criterion as
//!
//! ```text
//!     V(λ) = (n − M_p)/2 · log D_p + ½ log|H| − ½ log|S_λ|₊ + const,
//!     H = XᵀWX + S_λ,   S_λ = S_F + S_R,   S_F = Σ_{j∈F} λ_j S_j.
//! ```
//!
//! Let `N = ⋂_{j∈F} null(S_j)` be the common null space of the railed
//! penalties, `Z` an orthonormal basis of `N`, and `Q` an orthonormal basis of
//! its orthogonal complement inside the model space — the subspace the face
//! *releases*. In the `[Z, Q]` basis `S_F` lives entirely in the `QQ` block, so
//!
//! ```text
//!     log|H|    = log|S_F^{QQ}| + tr((S_F^{QQ})⁻¹·Schur_Z(K)) + log|ZᵀKZ|   + O(λ⁻²),
//!     log|S_λ|₊ = log|S_F^{QQ}| + tr((S_F^{QQ})⁻¹·Schur_Z(S_R)) + log|ZᵀS_RZ|₊ + O(λ⁻²),
//! ```
//!
//! with `K = XᵀWX + S_R` and `Schur_Z(M) = QᵀMQ − QᵀMZ(ZᵀMZ)⁻ZᵀMQ`. The
//! divergent `log|S_F^{QQ}|` — the `r_F·log λ` that makes each logdet blow up
//! on its own — **cancels exactly between the two**. What is left is finite:
//!
//! ```text
//!     V_∞ = (n − M_p)/2 · log D_p^∞ + ½log|Zᵀ(XᵀWX + S_R)Z| − ½log|ZᵀS_RZ|₊ + const,
//! ```
//!
//! which is precisely the REML criterion of the model restricted to `N` — the
//! λ→∞ limit model, fitted with the surviving penalties. `M_p = dim null(S_λ)`
//! is the same at the face as at any finite `λ > 0`, so no rank bookkeeping
//! moves. **An infinite smoothing parameter is not a numerical accident; it is
//! an ordinary fit of a smaller model, and this is its criterion value.**
//!
//! # The first-order term, and why one matrix decides the whole face
//!
//! Expanding the penalized fit (whose `O(λ⁻¹)` coefficient offset is
//! `S_F⁺g_c`, `g_c` the limit score) and the two traces above, everything of
//! first order collects into ONE symmetric form on the released subspace:
//!
//! ```text
//!     V(λ) = V_∞ + ½ tr( (QᵀS_FQ)⁻¹ C ) + O(λ⁻²),
//!     C = Schur_Z(XᵀWX + S_R) − Schur_Z(S_R) − g_Q g_Qᵀ/φ̂,   g_Q = Qᵀg_c.
//! ```
//!
//! Read it as the empirical-Bayes trade: `Schur_Z(K) − Schur_Z(S_R)` is the
//! *conditional information* the released directions would carry (the Occam
//! cost of un-freezing them), `g_Qg_Qᵀ/φ̂` is the *fit gain* from letting them
//! move. With no surviving penalty this is `I_{Q|Z} − g_Qg_Qᵀ/φ̂`, so `C ≻ 0`
//! is exactly the statement that the score test for releasing the constrained
//! directions falls below one: **no evidence, keep the null.**
//!
//! Three consequences make this a face certificate rather than a coordinate
//! one:
//!
//! 1. `(QᵀS_FQ)⁻¹ ≻ 0` for every positive `λ_F`, so `C ≻ 0` ⟹ `V(λ) > V_∞`
//!    for *all* finite smoothing parameters on the face — no direction, ray or
//!    mixture, needs to be enumerated.
//! 2. The same `C` governs every SUB-face. Releasing only `G ⊆ F` leaves the
//!    model constrained to `N_{F\G}`, whose released part is a subspace of
//!    `span(Q)`; the Schur complement against the *fixed* `Z` block restricts
//!    to it by compression. So one positive-definiteness test covers all
//!    `2^{|F|} − 1` ways to come off the face.
//! 3. Setting `S_F = λ_j S_j` recovers the measured law exactly:
//!    `∂V/∂ρ_j = −c_j e^{−ρ_j}` with `c_j = ½ tr((Q_jᵀS_jQ_j)⁻¹ Q_jᵀCQ_j)`.
//!    The probed pencil constant is a *measurement of this number*, and the
//!    two can be compared.
//!
//! A face coordinate whose own penalty releases nothing once the others are at
//! `λ = ∞` (its released subspace is empty) is **unidentified there**: `V` does
//! not depend on `λ_j` at all, exactly, not merely to first order. Such a
//! coordinate reports `c_j = 0` and is typed [`FaceCoordinateKind::Unidentified`]
//! — it must not be required to carry an outward derivative, and it must not
//! block the face.
//!
//! This module is pure linear algebra on the supplied [`RailFaceLimit`]; the
//! objective that owns the design and the penalties builds that input.

use faer::Side;
use gam_linalg::faer_ndarray::FaerEigh;
use ndarray::{Array1, Array2};

/// Relative eigenvalue threshold separating a subspace's range from its null
/// space. A symmetric eigensolver returns eigenvalues with backward error
/// `O(ε‖A‖)`; anything below `√ε·‖A‖` cannot be distinguished from an exact
/// zero by that spectrum, and everything above it is a genuine direction. This
/// is the same `√ε` split the outer certificate's PSD verdict uses.
fn subspace_split_threshold(spectrum_norm: f64) -> f64 {
    f64::EPSILON.sqrt() * spectrum_norm
}

/// Whether a face coordinate carries a strict outward derivative of its own.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum FaceCoordinateKind {
    /// Releasing this coordinate alone (the rest of the face held at `λ = ∞`)
    /// frees a nonempty subspace, and the criterion strictly increases along
    /// it: `c_j > 0`.
    StrictOutward,
    /// Releasing this coordinate alone frees nothing — the other face
    /// penalties already pin every direction this one would penalize. `V` is
    /// exactly independent of `λ_j` there, so the coordinate is unidentified
    /// at the face and carries no derivative to certify.
    Unidentified,
}

/// The analytic λ→∞ face-limit data an objective supplies for one rail face.
///
/// Every matrix is expressed in an orthonormal basis `Q` of the subspace the
/// face releases (`N^⊥`, `N = ⋂_{j∈F} null(S_j)`); `q` is its dimension.
#[derive(Clone, Debug)]
pub struct RailFaceLimit {
    /// The ρ-coordinates on the face, ascending.
    pub face: Vec<usize>,
    /// `ρ_j` at the certified point, in `face` order. Only used to price the
    /// remaining value gap and estimand travel of the shipped fit; the proof
    /// itself does not depend on it.
    pub face_rho: Vec<f64>,
    /// The symmetric `q×q` first-order form `C`.
    pub first_order_form: Array2<f64>,
    /// `QᵀS_jQ` for each face coordinate, in `face` order.
    pub released_penalties: Vec<Array2<f64>>,
    /// `Qᵀg_c`: the limit score in the released directions.
    pub released_score: Array1<f64>,
    /// Condition number of the `Z`-block solve that formed the Schur
    /// complements. It is the error-amplifying step in building `C`, so the
    /// certificate's curvature margin is scaled by it rather than by a bare
    /// `ε‖C‖`.
    pub form_conditioning: f64,
}

/// A proven rail face: the analytic first-order data behind the mint.
#[derive(Clone, Debug, PartialEq)]
pub struct RailFaceProof {
    /// Smallest eigenvalue of `C`; the proof is `min_curvature > curvature_margin`.
    pub min_curvature: f64,
    /// The numerical-error floor `min_curvature` had to clear:
    /// `q·ε·‖C‖·(1 + cond)`.
    pub curvature_margin: f64,
    /// `‖C‖₂`.
    pub form_norm: f64,
    /// Analytic pencil constants `c_j` in `face` order: `∂V/∂ρ_j → −c_j e^{−ρ_j}`.
    pub tail_constants: Vec<f64>,
    /// Per-coordinate identifiability at the face, in `face` order.
    pub coordinate_kinds: Vec<FaceCoordinateKind>,
    /// The joint pencil constant along the ray that releases the whole face
    /// together with unit weights.
    pub joint_tail_constant: f64,
    /// `V(ρ̂) − V_∞`: the entire criterion improvement still available by
    /// running the face out to `λ = ∞`, priced analytically at the certified ρ.
    pub value_gap: f64,
    /// `‖β(ρ̂) − β(∞)‖`, the exact first-order estimand travel to the limit fit.
    pub estimand_travel: f64,
}

/// Verdict of the analytic face certificate.
#[derive(Clone, Debug, PartialEq)]
pub enum RailFaceVerdict {
    /// `C ≻ 0`: the criterion strictly increases for every finite smoothing
    /// parameter on the face and on every sub-face.
    Certified(RailFaceProof),
    /// The analytic data does not prove the face. Carries the measured
    /// evidence, never a bare flag.
    Refused { reason: String },
}

/// Symmetrize exactly (`(M + Mᵀ)/2` is bitwise symmetric because `f64`
/// addition is commutative), which the strict self-adjoint eigensolver
/// requires.
fn symmetrized(matrix: &Array2<f64>) -> Array2<f64> {
    let n = matrix.nrows();
    let mut out = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            out[[i, j]] = 0.5 * (matrix[[i, j]] + matrix[[j, i]]);
        }
    }
    out
}

/// Eigenvectors of a PSD matrix whose eigenvalues are indistinguishable from
/// zero — an orthonormal basis of its null space. Returns `q×m`.
fn null_space_basis(matrix: &Array2<f64>) -> Result<Array2<f64>, String> {
    let q = matrix.nrows();
    let sym = symmetrized(matrix);
    let (values, vectors) = sym
        .eigh(Side::Lower)
        .map_err(|err| format!("null-space eigendecomposition failed: {err}"))?;
    let norm = values.iter().fold(0.0_f64, |acc, v| acc.max(v.abs()));
    let cut = subspace_split_threshold(norm);
    let keep: Vec<usize> = (0..values.len()).filter(|&i| values[i].abs() <= cut).collect();
    let mut basis = Array2::<f64>::zeros((q, keep.len()));
    for (col, &i) in keep.iter().enumerate() {
        for row in 0..q {
            basis[[row, col]] = vectors[[row, i]];
        }
    }
    Ok(basis)
}

/// Eigenpairs of a PSD matrix above the range/null split: an orthonormal
/// `m×r` basis of its range together with the matching eigenvalues.
fn range_eigenpairs(matrix: &Array2<f64>) -> Result<(Array2<f64>, Array1<f64>), String> {
    let m = matrix.nrows();
    let sym = symmetrized(matrix);
    let (values, vectors) = sym
        .eigh(Side::Lower)
        .map_err(|err| format!("range eigendecomposition failed: {err}"))?;
    let norm = values.iter().fold(0.0_f64, |acc, v| acc.max(v.abs()));
    let cut = subspace_split_threshold(norm);
    let keep: Vec<usize> = (0..values.len()).filter(|&i| values[i] > cut).collect();
    let mut basis = Array2::<f64>::zeros((m, keep.len()));
    let mut kept = Array1::<f64>::zeros(keep.len());
    for (col, &i) in keep.iter().enumerate() {
        kept[col] = values[i];
        for row in 0..m {
            basis[[row, col]] = vectors[[row, i]];
        }
    }
    Ok((basis, kept))
}

/// `½·tr(A⁻¹C)` for a symmetric positive-definite `A`, computed on `A`'s own
/// spectrum so a near-singular `A` reports its failure instead of amplifying
/// round-off through an explicit inverse.
fn half_trace_inverse_product(a: &Array2<f64>, c: &Array2<f64>) -> Result<f64, String> {
    let (basis, values) = range_eigenpairs(a)?;
    if values.len() != a.nrows() {
        return Err(format!(
            "released penalty is singular on the released subspace: rank {} < {}",
            values.len(),
            a.nrows()
        ));
    }
    // tr(A⁻¹C) = Σ_a (uₐᵀ C uₐ)/σₐ over A's eigenpairs.
    let mut total = 0.0_f64;
    for (col, &sigma) in values.iter().enumerate() {
        let u = basis.column(col);
        let cu = c.dot(&u);
        total += u.dot(&cu) / sigma;
    }
    Ok(0.5 * total)
}

/// Prove (or refuse) a rail face from its analytic λ→∞ limit data.
///
/// The proof is a single positive-definiteness test on the first-order form
/// `C`; see the module derivation. Refusals carry the measured margin so a
/// declined face explains itself.
pub fn certify_rail_face(limit: &RailFaceLimit) -> RailFaceVerdict {
    let refuse = |reason: String| RailFaceVerdict::Refused { reason };
    let q = limit.first_order_form.nrows();
    if limit.face.is_empty() {
        return refuse("empty rail face".to_string());
    }
    if q == 0 {
        return refuse("the face releases no direction".to_string());
    }
    if limit.first_order_form.ncols() != q
        || limit.released_score.len() != q
        || limit.released_penalties.len() != limit.face.len()
        || limit.face_rho.len() != limit.face.len()
        || limit
            .released_penalties
            .iter()
            .any(|a| a.nrows() != q || a.ncols() != q)
    {
        return refuse("face limit data has inconsistent dimensions".to_string());
    }
    if limit.first_order_form.iter().any(|v| !v.is_finite())
        || limit.released_score.iter().any(|v| !v.is_finite())
        || limit
            .released_penalties
            .iter()
            .any(|a| a.iter().any(|v| !v.is_finite()))
        || !limit.form_conditioning.is_finite()
        || limit.form_conditioning < 1.0
    {
        return refuse("face limit data is not finite".to_string());
    }

    let form = symmetrized(&limit.first_order_form);
    let (values, _) = match form.eigh(Side::Lower) {
        Ok(pair) => pair,
        Err(err) => return refuse(format!("first-order form eigendecomposition failed: {err}")),
    };
    let min_curvature = values.iter().fold(f64::INFINITY, |acc, v| acc.min(*v));
    let form_norm = values.iter().fold(0.0_f64, |acc, v| acc.max(v.abs()));
    // The eigenvalue backward error is O(q·ε‖C‖); forming C ran the released
    // directions through the Z-block solve, which amplifies by its condition
    // number. Below this the sign of `min_curvature` is not a measured fact.
    let curvature_margin =
        (q as f64) * f64::EPSILON * form_norm * (1.0 + limit.form_conditioning);
    if !(min_curvature > curvature_margin) {
        return refuse(format!(
            "face first-order form is not positive definite: λ_min(C)={min_curvature:.6e} \
             ≤ margin {curvature_margin:.3e} (‖C‖={form_norm:.3e}, cond={:.3e}) — releasing \
             the face would not raise the criterion, so λ=∞ is not proven optimal",
            limit.form_conditioning
        ));
    }

    // Per-coordinate law: hold the rest of the face at λ=∞ and release only j.
    // The directions it frees are the range of its own penalty inside the null
    // space of the others; on that subspace the ordinary one-coordinate tail
    // law holds with `c_j = ½tr((Q_jᵀS_jQ_j)⁻¹ Q_jᵀCQ_j)`.
    let mut tail_constants = Vec::with_capacity(limit.face.len());
    let mut coordinate_kinds = Vec::with_capacity(limit.face.len());
    for idx in 0..limit.face.len() {
        let mut rest = Array2::<f64>::zeros((q, q));
        for (other, penalty) in limit.released_penalties.iter().enumerate() {
            if other != idx {
                rest += penalty;
            }
        }
        let free = match null_space_basis(&rest) {
            Ok(basis) => basis,
            Err(err) => return refuse(err),
        };
        if free.ncols() == 0 {
            tail_constants.push(0.0);
            coordinate_kinds.push(FaceCoordinateKind::Unidentified);
            continue;
        }
        let own = free.t().dot(&limit.released_penalties[idx]).dot(&free);
        let (own_basis, own_values) = match range_eigenpairs(&own) {
            Ok(pair) => pair,
            Err(err) => return refuse(err),
        };
        if own_values.is_empty() {
            tail_constants.push(0.0);
            coordinate_kinds.push(FaceCoordinateKind::Unidentified);
            continue;
        }
        let released = free.dot(&own_basis);
        let compressed = released.t().dot(&form).dot(&released);
        let mut c_j = 0.0_f64;
        for (a, &sigma) in own_values.iter().enumerate() {
            c_j += compressed[[a, a]] / sigma;
        }
        c_j *= 0.5;
        if !(c_j > 0.0) {
            // `C ≻ 0` makes every compression positive definite, so this can
            // only be reached by a non-finite or numerically collapsed
            // released block — evidence the geometry is not resolvable here.
            return refuse(format!(
                "face coordinate {} has a non-positive analytic pencil constant {c_j:.6e} \
                 against a positive-definite face form — released geometry is degenerate",
                limit.face[idx]
            ));
        }
        tail_constants.push(c_j);
        coordinate_kinds.push(FaceCoordinateKind::StrictOutward);
    }

    // The joint law: release the whole face together with unit weights.
    let mut unit_face = Array2::<f64>::zeros((q, q));
    for penalty in limit.released_penalties.iter() {
        unit_face += penalty;
    }
    let joint_tail_constant = match half_trace_inverse_product(&unit_face, &form) {
        Ok(value) => value,
        Err(err) => return refuse(format!("joint face law unavailable: {err}")),
    };

    // Price the shipped point: `V(ρ̂) − V_∞ = ½tr((Σ_j λ_j QᵀS_jQ)⁻¹C)` and the
    // matching coefficient offset `(Σ_j λ_j QᵀS_jQ)⁻¹Qᵀg_c`.
    let mut certified_face = Array2::<f64>::zeros((q, q));
    for (penalty, &rho) in limit.released_penalties.iter().zip(limit.face_rho.iter()) {
        let lambda = rho.exp();
        if !lambda.is_finite() || lambda <= 0.0 {
            return refuse(format!(
                "face coordinate ρ={rho:.3e} does not exponentiate to a usable λ"
            ));
        }
        certified_face
            .iter_mut()
            .zip(penalty.iter())
            .for_each(|(dst, src)| *dst += lambda * src);
    }
    let value_gap = match half_trace_inverse_product(&certified_face, &form) {
        Ok(value) => value,
        Err(err) => return refuse(format!("value gap at the certified point unavailable: {err}")),
    };
    let estimand_travel = match range_eigenpairs(&certified_face) {
        Ok((basis, values)) if values.len() == q => {
            let mut offset = Array1::<f64>::zeros(q);
            for (col, &sigma) in values.iter().enumerate() {
                let u = basis.column(col);
                let coeff = u.dot(&limit.released_score) / sigma;
                for row in 0..q {
                    offset[row] += coeff * u[row];
                }
            }
            offset.dot(&offset).sqrt()
        }
        Ok(_) => {
            return refuse(
                "the certified-point face penalty is singular on the released subspace"
                    .to_string(),
            );
        }
        Err(err) => return refuse(err),
    };
    if !value_gap.is_finite() || !estimand_travel.is_finite() || value_gap < 0.0 {
        return refuse(format!(
            "face pricing is not usable: value_gap={value_gap:.3e} travel={estimand_travel:.3e}"
        ));
    }

    RailFaceVerdict::Certified(RailFaceProof {
        min_curvature,
        curvature_margin,
        form_norm,
        tail_constants,
        coordinate_kinds,
        joint_tail_constant,
        value_gap,
        estimand_travel,
    })
}

#[cfg(test)]
mod rail_face_tests {
    use super::*;

    fn limit(
        face: Vec<usize>,
        face_rho: Vec<f64>,
        form: Array2<f64>,
        penalties: Vec<Array2<f64>>,
        score: Array1<f64>,
    ) -> RailFaceLimit {
        RailFaceLimit {
            face,
            face_rho,
            first_order_form: form,
            released_penalties: penalties,
            released_score: score,
            form_conditioning: 1.0,
        }
    }

    fn diag(values: &[f64]) -> Array2<f64> {
        let n = values.len();
        let mut m = Array2::<f64>::zeros((n, n));
        for (i, &v) in values.iter().enumerate() {
            m[[i, i]] = v;
        }
        m
    }

    /// One railed coordinate, diagonal geometry: the analytic pencil constant
    /// is `½·Σ C_aa/σ_a` in closed form, the value gap is `c·e^{−ρ}` (the tail
    /// law's own integral), and the estimand travel is `‖(λA)⁻¹g‖`.
    #[test]
    fn single_coordinate_face_matches_the_closed_form_tail_law() {
        let rho = 30.0_f64;
        let lim = limit(
            vec![1],
            vec![rho],
            diag(&[4.0, 6.0]),
            vec![diag(&[2.0, 3.0])],
            Array1::from(vec![1.0, 2.0]),
        );
        match certify_rail_face(&lim) {
            RailFaceVerdict::Certified(proof) => {
                let expected = 0.5 * (4.0 / 2.0 + 6.0 / 3.0);
                assert!(
                    (proof.tail_constants[0] - expected).abs() <= 1.0e-12 * expected,
                    "analytic c_j={} should equal the closed form {expected}",
                    proof.tail_constants[0]
                );
                assert_eq!(proof.coordinate_kinds[0], FaceCoordinateKind::StrictOutward);
                assert!(
                    (proof.joint_tail_constant - expected).abs() <= 1.0e-12 * expected,
                    "a one-coordinate face's joint law is its own law"
                );
                // V(ρ̂) − V_∞ = ½tr((λA)⁻¹C) = c·e^{−ρ}: the tail law's exact
                // remaining value gap.
                let expected_gap = expected * (-rho).exp();
                assert!(
                    (proof.value_gap - expected_gap).abs() <= 1.0e-12 * expected_gap,
                    "value gap {} should be c·e^(−ρ) = {expected_gap}",
                    proof.value_gap
                );
                // ‖(λA)⁻¹g‖ with λ = e^ρ.
                let lambda = rho.exp();
                let expected_travel =
                    ((1.0 / (lambda * 2.0)).powi(2) + (2.0 / (lambda * 3.0)).powi(2)).sqrt();
                assert!(
                    (proof.estimand_travel - expected_travel).abs()
                        <= 1.0e-9 * expected_travel.max(f64::MIN_POSITIVE),
                    "estimand travel {} should be ‖(λA)⁻¹g‖ = {expected_travel}",
                    proof.estimand_travel
                );
            }
            other => panic!("a positive-definite face form must certify, got {other:?}"),
        }
    }

    /// The proof is a strict positive-definiteness test. A form with a negative
    /// direction means releasing the face LOWERS the criterion there — the rail
    /// is not the optimum — and must refuse, with the measured eigenvalue.
    #[test]
    fn indefinite_first_order_form_refuses_with_its_margin() {
        let lim = limit(
            vec![0],
            vec![25.0],
            diag(&[4.0, -1.0e-3]),
            vec![diag(&[1.0, 1.0])],
            Array1::from(vec![0.0, 0.0]),
        );
        match certify_rail_face(&lim) {
            RailFaceVerdict::Refused { reason } => {
                assert!(
                    reason.contains("not positive definite"),
                    "refusal should name the failed gate: {reason}"
                );
                assert!(
                    reason.contains("λ_min(C)") && reason.contains("margin"),
                    "refusal should carry the measured λ_min against its margin: {reason}"
                );
            }
            other => panic!("an indefinite face form must not certify, got {other:?}"),
        }
    }

    /// A flat direction is not a proof. `λ_min(C) = 0` leaves the criterion
    /// unchanged to first order along that release, so the face is undetermined
    /// and the certificate refuses rather than minting a maybe-optimum.
    #[test]
    fn semidefinite_form_is_not_a_proof() {
        let lim = limit(
            vec![0],
            vec![25.0],
            diag(&[4.0, 0.0]),
            vec![diag(&[1.0, 1.0])],
            Array1::from(vec![0.0, 0.0]),
        );
        assert!(
            matches!(certify_rail_face(&lim), RailFaceVerdict::Refused { .. }),
            "a face with an exactly flat released direction must refuse"
        );
    }

    /// Multi-coordinate face with OVERLAPPING penalties. Coordinate 0 penalizes
    /// only the first released direction; coordinate 1 penalizes both. With
    /// coordinate 1 at λ=∞ every direction is already pinned, so releasing
    /// coordinate 0 alone changes the model not at all: it is unidentified at
    /// the face and reports `c_0 = 0` — while the face itself still certifies
    /// and coordinate 1 keeps its own strict law on the direction that only it
    /// pins.
    #[test]
    fn coalesced_face_coordinate_is_unidentified_not_a_refusal() {
        let lim = limit(
            vec![2, 3],
            vec![30.0, 30.0],
            diag(&[4.0, 6.0]),
            vec![diag(&[1.0, 0.0]), diag(&[1.0, 1.0])],
            Array1::from(vec![0.0, 0.0]),
        );
        match certify_rail_face(&lim) {
            RailFaceVerdict::Certified(proof) => {
                assert_eq!(proof.coordinate_kinds[0], FaceCoordinateKind::Unidentified);
                assert_eq!(proof.tail_constants[0], 0.0);
                assert_eq!(proof.coordinate_kinds[1], FaceCoordinateKind::StrictOutward);
                // Releasing coordinate 1 alone frees the direction coordinate 0
                // does not penalize: c_1 = ½·C_11/σ_11 = ½·6/1.
                assert!(
                    (proof.tail_constants[1] - 3.0).abs() <= 1.0e-12,
                    "c_1={} should be ½·C₁₁/σ₁₁ = 3",
                    proof.tail_constants[1]
                );
                // The joint ray releases both with unit weights: A = diag(2,1).
                let expected_joint = 0.5 * (4.0 / 2.0 + 6.0 / 1.0);
                assert!(
                    (proof.joint_tail_constant - expected_joint).abs() <= 1.0e-12 * expected_joint,
                    "joint constant {} should be {expected_joint}",
                    proof.joint_tail_constant
                );
            }
            other => panic!("a coalesced but positive-definite face must certify, got {other:?}"),
        }
    }

    /// The face proof covers every SUB-face, not just the coordinate axes and
    /// the joint ray: for any positive weights the released-penalty sum stays
    /// positive definite on the released subspace, so the predicted gap
    /// `½tr((Σλ_jA_j)⁻¹C)` is positive. Check that against the certified form
    /// on a deliberately lopsided, non-diagonal geometry.
    #[test]
    fn certified_face_raises_the_criterion_for_every_positive_weighting() {
        let mut form = Array2::<f64>::zeros((2, 2));
        form[[0, 0]] = 3.0;
        form[[0, 1]] = 1.0;
        form[[1, 0]] = 1.0;
        form[[1, 1]] = 2.0;
        let mut a1 = Array2::<f64>::zeros((2, 2));
        a1[[0, 0]] = 2.0;
        a1[[0, 1]] = 0.5;
        a1[[1, 0]] = 0.5;
        a1[[1, 1]] = 1.0;
        let lim = limit(
            vec![0, 1],
            vec![20.0, 22.0],
            form.clone(),
            vec![diag(&[1.0, 0.25]), a1],
            Array1::from(vec![0.3, -0.7]),
        );
        let proof = match certify_rail_face(&lim) {
            RailFaceVerdict::Certified(proof) => proof,
            other => panic!("expected a certified face, got {other:?}"),
        };
        assert!(proof.min_curvature > 0.0);
        for (w0, w1) in [(1.0, 1.0e-6), (1.0e-6, 1.0), (1.0, 1.0), (17.0, 0.03)] {
            let mut mixed = Array2::<f64>::zeros((2, 2));
            mixed
                .iter_mut()
                .zip(lim.released_penalties[0].iter())
                .for_each(|(dst, src)| *dst += w0 * src);
            mixed
                .iter_mut()
                .zip(lim.released_penalties[1].iter())
                .for_each(|(dst, src)| *dst += w1 * src);
            let gap = half_trace_inverse_product(&mixed, &form)
                .expect("a positive weighting stays invertible on the released subspace");
            assert!(
                gap > 0.0,
                "releasing the face with weights ({w0}, {w1}) must raise the criterion, got {gap}"
            );
        }
    }

    /// Structural refusals never panic and never certify: an empty face, a
    /// dimension mismatch, and a non-finite form all decline with a reason.
    #[test]
    fn malformed_face_data_declines() {
        let empty = limit(
            Vec::new(),
            Vec::new(),
            diag(&[1.0]),
            Vec::new(),
            Array1::from(vec![0.0]),
        );
        assert!(matches!(
            certify_rail_face(&empty),
            RailFaceVerdict::Refused { .. }
        ));

        let mismatched = limit(
            vec![0],
            vec![30.0],
            diag(&[1.0, 1.0]),
            vec![diag(&[1.0])],
            Array1::from(vec![0.0, 0.0]),
        );
        assert!(matches!(
            certify_rail_face(&mismatched),
            RailFaceVerdict::Refused { .. }
        ));

        let non_finite = limit(
            vec![0],
            vec![30.0],
            diag(&[f64::NAN, 1.0]),
            vec![diag(&[1.0, 1.0])],
            Array1::from(vec![0.0, 0.0]),
        );
        assert!(matches!(
            certify_rail_face(&non_finite),
            RailFaceVerdict::Refused { .. }
        ));
    }
}
