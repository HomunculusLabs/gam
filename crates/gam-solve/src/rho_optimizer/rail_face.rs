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
//! # The LAML extension: families whose working weights move with `β̂`
//!
//! For a fixed-unit-dispersion LAML criterion (binomial, poisson)
//!
//! ```text
//!     V(λ) = −ℓ(β̂) + ½ β̂ᵀS_λβ̂ + ½ log|H| − ½ log|S_λ|₊,   H = I(β̂) + S_λ,
//!     I(β) = Xᵀ W(η) X,   W_i = −∂²ℓ_i/∂η_i²   (the criterion's observed weights),
//! ```
//!
//! the same expansion goes through with `K = I(β̂_∞) + S_R`, plus exactly one
//! new first-order term: the Laplace logdet's curvature moves with `β̂`, and
//! the face's `O(λ⁻¹)` coefficient offset `δβ = S_F⁺g_c` carries that motion
//! into first order,
//!
//! ```text
//!     Δ(½log|H|) = Σ_r [∂(½log|H_∞|)/∂β_r]·δβ_r = dᵀ δβ,
//!     d = ½ Xᵀ( c ⊙ a ),   c_i = dW_i/dη_i,   a_i = x_iᵀ Z (ZᵀKZ)⁻¹ Zᵀ x_i,
//! ```
//!
//! `a` the LIMIT model's leverage — no large-λ object anywhere. One care is
//! needed with `δβ = (K + S_F)⁻¹ g_c`: its PINNED component is `O(λ⁻¹)` too,
//! `δ_Z = −(ZᵀKZ)⁻¹ZᵀKQ·(QᵀS_FQ)⁻¹g_Q` (the block inverse's off-diagonal),
//! so `dᵀδβ` reads `d` through the K-OBLIQUE reduction onto the released
//! subspace — the same reduction the Schur complement applies to `K` — and
//! not the orthogonal compression:
//!
//! ```text
//!     d̃ = d − K Z (ZᵀKZ)⁻¹ Zᵀ d,     dᵀδβ = d̃_Qᵀ (QᵀS_FQ)⁻¹ g_Q.
//! ```
//!
//! (The Gaussian form never meets this subtlety: its only score-linear term
//! carries `g_c`, and `Zᵀg_c = 0` kills the oblique part exactly. Dropping
//! the reduction here is a FIRST-order error — measured at 1.7e-2 relative
//! against the production LAML gradient before it was applied.) In trace
//! form `dᵀδβ = ½tr((QᵀS_FQ)⁻¹(g_Q d̃_Qᵀ + d̃_Q g_Qᵀ))`, so the whole
//! extension is a symmetric rank-2 correction to the form:
//!
//! ```text
//!     C_LAML = Schur_Z(I(β̂_∞)+S_R) − Schur_Z(S_R) − g_Q g_Qᵀ + (g_Q d̃_Qᵀ + d̃_Q g_Qᵀ).
//! ```
//!
//! (The fit term's `−g_Qg_Qᵀ` carries no dispersion divisor: for these
//! families `−ℓ` and `H` are already in the same units, which is also why the
//! profiled-Gaussian form divides by `φ̂` — there the fit term is
//! `(n−M_p)/2·log D_p` and its first variation is `ΔD_p/(2φ̂)`.) The two
//! penalty logdets have no `β̂`-dependence, so their Schur content is
//! unchanged, and everything downstream — the `C ≻ 0` proof, per-coordinate
//! `c_j`, the `Unidentified` typing, the value-domain falsification — is
//! family-blind because none of it depends on how `C` was built. Gaussian
//! identity is the `c ≡ 0` member: the rank-2 term vanishes identically and
//! the form reduces to the REML one, which is the built-in exactness check.
//!
//! This module is pure linear algebra on the supplied [`RailFaceLimit`]; the
//! objective that owns the design and the penalties builds that input.

use faer::Side;
use gam_linalg::faer_ndarray::FaerEigh;
use gam_terms::construction::CanonicalPenalty;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};

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
    /// The λ=∞ fit itself: coefficients of the null-space-restricted model, in
    /// the model's own coefficient basis. This is the limit the certificate is
    /// about — the fit a face-certified optimum reports — and the same object a
    /// continuation anchored at maximal smoothing starts from.
    pub limit_beta: Array1<f64>,
    /// Profiled dispersion at the limit fit, the `φ̂` the first-order form's
    /// fit term is divided by.
    pub limit_dispersion: f64,
    /// `Qᵀd`: the LAML curvature-drift vector in the released directions,
    /// recorded so the proof object shows how much of `C` is the rank-2
    /// `g_Q d_Qᵀ + d_Q g_Qᵀ` correction. `None` on the profiled-Gaussian
    /// form, whose working weights do not move with the coefficients.
    pub released_curvature_drift: Option<Array1<f64>>,
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

/// Rank of a PSD block above the `√ε` range/null split — the same split every
/// other subspace decision in this module uses, exported so callers building
/// a reduced (limit-model) penalty layout count exactly as the form does.
pub(crate) fn released_rank(matrix: &Array2<f64>) -> Result<usize, String> {
    let (_, values) = range_eigenpairs(matrix)?;
    Ok(values.len())
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


/// Self-adjoint eigendecomposition of an exactly-symmetrized copy.
fn symmetric_eigh(matrix: &Array2<f64>) -> Option<(Array1<f64>, Array2<f64>)> {
    symmetrized(matrix).eigh(Side::Lower).ok()
}

/// [`subspace_split_threshold`] applied to an already-computed spectrum.
fn range_null_split(values: &Array1<f64>) -> f64 {
    subspace_split_threshold(values.iter().fold(0.0_f64, |acc, v| acc.max(v.abs())))
}

/// Gather selected eigenvectors into a `p×m` orthonormal basis.
fn basis_columns(vectors: &Array2<f64>, cols: &[usize]) -> Array2<f64> {
    let rows = vectors.nrows();
    let mut basis = Array2::<f64>::zeros((rows, cols.len()));
    for (out, &src) in cols.iter().enumerate() {
        for row in 0..rows {
            basis[[row, out]] = vectors[[row, src]];
        }
    }
    basis
}

/// Moore–Penrose inverse rebuilt from an already-computed spectrum, dropping
/// everything at or below `cut`.
fn spectral_pseudo_inverse(values: &Array1<f64>, vectors: &Array2<f64>, cut: f64) -> Array2<f64> {
    let n = values.len();
    let mut out = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        if values[i] > cut {
            let inv = 1.0 / values[i];
            let v = vectors.column(i);
            for a in 0..n {
                let va = inv * v[a];
                for b in 0..n {
                    out[[a, b]] += va * v[b];
                }
            }
        }
    }
    out
}

/// Why a rail-face limit is or is not available.
///
/// This is deliberately not an `Option`. There is more than one closed form for
/// the criterion (the profiled-Gaussian one here; a LAML one for families whose
/// working weights move with `β̂`), and a bare `None` would collapse two
/// statements that call for opposite responses: *"this criterion is outside the
/// closed form you asked"* — where a DIFFERENT form may still apply — and
/// *"the closed form applies but this face is unusable"*, which is a statement
/// about the face itself and which no other form will rescue.
#[derive(Clone, Debug)]
pub enum RailFaceLimitOutcome {
    /// The limit was formed.
    Available(Box<RailFaceLimit>),
    /// The criterion is outside the closed form that was asked for. Says
    /// nothing about the face; another closed form may still apply.
    OutsideClosedForm {
        /// Which clause of the form's scope failed.
        reason: String,
    },
    /// The closed form applies, but the face cannot be used: it releases no
    /// direction, the limit model is not identified, or a rank bookkeeping
    /// check did not hold. No other closed form changes this.
    FaceUnavailable {
        /// The measured evidence behind the refusal.
        reason: String,
    },
}

impl RailFaceLimitOutcome {
    /// The limit, when one was formed.
    pub fn available(self) -> Option<RailFaceLimit> {
        match self {
            Self::Available(limit) => Some(*limit),
            _ => None,
        }
    }

    /// A human-readable account of a decline, for the certificate's log.
    pub fn decline_reason(&self) -> Option<&str> {
        match self {
            Self::Available(_) => None,
            Self::OutsideClosedForm { reason } | Self::FaceUnavailable { reason } => {
                Some(reason.as_str())
            }
        }
    }
}

/// The per-face penalty geometry shared by every closed form: the face's
/// penalties at unit strength, the survivors at their certified strengths,
/// and the all-unit sum whose spectrum backs the rank bookkeeping. Factored
/// out so the Gaussian and LAML forms can never disagree about what the face
/// IS.
pub(crate) struct FacePenaltySplit {
    /// The face, ascending and deduplicated.
    pub(crate) face_sorted: Vec<usize>,
    /// `Σ_{j∈F} S_j` at unit strength.
    pub(crate) s_face_unit: Array2<f64>,
    /// `Σ_{j∉F} λ_j S_j` at the certified survivor strengths.
    pub(crate) s_rest: Array2<f64>,
    /// `Σ_j S_j` at unit strength, for the pseudo-determinant rank split.
    pub(crate) s_unit_all: Array2<f64>,
    /// Full-width unit `S_j` for each face coordinate, in `face_sorted` order.
    pub(crate) face_penalties: Vec<Array2<f64>>,
}

/// Split the canonical penalties around a face. `Err` carries the typed
/// decline the caller returns verbatim.
pub(crate) fn split_face_penalties(
    penalties: &[CanonicalPenalty],
    rho: &Array1<f64>,
    face: &[usize],
    p: usize,
) -> Result<FacePenaltySplit, RailFaceLimitOutcome> {
    let n_penalties = penalties.len();
    if p == 0 || n_penalties == 0 || face.is_empty() || rho.len() < n_penalties {
        return Err(RailFaceLimitOutcome::OutsideClosedForm {
            reason: "face-limit inputs are shape-inconsistent or empty".to_string(),
        });
    }
    let mut face_sorted = face.to_vec();
    face_sorted.sort_unstable();
    face_sorted.dedup();
    if face_sorted.len() != face.len()
        || face_sorted.last().copied().unwrap_or(usize::MAX) >= n_penalties
    {
        return Err(RailFaceLimitOutcome::FaceUnavailable {
            reason: "the face repeats a coordinate or indexes outside the penalty layout"
                .to_string(),
        });
    }

    let mut s_face_unit = Array2::<f64>::zeros((p, p));
    let mut s_rest = Array2::<f64>::zeros((p, p));
    let mut s_unit_all = Array2::<f64>::zeros((p, p));
    let mut face_penalties: Vec<Array2<f64>> = face_sorted
        .iter()
        .map(|_| Array2::<f64>::zeros((p, p)))
        .collect();
    for (j, penalty) in penalties.iter().enumerate() {
        let lambda = rho[j].exp();
        if !lambda.is_finite() || lambda <= 0.0 {
            return Err(RailFaceLimitOutcome::OutsideClosedForm {
                reason: "a smoothing parameter does not exponentiate to a usable lambda"
                    .to_string(),
            });
        }
        let cols = penalty.col_range.clone();
        if cols.end > p {
            return Err(RailFaceLimitOutcome::OutsideClosedForm {
                reason: "a penalty's column range runs past the coefficient layout".to_string(),
            });
        }
        let slot = face_sorted.iter().position(|&f| f == j);
        for (li, gi) in cols.clone().enumerate() {
            for (lj, gj) in cols.clone().enumerate() {
                let value = penalty.local[[li, lj]];
                s_unit_all[[gi, gj]] += value;
                match slot {
                    Some(idx) => {
                        s_face_unit[[gi, gj]] += value;
                        face_penalties[idx][[gi, gj]] += value;
                    }
                    None => s_rest[[gi, gj]] += lambda * value,
                }
            }
        }
    }
    Ok(FacePenaltySplit {
        face_sorted,
        s_face_unit,
        s_rest,
        s_unit_all,
        face_penalties,
    })
}

/// Orthonormal bases of the subspace a face releases (`Q`) and the one it
/// pins (`Z`), from the face's unit-strength penalty sum.
pub(crate) struct FaceBases {
    pub(crate) q_basis: Array2<f64>,
    pub(crate) z_basis: Array2<f64>,
}

pub(crate) fn face_release_bases(
    s_face_unit: &Array2<f64>,
) -> Result<FaceBases, RailFaceLimitOutcome> {
    let (face_values, face_vectors) = match symmetric_eigh(s_face_unit) {
        Some(pair) => pair,
        None => {
            return Err(RailFaceLimitOutcome::FaceUnavailable {
                reason: "a symmetric eigendecomposition of the face geometry failed".to_string(),
            });
        }
    };
    let face_cut = range_null_split(&face_values);
    let released_cols: Vec<usize> = (0..face_values.len())
        .filter(|&i| face_values[i] > face_cut)
        .collect();
    let pinned_cols: Vec<usize> = (0..face_values.len())
        .filter(|&i| face_values[i] <= face_cut)
        .collect();
    if released_cols.is_empty() {
        // The face penalizes nothing: λ=∞ there is not a statement about
        // the model at all.
        return Err(RailFaceLimitOutcome::FaceUnavailable {
            reason: "the face's penalties release no direction: lambda=infinity there says nothing about the model".to_string(),
        });
    }
    Ok(FaceBases {
        q_basis: basis_columns(&face_vectors, &released_cols),
        z_basis: basis_columns(&face_vectors, &pinned_cols),
    })
}

/// Eigenpairs of the pinned block `ZᵀKZ` — the λ=∞ model's curvature — with
/// the identification gate and the conditioning the certificate's margin is
/// scaled by. `Ok(None)` when the face pins nothing.
pub(crate) struct PinnedBlock {
    pub(crate) values: Array1<f64>,
    pub(crate) vectors: Array2<f64>,
    pub(crate) conditioning: f64,
}

pub(crate) fn pinned_block_eigenpairs(
    k_matrix: &Array2<f64>,
    z_basis: &Array2<f64>,
) -> Result<Option<PinnedBlock>, RailFaceLimitOutcome> {
    if z_basis.ncols() == 0 {
        return Ok(None);
    }
    let kzz = z_basis.t().dot(k_matrix).dot(z_basis);
    let (values, vectors) = match symmetric_eigh(&kzz) {
        Some(pair) => pair,
        None => {
            return Err(RailFaceLimitOutcome::FaceUnavailable {
                reason: "a symmetric eigendecomposition of the face geometry failed".to_string(),
            });
        }
    };
    let largest = values.iter().fold(0.0_f64, |acc, v| acc.max(v.abs()));
    let smallest = values.iter().fold(f64::INFINITY, |acc, v| acc.min(*v));
    if !(smallest > f64::EPSILON.sqrt() * largest) || !(largest > 0.0) {
        // The λ=∞ model is not identified; there is no limit fit to
        // certify against.
        return Err(RailFaceLimitOutcome::FaceUnavailable {
            reason: format!(
                "the lambda=infinity model is not identified: reduced Hessian spectrum spans {smallest:.3e}..{largest:.3e}"
            ),
        });
    }
    Ok(Some(PinnedBlock {
        conditioning: largest / smallest,
        values,
        vectors,
    }))
}

/// Everything the final assembly needs beyond the face geometry: the
/// criterion's data curvature `K = I + S_R`, the limit fit and its score, and
/// the two knobs that differ between closed forms — the dispersion the fit
/// term is divided by (`φ̂` for profiled-Gaussian REML, exactly `1` for
/// fixed-dispersion LAML) and the optional curvature-drift vector `d`, zero
/// exactly when the working weights do not move with the coefficients.
struct FaceLimitAssembly<'a> {
    rho: &'a Array1<f64>,
    split: FacePenaltySplit,
    bases: FaceBases,
    pinned: Option<PinnedBlock>,
    k_matrix: Array2<f64>,
    limit_beta: Array1<f64>,
    limit_score: Array1<f64>,
    /// Natural magnitude the pinned-stationarity residual is judged against.
    score_scale: f64,
    /// Absolute stationarity tolerance for the pinned residual: `√ε·scale`
    /// when the limit fit is a direct linear solve (Gaussian), the inner
    /// solver's own certified standard when it is an iterative P-IRLS mode —
    /// judging an iterative fit by `√ε` would decline every converged limit.
    stationarity_tolerance: f64,
    dispersion: f64,
    curvature_drift: Option<Array1<f64>>,
}

/// The shared tail of both closed forms: pinned-stationarity check, the two
/// Schur complements, the pseudo-determinant rank bookkeeping, and the
/// first-order form itself.
fn assemble_face_limit(input: FaceLimitAssembly<'_>) -> RailFaceLimitOutcome {
    let FaceLimitAssembly {
        rho,
        split,
        bases,
        pinned,
        k_matrix,
        limit_beta,
        limit_score,
        score_scale,
        stationarity_tolerance,
        dispersion,
        curvature_drift,
    } = input;
    let FacePenaltySplit {
        face_sorted,
        s_face_unit: _,
        s_rest,
        s_unit_all,
        face_penalties,
    } = split;
    let FaceBases { q_basis, z_basis } = bases;
    let released = q_basis.ncols();
    let pinned_count = z_basis.ncols();

    if pinned_count > 0 {
        // The limit fit is stationary in the pinned directions by
        // construction; a residual there means the reduced solve did not
        // hold, so the rest of the algebra is not trustworthy either.
        let pinned_residual = z_basis.t().dot(&limit_score);
        if pinned_residual.dot(&pinned_residual).sqrt() > stationarity_tolerance {
            return RailFaceLimitOutcome::FaceUnavailable {
                reason: format!(
                    "the limit fit is not stationary in the pinned directions: residual {:.3e} against tolerance {stationarity_tolerance:.3e} (scale {score_scale:.3e})",
                    pinned_residual.dot(&pinned_residual).sqrt()
                ),
            };
        }
    }
    let released_score = q_basis.t().dot(&limit_score);

    // ── the two logdets' analytic first-order content ───────────────────
    // `½log|H|` leaves `Schur_Z(K)` behind once its `log|QᵀS_FQ|`
    // divergence is removed; `−½log|S_λ|₊` leaves `Schur_Z(S_R)`. Their
    // divergences are the same matrix and cancel exactly.
    let qkq = q_basis.t().dot(&k_matrix).dot(&q_basis);
    let (schur_k, conditioning) = match pinned.as_ref() {
        None => (qkq, 1.0_f64),
        Some(block) => {
            let qkz = q_basis.t().dot(&k_matrix).dot(&z_basis);
            let kzz_inverse = spectral_pseudo_inverse(
                &block.values,
                &block.vectors,
                f64::EPSILON.sqrt()
                    * block.values.iter().fold(0.0_f64, |acc, v| acc.max(v.abs())),
            );
            (qkq - qkz.dot(&kzz_inverse).dot(&qkz.t()), block.conditioning)
        }
    };
    let qsq = q_basis.t().dot(&s_rest).dot(&q_basis);
    let (schur_s_rest, surviving_rank, survivor_conditioning) = if pinned_count == 0 {
        (qsq, 0usize, 1.0_f64)
    } else {
        let zsz = z_basis.t().dot(&s_rest).dot(&z_basis);
        let (values, vectors) = match symmetric_eigh(&zsz) {
            Some(pair) => pair,
            None => {
                return RailFaceLimitOutcome::FaceUnavailable {
                    reason: "a symmetric eigendecomposition of the face geometry failed"
                        .to_string(),
                };
            }
        };
        let cut = range_null_split(&values);
        let kept: Vec<f64> = values.iter().copied().filter(|v| *v > cut).collect();
        let cond = match (
            kept.iter().fold(0.0_f64, |acc, v| acc.max(*v)),
            kept.iter().fold(f64::INFINITY, |acc, v| acc.min(*v)),
        ) {
            (hi, lo) if lo > 0.0 && hi.is_finite() => hi / lo,
            _ => 1.0,
        };
        let zsz_pseudo = spectral_pseudo_inverse(&values, &vectors, cut);
        let qsz = q_basis.t().dot(&s_rest).dot(&z_basis);
        (qsq - qsz.dot(&zsz_pseudo).dot(&qsz.t()), kept.len(), cond)
    };
    // The pseudo-determinant split `log|S_λ|₊ = log|QᵀS_λQ| +
    // log|Schur_Q(S_λ)|₊` needs the ranks to add up. That is an identity
    // (`dim(range S_R + range S_F) = q + dim P_N range S_R`), so a
    // mismatch means a rank determination is unreliable here.
    let (all_values, _) = match symmetric_eigh(&s_unit_all) {
        Some(pair) => pair,
        None => {
            return RailFaceLimitOutcome::FaceUnavailable {
                reason: "a symmetric eigendecomposition of the face geometry failed".to_string(),
            };
        }
    };
    let all_cut = range_null_split(&all_values);
    let penalty_rank = all_values.iter().filter(|v| **v > all_cut).count();
    if penalty_rank != released + surviving_rank {
        return RailFaceLimitOutcome::FaceUnavailable {
            reason: format!(
                "rank bookkeeping does not close: rank(S_lambda)={penalty_rank} but released={released} + surviving={surviving_rank}"
            ),
        };
    }

    // ── the first-order form ────────────────────────────────────────────
    let mut first_order_form = &schur_k - &schur_s_rest;
    for a in 0..released {
        for b in 0..released {
            first_order_form[[a, b]] -= released_score[a] * released_score[b] / dispersion;
        }
    }
    let released_curvature_drift = curvature_drift.map(|drift| q_basis.t().dot(&drift));
    if let Some(drift_q) = released_curvature_drift.as_ref() {
        // The LAML rank-2 correction: the Laplace logdet's curvature moves
        // with β̂, and the face's O(λ⁻¹) coefficient offset carries that
        // motion into first order (see the module derivation).
        for a in 0..released {
            for b in 0..released {
                first_order_form[[a, b]] +=
                    released_score[a] * drift_q[b] + drift_q[a] * released_score[b];
            }
        }
    }
    if first_order_form.iter().any(|v| !v.is_finite()) {
        return RailFaceLimitOutcome::FaceUnavailable {
            reason: "the assembled first-order form is not finite".to_string(),
        };
    }

    let released_penalties: Vec<Array2<f64>> = face_penalties
        .iter()
        .map(|s| q_basis.t().dot(s).dot(&q_basis))
        .collect();
    let face_rho: Vec<f64> = face_sorted.iter().map(|&j| rho[j]).collect();

    RailFaceLimitOutcome::Available(Box::new(RailFaceLimit {
        face: face_sorted,
        face_rho,
        first_order_form,
        released_penalties,
        released_score,
        form_conditioning: conditioning.max(survivor_conditioning).max(1.0),
        limit_beta,
        limit_dispersion: dispersion,
        released_curvature_drift,
    }))
}

/// Build the analytic λ→∞ face-limit data from a model's parts.
///
/// This is the Gaussian-identity closed form derived at the top of this module,
/// factored so any caller that owns a design, a response, prior weights and
/// canonical penalties can obtain the λ=∞ limit — the null-space-restricted
/// fit, its profiled dispersion, the limit score, and the first-order form —
/// without going through a `RemlState`. A caller whose criterion is NOT the
/// profiled-Gaussian REML must not use it: with `β̂`-dependent working weights
/// the logdet terms gain a third-derivative contribution at the same order,
/// which is [`laml_rail_face_limit`]'s rank-2 term.
///
/// `response` must already be net of any offset. `penalties` are in ρ-block
/// order, so `rho[j]` is `penalties[j]`'s log smoothing parameter, and `face`
/// indexes the same order. A decline is typed, never an error: this
/// certificate simply has nothing to say there.
///
/// The returned `limit_beta` is the λ=∞ fit itself, which is also the canonical
/// maximal-smoothing anchor a continuation can start from (#2366) instead of
/// solving at a large-but-finite ρ.
pub fn gaussian_rail_face_limit(
    design: ArrayView2<'_, f64>,
    response: ArrayView1<'_, f64>,
    weights: ArrayView1<'_, f64>,
    penalties: &[CanonicalPenalty],
    rho: &Array1<f64>,
    face: &[usize],
) -> RailFaceLimitOutcome {
    let p = design.ncols();
    let n = design.nrows();
    if p == 0 || n == 0 || response.len() != n || weights.len() != n {
        return RailFaceLimitOutcome::OutsideClosedForm {
            reason: "face-limit inputs are shape-inconsistent or empty".to_string(),
        };
    }
    if weights.iter().any(|w| !w.is_finite() || *w < 0.0) {
        return RailFaceLimitOutcome::OutsideClosedForm {
            reason: "prior weights are not finite and non-negative".to_string(),
        };
    }
    let split = match split_face_penalties(penalties, rho, face, p) {
        Ok(split) => split,
        Err(outcome) => return outcome,
    };
    let bases = match face_release_bases(&split.s_face_unit) {
        Ok(bases) => bases,
        Err(outcome) => return outcome,
    };
    let pinned = bases.z_basis.ncols();

    // ── weighted design cross-products ──────────────────────────────────
    let weight_sqrt: Array1<f64> = weights.iter().map(|w| w.sqrt()).collect();
    let mut x_scaled = design.to_owned();
    for (i, mut row) in x_scaled.axis_iter_mut(Axis(0)).enumerate() {
        let scale = weight_sqrt[i];
        row.mapv_inplace(|v| v * scale);
    }
    let response_scaled: Array1<f64> = response
        .iter()
        .zip(weight_sqrt.iter())
        .map(|(y, w)| y * w)
        .collect();
    let xtwx = x_scaled.t().dot(&x_scaled);
    let xtwy = x_scaled.t().dot(&response_scaled);
    let k_matrix = &xtwx + &split.s_rest;

    // ── the limit fit: the model restricted to the pinned subspace ──────
    let pinned_block = match pinned_block_eigenpairs(&k_matrix, &bases.z_basis) {
        Ok(block) => block,
        Err(outcome) => return outcome,
    };
    let mut limit_beta = Array1::<f64>::zeros(p);
    if let Some(block) = pinned_block.as_ref() {
        let rhs = bases.z_basis.t().dot(&xtwy);
        let rotated = block.vectors.t().dot(&rhs);
        let mut alpha = Array1::<f64>::zeros(pinned);
        for i in 0..pinned {
            alpha[i] = rotated[i] / block.values[i];
        }
        limit_beta = bases.z_basis.dot(&block.vectors.dot(&alpha));
    }

    // ── the profiled dispersion at the limit fit ────────────────────────
    let fitted = design.dot(&limit_beta);
    let mut weighted_rss = 0.0_f64;
    for i in 0..n {
        let residual = response[i] - fitted[i];
        weighted_rss += weights[i] * residual * residual;
    }
    let penalty_energy = limit_beta.dot(&split.s_rest.dot(&limit_beta));
    let penalized_deviance = weighted_rss + penalty_energy;
    // The profiled scale's denominator must be the CRITERION's own
    // degrees-of-freedom bookkeeping, which counts the SUM of the
    // per-penalty ranks — not the rank of the summed penalty. The two agree
    // when the penalties have disjoint ranges and differ exactly when they
    // OVERLAP, which is the all-on Hilbert-scale case this certificate is
    // for. Reproducing the criterion means reproducing its bookkeeping.
    let criterion_penalty_rank: usize = penalties
        .iter()
        .map(|penalty| penalty.positive_eigenvalues.len())
        .sum();
    let null_dim = p.saturating_sub(criterion_penalty_rank);
    if n <= null_dim {
        return RailFaceLimitOutcome::FaceUnavailable {
            reason: format!("no residual degrees of freedom at the limit: n={n} <= M_p={null_dim}"),
        };
    }
    // The REML profiled scale, `D_p/(n − M_p)`, evaluated AT the limit fit.
    // `M_p` does not move along the face, so the same denominator holds at
    // ρ̂ and at λ=∞.
    let dispersion = penalized_deviance / ((n - null_dim) as f64);
    if !dispersion.is_finite() || dispersion <= 0.0 {
        return RailFaceLimitOutcome::FaceUnavailable {
            reason: format!("the limit fit's profiled dispersion is unusable: {dispersion:.3e}"),
        };
    }

    // ── the limit score ─────────────────────────────────────────────────
    let limit_score = &xtwy - &k_matrix.dot(&limit_beta);
    let score_scale = xtwy
        .dot(&xtwy)
        .sqrt()
        .max(limit_score.dot(&limit_score).sqrt())
        .max(1.0);

    assemble_face_limit(FaceLimitAssembly {
        rho,
        split,
        bases,
        pinned: pinned_block,
        k_matrix,
        limit_beta,
        limit_score,
        score_scale,
        // The Gaussian limit fit is a direct linear solve, so its pinned
        // residual is roundoff-scale and `√ε` is the honest standard.
        stationarity_tolerance: f64::EPSILON.sqrt() * score_scale,
        dispersion,
        curvature_drift: None,
    })
}

/// Row bundle the LAML closed form reads off the CONVERGED λ=∞ limit fit.
///
/// All row vectors are length-`n` in design row order, in the criterion's own
/// units — prior weights folded in, exactly as the inner solver reports them:
///
/// * `working_weights` — the observed-information diagonal
///   `w_i = −∂²ℓ_i/∂η_i²` at the limit fit, the same `W` whose `XᵀWX` the
///   criterion's `½log|H|` uses;
/// * `score_residuals` — `u_i = ∂ℓ_i/∂η_i`, so `∇ℓ = Xᵀu`;
/// * `weight_eta_derivatives` — `c_i = dW_i/dη_i`, the third-derivative array
///   the exact LAML ρ-gradient already consumes (`solve_c_array`).
pub struct LamlFaceParts<'a> {
    /// The λ=∞ fit in the model's own coefficient basis (`Zα̂`).
    pub limit_beta: Array1<f64>,
    pub working_weights: ArrayView1<'a, f64>,
    pub score_residuals: ArrayView1<'a, f64>,
    pub weight_eta_derivatives: ArrayView1<'a, f64>,
    /// The RELATIVE stationarity standard the limit fit's mode was certified
    /// to by its own solver (the inner KKT tolerance). The pinned-direction
    /// residual check applies this same scale-invariant standard in the model
    /// basis — an iterative mode judged by the Gaussian path's `√ε` would be
    /// declined every time, converged or not.
    pub convergence_tolerance: f64,
    /// The stabilization ridge `δ` the limit fit's own criterion carries, so
    /// this form expands `H = XᵀWX + S_λ + δI` — the operator the criterion
    /// actually used — rather than the bare `XᵀWX + S_λ`.
    ///
    /// This field exists because the two were allowed to differ. The inner
    /// solver carries δ as `RidgePolicy::exact_full_objective()`, so it enters
    /// the criterion through `½log|H|`, while this form rebuilt its own `H`
    /// from the design and penalties and never added it. The gap was hidden
    /// only because δ used to be zero whenever the bare Cholesky succeeded —
    /// and that selector is itself the #1575/#2519 discontinuity, because it
    /// makes δ a function of ρ. Carrying δ here is what lets that selector
    /// become unconditional without making this form expand the wrong matrix.
    pub stabilization_ridge: f64,
}

/// Build the analytic λ→∞ face-limit data for a fixed-unit-dispersion LAML
/// criterion (see the module derivation: `C_LAML` adds the symmetric rank-2
/// curvature-drift term to the Gaussian form and drops the `1/φ̂` fit-term
/// scale).
///
/// The caller supplies the converged limit fit — the model restricted to the
/// face's null space, solved by the SAME inner engine the criterion itself
/// uses — through [`LamlFaceParts`]. Everything here is then exact arithmetic
/// on those outputs: no probe, no finite difference, and no evaluation at a
/// large λ anywhere.
pub fn laml_rail_face_limit(
    design: ArrayView2<'_, f64>,
    penalties: &[CanonicalPenalty],
    rho: &Array1<f64>,
    face: &[usize],
    parts: LamlFaceParts<'_>,
) -> RailFaceLimitOutcome {
    let p = design.ncols();
    let n = design.nrows();
    if p == 0
        || n == 0
        || parts.limit_beta.len() != p
        || parts.working_weights.len() != n
        || parts.score_residuals.len() != n
        || parts.weight_eta_derivatives.len() != n
    {
        return RailFaceLimitOutcome::OutsideClosedForm {
            reason: "face-limit inputs are shape-inconsistent or empty".to_string(),
        };
    }
    if parts.limit_beta.iter().any(|v| !v.is_finite())
        || parts.working_weights.iter().any(|v| !v.is_finite())
        || parts.score_residuals.iter().any(|v| !v.is_finite())
        || parts.weight_eta_derivatives.iter().any(|v| !v.is_finite())
    {
        return RailFaceLimitOutcome::OutsideClosedForm {
            reason: "the limit fit's row bundle is not finite".to_string(),
        };
    }
    let split = match split_face_penalties(penalties, rho, face, p) {
        Ok(split) => split,
        Err(outcome) => return outcome,
    };
    let bases = match face_release_bases(&split.s_face_unit) {
        Ok(bases) => bases,
        Err(outcome) => return outcome,
    };
    if bases.z_basis.ncols() == 0 {
        return RailFaceLimitOutcome::FaceUnavailable {
            reason: "the face releases every direction: the lambda=infinity limit model is empty"
                .to_string(),
        };
    }

    // I(β̂_∞) = Xᵀ·diag(w)·X, formed WITHOUT a square-root split: observed
    // weights may be negative on non-canonical links, and `H ≻ 0` is a
    // statement about the sum, not about the rows.
    let mut x_by_w = design.to_owned();
    for (i, mut row) in x_by_w.axis_iter_mut(Axis(0)).enumerate() {
        let w = parts.working_weights[i];
        row.mapv_inplace(|v| v * w);
    }
    let information = symmetrized(&design.t().dot(&x_by_w));
    // `H = XᵀWX + S_rest + δI`. The δ is the limit fit's own stabilization
    // ridge, and it belongs here for the same reason it belongs in the
    // criterion: the inner solver factors `XᵀWX + S_λ + δI` and reports
    // `½log|·|` of THAT, so a face expansion of the bare matrix is expanding a
    // different function. Adding δ to the diagonal shifts every eigenvalue of
    // the pinned block by exactly δ, which is what the criterion saw.
    let mut k_matrix = &information + &split.s_rest;
    if parts.stabilization_ridge > 0.0 {
        for i in 0..p {
            k_matrix[[i, i]] += parts.stabilization_ridge;
        }
    }
    let k_matrix = k_matrix;

    let pinned_block = match pinned_block_eigenpairs(&k_matrix, &bases.z_basis) {
        Ok(Some(block)) => block,
        // `z_basis.ncols() > 0` was checked above, so this arm is dead — but
        // a decline is the right dead-arm behavior for a certificate: never
        // abort a fit over an internal expectation.
        Ok(None) => {
            return RailFaceLimitOutcome::FaceUnavailable {
                reason: "the face releases every direction: the lambda=infinity limit model is empty"
                    .to_string(),
            };
        }
        Err(outcome) => return outcome,
    };

    // g_c = ∇ℓ(β̂_∞) − S_R β̂_∞: the Lagrange force the face carries.
    let grad_ell = design.t().dot(&parts.score_residuals);
    // The ridge pulls on the score exactly as it curves the Hessian. With
    // `H = XᵀWX + S_rest + δI` the stationarity condition is
    // `∇ℓ - S_restβ - δβ = 0`, so omitting the δβ term leaves a residual of
    // order δ·|β| and the pinned-direction check reads a converged mode as
    // non-stationary. Measured: residual 1.983e-8 against tolerance 1.490e-8,
    // a 1.33x miss at exactly the δ = 1e-8 scale.
    let mut penalty_pull = split.s_rest.dot(&parts.limit_beta);
    if parts.stabilization_ridge > 0.0 {
        penalty_pull = &penalty_pull + &(&parts.limit_beta * parts.stabilization_ridge);
    }
    let limit_score = &grad_ell - &penalty_pull;
    let score_scale = grad_ell
        .dot(&grad_ell)
        .sqrt()
        .max(limit_score.dot(&limit_score).sqrt())
        .max(1.0);
    if !(parts.convergence_tolerance.is_finite() && parts.convergence_tolerance > 0.0) {
        return RailFaceLimitOutcome::OutsideClosedForm {
            reason: "the limit fit's convergence tolerance is not usable".to_string(),
        };
    }
    // The inner solver certified `‖g‖ ≤ tol·(1 + natural scale)` in its own
    // frame; apply the identical scale-invariant standard to the rebuilt
    // gradient in this frame, floored by the Gaussian path's `√ε·scale` so a
    // tolerance tighter than roundoff cannot decline an exact solve.
    let natural_scale =
        1.0 + grad_ell.dot(&grad_ell).sqrt() + penalty_pull.dot(&penalty_pull).sqrt();
    let stationarity_tolerance = (parts.convergence_tolerance * natural_scale)
        .max(f64::EPSILON.sqrt() * score_scale);

    // The curvature-drift vector `d = ½Xᵀ(c ⊙ a)`, `a` the LIMIT model's
    // leverage `a_i = x_iᵀ Z (ZᵀKZ)⁻¹ Zᵀ x_i` — assembled from the pinned
    // block's own eigenpairs so it prices exactly the inverse the Schur
    // complement uses.
    let xz = design.dot(&bases.z_basis);
    let rotated = xz.dot(&pinned_block.vectors);
    let mut drift_rows = Array1::<f64>::zeros(n);
    for i in 0..n {
        let mut leverage = 0.0_f64;
        for k in 0..rotated.ncols() {
            leverage += rotated[[i, k]] * rotated[[i, k]] / pinned_block.values[k];
        }
        drift_rows[i] = 0.5 * parts.weight_eta_derivatives[i] * leverage;
    }
    let raw_drift = design.t().dot(&drift_rows);
    // The K-oblique reduction (module derivation): the face's coefficient
    // offset has an O(λ⁻¹) pinned component, so the drift acts through
    // `d̃ = d − KZ(ZᵀKZ)⁻¹Zᵀd` — the same reduction the Schur complement
    // applies to K itself. The orthogonal compression alone is a
    // first-order error.
    let pinned_drift = bases.z_basis.t().dot(&raw_drift);
    let rotated_pinned = pinned_block.vectors.t().dot(&pinned_drift);
    let mut solved = Array1::<f64>::zeros(rotated_pinned.len());
    for k in 0..rotated_pinned.len() {
        solved[k] = rotated_pinned[k] / pinned_block.values[k];
    }
    let oblique = k_matrix.dot(&bases.z_basis.dot(&pinned_block.vectors.dot(&solved)));
    let curvature_drift = &raw_drift - &oblique;

    assemble_face_limit(FaceLimitAssembly {
        rho,
        split,
        bases,
        pinned: Some(pinned_block),
        k_matrix,
        limit_beta: parts.limit_beta,
        limit_score,
        score_scale,
        stationarity_tolerance,
        // The family's dispersion is exactly 1 for the gated families, so the
        // fit term's `−g_Qg_Qᵀ` carries no divisor and the reported limit
        // dispersion is the family's own.
        dispersion: 1.0,
        curvature_drift: Some(curvature_drift),
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
            limit_beta: Array1::zeros(0),
            limit_dispersion: 1.0,
            released_curvature_drift: None,
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

    /// The LAML form against a fully hand-expanded 2-coefficient fixture:
    /// `X = [[1,0],[1,1],[1,2]]`, one face penalty `diag(0,1)` (releases the
    /// slope, pins the intercept), no survivors. Every quantity — the Schur
    /// complement, the score term, the leverage, the rank-2 drift — is a
    /// scalar identity here, so a sign error or a misplaced factor of two in
    /// the assembly misses by a lot, not by rounding.
    #[test]
    fn laml_face_form_matches_the_hand_expansion() {
        use crate::estimate::PenaltySpec;
        let mut design = Array2::<f64>::zeros((3, 2));
        for (i, t) in [0.0_f64, 1.0, 2.0].iter().enumerate() {
            design[[i, 0]] = 1.0;
            design[[i, 1]] = *t;
        }
        let w = Array1::from(vec![1.0_f64, 0.5, 2.0]);
        // Pinned stationarity demands `Σu_i = 0` exactly (Z = intercept).
        let u = Array1::from(vec![0.3_f64, -0.5, 0.2]);
        let c = Array1::from(vec![0.4_f64, -0.2, 0.3]);
        let beta_inf = Array1::from(vec![0.7_f64, 0.0]);
        let mut bend = Array2::<f64>::zeros((2, 2));
        bend[[1, 1]] = 1.0;
        let (penalties, _) = gam_terms::construction::canonicalize_penalty_specs(
            &[PenaltySpec::Dense(bend)],
            &[1],
            2,
            "laml_face_hand_fixture",
        )
        .expect("canonicalize the hand fixture penalty");
        let rho = Array1::from(vec![20.0_f64]);

        // Hand expansion. K = XᵀWX with K00 = Σw, K01 = Σw·t, K11 = Σw·t²;
        // g = Xᵀu = [0, −0.1]; a_i = 1/K00; d₀ = ½·Σc_i/K00,
        // d₁ = ½·Σ(c_i·t_i)/K00. The drift acts through the K-oblique
        // reduction `d̃₁ = d₁ − (K₀₁/K₀₀)·d₀` — the pinned component of the
        // face's coefficient offset is first-order too, and this scalar case
        // makes the correction visible by hand: d₀ ≠ 0 and K₀₁ ≠ 0, so the
        // orthogonal compression alone would be measurably wrong.
        let k00 = 3.5_f64;
        let k01 = 4.5_f64;
        let k11 = 8.5_f64;
        let g1 = -0.1_f64;
        let d0 = 0.5 * (0.4 + (-0.2) + 0.3) / k00;
        let d1 = 0.5 * (0.0 * 0.4 + 1.0 * (-0.2) + 2.0 * 0.3) / k00;
        let d1_oblique = d1 - (k01 / k00) * d0;
        let c_hand = (k11 - k01 * k01 / k00) - g1 * g1 + 2.0 * g1 * d1_oblique;
        let expected_constant = 0.5 * c_hand;

        let limit = laml_rail_face_limit(
            design.view(),
            &penalties,
            &rho,
            &[0],
            LamlFaceParts {
                limit_beta: beta_inf.clone(),
                working_weights: w.view(),
                score_residuals: u.view(),
                weight_eta_derivatives: c.view(),
                convergence_tolerance: f64::EPSILON,
                stabilization_ridge: 0.0,
            },
        )
        .available()
        .expect("the hand fixture is inside the LAML closed form");
        let drift = limit
            .released_curvature_drift
            .as_ref()
            .expect("the LAML form must record its curvature drift");
        assert!(
            (drift[0].abs() - d1_oblique.abs()).abs() <= 1.0e-14,
            "released drift |d̃_Q|={} should be the hand value {}",
            drift[0].abs(),
            d1_oblique.abs()
        );
        match certify_rail_face(&limit) {
            RailFaceVerdict::Certified(proof) => {
                assert!(
                    (proof.tail_constants[0] - expected_constant).abs()
                        <= 1.0e-12 * expected_constant,
                    "analytic c_0={} should equal the hand expansion {expected_constant}",
                    proof.tail_constants[0]
                );
            }
            other => panic!("the hand fixture must certify, got {other:?}"),
        }

        // The `c ≡ 0` member is the Gaussian reduction: the drift is recorded
        // as exactly zero and the constant loses exactly the rank-2 term.
        let zeros = Array1::<f64>::zeros(3);
        let reduced = laml_rail_face_limit(
            design.view(),
            &penalties,
            &rho,
            &[0],
            LamlFaceParts {
                limit_beta: beta_inf,
                working_weights: w.view(),
                score_residuals: u.view(),
                weight_eta_derivatives: zeros.view(),
                convergence_tolerance: f64::EPSILON,
                stabilization_ridge: 0.0,
            },
        )
        .available()
        .expect("the zero-drift fixture is inside the LAML closed form");
        assert_eq!(
            reduced
                .released_curvature_drift
                .as_ref()
                .expect("drift recorded")[0],
            0.0
        );
        match certify_rail_face(&reduced) {
            RailFaceVerdict::Certified(proof) => {
                let expected_no_drift = 0.5 * ((k11 - k01 * k01 / k00) - g1 * g1);
                assert!(
                    (proof.tail_constants[0] - expected_no_drift).abs()
                        <= 1.0e-12 * expected_no_drift,
                    "zero-drift c_0={} should drop exactly the rank-2 term ({expected_no_drift})",
                    proof.tail_constants[0]
                );
            }
            other => panic!("the zero-drift fixture must certify, got {other:?}"),
        }
    }

}
