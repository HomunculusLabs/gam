// ═══════════════════════════════════════════════════════════════════════════
//  Smooth spectral regularization
// ═══════════════════════════════════════════════════════════════════════════
//
// For indefinite or near-singular Hessians, hard eigenvalue clamping
// `max(σ, ε)` is non-smooth and creates inconsistency between log|H| and
// H⁻¹ at the threshold boundary. We use instead the C∞ regularizer:
//
//   r_ε(σ) = ½(σ + √(σ² + 4ε²))
//
// Properties:
//   - C∞ and strictly positive for all σ ∈ ℝ
//   - r_ε(σ) → σ  as σ → +∞  (transparent for well-conditioned eigenvalues)
//   - r_ε(σ) → ε  as σ → 0   (smooth floor)
//   - r_ε(σ) → ε²/|σ| as σ → -∞  (damps negative eigenvalues)
//
// Its derivative is:
//
//   r'_ε(σ) = ½(1 + σ/√(σ² + 4ε²))
//
// Using the SAME r_ε for both log-determinant and inverse ensures the
// gradient is the exact derivative of a single scalar objective — no
// inconsistency from mixing different regularizations.

/// Smooth spectral regularizer: `r_ε(σ) = ½(σ + √(σ² + 4ε²))`.
///
/// Returns a strictly positive value for any real `sigma`. For large positive
/// `sigma` this is approximately `sigma`; near zero it smoothly floors at `epsilon`.
#[inline]
pub(crate) fn spectral_regularize(sigma: f64, epsilon: f64) -> f64 {
    let disc = sigma.hypot(2.0 * epsilon);
    if sigma >= 0.0 {
        0.5 * sigma + 0.5 * disc
    } else {
        // Avoid catastrophic cancellation in 0.5 * (σ + disc) when σ is
        // large and negative: r_ε(σ) = 2 ε² / (disc - σ).
        (2.0 * epsilon * epsilon) / (disc - sigma)
    }
}

/// Compute the spectral regularization scale for a set of eigenvalues.
///
/// `ε = √(machine_eps) · p` where `p` is the matrix dimension — a
/// ρ-INDEPENDENT numerical-stability floor on the smooth regularization
/// `r_ε(σ)`.  The previous formulation `ε = √(machine_eps) · max(|σ_max|, 1)`
/// coupled ε to the largest eigenvalue of H(ρ), which made ε a function of
/// ρ whenever the Hessian spectrum moved with ρ.  That coupling leaked a
/// spurious ∂log|H|_reg/∂ρ contribution through the near-zero eigenvalues:
/// for σ_j ≪ ε we have `log r_ε(σ_j) ≈ log ε`, so `d log r_ε(σ_j)/dρ`
/// picks up `(1/ε) · dε/dρ` whenever max|σ_j| moved.  That created a
/// first-order derivative mismatch in outer REML gradients (up to ~1.5% of
/// the dominant `d log|H|/dρ` term on problems with one near-singular
/// direction, e.g. multi-block GAMLSS wiggle models where the intercept/wiggle
/// direction is effectively in the null space of the likelihood curvature).
///
/// The analytic gradient formula `tr(G_ε(H) · dH/dρ_k)` assumes ε is
/// fixed; removing the ρ-coupling restores that assumption.  Scaling ε
/// by the matrix dimension `p` (a ρ-independent integer, set by the
/// problem geometry) gives numerical stability for larger systems without
/// reintroducing ρ leakage.  The absolute floor stays below any physically
/// meaningful eigenvalue (for p ≤ 10⁶, ε ≤ 1.5e-2; well-conditioned
/// problems have min σ ≫ ε and are unaffected).
#[inline]
pub(crate) fn spectral_epsilon(eigenvalues: &[f64]) -> f64 {
    f64::EPSILON.sqrt() * (eigenvalues.len() as f64).max(1.0)
}
