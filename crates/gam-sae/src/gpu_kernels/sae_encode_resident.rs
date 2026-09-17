//! Cyclic-Jacobi symmetric eigensolver for small dense `d×d` matrices.
//!
//! [`jacobi_eigh`] diagonalises a row-major symmetric matrix by cyclic Jacobi
//! rotations and certifies its own stopping point: it returns `true` only when
//! the off-diagonal mass has fallen inside the rounding band the rotations leave
//! on the diagonal ([`jacobi_off_diagonal_band_coefficient_squared`]), and
//! `false` when [`JACOBI_MAX_SWEEPS`] ran out first. `gam-pyffi` reaches it
//! through this module path.

/// Sweep budget of the cyclic Jacobi eigensolver. Cyclic Jacobi converges
/// quadratically once the off-diagonal mass is small, so for `d ≤ 8` the band
/// below is reached in a handful of sweeps; the budget only bounds the work on
/// an input the rotations cannot diagonalise (a non-finite matrix), and running
/// out of it is REPORTED by [`jacobi_eigh`], never absorbed.
pub const JACOBI_MAX_SWEEPS: usize = 30;

/// The squared coefficient of the Jacobi stopping band, for a `d×d` matrix.
///
/// The sweeps stop when `‖offdiag‖_F ≤ γ_{2d}·‖diag‖_F`: by Weyl's inequality
/// every eigenvalue of the iterate then lies within `‖offdiag‖_F` of a diagonal
/// entry, and `γ_{2d}·‖diag‖_F` is the rounding band the diagonal already carries
/// from the `2(d − 1)` rotation updates each of its entries receives per sweep.
/// `γ_n = n·u/(1 − n·u)` already carries the unit roundoff `u`; a second factor
/// of `u` made the band `O(u²)`, below anything binary64 resolves, so the stop
/// fired only once the upper triangle rounded to exact zeros (#2627). The band
/// belongs to the arithmetic, not to a magnitude: the former `1e-300` test was
/// inert for every matrix that was not already exactly diagonal.
pub fn jacobi_off_diagonal_band_coefficient_squared(d: usize) -> f64 {
    let coefficient = gam_linalg::roundoff::accumulation_growth(2 * d);
    coefficient * coefficient
}

/// Cyclic Jacobi symmetric eigensolver for a `d×d` matrix (row-major, `d ≤ 8`).
/// Returns eigenvalues `vals[i]` and eigenvectors as COLUMNS
/// `vecs[col*d + row]`.
///
/// The return value is the certificate: `true` when the sweeps drove the
/// off-diagonal mass below the arithmetic's own band
/// ([`jacobi_off_diagonal_band_coefficient_squared`]), `false` when
/// [`JACOBI_MAX_SWEEPS`] ran out first. `vals`/`vecs` hold the last iterate in
/// both cases; a caller must not certify anything on `false`.
#[must_use]
pub fn jacobi_eigh(a_in: &[f64], d: usize, vals: &mut [f64], vecs: &mut [f64]) -> bool {
    // Working copy A (row-major), V = I.
    let mut a = a_in.to_vec();
    for r in 0..d {
        for c in 0..d {
            vecs[c * d + r] = if r == c { 1.0 } else { 0.0 };
        }
    }
    if d == 1 {
        vals[0] = a[0];
        return true;
    }
    let band_coefficient_squared = jacobi_off_diagonal_band_coefficient_squared(d);
    let mut converged = false;
    for _sweep in 0..JACOBI_MAX_SWEEPS {
        // Off-diagonal mass (upper triangle, so `‖offdiag‖_F² = 2·off`) against
        // the diagonal's own rounding band.
        let mut off = 0.0_f64;
        let mut diag_sq = 0.0_f64;
        for r in 0..d {
            diag_sq += a[r * d + r] * a[r * d + r];
            for c in (r + 1)..d {
                off += a[r * d + c] * a[r * d + c];
            }
        }
        if 2.0 * off <= band_coefficient_squared * diag_sq {
            converged = true;
            break;
        }
        for pp in 0..d {
            for q in (pp + 1)..d {
                let apq = a[pp * d + q];
                if apq == 0.0 {
                    continue;
                }
                let app = a[pp * d + pp];
                let aqq = a[q * d + q];
                // Jacobi rotation angle (Golub & Van Loan 8.4.1).
                let tau = (aqq - app) / (2.0 * apq);
                let t = if tau >= 0.0 {
                    1.0 / (tau + (1.0 + tau * tau).sqrt())
                } else {
                    -1.0 / (-tau + (1.0 + tau * tau).sqrt())
                };
                let cph = 1.0 / (1.0 + t * t).sqrt();
                let sph = t * cph;
                // Apply rotation to A (rows/cols pp,q).
                for k in 0..d {
                    let akp = a[k * d + pp];
                    let akq = a[k * d + q];
                    a[k * d + pp] = cph * akp - sph * akq;
                    a[k * d + q] = sph * akp + cph * akq;
                }
                for k in 0..d {
                    let apk = a[pp * d + k];
                    let aqk = a[q * d + k];
                    a[pp * d + k] = cph * apk - sph * aqk;
                    a[q * d + k] = sph * apk + cph * aqk;
                }
                // Accumulate eigenvectors.
                for k in 0..d {
                    let vkp = vecs[pp * d + k];
                    let vkq = vecs[q * d + k];
                    vecs[pp * d + k] = cph * vkp - sph * vkq;
                    vecs[q * d + k] = sph * vkp + cph * vkq;
                }
            }
        }
    }
    for i in 0..d {
        vals[i] = a[i * d + i];
    }
    converged
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jacobi_eigh_matches_reference_2x2() {
        // Symmetric 2x2 spectral check: reconstruct A from V diag(vals) Vᵀ.
        let a = [4.0, 1.0, 1.0, 3.0];
        let mut vals = [0.0; 2];
        let mut vecs = [0.0; 4];
        assert!(
            jacobi_eigh(&a, 2, &mut vals, &mut vecs),
            "a 2×2 symmetric matrix diagonalises in one rotation"
        );
        // A_reconstructed[r][c] = Σ_k vals[k] v_k[r] v_k[c].
        for r in 0..2 {
            for c in 0..2 {
                let mut acc = 0.0;
                for k in 0..2 {
                    acc += vals[k] * vecs[k * 2 + r] * vecs[k * 2 + c];
                }
                assert!(
                    (acc - a[r * 2 + c]).abs() < 1e-12,
                    "eig reconstruct {r},{c}"
                );
            }
        }
        // Eigenvalues of [[4,1],[1,3]] are (7±√5)/2.
        let mut vs = vals.to_vec();
        vs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((vs[0] - (7.0 - 5.0_f64.sqrt()) / 2.0).abs() < 1e-12);
        assert!((vs[1] - (7.0 + 5.0_f64.sqrt()) / 2.0).abs() < 1e-12);
    }

    /// #2627 — a matrix whose off-diagonal mass already lies inside the band
    /// `γ_{2d}·‖diag‖_F` is certified before any rotation: V stays bitwise the
    /// identity and the eigenvalues are bitwise the input diagonal. With the
    /// band's former extra factor of `u` the solver rotated it (V gained ±1e-17).
    #[test]
    fn an_in_band_matrix_certifies_without_rotating_2627() {
        let d = 3;
        let tiny = 1.0e-17;
        let a = [1.0, tiny, tiny, tiny, 2.0, tiny, tiny, tiny, 3.0];
        let off_frobenius = (6.0_f64).sqrt() * tiny;
        let diag_frobenius = (14.0_f64).sqrt();
        assert!(
            off_frobenius <= gam_linalg::roundoff::accumulation_growth(2 * d) * diag_frobenius,
            "the fixture must lie inside the band it pins"
        );
        let mut vals = [0.0; 3];
        let mut vecs = [0.0; 9];
        assert!(
            jacobi_eigh(&a, d, &mut vals, &mut vecs),
            "an in-band matrix certifies"
        );
        for r in 0..d {
            for c in 0..d {
                assert_eq!(
                    vecs[c * d + r],
                    if r == c { 1.0 } else { 0.0 },
                    "V[{r},{c}] must be untouched by rotations"
                );
            }
            assert_eq!(vals[r], a[r * d + r], "eigenvalue {r} must be the input diagonal");
        }
    }

    /// The certified stop bounds the eigenvalue error by the off-diagonal band
    /// `γ_{2d}·‖diag‖_F` at stop plus at most `JACOBI_MAX_SWEEPS` sweeps of the
    /// diagonal's per-sweep rounding band, with `‖diag‖_F ≤ ‖A‖_F`. The input is
    /// exact, so the reference spectrum `2 − √2, 2, 2 + √2` carries no
    /// construction rounding.
    #[test]
    fn certified_eigenvalues_lie_within_the_band_of_an_exact_spectrum_2627() {
        let d = 3;
        let a = [2.0, -1.0, 0.0, -1.0, 2.0, -1.0, 0.0, -1.0, 2.0];
        let mut vals = [0.0; 3];
        let mut vecs = [0.0; 9];
        assert!(
            jacobi_eigh(&a, d, &mut vals, &mut vecs),
            "the exact tridiagonal certifies"
        );
        let frobenius = a.iter().map(|value| value * value).sum::<f64>().sqrt();
        let bound = (1 + JACOBI_MAX_SWEEPS) as f64
            * gam_linalg::roundoff::accumulation_growth(2 * d)
            * frobenius;
        let root_two = 2.0_f64.sqrt();
        let mut sorted = vals.to_vec();
        sorted.sort_by(f64::total_cmp);
        for (computed, exact) in sorted.iter().zip([2.0 - root_two, 2.0, 2.0 + root_two]) {
            assert!(
                (computed - exact).abs() <= bound,
                "eigenvalue {computed:e} vs {exact:e}: error beyond the certified band {bound:e}"
            );
        }
    }
}
