//! Dense row-major Cholesky helpers.
//!
//! `cholesky_logdet` factors a symmetric positive-definite `p×p` buffer in
//! place and returns its exact log-determinant; `chol_solve` solves
//! `L Lᵀ x = b` from that factor. The residual cascade in gam-solve uses them
//! for its dense penalized systems.

/// Dense lower-Cholesky in place (row-major `p×p`); returns the exact
/// `log det` (twice the log of the pivot products). The strict upper triangle
/// is zeroed so the buffer is exactly `L` afterwards.
pub fn cholesky_logdet(a: &mut [f64], p: usize) -> Result<f64, String> {
    let mut logdet = 0.0;
    for j in 0..p {
        let diag = a[j * p + j];
        let mut s = diag;
        let mut subtracted = 0.0_f64;
        for t in 0..j {
            let sq = a[j * p + t] * a[j * p + t];
            s -= sq;
            subtracted += sq;
        }
        // The pivot is `a_jj − Σ_t l_jt²`: one product and one subtraction per
        // term, so a pivot inside the rounding band of that accumulation is
        // not distinguishable from zero and the system is not positive definite
        // as computed (#2469). An absolute `1e-300` passed pure roundoff of an
        // O(1) row and refused an honest tiny pivot.
        let band = gam_linalg::roundoff::accumulation_growth(2 * j + 1) * (diag.abs() + subtracted);
        if !(s.is_finite() && s > band) {
            return Err(format!(
                "grid spline 2d: penalized system not positive definite at pivot {j} (value {s})"
            ));
        }
        let l = s.sqrt();
        a[j * p + j] = l;
        logdet += 2.0 * l.ln();
        for i in j + 1..p {
            let mut s2 = a[i * p + j];
            for t in 0..j {
                s2 -= a[i * p + t] * a[j * p + t];
            }
            a[i * p + j] = s2 / l;
        }
    }
    for i in 0..p {
        for j in i + 1..p {
            a[i * p + j] = 0.0;
        }
    }
    Ok(logdet)
}

/// Solve `L z = b` from a dense row-major lower-triangular factor.
fn lower_solve(l: &[f64], p: usize, b: &[f64]) -> Vec<f64> {
    let mut z = b.to_vec();
    for i in 0..p {
        let mut s = z[i];
        for t in 0..i {
            s -= l[i * p + t] * z[t];
        }
        z[i] = s / l[i * p + i];
    }
    z
}

/// Solve `L Lᵀ x = b` from the stored lower factor.
pub fn chol_solve(l: &[f64], p: usize, b: &[f64]) -> Vec<f64> {
    let mut z = lower_solve(l, p, b);
    for i in (0..p).rev() {
        let mut s = z[i];
        for t in i + 1..p {
            s -= l[t * p + i] * z[t];
        }
        z[i] = s / l[i * p + i];
    }
    z
}
