//! The conditional-transformation log-normalizer seam.
//!
//! The fitted CTN likelihood is the most-likely-transformation density
//! `f(y) = φ(h(y)) · h'(y)` — a proper density on the whole real line, with
//! `log Z ≡ 0`. gam#2600 measured what the previous *truncated* form
//! (`log Z = log[Φ(h(y_hi)) − Φ(h(y_lo))]`, the mass between two FITTED
//! endpoints) does to that objective: `log Z` is concave in the endpoints by
//! Prékopa, so subtracting it makes the negative log-likelihood non-convex,
//! and it cancels the `−½h²` coercivity so the supremum moves to `‖β‖ = ∞`.
//! [`LogNormalCdfDiffDerivatives::untruncated`] is the resulting normalizer and
//! is the only one the fit uses.
//!
//! Everything here is pure scalar math with no dependence on the family state,
//! so it lives in its own seam.


#[derive(Clone, Copy, Debug)]
pub(crate) struct LogNormalCdfDiffDerivatives {
    pub(super) log_z: f64,
    pub(super) first: [f64; 2],
    pub(super) second: [[f64; 2]; 2],
    pub(super) third: [[[f64; 2]; 2]; 2],
    pub(super) fourth: [[[[f64; 2]; 2]; 2]; 2],
}

impl LogNormalCdfDiffDerivatives {
    /// The normalizer of a transformation whose image is the WHOLE real line:
    /// `Z = Φ(+∞) − Φ(−∞) = 1`, so `log Z = 0` and every derivative of `log Z`
    /// with respect to the endpoints vanishes identically.
    ///
    /// This is the most-likely-transformation likelihood (`mlt`/`tram`,
    /// Hothorn–Möst–Bühlmann 2018): `f(y) = φ(h(y)) · h'(y)` with no
    /// renormalization. gam#2600 established that renormalizing by the mass
    /// between two FITTED endpoints is what makes the CTN objective
    /// non-concave and non-coercive — see the module header on
    /// `transformation_normal` for the measurement.
    pub(crate) const fn untruncated() -> Self {
        Self {
            log_z: 0.0,
            first: [0.0; 2],
            second: [[0.0; 2]; 2],
            third: [[[0.0; 2]; 2]; 2],
            fourth: [[[[0.0; 2]; 2]; 2]; 2],
        }
    }
}

pub(crate) fn endpoint_chain_first(q: &LogNormalCdfDiffDerivatives, a: [f64; 2]) -> f64 {
    q.first[0] * a[0] + q.first[1] * a[1]
}

pub(crate) fn endpoint_chain_second(
    q: &LogNormalCdfDiffDerivatives,
    a: [f64; 2],
    b: [f64; 2],
    ab: [f64; 2],
) -> f64 {
    let mut out = endpoint_chain_first(q, ab);
    for i in 0..2 {
        for j in 0..2 {
            out += q.second[i][j] * a[i] * b[j];
        }
    }
    out
}

pub(crate) fn endpoint_chain_third(
    q: &LogNormalCdfDiffDerivatives,
    a: [f64; 2],
    b: [f64; 2],
    c: [f64; 2],
    ab: [f64; 2],
    ac: [f64; 2],
    bc: [f64; 2],
    abc: [f64; 2],
) -> f64 {
    let mut out = endpoint_chain_first(q, abc);
    for i in 0..2 {
        for j in 0..2 {
            out += q.second[i][j] * (ab[i] * c[j] + ac[i] * b[j] + bc[i] * a[j]);
            for k in 0..2 {
                out += q.third[i][j][k] * a[i] * b[j] * c[k];
            }
        }
    }
    out
}

pub(crate) fn endpoint_chain_fourth(
    q: &LogNormalCdfDiffDerivatives,
    a: [f64; 2],
    b: [f64; 2],
    c: [f64; 2],
    d: [f64; 2],
    ab: [f64; 2],
    ac: [f64; 2],
    ad: [f64; 2],
    bc: [f64; 2],
    bd: [f64; 2],
    cd: [f64; 2],
    abc: [f64; 2],
    abd: [f64; 2],
    acd: [f64; 2],
    bcd: [f64; 2],
    abcd: [f64; 2],
) -> f64 {
    let mut out = endpoint_chain_first(q, abcd);
    for i in 0..2 {
        for j in 0..2 {
            out += q.second[i][j]
                * (abc[i] * d[j]
                    + abd[i] * c[j]
                    + acd[i] * b[j]
                    + bcd[i] * a[j]
                    + ab[i] * cd[j]
                    + ac[i] * bd[j]
                    + ad[i] * bc[j]);
            for k in 0..2 {
                out += q.third[i][j][k]
                    * (ab[i] * c[j] * d[k]
                        + ac[i] * b[j] * d[k]
                        + ad[i] * b[j] * c[k]
                        + bc[i] * a[j] * d[k]
                        + bd[i] * a[j] * c[k]
                        + cd[i] * a[j] * b[k]);
                for l in 0..2 {
                    out += q.fourth[i][j][k][l] * a[i] * b[j] * c[k] * d[l];
                }
            }
        }
    }
    out
}

