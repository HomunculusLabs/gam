//! #1011 — the certified two-sided enclosure of a block-SPD log-determinant.
//!
//! A [`LogdetEnclosure`] records certified bounds `lower ≤ log|S| ≤ upper` for a
//! block-partitioned SPD matrix `S`. With `D = blockdiag(S_11..S_KK)` and
//! `E = D^{-1/2}(S − D)D^{-1/2}`, it carries the exact `log|D|`, the moments
//! `tr E²` and `tr E³`, and the spectral-radius certificate `ρ < 1` behind the
//! bounds (derivation on issue #1011). Its midpoint stands in for the exact value
//! only in a decision whose margin exceeds the enclosure's width.

/// A certified enclosure of `log|S|` for a block-partitioned SPD matrix.
#[derive(Debug, Clone)]
pub struct LogdetEnclosure {
    /// Exact `log|D| = Σ_i log|S_ii|` from the per-block Cholesky factors.
    pub block_diag_logdet: f64,
    /// Certified lower bound on `log|S|` (i.e. `block_diag_logdet + correction_lower`).
    pub lower: f64,
    /// Certified upper bound on `log|S|`.
    pub upper: f64,
    /// The spectral-radius certificate used (`< 1` or this struct would not exist).
    pub rho: f64,
    /// Exact second moment `tr(E²)`.
    pub p2: f64,
    /// Exact third moment `tr(E³)` when the order-3 enclosure was requested.
    pub p3: Option<f64>,
}

impl LogdetEnclosure {
    /// Width of the enclosure — compare against the consuming decision's margin.
    pub fn gap(&self) -> f64 {
        self.upper - self.lower
    }
}
