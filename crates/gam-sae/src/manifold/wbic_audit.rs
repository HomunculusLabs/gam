//! Reconstruction spectrum behind the production rank charge (#2933 F30–F32).
//!
//! WHAT PRODUCTION CHARGES. The penalized quasi-Laplace criterion adds, per atom `k`,
//!
//! ```text
//! C_k    = ½ · r_k · edf_k · log max(N_eff,k, 1)
//! N_eff,k = Σ_i a_ik²,   G_k = Φ_kᵀ diag(a_k²) Φ_k,   edf_k = tr((G_k + λ_k S_k)⁻¹ G_k)
//! μ_j    = σ_j(G_k^½ B_k)² / N_eff,k,   e = R · (1 + √(p / N_eff,k))²
//! r_k    = #{j : μ_j > e}, promoted to 1 when that count is 0 but some μ_j > 0 (#2258)
//! ```
//!
//! This is a named criterion convention: a BIC-shaped charge on a thresholded
//! reconstruction rank. It is not a WBIC value, and nothing in this crate proves that
//! `r_k · edf_k` equals a real log-canonical threshold. The physical reconstruction
//! rank `#{μ_j > e}`, the chargeable rank `r_k`, the storage dimension `m · p`, the
//! basis EDF `edf_k`, the intrinsic manifold dimension and the RLCT are different
//! quantities.
//!
//! WHAT THE EDGE IS. `e` is the upper Marchenko–Pastur edge of the sample covariance
//! of an `N_eff × p` matrix of independent variance-`R` noise. The spectrum it
//! thresholds is not such a matrix. Conditional on coordinates, gates, dispersion and
//! the other atoms, a noise-only target `E` with independent `N(0, R)` entries gives
//! the ridge decoder `B̂ = (G + λS)⁻¹ Φᵀ diag(a) E`, so `M = G^½ B̂` has `p`
//! independent `N(0, R·C)` columns with `C = G^½ (G+λS)⁻¹ G (G+λS)⁻¹ G^½`, and
//! `M Mᵀ ~ Wishart_m(p, R·C)`. That law is scaled by `m`, `p` and `C`, not by
//! `N_eff`, and coordinate/gate fitting and selection add dependence it omits. `e`
//! is therefore a rank diagnostic, not a calibrated false-rank boundary or an
//! evidence dimension.
//!
//! BRANCHES. `r_k` is an integer, so `C_k` is piecewise smooth in the fitted state:
//! it jumps by `½ · edf_k · log N_eff,k` for every unit change of `r_k` when a
//! direction crosses `μ_j = e`. `production_rank_charge_derivative` differentiates
//! one fixed branch; it is not a derivative across a crossing.
//!
//! A RETIRED DERIVATION. This module once documented a tempered soft count
//! `½ μ / (μ + e · log N_eff)` as an exact closed-form WBIC coefficient. It is not
//! one. For a quadratic loss `L(w) = L(ŵ) + ½ h (w − ŵ)²`, a fixed prior
//! `N(0, 1/τ)` and inverse temperature `β`, the tempered posterior has mean
//! `m_β = β h ŵ / (β h + τ)` and variance `v_β = 1 / (β h + τ)`, and
//! `E_β[L(w)] − L(ŵ) = ½ h [(m_β − ŵ)² + v_β]`. The soft count kept only `v_β`: at
//! `h = τ = β = 1`, `ŵ = 2` it gives 0.25 where the expectation is 0.75. Its prior
//! precision also grew with `N_eff`, and a likelihood that is quadratic conditional
//! on coordinates does not make the joint decoder/coordinate model Gaussian.

use gam_linalg::faer_ndarray::{FaerEigh, FaerSvd};
use ndarray::Array2;

use super::Side;

/// The reconstruction spectrum of ONE atom. `mu` are the reconstruction-Gram
/// eigenvalues `sv(diag(√λ)·Uᵀ·D)²/n_eff` (with `(λ,U)=eigh(G)`) and `edge` the
/// Marchenko–Pastur reconstruction-rank edge `R·(1+√(p/n_eff))²`. This is the
/// decomposition inside `super::construction::realised_rank_charge_dof`, surfaced
/// so the rank-charge derivative classifies the same branch the value prices.
#[derive(Clone, Debug)]
pub struct ReconSpectrum {
    /// Reconstruction-Gram eigenvalues (per-observation signal+noise energy).
    mu: Vec<f64>,
    /// Marchenko–Pastur noise edge the hard rank count thresholds on.
    edge: f64,
}

impl ReconSpectrum {
    fn rank_classification(&self) -> super::construction::ReconstructionRankClassification {
        super::construction::classify_reconstruction_rank(&self.mu, self.edge)
    }

    /// #2258 production CHARGEABLE rank — the hard MP reconstruction count,
    /// with a below-rank-edge but numerically ALIVE atom promoted to the
    /// minimum non-degenerate rank 1. Mirrors the identical rule inside
    /// `super::construction::realised_rank_charge_dof` through the shared
    /// `super::construction::classify_reconstruction_rank` primitive;
    /// the ρ-derivative MUST take the same branch or the value/gradient pair
    /// desyncs (measured: real-GPT-2 fit priced finite by the promoted value
    /// path, then refused by the derivative's independent rank-zero invariant).
    /// Only an exactly zero reconstruction spectrum stays at rank 0 here.
    /// The stronger state-aware vanished-atom certificate runs before evidence
    /// pricing and may categorically refuse roundoff-indistinguishable signal.
    pub(crate) fn production_chargeable_rank(&self) -> usize {
        self.rank_classification().production_chargeable_rank
    }

}

/// Build the reconstruction spectrum from an atom's weighted basis Gram
/// `gram = Φᵀdiag(a²)Φ` (`m×m`), decoder `D` (`m×p`), effective sample size
/// `n_eff = Σ_row a²`, output dim `p_out`, noise floor `r_floor` (dispersion R),
/// and smoothness `(lam_smooth, smooth_penalty)`. Repeats the spectrum arithmetic
/// of `super::construction::realised_rank_charge_dof`, returning the spectrum
/// instead of the collapsed `rank_eff · basis_edf`.
pub(crate) fn recon_spectrum(
    gram: &Array2<f64>,
    decoder: &Array2<f64>,
    n_eff: f64,
    p_out: f64,
    r_floor: f64,
    lam_smooth: f64,
    smooth_penalty: Option<&Array2<f64>>,
) -> Result<ReconSpectrum, String> {
    let m = gram.nrows();
    super::construction::validate_rank_charge_problem(
        gram,
        decoder,
        n_eff,
        p_out,
        r_floor,
        lam_smooth,
        smooth_penalty,
    )?;
    if m == 0 || n_eff == 0.0 {
        return Ok(ReconSpectrum {
            mu: Vec::new(),
            edge: 0.0,
        });
    }
    let (evals, u) = gram
        .eigh(Side::Lower)
        .map_err(|e| format!("recon_spectrum: eigh(G): {e}"))?;
    let evals = super::construction::certified_psd_spectrum(evals.view(), "rank-charge Gram")?;
    let mut scaled = u.t().dot(decoder);
    let cols = scaled.ncols();
    for i in 0..m {
        let s = evals[i].sqrt();
        for j in 0..cols {
            scaled[[i, j]] *= s;
        }
    }
    let sv = match scaled.svd(false, false) {
        Ok((_, sv, _)) => sv,
        Err(e) => return Err(format!("recon_spectrum: recon svd: {e}")),
    };
    let edge = crate::null_battery::mp_reconstruction_rank_edge(n_eff, p_out, r_floor)
        .map_err(|error| format!("recon_spectrum: {error}"))?;
    let mu = sv
        .iter()
        .map(|&singular_value| {
            super::construction::normalized_reconstruction_energy(singular_value, n_eff)
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| format!("recon_spectrum: {error}"))?;
    // basis_edf = tr(G(G+λS)⁻¹), the same ridge trace the production core computes.
    let mut mmat = gram.clone();
    if let Some(pen) = smooth_penalty {
        for i in 0..m {
            for j in 0..m {
                mmat[[i, j]] += lam_smooth * pen[[i, j]];
            }
        }
    }
    Ok(ReconSpectrum {
        mu,
        edge,
    })
}

