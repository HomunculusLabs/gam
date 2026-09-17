//! Rank-charge strata, the conditional noise-only law of the reconstruction
//! spectrum, and the exact tempered posterior of a decoder block (#2933 F30–F32).
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
//! quantities, and [`AtomRankChargeAudit`] reports them as separate fields.
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
//! evidence dimension. [`conditional_noise_null`] evaluates the law's exact expected
//! total energy and an upper bound on its expected top energy, so `e` can be read
//! against the scale that noise reaches.
//!
//! BRANCHES. `r_k` is an integer, so `C_k` is piecewise smooth in the fitted state:
//! it jumps by `½ · edf_k · log N_eff,k` for every unit change of `r_k` when a
//! direction crosses `μ_j = e`. `rank_charge_stratum` is the one producer of that
//! branch: the value (`realised_rank_charge_dof`), its within-branch analytic
//! derivative (`production_rank_charge_derivative`) and
//! `SaeManifoldTerm::rank_charge_audit` all classify through it.
//! [`RankChargeStratum::nearest_mp_boundary`] names the direction closest to the edge
//! and the charge jump its crossing adds.
//!
//! TEMPERED POSTERIOR. For a quadratic loss `L(w) = L(ŵ) + ½ (w − ŵ)ᵀ H (w − ŵ)` and a
//! fixed Gaussian prior with precision `Λ` and mean `μ₀` (`Λ` may be singular when
//! `βH + Λ` is positive definite), the tempered posterior `∝ exp(−β L(w)) π(w)` has
//! precision `P = βH + Λ` and mean `m = P⁻¹ (βHŵ + Λμ₀)`, and
//!
//! ```text
//! E_β[L(w)] − L(ŵ) = ½ (m − ŵ)ᵀ H (m − ŵ) + ½ tr(H P⁻¹),   m − ŵ = P⁻¹ Λ (μ₀ − ŵ).
//! ```
//!
//! [`tempered_gaussian_excess_loss`] evaluates both terms. With `L = n L_n` and
//! `β = 1 / log n` the left side is Watanabe's WBIC minus `n L_n(ŵ)` for THAT model.
//! [`conditional_decoder_tempered_posterior`] applies it to one atom's decoder with
//! coordinates, gates, dispersion and the other atoms held fixed, unit row weights,
//! the identity output metric and the fitted smoothing prior (precision `λS / R`).
//! That conditional model is exactly Gaussian, so the value is exact for it. It is
//! not an RLCT estimate for the joint decoder/coordinate/gate model.
//!
//! A RETIRED DERIVATION. This module once documented a tempered soft count
//! `½ μ / (μ + e · log N_eff)` as an exact closed-form WBIC coefficient. It kept only
//! the spread term: for `H = h`, `Λ = τ`, `μ₀ = 0`, `β = h = τ = 1` and `ŵ = 2` the
//! spread is 0.25 while the expectation is 0.75. Its prior precision also grew with
//! `N_eff`, and a likelihood that is quadratic conditional on coordinates does not
//! make the joint decoder/coordinate model Gaussian.

use gam_linalg::faer_ndarray::{FaerCholesky, FaerEigh, FaerSvd};
use ndarray::{Array2, ArrayView1, ArrayView2, Axis};

use super::Side;
use super::construction::{
    ReconstructionRankClassification, certified_basis_edf, certified_psd_spectrum,
    classify_reconstruction_rank, normalized_reconstruction_energy, validate_rank_charge_problem,
};

/// One atom's priced rank-charge branch at one state: the per-observation
/// reconstruction energies `μ_j`, the Marchenko–Pastur rank edge `e` they are
/// classified against, the occupancy `N_eff`, the ridge-trace basis EDF and the
/// two integer ranks.
#[derive(Clone, Debug)]
pub struct RankChargeStratum {
    energies: Vec<f64>,
    edge: f64,
    effective_sample_size: f64,
    basis_edf: f64,
    classification: ReconstructionRankClassification,
}

/// The reconstruction direction closest to the MP rank edge, and what its
/// crossing changes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MpEdgeBoundary {
    /// Index into [`RankChargeStratum::reconstruction_energies`].
    pub direction: usize,
    /// `μ_direction − e`. Positive means the direction is counted in the MP rank.
    pub signed_gap: f64,
    /// Change of [`RankChargeStratum::production_charge`] when this direction
    /// alone crosses the edge. It is zero where the #2258 promotion keeps the
    /// chargeable rank at one.
    pub charge_jump: f64,
}

impl RankChargeStratum {
    /// Per-observation reconstruction energies `μ_j`, in singular-value order.
    pub fn reconstruction_energies(&self) -> &[f64] {
        &self.energies
    }

    /// Largest reconstruction energy, zero when there is none.
    pub fn top_reconstruction_energy(&self) -> f64 {
        self.classification.top_signal
    }

    /// The Marchenko–Pastur rank edge `R·(1+√(p/N_eff))²`.
    pub fn mp_reconstruction_rank_edge(&self) -> f64 {
        self.edge
    }

    /// Occupancy `N_eff = Σ_i a_i²`.
    pub fn effective_sample_size(&self) -> f64 {
        self.effective_sample_size
    }

    /// Ridge-trace basis EDF `tr((G + λS)⁻¹ G)`.
    pub fn basis_edf(&self) -> f64 {
        self.basis_edf
    }

    /// `#{j : μ_j > e}`, the physical reconstruction rank at the edge.
    pub fn mp_reconstruction_rank(&self) -> usize {
        self.classification.mp_reconstruction_rank
    }

    /// The rank the criterion charges, including the #2258 promotion of an alive
    /// decoder with MP rank zero to rank one.
    pub fn production_chargeable_rank(&self) -> usize {
        self.classification.production_chargeable_rank
    }

    /// `r · edf`, the DOF the criterion multiplies by `½ log max(N_eff, 1)`.
    pub fn production_dof(&self) -> f64 {
        self.production_chargeable_rank() as f64 * self.basis_edf
    }

    /// `½ · r · edf · log max(N_eff, 1)`, the arithmetic of the criterion's rank
    /// charge for this atom.
    pub fn production_charge(&self) -> f64 {
        charge_for_dof(self.production_dof(), self.effective_sample_size)
    }

    /// The direction whose energy is closest to the edge, with the charge jump a
    /// crossing adds. A direction above the edge crosses by falling to it; one at
    /// or below crosses by rising just past it. `None` without energies.
    pub fn nearest_mp_boundary(&self) -> Option<MpEdgeBoundary> {
        let (direction, energy) = self
            .energies
            .iter()
            .copied()
            .enumerate()
            .min_by(|(_, left), (_, right)| {
                (left - self.edge).abs().total_cmp(&(right - self.edge).abs())
            })?;
        let mut crossed = self.energies.clone();
        crossed[direction] = if energy > self.edge {
            self.edge
        } else {
            self.edge.next_up()
        };
        let crossed_rank =
            classify_reconstruction_rank(&crossed, self.edge).production_chargeable_rank;
        let crossed_dof = crossed_rank as f64 * self.basis_edf;
        Some(MpEdgeBoundary {
            direction,
            signed_gap: energy - self.edge,
            charge_jump: charge_for_dof(crossed_dof, self.effective_sample_size)
                - self.production_charge(),
        })
    }
}

fn charge_for_dof(dof: f64, effective_sample_size: f64) -> f64 {
    0.5 * dof * effective_sample_size.max(1.0).ln()
}

/// Classify one atom's rank-charge branch from its weighted basis Gram
/// `gram = Φᵀdiag(a²)Φ` (`m×m`), decoder `D` (`m×p`), occupancy `n_eff = Σ_row a²`,
/// output width `p_out`, dispersion `r_floor` and smoothing `(lam_smooth,
/// smooth_penalty)`. The production value, its analytic derivative and the audit
/// all call this, so they price one branch of one state.
pub(crate) fn rank_charge_stratum(
    gram: &Array2<f64>,
    decoder: &Array2<f64>,
    n_eff: f64,
    p_out: f64,
    r_floor: f64,
    lam_smooth: f64,
    smooth_penalty: Option<&Array2<f64>>,
) -> Result<RankChargeStratum, String> {
    let m = gram.nrows();
    validate_rank_charge_problem(
        gram,
        decoder,
        n_eff,
        p_out,
        r_floor,
        lam_smooth,
        smooth_penalty,
    )?;
    if m == 0 || n_eff == 0.0 {
        return Ok(RankChargeStratum {
            energies: Vec::new(),
            edge: 0.0,
            effective_sample_size: n_eff,
            basis_edf: 0.0,
            classification: classify_reconstruction_rank(&[], 0.0),
        });
    }
    // MP reconstruction rank on the reconstruction Gram. U orthogonal ⇒ svd of
    // diag(√λ)·Uᵀ·D equals svd of the reconstruction square root G^½·D.
    let (evals, u) = gram
        .eigh(Side::Lower)
        .map_err(|e| format!("rank-charge stratum: eigh(G): {e}"))?;
    let evals = certified_psd_spectrum(evals.view(), "rank-charge Gram")?;
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
        Err(e) => return Err(format!("rank-charge stratum: recon svd: {e}")),
    };
    let edge = crate::null_battery::mp_reconstruction_rank_edge(n_eff, p_out, r_floor)
        .map_err(|error| format!("rank-charge stratum: {error}"))?;
    let energies = sv
        .iter()
        .map(|&singular_value| normalized_reconstruction_energy(singular_value, n_eff))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| format!("rank-charge stratum: {error}"))?;
    let classification = classify_reconstruction_rank(&energies, edge);
    // basis_edf = tr(gram·(gram+λS)⁻¹).
    let factor = penalized_gram(gram, lam_smooth, smooth_penalty)
        .cholesky(Side::Lower)
        .map_err(|error| {
            format!("rank-charge stratum: G + lambda*S is not positive definite: {error}")
        })?;
    let x = factor.solve_mat(gram); // X = (G+λS)⁻¹ G
    let raw_basis_edf = (0..m).map(|i| x[[i, i]]).sum::<f64>();
    let basis_edf = certified_basis_edf(raw_basis_edf, m, "rank-charge stratum")?;
    Ok(RankChargeStratum {
        energies,
        edge,
        effective_sample_size: n_eff,
        basis_edf,
        classification,
    })
}

fn penalized_gram(
    gram: &Array2<f64>,
    lam_smooth: f64,
    smooth_penalty: Option<&Array2<f64>>,
) -> Array2<f64> {
    let mut penalized = gram.clone();
    if let Some(penalty) = smooth_penalty {
        penalized.scaled_add(lam_smooth, penalty);
    }
    penalized
}

/// The conditional noise-only law of an atom's reconstruction energies.
///
/// Hold the design `Φ`, gates `a`, smoothing `(λ, S)` and dispersion `R` fixed,
/// and let the target be independent `N(0, R)` noise in every output channel.
/// The ridge decoder gives `M = G^½ B̂ = √R · Q Z` with standard normal `Z` and
/// `Q Qᵀ = C`, so `M Mᵀ ~ Wishart_m(p, R·C)` and `μ_j = σ_j(M)² / N_eff`. With
/// `K = G^½ (G+λS)⁻¹ G^½` one has `C = K²`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ConditionalNoiseNull {
    /// `E[Σ_j μ_j] = R · p · tr(C) / N_eff`, exact because `E[M Mᵀ] = R · p · C`.
    pub expected_total_energy: f64,
    /// `R · [(√(p‖C‖) + √tr C)² + ‖C‖] / N_eff ≥ E[max_j μ_j]`. Gordon's comparison
    /// inequality gives `E σ_max(Q Z) ≤ ‖Q‖ √p + ‖Q‖_F`, and `σ_max(Q Z)` is
    /// `‖Q‖`-Lipschitz in `Z`, so its variance is at most `‖Q‖²` (Gaussian Poincaré
    /// inequality), with `‖Q‖² = ‖C‖` and `‖Q‖_F² = tr C`.
    pub top_energy_expectation_bound: f64,
}

/// Evaluate [`ConditionalNoiseNull`] for one atom from its weighted basis Gram
/// `gram = Φᵀdiag(a²)Φ`, occupancy `n_eff`, output width `p_out`, dispersion
/// `r_floor` and smoothing `(lam_smooth, smooth_penalty)`.
pub fn conditional_noise_null(
    gram: &Array2<f64>,
    n_eff: f64,
    p_out: usize,
    r_floor: f64,
    lam_smooth: f64,
    smooth_penalty: Option<&Array2<f64>>,
) -> Result<ConditionalNoiseNull, String> {
    let m = gram.nrows();
    validate_rank_charge_problem(
        gram,
        &Array2::<f64>::zeros((m, p_out)),
        n_eff,
        p_out as f64,
        r_floor,
        lam_smooth,
        smooth_penalty,
    )?;
    if !(n_eff > 0.0) {
        return Err(format!(
            "conditional noise null: needs a positive occupancy N_eff; got {n_eff}"
        ));
    }
    if m == 0 {
        return Ok(ConditionalNoiseNull {
            expected_total_energy: 0.0,
            top_energy_expectation_bound: 0.0,
        });
    }
    let (evals, u) = gram
        .eigh(Side::Lower)
        .map_err(|e| format!("conditional noise null: eigh(G): {e}"))?;
    let evals = certified_psd_spectrum(evals.view(), "rank-charge Gram")?;
    let mut root = u.clone();
    for (col, &value) in evals.iter().enumerate() {
        let s = value.sqrt();
        root.column_mut(col).mapv_inplace(|entry| entry * s);
    }
    let root = root.dot(&u.t());
    let factor = penalized_gram(gram, lam_smooth, smooth_penalty)
        .cholesky(Side::Lower)
        .map_err(|error| {
            format!("conditional noise null: G + lambda*S is not positive definite: {error}")
        })?;
    let shrinkage = symmetric_part(root.dot(&factor.solve_mat(&root)).view());
    let (kappa, _) = shrinkage
        .eigh(Side::Lower)
        .map_err(|e| format!("conditional noise null: eigh(K): {e}"))?;
    let kappa = certified_psd_spectrum(kappa.view(), "conditional noise-null shrinkage")?;
    let trace_c: f64 = kappa.iter().map(|value| value * value).sum();
    let norm_c = kappa
        .iter()
        .fold(0.0_f64, |largest, &value| largest.max(value * value));
    let p = p_out as f64;
    Ok(ConditionalNoiseNull {
        expected_total_energy: r_floor * p * trace_c / n_eff,
        top_energy_expectation_bound: r_floor
            * (((p * norm_c).sqrt() + trace_c.sqrt()).powi(2) + norm_c)
            / n_eff,
    })
}

/// `E_β[L(w)] − L(ŵ)` of a quadratic loss under its tempered Gaussian posterior,
/// split into its two terms.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TemperedGaussianExcessLoss {
    /// Inverse temperature `β` the likelihood is raised to.
    pub inverse_temperature: f64,
    /// `½ (m_β − ŵ)ᵀ H (m_β − ŵ)`, the squared posterior-mean displacement.
    pub mean_displacement: f64,
    /// `½ tr(H P_β⁻¹)`, the posterior spread.
    pub posterior_spread: f64,
}

impl TemperedGaussianExcessLoss {
    /// `mean_displacement + posterior_spread`.
    pub fn total(&self) -> f64 {
        self.mean_displacement + self.posterior_spread
    }
}

fn validate_inverse_temperature(inverse_temperature: f64) -> Result<(), String> {
    if inverse_temperature.is_finite() && inverse_temperature > 0.0 {
        Ok(())
    } else {
        Err(format!(
            "tempered posterior: inverse temperature must be finite and positive; got {inverse_temperature}"
        ))
    }
}

fn symmetric_part(matrix: ArrayView2<'_, f64>) -> Array2<f64> {
    (&matrix + &matrix.t()) * 0.5
}

fn certify_psd(matrix: &Array2<f64>, name: &str) -> Result<(), String> {
    let (eigenvalues, _) = matrix
        .eigh(Side::Lower)
        .map_err(|error| format!("{name} eigendecomposition: {error}"))?;
    certified_psd_spectrum(eigenvalues.view(), name)?;
    Ok(())
}

/// Exact tempered posterior excess loss of `L(w) = L(ŵ) + ½ (w − ŵ)ᵀ H (w − ŵ)`
/// with prior `N(μ₀, Λ⁻¹)` at inverse temperature `β` (see the module docs).
///
/// `information` is `H`, `minimizer` is `ŵ`, `prior_precision` is `Λ` and
/// `prior_mean` is `μ₀`. Only the symmetric parts of `H` and `Λ` enter the
/// quadratic forms, and both must be positive semidefinite. `Λ` may be singular;
/// an improper tempered posterior (`βH + Λ` not positive definite) is refused.
pub fn tempered_gaussian_excess_loss(
    information: ArrayView2<'_, f64>,
    prior_precision: ArrayView2<'_, f64>,
    prior_mean: ArrayView1<'_, f64>,
    minimizer: ArrayView1<'_, f64>,
    inverse_temperature: f64,
) -> Result<TemperedGaussianExcessLoss, String> {
    let dim = minimizer.len();
    if information.dim() != (dim, dim)
        || prior_precision.dim() != (dim, dim)
        || prior_mean.len() != dim
    {
        return Err(format!(
            "tempered Gaussian excess loss: dimension mismatch (information {:?}, prior \
             precision {:?}, prior mean {}, minimizer {dim})",
            information.dim(),
            prior_precision.dim(),
            prior_mean.len()
        ));
    }
    validate_inverse_temperature(inverse_temperature)?;
    if information
        .iter()
        .chain(prior_precision.iter())
        .chain(prior_mean.iter())
        .chain(minimizer.iter())
        .any(|value| !value.is_finite())
    {
        return Err("tempered Gaussian excess loss: inputs must be finite".to_string());
    }
    if dim == 0 {
        return Ok(TemperedGaussianExcessLoss {
            inverse_temperature,
            mean_displacement: 0.0,
            posterior_spread: 0.0,
        });
    }
    let information = symmetric_part(information);
    let prior_precision = symmetric_part(prior_precision);
    certify_psd(&information, "tempered posterior loss information")?;
    certify_psd(&prior_precision, "tempered posterior prior precision")?;
    let precision = &information * inverse_temperature + &prior_precision;
    let factor = precision.cholesky(Side::Lower).map_err(|error| {
        format!(
            "tempered Gaussian excess loss: beta*H + Lambda is not positive definite, so the \
             tempered posterior is improper: {error}"
        )
    })?;
    let offset = (&prior_mean - &minimizer).insert_axis(Axis(1));
    let displacement = factor.solve_mat(&prior_precision.dot(&offset));
    let mean_displacement = 0.5 * (&displacement * &information.dot(&displacement)).sum();
    let spread = factor.solve_mat(&information);
    let posterior_spread = 0.5 * (0..dim).map(|i| spread[[i, i]]).sum::<f64>();
    Ok(TemperedGaussianExcessLoss {
        inverse_temperature,
        mean_displacement,
        posterior_spread,
    })
}

/// Exact tempered posterior excess loss of one atom's decoder block, conditional
/// on coordinates, gates, dispersion and the other atoms.
///
/// The conditional negative log-likelihood of the `m×p` decoder `B` is
/// `(1/2R) Σ_i ‖y_i − a_i Bᵀφ_i‖²`, with `y` the target minus the other atoms'
/// reconstruction. Each output channel `j` is quadratic with information `G / R`
/// and minimizer `ŵ_j = G⁻¹ b_j`, where `score = b = Φᵀ diag(a) y`. The smoothing
/// prior has precision `λS / R` and mean zero. Channels are independent, so
///
/// ```text
/// mean_displacement = ½ Σ_j (m_j − ŵ_j)ᵀ G (m_j − ŵ_j) / R,   m_j − ŵ_j = −(βG + λS)⁻¹ λ S ŵ_j
/// posterior_spread  = ½ · p · tr((βG + λS)⁻¹ G)
/// ```
///
/// A singular `G` leaves `ŵ` unidentified and is refused.
pub fn conditional_decoder_tempered_posterior(
    gram: &Array2<f64>,
    score: &Array2<f64>,
    dispersion: f64,
    lam_smooth: f64,
    smooth_penalty: Option<&Array2<f64>>,
    inverse_temperature: f64,
) -> Result<TemperedGaussianExcessLoss, String> {
    let m = gram.nrows();
    if gram.ncols() != m || score.nrows() != m {
        return Err(format!(
            "conditional decoder tempered posterior: Gram {:?} and score {:?} do not share \
             the basis dimension",
            gram.dim(),
            score.dim()
        ));
    }
    if let Some(penalty) = smooth_penalty {
        if penalty.dim() != (m, m) {
            return Err(format!(
                "conditional decoder tempered posterior: smoothing penalty {:?} does not match \
                 the Gram ({m}, {m})",
                penalty.dim()
            ));
        }
    }
    if !(dispersion.is_finite() && dispersion > 0.0) {
        return Err(format!(
            "conditional decoder tempered posterior: dispersion must be finite and positive; \
             got {dispersion}"
        ));
    }
    if !(lam_smooth.is_finite() && lam_smooth >= 0.0) {
        return Err(format!(
            "conditional decoder tempered posterior: smoothing weight must be finite and \
             non-negative; got {lam_smooth}"
        ));
    }
    validate_inverse_temperature(inverse_temperature)?;
    if gram
        .iter()
        .chain(score.iter())
        .chain(smooth_penalty.into_iter().flat_map(|penalty| penalty.iter()))
        .any(|value| !value.is_finite())
    {
        return Err("conditional decoder tempered posterior: inputs must be finite".to_string());
    }
    let p = score.ncols();
    if m == 0 || p == 0 {
        return Ok(TemperedGaussianExcessLoss {
            inverse_temperature,
            mean_displacement: 0.0,
            posterior_spread: 0.0,
        });
    }
    let gram = symmetric_part(gram.view());
    let minimizer = gram
        .cholesky(Side::Lower)
        .map_err(|error| {
            format!(
                "conditional decoder tempered posterior: G is not positive definite, so the \
                 decoder minimizer is not identified: {error}"
            )
        })?
        .solve_mat(score);
    let prior = match smooth_penalty {
        Some(penalty) => {
            let penalty = symmetric_part(penalty.view());
            certify_psd(&penalty, "conditional decoder smoothing penalty")?;
            &penalty * lam_smooth
        }
        None => Array2::<f64>::zeros((m, m)),
    };
    let precision = &gram * inverse_temperature + &prior;
    let factor = precision.cholesky(Side::Lower).map_err(|error| {
        format!(
            "conditional decoder tempered posterior: beta*G + lambda*S is not positive \
             definite: {error}"
        )
    })?;
    let displacement = -factor.solve_mat(&prior.dot(&minimizer));
    let mean_displacement = 0.5 * (&displacement * &gram.dot(&displacement)).sum() / dispersion;
    let spread = factor.solve_mat(&gram);
    let posterior_spread = 0.5 * p as f64 * (0..m).map(|i| spread[[i, i]]).sum::<f64>();
    Ok(TemperedGaussianExcessLoss {
        inverse_temperature,
        mean_displacement,
        posterior_spread,
    })
}

/// One atom's rank-charge audit at one evaluated state; see
/// `SaeManifoldTerm::rank_charge_audit`.
#[derive(Clone, Debug)]
pub struct AtomRankChargeAudit {
    /// Atom index.
    pub atom: usize,
    /// Basis size `m`.
    pub basis_dim: usize,
    /// Output width `p`.
    pub output_dim: usize,
    /// Storage dimension of the decoder, `m · p`.
    pub storage_dim: usize,
    /// Tangent dimension of the atom's coordinate manifold. An ambient `S²` or
    /// `RP²` coordinate stores three numbers for two dimensions, so this can be
    /// smaller than the coordinate width.
    pub intrinsic_dim: usize,
    /// Reconstruction dispersion `R`, shared by the MP edge and the decoder
    /// likelihood.
    pub dispersion: f64,
    /// Smoothing weight `λ` of the atom.
    pub lambda_smooth: f64,
    /// `β = 1 / ln N_eff`, the inverse temperature of [`Self::tempered_posterior`].
    pub inverse_temperature: f64,
    /// The priced branch: energies, edge, occupancy, basis EDF, both integer ranks
    /// and the production charge.
    pub stratum: RankChargeStratum,
    /// The direction closest to the edge and the jump its crossing adds.
    pub nearest_mp_boundary: Option<MpEdgeBoundary>,
    /// Conditional noise-only law of the reconstruction energies.
    pub noise_null: ConditionalNoiseNull,
    /// Exact conditional tempered posterior excess loss of the decoder block.
    pub tempered_posterior: TemperedGaussianExcessLoss,
    /// Signed `stratum.production_charge() − tempered_posterior.total()`, both in
    /// nats. No sign is assumed: the two price different models.
    pub production_minus_tempered: f64,
}
