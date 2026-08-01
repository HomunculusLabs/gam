//! Constant-curvature (`M_κ`) smooth term: basis + penalty over the
//! κ-stereographic chart (#944, stage 3 step 1).
//!
//! The term is the κ-generic sibling of the intrinsic-S² Wahba smooth
//! (`sphere_spec.rs` / `build_spherical_spline_basis`): a reproducing-kernel
//! basis on a center set, with the kernel Gram on the centers as the RKHS
//! roughness penalty and a coefficient-space sum-to-zero constraint for
//! identifiability. Where the Wahba smooth hard-codes S² (lat/lon chart,
//! Legendre kernels), this term takes the geometry from
//! [`gam_geometry::constant_curvature::ConstantCurvature`] at an explicit
//! curvature κ, so one construction covers the whole interpolation
//! `S^d(1/√κ) → ℝ^d → H^d(1/√−κ)` through κ = 0.
//!
//! # Kernel
//!
//! `K_κ(x, y) = exp(−d_κ(x, y) / ℓ)` — the geodesic-exponential kernel, where
//! `d_κ` is the exact constant-curvature geodesic distance in the
//! κ-stereographic chart. The geodesic distance is a kernel of conditionally
//! negative type on all three constant-curvature space forms (Schoenberg 1942
//! for `S^d`; classical CND of `‖·‖` on `ℝ^d`; Faraut–Harzallah 1974 for
//! `H^d`), so `exp(−c·d_κ)` is positive definite for every `c > 0` and every
//! κ — the Gram on distinct centers is strictly PD, which is exactly what the
//! RKHS penalty construction needs. At κ = 0 the chart carries the doubled
//! gauge (`metric 4δ`, `d_0(x, y) = 2‖x − y‖`), so the κ = 0 term is the
//! Euclidean exponential (Matérn-½) kernel smooth with effective Euclidean
//! range `ℓ/2`.
//!
//! # κ-differentiability contract (what the ψ-channel stage consumes)
//!
//! Every κ-moving piece of this construction is differentiable in κ via the
//! exact κ-jets landed in stage 2, and every κ-FIXED piece is documented as
//! such so the later ψ-channel wiring (`∂X/∂κ`, `∂S/∂κ` into the LAML outer
//! gradient, Matérn iso-κ optimizer as the template) needs no new calculus:
//!
//! - **Centers are κ-fixed.** Center selection runs in chart coordinates
//!   (farthest-point / k-means / user-provided) and deliberately does NOT
//!   consult κ, so `∂(centers)/∂κ ≡ 0` and the design moves with κ only
//!   through the kernel. A κ-dependent center rule would add an
//!   uncontrolled, non-smooth term to the design drift.
//! - **The length scale ℓ is κ-fixed.** The auto-initialized ℓ is derived
//!   from chart-coordinate (κ = 0 gauge) center spacing only, and an
//!   explicit user ℓ is a constant. `∂ℓ/∂κ ≡ 0`.
//! - **The constraint transform `z` is κ-fixed.** Uniform coefficient
//!   weights; at fit time the global identifiability pipeline composes the
//!   parametric orthogonalization onto it and the result is FROZEN
//!   (mirroring `SphericalSplineIdentifiability::FrozenTransform`, #532), so
//!   the predict/ψ-trial rebuild replays the same `z` verbatim.
//! - **The kernel has exact κ-jets.** `∂K/∂κ` and `∂²K/∂κ²` follow from
//!   `distance_kappa_jet` (Tower4-exact, FD-gated) by the chain rule — see
//!   [`constant_curvature_kernel_kappa_jets`]. Therefore:
//!   `∂X_raw/∂κ = ∂K(data, centers)/∂κ`, realized design drift
//!   `∂X/∂κ = (∂K/∂κ)·z`, and penalty drift `∂S_raw/∂κ = zᵀ(∂K(centers,
//!   centers)/∂κ)z` are all available in closed form from this module today.
//!   (The penalty handed to the optimizer is Frobenius-normalized; the
//!   ψ-channel must route its κ-derivative through the same normalization
//!   rule — `normalize_penaltywith_psi_derivatives` is the existing seam.)
//! - **Available but not yet consumed:** `log_map_kappa_jet` /
//!   `exp_map_kappa_jet` cover future geodesic/normal-coordinate basis
//!   variants (e.g. tangent-space designs); the distance jet is the only one
//!   this kernel construction needs.

use ndarray::{Array1, Array2, ArrayView2, Axis};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use gam_geometry::constant_curvature::{ConstantCurvature, distance_kappa_jet};

use super::{
    ActivePenalty, BasisBuildResult, BasisError, BasisMetadata, BasisPsiDerivativeBundle,
    BasisPsiDerivativeResult, BasisPsiSecondDerivativeResult, CenterStrategy, CenterStrategyKind,
    ConstructiveQuadratic, PenaltyCandidate, PenaltySource, center_strategy_kind,
    filter_penalty_candidates, normalize_penalty, select_centers_by_strategy,
    weighted_coefficient_sum_to_zero_transform,
};

/// Realized-design identifiability policy for the constant-curvature smooth.
/// Mirrors [`super::SphericalSplineIdentifiability`] (#532): the fit-time
/// center-space sum-to-zero `z` gets the parametric orthogonalization composed
/// onto it by the global identifiability pipeline, and the composed transform
/// is frozen here so predict-time (and future per-ψ-trial) rebuilds replay it
/// verbatim instead of recomputing `z` from the centers.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum ConstantCurvatureIdentifiability {
    /// Fit-time default: uniform-weight coefficient sum-to-zero over the
    /// centers (`Σ_j α_j = 0`), then global parametric residualization.
    #[default]
    CenterSumToZero,
    /// Predict-time replay: the frozen composed transform captured at fit
    /// time. `transform.nrows()` equals the number of centers.
    FrozenTransform { transform: Array2<f64> },
}

/// Constant-curvature smooth configuration (`curv(x, z, kappa = …)`).
///
/// The chart inputs are the raw feature columns interpreted as
/// κ-stereographic chart coordinates: any finite point for κ ≥ 0, the open
/// ball `‖x‖ < 1/√(−κ)` for κ < 0. The default κ = 0 reproduces a Euclidean
/// exponential-kernel smooth (in the doubled κ = 0 chart gauge), so the term
/// is safe to use as a drop-in flat smooth until κ becomes a fitted
/// ψ-coordinate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstantCurvatureBasisSpec {
    /// Center/knot selection strategy in chart coordinates. Deliberately
    /// κ-independent (see the module-level κ-contract).
    pub center_strategy: CenterStrategy,
    /// Sectional curvature κ of the latent/feature geometry. Fixed at build
    /// time; when [`Self::kappa_fixed`] is `false` the later ψ-channel stage
    /// promotes it to a fitted outer coordinate consuming this module's exact
    /// κ-jets, and this field is only the seed. When `kappa_fixed` is `true`
    /// this value is the user's PINNED sectional curvature and the outer loop
    /// must hold it constant (never re-derive it).
    pub kappa: f64,
    /// Did the user explicitly pin the sectional curvature (`curv(.., kappa=K)`)?
    ///
    /// This is the mgcv-`sp=` convention applied to κ: an explicit `kappa=`
    /// selects a FIXED geometry (`Sᵈ` for κ>0, `ℝᵈ` for κ=0, `Hᵈ` for κ<0) and
    /// the fit builds/keeps the design and penalty at exactly that κ; an OMITTED
    /// `kappa=` leaves κ free, seeded at [`Self::kappa`] (default 0), for the
    /// #944/#1464 outer ψ-coordinate estimation to fit. The two paths must never
    /// be confused: honoring the pin is the whole contract of a fixed-curvature
    /// smooth (gam#2152), while the estimation path is the whole point of the
    /// κ-inference subsystem. Defaults to `false` (estimate) so the estimand
    /// machinery and every serialized pre-#2152 model keep their behaviour.
    #[serde(default)]
    pub kappa_fixed: bool,
    /// Geodesic kernel range ℓ in `K_κ = exp(−d_κ/ℓ)`. The `0.0` sentinel
    /// requests the κ-independent auto initialization
    /// ([`realized_constant_curvature_length_scale`]); the realized value is
    /// persisted in [`BasisMetadata::ConstantCurvature`] and frozen back into
    /// the spec for predict-time replay.
    pub length_scale: f64,
    /// Add the ridge-like shrinkage penalty alongside the RKHS Gram penalty.
    pub double_penalty: bool,
    /// Realized-design identifiability policy (see type docs).
    #[serde(default)]
    pub identifiability: ConstantCurvatureIdentifiability,
}

impl Default for ConstantCurvatureBasisSpec {
    fn default() -> Self {
        Self {
            center_strategy: CenterStrategy::FarthestPoint { num_centers: 50 },
            kappa: 0.0,
            kappa_fixed: false,
            length_scale: 0.0,
            // No double-penalty ridge by default (#1464). The RKHS Gram penalty
            // zᵀKz is strictly PD/full-rank on distinct centers, so it already
            // regularizes every coefficient direction — the ridge `I` adds no
            // stability. Worse, `I` is curvature-BLIND: with its own λ it absorbs
            // the data fit independently of κ. Curvature-sign identification
            // remains the separate #1464 problem; an extra curvature-blind
            // ridge cannot resolve it.
            double_penalty: false,
            identifiability: ConstantCurvatureIdentifiability::CenterSumToZero,
        }
    }
}

/// Validate that every row of `points` is finite and inside the
/// κ-stereographic chart (`1 + κ‖x‖² > 0`; automatic for κ ≥ 0, the open-ball
/// constraint for κ < 0).
pub(crate) fn validate_chart_points(
    points: ArrayView2<'_, f64>,
    kappa: f64,
    what: &str,
) -> Result<(), BasisError> {
    for (i, row) in points.outer_iter().enumerate() {
        let mut nx2 = 0.0_f64;
        for &v in row.iter() {
            if !v.is_finite() {
                crate::bail_invalid_basis!(
                    "constant-curvature {what} row {i} has a non-finite coordinate"
                );
            }
            nx2 += v * v;
        }
        if 1.0 + kappa * nx2 <= 0.0 {
            crate::bail_invalid_basis!(
                "constant-curvature {what} row {i} lies outside the κ-stereographic chart \
                 (need 1 + κ·‖x‖² > 0; got κ = {kappa}, ‖x‖² = {nx2}); for κ < 0 the chart is \
                 the open ball ‖x‖ < 1/√(−κ)"
            );
        }
    }
    Ok(())
}

/// `K_κ(data, centers)` — the geodesic-exponential kernel matrix
/// `exp(−d_κ(x_i, c_j)/ℓ)`.
pub fn constant_curvature_kernel_matrix(
    data: ArrayView2<'_, f64>,
    centers: ArrayView2<'_, f64>,
    kappa: f64,
    length_scale: f64,
) -> Result<Array2<f64>, BasisError> {
    if data.ncols() != centers.ncols() {
        crate::bail_dim_basis!(
            "constant-curvature kernel dimension mismatch: data d={} centers d={}",
            data.ncols(),
            centers.ncols()
        );
    }
    if !(length_scale.is_finite() && length_scale > 0.0) {
        crate::bail_invalid_basis!(
            "constant-curvature kernel needs a positive finite length_scale; got {length_scale}"
        );
    }
    validate_chart_points(data, kappa, "data")?;
    validate_chart_points(centers, kappa, "centers")?;
    let manifold = ConstantCurvature::new(data.ncols(), kappa);
    let mut out = Array2::<f64>::zeros((data.nrows(), centers.nrows()));
    out.axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .try_for_each(|(i, mut row)| -> Result<(), BasisError> {
            for (j, c) in centers.outer_iter().enumerate() {
                let d = manifold.distance(data.row(i), c).map_err(|e| {
                    BasisError::InvalidInput(format!(
                        "constant-curvature distance failed at (row {i}, center {j}): {e}"
                    ))
                })?;
                row[j] = (-d / length_scale).exp();
            }
            Ok(())
        })?;
    Ok(out)
}

/// `(K, ∂K/∂κ, ∂²K/∂κ²)` of the raw (pre-constraint) kernel matrix — the
/// ψ-channel hook. Exact: rides `distance_kappa_jet` (Tower4, FD-gated in
/// `geometry::constant_curvature`) through the chain rule for
/// `K = exp(−d/ℓ)` at κ-FIXED ℓ and centers (see the module κ-contract):
///
/// ```text
///   ∂K/∂κ  = −(d′/ℓ) · K
///   ∂²K/∂κ² = ((d′/ℓ)² − d″/ℓ) · K
/// ```
///
/// The realized design/penalty drifts follow by the κ-fixed transforms:
/// `∂X/∂κ = (∂K/∂κ)·z` and `∂S_raw/∂κ = zᵀ(∂K/∂κ)z` (centers×centers), with
/// the Frobenius penalty normalization differentiated by the existing
/// `normalize_penaltywith_psi_derivatives` seam.
pub fn constant_curvature_kernel_kappa_jets(
    data: ArrayView2<'_, f64>,
    centers: ArrayView2<'_, f64>,
    kappa: f64,
    length_scale: f64,
) -> Result<(Array2<f64>, Array2<f64>, Array2<f64>), BasisError> {
    if data.ncols() != centers.ncols() {
        crate::bail_dim_basis!(
            "constant-curvature kernel-jet dimension mismatch: data d={} centers d={}",
            data.ncols(),
            centers.ncols()
        );
    }
    if !(length_scale.is_finite() && length_scale > 0.0) {
        crate::bail_invalid_basis!(
            "constant-curvature kernel jets need a positive finite length_scale; got {length_scale}"
        );
    }
    validate_chart_points(data, kappa, "data")?;
    validate_chart_points(centers, kappa, "centers")?;
    let manifold = ConstantCurvature::new(data.ncols(), kappa);
    let n = data.nrows();
    let m = centers.nrows();
    let mut value = Array2::<f64>::zeros((n, m));
    let mut dk = Array2::<f64>::zeros((n, m));
    let mut dkk = Array2::<f64>::zeros((n, m));
    let rows: Vec<(usize, Vec<(f64, f64, f64)>)> = (0..n)
        .into_par_iter()
        .map(|i| -> Result<(usize, Vec<(f64, f64, f64)>), BasisError> {
            let mut row = Vec::with_capacity(m);
            for (j, c) in centers.outer_iter().enumerate() {
                let (d, d1, d2) = distance_kappa_jet(&manifold, data.row(i), c).map_err(|e| {
                    BasisError::InvalidInput(format!(
                        "constant-curvature distance κ-jet failed at (row {i}, center {j}): {e}"
                    ))
                })?;
                let k = (-d / length_scale).exp();
                let g = d1 / length_scale;
                row.push((k, -g * k, (g * g - d2 / length_scale) * k));
            }
            Ok((i, row))
        })
        .collect::<Result<Vec<_>, BasisError>>()?;
    for (i, row) in rows {
        for (j, (k, k1, k2)) in row.into_iter().enumerate() {
            value[(i, j)] = k;
            dk[(i, j)] = k1;
            dkk[(i, j)] = k2;
        }
    }
    Ok((value, dk, dkk))
}

/// `(K, ∂K/∂κ, ∂²K/∂κ²)` of the raw kernel matrix when the kernel uses the
/// fill-invariant effective length `L(κ)` (the #944 fix: `L` solves the fill
/// target `g(L,κ)=fill⋆`, holding the kernel's effective DoF κ-invariant). Both
/// the geodesic distance `d_κ` and the length `L(κ)` move with κ, so the exponent
/// is the quotient `q = d/L` and the chain rule carries both jets:
///
/// ```text
///   q  = d / L
///   q′ = d′/L − d·L′/L²
///   q″ = d″/L − 2 d′ L′/L² − d L″/L² + 2 d (L′)²/L³
///   K = e^{−q},  K′ = −q′K,  K″ = ((q′)² − q″) K
/// ```
///
/// `l_jet = (L, L′, L″)` is the effective-length κ-jet from
/// [`constant_curvature_effective_length_jet`]; at κ = 0 it reduces to the
/// fixed-ℓ jets (`L′ = L″` terms vanish only if the geometry is flat, but the
/// formula is exact for all κ).
pub(crate) fn constant_curvature_kernel_kappa_jets_scaled(
    data: ArrayView2<'_, f64>,
    centers: ArrayView2<'_, f64>,
    kappa: f64,
    l_jet: (f64, f64, f64),
) -> Result<(Array2<f64>, Array2<f64>, Array2<f64>), BasisError> {
    if data.ncols() != centers.ncols() {
        crate::bail_dim_basis!(
            "constant-curvature scaled kernel-jet dimension mismatch: data d={} centers d={}",
            data.ncols(),
            centers.ncols()
        );
    }
    let (l, l1, l2) = l_jet;
    if !(l.is_finite() && l > 0.0) {
        crate::bail_invalid_basis!(
            "constant-curvature scaled kernel jets need a positive finite effective length; got {l}"
        );
    }
    validate_chart_points(data, kappa, "data")?;
    validate_chart_points(centers, kappa, "centers")?;
    let manifold = ConstantCurvature::new(data.ncols(), kappa);
    let n = data.nrows();
    let m = centers.nrows();
    let mut value = Array2::<f64>::zeros((n, m));
    let mut dk = Array2::<f64>::zeros((n, m));
    let mut dkk = Array2::<f64>::zeros((n, m));
    let rows: Vec<(usize, Vec<(f64, f64, f64)>)> = (0..n)
        .into_par_iter()
        .map(|i| -> Result<(usize, Vec<(f64, f64, f64)>), BasisError> {
            let mut row = Vec::with_capacity(m);
            for (j, c) in centers.outer_iter().enumerate() {
                let (d, d1, d2) = distance_kappa_jet(&manifold, data.row(i), c).map_err(|e| {
                    BasisError::InvalidInput(format!(
                        "constant-curvature scaled distance κ-jet failed at (row {i}, center {j}): {e}"
                    ))
                })?;
                let q = d / l;
                let q1 = d1 / l - d * l1 / (l * l);
                let q2 = d2 / l - 2.0 * d1 * l1 / (l * l) - d * l2 / (l * l)
                    + 2.0 * d * l1 * l1 / (l * l * l);
                let k = (-q).exp();
                row.push((k, -q1 * k, (q1 * q1 - q2) * k));
            }
            Ok((i, row))
        })
        .collect::<Result<Vec<_>, BasisError>>()?;
    for (i, row) in rows {
        for (j, (k, k1, k2)) in row.into_iter().enumerate() {
            value[(i, j)] = k;
            dk[(i, j)] = k1;
            dkk[(i, j)] = k2;
        }
    }
    Ok((value, dk, dkk))
}

/// Resolve the realized kernel range ℓ. An explicit positive `spec_length_scale`
/// is used verbatim; the `0.0` sentinel auto-initializes from the median
/// pairwise CHART distance among the centers, doubled to match the κ = 0
/// chart gauge (`d_0 = 2‖Δ‖`).
///
/// κ-contract: the auto rule reads chart coordinates only — it never consults
/// κ — so the realized ℓ is a κ-CONSTANT and contributes no `∂ℓ/∂κ` term to
/// the design drift.
pub fn realized_constant_curvature_length_scale(
    centers: ArrayView2<'_, f64>,
    spec_length_scale: f64,
) -> Result<f64, BasisError> {
    if spec_length_scale.is_finite() && spec_length_scale > 0.0 {
        return Ok(spec_length_scale);
    }
    if spec_length_scale != 0.0 {
        crate::bail_invalid_basis!(
            "constant-curvature length_scale must be positive (or 0.0 for auto); got {spec_length_scale}"
        );
    }
    let m = centers.nrows();
    if m < 2 {
        return Err(BasisError::InsufficientColumnsForConstraint { found: m });
    }
    let mut dists: Vec<f64> = Vec::with_capacity(m * (m - 1) / 2);
    for i in 0..m {
        for j in (i + 1)..m {
            let mut s = 0.0_f64;
            for k in 0..centers.ncols() {
                let dlt = centers[(i, k)] - centers[(j, k)];
                s += dlt * dlt;
            }
            dists.push(2.0 * s.sqrt());
        }
    }
    dists.sort_by(|a, b| a.partial_cmp(b).expect("finite chart distances"));
    let median = dists[dists.len() / 2];
    if !(median.is_finite() && median > 0.0) {
        crate::bail_invalid_basis!(
            "constant-curvature auto length_scale failed: centers are degenerate \
             (median pairwise chart distance = {median})"
        );
    }
    Ok(median)
}

/// Reference kernel "fill" `fill⋆` — the κ = 0 mean data→center kernel entry
/// `(1/N) Σᵢⱼ exp(−d₀(xᵢ,cⱼ)/ℓ_ref)` with `d₀ = 2‖Δ‖` the κ = 0 chart gauge.
///
/// The fill is the scalar that measures the kernel's *effective resolution* (how
/// much each data row "sees" the centers): it is monotone in `ℓ/scale`, so
/// pinning it across κ pins the realized design's flexibility (its effective
/// degrees of freedom). [`constant_curvature_effective_length_jet`] solves
/// `g(L,κ) = fill⋆` for `L(κ)` so the fill — hence the basis flexibility — stays
/// κ-invariant and only the distance-matrix SHAPE (the genuine curvature signal)
/// moves with κ. At κ = 0 the solution is `L = ℓ_ref` by construction.
pub(crate) fn data_center_reference_fill(
    data: ArrayView2<'_, f64>,
    centers: ArrayView2<'_, f64>,
    ell_ref: f64,
) -> Result<f64, BasisError> {
    if !(ell_ref.is_finite() && ell_ref > 0.0) {
        crate::bail_invalid_basis!(
            "constant-curvature reference fill needs a positive finite ℓ_ref; got {ell_ref}"
        );
    }
    let mut sum = 0.0_f64;
    let mut cnt = 0.0_f64;
    for xi in data.outer_iter() {
        for cj in centers.outer_iter() {
            let mut s = 0.0_f64;
            for k in 0..centers.ncols() {
                let dlt = xi[k] - cj[k];
                s += dlt * dlt;
            }
            let d0 = 2.0 * s.sqrt(); // κ = 0 chart gauge d₀ = 2‖Δ‖
            sum += (-d0 / ell_ref).exp();
            cnt += 1.0;
        }
    }
    if cnt <= 0.0 {
        crate::bail_invalid_basis!(
            "constant-curvature reference fill needs at least one data row and one center"
        );
    }
    Ok(sum / cnt)
}

/// The mean-kernel-entry "fill" `g(L,κ) = (1/N) Σᵢⱼ exp(−d_κ(xᵢ,cⱼ)/L)` together
/// with the five partials needed by the implicit-function jet:
/// `(g, g_L, g_κ, g_LL, g_κκ, g_Lκ)`.
///
/// With `k = exp(−d/L)` and the per-pair geodesic jet `(d, d', d'')` (exact via
/// [`distance_kappa_jet`]):
///
/// ```text
///   ∂k/∂L = k·d/L²,                  ∂k/∂κ = −k·d'/L
///   g_LL  = (1/N)Σ k·d·(d − 2L)/L⁴
///   g_κκ  = (1/N)Σ k·((d')²/L − d'')/L
///   g_Lκ  = (1/N)Σ k·d'·(L − d)/L³
/// ```
///
/// (each obtained by differentiating `∂k/∂L` / `∂k/∂κ` once more). `g` and every
/// partial are smooth through κ = 0 because the distance jet is entire there.
pub(crate) fn data_center_fill_partials(
    data: ArrayView2<'_, f64>,
    centers: ArrayView2<'_, f64>,
    kappa: f64,
    l: f64,
) -> Result<(f64, f64, f64, f64, f64, f64), BasisError> {
    if !(l.is_finite() && l > 0.0) {
        crate::bail_invalid_basis!(
            "constant-curvature fill partials need a positive finite length; got {l}"
        );
    }
    let manifold = ConstantCurvature::new(centers.ncols(), kappa);
    let l2 = l * l;
    let l3 = l2 * l;
    let l4 = l2 * l2;
    let mut g = 0.0_f64;
    let mut g_l = 0.0_f64;
    let mut g_k = 0.0_f64;
    let mut g_ll = 0.0_f64;
    let mut g_kk = 0.0_f64;
    let mut g_lk = 0.0_f64;
    let mut cnt = 0.0_f64;
    for xi in data.outer_iter() {
        for cj in centers.outer_iter() {
            let (d, d1, d2) = distance_kappa_jet(&manifold, xi, cj).map_err(|e| {
                BasisError::InvalidInput(format!(
                    "constant-curvature data→center fill κ-jet failed: {e}"
                ))
            })?;
            let k = (-d / l).exp();
            g += k;
            g_l += k * d / l2;
            g_k += -k * d1 / l;
            g_ll += k * d * (d - 2.0 * l) / l4;
            g_kk += k * ((d1 * d1) / l - d2) / l;
            g_lk += k * d1 * (l - d) / l3;
            cnt += 1.0;
        }
    }
    if cnt <= 0.0 {
        crate::bail_invalid_basis!(
            "constant-curvature fill partials need at least one data row and one center"
        );
    }
    Ok((
        g / cnt,
        g_l / cnt,
        g_k / cnt,
        g_ll / cnt,
        g_kk / cnt,
        g_lk / cnt,
    ))
}

/// Effective kernel length `L(κ)` and its EXACT κ-jet `(L, L′, L″)`.
///
/// THE κ-IDENTIFICATION FIX (#944). A κ-FROZEN length makes the geodesic-
/// exponential kernel's *resolution* drift with κ: spherical (κ>0) geometries
/// compress geodesic distances, narrowing the kernel relative to the data and
/// inflating the basis's effective flexibility, so REML buys a lower deviance by
/// cranking κ up — κ rails to the chart bound for every truth (the #944/#1059
/// symptom). The earlier #1059 fix normalized by the mean data→center geodesic
/// distance `s_dc(κ)`; but holding the mean DISTANCE fixed does NOT hold the
/// kernel's flexibility fixed — the effective degrees of freedom still drift
/// ~30% across the bracket (verified), so the deviance stayed monotone in κ.
///
/// We instead hold the kernel's "fill" — the mean realized kernel entry
/// `g(L,κ) = (1/N) Σᵢⱼ exp(−d_κ(xᵢ,cⱼ)/L)` — κ-INVARIANT, which pins the
/// realized design's effective degrees of freedom (the EDF is flat to <0.5% in κ
/// under this rule, verified numerically). `L(κ)` is the implicit solution of
///
/// ```text
///   g(L(κ), κ) = fill⋆,   fill⋆ = g(ℓ_ref, 0)   (the κ=0 reference fill)
/// ```
///
/// so changing κ moves ONLY the distance-matrix SHAPE (the genuine curvature
/// signal), giving `V_p(κ)` an interior minimum at the data-generating κ for
/// curved truth. At κ = 0 the solution is `L = ℓ_ref` exactly.
///
/// The jet is EXACT via the implicit-function theorem. Differentiating
/// `g(L(κ),κ) ≡ fill⋆` once gives `g_L·L′ + g_κ = 0`, and once more gives
/// `g_LL·(L′)² + 2 g_Lκ·L′ + g_κκ + g_L·L″ = 0`:
///
/// ```text
///   L′  = −g_κ / g_L
///   L″  = −( g_LL·(L′)² + 2 g_Lκ·L′ + g_κκ ) / g_L .
/// ```
///
/// The partials come from [`data_center_fill_partials`] (exact, riding
/// `distance_kappa_jet`); the returned jet feeds `constant_curvature_kernel_
/// kappa_jets_scaled` through the quotient `q = d/L` chain rule.
///
/// Public scalar view of the κ-invariant effective kernel length `L(κ)` that the
/// realized constant-curvature design/penalty are built at (the #944 fill-
/// invariance fix). The forward build evaluates the geodesic-exponential kernel
/// at this `L(κ)`, NOT at the κ = 0 reference length `ell_ref`, so any external
/// consumer reconstructing `K(·)` to compare against the realized design must
/// use this length. Equals `ell_ref` exactly at κ = 0.
pub fn constant_curvature_effective_length(
    data: ArrayView2<'_, f64>,
    centers: ArrayView2<'_, f64>,
    ell_ref: f64,
    kappa: f64,
) -> Result<f64, BasisError> {
    Ok(constant_curvature_effective_length_jet(data, centers, ell_ref, kappa)?.0)
}

pub(crate) fn constant_curvature_effective_length_jet(
    data: ArrayView2<'_, f64>,
    centers: ArrayView2<'_, f64>,
    ell_ref: f64,
    kappa: f64,
) -> Result<(f64, f64, f64), BasisError> {
    let fill_star = data_center_reference_fill(data, centers, ell_ref)?;
    // Newton solve g(L, κ) = fill⋆ for L, warm-started at ℓ_ref (the exact root
    // at κ = 0). g is strictly increasing in L (g_L > 0: larger L ⇒ each entry
    // closer to 1), so Newton from ℓ_ref converges monotonically.
    let mut l = ell_ref;
    const NEWTON_MAX_ITER: usize = 100;
    const NEWTON_REL_TOL: f64 = 1.0e-13;
    let mut converged = false;
    for _ in 0..NEWTON_MAX_ITER {
        let (g, g_l, ..) = data_center_fill_partials(data, centers, kappa, l)?;
        if !(g_l.is_finite() && g_l > 0.0) {
            crate::bail_invalid_basis!(
                "constant-curvature effective length: non-positive fill slope g_L = {g_l} \
                 (degenerate data/centers at κ = {kappa})"
            );
        }
        let step = (g - fill_star) / g_l;
        l -= step;
        if !(l.is_finite() && l > 0.0) {
            crate::bail_invalid_basis!(
                "constant-curvature effective length: Newton left the positive axis (L = {l}) \
                 solving the fill target at κ = {kappa}"
            );
        }
        if step.abs() <= NEWTON_REL_TOL * l {
            converged = true;
            break;
        }
    }
    if !converged {
        crate::bail_invalid_basis!(
            "constant-curvature effective length: fill-target Newton did not converge at κ = {kappa}"
        );
    }
    // Exact implicit-function-theorem jet at the converged root.
    let (_, g_l, g_k, g_ll, g_kk, g_lk) = data_center_fill_partials(data, centers, kappa, l)?;
    let l1 = -g_k / g_l;
    let l2 = -(g_ll * l1 * l1 + 2.0 * g_lk * l1 + g_kk) / g_l;
    Ok((l, l1, l2))
}

/// Build the constant-curvature reproducing-kernel smooth: realized design
/// `K_κ(data, centers)·z`, RKHS penalty `zᵀ K_κ(centers, centers) z`, and the
/// replayable [`BasisMetadata::ConstantCurvature`]. Structure mirrors the
/// Wahba S² builder (`build_spherical_spline_basis`); geometry comes from
/// `ConstantCurvature` at the spec's fixed κ.
pub fn build_constant_curvature_basis(
    data: ArrayView2<'_, f64>,
    spec: &ConstantCurvatureBasisSpec,
) -> Result<BasisBuildResult, BasisError> {
    if data.ncols() == 0 {
        crate::bail_invalid_basis!("constant-curvature smooth needs at least one feature column");
    }
    if !spec.kappa.is_finite() {
        crate::bail_invalid_basis!("constant-curvature smooth needs a finite kappa");
    }
    validate_chart_points(data, spec.kappa, "data")?;
    let centers = select_constant_curvature_centers(data, &spec.center_strategy)?;
    if centers.nrows() < 2 {
        return Err(BasisError::InsufficientColumnsForConstraint {
            found: centers.nrows(),
        });
    }
    validate_chart_points(centers.view(), spec.kappa, "centers")?;
    // ℓ_ref is the κ = 0 reference length (auto = mean chart spacing, or the
    // user/frozen value); the kernel uses the κ-invariant effective length
    // L(κ) = ℓ_ref·s(κ)/s₀ so changing κ moves the geometry, not the kernel
    // resolution (the #1059 curvature-identification fix). At κ = 0, L = ℓ_ref.
    let length_scale = realized_constant_curvature_length_scale(centers.view(), spec.length_scale)?;
    // DESIGN effective length L(κ): solved against the DATA→center fill so the
    // realized design's effective DOF stays κ-invariant (#944/#1059). The design
    // X = K(data, centers)·z is built at this L.
    let (ell_eff, _, _) =
        constant_curvature_effective_length_jet(data, centers.view(), length_scale, spec.kappa)?;
    // PENALTY effective length L_S(κ): solved against the CENTER→center fill so
    // the penalty Gram S = zᵀK(centers,centers)z has a κ-INVARIANT resolution
    // (#1464). The data→center fill that pins L(κ) does NOT pin the center→center
    // penalty spectrum, so with the single shared L the penalty pseudo-determinant
    // logdet|S|₊ drifts freely with κ: as κ grows positive the geodesic kernel
    // collapses toward the constant, the center→center Gram eigenvalues bunch /
    // drop below the rank tolerance, logdet|S|₊ falls, and the REML Occam term
    // −½·logdet|S|₊ DECREASES — rewarding the +κ collapsed-kernel corner and
    // railing κ̂ to the +chart bound for any curved data (the headline #1464
    // sign-blindness: hyperbolic truth recovered as spherical, V_p(κ) monotone in
    // κ with no interior optimum). Building the penalty at L_S(κ) holds the
    // penalty eigenvalue SHAPE (hence logdet|S|₊ and its rank) κ-comparable, so
    // the Occam term stops rewarding the collapse and V_p regains an interior
    // minimum near the data-generating κ. At κ = 0, L_S = ℓ_ref = L, so the κ = 0
    // build is byte-identical.
    let (ell_eff_penalty, _, _) = constant_curvature_effective_length_jet(
        centers.view(),
        centers.view(),
        length_scale,
        spec.kappa,
    )?;
    let raw_penalty = constant_curvature_kernel_matrix(
        centers.view(),
        centers.view(),
        spec.kappa,
        ell_eff_penalty,
    )?;
    // Realized-design constraint transform: uniform coefficient sum-to-zero at
    // fit time; the frozen composed `z · z_parametric` at predict time (#532
    // pattern — see ConstantCurvatureIdentifiability).
    let z = match &spec.identifiability {
        ConstantCurvatureIdentifiability::FrozenTransform { transform } => {
            if transform.nrows() != centers.nrows() {
                crate::bail_dim_basis!(
                    "frozen constant-curvature identifiability transform mismatch: {} centers but transform has {} rows",
                    centers.nrows(),
                    transform.nrows()
                );
            }
            transform.clone()
        }
        ConstantCurvatureIdentifiability::CenterSumToZero => {
            let weights = Array1::<f64>::ones(centers.nrows());
            weighted_coefficient_sum_to_zero_transform(weights.view())?
        }
    };
    let gauge = gam_problem::Gauge::from_block_transforms(&[z.clone()]);
    let raw_penalty = ConstructiveQuadratic::try_from_dense_psd(
        (&raw_penalty + &raw_penalty.t()) * 0.5,
        "constant-curvature raw RKHS penalty",
    )?;
    let penalty =
        raw_penalty.restricted(&gauge, "constant-curvature identifiability restriction")?;
    let raw_design = constant_curvature_kernel_matrix(data, centers.view(), spec.kappa, ell_eff)?;
    let design = gam_linalg::matrix::DesignMatrix::Dense(
        gam_linalg::matrix::DenseDesignMatrix::from(gauge.restrict_design(&raw_design)),
    );
    // Keep the RKHS penalty RAW (the symmetric kernel Gram zᵀKz) with
    // normalization_scale = 1, rather than Frobenius-normalizing it. The Gram's
    // eigenvalues ARE the physical RKHS roughness energies of each coefficient
    // direction: the smoothest functions (the low-degree / degree-1 signal) sit
    // in the genuinely tiny-eigenvalue directions, while wiggly functions sit in
    // the large ones — a spread of many orders of magnitude. Frobenius-
    // normalizing divides the whole operator by ‖·‖_F (dominated by the large
    // wiggly eigenvalues), which compresses that spread and inflates the
    // smallest eigenvalues relative to their natural scale. REML's scale-
    // sensitive λ heuristics then drive a single λ high enough to suppress the
    // wiggly directions and, because the smooth directions are no longer
    // proportionally tiny, over-shrink the recoverable low-degree signal
    // (planted degree-1 sphere harmonic recovered at only R²≈0.84). Keeping the
    // raw physical operator (scale = 1, matching the sphere-harmonic Laplace-
    // Beltrami penalty) lets REML act on true roughness, leaving the smooth
    // signal essentially unpenalized while still shrinking the wiggly tail —
    // raising recovery toward the unconstrained RKHS ceiling. The penalty stays
    // exactly proportional to zᵀKz, so the constrained-kernel-Gram contract is
    // unchanged.
    let mut candidates = vec![PenaltyCandidate {
        matrix: penalty,
        source: PenaltySource::Primary,
        normalization_scale: 1.0,
        kronecker_factors: None,
        op: None,
    }];
    if spec.double_penalty {
        // #1531: the primary here is the RKHS kernel Gram zᵀKz, which is
        // strictly PD / full-rank on distinct centers. It has no unpenalized
        // function subspace, so an explicit second shrinkage coordinate must
        // target the whole coefficient chart. The full identity is therefore
        // intentional for this basis rather than a null-space penalty.
        // The regression test `constant_curvature_gram_is_full_rank_so_identity_is_the_only_double_penalty`
        // locks the full-rank fact that justifies this branch.
        let ridge = Array2::<f64>::eye(design.ncols());
        let (ridge_norm, c_ridge) = normalize_penalty(&ridge);
        candidates.push(PenaltyCandidate {
            matrix: ConstructiveQuadratic::try_from_dense_psd(
                ridge_norm,
                "constant-curvature whole-function ridge",
            )?,
            source: PenaltySource::DoublePenaltyNullspace,
            normalization_scale: c_ridge,
            kronecker_factors: None,
            op: None,
        });
    }
    let filtered = filter_penalty_candidates(candidates)?;
    Ok(BasisBuildResult {
        design,
        affine_offset: None,
        active_penalties: filtered.active,
        dropped_penalties: filtered.dropped,
        metadata: BasisMetadata::ConstantCurvature {
            centers,
            kappa: spec.kappa,
            length_scale,
            constraint_transform: Some(z),
        },
        kronecker_factored: None,
        joint_null_rotation: None,
    })
}

/// Select constant-curvature centers.
///
/// Upper bound on `max‖c‖²`, the largest squared chart radius among the centers
/// [`select_constant_curvature_centers`] will return for `strategy` on `data` —
/// computed WITHOUT materializing them.
///
/// Every κ bound is denominated in a chart radius, and the set that radius must
/// be taken over is the one the kernel EVALUATES: `K_κ` calls
/// `ConstantCurvature::distance(x, c)` for each (data row, center) pair and
/// `validate_chart_points` checks data **and** centers. Taking the radius over
/// `data` alone is only correct while every center is inside the data hull
/// (gam#2716). Two strategies break that:
///
/// * [`CenterStrategy::UserProvided`] — left verbatim by
///   [`select_constant_curvature_centers`], so a center may sit at any radius.
/// * [`CenterStrategy::UniformGrid`] — the Cartesian product of per-axis
///   linspaces over the data's *bounding box*, so a corner center sits at the
///   bounding box corner, radius up to `√d·max‖x‖`, outside the hull for `d ≥ 2`.
///
/// Every other strategy assigns either a data row verbatim (equal-mass leaves,
/// farthest point) or a convex combination of data rows (k-means centroids), so
/// `max‖c‖ ≤ max‖x‖` and this returns exactly the data radius — which is what
/// makes the κ box bit-identical to its pre-#2716 value on every data-driven
/// strategy. The origin-snap in [`select_constant_curvature_centers`] only ever
/// moves a center toward the origin, so it cannot invalidate an upper bound.
///
/// The match is exhaustive rather than wildcarded: a new strategy has to state
/// its own radius law instead of silently inheriting a wrong one.
pub fn constant_curvature_center_chart_radius2(
    data: ArrayView2<'_, f64>,
    feature_cols: &[usize],
    strategy: &CenterStrategy,
) -> f64 {
    match strategy {
        CenterStrategy::Auto(inner) => {
            constant_curvature_center_chart_radius2(data, feature_cols, inner)
        }
        CenterStrategy::DuchonSpectral { knots, .. } => {
            constant_curvature_center_chart_radius2(data, feature_cols, knots)
        }
        CenterStrategy::UserProvided(centers) => {
            let mut max_r2 = 0.0_f64;
            for row in centers.outer_iter() {
                let mut r2 = 0.0_f64;
                for &v in row.iter() {
                    if v.is_finite() {
                        r2 += v * v;
                    }
                }
                max_r2 = max_r2.max(r2);
            }
            max_r2
        }
        CenterStrategy::UniformGrid { .. } => {
            // The grid spans `[min_c, max_c]` per axis, so the extreme center
            // radius is realized at the bounding-box corner whose coordinate is
            // `max(|min_c|, |max_c|)` on every axis simultaneously.
            let mut corner_r2 = 0.0_f64;
            for &c in feature_cols.iter() {
                let mut lo = f64::INFINITY;
                let mut hi = f64::NEG_INFINITY;
                for row in data.outer_iter() {
                    if let Some(&v) = row.get(c)
                        && v.is_finite()
                    {
                        lo = lo.min(v);
                        hi = hi.max(v);
                    }
                }
                if lo.is_finite() && hi.is_finite() {
                    let extreme = lo.abs().max(hi.abs());
                    corner_r2 += extreme * extreme;
                }
            }
            corner_r2
        }
        CenterStrategy::EqualMass { .. }
        | CenterStrategy::EqualMassCovarRepresentative { .. }
        | CenterStrategy::FarthestPoint { .. }
        | CenterStrategy::KMeans { .. } => {
            constant_curvature_data_chart_radius2(data, feature_cols)
        }
    }
}

/// `max‖x‖²` over the term's feature columns of `data`, the other half of the
/// evaluated pair set. Non-finite coordinates are skipped rather than poisoning
/// the maximum (the basis build refuses them separately, by row and name).
pub fn constant_curvature_data_chart_radius2(
    data: ArrayView2<'_, f64>,
    feature_cols: &[usize],
) -> f64 {
    let mut max_r2 = 0.0_f64;
    for row in data.outer_iter() {
        let mut r2 = 0.0_f64;
        for &c in feature_cols.iter() {
            if let Some(&v) = row.get(c)
                && v.is_finite()
            {
                r2 += v * v;
            }
        }
        max_r2 = max_r2.max(r2);
    }
    max_r2
}

/// The stereographic constant-curvature chart has a distinguished pole: the
/// chart origin.  Curvature sign is visible first in the radial geodesic map
/// from that pole (`2 atan(√κ r)/√κ` versus `2 atanh(√|κ| r)/√|κ|`).  A pure
/// farthest-point subset can miss the pole on disk-like clouds, leaving the
/// radial mode to be reconstructed indirectly from boundary centers; then the
/// positive chart's distance compression becomes a generic interpolation
/// advantage and the κ profile is sign-blind.  Keep the user's requested center
/// count, but make data-driven center sets pole-aware by replacing the center
/// closest to the origin with the exact origin.  User-provided centers are left
/// verbatim.
fn select_constant_curvature_centers(
    data: ArrayView2<'_, f64>,
    strategy: &CenterStrategy,
) -> Result<Array2<f64>, BasisError> {
    let mut centers = select_centers_by_strategy(data, strategy)?;
    match strategy {
        CenterStrategy::UserProvided(_) => return Ok(centers),
        CenterStrategy::Auto(inner) => {
            if matches!(inner.as_ref(), CenterStrategy::UserProvided(_)) {
                return Ok(centers);
            }
        }
        CenterStrategy::DuchonSpectral { knots, .. } => {
            if center_strategy_kind(knots) == CenterStrategyKind::UserProvided {
                return Ok(centers);
            }
        }
        // Every data-driven strategy picks its centers from the cloud and can
        // therefore miss the chart origin, so all of them get the pole-aware
        // replacement below. Enumerated rather than wildcarded so a new
        // strategy has to state whether its centers are user-authored.
        CenterStrategy::EqualMass { .. }
        | CenterStrategy::EqualMassCovarRepresentative { .. }
        | CenterStrategy::FarthestPoint { .. }
        | CenterStrategy::KMeans { .. }
        | CenterStrategy::UniformGrid { .. } => {}
    }
    if centers.nrows() == 0 || centers.ncols() == 0 {
        return Ok(centers);
    }
    let (closest, _) = centers
        .outer_iter()
        .enumerate()
        .map(|(i, row)| (i, row.dot(&row)))
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .expect("centers has at least one row; the empty case returned above");
    for j in 0..centers.ncols() {
        centers[(closest, j)] = 0.0;
    }
    Ok(centers)
}

/// Symmetrize `M` in place to `(M + Mᵀ)/2` (the realized penalty is built from
/// the symmetric kernel Gram; the κ-derivative blocks inherit the same exact
/// symmetrization the value path applies before normalization).
pub(crate) fn symmetrize(m: &Array2<f64>) -> Array2<f64> {
    gam_linalg::matrix::symmetrize(m)
}

/// Map a single primary-penalty κ-derivative onto the active penalty list by
/// source — the constant-curvature analogue of the Matérn double-penalty
/// derivative selector. The RKHS Gram is the only κ-moving penalty; the
/// double-penalty ridge `I` is κ-independent, so its derivative is exactly
/// zero. Any other source would mean the basis grew a penalty whose κ-movement
/// is unaccounted for, so we refuse loudly rather than silently drop a term.
pub(crate) fn active_constant_curvature_penalty_derivatives(
    penalties: &[ActivePenalty],
    primary_derivative: &Array2<f64>,
) -> Result<Vec<Array2<f64>>, BasisError> {
    penalties
        .iter()
        .map(|penalty| match &penalty.info.source {
            PenaltySource::Primary => Ok(primary_derivative.clone()),
            PenaltySource::DoublePenaltyNullspace => {
                Ok(Array2::<f64>::zeros(primary_derivative.raw_dim()))
            }
            other => Err(BasisError::InvalidInput(format!(
                "unexpected constant-curvature penalty source in κ-derivative path: {other:?}"
            ))),
        })
        .collect()
}

/// κ-derivative bundle for the constant-curvature smooth — the ψ-channel hook
/// that lets κ join the outer LAML/REML optimization as one signed,
/// design-moving coordinate (#944 stage 3 final wiring).
///
/// The outer optimizer's ψ-coordinate here is the **raw, signed curvature κ
/// itself** (NOT `log κ` as for the Matérn kernel scale): κ = 0 must be a
/// reachable interior point of the `S^d ← ℝ^d → H^d` family, which `log κ`
/// cannot represent. So this returns `∂·/∂κ` and `∂²·/∂κ²` directly, and the
/// outer assembly treats the coordinate as `ψ = κ` with `∂/∂ψ = ∂/∂κ`.
///
/// Every κ-fixed piece (centers, length scale ℓ, the center-space constraint
/// transform `z`) is held constant exactly as documented in the module
/// κ-contract, so the design moves with κ only through the geodesic-exponential
/// kernel and:
///
/// ```text
///   X = K(data, centers)·z          ⇒  ∂X/∂κ  = (∂K_dc/∂κ)·z,
///                                       ∂²X/∂κ² = (∂²K_dc/∂κ²)·z
///   S_raw = symm(zᵀ K(centers,centers) z)
///                                   ⇒  ∂S_raw/∂κ  = symm(zᵀ(∂K_cc/∂κ)z), etc.
/// ```
///
/// and the Frobenius penalty normalization is differentiated with the exact
/// quotient rules through the shared `normalize_penaltywith_psi_derivatives`
/// seam — identical to how the Matérn operator penalties propagate their
/// normalization. The double-penalty ridge `I` is κ-independent (zero
/// derivative).
///
/// Mirrors [`build_constant_curvature_basis`] so the realized design and
/// penalties whose κ-derivatives this returns are byte-for-byte the same
/// construction the value path produced (same centers, same ℓ, same `z`).
pub fn build_constant_curvature_basis_kappa_derivatives(
    data: ArrayView2<'_, f64>,
    spec: &ConstantCurvatureBasisSpec,
) -> Result<BasisPsiDerivativeBundle, BasisError> {
    if data.ncols() == 0 {
        crate::bail_invalid_basis!("constant-curvature smooth needs at least one feature column");
    }
    if !spec.kappa.is_finite() {
        crate::bail_invalid_basis!("constant-curvature smooth needs a finite kappa");
    }
    validate_chart_points(data, spec.kappa, "data")?;
    // Pole-aware centers, IDENTICAL to `build_constant_curvature_basis` (#1464):
    // this bundle's whole contract is that the design/penalty whose κ-derivatives
    // it returns are byte-for-byte the SAME construction the value path produced
    // (see the doc above). The value builder replaces the near-origin center with
    // the exact pole for sign identifiability; if this bundle re-derived plain
    // farthest-point centers instead, ∂X/∂κ would be the derivative of a DIFFERENT
    // design than the frozen one the outer criterion is built on, desyncing the
    // analytic κ-gradient from the finite difference of the cost.
    let centers = select_constant_curvature_centers(data, &spec.center_strategy)?;
    if centers.nrows() < 2 {
        return Err(BasisError::InsufficientColumnsForConstraint {
            found: centers.nrows(),
        });
    }
    validate_chart_points(centers.view(), spec.kappa, "centers")?;
    let length_scale = realized_constant_curvature_length_scale(centers.view(), spec.length_scale)?;

    // κ-fixed constraint transform `z`, resolved exactly as the value builder.
    let z = match &spec.identifiability {
        ConstantCurvatureIdentifiability::FrozenTransform { transform } => {
            if transform.nrows() != centers.nrows() {
                crate::bail_dim_basis!(
                    "frozen constant-curvature identifiability transform mismatch: {} centers but transform has {} rows",
                    centers.nrows(),
                    transform.nrows()
                );
            }
            transform.clone()
        }
        ConstantCurvatureIdentifiability::CenterSumToZero => {
            let weights = Array1::<f64>::ones(centers.nrows());
            weighted_coefficient_sum_to_zero_transform(weights.view())?
        }
    };
    let gauge = gam_problem::Gauge::from_block_transforms(&[z.clone()]);

    // Effective-length κ-jet L(κ) = ℓ_ref·s(κ)/s₀ (the κ-invariant-resolution
    // fix). The kernel exponent is q = d/L with BOTH d and L moving in κ, so the
    // kernel κ-jets carry the full quotient chain rule — see
    // `constant_curvature_kernel_kappa_jets_scaled`.
    let l_jet =
        constant_curvature_effective_length_jet(data, centers.view(), length_scale, spec.kappa)?;

    // Design κ-jets: X = K(data, centers)·z, so the κ-derivatives are the
    // kernel κ-jets right-multiplied by the κ-fixed `z`.
    let (_k_dc, dk_dc, dkk_dc) =
        constant_curvature_kernel_kappa_jets_scaled(data, centers.view(), spec.kappa, l_jet)?;
    let design_first = gauge.restrict_design(&dk_dc);
    let design_second_diag = gauge.restrict_design(&dkk_dc);

    // Penalty κ-jets: S = symm(zᵀ K(centers,centers) z), kept RAW (no Frobenius
    // normalization) exactly as the value builder now does (scale = 1). The raw
    // symmetric penalty's κ-derivatives are therefore the symmetrized restricted
    // kernel κ-jets DIRECTLY — there is no normalization quotient rule to
    // propagate, which also removes the κ-dependent ‖S‖_F factor that the
    // normalized form had to differentiate.
    //
    // The penalty kernel is built at the CENTER→center effective-length jet
    // L_S(κ) (#1464), NOT the design's data→center L(κ), so the analytic κ-gradient
    // of logdet|S|₊ stays EXACT for the penalty-resolution-invariant value build
    // above. q_S = d/L_S with both d and L_S moving in κ, so the quotient chain
    // rule inside `constant_curvature_kernel_kappa_jets_scaled` carries the L_S jet.
    let l_jet_penalty = constant_curvature_effective_length_jet(
        centers.view(),
        centers.view(),
        length_scale,
        spec.kappa,
    )?;
    let (_k_cc, dk_cc, dkk_cc) = constant_curvature_kernel_kappa_jets_scaled(
        centers.view(),
        centers.view(),
        spec.kappa,
        l_jet_penalty,
    )?;
    let s_first = symmetrize(&gauge.restrict_penalty(&dk_cc));
    let s_second = symmetrize(&gauge.restrict_penalty(&dkk_cc));

    // Align the single primary-penalty derivative with the realized active
    // penalty list (primary always; ridge only when double_penalty, and
    // κ-independent). Rebuild the realized basis once to read `penaltyinfo`.
    let base = build_constant_curvature_basis(data, spec)?;
    let penalties_derivative =
        active_constant_curvature_penalty_derivatives(&base.active_penalties, &s_first)?;
    let penaltiessecond_derivative =
        active_constant_curvature_penalty_derivatives(&base.active_penalties, &s_second)?;

    Ok(BasisPsiDerivativeBundle {
        first: BasisPsiDerivativeResult {
            design_derivative: design_first,
            penalties_derivative,
            implicit_operator: None,
        },
        second: BasisPsiSecondDerivativeResult {
            designsecond_derivative: design_second_diag,
            penaltiessecond_derivative,
            implicit_operator: None,
        },
        implicit_operator: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use gam_linalg::faer_ndarray::FaerEigh;

    // Diagnostic (#1059 follow-up): show that a κ-FROZEN chart-scale length
    // makes the geodesic-exponential kernel COLLAPSE toward the constant
    // function as κ grows positive (sphere distances compress), which is the
    // degenerate optimum the REML criterion rails to. For a fixed center set we
    // print, per κ, the median geodesic distance and the kernel "spread"
    // 1 − mean(offdiag K). A collapsing kernel ⇒ spread → 0 as κ ↑.
    #[test]
    pub(crate) fn kernel_spread_collapses_with_kappa_at_frozen_length_scale() {
        // 8 centers in a disk of radius 0.45 (inside every κ∈[-2,2] chart).
        let centers = ndarray::array![
            [0.10, 0.05],
            [-0.20, 0.15],
            [0.30, -0.10],
            [-0.05, -0.25],
            [0.22, 0.20],
            [-0.30, -0.05],
            [0.05, 0.30],
            [-0.15, 0.10],
        ];
        // Frozen ℓ: the κ=0 chart-scale auto rule (median 2‖Δ‖).
        let ell_frozen = realized_constant_curvature_length_scale(centers.view(), 0.0)
            .expect("fixture centers span a positive pairwise distance");

        let spread = |kappa: f64, ell: f64| -> f64 {
            let k = constant_curvature_kernel_matrix(centers.view(), centers.view(), kappa, ell)
                .expect("fixture centers are distinct and the length scale is positive");
            let m = k.nrows();
            let mut s = 0.0;
            let mut cnt = 0.0;
            for i in 0..m {
                for j in 0..m {
                    if i != j {
                        s += k[(i, j)];
                        cnt += 1.0;
                    }
                }
            }
            1.0 - s / cnt
        };

        let s_neg = spread(-2.0, ell_frozen);
        let s_zero = spread(0.0, ell_frozen);
        let s_pos = spread(2.0, ell_frozen);
        eprintln!(
            "[κ-collapse] frozen ℓ={ell_frozen:.4}: spread κ=-2 {s_neg:.4} | κ=0 {s_zero:.4} | κ=+2 {s_pos:.4}"
        );

        // The degenerate signature: positive κ collapses the kernel toward the
        // constant (spread shrinks), so the criterion can buy cheap EDF by
        // pushing κ up — this is the unidentifiability we are fixing.
        assert!(
            s_pos < s_zero && s_zero < s_neg,
            "expected kernel spread to shrink with κ at frozen ℓ: κ=-2 {s_neg} κ=0 {s_zero} κ=+2 {s_pos}"
        );

        // Decompose the κ-monotone REML Occam term. The realized penalty is the
        // Frobenius-normalized centered Gram S~ = S_raw/‖S_raw‖_F with
        // S_raw = symm(zᵀ K z); the REML evidence carries +½ log|S~|_+ over its
        // range. Print log det₊(S~) per κ to see whether the penalty-normalization
        // Occam term (not just the modest kernel-spread shift) is what rails κ.
        let weights = Array1::<f64>::ones(centers.nrows());
        let z = weighted_coefficient_sum_to_zero_transform(weights.view())
            .expect("fixture weights are positive, so the sum-to-zero transform exists");
        let logdet_norm_penalty = |kappa: f64, ell: f64| -> f64 {
            let k = constant_curvature_kernel_matrix(centers.view(), centers.view(), kappa, ell)
                .expect("fixture centers are distinct and the length scale is positive");
            let s_raw = symmetrize(&z.t().dot(&k).dot(&z));
            let (s_norm, _c) = normalize_penalty(&s_raw);
            let sym = symmetrize(&s_norm);
            let (evals, _v) = FaerEigh::eigh(&sym, faer::Side::Lower)
                .expect("the fixture Gram is symmetric, so eigh converges");
            let max = evals.iter().cloned().fold(0.0_f64, f64::max);
            let tol = max * 1e-9;
            evals
                .iter()
                .filter(|&&e| e > tol)
                .map(|&e| e.ln())
                .sum::<f64>()
        };
        let l_neg = logdet_norm_penalty(-2.0, ell_frozen);
        let l_zero = logdet_norm_penalty(0.0, ell_frozen);
        let l_pos = logdet_norm_penalty(2.0, ell_frozen);
        eprintln!(
            "[κ-collapse] log|S~|_+ (frozen ℓ): κ=-2 {l_neg:.4} | κ=0 {l_zero:.4} | κ=+2 {l_pos:.4}"
        );

        // GEODESIC-SCALED ℓ removes the κ-dependence of the kernel resolution:
        // set ℓ(κ) = median geodesic distance d_κ among centers. Then the spread
        // should be ~κ-invariant. Print the geodesic-ℓ spread per κ.
        let geo_median_ell = |kappa: f64| -> f64 {
            let m = centers.nrows();
            let manifold = ConstantCurvature::new(centers.ncols(), kappa);
            let mut dists = Vec::with_capacity(m * (m - 1) / 2);
            for i in 0..m {
                for j in (i + 1)..m {
                    dists.push(
                        manifold
                            .distance(centers.row(i), centers.row(j))
                            .expect("fixture centers lie on the manifold"),
                    );
                }
            }
            dists.sort_by(|a, b| a.partial_cmp(b).expect("pairwise distances are finite"));
            dists[dists.len() / 2]
        };
        let gs_neg = spread(-2.0, geo_median_ell(-2.0));
        let gs_zero = spread(0.0, geo_median_ell(0.0));
        let gs_pos = spread(2.0, geo_median_ell(2.0));
        let gl_neg = logdet_norm_penalty(-2.0, geo_median_ell(-2.0));
        let gl_zero = logdet_norm_penalty(0.0, geo_median_ell(0.0));
        let gl_pos = logdet_norm_penalty(2.0, geo_median_ell(2.0));
        eprintln!(
            "[κ-collapse] geodesic ℓ: spread κ=-2 {gs_neg:.4} | κ=0 {gs_zero:.4} | κ=+2 {gs_pos:.4}"
        );
        eprintln!(
            "[κ-collapse] geodesic ℓ: log|S~|_+ κ=-2 {gl_neg:.4} | κ=0 {gl_zero:.4} | κ=+2 {gl_pos:.4}"
        );

        // CANDIDATE FIX: freeze the Frobenius normalization constant at κ=0 so
        // the REML Occam term log|S_λ|_+ carries only the GENUINE roughness
        // spectrum log|S_raw(κ)|_+ (minus a κ-independent constant), not the
        // spurious −r·log‖S_raw(κ)‖_F leak. Compare:
        //   (a) log|S_raw(κ)|_+        (un-normalized, true roughness Occam term)
        //   (b) log|S_raw(κ)/c₀|_+     (frozen-c₀ normalization at κ=0)
        // Both should be κ-IDENTIFYING (a real interior optimum), not monotone.
        let logdet_raw = |kappa: f64, ell: f64, c0: f64| -> f64 {
            let k = constant_curvature_kernel_matrix(centers.view(), centers.view(), kappa, ell)
                .expect("fixture centers are distinct and the length scale is positive");
            let s_raw = symmetrize(&z.t().dot(&k).dot(&z));
            let scaled = s_raw.mapv(|v| v / c0);
            let (evals, _v) = FaerEigh::eigh(&scaled, faer::Side::Lower)
                .expect("the fixture Gram is symmetric, so eigh converges");
            let max = evals.iter().cloned().fold(0.0_f64, f64::max);
            let tol = max * 1e-9;
            evals
                .iter()
                .filter(|&&e| e > tol)
                .map(|&e| e.ln())
                .sum::<f64>()
        };
        // c₀ = ‖S_raw(κ=0)‖_F at frozen ℓ.
        let k0 = constant_curvature_kernel_matrix(centers.view(), centers.view(), 0.0, ell_frozen)
            .expect("fixture centers are distinct and the length scale is positive");
        let s_raw0 = symmetrize(&z.t().dot(&k0).dot(&z));
        let c0 = s_raw0.iter().map(|v| v * v).sum::<f64>().sqrt();
        let r_neg = logdet_raw(-2.0, ell_frozen, c0);
        let r_zero = logdet_raw(0.0, ell_frozen, c0);
        let r_pos = logdet_raw(2.0, ell_frozen, c0);
        eprintln!(
            "[κ-collapse] frozen-c₀ log|S_raw/c₀|_+ (frozen ℓ): κ=-2 {r_neg:.4} | κ=0 {r_zero:.4} | κ=+2 {r_pos:.4}"
        );
        // Finer grid to see the shape of the un-normalized roughness Occam term.
        eprint!("[κ-collapse] frozen-c₀ grid:");
        for kk in [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0] {
            eprint!(" κ={kk}:{:.4}", logdet_raw(kk, ell_frozen, c0));
        }
        eprintln!();
    }

    /// The fill-invariant effective-length κ-jet `(L, L′, L″)` must be EXACT:
    /// `L` solves the fill target `g(L,κ)=fill⋆` (verify the fill is held
    /// κ-invariant), and `L′`, `L″` match central finite differences of the
    /// implicit solution `L(κ)` itself (re-solving the Newton root at κ±h). This
    /// is the gate the ψ-channel outer gradient depends on — `L′`,`L″` feed the
    /// kernel quotient jets in `constant_curvature_kernel_kappa_jets_scaled`.
    #[test]
    pub(crate) fn effective_length_jet_matches_fd_of_implicit_solution() {
        let (data, centers) = oracle_disk_design_centers();
        let ell_ref = realized_constant_curvature_length_scale(centers.view(), 0.0)
            .expect("fixture centers span a positive pairwise distance");
        // Reference fill at κ = 0 (the target L(κ) is pinned to).
        let fill_star = data_center_reference_fill(data.view(), centers.view(), ell_ref)
            .expect("fixture data and centers share an ambient dimension");
        // Solve-only helper: the converged Newton root L(κ) for FD of the jet.
        let solve_l = |kappa: f64| -> f64 {
            constant_curvature_effective_length_jet(data.view(), centers.view(), ell_ref, kappa)
                .expect(
                    "the Newton root for the effective length converges on the fixture geometry",
                )
                .0
        };
        let h = 1e-5_f64;
        for &kappa in &[-1.5_f64, -0.5, -1e-7, 0.0, 1e-7, 0.8, 1.7] {
            let (l, l1, l2) = constant_curvature_effective_length_jet(
                data.view(),
                centers.view(),
                ell_ref,
                kappa,
            )
            .expect("the Newton root for the effective length converges on the fixture geometry");
            // L solves the fill target: g(L, κ) = fill⋆.
            let (g, ..) = data_center_fill_partials(data.view(), centers.view(), kappa, l)
                .expect("fixture data and centers share an ambient dimension");
            assert!(
                (g - fill_star).abs() <= 1e-10 * (1.0 + fill_star.abs()),
                "κ={kappa}: fill not held invariant: g(L,κ)={g} vs fill⋆={fill_star}"
            );
            // κ = 0 ⇒ L = ℓ_ref exactly (the reference point).
            if kappa == 0.0 {
                assert!(
                    (l - ell_ref).abs() <= 1e-10 * ell_ref,
                    "L(0) must equal ℓ_ref; got {l} vs {ell_ref}"
                );
            }
            // L′, L″ vs central FD of the re-solved implicit root.
            let lp = solve_l(kappa + h);
            let lm = solve_l(kappa - h);
            let fd1 = (lp - lm) / (2.0 * h);
            let fd2 = (lp - 2.0 * l + lm) / (h * h);
            assert!(
                (l1 - fd1).abs() <= 1e-5 * (1.0 + fd1.abs()),
                "κ={kappa}: L′ analytic {l1} vs FD {fd1}"
            );
            assert!(
                (l2 - fd2).abs() <= 1e-3 * (1.0 + fd2.abs()),
                "κ={kappa}: L″ analytic {l2} vs FD {fd2}"
            );
        }
    }

    /// 8 data rows + 8 centers inside a disk of radius < 0.5 (valid in every
    /// κ ∈ [−3, 3] chart). Data ≠ centers so the data→center scale is nontrivial.
    pub(crate) fn oracle_disk_design_centers() -> (Array2<f64>, Array2<f64>) {
        let centers = ndarray::array![
            [0.10, 0.05],
            [-0.20, 0.15],
            [0.30, -0.10],
            [-0.05, -0.25],
            [0.22, 0.20],
            [-0.30, -0.05],
            [0.05, 0.30],
            [-0.15, 0.10],
        ];
        // Deterministic pseudo-random data on a slightly wider disk.
        let mut state = 0x2545_f491_4f6c_dd1d_u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            // map to (−0.42, 0.42)
            ((state >> 11) as f64 / (1u64 << 53) as f64 - 0.5) * 0.84
        };
        let n = 60usize;
        let mut data = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            data[(i, 0)] = next();
            data[(i, 1)] = next();
        }
        (data, centers)
    }

    /// #1531 regression: the constant-curvature RKHS primary penalty (the
    /// gauge-restricted kernel Gram `zᵀKz`) is strictly PD / full-rank, so it has
    /// NO null space. This is the fact that makes the `double_penalty` identity
    /// ridge at the top of `build_constant_curvature_basis` a deliberate
    /// whole-chart shrinkage coordinate rather than a null-space penalty.
    #[test]
    fn constant_curvature_gram_is_full_rank_so_identity_is_the_only_double_penalty() {
        // Centers inside every κ chart, several curvatures spanning sign.
        let centers = ndarray::array![
            [0.10, 0.05],
            [-0.20, 0.15],
            [0.30, -0.10],
            [-0.05, -0.25],
            [0.22, 0.20],
            [-0.30, -0.05],
            [0.05, 0.30],
            [-0.15, 0.10],
        ];
        let weights = Array1::<f64>::ones(centers.nrows());
        let z = weighted_coefficient_sum_to_zero_transform(weights.view())
            .expect("fixture weights are positive, so the sum-to-zero transform exists");
        // Frozen auto length scale (the κ=0 chart-scale rule; 0.0 ⇒ auto), reused
        // across κ so the full-rank check is on the same resolution the basis uses.
        let ell = realized_constant_curvature_length_scale(centers.view(), 0.0)
            .expect("fixture centers span a positive pairwise distance");

        for &kappa in &[-2.0_f64, -0.5, 0.0, 0.5, 2.0] {
            let k = constant_curvature_kernel_matrix(centers.view(), centers.view(), kappa, ell)
                .expect("fixture centers are distinct and the length scale is positive");
            // Primary penalty exactly as the basis builder forms it: symmetrized
            // gauge-restricted kernel Gram.
            let raw = symmetrize(&z.t().dot(&k).dot(&z));

            // (a) The primary is full-rank PD: smallest eigenvalue is strictly
            // positive (well above the spectral tolerance), so there is no null
            // space for a Marra-Wood ridge to shrink.
            let (evals, _v) = FaerEigh::eigh(&raw, faer::Side::Lower)
                .expect("the fixture Gram is symmetric, so eigh converges");
            let max = evals.iter().cloned().fold(0.0_f64, f64::max);
            let min = evals.iter().cloned().fold(f64::INFINITY, f64::min);
            assert!(
                max > 0.0 && min > max * 1e-9,
                "constant-curvature Gram must be full-rank PD at κ={kappa}: \
                 min eig {min:e}, max eig {max:e}"
            );
        }
    }

    /// The geodesic-exponential kernel is Matérn-½, so it has a CUSP at every
    /// center — and that is what blocks a constant-curvature SAE ATOM.
    ///
    /// The distinction is between the two ways a basis gets used. A GAM smooth
    /// evaluates its design at FIXED data rows and never differentiates with
    /// respect to the input, so a cusp in `x` is invisible to it and this kernel
    /// is exactly right there. A SAE atom's latent coordinate is a FITTED
    /// parameter: its solve consumes `∂Φ/∂t` and `∂²Φ/∂t²` (the
    /// `SaeBasisSecondJet` contract feeding the Newton/Schur assembly). At a
    /// center the first derivative is direction-dependent and the second is
    /// unbounded, so a Newton step there has no curvature to trust.
    ///
    /// This measures it rather than citing it: the central second difference of
    /// `K` along a fixed direction through a center is compared against the same
    /// quantity a smooth distance away. If the kernel were `C²` both would
    /// converge; instead the at-center value grows like `1/h` while the offset
    /// one converges, and the test asserts a decade of separation.
    ///
    /// The kernel is not at fault. Geodesic distance is conditionally negative
    /// definite on all three space forms, so `exp(−c·d_κ)` is PD for EVERY κ —
    /// which is precisely why it was chosen, and smoother radial families lose
    /// that guarantee on spheres. The obstruction is real and belongs to the
    /// atom design, not to this module.
    #[test]
    fn geodesic_exponential_kernel_has_unbounded_curvature_at_a_center() {
        let kappa = 0.0_f64;
        let ell = 1.0_f64;
        let center = ndarray::arr2(&[[0.0_f64, 0.0]]);

        // Second difference of `h -> K(center + h·e_x, center)` at the center,
        // and at a point a fixed distance away where the kernel is smooth.
        let curvature_at = |base: [f64; 2], h: f64| -> f64 {
            let probe = ndarray::arr2(&[
                [base[0] - h, base[1]],
                [base[0], base[1]],
                [base[0] + h, base[1]],
            ]);
            let k = constant_curvature_kernel_matrix(probe.view(), center.view(), kappa, ell)
                .expect("fixture centers are distinct and the length scale is positive");
            (k[[0, 0]] - 2.0 * k[[1, 0]] + k[[2, 0]]) / (h * h)
        };

        let mut at_center = Vec::new();
        let mut off_center = Vec::new();
        for &h in &[1.0e-2_f64, 1.0e-3, 1.0e-4] {
            at_center.push(curvature_at([0.0, 0.0], h).abs());
            off_center.push(curvature_at([0.5, 0.0], h).abs());
        }

        // Away from the center the second difference converges: successive
        // refinements agree.
        assert!(
            (off_center[2] - off_center[1]).abs() <= 1.0e-3 * off_center[1].max(1.0),
            "the kernel must be C² away from its centers; got {off_center:?}"
        );
        // At the center it diverges like 1/h: each 10x refinement multiplies it
        // by ~10, so three decades of h separate by ~100x.
        assert!(
            at_center[2] > 10.0 * at_center[0],
            "a Matérn-½ cusp must make the second difference diverge as h -> 0; \
             got {at_center:?}"
        );
        assert!(
            at_center[2] > 100.0 * off_center[2],
            "the at-center curvature must dwarf the smooth-region curvature; \
             at-center {:?} vs off-center {:?}",
            at_center[2],
            off_center[2]
        );
    }

    /// #2458 — the second κ-derivatives shipped WITHOUT a reader.
    ///
    /// `build_constant_curvature_basis_kappa_derivatives` returns a
    /// `BasisPsiDerivativeBundle` whose `.second` carries the κ-second
    /// derivatives of the design and of each penalty block. Those are consumed
    /// in production (`spatial_optimization.rs` destructures and rotates them
    /// for the spatial ψ path), but nothing anywhere pinned their VALUES:
    /// grepping for readers of `designsecond_derivative` /
    /// `penaltiessecond_derivative` outside the destructuring sites returns
    /// nothing. A second derivative that is shipped and consumed but never
    /// checked is exactly the input #2458 proposes to build a stationarity
    /// CERTIFICATE on, and a wrong certificate is worse than an honestly
    /// missing one.
    ///
    /// This differences the ANALYTIC FIRST derivative, which the κ-gradient
    /// path already exercises end to end, so a failure localizes to the
    /// second-order construction rather than to the basis itself.
    ///
    /// The bound is the central-difference error budget, not a tuned number.
    /// Truncation is order h^2 times the third derivative and roundoff is
    /// order eps times the first derivative over h, so at h = 1e-4 on an
    /// order-one chart both sit near 1e-8 relative. Asserting 1e-6 leaves two
    /// orders of headroom while still failing a missing or mis-scaled term —
    /// which is scale-invariant and would NOT shrink with h, hence the
    /// h-halving arm below.
    #[test]
    fn kappa_second_derivatives_match_a_central_difference_of_the_first_2458() {
        let data = ndarray::array![
            [0.10, 0.05],
            [-0.20, 0.15],
            [0.30, -0.10],
            [-0.05, -0.25],
            [0.22, 0.20],
            [-0.30, -0.05],
            [0.05, 0.30],
            [-0.15, 0.10],
        ];
        let spec = ConstantCurvatureBasisSpec {
            center_strategy: CenterStrategy::FarthestPoint { num_centers: 6 },
            ..Default::default()
        };
        let kappa0 = 0.35_f64;

        let first_at = |kappa: f64| {
            let mut probe = spec.clone();
            probe.kappa = kappa;
            let bundle = build_constant_curvature_basis_kappa_derivatives(data.view(), &probe)
                .expect("fixture points lie inside the chart for every probed kappa");
            (
                bundle.first.design_derivative,
                bundle.first.penalties_derivative,
            )
        };

        let mut exact_spec = spec.clone();
        exact_spec.kappa = kappa0;
        let analytic = build_constant_curvature_basis_kappa_derivatives(data.view(), &exact_spec)
            .expect("fixture points lie inside the chart at kappa0");
        let design_second = analytic.second.designsecond_derivative;
        let penalty_second = analytic.second.penaltiessecond_derivative;

        let max_rel_error_at = |h: f64| -> (f64, f64) {
            let (x_plus, s_plus) = first_at(kappa0 + h);
            let (x_minus, s_minus) = first_at(kappa0 - h);

            let mut design_error = 0.0_f64;
            let mut design_scale = 0.0_f64;
            for ((&plus, &minus), &exact) in
                x_plus.iter().zip(x_minus.iter()).zip(design_second.iter())
            {
                let fd = (plus - minus) / (2.0 * h);
                design_error = design_error.max((fd - exact).abs());
                design_scale = design_scale.max(exact.abs()).max(fd.abs());
            }

            assert_eq!(
                s_plus.len(),
                penalty_second.len(),
                "penalty block count must not depend on kappa"
            );
            let mut penalty_error = 0.0_f64;
            let mut penalty_scale = 0.0_f64;
            for ((block_plus, block_minus), block_exact) in
                s_plus.iter().zip(s_minus.iter()).zip(penalty_second.iter())
            {
                for ((&plus, &minus), &exact) in block_plus
                    .iter()
                    .zip(block_minus.iter())
                    .zip(block_exact.iter())
                {
                    let fd = (plus - minus) / (2.0 * h);
                    penalty_error = penalty_error.max((fd - exact).abs());
                    penalty_scale = penalty_scale.max(exact.abs()).max(fd.abs());
                }
            }
            (
                design_error / design_scale.max(1.0),
                penalty_error / penalty_scale.max(1.0),
            )
        };

        let h = 1.0e-4_f64;
        let (design_rel, penalty_rel) = max_rel_error_at(h);
        eprintln!(
            "[2458-second-fd] h={h:.1e}: design rel={design_rel:.3e} penalty rel={penalty_rel:.3e}"
        );
        assert!(
            design_rel < 1.0e-6,
            "d2X/dkappa2 disagrees with a central difference of dX/dkappa: rel={design_rel:.6e}"
        );
        assert!(
            penalty_rel < 1.0e-6,
            "d2S/dkappa2 disagrees with a central difference of dS/dkappa: rel={penalty_rel:.6e}"
        );

        // A MISSING term is scale-invariant: it does not shrink when h does.
        // Halving h must not inflate the disagreement, which it would if the
        // residual were a genuine missing contribution rather than truncation.
        let (design_rel_half, penalty_rel_half) = max_rel_error_at(0.5 * h);
        eprintln!(
            "[2458-second-fd] h={:.1e}: design rel={design_rel_half:.3e} penalty rel={penalty_rel_half:.3e}",
            0.5 * h
        );
        assert!(
            design_rel_half <= design_rel.max(1.0e-9) * 2.0,
            "halving h must not inflate the design disagreement (missing-term signature): \
             {design_rel:.6e} -> {design_rel_half:.6e}"
        );
        assert!(
            penalty_rel_half <= penalty_rel.max(1.0e-9) * 2.0,
            "halving h must not inflate the penalty disagreement (missing-term signature): \
             {penalty_rel:.6e} -> {penalty_rel_half:.6e}"
        );
    }
}
