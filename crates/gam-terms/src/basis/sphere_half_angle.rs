//! Half-angle geometry of a geodesic separation on `S²`, carried without ever
//! forming `cos γ`.
//!
//! # Why not `cos γ`
//!
//! Every zonal Wahba kernel in [`super::sphere_kernels`] is a function of the
//! half-angle pair
//!
//! ```text
//! u = sin²(γ/2) = (1 − cos γ)/2        (vanishes at coincidence)
//! v = cos²(γ/2) = (1 + cos γ)/2        (vanishes at the antipode)
//! ```
//!
//! and never of `cos γ` itself — the closed forms are `−ln u`, `Li₂(v)`,
//! `Li₃(u)`, `√u`, and their derivatives are `1/u`, `ln u / v`. Carrying the
//! separation as `cos γ` and recovering `u = (1 − cos γ)/2` at the point of use
//! destroys the answer for nearby points, because `cos γ = 1 − O(γ²)`: the
//! subtraction is exact (Sterbenz), but the information is already gone. Since
//! the spacing below `1.0` is `2⁻⁵³`, a `u` reconstructed from a dot product is
//! *quantized* to multiples of `ε/4 ≈ 5.6e-17`, so it can only be `0`, `ε/4`,
//! `ε/2`, … — measured against a 50-digit reference (#2489), that is `27 %`
//! relative error at `1e-6` degrees of latitude separation, `72×` at `1e-7`,
//! and *exactly zero* below `1e-8`, i.e. two distinct points reported
//! coincident. On Earth-scale coordinates `1e-5` degrees is about a metre.
//!
//! It also breaks rotation invariance of the Gram matrix. A zonal kernel's
//! diagonal is `K(0)` for every point, but whether `sin²φ + cos²φ·(cos²ψ +
//! sin²ψ)` rounds to exactly `1.0` depends on the particular `φ, ψ`: over 24
//! farthest-point centers, 17 landed on `1.0`, 5 one ulp below and 2 two ulps
//! below, giving **three different diagonal entries in one Gram matrix**. A
//! rotation relabels the coordinates and reshuffles which centers round which
//! way, so the shipped matrix was not a function of geodesic distance alone.
//!
//! # The chord form
//!
//! Both halves have a haversine expression that is a sum of non-negative terms,
//! so nothing cancels:
//!
//! ```text
//! u = sin²(Δφ/2)       + cos φ · cos φ_c · sin²(Δψ/2)
//! v = sin²((φ + φ_c)/2) + cos φ · cos φ_c · cos²(Δψ/2)
//! ```
//!
//! The `v` form is the `u` form evaluated against the *antipode* of the second
//! point (`φ_c → −φ_c`, `ψ_c → ψ_c + π`), which is the statement
//! `cos²(γ/2) = sin²((π − γ)/2)`. Their sum is `1` analytically:
//! `sin²((φ−φ_c)/2) + sin²((φ+φ_c)/2) = 1 − cos φ cos φ_c`.
//!
//! Each of the four half-angle squares is then taken in **chord form** rather
//! than from a half-angle trig call, because the callers have already
//! precomputed `(sin, cos)` of each latitude and longitude once per point and
//! reuse them across the whole `N × K` grid:
//!
//! ```text
//! 4 sin²(Δθ/2) = (sin θ₁ − sin θ₂)² + (cos θ₁ − cos θ₂)²      (chord²)
//! 4 cos²(Δθ/2) = (sin θ₁ + sin θ₂)² + (cos θ₁ + cos θ₂)²      (chord² to the antipode)
//! ```
//!
//! Both are `|w₁ ∓ w₂|²` for the unit plane vectors `wᵢ = (cos θᵢ, sin θᵢ)`,
//! and `|w₁ − w₂| = 2|sin(Δθ/2)|` is the chord subtending `Δθ`. This buys the
//! two properties the dot product lacks, for about a dozen extra flops per pair
//! and no transcendental calls:
//!
//! 1. **`u = 0` exactly iff the two coordinates are bitwise equal.** The same
//!    `sin_cos` on the same bits gives the same bits, so each difference is
//!    exactly `0`. Self-distance becomes a theorem instead of a rounding
//!    accident, and the Gram diagonal is one number for every center.
//! 2. **No catastrophic cancellation.** The error in `u` is now set by the
//!    absolute error of the stored `sin`/`cos` values (`≈ ε`) against a chord
//!    of length `≈ γ`, i.e. `O(ε/γ)` relative — instead of the dot product's
//!    `O(ε/γ²)`. At `1e-6` degrees that is `1e-8` instead of `27 %`, and it
//!    degrades linearly rather than falling off a cliff.
//!
//! # Genericity
//!
//! [`half_angle_separation`] is written once over any type closed under
//! `+ − ×` that can be built from an `f64` literal, so the scalar path, the
//! `wide::f64x4` SIMD path and the jet path share one derivation rather than
//! three transcriptions of it.

use std::ops::{Add, Mul, Sub};

/// A scalar or SIMD lane type the half-angle algebra can run in.
///
/// Satisfied by `f64` (via the reflexive `From` impl) and by `wide::f64x4`.
pub(crate) trait HalfAngleScalar:
    Copy + Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self> + From<f64>
{
}

impl<T> HalfAngleScalar for T where
    T: Copy + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + From<f64>
{
}

/// The precomputed trigonometry of one lat/lon point, in radians.
///
/// Callers build this once per point and reuse it across every pair, which is
/// what makes the chord form cheaper than a per-pair half-angle `sin`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct SphereTrig<T> {
    pub(crate) sin_lat: T,
    pub(crate) cos_lat: T,
    pub(crate) sin_lon: T,
    pub(crate) cos_lon: T,
}

impl SphereTrig<f64> {
    /// Build from a latitude and longitude already scaled to radians.
    #[inline]
    pub(crate) fn from_radians(lat: f64, lon: f64) -> Self {
        let (sin_lat, cos_lat) = lat.sin_cos();
        let (sin_lon, cos_lon) = lon.sin_cos();
        Self {
            sin_lat,
            cos_lat,
            sin_lon,
            cos_lon,
        }
    }
}

/// The half-angle pair of a geodesic separation `γ`: `u = sin²(γ/2)` and
/// `v = cos²(γ/2)`, with `u + v = 1` analytically.
///
/// `u` resolves the coincident end to full relative precision and `v` the
/// antipodal end; keeping both is what lets the Sobolev closed forms be
/// evaluated accurately at *either* singular end without reconstructing one
/// from the other (see [`super::sphere_kernels`], which needs `−ln u` near
/// `γ = 0` and `ln(1 − v)` near `γ = π`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct HalfAngleSeparation {
    /// `sin²(γ/2) = (1 − cos γ)/2 ∈ [0, 1]`.
    pub(crate) u: f64,
    /// `cos²(γ/2) = (1 + cos γ)/2 ∈ [0, 1]`.
    pub(crate) v: f64,
}

impl HalfAngleSeparation {
    /// `cos γ = v − u`, exact up to one rounding of a result of magnitude
    /// `≤ 1` formed from two non-negative summands of magnitude `≤ 1`.
    ///
    /// Used by the Legendre recurrences in [`super::sphere_spectral`], which
    /// are polynomials in `cos γ` and want the argument itself. Reconstructing
    /// `cos γ` from `u` runs the cancellation the *harmless* way: the small
    /// quantity (`u`) is the one carried to full relative precision, and the
    /// `O(1)` result absorbs its error as `O(ε)` absolute.
    #[inline]
    pub(crate) fn cos_gamma(self) -> f64 {
        (self.v - self.u).clamp(-1.0, 1.0)
    }
}

/// `u = sin²(γ/2)` and `v = cos²(γ/2)` from the precomputed trigonometry of
/// two lat/lon points, in chord form.
///
/// Returns `(u, v)` in the lane type, so the SIMD caller gets four pairs at
/// once. See the module docs for the derivation; in brief, with
/// `cc = cos φ · cos φ_c`,
///
/// ```text
/// u = ¼[(sinφ − sinφ_c)² + (cosφ − cosφ_c)²] + cc · ¼[(sinψ − sinψ_c)² + (cosψ − cosψ_c)²]
/// v = ¼[(sinφ + sinφ_c)² + (cosφ − cosφ_c)²] + cc · ¼[(sinψ + sinψ_c)² + (cosψ + cosψ_c)²]
/// ```
///
/// Note that `v`'s latitude term reuses the *difference* of cosines: reflecting
/// `φ_c → −φ_c` flips the sign of `sin φ_c` and leaves `cos φ_c` alone.
#[inline]
pub(crate) fn half_angle_separation<T: HalfAngleScalar>(
    a: SphereTrig<T>,
    b: SphereTrig<T>,
) -> (T, T) {
    let quarter = T::from(0.25);
    let d_sin_lat = a.sin_lat - b.sin_lat;
    let s_sin_lat = a.sin_lat + b.sin_lat;
    let d_cos_lat = a.cos_lat - b.cos_lat;
    let d_sin_lon = a.sin_lon - b.sin_lon;
    let s_sin_lon = a.sin_lon + b.sin_lon;
    let d_cos_lon = a.cos_lon - b.cos_lon;
    let s_cos_lon = a.cos_lon + b.cos_lon;

    // sin²(Δφ/2), sin²((φ+φ_c)/2), sin²(Δψ/2), cos²(Δψ/2).
    let hav_lat_diff = quarter * (d_sin_lat * d_sin_lat + d_cos_lat * d_cos_lat);
    let hav_lat_sum = quarter * (s_sin_lat * s_sin_lat + d_cos_lat * d_cos_lat);
    let hav_lon = quarter * (d_sin_lon * d_sin_lon + d_cos_lon * d_cos_lon);
    let cov_lon = quarter * (s_sin_lon * s_sin_lon + s_cos_lon * s_cos_lon);

    let cc = a.cos_lat * b.cos_lat;
    (hav_lat_diff + cc * hav_lon, hav_lat_sum + cc * cov_lon)
}

/// Scalar [`half_angle_separation`] packaged as a [`HalfAngleSeparation`], with
/// both halves clamped into `[0, 1]`.
///
/// The clamp is a range assertion, not a regularization: every summand is a
/// square or a product of non-negative cosines, so the exact values already lie
/// in `[0, 1]` and only the final roundings of `u + v = 1` can push a hair
/// outside.
#[inline]
pub(crate) fn half_angle_separation_scalar(
    a: SphereTrig<f64>,
    b: SphereTrig<f64>,
) -> HalfAngleSeparation {
    let (u, v) = half_angle_separation(a, b);
    HalfAngleSeparation {
        u: u.clamp(0.0, 1.0),
        v: v.clamp(0.0, 1.0),
    }
}

/// `u = sin²(γ/2)` and `v = cos²(γ/2)` for two AMBIENT unit vectors on
/// `S^{dim−1}`, where `cos γ = t · c`.
///
/// The chord form is even more direct here than in lat/lon coordinates, and
/// is an identity for unit vectors rather than a trigonometric rearrangement:
///
/// ```text
/// |t − c|² = |t|² + |c|² − 2 t·c = 2(1 − cos γ) = 4 sin²(γ/2)
/// |t + c|² = |t|² + |c|² + 2 t·c = 2(1 + cos γ) = 4 cos²(γ/2)
/// ```
///
/// so `u = |t − c|²/4` and `v = |t + c|²/4`. As in the lat/lon case, each is a
/// sum of squares — nothing cancels, `u` is exactly `0` iff the two vectors are
/// bitwise equal, and `v` is exactly `0` iff they are exact antipodes — whereas
/// `1 − t·c` throws away every bit below `2⁻⁵³` of the separation.
///
/// The two halves are normalized against `(|t|² + |c|²)/2` rather than against
/// the nominal `1`, which is what makes `u + v = 1` hold to a rounding even
/// when the inputs are unit vectors only to within their own storage error.
#[inline]
pub(crate) fn ambient_half_angle_separation(
    point: ndarray::ArrayView1<'_, f64>,
    center: ndarray::ArrayView1<'_, f64>,
) -> HalfAngleSeparation {
    let mut chord_sq = 0.0_f64;
    let mut anti_chord_sq = 0.0_f64;
    let mut norm_sq = 0.0_f64;
    for (t, c) in point.iter().zip(center.iter()) {
        let d = t - c;
        let s = t + c;
        chord_sq += d * d;
        anti_chord_sq += s * s;
        norm_sq += t * t + c * c;
    }
    // `chord_sq + anti_chord_sq = 2(|t|² + |c|²) = 2·norm_sq` exactly in exact
    // arithmetic, so dividing both by `2·norm_sq` is the scale-free reading of
    // the pair and keeps `u + v = 1`. A degenerate all-zero input leaves the
    // separation undefined; reporting the antipode-free `u = 0` there matches
    // the coincident convention every caller already handles.
    let scale = 2.0 * norm_sq;
    if !(scale > 0.0) {
        return HalfAngleSeparation { u: 0.0, v: 1.0 };
    }
    HalfAngleSeparation {
        u: (chord_sq / scale).clamp(0.0, 1.0),
        v: (anti_chord_sq / scale).clamp(0.0, 1.0),
    }
}

/// `∂u/∂φ` and `∂u/∂ψ` at the first point, in radian space.
///
/// The jet of a zonal kernel is `dK/du · ∂u/∂(φ, ψ)`, and taking it in `u`
/// rather than in `cos γ` is what makes the coincident limit computable at all.
/// With `cos γ = 1 − 2u`,
///
/// ```text
/// dK/d(cos γ) · ∂(cos γ)/∂φ = (dK/du · (−½)) · (−2 · ∂u/∂φ) = dK/du · ∂u/∂φ
/// ```
///
/// so the two divergent factors of the `cos γ` chain — `dK/d(cos γ) ~ 1/γ`
/// against `∂(cos γ)/∂φ ~ γ`, whose product is the finite `|γ|` cusp gradient —
/// never appear separately. The old form recovered that finite limit as a
/// numerical `∞ · 0`, and lost it: measured against the constant true value
/// `−0.002778` per degree for the pseudo `m = 1` cusp, it was `+17 %` off at
/// `1e-6°`, `−99 %` at `1e-10°`, and `−100 %` (exactly zero) *at* a center —
/// which is reached in ordinary use, since farthest-point selection picks
/// centers from the data rows themselves.
///
/// The partials are
///
/// ```text
/// ∂u/∂φ = ½ sin(Δφ) − sin φ · cos φ_c · sin²(Δψ/2)
/// ∂u/∂ψ = ½ cos φ · cos φ_c · sin(Δψ)
/// ```
///
/// with `sin(Δθ) = sin θ · cos θ_c − cos θ · sin θ_c` from the precomputed
/// values, which is exactly `0` when the two angles are bitwise equal. So a row
/// sitting on a center yields `∂u/∂φ = ∂u/∂ψ = 0` exactly, and the caller
/// resolves the cusp there rather than multiplying zero by an infinity.
///
/// Consistency check against the `cos γ` form these replace:
/// `−2 ∂u/∂φ = cos φ sin φ_c − sin φ cos φ_c cos Δψ = ∂(cos γ)/∂φ` and
/// `−2 ∂u/∂ψ = −cos φ cos φ_c sin Δψ = ∂(cos γ)/∂ψ`.
#[inline]
pub(crate) fn half_angle_partials(a: SphereTrig<f64>, b: SphereTrig<f64>) -> (f64, f64) {
    let sin_dlat = a.sin_lat * b.cos_lat - a.cos_lat * b.sin_lat;
    let sin_dlon = a.sin_lon * b.cos_lon - a.cos_lon * b.sin_lon;
    let d_sin_lon = a.sin_lon - b.sin_lon;
    let d_cos_lon = a.cos_lon - b.cos_lon;
    let hav_lon = 0.25 * (d_sin_lon * d_sin_lon + d_cos_lon * d_cos_lon);
    let du_dlat = 0.5 * sin_dlat - a.sin_lat * b.cos_lat * hav_lon;
    let du_dlon = 0.5 * a.cos_lat * b.cos_lat * sin_dlon;
    (du_dlat, du_dlon)
}
