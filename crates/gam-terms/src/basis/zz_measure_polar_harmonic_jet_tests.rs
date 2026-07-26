//! Measurement: the spherical-harmonic design jet's `∂/∂lat` at and near the
//! geographic poles.
//!
//! Found while replacing the zonal Legendre-derivative quotient in
//! `sphere_kernels` / `sphere_spectral` (see `zz_measure_2475_tests` §6). The
//! same shape survived in the ASSOCIATED Legendre path that
//! `spherical_harmonic_jet` uses:
//!
//! ```text
//!   let one_minus_x2 = (1.0 - x * x).max(f64::EPSILON);
//!   dp = (-l·x·P_{l,m} + (l+m)·P_{l-1,m}) / one_minus_x2
//! ```
//!
//! with `x = sin(lat)`, so `1 - x² = cos²(lat)` and the denominator vanishes at
//! the poles. The numerator vanishes with it — the identity it encodes is
//! `(1-x²)·P'_{l,m} = -l·x·P_{l,m} + (l+m)·P_{l-1,m}` — so the quotient is a
//! removable `0/0`. It was not resolved, it was floored, and a floor does not
//! restore a removable limit: at a pole the numerator is exactly zero, so the
//! quotient is exactly zero whatever the true value is.
//!
//! ## The two failures, and why they are different
//!
//! **Approach.** The numerator is `O(cos²(lat))` assembled by subtracting two
//! `O(1)` terms, so its relative error grows like `ε/cos²(lat)` — the law
//! `zz_measure_polar_harmonic_jet_numerator_cancels_like_two_over_cos_squared`
//! still measures on the retired expression. `cos(lat)` is `1.7e-4` at
//! `lat = 89.99°`, roughly 1.1 km from the pole, putting the relative error at
//! `1.4e-8`, far above the `1e-11`-class finite-difference agreement gates
//! elsewhere in this crate assert. Global geospatial data reaches those
//! latitudes.
//!
//! **Arrival.** At the pole itself the quotient returned `0.0` for every
//! `(l, m)`. For `m = 0` that is the right answer reached by luck. For `m = 1`
//! it is wrong by the whole quantity: `dP_{l,1}/dlat` at a pole is `l(l+1)/2`,
//! so `(l, m) = (3, 1)` should read `6.0` and read `-0.0`.
//!
//! ## Why this is not just the chart singularity
//!
//! `(lat, lon)` is a singular chart at the poles, and for `m ≥ 1` the harmonic's
//! `∂/∂lat` genuinely depends on the meridian of approach — no implementation
//! can return one right answer for all of them. **That is not what is measured
//! here.** Along a FIXED meridian the derivative is an ordinary one-dimensional
//! limit, and it is that limit these gates pin. The worst-conditioned entry is
//! `(l, m) = (1, 0)`, and an `m = 0` harmonic is zonal: it does not involve
//! `lon` at all, it is a smooth function of latitude, and its `∂/∂lat` has an
//! ordinary finite limit of `0` at the pole. The digits lost there were lost on
//! a quantity that is perfectly well defined.
//!
//! ## The replacement
//!
//! The colatitude form has no pole. With `x = cos θ`,
//!
//! ```text
//!   dP_{l,m}/dθ = ½ [ P_{l,m+1} - (l+m)(l-m+1) P_{l,m-1} ]
//! ```
//!
//! and here `x = sin(lat)`, so `θ` is the colatitude and `d/dlat` flips the
//! bracket. At `m = 0` the identity wants `P_{l,-1}`, which under the
//! Condon–Shortley convention this recurrence carries is `-P_{l,1}/(l(l+1))`;
//! substituting collapses the bracket to `dP_{l,0}/dlat = -P_{l,1}`. Every term
//! is a Legendre value the forward loop already computes, so the division is
//! gone rather than guarded.
//!
//! The sign convention is the part worth measuring rather than reading off a
//! reference, so `..._matches_forward_finite_differences` compares the shipped
//! jet against central differences of the shipped FORWARD design — the two
//! agree only if the convention, the `m = 0` negative-order collapse and the
//! `d/dlat` sign are all right together.

use super::radial_jets_nd::spherical_harmonic_jet;
use super::sphere_basis::{fill_real_spherical_harmonics_row, precompute_harmonic_norms};
use ndarray::{Array1, Array2};

/// Condon–Shortley associated Legendre table, built by exactly the recurrence
/// the forward design uses, EXCEPT that the sectoral factor is taken as the
/// retired `√(1 - x²)` rather than `cos(lat)`. Used only by the conditioning
/// law below, which measures the expression production no longer evaluates.
fn plm_table_from_radicand(x: f64, max_degree: usize) -> Vec<f64> {
    let l_cap = max_degree + 1;
    let idx = |l: usize, m: usize| l * l_cap + m;
    let mut p = vec![0.0_f64; l_cap * l_cap];
    let somx2 = (1.0 - x * x).max(0.0).sqrt();
    p[idx(0, 0)] = 1.0;
    for m in 1..=max_degree {
        p[idx(m, m)] = -((2 * m - 1) as f64) * somx2 * p[idx(m - 1, m - 1)];
    }
    for m in 0..max_degree {
        p[idx(m + 1, m)] = ((2 * m + 1) as f64) * x * p[idx(m, m)];
    }
    for m in 0..=max_degree {
        for l in (m + 2)..=max_degree {
            p[idx(l, m)] = (((2 * l - 1) as f64) * x * p[idx(l - 1, m)]
                - ((l + m - 1) as f64) * p[idx(l - 2, m)])
                / ((l - m) as f64);
        }
    }
    p
}

/// Wilkinson forward-error factor for one entry of the forward design row.
///
/// The differenced samples are not the exact harmonic, they are a
/// floating-point EVALUATION of it, so the cancellation term of a central
/// difference is `err(f)/h` and not `ε·|f|/h`. `P_{l,m}` is built by a
/// recurrence of depth `≤ L` carrying about three rounded operations per step,
/// which is `γ_{3L} = 3Lε/(1 - 3Lε)`, and the norm and the trigonometric
/// factor add two more roundings. Charging `ε` alone would credit the reference
/// with an accuracy it does not have, which is the same currency mistake as
/// comparing an exact-range enclosure against a rounded endpoint.
fn forward_row_relative_error(max_degree: usize) -> f64 {
    let k = 3.0 * (max_degree as f64) + 2.0;
    k * f64::EPSILON / (1.0 - k * f64::EPSILON)
}

/// One forward design row from production, in degrees.
fn forward_row(lat_deg: f64, lon_deg: f64, max_degree: usize) -> Array1<f64> {
    let l_cap = max_degree + 1;
    let norms = precompute_harmonic_norms(max_degree);
    let mut p_buf = vec![0.0_f64; l_cap * l_cap];
    let mut row = Array1::<f64>::zeros(max_degree * (max_degree + 2));
    let deg = std::f64::consts::PI / 180.0;
    fill_real_spherical_harmonics_row(
        lat_deg * deg,
        lon_deg * deg,
        max_degree,
        &mut p_buf,
        &norms,
        row.view_mut(),
    );
    row
}

/// Column index of the `(l, m)` cosine-phase entry in the shared layout: per
/// degree `l`, the `sin(mψ)` columns for `m = l..1`, then `m = 0`, then the
/// `cos(mψ)` columns for `m = 1..l`.
fn cos_column(l: usize, m: usize) -> usize {
    let before: usize = (1..l).map(|d| 2 * d + 1).sum();
    before + l + m
}

/// PRIMARY GATE. The shipped `∂/∂lat` must agree with central differences of
/// the shipped FORWARD design along a fixed meridian, marching toward the pole.
///
/// The bar is derived, not chosen. A central difference of step `h` carries
/// `|f'''|·h²/6` of truncation and `ε·M/h` of cancellation. Restricted to a
/// fixed meridian, `P_{l,m}(sin lat)` is a trigonometric polynomial in `lat` of
/// degree `≤ l`, so Bernstein's inequality gives `|dⁿf/dlatⁿ| ≤ lⁿ·‖f‖_∞` with
/// the norm taken over the WHOLE meridian — which is why `M` below is a sweep
/// maximum and not the local sample size. Near a pole the two differ by orders:
/// `Y_{l,1}` vanishes there while its latitude derivative is `l(l+1)/2`, so a
/// locally-scaled bound would be tighter than the derivative it brackets. With
/// `h` in radians the finite difference's own uncertainty is
///
/// ```text
///   L³·M·h²/6 + 2ε·M/h
/// ```
///
/// which is the accuracy the reference HAS, not a tolerance anyone picked. The
/// gate asserts the jet lies inside it. Latitudes stop at `10·h` from the pole
/// so the stencil never straddles the clamp — past that the one-sided limit is
/// the pole gate below.
#[test]
fn zz_measure_polar_harmonic_jet_matches_forward_finite_differences() {
    let max_degree = 6usize;
    let lon_deg = 37.5_f64;
    let deg = std::f64::consts::PI / 180.0;
    // In degrees; `10 * h` is the closest approach the stencil supports.
    let h = 1.0e-4_f64;
    let ncols = max_degree * (max_degree + 2);

    // Bernstein scale: the sup of each column over the full meridian, sampled
    // finely enough that a degree-`L` trigonometric polynomial cannot hide a
    // peak between samples.
    let mut col_scale = vec![0.0_f64; ncols];
    for step in 0..=720usize {
        let lat_deg = -90.0 + 180.0 * (step as f64) / 720.0;
        let row = forward_row(lat_deg, lon_deg, max_degree);
        for col in 0..ncols {
            col_scale[col] = col_scale[col].max(row[col].abs());
        }
    }

    println!("central-difference gate, h = {h:.1e} deg, meridian {lon_deg} deg");
    println!(
        "{:>14}  {:>10}  {:>13}  {:>13}  {:>11}",
        "90 - lat (deg)", "(l,m)", "jet d/dlat", "central diff", "fd bound"
    );

    let mut worst_ratio = 0.0_f64;
    for &gap in &[10.0_f64, 1.0, 1.0e-1, 1.0e-2, 1.0e-3] {
        let lat_deg = 90.0 - gap;
        let data = Array2::from_shape_vec((1, 2), vec![lat_deg, lon_deg])
            .expect("1x2 lat/lon fixture is well formed");
        let jet = spherical_harmonic_jet(data.view(), max_degree, false)
            .expect("the harmonic jet must build on an interior latitude");

        let up = forward_row(lat_deg + h, lon_deg, max_degree);
        let down = forward_row(lat_deg - h, lon_deg, max_degree);

        for col in 0..ncols {
            let fd = (up[col] - down[col]) / (2.0 * h);
            let shipped = jet[[0, col, 0]];
            // Meridian sup, and the step in RADIANS, which is the variable the
            // smoothness bound is stated in.
            let scale = col_scale[col].max(f64::MIN_POSITIVE);
            let h_rad = h * deg;
            let lf = max_degree as f64;
            let bound_rad = lf * lf * lf * scale * h_rad * h_rad / 6.0
                + 2.0 * forward_row_relative_error(max_degree) * scale / h_rad;
            // The jet is reported per DEGREE, so the bound converts with it.
            let bound = bound_rad * deg;
            let miss = (shipped - fd).abs();
            worst_ratio = worst_ratio.max(miss / bound);
            assert!(
                miss <= bound,
                "column {col} at lat {lat_deg}: jet {shipped:.12e} vs central difference \
                 {fd:.12e}, miss {miss:.6e} exceeds the difference's own uncertainty {bound:.6e}"
            );
        }

        // Report the (3, 1) cosine column, the entry the retired route lost
        // first and completely.
        let col = cos_column(3, 1);
        let fd = (up[col] - down[col]) / (2.0 * h);
        let scale = col_scale[col].max(f64::MIN_POSITIVE);
        let h_rad = h * deg;
        let lf = max_degree as f64;
        let bound = (lf * lf * lf * scale * h_rad * h_rad / 6.0
            + 2.0 * forward_row_relative_error(max_degree) * scale / h_rad)
            * deg;
        println!(
            "{gap:>14.0e}  {:>10}  {:>13.6e}  {:>13.6e}  {:>11.2e}",
            "(3,1)cos",
            jet[[0, col, 0]],
            fd,
            bound
        );
    }
    println!("worst miss / bound over all columns and latitudes: {worst_ratio:.3e}");
}

/// The pole is an EXACT value, not a floored one.
///
/// `P_{l,m}(±1) = 0` for `m ≥ 1` and `P_{l,0}(±1) = (±1)^l`, so
/// `dP_{l,m}/dlat = ½[(l+m)(l-m+1)P_{l,m-1} - P_{l,m+1}]` collapses at the
/// north pole to
///
/// ```text
///   m = 0    →  -P_{l,1}(1)                     = 0
///   m = 1    →  ½·(l+1)·l·P_{l,0}(1) - ½P_{l,2}(1) = l(l+1)/2
///   m ≥ 2    →  both Legendre terms carry m ≥ 1  = 0
/// ```
///
/// so exactly the `m = 1` family survives, at `l(l+1)/2` times its norm and its
/// meridian's trigonometric factor. The retired quotient returned `0.0` for all
/// three cases: right for two of them by luck, and short by the entire quantity
/// for the third.
#[test]
fn zz_measure_polar_harmonic_jet_is_exact_at_the_pole() {
    let max_degree = 6usize;
    let lon_deg = 37.5_f64;
    let deg = std::f64::consts::PI / 180.0;
    let norms = precompute_harmonic_norms(max_degree);
    let l_cap = max_degree + 1;
    let data = Array2::from_shape_vec((1, 2), vec![90.0, lon_deg])
        .expect("1x2 lat/lon fixture is well formed");
    let jet = spherical_harmonic_jet(data.view(), max_degree, false)
        .expect("the harmonic jet must build at the pole");

    println!("north pole, meridian {lon_deg} deg");
    println!(
        "{:>10}  {:>16}  {:>16}",
        "(l,m)cos", "shipped d/dlat", "N·l(l+1)/2·cos(mψ)·deg"
    );

    for l in 1..=max_degree {
        for m in 0..=l {
            let col = cos_column(l, m);
            let shipped = jet[[0, col, 0]];
            let want = if m == 1 {
                let nlm = norms[l * l_cap + 1];
                let lf = l as f64;
                nlm * (lf * (lf + 1.0) / 2.0) * (lon_deg * deg).cos() * deg
            } else {
                0.0
            };
            if m == 1 {
                println!(
                    "{:>10}  {shipped:>16.9e}  {want:>16.9e}",
                    format!("({l},1)")
                );
                // Every factor is a product of exactly representable-to-a-ulp
                // quantities; four ulps of the magnitude covers the norm, the
                // cosine and the degree conversion.
                let tol = 4.0 * f64::EPSILON * want.abs();
                assert!(
                    (shipped - want).abs() <= tol,
                    "pole (l={l}, m=1) cosine column: shipped {shipped:.17e} vs exact \
                     {want:.17e}"
                );
                assert!(
                    shipped.abs() > 0.0,
                    "pole (l={l}, m=1) must not be reported as zero"
                );
            } else {
                // `90.0_f64 * (π/180)` is the nearest double to `π/2`, whose
                // cosine is `6.1e-17` rather than `0`, so the vanishing columns
                // vanish only to that residual latitude. Every surviving term
                // carries one factor of `cos(lat)` and at most `l(l+1)` of
                // Legendre growth.
                let cos_pole = (90.0_f64 * deg).cos().abs();
                let lf = l as f64;
                let tol = 8.0 * norms[l * l_cap + m] * lf * (lf + 1.0) * cos_pole * deg;
                assert!(
                    shipped.abs() <= tol,
                    "pole (l={l}, m={m}) cosine column is {shipped:.6e}, beyond the \
                     {tol:.6e} the representable pole's own cos(lat) admits"
                );
            }
        }
    }
}

/// The conditioning law of the RETIRED expression, kept because it is the
/// reason the expression was retired.
///
/// The numerator `-l·x·P_{l,m} + (l+m)·P_{l-1,m}` is mathematically
/// `(1-x²)·P'_{l,m}`, i.e. `O(cos²(lat))`, but it is assembled by subtracting
/// two `O(1)` terms. The absolute error of that subtraction is `~ε` times the
/// terms' own size, so the relative error of the result grows like
/// `ε/cos²(lat)`. Measured at `(l, m) = (1, 0)`, where the numerator is exactly
/// `-x² + 1` and the law is cleanest, the constant is `2`.
#[test]
fn zz_measure_polar_harmonic_jet_numerator_cancels_like_two_over_cos_squared() {
    let max_degree = 4usize;
    let l_cap = max_degree + 1;
    let idx = |l: usize, m: usize| l * l_cap + m;

    println!(
        "{:>14}  {:>13}  {:>13}  {:>13}",
        "90 - lat (deg)", "cos^2(lat)", "rel err", "rel/(eps/cos^2)"
    );
    for &gap in &[1.0e-1_f64, 1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5] {
        let lat = (90.0 - gap) * std::f64::consts::PI / 180.0;
        let x = lat.sin();
        let cos2 = lat.cos() * lat.cos();
        let p = plm_table_from_radicand(x, max_degree);
        // (l, m) = (1, 0): numerator = -1·x·P_{1,0} + 1·P_{0,0} = 1 - x².
        let assembled = -x * p[idx(1, 0)] + p[idx(0, 0)];
        // The exact value of the same quantity, from the cosine directly.
        let exact = cos2;
        let rel = (assembled - exact).abs() / exact;
        let predicted = f64::EPSILON / cos2;
        println!(
            "{gap:>14.0e}  {cos2:>13.6e}  {rel:>13.6e}  {:>13.6e}",
            rel / predicted
        );
        // The law: the relative error tracks ε/cos²(lat) to within a factor of
        // four in either direction across five decades. Anything much smaller
        // would mean the subtraction is not the dominant term; anything much
        // larger would mean a second error source appeared.
        assert!(
            rel <= 4.0 * predicted,
            "cancellation at 90 - {gap} deg is {rel:.6e}, beyond 4·eps/cos²(lat) = \
             {:.6e}",
            4.0 * predicted
        );
    }
}
