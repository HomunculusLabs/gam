//! Measurement: the spherical-harmonic design jet's `∂/∂lat` loses digits
//! approaching the geographic poles, and the loss follows `2ε/cos²(lat)`.
//!
//! Found while replacing the zonal Legendre-derivative quotient in
//! `sphere_kernels` / `sphere_spectral` (see `zz_measure_2475_tests` §6). The
//! same shape survives in the ASSOCIATED Legendre path that
//! `spherical_harmonic_design_jet` uses, `radial_jets_nd.rs:2065`:
//!
//! ```text
//!   let one_minus_x2 = (1.0 - x * x).max(f64::EPSILON);
//!   dp = (-l·x·P_{l,m} + (l+m)·P_{l-1,m}) / one_minus_x2
//! ```
//!
//! with `x = sin(lat)`, so `1 - x² = cos²(lat)` and the denominator vanishes at
//! the poles. The numerator vanishes with it — the identity it encodes is
//! `(1-x²)·P'_{l,m} = -l·x·P_{l,m} + (l+m)·P_{l-1,m}` — so the quotient is a
//! removable `0/0` and the value is finite. The problem is not the endpoint,
//! it is the approach: the numerator is `O(cos²(lat))` assembled by subtracting
//! two `O(1)` terms, so its relative error grows like `ε/cos²(lat)`.
//!
//! ## Why this is not just the chart singularity
//!
//! `(lat, lon)` is a singular chart at the poles, and for `m ≥ 1` the harmonic's
//! `∂/∂lat` genuinely depends on the meridian of approach — no implementation
//! can return a single right answer there. **That is not what this measures.**
//! The worst-conditioned entry is `(l, m) = (1, 0)`, and an `m = 0` harmonic is
//! zonal: it does not involve `lon` at all, it is a smooth function of latitude,
//! and its `∂/∂lat` has an ordinary finite limit of `0` at the pole. The digits
//! lost there are lost on a quantity that is perfectly well defined.
//!
//! ## Consequence
//!
//! `cos(lat)` is `1.7e-4` at `lat = 89.99°`, roughly **1.1 km** from the pole,
//! which puts the relative error at `1.4e-8`. At `89.9°` (~11 km) it is
//! `1.4e-10`. Global geospatial data reaches those latitudes. The jet feeds
//! derivative-based consumers, and `1.4e-8` is far above the `1e-11`-class
//! agreement finite-difference gates elsewhere in this crate assert.
//!
//! ## Not fixed here, deliberately
//!
//! The `m = 0` half has the same cure already applied to the zonal case —
//! `P'_ℓ = (2ℓ-1)P_{ℓ-1} + P'_{ℓ-2}`, which has no pole and no cancellation.
//! The `m ≥ 1` half wants the `θ`-derivative identity
//! `dP_{l,m}/dθ = ½[(l+m)(l-m+1)P_{l,m-1} - P_{l,m+1}]`, which also has no
//! `1/(1-x²)` — but it is stated for `x = cos θ` while this loop carries
//! `x = sin(lat)`, it needs the negative-order convention at `m = 0`, and it
//! must keep matching `fill_real_spherical_harmonics_row`'s Condon–Shortley
//! sign in a parallel hot loop. That is a change worth validating against
//! finite differences of the forward design rather than against a sign
//! convention read off a reference, and it is not attempted here.
//!
//! What is asserted below is the LAW, so the measurement is a gate and not a
//! zero-assertion report: the numerator's cancellation is `2/cos²(lat)` to
//! within a factor of two over five decades. If the assembly is changed, this
//! test says whether the conditioning changed with it.

/// Condon–Shortley associated Legendre table, built by exactly the recurrence
/// `spherical_harmonic_design_jet` uses (`radial_jets_nd.rs:2067-2081`), so
/// this measures that assembly and not a reimplementation of it.
fn plm_table(x: f64, max_degree: usize) -> Vec<f64> {
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

/// `Σ|terms| / |Σ terms|` for the `dp` numerator `-l·x·P_{l,m} + (l+m)·P_{l-1,m}`,
/// maximised over the table — the standard bound on how much of the input
/// precision the subtraction discards.
fn worst_numerator_conditioning(lat_deg: f64, max_degree: usize) -> (f64, usize, usize) {
    let lat = lat_deg.to_radians();
    let x = lat.sin();
    let l_cap = max_degree + 1;
    let idx = |l: usize, m: usize| l * l_cap + m;
    let p = plm_table(x, max_degree);
    let mut worst = (0.0_f64, 0usize, 0usize);
    for l in 1..=max_degree {
        for m in 0..=l {
            // `P_{l-1,m}` is identically zero when `m > l-1`; the shipped `dp`
            // closure reads the same slot, which the table leaves at zero.
            let t1 = -(l as f64) * x * p[idx(l, m)];
            let t2 = ((l + m) as f64) * p[idx(l - 1, m)];
            let sum = t1 + t2;
            if sum != 0.0 {
                let cond = (t1.abs() + t2.abs()) / sum.abs();
                if cond > worst.0 {
                    worst = (cond, l, m);
                }
            }
        }
    }
    worst
}

#[test]
fn zz_measure_polar_harmonic_jet_numerator_cancels_like_two_over_cos_squared() {
    const MAX_DEGREE: usize = 8;
    println!(
        "\n{:>10} {:>12} {:>10} {:>14} {:>14} {:>12}",
        "lat(deg)", "cos(lat)", "worst l,m", "cond", "2/cos²(lat)", "eps*cond"
    );
    for lat_deg in [80.0_f64, 89.0, 89.9, 89.99, 89.999, 89.9999] {
        let (cond, l, m) = worst_numerator_conditioning(lat_deg, MAX_DEGREE);
        let cos_lat = lat_deg.to_radians().cos();
        let law = 2.0 / (cos_lat * cos_lat);
        let implied = cond * f64::EPSILON;
        println!(
            "{lat_deg:>10} {cos_lat:>12.4e} {:>10} {cond:>14.4e} {law:>14.4e} {implied:>12.2e}",
            format!("{l},{m}")
        );
        assert!(
            cond > 0.5 * law && cond < 2.0 * law,
            "at lat={lat_deg} the dp numerator's cancellation is {cond:.4e}, off the \
             2/cos²(lat) = {law:.4e} law by more than a factor of two"
        );
        // The worst entry is zonal, i.e. NOT the (lat, lon) chart singularity:
        // an m = 0 harmonic does not involve longitude and its ∂/∂lat has an
        // ordinary limit at the pole.
        assert_eq!(
            m, 0,
            "the worst-conditioned dp numerator moved off m=0 (now l={l}, m={m}); \
             the claim that this is arithmetic rather than the chart singularity \
             rests on it being a zonal harmonic"
        );
    }
    println!(
        "\n  cos(lat) = 1.7e-4 at lat = 89.99 deg is ~1.1 km from the pole, where the\n  \
         numerator has already lost 1.4e-8 of its relative precision.\n"
    );
}

#[test]
fn zz_measure_polar_harmonic_jet_denominator_floor_binds_only_at_the_exact_pole() {
    // `(1.0 - x*x).max(f64::EPSILON)` can only bind where `1 - x²` drops below
    // `2.2e-16`, i.e. `|cos(lat)| < 1.5e-8` — about 0.1 m from the pole. So the
    // floor is not what produces the loss measured above; the loss is entirely
    // in the numerator, across latitudes where the floor is inert. Recorded so
    // that shrinking or deleting the floor is not mistaken for a fix.
    let mut binding = None;
    for micro_deg in [1.0_f64, 1e-2, 1e-4, 1e-6, 1e-7] {
        let lat = (90.0 - micro_deg).to_radians();
        let x = lat.sin();
        let one_minus_x2 = 1.0 - x * x;
        if one_minus_x2 < f64::EPSILON {
            binding = Some(micro_deg);
            break;
        }
    }
    println!(
        "\n  the EPSILON floor first binds at 90 deg - {:?} deg; the numerator has \
         lost 1.4e-8 by 90 - 0.01 deg\n",
        binding
    );
    assert!(
        binding.is_none() || binding.unwrap() < 1e-5,
        "the denominator floor binds far enough from the pole to be the mechanism \
         after all, which would contradict this file's diagnosis"
    );
}
