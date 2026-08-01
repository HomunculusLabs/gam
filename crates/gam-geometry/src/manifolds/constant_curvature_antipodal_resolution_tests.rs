//! PROBE (temporary, #2687): how accurate are the shipped κ-stereographic
//! distance and its κ-jets as an evaluated pair approaches the antipodal fold?
//!
//! The κ search box's upper end is `CONSTANT_CURVATURE_KAPPA_CHART_FRACTION /
//! R²` with the fraction `0.5`, documented as a "half-margin" to the fold at
//! `κ‖x‖‖y‖ = 1`. Nothing in the tree states what the margin *buys*. This probe
//! measures it: for an anti-aligned equal-radius pair the geodesic distance has
//! the exact closed form
//!
//! ```text
//!   d(κ) = (4/√κ)·arctan(√κ·R)
//! ```
//!
//! (the two colatitudes `θ = 2·arctan(√κ R)` add), which is perfectly
//! conditioned for every `t = κR² ∈ (0, ∞)` — no cancellation, no fold. The
//! shipped route goes through `w = (−x) ⊕_κ y`, whose denominator `(1 − t)²`
//! collapses at `t = 1`. Differencing the two isolates exactly what the margin
//! is protecting.
#![allow(clippy::print_stderr)]

use super::constant_curvature::{ConstantCurvature, distance_kappa_jet};

/// `d(κ) = 4·arctan(R√κ)/√κ` for the anti-aligned pair `(±R, 0)`, and its exact
/// first and second κ-derivatives. Every expression below is a sum of
/// same-sign terms in `1 + R²s²`, so it never cancels: this is the oracle.
fn antipodal_pair_reference(r: f64, kappa: f64) -> (f64, f64, f64) {
    let s = kappa.sqrt();
    let rs = r * s;
    let den = 1.0 + rs * rs;
    let at = rs.atan();
    // d(s) = 4·atan(Rs)/s
    let d = 4.0 * at / s;
    // N(s)  = Rs/(1+R²s²) − atan(Rs);      d'(s) = 4N/s²
    let n = rs / den - at;
    let d_s = 4.0 * n / (s * s);
    // N'(s) = −2R³s²/(1+R²s²)²  ·  (in the `rs` variable: −2·R·rs²/den²)
    let n_s = -2.0 * r * rs * rs / (den * den);
    let d_ss = 4.0 * (n_s / (s * s) - 2.0 * n / (s * s * s));
    // κ = s² ⇒ ∂/∂κ = (1/2s)∂/∂s, ∂²/∂κ² = (1/4s²)(∂²/∂s² − (1/s)∂/∂s)
    let d_k = d_s / (2.0 * s);
    let d_kk = (d_ss - d_s / s) / (4.0 * s * s);
    (d, d_k, d_kk)
}

#[test]
fn probe_antipodal_resolution_law() {
    const R: f64 = 0.6;
    let x = ndarray::array![R, 0.0];
    let y = ndarray::array![-R, 0.0];
    eprintln!("t=kappa*R^2   D=(1-t)^2      rel_err(d)    rel_err(d')   rel_err(d'')");
    for t in [
        0.1_f64, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999, 0.9999, 1.0 - 1e-6, 1.0 - 1e-8,
        1.0 - 1e-10, 1.0 - 1e-12,
    ] {
        let kappa = t / (R * R);
        let m = ConstantCurvature::new(2, kappa);
        let (dr, dkr, dkkr) = antipodal_pair_reference(R, kappa);
        let d = m.distance(x.view(), y.view());
        let jet = distance_kappa_jet(&m, x.view(), y.view());
        match (d, jet) {
            (Ok(dv), Ok((jv, jg, jh))) => {
                let e0 = ((dv - dr) / dr).abs();
                let e1 = ((jg - dkr) / dkr).abs();
                let e2 = ((jh - dkkr) / dkkr).abs();
                let _ = jv;
                eprintln!(
                    "{t:<13.10} {:<14.4e} {e0:<13.3e} {e1:<13.3e} {e2:<13.3e}",
                    (1.0 - t) * (1.0 - t)
                );
            }
            (dv, jv) => eprintln!("{t:<13.10} REFUSED d={dv:?} jet_err={}", jv.is_err()),
        }
    }

    // (b) The HYPERBOLIC wall, for symmetry of treatment. There the pair
    // denominator for an ALIGNED pair is `(1 − |κ|‖x‖‖y‖)²` and the per-point
    // chart gauge is `1 + κ‖x‖²`; with `‖x‖ = ‖y‖ = R` the two coincide, so the
    // same `D` sweep reaches the same place from the other side. The oracle is
    // the mirrored closed form `d(κ) = (4/√−κ)·artanh(√−κ·R)`.
    eprintln!("\n[hyperbolic, aligned pair at (R,0) and (0.999R,0)]");
    eprintln!("|t|           lambda_min     rel_err(d)    rel_err(d')   rel_err(d'')");
    for t in [
        0.1_f64, 0.5, 0.9, 0.99, 0.999, 0.9999, 1.0 - 1e-6, 1.0 - 1e-8, 1.0 - 1e-10,
    ] {
        let kappa = -t / (R * R);
        let m = ConstantCurvature::new(2, kappa);
        let a = ndarray::array![R, 0.0];
        let b = ndarray::array![0.5 * R, 0.0];
        // Oracle: hyperbolic distance between two collinear radii, from the
        // cancellation-free artanh form d = (2/√−κ)·(artanh(√−κ r1) −
        // artanh(√−κ r2)) — geodesics through the origin are radial.
        let s = (-kappa).sqrt();
        let oracle = 2.0 * ((s * R).atanh() - (s * 0.5 * R).atanh()) / s;
        let d = m.distance(a.view(), b.view());
        let jet = distance_kappa_jet(&m, a.view(), b.view());
        match (d, jet) {
            (Ok(dv), Ok((_, jg, jh))) => {
                // FD reference for the derivatives, on the oracle expression.
                let h = 1e-5 * kappa.abs().max(1.0);
                let f = |k: f64| {
                    let s = (-k).sqrt();
                    2.0 * ((s * R).atanh() - (s * 0.5 * R).atanh()) / s
                };
                let g_ref = (f(kappa + h) - f(kappa - h)) / (2.0 * h);
                let hh_ref = (f(kappa + h) - 2.0 * f(kappa) + f(kappa - h)) / (h * h);
                eprintln!(
                    "{t:<13.10} {:<14.4e} {:<13.3e} {:<13.3e} {:<13.3e}",
                    1.0 - t,
                    ((dv - oracle) / oracle).abs(),
                    ((jg - g_ref) / g_ref).abs(),
                    ((jh - hh_ref) / hh_ref).abs()
                );
            }
            (dv, jv) => eprintln!("{t:<13.10} REFUSED d_err={} jet_err={}", dv.is_err(), jv.is_err()),
        }
    }

    // (c) The law is about `D`, not about `t`: a NON-collinear pair reaches the
    // same `D` at a different `t`, and must show the same error.
    eprintln!("\n[general pair, mu = cos angle] D            rel_err(d')   rel_err(d'')");
    for mu in [-0.999_f64, -0.99, -0.95] {
        for target_d in [1e-4_f64, 1e-6, 1e-8] {
            // Solve 1 + 2 t mu + t^2 = D for the smaller positive root.
            let disc = mu * mu - (1.0 - target_d);
            if disc < 0.0 {
                eprintln!("mu={mu}: D={target_d:.0e} unreachable (mu^2 < 1-D)");
                continue;
            }
            let t = -mu - disc.sqrt();
            // Pair with ‖x‖ = ‖y‖ = R and angle arccos(mu); t = κR².
            let kappa = t / (R * R);
            let ang = mu.acos();
            let x2 = ndarray::array![R, 0.0];
            let y2 = ndarray::array![R * ang.cos(), R * ang.sin()];
            let m = ConstantCurvature::new(2, kappa);
            let jet = distance_kappa_jet(&m, x2.view(), y2.view());
            // FD oracle on the shipped value path at a well-conditioned step.
            let h = 1e-4 * kappa.abs();
            let f = |k: f64| {
                ConstantCurvature::new(2, k)
                    .distance(x2.view(), y2.view())
                    .unwrap_or(f64::NAN)
            };
            match jet {
                Ok((_, jg, jh)) => {
                    let g_ref = (f(kappa + h) - f(kappa - h)) / (2.0 * h);
                    let hh_ref = (f(kappa + h) - 2.0 * f(kappa) + f(kappa - h)) / (h * h);
                    eprintln!(
                        "mu={mu:<7} t={t:<8.5} D={target_d:<9.0e} g_rel={:<11.3e} h_rel={:<11.3e}",
                        ((jg - g_ref) / g_ref).abs(),
                        ((jh - hh_ref) / hh_ref).abs()
                    );
                }
                Err(e) => eprintln!("mu={mu} D={target_d:.0e} REFUSED {e:?}"),
            }
        }
    }

    // Same sweep, but reported in the *cut-locus gauge* c = D/(λx·λy) =
    // cos²(√κ·d/2), the dimensionless quantity that is 1 at κ=0 and 0 at
    // antipodality — the candidate denomination for the box's retreat.
    eprintln!("\nt            c=cos^2(sqrt(k)d/2)  antipodal_fraction");
    for t in [0.1_f64, 0.25, 0.5, 0.54, 0.75, 0.81, 0.9248, 0.99] {
        let kappa = t / (R * R);
        let (d, _, _) = antipodal_pair_reference(R, kappa);
        let c = (kappa.sqrt() * d / 2.0).cos().powi(2);
        let frac = d * kappa.sqrt() / std::f64::consts::PI;
        let c_alg = (1.0 - t) * (1.0 - t) / ((1.0 + t) * (1.0 + t));
        eprintln!("{t:<12.4} {c:<19.10} {frac:<12.6} c_alg={c_alg:.10}");
    }
}
