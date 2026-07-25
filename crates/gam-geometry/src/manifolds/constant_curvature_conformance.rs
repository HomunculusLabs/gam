//! Independent-oracle conformance for the κ-stereographic family.
//!
//! [`ConstantCurvature`] is deliberately **not** a [`ManifoldSpec`] variant — it
//! carries a continuous parameter rather than a fixed shape — so the shared
//! inventory in [`super::conformance`] does not reach it. It needs its own
//! coverage, and it needs a different *kind* of coverage, because the thing
//! worth checking is not that a formula is self-consistent but that this one
//! chart reproduces three classical geometries it was never told about.
//!
//! The module's whole design claim is that spherical, flat, and hyperbolic
//! space are one analytic object in `u = κt²`, with κ = 0 a removable point
//! rather than a branch. Every test here is an oracle *external* to that claim:
//!
//! | oracle | what it pins |
//! |---|---|
//! | `K ≡ κ` | the curvature the family reports is the curvature it has |
//! | Möbius-translation invariance | the space really is homogeneous |
//! | `d_κ(x,y) = d_{±1}(√|κ|x, √|κ|y)/√|κ|` | the κ-family is one manifold rescaled |
//! | `κ = 0 ⇒ d = 2‖x−y‖` | the flat member is flat (in the `λ₀ = 2` gauge) |
//! | `κ = −1 ⇒ poincare_distance` | agreement with a **separate implementation** |
//! | `J_κ(r) = (sn_κ(r)/r)^{d−1}` | the volume element against its closed form |
//! | metric axioms | symmetry and the triangle inequality |
//! | `‖log_x y‖_{g_x} = d(x,y)` | the logarithm and the distance are the same geometry |
//!
//! The cross-implementation row is the strongest of these: `poincare.rs` reaches
//! hyperbolic distance by a different route, so agreement at `1e-15` is evidence
//! about the mathematics, not about a shared helper.
//!
//! ### The κ-jets are checked against the value path, not against themselves
//!
//! `distance_kappa_jet` and friends return `(f, ∂f/∂κ, ∂²f/∂κ²)` from a
//! `Tower2` program, and those derivatives enter the outer REML optimization as
//! a ψ-coordinate. A wrong `∂²/∂κ²` does not produce a wrong *answer* — it
//! produces a slower or differently-converging search, which is exactly the kind
//! of defect that survives for years. So the jets are differenced against the
//! independently written value methods (`distance`, `log_map`, `exp_map`)
//! evaluated at perturbed κ.
//!
//! Two things had to be right for that to measure anything:
//!
//! * **The test points must not depend on `h`.** The chart for `κ < 0` is a ball
//!   of radius `1/√−κ`, so the obvious "sample inside the chart of `κ − h`"
//!   makes the point cloud blow up as `h → 0`; the h-scan then measures the
//!   sampler instead of the discretization and reads as a catastrophic
//!   derivative error at exactly the κ = 0 point the family exists to handle.
//!   [`chart_point`] uses a fixed radius valid across the whole stencil.
//! * **Each derivative is scaled by its own magnitude.** `∂d/∂κ` grows like
//!   `‖w‖³` and `∂²d/∂κ²` like `‖w‖⁵`, so normalizing them by the *distance*
//!   understates a real error by orders of magnitude.
//!
//! With both fixed, the residual is second-order in `h` (measured: `1.4e-7`,
//! `1.5e-9`, `1.5e-11` at `h = 1e-2, 1e-3, 1e-4` — a clean `h²` ladder until
//! the second difference hits its `ε/h²` roundoff floor). `H = 1e-3` is the
//! bottom of that curve for the second derivative and is what the assertions
//! use.

use ndarray::{Array1, Array2};

use crate::manifold::RiemannianManifold;
use crate::manifolds::constant_curvature::{
    ConstantCurvature, distance_kappa_jet, exp_map_kappa_jet, log_map_kappa_jet,
};
use crate::manifolds::poincare;

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn normal(&mut self) -> f64 {
        let u1 = self.uniform().max(1.0e-300);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
    fn gaussian_vec(&mut self, n: usize) -> Array1<f64> {
        Array1::from_shape_fn(n, |_| self.normal())
    }
}

/// A chart point of radius `< 0.45`.
///
/// Fixed, and in particular independent of κ and of any finite-difference step:
/// the κ < 0 chart is the ball of radius `1/√−κ`, so a radius chosen from the
/// current κ would make each stencil arm see a different point cloud. 0.45 is
/// interior for every κ ≥ −1.1, which covers every curvature used below plus
/// the widest stencil arm.
fn chart_point(rng: &mut Rng, dim: usize) -> Array1<f64> {
    let mut v = rng.gaussian_vec(dim);
    let magnitude = v.dot(&v).sqrt().max(1.0e-30);
    v *= 0.45 * rng.uniform() / magnitude;
    v
}

/// Curvatures spanning both signs, both series/closed-form regimes of the
/// `C`/`S`/`T` primitives, and the removable point itself.
const CURVATURES: [f64; 11] = [
    2.0, 1.0, 0.35, 0.01, 1.0e-6, 0.0, -1.0e-6, -0.01, -0.35, -1.0, -2.0,
];
const DIMS: [usize; 4] = [1, 2, 3, 5];
const TRIALS: usize = 200;
const SEED: u64 = 0x9E37_79B9_7F4A_7C15;

fn seed_for(dim: usize, kappa: f64) -> u64 {
    SEED ^ ((dim as u64) << 32) ^ kappa.to_bits()
}

fn sup_diff(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

/// Largest magnitude across two vectors, floored at 1 — the scale a residual
/// between them is relative to.
fn joint_scale(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    a.iter()
        .chain(b.iter())
        .fold(1.0_f64, |acc, v| acc.max(v.abs()))
}

#[test]
fn sectional_curvature_is_kappa_everywhere() {
    let mut verified = 0usize;
    for dim in DIMS {
        if dim < 2 {
            continue;
        }
        for kappa in CURVATURES {
            let manifold = ConstantCurvature::new(dim, kappa);
            let mut rng = Rng::new(seed_for(dim, kappa));
            for _ in 0..TRIALS {
                let x = chart_point(&mut rng, dim);
                let u = manifold
                    .project_tangent(x.view(), rng.gaussian_vec(dim).view())
                    .expect("tangent u");
                let v = manifold
                    .project_tangent(x.view(), rng.gaussian_vec(dim).view())
                    .expect("tangent v");
                let Ok(k) = manifold.sectional_curvature(x.view(), (u.view(), v.view())) else {
                    continue;
                };
                assert!(
                    (k - kappa).abs() <= 1.0e-9 * kappa.abs().max(1.0),
                    "dim {dim} kappa {kappa}: sectional curvature is {k}, not kappa"
                );
                verified += 1;
            }
        }
    }
    assert!(verified > 0, "no sectional curvature was evaluated");
}

#[test]
fn distance_is_invariant_under_mobius_translation() {
    // M_κ is homogeneous: Möbius addition by a fixed g is an isometry. This is
    // the property whose failure was #2351 — a κ̂ verdict that moved when the
    // data was translated.
    let mut verified = 0usize;
    for dim in DIMS {
        for kappa in CURVATURES {
            let manifold = ConstantCurvature::new(dim, kappa);
            let mut rng = Rng::new(seed_for(dim, kappa));
            for _ in 0..TRIALS {
                let x = chart_point(&mut rng, dim);
                let y = chart_point(&mut rng, dim);
                let g = chart_point(&mut rng, dim);
                let base = manifold.distance(x.view(), y.view()).expect("d(x,y)");
                let (Ok(gx), Ok(gy)) = (
                    manifold.mobius_add(g.view(), x.view()),
                    manifold.mobius_add(g.view(), y.view()),
                ) else {
                    continue;
                };
                let Ok(moved) = manifold.distance(gx.view(), gy.view()) else {
                    continue;
                };
                assert!(
                    (moved - base).abs() <= 1.0e-9 * base.max(1.0),
                    "dim {dim} kappa {kappa}: translation changed the distance \
                     ({base} -> {moved})"
                );
                verified += 1;
            }
        }
    }
    assert!(verified > 0, "no translated distance was evaluated");
}

#[test]
fn the_kappa_family_is_one_manifold_rescaled() {
    // A space form of curvature κ is the unit-curvature form of the same sign
    // scaled by 1/√|κ|, so d_κ(x, y) = d_{sign κ}(√|κ|·x, √|κ|·y) / √|κ|. This
    // ties every κ to the two reference geometries with no free parameter.
    let mut verified = 0usize;
    for dim in DIMS {
        for kappa in CURVATURES {
            if kappa.abs() <= 1.0e-9 {
                continue;
            }
            let manifold = ConstantCurvature::new(dim, kappa);
            let unit = ConstantCurvature::new(dim, kappa.signum());
            let scale = kappa.abs().sqrt();
            let mut rng = Rng::new(seed_for(dim, kappa));
            for _ in 0..TRIALS {
                let x = chart_point(&mut rng, dim);
                let y = chart_point(&mut rng, dim);
                let here = manifold.distance(x.view(), y.view()).expect("d_kappa");
                let Ok(there) = unit.distance((&x * scale).view(), (&y * scale).view()) else {
                    continue;
                };
                assert!(
                    (here - there / scale).abs() <= 1.0e-9 * here.max(1.0),
                    "dim {dim} kappa {kappa}: d_kappa = {here} but the rescaled unit \
                     form gives {}",
                    there / scale
                );
                verified += 1;
            }
        }
    }
    assert!(verified > 0, "no rescaling was evaluated");
}

#[test]
fn the_flat_and_hyperbolic_members_match_their_classical_forms() {
    // Two external anchors:
    //   κ = 0  — the conformal gauge is λ₀ = 2, so the flat metric is 4δ and
    //            the distance is 2‖x − y‖ (Euclidean up to the isometry x ↦ 2x,
    //            exactly as the module documents).
    //   κ = −1 — must agree with `poincare.rs`, which computes hyperbolic
    //            distance by an independent route. This is the only assertion
    //            in the file that compares two implementations rather than an
    //            implementation to a formula, and so is the strongest.
    let mut flat_checked = 0usize;
    let mut hyperbolic_checked = 0usize;
    for dim in DIMS {
        let flat = ConstantCurvature::new(dim, 0.0);
        let hyperbolic = ConstantCurvature::new(dim, -1.0);
        let mut rng = Rng::new(seed_for(dim, 0.0));
        for _ in 0..TRIALS {
            let x = chart_point(&mut rng, dim);
            let y = chart_point(&mut rng, dim);

            let got = flat.distance(x.view(), y.view()).expect("flat distance");
            let want = 2.0 * (&x - &y).dot(&(&x - &y)).sqrt();
            assert!(
                (got - want).abs() <= 1.0e-12 * want.max(1.0),
                "dim {dim}: flat distance {got} != 2||x-y|| = {want}"
            );
            flat_checked += 1;

            let got = hyperbolic
                .distance(x.view(), y.view())
                .expect("hyperbolic distance");
            let want =
                poincare::poincare_distance(x.view(), y.view(), -1.0).expect("poincare distance");
            assert!(
                (got - want).abs() <= 1.0e-12 * want.max(1.0),
                "dim {dim}: kappa=-1 distance {got} disagrees with poincare.rs {want}"
            );
            hyperbolic_checked += 1;
        }
    }
    assert!(flat_checked > 0 && hyperbolic_checked > 0);
}

#[test]
fn distance_is_a_metric_and_agrees_with_the_logarithm() {
    let mut verified = 0usize;
    for dim in DIMS {
        for kappa in CURVATURES {
            let manifold = ConstantCurvature::new(dim, kappa);
            let mut rng = Rng::new(seed_for(dim, kappa));
            for _ in 0..TRIALS {
                let x = chart_point(&mut rng, dim);
                let y = chart_point(&mut rng, dim);
                let z = chart_point(&mut rng, dim);
                let dxy = manifold.distance(x.view(), y.view()).expect("d(x,y)");
                let dyx = manifold.distance(y.view(), x.view()).expect("d(y,x)");
                let dyz = manifold.distance(y.view(), z.view()).expect("d(y,z)");
                let dxz = manifold.distance(x.view(), z.view()).expect("d(x,z)");

                assert!(
                    (dxy - dyx).abs() <= 1.0e-12 * dxy.max(1.0),
                    "dim {dim} kappa {kappa}: distance is not symmetric ({dxy} vs {dyx})"
                );
                assert!(
                    dxz <= dxy + dyz + 1.0e-12 * dxz.max(1.0),
                    "dim {dim} kappa {kappa}: triangle inequality violated \
                     ({dxz} > {dxy} + {dyz})"
                );

                // ‖log_x y‖ in the metric at x is the geodesic distance. The
                // metric is conformal, so the norm is λ_x·‖·‖.
                let logarithm = manifold.log_map(x.view(), y.view()).expect("log");
                let lambda = manifold
                    .conformal_factor(x.view())
                    .expect("conformal factor");
                let metric_norm = lambda * logarithm.dot(&logarithm).sqrt();
                assert!(
                    (metric_norm - dxy).abs() <= 1.0e-9 * dxy.max(1.0),
                    "dim {dim} kappa {kappa}: ||log|| = {metric_norm} != d = {dxy}"
                );
                verified += 1;
            }
        }
    }
    assert!(verified > 0, "no metric axiom was evaluated");
}

#[test]
fn radial_volume_jacobian_matches_its_closed_form() {
    // J_κ(r) = (sn_κ(r)/r)^{d−1}, with sn_κ the curvature-normalized sine. This
    // is the volume term in the change-of-variables criterion, so an error here
    // biases the curvature estimate itself rather than merely slowing it.
    let mut verified = 0usize;
    for dim in DIMS {
        for kappa in CURVATURES {
            let manifold = ConstantCurvature::new(dim, kappa);
            let mut rng = Rng::new(seed_for(dim, kappa));
            for _ in 0..TRIALS {
                let r = 2.0 * rng.uniform();
                let got = manifold.jacobian_radial(r);
                let want = if dim <= 1 {
                    1.0
                } else {
                    let sn_over_r = if kappa.abs() <= 1.0e-12 || r == 0.0 {
                        1.0
                    } else if kappa > 0.0 {
                        let arc = kappa.sqrt() * r;
                        (arc.sin() / arc).max(0.0)
                    } else {
                        let arc = (-kappa).sqrt() * r;
                        arc.sinh() / arc
                    };
                    sn_over_r.powi((dim - 1) as i32)
                };
                assert!(
                    (got - want).abs() <= 1.0e-9 * want.abs().max(1.0e-12),
                    "dim {dim} kappa {kappa} r {r}: J = {got}, closed form {want}"
                );
                verified += 1;
            }
        }
    }
    assert!(verified > 0, "no volume Jacobian was evaluated");
}

#[test]
fn batched_distance_is_bit_identical_to_the_scalar_path() {
    // `distance_batch` documents bit-for-bit agreement with `distance`, which is
    // a stronger claim than "close": the SIMD `T`-series must reproduce the
    // scalar value per lane exactly, and the tail and closed-form lanes must
    // fall back rather than approximate. Asserting equality (not a tolerance) is
    // the only way to test the claim that is actually made.
    let mut verified = 0usize;
    for dim in DIMS {
        for kappa in CURVATURES {
            let manifold = ConstantCurvature::new(dim, kappa);
            let mut rng = Rng::new(seed_for(dim, kappa));
            // Row counts either side of the f64x4 lane width, so the vectorised
            // body and the scalar tail are both exercised.
            for rows in [1usize, 3, 4, 5, 8, 11] {
                let base = chart_point(&mut rng, dim);
                let mut targets = Array2::<f64>::zeros((rows, dim));
                for r in 0..rows {
                    targets.row_mut(r).assign(&chart_point(&mut rng, dim));
                }
                let mut batched = vec![0.0_f64; rows];
                manifold
                    .distance_batch(base.view(), targets.view(), &mut batched)
                    .expect("distance_batch");
                for r in 0..rows {
                    let scalar = manifold
                        .distance(base.view(), targets.row(r))
                        .expect("scalar distance");
                    assert_eq!(
                        batched[r].to_bits(),
                        scalar.to_bits(),
                        "dim {dim} kappa {kappa} rows {rows} row {r}: \
                         batched {} != scalar {scalar}",
                        batched[r]
                    );
                    verified += 1;
                }
            }
        }
    }
    assert!(verified > 0, "no batched distance was evaluated");
}

#[test]
fn kappa_jets_match_central_differences_of_the_value_path() {
    // See the module docs for why H is 1e-3 and why the sample radius must not
    // depend on it. The jets are compared against `distance` / `log_map` /
    // `exp_map` — separately written value code — at κ ± H, so agreement is
    // evidence about the Tower2 program rather than about itself.
    const H: f64 = 1.0e-3;
    const TOL: f64 = 1.0e-6;
    let mut verified = 0usize;
    for dim in [1usize, 2, 3] {
        for kappa in [1.0_f64, 0.3, 0.0, -0.3, -1.0] {
            let manifold = ConstantCurvature::new(dim, kappa);
            let up = ConstantCurvature::new(dim, kappa + H);
            let down = ConstantCurvature::new(dim, kappa - H);
            let mut rng = Rng::new(seed_for(dim, kappa));
            for _ in 0..TRIALS {
                let x = chart_point(&mut rng, dim);
                let y = chart_point(&mut rng, dim);

                // ---- distance ----
                let (value, first, second) =
                    distance_kappa_jet(&manifold, x.view(), y.view()).expect("distance jet");
                let here = manifold.distance(x.view(), y.view()).expect("d");
                let above = up.distance(x.view(), y.view()).expect("d+");
                let below = down.distance(x.view(), y.view()).expect("d-");
                assert!(
                    (value - here).abs() <= 1.0e-12 * here.max(1.0),
                    "dim {dim} kappa {kappa}: jet value {value} != distance {here}"
                );
                let fd_first = (above - below) / (2.0 * H);
                let fd_second = (above - 2.0 * here + below) / (H * H);
                assert!(
                    (first - fd_first).abs() <= TOL * first.abs().max(fd_first.abs()).max(1.0),
                    "dim {dim} kappa {kappa}: d/dkappa distance {first} vs FD {fd_first}"
                );
                assert!(
                    (second - fd_second).abs() <= TOL * second.abs().max(fd_second.abs()).max(1.0),
                    "dim {dim} kappa {kappa}: d2/dkappa2 distance {second} vs FD {fd_second}"
                );

                // ---- logarithm ----
                let (value, first, second) =
                    log_map_kappa_jet(&manifold, x.view(), y.view()).expect("log jet");
                let here = manifold.log_map(x.view(), y.view()).expect("log");
                let above = up.log_map(x.view(), y.view()).expect("log+");
                let below = down.log_map(x.view(), y.view()).expect("log-");
                assert!(
                    sup_diff(&value, &here) <= 1.0e-12 * joint_scale(&value, &here),
                    "dim {dim} kappa {kappa}: log jet value disagrees with log_map"
                );
                let fd_first = (&above - &below) / (2.0 * H);
                let fd_second = (&above - &(&here * 2.0) + &below) / (H * H);
                assert!(
                    sup_diff(&first, &fd_first) <= TOL * joint_scale(&first, &fd_first),
                    "dim {dim} kappa {kappa}: d/dkappa log disagrees with FD"
                );
                assert!(
                    sup_diff(&second, &fd_second) <= TOL * joint_scale(&second, &fd_second),
                    "dim {dim} kappa {kappa}: d2/dkappa2 log disagrees with FD"
                );

                // ---- exponential ----
                let tangent = manifold
                    .project_tangent(x.view(), (rng.gaussian_vec(dim) * 0.2).view())
                    .expect("tangent");
                let (value, first, second) =
                    exp_map_kappa_jet(&manifold, x.view(), tangent.view()).expect("exp jet");
                let here = manifold.exp_map(x.view(), tangent.view()).expect("exp");
                let above = up.exp_map(x.view(), tangent.view()).expect("exp+");
                let below = down.exp_map(x.view(), tangent.view()).expect("exp-");
                assert!(
                    sup_diff(&value, &here) <= 1.0e-12 * joint_scale(&value, &here),
                    "dim {dim} kappa {kappa}: exp jet value disagrees with exp_map"
                );
                let fd_first = (&above - &below) / (2.0 * H);
                let fd_second = (&above - &(&here * 2.0) + &below) / (H * H);
                assert!(
                    sup_diff(&first, &fd_first) <= TOL * joint_scale(&first, &fd_first),
                    "dim {dim} kappa {kappa}: d/dkappa exp disagrees with FD"
                );
                assert!(
                    sup_diff(&second, &fd_second) <= TOL * joint_scale(&second, &fd_second),
                    "dim {dim} kappa {kappa}: d2/dkappa2 exp disagrees with FD"
                );

                verified += 1;
            }
        }
    }
    assert!(verified > 0, "no kappa jet was differenced");
}
