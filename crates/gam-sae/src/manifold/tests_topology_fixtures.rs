#![cfg(test)]
//! Point clouds of KNOWN topology, shared by every test that needs a manifold
//! whose answer is decided in advance (#2280).
//!
//! Each generator samples one specific manifold, so a test asserting what the
//! atlas recovered is comparing against a fact rather than against another run of
//! the same code. Sampling is deterministic grids — no RNG anywhere — so an atlas
//! built on any of these is bit-reproducible.
//!
//! The zoo is deliberately built to separate the invariants pairwise: circle vs
//! trefoil separates the INTRINSIC topology from the ambient embedding, cylinder
//! vs Möbius separates the orientation class at fixed homology, sphere vs plane
//! separates `χ` at fixed (trivial) `π₁`, and torus vs cylinder separates `b₁`.

use ndarray::Array2;

/// A swiss roll: a flat 2-D sheet `(t, h)` rolled into ambient 3-D. Folded in
/// the ambient metric (ambient-near points can be geodesically far), yet
/// intrinsically flat, so the transition cocycle around a contractible triangle
/// must close.
pub(crate) fn swiss_roll(n_t: usize, n_h: usize) -> Array2<f64> {
    let n = n_t * n_h;
    let mut z = Array2::<f64>::zeros((n, 3));
    let mut r = 0usize;
    for it in 0..n_t {
        // t over ~1.5 turns.
        let t = 1.0 + 3.0 * std::f64::consts::PI * (it as f64) / (n_t as f64 - 1.0);
        for ih in 0..n_h {
            let h = 2.0 * (ih as f64) / (n_h as f64 - 1.0);
            z[[r, 0]] = t * t.cos();
            z[[r, 1]] = t * t.sin();
            z[[r, 2]] = h;
            r += 1;
        }
    }
    z
}

/// A flat 2-D lattice embedded isometrically into 4-D by a fixed orthonormal
/// pair of ambient directions. Local PCA recovers the exact plane, so every
/// transition is an exact isometry and the cocycle closes to rounding — the
/// sharp cocycle-closure fixture. Topologically a disk (`χ = 1`, `b₁ = 0`).
pub(crate) fn embedded_plane(n_x: usize, n_y: usize) -> Array2<f64> {
    // Two orthonormal ambient directions in R^4.
    let u = [0.5, 0.5, 0.5, 0.5];
    let v = [0.5, -0.5, 0.5, -0.5];
    let n = n_x * n_y;
    let mut z = Array2::<f64>::zeros((n, 4));
    let mut r = 0usize;
    for ix in 0..n_x {
        for iy in 0..n_y {
            let a = ix as f64;
            let b = iy as f64;
            for c in 0..4 {
                z[[r, c]] = a * u[c] + b * v[c];
            }
            r += 1;
        }
    }
    z
}

/// A BAND on the unit 2-sphere: a lat/lon grid stopping short of both poles.
///
/// Topologically this is an annulus — a cylinder with curvature — NOT a sphere.
/// Removing the two polar caps removes the two 2-cells that make `χ = 2`, leaving
/// `b₁ = 1`, `b₂ = 0`, `χ = 0`. It is kept under its true name because it is a
/// useful curved-cylinder fixture and because naming it `sphere` invites exactly
/// the mistake of asserting `χ = 2` on a surface that does not have it.
pub(crate) fn spherical_band(n_lat: usize, n_lon: usize) -> Array2<f64> {
    let n = n_lat * n_lon;
    let mut z = Array2::<f64>::zeros((n, 3));
    let mut r = 0usize;
    for i in 0..n_lat {
        let lat = -1.2 + 2.4 * (i as f64) / (n_lat as f64 - 1.0); // in (−π/2, π/2)
        for j in 0..n_lon {
            let lon = std::f64::consts::TAU * (j as f64) / (n_lon as f64);
            z[[r, 0]] = lat.cos() * lon.cos();
            z[[r, 1]] = lat.cos() * lon.sin();
            z[[r, 2]] = lat.sin();
            r += 1;
        }
    }
    z
}

/// The CLOSED unit 2-sphere, sampled by the golden-angle spiral.
///
/// `z` is swept linearly and the azimuth advances by the golden angle, which makes
/// the sample near-uniform in area — unlike a lat/lon grid, whose longitudes crowd
/// toward the poles and hand the atlas a density spike where the curvature is
/// hardest. Deterministic: no RNG, only the golden angle.
pub(crate) fn sphere(n: usize) -> Array2<f64> {
    let golden_angle = std::f64::consts::PI * (3.0 - 5.0_f64.sqrt());
    let mut out = Array2::<f64>::zeros((n, 3));
    for i in 0..n {
        let height = 1.0 - 2.0 * (i as f64 + 0.5) / (n as f64);
        let radius = (1.0 - height * height).max(0.0).sqrt();
        let azimuth = golden_angle * (i as f64);
        out[[i, 0]] = radius * azimuth.cos();
        out[[i, 1]] = radius * azimuth.sin();
        out[[i, 2]] = height;
    }
    out
}

/// A cylinder strip: loop coordinate `u`, width `v` on a FIXED ambient axis, so
/// the width frame never flips — orientable.
pub(crate) fn cylinder_strip(n_u: usize, n_v: usize) -> Array2<f64> {
    let n = n_u * n_v;
    let mut z = Array2::<f64>::zeros((n, 3));
    let mut r = 0usize;
    for iu in 0..n_u {
        let u = std::f64::consts::TAU * (iu as f64) / (n_u as f64);
        for iv in 0..n_v {
            let v = -0.4 + 0.8 * (iv as f64) / (n_v as f64 - 1.0);
            z[[r, 0]] = 2.0 * u.cos();
            z[[r, 1]] = 2.0 * u.sin();
            z[[r, 2]] = v;
            r += 1;
        }
    }
    z
}

/// A Möbius strip: the standard half-twist embedding, so the width frame
/// reverses once around the loop — non-orientable.
pub(crate) fn mobius_strip(n_u: usize, n_v: usize) -> Array2<f64> {
    let n = n_u * n_v;
    let mut z = Array2::<f64>::zeros((n, 3));
    let mut r = 0usize;
    for iu in 0..n_u {
        let u = std::f64::consts::TAU * (iu as f64) / (n_u as f64);
        for iv in 0..n_v {
            let v = -0.4 + 0.8 * (iv as f64) / (n_v as f64 - 1.0);
            let radial = 2.0 + v * (u / 2.0).cos();
            z[[r, 0]] = radial * u.cos();
            z[[r, 1]] = radial * u.sin();
            z[[r, 2]] = v * (u / 2.0).sin();
            r += 1;
        }
    }
    z
}

/// A round circle of radius `radius`, tilted off every ambient axis so no chart
/// frame can be read off a coordinate plane by accident. Intrinsically `S¹`.
pub(crate) fn circle(n: usize, radius: f64) -> Array2<f64> {
    // An orthonormal pair spanning a plane oblique to all three axes.
    let e1 = [
        1.0 / 3.0_f64.sqrt(),
        1.0 / 3.0_f64.sqrt(),
        1.0 / 3.0_f64.sqrt(),
    ];
    let e2 = [1.0 / 2.0_f64.sqrt(), -1.0 / 2.0_f64.sqrt(), 0.0];
    let mut z = Array2::<f64>::zeros((n, 3));
    for r in 0..n {
        let t = std::f64::consts::TAU * (r as f64) / (n as f64);
        for c in 0..3 {
            z[[r, c]] = radius * (t.cos() * e1[c] + t.sin() * e2[c]);
        }
    }
    z
}

/// An OPEN arc: three quarters of a circle, endpoints not joined. Intrinsically
/// an interval — the `b₁ = 0` one-manifold that separates "the cover closes up"
/// from "the cover is a chain".
pub(crate) fn open_arc(n: usize, radius: f64) -> Array2<f64> {
    let mut z = Array2::<f64>::zeros((n, 3));
    for r in 0..n {
        let t = 1.5 * std::f64::consts::PI * (r as f64) / (n as f64 - 1.0);
        z[[r, 0]] = radius * t.cos();
        z[[r, 1]] = radius * t.sin();
        z[[r, 2]] = 0.25 * radius * t.sin();
    }
    z
}

/// The trefoil knot `(sin t + 2 sin 2t, cos t − 2 cos 2t, −sin 3t)`, scaled by
/// `scale`, sampled at uniform PARAMETER (not arclength).
///
/// This is the fixture that separates intrinsic topology from ambient embedding:
/// the trefoil is a smooth `S¹`, but its three ambient principal directions carry
/// comparable spread, so a global-linear seed sees a three-dimensional blob and
/// no amount of PCA recovers the loop. Everything the atlas readout uses is a
/// transition between overlapping charts, which is intrinsic, so the knotting is
/// invisible to it and the verdict must be the same as for a round circle.
pub(crate) fn trefoil_knot(n: usize, scale: f64) -> Array2<f64> {
    let mut z = Array2::<f64>::zeros((n, 3));
    for r in 0..n {
        let t = std::f64::consts::TAU * (r as f64) / (n as f64);
        z[[r, 0]] = scale * (t.sin() + 2.0 * (2.0 * t).sin());
        z[[r, 1]] = scale * (t.cos() - 2.0 * (2.0 * t).cos());
        z[[r, 2]] = scale * -(3.0 * t).sin();
    }
    z
}

/// The standard torus of revolution with major radius `major` and minor radius
/// `minor`, on a uniform `(u, v)` grid. Intrinsically `T²`: `b₁ = 2`, `χ = 0`,
/// orientable — the two-handle case the cylinder's single loop must not be
/// confused with.
pub(crate) fn torus(n_u: usize, n_v: usize, major: f64, minor: f64) -> Array2<f64> {
    let n = n_u * n_v;
    let mut z = Array2::<f64>::zeros((n, 3));
    let mut r = 0usize;
    for iu in 0..n_u {
        let u = std::f64::consts::TAU * (iu as f64) / (n_u as f64);
        for iv in 0..n_v {
            let v = std::f64::consts::TAU * (iv as f64) / (n_v as f64);
            let radial = major + minor * v.cos();
            z[[r, 0]] = radial * u.cos();
            z[[r, 1]] = radial * u.sin();
            z[[r, 2]] = minor * v.sin();
            r += 1;
        }
    }
    z
}
