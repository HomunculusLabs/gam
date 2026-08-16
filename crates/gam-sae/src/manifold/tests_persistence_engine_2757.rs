#![cfg(test)]
//! #2757 — the rewritten Vietoris–Rips filtration engine produces the IDENTICAL
//! diagram to the one it replaces.
//!
//! ## What changed and why it needed a control
//!
//! `atom_topology_persistence` was measured at ~17 s per full-support atom, `K`
//! of them, serially — 547 s of a 547.8 s `fit_diagnostics_report` at
//! `n = 256, p = 64, charts = 32`. The mechanism was not the mathematics: the
//! `H₁` cover enumerates `C(256, 3) = 2 763 520` triangles, and the engine gave
//! each one a heap-allocated vertex list, a second one as a `HashMap` key, a
//! third as its boundary and a fourth as its reduced column, then reduced every
//! single one of them.
//!
//! Three changes, each of which must be invisible in the output:
//!
//! 1. **Inline simplices and a combinatorial-number-system ranking** replace the
//!    `Vec<usize>` vertex lists and the `HashMap<Vec<usize>, usize>` face index.
//!    A face's slot is now an array index computed in `d` additions.
//! 2. **The reduction is run one dimension at a time.** The boundary of a
//!    `d`-simplex contains only `(d−1)`-simplices, so pivots never cross
//!    dimensions and this is the same computation as the interleaved sweep.
//! 3. **A pair budget stops the top dimension early.** At the final filtration
//!    value the complex is the full `max_simplex_dim` skeleton of the simplex on
//!    `m` vertices, whose homology vanishes below the top dimension — so the
//!    number of pairs in dimension `d` is exactly the number of dimension-`(d−1)`
//!    columns that reduced to zero (less the one connected component at `d = 1`).
//!    Once that many are found, every remaining column of that dimension reduces
//!    to zero and contributes nothing. This is a theorem about the complex, not a
//!    truncation of the filtration: no simplex is dropped and no bar is lost.
//!
//! (3) is the one that could silently lose bars, which is why the gates below
//! difference the two engines BAR BY BAR rather than comparing Betti numbers or
//! the verdicts a downstream reader forms. A fixture family that all agreed on
//! `b₁` while disagreeing on a bar would pass a Betti test and fail this one.
//!
//! The fixtures span the shapes the audit actually meets: a circle (one loop),
//! a Clifford torus (two loops and a shell — the `H₂` path, so the tetrahedron
//! enumeration and the `max_homology_dim = 2` branch are covered), a line (no
//! loop), separated clusters (several `H₀` bars, and the case where the
//! enclosing structure is eccentric rather than round), coincident points
//! (exactly-tied filtration values, which is where an ordering change would
//! show), and a shape at the four-point floor.

use super::persistence::*;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use std::collections::HashMap;

/// The pre-#2757 filtration engine, kept verbatim as the reference the rewritten
/// one is judged against.
///
/// This is the whole point of the module: the rewrite is a COST change, so the
/// only acceptable evidence is that it produces the identical diagram — not a
/// diagram that agrees to a tolerance, and not one that agrees on the Betti
/// numbers a downstream verdict happens to read. Every gate below differences
/// the two bar lists element by element.
///
/// It is deliberately the naive algorithm: one `Vec<usize>` per simplex, one
/// `HashMap<Vec<usize>, usize>` keyed by the vertex list, one interleaved sweep
/// over the global order with no dimension decomposition and no pair budget.
/// Sharing any of the rewrite's reasoning would make it a mirror rather than a
/// control.
struct ReferenceSimplex {
    verts: Vec<usize>,
    filt: f64,
    dim: usize,
}

fn reference_symmetric_difference(a: &[usize], b: &[usize]) -> Vec<usize> {
    let mut out = Vec::with_capacity(a.len() + b.len());
    let mut ia = 0;
    let mut ib = 0;
    while ia < a.len() && ib < b.len() {
        match a[ia].cmp(&b[ib]) {
            std::cmp::Ordering::Less => {
                out.push(a[ia]);
                ia += 1;
            }
            std::cmp::Ordering::Greater => {
                out.push(b[ib]);
                ib += 1;
            }
            std::cmp::Ordering::Equal => {
                ia += 1;
                ib += 1;
            }
        }
    }
    out.extend_from_slice(&a[ia..]);
    out.extend_from_slice(&b[ib..]);
    out
}

pub(super) fn reference_dtm_vietoris_rips_persistence(
    points: ArrayView2<'_, f64>,
    weights: Option<ArrayView1<'_, f64>>,
    max_homology_dim: usize,
) -> PersistenceDiagram {
    let m = points.nrows();
    let mut h0 = Vec::new();
    let mut h1 = Vec::new();
    let mut h2 = Vec::new();
    if m == 0 {
        return PersistenceDiagram { h0, h1, h2 };
    }
    if m == 1 {
        // A single point has DTM radius 0 (`dtm_radii` returns 0 for `m <= 1`), so
        // its DTM-weighted vertex birth is 0 — historical behavior preserved.
        h0.push(PersistenceBar {
            birth: 0.0,
            death: f64::INFINITY,
        });
        return PersistenceDiagram { h0, h1, h2 };
    }

    let (dist, dtm) = dtm_weighted_distances_and_radii(points, weights);

    // Build simplices up to the coface dimension needed by the requested
    // homology: H₁ needs triangles, H₂ needs tetrahedra.
    let max_simplex_dim = (max_homology_dim + 1).min(3);
    let mut simplices: Vec<ReferenceSimplex> = Vec::new();
    // Standard `p = ∞` DTM-weighted Vietoris–Rips convention: a vertex is born at
    // its own DTM radius `w_i = dtm[i]`, NOT at 0. Edges/higher simplices already
    // carry `max(d_ij, w_i, w_j)` (see `dtm_weighted_distances_and_radii`), which
    // is `≥` each face's DTM birth, so face-before-coface ordering is preserved.
    for i in 0..m {
        simplices.push(ReferenceSimplex {
            verts: vec![i],
            filt: dtm[i],
            dim: 0,
        });
    }
    for i in 0..m {
        for j in (i + 1)..m {
            simplices.push(ReferenceSimplex {
                verts: vec![i, j],
                filt: dist[[i, j]],
                dim: 1,
            });
        }
    }
    if max_simplex_dim >= 2 {
        for i in 0..m {
            for j in (i + 1)..m {
                for k in (j + 1)..m {
                    let filt = dist[[i, j]].max(dist[[i, k]]).max(dist[[j, k]]);
                    simplices.push(ReferenceSimplex {
                        verts: vec![i, j, k],
                        filt,
                        dim: 2,
                    });
                }
            }
        }
    }
    if max_simplex_dim >= 3 {
        for i in 0..m {
            for j in (i + 1)..m {
                for k in (j + 1)..m {
                    for l in (k + 1)..m {
                        let filt = dist[[i, j]]
                            .max(dist[[i, k]])
                            .max(dist[[i, l]])
                            .max(dist[[j, k]])
                            .max(dist[[j, l]])
                            .max(dist[[k, l]]);
                        simplices.push(ReferenceSimplex {
                            verts: vec![i, j, k, l],
                            filt,
                            dim: 3,
                        });
                    }
                }
            }
        }
    }

    // Filtration order: ascending filtration, then ascending dimension (a face
    // must precede its coface), then lexicographic vertices for a total order.
    let mut order: Vec<usize> = (0..simplices.len()).collect();
    order.sort_by(|&a, &b| {
        let sa = &simplices[a];
        let sb = &simplices[b];
        sa.filt
            .partial_cmp(&sb.filt)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(sa.dim.cmp(&sb.dim))
            .then(sa.verts.cmp(&sb.verts))
    });
    // Global filtration index of each simplex, and a vertex-set -> index map.
    let mut filt_index = vec![0usize; simplices.len()];
    let mut key_to_index: HashMap<Vec<usize>, usize> = HashMap::with_capacity(simplices.len());
    for (fi, &orig) in order.iter().enumerate() {
        filt_index[orig] = fi;
        key_to_index.insert(simplices[orig].verts.clone(), fi);
    }

    // Ordered simplices (indexed by filtration position) with their boundaries
    // (as filtration indices of their codim-1 faces).
    let mut ordered_filt = vec![0.0_f64; simplices.len()];
    let mut ordered_dim = vec![0usize; simplices.len()];
    let mut boundary: Vec<Vec<usize>> = vec![Vec::new(); simplices.len()];
    for &orig in &order {
        let s = &simplices[orig];
        let fi = filt_index[orig];
        ordered_filt[fi] = s.filt;
        ordered_dim[fi] = s.dim;
        if s.dim == 0 {
            continue;
        }
        let mut faces = Vec::with_capacity(s.verts.len());
        for drop in 0..s.verts.len() {
            let mut face = Vec::with_capacity(s.verts.len() - 1);
            for (idx, &v) in s.verts.iter().enumerate() {
                if idx != drop {
                    face.push(v);
                }
            }
            if let Some(&face_fi) = key_to_index.get(&face) {
                faces.push(face_fi);
            }
        }
        faces.sort_unstable();
        boundary[fi] = faces;
    }

    // GF(2) reduction. `reduced[j]` holds the reduced column (sorted). `pivot`
    // maps a low-index to the column that owns it. `paired_birth` marks faces
    // that have been consumed as a birth (so leftover empty columns are the
    // essential classes).
    let n = simplices.len();
    let mut reduced: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut pivot: HashMap<usize, usize> = HashMap::new();
    let mut paired_birth = vec![false; n];

    for j in 0..n {
        let mut col = boundary[j].clone();
        while let Some(&low) = col.last() {
            if let Some(&owner) = pivot.get(&low) {
                col = reference_symmetric_difference(&col, &reduced[owner]);
            } else {
                break;
            }
        }
        if let Some(&low) = col.last() {
            pivot.insert(low, j);
            reduced[j] = col;
            paired_birth[low] = true;
            // Persistence pair: face `low` born, simplex `j` kills it.
            let birth = ordered_filt[low];
            let death = ordered_filt[j];
            let bar = PersistenceBar { birth, death };
            // `PersistenceDiagram` carries H0/H1/H2 only; classes of higher
            // dimension are not recorded.
            let dim = ordered_dim[low];
            if death > birth {
                if dim == 0 {
                    h0.push(bar);
                } else if dim == 1 {
                    h1.push(bar);
                } else if dim == 2 && max_homology_dim >= 2 {
                    h2.push(bar);
                }
            }
        }
    }

    // Essential classes: fully reduced zero columns that were never consumed as
    // a birth.
    for j in 0..n {
        if reduced[j].is_empty() && !paired_birth[j] {
            let bar = PersistenceBar {
                birth: ordered_filt[j],
                death: f64::INFINITY,
            };
            // As above: only H0/H1/H2 are recorded.
            let dim = ordered_dim[j];
            if dim == 0 {
                h0.push(bar);
            } else if dim == 1 {
                h1.push(bar);
            } else if dim == 2 && max_homology_dim >= 2 {
                h2.push(bar);
            }
        }
    }

    PersistenceDiagram { h0, h1, h2 }
}

fn lcg(s: &mut u64) -> f64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*s >> 11) as f64) / ((1u64 << 53) as f64)
}

fn circle(n: usize, r: f64) -> Array2<f64> {
    Array2::from_shape_fn((n, 2), |(i, c)| {
        let t = std::f64::consts::TAU * (i as f64) / (n as f64);
        if c == 0 { r * t.cos() } else { r * t.sin() }
    })
}

fn clifford_torus(nu: usize, nv: usize) -> Array2<f64> {
    let mut pts = Array2::<f64>::zeros((nu * nv, 4));
    for a in 0..nu {
        for b in 0..nv {
            let u = std::f64::consts::TAU * (a as f64) / (nu as f64);
            let v = std::f64::consts::TAU * (b as f64) / (nv as f64);
            let row = a * nv + b;
            pts[[row, 0]] = u.cos();
            pts[[row, 1]] = u.sin();
            pts[[row, 2]] = v.cos();
            pts[[row, 3]] = v.sin();
        }
    }
    pts
}

fn line(n: usize) -> Array2<f64> {
    Array2::from_shape_fn((n, 2), |(i, c)| if c == 0 { i as f64 } else { 0.0 })
}

/// `c` well-separated clusters, so the enclosing structure is eccentric and the
/// H₀ barcode carries several long bars.
fn clusters(c: usize, per: usize) -> Array2<f64> {
    let mut s = 0x2757_C105_0000_0001u64;
    let mut pts = Array2::<f64>::zeros((c * per, 2));
    for k in 0..c {
        for j in 0..per {
            let row = k * per + j;
            pts[[row, 0]] = 100.0 * k as f64 + lcg(&mut s);
            pts[[row, 1]] = lcg(&mut s);
        }
    }
    pts
}

/// A cloud with exact coordinate repeats, so a run of simplices shares one
/// filtration value to the last bit. Ties are where a change in the total order
/// — which the dimension decomposition could in principle have introduced —
/// would surface.
fn tied(n: usize) -> Array2<f64> {
    Array2::from_shape_fn((n, 2), |(i, c)| {
        let cell = (i / 3) as f64;
        if c == 0 { cell } else { (i % 3) as f64 }
    })
}

fn uniform_cloud(n: usize, dim: usize, seed: u64) -> Array2<f64> {
    let mut s = seed;
    Array2::from_shape_fn((n, dim), |_| lcg(&mut s) - 0.5)
}

fn assert_same_bars(label: &str, degree: &str, got: &[PersistenceBar], want: &[PersistenceBar]) {
    assert_eq!(
        got.len(),
        want.len(),
        "{label}: {degree} has {} bars against the reference engine's {}",
        got.len(),
        want.len()
    );
    for (index, (a, b)) in got.iter().zip(want.iter()).enumerate() {
        assert!(
            a.birth.to_bits() == b.birth.to_bits() && a.death.to_bits() == b.death.to_bits(),
            "{label}: {degree} bar {index} is [{}, {}) against the reference engine's [{}, {})",
            a.birth,
            a.death,
            b.birth,
            b.death
        );
    }
}

/// The gate: same points, same weights, same requested degree ⇒ same diagram,
/// bar for bar, bit for bit.
fn assert_engines_agree(
    label: &str,
    points: ArrayView2<'_, f64>,
    weights: Option<ArrayView1<'_, f64>>,
    max_homology_dim: usize,
) {
    let got = dtm_vietoris_rips_persistence(points, weights, max_homology_dim);
    let want = reference_dtm_vietoris_rips_persistence(points, weights, max_homology_dim);
    assert_same_bars(label, "H0", &got.h0, &want.h0);
    assert_same_bars(label, "H1", &got.h1, &want.h1);
    assert_same_bars(label, "H2", &got.h2, &want.h2);
}

#[test]
fn the_rewritten_engine_reproduces_the_reference_diagram_on_every_audited_shape() {
    let circle_pts = circle(40, 1.0);
    assert_engines_agree("circle(40)", circle_pts.view(), None, 1);

    let line_pts = line(24);
    assert_engines_agree("line(24)", line_pts.view(), None, 1);

    let cluster_pts = clusters(4, 8);
    assert_engines_agree("clusters(4x8)", cluster_pts.view(), None, 1);

    let tied_pts = tied(21);
    assert_engines_agree("tied(21)", tied_pts.view(), None, 1);

    let cloud = uniform_cloud(36, 3, 0x2757_0000_0000_00A1);
    assert_engines_agree("uniform(36,3)", cloud.view(), None, 1);

    // The four-point floor: a triangle plus one point is the smallest cloud that
    // can kill a loop, and it is where the pair budget is smallest.
    let floor = uniform_cloud(4, 2, 0x2757_0000_0000_00B1);
    assert_engines_agree("uniform(4,2)", floor.view(), None, 1);
    let five = uniform_cloud(5, 2, 0x2757_0000_0000_00B2);
    assert_engines_agree("uniform(5,2)", five.view(), None, 1);
}

#[test]
fn the_rewritten_engine_reproduces_the_reference_diagram_with_dtm_weights() {
    // The DTM weighting moves every vertex's birth off zero and every edge's
    // filtration off the raw distance, so this exercises a different filtration
    // function over the same combinatorics.
    let mut s = 0x2757_DDDD_0000_0001u64;
    let pts = circle(32, 1.0);
    let w = Array1::from_shape_fn(pts.nrows(), |_| 0.25 + lcg(&mut s));
    assert_engines_agree("circle(32) dtm-weighted", pts.view(), Some(w.view()), 1);

    let cloud = uniform_cloud(30, 4, 0x2757_DDDD_0000_0002);
    let w2 = Array1::from_shape_fn(cloud.nrows(), |_| 0.1 + lcg(&mut s));
    assert_engines_agree("uniform(30,4) dtm-weighted", cloud.view(), Some(w2.view()), 1);

    // A weight vector with an exact zero and an exact repeat: the DTM radii
    // collapse and several simplices tie.
    let flat = Array1::from_shape_fn(cloud.nrows(), |i| if i % 5 == 0 { 0.0 } else { 1.0 });
    assert_engines_agree("uniform(30,4) flat weights", cloud.view(), Some(flat.view()), 1);
}

#[test]
fn the_rewritten_engine_reproduces_the_reference_diagram_through_h2() {
    // `max_homology_dim = 2` is the tetrahedron branch: it builds C(m,4)
    // simplices, runs the reduction in four dimensions rather than three, and is
    // the only path on which dimension 2's zero-column SET is needed (both as
    // dimension 3's pair budget and as the H₂ essential classes), so the early
    // stop is taken in a different dimension than on the H₁ path.
    let torus = clifford_torus(5, 5);
    assert_engines_agree("clifford_torus(5x5)", torus.view(), None, 2);

    let sphere_like = uniform_cloud(18, 3, 0x2757_2222_0000_0001);
    assert_engines_agree("uniform(18,3) H2", sphere_like.view(), None, 2);

    let mut s = 0x2757_2222_0000_0002u64;
    let w = Array1::from_shape_fn(torus.nrows(), |_| 0.5 + lcg(&mut s));
    assert_engines_agree("clifford_torus(5x5) weighted H2", torus.view(), Some(w.view()), 2);
}

#[test]
fn the_rewritten_engine_reproduces_the_reference_diagram_on_degenerate_clouds() {
    // Coincident points: several distances are exactly zero, so a whole block of
    // simplices is born at the same instant and the tie-break in the ordering is
    // the only thing separating them.
    let mut pts = Array2::<f64>::zeros((12, 2));
    for i in 0..12 {
        pts[[i, 0]] = (i / 4) as f64;
        pts[[i, 1]] = 0.0;
    }
    assert_engines_agree("coincident(12)", pts.view(), None, 1);
    assert_engines_agree("coincident(12) H2", pts.view(), None, 2);

    // Every point identical: the entire filtration is one value.
    let same = Array2::<f64>::zeros((7, 3));
    assert_engines_agree("identical(7)", same.view(), None, 1);

    // Collinear with one far outlier: the enclosing structure is maximally
    // eccentric, which is the regime where the budget is reached earliest.
    let mut outlier = Array2::<f64>::zeros((14, 2));
    for i in 0..13 {
        outlier[[i, 0]] = i as f64 * 0.1;
    }
    outlier[[13, 0]] = 500.0;
    assert_engines_agree("outlier(14)", outlier.view(), None, 1);
}

#[test]
fn the_pair_budget_never_drops_a_bar_the_reference_engine_finds() {
    // The budget is the one change that could lose bars silently, so it gets a
    // gate that is about IT rather than about a fixture: over a family of random
    // clouds, the two engines must agree on the bar COUNT in every degree. A
    // budget that stopped one pair early would show here as a missing H₁ bar on
    // some member, which no single fixture is guaranteed to catch.
    for seed in 0..12u64 {
        let n = 8 + (seed as usize % 7) * 3;
        let cloud = uniform_cloud(n, 2 + (seed as usize % 3), 0x2757_B0D0_0000_0001 + seed * 7919);
        let got = dtm_vietoris_rips_persistence(cloud.view(), None, 1);
        let want = reference_dtm_vietoris_rips_persistence(cloud.view(), None, 1);
        assert_eq!(
            (got.h0.len(), got.h1.len()),
            (want.h0.len(), want.h1.len()),
            "seed {seed} (n={n}): bar counts diverged"
        );
        assert_same_bars(&format!("seed {seed}"), "H0", &got.h0, &want.h0);
        assert_same_bars(&format!("seed {seed}"), "H1", &got.h1, &want.h1);
    }
}
