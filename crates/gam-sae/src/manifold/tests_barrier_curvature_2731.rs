//! #2731 — the separation barrier's Gauss–Newton β curvature must leave the seam
//! in its FACTORED form, and must be the same operator it was when it left
//! expanded.
//!
//! ## What this module is defending
//!
//! [`SaeManifoldTerm::add_sae_separation_barrier`] derives, per co-firing
//! component, an `ne × ne` overlap-space Hessian `M` over the component's edges
//! and a per-edge carrier `v_a = ∂o_a/∂B`. The β-space curvature is
//!
//! ```text
//!   H_sep,C = Σ_{a,b} |M|[a,b] · v_a v_bᵀ .
//! ```
//!
//! Until #2731 that was handed out EXPANDED: `|M|` was diagonalized and the
//! eigen carriers `w_r = Σ_a e_r[a] v_a` were materialized, one
//! rank-1 operator each. The two forms are the same operator. The expanded one
//! costs `ne² · 2Mp` to build, `ne · s·M·p` to store and `ne · s·M·p` per
//! matvec, because every `v_a` touches exactly TWO atoms while every `w_r` is
//! dense over the whole component — and at the #2283 production shape
//! (`p = 2048`, 32 charts) that build term was **86.37% of the entire curved-tier
//! fit**, single-threaded, at 0.15% of socket memory bandwidth.
//!
//! The `ne²` is the whole story and it was previously discounted, on the ground
//! that the measured chart exponent (~2) was too small for a term that grows as
//! `K⁴` under `ne ~ K(K−1)/2`. That model of `ne` is wrong:
//! `barrier_coactive_support` emits one pair per pair of atoms **co-active on a
//! row**, so `ne` is bounded by the row count and by `top_k`-shaped
//! co-activation, and it SATURATES well below the complete graph. A saturating
//! `ne` makes `ne²` read as an exponent near 2 in the chart count, which is
//! exactly what was measured.
//!
//! ## The two tests, and what each can catch that the other cannot
//!
//! * `separation_barrier_curvature_carriers_are_two_atom_blocks_2731` is a
//!   STRUCTURAL guard: it asserts the exported carriers are exactly the two
//!   atom blocks their edge touches. It does not time anything, so it cannot
//!   flake, and it fails the instant anyone re-expands — an eigen carrier is
//!   `s·M·p` wide, not `2·M·p`.
//! * `separation_barrier_multi_edge_curvature_matches_dense_hbb_2731` is a
//!   NUMERICAL guard on a component with more than one edge, which the
//!   pre-existing two-atom parity gate
//!   (`separation_barrier_deferred_curvature_matches_dense_hbb_1610`) cannot be:
//!   at `ne = 1` the coupling is a scalar and `Σ_{a,b}` is indistinguishable
//!   from `Σ_a`. Every off-diagonal `|M|[a,b]` — the cross-edge coupling that
//!   the expansion used to fold into the eigenvectors — is only exercised here.

// `manifold/mod.rs` declares this module as
// `#[cfg(test)] mod tests_barrier_curvature_2731;` — its single declaration.
#![cfg(test)]
use super::*;

use crate::manifold::tests::{TestPeriodicEvaluator, periodic_basis};

/// `k_atoms` periodic circle atoms over `n = 6` rows under a SOFTMAX assignment,
/// so every atom carries non-negligible mass on every row and every pair
/// co-fires: one barrier component with `s = k_atoms` and `ne = s(s−1)/2` edges.
///
/// The two-atom fixture the #1610 parity gate uses cannot reach `ne > 1` by
/// construction, and `ne > 1` is where the coupling stops being a scalar.
/// Decoders are `3 × 1` with pairwise-distinct directions, so no edge is
/// degenerate and no overlap is exactly `0` or exactly `1` — a barrier that
/// abstained would make every assertion below vacuous, which is why the tests
/// assert the component's shape before reading anything off it.
fn co_firing_periodic_term(k_atoms: usize) -> SaeManifoldTerm {
    assert!(k_atoms >= 2, "a barrier component needs at least two atoms");
    let n = 6_usize;
    let mut coords_blocks: Vec<Array2<f64>> = Vec::with_capacity(k_atoms);
    let mut atoms: Vec<SaeManifoldAtom> = Vec::with_capacity(k_atoms);
    for atom_idx in 0..k_atoms {
        let mut coords = Array2::<f64>::zeros((n, 1));
        for row in 0..n {
            coords[[row, 0]] = (0.07 * (row as f64) + 0.13 * (atom_idx as f64)).rem_euclid(1.0);
        }
        let (phi, jet) = periodic_basis(&coords);
        // Distinct, non-orthogonal, non-parallel decoder directions: the overlap
        // `o_jk = cos²∠(b_j, b_k)` is then strictly inside `(0, 1)` for every
        // pair, which is the regime the barrier is defined on.
        let angle = 0.37 * (atom_idx as f64) + 0.11;
        let decoder = ndarray::array![[angle.cos()], [angle.sin()], [0.25 + 0.05 * angle]];
        atoms.push(
            SaeManifoldAtom::new_with_provided_function_gram(
                &format!("periodic{atom_idx}"),
                SaeAtomBasisKind::Periodic,
                1,
                phi,
                jet,
                decoder,
                Array2::<f64>::eye(3),
            )
            .expect("atom fixture: basis, jet, decoder and Gram shapes agree by construction")
            .with_basis_evaluator(Arc::new(TestPeriodicEvaluator)),
        );
        coords_blocks.push(coords);
    }
    let mut logits = Array2::<f64>::zeros((n, k_atoms));
    for row in 0..n {
        for atom_idx in 0..k_atoms {
            logits[[row, atom_idx]] = 0.3 * ((row + atom_idx) % 3) as f64 - 0.2;
        }
    }
    let manifolds = vec![LatentManifold::Circle { period: 1.0 }; k_atoms];
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        logits,
        coords_blocks,
        manifolds,
        AssignmentMode::softmax(0.8),
    )
    .expect("assignment fixture: logits, coordinate blocks and manifolds agree");
    SaeManifoldTerm::new(atoms, assignment)
        .expect("term fixture: every atom's row count matches the assignment's")
}

/// Run the barrier's deferred (matrix-free / framed) path and return the
/// exported curvature blocks alongside the term they came from.
fn deferred_curvature(term: &SaeManifoldTerm) -> Vec<SeparationBarrierCurvature> {
    let beta_dim = term.beta_dim();
    let mut sys = ArrowSchurSystem::new(0, 0, beta_dim);
    sys.gb = Array1::<f64>::zeros(beta_dim);
    sys.hbb = Array2::<f64>::zeros((0, 0));
    let mut atom_curv = vec![0.0_f64; term.k_atoms()];
    let mut curvature = Vec::new();
    assert!(
        term.add_sae_separation_barrier(&mut sys, 1.0, false, &mut atom_curv, &mut curvature),
        "fixture must activate the separation barrier"
    );
    curvature
}

/// #2731 — every exported carrier is exactly the TWO atom blocks its edge
/// touches. This is the shape the `ne²` expansion destroyed and the shape the
/// whole cost argument rests on.
#[test]
fn separation_barrier_curvature_carriers_are_two_atom_blocks_2731() {
    let k_atoms = 5_usize;
    let term = co_firing_periodic_term(k_atoms);
    let p = term.output_dim();
    let curvature = deferred_curvature(&term);
    assert_eq!(
        curvature.len(),
        1,
        "the softmax fixture co-fires every pair, so the barrier support is ONE \
         connected component"
    );
    let block = &curvature[0];
    let ne = block.carriers.len();
    // Non-vacuity: with a single edge the claim below ("two blocks, not `s`") is
    // trivially true, and every cross-edge coupling entry is absent.
    assert_eq!(
        ne,
        k_atoms * (k_atoms - 1) / 2,
        "every pair must co-fire, else the component is not the complete graph \
         this test reasons about"
    );
    assert_eq!(block.coupling.dim(), (ne, ne));
    let max_off_diagonal = (0..ne)
        .flat_map(|a| (0..ne).map(move |b| (a, b)))
        .filter(|&(a, b)| a != b)
        .map(|(a, b)| block.coupling[[a, b]].abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_off_diagonal > 0.0,
        "the coupling must carry live CROSS-EDGE curvature, else the factored \
         form is indistinguishable from a diagonal one"
    );

    let mut stored_values = 0_usize;
    for (edge, runs) in block.carriers.iter().enumerate() {
        assert_eq!(
            runs.len(),
            2,
            "carrier {edge} must be the two atom blocks its edge touches, not an \
             eigen carrier dense over the component"
        );
        assert!(
            runs[0].0 < runs[1].0,
            "carrier {edge} runs must be ascending in the atom index so the \
             consumers can walk them in order"
        );
        for (atom, values) in runs {
            assert_eq!(
                values.len(),
                term.atoms[*atom].basis_size() * p,
                "carrier {edge} run on atom {atom} must be that atom's whole \
                 decoder block"
            );
            stored_values += values.len();
        }
    }

    // The cost claim, as arithmetic on this fixture rather than as prose: the
    // expanded form stores `ne` carriers dense over the whole component.
    let component_width: usize = (0..k_atoms)
        .map(|atom| term.atoms[atom].basis_size() * p)
        .sum();
    let expanded_values = ne * component_width;
    assert!(
        expanded_values >= stored_values * 2,
        "with {k_atoms} atoms the expansion must be at least 2x the factored \
         storage, else this fixture cannot tell the two apart: expanded \
         {expanded_values}, factored {stored_values}"
    );
    assert_eq!(
        stored_values,
        block
            .carriers
            .iter()
            .map(|runs| runs
                .iter()
                .map(|(atom, _)| term.atoms[*atom].basis_size() * p)
                .sum::<usize>())
            .sum::<usize>()
    );
}

/// #2731 — the deferred (factored) curvature reconstructs the dense `hbb` on a
/// component with MANY edges, where the coupling is a matrix rather than a
/// scalar.
///
/// `separation_barrier_deferred_curvature_matches_dense_hbb_1610` pins the same
/// identity on the two-atom fixture, which has `ne = 1`: there the coupling is
/// `1 × 1` and a factored form that dropped every off-diagonal `|M|[a,b]` would
/// still pass it. Here it would not.
#[test]
fn separation_barrier_multi_edge_curvature_matches_dense_hbb_2731() {
    let term = co_firing_periodic_term(4);
    let beta_dim = term.beta_dim();
    let offsets = term.beta_offsets();

    let mut dense = ArrowSchurSystem::new(0, 0, beta_dim);
    dense.gb = Array1::<f64>::zeros(beta_dim);
    dense.hbb = Array2::<f64>::zeros((beta_dim, beta_dim));
    let mut dense_atom_curv = vec![0.0_f64; term.k_atoms()];
    let mut dense_curvature = Vec::new();
    assert!(
        term.add_sae_separation_barrier(
            &mut dense,
            1.0,
            true,
            &mut dense_atom_curv,
            &mut dense_curvature
        ),
        "fixture must activate the separation barrier on the dense path"
    );
    assert!(
        dense_curvature.is_empty(),
        "dense path expands into hbb rather than exporting the factored block"
    );

    let mut deferred = ArrowSchurSystem::new(0, 0, beta_dim);
    deferred.gb = Array1::<f64>::zeros(beta_dim);
    deferred.hbb = Array2::<f64>::zeros((0, 0));
    let mut atom_curv = vec![0.0_f64; term.k_atoms()];
    let mut curvature = Vec::new();
    assert!(
        term.add_sae_separation_barrier(&mut deferred, 1.0, false, &mut atom_curv, &mut curvature),
        "fixture must activate the separation barrier on the deferred path"
    );

    for idx in 0..beta_dim {
        assert!(
            (dense.gb[idx] - deferred.gb[idx]).abs() <= 1.0e-12,
            "the two paths must assemble the same barrier gradient at β[{idx}]"
        );
    }

    let mut reconstructed = Array2::<f64>::zeros((beta_dim, beta_dim));
    for atom_idx in 0..term.k_atoms() {
        let start = offsets[atom_idx];
        let end = start + term.atoms[atom_idx].basis_size() * term.output_dim();
        assert!(
            atom_curv[atom_idx] > 0.0,
            "the deferred path must export the per-atom Levenberg ridge for atom \
             {atom_idx}"
        );
        for idx in start..end {
            reconstructed[[idx, idx]] += atom_curv[atom_idx];
        }
    }
    // Read the factored block through the SAME operator production installs.
    for block in &curvature {
        reconstructed += &block.as_full_beta_op(beta_dim, &offsets).to_dense();
    }

    // Non-vacuity: the coupling has to have live off-diagonal mass, or this test
    // is the `ne = 1` gate again in a longer form.
    let coupling = &curvature.first().expect("one component").coupling;
    let ne = coupling.nrows();
    assert_eq!(ne, 6, "four fully co-firing atoms give six edges");
    let off_diagonal_mass: f64 = (0..ne)
        .flat_map(|a| (0..ne).map(move |b| (a, b)))
        .filter(|&(a, b)| a != b)
        .map(|(a, b)| coupling[[a, b]].abs())
        .sum();
    let diagonal_mass: f64 = (0..ne).map(|a| coupling[[a, a]].abs()).sum();
    assert!(
        off_diagonal_mass > 0.1 * diagonal_mass,
        "cross-edge coupling must be material, else the multi-edge claim is not \
         being tested: off-diagonal {off_diagonal_mass:e}, diagonal \
         {diagonal_mass:e}"
    );

    // The two routes reassociate the same f64 sum, so the bar is relative.
    let scale = dense.hbb.iter().fold(0.0_f64, |acc, v| acc.max(v.abs()));
    assert!(scale > 0.0, "the dense curvature must be non-trivial");
    let tolerance = 1.0e-12 * (1.0 + scale);
    for i in 0..beta_dim {
        for j in 0..beta_dim {
            assert!(
                (dense.hbb[[i, j]] - reconstructed[[i, j]]).abs() <= tolerance,
                "dense hbb and the reconstructed (ridge + factored) curvature must \
                 match at ({i},{j}): dense={} reconstructed={}",
                dense.hbb[[i, j]],
                reconstructed[[i, j]]
            );
        }
    }
}
