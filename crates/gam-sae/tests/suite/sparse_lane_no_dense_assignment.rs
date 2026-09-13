//! E1 (#985) demotion guard: the DENSE `SaeAssignment` representation must never
//! be constructed on the production SPARSE-code lane.
//!
//! `SaeAssignment` (the dense `N×K` routing state) is the dense-certification /
//! debug-and-research lane only. The production SAE path is the sparse-code lane
//! (`crate::sparse_dict`), whose per-row state is fixed-width `(indices, codes)` —
//! never an `N×K` assignment. This test LOCKS that architectural invariant by
//! scanning the sparse lane's own source: outside `#[cfg(test)]` regions, every
//! `sparse_dict/*.rs` file must contain ZERO `SaeAssignment` constructions.
//!
//! It scans ONLY the sparse lane, so it cannot break the overcomplete dense
//! research / certification path (which legitimately builds `SaeAssignment` at
//! K > P for small N) — that path lives in the manifold engine, not here.
//!
//! #2693 strengthened the scan twice, both tightenings:
//!
//! * it RECURSES through `sparse_dict/**` instead of reading only the top level,
//!   so a nested submodule cannot hold the dense state the flat `read_dir` missed;
//! * it also refuses the MECHANISM, not only the symptom. `SaeManifoldTerm` is the
//!   dense engine aggregate and its ONLY constructor is
//!   `SaeManifoldTerm::new(atoms, assignment: SaeAssignment)`, so naming that type
//!   in the sparse lane FORCES a dense `N×K` assignment into existence — there is
//!   no sparse in-core entry (that is the Stage 2
//!   `SaeAssignmentState::from_topk_support_heterogeneous` seam on #2023). Without this needle the invariant could be re-broken by any
//!   new dense-engine caller, and the `SaeAssignment` needle would only catch it
//!   at the last line. `crate::manifold::realised_rank_charge_dof` and friends are
//!   deliberately NOT needles: they are pure helpers that build no routing state.

use gam_sae::front_door::{SaeFitLane, admit_dense_certification, admit_sae_fit};
use gam_sae::sparse_dict::{SparseDictConfig, fit_sparse_dictionary};
use ndarray::Array2;
use std::path::PathBuf;

/// Drop every `#[cfg(test)]`-guarded item from Rust source, returning only the
/// production (non-test) code. For each `#[cfg(test)]` attribute we skip from it
/// through the matching close brace of the item it guards (`mod`/`fn`/`impl`),
/// brace-balanced from the first `{` after the attribute. All region boundaries
/// (`#`, `{`, `}`) are ASCII, so the retained `&str` slices stay valid UTF-8 even
/// though the source contains non-ASCII (e.g. `×`, `≤`) in comments.
fn strip_cfg_test_regions(src: &str) -> String {
    const ATTR: &str = "#[cfg(test)]";
    let mut kept = String::with_capacity(src.len());
    let mut cursor = 0usize;
    while let Some(rel) = src[cursor..].find(ATTR) {
        let attr = cursor + rel;
        kept.push_str(&src[cursor..attr]);
        match src[attr..].find('{') {
            Some(brace_rel) => {
                let brace = attr + brace_rel;
                let bytes = src.as_bytes();
                let mut depth = 0i32;
                let mut j = brace;
                while j < bytes.len() {
                    if bytes[j] == b'{' {
                        depth += 1;
                    } else if bytes[j] == b'}' {
                        depth -= 1;
                        if depth == 0 {
                            j += 1;
                            break;
                        }
                    }
                    j += 1;
                }
                cursor = j;
            }
            // An attribute with no following block is malformed; drop the tail.
            None => cursor = src.len(),
        }
    }
    kept.push_str(&src[cursor..]);
    kept
}

/// Guard-the-guard: the stripper removes `#[cfg(test)]` blocks and ONLY those, so
/// the scan below cannot be defeated by moving a construction into a test module,
/// nor does it vacuously pass by deleting production code.
#[test]
fn strip_cfg_test_regions_removes_guarded_blocks_only() {
    let src = "fn prod() { build(SaeAssignment::new()); }\n\
               #[cfg(test)]\n\
               mod tests { fn t() { build(SaeAssignment::from_blocks()); SaeManifoldTerm::new(); } }\n\
               fn also_prod() {}\n";
    let stripped = strip_cfg_test_regions(src);
    assert!(
        stripped.contains("SaeAssignment::new"),
        "production code before the cfg(test) block must survive"
    );
    assert!(
        stripped.contains("also_prod"),
        "production code after the cfg(test) block must survive"
    );
    assert!(
        !stripped.contains("SaeAssignment::from_blocks"),
        "the cfg(test) block must be stripped"
    );
    assert!(
        !stripped.contains("SaeManifoldTerm"),
        "the dense-engine needle must also be stripped inside a cfg(test) block"
    );
}

/// The production sparse-code lane constructs ZERO dense `SaeAssignment`s.
#[test]
fn sparse_lane_constructs_no_dense_assignment() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("src")
        .join("sparse_dict");
    assert!(
        dir.is_dir(),
        "sparse-code lane directory must exist at {dir:?}"
    );

    // A dense-assignment CONSTRUCTION is a struct literal (`SaeAssignment {`) or
    // any associated-fn call (`SaeAssignment::…`, covering every constructor —
    // `new`, `from_blocks_with_mode_and_manifolds`, …).
    // `SaeManifoldTerm` is the dense engine aggregate: its only constructor takes a
    // `SaeAssignment`, so naming it here forces the dense `N×K` state (#2693).
    const CONSTRUCTION_NEEDLES: [&str; 3] =
        ["SaeAssignment {", "SaeAssignment::", "SaeManifoldTerm"];

    // Recurse: a nested `sparse_dict/<sub>/*.rs` module is still the sparse lane.
    let mut sources: Vec<PathBuf> = Vec::new();
    let mut pending: Vec<PathBuf> = vec![dir.clone()];
    while let Some(d) = pending.pop() {
        for entry in std::fs::read_dir(&d).expect("read sparse_dict directory") {
            let path = entry.expect("sparse_dict dir entry").path();
            if path.is_dir() {
                pending.push(path);
            } else if path.extension().and_then(|e| e.to_str()) == Some("rs") {
                sources.push(path);
            }
        }
    }

    let mut scanned = 0usize;
    let mut offenders: Vec<String> = Vec::new();
    for path in &sources {
        let src = std::fs::read_to_string(path).expect("read sparse-lane source file");
        let production = strip_cfg_test_regions(&src);
        let name = path
            .strip_prefix(&dir)
            .unwrap_or(path)
            .to_string_lossy()
            .into_owned();
        for needle in CONSTRUCTION_NEEDLES {
            if production.contains(needle) {
                offenders.push(format!("{name}: production code contains `{needle}`"));
            }
        }
        scanned += 1;
    }

    assert!(
        scanned > 0,
        "expected to scan at least one sparse_dict source file"
    );
    assert!(
        offenders.is_empty(),
        "the production sparse-code lane must construct ZERO dense SaeAssignments and must not \
         enter the dense engine (`SaeManifoldTerm`, whose only constructor takes one) — that \
         dense N×K routing state is the certification / debug-and-research lane only (#985 / E1); \
         scanned {scanned} file(s), found {} offender(s): {offenders:?}",
        offenders.len()
    );
}

/// Runtime lock (complements the static source scan): at an overcomplete shape
/// (`K > P`) the front door demotes to the sparse-code lane, the dense-engine
/// guard REFUSES, and a real `fit_sparse_dictionary` returns fixed-width sparse
/// state `N×s` — never the dense `N×K` assignment the front door exists to avoid.
#[test]
fn large_k_sparse_fit_stays_fixed_width_and_never_materializes_dense_n_by_k() {
    // An overcomplete shape: K = 12 atoms into a P = 8 response, so the dense
    // assignment N·K is 1.5× the response N·P and the front door demotes it.
    //
    // #2822: this used to be K = 256 into P = 8 with s = 4 over 96 rows. That shape
    // cannot become a model any more: every epoch the trainer revives the atoms no
    // row routes to, and a fit is returned only from an epoch with zero accepted
    // births and a settled routing (the absolute fixed point, #2902). Pool job 578260
    // at 078a8f339 read `InnerNonConvergence { epochs: 30, accepted_births: 12,
    // decoder_fixed_point_residual: 1.0, routing_residual: 1.048 }` on it, which is
    // production's intended answer to the K ≫ rank limit cycle. The storage contract
    // this test locks does not need that regime, only K > P and a returned fit.
    let n_obs = 96usize;
    let p_out = 8usize;
    let k_atoms = 12usize;
    let active = 1usize;

    // Front-door admission: this shape routes to sparse codes, and the dense-engine
    // guard refuses it, pointing the caller at the sparse-code lane.
    let admission = admit_sae_fit(n_obs, p_out, k_atoms).expect("admission");
    assert_eq!(admission.lane, SaeFitLane::SparseCodes);
    let refusal = admit_dense_certification(n_obs, p_out, k_atoms)
        .expect_err("dense engine must refuse the K > P shape");
    assert!(
        refusal.contains("sparse-code lane"),
        "refusal must point at the sparse-code lane; got: {refusal}"
    );

    // Deterministic planted data (no RNG dependency): twelve distinct signed lines in
    // P = 8, the eight axes and the four normalized axis-pair sums, eight rows on each.
    // The largest cosine between two lines is 1/√2, so every row routes to its own
    // line and every atom has rows: no atom is dead, nothing is revived, and the
    // routing can settle.
    // #2822: a dictionary that reproduces every row exactly leaves zero residual,
    // and the shared-ρ Fellner–Schall step refuses the resulting `ρ = 0` as
    // boundary evidence. A deterministic perturbation keeps the residual above the
    // arithmetic floor while the planted line still dominates every row.
    let mut atoms = Array2::<f32>::zeros((k_atoms, p_out));
    for axis in 0..p_out {
        atoms[[axis, axis]] = 1.0;
    }
    for pair in 0..(k_atoms - p_out) {
        atoms[[p_out + pair, 2 * pair]] = std::f32::consts::FRAC_1_SQRT_2;
        atoms[[p_out + pair, 2 * pair + 1]] = std::f32::consts::FRAC_1_SQRT_2;
    }
    let mut x = Array2::<f32>::zeros((n_obs, p_out));
    for row in 0..n_obs {
        let atom = row % k_atoms;
        let scale = 1.0 + 0.01 * (row / k_atoms) as f32;
        for col in 0..p_out {
            let phase = (row as f32 + 1.0) * 12.9898 + (col as f32 + 1.0) * 78.233;
            x[[row, col]] =
                scale * atoms[[atom, col]] + 0.01 * (phase.sin() * 43758.5453).sin();
        }
    }

    let config = SparseDictConfig {
        active,
        ..SparseDictConfig::new(k_atoms)
    };
    let fit = fit_sparse_dictionary(x.view(), &config).expect("sparse dictionary fit");
    assert!(
        fit.convergence.certified && fit.convergence.accepted_births == 0,
        "a returned fit is a certified fixed point with no births; got certified={} births={}",
        fit.convergence.certified,
        fit.convergence.accepted_births
    );

    // The whole point: the fitted routing state is fixed-width sparse `N×s`, with
    // `s = min(active, K) ≪ K`. It is NEVER an `N×K` dense assignment.
    assert_eq!(fit.active, active.min(k_atoms));
    assert_eq!(fit.indices.dim(), (n_obs, fit.active));
    assert_eq!(fit.codes.dim(), (n_obs, fit.active));
    assert!(
        fit.indices.ncols() < k_atoms,
        "sparse routing width {} must be ≪ K = {k_atoms} — an N×K state is the demoted dense lane",
        fit.indices.ncols()
    );
    // The decoder is `K×P` (the dictionary itself), the only K-scaled state — and it
    // is P-wide, not N-wide, so no `N×K` object exists anywhere in the fit.
    assert_eq!(fit.decoder.dim(), (k_atoms, p_out));

    // The dense-Cholesky decline → certified block CG contract is pinned by
    // `sparse_dict::update::exact_solve_tests::a_declined_dense_cholesky_routes_through_certified_block_cg_2822`
    // on a constructed singular component, not on this fit.
}
