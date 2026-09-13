//! Regression: `SheafConsistencyPenalty::hessian_diag` agrees with the operator
//! its siblings `gradient` and `hvp` implement, including on a self-loop edge
//! `(v, v)`.
//!
//! ## The contract
//!
//! The penalty is the exact quadratic
//!
//!     P(s) = ½ · weight · sᵀ L s,    L = δᵀδ   (the sheaf Laplacian).
//!
//! Because it is quadratic, its Hessian is *exactly* `∂²P/∂s² = weight · L`,
//! constant in `s`. The crate exposes that operator three self-consistent ways:
//!
//!   * `gradient(s)       = weight · L · s`
//!   * `hvp(s, v)         = weight · L · v`
//!   * `hessian_diag(s)   = diag(weight · L)`   ← documented contract
//!
//! So `hessian_diag(s)[j]` MUST equal the `j`-th diagonal entry of the same
//! operator, i.e. `hvp(s, e_j)[j]` for the `j`-th standard basis vector `e_j`.
//! `hvp`/`gradient` route through the matrix-free coboundary matvec
//! (`delta` / `delta_transpose`), which is the ground-truth operator.
//!
//! ## The defect this test was written for
//!
//! For an edge `(u, v)` the coboundary is `δs[e] = R_uv·s_u − R_vu·s_v`, so the
//! edge's contribution to `L` is `Cᵀ C` with `C = [R_uv | −R_vu]` acting on
//! `(s_u, s_v)`. When `u ≠ v` the two stalk blocks are disjoint and the diagonal
//! of `Cᵀ C` splits cleanly into `colnorm²(R_uv)` on the `u` indices and
//! `colnorm²(R_vu)` on the `v` indices, which is what `hessian_diag`
//! accumulated.
//!
//! `SheafConsistencyPenalty::new` does not forbid a self-loop `u == v`. For a
//! self-loop the coboundary collapses to `δs[e] = (R_uv − R_vu)·s_v`, so the
//! true diagonal of that edge's `L` block is `colnorm²(R_uv − R_vu)`, which
//! carries the `−2·R_uv·R_vu` cross term. `hessian_diag` landed BOTH the
//! `u`-side `colnorm²(R_uv)` and the `v`-side `colnorm²(R_vu)` on the *same*
//! stalk indices and added them, dropping the cross term, so the reported
//! diagonal was `colnorm²(R_uv) + colnorm²(R_vu)` instead of
//! `colnorm²(R_uv − R_vu)` (≈100× too large in the case below).
//! `hessian_diag` (`crates/gam-terms/src/analytic_penalties/sheaf.rs`) now forms
//! `colnorm²(R_uv − R_vu)` on the shared block.
//!
//! `hessian_diag` feeds the inner-Newton / PIRLS diagonal preconditioner and the
//! PSD-curvature pipeline, so a sheaf carrying any self-loop consistency
//! constraint (two linear readouts of one stalk required to agree — a perfectly
//! ordinary cellular-sheaf edge) got a corrupted curvature block while the
//! gradient it is paired with was correct.
//!
//! ## Expectation
//!
//! `hessian_diag(s)` must equal the diagonal of the operator that `hvp` exposes.

use gam::terms::{EdgeRestriction, SheafConsistencyPenalty};
use ndarray::{Array1, array};

/// Build the exact Hessian diagonal from the operator `hvp` exposes: column `j`
/// of `weight·L` is `hvp(s, e_j)`, so the diagonal entry `j` is `hvp(s, e_j)[j]`.
/// This is the ground truth `hessian_diag` claims to reproduce.
fn operator_diagonal(pen: &SheafConsistencyPenalty) -> Array1<f64> {
    let n = pen.total_dim();
    let s = Array1::<f64>::zeros(n); // L is constant in s; any base point works.
    let mut diag = Array1::<f64>::zeros(n);
    for j in 0..n {
        let mut e = Array1::<f64>::zeros(n);
        e[j] = 1.0;
        diag[j] = pen.hvp(s.view(), e.view())[j];
    }
    diag
}

#[test]
fn sheaf_hessian_diag_matches_operator_diagonal_on_self_loop_edge() {
    // A single self-loop edge (0, 0) on a 2-dimensional stalk, with two distinct
    // restriction maps. The coboundary on this edge is (R_uv − R_vu)·s_0, a
    // bona-fide "two readouts of the same stalk must agree" consistency
    // constraint.
    let r_uv = array![[0.9_f64, 0.1], [-0.2, 0.7]];
    let r_vu = array![[1.0_f64, 0.5], [-0.3, 0.8]];
    let edges = vec![(0usize, 0usize)];
    let restrictions = vec![EdgeRestriction::paired(r_uv.clone(), r_vu.clone())];
    let pen = SheafConsistencyPenalty::new(edges, restrictions, 1.0, vec![2])
        .expect("self-loop sheaf penalty must build (new() does not forbid u == v)");

    let n = pen.total_dim();
    let s = array![0.4_f64, -0.7];

    // Sanity: the operator IS the Hessian — gradient(s) == weight·L·s, and L is
    // symmetric. (Establishes that operator_diagonal is the true Hessian diag.)
    let grad = pen.gradient(s.view());
    let l_times_s = {
        let mut acc = Array1::<f64>::zeros(n);
        for j in 0..n {
            let mut e = Array1::<f64>::zeros(n);
            e[j] = 1.0;
            let col = pen.hvp(s.view(), e.view());
            acc.scaled_add(s[j], &col);
        }
        acc
    };
    for i in 0..n {
        assert!(
            (grad[i] - l_times_s[i]).abs() < 1e-12,
            "operator self-consistency broken: gradient != L·s at {i}"
        );
    }

    let diag_true = operator_diagonal(&pen);
    let diag_reported = pen.hessian_diag(s.view());

    // The contract: hessian_diag == diag of the operator hvp/gradient expose.
    let max_err = (0..n)
        .map(|i| (diag_true[i] - diag_reported[i]).abs())
        .fold(0.0_f64, f64::max);

    assert!(
        max_err < 1e-10,
        "SheafConsistencyPenalty::hessian_diag disagrees with its own Hessian \
         operator on a self-loop edge: reported {diag_reported:?} but the \
         operator diagonal (diag of weight·L, the same L that gradient/hvp use) \
         is {diag_true:?} (max |err| = {max_err:.3e}). A self-loop's diagonal is \
         colnorm²(R_uv − R_vu), which carries the −2·R_uv·R_vu cross term."
    );
}
