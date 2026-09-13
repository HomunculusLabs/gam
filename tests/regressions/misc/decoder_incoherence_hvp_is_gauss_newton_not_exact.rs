//! Regression: `DecoderIncoherencePenalty::hvp` is the exact Hessian-vector
//! product the `AnalyticPenalty` contract promises, not the Gauss-Newton
//! curvature.
//!
//! The `AnalyticPenalty` trait draws a sharp line between two operators:
//!
//!   * `hvp` — "Hessian-vector product `H v = (∂²P/∂target²) v`, in closed
//!     form." The **exact** Hessian.
//!   * `psd_majorizer_hvp` — the **PSD surrogate** `B v` with `B ⪰ ∂²P`;
//!     nonconvex penalties override this to return a positive-definite stand-in
//!     instead of the indefinite true Hessian.
//!
//! `DecoderIncoherencePenalty`'s objective is built from the cross-Grams
//! `C_{jk}[a,b] = Σ_o B_j[a,o] B_k[b,o]` and is quartic in the decoder blocks
//! `B`, hence **nonconvex**. Differentiating its gradient along a direction `V`
//! gives **two** terms: the Gauss-Newton piece `Σ_b dC[a,b] B_k[b,o]`, with
//! `dC[a,b] = Σ_o (V_j[a,o] B_k[b,o] + B_j[a,o] V_k[b,o])`, and the residual
//! piece `Σ_b C[a,b] V_k[b,o]` (and the symmetric `_k` block).
//!
//! The defect this test was written for: `hvp` computed only the Gauss-Newton
//! piece and dropped the residual piece, and `psd_majorizer_hvp` was left at the
//! trait default, which delegated back to `hvp`. Both the exact path and the
//! surrogate path returned GN, so a consumer asking `hvp` for the genuine
//! penalized Hessian — an exact Newton step, or the penalized-Hessian log-det
//! that feeds the REML/Laplace marginal likelihood — silently received GN
//! whenever the cross-Gram `C` was nonzero (i.e. whenever the atoms are actually
//! incoherent, the regime the penalty targets). `hvp` now carries both terms and
//! `psd_majorizer_hvp` holds the Gauss-Newton block
//! (`crates/gam-terms/src/analytic_penalties/orthogonality.rs`).
//!
//! Reproduction is closed-form and small: two atoms, two basis rows each,
//! p_out = 2, unit pairwise coactivation. The reference is a central finite
//! difference of the (independently correct) analytic gradient, which by
//! definition is `H v`. The assertion encodes the documented `hvp == H v`
//! contract; the old Gauss-Newton-only `hvp` missed it by ≈ 0.26 here.
//!
//! Related: #809 (sibling: OrderedBetaBernoulliPenalty hvp drops the off-diagonal
//! block). Both are `AnalyticPenalty` curvature-contract defects; #804, #805,
//! #796, #794 are further SAE-penalty curvature/majorizer bugs.

use gam::terms::analytic_penalties::{AnalyticPenalty, DecoderIncoherencePenalty, PsiSlice};
use ndarray::{Array1, Array2};

#[test]
fn decoder_incoherence_hvp_equals_true_hessian_vector_product() {
    // Two decoder atoms, each B_k ∈ ℝ^{2×2} (M_k = 2 basis rows, p_out = 2).
    let block_sizes = vec![2usize, 2usize];
    let p_out = 2usize;
    let total = block_sizes.iter().map(|m| m * p_out).sum::<usize>(); // = 8

    // Pairwise coactivation: only the (0,1) atom pair is penalized, weight 1.
    let coactivation =
        Array2::from_shape_vec((2, 2), vec![0.0, 1.0, 1.0, 0.0]).expect("2x2 coactivation");

    let penalty = DecoderIncoherencePenalty::new(
        PsiSlice::full(total, None),
        block_sizes,
        p_out,
        coactivation,
        1.0,   // weight
        false, // not learnable -> empty rho
    )
    .expect("construct DecoderIncoherencePenalty");
    let rho = Array1::<f64>::zeros(0);

    // Decoder coefficients (β order: [atom0 row0 (o0,o1), atom0 row1, atom1 ...]).
    let target = Array1::from(vec![0.4, -0.3, 0.7, 0.2, -0.5, 0.6, 0.1, -0.8]);
    let v = Array1::from(vec![0.2, 0.5, -0.4, 0.3, 0.6, -0.1, 0.7, -0.2]);

    let analytic_hv = penalty.hvp(target.view(), rho.view(), v.view());

    // True Hessian-vector product = directional derivative of the (correct)
    // analytic gradient: H v = d/dε grad(target + ε v) |_{ε=0}.
    let h = 1e-6_f64;
    let grad_plus = penalty.grad_target((&target + &(&v * h)).view(), rho.view());
    let grad_minus = penalty.grad_target((&target - &(&v * h)).view(), rho.view());
    let fd_hv = (&grad_plus - &grad_minus) / (2.0 * h);

    let max_abs_diff = analytic_hv
        .iter()
        .zip(fd_hv.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);

    assert!(
        max_abs_diff < 1e-5,
        "DecoderIncoherencePenalty::hvp is not the exact Hessian-vector product: \
         max|hvp - H·v| = {max_abs_diff:.6e}. A Gauss-Newton-only hvp misses the \
         residual/cross term `Σ_b C[a,b]·V_k[b,o]`.\n\
         analytic hvp = {analytic_hv:?}\n\
         true   H·v   = {fd_hv:?}"
    );
}
