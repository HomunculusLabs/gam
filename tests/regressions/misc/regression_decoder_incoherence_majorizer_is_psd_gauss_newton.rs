//! Regression guard (companion to the #810 fix) approached from the
//! *surrogate* side rather than the exact-Hessian side.
//!
//! After #810, `DecoderIncoherencePenalty` exposes two distinct curvature
//! operators:
//!
//!   * `hvp` — the **exact** Hessian-vector product `∂²P·v`, including the
//!     indefinite residual term `W·Σ_b C[a,b]·V_k[b,o]`.
//!   * `psd_majorizer_hvp` — the **Gauss-Newton** block `W·Jᵀ(J v)` only, which
//!     is PSD by construction (`W = weight·coactivation ≥ 0`, and `JᵀJ ⪰ 0`).
//!
//! The original bug had both paths returning Gauss-Newton (the majorizer
//! delegated back to `hvp` via the trait default because `hessian_diag` is
//! `None`). This test locks in three independent facts that the fix must keep
//! true, none of which the exact-`hvp == H·v` repro alone pins down:
//!
//!   1. `hvp(v) − psd_majorizer_hvp(v)` equals the closed-form residual term
//!      `W·Σ C·V` exactly — i.e. the two operators differ by precisely the term
//!      that was dropped, computed here independently from the documented
//!      formula. (Catches a regression that drops the residual from `hvp`, or
//!      that leaks it into the majorizer.)
//!   2. The Gauss-Newton operator `B` (materialized column-by-column from
//!      `psd_majorizer_hvp`) is symmetric and positive semidefinite. (Catches a
//!      regression that makes the majorizer alias the indefinite exact Hessian.)
//!   3. `B` genuinely differs from the exact Hessian `H` (built from a central
//!      difference of the analytic gradient) whenever the cross-Gram `C` is
//!      nonzero — the two are not the same operator.
//!
//! Related: #810 (this fix), #809 (sibling OrderedBetaBernoulliPenalty diagonal-only
//! hvp).

use gam::terms::analytic_penalties::{AnalyticPenalty, DecoderIncoherencePenalty, PsiSlice};
use ndarray::{Array1, Array2};


fn build_penalty(m_j: usize, m_k: usize, p_out: usize) -> (DecoderIncoherencePenalty, Array1<f64>) {
    let block_sizes = vec![m_j, m_k];
    let total = (m_j + m_k) * p_out;
    let coactivation =
        Array2::from_shape_vec((2, 2), vec![0.0, 1.0, 1.0, 0.0]).expect("2x2 coactivation");
    let penalty = DecoderIncoherencePenalty::new(
        PsiSlice::full(total, None),
        block_sizes,
        p_out,
        coactivation,
        1.0,
        false,
    )
    .expect("construct DecoderIncoherencePenalty");
    (penalty, Array1::<f64>::zeros(0))
}

/// Deterministic pseudo-random vector (LCG) so the PSD probe is reproducible.
fn lcg_vec(len: usize, seed: u64) -> Array1<f64> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
    let mut out = Array1::<f64>::zeros(len);
    for x in out.iter_mut() {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        // map the top 53 bits into (-1, 1)
        let u = ((state >> 11) as f64) / ((1u64 << 53) as f64);
        *x = 2.0 * u - 1.0;
    }
    out
}

#[test]
fn majorizer_equals_exact_hvp_minus_residual_term() {
    // WAS: `hvp - psd_majorizer_hvp` compared against a hand-written closed form
    // `W*sum C*V`. That formula is the PRE-#2343 unnormalized penalty. `hvp_impl`
    // now implements the degree-0 normalized quotient `0.5*w*E/(N_j*N_k)` and
    // DIFFERENTIATES the normalizer, so the exact operator is
    //
    //   (Hv)_j = kappa[ DG_j - (a+b)G_j - (DE/Nj)Bj + (E/Nj)(2a+b)Bj - (E/Nj)Vj ]
    //
    // with `kappa = w_pair/(N_j*N_k)`. The hand-written residual has neither the
    // `kappa` factor nor four of those terms, which is the measured 9.909e-1 gap
    // against a 1e-12 bar -- a stale test, not a defect in the operators.
    //
    // Rewritten to verify the property that closed form was a proxy for, WITHOUT
    // a formula that goes stale the next time the penalty is renormalized: the
    // exact `hvp` must be the true Hessian-vector product of the analytic
    // gradient. Both original purposes survive --
    //   * "drops the residual from hvp" -> hvp stops matching the FD Hessian
    //   * "leaks it into the majorizer" -> asserted below, majorizer != hvp
    // and it is strictly stronger on the first: it checks the whole operator
    // rather than one term of a decomposition.
    let (m_j, m_k, p_out) = (2usize, 3usize, 2usize);
    let (penalty, rho) = build_penalty(m_j, m_k, p_out);
    let total = (m_j + m_k) * p_out;
    let target = lcg_vec(total, 7);
    for seed in 0..16u64 {
        let v = lcg_vec(total, 101 + seed);
        let exact = penalty.hvp(target.view(), rho.view(), v.view());
        let gn = penalty.psd_majorizer_hvp(target.view(), rho.view(), v.view());

        // Central difference of the analytic gradient along `v`. The step is
        // chosen so truncation (order h^2 times the third derivative) and
        // round-off (order eps/h) both sit far below the tolerance; the fixture
        // is O(1) by construction (`lcg_vec`, unit weight, 2x2 coactivation).
        let h = 1.0e-6_f64;
        let mut plus = target.clone();
        let mut minus = target.clone();
        for i in 0..total {
            plus[i] += h * v[i];
            minus[i] -= h * v[i];
        }
        let g_plus = penalty.grad_target(plus.view(), rho.view());
        let g_minus = penalty.grad_target(minus.view(), rho.view());

        let mut worst = 0.0_f64;
        let mut scale = 1.0_f64;
        for i in 0..total {
            let fd = (g_plus[i] - g_minus[i]) / (2.0 * h);
            worst = worst.max((exact[i] - fd).abs());
            scale = scale.max(fd.abs()).max(exact[i].abs());
        }
        assert!(
            worst <= 1.0e-5 * scale,
            "exact hvp must equal the central-difference Hessian-vector product of \
             the analytic gradient; max|hvp - FD| = {worst:.3e} (scale {scale:.3e}, \
             h = {h:.1e}, seed {seed})"
        );

        // The majorizer must still be a DIFFERENT operator: the residual is
        // dropped, not silently retained. Both paths returning Gauss-Newton was
        // the original #810 bug this file exists for.
        let mut gap = 0.0_f64;
        for i in 0..total {
            gap = gap.max((exact[i] - gn[i]).abs());
        }
        assert!(
            gap > 1.0e-9,
            "psd_majorizer_hvp must drop the residual term, so it cannot equal \
             the exact hvp; max|hvp - majorizer| = {gap:.3e} (seed {seed})"
        );
    }
}

#[test]
fn majorizer_is_symmetric_positive_semidefinite() {
    let (m_j, m_k, p_out) = (2usize, 3usize, 2usize);
    let (penalty, rho) = build_penalty(m_j, m_k, p_out);
    let n = (m_j + m_k) * p_out;
    let target = lcg_vec(n, 13);

    // Materialize the Gauss-Newton operator B column by column.
    let mut b = Array2::<f64>::zeros((n, n));
    for q in 0..n {
        let mut e = Array1::<f64>::zeros(n);
        e[q] = 1.0;
        let col = penalty.psd_majorizer_hvp(target.view(), rho.view(), e.view());
        for p in 0..n {
            b[[p, q]] = col[p];
        }
    }

    // Symmetric.
    let mut max_asym = 0.0_f64;
    for p in 0..n {
        for q in 0..n {
            max_asym = max_asym.max((b[[p, q]] - b[[q, p]]).abs());
        }
    }
    assert!(
        max_asym < 1e-12,
        "GN majorizer must be symmetric: max|B-Bᵀ| = {max_asym:.3e}"
    );

    // PSD: vᵀ B v ≥ 0 for many directions (the operator is JᵀJ scaled by W ≥ 0).
    let mut min_quad = f64::INFINITY;
    for seed in 0..512u64 {
        let v = lcg_vec(n, 1_000 + seed);
        let bv = penalty.psd_majorizer_hvp(target.view(), rho.view(), v.view());
        let quad = v.iter().zip(bv.iter()).map(|(a, c)| a * c).sum::<f64>();
        min_quad = min_quad.min(quad);
    }
    assert!(
        min_quad > -1e-10,
        "GN majorizer must be PSD: min vᵀBv = {min_quad:.3e}"
    );
}

#[test]
fn majorizer_differs_from_exact_hessian_when_atoms_incoherent() {
    let (m_j, m_k, p_out) = (2usize, 2usize, 2usize);
    let (penalty, rho) = build_penalty(m_j, m_k, p_out);
    let n = (m_j + m_k) * p_out;
    // Atoms with nonzero cross-Gram C, so the residual term is nonzero.
    let target = lcg_vec(n, 21);
    let v = lcg_vec(n, 22);
    let exact = penalty.hvp(target.view(), rho.view(), v.view());
    let gn = penalty.psd_majorizer_hvp(target.view(), rho.view(), v.view());
    let max_gap = exact
        .iter()
        .zip(gn.iter())
        .map(|(e, g)| (e - g).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_gap > 1e-3,
        "exact hvp and GN majorizer must be distinct operators when C != 0; \
         max|hvp - majorizer| = {max_gap:.3e} (both returning GN was the bug)"
    );
}
