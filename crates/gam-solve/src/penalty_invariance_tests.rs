//! Unit gates for the exact penalty-map invariance and the judged subspace
//! (#2676).

use super::*;
use gam_terms::construction::CanonicalPenalty;
use ndarray::{Array1, Array2, array};

fn penalty(local: Array2<f64>, total_dim: usize) -> CanonicalPenalty {
    let block = local.nrows();
    let (eigenvalues, eigenvectors) = {
        use gam_linalg::faer_ndarray::FaerEigh;
        local.eigh(faer::Side::Lower).expect("symmetric eigh")
    };
    let positive: Vec<f64> = eigenvalues
        .iter()
        .copied()
        .filter(|value| *value > 1e-12)
        .collect();
    let rank = positive.len();
    let mut root = Array2::<f64>::zeros((rank, block));
    let mut next = 0usize;
    for index in 0..block {
        if eigenvalues[index] > 1e-12 {
            let scale = eigenvalues[index].sqrt();
            for column in 0..block {
                root[[next, column]] = scale * eigenvectors[[column, index]];
            }
            next += 1;
        }
    }
    CanonicalPenalty {
        root,
        col_range: 0..block,
        total_dim,
        nullity: block - rank,
        local,
        prior_mean: Array1::zeros(block),
        positive_eigenvalues: positive,
        op: None,
    }
}

/// `S_2 = c * S_0` is the redundancy `geo_disease_matern` carries, and the
/// invariance must be the exact line `(c, 0, -1)` — recovered from the Gram, not
/// from a pairwise cosine threshold.
#[test]
fn proportional_penalties_yield_the_exact_lambda_null_direction_2676() {
    let s0 = array![[2.0, 0.5, 0.0], [0.5, 1.0, 0.0], [0.0, 0.0, 0.0]];
    let s1 = array![[0.0, 0.0, 0.0], [0.0, 3.0, 1.0], [0.0, 1.0, 2.0]];
    let scale = 0.75_f64;
    let s2 = s0.mapv(|value| scale * value);
    let bundle = vec![penalty(s0, 3), penalty(s1, 3), penalty(s2, 3)];

    let invariance = PenaltyMapInvariance::from_canonical_penalties(&bundle, 3)
        .expect("penalty map gram must decompose");
    assert_eq!(
        invariance.dimension(),
        1,
        "one proportional pair is exactly one linear redundancy"
    );
    let basis = invariance.lambda_basis();
    let w = basis.column(0).to_owned();
    // Fix the sign, then compare against (c, 0, -1) normalised.
    let expected = {
        let raw = Array1::from(vec![scale, 0.0, -1.0]);
        let norm = raw.dot(&raw).sqrt();
        raw.mapv(|value| value / norm)
    };
    let aligned = if w.dot(&expected) < 0.0 {
        w.mapv(|value| -value)
    } else {
        w.clone()
    };
    for index in 0..3 {
        assert!(
            (aligned[index] - expected[index]).abs() < 1e-12,
            "null direction component {index}: got {}, expected {}",
            aligned[index],
            expected[index],
        );
    }
}

/// The property that makes this a defect rather than a tolerance question:
/// along the lifted invariance the rho-curvature is EXACTLY the chain-rule
/// term, so `|t' H_rho t| == sum_k |g_k| t_k^2` whenever the gradient shares a
/// sign on the direction's support. Built from an `H_lambda` that is genuinely
/// positive definite on the identified block, so there is no negative curvature
/// anywhere in the construction.
#[test]
fn lifted_invariance_carries_only_the_chain_rule_term_2676() {
    // V depends on lambda through Lambda = lambda_0 + c*lambda_2 and lambda_1.
    let scale = 0.75_f64;
    let lambdas = Array1::from(vec![1.5_f64, 4.0, 2.25]);
    // A positive definite reduced Hessian in (Lambda, lambda_1).
    let reduced = array![[0.8_f64, -0.2], [-0.2, 0.5]];
    // Pull back: dLambda/dlambda = (1, 0, c), dlambda_1/dlambda = (0, 1, 0).
    let jacobian = array![[1.0_f64, 0.0], [0.0, 1.0], [scale, 0.0]];
    let h_lambda = jacobian.dot(&reduced).dot(&jacobian.t());
    // A gradient that is a genuine differential of the same reduced criterion,
    // so g_lambda = J g_reduced and g_rho = diag(lambda) g_lambda.
    let g_reduced = Array1::from(vec![-3.0e-5_f64, 7.0e-6]);
    let g_lambda = jacobian.dot(&g_reduced);
    let g_rho = Array1::from(vec![
        lambdas[0] * g_lambda[0],
        lambdas[1] * g_lambda[1],
        lambdas[2] * g_lambda[2],
    ]);
    let mut h_rho = Array2::<f64>::zeros((3, 3));
    for i in 0..3 {
        for j in 0..3 {
            h_rho[[i, j]] = lambdas[i] * h_lambda[[i, j]] * lambdas[j];
        }
        h_rho[[i, i]] += g_rho[i];
    }

    let s0 = array![[2.0, 0.5, 0.0], [0.5, 1.0, 0.0], [0.0, 0.0, 0.0]];
    let s1 = array![[0.0, 0.0, 0.0], [0.0, 3.0, 1.0], [0.0, 1.0, 2.0]];
    let s2 = s0.mapv(|value| scale * value);
    let bundle = vec![penalty(s0, 3), penalty(s1, 3), penalty(s2, 3)];
    let invariance =
        PenaltyMapInvariance::from_canonical_penalties(&bundle, 3).expect("gram decomposes");
    let lifted = invariance
        .theta_directions(&lambdas, 3, 0)
        .expect("one lifted direction");
    assert_eq!(lifted.ncols(), 1);
    let t = lifted.column(0).to_owned();

    let rayleigh = t.dot(&h_rho.dot(&t));
    let chain_rule: f64 = (0..3).map(|k| g_rho[k] * t[k] * t[k]).sum();
    let floor: f64 = (0..3).map(|k| g_rho[k].abs() * t[k] * t[k]).sum();
    // `t' (diag(l) H_lambda diag(l)) t` is zero as an IDENTITY, so what is left
    // of it in floating point is round-off of the matrix's own scale — not of
    // the curvature being reported. Judge it against that scale, which is the
    // whole asymmetry: the residual is `eps * ||H||`, and the quantity the gate
    // decides on is `10^11` times smaller than `||H||`.
    let matrix_scale = h_rho.iter().copied().map(f64::abs).fold(0.0_f64, f64::max);
    let intrinsic = rayleigh - chain_rule;
    assert!(
        intrinsic.abs() <= 64.0 * f64::EPSILON * matrix_scale,
        "the intrinsic part {intrinsic:.6e} must be round-off of the matrix scale \
         {matrix_scale:.6e}, i.e. the reparameterisation identity holds"
    );
    // The two gradient components on this direction's support share a sign
    // (both are `g_reduced[0]` times a positive number), so the identity is
    // tight: the quantity the gate compares IS the bound it compares against,
    // and the ONLY thing separating them is that round-off.
    let relative_gap = (rayleigh.abs() - floor).abs() / floor;
    assert!(
        relative_gap <= 1e-9,
        "|t'H t| = {:.17e} and the gradient floor {floor:.17e} must agree to at least 9 \
         digits (gap {relative_gap:.3e}); the gate's verdict is the SIGN of that gap",
        rayleigh.abs()
    );
    // And the whole matrix IS indefinite in rho even though H_lambda is PSD —
    // the negative eigenvalue is manufactured by the reparameterisation.
    use gam_linalg::faer_ndarray::FaerEigh;
    let (eigenvalues, _) = h_rho.eigh(faer::Side::Lower).expect("eigh");
    let minimum = eigenvalues.iter().copied().fold(f64::INFINITY, f64::min);
    assert!(
        minimum < 0.0,
        "the rho Hessian must carry the manufactured negative eigenvalue (got {minimum:.6e})"
    );
    // Deflating it leaves a strictly positive definite judged block.
    let judged = judged_subspace_basis(3, &[], Some(&lifted)).expect("complement is 2-dimensional");
    assert_eq!(judged.ncols(), 2);
    let compressed = compress_to_judged_subspace(&h_rho, &judged);
    let (compressed_eigenvalues, _) = compressed.eigh(faer::Side::Lower).expect("eigh");
    let compressed_minimum = compressed_eigenvalues
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    assert!(
        compressed_minimum > 0.0,
        "the judged complement must be positive definite (got {compressed_minimum:.6e})"
    );
}

/// With nothing to deflate the judged subspace is the interior indicator basis,
/// so `Z' H Z` is the historical sub-block BIT FOR BIT. This is what makes the
/// change inert on every model without a redundant penalty map.
#[test]
fn empty_deflation_reproduces_the_interior_sub_block_bitwise_2676() {
    let h = array![
        [4.0_f64, 1.0, -0.5, 0.25],
        [1.0, 3.0, 0.75, -0.125],
        [-0.5, 0.75, 2.5, 0.5],
        [0.25, -0.125, 0.5, 1.75],
    ];
    let excluded = [1usize, 3];
    let basis = judged_subspace_basis(4, &excluded, None).expect("interior is non-empty");
    let compressed = compress_to_judged_subspace(&h, &basis);
    assert_eq!(compressed.dim(), (2, 2));
    let expected = array![[4.0_f64, -0.5], [-0.5, 2.5]];
    for i in 0..2 {
        for j in 0..2 {
            assert_eq!(
                compressed[[i, j]].to_bits(),
                expected[[i, j]].to_bits(),
                "entry ({i},{j}) must be bit-identical to the sub-block"
            );
        }
    }
}

/// A penalty map with no redundancy must produce no invariance at all — the
/// deflation must not fire on ordinary models.
#[test]
fn independent_penalties_have_no_invariance_2676() {
    let s0 = array![[2.0_f64, 0.5, 0.0], [0.5, 1.0, 0.0], [0.0, 0.0, 0.0]];
    let s1 = array![[0.0_f64, 0.0, 0.0], [0.0, 3.0, 1.0], [0.0, 1.0, 2.0]];
    let s2 = array![[1.0_f64, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 4.0]];
    let bundle = vec![penalty(s0, 3), penalty(s1, 3), penalty(s2, 3)];
    let invariance =
        PenaltyMapInvariance::from_canonical_penalties(&bundle, 3).expect("gram decomposes");
    assert_eq!(invariance.dimension(), 0);
    let lambdas = Array1::from(vec![1.0_f64, 2.0, 3.0]);
    assert!(invariance.theta_directions(&lambdas, 3, 0).is_none());
}

/// A nonzero prior mean can BREAK a redundancy that the quadratic parts alone
/// would show: `S_2 = c S_0` but `mu_2 != mu_0` leaves the linear part of the
/// penalty varying along `(c, 0, -1)`, so the criterion is not invariant and
/// nothing may be deflated. The augmented Gram is what sees this.
#[test]
fn a_prior_mean_that_breaks_proportionality_removes_the_invariance_2676() {
    let s0 = array![[2.0_f64, 0.5, 0.0], [0.5, 1.0, 0.0], [0.0, 0.0, 0.0]];
    let s1 = array![[0.0_f64, 0.0, 0.0], [0.0, 3.0, 1.0], [0.0, 1.0, 2.0]];
    let scale = 0.75_f64;
    let s2 = s0.mapv(|value| scale * value);
    let mut bundle = vec![penalty(s0, 3), penalty(s1, 3), penalty(s2, 3)];
    assert_eq!(
        PenaltyMapInvariance::from_canonical_penalties(&bundle, 3)
            .expect("gram decomposes")
            .dimension(),
        1,
        "with zero prior means the pair is redundant"
    );
    bundle[2].prior_mean = Array1::from(vec![0.4_f64, -0.3, 0.0]);
    let invariance =
        PenaltyMapInvariance::from_canonical_penalties(&bundle, 3).expect("gram decomposes");
    assert_eq!(
        invariance.dimension(),
        0,
        "a differing prior mean makes the criterion genuinely depend on the direction"
    );
}

/// The lift is `diag(lambda)^{-1} w`, not the raw `w`: at unequal lambdas the
/// two are different directions, and only the lifted one annihilates
/// `diag(lambda) H_lambda diag(lambda)`.
#[test]
fn the_lift_is_the_lambda_weighted_tangent_not_the_index_pattern_2676() {
    let scale = 1.0_f64;
    let s0 = array![[1.0_f64, 0.0], [0.0, 0.0]];
    let s2 = s0.mapv(|value| scale * value);
    let bundle = vec![penalty(s0, 2), penalty(s2, 2)];
    let invariance =
        PenaltyMapInvariance::from_canonical_penalties(&bundle, 2).expect("gram decomposes");
    assert_eq!(invariance.dimension(), 1);
    let lambdas = Array1::from(vec![0.5_f64, 8.0]);
    let lifted = invariance
        .theta_directions(&lambdas, 2, 0)
        .expect("lift exists");
    let t = lifted.column(0).to_owned();
    // w ∝ (1, -1); t ∝ (1/0.5, -1/8) = (2, -0.125), which is 16:1, not 1:1.
    let ratio = (t[0] / t[1]).abs();
    assert!(
        (ratio - 16.0).abs() < 1e-9,
        "the lifted tangent must be lambda-weighted (|t0/t1| = {ratio:.6}, expected 16)"
    );
}

/// Embedding into a wider theta (the exact-joint `(rho, psi)` route) leaves the
/// non-rho coordinates at exactly zero, which is what makes the chain-rule
/// identity survive the embedding.
#[test]
fn embedding_into_a_wider_theta_zeroes_the_non_rho_block_2676() {
    let s0 = array![[1.0_f64, 0.0], [0.0, 0.0]];
    let s1 = s0.clone();
    let bundle = vec![penalty(s0, 2), penalty(s1, 2)];
    let invariance =
        PenaltyMapInvariance::from_canonical_penalties(&bundle, 2).expect("gram decomposes");
    let lambdas = Array1::from(vec![2.0_f64, 3.0]);
    let lifted = invariance
        .theta_directions(&lambdas, 5, 0)
        .expect("lift exists");
    assert_eq!(lifted.dim(), (5, 1));
    for row in 2..5 {
        assert_eq!(lifted[[row, 0]], 0.0, "psi coordinate {row} must be zero");
    }
}

/// The disjoint-supports fast path is a THEOREM, and it must agree with the
/// Gram it skips. An ordinary additive model — one penalty per smooth, each on
/// its own coefficient block — has no invariance, and the shortcut must reach
/// that answer for the reason stated (positive-definite diagonal plus a rank-1
/// PSD border) and not by accident.
#[test]
fn disjoint_penalty_supports_have_no_invariance_2676() {
    // Three blocks on 0..2, 2..5, 5..7 — the additive-model layout.
    let s0 = array![[2.0_f64, 0.5], [0.5, 1.0]];
    let s1 = array![[3.0_f64, 1.0, 0.0], [1.0, 2.0, 0.5], [0.0, 0.5, 1.5]];
    let s2 = array![[4.0_f64, -1.0], [-1.0, 2.0]];
    let mut bundle = vec![penalty(s0, 7), penalty(s1, 7), penalty(s2, 7)];
    bundle[1].col_range = 2..5;
    bundle[2].col_range = 5..7;
    let invariance =
        PenaltyMapInvariance::from_canonical_penalties(&bundle, 7).expect("gram decomposes");
    assert_eq!(invariance.dimension(), 0);

    // And the shortcut is not hiding a disagreement: force the general path by
    // overlapping the FIRST TWO ranges by one column with a penalty that is
    // zero on the shared column, so the Gram is still block-diagonal and the
    // answer must be unchanged.
    let mut overlapping = bundle.clone();
    let mut widened = Array2::<f64>::zeros((3, 3));
    widened
        .slice_mut(ndarray::s![..2, ..2])
        .assign(&overlapping[0].local);
    overlapping[0].local = widened;
    overlapping[0].col_range = 0..3;
    overlapping[0].prior_mean = Array1::zeros(3);
    let general =
        PenaltyMapInvariance::from_canonical_penalties(&overlapping, 7).expect("gram decomposes");
    assert_eq!(
        general.dimension(),
        0,
        "the general Gram path must agree with the disjoint-support shortcut"
    );
}
