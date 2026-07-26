//! #2500 — the assignment prior's sparse log-strength curvature operator
//! `∂H_tt/∂ρ_sparse` must be MODELLED for every family that mints a sparse outer
//! coordinate, not only for softmax.
//!
//! `AssignmentStrengthLayout` mints a sparse outer coordinate for exactly three
//! families: `Softmax` (`SoftmaxEntropy`) and `OrderedBetaBernoulli` /
//! `ThresholdGate` (`PenaltyWeight`); `TopK` is `FixedSupport` and carries none.
//! The ordered-Beta–Bernoulli operator is genuinely cross-row and is owned by
//! `dense_exact_a_ordered_bb_sparse_trace`, so the ONE family whose diagonal
//! operator was simply absent is `ThresholdGate` — even though
//! `assignment_prior_log_strength_hdiag_weighted` has always computed it exactly
//! and the arrow assembly writes precisely that diagonal into `block.htt`.
//!
//! These gates pin the three properties the fix must have:
//!   1. the operator map assembles at all for a ThresholdGate ρ (RED before);
//!   2. the sparse block IS `∂H_tt/∂ρ_sparse` of the ASSEMBLED operator, checked
//!      by central finite difference of the cache's own undamped row factors;
//!   3. the dense-route analytic outer gradient — the production consumer that
//!      aborted the outer-BFGS seed evaluation — runs to a finite result.

use super::*;
use ndarray::{Array1, Array2};

/// A `ThresholdGate` twin of `gamma_fd_tiny_fixture`: two periodic atoms on one
/// shared circle, K=2 free logits per row, and a target generated through the
/// SAME gate the fit uses (`a_k = σ((ℓ_k − θ)/τ)`), so the inner state is a real
/// fit rather than a mismatched-forward artefact.
///
/// The logits straddle the threshold on purpose: even rows sit BELOW it
/// (`a < ½` ⇒ prior curvature `λ·s·(1−2a)/τ² > 0`) and odd rows ABOVE it
/// (`a > ½` ⇒ curvature `< 0`). The ThresholdGate prior Hessian is signed, and a
/// fixture that only sampled one side would leave the sign of the operator
/// unexercised.
pub(crate) fn threshold_gate_tiny_fixture() -> (SaeManifoldTerm, Array2<f64>, SaeManifoldRho) {
    let n = 10usize;
    let p = 3usize;
    let k_atoms = 2usize;
    let m = 3usize;
    let tau = 1.0_f64;
    let threshold = 0.0_f64;
    let evaluator = Arc::new(PeriodicHarmonicEvaluator::new(m).unwrap());
    let mut logits = Array2::<f64>::zeros((n, k_atoms));
    let mut coords = vec![Array2::<f64>::zeros((n, 1)), Array2::<f64>::zeros((n, 1))];
    let weights = [
        [
            [0.10, -0.05, 0.03],
            [0.35, -0.20, 0.12],
            [-0.16, 0.18, 0.08],
        ],
        [
            [-0.08, 0.04, 0.06],
            [0.22, 0.10, -0.18],
            [0.11, -0.24, 0.15],
        ],
    ];
    let mut target = Array2::<f64>::zeros((n, p));
    for row in 0..n {
        let phase = (row as f64 + 0.35) / n as f64;
        coords[0][[row, 0]] = phase;
        coords[1][[row, 0]] = (phase + 0.21).fract();
        // Straddle the gate: below the threshold on even rows, above it on odd.
        logits[[row, 0]] = if row % 2 == 0 { -0.8 } else { 0.9 };
        logits[[row, 1]] = if row % 2 == 0 { 0.7 } else { -0.5 };
        for atom in 0..k_atoms {
            let gate = 1.0 / (1.0 + (-(logits[[row, atom]] - threshold) / tau).exp());
            let theta = std::f64::consts::TAU * coords[atom][[row, 0]];
            let basis = [1.0, theta.sin(), theta.cos()];
            for out_col in 0..p {
                for basis_col in 0..m {
                    target[[row, out_col]] +=
                        gate * basis[basis_col] * weights[atom][basis_col][out_col];
                }
            }
        }
    }
    let mut atoms = Vec::with_capacity(k_atoms);
    for atom in 0..k_atoms {
        let (phi, jet) = evaluator.evaluate(coords[atom].view()).unwrap();
        let decoder = Array2::from_shape_fn((m, p), |(basis_col, out_col)| {
            weights[atom][basis_col][out_col]
        });
        atoms.push(
            SaeManifoldAtom::new_with_provided_function_gram(
                format!("tgate_{atom}"),
                SaeAtomBasisKind::Periodic,
                1,
                phi,
                jet,
                decoder,
                Array2::<f64>::eye(m),
            )
            .unwrap()
            .with_basis_second_jet(evaluator.clone()),
        );
    }
    let mode = AssignmentMode::threshold_gate(tau, threshold);
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        logits,
        coords,
        vec![LatentManifold::Circle { period: 1.0 }; k_atoms],
        mode,
    )
    .unwrap();
    let term = SaeManifoldTerm::new(atoms, assignment).unwrap();
    // Moderate-penalty basin, mirroring `converged_state_with_residual`: every
    // channel stays live and the ±h FD probes stay factorizable.
    let rho = SaeManifoldRho::new(
        -1.0,
        -1.0,
        vec![Array1::from_vec(vec![-1.0]), Array1::from_vec(vec![-1.0])],
    )
    .for_assignment(mode);
    (term, target, rho)
}

/// Build the frozen-θ̂ cache the operator map is evaluated against. A ZERO inner
/// budget assembles `H(ρ) = H_data(θ̂) + penalty(ρ)` without re-running the fit,
/// which is exactly the fixed-stratum object `∂H/∂ρ` differentiates.
fn frozen_cache(
    term: &SaeManifoldTerm,
    target: &Array2<f64>,
    rho: &SaeManifoldRho,
) -> (SaeManifoldLoss, ArrowFactorCache) {
    let mut t = term.clone();
    let (_value, loss, cache) = t
        .penalized_quasi_laplace_criterion_with_cache(
            target.view(),
            rho,
            None,
            0,
            0.4,
            1.0e-6,
            1.0e-6,
        )
        .expect("threshold-gate fixed-theta cache");
    (loss, cache)
}

/// The flat logit slot `(row, atom)`'s global index in the cache's t-layout.
/// ThresholdGate always uses the dense (full-support) layout, where the row's
/// free logits occupy the first `assignment_coord_dim()` positions of the block.
fn logit_slots(term: &SaeManifoldTerm, cache: &ArrowFactorCache) -> Vec<(usize, usize, usize)> {
    let mut out = Vec::new();
    let dim = term.assignment.assignment_coord_dim();
    for row in 0..term.n_obs() {
        let base = cache.row_offsets[row];
        for atom in 0..dim.min(cache.row_dims[row]) {
            out.push((row, atom, base + atom));
        }
    }
    out
}

/// #2500 GATE 1 — the operator map must MODEL a ThresholdGate sparse coordinate.
///
/// RED before the fix: `penalty_curvature_operators_by_flat` fell into a
/// catch-all `_ =>` arm and refused with "rho carries a sparse log-strength
/// coordinate under an assignment prior whose ∂H/∂ρ_sparse operator this map does
/// not model". GREEN after: the sparse block is assembled, and it is not the
/// silently-zero operator the refusal existed to prevent.
#[test]
fn threshold_gate_sparse_curvature_operator_is_modelled_2500() {
    let (term, target, rho) = threshold_gate_tiny_fixture();
    let (_loss, cache) = frozen_cache(&term, &target, &rho);
    let sparse = rho
        .sparse_flat_index()
        .expect("a ThresholdGate rho must carry a sparse log-strength coordinate");

    let operators = term
        .penalty_curvature_operators_by_flat(&rho, &cache)
        .expect("#2500: the ThresholdGate sparse curvature operator must be modelled");
    let block = operators
        .get(&sparse)
        .expect("#2500: the sparse coordinate must own a curvature operator");
    let mass = block.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    assert!(
        mass > 1.0e-6,
        "#2500: the assembled sparse operator must be materially nonzero (else the gate \
         would be satisfied by exactly the silently-zero operator the refusal prevented); \
         max|entry| = {mass}"
    );

    // The operator is signed: the fixture straddles the gate, so both curvature
    // signs must be present. A one-sided fixture would let a `max(·,0)`-style
    // clamp ride through unnoticed.
    let mut saw_positive = false;
    let mut saw_negative = false;
    for (_row, _atom, slot) in logit_slots(&term, &cache) {
        let v = block[[slot, slot]];
        if v > 1.0e-8 {
            saw_positive = true;
        }
        if v < -1.0e-8 {
            saw_negative = true;
        }
    }
    assert!(
        saw_positive && saw_negative,
        "#2500: the ThresholdGate prior curvature is SIGNED (λ·s·(1−2a)/τ²); the fixture \
         must exercise both branches: saw_positive={saw_positive}, saw_negative={saw_negative}"
    );
}

/// #2500 GATE 2 — the modelled operator must BE the ρ_sparse-derivative of the
/// operator the arrow assembly actually installs, checked against a central
/// finite difference of the cache's own undamped per-row factors at frozen θ̂.
///
/// This is the property the refusal was protecting: an operator that assembles
/// but does not differentiate the installed `H` is worse than a refusal, because
/// it silently desyncs the ρ-gradient from the criterion.
#[test]
fn threshold_gate_sparse_curvature_operator_matches_finite_difference_2500() {
    let (term, target, rho) = threshold_gate_tiny_fixture();
    let (_loss, cache) = frozen_cache(&term, &target, &rho);
    let sparse = rho
        .sparse_flat_index()
        .expect("a ThresholdGate rho must carry a sparse log-strength coordinate");
    let operators = term
        .penalty_curvature_operators_by_flat(&rho, &cache)
        .expect("#2500: the ThresholdGate sparse curvature operator must be modelled");
    let block = operators
        .get(&sparse)
        .expect("#2500: the sparse coordinate must own a curvature operator");

    // `H_tt^(row) = L Lᵀ` from the cache's UNDAMPED factor: the operator the
    // Laplace normalizer and every ρ-trace are taken against.
    let htt_diag_at = |rho: &SaeManifoldRho| -> Vec<f64> {
        let (_loss, c) = frozen_cache(&term, &target, rho);
        let mut out = Vec::new();
        for row in 0..term.n_obs() {
            let l = c.undamped_factor(row);
            let q = c.row_dims[row];
            for slot in 0..q {
                // (L Lᵀ)_{slot,slot} = Σ_j L[slot, j]².
                let mut acc = 0.0_f64;
                for j in 0..=slot {
                    acc += l[[slot, j]] * l[[slot, j]];
                }
                out.push(acc);
            }
        }
        out
    };

    let base = rho.to_flat();
    let h = 1.0e-5;
    let shifted = |sign: f64| -> SaeManifoldRho {
        let mut flat = base.clone();
        flat[sparse] += sign * h;
        rho.from_flat(flat.view()).expect("shifted rho")
    };
    let plus = htt_diag_at(&shifted(1.0));
    let minus = htt_diag_at(&shifted(-1.0));
    assert_eq!(
        plus.len(),
        minus.len(),
        "#2500: the ±h caches must share one row layout (else the FD compares two strata)"
    );

    let mut worst = 0.0_f64;
    let mut worst_label = String::new();
    let mut exercised = 0usize;
    for (row, atom, slot) in logit_slots(&term, &cache) {
        let fd = (plus[slot] - minus[slot]) / (2.0 * h);
        let analytic = block[[slot, slot]];
        if analytic.abs() > 1.0e-8 {
            exercised += 1;
        }
        let err = (analytic - fd).abs();
        let tol = 1.0e-7 + 1.0e-5 * analytic.abs();
        if err / tol > worst {
            worst = err / tol;
            worst_label =
                format!("row {row} atom {atom}: analytic={analytic:.12e} fd={fd:.12e} tol={tol:.3e}");
        }
    }
    assert!(
        exercised >= 2,
        "#2500: the FD gate must touch at least two live logit slots, else it is vacuous \
         (exercised = {exercised})"
    );
    assert!(
        worst <= 1.0,
        "#2500: the sparse curvature operator must equal ∂H_tt/∂ρ_sparse of the ASSEMBLED \
         operator; worst normalized error {worst:.3} at {worst_label}"
    );
}

/// #2500 GATE 3 — the production consumer. `dense_exact_a_logdet_channels` (and
/// therefore `analytic_outer_rho_gradient_components`, and therefore every outer
/// BFGS seed evaluation) reads the operator map on the dense route. Before the
/// fix a ThresholdGate fit aborted there with a fatal
/// "Fatal outer-objective evaluation failure (outer BFGS seed evaluation)".
#[test]
fn threshold_gate_analytic_outer_gradient_assembles_2500() {
    let (term, target, rho) = threshold_gate_tiny_fixture();
    let (loss, cache) = frozen_cache(&term, &target, &rho);
    let sparse = rho
        .sparse_flat_index()
        .expect("a ThresholdGate rho must carry a sparse log-strength coordinate");

    let components = term
        .analytic_outer_rho_gradient_at_converged(target.view(), &rho, &loss, &cache)
        .expect("#2500: the ThresholdGate dense-route analytic outer gradient must assemble");
    let grad = components.gradient();
    assert!(
        grad.iter().all(|v| v.is_finite()),
        "#2500: the assembled outer gradient must be finite: {grad:?}"
    );
    assert!(
        grad[sparse].abs() > 0.0,
        "#2500: the sparse coordinate must carry a live gradient component, not a \
         structurally-zero one: {grad:?}"
    );
}
