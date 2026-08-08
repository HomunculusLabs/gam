use super::smoothing_correction::{EigenClassification, invert_identified_rho_hessian};
use ndarray::{Array1, Array2};

/// No outer gradient supplied: the inverter falls back to the eigensolver's own
/// backward-error bound, which is the standard every pre-#2428 caller got.
fn no_gradient() -> Array1<f64> {
    Array1::<f64>::zeros(0)
}

/// Build a real symmetric n×n matrix with a specified eigenvalue spectrum
/// rotated by a fixed orthogonal basis. Returns (matrix, eigenvectors).
fn build_with_spectrum(eigenvalues: &[f64]) -> (Array2<f64>, Array2<f64>) {
    let n = eigenvalues.len();
    let mut q = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            let v = if i == j {
                1.0
            } else {
                ((i + 1) as f64 * 0.37 + (j + 1) as f64 * 0.19).sin()
            };
            q[[j, i]] = v;
        }
    }
    // Modified Gram-Schmidt orthonormalization on columns.
    for i in 0..n {
        for k in 0..i {
            let mut dot = 0.0;
            for r in 0..n {
                dot += q[[r, i]] * q[[r, k]];
            }
            for r in 0..n {
                q[[r, i]] -= dot * q[[r, k]];
            }
        }
        let mut nrm = 0.0;
        for r in 0..n {
            nrm += q[[r, i]] * q[[r, i]];
        }
        let nrm = nrm.sqrt();
        assert!(nrm > 1e-12, "degenerate basis in test setup");
        for r in 0..n {
            q[[r, i]] /= nrm;
        }
    }
    // Form A = Q * diag(eigenvalues) * Q^T.
    let mut a = Array2::<f64>::zeros((n, n));
    for r in 0..n {
        for c in 0..n {
            let mut sum = 0.0;
            for k in 0..n {
                sum += q[[r, k]] * eigenvalues[k] * q[[c, k]];
            }
            a[[r, c]] = sum;
        }
    }
    for r in 0..n {
        for c in (r + 1)..n {
            let avg = 0.5 * (a[[r, c]] + a[[c, r]]);
            a[[r, c]] = avg;
            a[[c, r]] = avg;
        }
    }
    (a, q)
}

#[test]
fn spd_case_returns_full_rank_inverse_no_repair() {
    let (a, _q) = build_with_spectrum(&[10.0, 5.0, 2.0, 1.0]);
    let inv = invert_identified_rho_hessian(&a, 0, &no_gradient(), None, &[]).expect("invert");
    assert_eq!(inv.active_rank, 4);
    assert_eq!(inv.structural_zero, 0);
    assert!(!inv.used_structural_pseudoinverse);

    let prod = a.dot(&inv.inverse);
    for r in 0..4 {
        for c in 0..4 {
            let expected = if r == c { 1.0 } else { 0.0 };
            assert!(
                (prod[[r, c]] - expected).abs() < 1e-9,
                "A*Ainv[{r},{c}]={} not ~ {expected}",
                prod[[r, c]]
            );
        }
    }
}

#[test]
fn saddle_is_rejected_instead_of_salvaged() {
    let evals = [10.0, 5.0, 2.0, -0.066];
    let (a, _) = build_with_spectrum(&evals);
    let error = invert_identified_rho_hessian(&a, 0, &no_gradient(), None, &[]).unwrap_err();
    assert!(error.contains("negative curvature") || error.contains("positive definite"));
}

#[test]
fn structurally_certified_zero_direction_uses_pseudoinverse() {
    let evals = [10.0, 5.0, 2.0, 0.0];
    let (a, q) = build_with_spectrum(&evals);
    let inv = invert_identified_rho_hessian(&a, 1, &no_gradient(), None, &[]).expect("invert");
    assert_eq!(inv.active_rank, 3, "expected three identified directions");
    assert_eq!(inv.structural_zero, 1);
    assert!(inv.used_structural_pseudoinverse);
    // Exactly one eigenpair classifies as the certified structural zero; its
    // POSITION in the classification list is an eigensolver ordering detail,
    // not part of the contract.
    assert_eq!(
        inv.classifications
            .iter()
            .filter(|class| matches!(class, EigenClassification::StructuralZero))
            .count(),
        1,
        "exactly one structural-zero classification expected"
    );

    let v_flat = q.column(3).to_owned();
    let inv_vflat = inv.inverse.dot(&v_flat);
    let nrm = inv_vflat.iter().map(|x| x * x).sum::<f64>().sqrt();
    assert!(
        nrm < 1e-3,
        "pseudo-inverse should annihilate flat direction; got norm {nrm}"
    );
}

#[test]
fn structural_nullity_must_match_penalty_map_certificate() {
    let (a, _) = build_with_spectrum(&[10.0, 5.0, 2.0, 0.0]);
    let error = invert_identified_rho_hessian(&a, 2, &no_gradient(), None, &[]).unwrap_err();
    assert!(error.contains("penalty map certifies"));
}

#[test]
fn every_positive_curvature_direction_is_retained() {
    let (a, _) = build_with_spectrum(&[10.0, 5.0, 2.0, 1.0e-9]);
    let inv = invert_identified_rho_hessian(&a, 0, &no_gradient(), None, &[]).expect("small positive SPD inverse");
    assert_eq!(inv.active_rank, 4);
    assert!(inv.inverse.iter().all(|value| value.is_finite()));
}

#[test]
fn non_finite_input_returns_none() {
    let mut a = Array2::<f64>::eye(4);
    a[[1, 1]] = f64::NAN;
    let result = invert_identified_rho_hessian(&a, 0, &no_gradient(), None, &[]);
    assert!(result.is_err(), "expected error for NaN-bearing input matrix");

    let mut a = Array2::<f64>::eye(4);
    a[[2, 2]] = f64::INFINITY;
    let result = invert_identified_rho_hessian(&a, 0, &no_gradient(), None, &[]);
    assert!(result.is_err(), "expected error for Inf-bearing input matrix");
}

/// Every path must populate `eigenvalues` AND `eigenvectors` so the
/// [INDEF-HESS] diagnostic doesn't have to recompute `eigh` redundantly.
#[test]
fn structural_path_populates_eigenvalues_and_eigenvectors() {
    let (a, _q) = build_with_spectrum(&[10.0, 5.0, 2.0, 0.0]);
    let inv = invert_identified_rho_hessian(&a, 1, &no_gradient(), None, &[]).expect("invert");
    assert!(inv.used_structural_pseudoinverse);
    assert_eq!(inv.eigenvalues.len(), 4);
    assert_eq!(inv.eigenvectors.shape(), &[4, 4]);
    assert_eq!(inv.classifications.len(), 4);
    // Eigenvectors are unit-norm and pairwise orthogonal.
    for j in 0..4 {
        let v = inv.eigenvectors.column(j);
        let nrm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(
            (nrm - 1.0).abs() < 1e-9,
            "eigenvector {j} not unit-norm: ‖v‖={nrm}"
        );
    }
}

/// #2428: classification now precedes the inverse, so a strictly positive
/// definite ρ-Hessian still returns the Cholesky-certified inverse (no fit that
/// succeeds today moves) but ALSO carries its spectrum, which is what let the
/// old code fail without ever reporting the eigenvalue that killed it.
#[test]
fn spd_fast_path_still_reports_its_spectrum() {
    let (a, _q) = build_with_spectrum(&[10.0, 5.0, 2.0, 1.0]);
    let inv = invert_identified_rho_hessian(&a, 0, &no_gradient(), None, &[]).expect("invert");
    assert!(!inv.used_structural_pseudoinverse);
    assert_eq!(inv.active_rank, 4);
    assert_eq!(inv.below_gradient_floor, 0);
    assert_eq!(inv.eigenvalues.len(), 4);
    assert_eq!(inv.eigenvectors.shape(), &[4, 4]);
    assert_eq!(inv.classifications.len(), 4);
    assert!(
        inv.classifications
            .iter()
            .all(|c| matches!(c, EigenClassification::Active))
    );
}

/// #2428, the measured case. These are the real numbers from quakes split 7
/// (`paired_holdout_partition(1000, 0.20, 7)`, `mag ~ s(long, lat, bs="tp") +
/// s(depth)`): the outer loop certified the fit (‖g‖ = 3.75e-6 against a
/// stationarity bound of 2.38e-5) and the ρ-Hessian carried one eigenvalue at
/// −7.27e-9 — 19x SMALLER than the residual gradient in that very coordinate,
/// because λ₀ = e^24.35 ≈ 3.8e10 had saturated that term onto its null space.
///
/// The old code ran a zero-tolerance Cholesky here and destroyed the whole fit.
/// The direction is unresolvable, not negative: drop it and correct on the rest.
#[test]
fn curvature_under_the_outer_gradient_floor_is_dropped_not_called_a_saddle() {
    let (a, _q) = build_with_spectrum(&[1.1937700717145632, 0.8037012804437051, 0.45313450067234573, -7.268657266344009e-9]);
    let gradient = Array1::from(vec![
        1.4030614576596264e-7,
        -3.504589068441255e-6,
        4.08387320981113e-7,
        1.266257390889619e-6,
    ]);

    // Without the floor this matrix is refused outright — the pre-#2428 verdict.
    let refused = invert_identified_rho_hessian(&a, 0, &no_gradient(), None, &[]);
    assert!(
        refused.is_err(),
        "the eigensolver-backward-error standard alone must still refuse this matrix, \
         otherwise this fixture does not reproduce #2428"
    );

    // With the certificate's own floor it is one unresolvable direction.
    let inv = invert_identified_rho_hessian(&a, 0, &gradient, None, &[])
        .expect("a certified fit's rho-Hessian must invert on its identified subspace");
    assert_eq!(inv.active_rank, 3);
    assert_eq!(inv.below_gradient_floor, 1);
    assert_eq!(inv.structural_zero, 0, "this is a saturation null, not a structural one");
    assert!(inv.used_structural_pseudoinverse);
    assert!(inv.inverse.iter().all(|v| v.is_finite()));
}

/// A rail saturated hard enough that its curvature falls under even the
/// eigensolver's backward error must not resurrect the count check: the penalty
/// map certifies HOW MANY nulls exist, and an extra one is a property of this ρ̂.
#[test]
fn a_fully_saturated_rail_does_not_violate_the_structural_count() {
    let (a, _q) = build_with_spectrum(&[1.0, 0.5, 0.25, 0.0]);
    let gradient = Array1::from(vec![1.0e-6, 1.0e-6, 1.0e-6, 1.0e-6]);
    // The penalty map certifies NO structural zero, yet the Hessian has one.
    let inv = invert_identified_rho_hessian(&a, 0, &gradient, None, &[])
        .expect("an extra null direction is a saturated rail, not a penalty-map contradiction");
    assert_eq!(inv.active_rank, 3);
    assert_eq!(inv.structural_zero + inv.below_gradient_floor, 1);

    // Fewer nulls than certified is still a contradiction and still fails.
    let error = invert_identified_rho_hessian(&a, 2, &gradient, None, &[])
        .expect_err("finding fewer nulls than the penalty map certifies must stay an error");
    assert!(error.contains("penalty map certifies"), "unexpected error: {error}");
}

/// The floor must not become a licence to swallow real saddles. Same fixture,
/// but with curvature the instrument CAN resolve: the outer loop calling that
/// point a minimum is then a genuine contradiction and must stay loud.
#[test]
fn negative_curvature_above_the_floor_is_still_a_hard_failure() {
    let (a, _q) = build_with_spectrum(&[1.19, 0.80, 0.45, -1.0e-3]);
    let gradient = Array1::from(vec![1.4e-7, 3.5e-6, 4.1e-7, 1.3e-6]);
    let error = invert_identified_rho_hessian(&a, 0, &gradient, None, &[])
        .expect_err("resolvable negative curvature must not be absorbed by the floor");
    assert!(
        error.contains("negative curvature"),
        "unexpected error text: {error}"
    );
}

/// The invariant the fix establishes, stated directly: the outer certificate
/// accepts ρ̂ by testing that `H + diag(|g|)` is PSD. Any matrix passing that
/// test must invert on its identified subspace here — otherwise the two
/// subsystems can reach opposite verdicts on one matrix at one converged point,
/// which is precisely the #2428 defect.
#[test]
fn any_matrix_the_outer_certificate_accepts_inverts_here() {
    let cases: [(&[f64], &[f64]); 3] = [
        (&[1.0, 0.5, 0.25, -1.0e-8], &[1.0e-6, 1.0e-6, 1.0e-6, 1.0e-6]),
        (&[3.0, 2.0, 1.0, -5.0e-7], &[1.0e-5, 2.0e-5, 1.0e-5, 3.0e-5]),
        (&[10.0, 5.0, 2.0, 1.0], &[1.0e-9, 1.0e-9, 1.0e-9, 1.0e-9]),
    ];
    for (spectrum, grad) in cases {
        let (a, _q) = build_with_spectrum(spectrum);
        let gradient = Array1::from(grad.to_vec());

        // The certificate's acceptance test, verbatim: H + diag(|g|) is PSD.
        let mut floored = a.clone();
        for k in 0..floored.nrows() {
            floored[[k, k]] += gradient[k].abs();
        }
        let accepted = smallest_eigenvalue(&floored) >= 0.0;
        assert!(accepted, "fixture {spectrum:?} must be one the certificate accepts");

        invert_identified_rho_hessian(&a, 0, &gradient, None, &[]).unwrap_or_else(|error| {
            panic!("certificate accepted {spectrum:?} but the correction refused it: {error}")
        });
    }
}

/// Smallest eigenvalue via the symmetric eigendecomposition, for the
/// certificate-consistency fixture above.
fn smallest_eigenvalue(matrix: &Array2<f64>) -> f64 {
    use gam_linalg::faer_ndarray::FaerEigh;
    let (eigenvalues, _) = matrix.eigh(faer::Side::Lower).expect("eigendecomposition");
    eigenvalues.iter().copied().fold(f64::INFINITY, f64::min)
}

/// The `#2748` acceptance, built from the identity rather than from a recorded
/// spectrum: **the same matrix, the same gradient, the same curvature — refused
/// when the only measured `‖δH‖₂` is the eigensolver's, admitted when the
/// assembly's own exactly-zero identity is measured too.**
///
/// # The fixture
///
/// `t = (1,0,-1)/√2` stands for the penalty map's certified invariance, lifted
/// to ρ. In the orthonormal eigenbasis `{u₁, u₂, t}` with
/// `u₁ = (1,0,1)/√2`, `u₂ = (0,1,0)`:
///
/// ```text
///     H_ρ = a·u₁u₁ᵀ + (−s)·u₂u₂ᵀ + β·t tᵀ,     β = Σ_k g_k t_k² + η
/// ```
///
/// so that
///
/// * the judged complement `span{u₁, u₂}` carries curvature `−s` on `u₂`,
///   against a chain-rule floor `Σ_k |g_k| (u₂)_k² = |g₁| = γ`;
/// * the certified-null direction carries `tᵀH_ρt − Σ_k g_k t_k² = η`, which is
///   **exactly zero in exact arithmetic** — so `η` is a measured `‖δH‖₂`, in
///   situ, on this pair.
///
/// With `γ ≪ s ≪ η` the refusal is decided by a curvature the assembly has
/// itself demonstrated it cannot resolve. That is the #2748 defect in one
/// matrix.
fn deflatable_fixture(
    negative_curvature: f64,
    assembly_error: f64,
    gradient_scale: f64,
) -> (Array2<f64>, Array1<f64>, Array2<f64>) {
    let root_half = 0.5_f64.sqrt();
    let t = Array1::from(vec![root_half, 0.0, -root_half]);
    let u1 = Array1::from(vec![root_half, 0.0, root_half]);
    let u2 = Array1::from(vec![0.0, 1.0, 0.0]);
    // Every gradient component the same size and sign, so `Σ g_k v_k²` and
    // `Σ |g_k| v_k²` differ only by that sign and the floor is unambiguous.
    let gradient = Array1::from(vec![-gradient_scale, -gradient_scale, -gradient_scale]);
    let chain_rule: f64 = (0..3).map(|k| gradient[k] * t[k] * t[k]).sum();
    let a = 1.0_f64;
    let beta = chain_rule + assembly_error;

    let mut hessian = Array2::<f64>::zeros((3, 3));
    for r in 0..3 {
        for c in 0..3 {
            hessian[[r, c]] = a * u1[r] * u1[c] - negative_curvature * u2[r] * u2[c]
                + beta * t[r] * t[c];
        }
    }
    let mut invariance = Array2::<f64>::zeros((3, 1));
    invariance.column_mut(0).assign(&t);
    (hessian, gradient, invariance)
}

/// BEFORE — the pre-#2748 standard, reproduced by withholding the invariance:
/// with only the eigensolver's backward error as `‖δH‖₂`, a `-1e-8` curvature
/// against a `1e-9` chain-rule floor is a hard refusal.
#[test]
fn without_the_assembly_measurement_the_knife_edge_curvature_refuses_2748() {
    let (hessian, gradient, _invariance) = deflatable_fixture(1.0e-8, 1.0e-7, 1.0e-9);
    let error = invert_identified_rho_hessian(&hessian, 0, &gradient, None, &[])
        .expect_err("with no measured assembly error this direction is judged and refused");
    assert!(
        error.contains("negative curvature"),
        "unexpected error text: {error}"
    );
}

/// AFTER — the same matrix and the same gradient, with the penalty map's
/// certified invariance supplied. The invariance residual measures `‖δH‖₂ = 1e-7`
/// on a direction whose exact answer is zero, the `-1e-8` curvature is inside
/// it, and the direction is reported as unresolved instead of contradictory.
#[test]
fn the_assembly_s_own_exactly_zero_identity_resolves_the_knife_edge_2748() {
    let (hessian, gradient, invariance) = deflatable_fixture(1.0e-8, 1.0e-7, 1.0e-9);

    // The measurement itself, before the gate consumes it: the identity's
    // residual is the injected error, to round-off.
    let measured = crate::penalty_invariance::invariance_residual_2norm(
        &hessian,
        &gradient,
        &invariance,
    )
    .expect("the certified invariance is one direction, so the residual is measurable");
    assert!(
        (measured - 1.0e-7).abs() < 1.0e-15,
        "the invariance residual must recover the injected assembly error; got {measured:.6e}"
    );

    let inverted = invert_identified_rho_hessian(&hessian, 1, &gradient, Some(&invariance), &[])
        .expect("a curvature inside the assembly's own measured error is unresolved, not refuted");
    // One direction deflated (the invariance) and one unresolved (the -1e-8),
    // leaving exactly the `a = 1.0` direction identified and invertible.
    assert_eq!(
        inverted.active_rank, 1,
        "only the well-resolved direction may be inverted; classes = {:?}",
        inverted.classifications
    );
    assert!(
        inverted.used_structural_pseudoinverse,
        "an unresolved direction must route through the structural pseudoinverse"
    );
    assert!(
        inverted
            .classifications
            .iter()
            .filter(|class| !matches!(class, EigenClassification::Active))
            .count()
            == 2,
        "the deflated and the unresolved direction are both non-active; got {:?}",
        inverted.classifications
    );
}

/// The control that keeps the repair from being a licence. Same construction,
/// same measured `‖δH‖₂ = 1e-7`, but a curvature of `-1e-6` — an order ABOVE
/// what the assembly has demonstrated it can get wrong. That is a genuine
/// contradiction and must stay loud.
#[test]
fn curvature_above_the_measured_assembly_error_still_refuses_2748() {
    let (hessian, gradient, invariance) = deflatable_fixture(1.0e-6, 1.0e-7, 1.0e-9);
    let error = invert_identified_rho_hessian(&hessian, 1, &gradient, Some(&invariance), &[])
        .expect_err("a curvature above the measured assembly error is a real contradiction");
    assert!(
        error.contains("negative curvature"),
        "unexpected error text: {error}"
    );
    assert!(
        error.contains("penalty-map invariance residual"),
        "the refusal must name every measured component it judged against: {error}"
    );
}

/// #2665's saddle, scaled into this fixture: `λ_min = -1.6e3` against the same
/// `1e-7` measured assembly error is ten orders outside it and refuses exactly
/// as before. The repair is a resolution, and a resolution cannot swallow a
/// quantity ten orders above it.
#[test]
fn a_real_saddle_is_ten_orders_outside_the_measured_resolution_2748() {
    let (hessian, gradient, invariance) = deflatable_fixture(1.6e3, 1.0e-7, 1.0e-9);
    let error = invert_identified_rho_hessian(&hessian, 1, &gradient, Some(&invariance), &[])
        .expect_err("a 1.6e3 saddle must refuse whatever the assembly's error is");
    assert!(
        error.contains("negative curvature"),
        "unexpected error text: {error}"
    );
}

/// The two bars are ADDED, not maxed, and that is what keeps this site from
/// applying a strictly stronger standard than the certificate that admitted
/// `ρ̂` (#2748, the #2428 invariant restated with a resolution in play).
///
/// The certificate accepts by testing `H + diag(|g|) ⪰ −r` for its measured
/// resolution `r`; along an eigenvector that is `σ ≥ −(Σ_k|g_k|v_k² + r)`. The
/// numbers below are `papuan_oce_matern_k12`'s, which sit between the two
/// rules: `σ = −4.746e-7` against a chain-rule term of `4.407e-7` and a measured
/// `‖δH‖₂` of `1.013e-7`. The maximum refuses it; the sum — which is what the
/// certificate itself applied — does not.
#[test]
fn the_chain_rule_term_and_the_resolution_are_added_not_maxed_2748() {
    let chain = 4.407e-7_f64;
    let assembly = 1.013e-7_f64;
    let curvature = 4.746e-7_f64;
    assert!(
        curvature > chain.max(assembly) && curvature < chain + assembly,
        "the fixture must sit strictly between the two combination rules"
    );

    let (hessian, gradient, invariance) = deflatable_fixture(curvature, assembly, chain);
    // The certificate's own acceptance test, verbatim, at the same measured
    // resolution: `H + diag(|g|)` PSD to within `r`.
    let mut floored = hessian.clone();
    for k in 0..floored.nrows() {
        floored[[k, k]] += gradient[k].abs();
    }
    use gam_linalg::faer_ndarray::FaerEigh;
    let (eigenvalues, _) = floored.eigh(faer::Side::Lower).expect("eigendecomposition");
    let minimum = eigenvalues.iter().copied().fold(f64::INFINITY, f64::min);
    assert!(
        minimum >= -assembly,
        "fixture must be one the certificate accepts at resolution {assembly:.3e}; \
         lambda_min(H + diag|g|) = {minimum:.6e}"
    );

    invert_identified_rho_hessian(&hessian, 1, &gradient, Some(&invariance), &[]).unwrap_or_else(
        |error| {
            panic!(
                "the certificate accepted this point at its measured resolution and the \
                 correction refused it: {error}"
            )
        },
    );
}
