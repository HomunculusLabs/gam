// End-to-end finite-difference oracles for the isotropic-κ (log-κ) joint REML
// outer gradient on real Duchon / Matérn spatial smooths — the #901 gate.
//
// These tests are the headline reproduction #901 was filed against: they
// differentiate the *production* joint REML cost (`evaluate_cost_only`) by
// central finite differences in ρ=log λ and ψ=log κ, and compare it against
// the analytic outer gradient the κ-optimizer actually follows
// (`evaluate_joint_reml_outer_eval_at_theta`). The bug class #901 tracks — the
// range(Sλ)-projected logdet dropping both the penalty-null Schur curvature
// (ρ sign flips) and the moving-subspace dU_S/dψ term (~1e5 ψ blow-ups) — is a
// REML-criterion gradient error that ONLY surfaces on a non-Gaussian GLM
// spatial smooth with a genuinely rank-deficient penalty subspace, which the
// synthetic-matrix unit tests in `gam-custom-family` cannot exercise.
//
// The fixture set was authored in the pre-#1521 monolith under
// `tests/src_modules/smooths/smooth_design_assembly_constraint_tests.rs`. The
// #1521 crate carve moved its private dependencies
// (`SingleBlockExactJointDesignCache`, `try_build_spatial_log_kappa_hyper_dirs`,
// `evaluate_joint_reml_outer_eval_at_theta`, `external_opts_for_design`,
// `spatial_dims_per_term`, `spatial_length_scale_term_indices`,
// `try_build_spatial_term_log_kappa_derivative`) DOWN into the gam-models
// `fit_orchestration::drivers` module, but #1601 commented the test `include!`
// out of `gam_terms::smooth::tests` "for relocation" and the relocation never
// happened — so the #901 gate compiled into NO binary. It is re-homed here,
// where every private driver symbol is in scope via `use super::*` (the
// `design_construction.rs` + `spatial_optimization.rs` files are `include!`d
// flat into one module namespace), and the only cross-crate paths
// (`crate::estimate::ExternalJointHyperEvaluator`,
// `crate::solver::rho_optimizer::OuterEvalOrder`) are rewritten to their carved
// homes `gam_solve::estimate` / `gam_solve::rho_optimizer`.
#[cfg(test)]
mod iso_kappa_reml_gradient_fd_tests {
    use super::*;
    use super::test_support::SingleBlockExactJointDesignCacheTestExt;
    use gam_terms::basis::{
        DuchonBasisSpec, DuchonNullspaceOrder, DuchonOperatorPenaltySpec, MaternBasisSpec, MaternNu,
        OneDimensionalBoundary, SpatialIdentifiability,
    };
    use ndarray::{Array1, Array2, s};

#[test]
fn iso_kappa_duchon_binomial_probit_joint_gradient_matches_finite_difference() {
    let (pass, worst, violations, _) = iso_kappa_fd_variant_driver(
        "duchon_probit_n80",
        80,
        LikelihoodSpec::binomial_probit(),
        false,
        false,
        &[],
    );
    assert!(
        pass,
        "Duchon BinomialProbit n=80 FD failed; worst_psi_rel={worst:.3e}\n  {}",
        violations.join("\n  ")
    );
}

/// Shared driver for iso-κ joint REML gradient FD variants. Returns the
/// worst psi rel_err across the four theta probes (zero / psi_only / base /
/// alt) and panics with full violations only if `assert_pass` is true.
/// Knobs let one-at-a-time variants of the original BinomialProbit Duchon
/// failure isolate which dimension triggers the analytic-vs-FD blow-up.
///
/// `well_conditioned` selects a label set that keeps the inner probit fit well
/// inside the smooth IRLS regime (μ near ½, max|η| small). This matters at
/// small n: the analytic ψ=log κ outer gradient is mathematically exact (the
/// #901 intrinsic-pseudo-logdet kernel), but its GLM cubic-curvature trace term
/// `tr(H_pen⁺ · Xᵀdiag(c⊙X_ψβ̂)X)` is the near-cancellation of two O(10³) halves,
/// so it amplifies the inner PIRLS stationarity floor (‖g‖≈2e-6, the LM-ridge
/// noise floor on near-separable binary data) by ~1.5e3 ≈ 3e-3. Under genuine
/// near-separation (max|η|≈8.8 at n=20 with the original boundary-split labels)
/// BOTH the analytic gradient and the FD oracle inherit that floor on the
/// converged β̂, and their independent ~2e-6 errors blow up to ~1e-2 in the
/// amplified cubic — the FD comparison is then ill-posed, not the gradient.
/// A balanced label set keeps the cancellation halves O(1) so the oracle
/// verifies the *gradient formula* rather than the *conditioning floor*. Proof:
/// the same n=20 Duchon-probit config matches FD to 6e-7 under balanced labels
/// vs 8e-3 under the separated labels (and ρ matches to 1e-5 in both).
///
/// `extra_rho_probes` appends probes at large ρ in addition to the historical
/// near-origin probes. Every historical probe sits at ‖ρ‖ ≤ 1, but the
/// asymptote-rail certificate (#2348) that decides whether a railed joint fit
/// may be minted reads the gradient AT the rail — so the analytic gradient in
/// the only region the certificate consults had never been FD-checked by any
/// gate. Pass `&[]` for the historical probe set. `#2425`.
fn iso_kappa_fd_variant_driver(
    label: &str,
    n: usize,
    family: LikelihoodSpec,
    skip_psi: bool,
    well_conditioned: bool,
    extra_rho_probes: &[f64],
) -> (bool, f64, Vec<String>, Vec<(String, Array1<f64>)>) {
    // A `"*_2d"` label builds an ordinary 2-D feature cloud (the production
    // `matern(x1, x2)` regime: operator triplet {mass, tension, stiffness}, with
    // the per-axis tension and mixed-curvature stiffness blocks that only carry
    // cross-axis structure when d ≥ 2). This is the fast unit-level reproduction
    // of the #1122 stall, whose end-to-end pin is
    // `matern_2d_iso_kappa_outer_gradient_matches_fd`.
    let two_d = label.ends_with("_2d");
    let d = if two_d { 2 } else { 1 };
    let mut data = Array2::<f64>::zeros((n, d));
    let mut y = Array1::<f64>::zeros(n);
    for i in 0..n {
        let t = i as f64 / (n as f64 - 1.0);
        data[[i, 0]] = t;
        let eta = if two_d {
            // A low-discrepancy second axis (golden-ratio fill) keeps the 2-D
            // cloud well-spread, and a genuinely 2-D truth exercises both the
            // signal and the cross-axis curvature blocks.
            let t2 = (i as f64 * 0.618_033_988_749_894_9).fract();
            data[[i, 1]] = t2;
            1.4 * (2.0 * std::f64::consts::PI * t).sin()
                + 0.9 * (2.0 * std::f64::consts::PI * t2).cos()
                + 0.5 * (t - 0.5)
        } else {
            1.4 * (2.0 * std::f64::consts::PI * t).sin() + 0.5 * (t - 0.5)
        };
        let raw = eta + 0.7 * (3.7 * (i as f64) + 1.0).sin();
        y[i] = if family.is_gaussian_identity() {
            raw
        } else if well_conditioned {
            // Smooth, non-separating Bernoulli labels: a deterministic
            // logistic-probability threshold against a fixed phase grid keeps
            // the fitted μ away from {0,1} so the inner Newton system — and the
            // cubic-curvature ψ-trace built from it — stays well-conditioned.
            let p = 1.0 / (1.0 + (-0.6 * (2.0 * std::f64::consts::PI * t).sin()).exp());
            let u = 0.5 * ((5.0 * (i as f64) + 0.5).sin() + 1.0);
            if u < p { 1.0 } else { 0.0 }
        } else if raw > 0.0 {
            1.0
        } else {
            0.0
        };
    }
    let weights = Array1::ones(n);
    let offset = Array1::zeros(n);
    // Duchon is the historical iso-κ FD probe basis; a `"matern_*"` label
    // routes the Matérn ν=5/2 kernel instead so the same gold-standard
    // analytic-vs-FD outer-gradient check covers the Matérn iso-κ REML
    // gradient assembly (which has no other end-to-end FD pin). Thin-plate
    // is deliberately excluded from κ-axis enrollment (see
    // `spatial_term_supports_hyper_optimization`).
    let basis = if label.starts_with("matern") {
        SmoothBasisSpec::Matern {
            feature_cols: (0..d).collect(),
            spec: MaternBasisSpec {
                center_strategy: CenterStrategy::FarthestPoint { num_centers: 8 },
                periodic: None,
                length_scale: gam_terms::basis::MaternLengthScale::fixed(1.0),
                nu: MaternNu::FiveHalves,
                include_intercept: false,
                // The realized Matérn design ALWAYS carries the operator triplet
                // ({mass, tension, stiffness}, see
                // `matern_operator_penalty_triplet_at_length_scale`); the
                // `double_penalty` flag selects the COLD-build value-path penalty
                // but the κ-optimizer re-keys onto the operator triplet either
                // way. A `"*_dp"` label keeps the production default
                // `double_penalty: true` to mirror `matern(x1, x2)` exactly.
                double_penalty: label.contains("_dp"),
                identifiability: MaternIdentifiability::CenterSumToZero,
                aniso_log_scales: None,
            },
            input_scale: None,
        }
    } else {
        SmoothBasisSpec::Duchon {
            feature_cols: vec![0],
            spec: DuchonBasisSpec {
                radial_reparam: None,
                periodic: None,
                center_strategy: CenterStrategy::FarthestPoint { num_centers: 8 },
                length_scale: Some(1.0),
                power: 1.0,
                nullspace_order: DuchonNullspaceOrder::Linear,
                identifiability: SpatialIdentifiability::default(),
                aniso_log_scales: None,
                operator_penalties: DuchonOperatorPenaltySpec::default(),
                boundary: OneDimensionalBoundary::Open,
            },
            input_scale: None,
        }
    };
    let spec = TermCollectionSpec {
        linear_terms: vec![],
        random_effect_terms: vec![],
        smooth_terms: vec![SmoothTermSpec {
            name: "variant_1d".to_string(),
            basis,
            shape: ShapeConstraint::None,
            joint_null_rotation: None,
        }],
    };
    let fit_opts = FitOptions {
        compute_inference: false,
        max_iter: 200,
        tol: 1e-12,
        penalty_shrinkage_floor: None,
        ..FitOptions::default()
    };

    let design = build_term_collection_design(data.view(), &spec).unwrap_or_else(|e| panic!("{} failed: {:?}", "design", e));
    let frozen = freeze_term_collection_from_design(&spec, &design).unwrap_or_else(|e| panic!("{} failed: {:?}", "freeze", e));
    let frozen_design = build_term_collection_design(data.view(), &frozen).unwrap_or_else(|e| panic!("{} failed: {:?}", "frozen design", e));
    let spatial_terms = spatial_length_scale_term_indices(&frozen);
    let dims_per_term = spatial_dims_per_term(&frozen, &spatial_terms);
    // Isotropic κ: one log-κ axis regardless of feature dimension `d` (the 2-D
    // cloud still enrolls a single isotropic κ, not a per-axis η).
    assert_eq!(dims_per_term, vec![1], "{label}: expect one log-κ axis");
    let rho_dim = frozen_design.penalties.len();
    let psi_dim: usize = dims_per_term.iter().sum();
    assert!(psi_dim >= 1);
    eprintln!(
        "[{label} TOPOLOGY] d={d} rho_dim={rho_dim} psi_dim={psi_dim} \
         penalty_sources={:?}",
        frozen_design
            .penalties
            .iter()
            .map(|p| p.col_range.clone())
            .collect::<Vec<_>>()
    );

    let external_opts = external_opts_for_design(&family, &frozen_design, &fit_opts);
    let mut cache = SingleBlockExactJointDesignCache::new(
        data.view(),
        frozen.clone(),
        frozen_design.clone(),
        spatial_terms.clone(),
        rho_dim,
        dims_per_term.clone(),
    )
    .unwrap_or_else(|e| panic!("{} failed: {:?}", "single-block cache", e));
    let mut evaluator = gam_solve::estimate::ExternalJointHyperEvaluator::new(
        y.view(),
        weights.view(),
        &frozen_design.design,
        offset.view(),
        &frozen_design.penalties,
        &external_opts,
        "iso-κ variant FD evaluator",
    )
    .unwrap_or_else(|e| panic!("{} failed: {:?}", "evaluator", e));

    let cost_at = |theta: &Array1<f64>,
                   cache: &mut SingleBlockExactJointDesignCache<'_>,
                   evaluator: &mut gam_solve::estimate::ExternalJointHyperEvaluator<'_>|
     -> f64 {
        cache.ensure_theta(theta).unwrap_or_else(|e| panic!("{} failed: {:?}", "ensure_theta", e));
        let design = cache.design();
        evaluator
            .evaluate_cost_only(
                &design.design,
                &design.penalties,
                &design.nullspace_dims,
                design.linear_constraints.clone(),
                theta,
                rho_dim,
                None,
                "iso-κ variant FD cost-only",
                None,
            )
            .unwrap_or_else(|e| panic!("{} failed: {:?}", "cost-only eval", e))
    };

    let analytic_at = |theta: &Array1<f64>,
                       cache: &mut SingleBlockExactJointDesignCache<'_>,
                       evaluator: &mut gam_solve::estimate::ExternalJointHyperEvaluator<'_>|
     -> (f64, Array1<f64>) {
        cache.ensure_theta(theta).expect("ensure_theta");
        let hyper_dirs = try_build_spatial_log_kappa_hyper_dirs(
            data.view(),
            cache.spec(),
            cache.design(),
            &cache.spatial_terms,
        )
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "hyper dirs build", e))
        .expect("hyper dirs present");
        let (cost, grad, _hess) = evaluate_joint_reml_outer_eval_at_theta(
            evaluator,
            cache.design(),
            theta,
            rho_dim,
            hyper_dirs,
            None,
            gam_solve::rho_optimizer::OuterEvalOrder::ValueAndGradient,
            None,
        )
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "outer eval", e));
        (cost, grad)
    };

    let theta_dim = rho_dim + psi_dim;
    let theta_zero = Array1::<f64>::zeros(theta_dim);
    let mut theta_base = Array1::<f64>::zeros(theta_dim);
    for j in 0..rho_dim {
        theta_base[j] = 0.2 - 0.1 * j as f64;
    }
    let mut theta_psi_only = Array1::<f64>::zeros(theta_dim);
    for k in 0..psi_dim {
        theta_psi_only[rho_dim + k] = 0.4;
    }
    let mut theta_alt = theta_base.clone();
    for j in 0..rho_dim {
        theta_alt[j] = 1.0 + 0.05 * j as f64;
    }
    for k in 0..psi_dim {
        theta_alt[rho_dim + k] = 0.4;
    }

    // Rail / ladder probes (#2425). For each requested ρ value the driver emits
    // one probe per ρ coordinate holding that coordinate at the value, plus an
    // all-ρ probe. `&[11.5]` sits a half e-fold inside `JOINT_RHO_BOUND = 12` so
    // the centered stencil stays in the box; a longer ladder deliberately walks
    // PAST the box, because the evaluator is defined on all of θ and the question
    // "does V saturate at a λ=∞ face" is only answerable outside 12.
    let mut rail_probes: Vec<(String, Array1<f64>)> = Vec::new();
    for &value in extra_rho_probes {
        for j in 0..rho_dim {
            let mut theta_rail = theta_base.clone();
            theta_rail[j] = value;
            rail_probes.push((format!("rho{j}@{value}"), theta_rail));
        }
        let mut theta_all_rail = theta_base.clone();
        for j in 0..rho_dim {
            theta_all_rail[j] = value;
        }
        rail_probes.push((format!("rhoALL@{value}"), theta_all_rail));
    }

    // Central-difference step, DERIVED rather than picked (#2425).
    //
    // The historical `1e-5` assumed an exact evaluator, for which the optimal
    // central step is `h ≈ eps^(1/3) ≈ 6e-6`. This evaluator is not exact: each
    // cost runs an inner PIRLS solve, so its value carries an absolute noise
    // floor `ν`, and the total central-difference error is
    //
    //     ν/h  +  h²·S'''/6            (round-off/noise)  +  (truncation)
    //
    // minimized at `h* = (3ν/S''')^(1/3)`. `zz_measure_psi_only_rho1_fd_step_law_2425`
    // measures `ν` on the production objective by sweeping `h` over 1e-3…1e-7 and
    // reading which law the analytic-vs-FD gap follows. It is unambiguously the
    // NOISE law, not truncation — `gap·h` is flat at ~1.5e-11 across four decades
    // while `gap/h²` sweeps from 1e-2 to 1e10:
    //
    //     h        1e-3     1e-4     1e-5     1e-6     1e-7
    //     gap·h    9.6e-12  2.2e-12  1.0e-11  2.1e-11  1.7e-11
    //     gap/h²   9.6e-3   2.2e0    1.0e4    2.1e7    1.7e10
    //
    // So `ν ≈ 1.5e-11` and, with `S''' = O(1)`, `h* = (3·1.5e-11)^(1/3) ≈ 3.6e-4`.
    // At the old `1e-5` the oracle's OWN error is `ν/h ≈ 1.5e-6`, which is ~10%
    // of a near-zero component like the `psi_only` ρ₁ row (~1.5e-5) — that is
    // what made `iso_kappa_duchon_n_smaller_fd` fail at `rel=5.077e-3` against a
    // `5e-3` gate, with the analytic gradient correct all along. The measurement
    // confirms the tail of the argument too: at `h=1e-3`, where noise is
    // negligible, FD agrees with the analytic value to 0.6%.
    //
    // Raising `h` costs truncation on the WELL-scaled components, and that is
    // affordable: `h²·S'''/6 ≈ 9e-8·S'''/6`, four orders inside `rel_tol`.
    let h = 3e-4_f64;
    let rel_tol = 5e-3_f64;
    let mut violations: Vec<String> = Vec::new();
    let mut analytic_by_probe: Vec<(String, Array1<f64>)> = Vec::new();
    let mut worst_psi_rel = 0.0_f64;
    let base_probes: [(&str, &Array1<f64>); 4] = [
        ("zero", &theta_zero),
        ("psi_only", &theta_psi_only),
        ("base", &theta_base),
        ("alt", &theta_alt),
    ];
    let all_probes: Vec<(&str, &Array1<f64>)> = base_probes
        .into_iter()
        .chain(rail_probes.iter().map(|(n, t)| (n.as_str(), t)))
        .collect();
    for (probe, theta) in all_probes {
        let (cost_an, grad_an) = analytic_at(theta, &mut cache, &mut evaluator);
        assert!(cost_an.is_finite(), "{label} {probe}: cost not finite");
        analytic_by_probe.push((probe.to_string(), grad_an.clone()));
        // Objective↔gradient desync probe: the analytic gradient path
        // (evaluate_joint_reml_outer_eval_at_theta) and the cost-only FD
        // path (evaluate_cost_only) must agree on the COST itself at the
        // unperturbed θ. If they disagree, FD differences a different
        // function than the gradient differentiates and no gradient fix
        // can make them match. eprintln for the diagnostic build only.
        let cost_via_fd_path = cost_at(theta, &mut cache, &mut evaluator);
        eprintln!(
            "[{label} {probe}] COST an={:+.10e} fd_path={:+.10e} diff={:.3e}",
            cost_an,
            cost_via_fd_path,
            (cost_an - cost_via_fd_path).abs()
        );
        for j in 0..theta_dim {
            let is_psi = j >= rho_dim;
            if skip_psi && is_psi {
                continue;
            }
            let mut plus = theta.clone();
            plus[j] += h;
            let mut minus = theta.clone();
            minus[j] -= h;
            let cp = cost_at(&plus, &mut cache, &mut evaluator);
            let cm = cost_at(&minus, &mut cache, &mut evaluator);
            let fd = (cp - cm) / (2.0 * h);
            let denom = fd.abs().max(grad_an[j].abs()).max(1e-3);
            let rel = (grad_an[j] - fd).abs() / denom;
            let kind = if is_psi { "psi" } else { "rho" };
            eprintln!(
                "[{label} {probe}] {kind} j={j} an={:+.4e} fd={:+.4e} rel={:.3e}",
                grad_an[j], fd, rel
            );
            if is_psi && rel > worst_psi_rel {
                worst_psi_rel = rel;
            }
            if rel >= rel_tol {
                violations.push(format!(
                    "{probe} {kind} j={j}: analytic={:+.6e} fd={:+.6e} rel={:.3e}",
                    grad_an[j], fd, rel
                ));
            }
        }
    }
    let pass = violations.is_empty();
    eprintln!(
        "[{label} SUMMARY] pass={pass} worst_psi_rel={worst_psi_rel:.3e} \
             violations={}",
        violations.len()
    );
    (pass, worst_psi_rel, violations, analytic_by_probe)
}

#[test]
fn iso_kappa_duchon_gaussian_identity_fd() {
    let (pass, worst, violations, _) = iso_kappa_fd_variant_driver(
        "duchon_gaussian",
        80,
        LikelihoodSpec::gaussian_identity(),
        false,
        false,
        &[],
    );
    assert!(
        pass,
        "Gaussian Identity FD failed; worst_psi_rel={worst:.3e}\n  {}",
        violations.join("\n  ")
    );
}

/// The Matérn ν=5/2 analogue of `iso_kappa_duchon_gaussian_identity_fd`.
///
/// The isotropic-analytic κ optimizer was observed to stall at n≳1000 on a
/// well-conditioned 1-D Matérn Gaussian fit (grad_norm ≈ 0.5·|f|, nowhere
/// near stationary) while the Duchon path converges — and the Matérn iso-κ
/// *outer* REML gradient had no end-to-end FD pin (only basis-level log-κ
/// derivative tests). This closes that gap: it differences the same exact
/// analytic ψ=log κ outer gradient that the optimizer follows against a
/// central finite difference of the REML cost. If the analytic gradient is
/// wrong, the optimizer's stall is explained and this fails loudly.
#[test]
fn iso_kappa_matern_gaussian_identity_fd() {
    let (pass, worst, violations, _) = iso_kappa_fd_variant_driver(
        "matern_gaussian",
        80,
        LikelihoodSpec::gaussian_identity(),
        false,
        false,
        &[],
    );
    assert!(
        pass,
        "Matérn iso-κ Gaussian-identity outer-gradient FD failed; \
             worst_psi_rel={worst:.3e}\n  {}",
        violations.join("\n  ")
    );
}
/// Fast unit-level reproduction of the #1122 stall: an ordinary 2-D
/// `matern(x1, x2)` Gaussian fit whose isotropic-κ outer REML gradient must
/// match a central finite difference of the production REML cost. This is the
/// d=2 analogue of `iso_kappa_matern_gaussian_identity_fd`: the 1-D Matérn
/// already matched FD, so the desync that stalled the κ-optimizer at its
/// iteration cap (analytic ≠ FD on `psi_kappa`, #1122) lives in the cross-axis
/// tension / mixed-curvature stiffness operator blocks that only carry
/// off-diagonal structure when d ≥ 2.
#[test]
fn iso_kappa_matern_2d_gaussian_identity_fd() {
    let (pass, worst, violations, _) = iso_kappa_fd_variant_driver(
        "matern_gaussian_2d",
        120,
        LikelihoodSpec::gaussian_identity(),
        false,
        false,
        &[],
    );
    assert!(
        pass,
        "Matérn 2-D iso-κ Gaussian-identity outer-gradient FD failed (the #1122 \
             cross-axis operator-penalty ψ-derivative desync); worst_psi_rel={worst:.3e}\n  {}",
        violations.join("\n  ")
    );
}

/// DIAGNOSTIC (#1122): is the residual ~1.6e-3 relative gap in the production
/// end-to-end audit (`matern_2d_iso_kappa_outer_gradient_matches_fd`,
/// analytic=16.11 vs fd=16.08 at the small auto-init length scale) a genuine
/// missing derivative term, or a finite-difference truncation artifact of the
/// steep κ^{2m} operator penalty at the production-init `log κ ≈ 2.5`?
///
/// This sweeps the central-FD step `h` on the ψ=log κ axis at a high-κ θ that
/// mirrors the production init (the operator triplet scales like κ^{2m} with
/// m = ν + d/2 = 3.5, so V(ψ) has a large third derivative and the central-FD
/// truncation error ∝ h²·V''' dominates). A TRUNCATION artifact shrinks ≈ 100×
/// per 10× shrink in `h` (until the roundoff floor); a REAL derivative bug
/// leaves an `h`-independent floor. The analytic gradient is computed once;
/// only `h` changes. This is a diagnostic oracle (FD is sanctioned in tests).
#[test]
fn iso_kappa_matern_2d_psi_fd_step_sweep_diagnostic() {
    use ndarray::Array2 as NdArray2;
    let n = 150usize;
    let d = 2usize;
    // EXACT mirror of the end-to-end gate's dataset
    // (`matern_2d_iso_kappa_outer_gradient_matches_fd`): uniform-random 2-D
    // cloud via splitmix64 (seed 0x9A7E_7212_0001), truth sin(2πa)·sin(2πb) +
    // N(0, 0.05²). Reproducing the same X(ψ) is the only way the fast harness
    // sees the SAME analytic ψ-gradient (≈ +16.11) and the SAME h-flat gap.
    let mut st: u64 = 0x9A7E_7212_0001;
    fn splitmix(s: &mut u64) -> u64 {
        *s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *s;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn next_unit(s: &mut u64) -> f64 {
        (splitmix(s) >> 11) as f64 / (1u64 << 53) as f64
    }
    fn next_gauss(s: &mut u64) -> f64 {
        let u1 = next_unit(s).max(1.0e-12);
        let u2 = next_unit(s);
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
    let mut data = NdArray2::<f64>::zeros((n, d));
    let mut y = Array1::<f64>::zeros(n);
    let sigma = 0.05;
    for i in 0..n {
        let a = next_unit(&mut st);
        let b = next_unit(&mut st);
        data[[i, 0]] = a;
        data[[i, 1]] = b;
        y[i] = (2.0 * std::f64::consts::PI * a).sin() * (2.0 * std::f64::consts::PI * b).sin()
            + sigma * next_gauss(&mut st);
    }
    let weights = Array1::ones(n);
    let offset = Array1::zeros(n);
    // CRITICAL (#1122): the FrozenTransform `Z` and the nullspace-shrinkage
    // decision are computed ONCE at the BASE length scale the design is frozen
    // at, then held fixed across the κ-sweep. Production does a pilot ρ-fit /
    // data re-seed BEFORE freezing, so its frozen base ls is 0.28665832
    // (ψ_base = −ln(ls) = 1.2494 = the audit θ₀ ψ), NOT the raw
    // `auto_initial_length_scale` (0.0812 → ψ=2.51). Freezing the harness at the
    // auto-init ls gave a DIFFERENT `Z` (and a different objective: V≈16.31 vs
    // production ≈16.11), which is why the fast harness was internally
    // consistent yet never reproduced the production audit gap. Freeze at the
    // production base ls so the harness `Z` matches production byte-for-byte and
    // the probe ψ = 1.2494 lands AT the freeze point.
    let length_scale = 0.286_658_32_f64;
    eprintln!("[PSI-SWEEP] length_scale={length_scale:.6} log_kappa={:.4}", -length_scale.ln());
    // The production default center count for n=150, d=2 is 37 (see
    // `default_matern_center_count`/`default_num_centers`).
    let spec = TermCollectionSpec {
        linear_terms: vec![],
        random_effect_terms: vec![],
        smooth_terms: vec![SmoothTermSpec {
            name: "matern_2d".to_string(),
            basis: SmoothBasisSpec::Matern {
                feature_cols: (0..d).collect(),
                spec: MaternBasisSpec {
                    center_strategy: CenterStrategy::FarthestPoint { num_centers: 37 },
                    periodic: None,
                    length_scale: gam_terms::basis::MaternLengthScale::fixed(length_scale),
                    nu: MaternNu::FiveHalves,
                    include_intercept: false,
                    double_penalty: true,
                    identifiability: MaternIdentifiability::CenterSumToZero,
                    aniso_log_scales: None,
                },
                input_scale: None,
            },
            shape: ShapeConstraint::None,
            joint_null_rotation: None,
        }],
    };
    let fit_opts = FitOptions {
        compute_inference: false,
        max_iter: 200,
        tol: 1e-12,
        penalty_shrinkage_floor: None,
        ..FitOptions::default()
    };
    let design = build_term_collection_design(data.view(), &spec).unwrap();
    let frozen = freeze_term_collection_from_design(&spec, &design).unwrap();
    let frozen_design = build_term_collection_design(data.view(), &frozen).unwrap();
    let spatial_terms = spatial_length_scale_term_indices(&frozen);
    let dims_per_term = spatial_dims_per_term(&frozen, &spatial_terms);
    let rho_dim = frozen_design.penalties.len();
    let psi_dim: usize = dims_per_term.iter().sum();
    let theta_dim = rho_dim + psi_dim;
    let family = LikelihoodSpec::gaussian_identity();
    let external_opts = external_opts_for_design(&family, &frozen_design, &fit_opts);
    let mut cache = SingleBlockExactJointDesignCache::new(
        data.view(),
        frozen.clone(),
        frozen_design.clone(),
        spatial_terms.clone(),
        rho_dim,
        dims_per_term.clone(),
    )
    .unwrap();
    let mut evaluator = gam_solve::estimate::ExternalJointHyperEvaluator::new(
        y.view(),
        weights.view(),
        &frozen_design.design,
        offset.view(),
        &frozen_design.penalties,
        &external_opts,
        "psi-sweep FD evaluator",
    )
    .unwrap();

    // TEMP #1122 diagnostic: HARNESS DESIGN FINGERPRINT, to diff against the
    // production `/tmp/gam_prod_fingerprint.txt`. The harness is internally
    // consistent (analytic≈FD to 4.5e-9) but evaluates V≈16.31 while production
    // evaluates V≈16.11 — so the mirror is structurally imperfect. This dumps
    // the SAME fields the production FINGERPRINT block dumps. Removed before
    // merge.
    {
        eprintln!("[FINGERPRINT] HARNESS design.design dims = ({}, {})", frozen_design.design.nrows(), frozen_design.design.ncols());
        eprintln!("[FINGERPRINT] HARNESS n_penalties = {}", frozen_design.penalties.len());
        for (pi, p) in frozen_design.penalties.iter().enumerate() {
            let fro: f64 = p.local.iter().map(|v| v * v).sum::<f64>().sqrt();
            eprintln!(
                "[FINGERPRINT] HARNESS penalty[{pi}] col_range={:?} local_dims={:?} hint={:?} fro={fro:.10e}",
                p.col_range, p.local.dim(), p.structure_hint
            );
        }
        eprintln!("[FINGERPRINT] HARNESS nullspace_dims = {:?}", frozen_design.nullspace_dims);
        for &ti in spatial_terms.iter() {
            if let Some(t) = frozen.smooth_terms.get(ti) {
                let s = match &t.basis {
                    SmoothBasisSpec::Matern { spec, .. } => format!(
                        "Matern{{nu={:?}, ls={:.8}, dp={}, ident={}, aniso={:?}, centers_kind={}}}",
                        spec.nu,
                        spec.length_scale.resolved().unwrap(),
                        spec.double_penalty,
                        match &spec.identifiability {
                            MaternIdentifiability::FrozenTransform { transform } =>
                                format!("FrozenTransform{{z_dims={:?}}}", transform.dim()),
                            other => format!("{other:?}"),
                        },
                        spec.aniso_log_scales,
                        match &spec.center_strategy {
                            CenterStrategy::UserProvided(c) => format!("UserProvided(n={})", c.nrows()),
                            other => format!("{other:?}"),
                        },
                    ),
                    other => format!("{other:?}"),
                };
                eprintln!("[FINGERPRINT] HARNESS term[{ti}] basis_kind={s}");
            }
            if let Some(t) = frozen_design.smooth.terms.get(ti) {
                if let gam_terms::basis::BasisMetadata::Matern {
                    centers, input_scale, length_scale, ..
                } = &t.metadata
                {
                    let csum: f64 = centers.iter().map(|v| v.abs()).sum();
                    let c00 = centers.get((0, 0)).copied().unwrap_or(f64::NAN);
                    let c01 = centers.get((0, 1)).copied().unwrap_or(f64::NAN);
                    eprintln!(
                        "[FINGERPRINT] HARNESS meta.Matern length_scale={length_scale:.10} input_scale={input_scale:?} centers_abs_sum={csum:.10e} c[0,0]={c00:.10} c[0,1]={c01:.10}"
                    );
                }
            }
        }
    }

    let cost_at = |theta: &Array1<f64>,
                   cache: &mut SingleBlockExactJointDesignCache<'_>,
                   evaluator: &mut gam_solve::estimate::ExternalJointHyperEvaluator<'_>|
     -> f64 {
        cache.ensure_theta(theta).unwrap();
        let design = cache.design();
        evaluator
            .evaluate_cost_only(
                &design.design,
                &design.penalties,
                &design.nullspace_dims,
                design.linear_constraints.clone(),
                theta,
                rho_dim,
                None,
                "psi-sweep cost-only",
                None,
            )
            .unwrap()
    };
    let analytic_at = |theta: &Array1<f64>,
                       cache: &mut SingleBlockExactJointDesignCache<'_>,
                       evaluator: &mut gam_solve::estimate::ExternalJointHyperEvaluator<'_>|
     -> Array1<f64> {
        cache.ensure_theta(theta).unwrap();
        let hyper_dirs = try_build_spatial_log_kappa_hyper_dirs(
            data.view(),
            cache.spec(),
            cache.design(),
            &cache.spatial_terms,
        )
        .unwrap()
        .expect("hyper dirs present");
        let (_c, grad, _h) = evaluate_joint_reml_outer_eval_at_theta(
            evaluator,
            cache.design(),
            theta,
            rho_dim,
            hyper_dirs,
            None,
            gam_solve::rho_optimizer::OuterEvalOrder::ValueAndGradient,
            None,
        )
        .unwrap();
        grad
    };

    // θ mirroring the production audit θ₀ captured from the end-to-end gate
    // (`matern_2d_iso_kappa_outer_gradient_matches_fd`): the warm-started ρ is
    // strongly negative (λ ≈ e^{-4..-6}, the penalty is nearly OFF, so the
    // criterion is data-fit + ½log|H+Sλ| dominated) and ψ = log κ ≈ 1.25. This
    // is the regime that exposed the residual ~1.6e-3 gap; the earlier (ρ=0,
    // ψ=2.5) probe did not. rho_dim is 3 in both (operator triplet), so the ρ
    // slots line up; if rho_dim differs we pad with the last value.
    let prod_rho = [-3.632_687_635_594_657, -5.970_607_752_248_795, -4.804_720_434_766_625];
    let prod_psi = 1.249_464_308_750_002_1;
    let mut theta = Array1::<f64>::zeros(theta_dim);
    for j in 0..rho_dim {
        theta[j] = prod_rho.get(j).copied().unwrap_or(*prod_rho.last().unwrap());
    }
    for k in 0..psi_dim {
        theta[rho_dim + k] = prod_psi;
    }
    eprintln!("[PSI-SWEEP] rho_dim={rho_dim} probing theta={:?}", theta.to_vec());
    let grad = analytic_at(&theta, &mut cache, &mut evaluator);
    let psi_idx = rho_dim;
    let analytic = grad[psi_idx];
    eprintln!("[PSI-SWEEP] analytic ∂V/∂ψ (ValueAndGradient) = {analytic:+.8e}");
    {
        // TEMP #1122: value + data checksums at θ₀, to diff vs production. The
        // designs now fingerprint-match, but the harness ∂V/∂ψ (≈41) ≠ production
        // (≈16). Same X/penalty/θ → the COST itself differs: isolate whether it
        // is the data (y/weights/offset) or the evaluator options.
        let v0 = cost_at(&theta, &mut cache, &mut evaluator);
        let y_abs_sum: f64 = y.iter().map(|v| v.abs()).sum();
        let y0 = y.get(0).copied().unwrap_or(f64::NAN);
        let xd = cache.design().design.to_dense();
        let x_abs_sum: f64 = xd.iter().map(|v| v.abs()).sum();
        let x00 = xd.get((0, 0)).copied().unwrap_or(f64::NAN);
        let x01 = xd.get((0, 1)).copied().unwrap_or(f64::NAN);
        let x_row0_sum: f64 = xd.row(0).iter().map(|v| v.abs()).sum();
        eprintln!(
            "[FINGERPRINT] HARNESS V(theta0)={v0:.10e} y_abs_sum={y_abs_sum:.10e} y[0]={y0:.10} X_abs_sum={x_abs_sum:.10e} X[0,0]={x00:.10} X[0,1]={x01:.10} X_row0_abs={x_row0_sum:.10e} dims=({},{}) n={n}",
            xd.nrows(), xd.ncols()
        );
    }
    // The PRODUCTION audit takes its analytic gradient from a
    // ValueGradientHessian eval, NOT ValueAndGradient. If the ψ-gradient
    // returned by the two orders differs, the audit differences the value path
    // against a gradient computed in a different lane → an objective↔gradient
    // desync that no FD step can close. Probe both at the SAME θ₀.
    {
        cache.ensure_theta(&theta).unwrap();
        let hyper_dirs = try_build_spatial_log_kappa_hyper_dirs(
            data.view(),
            cache.spec(),
            cache.design(),
            &cache.spatial_terms,
        )
        .unwrap()
        .expect("hyper dirs present");
        let (_c, grad_vgh, _h) = evaluate_joint_reml_outer_eval_at_theta(
            &mut evaluator,
            cache.design(),
            &theta,
            rho_dim,
            hyper_dirs,
            None,
            gam_solve::rho_optimizer::OuterEvalOrder::ValueGradientHessian,
            None,
        )
        .unwrap();
        eprintln!(
            "[PSI-SWEEP] analytic ∂V/∂ψ (ValueGradientHessian) = {:+.8e} (delta vs V&G = {:.3e})",
            grad_vgh[psi_idx],
            (grad_vgh[psi_idx] - analytic).abs()
        );
    }
    // VALUE oracle #2: the production audit differences `eval_full(Value)` →
    // `evaluate_joint_reml_outer_eval_at_theta(.., Value)`, NOT
    // `evaluate_cost_only`. If THIS value path disagrees with the gradient while
    // `evaluate_cost_only` agrees, the desync is between the two value lanes.
    let value_via_outer_eval = |theta: &Array1<f64>,
                                cache: &mut SingleBlockExactJointDesignCache<'_>,
                                evaluator: &mut gam_solve::estimate::ExternalJointHyperEvaluator<'_>|
     -> f64 {
        cache.ensure_theta(theta).unwrap();
        let hyper_dirs = try_build_spatial_log_kappa_hyper_dirs(
            data.view(),
            cache.spec(),
            cache.design(),
            &cache.spatial_terms,
        )
        .unwrap()
        .expect("hyper dirs present");
        let (c, _g, _h) = evaluate_joint_reml_outer_eval_at_theta(
            evaluator,
            cache.design(),
            theta,
            rho_dim,
            hyper_dirs,
            None,
            gam_solve::rho_optimizer::OuterEvalOrder::Value,
            None,
        )
        .unwrap();
        c
    };

    let mut prev_gap: Option<f64> = None;
    let mut min_gap = f64::INFINITY;
    for &h in &[1e-2_f64, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7] {
        let mut plus = theta.clone();
        plus[psi_idx] += h;
        let mut minus = theta.clone();
        minus[psi_idx] -= h;
        let cp = cost_at(&plus, &mut cache, &mut evaluator);
        let cm = cost_at(&minus, &mut cache, &mut evaluator);
        let fd = (cp - cm) / (2.0 * h);
        let gap = (analytic - fd).abs();
        // Second FD using the outer-eval Value lane (the production audit's lane).
        let cp2 = value_via_outer_eval(&plus, &mut cache, &mut evaluator);
        let cm2 = value_via_outer_eval(&minus, &mut cache, &mut evaluator);
        let fd2 = (cp2 - cm2) / (2.0 * h);
        let gap2 = (analytic - fd2).abs();
        min_gap = min_gap.min(gap);
        let shrink = prev_gap.map(|p| p / gap).unwrap_or(f64::NAN);
        eprintln!(
            "[PSI-SWEEP] h={h:.0e} fd_costonly={fd:+.8e} gap={gap:.3e} | fd_outereval={fd2:+.8e} gap2={gap2:.3e} shrink={shrink:.2}"
        );
        prev_gap = Some(gap);
    }
    eprintln!("[PSI-SWEEP] min_gap_over_sweep={min_gap:.3e} analytic={analytic:.8e} (truncation→shrinks ~100×/decade; real bug→h-flat floor)");
    // The gradient is correct iff some step drives the gap well below the
    // audit's 1e-3·|fd| DESYNC band — i.e. the residual is FD truncation, not a
    // missing derivative term. A real derivative bug would floor the gap
    // regardless of `h`.
    assert!(
        min_gap < 5e-3 * analytic.abs().max(1.0),
        "ψ=log κ outer gradient never matches FD across the h-sweep \
         (min_gap={min_gap:.3e}, analytic={analytic:.6e}): this is a REAL \
         derivative bug, not FD truncation"
    );
}

/// The production-default (`double_penalty: true`) 2-D Matérn variant. This is
/// the closest unit-level mirror of `matern(x1, x2)` and isolates whether the
/// #1122 stall is driven by the double-penalty value-path / re-key topology.
#[test]
fn iso_kappa_matern_2d_dp_gaussian_identity_fd() {
    let (pass, worst, violations, _) = iso_kappa_fd_variant_driver(
        "matern_gaussian_2d_dp",
        120,
        LikelihoodSpec::gaussian_identity(),
        false,
        false,
        &[],
    );
    assert!(
        pass,
        "Matérn 2-D double-penalty iso-κ Gaussian-identity outer-gradient FD \
             failed (the #1122 stall); worst_psi_rel={worst:.3e}\n  {}",
        violations.join("\n  ")
    );
}

#[test]
fn iso_kappa_duchon_binomial_logit_fd() {
    let (pass, worst, violations, _) =
        iso_kappa_fd_variant_driver("duchon_logit", 80, LikelihoodSpec::binomial_logit(), false, false, &[]);
    assert!(
        pass,
        "BinomialLogit FD failed; worst_psi_rel={worst:.3e}\n  {}",
        violations.join("\n  ")
    );
}

// No `iso_kappa_thinplate_*_fd` companion to the Duchon FD tests above:
// thin-plate is deliberately excluded from the spatial κ-axis enrollment
// by `spatial_term_supports_hyper_optimization` (a scalar TPS κ creates
// the flat ρ/κ valleys tracked in #718 / #721 / #731 / #732), so there
// is no analytic κ-gradient on which an FD comparison could land.

#[test]
fn iso_kappa_duchon_n_smaller_fd() {
    let (pass, worst, violations, _) = iso_kappa_fd_variant_driver(
        "duchon_probit_n20",
        20,
        LikelihoodSpec::binomial_probit(),
        false,
        // Well-conditioned labels: at n=20 the separated label set drives the
        // inner probit fit to max|η|≈8.8, where the GLM cubic-curvature ψ-trace
        // amplifies the inner PIRLS KKT floor (~2e-6) by ~1.5e3 into a ~1e-2
        // analytic-vs-FD gap that is a conditioning artifact of BOTH sides, not
        // a gradient error (#901 kernel is exact: balanced labels match to 6e-7).
        true,
        &[],
    );
    assert!(
        pass,
        "Duchon Probit n=20 FD failed; worst_psi_rel={worst:.3e}\n  {}",
        violations.join("\n  ")
    );
}

#[test]
fn iso_kappa_duchon_no_psi_fd() {
    let (pass, _worst, violations, _) = iso_kappa_fd_variant_driver(
        "duchon_probit_rho_only",
        80,
        LikelihoodSpec::binomial_probit(),
        true,
        false,
        &[],
    );
    assert!(
        pass,
        "Duchon Probit ρ-only FD failed:\n  {}",
        violations.join("\n  ")
    );
}

/// #2425 MEASUREMENT (reports, never fails): is the analytic iso-κ outer
/// gradient still FD-correct NEAR THE RAIL?
///
/// Motivation. `spatial_length_scale_optimization_monotone_*` never reaches its
/// monotonicity assertion — the joint fit refuses to mint because the outer
/// certificate finds the railed coordinates non-stationary. The declining
/// certificate printed an 18-e-fold probe ladder in which
/// `ĉ = −e^ρ·∂V/∂ρ` — the quantity that is CONSTANT on a genuine λ→∞ tail —
/// instead tracks `e^ρ` across the whole box, i.e. `∂V/∂ρ ≈ const ≈ −0.3`, and
/// then GROWS to −1.9 at the ρ=11.5 rail rather than decaying to zero.
///
/// Two readings are possible and they demand opposite fixes:
///   1. the analytic gradient is right, the joint box `JOINT_RHO_BOUND = 12`
///      simply stops 18 e-folds short of the `RHO_BOUND = 30` rail the
///      asymptote certificate (#2348) was calibrated against, so the tail has
///      not begun and the certificate correctly declines; or
///   2. the analytic gradient is WRONG out there, and every railed joint fit
///      has been judged against a gradient no gate has ever checked.
///
/// Every historical FD probe in this file sits at ‖ρ‖ ≤ 1. The rail is the only
/// region the certificate consults and the only region never measured. This
/// test measures it on both bases and both link classes.
#[test]
fn zz_measure_iso_kappa_rail_gradient_fd_2425() {
    for (label, n, family) in [
        ("duchon_gaussian", 80usize, LikelihoodSpec::gaussian_identity()),
        ("matern_gaussian", 80, LikelihoodSpec::gaussian_identity()),
        ("duchon_logit", 80, LikelihoodSpec::binomial_logit()),
    ] {
        let (pass, worst, violations, _) =
            // #2444: probe BOTH faces. `+11.5` is the upper rail this gate was
            // written for; `-11.5` is its mirror a half e-fold inside the LOWER
            // bound, which is where every failing checkpoint in the kappa cluster
            // actually rails. A derivative wrong at one bound is not automatically
            // wrong at the other, and the rationale for measuring the rail at all
            // -- "the one region the certificate consults is the one region no gate
            // has ever measured" -- applied verbatim to the lower face until now.
            iso_kappa_fd_variant_driver(label, n, family, false, false, &[11.5, -11.5]);
        eprintln!(
            "[zz-rail-2425] {label}: pass={pass} worst_psi_rel={worst:.3e} \
             violations={}",
            violations.len()
        );
        for v in &violations {
            eprintln!("[zz-rail-2425] {label}: {v}");
        }
    }
}

/// #2444: the executable form of what the probe above measures.
///
/// The analytic outer gradient must match a central finite difference **at the
/// rails**, on both faces of the box. `zz_measure_iso_kappa_rail_gradient_fd_2425`
/// has computed exactly this since #2425 and printed `pass=false` into a run the
/// harness records as `ok`, so the violation has been visible and unenforced —
/// the same shape as every other false green in #2422. A measurement nobody is
/// obliged to read does not constrain anything.
///
/// Currently RED for Duchon and green for Matérn, which is the point: the
/// separation is 64x through the same optimizer at the lower face, and
/// `fd - analytic` is positive in every violation across both faces and both
/// links. Matern is the control — its worst also rose ~20x when the lower probes
/// were added and it still passes, so the lower face is harder for both families
/// and only Duchon exceeds.
#[test]
fn iso_kappa_rail_gradient_matches_fd_at_both_faces_2444() {
    let mut summary: Vec<String> = Vec::new();
    let mut failing: Vec<String> = Vec::new();
    for (label, n, family) in [
        ("duchon_gaussian", 80usize, LikelihoodSpec::gaussian_identity()),
        ("matern_gaussian", 80, LikelihoodSpec::gaussian_identity()),
        ("duchon_logit", 80, LikelihoodSpec::binomial_logit()),
    ] {
        let (pass, worst, violations, _) =
            iso_kappa_fd_variant_driver(label, n, family, false, false, &[11.5, -11.5]);
        summary.push(format!(
            "{label}: pass={pass} worst_psi_rel={worst:.3e} violations={}",
            violations.len()
        ));
        for violation in &violations {
            failing.push(format!("{label}: {violation}"));
        }
    }
    assert!(
        failing.is_empty(),
        "analytic outer gradient must match FD at both rails; {} violation(s)\n  {}\n  {}",
        failing.len(),
        summary.join("\n  "),
        failing.join("\n  ")
    );
}

/// #2425 MEASUREMENT (reports, never fails): does the iso-κ REML criterion
/// SATURATE at a λ=∞ face, or is it asymptotically linear in ρ?
///
/// `zz_measure_iso_kappa_rail_gradient_fd_2425` establishes that the analytic
/// gradient is FD-correct at ρ=11.5, so the monotone fixtures' refusal is not a
/// derivative defect: the criterion really is descending at the rail with
/// `∂V/∂ρ ≈ −0.3` and `ĉ = −e^ρ ∂V/∂ρ` growing like `e^ρ` instead of settling.
/// Two explanations survive and they demand opposite fixes.
///
///   1. The λ=∞ tail exists but begins OUTSIDE `JOINT_RHO_BOUND = 12`. The
///      asymptote certificate's own `ASYMPTOTE_PROBE_COUNT` comment says its
///      window was sized against rails at `RHO_BOUND = 30`, so a box that stops
///      at 12 can be 18 e-folds short of the region the certificate needs. Then
///      `V` saturates somewhere past 12 and the box is the bug.
///   2. There is no λ=∞ face at all, because the `½log|H| − ½log|S|₊`
///      cancellation leaves a residual linear term `(r_H − r_S)/2 · ρ`. Then `V`
///      keeps falling linearly forever and no box width can help; the rank
///      bookkeeping is the bug.
///
/// The discriminator is simply `V` far outside the box, which nothing forbids —
/// the evaluator is a function of θ and the ±12 clamp lives in the optimizer's
/// bound vectors, not in the criterion. Walking ρ out to 30 separates the two:
/// saturating `V` with `ĉ → const` is (1); `V` linear in ρ with `∂V/∂ρ → const`
/// is (2). Reported per ρ coordinate, so a per-block rank defect is visible as
/// a per-block slope.
#[test]
fn zz_measure_iso_kappa_face_saturation_ladder_2425() {
    // Out to `RHO_BOUND = 30` — the bound the asymptote certificate was
    // calibrated against — well past `JOINT_RHO_BOUND = 12`.
    const LADDER: [f64; 9] = [6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0, 27.0, 30.0];
    // `matern_gaussian_2d` vs `matern_gaussian_2d_dp` differ ONLY in
    // `double_penalty` (the driver reads `label.contains("_dp")`), so the pair
    // is a one-variable test of whether the double-penalty assembly is what
    // carries the spurious λ-linear term measured in #2454
    // (`∂V/∂ρ = −c·λ`, c = 2.87e-9, on the double-penalty monotone fixture).
    for (label, n, family) in [
        ("matern_gaussian", 80usize, LikelihoodSpec::gaussian_identity()),
        ("duchon_gaussian", 80, LikelihoodSpec::gaussian_identity()),
        ("matern_gaussian_2d", 120, LikelihoodSpec::gaussian_identity()),
        ("matern_gaussian_2d_dp", 120, LikelihoodSpec::gaussian_identity()),
    ] {
        let (pass, worst, violations, _) =
            iso_kappa_fd_variant_driver(label, n, family, false, false, &LADDER);
        eprintln!(
            "[zz-ladder-2425] {label}: fd_pass={pass} worst_psi_rel={worst:.3e} \
             violations={}",
            violations.len()
        );
        for v in &violations {
            eprintln!("[zz-ladder-2425] {label}: {v}");
        }
    }
}

/// #2450 — the derivation, executable: at large ρ the ENTIRE outer gradient is
/// the ρ-PRIOR, so the criterion has no λ=∞ face for any rail certificate to
/// find.
///
/// `FitOptions::default()` carries `RhoPrior::default() = Normal { mean: 0.0,
/// sd: 3.0 }` (`gam-spec/src/lib.rs`), and `rho_prior_eval` adds per coordinate
/// `cost += ½(ρ−mean)²/sd²`, `grad += (ρ−mean)/sd²`. So once ρ is large enough
/// that the REML/LAML part's own λ→∞ face is reached — its ρ-derivative decays
/// like `O(e^{−ρ})` — the surviving gradient is exactly `ρ/sd² = ρ/9`.
///
/// Why this is a gate and not a curiosity. Every rail path in
/// `rho_optimizer::run` decides by asking whether `ĉ = −e^ρ·∂V/∂ρ` is CONSTANT
/// over a probe run (`try_certify_asymptote_rail` #2348 Inc 1,
/// `try_tail_snap_to_rail`, `detect_wrong_rail_pullback` #2392). With
/// `∂V/∂ρ → ρ/9` we get `ĉ → −ρe^ρ/9 → −∞`, so that law can never hold and no
/// coordinate can ever be certified at an asymptote. That is measured, not
/// argued: `..._monotone_..._for_matern` refuses with `|Pg| = 1.333e0` = `12/9`
/// exactly, and `..._two_feature` with `1.886e0` = `√2·12/9`, both at ρ railed
/// on 12 — i.e. their entire residual projected gradient IS this prior.
///
/// `gam-custom-family`'s deterministic entry passes `RhoPrior::Flat` explicitly,
/// which is why the same certificates work there. If this test fails, the
/// default ρ-prior or its scale changed, or the criterion stopped including it —
/// any of which invalidates the reasoning recorded on #2450, so read that issue
/// before adjusting a tolerance here.
#[test]
fn outer_gradient_at_large_rho_is_exactly_the_rho_prior_2450() {
    /// `RhoPrior::default()`'s standard deviation. Not imported, deliberately:
    /// the point of this gate is to fail if the shipped default stops matching
    /// the value the #2450 derivation was carried out at.
    const PRIOR_SD: f64 = 3.0;
    // ρ ≥ 21 is where the ladder measured the REML part's own ρ-derivative to be
    // below 1e-10, so the prior is all that is left. (ψ's gradient decays with
    // it: 3.6e-11 at ρ=30.)
    const SATURATED: [f64; 4] = [21.0, 24.0, 27.0, 30.0];

    // Only `matern_gaussian`: these are the rungs the ladder actually measured
    // out to 30 (2.3333 / 2.6667 / 3.0000 / 3.3333 on every ρ coordinate, FD
    // agreeing to 1e-10, ψ decaying 2.9e-7 → 3.6e-11). The Duchon run's high
    // rungs were truncated out of that job's captured output, so asserting on
    // them here would be a guess; extend once they are measured.
    for (label, family) in [("matern_gaussian", LikelihoodSpec::gaussian_identity())] {
        let (_pass, _worst, _violations, grads) =
            iso_kappa_fd_variant_driver(label, 80, family, false, false, &SATURATED);
        let mut checked = 0usize;
        for value in SATURATED {
            let probe = format!("rhoALL@{value}");
            let grad = &grads
                .iter()
                .find(|(name, _)| *name == probe)
                .unwrap_or_else(|| panic!("{label}: probe {probe} missing"))
                .1;
            let expected = value / (PRIOR_SD * PRIOR_SD);
            // Only the ρ block carries the smoothing prior; ψ is a length-scale
            // coordinate and is asserted to have decayed instead.
            for (j, &observed) in grad.iter().enumerate() {
                if j + 1 == grad.len() {
                    assert!(
                        observed.abs() <= 1.0e-6,
                        "{label} {probe}: psi gradient should have decayed at a \
                         saturated rho, got {observed:+.6e}"
                    );
                    continue;
                }
                let rel = (observed - expected).abs() / expected;
                assert!(
                    rel <= 1.0e-4,
                    "{label} {probe} rho j={j}: outer gradient should be the \
                     Normal(0, sd={PRIOR_SD}) rho-prior's {expected:.10e} \
                     (= rho/sd^2) once the REML part has saturated, got \
                     {observed:.10e} (rel={rel:.3e}). See #2450."
                );
                checked += 1;
            }
        }
        assert!(checked >= SATURATED.len(), "{label}: nothing was checked");
        eprintln!("[#2450-gate] {label}: {checked} rho components == rho/sd^2");
    }
}

/// Owned 1-D Duchon BinomialProbit setup shared verbatim across the
/// `duchon_probit_*` mechanism pins. Holds only non-self-referential
/// owners; each test constructs its own `external_opts` / cache /
/// evaluator inline (the borrow-entangled, per-test-labelled parts).
struct DuchonProbitSetup {
    data: Array2<f64>,
    y: Array1<f64>,
    weights: Array1<f64>,
    offset: Array1<f64>,
    frozen: TermCollectionSpec,
    frozen_design: TermCollectionDesign,
    spatial_terms: Vec<usize>,
    dims_per_term: Vec<usize>,
    rho_dim: usize,
    psi_dim: usize,
}

/// Builds the verbatim 1-D Duchon BinomialProbit data + frozen design used
/// by the ψ-trace / per-row / PIRLS-determinism mechanism pins.
fn build_duchon_probit_setup() -> DuchonProbitSetup {
    let n = 80usize;
    let mut data = Array2::<f64>::zeros((n, 1));
    let mut y = Array1::<f64>::zeros(n);
    for i in 0..n {
        let t = i as f64 / (n as f64 - 1.0);
        data[[i, 0]] = t;
        let eta = 1.4 * (2.0 * std::f64::consts::PI * t).sin() + 0.5 * (t - 0.5);
        y[i] = if eta + 0.7 * (3.7 * (i as f64) + 1.0).sin() > 0.0 {
            1.0
        } else {
            0.0
        };
    }
    let weights = Array1::ones(n);
    let offset = Array1::zeros(n);
    let spec = TermCollectionSpec {
        linear_terms: vec![],
        random_effect_terms: vec![],
        smooth_terms: vec![SmoothTermSpec {
            name: "duchon_1d".to_string(),
            basis: SmoothBasisSpec::Duchon {
                feature_cols: vec![0],
                spec: DuchonBasisSpec {
                    radial_reparam: None,
                    periodic: None,
                    center_strategy: CenterStrategy::FarthestPoint { num_centers: 8 },
                    length_scale: Some(1.0),
                    power: 1.0,
                    nullspace_order: DuchonNullspaceOrder::Linear,
                    identifiability: SpatialIdentifiability::default(),
                    aniso_log_scales: None,
                    operator_penalties: DuchonOperatorPenaltySpec::all_active(),
                    boundary: OneDimensionalBoundary::Open,
                },
                input_scale: None,
            },
            shape: ShapeConstraint::None,
            joint_null_rotation: None,
        }],
    };
    let design = build_term_collection_design(data.view(), &spec).unwrap_or_else(|e| panic!("{} failed: {:?}", "design", e));
    let frozen = freeze_term_collection_from_design(&spec, &design).unwrap_or_else(|e| panic!("{} failed: {:?}", "freeze", e));
    let frozen_design = build_term_collection_design(data.view(), &frozen).unwrap_or_else(|e| panic!("{} failed: {:?}", "frozen design", e));
    let spatial_terms = spatial_length_scale_term_indices(&frozen);
    let dims_per_term = spatial_dims_per_term(&frozen, &spatial_terms);
    let rho_dim = frozen_design.penalties.len();
    let psi_dim: usize = dims_per_term.iter().sum();
    DuchonProbitSetup {
        data,
        y,
        weights,
        offset,
        frozen,
        frozen_design,
        spatial_terms,
        dims_per_term,
        rho_dim,
        psi_dim,
    }
}

/// #2425 MEASUREMENT (reports, never fails): is the marginal `psi_only` ρ₁
/// analytic-vs-FD gap central-difference TRUNCATION or a noise floor?
#[test]
fn zz_measure_psi_only_rho1_fd_step_law_2425() {
    // The one component that fails `iso_kappa_duchon_n_smaller_fd`, and it is
    // near-zero and marginal in EVERY config, not just at n=20:
    //
    //   duchon_probit_n20      an=-1.9454e-5 fd=-1.4377e-5  rel=5.077e-3  FAIL
    //   duchon_probit_rho_only an=-1.9108e-5 fd=-1.5746e-5  rel=3.362e-3  pass
    //   duchon_logit           an=+9.1413e-6 fd=+6.5455e-6  rel=2.596e-3  pass
    //   duchon_gaussian        an=+1.7330e-5 fd=+1.6750e-5  rel=5.799e-4  pass
    //
    // The driver's `denom = max(|fd|, |an|, 1e-3)` floors the denominator at
    // 1e-3, so on a component of size ~1.5e-5 the 5e-3 relative gate is really
    // an ABSOLUTE 5e-6 gate, and every config sits within a factor of two of it.
    // Before anyone moves that number, the h-law decides which side is wrong:
    //
    //   gap ∝ h²   → central-difference TRUNCATION. The analytic gradient is
    //                right and the ORACLE's fixed h=1e-5 is too coarse for a
    //                component with large third derivative; Richardson (or a
    //                smaller h) fixes it and no tolerance needs to move.
    //   gap flat, or growing as h shrinks → a noise floor (the inner PIRLS
    //                stationarity residual propagating into both sides). Then
    //                the absolute floor must be DERIVED from that residual
    //                rather than left at a magic 1e-3 denominator.
    //
    // Reports only. `ratio` is gap/h²: constant ⇒ truncation.
    let DuchonProbitSetup {
        data,
        y,
        weights,
        offset,
        frozen,
        frozen_design,
        spatial_terms,
        dims_per_term,
        rho_dim,
        psi_dim,
    } = build_duchon_probit_setup();
    let fit_opts = FitOptions {
        compute_inference: false,
        max_iter: 200,
        tol: 1e-12,
        penalty_shrinkage_floor: None,
        ..FitOptions::default()
    };
    let external_opts = external_opts_for_design(
        &LikelihoodSpec::binomial_probit(),
        &frozen_design,
        &fit_opts,
    );
    let mut cache = SingleBlockExactJointDesignCache::new(
        data.view(),
        frozen.clone(),
        frozen_design.clone(),
        spatial_terms.clone(),
        rho_dim,
        dims_per_term.clone(),
    )
    .unwrap_or_else(|e| panic!("{} failed: {:?}", "cache", e));
    let mut evaluator = gam_solve::estimate::ExternalJointHyperEvaluator::new(
        y.view(),
        weights.view(),
        &frozen_design.design,
        offset.view(),
        &frozen_design.penalties,
        &external_opts,
        "psi_only rho1 FD step law",
    )
    .unwrap_or_else(|e| panic!("{} failed: {:?}", "evaluator", e));

    // The driver's `psi_only` probe: every ρ at 0, ψ at 0.4.
    let theta_dim = rho_dim + psi_dim;
    let mut theta = Array1::<f64>::zeros(theta_dim);
    for k in 0..psi_dim {
        theta[rho_dim + k] = 0.4;
    }
    const COORD: usize = 1;

    let eval_at = |theta: &Array1<f64>,
                   order: gam_solve::rho_optimizer::OuterEvalOrder,
                   cache: &mut SingleBlockExactJointDesignCache<'_>,
                   evaluator: &mut gam_solve::estimate::ExternalJointHyperEvaluator<'_>| {
        cache.ensure_theta(theta).unwrap_or_else(|e| panic!("{} failed: {:?}", "ensure_theta", e));
        let hyper_dirs = try_build_spatial_log_kappa_hyper_dirs(
            data.view(),
            cache.spec(),
            cache.design(),
            &cache.spatial_terms,
        )
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "hyper dirs build", e))
        .expect("hyper dirs present");
        evaluate_joint_reml_outer_eval_at_theta(
            evaluator,
            cache.design(),
            theta,
            rho_dim,
            hyper_dirs,
            None,
            order,
            None,
        )
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "outer eval", e))
    };

    let (cost, grad, _h) = eval_at(
        &theta,
        gam_solve::rho_optimizer::OuterEvalOrder::ValueAndGradient,
        &mut cache,
        &mut evaluator,
    );
    let analytic = grad[COORD];
    eprintln!(
        "[zz-steplaw-2425] cost={cost:.12e} analytic_rho{COORD}={analytic:+.10e}"
    );
    let mut previous: Option<(f64, f64)> = None;
    for h in [
        1.0e-3, 3.0e-4, 1.0e-4, 3.0e-5, 1.0e-5, 3.0e-6, 1.0e-6, 3.0e-7, 1.0e-7,
    ] {
        let mut plus = theta.clone();
        plus[COORD] += h;
        let mut minus = theta.clone();
        minus[COORD] -= h;
        let (cp, _, _) = eval_at(
            &plus,
            gam_solve::rho_optimizer::OuterEvalOrder::Value,
            &mut cache,
            &mut evaluator,
        );
        let (cm, _, _) = eval_at(
            &minus,
            gam_solve::rho_optimizer::OuterEvalOrder::Value,
            &mut cache,
            &mut evaluator,
        );
        let fd = (cp - cm) / (2.0 * h);
        let gap = analytic - fd;
        // Richardson against the previous (3x coarser) rung: for a clean
        // O(h²) stencil this cancels the leading truncation term.
        let richardson = previous.map(|(hc, fdc)| {
            let ratio = (hc / h).powi(2);
            (ratio * fd - fdc) / (ratio - 1.0)
        });
        eprintln!(
            "[zz-steplaw-2425] h={h:.1e} fd={fd:+.10e} gap={gap:+.4e} \
             gap/h2={:.4e} richardson={}",
            gap / (h * h),
            richardson
                .map(|r| format!("{r:+.10e} (gap {:+.4e})", analytic - r))
                .unwrap_or_else(|| "-".to_string())
        );
        previous = Some((h, fd));
    }
}

/// Behavioral pin for the iso-κ Duchon ψ-axis under BinomialProbit: the
/// analytic outer gradient must agree with a centered finite difference of the
/// production objective.
#[test]
fn iso_kappa_duchon_outer_gradient_matches_centered_fd() {
    let DuchonProbitSetup {
        data,
        y,
        weights,
        offset,
        frozen,
        frozen_design,
        spatial_terms,
        dims_per_term,
        rho_dim,
        psi_dim,
    } = build_duchon_probit_setup();
    let fit_opts = FitOptions {
        compute_inference: false,
        max_iter: 200,
        tol: 1e-12,
        penalty_shrinkage_floor: None,
        ..FitOptions::default()
    };

    let external_opts = external_opts_for_design(
        &LikelihoodSpec::binomial_probit(),
        &frozen_design,
        &fit_opts,
    );
    let mut cache = SingleBlockExactJointDesignCache::new(
        data.view(),
        frozen.clone(),
        frozen_design.clone(),
        spatial_terms.clone(),
        rho_dim,
        dims_per_term.clone(),
    )
    .unwrap_or_else(|e| panic!("{} failed: {:?}", "cache", e));
    let mut evaluator = gam_solve::estimate::ExternalJointHyperEvaluator::new(
        y.view(),
        weights.view(),
        &frozen_design.design,
        offset.view(),
        &frozen_design.penalties,
        &external_opts,
        "iso-kappa Duchon gradient FD pin",
    )
    .unwrap_or_else(|e| panic!("{} failed: {:?}", "evaluator", e));

    let theta_dim = rho_dim + psi_dim;
    let theta_zero = Array1::<f64>::zeros(theta_dim);

    let eval_at =
        |theta: &Array1<f64>,
         order: gam_solve::rho_optimizer::OuterEvalOrder,
         cache: &mut SingleBlockExactJointDesignCache<'_>,
         evaluator: &mut gam_solve::estimate::ExternalJointHyperEvaluator<'_>| {
            cache.ensure_theta(theta).unwrap_or_else(|e| panic!("{} failed: {:?}", "ensure_theta", e));
            let hyper_dirs = try_build_spatial_log_kappa_hyper_dirs(
                data.view(),
                cache.spec(),
                cache.design(),
                &cache.spatial_terms,
            )
            .unwrap_or_else(|e| panic!("{} failed: {:?}", "hyper dirs build", e))
            .expect("hyper dirs present");
            evaluate_joint_reml_outer_eval_at_theta(
                evaluator,
                cache.design(),
                theta,
                rho_dim,
                hyper_dirs,
                None,
                order,
                None,
            )
            .unwrap_or_else(|e| panic!("{} failed: {:?}", "outer eval", e))
        };

    let (cost_at_zero, grad_at_zero, _hess) = eval_at(
        &theta_zero,
        gam_solve::rho_optimizer::OuterEvalOrder::ValueAndGradient,
        &mut cache,
        &mut evaluator,
    );

    let h = 1e-5_f64;
    let psi_idx = rho_dim;
    let mut theta_p = theta_zero.clone();
    theta_p[psi_idx] += h;
    let mut theta_m = theta_zero.clone();
    theta_m[psi_idx] -= h;
    let (cost_p, _, _) = eval_at(
        &theta_p,
        gam_solve::rho_optimizer::OuterEvalOrder::Value,
        &mut cache,
        &mut evaluator,
    );
    let (cost_m, _, _) = eval_at(
        &theta_m,
        gam_solve::rho_optimizer::OuterEvalOrder::Value,
        &mut cache,
        &mut evaluator,
    );
    let fd_psi_gradient = (cost_p - cost_m) / (2.0 * h);
    let analytic_psi_gradient = grad_at_zero[psi_idx];
    let scale = 1.0 + analytic_psi_gradient.abs().max(fd_psi_gradient.abs());
    let rel = (analytic_psi_gradient - fd_psi_gradient).abs() / scale;
    assert!(
        rel < 1e-3,
        "Duchon ψ outer gradient must match centered FD of the production objective: \
             analytic={:+.4e}, fd={:+.4e}, rel={:+.3e}",
        analytic_psi_gradient,
        fd_psi_gradient,
        rel
    );

    assert!(
        cost_at_zero.is_finite() && grad_at_zero.iter().all(|v| v.is_finite()),
        "ψ-gradient and cost must be finite at θ=0"
    );
}
#[test]
fn iso_kappa_duchon_dx_dpsi_matches_fd() {
    // Compare the production frozen-spec dX/dψ path against centered FD
    // of X(ψ+h) - X(ψ-h). This intentionally goes through
    // `try_build_spatial_term_log_kappa_derivative`: the formula layer owns
    // the frozen centers, length-scale compensation, and composed
    // identifiability transform.
    let n = 80usize;
    let mut data = Array2::<f64>::zeros((n, 1));
    for i in 0..n {
        data[[i, 0]] = i as f64 / (n as f64 - 1.0);
    }
    let spec_orig = TermCollectionSpec {
        linear_terms: vec![],
        random_effect_terms: vec![],
        smooth_terms: vec![SmoothTermSpec {
            name: "duchon_1d".to_string(),
            basis: SmoothBasisSpec::Duchon {
                feature_cols: vec![0],
                spec: DuchonBasisSpec {
                    radial_reparam: None,
                    periodic: None,
                    center_strategy: CenterStrategy::FarthestPoint { num_centers: 8 },
                    length_scale: Some(1.0),
                    power: 1.0,
                    nullspace_order: DuchonNullspaceOrder::Linear,
                    identifiability: SpatialIdentifiability::default(),
                    aniso_log_scales: None,
                    operator_penalties: DuchonOperatorPenaltySpec::default(),
                    boundary: OneDimensionalBoundary::Open,
                },
                input_scale: None,
            },
            shape: ShapeConstraint::None,
            joint_null_rotation: None,
        }],
    };
    let design = build_term_collection_design(data.view(), &spec_orig).unwrap_or_else(|e| panic!("{} failed: {:?}", "design", e));
    let frozen = freeze_term_collection_from_design(&spec_orig, &design).unwrap_or_else(|e| panic!("{} failed: {:?}", "freeze", e));

    let build_design_at = |psi: f64| -> Array2<f64> {
        // Rebuild design at psi via direct kernel build using frozen spec.
        let mut s = frozen.clone();
        if let SmoothBasisSpec::Duchon {
            spec: ref mut duchon,
            ..
        } = s.smooth_terms[0].basis
        {
            duchon.length_scale = Some((-psi).exp());
        }
        let d = build_term_collection_design(data.view(), &s).unwrap_or_else(|e| panic!("{} failed: {:?}", "rebuild", e));
        d.design.to_dense()
    };

    // Build derivative at psi=0.
    let psi_eval = 0.0_f64;
    let derivative_bundle =
        try_build_spatial_term_log_kappa_derivative(data.view(), &frozen, &design, 0)
            .unwrap_or_else(|e| panic!("{} failed: {:?}", "formula Duchon derivative should build", e))
            .expect("Duchon derivative should be available");
    let global_range = derivative_bundle.0;
    let p_total = derivative_bundle.1;
    let implicit_operator = derivative_bundle.8;
    let op = implicit_operator.unwrap_or_else(|| panic!("{} failed", "Duchon derivative should expose implicit operator"));
    let p = op.p_out();
    assert_eq!(p_total, design.design.ncols());
    assert_eq!(global_range.end - global_range.start, p);

    // FD reference.
    let h = 1e-4_f64;
    let x_plus = build_design_at(psi_eval + h);
    let x_minus = build_design_at(psi_eval - h);
    eprintln!(
        "[DXDPSI_FD] X(+h)[0,0..3]={:?} X(-h)[0,0..3]={:?}",
        x_plus.row(0).iter().take(3).copied().collect::<Vec<_>>(),
        x_minus.row(0).iter().take(3).copied().collect::<Vec<_>>(),
    );
    eprintln!(
        "[DXDPSI_FD] X(+h) shape={:?} X(-h) shape={:?} p_out={}",
        x_plus.shape(),
        x_minus.shape(),
        p,
    );
    // Also build at psi_eval to compare cols.
    let x_at = build_design_at(psi_eval);
    let orig_design = build_term_collection_design(data.view(), &spec_orig).unwrap_or_else(|e| panic!("{} failed: {:?}", "rebuild orig", e));
    eprintln!(
        "[DXDPSI_FD] X(psi_eval) shape={:?} orig_design.ncols={}",
        x_at.shape(),
        orig_design.design.ncols(),
    );

    // Multiply analytic operator by unit basis vectors.
    let mut analytic = Array2::<f64>::zeros((n, p));
    let mut basisv = Array1::<f64>::zeros(p);
    for j in 0..p {
        basisv[j] = 1.0;
        let col = op.forward_mul(0, &basisv.view()).unwrap_or_else(|e| panic!("{} failed: {:?}", "forward_mul", e));
        analytic.column_mut(j).assign(&col);
        basisv[j] = 0.0;
    }

    // Also check transpose_mul: X_tau^T v for v of length n.
    // FD reference: X_tau^T v should be (X(+h)^T - X(-h)^T)/(2h) · v.
    let smooth_start = global_range.start;
    let v_test = Array1::<f64>::from_shape_fn(n, |i| (i as f64 * 0.07).sin());
    let analytic_tv = op.transpose_mul(0, &v_test.view()).unwrap_or_else(|e| panic!("{} failed: {:?}", "transpose_mul", e));
    let fd_tv_full = (&x_plus.t() - &x_minus.t()) / (2.0 * h);
    let fd_tv = fd_tv_full.dot(&v_test);
    // Extract smooth portion only
    let fd_tv_smooth = fd_tv.slice(s![smooth_start..(smooth_start + p)]).to_owned();
    let max_tv_diff = analytic_tv
        .iter()
        .zip(fd_tv_smooth.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);
    let max_tv_abs = analytic_tv.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
    eprintln!(
        "[DXDPSI_TV] max|analytic_tv - fd_tv|={:.3e}  max|analytic_tv|={:.3e}",
        max_tv_diff, max_tv_abs
    );
    eprintln!(
        "[DXDPSI_TV] analytic_tv={:?}",
        analytic_tv.iter().take(p).copied().collect::<Vec<_>>()
    );
    eprintln!(
        "[DXDPSI_TV] fd_tv_smooth={:?}",
        fd_tv_smooth.iter().take(p).copied().collect::<Vec<_>>()
    );
    let fd_full = (&x_plus - &x_minus) / (2.0 * h);
    let fd = fd_full
        .slice(s![.., smooth_start..(smooth_start + p)])
        .to_owned();
    let mut max_diff = 0.0_f64;
    let mut max_abs = 0.0_f64;
    for i in 0..n {
        for j in 0..p {
            let d = (analytic[[i, j]] - fd[[i, j]]).abs();
            if d > max_diff {
                max_diff = d;
            }
            if analytic[[i, j]].abs() > max_abs {
                max_abs = analytic[[i, j]].abs();
            }
        }
    }
    eprintln!(
        "[DXDPSI_FD] max|analytic - fd|={:.3e}  max|analytic|={:.3e}",
        max_diff, max_abs
    );
    eprintln!(
        "[DXDPSI_FD] analytic[0,..]={:?}",
        analytic.row(0).iter().take(p).copied().collect::<Vec<_>>(),
    );
    eprintln!(
        "[DXDPSI_FD] fd[0,..]={:?}",
        fd.row(0).iter().take(p).copied().collect::<Vec<_>>(),
    );
    assert!(max_diff < 5e-3 * max_abs.max(1e-3), "dX/dψ mismatch");
}

/// #2454 MEASUREMENT (reports, never fails): put the MONOTONE fixture's own spec
/// through an evaluator whose gradient can be finite-difference-checked.
///
/// Everything claiming `∂V/∂ρ = −c·λ` on #2454 came from the standard-REML
/// certificate's printed probe gradients, which have never been FD-verified.
/// Every ladder that HAS been FD-verified runs the joint iso-κ evaluator and
/// saturates correctly. The two disagree, so the discriminator is to run the
/// SAME fixture — `length_scale = 12.0`, 12 centers, n=60, d=2,
/// `double_penalty: true`, `CenterSumToZero` — through the checkable evaluator.
///
///   `|g|` decaying, FD agreeing ⇒ the criterion is bounded for this fixture and
///        the standard-REML path's printed gradient is the defect;
///   `|g| ∝ e^ρ` reproduced AND FD-confirmed ⇒ the criterion really is unbounded
///        and both of my mechanism hypotheses were wrong.
///
/// `ĉ = −e^ρ·∂V/∂ρ` is reported alongside so the tail law is directly readable.
#[test]
fn zz_measure_monotone_fixture_through_checkable_evaluator_2454() {
    let n = 60usize;
    let d = 2usize;
    let mut data = Array2::<f64>::zeros((n, d));
    let mut y = Array1::<f64>::zeros(n);
    for i in 0..n {
        let x0 = i as f64 / (n as f64 - 1.0);
        let x1 = (i as f64 * 0.17).sin();
        data[[i, 0]] = x0;
        data[[i, 1]] = x1;
        y[i] = (3.0 * x0).cos() + 0.35 * x1;
    }
    let weights = Array1::ones(n);
    let offset = Array1::zeros(n);
    let spec = TermCollectionSpec {
        linear_terms: vec![],
        random_effect_terms: vec![],
        smooth_terms: vec![SmoothTermSpec {
            name: "matern".to_string(),
            basis: SmoothBasisSpec::Matern {
                feature_cols: vec![0, 1],
                spec: MaternBasisSpec {
                    periodic: None,
                    center_strategy: CenterStrategy::FarthestPoint { num_centers: 12 },
                    length_scale: gam_terms::basis::MaternLengthScale::fixed(12.0),
                    nu: MaternNu::FiveHalves,
                    include_intercept: false,
                    double_penalty: true,
                    identifiability: MaternIdentifiability::CenterSumToZero,
                    aniso_log_scales: None,
                },
                input_scale: None,
            },
            shape: ShapeConstraint::None,
            joint_null_rotation: None,
        }],
    };
    let fit_opts = FitOptions {
        compute_inference: false,
        max_iter: 200,
        tol: 1e-12,
        penalty_shrinkage_floor: None,
        ..FitOptions::default()
    };
    let family = LikelihoodSpec::gaussian_identity();

    let design = build_term_collection_design(data.view(), &spec)
        .unwrap_or_else(|e| panic!("design failed: {e:?}"));
    let frozen = freeze_term_collection_from_design(&spec, &design)
        .unwrap_or_else(|e| panic!("freeze failed: {e:?}"));
    let frozen_design = build_term_collection_design(data.view(), &frozen)
        .unwrap_or_else(|e| panic!("frozen design failed: {e:?}"));
    let spatial_terms = spatial_length_scale_term_indices(&frozen);
    let dims_per_term = spatial_dims_per_term(&frozen, &spatial_terms);
    let rho_dim = frozen_design.penalties.len();
    let psi_dim: usize = dims_per_term.iter().sum();
    eprintln!("[zz-mono-2454] rho_dim={rho_dim} psi_dim={psi_dim} p={}", frozen_design.design.ncols());

    let external_opts = external_opts_for_design(&family, &frozen_design, &fit_opts);
    let mut cache = SingleBlockExactJointDesignCache::new(
        data.view(),
        frozen.clone(),
        frozen_design.clone(),
        spatial_terms.clone(),
        rho_dim,
        dims_per_term.clone(),
    )
    .unwrap_or_else(|e| panic!("cache failed: {e:?}"));
    let mut evaluator = gam_solve::estimate::ExternalJointHyperEvaluator::new(
        y.view(),
        weights.view(),
        &frozen_design.design,
        offset.view(),
        &frozen_design.penalties,
        &external_opts,
        "#2454 monotone fixture evaluator",
    )
    .unwrap_or_else(|e| panic!("evaluator failed: {e:?}"));

    let cost_at = |theta: &Array1<f64>,
                   cache: &mut SingleBlockExactJointDesignCache<'_>,
                   evaluator: &mut gam_solve::estimate::ExternalJointHyperEvaluator<'_>|
     -> f64 {
        cache.ensure_theta(theta).unwrap_or_else(|e| panic!("ensure_theta: {e:?}"));
        let design = cache.design();
        evaluator
            .evaluate_cost_only(
                &design.design,
                &design.penalties,
                &design.nullspace_dims,
                design.linear_constraints.clone(),
                theta,
                rho_dim,
                None,
                "#2454 cost-only",
                None,
            )
            .unwrap_or_else(|e| panic!("cost-only: {e:?}"))
    };
    let analytic_at = |theta: &Array1<f64>,
                       cache: &mut SingleBlockExactJointDesignCache<'_>,
                       evaluator: &mut gam_solve::estimate::ExternalJointHyperEvaluator<'_>|
     -> (f64, Array1<f64>) {
        cache.ensure_theta(theta).expect("ensure_theta");
        let hyper_dirs = try_build_spatial_log_kappa_hyper_dirs(
            data.view(),
            cache.spec(),
            cache.design(),
            &cache.spatial_terms,
        )
        .unwrap_or_else(|e| panic!("hyper dirs: {e:?}"))
        .expect("hyper dirs present");
        let (cost, grad, _h) = evaluate_joint_reml_outer_eval_at_theta(
            evaluator,
            cache.design(),
            theta,
            rho_dim,
            hyper_dirs,
            None,
            gam_solve::rho_optimizer::OuterEvalOrder::ValueAndGradient,
            None,
        )
        .unwrap_or_else(|e| panic!("outer eval: {e:?}"));
        (cost, grad)
    };

    // #2454 STEP-LAW at the sign-disagreement point. At rho=15 the analytic and
    // FD gradients disagree by 166% AND on the SIGN (an=+6.499e-2, fd=-9.835e-2,
    // gap 0.163). A cost-noise floor of nu~1e-11 gives nu/h ~ 3e-8 at h=3e-4,
    // seven orders too small, so either the inner solve's noise explodes here or
    // truncation needs a third derivative of order 1e7. Sweeping h separates
    // them WITHOUT needing a higher-precision reference:
    //     gap proportional to 1/h  => NOISE (inner PIRLS not resolving at rho=15)
    //     gap proportional to h^2  => TRUNCATION (FD wrong; analytic stands)
    //     gap flat in h            => the ANALYTIC gradient is wrong
    {
        let value = 15.0_f64;
        let mut theta = Array1::<f64>::zeros(rho_dim + psi_dim);
        for j in 0..rho_dim {
            theta[j] = value;
        }
        let (cost, grad) = analytic_at(&theta, &mut cache, &mut evaluator);
        let an = grad[0];
        eprintln!("[zz-steplaw15-2454] rho=15 COST={cost:+.12e} analytic_rho0={an:+.8e}");
        for hh in [1e-2_f64, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6] {
            let mut plus = theta.clone();
            plus[0] += hh;
            let mut minus = theta.clone();
            minus[0] -= hh;
            let cp = cost_at(&plus, &mut cache, &mut evaluator);
            let cm = cost_at(&minus, &mut cache, &mut evaluator);
            let fd = (cp - cm) / (2.0 * hh);
            let gap = an - fd;
            eprintln!(
                "[zz-steplaw15-2454] h={hh:.1e} fd={fd:+.8e} gap={gap:+.4e} \
                 gap_times_h={:.4e} gap_over_h2={:.4e}",
                gap * hh,
                gap / (hh * hh)
            );
        }
    }
    let h = 3e-4_f64;
    for value in [6.0_f64, 9.0, 12.0, 15.0, 18.0, 21.0] {
        let mut theta = Array1::<f64>::zeros(rho_dim + psi_dim);
        for j in 0..rho_dim {
            theta[j] = value;
        }
        let (cost, grad) = analytic_at(&theta, &mut cache, &mut evaluator);
        for j in 0..rho_dim {
            let mut plus = theta.clone();
            plus[j] += h;
            let mut minus = theta.clone();
            minus[j] -= h;
            let cp = cost_at(&plus, &mut cache, &mut evaluator);
            let cm = cost_at(&minus, &mut cache, &mut evaluator);
            let fd = (cp - cm) / (2.0 * h);
            let an = grad[j];
            let denom = fd.abs().max(an.abs()).max(1e-12);
            eprintln!(
                "[zz-mono-2454] rho={value:5.1} j={j} COST={cost:+.10e} an={an:+.6e} \
                 fd={fd:+.6e} rel={:.3e} chat_an={:+.6e}",
                (an - fd).abs() / denom,
                -value.exp() * an
            );
        }
    }
}

}
