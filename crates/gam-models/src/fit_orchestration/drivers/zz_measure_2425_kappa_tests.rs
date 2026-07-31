// #2425 Half-A instrumentation. These are MEASUREMENTS, not gates: each one
// prints numbers and asserts only that the measurement was taken, so a change
// in the numbers is never a red test. They exist because the two remaining
// Half-A mechanisms are both "is the analytic quantity wrong, or is the thing
// we are comparing it against out of resolution", and that question cannot be
// answered from a pass/fail line.
//
// Probe 1 (`..._fd_step_sweep_at_duchon_probit_n20`) answers it for the lone
// surviving `iso_kappa_*` red. `iso_kappa_duchon_n_smaller_fd` reports exactly
// one violation, on the trend-ridge ρ coordinate, whose analytic gradient is
// −1.9454e-5 against a finite difference of −3.2117e-5 at h = 1e-5. Both
// readings cannot be right. A central difference of a profiled REML has error
//
//     err(h) ≈ eps_f / h  +  (h²/6)·|V'''|
//
// with `eps_f` the absolute accuracy of the objective (here: an inner PIRLS
// solve, a pseudo-logdet, and a Cholesky, so `eps_f ≫ eps·|V|`). The two terms
// have opposite h-dependence, so sweeping h separates them:
//
//   * analytic wrong  ⇒ |analytic − fd(h)| tends to a NONZERO constant as h
//     falls, until the 1/h noise arm takes over;
//   * oracle floor    ⇒ |analytic − fd(h)| has a clear minimum and then GROWS
//     like 1/h, and the Richardson-extrapolated value moves toward the
//     analytic one rather than away from it.
//
// Probe 2 (`..._matern_monotone_rail_geometry`) answers a different question
// for the five refusing fixtures. Their refusal decomposes as: one ρ coordinate
// pinned on the ±12 joint box carrying essentially the whole gradient norm, two
// interior ρ coordinates within a factor of ~6 of the stationarity bound, and a
// ψ coordinate holding ~1.33. The last one is only diagnosable against the ψ
// box the optimizer was actually given — if ψ* sits on (or inside the rail
// margin of) its data-derived bound, a `railed` set that lists only ρ
// coordinates can never certify, and the fit refuses forever. So the probe
// prints the realized box next to the realized checkpoint.
#[cfg(test)]
mod zz_measure_2425_kappa_tests {
    use super::*;
    use super::test_support::SingleBlockExactJointDesignCacheTestExt;
    use gam_terms::basis::{
        DuchonBasisSpec, DuchonNullspaceOrder, DuchonOperatorPenaltySpec, MaternBasisSpec,
        MaternNu, OneDimensionalBoundary, SpatialIdentifiability,
    };
    use ndarray::{Array1, Array2};

    /// The `duchon_probit_n20` fixture of `iso_kappa_fd_variant_driver`,
    /// reproduced verbatim (n = 20, `well_conditioned` labels, hybrid Duchon
    /// with the default operator-penalty set). Kept as its own function rather
    /// than reached through the driver so the probe can evaluate at arbitrary
    /// θ and h without perturbing the gate.
    fn duchon_probit_n20_fixture() -> (
        Array2<f64>,
        Array1<f64>,
        Array1<f64>,
        Array1<f64>,
        TermCollectionSpec,
        FitOptions,
    ) {
        let n = 20usize;
        let mut data = Array2::<f64>::zeros((n, 1));
        let mut y = Array1::<f64>::zeros(n);
        for i in 0..n {
            let t = i as f64 / (n as f64 - 1.0);
            data[[i, 0]] = t;
            // Balanced Bernoulli labels: a deterministic logistic-probability
            // threshold against a fixed phase grid keeps μ away from {0, 1}.
            let p = 1.0 / (1.0 + (-0.6 * (2.0 * std::f64::consts::PI * t).sin()).exp());
            let u = 0.5 * ((5.0 * (i as f64) + 0.5).sin() + 1.0);
            y[i] = if u < p { 1.0 } else { 0.0 };
        }
        let spec = TermCollectionSpec {
            linear_terms: vec![],
            random_effect_terms: vec![],
            smooth_terms: vec![SmoothTermSpec {
                name: "variant_1d".to_string(),
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
        let fit_opts = FitOptions {
            compute_inference: false,
            max_iter: 200,
            tol: 1e-12,
            ..FitOptions::default()
        };
        (
            data,
            y,
            Array1::ones(n),
            Array1::zeros(n),
            spec,
            fit_opts,
        )
    }

    #[test]
    fn zz_measure_2425_fd_step_sweep_at_duchon_probit_n20() {
        let (data, y, weights, offset, spec, fit_opts) = duchon_probit_n20_fixture();
        let design = build_term_collection_design(data.view(), &spec).expect("design");
        let frozen = freeze_term_collection_from_design(&spec, &design).expect("freeze");
        let frozen_design =
            build_term_collection_design(data.view(), &frozen).expect("frozen design");
        let spatial_terms = spatial_length_scale_term_indices(&frozen);
        let dims_per_term = spatial_dims_per_term(&frozen, &spatial_terms);
        let rho_dim = frozen_design.penalties.len();
        let psi_dim: usize = dims_per_term.iter().sum();
        let theta_dim = rho_dim + psi_dim;
        let family = LikelihoodSpec::binomial_probit();
        let external_opts = external_opts_for_design(&family, &frozen_design, &fit_opts);
        let mut cache = SingleBlockExactJointDesignCache::new(
            data.view(),
            frozen.clone(),
            frozen_design.clone(),
            spatial_terms.clone(),
            rho_dim,
            dims_per_term.clone(),
        )
        .expect("single-block cache");
        let mut evaluator = gam_solve::estimate::ExternalJointHyperEvaluator::new(
            y.view(),
            weights.view(),
            &frozen_design.design,
            offset.view(),
            &frozen_design.penalties,
            &external_opts,
            "2425 FD step-sweep evaluator",
        )
        .expect("evaluator");

        let cost_at = |theta: &Array1<f64>,
                           cache: &mut SingleBlockExactJointDesignCache<'_>,
                           evaluator: &mut gam_solve::estimate::ExternalJointHyperEvaluator<
            '_,
        >|
         -> f64 {
            cache.ensure_theta(theta).expect("ensure_theta");
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
                    "2425 sweep cost-only",
                    None,
                )
                .expect("cost-only eval")
        };

        // The `psi_only` probe of the gate: ρ = 0, ψ = 0.4. This is the probe
        // that violates; `zero` and `base` (both ψ = 0) agree to ~1e-10.
        let mut theta = Array1::<f64>::zeros(theta_dim);
        for k in 0..psi_dim {
            theta[rho_dim + k] = 0.4;
        }

        // (a) Is the objective a deterministic function of θ at all? If two
        //     evaluations at the SAME θ differ, every finite difference below
        //     inherits that difference divided by 2h and the sweep is
        //     meaningless. Measured before anything else for that reason.
        let c_first = cost_at(&theta, &mut cache, &mut evaluator);
        let c_second = cost_at(&theta, &mut cache, &mut evaluator);
        eprintln!(
            "[2425-SWEEP determinism] V(theta) call#1={c_first:+.17e} call#2={c_second:+.17e} \
             delta={:.3e}",
            (c_first - c_second).abs()
        );

        // (b) The analytic gradient the κ-optimizer follows.
        cache.ensure_theta(&theta).expect("ensure_theta");
        let hyper_dirs = try_build_spatial_log_kappa_hyper_dirs(
            data.view(),
            cache.spec(),
            cache.design(),
            &cache.spatial_terms,
        )
        .expect("hyper dirs build")
        .expect("hyper dirs present");
        let (cost_an, grad_an, _hess) = evaluate_joint_reml_outer_eval_at_theta(
            &mut evaluator,
            cache.design(),
            &theta,
            rho_dim,
            hyper_dirs,
            None,
            gam_solve::rho_optimizer::OuterEvalOrder::ValueAndGradient,
            None,
        )
        .expect("outer eval");
        eprintln!(
            "[2425-SWEEP] rho_dim={rho_dim} psi_dim={psi_dim} V={cost_an:+.10e} \
             analytic={:?}",
            grad_an.iter().map(|v| format!("{v:+.6e}")).collect::<Vec<_>>()
        );

        // (c) The sweep. `h` spans three decades either side of the gate's
        //     1e-5 so both arms of the error model are visible.
        const STEPS: [f64; 9] = [1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6];
        let mut fd_table: Vec<(f64, Vec<f64>)> = Vec::with_capacity(STEPS.len());
        for &h in STEPS.iter() {
            let mut row = Vec::with_capacity(theta_dim);
            for j in 0..theta_dim {
                let mut plus = theta.clone();
                plus[j] += h;
                let mut minus = theta.clone();
                minus[j] -= h;
                let cp = cost_at(&plus, &mut cache, &mut evaluator);
                let cm = cost_at(&minus, &mut cache, &mut evaluator);
                row.push((cp - cm) / (2.0 * h));
            }
            fd_table.push((h, row));
        }

        for j in 0..theta_dim {
            let kind = if j >= rho_dim { "psi" } else { "rho" };
            eprintln!("[2425-SWEEP] --- coordinate {kind} j={j} analytic={:+.8e}", grad_an[j]);
            for (h, row) in fd_table.iter() {
                eprintln!(
                    "[2425-SWEEP]   h={h:.1e} fd={:+.8e} |an-fd|={:.3e}",
                    row[j],
                    (grad_an[j] - row[j]).abs()
                );
            }
            // Richardson: with fd(h) = V' + C·h² + eps_f/h, the pair (h, 2h)
            // cancels the h² arm exactly. Where the h² arm dominates this lands
            // on V'; where the noise arm dominates it amplifies the noise, which
            // is itself the signature we are looking for.
            for w in fd_table.windows(2) {
                let (h_small, row_small) = &w[1];
                let (h_big, row_big) = &w[0];
                // `STEPS` descends, so `w[1]` is the smaller step. Richardson
                // is only exact for an exact halving; the 1e/3e ladder gives
                // ratios of 10/3 and 3, so use the general two-point form
                // fd_R = (r²·fd(h) − fd(rh)) / (r² − 1) with r = h_big/h_small.
                let r = h_big / h_small;
                let r2 = r * r;
                let fd_r = (r2 * row_small[j] - row_big[j]) / (r2 - 1.0);
                eprintln!(
                    "[2425-SWEEP]   richardson(h={h_small:.1e},{h_big:.1e}) fd_R={fd_r:+.8e} \
                     |an-fd_R|={:.3e}",
                    (grad_an[j] - fd_r).abs()
                );
            }
        }

        assert!(
            cost_an.is_finite() && grad_an.iter().all(|v| v.is_finite()),
            "the sweep must have something finite to report"
        );
    }

    #[test]
    fn zz_measure_2425_matern_monotone_rail_geometry() {
        // Verbatim fixture of
        // `spatial_length_scale_optimization_monotone_improves_or_keeps_score_for_matern`.
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
            max_iter: 40,
            ..FitOptions::default()
        };
        let kappa_options = SpatialLengthScaleOptimizationOptions {
            max_outer_iter: 16,
            rel_tol: 1e-5,
            pilot_subsample_threshold: 0,
            ..SpatialLengthScaleOptimizationOptions::default()
        };

        // Reproduce the driver's pre-optimization pipeline: baseline fit at the
        // seeded geometry, freeze, then the certified two-basin range choice.
        // Everything the optimizer sees downstream is a function of these.
        let baseline_options = superseded_fit_options(&fit_opts);
        let best = fit_term_collection_forspec(
            data.view(),
            y.view(),
            weights.view(),
            offset.view(),
            &spec,
            LikelihoodSpec::gaussian_identity(),
            &baseline_options,
        )
        .expect("baseline fit");
        let resolved = freeze_term_collection_from_design(&spec, &best.design).expect("freeze");
        let spatial_terms = spatial_length_scale_term_indices(&resolved);
        let basin = select_isotropic_matern_range_basin(
            data.view(),
            y.view(),
            weights.view(),
            offset.view(),
            resolved,
            best,
            &LikelihoodSpec::gaussian_identity(),
            &baseline_options,
            &kappa_options,
            &spatial_terms,
        );
        // A measurement probe must RECORD what it saw, never die on it. Under
        // the #2450 criterion fix this fixture's basin selection can refuse —
        // `RhoPrior::default()` is no longer a soft box holding rho at 12, so
        // the outer search runs out toward the infinite-smoothing face and the
        // certificate correctly declines. That refusal text IS the measurement
        // this probe exists to capture (it carries the full c-hat ladder), so
        // print it and return instead of turning the probe into a red test.
        let (resolved, best) = match basin {
            Ok(pair) => pair,
            Err(error) => {
                eprintln!("[2425-RAIL] basin selection REFUSED: {error}");
                return;
            }
        };

        // The box the joint optimizer is actually handed. `JOINT_RHO_BOUND` is
        // the private ±12 constant; ψ's window is data-derived per term.
        let (psi_lower, psi_upper) =
            spatial_term_psi_bounds(data.view(), &resolved, 0, &kappa_options)
                .expect("psi bounds");
        let seed_length_scale = get_spatial_length_scale(&resolved, 0).expect("resolved range");
        eprintln!(
            "[2425-RAIL] post-basin length_scale={seed_length_scale:.8} \
             psi_seed={:+.8} psi_box=[{psi_lower:+.8}, {psi_upper:+.8}] \
             rho_box=[-12, +12] lambdas={:?}",
            -seed_length_scale.ln(),
            best.fit.lambdas.iter().map(|v| format!("{v:.6e}")).collect::<Vec<_>>()
        );

        // The checkpoint the refusal reports, verbatim from the panic message.
        // θ = [ρ_mass, ρ_tension, ρ_stiffness, ψ].
        let checkpoint = Array1::from(vec![
            4.583_475_744_619_827_f64,
            -0.011_392_023_106_851_933,
            12.0,
            -0.011_839_795_303_285_494,
        ]);
        eprintln!(
            "[2425-RAIL] checkpoint={:?}",
            checkpoint.iter().map(|v| format!("{v:+.10e}")).collect::<Vec<_>>()
        );
        eprintln!(
            "[2425-RAIL] psi* distance to box: to_lower={:+.6e} to_upper={:+.6e}",
            checkpoint[3] - psi_lower,
            psi_upper - checkpoint[3]
        );

        // Is the gradient the optimizer stopped on the RIGHT gradient? Compare
        // analytic against a central difference of the production objective at
        // the checkpoint itself. If they agree, the search halted at a point it
        // correctly knows is not stationary, and the defect is in the stopping
        // or certification rule rather than in any derivative.
        let frozen_design =
            build_term_collection_design(data.view(), &resolved).expect("frozen design");
        let dims_per_term = spatial_dims_per_term(&resolved, &spatial_terms);
        let rho_dim = frozen_design.penalties.len();
        let psi_dim: usize = dims_per_term.iter().sum();
        let theta_dim = rho_dim + psi_dim;
        eprintln!("[2425-RAIL] rho_dim={rho_dim} psi_dim={psi_dim}");
        if theta_dim != checkpoint.len() {
            eprintln!(
                "[2425-RAIL] SKIP gradient probe: reproduced theta_dim={theta_dim} but the \
                 reported checkpoint has {} entries — the freeze differs from the run that \
                 produced it, so the two are not comparable",
                checkpoint.len()
            );
            return;
        }

        let family = LikelihoodSpec::gaussian_identity();
        let external_opts = external_opts_for_design(&family, &frozen_design, &fit_opts);
        let mut cache = SingleBlockExactJointDesignCache::new(
            data.view(),
            resolved.clone(),
            frozen_design.clone(),
            spatial_terms.clone(),
            rho_dim,
            dims_per_term.clone(),
        )
        .expect("single-block cache");
        let mut evaluator = gam_solve::estimate::ExternalJointHyperEvaluator::new(
            y.view(),
            weights.view(),
            &frozen_design.design,
            offset.view(),
            &frozen_design.penalties,
            &external_opts,
            "2425 rail-geometry evaluator",
        )
        .expect("evaluator");

        let cost_at = |theta: &Array1<f64>,
                           cache: &mut SingleBlockExactJointDesignCache<'_>,
                           evaluator: &mut gam_solve::estimate::ExternalJointHyperEvaluator<
            '_,
        >|
         -> f64 {
            cache.ensure_theta(theta).expect("ensure_theta");
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
                    "2425 rail cost-only",
                    None,
                )
                .expect("cost-only eval")
        };

        cache.ensure_theta(&checkpoint).expect("ensure_theta");
        let hyper_dirs = try_build_spatial_log_kappa_hyper_dirs(
            data.view(),
            cache.spec(),
            cache.design(),
            &cache.spatial_terms,
        )
        .expect("hyper dirs build")
        .expect("hyper dirs present");
        let (cost_an, grad_an, _hess) = evaluate_joint_reml_outer_eval_at_theta(
            &mut evaluator,
            cache.design(),
            &checkpoint,
            rho_dim,
            hyper_dirs,
            None,
            gam_solve::rho_optimizer::OuterEvalOrder::ValueAndGradient,
            None,
        )
        .expect("outer eval");

        let h = 1e-5_f64;
        for j in 0..theta_dim {
            let mut plus = checkpoint.clone();
            plus[j] += h;
            let mut minus = checkpoint.clone();
            minus[j] -= h;
            let cp = cost_at(&plus, &mut cache, &mut evaluator);
            let cm = cost_at(&minus, &mut cache, &mut evaluator);
            let fd = (cp - cm) / (2.0 * h);
            let kind = if j >= rho_dim { "psi" } else { "rho" };
            eprintln!(
                "[2425-RAIL] V={cost_an:+.8e} {kind} j={j} analytic={:+.8e} fd={fd:+.8e} \
                 |an-fd|={:.3e}",
                grad_an[j],
                (grad_an[j] - fd).abs()
            );
        }

        assert!(
            cost_an.is_finite() && grad_an.iter().all(|v| v.is_finite()),
            "the rail-geometry probe must have something finite to report"
        );
    }
}
