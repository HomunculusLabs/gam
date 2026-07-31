// Diagnostic probes for gam#2647 (binomial location-scale wiggle + Matérn
// spatial: β diverges monotonically while the objective descends every cycle).
//
// Every test here is `#[ignore]`d: these are measurement instruments, not
// gates. They are run explicitly with `-- --ignored --nocapture` and their
// output is the evidence recorded on the issue.
#![cfg(test)]

use super::*;

/// The exact fixture geometry from
/// `binomial_location_scalewiggle_termswith_matern_spatial_blocks_fit_finitely`,
/// factored so every probe below measures the SAME data.
fn fixture_2647() -> (Array2<f64>, Array1<f64>, Array1<f64>, Array1<f64>) {
    let n = 30usize;
    let mut data = Array2::<f64>::zeros((n, 2));
    for i in 0..n {
        let t = i as f64 / (n as f64 - 1.0);
        data[[i, 0]] = t;
        data[[i, 1]] = (2.5 * std::f64::consts::PI * t).sin();
    }
    let y = Array1::from_iter((0..n).map(|i| if i % 4 == 0 || i % 9 == 0 { 1.0 } else { 0.0 }));
    let weights = Array1::from_elem(n, 1.0);
    let q_seed = Array1::linspace(-1.5, 1.5, n);
    (data, y, weights, q_seed)
}

fn run_fixture_with(options: &BlockwiseFitOptions) -> Result<BlockwiseTermFitResult, String> {
    let (data, y, weights, q_seed) = fixture_2647();
    let (wiggle_block, knots) =
        BinomialLocationScaleWiggleFamily::buildwiggle_block_input(q_seed.view(), 2, 4, 2, false)
            .expect("wiggle block");
    let spec = BinomialLocationScaleWiggleTermSpec {
        y,
        weights,
        link_kind: InverseLink::Standard(StandardLink::Probit),
        thresholdspec: simple_matern_term_collection(&[0, 1], 0.45),
        log_sigmaspec: empty_term_collection(),
        threshold_offset: Array1::zeros(30),
        log_sigma_offset: Array1::zeros(30),
        wiggle_knots: knots,
        wiggle_degree: 2,
        wiggle_block,
    };
    fit_binomial_location_scalewiggle_terms(data.view(), spec, options, &spatial_kappa_options())
}

/// Probe 1 — does the refusal survive a much larger inner-cycle budget?
///
/// The refusal message reports `48 cycle(s)`, which is exactly the fixture's
/// `inner_max_cycles`. The derivative path in `psi_hyper` tightens `inner_tol`
/// from `1e-4` to `1e-11` and raises `inner_max_cycles` to `max(200)` — but
/// `fit_custom_family` stores `options.inner_max_cycles` into the shared
/// `outer_inner_max_iterations` atomic, and `capped_inner_max_cycles` mins
/// against it, so the raise never takes effect. If the divergence is really a
/// budget artefact this arm converges; if it is a runaway it does not.
#[test]
#[ignore]
fn probe_2647_budget_ladder() {
    gam_runtime::test_support::install_diagnostic_logger();
    for cycles in [48usize, 200, 600] {
        let options = BlockwiseFitOptions {
            inner_max_cycles: cycles,
            inner_tol: 1e-4,
            outer_max_iter: 3,
            outer_tol: 1e-4,
            ..BlockwiseFitOptions::default()
        };
        let started = std::time::Instant::now();
        let outcome = run_fixture_with(&options);
        let elapsed = started.elapsed().as_secs_f64();
        match outcome {
            Ok(fit) => println!(
                "[2647-budget] inner_max_cycles={cycles} OK in {elapsed:.1}s objective={:?}",
                fit.fit.penalized_objective()
            ),
            Err(error) => println!(
                "[2647-budget] inner_max_cycles={cycles} ERR in {elapsed:.1}s: {}",
                error.replace('\n', " ")
            ),
        }
    }
}

/// Probe 2 — how aliased are the threshold block and the warp block?
///
/// The BLS-wiggle model is `q = q0 + Σ_j β_w[j]·B_j(q0)` with
/// `q0 = −η_t·exp(−η_ls)`, so BOTH the threshold design `X_t` and the warp
/// basis `B(q0)` enter `q` as functions of the same rows. `fit_binomial_mean_wiggle`
/// removes exactly this alias in observation space (#1596): it fits the warp
/// through `B⊥ = (I − P_X)B`. This route does not. Measure the alias directly:
/// the fraction of `‖B‖` that survives projection onto `span(X_t)`, plus the
/// principal angles between the two column spaces.
#[test]
#[ignore]
fn probe_2647_threshold_warp_alias() {
    use faer::Side;
    use gam_linalg::faer_ndarray::FaerEigh;

    let (data, _y, _weights, q_seed) = fixture_2647();
    let thresholdspec = simple_matern_term_collection(&[0, 1], 0.45);
    let design = build_term_collection_design(data.view(), &thresholdspec)
        .expect("threshold term-collection design");
    let x: Array2<f64> = design.design.to_dense();

    let (_wiggle_block, knots) =
        BinomialLocationScaleWiggleFamily::buildwiggle_block_input(q_seed.view(), 2, 4, 2, false)
            .expect("wiggle block");
    let b = crate::wiggle::monotone_wiggle_basis_from_knots(q_seed.view(), &knots, 2)
        .expect("wiggle basis at the seed index");

    println!(
        "[2647-alias] X_t is {}x{}, B(q0) is {}x{} (n={})",
        x.nrows(),
        x.ncols(),
        b.nrows(),
        b.ncols(),
        x.nrows()
    );

    // Residualize B against span(X) exactly as `build_dealiased` does.
    let xtx = x.t().dot(&x);
    let xtb = x.t().dot(&b);
    let (evals, evecs) = xtx.eigh(Side::Lower).expect("XtX eigh");
    let max_eval = evals.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    let cutoff = 1.0e3 * f64::EPSILON * (xtx.nrows().max(1) as f64) * max_eval.max(1.0);
    let mut alias = Array2::<f64>::zeros((x.ncols(), b.ncols()));
    for k in 0..evals.len() {
        let lam = evals[k];
        if !lam.is_finite() || lam.abs() <= cutoff {
            continue;
        }
        let uk = evecs.column(k);
        let uk_xtb = uk.t().dot(&xtb);
        for i in 0..alias.nrows() {
            for j in 0..alias.ncols() {
                alias[[i, j]] += uk[i] * uk_xtb[j] / lam;
            }
        }
    }
    let bda = &b - &x.dot(&alias);
    for j in 0..b.ncols() {
        let full = b.column(j).iter().map(|v| v * v).sum::<f64>().sqrt();
        let resid = bda.column(j).iter().map(|v| v * v).sum::<f64>().sqrt();
        println!(
            "[2647-alias] warp column {j}: ‖b_j‖={full:.6e} ‖(I−P_X)b_j‖={resid:.6e} \
             retained={:.4}%",
            100.0 * resid / full.max(f64::MIN_POSITIVE)
        );
    }

    // Joint conditioning: the singular values of the STACKED [X | B] design are
    // what the inner joint Newton actually inverts.
    let mut stacked = Array2::<f64>::zeros((x.nrows(), x.ncols() + b.ncols()));
    for i in 0..x.nrows() {
        for j in 0..x.ncols() {
            stacked[[i, j]] = x[[i, j]];
        }
        for j in 0..b.ncols() {
            stacked[[i, x.ncols() + j]] = b[[i, j]];
        }
    }
    let gram = stacked.t().dot(&stacked);
    let (gevals, _) = gram.eigh(Side::Lower).expect("stacked Gram eigh");
    let lo = gevals.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = gevals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    println!(
        "[2647-alias] stacked [X|B] Gram spectrum: min={lo:.6e} max={hi:.6e} cond={:.6e}",
        hi / lo.max(f64::MIN_POSITIVE)
    );
    let gram_x = x.t().dot(&x);
    let (xevals, _) = gram_x.eigh(Side::Lower).expect("X Gram eigh");
    println!(
        "[2647-alias] X_t alone Gram spectrum: min={:.6e} max={:.6e}",
        xevals.iter().copied().fold(f64::INFINITY, f64::min),
        xevals.iter().copied().fold(f64::NEG_INFINITY, f64::max),
    );
    let gram_b = b.t().dot(&b);
    let (bevals, _) = gram_b.eigh(Side::Lower).expect("B Gram eigh");
    println!(
        "[2647-alias] B alone Gram spectrum: min={:.6e} max={:.6e}",
        bevals.iter().copied().fold(f64::INFINITY, f64::min),
        bevals.iter().copied().fold(f64::NEG_INFINITY, f64::max),
    );
}

/// Probe 3 — where does the joint penalized Hessian's smallest eigenvector live?
///
/// The alias hypothesis says the near-null direction of `H_L + S` straddles the
/// threshold and wiggle blocks (β_t ↓ traded against β_w ↑). A block-local
/// weakness would put the eigenvector inside ONE block. Report the block-wise
/// mass of the bottom eigenvectors of the exact joint penalized Hessian at
/// several `(β_t, β_w)` states, including the seed the fit starts from.
#[test]
#[ignore]
fn probe_2647_joint_hessian_null_direction() {
    use faer::Side;
    use gam_linalg::faer_ndarray::FaerEigh;

    let (data, y, weights, q_seed) = fixture_2647();
    let n = y.len();
    let thresholdspec = simple_matern_term_collection(&[0, 1], 0.45);
    let threshold_design = build_term_collection_design(data.view(), &thresholdspec)
        .expect("threshold design")
        .design;
    let log_sigma_design = DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
        Array2::<f64>::zeros((n, 0)),
    ));
    let (wiggle_block, knots) =
        BinomialLocationScaleWiggleFamily::buildwiggle_block_input(q_seed.view(), 2, 4, 2, false)
            .expect("wiggle block");

    let p_t = threshold_design.ncols();
    let p_w = wiggle_block.design.ncols();
    println!("[2647-hess] p_threshold={p_t} p_log_sigma=0 p_wiggle={p_w} n={n}");

    let family = BinomialLocationScaleWiggleFamily {
        y,
        weights,
        link_kind: InverseLink::Standard(StandardLink::Probit),
        threshold_design: Some(threshold_design.clone()),
        log_sigma_design: Some(log_sigma_design.clone()),
        wiggle_knots: knots,
        wiggle_degree: 2,
        policy: gam_runtime::resource::ResourcePolicy::default_library(),
    };

    let specs = vec![
        ParameterBlockSpec {
            name: "threshold".to_string(),
            design: threshold_design.clone(),
            offset: Array1::zeros(n),
            penalties: Vec::new(),
            nullspace_dims: Vec::new(),
            initial_log_lambdas: Array1::zeros(0),
            initial_beta: None,
            gauge_priority: 80,
            ..ParameterBlockSpec::defaults()
        },
        ParameterBlockSpec {
            name: "log_sigma".to_string(),
            design: log_sigma_design.clone(),
            offset: Array1::zeros(n),
            penalties: Vec::new(),
            nullspace_dims: Vec::new(),
            initial_log_lambdas: Array1::zeros(0),
            initial_beta: None,
            gauge_priority: 80,
            ..ParameterBlockSpec::defaults()
        },
        ParameterBlockSpec {
            name: "wiggle".to_string(),
            design: wiggle_block.design.clone(),
            offset: Array1::zeros(n),
            penalties: Vec::new(),
            nullspace_dims: Vec::new(),
            initial_log_lambdas: Array1::zeros(0),
            initial_beta: None,
            gauge_priority: 100,
            ..ParameterBlockSpec::defaults()
        },
    ];

    for (label, scale_t, scale_w) in [
        ("seed (β=0)", 0.0_f64, 0.0_f64),
        ("mild", 0.5, 0.2),
        ("late (|β|~10)", 5.0, 2.0),
    ] {
        let beta_t = Array1::from_shape_fn(p_t, |j| scale_t / ((j + 1) as f64));
        let beta_ls = Array1::<f64>::zeros(0);
        let beta_w = Array1::from_elem(p_w, scale_w);
        let states = vec![
            ParameterBlockState {
                eta: threshold_design.matrixvectormultiply(&beta_t),
                beta: beta_t,
            },
            ParameterBlockState {
                eta: Array1::zeros(n),
                beta: beta_ls,
            },
            ParameterBlockState {
                eta: {
                    // η_w must be the DYNAMIC design at the current q0, exactly
                    // as `refresh_all_block_etas` builds it.
                    let (x, off) = family
                        .block_geometry(
                            &[
                                ParameterBlockState {
                                    eta: threshold_design.matrixvectormultiply(
                                        &Array1::from_shape_fn(p_t, |j| scale_t / ((j + 1) as f64)),
                                    ),
                                    beta: Array1::from_shape_fn(p_t, |j| {
                                        scale_t / ((j + 1) as f64)
                                    }),
                                },
                                ParameterBlockState {
                                    eta: Array1::zeros(n),
                                    beta: Array1::zeros(0),
                                },
                                ParameterBlockState {
                                    eta: Array1::zeros(n),
                                    beta: Array1::from_elem(p_w, scale_w),
                                },
                            ],
                            &specs[2],
                        )
                        .expect("dynamic wiggle geometry");
                    x.matrixvectormultiply(&beta_w) + off
                },
                beta: beta_w,
            },
        ];

        let h = family
            .exact_newton_joint_hessian_with_specs(&states, &specs)
            .expect("joint hessian")
            .expect("joint hessian available");
        let (evals, evecs) = h.eigh(Side::Lower).expect("joint hessian eigh");
        let lo = evals.iter().copied().fold(f64::INFINITY, f64::min);
        let hi = evals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        println!(
            "[2647-hess] {label}: eig(H_L) min={lo:.6e} max={hi:.6e} cond={:.6e}",
            hi / lo.abs().max(f64::MIN_POSITIVE)
        );
        // Block mass of the two smallest-|λ| eigenvectors.
        let mut order: Vec<usize> = (0..evals.len()).collect();
        order.sort_by(|&a, &b| evals[a].abs().partial_cmp(&evals[b].abs()).unwrap());
        for &k in order.iter().take(2) {
            let v = evecs.column(k);
            let mass_t: f64 = (0..p_t).map(|i| v[i] * v[i]).sum();
            let mass_w: f64 = (p_t..p_t + p_w).map(|i| v[i] * v[i]).sum();
            println!(
                "[2647-hess] {label}: λ={:.6e} block mass threshold={:.4} wiggle={:.4}",
                evals[k], mass_t, mass_w
            );
        }
    }
}

/// Probe 4 — the scale gauge, demonstrated fit-free.
///
/// The model is `q = q0 + Σ_j β_w[j]·B_j(q0)` with `q0 = −η_t·exp(−η_ls)`. If the
/// warp span contains the linear function `u ↦ u − left`, then for any `s > 0`
///
///     (β_t, β_w) ↦ (β_t / s, β_w + (s−1)·ℓ)      where  B·ℓ = (u − left)
///
/// reproduces the SAME `q` on every row whose `q0` stays inside the knot hull:
/// `q0/s + (s−1)(q0/s − left) + w(q0/s) = q0 − (s−1)·left + w(q0/s)`. That is a
/// one-parameter gauge orbit of the LIKELIHOOD. It is not a gauge orbit of the
/// PENALTY: the threshold block is penalized, so `½βᵀSβ ∝ 1/s²` along it, while
/// the warp's linear direction `ℓ` lies in the null space of the order-2
/// roughness penalty (`double_penalty = false`) and costs nothing. So the
/// penalized objective DECREASES monotonically along the orbit and its infimum
/// sits at `s = ∞` — there is no minimiser, and no inner solve can converge.
///
/// This probe measures each link of that chain separately:
///   1. is `u ↦ u − left` in the warp span (least-squares residual of `Bℓ`)?
///   2. does `−ℓ(q)` stay flat along the orbit?
///   3. does `½βᵀSβ` fall like `1/s²`?
#[test]
#[ignore]
fn probe_2647_scale_gauge_orbit() {
    use faer::Side;
    use gam_linalg::faer_ndarray::FaerEigh;

    let (data, y, weights, q_seed) = fixture_2647();
    let n = y.len();
    let thresholdspec = simple_matern_term_collection(&[0, 1], 0.45);
    let threshold_collection =
        build_term_collection_design(data.view(), &thresholdspec).expect("threshold design");
    let threshold_design = threshold_collection.design.clone();
    let x_t: Array2<f64> = threshold_design.to_dense();
    let (wiggle_block, knots) =
        BinomialLocationScaleWiggleFamily::buildwiggle_block_input(q_seed.view(), 2, 4, 2, false)
            .expect("wiggle block");
    let p_t = x_t.ncols();
    let p_w = wiggle_block.design.ncols();

    // ---- link 1: is the linear function in the warp span? ----
    let left = knots[2];
    let right = knots[knots.len() - 3];
    println!("[2647-gauge] knot hull = [{left:.6}, {right:.6}], p_threshold={p_t}, p_wiggle={p_w}");
    let grid = Array1::linspace(left, right, 401);
    let b_grid = crate::wiggle::monotone_wiggle_basis_from_knots(grid.view(), &knots, 2)
        .expect("warp basis on the hull grid");
    let target = grid.mapv(|u| u - left);
    // Normal equations for min ‖B ℓ − (u − left)‖.
    let gram = b_grid.t().dot(&b_grid);
    let rhs = b_grid.t().dot(&target);
    let (gevals, gevecs) = gram.eigh(Side::Lower).expect("warp Gram eigh");
    let gmax = gevals.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    let mut ell = Array1::<f64>::zeros(p_w);
    for k in 0..gevals.len() {
        if gevals[k] <= 1e-12 * gmax.max(1.0) {
            continue;
        }
        let uk = gevecs.column(k);
        let coeff = uk.dot(&rhs) / gevals[k];
        for i in 0..p_w {
            ell[i] += coeff * uk[i];
        }
    }
    let fit = b_grid.dot(&ell);
    let resid = fit
        .iter()
        .zip(target.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    let scale = target.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    println!(
        "[2647-gauge] linear-in-hull residual: max|Bℓ − (u−left)| = {resid:.6e} against \
         range {scale:.6e} (relative {:.3e}); ℓ = {:?}",
        resid / scale.max(f64::MIN_POSITIVE),
        ell.iter().map(|v| format!("{v:.4}")).collect::<Vec<_>>(),
    );
    println!(
        "[2647-gauge] ℓ componentwise sign: all_nonneg={}",
        ell.iter().all(|&v| v >= -1e-12)
    );

    // ---- links 2 and 3: walk the orbit ----
    let family = BinomialLocationScaleWiggleFamily {
        y: y.clone(),
        weights: weights.clone(),
        link_kind: InverseLink::Standard(StandardLink::Probit),
        threshold_design: Some(threshold_design.clone()),
        log_sigma_design: Some(DesignMatrix::Dense(
            gam_linalg::matrix::DenseDesignMatrix::from(Array2::<f64>::zeros((n, 0))),
        )),
        wiggle_knots: knots.clone(),
        wiggle_degree: 2,
        policy: gam_runtime::resource::ResourcePolicy::default_library(),
    };
    // The real threshold penalty at ρ = 0 (λ = 1), i.e. the fixture's own seed.
    let s_t: Array2<f64> = threshold_collection
        .penalties_as_penalty_matrix()
        .first()
        .map(|p| p.to_dense())
        .unwrap_or_else(|| Array2::zeros((p_t, p_t)));
    let s_w: Array2<f64> = match wiggle_block.penalties.first() {
        Some(gam_terms::penalty_spec::PenaltySpec::Dense(m)) => m.clone(),
        Some(gam_terms::penalty_spec::PenaltySpec::DenseWithMean { matrix, .. }) => matrix.clone(),
        _ => Array2::zeros((p_w, p_w)),
    };
    println!(
        "[2647-gauge] ℓᵀ S_w ℓ = {:.6e}  (0 ⇒ the linear warp is free under the roughness penalty)",
        ell.dot(&s_w.dot(&ell))
    );

    // A non-degenerate starting index: spread q0 across the hull.
    let mut beta_t0 = Array1::<f64>::zeros(p_t);
    {
        // Least-squares β_t so that −X_t β_t ≈ q_seed (the index the knots were
        // planned for). Anything that spreads q0 over the hull works; using the
        // seed index makes the starting point the one the basis was built for.
        let g = x_t.t().dot(&x_t);
        let r = x_t.t().dot(&q_seed.mapv(|v| -v));
        let (ev, evec) = g.eigh(Side::Lower).expect("X_t Gram eigh");
        let emax = ev.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        for k in 0..ev.len() {
            if ev[k] <= 1e-12 * emax.max(1.0) {
                continue;
            }
            let uk = evec.column(k);
            let c = uk.dot(&r) / ev[k];
            for i in 0..p_t {
                beta_t0[i] += c * uk[i];
            }
        }
    }

    // ---- link 2/3, MEASURED rather than guessed ----
    //
    // A first attempt walked the ANALYTIC orbit `(β_t/s, β_w + (s−1)ℓ)` and it
    // was refuted: the warp's linear element is `a·(u − left)`, so it adds the
    // constant `−a·left` that neither a centred Matérn threshold nor the
    // log-σ block can absorb, and `−loglik` exploded (2.3e1 → 2.2e7 over
    // s = 1 → 1024) instead of staying flat. The naive scale orbit is NOT the
    // flat direction. So take the direction from the problem instead of from
    // algebra: the smallest-curvature eigenvector of the exact joint penalized
    // Hessian at a realistic state, and walk THAT.
    let s_w_pad = {
        let mut m = Array2::<f64>::zeros((p_t + p_w, p_t + p_w));
        for i in 0..p_t {
            for j in 0..p_t {
                m[[i, j]] = s_t[[i, j]];
            }
        }
        for i in 0..p_w {
            for j in 0..p_w {
                m[[p_t + i, p_t + j]] = s_w[[i, j]];
            }
        }
        m
    };
    let specs = vec![
        ParameterBlockSpec {
            name: "threshold".to_string(),
            design: threshold_design.clone(),
            offset: Array1::zeros(n),
            ..ParameterBlockSpec::defaults()
        },
        ParameterBlockSpec {
            name: "log_sigma".to_string(),
            design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
                Array2::<f64>::zeros((n, 0)),
            )),
            offset: Array1::zeros(n),
            ..ParameterBlockSpec::defaults()
        },
        ParameterBlockSpec {
            name: "wiggle".to_string(),
            design: wiggle_block.design.clone(),
            offset: Array1::zeros(n),
            ..ParameterBlockSpec::defaults()
        },
    ];
    let states_at = |beta_t: &Array1<f64>, beta_w: &Array1<f64>| -> Vec<ParameterBlockState> {
        let eta_t = x_t.dot(beta_t);
        let q0 = eta_t.mapv(|v| -v);
        let b_now = crate::wiggle::monotone_wiggle_basis_from_knots(q0.view(), &knots, 2)
            .expect("warp basis at the state");
        vec![
            ParameterBlockState {
                eta: eta_t,
                beta: beta_t.clone(),
            },
            ParameterBlockState {
                eta: Array1::zeros(n),
                beta: Array1::zeros(0),
            },
            ParameterBlockState {
                eta: b_now.dot(beta_w),
                beta: beta_w.clone(),
            },
        ]
    };

    let beta_w0 = Array1::<f64>::zeros(p_w);
    let base_states = states_at(&beta_t0, &beta_w0);
    let h_l = family
        .exact_newton_joint_hessian_with_specs(&base_states, &specs)
        .expect("joint hessian")
        .expect("joint hessian available");
    let h_pen = &h_l + &s_w_pad;
    let (hvals, hvecs) = h_pen.eigh(Side::Lower).expect("penalized joint hessian eigh");
    let mut order: Vec<usize> = (0..hvals.len()).collect();
    order.sort_by(|&a, &b| hvals[a].abs().partial_cmp(&hvals[b].abs()).unwrap());
    let hmax = hvals.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    println!(
        "[2647-gauge] eig(H_L + S) at the seed index: |λ|min={:.6e} |λ|max={hmax:.6e}",
        hvals[order[0]].abs()
    );
    for &k in order.iter().take(3) {
        let v = hvecs.column(k);
        let mass_t: f64 = (0..p_t).map(|i| v[i] * v[i]).sum();
        let mass_w: f64 = (p_t..p_t + p_w).map(|i| v[i] * v[i]).sum();
        println!(
            "[2647-gauge]   λ={:+.6e}  block mass threshold={mass_t:.4} wiggle={mass_w:.4}",
            hvals[k]
        );
    }

    let dir = hvecs.column(order[0]).to_owned();
    println!("[2647-gauge]      t | -loglik        | 0.5 b'Sb      | |beta|inf  | q0 in hull");
    for t in [0.0_f64, 1.0, 2.0, 4.0, 8.0, 16.0, 64.0, 256.0, 1024.0] {
        let mut beta_t = beta_t0.clone();
        let mut beta_w = beta_w0.clone();
        for i in 0..p_t {
            beta_t[i] += t * dir[i];
        }
        for i in 0..p_w {
            beta_w[i] += t * dir[p_t + i];
        }
        let states = states_at(&beta_t, &beta_w);
        let q0 = states[0].eta.mapv(|v| -v);
        let inside = q0.iter().filter(|&&u| u >= left && u <= right).count();
        let ll = family
            .log_likelihood_only(&states)
            .expect("orbit log-likelihood");
        let pen = 0.5 * (beta_t.dot(&s_t.dot(&beta_t)) + beta_w.dot(&s_w.dot(&beta_w)));
        let beta_inf = beta_t
            .iter()
            .chain(beta_w.iter())
            .map(|v| v.abs())
            .fold(0.0_f64, f64::max);
        println!(
            "[2647-gauge] {t:>8.1} | {:.8e} | {pen:.6e} | {beta_inf:.4e} | {inside}/{n}",
            -ll
        );
    }
}
