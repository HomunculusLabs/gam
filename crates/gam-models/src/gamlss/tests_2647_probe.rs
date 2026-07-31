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
