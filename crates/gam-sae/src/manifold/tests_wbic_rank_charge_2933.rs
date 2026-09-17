//! #2933 F30–F32. The tempered posterior expectation keeps its mean
//! displacement; the rank-charge audit fills every field from one evaluated
//! state; the MP edge's false-rank rate under a fitted noise-only null is
//! measured against the derived conditional law; and a stratum names the
//! boundary a crossing changes.

use super::tests::{TestPeriodicEvaluator, periodic_basis};
use super::wbic_audit::rank_charge_stratum;
use super::*;
use gam_linalg::utils::splitmix64;
use ndarray::array;

fn uniform01(state: &mut u64) -> f64 {
    (splitmix64(state) >> 11) as f64 / (1u64 << 53) as f64
}

fn standard_normal(state: &mut u64) -> f64 {
    let u1 = uniform01(state).max(f64::MIN_POSITIVE);
    let u2 = uniform01(state);
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

/// Trapezoid rule for `E[½h(w−ŵ)²]` under `exp(−½βh(w−ŵ)² − ½τw²)`. It evaluates
/// only that density, on a window reaching twelve curvature standard deviations
/// past both `0` and `ŵ` (the tempered mean lies between them), where the rule is
/// spectrally accurate for a Gaussian integrand.
fn tempered_excess_by_quadrature_1d(h: f64, tau: f64, mle: f64, beta: f64) -> f64 {
    let log_density = |w: f64| -0.5 * beta * h * (w - mle).powi(2) - 0.5 * tau * w * w;
    let curvature_sd = (beta * h + tau).sqrt().recip();
    let lower = mle.min(0.0) - 12.0 * curvature_sd;
    let upper = mle.max(0.0) + 12.0 * curvature_sd;
    let steps = 20_000usize;
    let width = (upper - lower) / steps as f64;
    let grid = |i: usize| lower + i as f64 * width;
    let peak = (0..=steps)
        .map(|i| log_density(grid(i)))
        .fold(f64::NEG_INFINITY, f64::max);
    let mut mass = 0.0_f64;
    let mut moment = 0.0_f64;
    for i in 0..=steps {
        let w = grid(i);
        let end_weight = if i == 0 || i == steps { 0.5 } else { 1.0 };
        let density = end_weight * (log_density(w) - peak).exp();
        mass += density;
        moment += density * 0.5 * h * (w - mle).powi(2);
    }
    moment / mass
}

fn two_by_two_extreme_eigenvalues(matrix: &Array2<f64>) -> (f64, f64) {
    let mid = 0.5 * (matrix[[0, 0]] + matrix[[1, 1]]);
    let radius =
        (0.25 * (matrix[[0, 0]] - matrix[[1, 1]]).powi(2) + matrix[[0, 1]].powi(2)).sqrt();
    (mid - radius, mid + radius)
}

fn half_quadratic(matrix: &Array2<f64>, center: &Array1<f64>, w: [f64; 2]) -> f64 {
    let d0 = w[0] - center[0];
    let d1 = w[1] - center[1];
    0.5 * (matrix[[0, 0]] * d0 * d0 + 2.0 * matrix[[0, 1]] * d0 * d1 + matrix[[1, 1]] * d1 * d1)
}

/// Tensor trapezoid rule for `E[½(w−ŵ)ᵀH(w−ŵ)]` under
/// `exp(−½β(w−ŵ)ᵀH(w−ŵ) − ½(w−μ₀)ᵀΛ(w−μ₀))`. The window is centred on `ŵ` and
/// reaches twelve standard deviations of the flattest curvature direction past
/// the largest possible mean shift `‖Λ‖·‖μ₀−ŵ‖/λ_min(βH+Λ)`.
fn tempered_excess_by_quadrature_2d(
    information: &Array2<f64>,
    prior_precision: &Array2<f64>,
    prior_mean: &Array1<f64>,
    minimizer: &Array1<f64>,
    beta: f64,
) -> f64 {
    let precision = information * beta + prior_precision;
    let lambda_min = two_by_two_extreme_eigenvalues(&precision).0;
    let prior_norm = two_by_two_extreme_eigenvalues(prior_precision).1;
    let offset = ((minimizer[0] - prior_mean[0]).powi(2)
        + (minimizer[1] - prior_mean[1]).powi(2))
    .sqrt();
    let half_width = 12.0 * lambda_min.sqrt().recip() + prior_norm * offset / lambda_min;
    let steps = 1600usize;
    let width = 2.0 * half_width / steps as f64;
    let coordinate = |axis: usize, i: usize| minimizer[axis] - half_width + i as f64 * width;
    let log_density = |w: [f64; 2]| {
        -beta * half_quadratic(information, minimizer, w)
            - half_quadratic(prior_precision, prior_mean, w)
    };
    let reference = log_density([minimizer[0], minimizer[1]]);
    let mut mass = 0.0_f64;
    let mut moment = 0.0_f64;
    for i in 0..=steps {
        let first = coordinate(0, i);
        let first_weight = if i == 0 || i == steps { 0.5 } else { 1.0 };
        for j in 0..=steps {
            let w = [first, coordinate(1, j)];
            let second_weight = if j == 0 || j == steps { 0.5 } else { 1.0 };
            let density = first_weight * second_weight * (log_density(w) - reference).exp();
            mass += density;
            moment += density * half_quadratic(information, minimizer, w);
        }
    }
    moment / mass
}

#[test]
fn tempered_gaussian_excess_loss_keeps_the_posterior_mean_displacement_2933() {
    // The audit's counterexample: h = τ = β = 1, ŵ = 2. The variance-only soft
    // count gives 0.25; the tempered posterior expectation is 0.75.
    let unit = array![[1.0]];
    let excess = tempered_gaussian_excess_loss(
        unit.view(),
        unit.view(),
        array![0.0].view(),
        array![2.0].view(),
        1.0,
    )
    .expect("proper tempered posterior");
    assert!((excess.posterior_spread - 0.25).abs() < 1.0e-14, "{excess:?}");
    assert!((excess.mean_displacement - 0.5).abs() < 1.0e-14, "{excess:?}");
    assert!((excess.total() - 0.75).abs() < 1.0e-14, "{excess:?}");

    // Closed form and an independent quadrature across temperatures, including
    // Watanabe's β = 1/ln n, and across prior strengths.
    for &(h, tau, mle) in &[(1.0, 1.0, 2.0), (40.0, 0.3, -1.7), (0.2, 5.0, 0.6)] {
        for beta in [1.0 / 1000.0_f64.ln(), 1.0 / 10.0_f64.ln(), 0.5, 1.0, 3.0] {
            let excess = tempered_gaussian_excess_loss(
                array![[h]].view(),
                array![[tau]].view(),
                array![0.0].view(),
                array![mle].view(),
                beta,
            )
            .expect("proper tempered posterior");
            let precision = beta * h + tau;
            let mean = beta * h * mle / precision;
            let closed = 0.5 * h * ((mean - mle).powi(2) + precision.recip());
            let quadrature = tempered_excess_by_quadrature_1d(h, tau, mle, beta);
            assert!(
                (excess.total() - closed).abs() <= 1.0e-12 * closed,
                "h={h} tau={tau} mle={mle} beta={beta}: primitive {excess:?} vs closed form {closed}"
            );
            assert!(
                (excess.total() - quadrature).abs() <= 1.0e-9 * quadrature,
                "h={h} tau={tau} mle={mle} beta={beta}: primitive {excess:?} vs quadrature {quadrature}"
            );
        }
    }

    // Two dimensions, correlated information, a prior that is flat along the
    // first axis and centred off zero along the second.
    let information = array![[3.0, 1.0], [1.0, 2.0]];
    let prior_precision = array![[0.0, 0.0], [0.0, 4.0]];
    let prior_mean = array![0.0, 0.5];
    let minimizer = array![1.0, -2.0];
    for beta in [0.4, 1.3] {
        let excess = tempered_gaussian_excess_loss(
            information.view(),
            prior_precision.view(),
            prior_mean.view(),
            minimizer.view(),
            beta,
        )
        .expect("proper tempered posterior");
        let quadrature = tempered_excess_by_quadrature_2d(
            &information,
            &prior_precision,
            &prior_mean,
            &minimizer,
            beta,
        );
        assert!(
            (excess.total() - quadrature).abs() <= 1.0e-9 * quadrature,
            "beta={beta}: primitive {excess:?} vs quadrature {quadrature}"
        );
    }
    let excess = tempered_gaussian_excess_loss(
        information.view(),
        prior_precision.view(),
        prior_mean.view(),
        minimizer.view(),
        0.4,
    )
    .expect("proper tempered posterior");
    assert!(
        excess.mean_displacement > 0.5 * excess.total(),
        "the displacement term must be material in this fixture: {excess:?}"
    );

    // An improper tempered posterior is refused, not priced.
    let zero = array![[0.0]];
    assert!(
        tempered_gaussian_excess_loss(
            zero.view(),
            zero.view(),
            array![0.0].view(),
            array![1.0].view(),
            1.0
        )
        .is_err()
    );
}

fn resolved_single_atom_state() -> (SaeManifoldTerm, Array2<f64>, SaeManifoldRho) {
    let n = 24usize;
    let coords = Array2::from_shape_fn((n, 1), |(row, _)| (row as f64 + 0.25) / n as f64);
    let (phi, jet) = periodic_basis(&coords);
    let decoder = array![[0.30, -0.10], [1.20, 0.20], [0.10, 1.10]];
    let mut target = phi.dot(&decoder);
    for row in 0..n {
        target[[row, 0]] += 1.0e-3 * (0.37 * row as f64).sin();
        target[[row, 1]] += 1.0e-3 * (0.29 * row as f64).cos();
    }
    let atom = SaeManifoldAtom::new_with_provided_function_gram(
        "periodic",
        SaeAtomBasisKind::Periodic,
        1,
        phi,
        jet,
        decoder,
        Array2::<f64>::eye(3),
    )
    .unwrap()
    .with_basis_evaluator(Arc::new(TestPeriodicEvaluator));
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        Array2::<f64>::zeros((n, 1)),
        vec![coords],
        vec![LatentManifold::Circle { period: 1.0 }],
        AssignmentMode::softmax(1.0),
    )
    .unwrap();
    let term = SaeManifoldTerm::new(vec![atom], assignment).unwrap();
    let rho = SaeManifoldRho::new(0.0, 0.8_f64.ln(), vec![array![250.0_f64.ln()]]);
    (term, target, rho)
}

#[test]
fn rank_charge_audit_fills_every_field_from_one_state_2933() {
    let (term, target, rho) = resolved_single_atom_state();
    let loss = term.loss(target.view(), &rho).unwrap();
    let sys = term
        .assemble_arrow_schur(target.view(), &rho, None)
        .unwrap();
    let options = ArrowSolveOptions::direct().with_positive_definite_evidence();
    let (_delta_t, _delta_beta, cache) =
        solve_arrow_newton_step_with_options(&sys, 0.0, 0.0, &options).unwrap();

    let audits = term
        .rank_charge_audit(target.view(), &rho, &loss, &cache)
        .expect("the plain single-atom state is auditable");
    assert_eq!(audits.len(), 1);
    let audit = &audits[0];
    assert_eq!(
        (
            audit.atom,
            audit.basis_dim,
            audit.output_dim,
            audit.storage_dim,
            audit.intrinsic_dim
        ),
        (0, 3, 2, 6, 1)
    );

    // Tied to the priced state: the same dispersion, the same DOF the criterion
    // charges, and the criterion's charge arithmetic.
    let residual = term.reconstruction_residual(target.view(), &rho).unwrap();
    let dispersion = term
        .reconstruction_dispersion(&loss, &cache, &rho, Some(residual.view()))
        .unwrap()
        .raw_output_noise_variance;
    assert_eq!(audit.dispersion.to_bits(), dispersion.to_bits());
    let mut grams = term.empty_decoder_gram_accumulator();
    term.accumulate_decoder_gram(&mut grams).unwrap();
    let n_eff = term.per_atom_effective_sample_size();
    let priced_dof = term
        .rank_dof_from_grams(&grams, &n_eff, &rho, dispersion)
        .unwrap();
    assert_eq!(audit.stratum.production_dof().to_bits(), priced_dof[0].to_bits());
    assert_eq!(
        audit.stratum.production_charge().to_bits(),
        (0.5 * priced_dof[0] * n_eff[0].max(1.0).ln()).to_bits()
    );
    assert!((audit.stratum.effective_sample_size() - 24.0).abs() <= 1.0e-12);
    assert!((audit.lambda_smooth - 0.8).abs() <= 1.0e-12);
    assert!((audit.inverse_temperature - 24.0_f64.ln().recip()).abs() <= 1.0e-15);
    assert_eq!(audit.stratum.mp_reconstruction_rank(), 2);
    assert_eq!(audit.stratum.production_chargeable_rank(), 2);
    let energies = audit.stratum.reconstruction_energies();
    let edge = audit.stratum.mp_reconstruction_rank_edge();
    assert_eq!(energies.len(), 2);
    assert!(energies.iter().all(|&energy| energy > edge));

    // Two directions above the edge: the nearest is the smaller one, and its
    // crossing drops the charge by one unit of ½·edf·ln N_eff.
    let boundary = audit
        .nearest_mp_boundary
        .expect("two reconstruction directions");
    assert_eq!(boundary.direction, 1);
    assert_eq!(boundary.signed_gap, energies[1] - edge);
    let unit_jump = 0.5 * audit.stratum.basis_edf() * 24.0_f64.ln();
    assert!((boundary.charge_jump + unit_jump).abs() <= 1.0e-12 * unit_jump);

    // Conditional noise law: a positive expected total energy, and a top-energy
    // bound no smaller than the average over the two nonzero energies.
    assert!(audit.noise_null.expected_total_energy > 0.0);
    assert!(
        audit.noise_null.top_energy_expectation_bound
            >= 0.5 * audit.noise_null.expected_total_energy
    );

    // The tempered posterior against an independent route: accumulate the row
    // design and the raw target (K = 1, so the target is the atom's conditional
    // response), solve the normal equations per channel, and hand the stacked
    // (m·p)-dimensional system to the general tempered-Gaussian primitive.
    let atom = &term.atoms[0];
    let phi = &atom.basis_values;
    let gates = term.assignment.assignments();
    let (n, m) = phi.dim();
    let p = target.ncols();
    let mut row_gram = Array2::<f64>::zeros((m, m));
    let mut row_score = Array2::<f64>::zeros((m, p));
    for row in 0..n {
        let gate = gates[[row, 0]];
        for i in 0..m {
            for j in 0..m {
                row_gram[[i, j]] += gate * gate * phi[[row, i]] * phi[[row, j]];
            }
            for c in 0..p {
                row_score[[i, c]] += gate * phi[[row, i]] * target[[row, c]];
            }
        }
    }
    let channel_minimizers = row_gram
        .cholesky(Side::Lower)
        .expect("resolved design")
        .solve_mat(&row_score);
    let dim = m * p;
    let penalty = atom.smooth_penalty();
    let mut information = Array2::<f64>::zeros((dim, dim));
    let mut prior_precision = Array2::<f64>::zeros((dim, dim));
    let mut minimizer = Array1::<f64>::zeros(dim);
    for c in 0..p {
        for i in 0..m {
            minimizer[c * m + i] = channel_minimizers[[i, c]];
            for j in 0..m {
                information[[c * m + i, c * m + j]] = row_gram[[i, j]] / dispersion;
                prior_precision[[c * m + i, c * m + j]] =
                    audit.lambda_smooth * penalty[[i, j]] / dispersion;
            }
        }
    }
    let reference = tempered_gaussian_excess_loss(
        information.view(),
        prior_precision.view(),
        Array1::<f64>::zeros(dim).view(),
        minimizer.view(),
        audit.inverse_temperature,
    )
    .expect("proper tempered posterior");
    let posterior = audit.tempered_posterior;
    assert_eq!(
        posterior.inverse_temperature.to_bits(),
        audit.inverse_temperature.to_bits()
    );
    assert!(
        (posterior.posterior_spread - reference.posterior_spread).abs()
            <= 1.0e-10 * reference.posterior_spread,
        "audit {posterior:?} vs stacked reference {reference:?}"
    );
    assert!(
        (posterior.mean_displacement - reference.mean_displacement).abs()
            <= 1.0e-7 * reference.mean_displacement,
        "audit {posterior:?} vs stacked reference {reference:?}"
    );
    assert!(
        posterior.mean_displacement > 1.0e-3 * posterior.total(),
        "the displacement term must be material in this fixture: {posterior:?}"
    );
    assert_eq!(
        audit.production_minus_tempered.to_bits(),
        (audit.stratum.production_charge() - posterior.total()).to_bits()
    );

    // Where the conditional model is not the data fit, the audit refuses.
    let mut weighted = resolved_single_atom_state().0;
    weighted
        .set_row_loss_weights((0..n).map(|row| if row % 2 == 0 { 1.5 } else { 0.5 }).collect())
        .unwrap();
    let refusal = weighted
        .rank_charge_audit(target.view(), &rho, &loss, &cache)
        .expect_err("weighted rows are refused");
    assert!(refusal.contains("row loss weights"), "{refusal}");
}

#[test]
fn mp_edge_false_rank_rate_under_a_fitted_noise_only_null_2933() {
    let n = 160usize;
    let m = 5usize;
    let p = 6usize;
    let dispersion = 0.7_f64;
    let lambda = 3.0_f64;
    let draws = 3000usize;
    let phi = Array2::from_shape_fn((n, m), |(row, col)| {
        let t = (row as f64 + 0.5) / n as f64;
        (std::f64::consts::PI * col as f64 * t).cos()
    });
    let mut difference = Array2::<f64>::zeros((m - 2, m));
    for i in 0..m - 2 {
        difference[[i, i]] = 1.0;
        difference[[i, i + 1]] = -2.0;
        difference[[i, i + 2]] = 1.0;
    }
    let penalty = difference.t().dot(&difference);
    let mut state = 0x2933_3032_2933_3032_u64;
    // Nonuniform gates, then the same gates rescaled.
    for gate_scale in [1.0_f64, 0.4] {
        let gates = Array1::from_shape_fn(n, |row| {
            gate_scale * (0.25 + 0.75 * ((row * 37) % n) as f64 / n as f64)
        });
        let design = Array2::from_shape_fn((n, m), |(row, col)| gates[row] * phi[[row, col]]);
        let gram = design.t().dot(&design);
        let n_eff = gates.iter().map(|gate| gate * gate).sum::<f64>();
        let edge =
            crate::null_battery::mp_reconstruction_rank_edge(n_eff, p as f64, dispersion).unwrap();
        let null = conditional_noise_null(&gram, n_eff, p, dispersion, lambda, Some(&penalty))
            .expect("conditional noise-only law");
        let ridge = (&gram + &(&penalty * lambda))
            .cholesky(Side::Lower)
            .expect("penalized Gram is positive definite");
        let noise_sd = dispersion.sqrt();
        let mut totals = Vec::with_capacity(draws);
        let mut top_sum = 0.0_f64;
        let mut mp_false_ranks = 0usize;
        let mut chargeable_false_ranks = 0usize;
        for _ in 0..draws {
            let noise =
                Array2::from_shape_simple_fn((n, p), || noise_sd * standard_normal(&mut state));
            // The fitted decoder of a noise-only target under the fixed design.
            let decoder = ridge.solve_mat(&design.t().dot(&noise));
            let stratum = rank_charge_stratum(
                &gram,
                &decoder,
                n_eff,
                p as f64,
                dispersion,
                lambda,
                Some(&penalty),
            )
            .expect("noise decoder stratum");
            totals.push(stratum.reconstruction_energies().iter().sum::<f64>());
            top_sum += stratum.top_reconstruction_energy();
            mp_false_ranks += usize::from(stratum.mp_reconstruction_rank() > 0);
            chargeable_false_ranks += usize::from(stratum.production_chargeable_rank() > 0);
        }
        let count = draws as f64;
        let mean_total = totals.iter().sum::<f64>() / count;
        let variance =
            totals.iter().map(|total| (total - mean_total).powi(2)).sum::<f64>() / (count - 1.0);
        let standard_error = (variance / count).sqrt();
        let mean_top = top_sum / count;
        let mp_rate = mp_false_ranks as f64 / count;
        let chargeable_rate = chargeable_false_ranks as f64 / count;
        eprintln!(
            "#2933 F32 fitted noise-only null: gate_scale={gate_scale} N_eff={n_eff:.4} \
             edge={edge:.6e} expected_total={:.6e} mc_total={mean_total:.6e} (se {standard_error:.2e}) \
             top_bound={:.6e} mc_top={mean_top:.6e} mp_false_rank_rate={mp_rate:.4} \
             chargeable_false_rank_rate={chargeable_rate:.4}",
            null.expected_total_energy, null.top_energy_expectation_bound
        );
        assert!(
            standard_error < 0.05 * null.expected_total_energy,
            "the Monte Carlo sample must resolve the expected energy"
        );
        assert!(
            (mean_total - null.expected_total_energy).abs() <= 4.0 * standard_error,
            "gate_scale={gate_scale}: mc total energy {mean_total} (se {standard_error}) vs the \
             exact conditional expectation {}",
            null.expected_total_energy
        );
        assert!(
            mean_top <= null.top_energy_expectation_bound,
            "gate_scale={gate_scale}: mc top energy {mean_top} exceeds its bound {}",
            null.top_energy_expectation_bound
        );
        // Markov's inequality on the top energy bounds the MP false-rank rate.
        let markov = (null.top_energy_expectation_bound / edge).min(1.0);
        assert!(
            mp_rate <= markov + 4.0 * (markov * (1.0 - markov) / count).sqrt(),
            "gate_scale={gate_scale}: MP false-rank rate {mp_rate} vs Markov bound {markov}"
        );
        // Every alive noise decoder is charged at least rank one (#2258).
        assert_eq!(chargeable_false_ranks, draws);
    }
}

fn decoder_with_energies(edge: f64, first: f64, second: f64, p: usize) -> Array2<f64> {
    let mut decoder = Array2::<f64>::zeros((2, p));
    decoder[[0, 0]] = (first * edge).sqrt();
    decoder[[1, 1]] = (second * edge).sqrt();
    decoder
}

#[test]
fn rank_charge_stratum_names_the_mp_edge_branch_2933() {
    let n_eff = 50.0_f64;
    let p = 3usize;
    let dispersion = 1.0_f64;
    // With G = N_eff·I the reconstruction energies are the squared decoder
    // singular values.
    let gram = Array2::<f64>::eye(2) * n_eff;
    let edge = crate::null_battery::mp_reconstruction_rank_edge(n_eff, p as f64, dispersion).unwrap();
    let stratum_at = |first: f64, second: f64| {
        rank_charge_stratum(
            &gram,
            &decoder_with_energies(edge, first, second, p),
            n_eff,
            p as f64,
            dispersion,
            0.0,
            None,
        )
        .expect("stratum")
    };
    let log_n = n_eff.ln();

    // One direction above the edge and one just below it.
    let resolved = stratum_at(1.05, 0.99);
    assert_eq!(
        (resolved.mp_reconstruction_rank(), resolved.production_chargeable_rank()),
        (1, 1)
    );
    let boundary = resolved.nearest_mp_boundary().expect("two directions");
    assert_eq!(boundary.direction, 1);
    assert!((resolved.reconstruction_energies()[1] - 0.99 * edge).abs() <= 1.0e-9 * edge);
    assert!(boundary.signed_gap < 0.0);
    let unit_jump = 0.5 * resolved.basis_edf() * log_n;
    assert!((boundary.charge_jump - unit_jump).abs() <= 1.0e-12 * unit_jump);
    // Inside the stratum the charge does not move with the decoder.
    let same_stratum = stratum_at(1.08, 0.97);
    assert_eq!(
        same_stratum.production_charge().to_bits(),
        resolved.production_charge().to_bits()
    );
    // Across the boundary it jumps by exactly the reported amount.
    let crossed = stratum_at(1.05, 1.01);
    assert_eq!(crossed.production_chargeable_rank(), 2);
    assert!(
        (crossed.production_charge() - resolved.production_charge() - boundary.charge_jump).abs()
            <= 1.0e-12 * unit_jump
    );

    // The promotion boundary: the only counted direction falling below the edge
    // leaves the chargeable rank at one, so that crossing adds no charge.
    let single = stratum_at(1.01, 0.5);
    let boundary = single.nearest_mp_boundary().expect("two directions");
    assert_eq!(boundary.direction, 0);
    assert!(boundary.signed_gap > 0.0);
    assert_eq!(boundary.charge_jump, 0.0);
    let promoted = stratum_at(0.99, 0.5);
    assert_eq!(
        (promoted.mp_reconstruction_rank(), promoted.production_chargeable_rank()),
        (0, 1)
    );
    assert_eq!(
        promoted.production_charge().to_bits(),
        single.production_charge().to_bits()
    );

    // The value path prices the branch the stratum names.
    let decoder = decoder_with_energies(edge, 1.05, 0.99, p);
    let priced = realised_rank_charge_dof(&gram, &decoder, n_eff, p as f64, dispersion, 0.0, None)
        .unwrap();
    assert_eq!(priced.to_bits(), resolved.production_dof().to_bits());
}
