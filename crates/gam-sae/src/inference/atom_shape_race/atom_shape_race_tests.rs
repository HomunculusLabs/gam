use super::*;
use ndarray::Array2;

#[test]
fn atom_shape_race_rejects_invalid_folds_before_any_reporting_fit() {
    // Coincident coordinates would fail the reporting gauge if fitting
    // began. The fold-contract error must win, proving validation happens
    // before any full-data or control fit is launched.
    let coords = Array2::<f64>::zeros((4, 2));
    for folds in [0usize, 1, 5] {
        let error = run_atom_shape_race(coords.view(), folds, 11)
            .expect_err("shape CV requires 2 <= folds <= n");
        assert_eq!(
            error,
            format!("adjudicate_atom_shape: require 2 <= folds <= n; got folds={folds}, n=4")
        );
    }

    let error = run_atom_shape_race(coords.view(), 2, 11)
        .expect_err("every ring-cluster training fold needs at least three rows");
    assert!(error.contains("minimum of 2 with n=4, folds=2"), "{error}");
}

#[test]
fn ring_of_clusters_owns_discrete_cyclic_verdict_2262() {
    let clusters = 7usize;
    let per_cluster = 40usize;
    let mut coords = Array2::<f64>::zeros((clusters * per_cluster, 2));
    for cluster in 0..clusters {
        let angle = std::f64::consts::TAU * cluster as f64 / clusters as f64;
        let (sin_angle, cos_angle) = angle.sin_cos();
        for sample in 0..per_cluster {
            let phase = std::f64::consts::TAU * sample as f64 / per_cluster as f64;
            // This is a cloud, not a constant-radius micro-circle, so the
            // planted candidate is a genuine Gaussian cluster model.
            let local_radius = 0.04 * (1.0 + 0.3 * (3.0 * phase).cos());
            let radial_noise = local_radius * phase.cos();
            let tangent_noise = local_radius * phase.sin();
            let radius = 2.0 + radial_noise;
            let row = cluster * per_cluster + sample;
            coords[[row, 0]] = 0.5 + radius * cos_angle - tangent_noise * sin_angle;
            coords[[row, 1]] = -0.25 + radius * sin_angle + tangent_noise * cos_angle;
        }
    }
    let verdict = run_atom_shape_race(coords.view(), 5, 2262).unwrap();
    assert_eq!(verdict.ring_clusters_reporting_k, 7);
    assert_eq!(verdict.winner_class, "ring_clusters");
    assert!(
        verdict.reporting_winner.starts_with("ring_clusters_k7"),
        "reporting_winner={} names={:?} weights={:?} bic={:?}",
        verdict.reporting_winner,
        verdict.candidate_names,
        verdict.stacking_weights,
        verdict.bic,
    );
    assert!(verdict.circle_wins);
    assert!(verdict.circular_margin > 0.0);
    assert_eq!(verdict.candidate_names.len(), 4);
}

#[test]
fn circular_gaussian_density_is_normalized_and_finite_at_center() {
    // Integrate 2 pi r p(r) dr with composite Simpson quadrature. The
    // model is a density on Cartesian R^2, so this must be one without a
    // truncated-radius correction or a singular center convention.
    const INTERVALS: usize = 200_000;
    for (radius, noise_variance) in [(0.0, 0.7), (2.0, 0.09), (20.0, 1.0e-4)] {
        let fit = CircularGaussianFit2d::from_parameters([1.25, -0.75], radius, noise_variance)
            .unwrap();
        let center = fit.center();
        assert!(fit.log_density(center[0], center[1]).is_finite());

        let upper = radius + 12.0 * noise_variance.sqrt();
        let step = upper / INTERVALS as f64;
        let radial_mass = |r: f64| {
            if r == 0.0 {
                0.0
            } else {
                std::f64::consts::TAU * r * fit.log_density(center[0] + r, center[1]).exp()
            }
        };
        let mut weighted_sum = radial_mass(0.0) + radial_mass(upper);
        for index in 1..INTERVALS {
            let weight = if index % 2 == 0 { 2.0 } else { 4.0 };
            weighted_sum += weight * radial_mass(index as f64 * step);
        }
        let integral = step * weighted_sum / 3.0;
        assert!(
            (integral - 1.0).abs() < 3.0e-6,
            "circular Gaussian failed to normalize: R={radius} s={noise_variance} integral={integral:.16e}"
        );
    }
}

#[test]
fn circular_gaussian_mle_satisfies_all_score_equations() {
    let n = 360usize;
    let mut coords = Array2::<f64>::zeros((n, 2));
    for row in 0..n {
        let angle = std::f64::consts::TAU * row as f64 / n as f64;
        coords[[row, 0]] = 0.2 + 1.7 * angle.cos() + 0.15 * (2.3 * angle).cos();
        coords[[row, 1]] = -0.4 + 1.7 * angle.sin() + 0.12 * (4.7 * angle).sin();
    }
    let rows = (0..n).collect::<Vec<_>>();
    let fit = CircularGaussianFit2d::fit(coords.view(), &rows).unwrap();
    let center = fit.center();
    let radius = fit.radius();
    let noise_variance = fit.noise_variance();
    let mut center_score = [0.0_f64; 2];
    let mut radius_score = 0.0_f64;
    let mut variance_score = 0.0_f64;
    for row in 0..n {
        let dx = coords[[row, 0]] - center[0];
        let dy = coords[[row, 1]] - center[1];
        let observed_radius = dx.hypot(dy);
        let kappa = radius * observed_radius / noise_variance;
        let (_, bessel_ratio) = gam_math::special::bessel_i0_log_minus_abs_and_ratio(kappa);
        let center_multiplier =
            (1.0 - radius * bessel_ratio / observed_radius) / noise_variance;
        center_score[0] += center_multiplier * dx;
        center_score[1] += center_multiplier * dy;
        radius_score += (observed_radius * bessel_ratio - radius) / noise_variance;
        let stable_expected_residual = (observed_radius - radius).powi(2)
            + 2.0 * radius * observed_radius * (1.0 - bessel_ratio);
        variance_score +=
            -1.0 / noise_variance + 0.5 * stable_expected_residual / noise_variance.powi(2);
    }
    let nf = n as f64;
    let score_scale = noise_variance.sqrt();
    assert!(
        (center_score[0] / nf * score_scale).abs() < 2.0e-8
            && (center_score[1] / nf * score_scale).abs() < 2.0e-8,
        "center score did not vanish: {:?}",
        center_score
    );
    assert!(
        (radius_score / nf * score_scale).abs() < 2.0e-8,
        "radius score did not vanish: {radius_score}"
    );
    assert!(
        (variance_score / nf * noise_variance).abs() < 2.0e-8,
        "variance score did not vanish: {variance_score}"
    );
}

#[test]
fn smooth_circle_density_and_evidence_are_translation_invariant() {
    let n = 160usize;
    let mut coords = Array2::<f64>::zeros((n, 2));
    for row in 0..n {
        let angle = std::f64::consts::TAU * row as f64 / n as f64;
        let radius = 2.0 + 0.08 * (3.0 * angle).cos() + 0.03 * (5.0 * angle).sin();
        coords[[row, 0]] = radius * angle.cos();
        coords[[row, 1]] = radius * angle.sin();
    }
    let mut translated = coords.clone();
    for mut row in translated.rows_mut() {
        row[0] += 137.25;
        row[1] -= 91.75;
    }
    let train = (0..120).collect::<Vec<_>>();
    let eval = (120..n).collect::<Vec<_>>();
    let base_density = ring_provider_2d(coords.clone())(&train, &eval).unwrap();
    let shifted_density = ring_provider_2d(translated.clone())(&train, &eval).unwrap();
    for (base, shifted) in base_density.iter().zip(&shifted_density) {
        assert!(
            (base - shifted).abs() < 2.0e-11,
            "translation changed held-out ring density: base={base:.16e}, shifted={shifted:.16e}"
        );
    }
    let base_evidence = ring_bic_2d(coords.view()).unwrap();
    let shifted_evidence = ring_bic_2d(translated.view()).unwrap();
    assert!(
        (base_evidence - shifted_evidence).abs() < 2.0e-9,
        "translation changed ring evidence: base={base_evidence:.16e}, shifted={shifted_evidence:.16e}"
    );
}

#[test]
fn smooth_circle_density_and_evidence_obey_the_2d_scale_law() {
    let n = 160usize;
    let scale = 7.25_f64;
    let mut coords = Array2::<f64>::zeros((n, 2));
    for row in 0..n {
        let angle = std::f64::consts::TAU * row as f64 / n as f64;
        let radius = 2.0 + 0.08 * (3.0 * angle).cos() + 0.03 * (5.0 * angle).sin();
        coords[[row, 0]] = radius * angle.cos();
        coords[[row, 1]] = radius * angle.sin();
    }
    let scaled = coords.mapv(|value| scale * value);
    let train = (0..120).collect::<Vec<_>>();
    let eval = (120..n).collect::<Vec<_>>();
    let base_density = ring_provider_2d(coords.clone())(&train, &eval).unwrap();
    let scaled_density = ring_provider_2d(scaled.clone())(&train, &eval).unwrap();
    for (base, transformed) in base_density.iter().zip(&scaled_density) {
        let expected = base - 2.0 * scale.ln();
        assert!(
            (transformed - expected).abs() < 2.0e-10,
            "ring density violated p_a(ax)=p(x)/a^2: expected={expected:.16e}, got={transformed:.16e}"
        );
    }

    let base_evidence = ring_bic_2d(coords.view()).unwrap();
    let scaled_evidence = ring_bic_2d(scaled.view()).unwrap();
    let expected = base_evidence + 2.0 * n as f64 * scale.ln();
    assert!(
        (scaled_evidence - expected).abs() < 2.0e-8,
        "ring evidence violated the 2-D scale law: expected={expected:.16e}, got={scaled_evidence:.16e}"
    );
}

#[test]
fn held_out_values_do_not_choose_smooth_candidate_gauges() {
    let train_rows = 120usize;
    let eval_rows = 24usize;
    let mut coords = Array2::<f64>::zeros((train_rows + eval_rows, 2));
    for row in 0..coords.nrows() {
        let angle = std::f64::consts::TAU * row as f64 / coords.nrows() as f64;
        let radius = 1.8 + 0.06 * (5.0 * angle).cos();
        coords[[row, 0]] = 0.4 + radius * angle.cos();
        coords[[row, 1]] = -0.7 + radius * angle.sin();
    }
    let train = (0..train_rows).collect::<Vec<_>>();
    let eval = (train_rows..train_rows + eval_rows).collect::<Vec<_>>();
    let circle_baseline = ring_provider_2d(coords.clone())(&train, &eval).unwrap();
    let gaussian_baseline = gaussian_provider_2d(coords.clone())(&train, &eval).unwrap();

    let perturbed_slot = eval_rows - 1;
    coords[[train_rows + perturbed_slot, 0]] = 1.0e12;
    coords[[train_rows + perturbed_slot, 1]] = -1.0e12;
    let circle_perturbed = ring_provider_2d(coords.clone())(&train, &eval).unwrap();
    let gaussian_perturbed = gaussian_provider_2d(coords)(&train, &eval).unwrap();
    assert_eq!(
        &circle_perturbed[..perturbed_slot],
        &circle_baseline[..perturbed_slot]
    );
    assert_eq!(
        &gaussian_perturbed[..perturbed_slot],
        &gaussian_baseline[..perturbed_slot]
    );
}

#[test]
fn shape_coordinate_gauge_handles_antipodal_finite_range() {
    let magnitude = 1.7e308_f64;
    let coords = ndarray::array![
        [magnitude, 0.0],
        [-magnitude, 0.0],
        [0.0, magnitude],
        [0.0, -magnitude],
    ];
    let gauge = fit_shape_coordinate_gauge(coords.view(), "extreme shape").unwrap();
    let canonical =
        apply_shape_coordinate_gauge(coords.view(), gauge, "extreme shape").unwrap();
    assert!(canonical.iter().all(|value| value.is_finite()));
    assert!(gauge.log_scale.is_finite());
    assert_eq!(canonical[[0, 0]], -canonical[[1, 0]]);
    assert_eq!(canonical[[2, 1]], -canonical[[3, 1]]);
}

#[test]
fn euclidean_gaussian_is_similarity_equivariant_with_a_normalized_density() {
    let n = 180usize;
    let mut coords = Array2::<f64>::zeros((n, 2));
    for row in 0..n {
        let phase = std::f64::consts::TAU * row as f64 / n as f64;
        let x = 2.5 * phase.cos() + 0.35 * (3.0 * phase).sin();
        let y = 0.4 * x + 0.7 * phase.sin() + 0.12 * (5.0 * phase).cos();
        coords[[row, 0]] = x;
        coords[[row, 1]] = y;
    }
    let train = (0..135).collect::<Vec<_>>();
    let eval = (135..n).collect::<Vec<_>>();
    let baseline = gaussian_provider_2d(coords.clone())(&train, &eval).unwrap();

    // A 2-D similarity x' = a Q x + b changes every normalized Cartesian
    // log density by exactly -log|det(aQ)| = -2 log(a).  In particular,
    // the covariance regularizer must scale as a^2 rather than living in
    // arbitrary absolute coordinate units.
    let scale = 1.0e-4_f64;
    let angle = 0.731_f64;
    let (sin_angle, cos_angle) = angle.sin_cos();
    let mut transformed = Array2::<f64>::zeros(coords.raw_dim());
    for row in 0..n {
        let x = coords[[row, 0]];
        let y = coords[[row, 1]];
        transformed[[row, 0]] = 0.25 + scale * (cos_angle * x - sin_angle * y);
        transformed[[row, 1]] = -0.5 + scale * (sin_angle * x + cos_angle * y);
    }
    let canonical = canonical_shape_coordinates(coords.view()).unwrap();
    let transformed_canonical = canonical_shape_coordinates(transformed.view()).unwrap();
    for row in 0..n {
        let expected_x = cos_angle * canonical[[row, 0]] - sin_angle * canonical[[row, 1]];
        let expected_y = sin_angle * canonical[[row, 0]] + cos_angle * canonical[[row, 1]];
        assert!((transformed_canonical[[row, 0]] - expected_x).abs() < 2.0e-11);
        assert!((transformed_canonical[[row, 1]] - expected_y).abs() < 2.0e-11);
    }
    let changed = gaussian_provider_2d(transformed.clone())(&train, &eval).unwrap();
    let jacobian_shift = -2.0 * scale.ln();
    for (&base, &under_similarity) in baseline.iter().zip(&changed) {
        assert!(
            (under_similarity - (base + jacobian_shift)).abs() < 2.0e-7,
            "Gaussian log density violated its similarity Jacobian: base={base:.16e}, transformed={under_similarity:.16e}"
        );
    }

    let baseline_evidence = gaussian_bic_2d(coords.view()).unwrap();
    let transformed_evidence = gaussian_bic_2d(transformed.view()).unwrap();
    let expected_evidence = baseline_evidence + 2.0 * n as f64 * scale.ln();
    assert!(
        (transformed_evidence - expected_evidence).abs() < 2.0e-5,
        "Gaussian evidence violated its similarity Jacobian: expected={expected_evidence:.16e}, actual={transformed_evidence:.16e}"
    );

    let mut translated = coords.clone();
    for mut row in translated.rows_mut() {
        row[0] += 1.0e6;
        row[1] -= 2.0e6;
    }
    let translated_density = gaussian_provider_2d(translated.clone())(&train, &eval).unwrap();
    for (&base, &shifted) in baseline.iter().zip(&translated_density) {
        assert!(
            (base - shifted).abs() < 2.0e-8,
            "translation changed Euclidean Gaussian density: base={base:.16e}, shifted={shifted:.16e}"
        );
    }
}

#[test]
fn euclidean_gaussian_spectral_constraint_stays_positive_on_a_line() {
    let mut line = Array2::<f64>::zeros((9, 2));
    for row in 0..line.nrows() {
        let x = row as f64 - 4.0;
        line[[row, 0]] = x;
        line[[row, 1]] = 2.0 * x;
    }
    let rows = (0..line.nrows()).collect::<Vec<_>>();
    let fit = GaussianFit2d::fit(line.view(), &rows).unwrap();
    let on_line = fit.log_density(0.0, 0.0);
    let off_line = fit.log_density(-0.02, 0.01);
    assert!(on_line.is_finite() && off_line.is_finite());
    assert!(
        off_line < on_line - 1.0e6,
        "the constrained covariance must penalize its null direction: on={on_line}, off={off_line}"
    );
    assert!(gaussian_bic_2d(line.view()).unwrap().is_finite());

    let coincident = Array2::<f64>::from_elem((5, 2), 3.0);
    let error = GaussianFit2d::fit(coincident.view(), &[0, 1, 2, 3, 4])
        .expect_err("a zero-dimensional point mass is not a 2-D Gaussian density");
    assert!(error.contains("zero"), "{error}");
}

#[test]
fn circular_verdict_aggregates_within_class_stacking_mass() {
    let kinds = [
        gam_solve::PredictiveCandidateKind::Fixed(gam_solve::AutoTopologyKind::Circle),
        gam_solve::PredictiveCandidateKind::Fixed(gam_solve::AutoTopologyKind::Euclidean),
        gam_solve::PredictiveCandidateKind::MixtureClass,
        gam_solve::PredictiveCandidateKind::RingOfClustersClass,
    ];
    // No individual circular candidate beats Euclidean (0.30 < 0.40), but
    // the circular topology class owns 0.60 total mass. A max-vs-max rule
    // would make the class verdict depend on how finely that class happened
    // to be represented in the candidate list.
    let (circular, noncircular, margin, circle_wins) =
        circular_stacking_summary(&kinds, &[0.30, 0.40, 0.0, 0.30]).unwrap();
    assert!((circular - 0.60).abs() < 1e-15);
    assert!((noncircular - 0.40).abs() < 1e-15);
    assert!((margin - 0.20).abs() < 1e-15);
    assert!(circle_wins);
}

#[test]
fn discrete_rung_orders_are_selected_inside_each_outer_training_fold() {
    use gam_solve::evidence::GaussianMixtureConfig;

    let train_rows = 90usize;
    let eval_rows = 18usize;
    let mut free_coords = Array2::<f64>::zeros((train_rows + eval_rows, 2));
    for row in 0..train_rows {
        let cluster = row % 2;
        let phase = std::f64::consts::TAU * (row / 2) as f64 / (train_rows / 2) as f64;
        free_coords[[row, 0]] = if cluster == 0 { -2.0 } else { 2.0 } + 0.12 * phase.cos();
        free_coords[[row, 1]] = 0.08 * phase.sin();
    }
    // These outer-held-out rows form a remote third cluster. If they leak
    // into order selection or parameter fitting, the returned density
    // cannot equal the independently fitted training-only prediction.
    for slot in 0..eval_rows {
        let row = train_rows + slot;
        let phase = std::f64::consts::TAU * slot as f64 / eval_rows as f64;
        free_coords[[row, 0]] = 40.0 + 0.2 * phase.cos();
        free_coords[[row, 1]] = -30.0 + 0.2 * phase.sin();
    }
    let train = (0..train_rows).collect::<Vec<_>>();
    let eval = (train_rows..train_rows + eval_rows).collect::<Vec<_>>();
    let config = GaussianMixtureConfig::default();
    let (train_coords, eval_coords, log_volume_scale) =
        canonical_shape_fold(free_coords.view(), &train, &eval, "test mixture").unwrap();
    let (train_selected_k, mut expected) = free_mixture_rung_predictive_density(
        train_coords.view(),
        eval_coords.view(),
        config,
    )
    .unwrap();
    for value in &mut expected {
        *value -= log_volume_scale;
    }
    let free_trace = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
    let actual = free_mixture_rung_provider_2d(
        free_coords.clone(),
        config,
        std::rc::Rc::clone(&free_trace),
    )(&train, &eval)
    .expect("outer provider must fit and select only on training rows");
    assert!(train_selected_k >= 2);
    assert_eq!(actual, expected);
    assert_eq!(&*free_trace.borrow(), &[train_selected_k]);

    // A held-out outlier may change its own predictive density, but it must
    // not change the training gauge, selected order, or any other held-out
    // row. The former full-data canonicalization failed this contract when
    // its absolute mixture covariance floor became active.
    let mut perturbed_free_coords = free_coords;
    let perturbed_slot = eval_rows - 1;
    perturbed_free_coords[[train_rows + perturbed_slot, 0]] = 1.0e12;
    perturbed_free_coords[[train_rows + perturbed_slot, 1]] = -1.0e12;
    let perturbed_free_trace = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
    let perturbed_actual = free_mixture_rung_provider_2d(
        perturbed_free_coords,
        config,
        std::rc::Rc::clone(&perturbed_free_trace),
    )(&train, &eval)
    .expect("held-out values must not alter mixture training preprocessing");
    assert_eq!(
        &perturbed_actual[..perturbed_slot],
        &actual[..perturbed_slot]
    );
    assert_eq!(&*perturbed_free_trace.borrow(), &[train_selected_k]);

    let clusters = 3usize;
    let per_cluster = 30usize;
    let mut ring_coords = Array2::<f64>::zeros((clusters * per_cluster + eval_rows, 2));
    for cluster in 0..clusters {
        let angle = std::f64::consts::TAU * cluster as f64 / clusters as f64;
        for sample in 0..per_cluster {
            let phase = std::f64::consts::TAU * sample as f64 / per_cluster as f64;
            let row = cluster * per_cluster + sample;
            ring_coords[[row, 0]] =
                (2.0 + 0.08 * phase.cos()) * angle.cos() - 0.05 * phase.sin() * angle.sin();
            ring_coords[[row, 1]] =
                (2.0 + 0.08 * phase.cos()) * angle.sin() + 0.05 * phase.sin() * angle.cos();
        }
    }
    for slot in 0..eval_rows {
        let row = clusters * per_cluster + slot;
        ring_coords[[row, 0]] = 25.0 + slot as f64;
        ring_coords[[row, 1]] = -20.0;
    }
    let ring_train = (0..clusters * per_cluster).collect::<Vec<_>>();
    let ring_eval =
        (clusters * per_cluster..clusters * per_cluster + eval_rows).collect::<Vec<_>>();
    let (ring_train_coords, ring_eval_coords, ring_log_volume_scale) = canonical_shape_fold(
        ring_coords.view(),
        &ring_train,
        &ring_eval,
        "test ring cluster",
    )
    .unwrap();
    let (ring_selected_k, mut ring_expected) = ring_cluster_rung_predictive_density(
        ring_train_coords.view(),
        ring_eval_coords.view(),
        config,
    )
    .unwrap();
    for value in &mut ring_expected {
        *value -= ring_log_volume_scale;
    }
    let ring_trace = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
    let ring_actual = ring_cluster_rung_provider_2d(
        ring_coords.clone(),
        config,
        std::rc::Rc::clone(&ring_trace),
    )(&ring_train, &ring_eval)
    .expect("ring provider must fit and select only on training rows");
    assert!(ring_selected_k >= 3);
    assert_eq!(ring_actual, ring_expected);
    assert_eq!(&*ring_trace.borrow(), &[ring_selected_k]);

    let mut perturbed_ring_coords = ring_coords;
    let perturbed_ring_slot = eval_rows - 1;
    perturbed_ring_coords[[clusters * per_cluster + perturbed_ring_slot, 0]] = -1.0e12;
    perturbed_ring_coords[[clusters * per_cluster + perturbed_ring_slot, 1]] = 1.0e12;
    let perturbed_ring_trace = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
    let perturbed_ring_actual = ring_cluster_rung_provider_2d(
        perturbed_ring_coords,
        config,
        std::rc::Rc::clone(&perturbed_ring_trace),
    )(&ring_train, &ring_eval)
    .expect("held-out values must not alter ring-cluster training preprocessing");
    assert_eq!(
        &perturbed_ring_actual[..perturbed_ring_slot],
        &ring_actual[..perturbed_ring_slot]
    );
    assert_eq!(&*perturbed_ring_trace.borrow(), &[ring_selected_k]);
}

/// #2262 rank-charge diagnostic: `reconstruction_rank_edge` surfaced by
/// `adjudicate_atom_shape` must be exactly the same closed-form MP edge
/// `mp_reconstruction_rank_edge` (and production reconstruction-Gram pricing)
/// compute — no independent reimplementation to drift out of sync.
#[test]
fn reconstruction_rank_edge_matches_closed_form_mp_edge_2262() {
    use crate::null_battery::mp_reconstruction_rank_edge;

    let n_eff = 84.0_f64;
    let ambient_p = 64.0_f64;
    let dispersion_r = 1.005_f64;
    let expected = dispersion_r * (1.0 + (ambient_p / n_eff).sqrt()).powi(2);
    let edge = mp_reconstruction_rank_edge(n_eff, ambient_p, dispersion_r)
        .expect("valid inputs must succeed");
    assert!(
        (edge - expected).abs() < 1e-12,
        "edge={edge} expected={expected}"
    );
    assert_eq!(
        shape_reconstruction_rank_edge(Some(n_eff), Some(ambient_p), Some(dispersion_r))
            .unwrap(),
        Some(expected)
    );
    assert_eq!(
        shape_reconstruction_rank_edge(None, None, None).unwrap(),
        None
    );
    for partial in [
        (Some(n_eff), None, None),
        (None, Some(ambient_p), None),
        (None, None, Some(dispersion_r)),
        (Some(n_eff), Some(ambient_p), None),
    ] {
        let error = shape_reconstruction_rank_edge(partial.0, partial.1, partial.2)
            .expect_err("a partial rank-charge diagnostic specification must be rejected");
        assert!(error.contains("supplied together"), "{error}");
    }
}
