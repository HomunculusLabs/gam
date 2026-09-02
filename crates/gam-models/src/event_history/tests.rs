use super::chain::{
    AtomTransition, GaussHermite, Grid, backward_axis_bases, forward_operators,
    interpolate_at_inner_points,
};
use super::cohort::{
    CovariateSegment, Event, EventHistoryCohort, MarkKind, SubjectHistory, SubjectNodes,
    design_rows, expand_nodes,
};
use super::family::{
    Directional, EventHistoryFamily, EventHistoryFit, EventHistorySpec, fit_event_history,
    fit_event_history_formula,
};
use super::forecast::{
    ForecastRequest, FutureSegment, PopulationForecastRequest, forecast,
    kolmogorov_smirnov_uniform, population_forecast, predictive_pit,
};
use super::marginal::{SubjectInputs, subject_marginal};
use crate::custom_family::{BlockwiseFitOptions, ParameterBlockState};
use gam_math::jet_scalar::{OneSeed, TwoSeed};
use gam_math::nested_dual::JetField;
use gam_terms::smooth::{
    LinearCoefficientGeometry, LinearTermSpec, TermCollectionSpec, build_term_collection_design,
};
use ndarray::{Array1, Array2, array};
use std::sync::Arc;

fn gaussian(x: f64, mean: f64, variance: f64) -> f64 {
    (-(x - mean).powi(2) / (2.0 * variance)).exp() / (2.0 * std::f64::consts::PI * variance).sqrt()
}

fn subject(times: &[f64], exposures: &[f64], counts: &[Vec<f64>]) -> SubjectNodes {
    let marks = counts[0].len();
    let mut matrix = Array2::<f64>::zeros((times.len(), marks));
    for (n, row) in counts.iter().enumerate() {
        for (d, &c) in row.iter().enumerate() {
            matrix[[n, d]] = c;
        }
    }
    let mut exposure_matrix = Array2::<f64>::zeros((times.len(), marks));
    for (n, &w) in exposures.iter().enumerate() {
        for d in 0..marks {
            exposure_matrix[[n, d]] = w;
        }
    }
    SubjectNodes {
        first_row: 0,
        times: times.to_vec(),
        gaps: times.windows(2).map(|w| w[1] - w[0]).collect(),
        weights: exposures.to_vec(),
        exposures: exposure_matrix,
        counts: matrix,
        covariate_rows: vec![0; times.len()],
    }
}

#[test]
fn forward_operator_is_exact_on_envelope_times_polynomial() {
    let gh = GaussHermite::new(15).expect("rule");
    let (mu, sigma) = (0.1, 0.9);
    let from = Grid::new(&gh, &[mu], &[sigma], &0.0);
    let to = Grid::new(&gh, &[0.4], &[0.5], &0.0);
    let kappa = 0.35;
    let transition = AtomTransition::new(&kappa);
    let phi = (-kappa).exp();
    let q = 1.0 - phi * phi;
    // f(z) = N(z; mu, sigma²) (1 + 0.3 z + 0.2 z²)
    let values: Vec<f64> = (0..from.size())
        .map(|i| {
            let z = *from.coordinate(i, 0);
            gaussian(z, mu, sigma * sigma) * (1.0 + 0.3 * z + 0.2 * z * z)
        })
        .collect();
    let forward = forward_operators(&gh, &from, &to, &[transition.clone()], 0);
    let predicted = forward.plain(&values);
    let tau2 = phi * phi * sigma * sigma + q;
    let s2 = sigma * sigma * q / tau2;
    for j in 0..to.size() {
        let z = *to.coordinate(j, 0);
        let m = mu + phi * sigma * sigma * (z - phi * mu) / tau2;
        let exact = gaussian(z, phi * mu, tau2) * (1.0 + 0.3 * m + 0.2 * (m * m + s2));
        assert!(
            (predicted[j] - exact).abs() < 1e-8 * exact.abs().max(1e-8),
            "node {j}: predicted {} exact {exact}",
            predicted[j]
        );
    }
    // A Gaussian of a different centre and width is not envelope × polynomial;
    // the interpolant is then an approximation, accurate to the interpolation
    // error of a degree-14 polynomial.
    let (mu0, sigma0) = (0.3, 0.7);
    let other: Vec<f64> = (0..from.size())
        .map(|i| gaussian(*from.coordinate(i, 0), mu0, sigma0 * sigma0))
        .collect();
    let predicted = forward.plain(&other);
    for j in 0..to.size() {
        let z = *to.coordinate(j, 0);
        let exact = gaussian(z, phi * mu0, phi * phi * sigma0 * sigma0 + q);
        assert!(
            (predicted[j] - exact).abs() < 1e-4 * exact.abs().max(1e-3),
            "node {j}: predicted {} exact {exact}",
            predicted[j]
        );
    }
    // The backward interpolation reproduces a constant at every inner point
    // and a linear function wherever the inner point lies inside the hull.
    let bases = backward_axis_bases(&gh, &from, &to, &[transition]);
    let ones = vec![1.0; to.size()];
    let at_inner = interpolate_at_inner_points(gh.order, &bases, &ones);
    assert_eq!(at_inner.len(), from.size() * to.size());
    for (i, v) in at_inner.iter().enumerate() {
        assert!((v - 1.0).abs() < 1e-9, "constant at inner point {i}: {v}");
    }
    let linear: Vec<f64> = (0..to.size()).map(|j| 0.5 + 0.3 * to.coordinate(j, 0)).collect();
    let at_inner = interpolate_at_inner_points(gh.order, &bases, &linear);
    let spread = (2.0 * q).sqrt();
    let hull = gh.nodes[gh.order - 1] * std::f64::consts::SQRT_2 * to.axes[0].sigma;
    for i in 0..from.size() {
        for (l, &x) in gh.nodes.iter().enumerate() {
            let zeta = phi * from.coordinate(i, 0) + spread * x;
            if (zeta - to.axes[0].mu).abs() < hull {
                let exact = 0.5 + 0.3 * zeta;
                let got = at_inner[i * to.size() + l];
                assert!((got - exact).abs() < 1e-9, "linear at ({i}, {l}): {got} vs {exact}");
            }
        }
    }
}

#[test]
fn transition_polynomials_are_exact_scores_of_the_log_density() {
    // Finite differences are permitted in tests: the gap polynomial must
    // equal the derivative of the log transition density in log-rate.
    let z = 0.4;
    let zp = -0.2;
    let gap = 0.6;
    let log_density = |rho: f64| {
        let phi = (-(rho.exp() * gap)).exp();
        let v = 1.0 - phi * phi;
        -0.5 * (2.0 * std::f64::consts::PI * v).ln() - (zp - phi * z).powi(2) / (2.0 * v)
    };
    let rho = -0.3;
    let h = 1e-5;
    let fd1 = (log_density(rho + h) - log_density(rho - h)) / (2.0 * h);
    let fd2 = (log_density(rho + h) - 2.0 * log_density(rho) + log_density(rho - h)) / (h * h);
    let kappa = rho.exp() * gap;
    let (t, dt) = super::marginal::transition_score_polynomials(kappa);
    let phi = (-kappa).exp();
    let u = (zp - phi * z) / (1.0 - phi * phi).sqrt();
    let evaluate = |c: &[f64]| -> f64 {
        let mut total = 0.0;
        for a in 0..5 {
            for b in 0..5 {
                total += c[a * 5 + b] * z.powi(a as i32) * u.powi(b as i32);
            }
        }
        total
    };
    assert!((evaluate(&t) - fd1).abs() < 1e-7, "score {} vs fd {fd1}", evaluate(&t));
    assert!(
        (evaluate(&dt) - fd2).abs() < 1e-5,
        "score derivative {} vs fd {fd2}",
        evaluate(&dt)
    );
}

#[test]
fn single_node_marginal_matches_numerical_integration() {
    let gh = GaussHermite::new(41).expect("rule");
    let nodes = subject(&[1.0], &[0.8], &[vec![2.0]]);
    let eta0 = [0.3];
    let loadings = [0.9];
    let log_rates = [0.0];
    let inputs = SubjectInputs {
        nodes: &nodes,
        eta0: &eta0,
        loadings: &loadings,
        log_rates: &log_rates,
        time_scale: 1.0,
        gh: &gh,
        continuation_gap: 0.0,
    designs: None,
    };
    let out = subject_marginal(&inputs, false).expect("marginal");
    // ∫ exp(y η − w e^η) N(z) dz with η = η0 − ½a² + a z, on a fine grid.
    let mut integral = 0.0;
    let steps = 200_000;
    let (lo, hi) = (-9.0, 9.0);
    let dz = (hi - lo) / steps as f64;
    for i in 0..=steps {
        let z = lo + i as f64 * dz;
        let eta = eta0[0] - 0.5 * loadings[0] * loadings[0] + loadings[0] * z;
        let weight = if i == 0 || i == steps { 0.5 } else { 1.0 };
        integral += weight * dz * (2.0 * eta - 0.8 * eta.exp()).exp() * gaussian(z, 0.0, 1.0);
    }
    assert!(
        (out.loglik - integral.ln()).abs() < 1e-8,
        "filter {} vs quadrature {}",
        out.loglik,
        integral.ln()
    );
}

#[test]
fn two_node_marginal_matches_brute_force_double_integral() {
    let gh = GaussHermite::new(25).expect("rule");
    let nodes = subject(&[0.0, 0.7], &[0.5, 0.6], &[vec![1.0], vec![0.0]]);
    let eta0 = [-0.2, 0.4];
    let loadings = [1.1];
    let log_rates = [0.2];
    let inputs = SubjectInputs {
        nodes: &nodes,
        eta0: &eta0,
        loadings: &loadings,
        log_rates: &log_rates,
        time_scale: 1.0,
        gh: &gh,
        continuation_gap: 0.0,
    designs: None,
    };
    let out = subject_marginal(&inputs, false).expect("marginal");
    let phi = (-(0.2f64.exp() * 0.7)).exp();
    let q = 1.0 - phi * phi;
    let steps = 1200;
    let (lo, hi) = (-7.0, 7.0);
    let dz = (hi - lo) / steps as f64;
    let mut integral = 0.0;
    for i in 0..=steps {
        let z1 = lo + i as f64 * dz;
        let w1 = if i == 0 || i == steps { 0.5 } else { 1.0 };
        let shift = -0.5 * loadings[0] * loadings[0];
        let eta1 = eta0[0] + shift + loadings[0] * z1;
        let l1 = (eta1 - 0.5 * eta1.exp()).exp() * gaussian(z1, 0.0, 1.0);
        let mut inner = 0.0;
        for j in 0..=steps {
            let z2 = lo + j as f64 * dz;
            let w2 = if j == 0 || j == steps { 0.5 } else { 1.0 };
            let eta2 = eta0[1] + shift + loadings[0] * z2;
            inner += w2 * dz * (-0.6 * eta2.exp()).exp() * gaussian(z2, phi * z1, q);
        }
        integral += w1 * dz * l1 * inner;
    }
    assert!(
        (out.loglik - integral.ln()).abs() < 1e-6,
        "filter {} vs brute force {}",
        out.loglik,
        integral.ln()
    );
}

#[test]
fn zero_loadings_reduce_to_the_poisson_likelihood() {
    let gh = GaussHermite::new(21).expect("rule");
    let nodes = subject(
        &[0.0, 0.5, 1.5],
        &[0.4, 0.0, 0.7],
        &[vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 2.0]],
    );
    let eta0 = [0.1, -0.3, 0.5, 0.2, -0.1, 0.4];
    let loadings = [0.0, 0.0, 0.0, 0.0];
    let log_rates = [0.0, 0.3];
    let inputs = SubjectInputs {
        nodes: &nodes,
        eta0: &eta0,
        loadings: &loadings,
        log_rates: &log_rates,
        time_scale: 1.0,
        gh: &gh,
        continuation_gap: 0.0,
    designs: None,
    };
    let out = subject_marginal(&inputs, true).expect("marginal");
    let mut expected = 0.0;
    for n in 0..3 {
        for d in 0..2 {
            let eta = eta0[n * 2 + d];
            expected += nodes.counts[[n, d]] * eta - nodes.exposures[[n, d]] * eta.exp();
        }
    }
    assert!((out.loglik - expected).abs() < 1e-12);
    for n in 0..3 {
        for d in 0..2 {
            let eta = eta0[n * 2 + d];
            let score = nodes.counts[[n, d]] - nodes.exposures[[n, d]] * eta.exp();
            // The smoothed marginal drops tail mass below the density noise
            // floor (1e-11 of the peak), which is the agreement limit here.
            assert!(
                (out.gradient[n * 2 + d] - score).abs() < 1e-9,
                "gradient {} vs score {score}",
                out.gradient[n * 2 + d]
            );
        }
    }
    // With zero loadings the rates are unidentified: their gradient is zero
    // up to the hull clamping of the backward quadrature.
    for k in 0..2 {
        assert!(
            out.gradient[6 + 4 + k].abs() < 1e-8,
            "rate gradient {}",
            out.gradient[6 + 4 + k]
        );
    }
}

fn finite_difference_subject() -> (SubjectNodes, Vec<f64>, Vec<f64>, Vec<f64>) {
    let nodes = subject(
        &[0.0, 0.4, 1.1, 1.6],
        &[0.3, 0.0, 0.5, 0.2],
        &[
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
        ],
    );
    let eta0 = vec![0.2, -0.1, 0.4, 0.0, -0.3, 0.5, 0.1, 0.2];
    let loadings = vec![0.8, -0.4, 0.3, 0.6];
    let log_rates = vec![-0.2, 0.5];
    (nodes, eta0, loadings, log_rates)
}

fn evaluate_at(
    nodes: &SubjectNodes,
    gh: &GaussHermite,
    theta: &[f64],
    derivatives: bool,
) -> super::marginal::SubjectOutput<f64> {
    let n = nodes.len();
    let (marks, atoms) = (2, 2);
    let eta0 = &theta[0..n * marks];
    let loadings = &theta[n * marks..n * marks + marks * atoms];
    let log_rates = &theta[n * marks + marks * atoms..];
    subject_marginal(
        &SubjectInputs {
            nodes,
            eta0,
            loadings,
            log_rates,
            time_scale: 1.0,
            gh,
            continuation_gap: 0.0,
        designs: None,
        },
        derivatives,
    )
    .expect("marginal")
}

#[test]
fn gradient_and_hessian_match_central_differences() {
    let gh = GaussHermite::new(31).expect("rule");
    let (nodes, eta0, loadings, log_rates) = finite_difference_subject();
    let mut theta = eta0.clone();
    theta.extend(loadings.iter());
    theta.extend(log_rates.iter());
    let p = theta.len();
    let base = evaluate_at(&nodes, &gh, &theta, true);
    let h = 1e-4;
    for i in 0..p {
        let mut plus = theta.clone();
        plus[i] += h;
        let mut minus = theta.clone();
        minus[i] -= h;
        let fp = evaluate_at(&nodes, &gh, &plus, true);
        let fm = evaluate_at(&nodes, &gh, &minus, true);
        let fd = (fp.loglik - fm.loglik) / (2.0 * h);
        assert!(
            (base.gradient[i] - fd).abs() < 1e-6 * (1.0 + fd.abs()),
            "gradient[{i}] = {} vs finite difference {fd}",
            base.gradient[i]
        );
        for j in 0..p {
            let fd_h = (fp.gradient[j] - fm.gradient[j]) / (2.0 * h);
            // Louis' Hessian interpolates the smoother residual with a cubic
            // spline; it agrees with the finite difference of the gradient to
            // that interpolation error.
            assert!(
                (base.hessian[i * p + j] - fd_h).abs() < 1e-3 * (1.0 + fd_h.abs()),
                "hessian[{i},{j}] = {} vs finite difference {fd_h}",
                base.hessian[i * p + j]
            );
            assert!((base.hessian[i * p + j] - base.hessian[j * p + i]).abs() < 1e-9);
        }
    }
}

#[test]
fn directional_duals_match_finite_differences_of_the_hessian() {
    let gh = GaussHermite::new(21).expect("rule");
    let (nodes, eta0, loadings, log_rates) = finite_difference_subject();
    let mut theta = eta0.clone();
    theta.extend(loadings.iter());
    theta.extend(log_rates.iter());
    let p = theta.len();
    let u: Vec<f64> = (0..p).map(|i| 0.3 * ((i as f64) * 0.7).sin() + 0.1).collect();
    let v: Vec<f64> = (0..p).map(|i| 0.2 * ((i as f64) * 1.3).cos() - 0.05).collect();
    let seed = |theta: &[f64], u: &[f64], v: &[f64]| -> Vec<TwoSeed<0>> {
        theta
            .iter()
            .zip(u.iter().zip(v.iter()))
            .map(|(&x, (&du, &dv))| super::family::seeded_two(x, du, dv))
            .collect()
    };
    let n = nodes.len();
    let (marks, atoms) = (2, 2);
    let evaluate_two = |theta: &[f64]| {
        let jets = seed(theta, &u, &v);
        subject_marginal(
            &SubjectInputs {
                nodes: &nodes,
                eta0: &jets[0..n * marks],
                loadings: &jets[n * marks..n * marks + marks * atoms],
                log_rates: &jets[n * marks + marks * atoms..],
                time_scale: 1.0,
                gh: &gh,
                continuation_gap: 0.0,
            designs: None,
            },
            true,
        )
        .expect("marginal")
    };
    let two = evaluate_two(&theta);
    let h = 1e-4;
    let shifted = |s: f64, dir: &[f64]| -> Vec<f64> {
        theta.iter().zip(dir.iter()).map(|(x, d)| x + s * d).collect()
    };
    let plus_u = evaluate_at(&nodes, &gh, &shifted(h, &u), true);
    let minus_u = evaluate_at(&nodes, &gh, &shifted(-h, &u), true);
    for idx in 0..p * p {
        let fd = (plus_u.hessian[idx] - minus_u.hessian[idx]) / (2.0 * h);
        let dual = two.hessian[idx].eps.value();
        assert!(
            (dual - fd).abs() < 2e-5 * (1.0 + fd.abs()),
            "D H[u] at {idx}: dual {dual} vs finite difference {fd}"
        );
    }
    // Mixed second derivative: difference of the u-directional derivative along v.
    let one_at = |theta: &[f64]| -> Vec<OneSeed<0>> {
        let jets: Vec<OneSeed<0>> = theta
            .iter()
            .zip(u.iter())
            .map(|(&x, &du)| super::family::seeded_one(x, du))
            .collect();
        let out = subject_marginal(
            &SubjectInputs {
                nodes: &nodes,
                eta0: &jets[0..n * marks],
                loadings: &jets[n * marks..n * marks + marks * atoms],
                log_rates: &jets[n * marks + marks * atoms..],
                time_scale: 1.0,
                gh: &gh,
                continuation_gap: 0.0,
            designs: None,
            },
            true,
        )
        .expect("marginal");
        out.hessian
    };
    let plus_v = one_at(&shifted(h, &v));
    let minus_v = one_at(&shifted(-h, &v));
    for idx in 0..p * p {
        let fd = (plus_v[idx].eps.value() - minus_v[idx].eps.value()) / (2.0 * h);
        let dual = two.hessian[idx].eps_del.value();
        assert!(
            (dual - fd).abs() < 5e-5 * (1.0 + fd.abs()),
            "D²H[u,v] at {idx}: dual {dual} vs finite difference {fd}"
        );
    }
}

#[test]
fn node_expansion_integrates_exposure_and_places_events() {
    let mut cohort = EventHistoryCohort {
        mark_names: vec!["a".to_string(), "b".to_string(), "c".to_string()],
        mark_kinds: vec![MarkKind::Recurrent, MarkKind::Once, MarkKind::Terminal],
        covariate_names: vec!["x".to_string()],
        covariate_levels: vec![Vec::new()],
        covariates: array![[1.0], [2.0], [3.0]],
        subjects: vec![SubjectHistory {
            id: "s".to_string(),
            entry: 1.0,
            exit: 4.0,
            events: vec![
                Event {
                    time: 2.5,
                    mark: 1,
                },
                Event {
                    time: 2.5,
                    mark: 0,
                },
                Event {
                    time: 4.0,
                    mark: 2,
                },
            ],
            segments: vec![
                CovariateSegment {
                    start: 0.0,
                    row: 0,
                },
                // A covariate change at the instant of the event: the event
                // node sees the left limit, the quadrature after it the new row.
                CovariateSegment {
                    start: 2.5,
                    row: 2,
                },
                CovariateSegment {
                    start: 3.0,
                    row: 1,
                },
            ],
        }],
    };
    cohort.validate().expect("valid");
    let nodes = expand_nodes(&cohort, 5, 0).expect("nodes");
    let s = &nodes.subjects[0];
    let total_weight: f64 = s.weights.iter().sum();
    assert!((total_weight - 3.0).abs() < 1e-12, "weight {total_weight}");
    // Recurrent and terminal marks are at risk throughout; the once-only
    // mark leaves the risk set after 2.5.
    let exposure = |d: usize| -> f64 { s.exposures.column(d).sum() };
    assert!((exposure(0) - 3.0).abs() < 1e-12);
    assert!((exposure(2) - 3.0).abs() < 1e-12);
    assert!((exposure(1) - 1.5).abs() < 1e-12, "once-only exposure {}", exposure(1));
    let event_node = s.times.iter().position(|&t| t == 2.5).expect("event node");
    assert_eq!(s.counts[[event_node, 0]], 1.0);
    assert_eq!(s.counts[[event_node, 1]], 1.0);
    assert_eq!(s.weights[event_node], 0.0);
    assert_eq!(s.covariate_rows[event_node], 0, "an event node takes the left limit");
    let terminal_node = s.len() - 1;
    assert_eq!(s.times[terminal_node], 4.0);
    assert_eq!(s.counts[[terminal_node, 2]], 1.0);
    assert!(s.gaps.iter().all(|&g| g > 0.0));
    for (n, &t) in s.times.iter().enumerate() {
        if n == event_node {
            continue;
        }
        let expected_row = if t >= 3.0 { 1 } else if t > 2.5 { 2 } else { 0 };
        assert_eq!(s.covariate_rows[n], expected_row, "node at {t}");
        assert_eq!(nodes.node_data[[n, 0]], cohort.covariates[[expected_row, 0]]);
        assert_eq!(nodes.node_data[[n, 1]], t);
    }
    // Every mesh cell is halved at refinement one: twice the quadrature
    // nodes, the same total weight, the same events.
    let refined = expand_nodes(&cohort, 5, 1).expect("refined nodes");
    let r = &refined.subjects[0];
    let quadrature_nodes = |s: &SubjectNodes| s.weights.iter().filter(|w| **w > 0.0).count();
    assert_eq!(quadrature_nodes(r), 2 * quadrature_nodes(s));
    assert!((r.weights.iter().sum::<f64>() - 3.0).abs() < 1e-12);
    assert_eq!(r.counts.sum(), s.counts.sum());
    // The design rows carry entry, exit, covariate changes and an
    // event-free quadrature. They are a function of the design alone: the
    // same subject with no events at all produces the identical rows, so no
    // data-adaptive basis can be shaped by where the events fell.
    let rows = design_rows(&cohort, 5).expect("design rows");
    let times: Vec<f64> = rows.column(1).to_vec();
    assert!(times.contains(&1.0) && times.contains(&4.0) && times.contains(&3.0));
    let interior = times.iter().filter(|t| **t > 1.0 && **t < 4.0 && **t != 3.0 && **t != 2.5).count();
    assert_eq!(interior, 15, "three event-free cells of five nodes");
    let mut event_free = cohort.clone();
    event_free.subjects[0].events.clear();
    event_free.subjects[0].exit = 4.0;
    event_free.validate().expect("valid without events");
    let without = design_rows(&event_free, 5).expect("design rows");
    assert_eq!(rows, without, "an event time must not enter a basis");
}

#[test]
fn validation_rejects_ill_formed_cohorts() {
    let base = || EventHistoryCohort {
        mark_names: vec!["relapse".to_string(), "death".to_string()],
        mark_kinds: vec![MarkKind::Recurrent, MarkKind::Terminal],
        covariate_names: vec!["x".to_string(), "arm".to_string()],
        covariate_levels: vec![Vec::new(), vec!["control".to_string(), "treated".to_string()]],
        covariates: array![[0.3, 0.0], [-0.1, 1.0]],
        subjects: vec![SubjectHistory {
            id: "a".to_string(),
            entry: 0.0,
            exit: 5.0,
            events: vec![Event { time: 2.0, mark: 0 }],
            segments: vec![CovariateSegment { start: 0.0, row: 0 }],
        }],
    };
    let mut ok = base();
    ok.validate().expect("the base cohort is valid");
    let expect_error = |mutate: &dyn Fn(&mut EventHistoryCohort), needle: &str| {
        // Each mutation starts from the same valid cohort.
        let mut cohort = base();
        mutate(&mut cohort);
        let error = cohort.validate().err().unwrap_or_else(|| panic!("expected an error containing {needle:?}"));
        assert!(error.to_string().contains(needle), "{error} lacks {needle:?}");
    };
    expect_error(&|c: &mut EventHistoryCohort| { let first = c.subjects[0].clone(); c.subjects.push(first); }, "duplicate subject identifier");
    expect_error(&|c: &mut EventHistoryCohort| c.subjects[0].segments.push(CovariateSegment { start: 0.0, row: 1 }), "two covariate segments starting at");
    expect_error(&|c: &mut EventHistoryCohort| c.subjects[0].segments.push(CovariateSegment { start: 7.0, row: 1 }), "outside (entry, exit)");
    expect_error(&|c: &mut EventHistoryCohort| c.subjects[0].events.push(Event { time: 3.0, mark: 1 }), "must end follow-up");
    // One mark firing twice is caught as that mark's own rule; two DIFFERENT
    // terminal marks each firing once is the case the follow-up rule catches.
    expect_error(
        &|c: &mut EventHistoryCohort| {
            c.subjects[0].events.push(Event { time: 3.0, mark: 1 });
            c.subjects[0].events.push(Event { time: 5.0, mark: 1 });
        },
        "can fire at most once",
    );
    expect_error(
        &|c: &mut EventHistoryCohort| {
            c.mark_kinds[0] = MarkKind::Terminal;
            c.subjects[0].events.push(Event { time: 3.0, mark: 1 });
        },
        "terminal events",
    );
    expect_error(
        &|c: &mut EventHistoryCohort| {
            c.mark_kinds[0] = MarkKind::Once;
            c.subjects[0].events.push(Event { time: 3.0, mark: 0 });
        },
        "can fire at most once",
    );
    expect_error(&|c: &mut EventHistoryCohort| c.covariates[[0, 1]] = 2.0, "categorical covariate");
    expect_error(&|c: &mut EventHistoryCohort| c.covariates[[0, 1]] = 0.5, "categorical covariate");
    expect_error(&|c: &mut EventHistoryCohort| c.mark_names[1] = "relapse".to_string(), "duplicate mark name");
    expect_error(&|c: &mut EventHistoryCohort| c.mark_kinds.truncate(1), "mark kinds");
    // A terminal event at exit is valid.
    let mut terminal = base();
    terminal.subjects[0].events.push(Event { time: 5.0, mark: 1 });
    terminal.validate().expect("a terminal event at exit ends follow-up");
    assert!(terminal.subjects[0].terminal_event(&terminal.mark_kinds).is_some());
}

struct Rng(u64);
impl Rng {
    fn uniform(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
    fn normal(&mut self) -> f64 {
        let u1 = self.uniform().max(1e-300);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

/// Simulate a marked cohort whose log-intensity of mark `d` is
/// `intercept_d + slope · x + loading_d · z(t)` with `z` a unit-variance
/// Ornstein–Uhlenbeck atom of the given rate, sampled exactly at the steps
/// of a fine grid and held constant between them. Conditional on the path,
/// each step's events are Poisson with the step's exact integrated
/// intensity, placed uniformly in the step: no thinning bound is needed, so
/// nothing is approximate beyond the piecewise-constant path itself, whose
/// error vanishes with the step. A terminal mark ends follow-up at its
/// event; a once-only mark leaves the risk set after its event.
fn simulate_marked_cohort(
    subjects: usize,
    follow_up: f64,
    intercepts: &[f64],
    slope: f64,
    loadings: &[f64],
    rate: f64,
    kinds: &[MarkKind],
    seed: u64,
) -> EventHistoryCohort {
    let marks = intercepts.len();
    let mut rng = Rng(seed);
    let steps = 400;
    let dt = follow_up / steps as f64;
    let phi = (-rate * dt).exp();
    let innovation = (1.0 - phi * phi).sqrt();
    let mut covariates = Array2::<f64>::zeros((subjects, 1));
    let mut histories = Vec::with_capacity(subjects);
    for s in 0..subjects {
        let x = rng.normal();
        covariates[[s, 0]] = x;
        let mut z = rng.normal();
        let mut events: Vec<Event> = Vec::new();
        let mut exit = follow_up;
        let mut at_risk = vec![true; marks];
        'steps: for step in 0..steps {
            let left = step as f64 * dt;
            let mut step_events: Vec<Event> = Vec::new();
            for d in 0..marks {
                if !at_risk[d] {
                    continue;
                }
                let mean = (intercepts[d] + slope * x + loadings[d] * z).exp() * dt;
                // Poisson(mean) by inversion of its cumulative sum.
                let threshold = rng.uniform();
                let mut count = 0usize;
                let mut term = (-mean).exp();
                let mut cumulative = term;
                while threshold > cumulative && count < 1000 {
                    count += 1;
                    term *= mean / count as f64;
                    cumulative += term;
                }
                for _ in 0..count {
                    step_events.push(Event {
                        time: left + dt * rng.uniform(),
                        mark: d,
                    });
                }
            }
            step_events.sort_by(|a, b| a.time.total_cmp(&b.time));
            for event in step_events {
                if !at_risk[event.mark] {
                    continue;
                }
                match kinds[event.mark] {
                    MarkKind::Recurrent => events.push(event),
                    MarkKind::Once => {
                        at_risk[event.mark] = false;
                        events.push(event);
                    }
                    MarkKind::Terminal => {
                        exit = event.time;
                        events.push(event);
                        break 'steps;
                    }
                }
            }
            z = phi * z + innovation * rng.normal();
        }
        // Distinct event times: a tie at the step resolution is resolved by
        // the sort, and an event at exactly zero is impossible.
        events.retain(|e| e.time > 0.0 && e.time <= exit);
        histories.push(SubjectHistory {
            id: format!("s{s}"),
            entry: 0.0,
            exit,
            events,
            segments: vec![CovariateSegment {
                start: 0.0,
                row: s,
            }],
        });
    }
    EventHistoryCohort {
        mark_names: (0..marks).map(|d| format!("mark{d}")).collect(),
        mark_kinds: kinds.to_vec(),
        covariate_names: vec!["x".to_string()],
        covariate_levels: vec![Vec::new()],
        covariates,
        subjects: histories,
    }
}

/// The single recurrent mark case of [`simulate_marked_cohort`].
fn simulate_cohort(
    subjects: usize,
    follow_up: f64,
    intercept: f64,
    slope: f64,
    loading: f64,
    rate: f64,
    seed: u64,
) -> EventHistoryCohort {
    let mut cohort = simulate_marked_cohort(
        subjects,
        follow_up,
        &[intercept],
        slope,
        &[loading],
        rate,
        &[MarkKind::Recurrent],
        seed,
    );
    cohort.mark_names = vec!["event".to_string()];
    cohort
}

fn linear_spec() -> TermCollectionSpec {
    TermCollectionSpec {
        linear_terms: vec![LinearTermSpec {
            name: "x".to_string(),
            feature_col: 0,
            feature_cols: vec![0],
            categorical_levels: Vec::new(),
            double_penalty: false,
            coefficient_geometry: LinearCoefficientGeometry::Unconstrained,
            coefficient_min: None,
            coefficient_max: None,
            frozen_function_mass: None,
        }],
        random_effect_terms: Vec::new(),
        smooth_terms: Vec::new(),
    }
}

#[test]
fn family_joint_hessian_matches_finite_differences_of_its_gradient() {
    let mut cohort = simulate_cohort(6, 3.0, -0.5, 0.4, 0.8, 0.5, 11);
    cohort.validate().expect("valid");
    let nodes = Arc::new(expand_nodes(&cohort, 3, 0).expect("nodes"));
    let total = nodes.total_nodes;
    let mut design = Array2::<f64>::zeros((total, 2));
    for row in 0..total {
        design[[row, 0]] = 1.0;
        design[[row, 1]] = nodes.node_data[[row, 0]];
    }
    let family = EventHistoryFamily::new(
        Arc::clone(&nodes),
        vec![Arc::new(design.clone())],
        1,
        31,
        cohort.time_scale(),
    )
    .expect("family");
    let beta = array![-0.4, 0.3];
    let latent = array![0.5, -0.2];
    let states = |beta: &Array1<f64>, latent: &Array1<f64>| {
        vec![
            ParameterBlockState {
                beta: beta.clone(),
                eta: design.dot(beta),
            },
            ParameterBlockState {
                beta: latent.clone(),
                eta: Array1::zeros(total),
            },
        ]
    };
    let base = family.joint_evaluation(&states(&beta, &latent)).expect("joint");
    let p = beta.len() + latent.len();
    let h = 1e-4;
    for i in 0..p {
        let mut plus_beta = beta.clone();
        let mut plus_latent = latent.clone();
        let mut minus_beta = beta.clone();
        let mut minus_latent = latent.clone();
        if i < beta.len() {
            plus_beta[i] += h;
            minus_beta[i] -= h;
        } else {
            plus_latent[i - beta.len()] += h;
            minus_latent[i - beta.len()] -= h;
        }
        let plus = family
            .joint_evaluation(&states(&plus_beta, &plus_latent))
            .expect("joint");
        let minus = family
            .joint_evaluation(&states(&minus_beta, &minus_latent))
            .expect("joint");
        let fd = (plus.log_likelihood - minus.log_likelihood) / (2.0 * h);
        assert!(
            (base.gradient[i] - fd).abs() < 1e-6 * (1.0 + fd.abs()),
            "gradient[{i}] {} vs {fd}",
            base.gradient[i]
        );
        for j in 0..p {
            let fd_h = -(plus.gradient[j] - minus.gradient[j]) / (2.0 * h);
            // Spline-interpolated smoother residual: agreement to its
            // interpolation error, not to roundoff.
            assert!(
                (base.hessian[[i, j]] - fd_h).abs() < 1e-3 * (1.0 + fd_h.abs()),
                "hessian[{i},{j}] {} vs {fd_h}",
                base.hessian[[i, j]]
            );
        }
    }
    // Directional derivative of the negative Hessian against finite differences.
    let u = array![0.2, -0.1, 0.3, 0.15];
    let du = family
        .directional_hessian(&states(&beta, &latent), &u)
        .expect("directional");
    let plus = family
        .joint_evaluation(&states(
            &(beta.clone() + &u.slice(ndarray::s![0..2]).to_owned().mapv(|x| x * h)),
            &(latent.clone() + &u.slice(ndarray::s![2..4]).to_owned().mapv(|x| x * h)),
        ))
        .expect("joint");
    let minus = family
        .joint_evaluation(&states(
            &(beta.clone() - &u.slice(ndarray::s![0..2]).to_owned().mapv(|x| x * h)),
            &(latent.clone() - &u.slice(ndarray::s![2..4]).to_owned().mapv(|x| x * h)),
        ))
        .expect("joint");
    for i in 0..p {
        for j in 0..p {
            let fd = (plus.hessian[[i, j]] - minus.hessian[[i, j]]) / (2.0 * h);
            assert!(
                (du[[i, j]] - fd).abs() < 2e-5 * (1.0 + fd.abs()),
                "D H[u][{i},{j}] {} vs {fd}",
                du[[i, j]]
            );
        }
    }
}

#[test]
fn fit_recovers_the_covariate_effect_and_a_positive_shared_risk_loading() {
    install_test_logger();
    let started = std::time::Instant::now();
    let mut cohort = simulate_cohort(80, 6.0, -0.8, 0.5, 1.0, 0.4, 7);
    let mut spec = EventHistorySpec::new(1, vec![linear_spec()]);
    spec.gauss_hermite_order = 11;
    let fit = fit_event_history(&mut cohort, &spec).expect("fit");
    let beta = fit.mark_coefficients(0);
    println!(
        "[fit] {:.1}s outer_iterations={} gh_order={} beta={:?} loading={} rate={} log_lambda={:?}",
        started.elapsed().as_secs_f64(),
        fit.fit.outer_iterations,
        fit.quadrature.gauss_hermite_order,
        beta.to_vec(),
        fit.loadings[[0, 0]],
        fit.rates[0],
        fit.atom_log_lambdas
    );
    assert_eq!(beta.len(), 2, "intercept and slope");
    assert!(
        (beta[1] - 0.5).abs() < 0.25,
        "slope {} should recover 0.5 within its sampling error",
        beta[1]
    );
    // The simulated intensity `exp(β₀ + β₁x + a z)` averages over the
    // stationary `z` to `exp(β₀ + ½a² + β₁x)`, and the fitted `η⁰` is
    // parameterised as that average, so the intercept estimates the
    // population log-rate `−0.8 + ½·1² = −0.3`, not the conditional `−0.8`.
    let population_log_rate = -0.8 + 0.5;
    assert!(
        (beta[0] - population_log_rate).abs() < 0.25,
        "intercept {} should be the population log-rate {population_log_rate}",
        beta[0]
    );
    let loading = fit.loadings[[0, 0]].abs();
    assert!(
        loading > 0.4,
        "a shared dynamic risk was simulated but the fitted loading is {loading}"
    );
    // A fit object only exists from a converged optimisation; its certificate
    // gradient, when reported, is finite.
    assert!(fit.fit.outer_gradient_norm.is_none_or(|g| g.is_finite()));
    assert!(fit.rates[0] > 0.0 && fit.rates[0].is_finite());
    // The certificate: the fitted coefficients are stationary under a
    // doubling of the Gauss-Hermite order and a halving of the mesh, to the
    // stated fraction of their posterior standard deviation.
    // The certificate: at the fitted smoothing parameters, doubling the
    // Gauss-Hermite order and halving the mesh each move the coefficients by
    // less than the tolerance, in posterior standard deviations.
    assert!(fit.quadrature.gauss_hermite.coefficient_shift <= spec.quadrature_tolerance);
    assert!(fit.quadrature.mesh.coefficient_shift <= spec.quadrature_tolerance);
    assert_eq!(fit.quadrature.gauss_hermite.candidate, 2 * fit.quadrature.gauss_hermite_order - 1);
    assert_eq!(fit.quadrature.mesh.candidate, fit.quadrature.mesh_refinement + 1);
    // Forecast: probabilities and expected counts are coherent.
    let request = ForecastRequest {
        history: &cohort.subjects[0],
        horizons: &[6.5, 7.0, 8.0],
        future: &[],
    };
    let f = forecast(&fit, &cohort, &request).expect("forecast");
    assert!(
        f.survival.iter().all(|&s| (0.0..=1.0).contains(&s)),
        "survival left [0, 1]: {:?}",
        f.survival
    );
    assert!((f.survival[0] - 1.0).abs() < 1e-12, "no terminal marks: survival stays one");
    assert!(f.expected_counts[[0, 0]] <= f.expected_counts[[1, 0]]);
    assert!(f.expected_counts[[1, 0]] <= f.expected_counts[[2, 0]]);
    assert!(f.expected_counts[[2, 0]] > 0.0);
    // Predictive PIT: uniform up to sampling error on the training cohort.
    let mut pits = Vec::new();
    for subject in &cohort.subjects {
        for event in predictive_pit(&fit, &cohort, subject).expect("pit") {
            assert_eq!(event.mark, 0);
            assert!((event.mark_probabilities[0] - 1.0).abs() < 1e-12);
            pits.push(event.pit);
        }
    }
    assert!(pits.iter().all(|&u| (0.0..=1.0).contains(&u)));
    // Under the model the PITs are independent uniforms (the Rosenblatt
    // transform of the event times); the fitted parameters make this a
    // sanity band around the Kolmogorov 95% quantile, not a formal test.
    let ks = kolmogorov_smirnov_uniform(&pits).expect("events");
    let n = pits.len() as f64;
    assert!(
        ks < 1.63 / n.sqrt() + 0.05,
        "PIT KS distance {ks} over {n} events exceeds the uniform band"
    );
}

/// Write one line to stdout from test support code that is not itself a
/// `#[test]` function.
fn emit(line: &str) {
    use std::io::Write;
    let mut stdout = std::io::stdout().lock();
    if stdout.write_all(line.as_bytes()).is_ok() && stdout.write_all(b"\n").is_ok() {
        return;
    }
}

/// A stdout logger for tests that watch the outer solve, installed once.
struct StdoutLogger;

impl log::Log for StdoutLogger {
    fn enabled(&self, metadata: &log::Metadata<'_>) -> bool {
        metadata.level() <= log::max_level()
    }
    fn log(&self, record: &log::Record<'_>) {
        if self.enabled(record.metadata()) {
            emit(&record.args().to_string());
        }
    }
    fn flush(&self) {}
}

static STDOUT_LOGGER: StdoutLogger = StdoutLogger;

fn install_test_logger() {
    if log::set_logger(&STDOUT_LOGGER).is_ok() {
        log::set_max_level(log::LevelFilter::Info);
    }
}

#[test]
fn formula_right_hand_side_resolves_against_the_node_columns() {
    let mut cohort = simulate_cohort(4, 3.0, -0.5, 0.4, 0.0, 0.5, 5);
    cohort.validate().expect("valid");
    let rows = design_rows(&cohort, 3).expect("rows");
    let spec = super::formula::covariate_spec_from_formula("x + s(time)", rows.view(), &cohort)
        .expect("spec");
    assert_eq!(spec.linear_terms.len(), 1);
    assert_eq!(spec.linear_terms[0].feature_col, 0);
    assert_eq!(spec.smooth_terms.len(), 1);
    let error = super::formula::covariate_spec_from_formula("nope", rows.view(), &cohort)
        .err()
        .expect("unknown column must fail");
    assert!(error.to_string().contains("nope"), "{error}");
}

#[test]
fn a_cohort_without_shared_risk_switches_the_atom_off() {
    install_test_logger();
    let mut cohort = simulate_cohort(80, 6.0, -0.8, 0.5, 0.0, 0.4, 3);
    let mut spec = EventHistorySpec::new(1, vec![linear_spec()]);
    spec.gauss_hermite_order = 11;
    let fit = fit_event_history(&mut cohort, &spec).expect("fit");
    assert!(fit.fit.outer_gradient_norm.is_none_or(|g| g.is_finite()));
    let loading = fit.loadings[[0, 0]].abs();
    assert!(
        loading < 0.35,
        "no shared risk was simulated but the fitted loading is {loading}"
    );
    let with_atom = fit.fit.log_likelihood;
    let null_spec = EventHistorySpec::new(0, vec![linear_spec()]);
    let null = fit_event_history(&mut cohort, &null_spec).expect("null fit");
    assert!(
        with_atom >= null.fit.log_likelihood - 1e-6,
        "the atom model must not lose likelihood against its own null"
    );
}

/// Solve `A x = b` for a small dense system by Gaussian elimination.
fn solve_small(a: &Array2<f64>, b: &Array1<f64>) -> Array1<f64> {
    let n = b.len();
    let mut m = a.clone();
    let mut r = b.clone();
    for col in 0..n {
        let pivot = (col..n)
            .max_by(|&i, &j| m[[i, col]].abs().total_cmp(&m[[j, col]].abs()))
            .expect("row");
        if pivot != col {
            for k in 0..n {
                let t = m[[col, k]];
                m[[col, k]] = m[[pivot, k]];
                m[[pivot, k]] = t;
            }
            let t = r[col];
            r[col] = r[pivot];
            r[pivot] = t;
        }
        for i in (col + 1)..n {
            let f = m[[i, col]] / m[[col, col]];
            for k in col..n {
                m[[i, k]] -= f * m[[col, k]];
            }
            r[i] -= f * r[col];
        }
    }
    let mut x = Array1::<f64>::zeros(n);
    for i in (0..n).rev() {
        let mut acc = r[i];
        for k in (i + 1)..n {
            acc -= m[[i, k]] * x[k];
        }
        x[i] = acc / m[[i, i]];
    }
    x
}

#[test]
fn newton_direction_decreases_the_penalised_objective_at_the_start() {
    let mut cohort = simulate_cohort(80, 6.0, -0.8, 0.5, 0.0, 0.4, 3);
    cohort.validate().expect("valid");
    let nodes = Arc::new(expand_nodes(&cohort, 9, 0).expect("nodes"));
    let total = nodes.total_nodes;
    let mut design = Array2::<f64>::zeros((total, 2));
    for row in 0..total {
        design[[row, 0]] = 1.0;
        design[[row, 1]] = nodes.node_data[[row, 0]];
    }
    let family = EventHistoryFamily::new(
        Arc::clone(&nodes),
        vec![Arc::new(design.clone())],
        1,
        11,
        cohort.time_scale(),
    )
    .expect("family");
    let states = |beta: &Array1<f64>, latent: &Array1<f64>| {
        vec![
            ParameterBlockState {
                beta: beta.clone(),
                eta: design.dot(beta),
            },
            ParameterBlockState {
                beta: latent.clone(),
                eta: Array1::zeros(total),
            },
        ]
    };
    // Ridge on the atom (loading and log-rate) at λ = 1.
    let penalty = |latent: &Array1<f64>| 0.5 * latent.dot(latent);
    let objective = |beta: &Array1<f64>, latent: &Array1<f64>| -> f64 {
        -family.log_likelihood(&states(beta, latent)).expect("value") + penalty(latent)
    };
    let beta0 = array![0.0, 0.0];
    let latent0 = array![0.0, 0.0];
    let joint = family
        .joint_evaluation(&states(&beta0, &latent0))
        .expect("joint");
    let value_only = family
        .log_likelihood(&states(&beta0, &latent0))
        .expect("value");
    assert_eq!(
        joint.log_likelihood, value_only,
        "value-only and derivative paths must agree bitwise"
    );
    let p = 4;
    for i in 0..p {
        let h_fd = 1e-6;
        let mut plus = Array1::<f64>::zeros(p);
        plus[i] = h_fd;
        let (bp, lp) = (
            &beta0 + &plus.slice(ndarray::s![0..2]),
            &latent0 + &plus.slice(ndarray::s![2..4]),
        );
        let (bm, lm) = (
            &beta0 - &plus.slice(ndarray::s![0..2]),
            &latent0 - &plus.slice(ndarray::s![2..4]),
        );
        let fd = (objective(&bp, &lp) - objective(&bm, &lm)) / (2.0 * h_fd);
        let analytic = -joint.gradient[i] + if i >= 2 { latent0[i - 2] } else { 0.0 };
        emit(&format!("coefficient {i}: analytic {analytic} vs finite difference {fd}"));
    }
    let mut h = joint.hessian.clone();
    let mut g = -joint.gradient.clone();
    for i in 2..p {
        h[[i, i]] += 1.0;
        g[i] += latent0[i - 2];
    }
    let direction = solve_small(&h, &g.mapv(|v| -v));
    let base = objective(&beta0, &latent0);
    let mut decreased = false;
    for &t in &[1.0, 0.5, 0.25, 0.125, 0.0625] {
        let beta = &beta0 + &direction.slice(ndarray::s![0..2]).to_owned().mapv(|v| v * t);
        let latent = &latent0 + &direction.slice(ndarray::s![2..4]).to_owned().mapv(|v| v * t);
        let trial = objective(&beta, &latent);
        let predicted = t * g.dot(&direction) + 0.5 * t * t * direction.dot(&h.dot(&direction));
        println!(
            "t={t}: objective {base} -> {trial} (actual {:+e}, model {predicted:+e})",
            trial - base
        );
        if trial < base {
            decreased = true;
        }
    }
    assert!(decreased, "no step along the Newton direction decreases the objective");
    // The gradient must be the derivative of the value along the direction.
    let h_fd = 1e-5;
    let beta = &beta0 + &direction.slice(ndarray::s![0..2]).to_owned().mapv(|v| v * h_fd);
    let latent = &latent0 + &direction.slice(ndarray::s![2..4]).to_owned().mapv(|v| v * h_fd);
    let beta_m = &beta0 - &direction.slice(ndarray::s![0..2]).to_owned().mapv(|v| v * h_fd);
    let latent_m = &latent0 - &direction.slice(ndarray::s![2..4]).to_owned().mapv(|v| v * h_fd);
    let fd = (objective(&beta, &latent) - objective(&beta_m, &latent_m)) / (2.0 * h_fd);
    let analytic = g.dot(&direction);
    println!("directional derivative: analytic {analytic} vs finite difference {fd}");
    assert!(
        (fd - analytic).abs() < 1e-4 * (1.0 + analytic.abs()),
        "directional derivative {analytic} vs finite difference {fd}"
    );
}

#[test]
fn lagrange_basis_keeps_its_derivative_within_roundoff_of_a_node() {
    // A point on, or within roundoff of, a node must carry the same dL_i/dx
    // as a point a hair away, where nothing is singular.
    let gh = GaussHermite::new(9).expect("rule");
    for hit in 0..gh.order {
        let x = gh.nodes[hit];
        let step = 1e-6;
        let plus = gh.lagrange_basis(&super::family::seeded_one(x + step, 1.0));
        let minus = gh.lagrange_basis(&super::family::seeded_one(x - step, 1.0));
        for offset in [0.0, 1e-17, -1e-17, 1e-13] {
            let on = gh.lagrange_basis(&super::family::seeded_one(x + offset, 1.0));
            for i in 0..gh.order {
                let fd_first = (plus[i].value() - minus[i].value()) / (2.0 * step);
                let expected_value = if i == hit { 1.0 } else { 0.0 } + fd_first * offset;
                assert!(
                    (on[i].value() - expected_value).abs() < 1e-12 + 1e-6 * offset.abs(),
                    "node {hit} basis {i} at offset {offset}: value {} vs {expected_value}",
                    on[i].value()
                );
                assert!(
                    (on[i].eps() - fd_first).abs() < 1e-6 * (1.0 + fd_first.abs()),
                    "node {hit} basis {i} at offset {offset}: derivative {} vs finite difference {fd_first}",
                    on[i].eps()
                );
            }
        }
    }
    // Partition of unity and exact reproduction of x^m for m below the order.
    let x = 0.371;
    let basis = gh.lagrange_basis(&x);
    for m in 0..gh.order {
        let reproduced: f64 = basis
            .iter()
            .zip(gh.nodes.iter())
            .map(|(l, node)| l * node.powi(m as i32))
            .sum();
        let exact = x.powi(m as i32);
        assert!(
            (reproduced - exact).abs() < 1e-11 * (1.0 + exact.abs()),
            "x^{m}: {reproduced} vs {exact}"
        );
    }
}

#[test]
fn transition_at_an_overflowed_rate_is_finite_with_zero_sensitivity() {
    // log-rate 800: exp overflows to infinity, φ is exactly zero, and every
    // derivative channel must be finite (zero), not ∞ · 0.
    let kappa = super::scalar::exp(&super::family::seeded_one(800.0, 1.0)).scale(0.7);
    let transition = AtomTransition::new(&kappa);
    assert_eq!(transition.phi.value(), 0.0);
    assert_eq!(transition.innovation.value(), 1.0);
    for value in [
        &transition.phi,
        &transition.innovation,
        &transition.dphi,
        &transition.d2phi,
    ] {
        assert!(value.value().is_finite() && value.eps().is_finite());
        assert_eq!(value.eps(), 0.0);
    }
    let plain = AtomTransition::new(&f64::INFINITY);
    assert_eq!(plain.dphi, 0.0);
    assert_eq!(plain.d2phi, 0.0);
}

/// The 80-subject loaded cohort the cost and diagnostic tests share.
fn loaded_cohort() -> EventHistoryCohort {
    let mut cohort = simulate_cohort(80, 6.0, -0.8, 0.5, 1.0, 0.4, 7);
    cohort.validate().expect("valid");
    cohort
}

/// Family on the loaded cohort at a Gauss-Hermite order, with the dense
/// linear design and zeroed block states.
fn loaded_family(
    cohort: &EventHistoryCohort,
    order: usize,
) -> (EventHistoryFamily, Arc<Array2<f64>>, Vec<ParameterBlockState>) {
    use gam_terms::smooth::build_term_collection_design;
    let nodes = Arc::new(expand_nodes(cohort, 9, 0).expect("nodes"));
    let design = build_term_collection_design(nodes.node_data.view(), &linear_spec()).expect("design");
    let dense = design.design.try_to_dense_arc("test design").expect("dense");
    let family = EventHistoryFamily::new(
        Arc::clone(&nodes),
        vec![Arc::clone(&dense)],
        1,
        order,
        cohort.time_scale(),
    )
    .expect("family");
    let states = vec![
        ParameterBlockState {
            beta: Array1::zeros(2),
            eta: Array1::zeros(nodes.total_nodes),
        },
        ParameterBlockState {
            beta: Array1::zeros(2),
            eta: Array1::zeros(nodes.total_nodes),
        },
    ];
    (family, dense, states)
}

#[test]
fn smallest_prefix_with_non_finite_louis_output_at_order_21() {
    let cohort = loaded_cohort();
    let (_, design_dense, _) = loaded_family(&cohort, 21);
    let beta = array![-0.9485, 0.5452];
    let eta = design_dense.dot(&beta);
    let gh = GaussHermite::new(21).expect("rule");
    let nodes = expand_nodes(&cohort, 9, 0).expect("nodes");
    let mut report = Vec::new();
    for (s, subj) in nodes.subjects.iter().enumerate() {
        let n = subj.len();
        let run = |len: usize| -> bool {
            let prefix = SubjectNodes {
                first_row: 0,
                times: subj.times[..len].to_vec(),
                gaps: subj.gaps[..len - 1].to_vec(),
                weights: subj.weights[..len].to_vec(),
                exposures: subj.exposures.slice(ndarray::s![..len, ..]).to_owned(),
                counts: subj.counts.slice(ndarray::s![..len, ..]).to_owned(),
                covariate_rows: subj.covariate_rows[..len].to_vec(),
            };
            let eta0: Vec<f64> = (0..len).map(|i| eta[subj.first_row + i]).collect();
            let inputs = SubjectInputs {
                nodes: &prefix,
                eta0: &eta0,
                loadings: &[1.2054],
                log_rates: &[1.0587],
                time_scale: cohort.time_scale(),
                gh: &gh,
                continuation_gap: 0.0,
            designs: None,
            };
            match subject_marginal(&inputs, true) {
                Ok(out) => {
                    out.loglik.is_finite()
                        && out.gradient.iter().all(|v| v.is_finite())
                        && out.hessian.iter().all(|v| v.is_finite())
                }
                Err(_) => false,
            }
        };
        if run(n) {
            continue;
        }
        let mut lo = 1;
        let mut hi = n;
        while lo < hi {
            let mid = (lo + hi) / 2;
            if run(mid) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        let bad = lo;
        report.push(format!(
            "subject {s}: {n} nodes, first non-finite prefix {bad}; node {}: time {} gap {} exposures {:?} counts {:?}; prefix log-lik at {bad}: {:?}",
            bad - 1,
            subj.times[bad - 1],
            if bad >= 2 { subj.gaps[bad - 2] } else { 0.0 },
            subj.exposures.row(bad - 1).to_vec(),
            subj.counts.row(bad - 1).to_vec(),
            {
                let prefix = SubjectNodes {
                    first_row: 0,
                    times: subj.times[..bad].to_vec(),
                    gaps: subj.gaps[..bad - 1].to_vec(),
                    weights: subj.weights[..bad].to_vec(),
                    exposures: subj.exposures.slice(ndarray::s![..bad, ..]).to_owned(),
                    counts: subj.counts.slice(ndarray::s![..bad, ..]).to_owned(),
                    covariate_rows: subj.covariate_rows[..bad].to_vec(),
                };
                let eta0: Vec<f64> = (0..bad).map(|i| eta[subj.first_row + i]).collect();
                let inputs = SubjectInputs {
                    nodes: &prefix,
                    eta0: &eta0,
                    loadings: &[1.2054],
                    log_rates: &[1.0587],
                    time_scale: cohort.time_scale(),
                    gh: &gh,
                    continuation_gap: 0.0,
                designs: None,
                };
                subject_marginal(&inputs, true).map(|o| {
                    (
                        o.loglik,
                        o.gradient.iter().filter(|v| !v.is_finite()).count(),
                        o.hessian.iter().filter(|v| !v.is_finite()).count(),
                        o.gradient.iter().take(4).copied().collect::<Vec<_>>(),
                    )
                })
            }
        ));
        if report.len() >= 3 {
            break;
        }
    }
    for line in &report {
        emit(line);
    }
    assert!(report.is_empty(), "{} subjects with non-finite Louis output", report.len());
}

#[test]
fn louis_hessian_converges_to_the_computed_curvature_as_the_quadrature_resolves() {
    // Louis' identity is the Hessian of the EXACT marginal; a finite
    // difference of the exact gradient is the Hessian of the COMPUTED one.
    // They are two approximations of the same curvature and differ by the
    // quadrature error, so the property that holds is convergence: raising
    // the Gauss-Hermite order must shrink the gap. Asserting a fixed
    // tolerance at one order instead would be asserting a number nobody
    // derived. The gradient itself is the exact derivative of the computed
    // value, so it agrees with its own finite difference to roundoff, and
    // step-size independence there is what rules out a ripple in the
    // objective.
    //
    // The state is where the fixed-λ inner solve lands on the loaded cohort:
    // the conditional intercept −0.9485 with loading 1.2054, which under the
    // population parameterisation is `−0.9485 + ½·1.2054²`.
    let cohort = loaded_cohort();
    let loading = 1.2054_f64;
    let beta = array![-0.9485 + 0.5 * loading * loading, 0.5452];
    let latent = array![loading, 1.0587];
    let p = 4;
    let mut discrepancy_by_order = Vec::new();
    for order in [11usize, 21] {
        let (family, design_dense, base_states) = loaded_family(&cohort, order);
        let at = |beta: &Array1<f64>, latent: &Array1<f64>| -> Vec<ParameterBlockState> {
            let mut states = base_states.clone();
            states[0].beta = beta.clone();
            states[0].eta = design_dense.dot(beta);
            states[1].beta = latent.clone();
            states
        };
        let base = family.joint_evaluation(&at(&beta, &latent)).expect("joint");
        let mut full = Array1::<f64>::zeros(p);
        full.slice_mut(ndarray::s![0..2]).assign(&beta);
        full.slice_mut(ndarray::s![2..4]).assign(&latent);
        let split = |v: &Array1<f64>| -> (Array1<f64>, Array1<f64>) {
            (v.slice(ndarray::s![0..2]).to_owned(), v.slice(ndarray::s![2..4]).to_owned())
        };
        let mut worst = 0.0_f64;
        let mut worst_at = (0usize, 0usize);
        for h in [1e-3, 1e-5] {
            for i in 0..p {
                let mut plus = full.clone();
                plus[i] += h;
                let mut minus = full.clone();
                minus[i] -= h;
                let (bp, lp) = split(&plus);
                let (bm, lm) = split(&minus);
                let gp = family.joint_evaluation(&at(&bp, &lp)).expect("joint").gradient.clone();
                let gm = family.joint_evaluation(&at(&bm, &lm)).expect("joint").gradient.clone();
                let vp = family.log_likelihood(&at(&bp, &lp)).expect("value");
                let vm = family.log_likelihood(&at(&bm, &lm)).expect("value");
                let fd_gradient = (vp - vm) / (2.0 * h);
                let fd_row: Vec<String> = (0..p).map(|j| format!("{:.5e}", -(gp[j] - gm[j]) / (2.0 * h))).collect();
                let louis_row: Vec<String> = (0..p).map(|j| format!("{:.5e}", base.hessian[[i, j]])).collect();
                emit(&format!(
                    "[final G={order} h={h:.0e}] coefficient {i}: gradient exact {:.6e} fd {:.6e}; -hessian louis [{}] fd [{}]",
                    base.gradient[i], fd_gradient, louis_row.join(", "), fd_row.join(", ")
                ));
                if h == 1e-5 {
                    // The exact gradient IS the derivative of the computed
                    // value, so it meets its own central difference at the
                    // step where truncation is negligible.
                    assert!(
                        (base.gradient[i] - fd_gradient).abs() < 1e-6 * (1.0 + fd_gradient.abs()),
                        "G={order}: gradient {i} exact {} vs finite difference {fd_gradient}",
                        base.gradient[i]
                    );
                    for j in 0..p {
                        let fd_h = -(gp[j] - gm[j]) / (2.0 * h);
                        let relative = (base.hessian[[i, j]] - fd_h).abs() / (1.0 + fd_h.abs());
                        if relative > worst {
                            worst = relative;
                            worst_at = (i, j);
                        }
                    }
                }
            }
        }
        emit(&format!(
            "[louis] G={order}: worst relative gap {worst:.4e} at entry {worst_at:?}"
        ));
        discrepancy_by_order.push(worst);
    }
    let (coarse, fine) = (discrepancy_by_order[0], discrepancy_by_order[1]);
    assert!(
        fine < 0.5 * coarse,
        "the gap between Louis' Hessian and the computed curvature must shrink with the quadrature: {coarse:.4e} at order 11, {fine:.4e} at order 21"
    );
    assert!(
        fine < 5e-2,
        "at order 21 the two curvatures should agree to a few percent; the worst entry differs by {fine:.4e}"
    );
}

#[test]
fn spline_basis_reproduces_cubics_with_a_bounded_operator_norm() {
    let gh = GaussHermite::new(21).expect("rule");
    let g = gh.order;
    // Exact on linear data, inside and beyond the hull.
    let linear: Vec<f64> = gh.nodes.iter().map(|x| 0.4 - 0.7 * x).collect();
    for &x in &[-9.0, -5.0, -1.3, 0.0, 0.27, 2.9, 5.6, 8.0] {
        let basis = gh.spline_basis(&x);
        let value: f64 = basis.iter().zip(linear.iter()).map(|(b, f)| b * f).sum();
        assert!((value - (0.4 - 0.7 * x)).abs() < 1e-12, "linear at {x}: {value}");
        let unity: f64 = basis.iter().sum();
        assert!((unity - 1.0).abs() < 1e-12, "partition of unity at {x}: {unity}");
    }
    // Exact on cubic data everywhere on the hull (not-a-knot).
    let cubic: Vec<f64> = gh.nodes.iter().map(|x| 0.2 * x * x * x - x * x + 0.5 * x - 1.0).collect();
    for step in 0..200 {
        let x = gh.nodes[0] + (gh.nodes[g - 1] - gh.nodes[0]) * step as f64 / 199.0;
        let value: f64 = gh.spline_basis(&x).iter().zip(cubic.iter()).map(|(b, f)| b * f).sum();
        let exact = 0.2 * x * x * x - x * x + 0.5 * x - 1.0;
        assert!((value - exact).abs() < 1e-9 * (1.0 + exact.abs()), "cubic at {x}: {value} vs {exact}");
    }
    // Interpolates the nodal values exactly.
    let data: Vec<f64> = gh.nodes.iter().map(|x| (-(x * x) / 3.0).exp() * (1.0 + x)).collect();
    for j in 0..g {
        let basis = gh.spline_basis(&gh.nodes[j]);
        let value: f64 = basis.iter().zip(data.iter()).map(|(b, f)| b * f).sum();
        assert!((value - data[j]).abs() < 1e-12, "node {j}: {value} vs {}", data[j]);
    }
    // Neither interpolant preserves the nodal range, but their operator
    // norms differ in kind: the cubic spline's `max_x Σ_j |S_j(x)|` is a
    // small constant on these nodes, the Lagrange interpolant's is the
    // Lebesgue constant, exponential in the order. The norm is a theorem
    // about the overshoot: `|S f(x) − c| ≤ ‖S‖ max_j |f_j − c|` for any
    // centre `c`, so centred data cannot overshoot by more than
    // `(‖S‖ − 1)` times its half-range.
    let mut spline_norm = 1.0_f64;
    let mut lagrange_norm = 1.0_f64;
    for step in 0..2000 {
        let x = gh.nodes[0] + (gh.nodes[g - 1] - gh.nodes[0]) * step as f64 / 1999.0;
        spline_norm = spline_norm.max(gh.spline_basis(&x).iter().map(|b| b.abs()).sum());
        lagrange_norm = lagrange_norm.max(gh.lagrange_basis(&x).iter().map(|b| b.abs()).sum());
    }
    assert!(spline_norm < 3.0, "cubic spline operator norm {spline_norm}");
    assert!(lagrange_norm > 100.0, "Lagrange operator norm {lagrange_norm}");
    assert!((gh.lebesgue_constant - lagrange_norm).abs() < 0.05 * lagrange_norm);
    let steep: Vec<f64> = gh.nodes.iter().map(|x| -12.0 * (x + 1.0).abs().powf(1.5) + 3.0).collect();
    let (lo, hi) = steep.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(l, h), v| (l.min(*v), h.max(*v)));
    let centre = 0.5 * (lo + hi);
    let half_range = 0.5 * (hi - lo);
    let mut spline_overshoot = 0.0_f64;
    let mut lagrange_overshoot = 0.0_f64;
    for step in 0..400 {
        let x = gh.nodes[0] + (gh.nodes[g - 1] - gh.nodes[0]) * step as f64 / 399.0;
        let spline: f64 = gh.spline_basis(&x).iter().zip(steep.iter()).map(|(b, f)| b * f).sum();
        spline_overshoot = spline_overshoot.max((spline - hi).max(lo - spline));
        let lagrange: f64 = gh.lagrange_basis(&x).iter().zip(steep.iter()).map(|(b, f)| b * f).sum();
        lagrange_overshoot = lagrange_overshoot.max((lagrange - hi).max(lo - lagrange));
    }
    assert!(
        spline_overshoot <= (spline_norm - 1.0) * half_range + 1e-9,
        "spline overshoot {spline_overshoot} exceeds its operator bound {} (centre {centre})",
        (spline_norm - 1.0) * half_range
    );
    assert!(spline_overshoot < 0.1 * (hi - lo), "spline overshoot {spline_overshoot} on a range of {}", hi - lo);
    assert!(lagrange_overshoot > hi - lo, "the control did not overshoot: {lagrange_overshoot}");
}

#[test]
fn lebesgue_constant_grows_with_the_order_and_is_recorded() {
    let mut previous = 0.0;
    for order in [5usize, 9, 17, 33] {
        let gh = GaussHermite::new(order).expect("rule");
        assert!(gh.lebesgue_constant > previous, "order {order}: {}", gh.lebesgue_constant);
        previous = gh.lebesgue_constant;
    }
    assert!(previous > 1e3, "the Lebesgue constant at order 33 is {previous}");
    let g9 = GaussHermite::new(9).expect("rule");
    assert!(g9.lebesgue_constant < 50.0, "order 9: {}", g9.lebesgue_constant);
}

#[test]
fn dual_loading_derivative_matches_finite_difference_at_zero_loading() {
    // At loading zero the computed marginal is exactly symmetric in the
    // loading, so its derivative is zero; the dual must reproduce that as
    // the node count grows and the grid starts moving with the loading.
    use super::scalar::Tangent;
    let gh = GaussHermite::new(9).expect("rule");
    for nodes in 1..=4 {
        let times: Vec<f64> = (0..nodes).map(|n| n as f64 * 0.7).collect();
        let exposures: Vec<f64> = (0..nodes).map(|n| if n == 0 { 0.0 } else { 0.7 }).collect();
        let counts: Vec<Vec<f64>> = (0..nodes).map(|n| vec![if n % 2 == 1 { 1.0 } else { 0.0 }]).collect();
        let subj = subject(&times, &exposures, &counts);
        let value = |a: f64| -> f64 {
            let eta0 = vec![0.3; nodes];
            let inputs = SubjectInputs {
                nodes: &subj,
                eta0: &eta0,
                loadings: &[a],
                log_rates: &[0.2],
                time_scale: 1.0,
                gh: &gh,
                continuation_gap: 0.0,
            designs: None,
            };
            subject_marginal(&inputs, false).expect("value").loglik
        };
        let eta0: Vec<Tangent<1>> = (0..nodes).map(|_| Tangent::seeded(0.3, [0.0])).collect();
        let inputs = SubjectInputs {
            nodes: &subj,
            eta0: &eta0,
            loadings: &[Tangent::seeded(0.0, [1.0])],
            log_rates: &[Tangent::seeded(0.2, [0.0])],
            time_scale: 1.0,
            gh: &gh,
            continuation_gap: 0.0,
        designs: None,
        };
        let dual = subject_marginal(&inputs, false).expect("dual").loglik;
        let h = 1e-5;
        let fd = (value(h) - value(-h)) / (2.0 * h);
        emit(&format!(
            "nodes={nodes}: value {} dual {} d/da {} vs finite difference {fd}",
            value(0.0),
            dual.value,
            dual.grad[0]
        ));
        assert_eq!(dual.value, value(0.0), "dual value channel must match the plain value");
        assert!(
            (dual.grad[0] - fd).abs() < 1e-6 * (1.0 + fd.abs()),
            "nodes={nodes}: d/da {} vs finite difference {fd}",
            dual.grad[0]
        );
    }
}

/// A tracing shell around the family that prints every engine call, so a
/// stalled inner solve can be read as the sequence of points it evaluated.
#[derive(Clone)]
struct Traced(EventHistoryFamily);

impl crate::custom_family::CustomFamily for Traced {
    fn evaluate(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<crate::custom_family::FamilyEvaluation, String> {
        let out = self.0.evaluate(block_states)?;
        emit(&format!(
            "[trace] evaluate latent={:?} beta={:?} ll={}",
            block_states[1].beta.as_slice().expect("slice"),
            block_states[0].beta.as_slice().expect("slice"),
            out.log_likelihood
        ));
        Ok(out)
    }
    fn log_likelihood_only(&self, block_states: &[ParameterBlockState]) -> Result<f64, String> {
        let out = self.0.log_likelihood_only(block_states);
        emit(&format!(
            "[trace] value latent={:?} beta={:?} -> {:?}",
            block_states[1].beta.as_slice().expect("slice"),
            block_states[0].beta.as_slice().expect("slice"),
            out
        ));
        out
    }
    fn classical_deviance(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<Option<f64>, String> {
        self.0.classical_deviance(block_states)
    }
    fn exact_newton_joint_hessian_beta_dependent(&self) -> bool {
        true
    }
    fn has_explicit_joint_hessian(&self) -> bool {
        true
    }
    fn requires_joint_outer_hyper_path(&self) -> bool {
        true
    }
    fn levenberg_on_ill_conditioning(&self) -> bool {
        true
    }
    fn coefficient_hessian_cost(&self, specs: &[crate::custom_family::ParameterBlockSpec]) -> u64 {
        self.0.coefficient_hessian_cost(specs)
    }
    fn output_channel_assignment(
        &self,
        specs: &[crate::custom_family::ParameterBlockSpec],
    ) -> Option<Vec<usize>> {
        self.0.output_channel_assignment(specs)
    }
    fn block_coefficient_coordinate(
        &self,
        block_states: &[ParameterBlockState],
        block_index: usize,
        block_spec: &crate::custom_family::ParameterBlockSpec,
    ) -> gam_problem::CoefficientCoordinate {
        self.0
            .block_coefficient_coordinate(block_states, block_index, block_spec)
    }
    fn exact_newton_joint_hessian(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<Option<Array2<f64>>, String> {
        let out = self.0.exact_newton_joint_hessian(block_states)?;
        emit(&format!(
            "[trace] joint hessian at latent={:?}: {:?}",
            block_states[1].beta.as_slice().expect("slice"),
            out.as_ref().map(|h| h.diag().to_vec())
        ));
        Ok(out)
    }
    fn exact_newton_joint_loglik_gradient(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<Option<Array1<f64>>, String> {
        let out = self.0.exact_newton_joint_loglik_gradient(block_states)?;
        emit(&format!(
            "[trace] joint gradient {:?}",
            out.as_ref().map(|g| g.to_vec())
        ));
        Ok(out)
    }
    fn exact_newton_joint_gradient_evaluation(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[crate::custom_family::ParameterBlockSpec],
    ) -> Result<Option<crate::custom_family::ExactNewtonJointGradientEvaluation>, String> {
        self.0.exact_newton_joint_gradient_evaluation(block_states, specs)
    }
    fn exact_newton_joint_hessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.0
            .exact_newton_joint_hessian_directional_derivative(block_states, d_beta_flat)
    }
    fn exact_newton_joint_hessiansecond_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        u: &Array1<f64>,
        v: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.0
            .exact_newton_joint_hessiansecond_directional_derivative(block_states, u, v)
    }
}

#[test]
fn traced_fixed_lambda_inner_solve_on_the_null_cohort() {
    use super::family::{latent_block_spec, mark_block_spec};
    use gam_terms::smooth::build_term_collection_design;
    let mut cohort = simulate_cohort(80, 6.0, -0.8, 0.5, 0.0, 0.4, 3);
    cohort.validate().expect("valid");
    let nodes = Arc::new(expand_nodes(&cohort, 9, 0).expect("nodes"));
    let design = build_term_collection_design(nodes.node_data.view(), &linear_spec()).expect("design");
    let dense = design
        .design
        .try_to_dense_arc("test design")
        .expect("dense");
    let family = EventHistoryFamily::new(
        Arc::clone(&nodes),
        vec![dense],
        1,
        11,
        cohort.time_scale(),
    )
    .expect("family");
    let specs = vec![
        mark_block_spec("event", &design),
        latent_block_spec(nodes.total_nodes, 1, 1).expect("latent spec"),
    ];
    let options = crate::custom_family::BlockwiseFitOptions::default();
    let result = crate::custom_family::fit_custom_family_fixed_log_lambdas(
        &Traced(family),
        &specs,
        &options,
        None,
    );
    match result {
        Ok(fit) => println!("[trace] converged: latent={:?}", fit.block_states[1].beta),
        Err(error) => println!("[trace] error: {error}"),
    }
}

#[test]
fn gradient_and_hessian_match_central_differences_at_tiny_gaps() {
    // Gaps of a few thousandths of the time scale drive `1 − φ²` to 1e-3;
    // the innovation-coordinate gap algebra must stay bounded there. At such
    // gaps the backward step is pure interpolation with no kernel smoothing,
    // so the agreement is limited by the Lagrange interpolation error of the
    // exponential likelihood factors on a 31-node grid, about 1e-4 relative.
    let gh = GaussHermite::new(31).expect("rule");
    let nodes = subject(
        &[0.0, 0.004, 0.011, 0.016],
        &[0.3, 0.0, 0.5, 0.2],
        &[
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
        ],
    );
    let mut theta = vec![0.2, -0.1, 0.4, 0.0, -0.3, 0.5, 0.1, 0.2];
    theta.extend([0.8, -0.4, 0.3, 0.6]);
    theta.extend([-0.2, 0.5]);
    let p = theta.len();
    let base = evaluate_at(&nodes, &gh, &theta, true);
    let h = 1e-4;
    let mut mismatches = Vec::new();
    for i in 0..p {
        let mut plus = theta.clone();
        plus[i] += h;
        let mut minus = theta.clone();
        minus[i] -= h;
        let fp = evaluate_at(&nodes, &gh, &plus, true);
        let fm = evaluate_at(&nodes, &gh, &minus, true);
        let fd = (fp.loglik - fm.loglik) / (2.0 * h);
        if (base.gradient[i] - fd).abs() >= 1e-4 * (1.0 + fd.abs()) {
            mismatches.push(format!("gradient[{i}] = {} vs {fd}", base.gradient[i]));
        }
        for j in 0..p {
            let fd_h = (fp.gradient[j] - fm.gradient[j]) / (2.0 * h);
            let value = base.hessian[i * p + j];
            if (value - fd_h).abs() >= 1e-3 * (1.0 + fd_h.abs()) {
                mismatches.push(format!(
                    "hessian[{i},{j}] = {value} vs {fd_h} (rel {:e})",
                    (value - fd_h).abs() / (1.0 + fd_h.abs())
                ));
            }
        }
    }
    for line in &mismatches {
        println!("{line}");
    }
    assert!(mismatches.is_empty(), "{} derivative entries disagree", mismatches.len());
}

#[test]
fn traced_fixed_lambda_inner_solve_on_the_loaded_cohort_reports_its_cost() {
    install_test_logger();
    use super::family::{latent_block_spec, mark_block_spec};
    use gam_terms::smooth::build_term_collection_design;
    let mut cohort = simulate_cohort(80, 6.0, -0.8, 0.5, 1.0, 0.4, 7);
    cohort.validate().expect("valid");
    let nodes = Arc::new(expand_nodes(&cohort, 9, 0).expect("nodes"));
    let design = build_term_collection_design(nodes.node_data.view(), &linear_spec()).expect("design");
    let dense = design
        .design
        .try_to_dense_arc("test design")
        .expect("dense");
    let family = EventHistoryFamily::new(
        Arc::clone(&nodes),
        vec![dense],
        1,
        11,
        cohort.time_scale(),
    )
    .expect("family");
    let total_nodes: usize = nodes.subjects.iter().map(|s| s.len()).sum();
    emit(&format!(
        "[cost] subjects={} nodes={} mean_nodes={:.1}",
        nodes.subjects.len(),
        total_nodes,
        total_nodes as f64 / nodes.subjects.len() as f64
    ));
    let states = vec![
        ParameterBlockState {
            beta: array![-0.8, 0.5],
            eta: Array1::zeros(nodes.total_nodes),
        },
        ParameterBlockState {
            beta: array![0.7, 0.0],
            eta: Array1::zeros(nodes.total_nodes),
        },
    ];
    let mut states = states;
    let design_dense = design.design.try_to_dense_arc("d").expect("dense");
    states[0].eta = design_dense.dot(&states[0].beta);
    let clock = std::time::Instant::now();
    family.log_likelihood(&states).expect("value");
    emit(&format!("[cost] value-only {:.3}s", clock.elapsed().as_secs_f64()));
    let clock = std::time::Instant::now();
    family.joint_evaluation(&states).expect("joint");
    emit(&format!("[cost] joint (value+gradient+hessian) {:.3}s", clock.elapsed().as_secs_f64()));
    let u = array![0.1, 0.2, 0.3, 0.4];
    let clock = std::time::Instant::now();
    family.directional_hessian(&states, &u).expect("directional");
    emit(&format!("[cost] directional hessian {:.3}s", clock.elapsed().as_secs_f64()));
    let clock = std::time::Instant::now();
    family
        .second_directional_hessian(&states, &u, &u)
        .expect("second directional");
    emit(&format!("[cost] second directional hessian {:.3}s", clock.elapsed().as_secs_f64()));
    let specs = vec![
        mark_block_spec("event", &design),
        latent_block_spec(nodes.total_nodes, 1, 1).expect("latent spec"),
    ];
    let options = crate::custom_family::BlockwiseFitOptions::default();
    let clock = std::time::Instant::now();
    let traced = Traced(family);
    let result = crate::custom_family::fit_custom_family_fixed_log_lambdas(
        &traced,
        &specs,
        &options,
        None,
    );
    match result {
        Ok(fit) => {
            emit(&format!(
                "[cost] fixed-lambda inner solve {:.1}s cycles={} latent={:?}",
                clock.elapsed().as_secs_f64(),
                fit.inner_cycles,
                fit.block_states[1].beta
            ));
            // Louis Hessian against a finite difference of the exact gradient
            // at the final state, and the exact gradient against a finite
            // difference of the value.
            let family = &traced.0;
            let at = |beta: &Array1<f64>, latent: &Array1<f64>| -> Vec<ParameterBlockState> {
                let mut states = fit.block_states.clone();
                states[0].beta = beta.clone();
                states[0].eta = design_dense.dot(beta);
                states[1].beta = latent.clone();
                states
            };
            let beta = fit.block_states[0].beta.clone();
            let latent = fit.block_states[1].beta.clone();
            let base = family.joint_evaluation(&at(&beta, &latent)).expect("joint");
            let p = beta.len() + latent.len();
            let h = 1e-5;
            let split = |v: &Array1<f64>| -> (Array1<f64>, Array1<f64>) {
                (
                    v.slice(ndarray::s![0..beta.len()]).to_owned(),
                    v.slice(ndarray::s![beta.len()..]).to_owned(),
                )
            };
            let mut full = Array1::<f64>::zeros(p);
            full.slice_mut(ndarray::s![0..beta.len()]).assign(&beta);
            full.slice_mut(ndarray::s![beta.len()..]).assign(&latent);
            for i in 0..p {
                let mut plus = full.clone();
                plus[i] += h;
                let mut minus = full.clone();
                minus[i] -= h;
                let (bp, lp) = split(&plus);
                let (bm, lm) = split(&minus);
                let gp = family.joint_evaluation(&at(&bp, &lp)).expect("joint").gradient.clone();
                let gm = family.joint_evaluation(&at(&bm, &lm)).expect("joint").gradient.clone();
                let vp = family.log_likelihood(&at(&bp, &lp)).expect("value");
                let vm = family.log_likelihood(&at(&bm, &lm)).expect("value");
                let fd_gradient = (vp - vm) / (2.0 * h);
                let fd_row: Vec<f64> = (0..p).map(|j| -(gp[j] - gm[j]) / (2.0 * h)).collect();
                let louis_row: Vec<f64> = (0..p).map(|j| base.hessian[[i, j]]).collect();
                emit(&format!(
                    "[final] coefficient {i}: gradient exact {:.9e} fd {:.9e}; -hessian row louis {:?} fd {:?}",
                    base.gradient[i], fd_gradient, louis_row, fd_row
                ));
            }
        }
        Err(error) => emit(&format!("[cost] error after {:.1}s: {error}", clock.elapsed().as_secs_f64())),
    }
}

/// Simulate a single-mark cohort whose log-intensity is `β₀ + b(t) g` for a
/// subject-level standard-normal score `g` and no latent state, by thinning
/// under the bound `exp(β₀ + slope_bound · |g|)` with `slope_bound ≥ max |b|`.
fn simulate_score_cohort(
    subjects: usize,
    follow_up: f64,
    intercept: f64,
    slope: &dyn Fn(f64) -> f64,
    slope_bound: f64,
    seed: u64,
) -> EventHistoryCohort {
    let mut rng = Rng(seed);
    let mut covariates = Array2::<f64>::zeros((subjects, 1));
    let mut histories = Vec::with_capacity(subjects);
    for s in 0..subjects {
        let g = rng.normal();
        covariates[[s, 0]] = g;
        let bound = (intercept + slope_bound * g.abs()).exp();
        let mut events = Vec::new();
        let mut t = 0.0;
        loop {
            t -= bound.recip() * rng.uniform().max(1e-300).ln();
            if t >= follow_up {
                break;
            }
            let intensity = (intercept + slope(t) * g).exp();
            assert!(intensity <= bound, "thinning bound violated");
            if rng.uniform() * bound < intensity {
                events.push(Event { time: t, mark: 0 });
            }
        }
        histories.push(SubjectHistory {
            id: format!("s{s}"),
            entry: 0.0,
            exit: follow_up,
            events,
            segments: vec![CovariateSegment {
                start: 0.0,
                row: s,
            }],
        });
    }
    EventHistoryCohort {
        mark_names: vec!["event".to_string()],
        mark_kinds: vec![MarkKind::Recurrent],
        covariate_names: vec!["g".to_string()],
        covariate_levels: vec![Vec::new()],
        covariates,
        subjects: histories,
    }
}

/// The fitted score slope `b(t) = η(g = 1, t) − η(g = 0, t)` of mark 0, for
/// a fit whose node columns are `[g, time]`.
fn fitted_score_slope(fit: &EventHistoryFit, times: &[f64]) -> Vec<f64> {
    let mut rows = Array2::<f64>::zeros((2 * times.len(), 2));
    for (i, &t) in times.iter().enumerate() {
        rows[[2 * i, 0]] = 1.0;
        rows[[2 * i, 1]] = t;
        rows[[2 * i + 1, 1]] = t;
    }
    let design =
        build_term_collection_design(rows.view(), &fit.frozen_specs[0]).expect("prediction design");
    let dense = design
        .design
        .try_to_dense_arc("score slope design")
        .expect("dense design");
    let beta = fit.mark_coefficients(0);
    let eta = |r: usize| -> f64 {
        design.affine_offset[r]
            + dense
                .row(r)
                .iter()
                .zip(beta.iter())
                .map(|(x, b)| x * b)
                .sum::<f64>()
    };
    (0..times.len()).map(|i| eta(2 * i) - eta(2 * i + 1)).collect()
}

/// An observed subject-level score enters the intensity as one penalised
/// slope surface `b(t) · g`: a continuous by-smooth keeps its constant, so
/// its wiggliness ridge decides how much the score's effect bends with time
/// and its null-space ridge decides whether the effect exists at all, both
/// selected by REML. A declining effect is
/// recovered as a decline; a score carrying nothing collapses to zero.
#[test]
fn an_observed_score_enters_as_a_penalised_slope_surface() {
    install_test_logger();
    let times = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5];
    let formula = "s(time, by=g)";
    let truth = |t: f64| 1.0 - 0.15 * t;
    let mut cohort = simulate_score_cohort(300, 6.0, -0.5, &truth, 1.0, 19);
    let events: usize = cohort.subjects.iter().map(|s| s.events.len()).sum();
    let started = std::time::Instant::now();
    let fit = fit_event_history_formula(&mut cohort, formula, 0, BlockwiseFitOptions::default())
        .expect("fit with a declining score effect");
    let slope = fitted_score_slope(&fit, &times);
    emit(&format!(
        "[score-slope] declining arm: {events} events, {:.1}s, outer_iterations={} log_lambdas={:?}",
        started.elapsed().as_secs_f64(),
        fit.fit.outer_iterations,
        fit.fit.log_lambdas
    ));
    for (t, b) in times.iter().zip(slope.iter()) {
        emit(&format!("[score-slope]   t={t} fitted={b:.3} truth={:.3}", truth(*t)));
    }
    // The whole fitted surface, densely, for plotting.
    let dense: Vec<f64> = (0..=60).map(|i| 0.1 * i as f64).collect();
    for (t, b) in dense.iter().zip(fitted_score_slope(&fit, &dense).iter()) {
        emit(&format!("[score-slope-curve] arm=declining t={t:.2} fitted={b:.5} truth={:.5}", truth(*t)));
    }
    assert!(
        slope[0] - slope[5] > 0.35,
        "the score's effect declines by 0.75 over the follow-up; the fit shows {} → {}",
        slope[0],
        slope[5]
    );
    for (t, b) in times.iter().zip(slope.iter()) {
        assert!(
            (b - truth(*t)).abs() < 0.3,
            "slope at t={t}: fitted {b}, truth {}",
            truth(*t)
        );
    }

    // Control: the score carries nothing. The same surface must collapse.
    let mut null = simulate_score_cohort(300, 6.0, -0.5, &|_| 0.0, 0.0, 23);
    let null_events: usize = null.subjects.iter().map(|s| s.events.len()).sum();
    let started = std::time::Instant::now();
    let null_fit = fit_event_history_formula(&mut null, formula, 0, BlockwiseFitOptions::default())
        .expect("fit with an uninformative score");
    let null_slope = fitted_score_slope(&null_fit, &times);
    emit(&format!(
        "[score-slope] null arm: {null_events} events, {:.1}s, outer_iterations={} log_lambdas={:?}",
        started.elapsed().as_secs_f64(),
        null_fit.fit.outer_iterations,
        null_fit.fit.log_lambdas
    ));
    for (t, b) in times.iter().zip(null_slope.iter()) {
        emit(&format!("[score-slope]   t={t} fitted={b:.3} truth=0.000"));
    }
    for (t, b) in dense.iter().zip(fitted_score_slope(&null_fit, &dense).iter()) {
        emit(&format!("[score-slope-curve] arm=null t={t:.2} fitted={b:.5} truth=0.00000"));
    }
    let amplitude = null_slope.iter().fold(0.0f64, |m, b| m.max(b.abs()));
    assert!(
        amplitude < 0.15,
        "an uninformative score should collapse to zero; the fitted surface reaches {amplitude}"
    );
}

/// The information hierarchy is three conditionings of one model. With a
/// standard-normal subject-level score `x` and a latent atom, the population
/// tier is the zero-count filter from the stationary prior at the population
/// score; the score-only tier is the same filter at the subject's own score;
/// the history tier continues from the subject's filtered state. A positive
/// score effect orders the first two by the score's sign, and a history
/// richer (poorer) in events than its score alone predicts raises (lowers)
/// the third against the second. No weight between the tiers is chosen.
#[test]
fn forecast_tiers_population_score_and_history_are_one_model_conditioned_on_more() {
    install_test_logger();
    let mut cohort = simulate_cohort(60, 6.0, -0.8, 0.5, 1.0, 0.4, 5);
    let mut spec = EventHistorySpec::new(1, vec![linear_spec()]);
    spec.gauss_hermite_order = 11;
    let started = std::time::Instant::now();
    let fit = fit_event_history(&mut cohort, &spec).expect("fit");
    let beta = fit.mark_coefficients(0);
    emit(&format!(
        "[tiers] {:.1}s beta={:?} loading={} rate={}",
        started.elapsed().as_secs_f64(),
        beta.to_vec(),
        fit.loadings[[0, 0]],
        fit.rates[0]
    ));
    assert!(beta[1] > 0.0, "the score effect was simulated positive; fitted {}", beta[1]);
    let horizons = [7.0, 8.0];
    let at = |start: f64, score: f64| -> Vec<FutureSegment> {
        vec![FutureSegment {
            start,
            covariates: vec![score],
        }]
    };
    let population = population_forecast(
        &fit,
        &cohort,
        &PopulationForecastRequest {
            start: 6.0,
            horizons: &horizons,
            future: &at(6.0, 0.0),
        },
    )
    .expect("population forecast");
    assert!((population.survival[1] - 1.0).abs() < 1e-12, "no terminal marks");
    let tiers: Vec<(f64, f64, f64, usize)> = cohort
        .subjects
        .iter()
        .enumerate()
        .map(|(i, subject)| {
            let score = cohort.covariates[[i, 0]];
            let alone = population_forecast(
                &fit,
                &cohort,
                &PopulationForecastRequest {
                    start: subject.exit,
                    horizons: &horizons,
                    future: &at(subject.exit, score),
                },
            )
            .expect("score-only forecast");
            let with_history = forecast(
                &fit,
                &cohort,
                &ForecastRequest {
                    history: subject,
                    horizons: &horizons,
                    future: &[],
                },
            )
            .expect("history forecast");
            (score, alone.expected_counts[[1, 0]], with_history.expected_counts[[1, 0]], subject.events.len())
        })
        .collect();
    let population_count = population.expected_counts[[1, 0]];
    emit(&format!("[tiers] population expected count by t=8: {population_count:.4}"));
    for tier in tiers.iter() {
        let (score, alone) = (tier.0, tier.1);
        assert!(
            (score > 0.0) == (alone > population_count) || score == 0.0,
            "score {score}: score-only tier {alone} against population {population_count}"
        );
    }
    // Within a band of near-population scores, the subject richest in events
    // sits above its score-only tier and the poorest below it.
    let band: Vec<&(f64, f64, f64, usize)> = tiers.iter().filter(|t| t.0.abs() < 0.5).collect();
    assert!(band.len() >= 4, "too few near-population scores to compare histories");
    let richest = band.iter().max_by_key(|t| t.3).expect("richest");
    let poorest = band.iter().min_by_key(|t| t.3).expect("poorest");
    for (label, t) in [("richest", richest), ("poorest", poorest)] {
        emit(&format!(
            "[tiers] {label}: score={:.3} events={} score-only={:.4} with-history={:.4}",
            t.0, t.3, t.1, t.2
        ));
    }
    assert!(richest.3 > poorest.3, "the band must contain unequal histories");
    // The history tier is compared against another history tier, not against
    // the score-only one. Observing a history does two things at once: it
    // moves the latent mean, and it narrows the latent variance. The
    // narrowing alone lowers `E[exp(a z)]` for any subject — the intensity
    // is convex in the state — so a rich history can carry more risk than
    // its score alone implied and still forecast below the score-only tier.
    // Between two subjects of the same score, both narrowed, what remains is
    // the information their histories carry.
    assert!(
        richest.2 > poorest.2,
        "within one score band, the history rich in events ({} events, {:.4}) must forecast above the poor one ({} events, {:.4})",
        richest.3,
        richest.2,
        poorest.3,
        poorest.2
    );
}

/// Intercept-only fit of a cohort: the maximum-likelihood rate of each mark
/// is its event count over its exposure, so every forecast has a closed
/// form to compare against.
fn intercept_only_spec() -> TermCollectionSpec {
    TermCollectionSpec {
        linear_terms: Vec::new(),
        random_effect_terms: Vec::new(),
        smooth_terms: Vec::new(),
    }
}

/// A competing-risks cohort: two terminal marks and one recurrent mark,
/// constant hazards, no latent state in the simulation.
fn competing_risks_cohort(seed: u64) -> EventHistoryCohort {
    simulate_marked_cohort(
        120,
        4.0,
        &[-1.4, -2.0, -0.6],
        0.0,
        &[0.0, 0.0, 0.0],
        1.0,
        &[MarkKind::Terminal, MarkKind::Terminal, MarkKind::Recurrent],
        seed,
    )
}

#[test]
fn terminal_forecasts_match_the_constant_hazard_solution() {
    install_test_logger();
    let mut cohort = competing_risks_cohort(31);
    let spec = EventHistorySpec::new(0, vec![intercept_only_spec()]);
    let fit = fit_event_history(&mut cohort, &spec).expect("intercept-only fit");
    // The maximum-likelihood rates: events over exposure, per mark.
    let exposure: f64 = cohort.subjects.iter().map(|s| s.exit - s.entry).sum();
    let counts: Vec<f64> = (0..3)
        .map(|d| cohort.subjects.iter().flat_map(|s| s.events.iter()).filter(|e| e.mark == d).count() as f64)
        .collect();
    let rates: Vec<f64> = counts.iter().map(|c| c / exposure).collect();
    for d in 0..3 {
        let fitted = fit.mark_coefficients(d)[0].exp();
        // The agreement is limited by the inner solve's own convergence
        // tolerance, not by the model: the closed form IS the maximiser.
        assert!(
            (fitted - rates[d]).abs() < 1e-4 * rates[d],
            "mark {d}: fitted rate {fitted} vs closed form {}",
            rates[d]
        );
    }
    let total_terminal = rates[0] + rates[1];
    // A censored subject: the forecast is the exponential competing-risks
    // solution, chronologically integrated.
    let censored = cohort
        .subjects
        .iter()
        .find(|s| s.terminal_event(&cohort.mark_kinds).is_none())
        .expect("a censored subject");
    let offsets = [0.5, 1.0, 2.0, 4.0];
    let horizons: Vec<f64> = offsets.iter().map(|h| censored.exit + h).collect();
    let f = forecast(
        &fit,
        &cohort,
        &ForecastRequest {
            history: censored,
            horizons: &horizons,
            future: &[],
        },
    )
    .expect("forecast");
    for (i, &h) in offsets.iter().enumerate() {
        let survival = (-total_terminal * h).exp();
        assert!(
            (f.survival[i] - survival).abs() < 1e-4,
            "survival at +{h}: {} vs {survival}",
            f.survival[i]
        );
        for d in 0..2 {
            let incidence = rates[d] / total_terminal * (1.0 - survival);
            assert!(
                (f.expected_counts[[i, d]] - incidence).abs() < 1e-4,
                "cumulative incidence of mark {d} at +{h}: {} vs {incidence}",
                f.expected_counts[[i, d]]
            );
        }
        // Recurrent events before termination: λ₂ ∫₀ʰ S = λ₂ (1 − S)/Λ.
        let recurrent = rates[2] / total_terminal * (1.0 - survival);
        assert!(
            (f.expected_counts[[i, 2]] - recurrent).abs() < 1e-4,
            "expected recurrent count at +{h}: {} vs {recurrent}",
            f.expected_counts[[i, 2]]
        );
    }
    // A subject who died has no future.
    let dead = cohort
        .subjects
        .iter()
        .find(|s| s.terminal_event(&cohort.mark_kinds).is_some())
        .expect("a subject with a terminal event");
    let gone = forecast(
        &fit,
        &cohort,
        &ForecastRequest {
            history: dead,
            horizons: &[dead.exit + 1.0],
            future: &[],
        },
    )
    .expect("forecast of an absorbed subject");
    assert_eq!(gone.survival, vec![0.0]);
    assert!(gone.expected_counts.iter().all(|c| *c == 0.0));
    // The population tier at the same rates is the same solution.
    let population = population_forecast(
        &fit,
        &cohort,
        &PopulationForecastRequest {
            start: 1.0,
            horizons: &[2.0, 3.0],
            future: &[FutureSegment {
                start: 1.0,
                covariates: vec![0.0],
            }],
        },
    )
    .expect("population forecast");
    assert!((population.survival[1] - (-2.0 * total_terminal).exp()).abs() < 1e-4);
    // Rosenblatt: with constant hazards the PIT of an event at `t` after the
    // previous event at `s` is `1 − exp(−Λ_all (t − s))` with the total rate.
    let total_rate: f64 = rates.iter().sum();
    let subject = cohort
        .subjects
        .iter()
        .max_by_key(|s| s.events.len())
        .expect("subject");
    let pits = predictive_pit(&fit, &cohort, subject).expect("pit");
    assert_eq!(pits.len(), subject.events.len());
    let mut previous = subject.entry;
    for (event, pit) in subject.events.iter().zip(pits.iter()) {
        let expected = 1.0 - (-total_rate * (event.time - previous)).exp();
        assert!(
            (pit.pit - expected).abs() < 1e-4,
            "PIT at {}: {} vs {expected}",
            event.time,
            pit.pit
        );
        assert_eq!(pit.mark, event.mark);
        let probability_sum: f64 = pit.mark_probabilities.iter().sum();
        assert!((probability_sum - 1.0).abs() < 1e-12);
        for d in 0..3 {
            assert!((pit.mark_probabilities[d] - rates[d] / total_rate).abs() < 1e-6);
        }
        previous = event.time;
    }
    assert!(kolmogorov_smirnov_uniform(&[]).is_none());
}

#[test]
fn forecast_probabilities_are_coherent_under_a_latent_state() {
    install_test_logger();
    let mut cohort = simulate_marked_cohort(
        60,
        4.0,
        &[-1.2, -1.8, -0.5],
        0.4,
        &[0.9, 0.5, 0.7],
        0.5,
        &[MarkKind::Terminal, MarkKind::Once, MarkKind::Recurrent],
        41,
    );
    let mut spec = EventHistorySpec::new(1, vec![linear_spec()]);
    spec.gauss_hermite_order = 9;
    let started = std::time::Instant::now();
    let fit = fit_event_history(&mut cohort, &spec).expect("fit");
    emit(&format!(
        "[coherent] {:.1}s order={} refinement={} loadings={:?} rate={}",
        started.elapsed().as_secs_f64(),
        fit.quadrature.gauss_hermite_order,
        fit.quadrature.mesh_refinement,
        fit.loadings.iter().copied().collect::<Vec<_>>(),
        fit.rates[0]
    ));
    let horizons_after = [0.5, 1.5, 3.0];
    for subject in cohort.subjects.iter().take(12) {
        if subject.terminal_event(&cohort.mark_kinds).is_some() {
            continue;
        }
        let horizons: Vec<f64> = horizons_after.iter().map(|h| subject.exit + h).collect();
        let f = forecast(
            &fit,
            &cohort,
            &ForecastRequest {
                history: subject,
                horizons: &horizons,
                future: &[],
            },
        )
        .expect("forecast");
        let had_once = subject.events.iter().any(|e| e.mark == 1);
        let mut previous_survival = 1.0;
        let mut previous_counts = vec![0.0; 3];
        for i in 0..horizons.len() {
            let s = f.survival[i];
            assert!((0.0..=1.0).contains(&s) && s <= previous_survival + 1e-12, "survival {s}");
            // One terminal mark: its cumulative incidence is 1 − S.
            assert!(
                (f.expected_counts[[i, 0]] - (1.0 - s)).abs() < 1e-4,
                "subject {}: F_terminal {} vs 1 − S {}",
                subject.id,
                f.expected_counts[[i, 0]],
                1.0 - s
            );
            // A once-only mark: a probability, zero if it already fired.
            let once = f.expected_counts[[i, 1]];
            if had_once {
                assert_eq!(once, 0.0);
            } else {
                assert!((0.0..=1.0 + 1e-9).contains(&once), "first-occurrence probability {once}");
                assert!(once + f.expected_counts[[i, 0]] <= 1.0 + 1e-6, "once + terminal exceeds one");
            }
            for d in 0..3 {
                assert!(f.expected_counts[[i, d]] >= previous_counts[d] - 1e-12);
                previous_counts[d] = f.expected_counts[[i, d]];
            }
            previous_survival = s;
        }
    }
    // The PITs carry mark probabilities that sum to one over the marks the
    // subject was at risk for.
    for subject in cohort.subjects.iter().take(12) {
        for pit in predictive_pit(&fit, &cohort, subject).expect("pit") {
            assert!((0.0..=1.0).contains(&pit.pit));
            let sum: f64 = pit.mark_probabilities.iter().sum();
            assert!((sum - 1.0).abs() < 1e-9, "mark probabilities sum to {sum}");
        }
    }
}

#[test]
fn the_latent_block_starts_its_atoms_apart() {
    // The likelihood is invariant under flipping an atom's sign with its
    // state, under permuting atoms, and — at equal rates — under rotating
    // them. The initial point is where that gauge is fixed: every loading at
    // one latent standard deviation per unit of log-intensity, and the
    // log-rates spaced by `ln 2` around the ridge centre, so a deterministic
    // Newton cannot keep two atoms identical to each other.
    for atoms in [1usize, 2, 3] {
        let latent = super::family::latent_block_spec(400, 2, atoms).expect("latent spec");
        let initial = latent.initial_beta.expect("initial");
        assert_eq!(initial.len(), 2 * atoms + atoms);
        for q in 0..2 * atoms {
            assert_eq!(initial[q], 1.0, "loading {q} of {atoms} atoms");
        }
        let log_rates: Vec<f64> = (0..atoms).map(|k| initial[2 * atoms + k]).collect();
        for pair in log_rates.windows(2) {
            assert!(
                (pair[1] - pair[0] - std::f64::consts::LN_2).abs() < 1e-12,
                "the log-rates must be spaced by ln 2: {log_rates:?}"
            );
        }
        let mean: f64 = log_rates.iter().sum::<f64>() / atoms as f64;
        assert!(mean.abs() < 1e-12, "the spacing is centred on the ridge: {log_rates:?}");
        assert_eq!(latent.penalties.len(), atoms, "one ridge per atom");
    }
}
