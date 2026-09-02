use super::chain::{
    AtomTransition, GaussHermite, Grid, backward_axis_bases, forward_operators,
    interpolate_at_inner_points,
};
use super::cohort::{
    CovariateSegment, Event, EventHistoryCohort, SubjectHistory, SubjectNodes, expand_nodes,
};
use super::family::{Directional, EventHistoryFamily, EventHistorySpec, fit_event_history};
use super::forecast::{ForecastRequest, forecast, kolmogorov_smirnov_uniform, predictive_pit};
use super::marginal::{SubjectInputs, subject_marginal};
use crate::custom_family::ParameterBlockState;
use gam_math::jet_scalar::{OneSeed, TwoSeed};
use gam_math::nested_dual::JetField;
use gam_terms::smooth::{LinearCoefficientGeometry, LinearTermSpec, TermCollectionSpec};
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
    SubjectNodes {
        first_row: 0,
        times: times.to_vec(),
        gaps: times.windows(2).map(|w| w[1] - w[0]).collect(),
        exposures: exposures.to_vec(),
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
    };
    let out = subject_marginal(&inputs, false).expect("marginal");
    // ∫ exp(y η − w e^η) N(z) dz with η = η0 + a z, on a fine grid.
    let mut integral = 0.0;
    let steps = 200_000;
    let (lo, hi) = (-9.0, 9.0);
    let dz = (hi - lo) / steps as f64;
    for i in 0..=steps {
        let z = lo + i as f64 * dz;
        let eta = eta0[0] + loadings[0] * z;
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
        let eta1 = eta0[0] + loadings[0] * z1;
        let l1 = (eta1 - 0.5 * eta1.exp()).exp() * gaussian(z1, 0.0, 1.0);
        let mut inner = 0.0;
        for j in 0..=steps {
            let z2 = lo + j as f64 * dz;
            let w2 = if j == 0 || j == steps { 0.5 } else { 1.0 };
            let eta2 = eta0[1] + loadings[0] * z2;
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
    };
    let out = subject_marginal(&inputs, true).expect("marginal");
    let mut expected = 0.0;
    for n in 0..3 {
        for d in 0..2 {
            let eta = eta0[n * 2 + d];
            expected += nodes.counts[[n, d]] * eta - nodes.exposures[n] * eta.exp();
        }
    }
    assert!((out.loglik - expected).abs() < 1e-12);
    for n in 0..3 {
        for d in 0..2 {
            let eta = eta0[n * 2 + d];
            let score = nodes.counts[[n, d]] - nodes.exposures[n] * eta.exp();
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
        mark_names: vec!["a".to_string(), "b".to_string()],
        covariate_names: vec!["x".to_string()],
        covariates: array![[1.0], [2.0]],
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
            ],
            segments: vec![
                CovariateSegment {
                    start: 0.0,
                    row: 0,
                },
                CovariateSegment {
                    start: 3.0,
                    row: 1,
                },
            ],
        }],
    };
    cohort.validate().expect("valid");
    let nodes = expand_nodes(&cohort, 5).expect("nodes");
    let s = &nodes.subjects[0];
    let exposure: f64 = s.exposures.iter().sum();
    assert!((exposure - 3.0).abs() < 1e-12, "exposure {exposure}");
    let event_node = s.times.iter().position(|&t| t == 2.5).expect("event node");
    assert_eq!(s.counts[[event_node, 0]], 1.0);
    assert_eq!(s.counts[[event_node, 1]], 1.0);
    assert_eq!(s.exposures[event_node], 0.0);
    assert!(s.gaps.iter().all(|&g| g > 0.0));
    for (n, &t) in s.times.iter().enumerate() {
        let expected_row = if t >= 3.0 { 1 } else { 0 };
        assert_eq!(s.covariate_rows[n], expected_row);
        assert_eq!(nodes.node_data[[n, 0]], cohort.covariates[[expected_row, 0]]);
        assert_eq!(nodes.node_data[[n, 1]], t);
    }
}

/// Simulate a cohort whose log-intensity is `β₀ + β₁ x + a z(t)` with `z` a
/// unit-variance Ornstein–Uhlenbeck atom of the given rate, by thinning.
fn simulate_cohort(
    subjects: usize,
    follow_up: f64,
    intercept: f64,
    slope: f64,
    loading: f64,
    rate: f64,
    seed: u64,
) -> EventHistoryCohort {
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
        let mut events = Vec::new();
        let bound = (intercept + slope * x + loading.abs() * 4.0).exp();
        let mut t = 0.0;
        while t < follow_up {
            let mut candidate = t - bound.recip() * rng.uniform().max(1e-300).ln();
            // advance the state to the candidate in `dt` steps
            while t + dt <= candidate && t + dt <= follow_up {
                z = phi * z + innovation * rng.normal();
                t += dt;
            }
            if candidate > follow_up {
                break;
            }
            let intensity = (intercept + slope * x + loading * z).exp();
            if rng.uniform() * bound < intensity {
                events.push(Event {
                    time: candidate,
                    mark: 0,
                });
            }
            if candidate <= t {
                candidate = t + 1e-9;
            }
            t = candidate;
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
        covariate_names: vec!["x".to_string()],
        covariates,
        subjects: histories,
    }
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
    let nodes = Arc::new(expand_nodes(&cohort, 3).expect("nodes"));
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
    println!(
        "[fit] {:.1}s outer_iterations={} gh_order={} loading={} rate={} log_lambda={:?}",
        started.elapsed().as_secs_f64(),
        fit.fit.outer_iterations,
        fit.quadrature.order,
        fit.loadings[[0, 0]],
        fit.rates[0],
        fit.atom_log_lambdas
    );
    let beta = fit.mark_coefficients(0);
    assert_eq!(beta.len(), 2, "intercept and slope");
    assert!(
        (beta[1] - 0.5).abs() < 0.25,
        "slope {} should recover 0.5 within its sampling error",
        beta[1]
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
    assert!(
        (fit.quadrature.log_likelihood - fit.quadrature.log_likelihood_at_checked_order).abs()
            <= spec.quadrature_tolerance * fit.quadrature.log_likelihood.abs().max(1.0)
    );
    // Forecast: probabilities and expected counts are coherent.
    let request = ForecastRequest {
        history: &cohort.subjects[0],
        horizons: &[6.5, 7.0, 8.0],
        absorbing: &[false],
        future_row: 0,
    };
    let f = forecast(&fit, &cohort, &request).expect("forecast");
    assert!(
        f.survival.iter().all(|&s| (0.0..=1.0).contains(&s)),
        "survival left [0, 1]: {:?}",
        f.survival
    );
    assert!((f.survival[0] - 1.0).abs() < 1e-12, "no absorbing marks: survival stays one");
    assert!(f.expected_counts[[0, 0]] <= f.expected_counts[[1, 0]]);
    assert!(f.expected_counts[[1, 0]] <= f.expected_counts[[2, 0]]);
    assert!(f.expected_counts[[2, 0]] > 0.0);
    // Predictive PIT: uniform up to sampling error on the training cohort.
    let mut pits = Vec::new();
    for subject in &cohort.subjects {
        pits.extend(predictive_pit(&fit, &cohort, subject).expect("pit"));
    }
    assert!(pits.iter().all(|&u| (0.0..=1.0).contains(&u)));
    let ks = kolmogorov_smirnov_uniform(&pits);
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
    let nodes = expand_nodes(&cohort, 3).expect("nodes");
    let spec = super::formula::covariate_spec_from_formula("x + s(time)", &nodes, &cohort.covariate_names)
        .expect("spec");
    assert_eq!(spec.linear_terms.len(), 1);
    assert_eq!(spec.linear_terms[0].feature_col, 0);
    assert_eq!(spec.smooth_terms.len(), 1);
    let error = super::formula::covariate_spec_from_formula("nope", &nodes, &cohort.covariate_names)
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
    let nodes = Arc::new(expand_nodes(&cohort, 9).expect("nodes"));
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
    let nodes = Arc::new(expand_nodes(cohort, 9).expect("nodes"));
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
    let nodes = expand_nodes(&cohort, 9).expect("nodes");
    let mut report = Vec::new();
    for (s, subj) in nodes.subjects.iter().enumerate() {
        let n = subj.len();
        let run = |len: usize| -> bool {
            let prefix = SubjectNodes {
                first_row: 0,
                times: subj.times[..len].to_vec(),
                gaps: subj.gaps[..len - 1].to_vec(),
                exposures: subj.exposures[..len].to_vec(),
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
            "subject {s}: {n} nodes, first non-finite prefix {bad}; node {}: time {} gap {} exposure {} counts {:?}; prefix log-lik at {bad}: {:?}",
            bad - 1,
            subj.times[bad - 1],
            if bad >= 2 { subj.gaps[bad - 2] } else { 0.0 },
            subj.exposures[bad - 1],
            subj.counts.row(bad - 1).to_vec(),
            {
                let prefix = SubjectNodes {
                    first_row: 0,
                    times: subj.times[..bad].to_vec(),
                    gaps: subj.gaps[..bad - 1].to_vec(),
                    exposures: subj.exposures[..bad].to_vec(),
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
fn louis_hessian_against_finite_differences_at_the_loaded_cohort_optimum() {
    // The state the fixed-λ inner solve reaches on the loaded cohort. Louis'
    // Hessian is compared with finite differences of the exact gradient at
    // two step sizes (a ripple in the objective shows up as step-size
    // dependence) and at two quadrature orders.
    let cohort = loaded_cohort();
    let beta = array![-0.9485, 0.5452];
    let latent = array![1.2054, 1.0587];
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
        let p = 4;
        let mut full = Array1::<f64>::zeros(p);
        full.slice_mut(ndarray::s![0..2]).assign(&beta);
        full.slice_mut(ndarray::s![2..4]).assign(&latent);
        let split = |v: &Array1<f64>| -> (Array1<f64>, Array1<f64>) {
            (v.slice(ndarray::s![0..2]).to_owned(), v.slice(ndarray::s![2..4]).to_owned())
        };
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
                    assert!(
                        (base.gradient[i] - fd_gradient).abs() < 1e-6 * (1.0 + fd_gradient.abs()),
                        "G={order}: gradient {i} exact {} vs finite difference {fd_gradient}",
                        base.gradient[i]
                    );
                }
                if order == 21 && h == 1e-5 {
                    for j in 0..p {
                        let fd_h = -(gp[j] - gm[j]) / (2.0 * h);
                        assert!(
                            (base.hessian[[i, j]] - fd_h).abs() < 1e-2 * (1.0 + fd_h.abs()),
                            "G=21: hessian[{i},{j}] louis {} vs finite difference {fd_h}",
                            base.hessian[[i, j]]
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn spline_basis_reproduces_lines_and_cubics_and_never_overshoots() {
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
    // A steep, smooth function with a wide range does not overshoot: the
    // spline stays within a small margin of the data's range everywhere on
    // the hull, where the degree-20 Lagrange interpolant would not.
    let steep: Vec<f64> = gh.nodes.iter().map(|x| -12.0 * (x + 1.0).abs().powf(1.5) + 3.0).collect();
    let (lo, hi) = steep.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(l, h), v| (l.min(*v), h.max(*v)));
    let margin = 0.1 * (hi - lo);
    let mut lagrange_overshoot = 0.0_f64;
    for step in 0..400 {
        let x = gh.nodes[0] + (gh.nodes[g - 1] - gh.nodes[0]) * step as f64 / 399.0;
        let spline: f64 = gh.spline_basis(&x).iter().zip(steep.iter()).map(|(b, f)| b * f).sum();
        assert!(spline >= lo - margin && spline <= hi + margin, "spline overshoot at {x}: {spline} outside [{lo}, {hi}]");
        let lagrange: f64 = gh.lagrange_basis(&x).iter().zip(steep.iter()).map(|(b, f)| b * f).sum();
        lagrange_overshoot = lagrange_overshoot.max((lagrange - hi).max(lo - lagrange));
    }
    assert!(lagrange_overshoot > hi - lo, "the control did not overshoot: {lagrange_overshoot}");
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
    let nodes = Arc::new(expand_nodes(&cohort, 9).expect("nodes"));
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
    let nodes = Arc::new(expand_nodes(&cohort, 9).expect("nodes"));
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
