use super::chain::{AtomTransition, GaussHermite, Grid, backward_operator, forward_operator};
use super::cohort::{
    CovariateSegment, Event, EventHistoryCohort, SubjectHistory, SubjectNodes, expand_nodes,
};
use super::family::{EventHistoryFamily, EventHistorySpec, fit_event_history};
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
    let forward = forward_operator(&gh, &from, &to, &[transition.clone()]);
    let predicted = forward.apply(&values);
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
    let predicted = forward.apply(&other);
    for j in 0..to.size() {
        let z = *to.coordinate(j, 0);
        let exact = gaussian(z, phi * mu0, phi * phi * sigma0 * sigma0 + q);
        assert!(
            (predicted[j] - exact).abs() < 1e-4 * exact.abs().max(1e-3),
            "node {j}: predicted {} exact {exact}",
            predicted[j]
        );
    }
    let backward = backward_operator(&gh, &from, &to, &[transition]);
    let ones = vec![1.0; to.size()];
    let integrated = backward.apply(&ones);
    for (i, v) in integrated.iter().enumerate() {
        assert!((v - 1.0).abs() < 1e-9, "backward of one at {i}: {v}");
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
    let evaluate = |c: &[f64]| -> f64 {
        let mut total = 0.0;
        for a in 0..5 {
            for b in 0..5 {
                total += c[a * 5 + b] * z.powi(a as i32) * zp.powi(b as i32);
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
            assert!((out.gradient[n * 2 + d] - score).abs() < 1e-10);
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
            assert!(
                (base.hessian[i * p + j] - fd_h).abs() < 1e-5 * (1.0 + fd_h.abs()),
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
            assert!(
                (base.hessian[[i, j]] - fd_h).abs() < 1e-5 * (1.0 + fd_h.abs()),
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
    let mut cohort = simulate_cohort(80, 6.0, -0.8, 0.5, 1.0, 0.4, 7);
    let mut spec = EventHistorySpec::new(1, vec![linear_spec()]);
    spec.gauss_hermite_order = 11;
    let fit = fit_event_history(&mut cohort, &spec).expect("fit");
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
    assert!(f.survival.iter().all(|&s| (0.0..=1.0).contains(&s)));
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

#[test]
fn a_cohort_without_shared_risk_switches_the_atom_off() {
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
