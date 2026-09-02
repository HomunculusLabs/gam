use super::cohort::{
    CovariateSegment, Event, EventHistoryCohort, MarkKind, SubjectHistory, SubjectNodes,
    expand_nodes,
};
use super::covariance::{SubjectResiduals, best_new_atom, covariance_score};
use super::family::{
    EventHistoryFamily, EventHistoryFit, EventHistorySpec, fit_event_history,
    fit_event_history_formula, seeded_one, seeded_two,
};
use super::forecast::{
    ForecastRequest, FutureSegment, PopulationForecastRequest, forecast,
    kolmogorov_smirnov_uniform, latent_exposure, latent_state, population_forecast, predictive_pit,
};
use super::laplace::{self, SubjectInputs, evidence, filter_pass, find_mode};
use super::scalar::Tangent;
use crate::custom_family::{BlockwiseFitOptions, ParameterBlockState};
use gam_math::nested_dual::JetField;
use gam_terms::smooth::{
    LinearCoefficientGeometry, LinearTermSpec, TermCollectionSpec, build_term_collection_design,
};
use ndarray::{Array1, Array2, array};
use std::sync::Arc;

fn gaussian(x: f64, mean: f64, variance: f64) -> f64 {
    (-(x - mean).powi(2) / (2.0 * variance)).exp() / (2.0 * std::f64::consts::PI * variance).sqrt()
}

/// A subject whose every mark is recurrent: exposure equals the node weight.
fn subject(times: &[f64], weights: &[f64], counts: &[Vec<f64>]) -> SubjectNodes {
    let marks = counts[0].len();
    let n = times.len();
    let mut count_matrix = Array2::<f64>::zeros((n, marks));
    let mut exposures = Array2::<f64>::zeros((n, marks));
    for (node, row) in counts.iter().enumerate() {
        for (d, &c) in row.iter().enumerate() {
            count_matrix[[node, d]] = c;
            exposures[[node, d]] = weights[node];
        }
    }
    SubjectNodes {
        first_row: 0,
        times: times.to_vec(),
        gaps: times.windows(2).map(|w| w[1] - w[0]).collect(),
        weights: weights.to_vec(),
        exposures,
        counts: count_matrix,
        covariate_rows: vec![0; n],
    }
}

fn inputs<'a, S: JetField>(
    nodes: &'a SubjectNodes,
    eta0: &'a [S],
    loadings: &'a [S],
    log_rates: &'a [S],
) -> SubjectInputs<'a, S> {
    SubjectInputs {
        nodes,
        eta0,
        loadings,
        log_rates,
        time_scale: 1.0,
    }
}

/// Value and gradient at `theta = [η⁰ | a | ρ]` on `f64`.
fn evaluate(nodes: &SubjectNodes, marks: usize, atoms: usize, theta: &[f64], derivatives: bool) -> (f64, Vec<f64>) {
    let n = nodes.len();
    let eta0 = &theta[..n * marks];
    let loadings = &theta[n * marks..n * marks + marks * atoms];
    let log_rates = &theta[n * marks + marks * atoms..];
    let inputs = inputs(nodes, eta0, loadings, log_rates);
    let mode = find_mode(&inputs, None).expect("mode");
    let out = evidence(&inputs, &mode, derivatives).expect("evidence");
    (out.loglik, out.gradient)
}

fn probe_subject() -> (SubjectNodes, Vec<f64>) {
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
    let mut theta = vec![0.2, -0.1, 0.4, 0.0, -0.3, 0.5, 0.1, 0.2];
    theta.extend([0.8, -0.4, 0.3, 0.6]);
    theta.extend([-0.2, 0.5]);
    (nodes, theta)
}

#[test]
fn zero_loadings_reduce_to_the_poisson_likelihood_exactly() {
    let nodes = subject(
        &[0.0, 0.5, 1.5],
        &[0.4, 0.0, 0.7],
        &[vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 2.0]],
    );
    let eta0 = [0.1, -0.3, 0.5, 0.2, -0.1, 0.4];
    let loadings = [0.0, 0.0, 0.0, 0.0];
    let log_rates = [0.0, 0.3];
    let inputs = inputs(&nodes, &eta0, &loadings, &log_rates);
    let mode = find_mode(&inputs, None).expect("mode");
    assert!(mode.iter().all(|z| z.abs() < 1e-12), "mode {mode:?}");
    let out = evidence(&inputs, &mode, true).expect("evidence");
    let mut expected = 0.0;
    for n in 0..3 {
        for d in 0..2 {
            let eta = eta0[n * 2 + d];
            expected += nodes.counts[[n, d]] * eta - nodes.exposures[[n, d]] * eta.exp();
        }
    }
    assert!((out.loglik - expected).abs() < 1e-12, "{} vs {expected}", out.loglik);
    for n in 0..3 {
        for d in 0..2 {
            let eta = eta0[n * 2 + d];
            let score = nodes.counts[[n, d]] - nodes.exposures[[n, d]] * eta.exp();
            assert!((out.gradient[n * 2 + d] - score).abs() < 1e-12);
        }
    }
    // At zero loading the evidence is even in every loading and the rates
    // are unidentified: both gradients vanish.
    for q in 6..12 {
        assert!(out.gradient[q].abs() < 1e-12, "gradient[{q}] = {}", out.gradient[q]);
    }
}

#[test]
fn single_node_evidence_approaches_the_exact_marginal_with_information() {
    // One node, one atom, one mark: the exact marginal is a one-dimensional
    // integral. The Laplace error is O(1 / expected count); scaling the
    // counts and exposure by 100 must shrink it by more than an order.
    let errors: Vec<f64> = [1.0, 100.0]
        .iter()
        .map(|&scale| {
            let nodes = subject(&[1.0], &[0.8 * scale], &[vec![2.0 * scale]]);
            let eta0 = [0.3];
            let loadings = [0.9];
            let log_rates = [0.0];
            let inputs = inputs(&nodes, &eta0, &loadings, &log_rates);
            let mode = find_mode(&inputs, None).expect("mode");
            let laplace = evidence(&inputs, &mode, false).expect("evidence").loglik;
            let shift = -0.5 * loadings[0] * loadings[0];
            let mut integral = 0.0;
            let steps = 400_000;
            let (lo, hi) = (-10.0, 10.0);
            let dz = (hi - lo) / steps as f64;
            let mut log_terms = Vec::with_capacity(steps + 1);
            for i in 0..=steps {
                let z = lo + i as f64 * dz;
                let eta = eta0[0] + shift + loadings[0] * z;
                let weight = if i == 0 || i == steps { 0.5 } else { 1.0 };
                log_terms.push((2.0 * scale * eta - 0.8 * scale * eta.exp()) + gaussian(z, 0.0, 1.0).ln() + (weight * dz).ln());
            }
            let top = log_terms.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            for t in &log_terms {
                integral += (t - top).exp();
            }
            let exact = top + integral.ln();
            let error = (laplace - exact).abs();
            println!("scale {scale}: laplace {laplace} exact {exact} error {error}");
            error
        })
        .collect();
    assert!(errors[0] < 0.1, "Laplace error at unit scale {}", errors[0]);
    assert!(errors[1] < errors[0] / 5.0, "errors {errors:?}");
    assert!(errors[1] < 1e-3, "Laplace error at scale 100 is {}", errors[1]);
}

#[test]
fn two_node_evidence_against_the_brute_force_double_integral_on_a_sparse_history() {
    // One event and one exposure node: the sparsest history there is, where
    // the Laplace approximation of a log-Gaussian Cox path is least
    // accurate. The error is printed and bounded, not hidden.
    let nodes = subject(&[0.0, 0.7], &[0.5, 0.6], &[vec![1.0], vec![0.0]]);
    let eta0 = [-0.2, 0.4];
    let loadings = [1.1];
    let log_rates = [0.2];
    let inputs = inputs(&nodes, &eta0, &loadings, &log_rates);
    let mode = find_mode(&inputs, None).expect("mode");
    let laplace = evidence(&inputs, &mode, false).expect("evidence").loglik;
    let shift = -0.5 * loadings[0] * loadings[0];
    let phi = (-(0.2f64.exp() * 0.7)).exp();
    let q = 1.0 - phi * phi;
    let steps = 1200;
    let (lo, hi) = (-7.0, 7.0);
    let dz = (hi - lo) / steps as f64;
    let mut integral = 0.0;
    for i in 0..=steps {
        let z1 = lo + i as f64 * dz;
        let w1 = if i == 0 || i == steps { 0.5 } else { 1.0 };
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
    let exact = integral.ln();
    let error = laplace - exact;
    println!("sparse two-node history: laplace {laplace} exact {exact} error {error}");
    assert!(error.abs() < 0.05, "Laplace error {error} nats on a one-event history");
}

#[test]
fn the_mode_is_the_maximiser_of_the_complete_data_objective() {
    let (nodes, theta) = probe_subject();
    let n = nodes.len();
    let (marks, atoms) = (2, 2);
    let eta0 = &theta[..n * marks];
    let loadings = &theta[n * marks..n * marks + marks * atoms];
    let log_rates = &theta[n * marks + marks * atoms..];
    let inputs = inputs(&nodes, eta0, loadings, log_rates);
    let mode = find_mode(&inputs, None).expect("mode");
    let warm = find_mode(&inputs, Some(&vec![0.7; mode.len()])).expect("warm mode");
    for (a, b) in mode.iter().zip(warm.iter()) {
        assert!((a - b).abs() < 1e-9, "mode depends on its start: {a} vs {b}");
    }
    let smoother = laplace::smoother(&inputs, &mode).expect("smoother");
    for node in 0..n {
        let cov = &smoother.covariances[node * atoms * atoms..(node + 1) * atoms * atoms];
        assert!(cov[0] > 0.0 && cov[3] > 0.0 && (cov[1] - cov[2]).abs() < 1e-12, "{cov:?}");
        assert!(cov[0] <= 1.0 + 1e-9 && cov[3] <= 1.0 + 1e-9, "posterior variance exceeds the prior: {cov:?}");
    }
}

#[test]
fn evidence_gradient_is_the_derivative_of_the_computed_value() {
    let (nodes, theta) = probe_subject();
    let (marks, atoms) = (2, 2);
    let (_, gradient) = evaluate(&nodes, marks, atoms, &theta, true);
    let h = 1e-5;
    for i in 0..theta.len() {
        let mut plus = theta.clone();
        plus[i] += h;
        let mut minus = theta.clone();
        minus[i] -= h;
        let fd = (evaluate(&nodes, marks, atoms, &plus, false).0 - evaluate(&nodes, marks, atoms, &minus, false).0) / (2.0 * h);
        assert!(
            (gradient[i] - fd).abs() < 1e-7 * (1.0 + fd.abs()),
            "gradient[{i}] = {} vs finite difference {fd}",
            gradient[i]
        );
    }
}

#[test]
fn tangent_channels_of_the_gradient_are_hessian_columns() {
    let (nodes, theta) = probe_subject();
    let n = nodes.len();
    let (marks, atoms) = (2, 2);
    let p = theta.len();
    const W: usize = 16;
    assert!(p <= W);
    let jets: Vec<Tangent<f64, W>> = (0..p)
        .map(|i| {
            let mut grad = [0.0; W];
            grad[i] = 1.0;
            Tangent::seeded(theta[i], grad)
        })
        .collect();
    let inputs_jet = inputs(&nodes, &jets[..n * marks], &jets[n * marks..n * marks + marks * atoms], &jets[n * marks + marks * atoms..]);
    let plain = inputs(&nodes, &theta[..n * marks], &theta[n * marks..n * marks + marks * atoms], &theta[n * marks + marks * atoms..]);
    let mode = find_mode(&plain, None).expect("mode");
    let jet = evidence(&inputs_jet, &mode, true).expect("jet evidence");
    let value = evidence(&plain, &mode, true).expect("evidence");
    assert!((jet.loglik.value - value.loglik).abs() < 1e-12);
    let h = 1e-5;
    for j in 0..p {
        let mut plus = theta.clone();
        plus[j] += h;
        let mut minus = theta.clone();
        minus[j] -= h;
        let gp = evaluate(&nodes, marks, atoms, &plus, true).1;
        let gm = evaluate(&nodes, marks, atoms, &minus, true).1;
        for i in 0..p {
            let fd = (gp[i] - gm[i]) / (2.0 * h);
            let dual = jet.gradient[i].grad[j];
            assert!(
                (dual - fd).abs() < 1e-6 * (1.0 + fd.abs()),
                "H[{i},{j}] dual {dual} vs finite difference {fd}"
            );
            let sym = jet.gradient[j].grad[i];
            assert!((dual - sym).abs() < 1e-8 * (1.0 + dual.abs()), "asymmetric H[{i},{j}]: {dual} vs {sym}");
        }
    }
}

#[test]
fn directional_duals_match_finite_differences_of_the_hessian() {
    let (nodes, theta) = probe_subject();
    let n = nodes.len();
    let (marks, atoms) = (2, 2);
    let p = theta.len();
    const W: usize = 16;
    let u: Vec<f64> = (0..p).map(|i| 0.3 * ((i as f64) * 0.7).sin() + 0.1).collect();
    let v: Vec<f64> = (0..p).map(|i| 0.2 * ((i as f64) * 1.3).cos() - 0.05).collect();
    let plain = inputs(&nodes, &theta[..n * marks], &theta[n * marks..n * marks + marks * atoms], &theta[n * marks + marks * atoms..]);
    let mode = find_mode(&plain, None).expect("mode");
    let hessian_at = |theta: &[f64]| -> Vec<Vec<f64>> {
        let jets: Vec<Tangent<f64, W>> = (0..p)
            .map(|i| {
                let mut grad = [0.0; W];
                grad[i] = 1.0;
                Tangent::seeded(theta[i], grad)
            })
            .collect();
        let inp = inputs(&nodes, &jets[..n * marks], &jets[n * marks..n * marks + marks * atoms], &jets[n * marks + marks * atoms..]);
        let plain = inputs(&nodes, &theta[..n * marks], &theta[n * marks..n * marks + marks * atoms], &theta[n * marks + marks * atoms..]);
        let mode = find_mode(&plain, None).expect("mode");
        let out = evidence(&inp, &mode, true).expect("evidence");
        out.gradient.iter().map(|g| g.grad[..p].to_vec()).collect()
    };
    let one_jets: Vec<Tangent<gam_math::jet_scalar::OneSeed<0>, W>> = (0..p)
        .map(|i| {
            let zero = seeded_one(0.0, 0.0);
            let mut grad = [zero; W];
            grad[i] = seeded_one(1.0, 0.0);
            Tangent::seeded(seeded_one(theta[i], u[i]), grad)
        })
        .collect();
    let one = evidence(
        &inputs(&nodes, &one_jets[..n * marks], &one_jets[n * marks..n * marks + marks * atoms], &one_jets[n * marks + marks * atoms..]),
        &mode,
        true,
    )
    .expect("one-seed evidence");
    let h = 1e-4;
    let shifted = |s: f64, dir: &[f64]| -> Vec<f64> { theta.iter().zip(dir.iter()).map(|(x, d)| x + s * d).collect() };
    let plus = hessian_at(&shifted(h, &u));
    let minus = hessian_at(&shifted(-h, &u));
    for i in 0..p {
        for j in 0..p {
            let fd = (plus[i][j] - minus[i][j]) / (2.0 * h);
            let dual = one.gradient[i].grad[j].eps.value();
            assert!(
                (dual - fd).abs() < 2e-5 * (1.0 + fd.abs()),
                "D H[u] at ({i},{j}): dual {dual} vs finite difference {fd}"
            );
            let base = one.gradient[i].grad[j].base.value();
            assert!((base - hessian_at(&theta)[i][j]).abs() < 1e-10 * (1.0 + base.abs()));
        }
    }
    let two_jets: Vec<Tangent<gam_math::jet_scalar::TwoSeed<0>, W>> = (0..p)
        .map(|i| {
            let zero = seeded_two(0.0, 0.0, 0.0);
            let mut grad = [zero; W];
            grad[i] = seeded_two(1.0, 0.0, 0.0);
            Tangent::seeded(seeded_two(theta[i], u[i], v[i]), grad)
        })
        .collect();
    let two = evidence(
        &inputs(&nodes, &two_jets[..n * marks], &two_jets[n * marks..n * marks + marks * atoms], &two_jets[n * marks + marks * atoms..]),
        &mode,
        true,
    )
    .expect("two-seed evidence");
    let one_at = |theta: &[f64]| -> Vec<Vec<f64>> {
        let jets: Vec<Tangent<gam_math::jet_scalar::OneSeed<0>, W>> = (0..p)
            .map(|i| {
                let zero = seeded_one(0.0, 0.0);
                let mut grad = [zero; W];
                grad[i] = seeded_one(1.0, 0.0);
                Tangent::seeded(seeded_one(theta[i], u[i]), grad)
            })
            .collect();
        let plain = inputs(&nodes, &theta[..n * marks], &theta[n * marks..n * marks + marks * atoms], &theta[n * marks + marks * atoms..]);
        let mode = find_mode(&plain, None).expect("mode");
        let out = evidence(
            &inputs(&nodes, &jets[..n * marks], &jets[n * marks..n * marks + marks * atoms], &jets[n * marks + marks * atoms..]),
            &mode,
            true,
        )
        .expect("evidence");
        out.gradient.iter().map(|g| g.grad[..p].iter().map(|t| t.eps.value()).collect()).collect()
    };
    let plus_v = one_at(&shifted(h, &v));
    let minus_v = one_at(&shifted(-h, &v));
    for i in 0..p {
        for j in 0..p {
            let fd = (plus_v[i][j] - minus_v[i][j]) / (2.0 * h);
            let dual = two.gradient[i].grad[j].eps_del.value();
            assert!(
                (dual - fd).abs() < 5e-5 * (1.0 + fd.abs()),
                "D²H[u,v] at ({i},{j}): dual {dual} vs finite difference {fd}"
            );
        }
    }
}

#[test]
fn the_filter_on_one_node_is_the_evidence_of_one_node() {
    let nodes = subject(&[1.0], &[0.8], &[vec![2.0, 0.0]]);
    let eta0 = [0.3, -0.2];
    let loadings = [0.9, -0.4, 0.2, 0.5];
    let log_rates = [0.0, 0.4];
    let inputs = inputs(&nodes, &eta0, &loadings, &log_rates);
    let mode = find_mode(&inputs, None).expect("mode");
    let joint = evidence(&inputs, &mode, false).expect("evidence").loglik;
    let pass = filter_pass(&inputs, None, &[true, true]).expect("filter");
    assert!((pass.log_normalisers[0] - joint).abs() < 1e-10, "{} vs {joint}", pass.log_normalisers[0]);
    // Restricting the compensator to the first mark changes the normaliser
    // to that of a node without the second mark's exposure.
    let restricted = filter_pass(&inputs, None, &[true, false]).expect("filter");
    let mut without = subject(&[1.0], &[0.8], &[vec![2.0, 0.0]]);
    without.exposures[[0, 1]] = 0.0;
    let inputs_without = SubjectInputs {
        nodes: &without,
        eta0: &eta0,
        loadings: &loadings,
        log_rates: &log_rates,
        time_scale: 1.0,
    };
    let mode_without = find_mode(&inputs_without, None).expect("mode");
    let joint_without = evidence(&inputs_without, &mode_without, false).expect("evidence").loglik;
    assert!((restricted.log_normalisers[0] - joint_without).abs() < 1e-10);
}

#[test]
fn transition_at_an_overflowed_rate_is_finite_with_zero_sensitivity() {
    let t = laplace::transition(&seeded_one(800.0, 1.0), 0.7, 1.0);
    assert_eq!(t.phi.value(), 0.0);
    assert_eq!(t.q.value(), 1.0);
    for value in [&t.phi, &t.q, &t.inv_q, &t.kappa] {
        assert!(value.value().is_finite() && value.eps.value().is_finite());
        assert_eq!(value.eps.value(), 0.0);
    }
}

#[test]
fn node_expansion_integrates_exposure_places_events_and_applies_risk_sets() {
    let mut cohort = EventHistoryCohort {
        mark_names: vec!["flu".to_string(), "diabetes".to_string(), "death".to_string()],
        mark_kinds: vec![MarkKind::Recurrent, MarkKind::Once, MarkKind::Terminal],
        covariate_names: vec!["x".to_string()],
        covariate_levels: vec![Vec::new()],
        covariates: array![[1.0], [2.0]],
        subjects: vec![
            SubjectHistory {
                id: "s".to_string(),
                entry: 1.0,
                exit: 4.0,
                events: vec![
                    Event { time: 2.5, mark: 1 },
                    Event { time: 2.5, mark: 0 },
                    Event { time: 4.0, mark: 2 },
                ],
                segments: vec![
                    CovariateSegment { start: 0.0, row: 0 },
                    CovariateSegment { start: 3.0, row: 1 },
                ],
            },
            SubjectHistory {
                id: "prior".to_string(),
                entry: 1.0,
                exit: 3.0,
                events: vec![Event { time: 0.5, mark: 1 }],
                segments: vec![CovariateSegment { start: 0.0, row: 0 }],
            },
        ],
    };
    cohort.validate().expect("valid");
    let nodes = expand_nodes(&cohort, 5, 0).expect("nodes");
    let s = &nodes.subjects[0];
    let weight: f64 = s.weights.iter().sum();
    assert!((weight - 3.0).abs() < 1e-12, "weight {weight}");
    let event_node = s.times.iter().position(|&t| t == 2.5).expect("event node");
    assert_eq!(s.counts[[event_node, 0]], 1.0);
    assert_eq!(s.counts[[event_node, 1]], 1.0);
    // The closed rule makes the event time a node of the compensator too, so
    // the instant the intensity is read carries its own exposure.
    assert!(s.weights[event_node] > 0.0, "the event node carries no weight");
    assert!(s.exposures[[event_node, 0]] > 0.0, "a recurrent mark is exposed at the event");
    assert!(
        s.exposures[[event_node, 1]] > 0.0 && s.exposures[[event_node, 1]] < s.weights[event_node],
        "a once-only mark keeps only the exposure of the cell that ended at its onset: {} of {}",
        s.exposures[[event_node, 1]],
        s.weights[event_node]
    );
    let death_node = s.times.iter().position(|&t| t == 4.0).expect("death node");
    assert_eq!(s.counts[[death_node, 2]], 1.0);
    assert!(s.gaps.iter().all(|&g| g > 0.0));
    let flu: f64 = (0..s.len()).map(|n| s.exposures[[n, 0]]).sum();
    let diabetes: f64 = (0..s.len()).map(|n| s.exposures[[n, 1]]).sum();
    let death: f64 = (0..s.len()).map(|n| s.exposures[[n, 2]]).sum();
    assert!((flu - 3.0).abs() < 1e-12 && (death - 3.0).abs() < 1e-12);
    assert!((diabetes - 1.5).abs() < 1e-12, "diabetes exposure stops at its onset: {diabetes}");
    for (n, &t) in s.times.iter().enumerate() {
        let expected_row = if t >= 3.0 { 1 } else { 0 };
        assert_eq!(s.covariate_rows[n], expected_row);
        assert_eq!(nodes.node_data[[n, 0]], cohort.covariates[[expected_row, 0]]);
        assert_eq!(nodes.node_data[[n, 1]], t);
        if t > 2.5 {
            assert_eq!(s.exposures[[n, 1]], 0.0);
        }
    }
    // Every node that carries a count also carries exposure for that mark:
    // the discretisation never reads an intensity at an instant its own
    // compensator steps over.
    for subject_nodes in &nodes.subjects {
        for n in 0..subject_nodes.len() {
            for d in 0..3 {
                if subject_nodes.counts[[n, d]] > 0.0 {
                    assert!(
                        subject_nodes.exposures[[n, d]] > 0.0,
                        "node {n} counts mark {d} with no exposure"
                    );
                }
            }
        }
    }
    let prior = &nodes.subjects[1];
    assert!((0..prior.len()).all(|n| prior.exposures[[n, 1]] == 0.0), "a prior diagnosis removes the risk");
    assert!(cohort.subjects[1].events.iter().any(|e| e.mark == 1 && e.time <= cohort.subjects[1].entry));
    // Invalid histories are refused.
    let mut twice = cohort.clone();
    twice.subjects[0].events.push(Event { time: 3.5, mark: 1 });
    assert!(twice.validate().is_err(), "a once-only mark twice must fail");
    let mut early_death = cohort.clone();
    early_death.subjects[0].events.retain(|e| e.mark != 2);
    early_death.subjects[0].events.push(Event { time: 3.0, mark: 2 });
    assert!(early_death.validate().is_err(), "a terminal event before exit must fail");
}

/// A xorshift generator for the simulations.
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
            segments: vec![CovariateSegment { start: 0.0, row: s }],
        });
    }
    EventHistoryCohort {
        mark_names: vec!["event".to_string()],
        mark_kinds: vec![MarkKind::Recurrent],
        covariate_names: vec!["x".to_string()],
        covariate_levels: vec![Vec::new()],
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
        2,
        cohort.time_scale(),
    )
    .expect("family");
    let beta = array![-0.4, 0.3];
    let latent = array![0.5, -0.3, -0.2, 0.4];
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
    let value_only = family.log_likelihood(&states(&beta, &latent)).expect("value");
    assert!((base.log_likelihood - value_only).abs() < 1e-10 * (1.0 + value_only.abs()));
    let p = beta.len() + latent.len();
    let h = 1e-5;
    let split = |v: &Array1<f64>| -> (Array1<f64>, Array1<f64>) {
        (v.slice(ndarray::s![0..2]).to_owned(), v.slice(ndarray::s![2..]).to_owned())
    };
    let mut full = Array1::<f64>::zeros(p);
    full.slice_mut(ndarray::s![0..2]).assign(&beta);
    full.slice_mut(ndarray::s![2..]).assign(&latent);
    for i in 0..p {
        let mut plus = full.clone();
        plus[i] += h;
        let mut minus = full.clone();
        minus[i] -= h;
        let (bp, lp) = split(&plus);
        let (bm, lm) = split(&minus);
        let jp = family.joint_evaluation(&states(&bp, &lp)).expect("joint");
        let jm = family.joint_evaluation(&states(&bm, &lm)).expect("joint");
        let fd = (jp.log_likelihood - jm.log_likelihood) / (2.0 * h);
        assert!(
            (base.gradient[i] - fd).abs() < 1e-6 * (1.0 + fd.abs()),
            "gradient[{i}] {} vs {fd}",
            base.gradient[i]
        );
        for j in 0..p {
            let fd_h = -(jp.gradient[j] - jm.gradient[j]) / (2.0 * h);
            assert!(
                (base.hessian[[i, j]] - fd_h).abs() < 1e-5 * (1.0 + fd_h.abs()),
                "hessian[{i},{j}] {} vs {fd_h}",
                base.hessian[[i, j]]
            );
        }
    }
    let u = array![0.2, -0.1, 0.3, 0.15, -0.2, 0.1];
    let du = family.directional_hessian(&states(&beta, &latent), &u).expect("directional");
    let (up, lp) = split(&(&full + &u.mapv(|x| x * h)));
    let (um, lm) = split(&(&full - &u.mapv(|x| x * h)));
    let plus = family.joint_evaluation(&states(&up, &lp)).expect("joint");
    let minus = family.joint_evaluation(&states(&um, &lm)).expect("joint");
    for i in 0..p {
        for j in 0..p {
            let fd = (plus.hessian[[i, j]] - minus.hessian[[i, j]]) / (2.0 * h);
            assert!(
                (du[[i, j]] - fd).abs() < 1e-5 * (1.0 + fd.abs()),
                "D H[u][{i},{j}] {} vs {fd}",
                du[[i, j]]
            );
        }
    }
    let v = array![-0.1, 0.25, 0.05, -0.3, 0.2, 0.1];
    let d2 = family.second_directional_hessian(&states(&beta, &latent), &u, &v).expect("second");
    let (vp, lvp) = split(&(&full + &v.mapv(|x| x * h)));
    let (vm, lvm) = split(&(&full - &v.mapv(|x| x * h)));
    let plus = family.directional_hessian(&states(&vp, &lvp), &u).expect("directional");
    let minus = family.directional_hessian(&states(&vm, &lvm), &u).expect("directional");
    for i in 0..p {
        for j in 0..p {
            let fd = (plus[[i, j]] - minus[[i, j]]) / (2.0 * h);
            assert!(
                (d2[[i, j]] - fd).abs() < 5e-5 * (1.0 + fd.abs()),
                "D²H[u,v][{i},{j}] {} vs {fd}",
                d2[[i, j]]
            );
        }
    }
}

#[test]
fn covariance_score_matches_its_definition_and_its_rate_derivatives() {
    let residuals = vec![
        SubjectResiduals {
            times: vec![0.0, 0.7, 1.9, 2.4],
            scores: vec![0.3, -0.2, -0.5, 0.4, 1.0, -0.1, 0.2, 0.6],
            curvatures: vec![0.2, 0.3, 0.5, 0.1, 0.4, 0.4, 0.2, 0.3],
        },
        SubjectResiduals {
            times: vec![0.2, 1.1],
            scores: vec![-0.4, 0.1, 0.8, -0.3],
            curvatures: vec![0.3, 0.2, 0.6, 0.5],
        },
    ];
    let marks = 2;
    let (rho, time_scale) = (0.3, 2.0);
    let brute = |rho: f64| -> Array2<f64> {
        let rate = rho.exp() / time_scale;
        let mut m = Array2::<f64>::zeros((marks, marks));
        for subject in &residuals {
            let n = subject.times.len();
            for a in 0..n {
                for b in 0..n {
                    let kernel = (-rate * (subject.times[a] - subject.times[b]).abs()).exp();
                    for d in 0..marks {
                        for e in 0..marks {
                            m[[d, e]] += kernel * subject.scores[a * marks + d] * subject.scores[b * marks + e];
                        }
                    }
                }
                for d in 0..marks {
                    m[[d, d]] -= subject.curvatures[a * marks + d];
                }
            }
        }
        m
    };
    let [m0, m1, m2] = covariance_score(&residuals, marks, rho, time_scale);
    let expected = brute(rho);
    for d in 0..marks {
        for e in 0..marks {
            assert!((m0[[d, e]] - expected[[d, e]]).abs() < 1e-12, "M[{d},{e}] {} vs {}", m0[[d, e]], expected[[d, e]]);
        }
    }
    let h = 1e-5;
    let fd1 = (brute(rho + h) - brute(rho - h)) / (2.0 * h);
    let fd2 = (brute(rho + h) - 2.0 * brute(rho) + brute(rho - h)) / (h * h);
    for d in 0..marks {
        for e in 0..marks {
            assert!((m1[[d, e]] - fd1[[d, e]]).abs() < 1e-8 * (1.0 + fd1[[d, e]].abs()));
            assert!((m2[[d, e]] - fd2[[d, e]]).abs() < 1e-5 * (1.0 + fd2[[d, e]].abs()));
        }
    }
    // A clearly correlated residual sequence proposes an atom; white
    // residuals whose curvature exceeds their square do not.
    let correlated = vec![SubjectResiduals {
        times: (0..40).map(|i| i as f64 * 0.1).collect(),
        scores: (0..40).flat_map(|i| [if i < 20 { 0.5 } else { -0.5 }, 0.0]).collect(),
        curvatures: vec![0.01; 80],
    }];
    let atom = best_new_atom(&correlated, marks, 1.0).expect("score").expect("an atom");
    assert!(atom.eigenvalue > 0.0 && atom.loading[0].abs() > 0.0 && atom.loading[1].abs() < 1e-9, "{atom:?}");
    assert!(atom.variance > 0.0 && atom.log_rate.is_finite());
    let white = vec![SubjectResiduals {
        times: (0..40).map(|i| i as f64 * 0.1).collect(),
        scores: (0..40).flat_map(|i| [if i % 2 == 0 { 0.1 } else { -0.1 }, 0.0]).collect(),
        curvatures: vec![1.0; 80],
    }];
    assert!(best_new_atom(&white, marks, 1.0).expect("score").is_none());
}

#[test]
fn reported_covariance_objects_are_consistent() {
    let loadings = array![[0.8, 0.1], [-0.3, 0.5], [0.2, -0.4]];
    let rates = [0.5, 2.0];
    let c0 = super::covariance::disease_covariance(&loadings);
    let at_zero = super::covariance::temporal_covariance(&loadings, &rates, 0.0);
    let at_lag = super::covariance::temporal_covariance(&loadings, &rates, 1.5);
    for d in 0..3 {
        for e in 0..3 {
            assert!((c0[[d, e]] - at_zero[[d, e]]).abs() < 1e-14);
            let expected: f64 = (0..2)
                .map(|k| loadings[[d, k]] * loadings[[e, k]] * (-rates[k] * 1.5).exp())
                .sum();
            assert!((at_lag[[d, e]] - expected).abs() < 1e-14);
        }
    }
    let (values, vectors) = super::covariance::eigenmodes(&c0).expect("eigen");
    assert!(values[0] >= values[1] && values[1] >= values[2]);
    assert!(values[2].abs() < 1e-12, "rank two: {values:?}");
    let rebuilt = vectors.dot(&Array2::from_diag(&values)).dot(&vectors.t());
    for d in 0..3 {
        for e in 0..3 {
            assert!((rebuilt[[d, e]] - c0[[d, e]]).abs() < 1e-12);
        }
    }
}

/// Write one line to stdout from test support code.
fn emit(line: &str) {
    use std::io::Write;
    let mut stdout = std::io::stdout().lock();
    if stdout.write_all(line.as_bytes()).is_ok() && stdout.write_all(b"\n").is_ok() {
        return;
    }
}

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
    let rows = super::cohort::design_rows(&cohort, 9).expect("design rows");
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
fn fit_recovers_the_covariate_effect_and_grows_one_shared_risk_direction() {
    install_test_logger();
    let started = std::time::Instant::now();
    let mut cohort = simulate_cohort(80, 6.0, -0.8, 0.5, 1.0, 0.4, 7);
    let spec = EventHistorySpec::new(vec![linear_spec()]);
    let fit = fit_event_history(&mut cohort, &spec).expect("fit");
    emit(&format!(
        "[fit] {:.1}s rank={} loadings={:?} rates={:?} evidence={:?} path={:?}",
        started.elapsed().as_secs_f64(),
        fit.rank(),
        fit.loadings,
        fit.rates,
        fit.atom_evidence,
        fit.rank_path
    ));
    let beta = fit.mark_coefficients(0);
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
    assert!(fit.rank() >= 1, "a shared dynamic risk was simulated but the evidence grew no atom");
    let loading = fit.loadings[[0, 0]].abs();
    assert!(loading > 0.4, "fitted loading {loading}");
    assert!(fit.rates[0] > 0.0 && fit.rates[0].is_finite());
    assert!(fit.atom_evidence[0] > 0.0);
    assert!(fit.fit.outer_gradient_norm.is_none_or(|g| g.is_finite()));
    let covariance = fit.disease_covariance();
    assert!((covariance[[0, 0]] - fit.loadings.row(0).dot(&fit.loadings.row(0))).abs() < 1e-12);
    // Forecast: probabilities and expected counts are coherent.
    let horizons = [6.5, 7.0, 8.0];
    let future = vec![FutureSegment { start: 0.0, covariates: cohort.covariates.row(0).to_vec() }];
    let f = forecast(
        &fit,
        &cohort,
        &ForecastRequest { history: &cohort.subjects[0], horizons: &horizons, future: &future },
    )
    .expect("forecast");
    assert!(f.survival.iter().all(|&s| (s - 1.0).abs() < 1e-12), "no terminal marks: survival stays one");
    for h in 0..3 {
        assert!((0.0..=1.0).contains(&f.risk[[h, 0]]), "risk {}", f.risk[[h, 0]]);
        assert!(f.expected_counts[[h, 0]] >= f.risk[[h, 0]] - 1e-12);
    }
    assert!(f.risk[[0, 0]] <= f.risk[[1, 0]] && f.risk[[1, 0]] <= f.risk[[2, 0]]);
    assert!(f.expected_counts[[2, 0]] > 0.0);
    // The smoothed latent state and its follow-up average are coherent.
    let path = latent_state(&fit, &cohort, &cohort.subjects[0]).expect("latent state");
    let (mean, cov) = latent_exposure(&fit, &cohort, &cohort.subjects[0]).expect("exposure");
    assert_eq!(path.means.ncols(), fit.rank());
    assert_eq!(mean.len(), fit.rank());
    let atoms = fit.rank();
    let max_variance = (0..path.times.len())
        .map(|n| path.covariances[n * atoms * atoms])
        .fold(0.0, f64::max);
    assert!(cov[0] > 0.0 && cov[0] <= max_variance + 1e-9, "exposure variance {} vs max node variance {max_variance}", cov[0]);
    // Predictive PIT: uniform up to sampling error on the training cohort.
    let mut pits = Vec::new();
    for subject in &cohort.subjects {
        pits.extend(predictive_pit(&fit, &cohort, subject).expect("pit").into_iter().map(|e| e.pit));
    }
    assert!(pits.iter().all(|&u| (0.0..=1.0).contains(&u)));
    let ks = kolmogorov_smirnov_uniform(&pits).expect("events to test");
    let n = pits.len() as f64;
    emit(&format!("[fit] pit events={} ks={ks}", pits.len()));
    assert!(
        ks < 1.63 / n.sqrt() + 0.05,
        "PIT KS distance {ks} over {n} events exceeds the uniform band"
    );
}

#[test]
fn a_cohort_without_shared_risk_carries_no_meaningful_atom() {
    install_test_logger();
    let mut cohort = simulate_cohort(80, 6.0, -0.8, 0.5, 0.0, 0.4, 3);
    let spec = EventHistorySpec::new(vec![linear_spec()]);
    let fit = fit_event_history(&mut cohort, &spec).expect("fit");
    emit(&format!(
        "[null] rank={} loadings={:?} evidence={:?} path={:?}",
        fit.rank(),
        fit.loadings,
        fit.atom_evidence,
        fit.rank_path
    ));
    assert!(fit.fit.outer_gradient_norm.is_none_or(|g| g.is_finite()));
    for a in fit.loadings.iter() {
        assert!(a.abs() < 0.35, "no shared risk was simulated but a fitted loading is {a}");
    }
    assert!(fit.rank_path.iter().all(|step| step.evidence_gain.is_finite()));
}

#[test]
fn first_onset_and_terminal_marks_forecast_per_mark_risks() {
    install_test_logger();
    // Two once-only diseases and death, 60 subjects, no covariate effect.
    let mut rng = Rng(17);
    let rates = [0.15, 0.25, 0.08];
    let mut subjects = Vec::new();
    let mut covariates = Array2::<f64>::zeros((60, 1));
    for s in 0..60 {
        covariates[[s, 0]] = rng.uniform() - 0.5;
        let mut events = Vec::new();
        let mut exit = 8.0;
        for (mark, &rate) in rates.iter().enumerate() {
            let t = -rng.uniform().max(1e-12).ln() / rate;
            if t < 8.0 {
                events.push(Event { time: t, mark });
            }
        }
        if let Some(death) = events.iter().find(|e| e.mark == 2).map(|e| e.time) {
            exit = death;
            events.retain(|e| e.time <= death);
        }
        if s % 10 == 0 {
            events.push(Event { time: -1.0, mark: 0 });
            events.retain(|e| e.mark != 0 || e.time <= 0.0);
        }
        subjects.push(SubjectHistory {
            id: format!("p{s}"),
            entry: 0.0,
            exit,
            events,
            segments: vec![CovariateSegment { start: 0.0, row: s }],
        });
    }
    let mut cohort = EventHistoryCohort {
        mark_names: vec!["a".to_string(), "b".to_string(), "death".to_string()],
        mark_kinds: vec![MarkKind::Once, MarkKind::Once, MarkKind::Terminal],
        covariate_names: vec!["x".to_string()],
        covariate_levels: vec![Vec::new()],
        covariates,
        subjects,
    };
    let spec = EventHistorySpec::new(vec![linear_spec()]);
    let fit = fit_event_history(&mut cohort, &spec).expect("fit");
    emit(&format!("[risk] rank={} path={:?}", fit.rank(), fit.rank_path));
    let alive = cohort
        .subjects
        .iter()
        .find(|s| s.exit == 8.0 && s.events.iter().all(|e| e.time > 0.0) && !s.events.iter().any(|e| e.mark == 0))
        .expect("an alive subject without disease a");
    let future = vec![FutureSegment { start: 0.0, covariates: cohort.covariates.row(0).to_vec() }];
    let f = forecast(
        &fit,
        &cohort,
        &ForecastRequest { history: alive, horizons: &[9.0, 10.0, 12.0], future: &future },
    )
    .expect("forecast");
    for h in 0..3 {
        assert!((0.0..=1.0).contains(&f.survival[h]));
        assert!(f.risk[[h, 0]].is_finite() && (0.0..=1.0).contains(&f.risk[[h, 0]]));
        assert!(f.risk[[h, 2]].is_finite() && (f.risk[[h, 2]] - (1.0 - f.survival[h])).abs() < 1e-9, "death risk is one minus survival");
        assert!((f.expected_counts[[h, 0]] - f.risk[[h, 0]]).abs() < 1e-12, "a once-only mark's expected count is its risk");
        if h > 0 {
            assert!(f.survival[h] <= f.survival[h - 1] + 1e-12);
            assert!(f.risk[[h, 0]] >= f.risk[[h - 1, 0]] - 1e-12);
        }
    }
    let had_b = alive.events.iter().any(|e| e.mark == 1);
    assert_eq!(f.risk[[0, 1]].is_nan(), had_b, "a disease already present carries no risk");
    let prior = cohort.subjects.iter().find(|s| s.events.iter().any(|e| e.time <= 0.0) && s.exit == 8.0).expect("prior");
    let g = forecast(
        &fit,
        &cohort,
        &ForecastRequest { history: prior, horizons: &[9.0], future: &future },
    )
    .expect("forecast");
    assert!(g.risk[[0, 0]].is_nan());
    let dead = cohort.subjects.iter().find(|s| s.exit < 8.0).expect("a dead subject");
    assert!(
        forecast(
            &fit,
            &cohort,
            &ForecastRequest { history: dead, horizons: &[9.0], future: &future },
        )
        .is_err(),
        "nothing to forecast after death"
    );
    let pits: Vec<f64> = cohort
        .subjects
        .iter()
        .flat_map(|s| predictive_pit(&fit, &cohort, s).expect("pit").into_iter().map(|e| e.pit))
        .collect();
    assert!(pits.iter().all(|&u| (0.0..=1.0).contains(&u)));
    let ks = kolmogorov_smirnov_uniform(&pits).expect("events to test");
    emit(&format!("[risk] pit events={} ks={ks}", pits.len()));
    assert!(ks < 1.63 / (pits.len() as f64).sqrt() + 0.05, "PIT KS {ks}");
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
/// selected by REML. A declining effect is recovered as a decline; a score
/// carrying nothing collapses to zero.
#[test]
fn an_observed_score_enters_as_a_penalised_slope_surface() {
    install_test_logger();
    let times = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5];
    let formula = "s(time, by=g)";
    let truth = |t: f64| 1.0 - 0.15 * t;
    let mut cohort = simulate_score_cohort(300, 6.0, -0.5, &truth, 1.0, 19);
    let events: usize = cohort.subjects.iter().map(|s| s.events.len()).sum();
    let started = std::time::Instant::now();
    let fit = fit_event_history_formula(&mut cohort, formula, BlockwiseFitOptions::default())
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
    let null_fit = fit_event_history_formula(&mut null, formula, BlockwiseFitOptions::default())
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
/// the history tier continues from the subject's smoothed state. A positive
/// score effect orders the first two by the score's sign, and a history
/// richer (poorer) in events than its score alone predicts raises (lowers)
/// the third against the second. No weight between the tiers is chosen.
#[test]
fn forecast_tiers_population_score_and_history_are_one_model_conditioned_on_more() {
    install_test_logger();
    let mut cohort = simulate_cohort(60, 6.0, -0.8, 0.5, 1.0, 0.4, 5);
    let spec = EventHistorySpec::new(vec![linear_spec()]);
    let started = std::time::Instant::now();
    let fit = fit_event_history(&mut cohort, &spec).expect("fit");
    let beta = fit.mark_coefficients(0);
    emit(&format!(
        "[tiers] {:.1}s rank={} beta={:?} loadings={:?} rates={:?}",
        started.elapsed().as_secs_f64(),
        fit.rank(),
        beta.to_vec(),
        fit.loadings,
        fit.rates
    ));
    assert!(beta[1] > 0.0, "the score effect was simulated positive; fitted {}", beta[1]);
    assert!(fit.rank() >= 1, "a latent atom was simulated");
    let horizons = [7.0, 8.0];
    let population = population_forecast(
        &fit,
        &cohort,
        &PopulationForecastRequest {
            start: 6.0,
            horizons: &horizons,
            future: &[FutureSegment { start: 6.0, covariates: vec![0.0] }],
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
            let future = vec![FutureSegment { start: subject.exit, covariates: vec![score] }];
            let alone = population_forecast(
                &fit,
                &cohort,
                &PopulationForecastRequest { start: subject.exit, horizons: &horizons, future: &future },
            )
            .expect("score-only forecast");
            let with_history = forecast(
                &fit,
                &cohort,
                &ForecastRequest { history: subject, horizons: &horizons, future: &future },
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
    assert!(
        richest.2 > richest.1,
        "a history rich in events must raise the forecast above its score-only tier"
    );
    assert!(
        poorest.2 < poorest.1,
        "a history poor in events must lower the forecast below its score-only tier"
    );
}

/// Simulate first diagnoses of three diseases plus death driven by a
/// rank-two latent covariance (the figure cohort), by thinning on a fine
/// Euler path.
fn simulate_four_marks(subjects: usize, seed: u64) -> EventHistoryCohort {
    let mut rng = Rng(seed);
    let follow_up: f64 = 10.0;
    let dt: f64 = 0.01;
    let base: [f64; 4] = [-2.6, -2.9, -2.4, -3.6];
    let slope: f64 = 0.3;
    let rates: [f64; 2] = [0.3, 1.5];
    let loadings: [[f64; 2]; 4] = [[0.9, 0.0], [0.7, 0.5], [0.0, 0.8], [0.3, 0.0]];
    let mut covariates = Array2::<f64>::zeros((subjects, 1));
    let mut histories = Vec::with_capacity(subjects);
    for s in 0..subjects {
        let x = rng.normal();
        covariates[[s, 0]] = x;
        let mut z = [rng.normal(), rng.normal()];
        let mut events = Vec::new();
        let mut had = [false; 4];
        if s % 10 == 0 {
            events.push(Event { time: -1.0, mark: 0 });
            had[0] = true;
        }
        let mut t = 0.0;
        let mut exit = follow_up;
        'time: while t < follow_up {
            for k in 0..2 {
                let phi = (-rates[k] * dt).exp();
                z[k] = phi * z[k] + (1.0 - phi * phi).sqrt() * rng.normal();
            }
            t += dt;
            for d in 0..4 {
                if had[d] {
                    continue;
                }
                let shift = -0.5 * (loadings[d][0] * loadings[d][0] + loadings[d][1] * loadings[d][1]);
                let eta = base[d] + slope * x + shift + loadings[d][0] * z[0] + loadings[d][1] * z[1];
                if rng.uniform() < eta.exp() * dt {
                    events.push(Event { time: t, mark: d });
                    had[d] = true;
                    if d == 3 {
                        exit = t;
                        break 'time;
                    }
                }
            }
        }
        histories.push(SubjectHistory {
            id: format!("p{s}"),
            entry: 0.0,
            exit,
            events,
            segments: vec![CovariateSegment { start: 0.0, row: s }],
        });
    }
    EventHistoryCohort {
        mark_names: ["a", "b", "c", "death"].iter().map(|s| s.to_string()).collect(),
        mark_kinds: vec![MarkKind::Once, MarkKind::Once, MarkKind::Once, MarkKind::Terminal],
        covariate_names: vec!["x".to_string()],
        covariate_levels: vec![Vec::new()],
        covariates,
        subjects: histories,
    }
}

/// A cohort of three first diagnoses and death, driven by a rank-two latent
/// covariance: the fit must recover a covariance whose leading structure is
/// the simulated one, at a rate the node mesh resolves. This is the shape
/// the design is for — several marks, per-mark risk sets, a shared latent
/// state — and it is where a rate proposal that ran past the mesh's
/// resolution used to send the loadings to infinity. The rate scan is
/// printed before the fit, so a refusal is readable as the shape of the
/// covariance score rather than as a bare error.
#[test]
fn four_marks_recover_a_shared_latent_covariance() {
    install_test_logger();
    use super::family::mark_block_spec;
    let started = std::time::Instant::now();
    let mut cohort = simulate_four_marks(150, 11);
    cohort.validate().expect("valid");
    let marks = 4;
    let events: Vec<usize> = (0..marks)
        .map(|d| cohort.subjects.iter().flat_map(|s| s.events.iter()).filter(|e| e.mark == d && e.time > 0.0).count())
        .collect();
    emit(&format!("[four] subjects={} events={events:?}", cohort.subjects.len()));
    // The covariance score across the rates the mesh can represent, at the
    // rank-zero fit.
    let nodes = Arc::new(expand_nodes(&cohort, 9, 0).expect("nodes"));
    let design = build_term_collection_design(nodes.node_data.view(), &linear_spec()).expect("design");
    let dense = design.design.try_to_dense_arc("d").expect("dense");
    let specs: Vec<_> = (0..marks).map(|d| mark_block_spec(&cohort.mark_names[d], &design)).collect();
    let family0 = EventHistoryFamily::new(Arc::clone(&nodes), vec![Arc::clone(&dense); marks], 0, cohort.time_scale())
        .expect("family");
    let fit0 = crate::custom_family::fit_custom_family(&family0, &specs, &BlockwiseFitOptions::default())
        .expect("rank-0 fit");
    let residuals = family0.residuals(&fit0.block_states).expect("residuals");
    let (lower, upper) = super::covariance::resolvable_log_rates(&residuals, cohort.time_scale())
        .expect("limits")
        .expect("a mesh with gaps");
    emit(&format!("[four] rank 0 criterion {:?}; resolvable log-rates [{lower:.3}, {upper:.3}]", fit0.reml_score()));
    for step in 0..=8 {
        let rho = lower + (upper - lower) * step as f64 / 8.0;
        let [m0, _, _] = super::covariance::covariance_score(&residuals, marks, rho, cohort.time_scale());
        let (values, vectors) = super::covariance::eigenmodes(&m0).expect("eigen");
        emit(&format!(
            "[four] scan ρ={rho:+.2} rate={:.3e} top={:+.4e} eigenvalues={:?} mode={:?}",
            rho.exp() / cohort.time_scale(),
            values[0],
            values.iter().map(|v| format!("{v:+.3e}")).collect::<Vec<_>>(),
            vectors.column(0).iter().map(|v| format!("{v:+.3}")).collect::<Vec<_>>()
        ));
    }
    match best_new_atom(&residuals, marks, cohort.time_scale()).expect("score") {
        Some(atom) => emit(&format!(
            "[four] proposal ρ={:.3} (limit {upper:.3}, at_limit={}) loading={:?} eigenvalue={:.4e} variance={:.4e}",
            atom.log_rate,
            atom.at_resolution_limit,
            atom.loading.iter().map(|v| format!("{v:+.3}")).collect::<Vec<_>>(),
            atom.eigenvalue,
            atom.variance
        )),
        None => emit("[four] proposal: none"),
    }
    let spec = EventHistorySpec::new(vec![linear_spec()]);
    let fit = fit_event_history(&mut cohort, &spec).expect("fit");
    let covariance = fit.disease_covariance();
    let (values, vectors) = fit.eigenmodes().expect("eigenmodes");
    emit(&format!(
        "[four] {:.1}s rank={} rates={:?} evidence={:?}",
        started.elapsed().as_secs_f64(),
        fit.rank(),
        fit.rates,
        fit.atom_evidence
    ));
    for step in &fit.rank_path {
        emit(&format!("[four] step {step:?}"));
    }
    emit(&format!("[four] C(0) diagonal {:?}", (0..marks).map(|d| covariance[[d, d]]).collect::<Vec<_>>()));
    emit(&format!("[four] C(0) full {:?}", covariance.rows().into_iter().map(|r| r.to_vec()).collect::<Vec<_>>()));
    emit(&format!("[four] eigenvalues {:?}", values.to_vec()));
    emit(&format!("[four] leading mode {:?}", vectors.column(0).to_vec()));
    emit(&format!("[four] betas {:?}", (0..marks).map(|d| fit.mark_coefficients(d).to_vec()).collect::<Vec<_>>()));
    assert!(fit.rank() >= 1, "a rank-two latent covariance was simulated but the evidence grew no atom");
    // The simulated marginal variances are 0.81, 0.74, 0.64, 0.09: no
    // mark's fitted variance may be the runaway the standardised gain and
    // the mesh's resolution limit exist to prevent.
    let diagonal: Vec<f64> = (0..marks).map(|d| covariance[[d, d]]).collect();
    for (d, v) in diagonal.iter().enumerate() {
        assert!(*v < 4.0, "mark {d} latent variance {v} is far beyond anything simulated");
    }
    assert!(
        diagonal[3] < diagonal[0].max(diagonal[2]),
        "death carries the smallest simulated loading; fitted diagonal {diagonal:?}"
    );
    for step in fit.rank_path.iter().filter(|s| s.accepted) {
        assert!(!step.at_resolution_limit, "an accepted atom sat on the mesh's resolution limit: {step:?}");
    }
    // The forecast surface is coherent for a subject still alive and free
    // of every disease.
    let alive = cohort
        .subjects
        .iter()
        .find(|s| s.exit >= 9.99 && s.events.is_empty())
        .expect("a subject with no events");
    let f = forecast(
        &fit,
        &cohort,
        &ForecastRequest {
            history: alive,
            horizons: &[11.0, 12.0],
            future: &[FutureSegment { start: 0.0, covariates: cohort.covariates.row(0).to_vec() }],
        },
    )
    .expect("forecast");
    for h in 0..2 {
        for d in 0..marks {
            assert!((0.0..=1.0).contains(&f.risk[[h, d]]), "risk {:?}", f.risk);
        }
        assert!((0.0..=1.0).contains(&f.survival[h]));
    }
    assert!(f.risk[[0, 0]] <= f.risk[[1, 0]] + 1e-12);
}

/// The rate a proposal may name is bounded by what the node mesh
/// resolves, because beyond it an event node's latent coordinate is held
/// only by the prior and the evidence grows without bound in the loading.
#[test]
fn the_rate_proposal_stays_inside_what_the_mesh_resolves() {
    // Residuals whose only structure is at lag zero (the same node, across
    // marks) drive the matched filter to the fastest rate there is.
    let times: Vec<f64> = (0..60).map(|i| i as f64 * 0.1).collect();
    let mut scores = Vec::with_capacity(120);
    let mut curvatures = Vec::with_capacity(120);
    for i in 0..60 {
        let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
        scores.push(0.6 * sign);
        scores.push(0.6 * sign);
        curvatures.push(0.05);
        curvatures.push(0.05);
    }
    let residuals = vec![SubjectResiduals { times: times.clone(), scores, curvatures }];
    let (lower, upper) = super::covariance::resolvable_log_rates(&residuals, 6.0)
        .expect("limits")
        .expect("a mesh with gaps");
    // κ = 1 at the median gap of 0.1 with T̄ = 6.
    assert!((upper - (6.0f64 / 0.1).ln()).abs() < 1e-12, "upper {upper}");
    assert!(lower < 0.0 && lower < upper);
    for step in 0..=6 {
        let rho = lower + (upper - lower) * step as f64 / 6.0;
        let [m0, _, _] = super::covariance::covariance_score(&residuals, 2, rho, 6.0);
        let (values, _) = super::covariance::eigenmodes(&m0).expect("eigen");
        emit(&format!("[mesh] alternating ρ={rho:+.2} top={:+.4e}", values[0]));
    }
    let atom = best_new_atom(&residuals, 2, 6.0).expect("score");
    emit(&format!("[mesh] alternating proposal {atom:?}"));
    if let Some(atom) = atom {
        assert!(atom.log_rate <= upper + 1e-12, "proposal {} ran past the mesh limit {upper}", atom.log_rate);
    }
    // A cohort whose residuals are correlated over a real time scale
    // proposes a rate well inside the limit and does not flag it.
    let smooth: Vec<f64> = (0..60).flat_map(|i| {
        let block = if i < 30 { 0.6 } else { -0.6 };
        [block, block]
    }).collect();
    let slow = vec![SubjectResiduals { times, scores: smooth, curvatures: vec![0.05; 120] }];
    let atom = best_new_atom(&slow, 2, 6.0).expect("score").expect("an atom");
    emit(&format!("[mesh] slow proposal {atom:?}"));
    assert!(!atom.at_resolution_limit, "a slowly varying residual must not need the limit: {atom:?}");
    assert!(atom.log_rate < upper - 1.0, "proposal {} against limit {upper}", atom.log_rate);
}
