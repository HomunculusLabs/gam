#![cfg(test)]
//! #2933 F39/F40 — the fitted response the reconstruction dispersion prices.
//!
//! F39: the effective degrees of freedom are `tr R` for the response
//! `R = ∂f̂/∂y` of the whole identified fit, gates and constrained coordinates
//! included. The oracle here measures `R` directly. It perturbs one target entry
//! by `±h`, drives the penalized stationarity `∇L(θ; y ± h·e) = 0` from the
//! fitted state to its roundoff plateau with exact-A Newton steps, and differences
//! every fitted scalar. The root is fixed by the assembled (tangent-projected)
//! gradient alone, so the oracle does not score the operator against itself. Each
//! re-solve declares the fitted state's collapse-prevention gates, so it
//! differentiates the same frozen-gate objective the operator describes.
//!
//! F40: the dispersion's scale equation uses the residual degrees of freedom
//! `‖I − R‖²_F` of that response, not `N − tr R`. A ridge smoother with a known
//! response is fitted repeatedly to noise around a zero mean and around a mean it
//! reproduces; the production dispersion must be unbiased and its 95% bands must
//! cover near nominal.
use super::*;
use crate::manifold::arrow_solver::SaeArrowVector;
use crate::manifold::construction::FittedResponseDivergence;
use gam_terms::latent::LatentManifold;
use ndarray::{Array1, Array2, ArrayView2};

/// Largest number of exact-A Newton steps one re-solve may take.
pub(super) const ROOT_POLISH_STEPS: usize = 40;

/// A root is admitted only once its gradient has reached the arithmetic floor
/// of an order-one objective on a few dozen scalar observations.
pub(super) const ROOT_GRADIENT_CEILING: f64 = 1.0e-10;

/// Central-difference step on a target entry. The targets are of order one, so
/// the truncation error `h²·f'''/6` sits far below the tolerance below.
const FD_STEP: f64 = 1.0e-4;

/// Relative agreement required between a priced quantity and its re-solved
/// value. The truncation error at `FD_STEP` and the roundoff of a root driven to
/// its plateau are both orders of magnitude smaller.
const RELATIVE_TOLERANCE: f64 = 1.0e-4;

fn gradient(system: &ArrowSchurSystem) -> SaeArrowVector {
    SaeArrowVector {
        t: system
            .rows
            .iter()
            .flat_map(|row| row.gt.iter().copied())
            .collect(),
        beta: system.gb.clone(),
    }
}

/// Drive `∇L(θ; target) = 0` by exact-A Newton steps until the gradient norm
/// stops decreasing or [`ROOT_POLISH_STEPS`] iterates are spent. Returns the
/// undamped evidence factorization at the last iterate and the gradient-norm
/// trajectory, whose last entry is that iterate's norm.
pub(super) fn polish_to_root(
    term: &mut SaeManifoldTerm,
    target: ArrayView2<'_, f64>,
    rho: &SaeManifoldRho,
) -> (ArrowFactorCache, Vec<f64>) {
    let options = term.evidence_factor_options();
    let mut trajectory: Vec<f64> = Vec::with_capacity(ROOT_POLISH_STEPS);
    loop {
        let system = term
            .assemble_arrow_schur(target, rho, None)
            .expect("the arrow system assembles at every re-solve iterate");
        let g = gradient(&system);
        let norm = (g.t.dot(&g.t) + g.beta.dot(&g.beta)).sqrt();
        let (_, _, cache) = solve_arrow_newton_step_with_options(&system, 0.0, 0.0, &options)
            .expect("the majorizer factors undamped at every re-solve iterate");
        let decreased = trajectory.last().is_none_or(|&previous| norm < previous);
        trajectory.push(norm);
        if !decreased || trajectory.len() == ROOT_POLISH_STEPS {
            return (cache, trajectory);
        }
        let step = term
            .solve_exact_stationarity(rho, target, &cache, &g)
            .expect("the exact stationarity pseudoinverse solves at every re-solve iterate");
        term.apply_newton_step((-&step.t).view(), (-&step.beta).view(), 1.0)
            .expect("the exact Newton step applies to the term state");
    }
}

/// The gradient norm a polish trajectory ended at.
pub(super) fn root_norm(trajectory: &[f64]) -> f64 {
    trajectory.last().copied().unwrap_or(f64::INFINITY)
}

/// `R = ∂f̂/∂y` over the `n·p` scalars (row-major), by central differences of fits
/// re-solved to their roots under the base term's declared gates.
pub(super) fn resolved_response(
    base: &SaeManifoldTerm,
    target: &Array2<f64>,
    rho: &SaeManifoldRho,
) -> Array2<f64> {
    let (n, p) = target.dim();
    let gates = base.collapse_prevention_gates();
    let mut response = Array2::<f64>::zeros((n * p, n * p));
    for row in 0..n {
        for col in 0..p {
            let mut fitted = Vec::with_capacity(2);
            for sign in [1.0_f64, -1.0] {
                let mut term = base.clone();
                term.declare_collapse_prevention_gates(&gates);
                let mut perturbed = target.clone();
                perturbed[[row, col]] += sign * FD_STEP;
                let (_, trajectory) = polish_to_root(&mut term, perturbed.view(), rho);
                let norm = root_norm(&trajectory);
                assert!(
                    norm <= ROOT_GRADIENT_CEILING,
                    "the re-solve at entry ({row}, {col}), side {sign} stalled at ‖g‖={norm:.3e}"
                );
                fitted.push(
                    term.try_fitted_for_rho(rho)
                        .expect("the re-solved fit reconstructs"),
                );
            }
            for (index, (plus, minus)) in fitted[0].iter().zip(fitted[1].iter()).enumerate() {
                response[[index, row * p + col]] = (plus - minus) / (2.0 * FD_STEP);
            }
        }
    }
    response
}

/// `(tr R, ‖I − R‖²_F)` of a response matrix.
pub(super) fn trace_and_residual_dof(response: &Array2<f64>) -> (f64, f64) {
    let trace = (0..response.nrows()).map(|i| response[[i, i]]).sum();
    let residual_dof = response
        .indexed_iter()
        .map(|((i, j), &value)| {
            let entry = if i == j { 1.0 - value } else { -value };
            entry * entry
        })
        .sum();
    (trace, residual_dof)
}

fn assert_agrees(label: &str, value: f64, resolved: f64) {
    let gap = (value - resolved).abs();
    assert!(
        gap <= RELATIVE_TOLERANCE * resolved.abs().max(1.0),
        "{label}: {value:.9e} against the re-solved response {resolved:.9e} (gap {gap:.3e})"
    );
}

/// What the production operator and the dispersion report at a fitted state.
struct PricedResponse {
    response: FittedResponseDivergence,
    /// `RSS/φ̂` on the raw output frame: the residual dof the scale equation used.
    priced_residual_dof: f64,
}

fn priced_response(
    term: &SaeManifoldTerm,
    target: &Array2<f64>,
    rho: &SaeManifoldRho,
    cache: &ArrowFactorCache,
) -> PricedResponse {
    let response = term
        .fitted_response_divergence(target.view(), rho, cache)
        .expect("the fitted state admits a fitted-response divergence");
    let loss = term.loss(target.view(), rho).expect("the fitted state has a loss");
    let residual = term
        .reconstruction_residual(target.view(), rho)
        .expect("the fitted state has a residual");
    let dispersion = term
        .reconstruction_dispersion(&loss, cache, rho, residual.view())
        .expect("the fitted state prices a dispersion");
    PricedResponse {
        response,
        priced_residual_dof: 2.0 * loss.data_fit / dispersion.raw_output_noise_variance,
    }
}

/// Require the exact spectral divergence, its residual dof, and the residual dof
/// the dispersion prices to match the re-solved response; return `tr R`.
pub(super) fn assert_prices_resolved_response(
    label: &str,
    term: &SaeManifoldTerm,
    target: &Array2<f64>,
    rho: &SaeManifoldRho,
    cache: &ArrowFactorCache,
) -> f64 {
    let (trace, residual_dof) = trace_and_residual_dof(&resolved_response(term, target, rho));
    let priced = priced_response(term, target, rho, cache);
    eprintln!(
        "[#2933 F39 {label}] N={} resolved tr R={trace:.9e} ‖I−R‖²={residual_dof:.9e}; divergence \
         {:.9e} ({:?}), residual dof {:.9e}, dispersion RSS/φ {:.9e}",
        target.len(),
        priced.response.divergence,
        priced.response.estimator,
        priced.response.likelihood_residual_dof,
        priced.priced_residual_dof
    );
    assert!(
        matches!(
            priced.response.estimator,
            FittedResponseDivergenceEstimator::ExactSpectral
        ),
        "{label}: a fixture this small is on the exact spectral route, got {:?}",
        priced.response.estimator
    );
    assert!(
        trace > 1.0 && residual_dof > 1.0,
        "{label}: the re-solved response (tr R {trace}, ‖I−R‖² {residual_dof}) must be material \
         and leave residual dof"
    );
    assert_agrees(&format!("{label} divergence"), priced.response.divergence, trace);
    assert_agrees(
        &format!("{label} residual dof"),
        priced.response.likelihood_residual_dof,
        residual_dof,
    );
    assert_agrees(
        &format!("{label} residual dof the dispersion prices"),
        priced.priced_residual_dof,
        residual_dof,
    );
    trace
}

/// A basis with fixed per-row values that do not depend on the coordinate: every
/// jet is identically zero, so the atom is a linear smoother in its decoder.
#[derive(Debug)]
struct FixedRowBasis {
    phi: Array2<f64>,
}

impl SaeBasisEvaluator for FixedRowBasis {
    fn evaluate(
        &self,
        coords: ArrayView2<'_, f64>,
    ) -> Result<(Array2<f64>, ndarray::Array3<f64>), String> {
        if coords.nrows() != self.phi.nrows() {
            return Err(format!(
                "FixedRowBasis: {} coordinate rows for {} basis rows",
                coords.nrows(),
                self.phi.nrows()
            ));
        }
        Ok((
            self.phi.clone(),
            ndarray::Array3::zeros((self.phi.nrows(), self.phi.ncols(), coords.ncols())),
        ))
    }

    fn second_jet_dyn(
        &self,
        coords: ArrayView2<'_, f64>,
    ) -> Option<Result<ndarray::Array4<f64>, String>> {
        Some(Ok(ndarray::Array4::zeros((
            coords.nrows(),
            self.phi.ncols(),
            coords.ncols(),
            coords.ncols(),
        ))))
    }

    fn third_jet_dyn(
        &self,
        coords: ArrayView2<'_, f64>,
    ) -> Result<SaeBasisThirdJetCapability, String> {
        if coords.nrows() != self.phi.nrows() {
            return Err(format!(
                "FixedRowBasis: {} coordinate rows for {} basis rows",
                coords.nrows(),
                self.phi.nrows()
            ));
        }
        Ok(SaeBasisThirdJetCapability::CertifiedZero)
    }
}

fn fixed_basis_atom(name: &str, phi: Array2<f64>, decoder: Array2<f64>) -> SaeManifoldAtom {
    let width = phi.ncols();
    let evaluator = Arc::new(FixedRowBasis { phi });
    let coords = Array2::<f64>::zeros((evaluator.phi.nrows(), 1));
    let (basis, jet) = evaluator
        .evaluate(coords.view())
        .expect("fixed basis evaluates");
    let penalty = Array2::<f64>::from_diag(&Array1::from_shape_fn(width, |basis| {
        if basis == 0 { 0.0 } else { 1.0 }
    }));
    SaeManifoldAtom::new_with_provided_function_gram(
        name.to_string(),
        SaeAtomBasisKind::Periodic,
        1,
        basis,
        jet,
        decoder,
        penalty,
    )
    .expect("fixed-basis atom: basis width, latent dimension and decoder shape agree")
    .with_basis_evaluator(evaluator)
}

/// Deterministic standard normal draws (splitmix64, Box–Muller).
struct NormalStream {
    state: u64,
    spare: Option<f64>,
}

impl NormalStream {
    fn next(&mut self) -> f64 {
        if let Some(value) = self.spare.take() {
            return value;
        }
        let mut uniform = || {
            ((gam_linalg::utils::splitmix64(&mut self.state) >> 11) as f64 + 0.5)
                / (1u64 << 53) as f64
        };
        let (u, v) = (uniform(), uniform());
        let radius = (-2.0 * u.ln()).sqrt();
        let angle = std::f64::consts::TAU * v;
        self.spare = Some(radius * angle.sin());
        radius * angle.cos()
    }
}

/// Largest number of majorizer descent steps a fixture root may take.
const DESCENT_STEPS: usize = 400;

/// Gradient norm at which descent hands the state to the exact-A polish, whose
/// undamped steps contract quadratically from there.
const POLISH_HANDOFF_GRADIENT: f64 = 1.0e-6;

/// #2933 F39 — two coordinate-free atoms routed by one free softmax gate, whose
/// fit is identified. The routing contrast is strong (the generating gate runs
/// from `σ(−11)` to `σ(11)` across the rows), and only the "wave" atom carries a
/// constant column, so the atoms cannot trade an intercept along the simplex. The
/// chart coordinates carry no data jets, so beyond the decoder border the only
/// fitted response is the gate logits'. The state starts at the generating
/// decoders and gate.
fn identified_gated_state(
    log_lambda_sparse: f64,
) -> (SaeManifoldTerm, Array2<f64>, SaeManifoldRho) {
    let n = 16usize;
    let p = 3usize;
    let centre = (n as f64 - 1.0) / 2.0;
    let theta = |row: usize| std::f64::consts::TAU * row as f64 / n as f64;
    let phi_wave = Array2::<f64>::from_shape_fn((n, 3), |(row, basis)| match basis {
        0 => 1.0,
        1 => theta(row).cos(),
        _ => theta(row).sin(),
    });
    let phi_ramp = Array2::<f64>::from_shape_fn((n, 3), |(row, basis)| {
        let x = (row as f64 - centre) / centre;
        [x, x * x - 0.4, x * x * x][basis]
    });
    let truth_wave = Array2::<f64>::from_shape_fn((3, p), |(basis, out)| {
        0.9 * ((2 * basis + out) as f64 * 0.8 + 0.3).sin()
    });
    let truth_ramp = Array2::<f64>::from_shape_fn((3, p), |(basis, out)| {
        0.8 * ((basis + 2 * out) as f64 * 1.1 + 0.6).cos()
    });
    let decoded_wave = phi_wave.dot(&truth_wave);
    let decoded_ramp = phi_ramp.dot(&truth_ramp);
    let logit = |row: usize| 1.5 * (row as f64 - centre);
    let target = Array2::<f64>::from_shape_fn((n, p), |(row, out)| {
        let gate = 1.0 / (1.0 + (-logit(row)).exp());
        gate * decoded_wave[[row, out]]
            + (1.0 - gate) * decoded_ramp[[row, out]]
            + 0.05 * (1.3 * row as f64 + 2.9 * out as f64).cos()
    });
    let atoms = vec![
        fixed_basis_atom("wave", phi_wave, truth_wave),
        fixed_basis_atom("ramp", phi_ramp, truth_ramp),
    ];
    let logits = Array2::<f64>::from_shape_fn((n, 2), |(row, atom)| {
        if atom == 0 { logit(row) } else { 0.0 }
    });
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        logits,
        vec![Array2::<f64>::zeros((n, 1)), Array2::<f64>::zeros((n, 1))],
        vec![
            LatentManifold::Circle { period: 1.0 },
            LatentManifold::Circle { period: 1.0 },
        ],
        AssignmentMode::softmax(1.0),
    )
    .expect("assignment: two logit columns and one coordinate block per atom");
    let term = SaeManifoldTerm::new(atoms, assignment).expect("the atoms match their blocks");
    let rho = SaeManifoldRho::new(
        log_lambda_sparse,
        -1.0,
        vec![Array1::<f64>::zeros(1), Array1::<f64>::zeros(1)],
    )
    .for_assignment(&term.assignment);
    (term, target, rho)
}

/// Reach and certify a fixture root under production's collapse-gate lifecycle.
///
/// While the term's gates are not declared, every assembly refreshes them from
/// its iterate, as the inner fit does before the outer objective adopts a root's
/// gates, and each step is priced under the gates its assembly read. Descent is
/// by majorizer arrow Newton steps halved until the objective decreases: the
/// observed information can be indefinite away from the root, so undamped exact-A
/// steps do not globalize. At the handoff the state's gates are adopted and held,
/// as the outer objective holds the gates of its first root, and exact-A Newton
/// steps polish that root to its arithmetic floor. A term whose gates are already
/// declared descends and polishes under them. The certificate is printed.
fn certify_production_root(
    label: &str,
    term: &mut SaeManifoldTerm,
    target: &Array2<f64>,
    rho: &SaeManifoldRho,
) -> ArrowFactorCache {
    let refreshing = !term.streaming_gates_frozen;
    let options = term.evidence_factor_options();
    let mut descent_steps = 0usize;
    let mut handoff_gradient = f64::INFINITY;
    while descent_steps < DESCENT_STEPS {
        let system = term
            .assemble_arrow_schur(target.view(), rho, None)
            .unwrap_or_else(|error| panic!("{label}: the descent iterate assembles: {error}"));
        let gates = term.collapse_prevention_gates();
        let g = gradient(&system);
        handoff_gradient = (g.t.dot(&g.t) + g.beta.dot(&g.beta)).sqrt();
        if handoff_gradient <= POLISH_HANDOFF_GRADIENT {
            break;
        }
        let (delta_t, delta_beta, _) =
            solve_arrow_newton_step_with_options(&system, 0.0, 0.0, &options).unwrap_or_else(
                |error| panic!("{label}: the majorizer factors at a descent iterate: {error}"),
            );
        let slope = g.t.dot(&delta_t) + g.beta.dot(&delta_beta);
        let start = term
            .penalized_objective_total(target.view(), rho, None, 1.0)
            .unwrap_or_else(|error| panic!("{label}: the descent iterate has a value: {error}"));
        let mut step = if slope < 0.0 { 1.0 } else { -1.0 };
        let accepted = loop {
            let mut trial = term.clone();
            trial.declare_collapse_prevention_gates(&gates);
            trial
                .apply_newton_step(delta_t.view(), delta_beta.view(), step)
                .unwrap_or_else(|error| panic!("{label}: the descent step applies: {error}"));
            let value = trial
                .penalized_objective_total(target.view(), rho, None, 1.0)
                .unwrap_or_else(|error| panic!("{label}: the descent trial has a value: {error}"));
            if value < start {
                break Some(trial);
            }
            step *= 0.5;
            if step.abs() < f64::EPSILON {
                break None;
            }
        };
        let Some(mut trial) = accepted else {
            break;
        };
        if refreshing {
            trial.streaming_gates_frozen = false;
        }
        *term = trial;
        descent_steps += 1;
    }
    let adopted = term.collapse_prevention_gates();
    term.declare_collapse_prevention_gates(&adopted);
    let (cache, trajectory) = polish_to_root(term, target.view(), rho);
    let norm = root_norm(&trajectory);
    eprintln!(
        "[#2933 F39 {label}] root certificate: gates {} then held; {descent_steps} descent \
         steps to ‖g‖={handoff_gradient:.3e}; repulsion {:?}; amplitude ε² {:?}; exact-A polish \
         ‖g‖ {} against the floor {ROOT_GRADIENT_CEILING:.0e}",
        if refreshing { "refreshed per assembly" } else { "declared" },
        adopted.decoder_repulsion,
        adopted.amplitude_barrier,
        trajectory
            .iter()
            .map(|value| format!("{value:.3e}"))
            .collect::<Vec<_>>()
            .join(" ")
    );
    assert!(
        norm <= ROOT_GRADIENT_CEILING,
        "{label}: the fixture's root stopped at ‖g‖={norm:.3e} after {descent_steps} descent steps"
    );
    cache
}

/// #2933 F39 — free gate logits are fitted directions. With the chart
/// coordinates carrying no data jets, the response beyond the decoder border is
/// the gates'. The priced response must match the re-solved one.
#[test]
fn free_gate_logits_enter_the_priced_response_2933_f39() {
    let (mut term, target, rho) = identified_gated_state(-2.0);
    let cache = certify_production_root("free gates", &mut term, &target, &rho);
    assert_prices_resolved_response("free gates", &term, &target, &rho, &cache);
}

/// #2933 F39 — every active penalty enters the response through the observed
/// information, not as a separately subtracted trace. From the decoder smoothness
/// and the collapse gates the production lifecycle adopted at the first root, add
/// the assignment sparsity, then an amplitude barrier, then a decoder repulsion
/// pair. After each addition the root is re-certified under the declared gates;
/// the priced response must match the re-solved one, and each penalty must move
/// the response by a hundred times the resolution the comparison is held to, so a
/// priced value that ignored it would fail.
#[test]
fn each_added_penalty_moves_the_priced_response_with_the_resolved_one_2933_f39() {
    let (mut term, target, mut rho) = identified_gated_state(-6.0);
    certify_production_root("adopted gates", &mut term, &target, &rho);
    let mut gates = term.collapse_prevention_gates();
    let mut previous: Option<(String, f64)> = None;
    let stages = [
        "smoothness and the adopted collapse gates",
        "+ assignment sparsity",
        "+ amplitude barrier",
        "+ decoder repulsion",
    ];
    for (stage, label) in stages.into_iter().enumerate() {
        if stage == 1 {
            rho.log_lambda_sparse = 1.0;
        } else if stage == 2 {
            let smallest_energy = term
                .atoms
                .iter()
                .map(|atom| atom.decoder_coefficients().iter().map(|v| v * v).sum::<f64>())
                .fold(f64::INFINITY, f64::min);
            gates.amplitude_barrier = Some(0.5 * smallest_energy);
        } else if stage == 3 {
            let adopted_weight = gates
                .decoder_repulsion
                .as_ref()
                .and_then(|pairs| pairs.first())
                .map_or(0.0, |&(_, _, weight)| weight);
            gates.decoder_repulsion = Some(vec![(0, 1, adopted_weight + 5.0)]);
        }
        term.declare_collapse_prevention_gates(&gates);
        let cache = certify_production_root(label, &mut term, &target, &rho);
        let trace = assert_prices_resolved_response(label, &term, &target, &rho, &cache);
        if let Some((previous_label, previous_trace)) = previous.as_ref() {
            assert!(
                (trace - previous_trace).abs() > 100.0 * RELATIVE_TOLERANCE * previous_trace,
                "{label}: the added penalty must move the response materially; tr R {trace} \
                 against {previous_trace} at '{previous_label}'"
            );
        }
        previous = Some((label.to_string(), trace));
    }
}

/// #2933 F40 — for a linear smoother `X̂ = RX` with `X = μ + ε`,
/// `E‖X − X̂‖² = ‖(I − R)μ‖² + σ²·‖I − R‖²_F`, so `RSS/(N − tr R)` has expectation
/// `σ²·(N − 2 tr R + ‖R‖²_F)/(N − tr R) < σ²` for any shrinking smoother, even with
/// a mean the smoother reproduces. One ungated atom with a coordinate-free
/// orthonormal basis `Φ` and penalty `λ·diag(0, 1, …, 1)` is the ridge smoother
/// `R = Φ·diag(s)·Φᵀ ⊗ I_p` with `s = (1, ½, …, ½)`: `tr R = 28` and
/// `‖I − R‖²_F = 42` on `N = 80`, where `RSS/(N − tr R)` averages `0.808·σ²` and
/// its 95% bands cover about 92%. Repeated noise around a zero mean and around a
/// mean in the unpenalized column must give an unbiased scale and nominal
/// coverage.
#[test]
fn ridge_smoother_scale_is_unbiased_and_covers_2933_f40() {
    let n = 10usize;
    let p = 8usize;
    let width = 6usize;
    let sigma = 0.3_f64;
    let replicates = 300usize;
    let z = 1.959_963_984_540_054_f64;
    let phi = Array2::<f64>::from_shape_fn((n, width), |(row, basis)| {
        let theta = std::f64::consts::TAU * row as f64 / n as f64;
        match basis {
            0 => 1.0 / (n as f64).sqrt(),
            _ => {
                let harmonic = ((basis + 1) / 2) as f64;
                let wave = if basis % 2 == 1 {
                    (harmonic * theta).cos()
                } else {
                    (harmonic * theta).sin()
                };
                wave * (2.0 / n as f64).sqrt()
            }
        }
    });
    let gram = phi.t().dot(&phi);
    for i in 0..width {
        for j in 0..width {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!(
                (gram[[i, j]] - expected).abs() < 1e-12,
                "the harmonic basis must be orthonormal at ({i}, {j}): {}",
                gram[[i, j]]
            );
        }
    }
    let penalty = Array2::<f64>::from_diag(&Array1::from_shape_fn(width, |basis| {
        if basis == 0 { 0.0 } else { 1.0 }
    }));
    let shrink: Vec<f64> = (0..width).map(|basis| 1.0 / (1.0 + penalty[[basis, basis]])).collect();
    // The oracle smoother on one output channel, and the diagonal of R Rᵀ.
    let smoother = Array2::<f64>::from_shape_fn((n, n), |(i, j)| {
        (0..width).map(|b| phi[[i, b]] * shrink[b] * phi[[j, b]]).sum::<f64>()
    });
    let response_variance: Vec<f64> = (0..n)
        .map(|i| (0..width).map(|b| (shrink[b] * phi[[i, b]]).powi(2)).sum::<f64>())
        .collect();
    let trace = p as f64 * shrink.iter().sum::<f64>();
    let residual_dof = p as f64
        * ((n - width) as f64 + shrink.iter().map(|s| (1.0 - s) * (1.0 - s)).sum::<f64>());
    assert!((trace - 28.0).abs() < 1e-12 && (residual_dof - 42.0).abs() < 1e-12);

    let base_term = {
        let atom = fixed_basis_atom("ridge", phi.clone(), Array2::<f64>::zeros((width, p)));
        let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
            Array2::<f64>::zeros((n, 1)),
            vec![Array2::<f64>::zeros((n, 1))],
            vec![LatentManifold::Circle { period: 1.0 }],
            AssignmentMode::softmax(1.0),
        )
        .expect("assignment: one logit column and one coordinate block");
        SaeManifoldTerm::new(vec![atom], assignment).expect("the atom matches its block")
    };
    let rho = SaeManifoldRho::new(0.0, 0.0, vec![Array1::<f64>::zeros(1)])
        .for_assignment(&base_term.assignment);
    let channel_means = [2.0, -1.0, 0.5, 3.0, -2.0, 1.0, 0.7, -0.4];
    for (case, mean_scale) in [("zero mean", 0.0_f64), ("reproduced nonzero mean", 1.0)] {
        let mean = Array2::<f64>::from_shape_fn((n, p), |(row, out)| {
            mean_scale * channel_means[out] * phi[[row, 0]]
        });
        let mut noise = NormalStream {
            state: 0x2933_F40A_0000_0001 + mean_scale.to_bits(),
            spare: None,
        };
        let mut ratio_sum = 0.0_f64;
        let mut covered = 0usize;
        for replicate in 0..replicates {
            let target = Array2::<f64>::from_shape_fn((n, p), |(row, out)| {
                mean[[row, out]] + sigma * noise.next()
            });
            let mut term = base_term.clone();
            // Start at the closed-form ridge solution; the exact-A polish certifies it
            // is the root of the assembled stationarity.
            let decoder = Array2::<f64>::from_shape_fn((width, p), |(basis, out)| {
                shrink[basis] * (0..n).map(|row| phi[[row, basis]] * target[[row, out]]).sum::<f64>()
            });
            term.atoms[0].decoder_coefficients_mut().assign(&decoder);
            let (cache, trajectory) = polish_to_root(&mut term, target.view(), &rho);
            let norm = root_norm(&trajectory);
            assert!(norm <= ROOT_GRADIENT_CEILING, "{case}: the ridge root stalled at ‖g‖={norm:.3e}");
            let fitted = term.try_fitted_for_rho(&rho).expect("the ridge fit reconstructs");
            let oracle_fitted = smoother.dot(&target);
            let fit_gap = (&fitted - &oracle_fitted).iter().fold(0.0_f64, |m, v| m.max(v.abs()));
            assert!(fit_gap < 1e-9, "{case}: the fit must be the oracle ridge smoother, gap {fit_gap:.3e}");
            let loss = term.loss(target.view(), &rho).expect("the ridge state has a loss");
            let residual = term
                .reconstruction_residual(target.view(), &rho)
                .expect("the ridge state has a residual");
            let dispersion = term
                .reconstruction_dispersion(&loss, &cache, &rho, residual.view())
                .expect("the ridge state prices a dispersion");
            let rss = 2.0 * loss.data_fit;
            let phi_hat = dispersion.raw_output_noise_variance;
            if replicate == 0 {
                eprintln!(
                    "[#2933 F40 ridge {case}] rss={rss:.9e} phi={phi_hat:.9e} implied dof \
                     {:.9e} against ‖I−R‖²={residual_dof} (N − tr R = {})",
                    rss / phi_hat,
                    (n * p) as f64 - trace
                );
                assert!(
                    (rss / phi_hat - residual_dof).abs() <= 1e-6 * residual_dof,
                    "{case}: the scale equation must use the residual dof ‖I − R‖²_F = \
                     {residual_dof}, got RSS/φ = {}",
                    rss / phi_hat
                );
                assert!(
                    (dispersion.likelihood_dispersion - phi_hat).abs() <= 1e-12 * phi_hat,
                    "{case}: with no row metric the two frames coincide"
                );
            }
            ratio_sum += phi_hat / (sigma * sigma);
            for row in 0..n {
                let half_width = z * (phi_hat * response_variance[row]).sqrt();
                for out in 0..p {
                    if (fitted[[row, out]] - mean[[row, out]]).abs() <= half_width {
                        covered += 1;
                    }
                }
            }
        }
        let mean_ratio = ratio_sum / replicates as f64;
        let coverage = covered as f64 / (replicates * n * p) as f64;
        eprintln!(
            "[#2933 F40 ridge {case}] mean φ̂/σ² = {mean_ratio:.6} over {replicates} replicates, \
             95% band coverage {coverage:.6}"
        );
        assert!(
            (mean_ratio - 1.0).abs() <= 0.05,
            "{case}: the scale must be unbiased for σ² when the smoother reproduces the mean; \
             mean φ̂/σ² = {mean_ratio}"
        );
        assert!(
            (0.935..=0.965).contains(&coverage),
            "{case}: 95% bands priced with φ̂ must cover near nominal; got {coverage}"
        );
    }
}
