//! #2933 F40 — the reconstruction dispersion's scale equation uses the residual
//! degrees of freedom `‖I − R‖²_F` of its fitted response, not `N − tr R`.
//!
//! A ridge smoother with a known response is fitted repeatedly to noise around a
//! zero mean and around a mean it reproduces. The production dispersion must be
//! unbiased for the noise variance and its 95% bands must cover near nominal.
//! Each fit is certified as the root of the assembled stationarity by exact-A
//! Newton steps before its dispersion is read.
use super::*;
use crate::manifold::arrow_solver::SaeArrowVector;
use gam_terms::latent::LatentManifold;
use ndarray::{Array1, Array2, ArrayView2};

/// Largest number of exact-A Newton steps one re-solve may take.
const ROOT_POLISH_STEPS: usize = 40;

/// A root is admitted only once its gradient has reached the arithmetic floor
/// of an order-one objective on a few dozen scalar observations.
const ROOT_GRADIENT_CEILING: f64 = 1.0e-10;

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
/// stops halving; return the undamped evidence factorization and the norm.
fn polish_to_root(
    term: &mut SaeManifoldTerm,
    target: ArrayView2<'_, f64>,
    rho: &SaeManifoldRho,
) -> (ArrowFactorCache, f64) {
    let options = term.evidence_factor_options();
    let mut previous = f64::INFINITY;
    for _ in 0..ROOT_POLISH_STEPS {
        let system = term
            .assemble_arrow_schur(target, rho, None)
            .expect("the arrow system assembles at every re-solve iterate");
        let g = gradient(&system);
        let norm = (g.t.dot(&g.t) + g.beta.dot(&g.beta)).sqrt();
        let (_, _, cache) = solve_arrow_newton_step_with_options(&system, 0.0, 0.0, &options)
            .expect("the majorizer factors undamped at every re-solve iterate");
        if !(norm < 0.5 * previous) {
            return (cache, norm);
        }
        let step = term
            .solve_exact_stationarity(rho, target, &cache, &g)
            .expect("the exact stationarity pseudoinverse solves at every re-solve iterate");
        term.apply_newton_step((-&step.t).view(), (-&step.beta).view(), 1.0)
            .expect("the exact Newton step applies to the term state");
        previous = norm;
    }
    panic!("the exact-A Newton re-solve did not reach its plateau in {ROOT_POLISH_STEPS} steps");
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
        _coords: ArrayView2<'_, f64>,
    ) -> Result<SaeBasisThirdJetCapability, String> {
        Ok(SaeBasisThirdJetCapability::CertifiedZero)
    }
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

    let coords = Array2::<f64>::zeros((n, 1));
    let evaluator = Arc::new(FixedRowBasis { phi: phi.clone() });
    let base_term = {
        let (basis, jet) = evaluator.evaluate(coords.view()).expect("fixed basis evaluates");
        let atom = SaeManifoldAtom::new_with_provided_function_gram(
            "ridge".to_string(),
            SaeAtomBasisKind::Periodic,
            1,
            basis,
            jet,
            Array2::<f64>::zeros((width, p)),
            penalty.clone(),
        )
        .expect("ridge atom: basis width, latent dimension and decoder shape agree")
        .with_basis_evaluator(evaluator);
        let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
            Array2::<f64>::zeros((n, 1)),
            vec![coords.clone()],
            vec![LatentManifold::Circle { period: 1.0 }],
            AssignmentMode::softmax(1.0),
        )
        .expect("assignment: one logit column and one coordinate block");
        SaeManifoldTerm::new(vec![atom], assignment).expect("the atom matches its block")
    };
    let rho = SaeManifoldRho::new(0.0, 0.0, vec![Array1::<f64>::zeros(1)])
        .for_assignment(AssignmentMode::softmax(1.0));
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
            let (cache, norm) = polish_to_root(&mut term, target.view(), &rho);
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
