#![cfg(test)]
//! Finite-difference gates for the survival marginal-slope ψ (design-moving)
//! first-order terms (#2765 / #2767, the `#979`/`#1040` lane).
//!
//! `psi_terms_inner` publishes the three objects the generic outer-REML hyper
//! assembly consumes for one design-moving coordinate:
//!
//! ```text
//!   objective_psi = ∂_ψ  ℓ̄ |_β
//!   score_psi     = ∂_ψ ∇_β ℓ̄ |_β
//!   hessian_psi   = ∂_ψ ∇²_β ℓ̄ |_β
//! ```
//!
//! where `ℓ̄` is the family's own joint objective (the one
//! `exact_newton_joint_gradient_evaluation` / `exact_newton_joint_hessian`
//! report) and `ψ` moves a block's design as `X(ψ) = X + ψ·X_ψ`.
//!
//! Nothing in the module's existing coverage compared those three against a
//! finite difference of the very functions they claim to differentiate: the
//! shipped gates check finiteness, subsample-vs-unsampled equality, and
//! batched-vs-per-axis agreement — all of which a *consistently wrong*
//! derivative passes. The end-to-end audit
//! (`tests/survival/survival/survival_marginal_slope_outer_gradient_fd_1040.rs`)
//! records the ψ block of this criterion disagreeing with its own Ridders
//! oracle by `1.000e0` and `1.377e-1` relative, with the oracle's uncertainty
//! six to seven orders below the gap. These gates put that comparison where it
//! can be attributed to a single term.
//!
//! The perturbation is **exactly linear** in ψ by construction, so `X_ψ` is the
//! true derivative to machine precision and a central difference is limited
//! only by the third derivative of `ℓ̄` in ψ — a Richardson pair certifies the
//! remainder in-test rather than assuming it.

use super::*;
use crate::custom_family::{
    CustomFamily, CustomFamilyBlockPsiDerivative, CustomFamilyHyperLayout,
};
use gam_linalg::matrix::DenseDesignMatrix;
use ndarray::{Array1, Array2, Axis};
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;

/// Which block's design the ψ coordinate moves.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum PsiBlock {
    /// Block 1 — the marginal (population survival index) covariate design.
    Marginal,
    /// Block 2 — the log-slope covariate design.
    LogSlope,
}

const N_ROWS: usize = 24;

/// A scalar unit latent-score covariance, matching the shipped fixtures.
fn unit_score_covariance() -> ScoreCovarianceField {
    ScoreCovarianceField::pooled(
        MarginalSlopeCovariance::diagonal(ndarray::array![1.0])
            .expect("a 1x1 unit latent-score covariance"),
    )
}

fn base_marginal_design() -> Array2<f64> {
    Array2::from_shape_fn((N_ROWS, 2), |(i, j)| {
        let t = (i as f64 + 0.5) / (N_ROWS as f64);
        match j {
            0 => 0.30 + 0.40 * t,
            _ => -0.20 + 0.55 * (2.3 * t).sin(),
        }
    })
}

fn base_logslope_design() -> Array2<f64> {
    Array2::from_shape_fn((N_ROWS, 2), |(i, j)| {
        let t = (i as f64 + 0.5) / (N_ROWS as f64);
        match j {
            0 => 0.20 + 0.50 * t,
            _ => 0.15 * (1.7 * t + 0.4).cos(),
        }
    })
}

fn marginal_design_derivative() -> Array2<f64> {
    Array2::from_shape_fn((N_ROWS, 2), |(i, j)| {
        let t = (i as f64 + 0.5) / (N_ROWS as f64);
        match j {
            0 => 0.40 + 0.30 * (3.1 * t).cos(),
            _ => -0.25 + 0.45 * t,
        }
    })
}

fn logslope_design_derivative() -> Array2<f64> {
    Array2::from_shape_fn((N_ROWS, 2), |(i, j)| {
        let t = (i as f64 + 0.5) / (N_ROWS as f64);
        match j {
            0 => 0.35 - 0.20 * t,
            _ => 0.10 + 0.30 * (2.7 * t).sin(),
        }
    })
}

/// The family at a ψ displacement `t`, with `X_block(t) = X_block + t·X_ψ`.
///
/// Every other input is held fixed, so `t` moves exactly the one design the ψ
/// coordinate owns and nothing else.
fn family_at(block: PsiBlock, t: f64) -> SurvivalMarginalSlopeFamily {
    let n = N_ROWS;
    let event: Array1<f64> =
        Array1::from_iter((0..n).map(|i| if (i * 31 + 7) % 5 >= 3 { 1.0 } else { 0.0 }));
    let weights: Array1<f64> =
        Array1::from_iter((0..n).map(|i| 0.5 + ((i * 13 + 4) % 5) as f64 * 0.1));
    let z: Array1<f64> = Array1::from_iter(
        (0..n).map(|i| -1.0 + 2.0 * ((i * 17 + 5) % n) as f64 / (n as f64)),
    );
    let offset_entry: Array1<f64> =
        Array1::from_iter((0..n).map(|i| -0.4 + 0.7 * ((i * 11 + 3) % n) as f64 / (n as f64)));
    let offset_exit: Array1<f64> =
        Array1::from_iter((0..n).map(|i| 0.1 + 0.6 * ((i * 19 + 7) % n) as f64 / (n as f64)));
    let derivative_offset_exit: Array1<f64> =
        Array1::from_iter((0..n).map(|i| 0.5 + 0.05 * ((i * 23 + 1) % 3) as f64));

    let mut marginal = base_marginal_design();
    let mut logslope = base_logslope_design();
    match block {
        PsiBlock::Marginal => marginal.scaled_add(t, &marginal_design_derivative()),
        PsiBlock::LogSlope => logslope.scaled_add(t, &logslope_design_derivative()),
    }

    SurvivalMarginalSlopeFamily {
        n,
        event: Arc::new(event),
        weights: Arc::new(weights),
        z: Arc::new(z.insert_axis(Axis(1))),
        score_covariance: unit_score_covariance(),
        gaussian_frailty_sd: None,
        family_hyper: SurvivalMarginalSlopeFamilyHyperState::default(),
        derivative_guard: 1e-6,
        design_entry: DesignMatrix::from(Array2::zeros((n, 0))),
        design_exit: DesignMatrix::from(Array2::zeros((n, 0))),
        design_derivative_exit: DesignMatrix::from(Array2::zeros((n, 0))),
        offset_entry: Arc::new(offset_entry),
        offset_exit: Arc::new(offset_exit),
        derivative_offset_exit: Arc::new(derivative_offset_exit),
        marginal_design: DesignMatrix::from(marginal),
        logslope_layout: (DesignMatrix::from(logslope)).into(),
        score_warp: None,
        link_dev: None,
        influence_absorber: None,
        time_linear_constraints: None,
        time_wiggle_knots: None,
        time_wiggle_degree: None,
        time_wiggle_ncols: 0,
        intercept_warm_starts: None,
        auto_subsample_phase_counter: Arc::new(AtomicUsize::new(0)),
        auto_subsample_last_rho: Arc::new(std::sync::Mutex::new(None)),
    }
}

fn marginal_beta() -> Array1<f64> {
    ndarray::array![0.35, -0.18]
}

fn logslope_beta() -> Array1<f64> {
    ndarray::array![0.22, 0.13]
}

/// Block states at fixed β for the family at displacement `t`.
///
/// `η` is a cached function of `(X(t), β)`, so it MUST be rebuilt at each
/// displacement: holding β fixed is the contract, holding a stale η fixed
/// would difference a different function from the one the ψ terms differentiate.
fn states_at(family: &SurvivalMarginalSlopeFamily) -> Vec<ParameterBlockState> {
    let m_beta = marginal_beta();
    let g_beta = logslope_beta();
    let m_design = family.marginal_design.to_dense().to_owned();
    let g_design = family
        .logslope_layout
        .coefficient_design()
        .to_dense()
        .to_owned();
    vec![
        ParameterBlockState {
            beta: Array1::zeros(0),
            eta: Array1::zeros(family.n),
        },
        ParameterBlockState {
            eta: m_design.dot(&m_beta),
            beta: m_beta,
        },
        ParameterBlockState {
            eta: g_design.dot(&g_beta),
            beta: g_beta,
        },
    ]
}

fn specs_for(family: &SurvivalMarginalSlopeFamily) -> Vec<ParameterBlockSpec> {
    vec![
        fd_blockspec(0),
        fd_blockspec(family.marginal_design.ncols()),
        fd_blockspec(family.logslope_layout.coefficient_design().ncols()),
    ]
}

fn fd_blockspec(cols: usize) -> ParameterBlockSpec {
    use std::sync::atomic::Ordering;
    static SEQ: AtomicUsize = AtomicUsize::new(0);
    let idx = SEQ.fetch_add(1, Ordering::Relaxed);
    ParameterBlockSpec {
        name: format!("psi_fd_{idx}"),
        design: DesignMatrix::Dense(DenseDesignMatrix::from(Array2::zeros((1, cols)))),
        offset: Array1::zeros(1),
        penalties: Vec::new(),
        nullspace_dims: Vec::new(),
        initial_log_lambdas: Array1::zeros(0),
        initial_beta: Some(Array1::zeros(cols)),
        gauge_priority: 100,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    }
}

/// One design-moving ψ axis on `block`, with the analytic `X_ψ` this fixture's
/// `family_at` realizes exactly.
fn derivative_blocks(block: PsiBlock) -> Vec<Vec<CustomFamilyBlockPsiDerivative>> {
    let (x_psi, p) = match block {
        PsiBlock::Marginal => (marginal_design_derivative(), 2),
        PsiBlock::LogSlope => (logslope_design_derivative(), 2),
    };
    let entry = vec![CustomFamilyBlockPsiDerivative::new(
        None,
        x_psi,
        Array2::zeros((p, p)),
        None,
        None,
        None,
        None,
    )];
    match block {
        PsiBlock::Marginal => vec![Vec::new(), entry, Vec::new()],
        PsiBlock::LogSlope => vec![Vec::new(), Vec::new(), entry],
    }
}

fn hyper_layout(block: PsiBlock) -> CustomFamilyHyperLayout {
    CustomFamilyHyperLayout::new(derivative_blocks(block), Vec::new(), Array1::zeros(1))
        .expect("one design ψ axis")
}

/// The family's own `(objective, score, Hessian)` triple at displacement `t`.
///
/// All three come from the hooks the outer assembly reads, so a derivative that
/// matches these matches the function the criterion is actually built on.
fn objective_score_hessian(block: PsiBlock, t: f64) -> (f64, Array1<f64>, Array2<f64>) {
    let family = family_at(block, t);
    let states = states_at(&family);
    let specs = specs_for(&family);
    let evaluation = family
        .exact_newton_joint_gradient_evaluation(&states, &specs)
        .expect("joint gradient evaluation")
        .expect("survival marginal-slope publishes a joint gradient evaluation");
    let hessian = family
        .exact_newton_joint_hessian(&states)
        .expect("joint hessian")
        .expect("survival marginal-slope publishes an explicit joint hessian");
    (evaluation.log_likelihood, evaluation.gradient, hessian)
}

/// Central difference plus its Richardson partner, so the gate reports the
/// oracle's own truncation instead of assuming it.
struct Ridders {
    value: f64,
    uncertainty: f64,
}

fn ridders(coarse: f64, fine: f64) -> Ridders {
    // Central differences are `O(h²)`: the `h/2` estimate carries a quarter of
    // the coarse remainder, so `(4·fine − coarse)/3` cancels it and the
    // difference of the two raw estimates bounds what is left.
    let value = (4.0 * fine - coarse) / 3.0;
    Ridders {
        value,
        uncertainty: (fine - coarse).abs() / 3.0,
    }
}

struct PsiFiniteDifference {
    objective: Ridders,
    score: Vec<Ridders>,
    hessian: Vec<Ridders>,
    dim: usize,
}

fn finite_difference(block: PsiBlock, h: f64) -> PsiFiniteDifference {
    let coarse_plus = objective_score_hessian(block, h);
    let coarse_minus = objective_score_hessian(block, -h);
    let fine_plus = objective_score_hessian(block, 0.5 * h);
    let fine_minus = objective_score_hessian(block, -0.5 * h);

    let dim = coarse_plus.1.len();
    let objective = ridders(
        (coarse_plus.0 - coarse_minus.0) / (2.0 * h),
        (fine_plus.0 - fine_minus.0) / h,
    );
    let score = (0..dim)
        .map(|i| {
            ridders(
                (coarse_plus.1[i] - coarse_minus.1[i]) / (2.0 * h),
                (fine_plus.1[i] - fine_minus.1[i]) / h,
            )
        })
        .collect();
    let hessian = (0..dim * dim)
        .map(|flat| {
            let (r, c) = (flat / dim, flat % dim);
            ridders(
                (coarse_plus.2[[r, c]] - coarse_minus.2[[r, c]]) / (2.0 * h),
                (fine_plus.2[[r, c]] - fine_minus.2[[r, c]]) / h,
            )
        })
        .collect();
    PsiFiniteDifference {
        objective,
        score,
        hessian,
        dim,
    }
}

fn analytic_terms(block: PsiBlock) -> (f64, Array1<f64>, Array2<f64>) {
    let family = family_at(block, 0.0);
    let states = states_at(&family);
    let layout = hyper_layout(block);
    let terms = family
        .psi_terms(&states, layout.design_derivative_blocks(), 0)
        .expect("analytic ψ terms")
        .expect("a design ψ axis on a supported block publishes terms");
    let total: usize = states.iter().map(|state| state.beta.len()).sum();
    let hessian = match terms.hessian_psi_operator.as_ref() {
        Some(operator) => operator.mul_mat(&Array2::<f64>::eye(total)),
        None => terms.hessian_psi.clone(),
    };
    (terms.objective_psi, terms.score_psi.clone(), hessian)
}

/// Grade `analytic` against a Ridders-certified `fd`, refusing to charge a gap
/// to the analytic term unless the oracle's own uncertainty is well below it.
fn assert_matches(label: &str, analytic: f64, fd: &Ridders, scale: f64) {
    let gap = (analytic - fd.value).abs();
    let denominator = scale.max(analytic.abs()).max(fd.value.abs()).max(1e-12);
    assert!(
        fd.uncertainty <= 0.05 * denominator,
        "{label}: the finite-difference oracle did not resolve this component \
         (value={:.6e} uncertainty={:.3e} scale={denominator:.3e}); the analytic \
         term is not on trial here",
        fd.value,
        fd.uncertainty,
    );
    assert!(
        gap <= 1e-6 * denominator + 4.0 * fd.uncertainty,
        "{label}: analytic={analytic:.9e} fd={:.9e} gap={gap:.3e} \
         rel={:.3e} oracle_uncertainty={:.3e}",
        fd.value,
        gap / denominator,
        fd.uncertainty,
    );
}

fn max_abs(values: &Array1<f64>) -> f64 {
    values.iter().fold(0.0_f64, |acc, v| acc.max(v.abs()))
}

fn max_abs2(values: &Array2<f64>) -> f64 {
    values.iter().fold(0.0_f64, |acc, v| acc.max(v.abs()))
}

fn run_first_order_gate(block: PsiBlock) {
    let (objective_psi, score_psi, hessian_psi) = analytic_terms(block);
    let fd = finite_difference(block, 1e-3);
    assert_eq!(
        score_psi.len(),
        fd.dim,
        "ψ score must live in the flattened joint coefficient space"
    );
    assert_eq!(hessian_psi.dim(), (fd.dim, fd.dim));

    let objective_scale = objective_psi.abs().max(fd.objective.value.abs());
    assert_matches(
        &format!("{block:?} objective_psi"),
        objective_psi,
        &fd.objective,
        objective_scale,
    );

    let score_scale = max_abs(&score_psi).max(
        fd.score
            .iter()
            .fold(0.0_f64, |acc, r| acc.max(r.value.abs())),
    );
    for (i, oracle) in fd.score.iter().enumerate() {
        assert_matches(
            &format!("{block:?} score_psi[{i}]"),
            score_psi[i],
            oracle,
            score_scale,
        );
    }

    let hessian_scale = max_abs2(&hessian_psi).max(
        fd.hessian
            .iter()
            .fold(0.0_f64, |acc, r| acc.max(r.value.abs())),
    );
    for (flat, oracle) in fd.hessian.iter().enumerate() {
        let (r, c) = (flat / fd.dim, flat % fd.dim);
        assert_matches(
            &format!("{block:?} hessian_psi[{r},{c}]"),
            hessian_psi[[r, c]],
            oracle,
            hessian_scale,
        );
    }
}

/// The marginal (population-index) design ψ axis: every published first-order
/// term must be the derivative of the function it names.
#[test]
fn marginal_psi_first_order_terms_match_finite_difference_2765() {
    run_first_order_gate(PsiBlock::Marginal);
}

/// The log-slope design ψ axis. Filed separately from the marginal axis because
/// the two enter the row program through different primary channels
/// (`q₀,q₁` versus `g`), so one being right says nothing about the other.
#[test]
fn logslope_psi_first_order_terms_match_finite_difference_2765() {
    run_first_order_gate(PsiBlock::LogSlope);
}

/// A ψ axis whose design derivative is exactly zero must publish exactly zero
/// terms — the degenerate end of the same contract, and the one case where the
/// finite difference is exact rather than certified.
#[test]
fn a_zero_psi_design_derivative_publishes_zero_terms_2765() {
    let family = family_at(PsiBlock::Marginal, 0.0);
    let states = states_at(&family);
    let blocks = vec![
        Vec::new(),
        vec![CustomFamilyBlockPsiDerivative::new(
            None,
            Array2::zeros((N_ROWS, 2)),
            Array2::zeros((2, 2)),
            None,
            None,
            None,
            None,
        )],
        Vec::new(),
    ];
    let terms = family
        .psi_terms(&states, &blocks, 0)
        .expect("zero ψ terms")
        .expect("a design ψ axis publishes terms even when its derivative vanishes");
    assert_eq!(terms.objective_psi, 0.0);
    assert!(terms.score_psi.iter().all(|value| *value == 0.0));
    let total: usize = states.iter().map(|state| state.beta.len()).sum();
    let hessian = match terms.hessian_psi_operator.as_ref() {
        Some(operator) => operator.mul_mat(&Array2::<f64>::eye(total)),
        None => terms.hessian_psi.clone(),
    };
    assert!(hessian.iter().all(|value| *value == 0.0));
}
