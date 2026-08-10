#![cfg(test)]
//! Finite-difference gates for every survival marginal-slope ψ lane
//! (#2765 / #2767, the `#979`/`#1040` lane).
//!
//! The family publishes three objects per outer ψ coordinate, and the generic
//! outer-REML hyper assembly (`build_psi_hyper_coords`) consumes all three:
//!
//! ```text
//!   objective_psi = ∂_ψ  ℓ̄ |_β
//!   score_psi     = ∂_ψ ∇_β ℓ̄ |_β
//!   hessian_psi   = ∂_ψ ∇²_β ℓ̄ |_β
//! ```
//!
//! where `ℓ̄` is the family's own joint objective — the one
//! `exact_newton_joint_gradient_evaluation` and `exact_newton_joint_hessian`
//! report — and there are two structurally different kinds of ψ:
//!
//! * a **design** axis, which moves one block's design (`X(ψ) = X + ψ·X_ψ`);
//!   this is the matern/duchon length scale, served by `psi_terms_inner`;
//! * a **baseline** axis, which moves the parametric baseline chart's own
//!   coordinates and therefore the three offset channels of the location
//!   index; served by `baseline_exact_joint_psi_terms_with_options`.
//!
//! Nothing compared any of them against a finite difference of the functions
//! they claim to differentiate. The shipped ψ coverage checks FINITENESS,
//! subsample-vs-unsampled equality, and batched-vs-per-axis agreement — every
//! one of which a *consistently wrong* derivative passes. Meanwhile the
//! end-to-end audit
//! (`tests/survival/survival/survival_marginal_slope_outer_gradient_fd_1040.rs`)
//! records this criterion's ψ gradient disagreeing with its own Ridders oracle
//! by `1.000e0` and `1.377e-1` relative, with the oracle's uncertainty six to
//! seven orders below the gap, and #2765's acceptance fit dies in a line search
//! along a pure baseline-ψ direction. Neither gap had anywhere to be
//! attributed, because no gate sat between the row program and the whole fit.
//!
//! Every arm is run in BOTH slope frames: the four-primary static frame
//! `(q₀, q₁, q̇₁, g)` and the six-primary follow-up-varying frame
//! `(q₀, q₁, q̇₁, g₀, g₁, ġ₁)` that #2765 introduced. A ψ calculus that is right
//! in one frame says nothing about the other — the pullback out of primary
//! space is a different map in each.
//!
//! The perturbation is exact by construction in both lanes (a design axis is
//! exactly linear in ψ; a baseline axis re-derives the chart at the displaced
//! coordinate), so the only error in the oracle is its own truncation, which a
//! Richardson pair certifies in-test rather than assuming.

use super::*;
use crate::custom_family::{
    BlockwiseFitOptions, CustomFamily, CustomFamilyBlockPsiDerivative, CustomFamilyHyperLayout,
};
use crate::survival::construction::{
    SurvivalBaselineConfig, SurvivalBaselineTarget, build_survival_marginal_slope_baseline_geometry,
    survival_baseline_config_from_theta,
};
use gam_linalg::matrix::DenseDesignMatrix;
use ndarray::{Array1, Array2, Axis};
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;

const N_ROWS: usize = 24;

/// A scalar unit latent-score covariance, matching the shipped fixtures.
fn unit_score_covariance() -> ScoreCovarianceField {
    ScoreCovarianceField::pooled(
        MarginalSlopeCovariance::diagonal(ndarray::array![1.0])
            .expect("a 1x1 unit latent-score covariance"),
    )
}

/// Which outer ψ coordinate is under test.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum PsiAxis {
    /// A design-moving ψ on block 1 — the marginal (population index) design.
    MarginalDesign,
    /// A design-moving ψ on block 2 — the log-slope design.
    LogSlopeDesign,
    /// A baseline-chart coordinate. Moves the offset channels of the location
    /// index on every row and touches no design.
    Baseline(usize),
}

/// Which slope frame the row program runs in.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum SlopeFrame {
    /// Four primaries `(q₀, q₁, q̇₁, g)` — a time-constant slope.
    Static,
    /// Six primaries `(q₀, q₁, q̇₁, g₀, g₁, ġ₁)` — a follow-up-varying slope.
    FollowUpVarying,
}

impl SlopeFrame {
    fn label(self) -> &'static str {
        match self {
            Self::Static => "static-slope",
            Self::FollowUpVarying => "follow-up-varying-slope",
        }
    }
}

// ── The fixture, as explicit functions of the displacement ──────────────────

fn age_entry() -> Array1<f64> {
    Array1::from_shape_fn(N_ROWS, |row| 0.25 + 0.10 * row as f64)
}

fn age_exit() -> Array1<f64> {
    let entry = age_entry();
    Array1::from_shape_fn(N_ROWS, |row| entry[row] + 0.75 + 0.03 * row as f64)
}

/// The baseline chart this fixture differentiates. Gompertz-Makeham carries
/// three coordinates whose partials are already FD-certified one layer down
/// (`marginal_slope_baseline_theta_partials_match_fd_for_gompertz_makeham`), so
/// a failure here is in the pullback, not in the chart.
fn baseline_config() -> SurvivalBaselineConfig {
    SurvivalBaselineConfig {
        target: SurvivalBaselineTarget::GompertzMakeham,
        scale: None,
        shape: Some(0.08),
        rate: Some(0.22),
        makeham: Some(0.04),
    }
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

fn base_logslope_exit_design() -> Array2<f64> {
    Array2::from_shape_fn((N_ROWS, 2), |(i, j)| {
        let t = (i as f64 + 0.5) / (N_ROWS as f64);
        match j {
            0 => 0.20 + 0.50 * t,
            _ => 0.15 * (1.7 * t + 0.4).cos(),
        }
    })
}

/// The entry-time margin of a follow-up-varying slope. Deliberately unequal to
/// the exit design: `g₀ = g₁` is exactly the degeneracy the static frame
/// already covers.
fn base_logslope_entry_design() -> Array2<f64> {
    Array2::from_shape_fn((N_ROWS, 2), |(i, j)| {
        let t = (i as f64 + 0.5) / (N_ROWS as f64);
        match j {
            0 => 0.14 + 0.31 * t,
            _ => 0.09 * (2.1 * t + 0.7).cos(),
        }
    })
}

/// The exit-RATE margin `∂B(log t)/∂t` of a follow-up-varying slope.
fn base_logslope_rate_design() -> Array2<f64> {
    Array2::from_shape_fn((N_ROWS, 2), |(i, j)| {
        let t = (i as f64 + 0.5) / (N_ROWS as f64);
        match j {
            0 => 0.07 - 0.05 * t,
            _ => 0.04 + 0.06 * (1.3 * t).sin(),
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

/// The design derivative of the follow-up ENTRY margin.
///
/// A design ψ is one length scale shared by the whole covariate factor, so when
/// the slope carries a time margin the SAME ψ moves all three of its channels.
/// Handing the derivative only to the exit channel would differentiate a model
/// nobody fitted.
fn logslope_entry_design_derivative() -> Array2<f64> {
    Array2::from_shape_fn((N_ROWS, 2), |(i, j)| {
        let t = (i as f64 + 0.5) / (N_ROWS as f64);
        match j {
            0 => 0.24 - 0.13 * t,
            _ => 0.06 + 0.21 * (2.7 * t).sin(),
        }
    })
}

fn logslope_rate_design_derivative() -> Array2<f64> {
    Array2::from_shape_fn((N_ROWS, 2), |(i, j)| {
        let t = (i as f64 + 0.5) / (N_ROWS as f64);
        match j {
            0 => 0.12 - 0.07 * t,
            _ => 0.02 + 0.10 * (2.7 * t).sin(),
        }
    })
}

/// The family at ψ displacement `t` along `axis`, in `frame`.
///
/// Every input the displaced coordinate does NOT own is held fixed, so `t`
/// moves exactly what the ψ coordinate owns and nothing else.
fn family_at(axis: PsiAxis, frame: SlopeFrame, t: f64) -> SurvivalMarginalSlopeFamily {
    let n = N_ROWS;
    let event: Array1<f64> =
        Array1::from_iter((0..n).map(|i| if (i * 31 + 7) % 5 >= 3 { 1.0 } else { 0.0 }));
    let weights: Array1<f64> =
        Array1::from_iter((0..n).map(|i| 0.5 + ((i * 13 + 4) % 5) as f64 * 0.1));
    let z: Array1<f64> =
        Array1::from_iter((0..n).map(|i| -1.0 + 2.0 * ((i * 17 + 5) % n) as f64 / (n as f64)));

    // Baseline chart: displaced along its own coordinate when that is the axis
    // under test, and re-derived from the displaced coordinate rather than
    // perturbed in place — the chart's own encoder/decoder round trip is part
    // of what the outer solve walks.
    let mut config = baseline_config();
    if let PsiAxis::Baseline(index) = axis {
        let mut theta = crate::survival::construction::survival_baseline_theta_from_config(&config)
            .expect("baseline theta")
            .expect("Gompertz-Makeham carries a baseline chart");
        theta[index] += t;
        config = survival_baseline_config_from_theta(config.target, &theta)
            .expect("the displaced baseline coordinate stays in the chart's domain");
    }
    let geometry = Arc::new(
        build_survival_marginal_slope_baseline_geometry(&age_entry(), &age_exit(), &config)
            .expect("build baseline geometry")
            .expect("Gompertz-Makeham has a nonlinear baseline chart"),
    );

    let mut marginal = base_marginal_design();
    let mut logslope_exit = base_logslope_exit_design();
    let mut logslope_entry = base_logslope_entry_design();
    let mut logslope_rate = base_logslope_rate_design();
    match axis {
        PsiAxis::MarginalDesign => marginal.scaled_add(t, &marginal_design_derivative()),
        PsiAxis::LogSlopeDesign => {
            logslope_exit.scaled_add(t, &logslope_design_derivative());
            logslope_entry.scaled_add(t, &logslope_entry_design_derivative());
            logslope_rate.scaled_add(t, &logslope_rate_design_derivative());
        }
        PsiAxis::Baseline(_) => {}
    }

    let logslope_layout: LogslopeLayout = match frame {
        SlopeFrame::Static => (DesignMatrix::from(logslope_exit)).into(),
        SlopeFrame::FollowUpVarying => {
            let layout: LogslopeLayout = (DesignMatrix::from(logslope_exit)).into();
            layout
                .with_follow_up(
                    DesignMatrix::from(logslope_entry),
                    DesignMatrix::from(logslope_rate),
                )
                .expect("a shared log-slope layout accepts a follow-up margin")
        }
    };

    SurvivalMarginalSlopeFamily {
        n,
        event: Arc::new(event),
        weights: Arc::new(weights),
        z: Arc::new(z.insert_axis(Axis(1))),
        score_covariance: unit_score_covariance(),
        gaussian_frailty_sd: None,
        family_hyper: SurvivalMarginalSlopeFamilyHyperState::new(Some(Arc::clone(&geometry)), None)
            .expect("install the baseline chart as the family's hyper state"),
        derivative_guard: 1e-8,
        design_entry: DesignMatrix::from(Array2::zeros((n, 0))),
        design_exit: DesignMatrix::from(Array2::zeros((n, 0))),
        design_derivative_exit: DesignMatrix::from(Array2::zeros((n, 0))),
        offset_entry: Arc::new(geometry.offset_entry.clone()),
        offset_exit: Arc::new(geometry.offset_exit.clone()),
        derivative_offset_exit: Arc::new(geometry.derivative_offset_exit.clone()),
        marginal_design: DesignMatrix::from(marginal),
        logslope_layout,
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

/// Block states at fixed β for a given family.
///
/// `η` is a cached function of `(X, β)`, so it MUST be rebuilt at each
/// displacement: holding β fixed is the contract, holding a stale `η` fixed
/// would difference a different function from the one the ψ terms
/// differentiate.
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

/// One design-moving ψ axis, carrying the analytic `X_ψ` that `family_at`
/// realizes exactly.
fn derivative_blocks(axis: PsiAxis) -> Vec<Vec<CustomFamilyBlockPsiDerivative>> {
    let x_psi = match axis {
        PsiAxis::MarginalDesign => marginal_design_derivative(),
        PsiAxis::LogSlopeDesign => logslope_design_derivative(),
        PsiAxis::Baseline(_) => panic!("a baseline ψ axis owns no design derivative"),
    };
    let entry = vec![CustomFamilyBlockPsiDerivative::new(
        None,
        x_psi,
        Array2::zeros((2, 2)),
        None,
        None,
        None,
        None,
    )];
    match axis {
        PsiAxis::MarginalDesign => vec![Vec::new(), entry, Vec::new()],
        PsiAxis::LogSlopeDesign => vec![Vec::new(), Vec::new(), entry],
        PsiAxis::Baseline(_) => unreachable!(),
    }
}

fn hyper_layout(axis: PsiAxis) -> CustomFamilyHyperLayout {
    CustomFamilyHyperLayout::new(derivative_blocks(axis), Vec::new(), Array1::zeros(1))
        .expect("one design ψ axis")
}

/// The family's own `(objective, score, Hessian)` triple at displacement `t`.
///
/// All three come from the hooks the outer assembly reads, so a derivative that
/// matches these matches the function the criterion is actually built on.
fn objective_score_hessian(
    axis: PsiAxis,
    frame: SlopeFrame,
    t: f64,
) -> (f64, Array1<f64>, Array2<f64>) {
    let family = family_at(axis, frame, t);
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
    Ridders {
        value: (4.0 * fine - coarse) / 3.0,
        uncertainty: (fine - coarse).abs() / 3.0,
    }
}

struct PsiFiniteDifference {
    objective: Ridders,
    score: Vec<Ridders>,
    hessian: Vec<Ridders>,
    dim: usize,
}

fn finite_difference(axis: PsiAxis, frame: SlopeFrame, h: f64) -> PsiFiniteDifference {
    let coarse_plus = objective_score_hessian(axis, frame, h);
    let coarse_minus = objective_score_hessian(axis, frame, -h);
    let fine_plus = objective_score_hessian(axis, frame, 0.5 * h);
    let fine_minus = objective_score_hessian(axis, frame, -0.5 * h);

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

fn analytic_terms(axis: PsiAxis, frame: SlopeFrame) -> (f64, Array1<f64>, Array2<f64>) {
    let family = family_at(axis, frame, 0.0);
    let states = states_at(&family);
    let terms = match axis {
        PsiAxis::Baseline(index) => family
            .baseline_exact_joint_psi_terms_with_options(
                &states,
                index,
                &BlockwiseFitOptions::default(),
            )
            .expect("analytic baseline ψ terms")
            .expect("a nonlinear baseline chart publishes ψ terms"),
        design_axis => {
            let layout = hyper_layout(design_axis);
            family
                .psi_terms(&states, layout.design_derivative_blocks(), 0)
                .expect("analytic design ψ terms")
                .expect("a design ψ axis on a supported block publishes terms")
        }
    };
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
        gap <= 1e-5 * denominator + 4.0 * fd.uncertainty,
        "{label}: analytic={analytic:.9e} fd={:.9e} gap={gap:.3e} \
         rel={:.3e} oracle_uncertainty={:.3e}",
        fd.value,
        gap / denominator,
        fd.uncertainty,
    );
}

fn max_abs<'a>(values: impl Iterator<Item = &'a f64>) -> f64 {
    values.fold(0.0_f64, |acc, v| acc.max(v.abs()))
}

fn run_first_order_gate(axis: PsiAxis, frame: SlopeFrame) {
    let (objective_psi, score_psi, hessian_psi) = analytic_terms(axis, frame);
    let fd = finite_difference(axis, frame, 1e-3);
    let tag = format!("{axis:?}/{}", frame.label());
    assert_eq!(
        score_psi.len(),
        fd.dim,
        "{tag}: the ψ score must live in the flattened joint coefficient space"
    );
    assert_eq!(hessian_psi.dim(), (fd.dim, fd.dim), "{tag}: ψ Hessian shape");

    assert_matches(
        &format!("{tag} objective_psi"),
        objective_psi,
        &fd.objective,
        objective_psi.abs().max(fd.objective.value.abs()),
    );

    let score_scale = max_abs(score_psi.iter()).max(
        fd.score
            .iter()
            .fold(0.0_f64, |acc, r| acc.max(r.value.abs())),
    );
    for (i, oracle) in fd.score.iter().enumerate() {
        assert_matches(
            &format!("{tag} score_psi[{i}]"),
            score_psi[i],
            oracle,
            score_scale,
        );
    }

    let hessian_scale = max_abs(hessian_psi.iter()).max(
        fd.hessian
            .iter()
            .fold(0.0_f64, |acc, r| acc.max(r.value.abs())),
    );
    for (flat, oracle) in fd.hessian.iter().enumerate() {
        let (r, c) = (flat / fd.dim, flat % fd.dim);
        assert_matches(
            &format!("{tag} hessian_psi[{r},{c}]"),
            hessian_psi[[r, c]],
            oracle,
            hessian_scale,
        );
    }
}

// ── Design ψ: the matern/duchon length-scale lane ───────────────────────────

/// The marginal (population-index) design ψ axis in the four-primary frame.
#[test]
fn marginal_design_psi_terms_match_finite_difference_static_2765() {
    run_first_order_gate(PsiAxis::MarginalDesign, SlopeFrame::Static);
}

/// The log-slope design ψ axis in the four-primary frame. Filed separately from
/// the marginal axis because the two enter the row program through different
/// primary channels (`q₀,q₁` versus `g`), so one being right says nothing about
/// the other.
#[test]
fn logslope_design_psi_terms_match_finite_difference_static_2765() {
    run_first_order_gate(PsiAxis::LogSlopeDesign, SlopeFrame::Static);
}

/// The marginal design ψ axis with a follow-up-varying slope. The location
/// index is unchanged by the frame, but the pullback out of primary space is
/// not: `∂H/∂ψ` now has three slope channels to cross against.
#[test]
fn marginal_design_psi_terms_match_finite_difference_follow_up_2765() {
    run_first_order_gate(PsiAxis::MarginalDesign, SlopeFrame::FollowUpVarying);
}

/// A design ψ on the log-slope surface of a follow-up-varying slope must be
/// REFUSED, by name, rather than lowered through one of its three channels.
///
/// With a time margin the block's channel designs are `X_cov ⊗ B_entry`,
/// `X_cov ⊗ B_exit` and `X_cov ⊗ B′_exit`, while the ψ design-derivative
/// contract carries a single `X_ψ` — the other two channels are not recoverable
/// from what the caller holds, so any answer this could return would be the
/// derivative of a model nobody asked for. `fit_entry` already refuses the
/// combination at construction; this pins the same refusal one layer down, so a
/// future caller reaching the primary-space helpers directly cannot get silent
/// wrong math instead.
#[test]
fn a_logslope_design_psi_on_a_follow_up_varying_slope_is_refused_2765() {
    let family = family_at(PsiAxis::LogSlopeDesign, SlopeFrame::FollowUpVarying, 0.0);
    let states = states_at(&family);
    let layout = hyper_layout(PsiAxis::LogSlopeDesign);
    let error = family
        .psi_terms(&states, layout.design_derivative_blocks(), 0)
        .expect_err("a log-slope design ψ has no lowering into the follow-up frame");
    assert!(
        error.contains("follow-up-varying log-slope"),
        "the refusal must name the surface it refuses: {error}"
    );
}

// ── Baseline ψ: the chart-coordinate lane #2765's fit dies on ───────────────

/// The baseline chart's first coordinate in the four-primary frame.
///
/// This is the lane #2765's acceptance fit refuses in: its search direction is
/// essentially pure baseline-ψ, and fifty successively halved steps all fail
/// Armijo. A gradient that disagrees with the criterion reproduces exactly that
/// signature, and nothing between the chart partials (already FD-certified) and
/// the whole fit had ever measured it.
#[test]
fn baseline_psi_terms_match_finite_difference_static_axis0_2765() {
    run_first_order_gate(PsiAxis::Baseline(0), SlopeFrame::Static);
}

/// The baseline chart's second coordinate — the Gompertz shape, which enters
/// the hazard through an exponential rather than a scale factor.
#[test]
fn baseline_psi_terms_match_finite_difference_static_axis1_2765() {
    run_first_order_gate(PsiAxis::Baseline(1), SlopeFrame::Static);
}

/// The baseline chart's third coordinate — the Makeham constant.
#[test]
fn baseline_psi_terms_match_finite_difference_static_axis2_2765() {
    run_first_order_gate(PsiAxis::Baseline(2), SlopeFrame::Static);
}

/// The baseline chart in the six-primary frame.
///
/// The baseline moves only the location index's offsets, so its primary
/// direction is zero on every slope channel in either frame — but the objects
/// it produces (`∂_ψ∇_βℓ̄` and `∂_ψ∇²_βℓ̄`) are NOT, and they have to be pulled
/// back through all three slope channels rather than one.
#[test]
fn baseline_psi_terms_match_finite_difference_follow_up_axis0_2765() {
    run_first_order_gate(PsiAxis::Baseline(0), SlopeFrame::FollowUpVarying);
}

#[test]
fn baseline_psi_terms_match_finite_difference_follow_up_axis1_2765() {
    run_first_order_gate(PsiAxis::Baseline(1), SlopeFrame::FollowUpVarying);
}

#[test]
fn baseline_psi_terms_match_finite_difference_follow_up_axis2_2765() {
    run_first_order_gate(PsiAxis::Baseline(2), SlopeFrame::FollowUpVarying);
}

// ── The degenerate end of the same contract ─────────────────────────────────

/// A ψ axis whose design derivative is exactly zero must publish exactly zero
/// terms — the one case where the finite difference is exact rather than
/// certified, and the cheapest possible check that the ψ lane is reading its
/// own input.
#[test]
fn a_zero_psi_design_derivative_publishes_zero_terms_2765() {
    let family = family_at(PsiAxis::MarginalDesign, SlopeFrame::Static, 0.0);
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
