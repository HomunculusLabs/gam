//! #2629 — the mixture/SAS flexible-link objective and the soft ρ-guard
//! barrier.
//!
//! #2545 removed the `log cosh` barrier from the certificate's view at railed
//! coordinates, but only for the objective that PUBLISHES it
//! (`OuterObjective::soft_rho_guard_gradient`). The mixture/SAS objective is
//! built on the SAME `RemlState` and evaluates through
//! `RemlState::evaluate_unified_with_link_ext` → `assemble_and_evaluate` →
//! `build_prior`, so it carries the identical barrier — and #2629 records that
//! it publishes nothing.
//!
//! The issue asked FIRST whether that route is reachable with the barrier, i.e.
//! whether this is a live defect or a latent one. That is a measurement, not a
//! grep, and it is the measurement this module makes: evaluate the link-ext
//! criterion on the saturated ρ ladder and decompose its ρ-gradient against the
//! barrier's own closed form.
#![cfg(test)]

use super::*;
use crate::mixture_link::state_from_sasspec;
use gam_problem::{InverseLink, LikelihoodSpec, ResponseFamily, SasLinkSpec};
use ndarray::{Array1, Array2};

/// The saturated ρ ladder #2450/#2545 measured the λ=∞ face on. Three e-folds
/// apart, so a genuine `c·e^{−ρ}` tail shows a constant `c` across the rungs
/// and an instrument artifact does not.
const SATURATED_LADDER: [f64; 4] = [21.0, 24.0, 27.0, 30.0];

fn tiny_design(n: usize) -> Array2<f64> {
    let mut x = Array2::<f64>::zeros((n, 3));
    for i in 0..n {
        let t = (i as f64 + 0.5) / n as f64;
        let x1 = -1.5 + 3.0 * t;
        x[[i, 0]] = 1.0;
        x[[i, 1]] = x1;
        x[[i, 2]] = (2.1 * x1).sin();
    }
    x
}

/// A binomial fixture whose link is the learnable SAS pair `(ε, log δ)` — the
/// configuration that sends `optimizer.rs` down the `mixture/SAS flexible link`
/// branch with `θ = [ρ (k), ε, log δ]`.
fn sas_binomial_state<'a>(
    y: &'a Array1<f64>,
    w: &'a Array1<f64>,
    offset: &'a Array1<f64>,
    x: &Array2<f64>,
    cfg: &'a RemlConfig,
) -> crate::estimate::reml::RemlState<'a> {
    let p = x.ncols();
    let mut s = Array2::<f64>::zeros((p, p));
    for j in 1..p {
        s[[j, j]] = 1.0;
    }
    let (canonical_penalties, active_nullspace_dims) =
        gam_terms::construction::canonicalize_penalty_specs(
            &[PenaltySpec::Dense(s)],
            &[1],
            p,
            "#2629 mixture/SAS rho-guard fixture",
        )
        .expect("canonicalize the one-penalty fixture");
    let mut state = crate::estimate::reml::RemlState::newwith_offset(
        y.view(),
        x.clone(),
        w.view(),
        offset.view(),
        canonical_penalties,
        p,
        cfg,
        Some(active_nullspace_dims),
        None,
        None,
    )
    .expect("build the SAS-link REML state");
    state.set_link_states(None, cfg.link_kind.sas_state().copied());
    state
}

fn sas_binomial_config() -> RemlConfig {
    let sas_state = state_from_sasspec(SasLinkSpec {
        initial_epsilon: 0.0,
        initial_log_delta: 0.0,
    })
    .expect("canonical zero-seed SAS state");
    RemlConfig::external(
        gam_spec::GlmLikelihoodSpec::canonical(LikelihoodSpec::new(
            ResponseFamily::Binomial,
            InverseLink::Sas(sas_state),
        )),
        1e-10,
        false,
    )
}

fn binomial_response(n: usize) -> Array1<f64> {
    Array1::from_iter((0..n).map(|i| if (i * 7 + 3) % 5 < 2 { 1.0 } else { 0.0 }))
}

/// The `mixture/SAS flexible link` criterion carries the soft ρ-guard barrier,
/// and the barrier is what its upper rail's KKT residual is made of (#2629).
///
/// This is the "live or latent?" question the issue opens with, answered by
/// measurement. Three things are asserted, and together they are the same
/// signature #2545 measured on standard REML — on a different objective, a
/// different evaluator and a different fixture:
///
/// 1. **The evaluation really is the link-ext route.** The returned gradient is
///    θ-length (`k` ρ-coordinates plus the two SAS coordinates), not ρ-length,
///    so this is the branch `optimizer.rs` builds at the `mixture/SAS flexible
///    link` site and not the standard-REML one beside it.
/// 2. **The barrier dominates the ρ-gradient at the rail**, and the residual
///    under it is a genuine λ=∞ face tail: `c = residual·e^ρ` is constant
///    across the ladder. Measured at `f1de531b8`: `c = −22.8248, −22.8248,
///    −22.8246` at ρ = 21, 24, 27 — six significant figures across two
///    three-e-fold steps — with the barrier at `1.332439e-7` of a total
///    `1.332418e-7` at ρ=30. Constancy is the control that says the instrument
///    reads the criterion's own tail rather than its own noise.
/// 3. **The total is POSITIVE at the upper rail**, which is precisely the sign
///    `project_gradient_vector` retains there (`gi.max(0.0)`). That is what
///    turns "the criterion carries a barrier" into "this objective's λ=∞ face
///    can never certify": the face tail pulls INWARD (`c < 0`) and is discarded
///    by the projection, and what survives is the barrier's outward pull alone.
///
/// The assertions are stated RELATIVELY (a fraction of the barrier, a relative
/// spread in `c`) so they cannot be satisfied by the fixture shrinking.
#[test]
fn the_mixture_sas_criterion_carries_the_soft_rho_guard_barrier_at_its_upper_rail() {
    let n = 60usize;
    let x = tiny_design(n);
    let y = binomial_response(n);
    let w = Array1::<f64>::ones(n);
    let offset = Array1::<f64>::zeros(n);
    let cfg = sas_binomial_config();
    let state = sas_binomial_state(&y, &w, &offset, &x, &cfg);

    let a = crate::estimate::RHO_SOFT_PRIOR_SHARPNESS / crate::estimate::RHO_BOUND;
    let saturation = crate::estimate::RHO_SOFT_PRIOR_WEIGHT * a;
    let anchor = state.rho_weight_anchor();

    let mut tail_constants = Vec::new();
    let mut rail_row = None;
    for probe in SATURATED_LADDER {
        let rho = Array1::from_elem(1, probe);
        let evaluation = state
            .evaluate_unified_with_link_ext(
                &rho,
                crate::estimate::reml::reml_outer_engine::EvalMode::ValueAndGradient,
            )
            .expect("the SAS link-ext criterion must evaluate on the saturated ladder");
        let gradient = evaluation
            .gradient
            .expect("ValueAndGradient must return a gradient");
        assert_eq!(
            gradient.len(),
            rho.len() + 2,
            "this gate is only about the mixture/SAS route, whose gradient is \
             theta-length (rho plus the SAS (epsilon, log delta) pair); a \
             rho-length gradient means the fixture fell back to standard REML \
             and measures nothing about #2629"
        );

        // The barrier's own closed form at the WEIGHT-ANCHORED coordinate — the
        // same `rho_tilde = rho - rho_weight_anchor` the criterion evaluates it
        // at. Compared against the state's published emission just below, so a
        // formula/emission disagreement is caught here rather than assumed.
        let closed_form = saturation * (a * (probe - anchor)).tanh();
        let published = state.soft_rho_guard_gradient(&rho);
        assert_eq!(
            published.len(),
            rho.len(),
            "the state publishes ONE entry per rho coordinate; the theta-length \
             embedding is the objective seam's job, not the state's"
        );
        assert!(
            (published[0] - closed_form).abs() <= 1.0e-18,
            "rho={probe}: the state's barrier emission must be the anchored closed \
             form {closed_form:.12e}, got {:.12e}",
            published[0]
        );

        let residual = gradient[0] - published[0];
        tail_constants.push(residual * probe.exp());
        if probe == crate::estimate::RHO_BOUND {
            rail_row = Some((gradient[0], published[0], residual));
        }
    }

    let (rail_gradient, rail_barrier, rail_residual) =
        rail_row.expect("the ladder must include the box bound itself");

    // (2) The barrier is essentially all of it. Relative, so a smaller fixture
    // cannot satisfy this by having a smaller gradient.
    assert!(
        rail_residual.abs() <= 1.0e-3 * rail_barrier.abs(),
        "at rho=RHO_BOUND the soft guard must dominate the criterion's \
         rho-gradient (it is a saturating term over a decayed face tail): \
         gradient={rail_gradient:.6e} barrier={rail_barrier:.6e} \
         residual={rail_residual:.6e}"
    );

    // ...and what is under it obeys the face law. The rail rung is excluded:
    // at rho=30 the residual is ~2e-12 against a 1.3e-7 gradient, which is the
    // instrument's own cancellation floor rather than a reading of the tail.
    let interior: Vec<f64> = tail_constants[..SATURATED_LADDER.len() - 1].to_vec();
    let lo = interior.iter().cloned().fold(f64::INFINITY, f64::min);
    let hi = interior.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    assert!(
        lo.is_finite() && hi.is_finite() && lo.abs() > 0.0,
        "the tail constants must be finite and nonzero, got {interior:?}"
    );
    assert!(
        (hi - lo).abs() <= 1.0e-4 * lo.abs(),
        "c = residual*e^rho must be CONSTANT across the ladder — that constancy \
         is what says the residual under the barrier is the criterion's own \
         lambda=infinity face tail and not instrument noise. Got {interior:?}"
    );

    // (3) The sign. This is the whole mechanism: the face tail pulls inward and
    // the projection discards it, so the only thing left at the rail is the
    // barrier's outward pull — a standing `|Pg| >= w*a` no fit can clear.
    assert!(
        rail_gradient > 0.0,
        "the rho-gradient at an UPPER rail must be positive (that is the branch \
         `project_gradient_vector` retains via `gi.max(0.0)`), got \
         {rail_gradient:.6e}"
    );
    assert!(
        lo < 0.0,
        "the criterion's own face tail must pull INWARD here (c < 0), else this \
         fixture does not exhibit the mechanism #2629 is about: c={interior:?}"
    );
    let bounds = (
        Array1::from_elem(1, -crate::estimate::RHO_BOUND),
        Array1::from_elem(1, crate::estimate::RHO_BOUND),
    );
    let rail = Array1::from_elem(1, crate::estimate::RHO_BOUND);
    let unpublished = crate::rho_optimizer::project_gradient_vector(
        &rail,
        &Array1::from_elem(1, rail_gradient),
        Some(&bounds),
    );
    assert!(
        unpublished[0] >= saturation * 0.999,
        "with the barrier still in the certificate's view this rail carries a \
         |Pg| floor of w*a = {saturation:.6e}; measured {:.6e}",
        unpublished[0]
    );
}
