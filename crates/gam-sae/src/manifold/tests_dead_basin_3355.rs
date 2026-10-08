//! #3355/#2939 — the dead-atom basin: what the criterion prefers when an atom's
//! effective dimension collapses.
//!
//! Two run receipts on the sweep fixture
//! (`tests.rs::planted_circle_noise_scale_sweep…`, cell 1: softmax, n=40,
//! σ=0.02), one per base, raise one question neither answers:
//!
//! * At base `ae828f54c1` (2026-10-01) the outer search converged INTO a basin
//!   with log λ_smooth ≈ 21.3 and certified it: seed EV 0.9998, final EV
//!   −0.0000. The `[SAE-CRITERION]` decomposition of that terminal state read
//!   `V=−8.4809: loss=−47.394, extra=0.030, ½log|A|=102.811, rank_charge=8.2e-8,
//!   occam=63.928` against the healthy incumbent `V=35.3068: loss=−29.244,
//!   ½log|A|=68.287, rank_charge=3.7227, occam=7.4587`. The criterion VALUE
//!   preferred the dead atom by 43.8 nats (identity
//!   loss+extra+½log|A|+rank−occam = V verified to 3e-5 on both), through three
//!   channels: Δloss = −18.15, Δ(½log|A| net of the cancelling λS/occam λ
//!   blocks) = −21.94, Δrank_charge = −3.72. The three big log-λ terms cancel
//!   exactly (+90.90 on ½log|A| vs −90.90 on −occam; the #2933 F26
//!   split-invariance working as designed), so what remains REWARDS death.
//! * At main `aef82cace7` (2026-10-08) the same cell never reaches that basin:
//!   EFS descends the rank-1 stratum 129 iterations to V=35.293, is refused on
//!   inner-solve noise ("no resolvable contraction"), and BFGS pins at the
//!   #2939 rank wall — 37 rank-2 refusals at +3.6976 nats each (the analytic
//!   jump ½·edf·log N_eff with edf≈2.005, N_eff≈39.7) — dying NOT STATIONARY at
//!   |g|=6.56e-2. The maintainer's named wall, not the dead basin.
//!
//! This test evaluates the criterion at the dead ρ (and, for contrast, the
//! checkpoint ρ) with a cold-start inner solve on the same seeded term, at
//! whatever main it runs on, and puts the comparison on the record. The
//! measured fact at main `aef82cace7`: the cold-start criterion at the dead ρ
//! prices V=19.794 (loss=−47.847, ½log|A|=131.567, rank_charge=8.2e-8,
//! occam=63.927) — BELOW every value the main search's own log evaluated (its
//! best converged live value was V=35.288). The dead basin is not gone; the
//! search now dies at the rank wall before reaching it. A per-stratum
//! selection that ranks strata by criterion value would select the dead
//! stratum here, so the #3355 redesign should be judged against this
//! measurement, not only against the wall.
//!
//! What is asserted (structure, honest at any main):
//! * the dead ρ evaluates from a cold-start inner solve — the basin is
//!   reachable by a value probe, a fact any value-based selection layer needs;
//! * both evaluations are finite and their decompositions are published.
//!
//! What is deliberately NOT asserted:
//! * `V(dead) < V(healthy)`. Which state is "the" healthy incumbent is
//!   inner-state-dependent (the cold-start solve at the checkpoint ρ finds a
//!   different inner state than the search's warm incumbent: V=64.830 vs
//!   V=35.288 at the same ρ — itself a fact the redesign's value comparisons
//!   must confront). The preference question is the maintainer's
//!   model-selection question; this test publishes the numbers it needs.

use super::outer_objective::SaeManifoldOuterObjective;
use super::rho::SaeManifoldRho;
use super::tests::{PlantedCircleAssignmentMode, planted_circle_data, planted_circle_seed_term};
use ndarray::array;

/// One criterion evaluation at an explicit ρ on a fresh term seeded exactly as
/// the sweep test seeds it (cell 1: softmax, n=40, σ=0.02). Returns the value
/// plus the installed loss breakdown. `None` when the evaluation refuses —
/// a refusal is itself a fact this receipt records.
#[derive(Debug, Clone, Copy)]
struct CriterionAt {
    value: f64,
    loss_total: f64,
    data_fit: f64,
    assignment_sparsity: f64,
    smoothness: f64,
    ard: f64,
}

fn evaluate_at(
    log_lambda_sparse: f64,
    log_lambda_smooth: f64,
    log_ard: f64,
) -> Option<CriterionAt> {
    let z = planted_circle_data(40, 0.02);
    let (term, _seed_dispersion) =
        planted_circle_seed_term(z.view(), PlantedCircleAssignmentMode::Softmax);
    let init_rho = SaeManifoldRho::new(
        log_lambda_sparse,
        log_lambda_smooth,
        vec![array![log_ard]],
    )
    // The receipts' `final_rho` are stored in the fitted state's own
    // coordinates (the dispersion shift was already applied to the seed the
    // search started from), so bind the layout without re-applying it.
    .for_assignment(&term.assignment);
    let flat = init_rho.to_flat(&term.assignment).ok()?;
    let mut objective =
        SaeManifoldOuterObjective::new(term, z.clone(), None, init_rho, 50, 0.04, 1.0e-6, 1.0e-6);
    let (value, _beta) = objective
        .evaluate_authoritative_criterion(flat.view())
        .ok()?;
    let loss = objective.last_loss?;
    Some(CriterionAt {
        value,
        loss_total: loss.total(),
        data_fit: loss.data_fit,
        assignment_sparsity: loss.assignment_sparsity,
        smoothness: loss.smoothness,
        ard: loss.ard,
    })
}

#[test]
fn dead_basin_is_reachable_by_value_probe_and_its_preference_is_published_3355() {
    gam_runtime::test_support::install_diagnostic_logger();
    // The two ρ from the receipts, cell 1 (softmax, n=40, σ=0.02). The dead ρ
    // is the old base's certified terminal state; its ARD coordinate printed
    // 3.681 there. The healthy ρ is the main run's checkpoint neighbourhood
    // (2.507, −1.061); the seed-side value near (−3.91, −1) prices the alive
    // incumbent the searches actually descended.
    let dead = evaluate_at(-3.912, 21.309, 3.681);
    let healthy = evaluate_at(2.507, -1.061, 3.681);
    let (d, h) = match (dead, healthy) {
        (Some(d), Some(h)) => (d, h),
        (None, _) => panic!(
            "the dead rho (log sparse=-3.912, log smooth=21.309, log ard=3.7) must evaluate: \
             the dead basin's reachability by value probes is a fact any value-based \
             selection layer needs"
        ),
        (_, None) => panic!("the healthy rho must evaluate"),
    };
    eprintln!(
        "[#3355-dead-basin] dead    V={:.4} loss={:.4} (data_fit={:.4} sparsity={:.4} smooth={:.4} ard={:.4})",
        d.value, d.loss_total, d.data_fit, d.assignment_sparsity, d.smoothness, d.ard
    );
    eprintln!(
        "[#3355-dead-basin] healthy V={:.4} loss={:.4} (data_fit={:.4} sparsity={:.4} smooth={:.4} ard={:.4})",
        h.value, h.loss_total, h.data_fit, h.assignment_sparsity, h.smoothness, h.ard
    );
    eprintln!(
        "[#3355-dead-basin] preference: V(dead) - V(healthy) = {:+.4} nats (negative: the criterion prefers the dead atom at this main)",
        d.value - h.value
    );
    // Structural assertions only: both states price, and the decomposition is
    // on the record. No value preference is pinned — see the module doc.
    assert!(
        d.value.is_finite() && h.value.is_finite(),
        "both criterion values must be finite"
    );
}
