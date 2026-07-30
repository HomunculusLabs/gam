//! #2080 — the outer penalized quasi-Laplace `ρ` search must terminate in a bounded number of
//! criterion evaluations at wide output dimension (`p ≈ 96`), where the outer
//! line search overshoots into the adjacent indefinite (non-PD Laplace) basin on
//! nearly every probe.
//!
//! Before the fix each such infeasible PROBE ground the inner refinement budget
//! (the FD-safeguard value probes routed through the ACCEPTED `16×/64×
//! inner_max_iter` budget, and the non-PD arm of
//! `converge_inner_for_undamped_logdet` refined the probe up to that budget before
//! refusing) — so a single wide-`p` gradient point issued ~2·d_ρ full-width inner
//! solves, each grinding thousands of inner iterations: the wide-`p` hang. The fix
//! makes an infeasible-ρ PROBE return the typed refusal after one diagnostic
//! factor pass (`refine_progress_extension == false` fast-fails the non-PD arm),
//! runs the FD safeguard's value probes on the PROBE budget over a THROWAWAY clone
//! (so they never mutate the accepted basin), and gates the full 2·d_ρ FD
//! escalation on the inner-criterion width.
//!
//! This exercises the FULL outer `OuterProblem::run` ("SAE manifold") path — the
//! existing #2027 width test explicitly bypasses the outer ρ-search — and asserts
//! a PROBE-COUNT budget (per SPEC's ban on wall-clock budgets), zero mutating
//! value probes, and a materially positive reconstruction EV.

use super::tests::{deterministic_circle_noise, global_ev};
use super::*;
use crate::basis::{PeriodicHarmonicEvaluator, SaeBasisSecondJet};
use gam_linalg::faer_ndarray::{FaerCholesky, fast_atb};
use gam_solve::rho_optimizer::{OuterEval, OuterEvalOrder, OuterObjective, OuterProblem};
use ndarray::{Array1, Array2, ArrayView2, array, s};
use std::sync::Arc;

/// #2080 — two atoms may each have a full-rank decoder design while their
/// CONCATENATED design is rank-deficient. With identical weighted constant
/// columns, `δB₀ = c, δB₁ = −c` leaves the reconstruction exactly unchanged;
/// an atom-local `G_k + λS_k` audit sees two positive scalar Grams and misses
/// this coupled redistribution gauge entirely.
#[test]
fn joint_decoder_gauge_quotients_full_rank_atom_redistribution_2080() -> Result<(), String> {
    let n = 4usize;
    let phi = Array2::<f64>::ones((n, 1));
    let jet = ndarray::Array3::<f64>::zeros((n, 1, 1));
    let make_atom = |name: &str, decoder: f64| {
        SaeManifoldAtom::new_with_provided_function_gram(
            name,
            SaeAtomBasisKind::Linear,
            1,
            phi.clone(),
            jet.clone(),
            array![[decoder]],
            Array2::<f64>::zeros((1, 1)),
        )
    };
    let coords = Array2::<f64>::zeros((n, 1));
    let assignment = SaeAssignment::from_blocks_with_mode(
        Array2::<f64>::zeros((n, 2)),
        vec![coords.clone(), coords],
        AssignmentMode::softmax(1.0),
    )?;
    let term = SaeManifoldTerm::new(
        vec![make_atom("shared-a", 1.0)?, make_atom("shared-b", -0.5)?],
        assignment,
    )?;

    // Both atom-local weighted designs are individually rank one (their full
    // possible rank), so the superseded per-atom eigensolves have no null.
    let weights = term.assignment.assignments();
    for atom_idx in 0..2 {
        let gram = (0..n)
            .map(|row| {
                let value = weights[[row, atom_idx]] * phi[[row, 0]];
                value * value
            })
            .sum::<f64>();
        assert!(
            gram > 0.0,
            "atom {atom_idx} must have a full-rank scalar Gram"
        );
    }

    let gauges = term.joint_decoder_beta_null_directions(&[0.0, 0.0])?;
    assert_eq!(
        gauges.len(),
        1,
        "the two identical scalar designs have exactly one coupled decoder gauge"
    );
    let coord_dim = n * term.assignment.row_block_dim();
    let delta_t = Array1::<f64>::zeros(coord_dim);
    let delta_beta = array![1.0_f64, -1.0];
    let raw = delta_beta.dot(&delta_beta);
    let quotient =
        term.quotient_newton_step_norm_sq(delta_t.view(), delta_beta.view(), raw, &[0.0, 0.0])?;
    assert!(
        quotient <= f64::EPSILON * (1.0 + raw),
        "joint redistribution must vanish on the identified quotient; raw={raw:.3e}, quotient={quotient:.3e}"
    );
    Ok(())
}

/// Two planted circles on DISJOINT ambient column parities (circle A on the even
/// output channels, circle B on the odd), driven by two independent phases on an
/// exact Cartesian product grid and per-column standardized. Together they span a
/// rank-4 subspace of the whitened `p`-dim cloud, so an honest K=2 dictionary
/// explains a materially positive fraction of the variance. `p` is the wide-`p`
/// knob that drives the outer hang.
pub(super) fn independent_two_circle_phases(n: usize, row: usize) -> (f64, f64) {
    let mut n1 = 1usize;
    let root = (n as f64).sqrt() as usize;
    for d in 1..=root.max(1) {
        if n % d == 0 {
            n1 = d;
        }
    }
    let n2 = n / n1.max(1);
    assert!(
        n1 > 1 && n2 > 1,
        "two-circle fixture needs a nontrivial Cartesian phase grid, got {n1}x{n2}"
    );
    let i = row % n1;
    let j = (row / n1) % n2;
    (
        std::f64::consts::TAU * (i as f64) / (n1 as f64),
        std::f64::consts::TAU * (j as f64) / (n2 as f64),
    )
}

fn two_circle_wide_target(n: usize, p: usize, sigma: f64) -> Array2<f64> {
    let mut fa = Array2::<f64>::zeros((2, p));
    let mut fb = Array2::<f64>::zeros((2, p));
    for j in 0..p {
        if j % 2 == 0 {
            fa[[0, j]] = deterministic_circle_noise(j, 0);
            fa[[1, j]] = deterministic_circle_noise(j, 1);
        } else {
            fb[[0, j]] = deterministic_circle_noise(j, 2);
            fb[[1, j]] = deterministic_circle_noise(j, 3);
        }
    }
    for f in [&mut fa, &mut fb] {
        for r in 0..2 {
            let nrm = (0..p).map(|j| f[[r, j]] * f[[r, j]]).sum::<f64>().sqrt();
            for j in 0..p {
                f[[r, j]] /= nrm.max(1.0e-300);
            }
        }
    }
    let mut z = Array2::<f64>::zeros((n, p));
    for row in 0..n {
        let (ta, tb) = independent_two_circle_phases(n, row);
        let (ca, sa) = (ta.cos(), ta.sin());
        let (cb, sb) = (tb.cos(), tb.sin());
        for j in 0..p {
            z[[row, j]] = ca * fa[[0, j]]
                + sa * fa[[1, j]]
                + cb * fb[[0, j]]
                + sb * fb[[1, j]]
                + sigma * deterministic_circle_noise(row, j + 7);
        }
    }
    for j in 0..p {
        let mut mean = 0.0_f64;
        for row in 0..n {
            mean += z[[row, j]];
        }
        mean /= n as f64;
        let mut var = 0.0_f64;
        for row in 0..n {
            let d = z[[row, j]] - mean;
            var += d * d;
        }
        let sd = (var / n as f64).sqrt().max(1.0e-12);
        for row in 0..n {
            z[[row, j]] = (z[[row, j]] - mean) / sd;
        }
    }
    z
}

/// A single centered circle embedded in `p` standardized ambient channels. This
/// is the cheap K=1 #2153 regression target; unlike the two-circle fixture, the
/// model is correctly specified, so any long Strong-Wolfe probe train is solver
/// pathology rather than target mismatch.
pub(super) fn one_circle_wide_target(n: usize, p: usize, sigma: f64) -> Array2<f64> {
    let mut frame = Array2::<f64>::zeros((2, p));
    for j in 0..p {
        frame[[0, j]] = deterministic_circle_noise(j, 0);
        frame[[1, j]] = deterministic_circle_noise(j, 1);
    }
    for r in 0..2 {
        let nrm = (0..p)
            .map(|j| frame[[r, j]] * frame[[r, j]])
            .sum::<f64>()
            .sqrt();
        for j in 0..p {
            frame[[r, j]] /= nrm.max(1.0e-300);
        }
    }
    let mut z = Array2::<f64>::zeros((n, p));
    for row in 0..n {
        let t = std::f64::consts::TAU * (row as f64) / (n as f64);
        let (c, s) = (t.cos(), t.sin());
        for j in 0..p {
            z[[row, j]] = c * frame[[0, j]]
                + s * frame[[1, j]]
                + sigma * deterministic_circle_noise(row, j + 7);
        }
    }
    for j in 0..p {
        let mut mean = 0.0_f64;
        for row in 0..n {
            mean += z[[row, j]];
        }
        mean /= n as f64;
        let mut var = 0.0_f64;
        for row in 0..n {
            let d = z[[row, j]] - mean;
            var += d * d;
        }
        let sd = (var / n as f64).sqrt().max(1.0e-12);
        for row in 0..n {
            z[[row, j]] = (z[[row, j]] - mean) / sd;
        }
    }
    z
}

/// Build a K-atom, d=1 periodic SAE term seeded the way the production cold path
/// does (PCA-seed the per-atom coordinates, ridge-LSQ each per-atom decoder), with
/// ordered Beta--Bernoulli assignment. Returns the term and the seed reconstruction dispersion the
/// outer cascade scales its ρ seed by. `harmonics` sets the basis size `m = 1 +
/// 2·harmonics`.
pub(super) fn two_circle_periodic_term(
    z: ArrayView2<'_, f64>,
    k: usize,
    harmonics: usize,
) -> (SaeManifoldTerm, f64) {
    let n = z.nrows();
    let p = z.ncols();
    let dim = 1usize;
    let num_basis = 1 + 2 * harmonics;
    let evaluator: Arc<dyn SaeBasisSecondJet> =
        Arc::new(
        PeriodicHarmonicEvaluator::new(num_basis)
            .expect("num_basis = 1 + 2*harmonics is a valid odd periodic basis width"),
    );
    let basis_kinds = vec![SaeAtomBasisKind::Periodic; k];
    let atom_dims = vec![dim; k];
    let seed_coords = sae_pca_seed_initial_coords(z, &basis_kinds, &atom_dims)
        .expect("one basis kind and one dim per atom, matching the fixture target");
    let mut atoms = Vec::with_capacity(k);
    let mut coords_blocks = Vec::with_capacity(k);
    let mut manifolds = Vec::with_capacity(k);
    let mut rss = 0.0_f64;
    for atom_idx in 0..k {
        let coords = seed_coords.slice(s![atom_idx, .., 0..dim]).to_owned();
        let (phi, jet) = evaluator
        .evaluate(coords.view())
        .expect("the seeded coords lie in the periodic chart domain");
        let mm = phi.ncols();
        let mut xtx = fast_atb(&phi, &phi);
        for i in 0..mm {
            xtx[[i, i]] += 1.0e-8;
        }
        let xtz = fast_atb(&phi, &z.to_owned());
        let decoder = xtx
        .cholesky(Side::Lower)
        .expect("phi^T phi + 1e-8 I is positive definite")
        .solve_mat(&xtz);
        let fitted = phi.dot(&decoder);
        for row in 0..n {
            for col in 0..p {
                let r = z[[row, col]] - fitted[[row, col]];
                rss += r * r;
            }
        }
        let atom = SaeManifoldAtom::new_with_provided_function_gram(
            "circle",
            SaeAtomBasisKind::Periodic,
            dim,
            phi,
            jet,
            decoder,
            Array2::<f64>::eye(mm),
        )
        .expect("phi, jet, decoder and gram were built with matching shapes")
        .with_basis_evaluator(evaluator.clone());
        atoms.push(atom);
        coords_blocks.push(coords);
        manifolds.push(LatentManifold::Circle { period: 1.0 });
    }
    let seed_dispersion = (rss / (k * n * p) as f64).max(1.0e-12);
    let logits = Array2::<f64>::from_elem((n, k), 6.0);
    let mode = AssignmentMode::ordered_beta_bernoulli(1.0, 1.0, false);
    let assignment =
        SaeAssignment::from_blocks_with_mode_and_manifolds(logits, coords_blocks, manifolds, mode)
            .expect("one logit column, coord block and manifold per atom");
    (
        SaeManifoldTerm::new(atoms, assignment)
            .expect("atoms and assignment were built over the same k atoms"),
        seed_dispersion,
    )
}

/// #2080 — a reactive legal entry must replace a nonzero but invalid cold
/// dictionary with a separated data-derived basin before asking for evidence.
/// The old zero-decoder gate skipped this production-shaped seed because both
/// independently fitted decoders had material norm; the fixed-ρ entry then spent
/// its whole refinement budget unwinding their double fit and never reached KKT.
#[test]
fn reactive_entry_reseeds_nonzero_k2_seed_to_strict_separated_root_2080() {
    let z = two_circle_wide_target(48, 24, 0.03);
    let k = 2usize;
    let (term, seed_dispersion) = two_circle_periodic_term(z.view(), k, 2);
    let seed_norms: Vec<f64> = term
        .atoms
        .iter()
        .map(|atom| {
            atom.decoder_coefficients()
                .iter()
                .map(|value| value * value)
                .sum::<f64>()
                .sqrt()
        })
        .collect();
    assert!(
        seed_norms
            .iter()
            .all(|norm| norm.is_finite() && *norm > 0.0),
        "regression requires the nonzero decoder seed that bypassed the old cold-entry placement; norms={seed_norms:?}"
    );

    let mode = AssignmentMode::ordered_beta_bernoulli(1.0, 1.0, false);
    let init_rho = SaeManifoldRho::new(0.02_f64.ln(), 1.0_f64.ln(), vec![array![0.0]; k])
        .seed_scaled_by_dispersion_for_assignment(seed_dispersion, mode)
        .expect("seed dispersion is finite and strictly positive");
    let mut objective =
        SaeManifoldOuterObjective::new(term, z.clone(), None, init_rho, 8, 0.04, 1.0e-6, 1.0e-6);
    let contract = OuterObjective::reactive_domain_scalar_contract(&objective)
        .expect("reactive scalar contract query")
        .expect("dense K=2 objective must own a reactive scalar contract");
    let entry_rho = OuterObjective::outer_domain_upper_bound(&objective)
        .expect("reactive rho entry query")
        .expect("dense K=2 objective must own a reactive rho entry");

    OuterObjective::begin_reactive_domain_waypoint(&mut objective)
        .expect("entry transaction must begin");
    OuterObjective::install_reactive_domain_scalar_state(&mut objective, contract.entry())
        .expect("separated legal entry must install");

    // The target's two factors occupy disjoint output parities. The objective-
    // owned entry placement may swap atom labels, but the two decoders must
    // specialize to opposite parities instead of both carrying the full target.
    let mut even_dominant = Vec::with_capacity(k);
    for atom in &objective.term.atoms {
        let mut even_energy = 0.0_f64;
        let mut odd_energy = 0.0_f64;
        for ((_, output), value) in atom.decoder_coefficients().indexed_iter() {
            if output % 2 == 0 {
                even_energy += value * value;
            } else {
                odd_energy += value * value;
            }
        }
        assert!(
            even_energy.is_finite()
                && odd_energy.is_finite()
                && (even_energy > 0.0 || odd_energy > 0.0)
                && even_energy != odd_energy,
            "entry decoder must carry a finite, parity-identifiable factor; even={even_energy:.6e}, odd={odd_energy:.6e}"
        );
        even_dominant.push(even_energy > odd_energy);
    }
    assert_ne!(
        even_dominant[0], even_dominant[1],
        "entry placement must put the two planted factors on distinct atoms; parity dominance={even_dominant:?}"
    );

    let entry_eval_result =
        OuterObjective::eval_with_order(&mut objective, &entry_rho, OuterEvalOrder::Value);
    if let Err(error) = &entry_eval_result {
        let entry_rho_state = objective
            .baseline_rho
            .from_flat(entry_rho.view())
            .expect("entry_rho came from `to_flat` on this same rho layout");
        let system = objective
            .term
            .assemble_arrow_schur(z.view(), &entry_rho_state, None)
            .expect("failed-entry KKT diagnostic assembly");
        let assignment_dim = objective.term.assignment.assignment_coord_dim();
        let mut assignment_grad_sq = 0.0_f64;
        let mut chart_grad_sq = 0.0_f64;
        for row in &system.rows {
            for (index, value) in row.gt.iter().enumerate() {
                if index < assignment_dim {
                    assignment_grad_sq += value * value;
                } else {
                    chart_grad_sq += value * value;
                }
            }
        }
        let decoder_grad = system
            .gb
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt();
        let assignments = objective
            .term
            .assignment
            .try_assignments()
            .expect("failed-entry assignment diagnostic");
        let (assignment_min, assignment_max) = assignments.iter().fold(
            (f64::INFINITY, f64::NEG_INFINITY),
            |(minimum, maximum), value| (minimum.min(*value), maximum.max(*value)),
        );
        let decoder_norms: Vec<f64> = objective
            .term
            .atoms
            .iter()
            .map(|atom| {
                atom.decoder_coefficients()
                    .iter()
                    .map(|value| value * value)
                    .sum::<f64>()
                    .sqrt()
            })
            .collect();
        eprintln!(
            "[#2080 reactive-entry KKT] error={error}; assignment_grad={:.6e}, \
             chart_grad={:.6e}, decoder_grad={decoder_grad:.6e}, \
             assignment_range=[{assignment_min:.6e},{assignment_max:.6e}], \
             decoder_norms={decoder_norms:?}, lambda_smooth={:?}",
            assignment_grad_sq.sqrt(),
            chart_grad_sq.sqrt(),
            entry_rho_state
                .lambda_smooth_vec()
                .expect("the fixture rho carries one smoothing block per atom"),
        );
    }
    let entry_eval =
        entry_eval_result.expect("separated legal entry must solve to finite evidence");
    assert!(
        entry_eval.cost.is_finite(),
        "separated legal entry returned non-finite evidence {}",
        entry_eval.cost
    );
    OuterObjective::commit_reactive_domain_waypoint(&mut objective, &entry_rho)
        .expect("finite entry must commit its full converged state");

    // Reassemble the exact committed entry and independently recheck the same
    // strict raw-or-quotient KKT predicate that authorizes penalized quasi-Laplace score. A
    // finite value alone cannot satisfy this regression.
    let committed_rho = objective.current_rho.clone();
    let system = objective
        .term
        .assemble_arrow_schur(z.view(), &committed_rho, None)
        .expect("committed entry KKT assembly");
    let raw_kkt_sq = SaeManifoldTerm::system_grad_norm_sq(&system);
    let raw_kkt = raw_kkt_sq.sqrt();
    let quotient_kkt = objective.term.quotient_gradient_norm_from_system(
        &system,
        raw_kkt_sq,
        &committed_rho
            .lambda_smooth_vec()
            .expect("the fixture rho carries one smoothing block per atom"),
    );
    let tolerance = SAE_MANIFOLD_INNER_GRAD_REL_TOL * objective.term.inner_iterate_scale();
    assert!(
        SaeManifoldTerm::quasi_laplace_kkt_stationary(raw_kkt, quotient_kkt, tolerance),
        "committed legal entry is not a strict envelope root: raw KKT={raw_kkt:.6e}, quotient KKT={quotient_kkt:.6e}, tolerance={tolerance:.6e}"
    );
}

/// Drive the full outer `OuterProblem::run` path on a wide two-circle fixture and
/// return `(reconstruction EV, probe telemetry)`.
fn run_wide_outer_fit(
    n: usize,
    p: usize,
    k: usize,
    harmonics: usize,
) -> (f64, OuterProbeTelemetry) {
    let z = two_circle_wide_target(n, p, 0.03);
    let (term, seed_dispersion) = two_circle_periodic_term(z.view(), k, harmonics);
    let mode = AssignmentMode::ordered_beta_bernoulli(1.0, 1.0, false);
    let init_rho = SaeManifoldRho::new(0.02_f64.ln(), 1.0_f64.ln(), vec![array![0.0]; k])
        .seed_scaled_by_dispersion_for_assignment(seed_dispersion, mode)
        .expect("seed dispersion is finite and strictly positive");
    let seed = init_rho.to_flat();
    let n_params = seed.len();
    let mut objective =
        SaeManifoldOuterObjective::new(term, z.clone(), None, init_rho, 8, 0.04, 1.0e-6, 1.0e-6);
    let result = OuterProblem::new(n_params)
        .with_initial_rho(seed)
        .with_max_iter(4)
        .with_seed_config(gam_problem::SeedConfig {
            max_seeds: 1,
            seed_budget: 1,
            ..Default::default()
        })
        .run(&mut objective, "SAE manifold")
        .expect("#2080 wide-p outer penalized quasi-Laplace fit must terminate, not hang / abort");
    assert!(
        result.converged(),
        "#2080 wide-p acceptance requires a CONVERGED outer penalized quasi-Laplace optimum, not a \
         finite max-iteration/line-search incumbent: iterations={}, final_value={:.6e}, \
         final_grad_norm={:?}",
        result.iterations,
        result.final_value,
        result.final_grad_norm,
    );
    let telemetry = objective.probe_telemetry();
    objective
        .certify_outer_result(&result)
        .expect("#2080 wide-p outer result must certify the installed state");
    let fitted = objective.into_fitted().expect("outer fit was evaluated");
    let ev = global_ev(z.view(), fitted.term.fitted().view());
    (ev, telemetry)
}

/// Same full outer path as `run_wide_outer_fit`, but intentionally starts from
/// the generated seed rather than pinning `initial_rho`. This is the K=1 cold
/// path that #2153 exposed: with the optimizer's raw identity iter-0 metric, the
/// first `-g` step can be orders too large, so Strong-Wolfe spends full
/// value-probe solves backtracking instead of making progress.
fn run_k1_generated_seed_outer_fit(
    n: usize,
    p: usize,
    harmonics: usize,
) -> (f64, OuterProbeTelemetry) {
    let z = one_circle_wide_target(n, p, 0.05);
    let (term, seed_dispersion) = two_circle_periodic_term(z.view(), 1, harmonics);
    let mode = AssignmentMode::ordered_beta_bernoulli(1.0, 1.0, false);
    let init_rho = SaeManifoldRho::new(0.02_f64.ln(), 1.0_f64.ln(), vec![array![0.0]])
        .seed_scaled_by_dispersion_for_assignment(seed_dispersion, mode)
        .expect("seed dispersion is finite and strictly positive");
    let n_params = init_rho.to_flat().len();
    let mut objective =
        SaeManifoldOuterObjective::new(term, z.clone(), None, init_rho, 8, 0.04, 1.0e-6, 1.0e-6);
    let result = OuterProblem::new(n_params)
        .with_max_iter(4)
        .with_seed_config(gam_problem::SeedConfig {
            max_seeds: 1,
            seed_budget: 1,
            ..Default::default()
        })
        .run(&mut objective, "SAE manifold K=1 generated seed")
        .expect("#2153 K=1 generated-seed circle fit must terminate");
    assert!(
        result.converged(),
        "#2153 K=1 acceptance requires a converged outer optimum: iterations={}, \
         final_value={:.6e}, final_grad_norm={:?}",
        result.iterations,
        result.final_value,
        result.final_grad_norm,
    );
    let telemetry = objective.probe_telemetry();
    objective
        .certify_outer_result(&result)
        .expect("#2153 outer result must certify the installed state");
    let fitted = objective.into_fitted().expect("outer fit was evaluated");
    let ev = global_ev(z.view(), fitted.term.fitted().view());
    (ev, telemetry)
}

#[derive(Clone, Copy, Debug)]
struct CeilingPathologyConfig {
    n: usize,
    p: usize,
    harmonics: usize,
    sigma: f64,
    inner_max_iter: usize,
    outer_max_iter: usize,
    initial_step_norm: f64,
    materialization_ratio_floor: f64,
    step_collapse_radius: f64,
    huge_final_gradient_floor: f64,
    pin_initial_rho: bool,
}

impl Default for CeilingPathologyConfig {
    fn default() -> Self {
        Self {
            n: 96,
            p: 96,
            harmonics: 2,
            sigma: 0.05,
            inner_max_iter: 8,
            outer_max_iter: 8,
            initial_step_norm: 0.25,
            materialization_ratio_floor: 0.05,
            step_collapse_radius: 1.0e-3,
            huge_final_gradient_floor: 10.0,
            pin_initial_rho: true,
        }
    }
}

#[derive(Clone, Debug)]
struct CeilingPathologyReport {
    initial_cost: f64,
    initial_grad_norm: f64,
    predicted_decrease: f64,
    actual_decrease: f64,
    materialization_ratio: f64,
    outer_converged: bool,
    outer_iterations: usize,
    final_value: f64,
    final_grad_norm: f64,
    rho_displacement: f64,
    ev: f64,
    telemetry: OuterProbeTelemetry,
    outer_error: Option<String>,
    predicted_decrease_not_materializing: bool,
    step_collapsed: bool,
    huge_final_gradient: bool,
    live_lock_present: bool,
}

fn l2_norm(v: &Array1<f64>) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

fn seeded_k1_circle_objective(
    cfg: CeilingPathologyConfig,
) -> (
    Array2<f64>,
    SaeManifoldRho,
    Array1<f64>,
    SaeManifoldOuterObjective,
) {
    let z = one_circle_wide_target(cfg.n, cfg.p, cfg.sigma);
    let (term, seed_dispersion) = two_circle_periodic_term(z.view(), 1, cfg.harmonics);
    let mode = AssignmentMode::ordered_beta_bernoulli(1.0, 1.0, false);
    let init_rho = SaeManifoldRho::new(0.02_f64.ln(), 1.0_f64.ln(), vec![array![0.0]])
        .seed_scaled_by_dispersion_for_assignment(seed_dispersion, mode)
        .expect("seed dispersion is finite and strictly positive");
    let seed = init_rho.to_flat();
    let objective = SaeManifoldOuterObjective::new(
        term,
        z.clone(),
        None,
        init_rho.clone(),
        cfg.inner_max_iter,
        0.04,
        1.0e-6,
        1.0e-6,
    );
    (z, init_rho, seed, objective)
}

/// CEILING-VS-PATHOLOGY decisive instrument (#2156): run the mid-scale curved
/// OUTER-REML fit and classify the line-search live-lock signature.
///
/// The first-step model check is intentionally independent of the optimizer's
/// private line-search trace: it measures whether the initial analytic descent
/// direction predicts a material REML decrease that the value path does not
/// realize. The full outer run then reports the operational symptoms that make
/// this a solver pathology rather than an information ceiling: collapsed
/// accepted ρ displacement and a large final gradient.
fn run_ceiling_vs_pathology_instrument(cfg: CeilingPathologyConfig) -> CeilingPathologyReport {
    let probe_seeded = seeded_k1_circle_objective(cfg);
    let seed_probe = probe_seeded.2;
    let mut probe_objective = probe_seeded.3;
    let initial = OuterObjective::eval(&mut probe_objective, &seed_probe)
        .expect("#2156 instrument initial REML gradient eval must complete");
    let initial_grad_norm = l2_norm(&initial.gradient);
    let mut trial = seed_probe.clone();
    if initial_grad_norm.is_finite() && initial_grad_norm > 0.0 {
        let scale = cfg.initial_step_norm / initial_grad_norm;
        for idx in 0..trial.len() {
            trial[idx] -= scale * initial.gradient[idx];
        }
    }
    let predicted_decrease = if initial_grad_norm.is_finite() {
        cfg.initial_step_norm * initial_grad_norm
    } else {
        f64::NAN
    };
    let trial_cost = OuterObjective::eval_cost(&mut probe_objective, &trial)
        .expect("#2156 instrument initial penalized quasi-Laplace value probe must complete");
    let actual_decrease = initial.cost - trial_cost;
    let materialization_ratio = if predicted_decrease.is_finite()
        && predicted_decrease > f64::MIN_POSITIVE
        && actual_decrease.is_finite()
    {
        actual_decrease / predicted_decrease
    } else {
        f64::NAN
    };
    let predicted_decrease_not_materializing = materialization_ratio.is_finite()
        && materialization_ratio < cfg.materialization_ratio_floor;

    let fit_seeded = seeded_k1_circle_objective(cfg);
    let z = fit_seeded.0;
    let seed = fit_seeded.2;
    let mut objective = fit_seeded.3;
    let n_params = seed.len();
    let mut problem = OuterProblem::new(n_params)
        .with_max_iter(cfg.outer_max_iter)
        .with_seed_config(gam_problem::SeedConfig {
            max_seeds: 1,
            seed_budget: 1,
            ..Default::default()
        });
    if cfg.pin_initial_rho {
        problem = problem.with_initial_rho(seed.clone());
    }
    let run = problem.run(&mut objective, "SAE manifold ceiling-vs-pathology #2156");
    let telemetry = objective.probe_telemetry();
    match run {
        Ok(result) => {
            let final_grad_norm = result.final_grad_norm.unwrap_or(f64::NAN);
            let rho_displacement = l2_norm(&(&result.rho - &seed));
            let step_collapsed =
                rho_displacement.is_finite() && rho_displacement <= cfg.step_collapse_radius;
            let huge_final_gradient =
                final_grad_norm.is_finite() && final_grad_norm >= cfg.huge_final_gradient_floor;
            objective
                .certify_outer_result(&result)
                .expect("ceiling-pathology outer result must certify the installed state");
            let fitted = objective.into_fitted().expect("outer fit was evaluated");
            let ev = global_ev(z.view(), fitted.term.fitted().view());
            let live_lock_present =
                predicted_decrease_not_materializing && step_collapsed && huge_final_gradient;
            CeilingPathologyReport {
                initial_cost: initial.cost,
                initial_grad_norm,
                predicted_decrease,
                actual_decrease,
                materialization_ratio,
                outer_converged: result.converged(),
                outer_iterations: result.iterations,
                final_value: result.final_value,
                final_grad_norm,
                rho_displacement,
                ev,
                telemetry,
                outer_error: None,
                predicted_decrease_not_materializing,
                step_collapsed,
                huge_final_gradient,
                live_lock_present,
            }
        }
        Err(err) => CeilingPathologyReport {
            initial_cost: initial.cost,
            initial_grad_norm,
            predicted_decrease,
            actual_decrease,
            materialization_ratio,
            outer_converged: false,
            outer_iterations: 0,
            final_value: f64::NAN,
            final_grad_norm: f64::NAN,
            rho_displacement: f64::NAN,
            ev: f64::NAN,
            telemetry,
            outer_error: Some(err.to_string()),
            predicted_decrease_not_materializing,
            step_collapsed: false,
            huge_final_gradient: false,
            live_lock_present: true,
        },
    }
}

/// #2080 — the wide-`p` (p=96) K=2 outer penalized quasi-Laplace fit must terminate in a bounded
/// number of criterion evaluations and recover a materially positive EV — even
/// though the outer line search overshoots into the non-PD basin on many probes.
#[test]
fn wide_p_outer_reml_terminates_within_probe_budget_2080() {
    let n = 96usize;
    let p = 96usize;
    let k = 2usize;
    let harmonics = 2usize; // m = 5: [1, sin2πt, cos2πt, sin4πt, cos4πt]
    let (ev, telemetry) = run_wide_outer_fit(n, p, k, harmonics);
    eprintln!(
        "[#2080] wide-p outer fit: ev={ev:.4}, criterion_calls={}, \
         infeasible(non_pd_per_row={},schur={},inner_nc={}), \
         infeasible_criterion_evals={}, reactive_scalar_installs={}, \
         reactive_target_restores={}",
        telemetry.criterion_calls,
        telemetry.infeasible_non_pd_per_row,
        telemetry.infeasible_schur,
        telemetry.infeasible_inner_not_converged,
        telemetry.infeasible_criterion_evals,
        telemetry.reactive_scalar_installs,
        telemetry.reactive_target_restores,
    );
    assert!(
        telemetry.reactive_scalar_installs > 0,
        "the initially undefined wide-K=2 seed must traverse genuine objective-installed scalar waypoints"
    );
    assert!(
        telemetry.reactive_target_restores > 0,
        "the wide-K=2 continuation must restore the objective's literal scalar target before certification"
    );
    // Bounded criterion (eval / eval_cost / efs) budget — a PROBE COUNT, not a
    // wall-clock limit (SPEC bans time budgets). With `with_max_iter(4)` and a
    // single seed the outer loop cannot issue an unbounded number of full
    // criterion evals; the pre-fix hang was UNBOUNDED inner work PER probe, not an
    // unbounded probe count, so this asserts the complementary invariant.
    assert!(
        telemetry.criterion_calls <= 64,
        "outer penalized quasi-Laplace issued {} criterion calls; expected a bounded (<= 64) probe budget",
        telemetry.criterion_calls
    );
    assert!(
        ev.is_finite() && ev > 0.20,
        "wide-p K=2 two-circle outer fit must recover a materially positive EV \
         (got {ev:.4}); two disjoint circles span a rank-4 subspace an honest K=2 \
         dictionary recovers"
    );
}

/// #2153 — K=1 manifold fits must not live-lock in Strong-Wolfe line search from
/// a cold generated seed. The regression is a probe-count assertion, not a
/// wall-clock deadline: the first BFGS step is normalized by the seed gradient
/// norm, so the line search should accept a bounded step instead of spending
/// repeated full inner solves on rejected value probes.
#[test]
fn k1_generated_seed_circle_outer_reml_does_not_livelock_2153() {
    let (ev, telemetry) = run_k1_generated_seed_outer_fit(32, 24, 1);
    eprintln!(
        "[#2153] K=1 generated-seed outer fit: ev={ev:.4}, criterion_calls={}, \
         infeasible_criterion_evals={}, infeasible_total={}",
        telemetry.criterion_calls,
        telemetry.infeasible_criterion_evals,
        telemetry.infeasible_total(),
    );
    assert!(
        telemetry.criterion_calls <= 32,
        "#2153 K=1 generated-seed fit issued {} criterion calls; expected a \
         bounded first-line-search probe budget",
        telemetry.criterion_calls
    );
    assert!(
        ev.is_finite() && ev > 0.30,
        "#2153 K=1 generated-seed circle fit must converge to a real positive-EV \
         basin (got {ev:.4})"
    );
}

/// Expert Test 1+2 / #2156 — decide "solver pathology" versus "real ceiling".
///
/// Before the adjoint-gradient fix, this fixture reports the combined pathology:
/// the analytic descent model predicts a material decrease, the value path does
/// not realize it, the accepted outer iterate barely moves, and the final
/// gradient remains large. After the fix the same source-level instrument should
/// report `live_lock_present=false`; the top-M envelope report then tells the
/// reader whether any remaining low curved EV is an information ceiling.
#[test]
fn ceiling_vs_pathology_outer_reml_instrument_2156() {
    let report = run_ceiling_vs_pathology_instrument(CeilingPathologyConfig::default());
    eprintln!(
        "[#2156 ceiling-vs-pathology] initial_cost={:.6e}, initial_grad_norm={:.6e}, \
         predicted_decrease={:.6e}, actual_decrease={:.6e}, materialization_ratio={:.6e}, \
         outer_converged={}, outer_iterations={}, final_value={:.6e}, final_grad_norm={:.6e}, \
         rho_displacement={:.6e}, ev={:.4}, criterion_calls={}, \
         infeasible_criterion_evals={}, infeasible_total={}, outer_error={:?}, \
         predicted_not_materializing={}, step_collapsed={}, huge_final_gradient={}, \
         live_lock_present={}",
        report.initial_cost,
        report.initial_grad_norm,
        report.predicted_decrease,
        report.actual_decrease,
        report.materialization_ratio,
        report.outer_converged,
        report.outer_iterations,
        report.final_value,
        report.final_grad_norm,
        report.rho_displacement,
        report.ev,
        report.telemetry.criterion_calls,
        report.telemetry.infeasible_criterion_evals,
        report.telemetry.infeasible_total(),
        report.outer_error,
        report.predicted_decrease_not_materializing,
        report.step_collapsed,
        report.huge_final_gradient,
        report.live_lock_present,
    );
    assert!(
        !report.live_lock_present,
        "#2156 CEILING-vs-PATHOLOGY instrument detected the live-lock signature: \
         predicted decrease did not materialize, accepted ρ step collapsed, and \
         final gradient stayed huge; report={report:?}"
    );
}

/// #2080 — heavier K=3 wide-`p` variant (the issue's headline shape). Same
/// bounded-probe-budget contract.
#[test]
fn wide_p_outer_reml_terminates_k3_heavy_2080() {
    let (ev, telemetry) = run_wide_outer_fit(96, 96, 3, 2);
    eprintln!(
        "[#2080 heavy] K=3 wide-p outer fit: ev={ev:.4}, criterion_calls={}, \
         infeasible_total={}",
        telemetry.criterion_calls,
        telemetry.infeasible_total(),
    );
    assert!(telemetry.criterion_calls <= 96);
    assert!(ev.is_finite() && ev > 0.15);
}

/// #2080 ENTANGLED two-circle target — two equal-variance circles on OVERLAPPING
/// (dense, all-column) 2-frames, unlike `two_circle_wide_target`'s even/odd
/// DISJOINT split. Equal variance in a shared 4-D signal subspace makes the
/// pairing rotation-AMBIGUOUS to PCA (any orthonormal rotation of the 4 leading
/// PCs has the same variance), so the PCA-residual chart seed hands both atoms a
/// MIXTURE of the two circles → they fit the same mixed subspace → `μ̂ = 1.0`
/// co-collapse. Fourth-order (kurtosis) independence — the joint-Jacobi ISA seed —
/// is what resolves the rotation. This is the minimal faithful repro of the
/// issue's entangled product-of-circles co-collapse.
fn entangled_two_circle_wide_target(n: usize, p: usize, sigma: f64) -> Array2<f64> {
    let mut fa = Array2::<f64>::zeros((2, p));
    let mut fb = Array2::<f64>::zeros((2, p));
    for j in 0..p {
        fa[[0, j]] = deterministic_circle_noise(j, 0);
        fa[[1, j]] = deterministic_circle_noise(j, 1);
        fb[[0, j]] = deterministic_circle_noise(j, 2);
        fb[[1, j]] = deterministic_circle_noise(j, 3);
    }
    for f in [&mut fa, &mut fb] {
        for r in 0..2 {
            let nrm = (0..p).map(|j| f[[r, j]] * f[[r, j]]).sum::<f64>().sqrt();
            for j in 0..p {
                f[[r, j]] /= nrm.max(1.0e-300);
            }
        }
    }
    // Tile the 2-TORUS on an independent grid: θ_a and θ_b must be STATISTICALLY
    // INDEPENDENT (a genuine product of two circles). A dependent parameterization
    // (θ_b = 2θ_a, the previous `2*row`) is a single 1-D Lissajous/(1,2)-knot curve
    // with only ONE true latent factor — a K=2 fit then CORRECTLY leaves one atom
    // redundant, which no seed can split and which is not the co-collapse we are
    // testing. ISA separates independent subspaces, so the fixture must contain two.
    // `independent_two_circle_phases` chooses the largest divisor of `n` at or
    // below √n, so `(row mod n1, row / n1)` is a bijection onto the n1×n2 grid:
    // `(θ_a, θ_b)` is jointly uniform and therefore independent.
    let mut z = Array2::<f64>::zeros((n, p));
    for row in 0..n {
        let (ta, tb) = independent_two_circle_phases(n, row);
        let (ca, sa) = (ta.cos(), ta.sin());
        let (cb, sb) = (tb.cos(), tb.sin());
        for jj in 0..p {
            z[[row, jj]] = ca * fa[[0, jj]]
                + sa * fa[[1, jj]]
                + cb * fb[[0, jj]]
                + sb * fb[[1, jj]]
                + sigma * deterministic_circle_noise(row, jj + 7);
        }
    }
    for j in 0..p {
        let mut mean = 0.0_f64;
        for row in 0..n {
            mean += z[[row, j]];
        }
        mean /= n as f64;
        let mut var = 0.0_f64;
        for row in 0..n {
            let d = z[[row, j]] - mean;
            var += d * d;
        }
        let sd = (var / n as f64).sqrt().max(1.0e-12);
        for row in 0..n {
            z[[row, j]] = (z[[row, j]] - mean) / sd;
        }
    }
    z
}

/// #2080/#2023 — the ENTANGLED co-collapse regression (fails-before / passes-after
/// the joint-Jacobi ISA chart seed). Matched K=2 on the overlapping-frame two-circle
/// target: with the PCA-residual seed both atoms co-collapse onto the same mixed
/// subspace (the outer solver then thrashes infeasible probes to a `Fatal` abort);
/// with the independence-separating ISA seed the two circles land on DISTINCT atoms.
/// Collapse is EV-INVISIBLE, so a positive EV alone is not enough — the load-bearing
/// assertion is that BOTH atoms carry material decoder norm (a weakest/strongest
/// ratio well above the ~0.13 collapse regime and inside the ~0.42 healthy regime
/// measured on this shape).
#[test]
fn entangled_two_circle_outer_reml_separates_2080() {
    let n = 240usize;
    let p = 96usize;
    let k = 2usize;
    let harmonics = 2usize;
    let z = entangled_two_circle_wide_target(n, p, 0.03);
    // Diagnostic: how many independent circle planes does the joint-Jacobi ISA
    // split κ-CERTIFY on this target? If < k, the seed falls back to the PCA peel
    // and cannot separate — distinguishing "certificate failed" from "engaged but
    // under-separated" (the two contingencies for a co-collapse red).
    let isa_certified = match super::isa_seed::capture_signal_span(z.view(), k) {
        Ok(Some(parts)) => super::isa_seed::isa_extract_certified_planes(
            z.view(),
            &parts,
            k,
            &super::isa_seed::IsaSeedConfig::default(),
        )
        .len(),
        _ => 0,
    };
    let (term, seed_dispersion) = two_circle_periodic_term(z.view(), k, harmonics);
    let mode = AssignmentMode::ordered_beta_bernoulli(1.0, 1.0, false);
    let init_rho = SaeManifoldRho::new(0.02_f64.ln(), 1.0_f64.ln(), vec![array![0.0]; k])
        .seed_scaled_by_dispersion_for_assignment(seed_dispersion, mode)
        .expect("seed dispersion is finite and strictly positive");
    let seed = init_rho.to_flat();
    let n_params = seed.len();
    let mut objective =
        SaeManifoldOuterObjective::new(term, z.clone(), None, init_rho, 8, 0.04, 1.0e-6, 1.0e-6);
    let result = OuterProblem::new(n_params)
        .with_initial_rho(seed)
        .with_max_iter(4)
        .with_seed_config(gam_problem::SeedConfig {
            max_seeds: 1,
            seed_budget: 1,
            ..Default::default()
        })
        .run(&mut objective, "SAE manifold entangled two-circle")
        .expect("#2080 entangled two-circle outer penalized quasi-Laplace fit must terminate, not abort");
    assert!(
        result.converged(),
        "#2080 entangled acceptance requires a converged outer penalized quasi-Laplace optimum: \
         iterations={}, final_value={:.6e}, final_grad_norm={:?}",
        result.iterations,
        result.final_value,
        result.final_grad_norm,
    );
    objective
        .certify_outer_result(&result)
        .expect("entangled two-circle outer result must certify the installed state");
    let fitted = objective.into_fitted().expect("outer fit was evaluated");
    let ev = global_ev(z.view(), fitted.term.fitted().view());
    let mut norms = vec![0.0_f64; k];
    for (i, atom) in fitted.term.atoms.iter().enumerate() {
        norms[i] = atom
            .decoder_coefficients()
            .iter()
            .map(|v| v * v)
            .sum::<f64>()
            .sqrt();
    }
    let hi = norms.iter().copied().fold(0.0_f64, f64::max);
    let lo = norms.iter().copied().fold(f64::INFINITY, f64::min);
    let ratio = lo / hi.max(1.0e-300);
    eprintln!(
        "[#2080 entangled] isa_certified_planes={isa_certified}/{k}, ev={ev:.4}, \
         decoder_norms={norms:?}, ratio={ratio:.3}"
    );
    assert!(
        ev.is_finite() && ev > 0.20,
        "entangled K=2 fit must recover a materially positive EV (got {ev:.4})"
    );
    assert!(hi > 0.0, "at least one atom must carry decoder norm");
    assert!(
        ratio > 0.30,
        "both entangled circles must be recovered on DISTINCT atoms (no co-collapse); \
         norms={norms:?} ratio={ratio:.3} — a ratio near the ~0.13 collapse regime is the \
         μ̂ = 1.0 shared-subspace collapse the joint-Jacobi ISA seed must prevent"
    );
}

/// gamfit#2138 (fit-robustness half) — the curved (periodic) atom's inner
/// Newton/arrow-Schur solve must CONVERGE on a small (`n = 35`) fold at high
/// working rank (`m = 9`, all harmonics genuinely excited, so no rank reduction
/// collapses the design), across the whole smoothing sweep — including the
/// over-smoothed tail where the undamped Laplace log-det system is worst
/// conditioned. Before the robustness work an ill-conditioned small-`n`
/// high-rank probe `ρ` could grind the inner refinement budget and surface a
/// `RemlConvergenceError` (the theory experiments had to work around it with a
/// lower rank + fixed smoothing); the inner joint fit's Armijo line search +
/// proximal-correction LM ridge escalation keeps every step a descent step, so
/// the fit reaches a finite, materially-positive-EV basin at every ρ instead of
/// diverging. Each fixed-ρ evaluation must return `Ok` with a finite penalized quasi-Laplace cost
/// and (for the feasible low-to-mid smoothing range) a materially positive EV.
#[test]
fn small_fold_high_rank_circle_inner_solve_converges_2138() {
    let n = 35usize;
    let p = 12usize;
    let harmonics = 4usize;
    let m = 1 + 2 * harmonics;
    let mut frames = Array2::<f64>::zeros((2 * harmonics, p));
    for h in 0..2 * harmonics {
        for j in 0..p {
            frames[[h, j]] = deterministic_circle_noise(h, j);
        }
    }
    let mut z = Array2::<f64>::zeros((n, p));
    for row in 0..n {
        let t = row as f64 / n as f64;
        for hh in 0..harmonics {
            let ang = std::f64::consts::TAU * (hh as f64 + 1.0) * t;
            let (c, s) = (ang.cos(), ang.sin());
            for j in 0..p {
                z[[row, j]] += c * frames[[2 * hh, j]] + s * frames[[2 * hh + 1, j]];
            }
        }
    }
    for j in 0..p {
        let mean: f64 = (0..n).map(|r| z[[r, j]]).sum::<f64>() / n as f64;
        let var: f64 = (0..n).map(|r| (z[[r, j]] - mean).powi(2)).sum::<f64>() / n as f64;
        let sd = var.sqrt().max(1.0e-12);
        for r in 0..n {
            z[[r, j]] = (z[[r, j]] - mean) / sd;
        }
    }
    let evaluator: Arc<dyn SaeBasisSecondJet> =
        Arc::new(
        PeriodicHarmonicEvaluator::new(m).expect("m is a valid odd periodic basis width"),
    );
    let seed_coords =
        sae_pca_seed_initial_coords(z.view(), &[SaeAtomBasisKind::Periodic], &[1])
            .expect("one periodic basis kind and one dim for the single atom");
    let coords = seed_coords.slice(s![0, .., 0..1]).to_owned();
    let (phi, jet) = evaluator
        .evaluate(coords.view())
        .expect("the seeded coords lie in the periodic chart domain");
    let mut xtx = fast_atb(&phi, &phi);
    for i in 0..m {
        xtx[[i, i]] += 1.0e-8;
    }
    let xtz = fast_atb(&phi, &z);
    let decoder = xtx
        .cholesky(Side::Lower)
        .expect("phi^T phi + 1e-8 I is positive definite")
        .solve_mat(&xtz);
    let atom = SaeManifoldAtom::new_with_provided_function_gram(
        "circle",
        SaeAtomBasisKind::Periodic,
        1,
        phi,
        jet,
        decoder,
        Array2::<f64>::eye(m),
    )
    .expect("phi, jet, decoder and gram were built with matching shapes")
    .with_basis_evaluator(evaluator.clone());
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        Array2::<f64>::from_elem((n, 1), 6.0),
        vec![coords],
        vec![LatentManifold::Circle { period: 1.0 }],
        AssignmentMode::ordered_beta_bernoulli(1.0, 1.0, false),
    )
    .expect("one logit column, coord block and manifold for the single atom");
    let base = SaeManifoldTerm::new(vec![atom], assignment)
        .expect("the single atom and its assignment agree on atom count");
    // The whole smoothing sweep, from flexible (-8) through the over-smoothed tail
    // (+8) where the undamped Laplace log-det is worst conditioned.
    for &smooth in &[-8.0_f64, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0] {
        let mut t = base.clone();
        let r = SaeManifoldRho::new(0.02_f64.ln(), smooth, vec![array![0.0]]);
        let evaluated = t
            .penalized_quasi_laplace_criterion_with_cache(
                z.view(),
                &r,
                None,
                60,
                0.04,
                1.0e-6,
                1.0e-6,
            )
            .unwrap_or_else(|err| {
                panic!(
                    "#2138: high-working-rank (m={m}) circle inner solve must converge at a \
                     small (n={n}) fold, smoothing={smooth}, not diverge into a \
                     RemlConvergenceError; got: {err}"
                )
            });
        let cost = evaluated.0;
        let ev = global_ev(z.view(), t.fitted().view());
        // The design is full working rank (all m columns excited): no rank
        // reduction should collapse it, so the ill-conditioned regime is real.
        assert_eq!(
            t.atoms[0].basis_size(),
            m,
            "#2138: the multi-harmonic target must keep the atom at full working rank m={m}",
        );
        assert!(
            cost.is_finite(),
            "#2138: inner solve returned a non-finite penalized quasi-Laplace cost at smoothing={smooth}",
        );
        assert!(
            ev.is_finite() && ev > 0.30,
            "#2138: high-working-rank small-fold circle fit must recover a materially positive \
             EV at smoothing={smooth} (got {ev:.4}), proving the inner solve reached a real \
             basin rather than a diverged / collapsed state",
        );
    }
}

/// #2080 COST-LANE PROFILER + criterion-finiteness gate.
/// Measures how a SINGLE outer penalized quasi-Laplace criterion evaluation scales in ambient width
/// `p` for the correctly-specified K=1 circle (no co-collapse). Splits the wall
/// time into: (A) the damped inner (t,β) Newton solve `run_joint_fit_arrow_schur`
/// and (B) the residual = the undamped-logdet re-converge + dense β-Schur factor
/// that `penalized_quasi_laplace_criterion_with_cache_refine_policy` adds on top. This localizes the
/// cubic-in-p term the issue tracks. Asserted invariant: the inner solve
/// converges finitely and the full criterion is FINITE (rankable) at every
/// width — a non-finite criterion on this correctly-specified probe is the
/// #1094-class outer refusal. Widths kept small enough for the standard shard;
/// the wide tail (64/96/128) is profiling territory for the #2080 owner's
/// dedicated runs.
#[test]
fn profile_wide_p_criterion_cost_2080() {
    let harmonics = 2usize; // m = 1 + 2*2 = 5 basis columns per atom
    for &p in &[16usize, 32, 48] {
        let n = 96usize;
        let z = one_circle_wide_target(n, p, 0.05);
        let (term, seed_dispersion) = two_circle_periodic_term(z.view(), 1, harmonics);
        let mode = AssignmentMode::ordered_beta_bernoulli(1.0, 1.0, false);
        let rho = SaeManifoldRho::new(0.02_f64.ln(), 1.0_f64.ln(), vec![array![0.0]])
            .seed_scaled_by_dispersion_for_assignment(seed_dispersion, mode)
            .expect("seed dispersion is finite and strictly positive");
        let beta_dim = term.beta_dim();

        // Phase A: damped inner solve alone.
        let mut ta = term.clone();
        let mut rho_a = rho.clone();
        let a0 = std::time::Instant::now();
        ta.run_joint_fit_arrow_schur(z.view(), &mut rho_a, None, 8, 0.04, 1.0e-6, 1.0e-6)
            .expect("inner solve");
        let dt_a = a0.elapsed().as_secs_f64();

        // Phase A+B: full criterion (inner solve + undamped logdet + dense Schur).
        let mut tb = term.clone();
        let b0 = std::time::Instant::now();
        let evaluated = tb
            .penalized_quasi_laplace_criterion_with_cache_refine_policy(
                z.view(),
                &rho,
                None,
                8,
                0.04,
                1.0e-6,
                1.0e-6,
                true,
            )
            .expect("full criterion");
        let dt_full = b0.elapsed().as_secs_f64();
        let dt_b = (dt_full - dt_a).max(0.0);
        assert!(
            evaluated.0.is_finite(),
            "#2080/#1094: the outer penalized quasi-Laplace criterion must be RANKABLE (finite) on a \
             correctly-specified K=1 wide-p circle at p={p}; a non-finite value here \
             is the probe-refusal failure class (got {})",
            evaluated.0
        );
        eprintln!(
            "[#2080 profile] p={p:>3} beta_dim={beta_dim:>4} | inner_solve={dt_a:8.3}s | logdet_phase={dt_b:8.3}s | full={dt_full:8.3}s | cost={:.4}",
            evaluated.0
        );
    }
}

/// #2080 wide-p per-eval localizer (zz_measure diagnostics). Splits a single
/// K=1 criterion evaluation into: (A) the damped inner (t,β) Newton solve,
/// (M) the dense exact-A materialization column-by-column, (E) the full exact
/// observed-information log-dets (M + two dense `eigh`), and the residual
/// refine-loop remainder = full − A − E, so the cubic-in-p term is localized
/// in ONE run. The in-suite sweep stops at p=32 to stay affordable (`#[ignore]`
/// is banned; widen the sweep locally when hunting the tail).
#[test]
fn zz_measure_wide_p_criterion_cost_localizer_2080() {
    let harmonics = 2usize; // m = 5 basis columns per atom
    for &p in &[16usize, 32] {
        let n = 96usize;
        let z = one_circle_wide_target(n, p, 0.05);
        let (term, seed_dispersion) = two_circle_periodic_term(z.view(), 1, harmonics);
        let mode = AssignmentMode::ordered_beta_bernoulli(1.0, 1.0, false);
        let rho = SaeManifoldRho::new(0.02_f64.ln(), 1.0_f64.ln(), vec![array![0.0]])
            .seed_scaled_by_dispersion_for_assignment(seed_dispersion, mode)
            .expect("seed dispersion is finite and strictly positive");
        let beta_dim = term.beta_dim();

        // Phase A: damped inner solve alone.
        let mut ta = term.clone();
        let mut rho_a = rho.clone();
        let a0 = std::time::Instant::now();
        ta.run_joint_fit_arrow_schur(z.view(), &mut rho_a, None, 8, 0.04, 1.0e-6, 1.0e-6)
            .expect("inner solve");
        let dt_a = a0.elapsed().as_secs_f64();

        // Full criterion (returns the converged undamped cache).
        let mut tb = term.clone();
        let f0 = std::time::Instant::now();
        let (_cost, _loss, cache) = tb
            .penalized_quasi_laplace_criterion_with_cache(
                z.view(),
                &rho,
                None,
                8,
                0.04,
                1.0e-6,
                1.0e-6,
            )
            .expect("full criterion");
        let dt_full = f0.elapsed().as_secs_f64();
        let total_t = cache.delta_t_len();
        let dim = total_t + cache.k;

        // Phase M: dense exact-A materialization (dim column matvecs).
        let m0 = std::time::Instant::now();
        let a_dense = tb
            .materialize_exact_hessian_dense(&rho, z.view(), &cache)
            .expect("materialize A");
        let dt_m = m0.elapsed().as_secs_f64();

        // Phase E: full exact observed-information log-dets (materialize + 2 eigh).
        let e0 = std::time::Instant::now();
        let log_dets = tb
            .exact_observed_information_log_dets(&rho, z.view(), &cache)
            .expect("observed-information log-dets");
        let dt_e = e0.elapsed().as_secs_f64();

        let dt_eigh = (dt_e - dt_m).max(0.0);
        let dt_refine = (dt_full - dt_a - dt_e).max(0.0);
        eprintln!(
            "[#2080 localize] p={p:>3} beta_dim={beta_dim:>4} dim={dim:>4} a_dim={} \
             log_dets={log_dets:?} | inner_A={dt_a:8.3}s | refine_rest={dt_refine:8.3}s | \
             materialize_M={dt_m:8.3}s | eigh_E-M={dt_eigh:8.3}s | full={dt_full:8.3}s",
            a_dense.nrows(),
        );
        // The cost attribution above is only about the right work if each phase
        // actually produced its object: a square exact A covering at least the
        // t-block, and finite log-dets. A refusal laundered into a non-finite
        // `log_dets` would otherwise be timed and reported as a successful phase.
        assert!(
            a_dense.nrows() == a_dense.ncols() && a_dense.nrows() >= total_t,
            "p={p}: the materialized exact A must be a square operator covering the {total_t} \
             t-coordinates, got {}x{}",
            a_dense.nrows(),
            a_dense.ncols()
        );
        assert!(
            log_dets.0.is_finite() && log_dets.1.is_finite(),
            "p={p}: the exact observed-information log-dets this phase is timed on must be \
             finite, got {log_dets:?}"
        );
        assert!(
            dim > 0 && beta_dim > 0,
            "p={p}: the localizer must be run on a nonempty system (dim={dim}, \
             beta_dim={beta_dim})"
        );
    }
}

/// #2439 MEASUREMENT (zz_measure) — WHY does asking for the gradient change the value?
///
/// `value_lane_prices_at_shared_fixed_point_2228` measures that the same objective
/// at the same ρ reports `1.1359321989218647e3` under `OuterEvalOrder::Value` and
/// `1.1363163948443218e3` under `ValueAndGradient` — 22,700× the certification
/// bound, with the `Value` lane matching the bare criterion to all 17 digits.
///
/// Two candidate causes, and this probe decides between them in one run:
///
/// 1. **the assembly differs** — both lanes converge to the SAME inner mode and
///    still price it differently, which would put the defect in what is summed;
/// 2. **the lanes evaluate at DIFFERENT inner modes** — `eval()` re-solves and
///    lands elsewhere, so the value is computed at one mode while the gradient
///    describes another. That is a violation of the envelope identity
///    `dV/dρ = ∂ℓ_p/∂ρ|_θ̂` by construction, not merely a discrepancy.
///
/// Both lanes publish `inner_beta_hint`, so comparing them bitwise separates the
/// two without instrumenting either lane.
///
/// The probe also varies something the failing test fixes: that test builds a
/// FRESH objective per lane, so the `#2080 (a)` hand-off — install the value
/// probe's converged inner state before the gradient lane's criterion loop, whose
/// stated purpose is "same converged optimum ⇒ identical criterion value" — never
/// happens. Production evaluates both on ONE objective. Arm B runs that ordering.
/// If B agrees where A disagrees, the hand-off is load-bearing and the value is a
/// functional of the starting state rather than of ρ; if B disagrees too, the
/// defect is independent of the hand-off.
#[test]
fn zz_measure_2439_value_vs_gradient_inner_mode() {
    let n = 96usize;
    let p = 48usize;
    let z = one_circle_wide_target(n, p, 0.05);
    let (term, seed_dispersion) = two_circle_periodic_term(z.view(), 1, 2);
    let mode = AssignmentMode::ordered_beta_bernoulli(1.0, 1.0, false);
    let imi = 16usize;
    let (lr, re, rb) = (0.04_f64, 1.0e-6_f64, 1.0e-6_f64);
    let rho = SaeManifoldRho::new(0.02_f64.ln(), 4.0_f64, vec![array![0.0]])
        .seed_scaled_by_dispersion_for_assignment(seed_dispersion, mode)
        .expect("seed dispersion is finite and strictly positive");
    let rho_flat = rho.to_flat();

    let report = |tag: &str, a: &OuterEval, b: &OuterEval| {
        let diff = (a.cost - b.cost).abs();
        let bound = f64::EPSILON.sqrt() * a.cost.abs().max(b.cost.abs()).max(1.0);
        eprintln!(
            "[#2439 {tag}] value={:.16e} grad_lane={:.16e} diff={diff:.6e} bound={bound:.6e}",
            a.cost, b.cost
        );
        // Both lanes ACCEPTED (the `.expect(...)`s above), so both must price a
        // finite cost — `diff` and the certification `bound` it is compared against
        // are otherwise undefined and the two candidate causes cannot be separated.
        assert!(
            a.cost.is_finite() && b.cost.is_finite(),
            "[#2439 {tag}] both accepted lanes must price a finite cost \
             (value={}, grad_lane={})",
            a.cost,
            b.cost
        );
        assert!(
            diff.is_finite() && bound > 0.0,
            "[#2439 {tag}] the lane gap and its certification bound must be well-defined \
             (diff={diff}, bound={bound})"
        );
        match (&a.inner_beta_hint, &b.inner_beta_hint) {
            (Some(bv), Some(bg)) if bv.len() == bg.len() => {
                let identical = bv
                    .iter()
                    .zip(bg.iter())
                    .all(|(x, y)| x.to_bits() == y.to_bits());
                let max_abs = bv
                    .iter()
                    .zip(bg.iter())
                    .map(|(x, y)| (x - y).abs())
                    .fold(0.0_f64, f64::max);
                eprintln!(
                    "[#2439 {tag}] beta len={} identical={identical} \
                     max|dbeta|={max_abs:.6e} => {}",
                    bv.len(),
                    if identical {
                        "SAME MODE: the assembly differs (cause 1)"
                    } else {
                        "DIFFERENT MODES: envelope identity broken by construction (cause 2)"
                    }
                );
            }
            (bv, bg) => eprintln!(
                "[#2439 {tag}] beta hints not comparable: value={:?} grad={:?}",
                bv.as_ref().map(|v| v.len()),
                bg.as_ref().map(|v| v.len())
            ),
        }
    };

    // Arm A — the failing test's construction: a FRESH objective per lane, so no
    // converged inner state is handed from the value probe to the gradient lane.
    let mut obj_v =
        SaeManifoldOuterObjective::new(term.clone(), z.clone(), None, rho.clone(), imi, lr, re, rb);
    let a_value = OuterObjective::eval_with_order(&mut obj_v, &rho_flat, OuterEvalOrder::Value)
        .expect("arm A value lane");
    let mut obj_g =
        SaeManifoldOuterObjective::new(term.clone(), z.clone(), None, rho.clone(), imi, lr, re, rb);
    let a_grad =
        OuterObjective::eval_with_order(&mut obj_g, &rho_flat, OuterEvalOrder::ValueAndGradient)
            .expect("arm A gradient lane");
    report("A fresh-objective-per-lane", &a_value, &a_grad);

    // Arm B — production's ordering: ONE objective, value probe first, so the
    // `#2080 (a)` hand-off can install the probe's converged inner state.
    let mut obj =
        SaeManifoldOuterObjective::new(term.clone(), z.clone(), None, rho.clone(), imi, lr, re, rb);
    let b_value = OuterObjective::eval_with_order(&mut obj, &rho_flat, OuterEvalOrder::Value)
        .expect("arm B value lane");
    let b_grad =
        OuterObjective::eval_with_order(&mut obj, &rho_flat, OuterEvalOrder::ValueAndGradient)
            .expect("arm B gradient lane");
    report("B shared-objective ", &b_value, &b_grad);
}

/// #2228 MEASUREMENT (zz_measure) — is the value-lane fixture's inner solve at a
/// FLOOR, or is it still converging and merely out of budget?
///
/// This is the question the #2267 ratchet left open. That work took one criterion
/// evaluation on this fixture from 808 inner iterations / 52 s to 272 / 4.36 s,
/// and ‖g‖ moved only 1.522323e-1 → 1.100073e-1 against an unchanged tolerance of
/// 1.435797e-3 — still 77× above. Iteration budget is therefore no longer what
/// stands between this fixture and its tolerance, but "no longer the BINDING
/// constraint" is not the same claim as "more budget cannot help", and the two
/// have opposite owners:
///
/// * still converging (some budget succeeds) ⇒ a RATE problem, and rate belongs to
///   the step rule and its accept rule (#2267), not to the refusal contract;
/// * floored (no budget succeeds) ⇒ the inner solve cannot reach `1e-5 ·
///   inner_iterate_scale()` on this problem at all, and a contract that demands it
///   is asking for something unattainable — which is #2228's own question.
///
/// The sweep answers it without parsing anything out of an error message: sweep the
/// refine budget and record only whether the criterion returns. A binary outcome per
/// budget is all the question needs, and it is exactly the acceptance bar this issue
/// is written against ("the fit must converge, not refuse").
#[test]
fn zz_measure_2228_value_lane_budget_sweep() {
    let n = 96usize;
    let p = 48usize;
    let z = one_circle_wide_target(n, p, 0.05);
    let mode = AssignmentMode::ordered_beta_bernoulli(1.0, 1.0, false);
    let (lr, re, rb) = (0.04_f64, 1.0e-6_f64, 1.0e-6_f64);
    // The sweep's binary CONVERGED/REFUSED outcome only answers the rate-vs-floor
    // question if every budget is run against the same well-posed wide-p target.
    assert!(
        z.nrows() == n && z.ncols() == p && z.iter().all(|v| v.is_finite()),
        "[#2228 sweep] the wide-p target must be a finite {n}x{p} matrix, got {}x{}",
        z.nrows(),
        z.ncols()
    );

    for imi in [8usize, 16, 32, 64, 128] {
        // Rebuild the term and rho per budget: the criterion mutates inner state,
        // so a shared term would make each budget start where the previous one
        // stopped and the sweep would measure a warm continuation instead of the
        // budget it names.
        let (term, seed_dispersion) = two_circle_periodic_term(z.view(), 1, 2);
        let rho = SaeManifoldRho::new(0.02_f64.ln(), 4.0_f64, vec![array![0.0]])
            .seed_scaled_by_dispersion_for_assignment(seed_dispersion, mode)
            .expect("seed dispersion is finite and strictly positive");
        // FULL budget (refine_progress_extension = true): the arm that must reach
        // the root the test prices against.
        let mut t = term.clone();
        let started = std::time::Instant::now();
        let full =
            t.penalized_quasi_laplace_criterion_with_cache(z.view(), &rho, None, imi, lr, re, rb);
        let full_secs = started.elapsed().as_secs_f64();
        // COARSE budget (refine_progress_extension = false): the raw diagnostic
        // policy, which the fixture requires to be inadequate at the same budget.
        // If raising `imi` ever made this one adequate too, the test's
        // coarse-vs-full premise would go vacuous, so sweep both together.
        let mut t_coarse = term.clone();
        let started = std::time::Instant::now();
        let coarse = t_coarse.penalized_quasi_laplace_criterion_with_cache_refine_policy(
            z.view(),
            &rho,
            None,
            imi,
            lr,
            re,
            rb,
            false,
        );
        let coarse_secs = started.elapsed().as_secs_f64();
        let full_report = match &full {
            Ok((value, _, _)) => format!("CONVERGED value={value:.9e} ({full_secs:.2}s)"),
            Err(err) => format!("REFUSED ({full_secs:.2}s): {err:?}"),
        };
        let coarse_report = match &coarse {
            Ok(evaluated) => format!("CONVERGED value={:.9e} ({coarse_secs:.2}s)", evaluated.0),
            Err(err) => format!("REFUSED ({coarse_secs:.2}s): {err:?}"),
        };
        eprintln!("[#2228 sweep] imi={imi:>4} full={full_report}");
        eprintln!("[#2228 sweep] imi={imi:>4} coarse={coarse_report}");
        // "CONVERGED" is the answer this sweep records, so an `Ok` carrying a
        // non-finite value would flip the rate-vs-floor verdict while printing as
        // a converged budget.
        if let Ok((value, _, _)) = &full {
            assert!(
                value.is_finite(),
                "[#2228 sweep] imi={imi}: a CONVERGED full-budget criterion must carry a finite \
                 value, got {value}"
            );
        }
        if let Ok(evaluated) = &coarse {
            assert!(
                evaluated.0.is_finite(),
                "[#2228 sweep] imi={imi}: a CONVERGED coarse-budget criterion must carry a \
                 finite value, got {}",
                evaluated.0
            );
        }
    }
}

/// #2228 regression — the line-search Value lane must price the shared inner
/// fixed point. The BFGS/ARC cost probe (`OuterEvalOrder::Value`) ranks steps
/// whose direction came from the gradient lane's exact implicit `∇f`, computed
/// at the FULLY converged idempotent inner root
/// (`penalized_quasi_laplace_criterion_with_cache`, `refine_progress_extension =
/// true`); the outer certification samples that same root. In the
/// ill-conditioned wide-`p` / over-smoothed regime a COARSE probe
/// (`refine_progress_extension = false`, a raw reduced-budget policy that no
/// outer ranking lane may consume) sits ~1% off that root, so (a) no step
/// reduces the coarsely-ranked value while pointing
/// down the analytic gradient — BFGS backtracks to `StepSizeTooSmall` at
/// iteration 1 — and (b) the shipped coarse terminal value fails the outer cert
/// value-agreement gate ("cost-only value disagrees with analytic-sample value",
/// #2228). This pins probe == analytic within the certification roundoff bound at
/// an admitted iterate, so a future rewrite that reintroduces the coarse budget on
/// the Value lane goes red here instead of silently shipping the desync.
#[test]
fn value_lane_prices_at_shared_fixed_point_2228() {
    let n = 96usize;
    let p = 48usize;
    let z = one_circle_wide_target(n, p, 0.05);
    let (term, seed_dispersion) = two_circle_periodic_term(z.view(), 1, 2);
    let mode = AssignmentMode::ordered_beta_bernoulli(1.0, 1.0, false);
    // Refine budget for BOTH arms. Measured (`zz_measure_2228_value_lane_budget_sweep`,
    // run 30150455983): the full arm REFUSES at 8 and converges from 16 on, with the
    // value identical to ten significant figures across 16/32/64/128, while the coarse
    // arm refuses at EVERY budget with its ‖g‖ pinned at ~1.9032e-1 from 16 through 128
    // — a 16× budget increase moves it by 4e-7 relative, i.e. the coarse path is at a
    // floor and no budget rescues it. So 8 no longer buys this fixture the converged
    // root its own precondition needs, and raising it cannot cost the coarse-inadequacy
    // premise the assertion below rests on. The stability check after `v_true` turns
    // "16 is enough" from an assumption into an assertion.
    let imi = 16usize;
    let (lr, re, rb) = (0.04_f64, 1.0e-6_f64, 1.0e-6_f64);
    // Over-smoothed rho: the undamped Laplace log-det is worst-conditioned here, so
    // the inner (t,beta) solve needs progress-extension refinement beyond the
    // coarse probe budget to reach the fixed point. The sanity check below asserts
    // the fixture actually exercises that regime (else the invariant is vacuous).
    let rho = SaeManifoldRho::new(0.02_f64.ln(), 4.0_f64, vec![array![0.0]])
        .seed_scaled_by_dispersion_for_assignment(seed_dispersion, mode)
        .expect("seed dispersion is finite and strictly positive");
    let rho_flat = rho.to_flat();

    // Sanity: the COARSE (false) refine budget must be demonstrably inadequate at
    // this rho, else the invariant assertion below would pass vacuously on any
    // fixture. Inadequacy has two admissible forms and the fixture is pinned to
    // whichever it lands in:
    //
    //   (a) the coarse budget returns a value that differs from the full-budget
    //       root by more than the certification roundoff bound, or
    //   (b) the coarse budget REFUSES with the typed non-convergence error — the
    //       STRONGER form of the same statement, since no value at all is further
    //       from the root than any finite disagreement.
    //
    // (b) is the DESIGNED raw coarse-policy behaviour, not a defect: per #2080 a
    // reduced-budget probe may return a typed verdict instead of grinding. No
    // production outer value/ranking entry point consumes that policy: they all
    // route through `authoritative_envelope_value_probe`, whose full-refine drive
    // either reaches the shared fixed point or refuses. This assertion is the
    // only caller that observes the raw coarse policy.
    //
    // Landing in (b) makes the Value-lane invariant below STRICTLY harder to
    // satisfy, not easier: a Value lane wrongly rebuilt on the coarse budget would
    // surface `OuterEval::infeasible` (+inf) rather than a merely-~1%-off value,
    // so the regression this test exists to catch still goes red.
    let v_true = {
        let mut t = term.clone();
        t.penalized_quasi_laplace_criterion_with_cache(z.view(), &rho, None, imi, lr, re, rb)
            .expect("full-budget bare criterion evaluates")
            .0
    };
    // The full budget must be ADEQUATE, not merely non-erroring: a root that still
    // moves when given twice the budget is not the root the Value lane is supposed to
    // price, and pinning the test to one would make every assertion below a statement
    // about a budget rather than about a fixed point. Doubling must not move it.
    let v_true_double = {
        let mut t = term.clone();
        t.penalized_quasi_laplace_criterion_with_cache(z.view(), &rho, None, 2 * imi, lr, re, rb)
            .expect("double-budget bare criterion evaluates")
            .0
    };
    let root_bound = f64::EPSILON.sqrt() * v_true.abs().max(v_true_double.abs()).max(1.0);
    assert!(
        (v_true - v_true_double).abs() <= root_bound,
        "the full budget must reach a converged root, but doubling it moved the value: \
         v_true={v_true:.16e}, v_true(2x)={v_true_double:.16e}, diff={:.3e} > {root_bound:.3e} \
         (raise `imi` until it stops moving)",
        (v_true - v_true_double).abs()
    );
    let coarse = {
        let mut t = term.clone();
        t.penalized_quasi_laplace_criterion_with_cache_refine_policy(
            z.view(),
            &rho,
            None,
            imi,
            lr,
            re,
            rb,
            false,
        )
        .map(|evaluated| evaluated.0)
    };
    match &coarse {
        Ok(v_false) => {
            let regime_bound = f64::EPSILON.sqrt() * v_false.abs().max(v_true.abs()).max(1.0);
            eprintln!(
                "[#2228] coarse budget returned a value: v_false={v_false:.16e} \
                 v_true={v_true:.16e} diff={:.3e} regime_bound={regime_bound:.3e}",
                (v_false - v_true).abs()
            );
            assert!(
                (v_false - v_true).abs() > regime_bound,
                "fixture must exercise the coarse-vs-full under-refinement regime, else this \
                 test is vacuous: v_false={v_false:.16e}, v_true={v_true:.16e}, diff={:.3e} \
                 <= regime_bound={regime_bound:.3e} (strengthen the fixture)",
                (v_false - v_true).abs()
            );
        }
        Err(err) => {
            let message = err.numerical_message().unwrap_or_else(|| {
                panic!(
                    "the coarse budget may only be inadequate via the typed NUMERICAL \
                     non-convergence refusal; got a different typed refusal: {err:?}"
                )
            });
            eprintln!(
                "[#2228] coarse budget REFUSED (the stronger form of coarse-vs-full \
                 disagreement); v_true={v_true:.16e}; refusal: {message}"
            );
            assert!(
                message.contains("inner solve did not converge at fixed \u{3c1}"),
                "the coarse budget must be inadequate here via the typed non-convergence \
                 refusal, not some other numerical failure; got: {message}"
            );
        }
    }

    // Invariant: the line-search Value lane prices the SAME fixed point the
    // analytic gradient lane differentiates, within the certification roundoff
    // bound (the exact gate `rho_optimizer::run` enforces at fit end).
    // ONE objective, value probe first — production's ordering. The outer search
    // always evaluates the gradient lane at the ρ of the line search's last
    // successful value probe on the same objective, which lets the `#2080 (a)`
    // hand-off install that probe's converged inner state before the gradient
    // lane's criterion loop ("same converged optimum ⇒ identical criterion value").
    //
    // A fresh objective per lane instead would assert this invariant against a
    // construction production never uses, and would fail for a reason that says
    // nothing about the Value lane: measured (#2439,
    // `zz_measure_2439_value_vs_gradient_inner_mode`), evaluating the two lanes on
    // SEPARATE objectives lands them on genuinely different inner modes
    // (`max|Δβ| = 4.394e-2` over 240 coordinates) whose criterion values differ by
    // 3.842e-1 — because the inner solve stops at a tolerance rather than at a
    // unique fixed point, so θ̂ depends on where it started. On one objective the
    // two lanes agree to 3.6e-12 at a BITWISE-identical β.
    let mut obj =
        SaeManifoldOuterObjective::new(term.clone(), z.clone(), None, rho.clone(), imi, lr, re, rb);
    let value_eval = OuterObjective::eval_with_order(&mut obj, &rho_flat, OuterEvalOrder::Value)
        .expect("value lane evaluates");
    let value_lane = value_eval.cost;
    let value_beta = value_eval
        .inner_beta_hint
        .clone()
        .expect("the Value lane publishes its converged inner state for the hand-off");
    let gradient_eval =
        OuterObjective::eval_with_order(&mut obj, &rho_flat, OuterEvalOrder::ValueAndGradient)
            .expect("gradient lane evaluates");
    let analytic = gradient_eval.cost;

    // The hand-off is what makes the agreement below hold, and it is load-bearing:
    // without it the lanes converge to different modes. Pin it directly, so a
    // refactor that drops it fails here saying so, instead of silently degrading
    // the value-agreement assertion into a statement about two unrelated modes.
    let gradient_beta = gradient_eval
        .inner_beta_hint
        .as_ref()
        .expect("the gradient lane publishes the inner state it differentiated at");
    assert_eq!(
        value_beta.len(),
        gradient_beta.len(),
        "the two lanes must describe the same inner coordinate system"
    );
    let beta_gap = value_beta
        .iter()
        .zip(gradient_beta.iter())
        .map(|(v, g)| (v - g).abs())
        .fold(0.0_f64, f64::max);
    assert_eq!(
        beta_gap,
        0.0,
        "#2228/#2439: the gradient lane must differentiate AT the mode the Value lane priced,          not at one it re-solved for itself; max|Δβ|={beta_gap:.6e} over {} coordinates",
        value_beta.len()
    );
    let cert_bound = f64::EPSILON.sqrt() * value_lane.abs().max(analytic.abs()).max(1.0);
    eprintln!(
        "[#2228] value_lane={value_lane:.16e} analytic={analytic:.16e} diff={:.3e} \
         cert_bound={cert_bound:.3e}",
        (value_lane - analytic).abs()
    );
    assert!(
        (value_lane - analytic).abs() <= cert_bound,
        "#2228: line-search Value lane must price the analytic fixed point within the \
         certification roundoff bound: value_lane={value_lane:.16e}, \
         analytic={analytic:.16e}, diff={:.3e}, bound={cert_bound:.3e}",
        (value_lane - analytic).abs()
    );
}

/// PROBE (#2080 Class-B localizer). The K=2 wide-`p` acceptance fixture
/// (`wide_p_outer_reml_terminates_within_probe_budget_2080`) times out, and its
/// outer search never starts: every candidate seed is rejected because the
/// domain-entry criterion refuses with `inner solve did not converge at fixed ρ`.
/// So the timeout is decided ENTIRELY inside one criterion evaluation, on the
/// fixture's own seed ρ, before any outer step exists.
///
/// This probe reads that one evaluation instead of inferring it. Two halves:
///
///  1. A BUDGET SWEEP of the raw inner solve. Per budget it reports the KKT
///     residual `‖g‖`, its gauge quotient `‖Π⊥g‖`, the admission tolerance the
///     criterion actually compares them against, the penalised objective, the
///     per-iteration contraction `(‖g‖ₖ₊₁/‖g‖ₖ)^(1/Δbudget)`, and wall clock.
///     The contraction column is the discriminator the issue asks for: a value
///     bounded away from 1 with `‖g‖` still descending is a solve that needs
///     more iterations (a rate problem); a value AT 1 with the objective still
///     falling is a solve that is moving somewhere the residual does not see.
///
///  2. ONE full criterion evaluation with the solver's own per-iterate trace
///     forwarded to stderr, so the accepted step length, the LM ridge ladder and
///     the rejection route are READ rather than guessed.
///
/// Diagnostic only: it asserts that the readings exist, never a rate. The
/// numbers are the deliverable.
#[test]
fn zz_measure_k2_wide_p_inner_trajectory_2080() {
    let n = 96usize;
    let p = 96usize;
    let k = 2usize;
    let harmonics = 2usize;
    let z = two_circle_wide_target(n, p, 0.03);
    let (base, seed_dispersion) = two_circle_periodic_term(z.view(), k, harmonics);
    let mode = AssignmentMode::ordered_beta_bernoulli(1.0, 1.0, false);
    let rho = SaeManifoldRho::new(0.02_f64.ln(), 1.0_f64.ln(), vec![array![0.0]; k])
        .seed_scaled_by_dispersion_for_assignment(seed_dispersion, mode)
        .expect("seed dispersion is finite and strictly positive");
    // The acceptance fixture's own inner settings, so this reads the evaluation
    // that fixture performs and not a nearby one.
    let (inner_max_iter, learning_rate, ridge) = (8usize, 0.04_f64, 1.0e-6_f64);

    eprintln!(
        "[2080-K2] n={} p_out={} k={} beta_dim={} coord_dim={} inner_max_iter={inner_max_iter} \
         lr={learning_rate} ridge={ridge:.1e}",
        base.n_obs(),
        base.output_dim(),
        base.k_atoms(),
        base.beta_dim(),
        base.n_obs() * base.assignment.row_block_dim(),
    );
    eprintln!(
        "[2080-K2] budget | ‖g‖ | ‖Π⊥gauge g‖ | tol | pen_obj | contraction/iter | wall_s"
    );

    let budgets = [8usize, 16, 32, 64, 128, 256, 512];
    let mut previous: Option<(usize, f64)> = None;
    let mut readings = 0usize;
    for &budget in &budgets {
        let mut term = base.clone();
        let mut rho_fixed = rho.clone();
        let started = std::time::Instant::now();
        let outcome = term.run_joint_fit_arrow_schur_for_quasi_laplace(
            z.view(),
            &mut rho_fixed,
            None,
            budget,
            learning_rate,
            ridge,
            ridge,
        );
        let wall = started.elapsed().as_secs_f64();
        let outcome = match outcome {
            Ok(outcome) => outcome,
            Err(err) => {
                eprintln!(
                    "[2080-K2] {budget:>6} | inner solve HARD-ERRORED after {wall:.2}s: {err}"
                );
                continue;
            }
        };
        let Ok(sys) = term.assemble_arrow_schur(z.view(), &rho_fixed, None) else {
            eprintln!("[2080-K2] {budget:>6} | reassembly failed after {wall:.2}s");
            continue;
        };
        let grad_norm_sq = SaeManifoldTerm::system_grad_norm_sq(&sys);
        let grad_norm = grad_norm_sq.sqrt();
        let quotient = term.quotient_gradient_norm_from_system(
            &sys,
            grad_norm_sq,
            &rho_fixed
                .lambda_smooth_vec()
                .expect("the fixture rho carries one smoothing block per atom"),
        );
        let tolerance = SAE_MANIFOLD_INNER_GRAD_REL_TOL * term.inner_iterate_scale();
        let objective = term
            .penalized_objective_total(z.view(), &rho_fixed, None, 1.0)
            .unwrap_or(f64::NAN);
        let contraction = match previous {
            Some((prev_budget, prev_grad)) if prev_grad > 0.0 && grad_norm > 0.0 => {
                (grad_norm / prev_grad).powf(1.0 / ((budget - prev_budget) as f64))
            }
            _ => f64::NAN,
        };
        eprintln!(
            "[2080-K2] {budget:>6} | {grad_norm:.6e} | {quotient:.6e} | {tolerance:.6e} | \
             {objective:.9e} | {contraction:.6} | {wall:.2} (fixed_point={}, gap={:?})",
            outcome.fixed_point, outcome.gap,
        );
        previous = Some((budget, grad_norm));
        readings += 1;
    }

    // Half 2: the traced criterion evaluation.
    struct ForwardingTestLogger;
    impl log::Log for ForwardingTestLogger {
        fn enabled(&self, metadata: &log::Metadata<'_>) -> bool {
            metadata.level() <= log::Level::Debug
        }
        fn log(&self, record: &log::Record<'_>) {
            eprintln!("[{}] {}", record.level(), record.args());
        }
        fn flush(&self) {}
    }
    static FORWARDING_TEST_LOGGER: ForwardingTestLogger = ForwardingTestLogger;
    if log::set_logger(&FORWARDING_TEST_LOGGER).is_ok() {
        log::set_max_level(log::LevelFilter::Debug);
    }

    let mut term = base.clone();
    let started = std::time::Instant::now();
    let evaluated = term.penalized_quasi_laplace_criterion_with_cache_refine_policy(
        z.view(),
        &rho,
        None,
        inner_max_iter,
        learning_rate,
        ridge,
        ridge,
        true,
    );
    let wall = started.elapsed().as_secs_f64();
    match evaluated {
        Ok(value) => eprintln!("[2080-K2] criterion CONVERGED cost={:.9e} in {wall:.2}s", value.0),
        Err(err) => eprintln!("[2080-K2] criterion REFUSED in {wall:.2}s: {err}"),
    }

    assert!(
        readings > 0,
        "[2080-K2] the budget sweep produced no reading at all; the probe measured nothing"
    );
}

/// PROBE (#2080 Class-B discriminator). Is the residual the inner gate measures
/// the gradient of the objective the inner line search descends?
///
/// [`zz_measure_k2_wide_p_inner_trajectory_2080`] shows a trajectory where the
/// two disagree in SIGN: over the tail of the K=2 wide-`p` solve the penalised
/// objective falls monotonically while `‖g‖` rises monotonically, iterate after
/// iterate. Only two readings of that are possible.
///
///   * The gate's `g` IS `∇(penalized_objective_total)`, and the solve is simply
///     unable to move the stiff directions that carry it. Then a steepest-descent
///     FINITE DIFFERENCE along `−g` must realise a decrease of `h·‖g‖²` to
///     leading order, because that is what a gradient means.
///   * The gate's `g` is NOT that gradient — a different objective, a different
///     chart, or a term present on one side only. Then the finite difference
///     along `−g` will not match `h·‖g‖²`, and no amount of iteration or budget
///     can make the gate clear, because the solve is minimising one function and
///     the gate is certifying the stationarity of another.
///
/// The measurement is direct: assemble the system at the iterate the solve
/// actually reaches, read `g` off it (`row.gt` per row plus `sys.gb`, the exact
/// blocks [`sae_manifold_newton_directional_decrease`] contracts), step along
/// `−g` with [`SaeManifoldTerm::apply_newton_step`] at several `h`, and compare
/// the realised decrease against the predicted `h·‖g‖²`. The ratio is reported
/// per `h`; a consistent ratio near 1 as `h → 0` says gradient, anything else
/// says desync. Diagnostic only — it asserts that the readings are finite.
#[test]
fn zz_measure_k2_wide_p_gradient_is_the_objectives_2080() {
    let z = two_circle_wide_target(96, 96, 0.03);
    let (base, seed_dispersion) = two_circle_periodic_term(z.view(), 2, 2);
    let mode = AssignmentMode::ordered_beta_bernoulli(1.0, 1.0, false);
    let rho = SaeManifoldRho::new(0.02_f64.ln(), 1.0_f64.ln(), vec![array![0.0]; 2])
        .seed_scaled_by_dispersion_for_assignment(seed_dispersion, mode)
        .expect("seed dispersion is finite and strictly positive");

    // Read the question at BOTH ends of the trajectory: the cold seed, where the
    // solve is descending happily, and the tail plateau, where `‖g‖` and the
    // objective move in opposite directions. A desync that is present at both is
    // structural; one that appears only at the plateau is state-dependent.
    for &warmup in &[0usize, 128] {
        let mut term = base.clone();
        let mut rho_fixed = rho.clone();
        if warmup > 0 {
            term.run_joint_fit_arrow_schur_for_quasi_laplace(
                z.view(),
                &mut rho_fixed,
                None,
                warmup,
                0.04,
                1.0e-6,
                1.0e-6,
            )
            .expect("inner evidence fit must not hard-error on the K=2 wide-p rung");
        }
        let sys = term
            .assemble_arrow_schur(z.view(), &rho_fixed, None)
            .expect("reassemble at the warmed iterate");

        // `d = −g`, in the variable-stride `(row_offsets, row_dims)` layout the
        // solver's own step vectors use.
        let total_t = sys.row_offsets[sys.rows.len()];
        let mut dir_t = Array1::<f64>::zeros(total_t);
        for (row_idx, row) in sys.rows.iter().enumerate() {
            let row_base = sys.row_offsets[row_idx];
            for axis in 0..sys.row_dims[row_idx] {
                dir_t[row_base + axis] = -row.gt[axis];
            }
        }
        let mut dir_b = Array1::<f64>::zeros(sys.k);
        for idx in 0..sys.k {
            dir_b[idx] = -sys.gb[idx];
        }
        let grad_norm_sq = SaeManifoldTerm::system_grad_norm_sq(&sys);
        // Cross-check the direction against the solver's own contraction, so a
        // layout mistake here cannot be read as a desync: for `d = −g` the
        // directional decrease `−gᵀd` must be exactly `‖g‖²`.
        let contracted =
            sae_manifold_newton_directional_decrease(&sys, dir_t.view(), dir_b.view());
        let base_objective = term
            .penalized_objective_total(z.view(), &rho_fixed, None, 1.0)
            .expect("objective at the warmed iterate");
        eprintln!(
            "[2080-FD] warmup={warmup} ‖g‖²={grad_norm_sq:.9e} (−gᵀd via solver contraction \
             ={contracted:.9e}, rel_gap={:.3e}) obj={base_objective:.12e}",
            (contracted - grad_norm_sq).abs() / grad_norm_sq.max(f64::MIN_POSITIVE)
        );

        let snapshot = term.snapshot_mutable_state();
        let mut finite_readings = 0usize;
        for &h in &[1.0e-4_f64, 1.0e-5, 1.0e-6, 1.0e-7, 1.0e-8] {
            term.restore_mutable_state(&snapshot)
                .expect("restore before each finite-difference trial");
            let stepped = term
                .apply_newton_step(dir_t.view(), dir_b.view(), h)
                .and_then(|()| term.penalized_objective_total(z.view(), &rho_fixed, None, 1.0));
            match stepped {
                Ok(objective) => {
                    let realised = base_objective - objective;
                    let predicted = h * grad_norm_sq;
                    eprintln!(
                        "[2080-FD] warmup={warmup} h={h:.1e} | predicted_decrease={predicted:.9e} \
                         | realised_decrease={realised:.9e} | ratio={:.6}",
                        realised / predicted
                    );
                    if realised.is_finite() {
                        finite_readings += 1;
                    }
                }
                Err(err) => eprintln!("[2080-FD] warmup={warmup} h={h:.1e} | step failed: {err}"),
            }
        }
        term.restore_mutable_state(&snapshot)
            .expect("restore after the finite-difference sweep");
        assert!(
            finite_readings > 0,
            "[2080-FD] warmup={warmup}: no finite finite-difference reading; the probe \
             measured nothing"
        );
    }
}

/// PROBE (#2080 Class-B, PRE-REGISTERED A/B). Would a gradient-related direction
/// clear the gate that the arrow-Schur direction cannot?
///
/// [`zz_measure_k2_wide_p_gradient_is_the_objectives_2080`] establishes that the
/// gate's `g` IS `∇(penalized_objective_total)` — a steepest-descent finite
/// difference realises `h·‖g‖²` to five digits at the plateau — so descent along
/// `−g` is genuinely available there. [`zz_measure_k2_wide_p_inner_trajectory_2080`]
/// establishes that the solver's own direction `Δ` is nearly ORTHOGONAL to `g`
/// (`gᵀΔ/(‖g‖‖Δ‖) ≈ 2e-3`), which is precisely the condition under which Armijo
/// backtracking stops guaranteeing `‖g‖ → 0`.
///
/// This probe runs the two arms from ONE shared state so the comparison is not
/// across fixtures: arm A continues the production inner solve, arm B takes
/// plain Armijo steepest-descent steps. Same iterate, same iteration count, same
/// objective, same sufficient-decrease constant.
///
/// **Registered in advance:** if arm B drives `‖g‖` materially down while arm A
/// holds it flat, the defect is the DIRECTION and a gradient-related fallback is
/// the fix. If arm B's `‖g‖` also stalls, the diagnosis is incomplete and no
/// angle condition should be built on it. Diagnostic only; it asserts that both
/// arms produced readings.
#[test]
fn zz_measure_k2_wide_p_gradient_arm_vs_solver_arm_2080() {
    let z = two_circle_wide_target(96, 96, 0.03);
    let (base, seed_dispersion) = two_circle_periodic_term(z.view(), 2, 2);
    let mode = AssignmentMode::ordered_beta_bernoulli(1.0, 1.0, false);
    let rho = SaeManifoldRho::new(0.02_f64.ln(), 1.0_f64.ln(), vec![array![0.0]; 2])
        .seed_scaled_by_dispersion_for_assignment(seed_dispersion, mode)
        .expect("seed dispersion is finite and strictly positive");

    // The shared starting state: the plateau the production solve reaches and
    // then cannot leave.
    let warmup = 128usize;
    let mut shared = base.clone();
    let mut rho_fixed = rho.clone();
    shared
        .run_joint_fit_arrow_schur_for_quasi_laplace(
            z.view(),
            &mut rho_fixed,
            None,
            warmup,
            0.04,
            1.0e-6,
            1.0e-6,
        )
        .expect("inner evidence fit must not hard-error on the K=2 wide-p rung");
    let entry = shared.snapshot_mutable_state();

    let report = |tag: &str, term: &mut SaeManifoldTerm, iter: usize| -> Option<(f64, f64, f64)> {
        let sys = term.assemble_arrow_schur(z.view(), &rho_fixed, None).ok()?;
        let grad_norm_sq = SaeManifoldTerm::system_grad_norm_sq(&sys);
        let quotient = term.quotient_gradient_norm_from_system(
            &sys,
            grad_norm_sq,
            &rho_fixed.lambda_smooth_vec().ok()?,
        );
        let objective = term
            .penalized_objective_total(z.view(), &rho_fixed, None, 1.0)
            .ok()?;
        let tolerance = SAE_MANIFOLD_INNER_GRAD_REL_TOL * term.inner_iterate_scale();
        eprintln!(
            "[2080-AB] {tag} iter={iter:>4} ‖g‖={:.6e} ‖Π⊥g‖={quotient:.6e} tol={tolerance:.6e} \
             obj={objective:.12e}",
            grad_norm_sq.sqrt()
        );
        Some((grad_norm_sq.sqrt(), quotient, objective))
    };

    // ── Arm A: the production inner solve, continued from the shared state.
    let mut arm_a = shared.clone();
    let mut a_readings = 0usize;
    report("A/solver ", &mut arm_a, 0);
    for round in 1..=8usize {
        let mut rho_a = rho_fixed.clone();
        if arm_a
            .run_joint_fit_arrow_schur_for_quasi_laplace(
                z.view(),
                &mut rho_a,
                None,
                25,
                0.04,
                1.0e-6,
                1.0e-6,
            )
            .is_err()
        {
            eprintln!("[2080-AB] A/solver  round {round} hard-errored");
            break;
        }
        if report("A/solver ", &mut arm_a, round * 25).is_some() {
            a_readings += 1;
        }
    }

    // ── Arm B: plain Armijo steepest descent from the same shared state.
    let mut arm_b = shared;
    arm_b
        .restore_mutable_state(&entry)
        .expect("arm B starts from the shared entry state");
    report("B/gradient", &mut arm_b, 0);
    let mut b_readings = 0usize;
    let mut trial_step = 1.0_f64;
    for iter in 1..=200usize {
        let Ok(sys) = arm_b.assemble_arrow_schur(z.view(), &rho_fixed, None) else {
            break;
        };
        let total_t = sys.row_offsets[sys.rows.len()];
        let mut dir_t = Array1::<f64>::zeros(total_t);
        for (row_idx, row) in sys.rows.iter().enumerate() {
            let row_base = sys.row_offsets[row_idx];
            for axis in 0..sys.row_dims[row_idx] {
                dir_t[row_base + axis] = -row.gt[axis];
            }
        }
        let mut dir_b = Array1::<f64>::zeros(sys.k);
        for idx in 0..sys.k {
            dir_b[idx] = -sys.gb[idx];
        }
        let grad_norm_sq = SaeManifoldTerm::system_grad_norm_sq(&sys);
        // Normalise, so the trial length is a distance and not a gradient scale.
        let grad_norm = grad_norm_sq.sqrt();
        if !(grad_norm.is_finite() && grad_norm > 0.0) {
            break;
        }
        dir_t.mapv_inplace(|v| v / grad_norm);
        dir_b.mapv_inplace(|v| v / grad_norm);
        // Directional decrease along the UNIT gradient direction is exactly ‖g‖.
        let Ok(pre) = arm_b.penalized_objective_total(z.view(), &rho_fixed, None, 1.0) else {
            break;
        };
        let snapshot = arm_b.snapshot_mutable_state();
        let mut accepted = false;
        let mut alpha = trial_step;
        for _ in 0..=SAE_MANIFOLD_MAX_LINESEARCH_HALVINGS {
            if arm_b.restore_mutable_state(&snapshot).is_err() {
                break;
            }
            let post = arm_b
                .apply_newton_step(dir_t.view(), dir_b.view(), alpha)
                .and_then(|()| arm_b.penalized_objective_total(z.view(), &rho_fixed, None, 1.0));
            if let Ok(post) = post
                && post.is_finite()
                && post <= pre - SAE_MANIFOLD_ARMIJO_C1 * alpha * grad_norm
            {
                accepted = true;
                break;
            }
            alpha *= 0.5;
        }
        if !accepted {
            // A DISCARDED restore result means the arm silently continues from a
            // half-rolled-back state and every later reading is still attributed
            // to the B arm. `restore_mutable_state` is documented as a TOTAL
            // structural inverse that commits nothing on failure, so an error
            // here is an invariant failure of this harness's own snapshot, not a
            // recoverable condition -- and `let _ok = ...` is banned in this
            // workspace for exactly that reason.
            arm_b.restore_mutable_state(&snapshot).unwrap_or_else(|error| {
                panic!("B-arm snapshot restore after a failed line search: {error:?}")
            });
            eprintln!("[2080-AB] B/gradient line search found no acceptable step at iter={iter}");
            break;
        }
        // Ratchet the trial length the same way the production loop does.
        trial_step = (alpha * 2.0).min(1.0e2);
        if iter % 25 == 0 && report("B/gradient", &mut arm_b, iter).is_some() {
            b_readings += 1;
        }
    }

    assert!(
        a_readings > 0 && b_readings > 0,
        "[2080-AB] the A/B needs a reading from BOTH arms to be a comparison; \
         got A={a_readings}, B={b_readings}"
    );
}
