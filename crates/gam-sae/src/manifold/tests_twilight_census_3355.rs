//! #3355 — the twilight zone census: the atoms the criterion neither kills
//! nor charges.
//!
//! The 10-08 receipt (`tests_dead_basin_3355`) measured that the criterion
//! VALUE still prefers the dead stratum at main (V(dead)=19.8 vs the best
//! converged live value 35.3 on the sweep fixture). This module answers the
//! question that receipt ends on: WHERE does the dead atom sit relative to the
//! two liveness boundaries the code already derives, and what predicate
//! separates it from the healthy atom?
//!
//! ## The two boundaries
//!
//! The criterion derives two distinct "is this atom real" boundaries, from two
//! distinct null models:
//!
//! 1. **The vanish boundary `S_*`** (`construction.rs::
//!    vanished_atoms_from_signal_upper_bound`): an atom is CERTIFIABLY
//!    VANISHED when its gated-signal upper bound `S_k` cannot change the
//!    residual's floating-point energy — `S_k <= S_*` with `S_*` derived from
//!    the residual reduction's backward-error floor (γ·R̄/(1−γ) scale). This
//!    boundary is about ROUNDING: below it the atom cannot even be
//!    distinguished from float noise. A violation is a categorical error
//!    (`SaeCriterionError::VanishedAtoms`, construction_quasi_laplace.rs:914).
//! 2. **The MP rank edge `e = R·(1+√(p/N_eff))²`** (`null_battery.rs::
//!    mp_reconstruction_rank_edge`): a reconstruction direction counts toward
//!    the charged rank only when its per-observation energy `μ_j > e`. This
//!    boundary is about NOISE: below it a direction is indistinguishable from
//!    an `N_eff × p` matrix of variance-`R` noise.
//!
//! ## The twilight zone (the hole)
//!
//! The two boundaries are NOT nested — they answer different questions on
//! different scales — so the set
//!
//! ```text
//! TWILIGHT(k) = { S_k > S_*  (alive: not certifiably vanished) }
//!             ∧ { μ_max,k ≤ e (below the MP edge: no direction counts)  }
//! ```
//!
//! is nonempty by construction, and #2258's promotion (`rank_charge_stratum`
//! module doc: "promoted to 1 when that count is 0 but some μ_j > 0") fills it
//! with a phantom rank: the atom is charged `½·1·edf_k·log N_eff,k` where
//! `edf_k = tr((G_k+λ_kS_k)⁻¹G_k) → 0` as λ_k → ∞. The promotion's "minimum
//! non-degenerate charge" is therefore NOT a minimum: on the smoothness rail
//! the charged dof collapses to roundoff (measured: 4.5e-8 at the dead basin)
//! while the atom still carries `S_k` above the vanish boundary. The dead
//! basin of the 10-08 receipt is exactly this zone: alive by rounding, dead by
//! noise, charged ~nothing.
//!
//! ## The census predicate
//!
//! This test classifies every atom of the sweep fixture's seeded term at TWO
//! ρ (the receipts' dead ρ and checkpoint ρ) into
//!
//!   alive-clearing  (some μ_j > e),  twilight  (none, but promoted),  vanished,
//!
//! repricing the SAME stratum producer the criterion uses
//! (`rank_charge_stratum`) and the SAME vanish proof
//! (`vanished_atoms_from_signal_upper_bound`) at each installed state, and
//! additionally reporting the noise-scale corroboration `S_k <= R̄` (the
//! atom's total signal under one residual-noise scale) for twilight atoms.
//!
//! ## Measured at main (the receipt this module froze)
//!
//! BOTH ρ sit on the promoted stratum — the true planted-circle atom at the
//! checkpoint ρ misses the MP edge by 0.8% (μmax=0.3767 vs e=0.3795), so the
//! #2258 promotion is the ONLY charge mechanism operating on this fixture,
//! and rank classification alone cannot separate the strata. What separates
//! them: (i) edf collapse — dead 4.5e-8 vs healthy 1.885 (7 orders); (ii)
//! signal content — dead `S_k` 20 orders under one residual noise scale,
//! healthy `S_k` 27× over it. The evidence predicate `S_k <= R̄` therefore
//! discriminates with ~20 orders of margin on each side, and using the
//! signal UPPER bound makes its refuse direction fail-closed: it can only
//! refuse an atom whose provable signal is under the noise scale.
//!
//! ## What a gate should do with it (the decomposition this census measures)
//!
//! * (a) **Charge-integrity floor**: promote with `r_k·edf_k >= κ > 0`.
//!   Necessary — a "rank" whose dof can be driven to 0 by λ is not a rank —
//!   but NOT sufficient as a soft floor on this fixture: with κ=1 the added
//!   charge is ½·1·log(39.7) ≈ 1.84 nats against the measured ≥15.5-nat
//!   value preference. As a hard veto (refuse `r·edf < κ`) it flips
//!   selection, but κ is uncalibrated across the 7-order gap; the null
//!   battery's false-rank Monte-Carlo rate is the natural calibrator.
//! * (b) **Evidence veto**: refuse the promoted stratum when `S_k <= R̄` —
//!   the twilight atom carries no signal above the residual's own noise
//!   scale, exactly the shape of the vanish veto one scale up. This is what
//!   flips value-based per-stratum selection on this fixture, with margin.
//!
//! Assertions are structural (the classification itself), honest at any main;
//! the printed receipt is the contribution.

use super::construction::VanishedAtomsProof;
use super::outer_objective::SaeManifoldOuterObjective;
use super::rank_charge_stratum::rank_charge_stratum;
use super::rho::SaeManifoldRho;
use super::tests::{PlantedCircleAssignmentMode, planted_circle_data, planted_circle_seed_term};
use ndarray::array;

/// The census's proposed floor on a promoted atom's charged dof. Calibrated
/// nothing yet — a placeholder between the measured dead value (4.5e-8) and
/// any healthy charge (O(1)); option (a) wants it tied to the null battery's
/// false-rank Monte-Carlo rate (`tests_wbic_rank_charge_2933`).
const TWILIGHT_DOF_FLOOR: f64 = 1.0e-3;

#[derive(Debug, Clone, Copy, PartialEq)]
enum CensusClass {
    /// Some reconstruction direction clears the MP edge: charged rank is real.
    AliveClearing,
    /// No direction clears the edge but the promotion charged rank one: the
    /// atom sits between the two boundaries. `signal_le_one_noise_scale`
    /// records whether its total signal is under one residual-noise scale.
    Twilight,
    /// Certified vanished by the signal upper bound (categorical error in
    /// production; listed here for census completeness).
    Vanished,
}

#[derive(Debug, Clone)]
struct AtomCensus {
    class: CensusClass,
    basis_size: usize,
    n_eff: f64,
    production_dof: f64,
    basis_edf: f64,
    chargeable_rank: usize,
    mp_rank: usize,
    top_energy: f64,
    mp_edge: f64,
    signal_upper_bound: Option<f64>,
    vanish_boundary: Option<f64>,
    /// `S_k <= R̄` — total gated signal under one residual mean square.
    signal_le_one_noise_scale: Option<bool>,
}

struct Census {
    atoms: Vec<AtomCensus>,
    residual_mean_square: f64,
    /// ½·Σ r_k·edf_k·log max(N_eff,k, 1) recomputed at the census's r_floor
    /// (the raw residual mean square — see the honest-scope note below).
    recomputed_rank_charge: f64,
}

fn classify(term: &super::SaeManifoldTerm, z: &ndarray::Array2<f64>, rho: &SaeManifoldRho) -> Result<Census, String> {
    // The same materialization the criterion performs
    // (construction_quasi_laplace.rs:902-910): residual, Grams, occupancies,
    // residual mean square.
    let residual = term.reconstruction_residual(z.view(), rho)?;
    let mut grams = term.empty_decoder_gram_accumulator();
    term.accumulate_decoder_gram(&mut grams)?;
    let n_eff = term.per_atom_effective_sample_size();
    let residual_energy = term.residual_energy_for_vanishing(residual.view())?;
    let rms = residual_energy.mean_square();

    // The vanish proof at the installed state, with the same inputs production
    // hands it. Unavailable is a fact, not a fallback: surface it.
    let proof = term.vanished_atoms_from_signal_upper_bound(&grams, &n_eff, rms)?;
    let (signal_bounds, vanish_boundary, vanished_set) = match &proof {
        VanishedAtomsProof::Certified {
            signal_upper_bounds,
            signal_vanish_boundary,
            atoms,
            ..
        } => (
            signal_upper_bounds.clone(),
            Some(*signal_vanish_boundary),
            atoms.as_ref().map(|a| a.iter().collect::<Vec<_>>()).unwrap_or_default(),
        ),
        VanishedAtomsProof::Unavailable { reason } => {
            return Err(format!("twilight census: vanish proof unavailable: {reason}"));
        }
    };

    let lam = rho.lambda_smooth_vec()?;
    let p_out = term.output_dim() as f64;
    let mut atoms = Vec::with_capacity(term.k_atoms());
    let mut charge = 0.0;
    for k in 0..term.k_atoms() {
        let atom = &term.atoms[k];
        let stratum = rank_charge_stratum(
            &grams[k],
            atom.decoder_coefficients(),
            n_eff[k],
            p_out,
            rms,
            lam[k],
            Some(atom.smooth_penalty()),
        )?;
        let mp_rank = stratum.mp_reconstruction_rank();
        let chargeable = stratum.production_chargeable_rank();
        let s_k = signal_bounds.get(k).copied();
        let is_vanished = vanished_set.contains(&k);
        let class = if is_vanished {
            CensusClass::Vanished
        } else if mp_rank == 0 && chargeable > 0 {
            CensusClass::Twilight
        } else {
            CensusClass::AliveClearing
        };
        let signal_le = s_k.map(|s| s <= rms);
        charge += 0.5 * chargeable as f64 * stratum_basis_edf(&stratum) * n_eff[k].max(1.0).ln();
        atoms.push(AtomCensus {
            class,
            basis_size: atom.basis_size(),
            n_eff: n_eff[k],
            production_dof: stratum.production_dof(),
            basis_edf: stratum_basis_edf(&stratum),
            chargeable_rank: chargeable,
            mp_rank,
            top_energy: stratum.top_reconstruction_energy(),
            mp_edge: stratum.mp_reconstruction_rank_edge(),
            signal_upper_bound: s_k,
            vanish_boundary,
            signal_le_one_noise_scale: signal_le,
        });
    }
    Ok(Census {
        atoms,
        residual_mean_square: rms,
        recomputed_rank_charge: charge,
    })
}

fn stratum_basis_edf(stratum: &super::rank_charge_stratum::RankChargeStratum) -> f64 {
    // production_dof = chargeable_rank * basis_edf, so recover basis_edf when
    // the rank is positive; rank-zero atoms carry no charge either way.
    let r = stratum.production_chargeable_rank();
    if r > 0 {
        stratum.production_dof() / r as f64
    } else {
        0.0
    }
}

/// Cold-start criterion evaluation + census at an explicit ρ, seeded exactly as
/// `tests_dead_basin_3355::evaluate_at` seeds it (cell 1: softmax, n=40,
/// σ=0.02; `for_assignment` binds the fitted-state coordinates).
fn census_at(
    log_lambda_sparse: f64,
    log_lambda_smooth: f64,
    log_ard: f64,
) -> Result<(f64, Census), String> {
    let z = planted_circle_data(40, 0.02);
    let (term, _seed_dispersion) =
        planted_circle_seed_term(z.view(), PlantedCircleAssignmentMode::Softmax);
    let init_rho = SaeManifoldRho::new(log_lambda_sparse, log_lambda_smooth, vec![array![log_ard]])
        .for_assignment(&term.assignment);
    let flat = init_rho.to_flat(&term.assignment)?;
    let mut objective =
        SaeManifoldOuterObjective::new(term, z.clone(), None, init_rho, 50, 0.04, 1.0e-6, 1.0e-6);
    let (value, _beta) = objective.evaluate_authoritative_criterion(flat.view())?;
    let rho = objective.current_rho.clone();
    let census = classify(&objective.term, &z, &rho)?;
    Ok((value, census))
}

fn print_census(label: &str, value: f64, census: &Census) {
    eprintln!("[#3355-twilight] === {label}: V={value:.4}, R̄={:.6e}, recomputed rank charge={:.6e}",
        census.residual_mean_square, census.recomputed_rank_charge);
    for (k, a) in census.atoms.iter().enumerate() {
        eprintln!(
            "[#3355-twilight] atom {k}: {:?} m={} n_eff={:.3} dof={:.6e} (edf={:.6e} × rank {}/mp {}) μmax={:.6e} vs e={:.6e}{}",
            a.class,
            a.basis_size,
            a.n_eff,
            a.production_dof,
            a.basis_edf,
            a.chargeable_rank,
            a.mp_rank,
            a.top_energy,
            a.mp_edge,
            match (a.signal_upper_bound, a.vanish_boundary, a.signal_le_one_noise_scale) {
                (Some(s), Some(b), Some(le)) =>
                    format!(", S_k={:.6e} vs S_*={:.6e}, S_k<=R̄: {le}", s, b),
                _ => String::new(),
            }
        );
    }
}

#[test]
fn twilight_zone_census_separates_the_dead_stratum_from_the_healthy_one_3355() {
    gam_runtime::test_support::install_diagnostic_logger();
    // The receipts' ρ (cell 1: softmax, n=40, σ=0.02): the old base's
    // certified dead terminal state, and the main run's checkpoint
    // neighbourhood.
    let (dead_v, dead) = census_at(-3.912, 21.309, 3.681)
        .expect("the dead rho must evaluate and classify: the basin is reachable (10-08 receipt)");
    let (healthy_v, healthy) = census_at(2.507, -1.061, 3.681)
        .expect("the healthy rho must evaluate and classify");

    print_census("dead rho (-3.912, 21.309)", dead_v, &dead);
    print_census("healthy rho (2.507, -1.061)", healthy_v, &healthy);

    // 1. The dead atom is NOT certified vanished: the vanish boundary is a
    //    rounding scale, and e^{-42}-shrunk decoders sit above it. This is the
    //    hole — production's categorical liveness veto cannot fire here.
    assert!(
        !dead.atoms.iter().any(|a| a.class == CensusClass::Vanished),
        "the dead atom must not be certified vanished (it was priced, so the \
         criterion's own vanish proof returned atoms: None there)"
    );

    // 2. At the dead ρ at least one atom is in the twilight zone: alive by
    //    the vanish bound, below the MP edge, promoted to a phantom rank whose
    //    charged dof is under the census floor.
    let dead_twilight: Vec<&AtomCensus> = dead
        .atoms
        .iter()
        .filter(|a| a.class == CensusClass::Twilight)
        .collect();
    assert!(
        !dead_twilight.is_empty(),
        "the dead rho must place at least one atom in the twilight zone \
         (alive per vanish, mp_rank 0, promoted)"
    );
    for a in &dead_twilight {
        assert!(
            a.production_dof < TWILIGHT_DOF_FLOOR,
            "twilight atom's charged dof {:.6e} must sit under the census floor \
             {TWILIGHT_DOF_FLOOR} (the promotion's charge collapsed with edf)",
            a.production_dof
        );
        assert_eq!((a.chargeable_rank, a.mp_rank), (1, 0));
        // The noise-scale corroboration: the atom's total gated signal is
        // under one residual mean square — option (b)'s veto predicate.
        assert_eq!(
            a.signal_le_one_noise_scale,
            Some(true),
            "the twilight atom's signal upper bound must sit under one residual \
             noise scale (the evidence-veto predicate)"
        );
    }

    // 3. The census DISCRIMINATES on both axes: at the healthy ρ every
    //    occupied atom charges real dof above the floor AND carries signal
    //    above one residual noise scale (the receipt: the healthy atom is
    //    ALSO promoted — μmax misses the MP edge by 0.8% at n=40 — so rank
    //    classification alone cannot separate the strata; edf and signal do).
    for (k, a) in healthy.atoms.iter().enumerate() {
        if a.n_eff > 0.0 {
            assert!(
                a.production_dof >= TWILIGHT_DOF_FLOOR,
                "healthy atom {k} charges dof {:.6e} above the floor: the census \
                 must not classify the healthy stratum as collapsed",
                a.production_dof
            );
            assert_eq!(
                a.signal_le_one_noise_scale,
                Some(false),
                "healthy atom {k} must carry signal above one residual noise scale: \
                 the evidence predicate discriminates the strata (dead: 20 orders \
                 under, healthy: 27x over)"
            );
        }
    }

    // 4. The charge-integrity quantification for option (a): a unit floor
    //    adds only ½·1·log max(N_eff,1) per twilight atom — printed, not
    //    asserted, because which κ is a design decision.
    let unit_floor_gain: f64 = dead_twilight
        .iter()
        .map(|a| 0.5 * (1.0 - a.production_dof).max(0.0) * a.n_eff.max(1.0).ln())
        .sum();
    eprintln!(
        "[#3355-twilight] option (a) unit-dof floor would add {unit_floor_gain:.4} nats \
         against the measured value preference (V(dead)={dead_v:.4} vs V(healthy)={healthy_v:.4})"
    );
}
