//! #2638 follow-up: gates on [`duchon_resolve_chart`], the seam that lets the
//! ψ-derivative entry point ANSWER for a cold spec instead of refusing it.
//!
//! `1f9171850` closed #2638 by routing the three ψ-derivative builders through
//! `duchon_frozen_radial_chart`, which refuses a spec carrying no frozen `V`
//! while the constrained kernel block is non-empty. That is the right answer for
//! those three builders — they receive `centers`, not `data`, so they *cannot*
//! solve the `Ω_c v = μ G_c v` eigenproblem that decides `V`.
//!
//! `build_duchon_basis_log_kappa_derivatives(data, spec)` does receive the data,
//! and its documented job is to return the ψ-jet of `build_duchon_basis(data,
//! spec)`. For a cold spec that forward build is perfectly well defined — it
//! adopts a fresh `V` — so the derivative entry point has no reason to refuse:
//! it resolves the same chart and differentiates in it.
//!
//! What is gated here:
//!
//!  1. `duchon_resolve_chart` is behaviour-preserving — `build_duchon_basis` on
//!     the RESOLVED spec reproduces the cold build bit-for-bit. That pins all
//!     five resolved decisions at once (centers, effective null-space order,
//!     seeded anisotropy, adopted `V`, identifiability transform `T`).
//!  2. `T` is read off the `V`-ROTATED design. The pre-#2638 derivative context
//!     derived it from the un-rotated one, which constrains a different function
//!     space.
//!  3. The cold-spec ψ-jet matches a finite difference of the forward taken at
//!     the resolved chart, on EVERY penalty block. Before the entry point
//!     resolved, this same call returned the raw-`Z`-chart jet: 32× the true
//!     Primary jet and 242× too small on OperatorMass (measured; see the table
//!     in the commit that added this file).
//!  4. The chart-motion decomposition itself, printed rather than asserted —
//!     the evidence that the residual #2638 reported is `|FD_cold − FD_frozen|`
//!     and not a dropped term.

#![cfg(test)]

use ndarray::Array2;

use super::*;

/// The `_no_ident` fixture from
/// `test_duchon_log_kappa_derivative_matchesfd_dim1_power1_linear_no_ident`:
/// 1-D, `power=1`, `Linear` null space, 8 farthest-point centers, no outer
/// identifiability constraint. Four penalties survive (Primary,
/// DoublePenaltyNullspace, OperatorMass, OperatorTension).
fn no_ident_fixture() -> (Array2<f64>, DuchonBasisSpec) {
    let n = 80usize;
    let mut data = Array2::<f64>::zeros((n, 1));
    for i in 0..n {
        data[[i, 0]] = i as f64 / (n as f64 - 1.0);
    }
    let spec = DuchonBasisSpec {
        radial_reparam: None,
        periodic: None,
        center_strategy: CenterStrategy::FarthestPoint { num_centers: 8 },
        length_scale: Some(1.0),
        power: 1.0,
        nullspace_order: DuchonNullspaceOrder::Linear,
        identifiability: SpatialIdentifiability::None,
        aniso_log_scales: None,
        operator_penalties: DuchonOperatorPenaltySpec::default(),
        boundary: OneDimensionalBoundary::Open,
    };
    (data, spec)
}

/// Same geometry, but with the default outer identifiability constraint, so the
/// realized `T` is non-trivial and pin 2 has something to discriminate.
fn constrained_fixture() -> (Array2<f64>, DuchonBasisSpec) {
    let (data, mut spec) = no_ident_fixture();
    spec.identifiability = SpatialIdentifiability::default();
    (data, spec)
}

/// A 2-D fixture whose auto-seeded anisotropy contrasts are non-trivial, so the
/// resolver's `auto_seed_aniso_contrasts` step is exercised rather than skipped.
fn aniso_fixture() -> (Array2<f64>, DuchonBasisSpec) {
    let n = 60usize;
    let mut data = Array2::<f64>::zeros((n, 2));
    for i in 0..n {
        let t = i as f64 / (n as f64 - 1.0);
        // Deliberately anisotropic support: axis 1 spans ~8× axis 0.
        data[[i, 0]] = t;
        data[[i, 1]] = 8.0 * (t * 7.0).sin();
    }
    let spec = DuchonBasisSpec {
        radial_reparam: None,
        periodic: None,
        center_strategy: CenterStrategy::FarthestPoint { num_centers: 10 },
        length_scale: Some(1.0),
        power: 2.0,
        nullspace_order: DuchonNullspaceOrder::Linear,
        identifiability: SpatialIdentifiability::None,
        // All-zero is the auto-seed sentinel: the forward replaces it with
        // geometry-derived contrasts, and a consumer that passes it through raw
        // builds a different kernel metric.
        aniso_log_scales: Some(vec![0.0, 0.0]),
        operator_penalties: DuchonOperatorPenaltySpec::default(),
        boundary: OneDimensionalBoundary::Open,
    };
    (data, spec)
}

fn fro(m: &Array2<f64>) -> f64 {
    m.iter().map(|v| v * v).sum::<f64>().sqrt()
}

fn duchon_metadata_chart(
    result: &BasisBuildResult,
) -> (Array2<f64>, Option<Array2<f64>>, Option<Array2<f64>>) {
    match &result.metadata {
        BasisMetadata::Duchon {
            centers,
            identifiability_transform,
            radial_reparam,
            ..
        } => (
            centers.clone(),
            radial_reparam.clone(),
            identifiability_transform.clone(),
        ),
        other => panic!(
            "expected Duchon metadata, got {:?}",
            std::mem::discriminant(other)
        ),
    }
}

/// Pin 1 + pin 2, over three fixtures that between them exercise every resolved
/// decision: the adopted `V`, a non-trivial `T`, and auto-seeded anisotropy.
#[test]
fn duchon_resolve_chart_reproduces_the_cold_build() {
    for (label, (data, spec)) in [
        ("no_ident", no_ident_fixture()),
        ("constrained", constrained_fixture()),
        ("aniso", aniso_fixture()),
    ] {
        let mut workspace = BasisWorkspace::default();
        let cold = build_duchon_basiswithworkspace(data.view(), &spec, &mut workspace)
            .unwrap_or_else(|e| panic!("[{label}] cold build: {e:?}"));
        let (cold_centers, cold_v, cold_t) = duchon_metadata_chart(&cold);

        let resolved = duchon_resolve_chart(data.view(), &spec, &mut workspace)
            .unwrap_or_else(|e| panic!("[{label}] resolve: {e:?}"));

        // The resolver must reach the SAME chart the forward reached.
        assert_eq!(
            resolved.centers, cold_centers,
            "[{label}] resolved centers differ from the cold build's"
        );
        match (&resolved.spec.radial_reparam, &cold_v) {
            (Some(a), Some(b)) => assert_eq!(
                a, b,
                "[{label}] resolved data-metric reparam V differs from the cold build's"
            ),
            (None, None) => {}
            (a, b) => panic!(
                "[{label}] reparam adoption disagrees: resolved={:?} cold={:?}",
                a.as_ref().map(|m| m.dim()),
                b.as_ref().map(|m| m.dim())
            ),
        }
        // Pin 2: `T` is a property of the ROTATED design. A `T` derived from the
        // un-rotated columns would differ here whenever `V` was adopted and the
        // spec asks for a constraint.
        match (&resolved.identifiability_transform, &cold_t) {
            (Some(a), Some(b)) => assert_eq!(
                a, b,
                "[{label}] resolved identifiability transform differs from the cold build's \
                 — it must be read off the V-rotated design"
            ),
            (None, None) => {}
            (a, b) => panic!(
                "[{label}] identifiability presence disagrees: resolved={:?} cold={:?}",
                a.as_ref().map(|m| m.dim()),
                b.as_ref().map(|m| m.dim())
            ),
        }

        // Pin 1: the resolution is behaviour-preserving. Building the RESOLVED
        // spec must reproduce the cold build exactly — same design, same
        // penalties. This is the gate that would catch a resolver which reached
        // the right `V` but the wrong effective null-space order or the wrong
        // seeded anisotropy, since either changes the realized basis.
        let replay = build_duchon_basiswithworkspace(data.view(), &resolved.spec, &mut workspace)
            .unwrap_or_else(|e| panic!("[{label}] replay build: {e:?}"));
        assert_eq!(
            replay.active_penalties.len(),
            cold.active_penalties.len(),
            "[{label}] replaying the resolved spec changed the penalty topology"
        );
        for (idx, (r, c)) in replay
            .active_penalties
            .iter()
            .zip(cold.active_penalties.iter())
            .enumerate()
        {
            assert_eq!(
                r.info.source, c.info.source,
                "[{label}] penalty {idx} source changed under replay"
            );
            // Bit-identical: the replay re-runs the same arithmetic on the same
            // inputs, so anything above exact equality would be hiding a
            // decision that was re-made rather than carried.
            assert_eq!(
                r.matrix, c.matrix,
                "[{label}] penalty {idx} ({:?}) changed under replay of the resolved spec",
                c.info.source
            );
        }
        let (replay_centers, replay_v, replay_t) = duchon_metadata_chart(&replay);
        assert_eq!(
            replay_centers, cold_centers,
            "[{label}] replay moved centers"
        );
        assert_eq!(replay_v, cold_v, "[{label}] replay re-derived a different V");
        assert_eq!(replay_t, cold_t, "[{label}] replay re-derived a different T");
    }
}

/// Pin 3: on a COLD spec the ψ-jet entry point answers, and its answer is the
/// derivative of the forward at the resolved chart — on every penalty block.
#[test]
fn duchon_cold_spec_psi_jet_matches_fd_at_the_resolved_chart() {
    // `(eps, relative arm)` per fixture, both read off
    // `zz_measure_2638_fd_step_sweep`: `eps` is that fixture's measured basin
    // bottom, the arm is ~10× the residual there. Measured bottoms are 5.6e-7
    // at eps=1e-4 (`no_ident`) and 3.4e-5 at eps=1e-5 (`constrained`) — the
    // latter sits higher because its outer identifiability transform amplifies
    // the operator blocks by ~85× (OperatorMass norm 1.68e2 against 1.97).
    // Either way the defect this gate exists for is 32×–242×, i.e. rel ≈ 1,
    // four to five orders above these bounds.
    for (label, eps, rel_arm, (data, spec)) in [
        ("no_ident", 1e-4_f64, 1e-5_f64, no_ident_fixture()),
        ("constrained", 1e-5_f64, 5e-4_f64, constrained_fixture()),
    ] {
        let mut workspace = BasisWorkspace::default();
        // The entry point must not refuse a cold spec: the forward it names is
        // well defined for one, so the jet is too.
        let jet = build_duchon_basis_log_kappa_derivatives(data.view(), &spec)
            .unwrap_or_else(|e| panic!("[{label}] cold-spec ψ-jet must build, got {e:?}"));
        let resolved = duchon_resolve_chart(data.view(), &spec, &mut workspace)
            .unwrap_or_else(|e| panic!("[{label}] resolve: {e:?}"));
        assert!(
            resolved.spec.radial_reparam.is_some(),
            "[{label}] fixture must adopt a V, otherwise this gate is vacuous"
        );

        // Central difference of the forward IN THE RESOLVED CHART.
        let kappa = 1.0
            / resolved
                .spec
                .length_scale
                .expect("hybrid Duchon length_scale");
        let mut plus_spec = resolved.spec.clone();
        let mut minus_spec = resolved.spec.clone();
        plus_spec.length_scale = Some(1.0 / (kappa * eps.exp()));
        minus_spec.length_scale = Some(1.0 / (kappa * (-eps).exp()));
        let plus = build_duchon_basiswithworkspace(data.view(), &plus_spec, &mut workspace)
            .unwrap_or_else(|e| panic!("[{label}] plus build: {e:?}"));
        let minus = build_duchon_basiswithworkspace(data.view(), &minus_spec, &mut workspace)
            .unwrap_or_else(|e| panic!("[{label}] minus build: {e:?}"));

        // The ±ε rebuilds must not have re-derived the chart — that is the
        // premise the finite difference rests on, and the premise whose absence
        // produced the whole of #2638.
        for (side, built) in [("plus", &plus), ("minus", &minus)] {
            let (c, v, t) = duchon_metadata_chart(built);
            assert_eq!(
                c, resolved.centers,
                "[{label}/{side}] ε-rebuild moved centers"
            );
            assert_eq!(
                v, resolved.spec.radial_reparam,
                "[{label}/{side}] ε-rebuild re-derived V — the FD would differentiate chart motion"
            );
            assert_eq!(
                t, resolved.identifiability_transform,
                "[{label}/{side}] ε-rebuild re-derived T"
            );
        }

        assert_eq!(
            jet.first.penalties_derivative.len(),
            plus.active_penalties.len(),
            "[{label}] ψ-jet block count does not match the forward's penalty list"
        );
        assert!(
            !jet.first.penalties_derivative.is_empty(),
            "[{label}] an empty derivative list would satisfy the loop below by absence"
        );
        for (idx, analytic) in jet.first.penalties_derivative.iter().enumerate() {
            let fd = (&plus.active_penalties[idx].matrix - &minus.active_penalties[idx].matrix)
                / (2.0 * eps);
            let err = fro(&(analytic - &fd));
            let a_norm = fro(analytic);
            let fd_norm = fro(&fd);
            eprintln!(
                "[2638:{label}] penalty {idx} ({:?}) analytic={a_norm:.6e} fd={fd_norm:.6e} \
                 err={err:.6e}",
                plus.active_penalties[idx].info.source,
            );
            // Absolute arm: 100× the measured 1e-14-per-entry rebuild roundoff,
            // in Frobenius, over the 2ε the difference divides by — the same
            // construction the `_frozen` sibling carries; it exists so a
            // legitimately-zero block (DoublePenaltyNullspace here) does not
            // divide its own roundoff. Relative arm: see the sweep note above.
            let entries = plus.active_penalties[idx].matrix.len() as f64;
            let floor = 1e2 * 1e-14 * entries.sqrt() / (2.0 * eps);
            let scale = a_norm.max(fd_norm).max(1.0);
            assert!(
                err.is_finite(),
                "[{label}] penalty {idx} residual is not finite"
            );
            assert!(
                err <= floor || err <= rel_arm * scale,
                "[{label}] penalty {idx} ({:?}) cold-spec ψ-jet disagrees with the finite \
                 difference of the forward at the RESOLVED chart: analytic={a_norm:.6e} \
                 fd={fd_norm:.6e} err={err:.6e} rel={:.6e} floor={floor:.6e} arm={rel_arm:.0e}",
                plus.active_penalties[idx].info.source,
                err / scale,
            );
        }
    }
}

/// The step sweep the gate above reads its `eps` and its relative arm off.
///
/// Printed, not asserted: it exists so the two numbers in
/// `duchon_cold_spec_psi_jet_matches_fd_at_the_resolved_chart` are measured
/// rather than chosen. A central difference has error
/// `(ε²/6)|S'''| + machine_eps·|S|/ε`, so the residual traces a V whose bottom
/// locates the best step; a gate placed a decade above that bottom is a gate on
/// the analytic derivative and not on the stencil.
#[test]
fn zz_measure_2638_fd_step_sweep() {
    for (label, (data, spec)) in [
        ("no_ident", no_ident_fixture()),
        ("constrained", constrained_fixture()),
    ] {
        let mut workspace = BasisWorkspace::default();
        let jet = build_duchon_basis_log_kappa_derivatives(data.view(), &spec).expect("jet");
        let resolved = duchon_resolve_chart(data.view(), &spec, &mut workspace).expect("resolve");
        let kappa = 1.0 / resolved.spec.length_scale.expect("ls");
        for eps in [1e-2_f64, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7] {
            let mut plus_spec = resolved.spec.clone();
            let mut minus_spec = resolved.spec.clone();
            plus_spec.length_scale = Some(1.0 / (kappa * eps.exp()));
            minus_spec.length_scale = Some(1.0 / (kappa * (-eps).exp()));
            let plus =
                build_duchon_basiswithworkspace(data.view(), &plus_spec, &mut workspace)
                    .expect("plus");
            let minus =
                build_duchon_basiswithworkspace(data.view(), &minus_spec, &mut workspace)
                    .expect("minus");
            let mut worst = 0.0_f64;
            let mut worst_idx = 0usize;
            for (idx, analytic) in jet.first.penalties_derivative.iter().enumerate() {
                let fd = (&plus.active_penalties[idx].matrix
                    - &minus.active_penalties[idx].matrix)
                    / (2.0 * eps);
                let scale = fro(analytic).max(fro(&fd)).max(1.0);
                let rel = fro(&(analytic - &fd)) / scale;
                if rel > worst {
                    worst = rel;
                    worst_idx = idx;
                }
            }
            eprintln!(
                "[2638:sweep:{label}] eps={eps:.0e} worst_rel={worst:.4e} \
                 at penalty {worst_idx} ({:?})",
                plus.active_penalties[worst_idx].info.source,
            );
        }
    }
}

/// Pin 4, printed not asserted: the decomposition that localized #2638. The
/// residual the three original gates reported is `|FD_cold − FD_frozen|` — the
/// chart moving between the ±ε rebuilds — not a dropped term in `dS/dψ`.
///
/// Deliberately not a gate. `FD_cold` differentiates a family nothing consumes
/// (the κ-optimizer replays a frozen chart at every trial), and the retained-
/// mode selection inside `thin_plate_radial_reparam_data_metric` is a discrete
/// decision, so the cold family need not even be continuous in ψ. Asserting a
/// bound on it would encode an answer about an object with no consumer.
#[test]
fn zz_measure_2638_chart_motion() {
    let (data, spec) = no_ident_fixture();
    let mut workspace = BasisWorkspace::default();
    let eps = 1e-5_f64;

    let resolved = duchon_resolve_chart(data.view(), &spec, &mut workspace).expect("resolve");
    eprintln!(
        "[2638b] resolved chart: V={:?} T={:?}",
        resolved.spec.radial_reparam.as_ref().map(|v| v.dim()),
        resolved.identifiability_transform.as_ref().map(|t| t.dim()),
    );

    let build = |s: &DuchonBasisSpec, ls: f64| {
        let mut local = s.clone();
        local.length_scale = Some(ls);
        build_duchon_basis(data.view(), &local).expect("build")
    };
    let cold_p = build(&spec, 1.0 / eps.exp());
    let cold_m = build(&spec, 1.0 / (-eps).exp());
    let frz_p = build(&resolved.spec, 1.0 / eps.exp());
    let frz_m = build(&resolved.spec, 1.0 / (-eps).exp());
    let base_cold = build_duchon_basis(data.view(), &spec).expect("cold base");
    let base_frz = build_duchon_basis(data.view(), &resolved.spec).expect("frozen base");

    let jet = build_duchon_basis_log_kappa_derivatives(data.view(), &spec).expect("jet");
    for (idx, analytic) in jet.first.penalties_derivative.iter().enumerate() {
        let fd_cold =
            (&cold_p.active_penalties[idx].matrix - &cold_m.active_penalties[idx].matrix)
                / (2.0 * eps);
        let fd_frz = (&frz_p.active_penalties[idx].matrix
            - &frz_m.active_penalties[idx].matrix)
            / (2.0 * eps);
        eprintln!(
            "[2638b] penalty {idx} ({:?}): |A|={:.4e} |FD_cold|={:.4e} |FD_frz|={:.4e} \
             |A-FD_frz|={:.4e} |FD_cold-FD_frz|={:.4e} |S_cold(0)-S_frz(0)|={:.4e}",
            base_frz.active_penalties[idx].info.source,
            fro(analytic),
            fro(&fd_cold),
            fro(&fd_frz),
            fro(&(analytic - &fd_frz)),
            fro(&(&fd_cold - &fd_frz)),
            fro(
                &(&base_cold.active_penalties[idx].matrix
                    - &base_frz.active_penalties[idx].matrix)
            ),
        );
    }
}
