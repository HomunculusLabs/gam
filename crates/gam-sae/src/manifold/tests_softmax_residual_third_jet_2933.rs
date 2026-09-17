// `manifold/mod.rs` declares this module only as
// `#[cfg(test)] mod tests_softmax_residual_third_jet_2933;`.
#![cfg(test)]
//! #2933 F01 — under a softmax gate the θ-derivative of the exact observed
//! information must carry the residual THIRD derivative `⟨M r, ∂³f_abw⟩` of the
//! whole reconstruction `f = Σ_k a_k(ℓ)·γ_k(t_k)`, cross-atom gate terms included.
//!
//! `patchd_residual_third_leg` returned zero for any triple with mixed atom labels
//! and for any logit unless the gate was ordered Beta–Bernoulli, while the outer
//! gradient of a softmax fit reaches both production θ-adjoints
//! (`dense_exact_a_logdet_channels` and the bundle's
//! `logdet_theta_adjoint_from_probes`) with the data target that switches the leg
//! on. So `∂_ℓj ∂²_tk f = ∂a_k/∂ℓ_j·γ_k''`, nonzero for `j ≠ k`, never reached `dA`.
//!
//! The arbiters are central differences of the MATERIALIZED dense `A` at endpoints
//! moved by `apply_newton_step`, each with its own `B` factor, so they are free to
//! disagree with every analytic leg. Each is paired with the same contraction run
//! without the target (`residual_target = None` skips the residual third leg) as a
//! positive control, which must miss the difference by far more than the analytic
//! side is allowed to.

use super::*;
use ndarray::{Array1, Array2, s};

/// Softmax temperature, deliberately not one.
const TAU: f64 = 0.7;
/// Coarsest central-difference step; Richardson uses `h`, `h/2` and `h/4`.
const STEP: f64 = 1.0e-3;
/// Rows whose `t` slots are differenced.
const PROBE_ROWS: [usize; 2] = [0, 3];
/// Amplitude of the target's offset from the fixture's reconstruction.
const RESIDUAL_AMPLITUDE: f64 = 0.05;

/// Four periodic atoms under a softmax gate: three free logits per row (so a triple
/// of three DISTINCT logits exists), unequal gates, harmonic curves whose third
/// coordinate derivative does not vanish, a target offset from the reconstruction
/// (so `M r ≠ 0`), and live sparsity, smoothness and ARD priors.
///
/// Every row factor stays off the deflation stratum by construction:
/// - Every coordinate lies within a sixth of a period of the chart origin, where the
///   periodic ARD curvature `α·cos(2πt) ≥ α/2` is positive and its majorizer
///   `α·softplus_{τ₀}(cos 2πt)` equals it, so every coordinate direction carries
///   prior curvature whatever the data rank. On the concave half that majorizer is
///   numerically zero, and a row with more such coordinates than its data Jacobian
///   resolves has an exactly singular `H_tt`.
/// - The decoders' phase frequency in the output column depends on the basis
///   column, so the decoded curves do not all lie in one plane of the outputs, as
///   they would for a decoder that is a sum of two outer products.
/// - The residual is small beside that curvature.
///
/// The premises in the tests check the outcome instead of assuming it.
fn softmax_residual_fixture() -> (SaeManifoldTerm, Array2<f64>, SaeManifoldRho) {
    let n = 6usize;
    let p = 3usize;
    let k_atoms = 4usize;
    let m = 3usize;
    let evaluator = Arc::new(
        PeriodicHarmonicEvaluator::new(m)
            .expect("m=3 is odd and positive, the basis width this evaluator requires"),
    );
    let mut logits = Array2::<f64>::zeros((n, k_atoms));
    let mut coords: Vec<Array2<f64>> =
        (0..k_atoms).map(|_| Array2::<f64>::zeros((n, 1))).collect();
    for row in 0..n {
        for atom in 0..k_atoms {
            let t = (0.2 * (row as f64 + 0.35) / n as f64 + 0.04 * atom as f64 - 0.15)
                .rem_euclid(1.0);
            assert!(
                (2.0 * std::f64::consts::PI * t).cos() > 0.0,
                "#2933 F01 premise: coordinate {t} of atom {atom} in row {row} must sit on the \
                 convex half of the periodic ARD prior"
            );
            coords[atom][[row, 0]] = t;
            logits[[row, atom]] = 0.9 * (1.3 * row as f64 + 0.8 * atom as f64 + 0.2).sin();
        }
    }
    let mut target = Array2::<f64>::zeros((n, p));
    for row in 0..n {
        for out_col in 0..p {
            target[[row, out_col]] =
                RESIDUAL_AMPLITUDE * (((row * 7 + out_col * 3) as f64) * 0.7).sin();
        }
    }
    let mut atoms = Vec::with_capacity(k_atoms);
    for atom in 0..k_atoms {
        let (phi, jet) = evaluator
            .evaluate(coords[atom].view())
            .expect("fixture coords are finite points of the unit-period circle chart");
        // The phase's frequency in `out_col` depends on `basis_col`, so the rows of
        // the decoder are not confined to one plane of the outputs.
        let decoder = Array2::from_shape_fn((m, p), |(basis_col, out_col)| {
            let (basis_col, out_col) = (basis_col as f64, out_col as f64);
            0.4 * (1.7 * atom as f64 + 1.1 * basis_col + (0.6 + 0.8 * basis_col) * out_col + 0.3)
                .sin()
        });
        let decoded = phi.dot(&decoder);
        for row in 0..n {
            let gate = softmax_row(logits.row(row), TAU)[atom];
            for out_col in 0..p {
                target[[row, out_col]] += gate * decoded[[row, out_col]];
            }
        }
        atoms.push(
            SaeManifoldAtom::new_with_provided_function_gram(
                format!("softmax_third_{atom}"),
                SaeAtomBasisKind::Periodic,
                1,
                phi,
                jet,
                decoder,
                Array2::<f64>::eye(m),
            )
            .expect("decoder is (m, p) and phi/jet come from the same evaluator")
            .with_basis_second_jet(evaluator.clone()),
        );
    }
    let mode = AssignmentMode::softmax(TAU);
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        logits,
        coords,
        vec![LatentManifold::Circle { period: 1.0 }; k_atoms],
        mode,
    )
    .expect("k_atoms coord blocks and k_atoms manifolds match the logits' k_atoms columns");
    let term = SaeManifoldTerm::new(atoms, assignment)
        .expect("the assignment was built with exactly one block per atom");
    let rho = SaeManifoldRho::new(-1.0, -1.0, vec![Array1::from_vec(vec![-1.0]); k_atoms])
        .for_assignment(&term.assignment);
    (term, target, rho)
}

/// The undamped `B` factor at the term's CURRENT state with no inner iteration: the
/// fixed-state operator every `A = B + ΔC` below is materialized against.
fn fixed_state_cache(
    term: &mut SaeManifoldTerm,
    target: &Array2<f64>,
    rho: &SaeManifoldRho,
) -> ArrowFactorCache {
    let sys = term
        .assemble_arrow_schur(target.view(), rho, None)
        .expect("arrow-Schur assembly at a fixture state");
    let options = ArrowSolveOptions::direct().with_positive_definite_evidence();
    let (_dt, _db, cache) = solve_arrow_newton_step_with_options(&sys, 0.0, 0.0, &options)
        .expect("undamped B factor at a fixture state");
    cache
}

/// The anchor and its cache, with the three per-assembly lagged gates pinned
/// (#2828): the objective whose Hessian `A` is holds them fixed, so a difference
/// across endpoints that re-derived their own gates would belong to no operator.
fn anchored_state(
    term: &SaeManifoldTerm,
    target: &Array2<f64>,
    rho: &SaeManifoldRho,
) -> (SaeManifoldTerm, ArrowFactorCache) {
    let mut anchor = term.clone();
    let cache = fixed_state_cache(&mut anchor, target, rho);
    anchor.streaming_gates_frozen = true;
    (anchor, cache)
}

fn deflated_direction_count(term: &SaeManifoldTerm, cache: &ArrowFactorCache) -> usize {
    (0..term.n_obs())
        .map(|row| cache.deflated_row_directions[row].len())
        .sum()
}

/// #2933 F01 premise — every row block of the materialized exact `A` is positive
/// definite, the property the fixture builds in. The deflation premises at the
/// anchor and at every finite-difference endpoint separately check that no row
/// factor changes stratum.
fn assert_exact_row_blocks_positive_definite(
    anchor: &SaeManifoldTerm,
    target: &Array2<f64>,
    rho: &SaeManifoldRho,
    cache: &ArrowFactorCache,
) {
    let a = anchor
        .materialize_exact_hessian_dense(rho, target.view(), cache)
        .expect("dense exact A at the anchor");
    for row in 0..anchor.n_obs() {
        let q = cache.row_dims[row];
        let base = cache.row_offsets[row];
        let block = a.slice(s![base..base + q, base..base + q]).to_owned();
        let (eigenvalues, _) = block
            .eigh(Side::Lower)
            .expect("a row block of the symmetric exact A");
        let smallest = eigenvalues.iter().copied().fold(f64::INFINITY, f64::min);
        let largest = eigenvalues.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        eprintln!(
            "[#2933 F01 premise] row={row} exact A_tt eigenvalues in [{smallest:.6e}, {largest:.6e}]"
        );
        assert!(
            smallest > 0.0,
            "#2933 F01 premise: row {row}'s exact A_tt must be positive definite by construction \
             (eigenvalues in [{smallest:.3e}, {largest:.3e}])"
        );
    }
}

fn log_abs_det(a: &Array2<f64>) -> f64 {
    let (eigenvalues, _) = a.eigh(Side::Lower).expect("the dense exact A is symmetric");
    eigenvalues.iter().map(|lambda| lambda.abs().ln()).sum()
}

/// The dense exact `A` at the anchor moved by `signed_step` along the unit `t` slot
/// `slot`, which is the variable `var` of `row`.
fn endpoint_exact_a(
    anchor: &SaeManifoldTerm,
    target: &Array2<f64>,
    rho: &SaeManifoldRho,
    anchor_cache: &ArrowFactorCache,
    row: usize,
    slot: usize,
    var: SaeLocalRowVar,
    signed_step: f64,
) -> Array2<f64> {
    let mut endpoint = anchor.clone();
    endpoint.decoder_repulsion_gate = anchor.decoder_repulsion_gate.clone();
    endpoint.barrier_coactivation_gate = anchor.barrier_coactivation_gate.clone();
    endpoint.amplitude_barrier_gate = anchor.amplitude_barrier_gate;
    endpoint.streaming_gates_frozen = true;
    // `apply_newton_step` refuses a non-positive step size, so the sign rides on
    // the direction.
    let mut direction = Array1::<f64>::zeros(anchor_cache.delta_t_len());
    direction[slot] = signed_step.signum();
    endpoint
        .apply_newton_step(
            direction.view(),
            Array1::<f64>::zeros(anchor_cache.k).view(),
            signed_step.abs(),
        )
        .expect("finite-difference endpoint step");
    let moved = match var {
        SaeLocalRowVar::Logit { atom } => {
            endpoint.assignment.logits[[row, atom]] - anchor.assignment.logits[[row, atom]]
        }
        SaeLocalRowVar::Coord { atom, axis } => {
            endpoint.assignment.coords[atom].row(row)[axis]
                - anchor.assignment.coords[atom].row(row)[axis]
        }
    };
    assert!(
        (moved - signed_step).abs() <= 1.0e-12,
        "#2933 F01 premise: slot {slot} of row {row} is {var:?}, but a step of {signed_step:e} \
         moved that variable by {moved:e}; the difference would not be in the variable the \
         adjoint names"
    );
    let cache = fixed_state_cache(&mut endpoint, target, rho);
    assert_eq!(
        deflated_direction_count(&endpoint, &cache),
        0,
        "#2933 F01 premise: a finite-difference endpoint must stay deflation-free, or the two \
         endpoints straddle a discrete deflation change and their difference is no derivative"
    );
    endpoint
        .materialize_exact_hessian_dense(rho, target.view(), &cache)
        .expect("dense exact A at a finite-difference endpoint")
}

/// Central differences, Richardson-extrapolated over `h, h/2, h/4`, of the dense
/// exact `A` and of `log|det A|` along one `t` slot. Each oracle is the finer
/// extrapolation minus the coarser one, charged to the same budget as the gap.
struct SlotDifference {
    d_a: Array2<f64>,
    d_a_oracle: Array2<f64>,
    d_log_det: f64,
    d_log_det_oracle: f64,
}

fn slot_difference(
    anchor: &SaeManifoldTerm,
    target: &Array2<f64>,
    rho: &SaeManifoldRho,
    cache: &ArrowFactorCache,
    row: usize,
    slot: usize,
    var: SaeLocalRowVar,
) -> SlotDifference {
    let mut d_a = Vec::with_capacity(3);
    let mut d_log_det = Vec::with_capacity(3);
    for divisor in [1.0_f64, 2.0, 4.0] {
        let h = STEP / divisor;
        let plus = endpoint_exact_a(anchor, target, rho, cache, row, slot, var, h);
        let minus = endpoint_exact_a(anchor, target, rho, cache, row, slot, var, -h);
        d_a.push((&plus - &minus) / (2.0 * h));
        d_log_det.push((log_abs_det(&plus) - log_abs_det(&minus)) / (2.0 * h));
    }
    let fine = (&d_a[2] * 4.0 - &d_a[1]) / 3.0;
    let coarse = (&d_a[1] * 4.0 - &d_a[0]) / 3.0;
    let fine_log_det = (4.0 * d_log_det[2] - d_log_det[1]) / 3.0;
    let coarse_log_det = (4.0 * d_log_det[1] - d_log_det[0]) / 3.0;
    SlotDifference {
        d_a_oracle: (&fine - &coarse).mapv(f64::abs),
        d_a: fine,
        d_log_det: fine_log_det,
        d_log_det_oracle: (fine_log_det - coarse_log_det).abs(),
    }
}

/// The product-rule term of `∂³(a_k·γ_k)` a triple of row variables selects, and
/// whether a logit in it belongs to a different atom than its coordinates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TripleKind {
    ThreeLogits { distinct: bool },
    TwoLogitsOneCoordinate { cross_atom: bool },
    OneLogitTwoCoordinates { cross_atom: bool },
    ThreeCoordinates,
    CoordinatesOfTwoAtoms,
}

fn triple_kind(vars: [SaeLocalRowVar; 3]) -> TripleKind {
    let mut logit_atoms = Vec::with_capacity(3);
    let mut coord_atoms = Vec::with_capacity(3);
    for var in vars {
        match var {
            SaeLocalRowVar::Logit { atom } => logit_atoms.push(atom),
            SaeLocalRowVar::Coord { atom, .. } => coord_atoms.push(atom),
        }
    }
    if coord_atoms.iter().any(|&atom| atom != coord_atoms[0]) {
        return TripleKind::CoordinatesOfTwoAtoms;
    }
    let cross_atom = coord_atoms
        .first()
        .is_some_and(|&owner| logit_atoms.iter().any(|&atom| atom != owner));
    match coord_atoms.len() {
        0 => TripleKind::ThreeLogits {
            distinct: logit_atoms[0] != logit_atoms[1]
                && logit_atoms[1] != logit_atoms[2]
                && logit_atoms[0] != logit_atoms[2],
        },
        1 => TripleKind::TwoLogitsOneCoordinate { cross_atom },
        2 => TripleKind::OneLogitTwoCoordinates { cross_atom },
        _ => TripleKind::ThreeCoordinates,
    }
}

struct CategoryReadout {
    count: usize,
    worst_gap: f64,
    worst_control_gap: f64,
    largest_difference: f64,
}

/// #2933 F01 — every row-local entry `∂A_ab/∂θ_w` the dense exact-A θ-adjoint
/// contracts, against a central difference of the materialized `A`.
///
/// A unit inverse `e_b e_aᵀ` turns `Γ_w = Σ inv[b,a]·dh_ab(w)` into the single entry
/// `∂A_ab/∂θ_w`, so every triple `(a, b, w)` of a probe row is read separately and
/// reported in its product-rule category.
#[test]
fn softmax_exact_a_derivative_entries_match_finite_difference_2933() {
    const ENTRY_TOL: f64 = 1.0e-6;
    let (term, target, rho) = softmax_residual_fixture();
    let (anchor, cache) = anchored_state(&term, &target, &rho);
    assert_eq!(
        deflated_direction_count(&anchor, &cache),
        0,
        "#2933 F01 premise: the anchor must be deflation-free"
    );
    assert_exact_row_blocks_positive_definite(&anchor, &target, &rho, &cache);
    let dim = cache.delta_t_len() + cache.k;
    let mut readouts: Vec<(TripleKind, CategoryReadout)> = Vec::new();
    for row in PROBE_ROWS {
        let q = cache.row_dims[row];
        let base = cache.row_offsets[row];
        let vars = anchor
            .row_vars_for_cache_row(row, &cache)
            .expect("row variables of a probe row");
        let differences: Vec<SlotDifference> = (0..q)
            .map(|w| slot_difference(&anchor, &target, &rho, &cache, row, base + w, vars[w]))
            .collect();
        for a in 0..q {
            for b in a..q {
                let mut unit = Array2::<f64>::zeros((dim, dim));
                unit[[base + b, base + a]] = 1.0;
                let with_target = anchor
                    .logdet_theta_adjoint_dense(&rho, &cache, &unit, true, true, Some(target.view()))
                    .expect("dense exact-A contraction with the data target");
                let without_target = anchor
                    .logdet_theta_adjoint_dense(&rho, &cache, &unit, true, true, None)
                    .expect("dense exact-A contraction without the data target");
                for w in 0..q {
                    let fd = differences[w].d_a[[base + a, base + b]];
                    let oracle = differences[w].d_a_oracle[[base + a, base + b]];
                    let analytic = with_target.t[base + w];
                    let control = without_target.t[base + w];
                    let scale = 1.0 + fd.abs();
                    let gap = ((analytic - fd).abs() + oracle) / scale;
                    let control_gap = (control - fd).abs() / scale;
                    let kind = triple_kind([vars[a], vars[b], vars[w]]);
                    eprintln!(
                        "[#2933 F01 dA] row={row} a={:?} b={:?} w={:?} kind={kind:?} fd={fd:.9e} \
                         analytic={analytic:.9e} without_target={control:.9e} gap={gap:.3e} \
                         control_gap={control_gap:.3e} oracle={oracle:.3e}",
                        vars[a], vars[b], vars[w],
                    );
                    let position = match readouts.iter().position(|(seen, _)| *seen == kind) {
                        Some(position) => position,
                        None => {
                            readouts.push((
                                kind,
                                CategoryReadout {
                                    count: 0,
                                    worst_gap: 0.0,
                                    worst_control_gap: 0.0,
                                    largest_difference: 0.0,
                                },
                            ));
                            readouts.len() - 1
                        }
                    };
                    let readout = &mut readouts[position].1;
                    readout.count += 1;
                    readout.worst_gap = readout.worst_gap.max(gap);
                    readout.worst_control_gap = readout.worst_control_gap.max(control_gap);
                    readout.largest_difference = readout.largest_difference.max(fd.abs());
                }
            }
        }
    }
    let summary: Vec<String> = readouts
        .iter()
        .map(|(kind, readout)| {
            format!(
                "{kind:?}: n={} worst_gap={:.3e} worst_control_gap={:.3e} max|fd|={:.3e}",
                readout.count,
                readout.worst_gap,
                readout.worst_control_gap,
                readout.largest_difference,
            )
        })
        .collect();
    for line in &summary {
        eprintln!("[#2933 F01 dA summary] {line}");
    }
    let worst_gap = readouts
        .iter()
        .map(|(_, readout)| readout.worst_gap)
        .fold(0.0_f64, f64::max);
    assert!(
        worst_gap <= ENTRY_TOL,
        "#2933 F01: the dense exact-A θ-adjoint's dA entries do not match a central difference of \
         the materialized A (worst relative gap {worst_gap:.3e} > {ENTRY_TOL:.1e}); per category: \
         {summary:#?}"
    );
    for required in [
        TripleKind::ThreeLogits { distinct: true },
        TripleKind::ThreeLogits { distinct: false },
        TripleKind::TwoLogitsOneCoordinate { cross_atom: true },
        TripleKind::TwoLogitsOneCoordinate { cross_atom: false },
        TripleKind::OneLogitTwoCoordinates { cross_atom: true },
        TripleKind::OneLogitTwoCoordinates { cross_atom: false },
        TripleKind::ThreeCoordinates,
    ] {
        let readout = readouts
            .iter()
            .find(|(kind, _)| *kind == required)
            .map(|(_, readout)| readout)
            .unwrap_or_else(|| panic!("#2933 F01: no triple of kind {required:?} was probed"));
        assert!(
            readout.worst_control_gap >= 100.0 * ENTRY_TOL,
            "#2933 F01 positive control: dropping the residual third leg must visibly break the \
             {required:?} entries, but the target-free contraction still matched to \
             {:.3e}; this gate cannot see that leg there. Per category: {summary:#?}",
            readout.worst_control_gap,
        );
    }
}

/// #2933 F01 — `Γ_w = tr(A⁻¹ ∂A/∂θ_w)`, the dense exact-A θ-adjoint contracted with
/// the exact inverse of the materialized `A`, against a central difference of
/// `log|det A|`: the derivative of the ranked `½log|A|` in each probe-row slot.
#[test]
fn softmax_exact_a_logdet_theta_adjoint_matches_finite_difference_2933() {
    const TRACE_TOL: f64 = 1.0e-6;
    let (term, target, rho) = softmax_residual_fixture();
    let (anchor, cache) = anchored_state(&term, &target, &rho);
    assert_eq!(
        deflated_direction_count(&anchor, &cache),
        0,
        "#2933 F01 premise: the anchor must be deflation-free"
    );
    assert_exact_row_blocks_positive_definite(&anchor, &target, &rho, &cache);
    let a = anchor
        .materialize_exact_hessian_dense(&rho, target.view(), &cache)
        .expect("dense exact A at the anchor");
    let (eigenvalues, eigenvectors) = a.eigh(Side::Lower).expect("the dense exact A is symmetric");
    let spectral_norm = eigenvalues.iter().map(|l| l.abs()).fold(0.0_f64, f64::max);
    let smallest = eigenvalues.iter().map(|l| l.abs()).fold(f64::INFINITY, f64::min);
    let negative = eigenvalues.iter().filter(|&&l| l < 0.0).count();
    eprintln!(
        "[#2933 F01 tr] dim={} ‖A‖₂={spectral_norm:.6e} min|λ|={smallest:.6e} negative={negative}",
        a.nrows()
    );
    assert!(
        smallest > 1.0e-8 * spectral_norm,
        "#2933 F01 premise: A must be nonsingular for log|det A| to be differentiable \
         (min|λ|={smallest:.3e}, ‖A‖₂={spectral_norm:.3e})"
    );
    let inverse = eigenvectors
        .dot(&Array2::from_diag(&eigenvalues.mapv(|l| 1.0 / l)))
        .dot(&eigenvectors.t());
    let with_target = anchor
        .logdet_theta_adjoint_dense(&rho, &cache, &inverse, true, true, Some(target.view()))
        .expect("dense exact-A θ-adjoint with the data target");
    let without_target = anchor
        .logdet_theta_adjoint_dense(&rho, &cache, &inverse, true, true, None)
        .expect("dense exact-A θ-adjoint without the data target");
    let mut worst_gap = 0.0_f64;
    let mut worst_logit_control_gap = 0.0_f64;
    for row in PROBE_ROWS {
        let q = cache.row_dims[row];
        let base = cache.row_offsets[row];
        let vars = anchor
            .row_vars_for_cache_row(row, &cache)
            .expect("row variables of a probe row");
        for w in 0..q {
            let difference = slot_difference(&anchor, &target, &rho, &cache, row, base + w, vars[w]);
            let fd = difference.d_log_det;
            let analytic = with_target.t[base + w];
            let control = without_target.t[base + w];
            let scale = 1.0 + fd.abs();
            let gap = ((analytic - fd).abs() + difference.d_log_det_oracle) / scale;
            let control_gap = (control - fd).abs() / scale;
            eprintln!(
                "[#2933 F01 tr] row={row} w={:?} fd={fd:.9e} analytic={analytic:.9e} \
                 without_target={control:.9e} gap={gap:.3e} control_gap={control_gap:.3e} \
                 oracle={:.3e}",
                vars[w], difference.d_log_det_oracle,
            );
            worst_gap = worst_gap.max(gap);
            if matches!(vars[w], SaeLocalRowVar::Logit { .. }) {
                worst_logit_control_gap = worst_logit_control_gap.max(control_gap);
            }
        }
    }
    assert!(
        worst_gap <= TRACE_TOL,
        "#2933 F01: tr(A⁻¹ ∂A/∂θ) from the dense exact-A θ-adjoint does not match a central \
         difference of log|det A| (worst relative gap {worst_gap:.3e} > {TRACE_TOL:.1e})"
    );
    assert!(
        worst_logit_control_gap >= 100.0 * TRACE_TOL,
        "#2933 F01 positive control: dropping the residual third leg must visibly move the logit \
         slots of tr(A⁻¹ ∂A/∂θ), but it still matched to {worst_logit_control_gap:.3e}"
    );
}

/// #2933 F01 — the bundle / matrix-free θ-adjoint `logdet_theta_adjoint_from_probes`
/// receives the data target in production
/// (`analytic_outer_rho_gradient_components_with_bundle`) and must carry the same
/// residual third leg as the dense route. At full-basis probes it reconstructs `B⁻¹`
/// exactly, so the leg's contribution (the adjoint with the target minus the adjoint
/// without it) must equal the dense route's on the same inverse, and must be far
/// from zero.
///
/// This is route parity, not the defect's arbiter: both towers call one
/// `patchd_residual_third_leg`, so a leg missing from that helper is missing from
/// both and the parity still holds (it did at F01's parent). The two
/// central-difference tests above arbitrate the leg on the dense route; this test
/// carries their verdict to the from-probes route.
#[test]
fn from_probes_residual_third_leg_matches_the_dense_route_2933() {
    let (term, target, rho) = softmax_residual_fixture();
    let (anchor, cache) = anchored_state(&term, &target, &rho);
    assert_eq!(
        deflated_direction_count(&anchor, &cache),
        0,
        "#2933 F01 premise: the anchor must be deflation-free"
    );
    assert_exact_row_blocks_positive_definite(&anchor, &target, &rho, &cache);
    assert!(
        cache.beta_schur_conditioning.is_none(),
        "#2933 F01 premise: a reduced-Schur basin reweights the from-probes row inverse, so the \
         two routes would contract the leg against different inverses"
    );
    let solver = DeflatedArrowSolver::plain(&cache);
    let inverse = anchor
        .materialize_joint_inverse(&cache, &solver)
        .expect("dense joint inverse of B");
    let k = cache.k;
    let sqrt_k = (k as f64).sqrt();
    let probes: Vec<Array1<f64>> = (0..k)
        .map(|j| {
            let mut probe = Array1::<f64>::zeros(k);
            probe[j] = sqrt_k;
            probe
        })
        .collect();
    let sinv: Vec<Array1<f64>> = probes
        .iter()
        .map(|probe| {
            cache
                .schur_inverse_apply(probe.view())
                .expect("exact reduced-Schur solve at a full-basis probe")
        })
        .collect();
    let flat = |v: &SaeArrowVector| -> Vec<f64> { v.t.iter().chain(v.beta.iter()).copied().collect() };
    let leg = |with: &SaeArrowVector, without: &SaeArrowVector| -> Vec<f64> {
        flat(with)
            .into_iter()
            .zip(flat(without))
            .map(|(x, y)| x - y)
            .collect()
    };
    let dense_leg = leg(
        &anchor
            .logdet_theta_adjoint_dense(&rho, &cache, &inverse, true, true, Some(target.view()))
            .expect("dense exact-A θ-adjoint with the data target"),
        &anchor
            .logdet_theta_adjoint_dense(&rho, &cache, &inverse, true, true, None)
            .expect("dense exact-A θ-adjoint without the data target"),
    );
    let probes_leg = leg(
        &anchor
            .logdet_theta_adjoint_from_probes(
                &rho,
                &cache,
                &probes,
                &sinv,
                EvidenceOperator::ExactObservedInformation,
                Some(target.view()),
            )
            .expect("from-probes exact-A θ-adjoint with the data target"),
        &anchor
            .logdet_theta_adjoint_from_probes(
                &rho,
                &cache,
                &probes,
                &sinv,
                EvidenceOperator::ExactObservedInformation,
                None,
            )
            .expect("from-probes exact-A θ-adjoint without the data target"),
    );
    let largest = dense_leg.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
    let gap = dense_leg
        .iter()
        .zip(&probes_leg)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max);
    eprintln!("[#2933 F01 probes] max|dense leg|={largest:.6e} max|probes − dense|={gap:.6e}");
    assert!(
        largest >= 1.0e-3,
        "#2933 F01: the residual third leg must carry real weight on this fixture \
         (max|leg|={largest:.3e}), or the parity below is a zero-vs-zero comparison"
    );
    assert!(
        gap <= 1.0e-9 * (1.0 + largest),
        "#2933 F01: the from-probes θ-adjoint's residual third leg differs from the dense route's \
         on the same inverse by {gap:.3e} (leg scale {largest:.3e})"
    );
}
