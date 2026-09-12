//! #2333 — rows the exact-A evidence factor prices as a clamp basin.
//!
//! `factor_spectral_deflated_criterion_row_with_geometry` prices an
//! `ExactADirectionClassification::ClampBasin` direction at its basin curvature
//! without unit-deflating it, so the row reaches the cache with a recorded,
//! non-identity spectrum and an EMPTY direction list. Every ρ/θ trace that
//! contracts the row's raw derivative must subtract `deflation_block_correction`
//! there. Before `SaeManifoldTerm::row_deflation_is_live`, the θ-adjoints, the
//! dense assignment log-strength trace and the ARD traces gated on the direction
//! list and skipped those rows.
//!
//! The arbiter reads the SAME geometry-factored cache as the producer under test
//! and already branched on the spectrum before the fix: the probes log-strength
//! trace for its dense sibling. A frozen-gate finite difference of
//! `exact_observed_information_log_dets` is NOT an arbiter for these rows: it
//! prices basins on the dense joint spectrum and never sees a per-row clamp basin.
#![cfg(test)]
use super::*;
use crate::assignment::AssignmentMode;
use crate::manifold::arrow_solver::DeflatedArrowSolver;
use crate::manifold::tests_sparse_curvature_operator_2500::threshold_gate_tiny_fixture;
use gam_solve::arrow_schur::{
    ArrowSolveOptions, SPECTRAL_DEFLATION_REL_FLOOR, solve_arrow_newton_step_with_options,
};
use ndarray::{Array1, Array2};

/// One declared state whose exact-A evidence factor carries clamp-basin rows.
struct ClampBasinState {
    term: SaeManifoldTerm,
    rho: SaeManifoldRho,
    cache: ArrowFactorCache,
    rows: Vec<usize>,
}

/// The exact-A evidence factor the matrix-free outer gradient consumes
/// (`BundleEvidenceGeometry::cache`): the majorizer system corrected to
/// `A = B + ΔC`, carrying the classification geometry, and factored under the
/// refusing unit-deflation policy of that lane.
fn exact_a_evidence_cache(
    term: &SaeManifoldTerm,
    target: &Array2<f64>,
    rho: &SaeManifoldRho,
) -> Result<(SaeManifoldTerm, ArrowFactorCache), String> {
    let mut anchor = term.clone();
    let mut majorizer = anchor.assemble_arrow_schur(target.view(), rho, None)?;
    SaeManifoldTerm::ensure_row_gauge_deflation_for_quasi_laplace(&mut majorizer);
    let exact = anchor.exact_a_evidence_system(target.view(), rho, &majorizer)?;
    let options = ArrowSolveOptions::direct()
        .with_newton_schur_tikhonov(SPECTRAL_DEFLATION_REL_FLOOR)
        .with_indefinite_refusing_evidence_unit_deflation(SPECTRAL_DEFLATION_REL_FLOOR);
    let (_delta_t, _delta_beta, cache) =
        solve_arrow_newton_step_with_options(&exact, 0.0, 0.0, &options)
            .map_err(|err| format!("{err:?}"))?;
    anchor.streaming_gates_frozen = true;
    Ok((anchor, cache))
}

/// Rows recorded with a spectrum that reprices at least one eigenvalue and no
/// deflated direction: the clamp-basin shape. An identity spectrum is not
/// counted, so a recorded-but-inert map cannot satisfy the premise.
fn clamp_basin_rows(cache: &ArrowFactorCache) -> Vec<usize> {
    cache
        .deflation_row_spectra
        .iter()
        .zip(cache.deflated_row_directions.iter())
        .enumerate()
        .filter_map(|(row, (spectrum, directions))| {
            let spectrum = spectrum.as_ref()?;
            let repriced = spectrum
                .raw_evals
                .iter()
                .zip(spectrum.cond_evals.iter())
                .any(|(raw, priced)| {
                    (raw - priced).abs() > f64::EPSILON * raw.abs().max(priced.abs()).max(1.0)
                });
            (directions.is_empty() && repriced).then_some(row)
        })
        .collect()
}

/// Walk a declared ARD-precision ladder and return the first state whose factor
/// records a clamp-basin row. The rung is selected on the cache's own
/// classification, never on agreement with any producer. Both clamp producers
/// can fire: the periodic ARD prior's concave half on coordinate slots, and the
/// ThresholdGate's concave remainder on logits above the threshold. The census
/// of every rung prints unconditionally.
fn first_clamp_basin_state(mode: AssignmentMode, straddle: bool) -> ClampBasinState {
    let (mut term, target, fixture_rho) = threshold_gate_tiny_fixture(straddle);
    term.assignment.mode = mode;
    let base_rho = fixture_rho.for_assignment(mode);
    let mut census = Vec::new();
    for log_ard in [-1.0_f64, 0.0, 1.0, 2.0, 3.0, 4.0] {
        let mut rho = base_rho.clone();
        for axes in rho.log_ard.iter_mut() {
            axes.fill(log_ard);
        }
        match exact_a_evidence_cache(&term, &target, &rho) {
            Ok((anchor, cache)) => {
                let rows = clamp_basin_rows(&cache);
                let spectral = cache
                    .deflation_row_spectra
                    .iter()
                    .filter(|spectrum| spectrum.is_some())
                    .count();
                let directions: usize = cache.deflated_row_directions.iter().map(Vec::len).sum();
                census.push(format!(
                    "log_ard={log_ard}: spectral_rows={spectral} \
                     deflated_directions={directions} clamp_basin_rows={rows:?}"
                ));
                if !rows.is_empty() {
                    eprintln!(
                        "#2333 CLAMP_BASIN_CENSUS mode={mode:?} straddle={straddle}\n{}",
                        census.join("\n")
                    );
                    return ClampBasinState {
                        term: anchor,
                        rho,
                        cache,
                        rows,
                    };
                }
            }
            Err(err) => census.push(format!("log_ard={log_ard}: no factor: {err}")),
        }
    }
    panic!(
        "#2333 premise: no rung of the declared ladder produced a clamp-basin row \
         (spectrum recorded and repriced, no deflated direction), so this gate \
         would compare producers where the two conventions coincide:\n{}",
        census.join("\n")
    );
}

/// The exact `(z_j, S⁻¹ z_j)` bundle at full-basis probes `√k·e_j`, where the
/// Hutchinson outer products are algebraically exact.
fn full_basis_bundle(cache: &ArrowFactorCache) -> (Vec<Array1<f64>>, Vec<Array1<f64>>) {
    let k = cache.k;
    let sqrt_k = (k as f64).sqrt();
    let probes: Vec<Array1<f64>> = (0..k)
        .map(|j| {
            let mut probe = Array1::<f64>::zeros(k);
            probe[j] = sqrt_k;
            probe
        })
        .collect();
    let sinv = probes
        .iter()
        .map(|probe| {
            cache
                .schur_inverse_apply(probe.view())
                .expect("exact reduced-Schur solve at a full-basis probe")
        })
        .collect();
    (probes, sinv)
}

#[test]
fn clamp_basin_rows_enter_the_dense_assignment_log_strength_trace_2333() {
    let state = first_clamp_basin_state(AssignmentMode::threshold_gate(1.0, 0.0), true);
    let solver = DeflatedArrowSolver::plain(&state.cache);
    let dense = state
        .term
        .assignment_log_strength_hessian_trace(&state.rho, &state.cache, &solver)
        .expect("dense assignment log-strength trace on the exact-A factor");
    let (probes, sinv) = full_basis_bundle(&state.cache);
    // The dense trace contracts the majorizer's prior curvature; the probes
    // sibling at the majorizer operator contracts the same curvature.
    let from_probes = state
        .term
        .assignment_log_strength_hessian_trace_from_probes(
            &state.rho,
            &state.cache,
            &probes,
            &sinv,
            EvidenceOperator::Majorizer,
        )
        .expect("probes assignment log-strength trace on the exact-A factor");
    let gap = (dense - from_probes).abs();
    let bar = 1.0e-9 * (1.0 + dense.abs().max(from_probes.abs()));
    eprintln!(
        "#2333 CLAMP_BASIN_LOG_STRENGTH rows={:?} dense={dense:.12e} \
         probes={from_probes:.12e} gap={gap:.6e}",
        state.rows
    );
    assert!(
        gap <= bar,
        "#2333: on clamp-basin rows the dense assignment log-strength trace must \
         subtract the correction its probes sibling subtracts: dense={dense:e} \
         probes={from_probes:e} gap={gap:e} bar={bar:e}"
    );
}

/// Largest per-entry relative departure of `actual` from `reference`
/// (`|r − a| / (1 + |r|)`), and the largest reference magnitude.
fn adjoint_gap(reference: &SaeArrowVector, actual: &SaeArrowVector) -> (f64, f64) {
    assert_eq!(reference.t.len(), actual.t.len());
    assert_eq!(reference.beta.len(), actual.beta.len());
    let mut gap = 0.0_f64;
    let mut scale = 0.0_f64;
    for (expected, observed) in reference
        .t
        .iter()
        .chain(reference.beta.iter())
        .zip(actual.t.iter().chain(actual.beta.iter()))
    {
        assert!(
            expected.is_finite() && observed.is_finite(),
            "non-finite adjoint entry: reference={expected} actual={observed}"
        );
        gap = gap.max((expected - observed).abs() / (1.0 + expected.abs()));
        scale = scale.max(expected.abs());
    }
    (gap, scale)
}

/// #2333 — the resident Trace θ-adjoint of an independent-logistic gate on
/// clamp-basin rows.
///
/// `logdet_theta_adjoint` reduces threshold-gate rows through the Trace seam:
/// the independent row program, the `E_tt` fold and the host post-folds. The
/// arbiter is `logdet_theta_adjoint_from_probes` at full-basis probes, where its
/// Hutchinson outer products are exact; it keeps its own hand tower loop
/// (host-whitened channels, contract-then-subtract correction), so the two share
/// no reduction code. A clamp-basin row is the shape on which the fold must
/// branch on the spectrum with an empty direction list. Non-vacuity is the
/// measured separation of the arbiter from itself on a deflation-blind copy of
/// the cache.
#[test]
fn threshold_gate_trace_theta_adjoint_matches_from_probes_on_clamp_basin_rows_2333() {
    let state = first_clamp_basin_state(AssignmentMode::threshold_gate(1.0, 0.0), true);
    let solver = DeflatedArrowSolver::plain(&state.cache);
    let resident = state
        .term
        .logdet_theta_adjoint(&state.rho, &state.cache, &solver)
        .expect("resident Trace theta-adjoint on the exact-A factor");
    let (probes, sinv) = full_basis_bundle(&state.cache);
    let reference = state
        .term
        .logdet_theta_adjoint_from_probes(
            &state.rho,
            &state.cache,
            &probes,
            &sinv,
            EvidenceOperator::Majorizer,
            None,
        )
        .expect("from-probes theta-adjoint on the exact-A factor");
    let (parity, scale) = adjoint_gap(&reference, &resident);
    let mut blind = state.cache.clone();
    let rows = blind.row_dims.len();
    blind.deflated_row_directions = std::sync::Arc::from(vec![Vec::new(); rows]);
    blind.deflation_row_spectra = std::sync::Arc::from(vec![None; rows]);
    let blind_reference = state
        .term
        .logdet_theta_adjoint_from_probes(
            &state.rho,
            &blind,
            &probes,
            &sinv,
            EvidenceOperator::Majorizer,
            None,
        )
        .expect("deflation-blind from-probes theta-adjoint");
    let (separation, _) = adjoint_gap(&reference, &blind_reference);
    eprintln!(
        "#2333 CLAMP_BASIN_TRACE_ADJOINT rows={:?} scale={scale:.6e} parity={parity:.6e} \
         deflation_separation={separation:.6e}",
        state.rows
    );
    assert!(
        scale > 1.0e-6,
        "#2333: a vanishing theta-adjoint cannot establish parity; scale={scale:e}"
    );
    assert!(
        parity <= 1.0e-10,
        "#2333: the resident Trace theta-adjoint must equal its from-probes arbiter on \
         clamp-basin rows: worst relative entry gap {parity:e}"
    );
    assert!(
        separation > 1.0e-10 && separation > 1.0e3 * parity,
        "#2333: the parity bar must reject a deflation-blind adjoint on this fixture: \
         separation={separation:e} parity={parity:e}"
    );
}

/// #2333 — the whitening pre-fold for independent-logistic gates.
///
/// Under a row-varying full-rank likelihood metric the resident route projects
/// the four semantic output bases into each row's metric chart before the seam
/// builds the tower, while the from-probes arbiter whitens every materialized
/// channel on the host. Non-vacuity is
/// the measured separation from the resident adjoint with the metric removed.
#[test]
fn independent_gate_trace_theta_adjoint_whitens_like_from_probes_2333() {
    let mode = AssignmentMode::threshold_gate(1.0, 0.0);
    let (mut term, target, fixture_rho) = threshold_gate_tiny_fixture(true);
    term.assignment.mode = mode;
    let rho = fixture_rho.for_assignment(mode);
    let (n, p) = (term.n_obs(), term.output_dim());
    assert_eq!(p, 3, "#2333 the metric cell below is 3x3");
    let cell = [
        [1.05_f64, 0.07, -0.03],
        [-0.04, 0.90, 0.06],
        [0.02, -0.05, 1.15],
    ];
    let drift_cell = [
        [0.08_f64, 0.0, 0.0],
        [0.0, -0.05, 0.0],
        [0.0, 0.0, -0.07],
    ];
    let factors = Array2::<f64>::from_shape_fn((n, p * p), |(row, col)| {
        let (out_col, rank_col) = (col / p, col % p);
        let drift = row as f64 / (n - 1) as f64;
        cell[out_col][rank_col] + drift * drift_cell[out_col][rank_col]
    });
    term.set_row_metric(
        gam_problem::RowMetric::behavioral_fisher(std::sync::Arc::new(factors), p, p)
            .expect("#2333 row metric"),
    )
    .expect("#2333 row metric installs");
    assert!(
        term.whiten_logdet_row_jets(),
        "#2333 the metric must engage row whitening, else the pre-fold is untested"
    );
    let mut system = term
        .assemble_arrow_schur(target.view(), &rho, None)
        .expect("#2333 arrow assembly under the row metric");
    SaeManifoldTerm::ensure_row_gauge_deflation_for_quasi_laplace(&mut system);
    let options = ArrowSolveOptions::direct().with_positive_definite_evidence();
    let (_delta_t, _delta_beta, cache) =
        solve_arrow_newton_step_with_options(&system, 0.0, 0.0, &options)
            .expect("#2333 spectrally conditioned evidence factor");
    let solver = DeflatedArrowSolver::plain(&cache);
    let resident = term
        .logdet_theta_adjoint(&rho, &cache, &solver)
        .expect("#2333 resident Trace theta-adjoint");
    let (probes, sinv) = full_basis_bundle(&cache);
    let reference = term
        .logdet_theta_adjoint_from_probes(
            &rho,
            &cache,
            &probes,
            &sinv,
            EvidenceOperator::Majorizer,
            None,
        )
        .expect("#2333 from-probes theta-adjoint");
    let (parity, scale) = adjoint_gap(&reference, &resident);
    let mut unwhitened = term.clone();
    unwhitened.row_metric = None;
    let counterfactual = unwhitened
        .logdet_theta_adjoint(&rho, &cache, &solver)
        .expect("#2333 resident theta-adjoint with the row metric removed");
    let (separation, _) = adjoint_gap(&reference, &counterfactual);
    let live_rows = (0..cache.row_dims.len())
        .filter(|&row| {
            SaeManifoldTerm::row_deflation_is_live(
                cache
                    .deflated_row_directions
                    .get(row)
                    .map(Vec::as_slice)
                    .unwrap_or(&[]),
                cache.deflation_row_spectra.get(row).and_then(Option::as_ref),
            )
        })
        .count();
    eprintln!(
        "#2333 INDEPENDENT_TRACE_WHITENING mode={mode:?} live_rows={live_rows} \
         scale={scale:.6e} parity={parity:.6e} whitening_separation={separation:.6e}"
    );
    assert!(
        scale > 1.0e-6,
        "#2333 ({mode:?}): a vanishing theta-adjoint cannot establish parity; scale={scale:e}"
    );
    assert!(
        parity <= 1.0e-10,
        "#2333 ({mode:?}): the resident Trace theta-adjoint must equal its from-probes \
         arbiter under row whitening: worst relative entry gap {parity:e}"
    );
    assert!(
        separation > 1.0e-10 && separation > 1.0e3 * parity,
        "#2333 ({mode:?}): the parity bar must reject an unwhitened adjoint on this \
         fixture: separation={separation:e} parity={parity:e}"
    );
}

/// #2915 — on the matrix-free lane a threshold gate's sparse log-det trace must
/// differentiate the exact observed information the lane's value prices.
///
/// The value is `½ arrow_log_det` of the exact-A evidence factor. Its central
/// difference in `log_lambda_sparse` over frozen θ̂, on one discrete stratum with
/// no clamp-basin row, is the arbiter; the clamp price's own derivative is a
/// separate leg this gate does not cover. The majorizer-operator trace on the
/// same factor differentiates `B` and is the positive control, because a
/// straddling gate carries a nonzero remainder `ΔC` on every switched-on logit.
#[test]
fn threshold_gate_exact_sparse_logdet_trace_matches_the_lane_value_2915() {
    let (term, target, fixture_rho) = threshold_gate_tiny_fixture(true);
    let mut census = Vec::new();
    let mut selected = None;
    for log_ard in [-1.0_f64, 0.0, 1.0, 2.0, 3.0, 4.0] {
        let mut rho = fixture_rho.clone();
        for axes in rho.log_ard.iter_mut() {
            axes.fill(log_ard);
        }
        match exact_a_evidence_cache(&term, &target, &rho) {
            Ok((anchor, cache)) => {
                let rows = clamp_basin_rows(&cache);
                census.push(format!("log_ard={log_ard}: clamp_basin_rows={rows:?}"));
                if rows.is_empty() {
                    selected = Some((anchor, cache, rho));
                    break;
                }
            }
            Err(err) => census.push(format!("log_ard={log_ard}: no factor: {err}")),
        }
    }
    eprintln!("#2915 LANE_SPARSE_TRACE census\n{}", census.join("\n"));
    let (anchor, cache, rho) = selected.unwrap_or_else(|| {
        panic!(
            "#2915 premise: no rung of the declared ladder gave an exact-A factor without \
             clamp-basin rows:\n{}",
            census.join("\n")
        )
    });
    let remainder = crate::assignment::threshold_gate_negative_hessian_remainder_weighted(
        &anchor.assignment,
        &rho,
        anchor.row_loss_weights.as_deref(),
    )
    .expect("#2915 threshold-gate remainder");
    assert!(
        remainder.iter().any(|value| *value != 0.0),
        "#2915 premise: a straddling gate must carry a live remainder"
    );
    let (probes, sinv) = full_basis_bundle(&cache);
    let exact = anchor
        .assignment_log_strength_hessian_trace_from_probes(
            &rho,
            &cache,
            &probes,
            &sinv,
            EvidenceOperator::ExactObservedInformation,
        )
        .expect("#2915 exact-operator sparse trace");
    let majorizer = anchor
        .assignment_log_strength_hessian_trace_from_probes(
            &rho,
            &cache,
            &probes,
            &sinv,
            EvidenceOperator::Majorizer,
        )
        .expect("#2915 majorizer-operator sparse trace");
    let stratum = |c: &ArrowFactorCache| -> (usize, usize) {
        let spectral = c
            .deflation_row_spectra
            .iter()
            .filter(|spectrum| spectrum.is_some())
            .count();
        let directions: usize = c.deflated_row_directions.iter().map(Vec::len).sum();
        (spectral, directions)
    };
    let anchor_stratum = stratum(&cache);
    let half_log_det = |log_lambda_sparse: f64| -> f64 {
        let mut moved = rho.clone();
        moved.log_lambda_sparse = log_lambda_sparse;
        let (_, endpoint) = exact_a_evidence_cache(&term, &target, &moved)
            .expect("#2915 exact-A factor at a finite-difference endpoint");
        assert_eq!(
            stratum(&endpoint),
            anchor_stratum,
            "#2915: both endpoints must sit on the anchor's discrete stratum"
        );
        assert!(
            clamp_basin_rows(&endpoint).is_empty(),
            "#2915: an endpoint must not price a clamp basin"
        );
        0.5 * endpoint
            .arrow_log_det()
            .expect("#2915 authoritative joint log-det of the exact-A factor")
    };
    let h = 1.0e-5;
    let fd = (half_log_det(rho.log_lambda_sparse + h) - half_log_det(rho.log_lambda_sparse - h))
        / (2.0 * h);
    let bar = 1.0e-6 * (1.0 + fd.abs());
    eprintln!(
        "#2915 LANE_SPARSE_TRACE stratum={anchor_stratum:?} fd={fd:.12e} exact={exact:.12e} \
         majorizer={majorizer:.12e} bar={bar:.3e}"
    );
    assert!(
        (exact - fd).abs() <= bar,
        "#2915: the exact-operator sparse log-det trace must be the central difference of \
         the lane value: exact={exact:e} fd={fd:e} bar={bar:e}"
    );
    assert!(
        (majorizer - fd).abs() > 1.0e3 * bar,
        "#2915 positive control: the majorizer-operator trace must miss the remainder on a \
         straddling gate: majorizer={majorizer:e} fd={fd:e} bar={bar:e}"
    );
}

/// #2915 — the clamp-basin price's own derivative on the matrix-free lane.
///
/// On a threshold-gate state whose exact-A evidence factor prices clamp-basin
/// rows, every exact-operator from-probes channel must be the central difference
/// of the lane value on one discrete stratum: the sparse log-strength trace and
/// one ARD precision trace against `½ arrow_log_det` over frozen θ̂, and the
/// θ-adjoint entry of a logit on a clamp-basin row against `arrow_log_det` at
/// frozen ρ. The premise requires a material basin price, so the clamp leg is
/// live on the rows the gate measures. Each gap prints before any assertion.
#[test]
fn clamp_basin_price_derivative_matches_the_lane_value_2915() {
    let mode = AssignmentMode::threshold_gate(1.0, 0.0);
    let state = first_clamp_basin_state(mode, true);
    let (mut term, target, _) = threshold_gate_tiny_fixture(true);
    term.assignment.mode = mode;
    let basin_price = state
        .rows
        .iter()
        .filter_map(|&row| state.cache.deflation_row_spectra[row].as_ref())
        .flat_map(|spectrum| {
            spectrum
                .raw_evals
                .iter()
                .zip(spectrum.cond_evals.iter())
                .zip(spectrum.conditioning.iter())
                .filter(|((raw, _), conditioning)| {
                    **raw < 0.0 && **conditioning == RowSpectralConditioning::Raw
                })
                .map(|((raw, priced), _)| priced - raw)
                .collect::<Vec<f64>>()
        })
        .fold(0.0_f64, f64::max);
    assert!(
        basin_price > 1.0e-3,
        "#2915 premise: the clamp-basin rows {:?} must carry a material price; largest \
         v'Ev = {basin_price:e}",
        state.rows
    );
    let (probes, sinv) = full_basis_bundle(&state.cache);
    let operator = EvidenceOperator::ExactObservedInformation;
    let stratum = |c: &ArrowFactorCache| -> (Vec<usize>, usize) {
        let directions: usize = c.deflated_row_directions.iter().map(Vec::len).sum();
        (clamp_basin_rows(c), directions)
    };
    let anchor_stratum = stratum(&state.cache);
    let log_det_at = |moved_term: &SaeManifoldTerm, moved_rho: &SaeManifoldRho| -> f64 {
        let (_, endpoint) = exact_a_evidence_cache(moved_term, &target, moved_rho)
            .expect("#2915 exact-A factor at a finite-difference endpoint");
        assert_eq!(
            stratum(&endpoint),
            anchor_stratum,
            "#2915: both endpoints must sit on the anchor's clamp-basin stratum"
        );
        endpoint
            .arrow_log_det()
            .expect("#2915 authoritative joint log-det of the exact-A factor")
    };
    let h = 1.0e-5;

    let sparse = state
        .term
        .assignment_log_strength_hessian_trace_from_probes(
            &state.rho,
            &state.cache,
            &probes,
            &sinv,
            operator,
        )
        .expect("#2915 exact-operator sparse trace");
    let sparse_fd = {
        let mut plus = state.rho.clone();
        let mut minus = state.rho.clone();
        plus.log_lambda_sparse += h;
        minus.log_lambda_sparse -= h;
        0.5 * (log_det_at(&term, &plus) - log_det_at(&term, &minus)) / (2.0 * h)
    };

    let ard_atom = (0..state.rho.log_ard.len())
        .find(|&atom| !state.rho.log_ard[atom].is_empty())
        .expect("#2915 premise: the fixture must carry an ARD precision");
    let ard = state
        .term
        .ard_log_precision_hessian_trace_from_probes(
            &state.rho,
            &state.cache,
            &probes,
            &sinv,
            operator,
        )
        .expect("#2915 exact-operator ARD trace")[ard_atom][0];
    let ard_fd = {
        let mut plus = state.rho.clone();
        let mut minus = state.rho.clone();
        plus.log_ard[ard_atom][0] += h;
        minus.log_ard[ard_atom][0] -= h;
        0.5 * (log_det_at(&term, &plus) - log_det_at(&term, &minus)) / (2.0 * h)
    };

    let row = state.rows[0];
    let variables = term
        .row_vars_for_cache_row(row, &state.cache)
        .expect("#2915 row variables");
    let (position, atom) = variables
        .iter()
        .enumerate()
        .find_map(|(position, variable)| match *variable {
            SaeLocalRowVar::Logit { atom } => Some((position, atom)),
            SaeLocalRowVar::Coord { .. } => None,
        })
        .expect("#2915 premise: a threshold-gate row carries a free logit");
    let theta = state
        .term
        .logdet_theta_adjoint_from_probes(
            &state.rho,
            &state.cache,
            &probes,
            &sinv,
            operator,
            Some(target.view()),
        )
        .expect("#2915 exact-operator theta-adjoint")
        .t[state.cache.row_offsets[row] + position];
    let theta_fd = {
        let mut plus = term.clone();
        let mut minus = term.clone();
        plus.assignment.logits[[row, atom]] += h;
        minus.assignment.logits[[row, atom]] -= h;
        (log_det_at(&plus, &state.rho) - log_det_at(&minus, &state.rho)) / (2.0 * h)
    };

    let rho_bar = |fd: f64| 1.0e-6 * (1.0 + fd.abs());
    let theta_bar = 1.0e-5 * (1.0 + theta_fd.abs());
    eprintln!(
        "#2915 CLAMP_BASIN_PRICE rows={:?} basin_price={basin_price:.6e}\n  sparse={sparse:.12e} \
         fd={sparse_fd:.12e} gap={:.3e}\n  ard[{ard_atom},0]={ard:.12e} fd={ard_fd:.12e} \
         gap={:.3e}\n  theta[row {row}, logit {atom}]={theta:.12e} fd={theta_fd:.12e} gap={:.3e}",
        state.rows,
        (sparse - sparse_fd).abs(),
        (ard - ard_fd).abs(),
        (theta - theta_fd).abs()
    );
    assert!(
        (sparse - sparse_fd).abs() <= rho_bar(sparse_fd),
        "#2915: the exact-operator sparse trace must differentiate the lane value on \
         clamp-basin rows: trace={sparse:e} fd={sparse_fd:e}"
    );
    assert!(
        (ard - ard_fd).abs() <= rho_bar(ard_fd),
        "#2915: the exact-operator ARD trace must differentiate the lane value on \
         clamp-basin rows: trace={ard:e} fd={ard_fd:e}"
    );
    assert!(
        (theta - theta_fd).abs() <= theta_bar,
        "#2915: the exact-operator theta-adjoint must differentiate the lane value on a \
         clamp-basin row: adjoint={theta:e} fd={theta_fd:e}"
    );
}

/// #2914 — the dense and from-probes ARD log-precision traces on clamp-basin rows.
///
/// Both siblings gate their Daleckii–Krein correction on `row_deflation_is_live`,
/// and neither had an arbiter on a row whose spectrum is recorded while its
/// direction list is empty. On the first clamp-basin rung they must agree per
/// entry. The from-probes trace on a deflation-blind copy of the same cache must
/// separate from them, or the bar would also accept the direction-list convention.
#[test]
fn ard_trace_dense_and_probes_agree_on_clamp_basin_rows_2914() {
    let state = first_clamp_basin_state(AssignmentMode::threshold_gate(1.0, 0.0), true);
    let solver = DeflatedArrowSolver::plain(&state.cache);
    let dense = state
        .term
        .ard_log_precision_hessian_trace(
            &state.rho,
            &state.cache,
            &solver,
            EvidenceOperator::Majorizer,
        )
        .expect("#2914 dense ARD trace on the exact-A factor");
    let (probes, sinv) = full_basis_bundle(&state.cache);
    let from_probes = state
        .term
        .ard_log_precision_hessian_trace_from_probes(
            &state.rho,
            &state.cache,
            &probes,
            &sinv,
            EvidenceOperator::Majorizer,
        )
        .expect("#2914 from-probes ARD trace on the exact-A factor");
    let mut blind = state.cache.clone();
    let rows = blind.row_dims.len();
    blind.deflated_row_directions = std::sync::Arc::from(vec![Vec::new(); rows]);
    blind.deflation_row_spectra = std::sync::Arc::from(vec![None; rows]);
    let blind_probes = state
        .term
        .ard_log_precision_hessian_trace_from_probes(
            &state.rho,
            &blind,
            &probes,
            &sinv,
            EvidenceOperator::Majorizer,
        )
        .expect("#2914 deflation-blind from-probes ARD trace");
    assert_eq!(dense.len(), from_probes.len());
    let mut gap = 0.0_f64;
    let mut separation = 0.0_f64;
    let mut scale = 0.0_f64;
    for atom in 0..dense.len() {
        assert_eq!(dense[atom].len(), from_probes[atom].len());
        for axis in 0..dense[atom].len() {
            let reference = dense[atom][axis];
            let probed = from_probes[atom][axis];
            gap = gap.max((reference - probed).abs() / (1.0 + reference.abs()));
            separation = separation
                .max((probed - blind_probes[atom][axis]).abs() / (1.0 + probed.abs()));
            scale = scale.max(reference.abs());
        }
    }
    eprintln!(
        "#2914 ARD_TRACE_CLAMP_BASIN rows={:?} scale={scale:.6e} gap={gap:.6e} \
         deflation_separation={separation:.6e}",
        state.rows
    );
    assert!(
        scale > 1.0e-6,
        "#2914: a vanishing ARD trace cannot establish agreement; scale={scale:e}"
    );
    assert!(
        gap <= 1.0e-9,
        "#2914: the dense and from-probes ARD traces must agree on clamp-basin rows: worst \
         relative entry gap {gap:e}"
    );
    assert!(
        separation > 1.0e-9 && separation > 1.0e3 * gap,
        "#2914: the agreement bar must reject a deflation-blind trace on this fixture: \
         separation={separation:e} gap={gap:e}"
    );
}
