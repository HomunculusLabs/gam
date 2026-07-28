//! One accounting for the penalized effective-degrees-of-freedom bundle.
//!
//! Every fitting route computes the per-block penalty traces
//! `tr_k = λ_k·tr(H⁻¹ S_k)` with whatever linear algebra its parameterisation
//! affords — a factorized solve against the canonical transformed blocks, a
//! dense product against a latent covariance, a Cholesky solve against an
//! assembled Hessian. That part is genuinely route-specific and stays where it
//! is. What is *not* route-specific is the **accounting** those traces feed:
//! which ceiling a trace is clamped to, what a non-finite trace resolves to,
//! what `edf_by_block` is measured against, and what floor `edf_total` may not
//! fall below.
//!
//! Those four rules were written out independently at six sites and did not
//! agree (issue #2470). Two disagreements were load-bearing:
//!
//! * **The per-block ceiling.** `rank(S_k)` and `block_cols` are not the same
//!   number; they differ by `nullity(S_k)`, which is a whole integer of reported
//!   complexity for every penalized block. `rank(S_k)` is the correct one, and
//!   not merely by convention: it is the quantity the REML criterion already
//!   prices as `rank(S_k)·ρ_k`, so it is the ceiling that agrees with the
//!   objective being optimized. Passing the ranks in explicitly is deliberate —
//!   a caller must *state* its rank oracle rather than reach for a column count
//!   because that is what happened to be in scope.
//! * **The floor.** `edf_total` cannot fall below the joint penalty null-space
//!   dimension `mp = p − rank(Σ_k S_k)`: those directions are unpenalized, so no
//!   amount of smoothing removes them. Clamping to `[0, p]` instead lets a noisy
//!   trace report an effective dimension below the mathematically attainable
//!   minimum, and nothing downstream notices.
//!
//! `edf_total` feeds `σ̂² = RSS/(n − edf_total)`, conditional AIC, the
//! likelihood-ratio reference df and every interval width, so a disagreement
//! here is not a reporting curiosity.
//!
//! Because the floor `mp` is stated here, this is also the one place that can
//! see `edf_total` LAND on it — a fit that kept none of the penalized
//! directions its design offered and returned the model no amount of smoothing
//! can remove. #2607 and #2579 both produced exactly that, by unrelated routes,
//! and both reported `Converged` with nothing said. `collapsed_to_penalty_null_space`
//! names the state by its outcome, so one check covers every route into it.

/// The three EDF quantities a fit publishes, produced together so they cannot
/// disagree with one another.
#[derive(Clone, Debug, PartialEq)]
pub struct EdfBundle {
    /// `p − Σ_k tr_k`, clamped to `[mp, p]`.
    pub edf_total: f64,
    /// `rank_k − tr_k` per penalty block, clamped to `[0, rank_k]`.
    pub edf_by_block: Vec<f64>,
    /// The admitted per-block traces `tr_k`, each clamped to `[0, rank_k]`.
    /// Retained because the per-term EDF decomposition is assembled from them
    /// (issue #1219), so downstream must read the same numbers this accounting
    /// used rather than re-clamping the raw values itself.
    pub penalty_block_trace: Vec<f64>,
}

/// Admit one raw penalty trace against its block rank.
///
/// A PSD penalty can absorb at most its own rank, so `tr_k` is mathematically
/// confined to `[0, rank_k]`. When the outer optimizer drives a redundant
/// block's `λ_k = exp(ρ_k)` to the ceiling, the raw product `λ_k·frob` can
/// overflow to `+∞` on a ridge-stabilized Hessian even though the true value is
/// exactly `rank_k` (gam#1379). `f64::clamp` does **not** rescue NaN — it
/// propagates it — and a NaN would reach the fit-result finiteness validator, so
/// any non-finite product resolves to the saturated bound, which is where the
/// `+∞` case lands anyway.
fn admit_trace(raw: f64, rank: usize) -> f64 {
    let ceiling = rank as f64;
    if raw.is_finite() {
        raw.clamp(0.0, ceiling)
    } else {
        ceiling
    }
}

/// Assemble the EDF bundle from already-computed per-block penalty traces.
///
/// `raw_block_traces` and `block_ranks` are aligned 1:1 with the penalty blocks.
/// `coefficient_count` is `p`. `joint_penalty_nullity` is `mp = p − rank(Σ_k S_k)`,
/// taken as a parameter rather than derived from `block_ranks`: the joint rank is
/// the rank of the *stacked* penalty root, which is not in general the sum of the
/// per-block ranks.
///
/// Traces are summed with compensated (Kahan) addition because `edf_total` is a
/// difference of two like-sized quantities, where naive summation error lands
/// directly in the reported effective dimension.
pub fn penalized_edf_bundle(
    raw_block_traces: &[f64],
    block_ranks: &[usize],
    coefficient_count: usize,
    joint_penalty_nullity: f64,
) -> EdfBundle {
    assert_blocks_aligned(raw_block_traces.len(), block_ranks.len());
    let penalty_block_trace: Vec<f64> = raw_block_traces
        .iter()
        .zip(block_ranks.iter())
        .map(|(&raw, &rank)| admit_trace(raw, rank))
        .collect();
    let edf_by_block: Vec<f64> = penalty_block_trace
        .iter()
        .zip(block_ranks.iter())
        .map(|(&trace, &rank)| {
            let ceiling = rank as f64;
            (ceiling - trace).clamp(0.0, ceiling)
        })
        .collect();
    let p = coefficient_count as f64;
    let edf_total = (p - super::penalty::kahan_sum(penalty_block_trace.iter().copied()))
        .clamp(joint_penalty_nullity.min(p), p);
    if collapsed_to_penalty_null_space(edf_total, coefficient_count, joint_penalty_nullity) {
        let mp = joint_penalty_nullity.clamp(0.0, p);
        log::warn!(
            "fit collapsed to its penalty null space: effective df {edf_total:.3} of {p} \
             coefficients, against a joint penalty nullity of {mp}. The {} penalized \
             directions this design offered were smoothed away entirely -- the model \
             returned is the one no amount of smoothing can remove. On a saturated or \
             near-saturated design that is the criterion's own optimum rather than a \
             solver failure; otherwise it is the signature of a lambda railed at its \
             ceiling (#2607).",
            p - mp,
        );
    }
    EdfBundle {
        edf_total,
        edf_by_block,
        penalty_block_trace,
    }
}

/// How many of the penalized directions a design offered survived into the fit,
/// and whether that count is small enough to call the result the penalty's own
/// null model.
///
/// `mp = p − rank(Σ_k S_k)` is the dimension smoothing cannot touch, so
/// `attainable = p − mp` is what λ-selection actually chooses over and
/// `spent = edf_total − mp` is what it kept. #2607 and #2579 reached the SAME
/// visible end state — `Converged`, `edf` within rounding of the intercept —
/// from two unrelated mechanisms (a saturated design whose REML optimum is
/// maximum smoothing; a df floor that iterated an emptied penalty list). Naming
/// the state by its *outcome* rather than by either route is what makes one
/// check cover both, and any third route that lands here.
///
/// The two bounds are what separate a collapse from an ordinary answer:
///
/// * `spent <= 0.5` — less than half of ONE direction retained out of every
///   direction on offer. `hifreq_tensor_k10` measured `edf = 1.294` against
///   `mp = 1`, i.e. `spent = 0.294` of an attainable 575.
/// * `attainable >= 20` — a single smooth term legitimately shrinks to its own
///   null space whenever the truth really is linear, and a `k = 10` marginal
///   offers only ~8 penalized directions, so firing there would report a
///   correct model selection as a defect. Twenty is the point past which a
///   design has been given substantial flexibility and kept none of it; it is a
///   threshold on how much was discarded, not on how well the fit did.
///
/// Returns `false` for any non-finite input rather than warning about arithmetic
/// that has already failed somewhere upstream.
pub fn collapsed_to_penalty_null_space(
    edf_total: f64,
    coefficient_count: usize,
    joint_penalty_nullity: f64,
) -> bool {
    /// Penalized directions a design must offer before keeping none of them is
    /// reported as a collapse rather than as a linear truth being found.
    const MIN_ATTAINABLE_DIRECTIONS: f64 = 20.0;
    /// Retained penalized df, below which the fit IS its penalty null model.
    const MAX_RETAINED_DF: f64 = 0.5;
    if !edf_total.is_finite() || !joint_penalty_nullity.is_finite() {
        return false;
    }
    let p = coefficient_count as f64;
    let mp = joint_penalty_nullity.clamp(0.0, p);
    let attainable = p - mp;
    let spent = edf_total - mp;
    attainable >= MIN_ATTAINABLE_DIRECTIONS && spent <= MAX_RETAINED_DF
}

/// Length agreement between the traces and their ranks is a caller contract, not
/// a runtime condition to recover from: a mismatch means the caller paired the
/// wrong penalty blocks, and silently zipping to the shorter of the two would
/// drop a block's complexity from `edf_total` without a word.
fn assert_blocks_aligned(traces: usize, ranks: usize) {
    assert_eq!(
        traces, ranks,
        "penalized_edf_bundle: {traces} traces against {ranks} block ranks; \
         they are aligned 1:1 with the penalty blocks"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_trace_is_admitted_against_its_block_rank_not_its_column_count() {
        // A rank-2 penalty on a 5-column block: the trace saturates at 2, and
        // the reported block EDF is measured against 2. Using the column count
        // as the ceiling would report `5 - 2 = 3` here instead of `0`, which is
        // exactly the nullity(S_k) = 3 overstatement this accounting exists to
        // remove.
        let bundle = penalized_edf_bundle(&[7.0], &[2], 5, 3.0);
        assert_eq!(bundle.penalty_block_trace, vec![2.0]);
        assert_eq!(bundle.edf_by_block, vec![0.0]);
    }

    #[test]
    fn a_non_finite_trace_resolves_to_the_saturated_rank_not_nan() {
        // A ceiling-λ redundant block overflows to +inf on a ridge-stabilized
        // Hessian; the true penalized trace is the block rank. NaN must resolve
        // the same way — `f64::clamp` propagates NaN, so this cannot be left to
        // the clamp alone.
        for raw in [f64::INFINITY, f64::NAN] {
            let bundle = penalized_edf_bundle(&[raw], &[3], 6, 3.0);
            assert_eq!(
                bundle.penalty_block_trace,
                vec![3.0],
                "non-finite raw trace {raw} must saturate at the block rank"
            );
            assert!(bundle.edf_total.is_finite());
        }
    }

    #[test]
    fn a_negative_trace_is_admitted_at_zero() {
        let bundle = penalized_edf_bundle(&[-0.25], &[4], 4, 0.0);
        assert_eq!(bundle.penalty_block_trace, vec![0.0]);
        assert_eq!(bundle.edf_by_block, vec![4.0]);
    }

    #[test]
    fn edf_total_cannot_fall_below_the_joint_penalty_null_space() {
        // p = 10 with mp = 3 unpenalized directions. Even a fully saturated
        // penalty cannot remove them, so the floor is 3, not 0. A `[0, p]` clamp
        // would report 0 here — an effective dimension below the mathematically
        // attainable minimum, with nothing downstream to notice.
        let bundle = penalized_edf_bundle(&[7.0], &[7], 10, 3.0);
        assert_eq!(bundle.edf_total, 3.0);
    }

    #[test]
    fn edf_total_is_p_minus_the_admitted_traces_when_interior() {
        let bundle = penalized_edf_bundle(&[1.5, 2.25], &[4, 5], 12, 3.0);
        assert_eq!(bundle.penalty_block_trace, vec![1.5, 2.25]);
        assert_eq!(bundle.edf_by_block, vec![2.5, 2.75]);
        assert_eq!(bundle.edf_total, 12.0 - 3.75);
    }

    #[test]
    fn an_unpenalized_fit_reports_every_coefficient() {
        let bundle = penalized_edf_bundle(&[], &[], 6, 6.0);
        assert_eq!(bundle.edf_total, 6.0);
        assert!(bundle.edf_by_block.is_empty());
        assert!(bundle.penalty_block_trace.is_empty());
    }

    #[test]
    #[should_panic(expected = "aligned 1:1 with the penalty blocks")]
    fn mismatched_traces_and_ranks_are_refused_not_zipped_short() {
        penalized_edf_bundle(&[1.0, 2.0], &[3], 5, 0.0);
    }

    #[test]
    fn the_state_2607_recorded_is_detected_by_its_outcome() {
        // The numbers `hifreq_tensor_k10` reported while it was saturated:
        // `edf = 1.294` of `p = 576`, joint penalty nullity 1 (the intercept —
        // a te() double penalty leaves nothing else unpenalized). Every one of
        // the 575 penalized directions was smoothed away, the fit reported
        // `Converged`, and nothing said so.
        assert!(collapsed_to_penalty_null_space(1.294, 576, 1.0));
        // The same fixture BEFORE the collapse, from the history #2585 records:
        // `edf = 227.938` of the same 576. Same design, same nullity — only the
        // outcome differs, which is the whole point of naming the state by its
        // outcome.
        assert!(!collapsed_to_penalty_null_space(227.938, 576, 1.0));
    }

    #[test]
    fn a_linear_truth_under_one_smooth_is_a_selection_not_a_collapse() {
        // A `k = 10` marginal offers ~8 penalized directions on top of a
        // 2-dimensional null space. When the truth really is linear, REML
        // shrinking that smooth to exactly its null space is the CORRECT
        // answer, and reporting it as a collapse would turn a good model
        // selection into a warning on a large fraction of honest fits.
        assert!(!collapsed_to_penalty_null_space(2.0, 10, 2.0));
        // Widen the same shape past the point where keeping nothing stops being
        // an ordinary selection, holding `spent` fixed at zero: the bound is on
        // how much was discarded, so this and the case above must disagree.
        assert!(collapsed_to_penalty_null_space(2.0, 30, 2.0));
    }

    #[test]
    fn an_unpenalized_design_can_never_collapse() {
        // `mp = p` means λ selects over nothing at all: `edf_total` is pinned to
        // `p` by construction, so there is no collapse available to report and a
        // predicate keyed on `spent` alone would fire on every such fit.
        assert!(!collapsed_to_penalty_null_space(40.0, 40, 40.0));
    }

    #[test]
    fn a_non_finite_edf_is_not_reported_as_a_collapse() {
        // NaN compares false against every bound, so `spent <= 0.5` would be
        // false for NaN but true for −inf. Neither is a statement about
        // smoothing: the arithmetic already failed upstream, and the finiteness
        // validators own that.
        for edf in [f64::NAN, f64::NEG_INFINITY, f64::INFINITY] {
            assert!(
                !collapsed_to_penalty_null_space(edf, 576, 1.0),
                "non-finite edf {edf} is an upstream arithmetic failure, not a collapse"
            );
        }
        assert!(!collapsed_to_penalty_null_space(1.294, 576, f64::NAN));
    }
}
