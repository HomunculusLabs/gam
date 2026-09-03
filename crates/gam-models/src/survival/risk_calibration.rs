//! Calibrating a scalar risk score into an absolute survival surface.
//!
//! A model that ranks subjects by risk does not by itself say what fraction of
//! them survive to time `t`. This module supplies the missing scale: fit a
//! univariate Cox proportional-hazards model of the observed training outcome
//! on the (centered) training risk score, take its Breslow baseline cumulative
//! hazard, and evaluate `S(t | r) = exp(-H₀(t) · exp(β·(r − r̄)))` for each
//! test risk on a shared time grid.
//!
//! This is the *benchmark comparator* path: it turns any risk ranking — gam's
//! own or a reference tool's — into the calibrated survival matrix that the
//! IPCW Brier / lifted-metric scorers consume, so both sides of a comparison
//! are scored on the same footing. It is deliberately a one-covariate model:
//! the covariate *is* the risk score under test, and the only free quantities
//! are its slope and the baseline.

use crate::survival::predict::KaplanMeier;
use ndarray::Array2;

/// Newton iterations the one-parameter partial-likelihood score equation is
/// allowed before the calibration is refused. On a strictly concave log
/// partial likelihood the monotone-safeguarded Newton step converges
/// quadratically, so this budget is reached only when no finite maximiser
/// exists — the risk score separates the outcomes — and that is reported as a
/// refusal, never returned as a slope (#2469: this loop used to fall through
/// to its last iterate, with a `1e-8` ridge on the score, a `±50` clamp on the
/// linear predictor, a `±2` step clamp and a `1e-10` absolute step tolerance
/// standing in for the arithmetic that now does the work exactly).
const NEWTON_MAX_ITERATIONS: usize = 50;

/// The marginal (covariate-free) Kaplan-Meier survival curve evaluated on
/// `grid`. Equivalent to `KaplanMeier::fit(times, events).on_grid(grid)`; kept
/// as a named function because it is the "null model" every calibrated matrix
/// is scored against.
pub fn km_curve_on_grid(times: &[f64], events: &[f64], grid: &[f64]) -> Vec<f64> {
    KaplanMeier::fit(times, events).on_grid(grid)
}

/// Calibrate `test_risk` into a `(test_risk.len(), grid.len())` survival matrix
/// using a univariate Cox fit of the training outcome on `train_risk`.
///
/// Rows of the training sample with a non-finite time, event, or risk, or with
/// a non-positive time, are dropped. If nothing survives that filter the result
/// is all-ones (no information to calibrate against); if fewer than two rows
/// remain, or the training risk has no spread, every test row receives the same
/// marginal Kaplan-Meier curve.
///
/// Each returned row is non-increasing in time and pinned to `1.0` in the first
/// grid column, so it is a valid survival function regardless of where `grid`
/// starts relative to the first event.
pub fn cox_calibrated_survival_matrix(
    train_times: &[f64],
    train_events: &[f64],
    train_risk: &[f64],
    test_risk: &[f64],
    grid: &[f64],
) -> Result<Array2<f64>, String> {
    if train_times.len() != train_events.len() || train_times.len() != train_risk.len() {
        return Err(format!(
            "survival calibration length mismatch: times={} events={} risk={}",
            train_times.len(),
            train_events.len(),
            train_risk.len()
        ));
    }
    let mut rows: Vec<(f64, f64, f64)> = train_times
        .iter()
        .zip(train_events.iter())
        .zip(train_risk.iter())
        .filter_map(|((&time, &event), &risk)| {
            (time.is_finite() && event.is_finite() && risk.is_finite() && time > 0.0)
                .then_some((time, event, risk))
        })
        .collect();
    if rows.is_empty() {
        return Ok(Array2::<f64>::ones((test_risk.len(), grid.len())));
    }
    let risk_mean = rows.iter().map(|row| row.2).sum::<f64>() / rows.len() as f64;
    let risk_sd = (rows
        .iter()
        .map(|row| {
            let d = row.2 - risk_mean;
            d * d
        })
        .sum::<f64>()
        / rows.len() as f64)
        .sqrt();
    // A spread inside the rounding band of the centering that produced it is
    // no spread: the risk score is constant to the arithmetic, and the
    // calibration is the covariate-free Kaplan–Meier curve.
    let max_abs_risk = rows.iter().fold(0.0_f64, |acc, row| acc.max(row.2.abs()));
    let spread_band = gam_linalg::roundoff::accumulation_growth(rows.len() + 2) * max_abs_risk;
    if rows.len() < 2 || risk_sd <= spread_band {
        let times: Vec<f64> = rows.iter().map(|row| row.0).collect();
        let events: Vec<f64> = rows.iter().map(|row| row.1).collect();
        let curve = km_curve_on_grid(&times, &events, grid);
        let mut out = Array2::<f64>::zeros((test_risk.len(), grid.len()));
        for mut row in out.rows_mut() {
            for (dst, src) in row.iter_mut().zip(curve.iter()) {
                *dst = *src;
            }
        }
        return Ok(out);
    }
    for row in &mut rows {
        row.2 -= risk_mean;
    }
    rows.sort_by(|a, b| a.0.total_cmp(&b.0));

    // Pool ties: one risk-set update per distinct event time, carrying the
    // death count and the summed covariate over the deaths at that time.
    let mut event_times = Vec::<(f64, usize, f64)>::new();
    let mut i = 0usize;
    while i < rows.len() {
        let time = rows[i].0;
        let mut j = i + 1;
        let mut d = usize::from(rows[i].1 > 0.5);
        let mut event_x = if rows[i].1 > 0.5 { rows[i].2 } else { 0.0 };
        while j < rows.len() && rows[j].0 == time {
            if rows[j].1 > 0.5 {
                d += 1;
                event_x += rows[j].2;
            }
            j += 1;
        }
        if d > 0 {
            event_times.push((time, d, event_x));
        }
        i = j;
    }

    // Newton on the Breslow partial-likelihood score. `s0/s1/s2` are the
    // risk-set weighted moments of the covariate at each event time.
    // Risk-set sums are evaluated with the log-sum-exp shift, so no clamp on
    // the linear predictor is needed for them to stay finite: the shift cancels
    // exactly in the mean and variance, and enters the Breslow increment only as
    // `exp(−m − ln Σ)`, which underflows honestly rather than saturating.
    struct RiskSetSums {
        log_s0: f64,
        mean: f64,
        var: f64,
    }
    let risk_set_sums = |beta: f64, time: f64| -> RiskSetSums {
        let mut shift = f64::NEG_INFINITY;
        for &(row_time, _, x) in &rows {
            if row_time >= time {
                shift = shift.max(beta * x);
            }
        }
        let (mut s0, mut s1) = (0.0_f64, 0.0_f64);
        for &(row_time, _, x) in &rows {
            if row_time >= time {
                let w = (beta * x - shift).exp();
                s0 += w;
                s1 += w * x;
            }
        }
        let mean = s1 / s0;
        // The variance is the centered second moment, summed as such: the
        // textbook `E[x²] − E[x]²` cancels catastrophically exactly where it
        // matters, at a slope large enough that the risk-set weights have
        // concentrated and the true variance is `e^{−βΔ}` below the roundoff of
        // the two O(x²) terms — which is where separation has to be told apart
        // from convergence.
        let mut centered = 0.0_f64;
        for &(row_time, _, x) in &rows {
            if row_time >= time {
                let w = (beta * x - shift).exp();
                let d = x - mean;
                centered += w * d * d;
            }
        }
        RiskSetSums {
            log_s0: shift + s0.ln(),
            mean,
            var: centered / s0,
        }
    };
    // Log partial likelihood, score, observed information, and the rounding
    // bands of the likelihood and of the score at `beta`. A band is what
    // "zero" means for this arithmetic: each is the accumulation over
    // `events × (rows + 2)` roundings of terms whose magnitudes are summed
    // alongside the value.
    struct PartialLikelihood {
        ell: f64,
        ell_band: f64,
        score: f64,
        score_band: f64,
        info: f64,
    }
    let partial_likelihood = |beta: f64| -> PartialLikelihood {
        let (mut ell, mut ell_abs, mut score, mut score_abs, mut info) =
            (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
        for &(time, d, event_x) in &event_times {
            let sums = risk_set_sums(beta, time);
            ell += beta * event_x - d as f64 * sums.log_s0;
            ell_abs += (beta * event_x).abs() + d as f64 * sums.log_s0.abs();
            score += event_x - d as f64 * sums.mean;
            score_abs += event_x.abs() + d as f64 * sums.mean.abs();
            info += d as f64 * sums.var;
        }
        let growth =
            gam_linalg::roundoff::accumulation_growth(event_times.len() * (rows.len() + 2));
        PartialLikelihood {
            ell,
            ell_band: growth * ell_abs,
            score,
            score_band: growth * score_abs,
            info,
        }
    };

    let mut beta = 0.0_f64;
    let mut at = partial_likelihood(beta);
    let mut converged = false;
    for _ in 0..NEWTON_MAX_ITERATIONS {
        if !(at.ell.is_finite() && at.score.is_finite() && at.info.is_finite()) {
            return Err(format!(
                "survival calibration: the partial likelihood is not finite at slope {beta} \
                 (score {}, information {}); the risk score separates the outcomes, so no \
                 finite calibration slope exists",
                at.score, at.info
            ));
        }
        // The log partial likelihood is a sum of log conditional event
        // probabilities, so it is at most 0 and equals 0 only when every event
        // is certain given its risk set — separation, where the maximiser is at
        // infinity. In floating point that supremum is reached at a finite
        // slope (the risk-set mean rounds onto the event's own risk and the
        // score becomes an exact zero), so a likelihood inside its own rounding
        // band of 0 is the certificate that no finite slope exists, and it is
        // what a stationarity test alone cannot see.
        if at.ell >= -at.ell_band {
            return Err(format!(
                "survival calibration: every event is certain given its risk set at slope \
                 {beta} (log partial likelihood {} within its rounding band {} of 0); the \
                 risk score separates the outcomes, so no finite calibration slope exists",
                at.ell, at.ell_band
            ));
        }
        if at.info <= 0.0 {
            return Err(format!(
                "survival calibration: the partial likelihood has no curvature at slope \
                 {beta} (score {}, information {}); the risk score separates the outcomes, \
                 so no finite calibration slope exists",
                at.score, at.info
            ));
        }
        // Stationary within the arithmetic: the score equation is solved to
        // its own rounding band.
        if at.score.abs() <= at.score_band {
            converged = true;
            break;
        }
        let step = at.score / at.info;
        // Monotone safeguard within the likelihood's own rounding: the Newton
        // direction ascends a concave log partial likelihood, and near the
        // maximum a step the score still needs can improve the likelihood by
        // less than its rounding band, so a candidate is accepted when it does
        // not measurably decrease it. No representable candidate at all while
        // the score is still outside its band is a stall, reported as one.
        let mut scale = 1.0_f64;
        let mut progressed = false;
        loop {
            let candidate = beta + scale * step;
            if candidate == beta {
                break;
            }
            let at_candidate = partial_likelihood(candidate);
            if at_candidate.ell.is_finite() && at_candidate.ell >= at.ell - at.ell_band {
                beta = candidate;
                at = at_candidate;
                progressed = true;
                break;
            }
            scale *= 0.5;
        }
        if !progressed {
            return Err(format!(
                "survival calibration: the partial-likelihood Newton search stalled at slope \
                 {beta} with score {} above its rounding band {}; no representable slope \
                 improves the likelihood, so no finite calibration slope exists",
                at.score, at.score_band
            ));
        }
    }
    if !converged {
        return Err(format!(
            "survival calibration: the partial-likelihood Newton search did not converge \
             within {NEWTON_MAX_ITERATIONS} iterations (slope {beta}, score {}, information \
             {}); the risk score separates the outcomes, so no finite calibration slope \
             exists",
            at.score, at.info
        ));
    }

    let mut baseline = Vec::<(f64, f64)>::new();
    let mut cumulative = 0.0;
    for &(time, d, _) in &event_times {
        let sums = risk_set_sums(beta, time);
        cumulative += d as f64 * (-sums.log_s0).exp();
        baseline.push((time, cumulative));
    }

    let mut out = Array2::<f64>::zeros((test_risk.len(), grid.len()));
    for (row_idx, &risk) in test_risk.iter().enumerate() {
        let x = if risk.is_finite() {
            risk - risk_mean
        } else {
            0.0
        };
        let log_mult = beta * x;
        let mut step_idx = 0usize;
        let mut h0 = 0.0;
        let mut prev = 1.0;
        for (col_idx, &time) in grid.iter().enumerate() {
            while step_idx < baseline.len() && baseline[step_idx].0 <= time {
                h0 = baseline[step_idx].1;
                step_idx += 1;
            }
            // `S = exp(−H₀·exp(βx))`, formed in log space so an extreme risk gives
            // the honest 0 or 1 instead of a clamped value; monotone in `t` by
            // construction of `H₀`, pinned by `min(prev)` against roundoff.
            let value = if col_idx == 0 || h0 <= 0.0 {
                1.0
            } else {
                (-(h0.ln() + log_mult).exp()).exp().min(prev)
            };
            out[[row_idx, col_idx]] = value;
            prev = value;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A risk score that separates the outcomes has no finite calibration
    /// slope; the search refuses instead of returning its last iterate (#2469).
    #[test]
    fn a_separating_risk_score_is_refused_not_calibrated() {
        // Every subject dies, and the risk score decreases with survival time,
        // so at each event time the event carries the smallest risk in its
        // risk set: the partial likelihood increases without bound as β → −∞.
        let times = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let events = [1.0; 6];
        let risk = [6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        let grid = [0.0, 2.5, 5.0];
        let err = cox_calibrated_survival_matrix(&times, &events, &risk, &[1.5, 4.5], &grid)
            .unwrap_err();
        assert!(
            err.contains("no finite calibration slope"),
            "unexpected message: {err}"
        );
    }

    #[test]
    fn length_mismatch_is_an_error() {
        let err = cox_calibrated_survival_matrix(&[1.0, 2.0], &[1.0], &[0.1, 0.2], &[0.0], &[1.0])
            .unwrap_err();
        assert!(err.contains("length mismatch"), "unexpected message: {err}");
    }

    #[test]
    fn no_usable_training_rows_gives_an_all_ones_surface() {
        let out = cox_calibrated_survival_matrix(
            &[-1.0, f64::NAN],
            &[1.0, 1.0],
            &[0.1, 0.2],
            &[0.0, 0.5],
            &[0.0, 1.0, 2.0],
        )
        .unwrap();
        assert_eq!(out.dim(), (2, 3));
        assert!(out.iter().all(|&v| v == 1.0));
    }

    #[test]
    fn constant_risk_degenerates_to_the_marginal_km_curve() {
        let times = [1.0, 2.0, 3.0, 4.0];
        let events = [1.0, 0.0, 1.0, 1.0];
        let risk = [0.5; 4];
        let grid = [0.0, 1.5, 2.5, 3.5];
        let out =
            cox_calibrated_survival_matrix(&times, &events, &risk, &[0.2, 0.9], &grid).unwrap();
        let km = km_curve_on_grid(&times, &events, &grid);
        for row in out.rows() {
            for (got, want) in row.iter().zip(km.iter()) {
                assert_eq!(got, want);
            }
        }
    }

    #[test]
    fn higher_risk_survives_less_and_rows_are_monotone() {
        // The risk score is mostly aligned with early failure, so beta > 0 and
        // the high-risk test subject must sit below the low-risk one — but it
        // must NOT be perfectly rank-aligned: with risks strictly decreasing in
        // time (the fixture this replaced) every event carries the largest risk
        // in its risk set, the log partial likelihood's supremum is 0, and the
        // calibration is refused (see the separation test above). The ordering
        // is broken at times 2 and 4, so a finite slope (≈ 0.74) exists.
        let times = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let events = [1.0, 1.0, 1.0, 1.0, 1.0, 0.0];
        let risk = [3.0, 1.0, 2.5, 0.5, 2.0, 0.0];
        let grid = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let out = cox_calibrated_survival_matrix(&times, &events, &risk, &[0.0, 3.0], &grid)
            .expect("calibration must succeed");
        for row in out.rows() {
            assert_eq!(row[0], 1.0, "first grid column is pinned to 1");
            for pair in row.as_slice().unwrap().windows(2) {
                assert!(pair[1] <= pair[0], "survival must be non-increasing");
            }
        }
        let last = grid.len() - 1;
        assert!(
            out[[1, last]] < out[[0, last]],
            "high risk {} must survive less than low risk {}",
            out[[1, last]],
            out[[0, last]]
        );
    }
}
