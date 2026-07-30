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

/// Newton iterations for the one-parameter partial-likelihood score equation.
/// A scalar Newton step on a strictly concave log partial likelihood converges
/// quadratically; the cap only bounds pathological data (complete separation),
/// where `NEWTON_STEP_CLAMP` already keeps the iterate finite.
const NEWTON_MAX_ITERATIONS: usize = 50;

/// Ridge added to the score and information. Guarantees a finite step under
/// complete separation, where the unpenalized information tends to zero.
const SCORE_RIDGE: f64 = 1.0e-8;

/// Clamp on the linear predictor before `exp`. `exp(50)` is ~5e21, already far
/// beyond any risk ratio a calibration is meaningful at, and keeps the risk-set
/// sums finite instead of saturating to `inf` and poisoning every ratio.
const LINEAR_PREDICTOR_CLAMP: f64 = 50.0;

/// Trust region on a single Newton step, in units of `β`.
const NEWTON_STEP_CLAMP: f64 = 2.0;

/// Convergence tolerance on the Newton step.
const NEWTON_STEP_TOLERANCE: f64 = 1.0e-10;

/// Floor on a reported survival probability, so a downstream `ln` is finite.
const SURVIVAL_FLOOR: f64 = 1.0e-12;

/// Risk scores whose spread is below this are treated as constant, and the
/// calibration degenerates to the covariate-free Kaplan-Meier curve.
const RISK_SPREAD_FLOOR: f64 = 1.0e-12;

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
    if rows.len() < 2 || risk_sd < RISK_SPREAD_FLOOR {
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
    let mut beta = 0.0;
    for _ in 0..NEWTON_MAX_ITERATIONS {
        let mut score = -SCORE_RIDGE * beta;
        let mut info = SCORE_RIDGE;
        for &(time, d, event_x) in &event_times {
            let mut s0 = 0.0;
            let mut s1 = 0.0;
            let mut s2 = 0.0;
            for &(row_time, _, x) in &rows {
                if row_time >= time {
                    let w = (beta * x)
                        .clamp(-LINEAR_PREDICTOR_CLAMP, LINEAR_PREDICTOR_CLAMP)
                        .exp();
                    s0 += w;
                    s1 += w * x;
                    s2 += w * x * x;
                }
            }
            if s0 > 0.0 {
                let mean = s1 / s0;
                score += event_x - d as f64 * mean;
                info += d as f64 * (s2 / s0 - mean * mean).max(0.0);
            }
        }
        if !info.is_finite() || info <= 0.0 {
            break;
        }
        let step = (score / info).clamp(-NEWTON_STEP_CLAMP, NEWTON_STEP_CLAMP);
        beta += step;
        if step.abs() < NEWTON_STEP_TOLERANCE {
            break;
        }
    }

    // Breslow baseline cumulative hazard at the converged beta.
    let mut baseline = Vec::<(f64, f64)>::new();
    let mut cumulative = 0.0;
    for &(time, d, _) in &event_times {
        let mut s0 = 0.0;
        for &(row_time, _, x) in &rows {
            if row_time >= time {
                s0 += (beta * x)
                    .clamp(-LINEAR_PREDICTOR_CLAMP, LINEAR_PREDICTOR_CLAMP)
                    .exp();
            }
        }
        if s0 > 0.0 {
            cumulative += d as f64 / s0;
            baseline.push((time, cumulative));
        }
    }

    let mut out = Array2::<f64>::zeros((test_risk.len(), grid.len()));
    for (row_idx, &risk) in test_risk.iter().enumerate() {
        let x = if risk.is_finite() {
            risk - risk_mean
        } else {
            0.0
        };
        let mult = (beta * x)
            .clamp(-LINEAR_PREDICTOR_CLAMP, LINEAR_PREDICTOR_CLAMP)
            .exp();
        let mut step_idx = 0usize;
        let mut h0 = 0.0;
        let mut prev = 1.0;
        for (col_idx, &time) in grid.iter().enumerate() {
            while step_idx < baseline.len() && baseline[step_idx].0 <= time {
                h0 = baseline[step_idx].1;
                step_idx += 1;
            }
            let value = if col_idx == 0 {
                1.0
            } else {
                (-(h0 * mult)).exp().clamp(SURVIVAL_FLOOR, 1.0).min(prev)
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
        // Risk score is perfectly rank-aligned with early failure, so beta > 0
        // and the high-risk test subject must sit below the low-risk one.
        let times = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let events = [1.0, 1.0, 1.0, 1.0, 1.0, 0.0];
        let risk = [3.0, 2.5, 2.0, 1.0, 0.5, 0.0];
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
