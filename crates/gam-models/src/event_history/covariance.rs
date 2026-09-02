//! The latent covariance an event-history fit reports, and the covariance
//! score that grows its rank.
//!
//! The atoms have unit variance, so with loadings `A` (marks × atoms) and
//! rates `r_k` the latent part of the log-intensities has covariance
//!
//! ```text
//! C(Δ) = A diag(e^{−r_k |Δ|}) Aᵀ,        C(0) = A Aᵀ.
//! ```
//!
//! `C` is the scientific object: two atoms with equal rates can be rotated
//! into each other without changing the process, but `C` is invariant. The
//! loadings are its factor coordinates.
//!
//! The rank is grown from zero by the evidence. At rank `K`, adding an atom
//! with loading vector `v` and rate `r` changes the evidence to second order
//! in `v` by `½ vᵀ M(r) v` (the first order vanishes: an atom is symmetric
//! under `v → −v`), with the covariance score
//!
//! ```text
//! M(r) = Σ_i [ Σ_{n,m} e^{−r |t_n − t_m|} s̄_{in} s̄_{im}ᵀ − Σ_n diag(c̄_{in}) ],
//! ```
//!
//! `s̄_{nd} = y_{nd} − w_{nd} μ̄_{nd}` the residual scores at the current
//! fit's posterior-mean intensities and `c̄_{nd} = w_{nd} μ̄_{nd}` their
//! curvatures. This is the score of the evidence with respect to the
//! covariance operator in the direction `v vᵀ e^{−r|Δ|}`; its top eigenpair
//! at the rate that maximises the top eigenvalue is the most
//! evidence-improving covariance direction the current rank omits, and it
//! initialises the next atom without any arbitrary symmetry breaking. The
//! double sum is evaluated by the forward–backward recursion of the
//! exponential kernel in `O(N D)` per subject, and so are its first two
//! derivatives in the log-rate, so the rate is found by Newton's method.

use super::cohort::EventHistoryError;
use faer::Side;
use gam_linalg::faer_ndarray::strict_symmetric_eigh;
use ndarray::{Array1, Array2};

/// `C(0) = A Aᵀ`.
pub fn disease_covariance(loadings: &Array2<f64>) -> Array2<f64> {
    loadings.dot(&loadings.t())
}

/// `C(Δ) = A diag(e^{−r_k |Δ|}) Aᵀ` for rates in the data's time unit.
pub fn temporal_covariance(loadings: &Array2<f64>, rates: &[f64], lag: f64) -> Array2<f64> {
    let (marks, atoms) = loadings.dim();
    let mut out = Array2::<f64>::zeros((marks, marks));
    for k in 0..atoms {
        let decay = (-rates[k] * lag.abs()).exp();
        for d in 0..marks {
            for e in 0..marks {
                out[[d, e]] += loadings[[d, k]] * loadings[[e, k]] * decay;
            }
        }
    }
    out
}

/// Eigenvalues (descending) and matching unit eigenvectors (columns) of a
/// symmetric matrix.
pub fn eigenmodes(matrix: &Array2<f64>) -> Result<(Array1<f64>, Array2<f64>), EventHistoryError> {
    let n = matrix.nrows();
    if n == 0 {
        return Ok((Array1::zeros(0), Array2::zeros((0, 0))));
    }
    let (values, vectors) = strict_symmetric_eigh(matrix, Side::Lower).map_err(|error| {
        EventHistoryError::NumericalFailure {
            reason: format!("eigendecomposition of a {n} × {n} covariance failed: {error}"),
        }
    })?;
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| values[b].total_cmp(&values[a]));
    let sorted_values = Array1::from_iter(order.iter().map(|&i| values[i]));
    let mut sorted_vectors = Array2::<f64>::zeros((n, n));
    for (j, &i) in order.iter().enumerate() {
        sorted_vectors.column_mut(j).assign(&vectors.column(i));
    }
    Ok((sorted_values, sorted_vectors))
}

/// One subject's residual scores at the current fit.
pub(crate) struct SubjectResiduals {
    pub times: Vec<f64>,
    /// `s̄_{nd}`, index `n * marks + d`.
    pub scores: Vec<f64>,
    /// `c̄_{nd}`, index `n * marks + d`.
    pub curvatures: Vec<f64>,
}

/// `Σ_m |t_n − t_m|^j e^{−r |t_n − t_m|} x_m` for `j = 0, 1, 2` at every
/// `n`, for a vector-valued sequence `x` of width `width`, by the
/// forward–backward recursion of the exponential kernel.
fn kernel_sums(times: &[f64], x: &[f64], width: usize, rate: f64) -> [Vec<f64>; 3] {
    let n = times.len();
    let mut forward = [vec![0.0; n * width], vec![0.0; n * width], vec![0.0; n * width]];
    let mut backward = [vec![0.0; n * width], vec![0.0; n * width], vec![0.0; n * width]];
    for i in 0..n {
        let (decay, delta) = if i == 0 {
            (0.0, 0.0)
        } else {
            let delta = times[i] - times[i - 1];
            ((-rate * delta).exp(), delta)
        };
        for c in 0..width {
            let idx = i * width + c;
            let prev = if i == 0 { 0 } else { (i - 1) * width + c };
            let (f0, f1, f2) = if i == 0 {
                (0.0, 0.0, 0.0)
            } else {
                (forward[0][prev], forward[1][prev], forward[2][prev])
            };
            forward[0][idx] = decay * f0 + x[idx];
            forward[1][idx] = decay * (f1 + delta * f0);
            forward[2][idx] = decay * (f2 + 2.0 * delta * f1 + delta * delta * f0);
        }
    }
    for i in (0..n).rev() {
        let (decay, delta) = if i + 1 == n {
            (0.0, 0.0)
        } else {
            let delta = times[i + 1] - times[i];
            ((-rate * delta).exp(), delta)
        };
        for c in 0..width {
            let idx = i * width + c;
            let next = if i + 1 == n { 0 } else { (i + 1) * width + c };
            let (b0, b1, b2) = if i + 1 == n {
                (0.0, 0.0, 0.0)
            } else {
                (backward[0][next], backward[1][next], backward[2][next])
            };
            backward[0][idx] = decay * b0 + x[idx];
            backward[1][idx] = decay * (b1 + delta * b0);
            backward[2][idx] = decay * (b2 + 2.0 * delta * b1 + delta * delta * b0);
        }
    }
    let mut out = forward;
    for c in 0..n * width {
        out[0][c] += backward[0][c] - x[c];
        out[1][c] += backward[1][c];
        out[2][c] += backward[2][c];
    }
    out
}

/// `M`, `dM/dρ`, `d²M/dρ²` at log-rate `ρ` (`rate = e^ρ / time_scale`).
pub(crate) fn covariance_score(
    subjects: &[SubjectResiduals],
    marks: usize,
    log_rate: f64,
    time_scale: f64,
) -> [Array2<f64>; 3] {
    let rate = log_rate.exp() / time_scale;
    let mut m0 = Array2::<f64>::zeros((marks, marks));
    let mut m1 = Array2::<f64>::zeros((marks, marks));
    let mut m2 = Array2::<f64>::zeros((marks, marks));
    for subject in subjects {
        let n = subject.times.len();
        let sums = kernel_sums(&subject.times, &subject.scores, marks, rate);
        for node in 0..n {
            for d in 0..marks {
                let s = subject.scores[node * marks + d];
                if s == 0.0 {
                    continue;
                }
                for e in 0..marks {
                    let idx = node * marks + e;
                    m0[[d, e]] += s * sums[0][idx];
                    m1[[d, e]] -= s * sums[1][idx];
                    m2[[d, e]] += s * sums[2][idx];
                }
            }
            for d in 0..marks {
                m0[[d, d]] -= subject.curvatures[node * marks + d];
            }
        }
    }
    // Symmetrise the double sums, then chain to the log-rate.
    let symmetric = |m: &Array2<f64>| 0.5 * (m + &m.t());
    let m0 = symmetric(&m0);
    let d1 = symmetric(&m1) * rate;
    let d2 = &d1 + symmetric(&m2) * (rate * rate);
    [m0, d1, d2]
}

/// The atom the covariance score proposes: its loading vector (the top
/// eigenvector of `M` scaled to the one-step estimate of its variance), its
/// log-rate, and the score's top eigenvalue there.
#[derive(Clone, Debug)]
pub(crate) struct NewAtom {
    pub log_rate: f64,
    pub loading: Vec<f64>,
    pub eigenvalue: f64,
    /// The one-step estimate of the variance component along the direction.
    pub variance: f64,
    /// The proposal wanted a rate the node mesh cannot resolve and was held
    /// at [`resolvable_log_rates`]'s upper limit: the residuals carry
    /// structure faster than the quadrature mesh, and a finer mesh would
    /// see more of it.
    pub at_resolution_limit: bool,
}

/// The log-rates the cohort's own node mesh can represent, as
/// `(lower, upper)` on `ρ = ln(rate · T̄)`.
///
/// Below `lower` the kernel is one to double precision across the longest
/// follow-up: the atom is a static frailty, and every slower rate is the
/// same model. That is a floating-point statement, and the fit is free to
/// sit on it — a static frailty is a perfectly good latent state.
///
/// Above `upper` the atom decorrelates inside one cell of the quadrature
/// mesh, and there the discretisation stops representing the process it
/// stands for. The limit matters because the likelihood diverges beyond it:
/// an event node carries a count but no exposure, so once its neighbours
/// decorrelate, its latent coordinate is held only by the prior and
/// `y a z − ½z²` is maximised at `z = a` with value `a²/2` — free evidence
/// per event, growing without bound in the loading. The continuous-time
/// model has no such corner (the exponential of white noise is not a random
/// measure, so those rates are not in the parameter space at all); the mesh
/// invents it. `κ = rate · gap ≤ 1` at the median gap is the statement that
/// consecutive nodes still share information, so the fit is measuring the
/// data rather than its own mesh. A proposal that wants to sit on this
/// limit is telling the caller the mesh is too coarse for the process the
/// residuals show, not that the process is infinitely fast.
pub(crate) fn resolvable_log_rates(
    subjects: &[SubjectResiduals],
    time_scale: f64,
) -> Result<Option<(f64, f64)>, EventHistoryError> {
    let mut longest = 0.0_f64;
    let mut gaps: Vec<f64> = Vec::new();
    for subject in subjects {
        if let (Some(first), Some(last)) = (subject.times.first(), subject.times.last()) {
            longest = longest.max(last - first);
        }
        gaps.extend(subject.times.windows(2).map(|w| w[1] - w[0]).filter(|g| *g > 0.0));
    }
    if !(longest > 0.0) || gaps.is_empty() || !(time_scale.is_finite() && time_scale > 0.0) {
        return Ok(None);
    }
    gaps.sort_by(f64::total_cmp);
    let median = gaps[gaps.len() / 2];
    let lower = (f64::EPSILON.sqrt() * time_scale / longest).ln();
    let upper = (time_scale / median).ln();
    if !(lower.is_finite() && upper.is_finite()) || upper <= lower {
        return Ok(None);
    }
    Ok(Some((lower, upper)))
}

/// The standardised evidence gain of the proposed direction at log-rate
/// `ρ`, with its exact derivative in `ρ`.
///
/// The score of the variance component `τ` along `v` at `τ = 0` is
/// `U = ½ vᵀM(r)v`, and its information is the variance of that score under
/// the fitted null. For a marked counting process the compensated score
/// `s = y − wμ` has, at each node and mark, the Poisson cumulants
/// `Var s = c`, `Var s² = c + 2c²`, so
///
/// ```text
/// Var U = ½ Σ_{n,m} e^{−2r|t_n−t_m|} κ_n κ_m + ¼ Σ_n Σ_d v_d⁴ c_{nd},
/// κ_n = Σ_d v_d² c_{nd}.
/// ```
///
/// The second term is the process's own diagonal — the variance a count
/// carries at a single instant — and it is what stops the standardised gain
/// from running away as the kernel narrows: the smooth part collapses to
/// `Σ_n κ_n²` there while the diagonal stays, so a kernel narrower than the
/// data's own structure buys nothing. Dropping it (the Gaussian-response
/// form of the same statistic) sends the proposal to white noise on any
/// cohort whose residuals are event spikes, which is every point process.
struct GainPoint {
    rho: f64,
    gain: f64,
    slope: f64,
    top: f64,
    vector: Vec<f64>,
    information: f64,
}

/// Propose the next atom: at every log-rate the top eigenpair `(λ, v)` of
/// `M(ρ)` is the direction with the largest second-order evidence slope
/// `½ λ` in its variance `τ`; the Fisher information of that slope is
/// `½ J` with `J = Σ_i Σ_{n,m} e^{−2r|t_n − t_m|} κ_n κ_m`,
/// `κ_n = Σ_d v_d² c̄_{nd}`, so the second-order model's best gain is
/// `λ² / (4 J)` at `τ̂ = λ / J`. The raw slope is always largest at rate
/// zero (a slower kernel dominates every faster one entrywise); the
/// standardised gain is the matched filter, largest at the rate that
/// generated the residual correlation. It is maximised over `ρ` by a secant
/// Newton on its exact derivative (Hellmann–Feynman for `λ`, first-order
/// eigenvector perturbation for `v`, the kernel recursion for `J`).
/// `None` when no direction raises the evidence.
pub(crate) fn best_new_atom(
    subjects: &[SubjectResiduals],
    marks: usize,
    time_scale: f64,
) -> Result<Option<NewAtom>, EventHistoryError> {
    let evaluate = |rho: f64| -> Result<GainPoint, EventHistoryError> {
        let [m0, m1, _] = covariance_score(subjects, marks, rho, time_scale);
        let (values, vectors) = eigenmodes(&m0)?;
        let top = values[0];
        let v: Vec<f64> = vectors.column(0).to_vec();
        let vv = Array1::from(v.clone());
        let m1v = m1.dot(&vv);
        let top_slope = vv.dot(&m1v);
        let mut v_slope = vec![0.0; marks];
        for j in 1..marks {
            let gap = top - values[j];
            if gap > 0.0 {
                let coupling = vectors.column(j).dot(&m1v) / gap;
                for d in 0..marks {
                    v_slope[d] += coupling * vectors[[d, j]];
                }
            }
        }
        let rate = rho.exp() / time_scale;
        // `information` is twice the score's variance, so that the one-step
        // variance is `λ / information` and the gain `λ² / (4 information)`.
        let mut information = 0.0;
        let mut information_slope = 0.0;
        for subject in subjects {
            let n = subject.times.len();
            let kappa: Vec<f64> = (0..n)
                .map(|node| (0..marks).map(|d| v[d] * v[d] * subject.curvatures[node * marks + d]).sum())
                .collect();
            let kappa_slope: Vec<f64> = (0..n)
                .map(|node| {
                    (0..marks)
                        .map(|d| 2.0 * v[d] * v_slope[d] * subject.curvatures[node * marks + d])
                        .sum()
                })
                .collect();
            let sums = kernel_sums(&subject.times, &kappa, 1, 2.0 * rate);
            for node in 0..n {
                information += kappa[node] * sums[0][node];
                // d/dρ of e^{−2r|Δ|} is −2 r |Δ| e^{−2r|Δ|}; the quadratic
                // form is symmetric, so the κ derivative enters twice.
                information_slope += -2.0 * rate * kappa[node] * sums[1][node]
                    + 2.0 * kappa_slope[node] * sums[0][node];
                // The process's own diagonal, ½ Σ_d v_d⁴ c: independent of
                // the rate, so it enters the slope only through `v`.
                for d in 0..marks {
                    let c = subject.curvatures[node * marks + d];
                    let v2 = v[d] * v[d];
                    information += 0.5 * v2 * v2 * c;
                    information_slope += 2.0 * v2 * v[d] * v_slope[d] * c;
                }
            }
        }
        if !(information > 0.0) {
            return Ok(GainPoint {
                rho,
                gain: 0.0,
                slope: 0.0,
                top,
                vector: v,
                information,
            });
        }
        // Signed gain λ|λ| / (4J): odd in λ, so the ascent also moves a
        // negative top eigenvalue toward where it turns positive.
        let gain = top * top.abs() / (4.0 * information);
        let slope = (2.0 * top.abs() * top_slope * information - top * top.abs() * information_slope)
            / (4.0 * information * information);
        Ok(GainPoint {
            rho,
            gain,
            slope,
            top,
            vector: v,
            information,
        })
    };
    let Some((lower, upper)) = resolvable_log_rates(subjects, time_scale)? else {
        return Ok(None);
    };
    // Start at the cohort's own time scale (rate = 1 / T̄).
    let mut point = evaluate(0.0_f64.clamp(lower, upper))?;
    let mut curvature: Option<f64> = None;
    for _ in 0..100 {
        let tolerance = f64::EPSILON.sqrt() * (1.0 + point.gain.abs());
        if !(point.slope.abs() > tolerance) {
            break;
        }
        // Secant Newton on the exact slope; a unit step in log-rate when no
        // negative curvature is known yet, never more than two per step.
        let direction = match curvature {
            Some(h) if h < 0.0 => (-point.slope / h).clamp(-2.0, 2.0),
            _ => point.slope.signum(),
        };
        let mut t = 1.0;
        let mut moved = false;
        for _ in 0..60 {
            let target = (point.rho + t * direction).clamp(lower, upper);
            if target == point.rho {
                break;
            }
            let trial = evaluate(target)?;
            if trial.gain > point.gain {
                curvature = Some((trial.slope - point.slope) / (trial.rho - point.rho));
                point = trial;
                moved = true;
                break;
            }
            t *= 0.5;
        }
        if !moved {
            break;
        }
    }
    if !(point.top > 0.0) || !(point.information > 0.0) {
        return Ok(None);
    }
    let variance = point.top / point.information;
    let scale = variance.sqrt();
    Ok(Some(NewAtom {
        log_rate: point.rho,
        loading: point.vector.iter().map(|x| scale * x).collect(),
        eigenvalue: point.top,
        variance,
        at_resolution_limit: point.rho >= upper - f64::EPSILON.sqrt() * (1.0 + upper.abs()),
    }))
}
