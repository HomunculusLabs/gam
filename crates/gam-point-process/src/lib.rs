//! Finite-state marked point processes with Matérn dynamics.
//!
//! This module is the direct state-space form of a piecewise-exponential
//! marked point-process model.  On risk interval `k`, mark `d` has intensity
//!
//! ```text
//! log lambda[k,d] = fixed_log_intensity[k,d] + loading[d]' z[k].
//! ```
//!
//! `z` is a stack of independent Matérn-1/2 or Matérn-3/2 factors.  Their
//! transition and innovation covariance are analytic, so the Gaussian prior
//! precision is block tridiagonal.  Conditional on the model parameters, the
//! point-process log likelihood is concave in `z`; [`smooth_laplace`] therefore
//! finds its unique mode with block Newton solves in linear time in the number
//! of intervals.  [`filter_laplace`] is the online assumed-Gaussian analogue:
//! it is constant-memory and constant-work per interval for fixed state size.
//!
//! Two approximation boundaries are deliberately visible in the API:
//!
//! * the intensity is constant within each supplied [`RiskInterval`];
//! * point-process filtering and posterior integration are Laplace/Monte Carlo,
//!   respectively.  The Matérn state transition itself is exact.

use ndarray::{Array1, Array2, ArrayView1, Axis};
use rand::{SeedableRng, rngs::StdRng};
use rand_distr::{Distribution, Poisson};
use serde::{Deserialize, Serialize};
use statrs::function::gamma::ln_gamma;
use thiserror::Error;

const LOG_TWO_PI: f64 = 1.8378770664093453;

/// Errors from construction, inference, or prediction of a marked process.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum MarkedPointProcessError {
    #[error("{reason}")]
    InvalidInput { reason: String },
    #[error("{context} is not positive definite at diagonal {diagonal}: {value}")]
    NonPositiveDefinite {
        context: &'static str,
        diagonal: usize,
        value: f64,
    },
    #[error("{context} is not positive semidefinite at diagonal {diagonal}: {value}")]
    NonPositiveSemidefinite {
        context: &'static str,
        diagonal: usize,
        value: f64,
    },
    #[error(
        "Laplace mode did not converge in {iterations} iterations (stationarity={stationarity})"
    )]
    NonConvergence {
        iterations: usize,
        stationarity: f64,
    },
    #[error(
        "Laplace hyperparameter search did not converge in {evaluations} evaluations (maximum transformed step={maximum_step})"
    )]
    HyperparameterNonConvergence {
        evaluations: usize,
        maximum_step: f64,
    },
    #[error("Newton line search could not find an improving finite step")]
    LineSearchFailure,
    #[error("non-finite numerical result while computing {context}")]
    NumericalFailure { context: &'static str },
}

/// The two scalar stationary Matérn processes with exact finite Markov states.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MaternMarkovOrder {
    /// Matérn `nu = 1/2`, an Ornstein–Uhlenbeck process with state `z`.
    Half,
    /// Matérn `nu = 3/2`, with state `(z, dz/dt)`.
    ThreeHalves,
}

impl MaternMarkovOrder {
    /// Number of state coordinates required by one factor.
    pub const fn state_dimension(self) -> usize {
        match self {
            Self::Half => 1,
            Self::ThreeHalves => 2,
        }
    }
}

/// One independent stationary Matérn factor.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MaternFactor {
    pub order: MaternMarkovOrder,
    /// Stationary marginal variance of the function value `z(t)`.
    pub marginal_variance: f64,
    /// Matérn length scale in the same units as interval times.
    pub length_scale: f64,
}

impl MaternFactor {
    fn validate(self, factor: usize) -> Result<(), MarkedPointProcessError> {
        if !self.marginal_variance.is_finite() || self.marginal_variance <= 0.0 {
            return Err(invalid(format!(
                "factor {factor} marginal_variance must be finite and positive"
            )));
        }
        if !self.length_scale.is_finite() || self.length_scale <= 0.0 {
            return Err(invalid(format!(
                "factor {factor} length_scale must be finite and positive"
            )));
        }
        if self.order == MaternMarkovOrder::ThreeHalves {
            let rate = 3.0_f64.sqrt() / self.length_scale;
            let derivative_scale = rate * self.marginal_variance.sqrt();
            let derivative_variance = derivative_scale * derivative_scale;
            if !rate.is_finite()
                || rate <= 0.0
                || !derivative_variance.is_finite()
                || derivative_variance <= 0.0
            {
                return Err(invalid(format!(
                    "factor {factor} Matérn-3/2 derivative scale is not representable"
                )));
            }
        }
        Ok(())
    }
}

/// Exact discrete-time transition of a stationary continuous-time state.
#[derive(Debug, Clone, PartialEq)]
pub struct StateTransition {
    pub transition: Array2<f64>,
    pub innovation_covariance: Array2<f64>,
}

/// Semantic role of an observed mark.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MarkRole {
    /// A one-time outcome which may participate in a competing-risk forecast.
    Absorbing,
    /// A repeatable outcome which informs the state but does not end follow-up.
    Recurrent,
    /// A healthcare-contact event.  It shares the state through its own loading
    /// and is excluded from competing-risk sets unless explicitly requested.
    Encounter,
}

/// Model parameters held fixed during the conditional latent-state solve.
///
/// The rows of `loadings` correspond to marks and its columns to Matérn factor
/// values (not derivative coordinates). `mark_impulses` maps a count at an
/// interval endpoint into an instantaneous change of the full Markov state;
/// this is the rational distributed-lag representation `exp(A lag) B`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MarkedPointProcessModel {
    pub factors: Vec<MaternFactor>,
    pub mark_names: Vec<String>,
    pub mark_roles: Vec<MarkRole>,
    pub loadings: Array2<f64>,
    pub mark_impulses: Array2<f64>,
}

impl MarkedPointProcessModel {
    /// Validate dimensions, names, roles, and all numerical parameters.
    pub fn validate(&self) -> Result<(), MarkedPointProcessError> {
        if self.factors.is_empty() {
            return Err(invalid("at least one Matérn factor is required"));
        }
        if self.mark_names.is_empty() {
            return Err(invalid("at least one mark is required"));
        }
        for (factor, spec) in self.factors.iter().copied().enumerate() {
            spec.validate(factor)?;
        }
        let marks = self.mark_names.len();
        if self.mark_roles.len() != marks {
            return Err(invalid(format!(
                "mark_roles has length {}, expected {marks}",
                self.mark_roles.len()
            )));
        }
        if self.mark_names.iter().any(String::is_empty) {
            return Err(invalid("mark names must be non-empty"));
        }
        for left in 0..marks {
            if self.mark_names[left + 1..]
                .iter()
                .any(|right| right == &self.mark_names[left])
            {
                return Err(invalid(format!(
                    "mark name {:?} is duplicated",
                    self.mark_names[left]
                )));
            }
        }
        if self.loadings.dim() != (marks, self.factors.len()) {
            return Err(invalid(format!(
                "loadings has shape {:?}, expected ({marks}, {})",
                self.loadings.dim(),
                self.factors.len()
            )));
        }
        let state_dimension = self.state_dimension();
        if self.mark_impulses.dim() != (state_dimension, marks) {
            return Err(invalid(format!(
                "mark_impulses has shape {:?}, expected ({state_dimension}, {marks})",
                self.mark_impulses.dim()
            )));
        }
        ensure_finite_matrix(&self.loadings, "loadings")?;
        ensure_finite_matrix(&self.mark_impulses, "mark_impulses")?;
        Ok(())
    }

    /// Dimension of the stacked Markov state, including derivative coordinates.
    pub fn state_dimension(&self) -> usize {
        self.factors
            .iter()
            .map(|factor| factor.order.state_dimension())
            .sum()
    }

    /// Number of event marks represented by the model.
    pub fn mark_count(&self) -> usize {
        self.mark_names.len()
    }

    fn factor_offsets(&self) -> Vec<usize> {
        let mut offsets = Vec::with_capacity(self.factors.len());
        let mut offset = 0;
        for factor in &self.factors {
            offsets.push(offset);
            offset += factor.order.state_dimension();
        }
        offsets
    }

    /// Mark-by-state observation matrix. Derivative state coordinates have zero
    /// direct loading; they affect future intensities through the transition.
    pub fn observation_matrix(&self) -> Array2<f64> {
        let mut out = Array2::zeros((self.mark_count(), self.state_dimension()));
        for (factor, &offset) in self.factor_offsets().iter().enumerate() {
            for mark in 0..self.mark_count() {
                out[[mark, offset]] = self.loadings[[mark, factor]];
            }
        }
        out
    }

    /// Stationary covariance of the full Markov state.
    pub fn stationary_covariance(&self) -> Result<Array2<f64>, MarkedPointProcessError> {
        self.validate()?;
        let mut covariance = Array2::zeros((self.state_dimension(), self.state_dimension()));
        for (factor, &offset) in self.factors.iter().zip(self.factor_offsets().iter()) {
            let variance = factor.marginal_variance;
            covariance[[offset, offset]] = variance;
            if factor.order == MaternMarkovOrder::ThreeHalves {
                let rate = 3.0_f64.sqrt() / factor.length_scale;
                let derivative_scale = rate * variance.sqrt();
                covariance[[offset + 1, offset + 1]] = derivative_scale * derivative_scale;
            }
        }
        ensure_finite_matrix(&covariance, "stationary state covariance")?;
        Ok(covariance)
    }

    /// Exact Matérn state transition over a positive elapsed time.
    pub fn transition(&self, elapsed: f64) -> Result<StateTransition, MarkedPointProcessError> {
        self.validate()?;
        if !elapsed.is_finite() || elapsed <= 0.0 {
            return Err(invalid(
                "transition elapsed time must be finite and positive",
            ));
        }
        let dimension = self.state_dimension();
        let mut transition = Array2::zeros((dimension, dimension));
        let mut innovation = Array2::zeros((dimension, dimension));
        for (factor, &offset) in self.factors.iter().zip(self.factor_offsets().iter()) {
            let variance = factor.marginal_variance;
            match factor.order {
                MaternMarkovOrder::Half => {
                    let scaled_time = elapsed / factor.length_scale;
                    if scaled_time == 0.0 {
                        return Err(MarkedPointProcessError::NumericalFailure {
                            context: "Matérn-1/2 elapsed-time resolution",
                        });
                    }
                    if scaled_time.is_infinite() {
                        innovation[[offset, offset]] = variance;
                    } else {
                        let decay = (-scaled_time).exp();
                        transition[[offset, offset]] = decay;
                        innovation[[offset, offset]] = variance * -(-2.0 * scaled_time).exp_m1();
                    }
                    if !innovation[[offset, offset]].is_finite()
                        || innovation[[offset, offset]] <= 0.0
                    {
                        return Err(MarkedPointProcessError::NonPositiveDefinite {
                            context: "Matérn-1/2 innovation covariance",
                            diagonal: offset,
                            value: innovation[[offset, offset]],
                        });
                    }
                }
                MaternMarkovOrder::ThreeHalves => {
                    let rate = 3.0_f64.sqrt() / factor.length_scale;
                    let position_scale = variance.sqrt();
                    let derivative_scale = rate * position_scale;
                    let derivative_variance = derivative_scale * derivative_scale;
                    let scaled_time = rate * elapsed;
                    if scaled_time == 0.0 {
                        return Err(MarkedPointProcessError::NumericalFailure {
                            context: "Matérn-3/2 elapsed-time resolution",
                        });
                    }
                    let decay = (-scaled_time).exp();
                    if scaled_time.is_infinite() || decay == 0.0 {
                        innovation[[offset, offset]] = variance;
                        innovation[[offset + 1, offset + 1]] = derivative_variance;
                        continue;
                    }
                    transition[[offset, offset]] = decay * (1.0 + scaled_time);
                    transition[[offset, offset + 1]] = decay * elapsed;
                    transition[[offset + 1, offset]] = -(decay * rate) * scaled_time;
                    transition[[offset + 1, offset + 1]] = decay * (1.0 - scaled_time);

                    let twice_scaled_time = 2.0 * scaled_time;
                    let decay2 = (-twice_scaled_time).exp();
                    let scaled_time2 = scaled_time * scaled_time;
                    innovation[[offset, offset]] = variance
                        * matern_three_halves_position_innovation_fraction(twice_scaled_time);
                    let variance_rate = position_scale * derivative_scale;
                    let cross = 2.0 * variance_rate * decay2 * scaled_time2;
                    innovation[[offset, offset + 1]] = cross;
                    innovation[[offset + 1, offset]] = cross;
                    innovation[[offset + 1, offset + 1]] = derivative_variance
                        * (-(-twice_scaled_time).exp_m1()
                            + decay2
                                * (twice_scaled_time
                                    - 0.5 * twice_scaled_time * twice_scaled_time));
                    let position_pivot = innovation[[offset, offset]];
                    if !position_pivot.is_finite() || position_pivot <= 0.0 {
                        return Err(MarkedPointProcessError::NonPositiveDefinite {
                            context: "Matérn-3/2 innovation covariance",
                            diagonal: offset,
                            value: position_pivot,
                        });
                    }
                    let derivative_residual = cross / position_pivot.sqrt();
                    let derivative_pivot =
                        innovation[[offset + 1, offset + 1]] - derivative_residual.powi(2);
                    if !derivative_pivot.is_finite() || derivative_pivot <= 0.0 {
                        return Err(MarkedPointProcessError::NonPositiveDefinite {
                            context: "Matérn-3/2 innovation covariance",
                            diagonal: offset + 1,
                            value: derivative_pivot,
                        });
                    }
                }
            }
        }
        ensure_finite_matrix(&transition, "state transition")?;
        ensure_finite_matrix(&innovation, "state innovation covariance")?;
        Ok(StateTransition {
            transition,
            innovation_covariance: innovation,
        })
    }

    /// State displacement at `lag` after one event of `mark` at lag zero.
    pub fn state_impulse_response(
        &self,
        mark: usize,
        lag: f64,
    ) -> Result<Array1<f64>, MarkedPointProcessError> {
        if mark >= self.mark_count() {
            return Err(invalid(format!("mark index {mark} is out of bounds")));
        }
        if !lag.is_finite() || lag < 0.0 {
            return Err(invalid(
                "impulse-response lag must be finite and non-negative",
            ));
        }
        self.validate()?;
        let impulse = self.mark_impulses.column(mark).to_owned();
        if lag == 0.0 {
            return Ok(impulse);
        }
        let response = self.transition(lag)?.transition.dot(&impulse);
        if response.iter().any(|value| !value.is_finite()) {
            return Err(MarkedPointProcessError::NumericalFailure {
                context: "state impulse response",
            });
        }
        Ok(response)
    }

    /// Stationary mark-level covariance induced by the latent factor values.
    ///
    /// This is `A diag(sigma_j^2) A'`. It reduces to `A A'` under the usual
    /// unit-variance factor convention and is invariant to a simultaneous
    /// rescaling of a factor and the inverse rescaling of its loading column.
    pub fn loading_covariance(&self) -> Result<Array2<f64>, MarkedPointProcessError> {
        self.validate()?;
        let factor_variances =
            Array1::from_iter(self.factors.iter().map(|factor| factor.marginal_variance));
        let variance_weighted_loadings = &self.loadings * &factor_variances.insert_axis(Axis(0));
        let covariance = variance_weighted_loadings.dot(&self.loadings.t());
        ensure_finite_matrix(&covariance, "loading covariance")?;
        Ok(covariance)
    }
}

/// `1 - exp(-x) * (1 + x + x²/2)`, evaluated without cancellation near zero.
fn matern_three_halves_position_innovation_fraction(x: f64) -> f64 {
    if x > 1.0 {
        return 1.0 - (-x).exp() * (1.0 + x + 0.5 * x * x);
    }

    // Integrating `exp(-t) * t²/2` from zero to x gives the target function.
    // On [0, 1] its alternating power series decreases term-by-term.
    let mut term = x * x * x / 6.0;
    let mut sum = term;
    for index in 0..64 {
        let k = index as f64;
        term *= -x * (k + 3.0) / ((k + 1.0) * (k + 4.0));
        let updated = sum + term;
        if updated == sum {
            break;
        }
        sum = updated;
    }
    sum
}

/// One piecewise-constant risk-set row for one subject.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RiskInterval {
    pub entry: f64,
    pub exit: f64,
    /// Event counts by mark during `[entry, exit)`. For sufficiently fine
    /// partitions these are ordinarily zero or one, but the exact Poisson row
    /// likelihood supports general counts.
    pub counts: Array1<u32>,
    /// Known mark-specific contribution to `log lambda` on this interval.
    pub fixed_log_intensity: Array1<f64>,
}

impl RiskInterval {
    /// Exposure time `exit - entry`.
    pub fn exposure(&self) -> f64 {
        self.exit - self.entry
    }
}

fn validate_risk_interval(
    interval: &RiskInterval,
    marks: usize,
    context: &str,
) -> Result<(), MarkedPointProcessError> {
    if !interval.entry.is_finite()
        || !interval.exit.is_finite()
        || interval.exit <= interval.entry
        || !interval.exposure().is_finite()
    {
        return Err(invalid(format!(
            "{context} must have finite entry < exit and finite exposure"
        )));
    }
    if interval.counts.len() != marks || interval.fixed_log_intensity.len() != marks {
        return Err(invalid(format!(
            "{context} has {} counts and {} fixed predictors, expected {marks} each",
            interval.counts.len(),
            interval.fixed_log_intensity.len()
        )));
    }
    if interval
        .fixed_log_intensity
        .iter()
        .any(|value| !value.is_finite())
    {
        return Err(invalid(format!(
            "{context} has a non-finite fixed predictor"
        )));
    }
    Ok(())
}

/// A time-ordered sequence of risk intervals for one subject.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SubjectHistory {
    pub subject: String,
    pub intervals: Vec<RiskInterval>,
}

impl SubjectHistory {
    pub fn validate(&self, marks: usize) -> Result<(), MarkedPointProcessError> {
        if self.subject.is_empty() {
            return Err(invalid("subject identifier must be non-empty"));
        }
        if self.intervals.is_empty() {
            return Err(invalid(format!(
                "subject {:?} has no risk intervals",
                self.subject
            )));
        }
        let mut previous_exit = None;
        for (index, interval) in self.intervals.iter().enumerate() {
            validate_risk_interval(
                interval,
                marks,
                &format!("subject {:?} interval {index}", self.subject),
            )?;
            if let Some(previous) = previous_exit {
                if interval.entry < previous {
                    return Err(invalid(format!(
                        "subject {:?} intervals overlap or are out of order at row {index}",
                        self.subject
                    )));
                }
            }
            previous_exit = Some(interval.exit);
        }
        Ok(())
    }
}

/// Exact Poisson-row likelihood value and derivatives with respect to `eta`.
#[derive(Debug, Clone, PartialEq)]
pub struct PoissonIntervalEvaluation {
    pub log_likelihood: f64,
    pub gradient: Array1<f64>,
    /// Diagonal of the negative Hessian with respect to `eta`.
    pub negative_hessian: Array1<f64>,
    /// Conditional mean counts `exposure * exp(eta)`.
    pub expected_counts: Array1<f64>,
}

/// Evaluate the exact piecewise-Poisson row likelihood, including `-log(y!)`.
pub fn evaluate_poisson_interval(
    counts: ArrayView1<'_, u32>,
    log_intensity: ArrayView1<'_, f64>,
    exposure: f64,
) -> Result<PoissonIntervalEvaluation, MarkedPointProcessError> {
    if counts.len() != log_intensity.len() || counts.is_empty() {
        return Err(invalid(
            "counts and log_intensity must have equal non-zero length",
        ));
    }
    if !exposure.is_finite() || exposure <= 0.0 {
        return Err(invalid(
            "Poisson interval exposure must be finite and positive",
        ));
    }
    if log_intensity.iter().any(|value| !value.is_finite()) {
        return Err(invalid("log_intensity must contain only finite values"));
    }
    let log_exposure = exposure.ln();
    let mut log_likelihood = 0.0;
    let mut gradient = Array1::zeros(counts.len());
    let mut negative_hessian = Array1::zeros(counts.len());
    let mut expected_counts = Array1::zeros(counts.len());
    for mark in 0..counts.len() {
        let log_mean = log_exposure + log_intensity[mark];
        let mean = log_mean.exp();
        if !mean.is_finite() {
            return Err(MarkedPointProcessError::NumericalFailure {
                context: "piecewise-Poisson mean",
            });
        }
        let count = f64::from(counts[mark]);
        log_likelihood += count * log_mean - mean - ln_gamma(count + 1.0);
        gradient[mark] = count - mean;
        negative_hessian[mark] = mean;
        expected_counts[mark] = mean;
    }
    if !log_likelihood.is_finite() {
        return Err(MarkedPointProcessError::NumericalFailure {
            context: "piecewise-Poisson log likelihood",
        });
    }
    Ok(PoissonIntervalEvaluation {
        log_likelihood,
        gradient,
        negative_hessian,
        expected_counts,
    })
}

/// Explicit globalization and convergence contract for a Laplace mode solve.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LaplaceControl {
    pub max_iterations: usize,
    pub absolute_stationarity_tolerance: f64,
    pub relative_stationarity_tolerance: f64,
    pub armijo_fraction: f64,
    pub step_shrink: f64,
    pub minimum_step: f64,
}

impl LaplaceControl {
    fn validate(self) -> Result<(), MarkedPointProcessError> {
        if self.max_iterations == 0 {
            return Err(invalid("max_iterations must be positive"));
        }
        if !self.absolute_stationarity_tolerance.is_finite()
            || self.absolute_stationarity_tolerance < 0.0
            || !self.relative_stationarity_tolerance.is_finite()
            || self.relative_stationarity_tolerance < 0.0
            || self.absolute_stationarity_tolerance + self.relative_stationarity_tolerance == 0.0
        {
            return Err(invalid(
                "stationarity tolerances must be finite, non-negative, and not both zero",
            ));
        }
        if !self.armijo_fraction.is_finite()
            || self.armijo_fraction <= 0.0
            || self.armijo_fraction >= 1.0
        {
            return Err(invalid(
                "armijo_fraction must lie strictly between zero and one",
            ));
        }
        if !self.step_shrink.is_finite() || self.step_shrink <= 0.0 || self.step_shrink >= 1.0 {
            return Err(invalid(
                "step_shrink must lie strictly between zero and one",
            ));
        }
        if !self.minimum_step.is_finite() || self.minimum_step <= 0.0 || self.minimum_step >= 1.0 {
            return Err(invalid(
                "minimum_step must lie strictly between zero and one",
            ));
        }
        Ok(())
    }
}

/// Global Gaussian Laplace approximation for one subject's latent trajectory.
#[derive(Debug, Clone, PartialEq)]
pub struct LaplaceSmootherResult {
    pub mode: Vec<Array1<f64>>,
    /// Diagonal time blocks of the inverse negative Hessian at the mode.
    pub marginal_covariances: Vec<Array2<f64>>,
    pub log_joint_at_mode: f64,
    pub laplace_log_marginal_likelihood: f64,
    pub iterations: usize,
    pub stationarity: f64,
}

/// One subject's contribution to a cohort Laplace approximation.
#[derive(Debug, Clone, PartialEq)]
pub struct SubjectLaplaceResult {
    pub subject: String,
    pub approximation: LaplaceSmootherResult,
}

/// Cohort evidence with independent latent-state chains by subject.
#[derive(Debug, Clone, PartialEq)]
pub struct CohortLaplaceResult {
    pub subjects: Vec<SubjectLaplaceResult>,
    /// Sum of the normalized subject-level approximate log marginal likelihoods.
    pub laplace_log_marginal_likelihood: f64,
}

/// One model parameter estimated by the outer Laplace marginal-likelihood solve.
///
/// Positive Matérn parameters are optimized on the log scale. Loadings are
/// optimized on their signed natural scale. A factor variance and a loading in
/// the same factor column cannot be estimated together because their reciprocal
/// rescaling leaves the induced intensity distribution unchanged.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LaplaceHyperparameter {
    FactorLengthScale { factor: usize },
    FactorMarginalVariance { factor: usize },
    Loading { mark: usize, factor: usize },
}

impl LaplaceHyperparameter {
    /// Read this parameter from a validated model on its natural scale.
    pub fn value(self, model: &MarkedPointProcessModel) -> Result<f64, MarkedPointProcessError> {
        model.validate()?;
        self.validate_index(model)?;
        Ok(match self {
            Self::FactorLengthScale { factor } => model.factors[factor].length_scale,
            Self::FactorMarginalVariance { factor } => model.factors[factor].marginal_variance,
            Self::Loading { mark, factor } => model.loadings[[mark, factor]],
        })
    }

    fn validate_index(
        self,
        model: &MarkedPointProcessModel,
    ) -> Result<(), MarkedPointProcessError> {
        match self {
            Self::FactorLengthScale { factor } | Self::FactorMarginalVariance { factor }
                if factor >= model.factors.len() =>
            {
                Err(invalid(format!("factor index {factor} is out of bounds")))
            }
            Self::Loading { mark, factor } if mark >= model.mark_count() => {
                Err(invalid(format!("mark index {mark} is out of bounds")))
            }
            Self::Loading { factor, .. } if factor >= model.factors.len() => {
                Err(invalid(format!("factor index {factor} is out of bounds")))
            }
            _ => Ok(()),
        }
    }

    fn is_positive(self) -> bool {
        matches!(
            self,
            Self::FactorLengthScale { .. } | Self::FactorMarginalVariance { .. }
        )
    }

    fn assign(
        self,
        model: &mut MarkedPointProcessModel,
        value: f64,
    ) -> Result<(), MarkedPointProcessError> {
        self.validate_index(model)?;
        if !value.is_finite() || (self.is_positive() && value <= 0.0) {
            return Err(invalid(
                "hyperparameter value is outside its natural domain",
            ));
        }
        match self {
            Self::FactorLengthScale { factor } => model.factors[factor].length_scale = value,
            Self::FactorMarginalVariance { factor } => {
                model.factors[factor].marginal_variance = value;
            }
            Self::Loading { mark, factor } => model.loadings[[mark, factor]] = value,
        }
        Ok(())
    }
}

/// Natural-scale box constraint for one estimated hyperparameter.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LaplaceHyperparameterSpec {
    pub parameter: LaplaceHyperparameter,
    pub lower: f64,
    pub upper: f64,
}

/// Deterministic bounded pattern-search controls for the outer LAML solve.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HyperparameterControl {
    pub max_evaluations: usize,
    /// Initial coordinate step as a fraction of each transformed box width.
    pub initial_step_fraction: f64,
    /// Multiplier applied to every coordinate step after an unsuccessful sweep.
    pub step_shrink: f64,
    /// Convergence threshold for the largest log-scale or loading-scale step.
    pub transformed_step_tolerance: f64,
    /// Smallest evidence gain accepted as a real improvement.
    pub evidence_tolerance: f64,
}

impl HyperparameterControl {
    fn validate(self) -> Result<(), MarkedPointProcessError> {
        if self.max_evaluations == 0 {
            return Err(invalid("max_evaluations must be positive"));
        }
        if !self.initial_step_fraction.is_finite()
            || self.initial_step_fraction <= 0.0
            || self.initial_step_fraction > 1.0
        {
            return Err(invalid(
                "initial_step_fraction must lie in the interval (0, 1]",
            ));
        }
        if !self.step_shrink.is_finite() || self.step_shrink <= 0.0 || self.step_shrink >= 1.0 {
            return Err(invalid(
                "hyperparameter step_shrink must lie strictly between zero and one",
            ));
        }
        if !self.transformed_step_tolerance.is_finite() || self.transformed_step_tolerance <= 0.0 {
            return Err(invalid(
                "transformed_step_tolerance must be finite and positive",
            ));
        }
        if !self.evidence_tolerance.is_finite() || self.evidence_tolerance < 0.0 {
            return Err(invalid(
                "evidence_tolerance must be finite and non-negative",
            ));
        }
        Ok(())
    }
}

/// Joint outer LAML estimate and the latent approximation at that estimate.
#[derive(Debug, Clone, PartialEq)]
pub struct HyperparameterFitResult {
    pub model: MarkedPointProcessModel,
    pub approximation: CohortLaplaceResult,
    /// Values in the natural scales and order of the supplied specifications.
    pub parameters: Vec<f64>,
    pub evaluations: usize,
    pub maximum_transformed_step: f64,
}

#[derive(Debug, Clone)]
struct PriorTransition {
    transition: Array2<f64>,
    innovation_precision: Array2<f64>,
    innovation_logdet: f64,
    shift: Array1<f64>,
}

#[derive(Debug, Clone)]
struct PriorChain {
    initial_precision: Array2<f64>,
    initial_logdet: f64,
    transitions: Vec<PriorTransition>,
}

fn prior_chain(
    model: &MarkedPointProcessModel,
    history: &SubjectHistory,
) -> Result<PriorChain, MarkedPointProcessError> {
    let initial_covariance = model.stationary_covariance()?;
    let (initial_precision, initial_logdet) =
        inverse_and_logdet_spd(&initial_covariance, "stationary state covariance")?;
    let mut transitions = Vec::with_capacity(history.intervals.len().saturating_sub(1));
    for index in 1..history.intervals.len() {
        let elapsed = history.intervals[index].exit - history.intervals[index - 1].exit;
        if elapsed <= 0.0 {
            return Err(invalid(format!(
                "subject {:?} interval exit times must be strictly increasing for state propagation",
                history.subject
            )));
        }
        let state_transition = model.transition(elapsed)?;
        let (innovation_precision, innovation_logdet) = inverse_and_logdet_spd(
            &state_transition.innovation_covariance,
            "Matérn innovation covariance",
        )?;
        let previous_counts = history.intervals[index - 1].counts.mapv(f64::from);
        let immediate_impulse = model.mark_impulses.dot(&previous_counts);
        let shift = state_transition.transition.dot(&immediate_impulse);
        transitions.push(PriorTransition {
            transition: state_transition.transition,
            innovation_precision,
            innovation_logdet,
            shift,
        });
    }
    Ok(PriorChain {
        initial_precision,
        initial_logdet,
        transitions,
    })
}

#[derive(Debug)]
struct PosteriorEvaluation {
    log_joint: f64,
    gradient: Vec<Array1<f64>>,
    negative_hessian_diagonal: Vec<Array2<f64>>,
    negative_hessian_lower: Vec<Array2<f64>>,
}

fn evaluate_log_joint(
    model: &MarkedPointProcessModel,
    history: &SubjectHistory,
    prior: &PriorChain,
    states: &[Array1<f64>],
) -> Result<PosteriorEvaluation, MarkedPointProcessError> {
    let interval_count = history.intervals.len();
    let state_dimension = model.state_dimension();
    if states.len() != interval_count || states.iter().any(|state| state.len() != state_dimension) {
        return Err(invalid(
            "latent state trajectory has incompatible dimensions",
        ));
    }
    let observation = model.observation_matrix();
    let mut gradient = vec![Array1::zeros(state_dimension); interval_count];
    let mut diagonal = vec![Array2::zeros((state_dimension, state_dimension)); interval_count];
    let mut lower = Vec::with_capacity(interval_count.saturating_sub(1));

    let initial_quadratic = states[0].dot(&prior.initial_precision.dot(&states[0]));
    let mut log_joint =
        -0.5 * (initial_quadratic + prior.initial_logdet + state_dimension as f64 * LOG_TWO_PI);
    gradient[0] -= &prior.initial_precision.dot(&states[0]);
    diagonal[0] += &prior.initial_precision;

    for index in 1..interval_count {
        let transition = &prior.transitions[index - 1];
        let residual =
            &states[index] - &transition.transition.dot(&states[index - 1]) - &transition.shift;
        let precision_residual = transition.innovation_precision.dot(&residual);
        let quadratic = residual.dot(&precision_residual);
        log_joint -=
            0.5 * (quadratic + transition.innovation_logdet + state_dimension as f64 * LOG_TWO_PI);
        gradient[index] -= &precision_residual;
        gradient[index - 1] += &transition.transition.t().dot(&precision_residual);
        diagonal[index] += &transition.innovation_precision;
        diagonal[index - 1] += &transition
            .transition
            .t()
            .dot(&transition.innovation_precision)
            .dot(&transition.transition);
        lower.push(-transition.innovation_precision.dot(&transition.transition));
    }

    for (index, interval) in history.intervals.iter().enumerate() {
        let eta = &interval.fixed_log_intensity + &observation.dot(&states[index]);
        let likelihood =
            evaluate_poisson_interval(interval.counts.view(), eta.view(), interval.exposure())?;
        log_joint += likelihood.log_likelihood;
        gradient[index] += &observation.t().dot(&likelihood.gradient);
        let weighted_observation = &observation * &likelihood.negative_hessian.insert_axis(Axis(1));
        diagonal[index] += &observation.t().dot(&weighted_observation);
    }
    if !log_joint.is_finite() {
        return Err(MarkedPointProcessError::NumericalFailure {
            context: "marked-process log joint",
        });
    }
    Ok(PosteriorEvaluation {
        log_joint,
        gradient,
        negative_hessian_diagonal: diagonal,
        negative_hessian_lower: lower,
    })
}

/// Compute the global latent-state mode and its Gaussian Laplace approximation.
///
/// The returned evidence conditions on the supplied loadings, impulse matrix,
/// Matérn variances, and length scales.  It is the objective an outer
/// hyperparameter optimizer must maximize; optimizing loadings jointly with
/// states is bilinear and is not claimed to be log-concave.
pub fn smooth_laplace(
    model: &MarkedPointProcessModel,
    history: &SubjectHistory,
    control: LaplaceControl,
) -> Result<LaplaceSmootherResult, MarkedPointProcessError> {
    model.validate()?;
    history.validate(model.mark_count())?;
    control.validate()?;
    let prior = prior_chain(model, history)?;
    let state_dimension = model.state_dimension();
    let interval_count = history.intervals.len();
    let mut states = vec![Array1::zeros(state_dimension); interval_count];
    let mut final_evaluation = None;

    for iteration in 0..control.max_iterations {
        let evaluation = evaluate_log_joint(model, history, &prior, &states)?;
        let stationarity = block_infinity_norm(&evaluation.gradient);
        let state_scale = block_infinity_norm(&states);
        let threshold = control.absolute_stationarity_tolerance
            + control.relative_stationarity_tolerance * state_scale.max(1.0);
        if stationarity <= threshold {
            final_evaluation = Some((iteration, stationarity, evaluation));
            break;
        }
        let direction = solve_block_tridiagonal(
            &evaluation.negative_hessian_diagonal,
            &evaluation.negative_hessian_lower,
            &evaluation.gradient,
        )?;
        let directional_derivative = block_dot(&evaluation.gradient, &direction);
        if !directional_derivative.is_finite() || directional_derivative <= 0.0 {
            return Err(MarkedPointProcessError::NumericalFailure {
                context: "Newton ascent direction",
            });
        }
        let mut step = 1.0;
        let accepted = loop {
            let candidate = add_scaled_blocks(&states, &direction, step);
            // A trial point that fails the Armijo test and one at which the
            // joint is not numerically evaluable are rejected the same way:
            // the step shrinks. Any other error is the caller's.
            let sufficient_ascent = match evaluate_log_joint(model, history, &prior, &candidate) {
                Ok(candidate_evaluation) => {
                    candidate_evaluation.log_joint
                        >= evaluation.log_joint
                            + control.armijo_fraction * step * directional_derivative
                }
                Err(MarkedPointProcessError::NumericalFailure { .. }) => false,
                Err(error) => return Err(error),
            };
            if sufficient_ascent {
                states = candidate;
                break true;
            }
            step *= control.step_shrink;
            if step < control.minimum_step {
                break false;
            }
        };
        if !accepted {
            return Err(MarkedPointProcessError::LineSearchFailure);
        }
    }

    let (iterations, stationarity, evaluation) = match final_evaluation {
        Some(result) => result,
        None => {
            let evaluation = evaluate_log_joint(model, history, &prior, &states)?;
            return Err(MarkedPointProcessError::NonConvergence {
                iterations: control.max_iterations,
                stationarity: block_infinity_norm(&evaluation.gradient),
            });
        }
    };
    let factorization = factor_block_tridiagonal(
        &evaluation.negative_hessian_diagonal,
        &evaluation.negative_hessian_lower,
    )?;
    let logdet_hessian: f64 = factorization.log_determinants.iter().sum();
    let marginal_covariances = inverse_diagonal_blocks(&factorization)?;
    let latent_dimension = (interval_count * state_dimension) as f64;
    let laplace_log_marginal_likelihood =
        evaluation.log_joint + 0.5 * latent_dimension * LOG_TWO_PI - 0.5 * logdet_hessian;
    if !laplace_log_marginal_likelihood.is_finite() {
        return Err(MarkedPointProcessError::NumericalFailure {
            context: "Laplace marginal likelihood",
        });
    }
    Ok(LaplaceSmootherResult {
        mode: states,
        marginal_covariances,
        log_joint_at_mode: evaluation.log_joint,
        laplace_log_marginal_likelihood,
        iterations,
        stationarity,
    })
}

/// Evaluate the one marginal-likelihood objective across independent subjects.
///
/// Every subject receives its own stationary initial state and Markov chain;
/// loadings, Matérn dynamics, and mark semantics are shared through `model`.
/// The returned scalar is therefore the cohort objective for outer estimation
/// of those shared parameters.
pub fn smooth_laplace_cohort(
    model: &MarkedPointProcessModel,
    histories: &[SubjectHistory],
    control: LaplaceControl,
) -> Result<CohortLaplaceResult, MarkedPointProcessError> {
    if histories.is_empty() {
        return Err(invalid("cohort requires at least one subject history"));
    }
    for left in 0..histories.len() {
        if histories[left + 1..]
            .iter()
            .any(|right| right.subject == histories[left].subject)
        {
            return Err(invalid(format!(
                "subject identifier {:?} is duplicated in the cohort",
                histories[left].subject
            )));
        }
    }
    let mut subjects = Vec::with_capacity(histories.len());
    let mut laplace_log_marginal_likelihood = 0.0;
    for history in histories {
        let approximation = smooth_laplace(model, history, control)?;
        laplace_log_marginal_likelihood += approximation.laplace_log_marginal_likelihood;
        subjects.push(SubjectLaplaceResult {
            subject: history.subject.clone(),
            approximation,
        });
    }
    if !laplace_log_marginal_likelihood.is_finite() {
        return Err(MarkedPointProcessError::NumericalFailure {
            context: "cohort Laplace marginal likelihood",
        });
    }
    Ok(CohortLaplaceResult {
        subjects,
        laplace_log_marginal_likelihood,
    })
}

/// Estimate selected dynamics and loading parameters by one cohort LAML objective.
///
/// The search is deterministic bounded coordinate pattern search. Positive
/// Matérn parameters use log coordinates, so every evaluated model remains in
/// its natural domain. The objective is generally not concave in these outer
/// parameters; the returned estimate is the bounded local optimum reached from
/// `initial_model`, while every inner latent-state solve retains its unique
/// log-concave mode.
pub fn fit_laplace_hyperparameters(
    initial_model: &MarkedPointProcessModel,
    histories: &[SubjectHistory],
    specifications: &[LaplaceHyperparameterSpec],
    laplace_control: LaplaceControl,
    hyperparameter_control: HyperparameterControl,
) -> Result<HyperparameterFitResult, MarkedPointProcessError> {
    initial_model.validate()?;
    laplace_control.validate()?;
    hyperparameter_control.validate()?;
    if specifications.is_empty() {
        return Err(invalid(
            "hyperparameter search requires at least one parameter",
        ));
    }

    let mut transformed_lower = Vec::with_capacity(specifications.len());
    let mut transformed_upper = Vec::with_capacity(specifications.len());
    let mut coordinates = Vec::with_capacity(specifications.len());
    for (index, specification) in specifications.iter().copied().enumerate() {
        specification.parameter.validate_index(initial_model)?;
        if specifications[..index]
            .iter()
            .any(|previous| previous.parameter == specification.parameter)
        {
            return Err(invalid(format!(
                "hyperparameter {:?} is duplicated",
                specification.parameter
            )));
        }
        if !specification.lower.is_finite()
            || !specification.upper.is_finite()
            || specification.lower >= specification.upper
            || (specification.parameter.is_positive() && specification.lower <= 0.0)
        {
            return Err(invalid(format!(
                "hyperparameter {:?} requires finite ordered bounds in its natural domain",
                specification.parameter
            )));
        }
        let initial = specification.parameter.value(initial_model)?;
        if initial < specification.lower || initial > specification.upper {
            return Err(invalid(format!(
                "initial value {initial} for hyperparameter {:?} lies outside [{}, {}]",
                specification.parameter, specification.lower, specification.upper
            )));
        }
        let transform = |value: f64| {
            if specification.parameter.is_positive() {
                value.ln()
            } else {
                value
            }
        };
        let lower = transform(specification.lower);
        let upper = transform(specification.upper);
        let coordinate = transform(initial);
        if !lower.is_finite() || !upper.is_finite() || !coordinate.is_finite() {
            return Err(invalid(format!(
                "transformed bounds for hyperparameter {:?} are not representable",
                specification.parameter
            )));
        }
        transformed_lower.push(lower);
        transformed_upper.push(upper);
        coordinates.push(coordinate);
    }

    for factor in 0..initial_model.factors.len() {
        let variance_is_estimated = specifications.iter().any(|specification| {
            specification.parameter == LaplaceHyperparameter::FactorMarginalVariance { factor }
        });
        let loading_is_estimated = specifications.iter().any(|specification| {
            matches!(
                specification.parameter,
                LaplaceHyperparameter::Loading {
                    factor: loading_factor,
                    ..
                } if loading_factor == factor
            )
        });
        if variance_is_estimated && loading_is_estimated {
            return Err(invalid(format!(
                "factor {factor} marginal variance and loading column cannot be estimated together because their scale is not identified"
            )));
        }
    }

    let mut evaluations = 0;
    let (mut best_model, mut best_approximation) = evaluate_hyperparameter_candidate(
        initial_model,
        histories,
        specifications,
        &coordinates,
        laplace_control,
    )?;
    evaluations += 1;
    let mut best_evidence = best_approximation.laplace_log_marginal_likelihood;
    let mut steps: Vec<f64> = transformed_lower
        .iter()
        .zip(transformed_upper.iter())
        .map(|(lower, upper)| hyperparameter_control.initial_step_fraction * (upper - lower))
        .collect();

    loop {
        let maximum_step = steps.iter().copied().fold(0.0_f64, f64::max);
        if maximum_step <= hyperparameter_control.transformed_step_tolerance {
            let parameters = specifications
                .iter()
                .map(|specification| specification.parameter.value(&best_model))
                .collect::<Result<Vec<_>, _>>()?;
            return Ok(HyperparameterFitResult {
                model: best_model,
                approximation: best_approximation,
                parameters,
                evaluations,
                maximum_transformed_step: maximum_step,
            });
        }

        let mut sweep_improved = false;
        for coordinate_index in 0..coordinates.len() {
            let center = coordinates[coordinate_index];
            let mut coordinate_best = None;
            let mut coordinate_best_evidence = best_evidence;
            for direction in [-1.0, 1.0] {
                let candidate_value = (center + direction * steps[coordinate_index]).clamp(
                    transformed_lower[coordinate_index],
                    transformed_upper[coordinate_index],
                );
                if candidate_value == center {
                    continue;
                }
                if evaluations >= hyperparameter_control.max_evaluations {
                    return Err(MarkedPointProcessError::HyperparameterNonConvergence {
                        evaluations,
                        maximum_step,
                    });
                }
                let mut candidate_coordinates = coordinates.clone();
                candidate_coordinates[coordinate_index] = candidate_value;
                let (candidate_model, candidate_approximation) = evaluate_hyperparameter_candidate(
                    initial_model,
                    histories,
                    specifications,
                    &candidate_coordinates,
                    laplace_control,
                )?;
                evaluations += 1;
                let candidate_evidence = candidate_approximation.laplace_log_marginal_likelihood;
                if candidate_evidence
                    > coordinate_best_evidence + hyperparameter_control.evidence_tolerance
                {
                    coordinate_best_evidence = candidate_evidence;
                    coordinate_best = Some((
                        candidate_coordinates,
                        candidate_model,
                        candidate_approximation,
                    ));
                }
            }
            if let Some((candidate_coordinates, candidate_model, candidate_approximation)) =
                coordinate_best
            {
                coordinates = candidate_coordinates;
                best_model = candidate_model;
                best_approximation = candidate_approximation;
                best_evidence = coordinate_best_evidence;
                sweep_improved = true;
            }
        }
        if !sweep_improved {
            for step in &mut steps {
                *step *= hyperparameter_control.step_shrink;
            }
        }
    }
}

fn evaluate_hyperparameter_candidate(
    initial_model: &MarkedPointProcessModel,
    histories: &[SubjectHistory],
    specifications: &[LaplaceHyperparameterSpec],
    coordinates: &[f64],
    laplace_control: LaplaceControl,
) -> Result<(MarkedPointProcessModel, CohortLaplaceResult), MarkedPointProcessError> {
    if specifications.len() != coordinates.len() {
        return Err(invalid(
            "hyperparameter specifications and coordinates have incompatible lengths",
        ));
    }
    let mut model = initial_model.clone();
    for (specification, &coordinate) in specifications.iter().zip(coordinates.iter()) {
        let value = if specification.parameter.is_positive() {
            coordinate.exp()
        } else {
            coordinate
        };
        specification.parameter.assign(&mut model, value)?;
    }
    model.validate()?;
    let approximation = smooth_laplace_cohort(&model, histories, laplace_control)?;
    Ok((model, approximation))
}

/// Gaussian state retained after one online point-process update.
#[derive(Debug, Clone, PartialEq)]
pub struct FilteredState {
    pub time: f64,
    pub mean: Array1<f64>,
    pub covariance: Array2<f64>,
}

/// Online assumed-Gaussian Laplace updates for one subject.
///
/// Unlike [`smooth_laplace`], this does not revisit earlier states. It is the
/// recursive deployment algorithm and therefore an approximation to the global
/// non-Gaussian posterior, not an exact point-process Kalman filter. Deployment
/// code that already retains the latest state can instead call
/// [`propagate_filtered_state`] and [`update_laplace_filter`] once per row.
pub fn filter_laplace(
    model: &MarkedPointProcessModel,
    history: &SubjectHistory,
    control: LaplaceControl,
) -> Result<Vec<FilteredState>, MarkedPointProcessError> {
    model.validate()?;
    history.validate(model.mark_count())?;
    control.validate()?;
    let mut filtered: Vec<FilteredState> = Vec::with_capacity(history.intervals.len());
    let mut predicted = FilteredState {
        time: history.intervals[0].exit,
        mean: Array1::zeros(model.state_dimension()),
        covariance: model.stationary_covariance()?,
    };

    for (index, interval) in history.intervals.iter().enumerate() {
        if index > 0 {
            let elapsed = interval.exit - history.intervals[index - 1].exit;
            predicted = propagate_filtered_state_validated(
                model,
                &filtered[index - 1],
                elapsed,
                history.intervals[index - 1].counts.view(),
            )?;
        }
        let updated = update_laplace_filter_validated(model, &predicted, interval, control)?;
        filtered.push(updated);
    }
    Ok(filtered)
}

/// Exact Gaussian prediction between two recursive point-process updates.
///
/// `event_counts` are the marks observed at `filtered.time`. Their impulses are
/// applied immediately and then propagated through the exact Matérn transition.
/// The returned time is `filtered.time + elapsed`.
pub fn propagate_filtered_state(
    model: &MarkedPointProcessModel,
    filtered: &FilteredState,
    elapsed: f64,
    event_counts: ArrayView1<'_, u32>,
) -> Result<FilteredState, MarkedPointProcessError> {
    model.validate()?;
    validate_filtered_state(filtered, model.state_dimension(), false, "filtered state")?;
    if event_counts.len() != model.mark_count() {
        return Err(invalid(format!(
            "filter propagation has {} event counts, expected {}",
            event_counts.len(),
            model.mark_count()
        )));
    }
    propagate_filtered_state_validated(model, filtered, elapsed, event_counts)
}

fn propagate_filtered_state_validated(
    model: &MarkedPointProcessModel,
    filtered: &FilteredState,
    elapsed: f64,
    event_counts: ArrayView1<'_, u32>,
) -> Result<FilteredState, MarkedPointProcessError> {
    let transition = model.transition(elapsed)?;
    let time = filtered.time + elapsed;
    if !time.is_finite() {
        return Err(invalid("propagated filter time must be finite"));
    }
    let counts = event_counts.mapv(f64::from);
    let impulse = model.mark_impulses.dot(&counts);
    let mean = transition.transition.dot(&(&filtered.mean + &impulse));
    if mean.iter().any(|value| !value.is_finite()) {
        return Err(MarkedPointProcessError::NumericalFailure {
            context: "propagated filter mean",
        });
    }
    let raw_covariance = transition
        .transition
        .dot(&filtered.covariance)
        .dot(&transition.transition.t())
        + transition.innovation_covariance;
    let covariance = 0.5 * (&raw_covariance + &raw_covariance.t());
    cholesky(&covariance, "propagated filter covariance")?;
    Ok(FilteredState {
        time,
        mean,
        covariance,
    })
}

/// Apply one point-process Laplace observation update to a predicted state.
///
/// `predicted.time` must equal the interval exit (within floating-point time
/// resolution), because the piecewise-constant row is represented by its exit
/// state. The result can be retained and passed to [`propagate_filtered_state`]
/// when the next row arrives; no earlier history is required.
pub fn update_laplace_filter(
    model: &MarkedPointProcessModel,
    predicted: &FilteredState,
    interval: &RiskInterval,
    control: LaplaceControl,
) -> Result<FilteredState, MarkedPointProcessError> {
    model.validate()?;
    control.validate()?;
    validate_risk_interval(interval, model.mark_count(), "filter interval")?;
    validate_filtered_state(
        predicted,
        model.state_dimension(),
        true,
        "predicted filter state",
    )?;
    let time_scale = predicted.time.abs().max(interval.exit.abs()).max(1.0);
    if (predicted.time - interval.exit).abs() > 16.0 * f64::EPSILON * time_scale {
        return Err(invalid(format!(
            "predicted filter time {} does not match interval exit {}",
            predicted.time, interval.exit
        )));
    }
    update_laplace_filter_validated(model, predicted, interval, control)
}

fn update_laplace_filter_validated(
    model: &MarkedPointProcessModel,
    predicted: &FilteredState,
    interval: &RiskInterval,
    control: LaplaceControl,
) -> Result<FilteredState, MarkedPointProcessError> {
    let observation = model.observation_matrix();
    let (prior_precision, _) =
        inverse_and_logdet_spd(&predicted.covariance, "predicted state covariance")?;
    let mut mode = predicted.mean.clone();
    let mut posterior_covariance = None;
    for _ in 0..control.max_iterations {
        let eta = &interval.fixed_log_intensity + &observation.dot(&mode);
        let likelihood =
            evaluate_poisson_interval(interval.counts.view(), eta.view(), interval.exposure())?;
        let displacement = &mode - &predicted.mean;
        let gradient =
            observation.t().dot(&likelihood.gradient) - prior_precision.dot(&displacement);
        let weighted_observation = &observation * &likelihood.negative_hessian.insert_axis(Axis(1));
        let negative_hessian = &prior_precision + &observation.t().dot(&weighted_observation);
        let stationarity = vector_infinity_norm(&gradient);
        let threshold = control.absolute_stationarity_tolerance
            + control.relative_stationarity_tolerance * vector_infinity_norm(&mode).max(1.0);
        if stationarity <= threshold {
            posterior_covariance =
                Some(inverse_and_logdet_spd(&negative_hessian, "filtered posterior precision")?.0);
            break;
        }
        let direction = solve_spd_vector(&negative_hessian, &gradient, "filtered Newton Hessian")?;
        let directional_derivative = gradient.dot(&direction);
        if !directional_derivative.is_finite() || directional_derivative <= 0.0 {
            return Err(MarkedPointProcessError::NumericalFailure {
                context: "filtered Newton ascent direction",
            });
        }
        let current = one_state_log_posterior(
            interval,
            &observation,
            &prior_precision,
            &predicted.mean,
            &mode,
        )?;
        let mut step = 1.0;
        let accepted = loop {
            let candidate = &mode + &(step * &direction);
            let sufficient_ascent = match one_state_log_posterior(
                interval,
                &observation,
                &prior_precision,
                &predicted.mean,
                &candidate,
            ) {
                Ok(value) => {
                    value >= current + control.armijo_fraction * step * directional_derivative
                }
                Err(MarkedPointProcessError::NumericalFailure { .. }) => false,
                Err(error) => return Err(error),
            };
            if sufficient_ascent {
                mode = candidate;
                break true;
            }
            step *= control.step_shrink;
            if step < control.minimum_step {
                break false;
            }
        };
        if !accepted {
            return Err(MarkedPointProcessError::LineSearchFailure);
        }
    }
    let covariance = match posterior_covariance {
        Some(covariance) => covariance,
        None => {
            let eta = &interval.fixed_log_intensity + &observation.dot(&mode);
            let likelihood =
                evaluate_poisson_interval(interval.counts.view(), eta.view(), interval.exposure())?;
            let gradient = observation.t().dot(&likelihood.gradient)
                - prior_precision.dot(&(&mode - &predicted.mean));
            return Err(MarkedPointProcessError::NonConvergence {
                iterations: control.max_iterations,
                stationarity: vector_infinity_norm(&gradient),
            });
        }
    };
    Ok(FilteredState {
        time: interval.exit,
        mean: mode,
        covariance,
    })
}

fn validate_filtered_state(
    state: &FilteredState,
    dimension: usize,
    require_positive_definite: bool,
    context: &'static str,
) -> Result<(), MarkedPointProcessError> {
    if !state.time.is_finite()
        || state.mean.len() != dimension
        || state.covariance.dim() != (dimension, dimension)
        || state.mean.iter().any(|value| !value.is_finite())
    {
        return Err(invalid(format!(
            "{context} has incompatible dimensions or non-finite values"
        )));
    }
    if require_positive_definite {
        cholesky(&state.covariance, context)?;
    } else {
        positive_semidefinite_square_root(&state.covariance, context)?;
    }
    Ok(())
}

fn one_state_log_posterior(
    interval: &RiskInterval,
    observation: &Array2<f64>,
    prior_precision: &Array2<f64>,
    prior_mean: &Array1<f64>,
    state: &Array1<f64>,
) -> Result<f64, MarkedPointProcessError> {
    let eta = &interval.fixed_log_intensity + &observation.dot(state);
    let likelihood =
        evaluate_poisson_interval(interval.counts.view(), eta.view(), interval.exposure())?;
    let displacement = state - prior_mean;
    let value =
        likelihood.log_likelihood - 0.5 * displacement.dot(&prior_precision.dot(&displacement));
    if !value.is_finite() {
        return Err(MarkedPointProcessError::NumericalFailure {
            context: "filtered log posterior",
        });
    }
    Ok(value)
}

/// Exact marginal mean intensity under a Gaussian state approximation.
///
/// This includes the `0.5 * a' Sigma a` correction and therefore is not the
/// plug-in intensity at the posterior mean state.
pub fn gaussian_mean_intensity(
    model: &MarkedPointProcessModel,
    fixed_log_intensity: ArrayView1<'_, f64>,
    state_mean: ArrayView1<'_, f64>,
    state_covariance: &Array2<f64>,
) -> Result<Array1<f64>, MarkedPointProcessError> {
    model.validate()?;
    if fixed_log_intensity.len() != model.mark_count()
        || state_mean.len() != model.state_dimension()
        || state_covariance.dim() != (model.state_dimension(), model.state_dimension())
    {
        return Err(invalid(
            "Gaussian intensity inputs have incompatible dimensions",
        ));
    }
    if fixed_log_intensity.iter().any(|value| !value.is_finite())
        || state_mean.iter().any(|value| !value.is_finite())
    {
        return Err(invalid("Gaussian intensity inputs must be finite"));
    }
    ensure_finite_matrix(state_covariance, "state covariance")?;
    positive_semidefinite_square_root(state_covariance, "state covariance")?;
    let observation = model.observation_matrix();
    let mut intensity = Array1::zeros(model.mark_count());
    for mark in 0..model.mark_count() {
        let loading = observation.row(mark);
        let log_mean = fixed_log_intensity[mark]
            + loading.dot(&state_mean)
            + 0.5 * loading.dot(&state_covariance.dot(&loading));
        intensity[mark] = log_mean.exp();
        if !intensity[mark].is_finite() {
            return Err(MarkedPointProcessError::NumericalFailure {
                context: "Gaussian posterior mean intensity",
            });
        }
    }
    Ok(intensity)
}

/// One future piecewise-constant forecast interval.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ForecastInterval {
    pub duration: f64,
    pub fixed_log_intensity: Array1<f64>,
}

/// Explicit simulation budget for posterior-integrated cumulative incidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ForecastMonteCarlo {
    pub trajectories: usize,
    /// Non-zero seed for the deterministic xorshift stream.
    pub seed: u64,
}

/// Posterior-integrated competing-risk prediction.
#[derive(Debug, Clone, PartialEq)]
pub struct CumulativeIncidenceForecast {
    /// End time of each forecast interval, relative to the landmark.
    pub horizons: Array1<f64>,
    /// Shape `(intervals, competing_marks.len())`.
    pub cumulative_incidence: Array2<f64>,
    pub cumulative_incidence_monte_carlo_se: Array2<f64>,
    pub survival: Array1<f64>,
    pub survival_monte_carlo_se: Array1<f64>,
    pub competing_marks: Vec<usize>,
}

/// Propagate latent uncertainty and average pathwise competing-risk incidence.
///
/// For every simulated state path, interval incidence uses the exact
/// piecewise-constant formula. Averaging happens afterward, so this computes a
/// Monte Carlo approximation to `E[S(t) lambda_d(t)]`, rather than substituting
/// a posterior mean state or a mean intensity into the nonlinear survival law.
/// Survival includes every mark declared [`MarkRole::Absorbing`], even when
/// `competing_marks` requests CIF output for only a subset of those causes.
/// Recurrent and encounter marks with non-zero impulse columns are sampled from
/// their exact conditional Poisson rows and applied at interval endpoints, so
/// their effects propagate into every later forecast interval.
pub fn forecast_cumulative_incidence(
    model: &MarkedPointProcessModel,
    landmark: &FilteredState,
    future: &[ForecastInterval],
    competing_marks: &[usize],
    monte_carlo: ForecastMonteCarlo,
) -> Result<CumulativeIncidenceForecast, MarkedPointProcessError> {
    model.validate()?;
    if landmark.mean.len() != model.state_dimension()
        || landmark.covariance.dim() != (model.state_dimension(), model.state_dimension())
        || !landmark.time.is_finite()
        || landmark.mean.iter().any(|value| !value.is_finite())
    {
        return Err(invalid(
            "landmark state has incompatible dimensions or non-finite values",
        ));
    }
    if future.is_empty() {
        return Err(invalid("forecast requires at least one future interval"));
    }
    if competing_marks.is_empty() {
        return Err(invalid("forecast requires at least one competing mark"));
    }
    if monte_carlo.trajectories < 2 || monte_carlo.seed == 0 {
        return Err(invalid(
            "forecast Monte Carlo requires at least two trajectories and a non-zero seed",
        ));
    }
    let mut selected = vec![false; model.mark_count()];
    for &mark in competing_marks {
        if mark >= model.mark_count() {
            return Err(invalid(format!(
                "competing mark index {mark} is out of bounds"
            )));
        }
        if selected[mark] {
            return Err(invalid(format!(
                "competing mark index {mark} is duplicated"
            )));
        }
        if model.mark_roles[mark] != MarkRole::Absorbing {
            return Err(invalid(format!(
                "mark {:?} is not declared absorbing",
                model.mark_names[mark]
            )));
        }
        selected[mark] = true;
    }
    let absorbing_marks: Vec<usize> = model
        .mark_roles
        .iter()
        .enumerate()
        .filter_map(|(mark, role)| (*role == MarkRole::Absorbing).then_some(mark))
        .collect();
    let mut elapsed = 0.0;
    let mut horizons = Vec::with_capacity(future.len());
    for (index, interval) in future.iter().enumerate() {
        if !interval.duration.is_finite() || interval.duration <= 0.0 {
            return Err(invalid(format!(
                "forecast interval {index} duration must be finite and positive"
            )));
        }
        if interval.fixed_log_intensity.len() != model.mark_count()
            || interval
                .fixed_log_intensity
                .iter()
                .any(|value| !value.is_finite())
        {
            return Err(invalid(format!(
                "forecast interval {index} fixed predictor is incompatible or non-finite"
            )));
        }
        elapsed += interval.duration;
        if !elapsed.is_finite() {
            return Err(invalid("forecast cumulative horizon must be finite"));
        }
        horizons.push(elapsed);
    }
    let initial_cholesky =
        positive_semidefinite_square_root(&landmark.covariance, "landmark covariance")?;
    let transitions: Vec<(StateTransition, Array2<f64>)> = future
        .iter()
        .map(|interval| {
            let transition = model.transition(interval.duration)?;
            let innovation_cholesky = cholesky(
                &transition.innovation_covariance,
                "forecast innovation covariance",
            )?;
            Ok((transition, innovation_cholesky))
        })
        .collect::<Result<_, MarkedPointProcessError>>()?;
    let observation = model.observation_matrix();
    let intervals = future.len();
    let causes = competing_marks.len();
    let feedback_marks: Vec<usize> = model
        .mark_roles
        .iter()
        .enumerate()
        .filter_map(|(mark, role)| {
            (*role != MarkRole::Absorbing
                && model
                    .mark_impulses
                    .column(mark)
                    .iter()
                    .any(|value| *value != 0.0))
            .then_some(mark)
        })
        .collect();
    let mut cif_sum: Array2<f64> = Array2::zeros((intervals, causes));
    let mut cif_square_sum: Array2<f64> = Array2::zeros((intervals, causes));
    let mut survival_sum: Array1<f64> = Array1::zeros(intervals);
    let mut survival_square_sum: Array1<f64> = Array1::zeros(intervals);
    let mut normals = NormalStream::new(monte_carlo.seed);
    let mut event_rng = StdRng::seed_from_u64(monte_carlo.seed ^ 0x9e37_79b9_7f4a_7c15_u64);
    let log_max_f64 = f64::MAX.ln();
    let log_max_exact_integer = 9_007_199_254_740_992.0_f64.ln();

    for _ in 0..monte_carlo.trajectories {
        let initial_noise = normals.vector(model.state_dimension());
        let mut state = &landmark.mean + &initial_cholesky.dot(&initial_noise);
        let mut survival = 1.0;
        let mut cif: Array1<f64> = Array1::zeros(causes);
        for index in 0..intervals {
            if survival == 0.0 {
                for remaining in index..intervals {
                    for cause in 0..causes {
                        cif_sum[[remaining, cause]] += cif[cause];
                        cif_square_sum[[remaining, cause]] += cif[cause] * cif[cause];
                    }
                }
                break;
            }
            let innovation = normals.vector(model.state_dimension());
            state =
                transitions[index].0.transition.dot(&state) + transitions[index].1.dot(&innovation);
            let eta = &future[index].fixed_log_intensity + &observation.dot(&state);
            if eta.iter().any(|value| !value.is_finite()) {
                return Err(MarkedPointProcessError::NumericalFailure {
                    context: "forecast log intensity",
                });
            }
            let maximum_log_rate = absorbing_marks
                .iter()
                .map(|&mark| eta[mark])
                .fold(f64::NEG_INFINITY, f64::max);
            let scaled_total_rate = absorbing_marks
                .iter()
                .map(|&mark| (eta[mark] - maximum_log_rate).exp())
                .sum::<f64>();
            let log_integrated_rate =
                maximum_log_rate + scaled_total_rate.ln() + future[index].duration.ln();
            let integrated_rate = if log_integrated_rate >= log_max_f64 {
                f64::INFINITY
            } else {
                log_integrated_rate.exp()
            };
            let event_probability = -(-integrated_rate).exp_m1();
            if event_probability > 0.0 {
                let event_mass = survival * event_probability;
                for (cause, &mark) in competing_marks.iter().enumerate() {
                    let cause_probability =
                        (eta[mark] - maximum_log_rate).exp() / scaled_total_rate;
                    cif[cause] += event_mass * cause_probability;
                }
                survival *= (-integrated_rate).exp();
            }
            if survival > 0.0 && index + 1 < intervals && !feedback_marks.is_empty() {
                let mut feedback_counts = Array1::zeros(model.mark_count());
                for &mark in &feedback_marks {
                    let log_mean = eta[mark] + future[index].duration.ln();
                    if log_mean >= log_max_exact_integer {
                        return Err(MarkedPointProcessError::NumericalFailure {
                            context: "forecast feedback-event count rate",
                        });
                    }
                    let mean = log_mean.exp();
                    if mean > 0.0 {
                        let distribution = Poisson::new(mean).map_err(|_| {
                            MarkedPointProcessError::NumericalFailure {
                                context: "forecast feedback-event distribution",
                            }
                        })?;
                        let draw: f64 = distribution.sample(&mut event_rng);
                        if !draw.is_finite() {
                            return Err(MarkedPointProcessError::NumericalFailure {
                                context: "forecast feedback-event draw",
                            });
                        }
                        feedback_counts[mark] = draw;
                    }
                }
                state += &model.mark_impulses.dot(&feedback_counts);
                if state.iter().any(|value| !value.is_finite()) {
                    return Err(MarkedPointProcessError::NumericalFailure {
                        context: "forecast feedback-event state",
                    });
                }
            }
            for cause in 0..causes {
                cif_sum[[index, cause]] += cif[cause];
                cif_square_sum[[index, cause]] += cif[cause] * cif[cause];
            }
            survival_sum[index] += survival;
            survival_square_sum[index] += survival * survival;
        }
    }
    let trajectories = monte_carlo.trajectories as f64;
    let cumulative_incidence = &cif_sum / trajectories;
    let survival = &survival_sum / trajectories;
    let mut cumulative_incidence_monte_carlo_se = Array2::zeros((intervals, causes));
    let mut survival_monte_carlo_se = Array1::zeros(intervals);
    for index in 0..intervals {
        for cause in 0..causes {
            let mean = cumulative_incidence[[index, cause]];
            let sample_variance: f64 = (cif_square_sum[[index, cause]]
                - trajectories * mean * mean)
                / (trajectories - 1.0);
            cumulative_incidence_monte_carlo_se[[index, cause]] =
                (sample_variance.max(0.0) / trajectories).sqrt();
        }
        let mean = survival[index];
        let sample_variance: f64 =
            (survival_square_sum[index] - trajectories * mean * mean) / (trajectories - 1.0);
        survival_monte_carlo_se[index] = (sample_variance.max(0.0) / trajectories).sqrt();
    }
    Ok(CumulativeIncidenceForecast {
        horizons: Array1::from_vec(horizons),
        cumulative_incidence,
        cumulative_incidence_monte_carlo_se,
        survival,
        survival_monte_carlo_se,
        competing_marks: competing_marks.to_vec(),
    })
}

#[derive(Debug)]
struct BlockFactorization {
    schur: Vec<Array2<f64>>,
    lower_multipliers: Vec<Array2<f64>>,
    lower_original: Vec<Array2<f64>>,
    log_determinants: Vec<f64>,
}

fn factor_block_tridiagonal(
    diagonal: &[Array2<f64>],
    lower: &[Array2<f64>],
) -> Result<BlockFactorization, MarkedPointProcessError> {
    if diagonal.is_empty() || lower.len() + 1 != diagonal.len() {
        return Err(invalid("block-tridiagonal dimensions are inconsistent"));
    }
    let dimension = diagonal[0].nrows();
    if dimension == 0
        || diagonal
            .iter()
            .any(|block| block.dim() != (dimension, dimension))
        || lower
            .iter()
            .any(|block| block.dim() != (dimension, dimension))
    {
        return Err(invalid(
            "block-tridiagonal blocks must be equally sized square matrices",
        ));
    }
    let mut schur = Vec::with_capacity(diagonal.len());
    let mut lower_multipliers = Vec::with_capacity(lower.len());
    let mut log_determinants = Vec::with_capacity(diagonal.len());
    schur.push(diagonal[0].clone());
    log_determinants.push(logdet_spd(&schur[0], "block Newton Hessian")?);
    for index in 1..diagonal.len() {
        let solved_transpose = solve_spd_matrix(
            &schur[index - 1],
            &lower[index - 1].t().to_owned(),
            "block Newton Hessian",
        )?;
        let multiplier = solved_transpose.t().to_owned();
        let next_schur = &diagonal[index] - &multiplier.dot(&lower[index - 1].t());
        log_determinants.push(logdet_spd(&next_schur, "block Newton Schur complement")?);
        lower_multipliers.push(multiplier);
        schur.push(next_schur);
    }
    Ok(BlockFactorization {
        schur,
        lower_multipliers,
        lower_original: lower.to_vec(),
        log_determinants,
    })
}

fn solve_block_tridiagonal(
    diagonal: &[Array2<f64>],
    lower: &[Array2<f64>],
    right_hand_side: &[Array1<f64>],
) -> Result<Vec<Array1<f64>>, MarkedPointProcessError> {
    let factorization = factor_block_tridiagonal(diagonal, lower)?;
    if right_hand_side.len() != diagonal.len() {
        return Err(invalid("block right-hand side has incompatible length"));
    }
    let mut transformed = Vec::with_capacity(right_hand_side.len());
    transformed.push(right_hand_side[0].clone());
    for index in 1..right_hand_side.len() {
        transformed.push(
            &right_hand_side[index]
                - &factorization.lower_multipliers[index - 1].dot(&transformed[index - 1]),
        );
    }
    let mut solution = vec![Array1::zeros(right_hand_side[0].len()); right_hand_side.len()];
    let last = right_hand_side.len() - 1;
    solution[last] = solve_spd_vector(
        &factorization.schur[last],
        &transformed[last],
        "block Newton backsolve",
    )?;
    for index in (0..last).rev() {
        let rhs = &transformed[index]
            - &factorization.lower_original[index]
                .t()
                .dot(&solution[index + 1]);
        solution[index] =
            solve_spd_vector(&factorization.schur[index], &rhs, "block Newton backsolve")?;
    }
    Ok(solution)
}

fn inverse_diagonal_blocks(
    factorization: &BlockFactorization,
) -> Result<Vec<Array2<f64>>, MarkedPointProcessError> {
    let count = factorization.schur.len();
    let dimension = factorization.schur[0].nrows();
    let mut covariance = vec![Array2::zeros((dimension, dimension)); count];
    covariance[count - 1] = inverse_and_logdet_spd(
        &factorization.schur[count - 1],
        "Laplace terminal Schur complement",
    )?
    .0;
    for index in (0..count - 1).rev() {
        let schur_inverse =
            inverse_and_logdet_spd(&factorization.schur[index], "Laplace Schur complement")?.0;
        let multiplier = &factorization.lower_multipliers[index];
        covariance[index] =
            &schur_inverse + &multiplier.t().dot(&covariance[index + 1]).dot(multiplier);
    }
    Ok(covariance)
}

fn cholesky(
    matrix: &Array2<f64>,
    context: &'static str,
) -> Result<Array2<f64>, MarkedPointProcessError> {
    if matrix.nrows() == 0 || matrix.nrows() != matrix.ncols() {
        return Err(invalid(format!(
            "{context} must be a non-empty square matrix"
        )));
    }
    ensure_finite_matrix(matrix, context)?;
    ensure_symmetric_matrix(matrix, context)?;
    let dimension = matrix.nrows();
    let mut lower = Array2::zeros((dimension, dimension));
    for row in 0..dimension {
        for column in 0..=row {
            let mut value = matrix[[row, column]];
            for inner in 0..column {
                value -= lower[[row, inner]] * lower[[column, inner]];
            }
            if row == column {
                if !value.is_finite() || value <= 0.0 {
                    return Err(MarkedPointProcessError::NonPositiveDefinite {
                        context,
                        diagonal: row,
                        value,
                    });
                }
                lower[[row, column]] = value.sqrt();
            } else {
                lower[[row, column]] = value / lower[[column, column]];
            }
        }
    }
    Ok(lower)
}

/// Lower-triangular square root for a symmetric positive-semidefinite matrix.
///
/// Unlike [`cholesky`], this deliberately accepts zero pivots.  A zero pivot
/// in a positive-semidefinite matrix requires the remaining residuals in that
/// column to be zero as well; enforcing that condition prevents an indefinite
/// matrix from slipping through as a degenerate Gaussian covariance.
fn positive_semidefinite_square_root(
    matrix: &Array2<f64>,
    context: &'static str,
) -> Result<Array2<f64>, MarkedPointProcessError> {
    if matrix.nrows() == 0 || matrix.nrows() != matrix.ncols() {
        return Err(invalid(format!(
            "{context} must be a non-empty square matrix"
        )));
    }
    ensure_finite_matrix(matrix, context)?;
    ensure_symmetric_matrix(matrix, context)?;
    let dimension = matrix.nrows();
    let scale = matrix.iter().map(|value| value.abs()).fold(0.0, f64::max);
    let tolerance = 128.0 * f64::EPSILON * dimension as f64 * scale;
    let mut lower = Array2::zeros((dimension, dimension));
    for row in 0..dimension {
        for column in 0..=row {
            let mut residual = matrix[[row, column]];
            for inner in 0..column {
                residual -= lower[[row, inner]] * lower[[column, inner]];
            }
            if !residual.is_finite() {
                return Err(MarkedPointProcessError::NumericalFailure { context });
            }
            if row == column {
                if residual < -tolerance {
                    return Err(MarkedPointProcessError::NonPositiveSemidefinite {
                        context,
                        diagonal: row,
                        value: residual,
                    });
                }
                lower[[row, column]] = if residual <= tolerance {
                    0.0
                } else {
                    residual.sqrt()
                };
            } else if lower[[column, column]] > 0.0 {
                lower[[row, column]] = residual / lower[[column, column]];
            } else if residual.abs() > tolerance {
                return Err(MarkedPointProcessError::NonPositiveSemidefinite {
                    context,
                    diagonal: column,
                    value: residual,
                });
            }
        }
    }
    ensure_finite_matrix(&lower, context)?;
    Ok(lower)
}

fn solve_cholesky_vector(lower: &Array2<f64>, rhs: &Array1<f64>) -> Array1<f64> {
    let dimension = lower.nrows();
    let mut intermediate = Array1::zeros(dimension);
    for row in 0..dimension {
        let mut value = rhs[row];
        for column in 0..row {
            value -= lower[[row, column]] * intermediate[column];
        }
        intermediate[row] = value / lower[[row, row]];
    }
    let mut solution = Array1::zeros(dimension);
    for row in (0..dimension).rev() {
        let mut value = intermediate[row];
        for column in row + 1..dimension {
            value -= lower[[column, row]] * solution[column];
        }
        solution[row] = value / lower[[row, row]];
    }
    solution
}

fn solve_spd_vector(
    matrix: &Array2<f64>,
    rhs: &Array1<f64>,
    context: &'static str,
) -> Result<Array1<f64>, MarkedPointProcessError> {
    if matrix.nrows() != rhs.len() {
        return Err(invalid(format!(
            "{context} right-hand side has incompatible length"
        )));
    }
    Ok(solve_cholesky_vector(&cholesky(matrix, context)?, rhs))
}

fn solve_spd_matrix(
    matrix: &Array2<f64>,
    rhs: &Array2<f64>,
    context: &'static str,
) -> Result<Array2<f64>, MarkedPointProcessError> {
    if matrix.nrows() != rhs.nrows() {
        return Err(invalid(format!(
            "{context} right-hand side has incompatible rows"
        )));
    }
    let lower = cholesky(matrix, context)?;
    let mut solution = Array2::zeros(rhs.dim());
    for column in 0..rhs.ncols() {
        solution.column_mut(column).assign(&solve_cholesky_vector(
            &lower,
            &rhs.column(column).to_owned(),
        ));
    }
    Ok(solution)
}

fn inverse_and_logdet_spd(
    matrix: &Array2<f64>,
    context: &'static str,
) -> Result<(Array2<f64>, f64), MarkedPointProcessError> {
    let lower = cholesky(matrix, context)?;
    let dimension = matrix.nrows();
    let mut inverse = Array2::zeros((dimension, dimension));
    for column in 0..dimension {
        let mut unit = Array1::zeros(dimension);
        unit[column] = 1.0;
        inverse
            .column_mut(column)
            .assign(&solve_cholesky_vector(&lower, &unit));
    }
    let logdet = 2.0 * lower.diag().iter().map(|value| value.ln()).sum::<f64>();
    Ok((inverse, logdet))
}

fn logdet_spd(matrix: &Array2<f64>, context: &'static str) -> Result<f64, MarkedPointProcessError> {
    let lower = cholesky(matrix, context)?;
    Ok(2.0 * lower.diag().iter().map(|value| value.ln()).sum::<f64>())
}

fn block_infinity_norm(blocks: &[Array1<f64>]) -> f64 {
    blocks
        .iter()
        .map(|block| vector_infinity_norm(block))
        .fold(0.0, f64::max)
}

fn vector_infinity_norm(vector: &Array1<f64>) -> f64 {
    vector.iter().map(|value| value.abs()).fold(0.0, f64::max)
}

fn block_dot(left: &[Array1<f64>], right: &[Array1<f64>]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(left, right)| left.dot(right))
        .sum()
}

fn add_scaled_blocks(
    base: &[Array1<f64>],
    direction: &[Array1<f64>],
    scale: f64,
) -> Vec<Array1<f64>> {
    base.iter()
        .zip(direction)
        .map(|(base, direction)| base + &(scale * direction))
        .collect()
}

fn ensure_finite_matrix(
    matrix: &Array2<f64>,
    name: &'static str,
) -> Result<(), MarkedPointProcessError> {
    if matrix.iter().any(|value| !value.is_finite()) {
        return Err(invalid(format!("{name} must contain only finite values")));
    }
    Ok(())
}

fn ensure_symmetric_matrix(
    matrix: &Array2<f64>,
    name: &'static str,
) -> Result<(), MarkedPointProcessError> {
    let dimension = matrix.nrows();
    let scale = matrix.iter().map(|value| value.abs()).fold(0.0, f64::max);
    let tolerance = 64.0 * f64::EPSILON * dimension as f64 * scale;
    for row in 0..dimension {
        for column in 0..row {
            if (matrix[[row, column]] - matrix[[column, row]]).abs() > tolerance {
                return Err(invalid(format!("{name} must be symmetric")));
            }
        }
    }
    Ok(())
}

fn invalid(reason: impl Into<String>) -> MarkedPointProcessError {
    MarkedPointProcessError::InvalidInput {
        reason: reason.into(),
    }
}

#[derive(Debug)]
struct NormalStream {
    state: u64,
    spare: Option<f64>,
}

impl NormalStream {
    fn new(seed: u64) -> Self {
        Self {
            state: seed,
            spare: None,
        }
    }

    fn uniform_open(&mut self) -> f64 {
        let mut value = self.state;
        value ^= value >> 12;
        value ^= value << 25;
        value ^= value >> 27;
        self.state = value;
        let bits = value.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 11;
        (bits as f64 + 0.5) * (1.0 / 9_007_199_254_740_992.0)
    }

    fn standard_normal(&mut self) -> f64 {
        if let Some(spare) = self.spare.take() {
            return spare;
        }
        let radius = (-2.0 * self.uniform_open().ln()).sqrt();
        let angle = std::f64::consts::TAU * self.uniform_open();
        self.spare = Some(radius * angle.sin());
        radius * angle.cos()
    }

    fn vector(&mut self, dimension: usize) -> Array1<f64> {
        Array1::from_shape_simple_fn(dimension, || self.standard_normal())
    }
}
