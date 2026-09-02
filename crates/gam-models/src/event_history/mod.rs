//! Event histories: marked counting processes with smooth covariate and time
//! effects per mark, a per-subject latent state of unit-variance
//! Ornstein–Uhlenbeck atoms that enters as the individual's deviation from
//! the population log-intensity, per-mark risk sets, and a latent
//! covariance whose rank is grown from zero by the evidence.
//!
//! The latent path is integrated out by a Laplace approximation on its
//! block-tridiagonal Markov structure ([`laplace`]), polynomial in the
//! number of atoms; the evidence gradient is exact, and the Hessian and its
//! directional derivatives are the tangent channels of that gradient,
//! contracted with the design rows node by node so nothing of size
//! `(nodes × marks)²` exists. The reported latent object is the covariance
//! `C(Δ) = A diag(e^{−r|Δ|}) Aᵀ` ([`covariance`]); the loadings are its
//! factor coordinates. Each new covariance direction is proposed by the
//! covariance score and accepted only if the outer LAML criterion improves.
//! Forecasts are per-mark first-occurrence probabilities under a sequential
//! Gaussian filter continued from the smoothed state at exit, and the
//! smoothed latent state itself is exposed with its posterior covariance.
//!
//! Survival with a static frailty, competing risks, first diagnoses of many
//! diseases, recurrent events and history-conditioned prediction are one
//! family here: they differ only in the rows and the mark kinds.

mod cohort;
mod covariance;
mod family;
mod forecast;
mod formula;
mod laplace;
mod scalar;

pub use cohort::{
    CohortNodes, CovariateSegment, Event, EventHistoryCohort, EventHistoryError, MarkKind,
    SubjectHistory, SubjectNodes, design_rows, expand_nodes, quadrature_order_for_degree,
};
pub use covariance::{disease_covariance, eigenmodes, temporal_covariance};
pub use laplace::Smoother;
pub use family::{
    EventHistoryFamily, EventHistoryFit, EventHistorySpec, JointEvaluation, RankStep,
    fit_event_history, fit_event_history_formula, latent_block_spec, mark_block_spec, seeded_one,
    seeded_two,
};
pub use forecast::{
    EventPit, Forecast, ForecastRequest, FutureSegment, LatentPath, PopulationForecastRequest,
    forecast, kolmogorov_smirnov_uniform, latent_exposure, latent_state, population_forecast,
    predictive_pit, training_eta,
};
pub use formula::{TIME_COLUMN, covariate_spec_from_formula, node_dataset};

#[cfg(test)]
mod tests;
