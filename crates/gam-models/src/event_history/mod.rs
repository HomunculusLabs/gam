//! Event histories: marked counting processes with smooth covariate and time
//! effects, a per-subject latent state made of unit-variance
//! Ornstein–Uhlenbeck atoms with evidence-selected loadings and rates, exact
//! marginalisation of the latent chain by adaptive Gauss-Hermite–Lagrange
//! filtering, and every derivative the LAML outer solve needs from the Fisher
//! and Louis identities. Forecasts and the predictive PIT are exact
//! expectations under the filtered state.
//!
//! Survival with a static frailty, competing risks, recurrent events and
//! history-conditioned prediction are all one family here: they differ only
//! in the rows. The latent block's ridges are ordinary REML smoothing
//! parameters, so the number of atoms the data support is read from the fit,
//! not chosen.

mod chain;
mod cohort;
mod family;
mod forecast;
mod marginal;
mod scalar;

pub use cohort::{
    CohortNodes, CovariateSegment, Event, EventHistoryCohort, EventHistoryError, SubjectHistory,
    SubjectNodes, expand_nodes, quadrature_order_for_degree,
};
pub use family::{
    EventHistoryFamily, EventHistoryFit, EventHistorySpec, JointEvaluation,
    QuadratureCertificate, fit_event_history, latent_block_spec, mark_block_spec, seeded_one,
    seeded_two,
};
pub use marginal::transition_score_polynomials;
pub use forecast::{
    Forecast, ForecastRequest, forecast, kolmogorov_smirnov_uniform, predictive_pit, training_eta,
};

#[cfg(test)]
mod tests;
