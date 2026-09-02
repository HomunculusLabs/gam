//! Event histories: marked counting processes with smooth covariate and time
//! effects, a per-subject latent state made of unit-variance
//! Ornstein–Uhlenbeck atoms with evidence-selected loadings and rates that
//! enters as the individual deviation from a population log-intensity (the
//! loadings' Gaussian mixing is cancelled in the intensity, so the smooth
//! surfaces are population-average rates), exact
//! marginalisation of the latent chain by adaptive Gauss-Hermite–Lagrange
//! filtering on grids centred at each node's posterior mean, the exact
//! gradient of that computed likelihood from forward-mode
//! duals through the same filter, and the Hessian and its directional
//! derivatives from Louis' identity. Forecasts and the predictive PIT are
//! exact expectations under the filtered state.
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
mod formula;
mod marginal;
mod scalar;

pub use cohort::{
    CohortNodes, CovariateSegment, Event, EventHistoryCohort, EventHistoryError, SubjectHistory,
    SubjectNodes, expand_nodes, quadrature_order_for_degree,
};
pub use family::{
    EventHistoryFamily, EventHistoryFit, EventHistorySpec, JointEvaluation,
    QuadratureCertificate, fit_event_history, fit_event_history_formula, latent_block_spec,
    mark_block_spec, seeded_one, seeded_two,
};
pub use formula::{TIME_COLUMN, covariate_spec_from_formula, node_dataset};
pub use marginal::transition_score_polynomials;
pub use forecast::{
    Forecast, ForecastRequest, PopulationForecastRequest, forecast, kolmogorov_smirnov_uniform,
    population_forecast, predictive_pit, training_eta,
};

#[cfg(test)]
mod tests;
