//! Event histories: marked counting processes with smooth covariate and time
//! effects, a per-subject latent state made of unit-variance
//! Ornstein–Uhlenbeck atoms with evidence-selected loadings and rates that
//! enters as the individual deviation from a population log-intensity (the
//! loadings' Gaussian mixing is cancelled in the intensity, so the smooth
//! surfaces are population-average rates), marginalisation of the latent
//! chain by adaptive Gauss-Hermite–Lagrange filtering on grids centred at
//! each node's posterior mean, the exact gradient of that computed
//! likelihood from forward-mode duals through the same filter, and the
//! Hessian and its directional derivatives from Louis' identity accumulated
//! in coefficient space by one forward sweep. Forecasts and the predictive
//! PIT are expectations under the filtered state, with every probability a
//! chronological integral of a killed process.
//!
//! What is exact and what is certified: the latent integral at the nodes is
//! resolved by Gauss-Hermite quadrature, the latent path is sampled at the
//! nodes of a mesh, and the fit refines both — the order and the mesh —
//! until the fitted coefficients are stationary under refinement to a
//! stated fraction of their posterior standard deviation. That certificate,
//! not the word "exact", is what the fit carries.
//!
//! Survival with a slow frailty, competing risks (terminal marks), once-only
//! marks, recurrent events and history-conditioned prediction are all one
//! family here: they differ only in the rows and the mark kinds. The latent
//! block's ridges are ordinary REML smoothing parameters, so an atom the
//! data do not support is switched off by its ridge; the count of atoms a
//! fit carries is the maximum offered, and which ones the data use is read
//! from the fitted loadings and ridges.

mod chain;
mod cohort;
mod family;
mod forecast;
mod formula;
mod marginal;
mod scalar;

pub use cohort::{
    CohortNodes, CovariateSegment, Event, EventHistoryCohort, EventHistoryError, MarkKind,
    SubjectHistory, SubjectNodes, design_rows, expand_nodes, quadrature_order_for_degree,
};
pub use family::{
    EventHistoryFamily, EventHistoryFit, EventHistorySpec, JointEvaluation,
    QuadratureCertificate, RefinementCheck, fit_event_history, fit_event_history_formula,
    latent_block_spec, mark_block_spec, seeded_one, seeded_two,
};
pub use formula::{TIME_COLUMN, covariate_spec_from_formula, node_dataset};
pub use marginal::transition_score_polynomials;
pub use forecast::{
    EventPit, Forecast, ForecastRequest, FutureSegment, PopulationForecastRequest, forecast,
    kolmogorov_smirnov_uniform, population_forecast, predictive_pit, training_eta,
};

#[cfg(test)]
mod tests;
