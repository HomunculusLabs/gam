//! Marked event histories with shared Gaussian OU latent factors.
//!
//! Reference centring is differentiated through the population evolution.
//! Its time discretisation is checked at fixed coefficients, and the final
//! coefficients and centring snapshot are exported together. Forecasts have
//! explicit reference-horizon and entry-conditioning contracts.
//!
//! Latent integration uses product Gauss-Hermite grids. It is an approximation
//! with limited rank capacity; streaming backward rows avoids a quadratic
//! state-memory allocation. Filtered residuals propose factor rates, while
//! likelihood derivatives check loading curvature. Sampled directional
//! profiles and Laplace rank comparisons remain approximations.

mod chain;
mod cohort;
mod covariance;
mod family;
mod forecast;
mod formula;
pub mod joint;
mod marginal;
mod preserve;
mod scalar;
mod static_state;

pub use cohort::{
    CohortNodes, CovariateSegment, Event, EventHistoryCohort, EventHistoryError, MarkKind,
    SubjectHistory, SubjectNodes,
};
pub use covariance::{
    DirectionEvidence, DirectionProfile, RidgeProfile, effective_rank, temporal_covariance,
};
pub use family::{
    EventHistoryFamily, EventHistoryFit, EventHistorySpec, QuadratureCertificate, RankStep,
    RefinementCheck, RiskSetCentring, fit_event_history_formulas,
};
pub use forecast::{
    Forecast, ForecastRequest, FutureSegment, HistoryForecastRequest, PopulationForecastRequest,
    SmoothedLatentState, SpellPit, forecast, forecast_history, latent_state, pit_uniform_distance,
    population_forecast, predictive_pit, baseline_log_rates,
};
pub use preserve::{ReferenceGrid, ReferenceStrata};

#[cfg(test)]
mod tests;
