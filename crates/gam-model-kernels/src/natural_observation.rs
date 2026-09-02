//! Common derivative carrier for scalar likelihoods in a natural predictor.
//!
//! Separable likelihoods differ in their row data and stable evaluation, but
//! the blockwise Laplace engine consumes the same five quantities.  Keeping
//! that contract here lets indexed families share one evaluation and outer-
//! derivative implementation without erasing the concrete row kernel.

/// Exact value and curvature tower for one unweighted scalar observation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NaturalDiagonalObservation {
    pub log_likelihood: f64,
    pub score: f64,
    pub negative_hessian: f64,
    pub negative_hessian_derivative: f64,
    pub negative_hessian_second_derivative: f64,
}

