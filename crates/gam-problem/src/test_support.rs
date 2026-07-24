//! Parameter-block test fixtures shared across the workspace.
//!
//! These fixtures build [`ParameterBlockSpec`] values, so they live in the crate
//! that owns that type rather than in a downstream test-support crate. Following
//! the workspace convention for `test_support` modules, this is a plain
//! always-compiled `pub mod`: feature gates and `#[cfg(test)]` module gates are
//! banned here, and a `cfg(test)` module would be invisible to downstream
//! crates' test builds anyway. The contents are `pub`, so they are reachable
//! (no dead-code lint) yet only ever called from `#[cfg(test)]` code.

use crate::PenaltyMatrix;
use crate::block_spec::ParameterBlockSpec;
use gam_linalg::matrix::{DenseDesignMatrix, DesignMatrix};
use ndarray::{Array1, Array2, array};

/// A minimal two-block binomial location–scale problem: `n = 7` Bernoulli
/// responses with an intercept-only threshold block and an intercept-only
/// log-σ block, each carrying a unit ridge penalty.
pub struct BinomialLocationScaleBaseFixture {
    pub n: usize,
    pub y: Array1<f64>,
    pub weights: Array1<f64>,
    pub threshold_design: DesignMatrix,
    pub log_sigma_design: DesignMatrix,
    pub threshold_spec: ParameterBlockSpec,
    pub log_sigma_spec: ParameterBlockSpec,
}

pub fn binomial_location_scale_base_fixture() -> BinomialLocationScaleBaseFixture {
    let n = 7usize;
    let y = Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]);
    let weights = Array1::from_vec(vec![1.0; n]);
    let threshold_design =
        DesignMatrix::Dense(DenseDesignMatrix::from(Array2::from_elem((n, 1), 1.0)));
    let log_sigma_design =
        DesignMatrix::Dense(DenseDesignMatrix::from(Array2::from_elem((n, 1), 1.0)));
    let threshold_spec = ParameterBlockSpec {
        name: "threshold".to_string(),
        design: threshold_design.clone(),
        offset: Array1::zeros(n),
        penalties: vec![PenaltyMatrix::Dense(Array2::eye(1))],
        nullspace_dims: vec![],
        initial_log_lambdas: array![0.0],
        initial_beta: Some(array![0.2]),
        gauge_priority: 100,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    };
    let log_sigma_spec = ParameterBlockSpec {
        name: "log_sigma".to_string(),
        design: log_sigma_design.clone(),
        offset: Array1::zeros(n),
        penalties: vec![PenaltyMatrix::Dense(Array2::eye(1))],
        nullspace_dims: vec![],
        initial_log_lambdas: array![-0.2],
        initial_beta: Some(array![-0.1]),
        gauge_priority: 100,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    };
    BinomialLocationScaleBaseFixture {
        n,
        y,
        weights,
        threshold_design,
        log_sigma_design,
        threshold_spec,
        log_sigma_spec,
    }
}

/// An unpenalized block wrapping a dense design under the default gauge
/// priority.
pub fn spec_from_dense(name: &str, design: Array2<f64>) -> ParameterBlockSpec {
    let n = design.nrows();
    ParameterBlockSpec {
        name: name.to_string(),
        design: DesignMatrix::Dense(DenseDesignMatrix::from(design)),
        offset: Array1::<f64>::zeros(n),
        penalties: Vec::new(),
        nullspace_dims: Vec::new(),
        initial_log_lambdas: Array1::<f64>::zeros(0),
        initial_beta: None,
        gauge_priority: 100,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    }
}

/// [`spec_from_dense`] with an explicit gauge priority, for fixtures that pin
/// which block wins a shared-affine gauge contest.
pub fn spec_from_dense_with_priority(
    name: &str,
    design: Array2<f64>,
    priority: u8,
) -> ParameterBlockSpec {
    let mut s = spec_from_dense(name, design);
    s.gauge_priority = priority;
    s
}
