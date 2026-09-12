//! Plumbing test: under a non-trivial frailty scale s_f = 1/√(1+σ²) (σ > 0,
//! so s_f < 1) the flat identifiability audit keeps the survival marginal-slope
//! slope block apart from the marginal block.
//!
//! At β = 0 the slope block's single-output effective design is
//! `diag(s_f · z) · G`. The audit on that design and a structurally distinct
//! marginal design must report no alias and drop no column.
//!
//! # σ-as-parameter verdict
//!
//! σ is always **fixed** in the survival marginal-slope family:
//! `validate_spec` rejects `FrailtySpec::GaussianShift { sigma_fixed: None }`
//! with an explicit error. There is no code path where σ is part of β.
//! Therefore `∂s_f/∂σ` never appears in the β-Jacobian and no σ-column is
//! needed. The `probit_frailty_scale` field on `FamilyLinearizationState`
//! carries the construction-time value through to the Jacobian callback at
//! evaluation time, ensuring outer-loop σ updates (which rebuild the family
//! with a new fixed σ) are reflected without requiring spec reconstruction.

// The direct `SlopeBlockJacobian` construction tests moved in-crate
// (crates/gam-models/src/survival/marginal_slope/tests.rs::
// slope_jacobian_reads_probit_scale_from_state_at_beta_zero) when the
// constructor went crate-internal (#2352). The public-surface audit guard
// below remains here.

use gam::custom_family::ParameterBlockSpec;
use gam::identifiability::audit::audit_identifiability;
use gam::linalg::matrix::{DenseDesignMatrix, DesignMatrix};
use ndarray::{Array1, Array2};

const N: usize = 40;
const P: usize = 5;

/// Build a random-ish design matrix deterministically.
fn make_design(seed: u64) -> Array2<f64> {
    let mut out = Array2::<f64>::zeros((N, P));
    let mut state = seed;
    for i in 0..N {
        for j in 0..P {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            out[[i, j]] = ((state >> 33) as f64) / (u32::MAX as f64) - 0.5;
        }
    }
    out
}

/// Build z scores deterministically (positive, mean ~1).
fn make_z(seed: u64) -> Vec<f64> {
    let mut state = seed ^ 0xdeadbeef;
    (0..N)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            0.3 + ((state >> 33) as f64) / (u32::MAX as f64) * 1.4
        })
        .collect()
}

/// Build a slope spec whose design is the β=0 effective slope design
/// `diag(s_f·z)·design`.
fn make_slope_spec(design: &Array2<f64>, z: &[f64], s_f: f64) -> ParameterBlockSpec {
    let mut scaled = design.clone();
    for (mut row, &zi) in scaled.rows_mut().into_iter().zip(z) {
        row *= s_f * zi;
    }
    ParameterBlockSpec {
        name: "slope_surface".to_string(),
        design: DesignMatrix::Dense(DenseDesignMatrix::from(scaled)),
        offset: Array1::<f64>::zeros(N),
        penalties: Vec::new(),
        nullspace_dims: Vec::new(),
        initial_log_lambdas: Array1::<f64>::zeros(0),
        initial_beta: None,
        gauge_priority: 120,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    }
}

/// Build a marginal block spec with a different design (no frailty involvement).
fn make_marginal_spec(design: &Array2<f64>) -> ParameterBlockSpec {
    ParameterBlockSpec {
        name: "marginal_surface".to_string(),
        design: DesignMatrix::Dense(DenseDesignMatrix::from(design.clone())),
        offset: Array1::<f64>::zeros(N),
        penalties: Vec::new(),
        nullspace_dims: Vec::new(),
        initial_log_lambdas: Array1::<f64>::zeros(0),
        initial_beta: None,
        gauge_priority: 150,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    }
}

/// The identifiability audit on a marginal + slope pair with non-trivial
/// s_f must pass (no fatal alias) because the slope effective design is
/// s_f·z·G (different row-weights than the marginal's unscaled design M).
#[test]
fn audit_marginal_slope_not_aliased_under_nontrivial_sf() {
    // Use different designs for marginal and slope so they are structurally distinct.
    let design_log = make_design(100);
    let design_marg = make_design(200); // deliberately different
    let z = make_z(100);
    let s_f = 0.6_f64;

    let slope_spec = make_slope_spec(&design_log, &z, s_f);
    let marginal_spec = make_marginal_spec(&design_marg);

    let specs = [marginal_spec, slope_spec];
    let audit = audit_identifiability(&specs).expect("audit must succeed");

    assert!(
        !audit.fatal,
        "marginal + slope with s_f={s_f} must not be fatal; summary: {}",
        audit.summary,
    );
    assert!(
        audit.dropped_columns.is_empty(),
        "no columns should be dropped for structurally distinct blocks; got: {:?}",
        audit.dropped_columns,
    );
}
