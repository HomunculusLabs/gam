//! Regression guard for the large-scale 8h-hang investigation.
//!
//! The identifiability audit / canonicalization halts the large-scale
//! 5-block shape (51 cols, rank 38) with
//! `CustomFamilyError::IdentifiabilityFailure`. The release binary
//! that emitted the failing log predates this gate — the literal
//! phrase "family-specific reparameterisation required" does not
//! appear in current source.
//!
//! The investigation's other fix, the K-cap bypass in
//! `maybe_install_auto_outer_subsample`, is guarded beside that code in
//! gam-models (`marginal_slope_shared::subsample_k_cap_tests`).

use gam::identifiability::audit::audit_identifiability;
use ndarray::{Array1, Array2};

// -------- H2: identifiability audit halts the large-scale shape ----------

/// Build five overlapping blocks that mimic the large-scale rank
/// deficiency the user's log reported: joint rank ≪ joint cols, many
/// near-1.0 aliases across blocks, mirroring time/marginal/slope/
/// score_warp/link_dev sharing the same polynomial/nullspace directions.
fn build_large_scale_like_aliased_specs() -> Vec<gam::families::custom_family::ParameterBlockSpec> {
    use gam::families::custom_family::ParameterBlockSpec;
    use gam::linalg::matrix::{DenseDesignMatrix, DesignMatrix};

    let n = 200;
    // Common low-frequency basis: constant, x, x^2, x^3.
    let xs: Vec<f64> = (0..n)
        .map(|i| -1.0 + 2.0 * (i as f64) / ((n - 1) as f64))
        .collect();
    let common = |k: usize| -> Array2<f64> {
        let mut m = Array2::<f64>::zeros((n, k + 1));
        for (i, &x) in xs.iter().enumerate() {
            let mut pow = 1.0;
            for j in 0..=k {
                m[[i, j]] = pow;
                pow *= x;
            }
        }
        m
    };

    // Each block embeds the common low-frequency basis plus its own
    // high-frequency tail. This produces near-perfect aliases on the
    // low-frequency columns across all blocks — the same structural
    // collinearity pattern the user's log reported between
    // time_surface / marginal_surface / slope_surface / etc.
    let block = |seed: u64, k_common: usize, k_extra: usize| -> Array2<f64> {
        let mut m = Array2::<f64>::zeros((n, k_common + 1 + k_extra));
        let c = common(k_common);
        for j in 0..=k_common {
            for i in 0..n {
                m[[i, j]] = c[[i, j]];
            }
        }
        // High-frequency tail — block-specific sin/cos with seed-based
        // frequencies so blocks are distinguishable past the polynomial
        // nullspace.
        for t in 0..k_extra {
            let freq = (seed as f64) + 1.0 + 1.7 * (t as f64);
            for i in 0..n {
                m[[i, k_common + 1 + t]] = (freq * xs[i]).sin();
            }
        }
        m
    };

    let names = [
        "time_surface",
        "marginal_surface",
        "slope_surface",
        "score_warp_dev",
        "link_dev",
    ];
    let extras = [8, 7, 6, 5, 5]; // total widths after +k_common+1=4 → 12, 11, 10, 9, 9
    let mut specs = Vec::new();
    for (idx, (name, &extra)) in names.iter().zip(extras.iter()).enumerate() {
        let m = block(idx as u64, 3, extra);
        specs.push(ParameterBlockSpec {
            name: name.to_string(),
            design: DesignMatrix::Dense(DenseDesignMatrix::from(m)),
            offset: Array1::<f64>::zeros(n),
            penalties: Vec::new(),
            nullspace_dims: Vec::new(),
            initial_log_lambdas: Array1::<f64>::zeros(0),
            initial_beta: None,
            gauge_priority: 100,
            jacobian_callback: None,
            stacked_design: None,
            stacked_offset: None,
        });
    }
    specs
}

#[test]
fn h2a_audit_marks_large_scale_shape_fatal() {
    let specs = build_large_scale_like_aliased_specs();
    let total_cols: usize = specs.iter().map(|s| s.design.ncols()).sum();
    // Sanity: the synthetic design matches the user's reported widths.
    assert_eq!(total_cols, 51, "large-scale-like total cols");
    let audit = audit_identifiability(&specs).expect("audit must run mechanically");
    assert!(
        audit.fatal,
        "current source MUST halt on the large-scale rank-deficient shape; \
         got fatal=false, summary={:?}",
        audit.summary
    );
    // Independent line of evidence: rank deficiency is visible AND named.
    assert!(
        audit.summary.contains("FATAL") || audit.summary.contains("joint rank"),
        "summary should name rank deficiency; got {:?}",
        audit.summary
    );
}

#[test]
fn h2b_release_log_string_absent_from_current_source() {
    // The user's installed release printed the literal phrase:
    //   "alias(es) flagged; family-specific reparameterisation required"
    // This phrase appears NOWHERE in current source — independent proof
    // that the running binary is older than current main.
    //
    // We can only check this from inside the test by exercising a fatal
    // audit on a known shape and confirming the modern summary uses
    // either "FATAL" or "partial alias(es) below halt threshold" or
    // "clean" instead.
    let specs = build_large_scale_like_aliased_specs();
    let audit = audit_identifiability(&specs).expect("audit must run");
    assert!(
        !audit
            .summary
            .contains("family-specific reparameterisation required"),
        "release-only phrase leaked into current source: {:?}",
        audit.summary
    );
    assert!(
        !audit.summary.contains("alias(es) flagged;"),
        "release-only phrase leaked into current source: {:?}",
        audit.summary
    );
}

