//! Model-level testing utilities.
//!
//! What lives here: harnesses that genuinely need the model layer — the
//! reference-tool bridge (`reference`), the CLI harness (`cli_harness`), and the
//! calibration fixtures (`calibration`).
//!
//! What deliberately does NOT live here: fixtures and assertions that own no
//! model-layer type. Those live in the leaf crate that owns the types they
//! exercise and are re-exported below, so a crate that only needs (say) a
//! finite-difference cross-check depends on that leaf rather than on this crate
//! — which pulls `gam-models`, and through it the whole solver stack, into every
//! dependent's test build. That back-edge is why `cargo test -p gam-solve --lib`
//! used to compile the entire model layer before running a single unit test.

pub mod calibration;
pub mod cli_harness;
pub mod reference;

// `no_densify_design` (and the operator-backed fixture behind it) is a
// linear-algebra fixture; it lives in `gam-linalg` alongside the operator traits
// it exercises and is re-exported here so this crate's consumers keep their
// familiar path. Single source of truth — the previous duplicate copy drifted
// out of the crate that owns the types.
pub use gam_linalg::test_support::no_densify_design;

// The stderr backend for production's `log::info!` diagnostics is `log` in,
// stderr out — it owns no model-layer type, so by the rule above it lives in
// `gam-runtime` (which already owns `span`/`process_monitor`/`loop_progress` and
// already depends on `log`) and is re-exported here. Without a backend installed
// the `log` facade DROPS every record, which is why the BMS intercept counters,
// the GL-ladder histogram and the certificate-bound discriminator have all been
// emitting into nothing in every test binary.
pub use gam_runtime::test_support::install_diagnostic_logger;

// Finite-difference derivative checking is `ndarray` in, `ndarray` out: it owns
// no model-layer type, so it lives in `gam-linalg`.
pub use gam_linalg::test_support::fd_checker;
pub use gam_linalg::test_support::fd_checker::{
    assert_matrix_derivativefd, assert_matrix_derivativefd_rel,
};

// `ParameterBlockSpec` fixtures live in `gam-problem`, the crate that owns the
// spec type.
pub use gam_problem::test_support::{
    BinomialLocationScaleBaseFixture, binomial_location_scale_base_fixture, spec_from_dense,
    spec_from_dense_with_priority,
};

// The central-difference macro is `ndarray`-only and expands `approx` at the
// call site, so it lives in `gam-linalg` with the rest of the FD harness.
pub use gam_linalg::assert_central_difference_array;
