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

/// Assert that a central difference of an array-producing function matches the analytical derivative.
#[macro_export]
macro_rules! assert_central_difference_array {
    ($x:expr, $h:expr, |$var:ident| $eval:expr, $analytical:expr, $tol:expr) => {
        let f_plus = {
            let $var = $x + $h;
            $eval
        };
        let f_minus = {
            let $var = $x - $h;
            $eval
        };
        assert_eq!(f_plus.len(), $analytical.len());
        for j in 0..$analytical.len() {
            let fd = (f_plus[j] - f_minus[j]) / (2.0 * $h);
            approx::assert_abs_diff_eq!(fd, $analytical[j], epsilon = $tol);
        }
    };
}
