//! Linear algebra helpers for `gam`: faer/ndarray bridges, matrix operators,
//! sparse solves, iterative solvers, and numerical stability utilities.

#[macro_export]
macro_rules! impl_reason_error_boilerplate {
    ($type:ident { $($variant:ident),+ $(,)? }) => {
        impl ::std::fmt::Display for $type {
            fn fmt(&self, f: &mut ::std::fmt::Formatter<'_>) -> ::std::fmt::Result {
                match self {
                    $(Self::$variant { reason })|+ => f.write_str(reason),
                }
            }
        }

        impl ::std::error::Error for $type {}

        impl From<$type> for String {
            fn from(err: $type) -> String {
                err.to_string()
            }
        }
    };
}

pub mod decision;
pub mod dense;
mod error;
pub mod faer_ndarray;
pub mod gaussian_weighted_ridge_backward;
pub mod governed_capture;
pub mod gpu_hook;
pub mod lanczos;
pub mod low_rank_weight;
pub mod matrix;
pub mod numeric_derivative;
pub mod pairwise_reduce;
pub mod parallel;
pub mod pcg;
pub mod psd_trust_region;
pub mod roundoff;
pub mod sparse_exact;
pub mod test_support;
pub mod triangular;
pub mod types;
pub mod utils;

pub use error::LinalgError;
pub use types::{RidgeDeterminantMode, RidgePolicy};

/// Assert that a central difference of an array-producing function matches the
/// analytical derivative.
///
/// Expands `approx::assert_abs_diff_eq!` at the call site, so the caller needs
/// `approx` in scope; it deliberately does not force that dependency on this
/// crate's non-test build.
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
