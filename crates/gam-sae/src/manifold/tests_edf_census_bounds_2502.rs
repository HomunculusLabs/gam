//! #2502 regression: the curvature census is a degrees-of-freedom at EVERY
//! lambda, with no precision cliff.
//!
//! `penalized_trace_and_null_dim` returns `tr((G + lambda*S)^-1 G)` and
//! `dim null(S)`; callers subtract the second from the first, so the result is
//! an effective degrees of freedom exactly when `null_dim <= trace <= m`. Both
//! bounds are structural. `lambda*S` is PSD, so `v'Gv <= v'(G + lambda*S)v`
//! and every mode contributes at most one. The modes spanning `null(S)` see
//! `G` alone and contribute exactly one each.
//!
//! Two defects had to be removed before those bounds held.
//!
//! The mode tolerance was scaled by `max|eigenvalue(G + lambda*S)|`, which
//! lambda dominates, so a large lambda discarded the very modes that guarantee
//! the lower bound; the census returned exactly `-null_dim`, and
//! Fellner-Schall, reading `signal = tau - null_dim < 0`, froze every atom
//! that reported it. Observed in production at `lambda = 6.339e15`.
//!
//! Rescaling that tolerance was not enough. While `G + lambda*S` was still
//! assembled, the sum itself rounds to `lambda*S` once `lambda*max|S|` passes
//! `max|G| / eps`, and `G` is gone from the matrix before any tolerance is
//! consulted. The census now uses the Jacobi identity that
//! `solve_penalized_normal_equations` already used for the same reason --
//! `tr((G + lambda*S)^-1 G) = tr((G~ + P~)^-1 G~)` with
//! `P~ = diag(lambda*s/(1+lambda*s))` bounded in [0, 1) -- so no matrix whose
//! condition number is lambda is ever formed, and the sweep below runs to
//! 1e230 rather than stopping at a cliff.

use super::support_term::penalized_trace_and_null_dim;
use ndarray::{array, Array2};

/// A penalty with a one-dimensional null space, spanned by `(1, 1)`, and a
/// Gram with mass in that direction -- so the lower bound has something to be
/// violated by.
fn fixture() -> (Array2<f64>, Array2<f64>) {
    (
        array![[2.0, 0.5], [0.5, 1.0]],
        array![[1.0, -1.0], [-1.0, 1.0]],
    )
}

/// Spans the benign range, the value measured in the K=8096 arms, and far past
/// the point where assembling `G + lambda*S` loses `G` to rounding.
const LAMBDAS: [f64; 9] = [
    0.0, 1.0e-8, 1.0, 1.0e6, 1.0e12, 6.339e15, 1.0e30, 1.0e100, 1.0e230,
];

#[test]
fn penalized_trace_stays_a_degrees_of_freedom_at_every_lambda_2502() {
    let (gram, penalty) = fixture();
    let m = gram.nrows() as f64;
    let mut previous = f64::INFINITY;
    for lambda in LAMBDAS {
        let (trace, null_dim) = penalized_trace_and_null_dim(&gram, &penalty, lambda, "test")
            .expect("the kernel solves a 2x2 symmetric problem");
        assert_eq!(null_dim, 1.0, "the penalty has a one-dimensional null space");
        assert!(
            trace <= m + 1.0e-9,
            "lambda*S is PSD so every mode contributes at most one: \
             trace {trace} exceeds m {m} at lambda {lambda:e}"
        );
        assert!(
            trace >= null_dim - 1.0e-9,
            "the modes spanning null(S) contribute one each: trace {trace} fell \
             below null_dim {null_dim} at lambda {lambda:e} -- the pre-#2502 \
             census returned edf = -null_dim here"
        );
        assert!(
            trace <= previous + 1.0e-9,
            "shrinkage is monotone: trace rose from {previous} to {trace} at lambda {lambda:e}"
        );
        previous = trace;
    }
}

#[test]
fn penalized_trace_reaches_both_limits_exactly_2502() {
    let (gram, penalty) = fixture();

    // No penalty: the fit spends every basis direction it has.
    let (unpenalized, null_dim) = penalized_trace_and_null_dim(&gram, &penalty, 0.0, "test")
        .expect("the kernel solves a 2x2 symmetric problem");
    assert!(
        (unpenalized - gram.nrows() as f64).abs() < 1.0e-9,
        "at lambda = 0 the trace is m, got {unpenalized}"
    );

    // Overwhelming penalty: only the null space survives. The Jacobi form makes
    // this limit exact rather than approaching a cliff -- before it, this
    // lambda produced a trace of 5e-231 and an edf of -1.
    let (railed, _) = penalized_trace_and_null_dim(&gram, &penalty, 1.0e230, "test")
        .expect("the kernel solves a 2x2 symmetric problem");
    assert!(
        (railed - null_dim).abs() < 1.0e-9,
        "at overwhelming lambda the trace is null_dim ({null_dim}), got {railed}"
    );
}
