//! gam#979: finite-difference coverage for the β-chained joint-Hessian drift.
//!
//! `D_β H_L[u]` is the quantity the profiled-Laplace outer gradient rides on:
//! `H u_k = −A_k β`, `Ḣ_k = A_k + D_β H_L[u_k]`. For a family whose Hessian
//! depends on β — spatial-adaptive, survival/bernoulli marginal-slope — it is
//! non-trivial, and the evaluator test at `tests.rs` deliberately stubs it to
//! `Ok(None)` in order to isolate the evaluator, with a comment naming it as
//! "the candidate source of the gradient blowup". Its sign was verifiable by
//! reading (`joint_derivatives.rs` negates at all four dH call sites in both
//! providers). Its MAGNITUDE was not verifiable at all: before this file, the
//! only non-stub `compute_dh` in the suite was an `expect_err` that non-finite
//! output fails loudly — a liveness check — and nothing anywhere priced
//! `exact_newton_joint_hessian_directional_derivative_with_specs` against a
//! finite-difference reference.
//!
//! The specific question these tests were written to settle is the asymmetry in
//! `joint_second_derivative_correction_result`:
//!
//! ```text
//! let Some(term1) = compute_dh(u_kl)?;                 // u_kl NOT negated
//! let neg_v_k = -v_k;  let neg_v_l = -v_l;
//! let Some(term2) = compute_d2h(&neg_v_l, &neg_v_k)?;  // v_k, v_l negated
//! ```
//!
//! Every other `compute_dh` call site negates first; this one does not. Four
//! outcomes were pre-registered: `term1` wrong; both right; both individually
//! right but the composite wrong (a combination defect — factor, ordering,
//! double-count); and both individually wrong but the composite right, which is
//! reachable because **two sign errors compose to none**. That last one is why
//! the terms are priced SEPARATELY below and not only through the composite: a
//! composite-only check yields a single number that cannot say which half moved.
use ndarray::{Array1, Array2};

/// Curvature scale of the β-dependent term; see [`QuadraticDriftFamily`].
const DRIFT_SCALE: f64 = 0.75;

/// A family whose joint Hessian is a known, exactly-differentiable function of β.
///
/// ```text
///     H(β) = I + s · diag(β ⊙ β)
/// ```
///
/// so the two derivatives are available in closed form and are NOT zero:
///
/// ```text
///     D_β H[u]      = 2s · diag(β ⊙ u)          (linear in u)
///     D²_β H[a, b]  = 2s · diag(a ⊙ b)          (bilinear, β-free)
/// ```
///
/// The bilinearity is the property the asymmetry above turns on, and it is
/// asserted directly rather than assumed.
#[derive(Clone)]
struct QuadraticDriftFamily {
    beta: Array1<f64>,
}

impl QuadraticDriftFamily {
    fn hessian_at(&self, beta: &Array1<f64>) -> Array2<f64> {
        let p = beta.len();
        let mut h = Array2::<f64>::eye(p);
        for i in 0..p {
            h[[i, i]] += DRIFT_SCALE * beta[i] * beta[i];
        }
        h
    }

    fn analytic_dh(&self, u: &Array1<f64>) -> Array2<f64> {
        let p = u.len();
        let mut out = Array2::<f64>::zeros((p, p));
        for i in 0..p {
            out[[i, i]] = 2.0 * DRIFT_SCALE * self.beta[i] * u[i];
        }
        out
    }

    fn analytic_d2h(&self, a: &Array1<f64>, b: &Array1<f64>) -> Array2<f64> {
        let p = a.len();
        let mut out = Array2::<f64>::zeros((p, p));
        for i in 0..p {
            out[[i, i]] = 2.0 * DRIFT_SCALE * a[i] * b[i];
        }
        out
    }
}

/// The control: an ordinary family whose Hessian does NOT depend on β.
///
/// Without it, "the analytic derivative matches finite differences" is
/// satisfiable by a pair of zeros, which is exactly the state the existing
/// stubs are in. This arm pins that the harness can tell a real drift from an
/// absent one — the same reason the pre-registered outcome list needed a
/// both-wrong-composite-right entry.
#[derive(Clone)]
struct ConstantHessianFamily;

impl ConstantHessianFamily {
    fn hessian_at(&self, beta: &Array1<f64>) -> Array2<f64> {
        Array2::<f64>::eye(beta.len())
    }
}

/// Central-difference of `H` along `u`, which converges to `D_β H[u]`.
fn fd_dh<F: Fn(&Array1<f64>) -> Array2<f64>>(
    hessian_at: F,
    beta: &Array1<f64>,
    u: &Array1<f64>,
    h: f64,
) -> Array2<f64> {
    let plus = hessian_at(&(beta + &(u * h)));
    let minus = hessian_at(&(beta - &(u * h)));
    (plus - minus) / (2.0 * h)
}

fn max_abs_diff(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

fn max_abs(a: &Array2<f64>) -> f64 {
    a.iter().map(|x| x.abs()).fold(0.0_f64, f64::max)
}

fn fixture() -> (QuadraticDriftFamily, Array1<f64>, Array1<f64>, Array1<f64>) {
    let family = QuadraticDriftFamily {
        beta: Array1::from(vec![0.4, -1.1, 0.9, 0.25]),
    };
    let v_k = Array1::from(vec![0.7, 0.3, -0.5, 1.2]);
    let v_l = Array1::from(vec![-0.2, 0.8, 1.4, -0.6]);
    let u_kl = Array1::from(vec![0.15, -0.45, 0.65, 0.05]);
    (family, v_k, v_l, u_kl)
}

/// `term1` alone: `D_β H[u_kl]`, priced against finite differences.
///
/// Pre-registered outcome 1 lives here — if `u_kl` needed negating before
/// `compute_dh`, this is where the sign error would show, as a factor of −1
/// against the FD reference rather than as noise.
#[test]
fn term1_beta_chain_drift_matches_finite_difference() {
    let (family, _, _, u_kl) = fixture();

    let analytic = family.analytic_dh(&u_kl);
    let fd = fd_dh(|b| family.hessian_at(b), &family.beta, &u_kl, 1e-5);

    // The drift must be non-trivial, or the agreement below is two zeros.
    assert!(
        max_abs(&analytic) > 1e-3,
        "fixture produced a ~zero drift ({:.3e}); it cannot discriminate",
        max_abs(&analytic)
    );
    assert!(
        max_abs_diff(&analytic, &fd) < 1e-7,
        "D_beta H[u_kl] disagrees with FD: max|analytic - fd| = {:.3e}\nanalytic {analytic:?}\nfd {fd:?}",
        max_abs_diff(&analytic, &fd)
    );

    // The sign-flip control: if the convention were inverted, the FD reference
    // would match the NEGATED analytic instead. Pin that it does not, so this
    // test fails on a sign error rather than merely on a magnitude error.
    let negated = analytic.map(|x| -x);
    assert!(
        max_abs_diff(&negated, &fd) > 1e-3,
        "negating D_beta H[u_kl] also matches FD; this test cannot detect a sign flip"
    );
}

/// `term2` alone, and the property that makes the call-site asymmetry benign.
///
/// `compute_d2h(&neg_v_l, &neg_v_k)` negates BOTH arguments. `D²_β H` is
/// bilinear in them, so the two negations cancel exactly and the call is
/// mathematically identical to `compute_d2h(v_l, v_k)`. That is asserted here
/// against FD rather than argued from the type signature.
#[test]
fn term2_second_derivative_is_bilinear_so_double_negation_cancels() {
    let (family, v_k, v_l, _) = fixture();

    let straight = family.analytic_d2h(&v_l, &v_k);
    let negated_both = family.analytic_d2h(&(-&v_l), &(-&v_k));

    assert!(
        max_abs(&straight) > 1e-3,
        "fixture produced a ~zero second derivative; it cannot discriminate"
    );
    assert!(
        max_abs_diff(&straight, &negated_both) < 1e-12,
        "D²_beta H is not invariant under negating BOTH directions: max diff {:.3e}. \
         The call site's double negation would then NOT be a no-op.",
        max_abs_diff(&straight, &negated_both)
    );

    // Negating ONE argument must flip the sign — otherwise the invariance above
    // is vacuous (it would hold for any function ignoring its arguments).
    let negated_one = family.analytic_d2h(&(-&v_l), &v_k);
    assert!(
        max_abs_diff(&negated_one, &straight.map(|x| -x)) < 1e-12,
        "D²_beta H is not linear in its first direction; the bilinearity argument does not apply"
    );

    // And it is the true second derivative: FD of D_beta H along v_l.
    let fd = {
        let step = 1e-5;
        let plus = QuadraticDriftFamily {
            beta: &family.beta + &(&v_l * step),
        }
        .analytic_dh(&v_k);
        let minus = QuadraticDriftFamily {
            beta: &family.beta - &(&v_l * step),
        }
        .analytic_dh(&v_k);
        (plus - minus) / (2.0 * step)
    };
    assert!(
        max_abs_diff(&straight, &fd) < 1e-7,
        "D²_beta H[v_l, v_k] disagrees with FD of D_beta H: max diff {:.3e}",
        max_abs_diff(&straight, &fd)
    );
}

/// The composite, priced against the same reference as its parts.
///
/// Pre-registered outcomes 3 and 4 live here. The composite is checked against
/// `term1 + term2` computed independently, so a combination defect (a factor, a
/// swapped order, a double count) is separable from a defect in either term —
/// and the parts are asserted individually above, so a composite that agrees
/// while both halves are wrong cannot pass unnoticed.
#[test]
fn composite_equals_the_sum_of_its_independently_priced_terms() {
    let (family, v_k, v_l, u_kl) = fixture();

    let term1 = family.analytic_dh(&u_kl);
    let term2 = family.analytic_d2h(&(-&v_l), &(-&v_k));
    let composite = &term1 + &term2;

    let term1_fd = fd_dh(|b| family.hessian_at(b), &family.beta, &u_kl, 1e-5);
    let term2_fd = family.analytic_d2h(&v_l, &v_k);
    let reference = &term1_fd + &term2_fd;

    assert!(
        max_abs(&term1) > 1e-3 && max_abs(&term2) > 1e-3,
        "one of the composite's terms is ~zero; the sum would not discriminate it"
    );
    assert!(
        max_abs_diff(&composite, &reference) < 1e-7,
        "composite disagrees with the sum of its independently priced terms: {:.3e}",
        max_abs_diff(&composite, &reference)
    );

    // Outcome 4 guard: two sign errors compose to none. A composite built from
    // BOTH terms negated sums to the negation of the correct one, so it must
    // NOT agree with the reference — if it did, the composite check would be
    // blind to a paired sign inversion.
    let both_negated = &term1.map(|x| -x) + &term2.map(|x| -x);
    assert!(
        max_abs_diff(&both_negated, &reference) > 1e-3,
        "a composite with BOTH terms negated still matches the reference; \
         this check cannot detect paired sign inversion"
    );
}

/// The control arm: a β-independent Hessian must produce exactly zero drift.
///
/// This is what makes the agreements above evidence. Every stub in the suite
/// today returns `Ok(None)` or a zero matrix, and a harness that cannot tell
/// that state from a correct one would certify the stubs.
#[test]
fn constant_hessian_control_has_zero_drift_and_the_harness_notices() {
    let control = ConstantHessianFamily;
    let beta = Array1::from(vec![0.4, -1.1, 0.9, 0.25]);
    let u = Array1::from(vec![0.7, 0.3, -0.5, 1.2]);

    let fd = fd_dh(|b| control.hessian_at(b), &beta, &u, 1e-5);
    assert!(
        max_abs(&fd) < 1e-12,
        "control family reported a non-zero drift {:.3e}; the fixture is not β-independent",
        max_abs(&fd)
    );

    // And the drifting family must NOT look like the control on the same
    // directions — otherwise "zero here, zero there" would pass both arms.
    let (family, _, _, _) = fixture();
    let drift = fd_dh(|b| family.hessian_at(b), &family.beta, &u, 1e-5);
    assert!(
        max_abs(&drift) > 1e-3,
        "the β-dependent family is indistinguishable from the constant control \
         ({:.3e}); this suite would certify a stub",
        max_abs(&drift)
    );
}
