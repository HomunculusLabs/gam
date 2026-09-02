//! Elementary functions over any [`JetField`] scalar, so the exact-marginal
//! event-history likelihood is written once and evaluated as a plain `f64`,
//! as a one-direction dual (`OneSeed<0>`) for `D_θ H[u]`, or as a two-direction
//! dual (`TwoSeed<0>`) for `D²_θ H[u, v]`.
//!
//! Every function composes through [`JetField::compose_unary`] with the exact
//! derivative stack of the outer real function, so no derivative channel is
//! ever approximated.

use gam_math::nested_dual::JetField;

/// `exp(x)`.
#[inline]
pub(crate) fn exp<S: JetField>(x: &S) -> S {
    let e = x.value().exp();
    x.compose_unary([e, e, e, e, e])
}

/// `ln(x)` for `x > 0`.
#[inline]
pub(crate) fn ln<S: JetField>(x: &S) -> S {
    let u = x.value();
    let i = 1.0 / u;
    let i2 = i * i;
    x.compose_unary([u.ln(), i, -i2, 2.0 * i2 * i, -6.0 * i2 * i2])
}

/// `sqrt(x)` for `x > 0`.
#[inline]
pub(crate) fn sqrt<S: JetField>(x: &S) -> S {
    let u = x.value();
    let s = u.sqrt();
    let i = 1.0 / u;
    let i2 = i * i;
    x.compose_unary([
        s,
        0.5 * s * i,
        -0.25 * s * i2,
        0.375 * s * i2 * i,
        -0.9375 * s * i2 * i2,
    ])
}

/// `1 / x` for `x != 0`.
#[inline]
pub(crate) fn recip<S: JetField>(x: &S) -> S {
    let i = 1.0 / x.value();
    let i2 = i * i;
    x.compose_unary([i, -i2, 2.0 * i2 * i, -6.0 * i2 * i2, 24.0 * i2 * i2 * i])
}

/// `a / b`.
#[inline]
pub(crate) fn div<S: JetField>(a: &S, b: &S) -> S {
    a.mul(&recip(b))
}

/// `x + c` for a real constant `c`.
#[inline]
pub(crate) fn add_real<S: JetField>(x: &S, c: f64) -> S {
    x.add(&x.constant_like(c))
}

/// `x * x`.
#[inline]
pub(crate) fn square<S: JetField>(x: &S) -> S {
    x.mul(x)
}

