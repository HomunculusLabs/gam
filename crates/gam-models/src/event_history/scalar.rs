//! Elementary functions over any [`JetField`] scalar, so the Laplace evidence
//! of an event history is written once and evaluated as a plain `f64`, as a
//! one-direction dual (`OneSeed<0>`) for `D_θ H[u]`, or as a two-direction
//! dual (`TwoSeed<0>`) for `D²_θ H[u, v]`, each optionally wrapped in a
//! [`Tangent`] that carries the coefficient-direction derivatives the
//! Hessian columns need.
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

/// A first-order forward-mode dual over a base scalar `B` with `W` inline
/// tangent slots: the value plus its derivative along each seeded
/// coefficient direction.
///
/// The base carries whatever derivative channels the caller needs on top
/// (none for `f64`, one direction for `OneSeed<0>`, two for `TwoSeed<0>`),
/// so the tangent slots of a quantity computed on `Tangent<OneSeed<0>, W>`
/// hold that quantity's mixed second derivatives: coefficient direction
/// times the base direction. That is how one generic evaluation of the exact
/// evidence gradient yields the Hessian, its directional derivative, and its
/// second directional derivative.
///
/// The slots live inline so the evaluation allocates nothing per scalar;
/// wider coefficient vectors are swept in chunks.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Tangent<B, const W: usize> {
    pub value: B,
    pub grad: [B; W],
}

impl<B: JetField + Copy, const W: usize> Tangent<B, W> {
    pub(crate) fn seeded(value: B, grad: [B; W]) -> Self {
        Self { value, grad }
    }
}

impl<B: JetField + Copy, const W: usize> JetField for Tangent<B, W> {
    #[inline]
    fn value(&self) -> f64 {
        self.value.value()
    }
    #[inline]
    fn add(&self, o: &Self) -> Self {
        let mut grad = self.grad;
        for (g, b) in grad.iter_mut().zip(o.grad.iter()) {
            *g = g.add(b);
        }
        Self {
            value: self.value.add(&o.value),
            grad,
        }
    }
    #[inline]
    fn sub(&self, o: &Self) -> Self {
        let mut grad = self.grad;
        for (g, b) in grad.iter_mut().zip(o.grad.iter()) {
            *g = g.sub(b);
        }
        Self {
            value: self.value.sub(&o.value),
            grad,
        }
    }
    #[inline]
    fn mul(&self, o: &Self) -> Self {
        let mut grad = self.grad;
        for (g, b) in grad.iter_mut().zip(o.grad.iter()) {
            *g = g.mul(&o.value).add(&self.value.mul(b));
        }
        Self {
            value: self.value.mul(&o.value),
            grad,
        }
    }
    #[inline]
    fn neg(&self) -> Self {
        let mut grad = self.grad;
        for g in grad.iter_mut() {
            *g = g.neg();
        }
        Self {
            value: self.value.neg(),
            grad,
        }
    }
    #[inline]
    fn scale(&self, s: f64) -> Self {
        let mut grad = self.grad;
        for g in grad.iter_mut() {
            *g = g.scale(s);
        }
        Self {
            value: self.value.scale(s),
            grad,
        }
    }
    /// `f(value + Σ ε_w grad_w) = f(value) + Σ ε_w f'(value) grad_w`, with
    /// `f'` itself evaluated on the base scalar through the shifted stack
    /// `[f', f'', f''', f'''', ·]`. The bases used here carry at most two
    /// derivative orders, so the fifth entry of the shifted stack is never
    /// read.
    #[inline]
    fn compose_unary(&self, d: [f64; 5]) -> Self {
        let derivative = self.value.compose_unary([d[1], d[2], d[3], d[4], 0.0]);
        let mut grad = self.grad;
        for g in grad.iter_mut() {
            *g = g.mul(&derivative);
        }
        Self {
            value: self.value.compose_unary(d),
            grad,
        }
    }
    #[inline]
    fn constant_like(&self, v: f64) -> Self {
        Self {
            value: self.value.constant_like(v),
            grad: [self.value.constant_like(0.0); W],
        }
    }
    #[inline]
    fn with_value(&self, v: f64) -> Self {
        Self {
            value: self.value.with_value(v),
            grad: self.grad,
        }
    }
}
