//! Gauss-Hermite–Lagrange representation of one subject's latent chain.
//!
//! The latent state at every node is a product of independent unit-variance
//! atoms. Every density or likelihood-weighted function on the state is
//! carried as its values on an adaptive product Gauss-Hermite grid whose
//! per-axis centre and scale follow the predicted moments. The two operators
//! the chain needs — predict forward across a gap and condition backward
//! across a gap — are Gaussian convolutions, and a Gaussian convolution of
//! `envelope × polynomial` is exact: the product of the transition kernel and
//! the grid envelope is again a Gaussian, so the convolution is an expectation
//! of the Lagrange interpolant under that Gaussian, evaluated by Gauss-Hermite
//! quadrature to which it is exact. Both operators are therefore `G × G`
//! matrices per axis, valid for gaps of any size, including gaps far shorter
//! than the grid spacing where direct kernel quadrature fails.
//!
//! Everything is generic over a [`JetField`] scalar so the same code yields
//! the value, one directional derivative, or a mixed second directional
//! derivative of every quantity downstream.

use super::cohort::EventHistoryError;
use super::scalar::{add_real, div, exp, recip, sqrt, square};
use gam_math::nested_dual::JetField;

/// Physicists' Gauss-Hermite rule with the derived constants the chain uses.
#[derive(Clone, Debug)]
pub(crate) struct GaussHermite {
    pub order: usize,
    /// Nodes `x_l` of `∫ e^{-x²} f(x) dx ≈ Σ w_l f(x_l)`.
    pub nodes: Vec<f64>,
    /// `w_l / √π`: weights of the standard-normal expectation
    /// `E[f(Z)] = Σ (w_l/√π) f(√2 x_l)`.
    pub normal_weights: Vec<f64>,
    /// `w_l e^{x_l²}`: weights of plain integration `∫ g(x) dx ≈ Σ (w_l e^{x_l²}) g(x_l)`.
    pub plain_weights: Vec<f64>,
    /// Barycentric weights of the nodes, scaled to unit maximum.
    pub barycentric: Vec<f64>,
}

impl GaussHermite {
    pub fn new(order: usize) -> Result<Self, EventHistoryError> {
        if order == 0 {
            return Err(EventHistoryError::InvalidInput {
                reason: "Gauss-Hermite order must be positive".to_string(),
            });
        }
        let rule = gam_math::quadrature::gauss_hermite_rule(order).map_err(|error| {
            EventHistoryError::NumericalFailure {
                reason: format!("Gauss-Hermite rule of order {order} failed: {error}"),
            }
        })?;
        let sqrt_pi = std::f64::consts::PI.sqrt();
        let nodes = rule.nodes.clone();
        let normal_weights: Vec<f64> = rule.weights.iter().map(|w| w / sqrt_pi).collect();
        let plain_weights: Vec<f64> = rule
            .weights
            .iter()
            .zip(nodes.iter())
            .map(|(w, x)| (w.ln() + x * x).exp())
            .collect();
        let mut log_bary = vec![0.0; order];
        let mut sign = vec![1.0; order];
        for i in 0..order {
            for m in 0..order {
                if m != i {
                    let d = nodes[i] - nodes[m];
                    log_bary[i] -= d.abs().ln();
                    if d < 0.0 {
                        sign[i] = -sign[i];
                    }
                }
            }
        }
        let max_log = log_bary.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let barycentric: Vec<f64> = log_bary
            .iter()
            .zip(sign.iter())
            .map(|(l, s)| s * (l - max_log).exp())
            .collect();
        Ok(Self {
            order,
            nodes,
            normal_weights,
            plain_weights,
            barycentric,
        })
    }

    /// Lagrange basis on the rule's nodes evaluated at `xi`.
    pub fn lagrange_basis<S: JetField>(&self, xi: &S) -> Vec<S> {
        let g = self.order;
        let value = xi.value();
        if let Some(hit) = self.nodes.iter().position(|&x| x == value) {
            return (0..g)
                .map(|i| xi.constant_like(if i == hit { 1.0 } else { 0.0 }))
                .collect();
        }
        let terms: Vec<S> = (0..g)
            .map(|i| recip(&add_real(xi, -self.nodes[i])).scale(self.barycentric[i]))
            .collect();
        let denominator = terms
            .iter()
            .fold(xi.constant_like(0.0), |acc, t| acc.add(t));
        let inverse = recip(&denominator);
        terms.iter().map(|t| t.mul(&inverse)).collect()
    }
}

/// One axis of a product grid: centre, scale, node values and plain weights.
#[derive(Clone, Debug)]
pub(crate) struct Axis<S> {
    pub mu: S,
    pub sigma: S,
    /// `z_i = mu + √2 sigma x_i`.
    pub points: Vec<S>,
    /// `√2 sigma · w_i e^{x_i²}`: plain integration weights in `z`.
    pub weights: Vec<S>,
}

/// Product Gauss-Hermite grid over `k` independent axes, axis 0 fastest.
#[derive(Clone, Debug)]
pub(crate) struct Grid<S> {
    pub axes: Vec<Axis<S>>,
    pub order: usize,
    /// Product integration weight of every flat point.
    pub weights: Vec<S>,
}

impl<S: JetField> Grid<S> {
    pub fn new(gh: &GaussHermite, centres: &[S], scales: &[S], like: &S) -> Self {
        let order = gh.order;
        let axes: Vec<Axis<S>> = centres
            .iter()
            .zip(scales.iter())
            .map(|(mu, sigma)| {
                let points = gh
                    .nodes
                    .iter()
                    .map(|&x| mu.add(&sigma.scale(std::f64::consts::SQRT_2 * x)))
                    .collect();
                let weights = gh
                    .plain_weights
                    .iter()
                    .map(|&w| sigma.scale(std::f64::consts::SQRT_2 * w))
                    .collect();
                Axis {
                    mu: mu.clone(),
                    sigma: sigma.clone(),
                    points,
                    weights,
                }
            })
            .collect();
        let size = order.pow(axes.len() as u32);
        let mut weights = Vec::with_capacity(size);
        for flat in 0..size {
            let mut w = like.constant_like(1.0);
            let mut rest = flat;
            for axis in &axes {
                let i = rest % order;
                rest /= order;
                w = w.mul(&axis.weights[i]);
            }
            weights.push(w);
        }
        Grid {
            axes,
            order,
            weights,
        }
    }

    pub fn dimension(&self) -> usize {
        self.axes.len()
    }

    pub fn size(&self) -> usize {
        self.weights.len()
    }

    /// Coordinate index of flat point `flat` along `axis`.
    #[inline]
    pub fn index(&self, flat: usize, axis: usize) -> usize {
        (flat / self.order.pow(axis as u32)) % self.order
    }

    /// The `axis` coordinate of flat point `flat`.
    #[inline]
    pub fn coordinate(&self, flat: usize, axis: usize) -> &S {
        &self.axes[axis].points[self.index(flat, axis)]
    }
}

/// Apply a per-axis `G × G` matrix (row-major `[j * G + i]`, output index
/// `j`, input index `i`) along `axis` of a flat tensor.
fn apply_axis<S: JetField>(values: &[S], matrix: &[S], order: usize, axis: usize) -> Vec<S> {
    let stride = order.pow(axis as u32);
    let block = stride * order;
    let total = values.len();
    let zero = values[0].constant_like(0.0);
    let mut out = vec![zero.clone(); total];
    let mut base = 0;
    while base < total {
        for inner in 0..stride {
            for j in 0..order {
                let mut acc = zero.clone();
                for i in 0..order {
                    acc = acc.add(&matrix[j * order + i].mul(&values[base + inner + i * stride]));
                }
                out[base + inner + j * stride] = acc;
            }
        }
        base += block;
    }
    out
}

/// A separable linear operator: one `G × G` matrix per axis.
#[derive(Clone, Debug)]
pub(crate) struct SeparableOperator<S> {
    pub matrices: Vec<Vec<S>>,
    pub order: usize,
}

impl<S: JetField> SeparableOperator<S> {
    pub fn apply(&self, values: &[S]) -> Vec<S> {
        let mut current = values.to_vec();
        for (axis, matrix) in self.matrices.iter().enumerate() {
            current = apply_axis(&current, matrix, self.order, axis);
        }
        current
    }
}

/// Transition of one unit-variance Ornstein–Uhlenbeck atom across a gap of
/// dimensionless length `kappa = rate · gap`: `z' | z ~ N(φ z, 1 − φ²)`.
#[derive(Clone, Debug)]
pub(crate) struct AtomTransition<S> {
    /// `φ = e^{-κ}`.
    pub phi: S,
    /// `q = 1 − φ²`, the innovation variance.
    pub innovation: S,
    /// `dφ/dρ = −κ φ` with `ρ = ln rate`.
    pub dphi: S,
    /// `d²φ/dρ² = κ φ (κ − 1)`.
    pub d2phi: S,
}

impl<S: JetField> AtomTransition<S> {
    pub fn new(kappa: &S) -> Self {
        let k = kappa.value();
        let e = (-k).exp();
        let one_minus_phi = kappa.compose_unary([-(-k).exp_m1(), e, -e, e, -e]);
        let phi = one_minus_phi.neg().add(&kappa.constant_like(1.0));
        let innovation = one_minus_phi.mul(&add_real(&phi, 1.0));
        let dphi = kappa.mul(&phi).neg();
        let d2phi = kappa.mul(&phi).mul(&add_real(kappa, -1.0));
        Self {
            phi,
            innovation,
            dphi,
            d2phi,
        }
    }
}

const LOG_SQRT_TWO_PI: f64 = 0.918_938_533_204_672_8;

/// `N(x; mean, variance)` as a scalar.
pub(crate) fn normal_density<S: JetField>(x: &S, mean: &S, variance: &S) -> S {
    let d = x.sub(mean);
    let quad = div(&square(&d), variance).scale(-0.5);
    let log_norm = super::scalar::ln(variance).scale(-0.5);
    exp(&quad.add(&log_norm).add(&x.constant_like(-LOG_SQRT_TWO_PI)))
}

/// Build the forward (predict) operator across one gap: input values on
/// `from`, output values on `to`.
pub(crate) fn forward_operator<S: JetField>(
    gh: &GaussHermite,
    from: &Grid<S>,
    to: &Grid<S>,
    transitions: &[AtomTransition<S>],
) -> SeparableOperator<S> {
    let g = gh.order;
    let mut matrices = Vec::with_capacity(from.dimension());
    for (axis, transition) in transitions.iter().enumerate() {
        let old = &from.axes[axis];
        let new = &to.axes[axis];
        let phi = &transition.phi;
        let q = &transition.innovation;
        let sigma2 = square(&old.sigma);
        let tau2 = square(phi).mul(&sigma2).add(q);
        let inv_tau2 = recip(&tau2);
        let ratio = sqrt(&q.mul(&inv_tau2));
        let phi_mu = phi.mul(&old.mu);
        // 1 / e(z_i) with e = N(z_i; mu, sigma²): √(2π) σ e^{x_i²}.
        let inverse_envelope: Vec<S> = gh
            .nodes
            .iter()
            .map(|&x| {
                old.sigma
                    .scale((2.0 * std::f64::consts::PI).sqrt() * (x * x).exp())
            })
            .collect();
        let mut matrix = vec![old.mu.constant_like(0.0); g * g];
        for j in 0..g {
            let d = new.points[j].sub(&phi_mu);
            let gauss = normal_density(&new.points[j], &phi_mu, &tau2);
            let centre = phi
                .mul(&old.sigma)
                .mul(&d)
                .mul(&inv_tau2)
                .scale(1.0 / std::f64::consts::SQRT_2);
            let mut accumulated = vec![old.mu.constant_like(0.0); g];
            for (l, &x) in gh.nodes.iter().enumerate() {
                let xi = centre.add(&ratio.scale(x));
                let basis = gh.lagrange_basis(&xi);
                for i in 0..g {
                    accumulated[i] = accumulated[i].add(&basis[i].scale(gh.normal_weights[l]));
                }
            }
            for i in 0..g {
                matrix[j * g + i] = gauss.mul(&accumulated[i]).mul(&inverse_envelope[i]);
            }
        }
        matrices.push(matrix);
    }
    SeparableOperator { matrices, order: g }
}

/// Build the backward (conditioning) operator across one gap: input values
/// on `to`, output values on `from`: `B[u](z) = ∫ N(z'; φ z, q) u(z') dz'`.
pub(crate) fn backward_operator<S: JetField>(
    gh: &GaussHermite,
    from: &Grid<S>,
    to: &Grid<S>,
    transitions: &[AtomTransition<S>],
) -> SeparableOperator<S> {
    let g = gh.order;
    let mut matrices = Vec::with_capacity(from.dimension());
    for (axis, transition) in transitions.iter().enumerate() {
        let old = &from.axes[axis];
        let new = &to.axes[axis];
        let phi = &transition.phi;
        let q = &transition.innovation;
        let sigma2_new = square(&new.sigma);
        let total = sigma2_new.add(q);
        let inv_total = recip(&total);
        let ratio = sqrt(&q.mul(&inv_total));
        let inverse_envelope: Vec<S> = gh
            .nodes
            .iter()
            .map(|&x| {
                new.sigma
                    .scale((2.0 * std::f64::consts::PI).sqrt() * (x * x).exp())
            })
            .collect();
        let mut matrix = vec![old.mu.constant_like(0.0); g * g];
        for i in 0..g {
            let phi_z = phi.mul(&old.points[i]);
            let gauss = normal_density(&phi_z, &new.mu, &total);
            let centre = new
                .sigma
                .mul(&phi_z.sub(&new.mu))
                .mul(&inv_total)
                .scale(1.0 / std::f64::consts::SQRT_2);
            let mut accumulated = vec![old.mu.constant_like(0.0); g];
            for (l, &x) in gh.nodes.iter().enumerate() {
                let xi = centre.add(&ratio.scale(x));
                let basis = gh.lagrange_basis(&xi);
                for j in 0..g {
                    accumulated[j] = accumulated[j].add(&basis[j].scale(gh.normal_weights[l]));
                }
            }
            for j in 0..g {
                matrix[i * g + j] = gauss.mul(&accumulated[j]).mul(&inverse_envelope[j]);
            }
        }
        matrices.push(matrix);
    }
    SeparableOperator { matrices, order: g }
}
