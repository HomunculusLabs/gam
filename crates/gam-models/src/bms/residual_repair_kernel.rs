//! The width-independent residual repair row kernel (gam#2924).
//!
//! # Five primaries, whatever the width
//!
//! The row likelihood of [`residual_row_nll`] reads the residual coefficients
//! `β` only through the row-varying read `t_i = βᵀr_i` and the anchor's two
//! moments of the drive, `u_i = βᵀγ(a_i)` and `v_i = βᵀΣ_rr(a_i)β`, with
//! `γ = Σ_r0` the first column of the declared joint covariance (the same at
//! every row under the pooled law). So the row program has the five primaries
//! `p_i = (η_m, g, t, u, v)` for any block width `K`, and its jets never grow
//! with `K`.
//!
//! # The chain rule through one quadratic map
//!
//! `t` and `u` are linear in `β` — a design row `r_i` and a coefficient row
//! `γ(a_i)` — and `v` is quadratic, with the constant second derivative
//! `D²v[d, e] = 2·d_rᵀΣ_rr e_r` and nothing above it. The coefficient-space
//! channels are therefore the generic pullback through the tangent Jacobian
//! `J_i = ∂p_i/∂θ`, whose `v` row is `2Σ_rr(a_i)β`, plus the terms in which the
//! curvature of `v` absorbs directions. Faà di Bruno with every block of size at
//! most two, writing `μ_d = 2Σ_rr d_r`, `h`/`T³`/`T⁴` the row's primary-space
//! derivatives and `h[v,·]` a row of `h`:
//!
//! ```text
//!   H        = Σ_i J_iᵀ h J_i + h_v·2Σ_rr
//!   H′[d]    = Σ_i J_iᵀ T³[Jd] J_i + μ_d ⊗ J_iᵀh[v,·] + J_iᵀh[v,·] ⊗ μ_d
//!              + (h·Jd)_v·2Σ_rr
//!   H″[d, e] = Σ_i J_iᵀ T⁴[Jd, Je] J_i + (d_rᵀμ_e)·J_iᵀ T³[e_v] J_i
//!              + μ_d ⊗ J_iᵀT³[Je][v,·] + its transpose + μ_e ⊗ J_iᵀT³[Jd][v,·] + its transpose
//!              + (T³[Jd]·Je)_v·2Σ_rr + (d_rᵀμ_e)·h_vv·2Σ_rr + h_vv·(μ_d ⊗ μ_e + μ_e ⊗ μ_d)
//! ```
//!
//! The residual block of a pullback is `O(K²)` per row and each row runs a fixed
//! number of five-primary jets, so a biobank block score with `26` or more block
//! columns costs what the rigid kernel costs plus `O(nK²)` assembly. There is no
//! width ceiling.

use super::family::*;
use super::hessian_paths::BlockSlices;
use super::residual_repair::{ResidualBlockRuntime, ResidualDrive, ResidualRowState, residual_row_nll};
use super::*;
use crate::row_kernel::{RowKernel, RowKernelCache, RowSet};
use gam_math::jet_scalar::{JetScalar, SymmetricQuadraticCoefficients};
use gam_math::jet_tower::RowProgram;
use std::borrow::Cow;
use std::ops::Range;

/// Primary index of the anchor's second drive moment `v = βᵀΣ_rrβ`.
const V: usize = 4;

/// The anchor's moments of the residual drive under one row's declared joint
/// covariance: `γ = Σ_r0`, `Σ_rrβ`, `u = γᵀβ` and `v = βᵀΣ_rrβ`.
#[derive(Clone, Debug)]
pub(super) struct DriveMoments {
    gamma: Vec<f64>,
    sigma_beta: Vec<f64>,
    u: f64,
    v: f64,
}

impl DriveMoments {
    pub(super) fn at(covariance: &MarginalSlopeCovariance, beta: &[f64]) -> Self {
        let k = beta.len();
        let mut lifted = vec![0.0_f64; k + 1];
        lifted[1..].copy_from_slice(beta);
        let mut image = vec![0.0_f64; k + 1];
        covariance.multiply(&lifted, &mut image);
        let sigma_beta = image[1..].to_vec();
        let v = beta.iter().zip(&sigma_beta).map(|(b, x)| b * x).sum();
        Self {
            gamma: (0..k).map(|j| covariance.coefficient(0, j + 1)).collect(),
            sigma_beta,
            u: image[0],
            v,
        }
    }
}

/// What the coefficient-space assembly needs beyond [`RowKernel`]: where the
/// residual coordinates sit, the residual rows of each row Jacobian, and the
/// curvature `D²v = 2Σ_rr(a_i)` of the one quadratic primary.
pub(super) trait ResidualDriveRows: RowKernel<5> {
    /// The residual coefficient range inside the flat coefficient vector.
    fn residual_range(&self) -> Range<usize>;

    /// The residual rows `(r_i, γ(a_i), 2Σ_rr(a_i)β)` of `J_i`: the `t`, `u` and
    /// `v` rows restricted to the residual coordinates.
    fn residual_jacobian_rows(&self, row: usize) -> [Vec<f64>; 3];

    /// `μ = 2Σ_rr(a_i)·d_r` for the residual part of a coefficient direction.
    fn curvature_action(&self, row: usize, d_r: &[f64]) -> Vec<f64>;

    /// `target[res, res] += scale·2Σ_rr(a_i)`.
    fn add_curvature(&self, row: usize, scale: f64, target: &mut Array2<f64>);
}

#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[inline]
fn scaled(h: &[[f64; 5]; 5], w: f64) -> [[f64; 5]; 5] {
    if w == 1.0 {
        return *h;
    }
    std::array::from_fn(|a| std::array::from_fn(|b| w * h[a][b]))
}

/// The residual parts `(rᵀd_r, γᵀd_r, 2(Σ_rrβ)ᵀd_r)` of `J_i·d`.
#[inline]
fn residual_action(rho: &[Vec<f64>; 3], d_r: &[f64]) -> [f64; 3] {
    [dot(&rho[0], d_r), dot(&rho[1], d_r), dot(&rho[2], d_r)]
}

/// `out_r += v_t·r_i + v_u·γ + v_v·2Σ_rrβ`.
#[inline]
fn residual_transpose_action(rho: &[Vec<f64>; 3], v: &[f64; 3], out_r: &mut [f64]) {
    for (j, out) in out_r.iter_mut().enumerate() {
        *out += v[0] * rho[0][j] + v[1] * rho[1][j] + v[2] * rho[2][j];
    }
}

/// The residual coordinates of `J_iᵀ·h·J_i`: the residual block is added to
/// `target_rr`; the residual columns of the two surface rows come back as the
/// K-vectors `c_q = Σ_b h[q][2+b]·ρ_b` and `c_g = Σ_b h[g][2+b]·ρ_b`, which the
/// caller scatters through its designs.
fn pullback_residual_block(
    h: &[[f64; 5]; 5],
    rho: &[Vec<f64>; 3],
    mut target_rr: ndarray::ArrayViewMut2<'_, f64>,
) -> [Vec<f64>; 2] {
    let k = rho[0].len();
    let mut projected = [vec![0.0_f64; k], vec![0.0_f64; k], vec![0.0_f64; k]];
    for (a, image) in projected.iter_mut().enumerate() {
        for b in 0..3 {
            let coefficient = h[2 + a][2 + b];
            if coefficient != 0.0 {
                for (x, r) in image.iter_mut().zip(&rho[b]) {
                    *x += coefficient * r;
                }
            }
        }
    }
    for j in 0..k {
        let (r0, r1, r2) = (rho[0][j], rho[1][j], rho[2][j]);
        let mut row = target_rr.row_mut(j);
        for l in 0..k {
            row[l] += r0 * projected[0][l] + r1 * projected[1][l] + r2 * projected[2][l];
        }
    }
    let surface = |primary: usize| -> Vec<f64> {
        (0..k)
            .map(|j| h[primary][2] * rho[0][j] + h[primary][3] * rho[1][j] + h[primary][4] * rho[2][j])
            .collect()
    };
    [surface(0), surface(1)]
}

/// `acc[res+j, c] += w·μ_j·g_c` and `acc[c, res+j] += w·g_c·μ_j`.
fn add_rank_two(acc: &mut Array2<f64>, residual_start: usize, mu: &[f64], g: &[f64], w: f64) {
    for (j, &m) in mu.iter().enumerate() {
        let scale = w * m;
        if scale == 0.0 {
            continue;
        }
        for (c, &gc) in g.iter().enumerate() {
            let value = scale * gc;
            acc[[residual_start + j, c]] += value;
            acc[[c, residual_start + j]] += value;
        }
    }
}

/// `J_iᵀ·v` as a flat coefficient vector.
fn transpose_vector(kern: &(impl ResidualDriveRows + ?Sized), row: usize, v: &[f64; 5]) -> Vec<f64> {
    let mut out = vec![0.0_f64; kern.n_coefficients()];
    kern.jacobian_transpose_action(row, v, &mut out);
    out
}

/// The negative log-likelihood Hessian in coefficient space from a row cache:
/// `Σ_i J_iᵀ h_i J_i + h_{i,v}·2Σ_rr(a_i)`.
pub(super) fn residual_hessian_dense(
    kern: &(impl ResidualDriveRows + ?Sized),
    cache: &RowKernelCache<5>,
) -> Result<Array2<f64>, String> {
    let n = kern.n_rows();
    let p = kern.n_coefficients();
    RowSet::All.par_try_reduce_fold(
        n,
        || Array2::<f64>::zeros((p, p)),
        |mut acc, row, w| -> Result<_, String> {
            kern.add_pullback_hessian(row, &scaled(&cache.hessians[row], w), &mut acc);
            kern.add_curvature(row, w * cache.gradients[row][V], &mut acc);
            Ok(acc)
        },
        |a, b| Ok(a + b),
    )
}

/// `H′[d] = ∂H/∂θ[d]` in coefficient space (see the module docs).
pub(super) fn residual_hessian_directional_derivative(
    kern: &(impl ResidualDriveRows + ?Sized),
    d_beta: &[f64],
) -> Result<Array2<f64>, String> {
    let n = kern.n_rows();
    let p = kern.n_coefficients();
    if d_beta.len() != p {
        return Err(format!(
            "residual repair directional derivative: direction has {} entries, expected {p}",
            d_beta.len()
        ));
    }
    let residual = kern.residual_range();
    RowSet::All.par_try_reduce_fold(
        n,
        || Array2::<f64>::zeros((p, p)),
        |mut acc, row, w| -> Result<_, String> {
            let jd = kern.jacobian_action(row, d_beta);
            let (_, _, h) = kern.row_kernel(row)?;
            let third = kern.row_third_contracted(row, &jd)?;
            kern.add_pullback_hessian(row, &scaled(&third, w), &mut acc);
            let mu_d = kern.curvature_action(row, &d_beta[residual.clone()]);
            let h_v = transpose_vector(kern, row, &h[V]);
            add_rank_two(&mut acc, residual.start, &mu_d, &h_v, w);
            kern.add_curvature(row, w * dot(&h[V], &jd), &mut acc);
            Ok(acc)
        },
        |a, b| Ok(a + b),
    )
}

/// `H″[d, e] = ∂²H/∂θ²[d, e]` in coefficient space (see the module docs).
pub(super) fn residual_hessian_second_directional_derivative(
    kern: &(impl ResidualDriveRows + ?Sized),
    d_beta: &[f64],
    e_beta: &[f64],
) -> Result<Array2<f64>, String> {
    let n = kern.n_rows();
    let p = kern.n_coefficients();
    if d_beta.len() != p || e_beta.len() != p {
        return Err(format!(
            "residual repair second directional derivative: directions have {} / {} entries, \
             expected {p}",
            d_beta.len(),
            e_beta.len()
        ));
    }
    let residual = kern.residual_range();
    let unit_v: [f64; 5] = std::array::from_fn(|a| if a == V { 1.0 } else { 0.0 });
    RowSet::All.par_try_reduce_fold(
        n,
        || Array2::<f64>::zeros((p, p)),
        |mut acc, row, w| -> Result<_, String> {
            let jd = kern.jacobian_action(row, d_beta);
            let je = kern.jacobian_action(row, e_beta);
            let (_, _, h) = kern.row_kernel(row)?;
            let fourth = kern.row_fourth_contracted(row, &jd, &je)?;
            kern.add_pullback_hessian(row, &scaled(&fourth, w), &mut acc);
            let third_d = kern.row_third_contracted(row, &jd)?;
            let third_e = kern.row_third_contracted(row, &je)?;
            let mu_d = kern.curvature_action(row, &d_beta[residual.clone()]);
            let mu_e = kern.curvature_action(row, &e_beta[residual.clone()]);
            let kappa = dot(&d_beta[residual.clone()], &mu_e);
            if kappa != 0.0 {
                let third_v = kern.row_third_contracted(row, &unit_v)?;
                kern.add_pullback_hessian(row, &scaled(&third_v, w * kappa), &mut acc);
            }
            add_rank_two(&mut acc, residual.start, &mu_d, &transpose_vector(kern, row, &third_e[V]), w);
            add_rank_two(&mut acc, residual.start, &mu_e, &transpose_vector(kern, row, &third_d[V]), w);
            kern.add_curvature(row, w * (dot(&third_d[V], &je) + kappa * h[V][V]), &mut acc);
            let h_vv = w * h[V][V];
            if h_vv != 0.0 {
                for (j, (&dj, &ej)) in mu_d.iter().zip(&mu_e).enumerate() {
                    for (l, (&dl, &el)) in mu_d.iter().zip(&mu_e).enumerate() {
                        acc[[residual.start + j, residual.start + l]] += h_vv * (dj * el + ej * dl);
                    }
                }
            }
            Ok(acc)
        },
        |a, b| Ok(a + b),
    )
}

/// `{H′[e_a]}_{a=0..p}`: one independent full-data sweep per coefficient axis.
pub(super) fn residual_hessian_directional_derivative_all_axes(
    kern: &(impl ResidualDriveRows + ?Sized + Sync),
) -> Result<Vec<Array2<f64>>, String> {
    let p = kern.n_coefficients();
    (0..p)
        .into_par_iter()
        .map(|a| {
            let mut axis = vec![0.0_f64; p];
            axis[a] = 1.0;
            gam_problem::with_nested_parallel(|| residual_hessian_directional_derivative(kern, &axis))
        })
        .collect()
}

/// `{H″[u, e_a]}_{a=0..p}`: one independent full-data sweep per coefficient axis.
pub(super) fn residual_hessian_second_directional_derivative_all_axes(
    kern: &(impl ResidualDriveRows + ?Sized + Sync),
    d_beta_u: &[f64],
) -> Result<Vec<Array2<f64>>, String> {
    let p = kern.n_coefficients();
    (0..p)
        .into_par_iter()
        .map(|a| {
            let mut axis = vec![0.0_f64; p];
            axis[a] = 1.0;
            gam_problem::with_nested_parallel(|| {
                residual_hessian_second_directional_derivative(kern, d_beta_u, &axis)
            })
        })
        .collect()
}

/// One row's local geometry in the coordinates `(η_m, g, β_1, …, β_K)`: the
/// five-primary channels pulled back through `J̃_i` (identity on the two
/// surfaces, the residual rows on `β`) plus the curvature of `v`. This is what
/// the saved-H ALO replay reads as a row's score and observed Hessian.
pub(super) fn residual_row_geometry(
    kern: &(impl ResidualDriveRows + ?Sized),
    row: usize,
) -> Result<(f64, Array1<f64>, Array2<f64>), String> {
    let (nll, g, h) = kern.row_kernel(row)?;
    let rho = kern.residual_jacobian_rows(row);
    let k = rho[0].len();
    let width = 2 + k;
    let mut score = Array1::<f64>::zeros(width);
    score[0] = g[0];
    score[1] = g[1];
    residual_transpose_action(
        &rho,
        &[g[2], g[3], g[4]],
        &mut score.as_slice_mut().ok_or("residual row geometry: non-contiguous score")?[2..],
    );
    let mut hessian = Array2::<f64>::zeros((width, width));
    hessian[[0, 0]] = h[0][0];
    hessian[[0, 1]] = h[0][1];
    hessian[[1, 0]] = h[1][0];
    hessian[[1, 1]] = h[1][1];
    let [c_q, c_g] = pullback_residual_block(&h, &rho, hessian.slice_mut(s![2.., 2..]));
    for j in 0..k {
        hessian[[0, 2 + j]] = c_q[j];
        hessian[[2 + j, 0]] = c_q[j];
        hessian[[1, 2 + j]] = c_g[j];
        hessian[[2 + j, 1]] = c_g[j];
    }
    // `add_curvature` addresses the kernel's flat coefficient layout; lift the
    // residual block out of it and back.
    let residual = kern.residual_range();
    let mut curvature = Array2::<f64>::zeros((residual.end, residual.end));
    kern.add_curvature(row, g[V], &mut curvature);
    hessian
        .slice_mut(s![2.., 2..])
        .scaled_add(1.0, &curvature.slice(s![residual.clone(), residual]));
    Ok((nll, score, hessian))
}

/// The residual repair kernel over the family's training rows.
pub(super) struct ResidualDriveKernel {
    family: BernoulliMarginalSlopeFamily,
    block_states: Vec<ParameterBlockState>,
    slices: BlockSlices,
    residual: Range<usize>,
    runtime: Arc<ResidualBlockRuntime>,
    /// The drive moments of the pooled law, computed once; `None` when the
    /// joint covariance varies by row.
    pooled: Option<DriveMoments>,
}

impl ResidualDriveKernel {
    pub(super) fn new(
        family: BernoulliMarginalSlopeFamily,
        block_states: Vec<ParameterBlockState>,
    ) -> Result<Self, String> {
        let runtime = family.residual.clone().ok_or_else(|| {
            "residual row kernel constructed on a family without a residual block".to_string()
        })?;
        if family.flex_active() {
            return Err(super::residual_repair::ResidualRepairRefusal::FlexBlocksUnsupported.to_string());
        }
        family.validate_exact_block_state_shapes(&block_states)?;
        let slices = super::hessian_paths::block_slices(&family);
        let residual = slices
            .residual
            .clone()
            .ok_or("residual row kernel: block slices carry no residual range")?;
        if residual.len() != runtime.width() || block_states[2].beta.len() != runtime.width() {
            return Err(format!(
                "residual row kernel width mismatch: range {}, coefficients {}, features {}",
                residual.len(),
                block_states[2].beta.len(),
                runtime.width()
            ));
        }
        let pooled = if runtime.field.is_conditional() || family.y.is_empty() {
            None
        } else {
            let beta = block_states[2]
                .beta
                .as_slice()
                .ok_or("residual beta not contiguous")?;
            Some(DriveMoments::at(runtime.field.at_row(0), beta))
        };
        Ok(Self {
            family,
            block_states,
            slices,
            residual,
            runtime,
            pooled,
        })
    }

    #[inline]
    fn beta(&self) -> &[f64] {
        self.block_states[2]
            .beta
            .as_slice()
            .expect("residual beta is validated contiguous at construction")
    }

    #[inline]
    fn moments(&self, row: usize) -> Cow<'_, DriveMoments> {
        match &self.pooled {
            Some(moments) => Cow::Borrowed(moments),
            None => Cow::Owned(DriveMoments::at(self.runtime.field.at_row(row), self.beta())),
        }
    }
}

impl RowProgram<5> for ResidualDriveKernel {
    fn n_rows(&self) -> usize {
        self.family.y.len()
    }

    fn primaries(&self, row: usize) -> Result<[f64; 5], String> {
        if row >= self.family.y.len() {
            return Err(format!("ResidualDriveKernel: row {row} out of range"));
        }
        let features = self.runtime.features.row(row);
        let t = features
            .iter()
            .zip(self.beta())
            .map(|(r, b)| r * b)
            .sum::<f64>();
        let moments = self.moments(row);
        Ok([
            self.block_states[0].eta[row],
            self.block_states[1].eta[row],
            t,
            moments.u,
            moments.v,
        ])
    }

    fn eval<S: JetScalar<5>>(&self, row: usize, p: &[S; 5]) -> Result<S, String> {
        if row >= self.family.y.len() {
            return Err(format!("ResidualDriveKernel: row {row} out of range"));
        }
        let marginal = self
            .family
            .marginal_link_map(self.block_states[0].eta[row])?;
        let grid = self
            .family
            .latent_measure
            .empirical_grid_for_training_row(row)?;
        let state = ResidualRowState {
            marginal,
            z: self.family.z[row],
            y: self.family.y[row],
            w: self.family.weights[row],
            probit_scale: self.family.probit_frailty_scale(),
            grid: grid.as_deref(),
        };
        let drive = ResidualDrive {
            t: p[2],
            u: p[3],
            v: p[V],
        };
        residual_row_nll(&state, &p[0], &p[1], &drive).map_err(|e| format!("row {row}: {e}"))
    }
}

impl RowKernel<5> for ResidualDriveKernel {
    fn n_coefficients(&self) -> usize {
        self.slices.total
    }

    fn jacobian_action(&self, row: usize, d_beta: &[f64]) -> [f64; 5] {
        let d_view = ndarray::ArrayView1::from(d_beta);
        let rho = self.residual_jacobian_rows(row);
        let [t, u, v] = residual_action(&rho, &d_beta[self.residual.clone()]);
        [
            self.family
                .marginal_design
                .dot_row_view(row, d_view.slice(s![self.slices.marginal.clone()])),
            self.family
                .slope_design
                .dot_row_view(row, d_view.slice(s![self.slices.slope.clone()])),
            t,
            u,
            v,
        ]
    }

    fn jacobian_transpose_action(&self, row: usize, v: &[f64; 5], out: &mut [f64]) {
        {
            let mut m = ndarray::ArrayViewMut1::from(&mut out[self.slices.marginal.clone()]);
            self.family
                .marginal_design
                .axpy_row_into(row, v[0], &mut m)
                .expect("marginal axpy dim mismatch");
        }
        {
            let mut g = ndarray::ArrayViewMut1::from(&mut out[self.slices.slope.clone()]);
            self.family
                .slope_design
                .axpy_row_into(row, v[1], &mut g)
                .expect("slope axpy dim mismatch");
        }
        let rho = self.residual_jacobian_rows(row);
        residual_transpose_action(&rho, &[v[2], v[3], v[4]], &mut out[self.residual.clone()]);
    }

    fn add_pullback_hessian(&self, row: usize, h: &[[f64; 5]; 5], target: &mut Array2<f64>) {
        let marginal = self.slices.marginal.clone();
        let slope = self.slices.slope.clone();
        let residual = self.residual.clone();
        self.family
            .marginal_design
            .syr_row_into_view(row, h[0][0], target.slice_mut(s![marginal.clone(), marginal.clone()]))
            .expect("marginal syr dim mismatch");
        if h[0][1] != 0.0 {
            self.family
                .marginal_design
                .row_outer_into_view(
                    row,
                    &self.family.slope_design,
                    h[0][1],
                    target.slice_mut(s![marginal.clone(), slope.clone()]),
                )
                .expect("marginal-slope outer dim mismatch");
            self.family
                .slope_design
                .row_outer_into_view(
                    row,
                    &self.family.marginal_design,
                    h[0][1],
                    target.slice_mut(s![slope.clone(), marginal.clone()]),
                )
                .expect("slope-marginal outer dim mismatch");
        }
        self.family
            .slope_design
            .syr_row_into_view(row, h[1][1], target.slice_mut(s![slope.clone(), slope.clone()]))
            .expect("slope syr dim mismatch");
        let rho = self.residual_jacobian_rows(row);
        let [c_q, c_g] =
            pullback_residual_block(h, &rho, target.slice_mut(s![residual.clone(), residual.clone()]));
        for (j, (&q, &g)) in c_q.iter().zip(&c_g).enumerate() {
            let col = residual.start + j;
            if q != 0.0 {
                self.family
                    .marginal_design
                    .axpy_row_into(row, q, &mut target.slice_mut(s![marginal.clone(), col]))
                    .expect("marginal-residual axpy dim mismatch");
                self.family
                    .marginal_design
                    .axpy_row_into(row, q, &mut target.slice_mut(s![col, marginal.clone()]))
                    .expect("residual-marginal axpy dim mismatch");
            }
            if g != 0.0 {
                self.family
                    .slope_design
                    .axpy_row_into(row, g, &mut target.slice_mut(s![slope.clone(), col]))
                    .expect("slope-residual axpy dim mismatch");
                self.family
                    .slope_design
                    .axpy_row_into(row, g, &mut target.slice_mut(s![col, slope.clone()]))
                    .expect("residual-slope axpy dim mismatch");
            }
        }
    }

    fn add_diagonal_quadratic(&self, row: usize, h: &[[f64; 5]; 5], diag: &mut [f64]) {
        {
            let mut md = ndarray::ArrayViewMut1::from(&mut diag[self.slices.marginal.clone()]);
            self.family
                .marginal_design
                .squared_axpy_row_into(row, h[0][0], &mut md)
                .expect("marginal squared_axpy dim mismatch");
        }
        {
            let mut gd = ndarray::ArrayViewMut1::from(&mut diag[self.slices.slope.clone()]);
            self.family
                .slope_design
                .squared_axpy_row_into(row, h[1][1], &mut gd)
                .expect("slope squared_axpy dim mismatch");
        }
        let rho = self.residual_jacobian_rows(row);
        for (j, out) in diag[self.residual.clone()].iter_mut().enumerate() {
            for a in 0..3 {
                for b in 0..3 {
                    *out += h[2 + a][2 + b] * rho[a][j] * rho[b][j];
                }
            }
        }
    }
}

impl ResidualDriveRows for ResidualDriveKernel {
    fn residual_range(&self) -> Range<usize> {
        self.residual.clone()
    }

    fn residual_jacobian_rows(&self, row: usize) -> [Vec<f64>; 3] {
        let moments = self.moments(row);
        [
            self.runtime.features.row(row).to_vec(),
            moments.gamma.clone(),
            moments.sigma_beta.iter().map(|x| 2.0 * x).collect(),
        ]
    }

    fn curvature_action(&self, row: usize, d_r: &[f64]) -> Vec<f64> {
        let covariance = self.runtime.field.at_row(row);
        let mut lifted = vec![0.0_f64; d_r.len() + 1];
        lifted[1..].copy_from_slice(d_r);
        let mut image = vec![0.0_f64; d_r.len() + 1];
        covariance.multiply(&lifted, &mut image);
        image[1..].iter().map(|x| 2.0 * x).collect()
    }

    fn add_curvature(&self, row: usize, scale: f64, target: &mut Array2<f64>) {
        if scale == 0.0 {
            return;
        }
        let covariance = self.runtime.field.at_row(row);
        let residual = self.residual.clone();
        for j in 0..residual.len() {
            for l in 0..residual.len() {
                target[[residual.start + j, residual.start + l]] +=
                    2.0 * scale * covariance.coefficient(j + 1, l + 1);
            }
        }
    }
}

#[cfg(test)]
mod residual_drive_kernel_tests {
    //! The five-primary kernel against the coefficient-primary row program it
    //! replaces: the same row statement lowered with `(η_m, g, β_1, …, β_K)` as
    //! the jet primaries (exact by construction, `O(K⁴)` per row) must agree
    //! with the chain-rule assembly channel for channel; at widths past any
    //! ladder the assembly is gated by finite differences.

    use super::super::residual_repair::declare_unit_score_variance;
    use super::super::tests_residual_repair_laws::{ResidualBlock, covariance, skewed_grid};
    use super::*;

    struct RowFixture {
        y: f64,
        w: f64,
        z: f64,
        s: f64,
        r: Vec<f64>,
        cov: MarginalSlopeCovariance,
        grid: Option<EmpiricalZGrid>,
    }

    impl RowFixture {
        fn new(k: usize, law: Option<EmpiricalZGrid>, seed: u64) -> Self {
            let cov = MarginalSlopeCovariance::full(
                declare_unit_score_variance(&covariance(k, seed).to_dense()).unwrap(),
            )
            .unwrap();
            Self {
                y: 1.0,
                w: 1.2,
                z: 0.7,
                s: 0.9,
                r: (0..k).map(|j| ((j as f64) * 0.37 + 0.2).sin()).collect(),
                cov,
                grid: law,
            }
        }

        fn state(&self, eta_m: f64) -> ResidualRowState<'_> {
            let link = InverseLink::Standard(StandardLink::Probit);
            ResidualRowState {
                marginal: bernoulli_marginal_link_map(&link, eta_m).unwrap(),
                z: self.z,
                y: self.y,
                w: self.w,
                probit_scale: self.s,
                grid: self.grid.as_ref(),
            }
        }
    }

    /// The five-primary kernel on one row with unit surface designs:
    /// `θ = (η_m, g, β)`.
    struct DriveRow<'a> {
        fixture: &'a RowFixture,
        theta: Vec<f64>,
    }

    impl DriveRow<'_> {
        fn moments(&self) -> DriveMoments {
            DriveMoments::at(&self.fixture.cov, &self.theta[2..])
        }
    }

    impl RowProgram<5> for DriveRow<'_> {
        fn n_rows(&self) -> usize {
            1
        }
        fn primaries(&self, row: usize) -> Result<[f64; 5], String> {
            assert_eq!(row, 0, "one-row fixture");
            let m = self.moments();
            Ok([
                self.theta[0],
                self.theta[1],
                dot(&self.fixture.r, &self.theta[2..]),
                m.u,
                m.v,
            ])
        }
        fn eval<S: JetScalar<5>>(&self, row: usize, p: &[S; 5]) -> Result<S, String> {
            assert_eq!(row, 0, "one-row fixture");
            let state = self.fixture.state(self.theta[0]);
            let drive = ResidualDrive {
                t: p[2],
                u: p[3],
                v: p[4],
            };
            residual_row_nll(&state, &p[0], &p[1], &drive)
        }
    }

    impl RowKernel<5> for DriveRow<'_> {
        fn n_coefficients(&self) -> usize {
            self.theta.len()
        }
        fn jacobian_action(&self, row: usize, d: &[f64]) -> [f64; 5] {
            let [t, u, v] = residual_action(&self.residual_jacobian_rows(row), &d[2..]);
            [d[0], d[1], t, u, v]
        }
        fn jacobian_transpose_action(&self, row: usize, v: &[f64; 5], out: &mut [f64]) {
            out[0] += v[0];
            out[1] += v[1];
            residual_transpose_action(&self.residual_jacobian_rows(row), &[v[2], v[3], v[4]], &mut out[2..]);
        }
        fn add_pullback_hessian(&self, row: usize, h: &[[f64; 5]; 5], target: &mut Array2<f64>) {
            for a in 0..2 {
                for b in 0..2 {
                    target[[a, b]] += h[a][b];
                }
            }
            let rho = self.residual_jacobian_rows(row);
            let [c_q, c_g] = pullback_residual_block(h, &rho, target.slice_mut(s![2.., 2..]));
            for j in 0..c_q.len() {
                target[[0, 2 + j]] += c_q[j];
                target[[2 + j, 0]] += c_q[j];
                target[[1, 2 + j]] += c_g[j];
                target[[2 + j, 1]] += c_g[j];
            }
        }
        fn add_diagonal_quadratic(&self, row: usize, h: &[[f64; 5]; 5], diag: &mut [f64]) {
            let mut dense = Array2::<f64>::zeros((self.theta.len(), self.theta.len()));
            self.add_pullback_hessian(row, h, &mut dense);
            for (j, out) in diag.iter_mut().enumerate() {
                *out += dense[[j, j]];
            }
        }
    }

    impl ResidualDriveRows for DriveRow<'_> {
        fn residual_range(&self) -> Range<usize> {
            2..self.theta.len()
        }
        fn residual_jacobian_rows(&self, row: usize) -> [Vec<f64>; 3] {
            assert_eq!(row, 0, "one-row fixture");
            let m = self.moments();
            [
                self.fixture.r.clone(),
                m.gamma,
                m.sigma_beta.iter().map(|x| 2.0 * x).collect(),
            ]
        }
        fn curvature_action(&self, row: usize, d_r: &[f64]) -> Vec<f64> {
            assert_eq!(row, 0, "one-row fixture");
            let mut out = vec![0.0; d_r.len()];
            ResidualBlock(&self.fixture.cov).multiply(d_r, &mut out);
            out.iter().map(|x| 2.0 * x).collect()
        }
        fn add_curvature(&self, row: usize, scale: f64, target: &mut Array2<f64>) {
            assert_eq!(row, 0, "one-row fixture");
            let k = self.theta.len() - 2;
            for j in 0..k {
                for l in 0..k {
                    target[[2 + j, 2 + l]] += 2.0 * scale * self.fixture.cov.coefficient(j + 1, l + 1);
                }
            }
        }
    }

    /// The same row with `(η_m, g, β_1, …, β_K)` as the jet primaries.
    struct CoefficientRow<'a, const P: usize> {
        fixture: &'a RowFixture,
        theta: [f64; P],
    }

    impl<const P: usize> RowProgram<P> for CoefficientRow<'_, P> {
        fn n_rows(&self) -> usize {
            1
        }
        fn primaries(&self, row: usize) -> Result<[f64; P], String> {
            assert_eq!(row, 0, "one-row fixture");
            Ok(self.theta)
        }
        fn eval<S: JetScalar<P>>(&self, row: usize, p: &[S; P]) -> Result<S, String> {
            assert_eq!(row, 0, "one-row fixture");
            let state = self.fixture.state(self.theta[0]);
            let mut gamma = vec![0.0; P - 2];
            for (j, entry) in gamma.iter_mut().enumerate() {
                *entry = self.fixture.cov.coefficient(0, j + 1);
            }
            let beta = &p[2..];
            let drive = ResidualDrive {
                t: S::linear_combination(beta, &self.fixture.r),
                u: S::linear_combination(beta, &gamma),
                v: S::symmetric_quadratic_form(beta, &ResidualBlock(&self.fixture.cov)),
            };
            residual_row_nll(&state, &p[0], &p[1], &drive)
        }
    }

    fn theta(k: usize) -> Vec<f64> {
        let mut theta = vec![-0.45, 0.55];
        theta.extend((0..k).map(|j| 0.3 * ((j as f64) * 0.71 + 0.4).cos() / (k as f64).sqrt()));
        theta
    }

    fn direction(p: usize, phase: f64) -> Vec<f64> {
        (0..p).map(|j| ((j as f64) * 0.53 + phase).sin()).collect()
    }

    fn close(what: &str, exact: f64, reference: f64, tolerance: f64) {
        assert!(
            (exact - reference).abs() <= tolerance * (1.0 + reference.abs()),
            "{what}: assembled {exact} vs reference {reference}"
        );
    }

    fn cache_of(row: &DriveRow<'_>) -> RowKernelCache<5> {
        crate::row_kernel::build_row_kernel_cache(row, &RowSet::All).unwrap()
    }

    fn equivalence<const P: usize>(law: Option<EmpiricalZGrid>) {
        let k = P - 2;
        let fixture = RowFixture::new(k, law, 17 + k as u64);
        let base = theta(k);
        let drive = DriveRow {
            fixture: &fixture,
            theta: base.clone(),
        };
        let reference = CoefficientRow::<P> {
            fixture: &fixture,
            theta: std::array::from_fn(|a| base[a]),
        };
        let label = if fixture.grid.is_some() { "declared law" } else { "normal law" };
        let (nll_ref, grad_ref, hess_ref) = gam_math::jet_tower::program_row_kernel(&reference, 0).unwrap();
        let cache = cache_of(&drive);
        close(&format!("{label} K={k} value"), cache.nll[0], nll_ref, 1e-12);
        let grad = transpose_vector(&drive, 0, &cache.gradients[0]);
        let hess = residual_hessian_dense(&drive, &cache).unwrap();
        let d = direction(P, 0.3);
        let e = direction(P, 1.9);
        let third = residual_hessian_directional_derivative(&drive, &d).unwrap();
        let fourth = residual_hessian_second_directional_derivative(&drive, &d, &e).unwrap();
        let d_arr: [f64; P] = std::array::from_fn(|a| d[a]);
        let e_arr: [f64; P] = std::array::from_fn(|a| e[a]);
        let third_ref = gam_math::jet_tower::program_third_contracted(&reference, 0, &d_arr).unwrap();
        let fourth_ref =
            gam_math::jet_tower::program_fourth_contracted(&reference, 0, &d_arr, &e_arr).unwrap();
        let (_, score, observed) = residual_row_geometry(&drive, 0).unwrap();
        for a in 0..P {
            close(&format!("{label} K={k} grad[{a}]"), grad[a], grad_ref[a], 1e-11);
            close(&format!("{label} K={k} ALO score[{a}]"), score[a], grad_ref[a], 1e-11);
            for b in 0..P {
                close(&format!("{label} K={k} hess[{a}][{b}]"), hess[[a, b]], hess_ref[a][b], 1e-10);
                close(&format!("{label} K={k} ALO hess[{a}][{b}]"), observed[[a, b]], hess_ref[a][b], 1e-10);
                close(&format!("{label} K={k} H'[{a}][{b}]"), third[[a, b]], third_ref[a][b], 1e-9);
                close(&format!("{label} K={k} H''[{a}][{b}]"), fourth[[a, b]], fourth_ref[a][b], 1e-9);
            }
        }
    }

    #[test]
    fn drive_kernel_equals_the_coefficient_primary_program_through_order_four() {
        for law in [None, Some(skewed_grid())] {
            equivalence::<3>(law.clone());
            equivalence::<5>(law.clone());
            equivalence::<14>(law);
        }
    }

    fn finite_difference_gate(k: usize, law: Option<EmpiricalZGrid>) {
        let fixture = RowFixture::new(k, law, 101 + k as u64);
        let base = theta(k);
        let p = base.len();
        let d = direction(p, 0.8);
        let e = direction(p, 2.6);
        let at = |theta: &[f64]| DriveRow {
            fixture: &fixture,
            theta: theta.to_vec(),
        };
        let displaced = |dir: &[f64], scale: f64| -> Vec<f64> {
            base.iter().zip(dir).map(|(x, v)| x + scale * v).collect()
        };
        let gradient = |theta: &[f64]| {
            let row = at(theta);
            let cache = cache_of(&row);
            transpose_vector(&row, 0, &cache.gradients[0])
        };
        let hessian = |theta: &[f64]| {
            let row = at(theta);
            residual_hessian_dense(&row, &cache_of(&row)).unwrap()
        };
        let h = 1.0e-5;
        let label = if fixture.grid.is_some() { "declared law" } else { "normal law" };
        let hess = hessian(&base);
        let third = residual_hessian_directional_derivative(&at(&base), &d).unwrap();
        let fourth = residual_hessian_second_directional_derivative(&at(&base), &d, &e).unwrap();
        let (gp, gm) = (gradient(&displaced(&d, h)), gradient(&displaced(&d, -h)));
        let (hp, hm) = (hessian(&displaced(&d, h)), hessian(&displaced(&d, -h)));
        let tp = residual_hessian_directional_derivative(&at(&displaced(&e, h)), &d).unwrap();
        let tm = residual_hessian_directional_derivative(&at(&displaced(&e, -h)), &d).unwrap();
        for a in 0..p {
            let hd: f64 = (0..p).map(|b| hess[[a, b]] * d[b]).sum();
            close(&format!("{label} K={k} H·d[{a}]"), hd, (gp[a] - gm[a]) / (2.0 * h), 2e-6);
            for b in 0..p {
                close(
                    &format!("{label} K={k} H'[{a}][{b}]"),
                    third[[a, b]],
                    (hp[[a, b]] - hm[[a, b]]) / (2.0 * h),
                    2e-6,
                );
                close(
                    &format!("{label} K={k} H''[{a}][{b}]"),
                    fourth[[a, b]],
                    (tp[[a, b]] - tm[[a, b]]) / (2.0 * h),
                    2e-6,
                );
            }
        }
    }

    #[test]
    fn drive_kernel_is_finite_difference_tight_at_block_widths_26_and_32() {
        for law in [None, Some(skewed_grid())] {
            finite_difference_gate(26, law.clone());
            finite_difference_gate(32, law);
        }
    }
}
