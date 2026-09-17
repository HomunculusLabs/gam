//! The metric a survival time margin tensors a covariate block against
//! (gam#2765, gam#2767; SPEC rule 5).
//!
//! A time-varying covariate block represents
//! `f(x, u) = Σ_{j,k} β_{jk} X_j(x) B_k(u)` with `u = log t`, and its penalties
//! must be functionals of `f`, not of the coefficient chart. With `G_x = XᵀX / n`
//! the covariate functions' Gram over the training rows (the measure a linear
//! term's null ridge already uses, `linear_function_mass`) and
//! `Ḡ_t = ∫_D B Bᵀ du / |D|` the margin's mean Gram over its modeling interval:
//!
//! ```text
//!   (1/n) Σ_i ∫_D (∂ᵐ_u f(x_i, u))² du   = βᵀ (G_x ⊗ S_t) β,
//!   (1/|D|) ∫_D J_x[f(·, u)] du          = βᵀ (S_x ⊗ Ḡ_t) β,
//! ```
//!
//! with `S_t = ∫_D B⁽ᵐ⁾ B⁽ᵐ⁾ᵀ du` and `J_x[g] = γᵀ S_x γ` the covariate term's
//! own functional. `S ⊗ I` agrees with these only where the other factor's Gram
//! is the identity in the chart at hand: re-centring a covariate column moves it
//! and leaves these unchanged. On a time-constant function `β = γ ⊗ 1`,
//! `1ᵀ Ḡ_t 1 = 1` returns the static block's `γᵀ S_x γ`, and an intercept-only
//! covariate block has `G_x = [1]`.
//!
//! `G_x` is assembled by the design's own `XᵀWX` kernel, so it keeps the design's
//! structure (a many-level random effect contributes its count Gram) and no
//! `n × p` covariate matrix is formed. `G_x` moves with a spatial length scale on
//! the covariate block, so the margin penalties' ψ-derivatives are published here
//! beside the covariate penalties'.

use crate::spatial_psi_bridge::{OwnedPenaltyPsiComponents, SpatialPsiAxisDesign, SpatialPsiBlockTransform};
use crate::survival::location_scale::SurvivalCovariateTermBlockTemplate;
use gam_linalg::matrix::{DesignMatrix, LinearOperator, symmetrize_in_place};
use gam_problem::penalty_matrix::kronecker_product;
use ndarray::{Array1, Array2};

/// `Ḡ_t`: the margin's exact L² Gram over its modeling interval divided by the
/// interval's length. `Σ Bᵢ = 1` there, so `1ᵀ G 1 = |D|`.
pub(crate) fn time_margin_mean_gram(
    knots: &Array1<f64>,
    degree: usize,
    block_name: &str,
) -> Result<Array2<f64>, String> {
    let gram = gam_terms::basis::bspline_function_gram(knots, degree)
        .map_err(|e| format!("{block_name} time-margin function Gram: {e}"))?;
    let length = gram.sum();
    if !(length.is_finite() && length > 0.0) {
        return Err(format!(
            "{block_name} time-margin modeling interval has non-positive length {length}"
        ));
    }
    Ok(gram.mapv(|value| value / length))
}

/// `G_x = XᵀX / n` through the design's own `XᵀWX` kernel.
fn covariate_mean_square_gram(
    design: &DesignMatrix,
    block_name: &str,
) -> Result<Array2<f64>, String> {
    let n = design.nrows();
    if n == 0 {
        return Err(format!("{block_name} covariate Gram needs at least one row"));
    }
    let mut gram = design
        .diag_xtw_x(&Array1::<f64>::ones(n))
        .map_err(|e| format!("{block_name} covariate Gram: {e}"))?;
    gram.mapv_inplace(|value| value / n as f64);
    symmetrize_in_place(&mut gram);
    if gram.iter().any(|value| !value.is_finite()) {
        return Err(format!("{block_name} covariate Gram is not finite"));
    }
    Ok(gram)
}

/// Null dimension of a symmetric PSD penalty against the relative eigenvalue
/// floor the penalty spectrum machinery uses elsewhere.
fn symmetric_nullspace_dimension(matrix: &Array2<f64>) -> Result<usize, String> {
    use faer::Side;
    use gam_linalg::faer_ndarray::FaerEigh;
    let (eigenvalues, _) = matrix
        .eigh(Side::Lower)
        .map_err(|error| format!("time-margin penalty eigendecomposition: {error}"))?;
    let largest = eigenvalues.iter().fold(0.0_f64, |acc, value| acc.max(*value));
    if largest <= 0.0 {
        return Ok(matrix.nrows());
    }
    let floor = largest * (matrix.nrows() as f64) * f64::EPSILON;
    Ok(eigenvalues.iter().filter(|value| **value <= floor).count())
}

/// Null dimension of `A ⊗ B` for symmetric PSD factors: the spectrum is the
/// products of the factors' spectra, so `p_a·p_b − rank(A)·rank(B)`.
pub(crate) fn kronecker_nullspace_dimension(
    left: &Array2<f64>,
    right: &Array2<f64>,
) -> Result<usize, String> {
    let left_rank = left.nrows() - symmetric_nullspace_dimension(left)?;
    let right_rank = right.nrows() - symmetric_nullspace_dimension(right)?;
    Ok(left.nrows() * right.nrows() - left_rank * right_rank)
}

/// The two Grams a time-varying covariate block's penalties are tensored
/// against, and the margin penalties whose covariate factor moves with ψ.
pub(crate) struct TimeMarginPenaltyMetric {
    /// `Ḡ_t`, the time factor of every covariate penalty.
    pub(crate) time_gram: Array2<f64>,
    /// `G_x`, the covariate factor of every margin penalty.
    pub(crate) covariate_gram: Array2<f64>,
    /// The margin penalties `S_t`.
    pub(crate) time_penalties: Vec<Array2<f64>>,
    /// The covariate design `G_x` is formed from, for its ψ-derivatives.
    pub(crate) covariate_design: DesignMatrix,
    /// Block index of the first margin penalty: the covariate penalties precede it.
    pub(crate) first_time_penalty: usize,
}

impl TimeMarginPenaltyMetric {
    /// The metric of a time-varying template over this covariate design, or
    /// `None` for a static template.
    pub(crate) fn from_template(
        template: &SurvivalCovariateTermBlockTemplate,
        covariate_design: &DesignMatrix,
        covariate_penalty_count: usize,
        block_name: &str,
    ) -> Result<Option<Self>, String> {
        let SurvivalCovariateTermBlockTemplate::TimeVarying {
            time_penalties,
            time_gram,
            ..
        } = template
        else {
            return Ok(None);
        };
        Ok(Some(Self {
            time_gram: time_gram.clone(),
            covariate_gram: covariate_mean_square_gram(covariate_design, block_name)?,
            time_penalties: time_penalties.clone(),
            covariate_design: covariate_design.clone(),
            first_time_penalty: covariate_penalty_count,
        }))
    }

    /// `S_x ⊗ Ḡ_t` for a covariate penalty over the covariate block's columns.
    pub(crate) fn covariate_penalty(&self, penalty: &Array2<f64>) -> Array2<f64> {
        kronecker_product(penalty, &self.time_gram)
    }

    /// `G_x ⊗ S_t` for every margin penalty, in margin order.
    pub(crate) fn margin_penalties(&self) -> Vec<Array2<f64>> {
        self.time_penalties
            .iter()
            .map(|penalty| kronecker_product(&self.covariate_gram, penalty))
            .collect()
    }
}

impl SpatialPsiBlockTransform for TimeMarginPenaltyMetric {
    fn transform_penalty(&self, penalty: Array2<f64>) -> Array2<f64> {
        self.covariate_penalty(&penalty)
    }

    fn owned_penalty_psi_components(
        &self,
        axes: &[SpatialPsiAxisDesign],
    ) -> Result<OwnedPenaltyPsiComponents, String> {
        time_penalty_psi_components(
            &self.covariate_design,
            axes,
            &self.time_penalties,
            self.first_time_penalty,
        )
    }
}

/// `∂G_x/∂ψ_a ⊗ S_t` and `∂²G_x/∂ψ_a∂ψ_b ⊗ S_t` for every margin penalty, at
/// block index `first_time_penalty + k`. `G_x = XᵀX / n` moves along a spatial
/// axis through its design derivative only:
///
/// ```text
///   ∂_a G_x     = (X_aᵀX + XᵀX_a) / n
///   ∂_a∂_b G_x  = (X_abᵀX + X_aᵀX_b + X_bᵀX_a + XᵀX_ab) / n
/// ```
///
/// A design derivative is nonzero only on the columns its term owns, so row `i`
/// of `X_aᵀX` is `Xᵀ(X_a e_i)` for `i` in the axis's range and zero elsewhere.
/// `X_ab` is nonzero only between axes of one term (`a = b`, or one anisotropy
/// group); `X_aᵀX_b` couples every pair of moving axes, because both reach the
/// same Gram. Everything is formed by operator and design matvecs over the moving
/// columns, so no `n × p` matrix is materialized and an axis-free block costs
/// nothing.
pub(crate) fn time_penalty_psi_components(
    covariate_design: &DesignMatrix,
    axes: &[SpatialPsiAxisDesign],
    time_penalties: &[Array2<f64>],
    first_time_penalty: usize,
) -> Result<OwnedPenaltyPsiComponents, String> {
    let n = covariate_design.nrows();
    let p = covariate_design.ncols();
    if n == 0 {
        return Err("time-margin covariate Gram derivative needs at least one row".to_string());
    }
    let scale = 1.0 / n as f64;
    let psi_dim = axes.len();
    let lift = |gram_derivative: &Array2<f64>| -> Vec<(usize, Array2<f64>)> {
        time_penalties
            .iter()
            .enumerate()
            .map(|(k, penalty)| {
                (
                    first_time_penalty + k,
                    kronecker_product(gram_derivative, penalty),
                )
            })
            .collect()
    };
    let unit = |j: usize| {
        let mut vector = Array1::<f64>::zeros(p);
        vector[j] = 1.0;
        vector
    };
    let transpose_rows = |columns: &[(usize, Array1<f64>)]| -> Result<Array2<f64>, String> {
        let mut out = Array2::<f64>::zeros((p, p));
        for (i, column) in columns {
            if column.len() != n {
                return Err(format!(
                    "time-margin covariate Gram derivative: operator returned {} rows for a \
                     {n}-row covariate block",
                    column.len()
                ));
            }
            out.row_mut(*i)
                .assign(&covariate_design.transpose_vector_multiply(column));
        }
        Ok(out)
    };

    let mut forward: Vec<Vec<(usize, Array1<f64>)>> = Vec::with_capacity(psi_dim);
    for axis in axes {
        let mut columns = Vec::with_capacity(axis.range.len());
        if let Some(operator) = axis.operator.as_ref() {
            for i in axis.range.clone() {
                let column = operator
                    .forward_mul(axis.axis, &unit(i).view())
                    .map_err(|e| e.to_string())?;
                columns.push((i, column));
            }
        }
        forward.push(columns);
    }

    let mut first = Vec::with_capacity(psi_dim);
    for (a, axis) in axes.iter().enumerate() {
        if axis.operator.is_none() {
            first.push(Vec::new());
            continue;
        }
        let moved = transpose_rows(&forward[a])?;
        let derivative = (&moved + &moved.t()).mapv(|value| value * scale);
        first.push(lift(&derivative));
    }

    let mut second = vec![vec![Vec::new(); psi_dim]; psi_dim];
    for a in 0..psi_dim {
        let Some(operator) = axes[a].operator.as_ref() else {
            continue;
        };
        for b in 0..psi_dim {
            if axes[b].operator.is_none() {
                continue;
            }
            let mut cross = Array2::<f64>::zeros((p, p));
            for (i, left) in &forward[a] {
                for (j, right) in &forward[b] {
                    cross[[*i, *j]] = left.dot(right);
                }
            }
            let mut derivative = &cross + &cross.t();
            let one_term = a == b || (axes[a].group.is_some() && axes[a].group == axes[b].group);
            if one_term {
                let mut curved_columns = Vec::with_capacity(axes[a].range.len());
                for i in axes[a].range.clone() {
                    let column = if a == b {
                        operator.forward_mul_second_diag(axes[a].axis, &unit(i).view())
                    } else {
                        operator.forward_mul_second_cross(axes[a].axis, axes[b].axis, &unit(i).view())
                    }
                    .map_err(|e| e.to_string())?;
                    curved_columns.push((i, column));
                }
                let curved = transpose_rows(&curved_columns)?;
                derivative += &curved;
                derivative += &curved.t();
            }
            derivative.mapv_inplace(|value| value * scale);
            second[a][b] = lift(&derivative);
        }
    }
    Ok(OwnedPenaltyPsiComponents { first, second })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::custom_family::build_embedded_dense_psi_operator;
    use crate::survival::construction::build_time_varying_survival_covariate_template;
    use gam_linalg::matrix::{BlockDesignOperator, DenseDesignMatrix, DesignBlock, RandomEffectOperator};
    use std::sync::Arc;

    fn fixture_times() -> (Array1<f64>, Array1<f64>) {
        let age_exit = Array1::from_iter((1..=40).map(|i| 0.25 + 0.35 * f64::from(i)));
        let age_entry = age_exit.mapv(|t| (t - 0.2).max(1e-3));
        (age_entry, age_exit)
    }

    fn quadratic_form(matrix: &Array2<f64>, beta: &Array1<f64>) -> f64 {
        beta.dot(&matrix.dot(beta))
    }

    /// SPEC rule 5: the margin penalties are functionals of the fitted surface.
    /// Re-centring the covariate column (`x → x + c`, the chart change
    /// `X → X M`, `β → (M⁻¹ ⊗ I) β`) leaves the function and every penalty value
    /// unchanged. The coefficient penalty `I ⊗ S_t` is the positive control: it
    /// moves under the same re-centring.
    #[test]
    fn margin_penalties_do_not_depend_on_the_covariate_chart_2765() {
        let (age_entry, age_exit) = fixture_times();
        let template =
            build_time_varying_survival_covariate_template(&age_entry, &age_exit, 5, 3, "slope")
                .expect("margin template");
        let SurvivalCovariateTermBlockTemplate::TimeVarying { time_penalties, .. } = &template
        else {
            panic!("a time-varying request must produce a time-varying template");
        };
        let n = age_exit.len();
        let p_time = time_penalties[0].nrows();
        let x = Array2::<f64>::from_shape_fn((n, 2), |(row, col)| {
            if col == 0 { 1.0 } else { 0.1 * (row as f64) - 1.3 }
        });
        let shift = 2.5;
        let m = ndarray::array![[1.0, shift], [0.0, 1.0]];
        let m_inverse = ndarray::array![[1.0, -shift], [0.0, 1.0]];
        let x_shifted = x.dot(&m);
        let metric = TimeMarginPenaltyMetric::from_template(
            &template,
            &DesignMatrix::from(x.clone()),
            0,
            "slope",
        )
        .expect("metric")
        .expect("time-varying metric");
        let metric_shifted = TimeMarginPenaltyMetric::from_template(
            &template,
            &DesignMatrix::from(x_shifted),
            0,
            "slope",
        )
        .expect("shifted metric")
        .expect("time-varying shifted metric");
        let beta = Array1::from_iter((0..2 * p_time).map(|i| (0.7 * i as f64).sin() + 0.2));
        let beta_shifted =
            kronecker_product(&m_inverse, &Array2::<f64>::eye(p_time)).dot(&beta);
        for (penalty, shifted) in metric
            .margin_penalties()
            .iter()
            .zip(metric_shifted.margin_penalties().iter())
        {
            let value = quadratic_form(penalty, &beta);
            let value_shifted = quadratic_form(shifted, &beta_shifted);
            assert!(
                (value - value_shifted).abs() <= 1e-12 * value.abs().max(1.0),
                "a margin penalty must not move under a covariate re-centring: \
                 {value:e} against {value_shifted:e}"
            );
        }
        let identity = Array2::<f64>::eye(2);
        let coefficient = kronecker_product(&identity, &time_penalties[0]);
        let control = quadratic_form(&coefficient, &beta);
        let control_shifted = quadratic_form(&coefficient, &beta_shifted);
        assert!(
            (control - control_shifted).abs() > 1e-6 * control.abs().max(1.0),
            "positive control: the coefficient penalty must move under the re-centring; \
             {control:e} against {control_shifted:e}"
        );
    }

    /// A time-constant covariate function `β = γ ⊗ 1` gets exactly the static
    /// block's covariate penalty `γᵀ S_x γ`.
    #[test]
    fn a_time_constant_function_keeps_the_static_covariate_penalty_2765() {
        let (age_entry, age_exit) = fixture_times();
        let template =
            build_time_varying_survival_covariate_template(&age_entry, &age_exit, 6, 2, "slope")
                .expect("margin template");
        let SurvivalCovariateTermBlockTemplate::TimeVarying { time_gram, .. } = &template else {
            panic!("a time-varying request must produce a time-varying template");
        };
        let p_time = time_gram.nrows();
        let s_x = ndarray::array![[2.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 2.0]];
        let gamma = ndarray::array![0.4, -1.1, 0.9];
        let beta = kronecker_product(
            &gamma.clone().insert_axis(ndarray::Axis(1)),
            &Array2::<f64>::ones((p_time, 1)),
        )
        .column(0)
        .to_owned();
        let tensored = kronecker_product(&s_x, time_gram);
        let static_value = quadratic_form(&s_x, &gamma);
        let tensored_value = quadratic_form(&tensored, &beta);
        assert!(
            (static_value - tensored_value).abs() <= 1e-12 * static_value.abs(),
            "a time-constant function must keep its static penalty: {static_value:e} against \
             {tensored_value:e}"
        );
    }

    /// A many-level random-effect slope surface is an operator-backed design,
    /// `[intercept | one-hot]`. Its covariate Gram is assembled by that design's
    /// own kernel, so it is exactly the group frequencies (intercept row and
    /// column, and the diagonal) with zero between groups, and no `n × p` matrix
    /// is formed: at this width a dense copy would be `n·p·8 ≈ 1.44 GB`.
    #[test]
    fn a_wide_random_effect_block_gets_its_count_gram_from_the_operator_2765() {
        let (age_entry, age_exit) = fixture_times();
        let template =
            build_time_varying_survival_covariate_template(&age_entry, &age_exit, 4, 2, "slope")
                .expect("margin template");
        let n = 60_000usize;
        let levels = 3_000usize;
        let group_ids: Vec<Option<usize>> = (0..n).map(|row| Some((row * 7 + row / 3) % levels)).collect();
        let mut counts = vec![0usize; levels];
        for id in group_ids.iter().flatten() {
            counts[*id] += 1;
        }
        let operator = BlockDesignOperator::new(vec![
            DesignBlock::Intercept(n),
            DesignBlock::RandomEffect(Arc::new(RandomEffectOperator::new(group_ids, levels))),
        ])
        .expect("random-effect block design");
        let design = DesignMatrix::Dense(DenseDesignMatrix::from(Arc::new(operator)));
        let metric = TimeMarginPenaltyMetric::from_template(&template, &design, 1, "slope")
            .expect("metric")
            .expect("time-varying metric");
        let gram = &metric.covariate_gram;
        assert_eq!(gram.dim(), (levels + 1, levels + 1));
        assert_eq!(gram[[0, 0]], 1.0, "the intercept's mean square is one");
        for g in 0..levels {
            let frequency = counts[g] as f64 / n as f64;
            assert_eq!(gram[[0, 1 + g]], frequency, "intercept × group {g}");
            assert_eq!(gram[[1 + g, 0]], frequency, "group {g} × intercept");
            assert_eq!(gram[[1 + g, 1 + g]], frequency, "group {g} diagonal");
        }
        let off_diagonal = (1..=levels)
            .flat_map(|i| (1..=levels).filter(move |j| *j != i).map(move |j| (i, j)))
            .fold(0.0_f64, |acc, (i, j)| acc.max(gram[[i, j]].abs()));
        assert_eq!(off_diagonal, 0.0, "distinct groups never share a row");
    }

    /// The published ψ components of `G_x ⊗ S_t` are the exact derivatives of the
    /// margin penalties along a design `X(ψ) = X₀ + ψ_a X_a + ψ_b X_b + ½ψ_a² X_aa
    /// + ψ_a ψ_b X_ab + ½ψ_b² X_bb`, whose penalties are polynomial in ψ, so a
    /// central difference of the rebuilt penalties is exact to truncation and
    /// roundoff.
    #[test]
    fn margin_penalty_psi_components_match_differences_of_the_rebuilt_penalties_2765() {
        let (age_entry, age_exit) = fixture_times();
        let template =
            build_time_varying_survival_covariate_template(&age_entry, &age_exit, 4, 2, "slope")
                .expect("margin template");
        let SurvivalCovariateTermBlockTemplate::TimeVarying { time_penalties, .. } = &template
        else {
            panic!("a time-varying request must produce a time-varying template");
        };
        let n = age_exit.len();
        let p = 3usize;
        let range = 1..3usize;
        let field = |phase: f64| {
            Array2::<f64>::from_shape_fn((n, range.len()), |(row, col)| {
                (0.13 * row as f64 + phase * (col + 1) as f64).sin()
            })
        };
        let x0 = Array2::<f64>::from_shape_fn((n, p), |(row, col)| {
            if col == 0 { 1.0 } else { (0.21 * row as f64 + col as f64).cos() }
        });
        let x_a = field(0.3);
        let x_b = field(1.1);
        let x_aa = field(1.7);
        let x_ab = field(2.3);
        let x_bb = field(2.9);
        let design_at = |psi_a: f64, psi_b: f64| {
            let mut x = x0.clone();
            let local = &x_a * psi_a
                + &x_b * psi_b
                + &x_aa * (0.5 * psi_a * psi_a)
                + &x_ab * (psi_a * psi_b)
                + &x_bb * (0.5 * psi_b * psi_b);
            let mut moving = x.slice_mut(ndarray::s![.., range.clone()]);
            moving += &local;
            x
        };
        let penalties_at = |psi_a: f64, psi_b: f64| {
            TimeMarginPenaltyMetric::from_template(
                &template,
                &DesignMatrix::from(design_at(psi_a, psi_b)),
                2,
                "slope",
            )
            .expect("metric")
            .expect("time-varying metric")
            .margin_penalties()
        };
        let axis_operator =
            |axis: usize, first: &Array2<f64>, diag: &Array2<f64>, cross_axis: usize| {
                build_embedded_dense_psi_operator(
                    first,
                    diag,
                    Some(&vec![(cross_axis, x_ab.clone())]),
                    range.clone(),
                    p,
                    axis,
                )
                .expect("dense psi operator")
            };
        let axes = vec![
            SpatialPsiAxisDesign {
                operator: Some(axis_operator(0, &x_a, &x_aa, 1)),
                axis: 0,
                group: Some(0),
                range: range.clone(),
            },
            SpatialPsiAxisDesign {
                operator: Some(axis_operator(1, &x_b, &x_bb, 0)),
                axis: 1,
                group: Some(0),
                range: range.clone(),
            },
        ];
        let components = time_penalty_psi_components(
            &DesignMatrix::from(x0.clone()),
            &axes,
            time_penalties,
            2,
        )
        .expect("psi components");
        let h_first = 1e-4;
        let h_second = 1e-3;
        let at = |psi_a: f64, psi_b: f64, k: usize| penalties_at(psi_a, psi_b)[k].clone();
        for k in 0..time_penalties.len() {
            let scale = time_penalties[k].iter().fold(0.0_f64, |acc, v| acc.max(v.abs()));
            for axis in 0..2 {
                let (index, published) = &components.first[axis][k];
                assert_eq!(*index, 2 + k);
                let difference = if axis == 0 {
                    (&at(h_first, 0.0, k) - &at(-h_first, 0.0, k)) / (2.0 * h_first)
                } else {
                    (&at(0.0, h_first, k) - &at(0.0, -h_first, k)) / (2.0 * h_first)
                };
                let gap = (published - &difference)
                    .iter()
                    .fold(0.0_f64, |acc, v| acc.max(v.abs()));
                assert!(
                    gap <= 1e-6 * scale.max(1.0),
                    "axis {axis} margin penalty {k}: first component differs from the \
                     difference by {gap:e}"
                );
            }
            for a in 0..2 {
                for b in 0..2 {
                    let (index, published) = &components.second[a][b][k];
                    assert_eq!(*index, 2 + k);
                    let h = h_second;
                    let difference = if a == b {
                        let along = |s: f64| if a == 0 { at(s, 0.0, k) } else { at(0.0, s, k) };
                        (&along(2.0 * h) - &(&along(0.0) * 2.0) + &along(-2.0 * h)) / (4.0 * h * h)
                    } else {
                        (&at(h, h, k) - &at(h, -h, k) - &at(-h, h, k) + &at(-h, -h, k))
                            / (4.0 * h * h)
                    };
                    let gap = (published - &difference)
                        .iter()
                        .fold(0.0_f64, |acc, v| acc.max(v.abs()));
                    assert!(
                        gap <= 1e-4 * scale.max(1.0),
                        "axes ({a}, {b}) margin penalty {k}: second component differs from the \
                         difference by {gap:e}"
                    );
                }
            }
        }
    }
}
