//! Exact fifth and sixth likelihood derivatives for rigid, time-constant slopes.
//!
//! The static row separates into functions of (q0,g), (q1,g), and qd1.
//! Differentiate each unary composition analytically in q first, then use the
//! finite Bell polynomial in g. This computes the eleven potentially nonzero
//! mixed fifth derivatives once per row, before any coefficient pullback. The
//! sixth is the same construction one order higher; the fourth information
//! derivative consumes it (gam#2894).

use super::*;
use crate::row_kernel::RowKernel;
use gam_math::jet_scalar::JetScalar;

pub(super) const FACTORIAL: [f64; 7] = [1.0, 1.0, 2.0, 6.0, 24.0, 120.0, 720.0];

fn product<const N: usize>(a: &[f64; N], b: &[f64; N]) -> [f64; N] {
    std::array::from_fn(|n| (0..=n).map(|j| a[j] * b[n - j]).sum())
}

/// Mixed derivatives d_q^m d_g^(K-m) F(q*c(g)+b*g), m=0,...,K, with K = N-1.
/// The leaf stack is with respect to its scalar index; c holds Taylor
/// coefficients, so factorials enter only at the final coefficient extraction.
fn mixed_order<const N: usize>(q: f64, c: &[f64; N], b: f64, leaf: &[f64; N]) -> [f64; N] {
    let order = N - 1;
    let mut delta = c.map(|value| q * value);
    delta[0] = 0.0;
    delta[1] += b;
    let mut powers = [[0.0; N]; N];
    powers[0][0] = 1.0;
    for j in 1..=order {
        powers[j] = product(&powers[j - 1], &delta);
    }
    let mut c_power = [0.0; N];
    c_power[0] = 1.0;
    let mut out = [0.0; N];
    for m in 0..=order {
        let k = order - m;
        let mut composed = [0.0; N];
        for j in 0..=k {
            for n in 0..=k {
                composed[n] += leaf[m + j] * powers[j][n] / FACTORIAL[j];
            }
        }
        out[m] = product(&c_power, &composed)[k] * FACTORIAL[k];
        if m < order {
            c_power = product(&c_power, c);
        }
    }
    out
}

/// [`mixed_order`] at order five, the frame the follow-up slope kernel shares.
pub(super) fn mixed_fifth(q: f64, c: &[f64; 6], b: f64, leaf: &[f64; 6]) -> [f64; 6] {
    mixed_order(q, c, b, leaf)
}

fn static_row_fifth(
    primaries: &[f64; STATIC_SLOPE_PRIMARIES],
    inputs: &RigidRowInputs,
) -> Result<[[[[[f64; 4]; 4]; 4]; 4]; 4], String> {
    if inputs.wi == 0.0 {
        return Ok([[[[[0.0; 4]; 4]; 4]; 4]; 4]);
    }
    let [neg_eta0, neg_eta1, derivative] =
        rigid_row_admission_witnesses::<4, StaticSlopeGeometry>(primaries, inputs);
    validate_rigid_row_admission::<4, StaticSlopeGeometry>(
        primaries[PRIMARY_QD1],
        inputs,
        neg_eta0,
        neg_eta1,
        derivative,
    )?;
    let g = primaries[PRIMARY_SLOPE];
    let a = inputs.probit_scale.powi(2) * inputs.covariance_ones;
    let b = inputs.probit_scale * inputs.z_sum;
    // c(g+t)^2 = 1+a(g+t)^2 gives this exact triangular recurrence.
    let mut c = [0.0; 6];
    c[0] = (1.0 + a * g * g).sqrt();
    for n in 1..=5 {
        let rhs = match n {
            1 => 2.0 * a * g,
            2 => a,
            _ => 0.0,
        };
        c[n] = (rhs - (1..n).map(|j| c[j] * c[n - j]).sum::<f64>()) / (2.0 * c[0]);
    }
    let neg_c = c.map(|value| -value);
    let entry = gam_math::probability::normal_logcdf_derivatives_through_fifth(neg_eta0)
        .map(|value| inputs.wi * value);
    let exit = gam_math::probability::normal_logcdf_derivatives_through_fifth(neg_eta1)
        .map(|value| -inputs.wi * (1.0 - inputs.di) * value);
    // The negative-index leaf's q derivative contributes (-c)^m.
    let q0 = mixed_fifth(primaries[PRIMARY_Q0], &neg_c, -b, &entry);
    let mut q1 = mixed_fifth(primaries[PRIMARY_Q1], &neg_c, -b, &exit);
    let event_weight = inputs.wi * inputs.di;
    let eta1 = -neg_eta1;
    let density = [
        0.5 * event_weight * eta1 * eta1,
        event_weight * eta1,
        event_weight,
        0.0,
        0.0,
        0.0,
    ];
    let density_fifth = mixed_fifth(primaries[PRIMARY_Q1], &c, b, &density);
    for m in 0..=5 {
        q1[m] += density_fifth[m];
    }
    // -w*d*log(qd1*c) separates exactly. The fifth log(c) coefficient
    // follows from log(c)=log(1+a*g^2)/2 and its finite Taylor series.
    let mut relative = [0.0; 6];
    relative[1] = 2.0 * a * g / c[0].powi(2);
    relative[2] = a / c[0].powi(2);
    let mut power = [0.0; 6];
    power[0] = 1.0;
    let mut log_c_fifth = 0.0;
    for j in 1..=5 {
        power = product(&power, &relative);
        let sign = if j % 2 == 1 { 1.0 } else { -1.0 };
        log_c_fifth += 0.5 * sign * power[5] * FACTORIAL[5] / j as f64;
    }
    let slope_fifth = q0[0] + q1[0] - event_weight * log_c_fifth;
    let derivative_fifth = -event_weight * 24.0 / primaries[PRIMARY_QD1].powi(5);
    Ok(std::array::from_fn(|i| {
        std::array::from_fn(|j| {
            std::array::from_fn(|k| {
                std::array::from_fn(|l| {
                    std::array::from_fn(|m| {
                        let axes = [i, j, k, l, m];
                        let count = |axis| axes.iter().filter(|&&value| value == axis).count();
                        let n0 = count(PRIMARY_Q0);
                        let n1 = count(PRIMARY_Q1);
                        let nd = count(PRIMARY_QD1);
                        if n0 + n1 + nd == 0 {
                            slope_fifth
                        } else if n0 > 0 && n1 + nd == 0 {
                            q0[n0]
                        } else if n1 > 0 && n0 + nd == 0 {
                            q1[n1]
                        } else if nd == 5 {
                            derivative_fifth
                        } else {
                            0.0
                        }
                    })
                })
            })
        })
    }))
}

fn static_row_sixth(
    primaries: &[f64; STATIC_SLOPE_PRIMARIES],
    inputs: &RigidRowInputs,
) -> Result<[[[[[[f64; 4]; 4]; 4]; 4]; 4]; 4], String> {
    if inputs.wi == 0.0 {
        return Ok([[[[[[0.0; 4]; 4]; 4]; 4]; 4]; 4]);
    }
    let [neg_eta0, neg_eta1, derivative] =
        rigid_row_admission_witnesses::<4, StaticSlopeGeometry>(primaries, inputs);
    validate_rigid_row_admission::<4, StaticSlopeGeometry>(
        primaries[PRIMARY_QD1],
        inputs,
        neg_eta0,
        neg_eta1,
        derivative,
    )?;
    let g = primaries[PRIMARY_SLOPE];
    let a = inputs.probit_scale.powi(2) * inputs.covariance_ones;
    let b = inputs.probit_scale * inputs.z_sum;
    // c(g+t)^2 = 1+a(g+t)^2 gives this exact triangular recurrence.
    let mut c = [0.0; 7];
    c[0] = (1.0 + a * g * g).sqrt();
    for n in 1..=6 {
        let rhs = match n {
            1 => 2.0 * a * g,
            2 => a,
            _ => 0.0,
        };
        c[n] = (rhs - (1..n).map(|j| c[j] * c[n - j]).sum::<f64>()) / (2.0 * c[0]);
    }
    let neg_c = c.map(|value| -value);
    let entry = gam_math::probability::normal_logcdf_derivatives_through_sixth(neg_eta0)
        .map(|value| inputs.wi * value);
    let exit = gam_math::probability::normal_logcdf_derivatives_through_sixth(neg_eta1)
        .map(|value| -inputs.wi * (1.0 - inputs.di) * value);
    // The negative-index leaf's q derivative contributes (-c)^m.
    let q0 = mixed_order(primaries[PRIMARY_Q0], &neg_c, -b, &entry);
    let mut q1 = mixed_order(primaries[PRIMARY_Q1], &neg_c, -b, &exit);
    let event_weight = inputs.wi * inputs.di;
    let eta1 = -neg_eta1;
    let density = [
        0.5 * event_weight * eta1 * eta1,
        event_weight * eta1,
        event_weight,
        0.0,
        0.0,
        0.0,
        0.0,
    ];
    let density_sixth = mixed_order(primaries[PRIMARY_Q1], &c, b, &density);
    for m in 0..=6 {
        q1[m] += density_sixth[m];
    }
    // -w*d*log(qd1*c) separates exactly. The sixth log(c) coefficient
    // follows from log(c)=log(1+a*g^2)/2 and its finite Taylor series.
    let mut relative = [0.0; 7];
    relative[1] = 2.0 * a * g / c[0].powi(2);
    relative[2] = a / c[0].powi(2);
    let mut power = [0.0; 7];
    power[0] = 1.0;
    let mut log_c_sixth = 0.0;
    for j in 1..=6 {
        power = product(&power, &relative);
        let sign = if j % 2 == 1 { 1.0 } else { -1.0 };
        log_c_sixth += 0.5 * sign * power[6] * FACTORIAL[6] / j as f64;
    }
    let slope_sixth = q0[0] + q1[0] - event_weight * log_c_sixth;
    // d⁶(−log qd1)/dqd1⁶ = 120/qd1⁶.
    let derivative_sixth = event_weight * 120.0 / primaries[PRIMARY_QD1].powi(6);
    Ok(std::array::from_fn(|i| {
        std::array::from_fn(|j| {
            std::array::from_fn(|k| {
                std::array::from_fn(|l| {
                    std::array::from_fn(|m| {
                        std::array::from_fn(|n| {
                            let axes = [i, j, k, l, m, n];
                            let count =
                                |axis| axes.iter().filter(|&&value| value == axis).count();
                            let n0 = count(PRIMARY_Q0);
                            let n1 = count(PRIMARY_Q1);
                            let nd = count(PRIMARY_QD1);
                            if n0 + n1 + nd == 0 {
                                slope_sixth
                            } else if n0 > 0 && n1 + nd == 0 {
                                q0[n0]
                            } else if n1 > 0 && n0 + nd == 0 {
                                q1[n1]
                            } else if nd == 6 {
                                derivative_sixth
                            } else {
                                0.0
                            }
                        })
                    })
                })
            })
        })
    }))
}

impl SurvivalMarginalSlopeRowKernel<STATIC_SLOPE_PRIMARIES, StaticSlopeGeometry> {
    pub(crate) fn third_information_all_axes(
        &self,
        u: &[f64],
        v: &[f64],
    ) -> Result<Vec<Array2<f64>>, String> {
        self.third_information_all_axes_from(u, v, static_row_fifth)
    }

    /// [`Self::primary_third_information_all_axes_from`] on the time-constant slope frame.
    pub(crate) fn primary_third_information_all_axes(
        &self,
        row_weights: &[f64],
        directions: impl Fn(usize) -> Result<PrimaryThirdDirections<STATIC_SLOPE_PRIMARIES>, String>,
    ) -> Result<Vec<Array2<f64>>, String> {
        self.primary_third_information_all_axes_from(row_weights, directions, static_row_fifth)
    }

    /// [`Self::design_psi_third_information_all_axes_from`] on the time-constant slope frame.
    pub(crate) fn design_psi_third_information_all_axes(
        &self,
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        d_beta: &[f64],
        row_weights: &[f64],
    ) -> Result<Option<Vec<Array2<f64>>>, String> {
        self.design_psi_third_information_all_axes_from(
            derivative_blocks,
            psi_index,
            d_beta,
            row_weights,
            static_row_fifth,
        )
    }

    /// [`Self::design_psi_pair_third_information_all_axes_from`] on the time-constant slope frame.
    pub(crate) fn design_psi_pair_third_information_all_axes(
        &self,
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_i: usize,
        psi_j: usize,
        row_weights: &[f64],
    ) -> Result<Option<Vec<Array2<f64>>>, String> {
        self.design_psi_pair_third_information_all_axes_from(
            derivative_blocks,
            psi_i,
            psi_j,
            row_weights,
            static_row_fifth,
        )
    }

    /// `⟨W, H³[u, e_a, e_b]⟩` for every axis pair: the fifth likelihood derivatives
    /// contracted with the row-projected trace weight and one direction, pulled back as a
    /// Hessian in one row pass (gam#2894). Linear in the symmetric weight `W`.
    pub(crate) fn contracted_trace_hessian_directional(
        &self,
        weight: &Array2<f64>,
        u: &[f64],
    ) -> Result<Array2<f64>, String> {
        let p = self.n_coefficients();
        if weight.dim() != (p, p) || u.len() != p || u.iter().any(|x| !x.is_finite()) {
            return Err(format!(
                "survival contracted trace Hessian derivative requires a ({p}, {p}) weight and a finite direction of length {p}"
            ));
        }
        self.chunked_pullback_reduce(p, |row, acc| -> Result<(), String> {
            let inputs = rigid_row_inputs(
                &self.family,
                &self.block_states,
                row,
                "contracted trace Hessian derivative",
            )?;
            let primaries = rigid_row_kernel_primaries::<
                STATIC_SLOPE_PRIMARIES,
                StaticSlopeGeometry,
            >(&self.family, &self.block_states, row)?;
            let fifth = static_row_fifth(&primaries, &inputs)?;
            let w_row = self.primary_trace_weight(row, weight)?;
            let du = self.jacobian_action(row, u);
            let mut coeff = [[0.0_f64; STATIC_SLOPE_PRIMARIES]; STATIC_SLOPE_PRIMARIES];
            for c in 0..STATIC_SLOPE_PRIMARIES {
                for d in 0..STATIC_SLOPE_PRIMARIES {
                    let mut sum = 0.0;
                    for a in 0..STATIC_SLOPE_PRIMARIES {
                        for b in 0..STATIC_SLOPE_PRIMARIES {
                            for e in 0..STATIC_SLOPE_PRIMARIES {
                                sum += w_row[a][b] * fifth[a][b][c][d][e] * du[e];
                            }
                        }
                    }
                    coeff[c][d] = sum;
                }
            }
            self.add_pullback_hessian(row, &coeff, acc);
            Ok(())
        })
    }

    /// `⟨W, H⁴[u, w, e_a, e_b]⟩` for every axis pair: the sixth likelihood derivatives
    /// contracted with the row-projected trace weight and two directions, pulled back as a
    /// Hessian in one row pass (gam#2894).
    pub(crate) fn contracted_trace_hessian_second_directional(
        &self,
        weight: &Array2<f64>,
        u: &[f64],
        w: &[f64],
    ) -> Result<Array2<f64>, String> {
        let p = self.n_coefficients();
        if weight.dim() != (p, p)
            || u.len() != p
            || w.len() != p
            || u.iter().chain(w).any(|x| !x.is_finite())
        {
            return Err(format!(
                "survival contracted trace Hessian second derivative requires a ({p}, {p}) weight and finite directions of length {p}"
            ));
        }
        self.chunked_pullback_reduce(p, |row, acc| -> Result<(), String> {
            let inputs = rigid_row_inputs(
                &self.family,
                &self.block_states,
                row,
                "contracted trace Hessian second derivative",
            )?;
            let primaries = rigid_row_kernel_primaries::<
                STATIC_SLOPE_PRIMARIES,
                StaticSlopeGeometry,
            >(&self.family, &self.block_states, row)?;
            let sixth = static_row_sixth(&primaries, &inputs)?;
            let w_row = self.primary_trace_weight(row, weight)?;
            let du = self.jacobian_action(row, u);
            let dw = self.jacobian_action(row, w);
            let mut coeff = [[0.0_f64; STATIC_SLOPE_PRIMARIES]; STATIC_SLOPE_PRIMARIES];
            for c in 0..STATIC_SLOPE_PRIMARIES {
                for d in 0..STATIC_SLOPE_PRIMARIES {
                    let mut sum = 0.0;
                    for a in 0..STATIC_SLOPE_PRIMARIES {
                        for b in 0..STATIC_SLOPE_PRIMARIES {
                            for e in 0..STATIC_SLOPE_PRIMARIES {
                                for f in 0..STATIC_SLOPE_PRIMARIES {
                                    sum += w_row[a][b] * sixth[a][b][c][d][e][f] * du[e] * dw[f];
                                }
                            }
                        }
                    }
                    coeff[c][d] = sum;
                }
            }
            self.add_pullback_hessian(row, &coeff, acc);
            Ok(())
        })
    }
}

/// One row's primary directions for
/// [`SurvivalMarginalSlopeRowKernel::primary_third_information_all_axes_from`]: the
/// pair the fifth likelihood derivatives contract with, and the direction, if any,
/// the fourth derivatives contract with.
pub(crate) type PrimaryThirdDirections<const P: usize> = ([f64; P], [f64; P], Option<[f64; P]>);

impl<const P: usize, G: SlopeRowGeometry<P>> SurvivalMarginalSlopeRowKernel<P, G> {
    pub(super) fn third_information_all_axes_from(
        &self,
        u: &[f64],
        v: &[f64],
        fifth: impl Fn(&[f64; P], &RigidRowInputs) -> Result<[[[[[f64; P]; P]; P]; P]; P], String>,
    ) -> Result<Vec<Array2<f64>>, String> {
        let p = self.n_coefficients();
        if u.len() != p || v.len() != p || u.iter().chain(v).any(|x| !x.is_finite()) {
            return Err(format!(
                "survival third information derivative requires finite directions of length {p}"
            ));
        }
        let unit_measure = vec![1.0; self.family.n];
        self.primary_third_information_all_axes_from(
            &unit_measure,
            |row| Ok((self.jacobian_action(row, u), self.jacobian_action(row, v), None)),
            fifth,
        )
    }

    /// `{D_β_a M}` along every coefficient axis `a`, where row `r` contributes
    /// `w_r·Jᵀ(T⁴[x_r, y_r] + T³[z_r])J` to `M`: its fifth likelihood derivatives
    /// contracted with the primary directions `x_r, y_r`, plus its fourth derivatives
    /// contracted with `z_r`. `row_weights` is the outer row measure, zero on a row
    /// the measure leaves out.
    ///
    /// The directions are primary-space vectors rather than coefficient directions
    /// because a baseline-chart coordinate moves the location index's offsets and
    /// leaves the coefficient map `J` fixed (gam#2765).
    pub(super) fn primary_third_information_all_axes_from(
        &self,
        row_weights: &[f64],
        directions: impl Fn(usize) -> Result<PrimaryThirdDirections<P>, String>,
        fifth: impl Fn(&[f64; P], &RigidRowInputs) -> Result<[[[[[f64; P]; P]; P]; P]; P], String>,
    ) -> Result<Vec<Array2<f64>>, String> {
        if row_weights.len() != self.family.n {
            return Err(format!(
                "survival third information derivative row measure has {} weights for {} rows",
                row_weights.len(),
                self.family.n,
            ));
        }
        let mut tensors = Vec::with_capacity(self.family.n);
        for row in 0..self.family.n {
            let mut tensor = [[[0.0; P]; P]; P];
            let weight = row_weights[row];
            if weight != 0.0 {
                let inputs = rigid_row_inputs(
                    &self.family,
                    &self.block_states,
                    row,
                    "third information derivative",
                )?;
                let primaries =
                    rigid_row_kernel_primaries::<P, G>(&self.family, &self.block_states, row)?;
                let fifth = fifth(&primaries, &inputs)?;
                let (x, y, z) = directions(row)?;
                for a in 0..P {
                    for b in 0..P {
                        for c in 0..P {
                            for d in 0..P {
                                for e in 0..P {
                                    tensor[a][b][c] += fifth[a][b][c][d][e] * x[d] * y[e];
                                }
                            }
                        }
                    }
                }
                if let Some(z) = z {
                    let vars: [SparseTower4<P, RIGID_LINEAR_MASK>; P] =
                        std::array::from_fn(|axis| SparseTower4::variable(primaries[axis], axis));
                    let tower = rigid_row_nll::<P, G, _>(&vars, &inputs)?;
                    for a in 0..P {
                        for b in 0..P {
                            for c in 0..P {
                                for d in 0..P {
                                    tensor[a][b][c] += tower.t4[a][b][c][d] * z[d];
                                }
                            }
                        }
                    }
                }
                if weight != 1.0 {
                    for plane in &mut tensor {
                        for line in plane {
                            for entry in line {
                                *entry *= weight;
                            }
                        }
                    }
                }
            }
            tensors.push(tensor);
        }
        self.all_axes_primary_tensor_pullback(&tensors)
    }

    /// `{D_β_a D_β ∂_ψ H[v]}` along every coefficient axis `a` for a design
    /// hyperparameter ψ that moves the marginal or the slope design (gam#2765).
    ///
    /// With the design motion `J_ψ = L ⊗ x_ψ` (primary loading `L`, design-derivative
    /// row `x_ψ`), a row contributes `Jᵀ(T⁵[J_ψβ, Jv, J e_a] + T⁴[J_ψv, J e_a])J`,
    /// `J_ψᵀ T⁴[J e_a, Jv] J` with its transpose, and, on an axis the design moves,
    /// `Jᵀ T⁴[J_ψ e_a, Jv] J`. Returns `None` where the family has no ψ block for the
    /// axis, as the first-order drift does.
    pub(super) fn design_psi_third_information_all_axes_from(
        &self,
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        d_beta: &[f64],
        row_weights: &[f64],
        fifth: impl Fn(&[f64; P], &RigidRowInputs) -> Result<[[[[[f64; P]; P]; P]; P]; P], String>,
    ) -> Result<Option<Vec<Array2<f64>>>, String> {
        let family = &self.family;
        let Some((block_idx, local_idx, p_psi, label)) =
            family.psi_block_info(derivative_blocks, psi_index)?
        else {
            return Ok(None);
        };
        let primary_array = |vector: &Array1<f64>| -> Result<[f64; P], String> {
            if vector.len() != P {
                return Err(format!(
                    "survival design ψ third information derivative: a primary vector has {} entries for a {P}-primary frame",
                    vector.len()
                ));
            }
            Ok(std::array::from_fn(|k| vector[k]))
        };
        let p = self.n_coefficients();
        if d_beta.len() != p || row_weights.len() != family.n {
            return Err(format!(
                "survival design ψ third information derivative requires a direction of length {p} and {} row weights",
                family.n
            ));
        }
        let (beta_block, psi_range) = match block_idx {
            1 => (&self.block_states[1].beta, self.slices.marginal.clone()),
            _ => (&self.block_states[2].beta, self.slices.slope.clone()),
        };
        let policy = gam_runtime::resource::ResourcePolicy::default_library();
        let psi_map = crate::custom_family::resolve_custom_family_x_psi_map(
            &derivative_blocks[block_idx][local_idx],
            family.n,
            p_psi,
            0..family.n,
            label,
            &policy,
        )
        .map_err(|error| error.to_string())?;
        let d_beta_block = ndarray::ArrayView1::from(&d_beta[psi_range.clone()]);
        let channels_at = |row: usize| -> Result<PsiRowChannels, String> {
            let psi_row = psi_map
                .row_vector(row)
                .map_err(|error| format!("survival design ψ third information row: {error}"))?;
            psi_row_channels(family, None, block_idx, psi_row)
        };

        let mut axes = self.primary_third_information_all_axes_from(
            row_weights,
            |row| {
                let channels = channels_at(row)?;
                Ok((
                    primary_array(&channels.direction(beta_block.view()))?,
                    self.jacobian_action(row, d_beta),
                    Some(primary_array(&channels.direction(d_beta_block))?),
                ))
            },
            fifth,
        )?;

        let slices = &self.slices;
        let p_h = slices.score_warp.as_ref().map_or(0, |range| range.len());
        let p_w = slices.link_dev.as_ref().map_or(0, |range| range.len());
        let p_i = slices.influence.as_ref().map_or(0, |range| range.len());
        let (p_t, p_m, p_g) = (slices.time.len(), slices.marginal.len(), slices.slope.len());
        let identity = Array2::<f64>::eye(p);
        let jacobians = self.jacobian_action_matrix(identity.view()).ok_or_else(|| {
            "survival design ψ third information derivative requires a dense J·I projection"
                .to_string()
        })?;
        let rows: Vec<usize> = (0..family.n).filter(|&row| row_weights[row] != 0.0).collect();
        let accumulators = crate::marginal_slope_shared::chunked_row_reduction(
            rows.as_slice(),
            || {
                (0..p)
                    .map(|_| BlockHessianAccumulator::new(p_t, p_m, p_g, p_h, p_w, p_i))
                    .collect::<Vec<_>>()
            },
            |row, accumulators| -> Result<(), String> {
                let inputs = rigid_row_inputs(
                    family,
                    &self.block_states,
                    row,
                    "design ψ third information derivative",
                )?;
                let primaries =
                    rigid_row_kernel_primaries::<P, G>(family, &self.block_states, row)?;
                let vars: [SparseTower4<P, RIGID_LINEAR_MASK>; P] =
                    std::array::from_fn(|axis| SparseTower4::variable(primaries[axis], axis));
                let tower = rigid_row_nll::<P, G, _>(&vars, &inputs)?;
                let jv = self.jacobian_action(row, d_beta);
                let channels = channels_at(row)?;
                // `K_c[k, γ] = w·T⁴[L_c, e_k, e_γ, Jv]` for each channel: the J_ψ-sided
                // kernel and, on an axis the design moves, that axis's pullback coefficient.
                let weight = row_weights[row];
                let kernels: Vec<Array2<f64>> = channels
                    .channels()
                    .iter()
                    .map(|(loading, _)| {
                        Array2::from_shape_fn((P, P), |(k, gamma)| {
                            let mut sum = 0.0;
                            for alpha in 0..P {
                                if loading[alpha] == 0.0 {
                                    continue;
                                }
                                for delta in 0..P {
                                    sum += loading[alpha] * tower.t4[alpha][k][gamma][delta] * jv[delta];
                                }
                            }
                            weight * sum
                        })
                    })
                    .collect();
                let mut right_primary = ndarray::Array1::<f64>::zeros(P);
                for (axis, accumulator) in accumulators.iter_mut().enumerate() {
                    for ((_, design_row), kernel) in channels.channels().iter().zip(&kernels) {
                        right_primary.fill(0.0);
                        for k in 0..P {
                            let axis_loading = jacobians[[row, k * p + axis]];
                            if axis_loading != 0.0 {
                                right_primary.scaled_add(axis_loading, &kernel.row(k));
                            }
                        }
                        accumulator.add_rank1_psi_cross(
                            family,
                            row,
                            block_idx,
                            design_row,
                            &right_primary,
                        )?;
                        if psi_range.contains(&axis) {
                            let coefficient = design_row[axis - psi_range.start];
                            if coefficient != 0.0 {
                                accumulator.add_pullback(
                                    family,
                                    row,
                                    &kernel.mapv(|value| value * coefficient),
                                )?;
                            }
                        }
                    }
                }
                Ok(())
            },
            |total, chunk| {
                for (left, right) in total.iter_mut().zip(chunk.iter()) {
                    left.add(right);
                }
            },
        )?;
        for (axis, accumulator) in accumulators.iter().enumerate() {
            axes[axis] += &accumulator.to_dense(slices);
        }
        Ok(Some(axes))
    }

    /// `{D_β_a ∂²_ψiψj H}` along every coefficient axis `a` for a pair of design
    /// hyperparameters (gam#2765): the rigid ψψ Hessian of
    /// `psi_second_order_terms_inner_with_options` differentiated once along `e_a`.
    ///
    /// With `d = L·(x_ψ·β)`, `d_ij = L_i·(x_ψiψj·β)` and `r_a = J e_a`, a row
    /// contributes `Jᵀ(T⁵[d_i, d_j, r_a] + T⁴[d_ij, r_a])J`; the ψ-sided crosses
    /// `x_i ⊗ T⁴[d_j, r_a]ᵀL_i`, `x_j ⊗ T⁴[d_i, r_a]ᵀL_j`, `x_ij ⊗ T³[r_a]ᵀL_i` and
    /// `x_i ⊗ x_j · L_iᵀT³[r_a]L_j` with their transposes; and, on the axes a design
    /// moves, the motion of `d_i`, `d_j` and `d_ij` themselves. Returns `None` where
    /// the family has no ψ block for either axis.
    pub(super) fn design_psi_pair_third_information_all_axes_from(
        &self,
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_i: usize,
        psi_j: usize,
        row_weights: &[f64],
        fifth: impl Fn(&[f64; P], &RigidRowInputs) -> Result<[[[[[f64; P]; P]; P]; P]; P], String>,
    ) -> Result<Option<Vec<Array2<f64>>>, String> {
        let family = &self.family;
        let n = family.n;
        let Some((block_i, local_i, p_psi_i, label_i)) =
            family.psi_block_info(derivative_blocks, psi_i)?
        else {
            return Ok(None);
        };
        let Some((block_j, local_j, p_psi_j, label_j)) =
            family.psi_block_info(derivative_blocks, psi_j)?
        else {
            return Ok(None);
        };
        let primary_array = |vector: &Array1<f64>| -> Result<[f64; P], String> {
            if vector.len() != P {
                return Err(format!(
                    "survival design ψ-pair third information derivative: a primary vector has {} entries for a {P}-primary frame",
                    vector.len()
                ));
            }
            Ok(std::array::from_fn(|k| vector[k]))
        };
        let p = self.n_coefficients();
        if row_weights.len() != n {
            return Err(format!(
                "survival design ψ-pair third information derivative has {} row weights for {n} rows",
                row_weights.len()
            ));
        }
        let block_range = |block: usize| match block {
            1 => self.slices.marginal.clone(),
            _ => self.slices.slope.clone(),
        };
        let (range_i, range_j) = (block_range(block_i), block_range(block_j));
        let beta_i = &self.block_states[block_i].beta;
        let beta_j = &self.block_states[block_j].beta;
        let policy = gam_runtime::resource::ResourcePolicy::default_library();
        let map_i = crate::custom_family::resolve_custom_family_x_psi_map(
            &derivative_blocks[block_i][local_i],
            n,
            p_psi_i,
            0..n,
            label_i,
            &policy,
        )
        .map_err(|error| error.to_string())?;
        let map_j = crate::custom_family::resolve_custom_family_x_psi_map(
            &derivative_blocks[block_j][local_j],
            n,
            p_psi_j,
            0..n,
            label_j,
            &policy,
        )
        .map_err(|error| error.to_string())?;
        let map_ij = if block_i == block_j {
            Some(
                crate::custom_family::resolve_custom_family_x_psi_psi_map(
                    &derivative_blocks[block_i][local_i],
                    &derivative_blocks[block_j][local_j],
                    local_j,
                    n,
                    p_psi_i,
                    0..n,
                    label_i,
                    &policy,
                )
                .map_err(|error| error.to_string())?,
            )
        } else {
            None
        };
        let row_error = |error| format!("survival design ψ-pair third information row: {error}");
        let channels_at = |row: usize| -> Result<
            (PsiRowChannels, PsiRowChannels, Option<PsiRowChannels>),
            String,
        > {
            let x_i = map_i.row_vector(row).map_err(row_error)?;
            let x_j = map_j.row_vector(row).map_err(row_error)?;
            let x_ij = map_ij
                .as_ref()
                .map(|map| map.row_vector(row).map_err(row_error))
                .transpose()?;
            Ok((
                psi_row_channels(family, None, block_i, x_i)?,
                psi_row_channels(family, None, block_j, x_j)?,
                x_ij.map(|x_ij| psi_row_channels(family, None, block_i, x_ij))
                    .transpose()?,
            ))
        };

        let mut axes = self.primary_third_information_all_axes_from(
            row_weights,
            |row| {
                let (channels_i, channels_j, channels_ij) = channels_at(row)?;
                Ok((
                    primary_array(&channels_i.direction(beta_i.view()))?,
                    primary_array(&channels_j.direction(beta_j.view()))?,
                    channels_ij
                        .map(|channels| primary_array(&channels.direction(beta_i.view())))
                        .transpose()?,
                ))
            },
            fifth,
        )?;

        let slices = &self.slices;
        let p_h = slices.score_warp.as_ref().map_or(0, |range| range.len());
        let p_w = slices.link_dev.as_ref().map_or(0, |range| range.len());
        let p_influence = slices.influence.as_ref().map_or(0, |range| range.len());
        let (p_t, p_m, p_g) = (slices.time.len(), slices.marginal.len(), slices.slope.len());
        let identity = Array2::<f64>::eye(p);
        let jacobians = self.jacobian_action_matrix(identity.view()).ok_or_else(|| {
            "survival design ψ-pair third information derivative requires a dense J·I projection"
                .to_string()
        })?;
        let rows: Vec<usize> = (0..n).filter(|&row| row_weights[row] != 0.0).collect();
        let accumulators = crate::marginal_slope_shared::chunked_row_reduction(
            rows.as_slice(),
            || {
                (0..p)
                    .map(|_| BlockHessianAccumulator::new(p_t, p_m, p_g, p_h, p_w, p_influence))
                    .collect::<Vec<_>>()
            },
            |row, accumulators| -> Result<(), String> {
                let inputs = rigid_row_inputs(
                    family,
                    &self.block_states,
                    row,
                    "design ψ-pair third information derivative",
                )?;
                let primaries =
                    rigid_row_kernel_primaries::<P, G>(family, &self.block_states, row)?;
                let vars: [SparseTower4<P, RIGID_LINEAR_MASK>; P] =
                    std::array::from_fn(|axis| SparseTower4::variable(primaries[axis], axis));
                let tower = rigid_row_nll::<P, G, _>(&vars, &inputs)?;
                let (channels_i, channels_j, channels_ij) = channels_at(row)?;
                let d_i = primary_array(&channels_i.direction(beta_i.view()))?;
                let d_j = primary_array(&channels_j.direction(beta_j.view()))?;
                // Row kernels, each contracted per axis with `r_a`: for a channel `c` of ψ_i,
                // `k3i_c[k, γ] = w·T³[L_c, e_k, e_γ]` and `k4i_c[k, γ] = w·T⁴[L_c, e_k, e_γ, d_j]`;
                // for a channel `c'` of ψ_j, `k4j_c'[k, γ] = w·T⁴[L_c', e_k, e_γ, d_i]`; and
                // `s3_cc'[k] = w·T³[L_c, L_c', e_k]`. The second-derivative rows carry ψ_i's
                // loadings.
                let weight = row_weights[row];
                let third_kernel = |loading: &Array1<f64>| {
                    Array2::from_shape_fn((P, P), |(k, gamma)| {
                        let mut third = 0.0;
                        for alpha in 0..P {
                            third += loading[alpha] * tower.t3[alpha][k][gamma];
                        }
                        weight * third
                    })
                };
                let fourth_kernel = |loading: &Array1<f64>, direction: &[f64; P]| {
                    Array2::from_shape_fn((P, P), |(k, gamma)| {
                        let mut fourth = 0.0;
                        for alpha in 0..P {
                            for delta in 0..P {
                                fourth += loading[alpha]
                                    * tower.t4[alpha][k][gamma][delta]
                                    * direction[delta];
                            }
                        }
                        weight * fourth
                    })
                };
                let k3i: Vec<Array2<f64>> = channels_i
                    .channels()
                    .iter()
                    .map(|(loading, _)| third_kernel(loading))
                    .collect();
                let k4i: Vec<Array2<f64>> = channels_i
                    .channels()
                    .iter()
                    .map(|(loading, _)| fourth_kernel(loading, &d_j))
                    .collect();
                let k4j: Vec<Array2<f64>> = channels_j
                    .channels()
                    .iter()
                    .map(|(loading, _)| fourth_kernel(loading, &d_i))
                    .collect();
                let k3ij: Vec<Array2<f64>> = channels_ij.as_ref().map_or_else(Vec::new, |channels| {
                    channels
                        .channels()
                        .iter()
                        .map(|(loading, _)| third_kernel(loading))
                        .collect()
                });
                let s3: Vec<Vec<Array1<f64>>> = k3i
                    .iter()
                    .map(|k3| {
                        channels_j
                            .channels()
                            .iter()
                            .map(|(loading, _)| k3.dot(loading))
                            .collect()
                    })
                    .collect();
                let mut right = ndarray::Array1::<f64>::zeros(P);
                for (axis, accumulator) in accumulators.iter_mut().enumerate() {
                    let axis_direction: [f64; P] =
                        std::array::from_fn(|k| jacobians[[row, k * p + axis]]);
                    let contract = |target: &mut Array1<f64>, kernel: &Array2<f64>| {
                        for k in 0..P {
                            if axis_direction[k] != 0.0 {
                                target.scaled_add(axis_direction[k], &kernel.row(k));
                            }
                        }
                    };
                    for (c, (_, row_i)) in channels_i.channels().iter().enumerate() {
                        right.fill(0.0);
                        contract(&mut right, &k4i[c]);
                        if range_j.contains(&axis) {
                            for (c_prime, (_, row_j)) in channels_j.channels().iter().enumerate() {
                                right.scaled_add(row_j[axis - range_j.start], &s3[c][c_prime]);
                            }
                        }
                        accumulator.add_rank1_psi_cross(family, row, block_i, row_i, &right)?;
                    }
                    for (c_prime, (_, row_j)) in channels_j.channels().iter().enumerate() {
                        right.fill(0.0);
                        contract(&mut right, &k4j[c_prime]);
                        if range_i.contains(&axis) {
                            for (c, (_, row_i)) in channels_i.channels().iter().enumerate() {
                                right.scaled_add(row_i[axis - range_i.start], &s3[c][c_prime]);
                            }
                        }
                        accumulator.add_rank1_psi_cross(family, row, block_j, row_j, &right)?;
                    }
                    if let Some(channels) = channels_ij.as_ref() {
                        for (c, (_, row_ij)) in channels.channels().iter().enumerate() {
                            right.fill(0.0);
                            contract(&mut right, &k3ij[c]);
                            accumulator.add_rank1_psi_cross(family, row, block_i, row_ij, &right)?;
                        }
                    }
                    for (c, (_, row_i)) in channels_i.channels().iter().enumerate() {
                        for (c_prime, (_, row_j)) in channels_j.channels().iter().enumerate() {
                            let cross: f64 = (0..P)
                                .filter(|&k| axis_direction[k] != 0.0)
                                .map(|k| axis_direction[k] * s3[c][c_prime][k])
                                .sum();
                            accumulator.add_psi_psi_outer(block_i, row_i, block_j, row_j, cross);
                        }
                    }
                    if range_i.contains(&axis) {
                        for (c, (_, row_i)) in channels_i.channels().iter().enumerate() {
                            let coefficient = row_i[axis - range_i.start];
                            if coefficient != 0.0 {
                                accumulator.add_pullback(
                                    family,
                                    row,
                                    &k4i[c].mapv(|value| value * coefficient),
                                )?;
                            }
                        }
                        if let Some(channels) = channels_ij.as_ref() {
                            for (c, (_, row_ij)) in channels.channels().iter().enumerate() {
                                let coefficient = row_ij[axis - range_i.start];
                                if coefficient != 0.0 {
                                    accumulator.add_pullback(
                                        family,
                                        row,
                                        &k3ij[c].mapv(|value| value * coefficient),
                                    )?;
                                }
                            }
                        }
                    }
                    if range_j.contains(&axis) {
                        for (c_prime, (_, row_j)) in channels_j.channels().iter().enumerate() {
                            let coefficient = row_j[axis - range_j.start];
                            if coefficient != 0.0 {
                                accumulator.add_pullback(
                                    family,
                                    row,
                                    &k4j[c_prime].mapv(|value| value * coefficient),
                                )?;
                            }
                        }
                    }
                }
                Ok(())
            },
            |total, chunk| {
                for (left, right) in total.iter_mut().zip(chunk.iter()) {
                    left.add(right);
                }
            },
        )?;
        for (axis, accumulator) in accumulators.iter().enumerate() {
            axes[axis] += &accumulator.to_dense(slices);
        }
        Ok(Some(axes))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gam_math::jet_scalar::JetScalar;

    #[test]
    fn static_fifth_matches_differentiated_fourth_for_events_and_censoring() {
        for event in [0.0, 1.0] {
            for slope in [-1.3, 0.0, 0.8] {
                let inputs = RigidRowInputs {
                    row: 0,
                    wi: 1.7,
                    di: event,
                    z_sum: 0.7,
                    covariance_ones: 1.2,
                    probit_scale: 0.9,
                    qd1_lower: 1e-8,
                };
                let point = [-0.9, 0.4, 1.1, slope];
                let exact = static_row_fifth(&point, &inputs).expect("admitted row");
                for axis in 0..4 {
                    let step = 1e-5;
                    let tower = |sign: f64| {
                        let mut shifted = point;
                        shifted[axis] += sign * step;
                        let vars: [SparseTower4<4, RIGID_LINEAR_MASK>; 4] =
                            std::array::from_fn(|a| SparseTower4::variable(shifted[a], a));
                        rigid_row_nll::<4, StaticSlopeGeometry, _>(&vars, &inputs)
                            .expect("perturbed row")
                    };
                    let plus = tower(1.0);
                    let minus = tower(-1.0);
                    for a in 0..4 {
                        for b in 0..4 {
                            for c in 0..4 {
                                for d in 0..4 {
                                    let fd =
                                        (plus.t4[a][b][c][d] - minus.t4[a][b][c][d]) / (2.0 * step);
                                    let actual = exact[a][b][c][d][axis];
                                    assert!(
                                        (actual - fd).abs() <= 2e-6 * (1.0 + fd.abs()),
                                        "event={event} slope={slope} axes={a},{b},{c},{d},{axis}: exact={actual} FD={fd}"
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn static_sixth_matches_differentiated_fifth_for_events_and_censoring_2894() {
        for event in [0.0, 1.0] {
            for slope in [-1.3, 0.0, 0.8] {
                let inputs = RigidRowInputs {
                    row: 0,
                    wi: 1.7,
                    di: event,
                    z_sum: 0.7,
                    covariance_ones: 1.2,
                    probit_scale: 0.9,
                    qd1_lower: 1e-8,
                };
                let point = [-0.9, 0.4, 1.1, slope];
                let exact = static_row_sixth(&point, &inputs).expect("admitted row");
                for axis in 0..4 {
                    let step = 1e-5;
                    let fifth = |sign: f64| {
                        let mut shifted = point;
                        shifted[axis] += sign * step;
                        static_row_fifth(&shifted, &inputs).expect("perturbed row")
                    };
                    let plus = fifth(1.0);
                    let minus = fifth(-1.0);
                    for a in 0..4 {
                        for b in 0..4 {
                            for c in 0..4 {
                                for d in 0..4 {
                                    for e in 0..4 {
                                        let fd = (plus[a][b][c][d][e] - minus[a][b][c][d][e])
                                            / (2.0 * step);
                                        let actual = exact[a][b][c][d][e][axis];
                                        assert!(
                                            (actual - fd).abs() <= 2e-6 * (1.0 + fd.abs()),
                                            "event={event} slope={slope} axes={a},{b},{c},{d},{e},{axis}: exact={actual} FD={fd}"
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
