//! The exact-Newton joint evaluation methods on the family: dense/gradient
//! dynamic-q evaluation, time-wiggle and
//! flex-no-wiggle directional derivatives, and the blockwise exact-Newton
//! dispatchers (rigid / per-z / flexible / time-wiggle).

use super::*;

impl SurvivalMarginalSlopeFamily {
    pub(crate) fn evaluate_exact_newton_joint_dynamic_q_dense(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<(f64, Array1<f64>, Array2<f64>), String> {
        let flex_active = self.effective_flex_active(block_states)?;
        if flex_active {
            self.validate_exact_monotonicity(block_states)?;
        }
        let slices = block_slices(self, block_states);
        let primary = flex_primary_slices(self);
        let p_total = slices.total;
        let identity_blocks = if flex_active {
            flex_identity_block_pairs(&primary, &slices)
        } else {
            vec![]
        };

        type Acc = (f64, Array1<f64>, Array2<f64>);
        // Per-thread accumulator carries a `SurvivalMarginalSlopeDynamicRow`
        // workspace alongside the (nll, gradient, hessian) tuple so the nine
        // Array2/Array1 buffers inside it are reused across all rows assigned
        // to a single rayon worker. At large scale this eliminates the
        // ~80 GB-per-sweep allocator traffic the fresh-allocation path used.
        type AccWithWs = (Acc, SurvivalMarginalSlopeDynamicRow);
        let make_acc = || -> AccWithWs {
            (
                (
                    0.0,
                    Array1::zeros(p_total),
                    Array2::zeros((p_total, p_total)),
                ),
                SurvivalMarginalSlopeDynamicRow::empty_workspace(),
            )
        };

        let final_acc = gam_problem::outer_subsample::RowSet::All.par_try_reduce_fold(
            self.n,
            make_acc,
            |mut acc, row, row_weight| -> Result<_, String> {
                // Full-data pass: `RowSet::All` folds with a literal weight of
                // 1, which is what licenses the unweighted sums below. A
                // Horvitz-Thompson weight from a subsampled row set would bias
                // every accumulated term instead, so refuse rather than
                // silently ignore the argument that carries the premise.
                if row_weight != 1.0 {
                    return Err(format!(
                        "survival marginal-slope joint eval: full-data pass saw row {row} with \
                         Horvitz-Thompson weight {row_weight}, expected 1"
                    ));
                }
                let (state, q_geom) = &mut acc;
                self.row_dynamic_q_geometry_into(row, block_states, q_geom)?;
                let (row_nll, f_pi, f_pipi) = if flex_active {
                    self.compute_row_flex_primary_gradient_hessian_exact(
                        row,
                        block_states,
                        q_geom,
                        &primary,
                    )?
                } else {
                    self.compute_row_primary_gradient_hessian_uncached(row, block_states)?
                };
                state.0 -= row_nll;
                self.accumulate_dynamic_q_joint_row(
                    row,
                    &slices,
                    q_geom,
                    f_pi.view(),
                    f_pipi.view(),
                    &identity_blocks,
                    &mut state.1,
                    &mut state.2,
                )?;
                Ok(acc)
            },
            |mut left, right| -> Result<_, String> {
                left.0.0 += right.0.0;
                left.0.1 += &right.0.1;
                left.0.2 += &right.0.2;
                Ok(left)
            },
        )?;
        Ok(final_acc.0)
    }

    pub(crate) fn evaluate_exact_newton_joint_dense(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<(f64, Array1<f64>, Array2<f64>), String> {
        if self.effective_flex_active(block_states)? || self.flex_timewiggle_active() {
            self.evaluate_exact_newton_joint_dynamic_q_dense(block_states)
        } else {
            in_slope_frame!(self, P, Frame, {
                let kern = SurvivalMarginalSlopeRowKernel::<P, Frame>::new(
                    self.clone(),
                    block_states.to_vec(),
                );
                let rows = crate::row_kernel::RowSet::All;
                let cache = build_row_kernel_cache(&kern, &rows)?;
                Ok((
                    row_kernel_log_likelihood(&cache, &rows),
                    -row_kernel_gradient(&kern, &cache, &rows),
                    row_kernel_hessian_dense(&kern, &cache, &rows)?,
                ))
            })
        }
    }

    pub(crate) fn evaluate_exact_newton_joint_gradient_dynamic_q(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<(f64, Array1<f64>), String> {
        let flex_active = self.effective_flex_active(block_states)?;
        let slices = block_slices(self, block_states);
        let primary = flex_primary_slices(self);
        let identity_blocks = if flex_active {
            flex_identity_block_pairs(&primary, &slices)
        } else {
            vec![]
        };
        type Acc = (f64, Array1<f64>);
        let make_acc = || -> Acc { (0.0, Array1::zeros(slices.total)) };

        gam_linalg::pairwise_reduce::par_deterministic_try_block_fold(
            self.n,
            |range| -> Result<_, String> {
                let mut acc = make_acc();
                for row in range {
                    let q_geom = self.row_dynamic_q_gradient(row, block_states)?;
                    let (row_nll, f_pi) = if flex_active {
                        self.compute_row_flex_primary_gradient_exact(
                            row,
                            block_states,
                            &q_geom,
                            &primary,
                        )?
                    } else {
                        let (nll, grad, _) =
                            self.compute_row_primary_gradient_hessian_uncached(row, block_states)?;
                        (nll, grad)
                    };
                    acc.0 -= row_nll;
                    self.accumulate_dynamic_q_core_gradient_first_order(
                        row,
                        &slices,
                        &q_geom,
                        f_pi.slice(s![0..self.core_primary_dimension()]),
                        &mut acc.1,
                    )?;
                    for (primary_range, joint_range) in &identity_blocks {
                        for local in 0..primary_range.len() {
                            acc.1[joint_range.start + local] -= f_pi[primary_range.start + local];
                        }
                    }
                    // The absorbed influence offset loads its coefficients through the row of
                    // `Z̃_infl` (#461), as accumulate_dynamic_q_joint_row pulls them back.
                    if flex_active
                        && let (Some(infl_primary), Some(infl_joint), Some(z_tilde)) = (
                            primary.infl,
                            slices.influence.as_ref(),
                            self.influence_absorber.as_ref(),
                        )
                    {
                        for (local, &z) in z_tilde.row(row).iter().enumerate() {
                            acc.1[infl_joint.start + local] -= f_pi[infl_primary] * z;
                        }
                    }
                }
                Ok(acc)
            },
            |mut left, right| -> Result<_, String> {
                left.0 += right.0;
                left.1 += &right.1;
                Ok(left)
            },
        )
        .map(|opt| opt.unwrap_or_else(make_acc))
    }

    /// Shared per-row pullback of the timewiggle q-map Jacobian/curvature
    /// derivatives into the joint Hessian accumulator.  Used by both the
    /// cached `_inner` path (with an empty `identity_blocks`) and the flex
    /// path (with non-empty `identity_blocks`); the only behavioural
    /// difference between those callers is whether the identity-block cross
    /// terms are added, which is driven entirely by `identity_blocks`.
    pub(crate) fn accumulate_timewiggle_directional_row(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        slices: &BlockSlices,
        q_geom: &SurvivalMarginalSlopeDynamicRow,
        f_pi: &Array1<f64>,
        h_pi: ArrayView2<'_, f64>,
        d_time: ndarray::ArrayView1<'_, f64>,
        d_marginal: ndarray::ArrayView1<'_, f64>,
        beta_time: &Array1<f64>,
        beta_time_w: ndarray::ArrayView1<'_, f64>,
        identity_blocks: &[(std::ops::Range<usize>, std::ops::Range<usize>)],
        acc: &mut Array2<f64>,
    ) -> Result<(), String> {
        let time_tail = self.time_wiggle_range();
        let p_base = time_tail.start;
        let p_time = slices.time.len();
        let p_marginal = slices.marginal.len();

        // ── Timewiggle Jacobian derivatives ────────────────
        let ec = self
            .design_entry
            .try_row_chunk(row..row + 1)
            .map_err(|e| format!("design_entry try_row_chunk: {e}"))?;
        let xc = self
            .design_exit
            .try_row_chunk(row..row + 1)
            .map_err(|e| format!("design_exit try_row_chunk: {e}"))?;
        let dc = self
            .design_derivative_exit
            .try_row_chunk(row..row + 1)
            .map_err(|e| format!("design_derivative_exit try_row_chunk: {e}"))?;
        let xe = ec.row(0).slice(s![..p_base]).to_owned();
        let xx = xc.row(0).slice(s![..p_base]).to_owned();
        let xd = dc.row(0).slice(s![..p_base]).to_owned();
        let mc = self
            .marginal_design
            .try_row_chunk(row..row + 1)
            .map_err(|e| format!("marginal_design try_row_chunk: {e}"))?;
        let mr = mc.row(0).to_owned();
        let dh0 = xe.dot(&d_time.slice(s![..p_base])) + mr.dot(&d_marginal);
        let dh1 = xx.dot(&d_time.slice(s![..p_base])) + mr.dot(&d_marginal);
        let ddr = xd.dot(&d_time.slice(s![..p_base]));
        let bm = block_states[1].eta[row];
        let h0 = xe.dot(&beta_time.slice(s![..p_base])) + self.offset_entry[row] + bm;
        let h1 = xx.dot(&beta_time.slice(s![..p_base])) + self.offset_exit[row] + bm;
        let dr = xd.dot(&beta_time.slice(s![..p_base])) + self.derivative_offset_exit[row];
        let eg = self
            .time_wiggle_geometry(Array1::from_vec(vec![h0]).view(), beta_time_w)?
            .ok_or_else(|| "timewiggle geometry missing at entry".to_string())?;
        let xg = self
            .time_wiggle_geometry(Array1::from_vec(vec![h1]).view(), beta_time_w)?
            .ok_or_else(|| "timewiggle geometry missing at exit".to_string())?;
        let (m2e, m3e) = (eg.d2q_dq02[0], eg.d3q_dq03[0]);
        let (m2x, m3x, m4x) = (xg.d2q_dq02[0], xg.d3q_dq03[0], xg.d4q_dq04[0]);
        // `m_k = Σ_l B_l^{(k)}(h)·γ_l` moves with the wiggle coefficients as well as with `h`:
        // along `d` it moves by `m_{k+1}·dh + Σ_l B_l^{(k)}(h)·dγ_l` (gam#2893).
        let d_wiggle = d_time.slice(s![time_tail.clone()]);
        let dm1e = m2e * dh0 + eg.basis_d1.row(0).dot(&d_wiggle);
        let dm2e = m3e * dh0 + eg.basis_d2.row(0).dot(&d_wiggle);
        let dm1x = m2x * dh1 + xg.basis_d1.row(0).dot(&d_wiggle);
        let dm2x = m3x * dh1 + xg.basis_d2.row(0).dot(&d_wiggle);
        let dm3x = m4x * dh1 + xg.basis_d3.row(0).dot(&d_wiggle);

        // dJ_{q,time}[a] / dβ[d]
        let mut dj0t = vec![0.0f64; p_time];
        let mut dj1t = vec![0.0f64; p_time];
        let mut djdt = vec![0.0f64; p_time];
        for a in 0..p_base {
            dj0t[a] = dm1e * xe[a];
            dj1t[a] = dm1x * xx[a];
            djdt[a] = dm2x * dr * xx[a] + m2x * ddr * xx[a] + dm1x * xd[a];
        }
        for li in 0..time_tail.len() {
            let ci = time_tail.start + li;
            dj0t[ci] = eg.basis_d1[[0, li]] * dh0;
            dj1t[ci] = xg.basis_d1[[0, li]] * dh1;
            djdt[ci] = xg.basis_d2[[0, li]] * dh1 * dr + xg.basis_d1[[0, li]] * ddr;
        }
        let djt = [&dj0t[..], &dj1t[..], &djdt[..]];
        let mut dj0m = vec![0.0f64; p_marginal];
        let mut dj1m = vec![0.0f64; p_marginal];
        let mut djdm = vec![0.0f64; p_marginal];
        for a in 0..p_marginal {
            dj0m[a] = dm1e * mr[a];
            dj1m[a] = dm1x * mr[a];
            djdm[a] = dm2x * dr * mr[a] + m2x * ddr * mr[a];
        }
        let djm = [&dj0m[..], &dj1m[..], &djdm[..]];
        let jt: [&Array1<f64>; 3] = [&q_geom.dq0_time, &q_geom.dq1_time, &q_geom.dqd1_time];
        let jm: [&Array1<f64>; 3] = [
            &q_geom.dq0_marginal,
            &q_geom.dq1_marginal,
            &q_geom.dqd1_marginal,
        ];

        // Term 2: (dJ/d)^T H J + J^T H (dJ/d)
        for a in 0..p_time {
            for b in 0..p_time {
                let mut v = 0.0;
                for qu in 0..3 {
                    for qv in 0..3 {
                        v += h_pi[[qu, qv]] * (djt[qu][a] * jt[qv][b] + jt[qu][a] * djt[qv][b]);
                    }
                }
                acc[[slices.time.start + a, slices.time.start + b]] += v;
            }
        }
        for a in 0..p_marginal {
            for b in 0..p_marginal {
                let mut v = 0.0;
                for qu in 0..3 {
                    for qv in 0..3 {
                        v += h_pi[[qu, qv]] * (djm[qu][a] * jm[qv][b] + jm[qu][a] * djm[qv][b]);
                    }
                }
                acc[[slices.marginal.start + a, slices.marginal.start + b]] += v;
            }
        }
        for a in 0..p_time {
            for b in 0..p_marginal {
                let mut v = 0.0;
                for qu in 0..3 {
                    for qv in 0..3 {
                        v += h_pi[[qu, qv]] * (djt[qu][a] * jm[qv][b] + jt[qu][a] * djm[qv][b]);
                    }
                }
                acc[[slices.time.start + a, slices.marginal.start + b]] += v;
                acc[[slices.marginal.start + b, slices.time.start + a]] += v;
            }
        }
        // Time×slope and marginal×slope: once per follow-up channel, each against
        // its own design row. A time-constant slope has one channel, so this is the
        // single `coefficient_design()` cross it was; a follow-up-varying slope has
        // three (gam#2767).
        for &(slope_primary, slope_design) in self.slope_layout.primary_channels().as_slice() {
            let gc = slope_design
                .try_row_chunk(row..row + 1)
                .map_err(|e| format!("slope_design try_row_chunk: {e}"))?;
            let gr = gc.row(0);
            for a in 0..p_time {
                let mut w = 0.0;
                for qu in 0..3 {
                    w += h_pi[[qu, slope_primary]] * djt[qu][a];
                }
                for b in 0..slices.slope.len() {
                    let v = w * gr[b];
                    acc[[slices.time.start + a, slices.slope.start + b]] += v;
                    acc[[slices.slope.start + b, slices.time.start + a]] += v;
                }
            }
            for a in 0..p_marginal {
                let mut w = 0.0;
                for qu in 0..3 {
                    w += h_pi[[qu, slope_primary]] * djm[qu][a];
                }
                for b in 0..slices.slope.len() {
                    let v = w * gr[b];
                    acc[[slices.marginal.start + a, slices.slope.start + b]] += v;
                    acc[[slices.slope.start + b, slices.marginal.start + a]] += v;
                }
            }
        }

        for (primary_range, joint_range) in identity_blocks {
            for local in 0..primary_range.len() {
                let primary_idx = primary_range.start + local;
                let joint_idx = joint_range.start + local;
                for a in 0..p_time {
                    let mut value = 0.0;
                    for qu in 0..3 {
                        value += h_pi[[qu, primary_idx]] * djt[qu][a];
                    }
                    acc[[slices.time.start + a, joint_idx]] += value;
                    acc[[joint_idx, slices.time.start + a]] += value;
                }
                for a in 0..p_marginal {
                    let mut value = 0.0;
                    for qu in 0..3 {
                        value += h_pi[[qu, primary_idx]] * djm[qu][a];
                    }
                    acc[[slices.marginal.start + a, joint_idx]] += value;
                    acc[[joint_idx, slices.marginal.start + a]] += value;
                }
            }
        }
        // The absorbed influence offset crosses the moving time and marginal Jacobians like an
        // identity primary, scaled by its row of `Z̃_infl` (#461).
        if let (Some(infl_primary), Some(infl_joint), Some(z_tilde)) = (
            flex_primary_slices(self).infl,
            slices.influence.as_ref(),
            self.influence_absorber.as_ref(),
        ) {
            for (local, &z) in z_tilde.row(row).iter().enumerate() {
                if z == 0.0 {
                    continue;
                }
                let joint_idx = infl_joint.start + local;
                for a in 0..p_time {
                    let mut value = 0.0;
                    for qu in 0..3 {
                        value += h_pi[[qu, infl_primary]] * djt[qu][a];
                    }
                    acc[[slices.time.start + a, joint_idx]] += value * z;
                    acc[[joint_idx, slices.time.start + a]] += value * z;
                }
                for a in 0..p_marginal {
                    let mut value = 0.0;
                    for qu in 0..3 {
                        value += h_pi[[qu, infl_primary]] * djm[qu][a];
                    }
                    acc[[slices.marginal.start + a, joint_idx]] += value * z;
                    acc[[joint_idx, slices.marginal.start + a]] += value * z;
                }
            }
        }

        // Term 4: Σ_r f_r dK_r/d
        for a in 0..p_base {
            for b in 0..p_base {
                let dk0 = dm2e * xe[a] * xe[b];
                let dk1 = dm2x * xx[a] * xx[b];
                let dkd = dm3x * dr * xx[a] * xx[b]
                    + m3x * ddr * xx[a] * xx[b]
                    + dm2x * (xx[a] * xd[b] + xd[a] * xx[b]);
                acc[[slices.time.start + a, slices.time.start + b]] +=
                    f_pi[0] * dk0 + f_pi[1] * dk1 + f_pi[2] * dkd;
            }
        }
        for li in 0..time_tail.len() {
            let ci = time_tail.start + li;
            for a in 0..p_base {
                let dk0 = eg.basis_d2[[0, li]] * dh0 * xe[a];
                let dk1 = xg.basis_d2[[0, li]] * dh1 * xx[a];
                let dkd = xg.basis_d3[[0, li]] * dh1 * dr * xx[a]
                    + xg.basis_d2[[0, li]] * ddr * xx[a]
                    + xg.basis_d2[[0, li]] * dh1 * xd[a];
                let v = f_pi[0] * dk0 + f_pi[1] * dk1 + f_pi[2] * dkd;
                acc[[slices.time.start + a, slices.time.start + ci]] += v;
                acc[[slices.time.start + ci, slices.time.start + a]] += v;
            }
        }
        for a in 0..p_base {
            for b in 0..p_marginal {
                let dk0 = dm2e * xe[a] * mr[b];
                let dk1 = dm2x * xx[a] * mr[b];
                let dkd = dm3x * dr * xx[a] * mr[b]
                    + m3x * ddr * xx[a] * mr[b]
                    + dm2x * xd[a] * mr[b];
                let v = f_pi[0] * dk0 + f_pi[1] * dk1 + f_pi[2] * dkd;
                acc[[slices.time.start + a, slices.marginal.start + b]] += v;
                acc[[slices.marginal.start + b, slices.time.start + a]] += v;
            }
        }
        for li in 0..time_tail.len() {
            let ci = time_tail.start + li;
            for b in 0..p_marginal {
                let dk0 = eg.basis_d2[[0, li]] * dh0 * mr[b];
                let dk1 = xg.basis_d2[[0, li]] * dh1 * mr[b];
                let dkd =
                    xg.basis_d3[[0, li]] * dh1 * dr * mr[b] + xg.basis_d2[[0, li]] * ddr * mr[b];
                let v = f_pi[0] * dk0 + f_pi[1] * dk1 + f_pi[2] * dkd;
                acc[[slices.time.start + ci, slices.marginal.start + b]] += v;
                acc[[slices.marginal.start + b, slices.time.start + ci]] += v;
            }
        }
        for a in 0..p_marginal {
            for b in 0..p_marginal {
                let dk0 = dm2e * mr[a] * mr[b];
                let dk1 = dm2x * mr[a] * mr[b];
                let dkd = dm3x * dr * mr[a] * mr[b] + m3x * ddr * mr[a] * mr[b];
                acc[[slices.marginal.start + a, slices.marginal.start + b]] +=
                    f_pi[0] * dk0 + f_pi[1] * dk1 + f_pi[2] * dkd;
            }
        }
        Ok(())
    }

    /// Exact directional derivative of the joint Hessian for timewiggle-only
    /// models (no score-warp / link-deviation).  Computes the derivative by
    /// differentiating the J^T H J + f·K pullback through the timewiggle
    /// q-map geometry (equation 47 of the unified pullback framework).
    pub(crate) fn exact_newton_joint_hessian_directional_derivative_timewiggle_inner(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_flat: &Array1<f64>,
        cache: Option<&EvalCache>,
    ) -> Result<Array2<f64>, String> {
        let slices = block_slices(self, block_states);
        let p_total = slices.total;
        let time_tail = self.time_wiggle_range();
        let d_time = d_beta_flat.slice(s![slices.time.clone()]);
        let d_marginal = d_beta_flat.slice(s![slices.marginal.clone()]);
        let beta_time = &block_states[0].beta;
        let beta_time_w = beta_time.slice(s![time_tail.clone()]);

        let result = gam_linalg::pairwise_reduce::par_deterministic_try_block_fold(
            self.n,
            |range| -> Result<Array2<f64>, String> {
                let mut acc = Array2::<f64>::zeros((p_total, p_total));
                for row in range {
                    let q_geom = self.row_dynamic_q_geometry(row, block_states)?;
                    let primary_owned;
                    let (f_pi, h_pi) = if let Some(cached) = cache {
                        self.row_primary_gradient_hessian(row, cached)
                    } else {
                        primary_owned =
                            self.compute_row_primary_gradient_hessian_uncached(row, block_states)?;
                        (&primary_owned.1, &primary_owned.2)
                    };
                    // Primary direction from the already-computed q_geom, in the
                    // family's own slope frame (one slope entry per follow-up
                    // channel, gam#2767).
                    let u_d = self.row_primary_direction_from_flat_dynamic_with_q_geometry(
                        row,
                        block_states,
                        &slices,
                        &q_geom,
                        d_beta_flat,
                    )?;
                    let t_ud = self.row_primary_third_contracted(row, block_states, u_d.view())?;
                    let h_ud = h_pi.dot(&u_d);

                    // Term 1 + 3: reuse core accumulator with (H·u^d, T[u^d])
                    self.accumulate_dynamic_q_core_hessian(
                        row,
                        &slices,
                        &q_geom,
                        h_ud.view(),
                        t_ud.view(),
                        &mut acc,
                    )?;

                    self.accumulate_timewiggle_directional_row(
                        row,
                        block_states,
                        &slices,
                        &q_geom,
                        f_pi,
                        h_pi.view(),
                        d_time,
                        d_marginal,
                        beta_time,
                        beta_time_w,
                        &[],
                        &mut acc,
                    )?;
                }
                Ok(acc)
            },
            |mut a, b| -> Result<_, String> {
                a += &b;
                Ok(a)
            },
        )?
        .unwrap_or_else(|| Array2::<f64>::zeros((p_total, p_total)));
        Ok(result)
    }

    /// Exact directional derivative of the joint Hessian for simultaneous
    /// timewiggle + flexible score/link warps.
    ///
    /// This extends the timewiggle-only transport by keeping the full flexible
    /// primary Hessian/third contraction live while only differentiating the
    /// q-geometry Jacobian and K tensors for the dynamic q coordinates. The
    /// score/link primary coordinates remain identity-mapped, so their
    /// contribution enters through the shared pullback term plus cross-columns
    /// against the dJ correction.
    pub(crate) fn exact_newton_joint_hessian_directional_derivative_timewiggle_flex(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_flat: &Array1<f64>,
    ) -> Result<Array2<f64>, String> {
        let slices = block_slices(self, block_states);
        let primary = flex_primary_slices(self);
        let identity_blocks = flex_identity_block_pairs(&primary, &slices);
        let p_total = slices.total;
        let time_tail = self.time_wiggle_range();
        let d_time = d_beta_flat.slice(s![slices.time.clone()]);
        let d_marginal = d_beta_flat.slice(s![slices.marginal.clone()]);
        let beta_time = &block_states[0].beta;
        let beta_time_w = beta_time.slice(s![time_tail.clone()]);

        let result = gam_linalg::pairwise_reduce::par_deterministic_try_block_fold(
            self.n,
            |range| -> Result<Array2<f64>, String> {
                let mut acc = Array2::<f64>::zeros((p_total, p_total));
                for row in range {
                    let q_geom = self.row_dynamic_q_geometry(row, block_states)?;
                    let (_, f_pi, h_pi) = self.compute_row_flex_primary_gradient_hessian_exact(
                        row,
                        block_states,
                        &q_geom,
                        &primary,
                    )?;
                    let u_d = self.row_primary_direction_from_flat_dynamic_with_q_geometry(
                        row,
                        block_states,
                        &slices,
                        &q_geom,
                        d_beta_flat,
                    )?;
                    let t_ud =
                        self.row_flex_primary_third_contracted_exact(row, block_states, &u_d)?;
                    let h_ud = h_pi.dot(&u_d);

                    self.accumulate_dynamic_q_joint_row(
                        row,
                        &slices,
                        &q_geom,
                        h_ud.view(),
                        t_ud.view(),
                        &identity_blocks,
                        &mut Array1::zeros(p_total),
                        &mut acc,
                    )?;

                    self.accumulate_timewiggle_directional_row(
                        row,
                        block_states,
                        &slices,
                        &q_geom,
                        &f_pi,
                        h_pi.view(),
                        d_time,
                        d_marginal,
                        beta_time,
                        beta_time_w,
                        &identity_blocks,
                        &mut acc,
                    )?;
                }
                Ok(acc)
            },
            |mut a, b| -> Result<_, String> {
                a += &b;
                Ok(a)
            },
        )?
        .unwrap_or_else(|| Array2::<f64>::zeros((p_total, p_total)));
        Ok(result)
    }

    /// Build-once all-axes variant of
    /// [`Self::exact_newton_joint_hessian_directional_derivative_timewiggle_flex`].
    ///
    /// The Jeffreys all-axes sweep asks for `D_β H[e_a]` along every coefficient axis. The
    /// single-direction routine rebuilds each row's `q`-geometry, flex primary Hessian and
    /// third-order base once per axis; this variant builds them once per row and contracts every
    /// axis against them through the same per-row assemblers as the single-axis call, so each
    /// matrix equals that call up to the cross-row reduction order (gam#2893).
    pub(crate) fn exact_newton_joint_hessian_directional_derivative_timewiggle_flex_all_axes(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<Vec<Array2<f64>>, String> {
        let slices = block_slices(self, block_states);
        let primary = flex_primary_slices(self);
        let identity_blocks = flex_identity_block_pairs(&primary, &slices);
        let p_total = slices.total;
        let time_tail = self.time_wiggle_range();
        let beta_time = &block_states[0].beta;
        let beta_time_w = beta_time.slice(s![time_tail.clone()]);
        let zeros = || vec![Array2::<f64>::zeros((p_total, p_total)); p_total];
        let result = gam_linalg::pairwise_reduce::par_deterministic_try_block_fold(
            self.n,
            |range| -> Result<Vec<Array2<f64>>, String> {
                let mut acc = zeros();
                // `accumulate_dynamic_q_joint_row` also scatters a gradient; the sweep reads
                // only the Hessian, so one scratch vector absorbs it.
                let mut gradient_scratch = Array1::<f64>::zeros(p_total);
                for row in range {
                    // Direction-independent per-row geometry, built once.
                    let q_geom = self.row_dynamic_q_geometry(row, block_states)?;
                    let (_, f_pi, h_pi) = self.compute_row_flex_primary_gradient_hessian_exact(
                        row,
                        block_states,
                        &q_geom,
                        &primary,
                    )?;
                    let base =
                        self.build_row_flex_third_base_with_states(row, block_states, &primary)?;
                    for axis_idx in 0..p_total {
                        let mut axis = Array1::<f64>::zeros(p_total);
                        axis[axis_idx] = 1.0;
                        let u_d = self.row_primary_direction_from_flat_dynamic_with_q_geometry(
                            row,
                            block_states,
                            &slices,
                            &q_geom,
                            &axis,
                        )?;
                        let t_ud = self.row_flex_third_contract_from_base(&base, &u_d)?;
                        let h_ud = h_pi.dot(&u_d);
                        self.accumulate_dynamic_q_joint_row(
                            row,
                            &slices,
                            &q_geom,
                            h_ud.view(),
                            t_ud.view(),
                            &identity_blocks,
                            &mut gradient_scratch,
                            &mut acc[axis_idx],
                        )?;
                        self.accumulate_timewiggle_directional_row(
                            row,
                            block_states,
                            &slices,
                            &q_geom,
                            &f_pi,
                            h_pi.view(),
                            axis.slice(s![slices.time.clone()]),
                            axis.slice(s![slices.marginal.clone()]),
                            beta_time,
                            beta_time_w,
                            &identity_blocks,
                            &mut acc[axis_idx],
                        )?;
                    }
                }
                Ok(acc)
            },
            |mut a, b| -> Result<_, String> {
                for (ai, bi) in a.iter_mut().zip(b.into_iter()) {
                    *ai += &bi;
                }
                Ok(a)
            },
        )?
        .unwrap_or_else(zeros);
        Ok(result)
    }

    pub(crate) fn exact_newton_joint_hessian_directional_derivative_timewiggle_cached(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_flat: &Array1<f64>,
        cache: &EvalCache,
    ) -> Result<Array2<f64>, String> {
        self.exact_newton_joint_hessian_directional_derivative_timewiggle_inner(
            block_states,
            d_beta_flat,
            Some(cache),
        )
    }

    pub(crate) fn exact_newton_joint_hessian_directional_derivative_timewiggle(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_flat: &Array1<f64>,
    ) -> Result<Array2<f64>, String> {
        self.exact_newton_joint_hessian_directional_derivative_timewiggle_inner(
            block_states,
            d_beta_flat,
            None,
        )
    }
    /// Fully exact second directional derivative D²H[d,e] for a time wiggle, with or
    /// without the flexible score/link warps. Differentiates DH[e] along d analytically
    /// using m₂–m₅ scalars and the wiggle basis derivatives: each `m_k = Σ_l B_l^{(k)}(h)·γ_l`
    /// moves with the wiggle coefficients as well as with `h` (gam#2893).
    ///
    /// D²H[d,e] = J^T Ψ J  +  Σ γ_r K_r
    ///   + Σ bilinear(W_k, left_k, right_k)  for k in {T_e×dJ_d, T_d×dJ_e, H×d²J, H×dJ_d×dJ_e}
    ///   + dK cross-terms: (Hu_d)·dK_e + (Hu_e)·dK_d + f·d²K
    ///
    /// where Ψ = Q[u_d,u_e] + T[du_e/dd], γ = T_d·u_e + H·du_e/dd.
    ///
    /// With a score warp or link deviation the primaries carry the flex coordinates:
    /// `H`, `T` and `Q` are the flex primary derivatives, and those coordinates are
    /// identity-mapped, so `dJ`, `d²J` and `dK` keep q rows only while `J^T Ψ J` and
    /// every `dJ^T W J` term also reach the identity blocks (gam#2893).
    pub(crate) fn exact_newton_joint_hessiansecond_directional_derivative_timewiggle(
        &self,
        block_states: &[ParameterBlockState],
        d_u: &Array1<f64>,
        d_v: &Array1<f64>,
    ) -> Result<Array2<f64>, String> {
        let slices = block_slices(self, block_states);
        let p_total = slices.total;
        let p_time = slices.time.len();
        let p_marginal = slices.marginal.len();
        let time_tail = self.time_wiggle_range();
        let p_base = time_tail.start;
        let du_t = d_u.slice(s![slices.time.clone()]);
        let du_m = d_u.slice(s![slices.marginal.clone()]);
        let dv_t = d_v.slice(s![slices.time.clone()]);
        let dv_m = d_v.slice(s![slices.marginal.clone()]);
        let beta_time = &block_states[0].beta;
        let beta_tw = beta_time.slice(s![time_tail.clone()]);
        let flex_primary = self
            .effective_flex_active(block_states)?
            .then(|| flex_primary_slices(self));
        let identity_blocks = flex_primary
            .as_ref()
            .map_or_else(Vec::new, |primary| flex_identity_block_pairs(primary, &slices));

        let result = gam_linalg::pairwise_reduce::par_deterministic_try_block_fold(
            self.n,
            |range| -> Result<Array2<f64>, String> {
                let mut acc = Array2::<f64>::zeros((p_total, p_total));
                for row in range {
                    let q_geom = self.row_dynamic_q_geometry(row, block_states)?;
                    let (f_pi, h_pi, ud, ue, t_d, t_e, q_de, flex_base) = if let Some(primary) =
                        flex_primary.as_ref()
                    {
                        let (_, f_pi, h_pi) = self.compute_row_flex_primary_gradient_hessian_exact(
                            row,
                            block_states,
                            &q_geom,
                            primary,
                        )?;
                        let ud = self.row_primary_direction_from_flat_dynamic_with_q_geometry(
                            row,
                            block_states,
                            &slices,
                            &q_geom,
                            d_u,
                        )?;
                        let ue = self.row_primary_direction_from_flat_dynamic_with_q_geometry(
                            row,
                            block_states,
                            &slices,
                            &q_geom,
                            d_v,
                        )?;
                        // One direction-independent row base serves every contraction of
                        // this row; a repeated direction reuses its third contraction.
                        let base =
                            self.build_row_flex_third_base_with_states(row, block_states, primary)?;
                        let t_d = self.row_flex_third_contract_from_base(&base, &ud)?;
                        let t_e = if ue == ud {
                            t_d.clone()
                        } else {
                            self.row_flex_third_contract_from_base(&base, &ue)?
                        };
                        let q_de = self.row_flex_fourth_contract_from_base(&base, &ud, &ue)?;
                        (f_pi, h_pi, ud, ue, t_d, t_e, q_de, Some(base))
                    } else {
                        let (_, f_pi, h_pi) =
                            self.compute_row_primary_gradient_hessian_uncached(row, block_states)?;

                        // Primary directions, in the family's own slope frame.
                        let ud = self.row_primary_direction_from_flat_dynamic_with_q_geometry(
                            row,
                            block_states,
                            &slices,
                            &q_geom,
                            d_u,
                        )?;
                        let ue = self.row_primary_direction_from_flat_dynamic_with_q_geometry(
                            row,
                            block_states,
                            &slices,
                            &q_geom,
                            d_v,
                        )?;

                        let t_d = self.row_primary_third_contracted(row, block_states, ud.view())?;
                        let t_e = self.row_primary_third_contracted(row, block_states, ue.view())?;
                        let q_de = self.row_primary_fourth_contracted(
                            row,
                            block_states,
                            ud.view(),
                            ue.view(),
                        )?;
                        (f_pi, h_pi, ud, ue, t_d, t_e, q_de, None)
                    };
                    let h_ud = h_pi.dot(&ud);
                    let h_ue = h_pi.dot(&ue);

                    // ── Timewiggle geometry ─────────────────────────────
                    let ec = self
                        .design_entry
                        .try_row_chunk(row..row + 1)
                        .map_err(|e| format!("design_entry try_row_chunk: {e}"))?;
                    let xc = self
                        .design_exit
                        .try_row_chunk(row..row + 1)
                        .map_err(|e| format!("design_exit try_row_chunk: {e}"))?;
                    let dc = self
                        .design_derivative_exit
                        .try_row_chunk(row..row + 1)
                        .map_err(|e| format!("design_derivative_exit try_row_chunk: {e}"))?;
                    let xe = ec.row(0).slice(s![..p_base]).to_owned();
                    let xx = xc.row(0).slice(s![..p_base]).to_owned();
                    let xd = dc.row(0).slice(s![..p_base]).to_owned();
                    let mc = self
                        .marginal_design
                        .try_row_chunk(row..row + 1)
                        .map_err(|e| format!("marginal_design try_row_chunk: {e}"))?;
                    let mr = mc.row(0).to_owned();

                    let bm = block_states[1].eta[row];
                    let h0 = xe.dot(&beta_time.slice(s![..p_base])) + self.offset_entry[row] + bm;
                    let h1 = xx.dot(&beta_time.slice(s![..p_base])) + self.offset_exit[row] + bm;
                    let dr =
                        xd.dot(&beta_time.slice(s![..p_base])) + self.derivative_offset_exit[row];

                    let eg = self
                        .time_wiggle_geometry(Array1::from_vec(vec![h0]).view(), beta_tw)?
                        .ok_or_else(|| "timewiggle geometry missing".to_string())?;
                    let xg = self
                        .time_wiggle_geometry(Array1::from_vec(vec![h1]).view(), beta_tw)?
                        .ok_or_else(|| "timewiggle geometry missing".to_string())?;

                    let m2_en = eg.d2q_dq02[0];
                    let m3_en = eg.d3q_dq03[0];
                    let m4_en = eg.d4q_dq04[0];
                    let m2_ex = xg.d2q_dq02[0];
                    let m3_ex = xg.d3q_dq03[0];
                    let m4_ex = xg.d4q_dq04[0];
                    let m5_ex = xg.d5q_dq05[0];

                    // Direction scalars (h linear in β ⇒ d²h/ded = 0)
                    let dh0d = xe.dot(&du_t.slice(s![..p_base])) + mr.dot(&du_m);
                    let dh1d = xx.dot(&du_t.slice(s![..p_base])) + mr.dot(&du_m);
                    let ddrd = xd.dot(&du_t.slice(s![..p_base]));
                    let dh0e = xe.dot(&dv_t.slice(s![..p_base])) + mr.dot(&dv_m);
                    let dh1e = xx.dot(&dv_t.slice(s![..p_base])) + mr.dot(&dv_m);
                    let ddre = xd.dot(&dv_t.slice(s![..p_base]));

                    // Moves of `m_k` along d, along e, and along d then e. `m_k` is linear in
                    // the wiggle coefficients and `h` is linear in β, so
                    //   dm_k[d]    = m_{k+1}·dh_d + B^{(k)}·dγ_d
                    //   d²m_k[d,e] = m_{k+2}·dh_d·dh_e + B^{(k+1)}·dγ_d·dh_e + B^{(k+1)}·dγ_e·dh_d
                    let dwd = du_t.slice(s![time_tail.clone()]);
                    let dwe = dv_t.slice(s![time_tail.clone()]);
                    let (b1e_d, b2e_d, b3e_d) = (
                        eg.basis_d1.row(0).dot(&dwd),
                        eg.basis_d2.row(0).dot(&dwd),
                        eg.basis_d3.row(0).dot(&dwd),
                    );
                    let (b1e_e, b2e_e, b3e_e) = (
                        eg.basis_d1.row(0).dot(&dwe),
                        eg.basis_d2.row(0).dot(&dwe),
                        eg.basis_d3.row(0).dot(&dwe),
                    );
                    let (b1x_d, b2x_d, b3x_d, b4x_d) = (
                        xg.basis_d1.row(0).dot(&dwd),
                        xg.basis_d2.row(0).dot(&dwd),
                        xg.basis_d3.row(0).dot(&dwd),
                        xg.basis_d4.row(0).dot(&dwd),
                    );
                    let (b1x_e, b2x_e, b3x_e, b4x_e) = (
                        xg.basis_d1.row(0).dot(&dwe),
                        xg.basis_d2.row(0).dot(&dwe),
                        xg.basis_d3.row(0).dot(&dwe),
                        xg.basis_d4.row(0).dot(&dwe),
                    );
                    let dm1_en_d = m2_en * dh0d + b1e_d;
                    let dm2_en_d = m3_en * dh0d + b2e_d;
                    let dm1_en_e = m2_en * dh0e + b1e_e;
                    let dm2_en_e = m3_en * dh0e + b2e_e;
                    let dm1_ex_d = m2_ex * dh1d + b1x_d;
                    let dm2_ex_d = m3_ex * dh1d + b2x_d;
                    let dm3_ex_d = m4_ex * dh1d + b3x_d;
                    let dm1_ex_e = m2_ex * dh1e + b1x_e;
                    let dm2_ex_e = m3_ex * dh1e + b2x_e;
                    let dm3_ex_e = m4_ex * dh1e + b3x_e;
                    let d2m1_en = m3_en * dh0d * dh0e + b2e_d * dh0e + b2e_e * dh0d;
                    let d2m2_en = m4_en * dh0d * dh0e + b3e_d * dh0e + b3e_e * dh0d;
                    let d2m1_ex = m3_ex * dh1d * dh1e + b2x_d * dh1e + b2x_e * dh1d;
                    let d2m2_ex = m4_ex * dh1d * dh1e + b3x_d * dh1e + b3x_e * dh1d;
                    let d2m3_ex = m5_ex * dh1d * dh1e + b4x_d * dh1e + b4x_e * dh1d;

                    // du_e/dd = (dJ/dd)·e_v — primary direction of e perturbed by d
                    // dJ[q0,time_a]/dd = dm1_en_d*xe[a] for base, basis_d1*dh0d for wiggle
                    let due_d = {
                        let mut v = [0.0f64; 4];
                        for a in 0..p_base {
                            v[0] += dm1_en_d * xe[a] * dv_t[a];
                            v[1] += dm1_ex_d * xx[a] * dv_t[a];
                            v[2] += (dm2_ex_d * dr * xx[a]
                                + m2_ex * ddrd * xx[a]
                                + dm1_ex_d * xd[a])
                                * dv_t[a];
                        }
                        for li in 0..time_tail.len() {
                            let ci = time_tail.start + li;
                            v[0] += eg.basis_d1[[0, li]] * dh0d * dv_t[ci];
                            v[1] += xg.basis_d1[[0, li]] * dh1d * dv_t[ci];
                            v[2] += (xg.basis_d2[[0, li]] * dh1d * dr
                                + xg.basis_d1[[0, li]] * ddrd)
                                * dv_t[ci];
                        }
                        for a in 0..p_marginal {
                            v[0] += dm1_en_d * mr[a] * dv_m[a];
                            v[1] += dm1_ex_d * mr[a] * dv_m[a];
                            v[2] += (dm2_ex_d * dr * mr[a] + m2_ex * ddrd * mr[a]) * dv_m[a];
                        }
                        // v[3] = 0 (slope J is constant), and the identity-mapped flex
                        // coordinates have a constant J as well.
                        let mut due_d = Array1::<f64>::zeros(h_pi.nrows());
                        for (index, value) in v.iter().enumerate() {
                            due_d[index] = *value;
                        }
                        due_d
                    };

                    // Ψ = Q[ud,ue] + T[due_d]
                    let t_due = if let Some(base) = flex_base.as_ref() {
                        self.row_flex_third_contract_from_base(base, &due_d)?
                    } else {
                        self.row_primary_third_contracted(row, block_states, due_d.view())?
                    };
                    let psi = &q_de + &t_due;

                    // γ = T_d·ue + H·due_d
                    let gamma = &t_d.dot(&ue) + &h_pi.dot(&due_d);

                    // ── Term A: J^T Ψ J + γ·K ─────────────────────────
                    self.accumulate_directional_joint_hessian_row(
                        row,
                        &slices,
                        &q_geom,
                        &identity_blocks,
                        gamma.view(),
                        psi.view(),
                        &mut acc,
                    )?;

                    let jt = [&q_geom.dq0_time, &q_geom.dq1_time, &q_geom.dqd1_time];
                    let jm = [
                        &q_geom.dq0_marginal,
                        &q_geom.dq1_marginal,
                        &q_geom.dqd1_marginal,
                    ];
                    // One design row per slope follow-up channel: one on a time-constant
                    // slope, three on a follow-up-varying one (gam#2767).
                    let slope_chunks = self
                        .slope_layout
                        .primary_channels()
                        .as_slice()
                        .iter()
                        .map(|&(primary, design)| {
                            design
                                .try_row_chunk(row..row + 1)
                                .map(|chunk| (primary, chunk))
                                .map_err(|e| format!("slope_design try_row_chunk: {e}"))
                        })
                        .collect::<Result<Vec<_>, String>>()?;

                    // ── Helper: accumulate a symmetrized bilinear term ──
                    // Adds Σ W[qu,qv] * (left[qu,a]*right[qv,b] + right[qu,a]*left[qv,b])
                    // for all block pairs into acc.
                    macro_rules! accum_bilinear {
                        ($w:expr, $lt:expr, $lm:expr, $rt:expr, $rm:expr) => {
                            for a in 0..p_time {
                                for b in 0..p_time {
                                    let mut v = 0.0;
                                    for qu in 0..3 {
                                        for qv in 0..3 {
                                            v += $w[[qu, qv]]
                                                * ($lt[qu][a] * $rt[qv][b]
                                                    + $rt[qu][a] * $lt[qv][b]);
                                        }
                                    }
                                    acc[[slices.time.start + a, slices.time.start + b]] += v;
                                }
                            }
                            for a in 0..p_marginal {
                                for b in 0..p_marginal {
                                    let mut v = 0.0;
                                    for qu in 0..3 {
                                        for qv in 0..3 {
                                            v += $w[[qu, qv]]
                                                * ($lm[qu][a] * $rm[qv][b]
                                                    + $rm[qu][a] * $lm[qv][b]);
                                        }
                                    }
                                    acc[[slices.marginal.start + a, slices.marginal.start + b]] +=
                                        v;
                                }
                            }
                            for a in 0..p_time {
                                for b in 0..p_marginal {
                                    let mut v = 0.0;
                                    for qu in 0..3 {
                                        for qv in 0..3 {
                                            v += $w[[qu, qv]]
                                                * ($lt[qu][a] * $rm[qv][b]
                                                    + $rt[qu][a] * $lm[qv][b]);
                                        }
                                    }
                                    acc[[slices.time.start + a, slices.marginal.start + b]] += v;
                                    acc[[slices.marginal.start + b, slices.time.start + a]] += v;
                                }
                            }
                        };
                    }

                    // `dJ` has q rows only, while `J` also carries the slope design and
                    // the identity-mapped flex coordinates: a `dJ^T W J` term reaches
                    // those columns and a `dJ^T W dJ` term does not.
                    macro_rules! accum_j_cross {
                        ($w:expr, $lt:expr, $lm:expr) => {
                            for (slope_primary, slope_chunk) in &slope_chunks {
                                let gr = slope_chunk.row(0);
                                for a in 0..p_time {
                                    let mut w2 = 0.0;
                                    for qu in 0..3 {
                                        w2 += $w[[qu, *slope_primary]] * $lt[qu][a];
                                    }
                                    for b in 0..slices.slope.len() {
                                        let v = w2 * gr[b];
                                        acc[[slices.time.start + a, slices.slope.start + b]] += v;
                                        acc[[slices.slope.start + b, slices.time.start + a]] += v;
                                    }
                                }
                                for a in 0..p_marginal {
                                    let mut w2 = 0.0;
                                    for qu in 0..3 {
                                        w2 += $w[[qu, *slope_primary]] * $lm[qu][a];
                                    }
                                    for b in 0..slices.slope.len() {
                                        let v = w2 * gr[b];
                                        acc[[slices.marginal.start + a, slices.slope.start + b]] +=
                                            v;
                                        acc[[slices.slope.start + b, slices.marginal.start + a]] +=
                                            v;
                                    }
                                }
                            }
                            for (primary_range, joint_range) in identity_blocks.iter() {
                                for local in 0..primary_range.len() {
                                    let primary_idx = primary_range.start + local;
                                    let joint_idx = joint_range.start + local;
                                    for a in 0..p_time {
                                        let mut w2 = 0.0;
                                        for qu in 0..3 {
                                            w2 += $w[[qu, primary_idx]] * $lt[qu][a];
                                        }
                                        acc[[slices.time.start + a, joint_idx]] += w2;
                                        acc[[joint_idx, slices.time.start + a]] += w2;
                                    }
                                    for a in 0..p_marginal {
                                        let mut w2 = 0.0;
                                        for qu in 0..3 {
                                            w2 += $w[[qu, primary_idx]] * $lm[qu][a];
                                        }
                                        acc[[slices.marginal.start + a, joint_idx]] += w2;
                                        acc[[joint_idx, slices.marginal.start + a]] += w2;
                                    }
                                }
                            }
                            // The absorbed influence offset crosses the moved Jacobians like an
                            // identity primary, scaled by its row of `Z̃_infl` (#461).
                            if let (Some(infl_primary), Some(infl_joint), Some(z_tilde)) = (
                                flex_primary_slices(self).infl,
                                slices.influence.as_ref(),
                                self.influence_absorber.as_ref(),
                            ) {
                                for (local, &z) in z_tilde.row(row).iter().enumerate() {
                                    if z == 0.0 {
                                        continue;
                                    }
                                    let joint_idx = infl_joint.start + local;
                                    for a in 0..p_time {
                                        let mut w2 = 0.0;
                                        for qu in 0..3 {
                                            w2 += $w[[qu, infl_primary]] * $lt[qu][a];
                                        }
                                        acc[[slices.time.start + a, joint_idx]] += w2 * z;
                                        acc[[joint_idx, slices.time.start + a]] += w2 * z;
                                    }
                                    for a in 0..p_marginal {
                                        let mut w2 = 0.0;
                                        for qu in 0..3 {
                                            w2 += $w[[qu, infl_primary]] * $lm[qu][a];
                                        }
                                        acc[[slices.marginal.start + a, joint_idx]] += w2 * z;
                                        acc[[joint_idx, slices.marginal.start + a]] += w2 * z;
                                    }
                                }
                            }
                        };
                    }

                    // ── Build dJ arrays for both directions ────────────
                    // (same code as first directional, for d and e)
                    type DjArrays = (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>);
                    // `dm1_en`, `dm1_ex` and `dm2_ex` are the direction's moves of `m₁` at entry
                    // and exit and of `m₂` at exit.
                    let build_dj = |dh0: f64,
                                    dh1: f64,
                                    ddr_val: f64,
                                    dm1_en: f64,
                                    dm1_ex: f64,
                                    dm2_ex: f64|
                     -> DjArrays {
                        let mut j0t = vec![0.0f64; p_time];
                        let mut j1t = vec![0.0f64; p_time];
                        let mut jdt = vec![0.0f64; p_time];
                        for a in 0..p_base {
                            j0t[a] = dm1_en * xe[a];
                            j1t[a] = dm1_ex * xx[a];
                            jdt[a] = dm2_ex * dr * xx[a]
                                + m2_ex * ddr_val * xx[a]
                                + dm1_ex * xd[a];
                        }
                        for li in 0..time_tail.len() {
                            let ci = time_tail.start + li;
                            j0t[ci] = eg.basis_d1[[0, li]] * dh0;
                            j1t[ci] = xg.basis_d1[[0, li]] * dh1;
                            jdt[ci] =
                                xg.basis_d2[[0, li]] * dh1 * dr + xg.basis_d1[[0, li]] * ddr_val;
                        }
                        let mut j0m = vec![0.0f64; p_marginal];
                        let mut j1m = vec![0.0f64; p_marginal];
                        let mut jdm = vec![0.0f64; p_marginal];
                        for a in 0..p_marginal {
                            j0m[a] = dm1_en * mr[a];
                            j1m[a] = dm1_ex * mr[a];
                            jdm[a] = dm2_ex * dr * mr[a] + m2_ex * ddr_val * mr[a];
                        }
                        (j0t, j1t, jdt, j0m, j1m, jdm)
                    };

                    let (djd0t, djd1t, djddt, djd0m, djd1m, djddm) =
                        build_dj(dh0d, dh1d, ddrd, dm1_en_d, dm1_ex_d, dm2_ex_d);
                    let djd_t = [&djd0t[..], &djd1t[..], &djddt[..]];
                    let djd_m = [&djd0m[..], &djd1m[..], &djddm[..]];

                    let (dje0t, dje1t, djedt, dje0m, dje1m, djedm) =
                        build_dj(dh0e, dh1e, ddre, dm1_en_e, dm1_ex_e, dm2_ex_e);
                    let dje_t = [&dje0t[..], &dje1t[..], &djedt[..]];
                    let dje_m = [&dje0m[..], &dje1m[..], &djedm[..]];

                    // ── d²J/ded (cross derivative, h linear ⇒ d²h=0) ──
                    let mut d2j0t = vec![0.0f64; p_time];
                    let mut d2j1t = vec![0.0f64; p_time];
                    let mut d2jdt = vec![0.0f64; p_time];
                    for a in 0..p_base {
                        d2j0t[a] = d2m1_en * xe[a];
                        d2j1t[a] = d2m1_ex * xx[a];
                        d2jdt[a] = d2m2_ex * dr * xx[a]
                            + (dm2_ex_d * ddre + dm2_ex_e * ddrd) * xx[a]
                            + d2m1_ex * xd[a];
                    }
                    for li in 0..time_tail.len() {
                        let ci = time_tail.start + li;
                        d2j0t[ci] = eg.basis_d2[[0, li]] * dh0d * dh0e;
                        d2j1t[ci] = xg.basis_d2[[0, li]] * dh1d * dh1e;
                        d2jdt[ci] = xg.basis_d3[[0, li]] * dh1d * dh1e * dr
                            + xg.basis_d2[[0, li]] * (dh1d * ddre + dh1e * ddrd);
                    }
                    let d2j_t = [&d2j0t[..], &d2j1t[..], &d2jdt[..]];
                    let mut d2j0m = vec![0.0f64; p_marginal];
                    let mut d2j1m = vec![0.0f64; p_marginal];
                    let mut d2jdm = vec![0.0f64; p_marginal];
                    for a in 0..p_marginal {
                        d2j0m[a] = d2m1_en * mr[a];
                        d2j1m[a] = d2m1_ex * mr[a];
                        d2jdm[a] =
                            d2m2_ex * dr * mr[a] + (dm2_ex_d * ddre + dm2_ex_e * ddrd) * mr[a];
                    }
                    let d2j_m = [&d2j0m[..], &d2j1m[..], &d2jdm[..]];

                    // `jt`/`jm` borrow owned `Array1` jet fields of `q_geom`,
                    // which are contiguous, so `as_slice` cannot return None.
                    let contiguous = "q_geom jets are owned contiguous Array1 buffers";
                    let jt_s: [&[f64]; 3] = [
                        jt[0].as_slice().expect(contiguous),
                        jt[1].as_slice().expect(contiguous),
                        jt[2].as_slice().expect(contiguous),
                    ];
                    let jm_s: [&[f64]; 3] = [
                        jm[0].as_slice().expect(contiguous),
                        jm[1].as_slice().expect(contiguous),
                        jm[2].as_slice().expect(contiguous),
                    ];

                    // ── Term B: bilinear cross-terms ────────────────────
                    // (dJ_d)^T T_e J + J^T T_e (dJ_d) — differentiated from Term 1
                    accum_bilinear!(t_e, djd_t, djd_m, jt_s, jm_s);
                    accum_j_cross!(t_e, djd_t, djd_m);
                    // (dJ_e)^T T_d J + J^T T_d (dJ_e) — symmetry partner
                    accum_bilinear!(t_d, dje_t, dje_m, jt_s, jm_s);
                    accum_j_cross!(t_d, dje_t, dje_m);
                    // (d²J)^T H J + J^T H (d²J) — from Term 2
                    accum_bilinear!(h_pi, d2j_t, d2j_m, jt_s, jm_s);
                    accum_j_cross!(h_pi, d2j_t, d2j_m);
                    // (dJ_d)^T H (dJ_e) + (dJ_e)^T H (dJ_d) — from Term 2. Neither factor
                    // has a slope or flex row, so this term stays in the time/marginal blocks.
                    accum_bilinear!(h_pi, djd_t, djd_m, dje_t, dje_m);

                    // ── Term C: dK cross-terms ──────────────────────────
                    // (H·ud)_r dK_r/de + (H·ue)_r dK_r/dd + f_r d²K_r/ded
                    //
                    // dK[q,a,b]/dd = d(K[q,a,b])/dd where K = m_{k}*product-of-design-rows
                    // d²K[q,a,b]/ded moves each m_k twice (d²h/ded = 0, m_k linear in γ)
                    //
                    // For q0 base×base: K = m2_en*xe[a]*xe[b]
                    //   dK/dd = dm2_en_d*xe[a]*xe[b]
                    //   d²K/ded = d2m2_en*xe[a]*xe[b]
                    // For q1 base×base: K = m2_ex*xx[a]*xx[b]
                    //   dK/dd = dm2_ex_d*xx[a]*xx[b]
                    //   d²K/ded = d2m2_ex*xx[a]*xx[b]
                    // For qd1 base×base: K = m3_ex*dr*xx[a]*xx[b] + m2_ex*(xx[a]*xd[b]+xd[a]*xx[b])
                    //   dK/dd = dm3_ex_d*dr*xx[a]*xx[b] + m3_ex*ddrd*xx[a]*xx[b]
                    //         + dm2_ex_d*(xx[a]*xd[b]+xd[a]*xx[b])
                    //   d²K/ded = d2m3_ex*dr*xx[a]*xx[b]
                    //           + (dm3_ex_d*ddre+dm3_ex_e*ddrd)*xx[a]*xx[b]
                    //           + d2m2_ex*(xx[a]*xd[b]+xd[a]*xx[b])

                    // base×base time×time
                    for a in 0..p_base {
                        for b in 0..p_base {
                            let dke_0 = dm2_en_e * xe[a] * xe[b];
                            let dke_1 = dm2_ex_e * xx[a] * xx[b];
                            let dke_d = dm3_ex_e * dr * xx[a] * xx[b]
                                + m3_ex * ddre * xx[a] * xx[b]
                                + dm2_ex_e * (xx[a] * xd[b] + xd[a] * xx[b]);
                            let dkd_0 = dm2_en_d * xe[a] * xe[b];
                            let dkd_1 = dm2_ex_d * xx[a] * xx[b];
                            let dkd_d = dm3_ex_d * dr * xx[a] * xx[b]
                                + m3_ex * ddrd * xx[a] * xx[b]
                                + dm2_ex_d * (xx[a] * xd[b] + xd[a] * xx[b]);
                            let d2k_0 = d2m2_en * xe[a] * xe[b];
                            let d2k_1 = d2m2_ex * xx[a] * xx[b];
                            let d2k_d = d2m3_ex * dr * xx[a] * xx[b]
                                + (dm3_ex_d * ddre + dm3_ex_e * ddrd) * xx[a] * xx[b]
                                + d2m2_ex * (xx[a] * xd[b] + xd[a] * xx[b]);
                            acc[[slices.time.start + a, slices.time.start + b]] += h_ud[0] * dke_0
                                + h_ud[1] * dke_1
                                + h_ud[2] * dke_d
                                + h_ue[0] * dkd_0
                                + h_ue[1] * dkd_1
                                + h_ue[2] * dkd_d
                                + f_pi[0] * d2k_0
                                + f_pi[1] * d2k_1
                                + f_pi[2] * d2k_d;
                        }
                    }

                    // base×wiggle time×time
                    for li in 0..time_tail.len() {
                        let ci = time_tail.start + li;
                        for a in 0..p_base {
                            // K[q0] for base×wiggle: d2q0/dβ_base[a] dβ_wiggle[li]
                            //   = basis_d1[li]*xe[a] at entry  (m2 * x * basis is wrong; correct is basis_d1*x)
                            // Actually from q_geom: d2q0_time_time[[a, ci]] = basis_d1_entry[li]*xe[a]
                            // dK/dd = basis_d2[li]*dh0d*xe[a]
                            // d²K/ded = basis_d3[li]*dh0d*dh0e*xe[a]
                            let dke_0 = eg.basis_d2[[0, li]] * dh0e * xe[a];
                            let dke_1 = xg.basis_d2[[0, li]] * dh1e * xx[a];
                            let dke_d = xg.basis_d3[[0, li]] * dh1e * dr * xx[a]
                                + xg.basis_d2[[0, li]] * ddre * xx[a]
                                + xg.basis_d2[[0, li]] * dh1e * xd[a];
                            let dkd_0 = eg.basis_d2[[0, li]] * dh0d * xe[a];
                            let dkd_1 = xg.basis_d2[[0, li]] * dh1d * xx[a];
                            let dkd_d = xg.basis_d3[[0, li]] * dh1d * dr * xx[a]
                                + xg.basis_d2[[0, li]] * ddrd * xx[a]
                                + xg.basis_d2[[0, li]] * dh1d * xd[a];
                            let d2k_0 = eg.basis_d3[[0, li]] * dh0d * dh0e * xe[a];
                            let d2k_1 = xg.basis_d3[[0, li]] * dh1d * dh1e * xx[a];
                            let d2k_d = xg.basis_d4[[0, li]] * dh1d * dh1e * dr * xx[a]
                                + xg.basis_d3[[0, li]] * (dh1d * ddre + dh1e * ddrd) * xx[a]
                                + xg.basis_d3[[0, li]] * dh1d * dh1e * xd[a];
                            let v = h_ud[0] * dke_0
                                + h_ud[1] * dke_1
                                + h_ud[2] * dke_d
                                + h_ue[0] * dkd_0
                                + h_ue[1] * dkd_1
                                + h_ue[2] * dkd_d
                                + f_pi[0] * d2k_0
                                + f_pi[1] * d2k_1
                                + f_pi[2] * d2k_d;
                            acc[[slices.time.start + a, slices.time.start + ci]] += v;
                            acc[[slices.time.start + ci, slices.time.start + a]] += v;
                        }
                    }

                    // base×marginal time×marginal
                    for a in 0..p_base {
                        for b in 0..p_marginal {
                            let dke_0 = dm2_en_e * xe[a] * mr[b];
                            let dke_1 = dm2_ex_e * xx[a] * mr[b];
                            let dke_d = dm3_ex_e * dr * xx[a] * mr[b]
                                + m3_ex * ddre * xx[a] * mr[b]
                                + dm2_ex_e * xd[a] * mr[b];
                            let dkd_0 = dm2_en_d * xe[a] * mr[b];
                            let dkd_1 = dm2_ex_d * xx[a] * mr[b];
                            let dkd_d = dm3_ex_d * dr * xx[a] * mr[b]
                                + m3_ex * ddrd * xx[a] * mr[b]
                                + dm2_ex_d * xd[a] * mr[b];
                            let d2k_0 = d2m2_en * xe[a] * mr[b];
                            let d2k_1 = d2m2_ex * xx[a] * mr[b];
                            let d2k_d = d2m3_ex * dr * xx[a] * mr[b]
                                + (dm3_ex_d * ddre + dm3_ex_e * ddrd) * xx[a] * mr[b]
                                + d2m2_ex * xd[a] * mr[b];
                            let v = h_ud[0] * dke_0
                                + h_ud[1] * dke_1
                                + h_ud[2] * dke_d
                                + h_ue[0] * dkd_0
                                + h_ue[1] * dkd_1
                                + h_ue[2] * dkd_d
                                + f_pi[0] * d2k_0
                                + f_pi[1] * d2k_1
                                + f_pi[2] * d2k_d;
                            acc[[slices.time.start + a, slices.marginal.start + b]] += v;
                            acc[[slices.marginal.start + b, slices.time.start + a]] += v;
                        }
                    }

                    // wiggle×marginal
                    for li in 0..time_tail.len() {
                        let ci = time_tail.start + li;
                        for b in 0..p_marginal {
                            let dke_0 = eg.basis_d2[[0, li]] * dh0e * mr[b];
                            let dke_1 = xg.basis_d2[[0, li]] * dh1e * mr[b];
                            let dke_d = xg.basis_d3[[0, li]] * dh1e * dr * mr[b]
                                + xg.basis_d2[[0, li]] * ddre * mr[b];
                            let dkd_0 = eg.basis_d2[[0, li]] * dh0d * mr[b];
                            let dkd_1 = xg.basis_d2[[0, li]] * dh1d * mr[b];
                            let dkd_d = xg.basis_d3[[0, li]] * dh1d * dr * mr[b]
                                + xg.basis_d2[[0, li]] * ddrd * mr[b];
                            let d2k_0 = eg.basis_d3[[0, li]] * dh0d * dh0e * mr[b];
                            let d2k_1 = xg.basis_d3[[0, li]] * dh1d * dh1e * mr[b];
                            let d2k_d = xg.basis_d4[[0, li]] * dh1d * dh1e * dr * mr[b]
                                + xg.basis_d3[[0, li]] * (dh1d * ddre + dh1e * ddrd) * mr[b];
                            let v = h_ud[0] * dke_0
                                + h_ud[1] * dke_1
                                + h_ud[2] * dke_d
                                + h_ue[0] * dkd_0
                                + h_ue[1] * dkd_1
                                + h_ue[2] * dkd_d
                                + f_pi[0] * d2k_0
                                + f_pi[1] * d2k_1
                                + f_pi[2] * d2k_d;
                            acc[[slices.time.start + ci, slices.marginal.start + b]] += v;
                            acc[[slices.marginal.start + b, slices.time.start + ci]] += v;
                        }
                    }

                    // marginal×marginal
                    for a in 0..p_marginal {
                        for b in 0..p_marginal {
                            let dke_0 = dm2_en_e * mr[a] * mr[b];
                            let dke_1 = dm2_ex_e * mr[a] * mr[b];
                            let dke_d =
                                dm3_ex_e * dr * mr[a] * mr[b] + m3_ex * ddre * mr[a] * mr[b];
                            let dkd_0 = dm2_en_d * mr[a] * mr[b];
                            let dkd_1 = dm2_ex_d * mr[a] * mr[b];
                            let dkd_d =
                                dm3_ex_d * dr * mr[a] * mr[b] + m3_ex * ddrd * mr[a] * mr[b];
                            let d2k_0 = d2m2_en * mr[a] * mr[b];
                            let d2k_1 = d2m2_ex * mr[a] * mr[b];
                            let d2k_d = d2m3_ex * dr * mr[a] * mr[b]
                                + (dm3_ex_d * ddre + dm3_ex_e * ddrd) * mr[a] * mr[b];
                            acc[[slices.marginal.start + a, slices.marginal.start + b]] += h_ud[0]
                                * dke_0
                                + h_ud[1] * dke_1
                                + h_ud[2] * dke_d
                                + h_ue[0] * dkd_0
                                + h_ue[1] * dkd_1
                                + h_ue[2] * dkd_d
                                + f_pi[0] * d2k_0
                                + f_pi[1] * d2k_1
                                + f_pi[2] * d2k_d;
                        }
                    }
                }
                Ok(acc)
            },
            |mut a, b| -> Result<_, String> {
                a += &b;
                Ok(a)
            },
        )?
        .unwrap_or_else(|| Array2::<f64>::zeros((p_total, p_total)));
        Ok(result)
    }

    /// Exact first directional derivative for flex without timewiggle.
    /// J is constant (no wiggle), so DH[d] = J^T T[u^d] J + Σ (Hu^d)_r K_r.
    /// Scatter one row's directional-derivative primary quantities
    /// (`primary_gradient` = directional change of the primary gradient, i.e.
    /// `H_primary · u`; `primary_hessian` = directional change of the primary
    /// Hessian) through the q-geometry / identity-block pullback into the joint
    /// `acc`. This is the per-row inner body shared by both the
    /// first-directional (`t_ud`, `h_ud`) and second-directional (`q_de`,
    /// `gamma`) flex-no-wiggle paths, and by the batched all-axes Jeffreys
    /// override — so the three callers cannot drift.
    pub(crate) fn accumulate_directional_joint_hessian_row(
        &self,
        row: usize,
        slices: &BlockSlices,
        q_geom: &SurvivalMarginalSlopeDynamicRow,
        identity_blocks: &[(std::ops::Range<usize>, std::ops::Range<usize>)],
        primary_gradient: ndarray::ArrayView1<'_, f64>,
        primary_hessian: ndarray::ArrayView2<'_, f64>,
        acc: &mut Array2<f64>,
    ) -> Result<(), String> {
        // Core q-geometry pullback (Hessian only)
        self.accumulate_dynamic_q_core_hessian(
            row,
            slices,
            q_geom,
            primary_gradient,
            primary_hessian,
            acc,
        )?;
        // Identity block Hessian: cross + diagonal + cross-cross
        for (primary_range, joint_range) in identity_blocks {
            for local in 0..primary_range.len() {
                self.accumulate_identity_primary_cross_hessian(
                    row,
                    slices,
                    q_geom,
                    primary_hessian.slice(s![0..self.core_primary_dimension(), primary_range.start + local]),
                    joint_range,
                    local,
                    acc,
                )?;
            }
            self.add_dense_submatrix(
                acc,
                joint_range,
                joint_range,
                primary_hessian.slice(s![primary_range.clone(), primary_range.clone()]),
            );
        }
        for li in 0..identity_blocks.len() {
            for ri in li + 1..identity_blocks.len() {
                let (lp, lj) = &identity_blocks[li];
                let (rp, rj) = &identity_blocks[ri];
                self.add_dense_symmetric_cross_submatrix(
                    acc,
                    lj,
                    rj,
                    primary_hessian.slice(s![lp.clone(), rp.clone()]),
                );
            }
        }
        // The absorbed influence offset (#461): one primary loading its coefficients through the
        // row of `Z̃_infl`, crossed with the core blocks, with itself and with the identity blocks,
        // as accumulate_dynamic_q_joint_row assembles the joint Hessian.
        if let (Some(infl_primary), Some(infl_joint), Some(z_tilde)) = (
            flex_primary_slices(self).infl,
            slices.influence.as_ref(),
            self.influence_absorber.as_ref(),
        ) {
            let z_row = z_tilde.row(row);
            let core_col =
                primary_hessian.slice(s![0..self.core_primary_dimension(), infl_primary]);
            for (local, &z) in z_row.iter().enumerate() {
                if z != 0.0 {
                    self.accumulate_identity_primary_cross_hessian_scaled(
                        row, slices, q_geom, core_col, z, infl_joint, local, acc,
                    )?;
                }
            }
            let ii_weight = primary_hessian[[infl_primary, infl_primary]];
            if ii_weight != 0.0 {
                for i in 0..z_row.len() {
                    for j in 0..z_row.len() {
                        acc[[infl_joint.start + i, infl_joint.start + j]] +=
                            ii_weight * z_row[i] * z_row[j];
                    }
                }
            }
            for (flex_primary, flex_joint) in identity_blocks {
                for f in 0..flex_primary.len() {
                    let weight = primary_hessian[[flex_primary.start + f, infl_primary]];
                    if weight == 0.0 {
                        continue;
                    }
                    for (i, &z) in z_row.iter().enumerate() {
                        let value = weight * z;
                        acc[[flex_joint.start + f, infl_joint.start + i]] += value;
                        acc[[infl_joint.start + i, flex_joint.start + f]] += value;
                    }
                }
            }
        }
        Ok(())
    }

    pub(crate) fn exact_newton_joint_hessian_directional_derivative_flex_no_wiggle(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_flat: &Array1<f64>,
    ) -> Result<Array2<f64>, String> {
        let slices = block_slices(self, block_states);
        let primary = flex_primary_slices(self);
        let p_total = slices.total;
        let identity_blocks = flex_identity_block_pairs(&primary, &slices);
        let result = gam_linalg::pairwise_reduce::par_deterministic_try_block_fold(
            self.n,
            |range| -> Result<Array2<f64>, String> {
                let mut acc = Array2::<f64>::zeros((p_total, p_total));
                for row in range {
                    let q_geom = self.row_dynamic_q_geometry(row, block_states)?;
                    let h_pi = self
                        .compute_row_flex_primary_gradient_hessian_exact(
                            row,
                            block_states,
                            &q_geom,
                            &primary,
                        )?
                        .2;
                    let u_d = self.row_primary_direction_from_flat_dynamic_with_q_geometry(
                        row,
                        block_states,
                        &slices,
                        &q_geom,
                        d_beta_flat,
                    )?;
                    let t_ud =
                        self.row_flex_primary_third_contracted_exact(row, block_states, &u_d)?;
                    let h_ud = h_pi.dot(&u_d);
                    self.accumulate_directional_joint_hessian_row(
                        row,
                        &slices,
                        &q_geom,
                        &identity_blocks,
                        h_ud.view(),
                        t_ud.view(),
                        &mut acc,
                    )?;
                }
                Ok(acc)
            },
            |mut a, b| -> Result<_, String> {
                a += &b;
                Ok(a)
            },
        )?
        .unwrap_or_else(|| Array2::<f64>::zeros((p_total, p_total)));
        Ok(result)
    }

    /// Build-once all-axes variant of
    /// [`Self::exact_newton_joint_hessian_directional_derivative_flex_no_wiggle`].
    ///
    /// The Jeffreys all-axes sweep needs the directional joint-Hessian for each
    /// of the `p` coordinate axes. Calling the single-direction routine `p`
    /// times rebuilds, per row and per axis, the direction-independent flex
    /// geometry (`q`-geometry, primary Hessian, intercept solves, cached
    /// partitions, exact base timepoints) — a `p`-fold redundant cost that is
    /// the #979 flex marginal-slope hot path. This variant builds that geometry
    /// once per row (`FlexThirdRowBase` + the primary Hessian) and contracts it
    /// against each axis, so only the per-axis directional pieces are repeated.
    /// Each output matrix routes through the same per-row assemblers as the
    /// corresponding single-axis call (`row_flex_third_contract_from_base` +
    /// `accumulate_directional_joint_hessian_row`), so it equals that call up to
    /// the cross-row rayon reduction order.
    pub(crate) fn exact_newton_joint_hessian_directional_derivative_flex_no_wiggle_all_axes(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<Vec<Array2<f64>>, String> {
        let slices = block_slices(self, block_states);
        let primary = flex_primary_slices(self);
        let p_total = slices.total;
        let identity_blocks = flex_identity_block_pairs(&primary, &slices);
        let zeros = || vec![Array2::<f64>::zeros((p_total, p_total)); p_total];
        let result = gam_linalg::pairwise_reduce::par_deterministic_try_block_fold(
            self.n,
            |range| -> Result<Vec<Array2<f64>>, String> {
                let mut acc = zeros();
                for row in range {
                    // Direction-independent per-row geometry, built once.
                    let q_geom = self.row_dynamic_q_geometry(row, block_states)?;
                    let h_pi = self
                        .compute_row_flex_primary_gradient_hessian_exact(
                            row,
                            block_states,
                            &q_geom,
                            &primary,
                        )?
                        .2;
                    let base =
                        self.build_row_flex_third_base_with_states(row, block_states, &primary)?;
                    // Per-axis: only the primary-direction pullback, the third
                    // contraction against it, and the row accumulation are repeated.
                    for axis_idx in 0..p_total {
                        let mut axis = Array1::<f64>::zeros(p_total);
                        axis[axis_idx] = 1.0;
                        let u_d = self.row_primary_direction_from_flat_dynamic_with_q_geometry(
                            row,
                            block_states,
                            &slices,
                            &q_geom,
                            &axis,
                        )?;
                        let t_ud = self.row_flex_third_contract_from_base(&base, &u_d)?;
                        let h_ud = h_pi.dot(&u_d);
                        self.accumulate_directional_joint_hessian_row(
                            row,
                            &slices,
                            &q_geom,
                            &identity_blocks,
                            h_ud.view(),
                            t_ud.view(),
                            &mut acc[axis_idx],
                        )?;
                    }
                }
                Ok(acc)
            },
            |mut a, b| -> Result<_, String> {
                for (ai, bi) in a.iter_mut().zip(b.into_iter()) {
                    *ai += &bi;
                }
                Ok(a)
            },
        )?
        .unwrap_or_else(zeros);
        Ok(result)
    }

    /// Third directional derivative `D³H[u, v, e_a]` of the joint Hessian along every
    /// coefficient axis, for flex without a time wiggle (gam#2893).
    ///
    /// Without a wiggle `q` is linear in β, so the q-map curvature vanishes and every
    /// primary Jacobian `J` is constant: `D³H[u, v, w] = Jᵀ F₅[J u, J v, J w] J`, with `F₅`
    /// the contracted fifth of the row NLL. `F₅` is linear in its third slot, so each row
    /// contracts once per primary axis, and every coefficient axis combines those matrices
    /// with its own primary image `J e_a` before one pullback.
    pub(crate) fn exact_newton_joint_hessian_third_directional_derivative_flex_no_wiggle_all_axes(
        &self,
        block_states: &[ParameterBlockState],
        d_u: &Array1<f64>,
        d_v: &Array1<f64>,
    ) -> Result<Vec<Array2<f64>>, String> {
        let slices = block_slices(self, block_states);
        let primary = flex_primary_slices(self);
        let p_total = slices.total;
        let identity_blocks = flex_identity_block_pairs(&primary, &slices);
        let zeros = || vec![Array2::<f64>::zeros((p_total, p_total)); p_total];
        let result = gam_linalg::pairwise_reduce::par_deterministic_try_block_fold(
            self.n,
            |range| -> Result<Vec<Array2<f64>>, String> {
                let mut acc = zeros();
                let zero_gradient = Array1::<f64>::zeros(primary.total);
                for row in range {
                    let q_geom = self.row_dynamic_q_geometry(row, block_states)?;
                    let u_pi = self.row_primary_direction_from_flat_dynamic_with_q_geometry(
                        row,
                        block_states,
                        &slices,
                        &q_geom,
                        d_u,
                    )?;
                    let v_pi = self.row_primary_direction_from_flat_dynamic_with_q_geometry(
                        row,
                        block_states,
                        &slices,
                        &q_geom,
                        d_v,
                    )?;
                    let base = self.build_row_flex_fifth_base_with_states(row, block_states, &primary)?;
                    let fifth =
                        self.row_flex_fifth_contract_all_primary_axes_from_base(&base, &u_pi, &v_pi)?;
                    for axis_idx in 0..p_total {
                        let mut axis = Array1::<f64>::zeros(p_total);
                        axis[axis_idx] = 1.0;
                        let w_pi = self.row_primary_direction_from_flat_dynamic_with_q_geometry(
                            row,
                            block_states,
                            &slices,
                            &q_geom,
                            &axis,
                        )?;
                        if w_pi.iter().all(|weight| *weight == 0.0) {
                            continue;
                        }
                        let mut contracted = Array2::<f64>::zeros((primary.total, primary.total));
                        for (primary_axis, &weight) in w_pi.iter().enumerate() {
                            if weight != 0.0 {
                                contracted.scaled_add(weight, &fifth[primary_axis]);
                            }
                        }
                        self.accumulate_directional_joint_hessian_row(
                            row,
                            &slices,
                            &q_geom,
                            &identity_blocks,
                            zero_gradient.view(),
                            contracted.view(),
                            &mut acc[axis_idx],
                        )?;
                    }
                }
                Ok(acc)
            },
            |mut a, b| -> Result<_, String> {
                for (ai, bi) in a.iter_mut().zip(b.into_iter()) {
                    *ai += &bi;
                }
                Ok(a)
            },
        )?
        .unwrap_or_else(zeros);
        Ok(result)
    }

    /// Exact second directional derivative for flex without timewiggle.
    /// J constant ⇒ D²H[d,e] = J^T Q[u^d,u^e] J + Σ (T_d·u^e)_r K_r.
    pub(crate) fn exact_newton_joint_hessiansecond_directional_derivative_flex_no_wiggle(
        &self,
        block_states: &[ParameterBlockState],
        d_u: &Array1<f64>,
        d_v: &Array1<f64>,
    ) -> Result<Array2<f64>, String> {
        let slices = block_slices(self, block_states);
        let primary = flex_primary_slices(self);
        let p_total = slices.total;
        let identity_blocks = flex_identity_block_pairs(&primary, &slices);
        let result = gam_linalg::pairwise_reduce::par_deterministic_try_block_fold(
            self.n,
            |range| -> Result<Array2<f64>, String> {
                let mut acc = Array2::<f64>::zeros((p_total, p_total));
                for row in range {
                    let q_geom = self.row_dynamic_q_geometry(row, block_states)?;
                    let ud = self.row_primary_direction_from_flat_dynamic_with_q_geometry(
                        row,
                        block_states,
                        &slices,
                        &q_geom,
                        d_u,
                    )?;
                    let ue = self.row_primary_direction_from_flat_dynamic_with_q_geometry(
                        row,
                        block_states,
                        &slices,
                        &q_geom,
                        d_v,
                    )?;
                    let q_de =
                        self.row_flex_primary_fourth_contracted_exact(row, block_states, &ud, &ue)?;
                    let t_d =
                        self.row_flex_primary_third_contracted_exact(row, block_states, &ud)?;
                    let gamma = t_d.dot(&ue);
                    // Hessian-only: accumulate q-core + identity block Hessian
                    self.accumulate_directional_joint_hessian_row(
                        row,
                        &slices,
                        &q_geom,
                        &identity_blocks,
                        gamma.view(),
                        q_de.view(),
                        &mut acc,
                    )?;
                }
                Ok(acc)
            },
            |mut a, b| -> Result<_, String> {
                a += &b;
                Ok(a)
            },
        )?
        .unwrap_or_else(|| Array2::<f64>::zeros((p_total, p_total)));
        Ok(result)
    }

    /// Build-once all-axes variant of
    /// [`Self::exact_newton_joint_hessiansecond_directional_derivative_flex_no_wiggle`]:
    /// `{D²H[u, e_a]}` along every coefficient axis `a` (gam#2893).
    ///
    /// Without a time wiggle `J` is constant, so
    /// `D²H[u, w] = Jᵀ ℓ⁴[Ju, Jw] J + Σ_r (ℓ³[Ju]·Jw)_r K_r`, linear in `Jw`. Each row builds
    /// its flex base once, contracts `ℓ³[Ju]` once and `ℓ⁴[Ju, e_k]` once per primary axis `k`,
    /// and every coefficient axis combines them with its own primary image `J e_a` before the
    /// single-axis assembler. Calling the single-direction routine once per axis rebuilds every
    /// row's base `p` times.
    pub(crate) fn exact_newton_joint_hessian_second_directional_derivative_flex_no_wiggle_all_axes(
        &self,
        block_states: &[ParameterBlockState],
        d_u: &Array1<f64>,
    ) -> Result<Vec<Array2<f64>>, String> {
        let slices = block_slices(self, block_states);
        let primary = flex_primary_slices(self);
        let p_total = slices.total;
        let identity_blocks = flex_identity_block_pairs(&primary, &slices);
        let zeros = || vec![Array2::<f64>::zeros((p_total, p_total)); p_total];
        let result = gam_linalg::pairwise_reduce::par_deterministic_try_block_fold(
            self.n,
            |range| -> Result<Vec<Array2<f64>>, String> {
                let mut acc = zeros();
                for row in range {
                    let q_geom = self.row_dynamic_q_geometry(row, block_states)?;
                    let ud = self.row_primary_direction_from_flat_dynamic_with_q_geometry(
                        row,
                        block_states,
                        &slices,
                        &q_geom,
                        d_u,
                    )?;
                    let base =
                        self.build_row_flex_third_base_with_states(row, block_states, &primary)?;
                    let t_d = self.row_flex_third_contract_from_base(&base, &ud)?;
                    let mut fourth = Vec::with_capacity(primary.total);
                    for primary_axis in 0..primary.total {
                        let mut unit = Array1::<f64>::zeros(primary.total);
                        unit[primary_axis] = 1.0;
                        fourth.push(self.row_flex_fourth_contract_from_base(&base, &ud, &unit)?);
                    }
                    for axis_idx in 0..p_total {
                        let mut axis = Array1::<f64>::zeros(p_total);
                        axis[axis_idx] = 1.0;
                        let ue = self.row_primary_direction_from_flat_dynamic_with_q_geometry(
                            row,
                            block_states,
                            &slices,
                            &q_geom,
                            &axis,
                        )?;
                        if ue.iter().all(|weight| *weight == 0.0) {
                            continue;
                        }
                        let mut q_de = Array2::<f64>::zeros((primary.total, primary.total));
                        for (primary_axis, &weight) in ue.iter().enumerate() {
                            if weight != 0.0 {
                                q_de.scaled_add(weight, &fourth[primary_axis]);
                            }
                        }
                        let gamma = t_d.dot(&ue);
                        self.accumulate_directional_joint_hessian_row(
                            row,
                            &slices,
                            &q_geom,
                            &identity_blocks,
                            gamma.view(),
                            q_de.view(),
                            &mut acc[axis_idx],
                        )?;
                    }
                }
                Ok(acc)
            },
            |mut a, b| -> Result<_, String> {
                for (ai, bi) in a.iter_mut().zip(b.into_iter()) {
                    *ai += &bi;
                }
                Ok(a)
            },
        )?
        .unwrap_or_else(zeros);
        Ok(result)
    }

    pub(crate) fn evaluate_blockwise_exact_newton(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<FamilyEvaluation, String> {
        if self.per_z_slope_active() {
            return self.evaluate_blockwise_exact_newton_per_z(block_states);
        }
        if self.effective_flex_active(block_states)? {
            return self.evaluate_blockwise_exact_newton_flexible(block_states);
        }
        if self.flex_timewiggle_active() {
            return self.evaluate_blockwise_exact_newton_timewiggle(block_states);
        }

        // Every rigid configuration takes the row-kernel block pullback. It is one
        // route at every width, and it runs the channel pair loop a follow-up-varying
        // slope needs (gam#2765).
        self.evaluate_blockwise_exact_newton_dense(block_states)
    }

    pub(crate) fn evaluate_blockwise_exact_newton_per_z(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<FamilyEvaluation, String> {
        if self.effective_flex_active(block_states)? || self.flex_timewiggle_active() {
            return Err(
                "survival marginal-slope per-z slope surfaces currently require the rigid row kernel"
                    .to_string(),
            );
        }
        let slices = block_slices(self, block_states);
        let p_t = slices.time.len();
        let p_m = slices.marginal.len();
        let p_g = slices.slope.len();
        let k = self.score_dim();
        let beta_time = &block_states[0].beta;
        let probit_scale = self.probit_frailty_scale();
        type PerZBlockAcc = (
            f64,
            Array1<f64>,
            Array1<f64>,
            Array1<f64>,
            Array2<f64>,
            Array2<f64>,
            Array2<f64>,
        );
        let make_per_z_acc = || -> PerZBlockAcc {
            (
                0.0,
                Array1::<f64>::zeros(p_t),
                Array1::<f64>::zeros(p_m),
                Array1::<f64>::zeros(p_g),
                Array2::<f64>::zeros((p_t, p_t)),
                Array2::<f64>::zeros((p_m, p_m)),
                Array2::<f64>::zeros((p_g, p_g)),
            )
        };
        let (ll, grad_t, grad_m, grad_g, hess_t, hess_m, hess_g): PerZBlockAcc =
            gam_linalg::pairwise_reduce::par_deterministic_try_block_fold(
                self.n,
                |range| -> Result<_, String> {
                    let mut acc = make_per_z_acc();
                    let mut row_jet_arena = VectorRowWorkspace::for_family(self)?;
                    let mut slope_workspace = self.slope_row_workspace()?;
                    for row in range {
                        let q0 = self.design_entry.dot_row(row, beta_time)
                            + self.offset_entry[row]
                            + block_states[1].eta[row];
                        let q1 = self.design_exit.dot_row(row, beta_time)
                            + self.offset_exit[row]
                            + block_states[1].eta[row];
                        let qd1 = self.design_derivative_exit.dot_row(row, beta_time)
                            + self.derivative_offset_exit[row];
                        self.fill_slope_values_for_row(
                            row,
                            block_states,
                            &mut slope_workspace,
                        )?;
                        let z_row = self.z.row(row);
                        let z = z_row.as_slice().ok_or_else(|| {
                            "per-score blockwise score row must be contiguous".to_string()
                        })?;
                        let nll = row_jet_arena.evaluate_row(
                            row,
                            q0,
                            q1,
                            qd1,
                            slope_workspace.values(),
                            z,
                            self.weights[row],
                            self.event[row],
                            self.derivative_guard,
                            probit_scale,
                        )?;
                        let (f_pi, f_pipi) = row_jet_arena.derivatives();
                        acc.0 -= nll;
                        self.design_entry
                            .axpy_row_into(row, -f_pi[0], &mut acc.1.view_mut())?;
                        self.design_exit
                            .axpy_row_into(row, -f_pi[1], &mut acc.1.view_mut())?;
                        self.design_derivative_exit.axpy_row_into(
                            row,
                            -f_pi[2],
                            &mut acc.1.view_mut(),
                        )?;
                        self.marginal_design.axpy_row_into(
                            row,
                            -(f_pi[0] + f_pi[1]),
                            &mut acc.2.view_mut(),
                        )?;
                        let channel_rows = slope_workspace.channel_rows();
                        for coord in 0..k {
                            let alpha = -f_pi[3 + coord];
                            for col in 0..p_g {
                                acc.3[col] += alpha * channel_rows[[coord, col]];
                            }
                        }
                        let time_designs = [
                            &self.design_entry,
                            &self.design_exit,
                            &self.design_derivative_exit,
                        ];
                        for a in 0..3 {
                            for b in 0..3 {
                                time_designs[a].row_outer_into(
                                    row,
                                    time_designs[b],
                                    f_pipi[[a, b]],
                                    &mut acc.4,
                                )?;
                            }
                        }
                        let alpha_mm =
                            f_pipi[[0, 0]] + f_pipi[[0, 1]] + f_pipi[[1, 0]] + f_pipi[[1, 1]];
                        self.marginal_design
                            .syr_row_into(row, alpha_mm, &mut acc.5)?;
                        for a in 0..k {
                            for b in 0..k {
                                let alpha = f_pipi[[3 + a, 3 + b]];
                                if alpha == 0.0 {
                                    continue;
                                }
                                for ca in 0..p_g {
                                    let va = channel_rows[[a, ca]] * alpha;
                                    for cb in 0..p_g {
                                        acc.6[[ca, cb]] += va * channel_rows[[b, cb]];
                                    }
                                }
                            }
                        }
                    }
                    Ok(acc)
                },
                |mut a, b| -> Result<_, String> {
                    a.0 += b.0;
                    a.1 += &b.1;
                    a.2 += &b.2;
                    a.3 += &b.3;
                    a.4 += &b.4;
                    a.5 += &b.5;
                    a.6 += &b.6;
                    Ok(a)
                },
            )?
            .unwrap_or_else(make_per_z_acc);
        Ok(FamilyEvaluation {
            log_likelihood: ll,
            blockworking_sets: vec![
                BlockWorkingSet::ExactNewton {
                    gradient: grad_t,
                    hessian: SymmetricMatrix::Dense(hess_t),
                },
                BlockWorkingSet::ExactNewton {
                    gradient: grad_m,
                    hessian: SymmetricMatrix::Dense(hess_m),
                },
                BlockWorkingSet::ExactNewton {
                    gradient: grad_g,
                    hessian: SymmetricMatrix::Dense(hess_g),
                },
            ],
        })
    }

    pub(crate) fn evaluate_exact_newton_joint_dense_per_z(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<(f64, Array1<f64>, Array2<f64>), String> {
        let slices = block_slices(self, block_states);
        let total = slices.total;
        let k = self.score_dim();
        let dim = 3 + k;
        let beta_time = &block_states[0].beta;
        let probit_scale = self.probit_frailty_scale();
        type PerZJointAcc = (f64, Array1<f64>, Array2<f64>);
        let make_per_z_joint_acc = || -> PerZJointAcc {
            (
                0.0,
                Array1::<f64>::zeros(total),
                Array2::<f64>::zeros((total, total)),
            )
        };
        let (ll, grad, hess): PerZJointAcc =
            gam_linalg::pairwise_reduce::par_deterministic_try_block_fold(
                self.n,
                |range| -> Result<_, String> {
                    let mut acc = make_per_z_joint_acc();
                    let mut row_jet_arena = VectorRowWorkspace::for_family(self)?;
                    let mut slope_workspace = self.slope_row_workspace()?;
                    let mut j = Array2::<f64>::zeros((dim, total));
                    for row in range {
                        let q0 = self.design_entry.dot_row(row, beta_time)
                            + self.offset_entry[row]
                            + block_states[1].eta[row];
                        let q1 = self.design_exit.dot_row(row, beta_time)
                            + self.offset_exit[row]
                            + block_states[1].eta[row];
                        let qd1 = self.design_derivative_exit.dot_row(row, beta_time)
                            + self.derivative_offset_exit[row];
                        self.fill_slope_values_for_row(
                            row,
                            block_states,
                            &mut slope_workspace,
                        )?;
                        let z_row = self.z.row(row);
                        let z = z_row.as_slice().ok_or_else(|| {
                            "per-score dense-joint score row must be contiguous".to_string()
                        })?;
                        let nll = row_jet_arena.evaluate_row(
                            row,
                            q0,
                            q1,
                            qd1,
                            slope_workspace.values(),
                            z,
                            self.weights[row],
                            self.event[row],
                            self.derivative_guard,
                            probit_scale,
                        )?;
                        let (f_pi, f_pipi) = row_jet_arena.derivatives();
                        acc.0 -= nll;
                        self.design_entry
                            .row_chunk_into(
                                row..row + 1,
                                j.slice_mut(s![0..1, slices.time.clone()]),
                            )
                            .map_err(|e| {
                                format!("evaluate_exact_newton_joint_dense_per_z entry row: {e}")
                            })?;
                        self.design_exit
                            .row_chunk_into(
                                row..row + 1,
                                j.slice_mut(s![1..2, slices.time.clone()]),
                            )
                            .map_err(|e| {
                                format!("evaluate_exact_newton_joint_dense_per_z exit row: {e}")
                            })?;
                        self.design_derivative_exit
                            .row_chunk_into(
                                row..row + 1,
                                j.slice_mut(s![2..3, slices.time.clone()]),
                            )
                            .map_err(|e| {
                                format!(
                                    "evaluate_exact_newton_joint_dense_per_z derivative row: {e}"
                                )
                            })?;
                        self.marginal_design
                            .row_chunk_into(
                                row..row + 1,
                                j.slice_mut(s![0..1, slices.marginal.clone()]),
                            )
                            .map_err(|e| {
                                format!("evaluate_exact_newton_joint_dense_per_z marginal row: {e}")
                            })?;
                        for col in slices.marginal.clone() {
                            j[[1, col]] = j[[0, col]];
                        }
                        let channel_rows = slope_workspace.channel_rows();
                        for coord in 0..k {
                            j.slice_mut(s![3 + coord, slices.slope.clone()])
                                .assign(&channel_rows.row(coord));
                        }
                        for a in 0..dim {
                            for col in 0..total {
                                acc.1[col] -= f_pi[a] * j[[a, col]];
                            }
                        }
                        for a in 0..dim {
                            for b in 0..dim {
                                let alpha = f_pipi[[a, b]];
                                if alpha == 0.0 {
                                    continue;
                                }
                                for ca in 0..total {
                                    let va = j[[a, ca]] * alpha;
                                    if va == 0.0 {
                                        continue;
                                    }
                                    for cb in 0..total {
                                        acc.2[[ca, cb]] += va * j[[b, cb]];
                                    }
                                }
                            }
                        }
                    }
                    Ok(acc)
                },
                |mut a, b| -> Result<_, String> {
                    a.0 += b.0;
                    a.1 += &b.1;
                    a.2 += &b.2;
                    Ok(a)
                },
            )?
            .unwrap_or_else(make_per_z_joint_acc);
        Ok((ll, grad, hess))
    }

    /// Blockwise exact-Newton for the flexible (score-warp / link-deviation)
    /// model.
    ///
    /// Accumulates exact per-block coefficient gradients and Hessians directly
    /// from the dynamic-q row jets. This preserves the exact block Newton
    /// update while avoiding dense full-joint assembly when the caller only
    /// needs block-local working sets.
    pub(crate) fn evaluate_blockwise_exact_newton_flexible(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<FamilyEvaluation, String> {
        self.validate_exact_monotonicity(block_states)?;
        let primary = flex_primary_slices(self);
        self.evaluate_blockwise_exact_newton_dynamic_q(block_states, &primary, |row, q_geom| {
            self.compute_row_flex_primary_gradient_hessian_exact(
                row,
                block_states,
                q_geom,
                &primary,
            )
        })
    }

    /// Blockwise exact-Newton for the time-wiggle model.
    ///
    /// Accumulates exact block-local Hessians directly from the 4D primary
    /// row calculus instead of materializing and slicing a dense joint Hessian.
    pub(crate) fn evaluate_blockwise_exact_newton_timewiggle(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<FamilyEvaluation, String> {
        let primary = flex_primary_slices(self);
        self.evaluate_blockwise_exact_newton_dynamic_q(block_states, &primary, |row, _| {
            self.compute_row_primary_gradient_hessian_uncached(row, block_states)
        })
    }

    // ── Rigid block-diagonal route ───────────────────────────────────

    pub(crate) fn evaluate_blockwise_exact_newton_dense(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<FamilyEvaluation, String> {
        in_slope_frame!(self, P, Frame, {
            self.evaluate_blockwise_exact_newton_dense_in_frame::<P, Frame>(block_states)
        })
    }

    fn evaluate_blockwise_exact_newton_dense_in_frame<
        const P: usize,
        Frame: SlopeRowGeometry<P>,
    >(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<FamilyEvaluation, String> {
        // Build RowKernel — the single source of truth for all exact-Newton
        // quantities.  The cache evaluates every row kernel once and stores
        // (nll_i, g_i[P], H_i[P×P]).
        let kern =
            SurvivalMarginalSlopeRowKernel::<P, Frame>::new(self.clone(), block_states.to_vec());
        let rows = crate::row_kernel::RowSet::All;
        let cache = build_row_kernel_cache(&kern, &rows)?;

        let ll = row_kernel_log_likelihood(&cache, &rows);

        // Joint gradient:  g = -Σ_i Jᵢᵀ gᵢ  (sign: gᵢ are NLL gradients,
        // we negate to get log-likelihood gradient).
        let nll_grad = row_kernel_gradient(&kern, &cache, &rows);
        let joint_gradient = -nll_grad;

        // Block-diagonal Hessians only — the inner solver consumes per-block
        // working sets, so we accumulate the principal time-time, m-m, and
        // g-g blocks directly instead of building the full joint Hessian and
        // slicing.  Cost falls from Θ(n·(p_t+p_m+p_g)²) to
        // Θ(n·(p_t²+p_m²+p_g²)).
        let slices = block_slices(self, block_states);
        let p_t = slices.time.len();
        let p_m = slices.marginal.len();
        let p_g = slices.slope.len();
        let mut hess_time = Array2::<f64>::zeros((p_t, p_t));
        let mut hess_marginal = Array2::<f64>::zeros((p_m, p_m));
        let mut hess_slope = Array2::<f64>::zeros((p_g, p_g));
        for row in 0..cache.n {
            let h = &cache.hessians[row];
            let mut h_arr = Array2::<f64>::zeros((P, P));
            for a in 0..P {
                for b in 0..P {
                    h_arr[[a, b]] = h[a][b];
                }
            }
            self.add_pullback_block_diagonals(
                row,
                &h_arr,
                &mut hess_time,
                &mut hess_marginal,
                &mut hess_slope,
            );
        }

        let mut blockworking_sets = vec![
            BlockWorkingSet::ExactNewton {
                gradient: joint_gradient.slice(s![slices.time.clone()]).to_owned(),
                hessian: SymmetricMatrix::Dense(hess_time),
            },
            BlockWorkingSet::ExactNewton {
                gradient: joint_gradient.slice(s![slices.marginal.clone()]).to_owned(),
                hessian: SymmetricMatrix::Dense(hess_marginal),
            },
            BlockWorkingSet::ExactNewton {
                gradient: joint_gradient.slice(s![slices.slope.clone()]).to_owned(),
                hessian: SymmetricMatrix::Dense(hess_slope),
            },
        ];
        if let Some(range) = slices.score_warp.as_ref() {
            // The 4-D row kernel does not span score_warp / link_dev primary
            // dimensions, so these blocks contribute zero gradient/Hessian
            // here — exactly what the joint-then-slice path produced.
            blockworking_sets.push(BlockWorkingSet::ExactNewton {
                gradient: joint_gradient.slice(s![range.clone()]).to_owned(),
                hessian: SymmetricMatrix::Dense(Array2::zeros((range.len(), range.len()))),
            });
        }
        if let Some(range) = slices.link_dev.as_ref() {
            blockworking_sets.push(BlockWorkingSet::ExactNewton {
                gradient: joint_gradient.slice(s![range.clone()]).to_owned(),
                hessian: SymmetricMatrix::Dense(Array2::zeros((range.len(), range.len()))),
            });
        }
        Ok(FamilyEvaluation {
            log_likelihood: ll,
            blockworking_sets,
        })
    }
}

// ── CustomFamily impl ─────────────────────────────────────────────────

pub(crate) fn time_wiggle_basis_ncols(knots: &Array1<f64>, degree: usize) -> Result<usize, String> {
    if knots.is_empty() {
        return Err(SurvivalMarginalSlopeError::InvalidInput {
            reason: "survival-marginal-slope timewiggle requires at least one knot".to_string(),
        }
        .into());
    }
    let probe = 0.5 * (knots[0] + knots[knots.len() - 1]);
    let h0 = Array1::from_vec(vec![probe]);
    Ok(monotone_wiggle_basis_with_derivative_order(h0.view(), knots, degree, 0)?.ncols())
}
