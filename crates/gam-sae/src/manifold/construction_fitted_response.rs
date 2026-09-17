// [#2933 F36] The joint fitted-response divergence `tr(∂f̂/∂y)` of the SAE
// reconstruction. It reads the materialized exact stationarity eigensystem owned
// by `construction_exact_hessian.rs`, so it is `include!`d beside that file into
// `construction.rs` and shares its module scope.

/// Why [`SaeManifoldTerm::fitted_response_divergence`] produced no divergence.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum FittedResponseDivergenceRefusal {
    /// An atom's basis exposes no second jet, so the residual curvature of the
    /// observed information is unknown. Unknown is not zero: no divergence is
    /// formed from the Gauss--Newton block alone.
    SecondJetsUnavailable { reason: String },
    /// The exact stationarity operator, the data curvature, or a solve failed at
    /// this state.
    Numerical { reason: String },
}

impl std::fmt::Display for FittedResponseDivergenceRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SecondJetsUnavailable { reason } => write!(
                f,
                "fitted-response divergence unavailable: the observed information needs \
                 every atom's second jet ({reason})"
            ),
            Self::Numerical { reason } => write!(f, "fitted-response divergence failed: {reason}"),
        }
    }
}

/// Which estimator produced a [`FittedResponseDivergence`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum FittedResponseDivergenceEstimator {
    /// `Σ_retained vᵢᵀ G vᵢ / λᵢ` on the materialized exact stationarity
    /// eigensystem: the trace itself, to the arithmetic of one symmetric
    /// eigendecomposition.
    ExactSpectral,
    /// The Rademacher average of `zᵀ A⁺ G z`, one Krylov `A⁺` solve per probe,
    /// where the dense eigensystem is not admitted. `standard_error` is the
    /// sample standard deviation of the probe values over `√probes`.
    Hutchinson { probes: usize, standard_error: f64 },
}

/// The within-basin Stein degrees of freedom of the fitted reconstruction.
///
/// Only the smooth response inside one basin is differentiated. The selected TopK
/// support, frozen routing, and the basin the inner solve converged to are held
/// fixed, so the selection (search) degrees of freedom of a support swap or a basin
/// switch are omitted, not estimated (#2933 F37).
///
/// A frame's residual degrees of freedom are `‖I − R‖²_F` for its response
/// `R = ∂f̂/∂y`. If the noise covariance is `φ` times the inverse of the frame's
/// weight, the residual of the linearized fit has
/// `E‖y − f̂‖² = ‖(I − R)μ‖² + φ·‖I − R‖²_F`, and
/// `‖I − R‖²_F = N − 2 tr R + ‖R‖²_F`. That equals `N − tr R` only when `R` is a
/// projection. For the ridge smoother `R = ½I` it is `N/4`, not `N/2` (#2933 F40).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct FittedResponseDivergence {
    pub(crate) divergence: f64,
    /// `‖I − Ω^{½} R Ω^{−½}‖²_F` over the likelihood-frame scalars of the rows
    /// with positive weight, `Ω = ⊕ᵢ wᵢMᵢ`.
    pub(crate) likelihood_residual_dof: f64,
    /// `‖I − R‖²_F` over the raw output scalars of the rows with positive weight.
    /// It equals the likelihood value unless the metric whitens.
    pub(crate) raw_residual_dof: f64,
    pub(crate) estimator: FittedResponseDivergenceEstimator,
}

/// The output inner product a dense data curvature is assembled with, over the
/// `√w`-weighted row jets `J̃ᵢ = √wᵢ Jᵢ`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ResponseOutputGram {
    /// `Mᵢ`: `Σᵢ J̃ᵢᵀ Mᵢ J̃ᵢ = JᵀΩJ`, the likelihood frame's curvature `G`.
    Likelihood,
    /// `wᵢMᵢ²`: `Σᵢ J̃ᵢᵀ wᵢMᵢ² J̃ᵢ = JᵀΩ²J`.
    SquaredLikelihood,
    /// `I/wᵢ` on rows with `wᵢ > 0`: `Σ_{wᵢ>0} JᵢᵀJᵢ`.
    UnweightedRaw,
}

/// The frame a residual degree-of-freedom count is taken in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FittedResponseFrame {
    /// The likelihood scalars, with noise covariance `φ·Ω⁻¹`.
    Likelihood,
    /// The raw output scalars, with noise covariance `φ·I`.
    Raw,
}

/// Group border channels by output vector. A channel's reconstruction jet is a
/// scalar times the output vector of its `(atom, output channel)` pair, which
/// every basis column of that atom shares, so the border data curvature needs
/// one inner product per pair of classes rather than per pair of channels.
fn border_output_classes(border: &[SaeBorderChannel]) -> (Vec<usize>, Vec<Vec<f64>>) {
    let mut class_of = Vec::with_capacity(border.len());
    let mut outputs: Vec<Vec<f64>> = Vec::new();
    let mut index: std::collections::HashMap<(usize, Vec<u64>), usize> =
        std::collections::HashMap::new();
    for channel in border {
        let key = (
            channel.atom,
            channel.output.iter().map(|value| value.to_bits()).collect::<Vec<u64>>(),
        );
        let class = *index.entry(key).or_insert_with(|| {
            outputs.push(channel.output.clone());
            outputs.len() - 1
        });
        class_of.push(class);
    }
    (class_of, outputs)
}

impl SaeManifoldTerm {
    /// Joint fitted-response divergence of the reconstruction at the converged
    /// inner state (#2933 F36).
    ///
    /// With row weights `wᵢ` and output metrics `Mᵢ`, the inner objective is
    /// `L(θ; y) = ½ Σᵢ wᵢ (fᵢ(θ) − yᵢ)ᵀ Mᵢ (fᵢ(θ) − yᵢ) + P(θ)` over every
    /// estimated coordinate `θ = (gate logits, chart coordinates, decoder border)`.
    /// Its stationarity `g = Σᵢ wᵢ Jᵢᵀ Mᵢ (fᵢ − yᵢ) + ∇P = 0` has `∂g/∂θ = A`, the
    /// exact observed information (residual curvature and the actual prior
    /// curvature included), and `∂g/∂yᵢ = −wᵢ Jᵢᵀ Mᵢ`. The implicit-function theorem
    /// gives `∂θ̂/∂yᵢ = A⁺ wᵢ Jᵢᵀ Mᵢ`, hence
    ///
    /// ```text
    ///   tr(∂f̂/∂y) = Σᵢ tr(Jᵢ A⁺ Jᵢᵀ wᵢ Mᵢ) = tr(A⁺ G),   G = Σᵢ J̃ᵢᵀ Mᵢ J̃ᵢ,  J̃ᵢ = √wᵢ Jᵢ.
    /// ```
    ///
    /// `A⁺` is the symmetric pseudoinverse with the null band of
    /// [`ExactHessianSpectralBlock::rank_floor`], the same classification the IFT
    /// stationarity solve owns: a direction the objective cannot resolve carries
    /// no identified response, and resolved negative modes are inverted rather
    /// than dropped. The trace couples every block through `A⁺`. Mixed coordinate
    /// curvature, cross-atom gate coupling, free logits, the decoder border, atoms
    /// without ARD, and a periodic prior on the concave side of its axis all enter
    /// through the one operator, never as a sum of separately inverted diagonal
    /// entries.
    ///
    /// The decoder frames are held at their fitted orientation, so on a framed
    /// term this is the divergence conditional on the fitted frames. The TopK
    /// support and the converged basin are held fixed in the same way, so the trace
    /// carries no selection degrees of freedom (#2933 F37).
    pub(crate) fn fitted_response_divergence(
        &self,
        target: ArrayView2<'_, f64>,
        rho: &SaeManifoldRho,
        cache: &ArrowFactorCache,
    ) -> Result<FittedResponseDivergence, FittedResponseDivergenceRefusal> {
        let numerical = |reason: String| FittedResponseDivergenceRefusal::Numerical { reason };
        if target.dim() != (self.n_obs(), self.output_dim()) {
            return Err(numerical(format!(
                "target {:?} != ({}, {})",
                target.dim(),
                self.n_obs(),
                self.output_dim()
            )));
        }
        if let Err(reason) = self.atom_second_jets() {
            return Err(FittedResponseDivergenceRefusal::SecondJetsUnavailable { reason });
        }
        let dim = sae_exact_stationarity_dim(cache.delta_t_len(), cache.k);
        if sae_exact_stationarity_admitted(dim, self.host_available_bytes) {
            let (divergence, likelihood_residual_dof, raw_residual_dof) = self
                .exact_spectral_fitted_response(rho, target, cache)
                .map_err(numerical)?;
            Ok(FittedResponseDivergence {
                divergence,
                likelihood_residual_dof,
                raw_residual_dof,
                estimator: FittedResponseDivergenceEstimator::ExactSpectral,
            })
        } else {
            // The divergence shares the ARD trace lane's probe budget and seed: both
            // are grouped Hutchinson traces of an arrow inverse at the same scale.
            let probes = Self::ARD_TRACE_HUTCHINSON_PROBES;
            let (divergence, standard_error) = self
                .hutchinson_fitted_response_divergence(
                    rho,
                    target,
                    cache,
                    probes,
                    Self::ARD_TRACE_HUTCHINSON_SEED,
                )
                .map_err(numerical)?;
            // The residual dofs are probed in output space, where each probe's
            // `‖z − Rz‖²` is non-negative, on streams disjoint from the divergence's.
            let likelihood_residual_dof = self
                .hutchinson_fitted_response_residual_dof(
                    rho,
                    target,
                    cache,
                    probes,
                    Self::ARD_TRACE_HUTCHINSON_SEED.wrapping_add(1 << 32),
                    FittedResponseFrame::Likelihood,
                )
                .map_err(numerical)?;
            let raw_residual_dof = if self.data_curvature_metric().map_err(numerical)?.is_some() {
                self.hutchinson_fitted_response_residual_dof(
                    rho,
                    target,
                    cache,
                    probes,
                    Self::ARD_TRACE_HUTCHINSON_SEED.wrapping_add(2 << 32),
                    FittedResponseFrame::Raw,
                )
                .map_err(numerical)?
            } else {
                likelihood_residual_dof
            };
            Ok(FittedResponseDivergence {
                divergence,
                likelihood_residual_dof,
                raw_residual_dof,
                estimator: FittedResponseDivergenceEstimator::Hutchinson {
                    probes,
                    standard_error,
                },
            })
        }
    }

    /// `tr(A⁺G)` and each frame's residual dof off the materialized exact
    /// stationarity eigensystem.
    ///
    /// With the retained eigenpairs `(λᵢ, vᵢ)` and `W_x = VᵀG_xV`,
    /// `tr(A⁺G) = Σᵢ Wᵢᵢ/λᵢ` and `tr(A⁺G_aA⁺G_b) = Σᵢⱼ (W_a)ᵢⱼ(W_b)ⱼᵢ/(λᵢλⱼ)`. The
    /// likelihood frame's `‖Ω^{½}RΩ^{−½}‖²_F` is `tr(A⁺GA⁺G)`. Under a whitening metric
    /// the raw frame's `‖R‖²_F` for `R = JA⁺JᵀΩ` is `tr(A⁺G_aA⁺G_b)` with
    /// `G_a = JᵀΩ²J` and `G_b = JᵀJ`.
    fn exact_spectral_fitted_response(
        &self,
        rho: &SaeManifoldRho,
        target: ArrayView2<'_, f64>,
        cache: &ArrowFactorCache,
    ) -> Result<(f64, f64, f64), String> {
        let geometry = self.materialize_exact_stationarity_geometry(rho, target, cache)?;
        let dim = geometry.eigenvalues.len();
        if geometry.eigenvectors.dim() != (dim, dim) {
            return Err(format!(
                "exact spectral divergence: eigenvectors {:?} do not match the \
                 {dim}-dimensional spectrum",
                geometry.eigenvectors.dim()
            ));
        }
        let retained: Vec<usize> = (0..dim)
            .filter(|&index| geometry.eigenvalues[index].abs() > geometry.rank_floor(index))
            .collect();
        let basis = geometry.eigenvectors.select(ndarray::Axis(1), &retained);
        let inverse: Vec<f64> = retained
            .iter()
            .map(|&index| 1.0 / geometry.eigenvalues[index])
            .collect();
        let project = |gram: ResponseOutputGram| -> Result<Array2<f64>, String> {
            let curvature = self.materialize_data_gauss_newton_dense(cache, gram)?;
            if curvature.dim() != (dim, dim) {
                return Err(format!(
                    "exact spectral divergence: data curvature {:?} does not match the \
                     {dim}-dimensional spectrum",
                    curvature.dim()
                ));
            }
            Ok(basis.t().dot(&curvature.dot(&basis)))
        };
        let paired_trace = |left: &Array2<f64>, right: &Array2<f64>| -> f64 {
            let mut total = 0.0_f64;
            for i in 0..inverse.len() {
                for j in 0..inverse.len() {
                    total += left[[i, j]] * right[[j, i]] * inverse[i] * inverse[j];
                }
            }
            total
        };
        let likelihood = project(ResponseOutputGram::Likelihood)?;
        let divergence: f64 = (0..inverse.len())
            .map(|i| likelihood[[i, i]] * inverse[i])
            .sum();
        let (likelihood_scalars, raw_scalars) = self.fitted_response_scalar_counts()?;
        let likelihood_residual_dof =
            likelihood_scalars - 2.0 * divergence + paired_trace(&likelihood, &likelihood);
        let raw_residual_dof = if self.data_curvature_metric()?.is_some() {
            let squared = project(ResponseOutputGram::SquaredLikelihood)?;
            let unweighted = project(ResponseOutputGram::UnweightedRaw)?;
            raw_scalars - 2.0 * divergence + paired_trace(&squared, &unweighted)
        } else {
            likelihood_residual_dof
        };
        for (label, value) in [
            ("divergence", divergence),
            ("likelihood residual dof", likelihood_residual_dof),
            ("raw residual dof", raw_residual_dof),
        ] {
            if !value.is_finite() {
                return Err(format!("exact spectral {label} is non-finite: {value}"));
            }
        }
        Ok((divergence, likelihood_residual_dof, raw_residual_dof))
    }

    /// Scalar observations of the likelihood frame and of the raw frame on the
    /// rows with positive weight. A zero-weight row is excluded from estimation,
    /// so it carries no scalar in either frame.
    pub(crate) fn fitted_response_scalar_counts(&self) -> Result<(f64, f64), String> {
        let live_rows = match self.row_loss_weights.as_deref() {
            Some(weights) => weights.iter().filter(|&&weight| weight > 0.0).count(),
            None => self.n_obs(),
        };
        let channels = match self.data_curvature_metric()? {
            Some(metric) => metric.metric_rank(),
            None => self.output_dim(),
        };
        Ok((
            (live_rows * channels) as f64,
            (live_rows * self.output_dim()) as f64,
        ))
    }

    /// Output-space Rademacher estimate of one frame's residual dof `‖I − R‖²_F`.
    ///
    /// For `z` with `E zzᵀ = I`, `E‖(I − R)z‖² = ‖I − R‖²_F`. Each probe is
    /// non-negative, whereas `N − 2 tr R + ‖R‖²_F` from separate trace estimates can
    /// cancel to either sign near interpolation. On the likelihood frame
    /// `Ω^{½} = √wᵢ Uᵢᵀ` with `Mᵢ = UᵢUᵢᵀ` (`√wᵢ I` unwhitened), so a probe contracts
    /// `J̃ᵢᵀUᵢzᵢ`, solves `A⁺`, and reads `UᵢᵀJ̃ᵢu`. On the raw frame `R = JA⁺JᵀΩ`, so
    /// it contracts `J̃ᵢᵀ√wᵢMᵢzᵢ` and reads `J̃ᵢu/√wᵢ`.
    fn hutchinson_fitted_response_residual_dof(
        &self,
        rho: &SaeManifoldRho,
        target: ArrayView2<'_, f64>,
        cache: &ArrowFactorCache,
        probes: usize,
        seed: u64,
        frame: FittedResponseFrame,
    ) -> Result<f64, String> {
        let n = self.n_obs();
        let p = self.output_dim();
        let total_t = cache.delta_t_len();
        let k = cache.k;
        let second_jets = self.atom_second_jets()?;
        let border = self.border_channels_for_cache(cache)?;
        let metric = self.data_curvature_metric()?;
        let prepared = self.prepare_decoder_prior_beta_curvature(1.0);
        let residual = self.prepare_residual_curvature_rows(target, cache)?;
        let apply_a = |vector: &SaeArrowVector| -> Result<SaeArrowVector, String> {
            self.apply_exact_hessian_prepared(rho, cache, vector, &prepared, &residual)
        };
        let apply_b = |vector: &SaeArrowVector| -> Result<SaeArrowVector, String> {
            crate::manifold::arrow_solver::apply_cached_arrow_hessian(
                cache,
                vector.t.view(),
                vector.beta.view(),
            )
        };
        let apply_b_raw = |vector: &SaeArrowVector| -> Result<SaeArrowVector, String> {
            apply_raw_cached_arrow_hessian(cache, vector.t.view(), vector.beta.view())
        };
        let weights = self.row_loss_weights.as_deref();
        let probes = probes.max(1);
        let mut accumulated = 0.0_f64;
        for probe in 0..probes {
            let mut state = seed.wrapping_add(probe as u64);
            let mut bits = 0u64;
            let mut remaining = 0u32;
            let mut draw = || {
                if remaining == 0 {
                    bits = gam_linalg::utils::splitmix64(&mut state);
                    remaining = 64;
                }
                let value = if bits & 1 == 1 { 1.0 } else { -1.0 };
                bits >>= 1;
                remaining -= 1;
                value
            };
            let mut rhs = SaeArrowVector {
                t: Array1::<f64>::zeros(total_t),
                beta: Array1::<f64>::zeros(k),
            };
            let mut row_probes: Vec<Vec<f64>> = Vec::with_capacity(n);
            let mut window: std::collections::VecDeque<SaeRowJets> =
                std::collections::VecDeque::new();
            let mut next = 0usize;
            for row in 0..n {
                if window.is_empty() {
                    next = self.refill_jet_window(next, cache, &second_jets, &border, &mut window)?;
                }
                let jets = window.pop_front().ok_or_else(|| {
                    format!("residual dof probe: the jet refill built no row {row}")
                })?;
                let weight = weights.map_or(1.0, |weights| weights[row]);
                if !(weight > 0.0) {
                    row_probes.push(Vec::new());
                    continue;
                }
                let base = cache.row_offsets[row];
                let q = cache.row_dims[row];
                if jets.vars.len() != q {
                    return Err(format!(
                        "residual dof probe: row {row} jets have {} variables, cache row has {q}",
                        jets.vars.len()
                    ));
                }
                let z: Vec<f64> = match (frame, metric) {
                    (FittedResponseFrame::Likelihood, Some(metric)) => {
                        (0..metric.metric_rank()).map(|_| draw()).collect()
                    }
                    _ => (0..p).map(|_| draw()).collect(),
                };
                let output: Vec<f64> = match (frame, metric) {
                    (FittedResponseFrame::Likelihood, Some(metric)) => (0..p)
                        .map(|out| {
                            z.iter()
                                .enumerate()
                                .map(|(column, &value)| metric.factor_entry(row, out, column) * value)
                                .sum()
                        })
                        .collect(),
                    (FittedResponseFrame::Raw, Some(metric)) => metric
                        .apply_metric_row(row, ndarray::aview1(&z))
                        .into_iter()
                        .map(|value| weight.sqrt() * value)
                        .collect(),
                    (_, None) => z.clone(),
                };
                for a in 0..q {
                    rhs.t[base + a] += sae_dot(jets.first(a), &output);
                }
                for (position, channel) in border.iter().enumerate() {
                    rhs.beta[channel.index] += sae_dot(jets.beta(position), &output);
                }
                row_probes.push(z);
            }
            let solved = solve_exact_stationarity_krylov(&rhs, &apply_a, &apply_b, &apply_b_raw)?;
            let mut window: std::collections::VecDeque<SaeRowJets> =
                std::collections::VecDeque::new();
            let mut next = 0usize;
            let mut response = vec![0.0_f64; p];
            for row in 0..n {
                if window.is_empty() {
                    next = self.refill_jet_window(next, cache, &second_jets, &border, &mut window)?;
                }
                let jets = window.pop_front().ok_or_else(|| {
                    format!("residual dof probe: the jet refill built no row {row}")
                })?;
                let z = &row_probes[row];
                if z.is_empty() {
                    continue;
                }
                let base = cache.row_offsets[row];
                let q = cache.row_dims[row];
                response.fill(0.0);
                for a in 0..q {
                    let coefficient = solved.t[base + a];
                    for (slot, &value) in response.iter_mut().zip(jets.first(a)) {
                        *slot += coefficient * value;
                    }
                }
                for (position, channel) in border.iter().enumerate() {
                    let coefficient = solved.beta[channel.index];
                    for (slot, &value) in response.iter_mut().zip(jets.beta(position)) {
                        *slot += coefficient * value;
                    }
                }
                let weight = weights.map_or(1.0, |weights| weights[row]);
                let image: Vec<f64> = match (frame, metric) {
                    (FittedResponseFrame::Likelihood, Some(metric)) => {
                        metric.whiten_residual_row(row, ndarray::aview1(&response))
                    }
                    (FittedResponseFrame::Raw, Some(_)) => {
                        response.iter().map(|value| value / weight.sqrt()).collect()
                    }
                    (_, None) => response.clone(),
                };
                accumulated += z
                    .iter()
                    .zip(&image)
                    .map(|(probe_value, image_value)| (probe_value - image_value).powi(2))
                    .sum::<f64>();
            }
        }
        let residual_dof = accumulated / probes as f64;
        if !residual_dof.is_finite() {
            return Err(format!("Hutchinson residual dof is non-finite: {residual_dof}"));
        }
        Ok(residual_dof)
    }

    /// Rademacher Hutchinson estimate of `tr(A⁺G)` and its standard error: each
    /// probe is one data curvature apply and one Krylov exact-stationarity solve
    /// against the cached arrow factorization, so nothing `dim × dim` is formed.
    /// The standard error is the sample standard deviation of the probe values
    /// over `√probes`.
    fn hutchinson_fitted_response_divergence(
        &self,
        rho: &SaeManifoldRho,
        target: ArrayView2<'_, f64>,
        cache: &ArrowFactorCache,
        probes: usize,
        seed: u64,
    ) -> Result<(f64, f64), String> {
        let total_t = cache.delta_t_len();
        let k = cache.k;
        let second_jets = self.atom_second_jets()?;
        let border = self.border_channels_for_cache(cache)?;
        let prepared = self.prepare_decoder_prior_beta_curvature(1.0);
        let residual = self.prepare_residual_curvature_rows(target, cache)?;
        let apply_a = |vector: &SaeArrowVector| -> Result<SaeArrowVector, String> {
            self.apply_exact_hessian_prepared(rho, cache, vector, &prepared, &residual)
        };
        let apply_b = |vector: &SaeArrowVector| -> Result<SaeArrowVector, String> {
            crate::manifold::arrow_solver::apply_cached_arrow_hessian(
                cache,
                vector.t.view(),
                vector.beta.view(),
            )
        };
        let apply_b_raw = |vector: &SaeArrowVector| -> Result<SaeArrowVector, String> {
            apply_raw_cached_arrow_hessian(cache, vector.t.view(), vector.beta.view())
        };
        if probes < 2 {
            return Err(format!(
                "Hutchinson divergence needs at least two probes for a standard error; got {probes}"
            ));
        }
        let mut values = Vec::with_capacity(probes);
        for probe in 0..probes {
            let mut state = seed.wrapping_add(probe as u64);
            let mut bits = 0u64;
            let mut remaining = 0u32;
            let mut draw = || {
                if remaining == 0 {
                    bits = gam_linalg::utils::splitmix64(&mut state);
                    remaining = 64;
                }
                let value = if bits & 1 == 1 { 1.0 } else { -1.0 };
                bits >>= 1;
                remaining -= 1;
                value
            };
            let z = SaeArrowVector {
                t: Array1::from_shape_fn(total_t, |_| draw()),
                beta: Array1::from_shape_fn(k, |_| draw()),
            };
            let gz = self.apply_data_gauss_newton(cache, &second_jets, &border, &z)?;
            let response = solve_exact_stationarity_krylov(&gz, &apply_a, &apply_b, &apply_b_raw)?;
            values.push(z.t.dot(&response.t) + z.beta.dot(&response.beta));
        }
        let count = probes as f64;
        let divergence = values.iter().sum::<f64>() / count;
        let spread = values
            .iter()
            .map(|value| (value - divergence) * (value - divergence))
            .sum::<f64>();
        let standard_error = (spread / (count - 1.0) / count).sqrt();
        if !(divergence.is_finite() && standard_error.is_finite()) {
            return Err(format!(
                "Hutchinson divergence is non-finite: {divergence} (standard error {standard_error})"
            ));
        }
        Ok((divergence, standard_error))
    }

    /// The whitening metric the data likelihood is assembled through, or `None`
    /// when the metric is the identity.
    fn data_curvature_metric(&self) -> Result<Option<&gam_problem::RowMetric>, String> {
        if !self.whiten_logdet_row_jets() {
            return Ok(None);
        }
        self.row_metric
            .as_ref()
            .map(Some)
            .ok_or_else(|| "data curvature: whitening metric absent".to_string())
    }

    /// `G v` for the data Gauss--Newton curvature `G = Σᵢ J̃ᵢᵀ Mᵢ J̃ᵢ`, in the joint
    /// `(t, β)` cache layout, read off the `√w`-weighted row jets the arrow
    /// assembly differentiates.
    fn apply_data_gauss_newton(
        &self,
        cache: &ArrowFactorCache,
        second_jets: &[Array4<f64>],
        border: &[SaeBorderChannel],
        v: &SaeArrowVector,
    ) -> Result<SaeArrowVector, String> {
        let n = self.n_obs();
        let p = self.output_dim();
        let total_t = cache.delta_t_len();
        if v.t.len() != total_t || v.beta.len() != cache.k {
            return Err(format!(
                "data curvature apply: vector (t={}, beta={}) != cache (t={total_t}, beta={})",
                v.t.len(),
                v.beta.len(),
                cache.k
            ));
        }
        let metric = self.data_curvature_metric()?;
        let mut out = SaeArrowVector {
            t: Array1::<f64>::zeros(total_t),
            beta: Array1::<f64>::zeros(cache.k),
        };
        let mut window: std::collections::VecDeque<SaeRowJets> = std::collections::VecDeque::new();
        let mut next = 0usize;
        let mut response = vec![0.0_f64; p];
        for row in 0..n {
            if window.is_empty() {
                next = self.refill_jet_window(next, cache, second_jets, border, &mut window)?;
            }
            let jets = window
                .pop_front()
                .ok_or_else(|| format!("data curvature apply: the jet refill built no row {row}"))?;
            let base = cache.row_offsets[row];
            let q = cache.row_dims[row];
            if jets.vars.len() != q {
                return Err(format!(
                    "data curvature apply: row {row} jets have {} variables, cache row has {q}",
                    jets.vars.len()
                ));
            }
            response.fill(0.0);
            for a in 0..q {
                let coefficient = v.t[base + a];
                if coefficient != 0.0 {
                    for (slot, &value) in response.iter_mut().zip(jets.first(a)) {
                        *slot += coefficient * value;
                    }
                }
            }
            for (position, channel) in border.iter().enumerate() {
                let coefficient = v.beta[channel.index];
                if coefficient != 0.0 {
                    for (slot, &value) in response.iter_mut().zip(jets.beta(position)) {
                        *slot += coefficient * value;
                    }
                }
            }
            let metric_response = match metric {
                Some(metric) => metric.apply_metric_row(row, ndarray::aview1(&response)),
                None => response.clone(),
            };
            for a in 0..q {
                out.t[base + a] = sae_dot(jets.first(a), &metric_response);
            }
            for (position, channel) in border.iter().enumerate() {
                out.beta[channel.index] += sae_dot(jets.beta(position), &metric_response);
            }
        }
        Ok(out)
    }

    /// The data Gauss--Newton curvature `G = Σᵢ J̃ᵢᵀ Mᵢ J̃ᵢ` as a dense matrix in the
    /// joint `(t, β)` cache layout. No prior, penalty, or residual curvature
    /// enters: those belong to `A`, and `G` is what the target perturbation reaches
    /// the stationarity through.
    ///
    /// A border channel's jet is `sᵢ_b u_b`, a scalar (`√wᵢ·a_k·Φ_b(tᵢ)`) times its
    /// output class vector, so the border block accumulates
    /// `Σᵢ sᵢ sᵢᵀ ∘ Ωᵢ` with `Ωᵢ[c, c'] = u_cᵀ Mᵢ u_c'`. The scalar is read back off
    /// the emitted jet (`⟨jet, u⟩/⟨u, u⟩`), so the row program stays the only
    /// definition of the channel.
    ///
    /// `output_gram` names the per-row output inner product (see
    /// [`ResponseOutputGram`]): the likelihood metric for `G` itself, or the two
    /// raw-frame Grams a whitened residual dof pairs.
    fn materialize_data_gauss_newton_dense(
        &self,
        cache: &ArrowFactorCache,
        output_gram: ResponseOutputGram,
    ) -> Result<Array2<f64>, String> {
        let n = self.n_obs();
        let total_t = cache.delta_t_len();
        let dim = sae_exact_stationarity_dim(total_t, cache.k);
        let second_jets = self.atom_second_jets()?;
        let border = self.border_channels_for_cache(cache)?;
        let metric = self.data_curvature_metric()?;
        let weights = self.row_loss_weights.as_deref();
        let (class_of, class_outputs) = border_output_classes(&border);
        let n_classes = class_outputs.len();
        let class_norm_sq: Vec<f64> = class_outputs.iter().map(|u| sae_dot(u, u)).collect();
        let gram = |left: &[Vec<f64>], right: &[Vec<f64>]| -> Vec<f64> {
            let mut omega = vec![0.0_f64; n_classes * n_classes];
            for (c, u) in left.iter().enumerate() {
                for (d, v) in right.iter().enumerate() {
                    omega[c * n_classes + d] = sae_dot(u, v);
                }
            }
            omega
        };
        let unwhitened_omega = (metric.is_none() && output_gram == ResponseOutputGram::Likelihood)
            .then(|| gram(&class_outputs, &class_outputs));
        let mut g = Array2::<f64>::zeros((dim, dim));
        let mut window: std::collections::VecDeque<SaeRowJets> = std::collections::VecDeque::new();
        let mut next = 0usize;
        let mut scalars = vec![0.0_f64; border.len()];
        let mut live: Vec<usize> = Vec::with_capacity(border.len());
        for row in 0..n {
            if window.is_empty() {
                next = self.refill_jet_window(next, cache, &second_jets, &border, &mut window)?;
            }
            let jets = window.pop_front().ok_or_else(|| {
                format!("data curvature materialization: the jet refill built no row {row}")
            })?;
            let base = cache.row_offsets[row];
            let q = cache.row_dims[row];
            if jets.vars.len() != q {
                return Err(format!(
                    "data curvature materialization: row {row} jets have {} variables, cache \
                     row has {q}",
                    jets.vars.len()
                ));
            }
            let weight = weights.map_or(1.0, |weights| weights[row]);
            let metric_apply = |values: &[f64]| -> Vec<f64> {
                match metric {
                    Some(metric) => metric.apply_metric_row(row, ndarray::aview1(values)),
                    None => values.to_vec(),
                }
            };
            let apply = |values: &[f64]| -> Vec<f64> {
                match output_gram {
                    ResponseOutputGram::Likelihood => metric_apply(values),
                    ResponseOutputGram::SquaredLikelihood => metric_apply(&metric_apply(values))
                        .into_iter()
                        .map(|value| weight * value)
                        .collect(),
                    ResponseOutputGram::UnweightedRaw if weight > 0.0 => {
                        values.iter().map(|value| value / weight).collect()
                    }
                    ResponseOutputGram::UnweightedRaw => vec![0.0; values.len()],
                }
            };
            let metric_first: Vec<Vec<f64>> = (0..q).map(|a| apply(jets.first(a))).collect();
            for a in 0..q {
                for b in 0..q {
                    g[[base + a, base + b]] += sae_dot(jets.first(a), &metric_first[b]);
                }
            }
            live.clear();
            for (position, &class) in class_of.iter().enumerate() {
                scalars[position] = if class_norm_sq[class] > 0.0 {
                    sae_dot(jets.beta(position), &class_outputs[class]) / class_norm_sq[class]
                } else {
                    0.0
                };
                if scalars[position] != 0.0 {
                    live.push(position);
                }
            }
            if live.is_empty() {
                continue;
            }
            let row_omega;
            let omega: &[f64] = match unwhitened_omega.as_ref() {
                Some(omega) => omega.as_slice(),
                None => {
                    let metric_outputs: Vec<Vec<f64>> =
                        class_outputs.iter().map(|u| apply(u)).collect();
                    row_omega = gram(&class_outputs, &metric_outputs);
                    row_omega.as_slice()
                }
            };
            for a in 0..q {
                let first_dot_class: Vec<f64> = class_outputs
                    .iter()
                    .map(|u| sae_dot(&metric_first[a], u))
                    .collect();
                for &position in &live {
                    let value = scalars[position] * first_dot_class[class_of[position]];
                    let border_slot = total_t + border[position].index;
                    g[[base + a, border_slot]] += value;
                    g[[border_slot, base + a]] += value;
                }
            }
            for &left in &live {
                let left_slot = total_t + border[left].index;
                let left_class = class_of[left];
                for &right in &live {
                    g[[left_slot, total_t + border[right].index]] += scalars[left]
                        * scalars[right]
                        * omega[left_class * n_classes + class_of[right]];
                }
            }
        }
        Ok(g)
    }
}
