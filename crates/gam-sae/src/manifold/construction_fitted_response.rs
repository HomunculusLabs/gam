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
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct FittedResponseDivergence {
    pub(crate) divergence: f64,
    pub(crate) estimator: FittedResponseDivergenceEstimator,
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
            let divergence = self
                .exact_spectral_fitted_response_divergence(rho, target, cache)
                .map_err(numerical)?;
            Ok(FittedResponseDivergence {
                divergence,
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
            Ok(FittedResponseDivergence {
                divergence,
                estimator: FittedResponseDivergenceEstimator::Hutchinson {
                    probes,
                    standard_error,
                },
            })
        }
    }

    /// `tr(A⁺G) = Σ_retained vᵢᵀ G vᵢ / λᵢ` off the materialized exact
    /// stationarity eigensystem.
    fn exact_spectral_fitted_response_divergence(
        &self,
        rho: &SaeManifoldRho,
        target: ArrayView2<'_, f64>,
        cache: &ArrowFactorCache,
    ) -> Result<f64, String> {
        let geometry = self.materialize_exact_stationarity_geometry(rho, target, cache)?;
        let data_curvature = self.materialize_data_gauss_newton_dense(cache)?;
        let dim = geometry.eigenvalues.len();
        if data_curvature.dim() != (dim, dim) || geometry.eigenvectors.dim() != (dim, dim) {
            return Err(format!(
                "exact spectral divergence: data curvature {:?} and eigenvectors {:?} do not \
                 match the {dim}-dimensional spectrum",
                data_curvature.dim(),
                geometry.eigenvectors.dim()
            ));
        }
        let applied = data_curvature.dot(&geometry.eigenvectors);
        let mut divergence = 0.0_f64;
        for index in 0..dim {
            let lambda = geometry.eigenvalues[index];
            if lambda.abs() > geometry.rank_floor(index) {
                divergence += geometry
                    .eigenvectors
                    .column(index)
                    .dot(&applied.column(index))
                    / lambda;
            }
        }
        if !divergence.is_finite() {
            return Err(format!("exact spectral divergence is non-finite: {divergence}"));
        }
        Ok(divergence)
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
    fn materialize_data_gauss_newton_dense(
        &self,
        cache: &ArrowFactorCache,
    ) -> Result<Array2<f64>, String> {
        let n = self.n_obs();
        let total_t = cache.delta_t_len();
        let dim = sae_exact_stationarity_dim(total_t, cache.k);
        let second_jets = self.atom_second_jets()?;
        let border = self.border_channels_for_cache(cache)?;
        let metric = self.data_curvature_metric()?;
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
        let unwhitened_omega = metric.is_none().then(|| gram(&class_outputs, &class_outputs));
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
            let apply = |values: &[f64]| -> Vec<f64> {
                match metric {
                    Some(metric) => metric.apply_metric_row(row, ndarray::aview1(values)),
                    None => values.to_vec(),
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
