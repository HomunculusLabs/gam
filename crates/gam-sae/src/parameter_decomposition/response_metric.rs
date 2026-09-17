//! Finite-intervention response metric and local state coordinates (#2951,
//! Doc B §9.3).
//!
//! # The metric
//!
//! A computational state `h ∈ ℝᵈ` (a per-input coordinate, never a
//! parameter-family label) is followed by a declared finite family of
//! interventions `α`. Each intervention is a native execution
//! `R_α : ℝᵈ → ℝ^{p_α}` from the state to a response, with an analytic Jacobian
//! `DR_α`. The experiment declares a finite measure `ν` on the family and, per
//! intervention, an output metric `M_α = L_αᵀ L_α` through its factor `L_α`
//! (`q_α × p_α`). The finite-intervention response metric is the pullback of the
//! declared output metrics,
//!
//! ```text
//! G(h) = Σ_α ν_α DR_α(h)ᵀ M_α DR_α(h) = A(h)ᵀ A(h),
//! A(h) = the rows √ν_α L_α DR_α(h), stacked over α.
//! ```
//!
//! Each member `S_α = √ν_α L_α R_α` ([`WhitenedResponse`]) is itself a native map,
//! so a finite response difference is measured in the metric that `G` measures to
//! first order:
//! `Σ_α ν_α (R_α h − R_α h')ᵀ M_α (R_α h − R_α h') = Σ_α ‖S_α h − S_α h'‖₂²`.
//!
//! * `G` is never formed. Its spectrum is read off the factor, `λᵢ(G) = σᵢ(A)²`
//!   ([`metric_eigenvalues`]), so a direction is resolved at the factor's own
//!   conditioning, not at its square. The factor is `(Σ_α q_α) × d` per state and
//!   its rank is resolved by [`constant_rank_check`](super::state::constant_rank_check).
//! * `ker G(h) = ⋂_α ker(L_α DR_α(h))`. A factor with a kernel removes the output
//!   directions the metric does not charge: with `L = [½, −½]` on two logits, a
//!   shift of both logits is invisible, as it is to their softmax.
//! * A finite family under a finite measure gives an exact sum. A family drawn from
//!   a law would make `G` an estimate of the law's metric, which this module does
//!   not claim.
//!
//! # Local state coordinates
//!
//! At an anchor `h₀` the resolved row space of `A(h₀)` gives `k` directions `V`
//! and the linear chart `E(h) = V*(h − h₀)` with section `D(z) = h₀ + V*ᵀ z`
//! ([`LinearStateChart`]). `V*` is the polar factor of `V`: it spans the same row
//! space with exactly orthonormal rows, so `E∘D = id` holds exactly, and every
//! roundoff bound of the chart carries `‖V − V*‖₂ ≤ ‖V Vᵀ − I‖₂`.
//!
//! * **Dimension.** The resolved rank `k` is a certified lower bound on the exact
//!   rank of `G(h₀)`. Where it reaches its ceiling `min(Σ_α q_α, d)` it is exact
//!   and, by lower semicontinuity, constant on a neighbourhood of `h₀`: the
//!   response map has certified constant rank there. The constant-rank theorem
//!   then gives local coordinates of dimension `k` whose differential at `h₀`
//!   spans the chart's directions, and for a generating family `k` is the local
//!   minimal dimension. Below the ceiling only `[k, ceiling]` is derived.
//! * **Directions.** Under the SVD's backward-error model, Wedin's theorem puts the
//!   computed directions within `band / (σ̂_k − σ̂_{k+1} − band)` (the sine of the
//!   largest principal angle) of the leading `k`-dimensional right singular
//!   subspace of the exact `A(h₀)`, through
//!   [`projector_error_bar`](gam_linalg::decision::projector_error_bar). Where the
//!   rank reaches `Σ_α q_α` the exact `σ_{k+1}` is zero.
//!
//! # The singular-point caveat: finite checks, not the kernel alone
//!
//! The kernel of `G(h₀)` names the directions the interventions do not see to
//! first order. It does not say the responses are locally constant along them.
//! `R(t) = t²` has `DR(0) = 0`, so `G(0) = 0` and the whole line is kernel, yet
//! `R(s) − R(0) = s²`. A regular point can fail the same way when the fibre is
//! curved: `R(a, b) = a² + b²` has certified constant rank 1 at `(1, 0)` with
//! kernel `e_b`, and `R(1, s) = 1 + s²`. The two refusals say different things.
//! At a rank drop no chart of the kernel's codimension need exist. At a certified
//! constant-rank point a curved chart (here the radius) exists, and it is the
//! linear chart that is not sufficient at that scale.
//!
//! So no chart is accepted from the kernel. [`LocalStateCoordinates::finite_check`]
//! executes the whitened responses at declared finite states and compares each
//! state with its section representative `D(E h)`, which differs from it only
//! along the kernel, through [`fiber_test`](super::state::fiber_test). A
//! separation above the declared fidelity refutes the chart at those states.
//! Agreement is a bound over the stated states only. The states and the fidelity
//! are experiment declarations with no default.
//!
//! # Categorical responses compared in KL
//!
//! When a response is a logit vector `z_α(h)` whose fidelity is declared in KL,
//! the finite check is the weighted divergence
//! `Σ_α ν_α KL(softmax z_α(h) ‖ softmax z_α(D(E h)))`
//! ([`CategoricalFamily::divergence_check`]). To second order in `v = h − D(E h)`
//! it is `½ vᵀ G v` for the output metric `F(z) = diag p − p pᵀ`, the Hessian of
//! `z′ ↦ KL(softmax z ‖ softmax z′)` at `z′ = z`, which does not charge a common
//! logit shift.
//!
//! Each divergence comes from
//! [`categorical_kl_from_logits_with_error`](gam_math::categorical::categorical_kl_from_logits_with_error),
//! whose bound covers the evaluation of the stored logits. The logits' own
//! roundoff `e` (sup norm, at both ends) adds `e(2 + osc(δ̂) + 4e)` with
//! `δ̂ = ẑ′ − ẑ`. Along the segment to the exact pair, `∇_{z′} KL = p′ − p` has
//! `ℓ₁` norm at most 2, and `∇_z KL = −F(z) δ` has `ℓ₁` norm
//! `Σᵢ pᵢ|δᵢ − E_p δ| ≤ osc(δ)`, where `δ` moves by at most `2e` per coordinate
//! and so its oscillation by at most `4e`.

use std::fmt;

use gam_linalg::decision::projector_error_bar;
use gam_linalg::roundoff::{accumulation_band, accumulation_growth};
use gam_math::categorical::{CategoricalError, categorical_kl_from_logits_with_error};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};

use super::state::{
    ConstantRankCheck, DifferentiableNativeMap, Evaluation, FiberVerdict, JacobianEvaluation,
    LocalQuotient, NativeMap, StateChart, StateDomain, StateError, StateRow, constant_rank_check,
    fiber_test,
};
use super::supports::{EvidenceStatus, EvidenceStatusError, Extremum};

/// One declared intervention of the family: its native response, its mass under
/// the declared finite measure, and the factor of its output metric.
pub struct ResponseMember<'a> {
    /// `R_α : ℝᵈ → ℝ^{p_α}`, with an analytic Jacobian.
    pub response: &'a dyn DifferentiableNativeMap,
    /// `ν_α`, finite and positive. An intervention without mass is not in the
    /// family.
    pub weight: f64,
    /// `L_α`, `q_α × p_α` with `q_α ≥ 1`, so that `M_α = L_αᵀ L_α`.
    pub metric_factor: Array2<f64>,
}

/// A member in the metric's own coordinates, `S_α(h) = √ν_α L_α R_α(h)`, with
/// derived roundoff.
///
/// Its values measure finite response differences in the declared metric, and
/// its Jacobians stack into the factor `A` of `G`.
pub struct WhitenedResponse<'a> {
    response: &'a dyn DifferentiableNativeMap,
    /// `fl(√ν_α)`. The square root is correctly rounded, so it lies within
    /// `u·√ν_α` of the exact root.
    weight_root: f64,
    metric_factor: Array2<f64>,
    /// `|L_α|`, entrywise.
    absolute_factor: Array2<f64>,
    /// `‖row i of L_α‖₁`, rounded up.
    factor_row_sums: Array1<f64>,
    /// `‖L_α‖_F`, rounded up: a bound on `‖L_α‖₂`.
    factor_norm_upper: f64,
}

impl NativeMap for WhitenedResponse<'_> {
    fn input_dimension(&self) -> usize {
        self.response.input_dimension()
    }

    fn output_dimension(&self) -> usize {
        self.metric_factor.nrows()
    }

    /// Row `i` of `fl(ŝ · fl(L r̂))` is within
    /// `ŝ(1 + γ₁)(2u|ŷᵢ| + γ_p Σⱼ|Lᵢⱼ r̂ⱼ| + ‖Lᵢ‖₁ e_R)` of `√ν (L R(x))ᵢ` for every
    /// exact `x` the response's bound `e_R` covers: the scaling rounds once, the
    /// root within `u√ν`, the inner product within `γ_p`, and the response error
    /// passes through `L` by its row sums. The last factor covers the bound's own
    /// arithmetic.
    fn evaluate(
        &self,
        input: ArrayView1<'_, f64>,
        input_roundoff: f64,
    ) -> Result<Evaluation, StateError> {
        let response = self.response.evaluate(input, input_roundoff)?;
        let height = self.response.output_dimension();
        if response.value.len() != height {
            return Err(StateError::DimensionMismatch {
                context: "whitened response: response value",
                expected: height,
                found: response.value.len(),
            });
        }
        if !response.roundoff.is_finite()
            || response.roundoff < 0.0
            || response.value.iter().any(|value| !value.is_finite())
        {
            return Err(StateError::NonFinite {
                context: "whitened response: response value",
            });
        }
        let projected = self.metric_factor.dot(&response.value);
        let magnitudes = self.absolute_factor.dot(&response.value.mapv(f64::abs));
        let single = accumulation_growth(1);
        let mut row_bound = 0.0_f64;
        for ((value, magnitude), row_sum) in projected
            .iter()
            .zip(magnitudes.iter())
            .zip(self.factor_row_sums.iter())
        {
            row_bound = row_bound.max(
                2.0 * single * value.abs()
                    + accumulation_band(height, *magnitude)
                    + row_sum * response.roundoff,
            );
        }
        Ok(Evaluation {
            value: projected.mapv(|value| self.weight_root * value),
            roundoff: rounded_up(self.weight_root * growth_factor(6) * row_bound),
        })
    }
}

impl DifferentiableNativeMap for WhitenedResponse<'_> {
    /// `fl(ŝ · fl(L Ĵ))` is within `ŝ(1 + γ₁)(2u‖X̂‖_F + ‖B‖_F + ‖L‖₂ e_J)` of
    /// `√ν L J` in spectral norm, where `X̂ = fl(L Ĵ)`, `Bᵢⱼ = γ_p Σₗ|Lᵢₗ Ĵₗⱼ|`
    /// bounds the product's rounding entrywise, and `e_J` is the response
    /// Jacobian's formation bound.
    fn jacobian(&self, input: ArrayView1<'_, f64>) -> Result<JacobianEvaluation, StateError> {
        let jacobian = self.response.jacobian(input)?;
        let height = self.response.output_dimension();
        let dimension = self.response.input_dimension();
        if jacobian.matrix.dim() != (height, dimension) {
            return Err(StateError::DimensionMismatch {
                context: "whitened response: Jacobian shape",
                expected: height * dimension,
                found: jacobian.matrix.len(),
            });
        }
        if !jacobian.roundoff.is_finite()
            || jacobian.roundoff < 0.0
            || jacobian.matrix.iter().any(|value| !value.is_finite())
        {
            return Err(StateError::NonFinite {
                context: "whitened response: Jacobian",
            });
        }
        let projected = self.metric_factor.dot(&jacobian.matrix);
        let magnitudes = self.absolute_factor.dot(&jacobian.matrix.mapv(f64::abs));
        let formation = frobenius_upper(
            magnitudes
                .iter()
                .map(|&magnitude| accumulation_band(height, magnitude)),
        );
        let spread = frobenius_upper(projected.iter().copied());
        let bound = 2.0 * accumulation_growth(1) * spread
            + formation
            + self.factor_norm_upper * jacobian.roundoff;
        Ok(JacobianEvaluation {
            matrix: projected.mapv(|value| self.weight_root * value),
            roundoff: rounded_up(self.weight_root * growth_factor(6) * bound),
        })
    }
}

/// The finite-intervention response metric `G(h) = Σ_α ν_α DR_αᵀ M_α DR_α` of a
/// declared family.
pub struct ResponseMetric<'a> {
    members: Vec<WhitenedResponse<'a>>,
    dimension: usize,
}

impl<'a> ResponseMetric<'a> {
    /// Validate the declared family and whiten each member.
    ///
    /// Refusals: an empty family, members on different state spaces, a weight
    /// that is not finite and positive, a factor with no rows or whose columns do
    /// not match its response, and a non-finite factor.
    pub fn new(members: Vec<ResponseMember<'a>>) -> Result<Self, ResponseMetricError> {
        let dimension = members
            .first()
            .ok_or(ResponseMetricError::EmptyFamily)?
            .response
            .input_dimension();
        let mut whitened = Vec::with_capacity(members.len());
        for (index, member) in members.into_iter().enumerate() {
            let found = member.response.input_dimension();
            if found != dimension {
                return Err(ResponseMetricError::StateDimension {
                    member: index,
                    expected: dimension,
                    found,
                });
            }
            if !(member.weight.is_finite() && member.weight > 0.0) {
                return Err(ResponseMetricError::InvalidWeight {
                    member: index,
                    weight: member.weight,
                });
            }
            let (rows, columns) = member.metric_factor.dim();
            let response_dimension = member.response.output_dimension();
            if rows == 0 || columns != response_dimension {
                return Err(ResponseMetricError::MetricFactorShape {
                    member: index,
                    rows,
                    columns,
                    response_dimension,
                });
            }
            if member.metric_factor.iter().any(|value| !value.is_finite()) {
                return Err(ResponseMetricError::NonFiniteMetricFactor { member: index });
            }
            let absolute_factor = member.metric_factor.mapv(f64::abs);
            let factor_row_sums = absolute_factor
                .sum_axis(Axis(1))
                .mapv(|sum| rounded_up(sum * growth_factor(columns)));
            let factor_norm_upper = frobenius_upper(member.metric_factor.iter().copied());
            whitened.push(WhitenedResponse {
                response: member.response,
                weight_root: member.weight.sqrt(),
                metric_factor: member.metric_factor,
                absolute_factor,
                factor_row_sums,
                factor_norm_upper,
            });
        }
        Ok(Self {
            members: whitened,
            dimension,
        })
    }

    /// Dimension `d` of the state space.
    pub fn input_dimension(&self) -> usize {
        self.dimension
    }

    /// The whitened members `S_α`, in declaration order.
    pub fn members(&self) -> &[WhitenedResponse<'a>] {
        &self.members
    }

    /// The factor `A(h)` of `G(h)` resolved at each stated state, the rows of
    /// `states`: its singular values `σᵢ(A) = √λᵢ(G)`, the band, the certified
    /// lower bound on the rank, its ceiling and the observed directions.
    pub fn at(&self, states: ArrayView2<'_, f64>) -> Result<ConstantRankCheck, ResponseMetricError> {
        let family: Vec<&dyn DifferentiableNativeMap> = self
            .members
            .iter()
            .map(|member| member as &dyn DifferentiableNativeMap)
            .collect();
        Ok(constant_rank_check(&family, states)?)
    }

    /// Local state coordinates at `anchor`: the factor resolved there, the linear
    /// chart on its observed directions, and the Wedin bound on those directions.
    pub fn local_coordinates(
        &self,
        anchor: ArrayView1<'_, f64>,
    ) -> Result<LocalStateCoordinates, ResponseMetricError> {
        let rank = self.at(anchor.insert_axis(Axis(0)))?;
        let quotient = rank
            .states
            .first()
            .ok_or(ResponseMetricError::State(StateError::EmptyFamily {
                context: "response metric: anchor",
            }))?;
        let resolved = quotient.resolved_rank;
        let subspace_error_bar = if resolved == 0 || resolved == self.dimension {
            0.0
        } else {
            let leading = quotient.singular_values[resolved - 1];
            let trailing = quotient.singular_values.get(resolved).copied().unwrap_or(0.0);
            projector_error_bar(leading - trailing, quotient.band)
        };
        let chart = LinearStateChart::new(anchor.to_owned(), quotient.observed_directions.clone())?;
        Ok(LocalStateCoordinates {
            rank,
            chart,
            subspace_error_bar,
        })
    }
}

/// Bounds on one eigenvalue of `G(h) = AᵀA`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MetricEigenvalue {
    /// `max(0, σ̂ − band)²`, rounded down.
    pub lower: f64,
    /// `(σ̂ + band)²`, rounded up.
    pub upper: f64,
}

impl MetricEigenvalue {
    /// The bracket as the evidence status of the eigenvalue at its one stated
    /// state. Both sides are derived by Weyl, so it stays unresolved until they
    /// meet.
    pub fn evidence(&self) -> Result<EvidenceStatus<StateRow, StateDomain>, EvidenceStatusError> {
        EvidenceStatus::unresolved(
            self.lower,
            self.upper,
            Extremum::Supremum,
            None,
            StateDomain::States { count: 1 },
        )
    }
}

/// The eigenvalues of `G` at a resolved state, largest first, read off the
/// factor's computed singular values: by Weyl each exact `σᵢ(A)` lies within the
/// band of `σ̂ᵢ`, and `λᵢ(G) = σᵢ(A)²`. When the factor has fewer rows than `d`,
/// the remaining `d − rows` eigenvalues are exactly zero and are not listed.
pub fn metric_eigenvalues(quotient: &LocalQuotient) -> Vec<MetricEigenvalue> {
    let band = quotient.band;
    quotient
        .singular_values
        .iter()
        .map(|&sigma| {
            let (below, above) = if band == 0.0 {
                (sigma, sigma)
            } else {
                ((sigma - band).next_down().max(0.0), (sigma + band).next_up())
            };
            MetricEigenvalue {
                lower: if below == 0.0 {
                    0.0
                } else {
                    (below * below).next_down().max(0.0)
                },
                upper: rounded_up(above * above),
            }
        })
        .collect()
}

/// A linear chart `E(h) = V*(h − h₀)` with section `D(z) = h₀ + V*ᵀ z`, where `V*`
/// is the polar factor of the declared directions `V` (`k × d`).
#[derive(Clone, Debug)]
pub struct LinearStateChart {
    anchor: Array1<f64>,
    directions: Array2<f64>,
    absolute_directions: Array2<f64>,
    orthonormality_defect: f64,
}

impl LinearStateChart {
    /// A chart at `anchor` on the rows of `directions`.
    ///
    /// `‖V Vᵀ − I_k‖₂` is bounded by the computed residual's Frobenius norm plus
    /// its entrywise rounding `γ_{d+1}(Σₗ|Vᵢₗ Vⱼₗ| + δᵢⱼ)`. Below 1 it certifies
    /// full row rank, and then `‖V − V*‖₂ = maxᵢ|σᵢ(V) − 1| ≤ ‖V Vᵀ − I_k‖₂`.
    /// Refused when the bound is not below 1.
    pub fn new(anchor: Array1<f64>, directions: Array2<f64>) -> Result<Self, ResponseMetricError> {
        let dimension = directions.ncols();
        if anchor.len() != dimension {
            return Err(StateError::DimensionMismatch {
                context: "linear chart: anchor",
                expected: dimension,
                found: anchor.len(),
            }
            .into());
        }
        if anchor
            .iter()
            .chain(directions.iter())
            .any(|value| !value.is_finite())
        {
            return Err(StateError::NonFinite {
                context: "linear chart: anchor and directions",
            }
            .into());
        }
        let absolute_directions = directions.mapv(f64::abs);
        let gram = directions.dot(&directions.t());
        let magnitudes = absolute_directions.dot(&absolute_directions.t());
        let residual = frobenius_upper(gram.indexed_iter().map(|((row, column), &value)| {
            if row == column { value - 1.0 } else { value }
        }));
        let formation = frobenius_upper(magnitudes.indexed_iter().map(
            |((row, column), &magnitude)| {
                let identity = if row == column { 1.0 } else { 0.0 };
                accumulation_band(dimension + 1, magnitude + identity)
            },
        ));
        let orthonormality_defect = rounded_up(residual + formation);
        if !(orthonormality_defect < 1.0) {
            return Err(ResponseMetricError::NotOrthonormal {
                defect: orthonormality_defect,
            });
        }
        Ok(Self {
            anchor,
            directions,
            absolute_directions,
            orthonormality_defect,
        })
    }

    /// The anchor `h₀`.
    pub fn anchor(&self) -> ArrayView1<'_, f64> {
        self.anchor.view()
    }

    /// The declared directions `V`, `k × d`.
    pub fn directions(&self) -> ArrayView2<'_, f64> {
        self.directions.view()
    }

    /// `k`, the code dimension.
    pub fn rank(&self) -> usize {
        self.directions.nrows()
    }

    /// A bound on `‖V − V*‖₂`.
    pub fn orthonormality_defect(&self) -> f64 {
        self.orthonormality_defect
    }

    /// `E(h) = V*(h − h₀)`.
    pub fn encoder(&self) -> ChartEncoder<'_> {
        ChartEncoder { chart: self }
    }

    /// `D(z) = h₀ + V*ᵀ z`.
    pub fn decoder(&self) -> ChartDecoder<'_> {
        ChartDecoder { chart: self }
    }
}

/// The encoder `E(h) = V*(h − h₀)` of a [`LinearStateChart`].
pub struct ChartEncoder<'a> {
    chart: &'a LinearStateChart,
}

impl NativeMap for ChartEncoder<'_> {
    fn input_dimension(&self) -> usize {
        self.chart.directions.ncols()
    }

    fn output_dimension(&self) -> usize {
        self.chart.directions.nrows()
    }

    /// `ŷ = fl(V fl(h − h₀))` is within
    /// `maxᵢ γ_{d+1} Σⱼ|Vᵢⱼ x̂ⱼ| + ‖V − V*‖₂ (1 + γ₁)‖x̂‖₂ + √d δ` of `V*(x − h₀)`
    /// in sup norm, for every exact `x` within `δ` of `h`: the product and the
    /// subtraction round, `V` differs from `V*`, and `V*` has unit rows.
    fn evaluate(
        &self,
        input: ArrayView1<'_, f64>,
        input_roundoff: f64,
    ) -> Result<Evaluation, StateError> {
        let chart = self.chart;
        let dimension = chart.directions.ncols();
        require_input(input, input_roundoff, dimension, "linear chart: encoder input")?;
        let offset = &input - &chart.anchor;
        let value = chart.directions.dot(&offset);
        let magnitudes = chart.absolute_directions.dot(&offset.mapv(f64::abs));
        let rounding = magnitudes.iter().fold(0.0_f64, |largest, &magnitude| {
            largest.max(accumulation_band(dimension + 1, magnitude))
        });
        let polar =
            chart.orthonormality_defect * growth_factor(1) * frobenius_upper(offset.iter().copied());
        let input_bound = (dimension as f64).sqrt().next_up() * input_roundoff;
        Ok(Evaluation {
            value,
            roundoff: rounded_up((rounding + polar + input_bound) * growth_factor(3)),
        })
    }
}

/// The section `D(z) = h₀ + V*ᵀ z` of a [`LinearStateChart`].
pub struct ChartDecoder<'a> {
    chart: &'a LinearStateChart,
}

impl NativeMap for ChartDecoder<'_> {
    fn input_dimension(&self) -> usize {
        self.chart.directions.nrows()
    }

    fn output_dimension(&self) -> usize {
        self.chart.directions.ncols()
    }

    /// `ĥ = fl(h₀ + fl(Vᵀ ẑ))` is within
    /// `maxⱼ(γ_k Σᵢ|Vᵢⱼ ẑᵢ| + γ₁|ĥⱼ|) + ‖V − V*‖₂ ‖ẑ‖₂ + √k δ` of `h₀ + V*ᵀ z` in sup
    /// norm, for every exact `z` within `δ` of `ẑ`.
    fn evaluate(
        &self,
        input: ArrayView1<'_, f64>,
        input_roundoff: f64,
    ) -> Result<Evaluation, StateError> {
        let chart = self.chart;
        let rank = chart.directions.nrows();
        require_input(input, input_roundoff, rank, "linear chart: decoder input")?;
        let value = &chart.anchor + &chart.directions.t().dot(&input);
        let magnitudes = chart
            .absolute_directions
            .t()
            .dot(&input.mapv(f64::abs));
        let single = accumulation_growth(1);
        let rounding = value
            .iter()
            .zip(magnitudes.iter())
            .fold(0.0_f64, |largest, (entry, magnitude)| {
                largest.max(accumulation_band(rank, *magnitude) + single * entry.abs())
            });
        let polar = chart.orthonormality_defect * frobenius_upper(input.iter().copied());
        let input_bound = (rank as f64).sqrt().next_up() * input_roundoff;
        Ok(Evaluation {
            value,
            roundoff: rounded_up((rounding + polar + input_bound) * growth_factor(3)),
        })
    }
}

/// Local state coordinates at an anchor, read off the response metric.
#[derive(Clone, Debug)]
pub struct LocalStateCoordinates {
    /// The factor `A(h₀)` resolved at the anchor, its only stated state (row 0).
    pub rank: ConstantRankCheck,
    /// The linear chart on the observed directions.
    pub chart: LinearStateChart,
    /// Sine of the largest principal angle between the observed directions and
    /// the exact factor's leading right singular subspace of the same dimension.
    /// Infinite when the band does not separate the resolved singular values from
    /// the rest.
    pub subspace_error_bar: f64,
}

impl LocalStateCoordinates {
    /// The chart's dimension as the evidence status of the exact rank of `G(h₀)`,
    /// through [`ConstantRankCheck::rank_evidence`] at the anchor. At its ceiling
    /// it is exact (Weyl plus the ceiling) and constant on a neighbourhood;
    /// otherwise only the resolved lower bound and the ceiling are derived.
    pub fn dimension(
        &self,
    ) -> Option<Result<EvidenceStatus<StateRow, StateDomain>, EvidenceStatusError>> {
        self.rank.rank_evidence(0)
    }

    /// The direction bound as the evidence status of the sine of the largest
    /// principal angle at the anchor: a uniform bound (rounded up, since the
    /// quotient rounds) where the band separates the resolved singular values,
    /// and unresolved with no upper side where it does not.
    pub fn direction_evidence(
        &self,
    ) -> Result<EvidenceStatus<StateRow, StateDomain>, EvidenceStatusError> {
        let domain = StateDomain::States { count: 1 };
        if self.subspace_error_bar.is_finite() {
            EvidenceStatus::uniform_bound(rounded_up(self.subspace_error_bar), 0.0, domain)
        } else {
            EvidenceStatus::unresolved(0.0, f64::INFINITY, Extremum::Supremum, None, domain)
        }
    }

    /// Test the chart at the declared finite states, the rows of `states`, on the
    /// whitened members of `family` at the declared `fidelity` (sup norm of the
    /// whitened responses).
    ///
    /// `family` may be the family that built the chart or a larger declared one,
    /// e.g. held-out interventions, on the same state space. Each state is
    /// compared with its section representative, which differs from it only along
    /// the chart's kernel, so this is the finite check the kernel alone cannot
    /// replace.
    pub fn finite_check(
        &self,
        family: &ResponseMetric<'_>,
        states: ArrayView2<'_, f64>,
        fidelity: f64,
    ) -> Result<FiberVerdict, ResponseMetricError> {
        let dimension = self.chart.directions.ncols();
        if family.dimension != dimension {
            return Err(StateError::DimensionMismatch {
                context: "finite check: family on the chart's state space",
                expected: dimension,
                found: family.dimension,
            }
            .into());
        }
        let encoder = self.chart.encoder();
        let decoder = self.chart.decoder();
        let chart = StateChart {
            encoder: &encoder,
            decoder: &decoder,
        };
        let futures: Vec<&dyn NativeMap> = family
            .members
            .iter()
            .map(|member| member as &dyn NativeMap)
            .collect();
        Ok(fiber_test(&chart, &futures, states, fidelity)?)
    }
}

/// One declared intervention whose response is a logit vector, compared in KL.
pub struct CategoricalMember<'a> {
    /// `z_α : ℝᵈ → ℝ^{K_α}`: finite logits, `K_α ≥ 1`, with an analytic Jacobian.
    pub logits: &'a dyn DifferentiableNativeMap,
    /// `ν_α`, finite and positive.
    pub weight: f64,
}

/// A declared family of categorical responses under a finite measure, compared
/// in KL.
pub struct CategoricalFamily<'a> {
    members: Vec<CategoricalMember<'a>>,
    dimension: usize,
}

impl<'a> CategoricalFamily<'a> {
    /// Validate the declared family.
    ///
    /// Refusals: an empty family, members on different state spaces, a weight
    /// that is not finite and positive, and a response with no categories.
    pub fn new(members: Vec<CategoricalMember<'a>>) -> Result<Self, ResponseMetricError> {
        let dimension = members
            .first()
            .ok_or(ResponseMetricError::EmptyFamily)?
            .logits
            .input_dimension();
        for (index, member) in members.iter().enumerate() {
            let found = member.logits.input_dimension();
            if found != dimension {
                return Err(ResponseMetricError::StateDimension {
                    member: index,
                    expected: dimension,
                    found,
                });
            }
            if !(member.weight.is_finite() && member.weight > 0.0) {
                return Err(ResponseMetricError::InvalidWeight {
                    member: index,
                    weight: member.weight,
                });
            }
            if member.logits.output_dimension() == 0 {
                return Err(ResponseMetricError::EmptyCategories { member: index });
            }
        }
        Ok(Self { members, dimension })
    }

    /// Dimension `d` of the state space.
    pub fn input_dimension(&self) -> usize {
        self.dimension
    }

    /// Test `coordinates`' chart at the declared finite states, the rows of
    /// `states`, in KL: each tested state `h` against
    /// `Σ_α ν_α KL(softmax z_α(h) ‖ softmax z_α(D(E h)))` and the declared
    /// `fidelity` (nats).
    ///
    /// The chart's section is exact (`E∘D = id` through its polar factor), so no
    /// section refusal is needed. A state within the decoder's roundoff of its
    /// representative tests nothing, and a sample of such states is refused.
    pub fn divergence_check(
        &self,
        coordinates: &LocalStateCoordinates,
        states: ArrayView2<'_, f64>,
        fidelity: f64,
    ) -> Result<DivergenceVerdict, ResponseMetricError> {
        if !fidelity.is_finite() || fidelity < 0.0 {
            return Err(StateError::InvalidFidelity { value: fidelity }.into());
        }
        let chart = &coordinates.chart;
        let dimension = chart.directions.ncols();
        if self.dimension != dimension {
            return Err(StateError::DimensionMismatch {
                context: "divergence check: family on the chart's state space",
                expected: dimension,
                found: self.dimension,
            }
            .into());
        }
        if states.nrows() == 0 {
            return Err(StateError::EmptyFamily {
                context: "divergence check: states",
            }
            .into());
        }
        let encoder = chart.encoder();
        let decoder = chart.decoder();
        let mut tested_states = 0_usize;
        let mut vacuous_states = 0_usize;
        let mut largest_upper = 0.0_f64;
        let mut largest_roundoff = 0.0_f64;
        // The state with the largest computed `divergence − roundoff`: if it is not
        // certified above the fidelity, no state is.
        let mut strongest: Option<(usize, f64, f64, f64)> = None;
        for (index, state) in states.rows().into_iter().enumerate() {
            let code = encoder.evaluate(state, 0.0)?;
            let representative = decoder.evaluate(code.value.view(), 0.0)?;
            if sup_distance(state, representative.value.view()) <= representative.roundoff {
                vacuous_states += 1;
                continue;
            }
            tested_states += 1;
            let (divergence, roundoff) = self.weighted_divergence(state, &representative)?;
            largest_upper = largest_upper.max(certified_sum(divergence, roundoff));
            largest_roundoff = largest_roundoff.max(roundoff);
            let larger = match strongest {
                Some((_, best, best_roundoff, _)) => divergence - roundoff > best - best_roundoff,
                None => true,
            };
            if larger {
                strongest = Some((index, divergence, roundoff, code.roundoff));
            }
        }
        let (state, divergence, roundoff, merge_bound) = match strongest {
            Some(strongest) => strongest,
            None => {
                return Err(StateError::VacuousFiberTest {
                    states: vacuous_states,
                }
                .into());
            }
        };
        let futures = self.members.len();
        Ok(if excess_over(divergence, roundoff, fidelity) {
            DivergenceVerdict::Separated {
                state: StateRow(state),
                divergence,
                roundoff,
                fidelity,
                merge_bound,
            }
        } else if largest_upper <= fidelity {
            DivergenceVerdict::WithinFidelity {
                tested_states,
                vacuous_states,
                futures,
                divergence_upper_bound: largest_upper,
                roundoff: largest_roundoff,
            }
        } else {
            DivergenceVerdict::Unresolved {
                tested_states,
                vacuous_states,
                futures,
                divergence_lower_bound: rounded_down(divergence, roundoff).max(0.0),
                divergence_upper_bound: largest_upper,
                witness: StateRow(state),
            }
        })
    }

    /// `Σ_α ν_α KL(softmax z_α(h) ‖ softmax z_α(r̂))` and a bound on its distance
    /// from the exact divergence between the exact logits at `h` and at the exact
    /// representative the decoder's roundoff covers.
    ///
    /// Per member the bound is the owner's evaluation error plus the logit drift
    /// `e(2 + osc(δ̂) + 4e)`, weighted, plus `γ₁` for each weighted product and
    /// `γ_m` for the sum of `m` non-negative terms. The last factor covers the
    /// bound's own arithmetic.
    fn weighted_divergence(
        &self,
        state: ArrayView1<'_, f64>,
        representative: &Evaluation,
    ) -> Result<(f64, f64), ResponseMetricError> {
        let mut divergence = 0.0_f64;
        let mut roundoff = 0.0_f64;
        for member in &self.members {
            let categories = member.logits.output_dimension();
            let native = member.logits.evaluate(state, 0.0)?;
            let represented = member
                .logits
                .evaluate(representative.value.view(), representative.roundoff)?;
            require_logits(&native, categories, "divergence check: logits at the state")?;
            require_logits(
                &represented,
                categories,
                "divergence check: logits at the representative",
            )?;
            let native_logits = native.value.to_vec();
            let represented_logits = represented.value.to_vec();
            let (kl, evaluation_error) =
                categorical_kl_from_logits_with_error(&native_logits, &represented_logits)
                    .map_err(ResponseMetricError::Categorical)?;
            let radius = native.roundoff.max(represented.roundoff);
            let (lowest, highest) = native_logits.iter().zip(&represented_logits).fold(
                (f64::INFINITY, f64::NEG_INFINITY),
                |(low, high), (reference, moved)| {
                    let gap = moved - reference;
                    (low.min(gap), high.max(gap))
                },
            );
            // Each gap rounds within `u` of its exact value, and the difference once more.
            let oscillation = rounded_up(
                (highest - lowest) * growth_factor(1)
                    + accumulation_growth(2) * (highest.abs() + lowest.abs()),
            );
            let drift = radius * (2.0 + oscillation + 4.0 * radius);
            divergence += member.weight * kl;
            roundoff += member.weight * (evaluation_error + drift)
                + accumulation_growth(1) * member.weight * kl;
        }
        let members = self.members.len();
        Ok((
            divergence,
            rounded_up(
                (roundoff + accumulation_growth(members) * divergence)
                    * growth_factor(4 * members + 2),
            ),
        ))
    }
}

/// What [`CategoricalFamily::divergence_check`] resolved over the stated states.
#[derive(Clone, Debug, PartialEq)]
pub enum DivergenceVerdict {
    /// At `state` the weighted divergence from its section representative exceeds
    /// the declared fidelity by more than its roundoff: the chart is refuted at
    /// that state. The chart merges the pair within `merge_bound` in code space.
    /// This is the state with the largest certified excess.
    Separated {
        state: StateRow,
        divergence: f64,
        roundoff: f64,
        fidelity: f64,
        merge_bound: f64,
    },
    /// At every tested state the exact weighted divergence is at most
    /// `divergence_upper_bound`, roundoff included, and that bound is within the
    /// declared fidelity. `roundoff` is the largest, kept for audit.
    WithinFidelity {
        tested_states: usize,
        vacuous_states: usize,
        futures: usize,
        divergence_upper_bound: f64,
        roundoff: f64,
    },
    /// No separation is certified above the declared fidelity, and the upper
    /// bound does not reach it. `witness` attains `divergence_lower_bound`.
    Unresolved {
        tested_states: usize,
        vacuous_states: usize,
        futures: usize,
        divergence_lower_bound: f64,
        divergence_upper_bound: f64,
        witness: StateRow,
    },
}

impl DivergenceVerdict {
    /// The verdict as the evidence status of
    /// `sup_h Σ_α ν_α KL(softmax z_α(h) ‖ softmax z_α(D(E h)))` over the tested
    /// states.
    pub fn evidence(&self) -> Result<EvidenceStatus<StateRow, StateDomain>, EvidenceStatusError> {
        match self {
            Self::Separated {
                state,
                divergence,
                roundoff,
                fidelity,
                ..
            } => EvidenceStatus::counterexample(*divergence, *roundoff, *fidelity, *state),
            Self::WithinFidelity {
                tested_states,
                vacuous_states,
                futures,
                divergence_upper_bound,
                roundoff,
            } => EvidenceStatus::uniform_bound(
                *divergence_upper_bound,
                *roundoff,
                StateDomain::FiberPairs {
                    tested_states: *tested_states,
                    vacuous_states: *vacuous_states,
                    futures: *futures,
                },
            ),
            Self::Unresolved {
                tested_states,
                vacuous_states,
                futures,
                divergence_lower_bound,
                divergence_upper_bound,
                witness,
            } => EvidenceStatus::unresolved(
                *divergence_lower_bound,
                *divergence_upper_bound,
                Extremum::Supremum,
                Some(*witness),
                StateDomain::FiberPairs {
                    tested_states: *tested_states,
                    vacuous_states: *vacuous_states,
                    futures: *futures,
                },
            ),
        }
    }
}

/// Errors and refusals of the response metric.
#[derive(Debug)]
pub enum ResponseMetricError {
    /// No intervention was declared.
    EmptyFamily,
    /// A member's response is on a different state space than the first member's.
    StateDimension {
        member: usize,
        expected: usize,
        found: usize,
    },
    /// A declared weight is not finite and positive.
    InvalidWeight { member: usize, weight: f64 },
    /// A metric factor has no rows, or its columns do not match its response.
    MetricFactorShape {
        member: usize,
        rows: usize,
        columns: usize,
        response_dimension: usize,
    },
    /// A metric factor has a non-finite entry.
    NonFiniteMetricFactor { member: usize },
    /// The bound on `‖V Vᵀ − I‖₂` is not below 1, so no polar factor of the
    /// chart's directions is certified within it.
    NotOrthonormal { defect: f64 },
    /// A categorical response has no categories.
    EmptyCategories { member: usize },
    /// A divergence between logit vectors was refused.
    Categorical(CategoricalError),
    /// A state check failed or refused.
    State(StateError),
}

impl From<StateError> for ResponseMetricError {
    fn from(source: StateError) -> Self {
        Self::State(source)
    }
}

impl fmt::Display for ResponseMetricError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyFamily => write!(formatter, "response metric: no intervention was declared"),
            Self::StateDimension {
                member,
                expected,
                found,
            } => write!(
                formatter,
                "response metric: member {member} acts on dimension {found}, the family on {expected}"
            ),
            Self::InvalidWeight { member, weight } => write!(
                formatter,
                "response metric: member {member} has weight {weight}, not a finite positive mass"
            ),
            Self::MetricFactorShape {
                member,
                rows,
                columns,
                response_dimension,
            } => write!(
                formatter,
                "response metric: member {member} has a {rows} x {columns} metric factor for a \
                 response of dimension {response_dimension}"
            ),
            Self::NonFiniteMetricFactor { member } => write!(
                formatter,
                "response metric: member {member} has a non-finite metric factor"
            ),
            Self::NotOrthonormal { defect } => write!(
                formatter,
                "linear chart: the orthonormality defect bound {defect:e} is not below 1"
            ),
            Self::EmptyCategories { member } => write!(
                formatter,
                "categorical family: member {member} has no categories"
            ),
            Self::Categorical(source) => write!(formatter, "{source}"),
            Self::State(source) => write!(formatter, "{source}"),
        }
    }
}

impl std::error::Error for ResponseMetricError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Categorical(source) => Some(source),
            Self::State(source) => Some(source),
            _ => None,
        }
    }
}

/// The next float above a positive bound, so a rounded bound is not below the
/// exact one. Zero stays zero.
fn rounded_up(value: f64) -> f64 {
    if value == 0.0 { value } else { value.next_up() }
}

/// A float not below `1 + γ_n`: the sum `1 + γ_n` rounds to nearest, and one ulp
/// above it covers that rounding.
fn growth_factor(operations: usize) -> f64 {
    (1.0 + accumulation_growth(operations)).next_up()
}

/// A bound on `√(Σ xᵢ²)` over the given entries. The naive sum of `n` squares is
/// within `γ_n` of the exact sum and the square root within `u`, so
/// `(1 + γ_{n+1})` over the computed root covers both.
fn frobenius_upper(entries: impl Iterator<Item = f64>) -> f64 {
    let mut count = 0_usize;
    let mut sum = 0.0_f64;
    for entry in entries {
        sum += entry * entry;
        count += 1;
    }
    rounded_up(sum.sqrt() * growth_factor(count + 1))
}

fn require_input(
    input: ArrayView1<'_, f64>,
    input_roundoff: f64,
    dimension: usize,
    context: &'static str,
) -> Result<(), StateError> {
    if input.len() != dimension {
        return Err(StateError::DimensionMismatch {
            context,
            expected: dimension,
            found: input.len(),
        });
    }
    if !input_roundoff.is_finite()
        || input_roundoff < 0.0
        || input.iter().any(|value| !value.is_finite())
    {
        return Err(StateError::NonFinite { context });
    }
    Ok(())
}

fn require_logits(
    evaluation: &Evaluation,
    categories: usize,
    context: &'static str,
) -> Result<(), StateError> {
    if evaluation.value.len() != categories {
        return Err(StateError::DimensionMismatch {
            context,
            expected: categories,
            found: evaluation.value.len(),
        });
    }
    if !evaluation.roundoff.is_finite()
        || evaluation.roundoff < 0.0
        || evaluation.value.iter().any(|value| !value.is_finite())
    {
        return Err(StateError::NonFinite { context });
    }
    Ok(())
}

fn sup_distance(left: ArrayView1<'_, f64>, right: ArrayView1<'_, f64>) -> f64 {
    left.iter()
        .zip(right.iter())
        .fold(0.0_f64, |largest, (a, b)| largest.max((a - b).abs()))
}

/// `value + err`, rounded up when `err` is nonzero, so it bounds the exact sum.
fn certified_sum(value: f64, err: f64) -> f64 {
    let sum = value + err;
    if err == 0.0 { sum } else { sum.next_up() }
}

/// `value − err`, rounded down when `err` is nonzero, so the exact difference
/// bounds it from above.
fn rounded_down(value: f64, err: f64) -> f64 {
    let difference = value - err;
    if err == 0.0 {
        difference
    } else {
        difference.next_down()
    }
}

/// Whether `value` exceeds `threshold` by more than `err`: the predicate
/// [`EvidenceStatus::counterexample`] reads, so a separation always converts.
fn excess_over(value: f64, err: f64, threshold: f64) -> bool {
    rounded_down(value, err) > threshold
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    /// A fixture response given by closed forms. `evaluate` returns the value and
    /// its roundoff bound for an input roundoff; the Jacobian is exact.
    struct Closed {
        input: usize,
        output: usize,
        evaluate: for<'v> fn(ArrayView1<'v, f64>, f64) -> (Array1<f64>, f64),
        jacobian: for<'v> fn(ArrayView1<'v, f64>) -> Array2<f64>,
    }

    impl NativeMap for Closed {
        fn input_dimension(&self) -> usize {
            self.input
        }

        fn output_dimension(&self) -> usize {
            self.output
        }

        fn evaluate(
            &self,
            input: ArrayView1<'_, f64>,
            input_roundoff: f64,
        ) -> Result<Evaluation, StateError> {
            let (value, roundoff) = (self.evaluate)(input, input_roundoff);
            Ok(Evaluation { value, roundoff })
        }
    }

    impl DifferentiableNativeMap for Closed {
        fn jacobian(&self, input: ArrayView1<'_, f64>) -> Result<JacobianEvaluation, StateError> {
            Ok(JacobianEvaluation {
                matrix: (self.jacobian)(input),
                roundoff: 0.0,
            })
        }
    }

    fn declared<'a>(
        response: &'a dyn DifferentiableNativeMap,
        weight: f64,
        metric_factor: Array2<f64>,
    ) -> ResponseMember<'a> {
        ResponseMember {
            response,
            weight,
            metric_factor,
        }
    }

    /// `R(t) = t²`. One product rounds once; moving the input by `δ` moves the
    /// value by at most `(2|t| + δ)δ`. `2t` is exact.
    fn square() -> Closed {
        Closed {
            input: 1,
            output: 1,
            evaluate: |h, delta| {
                let t = h[0];
                (
                    array![t * t],
                    accumulation_growth(1) * t * t + (2.0 * t.abs() + delta) * delta,
                )
            },
            jacobian: |h| array![[2.0 * h[0]]],
        }
    }

    /// `R(a, b) = a²`, constant along `b`.
    fn first_square() -> Closed {
        Closed {
            input: 2,
            output: 1,
            evaluate: |h, delta| {
                let a = h[0];
                (
                    array![a * a],
                    accumulation_growth(1) * a * a + (2.0 * a.abs() + delta) * delta,
                )
            },
            jacobian: |h| array![[2.0 * h[0], 0.0]],
        }
    }

    /// `R(a, b) = a² + b²`, whose fibres are circles. Two products and a sum round
    /// within `γ₂(a² + b²)`; moving the input by `δ` moves the value by at most
    /// `(2(|a| + |b|) + 2δ)δ`.
    fn radius_square() -> Closed {
        Closed {
            input: 2,
            output: 1,
            evaluate: |h, delta| {
                let (a, b) = (h[0], h[1]);
                (
                    array![a * a + b * b],
                    accumulation_growth(2) * (a * a + b * b)
                        + (2.0 * (a.abs() + b.abs()) + 2.0 * delta) * delta,
                )
            },
            jacobian: |h| array![[2.0 * h[0], 2.0 * h[1]]],
        }
    }

    /// Two logits `R(a, b) = (a + b, b)`. The sum rounds within `γ₁(|a| + |b|)`,
    /// and moving the input by `δ` moves a logit by at most `2δ`.
    fn shifted_logits() -> Closed {
        Closed {
            input: 2,
            output: 2,
            evaluate: |h, delta| {
                let (a, b) = (h[0], h[1]);
                (
                    array![a + b, b],
                    accumulation_growth(1) * (a.abs() + b.abs()) + 2.0 * delta,
                )
            },
            jacobian: |h| {
                Array2::from_shape_fn((2, h.len()), |(row, column)| {
                    if row == 0 || column == 1 { 1.0 } else { 0.0 }
                })
            },
        }
    }

    /// `R(a, b) = (ab, a² − b)`: quadratic, so a central difference is exact
    /// algebraically. The rows round within `γ₁|ab|` and `γ₂(a² + |b|)`; moving the
    /// input by `δ` moves a row by at most `(2(|a| + |b|) + 2δ + 1)δ`.
    fn quadratic_pair() -> Closed {
        Closed {
            input: 2,
            output: 2,
            evaluate: |h, delta| {
                let (a, b) = (h[0], h[1]);
                (
                    array![a * b, a * a - b],
                    accumulation_growth(2) * (a * a + b.abs() + (a * b).abs())
                        + (2.0 * (a.abs() + b.abs()) + 2.0 * delta + 1.0) * delta,
                )
            },
            jacobian: |h| array![[h[1], h[0]], [2.0 * h[0], -1.0]],
        }
    }

    /// The singular-point caveat. `R(t) = t²` has `G(0) = 0`: the rank at 0 is 0
    /// against a ceiling of 1, so the dimension is unresolved and the chart has no
    /// directions. The whole line is kernel, yet `R(0.5) = 0.25` against
    /// `R(0) = 0`, and the finite check refuses the chart. Control: `R(a, b) = a²`
    /// at `(1, 0)` has certified rank 1 and kernel `e_b`, along which it is
    /// constant, so no separation is certified at fidelity 0, and the check passes
    /// at the upper bound that verdict reported.
    #[test]
    fn the_kernel_at_a_singular_point_is_not_a_fibre_2951() {
        let square = square();
        let metric = ResponseMetric::new(vec![declared(&square, 1.0, array![[1.0]])])
            .expect("metric");
        let singular = metric
            .local_coordinates(array![0.0].view())
            .expect("coordinates");
        assert_eq!(singular.rank.states[0].singular_values, vec![0.0]);
        assert_eq!(
            (singular.rank.states[0].resolved_rank, singular.rank.states[0].rank_ceiling),
            (0, 1)
        );
        assert_eq!(
            metric_eigenvalues(&singular.rank.states[0]),
            vec![MetricEigenvalue {
                lower: 0.0,
                upper: 0.0
            }]
        );
        assert!(matches!(
            singular.dimension(),
            Some(Ok(EvidenceStatus::Unresolved { lower, upper, .. })) if lower == 0.0 && upper == 1.0
        ));
        assert_eq!(singular.chart.rank(), 0);
        assert_eq!(singular.subspace_error_bar, 0.0);

        let states = array![[0.5], [-0.25]];
        let verdict = singular
            .finite_check(&metric, states.view(), 0.0)
            .expect("finite check");
        assert!(
            matches!(verdict, FiberVerdict::Separated { .. }),
            "t² is not constant along its kernel at 0, got {verdict:?}"
        );
        let member = &metric.members()[0];
        let moved = member.evaluate(array![0.5].view(), 0.0).expect("evaluate");
        let anchored = member.evaluate(array![0.0].view(), 0.0).expect("evaluate");
        assert_eq!((moved.value[0], anchored.value[0]), (0.25, 0.0));
        assert!(0.25 - moved.roundoff - anchored.roundoff > 0.0);

        let flat = first_square();
        let control = ResponseMetric::new(vec![declared(&flat, 1.0, array![[1.0]])])
            .expect("metric");
        let regular = control
            .local_coordinates(array![1.0, 0.0].view())
            .expect("coordinates");
        assert!(matches!(
            regular.dimension(),
            Some(Ok(EvidenceStatus::Exact { value, .. })) if value == 1.0
        ));
        assert_eq!(regular.chart.rank(), 1);
        let kernel_states = array![[1.0, 0.5], [0.5, -2.0]];
        let upper = match regular
            .finite_check(&control, kernel_states.view(), 0.0)
            .expect("finite check")
        {
            FiberVerdict::Unresolved {
                separation_upper_bound,
                ..
            } => separation_upper_bound,
            other => panic!("a² is constant along e_b, so nothing separates at 0, got {other:?}"),
        };
        assert!(upper > 0.0);
        let passed = regular
            .finite_check(&control, kernel_states.view(), upper)
            .expect("finite check");
        assert!(
            matches!(passed, FiberVerdict::WithinFidelity { .. }),
            "the check passes at its own upper bound, got {passed:?}"
        );
    }

    /// Certified constant rank does not make the kernel a fibre at a finite scale.
    /// `R(a, b) = a² + b²` has rank 1 at `(1, 0)`, its ceiling, so the rank is 1
    /// on a neighbourhood and a one-dimensional quotient exists (the radius). The
    /// kernel `e_b` is tangent to the circle, not along it: `R(1, 0.5) = 1.25`
    /// against `R(1, 0) = 1`, and the finite check refuses the linear chart.
    /// Control: states moved only along the observed direction lie on the section
    /// and test nothing, and such a sample is refused.
    #[test]
    fn certified_constant_rank_does_not_make_the_kernel_a_fibre_2951() {
        let radius = radius_square();
        let metric = ResponseMetric::new(vec![declared(&radius, 1.0, array![[1.0]])])
            .expect("metric");
        let coordinates = metric
            .local_coordinates(array![1.0, 0.0].view())
            .expect("coordinates");
        assert!(matches!(
            coordinates.dimension(),
            Some(Ok(EvidenceStatus::Exact { value, .. })) if value == 1.0
        ));
        assert!(
            coordinates.subspace_error_bar > 0.0 && coordinates.subspace_error_bar < 1.0,
            "a resolved rank at its ceiling has a finite direction bound, found {}",
            coordinates.subspace_error_bar
        );
        let directions = coordinates
            .direction_evidence()
            .expect("a finite bar is a uniform bound");
        assert!(matches!(directions, EvidenceStatus::UniformBound { .. }));
        assert!(directions.certifies_at_most(coordinates.subspace_error_bar.next_up()));
        // Control: an unseparated band has no derived upper side.
        let mut unseparated = coordinates.clone();
        unseparated.subspace_error_bar = f64::INFINITY;
        let open = unseparated
            .direction_evidence()
            .expect("an infinite bar is unresolved");
        assert!(matches!(open, EvidenceStatus::Unresolved { upper, .. } if upper == f64::INFINITY));
        assert!(!open.certifies_at_most(1.0));
        let verdict = coordinates
            .finite_check(&metric, array![[1.0, 0.5]].view(), 0.0)
            .expect("finite check");
        assert!(
            matches!(verdict, FiberVerdict::Separated { .. }),
            "the circle leaves its tangent, got {verdict:?}"
        );
        match coordinates.finite_check(&metric, array![[1.5, 0.0], [0.25, 0.0]].view(), 0.0) {
            Err(ResponseMetricError::State(StateError::VacuousFiberTest { states })) => {
                assert_eq!(states, 2)
            }
            other => panic!("states on the section test no fibre, got {other:?}"),
        }
    }

    /// The output metric's factor removes the directions it does not charge. Two
    /// logits `R(a, b) = (a + b, b)` under `L = [½, −½]` see only `a`: a shift of
    /// both logits by `b` is invisible, `A = [½, 0]`, and the rank is 1, its
    /// ceiling. No separation along `b` is certified at fidelity 0. Controls: the
    /// Euclidean factor `I₂` charges the shift, so the rank is 2 and `G = JᵀJ` has
    /// trace 3 and determinant 1 inside the eigenvalue bounds. The weight scales
    /// the metric: `ν = ¼` halves the factor, and the eigenvalue moves from `¼` to
    /// `1/16`.
    #[test]
    fn the_metric_factor_removes_directions_the_output_metric_does_not_charge_2951() {
        let logits = shifted_logits();
        let origin = array![0.0, 0.0];
        let shift_blind =
            ResponseMetric::new(vec![declared(&logits, 1.0, array![[0.5, -0.5]])])
                .expect("metric");
        let coordinates = shift_blind
            .local_coordinates(origin.view())
            .expect("coordinates");
        assert!(
            (coordinates.rank.states[0].singular_values[0] - 0.5).abs() <= coordinates.rank.states[0].band
        );
        assert!(matches!(
            coordinates.dimension(),
            Some(Ok(EvidenceStatus::Exact { value, .. })) if value == 1.0
        ));
        let quarter = metric_eigenvalues(&coordinates.rank.states[0])[0];
        assert!(quarter.lower <= 0.25 && 0.25 <= quarter.upper, "{quarter:?}");
        let bracket = quarter.evidence().expect("a Weyl bracket is unresolved");
        assert!(matches!(
            bracket,
            EvidenceStatus::Unresolved { lower, upper, .. } if lower <= 0.25 && 0.25 <= upper
        ));
        assert!(bracket.certifies_at_most(quarter.upper) && bracket.refutes_at_most(0.0));
        match coordinates
            .finite_check(&shift_blind, array![[0.5, 3.0], [-1.0, -2.0]].view(), 0.0)
            .expect("finite check")
        {
            FiberVerdict::Unresolved { .. } => {}
            other => panic!("a shift of both logits is not charged, got {other:?}"),
        }

        let euclidean = ResponseMetric::new(vec![declared(
            &logits,
            1.0,
            array![[1.0, 0.0], [0.0, 1.0]],
        )])
        .expect("metric");
        let charged = euclidean
            .local_coordinates(origin.view())
            .expect("coordinates");
        assert_eq!(
            (charged.rank.states[0].resolved_rank, charged.rank.states[0].rank_ceiling),
            (2, 2)
        );
        assert!(matches!(
            charged.dimension(),
            Some(Ok(EvidenceStatus::Exact { value, .. })) if value == 2.0
        ));
        let spectrum = metric_eigenvalues(&charged.rank.states[0]);
        assert!(spectrum[0].lower + spectrum[1].lower <= 3.0);
        assert!(3.0 <= spectrum[0].upper + spectrum[1].upper);
        assert!(spectrum[0].lower * spectrum[1].lower <= 1.0);
        assert!(1.0 <= spectrum[0].upper * spectrum[1].upper);

        let weighted =
            ResponseMetric::new(vec![declared(&logits, 0.25, array![[0.5, -0.5]])])
                .expect("metric");
        let halved = weighted
            .local_coordinates(origin.view())
            .expect("coordinates");
        let sixteenth = metric_eigenvalues(&halved.rank.states[0])[0];
        assert!(
            sixteenth.lower <= 0.0625 && 0.0625 <= sixteenth.upper,
            "{sixteenth:?}"
        );
        assert!(sixteenth.upper < quarter.lower);
    }

    /// The whitened Jacobian is the analytic `√ν L DR`. For a quadratic response a
    /// central difference is exact algebraically, so the two agree within the
    /// Jacobian's formation bound plus the evaluations' roundoff over the step and
    /// the subtraction's rounding. Control: the unweighted member's Jacobian
    /// misses the weighted member's central difference by more than its bound.
    #[test]
    fn whitened_jacobian_matches_exact_central_differences_2951() {
        let pair = quadratic_pair();
        let factor = array![[1.0, 2.0], [0.5, -1.0], [3.0, 0.25]];
        let metric = ResponseMetric::new(vec![
            declared(&pair, 0.5625, factor.clone()),
            declared(&pair, 1.0, factor),
        ])
        .expect("metric");
        let state = array![0.75, -1.5];
        let step = 0.5;
        let weighted = &metric.members()[0];
        let unweighted = &metric.members()[1];
        let analytic = weighted.jacobian(state.view()).expect("jacobian");
        let plain = unweighted.jacobian(state.view()).expect("jacobian");
        for column in 0..2 {
            let mut forward = state.clone();
            forward[column] += step;
            let mut backward = state.clone();
            backward[column] -= step;
            let ahead = weighted.evaluate(forward.view(), 0.0).expect("evaluate");
            let behind = weighted.evaluate(backward.view(), 0.0).expect("evaluate");
            let difference = (&ahead.value - &behind.value) / (2.0 * step);
            let mut plain_misses = false;
            for row in 0..3 {
                let rounding = accumulation_growth(2)
                    * (ahead.value[row].abs() + behind.value[row].abs())
                    / (2.0 * step);
                let evaluations = (ahead.roundoff + behind.roundoff) / (2.0 * step);
                let gap = (difference[row] - analytic.matrix[[row, column]]).abs();
                assert!(
                    gap <= analytic.roundoff + evaluations + rounding,
                    "entry ({row}, {column}): central difference {} against analytic {}",
                    difference[row],
                    analytic.matrix[[row, column]]
                );
                let plain_gap = (difference[row] - plain.matrix[[row, column]]).abs();
                plain_misses |= plain_gap > plain.roundoff + evaluations + rounding;
            }
            assert!(plain_misses, "the unweighted Jacobian must miss column {column}");
        }
    }

    /// The declared family is validated: an empty family, a weight that is not a
    /// finite positive mass, a factor whose shape does not fit its response, a
    /// non-finite factor, and members on different state spaces are refused.
    /// Control: two valid members on one state space are accepted.
    #[test]
    fn declared_family_refusals_2951() {
        let square = square();
        let flat = first_square();
        assert!(matches!(
            ResponseMetric::new(Vec::new()).err(),
            Some(ResponseMetricError::EmptyFamily)
        ));
        for weight in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            match ResponseMetric::new(vec![declared(&square, weight, array![[1.0]])]).err() {
                Some(ResponseMetricError::InvalidWeight {
                    member,
                    weight: found,
                }) => {
                    assert_eq!(member, 0);
                    assert!(found.to_bits() == weight.to_bits());
                }
                other => panic!("weight {weight} must be refused, got {other:?}"),
            }
        }
        match ResponseMetric::new(vec![declared(&square, 1.0, array![[1.0, 2.0]])]).err() {
            Some(ResponseMetricError::MetricFactorShape {
                member,
                rows,
                columns,
                response_dimension,
            }) => assert_eq!((member, rows, columns, response_dimension), (0, 1, 2, 1)),
            other => panic!("a factor with the wrong columns must be refused, got {other:?}"),
        }
        match ResponseMetric::new(vec![declared(&square, 1.0, Array2::zeros((0, 1)))]).err() {
            Some(ResponseMetricError::MetricFactorShape { rows, .. }) => assert_eq!(rows, 0),
            other => panic!("a factor with no rows must be refused, got {other:?}"),
        }
        match ResponseMetric::new(vec![declared(&square, 1.0, array![[f64::NAN]])]).err() {
            Some(ResponseMetricError::NonFiniteMetricFactor { member }) => assert_eq!(member, 0),
            other => panic!("a non-finite factor must be refused, got {other:?}"),
        }
        match ResponseMetric::new(vec![
            declared(&square, 1.0, array![[1.0]]),
            declared(&flat, 1.0, array![[1.0]]),
        ])
        .err()
        {
            Some(ResponseMetricError::StateDimension {
                member,
                expected,
                found,
            }) => assert_eq!((member, expected, found), (1, 1, 2)),
            other => panic!("members on different state spaces must be refused, got {other:?}"),
        }
        let accepted = ResponseMetric::new(vec![
            declared(&square, 1.0, array![[1.0]]),
            declared(&square, 2.0, array![[1.0], [-3.0]]),
        ])
        .expect("a valid family");
        assert_eq!(accepted.members().len(), 2);
        assert_eq!(accepted.input_dimension(), 1);
    }

    /// A declared chart whose directions are not orthonormal is carried through its
    /// polar factor. Directions `[½, ½]` have `V Vᵀ = ½`, so the computed round trip
    /// `E(D(3))` is `1.5`, short of `3` by `1.5`. That defect is not roundoff; the
    /// bounds carry it through `‖V − V*‖₂ ≤ ‖V Vᵀ − I‖₂ = ½`, so the round trip stays
    /// within the encoder's bound at the decoder's roundoff, the predicate the fibre
    /// test reads. Controls: the orthonormal chart `[1, 0]` round-trips exactly with
    /// a bound below that defect, and directions `[1, 1]` have `‖V Vᵀ − I‖₂ = 1` and
    /// are refused.
    #[test]
    fn a_non_orthonormal_chart_is_carried_through_its_polar_factor_2951() {
        let anchor = array![1.0, -2.0];
        let code = array![3.0];
        let chart = LinearStateChart::new(anchor.clone(), array![[0.5, 0.5]]).expect("chart");
        assert!(chart.orthonormality_defect() >= 0.5 && chart.orthonormality_defect() < 1.0);
        let decoded = chart.decoder().evaluate(code.view(), 0.0).expect("decode");
        assert_eq!(decoded.value, array![2.5, -0.5]);
        let round_trip = chart
            .encoder()
            .evaluate(decoded.value.view(), decoded.roundoff)
            .expect("encode");
        let defect = (round_trip.value[0] - code[0]).abs();
        assert_eq!(defect, 1.5);
        assert!(
            defect <= round_trip.roundoff,
            "the polar term must carry the defect {defect}, bound {}",
            round_trip.roundoff
        );

        let orthonormal = LinearStateChart::new(anchor, array![[1.0, 0.0]]).expect("chart");
        assert!(orthonormal.orthonormality_defect() < chart.orthonormality_defect());
        let exact_decoded = orthonormal
            .decoder()
            .evaluate(code.view(), 0.0)
            .expect("decode");
        let exact_round_trip = orthonormal
            .encoder()
            .evaluate(exact_decoded.value.view(), exact_decoded.roundoff)
            .expect("encode");
        assert_eq!(exact_round_trip.value[0], 3.0);
        assert!(exact_round_trip.roundoff < defect);

        match LinearStateChart::new(array![0.0, 0.0], array![[1.0, 1.0]]).err() {
            Some(ResponseMetricError::NotOrthonormal { defect }) => assert!(defect >= 1.0),
            other => panic!("directions of norm √2 must be refused, got {other:?}"),
        }
    }

    fn categorical(logits: &dyn DifferentiableNativeMap, weight: f64) -> CategoricalMember<'_> {
        CategoricalMember { logits, weight }
    }

    /// Logits `z(t) = (t², 0)`. One product rounds once; moving `t` by `δ` moves the
    /// first logit by at most `(2|t| + δ)δ`.
    fn squared_logit() -> Closed {
        Closed {
            input: 1,
            output: 2,
            evaluate: |h, delta| {
                let t = h[0];
                (
                    array![t * t, 0.0],
                    accumulation_growth(1) * t * t + (2.0 * t.abs() + delta) * delta,
                )
            },
            jacobian: |h| array![[2.0 * h[0]], [0.0]],
        }
    }

    /// A response with no categories.
    fn no_categories() -> Closed {
        Closed {
            input: 1,
            output: 0,
            evaluate: |h, delta| (Array1::from_elem(0, h[0]), delta),
            jacobian: |h| Array2::from_elem((0, h.len()), 0.0),
        }
    }

    /// A common logit shift costs no divergence. Two logits `(a + b, b)` under the
    /// shift-blind factor `[½, −½]` give a chart on `e_a`, and states moved along
    /// `b` shift both logits equally: no divergence is certified at fidelity 0,
    /// and the check passes at the upper bound it reported. Control: the chart of
    /// a zero factor has no directions, so the state `(1.5, −1)` is merged with the
    /// anchor, whose first logit differs by 1, and the check refutes that chart.
    #[test]
    fn a_common_logit_shift_costs_no_divergence_2951() {
        let logits = shifted_logits();
        let anchor = array![0.5, -1.0];
        let shift_blind = ResponseMetric::new(vec![declared(&logits, 1.0, array![[0.5, -0.5]])])
            .expect("metric");
        let coordinates = shift_blind
            .local_coordinates(anchor.view())
            .expect("coordinates");
        assert_eq!(coordinates.chart.rank(), 1);
        let family = CategoricalFamily::new(vec![categorical(&logits, 1.0)]).expect("family");
        let states = array![[0.5, 2.0], [0.5, -3.0]];
        let upper = match family
            .divergence_check(&coordinates, states.view(), 0.0)
            .expect("divergence check")
        {
            DivergenceVerdict::Unresolved {
                tested_states,
                divergence_upper_bound,
                ..
            } => {
                assert_eq!(tested_states, 2);
                divergence_upper_bound
            }
            other => panic!("a common logit shift costs no divergence, got {other:?}"),
        };
        assert!(upper > 0.0 && upper.is_finite(), "found {upper}");
        let passed = family
            .divergence_check(&coordinates, states.view(), upper)
            .expect("divergence check");
        assert!(
            matches!(passed, DivergenceVerdict::WithinFidelity { .. }),
            "the check passes at its own upper bound, got {passed:?}"
        );
        assert!(passed.evidence().expect("a bound").certifies_at_most(upper));

        let blind = ResponseMetric::new(vec![declared(&logits, 1.0, array![[0.0, 0.0]])])
            .expect("metric");
        let anchored = blind.local_coordinates(anchor.view()).expect("coordinates");
        assert_eq!(anchored.chart.rank(), 0);
        let refuted = family
            .divergence_check(&anchored, array![[1.5, -1.0]].view(), 0.0)
            .expect("divergence check");
        assert!(
            matches!(refuted, DivergenceVerdict::Separated { .. }),
            "merging a moved first logit must be refuted, got {refuted:?}"
        );
        assert!(refuted.evidence().expect("a counterexample").refutes_at_most(0.0));
    }

    /// The singular-point caveat in KL. `z(t) = (t², 0)` has `J(0) = 0`, so the
    /// chart at 0 has no directions and the whole line is kernel. Yet
    /// `KL(softmax(¼, 0) ‖ softmax(0, 0)) ≈ 7.75e-3`, and the check refutes the
    /// chart. Two analytic bounds bracket the certified divergence with no
    /// tolerance of the test's own: P15's `osc(δ)²/8 = 1/128` above, and Pinsker's
    /// `½‖p − q‖₁² = 2(σ(¼) − ½)²` below.
    #[test]
    fn the_kernel_at_a_singular_point_is_refuted_in_kl_2951() {
        let logits = squared_logit();
        let euclidean = ResponseMetric::new(vec![declared(
            &logits,
            1.0,
            array![[1.0, 0.0], [0.0, 1.0]],
        )])
        .expect("metric");
        let coordinates = euclidean
            .local_coordinates(array![0.0].view())
            .expect("coordinates");
        assert_eq!(
            (
                coordinates.rank.states[0].resolved_rank,
                coordinates.rank.states[0].rank_ceiling
            ),
            (0, 1)
        );
        let family = CategoricalFamily::new(vec![categorical(&logits, 1.0)]).expect("family");
        let verdict = family
            .divergence_check(&coordinates, array![[0.5]].view(), 0.0)
            .expect("divergence check");
        match &verdict {
            DivergenceVerdict::Separated {
                state,
                divergence,
                roundoff,
                ..
            } => {
                assert_eq!(*state, StateRow(0));
                // P15: KL(softmax z ‖ softmax z′) ≤ osc(z′ − z)²/8, and osc = ¼ exactly.
                let oscillation_bound = 0.0625 / 8.0;
                assert!(
                    divergence - roundoff <= oscillation_bound,
                    "{divergence} ± {roundoff} above the P15 bound {oscillation_bound}"
                );
                // Pinsker: KL ≥ ½‖p − q‖₁² = 2(σ(¼) − ½)². σ(¼) rounds within γ₃
                // relative, σ(¼)/(σ(¼) − ½) < 10 turns that into at most 30u relative
                // on the difference, and the square with its products stays below 64u.
                let probability = 1.0 / (1.0 + (-0.25_f64).exp());
                let pinsker = 2.0 * (probability - 0.5) * (probability - 0.5);
                let pinsker_floor = pinsker * (1.0 - accumulation_growth(64));
                assert!(
                    divergence + roundoff >= pinsker_floor,
                    "{divergence} ± {roundoff} below the Pinsker bound {pinsker_floor}"
                );
            }
            other => panic!("t² is not constant along its kernel at 0, got {other:?}"),
        }
        assert!(verdict.evidence().expect("a counterexample").refutes_at_most(0.0));
    }

    /// The declared categorical family and the check's inputs are validated: an
    /// empty family, a weight that is not a finite positive mass, members on
    /// different state spaces, a response with no categories, a chart on another
    /// state space, and an invalid declared fidelity are refused. Control: the valid
    /// family and chart of the other tests are accepted.
    #[test]
    fn categorical_family_refusals_2951() {
        let squared = squared_logit();
        let pair = shifted_logits();
        let empty = no_categories();
        assert!(matches!(
            CategoricalFamily::new(Vec::new()).err(),
            Some(ResponseMetricError::EmptyFamily)
        ));
        for weight in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            match CategoricalFamily::new(vec![categorical(&squared, weight)]).err() {
                Some(ResponseMetricError::InvalidWeight {
                    member,
                    weight: found,
                }) => {
                    assert_eq!(member, 0);
                    assert!(found.to_bits() == weight.to_bits());
                }
                other => panic!("weight {weight} must be refused, got {other:?}"),
            }
        }
        match CategoricalFamily::new(vec![categorical(&squared, 1.0), categorical(&pair, 1.0)]).err()
        {
            Some(ResponseMetricError::StateDimension {
                member,
                expected,
                found,
            }) => assert_eq!((member, expected, found), (1, 1, 2)),
            other => panic!("members on different state spaces must be refused, got {other:?}"),
        }
        match CategoricalFamily::new(vec![categorical(&empty, 1.0)]).err() {
            Some(ResponseMetricError::EmptyCategories { member }) => assert_eq!(member, 0),
            other => panic!("a response with no categories must be refused, got {other:?}"),
        }

        let euclidean = ResponseMetric::new(vec![declared(
            &squared,
            1.0,
            array![[1.0, 0.0], [0.0, 1.0]],
        )])
        .expect("metric");
        let coordinates = euclidean
            .local_coordinates(array![0.0].view())
            .expect("coordinates");
        let family = CategoricalFamily::new(vec![categorical(&squared, 1.0)]).expect("family");
        assert_eq!(family.input_dimension(), 1);
        for fidelity in [-1.0, f64::NAN] {
            match family
                .divergence_check(&coordinates, array![[0.5]].view(), fidelity)
                .err()
            {
                Some(ResponseMetricError::State(StateError::InvalidFidelity { value })) => {
                    assert!(value.to_bits() == fidelity.to_bits())
                }
                other => panic!("fidelity {fidelity} must be refused, got {other:?}"),
            }
        }
        let planar = ResponseMetric::new(vec![declared(&pair, 1.0, array![[0.5, -0.5]])])
            .expect("metric");
        let planar_coordinates = planar
            .local_coordinates(array![0.0, 0.0].view())
            .expect("coordinates");
        match family
            .divergence_check(&planar_coordinates, array![[0.5, 0.5]].view(), 0.0)
            .err()
        {
            Some(ResponseMetricError::State(StateError::DimensionMismatch { expected, found, .. })) => {
                assert_eq!((expected, found), (2, 1))
            }
            other => panic!("a chart on another state space must be refused, got {other:?}"),
        }
    }
}
