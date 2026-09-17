//! Sufficient computational state over finite native intervention responses
//! (#2951, P13).
//!
//! # The quotient
//!
//! A computational state `h ∈ ℝᵈ` (a per-input coordinate, never a
//! parameter-family label) is followed by a declared finite family of permitted
//! futures `r_1..r_K`. Each future is a native execution: transitions under
//! declared interventions, then a readout. Two states are equivalent, `h ∼ h'`,
//! iff every permitted future agrees on them. A chart `E` is a sufficient state
//! iff it never merges two inequivalent states.
//!
//! * **Descent.** If the family is closed under prepending a transition `T_a`
//!   (`r ∈ F ⇒ r∘T_a ∈ F`), then `h ∼ h'` gives
//!   `r(T_a h) = (r∘T_a)(h) = (r∘T_a)(h') = r(T_a h')`. So `T_a` descends, and a
//!   minimal sufficient chart satisfies `E_next∘T_a = G_a∘E`.
//! * **Local dimension.** Let `R = (r_1..r_K)` *generate* the family: every
//!   permitted future is a function of `R`. Where `R` has constant rank `k` on a
//!   neighbourhood of `h`, the constant-rank theorem makes the fibres of `R` near
//!   `h` submanifolds of dimension `d − k` and the quotient a `k`-manifold, so `k`
//!   is the local minimal dimension. Both hypotheses are load-bearing.
//!   * A family that does not generate can have a smaller rank. Observing `a` of
//!     `(a, b)` under `T(a, b) = (a + b, b)` has rank 1; adding `a∘T` gives 2.
//!   * At a point where the rank drops (`o(h) = h²` at `h = 0`) there is no
//!     constant-rank chart, and the rank there is not the local dimension.
//! * **Fibres are global.** The rank at `h` tests local injectivity only.
//!   `o(h) = h²` has full rank at `h = ±1` yet merges them, and the future
//!   `h ↦ h + 1` separates them (`0` against `4`). [`fiber_test`] compares each
//!   stated native state with its section representative `D(E(h))`, which lies in
//!   the same fibre of `E`. Checking responses on decoded states `h = D(z)`
//!   compares each state with itself: a check on decoder sections does not test
//!   fibres, and the test refuses a sample on which every comparison is of that
//!   kind.
//!
//! # Two contracts that are not one
//!
//! * The **quotient contract** `E'∘T = g∘E` ([`QuotientContract`]) says the code
//!   descends. It is checked at native states, off any section.
//! * The **realization contract** `T∘D = D'∘g` ([`RealizationContract`]) says
//!   decoded representatives are carried to decoded representatives. It is
//!   checked at codes, on the section.
//!
//! Neither implies the other.
//! * `T(z, n) = (z, n + 1)` with only `z` observed satisfies the quotient
//!   contract with `E = E' = (z, n) ↦ z` and `g = id`. No section `D = D'`
//!   realizes it, because `T(D z)` has fibre coordinate one more than `D(z)`.
//! * `T(z, n) = (z + n, n)` fixes the section `D(z) = (z, 0)`, so the realization
//!   contract holds with `g = id`. The quotient contract fails at every state with
//!   `n ≠ 0`. With `E'∘D' = id` a realization gives `E'∘T∘D = g`, a statement on
//!   the section only.
//!
//! # Decisions
//!
//! Every native map reports a derived roundoff bound with its value
//! ([`Evaluation`]), so every verdict separates what the arithmetic resolved from
//! what it did not.
//! * A violation or separation is reported only when a computed defect exceeds
//!   the combined bounds.
//! * Otherwise the result is a uniform bound over the stated finite states or
//!   codes, roundoff included. It is never an exact equality.
//! * A rank counts the singular values above
//!   [`gam_linalg::roundoff::factor_singular_band`] plus the Jacobians' formation
//!   bound. That is a certified lower bound on the exact rank, and it is exact
//!   only where it reaches `min(rows, d)`.
//! * The fidelity tolerance is a declared experiment input with no default.

use std::fmt;

use gam_linalg::faer_ndarray::{FaerLinalgError, FaerSvd};
use gam_linalg::roundoff::factor_singular_band;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, s};

/// A computed value with a derived bound on its roundoff.
#[derive(Clone, Debug)]
pub struct Evaluation {
    /// The computed value.
    pub value: Array1<f64>,
    /// Bound on `‖value − f(x)‖∞` over every exact input `x` within the
    /// evaluation's input roundoff of the supplied input.
    pub roundoff: f64,
}

/// A native computation between coordinate spaces: a transition, a readout, a
/// chart's encoder or decoder, or a descended map. It owns its error model.
pub trait NativeMap {
    /// Dimension of the input coordinates.
    fn input_dimension(&self) -> usize;
    /// Dimension of the output coordinates.
    fn output_dimension(&self) -> usize;
    /// The computed value at `input`, with a bound on its sup-norm distance to
    /// the exact map's value at every input within `input_roundoff` (sup norm)
    /// of `input`.
    fn evaluate(
        &self,
        input: ArrayView1<'_, f64>,
        input_roundoff: f64,
    ) -> Result<Evaluation, StateError>;
}

/// An analytic Jacobian with a bound on its formation error.
#[derive(Clone, Debug)]
pub struct JacobianEvaluation {
    /// `output_dimension × input_dimension`.
    pub matrix: Array2<f64>,
    /// Bound on the spectral norm of `matrix` minus the exact Jacobian at the
    /// supplied input.
    pub roundoff: f64,
}

/// A native map with an analytic Jacobian.
pub trait DifferentiableNativeMap: NativeMap {
    /// The Jacobian at `input`.
    fn jacobian(&self, input: ArrayView1<'_, f64>) -> Result<JacobianEvaluation, StateError>;
}

/// The quotient contract `E'∘T = g∘E`: the code of the transitioned state is the
/// descended code.
pub struct QuotientContract<'a> {
    /// `T : H → H'`.
    pub transition: &'a dyn NativeMap,
    /// `E : H → Z`.
    pub encoder: &'a dyn NativeMap,
    /// `E' : H' → Z'`.
    pub next_encoder: &'a dyn NativeMap,
    /// `g : Z → Z'`.
    pub descended: &'a dyn NativeMap,
}

/// What [`QuotientContract::check`] resolved over the stated states.
#[derive(Clone, Debug, PartialEq)]
pub enum QuotientContractVerdict {
    /// At `state`, `‖E'(T h) − g(E h)‖∞ = defect` exceeds the combined roundoff
    /// of both sides: the code does not descend. This is the state with the
    /// largest excess.
    Violated {
        state: usize,
        defect: f64,
        roundoff: f64,
    },
    /// At every stated state the exact defect is at most `bound`, roundoff
    /// included.
    Bounded { bound: f64 },
}

impl QuotientContract<'_> {
    /// Check the contract at native states, the rows of `states`.
    pub fn check(
        &self,
        states: ArrayView2<'_, f64>,
    ) -> Result<QuotientContractVerdict, StateError> {
        let dimension = self.transition.input_dimension();
        require_dimension(
            self.encoder.input_dimension(),
            dimension,
            "quotient contract: E and T share a state space",
        )?;
        require_dimension(
            self.next_encoder.input_dimension(),
            self.transition.output_dimension(),
            "quotient contract: T into E'",
        )?;
        require_dimension(
            self.descended.input_dimension(),
            self.encoder.output_dimension(),
            "quotient contract: E into g",
        )?;
        require_dimension(
            self.descended.output_dimension(),
            self.next_encoder.output_dimension(),
            "quotient contract: E' and g share a code space",
        )?;
        require_sample(states, dimension, "quotient contract: states")?;
        let mut record = DefectRecord::new();
        for (index, state) in states.rows().into_iter().enumerate() {
            let moved = evaluated(self.transition, state, 0.0, "quotient contract: T(h)")?;
            let after = evaluated(
                self.next_encoder,
                moved.value.view(),
                moved.roundoff,
                "quotient contract: E'(T h)",
            )?;
            let code = evaluated(self.encoder, state, 0.0, "quotient contract: E(h)")?;
            let descended = evaluated(
                self.descended,
                code.value.view(),
                code.roundoff,
                "quotient contract: g(E h)",
            )?;
            record.observe(
                index,
                sup_distance(after.value.view(), descended.value.view()),
                after.roundoff + descended.roundoff,
            );
        }
        Ok(match record.violation {
            Some((state, defect, roundoff)) => QuotientContractVerdict::Violated {
                state,
                defect,
                roundoff,
            },
            None => QuotientContractVerdict::Bounded {
                bound: record.bound,
            },
        })
    }
}

/// The realization contract `T∘D = D'∘g`: decoded representatives are carried
/// to decoded representatives.
pub struct RealizationContract<'a> {
    /// `T : H → H'`.
    pub transition: &'a dyn NativeMap,
    /// `D : Z → H`.
    pub decoder: &'a dyn NativeMap,
    /// `D' : Z' → H'`.
    pub next_decoder: &'a dyn NativeMap,
    /// `g : Z → Z'`.
    pub descended: &'a dyn NativeMap,
}

/// What [`RealizationContract::check`] resolved over the stated codes.
#[derive(Clone, Debug, PartialEq)]
pub enum RealizationContractVerdict {
    /// At `code`, `‖T(D z) − D'(g z)‖∞ = defect` exceeds the combined roundoff
    /// of both sides: the decoded representative is not carried to the decoded
    /// representative. This is the code with the largest excess.
    Violated {
        code: usize,
        defect: f64,
        roundoff: f64,
    },
    /// At every stated code the exact defect is at most `bound`, roundoff
    /// included.
    Bounded { bound: f64 },
}

impl RealizationContract<'_> {
    /// Check the contract at codes, the rows of `codes`.
    pub fn check(
        &self,
        codes: ArrayView2<'_, f64>,
    ) -> Result<RealizationContractVerdict, StateError> {
        let dimension = self.decoder.input_dimension();
        require_dimension(
            self.descended.input_dimension(),
            dimension,
            "realization contract: D and g share a code space",
        )?;
        require_dimension(
            self.transition.input_dimension(),
            self.decoder.output_dimension(),
            "realization contract: D into T",
        )?;
        require_dimension(
            self.next_decoder.input_dimension(),
            self.descended.output_dimension(),
            "realization contract: g into D'",
        )?;
        require_dimension(
            self.next_decoder.output_dimension(),
            self.transition.output_dimension(),
            "realization contract: T and D' share a state space",
        )?;
        require_sample(codes, dimension, "realization contract: codes")?;
        let mut record = DefectRecord::new();
        for (index, code) in codes.rows().into_iter().enumerate() {
            let decoded = evaluated(self.decoder, code, 0.0, "realization contract: D(z)")?;
            let moved = evaluated(
                self.transition,
                decoded.value.view(),
                decoded.roundoff,
                "realization contract: T(D z)",
            )?;
            let descended = evaluated(self.descended, code, 0.0, "realization contract: g(z)")?;
            let represented = evaluated(
                self.next_decoder,
                descended.value.view(),
                descended.roundoff,
                "realization contract: D'(g z)",
            )?;
            record.observe(
                index,
                sup_distance(moved.value.view(), represented.value.view()),
                moved.roundoff + represented.roundoff,
            );
        }
        Ok(match record.violation {
            Some((code, defect, roundoff)) => RealizationContractVerdict::Violated {
                code,
                defect,
                roundoff,
            },
            None => RealizationContractVerdict::Bounded {
                bound: record.bound,
            },
        })
    }
}

/// A candidate sufficient state: an encoder together with a section of it.
pub struct StateChart<'a> {
    /// `E : H → Z`.
    pub encoder: &'a dyn NativeMap,
    /// `D : Z → H` with `E∘D = id` on the codes the encoder produces.
    pub decoder: &'a dyn NativeMap,
}

/// What [`fiber_test`] resolved over the stated states and futures.
#[derive(Clone, Debug, PartialEq)]
pub enum FiberVerdict {
    /// The chart maps `state` and its section representative within
    /// `merge_bound` of each other in code space, yet `future` separates them
    /// by at least `separation_lower_bound`, which exceeds the declared
    /// fidelity: the chart is not a sufficient state. This is the pair with the
    /// largest certified separation.
    Separated {
        state: usize,
        future: usize,
        representative: Array1<f64>,
        separation_lower_bound: f64,
        merge_bound: f64,
    },
    /// At every tested state every future agrees with its value at the
    /// representative within `separation_upper_bound`, roundoff included, and
    /// that bound is within the declared fidelity. `vacuous_states` lay on the
    /// section and tested nothing.
    WithinFidelity {
        tested_states: usize,
        vacuous_states: usize,
        separation_upper_bound: f64,
    },
    /// No separation is certified above the declared fidelity, but the largest
    /// separation's roundoff interval contains it.
    Unresolved {
        tested_states: usize,
        vacuous_states: usize,
        separation_lower_bound: f64,
        separation_upper_bound: f64,
    },
}

/// Test whether `chart` is a sufficient state for `futures` at the stated native
/// states, the rows of `states`.
///
/// Each state `h` is compared with its section representative `h̃ = D(E(h))`,
/// which the chart cannot tell apart from `h`, on every future. `fidelity` is
/// the declared largest sup-norm response difference that counts as agreement.
///
/// Refusals:
/// * [`StateError::NotASection`] when `E(D(E h))` differs from `E(h)` by more
///   than the arithmetic explains. Then `h̃` is not in the fibre of `h`, so a
///   separation would say nothing about `E`.
/// * [`StateError::VacuousFiberTest`] when every state lies on the section
///   within the decoder's roundoff, so every comparison is of a state with
///   itself.
pub fn fiber_test(
    chart: &StateChart<'_>,
    futures: &[&dyn NativeMap],
    states: ArrayView2<'_, f64>,
    fidelity: f64,
) -> Result<FiberVerdict, StateError> {
    if !fidelity.is_finite() || fidelity < 0.0 {
        return Err(StateError::InvalidFidelity { value: fidelity });
    }
    let dimension = chart.encoder.input_dimension();
    require_dimension(
        chart.decoder.input_dimension(),
        chart.encoder.output_dimension(),
        "fiber test: encoder into decoder",
    )?;
    require_dimension(
        chart.decoder.output_dimension(),
        dimension,
        "fiber test: decoder into the state space",
    )?;
    if futures.is_empty() {
        return Err(StateError::EmptyFamily {
            context: "fiber test: futures",
        });
    }
    for future in futures {
        require_dimension(
            future.input_dimension(),
            dimension,
            "fiber test: future on the state space",
        )?;
    }
    require_sample(states, dimension, "fiber test: states")?;
    let mut tested_states = 0;
    let mut vacuous_states = 0;
    let mut largest_lower = 0.0_f64;
    let mut largest_upper = 0.0_f64;
    let mut separated: Option<FiberVerdict> = None;
    for (index, state) in states.rows().into_iter().enumerate() {
        let code = evaluated(chart.encoder, state, 0.0, "fiber test: E(h)")?;
        let representative =
            evaluated(chart.decoder, code.value.view(), 0.0, "fiber test: D(E h)")?;
        let round_trip = evaluated(
            chart.encoder,
            representative.value.view(),
            representative.roundoff,
            "fiber test: E(D(E h))",
        )?;
        let code_defect = sup_distance(round_trip.value.view(), code.value.view());
        if code_defect > round_trip.roundoff {
            return Err(StateError::NotASection {
                state: index,
                code_defect,
                roundoff: round_trip.roundoff,
            });
        }
        if sup_distance(state, representative.value.view()) <= representative.roundoff {
            vacuous_states += 1;
            continue;
        }
        tested_states += 1;
        let merge_bound = code.roundoff + code_defect + round_trip.roundoff;
        for (future_index, future) in futures.iter().enumerate() {
            let native = evaluated(*future, state, 0.0, "fiber test: r(h)")?;
            let represented = evaluated(
                *future,
                representative.value.view(),
                0.0,
                "fiber test: r(D(E h))",
            )?;
            let distance = sup_distance(native.value.view(), represented.value.view());
            let roundoff = native.roundoff + represented.roundoff;
            let lower = distance - roundoff;
            largest_lower = largest_lower.max(lower);
            largest_upper = largest_upper.max(distance + roundoff);
            let larger = match &separated {
                Some(FiberVerdict::Separated {
                    separation_lower_bound,
                    ..
                }) => lower > *separation_lower_bound,
                _ => true,
            };
            if lower > fidelity && larger {
                separated = Some(FiberVerdict::Separated {
                    state: index,
                    future: future_index,
                    representative: representative.value.clone(),
                    separation_lower_bound: lower,
                    merge_bound,
                });
            }
        }
    }
    if tested_states == 0 {
        return Err(StateError::VacuousFiberTest {
            states: vacuous_states,
        });
    }
    if let Some(verdict) = separated {
        return Ok(verdict);
    }
    Ok(if largest_upper <= fidelity {
        FiberVerdict::WithinFidelity {
            tested_states,
            vacuous_states,
            separation_upper_bound: largest_upper,
        }
    } else {
        FiberVerdict::Unresolved {
            tested_states,
            vacuous_states,
            separation_lower_bound: largest_lower,
            separation_upper_bound: largest_upper,
        }
    })
}

/// The local quotient at one state, read off the stacked response Jacobian.
#[derive(Clone, Debug)]
pub struct LocalQuotient {
    /// Singular values of the stacked Jacobian, descending.
    pub singular_values: Vec<f64>,
    /// The SVD's backward-error band plus the stacked Jacobians' formation
    /// bound.
    pub band: f64,
    /// Number of singular values above `band`: a certified lower bound on the
    /// exact rank at the state.
    pub resolved_rank: usize,
    /// `min(response rows, d)`, the most the rank can be.
    pub rank_ceiling: usize,
    /// Orthonormal rows (`resolved_rank × d`) spanning the state directions the
    /// futures resolve. To first order their orthogonal complement is tangent to
    /// the fibre.
    pub observed_directions: Array2<f64>,
}

/// The local dimension a state's resolved rank supports.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LocalDimension {
    /// The resolved rank reaches its ceiling, so the rank is exact at the state.
    /// By lower semicontinuity it is constant on a neighbourhood. For a
    /// generating family this is the local minimal dimension.
    Certified(usize),
    /// The resolved rank equals the sample's generic rank but is below its
    /// ceiling. The exact rank is at least this; a larger one hidden below the
    /// band is not excluded.
    AtLeast(usize),
    /// The resolved rank is below the sample's generic rank. The constant-rank
    /// theorem gives no chart here, and this rank is not the local dimension.
    Singular {
        resolved_rank: usize,
        generic_rank: usize,
    },
}

/// The resolved ranks of a response family over stated states.
#[derive(Clone, Debug)]
pub struct ConstantRankCheck {
    /// One local quotient per stated state, in row order.
    pub states: Vec<LocalQuotient>,
    /// The largest resolved rank over the stated states: a certified lower
    /// bound on the family's largest rank on them.
    pub generic_rank: usize,
}

impl ConstantRankCheck {
    /// The local dimension at stated state `state`, relative to the sample's
    /// generic rank.
    pub fn local_dimension(&self, state: usize) -> Option<LocalDimension> {
        self.states.get(state).map(|local| {
            if local.resolved_rank < self.generic_rank {
                LocalDimension::Singular {
                    resolved_rank: local.resolved_rank,
                    generic_rank: self.generic_rank,
                }
            } else if local.resolved_rank == local.rank_ceiling {
                LocalDimension::Certified(local.resolved_rank)
            } else {
                LocalDimension::AtLeast(local.resolved_rank)
            }
        })
    }
}

/// Resolve the rank of the stacked Jacobian of `futures` at each stated state,
/// the rows of `states`.
///
/// This is the rank of the family as stated. It is the local minimal dimension
/// of the permitted family only if the stated futures generate it, and only at
/// states that are not singular. The check forms the stacked
/// `response rows × d` Jacobian of each state and its thin SVD.
pub fn constant_rank_check(
    futures: &[&dyn DifferentiableNativeMap],
    states: ArrayView2<'_, f64>,
) -> Result<ConstantRankCheck, StateError> {
    let first = futures.first().ok_or(StateError::EmptyFamily {
        context: "constant rank: futures",
    })?;
    let dimension = first.input_dimension();
    for future in futures {
        require_dimension(
            future.input_dimension(),
            dimension,
            "constant rank: future on the state space",
        )?;
    }
    require_sample(states, dimension, "constant rank: states")?;
    let rows: usize = futures.iter().map(|future| future.output_dimension()).sum();
    if rows == 0 {
        return Err(StateError::EmptyFamily {
            context: "constant rank: responses",
        });
    }
    let mut locals = Vec::with_capacity(states.nrows());
    for state in states.rows() {
        let mut stacked = Array2::<f64>::zeros((rows, dimension));
        let mut formation_squared = 0.0_f64;
        let mut offset = 0;
        for future in futures {
            let jacobian = future.jacobian(state)?;
            let height = future.output_dimension();
            if jacobian.matrix.dim() != (height, dimension) {
                return Err(StateError::DimensionMismatch {
                    context: "constant rank: Jacobian shape",
                    expected: height * dimension,
                    found: jacobian.matrix.len(),
                });
            }
            if !jacobian.roundoff.is_finite()
                || jacobian.roundoff < 0.0
                || jacobian.matrix.iter().any(|value| !value.is_finite())
            {
                return Err(StateError::NonFinite {
                    context: "constant rank: Jacobian",
                });
            }
            stacked
                .slice_mut(s![offset..offset + height, ..])
                .assign(&jacobian.matrix);
            formation_squared += jacobian.roundoff * jacobian.roundoff;
            offset += height;
        }
        // The spectral norm of a stacked error is at most the root sum of the
        // blocks' squared spectral norms.
        locals.push(local_quotient(&stacked, formation_squared.sqrt())?);
    }
    let generic_rank = locals
        .iter()
        .map(|local| local.resolved_rank)
        .max()
        .unwrap_or(0);
    Ok(ConstantRankCheck {
        states: locals,
        generic_rank,
    })
}

fn local_quotient(stacked: &Array2<f64>, formation: f64) -> Result<LocalQuotient, StateError> {
    let (rows, cols) = stacked.dim();
    let (_, sigma, vt) = stacked
        .svd(false, true)
        .map_err(|source| StateError::Svd {
            context: "constant rank: stacked Jacobian",
            source,
        })?;
    let vt = vt.ok_or(StateError::Svd {
        context: "constant rank: right singular vectors",
        source: FaerLinalgError::SvdNoConvergence {
            context: "constant rank: right singular vectors",
        },
    })?;
    let mut order: Vec<usize> = (0..sigma.len()).collect();
    order.sort_by(|&left, &right| sigma[right].total_cmp(&sigma[left]));
    let singular_values: Vec<f64> = order.iter().map(|&index| sigma[index]).collect();
    let sigma_max = singular_values.first().copied().unwrap_or(0.0);
    let band = factor_singular_band(rows, cols, sigma_max) + formation;
    let resolved_rank = singular_values.iter().filter(|&&value| value > band).count();
    let mut observed_directions = Array2::<f64>::zeros((resolved_rank, cols));
    for (row, &index) in order.iter().take(resolved_rank).enumerate() {
        observed_directions.row_mut(row).assign(&vt.row(index));
    }
    Ok(LocalQuotient {
        singular_values,
        band,
        resolved_rank,
        rank_ceiling: rows.min(cols),
        observed_directions,
    })
}

/// Errors and refusals of the sufficient-state checks.
#[derive(Debug)]
pub enum StateError {
    /// No futures, responses, states or codes were stated.
    EmptyFamily { context: &'static str },
    /// Two dimensions that must agree do not.
    DimensionMismatch {
        context: &'static str,
        expected: usize,
        found: usize,
    },
    /// A stated sample, value, Jacobian or roundoff bound is not finite.
    NonFinite { context: &'static str },
    /// The declared fidelity tolerance is negative or not finite.
    InvalidFidelity { value: f64 },
    /// A native map could not execute.
    Execution { reason: String },
    /// `E(D(E h))` differs from `E(h)` by more than the arithmetic explains, so
    /// the decoder is not a section of the encoder at this state's code.
    NotASection {
        state: usize,
        code_defect: f64,
        roundoff: f64,
    },
    /// Every stated state lies on the chart's section within the decoder's
    /// roundoff, so each comparison was of a state with itself and no fibre was
    /// tested.
    VacuousFiberTest { states: usize },
    /// A singular value decomposition failed.
    Svd {
        context: &'static str,
        source: FaerLinalgError,
    },
}

impl fmt::Display for StateError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyFamily { context } => write!(formatter, "{context}: nothing was stated"),
            Self::DimensionMismatch {
                context,
                expected,
                found,
            } => write!(formatter, "{context}: expected dimension {expected}, found {found}"),
            Self::NonFinite { context } => write!(formatter, "{context}: non-finite value"),
            Self::InvalidFidelity { value } => write!(
                formatter,
                "declared fidelity tolerance {value} is not a finite non-negative number"
            ),
            Self::Execution { reason } => write!(formatter, "native map failed: {reason}"),
            Self::NotASection {
                state,
                code_defect,
                roundoff,
            } => write!(
                formatter,
                "fiber test: at state {state} the decoder is not a section: E(D(E h)) differs \
                 from E(h) by {code_defect:e}, above the roundoff bound {roundoff:e}"
            ),
            Self::VacuousFiberTest { states } => write!(
                formatter,
                "fiber test: all {states} states lie on the chart's section, so no fibre was tested"
            ),
            Self::Svd { context, source } => {
                write!(formatter, "{context}: singular value decomposition failed: {source}")
            }
        }
    }
}

impl std::error::Error for StateError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Svd { source, .. } => Some(source),
            _ => None,
        }
    }
}

/// The stated item whose computed defect most exceeds its roundoff, and the
/// uniform bound over all of them.
struct DefectRecord {
    violation: Option<(usize, f64, f64)>,
    bound: f64,
}

impl DefectRecord {
    fn new() -> Self {
        Self {
            violation: None,
            bound: 0.0,
        }
    }

    fn observe(&mut self, index: usize, defect: f64, roundoff: f64) {
        self.bound = self.bound.max(defect + roundoff);
        let excess = defect - roundoff;
        let larger = match self.violation {
            Some((_, worst_defect, worst_roundoff)) => excess > worst_defect - worst_roundoff,
            None => true,
        };
        if excess > 0.0 && larger {
            self.violation = Some((index, defect, roundoff));
        }
    }
}

fn evaluated<M: NativeMap + ?Sized>(
    map: &M,
    input: ArrayView1<'_, f64>,
    input_roundoff: f64,
    context: &'static str,
) -> Result<Evaluation, StateError> {
    require_dimension(input.len(), map.input_dimension(), context)?;
    let evaluation = map.evaluate(input, input_roundoff)?;
    require_dimension(evaluation.value.len(), map.output_dimension(), context)?;
    if !evaluation.roundoff.is_finite()
        || evaluation.roundoff < 0.0
        || evaluation.value.iter().any(|value| !value.is_finite())
    {
        return Err(StateError::NonFinite { context });
    }
    Ok(evaluation)
}

fn require_dimension(found: usize, expected: usize, context: &'static str) -> Result<(), StateError> {
    if found == expected {
        Ok(())
    } else {
        Err(StateError::DimensionMismatch {
            context,
            expected,
            found,
        })
    }
}

fn require_sample(
    sample: ArrayView2<'_, f64>,
    dimension: usize,
    context: &'static str,
) -> Result<(), StateError> {
    if sample.nrows() == 0 {
        return Err(StateError::EmptyFamily { context });
    }
    require_dimension(sample.ncols(), dimension, context)?;
    if sample.iter().any(|value| !value.is_finite()) {
        return Err(StateError::NonFinite { context });
    }
    Ok(())
}

fn sup_distance(left: ArrayView1<'_, f64>, right: ArrayView1<'_, f64>) -> f64 {
    left.iter()
        .zip(right.iter())
        .fold(0.0_f64, |largest, (a, b)| largest.max((a - b).abs()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use gam_linalg::roundoff::{accumulation_band, accumulation_growth};
    use ndarray::array;

    /// A fixture map given by closed forms. `evaluate` returns the value and its
    /// roundoff bound for an input roundoff; `jacobian` returns the Jacobian and
    /// its formation bound.
    struct Closed {
        input: usize,
        output: usize,
        evaluate: for<'v> fn(ArrayView1<'v, f64>, f64) -> (Array1<f64>, f64),
        jacobian: for<'v> fn(ArrayView1<'v, f64>) -> (Array2<f64>, f64),
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
            let (matrix, roundoff) = (self.jacobian)(input);
            Ok(JacobianEvaluation { matrix, roundoff })
        }
    }

    /// A fixture affine map `x ↦ Mx + c`. Row `i` is an inner product of length
    /// `n` plus one addition, so it rounds within `γ_{n+1}(Σⱼ|Mᵢⱼxⱼ| + |cᵢ|)`.
    /// Moving the input by `δ` moves the value by at most `‖M‖∞·δ`. The Jacobian
    /// `M` is exact.
    struct Affine {
        matrix: Array2<f64>,
        offset: Array1<f64>,
    }

    impl NativeMap for Affine {
        fn input_dimension(&self) -> usize {
            self.matrix.ncols()
        }

        fn output_dimension(&self) -> usize {
            self.matrix.nrows()
        }

        fn evaluate(
            &self,
            input: ArrayView1<'_, f64>,
            input_roundoff: f64,
        ) -> Result<Evaluation, StateError> {
            let value = self.matrix.dot(&input) + &self.offset;
            let mut roundoff = 0.0_f64;
            let mut row_sum_norm = 0.0_f64;
            for (row, shift) in self.matrix.rows().into_iter().zip(self.offset.iter()) {
                let absolute_sum = row
                    .iter()
                    .zip(input.iter())
                    .map(|(entry, coordinate)| (entry * coordinate).abs())
                    .sum::<f64>()
                    + shift.abs();
                roundoff = roundoff.max(accumulation_band(row.len() + 1, absolute_sum));
                row_sum_norm = row_sum_norm.max(row.iter().map(|entry| entry.abs()).sum::<f64>());
            }
            Ok(Evaluation {
                value,
                roundoff: roundoff + row_sum_norm * input_roundoff,
            })
        }
    }

    impl DifferentiableNativeMap for Affine {
        fn jacobian(&self, input: ArrayView1<'_, f64>) -> Result<JacobianEvaluation, StateError> {
            require_dimension(input.len(), self.matrix.ncols(), "affine fixture: Jacobian")?;
            Ok(JacobianEvaluation {
                matrix: self.matrix.clone(),
                roundoff: 0.0,
            })
        }
    }

    /// `o(h) = h²` on `ℝ¹`: the current observation. One product rounds once;
    /// moving the input by `δ` moves the value by at most `(2|h| + δ)δ`.
    fn square() -> Closed {
        Closed {
            input: 1,
            output: 1,
            evaluate: |h, delta| {
                let x = h[0];
                (
                    array![x * x],
                    accumulation_growth(1) * x * x + (2.0 * x.abs() + delta) * delta,
                )
            },
            jacobian: |h| (array![[2.0 * h[0]]], 0.0),
        }
    }

    /// `o(T h) = (h + 1)²`: shift, then observe. The sum and the product give
    /// `γ₃(|h| + 1)²`.
    fn shifted_square() -> Closed {
        Closed {
            input: 1,
            output: 1,
            evaluate: |h, delta| {
                let x = h[0];
                let magnitude = x.abs() + 1.0;
                (
                    array![(x + 1.0) * (x + 1.0)],
                    accumulation_growth(3) * magnitude * magnitude
                        + (2.0 * magnitude + delta) * delta,
                )
            },
            jacobian: |h| {
                (
                    array![[2.0 * (h[0] + 1.0)]],
                    accumulation_growth(1) * 2.0 * (h[0].abs() + 1.0),
                )
            },
        }
    }

    /// The section `D(z) = √z` of the current observation. `√` is correctly
    /// rounded and ½-Hölder: `|√x − √z| ≤ √|x − z|`.
    fn square_root() -> Closed {
        Closed {
            input: 1,
            output: 1,
            evaluate: |z, delta| {
                let root = z[0].sqrt();
                (array![root], accumulation_growth(1) * root + delta.sqrt())
            },
            jacobian: |z| {
                let root = z[0].sqrt();
                (array![[0.5 / root]], accumulation_growth(2) * 0.5 / root)
            },
        }
    }

    /// `D(z) = 2√z`, which is not a section of `h²`.
    fn doubled_square_root() -> Closed {
        Closed {
            input: 1,
            output: 1,
            evaluate: |z, delta| {
                let root = z[0].sqrt();
                (
                    array![2.0 * root],
                    accumulation_growth(1) * 2.0 * root + 2.0 * delta.sqrt(),
                )
            },
            jacobian: |z| {
                let root = z[0].sqrt();
                (array![[1.0 / root]], accumulation_growth(2) / root)
            },
        }
    }

    /// `(a, b) ↦ a²` on `ℝ²`.
    fn first_square() -> Closed {
        Closed {
            input: 2,
            output: 1,
            evaluate: |h, delta| {
                let x = h[0];
                (
                    array![x * x],
                    accumulation_growth(1) * x * x + (2.0 * x.abs() + delta) * delta,
                )
            },
            jacobian: |h| (array![[2.0 * h[0], 0.0]], 0.0),
        }
    }

    /// `(a, b) ↦ (a + 1)²` on `ℝ²`.
    fn first_shifted_square() -> Closed {
        Closed {
            input: 2,
            output: 1,
            evaluate: |h, delta| {
                let x = h[0];
                let magnitude = x.abs() + 1.0;
                (
                    array![(x + 1.0) * (x + 1.0)],
                    accumulation_growth(3) * magnitude * magnitude
                        + (2.0 * magnitude + delta) * delta,
                )
            },
            jacobian: |h| {
                (
                    array![[2.0 * (h[0] + 1.0), 0.0]],
                    accumulation_growth(1) * 2.0 * (h[0].abs() + 1.0),
                )
            },
        }
    }

    /// `(z, n) ↦ z`: observe the first coordinate.
    fn first_coordinate() -> Affine {
        Affine {
            matrix: array![[1.0, 0.0]],
            offset: array![0.0],
        }
    }

    /// `(a, b) ↦ a + b`, the observation after `T(a, b) = (a + b, b)`.
    fn coordinate_sum() -> Affine {
        Affine {
            matrix: array![[1.0, 1.0]],
            offset: array![0.0],
        }
    }

    /// The section `z ↦ (z, 0)` of the first coordinate.
    fn zero_section() -> Affine {
        Affine {
            matrix: array![[1.0], [0.0]],
            offset: array![0.0, 0.0],
        }
    }

    /// The section `z ↦ (z, 3z)` of the first coordinate.
    fn tilted_section() -> Affine {
        Affine {
            matrix: array![[1.0], [3.0]],
            offset: array![0.0, 0.0],
        }
    }

    /// The identity on a one-dimensional code.
    fn code_identity() -> Affine {
        Affine {
            matrix: array![[1.0]],
            offset: array![0.0],
        }
    }

    /// `T(z, n) = (z, n + 1)`: the unobserved coordinate counts steps.
    fn counting_transition() -> Affine {
        Affine {
            matrix: array![[1.0, 0.0], [0.0, 1.0]],
            offset: array![0.0, 1.0],
        }
    }

    /// `T(z, n) = (z + n, n)`: the observed coordinate drifts by the unobserved
    /// one, and the section `n = 0` is fixed.
    fn drifting_transition() -> Affine {
        Affine {
            matrix: array![[1.0, 1.0], [0.0, 1.0]],
            offset: array![0.0, 0.0],
        }
    }

    /// A9, the h² fixture. The current observation `o(h) = h²` merges `±1`, and
    /// the future `h ↦ h + 1` separates them (`0` against `4`), so the fiber test
    /// refuses the current-observation chart. Two controls: the observation alone
    /// agrees at the merged pair, and the chart's full rank at `h = ±1` does not
    /// detect the merge.
    #[test]
    fn h_squared_fixture_refuses_the_current_observation_chart() {
        let observe = square();
        let observe_after_shift = shifted_square();
        let section = square_root();
        let chart = StateChart {
            encoder: &observe,
            decoder: &section,
        };
        let states = array![[-1.0], [1.0], [-0.5]];

        let futures: [&dyn NativeMap; 2] = [&observe, &observe_after_shift];
        match fiber_test(&chart, &futures, states.view(), 0.0).expect("fiber test") {
            FiberVerdict::Separated {
                state,
                future,
                representative,
                separation_lower_bound,
                merge_bound,
            } => {
                assert_eq!((state, future), (0, 1), "h = -1 against h = +1 under o∘T");
                assert_eq!(representative, array![1.0]);
                // 4 minus the two shifted-square bounds γ₃·2², less one rounding.
                assert!(separation_lower_bound <= 4.0);
                assert!(
                    separation_lower_bound >= 4.0 - 16.0 * accumulation_growth(3),
                    "separation {separation_lower_bound} must be 4 to roundoff"
                );
                // E(-1) rounds within γ₁, D(1) within γ₁, and E at D(1) within
                // γ₁ + (2 + γ₁)γ₁.
                assert!(
                    merge_bound <= 5.0 * accumulation_growth(1),
                    "the chart merges -1 and +1 to roundoff, found {merge_bound}"
                );
            }
            other => panic!("the current-observation chart must be refused, got {other:?}"),
        }

        let observation_only: [&dyn NativeMap; 1] = [&observe];
        match fiber_test(&chart, &observation_only, states.view(), 0.0).expect("fiber test") {
            FiberVerdict::Unresolved {
                tested_states,
                vacuous_states,
                separation_lower_bound,
                separation_upper_bound,
            } => {
                assert_eq!((tested_states, vacuous_states), (2, 1));
                assert_eq!(separation_lower_bound, 0.0);
                assert!(
                    separation_upper_bound <= 2.0 * accumulation_growth(1),
                    "the current observation agrees at the merged pair, found {separation_upper_bound}"
                );
            }
            other => panic!("the observation alone separates nothing, got {other:?}"),
        }

        let differentiable: [&dyn DifferentiableNativeMap; 1] = [&observe];
        let ranks = constant_rank_check(&differentiable, array![[-1.0], [1.0]].view())
            .expect("rank check");
        assert_eq!(ranks.local_dimension(0), Some(LocalDimension::Certified(1)));
        assert_eq!(ranks.local_dimension(1), Some(LocalDimension::Certified(1)));
    }

    /// A check on decoder sections does not test fibres. States on the section
    /// `h = √(h²)` compare each state with itself, and the test refuses rather
    /// than report agreement. A decoder that is not a section is refused, and so
    /// is an invalid declared fidelity.
    #[test]
    fn fiber_test_refuses_section_only_samples_and_non_sections() {
        let observe = square();
        let observe_after_shift = shifted_square();
        let section = square_root();
        let chart = StateChart {
            encoder: &observe,
            decoder: &section,
        };
        let futures: [&dyn NativeMap; 2] = [&observe, &observe_after_shift];

        let on_section = array![[0.5], [1.0], [2.0]];
        match fiber_test(&chart, &futures, on_section.view(), 0.0) {
            Err(StateError::VacuousFiberTest { states }) => assert_eq!(states, 3),
            other => panic!("a section-only sample must be refused, got {other:?}"),
        }

        let not_a_section = doubled_square_root();
        let broken = StateChart {
            encoder: &observe,
            decoder: &not_a_section,
        };
        match fiber_test(&broken, &futures, array![[-1.0]].view(), 0.0) {
            Err(StateError::NotASection {
                state, code_defect, ..
            }) => {
                assert_eq!(state, 0);
                assert_eq!(code_defect, 3.0, "E(D(E(-1))) = 4 against E(-1) = 1");
            }
            other => panic!("a non-section decoder must be refused, got {other:?}"),
        }

        match fiber_test(&chart, &futures, array![[-1.0]].view(), -1.0) {
            Err(StateError::InvalidFidelity { value }) => assert_eq!(value, -1.0),
            other => panic!("a negative declared fidelity must be refused, got {other:?}"),
        }
    }

    /// A sufficient chart with non-trivial fibres. The futures read only `a` of
    /// `(a, b)`, so the chart `(a, b) ↦ a` with section `a ↦ (a, 0)` passes at
    /// states off the section, within a fidelity declared at twice the futures'
    /// combined roundoff. Control: a declared fidelity of zero cannot be certified
    /// from floating point, so it stays unresolved.
    #[test]
    fn a_sufficient_chart_passes_off_the_section_within_its_roundoff() {
        let observe = first_square();
        let observe_after_shift = first_shifted_square();
        let encoder = first_coordinate();
        let section = zero_section();
        let chart = StateChart {
            encoder: &encoder,
            decoder: &section,
        };
        let futures: [&dyn NativeMap; 2] = [&observe, &observe_after_shift];
        let states = array![[-1.0, 5.0], [2.0, -3.0]];

        // The largest pair of shifted-square bounds is at |a| = 2: 2·γ₃·3².
        let fidelity = 2.0 * 18.0 * accumulation_growth(3);
        match fiber_test(&chart, &futures, states.view(), fidelity).expect("fiber test") {
            FiberVerdict::WithinFidelity {
                tested_states,
                vacuous_states,
                separation_upper_bound,
            } => {
                assert_eq!((tested_states, vacuous_states), (2, 0));
                assert!(separation_upper_bound <= fidelity);
            }
            other => panic!("the first-coordinate chart is sufficient, got {other:?}"),
        }

        match fiber_test(&chart, &futures, states.view(), 0.0).expect("fiber test") {
            FiberVerdict::Unresolved {
                separation_lower_bound,
                separation_upper_bound,
                ..
            } => {
                assert_eq!(separation_lower_bound, 0.0);
                assert!(separation_upper_bound > 0.0);
            }
            other => panic!("exact agreement is never certified, got {other:?}"),
        }
    }

    /// A9, the quotient-versus-realization counterexample `T(z, n) = (z, n + 1)`
    /// with only `z` observed. The quotient contract holds at every stated state.
    /// The stationary realization `D' = D`, the one an autonomous program needs,
    /// fails at every code, for the section `(z, 0)` and for the section `(z, 3z)`.
    /// Control: the per-step decoder `D'(z) = (z, 1)` after `D(z) = (z, 0)` does
    /// realize one step, so the refutation is of the stationary form only.
    #[test]
    fn counting_transition_satisfies_the_quotient_but_not_the_realization_contract() {
        let transition = counting_transition();
        let observe = first_coordinate();
        let identity = code_identity();
        let quotient = QuotientContract {
            transition: &transition,
            encoder: &observe,
            next_encoder: &observe,
            descended: &identity,
        };
        let states = array![[0.0, 0.0], [1.5, -2.0], [-3.0, 7.0]];
        match quotient.check(states.view()).expect("quotient check") {
            QuotientContractVerdict::Bounded { bound } => {
                // At (-3, 7): T rounds within 8γ₃, E'(T h) within 3γ₃ + 8γ₃,
                // E(h) within 3γ₃ and g(E h) within 3γ₂ + 3γ₃: 14γ₃ + 3γ₂ ≤ 17γ₃.
                assert!(
                    bound <= 17.0 * accumulation_growth(3),
                    "the code descends to roundoff, found {bound}"
                );
            }
            other => panic!("the quotient contract holds, got {other:?}"),
        }

        let codes = array![[0.0], [1.5], [-3.0]];
        let zero = zero_section();
        let tilted = tilted_section();
        let sections: [&dyn NativeMap; 2] = [&zero, &tilted];
        for section in sections {
            let realization = RealizationContract {
                transition: &transition,
                decoder: section,
                next_decoder: section,
                descended: &identity,
            };
            match realization.check(codes.view()).expect("realization check") {
                RealizationContractVerdict::Violated { code, defect, .. } => {
                    assert_eq!(code, 0);
                    assert_eq!(defect, 1.0, "T(D z) sits one step along the fibre from D(z)");
                }
                other => panic!("no section realizes the counting transition, got {other:?}"),
            }
        }

        let unit_section = Affine {
            matrix: array![[1.0], [0.0]],
            offset: array![0.0, 1.0],
        };
        let per_step = RealizationContract {
            transition: &transition,
            decoder: &zero,
            next_decoder: &unit_section,
            descended: &identity,
        };
        match per_step.check(codes.view()).expect("realization check") {
            RealizationContractVerdict::Bounded { bound } => {
                // At z = -3: T(D z) rounds within 3γ₃ + 3γ₂ and D'(g z) within
                // 3γ₂ + 3γ₂: 3γ₃ + 9γ₂ ≤ 12γ₃.
                assert!(
                    bound <= 12.0 * accumulation_growth(3),
                    "a per-step decoder realizes one step, found {bound}"
                );
            }
            other => panic!("the per-step decoder realizes one step, got {other:?}"),
        }
    }

    /// The converse: `T(z, n) = (z + n, n)` fixes the section `(z, 0)`, so the
    /// realization contract holds, while the quotient contract fails at states
    /// with `n ≠ 0`, most at `n = 7`.
    #[test]
    fn drifting_transition_satisfies_the_realization_but_not_the_quotient_contract() {
        let transition = drifting_transition();
        let observe = first_coordinate();
        let identity = code_identity();
        let section = zero_section();
        let realization = RealizationContract {
            transition: &transition,
            decoder: &section,
            next_decoder: &section,
            descended: &identity,
        };
        match realization
            .check(array![[0.0], [1.5], [-3.0]].view())
            .expect("realization check")
        {
            RealizationContractVerdict::Bounded { bound } => {
                // At z = -3: D(z) rounds within 3γ₂, T(D z) within 3γ₃ + 2·3γ₂,
                // g(z) within 3γ₂ and D'(g z) within 3γ₂ + 3γ₂: 3γ₃ + 12γ₂ ≤ 15γ₃.
                assert!(
                    bound <= 15.0 * accumulation_growth(3),
                    "the section is fixed to roundoff, found {bound}"
                );
            }
            other => panic!("the section realizes the drift, got {other:?}"),
        }

        let quotient = QuotientContract {
            transition: &transition,
            encoder: &observe,
            next_encoder: &observe,
            descended: &identity,
        };
        match quotient
            .check(array![[0.0, 0.0], [1.5, -2.0], [-3.0, 7.0]].view())
            .expect("quotient check")
        {
            QuotientContractVerdict::Violated { state, defect, .. } => {
                assert_eq!(state, 2);
                assert_eq!(defect, 7.0);
            }
            other => panic!("the observed coordinate does not descend, got {other:?}"),
        }
    }

    /// The singular-point caveat and the generation assumption. The current
    /// observation `h²` loses rank at `h = 0`, while the generating family
    /// `{h², (h + 1)²}` has constant rank 1 there. Observing `a` of `(a, b)` has
    /// rank 1, and adding the observation after `T(a, b) = (a + b, b)` gives 2: a
    /// family that does not generate understates the minimal dimension.
    #[test]
    fn rank_drops_at_singular_points_and_depends_on_generation() {
        let observe = square();
        let observe_after_shift = shifted_square();
        let states = array![[-1.0], [0.0], [1.0]];

        let current: [&dyn DifferentiableNativeMap; 1] = [&observe];
        let ranks = constant_rank_check(&current, states.view()).expect("rank check");
        assert_eq!(ranks.generic_rank, 1);
        assert_eq!(ranks.local_dimension(0), Some(LocalDimension::Certified(1)));
        assert_eq!(
            ranks.local_dimension(1),
            Some(LocalDimension::Singular {
                resolved_rank: 0,
                generic_rank: 1,
            })
        );
        assert_eq!(ranks.local_dimension(2), Some(LocalDimension::Certified(1)));
        assert_eq!(ranks.local_dimension(3), None);

        let generating: [&dyn DifferentiableNativeMap; 2] = [&observe, &observe_after_shift];
        let generated = constant_rank_check(&generating, states.view()).expect("rank check");
        for state in 0..3 {
            assert_eq!(
                generated.local_dimension(state),
                Some(LocalDimension::Certified(1)),
                "state {state}"
            );
        }

        let first = first_coordinate();
        let after_transition = coordinate_sum();
        let pair_states = array![[0.5, -1.0], [2.0, 3.0]];
        let observed_only: [&dyn DifferentiableNativeMap; 1] = [&first];
        let partial = constant_rank_check(&observed_only, pair_states.view()).expect("rank check");
        assert_eq!(partial.generic_rank, 1);
        assert_eq!(partial.states[0].observed_directions.dim(), (1, 2));
        let closed: [&dyn DifferentiableNativeMap; 2] = [&first, &after_transition];
        let full = constant_rank_check(&closed, pair_states.view()).expect("rank check");
        assert_eq!(full.generic_rank, 2);
        assert_eq!(full.local_dimension(0), Some(LocalDimension::Certified(2)));
    }
}
