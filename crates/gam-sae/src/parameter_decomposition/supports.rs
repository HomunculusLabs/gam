//! Robust supports and the evidence status of every reported number (#2951).
//!
//! # Evidence status
//!
//! A reported number is a claim about a quantity `Q` over a declared domain `D`:
//! a mask domain, a finite family of masks, a covering region, or a sampling law.
//! The number is an extremum of `Q` over the domain (a worst-case divergence, a
//! minimum support code) or an expectation of `Q` under a law. [`EvidenceStatus`]
//! states what the number is allowed to mean and keeps five cases apart.
//!
//! * **Exact.** An algebraic identity, or exhaustive evaluation over a stated
//!   finite family. The value carries the numerical error of its own evaluation.
//! * **Uniform bound.** `Q(w) <= upper` at every `w` of a stated region, with the
//!   numerical error already inside `upper`.
//! * **Statistical estimate.** An expectation under a stated law, with its
//!   standard error. It bounds no extremum. A stochastic-mask mean, an observed
//!   worst case and a certified bound are three different numbers (P10), so
//!   [`EvidenceStatus::upper_bound`] and [`EvidenceStatus::lower_bound`] return
//!   `None` for it.
//! * **Counterexample.** A witness at which `Q` exceeds a threshold by more than
//!   the numerical error of the witness value.
//! * **Unresolved.** A lower and an upper bound that have not met, the gap
//!   between them, and which side a witness attains.
//!
//! A result never returns a stronger status than it proved. Every variant carries
//! a [`Validated`] field that only this module can build, so each status passes
//! through a constructor. The constructors refuse non-finite values, negative
//! numerical or standard error, an empty exhaustive family, an inverted interval,
//! a witness for an underived side, and a counterexample that roundoff could
//! explain.
//!
//! # Supports as hitting sets
//!
//! Fix a decomposition with `C` components and an input-independent control
//! program `Theta : M -> P` with `Theta(1) = theta_*` over a declared mask domain
//! `M`. A support `S` keeps its components on while the rest range over the
//! domain. Its risk is `R(S) = sup { d(f_Theta(m), f_theta_*) : m in M, m_S = 1 }`,
//! and `S` is sufficient at the declared tolerance `eps` when `R(S) <= eps`.
//!
//! * **Upward closure (P7).** Clamping more controls shrinks the admissible set,
//!   so `R` is monotone under supersets, and a union of per-input sufficient
//!   supports is sufficient at each input's own tolerance. This holds for any
//!   control set containing the all-on value, which keeps every clamped set
//!   nonempty, and for any nonnegative discrepancy `d`. A product domain only gives
//!   "the complement controls are free" its meaning: on a tied domain, clamping `S`
//!   may clamp other controls too, so supports are drawn from the declared
//!   admissible family (group-closed supports are closed under union).
//! * **Hitting sets (P12).** A mask `m` is bad when `d > eps`, and it perturbs
//!   `A(m) = {c : m_c != 1}`. `S` is sufficient iff it hits `A(m)` for every bad
//!   mask, so redundancy is an OR constraint over components, not two scalar
//!   importances. A refutation with no witness mask still cuts: every bad mask
//!   admissible under `S` perturbs only components outside `S`, so the complement
//!   of `S` is an edge.
//! * **Minimum code.** When the support code depends only on the support size, as
//!   P18's enumerative subset code does, every superset of a hitting set is a
//!   hitting set. The minimum code over the known edges is then
//!   `min_{k >= tau} L(C, k)`, with `tau` the minimum hitting-set size from an
//!   exact branch and bound. Counterexample-guided refinement (CEGAR) alternates
//!   that solve with the declared separation oracle. The result is optimal for the
//!   chosen support code and the fixed decomposition only. When the oracle neither
//!   refutes nor certifies a candidate, the result is unresolved and reports its
//!   gap.
//! * **Conflict replay.** When the decomposition changes, each recorded bad mask
//!   is mapped into the new coordinates and re-evaluated by the new oracle. A mask
//!   with no representative is dropped, and a mask that is no longer bad is
//!   discarded, never kept on trust.

use std::cmp::Ordering;
use std::convert::Infallible;
use std::fmt;

/// Proof that an [`EvidenceStatus`] passed through its constructor. Only this
/// module can build one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Validated(());

/// Why an [`EvidenceStatus::Exact`] value is exact.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExactBasis {
    /// An algebraic identity over the whole domain, e.g. P8's support function
    /// `h(u) = sum_c max(0, u'v_c)`.
    Algebraic,
    /// Every one of the `cardinality` members of the finite family named by the
    /// domain was evaluated.
    Exhaustive { cardinality: u64 },
    /// A search over the finite family named by the domain closed: every member
    /// was evaluated or excluded by a proven bound, e.g. P12's branch and bound
    /// over the `2^C` supports of `C` components, a family too large to count in a
    /// `u64`.
    ClosedSearch,
}

/// Which extremum a reported number is, and so which side of an interval a
/// witness attains.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Extremum {
    /// `sup_{w in D} Q(w)`. A witness attains a lower bound; the upper bound needs
    /// a derivation (a Lipschitz covering, P9 convexity).
    Supremum,
    /// `inf_{w in D} Q(w)`. A witness attains an upper bound; the lower bound
    /// needs a derivation (a relaxation, a closed search tree).
    Infimum,
}

/// What a reported number about `Q` over the declared domain `D` is allowed to
/// mean. `W` is a witness point of the domain (a mask, a support, an input).
#[derive(Clone, Debug, PartialEq)]
pub enum EvidenceStatus<W, D> {
    /// The extremum is `value`, up to `numerical_error` from its evaluation.
    Exact {
        value: f64,
        numerical_error: f64,
        basis: ExactBasis,
        witness: Option<W>,
        domain: D,
        validated: Validated,
    },
    /// `Q(w) <= upper` for every `w` in `region`. `upper` already includes
    /// `numerical_error`, which is kept for audit.
    UniformBound {
        upper: f64,
        numerical_error: f64,
        region: D,
        validated: Validated,
    },
    /// An estimate of `E_law Q` with its standard error, from `samples` draws. It
    /// is a different number from any extremum over the domain.
    StatisticalEstimate {
        estimate: f64,
        standard_error: f64,
        samples: u64,
        law: D,
        validated: Validated,
    },
    /// `Q(witness) = value` and `value - numerical_error > threshold`, so the
    /// claim `sup_D Q <= threshold` is false.
    Counterexample {
        value: f64,
        numerical_error: f64,
        threshold: f64,
        witness: W,
        validated: Validated,
    },
    /// `lower <= extremum <= upper` over `domain`, each bound already widened by
    /// its numerical error. An underived side is infinite. For a supremum the
    /// witness attains `lower`; for an infimum it attains `upper`.
    Unresolved {
        lower: f64,
        upper: f64,
        extremum: Extremum,
        witness: Option<W>,
        domain: D,
        validated: Validated,
    },
}

/// Why a constructor refused to build an [`EvidenceStatus`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum EvidenceStatusError {
    /// A value that must be finite was NaN or infinite.
    NonFinite { field: &'static str, value: f64 },
    /// A numerical error or standard error was negative.
    Negative { field: &'static str, value: f64 },
    /// An exhaustive evaluation over a family with no members.
    EmptyExhaustiveFamily,
    /// A statistical estimate from no samples.
    NoSamples,
    /// An interval bound that is NaN, or infinite on the side that excludes every
    /// value (`lower = +inf` or `upper = -inf`).
    InvalidBound { field: &'static str, value: f64 },
    /// `lower > upper`.
    InvertedInterval { lower: f64, upper: f64 },
    /// A witness for the side of an interval that has no finite value.
    UnattainedWitness { extremum: Extremum },
    /// `value - numerical_error` does not exceed `threshold`, so roundoff could
    /// explain the violation.
    NotAViolation {
        value: f64,
        numerical_error: f64,
        threshold: f64,
    },
}

impl fmt::Display for EvidenceStatusError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFinite { field, value } => {
                write!(f, "evidence field {field} must be finite, got {value}")
            }
            Self::Negative { field, value } => {
                write!(f, "evidence field {field} must be nonnegative, got {value}")
            }
            Self::EmptyExhaustiveFamily => {
                write!(f, "an exhaustive evaluation needs a nonempty finite family")
            }
            Self::NoSamples => write!(f, "a statistical estimate needs at least one sample"),
            Self::InvalidBound { field, value } => {
                write!(f, "interval bound {field} = {value} is NaN or excludes every value")
            }
            Self::InvertedInterval { lower, upper } => {
                write!(f, "interval lower bound {lower} exceeds upper bound {upper}")
            }
            Self::UnattainedWitness { extremum } => {
                write!(f, "a witness of the {extremum:?} must attain a finite bound")
            }
            Self::NotAViolation {
                value,
                numerical_error,
                threshold,
            } => write!(
                f,
                "value {value} with numerical error {numerical_error} does not exceed threshold {threshold}"
            ),
        }
    }
}

impl std::error::Error for EvidenceStatusError {}

fn require_finite(field: &'static str, value: f64) -> Result<(), EvidenceStatusError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(EvidenceStatusError::NonFinite { field, value })
    }
}

fn require_nonnegative(field: &'static str, value: f64) -> Result<(), EvidenceStatusError> {
    require_finite(field, value)?;
    if value < 0.0 {
        Err(EvidenceStatusError::Negative { field, value })
    } else {
        Ok(())
    }
}

/// `value + numerical_error` rounded up, so it bounds the exact sum. Adding zero is
/// exact, so an exactly evaluated value is not widened.
fn rounded_up_sum(value: f64, numerical_error: f64) -> f64 {
    let sum = value + numerical_error;
    if numerical_error == 0.0 { sum } else { sum.next_up() }
}

/// `value - numerical_error` rounded down, so the exact difference bounds it.
fn rounded_down_difference(value: f64, numerical_error: f64) -> f64 {
    let difference = value - numerical_error;
    if numerical_error == 0.0 {
        difference
    } else {
        difference.next_down()
    }
}

impl<W, D> EvidenceStatus<W, D> {
    /// An exact extremum: an algebraic identity, or an exhaustive evaluation over
    /// the finite family `domain`.
    pub fn exact(
        value: f64,
        numerical_error: f64,
        basis: ExactBasis,
        witness: Option<W>,
        domain: D,
    ) -> Result<Self, EvidenceStatusError> {
        require_finite("value", value)?;
        require_nonnegative("numerical_error", numerical_error)?;
        if matches!(basis, ExactBasis::Exhaustive { cardinality: 0 }) {
            return Err(EvidenceStatusError::EmptyExhaustiveFamily);
        }
        Ok(Self::Exact {
            value,
            numerical_error,
            basis,
            witness,
            domain,
            validated: Validated(()),
        })
    }

    /// A bound `Q(w) <= upper` derived for every `w` in `region`, with
    /// `numerical_error` already included in `upper`.
    pub fn uniform_bound(
        upper: f64,
        numerical_error: f64,
        region: D,
    ) -> Result<Self, EvidenceStatusError> {
        require_finite("upper", upper)?;
        require_nonnegative("numerical_error", numerical_error)?;
        Ok(Self::UniformBound {
            upper,
            numerical_error,
            region,
            validated: Validated(()),
        })
    }

    /// An estimate of the expectation of `Q` under `law`, with its standard error.
    pub fn statistical_estimate(
        estimate: f64,
        standard_error: f64,
        samples: u64,
        law: D,
    ) -> Result<Self, EvidenceStatusError> {
        require_finite("estimate", estimate)?;
        require_nonnegative("standard_error", standard_error)?;
        if samples == 0 {
            return Err(EvidenceStatusError::NoSamples);
        }
        Ok(Self::StatisticalEstimate {
            estimate,
            standard_error,
            samples,
            law,
            validated: Validated(()),
        })
    }

    /// A witness refuting `sup_D Q <= threshold`. Refused unless the violation
    /// exceeds the numerical error of `value`.
    pub fn counterexample(
        value: f64,
        numerical_error: f64,
        threshold: f64,
        witness: W,
    ) -> Result<Self, EvidenceStatusError> {
        require_finite("value", value)?;
        require_nonnegative("numerical_error", numerical_error)?;
        require_finite("threshold", threshold)?;
        if rounded_down_difference(value, numerical_error) <= threshold {
            return Err(EvidenceStatusError::NotAViolation {
                value,
                numerical_error,
                threshold,
            });
        }
        Ok(Self::Counterexample {
            value,
            numerical_error,
            threshold,
            witness,
            validated: Validated(()),
        })
    }

    /// Bounds `lower <= extremum <= upper` that have not met. An underived side is
    /// infinite; a witness must attain a finite side (`lower` for a supremum,
    /// `upper` for an infimum).
    pub fn unresolved(
        lower: f64,
        upper: f64,
        extremum: Extremum,
        witness: Option<W>,
        domain: D,
    ) -> Result<Self, EvidenceStatusError> {
        if lower.is_nan() || lower == f64::INFINITY {
            return Err(EvidenceStatusError::InvalidBound {
                field: "lower",
                value: lower,
            });
        }
        if upper.is_nan() || upper == f64::NEG_INFINITY {
            return Err(EvidenceStatusError::InvalidBound {
                field: "upper",
                value: upper,
            });
        }
        if lower > upper {
            return Err(EvidenceStatusError::InvertedInterval { lower, upper });
        }
        let attained = match extremum {
            Extremum::Supremum => lower.is_finite(),
            Extremum::Infimum => upper.is_finite(),
        };
        if witness.is_some() && !attained {
            return Err(EvidenceStatusError::UnattainedWitness { extremum });
        }
        Ok(Self::Unresolved {
            lower,
            upper,
            extremum,
            witness,
            domain,
            validated: Validated(()),
        })
    }

    /// An upper bound on the reported extremum that this status proves, if any.
    pub fn upper_bound(&self) -> Option<f64> {
        match self {
            Self::Exact {
                value,
                numerical_error,
                ..
            } => Some(rounded_up_sum(*value, *numerical_error)),
            Self::UniformBound { upper, .. } => Some(*upper),
            Self::Unresolved { upper, .. } if upper.is_finite() => Some(*upper),
            Self::Unresolved { .. } | Self::StatisticalEstimate { .. } | Self::Counterexample { .. } => {
                None
            }
        }
    }

    /// A lower bound on the reported extremum that this status proves, if any. A
    /// counterexample bounds the supremum it refutes from below.
    pub fn lower_bound(&self) -> Option<f64> {
        match self {
            Self::Exact {
                value,
                numerical_error,
                ..
            }
            | Self::Counterexample {
                value,
                numerical_error,
                ..
            } => Some(rounded_down_difference(*value, *numerical_error)),
            Self::Unresolved { lower, .. } if lower.is_finite() => Some(*lower),
            Self::Unresolved { .. } | Self::UniformBound { .. } | Self::StatisticalEstimate { .. } => {
                None
            }
        }
    }

    /// `upper - lower` for an unresolved status, infinite when a side is underived.
    pub fn gap(&self) -> Option<f64> {
        match self {
            Self::Unresolved { lower, upper, .. } => Some(upper - lower),
            Self::Exact { .. }
            | Self::UniformBound { .. }
            | Self::StatisticalEstimate { .. }
            | Self::Counterexample { .. } => None,
        }
    }

    /// True when this status proves the extremum is at most `threshold`.
    pub fn certifies_at_most(&self, threshold: f64) -> bool {
        self.upper_bound().is_some_and(|upper| upper <= threshold)
    }

    /// True when this status proves the extremum exceeds `threshold`.
    pub fn refutes_at_most(&self, threshold: f64) -> bool {
        self.lower_bound().is_some_and(|lower| lower > threshold)
    }

    /// The witness the status carries, if any.
    pub fn witness(&self) -> Option<&W> {
        match self {
            Self::Exact { witness, .. } | Self::Unresolved { witness, .. } => witness.as_ref(),
            Self::Counterexample { witness, .. } => Some(witness),
            Self::UniformBound { .. } | Self::StatisticalEstimate { .. } => None,
        }
    }

    /// The status's witness, if any, taken by value.
    pub fn into_witness(self) -> Option<W> {
        match self {
            Self::Exact { witness, .. } | Self::Unresolved { witness, .. } => witness,
            Self::Counterexample { witness, .. } => Some(witness),
            Self::UniformBound { .. } | Self::StatisticalEstimate { .. } => None,
        }
    }

    /// The declared domain, region or law the status is stated over. A
    /// counterexample states only its witness.
    pub fn domain(&self) -> Option<&D> {
        match self {
            Self::Exact { domain, .. } | Self::Unresolved { domain, .. } => Some(domain),
            Self::UniformBound { region, .. } => Some(region),
            Self::StatisticalEstimate { law, .. } => Some(law),
            Self::Counterexample { .. } => None,
        }
    }

    /// The order of what two statuses about one quantity prove. `self` is stronger
    /// when the interval it proves, `[lower_bound or -inf, upper_bound or +inf]`,
    /// lies inside `other`'s: every threshold the weaker status certifies or refutes,
    /// the stronger one certifies or refutes too. The kind of evidence does not
    /// order anything by itself: an exact value with a wide numerical error proves
    /// less about `<= t` than a tight uniform bound at `t`. Statuses over different
    /// domains, about different extrema, or involving a statistical estimate (a
    /// different number) or a counterexample (which states no domain) compare as
    /// `None`, incomparable, as do intervals neither of which contains the other.
    pub fn compare_strength(&self, other: &Self) -> Option<Ordering>
    where
        D: PartialEq,
    {
        if matches!(self, Self::StatisticalEstimate { .. })
            || matches!(other, Self::StatisticalEstimate { .. })
        {
            return None;
        }
        let (Some(domain), Some(other_domain)) = (self.domain(), other.domain()) else {
            return None;
        };
        if domain != other_domain {
            return None;
        }
        if let (Some(extremum), Some(other_extremum)) =
            (self.bounded_extremum(), other.bounded_extremum())
            && extremum != other_extremum
        {
            return None;
        }
        let (lower, upper) = self.proven_interval();
        let (other_lower, other_upper) = other.proven_interval();
        let inside = lower >= other_lower && upper <= other_upper;
        let contains = lower <= other_lower && upper >= other_upper;
        match (inside, contains) {
            (true, true) => Some(Ordering::Equal),
            (true, false) => Some(Ordering::Greater),
            (false, true) => Some(Ordering::Less),
            (false, false) => None,
        }
    }

    /// The interval this status proves the extremum lies in, with an unproved side
    /// infinite.
    fn proven_interval(&self) -> (f64, f64) {
        (
            self.lower_bound().unwrap_or(f64::NEG_INFINITY),
            self.upper_bound().unwrap_or(f64::INFINITY),
        )
    }

    /// The extremum a status names: a uniform bound and a counterexample are about a
    /// supremum, an unresolved bracket states its own, and an exact value is either.
    fn bounded_extremum(&self) -> Option<Extremum> {
        match self {
            Self::UniformBound { .. } | Self::Counterexample { .. } => Some(Extremum::Supremum),
            Self::Unresolved { extremum, .. } => Some(*extremum),
            Self::Exact { .. } | Self::StatisticalEstimate { .. } => None,
        }
    }
}

/// Why a component set or a failure hypergraph was refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HypergraphError {
    /// A component index outside `0..components`.
    ComponentOutOfRange { index: usize, components: usize },
    /// Two sets, or a set and a hypergraph or oracle, over different component
    /// counts.
    ComponentCountMismatch { expected: usize, found: usize },
    /// An edge that perturbs no component: no support can hit it.
    EmptyEdge,
}

impl fmt::Display for HypergraphError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ComponentOutOfRange { index, components } => {
                write!(f, "component {index} is outside 0..{components}")
            }
            Self::ComponentCountMismatch { expected, found } => {
                write!(f, "expected {expected} components, found {found}")
            }
            Self::EmptyEdge => write!(f, "a failure edge must perturb at least one component"),
        }
    }
}

impl std::error::Error for HypergraphError {}

fn require_same_components(expected: usize, found: usize) -> Result<(), HypergraphError> {
    if expected == found {
        Ok(())
    } else {
        Err(HypergraphError::ComponentCountMismatch { expected, found })
    }
}

/// Component indices out of `0..components`, sorted and without repeats.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ComponentSet {
    components: usize,
    members: Vec<usize>,
}

impl ComponentSet {
    /// Sorts `members` and removes repeats; refuses an index outside
    /// `0..components`.
    pub fn new(components: usize, mut members: Vec<usize>) -> Result<Self, HypergraphError> {
        members.sort_unstable();
        members.dedup();
        if let Some(&index) = members.last()
            && index >= components
        {
            return Err(HypergraphError::ComponentOutOfRange { index, components });
        }
        Ok(Self {
            components,
            members,
        })
    }

    /// Every component: the all-on support, sufficient because
    /// `Theta(1) = theta_*`.
    pub fn all(components: usize) -> Self {
        Self {
            components,
            members: (0..components).collect(),
        }
    }

    /// The number of components `C` the set is drawn from.
    pub fn components(&self) -> usize {
        self.components
    }

    /// The member indices, sorted.
    pub fn members(&self) -> &[usize] {
        &self.members
    }

    /// The number of members.
    pub fn len(&self) -> usize {
        self.members.len()
    }

    /// True when the set has no members.
    pub fn is_empty(&self) -> bool {
        self.members.is_empty()
    }

    /// True when the two sets share a member.
    pub fn intersects(&self, other: &Self) -> bool {
        let (mut left, mut right) = (0, 0);
        while left < self.members.len() && right < other.members.len() {
            match self.members[left].cmp(&other.members[right]) {
                Ordering::Less => left += 1,
                Ordering::Greater => right += 1,
                Ordering::Equal => return true,
            }
        }
        false
    }

    /// True when every member of `self` is a member of `other`.
    pub fn is_subset_of(&self, other: &Self) -> bool {
        self.members
            .iter()
            .all(|member| other.members.binary_search(member).is_ok())
    }

    /// The union of two sets over the same components.
    pub fn union(&self, other: &Self) -> Result<Self, HypergraphError> {
        require_same_components(self.components, other.components)?;
        let mut members = self.members.clone();
        members.extend_from_slice(&other.members);
        Self::new(self.components, members)
    }

    /// The components outside the set.
    pub fn complement(&self) -> Self {
        Self {
            components: self.components,
            members: (0..self.components)
                .filter(|index| self.members.binary_search(index).is_err())
                .collect(),
        }
    }

    /// The set grown to `size` members by adding the smallest absent indices.
    fn extended_to(&self, size: usize) -> Self {
        let mut members = self.members.clone();
        for index in 0..self.components {
            if members.len() >= size {
                break;
            }
            if self.members.binary_search(&index).is_err() {
                members.push(index);
            }
        }
        members.sort_unstable();
        Self {
            components: self.components,
            members,
        }
    }
}

/// A recorded failure: the components a bad mask perturbs, and the mask.
#[derive(Clone, Debug, PartialEq)]
pub struct FailureEdge<M> {
    /// `A(m) = {c : m_c != 1}`, or the complement of a support refuted without a
    /// witness.
    pub perturbed: ComponentSet,
    /// The bad mask, or `None` for a complement cut.
    pub witness: Option<M>,
}

/// The inclusion-minimal failure edges recorded for a fixed decomposition. A
/// support is sufficient only if it hits every edge.
#[derive(Clone, Debug, PartialEq)]
pub struct FailureHypergraph<M> {
    components: usize,
    edges: Vec<FailureEdge<M>>,
}

impl<M> FailureHypergraph<M> {
    /// A hypergraph over `components` components with no edges.
    pub fn new(components: usize) -> Self {
        Self {
            components,
            edges: Vec::new(),
        }
    }

    /// The number of components `C`.
    pub fn components(&self) -> usize {
        self.components
    }

    /// The recorded inclusion-minimal edges.
    pub fn edges(&self) -> &[FailureEdge<M>] {
        &self.edges
    }

    /// Records an edge and returns whether it was kept. A recorded subset makes it
    /// redundant, since hitting the subset hits it; it makes recorded supersets
    /// redundant in turn.
    pub fn insert(&mut self, edge: FailureEdge<M>) -> Result<bool, HypergraphError> {
        require_same_components(self.components, edge.perturbed.components())?;
        if edge.perturbed.is_empty() {
            return Err(HypergraphError::EmptyEdge);
        }
        if self
            .edges
            .iter()
            .any(|recorded| recorded.perturbed.is_subset_of(&edge.perturbed))
        {
            return Ok(false);
        }
        self.edges
            .retain(|recorded| !edge.perturbed.is_subset_of(&recorded.perturbed));
        self.edges.push(edge);
        Ok(true)
    }

    /// A minimum-cardinality hitting set of the recorded edges, by exact branch and
    /// bound. With no edges it is the empty set.
    pub fn minimum_hitting_set(&self) -> ComponentSet {
        let edges: Vec<&[usize]> = self
            .edges
            .iter()
            .map(|edge| edge.perturbed.members())
            .collect();
        let mut search = HittingSetSearch {
            best: greedy_hitting_set(&edges, self.components),
            edges,
            chosen: vec![false; self.components],
            forbidden: vec![false; self.components],
            picked: Vec::new(),
        };
        search.branch();
        search.best.sort_unstable();
        ComponentSet {
            components: self.components,
            members: search.best,
        }
    }
}

/// A hitting set built by repeatedly taking the component that hits the most
/// unhit edges. It seeds the branch and bound with an incumbent.
fn greedy_hitting_set(edges: &[&[usize]], components: usize) -> Vec<usize> {
    let mut hit = vec![false; edges.len()];
    let mut picked = Vec::new();
    loop {
        let mut counts = vec![0usize; components];
        for (edge, done) in edges.iter().zip(&hit) {
            if !*done {
                for &component in edge.iter() {
                    counts[component] += 1;
                }
            }
        }
        match (0..components).max_by_key(|&component| counts[component]) {
            Some(component) if counts[component] > 0 => {
                picked.push(component);
                for (edge, done) in edges.iter().zip(hit.iter_mut()) {
                    if edge.contains(&component) {
                        *done = true;
                    }
                }
            }
            Some(..) | None => return picked,
        }
    }
}

/// The number of pairwise disjoint edges a greedy packing finds. Disjoint edges
/// need distinct components, so it bounds the components still needed from below.
fn disjoint_packing(edges: &[Vec<usize>], components: usize) -> usize {
    let mut order: Vec<usize> = (0..edges.len()).collect();
    order.sort_by_key(|&index| edges[index].len());
    let mut used = vec![false; components];
    let mut count = 0;
    for index in order {
        if edges[index].iter().all(|&component| !used[component]) {
            for &component in &edges[index] {
                used[component] = true;
            }
            count += 1;
        }
    }
    count
}

/// Branch and bound for a minimum hitting set. Every hitting set contains a
/// component of any unhit edge, so branching on the components of one unhit edge,
/// and forbidding each after its branch, visits every minimal candidate once; a
/// branch whose picked count plus the packing bound cannot beat the incumbent is
/// pruned.
struct HittingSetSearch<'a> {
    edges: Vec<&'a [usize]>,
    chosen: Vec<bool>,
    forbidden: Vec<bool>,
    picked: Vec<usize>,
    best: Vec<usize>,
}

impl HittingSetSearch<'_> {
    fn branch(&mut self) {
        let mut unhit: Vec<Vec<usize>> = Vec::new();
        for edge in &self.edges {
            if edge.iter().any(|&component| self.chosen[component]) {
                continue;
            }
            let allowed: Vec<usize> = edge
                .iter()
                .copied()
                .filter(|&component| !self.forbidden[component])
                .collect();
            if allowed.is_empty() {
                return;
            }
            unhit.push(allowed);
        }
        if unhit.is_empty() {
            if self.picked.len() < self.best.len() {
                self.best = self.picked.clone();
            }
            return;
        }
        if self.picked.len() + disjoint_packing(&unhit, self.chosen.len()) >= self.best.len() {
            return;
        }
        let mut pivot = 0;
        for (index, edge) in unhit.iter().enumerate() {
            if edge.len() < unhit[pivot].len() {
                pivot = index;
            }
        }
        let pivot = std::mem::take(&mut unhit[pivot]);
        for &component in &pivot {
            self.chosen[component] = true;
            self.picked.push(component);
            self.branch();
            self.picked.pop();
            self.chosen[component] = false;
            self.forbidden[component] = true;
        }
        for &component in &pivot {
            self.forbidden[component] = false;
        }
    }
}

/// A support code whose length depends only on how many of the `C` components a
/// support keeps, like P18's enumerative subset code
/// `L_int(k + 1) + ceil(log2 binom(C, k))`.
pub trait CardinalityCode {
    /// Why the code has no codeword for a support size.
    type Error;

    /// The length in bits of a support keeping `size` of `components` components.
    fn support_bits(&self, components: usize, size: usize) -> Result<u64, Self::Error>;
}

/// The cheapest support size at or above `least`, the smallest on ties, and its
/// length in bits.
fn cheapest_size<K: CardinalityCode + ?Sized>(
    code: &K,
    components: usize,
    least: usize,
) -> Result<(usize, u64), K::Error> {
    let mut best = (least, code.support_bits(components, least)?);
    for size in least + 1..=components {
        let bits = code.support_bits(components, size)?;
        if bits < best.1 {
            best = (size, bits);
        }
    }
    Ok(best)
}

/// The declared separation oracle of a fixed decomposition, at one input or over a
/// declared family of inputs.
pub trait SeparationOracle {
    /// A mask of the declared domain.
    type Mask;
    /// The declared domain the oracle's evidence is stated over.
    type Domain;
    /// Why an evaluation failed.
    type Error;

    /// The number of components `C`.
    fn components(&self) -> usize;

    /// `A(m) = {c : m_c != 1}`, the components `mask` perturbs.
    fn perturbed_components(&self, mask: &Self::Mask) -> Vec<usize>;

    /// Evidence about the risk
    /// `R(S) = sup { d(f_Theta(m), f_theta_*) : m in M, m_S = 1 }` of `support`.
    /// A refuting witness must keep every component of `support` on.
    fn separate(
        &mut self,
        support: &ComponentSet,
    ) -> Result<EvidenceStatus<Self::Mask, Self::Domain>, Self::Error>;

    /// Evidence about `d(f_Theta(m), f_theta_*)` at the single mask `mask`.
    fn evaluate(
        &mut self,
        mask: &Self::Mask,
    ) -> Result<EvidenceStatus<Self::Mask, Self::Domain>, Self::Error>;
}

/// Why a support search, a union or a replay was refused. `E` is the oracle's
/// error and `C` the support code's.
#[derive(Debug)]
pub enum SupportSearchError<E, C = Infallible> {
    /// A component set or hypergraph was refused.
    Hypergraph(HypergraphError),
    /// A status could not be built.
    Evidence(EvidenceStatusError),
    /// The declared tolerance is negative or not finite.
    InvalidTolerance { tolerance: f64 },
    /// A refutation perturbs no component, so the all-on setting itself violates
    /// the tolerance.
    AllOnViolation,
    /// A refuting witness perturbs a component of the support it refutes, so it is
    /// not admissible under that support.
    InadmissibleWitness {
        support: ComponentSet,
        perturbed: ComponentSet,
    },
    /// A per-input support whose evidence does not certify it at its own
    /// tolerance.
    InsufficientInputSupport { input: usize },
    /// The oracle failed.
    Oracle(E),
    /// The support code has no codeword for a support size.
    Code(C),
}

impl<E: fmt::Display, C: fmt::Display> fmt::Display for SupportSearchError<E, C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Hypergraph(error) => write!(f, "{error}"),
            Self::Evidence(error) => write!(f, "{error}"),
            Self::InvalidTolerance { tolerance } => {
                write!(f, "the declared tolerance must be finite and nonnegative, got {tolerance}")
            }
            Self::AllOnViolation => {
                write!(f, "a refutation perturbs no component: the all-on setting violates the tolerance")
            }
            Self::InadmissibleWitness { support, perturbed } => write!(
                f,
                "witness perturbs {:?}, which meets the refuted support {:?}",
                perturbed.members(),
                support.members()
            ),
            Self::InsufficientInputSupport { input } => {
                write!(f, "the support of input {input} is not certified at its tolerance")
            }
            Self::Oracle(error) => write!(f, "separation oracle failed: {error}"),
            Self::Code(error) => write!(f, "support code failed: {error}"),
        }
    }
}

impl<E: fmt::Debug + fmt::Display, C: fmt::Debug + fmt::Display> std::error::Error
    for SupportSearchError<E, C>
{
}

impl<E, C> From<HypergraphError> for SupportSearchError<E, C> {
    fn from(error: HypergraphError) -> Self {
        Self::Hypergraph(error)
    }
}

impl<E, C> From<EvidenceStatusError> for SupportSearchError<E, C> {
    fn from(error: EvidenceStatusError) -> Self {
        Self::Evidence(error)
    }
}

fn require_tolerance<E, C>(tolerance: f64) -> Result<(), SupportSearchError<E, C>> {
    if tolerance.is_finite() && tolerance >= 0.0 {
        Ok(())
    } else {
        Err(SupportSearchError::InvalidTolerance { tolerance })
    }
}

/// The family a support search reports over: the supports of `components`
/// components, at the declared `tolerance`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SupportDomain {
    pub components: usize,
    pub tolerance: f64,
}

/// A support with the oracle's evidence about its risk.
#[derive(Clone, Debug, PartialEq)]
pub struct SupportEvidence<M, D> {
    pub support: ComponentSet,
    pub evidence: EvidenceStatus<M, D>,
}

/// The outcome of counterexample-guided support search.
#[derive(Clone, Debug, PartialEq)]
pub struct SupportSearch<M, D> {
    /// The minimum code in bits over sufficient supports, for the chosen support
    /// code and the fixed decomposition only. Exact when the search closed;
    /// otherwise unresolved, with the certified support attaining `upper` as its
    /// witness.
    pub code: EvidenceStatus<ComponentSet, SupportDomain>,
    /// The certified support with the smallest code found, and its evidence.
    pub certified: Option<SupportEvidence<M, D>>,
    /// The minimum-code candidate the oracle neither refuted nor certified, when
    /// the search did not close.
    pub undecided: Option<SupportEvidence<M, D>>,
    /// The failure hypergraph at termination, to replay after the decomposition
    /// changes.
    pub hypergraph: FailureHypergraph<M>,
    /// The oracle separations performed.
    pub separations: usize,
}

/// CEGAR for a minimum-code sufficient support (P12), starting from `hypergraph`
/// (empty, or replayed after a decomposition change).
///
/// Each round solves the exact minimum hitting set, takes the cheapest size at or
/// above it, and asks the oracle about the candidate. A certificate closes the
/// search exactly. A refutation records the witness's perturbed set, or the
/// candidate's complement when there is no witness. That edge misses the candidate
/// while every recorded edge hits it, so each round marks a new support unsafe and
/// the loop ends after at most `2^C` rounds, with no iteration cap. A candidate
/// that is neither refuted nor certified ends the search unresolved: the lower
/// bound is the candidate's code, and the upper bound is the full support's code
/// when the oracle certifies the full support.
pub fn minimum_code_support<O, K>(
    oracle: &mut O,
    code: &K,
    tolerance: f64,
    mut hypergraph: FailureHypergraph<O::Mask>,
) -> Result<SupportSearch<O::Mask, O::Domain>, SupportSearchError<O::Error, K::Error>>
where
    O: SeparationOracle,
    K: CardinalityCode + ?Sized,
{
    require_tolerance::<O::Error, K::Error>(tolerance)?;
    let components = oracle.components();
    require_same_components(components, hypergraph.components())?;
    let domain = SupportDomain {
        components,
        tolerance,
    };
    let mut separations = 0;
    loop {
        let least = hypergraph.minimum_hitting_set();
        let (size, bits) =
            cheapest_size(code, components, least.len()).map_err(SupportSearchError::Code)?;
        let candidate = least.extended_to(size);
        let evidence = oracle
            .separate(&candidate)
            .map_err(SupportSearchError::Oracle)?;
        separations += 1;
        if evidence.certifies_at_most(tolerance) {
            // Integer code lengths convert to f64 exactly below 2^53 bits.
            let minimum_code = EvidenceStatus::exact(
                bits as f64,
                0.0,
                ExactBasis::ClosedSearch,
                Some(candidate.clone()),
                domain,
            )?;
            return Ok(SupportSearch {
                code: minimum_code,
                certified: Some(SupportEvidence {
                    support: candidate,
                    evidence,
                }),
                undecided: None,
                hypergraph,
                separations,
            });
        }
        if !evidence.refutes_at_most(tolerance) {
            let all = ComponentSet::all(components);
            let certified = if candidate == all {
                None
            } else {
                let full = oracle.separate(&all).map_err(SupportSearchError::Oracle)?;
                separations += 1;
                full.certifies_at_most(tolerance).then_some(SupportEvidence {
                    support: all,
                    evidence: full,
                })
            };
            let upper = if certified.is_some() {
                code.support_bits(components, components)
                    .map_err(SupportSearchError::Code)? as f64
            } else {
                f64::INFINITY
            };
            let minimum_code = EvidenceStatus::unresolved(
                bits as f64,
                upper,
                Extremum::Infimum,
                certified.as_ref().map(|found| found.support.clone()),
                domain,
            )?;
            return Ok(SupportSearch {
                code: minimum_code,
                certified,
                undecided: Some(SupportEvidence {
                    support: candidate,
                    evidence,
                }),
                hypergraph,
                separations,
            });
        }
        let perturbed = match evidence.witness() {
            Some(mask) => {
                let perturbed = ComponentSet::new(components, oracle.perturbed_components(mask))?;
                if perturbed.intersects(&candidate) {
                    return Err(SupportSearchError::InadmissibleWitness {
                        support: candidate,
                        perturbed,
                    });
                }
                perturbed
            }
            None => candidate.complement(),
        };
        if perturbed.is_empty() {
            return Err(SupportSearchError::AllOnViolation);
        }
        hypergraph.insert(FailureEdge {
            perturbed,
            witness: evidence.into_witness(),
        })?;
    }
}

/// A support certified sufficient at one input, at that input's own tolerance.
#[derive(Clone, Debug, PartialEq)]
pub struct InputSupport<M, D> {
    pub support: ComponentSet,
    pub tolerance: f64,
    pub evidence: EvidenceStatus<M, D>,
}

/// The union of per-input sufficient supports and the bound each input inherits.
#[derive(Clone, Debug, PartialEq)]
pub struct SufficientUnion<M, D> {
    pub support: ComponentSet,
    /// Per input, in input order, `R_x(union) <= R_x(S_x) <= upper`: a uniform
    /// bound over the region the input's evidence was stated over, which contains
    /// the union's clamped set.
    pub per_input: Vec<EvidenceStatus<M, D>>,
}

/// P7: the union of per-input sufficient supports is sufficient at each input's
/// own tolerance, because clamping more controls shrinks the admissible set and
/// the all-on value keeps it nonempty. Each input's evidence must certify its
/// support at its tolerance. No oracle is consulted, so the oracle error is
/// `Infallible`.
pub fn sufficient_union<M, D: Clone>(
    components: usize,
    per_input: &[InputSupport<M, D>],
) -> Result<SufficientUnion<M, D>, SupportSearchError<Infallible>> {
    let mut support = ComponentSet::new(components, Vec::new())?;
    let mut inherited = Vec::with_capacity(per_input.len());
    for (input, sufficient) in per_input.iter().enumerate() {
        require_tolerance::<Infallible, Infallible>(sufficient.tolerance)?;
        support = support.union(&sufficient.support)?;
        let Some(upper) = sufficient
            .evidence
            .upper_bound()
            .filter(|upper| *upper <= sufficient.tolerance)
        else {
            return Err(SupportSearchError::InsufficientInputSupport { input });
        };
        let Some(region) = sufficient.evidence.domain() else {
            return Err(SupportSearchError::InsufficientInputSupport { input });
        };
        let numerical_error = match &sufficient.evidence {
            EvidenceStatus::Exact {
                numerical_error, ..
            }
            | EvidenceStatus::UniformBound {
                numerical_error, ..
            } => *numerical_error,
            EvidenceStatus::StatisticalEstimate { .. }
            | EvidenceStatus::Counterexample { .. }
            | EvidenceStatus::Unresolved { .. } => 0.0,
        };
        inherited.push(EvidenceStatus::uniform_bound(
            upper,
            numerical_error,
            region.clone(),
        )?);
    }
    Ok(SufficientUnion {
        support,
        per_input: inherited,
    })
}

/// The outcome of replaying recorded conflicts on a changed decomposition.
#[derive(Clone, Debug, PartialEq)]
pub struct ConflictReplay<M> {
    /// The replayed conflicts the new oracle still refutes, over the new
    /// components.
    pub hypergraph: FailureHypergraph<M>,
    /// Recorded conflicts whose mask has no representative in the new
    /// decomposition.
    pub unrepresentable: usize,
    /// Recorded conflicts the new oracle no longer refutes at the tolerance.
    pub no_longer_bad: usize,
    /// Complement cuts, which record no mask and so cannot be replayed.
    pub without_witness: usize,
}

/// Replays the conflicts of `recorded` after the decomposition changes.
/// `represent` maps an old mask to a mask of the new decomposition that sets the
/// same parameters (for a refinement, each piece copies its parent's mask value),
/// or `None` when no mask does. Every mapped mask is re-evaluated by the new oracle;
/// none is kept on trust.
pub fn replay_conflicts<O, OldMask, F>(
    recorded: &FailureHypergraph<OldMask>,
    mut represent: F,
    oracle: &mut O,
    tolerance: f64,
) -> Result<ConflictReplay<O::Mask>, SupportSearchError<O::Error>>
where
    O: SeparationOracle,
    F: FnMut(&OldMask) -> Option<O::Mask>,
{
    require_tolerance::<O::Error, Infallible>(tolerance)?;
    let components = oracle.components();
    let mut replay = ConflictReplay {
        hypergraph: FailureHypergraph::new(components),
        unrepresentable: 0,
        no_longer_bad: 0,
        without_witness: 0,
    };
    for edge in recorded.edges() {
        let Some(old_mask) = &edge.witness else {
            replay.without_witness += 1;
            continue;
        };
        let Some(mask) = represent(old_mask) else {
            replay.unrepresentable += 1;
            continue;
        };
        let evidence = oracle.evaluate(&mask).map_err(SupportSearchError::Oracle)?;
        if !evidence.refutes_at_most(tolerance) {
            replay.no_longer_bad += 1;
            continue;
        }
        let perturbed = ComponentSet::new(components, oracle.perturbed_components(&mask))?;
        if perturbed.is_empty() {
            return Err(SupportSearchError::AllOnViolation);
        }
        replay.hypergraph.insert(FailureEdge {
            perturbed,
            witness: Some(mask),
        })?;
    }
    Ok(replay)
}

#[cfg(test)]
mod tests {
    use super::{
        CardinalityCode, ComponentSet, ConflictReplay, EvidenceStatus, EvidenceStatusError,
        ExactBasis, Extremum, FailureEdge, FailureHypergraph, InputSupport, SeparationOracle,
        SupportSearchError, greedy_hitting_set, minimum_code_support, replay_conflicts,
        sufficient_union,
    };
    use std::cmp::Ordering;
    use std::convert::Infallible;

    type Status = EvidenceStatus<Vec<usize>, &'static str>;

    #[test]
    fn a_statistical_estimate_never_bounds_the_extremum_while_the_same_number_as_a_bound_does() {
        // P10: n pieces of a cancelling pair +-a with iid U[0,1] masks give
        // E E_n^2 = a^2 / (6n), a mean, while sup |E_n| = |a| is the extremum.
        let a = 1.0_f64;
        let pieces = 8.0_f64;
        let mean = a * a / (6.0 * pieces);
        let estimate = Status::statistical_estimate(mean, 0.01, 1000, "iid U[0,1] masks")
            .expect("a finite estimate with a nonnegative standard error");
        assert_eq!(estimate.upper_bound(), None);
        assert_eq!(estimate.lower_bound(), None);
        assert!(!estimate.certifies_at_most(mean));
        assert!(!estimate.certifies_at_most(f64::MAX));
        assert!(!estimate.refutes_at_most(f64::MIN));
        // Positive control: the same number as a derived bound does certify.
        let bound = Status::uniform_bound(mean, 0.0, "[0,1]^C").expect("a finite bound");
        assert!(bound.certifies_at_most(mean));
        assert!(!bound.certifies_at_most(mean.next_down()));
    }

    #[test]
    fn an_exact_value_brackets_itself_by_its_numerical_error_and_an_exact_integer_is_not_widened() {
        let code = Status::exact(
            5.0,
            0.0,
            ExactBasis::Exhaustive { cardinality: 32 },
            Some(vec![1, 2]),
            "subsets of five components",
        )
        .expect("an exhaustive exact value");
        assert_eq!(code.upper_bound(), Some(5.0));
        assert_eq!(code.lower_bound(), Some(5.0));
        assert!(code.certifies_at_most(5.0));
        assert!(!code.certifies_at_most(4.0));
        assert!(code.refutes_at_most(4.0));
        assert!(!code.refutes_at_most(5.0));
        assert_eq!(code.gap(), None);

        let numerical_error = 1e-15;
        let evaluated = Status::exact(1.0, numerical_error, ExactBasis::Algebraic, None, "all u")
            .expect("an algebraic exact value");
        let upper = evaluated.upper_bound().expect("an exact value bounds itself");
        let lower = evaluated.lower_bound().expect("an exact value bounds itself");
        assert!(upper > 1.0 + numerical_error);
        assert!(lower < 1.0 - numerical_error);
        assert!(!evaluated.certifies_at_most(1.0));
    }

    #[test]
    fn constructors_refuse_inputs_that_do_not_support_the_status() {
        assert!(matches!(
            Status::uniform_bound(f64::NAN, 0.0, "r"),
            Err(EvidenceStatusError::NonFinite { field: "upper", .. })
        ));
        assert!(matches!(
            Status::uniform_bound(f64::INFINITY, 0.0, "r"),
            Err(EvidenceStatusError::NonFinite { field: "upper", .. })
        ));
        assert!(matches!(
            Status::uniform_bound(1.0, -1e-16, "r"),
            Err(EvidenceStatusError::Negative {
                field: "numerical_error",
                ..
            })
        ));
        assert!(matches!(
            Status::exact(1.0, 0.0, ExactBasis::Exhaustive { cardinality: 0 }, None, "empty"),
            Err(EvidenceStatusError::EmptyExhaustiveFamily)
        ));
        assert!(matches!(
            Status::statistical_estimate(0.5, -0.1, 10, "law"),
            Err(EvidenceStatusError::Negative {
                field: "standard_error",
                ..
            })
        ));
        assert!(matches!(
            Status::statistical_estimate(0.5, 0.1, 0, "law"),
            Err(EvidenceStatusError::NoSamples)
        ));
        // Positive controls: the same constructors accept supported inputs.
        assert!(Status::uniform_bound(1.0, 0.0, "r").is_ok());
        assert!(
            Status::exact(1.0, 0.0, ExactBasis::Exhaustive { cardinality: 1 }, None, "one").is_ok()
        );
        assert!(Status::exact(2.0, 0.0, ExactBasis::ClosedSearch, Some(vec![0, 1]), "2^C").is_ok());
        assert!(Status::statistical_estimate(0.5, 0.1, 10, "law").is_ok());
    }

    #[test]
    fn a_violation_within_its_numerical_error_is_not_a_counterexample() {
        let threshold = 0.25;
        let numerical_error = 1e-12;
        assert!(matches!(
            Status::counterexample(threshold + numerical_error, numerical_error, threshold, vec![0, 1]),
            Err(EvidenceStatusError::NotAViolation { .. })
        ));
        assert!(matches!(
            Status::counterexample(threshold, 0.0, threshold, vec![0, 1]),
            Err(EvidenceStatusError::NotAViolation { .. })
        ));
        // Positive control: a violation beyond the numerical error is accepted.
        let found = Status::counterexample(
            threshold + 4.0 * numerical_error,
            numerical_error,
            threshold,
            vec![0, 1],
        )
        .expect("a violation beyond roundoff");
        assert!(found.refutes_at_most(threshold));
        assert_eq!(found.upper_bound(), None);
        assert!(!found.certifies_at_most(f64::MAX));
    }

    #[test]
    fn an_unresolved_status_reports_its_gap_and_refuses_an_inverted_or_unattained_interval() {
        let witnessed = Status::unresolved(
            0.3,
            f64::INFINITY,
            Extremum::Supremum,
            Some(vec![1, 0, 1]),
            "[0,1]^3",
        )
        .expect("a lower witness with no derived upper bound");
        assert_eq!(witnessed.gap(), Some(f64::INFINITY));
        assert_eq!(witnessed.upper_bound(), None);
        assert_eq!(witnessed.lower_bound(), Some(0.3));
        assert!(!witnessed.certifies_at_most(f64::MAX));

        let bracketed =
            Status::unresolved(2.0, 3.0, Extremum::Infimum, Some(vec![0, 2, 4]), "supports")
                .expect("a relaxation bound below an incumbent");
        assert_eq!(bracketed.gap(), Some(1.0));
        assert!(bracketed.certifies_at_most(3.0));
        assert!(!bracketed.certifies_at_most(2.5));
        assert!(bracketed.refutes_at_most(1.5));
        assert!(!bracketed.refutes_at_most(2.0));

        assert!(matches!(
            Status::unresolved(3.0, 2.0, Extremum::Supremum, None, "d"),
            Err(EvidenceStatusError::InvertedInterval { .. })
        ));
        assert!(matches!(
            Status::unresolved(f64::NEG_INFINITY, 3.0, Extremum::Supremum, Some(vec![1]), "d"),
            Err(EvidenceStatusError::UnattainedWitness {
                extremum: Extremum::Supremum
            })
        ));
        assert!(matches!(
            Status::unresolved(2.0, f64::INFINITY, Extremum::Infimum, Some(vec![1]), "d"),
            Err(EvidenceStatusError::UnattainedWitness {
                extremum: Extremum::Infimum
            })
        ));
        assert!(matches!(
            Status::unresolved(f64::INFINITY, f64::INFINITY, Extremum::Supremum, None, "d"),
            Err(EvidenceStatusError::InvalidBound { field: "lower", .. })
        ));
        assert!(matches!(
            Status::unresolved(f64::NAN, 1.0, Extremum::Supremum, None, "d"),
            Err(EvidenceStatusError::InvalidBound { field: "lower", .. })
        ));
        // Positive control: an interval with both sides underived and no witness is
        // a valid, uninformative status.
        let unknown = Status::unresolved(f64::NEG_INFINITY, f64::INFINITY, Extremum::Supremum, None, "d")
            .expect("an uninformative interval");
        assert_eq!(unknown.gap(), Some(f64::INFINITY));
        assert!(!unknown.certifies_at_most(f64::MAX));
        assert!(!unknown.refutes_at_most(f64::MIN));
    }

    #[test]
    fn a_status_hands_back_its_witness_and_domain_and_a_counterexample_states_no_domain() {
        let exact = Status::exact(1.0, 0.0, ExactBasis::Algebraic, Some(vec![3]), "all u")
            .expect("an exact value");
        assert_eq!(exact.witness(), Some(&vec![3]));
        assert_eq!(exact.domain(), Some(&"all u"));
        assert_eq!(exact.into_witness(), Some(vec![3]));
        let found = Status::counterexample(1.0, 0.0, 0.5, vec![4]).expect("a violation");
        assert_eq!(found.domain(), None);
        assert_eq!(found.into_witness(), Some(vec![4]));
        let bound = Status::uniform_bound(1.0, 0.0, "r").expect("a finite bound");
        assert_eq!(bound.witness(), None);
        assert_eq!(bound.domain(), Some(&"r"));
    }

    /// Exhaustive separation over a declared finite grid of mask levels. The
    /// fixtures' responses are polynomials in dyadic levels with few bits, so every
    /// value is exact in f64 and the numerical error is zero.
    struct GridOracle<F> {
        components: usize,
        levels: Vec<f64>,
        response: F,
    }

    impl<F: Fn(&[f64]) -> f64> GridOracle<F> {
        fn distance(&self, mask: &[f64]) -> f64 {
            ((self.response)(mask) - (self.response)(&vec![1.0; self.components])).abs()
        }
    }

    fn perturbed(mask: &[f64]) -> Vec<usize> {
        mask.iter()
            .enumerate()
            .filter_map(|(index, level)| (*level != 1.0).then_some(index))
            .collect()
    }

    impl<F: Fn(&[f64]) -> f64> SeparationOracle for GridOracle<F> {
        type Mask = Vec<f64>;
        type Domain = &'static str;
        type Error = EvidenceStatusError;

        fn components(&self) -> usize {
            self.components
        }

        fn perturbed_components(&self, mask: &Vec<f64>) -> Vec<usize> {
            perturbed(mask)
        }

        fn separate(
            &mut self,
            support: &ComponentSet,
        ) -> Result<EvidenceStatus<Vec<f64>, &'static str>, EvidenceStatusError> {
            let free: Vec<usize> = (0..self.components)
                .filter(|index| support.members().binary_search(index).is_err())
                .collect();
            let mut digits = vec![0usize; free.len()];
            let mut mask = vec![1.0; self.components];
            let mut best = (f64::NEG_INFINITY, mask.clone());
            let mut cardinality = 0u64;
            'masks: loop {
                for (slot, &component) in free.iter().enumerate() {
                    mask[component] = self.levels[digits[slot]];
                }
                let distance = self.distance(&mask);
                cardinality += 1;
                if distance > best.0 {
                    best = (distance, mask.clone());
                }
                for digit in digits.iter_mut() {
                    *digit += 1;
                    if *digit < self.levels.len() {
                        continue 'masks;
                    }
                    *digit = 0;
                }
                break;
            }
            EvidenceStatus::exact(
                best.0,
                0.0,
                ExactBasis::Exhaustive { cardinality },
                Some(best.1),
                "declared mask grid",
            )
        }

        fn evaluate(
            &mut self,
            mask: &Vec<f64>,
        ) -> Result<EvidenceStatus<Vec<f64>, &'static str>, EvidenceStatusError> {
            EvidenceStatus::exact(
                self.distance(mask),
                0.0,
                ExactBasis::Exhaustive { cardinality: 1 },
                Some(mask.clone()),
                "one mask",
            )
        }
    }

    /// Code length = support size, so the minimum code is the minimum cardinality.
    struct SizeCode;

    impl CardinalityCode for SizeCode {
        type Error = Infallible;

        fn support_bits(&self, components: usize, size: usize) -> Result<u64, Infallible> {
            Ok(size.min(components) as u64)
        }
    }

    /// A P18-shaped subset code with Elias gamma standing in for `L_int`:
    /// `gamma(k + 1) + ceil(log2 binom(C, k))`.
    struct SubsetCode;

    impl CardinalityCode for SubsetCode {
        type Error = Infallible;

        fn support_bits(&self, components: usize, size: usize) -> Result<u64, Infallible> {
            let count = (size + 1) as u64;
            let gamma = 2 * u64::from(63 - count.leading_zeros()) + 1;
            let mut binomial = 1u64;
            for step in 0..size {
                binomial = binomial * (components - step) as u64 / (step + 1) as u64;
            }
            let enumerative = if binomial <= 1 {
                0
            } else {
                u64::from(64 - (binomial - 1).leading_zeros())
            };
            Ok(gamma + enumerative)
        }
    }

    fn set(components: usize, members: &[usize]) -> ComponentSet {
        ComponentSet::new(components, members.to_vec()).expect("members in range")
    }

    #[test]
    fn minimum_hitting_set_is_exact_where_greedy_is_not() {
        // A hub joined to a, b, c, each with a pendant: {a, b, c} hits every edge,
        // greedy takes the hub first and needs four.
        let edges = [[0, 1], [0, 2], [0, 3], [1, 4], [2, 5], [3, 6]];
        let mut hypergraph = FailureHypergraph::<()>::new(7);
        for edge in &edges {
            assert!(
                hypergraph
                    .insert(FailureEdge {
                        perturbed: set(7, edge),
                        witness: None,
                    })
                    .expect("a nonempty edge")
            );
        }
        let exhaustive_minimum = (0u32..1 << 7)
            .filter(|subset| edges.iter().all(|edge| edge.iter().any(|&c| subset >> c & 1 == 1)))
            .map(u32::count_ones)
            .min()
            .expect("the full set hits every edge");
        let solved = hypergraph.minimum_hitting_set();
        assert_eq!(solved.len() as u32, exhaustive_minimum);
        assert_eq!(solved.members(), &[1, 2, 3]);
        assert!(hypergraph.edges().iter().all(|edge| edge.perturbed.intersects(&solved)));
        // Positive control: the greedy incumbent the search starts from is not optimal.
        let members: Vec<&[usize]> = hypergraph.edges().iter().map(|edge| edge.perturbed.members()).collect();
        assert_eq!(greedy_hitting_set(&members, 7).len(), 4);
    }

    #[test]
    fn minimum_hitting_set_equals_the_exhaustive_minimum_on_deterministic_hypergraphs() {
        for components in 3..=8usize {
            for seed in 0..12u64 {
                let full = (1u64 << components) - 1;
                let mut hypergraph = FailureHypergraph::<()>::new(components);
                let mut bitmasks = Vec::new();
                for index in 0..(components as u64 + seed % 5) {
                    let hash = (seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ index.wrapping_mul(0xBF58_476D_1CE4_E5B9))
                        .wrapping_mul(0x94D0_49BB_1331_11EB)
                        >> 17;
                    let bits = hash % full + 1;
                    bitmasks.push(bits);
                    let members: Vec<usize> = (0..components).filter(|c| bits >> c & 1 == 1).collect();
                    hypergraph
                        .insert(FailureEdge {
                            perturbed: set(components, &members),
                            witness: None,
                        })
                        .expect("a nonempty edge");
                }
                let exhaustive_minimum = (0..=full)
                    .filter(|subset| bitmasks.iter().all(|edge| edge & subset != 0))
                    .map(u64::count_ones)
                    .min()
                    .expect("the full set hits every edge");
                let solved = hypergraph.minimum_hitting_set();
                assert_eq!(solved.len() as u32, exhaustive_minimum, "components {components} seed {seed}");
                assert!(bitmasks.iter().all(|edge| solved.members().iter().any(|&c| edge >> c & 1 == 1)));
            }
        }
    }

    #[test]
    fn cegar_keeps_the_interior_component_that_endpoint_masks_miss() {
        // P12: F = m1 m2 + 4 m3 (1 - m3). Endpoint masks never see m3.
        let response = |mask: &[f64]| mask[0] * mask[1] + 4.0 * mask[2] * (1.0 - mask[2]);
        let tolerance = 0.5;

        let mut endpoints = GridOracle { components: 3, levels: vec![0.0, 1.0], response };
        let found = minimum_code_support(&mut endpoints, &SizeCode, tolerance, FailureHypergraph::new(3))
            .expect("the endpoint search closes");
        let certified = found.certified.expect("a certified support");
        assert_eq!(certified.support.members(), &[0, 1]);
        assert!(matches!(found.code, EvidenceStatus::Exact { basis: ExactBasis::ClosedSearch, .. }));
        assert_eq!(found.code.upper_bound(), Some(2.0));
        assert_eq!(found.code.lower_bound(), Some(2.0));
        assert!(found.undecided.is_none());
        assert!(found.hypergraph.edges().iter().all(|edge| !edge.perturbed.is_subset_of(&set(3, &[2]))));

        let mut interior = GridOracle { components: 3, levels: vec![0.0, 0.5, 1.0], response };
        let found = minimum_code_support(&mut interior, &SizeCode, tolerance, FailureHypergraph::new(3))
            .expect("the interior search closes");
        let certified = found.certified.expect("a certified support");
        assert_eq!(certified.support.members(), &[0, 1, 2]);
        assert!(matches!(found.code, EvidenceStatus::Exact { basis: ExactBasis::ClosedSearch, .. }));
        assert_eq!(found.code.upper_bound(), Some(3.0));
        assert!(found.hypergraph.edges().iter().any(|edge| edge.perturbed.members() == [2]));
    }

    #[test]
    fn a_redundant_pair_is_one_or_constraint_not_two_scalar_importances() {
        let response = |mask: &[f64]| mask[0] + mask[1] - mask[0] * mask[1];
        let tolerance = 0.5;
        let mut oracle = GridOracle { components: 2, levels: vec![0.0, 1.0], response };
        let found = minimum_code_support(&mut oracle, &SizeCode, tolerance, FailureHypergraph::new(2))
            .expect("the search closes");
        assert_eq!(found.hypergraph.edges().len(), 1);
        assert_eq!(found.hypergraph.edges()[0].perturbed.members(), &[0, 1]);
        assert!(matches!(found.code, EvidenceStatus::Exact { .. }));
        assert_eq!(found.code.upper_bound(), Some(1.0));
        // Negative control: each single deletion is harmless, so scalar importances
        // keep neither component, and the empty support they select is refuted.
        for mask in [vec![0.0, 1.0], vec![1.0, 0.0]] {
            assert!(!oracle.evaluate(&mask).expect("an exact evaluation").refutes_at_most(tolerance));
        }
        assert!(oracle.separate(&set(2, &[])).expect("an exhaustive separation").refutes_at_most(tolerance));
    }

    #[test]
    fn a_full_support_is_chosen_when_the_subset_code_makes_it_shortest() {
        // P18: with C = 4 and tau = 2, gamma(3) + ceil(log2 6) = 6 bits, while the
        // full support costs gamma(5) + 0 = 5 bits.
        let response = |mask: &[f64]| mask[0] * mask[1];
        let tolerance = 0.5;
        let mut oracle = GridOracle { components: 4, levels: vec![0.0, 1.0], response };
        assert_eq!(SubsetCode.support_bits(4, 2), Ok(6));
        assert_eq!(SubsetCode.support_bits(4, 4), Ok(5));
        let found = minimum_code_support(&mut oracle, &SubsetCode, tolerance, FailureHypergraph::new(4))
            .expect("the search closes");
        assert_eq!(found.certified.expect("a certified support").support.members(), &[0, 1, 2, 3]);
        assert!(matches!(found.code, EvidenceStatus::Exact { .. }));
        assert_eq!(found.code.upper_bound(), Some(5.0));
        // Positive control: a size code on the same oracle keeps only the pair.
        let found = minimum_code_support(&mut oracle, &SizeCode, tolerance, FailureHypergraph::new(4))
            .expect("the search closes");
        assert_eq!(found.certified.expect("a certified support").support.members(), &[0, 1]);
    }

    /// An oracle that neither refutes nor certifies a partial support.
    struct UndecidedOracle {
        components: usize,
        full_certified: bool,
    }

    impl SeparationOracle for UndecidedOracle {
        type Mask = Vec<f64>;
        type Domain = &'static str;
        type Error = EvidenceStatusError;

        fn components(&self) -> usize {
            self.components
        }

        fn perturbed_components(&self, mask: &Vec<f64>) -> Vec<usize> {
            perturbed(mask)
        }

        fn separate(
            &mut self,
            support: &ComponentSet,
        ) -> Result<EvidenceStatus<Vec<f64>, &'static str>, EvidenceStatusError> {
            if support.len() == self.components && self.full_certified {
                EvidenceStatus::exact(0.0, 0.0, ExactBasis::Algebraic, None, "all on")
            } else {
                EvidenceStatus::unresolved(0.25, 0.75, Extremum::Supremum, None, "[0,1]^C")
            }
        }

        fn evaluate(
            &mut self,
            mask: &Vec<f64>,
        ) -> Result<EvidenceStatus<Vec<f64>, &'static str>, EvidenceStatusError> {
            EvidenceStatus::unresolved(0.25, 0.75, Extremum::Supremum, Some(mask.clone()), "one mask")
        }
    }

    #[test]
    fn an_undecided_candidate_ends_the_search_unresolved_with_its_gap() {
        let tolerance = 0.5;
        let mut oracle = UndecidedOracle { components: 3, full_certified: true };
        let found = minimum_code_support(&mut oracle, &SizeCode, tolerance, FailureHypergraph::new(3))
            .expect("the search ends");
        assert!(matches!(found.code, EvidenceStatus::Unresolved { extremum: Extremum::Infimum, .. }));
        assert_eq!(found.code.lower_bound(), Some(0.0));
        assert_eq!(found.code.upper_bound(), Some(3.0));
        assert_eq!(found.code.gap(), Some(3.0));
        assert!(!found.code.certifies_at_most(2.0));
        assert_eq!(found.certified.expect("the full support is certified").support.len(), 3);
        assert!(found.undecided.expect("an undecided candidate").support.is_empty());
        assert_eq!(found.separations, 2);

        let mut oracle = UndecidedOracle { components: 3, full_certified: false };
        let found = minimum_code_support(&mut oracle, &SizeCode, tolerance, FailureHypergraph::new(3))
            .expect("the search ends");
        assert_eq!(found.code.gap(), Some(f64::INFINITY));
        assert!(found.certified.is_none());
    }

    #[test]
    fn the_union_of_per_input_sufficient_supports_is_sufficient_at_each_inputs_own_tolerance() {
        let first = |mask: &[f64]| mask[0];
        let second = |mask: &[f64]| mask[1] * mask[2];
        let mut at_first = GridOracle { components: 3, levels: vec![0.0, 1.0], response: first };
        let mut at_second = GridOracle { components: 3, levels: vec![0.0, 1.0], response: second };
        let tolerances = [0.5, 0.25];
        let mut per_input = Vec::new();
        for (index, tolerance) in tolerances.into_iter().enumerate() {
            let found = if index == 0 {
                minimum_code_support(&mut at_first, &SizeCode, tolerance, FailureHypergraph::new(3))
            } else {
                minimum_code_support(&mut at_second, &SizeCode, tolerance, FailureHypergraph::new(3))
            }
            .expect("the search closes");
            let certified = found.certified.expect("a certified support");
            per_input.push(InputSupport { support: certified.support, tolerance, evidence: certified.evidence });
        }
        assert_eq!(per_input[0].support.members(), &[0]);
        assert_eq!(per_input[1].support.members(), &[1, 2]);

        let union = sufficient_union(3, &per_input).expect("both inputs are certified");
        assert_eq!(union.support.members(), &[0, 1, 2]);
        let direct = [
            at_first.separate(&union.support).expect("an exhaustive separation"),
            at_second.separate(&union.support).expect("an exhaustive separation"),
        ];
        for ((inherited, direct), tolerance) in union.per_input.iter().zip(&direct).zip(tolerances) {
            assert!(inherited.certifies_at_most(tolerance));
            assert!(direct.upper_bound().expect("exact") <= inherited.upper_bound().expect("a bound"));
        }

        // Monotonicity pinned exhaustively: R(U) <= R(S) for every S within U.
        for oracle in [&mut at_first as &mut dyn SeparationOracle<Mask = Vec<f64>, Domain = &'static str, Error = EvidenceStatusError>, &mut at_second] {
            for subset in 0u32..8 {
                for superset in (0u32..8).filter(|superset| superset & subset == subset) {
                    let members = |bits: u32| (0..3usize).filter(|c| bits >> c & 1 == 1).collect::<Vec<usize>>();
                    let small = oracle.separate(&set(3, &members(subset))).expect("exhaustive");
                    let large = oracle.separate(&set(3, &members(superset))).expect("exhaustive");
                    assert!(large.upper_bound().expect("exact") <= small.lower_bound().expect("exact"));
                }
            }
        }

        // Negative control: the intersection of the per-input supports is refuted at both.
        let intersection = set(3, &[]);
        assert!(at_first.separate(&intersection).expect("exhaustive").refutes_at_most(tolerances[0]));
        assert!(at_second.separate(&intersection).expect("exhaustive").refutes_at_most(tolerances[1]));

        // An input whose evidence does not certify its support is refused.
        let undecided: InputSupport<Vec<f64>, &'static str> = InputSupport {
            support: set(3, &[0]),
            tolerance: 0.5,
            evidence: EvidenceStatus::unresolved(0.25, 0.75, Extremum::Supremum, None, "[0,1]^C")
                .expect("a valid interval"),
        };
        assert!(matches!(
            sufficient_union(3, &[per_input[1].clone(), undecided]),
            Err(SupportSearchError::InsufficientInputSupport { input: 1 })
        ));
    }

    #[test]
    fn strength_is_containment_of_the_proven_interval_on_one_domain() {
        // mpd-verify's two counterexamples to ordering by kind. An exact value with a
        // wide numerical error proves less about `<= 0.45` than a uniform bound at
        // 0.45, and that bound proves less than the bracket [0.3, 0.4].
        let exact = Status::exact(0.5, 0.1, ExactBasis::Algebraic, None, "d").expect("exact");
        let bound = Status::uniform_bound(0.45, 0.0, "d").expect("a bound");
        let bracket =
            Status::unresolved(0.3, 0.4, Extremum::Supremum, None, "d").expect("an interval");
        assert!(bound.certifies_at_most(0.45) && !exact.certifies_at_most(0.45));
        assert_eq!(exact.compare_strength(&bound), None);
        assert_eq!(bound.compare_strength(&exact), None);
        assert!(bracket.certifies_at_most(0.4) && !bound.certifies_at_most(0.4));
        assert!(bracket.refutes_at_most(0.29) && !bound.refutes_at_most(0.29));
        assert_eq!(bound.compare_strength(&bracket), Some(Ordering::Less));
        assert_eq!(bracket.compare_strength(&bound), Some(Ordering::Greater));
        // Positive controls: a tight exact value inside the bracket is stronger, and a
        // status equals itself.
        let tight = Status::exact(0.35, 0.0, ExactBasis::Algebraic, None, "d").expect("exact");
        assert_eq!(tight.compare_strength(&bracket), Some(Ordering::Greater));
        assert_eq!(bracket.compare_strength(&tight), Some(Ordering::Less));
        assert_eq!(bracket.compare_strength(&bracket), Some(Ordering::Equal));
        // Another domain, another extremum, an estimate and a counterexample are
        // incomparable.
        let elsewhere = Status::uniform_bound(0.45, 0.0, "other").expect("a bound");
        assert_eq!(bound.compare_strength(&elsewhere), None);
        let infimum =
            Status::unresolved(0.3, 0.4, Extremum::Infimum, None, "d").expect("an interval");
        assert_eq!(bracket.compare_strength(&infimum), None);
        let estimate = Status::statistical_estimate(0.35, 0.01, 10, "d").expect("an estimate");
        assert_eq!(estimate.compare_strength(&estimate), None);
        assert_eq!(bracket.compare_strength(&estimate), None);
        let found = Status::counterexample(1.0, 0.0, 0.5, vec![0]).expect("a violation");
        assert_eq!(found.compare_strength(&found), None);
    }

    #[test]
    fn conflict_replay_carries_a_refinement_and_drops_unrepresentable_repaired_and_witnessless_conflicts() {
        let tolerance = 0.5;
        let old_response = |mask: &[f64]| mask[0] + mask[1] - mask[0] * mask[1];
        let mut old = GridOracle { components: 2, levels: vec![0.0, 1.0], response: old_response };
        let recorded = minimum_code_support(&mut old, &SizeCode, tolerance, FailureHypergraph::new(2))
            .expect("the old search closes")
            .hypergraph;
        assert_eq!(recorded.edges().len(), 1);

        // Refinement: old component 0 splits into new pieces 0 and 1 carrying half of
        // its moment each; old component 1 becomes new component 2.
        let refined_response = |mask: &[f64]| {
            let merged = 0.5 * (mask[0] + mask[1]);
            merged + mask[2] - merged * mask[2]
        };
        let mut refined = GridOracle { components: 3, levels: vec![0.0, 1.0], response: refined_response };
        let replay: ConflictReplay<Vec<f64>> =
            replay_conflicts(&recorded, |old_mask: &Vec<f64>| Some(vec![old_mask[0], old_mask[0], old_mask[1]]), &mut refined, tolerance)
                .expect("the replay evaluates");
        assert_eq!(replay.hypergraph.edges().len(), 1);
        assert_eq!(replay.hypergraph.edges()[0].perturbed.members(), &[0, 1, 2]);
        assert_eq!((replay.unrepresentable, replay.no_longer_bad, replay.without_witness), (0, 0, 0));

        let cold = minimum_code_support(&mut refined, &SizeCode, tolerance, FailureHypergraph::new(3))
            .expect("the cold search closes");
        let warm = minimum_code_support(&mut refined, &SizeCode, tolerance, replay.hypergraph)
            .expect("the warm search closes");
        assert_eq!(cold.code, warm.code);
        assert!(warm.separations < cold.separations);

        // A merge has no representative for a mask that deletes only one old piece.
        let mut split = FailureHypergraph::new(2);
        split
            .insert(FailureEdge { perturbed: set(2, &[0]), witness: Some(vec![0.0, 1.0]) })
            .expect("a nonempty edge");
        split.insert(FailureEdge { perturbed: set(2, &[1]), witness: None }).expect("a nonempty edge");
        let mut merged = GridOracle { components: 1, levels: vec![0.0, 1.0], response: |mask: &[f64]| mask[0] };
        let replay = replay_conflicts(&split, |old_mask: &Vec<f64>| (old_mask[0] == old_mask[1]).then(|| vec![old_mask[0]]), &mut merged, tolerance)
            .expect("the replay evaluates");
        assert_eq!((replay.unrepresentable, replay.no_longer_bad, replay.without_witness), (1, 0, 1));
        assert!(replay.hypergraph.edges().is_empty());

        // A repaired decomposition no longer fails at the replayed mask.
        let mut repaired = GridOracle { components: 3, levels: vec![0.0, 1.0], response: |mask: &[f64]| mask[0].max(1.0) };
        let replay = replay_conflicts(&recorded, |old_mask: &Vec<f64>| Some(vec![old_mask[0], old_mask[0], old_mask[1]]), &mut repaired, tolerance)
            .expect("the replay evaluates");
        assert_eq!(replay.no_longer_bad, 1);
        assert!(replay.hypergraph.edges().is_empty());
    }
}
