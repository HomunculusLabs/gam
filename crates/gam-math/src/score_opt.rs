//! Certified global optimization of one-dimensional scores on a bounded
//! domain, together with the affine-pencil spectral profile shared by the
//! Gaussian REML smoothing-parameter searches.
//!
//! Point samples alone cannot prove that a smooth function has no narrow
//! stationary pair between them.  The search therefore requires two pieces of
//! information from its caller:
//!
//! * a nearest-rounded point evaluation `(value, first derivative, second
//!   derivative)`, used only as a representative and to propose refinements;
//! * an OUTER enclosure of the exact score value and both exact derivatives
//!   over every requested interval, accompanied by a certified forward-error
//!   bound for the scalar score evaluator.
//!
//! An interval is discarded only when its first-derivative enclosure excludes
//! zero.  A stationary point is refined only after the second-derivative
//! enclosure excludes zero, proving that the first derivative is monotone and
//! hence that certified endpoint derivative ranges of opposite sign contain
//! exactly one root. Every other interval is subdivided unless the exact score
//! range is narrower than the score evaluator's certified pairwise
//! forward-error floor. Such a region is returned explicitly as a
//! [`ResolutionFlatRegion`]; it is never mislabeled as a stationary point. If
//! neither stationary structure nor score-value flatness is proved before the
//! requested abscissa resolution, the result is a typed
//! [`ScoreSearchError::Unresolved`] rather than a best-effort optimum.
//!
//! [`AffineRemlProfile`] supplies both the point jets and rigorous interval
//! formulas for scores whose penalized Hessian has simultaneously diagonal
//! affine modes `h_i(lambda) = g_i + lambda s_i`.  This covers an ordinary
//! Demmler--Reinsch eigensystem (`g_i = 1`) and a reference-Hessian pencil
//! (`g_i = 1 - lambda_0 mu_i`, `s_i = mu_i`) without any matrix dependency in
//! this crate.

use std::fmt;
use std::sync::OnceLock;

/// Closed real interval `[lo, hi]`.
///
/// Search callbacks may use infinite endpoints for conservative bounds, but
/// neither endpoint may be NaN and `lo <= hi` must hold.  The search validates
/// every enclosure returned by a callback.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClosedInterval {
    pub lo: f64,
    pub hi: f64,
}

impl ClosedInterval {
    #[inline]
    pub const fn new(lo: f64, hi: f64) -> Self {
        Self { lo, hi }
    }

    #[inline]
    pub const fn point(value: f64) -> Self {
        Self {
            lo: value,
            hi: value,
        }
    }

    #[inline]
    pub const fn entire() -> Self {
        Self {
            lo: f64::NEG_INFINITY,
            hi: f64::INFINITY,
        }
    }

    #[inline]
    pub fn contains(self, value: f64) -> bool {
        self.lo <= value && value <= self.hi
    }

    #[inline]
    pub fn contains_zero(self) -> bool {
        self.contains(0.0)
    }

    #[inline]
    fn is_valid(self) -> bool {
        !self.lo.is_nan() && !self.hi.is_nan() && self.lo <= self.hi
    }

    #[inline]
    fn hull(self, other: Self) -> Self {
        Self {
            lo: self.lo.min(other.lo),
            hi: self.hi.max(other.hi),
        }
    }

    #[inline]
    fn intersection(self, other: Self) -> Option<Self> {
        let intersection = Self {
            lo: self.lo.max(other.lo),
            hi: self.hi.min(other.hi),
        };
        (intersection.lo <= intersection.hi).then_some(intersection)
    }

    #[inline]
    fn max_abs(self) -> f64 {
        self.lo.abs().max(self.hi.abs())
    }

    #[inline]
    fn widen(self, radius: f64) -> Self {
        if radius == 0.0 {
            return self;
        }
        if radius == f64::INFINITY {
            return Self::entire();
        }
        Self {
            lo: next_down(self.lo - radius),
            hi: next_up(self.hi + radius),
        }
    }

    #[inline]
    /// Directed outer enclosure of the exact sum of two intervals.
    pub fn add(self, other: Self) -> Self {
        Self {
            lo: sum_down(self.lo, other.lo),
            hi: sum_up(self.hi, other.hi),
        }
    }

    #[inline]
    /// Directed outer enclosure of the exact interval difference.
    pub fn sub(self, other: Self) -> Self {
        Self {
            lo: sum_down(self.lo, -other.hi),
            hi: sum_up(self.hi, -other.lo),
        }
    }

    #[inline]
    /// Exact sign reversal of the interval.
    pub fn neg(self) -> Self {
        Self {
            lo: -self.hi,
            hi: -self.lo,
        }
    }

    /// Directed outer enclosure of the exact interval product.
    pub fn mul(self, other: Self) -> Self {
        let pairs = [
            (self.lo, other.lo),
            (self.lo, other.hi),
            (self.hi, other.lo),
            (self.hi, other.hi),
        ];
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for (left, right) in pairs {
            lo = lo.min(product_down(left, right));
            hi = hi.max(product_up(left, right));
        }
        Self { lo, hi }
    }

    #[inline]
    /// Directed outer enclosure after multiplication by an exact binary64
    /// scalar.
    pub fn scale(self, value: f64) -> Self {
        self.mul(Self::point(value))
    }

    fn square(self) -> Self {
        if self.lo >= 0.0 {
            Self {
                lo: product_down(self.lo, self.lo).max(0.0),
                hi: product_up(self.hi, self.hi),
            }
        } else if self.hi <= 0.0 {
            Self {
                lo: product_down(self.hi, self.hi).max(0.0),
                hi: product_up(self.lo, self.lo),
            }
        } else {
            Self {
                lo: 0.0,
                hi: product_up(self.lo, self.lo).max(product_up(self.hi, self.hi)),
            }
        }
    }

    /// Natural logarithm of an interval known to be strictly positive.
    fn ln_positive(self) -> Self {
        assert!(
            self.lo > 0.0,
            "ln_positive requires a strictly positive interval, got lo={}",
            self.lo
        );
        let lo = certified_ln_positive(self.lo)
            .expect("ln_positive lower endpoint is finite and positive");
        let hi = certified_ln_positive(self.hi)
            .expect("ln_positive upper endpoint is finite and positive");
        Self::new(lo.lo, hi.hi)
    }

    /// Divide by an interval known to be strictly positive.
    fn div_positive(self, denominator: Self) -> Self {
        assert!(
            denominator.lo > 0.0,
            "div_positive requires a strictly positive denominator interval, got lo={}",
            denominator.lo
        );
        let reciprocal = Self {
            lo: quotient_down(1.0, denominator.hi).max(0.0),
            hi: quotient_up(1.0, denominator.lo),
        };
        self.mul(reciprocal)
    }

    /// Divide by an interval that excludes zero.
    fn div_nonzero(self, denominator: Self) -> Self {
        if denominator.lo > 0.0 {
            self.div_positive(denominator)
        } else {
            assert!(
                denominator.hi < 0.0,
                "div_nonzero requires a denominator interval excluding zero, got {denominator:?}"
            );
            self.div_positive(denominator.neg()).neg()
        }
    }

    #[inline]
    fn nonnegative(self) -> Self {
        Self {
            lo: self.lo.max(0.0),
            hi: self.hi.max(0.0),
        }
    }
}

/// Nearest-rounded value and analytic derivatives at one abscissa.
///
/// `third` is carried alongside the first two because every endpoint-anchored
/// [`DerivativeEnclosure`] in this workspace is built from the endpoint
/// curvature and third derivative. Dropping it here used to force the enclosure
/// oracle to RE-EVALUATE the criterion at both endpoints of every
/// branch-and-bound cell — endpoints the search had already sampled — which
/// tripled the number of criterion evaluations the search actually paid for.
/// Oracles that have no third derivative to report set it to zero; enclosures
/// that do not consult it are unaffected.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScoreJet {
    pub value: f64,
    pub derivative: f64,
    pub curvature: f64,
    pub third: f64,
}

/// A point evaluation augmented with its abscissa.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScoreSample {
    pub x: f64,
    pub value: f64,
    pub derivative: f64,
    pub curvature: f64,
    pub third: f64,
}

/// Exact score-value range and the numerical resolution of point values.
///
/// `value` contains the exact-real score at every point of the cell.
/// `evaluation_error` is an absolute forward-error bound for both endpoint
/// values supplied with that cell:
///
/// `|endpoint.value - exact_score(endpoint.x)| <= evaluation_error`.
///
/// An interval-extension oracle may provide the stronger cell-uniform bound.
/// The search needs only the endpoint statement: every representative it
/// retains is an evaluated cell endpoint. The corresponding uncertainty of a
/// comparison between the two endpoints is at most `2 * evaluation_error`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScoreValueEnclosure {
    pub value: ClosedInterval,
    pub evaluation_error: f64,
}

/// Exact-real score and derivative ranges supplied to the certified search.
///
/// Scalar derivative estimates are proposals only. Exclusion, monotonicity,
/// and root-sign decisions use these mathematical ranges directly, so
/// derivative-evaluator roundoff never becomes part of a proof predicate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DerivativeEnclosure {
    pub score: ScoreValueEnclosure,
    pub derivative: ClosedInterval,
    pub curvature: ClosedInterval,
}

/// A region whose unresolved stationary structure is immaterial at the
/// representable resolution of its score.
///
/// `max_score_gap` is the width of the cell's exact score-value enclosure.
/// `score_resolution` is the certified forward-error bound for comparing two
/// point score evaluations. The search records this region only when
/// `max_score_gap <= score_resolution`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ResolutionFlatRegion {
    pub sample: ScoreSample,
    pub bracket: ClosedInterval,
    /// Exact score range over `bracket`.
    pub score: ClosedInterval,
    pub max_score_gap: f64,
    pub score_resolution: f64,
}

/// One stationary point together with the final bracket that certifies its
/// location.  The bracket width is no larger than the requested resolution,
/// unless the point was represented exactly (a zero-width bracket).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StationaryPoint {
    pub sample: ScoreSample,
    pub bracket: ClosedInterval,
    /// Exact score range over `bracket` and endpoint evaluation resolution.
    pub score: ScoreValueEnclosure,
}

/// Exact-value certificate for the representative selected by the rounded
/// evaluator.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GlobalScoreCertificate {
    /// Exact score at the returned representative.
    pub selected: ClosedInterval,
    /// Outer range containing the exact global maximum.
    pub maximum: ClosedInterval,
    /// Outward bound on `global maximum - exact score(representative)`.
    /// Repeated certificates of the same represented point contribute zero:
    /// they name the same exact real value, rather than independent uncertain
    /// quantities.
    pub maximum_excess: f64,
    /// Outward sum of the selected point evaluator's forward error and the
    /// largest competing representative's forward error. Exact terminal
    /// ranges remain separate in [`Self::maximum_excess`].
    pub comparison_resolution: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScoreOptimumLocation {
    LowerBoundary,
    UpperBoundary,
    Stationary(usize),
    ResolutionFlat(usize),
}

/// Complete successful search result. Endpoints, isolated stationary points,
/// and resolution-flat regions are retained explicitly so the rounded-value
/// comparison and every value-resolution certificate are independently
/// checkable by the caller.
#[derive(Clone, Debug, PartialEq)]
pub struct ScoreSearchResult {
    pub optimum: ScoreSample,
    pub location: ScoreOptimumLocation,
    pub lower_boundary: ScoreSample,
    pub upper_boundary: ScoreSample,
    pub stationary_points: Vec<StationaryPoint>,
    pub resolution_flat_regions: Vec<ResolutionFlatRegion>,
    pub value_certificate: GlobalScoreCertificate,
}

/// Failure of the generic certified search.
#[derive(Debug)]
pub enum ScoreSearchError<E> {
    InvalidDomain {
        lo: f64,
        hi: f64,
    },
    InvalidResolution {
        resolution: f64,
    },
    PointEvaluation {
        x: f64,
        source: E,
    },
    EnclosureEvaluation {
        lo: f64,
        hi: f64,
        source: E,
    },
    NonFiniteSample {
        sample: ScoreSample,
    },
    InvalidEnclosure {
        lo: f64,
        hi: f64,
        enclosure: DerivativeEnclosure,
    },
    ScoreValueEnclosureMissesEndpoint {
        lo: f64,
        hi: f64,
        endpoint: ScoreSample,
        score: ScoreValueEnclosure,
    },
    DisjointEndpointEnclosure {
        lo: f64,
        hi: f64,
        endpoint: ScoreSample,
        endpoint_derivative: ClosedInterval,
        enclosure: DerivativeEnclosure,
    },
    /// Neither stationary exclusion/isolation nor score flatness could be
    /// proved before the requested or floating-point abscissa-resolution
    /// floor.
    Unresolved {
        lo: f64,
        hi: f64,
        requested_resolution: f64,
        enclosure: DerivativeEnclosure,
    },
    /// The traversal asked for more cell subdivisions than
    /// [`subdivision_budget`] allows for this domain and resolution. Reported
    /// with the cell that was being split when the budget ran out, so the
    /// caller can see WHERE the criterion stopped being decomposable.
    SubdivisionBudget {
        lo: f64,
        hi: f64,
        cell_lo: f64,
        cell_hi: f64,
        requested_resolution: f64,
        subdivisions: usize,
        budget: usize,
        depth_bound: u32,
    },
}

/// Total cell subdivisions a converging certified 1-D search may spend on
/// `[lo, hi]` at `resolution`.
///
/// Two facts set the scale. First, no cell can be halved more than
/// `D = ceil(log2((hi - lo) / resolution))` times before it is narrower than
/// `resolution`, where the search already stops with
/// [`ScoreSearchError::Unresolved`] — so `D` bounds the depth of the
/// subdivision tree outright. Second, a search that is ISOLATING structure
/// spends at most `D` subdivisions per cell it finally certifies, because each
/// one halves the cell it is working in.
///
/// So the whole traversal costs at most `D` times the size of the certified
/// decomposition, and the budget is that product with the decomposition
/// allowed `2 D` cells — twice as many certified cells as the domain has
/// resolvable binary levels. Measured on #2546's cascade: every terminating
/// search on that surface spent 33–39 subdivisions at `D = 32`, i.e. about `D`,
/// against a budget of `2 D² = 2048`; the non-terminating one passes 40 000
/// with its bracket still halving cleanly at every node. The margin over the
/// deepest currently-successful search is ~60x, so the budget is invisible to
/// every search that converges and is reached in under a second by one that
/// does not.
///
/// gam#2614 — that ~60x margin is NOT general, and the `2 D` cell allowance is
/// the reason. The `D` factor is derived: no cell survives more than `D`
/// halvings before it is narrower than `resolution`. The cell allowance is an
/// assumption about how many cells a criterion's certified decomposition
/// contains, which is exactly what the search cannot know in advance.
///
/// Read the calibration above again: spending about `D` subdivisions IN TOTAL,
/// at `D` per certified cell, means that surface's decomposition was about ONE
/// cell. The `2 D` allowance (64 cells at `D = 32`) was never exercised there,
/// so the quoted margin is headroom over a single-cell case.
///
/// Measured since, at the same `D = 32`:
/// `spline_scan::tests::order_one_scan_matches_dense_random_walk_posterior`
/// exceeds 2048 subdivisions — its decomposition needs MORE than 64 cells — and
/// it terminates and PASSES once the budget is `100 D²`. It is not going deeper
/// than `D` per cell; the depth bound is a hard geometric fact. It is isolating
/// structure over a wider decomposition than this constant anticipates, so
/// against that surface the margin is negative.
///
/// A larger allowance does not repair every scan refusal.
/// `weighted_scan_dgp_2300_search_terminates_in_bounded_evaluations` still fails
/// with the budget raised 25x, because its endpoint certificates carry
/// `eval_err ~ 1e-6` in one region of the domain while the search requests
/// `resolution = 1.49e-8`. That is an evaluation-conditioning defect; no cell
/// allowance reaches it.
///
/// A degenerate domain still gets a budget of at least one subdivision: the
/// bound is a backstop against unbounded breadth, never a refusal of the first
/// split.
pub fn subdivision_budget(lo: f64, hi: f64, resolution: f64) -> (usize, u32) {
    let width = hi - lo;
    if !(width.is_finite() && width > 0.0 && resolution.is_finite() && resolution > 0.0) {
        return (1, 0);
    }
    let levels = (width / resolution).log2().ceil();
    let depth_bound = if levels.is_finite() && levels >= 1.0 {
        // `f64::MANTISSA_DIGITS`-scaled domains cannot exceed the exponent
        // range, so the cast is saturating in practice and clamped in fact.
        levels.min(u32::MAX as f64) as u32
    } else {
        1
    };
    let depth = depth_bound as usize;
    (2 * depth * depth, depth_bound)
}

impl<E: fmt::Display> fmt::Display for ScoreSearchError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidDomain { lo, hi } => {
                write!(f, "score search: invalid domain [{lo}, {hi}]")
            }
            Self::InvalidResolution { resolution } => {
                write!(f, "score search: invalid resolution {resolution}")
            }
            Self::PointEvaluation { x, source } => {
                write!(f, "score search: evaluation failed at {x}: {source}")
            }
            Self::EnclosureEvaluation { lo, hi, source } => write!(
                f,
                "score search: score/derivative enclosure failed on [{lo}, {hi}]: {source}"
            ),
            Self::NonFiniteSample { sample } => write!(
                f,
                "score search: non-finite jet at {} (value {}, derivative {}, curvature {}, third {})",
                sample.x, sample.value, sample.derivative, sample.curvature, sample.third
            ),
            Self::InvalidEnclosure { lo, hi, enclosure } => write!(
                f,
                "score search: invalid score/derivative enclosure on [{lo}, {hi}]: {enclosure:?}"
            ),
            Self::ScoreValueEnclosureMissesEndpoint {
                lo,
                hi,
                endpoint,
                score,
            } => write!(
                f,
                "score search: exact score range {:?} plus evaluator error {} on [{lo}, {hi}] misses the rounded endpoint value {} at {}",
                score.value,
                score.evaluation_error,
                endpoint.value,
                endpoint.x
            ),
            Self::DisjointEndpointEnclosure {
                lo,
                hi,
                endpoint,
                endpoint_derivative,
                enclosure,
            } => write!(
                f,
                "score search: derivative enclosures on [{lo}, {hi}] and its endpoint {} are disjoint: endpoint range {endpoint_derivative:?}, cell {enclosure:?}; point estimate {endpoint:?}",
                endpoint.x
            ),
            Self::Unresolved {
                lo,
                hi,
                requested_resolution,
                enclosure,
            } => write!(
                f,
                "score search: stationary structure unresolved on [{lo}, {hi}] at requested resolution {requested_resolution}: {enclosure:?}"
            ),
            Self::SubdivisionBudget {
                lo,
                hi,
                cell_lo,
                cell_hi,
                requested_resolution,
                subdivisions,
                budget,
                depth_bound,
            } => write!(
                f,
                "score search: {subdivisions} cell subdivisions on [{lo}, {hi}] at requested \
                 resolution {requested_resolution} exceed the budget {budget} derived from this \
                 domain's subdivision depth bound {depth_bound}; the criterion is still \
                 undecomposable at [{cell_lo}, {cell_hi}], so it neither excludes nor isolates \
                 stationary structure over a region the search can only enumerate"
            ),
        }
    }
}

impl<E: std::error::Error + 'static> std::error::Error for ScoreSearchError<E> {}

#[derive(Clone, Copy)]
struct SearchSample {
    sample: ScoreSample,
    point_enclosure: Option<DerivativeEnclosure>,
}

#[derive(Clone, Copy)]
struct SearchNode {
    left: SearchSample,
    right: SearchSample,
}

#[derive(Clone, Copy)]
struct TerminalScoreCandidate {
    score: ScoreValueEnclosure,
    /// Forward error of the rounded representative used to compare this
    /// candidate with the selected representative. For a region certificate
    /// this is kept separate from the exact range: the range bounds the
    /// terminal maximum, while the error belongs to an actually evaluated
    /// point.
    comparison_error: f64,
    /// Present only when the terminal maximum is the exact score at this
    /// represented point. Region certificates deliberately carry `None` so
    /// their possible improvement over a representative is retained.
    point_x: Option<f64>,
}

impl TerminalScoreCandidate {
    #[inline]
    fn point(x: f64, score: ScoreValueEnclosure) -> Self {
        Self {
            score,
            comparison_error: score.evaluation_error,
            point_x: Some(x),
        }
    }

    #[inline]
    fn region(score: ScoreValueEnclosure, comparison_error: f64) -> Self {
        Self {
            score,
            comparison_error,
            point_x: None,
        }
    }
}

fn evaluate_sample<E, F>(x: f64, evaluate: &mut F) -> Result<SearchSample, ScoreSearchError<E>>
where
    F: FnMut(f64) -> Result<ScoreJet, E>,
{
    let jet = evaluate(x).map_err(|source| ScoreSearchError::PointEvaluation { x, source })?;
    let sample = ScoreSample {
        x,
        value: jet.value,
        derivative: jet.derivative,
        curvature: jet.curvature,
        third: jet.third,
    };
    if sample.value.is_finite()
        && sample.derivative.is_finite()
        && sample.curvature.is_finite()
        && sample.third.is_finite()
    {
        Ok(SearchSample {
            sample,
            point_enclosure: None,
        })
    } else {
        Err(ScoreSearchError::NonFiniteSample { sample })
    }
}

fn checked_enclosure<E, F>(
    left: ScoreSample,
    right: ScoreSample,
    enclose: &mut F,
) -> Result<DerivativeEnclosure, ScoreSearchError<E>>
where
    F: FnMut(ScoreSample, ScoreSample) -> Result<DerivativeEnclosure, E>,
{
    let lo = left.x;
    let hi = right.x;
    // The cell's endpoints are handed to the oracle as the SAMPLES the search
    // already paid for, not as bare abscissae. An endpoint-anchored enclosure
    // needs the endpoint jets and nothing else, so this is what makes it free:
    // the oracle reads `left`/`right` instead of re-evaluating the criterion at
    // two points it has already evaluated.
    let enclosure =
        enclose(left, right).map_err(|source| ScoreSearchError::EnclosureEvaluation {
            lo,
            hi,
            source,
        })?;
    if !(enclosure.derivative.is_valid()
        && enclosure.curvature.is_valid()
        && enclosure.score.value.is_valid()
        && enclosure.score.evaluation_error.is_finite()
        && enclosure.score.evaluation_error >= 0.0)
    {
        return Err(ScoreSearchError::InvalidEnclosure { lo, hi, enclosure });
    }
    let score = enclosure.score;
    let resolved_score = score.value.widen(score.evaluation_error);
    for endpoint in [left, right] {
        if !resolved_score.contains(endpoint.value) {
            return Err(ScoreSearchError::ScoreValueEnclosureMissesEndpoint {
                lo,
                hi,
                endpoint,
                score,
            });
        }
    }
    Ok(enclosure)
}

/// Attach the oracle's exact score/derivative ranges at one represented point.
///
/// A nearest-rounded scalar jet is not required to lie inside an exact-real
/// interval extension.  Instead, proof decisions use this degenerate-cell
/// enclosure.  Both the point range and its parent-cell range contain the same
/// exact endpoint derivative, so disjointness remains a valid contract check.
fn certify_point<E, F>(
    point: &mut SearchSample,
    enclose: &mut F,
) -> Result<DerivativeEnclosure, ScoreSearchError<E>>
where
    F: FnMut(ScoreSample, ScoreSample) -> Result<DerivativeEnclosure, E>,
{
    let enclosure = match point.point_enclosure {
        Some(enclosure) => enclosure,
        None => {
            let enclosure = checked_enclosure(point.sample, point.sample, enclose)?;
            point.point_enclosure = Some(enclosure);
            enclosure
        }
    };
    Ok(enclosure)
}

fn certify_endpoint_derivative<E, F>(
    point: &mut SearchSample,
    cell_lo: f64,
    cell_hi: f64,
    cell: DerivativeEnclosure,
    enclose: &mut F,
) -> Result<ClosedInterval, ScoreSearchError<E>>
where
    F: FnMut(ScoreSample, ScoreSample) -> Result<DerivativeEnclosure, E>,
{
    let endpoint_derivative = certify_point(point, enclose)?.derivative;
    endpoint_derivative
        .intersection(cell.derivative)
        .ok_or(ScoreSearchError::DisjointEndpointEnclosure {
            lo: cell_lo,
            hi: cell_hi,
            endpoint: point.sample,
            endpoint_derivative,
            enclosure: cell,
        })
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum StrictSign {
    Negative,
    Positive,
}

#[inline]
fn strict_sign(interval: ClosedInterval) -> Option<StrictSign> {
    if interval.hi < 0.0 {
        Some(StrictSign::Negative)
    } else if interval.lo > 0.0 {
        Some(StrictSign::Positive)
    } else {
        None
    }
}

#[inline]
fn is_exact_zero(interval: ClosedInterval) -> bool {
    interval.lo == 0.0 && interval.hi == 0.0
}

fn certify_bracket_score<E, Eval, Enclose>(
    bracket: ClosedInterval,
    representative: SearchSample,
    evaluate: &mut Eval,
    enclose: &mut Enclose,
) -> Result<ScoreValueEnclosure, ScoreSearchError<E>>
where
    Eval: FnMut(f64) -> Result<ScoreJet, E>,
    Enclose: FnMut(ScoreSample, ScoreSample) -> Result<DerivativeEnclosure, E>,
{
    if bracket.lo == bracket.hi {
        let mut representative = representative;
        return Ok(certify_point(&mut representative, enclose)?.score);
    }
    let left = if representative.sample.x == bracket.lo {
        representative
    } else {
        evaluate_sample(bracket.lo, evaluate)?
    };
    let right = if representative.sample.x == bracket.hi {
        representative
    } else {
        evaluate_sample(bracket.hi, evaluate)?
    };
    Ok(checked_enclosure(left.sample, right.sample, enclose)?.score)
}

/// Refine a UNIQUE derivative root.  The caller has already proved uniqueness
/// by a curvature enclosure that excludes zero and supplied endpoint
/// derivative enclosures of opposite sign.
fn refine_unique_root<E, Eval, Enclose>(
    mut left: SearchSample,
    mut right: SearchSample,
    resolution: f64,
    enclosure: DerivativeEnclosure,
    evaluate: &mut Eval,
    enclose: &mut Enclose,
) -> Result<StationaryPoint, ScoreSearchError<E>>
where
    Eval: FnMut(f64) -> Result<ScoreJet, E>,
    Enclose: FnMut(ScoreSample, ScoreSample) -> Result<DerivativeEnclosure, E>,
{
    let bracket_lo = left.sample.x;
    let bracket_hi = right.sample.x;
    let mut left_derivative = certify_endpoint_derivative(
        &mut left,
        bracket_lo,
        bracket_hi,
        enclosure,
        enclose,
    )?;
    let mut right_derivative = certify_endpoint_derivative(
        &mut right,
        bracket_lo,
        bracket_hi,
        enclosure,
        enclose,
    )?;
    if strict_sign(left_derivative).zip(strict_sign(right_derivative)).map(
        |(left_sign, right_sign)| left_sign == right_sign,
    ) != Some(false)
    {
        return Err(ScoreSearchError::InvalidEnclosure {
            lo: left.sample.x,
            hi: right.sample.x,
            enclosure,
        });
    }

    let increasing = enclosure.curvature.lo > 0.0;
    let mut force_midpoint = false;
    while right.sample.x - left.sample.x > resolution {
        let width = right.sample.x - left.sample.x;
        let midpoint = left.sample.x + 0.5 * width;
        if !(midpoint > left.sample.x && midpoint < right.sample.x) {
            return Err(ScoreSearchError::Unresolved {
                lo: left.sample.x,
                hi: right.sample.x,
                requested_resolution: resolution,
                enclosure,
            });
        }

        // Newton is accepted only in the central half of the bracket.  Thus
        // every accepted point, Newton or midpoint, contracts the maintained
        // sign bracket by at least one quarter.  The loop has no iteration cap
        // because its geometric termination follows from this safeguard.
        // Point derivatives are refinement proposals, not proof currency. In
        // particular, a nonzero derivative can round to scalar zero. Rank the
        // Newton anchors by their certified point ranges so that false scalar
        // zeros cannot control either the refinement path or its eventual
        // endpoint representative.
        let base = if left_derivative.max_abs() <= right_derivative.max_abs() {
            left.sample
        } else {
            right.sample
        };
        let newton = if base.curvature != 0.0 {
            base.x - base.derivative / base.curvature
        } else {
            f64::NAN
        };
        let guard = 0.25 * width;
        let x = if !force_midpoint
            && newton.is_finite()
            && newton >= left.sample.x + guard
            && newton <= right.sample.x - guard
        {
            newton
        } else {
            midpoint
        };
        force_midpoint = false;
        if !(x > left.sample.x && x < right.sample.x) {
            return Err(ScoreSearchError::Unresolved {
                lo: left.sample.x,
                hi: right.sample.x,
                requested_resolution: resolution,
                enclosure,
            });
        }
        let mut sample = evaluate_sample(x, evaluate)?;
        let probe_x = sample.sample.x;
        let mut point_derivative = certify_endpoint_derivative(
            &mut sample,
            left.sample.x,
            right.sample.x,
            enclosure,
            enclose,
        )?;
        let mut root_curvature = enclosure.curvature;
        if !is_exact_zero(point_derivative) && strict_sign(point_derivative).is_none() {
            // A degenerate-cell derivative enclosure can remain wide when its
            // analytic formula contains cancellation. The two adjacent cell
            // extensions are independent exact evidence about their shared
            // endpoint. Intersect all three rather than discarding the cell
            // information after merely checking overlap.
            let left_cell = checked_enclosure(left.sample, sample.sample, enclose)?;
            let right_cell = checked_enclosure(sample.sample, right.sample, enclose)?;
            let left_probe_derivative = certify_endpoint_derivative(
                &mut sample,
                left.sample.x,
                probe_x,
                left_cell,
                enclose,
            )?;
            let right_probe_derivative = certify_endpoint_derivative(
                &mut sample,
                probe_x,
                right.sample.x,
                right_cell,
                enclose,
            )?;
            point_derivative = left_probe_derivative
                .intersection(right_probe_derivative)
                .ok_or(ScoreSearchError::DisjointEndpointEnclosure {
                    lo: left.sample.x,
                    hi: right.sample.x,
                    endpoint: sample.sample,
                    endpoint_derivative: left_probe_derivative,
                    enclosure: right_cell,
                })?;
            let child_curvature = left_cell.curvature.hull(right_cell.curvature);
            root_curvature = enclosure
                .curvature
                .intersection(child_curvature)
                .ok_or(ScoreSearchError::InvalidEnclosure {
                    lo: left.sample.x,
                    hi: right.sample.x,
                    enclosure: right_cell,
                })?;
        }
        if is_exact_zero(point_derivative) {
            let bracket = ClosedInterval::point(x);
            let score = certify_bracket_score(bracket, sample, evaluate, enclose)?;
            return Ok(StationaryPoint {
                sample: sample.sample,
                bracket,
                score,
            });
        }
        if let Some(sign) = strict_sign(point_derivative) {
            match (increasing, sign) {
                (true, StrictSign::Negative) | (false, StrictSign::Positive) => {
                    left = sample;
                    left_derivative = point_derivative;
                }
                (true, StrictSign::Positive) | (false, StrictSign::Negative) => {
                    right = sample;
                    right_derivative = point_derivative;
                }
            }
            continue;
        }

        // The point derivative is itself unresolved at f64 precision.  The
        // mean-value theorem and the sign-definite cell curvature still give an
        // interval-Newton enclosure of the exact root:
        //
        //   root = x - f'(x) / f''(ξ),  ξ between x and root.
        let root = ClosedInterval::point(x)
            .sub(point_derivative.div_nonzero(root_curvature))
            .intersection(ClosedInterval::new(left.sample.x, right.sample.x));
        if let Some(root) = root {
            if root.hi - root.lo <= resolution {
                let score = certify_bracket_score(root, sample, evaluate, enclose)?;
                return Ok(StationaryPoint {
                    sample: sample.sample,
                    bracket: root,
                    score,
                });
            }
            if root.lo > left.sample.x || root.hi < right.sample.x {
                let mut new_left = if root.lo == sample.sample.x {
                    sample
                } else {
                    evaluate_sample(root.lo, evaluate)?
                };
                let mut new_right = if root.hi == sample.sample.x {
                    sample
                } else {
                    evaluate_sample(root.hi, evaluate)?
                };
                let new_left_derivative = if new_left.sample.x == sample.sample.x {
                    point_derivative
                } else {
                    certify_endpoint_derivative(
                        &mut new_left,
                        root.lo,
                        root.hi,
                        enclosure,
                        enclose,
                    )?
                };
                let new_right_derivative = if new_right.sample.x == sample.sample.x {
                    point_derivative
                } else {
                    certify_endpoint_derivative(
                        &mut new_right,
                        root.lo,
                        root.hi,
                        enclosure,
                        enclose,
                    )?
                };
                left = new_left;
                right = new_right;
                left_derivative = new_left_derivative;
                right_derivative = new_right_derivative;
                continue;
            }
        }
        if x != midpoint {
            force_midpoint = true;
            continue;
        }
        return Err(ScoreSearchError::Unresolved {
            lo: left.sample.x,
            hi: right.sample.x,
            requested_resolution: resolution,
            enclosure,
        });
    }

    let midpoint = left.sample.x + 0.5 * (right.sample.x - left.sample.x);
    let sample = if midpoint > left.sample.x && midpoint < right.sample.x {
        evaluate_sample(midpoint, evaluate)?.sample
    } else if left_derivative.max_abs() <= right_derivative.max_abs() {
        left.sample
    } else {
        right.sample
    };
    let bracket = ClosedInterval::new(left.sample.x, right.sample.x);
    let representative = SearchSample {
        sample,
        point_enclosure: None,
    };
    let score = certify_bracket_score(bracket, representative, evaluate, enclose)?;
    Ok(StationaryPoint { sample, bracket, score })
}

/// When subdivision lands exactly on a stationary abscissa, a rigorous point
/// interval can contain zero without proving the derivative is exactly zero.
/// Probe symmetrically within one requested-resolution bracket and accept the
/// shared endpoint only if those two certified derivative ranges have opposite
/// signs and the probe-cell curvature proves uniqueness.
fn isolate_shared_endpoint_root<E, Eval, Enclose>(
    endpoint: SearchSample,
    domain_lo: f64,
    domain_hi: f64,
    resolution: f64,
    evaluate: &mut Eval,
    enclose: &mut Enclose,
) -> Result<Option<StationaryPoint>, ScoreSearchError<E>>
where
    Eval: FnMut(f64) -> Result<ScoreJet, E>,
    Enclose: FnMut(ScoreSample, ScoreSample) -> Result<DerivativeEnclosure, E>,
{
    let radius = 0.5 * resolution;
    let left_x = endpoint.sample.x - radius;
    let mut right_x = endpoint.sample.x + radius;
    if !(left_x >= domain_lo
        && right_x <= domain_hi
        && left_x < endpoint.sample.x
        && right_x > endpoint.sample.x)
    {
        return Ok(None);
    }
    while right_x - left_x > resolution {
        right_x = next_down(right_x);
    }
    if !(right_x > endpoint.sample.x && right_x - left_x <= resolution) {
        return Ok(None);
    }

    let mut left = evaluate_sample(left_x, evaluate)?;
    let mut right = evaluate_sample(right_x, evaluate)?;
    let probe_enclosure = checked_enclosure(left.sample, right.sample, enclose)?;
    if probe_enclosure.curvature.contains_zero() {
        return Ok(None);
    }
    let left_derivative =
        certify_endpoint_derivative(&mut left, left_x, right_x, probe_enclosure, enclose)?;
    let right_derivative =
        certify_endpoint_derivative(&mut right, left_x, right_x, probe_enclosure, enclose)?;
    if strict_sign(left_derivative)
        .zip(strict_sign(right_derivative))
        .is_some_and(|(left_sign, right_sign)| left_sign != right_sign)
    {
        Ok(Some(StationaryPoint {
            sample: endpoint.sample,
            bracket: ClosedInterval::new(left_x, right_x),
            score: probe_enclosure.score,
        }))
    } else {
        Ok(None)
    }
}

/// Prove that every score value in a cell is indistinguishable from one of its
/// endpoint samples at the point evaluator's certified f64 resolution.
///
/// If the exact score range is `[L, U]`, every pair of exact scores in the cell
/// differs by at most `U-L`. If each nearest-rounded point value has absolute
/// forward error at most `rho`, a comparison of two such values has uncertainty
/// at most `2 rho`. The cell is resolution-flat only when `U-L <= 2 rho`.
///
/// Both sides are expressed in score-value units and are invariant under
/// adding a constant to the objective. Derivative-evaluator error is
/// deliberately absent: integrating it would bound the error of a hypothetical
/// numerical quadrature, not the forward error of `ScoreJet::value`.
fn resolution_flat_region(
    node: SearchNode,
    enclosure: DerivativeEnclosure,
) -> Option<ResolutionFlatRegion> {
    let score = enclosure.score;
    let max_score_gap = if score.value.lo == score.value.hi {
        0.0
    } else {
        next_up(score.value.hi - score.value.lo)
    };
    let score_resolution = if score.evaluation_error == 0.0 {
        0.0
    } else {
        next_up(2.0 * score.evaluation_error)
    };
    if !(max_score_gap.is_finite() && score_resolution.is_finite()) {
        return None;
    }
    let sample = if node.right.sample.value > node.left.sample.value {
        node.right.sample
    } else {
        node.left.sample
    };
    (max_score_gap <= score_resolution).then_some(
        ResolutionFlatRegion {
            sample,
            bracket: ClosedInterval::new(node.left.sample.x, node.right.sample.x),
            score: score.value,
            max_score_gap,
            score_resolution,
        },
    )
}

/// Select a domain boundary only when one proof cell covers the whole domain
/// and its derivative has one strict sign throughout.
///
/// Rounded endpoint values may tie even when their exact-real ordering is
/// strict. A whole-domain monotonicity certificate resolves that ordering
/// directly. A proper subcell cannot select the global representative because
/// its endpoint has not been compared with maxima in the other cells.
fn certified_domain_boundary(
    node: &SearchNode,
    derivative_sign: StrictSign,
    domain_lo: f64,
    domain_hi: f64,
) -> Option<(ScoreSample, ScoreOptimumLocation)> {
    if node.left.sample.x != domain_lo || node.right.sample.x != domain_hi {
        return None;
    }
    Some(match derivative_sign {
        StrictSign::Positive => (
            node.right.sample,
            ScoreOptimumLocation::UpperBoundary,
        ),
        StrictSign::Negative => (
            node.left.sample,
            ScoreOptimumLocation::LowerBoundary,
        ),
    })
}

/// Globally maximize a smooth score on `[lo, hi]` by certified stationary
/// isolation.
///
/// `evaluate` returns a nearest-rounded score jet at a point. `enclose(a, b)`
/// receives the cell's two ENDPOINT SAMPLES — the jets the search already
/// obtained from `evaluate` — and must return OUTER ranges containing the exact
/// first and second derivative at every point of `[a.x, b.x]`.
///
/// Handing the samples in (rather than the bare abscissae) is what keeps an
/// endpoint-anchored enclosure free: such an oracle is a Taylor pad around the
/// endpoint jets, so with the jets in hand it performs no criterion evaluation
/// of its own. An oracle whose enclosure is a genuine interval extension may
/// ignore the jets and use `a.x`/`b.x`.
///
/// The scalar derivatives are never treated as proofs: when an endpoint sign
/// matters, the search asks `enclose(a, a)` for its exact derivative range.
/// The point and parent-cell ranges must overlap, but the exact-real parent
/// range is intentionally not required to contain a separately rounded scalar
/// estimate.
///
/// A successful return means every stationary interval was excluded, isolated
/// to `resolution`, or proved score-flat at the local representable value
/// resolution. Any interval that satisfies none of those conditions produces
/// [`ScoreSearchError::Unresolved`].
///
/// The traversal is bounded by [`subdivision_budget`]. The per-cell resolution
/// floor bounds the DEPTH of the subdivision and never its BREADTH, and those
/// are different failures. A criterion that certifies NOTHING bottoms out on the
/// floor after `D` subdivisions and is already typed
/// [`ScoreSearchError::Unresolved`]. The unbounded case is the one where cells
/// DO certify, at widths far above the floor, and there are simply too many of
/// them: a criterion whose derivative and curvature enclosures both straddle
/// zero over a wide region excludes no cell by a sign and isolates no root, so
/// every cell it reaches is split until its score range collapses under the
/// evaluator's own error — and the leaf count of that tree is exponential in the
/// depth, 2^32 cells on a 58-wide log-λ domain at `sqrt(eps)` resolution, which
/// is non-termination rather than slowness (#2546). Exceeding the budget is
/// [`ScoreSearchError::SubdivisionBudget`], a statement about the CRITERION and
/// not about the machine: the search was asked to certify more cells than a
/// converging 1-D decomposition at this resolution consists of.
pub fn maximize_score_1d<E, Eval, Enclose>(
    lo: f64,
    hi: f64,
    resolution: f64,
    mut evaluate: Eval,
    mut enclose: Enclose,
) -> Result<ScoreSearchResult, ScoreSearchError<E>>
where
    Eval: FnMut(f64) -> Result<ScoreJet, E>,
    Enclose: FnMut(ScoreSample, ScoreSample) -> Result<DerivativeEnclosure, E>,
{
    if !(lo.is_finite() && hi.is_finite() && lo <= hi && (hi - lo).is_finite()) {
        return Err(ScoreSearchError::InvalidDomain { lo, hi });
    }
    if !(resolution.is_finite() && resolution > 0.0) {
        return Err(ScoreSearchError::InvalidResolution { resolution });
    }

    let lower_boundary = evaluate_sample(lo, &mut evaluate)?;
    if lo == hi {
        let score = checked_enclosure(
            lower_boundary.sample,
            lower_boundary.sample,
            &mut enclose,
        )?
        .score;
        return Ok(ScoreSearchResult {
            optimum: lower_boundary.sample,
            location: ScoreOptimumLocation::LowerBoundary,
            lower_boundary: lower_boundary.sample,
            upper_boundary: lower_boundary.sample,
            stationary_points: Vec::new(),
            resolution_flat_regions: Vec::new(),
            value_certificate: GlobalScoreCertificate {
                selected: score.value,
                maximum: score.value,
                maximum_excess: 0.0,
                comparison_resolution: 0.0,
            },
        });
    }
    let upper_boundary = evaluate_sample(hi, &mut evaluate)?;
    let (mut optimum, mut location) =
        if upper_boundary.sample.value > lower_boundary.sample.value {
            (upper_boundary.sample, ScoreOptimumLocation::UpperBoundary)
        } else {
            (lower_boundary.sample, ScoreOptimumLocation::LowerBoundary)
        };

    let (budget, depth_bound) = subdivision_budget(lo, hi, resolution);
    let mut subdivisions = 0usize;
    let mut stationary_points = Vec::<StationaryPoint>::new();
    let mut resolution_flat_regions = Vec::<ResolutionFlatRegion>::new();
    let mut terminal_maxima = Vec::<TerminalScoreCandidate>::new();
    let mut stack = vec![SearchNode {
        left: lower_boundary,
        right: upper_boundary,
    }];
    while let Some(mut node) = stack.pop() {
        let mathematical_enclosure =
            checked_enclosure(node.left.sample, node.right.sample, &mut enclose)?;
        let enclosure = mathematical_enclosure;
        if !enclosure.derivative.contains_zero() {
            let derivative_sign = if enclosure.derivative.lo > 0.0 {
                StrictSign::Positive
            } else {
                StrictSign::Negative
            };
            if let Some((proven_optimum, proven_location)) =
                certified_domain_boundary(&node, derivative_sign, lo, hi)
            {
                optimum = proven_optimum;
                location = proven_location;
            }
            let endpoint = match derivative_sign {
                StrictSign::Positive => &mut node.right,
                StrictSign::Negative => &mut node.left,
            };
            terminal_maxima.push(TerminalScoreCandidate::point(
                endpoint.sample.x,
                certify_point(endpoint, &mut enclose)?.score,
            ));
            continue;
        }

        let monotone = !enclosure.curvature.contains_zero();
        if monotone {
            let node_lo = node.left.sample.x;
            let node_hi = node.right.sample.x;
            let left_derivative =
                certify_endpoint_derivative(
                    &mut node.left,
                    node_lo,
                    node_hi,
                    enclosure,
                    &mut enclose,
                )?;
            let right_derivative =
                certify_endpoint_derivative(
                    &mut node.right,
                    node_lo,
                    node_hi,
                    enclosure,
                    &mut enclose,
                )?;
            let left_sign = strict_sign(left_derivative);
            let right_sign = strict_sign(right_derivative);
            let stationary = if is_exact_zero(left_derivative) {
                let score = certify_point(&mut node.left, &mut enclose)?.score;
                Some(StationaryPoint {
                    sample: node.left.sample,
                    bracket: ClosedInterval::point(node.left.sample.x),
                    score,
                })
            } else if is_exact_zero(right_derivative) {
                let score = certify_point(&mut node.right, &mut enclose)?.score;
                Some(StationaryPoint {
                    sample: node.right.sample,
                    bracket: ClosedInterval::point(node.right.sample.x),
                    score,
                })
            } else if left_sign.zip(right_sign).is_some_and(
                |(left_sign, right_sign)| left_sign != right_sign,
            ) {
                Some(refine_unique_root(
                    node.left,
                    node.right,
                    resolution,
                    enclosure,
                    &mut evaluate,
                    &mut enclose,
                )?)
            } else if left_sign.is_none() {
                isolate_shared_endpoint_root(
                    node.left,
                    lo,
                    hi,
                    resolution,
                    &mut evaluate,
                    &mut enclose,
                )?
            } else if right_sign.is_none() {
                isolate_shared_endpoint_root(
                    node.right,
                    lo,
                    hi,
                    resolution,
                    &mut evaluate,
                    &mut enclose,
                )?
            } else {
                None
            };

            if let Some(stationary) = stationary {
                // Two adjacent certified cells can report the same exact root
                // when it lies on their common boundary.  Preserve one copy.
                let duplicate = stationary_points
                    .last()
                    .is_some_and(|previous| previous.sample.x == stationary.sample.x);
                if !duplicate {
                    let index = stationary_points.len();
                    if stationary.sample.value > optimum.value {
                        optimum = stationary.sample;
                        location = ScoreOptimumLocation::Stationary(index);
                    }
                    stationary_points.push(stationary);
                }
                if enclosure.curvature.hi < 0.0 {
                    let score = stationary.score;
                    terminal_maxima.push(if stationary.bracket.lo == stationary.bracket.hi {
                        TerminalScoreCandidate::point(stationary.sample.x, score)
                    } else {
                        let mut representative = SearchSample {
                            sample: stationary.sample,
                            point_enclosure: None,
                        };
                        let representative_error =
                            certify_point(&mut representative, &mut enclose)?
                                .score
                                .evaluation_error;
                        TerminalScoreCandidate::region(
                            score,
                            representative_error.max(score.evaluation_error),
                        )
                    });
                } else {
                    terminal_maxima.push(TerminalScoreCandidate::point(
                        node.left.sample.x,
                        certify_point(&mut node.left, &mut enclose)?.score,
                    ));
                    terminal_maxima.push(TerminalScoreCandidate::point(
                        node.right.sample.x,
                        certify_point(&mut node.right, &mut enclose)?.score,
                    ));
                }
                continue;
            }

            // Definite equal endpoint signs plus strict monotonicity exclude a
            // root.  An endpoint range that straddles zero is not silently
            // replaced by the rounded point sign; it proceeds to the value-flat
            // proof or subdivision below.
            if let Some((left_sign, right_sign)) = left_sign.zip(right_sign)
                && left_sign == right_sign
            {
                if let Some((proven_optimum, proven_location)) =
                    certified_domain_boundary(&node, left_sign, lo, hi)
                {
                    optimum = proven_optimum;
                    location = proven_location;
                }
                let endpoint = match left_sign {
                    StrictSign::Positive => &mut node.right,
                    StrictSign::Negative => &mut node.left,
                };
                terminal_maxima.push(TerminalScoreCandidate::point(
                    endpoint.sample.x,
                    certify_point(endpoint, &mut enclose)?.score,
                ));
                continue;
            }
        }

        if let Some(flat) = resolution_flat_region(node, mathematical_enclosure) {
            let index = resolution_flat_regions.len();
            if flat.sample.value > optimum.value {
                optimum = flat.sample;
                location = ScoreOptimumLocation::ResolutionFlat(index);
            }
            terminal_maxima.push(TerminalScoreCandidate::region(
                enclosure.score,
                enclosure.score.evaluation_error,
            ));
            resolution_flat_regions.push(flat);
            continue;
        }

        let width = node.right.sample.x - node.left.sample.x;
        let midpoint = node.left.sample.x + 0.5 * width;
        if width <= resolution
            || !(midpoint > node.left.sample.x && midpoint < node.right.sample.x)
        {
            return Err(ScoreSearchError::Unresolved {
                lo: node.left.sample.x,
                hi: node.right.sample.x,
                requested_resolution: resolution,
                enclosure,
            });
        }
        subdivisions += 1;
        if subdivisions > budget {
            return Err(ScoreSearchError::SubdivisionBudget {
                lo,
                hi,
                cell_lo: node.left.sample.x,
                cell_hi: node.right.sample.x,
                requested_resolution: resolution,
                subdivisions,
                budget,
                depth_bound,
            });
        }
        let middle = evaluate_sample(midpoint, &mut evaluate)?;
        // Right first, then left: the LIFO traversal emits stationary points
        // in ascending x, which makes exact-boundary de-duplication stable.
        stack.push(SearchNode {
            left: middle,
            right: node.right,
        });
        stack.push(SearchNode {
            left: node.left,
            right: middle,
        });
    }

    let mut selected_sample = SearchSample {
        sample: optimum,
        point_enclosure: None,
    };
    let selected_score = certify_point(&mut selected_sample, &mut enclose)?.score;
    let global_lower = terminal_maxima
        .iter()
        .map(|candidate| candidate.score.value.lo)
        .fold(selected_score.value.lo, f64::max);
    let global_upper = terminal_maxima
        .iter()
        .map(|candidate| candidate.score.value.hi)
        .fold(selected_score.value.hi, f64::max);
    let candidate_evaluation_error = terminal_maxima
        .iter()
        .filter(|candidate| candidate.point_x != Some(optimum.x))
        .map(|candidate| candidate.comparison_error)
        .fold(0.0_f64, f64::max);
    let maximum_excess = terminal_maxima
        .iter()
        .filter(|candidate| candidate.point_x != Some(optimum.x))
        .map(|candidate| {
            if candidate.score.value.hi <= selected_score.value.lo {
                0.0
            } else {
                next_up(candidate.score.value.hi - selected_score.value.lo)
            }
        })
        .fold(0.0_f64, f64::max);
    let comparison_resolution = add_nonnegative_upward(
        selected_score.evaluation_error,
        candidate_evaluation_error,
    );

    Ok(ScoreSearchResult {
        optimum,
        location,
        lower_boundary: lower_boundary.sample,
        upper_boundary: upper_boundary.sample,
        stationary_points,
        resolution_flat_regions,
        value_certificate: GlobalScoreCertificate {
            selected: selected_score.value,
            maximum: ClosedInterval::new(global_lower, global_upper),
            maximum_excess,
            comparison_resolution,
        },
    })
}

/// Repeat a certified global score search until its exact winning value is
/// orderable at the evaluator's certified comparison resolution.
///
/// Location resolution and value resolution are different proof currencies:
/// isolating every stationary point to `initial_resolution` can still leave
/// the winning candidate's exact score range wider than the point evaluator's
/// forward-error comparison permits. Each pass here independently rebuilds
/// the complete global certificate at a smaller location target. The observed
/// ratio between maximum excess and comparison resolution is only a refinement
/// strategy; it is never used as acceptance evidence.
///
/// There is no retry cap or acceptance fallback. Each retry contracts the
/// target by at least one binary subdivision. If the next target is no longer
/// representable, or the oracle cannot resolve stationary structure at that
/// finer target, or the finer traversal exceeds its [`subdivision_budget`], the
/// last complete certificate is returned unchanged so the caller can issue its
/// domain-specific typed refusal.
pub fn maximize_score_1d_value_ordered<E, Eval, Enclose>(
    lo: f64,
    hi: f64,
    initial_resolution: f64,
    mut evaluate: Eval,
    mut enclose: Enclose,
) -> Result<ScoreSearchResult, ScoreSearchError<E>>
where
    Eval: FnMut(f64) -> Result<ScoreJet, E>,
    Enclose: FnMut(ScoreSample, ScoreSample) -> Result<DerivativeEnclosure, E>,
{
    let mut resolution = initial_resolution;
    let mut search = maximize_score_1d(
        lo,
        hi,
        resolution,
        &mut evaluate,
        &mut enclose,
    )?;
    loop {
        let certificate = search.value_certificate;
        if certificate.maximum_excess <= certificate.comparison_resolution {
            return Ok(search);
        }
        let binary_refinement = 0.5 * resolution;
        let value_directed_refinement = if certificate.comparison_resolution > 0.0 {
            resolution * (certificate.comparison_resolution / certificate.maximum_excess)
        } else {
            binary_refinement
        };
        let next_resolution = binary_refinement.min(value_directed_refinement);
        if !(next_resolution.is_finite()
            && next_resolution > 0.0
            && next_resolution < resolution)
        {
            return Ok(search);
        }
        match maximize_score_1d(
            lo,
            hi,
            next_resolution,
            &mut evaluate,
            &mut enclose,
        ) {
            Ok(refined) => {
                search = refined;
                resolution = next_resolution;
            }
            // A finer requested location is optional proof strengthening.
            // Preserve the last complete global certificate when the oracle
            // cannot resolve stationary structure at that finer currency; the
            // caller will still reject it if its values remain unordered.
            //
            // A retry that exhausts its subdivision budget is the same kind of
            // outcome and ENDS the loop rather than contracting again: each
            // retry's budget grows as its target shrinks, so continuing past
            // one exhaustion would pay a whole traversal per halving down to
            // the denormal floor — a second unbounded axis (#2546).
            Err(
                ScoreSearchError::Unresolved { .. }
                | ScoreSearchError::SubdivisionBudget { .. },
            ) => return Ok(search),
            Err(error) => return Err(error),
        }
    }
}

/// Static validation or evaluation failure for [`AffineRemlProfile`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AffineRemlError {
    EmptyModes,
    EmptyResponses,
    ShapeMismatch {
        gram_modes: usize,
        penalty_modes: usize,
        projected_rhs_squared: usize,
        responses: usize,
    },
    InvalidMode {
        index: usize,
        gram: f64,
        penalty: f64,
    },
    InvalidProjectedSquare {
        index: usize,
        value: f64,
    },
    InvalidResponseEnergy {
        output: usize,
        value: f64,
    },
    ZeroLambdaResidualUnavailable {
        output: usize,
    },
    InvalidResidualDof {
        value: f64,
    },
    InvalidLogdetConstant {
        value: f64,
    },
    RankMismatch {
        supplied: usize,
        inferred: usize,
    },
    InvalidLogLambda {
        value: f64,
    },
    InvalidLogLambdaInterval {
        lo: f64,
        hi: f64,
    },
    ElementaryEnclosureUnavailable {
        function: &'static str,
        lo: f64,
        hi: f64,
    },
    NonPositiveMode {
        index: usize,
        log_lambda: f64,
        value: f64,
    },
    NonPositiveResidual {
        output: usize,
        log_lambda: f64,
        value: f64,
    },
    NonPositiveResidualInterval {
        output: usize,
        lo: f64,
        hi: f64,
        lower_bound: f64,
    },
    InconsistentResidualEnclosures {
        output: usize,
        lo: f64,
        hi: f64,
        direct: ClosedInterval,
        complement: ClosedInterval,
    },
    UnboundedScoreEvaluationError {
        lo: f64,
        hi: f64,
        error: f64,
    },
}

impl fmt::Display for AffineRemlError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyModes => write!(f, "affine REML profile has no modes"),
            Self::EmptyResponses => write!(f, "affine REML profile has no responses"),
            Self::ShapeMismatch {
                gram_modes,
                penalty_modes,
                projected_rhs_squared,
                responses,
            } => write!(
                f,
                "affine REML profile shape mismatch: gram {gram_modes}, penalty {penalty_modes}, projected squares {projected_rhs_squared}, responses {responses}"
            ),
            Self::InvalidMode {
                index,
                gram,
                penalty,
            } => write!(
                f,
                "affine REML mode {index} must have finite nonnegative (g,s), not both zero; got ({gram}, {penalty})"
            ),
            Self::InvalidProjectedSquare { index, value } => write!(
                f,
                "affine REML projected square {index} must be finite and nonnegative, got {value}"
            ),
            Self::InvalidResponseEnergy { output, value } => write!(
                f,
                "affine REML response energy {output} must be finite and nonnegative, got {value}"
            ),
            Self::ZeroLambdaResidualUnavailable { output } => write!(
                f,
                "affine REML could not certify the zero-smoothing residual for response {output}"
            ),
            Self::InvalidResidualDof { value } => {
                write!(
                    f,
                    "affine REML residual dof must be finite and positive, got {value}"
                )
            }
            Self::InvalidLogdetConstant { value } => write!(
                f,
                "affine REML log-determinant constant must be finite, got {value}"
            ),
            Self::RankMismatch { supplied, inferred } => write!(
                f,
                "affine REML determinant rank {supplied} disagrees with {inferred} positive penalty modes"
            ),
            Self::InvalidLogLambda { value } => {
                write!(f, "affine REML invalid log lambda {value}")
            }
            Self::InvalidLogLambdaInterval { lo, hi } => {
                write!(f, "affine REML invalid log-lambda interval [{lo}, {hi}]")
            }
            Self::ElementaryEnclosureUnavailable { function, lo, hi } => write!(
                f,
                "affine REML has no finite source-derived {function} enclosure on [{lo}, {hi}]"
            ),
            Self::NonPositiveMode {
                index,
                log_lambda,
                value,
            } => write!(
                f,
                "affine REML mode {index} is nonpositive at log lambda {log_lambda}: {value}"
            ),
            Self::NonPositiveResidual {
                output,
                log_lambda,
                value,
            } => write!(
                f,
                "affine REML residual {output} is nonpositive at log lambda {log_lambda}: {value}"
            ),
            Self::NonPositiveResidualInterval {
                output,
                lo,
                hi,
                lower_bound,
            } => write!(
                f,
                "affine REML residual {output} is not certified positive on [{lo}, {hi}] (lower bound {lower_bound})"
            ),
            Self::InconsistentResidualEnclosures {
                output,
                lo,
                hi,
                direct,
                complement,
            } => write!(
                f,
                "affine REML residual {output} has disjoint direct {direct:?} and zero-smoothing-complement {complement:?} enclosures on [{lo}, {hi}]"
            ),
            Self::UnboundedScoreEvaluationError { lo, hi, error } => write!(
                f,
                "affine REML score evaluator has no finite forward-error bound on [{lo}, {hi}] (bound {error})"
            ),
        }
    }
}

impl std::error::Error for AffineRemlError {}

/// Spectral REML/profile score with affine diagonal modes
/// `h_i(lambda) = g_i + lambda s_i`.
///
/// `projected_rhs_squared` is RESPONSE-MAJOR: entry `(d, i)` is stored at
/// `d * n_modes + i`.  The score is
///
/// `-1/2 { D [logdet_constant + sum log h_i - rank log(lambda)]
///          + residual_dof * sum_d log(R_d / residual_dof) }`,
///
/// where `R_d = response_energy[d] - sum_i q[d,i] / h_i`.
#[derive(Clone, Debug)]
pub struct AffineRemlProfile<'a> {
    gram_modes: &'a [f64],
    penalty_modes: &'a [f64],
    projected_rhs_squared: &'a [f64],
    response_energy: &'a [f64],
    /// Exact-real residuals on the finite part of the zero-smoothing face,
    ///
    /// `response_energy[d] - sum_{i:g_i>0} q[d,i] / g_i`.
    ///
    /// These invariants are computed once with an error-free leading sum and
    /// FMA-certified division corrections. Re-forming them independently in
    /// every interval evaluation loses the small Schur complement to the
    /// rounding scale of its O(energy) operands.
    zero_lambda_residual: Vec<ClosedInterval>,
    residual_dof: f64,
    logdet_constant: f64,
}

/// O(n) exact-leading accumulator.
///
/// `leading + correction` encloses the exact sum of every value submitted so
/// far. `leading` follows the ordinary binary64 accumulation path. Knuth's
/// TwoSum identity moves each discarded low part into `correction`, whose
/// directed interval accumulation never again mixes it with an O(energy)
/// operand. This is the fixed-size analogue of a floating-point expansion:
/// exact proof information, without an O(n²) expansion walk or arbitrary
/// precision dependency.
struct CertifiedCompensatedSum {
    leading: f64,
    correction: ClosedInterval,
}

impl CertifiedCompensatedSum {
    fn new(value: f64) -> Self {
        Self {
            leading: value,
            correction: ClosedInterval::point(0.0),
        }
    }

    /// Add one exact binary64 value, retaining the exact TwoSum residual.
    fn add_exact(&mut self, value: f64) -> bool {
        let sum = self.leading + value;
        if !sum.is_finite() {
            return false;
        }
        let virtual_value = sum - self.leading;
        let virtual_leading = sum - virtual_value;
        let value_residual = value - virtual_value;
        let leading_residual = self.leading - virtual_leading;
        // Under round-to-nearest with gradual underflow, the two residuals and
        // their final sum are exact (Knuth/Møller TwoSum).
        let error = leading_residual + value_residual;
        self.leading = sum;
        self.correction = self.correction.add(ClosedInterval::point(error));
        self.correction.is_valid()
    }

    fn subtract_interval(&mut self, value: ClosedInterval) -> bool {
        self.correction = self.correction.sub(value);
        self.correction.is_valid()
    }

    fn enclosure(self) -> Option<ClosedInterval> {
        let enclosure = ClosedInterval::point(self.leading).add(self.correction);
        (enclosure.is_valid()
            && enclosure.lo.is_finite()
            && enclosure.hi.is_finite())
        .then_some(enclosure)
    }
}

/// Split one exact-real positive quotient into a binary64 leading value and a
/// rigorous low correction:
///
/// `numerator / denominator ∈ leading + correction`.
///
/// The fused residual `numerator - leading*denominator` rounds only once.
/// Its two adjacent binary64 values therefore enclose the exact residual; a
/// directed division by the positive denominator transports that enclosure
/// into quotient units. The correction is normally O(u²) relative to the
/// quotient, rather than the O(u) width of independently directed division.
fn quotient_leading_and_correction(
    numerator: f64,
    denominator: f64,
) -> Option<(f64, ClosedInterval)> {
    if numerator == 0.0 {
        return Some((0.0, ClosedInterval::point(0.0)));
    }
    if !(numerator.is_finite()
        && numerator > 0.0
        && denominator.is_finite()
        && denominator > 0.0)
    {
        return None;
    }
    let leading = numerator / denominator;
    if !(leading.is_finite() && leading >= 0.0) {
        return None;
    }
    if denominator == 1.0 {
        return Some((leading, ClosedInterval::point(0.0)));
    }
    let fused_residual = (-leading).mul_add(denominator, numerator);
    if !fused_residual.is_finite() {
        return None;
    }
    let exact_residual = ClosedInterval::new(
        next_down(fused_residual),
        next_up(fused_residual),
    );
    let correction =
        exact_residual.div_positive(ClosedInterval::point(denominator));
    (correction.is_valid()
        && correction.lo.is_finite()
        && correction.hi.is_finite())
    .then_some((leading, correction))
}

fn certified_zero_lambda_residual(
    energy: f64,
    gram_modes: &[f64],
    projected_squares: &[f64],
) -> Option<ClosedInterval> {
    let mut residual = CertifiedCompensatedSum::new(energy);
    for (&gram, &projected_square) in
        gram_modes.iter().zip(projected_squares)
    {
        if gram == 0.0 || projected_square == 0.0 {
            continue;
        }
        let (leading, correction) =
            quotient_leading_and_correction(projected_square, gram)?;
        if !(residual.add_exact(-leading)
            && residual.subtract_interval(correction))
        {
            return None;
        }
    }
    residual.enclosure()
}

// Operation counts in the scalar evaluator's per-mode accumulators.  They are
// kept beside the profile rather than written as anonymous roundoff factors:
// determinant value = at most exponential, multiply, divide, two logarithms,
// two additions/subtractions, and accumulator update;
// determinant first = fused h, the cancellation-free complement g/h, and sum;
// determinant second adds u and the product;
// residual value = at most two ratio divisions, scaling, and subtraction;
// residual first adds numerator product, division, and sum to the `u` path;
// residual second additionally forms `2u`, `1-2u`, and its product.
const DETERMINANT_VALUE_OPS_PER_MODE: usize = 8;
const RESIDUAL_VALUE_OPS_PER_MODE: usize = 4;
const RESIDUAL_LOG_OPS_PER_RESPONSE: usize = 3;
const SCORE_COMBINE_OPS: usize = 4;

impl<'a> AffineRemlProfile<'a> {
    pub fn new(
        gram_modes: &'a [f64],
        penalty_modes: &'a [f64],
        projected_rhs_squared: &'a [f64],
        response_energy: &'a [f64],
        residual_dof: f64,
        determinant_rank: usize,
        logdet_constant: f64,
    ) -> Result<Self, AffineRemlError> {
        let modes = gram_modes.len();
        let responses = response_energy.len();
        if modes == 0 {
            return Err(AffineRemlError::EmptyModes);
        }
        if responses == 0 {
            return Err(AffineRemlError::EmptyResponses);
        }
        if penalty_modes.len() != modes
            || projected_rhs_squared.len() != modes.saturating_mul(responses)
        {
            return Err(AffineRemlError::ShapeMismatch {
                gram_modes: modes,
                penalty_modes: penalty_modes.len(),
                projected_rhs_squared: projected_rhs_squared.len(),
                responses,
            });
        }
        for (index, (&gram, &penalty)) in gram_modes.iter().zip(penalty_modes).enumerate() {
            if !(gram.is_finite()
                && penalty.is_finite()
                && gram >= 0.0
                && penalty >= 0.0
                && (gram > 0.0 || penalty > 0.0))
            {
                return Err(AffineRemlError::InvalidMode {
                    index,
                    gram,
                    penalty,
                });
            }
        }
        for (index, &value) in projected_rhs_squared.iter().enumerate() {
            if !(value.is_finite() && value >= 0.0) {
                return Err(AffineRemlError::InvalidProjectedSquare { index, value });
            }
        }
        for (output, &value) in response_energy.iter().enumerate() {
            if !(value.is_finite() && value >= 0.0) {
                return Err(AffineRemlError::InvalidResponseEnergy { output, value });
            }
        }
        if !(residual_dof.is_finite() && residual_dof > 0.0) {
            return Err(AffineRemlError::InvalidResidualDof {
                value: residual_dof,
            });
        }
        if !logdet_constant.is_finite() {
            return Err(AffineRemlError::InvalidLogdetConstant {
                value: logdet_constant,
            });
        }
        let inferred_rank = penalty_modes.iter().filter(|&&value| value > 0.0).count();
        if determinant_rank != inferred_rank {
            return Err(AffineRemlError::RankMismatch {
                supplied: determinant_rank,
                inferred: inferred_rank,
            });
        }
        let mut zero_lambda_residual = Vec::with_capacity(responses);
        for (output, &energy) in response_energy.iter().enumerate() {
            let start = output * modes;
            let end = start + modes;
            zero_lambda_residual.push(
                certified_zero_lambda_residual(
                    energy,
                    gram_modes,
                    &projected_rhs_squared[start..end],
                )
                .ok_or(
                    AffineRemlError::ZeroLambdaResidualUnavailable { output },
                )?,
            );
        }
        Ok(Self {
            gram_modes,
            penalty_modes,
            projected_rhs_squared,
            response_energy,
            zero_lambda_residual,
            residual_dof,
            logdet_constant,
        })
    }

    #[inline]
    pub fn num_modes(&self) -> usize {
        self.gram_modes.len()
    }

    #[inline]
    pub fn num_responses(&self) -> usize {
        self.response_energy.len()
    }

    /// Nearest-rounded score value, first derivative, and second derivative in
    /// `log(lambda)`. [`Self::enclose`] supplies the proof-grade outer ranges.
    pub fn evaluate(&self, log_lambda: f64) -> Result<ScoreJet, AffineRemlError> {
        if !log_lambda.is_finite() {
            return Err(AffineRemlError::InvalidLogLambda { value: log_lambda });
        }
        let lambda = certified_exp_representative(log_lambda)
            .ok_or(AffineRemlError::InvalidLogLambda { value: log_lambda })?;
        if !(lambda.is_finite() && lambda > 0.0) {
            return Err(AffineRemlError::InvalidLogLambda { value: log_lambda });
        }

        let mut normalized_logdet = self.logdet_constant;
        let mut determinant_derivative = 0.0;
        let mut determinant_curvature = 0.0;
        let exp_neg_log_lambda = if log_lambda >= 0.0 {
            certified_exp_representative(-log_lambda)
        } else {
            None
        };
        for (index, (&gram, &penalty)) in self.gram_modes.iter().zip(self.penalty_modes).enumerate()
        {
            // A gram-zero penalized mode is structurally
            //
            //   log(exp(rho) s) - rho = log(s),
            //
            // with first and second derivatives exactly zero. Do not form
            // `exp(rho) s`: its rounded product may be zero or infinity even
            // though every normalized determinant quantity is finite.
            if gram == 0.0 {
                normalized_logdet +=
                    certified_ln_value(penalty).ok_or(AffineRemlError::NonPositiveMode {
                        index,
                        log_lambda,
                        value: penalty,
                    })?;
                continue;
            }
            let h = lambda.mul_add(penalty, gram);
            if !(h.is_finite() && h > 0.0) {
                return Err(AffineRemlError::NonPositiveMode {
                    index,
                    log_lambda,
                    value: h,
                });
            }
            let u = lambda * penalty / h;
            // For a penalized mode,
            //
            //   d/d rho [log(g + exp(rho)s) - rho]
            //     = u - 1 = -g/h,
            //
            // and its second derivative is u*g/h. Accumulate in that
            // cancellation-free complement currency rather than adding u to a
            // separately rounded `-rank`; the latter loses the derivative's
            // sign when u rounds to one. An unpenalized mode has no `-rho`
            // normalization and contributes exactly zero.
            let determinant_complement = if penalty == 0.0 {
                0.0
            } else {
                gram / h
            };
            // Accumulate the determinant in the normalized per-mode form
            // instead of forming two O(rho) quantities and subtracting them
            // after the sum. Both branches keep their exponential in (0, 1]:
            //
            // log(g + exp(rho)s) - rho
            //   = log(s + g exp(-rho))                         rho >= 0
            //   = log(g) - rho + log1p(exp(rho)s/g)            g dominates
            //   = log(s) + log1p(g/(exp(rho)s))                s dominates.
            //
            // The selected log1p ratio is always in [0, 1], so neither tail
            // forms an overflowing exponential or divides by its small term.
            let normalized_mode = if penalty == 0.0 {
                certified_ln_value(gram)
            } else if log_lambda >= 0.0 {
                exp_neg_log_lambda
                    .and_then(|exp_neg_rho| certified_ln_value(penalty + gram * exp_neg_rho))
            } else if gram >= penalty * lambda {
                certified_ln_value(gram).zip(certified_ln_1p_value(penalty * lambda / gram))
                    .map(|(log_gram, correction)| log_gram - log_lambda + correction)
            } else {
                certified_ln_value(penalty)
                    .zip(certified_ln_1p_value(gram / (penalty * lambda)))
                    .map(|(log_penalty, correction)| log_penalty + correction)
            }
            .ok_or(AffineRemlError::NonPositiveMode {
                index,
                log_lambda,
                value: h,
            })?;
            normalized_logdet += normalized_mode;
            determinant_derivative -= determinant_complement;
            determinant_curvature += u * determinant_complement;
        }

        let modes = self.num_modes();
        let mut residual_log_sum = 0.0;
        let mut residual_derivative_sum = 0.0;
        let mut residual_curvature_sum = 0.0;
        for (output, &energy) in self.response_energy.iter().enumerate() {
            let mut residual = energy;
            let mut first = 0.0;
            let mut second = 0.0;
            for i in 0..modes {
                let projected_square = self.projected_rhs_squared[output * modes + i];
                if projected_square == 0.0 {
                    continue;
                }
                if self.gram_modes[i] == 0.0 {
                    let fitted = positive_ratio_over_product(
                        projected_square,
                        self.penalty_modes[i],
                        lambda,
                    )
                    .ok_or(AffineRemlError::ElementaryEnclosureUnavailable {
                        function: "gram-zero residual quotient",
                        lo: log_lambda,
                        hi: log_lambda,
                    })?;
                    residual -= fitted;
                    first += fitted;
                    second -= fitted;
                    continue;
                }
                let h = lambda.mul_add(self.penalty_modes[i], self.gram_modes[i]);
                let u = lambda * self.penalty_modes[i] / h;
                residual -= projected_square / h;
                first += projected_square * u / h;
                second += projected_square * u * (1.0 - 2.0 * u) / h;
            }
            if !(residual.is_finite() && residual > 0.0) {
                return Err(AffineRemlError::NonPositiveResidual {
                    output,
                    log_lambda,
                    value: residual,
                });
            }
            let log_derivative = first / residual;
            residual_log_sum += certified_ln_value(residual / self.residual_dof).ok_or(
                AffineRemlError::NonPositiveResidual {
                    output,
                    log_lambda,
                    value: residual,
                },
            )?;
            residual_derivative_sum += log_derivative;
            residual_curvature_sum += second / residual - log_derivative * log_derivative;
        }

        let outputs = self.num_responses() as f64;
        Ok(ScoreJet {
            value: -0.5
                * (outputs * normalized_logdet + self.residual_dof * residual_log_sum),
            derivative: -0.5
                * (outputs * determinant_derivative + self.residual_dof * residual_derivative_sum),
            curvature: -0.5
                * (outputs * determinant_curvature + self.residual_dof * residual_curvature_sum),
            // This profile's companion `enclose` is a true interval extension
            // over the whole cell (it evaluates the mode kernels on interval
            // lambda), not an endpoint-anchored Taylor pad, so it never reads
            // the endpoint third derivative and none is computed here.
            third: 0.0,
        })
    }

    /// Outward enclosure of the score value and first two derivatives on a
    /// bounded log-lambda interval.
    ///
    /// The interval kernels enclose the exact-real ranges. The score value uses
    /// the same cancellation-free normalized determinant identity as
    /// [`Self::evaluate`]. Its separate `evaluation_error` charges each
    /// source-derived elementary-function interval, error propagation through
    /// `log(residual)`, and Wilkinson `gamma_k * sum |term|` bounds for the
    /// actual sequential accumulators.
    pub fn enclose(&self, lo: f64, hi: f64) -> Result<DerivativeEnclosure, AffineRemlError> {
        if !(lo.is_finite() && hi.is_finite() && lo <= hi) {
            return Err(AffineRemlError::InvalidLogLambdaInterval { lo, hi });
        }
        let lambda = exp_interval(lo, hi)?;
        if !(lambda.lo.is_finite() && lambda.lo > 0.0 && lambda.hi.is_finite()) {
            return Err(AffineRemlError::InvalidLogLambdaInterval { lo, hi });
        }
        // Exp's range-reduction, arithmetic, and truncation errors are
        // multiplicative and must remain in relative currency across a wide
        // interval. Only gradual underflow is additive and is charged against
        // the certified positive lower endpoint. This avoids coupling an
        // upper-endpoint absolute error to the lower-endpoint scale.
        let lambda_relative_error =
            certified_exp_relative_forward_error(ClosedInterval::new(lo, hi), lambda);
        if !lambda_relative_error.is_finite() {
            return Err(AffineRemlError::UnboundedScoreEvaluationError {
                lo,
                hi,
                error: lambda_relative_error,
            });
        }

        let mut normalized_logdet = ClosedInterval::point(self.logdet_constant);
        let mut normalized_logdet_magnitude = self.logdet_constant.abs();
        let mut normalized_logdet_error = 0.0;
        let mut determinant_first = ClosedInterval::point(0.0);
        let mut determinant_second = ClosedInterval::point(0.0);
        for i in 0..self.num_modes() {
            let (normalized_mode, normalized_mode_error) = normalized_log_mode_enclosure(
                self.gram_modes[i],
                self.penalty_modes[i],
                lo,
                hi,
            )?;
            normalized_logdet = normalized_logdet.add(normalized_mode);
            normalized_logdet_magnitude = add_nonnegative_upward(
                normalized_logdet_magnitude,
                add_nonnegative_upward(normalized_mode.max_abs(), normalized_mode_error),
            );
            normalized_logdet_error =
                add_nonnegative_upward(normalized_logdet_error, normalized_mode_error);

            let ranges =
                mode_ranges(self.gram_modes[i], self.penalty_modes[i], 0.0, lambda)?;
            determinant_first = determinant_first.sub(ranges.c);
            determinant_second = determinant_second.add(ranges.w);
        }

        let mut residual_first_sum = ClosedInterval::point(0.0);
        let mut residual_second_sum = ClosedInterval::point(0.0);
        let mut residual_log_sum = ClosedInterval::point(0.0);
        let mut residual_log_magnitude = 0.0;
        let mut residual_log_error = 0.0;
        let modes = self.num_modes();
        for (output, &energy) in self.response_energy.iter().enumerate() {
            let mut fitted_quadratic = ClosedInterval::point(0.0);
            let mut smoothing_increment = ClosedInterval::point(0.0);
            let mut singular_fitted = ClosedInterval::point(0.0);
            let mut first = ClosedInterval::point(0.0);
            let mut second = ClosedInterval::point(0.0);
            let mut fitted_magnitude = energy;
            for i in 0..modes {
                let ranges = mode_ranges(
                    self.gram_modes[i],
                    self.penalty_modes[i],
                    self.projected_rhs_squared[output * modes + i],
                    lambda,
                )?;
                fitted_quadratic = fitted_quadratic.add(ranges.v);
                smoothing_increment =
                    smoothing_increment.add(ranges.smoothing_increment);
                singular_fitted = singular_fitted.add(ranges.singular_fitted);
                first = first.add(ranges.p);
                second = second.add(ranges.q);
                fitted_magnitude =
                    add_nonnegative_upward(fitted_magnitude, ranges.v.max_abs());
            }
            // Two exact identities describe the same residual:
            //
            //   R = E - sum_i q_i/(g_i + lambda*s_i)
            //
            // and, for every positive-Gram mode,
            //
            //   q_i/(g_i + lambda*s_i)
            //     = q_i/g_i - (q_i/g_i) * lambda*s_i/(g_i + lambda*s_i).
            //
            // The direct form is well-conditioned away from interpolation.
            // Near the zero-smoothing face it subtracts many independently
            // rounded near-one fitted fractions from `E`, even though their
            // deviations from one are perfectly correlated with lambda.  The
            // complement form carries that correlation explicitly as a
            // fixed zero-smoothing residual plus nonnegative smoothing
            // increments.  Both are rigorous outer enclosures, so their
            // intersection is rigorous and never an acceptance tolerance.
            let direct_residual = ClosedInterval::point(energy).sub(fitted_quadratic);
            let complement_residual = self.zero_lambda_residual[output]
                .add(smoothing_increment)
                .sub(singular_fitted);
            let residual = direct_residual.intersection(complement_residual).ok_or(
                AffineRemlError::InconsistentResidualEnclosures {
                    output,
                    lo,
                    hi,
                    direct: direct_residual,
                    complement: complement_residual,
                },
            )?;
            if !(residual.lo > 0.0 && residual.is_valid()) {
                return Err(AffineRemlError::NonPositiveResidualInterval {
                    output,
                    lo,
                    hi,
                    lower_bound: residual.lo,
                });
            }
            let first_ratio = first.div_positive(residual).nonnegative();
            let second_ratio = second.div_positive(residual);
            residual_first_sum = residual_first_sum.add(first_ratio);
            residual_second_sum = residual_second_sum.add(second_ratio.sub(first_ratio.square()));

            let fitted_arithmetic_error = wilkinson_roundoff(
                fitted_magnitude,
                modes.saturating_mul(RESIDUAL_VALUE_OPS_PER_MODE),
            );
            // `first = d fitted/d rho`, so the MVT propagates the exp error in
            // rho-space without a condition-number guess.
            let fitted_exp_error =
                next_up(first.max_abs() * lambda_relative_error);
            let resolved_fitted_quadratic = fitted_quadratic.widen(
                add_nonnegative_upward(fitted_arithmetic_error, fitted_exp_error),
            );
            let resolved_residual =
                ClosedInterval::point(energy).sub(resolved_fitted_quadratic);
            if !(resolved_residual.lo > 0.0 && resolved_residual.is_valid()) {
                return Err(AffineRemlError::NonPositiveResidualInterval {
                    output,
                    lo,
                    hi,
                    lower_bound: resolved_residual.lo,
                });
            }
            let residual_over_dof =
                residual.div_positive(ClosedInterval::point(self.residual_dof));
            if !(residual_over_dof.lo > 0.0 && residual_over_dof.hi.is_finite()) {
                return Err(AffineRemlError::ElementaryEnclosureUnavailable {
                    function: "ln",
                    lo: residual_over_dof.lo,
                    hi: residual_over_dof.hi,
                });
            }
            let residual_log = residual_over_dof.ln_positive();
            residual_log_sum = residual_log_sum.add(residual_log);

            // `evaluate` first forms the residual and then takes
            // `ln(residual/dof)`. The residual forward error is already
            // represented by `resolved_residual`. On the strictly positive
            // resolved range, the mean-value theorem bounds its propagation
            // through log by `delta_R / min(R)`. The division and logarithm
            // each add one directed basic-operation contribution; the
            // source-derived logarithm error is absolute, so it remains valid
            // when the logarithm's result is near zero.
            let residual_error = enclosure_excess(residual, resolved_residual);
            let propagated_residual_error = next_up(residual_error / resolved_residual.lo);
            let elementary_error = certified_log_forward_error(
                residual.div_positive(ClosedInterval::point(self.residual_dof)),
            );
            let local_log_error = add_nonnegative_upward(
                propagated_residual_error,
                add_nonnegative_upward(
                    elementary_error,
                    wilkinson_roundoff(
                        add_nonnegative_upward(1.0, residual_log.max_abs()),
                        RESIDUAL_LOG_OPS_PER_RESPONSE,
                    ),
                ),
            );
            residual_log_error =
                add_nonnegative_upward(residual_log_error, local_log_error);
            residual_log_magnitude = add_nonnegative_upward(
                residual_log_magnitude,
                add_nonnegative_upward(residual_log.max_abs(), local_log_error),
            );
        }

        let outputs = self.num_responses() as f64;
        let first_bracket = determinant_first
            .scale(outputs)
            .add(residual_first_sum.scale(self.residual_dof));
        let second_bracket = determinant_second
            .scale(outputs)
            .add(residual_second_sum.scale(self.residual_dof));
        let derivative = first_bracket.scale(-0.5);
        let curvature = second_bracket.scale(-0.5);
        let score_value = normalized_logdet
            .scale(outputs)
            .add(residual_log_sum.scale(self.residual_dof))
            .scale(-0.5);
        let score_magnitude = add_nonnegative_upward(
            next_up(outputs * normalized_logdet_magnitude),
            next_up(self.residual_dof * residual_log_magnitude),
        );
        normalized_logdet_error = add_nonnegative_upward(
            normalized_logdet_error,
            wilkinson_roundoff(normalized_logdet_magnitude, self.num_modes()),
        );
        residual_log_error = add_nonnegative_upward(
            residual_log_error,
            wilkinson_roundoff(residual_log_magnitude, self.num_responses()),
        );
        let final_arithmetic_error =
            wilkinson_roundoff(score_magnitude, SCORE_COMBINE_OPS);
        let weighted_component_error = add_nonnegative_upward(
            next_up(outputs * normalized_logdet_error),
            next_up(self.residual_dof * residual_log_error),
        );
        let value_evaluation_error = next_up(
            0.5
                * add_nonnegative_upward(
                    weighted_component_error,
                    final_arithmetic_error,
                ),
        );
        if !(score_value.is_valid() && value_evaluation_error.is_finite()) {
            return Err(AffineRemlError::UnboundedScoreEvaluationError {
                lo,
                hi,
                error: value_evaluation_error,
            });
        }
        let score = ScoreValueEnclosure {
            value: score_value,
            evaluation_error: value_evaluation_error,
        };
        Ok(DerivativeEnclosure {
            score,
            derivative,
            curvature,
        })
    }

    pub fn maximize(
        &self,
        lo: f64,
        hi: f64,
        resolution: f64,
    ) -> Result<ScoreSearchResult, ScoreSearchError<AffineRemlError>> {
        maximize_score_1d(
            lo,
            hi,
            resolution,
            |x| self.evaluate(x),
            |a, b| self.enclose(a.x, b.x),
        )
    }

    /// Isolate every finite stationary candidate and tighten location
    /// resolution until the selected exact score is globally orderable at the
    /// point evaluator's certified comparison resolution.
    ///
    /// The first pass honors the caller's requested location resolution.  A
    /// successful root isolation can still leave a wider exact score range
    /// than the rounded point comparison can distinguish, because location and
    /// value are different currencies.  In that case this repeats the same
    /// exact search with a smaller location target.  The observed ratio between
    /// comparison resolution and maximum excess is only an iteration strategy,
    /// never proof currency; every pass independently rebuilds the complete
    /// global certificate and the loop exits only on its verdict.
    ///
    /// There is no retry cap or acceptance fallback.  Each retry contracts the
    /// target by at least one binary subdivision.  If the target can no longer
    /// be represented, or the oracle cannot resolve structure at that finer
    /// target, the last complete certificate is returned unchanged for the
    /// caller's existing typed refusal.
    pub fn maximize_value_ordered(
        &self,
        lo: f64,
        hi: f64,
        initial_resolution: f64,
    ) -> Result<ScoreSearchResult, ScoreSearchError<AffineRemlError>> {
        maximize_score_1d_value_ordered(
            lo,
            hi,
            initial_resolution,
            |x| self.evaluate(x),
            |a, b| self.enclose(a.x, b.x),
        )
    }
}

#[derive(Clone, Copy)]
struct ModeRanges {
    /// Cancellation-free determinant complement `c = g/h` for a penalized
    /// mode. An unpenalized mode contributes exactly zero because its
    /// normalized log determinant has no `-rho` term.
    c: ClosedInterval,
    /// `u(1-u)`.
    w: ClosedInterval,
    /// `projected_square / h`.
    v: ClosedInterval,
    /// The nonnegative loss of fitted energy caused by smoothing,
    /// `(projected_square / gram) * lambda*penalty / h`, for a positive Gram
    /// mode.
    smoothing_increment: ClosedInterval,
    /// The complete fitted contribution of a Gram-zero mode. Such a mode
    /// cannot participate in the zero-smoothing complement identity.
    singular_fitted: ClosedInterval,
    /// First derivative of the residual contribution:
    /// `projected_square * lambda s / h^2`.
    p: ClosedInterval,
    /// Second derivative of the residual contribution:
    /// `projected_square * lambda s (g-lambda s) / h^3`.
    q: ClosedInterval,
}

/// Exact-real range and a uniform forward-error bound for the normalized
/// determinant contribution of one affine mode.
///
/// The exact function is monotone, so outward endpoint evaluation gives its
/// range. [`AffineRemlProfile::evaluate`] uses algebraically equivalent stable
/// sign/dominance branches whose exponential or `ln_1p` argument is in
/// `[0, 1]`. The elementary-function bounds come from the source-derived
/// range-reduced series above, not a platform-libm accuracy assumption;
/// Wilkinson's bound charges the surrounding IEEE basic operations and the
/// elementary input perturbation. The leading `1` is the analytic sensitivity
/// bound: the mode's rho derivative lies in `[-1, 0]`.
fn normalized_log_mode_enclosure(
    gram: f64,
    penalty: f64,
    lo: f64,
    hi: f64,
) -> Result<(ClosedInterval, f64), AffineRemlError> {
    if penalty == 0.0 {
        let range = ClosedInterval::point(gram).ln_positive();
        return Ok((
            range,
            certified_log_forward_error(ClosedInterval::point(gram)),
        ));
    }
    if gram == 0.0 {
        let range = ClosedInterval::point(penalty).ln_positive();
        return Ok((
            range,
            certified_log_forward_error(ClosedInterval::point(penalty)),
        ));
    }

    let at_lo = normalized_log_mode_at(gram, penalty, lo)?;
    let at_hi = normalized_log_mode_at(gram, penalty, hi)?;
    // The normalized contribution has derivative `u - 1` in [-1, 0].
    let range = ClosedInterval::new(at_hi.lo, at_lo.hi);
    // In the negative branch the final subtraction can cancel `log(h)` and
    // `rho`; charge both pre-cancellation operands. Since
    // `log(h) = normalized_mode + rho`, `|mode| + 2|rho|` bounds their absolute
    // sum without evaluating a second logarithm.
    let negative_rho_abs = if lo < 0.0 { -lo } else { 0.0 };
    let arithmetic_scale = add_nonnegative_upward(
        add_nonnegative_upward(1.0, range.max_abs()),
        next_up(2.0 * negative_rho_abs),
    );
    let arithmetic_error =
        wilkinson_roundoff(arithmetic_scale, DETERMINANT_VALUE_OPS_PER_MODE);
    let mut exp_input_error = 0.0_f64;
    if hi >= 0.0 {
        let positive_lo = lo.max(0.0);
        let exp_neg_rho = exp_interval(-hi, -positive_lo)?;
        let argument_lo = ClosedInterval::point(penalty)
            .add(ClosedInterval::point(gram).mul(exp_neg_rho))
            .lo;
        if argument_lo > 0.0 {
            exp_input_error = exp_input_error.max(next_up(
                gram
                    * certified_exp_forward_error(
                        ClosedInterval::new(-hi, -positive_lo),
                        exp_neg_rho,
                    )
                    / argument_lo,
            ));
        } else {
            exp_input_error = f64::INFINITY;
        }
    }
    if lo < 0.0 {
        let negative_hi = hi.min(0.0);
        let exp_rho = exp_interval(lo, negative_hi)?;
        if exp_rho.lo > 0.0 {
            // The two stable negative-rho branches have log-lambda
            // sensitivities `u` and `1-u`, respectively. Both are at most one,
            // so the scale-safe relative exp error is a uniform bound even if
            // the dominance branch changes.
            exp_input_error = exp_input_error.max(
                certified_exp_relative_forward_error(
                    ClosedInterval::new(lo, negative_hi),
                    exp_rho,
                ),
            );
        } else {
            exp_input_error = f64::INFINITY;
        }
    }
    let log_output_error =
        certified_log_error_from_output(at_lo).max(certified_log_error_from_output(at_hi));
    let log_gram_error = certified_log_forward_error(ClosedInterval::point(gram));
    let log_penalty_error = certified_log_forward_error(ClosedInterval::point(penalty));
    let log1p_error = certified_ln1p_forward_error();
    let elementary_error = add_nonnegative_upward(
        exp_input_error,
        add_nonnegative_upward(
            log_output_error,
            add_nonnegative_upward(
                log_gram_error,
                add_nonnegative_upward(log_penalty_error, log1p_error),
            ),
        ),
    );
    Ok((
        range,
        add_nonnegative_upward(arithmetic_error, elementary_error),
    ))
}

fn normalized_log_mode_at(
    gram: f64,
    penalty: f64,
    rho: f64,
) -> Result<ClosedInterval, AffineRemlError> {
    if rho >= 0.0 {
        let exp_neg_rho = exp_interval(-rho, -rho)?;
        let argument = ClosedInterval::point(penalty)
            .add(ClosedInterval::point(gram).mul(exp_neg_rho));
        if !(argument.lo > 0.0 && argument.hi.is_finite()) {
            return Err(AffineRemlError::ElementaryEnclosureUnavailable {
                function: "ln",
                lo: argument.lo,
                hi: argument.hi,
            });
        }
        Ok(argument.ln_positive())
    } else {
        let exp_rho = exp_interval(rho, rho)?;
        let argument =
            ClosedInterval::point(gram).add(ClosedInterval::point(penalty).mul(exp_rho));
        if !(argument.lo > 0.0 && argument.hi.is_finite()) {
            return Err(AffineRemlError::ElementaryEnclosureUnavailable {
                function: "ln",
                lo: argument.lo,
                hi: argument.hi,
            });
        }
        Ok(argument
            .ln_positive()
            .sub(ClosedInterval::point(rho)))
    }
}

fn exp_interval(lo: f64, hi: f64) -> Result<ClosedInterval, AffineRemlError> {
    let unavailable = || AffineRemlError::ElementaryEnclosureUnavailable {
        function: "exp",
        lo,
        hi,
    };
    if !(lo.is_finite() && hi.is_finite() && lo <= hi) {
        return Err(unavailable());
    }
    let lower = certified_exp(lo).ok_or_else(unavailable)?;
    let upper = certified_exp(hi).ok_or_else(unavailable)?;
    let enclosure = ClosedInterval::new(lower.lo.max(0.0), upper.hi).nonnegative();
    if !enclosure.is_valid() {
        return Err(unavailable());
    }
    Ok(enclosure)
}

/// Directed division for a nonnegative numerator and a strictly positive
/// denominator without first materializing the reciprocal.
///
/// Forming `1 / denominator.lo` can overflow even when the final quotient is
/// finite because a correspondingly tiny numerator cancels that scale. Direct
/// endpoint quotients preserve that finite result. Invalid preconditions and a
/// nonfinite upper bound are typed refusals rather than assertions.
fn finite_nonnegative_quotient(
    numerator: ClosedInterval,
    denominator: ClosedInterval,
    function: &'static str,
) -> Result<ClosedInterval, AffineRemlError> {
    if !(numerator.is_valid()
        && numerator.lo >= 0.0
        && denominator.is_valid()
        && denominator.lo > 0.0)
    {
        return Err(AffineRemlError::ElementaryEnclosureUnavailable {
            function,
            lo: denominator.lo,
            hi: denominator.hi,
        });
    }
    let quotient = ClosedInterval::new(
        quotient_down(numerator.lo, denominator.hi).max(0.0),
        quotient_up(numerator.hi, denominator.lo),
    );
    if !(quotient.is_valid() && quotient.hi.is_finite()) {
        return Err(AffineRemlError::ElementaryEnclosureUnavailable {
            function,
            lo: quotient.lo,
            hi: quotient.hi,
        });
    }
    Ok(quotient.nonnegative())
}

fn mode_ranges(
    gram: f64,
    penalty: f64,
    projected_square: f64,
    lambda: ClosedInterval,
) -> Result<ModeRanges, AffineRemlError> {
    if penalty == 0.0 {
        let v = ClosedInterval::point(projected_square)
            .div_positive(ClosedInterval::point(gram))
            .nonnegative();
        return Ok(ModeRanges {
            c: ClosedInterval::point(0.0),
            w: ClosedInterval::point(0.0),
            v,
            smoothing_increment: ClosedInterval::point(0.0),
            singular_fitted: ClosedInterval::point(0.0),
            p: ClosedInterval::point(0.0),
            q: ClosedInterval::point(0.0),
        });
    }
    if gram == 0.0 {
        let zero = ClosedInterval::point(0.0);
        if projected_square == 0.0 {
            return Ok(ModeRanges {
                c: zero,
                w: zero,
                v: zero,
                smoothing_increment: zero,
                singular_fitted: zero,
                p: zero,
                q: zero,
            });
        }

        // The normalized determinant is exactly constant for g=0. For the
        // residual, v = A/(lambda*s). Use the direct product only when its
        // outward lower bound is strictly positive. If that lower bound rounds
        // to zero, cancel the exact scalar penalty first and divide the
        // resulting nonnegative interval directly by lambda. This preserves a
        // finite quotient such as min_subnormal/(0.01*lambda) without ever
        // asking `div_positive` to accept a denominator containing zero.
        let h = lambda
            .mul(ClosedInterval::point(penalty))
            .nonnegative();
        let projected = ClosedInterval::point(projected_square);
        let v = if h.lo > 0.0 {
            finite_nonnegative_quotient(projected, h, "gram-zero residual quotient")?
        } else {
            let scaled = finite_nonnegative_quotient(
                projected,
                ClosedInterval::point(penalty),
                "gram-zero residual quotient",
            )?;
            finite_nonnegative_quotient(
                scaled,
                lambda,
                "gram-zero residual quotient",
            )?
        };
        return Ok(ModeRanges {
            c: ClosedInterval::point(0.0),
            w: ClosedInterval::point(0.0),
            v,
            smoothing_increment: zero,
            singular_fitted: v,
            p: v,
            q: v.neg(),
        });
    }

    // Normalize by g: h = g(1+t), t = lambda*s/g.  The four kernels below
    // have known global critical points, so endpoint evaluation plus any
    // critical point contained by the t-window gives an exact real range;
    // interval arithmetic rounds every primitive outward.
    let t = lambda
        .mul(ClosedInterval::point(penalty))
        .div_positive(ClosedInterval::point(gram))
        .nonnegative();
    let scale = ClosedInterval::point(projected_square)
        .div_positive(ClosedInterval::point(gram))
        .nonnegative();
    let kernels = kernel_ranges(t);
    Ok(ModeRanges {
        c: kernels.v,
        w: kernels.w,
        v: scale.mul(kernels.v).nonnegative(),
        smoothing_increment: scale.mul(kernels.u).nonnegative(),
        singular_fitted: ClosedInterval::point(0.0),
        p: scale.mul(kernels.w).nonnegative(),
        q: scale.mul(kernels.k),
    })
}

#[derive(Clone, Copy)]
struct KernelRanges {
    /// `1/(1+t)`.
    v: ClosedInterval,
    /// `t/(1+t)`.
    u: ClosedInterval,
    /// `t/(1+t)^2`.
    w: ClosedInterval,
    /// `t(1-t)/(1+t)^3`.
    k: ClosedInterval,
}

fn kernel_at(t: ClosedInterval) -> KernelRanges {
    let one = ClosedInterval::point(1.0);
    let denom = one.add(t);
    let v = one.div_positive(denom).nonnegative();
    let u = t.mul(v).nonnegative();
    let w = u.mul(v).nonnegative();
    let k = w.mul(one.sub(t)).div_positive(denom);
    KernelRanges { v, u, w, k }
}

fn kernel_ranges(t: ClosedInterval) -> KernelRanges {
    let left = kernel_at(ClosedInterval::point(t.lo));
    let right = kernel_at(ClosedInterval::point(t.hi));
    let mut v = ClosedInterval::new(right.v.lo, left.v.hi).nonnegative();
    let u = ClosedInterval::new(left.u.lo, right.u.hi).nonnegative();
    let mut w = left.w.hull(right.w).nonnegative();
    let mut k = left.k.hull(right.k);

    if t.contains(1.0) {
        let critical = kernel_at(ClosedInterval::point(1.0));
        w = w.hull(critical.w).nonnegative();
    }

    // k'(t) has its only positive roots at 2 +/- sqrt(3).  Enclose sqrt(3)
    // itself before subtraction/addition so the exact irrational critical
    // points are not lost to nearest-rounded scalar arithmetic.
    let sqrt_three = certified_sqrt_positive(3.0)
        .expect("three is a finite positive square-root argument");
    let critical_points = [
        ClosedInterval::point(2.0).sub(sqrt_three),
        ClosedInterval::point(2.0).add(sqrt_three),
    ];
    for critical in critical_points {
        if critical.hi >= t.lo && critical.lo <= t.hi {
            k = k.hull(kernel_at(critical).k);
        }
    }

    // Monotonicity gives tighter endpoint ranges than a dependency-heavy
    // interval evaluation, but retain outward endpoint arithmetic.
    v.lo = v.lo.max(0.0);
    v.hi = v.hi.min(next_up(1.0));
    KernelRanges { v, u, w, k }
}

const LOG_SERIES_TERMS: usize = 18;
const EXP_SERIES_TERMS: usize = 18;
const EXP_RANGE_SQUARINGS: usize = 6;

fn certified_sqrt_positive(value: f64) -> Option<ClosedInterval> {
    if !(value.is_finite() && value > 0.0) {
        return None;
    }
    // `sqrt` supplies only a starting guess. Directed squaring proves and, if
    // necessary, expands the two sides, so no platform sqrt accuracy contract
    // is a premise of the returned interval.
    let guess = value.sqrt();
    if !(guess.is_finite() && guess > 0.0) {
        return None;
    }
    let mut lo = next_down(guess);
    for _ in 0..8 {
        if ClosedInterval::point(lo).square().hi <= value {
            break;
        }
        lo = next_down(lo);
    }
    let mut hi = next_up(guess);
    for _ in 0..8 {
        if ClosedInterval::point(hi).square().lo >= value {
            break;
        }
        hi = next_up(hi);
    }
    (ClosedInterval::point(lo).square().hi <= value
        && ClosedInterval::point(hi).square().lo >= value)
        .then(|| ClosedInterval::new(lo, hi))
}

/// `2·atanh(z)` by its positive odd-power series, with the omitted tail
/// bounded geometrically. The caller supplies `|z| <= 1/3`.
fn certified_log_from_atanh(z: ClosedInterval) -> ClosedInterval {
    let z_abs = z.max_abs();
    assert!(z_abs <= 1.0 / 3.0 + f64::EPSILON);
    let z2 = z.square();
    let mut power = z;
    let mut sum = z;
    for term in 1..LOG_SERIES_TERMS {
        power = power.mul(z2);
        sum = sum.add(
            power.div_positive(ClosedInterval::point((2 * term + 1) as f64)),
        );
    }
    let next_power = power.mul(z2).max_abs();
    let first_denominator = (2 * LOG_SERIES_TERMS + 1) as f64;
    let geometric_denominator = next_down(1.0 - next_up(z_abs * z_abs));
    let tail = if geometric_denominator > 0.0 {
        next_up(
            next_up(2.0 * next_power)
                / next_down(first_denominator * geometric_denominator),
        )
    } else {
        f64::INFINITY
    };
    sum.scale(2.0).widen(tail)
}

fn certified_ln_two() -> ClosedInterval {
    static LN_TWO: OnceLock<ClosedInterval> = OnceLock::new();
    *LN_TWO.get_or_init(|| {
        // ln(2) = 2 atanh(1/3). Both the rational 1/3 and the series are
        // evaluated with directed IEEE basic operations; no platform libm
        // result participates in this constant.
        let third = ClosedInterval::point(1.0)
            .div_positive(ClosedInterval::point(3.0));
        certified_log_from_atanh(third)
    })
}

/// Exact decomposition `value = mantissa * 2^exponent` with
/// `mantissa in [1, 2)` for every positive finite binary64 value.
fn positive_binary64_parts(value: f64) -> Option<(f64, i32)> {
    if !(value.is_finite() && value > 0.0) {
        return None;
    }
    let bits = value.to_bits();
    let exponent_bits = ((bits >> 52) & 0x7ff) as i32;
    let fraction = bits & ((1_u64 << 52) - 1);
    if exponent_bits == 0 {
        // value = fraction*2^-1074. Normalize the integer significand into
        // [2^52,2^53), then install it under exponent zero.
        let highest = 63_i32 - fraction.leading_zeros() as i32;
        let normalized = fraction << (52 - highest);
        let mantissa_bits = (1023_u64 << 52) | (normalized - (1_u64 << 52));
        Some((f64::from_bits(mantissa_bits), highest - 1074))
    } else {
        let mantissa_bits = (1023_u64 << 52) | fraction;
        Some((
            f64::from_bits(mantissa_bits),
            exponent_bits - 1023,
        ))
    }
}

/// Rigorous exact-real enclosure of `ln(value)` for every finite positive
/// binary64 input, including subnormals.
///
/// Bit decomposition writes `value = m·2^k` exactly with `m in [1,2)`.
/// `ln(m) = 2·atanh((m-1)/(m+1))` then has `z in [0,1/3]`, so the fixed
/// positive series above has a closed geometric remainder. Only directed
/// binary64 basic operations are used.
pub fn certified_ln_positive(value: f64) -> Option<ClosedInterval> {
    if !(value.is_finite() && value > 0.0) {
        return None;
    }
    if value == 1.0 {
        return Some(ClosedInterval::point(0.0));
    }
    let (mantissa, exponent) = positive_binary64_parts(value)?;
    let m = ClosedInterval::point(mantissa);
    let z = m
        .sub(ClosedInterval::point(1.0))
        .div_positive(m.add(ClosedInterval::point(1.0)));
    Some(
        certified_log_from_atanh(z)
            .add(certified_ln_two().scale(exponent as f64)),
    )
}

/// Rigorous exact-real enclosure of `ln(1+value)`.
///
/// The nonnegative lane used by the affine score evaluates
/// `2·atanh(value/(2+value))` directly when `value <= 1`, preserving tiny
/// `value` without the rounded `1+value` cancellation. For larger values the
/// exact identity `ln(1+x) = ln(x) + ln(1+1/x)` keeps the atanh argument below
/// `1/3` and avoids overflow in `1+x`. Negative valid inputs route through the
/// certified positive logarithm of an outward `1+value` interval.
pub fn certified_ln_1p(value: f64) -> Option<ClosedInterval> {
    if !(value.is_finite() && value > -1.0) {
        return None;
    }
    if value == 0.0 {
        return Some(ClosedInterval::point(0.0));
    }
    if (0.0..=1.0).contains(&value) {
        let x = ClosedInterval::point(value);
        let z = x.div_positive(ClosedInterval::point(2.0).add(x));
        return Some(certified_log_from_atanh(z));
    }
    if value > 1.0 {
        let reciprocal = ClosedInterval::point(1.0)
            .div_positive(ClosedInterval::point(value))
            .nonnegative();
        let z = reciprocal
            .div_positive(ClosedInterval::point(2.0).add(reciprocal))
            .nonnegative();
        return Some(certified_ln_positive(value)?.add(certified_log_from_atanh(z)));
    }
    let argument = ClosedInterval::point(1.0).add(ClosedInterval::point(value));
    if !(argument.lo > 0.0) {
        return None;
    }
    let lo = certified_ln_positive(argument.lo)?;
    let hi = certified_ln_positive(argument.hi)?;
    Some(ClosedInterval::new(lo.lo, hi.hi))
}

fn exact_power_of_two(exponent: i32) -> Option<f64> {
    match exponent {
        -1074..=-1023 => {
            let bit = (exponent + 1074) as u32;
            Some(f64::from_bits(1_u64 << bit))
        }
        -1022..=1023 => Some(f64::from_bits(((exponent + 1023) as u64) << 52)),
        _ => None,
    }
}

/// Stable rounded representative of `numerator/(first*second)`.
///
/// Exact binary exponent extraction prevents the denominator product from
/// underflowing or overflowing before its scale cancels against the numerator.
/// Only two mantissa divisions and the final binary scaling round.
fn positive_ratio_over_product(
    numerator: f64,
    first_denominator: f64,
    second_denominator: f64,
) -> Option<f64> {
    if numerator == 0.0 {
        return Some(0.0);
    }
    let (numerator_mantissa, numerator_exponent) =
        positive_binary64_parts(numerator)?;
    let (first_mantissa, first_exponent) =
        positive_binary64_parts(first_denominator)?;
    let (second_mantissa, second_exponent) =
        positive_binary64_parts(second_denominator)?;
    let mut mantissa = numerator_mantissa / first_mantissa / second_mantissa;
    let mut exponent = numerator_exponent - first_exponent - second_exponent;
    if !(mantissa.is_finite() && mantissa > 0.0) {
        return None;
    }
    while mantissa < 1.0 {
        mantissa *= 2.0;
        exponent -= 1;
    }
    while mantissa >= 2.0 {
        mantissa *= 0.5;
        exponent += 1;
    }
    if exponent < -1075 {
        return Some(0.0);
    }
    if exponent > 1023 {
        return None;
    }
    let value = if exponent == -1075 {
        (0.5 * mantissa) * exact_power_of_two(-1074)?
    } else {
        mantissa * exact_power_of_two(exponent)?
    };
    (value.is_finite() && value >= 0.0).then_some(value)
}

/// Rigorous exact-real enclosure of `exp(value)` for a finite binary64 input.
///
/// Range reduction uses the independently certified `ln(2)` interval:
/// `value = k ln(2) + r`. After six exact halvings, `|r/64| < 1/16`; a fixed
/// Taylor polynomial encloses `exp(r/64)` and a geometric bound encloses its
/// positive tail. Six interval squarings and multiplication by the exact
/// binary power `2^k` restore the result. Subnormal outputs remain intervals
/// with an absolute (possibly zero) lower endpoint instead of being forced
/// through an invalid relative-error model.
pub fn certified_exp(value: f64) -> Option<ClosedInterval> {
    if !value.is_finite() {
        return None;
    }
    if value == 0.0 {
        return Some(ClosedInterval::point(1.0));
    }
    // This quotient merely chooses an integer identity; its accuracy is not a
    // proof premise because `r = value-k·ln(2)` is subsequently enclosed using
    // the certified ln(2) interval and validated below.
    let mut exponent = (value / std::f64::consts::LN_2).round() as i32;
    exponent = exponent.clamp(-1074, 1023);
    let remainder = ClosedInterval::point(value)
        .sub(certified_ln_two().scale(exponent as f64));
    if !(remainder.is_valid() && remainder.max_abs() < 4.0) {
        return None;
    }
    let reduction = (1_u64 << EXP_RANGE_SQUARINGS) as f64;
    let reduced = remainder.scale(1.0 / reduction);
    if !(reduced.max_abs() < 1.0 / 16.0) {
        return None;
    }
    let mut term = ClosedInterval::point(1.0);
    let mut sum = term;
    for degree in 1..=EXP_SERIES_TERMS {
        term = term
            .mul(reduced)
            .div_positive(ClosedInterval::point(degree as f64));
        sum = sum.add(term);
    }
    let z = reduced.max_abs();
    let first_omitted = next_up(term.max_abs() * z / (EXP_SERIES_TERMS + 1) as f64);
    // Every later term ratio is at most z, so a geometric majorant is valid.
    let tail = next_up(first_omitted / next_down(1.0 - z));
    let mut result = sum.widen(tail);
    for _ in 0..EXP_RANGE_SQUARINGS {
        result = result.square();
    }
    result = result.mul(ClosedInterval::point(exact_power_of_two(exponent)?));
    Some(result.nonnegative())
}

#[inline]
fn certified_midpoint(interval: ClosedInterval) -> f64 {
    let midpoint = interval.lo + 0.5 * (interval.hi - interval.lo);
    midpoint.max(interval.lo).min(interval.hi)
}

/// Deterministic representative of [`certified_exp`].
///
/// This midpoint is for downstream floating-point evaluation only; callers
/// needing a proof must retain the full enclosure returned by
/// [`certified_exp`].
#[inline]
pub fn certified_exp_representative(value: f64) -> Option<f64> {
    certified_exp(value).map(certified_midpoint)
}

#[inline]
fn certified_ln_value(value: f64) -> Option<f64> {
    certified_ln_positive(value).map(certified_midpoint)
}

#[inline]
fn certified_ln_1p_value(value: f64) -> Option<f64> {
    certified_ln_1p(value).map(certified_midpoint)
}

fn interval_diameter(interval: ClosedInterval) -> f64 {
    if interval.lo == interval.hi {
        0.0
    } else {
        next_up(interval.hi - interval.lo)
    }
}

fn log_series_tail_max() -> f64 {
    let z = next_up(1.0 / 3.0);
    let z2 = next_up(z * z);
    let mut power = z;
    for _ in 1..LOG_SERIES_TERMS {
        power = next_up(power * z2);
    }
    power = next_up(power * z2);
    let denominator = next_down(
        (2 * LOG_SERIES_TERMS + 1) as f64 * next_down(1.0 - z2),
    );
    next_up(next_up(2.0 * power) / denominator)
}

/// Uniform absolute remainder of the reduced exponential Taylor series on
/// `[-1/16, 1/16]`, propagated through the six restoring squarings as a
/// relative error. This is computed only with outward basic arithmetic.
fn exp_series_relative_tail_max() -> f64 {
    let z = next_up(1.0 / 16.0);
    let mut term = 1.0;
    for degree in 1..=EXP_SERIES_TERMS {
        term = next_up(next_up(term * z) / degree as f64);
    }
    let first_omitted =
        next_up(next_up(term * z) / (EXP_SERIES_TERMS + 1) as f64);
    let absolute_tail = next_up(first_omitted / next_down(1.0 - z));
    // exp(reduced) >= exp(-1/16) > 1/2, hence its relative error is at most
    // twice the absolute Taylor tail. Raising the reduced result to 64 raises
    // the multiplicative error factor to the same power.
    let mut factor = ClosedInterval::point(1.0)
        .add(ClosedInterval::point(next_up(2.0 * absolute_tail)));
    for _ in 0..EXP_RANGE_SQUARINGS {
        factor = factor.square();
    }
    next_up(factor.hi - 1.0).max(0.0)
}

/// Uniform forward-error bound for the midpoint returned by
/// [`certified_ln_value`] over a positive input interval.
fn certified_log_forward_error(input: ClosedInterval) -> f64 {
    if !(input.lo > 0.0 && input.hi.is_finite()) {
        return f64::INFINITY;
    }
    let exponent_abs = [input.lo, input.hi]
        .into_iter()
        .map(|value| {
            let bits = value.to_bits();
            let exponent_bits = ((bits >> 52) & 0x7ff) as i32;
            if exponent_bits == 0 {
                let fraction = bits & ((1_u64 << 52) - 1);
                let highest = 63_i32 - fraction.leading_zeros() as i32;
                (highest - 1074).unsigned_abs() as f64
            } else {
                (exponent_bits - 1023).unsigned_abs() as f64
            }
        })
        .fold(0.0_f64, f64::max);
    let ln_two_uncertainty =
        next_up(exponent_abs * interval_diameter(certified_ln_two()));
    // Per term: power multiply, division, and accumulation, with two directed
    // endpoints; the remainder and range-combination path add 32 operations.
    let mantissa_ops = 6 * LOG_SERIES_TERMS + 32;
    let mantissa_error = add_nonnegative_upward(
        wilkinson_roundoff(1.0, mantissa_ops),
        log_series_tail_max(),
    );
    add_nonnegative_upward(ln_two_uncertainty, mantissa_error)
}

fn certified_log_error_from_output(output: ClosedInterval) -> f64 {
    if !output.is_valid() {
        return f64::INFINITY;
    }
    // |ln(input)|/ln(2) bounds the binary exponent to one neighboring bin.
    let exponent_abs =
        next_up(output.max_abs() / certified_ln_two().lo.abs()).ceil() + 1.0;
    let ln_two_uncertainty =
        next_up(exponent_abs * interval_diameter(certified_ln_two()));
    let mantissa_ops = 6 * LOG_SERIES_TERMS + 32;
    add_nonnegative_upward(
        ln_two_uncertainty,
        add_nonnegative_upward(
            wilkinson_roundoff(1.0, mantissa_ops),
            log_series_tail_max(),
        ),
    )
}

fn certified_ln1p_forward_error() -> f64 {
    let operations = 6 * LOG_SERIES_TERMS + 36;
    add_nonnegative_upward(
        wilkinson_roundoff(1.0, operations),
        log_series_tail_max(),
    )
}

/// Uniform absolute forward-error bound for [`certified_exp_representative`] on an
/// input interval, including range-reduction uncertainty and gradual
/// underflow.
fn certified_exp_forward_error(input: ClosedInterval, output: ClosedInterval) -> f64 {
    if !(input.is_valid() && output.is_valid() && output.lo >= 0.0) {
        return f64::INFINITY;
    }
    let exponent_abs = next_up(input.max_abs() / certified_ln_two().lo).ceil() + 1.0;
    let reduction_error =
        next_up(exponent_abs * interval_diameter(certified_ln_two()));
    if !(reduction_error < 1.0) {
        return f64::INFINITY;
    }
    // exp(delta)-1 <= delta/(1-delta) for 0 <= delta < 1.
    let propagated_reduction =
        next_up(output.max_abs() * reduction_error / next_down(1.0 - reduction_error));
    // Taylor recurrence, remainder, six squarings, and final binary scaling;
    // count both directed endpoints of each basic operation.
    let operations = 6 * EXP_SERIES_TERMS + 4 * EXP_RANGE_SQUARINGS + 40;
    let arithmetic = wilkinson_roundoff(output.max_abs(), operations);
    let truncation =
        next_up(output.max_abs() * exp_series_relative_tail_max());
    add_nonnegative_upward(
        propagated_reduction,
        add_nonnegative_upward(arithmetic, truncation),
    )
}

/// Uniform relative forward-error bound for
/// [`certified_exp_representative`] on an input interval whose exponential is
/// certified strictly positive.
///
/// The absolute bound above scales every multiplicative contribution by the
/// largest output in the interval. Dividing that result by the smallest output
/// couples opposite endpoints and can overflow on a wide interval even though
/// exp has a finite scale-independent relative error. Keep the range-reduction,
/// arithmetic, and truncation terms in relative currency instead. Only gradual
/// underflow is genuinely additive, so only that allowance is divided by the
/// certified positive lower output.
fn certified_exp_relative_forward_error(
    input: ClosedInterval,
    output: ClosedInterval,
) -> f64 {
    if !(input.is_valid()
        && output.is_valid()
        && output.lo > 0.0
        && output.hi.is_finite())
    {
        return f64::INFINITY;
    }
    let exponent_abs = next_up(input.max_abs() / certified_ln_two().lo).ceil() + 1.0;
    let reduction_error =
        next_up(exponent_abs * interval_diameter(certified_ln_two()));
    if !(reduction_error < 1.0) {
        return f64::INFINITY;
    }
    let relative_reduction =
        next_up(reduction_error / next_down(1.0 - reduction_error));
    let operations = 6 * EXP_SERIES_TERMS + 4 * EXP_RANGE_SQUARINGS + 40;
    let relative_arithmetic = wilkinson_roundoff(1.0, operations);
    let relative_underflow =
        next_up(wilkinson_roundoff(0.0, operations) / output.lo);
    add_nonnegative_upward(
        relative_reduction,
        add_nonnegative_upward(
            relative_arithmetic,
            add_nonnegative_upward(
                exp_series_relative_tail_max(),
                relative_underflow,
            ),
        ),
    )
}

/// Upward-rounded accumulation of a nonnegative magnitude bound.
fn add_nonnegative_upward(accumulator: f64, term: f64) -> f64 {
    if accumulator == f64::INFINITY || term == f64::INFINITY {
        f64::INFINITY
    } else if term == 0.0 {
        accumulator
    } else {
        next_up(accumulator + term)
    }
}

/// Symmetric absolute radius needed to widen `mathematical` until it contains
/// the already-computed `resolved` interval.
fn enclosure_excess(mathematical: ClosedInterval, resolved: ClosedInterval) -> f64 {
    let lower = if mathematical.lo == resolved.lo {
        0.0
    } else {
        next_up(mathematical.lo - resolved.lo)
    };
    let upper = if mathematical.hi == resolved.hi {
        0.0
    } else {
        next_up(resolved.hi - mathematical.hi)
    };
    lower.max(upper).max(0.0)
}

/// Wilkinson forward-error bound for `k` round-to-nearest binary64
/// operations. The normal-range `gamma_k * magnitude` term is accompanied by
/// `k` minimum-subnormal units, covering gradual-underflow roundoff where a
/// purely relative model is invalid.
fn wilkinson_roundoff(magnitude: f64, operations: usize) -> f64 {
    if operations == 0 {
        return 0.0;
    }
    if !(magnitude.is_finite() && magnitude >= 0.0) {
        return f64::INFINITY;
    }
    // Convert the integer count upward before either product. For counts above
    // 2^53, `as f64` can round down; charging only one ulp after multiplication
    // would then combine two rounding steps into an unjustified one-step
    // bound.
    let operation_count = next_up(operations as f64);
    let underflow = next_up(operation_count * f64::from_bits(1));
    if magnitude == 0.0 {
        return underflow;
    }
    // IEEE-754 binary64 unit roundoff under round-to-nearest.
    let unit_roundoff = 0.5 * f64::EPSILON;
    let ku = next_up(operation_count * unit_roundoff);
    if !(ku < 1.0) {
        return f64::INFINITY;
    }
    let denominator = next_down(1.0 - ku);
    if !(denominator > 0.0) {
        return f64::INFINITY;
    }
    let gamma = next_up(ku / denominator);
    add_nonnegative_upward(next_up(gamma * magnitude), underflow)
}

#[inline]
fn sum_down(left: f64, right: f64) -> f64 {
    let value = left + right;
    if sum_is_exact(left, right, value) {
        value
    } else {
        next_down(value)
    }
}

#[inline]
fn sum_up(left: f64, right: f64) -> f64 {
    let value = left + right;
    if sum_is_exact(left, right, value) {
        value
    } else {
        next_up(value)
    }
}

/// Whether binary64 addition produced the exact-real sum.
///
/// Knuth's `TwoSum` residual is itself exact under IEEE round-to-nearest with
/// gradual underflow. Besides avoiding needless interval inflation, retaining
/// exact cancellation is semantically important: structural zeros in diffuse
/// covariance recurrences must remain `[0, 0]`, not become artificial
/// minimum-subnormal uncertainty.
#[inline]
fn sum_is_exact(left: f64, right: f64, value: f64) -> bool {
    if left == 0.0 || right == 0.0 {
        return true;
    }
    if !(left.is_finite() && right.is_finite() && value.is_finite()) {
        return value == left || value == right;
    }
    let virtual_right = value - left;
    let virtual_left = value - virtual_right;
    let right_residual = right - virtual_right;
    let left_residual = left - virtual_left;
    left_residual + right_residual == 0.0
}

#[inline]
fn product_is_exact(left: f64, right: f64) -> bool {
    left == 0.0 || right == 0.0 || left.abs() == 1.0 || right.abs() == 1.0
}

#[inline]
fn product_down(left: f64, right: f64) -> f64 {
    let value = left * right;
    if product_is_exact(left, right) {
        if value.is_nan() { 0.0 } else { value }
    } else {
        next_down(value)
    }
}

#[inline]
fn product_up(left: f64, right: f64) -> f64 {
    let value = left * right;
    if product_is_exact(left, right) {
        if value.is_nan() { 0.0 } else { value }
    } else {
        next_up(value)
    }
}

#[inline]
fn quotient_down(numerator: f64, denominator: f64) -> f64 {
    let value = numerator / denominator;
    if numerator == 0.0 || denominator.abs() == 1.0 {
        value
    } else {
        next_down(value)
    }
}

#[inline]
fn quotient_up(numerator: f64, denominator: f64) -> f64 {
    let value = numerator / denominator;
    if numerator == 0.0 || denominator.abs() == 1.0 {
        value
    } else {
        next_up(value)
    }
}

/// Next representable number below `value`, used for directed outward
/// rounding of interval lower bounds.
fn next_down(value: f64) -> f64 {
    if value.is_nan() || value == f64::NEG_INFINITY {
        return value;
    }
    if value == 0.0 {
        return -f64::from_bits(1);
    }
    let bits = value.to_bits();
    f64::from_bits(if value > 0.0 { bits - 1 } else { bits + 1 })
}

/// Next representable number above `value`, used for directed outward
/// rounding of interval upper bounds.
fn next_up(value: f64) -> f64 {
    if value.is_nan() || value == f64::INFINITY {
        return value;
    }
    if value == 0.0 {
        return f64::from_bits(1);
    }
    let bits = value.to_bits();
    f64::from_bits(if value > 0.0 { bits + 1 } else { bits - 1 })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn polynomial_hidden_bump_jet(x: f64) -> ScoreJet {
        let p = x * (x - 0.5) * (x - 1.0);
        let dp = 3.0 * x * x - 3.0 * x + 0.5;
        let ddp = 6.0 * x - 3.0;
        ScoreJet {
            value: x + 1000.0 * p * p,
            derivative: 1.0 + 2000.0 * p * dp,
            curvature: 2000.0 * (dp * dp + p * ddp),
            third: 2000.0 * (3.0 * dp * ddp + p * 6.0),
        }
    }

    fn polynomial_hidden_bump_enclosure(lo: f64, hi: f64) -> DerivativeEnclosure {
        let x = ClosedInterval::new(lo, hi);
        let p = x
            .mul(x.sub(ClosedInterval::point(0.5)))
            .mul(x.sub(ClosedInterval::point(1.0)));
        let dp = x
            .square()
            .scale(3.0)
            .sub(x.scale(3.0))
            .add(ClosedInterval::point(0.5));
        let ddp = x.scale(6.0).sub(ClosedInterval::point(3.0));
        let value = x.add(p.square().scale(1000.0));
        DerivativeEnclosure {
            score: ScoreValueEnclosure {
                value,
                evaluation_error: wilkinson_roundoff(value.max_abs(), 7),
            },
            derivative: ClosedInterval::point(1.0).add(p.mul(dp).scale(2000.0)),
            curvature: dp.square().add(p.mul(ddp)).scale(2000.0),
        }
    }

    #[test]
    fn hidden_between_endpoint_and_midpoint_samples_is_found() {
        let result = maximize_score_1d(
            0.0,
            1.0,
            1.0e-9,
            |x| -> Result<_, String> { Ok(polynomial_hidden_bump_jet(x)) },
            |lo, hi| -> Result<_, String> { Ok(polynomial_hidden_bump_enclosure(lo.x, hi.x)) },
        )
        .expect("certified search");

        // At x=0, 1/2, 1 both value and derivative agree exactly with f=x;
        // the former midpoint/Hermite heuristic therefore returned x=1.
        assert_eq!(polynomial_hidden_bump_jet(0.0).derivative, 1.0);
        assert_eq!(polynomial_hidden_bump_jet(0.5).derivative, 1.0);
        assert_eq!(polynomial_hidden_bump_jet(1.0).derivative, 1.0);
        assert!(result.optimum.x > 0.5 && result.optimum.x < 1.0);
        assert!(result.optimum.value > 2.9);
        assert_eq!(result.stationary_points.len(), 4);
    }

    fn quartic_jet(x: f64) -> ScoreJet {
        ScoreJet {
            value: -(x * x - 1.0).powi(2),
            derivative: 4.0 * x - 4.0 * x * x * x,
            curvature: 4.0 - 12.0 * x * x,
            third: -24.0 * x,
        }
    }

    fn quartic_enclosure(lo: f64, hi: f64) -> DerivativeEnclosure {
        let x = ClosedInterval::new(lo, hi);
        let shifted_square = x.square().sub(ClosedInterval::point(1.0));
        let value = shifted_square.square().neg();
        if lo == hi && (lo == -1.0 || lo == 0.0 || lo == 1.0) {
            return DerivativeEnclosure {
                score: ScoreValueEnclosure {
                    value,
                    evaluation_error: wilkinson_roundoff(value.max_abs(), 4),
                },
                derivative: ClosedInterval::point(0.0),
                curvature: ClosedInterval::point(quartic_jet(lo).curvature),
            };
        }
        DerivativeEnclosure {
            score: ScoreValueEnclosure {
                value,
                evaluation_error: wilkinson_roundoff(value.max_abs(), 4),
            },
            derivative: x.scale(4.0).sub(x.mul(x).mul(x).scale(4.0)),
            curvature: ClosedInterval::point(4.0).sub(x.square().scale(12.0)),
        }
    }

    #[test]
    fn multiple_roots_in_initial_bracket_are_all_isolated() {
        let result = maximize_score_1d(
            -2.0,
            2.0,
            1.0e-10,
            |x| -> Result<_, String> { Ok(quartic_jet(x)) },
            |lo, hi| -> Result<_, String> { Ok(quartic_enclosure(lo.x, hi.x)) },
        )
        .expect("certified search");
        assert_eq!(result.stationary_points.len(), 3);
        for (point, expected) in result.stationary_points.iter().zip([-1.0_f64, 0.0, 1.0]) {
            assert!((point.sample.x - expected).abs() <= 1.0e-9);
            assert!(point.bracket.hi - point.bracket.lo <= 1.0e-10);
        }
        assert!((result.optimum.x.abs() - 1.0).abs() <= 1.0e-9);
    }

    /// The abscissa at which this fixture's point oracle reports a derivative
    /// of exactly zero while the exact derivative is two.
    const ROUNDED_ZERO_ABSCISSA: f64 = 1.5;

    /// A point derivative that rounds to zero cannot close its parent cell.
    ///
    /// The exact score is the concave quadratic `1 - (x - 2.5)^2` on `[0, 3]`.
    /// Its only stationary point and maximum is exactly representable at
    /// `x=2.5`.
    /// The point oracle deliberately loses the nonzero derivative at the
    /// safeguarded midpoint `x=1.5`, while the exact-real enclosure reports the
    /// true derivative range through interval arithmetic. The initial Newton
    /// proposal at `x=3` is `2.5`, outside the central-half guard, so the
    /// midpoint is exercised deterministically. Treating its rounded scalar
    /// zero as a root closes the left half, discards the real maximum, and
    /// leaves the rounded-value selection at boundary `x=3`, whose score is
    /// `0.75`. The point enclosure introduced by the exact-real repair
    /// distinguishes that false zero from the quadratic's exact zero at
    /// `x=2.5`.
    #[test]
    fn a_rounded_zero_at_a_cell_endpoint_does_not_close_the_cell() {
        let mut rounded_zeros = 0_usize;
        let result = maximize_score_1d(
            0.0,
            3.0,
            1.0e-9,
            |x| -> Result<_, String> {
                let shifted = x - 2.5;
                let derivative = if x == ROUNDED_ZERO_ABSCISSA {
                    rounded_zeros += 1;
                    0.0
                } else {
                    -2.0 * shifted
                };
                Ok(ScoreJet {
                    value: 1.0 - shifted * shifted,
                    derivative,
                    curvature: -2.0,
                    third: 0.0,
                })
            },
            |left, right| -> Result<_, String> {
                let x = ClosedInterval::new(left.x, right.x);
                let shifted = x.sub(ClosedInterval::point(2.5));
                let value = ClosedInterval::point(1.0).sub(shifted.square());
                Ok(DerivativeEnclosure {
                    score: ScoreValueEnclosure {
                        value,
                        evaluation_error: wilkinson_roundoff(value.max_abs(), 3),
                    },
                    // Preserve the quadratic's structural zero at x=2.5 instead
                    // of manufacturing cancellation through `5 - 2x`.
                    derivative: shifted.scale(-2.0),
                    curvature: ClosedInterval::point(-2.0),
                })
            },
        )
        .expect("certified search");

        assert!(
            rounded_zeros > 0,
            "fixture premise unmet: the search never evaluated x = {ROUNDED_ZERO_ABSCISSA}"
        );
        assert!(
            (result.optimum.x - 2.5).abs() <= 1.0e-9,
            "reported the maximum at x={} (value {}) instead of x=2.5",
            result.optimum.x,
            result.optimum.value,
        );
        assert!(
            result.value_certificate.maximum.contains(1.0),
            "the exact maximum escaped the global score certificate: {:?}",
            result.value_certificate,
        );
        assert!(
            result
                .stationary_points
                .iter()
                .all(|point| point.sample.x != ROUNDED_ZERO_ABSCISSA),
            "a derivative that rounded to zero was reported as a stationary point",
        );
        let root = result
            .stationary_points
            .iter()
            .find(|point| point.bracket.contains(2.5))
            .expect("the exact quadratic root must be isolated");
        assert_eq!(
            root.bracket,
            ClosedInterval::point(2.5),
            "the cancellation-free point enclosure must preserve the exact dyadic root"
        );
    }

    #[test]
    fn adjacent_cell_evidence_is_retained_when_point_derivative_is_uninformative() {
        let planted = 0.7_f64;
        let result = maximize_score_1d(
            0.0,
            1.0,
            1.0e-9,
            |x| -> Result<_, String> {
                let shifted = x - planted;
                Ok(ScoreJet {
                    value: 1.0 - shifted * shifted,
                    derivative: -2.0 * shifted,
                    curvature: -2.0,
                    third: 0.0,
                })
            },
            |left, right| -> Result<_, String> {
                let x = ClosedInterval::new(left.x, right.x);
                let shifted = x.sub(ClosedInterval::point(planted));
                let value = ClosedInterval::point(1.0).sub(shifted.square());
                let interior_point = left.x == right.x && left.x > 0.0 && left.x < 1.0;
                Ok(DerivativeEnclosure {
                    score: ScoreValueEnclosure {
                        value,
                        evaluation_error: wilkinson_roundoff(value.max_abs(), 3),
                    },
                    // Model a cancellation-heavy degenerate-cell formula: by
                    // itself it carries no sign. Each adjacent nondegenerate
                    // interval remains a tight exact extension, and their
                    // intersection at the shared endpoint isolates the root.
                    derivative: if interior_point {
                        ClosedInterval::new(-2.0, 2.0)
                    } else {
                        shifted.scale(-2.0)
                    },
                    curvature: ClosedInterval::point(-2.0),
                })
            },
        )
        .expect("adjacent exact cell evidence must isolate the unique root");

        assert!(
            (result.optimum.x - planted).abs() <= 1.0e-9,
            "selected {}, expected {planted}",
            result.optimum.x
        );
        let stationary = result
            .stationary_points
            .iter()
            .find(|point| point.bracket.contains(planted))
            .expect("the planted stationary point must be certified");
        assert!(stationary.bracket.hi - stationary.bracket.lo <= 1.0e-9);
    }

    #[test]
    fn monotone_score_selects_exact_boundary() {
        let result = maximize_score_1d(
            -4.0,
            9.0,
            1.0e-9,
            |x| -> Result<_, String> {
                Ok(ScoreJet {
                    value: 0.3 * x,
                    derivative: 0.3,
                    curvature: 0.0,
                    third: 0.0,
                })
            },
            |left, right| -> Result<_, String> {
                let value = ClosedInterval::new(left.x, right.x).scale(0.3);
                Ok(DerivativeEnclosure {
                    score: ScoreValueEnclosure {
                        value,
                        evaluation_error: wilkinson_roundoff(value.max_abs(), 1),
                    },
                    derivative: ClosedInterval::point(0.3),
                    curvature: ClosedInterval::point(0.0),
                })
            },
        )
        .expect("certified search");
        assert_eq!(result.location, ScoreOptimumLocation::UpperBoundary);
        assert_eq!(result.optimum.x, 9.0);
        assert!(result.stationary_points.is_empty());
        assert_eq!(
            result.value_certificate.maximum_excess, 0.0,
            "the exact same terminal point is not a competing uncertain value"
        );
    }

    #[test]
    fn certified_increase_selects_upper_boundary_when_rounded_values_tie() {
        let result = maximize_score_1d(
            -1.0,
            1.0,
            1.0e-9,
            |_| -> Result<_, String> {
                Ok(ScoreJet {
                    value: 0.0,
                    derivative: 1.0,
                    curvature: 0.0,
                    third: 0.0,
                })
            },
            |left, right| -> Result<_, String> {
                Ok(DerivativeEnclosure {
                    score: ScoreValueEnclosure {
                        value: ClosedInterval::new(left.x, right.x),
                        evaluation_error: 1.0,
                    },
                    derivative: ClosedInterval::point(1.0),
                    curvature: ClosedInterval::point(0.0),
                })
            },
        )
        .expect("a whole-domain positive derivative orders tied rounded endpoints");
        assert_eq!(result.lower_boundary.value, result.upper_boundary.value);
        assert_eq!(result.location, ScoreOptimumLocation::UpperBoundary);
        assert_eq!(result.optimum.x, 1.0);
        assert_eq!(result.value_certificate.maximum_excess, 0.0);
    }

    #[test]
    fn certified_decrease_selects_lower_boundary_when_rounded_values_tie() {
        let result = maximize_score_1d(
            -1.0,
            1.0,
            1.0e-9,
            |_| -> Result<_, String> {
                Ok(ScoreJet {
                    value: 0.0,
                    derivative: -1.0,
                    curvature: 0.0,
                    third: 0.0,
                })
            },
            |left, right| -> Result<_, String> {
                Ok(DerivativeEnclosure {
                    score: ScoreValueEnclosure {
                        value: ClosedInterval::new(-right.x, -left.x),
                        evaluation_error: 1.0,
                    },
                    derivative: ClosedInterval::point(-1.0),
                    curvature: ClosedInterval::point(0.0),
                })
            },
        )
        .expect("a whole-domain negative derivative orders tied rounded endpoints");
        assert_eq!(result.lower_boundary.value, result.upper_boundary.value);
        assert_eq!(result.location, ScoreOptimumLocation::LowerBoundary);
        assert_eq!(result.optimum.x, -1.0);
        assert_eq!(result.value_certificate.maximum_excess, 0.0);
    }

    #[test]
    fn tangential_stationary_structure_is_accepted_only_after_value_flat_proof() {
        let result = maximize_score_1d(
            -1.0,
            1.0,
            1.0e-8,
            |x| -> Result<_, String> {
                Ok(ScoreJet {
                    value: x * x * x,
                    derivative: 3.0 * x * x,
                    curvature: 6.0 * x,
                    third: 6.0,
                })
            },
            |lo, hi| -> Result<_, String> {
                let x = ClosedInterval::new(lo.x, hi.x);
                Ok(DerivativeEnclosure {
                    score: ScoreValueEnclosure {
                        value: x.mul(x).mul(x),
                        evaluation_error: f64::EPSILON,
                    },
                    derivative: x.square().scale(3.0),
                    curvature: x.scale(6.0),
                })
            },
        )
        .expect("the unresolved inflection is immaterial at certified score-value resolution");
        assert_eq!(result.location, ScoreOptimumLocation::UpperBoundary);
        assert!(
            !result.resolution_flat_regions.is_empty(),
            "the search must record the value-side proof instead of silently dropping the cell"
        );
        for flat in result.resolution_flat_regions {
            assert!(flat.max_score_gap <= flat.score_resolution);
        }
    }

    #[test]
    fn unresolved_nonflat_cell_remains_typed() {
        let error = maximize_score_1d(
            0.0,
            1.0e-8,
            1.0e-8,
            |x| -> Result<_, String> {
                Ok(ScoreJet {
                    value: x,
                    derivative: 0.0,
                    curvature: 0.0,
                    third: 0.0,
                })
            },
            |lo, hi| -> Result<_, String> {
                Ok(DerivativeEnclosure {
                    score: ScoreValueEnclosure {
                        value: ClosedInterval::new(lo.x, hi.x),
                        evaluation_error: 0.0,
                    },
                    derivative: ClosedInterval::new(-1.0, 1.0),
                    curvature: ClosedInterval::new(-1.0, 1.0),
                })
            },
        )
        .expect_err("a derivative enclosure admitting visible score motion is not flat");
        assert!(matches!(error, ScoreSearchError::Unresolved { .. }));
    }

    /// BREADTH exhaustion, which is a different failure from the per-cell depth
    /// floor and was #2546's non-termination.
    ///
    /// The oracle's derivative and curvature enclosures always straddle zero, so
    /// no cell is ever excluded by a sign or isolated as a root; but its score
    /// range collapses with the cell against a FIXED evaluation error, so every
    /// cell does terminate — as resolution-flat — once it is narrower than
    /// `2 * evaluation_error`. That is the regime the cascade is in: cells
    /// certify, at widths far above `resolution`, and the traversal simply needs
    /// too many of them. The flat width here is 1e-3 of a 32-wide domain, so the
    /// decomposition is ~2^15 = 32 768 cells and no cell ever reaches the
    /// resolution floor — `ScoreSearchError::Unresolved` cannot fire, and
    /// without a breadth budget nothing else can either.
    ///
    /// Contrast `unresolved_nonflat_cell_remains_typed`, whose oracle certifies
    /// NOTHING at any width: that one bottoms out on the depth floor after `D`
    /// subdivisions and is already typed. The two are not interchangeable.
    #[test]
    fn undecomposable_criterion_exhausts_the_budget_instead_of_enumerating_the_domain() {
        let lo = 0.0;
        let hi = 32.0;
        let resolution = f64::EPSILON.sqrt();
        let flat_error = 5.0e-4;
        let (budget, depth_bound) = subdivision_budget(lo, hi, resolution);
        assert_eq!(depth_bound, 31, "log2(32 / sqrt(eps)) rounds up to 31");
        assert_eq!(budget, 2 * 31 * 31);
        let error = maximize_score_1d(
            lo,
            hi,
            resolution,
            |x| -> Result<_, String> {
                Ok(ScoreJet {
                    value: x,
                    derivative: 0.0,
                    curvature: 0.0,
                    third: 0.0,
                })
            },
            |left, right| -> Result<_, String> {
                Ok(DerivativeEnclosure {
                    score: ScoreValueEnclosure {
                        value: ClosedInterval::new(left.x, right.x),
                        evaluation_error: flat_error,
                    },
                    derivative: ClosedInterval::new(-1.0, 1.0),
                    curvature: ClosedInterval::new(-1.0, 1.0),
                })
            },
        )
        .expect_err("a decomposition this large must refuse, not enumerate");
        let ScoreSearchError::SubdivisionBudget {
            subdivisions,
            budget: reported_budget,
            depth_bound: reported_depth,
            cell_lo,
            cell_hi,
            ..
        } = error
        else {
            panic!("expected a subdivision-budget refusal, got {error}");
        };
        assert_eq!(subdivisions, budget + 1, "the budget stops the split that exceeds it");
        assert_eq!(reported_budget, budget);
        assert_eq!(reported_depth, depth_bound);
        assert!(
            cell_hi - cell_lo > 2.0 * flat_error,
            "the reported cell must be one the search could still have split and \
             had not yet certified ({cell_lo}, {cell_hi}); a narrower cell would \
             mean the depth floor, not the breadth budget, was binding"
        );
    }

    /// The same budget must be invisible to a search that converges. A strictly
    /// concave criterion over the same wide domain isolates its stationary point
    /// in subdivisions proportional to the DEPTH, so the number of criterion
    /// evaluations stays far below a budget scaled by the depth SQUARED.
    #[test]
    fn a_converging_search_stays_far_under_the_subdivision_budget() {
        let lo = 0.0;
        let hi = 32.0;
        let resolution = f64::EPSILON.sqrt();
        let (budget, depth_bound) = subdivision_budget(lo, hi, resolution);
        let evaluations = std::cell::Cell::new(0usize);
        let result = maximize_score_1d(
            lo,
            hi,
            resolution,
            |x| -> Result<_, String> {
                evaluations.set(evaluations.get() + 1);
                let shifted = x - 7.0;
                Ok(ScoreJet {
                    value: -shifted * shifted,
                    derivative: -2.0 * shifted,
                    curvature: -2.0,
                    third: 0.0,
                })
            },
            |left, right| -> Result<_, String> {
                let x = ClosedInterval::new(left.x, right.x);
                let shifted = x.sub(ClosedInterval::point(7.0));
                Ok(DerivativeEnclosure {
                    score: ScoreValueEnclosure {
                        value: shifted.square().scale(-1.0),
                        evaluation_error: f64::EPSILON * 1024.0,
                    },
                    derivative: shifted.scale(-2.0),
                    curvature: ClosedInterval::point(-2.0),
                })
            },
        )
        .expect("a strictly concave criterion is decomposable");
        let ScoreOptimumLocation::Stationary(index) = result.location else {
            panic!("expected the interior maximum, got {:?}", result.location);
        };
        let bracket = result.stationary_points[index].bracket;
        assert!(
            bracket.lo <= 7.0 && bracket.hi >= 7.0,
            "certified bracket {bracket:?} must contain the planted maximum"
        );
        // Every subdivision costs one midpoint evaluation, so the evaluation
        // count bounds the subdivisions from above.
        assert!(
            evaluations.get() < budget / 8,
            "a converging search used {} evaluations against budget {budget} at depth \
             bound {depth_bound}; a budget within 8x of a converging search is a \
             tuning parameter, not a backstop",
            evaluations.get()
        );
    }

    #[test]
    fn resolution_flatness_is_exactly_value_diameter_vs_pairwise_error() {
        let sample = SearchSample {
            sample: ScoreSample {
                x: 0.0,
                value: 7.0,
                derivative: 0.0,
                curvature: 0.0,
                third: 0.0,
            },
            point_enclosure: None,
        };
        let node = SearchNode {
            left: sample,
            right: SearchSample {
                sample: ScoreSample {
                    x: 1.0,
                    ..sample.sample
                },
                point_enclosure: None,
            },
        };
        let error = 0.125;
        for (upper, expected) in [(1024.25, true), (next_up(1024.25), false)] {
            let enclosure = DerivativeEnclosure {
                score: ScoreValueEnclosure {
                    // Translation by a large exactly represented constant must
                    // not change either side of the flatness comparison.
                    value: ClosedInterval::new(1024.0, upper),
                    evaluation_error: error,
                },
                derivative: ClosedInterval::new(-1.0, 1.0),
                curvature: ClosedInterval::new(-1.0, 1.0),
            };
            assert_eq!(
                resolution_flat_region(node, enclosure).is_some(),
                expected,
                "flatness must be equivalent to outward diameter <= outward 2*value error"
            );
        }
    }

    #[test]
    fn resolution_flat_cells_remain_regions_instead_of_fake_points() {
        let resolution = 0.25;
        let result = maximize_score_1d(
            0.0,
            1.0,
            resolution,
            |_| -> Result<_, String> {
                Ok(ScoreJet {
                    value: 3.0,
                    derivative: 0.0,
                    curvature: 0.0,
                    third: 0.0,
                })
            },
            |_, _| -> Result<_, String> {
                Ok(DerivativeEnclosure {
                    score: ScoreValueEnclosure {
                        value: ClosedInterval::point(3.0),
                        evaluation_error: 0.0,
                    },
                    derivative: ClosedInterval::new(-1.0, 1.0),
                    curvature: ClosedInterval::new(-1.0, 1.0),
                })
            },
        )
        .expect("an exactly constant score is resolution-flat");
        assert_eq!(result.resolution_flat_regions.len(), 1);
        assert!(
            result.resolution_flat_regions[0].bracket.hi
                - result.resolution_flat_regions[0].bracket.lo
                > resolution,
            "value resolution may close a wide cell, so callers must not reinterpret it \
             as an abscissa-resolved stationary point"
        );
    }

    #[test]
    fn directed_arithmetic_preserves_cancellation_and_subnormal_error() {
        assert_eq!(
            ClosedInterval::point(1.0).sub(ClosedInterval::point(1.0)),
            ClosedInterval::point(0.0),
            "an exact structural zero must not acquire artificial uncertainty"
        );
        let minimum_subnormal = f64::from_bits(1);
        let underflowing_product =
            ClosedInterval::point(minimum_subnormal).mul(ClosedInterval::point(0.5));
        assert!(
            underflowing_product.lo <= 0.5 * minimum_subnormal
                && underflowing_product.hi >= 0.5 * minimum_subnormal
                && underflowing_product.lo < 0.0
                && underflowing_product.hi > 0.0,
            "a nonzero exact product that rounds to zero needs additive subnormal width"
        );
        assert!(
            wilkinson_roundoff(0.0, 1) >= minimum_subnormal,
            "a zero-magnitude relative model must still charge additive underflow"
        );
    }

    #[test]
    fn certified_elementary_intervals_cover_normal_and_subnormal_lanes() {
        for value in [
            f64::from_bits(1),
            f64::MIN_POSITIVE,
            0.5,
            1.0,
            2.0,
            f64::MAX,
        ] {
            let enclosure = certified_ln_positive(value).expect("certified positive log");
            assert!(enclosure.is_valid() && enclosure.lo.is_finite() && enclosure.hi.is_finite());
            assert!(
                enclosure.contains(value.ln()),
                "independent platform log sanity value {} escaped {:?}",
                value.ln(),
                enclosure
            );
        }
        for value in [-744.0_f64, -708.0, -1.0, 0.0, 1.0, 709.0] {
            let enclosure = certified_exp(value).expect("certified exponential");
            assert!(enclosure.is_valid() && enclosure.lo >= 0.0);
            assert!(
                enclosure.contains(value.exp()),
                "independent platform exp sanity value {} escaped {:?}",
                value.exp(),
                enclosure
            );
        }
        for value in [f64::from_bits(1), 1.0e-12, 0.25, 1.0] {
            let enclosure = certified_ln_1p(value).expect("certified log1p");
            assert!(
                enclosure.contains(value.ln_1p()),
                "independent platform log1p sanity value {} escaped {:?}",
                value.ln_1p(),
                enclosure
            );
        }
    }

    #[test]
    fn exact_range_is_not_compared_to_a_separately_rounded_curvature() {
        let denormal = f64::from_bits(1);
        let result = maximize_score_1d(
            0.0,
            1.0,
            1.0e-8,
            |x| -> Result<_, String> {
                Ok(ScoreJet {
                    value: x,
                    derivative: 1.0,
                    // A real negative denormal rounds to signed zero in the
                    // point-jet arithmetic represented by this fixture.
                    curvature: -0.0,
                    third: 0.0,
                })
            },
            |left, right| -> Result<_, String> {
                Ok(DerivativeEnclosure {
                    score: ScoreValueEnclosure {
                        value: ClosedInterval::new(left.x, right.x),
                        evaluation_error: 0.0,
                    },
                    derivative: ClosedInterval::point(1.0),
                    curvature: ClosedInterval::point(-denormal),
                })
            },
        )
        .expect("an exact-real enclosure need not contain a separately rounded scalar jet");
        assert_eq!(result.location, ScoreOptimumLocation::UpperBoundary);
    }

    fn affine_fixture() -> AffineRemlProfile<'static> {
        const G: &[f64] = &[2.0, 0.5, 0.0, 3.0];
        const S: &[f64] = &[1.0, 0.0, 2.0, 0.25];
        const Q: &[f64] = &[
            0.6, 0.1, 0.02, 0.3, // response 0
            0.2, 0.4, 0.01, 0.5, // response 1
        ];
        const Y2: &[f64] = &[8.0, 10.0];
        AffineRemlProfile::new(G, S, Q, Y2, 12.0, 3, 0.7).expect("valid fixture")
    }

    #[test]
    fn affine_reml_jet_matches_test_only_differences() {
        let profile = affine_fixture();
        for x in [-2.0_f64, -0.4, 0.7, 2.0] {
            let h = 1.0e-5;
            let center = profile.evaluate(x).unwrap();
            let left = profile.evaluate(x - h).unwrap();
            let right = profile.evaluate(x + h).unwrap();
            let derivative = (right.value - left.value) / (2.0 * h);
            let curvature = (right.derivative - left.derivative) / (2.0 * h);
            assert!(
                (center.derivative - derivative).abs() <= 2.0e-8 * (1.0 + derivative.abs()),
                "first derivative mismatch at {x}: analytic {}, difference {derivative}",
                center.derivative
            );
            assert!(
                (center.curvature - curvature).abs() <= 2.0e-8 * (1.0 + curvature.abs()),
                "curvature mismatch at {x}: analytic {}, difference {curvature}",
                center.curvature
            );
        }
    }

    #[test]
    fn affine_reml_enclosure_contains_value_jets() {
        let profile = affine_fixture();
        let enclosure = profile.enclose(-2.5, 1.75).expect("enclosure");
        let score = enclosure.score;
        let resolved_score = score.value.widen(score.evaluation_error);
        for x in [-2.5_f64, -1.7, -0.3, 0.0, 0.9, 1.75] {
            let jet = profile.evaluate(x).unwrap();
            let point = profile.enclose(x, x).expect("point enclosure");
            assert!(
                resolved_score.contains(jet.value),
                "score {} at {x} outside {:?} ± {}",
                jet.value,
                score.value,
                score.evaluation_error
            );
            assert!(
                enclosure.derivative.intersection(point.derivative).is_some(),
                "exact point gradient {:?} at {x} is disjoint from {:?}",
                point.derivative,
                enclosure.derivative
            );
            assert!(
                enclosure.curvature.intersection(point.curvature).is_some(),
                "exact point curvature {:?} at {x} is disjoint from {:?}",
                point.curvature,
                enclosure.curvature
            );
        }
    }

    #[test]
    fn affine_reml_zero_smoothing_complement_retains_residual_correlation() {
        // With E = sum(q/g), the zero-smoothing residual is exactly zero and
        //
        //   R(lambda) = sum_i (q_i/g_i) * lambda/(1 + lambda).
        //
        // The determinant and profiled-residual derivatives then cancel
        // identically when residual_dof equals the mode count, so the exact
        // score derivative is zero. Forming R as
        // `E - sum q/(g + lambda*s)` loses the shared near-one factor once per
        // mode: at lambda=1e-10 its interval width is large enough to fabricate
        // a material derivative range even though every term has the same
        // analytic complement. The zero-smoothing form carries that
        // correlation explicitly.
        const MODES: usize = 64;
        let grams = [1.0; MODES];
        let penalties = [1.0; MODES];
        let projected = [1.0; MODES];
        let energies = [MODES as f64];
        let profile = AffineRemlProfile::new(
            &grams,
            &penalties,
            &projected,
            &energies,
            MODES as f64,
            MODES,
            0.0,
        )
        .expect("valid cancellation fixture");
        let rho = -23.025850929940457_f64; // nearest binary64 to ln(1e-10)
        let enclosure = profile
            .enclose(rho, rho)
            .expect("equivalent residual forms must retain their intersection");

        assert!(
            enclosure.derivative.contains_zero(),
            "the analytically constant profile must contain zero derivative: {:?}",
            enclosure.derivative
        );
        assert!(
            enclosure.derivative.hi - enclosure.derivative.lo < 1.0e-6,
            "the residual complement must remove the independent near-one dependency: {:?}",
            enclosure.derivative
        );
    }

    #[test]
    fn affine_reml_zero_smoothing_schur_residual_keeps_division_low_parts() {
        // Three exact-real quotients 1/3 sum to one, although no individual
        // quotient is representable in binary64. A directed interval around
        // each independently rounded quotient leaves an O(u) uncertainty in
        // `1 - 3*(1/3)`, larger than this profile's residual at lambda=1e-10.
        // The one-time TwoSum/FMA construction retains the division low parts,
        // so the exact zero Schur residual remains resolved near O(u²).
        let grams = [3.0; 3];
        let penalties = [1.0; 3];
        let projected = [1.0; 3];
        let energies = [1.0];
        let profile = AffineRemlProfile::new(
            &grams,
            &penalties,
            &projected,
            &energies,
            3.0,
            3,
            0.0,
        )
        .expect("valid nonrepresentable-quotient fixture");

        let zero_residual = profile.zero_lambda_residual[0];
        assert!(
            zero_residual.contains_zero(),
            "the exact identity 1 - 3*(1/3) = 0 must be retained: {zero_residual:?}"
        );
        assert!(
            zero_residual.hi - zero_residual.lo < 1.0e-28,
            "division corrections must live below ordinary binary64 cancellation scale: \
             {zero_residual:?}"
        );

        let rho = -23.025850929940457_f64;
        let enclosure = profile
            .enclose(rho, rho)
            .expect("the small positive smoothing residual must remain resolved");
        assert!(
            enclosure.derivative.contains_zero(),
            "determinant and residual derivatives cancel analytically: {:?}",
            enclosure.derivative
        );
        assert!(
            enclosure.derivative.hi - enclosure.derivative.lo < 1.0e-6,
            "the exact Schur residual must control the profiled derivative: {:?}",
            enclosure.derivative
        );
    }

    #[test]
    fn affine_reml_saturated_tail_preserves_complement_signs() {
        let profile =
            AffineRemlProfile::new(&[1.0], &[1.0], &[0.0], &[1.0], 4.0, 1, 0.0)
                .expect("valid saturated-tail fixture");
        let log_lambda = 700.0;
        let jet = profile.evaluate(log_lambda).expect("point jet");
        let enclosure = profile
            .enclose(log_lambda, log_lambda)
            .expect("point enclosure");

        assert!(
            jet.derivative > 0.0,
            "the point derivative must preserve +0.5/(1+exp(rho)), got {}",
            jet.derivative
        );
        assert!(
            jet.curvature < 0.0,
            "the point curvature must preserve its negative u*c sign, got {}",
            jet.curvature
        );
        assert!(
            enclosure.curvature.hi <= 0.0,
            "the exact saturated curvature remains nonpositive: {:?}",
            enclosure.curvature
        );
        assert!(
            enclosure.derivative.lo >= 0.0,
            "the exact saturated derivative remains nonnegative: {:?}",
            enclosure.derivative
        );
        let score = enclosure.score;
        assert!(score.evaluation_error.is_finite());
        assert!(
            score.value.widen(score.evaluation_error).contains(jet.value),
            "the stable score evaluator must lie inside its exact value range plus forward error"
        );
    }

    #[test]
    fn affine_reml_saturated_tail_uses_complement_sign_before_value_flatness() {
        let profile =
            AffineRemlProfile::new(&[1.0], &[1.0], &[0.0], &[1.0], 4.0, 1, 0.0)
                .expect("valid saturated-tail fixture");
        let result = profile
            .maximize(600.0, 700.0, f64::EPSILON.sqrt())
            .expect("the cancellation-free derivative proves the tail monotone");
        assert_eq!(result.location, ScoreOptimumLocation::UpperBoundary);
        assert_eq!(result.optimum.x, 700.0);
        assert!(
            result.resolution_flat_regions.is_empty(),
            "a strictly positive derivative should resolve before value-flat fallback"
        );
    }

    #[test]
    fn affine_reml_extreme_domain_one_direction_encloses_and_maximizes_repeatably() {
        // The normalized one-direction ridge profile behind the gam-sae
        // regressions has
        //
        //   R(lambda) = 10 - 4/(1 + lambda),
        //
        // so its exact residual stays in [6, 10] over the complete finite
        // lambda domain. Repeating the direction three times reproduces the
        // response multiplicity of that caller and plants the stationary point
        // at lambda/gamma_max = 0.6.
        let gram_modes = [1.0, 1.0, 1.0];
        let penalty_modes = [1.0, 1.0, 1.0];
        let projected_rhs_squared = [4.0 / 3.0, 4.0 / 3.0, 4.0 / 3.0];
        let response_energy = [10.0];
        let profile = AffineRemlProfile::new(
            &gram_modes,
            &penalty_modes,
            &projected_rhs_squared,
            &response_energy,
            15.0,
            3,
            0.0,
        )
        .expect("valid normalized one-direction ridge profile");
        let rho_lo = certified_ln_positive(f64::MIN_POSITIVE)
            .expect("finite-domain lower log bound")
            .lo;
        let rho_hi = certified_ln_positive(f64::MAX / 2.0)
            .expect("finite-domain upper log bound")
            .hi;

        let whole_domain = profile
            .enclose(rho_lo, rho_hi)
            .expect("scale-safe relative exp error keeps the full-domain residual finite");
        assert!(
            whole_domain.score.value.is_valid()
                && whole_domain.score.value.lo.is_finite()
                && whole_domain.score.value.hi.is_finite()
        );
        assert!(whole_domain.score.evaluation_error.is_finite());
        assert!(whole_domain.derivative.contains_zero());

        let resolution = f64::EPSILON.sqrt();
        let first = profile
            .maximize_value_ordered(rho_lo, rho_hi, resolution)
            .expect("finite subdivision must certify the planted stationary optimum");
        let repeated = profile
            .maximize_value_ordered(rho_lo, rho_hi, resolution)
            .expect("the same exact search must be repeatable");
        assert_eq!(first, repeated);
        let ScoreOptimumLocation::Stationary(index) = first.location else {
            panic!(
                "the planted one-direction optimum must be stationary, got {:?}",
                first.location
            );
        };
        let stationary = first
            .stationary_points
            .get(index)
            .expect("stationary result index");
        let expected = certified_ln_positive(0.6).expect("analytic stationary log");
        assert!(
            stationary.bracket.lo <= expected.lo && stationary.bracket.hi >= expected.hi,
            "certified bracket {:?} must contain analytic log(0.6) {:?}",
            stationary.bracket,
            expected
        );
        assert!(
            first.value_certificate.maximum_excess
                <= first.value_certificate.comparison_resolution,
            "an isolated stationary root is not yet a globally ordered score candidate: \
             maximum excess {}, comparison resolution {}, bracket {:?}",
            first.value_certificate.maximum_excess,
            first.value_certificate.comparison_resolution,
            stationary.bracket,
        );
    }

    #[test]
    fn affine_reml_gram_zero_subnormal_zero_projection_is_structural() {
        let minimum_subnormal = f64::from_bits(1);
        let log_lambda = -740.0;
        let lambda = exp_interval(log_lambda, log_lambda)
            .expect("the fixture needs a certified subnormal lambda");
        assert!(lambda.lo > 0.0 && lambda.hi < f64::MIN_POSITIVE);
        let raw_h = lambda.mul(ClosedInterval::point(minimum_subnormal));
        assert!(
            raw_h.lo < 0.0 && raw_h.hi > 0.0,
            "the raw outward product must cross rounded zero: {raw_h:?}"
        );
        let h = raw_h.nonnegative();
        assert_eq!(
            h.lo, 0.0,
            "known nonnegative product must clamp its outward lower bound to zero"
        );

        let ranges = mode_ranges(
            0.0,
            minimum_subnormal,
            0.0,
            lambda,
        )
        .expect("the zero projection cancels before any residual division");
        assert_eq!(ranges.c, ClosedInterval::point(0.0));
        assert_eq!(ranges.w, ClosedInterval::point(0.0));
        assert_eq!(ranges.v, ClosedInterval::point(0.0));
        assert_eq!(ranges.p, ClosedInterval::point(0.0));
        assert_eq!(ranges.q, ClosedInterval::point(0.0));

        let gram_modes = [0.0];
        let penalty_modes = [minimum_subnormal];
        let projected_rhs_squared = [0.0];
        let response_energy = [1.0];
        let profile = AffineRemlProfile::new(
            &gram_modes,
            &penalty_modes,
            &projected_rhs_squared,
            &response_energy,
            1.0,
            1,
            0.0,
        )
        .expect("valid gram-zero structural fixture");
        let jet = profile
            .evaluate(log_lambda)
            .expect("normalized determinant and zero residual projection stay finite");
        let enclosure = profile
            .enclose(log_lambda, log_lambda)
            .expect("the proof path must not divide by a zero-containing h interval");
        assert_eq!(jet.derivative, 0.0);
        assert_eq!(jet.curvature, 0.0);
        assert!(is_exact_zero(enclosure.derivative));
        assert!(is_exact_zero(enclosure.curvature));
        assert!(
            enclosure
                .score
                .value
                .widen(enclosure.score.evaluation_error)
                .contains(jet.value)
        );
    }

    #[test]
    fn affine_reml_gram_zero_subnormal_nonzero_projection_stays_finite() {
        let minimum_subnormal = f64::from_bits(1);
        let log_lambda = -740.0;
        let lambda = exp_interval(log_lambda, log_lambda)
            .expect("the fixture needs a certified subnormal lambda");
        let penalty = 0.01;
        let h = lambda
            .mul(ClosedInterval::point(penalty))
            .nonnegative();
        assert_eq!(
            h.lo, 0.0,
            "the fixture must enter the structural quotient path"
        );

        let ranges = mode_ranges(
            0.0,
            penalty,
            minimum_subnormal,
            lambda,
        )
        .expect("the scaled quotient has a finite representable range");
        assert_eq!(ranges.c, ClosedInterval::point(0.0));
        assert_eq!(ranges.w, ClosedInterval::point(0.0));
        assert!(ranges.v.lo > 0.0 && ranges.v.hi.is_finite());
        assert_eq!(ranges.p, ranges.v);
        assert_eq!(ranges.q, ranges.v.neg());

        let gram_modes = [0.0];
        let penalty_modes = [penalty];
        let projected_rhs_squared = [minimum_subnormal];
        let response_energy = [10.0];
        let profile = AffineRemlProfile::new(
            &gram_modes,
            &penalty_modes,
            &projected_rhs_squared,
            &response_energy,
            1.0,
            1,
            0.0,
        )
        .expect("valid gram-zero finite-ratio fixture");
        let jet = profile
            .evaluate(log_lambda)
            .expect("the point ratio must avoid the underflowing product");
        let enclosure = profile
            .enclose(log_lambda, log_lambda)
            .expect("the interval ratio must remain finite without a reciprocal overflow");
        assert!(
            enclosure
                .score
                .value
                .widen(enclosure.score.evaluation_error)
                .contains(jet.value)
        );
    }

    #[test]
    fn affine_reml_gram_zero_unrepresentable_projection_is_typed() {
        let minimum_subnormal = f64::from_bits(1);
        let log_lambda = -740.0;
        let gram_modes = [0.0];
        let penalty_modes = [minimum_subnormal];
        let projected_rhs_squared = [1.0];
        let response_energy = [10.0];
        let profile = AffineRemlProfile::new(
            &gram_modes,
            &penalty_modes,
            &projected_rhs_squared,
            &response_energy,
            1.0,
            1,
            0.0,
        )
        .expect("valid gram-zero refusal fixture");
        assert!(matches!(
            profile.evaluate(log_lambda),
            Err(AffineRemlError::ElementaryEnclosureUnavailable {
                function: "gram-zero residual quotient",
                ..
            })
        ));
        assert!(matches!(
            profile.enclose(log_lambda, log_lambda),
            Err(AffineRemlError::ElementaryEnclosureUnavailable {
                function: "gram-zero residual quotient",
                ..
            })
        ));
    }

    #[test]
    fn affine_reml_rejects_nonpositive_profile_residual() {
        let profile = AffineRemlProfile::new(&[1.0], &[1.0], &[2.0], &[1.0], 4.0, 1, 0.0)
            .expect("statically valid");
        assert!(matches!(
            profile.evaluate(-2.0),
            Err(AffineRemlError::NonPositiveResidual { .. })
        ));
    }
}
