//! Certified global optimization of one-dimensional scores on a bounded
//! domain, together with the affine-pencil spectral profile shared by the
//! Gaussian REML smoothing-parameter searches.
//!
//! Point samples alone cannot prove that a smooth function has no narrow
//! stationary pair between them.  The search therefore requires two pieces of
//! information from its caller:
//!
//! * a point evaluation `(value, first derivative, second derivative)`;
//! * an OUTER enclosure of both derivatives over every requested interval.
//!
//! The enclosure must contain the values the POINT EVALUATOR returns on the
//! interval, not merely the range of the ideal derivatives.  For an oracle that
//! pads endpoint jets with Lipschitz constants this is automatic.  For a
//! genuine interval extension it is not: an exact-range bound and a
//! floating-point jet are different currencies, and the search checks
//! containment at zero tolerance because that check is what keeps its
//! sign-based decisions consistent with its exclusion-based ones.  Such an
//! oracle owes the search a range widened by its own forward error — see
//! [`AffineRemlProfile::enclose`], which carries both together (#2513).
//!
//! An interval is discarded only when its first-derivative enclosure excludes
//! zero.  A stationary point is refined only after the second-derivative
//! enclosure excludes zero, proving that the first derivative is monotone and
//! hence that a straddling interval contains exactly one root.
//!
//! Two intervals need neither fact, because the object being certified is the
//! MAXIMUM and not the stationary structure: one whose derivative enclosure has
//! a constant sign is monotone, so its maximum is an endpoint already in hand;
//! and one already narrowed to the requested resolution over which the score
//! varies by less than one unit in the last place of its own magnitude cannot
//! be told from a constant by this arithmetic.  Both are closed as a
//! [`BoundedCell`] carrying a certified bound on what they could still hide,
//! summarized by [`ScoreSearchResult::value_uncertainty`].
//!
//! Every other interval is subdivided.  If floating-point spacing or the
//! caller-requested resolution is reached before any of the above is proved,
//! the result is a typed [`ScoreSearchError::Unresolved`] rather than a
//! best-effort optimum.
//!
//! [`AffineRemlProfile`] supplies both the point jets and rigorous interval
//! formulas for scores whose penalized Hessian has simultaneously diagonal
//! affine modes `h_i(lambda) = g_i + lambda s_i`.  This covers an ordinary
//! Demmler--Reinsch eigensystem (`g_i = 1`) and a reference-Hessian pencil
//! (`g_i = 1 - lambda_0 mu_i`, `s_i = mu_i`) without any matrix dependency in
//! this crate.

use std::fmt;

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

    /// Construct an interval and round both supplied bounds one representable
    /// value outward.  This is the public bridge for callers that derive a
    /// real-valued bound with ordinary nearest-rounded scalar arithmetic.
    #[inline]
    pub fn outward(lo: f64, hi: f64) -> Self {
        Self {
            lo: next_down(lo),
            hi: next_up(hi),
        }
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
    fn add(self, other: Self) -> Self {
        Self {
            lo: sum_down(self.lo, other.lo),
            hi: sum_up(self.hi, other.hi),
        }
    }

    #[inline]
    fn sub(self, other: Self) -> Self {
        Self {
            lo: sum_down(self.lo, -other.hi),
            hi: sum_up(self.hi, -other.lo),
        }
    }

    #[inline]
    fn neg(self) -> Self {
        // Negation is exact.
        Self {
            lo: -self.hi,
            hi: -self.lo,
        }
    }

    fn mul(self, other: Self) -> Self {
        let pairs = [
            (self.lo, other.lo),
            (self.lo, other.hi),
            (self.hi, other.lo),
            (self.hi, other.hi),
        ];
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for (a, b) in pairs {
            lo = lo.min(product_down(a, b));
            hi = hi.max(product_up(a, b));
        }
        Self { lo, hi }
    }

    #[inline]
    fn scale(self, value: f64) -> Self {
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

    #[inline]
    fn nonnegative(self) -> Self {
        Self {
            lo: self.lo.max(0.0),
            hi: self.hi.max(0.0),
        }
    }
}

/// Value and the analytic derivatives at one abscissa.
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

/// Outer derivative ranges supplied to the certified search.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DerivativeEnclosure {
    pub derivative: ClosedInterval,
    pub curvature: ClosedInterval,
}

/// One stationary point together with the final bracket that certifies its
/// location.  The bracket width is no larger than the requested resolution,
/// unless the point was represented exactly (a zero-width bracket).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StationaryPoint {
    pub sample: ScoreSample,
    pub bracket: ClosedInterval,
}

/// A cell closed on the VALUE side instead of by isolating its stationary
/// structure.
///
/// Two situations produce one.  A cell whose derivative enclosure has a
/// constant sign but touches zero is monotone: its maximum IS an endpoint, and
/// `excess` is the representational floor alone.  A cell over which the score
/// is flat to that same floor cannot be told apart from a constant, and every
/// point of it maximizes the score to the resolution the arithmetic can carry.
///
/// Neither is a best-effort answer.  `excess` is a certified bound on how much
/// the cell's true maximum can exceed `sample.value`, derived from the cell's
/// endpoint values and its derivative enclosure, so the search's global
/// statement survives intact — see [`ScoreSearchResult::value_uncertainty`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BoundedCell {
    /// The better of the cell's two endpoint samples.
    pub sample: ScoreSample,
    pub cell: ClosedInterval,
    /// Certified bound on `max_{x in cell} V(x) - sample.value`.
    pub excess: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScoreOptimumLocation {
    LowerBoundary,
    UpperBoundary,
    Stationary(usize),
    /// The optimum is the representative of a cell closed on the value side;
    /// the index selects it from [`ScoreSearchResult::bounded_cells`].
    BoundedCell(usize),
}

/// Complete successful search result.  Endpoints are retained explicitly so
/// the global comparison is independently checkable by the caller.
#[derive(Clone, Debug, PartialEq)]
pub struct ScoreSearchResult {
    pub optimum: ScoreSample,
    pub location: ScoreOptimumLocation,
    pub lower_boundary: ScoreSample,
    pub upper_boundary: ScoreSample,
    pub stationary_points: Vec<StationaryPoint>,
    /// Cells closed on the value side; see [`BoundedCell`].
    pub bounded_cells: Vec<BoundedCell>,
    /// Certified bound on `global maximum - optimum.value`.
    ///
    /// Zero when every cell was excluded or isolated, which is the classical
    /// guarantee.  Otherwise it is the largest amount by which any cell closed
    /// on the value side could still hide a better point — a number the caller
    /// can compare its own tolerances against instead of having to assume one.
    ///
    /// It is stated in the currency the search works in: the score AS
    /// EVALUATED.  Every comparison the search makes, this one included, is
    /// between values the oracle returned, so the oracle's own error in the
    /// VALUE is not represented here.  This is the uncertainty the SEARCH adds.
    pub value_uncertainty: f64,
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
    EnclosureMissesEndpoint {
        lo: f64,
        hi: f64,
        endpoint: ScoreSample,
        enclosure: DerivativeEnclosure,
    },
    /// The enclosure still admits both a stationary point and a curvature
    /// zero, so uniqueness could not be proved before the requested or
    /// floating-point resolution floor — AND the cell's own values are not
    /// flat enough for the value side to close it either.
    Unresolved {
        lo: f64,
        hi: f64,
        requested_resolution: f64,
        enclosure: DerivativeEnclosure,
        /// Certified bound on how much the cell's interior can exceed its
        /// better endpoint, and the floor it had to fall below to be closed on
        /// the value side.  A refusal carries the quantity it was decided
        /// against.
        value_excess: f64,
        value_floor: f64,
    },
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
                "score search: derivative enclosure failed on [{lo}, {hi}]: {source}"
            ),
            Self::NonFiniteSample { sample } => write!(
                f,
                "score search: non-finite jet at {} (value {}, derivative {}, curvature {})",
                sample.x, sample.value, sample.derivative, sample.curvature
            ),
            Self::InvalidEnclosure { lo, hi, enclosure } => write!(
                f,
                "score search: invalid derivative enclosure on [{lo}, {hi}]: {enclosure:?}"
            ),
            Self::EnclosureMissesEndpoint {
                lo,
                hi,
                endpoint,
                enclosure,
            } => write!(
                f,
                "score search: enclosure on [{lo}, {hi}] misses endpoint jet at {}: {endpoint:?} not in {enclosure:?}",
                endpoint.x
            ),
            Self::Unresolved {
                lo,
                hi,
                requested_resolution,
                enclosure,
                value_excess,
                value_floor,
            } => write!(
                f,
                "score search: stationary structure unresolved on [{lo}, {hi}] at requested resolution {requested_resolution}: {enclosure:?}; the cell's interior can still exceed its better endpoint by {value_excess} against a value floor of {value_floor}"
            ),
        }
    }
}

impl<E: std::error::Error + 'static> std::error::Error for ScoreSearchError<E> {}

#[derive(Clone, Copy)]
struct SearchNode {
    left: ScoreSample,
    right: ScoreSample,
}

fn evaluate_sample<E, F>(x: f64, evaluate: &mut F) -> Result<ScoreSample, ScoreSearchError<E>>
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
    if sample.value.is_finite() && sample.derivative.is_finite() && sample.curvature.is_finite() {
        Ok(sample)
    } else {
        Err(ScoreSearchError::NonFiniteSample { sample })
    }
}

fn checked_enclosure<E, F>(
    node: SearchNode,
    enclose: &mut F,
) -> Result<DerivativeEnclosure, ScoreSearchError<E>>
where
    F: FnMut(ScoreSample, ScoreSample) -> Result<DerivativeEnclosure, E>,
{
    let lo = node.left.x;
    let hi = node.right.x;
    // The cell's endpoints are handed to the oracle as the SAMPLES the search
    // already paid for, not as bare abscissae. An endpoint-anchored enclosure
    // needs the endpoint jets and nothing else, so this is what makes it free:
    // the oracle reads `left`/`right` instead of re-evaluating the criterion at
    // two points it has already evaluated.
    let enclosure = enclose(node.left, node.right)
        .map_err(|source| ScoreSearchError::EnclosureEvaluation { lo, hi, source })?;
    if !(enclosure.derivative.is_valid() && enclosure.curvature.is_valid()) {
        return Err(ScoreSearchError::InvalidEnclosure { lo, hi, enclosure });
    }
    for endpoint in [node.left, node.right] {
        if !(enclosure.derivative.contains(endpoint.derivative)
            && enclosure.curvature.contains(endpoint.curvature))
        {
            return Err(ScoreSearchError::EnclosureMissesEndpoint {
                lo,
                hi,
                endpoint,
                enclosure,
            });
        }
    }
    Ok(enclosure)
}

/// Refine a UNIQUE derivative root.  The caller has already proved uniqueness
/// by a curvature enclosure that excludes zero and supplied endpoint
/// derivatives of opposite sign.
fn refine_unique_root<E, F>(
    mut left: ScoreSample,
    mut right: ScoreSample,
    resolution: f64,
    enclosure: DerivativeEnclosure,
    evaluate: &mut F,
) -> Result<StationaryPoint, ScoreSearchError<E>>
where
    F: FnMut(f64) -> Result<ScoreJet, E>,
{
    // A unique-root refinement is only meaningful on a strict sign-change
    // bracket; anything else is a caller error surfaced as a typed rejection.
    if left.derivative == 0.0
        || right.derivative == 0.0
        || left.derivative.is_sign_positive() == right.derivative.is_sign_positive()
    {
        return Err(ScoreSearchError::InvalidEnclosure {
            lo: left.x,
            hi: right.x,
            enclosure,
        });
    }

    while right.x - left.x > resolution {
        let width = right.x - left.x;
        let midpoint = left.x + 0.5 * width;
        if !(midpoint > left.x && midpoint < right.x) {
            // The bracket is two adjacent doubles.  The root is isolated as
            // tightly as the representation admits, which is TIGHTER than the
            // caller asked for — not a failure to reach the request.
            break;
        }

        // Newton is accepted only in the central half of the bracket.  Thus
        // every accepted point, Newton or midpoint, contracts the maintained
        // sign bracket by at least one quarter.  The loop has no iteration cap
        // because its geometric termination follows from this safeguard.
        let base = if left.derivative.abs() <= right.derivative.abs() {
            left
        } else {
            right
        };
        let newton = if base.curvature != 0.0 {
            base.x - base.derivative / base.curvature
        } else {
            f64::NAN
        };
        let guard = 0.25 * width;
        let mut x = if newton.is_finite() && newton >= left.x + guard && newton <= right.x - guard {
            newton
        } else {
            midpoint
        };
        if !(x > left.x && x < right.x) {
            // `left.x + guard` can round back onto `left.x` on a bracket only a
            // few ulps wide, admitting a Newton step that is not interior.  The
            // midpoint is known interior here, so fall back to it rather than
            // abandoning a bracket that is still contracting.
            x = midpoint;
        }
        let sample = evaluate_sample(x, evaluate)?;
        if sample.derivative == 0.0 {
            return Ok(StationaryPoint {
                sample,
                bracket: ClosedInterval::point(x),
            });
        }
        if sample.derivative.is_sign_positive() == left.derivative.is_sign_positive() {
            left = sample;
        } else {
            right = sample;
        }
    }

    // The BRACKET is the certificate; the returned sample is the best ESTIMATE
    // inside it, and the two are not the same thing.  Reporting the midpoint of
    // the final bracket threw away every iterate the refinement had already
    // paid for: the derivative is monotone here (that is what certified the
    // root), so each accepted sample is closer to the root than the endpoint it
    // replaced, and after a Newton step the winning endpoint is typically
    // within an ulp of the root while the midpoint is a full half-resolution
    // away.  Measured on the #2513 fixture at resolution sqrt(eps), the
    // midpoint answered lambda=1.2000000050 where the bracket endpoint answers
    // 1.2000000000 — a 5e-9 error on a root the search had already located to
    // 1e-16.
    let bracket = ClosedInterval::new(left.x, right.x);
    let best_endpoint = if left.derivative.abs() <= right.derivative.abs() {
        left
    } else {
        right
    };
    // One false-position step costs the same single evaluation the midpoint
    // used to cost and dominates both endpoints: on a pure-bisection bracket it
    // IS the midpoint, and on a Newton-contracted one it lands on the root.
    let total = left.derivative.abs() + right.derivative.abs();
    let interpolated = left.x + (right.x - left.x) * (left.derivative.abs() / total);
    let sample =
        if total.is_finite() && total > 0.0 && interpolated > left.x && interpolated < right.x {
            let candidate = evaluate_sample(interpolated, evaluate)?;
            if candidate.derivative.abs() < best_endpoint.derivative.abs() {
                candidate
            } else {
                best_endpoint
            }
        } else {
            best_endpoint
        };
    Ok(StationaryPoint { sample, bracket })
}

/// Globally maximize a smooth score on `[lo, hi]` by certified stationary
/// isolation.
///
/// `evaluate` returns the score jet at a point. `enclose(a, b)` receives the
/// cell's two ENDPOINT SAMPLES — the jets the search already obtained from
/// `evaluate` — and must return OUTER ranges containing the first and second
/// derivative at every point of `[a.x, b.x]`.  The search additionally checks
/// that both endpoint jets lie inside every returned enclosure.
///
/// Handing the samples in (rather than the bare abscissae) is what keeps an
/// endpoint-anchored enclosure free: such an oracle is a Taylor pad around the
/// endpoint jets, so with the jets in hand it performs no criterion evaluation
/// of its own. An oracle whose enclosure is a genuine interval extension may
/// ignore the jets and use `a.x`/`b.x`.
///
/// There is no evaluation or subdivision budget.  A successful return means
/// every cell was excluded, isolated to `resolution`, or closed on the value
/// side with a certified bound on what it could still hide — the last case is
/// summarized by [`ScoreSearchResult::value_uncertainty`], which is zero when
/// it did not arise.  Any cell that none of the three can close produces
/// [`ScoreSearchError::Unresolved`].
///
/// The value side matters because THIS IS A MAXIMIZER.  Isolating stationary
/// structure is the means; bounding the maximum is the end.  A cell whose
/// derivative enclosure has a constant sign — even one that touches zero, so
/// the derivative test cannot discard it — attains its maximum at an endpoint
/// the search has already evaluated.  And a cell already narrowed to
/// `resolution` over which the score varies by less than one unit in the last
/// place of its own magnitude cannot be told apart from a constant by any
/// arithmetic this search can perform.  Refusing there is not conservatism; it
/// is asking arithmetic for a distinction it does not carry (#2513).
///
/// Flatness closes a cell only once it is already at `resolution`, never
/// earlier.  The caller asked for that abscissa accuracy, and flatness improves
/// only the VALUE — a wide flat cell would answer a question nobody asked.
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
        return Ok(ScoreSearchResult {
            optimum: lower_boundary,
            location: ScoreOptimumLocation::LowerBoundary,
            lower_boundary,
            upper_boundary: lower_boundary,
            stationary_points: Vec::new(),
            bounded_cells: Vec::new(),
            value_uncertainty: 0.0,
        });
    }
    let upper_boundary = evaluate_sample(hi, &mut evaluate)?;
    let (mut optimum, mut location) = if upper_boundary.value > lower_boundary.value {
        (upper_boundary, ScoreOptimumLocation::UpperBoundary)
    } else {
        (lower_boundary, ScoreOptimumLocation::LowerBoundary)
    };

    let mut stationary_points = Vec::<StationaryPoint>::new();
    let mut bounded_cells = Vec::<BoundedCell>::new();
    let mut stack = vec![SearchNode {
        left: lower_boundary,
        right: upper_boundary,
    }];
    while let Some(node) = stack.pop() {
        let enclosure = checked_enclosure(node, &mut enclose)?;
        if !enclosure.derivative.contains_zero() {
            continue;
        }

        let monotone = !enclosure.curvature.contains_zero();
        if monotone {
            let stationary = if node.left.derivative == 0.0 {
                Some(StationaryPoint {
                    sample: node.left,
                    bracket: ClosedInterval::point(node.left.x),
                })
            } else if node.right.derivative == 0.0 {
                Some(StationaryPoint {
                    sample: node.right,
                    bracket: ClosedInterval::point(node.right.x),
                })
            } else if node.left.derivative.is_sign_positive()
                != node.right.derivative.is_sign_positive()
            {
                Some(refine_unique_root(
                    node.left,
                    node.right,
                    resolution,
                    enclosure,
                    &mut evaluate,
                )?)
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
            }
            continue;
        }

        let width = node.right.x - node.left.x;
        let midpoint = node.left.x + 0.5 * width;
        let at_floor = width <= resolution || !(midpoint > node.left.x && midpoint < node.right.x);

        // The value side, tried after the monotone branch so a well-conditioned
        // optimum is still located by root refinement.
        let value_floor = f64::EPSILON
            * lower_boundary
                .value
                .abs()
                .max(upper_boundary.value.abs())
                .max(optimum.value.abs())
                .max(node.left.value.abs())
                .max(node.right.value.abs());
        let excess = cell_excess_bound(node.left, node.right, enclosure.derivative);
        // A one-signed enclosure closes the cell at ANY width: the maximum IS
        // an endpoint, exactly, and no arithmetic was involved in saying so.
        // Flatness only closes a cell the search has already subdivided to the
        // caller's `resolution` — the caller asked for that abscissa accuracy,
        // and a cell being flat improves the VALUE nothing further can improve
        // while saying nothing about the location.  Closing a wide flat cell
        // early would answer a question the caller did not ask.
        if excess.exact || (at_floor && excess.bound <= value_floor) {
            let sample = if node.right.value > node.left.value {
                node.right
            } else {
                node.left
            };
            let index = bounded_cells.len();
            if sample.value > optimum.value {
                optimum = sample;
                location = ScoreOptimumLocation::BoundedCell(index);
            }
            bounded_cells.push(BoundedCell {
                sample,
                cell: ClosedInterval::new(node.left.x, node.right.x),
                // The crossing formula is itself evaluated in double precision,
                // so its own roundoff is of order the floor.  Charging the
                // floor makes the recorded bound one the arithmetic can stand
                // behind rather than one it merely computed.  The exact branch
                // needs no such charge, and gets none.
                excess: if excess.exact {
                    0.0
                } else {
                    excess.bound + value_floor
                },
            });
            continue;
        }

        if at_floor {
            return Err(ScoreSearchError::Unresolved {
                lo: node.left.x,
                hi: node.right.x,
                requested_resolution: resolution,
                enclosure,
                value_excess: excess.bound,
                value_floor,
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

    // The only place the reported optimum can be beaten is inside a cell that
    // was closed without isolating it, and only by that cell's own certified
    // excess over the sample it was closed on.
    let value_uncertainty = bounded_cells
        .iter()
        .map(|cell| (cell.sample.value + cell.excess - optimum.value).max(0.0))
        .fold(0.0_f64, f64::max);

    Ok(ScoreSearchResult {
        optimum,
        location,
        lower_boundary,
        upper_boundary,
        stationary_points,
        bounded_cells,
        value_uncertainty,
    })
}

/// Certified bound on how much the maximum of a smooth score over `[a, b]` can
/// exceed the larger of its two endpoint values, given an OUTER enclosure of
/// the derivative on the cell.
///
/// For `x` in the cell, `t = x - a` and `w = b - a`, the mean value theorem
/// gives BOTH `V(x) <= V(a) + d_hi*t` (rising from the left) and
/// `V(x) <= V(b) - d_lo*(w - t)` (falling back to the right).  The cell's
/// maximum is under the smaller of the two lines everywhere, so it is under
/// their crossing.  On the symmetric case `V(a) = V(b) = v` this collapses to
/// the classical `v + w*(d_hi - d_lo)/4`, which shrinks QUADRATICALLY as the
/// cell does — so a cell that is flat at the resolution floor is flat by orders
/// of magnitude, not marginally, while a genuine peak stays well clear.
///
/// A one-signed derivative enclosure gives exactly zero: the cell is monotone
/// and its maximum is an endpoint the search already holds.  `exact` reports
/// which of the two cases produced the bound, because the caller charges the
/// crossing formula's own roundoff and must not charge the structural answer's.
struct CellExcess {
    bound: f64,
    exact: bool,
}

fn cell_excess_bound(
    left: ScoreSample,
    right: ScoreSample,
    derivative: ClosedInterval,
) -> CellExcess {
    let width = right.x - left.x;
    if !(width > 0.0) || derivative.lo >= 0.0 || derivative.hi <= 0.0 {
        return CellExcess {
            bound: 0.0,
            exact: true,
        };
    }
    if !(derivative.lo.is_finite() && derivative.hi.is_finite()) {
        return CellExcess {
            bound: f64::INFINITY,
            exact: false,
        };
    }
    let span = derivative.hi - derivative.lo;
    let crossing = (((right.value - left.value) - derivative.lo * width) / span).clamp(0.0, width);
    let ceiling = (left.value + derivative.hi * crossing)
        .min(right.value - derivative.lo * (width - crossing));
    let excess = ceiling - left.value.max(right.value);
    CellExcess {
        bound: if excess.is_nan() {
            f64::INFINITY
        } else {
            excess.max(0.0)
        },
        exact: false,
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
#[derive(Clone, Copy, Debug)]
pub struct AffineRemlProfile<'a> {
    gram_modes: &'a [f64],
    penalty_modes: &'a [f64],
    projected_rhs_squared: &'a [f64],
    response_energy: &'a [f64],
    residual_dof: f64,
    logdet_constant: f64,
}

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
        Ok(Self {
            gram_modes,
            penalty_modes,
            projected_rhs_squared,
            response_energy,
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

    /// Score value, first derivative, and second derivative in `log(lambda)`.
    ///
    /// Every quantity is formed through the SAME normalized mode variable
    /// `t = lambda*s/g` that [`Self::enclose`] evaluates on intervals, so the
    /// point evaluator and the interval extension are one expression graph
    /// walked two ways rather than two independent derivations that happen to
    /// agree in the middle of the domain.  Three cancellations that the
    /// textbook form carries are absent as a result, and all three used to be
    /// fatal at the ends of the representable log-lambda domain:
    ///
    /// * `sum_i log h_i - rank*log(lambda)` differences two numbers of size
    ///   `rank*|log lambda|` (up to ~2100 at the domain edge) to recover a
    ///   result of size `e^-|log lambda|`.  Per mode this is instead
    ///   `log(h/lambda) = log(g/lambda + s)`, evaluated in whichever of its two
    ///   factorizations keeps the added term below one.
    /// * `sum_i u_i - rank` differences `rank` from a sum of numbers that each
    ///   round to exactly `1.0` once `lambda*s >> g`, so the result is a
    ///   multiple of `rank*eps` where the true value is `sum_i g_i/h_i`.  Per
    ///   mode this is instead `-c_i` with `c_i = 1 - u_i = g_i/h_i` computed
    ///   from `t` directly.
    /// * `u(1-u)` recovers nothing once `u` rounds to one; it is `u*c` here.
    pub fn evaluate(&self, log_lambda: f64) -> Result<ScoreJet, AffineRemlError> {
        if !log_lambda.is_finite() {
            return Err(AffineRemlError::InvalidLogLambda { value: log_lambda });
        }
        let lambda = log_lambda.exp();
        if !(lambda.is_finite() && lambda > 0.0) {
            return Err(AffineRemlError::InvalidLogLambda { value: log_lambda });
        }

        let modes = self.num_modes();
        let mut normalized_logdet = self.logdet_constant;
        let mut determinant_derivative = 0.0;
        let mut determinant_curvature = 0.0;
        // Mode-major so the kernels are formed once and shared by every
        // response, instead of once per (mode, response) pair.
        let mut residual: Vec<f64> = self.response_energy.to_vec();
        let mut first = vec![0.0_f64; residual.len()];
        let mut second = vec![0.0_f64; residual.len()];
        for (index, (&gram, &penalty)) in self.gram_modes.iter().zip(self.penalty_modes).enumerate()
        {
            let point = mode_point(gram, penalty, lambda, log_lambda);
            if !(point.determinant_first.is_finite()
                && point.determinant_second.is_finite()
                && point.normalized_log_h.is_finite())
            {
                return Err(AffineRemlError::NonPositiveMode {
                    index,
                    log_lambda,
                    value: lambda.mul_add(penalty, gram),
                });
            }
            normalized_logdet += point.normalized_log_h;
            determinant_derivative += point.determinant_first;
            determinant_curvature += point.determinant_second;
            for output in 0..residual.len() {
                let projected_square = self.projected_rhs_squared[output * modes + index];
                residual[output] -= projected_square * point.fitted;
                first[output] += projected_square * point.first;
                second[output] += projected_square * point.second;
            }
        }

        let mut residual_log_sum = 0.0;
        let mut residual_derivative_sum = 0.0;
        let mut residual_curvature_sum = 0.0;
        for output in 0..residual.len() {
            let residual = residual[output];
            if !(residual.is_finite() && residual > 0.0) {
                return Err(AffineRemlError::NonPositiveResidual {
                    output,
                    log_lambda,
                    value: residual,
                });
            }
            let log_derivative = first[output] / residual;
            residual_log_sum += (residual / self.residual_dof).ln();
            residual_derivative_sum += log_derivative;
            residual_curvature_sum += second[output] / residual - log_derivative * log_derivative;
        }

        let outputs = self.num_responses() as f64;
        Ok(ScoreJet {
            value: -0.5 * (outputs * normalized_logdet + self.residual_dof * residual_log_sum),
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

    /// Outward enclosure of the first two score derivatives on a bounded
    /// log-lambda interval.
    ///
    /// The returned ranges enclose the IMPLEMENTED jet, not only the ideal
    /// one: every mode kernel is carried as a [`RoundedRange`], so the range of
    /// the exact derivative and a bound on [`Self::evaluate`]'s own forward
    /// error travel together and the result is widened by the latter.  That is
    /// what makes the search's zero-tolerance endpoint-containment check a
    /// property this oracle can actually have rather than a currency mismatch
    /// between an exact-range enclosure and a floating-point point jet (#2513).
    pub fn enclose(&self, lo: f64, hi: f64) -> Result<DerivativeEnclosure, AffineRemlError> {
        if !(lo.is_finite() && hi.is_finite() && lo <= hi) {
            return Err(AffineRemlError::InvalidLogLambdaInterval { lo, hi });
        }
        let lambda = ClosedInterval::new(next_down(lo.exp()), next_up(hi.exp()));
        if !(lambda.lo.is_finite() && lambda.lo > 0.0 && lambda.hi.is_finite()) {
            return Err(AffineRemlError::InvalidLogLambdaInterval { lo, hi });
        }

        let modes = self.num_modes();
        let responses = self.num_responses();
        let mut determinant_first = RoundedSum::seeded(ClosedInterval::point(0.0));
        let mut determinant_second = RoundedSum::seeded(ClosedInterval::point(0.0));
        let mut fitted_quadratic: Vec<RoundedSum> = self
            .response_energy
            .iter()
            .map(|&energy| RoundedSum::seeded(ClosedInterval::point(energy)))
            .collect();
        let mut response_first = vec![RoundedSum::seeded(ClosedInterval::point(0.0)); responses];
        let mut response_second = vec![RoundedSum::seeded(ClosedInterval::point(0.0)); responses];
        for i in 0..modes {
            let ranges = mode_ranges(self.gram_modes[i], self.penalty_modes[i], lambda);
            determinant_first.add(ranges.determinant_first);
            determinant_second.add(ranges.determinant_second);
            for output in 0..responses {
                let projected = self.projected_rhs_squared[output * modes + i];
                // One rounding for the `projected * kernel` product itself; the
                // kernel's own error scales with the same exact factor.
                fitted_quadratic[output].sub(ranges.fitted.scaled(projected));
                response_first[output].add(ranges.first.scaled(projected));
                response_second[output].add(ranges.second.scaled(projected));
            }
        }
        let determinant_first = determinant_first.finish();
        let determinant_second = determinant_second.finish();

        let mut residual_first_sum = RoundedSum::seeded(ClosedInterval::point(0.0));
        let mut residual_second_sum = RoundedSum::seeded(ClosedInterval::point(0.0));
        for output in 0..responses {
            let first = response_first[output].finish();
            let second = response_second[output].finish();
            let residual = fitted_quadratic[output].finish();
            if !(residual.range.lo > 0.0 && residual.range.is_valid()) {
                return Err(AffineRemlError::NonPositiveResidualInterval {
                    output,
                    lo,
                    hi,
                    lower_bound: residual.range.lo,
                });
            }
            let first_ratio = first.div_positive(residual).nonnegative();
            let second_ratio = second.div_positive(residual);
            residual_first_sum.add(first_ratio);
            residual_second_sum.add(second_ratio.sub(first_ratio.square()));
        }
        let residual_first_sum = residual_first_sum.finish();
        let residual_second_sum = residual_second_sum.finish();

        let outputs = responses as f64;
        let derivative = determinant_first
            .scaled(outputs)
            .add(residual_first_sum.scaled(self.residual_dof))
            .scaled(-0.5);
        let curvature = determinant_second
            .scaled(outputs)
            .add(residual_second_sum.scaled(self.residual_dof))
            .scaled(-0.5);
        Ok(DerivativeEnclosure {
            derivative: derivative.observable(),
            curvature: curvature.observable(),
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
}

/// The relative error bound of one correctly rounded double operation.
const UNIT_ROUNDOFF: f64 = f64::EPSILON / 2.0;

/// Wilkinson's `gamma_k = k*u/(1 - k*u)`: the relative error bound accumulated
/// by `k` chained roundings.  Saturates to infinity rather than turning
/// negative once `k*u >= 1/2`, which would take ~2e15 modes.
fn gamma(operations: usize) -> f64 {
    let scaled = operations as f64 * UNIT_ROUNDOFF;
    if scaled >= 0.5 {
        f64::INFINITY
    } else {
        scaled / (1.0 - scaled)
    }
}

#[inline]
fn magnitude_of(range: ClosedInterval) -> f64 {
    range.lo.abs().max(range.hi.abs())
}

/// A range of an exact quantity over a cell, together with a bound on the
/// forward error the POINT evaluator makes computing that same quantity
/// anywhere in the cell.
///
/// This pairing is the whole content of #2513.  An interval extension bounds
/// the range of the EXACT function; the search compares that bound against
/// `fl(f(x))`, and no theorem says an exact-range enclosure contains an
/// inexact point evaluation.  Carrying the evaluator's own error alongside the
/// range makes the two commensurable: [`Self::observable`] is the set of values
/// the implemented evaluator may return on the cell, which is what
/// `checked_enclosure` is entitled to demand at zero tolerance.
///
/// Widening is also the safe direction for every decision the search makes.  A
/// wider derivative range excludes zero less often, so fewer cells are
/// discarded; a wider curvature range admits zero more readily, so uniqueness
/// is certified less often.  The pad can cost a certificate; it cannot mint one.
#[derive(Clone, Copy, Debug)]
struct RoundedRange {
    range: ClosedInterval,
    error: f64,
}

impl RoundedRange {
    #[inline]
    fn new(range: ClosedInterval, error: f64) -> Self {
        Self { range, error }
    }

    /// A quantity the point evaluator obtains through `operations` chained
    /// roundings with no cancelling subtraction, so its error is relative.
    #[inline]
    fn rounded(range: ClosedInterval, operations: usize) -> Self {
        Self {
            range,
            error: gamma(operations) * magnitude_of(range),
        }
    }

    #[inline]
    fn exact(range: ClosedInterval) -> Self {
        Self { range, error: 0.0 }
    }

    #[inline]
    fn magnitude(self) -> f64 {
        magnitude_of(self.range)
    }

    /// Bound on the magnitude of the value the POINT evaluator holds, which is
    /// what the next rounding is charged against — one step further out than
    /// the exact range.
    #[inline]
    fn bound(self) -> f64 {
        self.magnitude() + self.error
    }

    #[inline]
    fn scaled(self, factor: f64) -> Self {
        let range = self.range.scale(factor);
        let propagated = factor.abs() * self.error;
        Self {
            range,
            error: propagated + UNIT_ROUNDOFF * (magnitude_of(range) + propagated),
        }
    }

    #[inline]
    fn add(self, other: Self) -> Self {
        Self {
            range: self.range.add(other.range),
            error: self.error + other.error + UNIT_ROUNDOFF * (self.bound() + other.bound()),
        }
    }

    #[inline]
    fn sub(self, other: Self) -> Self {
        Self {
            range: self.range.sub(other.range),
            error: self.error + other.error + UNIT_ROUNDOFF * (self.bound() + other.bound()),
        }
    }

    #[inline]
    fn square(self) -> Self {
        let range = self.range.square();
        let propagated = (2.0 * self.magnitude() + self.error) * self.error;
        Self {
            range,
            error: propagated + UNIT_ROUNDOFF * (magnitude_of(range) + propagated),
        }
    }

    /// `self / denominator` where the denominator's EXACT range is strictly
    /// positive.  The point evaluator divides by its own rounded denominator,
    /// which this can only place above `range.lo - error`; when that floor is
    /// not positive the quotient's forward error is genuinely unbounded and
    /// saying so is the honest answer.  The resulting entire enclosure costs
    /// the search a subdivision, never a wrong certificate.
    fn div_positive(self, denominator: Self) -> Self {
        let range = self.range.div_positive(denominator.range);
        let quotient = magnitude_of(range);
        let floor = denominator.range.lo - denominator.error;
        let error = if floor > 0.0 {
            let propagated = (self.error + quotient * denominator.error) / floor;
            propagated + UNIT_ROUNDOFF * (quotient + propagated)
        } else {
            f64::INFINITY
        };
        Self { range, error }
    }

    #[inline]
    fn nonnegative(self) -> Self {
        Self {
            range: self.range.nonnegative(),
            error: self.error,
        }
    }

    /// The set of values the POINT evaluator may return on this cell.
    fn observable(self) -> ClosedInterval {
        if !(self.error >= 0.0) {
            return ClosedInterval::entire();
        }
        ClosedInterval::new(
            next_down(self.range.lo - self.error),
            next_up(self.range.hi + self.error),
        )
    }
}

/// A sequentially accumulated sum, carrying what Wilkinson's bound needs: the
/// sum of the per-term MAGNITUDES, which is what the accumulated roundings are
/// charged against, rather than the (possibly cancelled) total.
#[derive(Clone, Copy, Debug)]
struct RoundedSum {
    range: ClosedInterval,
    magnitude: f64,
    error: f64,
    additions: usize,
}

impl RoundedSum {
    fn seeded(seed: ClosedInterval) -> Self {
        Self {
            range: seed,
            magnitude: magnitude_of(seed),
            error: 0.0,
            additions: 0,
        }
    }

    fn add(&mut self, term: RoundedRange) {
        self.range = self.range.add(term.range);
        // The roundings are charged against what the evaluator actually held,
        // which is the exact term plus its own error.
        self.magnitude += term.bound();
        self.error += term.error;
        self.additions += 1;
    }

    fn sub(&mut self, term: RoundedRange) {
        self.range = self.range.sub(term.range);
        self.magnitude += term.bound();
        self.error += term.error;
        self.additions += 1;
    }

    fn finish(self) -> RoundedRange {
        RoundedRange::new(
            self.range,
            self.error + gamma(self.additions) * self.magnitude,
        )
    }
}

/// One mode's contribution to the score jet, per unit `projected_square`.
///
/// The interval counterpart is [`ModeRanges`], field for field.
#[derive(Clone, Copy)]
struct ModePoint {
    /// Contribution to `d/dlog(lambda) [sum_i log h_i - rank*log lambda]`,
    /// namely `u - 1 = -c` for a penalized mode and `0` otherwise.
    determinant_first: f64,
    /// Contribution to its second derivative, `u*c`.
    determinant_second: f64,
    /// `log(h/lambda)` for a penalized mode, `log h` for an unpenalized one;
    /// summing this over the modes gives `sum_i log h_i - rank*log lambda`
    /// with no cancellation to undo.
    normalized_log_h: f64,
    /// `1/h`: the fitted-energy weight, per unit `projected_square`.
    fitted: f64,
    /// `lambda s / h^2`, per unit `projected_square`.
    first: f64,
    /// `lambda s (g - lambda s) / h^3`, per unit `projected_square`.
    second: f64,
}

fn mode_point(gram: f64, penalty: f64, lambda: f64, log_lambda: f64) -> ModePoint {
    if penalty == 0.0 {
        return ModePoint {
            determinant_first: 0.0,
            determinant_second: 0.0,
            normalized_log_h: gram.ln(),
            fitted: 1.0 / gram,
            first: 0.0,
            second: 0.0,
        };
    }
    if gram == 0.0 {
        let fitted = 1.0 / (lambda * penalty);
        return ModePoint {
            determinant_first: 0.0,
            determinant_second: 0.0,
            normalized_log_h: penalty.ln(),
            fitted,
            first: fitted,
            second: -fitted,
        };
    }

    let t = lambda * (penalty / gram);
    let kernels = kernel_point(t);
    // `log(h/lambda) = log(g/lambda + s)`.  Both factorizations below add a
    // term bounded by one to a logarithm, so neither cancels: `t <= 1` is the
    // small-lambda half where `g/lambda` dominates, `t > 1` the large-lambda
    // half where `s` does.  Neither forms `g/lambda` itself, which overflows
    // at the bottom of the log-lambda domain.
    let normalized_log_h = if t <= 1.0 {
        gram.ln() - log_lambda + t.ln_1p()
    } else {
        penalty.ln() + (1.0 / t).ln_1p()
    };
    ModePoint {
        determinant_first: -kernels.c,
        determinant_second: kernels.w,
        normalized_log_h,
        fitted: kernels.c / gram,
        first: kernels.w / gram,
        second: kernels.k / gram,
    }
}

/// The four normalized mode kernels at one `t = lambda*s/g`.
///
/// `c` is the complement `1 - u`, returned in its own right rather than
/// recovered by subtraction: past `t ~ 1/eps` the double `u` IS exactly one and
/// `1 - u` carries no digits of `c` at all, while `c = 1/(1+t)` keeps its own
/// relative accuracy across the whole domain.  Every kernel that a difference
/// would otherwise destroy — `w = u*c` and `k = w*(c-u)` — is built from `c`.
/// `u` itself is deliberately NOT a field: nothing outside this function needs
/// it, and handing it out is how the subtraction gets reintroduced.
#[derive(Clone, Copy)]
struct KernelPoint {
    /// `c = 1/(1+t)`.
    c: f64,
    /// `w = u*c = t/(1+t)^2`.
    w: f64,
    /// `k = w*(c-u) = t(1-t)/(1+t)^3`.
    k: f64,
}

fn kernel_point(t: f64) -> KernelPoint {
    if t.is_infinite() {
        // The `t -> infinity` limits, exactly. Reached when `lambda*s/g`
        // overflows, which the score's own limit does not care about.
        return KernelPoint {
            c: 0.0,
            w: 0.0,
            k: 0.0,
        };
    }
    let (u, c) = if t <= 1.0 {
        let c = 1.0 / (1.0 + t);
        (t * c, c)
    } else {
        // Mirror image: `1/t` is the quantity below one here, so the SAME
        // three roundings deliver `c` to its own relative accuracy.
        let inverse = 1.0 / t;
        let denominator = 1.0 + inverse;
        (1.0 / denominator, inverse / denominator)
    };
    let w = u * c;
    KernelPoint {
        c,
        w,
        k: w * (c - u),
    }
}

/// One mode's contribution to the score jet over a `lambda` cell, per unit
/// `projected_square`.  Field for field the interval image of [`ModePoint`],
/// each field carrying the bound on that point evaluation's own forward error.
///
/// The rounding counts below are read off [`mode_point`] and [`kernel_point`]
/// directly.  `lambda` itself is NOT charged: the cell's `lambda` interval is
/// built by rounding `exp` outward at both ends, so it already contains the
/// point evaluator's `log_lambda.exp()`.
#[derive(Clone, Copy)]
struct ModeRanges {
    determinant_first: RoundedRange,
    determinant_second: RoundedRange,
    fitted: RoundedRange,
    first: RoundedRange,
    second: RoundedRange,
}

fn mode_ranges(gram: f64, penalty: f64, lambda: ClosedInterval) -> ModeRanges {
    let zero = RoundedRange::exact(ClosedInterval::point(0.0));
    if penalty == 0.0 {
        return ModeRanges {
            determinant_first: zero,
            determinant_second: zero,
            // `1.0 / gram`.
            fitted: RoundedRange::rounded(
                ClosedInterval::point(1.0)
                    .div_positive(ClosedInterval::point(gram))
                    .nonnegative(),
                1,
            ),
            first: zero,
            second: zero,
        };
    }
    if gram == 0.0 {
        let h = lambda.mul(ClosedInterval::point(penalty)).nonnegative();
        // `lambda*s` can underflow to zero at the bottom of the domain even
        // though `1/h` is a perfectly good real number there.  A reciprocal
        // that cannot be bounded is reported as the entire line: useless to
        // the search, which will subdivide, but never a false enclosure and
        // never a panic inside `div_positive`.
        let fitted = if h.lo > 0.0 {
            // `1.0 / (lambda * penalty)`.
            RoundedRange::rounded(ClosedInterval::point(1.0).div_positive(h).nonnegative(), 2)
        } else {
            RoundedRange::new(ClosedInterval::new(0.0, f64::INFINITY), f64::INFINITY)
        };
        return ModeRanges {
            determinant_first: zero,
            determinant_second: zero,
            fitted,
            first: fitted,
            second: RoundedRange::new(fitted.range.neg(), fitted.error),
        };
    }

    // Normalize by g: h = g(1+t), t = lambda*s/g.  The four kernels below
    // have known global critical points, so endpoint evaluation plus any
    // critical point contained by the t-window gives an exact real range;
    // interval arithmetic rounds every primitive outward.  The association is
    // `lambda*(s/g)`, matching [`mode_point`], because `lambda*s` alone
    // overflows at the top of the domain for a well-scaled `s/g`.
    let ratio = ClosedInterval::point(penalty).div_positive(ClosedInterval::point(gram));
    let t = lambda.mul(ratio).nonnegative();
    let inverse_gram = ClosedInterval::point(1.0)
        .div_positive(ClosedInterval::point(gram))
        .nonnegative();
    let kernels = kernel_ranges(t);
    // `mode_point` reaches `c` and `u` in five roundings (ratio, t, and three
    // in the branch) and `w = u*c` in six.  Negation and the `1/gram` scaling
    // add one each.
    let first = inverse_gram.mul(kernels.w).nonnegative();
    ModeRanges {
        determinant_first: RoundedRange::rounded(kernels.c.neg(), 5),
        determinant_second: RoundedRange::rounded(kernels.w, 6),
        fitted: RoundedRange::rounded(inverse_gram.mul(kernels.c).nonnegative(), 6),
        first: RoundedRange::rounded(first, 7),
        // `k = w*(c-u)` is the ONE kernel with a cancelling subtraction: `c-u`
        // vanishes at `t = 1` while its own error does not, so `k` has no
        // bounded relative error and charging `|k|` would understate it by
        // `1/|c-u|` — the trap #2513 records, where the first attempt was off
        // by a factor of 1e16 at the far end of the domain.
        //
        // The pre-cancellation magnitude is `|w|`.  Chaining absolute bounds
        // with `|c| , |u| , |c-u| <= 1`: `c-u` carries `2*gamma_5 + u <=
        // gamma_12` absolute; `w*(c-u)` then carries
        // `|w|*(gamma_12 + gamma_6) + u|w| <= |w|*gamma_20`; and the `1/gram`
        // scaling makes it `|w/gram|*gamma_21 = |first|*gamma_21`.
        second: RoundedRange::new(inverse_gram.mul(kernels.k), gamma(21) * magnitude_of(first)),
    }
}

#[derive(Clone, Copy)]
struct KernelRanges {
    /// `t/(1+t)`.
    u: ClosedInterval,
    /// `1/(1+t)`.
    c: ClosedInterval,
    /// `t/(1+t)^2`.
    w: ClosedInterval,
    /// `t(1-t)/(1+t)^3`.
    k: ClosedInterval,
}

/// [`kernel_point`] on a POINT interval, primitive for primitive.  Called only
/// with degenerate or ulp-wide arguments, so no dependency is lost to the
/// `c - u` difference.
fn kernel_at(t: ClosedInterval) -> KernelRanges {
    if t.lo.is_infinite() {
        return KernelRanges {
            u: ClosedInterval::point(1.0),
            c: ClosedInterval::point(0.0),
            w: ClosedInterval::point(0.0),
            k: ClosedInterval::point(0.0),
        };
    }
    let one = ClosedInterval::point(1.0);
    let denom = one.add(t);
    let c = one.div_positive(denom).nonnegative();
    let u = t.mul(c).nonnegative();
    let w = u.mul(c).nonnegative();
    let k = w.mul(c.sub(u));
    KernelRanges { u, c, w, k }
}

fn kernel_ranges(t: ClosedInterval) -> KernelRanges {
    let left = kernel_at(ClosedInterval::point(t.lo));
    let right = kernel_at(ClosedInterval::point(t.hi));
    let mut u = ClosedInterval::new(left.u.lo, right.u.hi).nonnegative();
    let mut c = ClosedInterval::new(right.c.lo, left.c.hi).nonnegative();
    let mut w = left.w.hull(right.w).nonnegative();
    let mut k = left.k.hull(right.k);

    if t.contains(1.0) {
        let critical = kernel_at(ClosedInterval::point(1.0));
        w = w.hull(critical.w).nonnegative();
    }

    // k'(t) has its only positive roots at 2 +/- sqrt(3).  Enclose sqrt(3)
    // itself before subtraction/addition so the exact irrational critical
    // points are not lost to nearest-rounded scalar arithmetic.
    let sqrt_three = ClosedInterval::new(next_down(3.0_f64.sqrt()), next_up(3.0_f64.sqrt()));
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
    u.lo = u.lo.max(0.0);
    u.hi = u.hi.min(next_up(1.0));
    c.lo = c.lo.max(0.0);
    c.hi = c.hi.min(next_up(1.0));
    KernelRanges { u, c, w, k }
}

/// Directed outward rounding that does NOT widen an operation IEEE-754
/// performs exactly.
///
/// Structural zeros are the case that matters.  A kernel that is identically
/// zero on a cell, or the `0.0` an accumulator starts at, produced a
/// `-4.9e-324` lower bound once it met a `next_down` — enough to turn a
/// provably one-signed derivative enclosure into one that straddles zero, and
/// so to lose the cheapest fact the search has: that a monotone cell attains
/// its maximum at an endpoint.  `x + 0` and `x * 0` are exact for every finite
/// `x`, as is scaling by a unit, and rounding them outward claims an
/// uncertainty that is not there.
#[inline]
fn sum_down(a: f64, b: f64) -> f64 {
    let sum = a + b;
    if a == 0.0 || b == 0.0 {
        sum
    } else {
        next_down(sum)
    }
}

#[inline]
fn sum_up(a: f64, b: f64) -> f64 {
    let sum = a + b;
    if a == 0.0 || b == 0.0 {
        sum
    } else {
        next_up(sum)
    }
}

#[inline]
fn product_is_exact(a: f64, b: f64) -> bool {
    a == 0.0 || b == 0.0 || a.abs() == 1.0 || b.abs() == 1.0
}

#[inline]
fn quotient_down(a: f64, b: f64) -> f64 {
    let quotient = a / b;
    if a == 0.0 || b.abs() == 1.0 {
        quotient
    } else {
        next_down(quotient)
    }
}

#[inline]
fn quotient_up(a: f64, b: f64) -> f64 {
    let quotient = a / b;
    if a == 0.0 || b.abs() == 1.0 {
        quotient
    } else {
        next_up(quotient)
    }
}

#[inline]
fn product_down(a: f64, b: f64) -> f64 {
    let product = a * b;
    if product_is_exact(a, b) {
        product
    } else {
        next_down(product)
    }
}

#[inline]
fn product_up(a: f64, b: f64) -> f64 {
    let product = a * b;
    if product_is_exact(a, b) {
        product
    } else {
        next_up(product)
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
        DerivativeEnclosure {
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

    /// Outward rounding is for operations the hardware cannot do exactly.
    /// Applying it to ones it can costs the search real facts — most
    /// importantly the one-signed derivative enclosure of a cell where a
    /// kernel vanishes identically, which is the cheapest close it has.
    #[test]
    fn interval_primitives_do_not_widen_exact_operations() {
        let zero = ClosedInterval::point(0.0);
        assert_eq!(
            zero.add(ClosedInterval::point(5.0)),
            ClosedInterval::point(5.0)
        );
        assert_eq!(
            zero.sub(ClosedInterval::point(5.0)),
            ClosedInterval::point(-5.0)
        );
        assert_eq!(ClosedInterval::point(3.0).mul(zero), zero);
        assert_eq!(
            ClosedInterval::point(7.0).scale(1.0),
            ClosedInterval::point(7.0)
        );
        assert_eq!(
            ClosedInterval::point(7.0).div_positive(ClosedInterval::point(1.0)),
            ClosedInterval::point(7.0)
        );
        assert_eq!(
            ClosedInterval::new(-2.0, 3.0).neg(),
            ClosedInterval::new(-3.0, 2.0)
        );
        // The #2513 case: `3*x^2` over a cell straddling the origin is
        // nonnegative, and the enclosure has to say so.
        let derivative = ClosedInterval::new(-1.0, 1.0).square().scale(3.0);
        assert_eq!(derivative.lo, 0.0);
        assert!(derivative.contains_zero());
        // ... while a genuinely inexact primitive is still rounded outward.
        let inexact = ClosedInterval::point(0.1).add(ClosedInterval::point(0.2));
        assert!(inexact.lo < inexact.hi);
        assert!(inexact.contains(0.1 + 0.2));
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
        DerivativeEnclosure {
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
            |_, _| -> Result<_, String> {
                Ok(DerivativeEnclosure {
                    derivative: ClosedInterval::point(0.3),
                    curvature: ClosedInterval::point(0.0),
                })
            },
        )
        .expect("certified search");
        assert_eq!(result.location, ScoreOptimumLocation::UpperBoundary);
        assert_eq!(result.optimum.x, 9.0);
        assert!(result.stationary_points.is_empty());
    }

    /// `x^3` has a tangential stationary point at the origin that no
    /// derivative/curvature enclosure can isolate.  It is also irrelevant to
    /// the MAXIMUM: the derivative enclosure is one-signed, so the cell is
    /// monotone and its maximum is an endpoint.  The search must say so
    /// exactly — with an empty stationary list, because nothing was certified
    /// stationary — rather than refuse.
    #[test]
    fn tangential_stationary_point_is_not_certified_and_does_not_block_the_maximum() {
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
                    derivative: x.square().scale(3.0),
                    curvature: x.scale(6.0),
                })
            },
        )
        .expect("a monotone cell needs no structural bound at all");
        assert_eq!(result.location, ScoreOptimumLocation::UpperBoundary);
        assert_eq!(result.optimum.x, 1.0);
        assert!(result.stationary_points.is_empty());
        assert_eq!(result.bounded_cells.len(), 1);
        assert_eq!(result.bounded_cells[0].cell, ClosedInterval::new(-1.0, 1.0));
        assert_eq!(result.value_uncertainty, 0.0);
    }

    /// A degenerate quartic peak off the bisection lattice: `V'` and `V''`
    /// vanish together at the maximum, so no curvature enclosure ever excludes
    /// zero there.  Whether that is resolvable is a VALUE question, and the two
    /// arms below differ only in the requested resolution.
    fn quartic_peak_jet(x: f64) -> ScoreJet {
        let d = x - 0.3;
        ScoreJet {
            value: -d * d * d * d,
            derivative: -4.0 * d * d * d,
            curvature: -12.0 * d * d,
            third: -24.0 * d,
        }
    }

    fn quartic_peak_enclosure(lo: f64, hi: f64) -> DerivativeEnclosure {
        let d = ClosedInterval::new(lo, hi).sub(ClosedInterval::point(0.3));
        DerivativeEnclosure {
            derivative: d.mul(d).mul(d).scale(-4.0),
            curvature: d.square().scale(-12.0),
        }
    }

    #[test]
    fn degenerate_peak_is_refused_while_its_cell_still_carries_value() {
        let error = maximize_score_1d(
            -1.0,
            1.0,
            1.0e-3,
            |x| -> Result<_, String> { Ok(quartic_peak_jet(x)) },
            |lo, hi| -> Result<_, String> { Ok(quartic_peak_enclosure(lo.x, hi.x)) },
        )
        .expect_err("at 1e-3 the peak cell still varies far above its own ulp");
        let ScoreSearchError::Unresolved {
            value_excess,
            value_floor,
            ..
        } = error
        else {
            panic!("expected an unresolved cell, got {error}");
        };
        assert!(
            value_excess > value_floor,
            "excess {value_excess} must be the reason, against floor {value_floor}"
        );
    }

    #[test]
    fn degenerate_peak_is_value_certified_once_its_cell_is_flat_to_roundoff() {
        let result = maximize_score_1d(
            -1.0,
            1.0,
            1.0e-8,
            |x| -> Result<_, String> { Ok(quartic_peak_jet(x)) },
            |lo, hi| -> Result<_, String> { Ok(quartic_peak_enclosure(lo.x, hi.x)) },
        )
        .expect("a cell flat below its own ulp is resolved, not refused");
        assert!(
            matches!(result.location, ScoreOptimumLocation::BoundedCell(_)),
            "location was {:?}",
            result.location
        );
        assert!((result.optimum.x - 0.3).abs() <= 1.0e-8);
        // The true maximum is exactly zero, and the search says so to within
        // the uncertainty it reports rather than to an assumed tolerance.
        assert!(result.optimum.value <= 0.0);
        assert!(-result.optimum.value <= result.value_uncertainty);
        assert!(result.value_uncertainty <= 1.0e-14);
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
        for x in [-2.5_f64, -1.7, -0.3, 0.0, 0.9, 1.75] {
            let jet = profile.evaluate(x).unwrap();
            assert!(
                enclosure.derivative.contains(jet.derivative),
                "gradient {} at {x} outside {:?}",
                jet.derivative,
                enclosure.derivative
            );
            assert!(
                enclosure.curvature.contains(jet.curvature),
                "curvature {} at {x} outside {:?}",
                jet.curvature,
                enclosure.curvature
            );
        }
    }

    /// The fixture behind #2513, in the shape `ridge_reml_select_weight`
    /// builds: one spectral direction normalized to `gamma = 1`, duplicated
    /// once per response, one pooled residual.
    fn tail_fixture() -> AffineRemlProfile<'static> {
        const G: &[f64] = &[1.0, 1.0, 1.0];
        const S: &[f64] = &[1.0, 1.0, 1.0];
        const Q: &[f64] = &[4.0 / 3.0, 4.0 / 3.0, 4.0 / 3.0];
        const Y2: &[f64] = &[10.0];
        AffineRemlProfile::new(G, S, Q, Y2, 15.0, 3, 0.0).expect("valid fixture")
    }

    /// Closed form for [`tail_fixture`], written out by hand in the
    /// complement variable `c = 1/(1+lambda)` so the REFERENCE never
    /// differences two numbers that round to the same double.
    fn tail_reference(log_lambda: f64) -> (f64, f64, f64) {
        let lambda = log_lambda.exp();
        let c = 1.0 / (1.0 + lambda);
        let u = lambda / (1.0 + lambda);
        let normalized_logdet = 3.0 * (1.0 / lambda).ln_1p();
        let residual = 10.0 - 4.0 * c;
        let first = 4.0 * u * c;
        let second = 4.0 * u * c * (c - u);
        let log_derivative = first / residual;
        (
            -0.5 * (normalized_logdet + 15.0 * (residual / 15.0).ln()),
            -0.5 * (-3.0 * c + 15.0 * log_derivative),
            -0.5 * (3.0 * u * c + 15.0 * (second / residual - log_derivative * log_derivative)),
        )
    }

    /// #2513: the shipped jet formed `sum_i log h_i - rank*log lambda` and
    /// `sum_i u_i - rank`, both of which difference numbers of size
    /// `rank*|log lambda|` and `rank` to recover results of size `e^-rho`.
    /// Past `rho ~ 36` that left the derivative and the curvature quantized to
    /// multiples of `rank*eps` — 100% error on their own scale — which is what
    /// made an exact-range enclosure unable to contain them.
    #[test]
    fn affine_reml_jet_keeps_relative_accuracy_across_the_whole_log_lambda_domain() {
        let profile = tail_fixture();
        for rho in [
            f64::MIN_POSITIVE.ln(),
            -300.0,
            -1.0,
            0.0,
            1.0,
            30.0,
            37.0,
            40.0,
            80.0,
            300.0,
            700.0,
            (f64::MAX / 2.0).ln(),
        ] {
            let jet = profile.evaluate(rho).expect("finite jet");
            let (value, derivative, curvature) = tail_reference(rho);
            // Eight roundings is the whole budget of either evaluation path;
            // the shipped form was off by a FACTOR at the tail rungs.
            let tolerance = 8.0 * f64::EPSILON;
            assert!(
                (jet.value - value).abs() <= tolerance * value.abs(),
                "value at rho={rho}: {} vs reference {value}",
                jet.value
            );
            assert!(
                (jet.derivative - derivative).abs() <= tolerance * derivative.abs(),
                "derivative at rho={rho}: {} vs reference {derivative} (relative {})",
                jet.derivative,
                (jet.derivative - derivative).abs() / derivative.abs()
            );
            assert!(
                (jet.curvature - curvature).abs() <= tolerance * curvature.abs(),
                "curvature at rho={rho}: {} vs reference {curvature} (relative {})",
                jet.curvature,
                (jet.curvature - curvature).abs() / curvature.abs()
            );
        }
    }

    /// The four cells `ridge_reml_select_weight` actually refused on, by
    /// abscissa. An exact-range enclosure has to contain the range of the
    /// exact score derivatives; it is only obliged to contain the POINT jet
    /// once the point jet is accurate on its own scale.
    #[test]
    fn affine_reml_enclosure_contains_the_tail_jets_it_certifies_against() {
        let profile = tail_fixture();
        for (lo, hi) in [
            (37.029560487247664_f64, 37.72169231549234_f64),
            (36.337428659002995, 37.72169231549234),
            (33.45338621883019, 33.45338622914376),
            (700.0, (f64::MAX / 2.0).ln()),
            (f64::MIN_POSITIVE.ln(), -700.0),
        ] {
            let enclosure = profile.enclose(lo, hi).expect("enclosure");
            for x in [lo, hi] {
                let jet = profile.evaluate(x).expect("jet");
                assert!(
                    enclosure.derivative.contains(jet.derivative),
                    "derivative {} at {x} outside {:?} on [{lo}, {hi}]",
                    jet.derivative,
                    enclosure.derivative
                );
                assert!(
                    enclosure.curvature.contains(jet.curvature),
                    "curvature {} at {x} outside {:?} on [{lo}, {hi}]",
                    jet.curvature,
                    enclosure.curvature
                );
            }
        }
    }

    /// #2513: the refinement's own iterates locate the root far below the
    /// requested bracket resolution, and the reported sample must be one of
    /// them rather than a fresh midpoint of the final bracket.  The bracket
    /// stays the certificate and is still honoured at the requested width.
    #[test]
    fn refined_root_is_reported_far_inside_its_own_bracket() {
        let profile = tail_fixture();
        let resolution = f64::EPSILON.sqrt();
        let search = profile
            .maximize(f64::MIN_POSITIVE.ln(), (f64::MAX / 2.0).ln(), resolution)
            .expect("certified search");
        assert_eq!(search.location, ScoreOptimumLocation::Stationary(0));
        assert_eq!(search.stationary_points.len(), 1);
        let stationary = search.stationary_points[0];
        // gamma_max = 2 for the fixture this profile was normalized from, and
        // the one-direction stationarity equation gives lambda = 1.2 exactly.
        let lambda = 2.0 * stationary.sample.x.exp();
        assert!(
            (lambda - 1.2).abs() <= 1.0e-13,
            "lambda={lambda}; the midpoint-of-bracket report was 1.2000000050"
        );
        assert!(stationary.bracket.hi - stationary.bracket.lo <= resolution);
        assert!(stationary.bracket.contains(stationary.sample.x));
    }

    /// The containment the search demands at zero tolerance is a PROPERTY of
    /// this oracle pair, not a coincidence of the fixtures that happened to be
    /// tried.  A deterministic sweep over ill-conditioned spectra, several
    /// responses, and cells from a single point up to the width of the whole
    /// representable log-lambda domain pins it as one.
    #[test]
    fn affine_reml_enclosure_contains_every_point_jet_it_is_checked_against() {
        // A deterministic LCG: the point of the sweep is coverage, not chance,
        // and a random seed would make a failure unreproducible.
        let mut state = 0x2513_2513_2513_2513_u64;
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 11) as f64) / ((1u64 << 53) as f64)
        };
        let domain_lo = f64::MIN_POSITIVE.ln();
        let domain_hi = (f64::MAX / 2.0).ln();
        let mut checked = 0usize;
        for case in 0..64 {
            let modes = 1 + case % 5;
            let responses = 1 + case % 3;
            // Spectra spanning up to 12 decades, and every case gives one mode
            // an exact zero gram or penalty so both degenerate branches run.
            let mut gram = Vec::new();
            let mut penalty = Vec::new();
            for i in 0..modes {
                let g = 10.0_f64.powf(12.0 * next() - 6.0);
                let s = 10.0_f64.powf(12.0 * next() - 6.0);
                if i == 0 && case % 3 == 1 {
                    gram.push(0.0);
                    penalty.push(s);
                } else if i == 0 && case % 3 == 2 {
                    gram.push(g);
                    penalty.push(0.0);
                } else {
                    gram.push(g);
                    penalty.push(s);
                }
            }
            let rank = penalty.iter().filter(|&&value| value > 0.0).count();
            let projected: Vec<f64> = (0..modes * responses).map(|_| next()).collect();
            // Keep the profiled residual comfortably positive: the enclosure
            // rejects a cell it cannot certify positive, which is a different
            // (and already tested) verdict.
            let energy: Vec<f64> = (0..responses)
                .map(|output| {
                    let fitted: f64 = (0..modes)
                        .map(|i| projected[output * modes + i] / gram[i].max(penalty[i]).min(1.0))
                        .sum();
                    fitted * (2.0 + next()) + 1.0
                })
                .collect();
            let Ok(profile) = AffineRemlProfile::new(
                &gram,
                &penalty,
                &projected,
                &energy,
                7.0 + case as f64,
                rank,
                0.25,
            ) else {
                continue;
            };
            for cell in 0..12 {
                let width = (domain_hi - domain_lo) * 0.5_f64.powi(cell);
                let lo = domain_lo + (domain_hi - domain_lo - width) * next();
                let hi = (lo + width).min(domain_hi);
                let Ok(enclosure) = profile.enclose(lo, hi) else {
                    continue;
                };
                for at in [lo, hi, 0.5 * (lo + hi)] {
                    let Ok(jet) = profile.evaluate(at) else {
                        continue;
                    };
                    checked += 1;
                    assert!(
                        enclosure.derivative.contains(jet.derivative),
                        "case {case}: derivative {} at {at} outside {:?} on [{lo}, {hi}]",
                        jet.derivative,
                        enclosure.derivative
                    );
                    assert!(
                        enclosure.curvature.contains(jet.curvature),
                        "case {case}: curvature {} at {at} outside {:?} on [{lo}, {hi}]",
                        jet.curvature,
                        enclosure.curvature
                    );
                }
            }
        }
        assert!(checked > 1000, "sweep only checked {checked} jets");
    }

    /// The widening is not decorative: on a degenerate cell the exact range of
    /// each derivative is a single real number, so the whole width of the
    /// returned enclosure IS the evaluator's forward-error bound.
    #[test]
    fn affine_reml_enclosure_carries_a_real_evaluator_pad() {
        let profile = tail_fixture();
        for rho in [-2.0_f64, 0.0, 5.0, 40.0] {
            let enclosure = profile.enclose(rho, rho).expect("degenerate cell");
            let jet = profile.evaluate(rho).expect("jet");
            let width = enclosure.derivative.hi - enclosure.derivative.lo;
            assert!(width > 0.0, "no pad at rho={rho}");
            // The pad tracks the quantity it brackets rather than sitting at an
            // absolute floor: that is what keeps the far tail certifiable.
            assert!(
                width <= 1.0e-12 * (1.0 + jet.derivative.abs()),
                "pad {width} at rho={rho} is not proportional to |derivative| {}",
                jet.derivative.abs()
            );
            assert!(enclosure.derivative.contains(jet.derivative));
            assert!(enclosure.curvature.contains(jet.curvature));
        }
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
