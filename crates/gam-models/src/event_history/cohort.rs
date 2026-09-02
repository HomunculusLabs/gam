//! Event-history data: subjects, their marked events, their covariate
//! segments, the kind of every mark, and the node expansion that turns
//! continuous follow-up into the quadrature/event nodes the likelihood is
//! evaluated on.
//!
//! The compensator `∫ R_d(t) λ_d(t) dt` of every mark is integrated by
//! Gauss-Lobatto quadrature on every segment between consecutive breakpoints
//! (entry, exit, covariate changes, events). `R_d` is the risk-set indicator
//! of the mark: one for a recurrent mark, one until the first occurrence for
//! a mark that can happen once, zero for a mark the subject had before
//! entry. It is constant on the open segment, so each segment's nodes carry
//! that segment's risk, and a node shared by two segments adds the two.
//!
//! The rule is closed — its endpoints are nodes — and that is the point. An
//! event time is a breakpoint, so with an open rule the instant at which the
//! intensity is read carries no exposure at all, and the latent state there
//! is held only by its neighbours. The subject's latent path is represented
//! at the nodes, so once the process decorrelates across a mesh cell that
//! event node's state is free, `y a z − ½z²` is maximised at `z = a`, and
//! the likelihood gains `a²/2` per event without bound — a divergence the
//! continuous-time model does not have (the exponential of white noise is
//! not a random measure) and the mesh invents. A closed rule gives the event
//! node its own exposure, so its own compensator holds its state down, and
//! the discretisation error is `O(rate · gap)` in every term.

use ndarray::Array2;
use thiserror::Error;

/// Errors raised by the event-history family.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum EventHistoryError {
    #[error("{reason}")]
    InvalidInput { reason: String },
    #[error("{reason}")]
    NumericalFailure { reason: String },
    #[error("{reason}")]
    Fit { reason: String },
}

impl From<EventHistoryError> for String {
    fn from(error: EventHistoryError) -> Self {
        error.to_string()
    }
}

fn invalid(reason: impl Into<String>) -> EventHistoryError {
    EventHistoryError::InvalidInput {
        reason: reason.into(),
    }
}

/// What an occurrence of a mark does to the subject's risk sets.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MarkKind {
    /// May recur; the subject stays at risk for it after every occurrence.
    Recurrent,
    /// Happens at most once: the subject leaves this mark's risk set at its
    /// first occurrence and stays at risk for every other mark (a first
    /// diagnosis).
    Once,
    /// Ends follow-up for every mark (death).
    Terminal,
}

impl MarkKind {
    pub fn as_str(self) -> &'static str {
        match self {
            MarkKind::Recurrent => "recurrent",
            MarkKind::Once => "once",
            MarkKind::Terminal => "terminal",
        }
    }
}

/// One observed event: its time and its mark index. An event at or before
/// the subject's entry is history: it is not a likelihood contribution, but
/// it removes the subject from the risk set of a mark that happens once.
#[derive(Clone, Debug, PartialEq)]
pub struct Event {
    pub time: f64,
    pub mark: usize,
}

/// A piece of the follow-up during which the subject's covariate row is
/// constant: from `start` until the next segment's `start` (or exit).
#[derive(Clone, Debug, PartialEq)]
pub struct CovariateSegment {
    pub start: f64,
    /// Row of [`EventHistoryCohort::covariates`] in force on this segment.
    pub row: usize,
}

/// One subject's observed history.
#[derive(Clone, Debug, PartialEq)]
pub struct SubjectHistory {
    pub id: String,
    /// Start of observation (delayed entry allowed).
    pub entry: f64,
    /// End of observation; a terminal event happens at `exit`.
    pub exit: f64,
    /// Events, any order (sorted during validation). Events at or before
    /// `entry` are prior history.
    pub events: Vec<Event>,
    /// Covariate segments; the first must start at or before `entry`.
    pub segments: Vec<CovariateSegment>,
}

impl SubjectHistory {
    /// Marks the subject had at or before entry.
    pub fn prior_marks(&self) -> impl Iterator<Item = usize> + '_ {
        self.events
            .iter()
            .filter(move |e| e.time <= self.entry)
            .map(|e| e.mark)
    }

    /// Events inside the observation window `(entry, exit]`.
    pub fn observed_events(&self) -> impl Iterator<Item = &Event> + '_ {
        self.events.iter().filter(move |e| e.time > self.entry)
    }
}

/// A cohort of subjects with a shared covariate table and mark vocabulary.
#[derive(Clone, Debug)]
pub struct EventHistoryCohort {
    pub mark_names: Vec<String>,
    /// The kind of every mark, parallel to `mark_names`.
    pub mark_kinds: Vec<MarkKind>,
    /// Names of the covariate table's columns, visible to the formula.
    pub covariate_names: Vec<String>,
    /// Covariate table; rows are referenced by [`CovariateSegment::row`].
    pub covariates: Array2<f64>,
    pub subjects: Vec<SubjectHistory>,
}

impl EventHistoryCohort {
    pub fn marks(&self) -> usize {
        self.mark_names.len()
    }

    /// Validate every subject and sort its events and segments in time.
    pub fn validate(&mut self) -> Result<(), EventHistoryError> {
        if self.mark_names.is_empty() {
            return Err(invalid("an event-history cohort needs at least one mark"));
        }
        if self.mark_kinds.len() != self.mark_names.len() {
            return Err(invalid(format!(
                "{} mark kinds for {} marks",
                self.mark_kinds.len(),
                self.mark_names.len()
            )));
        }
        if self.subjects.is_empty() {
            return Err(invalid("an event-history cohort needs at least one subject"));
        }
        if self.covariates.iter().any(|v| !v.is_finite()) {
            return Err(invalid("covariate table contains a non-finite value"));
        }
        if self.covariate_names.len() != self.covariates.ncols() {
            return Err(invalid(format!(
                "{} covariate names for a table with {} columns",
                self.covariate_names.len(),
                self.covariates.ncols()
            )));
        }
        let marks = self.marks();
        let rows = self.covariates.nrows();
        for subject in &mut self.subjects {
            if subject.id.is_empty() {
                return Err(invalid("subject identifier must be non-empty"));
            }
            if !subject.entry.is_finite() || !subject.exit.is_finite() || subject.exit <= subject.entry
            {
                return Err(invalid(format!(
                    "subject {:?} needs finite entry < exit, got {} and {}",
                    subject.id, subject.entry, subject.exit
                )));
            }
            if subject.segments.is_empty() {
                return Err(invalid(format!(
                    "subject {:?} needs at least one covariate segment",
                    subject.id
                )));
            }
            subject
                .segments
                .sort_by(|a, b| a.start.total_cmp(&b.start));
            if !subject.segments[0].start.is_finite() || subject.segments[0].start > subject.entry {
                return Err(invalid(format!(
                    "subject {:?}: the first covariate segment must start at or before entry",
                    subject.id
                )));
            }
            for segment in &subject.segments {
                if !segment.start.is_finite() {
                    return Err(invalid(format!(
                        "subject {:?} has a non-finite segment start",
                        subject.id
                    )));
                }
                if segment.row >= rows {
                    return Err(invalid(format!(
                        "subject {:?} references covariate row {} of a {}-row table",
                        subject.id, segment.row, rows
                    )));
                }
            }
            subject.events.sort_by(|a, b| a.time.total_cmp(&b.time));
            let mut once_seen = vec![false; marks];
            for event in &subject.events {
                if !event.time.is_finite() || event.time > subject.exit {
                    return Err(invalid(format!(
                        "subject {:?} has an event at {} after its exit {}",
                        subject.id, event.time, subject.exit
                    )));
                }
                if event.mark >= marks {
                    return Err(invalid(format!(
                        "subject {:?} has an event with mark {} but only {} marks are declared",
                        subject.id, event.mark, marks
                    )));
                }
                match self.mark_kinds[event.mark] {
                    MarkKind::Recurrent => {}
                    MarkKind::Once => {
                        if once_seen[event.mark] && event.time > subject.entry {
                            return Err(invalid(format!(
                                "subject {:?} has mark {:?} (which happens once) more than once",
                                subject.id, self.mark_names[event.mark]
                            )));
                        }
                        once_seen[event.mark] = true;
                    }
                    MarkKind::Terminal => {
                        if event.time != subject.exit {
                            return Err(invalid(format!(
                                "subject {:?} has terminal mark {:?} at {} but exits at {}",
                                subject.id, self.mark_names[event.mark], event.time, subject.exit
                            )));
                        }
                    }
                }
            }
            let terminal_events = subject
                .events
                .iter()
                .filter(|e| self.mark_kinds[e.mark] == MarkKind::Terminal)
                .count();
            if terminal_events > 1 {
                return Err(invalid(format!(
                    "subject {:?} has {terminal_events} terminal events",
                    subject.id
                )));
            }
        }
        Ok(())
    }

    /// Mean follow-up length across subjects: the cohort's own time scale.
    ///
    /// Latent rates are parameterised as `log(rate · time_scale)`, so a rate
    /// of one is "memory comparable to a follow-up", whatever the time unit.
    pub fn time_scale(&self) -> f64 {
        let total: f64 = self.subjects.iter().map(|s| s.exit - s.entry).sum();
        total / self.subjects.len() as f64
    }

    /// Whether `subject` is at risk for `mark` at `time` given its history:
    /// a once-only mark is off after (or at) its first occurrence.
    pub fn at_risk(&self, subject: &SubjectHistory, mark: usize, time: f64) -> bool {
        match self.mark_kinds[mark] {
            MarkKind::Recurrent | MarkKind::Terminal => true,
            MarkKind::Once => !subject
                .events
                .iter()
                .any(|e| e.mark == mark && e.time <= time),
        }
    }
}

/// The node expansion of one subject.
#[derive(Clone, Debug)]
pub struct SubjectNodes {
    /// Global row offset of this subject's first node.
    pub first_row: usize,
    /// Strictly increasing node times.
    pub times: Vec<f64>,
    /// `times[n+1] - times[n]`, length `times.len() - 1`.
    pub gaps: Vec<f64>,
    /// Quadrature weight of each node: the length of follow-up it stands
    /// for. An event time is a node of the closed rule, so it carries the
    /// weight of the segment that ended there.
    pub weights: Vec<f64>,
    /// Exposure of each node to each mark, `weights[n] · R_d(t_n)`, shape
    /// `(nodes, marks)`.
    pub exposures: Array2<f64>,
    /// Event counts per node and mark, shape `(nodes, marks)`.
    pub counts: Array2<f64>,
    /// Covariate row in force at each node.
    pub covariate_rows: Vec<usize>,
}

impl SubjectNodes {
    pub fn len(&self) -> usize {
        self.times.len()
    }

    pub fn is_empty(&self) -> bool {
        self.times.is_empty()
    }

    /// Whether node `n` carries any likelihood: a count or an exposure.
    pub fn informative(&self, n: usize) -> bool {
        self.counts.row(n).iter().any(|&y| y != 0.0)
            || self.exposures.row(n).iter().any(|&w| w != 0.0)
    }
}

/// Node expansion of a whole cohort plus the per-node data matrix.
#[derive(Clone, Debug)]
pub struct CohortNodes {
    pub marks: usize,
    pub subjects: Vec<SubjectNodes>,
    /// Per-node data: the covariate row in force followed by the node time
    /// in the last column, so a time smooth is a smooth of `time_column`.
    pub node_data: Array2<f64>,
    pub time_column: usize,
    pub total_nodes: usize,
}

/// Gauss-Lobatto order per segment for a time smooth of this B-spline
/// degree: enough nodes that the closed rule (exact to degree `2n − 3`)
/// integrates the basis products exactly, matching the convention the
/// smooth penalties use.
pub fn quadrature_order_for_degree(degree: usize) -> usize {
    2 * degree + 3
}

struct RawNode {
    time: f64,
    /// Quadrature weight this node carries from the segment it came from.
    weight: f64,
    /// Whether the subject is at risk for each mark on that segment.
    at_risk: Vec<bool>,
    mark: Option<usize>,
}

/// Expand every subject into quadrature and event nodes.
pub fn expand_nodes(
    cohort: &EventHistoryCohort,
    quadrature_order: usize,
) -> Result<CohortNodes, EventHistoryError> {
    if quadrature_order < 2 {
        return Err(invalid("a closed quadrature rule needs at least two nodes"));
    }
    let marks = cohort.marks();
    let (rule_nodes, rule_weights) = gam_math::special::gauss_lobatto(quadrature_order);
    let n_cov = cohort.covariates.ncols();
    let mut subjects = Vec::with_capacity(cohort.subjects.len());
    let mut data_rows: Vec<Vec<f64>> = Vec::new();
    let mut first_row = 0usize;
    for subject in &cohort.subjects {
        let mut breakpoints = vec![subject.entry, subject.exit];
        breakpoints.extend(
            subject
                .segments
                .iter()
                .map(|s| s.start)
                .filter(|&t| t > subject.entry && t < subject.exit),
        );
        breakpoints.extend(subject.observed_events().map(|e| e.time));
        breakpoints.sort_by(|a, b| a.total_cmp(b));
        breakpoints.dedup();
        let mut raw: Vec<RawNode> = Vec::new();
        for pair in breakpoints.windows(2) {
            let (left, right) = (pair[0], pair[1]);
            // The risk indicator is constant on the open segment: a mark
            // that happens once is off from its own occurrence onward, so a
            // segment whose left end is that occurrence carries no exposure
            // for it, and the occurrence's own node keeps only the exposure
            // of the segment that ended there.
            let at_risk: Vec<bool> = (0..marks)
                .map(|d| match cohort.mark_kinds[d] {
                    MarkKind::Recurrent | MarkKind::Terminal => true,
                    MarkKind::Once => !subject.events.iter().any(|e| e.mark == d && e.time <= left),
                })
                .collect();
            let half = 0.5 * (right - left);
            let mid = 0.5 * (right + left);
            for (x, w) in rule_nodes.iter().zip(rule_weights.iter()) {
                raw.push(RawNode {
                    time: mid + half * x,
                    weight: half * w,
                    at_risk: at_risk.clone(),
                    mark: None,
                });
            }
        }
        let no_risk = vec![false; marks];
        for event in subject.observed_events() {
            raw.push(RawNode {
                time: event.time,
                weight: 0.0,
                at_risk: no_risk.clone(),
                mark: Some(event.mark),
            });
        }
        raw.sort_by(|a, b| a.time.total_cmp(&b.time));
        // Two nodes closer than the resolution of `f64` on this follow-up
        // are one node: their weights add, their counts add.
        let resolution = 8.0 * f64::EPSILON * (subject.exit.abs().max(subject.entry.abs()) + (subject.exit - subject.entry));
        let mut times: Vec<f64> = Vec::new();
        let mut weights: Vec<f64> = Vec::new();
        let mut exposure_rows: Vec<Vec<f64>> = Vec::new();
        let mut counts: Vec<Vec<f64>> = Vec::new();
        for node in raw {
            if let Some(&last) = times.last()
                && node.time - last <= resolution
            {
                let index = times.len() - 1;
                weights[index] += node.weight;
                for d in 0..marks {
                    if node.at_risk[d] {
                        exposure_rows[index][d] += node.weight;
                    }
                }
                if let Some(mark) = node.mark {
                    counts[index][mark] += 1.0;
                }
            } else {
                times.push(node.time);
                weights.push(node.weight);
                exposure_rows.push(
                    (0..marks)
                        .map(|d| if node.at_risk[d] { node.weight } else { 0.0 })
                        .collect(),
                );
                let mut row = vec![0.0; marks];
                if let Some(mark) = node.mark {
                    row[mark] += 1.0;
                }
                counts.push(row);
            }
        }
        let gaps: Vec<f64> = times.windows(2).map(|w| w[1] - w[0]).collect();
        if gaps.iter().any(|&g| !(g > 0.0)) {
            return Err(EventHistoryError::NumericalFailure {
                reason: format!(
                    "subject {:?}: node times are not strictly increasing after merging",
                    subject.id
                ),
            });
        }
        let n = times.len();
        let mut covariate_rows = Vec::with_capacity(n);
        let mut exposures = Array2::<f64>::zeros((n, marks));
        let mut count_matrix = Array2::<f64>::zeros((n, marks));
        for (i, &t) in times.iter().enumerate() {
            let row = subject
                .segments
                .iter()
                .rev()
                .find(|s| s.start <= t)
                .map(|s| s.row)
                .unwrap_or(subject.segments[0].row);
            covariate_rows.push(row);
            let mut data = Vec::with_capacity(n_cov + 1);
            data.extend(cohort.covariates.row(row).iter().copied());
            data.push(t);
            data_rows.push(data);
            for d in 0..marks {
                count_matrix[[i, d]] = counts[i][d];
                exposures[[i, d]] = exposure_rows[i][d];
            }
        }
        subjects.push(SubjectNodes {
            first_row,
            times,
            gaps,
            weights,
            exposures,
            counts: count_matrix,
            covariate_rows,
        });
        first_row += n;
    }
    let total_nodes = first_row;
    let mut node_data = Array2::<f64>::zeros((total_nodes, n_cov + 1));
    for (i, row) in data_rows.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            node_data[[i, j]] = v;
        }
    }
    Ok(CohortNodes {
        marks,
        subjects,
        node_data,
        time_column: n_cov,
        total_nodes,
    })
}
