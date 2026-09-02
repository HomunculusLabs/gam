//! Event-history data: subjects, their marked events, their covariate
//! segments, and the node expansion that turns continuous follow-up into the
//! quadrature/event nodes the exact-marginal likelihood is evaluated on.
//!
//! The compensator `∫ λ_d(t) dt` of every mark is integrated by Gauss-Legendre
//! quadrature on every segment between consecutive breakpoints (entry, exit,
//! covariate changes, events). Every quadrature node carries an exposure
//! weight; every event carries a unit count and no exposure. The latent state
//! of the subject is represented at every node, so within-segment latent
//! variation is resolved at the same resolution as the compensator.

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

/// One observed event: its time and its mark index.
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
    /// End of observation; an absorbing event at `exit` ends follow-up.
    pub exit: f64,
    /// Events in `(entry, exit]`, any order (sorted during validation).
    pub events: Vec<Event>,
    /// Covariate segments; the first must start at or before `entry`.
    pub segments: Vec<CovariateSegment>,
}

/// A cohort of subjects with a shared covariate table and mark vocabulary.
#[derive(Clone, Debug)]
pub struct EventHistoryCohort {
    pub mark_names: Vec<String>,
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
            for event in &subject.events {
                if !event.time.is_finite() || event.time <= subject.entry || event.time > subject.exit {
                    return Err(invalid(format!(
                        "subject {:?} has an event at {} outside (entry, exit] = ({}, {}]",
                        subject.id, event.time, subject.entry, subject.exit
                    )));
                }
                if event.mark >= marks {
                    return Err(invalid(format!(
                        "subject {:?} has an event with mark {} but only {} marks are declared",
                        subject.id, event.mark, marks
                    )));
                }
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
    /// Exposure (quadrature) weight of each node; zero on pure event nodes.
    pub exposures: Vec<f64>,
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

/// Gauss-Legendre order per segment for a time smooth of this B-spline
/// degree: the order that integrates the basis products exactly, matching
/// the convention the smooth penalties use.
pub fn quadrature_order_for_degree(degree: usize) -> usize {
    2 * degree + 3
}

struct RawNode {
    time: f64,
    exposure: f64,
    mark: Option<usize>,
}

/// Expand every subject into quadrature and event nodes.
pub fn expand_nodes(
    cohort: &EventHistoryCohort,
    quadrature_order: usize,
) -> Result<CohortNodes, EventHistoryError> {
    if quadrature_order == 0 {
        return Err(invalid("quadrature order must be positive"));
    }
    let marks = cohort.marks();
    let (gl_nodes, gl_weights) = gam_math::special::gauss_legendre(quadrature_order);
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
        breakpoints.extend(subject.events.iter().map(|e| e.time));
        breakpoints.sort_by(|a, b| a.total_cmp(b));
        breakpoints.dedup();
        let mut raw: Vec<RawNode> = Vec::new();
        for pair in breakpoints.windows(2) {
            let (left, right) = (pair[0], pair[1]);
            let half = 0.5 * (right - left);
            let mid = 0.5 * (right + left);
            for (x, w) in gl_nodes.iter().zip(gl_weights.iter()) {
                raw.push(RawNode {
                    time: mid + half * x,
                    exposure: half * w,
                    mark: None,
                });
            }
        }
        for event in &subject.events {
            raw.push(RawNode {
                time: event.time,
                exposure: 0.0,
                mark: Some(event.mark),
            });
        }
        raw.sort_by(|a, b| a.time.total_cmp(&b.time));
        let mut times: Vec<f64> = Vec::new();
        let mut exposures: Vec<f64> = Vec::new();
        let mut counts: Vec<Vec<f64>> = Vec::new();
        for node in raw {
            if let Some(&last) = times.last()
                && node.time == last
            {
                let index = times.len() - 1;
                exposures[index] += node.exposure;
                if let Some(mark) = node.mark {
                    counts[index][mark] += 1.0;
                }
            } else {
                times.push(node.time);
                exposures.push(node.exposure);
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
        let mut covariate_rows = Vec::with_capacity(times.len());
        for &t in &times {
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
        }
        let n = times.len();
        let mut count_matrix = Array2::<f64>::zeros((n, marks));
        for (i, row) in counts.iter().enumerate() {
            for (d, &c) in row.iter().enumerate() {
                count_matrix[[i, d]] = c;
            }
        }
        subjects.push(SubjectNodes {
            first_row,
            times,
            gaps,
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
