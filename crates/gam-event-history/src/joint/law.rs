//! Complete-path probability law of the joint event model (#2961).
//!
//! A history is its node sequence: an unexposed entry anchor, nodes that carry
//! at-risk exposure, and zero-exposure event nodes. Entry is observed context.
//! A once-only mark that fired at or before entry opens the window outside its
//! risk set, and no event-free pre-entry exposure is invented.
//!
//! At rank zero every mark's intensity is a constant rate `r_d` and the
//! reference normaliser is identically one, so the complete-path log density of
//! a history is `sum_d y_d log r_d - E_d r_d`. The event counts `y_d` and the
//! at-risk exposures `E_d` are sufficient statistics.

use crate::{EventHistoryError, MarkKind, SubjectHistory};
use serde::{Deserialize, Serialize};

pub(super) fn invalid(reason: impl Into<String>) -> EventHistoryError {
    EventHistoryError::InvalidInput {
        reason: reason.into(),
    }
}

pub(super) fn numerical(reason: impl Into<String>) -> EventHistoryError {
    EventHistoryError::NumericalFailure {
        reason: reason.into(),
    }
}

/// The declared kind of every mark.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(super) struct JointSpecification {
    pub(super) marks: Vec<MarkKind>,
}

/// One follow-up window as its node sequence.
pub(super) struct JointHistory {
    /// At-risk exposure carried by each node: zero on the entry anchor and on
    /// every event node.
    exposure: Vec<f64>,
    events: Vec<Option<usize>>,
    initially_at_risk: Vec<bool>,
}

impl JointSpecification {
    pub(super) fn new(marks: Vec<MarkKind>) -> Result<Self, EventHistoryError> {
        if marks.is_empty() {
            return Err(invalid("the joint event model needs at least one mark"));
        }
        Ok(Self { marks })
    }

    /// The node sequence of one follow-up record. An event at or before entry
    /// is prior history: a once-only mark leaves the risk set it opens with, a
    /// recurrent one is not compensated, and a terminal one leaves nothing to
    /// model. Simultaneous events are kept, with a terminal event last.
    pub(super) fn history(&self, subject: &SubjectHistory) -> Result<JointHistory, EventHistoryError> {
        let id = &subject.id;
        if !subject.segments.is_empty() {
            return Err(invalid(format!(
                "subject {id:?}: the rank-zero joint model has no covariate terms, so a history takes no covariate segments"
            )));
        }
        if !(subject.entry.is_finite() && subject.exit.is_finite() && subject.entry < subject.exit) {
            return Err(invalid(format!(
                "subject {id:?} needs finite entry < exit, got {} and {}",
                subject.entry, subject.exit
            )));
        }
        let marks = self.marks.len();
        if let Some(event) = subject
            .events
            .iter()
            .find(|e| e.mark >= marks || !e.time.is_finite() || e.time > subject.exit)
        {
            return Err(invalid(format!(
                "subject {id:?} has an event of mark {} at {}; marks are 0..{marks} and no event follows the exit {}",
                event.mark, event.time, subject.exit
            )));
        }
        let terminal = |mark: usize| self.marks[mark] == MarkKind::Terminal;
        let mut events: Vec<_> = subject.events.iter().collect();
        events.sort_by(|a, b| {
            a.time
                .total_cmp(&b.time)
                .then_with(|| terminal(a.mark).cmp(&terminal(b.mark)))
        });
        let mut history = JointHistory {
            exposure: vec![0.0],
            events: vec![None],
            initially_at_risk: vec![true; marks],
        };
        let mut fired = vec![false; marks];
        let mut terminated = false;
        let mut previous = subject.entry;
        for event in events {
            let kind = self.marks[event.mark];
            if terminated {
                return Err(invalid(format!(
                    "subject {id:?} has an event of mark {} after the terminal event that ended its follow-up",
                    event.mark
                )));
            }
            if kind != MarkKind::Recurrent && std::mem::replace(&mut fired[event.mark], true) {
                return Err(invalid(format!(
                    "subject {id:?} has two events of the {} mark {}, which fires at most once",
                    kind.name(),
                    event.mark
                )));
            }
            if event.time <= subject.entry {
                match kind {
                    MarkKind::Once => history.initially_at_risk[event.mark] = false,
                    MarkKind::Terminal => {
                        return Err(invalid(format!(
                            "subject {id:?} has a terminal event at {}, at or before its entry {}; it has no follow-up to model",
                            event.time, subject.entry
                        )));
                    }
                    MarkKind::Recurrent => (),
                }
                continue;
            }
            if kind == MarkKind::Terminal {
                if event.time != subject.exit {
                    return Err(invalid(format!(
                        "subject {id:?}: a terminal event at {} must end follow-up, but the exit is {}",
                        event.time, subject.exit
                    )));
                }
                terminated = true;
            }
            if event.time > previous {
                history.exposure.push(event.time - previous);
                history.events.push(None);
                previous = event.time;
            }
            history.exposure.push(0.0);
            history.events.push(Some(event.mark));
        }
        if subject.exit > previous {
            history.exposure.push(subject.exit - previous);
            history.events.push(None);
        }
        Ok(history)
    }
}

impl JointHistory {
    /// Event counts and at-risk exposures by mark: the sufficient statistics of
    /// the rank-zero complete-path density.
    pub(super) fn rate_statistics(&self, marks: &[MarkKind]) -> (Vec<u64>, Vec<f64>) {
        let mut risk = self.initially_at_risk.clone();
        let mut counts = vec![0u64; marks.len()];
        let mut exposure = vec![0.0; marks.len()];
        for (&weight, event) in self.exposure.iter().zip(&self.events) {
            for d in 0..marks.len() {
                if risk[d] {
                    exposure[d] += weight;
                }
            }
            if let Some(d) = *event {
                counts[d] += 1;
                if marks[d] == MarkKind::Once {
                    risk[d] = false;
                }
            }
        }
        (counts, exposure)
    }

    /// The risk set the window leaves open at its exit, or `None` when a
    /// terminal event ended it.
    pub(super) fn open_risk_set(&self, marks: &[MarkKind]) -> Option<Vec<bool>> {
        let mut risk = self.initially_at_risk.clone();
        for &d in self.events.iter().flatten() {
            match marks[d] {
                MarkKind::Recurrent => (),
                MarkKind::Once => risk[d] = false,
                MarkKind::Terminal => return None,
            }
        }
        Some(risk)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Event;

    fn subject(entry: f64, exit: f64, events: &[(f64, usize)]) -> SubjectHistory {
        SubjectHistory {
            id: "s".to_string(),
            entry,
            exit,
            events: events
                .iter()
                .map(|&(time, mark)| Event { time, mark })
                .collect(),
            segments: Vec::new(),
        }
    }

    #[test]
    fn node_path_density_equals_the_rank_zero_sufficient_statistics() {
        let spec = JointSpecification::new(vec![
            MarkKind::Once,
            MarkKind::Recurrent,
            MarkKind::Once,
            MarkKind::Terminal,
        ])
        .unwrap();
        // Mark 2 is prevalent at entry, recurrent mark 1 fires twice, and the
        // incident once-only mark 0 fires at the instant of the terminal event.
        let history = spec
            .history(&subject(
                1.0,
                7.5,
                &[(7.5, 3), (0.5, 2), (2.0, 1), (3.0, 1), (7.5, 0)],
            ))
            .unwrap();
        let log_rates = [-1.3, 0.4, -0.2, -2.1];
        // Node by node: every exposed node compensates the marks at risk there
        // and every event adds its log rate.
        let mut risk = history.initially_at_risk.clone();
        let mut path = 0.0;
        let mut path_abs = 0.0;
        let mut summands = 0usize;
        for (&weight, event) in history.exposure.iter().zip(&history.events) {
            for d in 0..4 {
                if risk[d] {
                    let compensator = weight * f64::exp(log_rates[d]);
                    path -= compensator;
                    path_abs += compensator;
                    summands += 1;
                }
            }
            if let Some(d) = *event {
                assert!(risk[d], "event of mark {d} outside its risk set");
                path += log_rates[d];
                path_abs += log_rates[d].abs();
                summands += 1;
                if spec.marks[d] != MarkKind::Recurrent {
                    risk[d] = false;
                }
            }
        }
        let (counts, exposure) = history.rate_statistics(&spec.marks);
        assert_eq!(counts, [1, 2, 0, 1]);
        assert_eq!(exposure, [6.5, 6.5, 0.0, 6.5]);
        let terms: Vec<f64> = (0..4)
            .flat_map(|d| [counts[d] as f64 * log_rates[d], -exposure[d] * f64::exp(log_rates[d])])
            .collect();
        let sufficient: f64 = terms.iter().sum();
        // The same summands in two orders: the bar is the summation roundoff floor.
        let s_abs = path_abs + terms.iter().map(|t| t.abs()).sum::<f64>();
        let bar = f64::EPSILON * s_abs * ((summands + terms.len()) as f64).log2().ceil();
        assert!(sufficient.abs() > bar);
        assert!((path - sufficient).abs() <= bar, "{path} vs {sufficient}, bar {bar}");
        assert_eq!(history.open_risk_set(&spec.marks), None);
        let open = spec.history(&subject(0.0, 4.0, &[(1.0, 0), (0.0, 2)])).unwrap();
        assert_eq!(
            open.open_risk_set(&spec.marks),
            Some(vec![false, true, false, true])
        );
        assert_eq!(open.rate_statistics(&spec.marks).1, [1.0, 4.0, 0.0, 4.0]);
    }

    #[test]
    fn histories_refuse_records_the_law_cannot_hold() {
        let spec = JointSpecification::new(vec![MarkKind::Once, MarkKind::Terminal, MarkKind::Terminal])
            .unwrap();
        for (record, reason) in [
            (subject(0.0, 5.0, &[(3.0, 1)]), "must end follow-up"),
            (subject(0.0, 5.0, &[(1.0, 0), (2.0, 0)]), "fires at most once"),
            (subject(0.0, 5.0, &[(6.0, 0)]), "no event follows the exit"),
            (subject(0.0, 5.0, &[(0.0, 1)]), "no follow-up to model"),
            (subject(0.0, 5.0, &[(5.0, 1), (5.0, 2)]), "after the terminal event"),
            (subject(5.0, 5.0, &[]), "entry < exit"),
        ] {
            let error = spec.history(&record).err().unwrap().to_string();
            assert!(error.contains(reason), "{error}");
        }
        let mut segmented = subject(0.0, 5.0, &[]);
        segmented.segments.push(crate::CovariateSegment { start: 0.0, row: 0 });
        let error = spec.history(&segmented).err().unwrap().to_string();
        assert!(error.contains("no covariate terms"), "{error}");
        assert!(JointSpecification::new(Vec::new()).is_err());
    }
}
