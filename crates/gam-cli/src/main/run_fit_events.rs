//! `gam fit-events`: fit an event-history model from three CSV tables and
//! write a JSON summary with optional per-mark risk forecasts.

use crate::cli_args::FitEventsArgs;
use gam::families::custom_family::BlockwiseFitOptions;
use gam::families::event_history::{
    CovariateSegment, Event, EventHistoryCohort, MarkKind, SubjectHistory,
    fit_event_history_formula, forecast, kolmogorov_smirnov_uniform, population_forecast,
    predictive_pit,
};
use ndarray::Array2;
use serde_json::{Map, Value, json};
use std::collections::HashMap;
use std::path::Path;

fn read_csv(path: &Path) -> Result<(Vec<String>, Vec<Vec<String>>), String> {
    let mut reader = csv::Reader::from_path(path)
        .map_err(|error| format!("cannot open {}: {error}", path.display()))?;
    let headers: Vec<String> = reader
        .headers()
        .map_err(|error| format!("{}: {error}", path.display()))?
        .iter()
        .map(|h| h.trim().to_string())
        .collect();
    let mut rows = Vec::new();
    for record in reader.records() {
        let record = record.map_err(|error| format!("{}: {error}", path.display()))?;
        rows.push(record.iter().map(|v| v.trim().to_string()).collect());
    }
    Ok((headers, rows))
}

fn column<'a>(headers: &[String], rows: &'a [Vec<String>], name: &str, path: &Path) -> Result<Vec<&'a str>, String> {
    let index = headers
        .iter()
        .position(|h| h == name)
        .ok_or_else(|| format!("{} has no column {name:?}", path.display()))?;
    rows.iter()
        .map(|row| {
            row.get(index)
                .map(|v| v.as_str())
                .ok_or_else(|| format!("{}: a row is shorter than its header", path.display()))
        })
        .collect()
}

fn parse_f64(value: &str, what: &str) -> Result<f64, String> {
    value
        .parse::<f64>()
        .map_err(|_| format!("{what}: {value:?} is not a number"))
}

fn matrix_rows(matrix: &Array2<f64>) -> Vec<Vec<Option<f64>>> {
    matrix
        .rows()
        .into_iter()
        .map(|r| r.iter().map(|v| if v.is_finite() { Some(*v) } else { None }).collect())
        .collect()
}

pub(crate) fn run_fit_events(args: FitEventsArgs) -> Result<(), String> {
    let (subject_headers, subject_rows) = read_csv(&args.subjects)?;
    let ids = column(&subject_headers, &subject_rows, "id", &args.subjects)?;
    let entries = column(&subject_headers, &subject_rows, "entry", &args.subjects)?;
    let exits = column(&subject_headers, &subject_rows, "exit", &args.subjects)?;
    let mut index: HashMap<String, usize> = HashMap::new();
    let mut subjects: Vec<SubjectHistory> = Vec::with_capacity(ids.len());
    for (i, ((id, entry), exit)) in ids.iter().zip(entries.iter()).zip(exits.iter()).enumerate() {
        if index.insert((*id).to_string(), i).is_some() {
            return Err(format!("duplicate subject id {id:?} in {}", args.subjects.display()));
        }
        subjects.push(SubjectHistory {
            id: (*id).to_string(),
            entry: parse_f64(entry, "entry")?,
            exit: parse_f64(exit, "exit")?,
            events: Vec::new(),
            segments: Vec::new(),
        });
    }
    let (event_headers, event_rows) = read_csv(&args.events)?;
    let event_ids = column(&event_headers, &event_rows, "id", &args.events)?;
    let event_times = column(&event_headers, &event_rows, "time", &args.events)?;
    let event_marks = column(&event_headers, &event_rows, "mark", &args.events)?;
    let mut mark_names: Vec<String> = event_marks.iter().map(|m| (*m).to_string()).collect();
    mark_names.extend(args.once.iter().cloned());
    mark_names.extend(args.terminal.iter().cloned());
    mark_names.sort();
    mark_names.dedup();
    if mark_names.is_empty() {
        return Err("the events table has no rows".to_string());
    }
    if let Some(both) = args.once.iter().find(|m| args.terminal.contains(m)) {
        return Err(format!("mark {both:?} cannot be both once-only and terminal"));
    }
    let mark_kinds: Vec<MarkKind> = mark_names
        .iter()
        .map(|m| {
            if args.terminal.contains(m) {
                MarkKind::Terminal
            } else if args.once.contains(m) {
                MarkKind::Once
            } else {
                MarkKind::Recurrent
            }
        })
        .collect();
    for ((id, time), mark) in event_ids.iter().zip(event_times.iter()).zip(event_marks.iter()) {
        let subject = *index
            .get(*id)
            .ok_or_else(|| format!("event subject {id:?} is not in the subjects table"))?;
        let mark_index = mark_names.iter().position(|m| m == mark).unwrap_or(0);
        subjects[subject].events.push(Event {
            time: parse_f64(time, "event time")?,
            mark: mark_index,
        });
    }
    let (cov_headers, cov_rows) = read_csv(&args.covariates)?;
    let cov_ids = column(&cov_headers, &cov_rows, "id", &args.covariates)?;
    let cov_starts = column(&cov_headers, &cov_rows, "start", &args.covariates)?;
    let covariate_names: Vec<String> = cov_headers
        .iter()
        .filter(|h| h.as_str() != "id" && h.as_str() != "start")
        .cloned()
        .collect();
    if covariate_names.is_empty() {
        return Err("the covariates table needs at least one covariate column".to_string());
    }
    let mut table = Array2::<f64>::zeros((cov_rows.len(), covariate_names.len()));
    for (j, name) in covariate_names.iter().enumerate() {
        let values = column(&cov_headers, &cov_rows, name, &args.covariates)?;
        for (i, value) in values.iter().enumerate() {
            table[[i, j]] = parse_f64(value, name)?;
        }
    }
    for (row, (id, start)) in cov_ids.iter().zip(cov_starts.iter()).enumerate() {
        let subject = *index
            .get(*id)
            .ok_or_else(|| format!("covariate subject {id:?} is not in the subjects table"))?;
        subjects[subject].segments.push(CovariateSegment {
            start: parse_f64(start, "segment start")?,
            row,
        });
    }
    let mut cohort = EventHistoryCohort {
        mark_names: mark_names.clone(),
        mark_kinds: mark_kinds.clone(),
        covariate_names: covariate_names.clone(),
        covariates: table,
        subjects,
    };
    let fit = fit_event_history_formula(&mut cohort, &args.formula, BlockwiseFitOptions::default())
        .map_err(|e| e.to_string())?;

    let mut summary = Map::new();
    summary.insert("marks".to_string(), json!(mark_names));
    summary.insert(
        "mark_kinds".to_string(),
        json!(mark_kinds.iter().map(|k| k.as_str()).collect::<Vec<_>>()),
    );
    summary.insert("covariates".to_string(), json!(covariate_names));
    summary.insert("formula".to_string(), json!(args.formula));
    summary.insert("rank".to_string(), json!(fit.rank()));
    summary.insert("log_likelihood".to_string(), json!(fit.fit.log_likelihood));
    summary.insert("reml_score".to_string(), json!(fit.fit.reml_score()));
    summary.insert("outer_iterations".to_string(), json!(fit.fit.outer_iterations));
    summary.insert("time_scale".to_string(), json!(fit.time_scale));
    summary.insert("loadings".to_string(), json!(matrix_rows(&fit.loadings)));
    summary.insert("rates".to_string(), json!(fit.rates));
    summary.insert("atom_log_lambdas".to_string(), json!(fit.atom_log_lambdas));
    summary.insert("atom_evidence".to_string(), json!(fit.atom_evidence));
    summary.insert(
        "rank_path".to_string(),
        json!(fit
            .rank_path
            .iter()
            .map(|step| {
                json!({
                    "rank": step.rank,
                    "score_eigenvalue": step.score_eigenvalue,
                    "proposed_log_rate": step.proposed_log_rate,
                    "at_resolution_limit": step.at_resolution_limit,
                    "evidence_gain": step.evidence_gain,
                    "accepted": step.accepted,
                })
            })
            .collect::<Vec<_>>()),
    );
    summary.insert(
        "disease_covariance".to_string(),
        json!(matrix_rows(&fit.disease_covariance())),
    );
    let (eigenvalues, eigenvectors) = fit.eigenmodes().map_err(|e| e.to_string())?;
    summary.insert("eigenvalues".to_string(), json!(eigenvalues.to_vec()));
    summary.insert("eigenvectors".to_string(), json!(matrix_rows(&eigenvectors)));
    summary.insert(
        "coefficients".to_string(),
        json!((0..fit.marks())
            .map(|d| fit.mark_coefficients(d).to_vec())
            .collect::<Vec<_>>()),
    );
    let mut pits = Vec::new();
    for subject in &cohort.subjects {
        pits.extend(predictive_pit(&fit, &cohort, subject).map_err(|e| e.to_string())?);
    }
    summary.insert("pit_events".to_string(), json!(pits.len()));
    summary.insert(
        "pit_ks".to_string(),
        json!(kolmogorov_smirnov_uniform(&pits)),
    );
    if !args.horizons.is_empty() {
        let mut forecasts = Vec::with_capacity(cohort.subjects.len());
        for subject in &cohort.subjects {
            if subject
                .events
                .iter()
                .any(|e| mark_kinds[e.mark] == MarkKind::Terminal)
            {
                continue;
            }
            let horizons: Vec<f64> = args.horizons.iter().map(|h| subject.exit + h).collect();
            let future_row = subject.segments.last().map(|s| s.row).unwrap_or(0);
            let f = forecast(&fit, &cohort, subject, &horizons, future_row).map_err(|e| e.to_string())?;
            // The same window from the stationary prior at the subject's
            // covariates: what the model says without its history.
            let alone = population_forecast(
                &fit,
                &cohort,
                &cohort.covariates.row(future_row).to_vec(),
                subject.exit,
                &horizons,
            )
            .map_err(|e| e.to_string())?;
            forecasts.push(json!({
                "id": subject.id,
                "horizons": f.horizons,
                "risk": matrix_rows(&f.risk),
                "survival": f.survival,
                "expected_counts": matrix_rows(&f.expected_counts),
                "without_history": {
                    "risk": matrix_rows(&alone.risk),
                    "survival": alone.survival,
                    "expected_counts": matrix_rows(&alone.expected_counts),
                },
            }));
        }
        summary.insert("forecasts".to_string(), Value::Array(forecasts));
    }
    let text = serde_json::to_string_pretty(&Value::Object(summary))
        .map_err(|error| format!("serialising the summary: {error}"))?;
    match &args.out {
        Some(path) => std::fs::write(path, text)
            .map_err(|error| format!("writing {}: {error}", path.display())),
        None => {
            use std::io::Write;
            let mut stdout = std::io::stdout().lock();
            stdout
                .write_all(text.as_bytes())
                .and_then(|()| stdout.write_all(b"\n"))
                .map_err(|error| format!("writing the summary to stdout: {error}"))
        }
    }
}
