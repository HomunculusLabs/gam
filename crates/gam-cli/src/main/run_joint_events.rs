//! `gam joint-events`: fit the joint event model to subjects and events tables
//! and save it, or condition a saved model on histories and forecast (#2961).
//! Both actions run the one Rust model path the Python library also calls.

use crate::cli_args::{
    JointEventsAction, JointEventsArgs, JointEventsFitArgs, JointEventsForecastArgs,
};
use gam::event_history::joint::{JointEventModel, fit_joint_event_model};
use gam::event_history::{Event, MarkKind, SubjectHistory, mark_index_of, resolve_mark_vocabulary};
use ndarray::Array2;
use serde_json::json;
use std::collections::HashMap;
use std::path::Path;

/// The named columns of a CSV table, in the order named.
fn read_columns(path: &Path, names: &[&str]) -> Result<Vec<Vec<String>>, String> {
    let mut reader = csv::Reader::from_path(path)
        .map_err(|error| format!("cannot open {}: {error}", path.display()))?;
    let headers: Vec<String> = reader
        .headers()
        .map_err(|error| format!("{}: {error}", path.display()))?
        .iter()
        .map(|h| h.trim().to_string())
        .collect();
    let indices = names
        .iter()
        .map(|name| {
            headers
                .iter()
                .position(|h| h == name)
                .ok_or_else(|| format!("{} has no column {name:?}", path.display()))
        })
        .collect::<Result<Vec<usize>, String>>()?;
    let mut columns = vec![Vec::new(); names.len()];
    for record in reader.records() {
        let record = record.map_err(|error| format!("{}: {error}", path.display()))?;
        for (column, &index) in columns.iter_mut().zip(&indices) {
            let value = record
                .get(index)
                .ok_or_else(|| format!("{}: a row is shorter than its header", path.display()))?;
            column.push(value.trim().to_string());
        }
    }
    Ok(columns)
}

fn parse_f64(value: &str, what: &str) -> Result<f64, String> {
    value
        .parse::<f64>()
        .map_err(|_| format!("{what}: {value:?} is not a number"))
}

/// The subjects with no events yet, and every event row as its subject's index,
/// time and mark label.
fn read_histories(
    subjects: &Path,
    events: &Path,
) -> Result<(Vec<SubjectHistory>, Vec<(usize, f64, String)>), String> {
    let columns = read_columns(subjects, &["id", "entry", "exit"])?;
    let mut index = HashMap::new();
    let mut histories = Vec::with_capacity(columns[0].len());
    for (i, ((id, entry), exit)) in columns[0]
        .iter()
        .zip(&columns[1])
        .zip(&columns[2])
        .enumerate()
    {
        if index.insert(id.clone(), i).is_some() {
            return Err(format!("duplicate subject id {id:?} in {}", subjects.display()));
        }
        histories.push(SubjectHistory {
            id: id.clone(),
            entry: parse_f64(entry, "entry")?,
            exit: parse_f64(exit, "exit")?,
            events: Vec::new(),
            segments: Vec::new(),
        });
    }
    let rows = read_columns(events, &["id", "time", "mark"])?;
    let mut records = Vec::with_capacity(rows[0].len());
    for ((id, time), mark) in rows[0].iter().zip(&rows[1]).zip(&rows[2]) {
        let subject = *index
            .get(id)
            .ok_or_else(|| format!("event subject {id:?} is not in the subjects table"))?;
        records.push((subject, parse_f64(time, "event time")?, mark.clone()));
    }
    Ok((histories, records))
}

fn fit(args: JointEventsFitArgs) -> Result<(), String> {
    let (mut histories, records) = read_histories(&args.subjects, &args.events)?;
    let declared = if args.marks.is_empty() {
        None
    } else {
        let mut pairs = Vec::with_capacity(args.marks.len());
        for spec in &args.marks {
            let (name, kind) = spec
                .split_once(':')
                .ok_or_else(|| format!("--marks entry {spec:?} is not name:kind"))?;
            pairs.push((name.trim().to_string(), MarkKind::parse(kind).map_err(|e| e.to_string())?));
        }
        Some(pairs)
    };
    let labels: Vec<&str> = records.iter().map(|record| record.2.as_str()).collect();
    let (mark_names, mark_kinds, marks) =
        resolve_mark_vocabulary(declared, &labels).map_err(|e| e.to_string())?;
    for (record, mark) in records.iter().zip(marks) {
        histories[record.0].events.push(Event {
            time: record.1,
            mark,
        });
    }
    let model =
        fit_joint_event_model(mark_names, mark_kinds, &histories).map_err(|e| e.to_string())?;
    model.save(&args.out).map_err(|e| e.to_string())
}

fn rows(matrix: &Array2<f64>) -> Vec<Vec<f64>> {
    matrix.rows().into_iter().map(|r| r.to_vec()).collect()
}

fn forecast(args: JointEventsForecastArgs) -> Result<(), String> {
    let model = JointEventModel::load(&args.model).map_err(|e| e.to_string())?;
    let (mut histories, records) = read_histories(&args.subjects, &args.events)?;
    for (subject, time, label) in records {
        let mark = mark_index_of(model.mark_names(), &label).map_err(|e| e.to_string())?;
        histories[subject].events.push(Event { time, mark });
    }
    let mut forecasts = Vec::with_capacity(histories.len());
    for history in &histories {
        let f = model
            .condition(history)
            .and_then(|conditioned| conditioned.forecast(&args.horizons))
            .map_err(|e| e.to_string())?;
        forecasts.push(json!({
            "id": history.id,
            "time": history.exit,
            "horizons": f.horizons,
            "survival": f.survival,
            "incidence": rows(&f.incidence),
            "incidence_error": rows(&f.incidence_error),
        }));
    }
    let summary = json!({
        "marks": model.mark_names(),
        "mark_kinds": model.mark_kinds().iter().map(|k| k.name()).collect::<Vec<_>>(),
        "forecasts": forecasts,
    });
    let text = serde_json::to_string_pretty(&summary)
        .map_err(|error| format!("serialising the forecasts: {error}"))?;
    match &args.out {
        Some(path) => std::fs::write(path, text)
            .map_err(|error| format!("writing {}: {error}", path.display())),
        None => {
            use std::io::Write;
            let mut stdout = std::io::stdout().lock();
            stdout
                .write_all(text.as_bytes())
                .and_then(|()| stdout.write_all(b"\n"))
                .map_err(|error| format!("writing the forecasts to stdout: {error}"))
        }
    }
}

pub(crate) fn run_joint_events(args: JointEventsArgs) -> Result<(), String> {
    match args.action {
        JointEventsAction::Fit(args) => fit(args),
        JointEventsAction::Forecast(args) => forecast(args),
    }
}
