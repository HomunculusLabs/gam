//! `gam fit-events`: fit an event-history model from three CSV tables and
//! write a JSON summary with optional forecasts.

use crate::cli_args::FitEventsArgs;
use gam::families::custom_family::BlockwiseFitOptions;
use gam::families::event_history::{
    CovariateSegment, Event, EventHistoryCohort, ForecastRequest, FutureSegment, MarkKind,
    PopulationForecastRequest, SubjectHistory, fit_event_history_formula, forecast,
    kolmogorov_smirnov_uniform, population_forecast, predictive_pit,
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

/// A covariate column: continuous when every value parses as a number,
/// categorical (coded by its sorted distinct labels) otherwise.
fn encode_column(values: &[&str], name: &str) -> (Vec<f64>, Vec<String>) {
    if let Ok(numbers) = values
        .iter()
        .map(|v| parse_f64(v, name))
        .collect::<Result<Vec<f64>, String>>()
    {
        return (numbers, Vec::new());
    }
    let mut levels: Vec<String> = values.iter().map(|v| (*v).to_string()).collect();
    levels.sort();
    levels.dedup();
    let codes = values
        .iter()
        .map(|v| levels.iter().position(|l| l == v).expect("level present") as f64)
        .collect();
    (codes, levels)
}

fn forecast_json(f: &gam::families::event_history::Forecast) -> Value {
    json!({
        "horizons": f.horizons,
        "survival": f.survival,
        "expected_counts": f.expected_counts.rows().into_iter().map(|r| r.to_vec()).collect::<Vec<_>>(),
    })
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
    // The mark vocabulary: declared with kinds, or the observed marks, all
    // recurrent.
    let (mark_names, mark_kinds): (Vec<String>, Vec<MarkKind>) = if args.marks.is_empty() {
        let mut names: Vec<String> = event_marks.iter().map(|m| (*m).to_string()).collect();
        names.sort();
        names.dedup();
        if names.is_empty() {
            return Err(
                "the events table has no rows, so the marks must be declared with --marks name:kind,..."
                    .to_string(),
            );
        }
        let kinds = vec![MarkKind::Recurrent; names.len()];
        (names, kinds)
    } else {
        let mut names = Vec::with_capacity(args.marks.len());
        let mut kinds = Vec::with_capacity(args.marks.len());
        for spec in &args.marks {
            let (name, kind) = spec
                .split_once(':')
                .ok_or_else(|| format!("--marks entry {spec:?} is not name:kind"))?;
            names.push(name.trim().to_string());
            kinds.push(MarkKind::parse(kind).map_err(|e| e.to_string())?);
        }
        (names, kinds)
    };
    for ((id, time), mark) in event_ids.iter().zip(event_times.iter()).zip(event_marks.iter()) {
        let subject = *index
            .get(*id)
            .ok_or_else(|| format!("event subject {id:?} is not in the subjects table"))?;
        let mark_index = mark_names
            .iter()
            .position(|m| m == mark)
            .ok_or_else(|| format!("event mark {mark:?} is not in the mark vocabulary {mark_names:?}"))?;
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
    let mut table = Array2::<f64>::zeros((cov_rows.len(), covariate_names.len()));
    let mut covariate_levels = Vec::with_capacity(covariate_names.len());
    for (j, name) in covariate_names.iter().enumerate() {
        let values = column(&cov_headers, &cov_rows, name, &args.covariates)?;
        let (codes, levels) = encode_column(&values, name);
        for (i, code) in codes.iter().enumerate() {
            table[[i, j]] = *code;
        }
        covariate_levels.push(levels);
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
        covariate_levels: covariate_levels.clone(),
        covariates: table,
        subjects,
    };
    let fit = fit_event_history_formula(
        &mut cohort,
        &args.formula,
        args.atoms,
        BlockwiseFitOptions::default(),
    )
    .map_err(|e| e.to_string())?;

    let mut summary = Map::new();
    summary.insert("marks".to_string(), json!(mark_names));
    summary.insert(
        "mark_kinds".to_string(),
        json!(mark_kinds.iter().map(|k| k.name()).collect::<Vec<_>>()),
    );
    summary.insert("covariates".to_string(), json!(covariate_names));
    summary.insert("covariate_levels".to_string(), json!(covariate_levels));
    summary.insert("formula".to_string(), json!(args.formula));
    summary.insert("atoms".to_string(), json!(args.atoms));
    summary.insert("log_likelihood".to_string(), json!(fit.fit.log_likelihood));
    summary.insert("reml_score".to_string(), json!(fit.fit.reml_score()));
    summary.insert("outer_iterations".to_string(), json!(fit.fit.outer_iterations));
    summary.insert("time_scale".to_string(), json!(fit.time_scale));
    summary.insert(
        "loadings".to_string(),
        json!(fit.loadings.rows().into_iter().map(|r| r.to_vec()).collect::<Vec<_>>()),
    );
    summary.insert("rates".to_string(), json!(fit.rates));
    summary.insert("atom_log_lambdas".to_string(), json!(fit.atom_log_lambdas));
    summary.insert(
        "coefficients".to_string(),
        json!((0..fit.marks())
            .map(|d| fit.mark_coefficients(d).to_vec())
            .collect::<Vec<_>>()),
    );
    let q = &fit.quadrature;
    summary.insert(
        "quadrature".to_string(),
        json!({
            "gauss_hermite_order": q.gauss_hermite_order,
            "mesh_refinement": q.mesh_refinement,
            "log_likelihood": q.log_likelihood,
            "gauss_hermite_check": {
                "order": q.gauss_hermite.candidate,
                "coefficient_shift": q.gauss_hermite.coefficient_shift,
                "log_likelihood": q.gauss_hermite.log_likelihood,
            },
            "mesh_check": {
                "refinement": q.mesh.candidate,
                "coefficient_shift": q.mesh.coefficient_shift,
                "log_likelihood": q.mesh.log_likelihood,
            },
        }),
    );
    let mut pits = Vec::new();
    for subject in &cohort.subjects {
        pits.extend(
            predictive_pit(&fit, &cohort, subject)
                .map_err(|e| e.to_string())?
                .into_iter()
                .map(|p| p.pit),
        );
    }
    summary.insert("pit_events".to_string(), json!(pits.len()));
    summary.insert(
        "pit_ks".to_string(),
        json!(kolmogorov_smirnov_uniform(&pits)),
    );
    if !args.horizons_after_exit.is_empty() {
        let mut forecasts = Vec::with_capacity(cohort.subjects.len());
        for subject in &cohort.subjects {
            let horizons: Vec<f64> = args.horizons_after_exit.iter().map(|h| subject.exit + h).collect();
            let f = forecast(
                &fit,
                &cohort,
                &ForecastRequest {
                    history: subject,
                    horizons: &horizons,
                    future: &[],
                },
            )
            .map_err(|e| e.to_string())?;
            // The same window from the stationary prior at the subject's
            // covariates at exit: what the model says without its history.
            let row = subject.covariate_row_at(subject.exit, false);
            let alone = population_forecast(
                &fit,
                &cohort,
                &PopulationForecastRequest {
                    start: subject.exit,
                    horizons: &horizons,
                    future: &[FutureSegment {
                        start: subject.exit,
                        covariates: cohort.covariates.row(row).to_vec(),
                    }],
                },
            )
            .map_err(|e| e.to_string())?;
            let mut entry = forecast_json(&f);
            entry["id"] = json!(subject.id);
            entry["without_history"] = json!({
                "survival": alone.survival,
                "expected_counts": alone.expected_counts.rows().into_iter().map(|r| r.to_vec()).collect::<Vec<_>>(),
            });
            forecasts.push(entry);
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
