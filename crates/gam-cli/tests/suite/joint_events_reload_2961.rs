//! #2961 A11 across surfaces: a model fitted and saved by `gam joint-events fit`
//! in its own process reloads in-process and forecasts bit for bit what
//! `gam joint-events forecast` writes, and what the in-memory fit forecasts.

use gam::event_history::joint::{JointEventModel, fit_joint_event_model};
use gam::event_history::{Event, MarkKind, SubjectHistory};
use std::process::{Command, Output};

fn gam(args: &[&str]) -> Output {
    Command::new(gam_test_support::gam_binary!())
        .args(args)
        .output()
        .expect("spawn gam CLI")
}

fn stderr(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

fn history(id: &str, entry: f64, exit: f64, events: &[(f64, usize)]) -> SubjectHistory {
    SubjectHistory {
        id: id.to_string(),
        entry,
        exit,
        events: events
            .iter()
            .map(|&(time, mark)| Event { time, mark })
            .collect(),
        segments: Vec::new(),
    }
}

fn bits(values: impl IntoIterator<Item = f64>) -> Vec<u64> {
    values.into_iter().map(f64::to_bits).collect()
}

/// Every number in a forecast field, flattened row-major, as bits.
fn json_bits(value: &serde_json::Value) -> Vec<u64> {
    match value {
        serde_json::Value::Array(items) => items.iter().flat_map(json_bits).collect(),
        serde_json::Value::Number(number) => {
            vec![number.as_f64().expect("a finite number").to_bits()]
        }
        other => vec![f64::NAN.to_bits(), other.to_string().len() as u64],
    }
}

#[test]
fn a_cli_saved_joint_model_forecasts_bit_identically_in_every_process() {
    let scratch = tempfile::tempdir().expect("scratch directory");
    let path = |name: &str| {
        scratch
            .path()
            .join(name)
            .to_str()
            .expect("UTF-8 path")
            .to_string()
    };
    std::fs::write(path("subjects.csv"), "id,entry,exit\na,0,4\nb,0,6\n").expect("write subjects");
    std::fs::write(
        path("events.csv"),
        "id,time,mark\na,4,cvd_death\nb,0,diagnosis\nb,1,visit\nb,5,visit\n",
    )
    .expect("write events");
    std::fs::write(path("new_subjects.csv"), "id,entry,exit\nnew,0.5,2.5\n")
        .expect("write new subjects");
    std::fs::write(path("new_events.csv"), "id,time,mark\nnew,1,visit\n")
        .expect("write new events");
    let marks = "diagnosis:once,cvd_death:terminal,other_death:terminal,visit:recurrent";

    let fit = gam(&[
        "joint-events",
        "fit",
        "--subjects",
        &path("subjects.csv"),
        "--events",
        &path("events.csv"),
        "--marks",
        marks,
        "--out",
        &path("model.json"),
    ]);
    assert_eq!(fit.status.code(), Some(0), "{}", stderr(&fit));
    let forecast = gam(&[
        "joint-events",
        "forecast",
        "--model",
        &path("model.json"),
        "--subjects",
        &path("new_subjects.csv"),
        "--events",
        &path("new_events.csv"),
        "--horizons",
        "0.25,3,40",
        "--out",
        &path("forecast.json"),
    ]);
    assert_eq!(forecast.status.code(), Some(0), "{}", stderr(&forecast));
    let written: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(path("forecast.json")).expect("read forecast"))
            .expect("parse forecast");
    let cli = &written["forecasts"][0];
    assert_eq!(cli["id"], "new");

    let new = history("new", 0.5, 2.5, &[(1.0, 3)]);
    let horizons = [0.25, 3.0, 40.0];
    let in_memory = fit_joint_event_model(
        ["diagnosis", "cvd_death", "other_death", "visit"]
            .map(str::to_string)
            .to_vec(),
        vec![
            MarkKind::Once,
            MarkKind::Terminal,
            MarkKind::Terminal,
            MarkKind::Recurrent,
        ],
        &[
            history("a", 0.0, 4.0, &[(4.0, 1)]),
            history("b", 0.0, 6.0, &[(0.0, 0), (1.0, 3), (5.0, 3)]),
        ],
    )
    .expect("in-memory fit")
    .condition(&new)
    .expect("condition")
    .forecast(&horizons)
    .expect("forecast");
    let reloaded = JointEventModel::load(scratch.path().join("model.json").as_path())
        .expect("reload the CLI's model")
        .condition(&new)
        .expect("condition")
        .forecast(&horizons)
        .expect("forecast");
    // The once-only diagnosis against both terminal causes takes the bracketed
    // quadrature route, so the comparison covers its reported error too.
    assert!(reloaded.incidence_error.iter().any(|&e| e > 0.0));
    for forecast in [&in_memory, &reloaded] {
        assert_eq!(json_bits(&cli["horizons"]), bits(forecast.horizons.iter().copied()));
        assert_eq!(json_bits(&cli["survival"]), bits(forecast.survival.iter().copied()));
        assert_eq!(json_bits(&cli["incidence"]), bits(forecast.incidence.iter().copied()));
        assert_eq!(
            json_bits(&cli["incidence_error"]),
            bits(forecast.incidence_error.iter().copied())
        );
    }
}
