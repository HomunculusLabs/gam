//! A `gam fit` line names each absence of the comparable REML/LAML criterion by
//! its cause (#2627).
//!
//! `reml_score` is the criterion made comparable across fits by the
//! Tierney-Kadane normalizer over the penalty null space. It can be absent for
//! two different reasons:
//! - a fit without null-space metadata (a survival fit, through the library
//!   formula route) still has its raw criterion, so it has none COMPARABLE;
//! - an exactly-interpolating Gaussian fit has no criterion at all.
//!
//! Both lines used to print "none (exact fit: criterion unbounded)", which is
//! false for the first case. The words now come from one owner,
//! `gam::report::criterion_row` / `criterion_display`, shared with the HTML report.

use std::fmt::Write as _;
use std::process::Command;

fn fit_line(csv: &str, formula: &str) -> String {
    let dir = tempfile::tempdir().expect("temp dir");
    let data = dir.path().join("data.csv");
    std::fs::write(&data, csv).expect("write fixture");
    let out = dir.path().join("model.gam");
    let output = Command::new(gam_test_support::gam_binary!())
        .arg("fit")
        .arg(&data)
        .arg(formula)
        .arg("--out")
        .arg(&out)
        .output()
        .expect("spawn gam fit");
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "gam fit {formula:?} failed (exit {:?}).\nstderr tail: {}",
        output.status.code(),
        stderr.lines().rev().take(6).collect::<Vec<_>>().join("\n")
    );
    stdout
        .lines()
        .find(|line| line.contains(" fit | ") && line.contains("reml_score="))
        .unwrap_or_else(|| panic!("no fit line with reml_score= in stdout:\n{stdout}"))
        .to_string()
}

#[test]
fn a_fit_without_null_space_metadata_prints_none_comparable_and_its_raw_criterion_2627() {
    // Right-censored survival data with a smooth covariate effect.
    let mut csv = String::from("entry,exit,event,x\n");
    for i in 0..120 {
        let x = -1.0 + 2.0 * (i as f64) / 119.0;
        let exit = 0.5 + ((i * 37) % 97) as f64 / 40.0 + 0.4 * (1.0 + x.sin());
        let event = u8::from((i * 11) % 5 != 0);
        writeln!(csv, "0,{exit},{event},{x}").expect("format row");
    }
    let line = fit_line(&csv, "Surv(entry, exit, event) ~ s(x)");
    assert!(
        line.contains("reml_score=none comparable (no null-space metadata); raw criterion "),
        "a survival fit has its raw criterion but no null-space metadata: {line}"
    );
    assert!(
        !line.contains("exact fit"),
        "a fit that only lacks null-space metadata is not an exact fit: {line}"
    );
}

#[test]
fn an_exactly_interpolating_gaussian_fit_prints_the_exact_fit_words_2627() {
    // y is an exact affine function of x, so the profiled Gaussian scale is zero
    // and the restricted likelihood is unbounded: there is no criterion at all.
    let mut csv = String::from("x,y\n");
    for i in 0..80 {
        let x = -1.0 + 2.0 * (i as f64) / 79.0;
        let y = 0.5 + 1.25 * x;
        writeln!(csv, "{x},{y}").expect("format row");
    }
    let line = fit_line(&csv, "y ~ x");
    let words = gam::report::criterion_display(None);
    assert!(
        line.contains(&format!("reml_score={words} | raw_reml_score={words}")),
        "an exact fit has neither criterion, in the one owner's words: {line}"
    );
}
