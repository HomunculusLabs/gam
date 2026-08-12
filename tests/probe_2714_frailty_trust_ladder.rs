//! #2714 probe: the same fit as `repro_2714_latent_frailty_inner_solve`, run
//! with the solver's own diagnostic backend installed.
//!
//! The witness's binary installs no `log` backend, so every `log::info!` the
//! joint-Newton inner emits — the per-attempt trust ladder, the
//! `TrustRatioRefinementWitness` model-consistency fault, the objective's
//! measured resolution — is discarded. Reading those lines is what separates
//! "the region is wrong" from "the two ends of ρ measure different functions",
//! and it costs one line to turn on.
//!
//! `#[ignore]` because it is an instrument: it prints, it does not grade. Run it
//! with
//!
//! ```text
//! cargo test --test probe_2714_frailty_trust_ladder -- --ignored --nocapture
//! ```
//!
//! The graded half is `repro_2714_latent_frailty_inner_solve`, which asserts the
//! fit converges and is not `#[ignore]`d.

use csv::StringRecord;
use gam::families::survival::lognormal_kernel::{FrailtyScale, FrailtySpec, HazardLoading};
use gam::{
    FitConfig, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
    load_csvwith_inferred_schema,
};
use std::path::Path;

const VETERAN_CSV: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/bench/datasets/veteran_lung.csv"
);

#[test]
#[ignore = "instrument: prints the inner solve's trust ladder, asserts nothing about it"]
fn probe_2714_latent_frailty_trust_ladder() {
    gam::test_support::install_diagnostic_logger();
    init_parallelism();

    let raw = load_csvwith_inferred_schema(Path::new(VETERAN_CSV)).expect("load veteran_lung.csv");
    let rcol = raw.column_map();
    let (r_time, r_status, r_karno, r_celltype) = (
        rcol["time"],
        rcol["status"],
        rcol["karno"],
        rcol["celltype"],
    );
    let celltype_levels = &raw.schema.columns[r_celltype].levels;
    let n = raw.values.nrows();

    let time: Vec<f64> = raw.values.column(r_time).to_vec();
    let status: Vec<f64> = raw.values.column(r_status).to_vec();
    let karno: Vec<f64> = raw.values.column(r_karno).to_vec();
    let celltype_label: Vec<String> = raw
        .values
        .column(r_celltype)
        .iter()
        .map(|&code| celltype_levels[code as usize].clone())
        .collect();

    let train_rows: Vec<usize> = (0..n).filter(|i| i % 4 != 0).collect();
    let headers = vec![
        "time".to_string(),
        "status".to_string(),
        "karno".to_string(),
        "celltype".to_string(),
    ];
    let train_records: Vec<StringRecord> = train_rows
        .iter()
        .map(|&i| {
            StringRecord::from(vec![
                time[i].to_string(),
                status[i].to_string(),
                karno[i].to_string(),
                celltype_label[i].clone(),
            ])
        })
        .collect();
    let train_ds = encode_recordswith_inferred_schema(headers, train_records)
        .expect("encode veteran train survival data");

    let cfg = FitConfig {
        survival_likelihood: Some("latent".to_string()),
        baseline_target: "weibull".to_string(),
        time_basis: "ispline".to_string(),
        frailty: FrailtySpec::HazardMultiplier {
            scale: FrailtyScale::Learned { initial_sigma: 0.5 },
            loading: HazardLoading::Full,
        },
        ..FitConfig::default()
    };

    match fit_from_formula("Surv(time, status) ~ karno", &train_ds, &cfg) {
        Ok(_) => eprintln!("[2714-probe] the fit returned Ok"),
        Err(error) => eprintln!("[2714-probe] the fit returned Err:\n{error}"),
    }
    assert_eq!(
        gam::test_support::diagnostic_write_failures(),
        0,
        "#2714 probe: diagnostics were dropped, so this run measured nothing"
    );
}
