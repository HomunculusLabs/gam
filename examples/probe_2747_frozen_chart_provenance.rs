//! gam#2747: WHY is a Matérn smooth the one family that ships un-orthogonalized?
//!
//! `probe_2747_parametric_orthogonality` measures the symptom through the real
//! pipeline: with the projection arm landed, `tps`, `duchon` and `curv` all sit
//! at `1e-14` against `[1 | x1]` while `matern` sits at `4.15e-1` — three
//! decades of difference between families that go through the *same* global
//! step. So the arm is not at fault and the difference is upstream of it.
//!
//! The candidate is a marker, not a computation. `apply_global_smooth_
//! identifiability` skips its whole pass when `smooth_has_frozen_identifiability`
//! is true, on the documented premise that *"a transform frozen by this pipeline
//! already has the parametric orthogonalization composed in"* — which is exactly
//! what `with_identifiability_transform` does when the global step itself
//! freezes. But `freeze_geometry_from_metadata` (`spatial_optimization.rs`)
//! ALSO writes that marker, from the κ optimizer's SINGLE-TERM local build,
//! whose own comment says it "never runs the global ownership pass". For that
//! producer the premise is false.
//!
//! This probe decides it by A/B on one fitted spec, so the claim is a
//! measurement rather than a reading:
//!
//!   1. fit `y ~ x1 + matern(x1, x2)` and print the resolved spec's
//!      identifiability variant (`FrozenTransform` ⇒ the marker is set);
//!   2. rebuild the design from that spec verbatim → the shipped number;
//!   3. rebuild from the SAME spec with the identifiability reverted to
//!      `CenterSumToZero` and nothing else touched → if the number collapses to
//!      the shipped bar, the marker is the cause and the transform's CONTENT is
//!      not;
//!   4. print `Z`'s shape beside the local chart's, because a `Z` that is the
//!      LOCAL chart alone (`centers × centers−1`) is the direct evidence that
//!      nothing global was ever composed into it.
//!
//! Run: `cargo run --release --example probe_2747_frozen_chart_provenance`

use csv::StringRecord;
use gam::basis::MaternIdentifiability;
use gam::smooth::SmoothBasisSpec;
use gam::{FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula};
use ndarray::Array2;

const N: usize = 400;
const CENTERS: usize = 24;

fn rows() -> Vec<(f64, f64, f64)> {
    let mut state = 0x2747_0000_0000_0002_u64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / (1u64 << 53) as f64
    };
    let mut out = Vec::with_capacity(N);
    while out.len() < N {
        let a = 2.0 * next() - 1.0;
        let b = 2.0 * next() - 1.0;
        if a * a + b * b <= 1.0 {
            let signal = 1.3 * a + (3.0 * (a * a + b * b)).sin();
            let noise = 0.1 * (2.0 * next() - 1.0);
            out.push((a, b, signal + noise));
        }
    }
    out
}

fn relative_cross(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    let cross = a.t().dot(b);
    let num = cross.iter().map(|v| v * v).sum::<f64>().sqrt();
    let a_norm = a.iter().map(|v| v * v).sum::<f64>().sqrt();
    let b_norm = b.iter().map(|v| v * v).sum::<f64>().sqrt();
    num / (a_norm * b_norm).max(1e-300)
}

fn smooth_block_offset(design: &gam::smooth::TermCollectionDesign) -> usize {
    design
        .intercept_range
        .end
        .max(
            design
                .linear_ranges
                .iter()
                .map(|(_, range)| range.end)
                .max()
                .unwrap_or(0),
        )
        .max(
            design
                .random_effect_ranges
                .iter()
                .map(|(_, range)| range.end)
                .max()
                .unwrap_or(0),
        )
}

/// `‖X_smoothᵀ C‖/(‖X‖‖C‖)` and the block width, for one term of one spec.
fn measure_spec(
    frame: &Array2<f64>,
    spec: &gam::smooth::TermCollectionSpec,
    term_name: &str,
    constraint: &Array2<f64>,
) -> Option<(f64, usize)> {
    let design = match gam::smooth::build_term_collection_design(frame.view(), spec) {
        Ok(design) => design,
        Err(err) => {
            println!("        rebuild refused: {err}");
            return None;
        }
    };
    let dense = design.design.to_dense();
    let offset = smooth_block_offset(&design);
    let realized = design
        .smooth
        .terms
        .iter()
        .find(|built| built.name == term_name)?;
    let block = dense
        .slice(ndarray::s![
            ..,
            offset + realized.coeff_range.start..offset + realized.coeff_range.end
        ])
        .to_owned();
    let cross = relative_cross(&block, constraint);
    Some((cross, block.ncols()))
}

fn main() {
    let data = rows();
    let headers = ["x1", "x2", "y"].into_iter().map(String::from).collect();
    let records: Vec<StringRecord> = data
        .iter()
        .map(|(x1, x2, y)| StringRecord::from(vec![x1.to_string(), x2.to_string(), y.to_string()]))
        .collect();
    let encoded = encode_recordswith_inferred_schema(headers, records).expect("encode");
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };

    let mut frame = Array2::<f64>::zeros((data.len(), 3));
    for (i, (x1, x2, y)) in data.iter().enumerate() {
        frame[(i, 0)] = *x1;
        frame[(i, 1)] = *x2;
        frame[(i, 2)] = *y;
    }

    for formula in [
        &format!("y ~ matern(x1, x2, centers={CENTERS})")[..],
        &format!("y ~ x1 + matern(x1, x2, centers={CENTERS})")[..],
    ] {
        println!("{formula}");
        let fit = match fit_from_formula(formula, &encoded, &cfg) {
            Ok(FitResult::Standard(fit)) => fit,
            Ok(_) => {
                println!("    not a standard fit");
                continue;
            }
            Err(err) => {
                println!("    refused: {err}");
                continue;
            }
        };

        // `[1 | x1]` when the formula names `x1`, `[1]` otherwise.
        let names_x1 = formula.contains("x1 +");
        let width = if names_x1 { 2 } else { 1 };
        let mut constraint = Array2::<f64>::ones((data.len(), width));
        if names_x1 {
            for i in 0..data.len() {
                constraint[(i, 1)] = frame[(i, 0)];
            }
        }

        let shipped = fit.resolvedspec.clone();
        let term_name = shipped.smooth_terms[0].name.clone();
        match &shipped.smooth_terms[0].basis {
            SmoothBasisSpec::Matern { spec, .. } => {
                let label = match &spec.identifiability {
                    MaternIdentifiability::None => "None".to_string(),
                    MaternIdentifiability::CenterSumToZero => "CenterSumToZero".to_string(),
                    MaternIdentifiability::CenterLinearOrthogonal => {
                        "CenterLinearOrthogonal".to_string()
                    }
                    MaternIdentifiability::FrozenTransform { transform } => {
                        format!("FrozenTransform{{Z: {:?}}}", transform.dim())
                    }
                };
                println!("    resolved identifiability = {label}");
                println!(
                    "    LOCAL center-sum-to-zero chart would be ({CENTERS}, {}) — a Z of that \
                     exact shape carries NOTHING global",
                    CENTERS - 1
                );
            }
            _ => println!("    not a Matérn term"),
        }

        println!(
            "    frozen_parametric_residualization = {}   joint_null_rotation = {}",
            shipped.smooth_terms[0]
                .frozen_parametric_residualization
                .is_some(),
            shipped.smooth_terms[0].joint_null_rotation.is_some(),
        );
        if let SmoothBasisSpec::Matern { spec, .. } = &shipped.smooth_terms[0].basis {
            println!(
                "    center_strategy = {:?}   include_intercept = {}   double_penalty = {}",
                std::mem::discriminant(&spec.center_strategy),
                spec.include_intercept,
                spec.double_penalty
            );
        }

        // THE decisive split: the design the coefficients were ESTIMATED in,
        // against the design a rebuild from the resolved spec produces. If the
        // first is orthogonal and the second is not, the defect is not "the fit
        // lost the hierarchy" — it is that predict replays a different model.
        {
            let dense = fit.design.design.to_dense();
            let offset = smooth_block_offset(&fit.design);
            if let Some(realized) = fit
                .design
                .smooth
                .terms
                .iter()
                .find(|built| built.name == term_name)
            {
                let block = dense
                    .slice(ndarray::s![
                        ..,
                        offset + realized.coeff_range.start..offset + realized.coeff_range.end
                    ])
                    .to_owned();
                println!(
                    "        FIT's own design    ||X'C||/(||X||||C||) = {:.6e}   cols = {}",
                    relative_cross(&block, &constraint),
                    block.ncols()
                );
            }
        }

        // Each arm reverts ONE more piece of what the fit froze onto the spec,
        // so whichever line moves the number names the gate.
        let arms: Vec<(&str, gam::smooth::TermCollectionSpec)> = {
            let mut out = Vec::new();
            out.push(("shipped spec        ", shipped.clone()));

            let mut unfrozen = shipped.clone();
            if let SmoothBasisSpec::Matern { spec, .. } = &mut unfrozen.smooth_terms[0].basis {
                spec.identifiability = MaternIdentifiability::CenterSumToZero;
            }
            out.push(("- frozen Z          ", unfrozen.clone()));

            let mut no_chart = unfrozen.clone();
            no_chart.smooth_terms[0].frozen_parametric_residualization = None;
            out.push(("- frozen Z, - chart ", no_chart.clone()));

            let mut chart_only = shipped.clone();
            chart_only.smooth_terms[0].frozen_parametric_residualization = None;
            out.push(("- chart only        ", chart_only));

            out
        };
        for (label, spec) in arms {
            if let Some((cross, cols)) = measure_spec(&frame, &spec, &term_name, &constraint) {
                println!("        {label}||X'C||/(||X||||C||) = {cross:.6e}   cols = {cols}");
            }
        }
    }
}
