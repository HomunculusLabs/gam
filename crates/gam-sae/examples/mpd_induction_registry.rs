//! Run mpd-induction's S1 stage receipt for every linear use against torch's executed stages (#2951).
//!
//! `mpd_induction_registry --run EXECUTE_DIR --settings SETTINGS_JSON --out REPORT_JSON`
//!
//! This is the example half of the direct-lane receipt runner. `EXECUTE_DIR` is what
//! `bench/mpd_induction_2951.py execute` writes:
//! * `execute.json`, naming the registry export it read (the harvest), the declared settings and the
//!   executor's dtype, device and TF32 flag;
//! * the edit's two factors;
//! * one `<f8` `.npy` per stage, holding every setting's rows in setting order.
//!
//! The registry export holds `registry.json`, one little-endian `<f4` `.npy` per stored tensor in its
//! trained dtype, and a `<f8` `.npy` of the native logits.
//!
//! The example:
//! * refuses a settings file other than the one `execute.json` declares, and exports with different
//!   checkpoints or token rows;
//! * widens every tensor to binary64 (exact for every `f32`) and registers it in a
//!   [`TensorRegistry`], with its aliases and every discovered use site.
//!   `torch.nn.functional.linear` applies `x Wᵀ`, the identity orientation; an index or an addition
//!   reads the stored values; any other op is refused;
//! * checks that the export's edit factors are a layer-0 head's output block of the registered
//!   `W_O.0` (negated) and its unit columns, and builds each setting's [`ParameterEditRecord`] at
//!   `W_O.0#0`;
//! * wraps each (setting, sequence) of torch's executed rows as an [`ExecutedExperiment`] and refuses
//!   any block whose shape the registry does not declare;
//! * runs the stage receipt of every linear use the export covers:
//!   * `W_O.0#0` and `W_O.1#0`, from the mixed head outputs to the write;
//!   * `W_U#0`, from the last residual to the logits.
//!
//!   Each receipt runs the native kernel on torch's input rows: `apply_anchored_linear` on the rows
//!   an edit reaches, `native_linear` on the rest. [`compare_stage`] decides agreement against the
//!   derived bands. The tallies go to `REPORT_JSON`.
//!
//! The attention stages wait on the attention-only layer in `block` (mpd-block). A certified
//! refutation exits with an error after the report is written.

use gam_sae::inference::intervention_shard::ExperimentUnit;
use gam_sae::parameter_decomposition::apply::{
    FactorView, FactoredEdit, apply_anchored_linear, native_linear,
};
use gam_sae::parameter_decomposition::lift::{
    ExecutedExperiment, ForwardRoundoff, LiftError, ParameterExperiment, ParameterReadout, TensorId,
    TensorRegistry, TieOrientation, UseMap, UseSiteId,
};
use gam_sae::parameter_decomposition::occurrence::{EditScope, ParameterEditRecord, PositionScope};
use gam_sae::parameter_decomposition::receipts::{
    ExternalExecution, ReceiptRefusal, StageAgreement, affine_stage_band, compare_stage,
    factored_edit_stage_band,
};
use ndarray::{Array1, Array2, ArrayD, ArrayView1, ArrayView2, Axis, IxDyn, s};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

#[path = "support/npy_header.rs"]
mod npy_header;
use npy_header::{NpyFloat, parse_npy_float_header, parse_npy_header};

const USAGE: &str = "usage: mpd_induction_registry --run EXECUTE_DIR --settings SETTINGS_JSON --out REPORT_JSON";

/// The fields of `registry.json` this example reads; the others are ignored.
#[derive(Deserialize)]
struct Export {
    stage: String,
    checkpoint: u64,
    trained_dtype: String,
    sequences: usize,
    tokens: Vec<Vec<i64>>,
    config: Config,
    tensors: Vec<Tensor>,
    use_sites: Vec<Use>,
    files: BTreeMap<String, ExportFile>,
}

#[derive(Deserialize)]
struct Config {
    vocab: usize,
    seq_len: usize,
    d_model: usize,
    n_heads: usize,
    d_head: usize,
}

#[derive(Deserialize)]
struct Tensor {
    tensor_id: String,
    aliases: Vec<String>,
    shape: Vec<usize>,
}

#[derive(Deserialize)]
struct Use {
    tensor_id: String,
    ordinal: usize,
    transposed: bool,
    op: String,
}

#[derive(Deserialize)]
struct ExportFile {
    path: String,
}

/// The fields of `execute.json` this example reads.
#[derive(Deserialize)]
struct Execute {
    stage: String,
    harvest: String,
    declared: serde_json::Value,
    checkpoint: u64,
    sequences: usize,
    tokens: Vec<Vec<i64>>,
    edit: EditDeclaration,
    settings: Vec<Setting>,
    rows_per_setting: BTreeMap<String, usize>,
    external_dtype: String,
    external_device: String,
    tf32_matmul: bool,
    files: BTreeMap<String, ExportFile>,
}

#[derive(Deserialize)]
struct EditDeclaration {
    tensor_id: String,
    ordinal: usize,
    head: usize,
    left: String,
    right: String,
}

#[derive(Deserialize)]
struct Setting {
    name: String,
    positions: Option<Vec<usize>>,
    substituted: Vec<String>,
}

/// One (setting, use) of the report.
#[derive(Serialize)]
struct UseReport {
    setting: String,
    use_site: String,
    sequences: usize,
    agrees: usize,
    refutes: usize,
    unresolved: usize,
    worst_sequence: usize,
    worst_witness: (usize, usize),
    worst_discrepancy: f64,
    worst_band: f64,
    worst_ratio: f64,
}

#[derive(Serialize)]
struct Report {
    checkpoint: u64,
    teacher_fingerprint: String,
    external_dtype: String,
    external_device: String,
    tf32_matmul: bool,
    receipts: Vec<UseReport>,
}

/// The value following `name` on the command line.
fn flag<'a>(args: &'a [String], name: &str) -> Result<&'a str, String> {
    args.windows(2)
        .find(|pair| pair[0] == name)
        .map(|pair| pair[1].as_str())
        .ok_or_else(|| format!("missing {name}; {USAGE}"))
}

/// The bytes of one exported array, found by its file name inside `dir`, so an export can move.
fn read_array(
    files: &BTreeMap<String, ExportFile>,
    dir: &Path,
    array_id: &str,
) -> Result<(PathBuf, Vec<u8>), String> {
    let file = files
        .get(array_id)
        .ok_or_else(|| format!("the manifest names no file for {array_id}"))?;
    let name = Path::new(&file.path)
        .file_name()
        .ok_or_else(|| format!("{}: no file name", file.path))?;
    let path = dir.join(name);
    let bytes = std::fs::read(&path).map_err(|error| format!("read {}: {error}", path.display()))?;
    Ok((path, bytes))
}

/// The data bytes of `count` elements of `width` bytes, refusing a truncated or padded file.
fn payload<'a>(
    bytes: &'a [u8],
    data_off: usize,
    count: usize,
    width: usize,
    path: &Path,
) -> Result<&'a [u8], String> {
    let end = count
        .checked_mul(width)
        .and_then(|size| size.checked_add(data_off))
        .ok_or_else(|| format!("{}: size overflow", path.display()))?;
    if end != bytes.len() {
        return Err(format!(
            "{}: {} bytes, expected {end}",
            path.display(),
            bytes.len()
        ));
    }
    Ok(&bytes[data_off..end])
}

/// One stored tensor: a two-axis `<f4` array, widened to binary64.
fn stored_tensor(export: &Export, dir: &Path, tensor: &Tensor) -> Result<ArrayD<f64>, String> {
    let (path, bytes) = read_array(&export.files, dir, &tensor.tensor_id)?;
    let (rows, cols, width, is_f4, data_off) = parse_npy_header(&bytes, &path)?;
    if !is_f4 {
        return Err(format!("{}: expected the trained <f4 values", path.display()));
    }
    if [rows, cols][..] != tensor.shape[..] {
        return Err(format!(
            "{}: shape ({rows}, {cols}), but registry.json records {:?}",
            path.display(),
            tensor.shape
        ));
    }
    let values = payload(&bytes, data_off, rows * cols, width, &path)?
        .chunks_exact(width)
        .map(|chunk| f64::from(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])))
        .collect();
    ArrayD::from_shape_vec(IxDyn(&[rows, cols]), values)
        .map_err(|error| format!("{}: {error}", path.display()))
}

/// One executed array: a two-axis `<f8` array of the declared shape.
fn float64_array(
    files: &BTreeMap<String, ExportFile>,
    dir: &Path,
    array_id: &str,
    expected: (usize, usize),
) -> Result<Array2<f64>, String> {
    let (path, bytes) = read_array(files, dir, array_id)?;
    let header = parse_npy_float_header(&bytes, &path)?;
    if header.float != NpyFloat::F8 {
        return Err(format!(
            "{}: expected <f8 values from the binary64 executor",
            path.display()
        ));
    }
    let [rows, cols] = header.shape[..] else {
        return Err(format!(
            "{}: expected two axes, got {:?}",
            path.display(),
            header.shape
        ));
    };
    if (rows, cols) != expected {
        return Err(format!(
            "{}: shape ({rows}, {cols}), expected {expected:?}",
            path.display()
        ));
    }
    let values = payload(&bytes, header.data_off, rows * cols, header.float.bytes(), &path)?
        .chunks_exact(8)
        .map(|chunk| {
            f64::from_le_bytes([
                chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
            ])
        })
        .collect();
    Array2::from_shape_vec((rows, cols), values)
        .map_err(|error| format!("{}: {error}", path.display()))
}

/// What a discovered use site does with its tensor, for the ops this export produces.
fn use_map(site: &Use) -> Result<UseMap, String> {
    match (site.op.as_str(), site.transposed) {
        ("torch.nn.functional.linear", false) => Ok(UseMap::Linear(TieOrientation::Identity)),
        ("torch.Tensor.__getitem__" | "torch.Tensor.add", false) => Ok(UseMap::Stored),
        (op, transposed) => Err(format!(
            "use site {}#{}: no use map is declared for op {op} (transposed: {transposed})",
            site.tensor_id, site.ordinal
        )),
    }
}

/// The native stage of one linear use on the external input rows, and its receipt.
///
/// Rows the edit reaches run `x (W + L diag(s) Rᵀ)ᵀ` through `apply_anchored_linear`, under the
/// factored-edit band; every other row runs `x Wᵀ` through `native_linear`, under the affine band.
fn linear_receipt(
    execution: ExternalExecution<'_>,
    weight: ArrayView2<'_, f64>,
    input: ArrayView2<'_, f64>,
    external: ArrayView2<'_, f64>,
    edit: Option<(ArrayView2<'_, f64>, ArrayView1<'_, f64>, ArrayView2<'_, f64>, &PositionScope)>,
) -> Result<StageAgreement, String> {
    let native_rows = native_linear(weight, input).map_err(|error| LiftError::from(error).to_string())?;
    let mut native = (*native_rows).to_owned();
    let mut band = affine_stage_band(weight, None, input)
        .map_err(|mismatch| ReceiptRefusal::from(mismatch).to_string())?;
    if let Some((left, coefficients, right, scope)) = edit {
        let reached: Vec<usize> = (0..input.nrows()).filter(|&row| scope.reaches(row)).collect();
        if !reached.is_empty() {
            let reached_input = input.select(Axis(0), &reached);
            let factors = FactorView::new(left, right).map_err(|error| LiftError::from(error).to_string())?;
            let edited = apply_anchored_linear(weight, 1.0, factors, coefficients, reached_input.view())
                .map_err(|error| LiftError::from(error).to_string())?;
            let edited_band =
                factored_edit_stage_band(weight, left, coefficients, right, None, reached_input.view())
                    .map_err(|mismatch| ReceiptRefusal::from(mismatch).to_string())?;
            for (index, &row) in reached.iter().enumerate() {
                native.row_mut(row).assign(&edited.row(index));
                band.row_mut(row).assign(&edited_band.row(index));
            }
        }
    }
    compare_stage(execution, external, native.view(), band.view(), band.view())
        .map_err(|refusal| refusal.to_string())
}

/// Agreement counts of one (setting, use) over the export's sequences.
#[derive(Default)]
struct Tally {
    agrees: usize,
    refutes: usize,
    unresolved: usize,
    worst: Option<(usize, StageAgreement)>,
}

impl Tally {
    fn add(&mut self, sequence: usize, agreement: StageAgreement) {
        if agreement.agrees {
            self.agrees += 1;
        } else if agreement.refutes {
            self.refutes += 1;
        } else {
            self.unresolved += 1;
        }
        if self.worst.is_none_or(|(_, worst)| agreement.ratio > worst.ratio) {
            self.worst = Some((sequence, agreement));
        }
    }
}

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 7 {
        return Err(USAGE.to_string());
    }
    let execute_dir = PathBuf::from(flag(&args, "--run")?);
    let settings_path = PathBuf::from(flag(&args, "--settings")?);
    let report_path = PathBuf::from(flag(&args, "--out")?);
    let text = std::fs::read_to_string(execute_dir.join("execute.json"))
        .map_err(|error| format!("read execute.json in {}: {error}", execute_dir.display()))?;
    let execute: Execute =
        serde_json::from_str(&text).map_err(|error| format!("execute.json: {error}"))?;
    let text = std::fs::read_to_string(&settings_path)
        .map_err(|error| format!("read {}: {error}", settings_path.display()))?;
    let declared: serde_json::Value =
        serde_json::from_str(&text).map_err(|error| format!("{}: {error}", settings_path.display()))?;
    if declared != execute.declared {
        return Err(format!(
            "{} is not the settings execute.json declares",
            settings_path.display()
        ));
    }
    let registry_dir = PathBuf::from(&execute.harvest);
    let text = std::fs::read_to_string(registry_dir.join("registry.json"))
        .map_err(|error| format!("read registry.json in {}: {error}", registry_dir.display()))?;
    let export: Export =
        serde_json::from_str(&text).map_err(|error| format!("registry.json: {error}"))?;
    if export.stage != "registry" || execute.stage != "execute" {
        return Err(format!(
            "stages {:?} and {:?}; expected registry and execute",
            export.stage, execute.stage
        ));
    }
    if export.trained_dtype != "float32" {
        return Err(format!(
            "trained dtype {:?}; the tensors are read as <f4",
            export.trained_dtype
        ));
    }
    let config = &export.config;
    if export.checkpoint != execute.checkpoint
        || export.sequences != execute.sequences
        || export.tokens != execute.tokens
        || export.tokens.len() != export.sequences
        || export.tokens.iter().any(|row| row.len() != config.seq_len)
    {
        return Err("the registry and execute exports hold different checkpoints or token rows".to_string());
    }

    let mut registry = TensorRegistry::default();
    let mut values = BTreeMap::new();
    for tensor in &export.tensors {
        let stored = stored_tensor(&export, &registry_dir, tensor)?;
        registry
            .register_storage(TensorId(tensor.tensor_id.clone()), stored.view())
            .map_err(|error| error.to_string())?;
        values.insert(tensor.tensor_id.clone(), stored);
    }
    for tensor in &export.tensors {
        for alias in &tensor.aliases {
            registry
                .register_alias(TensorId(alias.clone()), TensorId(tensor.tensor_id.clone()))
                .map_err(|error| error.to_string())?;
        }
    }
    for site in &export.use_sites {
        let storage = TensorId(site.tensor_id.clone());
        registry
            .register_use_site(UseSiteId::read(&storage, site.ordinal), storage, use_map(site)?)
            .map_err(|error| error.to_string())?;
    }
    let fingerprint = registry.teacher_fingerprint();
    let native_logits = float64_array(
        &export.files,
        &registry_dir,
        "native_logits",
        (export.sequences * config.seq_len, config.vocab),
    )?;
    println!(
        "[load] checkpoint={} tensors={} use_sites={} teacher_fingerprint={:#018x} native_logits={}x{}",
        export.checkpoint,
        export.tensors.len(),
        export.use_sites.len(),
        fingerprint.0,
        native_logits.nrows(),
        native_logits.ncols()
    );

    let weight = |id: &str| -> Result<Array2<f64>, String> {
        values
            .get(id)
            .ok_or_else(|| format!("the registry export holds no {id}"))?
            .view()
            .into_dimensionality::<ndarray::Ix2>()
            .map(|view| view.to_owned())
            .map_err(|error| format!("{id}: {error}"))
    };
    let output = weight("W_O.0")?;
    let later_output = weight("W_O.1")?;
    let unembedding = weight("W_U")?;
    let width = config.n_heads * config.d_head;
    if execute.edit.tensor_id != "W_O.0" || output.dim() != (config.d_model, width) || execute.edit.head >= config.n_heads {
        return Err(format!(
            "execute.json edits {} head {}; expected W_O.0 of shape ({}, {width}) and a head below {}",
            execute.edit.tensor_id, execute.edit.head, config.d_model, config.n_heads
        ));
    }
    // The export's factors must be the registered head block, negated, and its unit columns.
    let block = execute.edit.head * config.d_head..(execute.edit.head + 1) * config.d_head;
    let left = float64_array(&execute.files, &execute_dir, &execute.edit.left, (config.d_model, config.d_head))?;
    let right = float64_array(&execute.files, &execute_dir, &execute.edit.right, (width, config.d_head))?;
    let expected_left = output.slice(s![.., block.clone()]).mapv(|value| -value);
    let expected_right = Array2::from_shape_fn((width, config.d_head), |(column, term)| {
        if column == block.start + term { 1.0 } else { 0.0 }
    });
    if left != expected_left || right != expected_right {
        return Err("the exported edit factors are not the registered W_O.0 head block and its unit columns".to_string());
    }
    let coefficients = Array1::<f64>::ones(config.d_head);
    let site = UseSiteId::read(&TensorId(execute.edit.tensor_id.clone()), execute.edit.ordinal);
    let later_site = UseSiteId::read(&TensorId("W_O.1".to_string()), 0);
    let unembedding_site = UseSiteId::read(&TensorId("W_U".to_string()), 0);
    let execution = ExternalExecution {
        dtype: &execute.external_dtype,
        device: &execute.external_device,
        tf32_matmul: execute.tf32_matmul,
    };
    println!(
        "[setting] edit={} head={} executor dtype={} device={} tf32_matmul={} settings={}",
        site.0,
        execute.edit.head,
        execute.external_dtype,
        execute.external_device,
        execute.tf32_matmul,
        execute.settings.len()
    );

    let stage_ids = ["mixed.0", "write.0", "mixed.1", "write.1", "resid_post.1", "logits"];
    let mut stages = BTreeMap::new();
    for id in stage_ids {
        let per_setting = *execute
            .rows_per_setting
            .get(id)
            .ok_or_else(|| format!("execute.json has no rows for {id}"))?;
        if per_setting != execute.sequences * config.seq_len {
            return Err(format!("{id}: {per_setting} rows per setting, expected sequences x T"));
        }
        let columns = if id == "logits" { config.vocab } else if id.starts_with("mixed") { width } else { config.d_model };
        let rows = float64_array(&execute.files, &execute_dir, id, (per_setting * execute.settings.len(), columns))?;
        stages.insert(id, rows);
    }
    // One (setting, sequence) of a stage: its T rows, in the setting and sequence order execute.json declares.
    let rows_of = |id: &str, setting: usize, sequence: usize| -> Array2<f64> {
        let start = (setting * execute.sequences + sequence) * config.seq_len;
        stages[id].slice(s![start..start + config.seq_len, ..]).to_owned()
    };

    let mut reports = Vec::new();
    for (index, setting) in execute.settings.iter().enumerate() {
        let scope = match &setting.positions {
            Some(positions) => Some(PositionScope::declared(positions.clone()).map_err(|error| error.to_string())?),
            None if setting.name == "all_on" => None,
            None => Some(PositionScope::every()),
        };
        let expected_substituted = if scope.is_some() { vec![site.0.clone()] } else { Vec::new() };
        if setting.substituted != expected_substituted {
            return Err(format!(
                "setting {} substituted {:?}, expected {:?}",
                setting.name, setting.substituted, expected_substituted
            ));
        }
        let record = match &scope {
            Some(scope) => {
                scope.check_within(config.seq_len).map_err(|error| error.to_string())?;
                let delta = FactoredEdit::new(left.clone(), right.clone())
                    .map_err(|error| LiftError::from(error).to_string())?;
                Some(
                    ParameterEditRecord::new(&registry, EditScope::UseSite(site.clone()), scope.clone(), delta)
                        .map_err(|error| error.to_string())?,
                )
            }
            None => None,
        };
        if let Some(record) = &record {
            if record.registry() != fingerprint {
                return Err(format!("setting {}: the record was checked against a different registry", setting.name));
            }
        }
        let mut tallies: [Tally; 3] = Default::default();
        for sequence in 0..execute.sequences {
            let experiment = ParameterExperiment {
                unit: ExperimentUnit {
                    group: 0,
                    sequence: sequence as i64,
                    length: config.seq_len,
                },
                edits: record.iter().cloned().collect(),
                readouts: vec![
                    ParameterReadout::UseSiteInput(site.clone()),
                    ParameterReadout::UseSiteOutput(site.clone()),
                    ParameterReadout::UseSiteInput(later_site.clone()),
                    ParameterReadout::UseSiteOutput(later_site.clone()),
                    ParameterReadout::UseSiteInput(unembedding_site.clone()),
                    ParameterReadout::Output {
                        positions: (0..config.seq_len).collect(),
                    },
                ],
            };
            let shapes = experiment
                .readout_shapes(&registry, config.vocab)
                .map_err(|error| error.to_string())?;
            let executed = ExecutedExperiment {
                readouts: stage_ids
                    .iter()
                    .map(|id| rows_of(*id, index, sequence))
                    .collect(),
                roundoff: ForwardRoundoff::Unresolved,
            };
            executed.check(&shapes).map_err(|error| error.to_string())?;
            let blocks = &executed.readouts;
            tallies[0].add(
                sequence,
                linear_receipt(
                    execution,
                    output.view(),
                    blocks[0].view(),
                    blocks[1].view(),
                    scope.as_ref().map(|scope| (left.view(), coefficients.view(), right.view(), scope)),
                )?,
            );
            tallies[1].add(
                sequence,
                linear_receipt(execution, later_output.view(), blocks[2].view(), blocks[3].view(), None)?,
            );
            tallies[2].add(
                sequence,
                linear_receipt(execution, unembedding.view(), blocks[4].view(), blocks[5].view(), None)?,
            );
        }
        for (use_site, tally) in [&site, &later_site, &unembedding_site].iter().zip(tallies) {
            let (worst_sequence, worst) = tally
                .worst
                .ok_or_else(|| format!("setting {}: the export holds no sequences", setting.name))?;
            println!(
                "[receipt] setting={} use={} sequences={} agrees={} refutes={} unresolved={} worst_ratio={:.3e} sequence={worst_sequence} witness={:?} discrepancy={:.3e} band={:.3e}",
                setting.name,
                use_site.0,
                execute.sequences,
                tally.agrees,
                tally.refutes,
                tally.unresolved,
                worst.ratio,
                worst.witness,
                worst.discrepancy,
                worst.band
            );
            reports.push(UseReport {
                setting: setting.name.clone(),
                use_site: use_site.0.clone(),
                sequences: execute.sequences,
                agrees: tally.agrees,
                refutes: tally.refutes,
                unresolved: tally.unresolved,
                worst_sequence,
                worst_witness: worst.witness,
                worst_discrepancy: worst.discrepancy,
                worst_band: worst.band,
                worst_ratio: worst.ratio,
            });
        }
    }
    let refuted: usize = reports.iter().map(|report| report.refutes).sum();
    let report = Report {
        checkpoint: export.checkpoint,
        teacher_fingerprint: format!("{:#018x}", fingerprint.0),
        external_dtype: execute.external_dtype.clone(),
        external_device: execute.external_device.clone(),
        tf32_matmul: execute.tf32_matmul,
        receipts: reports,
    };
    let text = serde_json::to_string_pretty(&report).map_err(|error| format!("report: {error}"))?;
    std::fs::write(&report_path, text).map_err(|error| format!("write {}: {error}", report_path.display()))?;
    if refuted > 0 {
        return Err(format!("{refuted} stage receipts certify a violation"));
    }
    Ok(())
}
