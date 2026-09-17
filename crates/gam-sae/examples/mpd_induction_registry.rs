//! Load mpd-induction's S1 export into the native lift (#2951).
//!
//! `cargo run --release -p gam-sae --example mpd_induction_registry -- EXPORT_DIR HEAD`
//!
//! `EXPORT_DIR` is what `bench/mpd_induction_2951.py registry` writes:
//! * `registry.json`, the tensors and use sites of one unedited forward;
//! * one little-endian `<f4` `.npy` per stored tensor, in its trained dtype;
//! * a `<f8` `.npy` of the native logits, one row per (sequence, position).
//!
//! The example:
//! * widens every tensor to binary64 (exact for every `f32`) and registers it, with its aliases,
//!   in a [`TensorRegistry`];
//! * registers every discovered use site. `torch.nn.functional.linear` applies `x Wᵀ`, the
//!   identity orientation; an index or an addition reads the stored values; any other op is
//!   refused;
//! * builds the occurrence experiment's records: the output block of layer-0 head `HEAD` zeroed at
//!   use site `W_O.0#0`, once per single position `j = 1..T-1` and once at every position;
//! * prints the teacher fingerprint and each record's affected uses.
//!
//! Executing a record natively waits on the lift executor (mpd-lift) and the attention-only
//! layer in `block` (mpd-block), so nothing here scores an agreement.

use gam_sae::parameter_decomposition::apply::FactoredEdit;
use gam_sae::parameter_decomposition::lift::{
    LiftError, TensorId, TensorRegistry, TieOrientation, UseMap, UseSiteId,
};
use gam_sae::parameter_decomposition::occurrence::{EditScope, ParameterEditRecord, PositionScope};
use ndarray::{Array2, ArrayD, Ix2, IxDyn, s};
use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

#[path = "support/npy_header.rs"]
mod npy_header;
use npy_header::{NpyFloat, parse_npy_float_header, parse_npy_header};

const USAGE: &str = "usage: mpd_induction_registry EXPORT_DIR HEAD";

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

/// The bytes of one exported array, found by its file name inside `dir`, so the export can move.
fn read_array(export: &Export, dir: &Path, array_id: &str) -> Result<(PathBuf, Vec<u8>), String> {
    let file = export
        .files
        .get(array_id)
        .ok_or_else(|| format!("registry.json names no file for {array_id}"))?;
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
    let (path, bytes) = read_array(export, dir, &tensor.tensor_id)?;
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

/// The native logits: a two-axis `<f8` array of (sequence, position) rows over the vocabulary.
fn native_logits(export: &Export, dir: &Path) -> Result<Array2<f64>, String> {
    let (path, bytes) = read_array(export, dir, "native_logits")?;
    let header = parse_npy_float_header(&bytes, &path)?;
    if header.float != NpyFloat::F8 {
        return Err(format!(
            "{}: expected <f8 logits from the binary64 executor",
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
    let expected = (export.sequences * export.config.seq_len, export.config.vocab);
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

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        return Err(USAGE.to_string());
    }
    let dir = PathBuf::from(&args[1]);
    let head: usize = args[2]
        .parse()
        .map_err(|error| format!("HEAD {:?}: {error}; {USAGE}", args[2]))?;
    let text = std::fs::read_to_string(dir.join("registry.json"))
        .map_err(|error| format!("read registry.json in {}: {error}", dir.display()))?;
    let export: Export =
        serde_json::from_str(&text).map_err(|error| format!("registry.json: {error}"))?;
    if export.stage != "registry" {
        return Err(format!("registry.json is stage {:?}, not registry", export.stage));
    }
    if export.trained_dtype != "float32" {
        return Err(format!(
            "trained dtype {:?}; the tensors are read as <f4",
            export.trained_dtype
        ));
    }
    let config = &export.config;
    if export.tokens.len() != export.sequences
        || export.tokens.iter().any(|row| row.len() != config.seq_len)
    {
        return Err(format!(
            "registry.json holds {} token rows, expected {} of length {}",
            export.tokens.len(),
            export.sequences,
            config.seq_len
        ));
    }
    if head >= config.n_heads {
        return Err(format!("HEAD {head} is outside {} heads", config.n_heads));
    }

    let mut registry = TensorRegistry::default();
    let mut values = BTreeMap::new();
    for tensor in &export.tensors {
        let stored = stored_tensor(&export, &dir, tensor)?;
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
    let logits = native_logits(&export, &dir)?;
    println!(
        "checkpoint={} tensors={} use_sites={} teacher_fingerprint={:#018x} native_logits={}x{}",
        export.checkpoint,
        export.tensors.len(),
        export.use_sites.len(),
        fingerprint.0,
        logits.nrows(),
        logits.ncols()
    );

    let output = TensorId("W_O.0".to_string());
    let stored = values
        .get(&output.0)
        .ok_or_else(|| "the export holds no W_O.0".to_string())?;
    let width = config.n_heads * config.d_head;
    if stored.shape() != [config.d_model, width] {
        return Err(format!(
            "W_O.0 has shape {:?}, expected [{}, {width}]",
            stored.shape(),
            config.d_model
        ));
    }
    // The edit zeroes head HEAD's output block: left = -W_O.0[:, block], right = the block's unit columns.
    let block = head * config.d_head..(head + 1) * config.d_head;
    let left = stored
        .slice(s![.., block.clone()])
        .mapv(|value| -value)
        .into_dimensionality::<Ix2>()
        .map_err(|error| format!("W_O.0 block: {error}"))?;
    let right = Array2::from_shape_fn((width, config.d_head), |(column, term)| {
        if column == block.start + term { 1.0 } else { 0.0 }
    });
    let site = UseSiteId::read(&output, 0);
    let mut scopes = Vec::new();
    for position in 1..config.seq_len {
        let scope = PositionScope::declared(vec![position]).map_err(|error| error.to_string())?;
        scope
            .check_within(config.seq_len)
            .map_err(|error| error.to_string())?;
        scopes.push(scope);
    }
    scopes.push(PositionScope::every());
    for scope in scopes {
        let delta = FactoredEdit::new(left.clone(), right.clone())
            .map_err(|error| LiftError::from(error).to_string())?;
        let record =
            ParameterEditRecord::new(&registry, EditScope::UseSite(site.clone()), scope, delta)
                .map_err(|error| error.to_string())?;
        if record.registry() != fingerprint {
            return Err("a record was checked against a different registry".to_string());
        }
        let uses = record
            .scope()
            .affected_uses(&registry)
            .map_err(|error| error.to_string())?;
        println!(
            "record site={} positions={:?} rank={} affected_uses={:?}",
            site.0,
            record.positions().positions(),
            record.delta().term_count(),
            uses.iter().map(|used| used.0.as_str()).collect::<Vec<&str>>()
        );
    }
    Ok(())
}
