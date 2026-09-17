//! `bench_fixtures FIXTURE OUT_DIR [KEY=VALUE]...`
//!
//! The benchmark suite's source of seeded panels ([`gam_test_support::synthetic`])
//! and of its cross-validation folds and per-fold z-scores
//! ([`gam_test_support::bench_fixtures`]). Numeric outputs are written into
//! `OUT_DIR` as `.npy` (float64, or uint64 for row indices) and string columns as
//! `.txt`, one value per line. Array inputs are C-order `.npy` paths, scalar inputs
//! are decimal text, and a key the fixture does not read is refused.
//! `bench/_bench_fixtures.py` is its caller.

use gam_test_support::{bench_fixtures, synthetic};
use ndarray::{Array2, ArrayView1, ArrayView2};
use npyz::{NpyFile, Order, WriterBuilder};
use std::collections::BTreeMap;
use std::fmt::Display;
use std::io::{BufReader, BufWriter};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::str::FromStr;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("bench_fixtures: {message}");
            ExitCode::FAILURE
        }
    }
}

/// The `KEY=VALUE` arguments. Every key is read at most once, and [`Params::finish`]
/// refuses the ones nothing read, so a misspelled key cannot fall back to a default.
struct Params(BTreeMap<String, String>);

impl Params {
    fn parse(pairs: &[String]) -> Result<Self, String> {
        let mut map = BTreeMap::new();
        for pair in pairs {
            let Some((key, value)) = pair.split_once('=') else {
                return Err(format!("argument '{pair}' is not KEY=VALUE"));
            };
            if map.insert(key.to_string(), value.to_string()).is_some() {
                return Err(format!("key '{key}' is given twice"));
            }
        }
        Ok(Self(map))
    }

    fn text(&mut self, key: &str) -> Result<String, String> {
        self.0
            .remove(key)
            .ok_or_else(|| format!("missing key '{key}'"))
    }

    fn value<T: FromStr>(&mut self, key: &str) -> Result<T, String>
    where
        T::Err: Display,
    {
        let raw = self.text(key)?;
        raw.parse::<T>()
            .map_err(|err| format!("key '{key}' = '{raw}': {err}"))
    }

    fn optional_value<T: FromStr>(&mut self, key: &str) -> Result<Option<T>, String>
    where
        T::Err: Display,
    {
        if self.0.contains_key(key) {
            self.value(key).map(Some)
        } else {
            Ok(None)
        }
    }

    fn path(&mut self, key: &str) -> Result<PathBuf, String> {
        self.text(key).map(PathBuf::from)
    }

    fn finish(self) -> Result<(), String> {
        match self.0.keys().next() {
            Some(key) => Err(format!("unused key '{key}'")),
            None => Ok(()),
        }
    }
}

fn run(args: &[String]) -> Result<(), String> {
    let [fixture, out_dir, pairs @ ..] = args else {
        return Err("usage: bench_fixtures FIXTURE OUT_DIR [KEY=VALUE]...".to_string());
    };
    let out = Path::new(out_dir);
    std::fs::create_dir_all(out).map_err(|err| format!("create {}: {err}", out.display()))?;
    let mut params = Params::parse(pairs)?;
    match fixture.as_str() {
        "cv-folds" => {
            let y = read_vector::<f64>(&params.path("y")?)?;
            let folds = bench_fixtures::cv_folds(
                &y,
                params.value("n_splits")?,
                params.value("seed")?,
                params.value("stratified")?,
            )?;
            for (k, (train, test)) in folds.iter().enumerate() {
                write_indices(&out.join(format!("train_{k}.npy")), train)?;
                write_indices(&out.join(format!("test_{k}.npy")), test)?;
            }
        }
        "zscore" => {
            let train = read_matrix(&params.path("train")?)?;
            let test = read_matrix(&params.path("test")?)?;
            let (train, test) = bench_fixtures::zscore_train_test(train.view(), test.view())?;
            write_matrix(&out.join("train.npy"), train.view())?;
            write_matrix(&out.join("test.npy"), test.view())?;
        }
        "binomial-columns" => {
            let (x, y) = synthetic::binomial_columns(
                params.value("n")?,
                params.value("p")?,
                params.value("seed")?,
            );
            write_matrix(&out.join("x.npy"), x.view())?;
            write_vector(&out.join("y.npy"), y.view())?;
        }
        "geo-disease-columns" => {
            let (x, y) = synthetic::geo_disease_columns(params.value("n")?, params.value("seed")?);
            write_matrix(&out.join("x.npy"), x.view())?;
            write_vector(&out.join("y.npy"), y.view())?;
        }
        "continuous-order-columns" => {
            let (x, y) = synthetic::continuous_order_columns(
                &params.text("mode")?,
                params.value("n")?,
                params.value("seed")?,
                params.optional_value("true_nu")?,
                params.optional_value("true_kappa2")?,
            )?;
            write_vector(&out.join("x.npy"), x.view())?;
            write_vector(&out.join("y.npy"), y.view())?;
        }
        "thread3-admixture-cliff-columns" => {
            let (x, y) = synthetic::thread3_admixture_cliff_columns(
                params.value("n")?,
                params.value("seed")?,
            );
            write_matrix(&out.join("x.npy"), x.view())?;
            write_vector(&out.join("y.npy"), y.view())?;
        }
        "geo-disease-eas-columns" => {
            let (x, y) = synthetic::geo_disease_eas_columns(
                params.value("n")?,
                params.value("seed")?,
                params.value("n_pcs")?,
            );
            write_matrix(&out.join("x.npy"), x.view())?;
            write_vector(&out.join("y.npy"), y.view())?;
        }
        "papuan-oce-columns" => {
            let (x, y) = synthetic::papuan_oce_columns(
                params.value("n")?,
                params.value("seed")?,
                params.value("n_pcs")?,
            );
            write_matrix(&out.join("x.npy"), x.view())?;
            write_vector(&out.join("y.npy"), y.view())?;
        }
        "hgdp-pc-panel" => {
            let panel = synthetic::hgdp_pc_panel(params.value("seed")?);
            write_matrix(&out.join("pc.npy"), panel.pc.view())?;
            write_vector(&out.join("latitude.npy"), ArrayView1::from(&panel.latitudes))?;
            write_vector(&out.join("longitude.npy"), ArrayView1::from(&panel.longitudes))?;
            write_lines(&out.join("sample_id.txt"), &panel.sample_ids)?;
            write_lines(&out.join("superpopulation.txt"), &panel.superpopulations)?;
            write_lines(&out.join("subpopulation.txt"), &panel.subpopulations)?;
        }
        "geo-subpop-response" => {
            let codes = read_codes(&params.path("subpop_codes")?)?;
            let y = synthetic::geo_subpop_response(
                &codes,
                params.value("seed")?,
                params.value("prevalence_min")?,
                params.value("prevalence_max")?,
                params.value("noise_scale_min")?,
                params.value("noise_scale_max")?,
                params.value("random_scale")?,
            );
            write_vector(&out.join("y.npy"), y.view())?;
        }
        "geo-latlon-response" => {
            let mode = params.text("mode")?;
            let codes = read_codes(&params.path("superpop_codes")?)?;
            let latitudes = read_vector::<f64>(&params.path("latitudes")?)?;
            let longitudes = read_vector::<f64>(&params.path("longitudes")?)?;
            let y = synthetic::geo_latlon_response(
                &mode,
                &codes,
                &latitudes,
                &longitudes,
                params.value("seed")?,
                params.value("prevalence_min")?,
                params.value("prevalence_max")?,
            )?;
            write_vector(&out.join("y.npy"), y.view())?;
        }
        "cliff-gradient-magnitude" => {
            let points = read_matrix(&params.path("points")?)?;
            let coefficients = read_vector::<f64>(&params.path("coefficients")?)?;
            // No `magnitude.npy` means the cliff has no usable geometry here.
            if let Some(magnitude) = synthetic::cliff_gradient_magnitude(
                points.view(),
                &coefficients,
                params.value("jump")?,
                params.value("sharpness")?,
            ) {
                write_vector(&out.join("magnitude.npy"), magnitude.view())?;
            }
        }
        other => return Err(format!("unknown fixture '{other}'")),
    }
    params.finish()
}

fn read_npy<T: npyz::Deserialize>(path: &Path) -> Result<(Vec<u64>, Vec<T>), String> {
    let file = std::fs::File::open(path).map_err(|err| format!("open {}: {err}", path.display()))?;
    let npy = NpyFile::new(BufReader::new(file))
        .map_err(|err| format!("read NPY header {}: {err}", path.display()))?;
    if let Order::Fortran = npy.order() {
        return Err(format!(
            "{} is Fortran-ordered; pass a C-order array",
            path.display()
        ));
    }
    let shape = npy.shape().to_vec();
    match npy.try_data::<T>() {
        Ok(reader) => reader
            .collect::<std::io::Result<Vec<T>>>()
            .map(|values| (shape, values))
            .map_err(|err| format!("read {}: {err}", path.display())),
        Err(npy) => Err(format!(
            "{} has dtype {}, which this input does not take",
            path.display(),
            npy.dtype().descr()
        )),
    }
}

fn dimension(path: &Path, extent: u64) -> Result<usize, String> {
    usize::try_from(extent)
        .map_err(|_| format!("{} extent {extent} exceeds this platform", path.display()))
}

fn read_vector<T: npyz::Deserialize>(path: &Path) -> Result<Vec<T>, String> {
    let (shape, values) = read_npy::<T>(path)?;
    if shape.len() != 1 {
        return Err(format!(
            "{} must be 1-D; it has {} axes",
            path.display(),
            shape.len()
        ));
    }
    Ok(values)
}

fn read_matrix(path: &Path) -> Result<Array2<f64>, String> {
    let (shape, values) = read_npy::<f64>(path)?;
    let [rows, cols] = shape.as_slice() else {
        return Err(format!(
            "{} must be 2-D; it has {} axes",
            path.display(),
            shape.len()
        ));
    };
    Array2::from_shape_vec((dimension(path, *rows)?, dimension(path, *cols)?), values)
        .map_err(|err| format!("{} has an invalid shape: {err}", path.display()))
}

fn read_codes(path: &Path) -> Result<Vec<usize>, String> {
    read_vector::<i64>(path)?
        .into_iter()
        .map(|code| {
            usize::try_from(code)
                .map_err(|_| format!("{} holds the negative code {code}", path.display()))
        })
        .collect()
}

fn write_npy<T: npyz::AutoSerialize>(
    path: &Path,
    shape: &[u64],
    values: impl IntoIterator<Item = T>,
) -> Result<(), String> {
    let file =
        std::fs::File::create(path).map_err(|err| format!("create {}: {err}", path.display()))?;
    let mut writer = npyz::WriteOptions::new()
        .default_dtype()
        .shape(shape)
        .writer(BufWriter::new(file))
        .begin_nd()
        .map_err(|err| format!("begin {}: {err}", path.display()))?;
    writer
        .extend(values)
        .map_err(|err| format!("write {}: {err}", path.display()))?;
    writer
        .finish()
        .map_err(|err| format!("finish {}: {err}", path.display()))
}

fn write_matrix(path: &Path, matrix: ArrayView2<'_, f64>) -> Result<(), String> {
    write_npy(
        path,
        &[matrix.nrows() as u64, matrix.ncols() as u64],
        matrix.iter().copied(),
    )
}

fn write_vector(path: &Path, vector: ArrayView1<'_, f64>) -> Result<(), String> {
    write_npy(path, &[vector.len() as u64], vector.iter().copied())
}

fn write_indices(path: &Path, indices: &[usize]) -> Result<(), String> {
    write_npy(
        path,
        &[indices.len() as u64],
        indices.iter().map(|&index| index as u64),
    )
}

fn write_lines(path: &Path, lines: &[String]) -> Result<(), String> {
    let mut text = lines.join("\n");
    if !lines.is_empty() {
        text.push('\n');
    }
    std::fs::write(path, text).map_err(|err| format!("write {}: {err}", path.display()))
}
