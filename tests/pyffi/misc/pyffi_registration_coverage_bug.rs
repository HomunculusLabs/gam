use std::collections::BTreeSet;
use std::fs;
use std::path::PathBuf;

/// Every `#[pyfunction]` exported by the PyO3 boundary must be registered in the
/// `#[pymodule]` body, otherwise the symbol compiles but is unreachable from
/// Python.
///
/// Registration has two forms. Functions in the `include!`d fragments are listed
/// one by one in `fn rust_extension`. A concern module (`event_history_ffi`,
/// `inference_instruments`) owns `fn register(module)`, and the pymodule body
/// calls it with one `<module>::register(module)?;` line. So the registered set
/// is the `wrap_pyfunction!` sites inside the pymodule body plus those inside the
/// body of every `register` function that body calls, and the definitions are
/// every `#[pyfunction]` in every source file of the crate. Scanning only the
/// `include!` fragments left every concern module's functions unchecked.
#[test]
fn pyffi_every_pyfunction_is_registered_once() {
    let sources = pyffi_sources();
    let scan = scan_registrations(&sources).unwrap_or_else(|e| panic!("{e}"));
    // Positive controls on the real tree: a fragment function, and a function of
    // a concern module registered through its own `register`, must each be seen
    // both as a definition and as a registration.
    for known in ["intervention_calibration_plan", "fit_event_history"] {
        assert!(
            scan.definitions.contains(known),
            "the scan missed the #[pyfunction] definition of {known}"
        );
        assert!(
            scan.registered.contains(known),
            "the scan missed the registration of {known}"
        );
    }
    let missing: Vec<&String> = scan.definitions.difference(&scan.registered).collect();
    assert!(
        missing.is_empty(),
        "unregistered #[pyfunction] exports: {missing:?}"
    );
}

/// Each way a function can be unreachable must be reported, and a registration
/// outside the pymodule body must not count.
#[test]
fn registration_scan_reports_every_unreachable_pyfunction() {
    let call = "    crate::concern_ffi::register(module)?;";

    // Clean: a fragment function and a concern-module function behind a
    // multi-line `#[pyo3(signature = ...)]` attribute, both registered.
    assert_eq!(
        unregistered(&[
            synthetic_pymodule(call, ""),
            source("concern_ffi", CONCERN_MODULE)
        ]),
        Ok(Vec::new())
    );

    // A concern-module function that its `register` omits.
    let with_forgotten = format!("{CONCERN_MODULE}\n#[pyfunction]\nfn forgotten() {{}}\n");
    assert_eq!(
        unregistered(&[
            synthetic_pymodule(call, ""),
            source("concern_ffi", &with_forgotten)
        ]),
        Ok(vec!["forgotten".to_string()])
    );

    // A concern module whose `register` the pymodule body never calls.
    assert_eq!(
        unregistered(&[
            synthetic_pymodule("", ""),
            source("concern_ffi", CONCERN_MODULE)
        ]),
        Ok(vec!["wired".to_string()])
    );

    // A `wrap_pyfunction!` in a function after the pymodule body does not
    // register anything. This also fails if a `{` inside a string or a `}` inside
    // a comment moves the end of the body.
    let stray = "#[pyfunction]\nfn stray() {}\n\nfn after(module: &Bound<'_, PyModule>) -> PyResult<()> {\n    module.add_function(wrap_pyfunction!(stray, module)?)\n}\n";
    assert_eq!(
        unregistered(&[
            synthetic_pymodule(call, stray),
            source("concern_ffi", CONCERN_MODULE)
        ]),
        Ok(vec!["stray".to_string()])
    );

    // A `register` call that names no source file is refused, not skipped.
    assert!(unregistered(&[synthetic_pymodule(call, "")]).is_err());
}

/// One crate source file: its stem, which is the module name a `register` call
/// names, and its text.
struct Source {
    stem: String,
    text: String,
}

/// The functions defined under `#[pyfunction]` and the functions reachable from
/// the pymodule body through `wrap_pyfunction!`.
struct RegistrationScan {
    definitions: BTreeSet<String>,
    registered: BTreeSet<String>,
}

/// Every `.rs` file under `crates/gam-pyffi/src`, walked recursively.
fn pyffi_sources() -> Vec<Source> {
    let mut pending = vec![PathBuf::from("crates/gam-pyffi/src")];
    let mut sources = Vec::new();
    while let Some(dir) = pending.pop() {
        let entries =
            fs::read_dir(&dir).unwrap_or_else(|e| panic!("read_dir {}: {e}", dir.display()));
        for entry in entries {
            let path = entry
                .unwrap_or_else(|e| panic!("read_dir entry in {}: {e}", dir.display()))
                .path();
            if path.is_dir() {
                pending.push(path);
            } else if path.extension().is_some_and(|ext| ext == "rs") {
                let text = fs::read_to_string(&path)
                    .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
                let stem = path
                    .file_stem()
                    .and_then(|stem| stem.to_str())
                    .unwrap_or_else(|| panic!("non-UTF-8 file name {}", path.display()))
                    .to_string();
                sources.push(Source { stem, text });
            }
        }
    }
    assert!(
        !sources.is_empty(),
        "no source files found under crates/gam-pyffi/src"
    );
    sources
}

fn scan_registrations(sources: &[Source]) -> Result<RegistrationScan, String> {
    let mut definitions = BTreeSet::new();
    for source in sources {
        definitions.extend(pyfunction_definitions(&source.text)?);
    }
    if definitions.is_empty() {
        return Err("no #[pyfunction] definitions found; the boundary layout changed".to_string());
    }

    let pymodules: Vec<&Source> = sources
        .iter()
        .filter(|source| source.text.contains("fn rust_extension("))
        .collect();
    let [pymodule] = pymodules.as_slice() else {
        return Err(format!(
            "expected exactly one `fn rust_extension` pymodule body, found {}",
            pymodules.len()
        ));
    };
    let body = item_body(&pymodule.text, "fn rust_extension(")?;
    let mut registered = wrapped_functions(body);
    for module in register_calls(body) {
        let owners: Vec<&Source> = sources
            .iter()
            .filter(|source| source.stem == module)
            .collect();
        let [owner] = owners.as_slice() else {
            return Err(format!(
                "the pymodule body calls `{module}::register`, but {} source files are named {module}.rs",
                owners.len()
            ));
        };
        registered.extend(wrapped_functions(item_body(&owner.text, "fn register(")?));
    }
    Ok(RegistrationScan {
        definitions,
        registered,
    })
}

fn unregistered(sources: &[Source]) -> Result<Vec<String>, String> {
    let scan = scan_registrations(sources)?;
    Ok(scan
        .definitions
        .difference(&scan.registered)
        .cloned()
        .collect())
}

/// The name of the `fn` item under each `#[pyfunction]` attribute. The item line
/// is the first line after the attribute that starts with `fn` once a visibility
/// prefix is stripped, so further attributes (including a multi-line
/// `#[pyo3(signature = (...))]`) and doc comments in between do not hide it.
fn pyfunction_definitions(text: &str) -> Result<Vec<String>, String> {
    let mut names = Vec::new();
    let mut lines = text.lines();
    while let Some(line) = lines.next() {
        if !line.trim_start().starts_with("#[pyfunction") {
            continue;
        }
        let item = lines
            .by_ref()
            .map(str::trim_start)
            .map(|item| {
                item.strip_prefix("pub(crate) ")
                    .or_else(|| item.strip_prefix("pub(super) "))
                    .or_else(|| item.strip_prefix("pub "))
                    .unwrap_or(item)
            })
            .find_map(|item| item.strip_prefix("fn "))
            .ok_or_else(|| format!("a #[pyfunction] attribute has no fn item after it: {line:?}"))?;
        names.push(
            item.chars()
                .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                .collect(),
        );
    }
    Ok(names)
}

/// The text inside the braces of the first item whose header starts with
/// `header`. Braces are matched past string literals and line comments, so a `{`
/// in a message or a `}` in a comment does not move the end of the body.
fn item_body<'a>(text: &'a str, header: &str) -> Result<&'a str, String> {
    let start = text
        .find(header)
        .ok_or_else(|| format!("no `{header}` item found"))?;
    let open = start
        + text[start..]
            .find('{')
            .ok_or_else(|| format!("`{header}` has no body"))?;
    let bytes = text.as_bytes();
    let mut depth = 0usize;
    let mut in_string = false;
    let mut i = open;
    while i < bytes.len() {
        let byte = bytes[i];
        if in_string {
            if byte == b'\\' {
                i += 1;
            } else if byte == b'"' {
                in_string = false;
            }
        } else if byte == b'"' {
            in_string = true;
        } else if byte == b'/' && bytes.get(i + 1) == Some(&b'/') {
            while i < bytes.len() && bytes[i] != b'\n' {
                i += 1;
            }
        } else if byte == b'{' {
            depth += 1;
        } else if byte == b'}' {
            depth -= 1;
            if depth == 0 {
                return Ok(&text[open + 1..i]);
            }
        }
        i += 1;
    }
    Err(format!("`{header}` body is not closed"))
}

/// The registered function of each `wrap_pyfunction!(path, module)`: the last
/// segment of the path, so `crate::io::x` registers `x`.
fn wrapped_functions(body: &str) -> BTreeSet<String> {
    body.split("wrap_pyfunction!(")
        .skip(1)
        .filter_map(|rest| {
            let path: String = rest
                .trim_start()
                .chars()
                .take_while(|c| c.is_ascii_alphanumeric() || *c == '_' || *c == ':')
                .collect();
            path.rsplit("::")
                .next()
                .filter(|name| !name.is_empty())
                .map(str::to_string)
        })
        .collect()
}

/// The module named by each `<path>::<module>::register(` call in a body.
fn register_calls(body: &str) -> Vec<String> {
    let needle = "::register(";
    let mut modules = Vec::new();
    let mut from = 0usize;
    while let Some(off) = body[from..].find(needle) {
        let at = from + off;
        let head = &body[..at];
        let start = head
            .rfind(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
            .map_or(0, |i| i + 1);
        modules.push(head[start..].to_string());
        from = at + needle.len();
    }
    modules
}

fn source(stem: &str, text: &str) -> Source {
    Source {
        stem: stem.to_string(),
        text: text.to_string(),
    }
}

/// A pymodule fragment that defines and registers `listed`, runs `extra_body`
/// inside the pymodule body, and appends `after_body` after it.
fn synthetic_pymodule(extra_body: &str, after_body: &str) -> Source {
    source(
        "geometry_ffi",
        &format!(
            r#"#[pyfunction]
fn listed() {{}}

#[pymodule(name = "_rust", gil_used = false)]
fn rust_extension(module: &Bound<'_, PyModule>) -> PyResult<()> {{
    module.add("__doc__", "a {{ inside a string")?; // and a }} inside a comment
    module.add_function(wrap_pyfunction!(
        listed,
        module
    )?)?;
{extra_body}
    Ok(())
}}
{after_body}"#
        ),
    )
}

const CONCERN_MODULE: &str = r#"#[pyfunction]
#[pyo3(signature = (
    x,
    y = None
))]
pub(crate) fn wired(x: f64, y: Option<f64>) -> f64 {
    x
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(crate::concern_ffi::wired, module)?)?;
    Ok(())
}
"#;
