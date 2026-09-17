//! Saved event-history models (#2966): one versioned JSON envelope for every
//! event engine, written and read through the same Rust code by the CLI, pyffi
//! and gamfit.
//!
//! A document is `{"kind": ..., "version": ..., "model": ...}`. Loading probes
//! the kind and the version before it parses the model, and refuses any other
//! kind or version with a typed error; an older payload is never migrated.
//! JSON has no encoding for a non-finite float, so saving refuses one instead
//! of writing `null`. With serde_json's exact float round trip, save → reload
//! reproduces every f64 bit.

use crate::EventHistoryError;
use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::Value;
use std::path::Path;

/// Which event engine a saved model belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SavedModelKind {
    /// The joint latent-signature event model.
    Joint,
}

impl SavedModelKind {
    fn name(self) -> &'static str {
        match self {
            Self::Joint => "joint",
        }
    }
}

/// Why a saved model could not be written, or was refused.
#[derive(Debug, thiserror::Error)]
pub enum SavedModelError {
    #[error("the saved model is of kind {found:?}; this reader expects {expected:?}")]
    Kind {
        found: Option<String>,
        expected: SavedModelKind,
    },
    #[error(
        "the saved {kind:?} model has version {found:?}; this build reads only version {expected}"
    )]
    Version {
        kind: SavedModelKind,
        found: Option<u64>,
        expected: u64,
    },
    #[error("the saved model is not a model document: {reason}")]
    Malformed { reason: String },
    #[error("the saved model holds a state its law cannot: {reason}")]
    Inconsistent { reason: EventHistoryError },
    #[error("{path}: {reason}")]
    Io { path: String, reason: String },
}

#[derive(Serialize)]
struct Envelope<'m, T> {
    kind: SavedModelKind,
    version: u64,
    model: &'m T,
}

/// The saved document of a model.
pub(crate) fn saved_model_text<T: Serialize>(
    kind: SavedModelKind,
    version: u64,
    model: &T,
) -> Result<String, SavedModelError> {
    gam_problem::ensure_serialized_floats_are_finite(model).map_err(|found| {
        SavedModelError::Inconsistent {
            reason: EventHistoryError::NumericalFailure {
                reason: format!("a saved model cannot hold a non-finite float: {found}"),
            },
        }
    })?;
    serde_json::to_string_pretty(&Envelope {
        kind,
        version,
        model,
    })
    .map_err(|error| SavedModelError::Malformed {
        reason: error.to_string(),
    })
}

/// The model in a saved document of `kind` at `version`.
pub(crate) fn read_saved_model_text<T: DeserializeOwned>(
    text: &str,
    kind: SavedModelKind,
    version: u64,
) -> Result<T, SavedModelError> {
    let malformed = |error: serde_json::Error| SavedModelError::Malformed {
        reason: error.to_string(),
    };
    let mut document: Value = serde_json::from_str(text).map_err(malformed)?;
    let found_kind = document.get("kind").and_then(Value::as_str);
    if found_kind != Some(kind.name()) {
        return Err(SavedModelError::Kind {
            found: found_kind.map(str::to_string),
            expected: kind,
        });
    }
    let found = document.get("version").and_then(Value::as_u64);
    if found != Some(version) {
        return Err(SavedModelError::Version {
            kind,
            found,
            expected: version,
        });
    }
    let model = document
        .get_mut("model")
        .map(Value::take)
        .ok_or_else(|| SavedModelError::Malformed {
            reason: "the document has no model".to_string(),
        })?;
    serde_json::from_value(model).map_err(malformed)
}

/// Write a saved document atomically: to a sibling temporary file, renamed
/// over `path`.
pub(crate) fn write_saved_model(path: &Path, text: &str) -> Result<(), SavedModelError> {
    let io = |reason: String| SavedModelError::Io {
        path: path.display().to_string(),
        reason,
    };
    let name = path
        .file_name()
        .ok_or_else(|| io("not a file path".to_string()))?;
    let mut temporary = name.to_os_string();
    temporary.push(format!(".tmp{}", std::process::id()));
    let temporary = path.with_file_name(temporary);
    std::fs::write(&temporary, text).map_err(|error| io(error.to_string()))?;
    std::fs::rename(&temporary, path).map_err(|error| {
        drop(std::fs::remove_file(&temporary));
        io(error.to_string())
    })
}

/// The text of a saved document.
pub(crate) fn read_saved_model_file(path: &Path) -> Result<String, SavedModelError> {
    std::fs::read_to_string(path).map_err(|error| SavedModelError::Io {
        path: path.display().to_string(),
        reason: error.to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn envelopes_refuse_other_kinds_versions_and_non_finite_floats() {
        let values = vec![0.1_f64, 2.5e300];
        let text = saved_model_text(SavedModelKind::Joint, 3, &values).unwrap();
        let reloaded: Vec<f64> = read_saved_model_text(&text, SavedModelKind::Joint, 3).unwrap();
        assert_eq!(
            reloaded.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            values.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        assert!(matches!(
            read_saved_model_text::<Vec<f64>>(&text, SavedModelKind::Joint, 4),
            Err(SavedModelError::Version { found: Some(3), expected: 4, .. })
        ));
        assert!(text.contains("\"kind\": \"joint\""));
        assert!(matches!(
            read_saved_model_text::<Vec<f64>>(
                &text.replace("\"kind\": \"joint\"", "\"kind\": \"event_history\""),
                SavedModelKind::Joint,
                3
            ),
            Err(SavedModelError::Kind { .. })
        ));
        assert!(matches!(
            read_saved_model_text::<Vec<f64>>("{", SavedModelKind::Joint, 3),
            Err(SavedModelError::Malformed { .. })
        ));
        assert!(matches!(
            saved_model_text(SavedModelKind::Joint, 3, &vec![f64::NAN]),
            Err(SavedModelError::Inconsistent { .. })
        ));
        let path = std::env::temp_dir().join(format!("gam-saved-model-{}.json", std::process::id()));
        write_saved_model(&path, &text).unwrap();
        assert_eq!(read_saved_model_file(&path).unwrap(), text);
        std::fs::remove_file(&path).unwrap();
        assert!(matches!(
            read_saved_model_file(&path),
            Err(SavedModelError::Io { .. })
        ));
    }
}
