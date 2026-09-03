use super::*;

pub(crate) trait CliCauseCountResult {
    fn into_cli_result(self) -> Result<usize, String>;
}

impl CliCauseCountResult for usize {
    fn into_cli_result(self) -> Result<usize, String> {
        Ok(self)
    }
}

impl<E: ToString> CliCauseCountResult for Result<usize, E> {
    fn into_cli_result(self) -> Result<usize, String> {
        self.map_err(|err| err.to_string())
    }
}

pub(crate) type CliResult<T> = Result<T, CliError>;

#[derive(Debug, Error)]
pub(crate) enum CliError {
    #[error("{message}")]
    Message {
        message: String,
        advice: Option<String>,
    },
    #[error("{reason}")]
    ArgumentInvalid { reason: String },
    #[error("{reason}")]
    IncompatibleConfig { reason: String },
    #[error("{reason}")]
    FileWriteFailed { reason: String },
    #[error("{reason}")]
    Internal { reason: String },
}

impl CliError {
    pub(crate) fn advice(&self) -> Option<&str> {
        match self {
            Self::Message { advice, .. } => advice.as_deref(),
            Self::ArgumentInvalid { .. }
            | Self::IncompatibleConfig { .. }
            | Self::FileWriteFailed { .. }
            | Self::Internal { .. } => None,
        }
    }
}

impl From<String> for CliError {
    fn from(message: String) -> Self {
        // A bare string carries no typed identity and therefore no advice:
        // remediation is a property of the typed error that produced the
        // failure (`EstimationError::advice` and friends), never something
        // re-derived from the rendered text.
        Self::Message {
            message,
            advice: None,
        }
    }
}

impl From<CliError> for String {
    fn from(err: CliError) -> Self {
        err.to_string()
    }
}

// Cross-module `?` cascade: typed library errors flow into `CliError` with
// the advice their own type declares, so the `help:` line the CLI prints is
// the same remediation the Python exception carries.

impl From<gam::inference::formula_dsl::FormulaDslError> for CliError {
    fn from(err: gam::inference::formula_dsl::FormulaDslError) -> Self {
        // Every formula-DSL failure is, from the CLI's point of view, an
        // argument-validation failure: the user-supplied formula string did
        // not parse / type-check / use a supported identifier.
        Self::ArgumentInvalid {
            reason: err.to_string(),
        }
    }
}

impl From<gam::data::DataError> for CliError {
    fn from(err: gam::data::DataError) -> Self {
        Self::Message {
            message: err.to_string(),
            advice: err.advice(),
        }
    }
}

impl From<WorkflowError> for CliError {
    fn from(err: WorkflowError) -> Self {
        Self::Message {
            message: err.to_string(),
            advice: err.advice(),
        }
    }
}

impl From<gam::estimate::EstimationError> for CliError {
    fn from(err: gam::estimate::EstimationError) -> Self {
        Self::Message {
            message: err.to_string(),
            advice: err.advice(),
        }
    }
}
