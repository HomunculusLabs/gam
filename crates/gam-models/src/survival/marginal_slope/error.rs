//! Errors for survival marginal-slope fitting, with their
//! `Display`/`Error`/`From` conversions. Self-contained: no dependency on the
//! fitting machinery.

#[derive(Debug, Clone)]
pub enum SurvivalMarginalSlopeError {
    /// Spec, data, or runtime configuration failed input validation
    /// (finite/non-negative weights, derivative_guard > 0, supported
    /// base_link, frailty constraints, missing block state, etc.).
    InvalidInput { reason: String },
    /// Lengths, row/column counts, basis widths, or coefficient block
    /// sizes do not agree (covariance dim vs z, design rows vs n,
    /// basis/beta length mismatch, post-update beta length, time
    /// constraints A vs b, hessian_matvec dim mismatch, ...).
    IncompatibleDimensions { reason: String },
    /// A row's transformed time derivative or structural slack fell
    /// below `derivative_guard` (`qd1 < guard`), violating the
    /// monotonicity contract.
    MonotonicityViolation { reason: String },
    /// A numerical step produced a non-finite, non-positive, or
    /// internally inconsistent quantity that downstream code cannot
    /// consume (e.g. non-positive `D`, non-positive `chi1`, calibration
    /// derivative disagrees with the direct evaluation, transformed
    /// derivative not strictly positive).
    NumericalFailure { reason: String },
    /// A quadrature or numerical integration did not reach its tolerance.
    IntegrationFailed { reason: String },
    /// A root solve stopped with a residual above its tolerance, e.g. the
    /// per-row intercept solve. It was reported as `IntegrationFailed`, which
    /// sent a reader after quadrature that was never involved (#2937).
    RootSolveFailed { reason: String },
    /// The requested combination of options is not implemented (non-
    /// probit base link, flexible row calculus with K > 1, spatial psi
    /// for unsupported block roles, ...).
    UnsupportedConfiguration { reason: String },
}

impl_reason_error_boilerplate! {
    SurvivalMarginalSlopeError {
        InvalidInput,
        IncompatibleDimensions,
        MonotonicityViolation,
        NumericalFailure,
        IntegrationFailed,
        RootSolveFailed,
        UnsupportedConfiguration,
    }
}

impl SurvivalMarginalSlopeError {
    /// The fixed category of this failure (#2937). Exhaustive with no wildcard
    /// arm: a new variant is categorized by whoever adds it.
    #[must_use]
    pub fn failure_category(&self) -> gam_problem::FailureCategory {
        use gam_problem::FailureCategory;
        match self {
            Self::InvalidInput { .. } | Self::UnsupportedConfiguration { .. } => {
                FailureCategory::Input
            }
            Self::IncompatibleDimensions { .. } => FailureCategory::Invariant,
            Self::MonotonicityViolation { .. }
            | Self::NumericalFailure { .. }
            | Self::RootSolveFailed { .. } => FailureCategory::Numerical,
            Self::IntegrationFailed { .. } => FailureCategory::Integration,
        }
    }

    /// The `Enum::Variant` name a front end prints beside the message (#2937).
    #[must_use]
    pub fn variant_name(&self) -> &'static str {
        match self {
            Self::InvalidInput { .. } => "SurvivalMarginalSlopeError::InvalidInput",
            Self::IncompatibleDimensions { .. } => {
                "SurvivalMarginalSlopeError::IncompatibleDimensions"
            }
            Self::MonotonicityViolation { .. } => "SurvivalMarginalSlopeError::MonotonicityViolation",
            Self::NumericalFailure { .. } => "SurvivalMarginalSlopeError::NumericalFailure",
            Self::IntegrationFailed { .. } => "SurvivalMarginalSlopeError::IntegrationFailed",
            Self::RootSolveFailed { .. } => "SurvivalMarginalSlopeError::RootSolveFailed",
            Self::UnsupportedConfiguration { .. } => {
                "SurvivalMarginalSlopeError::UnsupportedConfiguration"
            }
        }
    }
}

impl From<String> for SurvivalMarginalSlopeError {
    /// Inbound conversion from helpers in this module (and adjacent
    /// families) that still surface `Result<_, String>`. The text is
    /// preserved verbatim; `InvalidInput` is the catch-all category for
    /// strings produced outside this module.
    fn from(reason: String) -> SurvivalMarginalSlopeError {
        SurvivalMarginalSlopeError::InvalidInput { reason }
    }
}
