use ndarray::{Array1, ArrayView1};
use serde::{Deserialize, Serialize};
use std::ops::{Deref, DerefMut};

pub use gam_linalg::RidgePolicy;

pub use gam_spec::*;

/// Storage form of the ridge penalty matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RidgeMatrixForm {
    /// Ridge matrix is `delta * I`.
    ScaledIdentity,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InvalidStabilization {
    reason: String,
}

impl InvalidStabilization {
    fn new(reason: impl Into<String>) -> Self {
        Self {
            reason: reason.into(),
        }
    }

    pub fn reason(&self) -> &str {
        &self.reason
    }
}

impl std::fmt::Display for InvalidStabilization {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "invalid stabilization metadata: {}", self.reason)
    }
}

impl std::error::Error for InvalidStabilization {}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
struct RidgePassportWire {
    delta: f64,
    matrix_form: RidgeMatrixForm,
    policy: RidgePolicy,
}

/// Validated ridge metadata stamped into a fitted PIRLS result.
///
/// Construction and deserialization both reject non-finite or negative
/// magnitudes; fields are private so invalid state cannot be assembled with a
/// literal.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "RidgePassportWire", into = "RidgePassportWire")]
pub struct RidgePassport {
    delta: f64,
    matrix_form: RidgeMatrixForm,
    policy: RidgePolicy,
}

impl RidgePassport {
    pub fn scaled_identity(delta: f64, policy: RidgePolicy) -> Result<Self, InvalidStabilization> {
        if !(delta.is_finite() && delta >= 0.0) {
            return Err(InvalidStabilization::new(format!(
                "ridge delta must be finite and non-negative, got {delta:?}"
            )));
        }
        Ok(Self {
            delta: if delta == 0.0 { 0.0 } else { delta },
            matrix_form: RidgeMatrixForm::ScaledIdentity,
            policy,
        })
    }

    /// Exact zero-ridge passport; this fixed sentinel has no unchecked input.
    pub const fn zero(policy: RidgePolicy) -> Self {
        Self {
            delta: 0.0,
            matrix_form: RidgeMatrixForm::ScaledIdentity,
            policy,
        }
    }

    #[inline]
    pub const fn delta(self) -> f64 {
        self.delta
    }

    #[inline]
    pub const fn matrix_form(self) -> RidgeMatrixForm {
        self.matrix_form
    }

    #[inline]
    pub const fn policy(self) -> RidgePolicy {
        self.policy
    }

    #[inline]
    pub const fn penalty_logdet_ridge(self) -> f64 {
        if self.policy.accounts_for_objective() {
            self.delta
        } else {
            0.0
        }
    }

}

impl TryFrom<RidgePassportWire> for RidgePassport {
    type Error = InvalidStabilization;

    fn try_from(wire: RidgePassportWire) -> Result<Self, Self::Error> {
        let mut passport = Self::scaled_identity(wire.delta, wire.policy)?;
        passport.matrix_form = wire.matrix_form;
        Ok(passport)
    }
}

impl From<RidgePassport> for RidgePassportWire {
    fn from(passport: RidgePassport) -> Self {
        Self {
            delta: passport.delta,
            matrix_form: passport.matrix_form,
            policy: passport.policy,
        }
    }
}

/// Inertia of a symmetric matrix (count of positive / zero / negative
/// eigenvalues). Used by `bump_with_matrix` and other indefinite-aware
/// stabilization rules to drive δ from spectral evidence rather than a
/// condition-number heuristic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "InertiaWire", into = "InertiaWire")]
pub struct Inertia {
    positive: usize,
    zero: usize,
    negative: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
struct InertiaWire {
    positive: usize,
    zero: usize,
    negative: usize,
}

impl Inertia {
    pub fn new(
        positive: usize,
        zero: usize,
        negative: usize,
    ) -> Result<Self, InvalidStabilization> {
        let total = positive
            .checked_add(zero)
            .and_then(|value| value.checked_add(negative))
            .ok_or_else(|| InvalidStabilization::new("inertia count sum overflows usize"))?;
        if total == 0 {
            return Err(InvalidStabilization::new(
                "inertia must describe a non-empty matrix",
            ));
        }
        Ok(Self {
            positive,
            zero,
            negative,
        })
    }

    pub const fn positive(self) -> usize {
        self.positive
    }

    pub const fn zero(self) -> usize {
        self.zero
    }

    pub const fn negative(self) -> usize {
        self.negative
    }

    pub fn total(self) -> usize {
        self.positive + self.zero + self.negative
    }
}

impl TryFrom<InertiaWire> for Inertia {
    type Error = InvalidStabilization;

    fn try_from(wire: InertiaWire) -> Result<Self, Self::Error> {
        Self::new(wire.positive, wire.zero, wire.negative)
    }
}

impl From<Inertia> for InertiaWire {
    fn from(inertia: Inertia) -> Self {
        Self {
            positive: inertia.positive,
            zero: inertia.zero,
            negative: inertia.negative,
        }
    }
}

/// Generate a `#[repr(transparent)]` `Array1<f64>` newtype with the
/// `new`/`Deref`/`DerefMut`/`AsRef`/`From` boilerplate used by unconstrained
/// numeric vectors in this module.
macro_rules! array1_f64_newtype {
    ($name:ident) => {
        #[repr(transparent)]
        #[derive(Clone, Debug, PartialEq)]
        pub struct $name(pub Array1<f64>);

        impl $name {
            #[inline]
            pub fn new(values: Array1<f64>) -> Self {
                Self(values)
            }

            #[inline]
            pub fn zeros(len: usize) -> Self {
                Self(Array1::zeros(len))
            }
        }

        impl Deref for $name {
            type Target = Array1<f64>;
            #[inline]
            fn deref(&self) -> &Self::Target {
                &self.0
            }
        }

        impl DerefMut for $name {
            #[inline]
            fn deref_mut(&mut self) -> &mut Self::Target {
                &mut self.0
            }
        }

        impl AsRef<Array1<f64>> for $name {
            #[inline]
            fn as_ref(&self) -> &Array1<f64> {
                &self.0
            }
        }

        impl From<Array1<f64>> for $name {
            #[inline]
            fn from(values: Array1<f64>) -> Self {
                Self(values)
            }
        }

        impl From<$name> for Array1<f64> {
            #[inline]
            fn from(values: $name) -> Self {
                values.0
            }
        }
    };
}

array1_f64_newtype!(Coefficients);
array1_f64_newtype!(LinearPredictor);

/// Index into `TermCollectionSpec::smooth_terms` (and the parallel
/// `TermCollectionDesign::smooth.terms` slice produced from it).
///
/// This is **not** a penalty/ρ index, **not** a column index, and **not** a
/// coefficient-offset index. Keeping it behind a `#[repr(transparent)]`
/// newtype makes those confusables a compile error: a `SmoothTermIdx` cannot
/// be silently used to index `rho`, `beta`, or a design column.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct SmoothTermIdx(usize);

impl SmoothTermIdx {
    #[inline]
    pub const fn new(idx: usize) -> Self {
        Self(idx)
    }

    /// Sentinel used by transient builders that must allocate a coord config
    /// before the smooth term it references has been positioned in the spec.
    /// Every code path that constructs a sentinel must overwrite it before
    /// the value escapes the builder.
    #[inline]
    pub const fn placeholder() -> Self {
        Self(usize::MAX)
    }

    #[inline]
    pub const fn get(self) -> usize {
        self.0
    }

}

impl std::fmt::Display for SmoothTermIdx {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Index into the user-facing design matrix `data: Array2<f64>` — i.e. the
/// position of a covariate column in the raw input frame, *before* any
/// per-family basis expansion or intercept/parametric layout is applied.
///
/// Distinct from:
///   * [`SmoothTermIdx`] — position in `TermCollectionSpec::smooth_terms`.
///   * A coefficient-vector offset `β[i]` — spans the combined design after
///     expansion, which is much wider than the user-facing data matrix.
///
/// Keeping this as its own `#[repr(transparent)]` newtype rules out the easy
/// confusion of indexing the raw data frame with an expanded-column offset.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ColIdx(usize);

impl ColIdx {
    #[inline]
    pub const fn new(idx: usize) -> Self {
        Self(idx)
    }

    #[inline]
    pub const fn get(self) -> usize {
        self.0
    }
}

impl std::fmt::Display for ColIdx {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[repr(transparent)]
#[derive(Clone, Copy, Debug)]
pub struct LogSmoothingParamsView<'a>(ArrayView1<'a, f64>);

impl<'a> LogSmoothingParamsView<'a> {
    /// Borrow a smoothing vector only after every coordinate satisfies the
    /// exact shared logarithmic-strength contract.
    pub fn new(values: ArrayView1<'a, f64>) -> Result<Self, crate::IndexedLogStrengthDomainError> {
        crate::validate_log_strengths(values.iter().copied())?;
        Ok(Self(values))
    }

    /// Exact physical strengths for this already-validated vector.
    pub fn exact_exp(&self) -> Array1<f64> {
        // `new` established the private invariant; the borrow prevents the
        // source array from being mutated for this view's lifetime.
        self.0.mapv(f64::exp)
    }
}

impl<'a> Deref for LogSmoothingParamsView<'a> {
    type Target = ArrayView1<'a, f64>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[cfg(test)]
mod newtype_tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn smooth_term_idx_ordering() {
        let a = SmoothTermIdx::new(1);
        let b = SmoothTermIdx::new(2);
        assert!(a < b);
        assert_eq!(a, SmoothTermIdx::new(1));
    }

    #[test]
    fn coefficients_zeros_and_deref() {
        let c = Coefficients::zeros(3);
        assert_eq!(c.len(), 3);
        assert!(c.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn coefficients_from_array1() {
        let arr = array![1.0, 2.0, 3.0];
        let c = Coefficients::from(arr.clone());
        assert_eq!(*c, arr);
    }

    #[test]
    fn log_smoothing_params_view_is_validated_and_exponentiates_exactly() {
        let arr = array![crate::LOG_STRENGTH_MIN, 0.0, crate::LOG_STRENGTH_MAX];
        let rho = LogSmoothingParamsView::new(arr.view()).expect("closed domain");
        for (actual, expected) in rho.exact_exp().iter().zip(arr.iter()) {
            assert_eq!(actual.to_bits(), expected.exp().to_bits());
        }

        let invalid = array![0.0, crate::LOG_STRENGTH_MAX + 1.0];
        let error = LogSmoothingParamsView::new(invalid.view()).unwrap_err();
        assert_eq!(error.coordinate, 1);
        assert_eq!(error.value, crate::LOG_STRENGTH_MAX + 1.0);
    }

    #[test]
    fn linear_predictor_zeros_and_deref() {
        let lp = LinearPredictor::zeros(4);
        assert_eq!(lp.len(), 4);
        assert!(lp.iter().all(|&v| v == 0.0));
    }
}

#[cfg(test)]
mod ridge_policy_tests {
    use super::{RidgePassport, RidgePolicy};
    use serde_json::json;

    #[test]
    fn serde_cannot_bypass_passport_validation() {
        let negative = json!({
            "delta": -1.0,
            "matrix_form": "ScaledIdentity",
            "policy": "SolverOnly"
        });
        assert!(serde_json::from_value::<RidgePassport>(negative).is_err());

        let passport = RidgePassport::scaled_identity(
            2.5e-7,
            RidgePolicy::exact_full_objective(),
        )
        .expect("valid ridge");
        let roundtrip: RidgePassport =
            serde_json::from_value(serde_json::to_value(passport).expect("serialize passport"))
                .expect("deserialize validated passport");
        assert_eq!(roundtrip, passport);
    }

}
