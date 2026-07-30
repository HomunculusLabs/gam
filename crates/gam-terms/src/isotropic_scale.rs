//! A scalar coordinate scale for isotropic Euclidean smooths.
//!
//! Isotropic kernels admit one uniform change of coordinate units.  Encoding
//! that scale as a vector made anisotropic states representable in the frozen
//! model even though the kernel and its scale contract require one value in
//! every direction.  `IsotropicScale` makes the geometric invariant explicit:
//! anisotropy belongs to the separate ARD parameters, never to this frame.

use ndarray::Array2;
use serde::{Deserialize, Serialize};
use std::fmt;

/// A coordinate-valued length in the user's ORIGINAL covariate units.
///
/// A basis that auto-standardizes its Euclidean input divides the coordinates
/// by an [`IsotropicScale`] before building anything, so its `centers` — and
/// every radius the kernel evaluates — live in the standardized frame while
/// the range the user asked for lives here.  Keeping the two frames in
/// distinct types is what stops a consumer from pairing them: the kernel
/// range and the radii it is compared against must agree, and getting that
/// wrong is a silent O(1) relative error in the kernel bandwidth that no
/// shape or count check can see.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct OriginalUnits(f64);

/// A coordinate-valued length in the STANDARDIZED frame, i.e. the frame the
/// auto-standardized `centers` and the kernel radii already live in.
///
/// This is the frame every radial kernel evaluation must be denominated in.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct StandardizedUnits(f64);

impl OriginalUnits {
    pub const fn new(value: f64) -> Self {
        Self(value)
    }

    /// The scalar, named for its frame so a bare extraction that feeds
    /// standardized-frame math reads wrong at the call site.
    pub const fn original_value(self) -> f64 {
        self.0
    }
}

impl StandardizedUnits {
    pub const fn new(value: f64) -> Self {
        Self(value)
    }

    /// The scalar, named for its frame; see [`OriginalUnits::original_value`].
    pub const fn standardized_value(self) -> f64 {
        self.0
    }
}

impl fmt::Display for OriginalUnits {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, formatter)
    }
}

impl fmt::Display for StandardizedUnits {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, formatter)
    }
}

/// A positive, finite scalar whose reciprocal is also representable.
///
/// The field is private so construction, deserialization, and every frozen
/// replay path enforce the same invariant.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "f64", into = "f64")]
pub struct IsotropicScale(f64);

impl IsotropicScale {
    pub const ONE: Self = Self(1.0);

    pub fn new(value: f64) -> Result<Self, IsotropicScaleError> {
        if value.is_finite() && value > 0.0 && value.recip().is_finite() {
            Ok(Self(value))
        } else {
            Err(IsotropicScaleError { value })
        }
    }

    pub fn get(self) -> f64 {
        self.0
    }

    pub fn reciprocal(self) -> f64 {
        self.0.recip()
    }

    pub fn to_bits(self) -> u64 {
        self.0.to_bits()
    }

    /// Convert a coordinate-valued length from original to standardized units.
    ///
    /// This is the ONLY conversion between the two frames.  Because it
    /// consumes an [`OriginalUnits`] and produces a [`StandardizedUnits`],
    /// applying it to a value that is already standardized — the
    /// double-divide that silently halves or doubles a kernel bandwidth — is
    /// a type error rather than a comment for the next reader to remember.
    pub fn to_standardized_units(self, value: OriginalUnits) -> StandardizedUnits {
        StandardizedUnits(value.original_value() * self.reciprocal())
    }

    /// The inverse of [`Self::to_standardized_units`]: express a standardized
    /// length back in the user's original coordinate units.
    pub fn to_original_units(self, value: StandardizedUnits) -> OriginalUnits {
        OriginalUnits(value.standardized_value() * self.0)
    }

    /// Apply the uniform coordinate pullback in place.
    pub fn standardize(self, coordinates: &mut Array2<f64>) {
        let reciprocal = self.reciprocal();
        coordinates.mapv_inplace(|value| value * reciprocal);
    }
}

impl TryFrom<f64> for IsotropicScale {
    type Error = IsotropicScaleError;

    fn try_from(value: f64) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl From<IsotropicScale> for f64 {
    fn from(value: IsotropicScale) -> Self {
        value.get()
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IsotropicScaleError {
    value: f64,
}

impl fmt::Display for IsotropicScaleError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "isotropic scale must be positive and finite with a finite reciprocal, got {}",
            self.value
        )
    }
}

impl std::error::Error for IsotropicScaleError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn construction_enforces_the_operational_invariant() {
        assert_eq!(IsotropicScale::new(1.0), Ok(IsotropicScale::ONE));
        assert!(IsotropicScale::new(f64::MIN_POSITIVE).is_ok());
        assert!(IsotropicScale::new(0.0).is_err());
        assert!(IsotropicScale::new(-1.0).is_err());
        assert!(IsotropicScale::new(f64::NAN).is_err());
        assert!(IsotropicScale::new(f64::INFINITY).is_err());
        assert!(IsotropicScale::new(f64::from_bits(1)).is_err());
    }

    #[test]
    fn the_two_frames_round_trip_through_the_only_conversion() {
        let scale = IsotropicScale::new(4.0).unwrap();
        let original = OriginalUnits::new(500.0);
        let standardized = scale.to_standardized_units(original);
        assert_eq!(standardized, StandardizedUnits::new(125.0));
        assert_eq!(scale.to_original_units(standardized), original);
        // The frame tag is what distinguishes the two, not the magnitude:
        // a bare `125.0` carries no evidence of which side it came from,
        // which is precisely the hazard the tags remove.
        assert_eq!(
            scale.to_standardized_units(OriginalUnits::new(125.0)),
            StandardizedUnits::new(31.25)
        );
    }

    #[test]
    fn wire_representation_is_a_checked_scalar() {
        let encoded = serde_json::to_string(&IsotropicScale::new(2.5).unwrap()).unwrap();
        assert_eq!(encoded, "2.5");
        assert_eq!(
            serde_json::from_str::<IsotropicScale>(&encoded).unwrap(),
            IsotropicScale::new(2.5).unwrap()
        );
        assert!(serde_json::from_str::<IsotropicScale>("[2.5,2.5]").is_err());
        assert!(serde_json::from_str::<IsotropicScale>("0.0").is_err());
        assert!(serde_json::from_str::<IsotropicScale>("-1.0").is_err());
    }
}
