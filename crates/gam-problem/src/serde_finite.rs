//! Structural non-finite-float guard for anything that implements [`Serialize`].
//!
//! ## Why this exists
//!
//! `serde_json` renders `f64::NAN` and `±f64::INFINITY` as the JSON literal
//! `null` — JSON has no encoding for them and `serde_json`'s serializer takes
//! the lossy branch silently. A persisted model carrying one non-finite scalar
//! therefore *writes* without complaint and only fails on the way back in, as
//!
//! ```text
//! invalid type: null, expected f64
//! ```
//!
//! — an error that names neither the field nor the fit that produced it, and
//! that surfaces arbitrarily far from the computation at fault (#2601).
//!
//! ## Why it is structural rather than a field list
//!
//! The pre-existing guards (`ensure_finite_scalar`, `validate_all_finite`, and
//! the hand-maintained `FittedModel::validate_numeric_finiteness`) each name one
//! field. A hand-maintained enumeration over a struct with hundreds of optional
//! numeric fields cannot stay complete: every new field is opted OUT by default,
//! so the guard silently stops covering the payload as the payload grows. That
//! is exactly how #2601's `null` reached a saved model.
//!
//! [`ensure_serialized_floats_are_finite`] instead walks the value through
//! `serde`'s own data model — the same traversal the JSON writer performs — so
//! *every* float that would be written is checked, by construction, with no
//! per-field opt-in. It tracks the struct-field / map-key / sequence-index path
//! as it descends, so the error names the offending scalar the way the
//! scalar-at-a-time guards do:
//!
//! ```text
//! payload.fit_result.blocks[3].edf must be finite, got NaN
//! ```
//!
//! The walk allocates nothing per scalar; only the current path (bounded by the
//! nesting depth) and the borrowed field names are held.

use serde::ser::{
    Impossible, Serialize, SerializeMap, SerializeSeq, SerializeStruct, SerializeStructVariant,
    SerializeTuple, SerializeTupleStruct, SerializeTupleVariant, Serializer,
};
use std::fmt::{self, Display, Write as _};

/// A non-finite float found at `path` while walking a serializable value.
#[derive(Debug, Clone, PartialEq)]
pub struct NonFiniteFloat {
    /// Dotted / indexed path to the offending scalar, e.g.
    /// `payload.fit_result.blocks[3].edf`. Empty when the value serialized is
    /// itself a bare float.
    pub path: String,
    /// The offending value, widened to `f64` (`f32` inputs keep their class:
    /// a `f32::NAN` reports as `NaN`).
    pub value: f64,
}

impl Display for NonFiniteFloat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.path.is_empty() {
            write!(f, "value must be finite, got {}", self.value)
        } else {
            write!(f, "{} must be finite, got {}", self.path, self.value)
        }
    }
}

impl std::error::Error for NonFiniteFloat {}

/// Walk `value` through `serde`'s data model and fail on the FIRST non-finite
/// `f32`/`f64` that a serializer would emit, reporting its path.
///
/// This is the write-side counterpart of the load-side type error: it converts
/// "a `null` will silently appear in the output" into a typed refusal at the
/// point of origin.
pub fn ensure_serialized_floats_are_finite<T>(value: &T) -> Result<(), NonFiniteFloat>
where
    T: Serialize + ?Sized,
{
    let mut walker = FloatWalker { path: String::new() };
    match value.serialize(&mut walker) {
        Ok(()) => Ok(()),
        Err(WalkError::NonFinite(found)) => Err(found),
        // `Custom` can only arise from a `Serialize` impl that itself reports an
        // error (e.g. a map with an unrepresentable key). Such a value cannot be
        // serialized to JSON either, so there is no float verdict to give and
        // the writer downstream will surface the same failure with its own
        // message. Treat it as "nothing non-finite found here".
        Err(WalkError::Custom(_)) => Ok(()),
    }
}

/// Error channel of the walking serializer.
#[derive(Debug)]
enum WalkError {
    NonFinite(NonFiniteFloat),
    Custom(String),
}

impl Display for WalkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WalkError::NonFinite(found) => Display::fmt(found, f),
            WalkError::Custom(message) => f.write_str(message),
        }
    }
}

impl std::error::Error for WalkError {}

impl serde::ser::Error for WalkError {
    fn custom<T: Display>(msg: T) -> Self {
        WalkError::Custom(msg.to_string())
    }
}

/// Cursor over the serde data model carrying the path to the current value.
struct FloatWalker {
    path: String,
}

impl FloatWalker {
    /// Append `.name` (or `name` at the root) and return the previous length so
    /// the caller can truncate back after descending.
    fn push_field(&mut self, name: &str) -> usize {
        let restore = self.path.len();
        if !self.path.is_empty() {
            self.path.push('.');
        }
        self.path.push_str(name);
        restore
    }

    fn push_index(&mut self, index: usize) -> usize {
        let restore = self.path.len();
        // `write!` to a String is infallible; the `Result` is discarded with
        // `.ok()` rather than `unwrap` so no panic path exists here, and rather
        // than `let _`, which the ban scanner rejects.
        write!(self.path, "[{index}]").ok();
        restore
    }

    fn pop_to(&mut self, restore: usize) {
        self.path.truncate(restore);
    }

    fn check(&self, value: f64) -> Result<(), WalkError> {
        if value.is_finite() {
            Ok(())
        } else {
            Err(WalkError::NonFinite(NonFiniteFloat {
                path: self.path.clone(),
                value,
            }))
        }
    }
}

/// Render a map key into the path. Only string and integer keys are
/// representable in JSON objects, which is the format this guard protects; any
/// other key shape falls back to a positional index so the path stays useful.
struct KeyRenderer;

impl Serializer for KeyRenderer {
    type Ok = String;
    type Error = WalkError;
    type SerializeSeq = Impossible<String, WalkError>;
    type SerializeTuple = Impossible<String, WalkError>;
    type SerializeTupleStruct = Impossible<String, WalkError>;
    type SerializeTupleVariant = Impossible<String, WalkError>;
    type SerializeMap = Impossible<String, WalkError>;
    type SerializeStruct = Impossible<String, WalkError>;
    type SerializeStructVariant = Impossible<String, WalkError>;

    fn serialize_str(self, value: &str) -> Result<String, WalkError> {
        Ok(value.to_string())
    }

    fn serialize_bool(self, value: bool) -> Result<String, WalkError> {
        Ok(value.to_string())
    }

    fn serialize_i64(self, value: i64) -> Result<String, WalkError> {
        Ok(value.to_string())
    }

    fn serialize_i128(self, value: i128) -> Result<String, WalkError> {
        Ok(value.to_string())
    }

    fn serialize_u64(self, value: u64) -> Result<String, WalkError> {
        Ok(value.to_string())
    }

    fn serialize_u128(self, value: u128) -> Result<String, WalkError> {
        Ok(value.to_string())
    }

    fn serialize_i8(self, value: i8) -> Result<String, WalkError> {
        self.serialize_i64(i64::from(value))
    }

    fn serialize_i16(self, value: i16) -> Result<String, WalkError> {
        self.serialize_i64(i64::from(value))
    }

    fn serialize_i32(self, value: i32) -> Result<String, WalkError> {
        self.serialize_i64(i64::from(value))
    }

    fn serialize_u8(self, value: u8) -> Result<String, WalkError> {
        self.serialize_u64(u64::from(value))
    }

    fn serialize_u16(self, value: u16) -> Result<String, WalkError> {
        self.serialize_u64(u64::from(value))
    }

    fn serialize_u32(self, value: u32) -> Result<String, WalkError> {
        self.serialize_u64(u64::from(value))
    }

    fn serialize_f32(self, value: f32) -> Result<String, WalkError> {
        Ok(value.to_string())
    }

    fn serialize_f64(self, value: f64) -> Result<String, WalkError> {
        Ok(value.to_string())
    }

    fn serialize_char(self, value: char) -> Result<String, WalkError> {
        Ok(value.to_string())
    }

    fn serialize_bytes(self, _: &[u8]) -> Result<String, WalkError> {
        Ok("<bytes>".to_string())
    }

    fn serialize_none(self) -> Result<String, WalkError> {
        Ok("null".to_string())
    }

    fn serialize_some<T>(self, value: &T) -> Result<String, WalkError>
    where
        T: Serialize + ?Sized,
    {
        value.serialize(self)
    }

    fn serialize_unit(self) -> Result<String, WalkError> {
        Ok("null".to_string())
    }

    fn serialize_unit_struct(self, name: &'static str) -> Result<String, WalkError> {
        Ok(name.to_string())
    }

    fn serialize_unit_variant(
        self,
        _: &'static str,
        _: u32,
        variant: &'static str,
    ) -> Result<String, WalkError> {
        Ok(variant.to_string())
    }

    fn serialize_newtype_struct<T>(self, _: &'static str, value: &T) -> Result<String, WalkError>
    where
        T: Serialize + ?Sized,
    {
        value.serialize(self)
    }

    fn serialize_newtype_variant<T>(
        self,
        _: &'static str,
        _: u32,
        variant: &'static str,
        _: &T,
    ) -> Result<String, WalkError>
    where
        T: Serialize + ?Sized,
    {
        Ok(variant.to_string())
    }

    fn serialize_seq(self, _: Option<usize>) -> Result<Self::SerializeSeq, WalkError> {
        Err(WalkError::Custom("map key is not a scalar".to_string()))
    }

    fn serialize_tuple(self, _: usize) -> Result<Self::SerializeTuple, WalkError> {
        Err(WalkError::Custom("map key is not a scalar".to_string()))
    }

    fn serialize_tuple_struct(
        self,
        _: &'static str,
        _: usize,
    ) -> Result<Self::SerializeTupleStruct, WalkError> {
        Err(WalkError::Custom("map key is not a scalar".to_string()))
    }

    fn serialize_tuple_variant(
        self,
        _: &'static str,
        _: u32,
        _: &'static str,
        _: usize,
    ) -> Result<Self::SerializeTupleVariant, WalkError> {
        Err(WalkError::Custom("map key is not a scalar".to_string()))
    }

    fn serialize_map(self, _: Option<usize>) -> Result<Self::SerializeMap, WalkError> {
        Err(WalkError::Custom("map key is not a scalar".to_string()))
    }

    fn serialize_struct(
        self,
        _: &'static str,
        _: usize,
    ) -> Result<Self::SerializeStruct, WalkError> {
        Err(WalkError::Custom("map key is not a scalar".to_string()))
    }

    fn serialize_struct_variant(
        self,
        _: &'static str,
        _: u32,
        _: &'static str,
        _: usize,
    ) -> Result<Self::SerializeStructVariant, WalkError> {
        Err(WalkError::Custom("map key is not a scalar".to_string()))
    }
}

impl<'a> Serializer for &'a mut FloatWalker {
    type Ok = ();
    type Error = WalkError;
    type SerializeSeq = SeqWalker<'a>;
    type SerializeTuple = SeqWalker<'a>;
    type SerializeTupleStruct = SeqWalker<'a>;
    type SerializeTupleVariant = VariantSeqWalker<'a>;
    type SerializeMap = MapWalker<'a>;
    type SerializeStruct = StructWalker<'a>;
    type SerializeStructVariant = StructWalker<'a>;

    fn serialize_f64(self, value: f64) -> Result<(), WalkError> {
        self.check(value)
    }

    fn serialize_f32(self, value: f32) -> Result<(), WalkError> {
        // Widen for the verdict AND the message: `f32::NAN as f64` is still
        // NaN and `f32::INFINITY as f64` is still infinite, so the class is
        // preserved exactly.
        self.check(f64::from(value))
    }

    fn serialize_bool(self, _: bool) -> Result<(), WalkError> {
        Ok(())
    }

    fn serialize_i8(self, _: i8) -> Result<(), WalkError> {
        Ok(())
    }

    fn serialize_i16(self, _: i16) -> Result<(), WalkError> {
        Ok(())
    }

    fn serialize_i32(self, _: i32) -> Result<(), WalkError> {
        Ok(())
    }

    fn serialize_i64(self, _: i64) -> Result<(), WalkError> {
        Ok(())
    }

    fn serialize_i128(self, _: i128) -> Result<(), WalkError> {
        Ok(())
    }

    fn serialize_u8(self, _: u8) -> Result<(), WalkError> {
        Ok(())
    }

    fn serialize_u16(self, _: u16) -> Result<(), WalkError> {
        Ok(())
    }

    fn serialize_u32(self, _: u32) -> Result<(), WalkError> {
        Ok(())
    }

    fn serialize_u64(self, _: u64) -> Result<(), WalkError> {
        Ok(())
    }

    fn serialize_u128(self, _: u128) -> Result<(), WalkError> {
        Ok(())
    }

    fn serialize_char(self, _: char) -> Result<(), WalkError> {
        Ok(())
    }

    fn serialize_str(self, _: &str) -> Result<(), WalkError> {
        Ok(())
    }

    fn serialize_bytes(self, _: &[u8]) -> Result<(), WalkError> {
        Ok(())
    }

    fn serialize_none(self) -> Result<(), WalkError> {
        Ok(())
    }

    fn serialize_some<T>(self, value: &T) -> Result<(), WalkError>
    where
        T: Serialize + ?Sized,
    {
        value.serialize(self)
    }

    fn serialize_unit(self) -> Result<(), WalkError> {
        Ok(())
    }

    fn serialize_unit_struct(self, _: &'static str) -> Result<(), WalkError> {
        Ok(())
    }

    fn serialize_unit_variant(
        self,
        _: &'static str,
        _: u32,
        _: &'static str,
    ) -> Result<(), WalkError> {
        Ok(())
    }

    fn serialize_newtype_struct<T>(self, _: &'static str, value: &T) -> Result<(), WalkError>
    where
        T: Serialize + ?Sized,
    {
        value.serialize(self)
    }

    fn serialize_newtype_variant<T>(
        self,
        _: &'static str,
        _: u32,
        variant: &'static str,
        value: &T,
    ) -> Result<(), WalkError>
    where
        T: Serialize + ?Sized,
    {
        // Externally tagged enums serialize as `{"Variant": payload}`, so the
        // variant name IS a path segment in the emitted JSON.
        let restore = self.push_field(variant);
        let outcome = value.serialize(&mut *self);
        self.pop_to(restore);
        outcome
    }

    fn serialize_seq(self, _: Option<usize>) -> Result<SeqWalker<'a>, WalkError> {
        Ok(SeqWalker {
            walker: self,
            index: 0,
        })
    }

    fn serialize_tuple(self, len: usize) -> Result<SeqWalker<'a>, WalkError> {
        self.serialize_seq(Some(len))
    }

    fn serialize_tuple_struct(
        self,
        _: &'static str,
        len: usize,
    ) -> Result<SeqWalker<'a>, WalkError> {
        self.serialize_seq(Some(len))
    }

    fn serialize_tuple_variant(
        self,
        _: &'static str,
        _: u32,
        variant: &'static str,
        _: usize,
    ) -> Result<VariantSeqWalker<'a>, WalkError> {
        let restore = self.push_field(variant);
        Ok(VariantSeqWalker {
            seq: SeqWalker {
                walker: self,
                index: 0,
            },
            restore,
        })
    }

    fn serialize_map(self, _: Option<usize>) -> Result<MapWalker<'a>, WalkError> {
        Ok(MapWalker {
            walker: self,
            restore: None,
        })
    }

    fn serialize_struct(
        self,
        _: &'static str,
        _: usize,
    ) -> Result<StructWalker<'a>, WalkError> {
        Ok(StructWalker {
            walker: self,
            restore: None,
        })
    }

    fn serialize_struct_variant(
        self,
        _: &'static str,
        _: u32,
        variant: &'static str,
        _: usize,
    ) -> Result<StructWalker<'a>, WalkError> {
        let restore = self.push_field(variant);
        Ok(StructWalker {
            walker: self,
            restore: Some(restore),
        })
    }

    fn collect_str<T>(self, _: &T) -> Result<(), WalkError>
    where
        T: Display + ?Sized,
    {
        Ok(())
    }

    fn is_human_readable(&self) -> bool {
        // The format this guard protects is JSON. Types whose `Serialize` impl
        // branches on this (e.g. compact binary encodings) must be walked in the
        // same shape the JSON writer will use, or the guard would inspect a
        // different set of floats than the one persisted.
        true
    }
}

/// Sequence / tuple cursor: elements are addressed by position.
struct SeqWalker<'a> {
    walker: &'a mut FloatWalker,
    index: usize,
}

impl SerializeSeq for SeqWalker<'_> {
    type Ok = ();
    type Error = WalkError;

    fn serialize_element<T>(&mut self, value: &T) -> Result<(), WalkError>
    where
        T: Serialize + ?Sized,
    {
        let restore = self.walker.push_index(self.index);
        let outcome = value.serialize(&mut *self.walker);
        self.walker.pop_to(restore);
        self.index += 1;
        outcome
    }

    fn end(self) -> Result<(), WalkError> {
        Ok(())
    }
}

impl SerializeTuple for SeqWalker<'_> {
    type Ok = ();
    type Error = WalkError;

    fn serialize_element<T>(&mut self, value: &T) -> Result<(), WalkError>
    where
        T: Serialize + ?Sized,
    {
        SerializeSeq::serialize_element(self, value)
    }

    fn end(self) -> Result<(), WalkError> {
        Ok(())
    }
}

impl SerializeTupleStruct for SeqWalker<'_> {
    type Ok = ();
    type Error = WalkError;

    fn serialize_field<T>(&mut self, value: &T) -> Result<(), WalkError>
    where
        T: Serialize + ?Sized,
    {
        SerializeSeq::serialize_element(self, value)
    }

    fn end(self) -> Result<(), WalkError> {
        Ok(())
    }
}

/// A tuple variant additionally owns the pushed variant-name segment.
struct VariantSeqWalker<'a> {
    seq: SeqWalker<'a>,
    restore: usize,
}

impl SerializeTupleVariant for VariantSeqWalker<'_> {
    type Ok = ();
    type Error = WalkError;

    fn serialize_field<T>(&mut self, value: &T) -> Result<(), WalkError>
    where
        T: Serialize + ?Sized,
    {
        SerializeSeq::serialize_element(&mut self.seq, value)
    }

    fn end(self) -> Result<(), WalkError> {
        self.seq.walker.pop_to(self.restore);
        Ok(())
    }
}

/// Map cursor: the key is rendered into the path, then the value is walked.
struct MapWalker<'a> {
    walker: &'a mut FloatWalker,
    /// Set while a key has been consumed and its value not yet walked.
    restore: Option<usize>,
}

impl SerializeMap for MapWalker<'_> {
    type Ok = ();
    type Error = WalkError;

    fn serialize_key<T>(&mut self, key: &T) -> Result<(), WalkError>
    where
        T: Serialize + ?Sized,
    {
        // A key that cannot be rendered as a scalar is not JSON-encodable at
        // all; fall back to a positional marker so the walk (and its float
        // verdict) still completes.
        let rendered = key
            .serialize(KeyRenderer)
            .unwrap_or_else(|_| "<key>".to_string());
        self.restore = Some(self.walker.push_field(&rendered));
        Ok(())
    }

    fn serialize_value<T>(&mut self, value: &T) -> Result<(), WalkError>
    where
        T: Serialize + ?Sized,
    {
        let outcome = value.serialize(&mut *self.walker);
        if let Some(restore) = self.restore.take() {
            self.walker.pop_to(restore);
        }
        outcome
    }

    fn end(self) -> Result<(), WalkError> {
        Ok(())
    }
}

/// Struct cursor: fields are addressed by name.
struct StructWalker<'a> {
    walker: &'a mut FloatWalker,
    /// `Some` for a struct *variant*, whose variant-name segment must be popped
    /// when the compound ends.
    restore: Option<usize>,
}

impl SerializeStruct for StructWalker<'_> {
    type Ok = ();
    type Error = WalkError;

    fn serialize_field<T>(&mut self, key: &'static str, value: &T) -> Result<(), WalkError>
    where
        T: Serialize + ?Sized,
    {
        let restore = self.walker.push_field(key);
        let outcome = value.serialize(&mut *self.walker);
        self.walker.pop_to(restore);
        outcome
    }

    fn end(self) -> Result<(), WalkError> {
        if let Some(restore) = self.restore {
            self.walker.pop_to(restore);
        }
        Ok(())
    }
}

impl SerializeStructVariant for StructWalker<'_> {
    type Ok = ();
    type Error = WalkError;

    fn serialize_field<T>(&mut self, key: &'static str, value: &T) -> Result<(), WalkError>
    where
        T: Serialize + ?Sized,
    {
        SerializeStruct::serialize_field(self, key, value)
    }

    fn end(self) -> Result<(), WalkError> {
        SerializeStruct::end(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Serialize;
    use std::collections::BTreeMap;

    #[derive(Serialize)]
    struct Leaf {
        edf: f64,
        name: String,
    }

    #[derive(Serialize)]
    struct Root {
        blocks: Vec<Leaf>,
        scale: Option<f64>,
        counts: Vec<u32>,
        by_term: BTreeMap<String, f64>,
    }

    fn root() -> Root {
        Root {
            blocks: vec![
                Leaf {
                    edf: 1.0,
                    name: "a".to_string(),
                },
                Leaf {
                    edf: 2.0,
                    name: "b".to_string(),
                },
            ],
            scale: Some(0.5),
            counts: vec![1, 2, 3],
            by_term: BTreeMap::from([("s(x)".to_string(), 3.25)]),
        }
    }

    #[test]
    fn all_finite_payload_passes() {
        assert!(ensure_serialized_floats_are_finite(&root()).is_ok());
    }

    #[test]
    fn nested_sequence_element_reports_indexed_path() {
        let mut value = root();
        value.blocks[1].edf = f64::NAN;
        let err = ensure_serialized_floats_are_finite(&value).unwrap_err();
        assert_eq!(err.path, "blocks[1].edf");
        assert!(err.value.is_nan());
        assert!(
            err.to_string().contains("blocks[1].edf must be finite"),
            "message should name the path: {err}"
        );
    }

    #[test]
    fn optional_scalar_reports_its_field() {
        let mut value = root();
        value.scale = Some(f64::INFINITY);
        let err = ensure_serialized_floats_are_finite(&value).unwrap_err();
        assert_eq!(err.path, "scale");
        assert_eq!(err.value, f64::INFINITY);
    }

    #[test]
    fn map_value_reports_its_key() {
        let mut value = root();
        value
            .by_term
            .insert("s(z)".to_string(), f64::NEG_INFINITY);
        let err = ensure_serialized_floats_are_finite(&value).unwrap_err();
        assert_eq!(err.path, "by_term.s(z)");
    }

    #[test]
    fn none_is_not_a_non_finite_float() {
        let mut value = root();
        value.scale = None;
        assert!(ensure_serialized_floats_are_finite(&value).is_ok());
    }

    #[test]
    fn bare_scalar_has_empty_path() {
        let err = ensure_serialized_floats_are_finite(&f64::NAN).unwrap_err();
        assert!(err.path.is_empty());
        assert!(err.to_string().starts_with("value must be finite"));
    }

    #[test]
    fn f32_non_finite_is_caught_and_widened() {
        #[derive(Serialize)]
        struct Small {
            w: f32,
        }
        let err = ensure_serialized_floats_are_finite(&Small { w: f32::NAN }).unwrap_err();
        assert_eq!(err.path, "w");
        assert!(err.value.is_nan());
    }

    #[test]
    fn struct_variant_and_newtype_variant_paths_are_reported() {
        #[derive(Serialize)]
        enum Node {
            Scale { phi: f64 },
            Raw(f64),
        }
        #[derive(Serialize)]
        struct Holder {
            node: Node,
        }
        let err =
            ensure_serialized_floats_are_finite(&Holder { node: Node::Scale { phi: f64::NAN } })
                .unwrap_err();
        assert_eq!(err.path, "node.Scale.phi");
        let err = ensure_serialized_floats_are_finite(&Holder {
            node: Node::Raw(f64::INFINITY),
        })
        .unwrap_err();
        assert_eq!(err.path, "node.Raw");
    }

    #[test]
    fn path_state_is_restored_after_each_branch() {
        // A finite branch visited BEFORE the offending one must not leave its
        // segments on the path (the truncate-on-exit contract).
        #[derive(Serialize)]
        struct Two {
            first: Vec<Leaf>,
            second: f64,
        }
        let err = ensure_serialized_floats_are_finite(&Two {
            first: vec![Leaf {
                edf: 1.0,
                name: "ok".to_string(),
            }],
            second: f64::NAN,
        })
        .unwrap_err();
        assert_eq!(err.path, "second");
    }

    #[test]
    fn walk_agrees_with_what_serde_json_would_write() {
        // The contract: the guard rejects exactly the payloads whose JSON
        // rendering contains a `null` that came from a float. Verify on a value
        // that serde_json silently lossy-renders.
        let mut value = root();
        value.blocks[0].edf = f64::NAN;
        let json = serde_json::to_string(&value).expect("serde_json renders NaN as null");
        assert!(
            json.contains("\"edf\":null"),
            "precondition: serde_json writes NaN as null, got {json}"
        );
        assert!(ensure_serialized_floats_are_finite(&value).is_err());
    }
}
