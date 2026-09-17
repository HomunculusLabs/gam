//! The single Rust entry behind MPD's Python and CLI surfaces (#2951).
//!
//! A request is a versioned JSON document plus named `f64` arrays, and a report is
//! a versioned JSON document plus named `f64` arrays. Arrays never travel inside
//! the JSON: an eigenbasis at model width is not text, so the document names each
//! array by id and the transport carries it (numpy in `gam-pyffi`, NPY files in
//! `gam-cli`). Both front ends call only [`run_parameter_decomposition`], so
//! parsing, validation, dispatch and projection happen once, and an operation
//! added here reaches Python and the CLI with no front-end change.
//!
//! Nothing here computes. Each operation calls its owner and projects the owner's
//! result into the wire report without strengthening any claim. A value the owner
//! reports as `+inf` to mean "no such quantity" (a cluster that is the whole
//! spectrum has no separation; a refused Davis–Kahan bound has no bar) becomes an
//! absent value with that stated meaning. Any other non-finite value is refused
//! instead of being written as JSON `null`.

use std::collections::BTreeMap;
use std::fmt;

use ndarray::{ArrayD, Ix2};
use serde::{Deserialize, Serialize};

use super::spectral::{
    PlaneRotationError, PlaneRotationRecovery, RotationAmbiguity, RotationClusterKind,
    recover_plane_rotations,
};

/// Identity of the request document.
pub const MPD_REQUEST_SCHEMA: &str = "gam.mpd-request";

/// Identity of the report document.
pub const MPD_REPORT_SCHEMA: &str = "gam.mpd-report";

/// Version shared by the request and report documents.
pub const MPD_SCHEMA_VERSION: u32 = 1;

/// A complete, front-end-neutral MPD request. Arrays are not embedded; each
/// operation names the input arrays it reads.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MpdRequest {
    pub schema: String,
    pub schema_version: u32,
    pub operation: MpdOperation,
}

/// The closed set of operations. Each variant carries exactly its owner's declared
/// inputs, with no defaults.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum MpdOperation {
    /// P3: the rotation planes of one square matrix (`spectral`).
    RecoverPlaneRotations {
        /// Id of the input array holding the matrix.
        tensor: String,
        /// The declared 2-norm distance from the matrix to the orthogonal group. It is an
        /// experiment declaration, required and with no default; the owner refuses a matrix
        /// provably further away.
        declared_error: f64,
    },
}

impl MpdRequest {
    /// Parses and validates a request document.
    pub fn from_json(raw: &str) -> Result<Self, MpdSurfaceError> {
        let request: Self = serde_json::from_str(raw)
            .map_err(|error| MpdSurfaceError::InvalidRequest(error.to_string()))?;
        if request.schema != MPD_REQUEST_SCHEMA {
            return Err(MpdSurfaceError::InvalidRequest(format!(
                "request schema must be {MPD_REQUEST_SCHEMA:?}, got {:?}",
                request.schema
            )));
        }
        if request.schema_version != MPD_SCHEMA_VERSION {
            return Err(MpdSurfaceError::InvalidRequest(format!(
                "unsupported request schema_version {}; expected {MPD_SCHEMA_VERSION}",
                request.schema_version
            )));
        }
        Ok(request)
    }
}

/// The report document.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct MpdReport {
    pub schema: &'static str,
    pub schema_version: u32,
    pub result: MpdResult,
}

/// One operation's projected result.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum MpdResult {
    RecoverPlaneRotations(PlaneRotationReport),
}

/// [`PlaneRotationRecovery`] on the wire.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct PlaneRotationReport {
    /// The input array the matrix came from.
    pub tensor: String,
    pub orthogonality_defect: f64,
    pub perturbation_bound: f64,
    /// Clusters in increasing cosine order.
    pub clusters: Vec<RotationClusterReport>,
    pub ambiguities: Vec<RotationAmbiguityReport>,
}

/// One spectral cluster on the wire.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct RotationClusterReport {
    /// Id of the output array holding the cluster's orthonormal basis (`d x m`).
    pub basis: String,
    /// `m`, the dimension of the cluster's eigenspace.
    pub dimension: usize,
    pub cosine_interval: [f64; 2],
    /// Distance to the nearest other computed eigenvalue; absent when the cluster
    /// is the whole spectrum.
    pub separation: Option<f64>,
    /// Davis–Kahan bar on the projector; absent when the bound refuses because the
    /// separation does not exceed the perturbation bound, so no subspace claim
    /// stands.
    pub projector_bar: Option<f64>,
    pub structure: RotationClusterStructure,
}

/// [`RotationClusterKind`] on the wire.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RotationClusterStructure {
    Fixed {
        max_hidden_angle: f64,
    },
    Rotation {
        planes: usize,
        angle: f64,
        /// Id of the output array holding `J` in the cluster basis; absent when the
        /// orientation is not certified.
        complex_structure: Option<String>,
    },
    HalfTurn {
        min_hidden_angle: f64,
    },
    Unresolved,
}

/// [`RotationAmbiguity`] on the wire.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RotationAmbiguityReport {
    Identity,
    RepeatedCosine { cluster: usize, planes: usize },
    HalfTurn { cluster: usize },
    Unresolved { cluster: usize },
    Winding,
}

impl From<RotationAmbiguity> for RotationAmbiguityReport {
    fn from(ambiguity: RotationAmbiguity) -> Self {
        match ambiguity {
            RotationAmbiguity::Identity => Self::Identity,
            RotationAmbiguity::RepeatedCosine { cluster, planes } => {
                Self::RepeatedCosine { cluster, planes }
            }
            RotationAmbiguity::HalfTurn { cluster } => Self::HalfTurn { cluster },
            RotationAmbiguity::Unresolved { cluster } => Self::Unresolved { cluster },
            RotationAmbiguity::Winding => Self::Winding,
        }
    }
}

/// A report and the arrays it names.
#[derive(Clone, Debug, PartialEq)]
pub struct MpdOutput {
    pub report: MpdReport,
    pub arrays: BTreeMap<String, ArrayD<f64>>,
}

impl MpdOutput {
    /// The report document, the same bytes for every front end.
    pub fn report_json(&self) -> Result<String, MpdSurfaceError> {
        serde_json::to_string_pretty(&self.report)
            .map_err(|error| MpdSurfaceError::Serialize(error.to_string()))
    }
}

/// Why a request produced no report.
#[derive(Debug)]
pub enum MpdSurfaceError {
    /// The document did not parse, or named another schema or version.
    InvalidRequest(String),
    /// The operation names an input array that was not supplied.
    MissingTensor { tensor: String },
    /// An input array does not have the shape the operation reads.
    TensorShape { tensor: String, reason: String },
    PlaneRotation(PlaneRotationError),
    /// An owner returned a non-finite value where the wire report has no meaning
    /// for one.
    NonFiniteReport { field: &'static str, value: f64 },
    Serialize(String),
}

impl fmt::Display for MpdSurfaceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidRequest(reason) => write!(formatter, "invalid MPD request: {reason}"),
            Self::MissingTensor { tensor } => write!(
                formatter,
                "MPD request names input array {tensor:?}, which was not supplied"
            ),
            Self::TensorShape { tensor, reason } => {
                write!(formatter, "MPD input array {tensor:?}: {reason}")
            }
            Self::PlaneRotation(error) => write!(formatter, "{error}"),
            Self::NonFiniteReport { field, value } => write!(
                formatter,
                "MPD report field {field} is {value}, which the wire report cannot state"
            ),
            Self::Serialize(reason) => write!(formatter, "serialize MPD report: {reason}"),
        }
    }
}

impl std::error::Error for MpdSurfaceError {}

/// Runs one MPD request against its named input arrays.
pub fn run_parameter_decomposition(
    request_json: &str,
    tensors: &BTreeMap<String, ArrayD<f64>>,
) -> Result<MpdOutput, MpdSurfaceError> {
    let request = MpdRequest::from_json(request_json)?;
    match request.operation {
        MpdOperation::RecoverPlaneRotations {
            tensor,
            declared_error,
        } => {
            let input = tensors
                .get(&tensor)
                .ok_or_else(|| MpdSurfaceError::MissingTensor {
                    tensor: tensor.clone(),
                })?;
            let matrix = input.view().into_dimensionality::<Ix2>().map_err(|error| {
                MpdSurfaceError::TensorShape {
                    tensor: tensor.clone(),
                    reason: format!("expected a matrix, got shape {:?}: {error}", input.shape()),
                }
            })?;
            let recovery = recover_plane_rotations(matrix, declared_error)
                .map_err(MpdSurfaceError::PlaneRotation)?;
            project_plane_rotations(tensor, recovery)
        }
    }
}

fn project_plane_rotations(
    tensor: String,
    recovery: PlaneRotationRecovery,
) -> Result<MpdOutput, MpdSurfaceError> {
    let ambiguities = recovery
        .ambiguities()
        .into_iter()
        .map(RotationAmbiguityReport::from)
        .collect();
    let orthogonality_defect = finite("orthogonality_defect", recovery.orthogonality_defect)?;
    let perturbation_bound = finite("perturbation_bound", recovery.perturbation_bound)?;
    let mut arrays = BTreeMap::new();
    let mut clusters = Vec::with_capacity(recovery.clusters.len());
    for (index, cluster) in recovery.clusters.into_iter().enumerate() {
        let structure = match cluster.kind {
            RotationClusterKind::Fixed { max_hidden_angle } => RotationClusterStructure::Fixed {
                max_hidden_angle: finite("max_hidden_angle", max_hidden_angle)?,
            },
            RotationClusterKind::Rotation {
                planes,
                angle,
                complex_structure,
            } => RotationClusterStructure::Rotation {
                planes,
                angle: finite("angle", angle)?,
                complex_structure: complex_structure.map(|structure| {
                    let id = format!("clusters/{index}/complex_structure");
                    arrays.insert(id.clone(), structure.into_dyn());
                    id
                }),
            },
            RotationClusterKind::HalfTurn { min_hidden_angle } => {
                RotationClusterStructure::HalfTurn {
                    min_hidden_angle: finite("min_hidden_angle", min_hidden_angle)?,
                }
            }
            RotationClusterKind::Unresolved => RotationClusterStructure::Unresolved,
        };
        let basis = format!("clusters/{index}/basis");
        let dimension = cluster.basis.ncols();
        arrays.insert(basis.clone(), cluster.basis.into_dyn());
        clusters.push(RotationClusterReport {
            basis,
            dimension,
            cosine_interval: [
                finite("cosine_interval", cluster.cosine_interval.0)?,
                finite("cosine_interval", cluster.cosine_interval.1)?,
            ],
            separation: absent_when_infinite("separation", cluster.separation)?,
            projector_bar: absent_when_infinite("projector_bar", cluster.projector_bar)?,
            structure,
        });
    }
    Ok(MpdOutput {
        report: MpdReport {
            schema: MPD_REPORT_SCHEMA,
            schema_version: MPD_SCHEMA_VERSION,
            result: MpdResult::RecoverPlaneRotations(PlaneRotationReport {
                tensor,
                orthogonality_defect,
                perturbation_bound,
                clusters,
                ambiguities,
            }),
        },
        arrays,
    })
}

fn finite(field: &'static str, value: f64) -> Result<f64, MpdSurfaceError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(MpdSurfaceError::NonFiniteReport { field, value })
    }
}

/// `+inf` is an owner's "no such quantity"; every other non-finite value is refused.
fn absent_when_infinite(field: &'static str, value: f64) -> Result<Option<f64>, MpdSurfaceError> {
    if value == f64::INFINITY {
        Ok(None)
    } else {
        finite(field, value).map(Some)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gam_linalg::roundoff::accumulation_growth;
    use ndarray::{Array2, array};

    fn request_json(operation: &str) -> String {
        format!(
            r#"{{"schema": "{MPD_REQUEST_SCHEMA}", "schema_version": {MPD_SCHEMA_VERSION}, "operation": {operation}}}"#
        )
    }

    fn plane_request(declared_error: f64) -> String {
        request_json(&format!(
            r#"{{"kind": "recover_plane_rotations", "tensor": "w", "declared_error": {declared_error:?}}}"#
        ))
    }

    fn inputs(id: &str, matrix: Array2<f64>) -> BTreeMap<String, ArrayD<f64>> {
        BTreeMap::from([(id.to_string(), matrix.into_dyn())])
    }

    fn frobenius(matrix: &Array2<f64>) -> f64 {
        matrix.iter().map(|value| value * value).sum::<f64>().sqrt()
    }

    /// A declared distance to the orthogonal group that covers the true one:
    /// `||W - W_o||_2 = max |sigma_i - 1| <= ||W^T W - I||_F`, plus the rounding of the
    /// computed Gram, `gamma_n` times the Frobenius norm of `|W|^T |W|`.
    fn orthogonality_declaration(matrix: &Array2<f64>) -> f64 {
        let columns = matrix.ncols();
        let gram = matrix.t().dot(matrix) - Array2::<f64>::eye(columns);
        let absolute = matrix.mapv(f64::abs);
        frobenius(&gram) + accumulation_growth(columns) * frobenius(&absolute.t().dot(&absolute))
    }

    /// Rotations by `alpha` and `beta` in the planes (e1, e2) and (e3, e4), with e5
    /// fixed.
    fn two_plane_rotation(alpha: f64, beta: f64) -> Array2<f64> {
        let (sa, ca) = alpha.sin_cos();
        let (sb, cb) = beta.sin_cos();
        array![
            [ca, -sa, 0.0, 0.0, 0.0],
            [sa, ca, 0.0, 0.0, 0.0],
            [0.0, 0.0, cb, -sb, 0.0],
            [0.0, 0.0, sb, cb, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0]
        ]
    }

    #[test]
    fn request_document_round_trips_and_refuses_what_it_does_not_declare() {
        let valid = plane_request(0.0);
        let request = MpdRequest::from_json(&valid).expect("a declared request parses");
        assert_eq!(
            request.operation,
            MpdOperation::RecoverPlaneRotations {
                tensor: "w".to_string(),
                declared_error: 0.0,
            }
        );
        let reserialized = serde_json::to_string(&request).expect("serialize request");
        assert_eq!(
            MpdRequest::from_json(&reserialized).expect("round trip parses"),
            request
        );

        // Each refusal is a small change to the accepted document above, so a guard
        // that refused everything would already have failed.
        let unknown_top = valid.replacen("\"operation\"", "\"extra\": 1, \"operation\"", 1);
        assert!(MpdRequest::from_json(&unknown_top).is_err());
        let unknown_operation_field = request_json(
            r#"{"kind": "recover_plane_rotations", "tensor": "w", "declared_error": 0.0, "tolerance": 1e-8}"#,
        );
        assert!(MpdRequest::from_json(&unknown_operation_field).is_err());
        let missing_tensor = request_json(r#"{"kind": "recover_plane_rotations", "declared_error": 0.0}"#);
        assert!(MpdRequest::from_json(&missing_tensor).is_err());
        // The declared error is an experiment declaration with no default.
        let missing_declaration = request_json(r#"{"kind": "recover_plane_rotations", "tensor": "w"}"#);
        assert!(MpdRequest::from_json(&missing_declaration).is_err());
        let unknown_kind =
            request_json(r#"{"kind": "guess_the_planes", "tensor": "w", "declared_error": 0.0}"#);
        assert!(MpdRequest::from_json(&unknown_kind).is_err());
        let other_schema = valid.replacen(MPD_REQUEST_SCHEMA, "gam.fit-request", 1);
        assert!(MpdRequest::from_json(&other_schema).is_err());
        let other_version = valid.replacen(
            &format!("\"schema_version\": {MPD_SCHEMA_VERSION}"),
            "\"schema_version\": 0",
            1,
        );
        assert!(MpdRequest::from_json(&other_version).is_err());
    }

    #[test]
    fn plane_rotation_report_is_the_owner_result_field_for_field() {
        let matrix = two_plane_rotation(0.7, 1.9);
        let declared = orthogonality_declaration(&matrix);
        let direct = recover_plane_rotations(matrix.view(), declared).expect("owner recovery");
        let output = run_parameter_decomposition(&plane_request(declared), &inputs("w", matrix))
            .expect("surface run");
        let MpdResult::RecoverPlaneRotations(report) = &output.report.result;

        assert_eq!(report.tensor, "w");
        assert_eq!(report.orthogonality_defect, direct.orthogonality_defect);
        assert_eq!(report.perturbation_bound, direct.perturbation_bound);
        assert_eq!(report.clusters.len(), direct.clusters.len());
        let rotations = report
            .clusters
            .iter()
            .filter(|cluster| matches!(cluster.structure, RotationClusterStructure::Rotation { .. }))
            .count();
        assert_eq!(rotations, 2, "two planted planes with distinct cosines");
        for (wire, owner) in report.clusters.iter().zip(&direct.clusters) {
            assert_eq!(output.arrays[&wire.basis], owner.basis.clone().into_dyn());
            assert_eq!(wire.dimension, owner.basis.ncols());
            assert_eq!(
                wire.cosine_interval,
                [owner.cosine_interval.0, owner.cosine_interval.1]
            );
            assert_eq!(
                wire.separation,
                (owner.separation != f64::INFINITY).then_some(owner.separation)
            );
            assert_eq!(
                wire.projector_bar,
                (owner.projector_bar != f64::INFINITY).then_some(owner.projector_bar)
            );
            match (&wire.structure, &owner.kind) {
                (
                    RotationClusterStructure::Fixed { max_hidden_angle },
                    RotationClusterKind::Fixed {
                        max_hidden_angle: owner_angle,
                    },
                ) => assert_eq!(max_hidden_angle, owner_angle),
                (
                    RotationClusterStructure::HalfTurn { min_hidden_angle },
                    RotationClusterKind::HalfTurn {
                        min_hidden_angle: owner_angle,
                    },
                ) => assert_eq!(min_hidden_angle, owner_angle),
                (RotationClusterStructure::Unresolved, RotationClusterKind::Unresolved) => assert!(
                    wire.cosine_interval[0] <= -1.0 && wire.cosine_interval[1] >= 1.0,
                    "an unresolved cluster admits both +1 and -1"
                ),
                (
                    RotationClusterStructure::Rotation {
                        planes,
                        angle,
                        complex_structure,
                    },
                    RotationClusterKind::Rotation {
                        planes: owner_planes,
                        angle: owner_angle,
                        complex_structure: owner_structure,
                    },
                ) => {
                    assert_eq!(planes, owner_planes);
                    assert_eq!(angle, owner_angle);
                    assert_eq!(
                        complex_structure.is_some(),
                        owner_structure.is_some(),
                        "orientation certificate changed on the wire"
                    );
                    if let (Some(id), Some(j)) = (complex_structure, owner_structure) {
                        assert_eq!(output.arrays[id], j.clone().into_dyn());
                    }
                }
                (wire_structure, owner_kind) => panic!(
                    "cluster structure changed on the wire: {wire_structure:?} vs {owner_kind:?}"
                ),
            }
        }
        let expected: Vec<RotationAmbiguityReport> = direct
            .ambiguities()
            .into_iter()
            .map(RotationAmbiguityReport::from)
            .collect();
        assert_eq!(report.ambiguities, expected);

        // Every array the report names is supplied, and nothing else is.
        let mut named: Vec<&String> = report.clusters.iter().map(|cluster| &cluster.basis).collect();
        for cluster in &report.clusters {
            if let RotationClusterStructure::Rotation {
                complex_structure: Some(id),
                ..
            } = &cluster.structure
            {
                named.push(id);
            }
        }
        named.sort();
        assert_eq!(named, output.arrays.keys().collect::<Vec<_>>());

        let json: serde_json::Value =
            serde_json::from_str(&output.report_json().expect("report json")).expect("parse report");
        assert_eq!(json["schema"], MPD_REPORT_SCHEMA);
        assert_eq!(json["result"]["kind"], "recover_plane_rotations");
    }

    #[test]
    fn identity_reports_its_ambiguity_and_an_absent_separation() {
        let output = run_parameter_decomposition(&plane_request(0.0), &inputs("w", Array2::eye(3)))
            .expect("surface run on the identity");
        let MpdResult::RecoverPlaneRotations(report) = &output.report.result;
        assert_eq!(report.ambiguities, vec![RotationAmbiguityReport::Identity]);
        assert_eq!(report.clusters.len(), 1);
        assert_eq!(report.clusters[0].separation, None);
        // The whole spectrum's projector is the identity, a bar of exactly 0, not an
        // absent one.
        assert_eq!(report.clusters[0].projector_bar, Some(0.0));
        let json: serde_json::Value =
            serde_json::from_str(&output.report_json().expect("report json")).expect("parse report");
        assert!(json["result"]["clusters"][0]["separation"].is_null());
        assert_eq!(json["result"]["clusters"][0]["projector_bar"], 0.0);
    }

    #[test]
    fn the_owners_refusal_of_a_non_orthogonal_matrix_reaches_the_caller() {
        // 2 I is 1 away from the orthogonal group, far beyond a declared error of 0.
        assert!(matches!(
            run_parameter_decomposition(&plane_request(0.0), &inputs("w", Array2::eye(2) * 2.0)),
            Err(MpdSurfaceError::PlaneRotation(
                PlaneRotationError::NotOrthogonal { .. }
            ))
        ));
        // Positive control: the same request on an orthogonal matrix is accepted.
        assert!(run_parameter_decomposition(&plane_request(0.0), &inputs("w", Array2::eye(2))).is_ok());
    }

    #[test]
    fn missing_or_misshapen_inputs_are_refused() {
        let json = plane_request(0.0);
        assert!(run_parameter_decomposition(&json, &inputs("w", Array2::eye(2))).is_ok());
        assert!(matches!(
            run_parameter_decomposition(&json, &inputs("other", Array2::eye(2))),
            Err(MpdSurfaceError::MissingTensor { .. })
        ));
        let cube = BTreeMap::from([("w".to_string(), ArrayD::<f64>::zeros(vec![2, 2, 2]))]);
        assert!(matches!(
            run_parameter_decomposition(&json, &cube),
            Err(MpdSurfaceError::TensorShape { .. })
        ));
    }

    #[test]
    fn only_an_owners_infinite_no_such_quantity_becomes_absent() {
        assert_eq!(absent_when_infinite("x", 0.5).expect("finite"), Some(0.5));
        assert_eq!(absent_when_infinite("x", f64::INFINITY).expect("+inf"), None);
        assert!(absent_when_infinite("x", f64::NEG_INFINITY).is_err());
        assert!(absent_when_infinite("x", f64::NAN).is_err());
        assert!(finite("x", f64::INFINITY).is_err());
        assert_eq!(finite("x", -2.0).expect("finite"), -2.0);
    }
}
