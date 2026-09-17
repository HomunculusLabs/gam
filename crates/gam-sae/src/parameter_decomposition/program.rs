//! The mechanism program (#2951): a typed graph of sums, compositions, native
//! primitives, reads and writes, and calls to shared bodies, executed against
//! its native reference.
//!
//! # The graph
//!
//! A [`Program`] is a set of [`Body`]s and an entry body. A body is a DAG whose
//! nodes are listed in topological order: a node may only read nodes listed
//! before it. Values are row-valued matrices (rows are positions, columns are
//! widths). A body has input ports, one output node, and may read and write the
//! program's named state slots (a residual stream, a cache).
//!
//! * [`Node::Native`] runs one of the source's primitives on the source's
//!   tensors, bound through a [`ParameterSource`].
//! * [`Node::Sum`] is `y = sum_i m_i x_i`: mechanisms that add.
//! * [`Node::Compose`] applies stage bodies in sequence, each stage deleted along
//!   the straight path `x -> x + m (f(x) - x)`. For linear stages
//!   `f_k = I + Delta_k` the output is `prod_k (I + m_k Delta_k) x`, so the cross
//!   term `m_A m_B Delta_B Delta_A` is induced by execution and has no control of
//!   its own.
//! * [`Node::Call`] invokes a shared body. Every call site is an invocation, and a
//!   control inside the body can be set at every invocation or at one.
//! * [`Node::Refine`] pairs a mechanism body with the native body it decomposes.
//!   When every control the mechanism reaches resolves to exactly `1` at this
//!   invocation, the native body runs on its original path, so the all-on program
//!   is bit-identical to the source; otherwise the mechanism runs.
//!
//! # Controls, mask groups and macros
//!
//! A control is one masking site: a sum term, a composition stage, one component
//! coordinate of a [`NativePrimitive::CoordinateMask`], or the component anchor
//! of a parameter. Every control belongs to exactly one [`MaskGroup`], and a
//! group is the unit of assignment: tied controls take one value. A macro is a
//! different object, a called body whose internal controls stay independent: a
//! [`MaskAssignment`] scoped to one invocation changes that call only. A group
//! ties controls; a call shares structure without tying anything.
//!
//! A mask value is any finite real (continuous, binary or signed). An unassigned
//! group is `1`, the identity intervention, not a declared mask domain: the
//! domain an experiment searches over is declared by that experiment.

use gam_math::gaussian_activation::{
    GaussianActivation, GaussianActivationError, gaussian_smoothing_derivatives,
};
use ndarray::{Array1, Array2, ArrayView2};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt;

/// The serialized schema tag of a [`Program`] document.
pub const MECHANISM_PROGRAM_SCHEMA: &str = "gamfit.MechanismProgram/v1";

/// Index of a body in the program.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct BodyId(pub u32);

/// Index of a node inside its body.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct NodeId(pub u32);

/// Index of a named state slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct SlotId(pub u32);

/// Index of a formal parameter: a tensor the program reads, bound to the source's
/// tensors by a [`ParameterSource`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ParameterSlot(pub u32);

/// Index of a control (one masking site).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ControlId(pub u32);

/// Index of a mask group (tied controls).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct MaskGroupId(pub u32);

impl BodyId {
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

impl NodeId {
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

impl SlotId {
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

impl ParameterSlot {
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

impl ControlId {
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

impl MaskGroupId {
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

/// One step of an invocation path: the call site `node` in `body`, and for a
/// composition the stage index (`0` for a call or a refinement).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct CallSite {
    pub body: BodyId,
    pub node: NodeId,
    pub stage: u32,
}

/// The source's elementwise activation, evaluated by the gam-math owner.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum NativeActivation {
    /// `max(t, 0)`.
    Relu,
    /// The exact GELU `t Phi(t)`.
    ExactGelu,
}

impl NativeActivation {
    fn owner(self) -> GaussianActivation {
        match self {
            Self::Relu => GaussianActivation::Relu,
            Self::ExactGelu => GaussianActivation::ExactGelu,
        }
    }
}

/// A primitive of the source network.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum NativePrimitive {
    /// Rows mapped through a parameter matrix: `y = x Theta^T`.
    Linear { weight: ParameterSlot },
    /// A parameter vector added to every row.
    AddBias { bias: ParameterSlot },
    /// The source's elementwise activation.
    Activation { activation: NativeActivation },
    /// The elementwise product of two values of one shape.
    Hadamard,
    /// A diagonal mask on component coordinates: column `c` is multiplied by the
    /// value of `controls[c]`. By the mask gauge (P1) only a group mask `cI` on a
    /// declared group, not independent diagonal entries in an arbitrary basis, is
    /// an intrinsic intervention; tie the coordinates with a [`MaskGroup`].
    CoordinateMask { controls: Vec<ControlId> },
}

impl NativePrimitive {
    fn arity(&self) -> usize {
        match self {
            Self::Hadamard => 2,
            Self::Linear { .. }
            | Self::AddBias { .. }
            | Self::Activation { .. }
            | Self::CoordinateMask { .. } => 1,
        }
    }
}

/// One term of a [`Node::Sum`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SumTerm {
    pub value: NodeId,
    pub control: Option<ControlId>,
}

/// One stage of a [`Node::Compose`]: a one-input body and its deletion control.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ComposeStage {
    pub body: BodyId,
    pub control: Option<ControlId>,
}

/// A node of a body.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Node {
    /// The body's input `port`.
    Input { port: u32 },
    /// The current value of a state slot.
    Read { slot: SlotId },
    /// Stores `value` in a state slot; the node's value is `value`.
    Write { slot: SlotId, value: NodeId },
    /// A source primitive applied to `arguments`.
    Native {
        primitive: NativePrimitive,
        arguments: Vec<NodeId>,
    },
    /// `sum_i m_i x_i` over the terms, in term order.
    Sum { terms: Vec<SumTerm> },
    /// The stages applied in order to `value`, stage `k` as
    /// `x -> x + m_k (f_k(x) - x)`.
    Compose {
        value: NodeId,
        stages: Vec<ComposeStage>,
    },
    /// An invocation of a shared body.
    Call {
        body: BodyId,
        arguments: Vec<NodeId>,
    },
    /// A mechanism body and the native body it decomposes, on the same arguments.
    Refine {
        native: BodyId,
        mechanism: BodyId,
        arguments: Vec<NodeId>,
    },
}

impl Node {
    /// The nodes this node reads, in argument order.
    pub fn arguments(&self) -> Vec<NodeId> {
        match self {
            Self::Input { .. } | Self::Read { .. } => Vec::new(),
            Self::Write { value, .. } | Self::Compose { value, .. } => vec![*value],
            Self::Native { arguments, .. }
            | Self::Call { arguments, .. }
            | Self::Refine { arguments, .. } => arguments.clone(),
            Self::Sum { terms } => terms.iter().map(|term| term.value).collect(),
        }
    }
}

/// A DAG of nodes with input ports and one output node.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Body {
    pub name: String,
    pub inputs: u32,
    pub nodes: Vec<Node>,
    pub output: NodeId,
}

/// A formal parameter and the controls of its component anchor. A parameter with
/// controls is affine in them (the anchor `m_Delta Theta_* + B sum_c (m_c -
/// m_Delta) v_c`); the binding applies it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ParameterDecl {
    pub name: String,
    pub controls: Vec<ControlId>,
}

/// A named state slot.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SlotDecl {
    pub name: String,
}

/// A named control.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ControlDecl {
    pub name: String,
}

/// Controls tied to one assigned value.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MaskGroup {
    pub name: String,
    pub controls: Vec<ControlId>,
}

/// The unvalidated parts of a program; [`Program::new`] validates them.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProgramParts {
    pub parameters: Vec<ParameterDecl>,
    pub slots: Vec<SlotDecl>,
    pub controls: Vec<ControlDecl>,
    pub mask_groups: Vec<MaskGroup>,
    pub bodies: Vec<Body>,
    pub entry: BodyId,
}

/// A refused program, assignment, execution or document.
#[derive(Clone, Debug, PartialEq)]
pub enum ProgramError {
    UnknownBody { body: BodyId },
    UnknownSlot { slot: SlotId },
    UnknownParameter { parameter: ParameterSlot },
    UnknownControl { control: ControlId },
    UnknownMaskGroup { group: MaskGroupId },
    UnknownInputPort { body: BodyId, node: NodeId, port: u32 },
    UnknownOutput { body: BodyId, output: NodeId },
    /// A node reads a node that is not listed before it.
    ForwardReference { body: BodyId, node: NodeId, argument: NodeId },
    PrimitiveArity { body: BodyId, node: NodeId, expected: usize, found: usize },
    CallArity { body: BodyId, node: NodeId, callee: BodyId, expected: usize, found: usize },
    EmptySum { body: BodyId, node: NodeId },
    EmptyCompose { body: BodyId, node: NodeId },
    /// The call graph has a cycle through `body`.
    RecursiveCall { body: BodyId },
    ControlUnused { control: ControlId },
    ControlUsedTwice { control: ControlId },
    ControlUngrouped { control: ControlId },
    ControlInTwoGroups { control: ControlId },
    EmptyMaskGroup { group: MaskGroupId },
    /// A composition stage or refinement body writes a state slot, so deleting the
    /// stage or choosing the native path would change which writes happen.
    ImpureBody { body: BodyId, node: NodeId, callee: BodyId },
    /// A refinement's native body reaches a control or another refinement.
    NativeReferenceNotNative { body: BodyId, node: NodeId, native: BodyId },
    InputCount { body: BodyId, expected: usize, found: usize },
    SlotCount { expected: usize, found: usize },
    SlotUnwritten { slot: SlotId },
    ShapeMismatch { body: BodyId, node: NodeId, left: (usize, usize), right: (usize, usize) },
    WidthMismatch { body: BodyId, node: NodeId, expected: usize, found: usize },
    RowCountChanged { body: BodyId, node: NodeId, expected: usize, found: usize },
    Activation { body: BodyId, node: NodeId, error: GaussianActivationError },
    NonFiniteMask { group: MaskGroupId, value: f64 },
    DuplicateMaskScope { group: MaskGroupId },
    /// An executor invariant: a value was released before its last use.
    ValueReleased { body: BodyId, node: NodeId },
    SchemaMismatch { found: String },
    Json { message: String },
}

impl fmt::Display for ProgramError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownBody { body } => write!(formatter, "no body {}", body.0),
            Self::UnknownSlot { slot } => write!(formatter, "no state slot {}", slot.0),
            Self::UnknownParameter { parameter } => {
                write!(formatter, "no formal parameter {}", parameter.0)
            }
            Self::UnknownControl { control } => write!(formatter, "no control {}", control.0),
            Self::UnknownMaskGroup { group } => write!(formatter, "no mask group {}", group.0),
            Self::UnknownInputPort { body, node, port } => write!(
                formatter,
                "node {} of body {} reads input port {port}, which the body does not declare",
                node.0, body.0
            ),
            Self::UnknownOutput { body, output } => {
                write!(formatter, "body {} outputs missing node {}", body.0, output.0)
            }
            Self::ForwardReference { body, node, argument } => write!(
                formatter,
                "node {} of body {} reads node {}, which is not listed before it",
                node.0, body.0, argument.0
            ),
            Self::PrimitiveArity { body, node, expected, found } => write!(
                formatter,
                "native node {} of body {} takes {expected} arguments, got {found}",
                node.0, body.0
            ),
            Self::CallArity { body, node, callee, expected, found } => write!(
                formatter,
                "node {} of body {} passes {found} values to body {}, which takes {expected}",
                node.0, body.0, callee.0
            ),
            Self::EmptySum { body, node } => {
                write!(formatter, "sum node {} of body {} has no terms", node.0, body.0)
            }
            Self::EmptyCompose { body, node } => {
                write!(formatter, "compose node {} of body {} has no stages", node.0, body.0)
            }
            Self::RecursiveCall { body } => {
                write!(formatter, "the call graph has a cycle through body {}", body.0)
            }
            Self::ControlUnused { control } => {
                write!(formatter, "control {} is used at no site", control.0)
            }
            Self::ControlUsedTwice { control } => write!(
                formatter,
                "control {} is used at two sites; tie two sites with a mask group instead",
                control.0
            ),
            Self::ControlUngrouped { control } => {
                write!(formatter, "control {} belongs to no mask group", control.0)
            }
            Self::ControlInTwoGroups { control } => {
                write!(formatter, "control {} belongs to two mask groups", control.0)
            }
            Self::EmptyMaskGroup { group } => {
                write!(formatter, "mask group {} ties no controls", group.0)
            }
            Self::ImpureBody { body, node, callee } => write!(
                formatter,
                "node {} of body {} deletes or refines body {}, which writes a state slot",
                node.0, body.0, callee.0
            ),
            Self::NativeReferenceNotNative { body, node, native } => write!(
                formatter,
                "refinement node {} of body {} names body {} as native, but it reaches a control or a refinement",
                node.0, body.0, native.0
            ),
            Self::InputCount { body, expected, found } => write!(
                formatter,
                "body {} takes {expected} inputs, got {found}",
                body.0
            ),
            Self::SlotCount { expected, found } => write!(
                formatter,
                "the program declares {expected} state slots, got {found} initial values"
            ),
            Self::SlotUnwritten { slot } => {
                write!(formatter, "state slot {} is read before any value is written", slot.0)
            }
            Self::ShapeMismatch { body, node, left, right } => write!(
                formatter,
                "node {} of body {} combines shapes {left:?} and {right:?}",
                node.0, body.0
            ),
            Self::WidthMismatch { body, node, expected, found } => write!(
                formatter,
                "node {} of body {} needs width {expected}, got {found}",
                node.0, body.0
            ),
            Self::RowCountChanged { body, node, expected, found } => write!(
                formatter,
                "native node {} of body {} returned {found} rows for {expected} input rows",
                node.0, body.0
            ),
            Self::Activation { body, node, error } => write!(
                formatter,
                "activation node {} of body {}: {error}",
                node.0, body.0
            ),
            Self::NonFiniteMask { group, value } => {
                write!(formatter, "mask group {} was assigned non-finite {value}", group.0)
            }
            Self::DuplicateMaskScope { group } => write!(
                formatter,
                "mask group {} is assigned twice at one invocation scope",
                group.0
            ),
            Self::ValueReleased { body, node } => write!(
                formatter,
                "executor invariant: node {} of body {} was released before its last use",
                node.0, body.0
            ),
            Self::SchemaMismatch { found } => write!(
                formatter,
                "expected schema {MECHANISM_PROGRAM_SCHEMA}, found {found}"
            ),
            Self::Json { message } => write!(formatter, "program document: {message}"),
        }
    }
}

impl std::error::Error for ProgramError {}

/// A validated program.
#[derive(Clone, Debug, PartialEq)]
pub struct Program {
    parts: ProgramParts,
    control_group: Vec<MaskGroupId>,
    use_counts: Vec<Vec<u32>>,
    controls_below: Vec<Vec<ControlId>>,
}

impl Program {
    /// Validates the graph: ids in range, topological node order, arities, an
    /// acyclic call graph, every control used at exactly one site and in exactly
    /// one group, composition stages and refinements that write no slots, and
    /// native bodies that reach no control.
    pub fn new(parts: ProgramParts) -> Result<Self, ProgramError> {
        let body_count = parts.bodies.len();
        callee_body(&parts, parts.entry)?;
        let mut control_sites = vec![0u32; parts.controls.len()];
        for decl in &parts.parameters {
            for &control in &decl.controls {
                record_control_site(&mut control_sites, control)?;
            }
        }
        let mut use_counts = Vec::with_capacity(body_count);
        let mut callees: Vec<Vec<BodyId>> = Vec::with_capacity(body_count);
        let mut controls_below: Vec<Vec<ControlId>> = Vec::with_capacity(body_count);
        let mut writes_below = Vec::with_capacity(body_count);
        let mut refines_below = Vec::with_capacity(body_count);
        for (b, body) in parts.bodies.iter().enumerate() {
            let body_id = BodyId(b as u32);
            let mut counts = vec![0u32; body.nodes.len()];
            let mut body_callees = Vec::new();
            let mut controls = Vec::new();
            let mut writes = false;
            let mut refines = false;
            for (i, node) in body.nodes.iter().enumerate() {
                let node_id = NodeId(i as u32);
                for argument in node.arguments() {
                    if argument.index() >= i {
                        return Err(ProgramError::ForwardReference {
                            body: body_id,
                            node: node_id,
                            argument,
                        });
                    }
                    counts[argument.index()] += 1;
                }
                match node {
                    Node::Input { port } => {
                        if *port >= body.inputs {
                            return Err(ProgramError::UnknownInputPort {
                                body: body_id,
                                node: node_id,
                                port: *port,
                            });
                        }
                    }
                    Node::Read { slot } => check_slot(&parts, *slot)?,
                    Node::Write { slot, .. } => {
                        check_slot(&parts, *slot)?;
                        writes = true;
                    }
                    Node::Native { primitive, arguments } => {
                        if arguments.len() != primitive.arity() {
                            return Err(ProgramError::PrimitiveArity {
                                body: body_id,
                                node: node_id,
                                expected: primitive.arity(),
                                found: arguments.len(),
                            });
                        }
                        match primitive {
                            NativePrimitive::Linear { weight: parameter }
                            | NativePrimitive::AddBias { bias: parameter } => {
                                let decl = parts.parameters.get(parameter.index()).ok_or(
                                    ProgramError::UnknownParameter { parameter: *parameter },
                                )?;
                                controls.extend(decl.controls.iter().copied());
                            }
                            NativePrimitive::CoordinateMask { controls: mask } => {
                                for &control in mask {
                                    record_control_site(&mut control_sites, control)?;
                                    controls.push(control);
                                }
                            }
                            NativePrimitive::Activation { .. } | NativePrimitive::Hadamard => {}
                        }
                    }
                    Node::Sum { terms } => {
                        if terms.is_empty() {
                            return Err(ProgramError::EmptySum { body: body_id, node: node_id });
                        }
                        for control in terms.iter().filter_map(|term| term.control) {
                            record_control_site(&mut control_sites, control)?;
                            controls.push(control);
                        }
                    }
                    Node::Compose { stages, .. } => {
                        if stages.is_empty() {
                            return Err(ProgramError::EmptyCompose {
                                body: body_id,
                                node: node_id,
                            });
                        }
                        for stage in stages {
                            let inputs = callee_body(&parts, stage.body)?.inputs as usize;
                            if inputs != 1 {
                                return Err(ProgramError::CallArity {
                                    body: body_id,
                                    node: node_id,
                                    callee: stage.body,
                                    expected: inputs,
                                    found: 1,
                                });
                            }
                            body_callees.push(stage.body);
                            if let Some(control) = stage.control {
                                record_control_site(&mut control_sites, control)?;
                                controls.push(control);
                            }
                        }
                    }
                    Node::Call { body: callee, arguments } => {
                        let inputs = callee_body(&parts, *callee)?.inputs as usize;
                        if inputs != arguments.len() {
                            return Err(ProgramError::CallArity {
                                body: body_id,
                                node: node_id,
                                callee: *callee,
                                expected: inputs,
                                found: arguments.len(),
                            });
                        }
                        body_callees.push(*callee);
                    }
                    Node::Refine { native, mechanism, arguments } => {
                        for callee in [*native, *mechanism] {
                            let inputs = callee_body(&parts, callee)?.inputs as usize;
                            if inputs != arguments.len() {
                                return Err(ProgramError::CallArity {
                                    body: body_id,
                                    node: node_id,
                                    callee,
                                    expected: inputs,
                                    found: arguments.len(),
                                });
                            }
                            body_callees.push(callee);
                        }
                        refines = true;
                    }
                }
            }
            let output_count = counts.get_mut(body.output.index()).ok_or(
                ProgramError::UnknownOutput { body: body_id, output: body.output },
            )?;
            *output_count += 1;
            use_counts.push(counts);
            callees.push(body_callees);
            controls_below.push(controls);
            writes_below.push(writes);
            refines_below.push(refines);
        }
        if let Some(unused) = control_sites.iter().position(|&sites| sites == 0) {
            return Err(ProgramError::ControlUnused { control: ControlId(unused as u32) });
        }
        let mut group_of: Vec<Option<MaskGroupId>> = vec![None; parts.controls.len()];
        for (g, group) in parts.mask_groups.iter().enumerate() {
            let group_id = MaskGroupId(g as u32);
            if group.controls.is_empty() {
                return Err(ProgramError::EmptyMaskGroup { group: group_id });
            }
            for &control in &group.controls {
                let entry = group_of
                    .get_mut(control.index())
                    .ok_or(ProgramError::UnknownControl { control })?;
                if entry.is_some() {
                    return Err(ProgramError::ControlInTwoGroups { control });
                }
                *entry = Some(group_id);
            }
        }
        let control_group = group_of
            .iter()
            .enumerate()
            .map(|(c, group)| {
                group.ok_or(ProgramError::ControlUngrouped { control: ControlId(c as u32) })
            })
            .collect::<Result<Vec<_>, _>>()?;
        for body in call_graph_post_order(&callees)? {
            let b = body.index();
            for callee in callees[b].iter().map(|callee| callee.index()) {
                writes_below[b] = writes_below[b] || writes_below[callee];
                refines_below[b] = refines_below[b] || refines_below[callee];
                let below = controls_below[callee].clone();
                controls_below[b].extend(below);
            }
            controls_below[b].sort_unstable();
            controls_below[b].dedup();
        }
        for (b, body) in parts.bodies.iter().enumerate() {
            let body_id = BodyId(b as u32);
            for (i, node) in body.nodes.iter().enumerate() {
                let node_id = NodeId(i as u32);
                let pure_callees: Vec<BodyId> = match node {
                    Node::Compose { stages, .. } => stages.iter().map(|stage| stage.body).collect(),
                    Node::Refine { native, mechanism, .. } => {
                        if !controls_below[native.index()].is_empty()
                            || refines_below[native.index()]
                        {
                            return Err(ProgramError::NativeReferenceNotNative {
                                body: body_id,
                                node: node_id,
                                native: *native,
                            });
                        }
                        vec![*native, *mechanism]
                    }
                    Node::Input { .. }
                    | Node::Read { .. }
                    | Node::Write { .. }
                    | Node::Native { .. }
                    | Node::Sum { .. }
                    | Node::Call { .. } => Vec::new(),
                };
                if let Some(callee) = pure_callees.into_iter().find(|callee| writes_below[callee.index()]) {
                    return Err(ProgramError::ImpureBody { body: body_id, node: node_id, callee });
                }
            }
        }
        Ok(Self { parts, control_group, use_counts, controls_below })
    }

    /// The validated parts.
    pub fn parts(&self) -> &ProgramParts {
        &self.parts
    }

    /// The mask group a control belongs to.
    pub fn mask_group_of(&self, control: ControlId) -> Option<MaskGroupId> {
        self.control_group.get(control.index()).copied()
    }

    /// Every control a body reaches, directly or through the bodies it calls,
    /// composes or refines, in increasing order.
    pub fn controls_reached(&self, body: BodyId) -> Option<&[ControlId]> {
        self.controls_below.get(body.index()).map(Vec::as_slice)
    }
}

fn callee_body(parts: &ProgramParts, body: BodyId) -> Result<&Body, ProgramError> {
    parts.bodies.get(body.index()).ok_or(ProgramError::UnknownBody { body })
}

fn check_slot(parts: &ProgramParts, slot: SlotId) -> Result<(), ProgramError> {
    if slot.index() < parts.slots.len() {
        Ok(())
    } else {
        Err(ProgramError::UnknownSlot { slot })
    }
}

fn record_control_site(sites: &mut [u32], control: ControlId) -> Result<(), ProgramError> {
    let count = sites
        .get_mut(control.index())
        .ok_or(ProgramError::UnknownControl { control })?;
    if *count > 0 {
        return Err(ProgramError::ControlUsedTwice { control });
    }
    *count = 1;
    Ok(())
}

/// Callees before callers, refusing a cycle.
fn call_graph_post_order(callees: &[Vec<BodyId>]) -> Result<Vec<BodyId>, ProgramError> {
    const UNVISITED: u8 = 0;
    const ON_STACK: u8 = 1;
    let mut state = vec![UNVISITED; callees.len()];
    let mut order = Vec::with_capacity(callees.len());
    for root in 0..callees.len() {
        if state[root] != UNVISITED {
            continue;
        }
        state[root] = ON_STACK;
        let mut stack: Vec<(usize, usize)> = vec![(root, 0)];
        while let Some(frame) = stack.last_mut() {
            let (body, next) = *frame;
            if next < callees[body].len() {
                frame.1 += 1;
                let callee = callees[body][next].index();
                if state[callee] == ON_STACK {
                    return Err(ProgramError::RecursiveCall { body: BodyId(callee as u32) });
                }
                if state[callee] == UNVISITED {
                    state[callee] = ON_STACK;
                    stack.push((callee, 0));
                }
            } else {
                state[body] = ON_STACK + 1;
                order.push(BodyId(body as u32));
                stack.pop();
            }
        }
    }
    Ok(order)
}

/// A formal parameter at one use: the node that reads it, the invocation it runs
/// in, and the values its anchor controls resolve to there, in the order of
/// [`ParameterDecl::controls`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ParameterUse<'a> {
    pub parameter: ParameterSlot,
    pub body: BodyId,
    pub node: NodeId,
    pub invocation: &'a [CallSite],
    pub controls: &'a [f64],
}

/// Binds formal parameters to the source's tensors. The rows handed to
/// [`ParameterSource::apply_linear`] are the current input of that use, after
/// every upstream intervention, and an implementation applies its tensor to them
/// without materializing an edited copy. At controls all `1` it executes the
/// original tensor.
pub trait ParameterSource {
    type Error: std::error::Error;

    /// `x Theta^T` for the matrix parameter at this use.
    fn apply_linear(
        &self,
        parameter: ParameterUse<'_>,
        rows: ArrayView2<'_, f64>,
    ) -> Result<Array2<f64>, Self::Error>;

    /// The vector parameter at this use.
    fn vector(&self, parameter: ParameterUse<'_>) -> Result<Array1<f64>, Self::Error>;
}

/// A refused execution: the program itself, or the parameter binding at one use.
#[derive(Debug)]
pub enum ExecutionError<E> {
    Program(ProgramError),
    Source {
        body: BodyId,
        node: NodeId,
        parameter: ParameterSlot,
        error: E,
    },
}

impl<E> From<ProgramError> for ExecutionError<E> {
    fn from(error: ProgramError) -> Self {
        Self::Program(error)
    }
}

impl<E: fmt::Display> fmt::Display for ExecutionError<E> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Program(error) => write!(formatter, "{error}"),
            Self::Source { body, node, parameter, error } => write!(
                formatter,
                "parameter {} at node {} of body {}: {error}",
                parameter.0, node.0, body.0
            ),
        }
    }
}

impl<E: std::error::Error> std::error::Error for ExecutionError<E> {}

/// One assigned group value at an invocation scope. The empty scope is global. A
/// scope applies at every invocation path it prefixes, and the deepest applicable
/// scope wins.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScopedMask {
    pub group: MaskGroupId,
    pub scope: Vec<CallSite>,
    pub value: f64,
}

/// Mask group values by invocation scope. A group with no applicable entry is
/// `1`, the identity intervention.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MaskAssignment {
    entries: Vec<ScopedMask>,
}

impl MaskAssignment {
    /// No intervention.
    pub fn all_on() -> Self {
        Self { entries: Vec::new() }
    }

    /// Assigns `value` to `group` at `scope`, refusing a non-finite value and a
    /// second value at the same scope.
    pub fn set(
        &mut self,
        group: MaskGroupId,
        scope: Vec<CallSite>,
        value: f64,
    ) -> Result<(), ProgramError> {
        if !value.is_finite() {
            return Err(ProgramError::NonFiniteMask { group, value });
        }
        if self
            .entries
            .iter()
            .any(|entry| entry.group == group && entry.scope == scope)
        {
            return Err(ProgramError::DuplicateMaskScope { group });
        }
        self.entries.push(ScopedMask { group, scope, value });
        Ok(())
    }

    pub fn entries(&self) -> &[ScopedMask] {
        &self.entries
    }
}

/// The entry body's output and the final state slots.
#[derive(Clone, Debug, PartialEq)]
pub struct Execution {
    pub output: Array2<f64>,
    pub slots: Vec<Option<Array2<f64>>>,
}

/// At one refinement invocation, the mechanism body against the native body on
/// the same arguments. These are measured numbers, not a bound.
#[derive(Clone, Debug, PartialEq)]
pub struct RefinementResidual {
    pub invocation: Vec<CallSite>,
    /// Every control the mechanism reaches resolved to `1`, so the native output
    /// was carried downstream.
    pub all_on: bool,
    /// `max |mechanism - native|` over every entry (NaN if either output has one).
    pub max_abs_difference: f64,
    /// `max |native|` over every entry, the scale of the difference.
    pub native_max_abs: f64,
}

impl Program {
    /// Runs the entry body on `inputs` with state slots starting at `slots`.
    pub fn execute<S: ParameterSource>(
        &self,
        source: &S,
        masks: &MaskAssignment,
        inputs: Vec<Array2<f64>>,
        slots: Vec<Option<Array2<f64>>>,
    ) -> Result<Execution, ExecutionError<S::Error>> {
        let mut executor = Executor::new(self, source, masks, slots, false)?;
        let output = executor.run(inputs)?;
        Ok(Execution { output, slots: executor.slots })
    }

    /// Runs like [`Program::execute`], and at every refinement invocation also runs
    /// the body that was not carried downstream, reporting the residual between the
    /// mechanism and its native reference.
    pub fn refinement_residuals<S: ParameterSource>(
        &self,
        source: &S,
        masks: &MaskAssignment,
        inputs: Vec<Array2<f64>>,
        slots: Vec<Option<Array2<f64>>>,
    ) -> Result<(Execution, Vec<RefinementResidual>), ExecutionError<S::Error>> {
        let mut executor = Executor::new(self, source, masks, slots, true)?;
        let output = executor.run(inputs)?;
        let residuals = executor.residuals.take().unwrap_or_default();
        Ok((Execution { output, slots: executor.slots }, residuals))
    }
}

struct Executor<'a, S> {
    program: &'a Program,
    source: &'a S,
    masks: BTreeMap<MaskGroupId, Vec<(&'a [CallSite], f64)>>,
    slots: Vec<Option<Array2<f64>>>,
    residuals: Option<Vec<RefinementResidual>>,
}

impl<'a, S: ParameterSource> Executor<'a, S> {
    fn new(
        program: &'a Program,
        source: &'a S,
        masks: &'a MaskAssignment,
        slots: Vec<Option<Array2<f64>>>,
        record_residuals: bool,
    ) -> Result<Self, ProgramError> {
        if slots.len() != program.parts.slots.len() {
            return Err(ProgramError::SlotCount {
                expected: program.parts.slots.len(),
                found: slots.len(),
            });
        }
        let mut index: BTreeMap<MaskGroupId, Vec<(&'a [CallSite], f64)>> = BTreeMap::new();
        for entry in &masks.entries {
            if entry.group.index() >= program.parts.mask_groups.len() {
                return Err(ProgramError::UnknownMaskGroup { group: entry.group });
            }
            if !entry.value.is_finite() {
                return Err(ProgramError::NonFiniteMask { group: entry.group, value: entry.value });
            }
            let scoped = index.entry(entry.group).or_default();
            if scoped.iter().any(|assigned| assigned.0 == entry.scope.as_slice()) {
                return Err(ProgramError::DuplicateMaskScope { group: entry.group });
            }
            scoped.push((entry.scope.as_slice(), entry.value));
        }
        Ok(Self {
            program,
            source,
            masks: index,
            slots,
            residuals: record_residuals.then(Vec::new),
        })
    }

    fn run(&mut self, inputs: Vec<Array2<f64>>) -> Result<Array2<f64>, ExecutionError<S::Error>> {
        let mut path = Vec::new();
        self.body(self.program.parts.entry, inputs, &mut path)
    }

    /// The value of a control at an invocation path: its group's deepest
    /// applicable scoped value, or `1`.
    fn resolve(&self, control: ControlId, path: &[CallSite]) -> f64 {
        let group = self.program.control_group[control.index()];
        let mut value = 1.0;
        let mut deepest: Option<usize> = None;
        if let Some(scoped) = self.masks.get(&group) {
            for &(scope, assigned) in scoped {
                if path.starts_with(scope) && deepest.is_none_or(|depth| scope.len() > depth) {
                    value = assigned;
                    deepest = Some(scope.len());
                }
            }
        }
        value
    }

    fn parameter_controls(&self, parameter: ParameterSlot, path: &[CallSite]) -> Vec<f64> {
        self.program.parts.parameters[parameter.index()]
            .controls
            .iter()
            .map(|&control| self.resolve(control, path))
            .collect()
    }

    fn body(
        &mut self,
        body_id: BodyId,
        inputs: Vec<Array2<f64>>,
        path: &mut Vec<CallSite>,
    ) -> Result<Array2<f64>, ExecutionError<S::Error>> {
        let program = self.program;
        let body = &program.parts.bodies[body_id.index()];
        if inputs.len() != body.inputs as usize {
            return Err(ProgramError::InputCount {
                body: body_id,
                expected: body.inputs as usize,
                found: inputs.len(),
            }
            .into());
        }
        let mut remaining = program.use_counts[body_id.index()].clone();
        let mut values: Vec<Option<Array2<f64>>> = vec![None; body.nodes.len()];
        for (i, node) in body.nodes.iter().enumerate() {
            let node_id = NodeId(i as u32);
            let site = CallSite { body: body_id, node: node_id, stage: 0 };
            let value = match node {
                Node::Input { port } => inputs[*port as usize].clone(),
                Node::Read { slot } => self.slots[slot.index()]
                    .clone()
                    .ok_or(ProgramError::SlotUnwritten { slot: *slot })?,
                Node::Write { slot, value } => {
                    let written = argument(&values, body_id, *value)?.clone();
                    self.slots[slot.index()] = Some(written.clone());
                    written
                }
                Node::Native { primitive, arguments } => {
                    self.native(site, primitive, arguments, &values, path)?
                }
                Node::Sum { terms } => self.sum(site, terms, &values, path)?,
                Node::Compose { value, stages } => {
                    let mut current = argument(&values, body_id, *value)?.clone();
                    for (k, stage) in stages.iter().enumerate() {
                        let m = stage
                            .control
                            .map_or(1.0, |control| self.resolve(control, path));
                        if m == 0.0 {
                            continue;
                        }
                        path.push(CallSite { stage: k as u32, ..site });
                        let staged = self.body(stage.body, vec![current.clone()], path);
                        path.pop();
                        let staged = staged?;
                        if staged.dim() != current.dim() {
                            return Err(ProgramError::ShapeMismatch {
                                body: body_id,
                                node: node_id,
                                left: current.dim(),
                                right: staged.dim(),
                            }
                            .into());
                        }
                        current = if m == 1.0 {
                            staged
                        } else {
                            let mut interpolated = staged;
                            interpolated -= &current;
                            interpolated *= m;
                            interpolated += &current;
                            interpolated
                        };
                    }
                    current
                }
                Node::Call { body: callee, arguments } => {
                    let passed = collect_arguments(&values, body_id, arguments)?;
                    path.push(site);
                    let called = self.body(*callee, passed, path);
                    path.pop();
                    called?
                }
                Node::Refine { native, mechanism, arguments } => {
                    let passed = collect_arguments(&values, body_id, arguments)?;
                    path.push(site);
                    let refined = self.refine(site, *native, *mechanism, passed, path);
                    path.pop();
                    refined?
                }
            };
            for argument in node.arguments() {
                let count = &mut remaining[argument.index()];
                *count -= 1;
                if *count == 0 {
                    values[argument.index()] = None;
                }
            }
            if remaining[i] > 0 {
                values[i] = Some(value);
            }
        }
        values[body.output.index()].take().ok_or(ExecutionError::Program(
            ProgramError::ValueReleased { body: body_id, node: body.output },
        ))
    }

    fn native(
        &self,
        site: CallSite,
        primitive: &NativePrimitive,
        arguments: &[NodeId],
        values: &[Option<Array2<f64>>],
        path: &[CallSite],
    ) -> Result<Array2<f64>, ExecutionError<S::Error>> {
        let (body, node) = (site.body, site.node);
        let x = argument(values, body, arguments[0])?;
        match primitive {
            NativePrimitive::Linear { weight } => {
                let controls = self.parameter_controls(*weight, path);
                let parameter_use = ParameterUse {
                    parameter: *weight,
                    body,
                    node,
                    invocation: path,
                    controls: &controls,
                };
                let rows = self
                    .source
                    .apply_linear(parameter_use, x.view())
                    .map_err(|error| ExecutionError::Source { body, node, parameter: *weight, error })?;
                if rows.nrows() != x.nrows() {
                    return Err(ProgramError::RowCountChanged {
                        body,
                        node,
                        expected: x.nrows(),
                        found: rows.nrows(),
                    }
                    .into());
                }
                Ok(rows)
            }
            NativePrimitive::AddBias { bias } => {
                let controls = self.parameter_controls(*bias, path);
                let parameter_use = ParameterUse {
                    parameter: *bias,
                    body,
                    node,
                    invocation: path,
                    controls: &controls,
                };
                let vector = self
                    .source
                    .vector(parameter_use)
                    .map_err(|error| ExecutionError::Source { body, node, parameter: *bias, error })?;
                if vector.len() != x.ncols() {
                    return Err(ProgramError::WidthMismatch {
                        body,
                        node,
                        expected: x.ncols(),
                        found: vector.len(),
                    }
                    .into());
                }
                let mut rows = x.clone();
                rows += &vector;
                Ok(rows)
            }
            NativePrimitive::Activation { activation } => {
                let mut rows = x.clone();
                let mut value = [0.0];
                for entry in rows.iter_mut() {
                    gaussian_smoothing_derivatives(activation.owner(), *entry, 0.0, &mut value)
                        .map_err(|error| ProgramError::Activation { body, node, error })?;
                    *entry = value[0];
                }
                Ok(rows)
            }
            NativePrimitive::Hadamard => {
                let y = argument(values, body, arguments[1])?;
                if x.dim() != y.dim() {
                    return Err(ProgramError::ShapeMismatch {
                        body,
                        node,
                        left: x.dim(),
                        right: y.dim(),
                    }
                    .into());
                }
                Ok(x * y)
            }
            NativePrimitive::CoordinateMask { controls } => {
                if x.ncols() != controls.len() {
                    return Err(ProgramError::WidthMismatch {
                        body,
                        node,
                        expected: controls.len(),
                        found: x.ncols(),
                    }
                    .into());
                }
                let mut rows = x.clone();
                for (c, &control) in controls.iter().enumerate() {
                    let m = self.resolve(control, path);
                    if m != 1.0 {
                        let mut column = rows.column_mut(c);
                        column *= m;
                    }
                }
                Ok(rows)
            }
        }
    }

    fn sum(
        &self,
        site: CallSite,
        terms: &[SumTerm],
        values: &[Option<Array2<f64>>],
        path: &[CallSite],
    ) -> Result<Array2<f64>, ExecutionError<S::Error>> {
        let first = argument(values, site.body, terms[0].value)?;
        let mut total: Option<Array2<f64>> = None;
        for term in terms {
            let x = argument(values, site.body, term.value)?;
            if x.dim() != first.dim() {
                return Err(ProgramError::ShapeMismatch {
                    body: site.body,
                    node: site.node,
                    left: first.dim(),
                    right: x.dim(),
                }
                .into());
            }
            let m = term.control.map_or(1.0, |control| self.resolve(control, path));
            if m == 0.0 {
                continue;
            }
            total = Some(match total.take() {
                None if m == 1.0 => x.clone(),
                None => x * m,
                Some(mut accumulated) => {
                    if m == 1.0 {
                        accumulated += x;
                    } else {
                        accumulated.scaled_add(m, x);
                    }
                    accumulated
                }
            });
        }
        Ok(total.unwrap_or_else(|| Array2::zeros(first.dim())))
    }

    fn refine(
        &mut self,
        site: CallSite,
        native: BodyId,
        mechanism: BodyId,
        arguments: Vec<Array2<f64>>,
        path: &mut Vec<CallSite>,
    ) -> Result<Array2<f64>, ExecutionError<S::Error>> {
        let program = self.program;
        let all_on = program.controls_below[mechanism.index()]
            .iter()
            .all(|&control| self.resolve(control, path) == 1.0);
        if self.residuals.is_none() {
            let chosen = if all_on { native } else { mechanism };
            return self.body(chosen, arguments, path);
        }
        let native_output = self.body(native, arguments.clone(), path)?;
        let mechanism_output = self.body(mechanism, arguments, path)?;
        if native_output.dim() != mechanism_output.dim() {
            return Err(ProgramError::ShapeMismatch {
                body: site.body,
                node: site.node,
                left: native_output.dim(),
                right: mechanism_output.dim(),
            }
            .into());
        }
        let max_abs_difference = nan_propagating_max(
            native_output
                .iter()
                .zip(mechanism_output.iter())
                .map(|(native_entry, mechanism_entry)| (mechanism_entry - native_entry).abs()),
        );
        let native_max_abs = nan_propagating_max(native_output.iter().map(|entry| entry.abs()));
        if let Some(residuals) = self.residuals.as_mut() {
            residuals.push(RefinementResidual {
                invocation: path.clone(),
                all_on,
                max_abs_difference,
                native_max_abs,
            });
        }
        Ok(if all_on { native_output } else { mechanism_output })
    }
}

fn argument<E>(
    values: &[Option<Array2<f64>>],
    body: BodyId,
    node: NodeId,
) -> Result<&Array2<f64>, ExecutionError<E>> {
    values[node.index()]
        .as_ref()
        .ok_or(ExecutionError::Program(ProgramError::ValueReleased { body, node }))
}

fn collect_arguments<E>(
    values: &[Option<Array2<f64>>],
    body: BodyId,
    arguments: &[NodeId],
) -> Result<Vec<Array2<f64>>, ExecutionError<E>> {
    arguments
        .iter()
        .map(|&node| argument(values, body, node).cloned())
        .collect()
}

/// The largest value, or NaN once any value is NaN; `0` for no values.
fn nan_propagating_max(values: impl Iterator<Item = f64>) -> f64 {
    values.fold(0.0, |largest, value| {
        if value.is_nan() || value > largest { value } else { largest }
    })
}

/// How a value depends on the mask values: a polynomial through linear
/// operations, or through a nonlinearity.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum MaskDependence {
    /// A polynomial of this total degree in the controls; `0` is mask-free.
    Polynomial { degree: u32 },
    /// A mask-dependent value reaches a nonlinearity.
    Nonlinear,
}

impl MaskDependence {
    const FREE: Self = Self::Polynomial { degree: 0 };

    /// Affine in the controls (degree at most one). The exact affine-logit vertex
    /// adversary (P9) needs this over the whole zonotope. Masks that enter two
    /// multiplied layers, including the induced cross term of a composition, give
    /// degree two.
    pub fn is_affine(self) -> bool {
        matches!(self, Self::Polynomial { degree } if degree <= 1)
    }

    fn product(self, other: Self) -> Self {
        match (self, other) {
            (Self::Polynomial { degree: left }, Self::Polynomial { degree: right }) => {
                Self::Polynomial { degree: left.saturating_add(right) }
            }
            (Self::Nonlinear, _) | (_, Self::Nonlinear) => Self::Nonlinear,
        }
    }

    fn join(self, other: Self) -> Self {
        match (self, other) {
            (Self::Polynomial { degree: left }, Self::Polynomial { degree: right }) => {
                Self::Polynomial { degree: left.max(right) }
            }
            (Self::Nonlinear, _) | (_, Self::Nonlinear) => Self::Nonlinear,
        }
    }

    fn scaled(self, controlled: bool) -> Self {
        if controlled { self.product(Self::Polynomial { degree: 1 }) } else { self }
    }
}

impl Program {
    /// The mask dependence of the entry body's output, with inputs and initial
    /// slots mask-free. A refinement contributes its mechanism body: the native
    /// body it selects at all-on reaches no control.
    pub fn output_mask_dependence(&self) -> MaskDependence {
        let entry = self.parts.entry;
        let inputs = vec![MaskDependence::FREE; self.parts.bodies[entry.index()].inputs as usize];
        let mut slots = vec![MaskDependence::FREE; self.parts.slots.len()];
        self.body_mask_dependence(entry, &inputs, &mut slots)
    }

    fn body_mask_dependence(
        &self,
        body_id: BodyId,
        inputs: &[MaskDependence],
        slots: &mut [MaskDependence],
    ) -> MaskDependence {
        let body = &self.parts.bodies[body_id.index()];
        let mut values: Vec<MaskDependence> = Vec::with_capacity(body.nodes.len());
        for node in &body.nodes {
            let value = match node {
                Node::Input { port } => inputs[*port as usize],
                Node::Read { slot } => slots[slot.index()],
                Node::Write { slot, value } => {
                    slots[slot.index()] = values[value.index()];
                    values[value.index()]
                }
                Node::Native { primitive, arguments } => {
                    let x = values[arguments[0].index()];
                    match primitive {
                        NativePrimitive::Linear { weight } => {
                            x.product(self.parameter_mask_dependence(*weight))
                        }
                        NativePrimitive::AddBias { bias } => {
                            x.join(self.parameter_mask_dependence(*bias))
                        }
                        NativePrimitive::Activation { .. } => {
                            if x == MaskDependence::FREE { x } else { MaskDependence::Nonlinear }
                        }
                        NativePrimitive::Hadamard => x.product(values[arguments[1].index()]),
                        NativePrimitive::CoordinateMask { controls } => {
                            x.scaled(!controls.is_empty())
                        }
                    }
                }
                Node::Sum { terms } => terms.iter().fold(MaskDependence::FREE, |total, term| {
                    total.join(values[term.value.index()].scaled(term.control.is_some()))
                }),
                Node::Compose { value, stages } => {
                    let mut current = values[value.index()];
                    for stage in stages {
                        let staged = self.body_mask_dependence(stage.body, &[current], slots);
                        current = if stage.control.is_some() {
                            current.join(staged).scaled(true)
                        } else {
                            staged
                        };
                    }
                    current
                }
                Node::Call { body: callee, arguments }
                | Node::Refine { mechanism: callee, arguments, .. } => {
                    let passed: Vec<MaskDependence> =
                        arguments.iter().map(|argument| values[argument.index()]).collect();
                    self.body_mask_dependence(*callee, &passed, slots)
                }
            };
            values.push(value);
        }
        values[body.output.index()]
    }

    fn parameter_mask_dependence(&self, parameter: ParameterSlot) -> MaskDependence {
        let controlled = !self.parts.parameters[parameter.index()].controls.is_empty();
        MaskDependence::Polynomial { degree: u32::from(controlled) }
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ProgramDocument {
    schema: String,
    program: ProgramParts,
}

fn json_error(error: serde_json::Error) -> ProgramError {
    ProgramError::Json { message: error.to_string() }
}

impl Program {
    /// The program as a [`MECHANISM_PROGRAM_SCHEMA`] JSON document.
    pub fn to_json(&self) -> Result<String, ProgramError> {
        let document = ProgramDocument {
            schema: MECHANISM_PROGRAM_SCHEMA.to_string(),
            program: self.parts.clone(),
        };
        serde_json::to_string(&document).map_err(json_error)
    }

    /// Reads a document, refusing any other schema tag, unknown fields and an
    /// invalid program. Older schemas are refused, not migrated.
    pub fn from_json(text: &str) -> Result<Self, ProgramError> {
        let value: serde_json::Value = serde_json::from_str(text).map_err(json_error)?;
        let found = value
            .get("schema")
            .and_then(serde_json::Value::as_str)
            .unwrap_or_default();
        if found != MECHANISM_PROGRAM_SCHEMA {
            return Err(ProgramError::SchemaMismatch { found: found.to_string() });
        }
        let document: ProgramDocument = serde_json::from_value(value).map_err(json_error)?;
        Self::new(document.program)
    }
}

/// A dense tensor bound to a formal parameter.
#[derive(Clone, Debug, PartialEq)]
pub enum DenseTensor {
    Matrix(Array2<f64>),
    Vector(Array1<f64>),
}

/// Formal parameters bound to dense tensors with no component structure, the
/// native binding of a small program such as a planted teacher. A parameter with
/// anchor controls needs a component binding and is refused here.
#[derive(Clone, Debug, PartialEq)]
pub struct DenseParameters {
    tensors: Vec<DenseTensor>,
}

/// A refused dense parameter use.
#[derive(Clone, Debug, PartialEq)]
pub enum DenseParameterError {
    Unbound { parameter: ParameterSlot },
    NotAMatrix { parameter: ParameterSlot },
    NotAVector { parameter: ParameterSlot },
    WidthMismatch { parameter: ParameterSlot, expected: usize, found: usize },
    /// A dense tensor has no component anchor for controls to act on.
    Controlled { parameter: ParameterSlot },
}

impl fmt::Display for DenseParameterError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unbound { parameter } => {
                write!(formatter, "formal parameter {} has no dense tensor", parameter.0)
            }
            Self::NotAMatrix { parameter } => {
                write!(formatter, "formal parameter {} is bound to a vector, not a matrix", parameter.0)
            }
            Self::NotAVector { parameter } => {
                write!(formatter, "formal parameter {} is bound to a matrix, not a vector", parameter.0)
            }
            Self::WidthMismatch { parameter, expected, found } => write!(
                formatter,
                "formal parameter {} takes rows of width {expected}, got {found}",
                parameter.0
            ),
            Self::Controlled { parameter } => write!(
                formatter,
                "formal parameter {} declares anchor controls, which a dense tensor cannot apply",
                parameter.0
            ),
        }
    }
}

impl std::error::Error for DenseParameterError {}

impl DenseParameters {
    /// Binds formal parameter `k` to `tensors[k]`.
    pub fn new(tensors: Vec<DenseTensor>) -> Self {
        Self { tensors }
    }

    fn tensor(&self, parameter_use: ParameterUse<'_>) -> Result<&DenseTensor, DenseParameterError> {
        let parameter = parameter_use.parameter;
        if !parameter_use.controls.is_empty() {
            return Err(DenseParameterError::Controlled { parameter });
        }
        self.tensors
            .get(parameter.index())
            .ok_or(DenseParameterError::Unbound { parameter })
    }
}

impl ParameterSource for DenseParameters {
    type Error = DenseParameterError;

    fn apply_linear(
        &self,
        parameter_use: ParameterUse<'_>,
        rows: ArrayView2<'_, f64>,
    ) -> Result<Array2<f64>, Self::Error> {
        let parameter = parameter_use.parameter;
        match self.tensor(parameter_use)? {
            DenseTensor::Matrix(matrix) => {
                if matrix.ncols() != rows.ncols() {
                    return Err(DenseParameterError::WidthMismatch {
                        parameter,
                        expected: matrix.ncols(),
                        found: rows.ncols(),
                    });
                }
                Ok(rows.dot(&matrix.t()))
            }
            DenseTensor::Vector(..) => Err(DenseParameterError::NotAMatrix { parameter }),
        }
    }

    fn vector(&self, parameter_use: ParameterUse<'_>) -> Result<Array1<f64>, Self::Error> {
        let parameter = parameter_use.parameter;
        match self.tensor(parameter_use)? {
            DenseTensor::Vector(vector) => Ok(vector.clone()),
            DenseTensor::Matrix(..) => Err(DenseParameterError::NotAVector { parameter }),
        }
    }
}

#[cfg(test)]
mod tests {
    //! Fixtures whose entries and mask values are dyadic rationals with small
    //! numerators: every product and sum they reach is representable, so no
    //! operation rounds and the comparisons are exact.

    use super::*;
    use ndarray::array;

    fn controls(indices: &[u32]) -> Vec<ControlId> {
        indices.iter().map(|&index| ControlId(index)).collect()
    }

    fn named_controls(count: u32) -> Vec<ControlDecl> {
        (0..count).map(|index| ControlDecl { name: format!("c{index}") }).collect()
    }

    fn singleton_groups(count: u32) -> Vec<MaskGroup> {
        (0..count)
            .map(|index| MaskGroup { name: format!("g{index}"), controls: vec![ControlId(index)] })
            .collect()
    }

    fn free_parameters(count: u32) -> Vec<ParameterDecl> {
        (0..count)
            .map(|index| ParameterDecl { name: format!("p{index}"), controls: Vec::new() })
            .collect()
    }

    fn native(primitive: NativePrimitive, arguments: &[u32]) -> Node {
        Node::Native { primitive, arguments: arguments.iter().map(|&index| NodeId(index)).collect() }
    }

    fn linear(parameter: u32, argument: u32) -> Node {
        native(NativePrimitive::Linear { weight: ParameterSlot(parameter) }, &[argument])
    }

    fn mask(indices: &[u32], argument: u32) -> Node {
        native(NativePrimitive::CoordinateMask { controls: controls(indices) }, &[argument])
    }

    fn term(value: u32, control: Option<u32>) -> SumTerm {
        SumTerm { value: NodeId(value), control: control.map(ControlId) }
    }

    fn body(name: &str, inputs: u32, nodes: Vec<Node>) -> Body {
        let output = NodeId(nodes.len() as u32 - 1);
        Body { name: name.to_string(), inputs, nodes, output }
    }

    fn row_map(rows: &Array2<f64>, matrix: &Array2<f64>) -> Array2<f64> {
        rows.dot(&matrix.t())
    }

    fn global(values: &[(u32, f64)]) -> MaskAssignment {
        let mut masks = MaskAssignment::all_on();
        for &(group, value) in values {
            masks.set(MaskGroupId(group), Vec::new(), value).expect("a finite mask at a new scope");
        }
        masks
    }

    fn run(program: &Program, source: &DenseParameters, masks: &MaskAssignment, x: &Array2<f64>) -> Array2<f64> {
        program
            .execute(source, masks, vec![x.clone()], Vec::new())
            .expect("execution of a valid program")
            .output
    }

    #[test]
    fn tied_coordinate_masks_and_a_controlled_sum_execute_exactly() {
        let r = array![[1.0, 0.5], [-0.25, 2.0], [0.75, -1.0]];
        let u = array![[0.5, -1.0, 0.25], [2.0, 0.125, -0.5]];
        let x = array![[1.0, -2.0], [0.5, 4.0]];
        let parts = ProgramParts {
            parameters: free_parameters(2),
            slots: Vec::new(),
            controls: named_controls(4),
            mask_groups: vec![
                MaskGroup { name: "tied".to_string(), controls: controls(&[0, 1]) },
                MaskGroup { name: "third".to_string(), controls: controls(&[2]) },
                MaskGroup { name: "residual".to_string(), controls: controls(&[3]) },
            ],
            bodies: vec![body(
                "entry",
                1,
                vec![
                    Node::Input { port: 0 },
                    linear(0, 0),
                    mask(&[0, 1, 2], 1),
                    linear(1, 2),
                    Node::Sum { terms: vec![term(0, None), term(3, Some(3))] },
                ],
            )],
            entry: BodyId(0),
        };
        let program = Program::new(parts).expect("a valid program");
        let source = DenseParameters::new(vec![DenseTensor::Matrix(r.clone()), DenseTensor::Matrix(u.clone())]);
        let expected = |coordinates: [f64; 3], residual: f64| {
            let diagonal = Array2::from_diag(&Array1::from(coordinates.to_vec()));
            &x + &(row_map(&row_map(&row_map(&x, &r), &diagonal), &u) * residual)
        };

        let all_on = run(&program, &source, &MaskAssignment::all_on(), &x);
        assert_eq!(all_on, expected([1.0, 1.0, 1.0], 1.0));

        let masked = run(&program, &source, &global(&[(0, 0.5), (1, -1.25), (2, 0.75)]), &x);
        assert_eq!(masked, expected([0.5, 0.5, -1.25], 0.75));
        // Positive controls: the mask moved the output, and the group set BOTH tied
        // coordinates (setting only the first gives a different output).
        assert_ne!(masked, all_on);
        assert_ne!(masked, expected([0.5, 1.0, -1.25], 0.75));
        assert_eq!(program.output_mask_dependence(), MaskDependence::Polynomial { degree: 2 });
    }

    #[test]
    fn composition_induces_the_cross_term_without_a_control_of_its_own() {
        let delta_a = array![[0.5, -0.25], [0.0, 1.0]];
        let delta_b = array![[-1.0, 0.5], [0.25, 0.0]];
        let identity = Array2::<f64>::eye(2);
        let x = array![[1.0, 2.0], [-0.5, 0.25]];
        let parts = ProgramParts {
            parameters: free_parameters(2),
            slots: Vec::new(),
            controls: named_controls(2),
            mask_groups: singleton_groups(2),
            bodies: vec![
                body(
                    "entry",
                    1,
                    vec![
                        Node::Input { port: 0 },
                        Node::Compose {
                            value: NodeId(0),
                            stages: vec![
                                ComposeStage { body: BodyId(1), control: Some(ControlId(0)) },
                                ComposeStage { body: BodyId(2), control: Some(ControlId(1)) },
                            ],
                        },
                    ],
                ),
                body("a", 1, vec![Node::Input { port: 0 }, linear(0, 0)]),
                body("b", 1, vec![Node::Input { port: 0 }, linear(1, 0)]),
            ],
            entry: BodyId(0),
        };
        let program = Program::new(parts).expect("a valid program");
        let source = DenseParameters::new(vec![
            DenseTensor::Matrix(&identity + &delta_a),
            DenseTensor::Matrix(&identity + &delta_b),
        ]);
        let (m_a, m_b) = (0.5, -0.25);
        let output = run(&program, &source, &global(&[(0, m_a), (1, m_b)]), &x);
        // Rows transform by transposes: stage A then stage B is
        // (I + m_B Delta_B)(I + m_A Delta_A) = I + m_A Delta_A + m_B Delta_B + m_A m_B Delta_B Delta_A.
        let first_order = &x + &(row_map(&x, &delta_a) * m_a) + &(row_map(&x, &delta_b) * m_b);
        let cross = row_map(&row_map(&x, &delta_a), &delta_b) * (m_a * m_b);
        assert_eq!(output, &first_order + &cross);
        // Negative control: the cross term is nonzero, so a sum of the two
        // first-order mechanisms is refuted.
        assert_ne!(output, first_order);
        assert_eq!(program.output_mask_dependence(), MaskDependence::Polynomial { degree: 2 });
        assert!(!program.output_mask_dependence().is_affine());
        // A deleted stage is skipped exactly.
        let deleted = run(&program, &source, &global(&[(0, 0.0), (1, 1.0)]), &x);
        assert_eq!(deleted, row_map(&x, &(&identity + &delta_b)));
    }

    fn two_control_program(nodes: Vec<Node>) -> Program {
        Program::new(ProgramParts {
            parameters: free_parameters(2),
            slots: Vec::new(),
            controls: named_controls(2),
            mask_groups: singleton_groups(2),
            bodies: vec![body("entry", 1, nodes)],
            entry: BodyId(0),
        })
        .expect("a valid program")
    }

    #[test]
    fn mask_dependence_separates_affine_sums_from_products_and_nonlinearities() {
        let affine_sum = vec![
            Node::Input { port: 0 },
            linear(0, 0),
            linear(1, 0),
            Node::Sum { terms: vec![term(0, None), term(1, Some(0)), term(2, Some(1))] },
        ];
        let affine = two_control_program(affine_sum.clone());
        assert_eq!(affine.output_mask_dependence(), MaskDependence::Polynomial { degree: 1 });
        assert!(affine.output_mask_dependence().is_affine());

        let mut through_activation = affine_sum;
        through_activation.push(native(NativePrimitive::Activation { activation: NativeActivation::ExactGelu }, &[3]));
        assert_eq!(two_control_program(through_activation).output_mask_dependence(), MaskDependence::Nonlinear);

        let multiplied_layers = two_control_program(vec![
            Node::Input { port: 0 },
            linear(0, 0),
            Node::Sum { terms: vec![term(1, Some(0))] },
            linear(1, 2),
            Node::Sum { terms: vec![term(3, Some(1))] },
        ]);
        assert_eq!(multiplied_layers.output_mask_dependence(), MaskDependence::Polynomial { degree: 2 });

        let activation_before_masks = two_control_program(vec![
            Node::Input { port: 0 },
            linear(0, 0),
            native(NativePrimitive::Activation { activation: NativeActivation::Relu }, &[1]),
            linear(1, 2),
            Node::Sum { terms: vec![term(3, Some(0)), term(0, Some(1))] },
        ]);
        assert_eq!(activation_before_masks.output_mask_dependence(), MaskDependence::Polynomial { degree: 1 });
    }

    #[test]
    fn a_refinement_runs_its_native_body_exactly_when_every_reached_control_is_one() {
        let w = array![[1.0, -0.5], [0.25, 2.0]];
        let r = Array2::<f64>::eye(2);
        // A deliberately wrong factor, U R = 2 W, so the two paths are told apart.
        let u = &w * 2.0;
        let x = array![[1.0, -2.0], [0.5, 4.0]];
        let parts = ProgramParts {
            parameters: free_parameters(3),
            slots: Vec::new(),
            controls: named_controls(2),
            mask_groups: singleton_groups(2),
            bodies: vec![
                body(
                    "entry",
                    1,
                    vec![
                        Node::Input { port: 0 },
                        Node::Refine { native: BodyId(1), mechanism: BodyId(2), arguments: vec![NodeId(0)] },
                    ],
                ),
                body("native", 1, vec![Node::Input { port: 0 }, linear(0, 0)]),
                body("mechanism", 1, vec![Node::Input { port: 0 }, linear(1, 0), mask(&[0, 1], 1), linear(2, 2)]),
            ],
            entry: BodyId(0),
        };
        let program = Program::new(parts).expect("a valid program");
        let source = DenseParameters::new(vec![
            DenseTensor::Matrix(w.clone()),
            DenseTensor::Matrix(r.clone()),
            DenseTensor::Matrix(u.clone()),
        ]);
        let native_output = row_map(&x, &w);
        let mechanism_all_on = row_map(&row_map(&x, &r), &u);
        assert_ne!(mechanism_all_on, native_output);

        assert_eq!(run(&program, &source, &MaskAssignment::all_on(), &x), native_output);
        assert_eq!(run(&program, &source, &global(&[(0, 1.0), (1, 1.0)]), &x), native_output);

        let half = global(&[(0, 0.5)]);
        let diagonal = array![[0.5, 0.0], [0.0, 1.0]];
        assert_eq!(run(&program, &source, &half, &x), row_map(&row_map(&row_map(&x, &r), &diagonal), &u));

        let (execution, residuals) = program
            .refinement_residuals(&source, &MaskAssignment::all_on(), vec![x.clone()], Vec::new())
            .expect("residual execution");
        assert_eq!(execution.output, native_output);
        assert_eq!(residuals.len(), 1);
        let residual = &residuals[0];
        assert!(residual.all_on);
        assert_eq!(residual.invocation, vec![CallSite { body: BodyId(0), node: NodeId(1), stage: 0 }]);
        // U R - W = W, so the residual equals the native scale.
        let native_scale = native_output.iter().fold(0.0_f64, |largest, entry| largest.max(entry.abs()));
        assert_eq!(residual.max_abs_difference, native_scale);
        assert_eq!(residual.native_max_abs, native_scale);

        let (half_execution, half_residuals) = program
            .refinement_residuals(&source, &half, vec![x.clone()], Vec::new())
            .expect("residual execution");
        assert!(!half_residuals[0].all_on);
        assert_eq!(half_execution.output, run(&program, &source, &half, &x));
    }

    #[test]
    fn an_invocation_scope_edits_one_call_and_a_group_ties_every_use() {
        let build = |mask_groups: Vec<MaskGroup>| {
            Program::new(ProgramParts {
                parameters: Vec::new(),
                slots: Vec::new(),
                controls: named_controls(2),
                mask_groups,
                bodies: vec![
                    body(
                        "entry",
                        1,
                        vec![
                            Node::Input { port: 0 },
                            Node::Call { body: BodyId(1), arguments: vec![NodeId(0)] },
                            Node::Call { body: BodyId(1), arguments: vec![NodeId(1)] },
                            Node::Call { body: BodyId(2), arguments: vec![NodeId(2)] },
                        ],
                    ),
                    body("shared", 1, vec![Node::Input { port: 0 }, mask(&[0], 0)]),
                    body("other", 1, vec![Node::Input { port: 0 }, mask(&[1], 0)]),
                ],
                entry: BodyId(0),
            })
            .expect("a valid program")
        };
        let source = DenseParameters::new(Vec::new());
        let x = array![[2.0]];
        let first = CallSite { body: BodyId(0), node: NodeId(1), stage: 0 };
        let second = CallSite { body: BodyId(0), node: NodeId(2), stage: 0 };
        let separate = build(singleton_groups(2));

        assert_eq!(run(&separate, &source, &MaskAssignment::all_on(), &x), array![[2.0]]);
        // A global value reaches both invocations of the shared body.
        assert_eq!(run(&separate, &source, &global(&[(0, 0.5)]), &x), array![[0.5]]);
        // A scoped value reaches only its invocation.
        let mut scoped = MaskAssignment::all_on();
        scoped.set(MaskGroupId(0), vec![first], 0.5).expect("a new scope");
        assert_eq!(run(&separate, &source, &scoped, &x), array![[1.0]]);
        // The deepest applicable scope wins over the global value.
        let mut layered = global(&[(0, 0.25)]);
        layered.set(MaskGroupId(0), vec![second], 0.5).expect("a new scope");
        assert_eq!(run(&separate, &source, &layered, &x), array![[0.25]]);

        let tied = build(vec![MaskGroup { name: "both".to_string(), controls: controls(&[0, 1]) }]);
        assert_eq!(run(&tied, &source, &global(&[(0, 0.5)]), &x), array![[0.25]]);
        // Positive control: untied, the same value leaves the other body's control on.
        assert_ne!(run(&separate, &source, &global(&[(0, 0.5)]), &x), array![[0.25]]);
    }

    #[test]
    fn activation_nodes_evaluate_the_gam_math_owner() {
        let x = array![[-1.5, 0.0, 2.25]];
        let mut outputs = Vec::new();
        for (activation, owner) in [
            (NativeActivation::Relu, GaussianActivation::Relu),
            (NativeActivation::ExactGelu, GaussianActivation::ExactGelu),
        ] {
            let program = Program::new(ProgramParts {
                parameters: Vec::new(),
                slots: Vec::new(),
                controls: Vec::new(),
                mask_groups: Vec::new(),
                bodies: vec![body(
                    "entry",
                    1,
                    vec![Node::Input { port: 0 }, native(NativePrimitive::Activation { activation }, &[0])],
                )],
                entry: BodyId(0),
            })
            .expect("a valid program");
            let output = run(&program, &DenseParameters::new(Vec::new()), &MaskAssignment::all_on(), &x);
            let expected = x.mapv(|t| {
                let mut value = [0.0];
                gaussian_smoothing_derivatives(owner, t, 0.0, &mut value).expect("a finite point");
                value[0]
            });
            assert_eq!(output, expected);
            outputs.push(output);
        }
        assert_eq!(outputs[0], array![[0.0, 0.0, 2.25]]);
        // The exact GELU is negative at -1.5 and strictly between 0 and t at 2.25,
        // so the two kinds are not routed to one function.
        assert!(outputs[1][[0, 0]] < 0.0);
        assert!(outputs[1][[0, 2]] > 0.0 && outputs[1][[0, 2]] < 2.25);
    }

    fn validation_base() -> ProgramParts {
        ProgramParts {
            parameters: free_parameters(1),
            slots: vec![SlotDecl { name: "stream".to_string() }],
            controls: named_controls(1),
            mask_groups: singleton_groups(1),
            bodies: vec![
                body(
                    "entry",
                    1,
                    vec![
                        Node::Input { port: 0 },
                        Node::Compose {
                            value: NodeId(0),
                            stages: vec![ComposeStage { body: BodyId(1), control: Some(ControlId(0)) }],
                        },
                    ],
                ),
                body("stage", 1, vec![Node::Input { port: 0 }, linear(0, 0)]),
            ],
            entry: BodyId(0),
        }
    }

    #[test]
    fn validation_refuses_each_malformed_graph() {
        assert!(Program::new(validation_base()).is_ok());

        let mut forward = validation_base();
        forward.bodies[1].nodes[1] = linear(0, 1);
        assert!(matches!(Program::new(forward), Err(ProgramError::ForwardReference { .. })));

        let mut recursive = validation_base();
        recursive.bodies[1].nodes.push(Node::Call { body: BodyId(1), arguments: vec![NodeId(1)] });
        assert!(matches!(
            Program::new(recursive),
            Err(ProgramError::RecursiveCall { body: BodyId(1) })
        ));

        let mut twice = validation_base();
        twice.bodies[1].nodes.push(mask(&[0], 1));
        assert!(matches!(
            Program::new(twice),
            Err(ProgramError::ControlUsedTwice { control: ControlId(0) })
        ));

        let mut two_groups = validation_base();
        two_groups.mask_groups.push(MaskGroup { name: "again".to_string(), controls: controls(&[0]) });
        assert!(matches!(Program::new(two_groups), Err(ProgramError::ControlInTwoGroups { .. })));

        let mut ungrouped = validation_base();
        ungrouped.mask_groups.clear();
        assert!(matches!(Program::new(ungrouped), Err(ProgramError::ControlUngrouped { .. })));

        let mut unused = validation_base();
        unused.controls.push(ControlDecl { name: "extra".to_string() });
        assert!(matches!(
            Program::new(unused),
            Err(ProgramError::ControlUnused { control: ControlId(1) })
        ));

        let mut impure = validation_base();
        impure.bodies[1].nodes.push(Node::Write { slot: SlotId(0), value: NodeId(1) });
        assert!(matches!(
            Program::new(impure),
            Err(ProgramError::ImpureBody { callee: BodyId(1), .. })
        ));

        let refinement = |native: u32, mechanism: u32| ProgramParts {
            parameters: free_parameters(1),
            slots: Vec::new(),
            controls: named_controls(1),
            mask_groups: singleton_groups(1),
            bodies: vec![
                body(
                    "entry",
                    1,
                    vec![
                        Node::Input { port: 0 },
                        Node::Refine {
                            native: BodyId(native),
                            mechanism: BodyId(mechanism),
                            arguments: vec![NodeId(0)],
                        },
                    ],
                ),
                body("masked", 1, vec![Node::Input { port: 0 }, mask(&[0], 0)]),
                body("plain", 1, vec![Node::Input { port: 0 }, linear(0, 0)]),
            ],
            entry: BodyId(0),
        };
        assert!(Program::new(refinement(2, 1)).is_ok());
        assert!(matches!(
            Program::new(refinement(1, 2)),
            Err(ProgramError::NativeReferenceNotNative { native: BodyId(1), .. })
        ));
    }

    #[test]
    fn a_program_document_round_trips_and_refuses_other_schemas() {
        let program = Program::new(validation_base()).expect("a valid program");
        let text = program.to_json().expect("a serializable program");
        assert_eq!(Program::from_json(&text).expect("a round trip"), program);

        let older = text.replace(MECHANISM_PROGRAM_SCHEMA, "gamfit.MechanismProgram/v0");
        assert_ne!(older, text);
        assert!(matches!(
            Program::from_json(&older),
            Err(ProgramError::SchemaMismatch { found }) if found == "gamfit.MechanismProgram/v0"
        ));

        let extra_field = text.replacen('{', "{\"extra\":1,", 1);
        assert!(matches!(Program::from_json(&extra_field), Err(ProgramError::Json { .. })));

        let mut invalid = validation_base();
        invalid.mask_groups.clear();
        let invalid_text = serde_json::to_string(&ProgramDocument {
            schema: MECHANISM_PROGRAM_SCHEMA.to_string(),
            program: invalid,
        })
        .expect("a serializable document");
        assert!(matches!(Program::from_json(&invalid_text), Err(ProgramError::ControlUngrouped { .. })));
    }

    #[test]
    fn slots_carry_state_and_refusals_name_their_cause() {
        let w = array![[0.5, 0.0], [-0.25, 1.0]];
        let x = array![[2.0, -1.0]];
        let program = Program::new(ProgramParts {
            parameters: free_parameters(1),
            slots: vec![SlotDecl { name: "stream".to_string() }],
            controls: Vec::new(),
            mask_groups: Vec::new(),
            bodies: vec![body(
                "entry",
                0,
                vec![
                    Node::Read { slot: SlotId(0) },
                    linear(0, 0),
                    Node::Sum { terms: vec![term(0, None), term(1, None)] },
                    Node::Write { slot: SlotId(0), value: NodeId(2) },
                ],
            )],
            entry: BodyId(0),
        })
        .expect("a valid program");
        let source = DenseParameters::new(vec![DenseTensor::Matrix(w.clone())]);
        let execution = program
            .execute(&source, &MaskAssignment::all_on(), Vec::new(), vec![Some(x.clone())])
            .expect("a written slot");
        let updated = &x + &row_map(&x, &w);
        assert_eq!(execution.output, updated);
        assert_eq!(execution.slots, vec![Some(updated)]);
        assert!(matches!(
            program.execute(&source, &MaskAssignment::all_on(), Vec::new(), vec![None]),
            Err(ExecutionError::Program(ProgramError::SlotUnwritten { .. }))
        ));
        assert!(matches!(
            program.execute(&source, &MaskAssignment::all_on(), Vec::new(), Vec::new()),
            Err(ExecutionError::Program(ProgramError::SlotCount { expected: 1, found: 0 }))
        ));

        let mut masks = MaskAssignment::all_on();
        assert!(matches!(
            masks.set(MaskGroupId(0), Vec::new(), f64::NAN),
            Err(ProgramError::NonFiniteMask { .. })
        ));
        assert!(masks.set(MaskGroupId(0), Vec::new(), 0.5).is_ok());
        assert!(matches!(
            masks.set(MaskGroupId(0), Vec::new(), 0.25),
            Err(ProgramError::DuplicateMaskScope { .. })
        ));
        assert!(matches!(
            program.execute(&source, &masks, Vec::new(), vec![Some(x.clone())]),
            Err(ExecutionError::Program(ProgramError::UnknownMaskGroup { .. }))
        ));

        let anchored = Program::new(ProgramParts {
            parameters: vec![ParameterDecl { name: "anchored".to_string(), controls: controls(&[0]) }],
            slots: Vec::new(),
            controls: named_controls(1),
            mask_groups: singleton_groups(1),
            bodies: vec![body("entry", 1, vec![Node::Input { port: 0 }, linear(0, 0)])],
            entry: BodyId(0),
        })
        .expect("a valid program");
        assert_eq!(anchored.output_mask_dependence(), MaskDependence::Polynomial { degree: 1 });
        assert!(matches!(
            anchored.execute(&source, &MaskAssignment::all_on(), vec![x.clone()], Vec::new()),
            Err(ExecutionError::Source { error: DenseParameterError::Controlled { .. }, .. })
        ));
    }
}
