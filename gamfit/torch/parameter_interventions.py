"""Execute parameter edits on a torch module for manifold parameter
decomposition (#2951).

This is the torch end of MPD's native lift
(``crates/gam-sae/src/parameter_decomposition/``). All math that decides an
experiment stays in Rust: the lift declares each edit's delta, scope and
positions and computes the teacher fingerprint from what this module exports.
This module only

* exports the parameter registry: one tensor id per ``Parameter`` object (its
  first name in ``named_parameters()``), its aliases (every other qualified
  name bound to that same object, e.g. a tied embedding and output head), its
  shape and source dtype, with the values fetched one tensor at a time as an
  exact float64 widening;
* discovers use sites: within one forward, every torch function call that
  reads a registered parameter and returns a tensor, keyed
  ``{tensor_id}#{ordinal}`` in execution order;
* executes a forward in which a tensor reads ``W + ΔW`` at every use (a global
  edit, which moves every tied use) or at exactly one use site (a use-specific
  edit), at every position or only at declared positions, and returns the
  executed output with the forward's use sites.

``ΔW`` arrives in the tensor's stored orientation, factored as
``left · rightᵀ`` (the canonical record) or dense for small blocks; applying it
at the use is the only arithmetic here. The native path is ``model(inputs)``
with nothing installed, so the all-on setting runs the original tensors on
their original path. Edited forwards substitute arguments at the
torch-function boundary and never write into a parameter, so the module is
unchanged afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

import numpy as np
import torch
from torch.overrides import TorchFunctionMode, resolve_name

__all__ = [
    "EditedExecution",
    "ExecutedOutput",
    "FactoredDelta",
    "GlobalParameterEdit",
    "ParameterTensor",
    "ParameterUseSite",
    "UseSiteParameterEdit",
    "discover_parameter_use_sites",
    "execute_native",
    "execute_parameter_edits",
    "parameter_registry",
    "parameter_values",
]


def _is_index(value: Any) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _require_ordinal(ordinal: Any) -> None:
    if not _is_index(ordinal) or ordinal < 0:
        raise ValueError(f"ordinal must be a non-negative integer; got {ordinal!r}")


def _declared_positions(positions: Any) -> tuple[int, ...] | None:
    if positions is None:
        return None
    items = tuple(positions)
    if (
        not items
        or not all(_is_index(item) and item >= 0 for item in items)
        or any(later <= earlier for earlier, later in zip(items, items[1:]))
    ):
        raise ValueError(
            "positions must be None (every position) or strictly increasing non-negative "
            f"integers; got {positions!r}"
        )
    return tuple(int(item) for item in items)


@dataclass(frozen=True)
class ParameterTensor:
    """One registered parameter tensor.

    Attributes
    ----------
    tensor_id
        The first qualified name ``named_parameters()`` yields for this
        ``Parameter`` object.
    aliases
        Every later qualified name bound to the same object. An alias is
        identity only; a global edit of ``tensor_id`` moves all of them.
    shape
        The tensor's shape.
    dtype
        The source dtype (``"float32"``, ``"bfloat16"``, ...).
    """

    tensor_id: str
    aliases: tuple[str, ...]
    shape: tuple[int, ...]
    dtype: str


@dataclass(frozen=True)
class ParameterUseSite:
    """One read of a registered parameter within a single forward.

    ``tensor_id`` and ``ordinal`` are the key, and a read always reads
    ``tensor_id`` whichever alias the code used; the other fields describe the
    read.

    Attributes
    ----------
    tensor_id
        The registry id of the tensor read.
    ordinal
        ``k`` for the ``k``-th (from 0) torch function call in the forward that
        read this tensor and returned a tensor. Calls that return no tensor,
        such as dtype or shape queries, are not uses.
    transposed
        True when the read returned the tensor's transpose as a view (``W.t()``,
        ``W.T``), so the op consuming that view sees ``Wᵀ``.
    op
        :func:`torch.overrides.resolve_name` of the function that read it.
    module
        Qualified name of the innermost executing module (``""`` for the root
        module).
    """

    tensor_id: str
    ordinal: int
    transposed: bool
    op: str
    module: str

    def __post_init__(self) -> None:
        _require_ordinal(self.ordinal)

    @property
    def use_site_id(self) -> str:
        """``{tensor_id}#{ordinal}``."""
        return f"{self.tensor_id}#{self.ordinal}"


@dataclass(frozen=True)
class FactoredDelta:
    """``ΔW = left · rightᵀ`` in the tensor's stored orientation: ``left`` is
    ``(d_out, rank)`` and ``right`` is ``(d_in, rank)`` for a ``(d_out, d_in)``
    tensor."""

    left: Any
    right: Any


@dataclass(frozen=True)
class GlobalParameterEdit:
    """Every use of ``tensor_id`` reads ``W + delta``, so every tied name sees
    the edit.

    ``delta`` is a :class:`FactoredDelta` or a dense array of the tensor's
    shape. ``positions`` is ``None`` (every position) or strictly increasing
    sequence positions: at each use only the result rows at those positions
    read ``W + delta``, and the rest read ``W``.
    """

    tensor_id: str
    delta: Any
    positions: Any

    def __post_init__(self) -> None:
        object.__setattr__(self, "positions", _declared_positions(self.positions))


@dataclass(frozen=True)
class UseSiteParameterEdit:
    """Use ``{tensor_id}#{ordinal}`` reads ``W + delta`` (at every position, or
    only at declared ``positions``); every other use reads ``W``."""

    tensor_id: str
    ordinal: int
    delta: Any
    positions: Any

    def __post_init__(self) -> None:
        _require_ordinal(self.ordinal)
        object.__setattr__(self, "positions", _declared_positions(self.positions))


@dataclass(frozen=True)
class ExecutedOutput:
    """The model's output, promoted exactly to float64, with its source
    dtype."""

    values: np.ndarray
    dtype: str


@dataclass(frozen=True)
class EditedExecution:
    """An edited forward: its output, every use observed in execution order
    (the forward's path, to compare against discovery), and the uses that
    received edited values."""

    output: ExecutedOutput
    use_sites: tuple[ParameterUseSite, ...]
    substituted: tuple[ParameterUseSite, ...]


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _registered_parameters(
    model: torch.nn.Module,
) -> dict[str, tuple[torch.nn.Parameter, list[str]]]:
    """``tensor_id -> (parameter, aliases)`` in ``named_parameters`` order."""
    table: dict[str, tuple[torch.nn.Parameter, list[str]]] = {}
    id_of_object: dict[int, str] = {}
    id_of_storage: dict[int, str] = {}
    for name, param in model.named_parameters(remove_duplicate=False):
        if id(param) in id_of_object:
            table[id_of_object[id(param)]][1].append(name)
            continue
        if param.numel() > 0:
            storage = param.untyped_storage().data_ptr()
            if storage in id_of_storage:
                raise ValueError(
                    f"parameters {id_of_storage[storage]!r} and {name!r} are distinct tensors "
                    "sharing one storage; a tie must be one Parameter object under several names"
                )
            id_of_storage[storage] = name
        id_of_object[id(param)] = name
        table[name] = (param, [])
    return table


def parameter_registry(model: torch.nn.Module) -> tuple[ParameterTensor, ...]:
    """Every distinct parameter tensor of ``model``, with its aliases."""
    rows = []
    for tensor_id, (param, aliases) in _registered_parameters(model).items():
        if not param.is_floating_point():
            raise TypeError(f"parameter {tensor_id!r} has non-floating dtype {param.dtype}")
        rows.append(
            ParameterTensor(
                tensor_id=tensor_id,
                aliases=tuple(aliases),
                shape=tuple(param.shape),
                dtype=_dtype_name(param.dtype),
            )
        )
    return tuple(rows)


def parameter_values(model: torch.nn.Module, tensor_id: str) -> np.ndarray:
    """A float64 copy of one registered tensor. Every float format torch stores
    widens to float64 exactly, and its registry row carries the source dtype."""
    param = _lookup(_registered_parameters(model), tensor_id)
    if not param.is_floating_point():
        raise TypeError(f"parameter {tensor_id!r} has non-floating dtype {param.dtype}")
    return param.detach().to(device="cpu", dtype=torch.float64, copy=True).numpy()


def _lookup(
    table: dict[str, tuple[torch.nn.Parameter, list[str]]], tensor_id: str
) -> torch.nn.Parameter:
    if tensor_id not in table:
        raise ValueError(f"unknown tensor id {tensor_id!r}; registered ids are {sorted(table)}")
    return table[tensor_id][0]


def _finite_array(tensor_id: str, values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"the delta for {tensor_id!r} must be finite")
    return np.ascontiguousarray(array)


def _edited_tensor(
    table: dict[str, tuple[torch.nn.Parameter, list[str]]], tensor_id: str, delta: Any
) -> torch.Tensor:
    """``W + delta`` in the tensor's own dtype and device."""
    param = _lookup(table, tensor_id)
    shape = tuple(param.shape)

    def cast(array: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(array).to(device=param.device, dtype=param.dtype)

    if isinstance(delta, FactoredDelta):
        left = _finite_array(tensor_id, delta.left)
        right = _finite_array(tensor_id, delta.right)
        if (
            len(shape) != 2
            or left.ndim != 2
            or right.ndim != 2
            or left.shape[0] != shape[0]
            or right.shape[0] != shape[1]
            or left.shape[1] != right.shape[1]
        ):
            raise ValueError(
                f"the factored delta for {tensor_id!r} has left {left.shape} and right "
                f"{right.shape}; a tensor of shape {shape} needs (d_out, rank) and (d_in, rank)"
            )
        with torch.no_grad():
            edited = param.detach() + cast(left) @ cast(right).T
    else:
        dense = _finite_array(tensor_id, delta)
        if dense.shape != shape:
            raise ValueError(
                f"the dense delta for {tensor_id!r} has shape {dense.shape}; "
                f"the tensor has shape {shape}"
            )
        with torch.no_grad():
            edited = param.detach() + cast(dense)
    if not bool(torch.isfinite(edited).all()):
        raise ValueError(
            f"the edited {tensor_id!r} overflows the tensor's {_dtype_name(param.dtype)} format"
        )
    return edited


def _collect_reads(
    value: Any, parameters: dict[int, tuple[torch.nn.Parameter, str]], found: dict[int, str]
) -> None:
    if isinstance(value, torch.Tensor):
        entry = parameters.get(id(value))
        if entry is not None and entry[0] is value:
            found.setdefault(id(value), entry[1])
    elif isinstance(value, (list, tuple)):
        for item in value:
            _collect_reads(item, parameters, found)
    elif isinstance(value, dict):
        for item in value.values():
            _collect_reads(item, parameters, found)


def _input_tensors(
    value: Any, parameters: dict[int, tuple[torch.nn.Parameter, str]]
) -> list[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        entry = parameters.get(id(value))
        return [] if entry is not None and entry[0] is value else [value]
    if isinstance(value, (list, tuple)):
        return [tensor for item in value for tensor in _input_tensors(item, parameters)]
    if isinstance(value, dict):
        return [tensor for item in value.values() for tensor in _input_tensors(item, parameters)]
    return []


def _substitute(value: Any, replacements: dict[int, torch.Tensor]) -> Any:
    if isinstance(value, torch.Tensor):
        return replacements.get(id(value), value)
    if isinstance(value, list):
        return [_substitute(item, replacements) for item in value]
    if isinstance(value, tuple):
        return tuple(_substitute(item, replacements) for item in value)
    if isinstance(value, dict):
        return {key: _substitute(item, replacements) for key, item in value.items()}
    return value


def _holds_tensor(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return True
    if isinstance(value, (list, tuple)):
        return any(_holds_tensor(item) for item in value)
    return False


def _is_transpose_view(result: Any, read: torch.Tensor) -> bool:
    return (
        isinstance(result, torch.Tensor)
        and result.dim() == 2
        and read.dim() == 2
        and result.data_ptr() == read.data_ptr()
        and tuple(result.shape) == tuple(reversed(read.shape))
        and result.stride() == tuple(reversed(read.stride()))
    )


class _ParameterUseMode(TorchFunctionMode):
    """Counts each tensor-returning read of a registered parameter and swaps in
    edited tensors at the planned reads."""

    def __init__(
        self,
        parameters: dict[int, tuple[torch.nn.Parameter, str]],
        module_stack: list[str],
        global_values: dict[str, tuple[torch.Tensor, tuple[int, ...] | None]],
        site_values: dict[tuple[str, int], tuple[torch.Tensor, tuple[int, ...] | None]],
        leading_shape: tuple[int, int] | None,
    ) -> None:
        super().__init__()
        self._parameters = parameters
        self._module_stack = module_stack
        self._global_values = global_values
        self._site_values = site_values
        self._leading_shape = leading_shape
        self._counts: dict[str, int] = {}
        self.use_sites: list[ParameterUseSite] = []
        self.substituted: list[ParameterUseSite] = []

    def __torch_function__(
        self, func: Callable[..., Any], types: Any, args: Any = (), kwargs: Any = None
    ) -> Any:
        kwargs = {} if kwargs is None else kwargs
        found: dict[int, str] = {}
        _collect_reads((args, kwargs), self._parameters, found)
        if not found:
            return func(*args, **kwargs)
        replacements: dict[int, torch.Tensor] = {}
        declared: dict[int, tuple[int, ...]] = {}
        reads = []
        for key, tensor_id in found.items():
            ordinal = self._counts.get(tensor_id, 0)
            # A call that returns no tensor keeps the ordinal, and an edited
            # tensor answers dtype/shape/device queries exactly as the original.
            planned = self._global_values.get(tensor_id)
            if planned is None:
                planned = self._site_values.get((tensor_id, ordinal))
            if planned is not None:
                replacements[key] = planned[0]
                if planned[1] is not None:
                    declared[key] = planned[1]
            reads.append((key, tensor_id, ordinal))
        if not declared:
            result = func(*_substitute(args, replacements), **_substitute(kwargs, replacements))
        else:
            result = self._select_declared_rows(func, args, kwargs, replacements, declared, reads)
        if not _holds_tensor(result):
            return result
        op = resolve_name(func) or getattr(func, "__qualname__", type(func).__name__)
        module = self._module_stack[-1] if self._module_stack else ""
        for key, tensor_id, ordinal in reads:
            read = replacements.get(key, self._parameters[key][0])
            site = ParameterUseSite(
                tensor_id=tensor_id,
                ordinal=ordinal,
                transposed=_is_transpose_view(result, read),
                op=op,
                module=module,
            )
            self._counts[tensor_id] = ordinal + 1
            self.use_sites.append(site)
            if key in replacements:
                self.substituted.append(site)
        return result

    def _select_declared_rows(
        self,
        func: Callable[..., Any],
        args: Any,
        kwargs: Any,
        replacements: dict[int, torch.Tensor],
        declared: dict[int, tuple[int, ...]],
        reads: list[tuple[int, str, int]],
    ) -> Any:
        """Run the op on the same inputs with ``W`` and with ``W + ΔW`` at the
        declared reads, then take the declared rows from the edited result."""
        sites = [f"{tensor_id}#{ordinal}" for key, tensor_id, ordinal in reads if key in declared]
        position_sets = set(declared.values())
        if len(position_sets) > 1:
            raise ValueError(
                f"use sites {sites} are read by one call with different declared positions"
            )
        (positions,) = position_sets
        everywhere = {key: value for key, value in replacements.items() if key not in declared}
        clean = func(*_substitute(args, everywhere), **_substitute(kwargs, everywhere))
        edited = func(*_substitute(args, replacements), **_substitute(kwargs, replacements))
        if not _holds_tensor(clean):
            return clean
        leading = self._leading_shape
        views_a_read = isinstance(clean, torch.Tensor) and any(
            clean.data_ptr() == tensor.data_ptr()
            for key, tensor_id, ordinal in reads
            for tensor in (self._parameters[key][0], replacements.get(key, self._parameters[key][0]))
        )
        if (
            not isinstance(clean, torch.Tensor)
            or clean.dim() < 2
            or tuple(clean.shape[:2]) != leading
            or views_a_read
            or not any(
                tensor.dim() >= 2 and tuple(tensor.shape[:2]) == leading
                for tensor in _input_tensors((args, kwargs), self._parameters)
            )
        ):
            shape = tuple(clean.shape) if isinstance(clean, torch.Tensor) else type(clean).__name__
            raise ValueError(
                f"use sites {sites} return {shape}, which does not keep the declared leading "
                f"axes (batch, seq) = {leading} from its input; declared positions cannot be "
                "applied there"
            )
        index = torch.as_tensor(positions, device=clean.device)
        combined = clean.clone()
        combined[:, index] = edited[:, index]
        return combined


def _push_module(stack: list[str], name: str) -> Callable[..., None]:
    def hook(module: torch.nn.Module, args: Any) -> None:
        stack.append(name)

    return hook


def _pop_module(stack: list[str]) -> Callable[..., None]:
    def hook(module: torch.nn.Module, args: Any, output: Any) -> None:
        stack.pop()

    return hook


def _run_under_mode(
    model: torch.nn.Module,
    inputs: Any,
    global_values: dict[str, tuple[torch.Tensor, tuple[int, ...] | None]],
    site_values: dict[tuple[str, int], tuple[torch.Tensor, tuple[int, ...] | None]],
    leading_shape: tuple[int, int] | None,
) -> tuple[Any, _ParameterUseMode]:
    table = _registered_parameters(model)
    parameters = {id(param): (param, tensor_id) for tensor_id, (param, aliases) in table.items()}
    stack: list[str] = []
    mode = _ParameterUseMode(parameters, stack, global_values, site_values, leading_shape)
    handles = []
    try:
        for name, module in model.named_modules():
            handles.append(module.register_forward_pre_hook(_push_module(stack, name)))
            handles.append(module.register_forward_hook(_pop_module(stack), always_call=True))
        with torch.no_grad(), mode:
            output = model(inputs)
    finally:
        for handle in handles:
            handle.remove()
    return output, mode


def _executed_output(output: Any) -> ExecutedOutput:
    if not isinstance(output, torch.Tensor) or not output.is_floating_point():
        raise TypeError(
            f"the model must return a floating-point tensor; got {type(output).__name__}"
        )
    return ExecutedOutput(
        values=output.detach().to(device="cpu", dtype=torch.float64, copy=True).numpy(),
        dtype=_dtype_name(output.dtype),
    )


def execute_native(model: torch.nn.Module, inputs: Any) -> ExecutedOutput:
    """The all-on setting: ``model(inputs)`` on the original tensors and path."""
    with torch.no_grad():
        output = model(inputs)
    return _executed_output(output)


def discover_parameter_use_sites(
    model: torch.nn.Module, inputs: Any
) -> tuple[ParameterUseSite, ...]:
    """Every use of every registered tensor in one unedited forward."""
    output, mode = _run_under_mode(model, inputs, {}, {}, None)
    return tuple(mode.use_sites)


def execute_parameter_edits(
    model: torch.nn.Module,
    inputs: Any,
    edits: Iterable[GlobalParameterEdit | UseSiteParameterEdit],
    leading_shape: tuple[int, int] | None = None,
) -> EditedExecution:
    """Run ``model(inputs)`` with every edit applied for this forward only.

    A global edit makes every use read ``W + ΔW``. A use-site edit does so only
    at ``{tensor_id}#{ordinal}``, which this forward must reach. A tensor
    cannot carry both kinds, since a global edit already reaches every use.

    An edit with declared positions needs ``leading_shape``, the unit's
    ``(batch, seq)``. At each of its uses the op runs twice on the same current
    input, with ``W`` and with ``W + ΔW``, and the result rows at the declared
    positions (axis 1) come from the edited run. A use whose result does not
    keep those leading axes from its input, such as ``W.t()`` or an attention
    score ``(batch, heads, seq, seq)``, refuses.

    The returned ``use_sites`` are this forward's path; a caller holding the
    discovered path compares the two to confirm the edit ran on the same one.
    """
    edits = tuple(edits)
    if not edits:
        raise ValueError(
            "execute_parameter_edits needs at least one edit; the all-on setting is execute_native"
        )
    if leading_shape is not None:
        leading_shape = tuple(leading_shape)
        if len(leading_shape) != 2 or not all(_is_index(size) and size > 0 for size in leading_shape):
            raise ValueError(
                f"leading_shape must be the unit's (batch, seq) as positive integers; got {leading_shape!r}"
            )
        leading_shape = (int(leading_shape[0]), int(leading_shape[1]))
    table = _registered_parameters(model)
    global_values: dict[str, tuple[torch.Tensor, tuple[int, ...] | None]] = {}
    site_values: dict[tuple[str, int], tuple[torch.Tensor, tuple[int, ...] | None]] = {}
    for edit in edits:
        if not isinstance(edit, (GlobalParameterEdit, UseSiteParameterEdit)):
            raise TypeError(
                f"edits must be GlobalParameterEdit or UseSiteParameterEdit; got {type(edit).__name__}"
            )
        if edit.positions is not None:
            if leading_shape is None:
                raise ValueError(
                    f"an edit of {edit.tensor_id!r} declares positions, which need the unit's "
                    "leading_shape (batch, seq)"
                )
            if edit.positions[-1] >= leading_shape[1]:
                raise ValueError(
                    f"position {edit.positions[-1]} of an edit of {edit.tensor_id!r} lies outside "
                    f"the unit length {leading_shape[1]}"
                )
        planned = (_edited_tensor(table, edit.tensor_id, edit.delta), edit.positions)
        if isinstance(edit, GlobalParameterEdit):
            if edit.tensor_id in global_values:
                raise ValueError(f"more than one global edit of {edit.tensor_id!r}")
            global_values[edit.tensor_id] = planned
        else:
            key = (edit.tensor_id, int(edit.ordinal))
            if key in site_values:
                raise ValueError(f"more than one edit at use site {edit.tensor_id}#{edit.ordinal}")
            site_values[key] = planned
    clash = sorted({tensor_id for tensor_id, ordinal in site_values} & global_values.keys())
    if clash:
        raise ValueError(
            f"tensors {clash} carry both a global edit and a use-site edit; "
            "a global edit already reaches every use"
        )
    output, mode = _run_under_mode(model, inputs, global_values, site_values, leading_shape)
    reached = {(site.tensor_id, site.ordinal) for site in mode.substituted}
    missed = sorted(f"{tensor_id}#{ordinal}" for tensor_id, ordinal in set(site_values) - reached)
    if missed:
        raise ValueError(f"use sites {missed} were never reached in this forward")
    return EditedExecution(
        output=_executed_output(output),
        use_sites=tuple(mode.use_sites),
        substituted=tuple(mode.substituted),
    )
