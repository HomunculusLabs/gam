"""Parameter-edit execution contract for manifold parameter decomposition (#2951).

A float64 fixture ties one ``Parameter`` under two names (``embed.weight`` and
``head.weight``) and runs one MLP body twice, so the embedding and every body
tensor each have two use sites. References are hand-built forwards through the
same torch ops, so every comparison is bitwise:

* the registry names one id per tensor object and lists its aliases, and
  values leave as an exact float64 widening of every float format;
* use sites are the tensor-returning reads in execution order, keyed
  ``{tensor_id}#{ordinal}``, with a transposed flag, and a dtype query is not a
  use;
* a zero delta reproduces the native forward bitwise, and a nonzero one does not;
* a global edit, dense or factored, equals ``torch.func.functional_call`` on
  ``W + ΔW`` with tied weights;
* a use-site edit of the tied tensor or of the shared body moves only that use,
  and differs from the global edit;
* declared positions take only those rows from the edited run of the op, on
  the op's current intervened input, and declaring every row equals the
  every-position edit;
* every refusal fires on an input built to trigger it.

Fixed seeds throughout; no clock entropy.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch.overrides import resolve_name

from gamfit.torch.parameter_interventions import (
    FactoredDelta,
    GlobalParameterEdit,
    ParameterUseSite,
    UseSiteParameterEdit,
    discover_parameter_use_sites,
    execute_native,
    execute_parameter_edits,
    parameter_registry,
    parameter_values,
)

_TOKENS = torch.tensor([[3, 1, 4, 1, 5, 2]])
_LEADING = (1, 6)


class _WeightScale(torch.nn.Module):
    """``x`` cast to the weight's dtype, times the weight: one dtype query and
    one value read of ``weight``."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(width, dtype=torch.float64))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.to(self.weight.dtype) * self.weight


class _TiedSharedMlp(torch.nn.Module):
    """Tokens -> embedding -> one MLP body applied twice -> weight scale -> tied head."""

    def __init__(self) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(7, 4, dtype=torch.float64)
        self.body = torch.nn.Sequential(
            torch.nn.Linear(4, 6, dtype=torch.float64),
            torch.nn.GELU(),
            torch.nn.Linear(6, 4, dtype=torch.float64),
        )
        self.scale = _WeightScale(4)
        self.head = torch.nn.Linear(4, 7, bias=False, dtype=torch.float64)
        self.head.weight = self.embed.weight

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embed(tokens)
        x = x + self.body(x)
        x = x + self.body(x)
        return self.head(self.scale(x))


class _TransposedTie(torch.nn.Module):
    """Tokens -> embedding, read out through the embedding's transpose in the
    root forward."""

    def __init__(self) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(7, 4, dtype=torch.float64)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.embed(tokens) @ self.embed.weight.t()


class _TwoViewsOfOneStorage(torch.nn.Module):
    """Two distinct ``Parameter`` objects over halves of one storage."""

    def __init__(self) -> None:
        super().__init__()
        base = torch.zeros(4, dtype=torch.float64)
        self.left = torch.nn.Parameter(base[:2])
        self.right = torch.nn.Parameter(base[2:])


class _TupleOutput(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(2, 2, dtype=torch.float64)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        return (self.linear(x),)


def _seeded(model: torch.nn.Module) -> torch.nn.Module:
    generator = torch.Generator().manual_seed(2951)
    with torch.no_grad():
        for param in model.parameters():
            param.copy_(torch.randn(param.shape, generator=generator, dtype=param.dtype))
    return model


def _model() -> _TiedSharedMlp:
    return _seeded(_TiedSharedMlp())


def _sites_of(sites: tuple[ParameterUseSite, ...], tensor_id: str) -> list[ParameterUseSite]:
    return [site for site in sites if site.tensor_id == tensor_id]


def _ramp(shape: tuple[int, ...]) -> np.ndarray:
    """A deterministic dense delta that moves every element but the first by a
    clearly nonzero amount."""
    size = int(np.prod(shape))
    return 0.25 * np.arange(size, dtype=np.float64).reshape(shape) / size


def _original(model: torch.nn.Module, tensor_id: str) -> torch.Tensor:
    return torch.from_numpy(parameter_values(model, tensor_id))


def _splice(clean: torch.Tensor, moved: torch.Tensor, positions: tuple[int, ...]) -> torch.Tensor:
    index = torch.as_tensor(positions)
    combined = clean.clone()
    combined[:, index] = moved[:, index]
    return combined


def test_registry_names_one_id_per_tensor_object_and_lists_its_aliases() -> None:
    model = _model()
    rows = parameter_registry(model)
    assert [row.tensor_id for row in rows] == [
        "embed.weight",
        "body.0.weight",
        "body.0.bias",
        "body.2.weight",
        "body.2.bias",
        "scale.weight",
    ]
    assert rows[0].aliases == ("head.weight",)
    assert all(row.aliases == () for row in rows[1:])
    assert rows[1].shape == (6, 4)
    assert all(row.dtype == "float64" for row in rows)

    exported = parameter_values(model, "embed.weight")
    assert np.array_equal(exported, model.embed.weight.detach().numpy())
    exported[0, 0] += 1.0
    assert not np.array_equal(exported, model.embed.weight.detach().numpy())


def test_values_leave_as_an_exact_float64_widening_with_the_source_dtype() -> None:
    single = _seeded(torch.nn.Linear(3, 2, dtype=torch.float32))
    exported = parameter_values(single, "weight")
    assert exported.dtype == np.float64
    # Narrowing back recovers the source bits, so the widening lost nothing.
    assert exported.astype(np.float32).tobytes() == single.weight.detach().numpy().tobytes()
    assert parameter_registry(single)[0].dtype == "float32"
    for dtype, name in ((torch.float16, "float16"), (torch.bfloat16, "bfloat16")):
        narrow = _seeded(torch.nn.Linear(3, 2, dtype=torch.float32)).to(dtype)
        widened = parameter_values(narrow, "weight")
        assert widened.dtype == np.float64
        assert torch.equal(torch.from_numpy(widened).to(dtype), narrow.weight.detach())
        assert parameter_registry(narrow)[0].dtype == name


def test_registry_refuses_distinct_parameters_sharing_one_storage() -> None:
    with pytest.raises(ValueError, match="sharing one storage"):
        parameter_registry(_TwoViewsOfOneStorage())


def test_use_sites_are_tensor_returning_reads_in_execution_order() -> None:
    sites = discover_parameter_use_sites(_model(), _TOKENS)
    body = ["body.0.weight", "body.0.bias", "body.2.weight", "body.2.bias"]
    assert [site.tensor_id for site in sites] == [
        "embed.weight",
        *body,
        *body,
        "scale.weight",
        "embed.weight",
    ]
    embedding, head = _sites_of(sites, "embed.weight")
    assert (embedding.use_site_id, embedding.transposed, embedding.op, embedding.module) == (
        "embed.weight#0",
        False,
        resolve_name(F.embedding),
        "embed",
    )
    # The head reads the tied tensor through its alias; the read is still embed.weight's.
    assert (head.use_site_id, head.transposed, head.op, head.module) == (
        "embed.weight#1",
        False,
        resolve_name(F.linear),
        "head",
    )
    for tensor_id in body:
        assert [(site.use_site_id, site.module) for site in _sites_of(sites, tensor_id)] == [
            (f"{tensor_id}#0", tensor_id[:6]),
            (f"{tensor_id}#1", tensor_id[:6]),
        ]
    # ``_WeightScale`` queries ``weight.dtype`` before multiplying; only the
    # multiply returns a tensor, so the weight has exactly one use.
    assert [site.use_site_id for site in _sites_of(sites, "scale.weight")] == ["scale.weight#0"]


def test_a_transposed_read_is_a_use_site_of_the_same_tensor() -> None:
    model = _seeded(_TransposedTie())
    sites = discover_parameter_use_sites(model, _TOKENS)
    assert [(site.use_site_id, site.transposed, site.op, site.module) for site in sites] == [
        ("embed.weight#0", False, resolve_name(F.embedding), "embed"),
        ("embed.weight#1", True, resolve_name(torch.Tensor.t), ""),
    ]
    original = _original(model, "embed.weight")
    delta = _ramp((7, 4))
    run = execute_parameter_edits(
        model, _TOKENS, [UseSiteParameterEdit("embed.weight", 1, delta, None)]
    )
    reference = F.embedding(_TOKENS, original) @ (original + torch.from_numpy(delta)).t()
    assert np.array_equal(run.output.values, reference.numpy())
    assert [site.use_site_id for site in run.substituted] == ["embed.weight#1"]
    assert [site.transposed for site in run.substituted] == [True]


def test_native_execution_is_the_plain_forward() -> None:
    model = _model()
    native = execute_native(model, _TOKENS)
    with torch.no_grad():
        plain = model(_TOKENS)
    assert native.dtype == "float64"
    assert np.array_equal(native.values, plain.numpy())


def test_zero_deltas_reproduce_native_bitwise_and_nonzero_deltas_do_not() -> None:
    model = _model()
    native = execute_native(model, _TOKENS).values
    zero = np.zeros((7, 4))
    zero_factors = FactoredDelta(np.zeros((7, 2)), np.zeros((4, 2)))
    for edit in (
        GlobalParameterEdit("embed.weight", zero, None),
        UseSiteParameterEdit("embed.weight", 1, zero, None),
        GlobalParameterEdit("embed.weight", zero_factors, None),
    ):
        assert np.array_equal(execute_parameter_edits(model, _TOKENS, [edit]).output.values, native)
    moved = _ramp((7, 4))
    for edit in (
        GlobalParameterEdit("embed.weight", moved, None),
        UseSiteParameterEdit("embed.weight", 1, moved, None),
    ):
        assert not np.array_equal(
            execute_parameter_edits(model, _TOKENS, [edit]).output.values, native
        )


def test_global_edit_moves_every_tied_use_like_functional_call() -> None:
    model = _model()
    delta = _ramp((7, 4))
    run = execute_parameter_edits(
        model, _TOKENS, [GlobalParameterEdit("embed.weight", delta, None)]
    )
    edited = _original(model, "embed.weight") + torch.from_numpy(delta)
    with torch.no_grad():
        reference = torch.func.functional_call(model, {"embed.weight": edited}, (_TOKENS,))
    assert np.array_equal(run.output.values, reference.numpy())
    assert [site.use_site_id for site in run.substituted] == ["embed.weight#0", "embed.weight#1"]
    assert run.use_sites == discover_parameter_use_sites(model, _TOKENS)


def test_a_factored_delta_is_read_as_w_plus_left_right_transpose() -> None:
    model = _model()
    left = _ramp((6, 2)) + 0.5
    right = _ramp((4, 2)) - 0.5
    run = execute_parameter_edits(
        model, _TOKENS, [GlobalParameterEdit("body.0.weight", FactoredDelta(left, right), None)]
    )
    edited = _original(model, "body.0.weight") + torch.from_numpy(left) @ torch.from_numpy(right).T
    with torch.no_grad():
        reference = torch.func.functional_call(model, {"body.0.weight": edited}, (_TOKENS,))
    assert np.array_equal(run.output.values, reference.numpy())
    # Positive control: swapping the factor roles is a different edit, and it refuses on shape.
    with pytest.raises(ValueError, match=r"needs \(d_out, rank\) and \(d_in, rank\)"):
        execute_parameter_edits(
            model, _TOKENS, [GlobalParameterEdit("body.0.weight", FactoredDelta(right, left), None)]
        )


def test_use_site_edit_of_the_tied_tensor_moves_only_the_head() -> None:
    model = _model()
    original = _original(model, "embed.weight")
    delta = _ramp((7, 4))
    use_site = execute_parameter_edits(
        model, _TOKENS, [UseSiteParameterEdit("embed.weight", 1, delta, None)]
    )
    global_edit = execute_parameter_edits(
        model, _TOKENS, [GlobalParameterEdit("embed.weight", delta, None)]
    )
    with torch.no_grad():
        x = F.embedding(_TOKENS, original)
        x = x + model.body(x)
        x = x + model.body(x)
        reference = F.linear(model.scale(x), original + torch.from_numpy(delta))
    assert np.array_equal(use_site.output.values, reference.numpy())
    assert not np.array_equal(use_site.output.values, global_edit.output.values)
    assert [site.use_site_id for site in use_site.substituted] == ["embed.weight#1"]
    assert use_site.use_sites == discover_parameter_use_sites(model, _TOKENS)


def test_use_site_edit_of_the_shared_body_moves_only_its_second_call() -> None:
    model = _model()
    delta = _ramp((6, 4))
    edited = _original(model, "body.0.weight") + torch.from_numpy(delta)
    run = execute_parameter_edits(
        model, _TOKENS, [UseSiteParameterEdit("body.0.weight", 1, delta, None)]
    )
    with torch.no_grad():
        x = model.embed(_TOKENS)
        x = x + model.body(x)
        x = x + model.body[2](model.body[1](F.linear(x, edited, model.body[0].bias)))
        reference = model.head(model.scale(x))
    assert np.array_equal(run.output.values, reference.numpy())
    assert [site.use_site_id for site in run.substituted] == ["body.0.weight#1"]


def test_declared_positions_read_the_edit_only_at_those_rows() -> None:
    model = _model()
    delta = _ramp((6, 4))
    positions = (1, 4)
    run = execute_parameter_edits(
        model,
        _TOKENS,
        [GlobalParameterEdit("body.0.weight", delta, positions)],
        leading_shape=_LEADING,
    )
    original = _original(model, "body.0.weight")
    edited = original + torch.from_numpy(delta)
    bias = model.body[0].bias

    def first_linear(x: torch.Tensor) -> torch.Tensor:
        return _splice(F.linear(x, original, bias), F.linear(x, edited, bias), positions)

    with torch.no_grad():
        x = model.embed(_TOKENS)
        x = x + model.body[2](model.body[1](first_linear(x)))
        x = x + model.body[2](model.body[1](first_linear(x)))
        reference = model.head(model.scale(x))
    assert np.array_equal(run.output.values, reference.numpy())
    every = execute_parameter_edits(
        model, _TOKENS, [GlobalParameterEdit("body.0.weight", delta, None)]
    )
    assert not np.array_equal(run.output.values, every.output.values)
    # Control: declaring every row is the every-position edit, bitwise.
    all_rows = execute_parameter_edits(
        model,
        _TOKENS,
        [GlobalParameterEdit("body.0.weight", delta, tuple(range(_LEADING[1])))],
        leading_shape=_LEADING,
    )
    assert np.array_equal(all_rows.output.values, every.output.values)


def test_declared_positions_at_the_embedding_edit_a_suffix_of_rows() -> None:
    model = _model()
    delta = _ramp((7, 4))
    suffix = (3, 4, 5)
    run = execute_parameter_edits(
        model,
        _TOKENS,
        [UseSiteParameterEdit("embed.weight", 0, delta, suffix)],
        leading_shape=_LEADING,
    )
    original = _original(model, "embed.weight")
    edited = original + torch.from_numpy(delta)
    with torch.no_grad():
        x = _splice(F.embedding(_TOKENS, original), F.embedding(_TOKENS, edited), suffix)
        x = x + model.body(x)
        x = x + model.body(x)
        reference = F.linear(model.scale(x), original)
    assert np.array_equal(run.output.values, reference.numpy())
    assert [site.use_site_id for site in run.substituted] == ["embed.weight#0"]


def test_both_runs_of_a_declared_use_see_the_current_intervened_input() -> None:
    model = _model()
    embed_delta = _ramp((7, 4))
    body_delta = _ramp((6, 4))
    positions = (1, 4)
    run = execute_parameter_edits(
        model,
        _TOKENS,
        [
            GlobalParameterEdit("embed.weight", embed_delta, None),
            UseSiteParameterEdit("body.0.weight", 0, body_delta, positions),
        ],
        leading_shape=_LEADING,
    )
    embed = _original(model, "embed.weight") + torch.from_numpy(embed_delta)
    weight = _original(model, "body.0.weight")
    edited = weight + torch.from_numpy(body_delta)
    bias = model.body[0].bias
    with torch.no_grad():
        x = F.embedding(_TOKENS, embed)
        h = _splice(F.linear(x, weight, bias), F.linear(x, edited, bias), positions)
        x = x + model.body[2](model.body[1](h))
        x = x + model.body(x)
        reference = F.linear(model.scale(x), embed)
    assert np.array_equal(run.output.values, reference.numpy())


def test_edited_execution_never_writes_into_the_module() -> None:
    model = _model()

    def raw_values() -> dict[str, bytes]:
        return {
            row.tensor_id: parameter_values(model, row.tensor_id).tobytes()
            for row in parameter_registry(model)
        }

    before = raw_values()
    execute_parameter_edits(
        model,
        _TOKENS,
        [
            GlobalParameterEdit("embed.weight", _ramp((7, 4)), None),
            UseSiteParameterEdit(
                "body.2.weight", 0, FactoredDelta(_ramp((4, 1)), _ramp((6, 1))), (2,)
            ),
        ],
        leading_shape=_LEADING,
    )
    assert raw_values() == before
    # Positive control: a real write into one element is visible to the same comparison.
    with torch.no_grad():
        model.body[2].weight[0, 0] += 1.0
    assert raw_values() != before


def test_edits_refuse_what_they_cannot_execute_faithfully() -> None:
    model = _model()
    delta = _ramp((6, 4))

    def run(*edits: GlobalParameterEdit | UseSiteParameterEdit) -> None:
        execute_parameter_edits(model, _TOKENS, edits)

    with pytest.raises(ValueError, match="at least one edit"):
        run()
    with pytest.raises(TypeError, match="must be GlobalParameterEdit or UseSiteParameterEdit"):
        execute_parameter_edits(model, _TOKENS, ["body.0.weight"])
    # An alias is not an id: the tied tensor is edited through ``embed.weight``.
    with pytest.raises(ValueError, match="unknown tensor id"):
        run(GlobalParameterEdit("head.weight", _ramp((7, 4)), None))
    with pytest.raises(ValueError, match="dense delta"):
        run(GlobalParameterEdit("body.0.weight", delta[:, :3], None))
    with pytest.raises(ValueError, match=r"needs \(d_out, rank\) and \(d_in, rank\)"):
        run(GlobalParameterEdit("body.0.bias", FactoredDelta(_ramp((6, 1)), _ramp((1, 1))), None))
    nonfinite = delta.copy()
    nonfinite[1, 1] = np.nan
    with pytest.raises(ValueError, match="must be finite"):
        run(GlobalParameterEdit("body.0.weight", nonfinite, None))
    with pytest.raises(ValueError, match="more than one global edit"):
        run(
            GlobalParameterEdit("body.0.weight", delta, None),
            GlobalParameterEdit("body.0.weight", delta, None),
        )
    with pytest.raises(ValueError, match="more than one edit at use site"):
        run(
            UseSiteParameterEdit("body.0.weight", 0, delta, None),
            UseSiteParameterEdit("body.0.weight", 0, delta, None),
        )
    with pytest.raises(ValueError, match="both a global edit and a use-site edit"):
        run(
            GlobalParameterEdit("body.0.weight", delta, None),
            UseSiteParameterEdit("body.0.weight", 0, delta, None),
        )
    with pytest.raises(ValueError, match="never reached"):
        run(UseSiteParameterEdit("body.0.weight", 2, delta, None))
    with pytest.raises(ValueError, match="non-negative integer"):
        UseSiteParameterEdit("body.0.weight", -1, delta, None)
    with pytest.raises(ValueError, match="overflows"):
        execute_parameter_edits(
            torch.nn.Linear(2, 2, dtype=torch.float32),
            torch.ones(1, 2),
            [GlobalParameterEdit("weight", np.full((2, 2), 1e39), None)],
        )
    with pytest.raises(TypeError, match="floating-point tensor"):
        execute_native(_TupleOutput(), torch.ones(1, 2, dtype=torch.float64))


def test_declared_positions_refuse_where_rows_are_not_positions() -> None:
    model = _model()
    delta = _ramp((6, 4))
    for bad in ((), (2, 1), (1, 1), (-1,)):
        with pytest.raises(ValueError, match="strictly increasing non-negative"):
            GlobalParameterEdit("body.0.weight", delta, bad)
    with pytest.raises(ValueError, match="need the unit's leading_shape"):
        execute_parameter_edits(model, _TOKENS, [GlobalParameterEdit("body.0.weight", delta, (1,))])
    with pytest.raises(ValueError, match="leading_shape must be"):
        execute_parameter_edits(
            model, _TOKENS, [GlobalParameterEdit("body.0.weight", delta, (1,))], leading_shape=(0, 6)
        )
    with pytest.raises(ValueError, match="outside the unit length"):
        execute_parameter_edits(
            model,
            _TOKENS,
            [GlobalParameterEdit("body.0.weight", delta, (6,))],
            leading_shape=_LEADING,
        )
    # A declared leading shape the op's input and result do not carry.
    with pytest.raises(ValueError, match="does not keep the declared leading axes"):
        execute_parameter_edits(
            model,
            _TOKENS,
            [UseSiteParameterEdit("embed.weight", 0, _ramp((7, 4)), (0,))],
            leading_shape=(1, 5),
        )
    # ``W.t()`` returns the weight's transpose, which has no sequence axis.
    transposed = _seeded(_TransposedTie())
    with pytest.raises(ValueError, match="does not keep the declared leading axes"):
        execute_parameter_edits(
            transposed,
            _TOKENS,
            [UseSiteParameterEdit("embed.weight", 1, _ramp((7, 4)), (0,))],
            leading_shape=_LEADING,
        )
    # One call reading two edited tensors with different declared rows.
    with pytest.raises(ValueError, match="different declared positions"):
        execute_parameter_edits(
            model,
            _TOKENS,
            [
                GlobalParameterEdit("body.0.weight", delta, (1,)),
                GlobalParameterEdit("body.0.bias", _ramp((6,)), (2,)),
            ],
            leading_shape=_LEADING,
        )
