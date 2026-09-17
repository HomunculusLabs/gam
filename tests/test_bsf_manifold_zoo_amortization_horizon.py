"""The manifold-zoo bench scores Eq. 4 against a declared amortisation horizon.

``bench/bsf_manifold_zoo.py`` called ``description_length(fitted, test_x)``
without the required ``amortization_horizon`` keyword, so every bench run raised
``TypeError`` at the description-length step, after its featurizers were fitted.
This runs the bench's own ``main`` on a tiny toy mixture with only the numpy
oracle featurizer. The package facade is loaded against a stub Rust boundary
that records what it was handed, so no compiled extension is needed.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
_FACADE_PATH = _REPO_ROOT / "gamfit" / "_description_length.py"


class _RecordingRustBoundary:
    """Stand-in for ``gamfit._binding.rust_module()`` that records each call."""

    def __init__(self) -> None:
        self.calls: list[dict[str, int]] = []

    def sae_eq4_description_length(
        self,
        test_x,
        recon,
        gate,
        code_dims,
        dictionary_params,
        amortization_horizon,
        fetch,
        *,
        r2_targets=None,
        native_bits_per_token=None,
    ):
        call = {
            "estimation_rows": int(np.asarray(test_x).shape[0]),
            "amortization_horizon": int(amortization_horizon),
        }
        self.calls.append(call)
        return dict(call)


def _install_stub_gamfit(monkeypatch, rust_boundary: _RecordingRustBoundary) -> None:
    """Make ``gamfit._description_length`` the real facade over a stub binding."""
    package = types.ModuleType("gamfit")
    package.__path__ = []
    binding = types.ModuleType("gamfit._binding")
    binding.rust_module = lambda: rust_boundary
    monkeypatch.setitem(sys.modules, "gamfit", package)
    monkeypatch.setitem(sys.modules, "gamfit._binding", binding)
    spec = importlib.util.spec_from_file_location(
        "gamfit._description_length", _FACADE_PATH
    )
    facade = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "gamfit._description_length", facade)
    spec.loader.exec_module(facade)


def test_bench_amortises_the_dictionary_over_the_training_rows(monkeypatch, tmp_path):
    rust = _RecordingRustBoundary()
    _install_stub_gamfit(monkeypatch, rust)
    monkeypatch.syspath_prepend(str(_REPO_ROOT))
    for name in [m for m in sys.modules if m == "bench" or m.startswith("bench.")]:
        monkeypatch.delitem(sys.modules, name)
    before = set(sys.modules)
    out = tmp_path / "zoo.jsonl"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bsf_manifold_zoo",
            "--featurizers", "oracle",
            "--factors", "3",
            "--ambient", "12",
            "--l0", "2",
            "--n-train", "64",
            "--n-test", "48",
            "--seed", "0",
            "--out", str(out),
        ],
    )
    try:
        zoo = importlib.import_module("bench.bsf_manifold_zoo")
        assert zoo.main() == 0
    finally:
        for name in set(sys.modules) - before:
            sys.modules.pop(name, None)

    # One featurizer, scored once: the horizon is the training-row count and the
    # estimation rows are the test rows, never the same number.
    assert rust.calls == [{"estimation_rows": 48, "amortization_horizon": 64}]
    records = [json.loads(line) for line in out.read_text().splitlines()]
    results = [record for record in records if record["record"] == "result"]
    assert [record["featurizer"] for record in results] == ["oracle"]
    assert results[0]["mdl"]["amortization_horizon"] == 64
