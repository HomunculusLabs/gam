"""Numpy-only contract gates for the #2263 gate-5 committed activation panel."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "qwen_l20_structure_panel_acts.py"
SPEC = importlib.util.spec_from_file_location("qwen_l20_structure_panel_acts", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
PANEL = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PANEL
SPEC.loader.exec_module(PANEL)


def test_the_panel_is_the_gate_five_model_layer_and_features() -> None:
    protocol = PANEL.PANEL_PROTOCOL
    assert protocol["model"] == "Qwen/Qwen3-8B"
    assert protocol["layer"] == 20
    assert protocol["features"] == ("weekday", "month")


def test_every_feature_has_fifty_distinct_heads_and_one_row_per_head_and_label() -> None:
    for feature, n_labels in (("weekday", 7), ("month", 12)):
        heads = PANEL.panel_heads(feature)
        assert len(heads) == 50
        assert len(set(heads)) == 50
        assert all("{label}" not in head and head == head.strip() for head in heads)
        prompts = PANEL.panel_prompts(feature)
        assert len(prompts) == 50 * n_labels
        assert all(template.endswith(" {label}") for _, _, template, _ in prompts)
        assert {(head, label) for head, label, _, _ in prompts} == {
            (head, label) for head in range(50) for label in range(n_labels)
        }
    with pytest.raises(ValueError):
        PANEL.panel_heads("color")


def test_the_prompt_bank_hash_is_stable_and_names_the_bank() -> None:
    first = PANEL.prompt_bank_sha256(("weekday", "month"))
    assert first == PANEL.prompt_bank_sha256(("weekday", "month"))
    assert first != PANEL.prompt_bank_sha256(("month", "weekday"))


def test_the_panel_chart_keeps_a_fraction_of_the_centered_variance() -> None:
    rng = np.random.default_rng(2263)
    x = rng.standard_normal((70, 12)) @ np.diag(np.linspace(3.0, 0.1, 12))
    chart, kept = PANEL.panel_chart(x, 4)
    assert chart.shape == (70, 4)
    np.testing.assert_allclose(chart.mean(axis=0), 0.0, atol=1e-12)
    assert 0.0 < kept < 1.0
    full, everything = PANEL.panel_chart(x, 12)
    assert full.shape == (70, 12)
    assert everything == pytest.approx(1.0, rel=1e-12)
    with pytest.raises(ValueError):
        PANEL.panel_chart(np.ones((5, 3)), 2)
