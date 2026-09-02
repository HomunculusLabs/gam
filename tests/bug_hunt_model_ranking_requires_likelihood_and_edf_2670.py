"""Ranking must not change estimands when conditional-AIC inputs are absent."""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("gamfit._rust")

import gamfit


@pytest.mark.parametrize(
    "payload, missing",
    [
        ({"reml_score": 12.0, "edf_total": 3.0}, "log_likelihood"),
        ({"reml_score": 12.0, "log_likelihood": -4.0}, "edf_total"),
    ],
)
def test_compare_models_refuses_raw_reml_fallback_2670(
    payload: dict[str, Any], missing: str
) -> None:
    with pytest.raises(ValueError, match=missing):
        gamfit.compare_models([payload], names=["incomplete"])
