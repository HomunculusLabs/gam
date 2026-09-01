"""Regression for #2785: standard prediction tables name every estimand.

The default point remains the response-scale posterior mean required by
SPEC.md. The table also exposes a complete plug-in pair, so no linear-predictor
column is silently paired with a response value from a different estimand.
"""

from __future__ import annotations

import importlib
from typing import Any

pytest: Any = importlib.import_module("pytest")
np = pytest.importorskip("numpy")
pytest.importorskip("gamfit._rust")

import gamfit


def _simulate(family: str, n: int = 180) -> dict[str, Any]:
    rng = np.random.default_rng(1)
    x1 = rng.normal(size=n)
    x2 = rng.uniform(-1.0, 1.0, n)
    eta = 0.3 + 1.5 * x1 - 0.6 * x2
    response = {
        "gaussian": lambda: eta + 0.3 * rng.standard_normal(n),
        "binomial-logit": lambda: (
            rng.random(n) < 1.0 / (1.0 + np.exp(-eta))
        ).astype(float),
        "poisson": lambda: rng.poisson(np.exp(eta)).astype(float),
        "gamma": lambda: rng.gamma(6.0, np.exp(eta) / 6.0),
    }[family]()
    return {"x1": x1, "x2": x2, "y": response}


_GRID = {"x1": np.linspace(-2.5, 2.5, 15), "x2": np.zeros(15)}
_INVERSE_LINK = {
    "gaussian": lambda eta: eta,
    "binomial-logit": lambda eta: 1.0 / (1.0 + np.exp(-eta)),
    "poisson": np.exp,
    "gamma": np.exp,
}


@pytest.mark.parametrize(
    "family", ["gaussian", "binomial-logit", "poisson", "gamma"]
)
def test_standard_prediction_table_has_three_explicit_estimands(family: str) -> None:
    model = gamfit.fit(_simulate(family), "y ~ x1 + x2", family=family)
    table = model.predict(_GRID, return_type="dict")

    required = {"linear_predictor_plugin", "mean_plugin", "posterior_mean"}
    assert required <= table.keys()
    assert {"linear_predictor", "mean"}.isdisjoint(table.keys())

    eta_plugin = np.asarray(table["linear_predictor_plugin"], dtype=float)
    mean_plugin = np.asarray(table["mean_plugin"], dtype=float)
    posterior_mean = np.asarray(table["posterior_mean"], dtype=float)

    design = np.asarray(model.design_matrix(_GRID).matrix, dtype=float)
    beta = np.asarray(
        [coefficient["estimate"] for coefficient in model.summary().coefficients],
        dtype=float,
    )
    np.testing.assert_allclose(eta_plugin, design @ beta, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        mean_plugin, _INVERSE_LINK[family](eta_plugin), rtol=1e-12, atol=1e-12
    )

    # The non-tabular default is still the SPEC-mandated posterior estimand.
    np.testing.assert_array_equal(model.predict(_GRID), posterior_mean)


@pytest.mark.parametrize("family", ["binomial-logit", "poisson", "gamma"])
def test_curved_link_keeps_posterior_mean_distinct_from_plugin(family: str) -> None:
    model = gamfit.fit(_simulate(family, 100), "y ~ x1 + x2", family=family)
    table = model.predict(_GRID, return_type="dict")
    mean_plugin = np.asarray(table["mean_plugin"], dtype=float)
    posterior_mean = np.asarray(table["posterior_mean"], dtype=float)
    assert np.max(np.abs(posterior_mean - mean_plugin)) > 1e-6


def test_identity_link_collapses_all_three_estimands_exactly() -> None:
    model = gamfit.fit(_simulate("gaussian"), "y ~ x1 + x2", family="gaussian")
    table = model.predict(_GRID, return_type="dict")
    np.testing.assert_array_equal(
        table["linear_predictor_plugin"], table["mean_plugin"]
    )
    np.testing.assert_array_equal(table["mean_plugin"], table["posterior_mean"])


def test_standard_interval_columns_name_the_posterior_estimand() -> None:
    model = gamfit.fit(_simulate("poisson"), "y ~ x1 + x2", family="poisson")
    table = model.predict(_GRID, interval=0.9, return_type="dict")
    required = {
        "posterior_mean_standard_error",
        "posterior_mean_lower",
        "posterior_mean_upper",
    }
    assert required <= table.keys()
    assert {"std_error", "mean_lower", "mean_upper"}.isdisjoint(table.keys())
