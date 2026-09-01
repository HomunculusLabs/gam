"""``predict`` reports two estimands in one row, and the contract between
them is Jensen's inequality, not the inverse link.

``linear_predictor`` is the plug-in ``eta_hat = X beta_hat``. ``mean`` is the
posterior mean of the response, ``E[g^-1(eta)]`` integrated over the
conditional posterior ``eta ~ N(eta_hat, Var(eta_hat))`` -- the default
``SPEC.md`` mandates for every point prediction ("posterior mean must always
be the default, never MAP"). For a curved inverse link ``mean`` therefore
differs from ``g^-1(linear_predictor)`` by an ``O(Var(eta_hat)) = O(1/n)``
term whose sign follows the curvature of ``g^-1``; for the identity link the
two coincide exactly (#2785, which asked for the gap to be closed; the
resolution is that the gap IS the documented estimand, see
``docs/predictions.md``).

This file pins that contract from three directions: the plug-in column is
exactly ``X @ coefficients``; the identity link has no gap; and for the curved
links the gap has Jensen's sign at every grid point and shrinks as ``n``
grows.
"""

from __future__ import annotations

import importlib
from typing import Any

pytest: Any = importlib.import_module("pytest")
np = pytest.importorskip("numpy")
pytest.importorskip("gamfit._rust")

import gamfit

_INVERSE_LINK = {
    "gaussian": lambda e: e,
    "binomial-logit": lambda e: 1.0 / (1.0 + np.exp(-e)),
    "poisson": np.exp,
    "gamma": np.exp,
}


def _simulate(family: str, n: int) -> dict[str, Any]:
    rng = np.random.default_rng(1)
    x1 = rng.normal(size=n)
    x2 = rng.uniform(-1.0, 1.0, n)
    eta = 0.3 + 1.5 * x1 - 0.6 * x2
    response = {
        "gaussian": lambda: eta + 0.3 * rng.standard_normal(n),
        "binomial-logit": lambda: (rng.random(n) < 1.0 / (1.0 + np.exp(-eta))).astype(float),
        "poisson": lambda: rng.poisson(np.exp(eta)).astype(float),
        "gamma": lambda: rng.gamma(6.0, np.exp(eta) / 6.0),
    }[family]()
    return {"x1": x1, "x2": x2, "y": response}


_GRID = {"x1": np.linspace(-2.5, 2.5, 15), "x2": np.zeros(15)}


def _columns(family: str, n: int) -> tuple[Any, np.ndarray, np.ndarray]:
    model = gamfit.fit(_simulate(family, n), "y ~ x1 + x2", family=family)
    table = model.predict(_GRID, return_type="dict")
    eta = np.asarray(table["linear_predictor"], dtype=float)
    mean = np.asarray(table["mean"], dtype=float)
    return model, eta, mean


@pytest.mark.parametrize("family", ["gaussian", "binomial-logit", "poisson", "gamma"])
def test_linear_predictor_is_the_plug_in(family: str) -> None:
    """``linear_predictor`` is ``X beta_hat`` to float precision."""
    model, eta, _ = _columns(family, 300)
    design = np.asarray(model.design_matrix(_GRID).matrix, dtype=float)
    beta = np.asarray([c["estimate"] for c in model.summary().coefficients], dtype=float)
    np.testing.assert_allclose(eta, design @ beta, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("n", [100, 1000])
def test_identity_link_has_no_retransformation_gap(n: int) -> None:
    """Jensen's term vanishes for a linear ``g^-1``: the two columns agree exactly."""
    _, eta, mean = _columns("gaussian", n)
    np.testing.assert_array_equal(mean, eta)


@pytest.mark.parametrize("family", ["poisson", "gamma"])
def test_convex_inverse_link_posterior_mean_exceeds_the_plug_in(family: str) -> None:
    """``exp`` is convex, so ``E[exp(eta)] > exp(E[eta])`` at every grid point."""
    _, eta, mean = _columns(family, 100)
    plugin = _INVERSE_LINK[family](eta)
    assert np.all(mean > plugin), (
        f"{family}: the posterior mean of the response must sit above the plug-in "
        f"transform everywhere; min(mean - plugin) = {(mean - plugin).min():.3e}"
    )


def test_logit_posterior_mean_pulls_toward_one_half() -> None:
    """The logistic function is convex below ``p = 0.5`` and concave above it,
    so the posterior mean is above the plug-in where ``p < 0.5`` and below it
    where ``p > 0.5``."""
    _, eta, mean = _columns("binomial-logit", 100)
    plugin = _INVERSE_LINK["binomial-logit"](eta)
    below = plugin < 0.5 - 0.05
    above = plugin > 0.5 + 0.05
    assert below.any() and above.any(), "the grid must straddle p = 0.5 for this test to bite"
    assert np.all(mean[below] > plugin[below])
    assert np.all(mean[above] < plugin[above])


@pytest.mark.parametrize("family", ["binomial-logit", "poisson", "gamma"])
def test_retransformation_gap_shrinks_with_n(family: str) -> None:
    """The gap is ``O(Var(eta_hat)) = O(1/n)``: ten times the data, a materially
    smaller gap."""
    _, eta_small, mean_small = _columns(family, 100)
    _, eta_large, mean_large = _columns(family, 1000)
    gap_small = np.abs(mean_small - _INVERSE_LINK[family](eta_small)) / np.abs(mean_small)
    gap_large = np.abs(mean_large - _INVERSE_LINK[family](eta_large)) / np.abs(mean_large)
    assert gap_large.max() < 0.5 * gap_small.max(), (
        f"{family}: max relative gap {gap_small.max():.3e} at n=100 vs "
        f"{gap_large.max():.3e} at n=1000"
    )
