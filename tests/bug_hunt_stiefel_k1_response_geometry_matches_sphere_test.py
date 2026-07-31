"""Regression (#2140): ``response_geometry="stiefel(k=1)"`` is operationally
equivalent to ``response_geometry="sphere"`` on identifiable spherical data.

``St(n, 1)`` is *exactly* the sphere ``S^{n-1}`` — the Stiefel ``k=1`` exp/log/
metric dispatch straight to the sphere
(``crates/gam-geometry/src/manifolds/stiefel.rs::as_sphere``).  Before #2140,
the generic Stiefel Fréchet driver could abort even on data with a well-defined
intrinsic mean:

    response geometry Fréchet mean did not reach stationarity within max_iter

The current generic response-geometry contract is deliberately stronger than
the original issue's whole-sphere fixture: an implicit base is accepted only
when the support lies inside the geometry-derived strong-convexity radius, so
the stationary point is certified as the unique global Fréchet mean. A nearly
uniform cover of the whole sphere does not satisfy that contract and must use
an explicit base. This regression therefore uses a deterministic, nontrivial
spherical cap whose support is comfortably inside ``π/4``. It asserts that
BOTH public spellings fit the same responses and predict unit-norm rows without
weakening the global-identifiability guarantee.
"""

from __future__ import annotations

import importlib
from typing import Any

pytest: Any = importlib.import_module("pytest")
np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("gamfit._rust")

import gamfit


def _fibonacci_spherical_cap(n: int) -> np.ndarray:
    """A deterministic, two-dimensional cap on S² with angular radius < 0.2.

    Starting from a Fibonacci cover keeps the responses genuinely
    two-dimensional; scaling the transverse coordinates and renormalizing puts
    every point in a globally certifiable ball around +e₀.
    """
    i = np.arange(n) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    theta = np.pi * (1.0 + 5.0**0.5) * i
    cover = np.column_stack(
        [np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)]
    )
    cap = np.column_stack([np.ones(n), 0.18 * cover[:, 1], 0.18 * cover[:, 2]])
    return cap / np.linalg.norm(cap, axis=1, keepdims=True)


def _predict_matrix(prediction: Any) -> np.ndarray:
    if hasattr(prediction, "values"):
        return np.asarray(prediction.values, dtype=float)
    return np.asarray(prediction, dtype=float)


def _fit_and_check_unit(geometry: str) -> None:
    v = _fibonacci_spherical_cap(80)
    x = np.linspace(0.0, 1.0, 80)
    df = pd.DataFrame({"x": x, "a": v[:, 0], "b": v[:, 1], "c": v[:, 2]})

    model = gamfit.fit(
        df,
        "r ~ s(x)",
        response_geometry=geometry,
        response_columns=["a", "b", "c"],
    )

    preds = _predict_matrix(model.predict(df.head(20)))
    assert preds.shape[1] == 3, f"{geometry}: expected 3 columns, got {preds.shape}"
    norms = np.linalg.norm(preds, axis=1)
    worst = float(np.max(np.abs(norms - 1.0)))
    assert worst < 1e-8, (
        f"{geometry}: predictions must lie on the sphere (unit-norm); "
        f"max|‖ŷ‖ - 1| = {worst:.3e}"
    )


def test_sphere_control_fits_identifiable_cap() -> None:
    """Control: the dedicated sphere driver fits the identifiable cap."""
    _fit_and_check_unit("sphere")


def test_stiefel_k1_fits_the_same_data_as_sphere() -> None:
    """#2140: stiefel(k=1) IS the sphere; it must fit the identical data the
    sphere geometry fits when the implicit mean is globally identifiable."""
    _fit_and_check_unit("stiefel(k=1)")
