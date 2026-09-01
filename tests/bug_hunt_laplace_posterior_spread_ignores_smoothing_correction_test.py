"""Bug hunt: the closed-form Laplace posterior draws from the rho-hat-CONDITIONAL
coefficient covariance while ``summary()`` and ``predict(interval=...)`` publish
the SMOOTHING-CORRECTED one, so one fitted object hands back two credible
intervals for the same quantity that disagree by up to ~45%.

``crates/gam-inference/src/sample.rs`` ``laplace_gaussian_fallback`` states the
invariant this file gates, in its own comment at the draw site:

    // `cov_scale` is the *coefficient-covariance* scale the fit uses for `Vb`
    // -- exactly the quantity `summary()`'s Wald SE is built from.
    ...
    // This keeps the draw spread identical to the reported
    // `summary().std_error`, like the sibling bounded-coefficient path
    // (gam#1514).

That holds only when the fit has NO smoothing parameters. The sampler draws
``N(beta_hat, cov_scale * H^-1)`` -- the plug-in ``Vb``, conditional on
``rho_hat`` -- but ``summary()`` reports
``coefficient_se_source = 'smoothing-corrected'`` for any REML fit that carries
a smooth, and ``predict(interval=...)`` defaults to that same corrected
covariance (``docs/predictions.md:27``: "``None`` requires smoothing-corrected
covariance and errors when the fit cannot supply it").

Measured on ``y ~ s(x)``, n=400, a clean ``sin(2*pi*x)`` signal:

* posterior SD / ``summary().std_error`` ranges over ``0.57 .. 1.00``;
* ``posterior.predict(grid, level=0.95)`` band width 0.1790 vs
  ``model.predict(grid, interval=0.95)`` band width 0.1995 (-10%);
* the posterior band reproduces ``covariance_mode="conditional"`` (0.1792) to
  four digits, which identifies the covariance the sampler actually used.

The parametric control ``y ~ x`` (zero smoothing parameters, so corrected ==
conditional by identity) matches on every one of those numbers, which is what
pins the mechanism on the missing smoothing correction rather than on Monte
Carlo error or a dispersion factor.

``Model.sample`` exposes no ``covariance_mode``, so a caller cannot reconcile the
two surfaces.

Observed: posterior draw spread is the conditional ``Vb`` while every other
uncertainty surface on the same model reports the corrected ``Vp``.

Expected: the posterior draws and the reported Wald SEs / default prediction
bands describe one posterior. Whichever covariance the project picks as the
published one, ``summary().std_error``, ``predict(interval=...)`` and
``Model.sample`` must agree on it -- as they already do for a parametric fit.
"""

from __future__ import annotations

import importlib
from typing import Any

pytest: Any = importlib.import_module("pytest")
np = pytest.importorskip("numpy")
pytest.importorskip("gamfit._rust")

import gamfit

_SEED = 20260823
_DRAWS = 40_000
_GRID = {"x": np.linspace(0.05, 0.95, 9)}

# 40k iid Laplace draws give a relative SE on a standard deviation of
# 1/sqrt(2*N) ~ 0.0035, so 2% is ~6 Monte-Carlo sigma: wide enough never to
# flake, far tighter than the 43% gap being reported.
_MC_TOL = 0.02


def _data() -> dict[str, Any]:
    rng = np.random.default_rng(_SEED)
    x = np.linspace(0.0, 1.0, 400)
    return {"x": x, "y": np.sin(2.0 * np.pi * x) + 0.35 * rng.standard_normal(x.size)}


def _fit_and_sample(formula: str) -> tuple[Any, Any, np.ndarray, np.ndarray]:
    data = _data()
    model = gamfit.fit(data, formula, family="gaussian")
    summary = model.summary()
    posterior = model.sample(data, seed=11, samples=_DRAWS, chains=2)
    reported = np.asarray(
        [c["std_error"] for c in summary.coefficients], dtype=float
    )
    drawn = np.asarray(posterior.std, dtype=float)
    return model, posterior, reported, drawn


def _band_width(table: Any) -> float:
    return float(
        np.mean(
            np.asarray(table["mean_upper"], dtype=float)
            - np.asarray(table["mean_lower"], dtype=float)
        )
    )


def test_parametric_control_posterior_spread_matches_summary() -> None:
    """No smoothing parameters => corrected == conditional; this must pass today."""
    _, _, reported, drawn = _fit_and_sample("y ~ x")
    np.testing.assert_allclose(drawn, reported, rtol=_MC_TOL)


def test_smooth_posterior_spread_matches_reported_standard_errors() -> None:
    """sample.rs: "keeps the draw spread identical to the reported summary().std_error"."""
    _, _, reported, drawn = _fit_and_sample("y ~ s(x)")
    ratio = drawn / reported
    assert np.all(np.abs(ratio - 1.0) <= _MC_TOL), (
        "posterior draw spread does not match summary().std_error; "
        f"ratio range [{ratio.min():.4f}, {ratio.max():.4f}]"
    )


def test_posterior_band_matches_default_prediction_band() -> None:
    """Two credible intervals for the same quantity, from the same object."""
    model, posterior, _, _ = _fit_and_sample("y ~ s(x)")

    posterior_width = _band_width(posterior.predict(_GRID, level=0.95))
    predict_width = _band_width(
        model.predict(_GRID, interval=0.95, return_type="dict")
    )

    assert posterior_width == pytest.approx(predict_width, rel=_MC_TOL), (
        "posterior.predict and model.predict disagree on the 95% band: "
        f"{posterior_width:.6f} vs {predict_width:.6f} "
        f"(ratio {posterior_width / predict_width:.4f})"
    )
