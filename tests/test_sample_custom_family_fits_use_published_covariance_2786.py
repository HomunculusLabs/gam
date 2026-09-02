"""``Model.sample`` on a custom-family fit (location-scale, marginal-slope) draws
from the covariance the fit publishes, exactly as a standard fit does (#2786).

Those fits store their conditional ``Vb`` explicitly and carry no engine-level
GLM family, so the Laplace fallback used to die asking them for a scalar
coefficient-covariance scale (``this fit has no engine-level family and
therefore no scalar coefficient-covariance scale``). The draw spread must match
the standard errors ``summary()`` prints for the same coefficients, under the
same covariance definition the draws name.
"""

from __future__ import annotations

import importlib
from typing import Any

pytest: Any = importlib.import_module("pytest")
np = pytest.importorskip("numpy")
pytest.importorskip("gamfit._rust")

import gamfit

_DRAWS = 20_000
# 20k iid Laplace draws: relative SE of a standard deviation is 1/sqrt(2N) ~ 0.005.
_MC_TOL = 0.03


def _data() -> dict[str, Any]:
    rng = np.random.default_rng(2786)
    x = np.linspace(0.0, 1.0, 400)
    sigma = np.exp(-0.5 + 1.0 * x)
    return {"x": x, "y": np.sin(2.0 * np.pi * x) + sigma * rng.standard_normal(x.size)}


def test_gaussian_location_scale_sample_matches_summary_standard_errors() -> None:
    data = _data()
    model = gamfit.fit(data, "y ~ s(x)", family="gaussian", noise_formula="s(x)")
    summary = model.summary()
    posterior = model.sample(data, seed=3, samples=_DRAWS, chains=1)

    assert posterior.covariance_source in {"smoothing-corrected", "conditional"}
    assert posterior.covariance_source == summary.covariance_kind, (
        "the draws must name the covariance summary() prices its SEs from: "
        f"{posterior.covariance_source!r} vs {summary.covariance_kind!r}"
    )
    reported = np.asarray([c["std_error"] for c in summary.coefficients], dtype=float)
    drawn = np.asarray(posterior.std, dtype=float)
    assert drawn.shape == reported.shape
    assert np.all(np.isfinite(drawn)) and np.all(drawn > 0.0)
    ratio = drawn / reported
    assert np.all(np.abs(ratio - 1.0) <= _MC_TOL), (
        "posterior draw spread does not match summary().std_error on a location-scale fit; "
        f"ratio range [{ratio.min():.4f}, {ratio.max():.4f}]"
    )
