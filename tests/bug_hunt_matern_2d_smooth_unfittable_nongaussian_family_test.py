"""Contract: a 2-D ``matern(x, z)`` isotropic-Matérn GP smooth fits and recovers
a surface under a non-Gaussian GLM family (Gamma), as ``te(x, z)`` does on the
same data.

The defect this test was written for: the fit aborted deterministically with

    IntegrationError: REML smoothing optimization failed to converge:
    spatial kappa optimization failed: Invalid input:
    bounded linear terms are not supported for GammaLog fits

even though the *same data* fit fine with ``te(x, z)`` and ``thinplate(x, z)``,
and ``matern(x, z)`` itself fit fine under the Gaussian and Binomial families.
The Matérn isotropic length-scale (κ / range) search evaluated its candidate κ
through a *bounded-linear-term* inner observation-state builder
(``crates/gam-models/src/fit_orchestration/drivers/design_construction.rs``)
whose Poisson, Tweedie, negative-binomial, Beta and Gamma arms were hard
``bail_invalid_estim!("bounded linear terms are not supported for <Family> fits")``
refusals, so the family alone refused the forward fit. This is distinct from
the Gaussian-only Matérn defects #1270 (penalty-topology staging), #1357
(Gaussian κ collapse) and #1379 (univariate Gaussian range penalty).

This test builds a Gamma-distributed 2-D surface, confirms the surface IS
recoverable via a control ``te(x, z)`` fit, and then asserts that the documented
``matern(x, z)`` smooth also fits and recovers it.
"""

from __future__ import annotations

import importlib
from typing import Any

pytest: Any = importlib.import_module("pytest")
np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("gamfit._rust")

import gamfit


def _gamma_surface(seed: int = 0, n: int = 600):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, n)
    z = rng.uniform(0.0, 1.0, n)
    eta = 0.8 * np.sin(3.0 * x) + 0.5 * z
    mu = np.exp(eta)
    shape = 5.0
    y = rng.gamma(shape, mu / shape)  # Gamma mean=mu, var=mu^2/shape
    df = pd.DataFrame({"x": x, "z": z, "y": y})
    return df, mu


def test_matern_2d_smooth_is_fittable_under_gamma_family() -> None:
    df, mu = _gamma_surface(seed=0)

    # Control: the SAME data fits cleanly with a tensor-product smooth under the
    # Gamma family, so the data / family / formula machinery is sound — only the
    # matern κ optimizer's family handling is at fault.
    te_model = gamfit.fit(df, "y ~ te(x, z)", family="gamma")
    te_pred = np.asarray(te_model.predict(df), dtype=float)
    assert np.all(np.isfinite(te_pred)), "te control produced non-finite predictions"
    te_corr = float(np.corrcoef(te_pred, mu)[0, 1])
    assert te_corr > 0.4, f"sanity: te(x,z) should recover the surface, got corr={te_corr:.3f}"

    # The documented matern smooth must fit the same data under the same family.
    matern_model = gamfit.fit(df, "y ~ matern(x, z)", family="gamma")
    matern_pred = np.asarray(matern_model.predict(df), dtype=float)
    assert np.all(np.isfinite(matern_pred)), "matern produced non-finite predictions"
    matern_corr = float(np.corrcoef(matern_pred, mu)[0, 1])
    assert matern_corr > 0.5, (
        f"matern(x,z) under Gamma must recover the surface like te/thinplate do, "
        f"got corr={matern_corr:.3f}"
    )
