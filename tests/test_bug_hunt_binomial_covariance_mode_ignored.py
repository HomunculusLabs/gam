"""Contract: ``predict(covariance_mode=...)`` is honoured for binomial smooth
models, so the smoothing-parameter-uncertainty correction reaches the interval.

``Model.predict``'s ``covariance_mode`` selects the covariance source for the
response-scale SE: ``"conditional"`` = ``H^{-1}`` only; ``"smoothing"`` /
``None`` (the default required ``SmoothingCorrected`` mode) adds the
first-order smoothing correction ``J·Var(rho_hat)·J^T`` and errors if it cannot
be formed (``gamfit/_model.py:84-95``).
For a smooth model with REML-selected ``rho``, the correction is non-trivial, so
the modes must produce *different* standard errors, as they do for Poisson,
Gamma and Gaussian.

The defect this test was written for: binomial returned **bitwise-identical**
standard errors under both modes, so a binomial ``s(x)`` model's default
credible intervals omitted the smoothing uncertainty every other family
includes, and ``covariance_mode`` was a no-op. ``predict_columns`` in
``crates/gam-pyffi`` dispatched on
``(interval, model.prediction_uses_posterior_mean())``, which is ``true`` for
exactly the binomial family (every link) and the wiggle models. The
``(Some(level), true)`` arm called ``predict_posterior_mean`` and never parsed
``options.covariance_mode``, while the sibling ``(Some(level), false)`` arm
taken by Poisson/Gamma/Gaussian parsed it and fed it into
``predict_full_uncertainty``.

This test asserts that ``covariance_mode`` is honoured for binomial: the
smoothing-corrected SE must differ from (and not be smaller than) the
conditional SE, exactly as it does for the Poisson control.

Related: #811 (the same dispatch arm dropped ``observation_interval`` for
binomial).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import gamfit


def _fit(family: str, gen) -> "gamfit.Model":
    rng = np.random.default_rng(1)
    n = 1500
    x = rng.uniform(0.0, 1.0, n)
    y = gen(rng, x)
    return gamfit.fit(pd.DataFrame({"y": y, "x": x}), "y ~ s(x)", family=family)


def _se(model: "gamfit.Model", mode: str) -> np.ndarray:
    grid = pd.DataFrame({"x": np.linspace(0.05, 0.95, 12)})
    out = model.predict(grid, interval=0.95, covariance_mode=mode)
    return np.asarray(out["posterior_mean_standard_error"], dtype=float)


def test_binomial_smooth_se_responds_to_covariance_mode() -> None:
    model = _fit(
        "binomial",
        lambda r, x: r.binomial(1, 1.0 / (1.0 + np.exp(-(np.sin(3.0 * x) * 2.0)))),
    )
    se_cond = _se(model, "conditional")
    se_smooth = _se(model, "smoothing")

    # The smoothing correction J·Var(rho)·J^T is non-trivial for a REML-selected
    # smooth, so the smoothing-corrected SE must differ from the conditional SE.
    assert not np.allclose(se_cond, se_smooth, rtol=1e-6, atol=1e-12), (
        "binomial std_error is identical for covariance_mode='conditional' and "
        "'smoothing' — the smoothing-parameter-uncertainty correction is being "
        f"dropped. conditional={se_cond}, smoothing={se_smooth}"
    )
    # The correction adds variance (H^{-1} + J·Var(rho)·J^T), so the
    # smoothing-corrected SE is never smaller than the conditional SE.
    assert np.mean(se_smooth) >= np.mean(se_cond) - 1e-12, (
        "smoothing-corrected SE is smaller than the conditional SE on average: "
        f"mean conditional={np.mean(se_cond)}, mean smoothing={np.mean(se_smooth)}"
    )


def test_poisson_smooth_se_responds_to_covariance_mode_control() -> None:
    # Control: the same mechanism is honoured for a family routed through the
    # full-uncertainty arm (Poisson is `uses_posterior_mean=False`).
    model = _fit("poisson", lambda r, x: r.poisson(np.exp(0.5 + np.sin(3.0 * x))))
    se_cond = _se(model, "conditional")
    se_smooth = _se(model, "smoothing")
    assert not np.allclose(se_cond, se_smooth, rtol=1e-6, atol=1e-12), (
        "Poisson control: covariance_mode unexpectedly has no effect; "
        f"conditional={se_cond}, smoothing={se_smooth}"
    )
    assert np.mean(se_smooth) >= np.mean(se_cond) - 1e-12
