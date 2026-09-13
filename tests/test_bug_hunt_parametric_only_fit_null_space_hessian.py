"""Regression: a purely parametric (smooth-free) GLM must be fittable through
the documented Python API ``gamfit.fit``.

An ordinary linear model ``y ~ x1 + x2`` — no ``s()``/``te()``/``matern()``
smooth, just penalized linear terms — is the most basic model a GAM engine can
be asked for, and the CLI (``gam fit data.csv 'y ~ x1 + x2'``) fits it fine and
recovers the OLS coefficients. The README promises the CLI and ``gamfit`` "share
one engine"; the two should therefore agree on this trivial fit.

They did not. ``gamfit.fit(df, "y ~ x1 + x2")`` raised

    gamfit ... InvalidConfigurationError:
        null-space Hessian is not positive definite:
        Cholesky factorization failed: NonPositivePivot { index: 0 }

The fit itself converged; the abort came afterwards, in the Python-only
payload-builder step that restricted the fitted penalized Hessian to the
penalty null-space basis and Cholesky-factorized the result to record a
``null_space_logdet`` for the saved payload. For a smooth-free design that
restricted matrix was reported indefinite and the whole ``fit_table`` call
failed. The CLI never ran this metadata path, which is why it succeeded on
identical data.

The defect was not family-specific: smooth-free Gaussian, binomial, Poisson and
Gamma models all failed the same way, while *adding a single smooth term* (e.g.
``y ~ s(x1) + x2``) made the identical-otherwise model fit. The intercept-only
model ``y ~ 1`` also fit, so the trigger was "≥1 non-intercept parametric term
and no penalized smooth".

This test fits a deterministic linear-Gaussian dataset and asserts the fit
succeeds and tracks the closed-form OLS solution.

Committed for the bug hunt; see the GitHub issue for the full write-up.
"""

from __future__ import annotations

import numpy as np

import gamfit


def _make_linear_dataset(n: int = 200):
    """Deterministic y = 1.5 + 2.0*x1 - 0.7*x2 + small noise."""
    rng = np.random.default_rng(20240605)
    x1 = rng.normal(0.0, 1.0, n)
    x2 = rng.normal(0.0, 1.0, n)
    noise = rng.normal(0.0, 0.3, n)
    y = 1.5 + 2.0 * x1 - 0.7 * x2 + noise
    return {"x1": x1, "x2": x2, "y": y}


def _ols_predictions(data) -> np.ndarray:
    n = len(data["y"])
    design = np.column_stack([np.ones(n), data["x1"], data["x2"]])
    beta, *_ = np.linalg.lstsq(design, data["y"], rcond=None)
    return design @ beta


def test_parametric_only_gaussian_fit_recovers_ols():
    data = _make_linear_dataset()

    # Fitting an ordinary linear model through the public Python API must not
    # raise: the post-fit null-space-logdet metadata step once refused it with
    # "null-space Hessian is not positive definite" although the fit converged.
    model = gamfit.fit(data, "y ~ x1 + x2")

    preds = np.asarray(model.predict(data), dtype=float)
    ols = _ols_predictions(data)

    # A penalized linear fit with the default (tiny) ridge must reproduce the
    # OLS fitted values closely — the CLI fit of this exact dataset matches OLS
    # to < 2e-3 max abs deviation. Use a generous tolerance so the assertion is
    # about correctness, not the exact ridge strength.
    max_dev = float(np.max(np.abs(preds - ols)))
    assert max_dev < 1e-1, (
        f"parametric-only fit does not track the OLS solution: "
        f"max|pred - OLS| = {max_dev:.3e}"
    )

    # And it must explain the (essentially linear) response.
    truth = 1.5 + 2.0 * data["x1"] - 0.7 * data["x2"]
    ss_res = float(np.sum((preds - truth) ** 2))
    ss_tot = float(np.sum((truth - truth.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot
    assert r2 > 0.99, f"parametric-only fit R^2 vs true linear signal = {r2:.4f}"
