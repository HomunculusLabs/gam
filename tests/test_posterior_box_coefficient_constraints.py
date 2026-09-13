"""Regression tests for issue #1507 — posterior sampling must respect the
``nonnegative()`` / ``nonpositive()`` / ``linear(min, max)`` box constraints on
a parametric coefficient.

Root cause: the constrained P-IRLS fit enforces the box bounds as KKT
inequality rows, so the constrained MAP pins to an active boundary, but
``model.sample()`` drew a plain unconstrained Gaussian ``N(mode, φ·H⁻¹)``
centred on that boundary — so ~half the posterior mass landed on the forbidden
side. The fix routes a constrained model through a truncated-Gaussian sampler
(exact reflective HMC over the feasible polytope ``A β ≥ b``), so every draw is
feasible.

The reported coefficient is the posterior MEAN (SPEC: the posterior mean is the
default, never MAP). With an active bound only the MAP sits on the bound; the
mean of the truncated posterior lies strictly inside it, about one posterior SD
away. The Rust regression
``tests/regressions/smooths/posterior_sampling_box_coefficient_constraint_1507.rs``
checks the persisted MAP; these tests check the reported mean against the draws.

These tests attack the fix from several independent angles so a regression of
the root cause is caught even if one exact assertion drifts:

* an *active* lower bound (true slope on the wrong side of the bound) — the
  failure mode in the issue — for both Gaussian and binomial families;
* an *active* two-sided ``linear(min, max)`` bound;
* the ``nonpositive()`` mirror image;
* an *inactive* bound (true slope safely interior) must NOT be over-truncated:
  the posterior should still be centred on the unconstrained estimate with a
  non-degenerate spread.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import gamfit

# Feasibility tolerance: the reflective sampler advances *to* a wall, never
# through it, so violations are bounded by reflection round-off (~1e-8).
FEASIBLE_TOL = 1e-6


def _coef_draws(model, frame, *, samples=3000, seed=1, col=1):
    return model.sample(frame, samples=samples, seed=seed).to_numpy()[:, col]


def _assert_active_bound_posterior_mean(est, draws, *, bound, sign) -> None:
    """The reported estimate of an actively bounded coefficient is its posterior mean.

    ``sign`` is +1 for a lower bound and -1 for an upper bound. With the bound
    active the truncated posterior is close to an exponential tail, so its mean
    sits about one posterior SD inside the bound. The estimate must be strictly
    interior, within a few draw SDs of the bound, and the centre of the feasible
    draws. The MAP sits on the bound, about one SD from the draws' mean, so
    reporting it fails the last check.
    """
    spread = float(draws.std())
    excess = sign * (est - bound)
    assert spread > 0.0, "the truncated posterior must have a real spread"
    assert 0.0 < excess < 3.0 * spread, (
        f"estimate {est} must sit strictly inside the active bound {bound}, within 3 "
        f"posterior SDs ({spread:.4g}); it is {excess:.4g} inside"
    )
    assert abs(float(draws.mean()) - est) < 0.5 * spread, (
        f"estimate {est} must be the posterior mean of the feasible draws "
        f"(draw mean {draws.mean():.6g}, SD {spread:.4g})"
    )


def test_nonnegative_active_bound_gaussian() -> None:
    np.random.seed(0)
    n = 200
    x = np.random.randn(n)
    y = -3.0 * x + np.random.randn(n) * 0.5  # true slope strongly NEGATIVE
    df = pd.DataFrame({"x": x, "y": y})
    m = gamfit.fit(df, "y ~ nonnegative(x)")
    est = m.summary().coefficients[1]["estimate"]

    d = _coef_draws(m, df)
    assert d.min() >= -FEASIBLE_TOL, f"draw escaped β ≥ 0: min {d.min()}"
    # Essentially no mass below the bound (was ~52% before the fix).
    assert (d < -FEASIBLE_TOL).mean() < 1e-3
    _assert_active_bound_posterior_mean(est, d, bound=0.0, sign=1.0)


def test_nonnegative_active_bound_binomial() -> None:
    rng = np.random.default_rng(11)
    n = 400
    x = rng.standard_normal(n)
    p = 1.0 / (1.0 + np.exp(-(-0.3 - 2.0 * x)))  # slope NEGATIVE on the logit
    y = rng.binomial(1, p, n)
    df = pd.DataFrame({"x": x, "y": y})
    m = gamfit.fit(df, "y ~ nonnegative(x)", family="binomial")
    est = m.summary().coefficients[1]["estimate"]

    d = _coef_draws(m, df)
    # Was 100% negative before the fix.
    assert d.min() >= -FEASIBLE_TOL, f"binomial draw escaped β ≥ 0: min {d.min()}"
    assert (d < -FEASIBLE_TOL).mean() < 1e-3
    _assert_active_bound_posterior_mean(est, d, bound=0.0, sign=1.0)


def test_linear_min_max_active_lower_bound() -> None:
    np.random.seed(2)
    n = 250
    x = np.random.randn(n)
    y = -3.0 * x + np.random.randn(n) * 0.5  # true slope -3, lower bound -1 active
    df = pd.DataFrame({"x": x, "y": y})
    m = gamfit.fit(df, "y ~ linear(x, min=-1, max=1)")
    est = m.summary().coefficients[1]["estimate"]

    d = _coef_draws(m, df)
    assert d.min() >= -1.0 - FEASIBLE_TOL and d.max() <= 1.0 + FEASIBLE_TOL
    assert ((d < -1.0 - FEASIBLE_TOL) | (d > 1.0 + FEASIBLE_TOL)).mean() < 1e-3
    _assert_active_bound_posterior_mean(est, d, bound=-1.0, sign=1.0)


def test_nonpositive_active_bound() -> None:
    np.random.seed(3)
    n = 220
    x = np.random.randn(n)
    y = 3.0 * x + np.random.randn(n) * 0.5  # true slope POSITIVE, nonpositive bound active
    df = pd.DataFrame({"x": x, "y": y})
    m = gamfit.fit(df, "y ~ nonpositive(x)")
    est = m.summary().coefficients[1]["estimate"]

    d = _coef_draws(m, df)
    assert d.max() <= FEASIBLE_TOL, f"draw escaped β ≤ 0: max {d.max()}"
    assert (d > FEASIBLE_TOL).mean() < 1e-3
    _assert_active_bound_posterior_mean(est, d, bound=0.0, sign=-1.0)


def test_inactive_bound_is_not_over_truncated() -> None:
    # True slope safely positive; nonnegative() bound is INACTIVE. The posterior
    # must remain centred on the (interior) estimate with a real spread, not
    # collapse onto the bound.
    rng = np.random.default_rng(7)
    n = 300
    x = rng.standard_normal(n)
    y = 2.0 * x + rng.standard_normal(n) * 0.5  # interior, far from 0
    df = pd.DataFrame({"x": x, "y": y})
    m = gamfit.fit(df, "y ~ nonnegative(x)")
    est = m.summary().coefficients[1]["estimate"]
    assert est > 1.5  # interior, well away from the bound

    d = _coef_draws(m, df, samples=4000)
    assert d.min() >= -FEASIBLE_TOL
    # Posterior mean tracks the interior estimate; spread is non-degenerate and
    # almost no mass is truncated (the bound is far away).
    assert abs(d.mean() - est) < 0.1
    assert d.std() > 1e-3
    assert (d < -FEASIBLE_TOL).mean() < 1e-3
