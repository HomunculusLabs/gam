"""Contract: ``matern(x, periodic=true, period=...)`` fits a periodic Matern smooth
that tracks a periodic signal.

The defect this test was written for: commit c8c3192fa ("feat(#580): periodic
period derivation for radial builders — boolean periodic= (scalar + per-axis
list) on duchon/tps/matern") threaded ``periodic`` into the Matern basis spec,
and the Matern builder consumed it through ``expand_periodic_centers``. But the
``matern`` arm's option whitelist in ``crates/gam-terms/src/term_builder.rs``
omitted ``periodic``/``cyclic``/``period``/``period_start``/``period_end``, while
the sibling ``duchon`` arm accepted them. Every spelling was rejected before the
builder ran:

    InvalidConfigurationError: matern() does not accept option `periodic`.

``MATERN_SMOOTH_OPTION_KEYS`` now lists all five keys.

With the option accepted, the fit did not return. Python Contracts run
34670725591 censored this test at its 300 s per-test bound while it was inside
``gamfit.fit``. Run alone without xdist (MSI job 602008), the spatial
length-scale search ran for 490 s: its REML criterion jumped where a Matern
operator penalty eigenvalue crossed the spectral rank cutoff, and the fit ended
with "incremental realizer lost cached penalty 1" at a trial far out in the
length scale.

This test fits a clean periodic signal on ``[0, 2*pi)`` with a periodic Matern
smooth and asserts that (a) the call is accepted, (b) the predictions are finite,
and (c) the fit is non-degenerate: it tracks the periodic signal rather than
collapsing to a flat line.
"""

from __future__ import annotations

import importlib
from typing import Any

pytest: Any = importlib.import_module("pytest")
np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("gamfit._rust")

import gamfit

TWO_PI = 2.0 * np.pi


def test_matern_periodic_smooth_is_accepted_and_fits() -> None:
    rng = np.random.default_rng(1)
    n = 400
    x = rng.uniform(0.0, TWO_PI, n)
    f = np.sin(x) + 0.5 * np.cos(2.0 * x)
    y = f + rng.normal(0.0, 0.15, n)
    df = pd.DataFrame({"x": x, "y": y})

    # The periodic options must be accepted and the fit must complete.
    model = gamfit.fit(
        df,
        "y ~ matern(x, periodic=true, period=6.283185307179586)",
        family="gaussian",
    )

    grid = np.linspace(0.1, TWO_PI - 0.1, 24)
    preds = np.asarray(model.predict(pd.DataFrame({"x": grid}))).ravel()
    truth = np.sin(grid) + 0.5 * np.cos(2.0 * grid)

    assert np.all(np.isfinite(preds)), f"periodic Matern predictions non-finite: {preds}"
    # Non-degenerate: a real periodic fit tracks the signal (truth std ~0.8),
    # not a collapsed flat line.
    assert preds.std() > 0.3, f"periodic Matern fit looks flat (std={preds.std():.3f})"
    corr = float(np.corrcoef(preds, truth)[0, 1])
    assert corr > 0.6, f"periodic Matern fit does not track the periodic signal (corr={corr:.3f})"
