"""Bug hunt: the order-3 (quintic) exact O(n) spline scan aborts the fit on
ordinary Gaussian data -- and adding a constant to the response, a shift the
estimator is exactly equivariant under, makes the identical fit succeed.

``y ~ s(x, bs="ps", degree=5, penalty_order=3, double_penalty=False)`` is the
documented order-3 scan-routed form (``spline_scan_fast_path`` requires
``degree == 2*order - 1`` and ``!double_penalty``;
``crates/gam-models/src/fit_orchestration/entry.rs:2125-2155``). It is
exercised as a first-class configuration by
``tests/bug_hunt_spline_scan_model_summary_unavailable_test.py``.

Measured on ``y = sin(2*pi*x) + N(0, 0.2)``, ``x`` sorted ``U(0, 1)``, seeds
0..19:

    n = 100  ->  12 of 20 seeds raise ``IntegrationError``
    n = 140  ->  11 of 20
    n = 200  ->  11 of 25

The cubic (``degree=3, penalty_order=2``) scan form raises on 0 of 25 of the same
frames, and the dense ``double_penalty=True`` form on 0 of 25.

The error is a refusal of the certified REML search, not a property of the data
(``crates/gam-math/src/score_opt.rs:692-714``). Seed 0 at n=100::

    spline scan: REML stationary isolation failed: score search: stationary
    structure unresolved on [-12.105374438144967, -12.104760848454575] at
    requested resolution 1.4901161193847656e-8 (certified evaluation error
    3.966754013333685e-4) -- the REQUEST is unsatisfiable: the certified
    evaluation error at this cell already reaches the requested resolution ...
    DerivativeEnclosure { score: { value: [134.05335199505896, 134.05427925955365],
    evaluation_error: 3.9668e-4 }, derivative: [-1.8562e-3, 1.6607e-3],
    curvature: [-2.2666, -0.2358] }

The search has already isolated the optimum to a cell 6.1e-4 wide in ``log
lambda`` over which the criterion is certifiably strictly concave -- the
curvature enclosure is entirely negative, in every failure measured -- and the
REML value is pinned to +-5e-4. It then raises rather than returning that
bracket, because
``crates/gam-solve/src/spline_scan.rs:4581-4584`` asks
``maximize_score_1d`` for a fixed ``f64::EPSILON.sqrt()`` = 1.49e-8 resolution
regardless of the order, while the order-3
``certified_concentrated_criterion_jet`` enclosure is four orders wider than
that. The error text names the defect itself: "the REQUEST is unsatisfiable ...
the request is the defect, not the search (#2614)" -- but the caller never
adapts its request, so the user gets no fit.

That the refused fit exists and is well conditioned is measured two independent
ways, on the same frames that raise. Both are exact symmetries of an order-3
smoothing spline (constants and linear-in-x rescalings live in its
``lambda*integral (f''')^2`` null space / scale group):

    seed   y                 y + 1000        x/2             x*2         cubic     dense
      0    IntegrationError  edf 6.8964      edf 6.8974      Integ.Err   10.4193   6.8527
      1    IntegrationError  edf 6.2776      Integ.Err       edf 6.2781   8.8368   6.2448
      2    IntegrationError  edf 6.4704      Integ.Err       Integ.Err    9.5445   6.4313
      4    IntegrationError  edf 6.7166      Integ.Err       edf 6.7165   9.0091   6.6671
      7    IntegrationError  edf 6.8430      Integ.Err       edf 6.8427   9.9119   6.7916
      8    IntegrationError  edf 7.0044      Integ.Err       edf 7.0044   9.5113   6.9241

The shifted and the rescaled fits of seed 0 -- two different reparameterizations
of the refused problem -- agree to 2.4e-5 out of a fitted range of 2.02, and both
land within 0.6% of the dense reference EDF. Nothing about the problem is
undecidable; only this particular placement of the search's cells is.

Observed: ``gamfit.fit`` raises ``IntegrationError`` on ~half of ordinary
n=100..200 Gaussian frames for the order-3 scan form, and ``fit(x, y + c)``
succeeds on the same frame.

Expected: an estimator that is exactly equivariant under ``y -> y + c`` either
fits both or neither. A certified search that has bracketed a strictly concave
optimum to 7.8e-5 in ``log lambda`` should return that optimum at the resolution
its evaluator can support, not abort the fit.
"""

from __future__ import annotations

import importlib
from typing import Any

pytest: Any = importlib.import_module("pytest")
np = pytest.importorskip("numpy")
pytest.importorskip("gamfit._rust")

import gamfit

QUINTIC_SCAN = 'y ~ s(x, bs="ps", degree=5, penalty_order=3, double_penalty=False)'
CUBIC_SCAN = 'y ~ s(x, bs="ps", degree=3, penalty_order=2, double_penalty=False)'
QUINTIC_DENSE = 'y ~ s(x, bs="ps", degree=5, penalty_order=3, double_penalty=True)'

N_ROWS = 100
#: Seeds measured to raise on the plain frame at n=100 and to fit after a
#: constant response shift.
SEEDS = [0, 1, 2]

#: A shift in the order-3 penalty null space, so the estimator is exactly
#: equivariant under it: f_hat(x; y + SHIFT) == f_hat(x; y) + SHIFT.
SHIFT = 1000.0


def _frame(seed: int) -> tuple[Any, Any]:
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(0.0, 1.0, N_ROWS))
    return x, np.sin(2.0 * np.pi * x) + 0.2 * rng.standard_normal(N_ROWS)


def _fit(x: Any, y: Any, formula: str = QUINTIC_SCAN) -> gamfit.Model:
    return gamfit.fit({"x": x, "y": y}, formula, family="gaussian")


@pytest.mark.parametrize("seed", SEEDS)
def test_control_the_shifted_frame_fits(seed: int) -> None:
    """Green today, and the anchor for the failures below: the same data with a
    constant added to the response fits and reports an ordinary EDF."""
    x, y = _frame(seed)
    edf = float(_fit(x, y + SHIFT).summary().edf_total)
    assert 3.0 < edf < N_ROWS, f"seed={seed}: shifted fit reported edf={edf}"


@pytest.mark.parametrize("seed", SEEDS)
def test_control_the_cubic_scan_and_the_dense_quintic_both_fit(seed: int) -> None:
    """Green today: only the order-3 SCAN route refuses these frames."""
    x, y = _frame(seed)
    assert float(_fit(x, y, CUBIC_SCAN).summary().edf_total) > 3.0
    assert float(_fit(x, y, QUINTIC_DENSE).summary().edf_total) > 3.0


@pytest.mark.parametrize("seed", SEEDS)
def test_quintic_scan_fits_ordinary_gaussian_data(seed: int) -> None:
    x, y = _frame(seed)
    model = _fit(x, y)
    edf = float(model.summary().edf_total)
    assert 3.0 < edf < N_ROWS, f"seed={seed}: edf={edf}"


@pytest.mark.parametrize("seed", SEEDS)
def test_quintic_scan_is_equivariant_under_a_constant_response_shift(seed: int) -> None:
    """``y -> y + c`` lies in the order-3 penalty null space, so the fitted curve
    must shift by exactly ``c`` and the EDF must not move. Today one side of
    this identity raises while the other returns a well-conditioned fit."""
    x, y = _frame(seed)
    shifted = _fit(x, y + SHIFT)
    plain = _fit(x, y)

    shifted_curve = np.asarray(shifted.predict({"x": x}), dtype=float) - SHIFT
    plain_curve = np.asarray(plain.predict({"x": x}), dtype=float)
    span = float(plain_curve.max() - plain_curve.min())
    assert np.max(np.abs(plain_curve - shifted_curve)) < 1.0e-3 * span, (
        f"seed={seed}: the shifted fit is not the plain fit shifted back"
    )

    plain_edf = float(plain.summary().edf_total)
    shifted_edf = float(shifted.summary().edf_total)
    assert abs(plain_edf - shifted_edf) < 1.0e-3 * shifted_edf, (
        f"seed={seed}: edf {plain_edf} vs {shifted_edf} under a null-space shift"
    )
