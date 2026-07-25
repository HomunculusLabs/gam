"""#2412: a grouped-binomial fit whose variance components sit at the boundary
must certify, not grind out its outer budget.

The design class: a grouped-binomial response with ridge-penalized factor
blocks of mixed cardinality, a univariate smooth, and a factor smooth over one
of those factors. Two of the three blocks are generated with **true variance
exactly zero**, so their smoothing parameters have a REML optimum at the
infinite-smoothing boundary. That is the load-bearing part of the fixture: a
smoothing parameter running to the ceiling is approached asymptotically and
never lands on the bound exactly, which is what made the search loop and the
terminal certificate disagree about whether it was railed.

While they disagreed, the cost-stall guard measured a stationarity residual
made almost entirely of a rail pull the certificate would have discarded, read
the still-descending rail direction as negative curvature, concluded it had
found a strict saddle, and refused to halt — resetting its no-improvement
streak and burning its escape budget until the outer iteration cap ran out.
The reported symptom was a projected gradient over two orders of magnitude
above the bound after 200 iterations.

What is asserted here is the certification contract on that geometry, not a
timing: the fit is minted, the certificate is self-consistent, the budget is
not exhausted, and no rail is reported as indefinite curvature.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import gamfit

# Kept small on purpose: the defect is a property of the design's geometry, not
# of its size, and a regression guard that costs minutes will be the first
# thing disabled. See the issue for the full-scale (n=13,897) measurements.
N_ROWS = 800
N_LEVELS_A = 60

# The outer iteration cap. Certifying well inside it is the anti-grind
# assertion; the reported failure sat exactly at the cap.
OUTER_ITERATION_CAP = 200

FORMULA = (
    "y ~ x1 + x2 + x3"
    " + group(f_a) + group(f_b) + group(f_c)"
    " + s(t, k=6) + s(t, f_c, bs='fs', k=4)"
)


def _rail_fixture() -> pd.DataFrame:
    rng = np.random.default_rng(2412)

    # Long-tailed level sizes: many levels carry only a handful of rows, which
    # is what leaves the block's variance component weakly identified and sends
    # its smoothing parameter to the ceiling.
    weights = rng.pareto(1.5, size=N_LEVELS_A) + 0.2
    weights /= weights.sum()
    f_a = rng.choice(N_LEVELS_A, size=N_ROWS, p=weights)
    f_b = rng.integers(0, 8, size=N_ROWS)
    f_c = rng.choice(5, size=N_ROWS, p=np.array([0.12, 0.29, 0.24, 0.23, 0.12]))

    t = rng.uniform(0.0, 1.0, N_ROWS)
    x = rng.standard_normal((N_ROWS, 3))

    # f_a and f_b contribute NOTHING to the linear predictor: their true
    # variance is zero, so lambda -> infinity is the correct answer for both and
    # the optimizer must be able to mint it. f_c keeps a real variance so the
    # design is not uniformly railed — a fit that grinds here cannot be excused
    # as "every component was degenerate".
    eta = (
        0.2
        + x @ np.array([0.35, -0.25, 0.20])
        + 0.8 * np.sin(2.0 * np.pi * t)
        + np.array([0.30, -0.20, 0.10, 0.0, -0.15])[f_c] * np.cos(np.pi * t)
        + rng.normal(0.0, 0.25, size=5)[f_c]
    )
    mu = 1.0 / (1.0 + np.exp(-eta))

    columns = {f"x{i + 1}": x[:, i] for i in range(3)}
    columns.update(
        {
            # Grouped binomial: two trials per row, response in {0, 0.5, 1}.
            "y": rng.binomial(2, mu) / 2.0,
            "w": np.full(N_ROWS, 2.0),
            "t": t,
            "f_a": [f"a{i}" for i in f_a],
            "f_b": [f"b{i}" for i in f_b],
            "f_c": [f"c{i}" for i in f_c],
        }
    )
    return pd.DataFrame(columns)


@pytest.fixture(scope="module")
def rail_convergence() -> dict:
    """Fit once; every assertion below reads the same certificate."""
    model = gamfit.fit(_rail_fixture(), FORMULA, family="binomial", weights="w")
    convergence = model.summary().convergence
    assert convergence is not None, "the fit reported no convergence certificate"
    return convergence


def test_boundary_variance_components_do_not_block_certification(rail_convergence) -> None:
    """The fit is minted, and its own verdict says so."""
    assert rail_convergence["certified"] is True
    assert rail_convergence["inner_status"] == "Converged", rail_convergence["inner_status"]


def test_the_outer_budget_is_not_exhausted(rail_convergence) -> None:
    """The anti-grind assertion.

    The reported failure did not miss the bound by a little — it ran the outer
    optimizer to its cap and stopped there. Certifying comfortably inside the
    budget is what distinguishes a solved problem from a lucky one.
    """
    iterations = int(rail_convergence["outer_iterations"])
    assert 0 < iterations < OUTER_ITERATION_CAP // 2, (
        f"outer search used {iterations} of {OUTER_ITERATION_CAP} iterations; "
        "the rail-grind regression burns the budget"
    )


def test_the_stationarity_residual_clears_its_own_bound(rail_convergence) -> None:
    outer = rail_convergence["outer"]
    assert outer is not None, "a fit with smoothing parameters must carry an outer certificate"

    raw = float(outer["gradient_norm"])
    projected = float(outer["projected_gradient_norm"])
    bound = float(outer["stationarity_bound"])
    assert np.isfinite(projected) and np.isfinite(bound)
    assert projected <= bound, (projected, bound)

    # Projection removes components; it cannot manufacture them. When a
    # coordinate IS railed this is a strict inequality, which is exactly the
    # KKT-multiplier mass the search loop used to keep counting.
    assert projected <= raw + 1e-12 * max(1.0, raw), (projected, raw)


def test_a_rail_is_not_reported_as_indefinite_curvature(rail_convergence) -> None:
    """The curvature half of the same defect.

    A smoothing parameter running to the ceiling is still descending along that
    direction, so if it is left in the critical cone its curvature reads
    negative. The guard then treats the rail as a strict saddle with an escape
    direction available, refuses to halt, and spends its escape budget
    rediscovering the same point.
    """
    outer = rail_convergence["outer"]
    assert outer["hessian_psd"] is not False, (
        "a boundary variance component was read as indefinite curvature: "
        f"hessian_psd={outer['hessian_psd']}, railed={outer['lambdas_railed']}"
    )
