"""Bug hunt: ``predict`` returns ``linear_predictor`` and ``mean`` in the same
table, and ``mean`` is NOT the inverse link of that ``linear_predictor``.

``mean`` carries an O(n^-1) retransformation / bias correction that
``linear_predictor`` does not, so two columns of one row describe two different
estimands with nothing in the returned object saying so:

    binomial-logit   n=  100  max|mean - linkinv(eta)| = 1.838e-2   rel = 2.81e-1
    binomial-logit   n= 1000  max|mean - linkinv(eta)| = 1.258e-3   rel = 2.73e-2
    binomial-probit  n=  100  max|mean - linkinv(eta)| = 1.450e-2   rel = 9.66e-1
    poisson          n=  100  max|mean - linkinv(eta)| = 6.574e-1   rel = 4.81e-2
    gamma            n=  100  max|mean - linkinv(eta)| = 4.277e-1   rel = 7.29e-3

The gap is exactly O(1/n) (2.63e-3 -> 6.81e-4 -> 1.73e-4 for logit at
n = 500 / 2000 / 8000) and is identically zero for the Gaussian identity link,
which is the signature of a Jensen / retransformation term
``½ · (g^-1)''(eta) · Var(eta)`` applied on the response scale only. Its sign
agrees: positive for the convex ``exp`` (Poisson, Gamma), and for logit it
crosses zero at ``p = 0.5`` and goes negative above it. It reaches 44% of the
reported band half-width (probit, n=100).

``linear_predictor`` is the plain plug-in: ``X @ summary().coefficients`` matches
it to 1e-17, and second differences along a line are ~1e-16, so eta is exactly
affine. It is ``mean`` that carries the extra term.

The CLI shows the same thing in one CSV -- ``gam predict`` on a Poisson model:

    eta,mean
    1.507978486706,4.521651697760      exp(eta) = 4.517589190885   diff +4.06e-3
    0.636245880279,1.892349441949      exp(eta) = 1.889374610274   diff +2.98e-3

Nothing documents this. ``docs/predictions.md:46-48`` lists the columns as
"``linear_predictor``, ``mean``" with no statement that they are different
estimands, and ``gam predict --no-bias-correction``'s own help calls the point
prediction "the plain plug-in / posterior-mean estimate (``eta``/``mean``)" --
one estimate, two spellings. ``docs/cli.md:98`` advertises that flag as
"Disable the prediction-time ``O(n^-1)`` bias correction", but the flag's help
scopes it to "the survival uncertainty paths", and ``Model.predict`` has no such
parameter at all (``data, *, interval, conformal_level, covariance_mode,
observation_interval, return_type, id_column``), so a Python caller cannot turn
it off or even see it.

Observed: ``mean != inverse_link(linear_predictor)`` by up to 97% relative
(probit, n=100) and ~2.7% relative at n=1000.

Expected: the two columns of one prediction row describe one estimand. Either
``mean`` is ``g^-1(linear_predictor)``, or ``linear_predictor`` is
``g(mean)`` -- either choice makes this file pass. The Gaussian identity case
already satisfies it exactly, and is kept here as a control.
"""

from __future__ import annotations

import importlib
from typing import Any

pytest: Any = importlib.import_module("pytest")
np = pytest.importorskip("numpy")
pytest.importorskip("gamfit._rust")

import gamfit

# Float noise in one link round trip is ~1e-15 relative; the smallest gap
# measured above is 1.2e-4 relative, so 1e-9 separates them by five orders.
_TOL = 1e-9

_INVERSE_LINK = {
    "gaussian": lambda e: e,
    "binomial-logit": lambda e: 1.0 / (1.0 + np.exp(-e)),
    "poisson": np.exp,
    "gamma": np.exp,
}


def _simulate(family: str, n: int) -> dict[str, Any]:
    rng = np.random.default_rng(1)
    x1 = rng.normal(size=n)
    x2 = rng.uniform(-1.0, 1.0, n)
    eta = 0.3 + 1.5 * x1 - 0.6 * x2
    response = {
        "gaussian": lambda: eta + 0.3 * rng.standard_normal(n),
        "binomial-logit": lambda: (rng.random(n) < 1.0 / (1.0 + np.exp(-eta))).astype(float),
        "poisson": lambda: rng.poisson(np.exp(eta)).astype(float),
        "gamma": lambda: rng.gamma(6.0, np.exp(eta) / 6.0),
    }[family]()
    return {"x1": x1, "x2": x2, "y": response}


def _gap(family: str, n: int, formula: str = "y ~ x1 + x2") -> tuple[float, float]:
    model = gamfit.fit(_simulate(family, n), formula, family=family)
    grid = {"x1": np.linspace(-2.5, 2.5, 15), "x2": np.zeros(15)}
    table = model.predict(grid, return_type="dict")
    eta = np.asarray(table["linear_predictor"], dtype=float)
    mean = np.asarray(table["mean"], dtype=float)
    plugin = _INVERSE_LINK[family](eta)
    absolute = float(np.abs(mean - plugin).max())
    relative = float((np.abs(mean - plugin) / np.maximum(np.abs(mean), 1e-12)).max())
    return absolute, relative


@pytest.mark.parametrize("n", [100, 1000])
def test_gaussian_identity_control(n: int) -> None:
    """Green today: the identity link has no retransformation term."""
    absolute, relative = _gap("gaussian", n)
    assert absolute == 0.0 and relative == 0.0


@pytest.mark.parametrize("family", ["binomial-logit", "poisson", "gamma"])
@pytest.mark.parametrize("n", [100, 1000])
def test_mean_is_the_inverse_link_of_the_reported_linear_predictor(
    family: str, n: int
) -> None:
    absolute, relative = _gap(family, n)
    assert relative <= _TOL, (
        f"{family} n={n}: predict() returned mean != inverse_link(linear_predictor) "
        f"in the same table (max abs {absolute:.3e}, max rel {relative:.3e})"
    )


def test_gap_is_present_with_a_smooth_term_too() -> None:
    absolute, relative = _gap("poisson", 500, formula="y ~ s(x1) + x2")
    assert relative <= _TOL, (
        "y ~ s(x1) + x2 (poisson): mean != inverse_link(linear_predictor) "
        f"(max abs {absolute:.3e}, max rel {relative:.3e})"
    )
