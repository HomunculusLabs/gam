"""Bug hunt: ``te(...)`` / ``ti(...)`` silently ignore their documented ``degree``
and ``penalty_order`` options on the DEFAULT marginal basis -- the fit is
bit-identical, down to the predictions.

``docs/formulas.md:342-370`` documents both options for the tensor path:

    | `degree`        | 3 | Polynomial degree (all margins).  |
    | `penalty_order` | 2 | Difference-penalty order.         |

and ``ti(...)`` "takes the same options as ``te(...)``". Measured (n=400,
``y = sin(2*pi*x) + 0.7*z^2 + N(0, 0.3)``), every one of these is bit-identical:

    y ~ te(x, z, k=5)                          dev=32.70775210081
    y ~ te(x, z, k=5, degree=1)                dev=32.70775210081
    y ~ te(x, z, k=5, degree=2)                dev=32.70775210081
    y ~ te(x, z, k=5, degree=4)                dev=32.70775210081
    y ~ te(x, z, k=5, degree=[1,3])            dev=32.70775210081
    y ~ te(x, z, k=5, penalty_order=1)         dev=32.70775210081
    y ~ te(x, z, k=5, penalty_order=3)         dev=32.70775210081
    y ~ te(x, z, k=5, penalty_order=[1,2])     dev=32.70775210081

-- identical EDF (13.474505660702045) and identical predictions too, so the
options have no effect at all. ``ti`` behaves the same (260.22194288488504 in
every case). No error, no ``GamInferenceWarning``, nothing in ``model.notes``.

Two controls show the options are not intrinsically inert:

    y ~ s(x, k=7)                                      dev=48.36098246  edf=6.46537
    y ~ s(x, k=7, degree=1)                            dev=48.53185362  edf=6.88977
    y ~ s(x, k=7, degree=2)                            dev=48.20030281  edf=6.67264
    y ~ s(x, k=7, penalty_order=1)                     dev=48.33670604  edf=6.83637

    y ~ te(x, z, k=5, bs=['ps','ps'])                  dev=33.905928918
    y ~ te(x, z, k=5, bs=['ps','ps'], degree=1)        dev=35.980428547
    y ~ te(x, z, k=5, bs=['ps','ps'], degree=2)        dev=34.636309210
    y ~ te(x, z, k=5, bs=['ps','ps'], penalty_order=1) dev=33.575485605

Mechanism (``crates/gam-terms/src/term_builder.rs``, the
``"tensor" | "te" | "ti" | "t2"`` arm at ~line 3462). Both options are parsed
(~lines 3552-3555) and per-axis ``effective_degree`` / ``effective_penalty_order``
are derived (~lines 3653-3654), but the margin knot spec is chosen at ~line 3728
by

    } else if margin_wants_cr(&per_axis_bs[axis])
        && requested_knot_placement != BSplineKnotPlacement::Quantile
        && k_axis >= 3
    {
        let cr_knots = select_cr_knots(ds.values.column(c), k_axis)?;
        (BSplineKnotSpec::NaturalCubicRegression { knots: cr_knots },
         OneDimensionalBoundary::Open, None)
    } else {
        // ... B-spline branch: num_internal_knots = k - degree - 1, honours both
    }

with ``margin_wants_cr(&None) == true``, so the DEFAULT margin is a natural cubic
regression spline whose basis and penalty are fixed by its ``k`` value-knots. The
``degree`` / ``penalty_order`` fields are still attached to the pushed
``BSplineBasisSpec`` (~line 3789), where that knotspec ignores them. That is
exactly why the two escapes above work: ``bs=['ps','ps']`` fails
``margin_wants_cr``, and ``knot_placement='quantile'`` fails the second conjunct
(``y ~ te(x, z, k=5, knot_placement='quantile')`` gives 33.905530590 and
``+ degree=2`` gives 35.199050914) -- both land on the B-spline branch.

(``bc='clamped'`` on a tensor is a different matter and is deliberately inert:
``validate_tensor_boundary_tokens`` at ~line 1460 classifies ``clamped`` among
the "non-periodic markers", with the local variable literally named ``inert``,
and the refusal message it raises for ``anchored`` says so. ``docs/formulas.md:365``
calling ``clamped`` "rejected" alongside ``anchored`` is a documentation error
rather than a silent drop, so this file does not gate it.)

Observed: documented tensor options accepted and discarded, with no error,
warning, or inference note.

Expected: ``degree`` and ``penalty_order`` change the tensor fit, as they do for
``s()`` and for an explicit ``bs=['ps','ps']`` tensor.
"""

from __future__ import annotations

import importlib
from typing import Any

pytest: Any = importlib.import_module("pytest")
np = pytest.importorskip("numpy")
pytest.importorskip("gamfit._rust")

import gamfit

_GRID = {"x": np.linspace(0.05, 0.95, 7), "z": np.linspace(0.05, 0.95, 7)}


def _data() -> dict[str, Any]:
    rng = np.random.default_rng(21)
    x = rng.uniform(0.0, 1.0, 400)
    z = rng.uniform(0.0, 1.0, 400)
    return {
        "x": x,
        "z": z,
        "y": np.sin(2.0 * np.pi * x) + 0.7 * z * z + 0.3 * rng.standard_normal(x.size),
    }


def _signature(formula: str) -> tuple[float, float, int, tuple[float, ...]]:
    model = gamfit.fit(_data(), formula, family="gaussian")
    summary = model.summary()
    predictions = np.asarray(
        model.predict(_GRID, return_type="dict")["mean"], dtype=float
    )
    return (
        float(summary.deviance),
        float(summary.edf_total),
        len(summary.coefficients),
        tuple(predictions.tolist()),
    )


def test_control_options_change_the_one_dimensional_fit() -> None:
    """Green today: nothing is wrong with the options themselves."""
    base = _signature("y ~ s(x, k=7)")
    for variant in ("y ~ s(x, k=7, degree=2)", "y ~ s(x, k=7, penalty_order=1)"):
        assert _signature(variant) != base, f"{variant} was inert on the 1-D path too"


def test_control_options_change_a_bspline_margin_tensor_fit() -> None:
    """Green today: with bs=['ps','ps'] the tensor honours both options."""
    base = _signature("y ~ te(x, z, k=5, bs=['ps','ps'])")
    for variant in (
        "y ~ te(x, z, k=5, bs=['ps','ps'], degree=1)",
        "y ~ te(x, z, k=5, bs=['ps','ps'], degree=2)",
        "y ~ te(x, z, k=5, bs=['ps','ps'], penalty_order=1)",
    ):
        assert _signature(variant) != base, f"{variant} was inert on the ps-margin tensor"


@pytest.mark.parametrize("kind", ["te", "ti"])
@pytest.mark.parametrize(
    "option",
    ["degree=1", "degree=2", "degree=4", "penalty_order=1", "penalty_order=3"],
)
def test_default_margin_tensor_honours_degree_and_penalty_order(kind: str, option: str) -> None:
    base = _signature(f"y ~ {kind}(x, z, k=5)")
    variant = _signature(f"y ~ {kind}(x, z, k=5, {option})")
    assert variant != base, (
        f"{kind}(x, z, k=5, {option}) produced a bit-identical fit to "
        f"{kind}(x, z, k=5): deviance {variant[0]!r}, edf {variant[1]!r}"
    )


@pytest.mark.parametrize("kind", ["te", "ti"])
@pytest.mark.parametrize("option", ["degree=[1,3]", "penalty_order=[1,2]"])
def test_default_margin_tensor_honours_per_margin_lists(kind: str, option: str) -> None:
    base = _signature(f"y ~ {kind}(x, z, k=5)")
    variant = _signature(f"y ~ {kind}(x, z, k=5, {option})")
    assert variant != base, (
        f"{kind}(x, z, k=5, {option}) produced a bit-identical fit to "
        f"{kind}(x, z, k=5): deviance {variant[0]!r}, edf {variant[1]!r}"
    )
