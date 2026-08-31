"""Bug hunt: ``s(x, k=...)`` silently ignores the three smallest usable basis
sizes above the degree-reduction band. For a cubic smooth, ``k=4``, ``k=5`` and
``k=6`` all realize the SAME 6-column basis and produce bit-identical fits, with
no warning -- while the equivalent ``knots=`` spelling of the same bases is
honoured exactly, so the two documented ways of asking for one basis disagree.

``docs/formulas.md:129-130`` gives both spellings:

    y ~ s(x, k=15)              # basis dimension 15
    y ~ s(x, knots=10)          # 10 interior knots

and ``docs/formulas.md:187-188`` gives the exact identity between them:

    The basis dimension is then `k = internal_knots + degree + 1`.

So ``s(x, knots=0)`` and ``s(x, k=4)`` name the same cubic basis. They are not
the same fit.

Measured (n=400, ``y = sin(2*pi*x) + N(0, 0.2)``, default ``degree=3``;
``deviance`` and the coefficient count from ``summary()``):

    k=2   2 coefficients   dev 87.621653679
    k=3   3 coefficients   dev 87.621668426
    k=4   6 coefficients   dev 15.212885592   <-- requested 4
    k=5   6 coefficients   dev 15.212885592   <-- requested 5, bit-identical
    k=6   6 coefficients   dev 15.212885592
    k=7   7 coefficients   dev 15.109383241
    k=8   8 coefficients   dev 15.067894873

    knots=0   4 coefficients   dev 17.702436675    (== k=4 by the doc identity)
    knots=1   5 coefficients   dev 17.690911407    (== k=5)
    knots=2   6 coefficients   dev 15.212885592    (== k=6)
    knots=3   7 coefficients   dev 15.109383241    (== k=7)

The same collapse happens at every degree, always onto ``2 + degree + 1``:

    degree=1: k=2,3,4 all give 4 coefficients, dev 20.213147620
    degree=2: k=3,4,5 all give 5 coefficients, dev 18.007113583
    degree=3: k=4,5,6 all give 6 coefficients, dev 15.212885592

Mechanism (``crates/gam-terms/src/term_builder.rs:4635-4691``,
``parse_ps_internal_knots``)::

    const MIN_EXPRESSIVE_INTERNAL_KNOTS: usize = 2;
    ...
    let effective_degree = degree.min(k - 1).max(1);
    let num_internal_knots = if effective_degree < degree {
        // Reproduce the requested basis size exactly when degree was
        // reduced for a low-cardinality axis: num_basis = k.
        k.saturating_sub(effective_degree + 1)
    } else {
        (k - degree - 1).max(MIN_EXPRESSIVE_INTERNAL_KNOTS)
    };

The first branch -- reached only when ``k <= degree``, i.e. ``k in {2, 3}`` for a
cubic -- is documented to "reproduce the requested basis size exactly" and does.
The second branch applies a floor of 2 internal knots, which silently overrides
``k`` for exactly ``k in {degree+1, degree+2}`` and leaves ``k = degree+3`` as the
smallest honoured request. The ``knots=`` path never passes through this floor
(the ``else`` arm at line 4685 returns the count verbatim), which is why the two
spellings diverge.

Nothing warns. A caller sweeping ``k`` for basis-size selection (AIC / REML over
k, or a refit after a #2774 basis-adequacy flag) silently scores the same model
three times and reads the flat stretch as evidence that k does not matter there.

Observed: ``s(x, k=4)`` and ``s(x, k=5)`` build a 6-column basis and fit
bit-identically to ``s(x, k=6)``; ``s(x, knots=0)`` builds the 4-column basis
``k=4`` names and fits differently.

Expected: either ``k`` is honoured down to ``degree+1`` (matching the ``knots=``
spelling and the documented identity), or the inflation is rejected/announced
instead of applied silently.
"""

from __future__ import annotations

import importlib
from typing import Any

pytest: Any = importlib.import_module("pytest")
np = pytest.importorskip("numpy")
pytest.importorskip("gamfit._rust")

import gamfit

N_ROWS = 400
DEFAULT_DEGREE = 3


def _data() -> dict[str, Any]:
    rng = np.random.default_rng(3)
    x = rng.uniform(0.0, 1.0, N_ROWS)
    return {"x": x, "y": np.sin(2.0 * np.pi * x) + 0.2 * rng.standard_normal(N_ROWS)}


def _fit(term: str) -> tuple[int, float]:
    summary = gamfit.fit(_data(), f"y ~ {term}", family="gaussian").summary()
    return len(summary.coefficients), float(summary.deviance)


def test_control_the_knots_spelling_is_honoured_exactly() -> None:
    """Green today: each extra interior knot adds exactly one coefficient and
    changes the fit, so the ladder itself is well-posed."""
    widths = [_fit(f"s(x, knots={j})") for j in range(4)]
    assert [w for w, _ in widths] == [
        j + DEFAULT_DEGREE + 1 for j in range(4)
    ], f"knots= ladder: {widths}"
    assert len({round(d, 9) for _, d in widths}) == 4, f"knots= ladder collapsed: {widths}"


def test_control_k_below_the_degree_is_honoured_exactly() -> None:
    """Green today: `k=2` and `k=3` take the degree-reduction branch, which the
    source says reproduces the requested basis size exactly -- and it does."""
    for k in (2, 3):
        width, _ = _fit(f"s(x, k={k})")
        assert width == k, f"k={k} realized {width} coefficients"


def test_control_k_at_and_above_degree_plus_three_is_honoured() -> None:
    """Green today: from `k = degree + 3` up, `k` is the coefficient count."""
    for k in (6, 7, 8, 12):
        width, _ = _fit(f"s(x, k={k})")
        assert width == k, f"k={k} realized {width} coefficients"


@pytest.mark.parametrize("k", [4, 5])
def test_small_k_is_honoured(k: int) -> None:
    width, deviance = _fit(f"s(x, k={k})")
    assert width == k, (
        f"s(x, k={k}) realized {width} coefficients (deviance {deviance!r}); "
        "the MIN_EXPRESSIVE_INTERNAL_KNOTS floor overrode the request"
    )


@pytest.mark.parametrize("k", [4, 5])
def test_small_k_is_not_bit_identical_to_the_next_honoured_size(k: int) -> None:
    assert _fit(f"s(x, k={k})") != _fit("s(x, k=6)"), (
        f"s(x, k={k}) is bit-identical to s(x, k=6): {_fit(f's(x, k={k})')}"
    )


@pytest.mark.parametrize("internal", [0, 1, 2, 3])
def test_k_and_knots_name_the_same_basis(internal: int) -> None:
    """``k = internal_knots + degree + 1`` (docs/formulas.md:187-188)."""
    by_knots = _fit(f"s(x, knots={internal})")
    by_k = _fit(f"s(x, k={internal + DEFAULT_DEGREE + 1})")
    assert by_knots[0] == by_k[0] and abs(by_knots[1] - by_k[1]) < 1.0e-9, (
        f"s(x, knots={internal}) -> {by_knots} but "
        f"s(x, k={internal + DEFAULT_DEGREE + 1}) -> {by_k}"
    )


@pytest.mark.parametrize("degree", [1, 2, 3])
def test_the_collapse_band_is_present_at_every_degree(degree: int) -> None:
    """The floor is on internal knots, so the collapse band moves with the
    degree: ``k in {degree+1, degree+2}`` all land on ``degree+3``."""
    fits = [
        _fit(f's(x, bs="ps", degree={degree}, k={k})')
        for k in (degree + 1, degree + 2, degree + 3)
    ]
    assert len({(w, round(d, 9)) for w, d in fits}) == 3, (
        f"degree={degree}: k in {{{degree + 1}, {degree + 2}, {degree + 3}}} "
        f"produced {len({(w, round(d, 9)) for w, d in fits})} distinct fits: {fits}"
    )
