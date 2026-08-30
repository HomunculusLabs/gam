"""Bug hunt: ``Model.report()`` prints the UN-normalized outer-optimizer
criterion under the label ``REML / LAML``, while ``Model.summary()`` publishes the
rank-aware normalized one under the same name. Two first-party surfaces of one
fitted model, one label, two different numbers -- and the gap is model-dependent,
so it is not a convention a reader can subtract out.

``Summary`` deliberately carries both, and says which is which
(``crates/gam-pyffi/src/model/model_ffi.rs:272-284``)::

    /// Cross-model comparable criterion: `raw_reml_score` plus the rank-aware
    /// Tierney-Kadane normalizer over the penalty null space.
    reml_score: Option<f64>,
    /// The outer optimizer's own criterion value, un-normalized.
    raw_reml_score: Option<f64>,

``docs/getting-started.md`` points a reader at the first one ("REML/LAML
criterion (in the ``reml_score`` field)"). The report is wired to the second:
``crates/gam-report/src/lib.rs:18-20`` documents its own field as
``UnifiedFitResult::reml_score`` -- the raw one -- and renders it at
``crates/gam-report/src/lib.rs:546`` with the row label ``"REML / LAML"``.

Measured (n=500, ``y = sin(2*pi*x) + 0.15*z + N(0, 0.3)``, one seed; the report's
value equals ``summary().raw_reml_score`` to its printed precision in every row):

    formula                summary().reml_score   report "REML / LAML"   gap    null_dim
    y ~ s(x)                    142.39836              140.21000        2.188      1
    y ~ s(x) + s(z)             134.68015              132.49179        2.188      1
    y ~ z + s(x)                146.07769              142.93249        3.145      2
    y ~ w + s(x)                156.12697              151.70240        4.425      2
    y ~ w + v + s(x)            170.03508              163.44422        6.591      3
    y ~ w + v + z + s(x)        174.09721              166.55231        7.545      4

The gap is exactly the normalizer the summary applies,
``0.5*null_space_logdet - 0.5*null_dim*log(2*pi)`` (checked against
``summary()["null_space_logdet"]`` / ``summary()["null_dim"]``: at n=500 with
``null_dim=1``, ``null_space_logdet = log(500) = 6.2146`` and
``0.5*6.2146 - 0.5*log(2*pi) = 2.1884``). It therefore GROWS with the model's
unpenalized dimension -- 2.19 to 7.55 across the six models above, on identical
rows -- so two models read off their reports are separated by a different number
of log-evidence units than the same two models read off ``summary()`` or ranked
by ``gamfit.compare_models``.

That normalizer is not decorative. It is the whole reason `Summary` keeps the two
apart: the raw criterion is the optimizer's own objective at the fitted lambda,
and the codebase's own comparison surfaces refuse to rank on quantities that are
not made comparable first (see the `n_obs` guard on the same struct, #1384/#2595).
The report is the artifact a user hands to someone else, with no second column
and nothing saying which criterion it is.

Every other headline field in the report agrees with `summary()` exactly
(deviance, total EDF, observation count), which is what makes this one field a
defect rather than a different report design.

Observed: ``report()`` and ``summary()`` disagree on ``REML / LAML`` by a
model-dependent 2.19-7.55 on the same rows; the report shows
``summary().raw_reml_score``.

Expected: the report's ``REML / LAML`` row is the criterion ``summary()`` calls
``reml_score`` -- or, if the raw value is what a report should carry, it is
labelled as the raw value and the normalized one appears beside it.
"""

from __future__ import annotations

import importlib
import math
import re
from typing import Any

pytest: Any = importlib.import_module("pytest")
np = pytest.importorskip("numpy")
pytest.importorskip("gamfit._rust")

import gamfit

N_ROWS = 500
SEED = 7

FORMULAS = [
    "y ~ s(x)",
    "y ~ s(x) + s(z)",
    "y ~ z + s(x)",
    "y ~ w + s(x)",
    "y ~ w + v + s(x)",
    "y ~ w + v + z + s(x)",
]


def _data() -> dict[str, Any]:
    rng = np.random.default_rng(SEED)
    x = rng.uniform(0.0, 1.0, N_ROWS)
    z = rng.uniform(0.0, 1.0, N_ROWS)
    return {
        "x": x,
        "z": z,
        "w": rng.standard_normal(N_ROWS),
        "v": rng.standard_normal(N_ROWS),
        "y": np.sin(2.0 * np.pi * x) + 0.15 * z + 0.3 * rng.standard_normal(N_ROWS),
    }


def _report_rows(model: gamfit.Model, tmp_path: Any) -> dict[str, float]:
    """Headline stat rows of the HTML report, by label."""
    path = tmp_path / "report.html"
    model.report(str(path))
    body = path.read_text().split("</style>")[-1]
    flat = re.sub(r"(\|\s*)+", "|", re.sub(r"<[^>]+>", "|", body))
    out: dict[str, float] = {}
    for label in ("Deviance", "REML / LAML", "EDF (total)", "Observations"):
        hit = re.search(re.escape(label) + r"\|(-?[0-9][0-9.eE+-]*)", flat)
        assert hit is not None, f"report has no {label!r} row"
        out[label] = float(hit.group(1))
    return out


def _close(a: float, b: float) -> bool:
    """Agreement to the report's own printed precision (4 decimals)."""
    return abs(a - b) <= 5.0e-4 * max(1.0, abs(b))


@pytest.mark.parametrize("formula", FORMULAS)
def test_control_the_other_headline_fields_agree(formula: str, tmp_path: Any) -> None:
    """Green today: deviance, total EDF and the observation count round-trip into
    the report exactly, so the report is faithful everywhere else."""
    model = gamfit.fit(_data(), formula, family="gaussian")
    summary = model.summary()
    rows = _report_rows(model, tmp_path)
    assert _close(rows["Deviance"], float(summary.deviance))
    assert _close(rows["EDF (total)"], float(summary.edf_total))
    assert rows["Observations"] == float(N_ROWS)


@pytest.mark.parametrize("formula", FORMULAS)
def test_control_the_report_value_is_the_raw_criterion(formula: str, tmp_path: Any) -> None:
    """Green today, and the diagnosis: the number the report prints is
    ``summary().raw_reml_score``, not ``summary().reml_score``."""
    model = gamfit.fit(_data(), formula, family="gaussian")
    summary = model.summary()
    rows = _report_rows(model, tmp_path)
    assert _close(rows["REML / LAML"], float(summary.raw_reml_score))


@pytest.mark.parametrize("formula", FORMULAS)
def test_report_reml_matches_the_summary_criterion(formula: str, tmp_path: Any) -> None:
    model = gamfit.fit(_data(), formula, family="gaussian")
    summary = model.summary()
    rows = _report_rows(model, tmp_path)
    assert _close(rows["REML / LAML"], float(summary.reml_score)), (
        f"{formula}: report prints {rows['REML / LAML']} where summary().reml_score "
        f"is {summary.reml_score} (raw_reml_score {summary.raw_reml_score})"
    )


def test_the_discrepancy_is_not_a_fixed_offset(tmp_path: Any) -> None:
    """A constant convention gap could be read around. This one tracks the
    model's unpenalized dimension, so report-to-report gaps between two models
    are not the summary's gaps."""
    data = _data()
    gaps: list[tuple[str, float, float]] = []
    for formula in FORMULAS:
        model = gamfit.fit(data, formula, family="gaussian")
        summary = model.summary()
        rows = _report_rows(model, tmp_path)
        gap = float(summary.reml_score) - rows["REML / LAML"]
        null_dim = float(summary["null_dim"])
        # The gap is exactly the Tierney-Kadane normalizer the summary applies.
        predicted = 0.5 * float(summary["null_space_logdet"]) - 0.5 * null_dim * math.log(
            2.0 * math.pi
        )
        assert abs(gap - predicted) < 1.0e-3, (
            f"{formula}: gap {gap} is not the null-space normalizer {predicted}"
        )
        gaps.append((formula, null_dim, gap))

    spread = max(g for _, _, g in gaps) - min(g for _, _, g in gaps)
    assert spread < 1.0e-3, (
        "the report/summary REML gap varies with the model's unpenalized "
        f"dimension, so report-to-report comparisons are not summary comparisons: {gaps}"
    )
