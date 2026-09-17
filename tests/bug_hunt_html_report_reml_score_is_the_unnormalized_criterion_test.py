"""Bug hunt: ``Model.report()`` printed the UN-normalized outer-optimizer
criterion under the label ``REML / LAML``, while ``Model.summary()`` publishes the
rank-aware normalized one under the same name. Two first-party surfaces of one
fitted model, one label, two different numbers.

``Summary`` deliberately carries both, and says which is which::

    /// Cross-model comparable criterion: `raw_reml_score` plus the rank-aware
    /// Tierney-Kadane normalizer over the penalty null space.
    reml_score: Option<f64>,
    /// The outer optimizer's own criterion value, un-normalized.
    raw_reml_score: Option<f64>,

``docs/getting-started.md`` points a reader at the first one ("REML/LAML
criterion (in the ``reml_score`` field)"). The report was wired to the second:
``ReportInput::reml_score`` was filled from ``UnifiedFitResult::reml_score`` --
the raw one -- and rendered with the row label ``"REML / LAML"``.

Measured (n=500, ``y = sin(2*pi*x) + 0.15*z + N(0, 0.3)``, one seed; the report's
value equalled ``summary().raw_reml_score`` to its printed precision in every row):

    formula                summary().reml_score   report "REML / LAML"   gap    null_dim
    y ~ s(x)                    142.39836              140.21000        2.188      1
    y ~ s(x) + s(z)             134.68015              132.49179        2.188      1
    y ~ z + s(x)                146.07769              142.93249        3.145      2
    y ~ w + s(x)                156.12697              151.70240        4.425      2
    y ~ w + v + s(x)            170.03508              163.44422        6.591      3
    y ~ w + v + z + s(x)        174.09721              166.55231        7.545      4

The gap is exactly the normalizer the summary applies,
``0.5*null_space_logdet - 0.5*null_dim*log(2*pi)``. At 69befd10c5 every formula
above reports ``null_dim == 1``, so the gap measured 2.188 in all six rows, but the
two surfaces still disagreed on the one label they share.

The comparable criterion is now one Rust function,
``gam_solve::topology_selector::comparable_reml_score`` (reached through
``UnifiedFitResult::comparable_reml_score``). The summary, ``compare_models``, and
both report producers (``gam report`` and ``Model.report()``) read it, so the
report's ``REML / LAML`` row is the criterion ``summary()`` calls ``reml_score``.
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
    """Deviance, total EDF and the observation count round-trip into the report
    exactly, so the report is faithful everywhere else."""
    model = gamfit.fit(_data(), formula, family="gaussian")
    summary = model.summary()
    rows = _report_rows(model, tmp_path)
    assert _close(rows["Deviance"], float(summary.deviance))
    assert _close(rows["EDF (total)"], float(summary.edf_total))
    assert rows["Observations"] == float(N_ROWS)


@pytest.mark.parametrize("formula", FORMULAS)
def test_control_the_report_value_is_the_raw_criterion_plus_the_normalizer(
    formula: str, tmp_path: Any
) -> None:
    """The report's row sits above ``summary().raw_reml_score`` by exactly the
    null-space normalizer. Every formula here has a nonzero normalizer, so a
    report that went back to printing the raw criterion fails this check."""
    model = gamfit.fit(_data(), formula, family="gaussian")
    summary = model.summary()
    rows = _report_rows(model, tmp_path)
    null_dim = float(summary["null_dim"])
    normalizer = 0.5 * float(summary["null_space_logdet"]) - 0.5 * null_dim * math.log(
        2.0 * math.pi
    )
    assert abs(normalizer) > 1.0e-2, (
        f"{formula}: normalizer {normalizer} is too small to tell the raw criterion apart"
    )
    gap = rows["REML / LAML"] - float(summary.raw_reml_score)
    assert _close(gap, normalizer), (
        f"{formula}: report sits {gap} above raw_reml_score, not the normalizer {normalizer}"
    )


@pytest.mark.parametrize("formula", FORMULAS)
def test_report_reml_matches_the_summary_criterion(formula: str, tmp_path: Any) -> None:
    model = gamfit.fit(_data(), formula, family="gaussian")
    summary = model.summary()
    rows = _report_rows(model, tmp_path)
    assert _close(rows["REML / LAML"], float(summary.reml_score)), (
        f"{formula}: report prints {rows['REML / LAML']} where summary().reml_score "
        f"is {summary.reml_score} (raw_reml_score {summary.raw_reml_score})"
    )
