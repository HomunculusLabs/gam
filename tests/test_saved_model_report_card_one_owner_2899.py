"""One report card for a saved model on both front doors (#2899 F3).

``gam report`` and ``Model.report()`` each assembled their own card from the saved
model, and the two disagreed:

* the CLI printed the model class as the enum's Debug name (``Standard``), while
  Python and every prediction payload print ``standard``;
* the Python card had no optimality certificate and no smoothing-forensics section.

gam-models ``inference::saved_summary::saved_model_report_input`` is now the card
both render; ``gam report`` adds only what the data it is given can show.
"""

from __future__ import annotations

import importlib
import os
import re
import shutil
import subprocess
from typing import Any

pytest: Any = importlib.import_module("pytest")
np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("gamfit._rust")

import gamfit

N_ROWS = 400
SEED = 11
FORMULA = "y ~ s(x) + z"
CARD_ITEM = re.compile(
    r'<span class="stat-label">(.*?)</span><span class="stat-value">(.*?)</span>', re.S
)


def gam_bin() -> str:
    candidates = [
        os.environ.get("GAM_BIN"),
        "target/release/gam",
        "target/debug/gam",
        shutil.which("gam"),
    ]
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    pytest.skip("gam binary not built")


def _data() -> Any:
    rng = np.random.default_rng(SEED)
    x = rng.uniform(0.0, 1.0, N_ROWS)
    z = rng.standard_normal(N_ROWS)
    y = np.sin(2.0 * np.pi * x) + 0.2 * z + 0.3 * rng.standard_normal(N_ROWS)
    return pd.DataFrame({"x": x, "z": z, "y": y})


def _card(html: str) -> dict[str, str]:
    return {label: value for label, value in CARD_ITEM.findall(html)}


def _both_reports(tmp_path: Any) -> tuple[Any, str, str]:
    data = _data()
    model = gamfit.fit(data, FORMULA, family="gaussian")
    model_path = tmp_path / "model.json"
    model.save(model_path)
    data_path = tmp_path / "train.csv"
    data.to_csv(data_path, index=False)
    python_html = model.report()
    cli_path = tmp_path / "cli.report.html"
    subprocess.run(
        [gam_bin(), "report", str(model_path), str(data_path), str(cli_path)], check=True
    )
    return model, python_html, cli_path.read_text()


def test_both_front_doors_print_the_same_card(tmp_path: Any) -> None:
    model, python_html, cli_html = _both_reports(tmp_path)
    python_card = _card(python_html)
    cli_card = _card(cli_html)
    assert python_card["Model Class"] == model.summary().model_class
    for label in ("Family", "Model Class", "Observations", "Deviance", "REML / LAML"):
        assert python_card[label] == cli_card[label], (
            f"{label!r}: python card {python_card[label]!r}, CLI card {cli_card[label]!r}"
        )
    assert python_card["Observations"] == str(N_ROWS)


def test_the_python_card_carries_what_the_fit_recorded(tmp_path: Any) -> None:
    _model, python_html, cli_html = _both_reports(tmp_path)
    assert "<h2>Smoothing Forensics</h2>" in cli_html
    assert "<h2>Smoothing Forensics</h2>" in python_html
    assert ("Optimality Certificate" in _card(python_html)) == (
        "Optimality Certificate" in _card(cli_html)
    )
