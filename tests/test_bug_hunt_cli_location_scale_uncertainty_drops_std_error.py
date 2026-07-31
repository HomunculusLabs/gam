"""Regression for #2148: CLI Gaussian location-scale uncertainty must emit std_error.

The Rust prediction path computes the response-scale mean standard error and uses
it to form the symmetric mean band. This test exercises the CLI serialization
boundary so the location-scale CSV writer cannot silently drop that column.
"""

import csv
import math
import os
import random
import shutil
import subprocess
from pathlib import Path

_Z_975 = 1.959963984540054


def test_python_contracts_builds_cli_before_full_suite() -> None:
    """The wheel-only inventory must provision the binary this test exercises."""

    repo_root = Path(__file__).resolve().parents[1]
    workflow = (repo_root / ".github" / "workflows" / "python-contracts.yml").read_text()
    cli_build = workflow.index("cargo build --release -p gam-cli")
    full_suite = workflow.index("Triage full Python suite on the fresh wheel")
    assert cli_build < full_suite


def _gam_binary() -> str:
    """Resolve the `gam` CLI the way every other CLI repro in tests/ does.

    CI builds the RELEASE binary (`cargo build --release -p gam-cli`) before the
    full Python suite; this test named `target/debug/gam` unconditionally, so it
    never reached the CLI serialization boundary it exists to guard — it died in
    `subprocess.Popen` with `FileNotFoundError: 'target/debug/gam'`.
    """

    repo_root = Path(__file__).resolve().parents[1]
    for candidate in (
        os.environ.get("GAM_BIN"),
        repo_root / "target" / "release" / "gam",
        repo_root / "target" / "debug" / "gam",
        shutil.which("gam"),
    ):
        if candidate and Path(candidate).exists():
            return str(candidate)
    # No skip (SPEC 16): an unbuilt CLI is a real gap in this measurement and
    # must read red, but it must say so instead of surfacing as a bare ENOENT.
    raise AssertionError(
        "no `gam` CLI binary found (GAM_BIN, target/release/gam, target/debug/gam, PATH)"
    )


def test_cli_location_scale_uncertainty_preserves_std_error(tmp_path: Path) -> None:
    data_path = tmp_path / "ls.csv"
    model_path = tmp_path / "ls.gam"
    pred_path = tmp_path / "ls_pred.csv"

    rng = random.Random(51)
    with data_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["y", "x"])
        for _ in range(150):
            x = rng.gauss(0.0, 1.0)
            y = 1.0 + 2.0 * x + rng.gauss(0.0, math.exp(0.2 + 0.3 * x))
            writer.writerow([f"{y:.8f}", f"{x:.8f}"])

    gam = _gam_binary()
    subprocess.run(
        [
            gam,
            "fit",
            "--family",
            "gaussian",
            "--predict-noise",
            "x",
            "--out",
            str(model_path),
            str(data_path),
            "y ~ x",
        ],
        check=True,
    )
    subprocess.run(
        [
            gam,
            "predict",
            "--uncertainty",
            "--out",
            str(pred_path),
            str(model_path),
            str(data_path),
        ],
        check=True,
    )

    with pred_path.open(newline="") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == [
            "eta",
            "mean",
            "sigma",
            "std_error",
            "mean_lower",
            "mean_upper",
        ]
        rows = list(reader)

    assert rows
    for row in rows[:10]:
        mean = float(row["mean"])
        std_error = float(row["std_error"])
        mean_upper = float(row["mean_upper"])
        assert std_error > 0.0
        assert math.isclose(
            mean_upper - mean,
            std_error * _Z_975,
            rel_tol=1e-8,
            abs_tol=1e-8,
        )
