"""Keep the torch-dependent Python examples executable as standalone programs.

``tests/examples_smoke_test.py`` runs every example that does not import torch.
The ones that do run here, where ``tests/torch/conftest.py`` guarantees torch is
installed, so they are measured by the torch lane instead of failing on every
worker without the extra.
"""

from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess
import sys

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
IMPORTS_TORCH = re.compile(r"^\s*(?:import|from)\s+torch\b", re.MULTILINE)
TORCH_EXAMPLES = tuple(
    path
    for path in sorted((REPOSITORY_ROOT / "examples").glob("*.py"))
    if IMPORTS_TORCH.search(path.read_text(encoding="utf-8"))
)


def test_the_torch_example_population_is_not_empty() -> None:
    assert TORCH_EXAMPLES, "no example imports torch; the partition in examples_smoke_test is stale"


@pytest.mark.parametrize("example", TORCH_EXAMPLES, ids=lambda path: path.name)
def test_torch_example_runs(example: Path) -> None:
    environment = os.environ.copy()
    environment.setdefault("MPLBACKEND", "Agg")

    completed = subprocess.run(
        [sys.executable, str(example)],
        cwd=REPOSITORY_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert completed.returncode == 0, (
        f"{example.name} exited with {completed.returncode}\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )
