"""Keep the user-facing Python examples executable as standalone programs."""

from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess
import sys

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
# An example that imports torch belongs to the torch population: it runs in
# tests/torch/test_torch_examples_run.py, which is collected where torch is
# installed, instead of failing here on every worker without torch.
IMPORTS_TORCH = re.compile(r"^\s*(?:import|from)\s+torch\b", re.MULTILINE)
EXAMPLES = tuple(
    path
    for path in sorted((REPOSITORY_ROOT / "examples").glob("*.py"))
    if not IMPORTS_TORCH.search(path.read_text(encoding="utf-8"))
)


@pytest.mark.parametrize("example", EXAMPLES, ids=lambda path: path.name)
def test_python_example_runs(example: Path) -> None:
    environment = os.environ.copy()
    # Plotting demonstrations must also run on headless CI workers.
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
