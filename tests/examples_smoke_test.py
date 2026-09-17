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
# An example whose measured cost is legitimate but too large for the default job is
# marked slow, beside its measurement, and runs where slow tests are scheduled.
SLOW_EXAMPLES = {
    # 42.3 s end to end on acn112 at RAYON_NUM_THREADS=4 against the 60 s cap below.
    # 31.28 s of it is the torus te(k=[20, 20]) candidate: p = 400 dense
    # factorizations at 3.9 s per outer iteration, the p³ cost of Torus()'s default
    # basis (the cylinder at p = 160 runs 0.24 s per iteration, and (400/160)³ = 15.6).
    "topology_selection_demo.py",
}
EXAMPLES = tuple(
    pytest.param(
        path,
        id=path.name,
        marks=(pytest.mark.slow,) if path.name in SLOW_EXAMPLES else (),
    )
    for path in sorted((REPOSITORY_ROOT / "examples").glob("*.py"))
    if not IMPORTS_TORCH.search(path.read_text(encoding="utf-8"))
)


@pytest.mark.parametrize("example", EXAMPLES)
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
