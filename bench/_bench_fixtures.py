"""The benchmark suite's fixtures, served by the Rust `bench_fixtures` binary.

Seeded synthetic panels, cross-validation folds and per-fold z-scores are harness
choices, so they live in `crates/gam-test-support` rather than in the production
`gamfit._rust` extension. Each call runs the binary once: arrays go in as C-order
`.npy` files and scalars as decimal text, and every `.npy` array and `.txt` string
column the binary writes comes back under its file stem.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import typing
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
_BINARY: Path | None = None


def bench_fixtures_binary() -> Path:
    """`BENCH_FIXTURES_BIN` when it is set, otherwise a release build from this checkout."""
    global _BINARY
    if _BINARY is not None:
        return _BINARY
    prebuilt = os.environ.get("BENCH_FIXTURES_BIN", "").strip()
    if prebuilt:
        candidate = Path(prebuilt).expanduser().resolve()
        if not candidate.is_file() or not os.access(candidate, os.X_OK):
            raise RuntimeError(f"BENCH_FIXTURES_BIN is not an executable file: {candidate}")
        _BINARY = candidate
        return _BINARY
    build = subprocess.run(
        ["cargo", "build", "--release", "-p", "gam-test-support", "--bin", "bench_fixtures"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if build.returncode != 0:
        raise RuntimeError((build.stderr or build.stdout).strip() or "failed to build bench_fixtures")
    built = ROOT / "target" / "release" / "bench_fixtures"
    if not built.is_file():
        raise RuntimeError(f"missing bench_fixtures binary at {built}")
    _BINARY = built
    return _BINARY


def _scalar_text(value: typing.Any) -> str:
    if isinstance(value, (bool, np.bool_)):
        return "true" if value else "false"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return repr(float(value))
    return str(value)


def run_fixture(
    fixture: str,
    *,
    arrays: typing.Mapping[str, typing.Any] | None = None,
    **scalars: typing.Any,
) -> dict[str, typing.Any]:
    """Run one fixture and return its outputs: `.npy` stems as arrays, `.txt` stems as string lists."""
    with tempfile.TemporaryDirectory(prefix="bench-fixtures-") as tmp:
        tmp_path = Path(tmp)
        out_dir = tmp_path / "out"
        args = [str(bench_fixtures_binary()), fixture, str(out_dir)]
        for key, array in (arrays or {}).items():
            path = tmp_path / f"in_{key}.npy"
            np.save(path, np.ascontiguousarray(array))
            args.append(f"{key}={path}")
        for key, value in scalars.items():
            args.append(f"{key}={_scalar_text(value)}")
        done = subprocess.run(args, capture_output=True, text=True)
        if done.returncode != 0:
            raise RuntimeError(f"bench_fixtures {fixture} failed: {done.stderr.strip()}")
        outputs: dict[str, typing.Any] = {}
        for path in sorted(out_dir.iterdir()):
            if path.suffix == ".npy":
                outputs[path.stem] = np.load(path)
            elif path.suffix == ".txt":
                outputs[path.stem] = path.read_text().splitlines()
        return outputs
