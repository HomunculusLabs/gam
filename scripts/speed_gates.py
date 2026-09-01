#!/usr/bin/env python3
"""Derive and run every wall-clock speed gate in the workspace (#932).

A speed gate is a `#[test]` whose body opens a `gam_math::paired_timing::SpeedGate`.
That call site is the ONLY marker: the population is derived from source, never
from a name prefix, so a gate cannot be missed by a filter, and the run refuses
to report success unless every derived gate actually executed and passed.

    scripts/speed_gates.py                 # list the derived population
    scripts/speed_gates.py --run           # build (release) + run every gate
    scripts/speed_gates.py --run --package gam-models --summary out.md

Why the release profile: `[profile.test.package.gam-models]` sets
`codegen-units = 16` and the test profile carries no LTO, while the shipped
profile is `codegen-units = 1` + thin-LTO. A compiled-vs-hand ratio whose margin
is cross-CGU inlining measures a different program in the test profile, which is
why `SpeedGate::open` returns `None` outside the release profile and why this
runner always passes `--release`.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

MARKER = "SpeedGate::open("
TEST_ATTR = re.compile(r"#\[test\]")
FN_HEAD = re.compile(r"\bfn\s+([A-Za-z0-9_]+)\s*(?:<[^>]*>)?\s*\(")


@dataclass(frozen=True)
class Gate:
    package: str
    target_args: tuple[str, ...]
    name: str
    path: str

    @property
    def target_key(self) -> tuple[str, tuple[str, ...]]:
        return (self.package, self.target_args)


def repo_root() -> Path:
    out = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True)
    return Path(out.strip())


def package_name(crate_dir: Path) -> str:
    manifest = (crate_dir / "Cargo.toml").read_text()
    match = re.search(r'^\s*name\s*=\s*"([^"]+)"', manifest, re.M)
    if not match:
        raise SystemExit(f"no package name in {crate_dir / 'Cargo.toml'}")
    return match.group(1)


def target_for(crate_dir: Path, file: Path) -> tuple[str, ...] | None:
    relative = file.relative_to(crate_dir)
    parts = relative.parts
    if parts[0] == "src":
        return ("--lib",)
    if parts[0] == "tests":
        if len(parts) == 2 and parts[1].endswith(".rs"):
            return ("--test", parts[1][: -len(".rs")])
        if len(parts) >= 3:
            return ("--test", parts[1])
    return None


def test_fns_with_marker(source: str):
    """Yield the name of every `#[test]` fn whose body contains MARKER."""
    for attr in TEST_ATTR.finditer(source):
        head = FN_HEAD.search(source, attr.end())
        if not head:
            continue
        # Another `#[test]` before this fn means the attribute belonged to an
        # item that is not a fn with a body we can find; skip it.
        between = source[attr.end() : head.start()]
        if TEST_ATTR.search(between):
            continue
        open_brace = source.find("{", head.end())
        if open_brace < 0 or ";" in source[head.end() : open_brace]:
            continue
        depth = 0
        cursor = open_brace
        while cursor < len(source):
            char = source[cursor]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    break
            cursor += 1
        body = source[open_brace : cursor + 1]
        if MARKER in body:
            yield head.group(1)


def derive(root: Path) -> list[Gate]:
    gates: list[Gate] = []
    for crate_dir in sorted((root / "crates").iterdir()):
        if not (crate_dir / "Cargo.toml").is_file():
            continue
        package = package_name(crate_dir)
        for file in sorted(crate_dir.rglob("*.rs")):
            if "target" in file.parts:
                continue
            source = file.read_text()
            if MARKER not in source:
                continue
            target = target_for(crate_dir, file)
            if target is None:
                raise SystemExit(f"{file}: a speed gate must live under src/ or tests/")
            for name in test_fns_with_marker(source):
                gates.append(Gate(package, target, name, str(file.relative_to(root))))
    return gates


def cargo_test(package: str, target_args: tuple[str, ...], extra: list[str], root: Path):
    command = ["cargo", "test", "--release", "-p", package, *target_args, *extra]
    print("$", " ".join(command), flush=True)
    return subprocess.run(command, cwd=root, text=True, capture_output=True)


def resolve_exact(package: str, target_args: tuple[str, ...], names: list[str], root: Path):
    """Map each bare gate name to its exact test path in the compiled binary."""
    listing = cargo_test(package, target_args, ["--", "--list", "--format", "terse"], root)
    if listing.returncode != 0:
        sys.stdout.write(listing.stdout)
        sys.stderr.write(listing.stderr)
        raise SystemExit(f"{package} {' '.join(target_args)}: listing the test binary failed")
    listed = [line[: -len(": test")] for line in listing.stdout.splitlines() if line.endswith(": test")]
    resolved = {}
    for name in names:
        matches = [path for path in listed if path == name or path.endswith("::" + name)]
        if len(matches) != 1:
            raise SystemExit(
                f"{package} {' '.join(target_args)}: gate `{name}` resolved to {len(matches)} "
                f"tests in the compiled binary ({matches}); a derived gate must resolve to exactly one"
            )
        resolved[name] = matches[0]
    return resolved


RESULT_LINE = re.compile(r"^test result: (ok|FAILED)\. (\d+) passed; (\d+) failed", re.M)


def run(gates: list[Gate], root: Path, summary_path: str | None) -> int:
    failures: list[str] = []
    rows: list[str] = []
    by_target: dict[tuple[str, tuple[str, ...]], list[Gate]] = {}
    for gate in gates:
        by_target.setdefault(gate.target_key, []).append(gate)
    for (package, target_args), members in by_target.items():
        build = cargo_test(package, target_args, ["--no-run"], root)
        sys.stdout.write("\n".join(line for line in build.stderr.splitlines() if not line.lstrip().startswith(("Compiling", "Checking"))) + "\n")
        if build.returncode != 0:
            raise SystemExit(f"{package} {' '.join(target_args)}: the release test target did not build")
        exact = resolve_exact(package, target_args, [g.name for g in members], root)
        for gate in members:
            path = exact[gate.name]
            result = cargo_test(
                package,
                target_args,
                ["--", "--exact", path, "--nocapture", "--test-threads=1"],
                root,
            )
            output = result.stdout + result.stderr
            verdict_lines = [line for line in output.splitlines() if " verdict=" in line]
            for line in verdict_lines:
                print(line, flush=True)
                rows.append(f"| `{gate.name}` | `{line.strip()}` |")
            match = RESULT_LINE.search(output)
            if not match:
                failures.append(f"{gate.name}: no `test result:` line (binary aborted?)")
                sys.stdout.write(output)
                continue
            status, passed, failed = match.group(1), int(match.group(2)), int(match.group(3))
            if passed + failed != 1:
                failures.append(f"{gate.name}: expected exactly one test to execute, saw {passed + failed}")
            if status != "ok" or result.returncode != 0:
                failures.append(f"{gate.name}: FAILED")
                sys.stdout.write(output)
            if not verdict_lines:
                failures.append(f"{gate.name}: printed no cell verdict; a gate that opens must record a cell")
            print(f"[{status}] {gate.package} {' '.join(gate.target_args)} {path}", flush=True)
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as handle:
            handle.write(f"### Speed gates: {len(gates)} derived, {len(gates) - len(failures)} passed\n\n")
            handle.write("| gate | cell |\n|---|---|\n")
            handle.write("\n".join(rows) + "\n")
            if failures:
                handle.write("\n**Failures**\n\n" + "\n".join(f"- {f}" for f in failures) + "\n")
    if failures:
        print("\nSPEED GATES FAILED:\n" + "\n".join(failures), file=sys.stderr)
        return 1
    print(f"\nall {len(gates)} derived speed gates executed and passed")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="store_true", help="build in release and run every derived gate")
    parser.add_argument("--package", action="append", help="restrict to these packages (repeatable)")
    parser.add_argument("--summary", help="append a markdown table of cell verdicts to this file")
    parser.add_argument("--expect", type=int, help="fail unless exactly this many gates are derived")
    args = parser.parse_args()
    root = repo_root()
    gates = derive(root)
    if args.package:
        gates = [g for g in gates if g.package in set(args.package)]
    if not gates:
        print("no speed gates derived", file=sys.stderr)
        return 1
    if args.expect is not None and len(gates) != args.expect:
        print(f"derived {len(gates)} gates, expected {args.expect}", file=sys.stderr)
        return 1
    for gate in gates:
        print(f"{gate.package}\t{' '.join(gate.target_args)}\t{gate.name}\t{gate.path}")
    if not args.run:
        return 0
    return run(gates, root, args.summary)


if __name__ == "__main__":
    os.environ.setdefault("CARGO_TERM_COLOR", "never")
    sys.exit(main())
