#!/usr/bin/env python3
"""Run every original #2668 contract; missing tests and timeouts are not passes."""

import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time


# Each row names the test binary it lives in. #2899 (5f1c8e4d80) moved rows 5 and 18
# out of the root regressions binary into gam-predict's lib tests with their names kept.
HARNESSES = {
    "regressions": (["test", "-p", "gam", "--test", "regressions", "--no-run"], "regressions"),
    "gam-predict-lib": (["test", "-p", "gam-predict", "--lib", "--no-run"], "gam_predict"),
}


def build(root, cargo_args):
    """Build under the test profile; return cargo's artifact records for this invocation."""
    # Always the test profile. `cargo build -p gam-cli` alone is the dev profile, which
    # leaves the workspace crates at opt-level 0 and still writes target/debug/gam: row 29's
    # `gam fit` took 198 s with that binary and 13 s with the test-profile one, at the same
    # 116 outer evaluations (MSI job 1113131 at de910ee411).
    process = subprocess.run(
        ["cargo", *cargo_args, "--profile", "test", "--message-format=json-render-diagnostics"],
        cwd=root, check=True, stdout=subprocess.PIPE, text=True,
    )
    records = (json.loads(line) for line in process.stdout.splitlines() if line.startswith("{"))
    return [r for r in records if r.get("reason") == "compiler-artifact" and r.get("executable")]


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()
    if args.workers < 1 or args.timeout <= 0:
        parser.error("workers and timeout must be positive")
    root = Path(__file__).resolve().parent.parent
    entries = json.loads((root / "tests/data/issue_2668_regressions.json").read_text())
    unknown = sorted({entry["harness"] for entry in entries} - HARNESSES.keys())
    if unknown:
        parser.error(f"rows name unknown harnesses {unknown}")
    args.output.mkdir(parents=True, exist_ok=True)

    cli_artifacts = [r for r in build(root, ["build", "-p", "gam-cli", "--bin", "gam"])
                     if r["target"]["name"] == "gam" and "bin" in r["target"]["kind"]]
    if len(cli_artifacts) != 1 or cli_artifacts[0]["profile"]["opt_level"] == "0":
        parser.error(f"expected one optimized gam CLI artifact, got {cli_artifacts}")
    cli = Path(cli_artifacts[0]["executable"])
    binaries = {}
    for harness, (cargo_args, target) in HARNESSES.items():
        artifacts = [r for r in build(root, cargo_args)
                     if r["target"]["name"] == target and r["profile"]["test"]]
        if len(artifacts) != 1:
            parser.error(f"expected one {harness} test binary, got {artifacts}")
        binaries[harness] = Path(artifacts[0]["executable"])
    # Rows that drive the CLI spawn `target/<profile>/gam` beside the test binary
    # (`gam_test_support::cli_harness::resolve_gam_binary`), so that path must be the
    # binary just built, or the receipt does not name the code those rows ran.
    if binaries["regressions"].parent.parent / "gam" != cli:
        parser.error(f"{cli} is not the gam the regressions binary spawns")

    names = {}
    for harness, binary in binaries.items():
        inventory = subprocess.run(
            [str(binary), "--list", "--format", "terse"],
            check=True, capture_output=True, text=True,
        ).stdout
        (args.output / f"inventory-{harness}.txt").write_text(inventory)
        names[harness] = [line.removesuffix(": test") for line in inventory.splitlines()
                          if line.endswith(": test")]
    scratch = args.output.resolve().with_name(args.output.name + "-scratch")
    scratch.mkdir(exist_ok=True)
    digests = {harness: sha256(binary) for harness, binary in binaries.items()}
    cli_digest = sha256(cli)
    environment = dict(os.environ, TMPDIR=str(scratch), RAYON_NUM_THREADS="2",
                       OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")

    def run(entry):
        harness = entry["harness"]
        matches = [name for name in names[harness] if name.split("::")[-1] == entry["test"]]
        record = dict(entry, matches=matches)
        if len(matches) != 1:
            return dict(record, status="missing" if not matches else "ambiguous")
        log = args.output / (entry["test"] + ".log")
        start = time.monotonic()
        with log.open("w") as stream:
            process = subprocess.Popen(
                [str(binaries[harness]), "--exact", matches[0], "--nocapture", "--test-threads=1"],
                cwd=root, env=environment, stdout=stream, stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                code = process.wait(timeout=args.timeout)
                status = "passed" if code == 0 else "failed"
            except subprocess.TimeoutExpired:
                # This process group belongs exclusively to the test we started.
                os.killpg(process.pid, signal.SIGKILL)
                code = process.wait()
                status = "timeout"
        if status == "passed" and "1 passed; 0 failed; 0 ignored;" not in log.read_text():
            status = "unmeasured"
        record.update(status=status, exit_code=code,
                      seconds=time.monotonic() - start, log=str(log))
        return record

    records = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run, entry) for entry in entries]
        for future in concurrent.futures.as_completed(futures):
            record = future.result()
            records.append(record)
            print(f"{record['status']:10} {record['original']}", flush=True)
            receipt = dict(binaries={h: str(b) for h, b in binaries.items()},
                           binary_sha256=digests, gam_binary=str(cli), gam_sha256=cli_digest,
                           gam_opt_level=cli_artifacts[0]["profile"]["opt_level"],
                           test_timeout_seconds=args.timeout, workers=args.workers,
                           results=records)
            (args.output / "results.json").write_text(json.dumps(receipt, indent=2) + "\n")
    counts = {status: sum(r["status"] == status for r in records)
              for status in sorted({r["status"] for r in records})}
    final_digests = {harness: sha256(binary) for harness, binary in binaries.items()}
    cli_final_digest = sha256(cli)
    receipt["binary_sha256_after"] = final_digests
    receipt["binary_unchanged"] = final_digests == digests
    receipt["gam_sha256_after"] = cli_final_digest
    receipt["gam_unchanged"] = cli_final_digest == cli_digest
    (args.output / "results.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(counts, sort_keys=True), flush=True)
    unchanged = final_digests == digests and cli_final_digest == cli_digest
    return 0 if unchanged and len(records) == 30 and counts == {"passed": 30} else 1


if __name__ == "__main__":
    raise SystemExit(main())
