"""Regression guard for #1393 — a slow test is named, never bulk-killed.

Issue #1393: after #1146 folded ~700 per-file integration crates into a handful
of aggregator binaries, a few of those binaries hold hundreds of `#[test]`s
each (quality ~427, basis_smooth ~292, manifolds ~152). Run serially under
`--test-threads 1` (the peak-RSS guard), the SUM of the legitimately-slow
REML/PIRLS/joint-Newton and R/Python reference-comparison tests in one binary
overran an 1800s per-binary GNU `timeout`, which SIGKILLed the whole target
(exit 124/137) BEFORE the per-test summary printed — so the entire target
failed in bulk with NO attribution of which test was slow.

THE PRINCIPLE, which is what this file pins:

  1. ATTRIBUTION. An overrunning test is terminated and reported BY NAME. A
     wall-clock cap must never be applied to a whole test binary, because
     killing the binary destroys the very evidence that says which test was
     slow.
  2. DIVISION. The oversized test surface is divided across parallel runners,
     so no single runner serially absorbs hundreds of multi-minute fits.
  3. COMPLETENESS. Every shard reports every one of its failures — a shard
     does not stop at the first one.

THE MECHANISM HAS CHANGED ONCE, and the principle survived it. The original
fix (commit d74ef8af) kept the GNU `timeout` wrapper and merely lowered what
it wrapped, splitting each oversized BINARY into `--partition count:i/N`
shards under a second, larger `PARTITIONED_BINARY_TIMEOUT` cap. The workflow
has since been redesigned around nextest's build-once/run-many archive model:
one `cargo nextest archive` job, then a `strategy.matrix` of run-shards that
each filter the WHOLE prebuilt archive with `--partition count:i/N`. That
redesign deleted the per-binary partition map, both GNU `timeout` caps, and
the `timeout`-wrapped invocation altogether — attribution now rests entirely
on nextest's own per-test `slow-timeout`/`terminate-after`, which names the
test it kills. That is strictly stronger than what #1393 asked for.

So this file asserts the three properties above, NOT the retired spelling of
them. Pinning `partitions_for_binary` / `PARTITIONED_BINARY_TIMEOUT` would
have made a genuine improvement look like a regression, which is how a
contract test turns into a brake. In particular, property 1 is now stated in
the form that can still bite: the bulk-kill mechanism #1393 was filed against
must be ABSENT.

This is a pure-source contract test — it reads the workflow and the nextest
config, and builds nothing.
"""

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def _shell_commands(text):
    """Rejoin backslash-continued shell lines into whole commands.

    Flags of one invocation are spread over many YAML lines, so a per-line or
    whole-file substring test cannot tell "this command has the flag" from
    "some other command in the file has it".
    """
    commands, current = [], []
    for line in text.splitlines():
        stripped = line.strip()
        current.append(stripped[:-1].strip() if stripped.endswith("\\") else stripped)
        if not stripped.endswith("\\"):
            commands.append(" ".join(current))
            current = []
    if current:
        commands.append(" ".join(current))
    return commands


def test_nextest_profile_terminates_and_names_slow_tests():
    """Property 1a — the per-test layer that produces the attribution exists."""
    cfg = _read(".config/nextest.toml")
    assert "[profile.ci]" in cfg, "ci nextest profile missing"
    # A single overrunning #[test] must be SIGKILLed and reported by name —
    # this is the layer that makes a bulk binary cap unnecessary.
    assert "slow-timeout" in cfg, "per-test slow-timeout removed (#1393)"
    assert "terminate-after" in cfg, "terminate-after removed; hangs would run unbounded (#1393)"
    # no-fail-fast preserves the no-fail-fast contract across all shards.
    assert "fail-fast = false" in cfg, "fail-fast must stay false so all shards report (#1393)"


def test_no_wall_clock_cap_bulk_kills_a_test_binary():
    """Property 1b — the bulk-kill mechanism #1393 was filed against is absent.

    A GNU `timeout` wrapped around a test-runner invocation is exactly the
    defect: it SIGKILLs the process group before the per-test summary prints,
    so the run reports "the binary died" instead of "this test was slow".
    Attribution belongs to nextest's per-test terminate-after (asserted
    above), which is the only layer that knows a test's name.
    """
    wf = _read(".github/workflows/test.yml")
    offenders = []
    for lineno, line in enumerate(wf.splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue  # prose describing the retired mechanism is not the mechanism
        # `timeout 1800 cargo nextest ...` / `timeout --signal=... <runner>`
        if re.match(r"^timeout\s+(--\S+\s+)*\d", stripped) or re.search(
            r"\btimeout\s+(--\S+\s+)*\d+\s+\S*(cargo|nextest|/deps/)", stripped
        ):
            offenders.append(f"{lineno}: {stripped}")
    assert not offenders, (
        "a GNU `timeout` wall-clock cap wraps a test invocation again — that "
        "bulk-kills the binary and destroys per-test attribution, which is the "
        "exact defect of #1393:\n  " + "\n  ".join(offenders)
    )


def test_workflow_divides_the_test_surface_across_parallel_shards():
    """Properties 2 and 3 — the surface is partitioned, and every shard reports."""
    # Read the RUN steps only. The header prose documents the design using the
    # same `--partition count:i/N` spelling, and a guard that reads its own
    # documentation proves nothing about the workflow.
    wf_code = "\n".join(
        line for line in _read(".github/workflows/test.yml").splitlines()
        if not line.strip().startswith("#")
    )

    # The run-shards must filter the prebuilt surface through nextest's
    # partitioner. `count:i/N` is the form that divides by test COUNT; the
    # denominator must come from the same value that sizes the matrix, or
    # shards would silently overlap or leave a gap in the surface.
    assert "--partition" in wf_code, "--partition sharding removed (#1393)"
    # `${{ ... }}` expressions contain spaces, so the quoted form must be read
    # to its closing quote rather than to the first space.
    partitions = re.findall(r'--partition\s+"count:([^"]+)"', wf_code)
    partitions += re.findall(r"--partition\s+count:(\S+)", wf_code)
    assert partitions, "partition must use nextest's count:i/N filter form (#1393)"
    for spec in partitions:
        assert "/" in spec, f"partition spec {spec!r} is not i/N (#1393)"
        index, _, denominator = spec.partition("/")
        assert "matrix." in index, (
            f"partition index {index!r} must come from the job matrix so the "
            "shards together cover the whole surface (#1393)"
        )
        # The denominator must be the SAME value that sizes the matrix, not a
        # literal. A hardcoded `count:i/1` would make "division" a no-op and
        # put the whole serial surface back on one runner.
        assert "needs." in denominator or "matrix." in denominator, (
            f"partition denominator {denominator!r} is a literal; it must be "
            "the value that sizes the shard matrix, or the shards will not "
            "tile the surface (#1393)"
        )

    # A shard that stops at its first failure hides the rest of its surface,
    # which is the same "no attribution" defect one level down. This must be
    # checked on the PARTITIONED invocation itself: `--no-fail-fast` elsewhere
    # in the file (the gam-pyffi job has its own) would otherwise satisfy a
    # bare substring test while the shard step quietly lost the flag.
    partitioned_commands = [
        cmd for cmd in _shell_commands(wf_code) if "--partition" in cmd
    ]
    assert partitioned_commands, "no partitioned test invocation found (#1393)"
    for cmd in partitioned_commands:
        assert "--no-fail-fast" in cmd, (
            "the partitioned run-shard invocation lost --no-fail-fast; the "
            "shard would stop at its first failure and the rest of its "
            "surface would go unreported (#1393):\n  " + " ".join(cmd.split())
        )
