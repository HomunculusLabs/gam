"""Recover the diagnosis for tests a timeout kill erases from the JUnit inventory.

`--timeout-method=thread` terminates the xdist worker instead of raising inside
it, so a test that overruns the per-test bound lands in the JUnit as
`<error message="worker 'gwN' crashed while running ..."/>` with `time="0.000"`.
That record carries no reason and no duration: the reader of the published
inventory cannot tell a hung fit from a crashed interpreter, and cannot tell how
long the test actually ran. On run 30637393388 that was 34 of 137 red tests —
a quarter of the red population reported as unexplained.

The diagnosis is not actually lost. pytest's faulthandler plugin is armed at
`faulthandler_timeout` (below the kill bound) and writes its dump from C
straight to fd 2, so it survives the `os._exit` that eats pytest-timeout's own
dump. Every one of those 34 workers therefore printed a full stack naming the
hung test and the frame it was stuck in — into a five-figure line count of
build output that nothing reads.

This script joins the two: the JUnit says WHICH tests were censored, the stdout
log says WHERE each one was stuck, and the result is written to the
`py1512_summary.txt` that `python-contracts.yml` already promised to publish and
never produced.

Usage:  py1512_summary.py <junit.xml> <pytest-stdout.log> <out.txt>
"""

from __future__ import annotations

import collections
import re
import sys
import xml.etree.ElementTree as ET

# `gh run view --log` prefixes every line with "<job>\t<step>\t<timestamp> ".
# Raw step output has no prefix. Tolerate both so the script gives the same
# answer on a downloaded log as it does in-job.
_LOG_PREFIX = re.compile(r"^[^\t]*\t[^\t]*\t\d{4}-\d\d-\d\dT[\d:.]+Z ")
# NOT anchored to the start of the line. faulthandler writes its header to fd 2
# with no regard for whether pytest's progress stream has emitted a newline yet,
# so `.....Timeout (0:04:00)!` is as common as a header on its own line — and
# which one you get is a race between two writers. Anchoring cost 24 of 29
# recoveries on run 30657860492, where 37 dumps were present and NOT ONE started
# a line, while the run before it had 43 that mostly did.
_DUMP_HEADER = re.compile(r"Timeout \(\d+:\d\d:\d\d\)!")
# Same reasoning for the end of a dump: the worker's last frame can be followed
# on the same line by whatever the controller writes next.
_DUMP_TAIL = 'File "<string>", line 1 in <module>'
_TEST_FRAME = re.compile(r'File "([^"]*[/\\]tests[/\\][^"]+)", line \d+ in (\w+)')
_PKG_FRAME = re.compile(r'File "[^"]*[/\\]site-packages[/\\]gamfit[/\\]([^"]+)", line (\d+) in (\w+)')


def _crashed_message(message: str) -> bool:
    return "crashed while running" in message


def read_junit(path):
    """Return (counts, censored, diagnosed) from the JUnit inventory."""
    root = ET.parse(path).getroot()
    counts = collections.Counter()
    censored = []
    diagnosed = []
    for case in root.iter("testcase"):
        name = case.get("name") or "?"
        where = case.get("file") or case.get("classname") or "?"
        seconds = float(case.get("time") or 0.0)
        failure = case.find("failure")
        error = case.find("error")
        skipped = case.find("skipped")
        if failure is not None:
            counts["failed"] += 1
            diagnosed.append((where, name, seconds, failure.get("message") or ""))
        elif error is not None:
            message = error.get("message") or ""
            if _crashed_message(message):
                counts["censored"] += 1
                censored.append((where, name))
            else:
                counts["errored"] += 1
                diagnosed.append((where, name, seconds, message))
        elif skipped is not None:
            counts["skipped"] += 1
        else:
            counts["passed"] += 1
    return counts, censored, diagnosed


def read_dumps(path):
    """Map test function name -> Counter of the gamfit frame it was stuck in.

    A faulthandler dump reports every frame, innermost first. The INNERMOST
    tests/ frame is not the test item whenever the test fits through a
    module-level helper — `test_structure_certificate_1058` fits inside `_fit`,
    so keying on the innermost frame attributes all four of its tests to `_fit`
    and none of them to a name the JUnit uses. The test item is the OUTERMOST
    tests/ frame whose function is named `test_*`, which is what pytest called.
    """
    stuck = collections.defaultdict(collections.Counter)
    dumps = 0
    block: list[str] | None = None

    def flush(lines):
        text = "\n".join(lines)
        tests = _TEST_FRAME.findall(text)
        if not tests:
            return
        items = [func for _, func in tests if func.startswith("test_")]
        if not items:
            return
        package = _PKG_FRAME.findall(text)
        frame = "gamfit/%s:%s in %s" % package[0] if package else "(no gamfit frame)"
        stuck[items[-1]][frame] += 1

    with open(path, errors="replace") as handle:
        for raw in handle:
            line = _LOG_PREFIX.sub("", raw.rstrip("\n"))
            header = _DUMP_HEADER.search(line)
            if header:
                if block is not None:
                    flush(block)
                # Keep whatever followed the header on the same line: with two
                # writers on one fd, the first frame can share the header's line.
                block = [line[header.end():]]
                dumps += 1
                continue
            if block is not None:
                block.append(line)
                # A dump ends at the outermost frame of the worker's main thread.
                if _DUMP_TAIL in line:
                    flush(block)
                    block = None
    if block is not None:
        flush(block)
    return stuck, dumps


def main(argv):
    junit_path, log_path, out_path = argv[1], argv[2], argv[3]
    counts, censored, diagnosed = read_junit(junit_path)
    stuck, dumps = read_dumps(log_path)

    out = []
    w = out.append
    total = sum(counts.values())
    w("# python-contracts full-suite triage (#1512) — recovered summary")
    w("")
    w("collected            %d" % total)
    w("passed               %d" % counts["passed"])
    w("skipped              %d" % counts["skipped"])
    w("failed (diagnosed)   %d" % counts["failed"])
    w("errored (diagnosed)  %d" % counts["errored"])
    w("CENSORED             %d   (worker killed at the per-test bound;" % counts["censored"])
    w("                          the JUnit records no reason and time=0.000)")
    w("")
    w("faulthandler dumps in the step log: %d" % dumps)
    w("")

    if not censored:
        w("No censored tests in this run.")
    else:
        w("## The %d censored tests, with the frame each was stuck in" % len(censored))
        w("")
        w("Recovered from the faulthandler dumps, which fire before the kill and")
        w("write straight to fd 2. A test listed here is NOT a crashed interpreter:")
        w("it is a call that had not returned when the bound expired.")
        w("")
        by_file = collections.defaultdict(list)
        for where, name in censored:
            by_file[where].append(name)
        for where in sorted(by_file, key=lambda k: (-len(by_file[k]), k)):
            w("[%d] %s" % (len(by_file[where]), where))
            for name in sorted(by_file[where]):
                # Parametrised items carry a [param] suffix the dump cannot see.
                base = name.split("[", 1)[0]
                frames = stuck.get(base)
                if frames:
                    frame = frames.most_common(1)[0][0]
                else:
                    frame = "(no dump recovered — dump may have interleaved with a concurrent worker)"
                w("      %s" % name)
                w("          stuck in %s" % frame)
            w("")

        w("## Entry-point census over the censored population")
        w("")
        entries = collections.Counter()
        for where, name in censored:
            frames = stuck.get(name.split("[", 1)[0])
            entries[frames.most_common(1)[0][0] if frames else "(no dump recovered)"] += 1
        for frame, count in entries.most_common():
            w("  %4d  %s" % (count, frame))
        w("")

    slowest = sorted(diagnosed, key=lambda r: -r[2])[:20]
    if slowest:
        w("## 20 slowest tests that DID report a duration")
        w("")
        for where, name, seconds, _ in slowest:
            w("  %9.1fs  %s::%s" % (seconds, where, name))
        w("")

    text = "\n".join(out) + "\n"
    with open(out_path, "w") as handle:
        handle.write(text)
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
