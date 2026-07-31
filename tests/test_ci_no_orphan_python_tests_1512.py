"""#1512 orphan guard: every Python test file must be reached by some CI step.

The bug behind #1512: ``pyproject.toml`` declares ``testpaths = ["tests"]`` so a
bare ``pytest`` collects that directory, but the workflows historically named
only a handful of files explicitly. Every other test file therefore ran in NO CI
job: silent orphans that asserted nothing in CI while looking like coverage.

Three distinct mechanisms produce an orphan, and this module gates all three.

1. NOT NAMED. No CI step targets the file or any directory above it. A
   directory-level ``pytest <dir>`` step covers everything beneath it at once,
   which is why one is required — but "a directory-level step exists somewhere"
   is NOT the same claim as "this file is under it", and the previous version of
   this guard conflated them: it returned early the moment it saw any
   ``pytest tests/``, which made its file-by-file check unreachable. Files
   living OUTSIDE ``testpaths`` (``bench/``, ``experiments/``) were invisible to
   it for that reason, and six of them — 29 test functions — ran nowhere.

2. NOT COLLECTIBLE BY NAME. ``pytest <dir>`` only collects files whose name
   matches the ``python_files`` ini setting. A file named anything else is
   skipped no matter how many ``def test_*`` it defines, and nothing reports it.

3. DESELECTED. A step that reaches the file may still filter it out: ``-m "not
   slow"`` drops every ``slow``-marked test, and a ``conftest.py`` that sets
   ``collect_ignore``/``collect_ignore_glob`` drops a whole directory (as
   ``tests/torch/conftest.py`` does when torch is absent). Both mechanisms are
   legitimate — they are not legitimate if NOTHING else runs that population.

THE DISCOVERY RULE ITSELF IS NOT REIMPLEMENTED HERE. The previous version
hard-coded ``name.startswith("test_") or name.endswith("_test.py")`` and then
globbed with ``tests/test_*.py`` — dropping the ``*_test.py`` half at the one
place it mattered. Two files (``tests/workflow_tasks_2623_test.py``, which held
every assertion about the #2623 benchmark-verdict contract, and
``tests/test_benchmark_build_matrix.py``) were orphaned behind exactly that
mismatch: collected by pytest, invisible to the guard whose purpose is to prove
every test file is named by some CI step. So the patterns now come from
``pytestconfig.getini("python_files")`` — pytest's own resolved configuration,
honouring ``pyproject.toml`` / ``pytest.ini`` / ``setup.cfg`` — and the standalone
fallback used by ``__main__`` is checked against it in
``test_fallback_ini_reader_agrees_with_pytest``. A guard that can disagree with
pytest about what a test file is has the same defect it exists to prevent.

It is pure-Python (no ``gamfit`` / ``gamfit._rust`` dependency) so it runs even
when the Rust extension is not built, and it lives in ``tests/`` so it is itself
covered by the very directory-level step it guards.
"""

from __future__ import annotations

import fnmatch
import json
import re
import shlex
import subprocess
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_WORKFLOW_DIR = _REPO_ROOT / ".github" / "workflows"
_TESTS_DIR = _REPO_ROOT / "tests"
_PYPROJECT = _REPO_ROOT / "pyproject.toml"

# pytest's documented default, used only when no config file sets the value.
_DEFAULT_PYTHON_FILES = ("test_*.py", "*_test.py")

# A ``def test...`` at module level, or a method of a ``Test...`` class (pytest's
# default ``python_functions = test*`` / ``python_classes = Test*``). Indented
# ``def test`` inside a plain helper function is not collectible either way, but
# counting it only ever makes the guard stricter about a file that clearly means
# to hold tests.
_TEST_FUNCTION = re.compile(r"^\s*def test\w*\s*\(", re.MULTILINE)

# Names pytest treats as infrastructure rather than as a test module.
_INFRASTRUCTURE = frozenset({"conftest.py", "__init__.py", "setup.py"})

# Directories that are never part of this repo's own test surface: vendored
# sources, build output, virtualenvs, and agent scratch checkouts (which contain
# verbatim COPIES of tests/ and would otherwise be reported as orphans).
_EXCLUDED_TOP_LEVEL = frozenset(
    {".cargo", ".codex-artifacts", ".git", "target", "dist", "node_modules"}
)
_EXCLUDED_PARTS = frozenset({"__pycache__", ".pytest_cache", ".venv", "node_modules"})

# Options that consume the following token, so that token is a value and never a
# collection target.
_VALUE_TAKING = frozenset(
    {
        "-m",
        "-k",
        "-n",
        "-p",
        "-o",
        "-c",
        "-W",
        "--markers",
        "--rootdir",
        "--tb",
        "--timeout",
        "--timeout-method",
        "--junitxml",
        "--junit-xml",
        "--max-worker-restart",
        "--deselect",
        "--ignore",
        "--ignore-glob",
        "--import-mode",
        "--dist",
    }
)

# Where the pytest command line ends and the surrounding shell begins.
_SHELL_OPERATORS = frozenset({"|", "||", "&&", ";", "&", "2>&1"})

# GitHub-expression / shell prefixes that resolve to the checkout root.
_WORKSPACE_PREFIXES = (
    "${{ github.workspace }}/",
    "${{github.workspace}}/",
    "$GITHUB_WORKSPACE/",
    "${GITHUB_WORKSPACE}/",
)

# Triggers that fire without a human asking. ``workflow_dispatch`` does not
# count: a step that only ever runs when somebody remembers to press the button
# is not meaningfully different from an orphan, and this repo has the receipt —
# `python-contracts.yml`'s own header records that every run of the #1512
# inventory between 2026-07-07 and the day a schedule was added was a
# hand-dispatched one.
_AUTOMATIC_TRIGGERS = frozenset({"push", "pull_request", "pull_request_target", "schedule"})


# ---------------------------------------------------------------------------
# pytest configuration (the authority on what a test file is)
# ---------------------------------------------------------------------------


def _fallback_ini(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    """Read a whitespace/newline-separated pytest ini value without pytest.

    Only used by ``__main__`` (and cross-checked against pytest itself in
    ``test_fallback_ini_reader_agrees_with_pytest``), so the two can never drift
    into disagreeing about the collection rule.
    """
    for path, section in (
        (_REPO_ROOT / "pytest.ini", "[pytest]"),
        (_REPO_ROOT / "setup.cfg", "[tool:pytest]"),
        (_PYPROJECT, "[tool.pytest.ini_options]"),
    ):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        start = text.find(section)
        if start < 0:
            continue
        body = text[start + len(section) :]
        end = re.search(r"^\[", body, re.MULTILINE)
        if end:
            body = body[: end.start()]
        match = re.search(rf"^\s*{re.escape(name)}\s*=\s*(.+)$", body, re.MULTILINE)
        if not match:
            continue
        raw = match.group(1).strip()
        if raw.startswith("["):  # TOML array, possibly multi-line
            tail = body[match.end(1) - len(raw) :]
            raw = tail[: tail.index("]") + 1]
            return tuple(v.strip().strip("\"'") for v in raw.strip("[]").split(",") if v.strip())
        return tuple(v.strip().strip("\"'") for v in raw.split())
    return default


def _python_file_patterns(config: object | None = None) -> tuple[str, ...]:
    """The ``python_files`` patterns, from pytest when pytest is running."""
    if config is not None:
        return tuple(config.getini("python_files"))  # type: ignore[attr-defined]
    return _fallback_ini("python_files", _DEFAULT_PYTHON_FILES)


def _is_collectible_name(name: str, patterns: tuple[str, ...]) -> bool:
    """Whether pytest would consider ``name`` a test module.

    ``python_files`` patterns are matched against the file's basename, which is
    what pytest's own ``path_matches_patterns`` does for slash-free patterns. A
    pattern containing a separator would mean something else, so refuse to guess.
    """
    assert all("/" not in p for p in patterns), (
        f"python_files contains a path pattern ({patterns!r}); this guard matches "
        "on the basename like pytest does for slash-free patterns and cannot "
        "honour a path pattern. Teach _is_collectible_name the new rule."
    )
    return any(fnmatch.fnmatch(name, p) for p in patterns)


# ---------------------------------------------------------------------------
# repository inventory
# ---------------------------------------------------------------------------


def _tracked_python_files() -> list[Path]:
    """Every tracked ``*.py`` in the repo, as paths relative to the root.

    ``git ls-files`` rather than ``rglob`` so untracked scratch output and
    ignored trees are excluded by the same rule the repo already uses. Falls
    back to a filtered walk where git is unavailable.
    """
    try:
        out = subprocess.run(
            ["git", "-C", str(_REPO_ROOT), "ls-files", "-z", "--", "*.py"],
            capture_output=True,
            check=True,
            text=True,
            timeout=120,
        ).stdout
        paths = [Path(p) for p in out.split("\0") if p]
    except (OSError, subprocess.SubprocessError):
        paths = [p.relative_to(_REPO_ROOT) for p in _REPO_ROOT.rglob("*.py")]
        paths = [p for p in paths if p.parts and p.parts[0] not in _EXCLUDED_TOP_LEVEL]
    return sorted(
        p
        for p in paths
        if not _EXCLUDED_PARTS.intersection(p.parts)
        and (not p.parts or p.parts[0] not in _EXCLUDED_TOP_LEVEL)
    )


def _collectible_test_files(patterns: tuple[str, ...]) -> list[Path]:
    return [p for p in _tracked_python_files() if _is_collectible_name(p.name, patterns)]


# ---------------------------------------------------------------------------
# CI inventory
# ---------------------------------------------------------------------------


class Invocation:
    """One pytest command line found in a workflow."""

    def __init__(self, workflow: str, automatic: bool, targets: list[str], markers: str | None):
        self.workflow = workflow
        self.automatic = automatic
        self.targets = targets
        self.markers = markers

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Invocation({self.workflow}, auto={self.automatic}, {self.targets}, -m {self.markers!r})"


def _normalise_target(token: str) -> str | None:
    """Turn a command-line token into a repo-relative path, or None."""
    for prefix in _WORKSPACE_PREFIXES:
        if token.startswith(prefix):
            token = token[len(prefix) :]
            break
    if "$" in token or "{{" in token:
        return None  # opaque shell/GitHub expansion: cannot be attributed
    token = token.rstrip("/")
    if not token or token.startswith("-"):
        return None
    return token


def _parse_command(line: str, workflow: str, automatic: bool) -> Invocation | None:
    try:
        tokens = shlex.split(line)
    except ValueError:
        tokens = line.split()
    if "pytest" not in tokens:
        return None
    tokens = tokens[tokens.index("pytest") + 1 :]
    # The command ends where the shell takes over: `pytest ... 2>&1 | tee log`
    # would otherwise contribute `2>&1`, `|` and `tee` as collection targets.
    for stop, token in enumerate(tokens):
        if token in _SHELL_OPERATORS or token.startswith(">") or token.endswith(">&1"):
            tokens = tokens[:stop]
            break
    targets: list[str] = []
    markers: str | None = None
    skip_next = False
    for index, token in enumerate(tokens):
        if skip_next:
            skip_next = False
            continue
        if token.startswith("-"):
            flag, _, inline = token.partition("=")
            if flag == "-m":
                markers = inline if inline else (tokens[index + 1] if index + 1 < len(tokens) else None)
            if not inline and flag in _VALUE_TAKING:
                skip_next = True
            continue
        target = _normalise_target(token)
        if target is not None:
            targets.append(target)
    return Invocation(workflow, automatic, targets, markers)


def _workflow_is_automatic(text: str) -> bool:
    """Whether the workflow fires without a human pressing a button."""
    inline = re.search(r"^on:[ \t]+(\S.*)$", text, re.MULTILINE)
    if inline:
        return any(t in inline.group(1) for t in _AUTOMATIC_TRIGGERS)
    block = re.search(r"^on:\s*$(.*?)^\S", text + "\n\x00", re.MULTILINE | re.DOTALL)
    assert block, "workflow has no parsable `on:` trigger block"
    return any(re.search(rf"^\s+{t}:", block.group(1), re.MULTILINE) for t in _AUTOMATIC_TRIGGERS)


def _manifest_invocations(text: str, workflow: str, automatic: bool) -> list[Invocation]:
    """Selectors declared in a ``COVERAGE_POPULATIONS_JSON`` manifest.

    ``missing-populations.yml`` does not write literal pytest command lines: it
    validates a checked-in JSON manifest and drives the runners from its
    ``selector`` arrays. Those selectors are real invocations and are what make
    the ``slow`` and ``tests/torch`` populations non-orphaned, so read them as
    such instead of pattern-matching the shell that expands them.
    """
    marker = "COVERAGE_POPULATIONS_JSON:"
    start = text.find(marker)
    if start < 0:
        return []
    lines = text[start:].splitlines()[1:]
    indent = len(lines[0]) - len(lines[0].lstrip()) if lines else 0
    body: list[str] = []
    for line in lines:
        if line.strip() and (len(line) - len(line.lstrip())) < indent:
            break
        body.append(line.strip())
    try:
        populations = json.loads(" ".join(body))
    except json.JSONDecodeError:
        return []
    out: list[Invocation] = []
    for population in populations:
        if population.get("runner") != "pytest":
            continue
        selector = population.get("selector") or []
        inv = _parse_command("pytest " + " ".join(shlex.quote(s) for s in selector), workflow, automatic)
        if inv is not None:
            out.append(inv)
    return out


def _ci_invocations() -> list[Invocation]:
    assert _WORKFLOW_DIR.is_dir(), f"no workflow directory at {_WORKFLOW_DIR}"
    invocations: list[Invocation] = []
    for workflow in sorted(_WORKFLOW_DIR.glob("*.yml")) + sorted(_WORKFLOW_DIR.glob("*.yaml")):
        text = workflow.read_text(encoding="utf-8")
        automatic = _workflow_is_automatic(text)
        # Fold shell line continuations so a wrapped command is one logical line.
        folded = text.replace("\\\n", " ")
        for raw in folded.splitlines():
            stripped = raw.strip()
            if stripped.startswith("#") or "pytest" not in stripped:
                continue
            inv = _parse_command(stripped, workflow.name, automatic)
            if inv is not None and inv.targets:
                invocations.append(inv)
        invocations.extend(_manifest_invocations(text, workflow.name, automatic))
    return invocations


def _reaching(path: Path, invocations: list[Invocation]) -> list[Invocation]:
    """Invocations whose target set contains ``path`` (file or ancestor dir)."""
    candidates = {path.as_posix()} | {parent.as_posix() for parent in path.parents}
    return [inv for inv in invocations if candidates.intersection(inv.targets)]


# ---------------------------------------------------------------------------
# the guard
# ---------------------------------------------------------------------------


def test_fallback_ini_reader_agrees_with_pytest(pytestconfig) -> None:  # type: ignore[no-untyped-def]
    """The standalone reader must resolve ``python_files`` exactly as pytest does.

    Under pytest, discovery uses ``pytestconfig`` and cannot be wrong. The
    ``__main__`` path has no pytest to ask, so it parses the ini itself — and a
    hand-rolled parser that silently disagrees is the #1512 defect class in
    miniature (the previous guard's hard-coded defaults dropped ``*_test.py``).
    Pin them together here so the fallback can never drift.
    """
    assert _fallback_ini("python_files", _DEFAULT_PYTHON_FILES) == tuple(
        pytestconfig.getini("python_files")
    ), (
        "the standalone python_files reader disagrees with pytest: "
        f"{_fallback_ini('python_files', _DEFAULT_PYTHON_FILES)!r} != "
        f"{tuple(pytestconfig.getini('python_files'))!r}"
    )


def test_workflow_has_a_directory_level_pytest_step() -> None:
    """Some automatically-triggered workflow must run a whole test DIRECTORY.

    A catch-all is what makes a newly added test file covered on the day it
    lands rather than on the day somebody remembers to name it.
    """
    directory_steps = [
        inv
        for inv in _ci_invocations()
        if inv.automatic and any(not t.endswith(".py") for t in inv.targets)
    ]
    assert directory_steps, (
        "No automatically-triggered workflow runs a directory-level pytest step. "
        "Without one, every test file that is not named explicitly becomes a "
        "silent CI orphan (#1512). Add a step that runs a whole test directory, "
        "e.g. `python -m pytest tests/ -m 'not slow'`, in a workflow with a "
        "push/pull_request/schedule trigger."
    )


def test_every_collectible_test_file_is_reached_by_some_ci_step(pytestconfig) -> None:  # type: ignore[no-untyped-def]
    """Every tracked file pytest would collect must be reached by a CI step.

    "Reached" means some workflow targets the file or a directory above it. The
    check runs unconditionally: the old version short-circuited on the mere
    presence of a ``pytest tests/`` step, which asserted nothing about files
    outside ``tests/`` and let ``bench/test_*.py`` run nowhere for months.
    """
    patterns = _python_file_patterns(pytestconfig)
    invocations = _ci_invocations()
    files = _collectible_test_files(patterns)
    assert files, "no collectible test files found — discovery is broken"

    orphans = [p for p in files if not _reaching(p, invocations)]
    assert not orphans, (
        f"{len(orphans)} test file(s) matching python_files={list(patterns)} are "
        "reached by NO CI step, so they run in NO CI job (silent orphans, "
        "#1512):\n  "
        + "\n  ".join(p.as_posix() for p in orphans)
        + "\nWire each into the job where it belongs (a directory-level pytest "
        "step over its directory is the durable fix), or delete it if it is dead."
    )


def test_no_test_file_is_reached_only_by_manual_dispatch(pytestconfig) -> None:  # type: ignore[no-untyped-def]
    """A file covered only by a ``workflow_dispatch``-only workflow is an orphan.

    ``test.yml`` (Rust CI) has no automatic trigger, so every pytest step in it
    runs only when a human asks. A guard that counts those steps as coverage
    reports green over a surface nothing measures.
    """
    patterns = _python_file_patterns(pytestconfig)
    invocations = _ci_invocations()
    manual_only = []
    for path in _collectible_test_files(patterns):
        reaching = _reaching(path, invocations)
        if reaching and not any(inv.automatic for inv in reaching):
            workflows = sorted({inv.workflow for inv in reaching})
            manual_only.append(f"{path.as_posix()} (only: {', '.join(workflows)})")
    assert not manual_only, (
        f"{len(manual_only)} test file(s) are reached ONLY by workflows with no "
        "automatic trigger, so they run only when a human dispatches them "
        "(#1512):\n  " + "\n  ".join(manual_only)
    )


def test_every_deselected_marker_has_a_dedicated_ci_selector() -> None:
    """A marker every catch-all excludes must be run positively somewhere.

    ``-m "not slow"`` is a legitimate scheduling choice and an illegitimate way
    to make a population disappear. #2642 found three such populations at once
    (``slow``, ``tests/torch``, Rust doctests); this makes a fourth impossible
    to add silently, and keeps the first one visible for as long as it is open.
    """
    excluded: set[str] = set()
    selected: set[str] = set()
    for inv in _ci_invocations():
        # Only automatic lanes count, on both sides: an exclusion in a
        # dispatch-only workflow costs nothing, and a positive selector in one
        # is not coverage either.
        if not inv.markers or not inv.automatic:
            continue
        # Tokenise the marker expression and split names by whether the token
        # immediately before them negates: `not slow` excludes, bare `slow`
        # selects. Anything more exotic than that is not in use here and would
        # show up as an uncovered marker rather than as a silent pass.
        tokens = re.findall(r"\w+|\(|\)", inv.markers)
        for index, token in enumerate(tokens):
            if token in {"not", "and", "or"} or not token.isidentifier():
                continue
            if index and tokens[index - 1] == "not":
                excluded.add(token)
            else:
                selected.add(token)
    uncovered = sorted(excluded - selected)
    assert not uncovered, (
        f"{len(uncovered)} pytest marker(s) are deselected by CI and selected by "
        "no CI step, so every test carrying them runs in NO job (#1512/#2642):\n  "
        + "\n  ".join(uncovered)
        + "\nAdd a lane with an automatic trigger that runs `-m <marker>` "
        "positively. A marker is a scheduling decision; it is not a licence to "
        "stop measuring the tests that carry it."
    )


def test_every_conftest_ignored_directory_has_a_dedicated_ci_selector() -> None:
    """A directory a conftest can drop from collection needs its own lane.

    ``tests/torch/conftest.py`` removes the whole directory when torch is not
    installed. That is correct behaviour for a job without torch and a total
    blackout if no job ever has torch — which is what it was until #2642. The
    directory must therefore be an EXPLICIT target of an AUTOMATIC workflow, not
    merely swept up by an ancestor whose collection the conftest is free to
    cancel, and not merely named by a workflow somebody has to dispatch by hand.
    """
    targets = {t for inv in _ci_invocations() if inv.automatic for t in inv.targets}
    unlanded = []
    for conftest in _tracked_python_files():
        if conftest.name != "conftest.py":
            continue
        if "collect_ignore" not in (_REPO_ROOT / conftest).read_text(encoding="utf-8"):
            continue
        rel = conftest.parent.as_posix()
        if rel not in targets:
            unlanded.append(rel)
    assert not unlanded, (
        f"{len(unlanded)} directory(ies) whose conftest.py can cancel collection "
        "are named by no CI step of their own, so the conftest's skip makes them "
        "vanish from every verdict (#1512/#2642):\n  "
        + "\n  ".join(unlanded)
        + "\nGive each an explicit selector in a workflow that satisfies the "
        "condition the conftest tests for."
    )


def test_no_test_file_is_uncollectible_by_name(pytestconfig) -> None:  # type: ignore[no-untyped-def]
    """Every file under ``tests/`` that defines tests must be named collectibly.

    A directory-level ``pytest tests/`` step still skips a file whose name
    matches no ``python_files`` pattern, so such a file's tests never run and
    nothing says so — #1512 arriving by a different route than a missing step.
    """
    patterns = _python_file_patterns(pytestconfig)
    orphans: list[str] = []
    for path in sorted(_TESTS_DIR.rglob("*.py")):
        if _EXCLUDED_PARTS.intersection(path.parts):
            continue
        if path.name in _INFRASTRUCTURE or _is_collectible_name(path.name, patterns):
            continue
        found = _TEST_FUNCTION.findall(path.read_text(encoding="utf-8"))
        if found:
            rel = path.relative_to(_REPO_ROOT)
            orphans.append(f"{rel} ({len(found)} test function(s))")

    assert not orphans, (
        f"{len(orphans)} file(s) under tests/ define test functions but are "
        "named so that pytest never collects them, so those tests run in NO CI "
        "job (silent orphans, #1512):\n  "
        + "\n  ".join(orphans)
        + f"\nRename each to match one of {list(patterns)} so the directory-level "
        "step picks it up — or delete it if it is a scratch script rather than a "
        "test. A helper module that legitimately holds no tests should not define "
        "`def test...`."
    )


if __name__ == "__main__":
    # #1512: this guard is pure-Python and must run even where ``pytest`` is not
    # installed (the triage / Rust-only environments). Running it as a plain
    # script exercises the same assertions and exits non-zero on failure. The
    # ``pytestconfig``-taking checks fall back to `_fallback_ini`, which
    # `test_fallback_ini_reader_agrees_with_pytest` pins to pytest's own answer.
    _CHECKS = (
        test_workflow_has_a_directory_level_pytest_step,
        test_every_collectible_test_file_is_reached_by_some_ci_step,
        test_no_test_file_is_reached_only_by_manual_dispatch,
        test_every_deselected_marker_has_a_dedicated_ci_selector,
        test_every_conftest_ignored_directory_has_a_dedicated_ci_selector,
        test_no_test_file_is_uncollectible_by_name,
    )
    _failures = 0
    for _check in _CHECKS:
        try:
            if _check.__code__.co_argcount:
                _check(None)  # type: ignore[arg-type]
            else:
                _check()
        except AssertionError as exc:  # pragma: no cover - exercised via __main__
            _failures += 1
            print(f"FAIL {_check.__name__}:\n{exc}")
        else:
            print(f"PASS {_check.__name__}")
    if _failures:
        raise SystemExit(f"{_failures} guard check(s) failed (#1512)")
    print("OK: all #1512 orphan-guard checks passed")
