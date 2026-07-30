"""#1512 orphan guard: every ``tests/test_*.py`` must be reached by some CI step.

The bug behind #1512: ``pyproject.toml`` declares ``testpaths = ["tests"]`` so a
bare ``pytest`` collects the whole directory, but ``.github/workflows/test.yml``
historically named only a handful of files explicitly. Every other
``tests/test_*.py`` — including dozens of ``test_bug_hunt_*`` contract repros
committed by fix runs — therefore ran in NO CI job: they were silent orphans
that asserted nothing in CI.

This meta-test is the anti-regression. It parses the workflow and proves that a
*directory-level* pytest invocation (``pytest tests/`` / ``pytest tests``, i.e.
NOT a single named ``*.py`` file) exists. A directory-level step collects every
current and future ``test_*.py`` automatically, so as long as one such step is
present no new test file can become an orphan. If a future edit deletes that
catch-all step and reverts to naming individual files, this test fails and names
the ``test_*.py`` files that would no longer be collected by any CI step.

A directory-level step is necessary but NOT sufficient, which is the second hole
this guard closes. ``pytest tests/`` only collects files whose *name* matches
``python_files`` (defaults: ``test_*.py`` and ``*_test.py``). A file named
anything else — ``bug_hunt_foo.py``, ``response_geometry_bar.py`` — is skipped by
collection no matter how many ``def test_*`` functions it defines, and nothing
reports it: three such files sat in ``tests/`` for months, two of them the
anti-regression guards committed alongside the #1365 and #1366 fixes, and had
never executed once. Same #1512 defect, different mechanism, so
``test_no_test_file_is_uncollectible_by_name`` below enforces the naming half.

It is pure-Python (no ``gamfit`` / ``gamfit._rust`` dependency) so it runs even
when the Rust extension is not built, and it lives in ``tests/`` so it is itself
covered by the very directory-level step it guards.
"""

from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_WORKFLOW = _REPO_ROOT / ".github" / "workflows" / "test.yml"
_TESTS_DIR = _REPO_ROOT / "tests"
_PYPROJECT = _REPO_ROOT / "pyproject.toml"

# A pytest invocation whose target is the ``tests`` directory itself (optionally
# with a trailing slash), as opposed to a single ``tests/<file>.py``. The target
# must be followed by whitespace, a line continuation, end-of-line, or another
# flag — never a path segment like ``tests/test_foo.py``.
_DIR_LEVEL_PYTEST = re.compile(r"pytest\s+tests/?(?=\s|\\|$)")

# Any ``pytest`` invocation that names a specific ``tests/...py`` file.
_FILE_LEVEL_PYTEST = re.compile(r"pytest\s+(tests/[^\s\\]+\.py)")

# A ``def test...`` at module level, or a method of a ``Test...`` class (pytest's
# default ``python_functions = test*`` / ``python_classes = Test*``). Indented
# ``def test`` inside a plain helper function is not collectible either way, but
# counting it only ever makes the guard stricter about a file that clearly means
# to hold tests.
_TEST_FUNCTION = re.compile(r"^\s*def test\w*\s*\(", re.MULTILINE)

# ``python_files`` would change which names pytest collects and therefore
# invalidate the hard-coded defaults below.
_PYTHON_FILES_OVERRIDE = re.compile(r"^\s*python_files\s*=", re.MULTILINE)

# Directories that are never collected regardless of file name.
_UNCOLLECTED_DIRS = frozenset({"__pycache__", ".pytest_cache", ".venv", "node_modules"})

# Names pytest treats as infrastructure rather than as a test module.
_INFRASTRUCTURE = frozenset({"conftest.py", "__init__.py", "setup.py"})


def _is_collectible_name(name: str) -> bool:
    """Mirror pytest's default ``python_files = test_*.py *_test.py``."""
    return name.startswith("test_") or name.endswith("_test.py")


def _workflow_text() -> str:
    assert _WORKFLOW.is_file(), f"CI workflow not found at {_WORKFLOW}"
    return _WORKFLOW.read_text(encoding="utf-8")


def test_workflow_has_a_directory_level_pytest_step() -> None:
    """A catch-all ``pytest tests/`` step must exist so newly added
    ``test_*.py`` files are collected automatically (the #1512 fix)."""
    text = _workflow_text()
    assert _DIR_LEVEL_PYTEST.search(text), (
        "No directory-level `pytest tests/` invocation found in "
        f"{_WORKFLOW.relative_to(_REPO_ROOT)}. Without it, every tests/test_*.py "
        "that is not named explicitly becomes a silent CI orphan (#1512). Add a "
        "step that runs the whole tests/ directory (honoring testpaths), e.g. "
        "`python -m pytest tests/ -m 'not slow'`."
    )


def test_every_test_file_is_collected_by_some_ci_step() -> None:
    """Every ``tests/test_*.py`` on disk must be reachable by a CI step.

    A directory-level ``pytest tests/`` step covers all of them at once; only if
    that catch-all is absent do we fall back to checking explicit file names, and
    then report any file that no step would collect.
    """
    text = _workflow_text()
    test_files = sorted(p.name for p in _TESTS_DIR.glob("test_*.py"))
    assert test_files, "no tests/test_*.py files found — discovery is broken"

    if _DIR_LEVEL_PYTEST.search(text):
        # The directory-level step collects the entire suite; nothing can be an
        # orphan. This is the expected, healthy state.
        return

    named = {Path(m).name for m in _FILE_LEVEL_PYTEST.findall(text)}
    orphans = [f for f in test_files if f not in named]
    assert not orphans, (
        "No directory-level `pytest tests/` step exists, and these "
        f"{len(orphans)} tests/test_*.py file(s) are named by no CI step, so "
        "they run in NO CI job (silent orphans, #1512):\n  "
        + "\n  ".join(orphans)
        + "\nEither restore the directory-level catch-all step or name these "
        "files explicitly."
    )


def test_pyproject_does_not_override_python_files() -> None:
    """The naming guard below hard-codes pytest's default ``python_files``.

    If a future edit adds a ``python_files`` setting, the set of collectible
    names changes and ``_is_collectible_name`` silently stops matching reality —
    so fail here and point at the one function that has to be updated with it.
    """
    text = _PYPROJECT.read_text(encoding="utf-8")
    assert not _PYTHON_FILES_OVERRIDE.search(text), (
        "pyproject.toml now sets `python_files`, but "
        "`_is_collectible_name` in this file still hard-codes pytest's defaults "
        "(test_*.py, *_test.py). Update it to match, or the naming guard will "
        "report the wrong files."
    )


def test_no_test_file_is_uncollectible_by_name() -> None:
    """Every file under ``tests/`` that defines tests must be named collectibly.

    A directory-level ``pytest tests/`` step still skips a file whose name
    matches neither ``test_*.py`` nor ``*_test.py``, so such a file's tests
    never run and nothing says so — the #1512 defect arriving by a different
    route than a missing CI step.
    """
    orphans: list[str] = []
    for path in sorted(_TESTS_DIR.rglob("*.py")):
        if _UNCOLLECTED_DIRS.intersection(path.parts):
            continue
        if path.name in _INFRASTRUCTURE or _is_collectible_name(path.name):
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
        + "\nRename each to `test_<name>.py` (or `<name>_test.py`) so the "
        "directory-level step picks it up — or delete it if it is a scratch "
        "script rather than a test. A helper module that legitimately holds no "
        "tests should not define `def test...`."
    )


if __name__ == "__main__":
    # #1512: this guard is pure-Python and must run even where ``pytest`` is not
    # installed (the triage / Rust-only environments). Running it as a plain
    # script exercises the same assertions as ``pytest`` and exits non-zero on
    # the first failure, so it can gate locally without the test runner.
    _CHECKS = (
        test_workflow_has_a_directory_level_pytest_step,
        test_every_test_file_is_collected_by_some_ci_step,
        test_pyproject_does_not_override_python_files,
        test_no_test_file_is_uncollectible_by_name,
    )
    _failures = 0
    for _check in _CHECKS:
        try:
            _check()
        except AssertionError as exc:  # pragma: no cover - exercised via __main__
            _failures += 1
            print(f"FAIL {_check.__name__}:\n{exc}")
        else:
            print(f"PASS {_check.__name__}")
    if _failures:
        raise SystemExit(f"{_failures} guard check(s) failed (#1512)")
    print("OK: all #1512 orphan-guard checks passed")
