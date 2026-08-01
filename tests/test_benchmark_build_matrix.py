"""Regression test for #1560 — the benchmark suite ran zero scenarios.

`.github/workflows/benchmark.yml` fans the selected scenarios across two jobs
and reads four outputs from the `prepare` job's matrix step:

    parallel_matrix / parallel_count   (bench-shard,        max-parallel 8)
    serial_matrix   / serial_count     (bench-shard-serial, max-parallel 1)

`build_matrix()` previously emitted only a single `matrix=` output (plus
`is_nightly=`), leaving all four of those downstream outputs empty. The shard
jobs then expanded `fromJSON('')` to nothing and ran **zero** scenarios, so the
nightly Benchmark Suite went green while benchmarking nothing.

These assertions pin the output contract the workflow actually consumes.

#2737 extends that contract: each matrix entry now also carries the scenario's
own `wall_budget_seconds` (the GNU `timeout` the shard runs under) and
`timeout_minutes` (the GitHub job ceiling, which must sit strictly above it, or a
censored shard is CANCELLED instead of exiting 124 and the run can no longer tell
"cut short" from "failed"). Both come from `bench/ci_wall_budgets.json`,
replacing a hand-maintained shell `case` that was duplicated in both shard jobs.
"""

import importlib.util
import json
import pathlib

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
WORKFLOW_TASKS = REPO_ROOT / ".github" / "scripts" / "workflow_tasks.py"
SCENARIOS = REPO_ROOT / "bench" / "scenarios.json"

# Serial scenarios are kept in lockstep with build_matrix()'s SERIAL_SCENARIOS.
SERIAL_SCENARIOS = {"icu_survival_death", "cirrhosis_survival"}


def _load_workflow_tasks():
    spec = importlib.util.spec_from_file_location("workflow_tasks", WORKFLOW_TASKS)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _names(matrix):
    """Scenario names out of the `include`-shaped matrix build_matrix emits."""
    return {entry["scenario"] for entry in matrix["include"]}


def _configured_scenarios():
    return {
        s["name"] for s in json.loads(SCENARIOS.read_text())["scenarios"] if "name" in s
    }


def _run_build_matrix(monkeypatch, tmp_path, event_name):
    mod = _load_workflow_tasks()
    out = tmp_path / "github_output"
    out.write_text("")
    monkeypatch.setenv("GITHUB_OUTPUT", str(out))
    monkeypatch.setenv("GITHUB_EVENT_NAME", event_name)
    # Imported workflow helpers must resolve checked-in inputs from their own
    # location, not from whichever directory happens to invoke them.
    monkeypatch.chdir(tmp_path)
    requested = None if event_name == "schedule" else "wine_temp_vs_year"
    mod.build_matrix(requested)
    parsed = {}
    for line in out.read_text().splitlines():
        if not line or "=" not in line:
            continue
        key, _, value = line.partition("=")
        parsed[key] = value
    return parsed


@pytest.mark.parametrize("event_name", ["schedule", "workflow_dispatch"])
def test_build_matrix_emits_the_four_outputs_the_workflow_reads(
    monkeypatch, tmp_path, event_name
):
    outputs = _run_build_matrix(monkeypatch, tmp_path, event_name)

    # The exact four output names benchmark.yml's `prepare` job declares.
    for key in ("parallel_matrix", "parallel_count", "serial_matrix", "serial_count"):
        assert key in outputs, f"build_matrix() must emit `{key}` (consumed by benchmark.yml)"

    # Each matrix must be valid JSON of the shape `fromJSON(...)` expects.
    parallel = json.loads(outputs["parallel_matrix"])
    serial = json.loads(outputs["serial_matrix"])
    assert isinstance(parallel.get("include"), list)
    assert isinstance(serial.get("include"), list)

    # Counts must match the matrices the shard jobs expand, and the `!= '0'`
    # gate must actually open: at least the parallel shards must have work.
    assert outputs["parallel_count"] == str(len(parallel["include"]))
    assert outputs["serial_count"] == str(len(serial["include"]))
    assert int(outputs["parallel_count"]) > 0, "no parallel scenarios scheduled — suite would run nothing"

    # Serial scenarios route to the serial (max-parallel 1) job, never the
    # parallel one; everything else is parallel. No scenario is dropped.
    parallel_names = _names(parallel)
    serial_names = _names(serial)
    for name in SERIAL_SCENARIOS:
        assert name not in parallel_names
    assert not (parallel_names & serial_names)


def test_nightly_selects_every_scenario_split_across_both_jobs(monkeypatch, tmp_path):
    outputs = _run_build_matrix(monkeypatch, tmp_path, "schedule")
    all_names = _configured_scenarios()
    scheduled = _names(json.loads(outputs["parallel_matrix"])) | _names(
        json.loads(outputs["serial_matrix"])
    )
    # Nightly runs the whole suite — nothing silently dropped.
    assert scheduled == all_names
    # Serial scenarios that exist in the suite land in the serial bucket.
    expected_serial = all_names & SERIAL_SCENARIOS
    assert _names(json.loads(outputs["serial_matrix"])) == expected_serial


def test_every_scheduled_scenario_carries_its_own_budget_under_its_own_ceiling(
    monkeypatch, tmp_path
):
    """#2737: no scenario may run on a budget nobody derived for it.

    The failure this pins: a flat 42m `case` default applied to every cell of a
    110-scenario matrix, so 22 shards were cut short at the wall and reported as
    ordinary failures — which is what let 20 scenarios failing WELL INSIDE the
    budget hide behind a budget headline.
    """

    outputs = _run_build_matrix(monkeypatch, tmp_path, "schedule")
    entries = (
        json.loads(outputs["parallel_matrix"])["include"]
        + json.loads(outputs["serial_matrix"])["include"]
    )
    assert {entry["scenario"] for entry in entries} == _configured_scenarios()

    for entry in entries:
        budget_seconds = entry["wall_budget_seconds"]
        ceiling_minutes = entry["timeout_minutes"]
        assert isinstance(budget_seconds, int) and budget_seconds > 0
        assert isinstance(ceiling_minutes, int)
        # The GitHub ceiling must sit STRICTLY above the GNU-timeout budget. If
        # it does not, GitHub cancels the job first: a cancelled job carries no
        # exit code, so 124/137 never happens and censored becomes
        # indistinguishable from failed — the exact conflation #2737 is about.
        assert ceiling_minutes * 60 > budget_seconds, (
            f"{entry['scenario']}: job ceiling {ceiling_minutes}m does not exceed "
            f"its {budget_seconds}s wall budget"
        )


def test_budget_table_is_not_a_flat_number_and_is_keyed_on_real_scenarios(
    monkeypatch, tmp_path
):
    """The calibration must be per scenario, and must describe THIS suite.

    A single value repeated across the matrix is the stale-`case` failure mode
    wearing a JSON costume; entries naming scenarios that no longer exist are how
    a hand-maintained table drifts away from the suite it claims to budget.
    """

    mod = _load_workflow_tasks()
    monkeypatch.chdir(tmp_path)
    budgets = mod.scenario_wall_budgets()
    assert set(budgets) == _configured_scenarios()
    assert len({b["budget_seconds"] for b in budgets.values()}) > 1

    table = json.loads((REPO_ROOT / "bench" / "ci_wall_budgets.json").read_text())
    # Provenance, not folklore: the measurements must name the run they came from.
    assert table["provenance"]["run_id"]
    assert table["provenance"]["source"]
    measured = set(table["measurements"])
    assert measured, "no measured history to calibrate against"
    unknown = measured - _configured_scenarios()
    assert not unknown, f"budget table names scenarios that no longer exist: {sorted(unknown)}"
    for name, entry in table["measurements"].items():
        assert entry["outcome"] in {"completed", "censored", "errored"}, name
        assert isinstance(entry["job_seconds"], int) and entry["job_seconds"] > 0, name
