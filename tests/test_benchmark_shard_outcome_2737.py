"""#2737 — a benchmark shard that was CUT SHORT measured nothing.

Run 30687540475 reported 42 red Bench jobs on a 110-scenario matrix. The headline
that came out of it was "the scenarios exceed the 42m per-scenario wall budget —
a budget/matrix mismatch". Only half of that was true. Of the 42:

  * 22 ran at least as long as the budget then in force: budget-CENSORED. They
    were killed before they could produce a verdict, so they measured NOTHING.
  * 20 FAILED WELL INSIDE their budget — several `matern` scenarios died at
    220-300s against a 2520s budget. Those are real defects, and there was no
    machine-readable way to see them: the workflow knew the difference (GNU
    `timeout` reports 124/137 for a killed command and anything else for one
    that ran to its own conclusion) and then collapsed both onto `exit 1`.

Two rules are pinned here.

1. The marker is a function of the EXIT CODE, so no two shards can disagree about
   the same input, and 124/137 can never be reported as a measurement.
2. The aggregate counts a censored scenario as NOT MEASURED — never as a pass,
   never as a fail. This is the rule 5d40cf900 landed for #2743: a total is a
   total only when every contributor was measured, and silent truncation reads
   as coverage.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
WORKFLOW_TASKS = REPO_ROOT / ".github" / "scripts" / "workflow_tasks.py"


def _load_workflow_tasks():
    spec = importlib.util.spec_from_file_location("workflow_tasks", WORKFLOW_TASKS)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _configured_scenario():
    scenarios = json.loads((REPO_ROOT / "bench" / "scenarios.json").read_text())
    return scenarios["scenarios"][0]["name"]


# GNU `timeout` exits 124 when the command dies on the TERM it sent, and 137
# (128+SIGKILL) when the command outlived the `--kill-after` grace period. Both
# mean "cut short at the budget"; every other non-zero code means the shard
# reached its own conclusion inside the budget.
@pytest.mark.parametrize("exit_code", [124, 137])
def test_timeout_exit_codes_are_recorded_as_censored(tmp_path, exit_code):
    mod = _load_workflow_tasks()
    out = tmp_path / "marker.json"
    marker = mod.record_shard_outcome("scenario_x", exit_code, 2520, str(out))
    assert marker["outcome"] == mod.SHARD_OUTCOME_CENSORED
    assert marker["measured"] is False
    assert json.loads(out.read_text()) == marker
    assert marker["schema"] == mod.SHARD_OUTCOME_SCHEMA


@pytest.mark.parametrize("exit_code", [1, 2, 101, 139])
def test_other_nonzero_exit_codes_are_recorded_as_errored(tmp_path, exit_code):
    mod = _load_workflow_tasks()
    marker = mod.record_shard_outcome(
        "scenario_x", exit_code, 2520, str(tmp_path / "marker.json")
    )
    assert marker["outcome"] == mod.SHARD_OUTCOME_ERRORED
    assert marker["measured"] is False


def test_exit_zero_is_recorded_as_measured(tmp_path):
    mod = _load_workflow_tasks()
    marker = mod.record_shard_outcome(
        "scenario_x", 0, 2520, str(tmp_path / "marker.json")
    )
    assert marker["outcome"] == mod.SHARD_OUTCOME_MEASURED
    assert marker["measured"] is True


def test_a_censored_scenario_is_not_measured_and_is_never_a_failure():
    """The censored population must never be scored as a verdict."""

    mod = _load_workflow_tasks()
    scenario = _configured_scenario()
    verdict = mod.matched_benchmark_verdict(
        [],
        shard_outcomes=[
            {
                "schema": mod.SHARD_OUTCOME_SCHEMA,
                "scenario_name": scenario,
                "outcome": mod.SHARD_OUTCOME_CENSORED,
            }
        ],
    )
    assert verdict["censored_scenarios"] == [scenario]
    assert scenario in verdict["not_measured_scenarios"]
    assert verdict["errored_scenarios"] == []
    # Not counted as measured coverage either: the denominator has to admit it.
    assert scenario not in [c["scenario_name"] for c in verdict["comparisons"]]

    outcome = mod.benchmark_verdict_enforcement(verdict, require_full_suite=False)
    report = "\n".join(outcome["summary_lines"])
    assert "NOT MEASURED" in report
    # A censored scenario is an absence, so on a targeted (non-full-suite) run it
    # must not be reported as that scenario having FAILED.
    assert not any(
        scenario in failure and "FAILED inside" in failure for failure in outcome["failures"]
    )
    # But the coverage gap is not silent: the full-suite contract is unmet.
    full = mod.benchmark_verdict_enforcement(verdict, require_full_suite=True)
    assert not full["passed"]
    assert any("NOT MEASURED" in failure for failure in full["failures"])


def test_a_scenario_that_failed_inside_its_budget_fails_the_gate_by_itself():
    """The 20 sub-budget failures must be their own population, and must be red.

    Before #2737 an errored shard was one more anonymous "missing scenario",
    reported only when the full-suite contract was being enforced — so a cluster
    of `matern` scenarios dying at 220-300s was readable as a budget symptom.
    """

    mod = _load_workflow_tasks()
    scenario = _configured_scenario()
    verdict = mod.matched_benchmark_verdict(
        [],
        shard_outcomes=[
            {
                "schema": mod.SHARD_OUTCOME_SCHEMA,
                "scenario_name": scenario,
                "outcome": mod.SHARD_OUTCOME_ERRORED,
            }
        ],
    )
    assert verdict["errored_scenarios"] == [scenario]
    assert verdict["censored_scenarios"] == []
    assert scenario not in verdict["not_measured_scenarios"]

    outcome = mod.benchmark_verdict_enforcement(verdict, require_full_suite=False)
    assert not outcome["passed"]
    assert any(
        "FAILED inside their own wall budget" in failure for failure in outcome["failures"]
    )


def test_censored_and_errored_are_reported_as_separate_populations():
    """One red is not two reds. The report must resolve the split."""

    mod = _load_workflow_tasks()
    names = [
        s["name"]
        for s in json.loads((REPO_ROOT / "bench" / "scenarios.json").read_text())[
            "scenarios"
        ][:2]
    ]
    censored, errored = names
    verdict = mod.matched_benchmark_verdict(
        [],
        shard_outcomes=[
            {"scenario_name": censored, "outcome": mod.SHARD_OUTCOME_CENSORED},
            {"scenario_name": errored, "outcome": mod.SHARD_OUTCOME_ERRORED},
        ],
    )
    assert verdict["censored_scenarios"] == [censored]
    assert verdict["errored_scenarios"] == [errored]
    report = "\n".join(mod._shard_outcome_summary_lines(verdict))
    assert censored in report and errored in report
    assert "censored" in report and "errored" in report


def test_a_scenario_with_no_marker_at_all_is_reported_as_unreported():
    """Absent-and-unexplained is a third state, and it is still not a failure."""

    mod = _load_workflow_tasks()
    verdict = mod.matched_benchmark_verdict([], shard_outcomes=[])
    assert verdict["shard_outcomes_seen"] == 0
    assert verdict["unreported_scenarios"] == verdict["missing_scenarios"]
    assert verdict["not_measured_scenarios"] == verdict["missing_scenarios"]
    assert verdict["errored_scenarios"] == []
