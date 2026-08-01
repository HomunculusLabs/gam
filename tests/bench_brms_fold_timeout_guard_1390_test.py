"""Regression guard for #1390 — brms MCMC CV per-fold timeout wiring.

Issue #1390: on the `us48_demand_31day` benchmark shard the brms reference
contender kept running full Bayesian MCMC cross-validation after the GAM lane
had finished, overrunning the 42-minute shard budget. The GNU `timeout`
SIGKILLed the whole shard (exit 124) AFTER gam was done, discarding every
result with no per-fold attribution.

The fix (commit d74ef8af) bounds each brms CV fold with a per-invocation
timeout so a slow/hung fold becomes a recorded, visible failure instead of
consuming the shard, and pairs it with a scenario-aware shard budget for the
heavy daily-demand panel.

This is a pure-source contract test (no Rust build, no subprocess). It pins the
three load-bearing pieces of that fix so a future refactor cannot silently drop
them and re-open the bulk-shard-kill:

  1. `run_cmd` accepts a `timeout_sec` override and actually enforces it
     (waits with that deadline, then terminates/kills the child).
  2. the brms CV driver reads `BENCH_BRMS_FOLD_TIMEOUT_SEC` and passes the
     resulting per-fold cap into `run_cmd(..., timeout_sec=...)`.
  3. the benchmark CI gives `us48_demand_31day` a larger shard budget than a
     light scenario, and the brms-fold cap is referenced where that shard budget
     is applied.

#2737 moved (3) without weakening it. The heavy scenario's budget no longer
lives in a hand-maintained shell `case` in `.github/workflows/benchmark.yml`; it
is derived per scenario from measured history in `bench/ci_wall_budgets.json` and
carried on the matrix entry, and the shard body itself now lives once in
`.github/actions/run-bench-shard/action.yml` instead of twice in the workflow.
The assertions below therefore check the PROPERTY (this scenario gets more wall
time than a light one; the outer budget still names the inner per-fold cap)
rather than the string that used to express it.
"""

import importlib.util
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def test_run_cmd_accepts_and_enforces_per_invocation_timeout():
    src = _read("bench/_run_suite_datasets.py")
    # The override parameter must exist on run_cmd.
    assert "def run_cmd(" in src, "run_cmd helper missing"
    assert "timeout_sec" in src, "run_cmd lost its timeout_sec override (#1390)"
    # It must actually be enforced: a bounded wait that escalates to kill on
    # overrun (not merely accepted and ignored).
    assert "effective_timeout" in src
    assert "proc.wait(timeout=effective_timeout)" in src, (
        "run_cmd no longer waits on the per-invocation timeout (#1390)"
    )
    assert "except subprocess.TimeoutExpired" in src, (
        "run_cmd no longer catches the per-invocation timeout (#1390)"
    )
    assert "proc.kill()" in src, "run_cmd no longer kills an overrunning child (#1390)"
    # rc=124 is the conventional timeout exit code the shard log greps for.
    assert "rc=124" in src, "run_cmd lost the timeout exit-code attribution (#1390)"


def test_brms_cv_driver_caps_each_fold():
    src = _read("bench/_run_suite_external.py")
    assert "def run_external_r_brms_cv(" in src, "brms CV driver missing"
    # The env override the fix introduced, with a finite default.
    assert "BENCH_BRMS_FOLD_TIMEOUT_SEC" in src, (
        "brms per-fold timeout env override removed (#1390)"
    )
    assert "brms_fold_timeout" in src
    # The cap must be threaded into the actual run_cmd call for the fold, not
    # just computed and dropped.
    assert "timeout_sec=brms_fold_timeout" in src, (
        "brms fold timeout no longer reaches run_cmd (#1390)"
    )
    # A capped fold must be tagged as a timeout (rc=124), distinct from a model
    # failure, so the recorded outcome is attributable.
    assert '"status": "timeout"' in src or "'status': 'timeout'" in src, (
        "brms timeout is no longer tagged distinctly from a model failure (#1390)"
    )
    assert "code == 124" in src, (
        "brms timeout no longer keyed off the rc=124 budget-overrun signal (#1390)"
    )


def test_brms_r_script_caps_mcmc_sampling_budget():
    src = _read("bench/_run_suite_external.py")
    # The brms MCMC sampling budget must be overridable per #1390 so a heavy
    # scenario can run a lighter posterior instead of being killed mid-sample.
    for knob in ("BENCH_BRMS_CHAINS", "BENCH_BRMS_ITER", "BENCH_BRMS_WARMUP"):
        assert knob in src, f"brms sampling knob {knob} removed (#1390)"
    # The R fit must consume the overridable values, not the old hardcoded ones.
    assert "chains = brms_chains" in src, "brms chains no longer overridable (#1390)"
    assert "iter = brms_iter" in src, "brms iter no longer overridable (#1390)"
    assert "warmup = brms_warmup" in src, "brms warmup no longer overridable (#1390)"


def test_benchmark_ci_gives_heavy_shard_a_larger_budget_than_a_light_one():
    spec = importlib.util.spec_from_file_location(
        "workflow_tasks", ROOT / ".github" / "scripts" / "workflow_tasks.py"
    )
    tasks = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tasks)
    budgets = tasks.scenario_wall_budgets()

    heavy = "us48_demand_31day"
    assert heavy in budgets, "heavy demand shard scenario missing from the budget table"
    # The scenario that triggered #1390 must still get more wall time than the
    # cheapest scenario in the suite -- that is the whole content of "explicit
    # (larger) budget", and it survives the budget moving out of the workflow.
    lightest = min(b["budget_seconds"] for b in budgets.values())
    assert budgets[heavy]["budget_seconds"] > lightest, (
        f"{heavy} no longer gets a larger shard budget than the lightest scenario (#1390)"
    )
    # And the per-shard budget knob must still reach the GNU timeout.
    action = _read(".github/actions/run-bench-shard/action.yml")
    assert "BENCH_SHARD_TIMEOUT_SEC" in action, (
        "per-shard budget knob removed from the shard runner (#1390)"
    )
    assert 'timeout --signal=TERM --kill-after=30s "${BENCH_SHARD_TIMEOUT_SEC}s"' in action, (
        "the per-shard budget no longer bounds the bench command (#1390)"
    )
    # The rationale where the OUTER budget is applied references the INNER
    # per-fold cap, so the two cannot drift apart unnoticed.
    assert "BENCH_BRMS_FOLD_TIMEOUT_SEC" in action, (
        "shard runner no longer references the brms per-fold cap (#1390)"
    )
