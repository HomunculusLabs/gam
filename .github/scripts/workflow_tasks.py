import argparse
import glob
import importlib.util
import json
import math
import os
import pathlib
import shutil
import subprocess
import sys
import zipfile


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_RUN_SUITE_PATH = _REPO_ROOT / "bench" / "run_suite.py"
_SCENARIOS_PATH = _REPO_ROOT / "bench" / "scenarios.json"
_WALL_BUDGETS_PATH = _REPO_ROOT / "bench" / "ci_wall_budgets.json"

# #2737: the marker every bench shard writes next to (or instead of) its result
# JSON, so the aggregate can tell a shard that was CUT SHORT from one that ran
# to a verdict. A censored shard measured nothing; counting it as a failure is
# as wrong as counting it as a pass.
SHARD_OUTCOME_SCHEMA = "benchmark-shard-outcome/v1"
SHARD_OUTCOME_MEASURED = "measured"
SHARD_OUTCOME_CENSORED = "censored"
SHARD_OUTCOME_ERRORED = "errored"


def _load_scenario_config():
    return json.loads(_SCENARIOS_PATH.read_text())


def _load_wall_budgets():
    return json.loads(_WALL_BUDGETS_PATH.read_text())


def scenario_wall_budgets():
    """Per-scenario CI wall budgets derived from each scenario's OWN history.

    #2737: the budget used to be a hand-maintained shell `case` that named one
    scenario at 145m and left every other cell of a 110-scenario matrix on a
    flat 42m. A flat number applied to a heterogeneous matrix goes stale the
    moment the matrix grows, and it did: 22 shards were cut short at 42m and
    reported as ordinary failures.

    The replacement is data, not a formula over scenario shape: each scenario is
    budgeted from its own measured `job_seconds` (see bench/ci_wall_budgets.json
    for the measurements, the provenance, and why a shape-based cost model was
    rejected). Scenarios with no completed measurement get the largest budget
    any measured scenario earns.

    Returns {scenario: {"budget_seconds", "timeout_minutes", "basis"}} for every
    CONFIGURED scenario, so a scenario added since the last calibration is
    covered by construction rather than by remembering to edit a table.
    """

    table = _load_wall_budgets()
    policy = table["policy"]
    safety_factor = float(policy["safety_factor"])
    # The GitHub job ceiling must clear the GNU-timeout budget by everything the
    # budget does NOT cover (checkout, Python/R setup, runtime download, smoke
    # imports, upload). Same headroom policy as the budget, applied to the
    # largest such gap ever measured -- see the rationale in the table.
    overhead_minutes = int(
        math.ceil(safety_factor * float(policy["job_overhead_seconds_observed_max"]) / 60.0)
    )
    measurements = table["measurements"]

    def budget_minutes_for(job_seconds):
        return int(math.ceil(safety_factor * float(job_seconds) / 60.0))

    completed = [
        entry["job_seconds"]
        for entry in measurements.values()
        if entry.get("outcome") == "completed"
    ]
    if not completed:
        raise SystemExit(
            f"{_WALL_BUDGETS_PATH} records no completed scenario, so no budget can be "
            "derived from measured history."
        )
    unmeasured_minutes = max(budget_minutes_for(seconds) for seconds in completed)

    budgets = {}
    for scenario in _load_scenario_config().get("scenarios", []):
        name = scenario.get("name")
        if not name:
            continue
        entry = measurements.get(name)
        if entry is not None and entry.get("outcome") == "completed":
            minutes = budget_minutes_for(entry["job_seconds"])
            basis = (
                f"{safety_factor}x its own measured {entry['job_seconds']}s completed run"
            )
        else:
            minutes = unmeasured_minutes
            observed = "no measurement on record" if entry is None else entry["outcome"]
            basis = (
                f"no completed measurement ({observed}); granted the largest budget any "
                "measured scenario earns"
            )
        budgets[name] = {
            "budget_seconds": minutes * 60,
            "timeout_minutes": minutes + overhead_minutes,
            "basis": basis,
        }
    return budgets


def validate_schemas():
    spec = importlib.util.spec_from_file_location("run_suite_mod", _RUN_SUITE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    cfg = _load_scenario_config()
    scenarios = cfg.get("scenarios", [])
    if not scenarios:
        raise SystemExit(f"No benchmark scenarios found in {_SCENARIOS_PATH}")

    for s in scenarios:
        mod.validate_scenario_schema(s)
    print(f"validated {len(scenarios)} scenario dataset schemas")


def validate_geo_subpop():
    spec = importlib.util.spec_from_file_location("run_suite_mod", _RUN_SUITE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    cfg = _load_scenario_config()
    scenarios = cfg.get("scenarios", [])
    geo_subpop = [s for s in scenarios if str(s.get("name", "")).startswith("geo_subpop16_")]
    if not geo_subpop:
        raise SystemExit(f"No geo_subpop16 scenarios found in {_SCENARIOS_PATH}")

    for s in geo_subpop:
        mod.validate_scenario_schema(s)
    print(f"validated geo_subpop16 simulation for {len(geo_subpop)} scenarios")


def build_matrix(requested_scenarios=None):
    SERIAL_SCENARIOS = {
        "icu_survival_death",
        "cirrhosis_survival",
    }

    cfg = _load_scenario_config()
    scenarios = cfg.get("scenarios", [])
    names = [s["name"] for s in scenarios if "name" in s]
    if not names:
        raise SystemExit(f"No benchmark scenarios found in {_SCENARIOS_PATH}")

    event_name = os.environ.get("GITHUB_EVENT_NAME", "").strip().lower()
    is_nightly = event_name == "schedule"
    requested = str(requested_scenarios or "").strip()
    selects_all = not requested or requested.lower() == "all" or requested == "*"

    if is_nightly:
        if not selects_all:
            raise SystemExit("Scheduled benchmark runs cannot select a scenario subset")
        selected = names
    elif selects_all:
        selected = names
    else:
        selected = list(
            dict.fromkeys(
                name.strip()
                for name in requested.split(",")
                if name.strip()
            )
        )
        unknown = sorted(set(selected) - set(names))
        if unknown:
            raise SystemExit(f"Unknown benchmark scenario(s): {', '.join(unknown)}")

    # benchmark.yml fans the selected scenarios across two jobs:
    #   `bench-shard`        (max-parallel 8, gated on `parallel_count != '0'`)
    #   `bench-shard-serial` (max-parallel 1, gated on `serial_count != '0'`)
    # and reads `prepare`'s `parallel_matrix` / `parallel_count` /
    # `serial_matrix` / `serial_count` outputs to do so. Emitting only a single
    # `matrix` output here (the historical shape) left all four of those
    # downstream outputs empty, so `fromJSON('')` expanded to nothing and BOTH
    # shard jobs ran zero scenarios — the suite went green while benchmarking
    # nothing (#1560). Split `selected` into the serial and parallel buckets and
    # emit exactly the four outputs the workflow consumes.
    #
    # #2737: each matrix entry now CARRIES its own wall budget and its own GitHub
    # job ceiling, replacing the shell `case` statement that was duplicated
    # verbatim in both shard jobs. Two copies of a hand-maintained budget list is
    # two places for it to go stale; this is one place, derived from measured
    # history (see `scenario_wall_budgets`).
    serial = [s for s in selected if s in SERIAL_SCENARIOS]
    parallel = [s for s in selected if s not in SERIAL_SCENARIOS]
    budgets = scenario_wall_budgets()

    def matrix_for(names):
        include = []
        for name in names:
            budget = budgets[name]
            include.append(
                {
                    "scenario": name,
                    "wall_budget_seconds": budget["budget_seconds"],
                    "timeout_minutes": budget["timeout_minutes"],
                }
            )
            print(
                f"budget {name}: {budget['budget_seconds']}s shard / "
                f"{budget['timeout_minutes']}m job ceiling -- {budget['basis']}"
            )
        return {"include": include}

    parallel_matrix = matrix_for(parallel)
    serial_matrix = matrix_for(serial)
    with open(os.environ["GITHUB_OUTPUT"], "a") as f:
        f.write(f"parallel_matrix={json.dumps(parallel_matrix)}\n")
        f.write(f"parallel_count={len(parallel)}\n")
        f.write(f"serial_matrix={json.dumps(serial_matrix)}\n")
        f.write(f"serial_count={len(serial)}\n")
        f.write(f"is_nightly={'true' if is_nightly else 'false'}\n")

def extract_maturin_wheel(out_dir_arg="gamfit"):
    wheels = sorted(glob.glob("dist/*.whl"))
    if not wheels:
        sys.exit("maturin produced no wheel under dist/")
    wheel = wheels[-1]
    out_dir = pathlib.Path(out_dir_arg)
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(wheel) as zf:
        matches = [
            m for m in zf.namelist()
            if m.startswith("gamfit/_rust") and (m.endswith(".so") or m.endswith(".pyd"))
        ]
        if not matches:
            sys.exit(f"no _rust*.so found inside {wheel}")
        for member in matches:
            target = out_dir / pathlib.Path(member).name
            with zf.open(member) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            print(f"extracted {target}")

def download_artifacts(target_name, out_dir_arg):
    repo = os.environ.get("GITHUB_REPOSITORY")
    run_id = os.environ.get("GITHUB_RUN_ID")
    out_dir = pathlib.Path(out_dir_arg)
    out_dir.mkdir(parents=True, exist_ok=True)

    proc = subprocess.run(
        ["gh", "api", f"/repos/{repo}/actions/runs/{run_id}/artifacts?per_page=100"],
        check=True,
        capture_output=True,
        text=True,
    )
    artifacts = json.loads(proc.stdout).get("artifacts", [])
    if target_name == "bench-runtime":
        shard_artifacts = [a for a in artifacts if a["name"] == "bench-runtime"]
    else:
        # Shard result artifacts are named `bench-<scenario>`; the heavy
        # `bench-runtime` toolchain bundle shares the `bench-` prefix but is
        # NOT a result shard, so exclude it from prefix matches (the aggregate
        # passes `bench-` to collect every shard without dragging the runtime
        # bundle back down).
        shard_artifacts = [
            a
            for a in artifacts
            if a["name"].startswith(target_name) and a["name"] != "bench-runtime"
        ]
    
    if not shard_artifacts:
        print(f"no artifacts matching {target_name}")
        return

    for a in shard_artifacts:
        # `gh api <archive_download_url>` follows the redirect and streams the
        # zip bytes to stdout; capture that stdout straight into the file.
        # The previous form passed a LIST with `shell=True`, which on POSIX
        # runs only argv[0] (`gh`) as the shell command and hands the rest to
        # it as positional params ($0, $1, …) — so it executed a bare `gh`
        # (which prints help and exits 0), the `>` redirect and `artifact.zip`
        # were never honored, and the file never existed. ZipFile then raised
        # FileNotFoundError, failing every bench shard before it could run.
        with open("artifact.zip", "wb") as fh:
            subprocess.run(
                ["gh", "api", a["archive_download_url"]],
                check=True,
                stdout=fh,
            )
        with zipfile.ZipFile("artifact.zip") as zf:
            zf.extractall(out_dir)
        os.remove("artifact.zip")
        print(f"extracted {a['name']} to {out_dir}")

def record_shard_outcome(scenario, exit_code, budget_seconds, out_path):
    """Write the machine-readable outcome marker for one bench shard (#2737).

    The shard step already KNEW the difference -- GNU `timeout` reports 124 (TERM
    honoured) or 137 (KILL after the grace period) for an over-budget command,
    and anything else for a command that ran to its own conclusion -- and then
    collapsed both onto a bare `exit 1` and one indistinguishable red. 22 of the
    42 failing shards in run 30687540475 were censored at the 42m wall and 20 had
    already failed well under it; nothing downstream could separate them, so a
    real cluster of sub-budget failures hid inside a budget headline.

    A censored shard is NOT a failed measurement. It is an ABSENT one, and the
    aggregate must say so rather than fold it into a total (the rule 5d40cf900
    landed for #2743).
    """

    exit_code = int(exit_code)
    if exit_code == 0:
        outcome = SHARD_OUTCOME_MEASURED
    elif exit_code in (124, 137):
        outcome = SHARD_OUTCOME_CENSORED
    else:
        outcome = SHARD_OUTCOME_ERRORED
    marker = {
        "schema": SHARD_OUTCOME_SCHEMA,
        "scenario_name": scenario,
        "outcome": outcome,
        "exit_code": exit_code,
        "wall_budget_seconds": int(budget_seconds),
        "measured": outcome == SHARD_OUTCOME_MEASURED,
        "run_id": os.environ.get("GITHUB_RUN_ID"),
        "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
        "job_url": (
            f"https://github.com/{os.environ.get('GITHUB_REPOSITORY')}/actions/runs/"
            f"{os.environ.get('GITHUB_RUN_ID')}"
        ),
    }
    path = pathlib.Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(marker, indent=2) + "\n")
    print(f"shard outcome: {scenario} -> {outcome} (exit {exit_code}) written to {path}")
    return marker


def check_python_deps():
    scenario_name = os.environ.get("SCENARIO_NAME", "unknown")
    spec = importlib.util.spec_from_file_location("run_suite_mod", _RUN_SUITE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    import numpy, pandas, lifelines, sksurv, xgboost
    print(f"python deps ok (scenario={scenario_name})")

def _numeric_fit_sec(row):
    """`fit_sec` as a positive float, or None if the row carries no usable time."""
    if str(row.get("status", "")) != "ok":
        return None
    try:
        value = float(row.get("fit_sec"))
    except (TypeError, ValueError):
        return None
    if value <= 0.0 or value != value or value in (float("inf"), float("-inf")):
        return None
    return value


def fit_sec_ratio_rows(rows):
    """gam-vs-reference `fit_sec` ratios, one row per (gam arm, reference arm).

    #2623: every shard has always recorded `fit_sec` for every contender and
    nothing ever compared gam's against the mature reference tool's, so a
    244x row on `papuan_oce4_duchon_k6` sat unread across three nightlies.
    Pairing is by scenario: contenders named `rust_*` are gam's arms, every
    other contender in the same scenario is a reference. That is deliberately
    coarse — it pairs arms that differ in family or basis, and it pairs the
    MCMC references (brms samples, so gam is legitimately faster there) — but
    a coarse ratio that is PRINTED beats an exact one that is skipped. Rows
    come back worst-first so the headline is the largest gap.

    `fit_sec` is not symmetric between arms and the table says so: gam's is
    wall-clock around the `gam fit` subprocess (process start + CSV read +
    model JSON write included), the R arms' is `proc.time()` around the
    fitting call alone. That asymmetry is worth well under a second per fold.
    """
    by_scenario = {}
    for row in rows:
        scenario = str(row.get("scenario_name", "unknown"))
        by_scenario.setdefault(scenario, []).append(row)

    pairs = []
    for scenario in sorted(by_scenario):
        scenario_rows = by_scenario[scenario]
        gam_arms = []
        reference_arms = []
        for row in scenario_rows:
            fit_sec = _numeric_fit_sec(row)
            if fit_sec is None:
                continue
            contender = str(row.get("contender", "unknown"))
            entry = (contender, fit_sec)
            if contender.startswith("rust_"):
                gam_arms.append(entry)
            else:
                reference_arms.append(entry)
        for gam_contender, gam_fit in sorted(gam_arms):
            for ref_contender, ref_fit in sorted(reference_arms):
                pairs.append(
                    {
                        "scenario_name": scenario,
                        "gam_contender": gam_contender,
                        "gam_fit_sec": gam_fit,
                        "reference_contender": ref_contender,
                        "reference_fit_sec": ref_fit,
                        "gam_over_reference": gam_fit / ref_fit,
                    }
                )
    pairs.sort(key=lambda p: -p["gam_over_reference"])
    return pairs


MATCHED_BENCHMARK_CONTENDERS = {
    "rust_gam": "r_mgcv",
    "rust_gamlss": "r_mgcv_gaulss",
}
PERFORMANCE_MEASURES = ("fit_sec", "predict_sec")
ACCURACY_DIRECTIONS = {
    "auc": "higher",
    "c_index": "higher",
    "nagelkerke_r2": "higher",
    "r2": "higher",
    "brier": "lower",
    "logloss": "lower",
    "mse": "lower",
    "rmse": "lower",
    "mae": "lower",
}
# Independent binary64 LAPACK/optimizer paths are not bitwise comparable.
# sqrt(epsilon) is the conventional forward-error scale at which a stable
# floating-point result loses a meaningful directional ordering: below it,
# calling one arm "more accurate" than the other is reporting roundoff, not a
# model-quality difference. Keep this many orders tighter than sampling error.
ACCURACY_NUMERICAL_EQUIVALENCE = sys.float_info.epsilon ** 0.5


def _finite_number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and number not in (float("inf"), float("-inf")) else None


def matched_benchmark_verdict(rows, *, maximum_slowdown=1.2, shard_outcomes=None):
    """Strict #2623 performance/accuracy verdict for genuinely matched arms.

    `shard_outcomes` is the list of per-scenario markers written by
    `record_shard_outcome` (#2737). They partition the scenarios that produced no
    result rows into ones that were CUT SHORT at their wall budget (censored:
    measured nothing, so they are an absence and never a verdict) and ones that
    FAILED under their budget (errored: a real defect that must not hide inside a
    budget headline). Without them every such scenario is one undifferentiated
    "missing", which is exactly how 20 sub-budget failures stayed invisible
    behind 22 budget-censored ones in run 30687540475.
    """

    by_scenario_contender = {}
    observed_scenarios = set()
    for row in rows:
        scenario = str(row.get("scenario_name", ""))
        contender = str(row.get("contender", ""))
        if scenario:
            observed_scenarios.add(scenario)
        by_scenario_contender[(scenario, contender)] = row

    configured = _load_scenario_config().get("scenarios", [])
    expected_scenarios = {str(s["name"]) for s in configured if s.get("name")}
    missing_scenarios = sorted(expected_scenarios - observed_scenarios)

    outcome_by_scenario = {}
    for marker in shard_outcomes or []:
        name = str(marker.get("scenario_name", ""))
        if name:
            outcome_by_scenario[name] = str(marker.get("outcome", ""))
    censored_scenarios = sorted(
        name
        for name, outcome in outcome_by_scenario.items()
        if outcome == SHARD_OUTCOME_CENSORED
    )
    errored_scenarios = sorted(
        name
        for name, outcome in outcome_by_scenario.items()
        if outcome == SHARD_OUTCOME_ERRORED
    )
    # A censored shard is cut short before it can write its result file, so it
    # must not also be reporting rows. If it does, the marker and the rows
    # disagree about what happened and neither may be quietly preferred.
    censored_with_results = sorted(set(censored_scenarios) & observed_scenarios)
    # Neither a marker nor rows: the job never reached the shard step at all
    # (setup failure, cancellation, lost artifact). Also not measured, and also
    # not a failed measurement.
    unreported_scenarios = sorted(
        name
        for name in missing_scenarios
        if name not in outcome_by_scenario
    )
    not_measured_scenarios = sorted(set(censored_scenarios) | set(unreported_scenarios))
    comparisons = []

    for scenario in sorted(observed_scenarios):
        for gam_contender, reference_contender in MATCHED_BENCHMARK_CONTENDERS.items():
            gam_row = by_scenario_contender.get((scenario, gam_contender))
            reference_row = by_scenario_contender.get((scenario, reference_contender))
            # Reference presence is the applicability witness for a matched
            # pair. The runner always emits an explicit failed row when an
            # enabled reference errors, so absence means that pair is not part
            # of this scenario (for example gaulss on a binomial GAM). A stray
            # failed diagnostic gam arm must not invent a missing comparison.
            if reference_row is None:
                continue
            comparison = {
                "scenario_name": scenario,
                "gam_contender": gam_contender,
                "reference_contender": reference_contender,
                "gam_status": None if gam_row is None else gam_row.get("status"),
                "reference_status": None if reference_row is None else reference_row.get("status"),
                "performance": [],
                "accuracy": [],
                "passed": False,
            }
            if (
                gam_row is None
                or reference_row is None
                or gam_row.get("status") != "ok"
                or reference_row.get("status") != "ok"
            ):
                comparison["failure"] = "missing or failed matched contender"
                comparisons.append(comparison)
                continue

            for measure in PERFORMANCE_MEASURES:
                gam_value = _finite_number(gam_row.get(measure))
                reference_value = _finite_number(reference_row.get(measure))
                if gam_value is None or reference_value is None or reference_value <= 0.0:
                    comparison["performance"].append(
                        {"measure": measure, "passed": False, "failure": "missing/non-positive value"}
                    )
                    continue
                ratio = gam_value / reference_value
                comparison["performance"].append(
                    {
                        "measure": measure,
                        "gam": gam_value,
                        "reference": reference_value,
                        "gam_over_reference": ratio,
                        "passed": ratio <= maximum_slowdown,
                    }
                )

            for measure, direction in ACCURACY_DIRECTIONS.items():
                gam_value = _finite_number(gam_row.get(measure))
                reference_value = _finite_number(reference_row.get(measure))
                if gam_value is None and reference_value is None:
                    continue
                if gam_value is None or reference_value is None:
                    comparison["accuracy"].append(
                        {
                            "measure": measure,
                            "direction": direction,
                            "passed": False,
                            "failure": "measure missing from one matched arm",
                        }
                    )
                    continue
                tolerance = ACCURACY_NUMERICAL_EQUIVALENCE * max(
                    1.0, abs(gam_value), abs(reference_value)
                )
                passed = (
                    gam_value + tolerance >= reference_value
                    if direction == "higher"
                    else gam_value <= reference_value + tolerance
                )
                comparison["accuracy"].append(
                    {
                        "measure": measure,
                        "direction": direction,
                        "gam": gam_value,
                        "reference": reference_value,
                        "gam_minus_reference": gam_value - reference_value,
                        "numerical_equivalence_tolerance": tolerance,
                        "passed": passed,
                    }
                )

            measures = comparison["performance"] + comparison["accuracy"]
            comparison["passed"] = (
                len(comparison["performance"]) == len(PERFORMANCE_MEASURES)
                and bool(comparison["accuracy"])
                and all(measure["passed"] for measure in measures)
            )
            comparisons.append(comparison)

    performance_rows = [
        {
            **measure,
            "scenario_name": comparison["scenario_name"],
            "gam_contender": comparison["gam_contender"],
            "reference_contender": comparison["reference_contender"],
        }
        for comparison in comparisons
        for measure in comparison["performance"]
        if "gam_over_reference" in measure
    ]
    worst_performance = (
        max(performance_rows, key=lambda measure: measure["gam_over_reference"])
        if performance_rows
        else None
    )
    complete = not missing_scenarios and observed_scenarios == expected_scenarios
    observed_scope_certified = bool(comparisons) and all(c["passed"] for c in comparisons)
    certified = complete and observed_scope_certified
    return {
        # #2737: the coverage denominator, split by WHY a scenario is absent.
        # `not_measured_scenarios` are absences (cut short at the wall budget, or
        # never reported at all): they are neither passes nor failures and no
        # total may sum over them silently. `errored_scenarios` are failures that
        # ran to their own conclusion inside their budget -- a defect population,
        # not a budget one.
        "shard_outcome_schema": SHARD_OUTCOME_SCHEMA,
        "shard_outcomes_seen": len(outcome_by_scenario),
        "censored_scenarios": censored_scenarios,
        "errored_scenarios": errored_scenarios,
        "unreported_scenarios": unreported_scenarios,
        "not_measured_scenarios": not_measured_scenarios,
        "censored_with_results": censored_with_results,
        "contract": {
            "maximum_slowdown": maximum_slowdown,
            "accuracy": (
                "no loss on every shared reported accuracy measure, with "
                "sqrt(binary64 epsilon) relative numerical equivalence"
            ),
            "accuracy_numerical_equivalence": ACCURACY_NUMERICAL_EQUIVALENCE,
            "missing_or_failed_pairs": "fail",
            "full_suite_required": True,
        },
        "configured_scenario_count": len(expected_scenarios),
        "observed_scenario_count": len(observed_scenarios),
        "missing_scenarios": missing_scenarios,
        "full_suite": complete,
        "comparisons": comparisons,
        "worst_performance_measure": worst_performance,
        "observed_scope_certified": observed_scope_certified,
        "certified": certified,
    }


def benchmark_verdict_enforcement(verdict, *, require_full_suite=False):
    """Turn the published verdict into a pass/fail decision with named reasons.

    #2623: the verdict was computed, written to `benchmark-verdict.json` and
    printed into the step summary, and then nothing read it — so a matched arm
    could regress by 250x and every job still reported success. This is the
    consumer that makes it a gate.

    The bound is relative by construction: each measure is compared against its
    matched reference arm measured in the same job on the same runner, so there
    is no absolute second-count to go stale and no cross-host baseline to be
    invalidated by a runner change.

    Zero observed comparisons is a FAILURE, not a pass. A run whose shards all
    died measures nothing, and "nothing failed" is exactly how this regression
    stayed invisible.

    `require_full_suite` is opt-in so a deliberately targeted dispatch can be
    graded on the scope it actually ran; the scheduled full-matrix path passes
    it, and missing scenarios are reported either way.
    """

    failures = []
    comparisons = verdict.get("comparisons") or []
    contract = verdict.get("contract") or {}
    maximum_slowdown = contract.get("maximum_slowdown")

    if not comparisons:
        failures.append(
            "no matched comparison was measured: "
            f"{verdict.get('observed_scenario_count', 0)}/"
            f"{verdict.get('configured_scenario_count', 0)} scenarios observed. "
            "A run that measured nothing cannot certify anything."
        )

    for comparison in comparisons:
        if comparison.get("passed"):
            continue
        label = (
            f"{comparison.get('scenario_name')} "
            f"[{comparison.get('gam_contender')} vs {comparison.get('reference_contender')}]"
        )
        if comparison.get("failure"):
            failures.append(f"{label}: {comparison['failure']}")
        for measure in comparison.get("performance") or []:
            if measure.get("passed"):
                continue
            if "gam_over_reference" in measure:
                failures.append(
                    f"{label}: {measure['measure']} is "
                    f"{measure['gam_over_reference']:.4f}x its matched reference "
                    f"({measure['gam']:.6g}s vs {measure['reference']:.6g}s), "
                    f"over the {maximum_slowdown}x bound"
                )
            else:
                failures.append(
                    f"{label}: {measure.get('measure')} "
                    f"{measure.get('failure', 'did not pass')}"
                )
        for measure in comparison.get("accuracy") or []:
            if measure.get("passed"):
                continue
            if "gam_minus_reference" in measure:
                failures.append(
                    f"{label}: {measure['measure']} regressed "
                    f"({measure['gam']:.12g} vs reference {measure['reference']:.12g}, "
                    f"{measure['direction']} is better, "
                    f"delta {measure['gam_minus_reference']:.3e} beyond "
                    f"{measure['numerical_equivalence_tolerance']:.3e})"
                )
            else:
                failures.append(
                    f"{label}: {measure.get('measure')} "
                    f"{measure.get('failure', 'did not pass')}"
                )

    # #2737: absence has causes, and they are not interchangeable.
    censored = verdict.get("censored_scenarios") or []
    errored = verdict.get("errored_scenarios") or []
    unreported = verdict.get("unreported_scenarios") or []
    not_measured = verdict.get("not_measured_scenarios") or []
    conflicted = verdict.get("censored_with_results") or []
    missing = verdict.get("missing_scenarios") or []

    def _name_list(names, limit=10):
        return ", ".join(names[:limit]) + (" ..." if len(names) > limit else "")

    # A shard that ran to its own conclusion and failed INSIDE its wall budget is
    # a defect, not a budget case, and it fails the gate whatever the requested
    # scope is. Previously it was reported only as one more "missing" scenario,
    # so a cluster of sub-budget failures could be read off as a budget headline.
    if errored:
        failures.append(
            f"{len(errored)} scenario(s) FAILED inside their own wall budget "
            "(a defect population, not a budget one): " + _name_list(errored)
        )
    if conflicted:
        failures.append(
            f"{len(conflicted)} scenario(s) reported result rows AND a censored "
            "shard marker, so the marker and the rows disagree about whether the "
            "shard finished: " + _name_list(conflicted)
        )
    if require_full_suite and not_measured:
        failures.append(
            f"{len(not_measured)} configured scenario(s) were NOT MEASURED "
            f"({len(censored)} cut short at their wall budget, {len(unreported)} "
            "never reported), so the full-suite contract is unmet -- this is a "
            "coverage gap, not a verdict on those scenarios: "
            + _name_list(not_measured)
        )

    worst = verdict.get("worst_performance_measure")
    configured_count = verdict.get("configured_scenario_count", 0)
    measured_count = verdict.get("observed_scenario_count", 0)
    lines = [
        "",
        "### #2623 benchmark regression gate",
        "",
        (
            f"Bound: every matched performance measure <= {maximum_slowdown}x its "
            "reference arm, and no accuracy measure worse than numerical equivalence."
        ),
        (
            f"Measured: {len(comparisons)} matched comparison(s) over "
            f"{measured_count}/{configured_count} configured scenarios."
        ),
        (
            # #2737 / #2743: a total that sums over a scenario which was cut short
            # must say so. Silent truncation reads as coverage.
            f"NOT MEASURED: {len(not_measured)}/{configured_count} scenarios "
            f"({len(censored)} censored at their wall budget, {len(unreported)} "
            "never reported). These are absences: they are neither passes nor "
            "failures and no count above includes them."
        ),
        (
            f"Failed inside budget: {len(errored)}/{configured_count} scenarios. "
            "These ran to their own conclusion and are real defects."
        ),
    ]
    if censored:
        lines.append(f"Censored: {_name_list(censored, limit=len(censored))}")
    if errored:
        lines.append(f"Errored: {_name_list(errored, limit=len(errored))}")
    if missing and not verdict.get("shard_outcomes_seen"):
        lines.append(
            f"No shard outcome markers were published, so the {len(missing)} absent "
            "scenario(s) could not be split into censored vs errored: "
            + _name_list(missing)
        )
    if worst is not None:
        lines.append(
            f"Worst performance ratio: {worst['gam_over_reference']:.4f}x on "
            f"`{worst['scenario_name']}` / `{worst['measure']}`."
        )
    if failures:
        lines.append("")
        lines.append("**GATE FAILED**")
        lines.extend(f"- {failure}" for failure in failures)
    else:
        lines.append("")
        lines.append("**GATE PASSED**")

    return {"passed": not failures, "failures": failures, "summary_lines": lines}


def enforce_verdict(require_full_suite=False):
    """Fail the job when the published benchmark verdict does not pass."""

    verdict_path = pathlib.Path("bench/benchmark-verdict.json")
    if not verdict_path.exists():
        sys.exit(
            f"#2623 benchmark regression gate: {verdict_path} does not exist. "
            "The merge step must publish a verdict before it can be enforced."
        )
    outcome = benchmark_verdict_enforcement(
        json.loads(verdict_path.read_text()), require_full_suite=require_full_suite
    )
    summary = "\n".join(outcome["summary_lines"])
    print(summary)
    step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if step_summary:
        with open(step_summary, "a") as f:
            f.write(summary + "\n")
    if not outcome["passed"]:
        sys.exit(1)


def _fit_sec_ratio_summary_lines(pairs):
    if not pairs:
        return [
            "",
            "### gam vs reference fit time",
            "",
            "No scenario produced an `ok` gam arm and an `ok` reference arm with "
            "a positive `fit_sec`, so there is no ratio to report.",
        ]
    worst = pairs[0]
    lines = [
        "",
        "### gam vs reference fit time",
        "",
        (
            f"Worst gam/reference `fit_sec` ratio: **{worst['gam_over_reference']:.1f}x** "
            f"({worst['scenario_name']}: {worst['gam_contender']} "
            f"{worst['gam_fit_sec']:.2f}s vs {worst['reference_contender']} "
            f"{worst['reference_fit_sec']:.2f}s)."
        ),
        "",
        (
            "`fit_sec` is the sum over CV folds. gam's is wall-clock around the "
            "`gam fit` subprocess; the R arms' is `proc.time()` around the fitting "
            "call alone. Rows are worst-first; gam is expected to win against the "
            "MCMC arms, which sample rather than optimize."
        ),
        "",
        "| Scenario | gam arm | gam fit (s) | Reference | ref fit (s) | gam/ref |",
        "|----------|---------|-------------|-----------|-------------|---------|",
    ]
    for pair in pairs:
        lines.append(
            f"| {pair['scenario_name']} | {pair['gam_contender']} | "
            f"{pair['gam_fit_sec']:.2f} | {pair['reference_contender']} | "
            f"{pair['reference_fit_sec']:.2f} | {pair['gam_over_reference']:.1f}x |"
        )
    return lines


def _shard_outcome_summary_lines(verdict):
    """#2737: report coverage with its causes, so no absence reads as a verdict."""

    configured = verdict.get("configured_scenario_count", 0)
    measured = verdict.get("observed_scenario_count", 0)
    censored = verdict.get("censored_scenarios") or []
    errored = verdict.get("errored_scenarios") or []
    unreported = verdict.get("unreported_scenarios") or []
    lines = [
        "",
        "### Scenario coverage",
        "",
        (
            f"**{measured} of {configured} configured scenarios were MEASURED.** "
            f"{len(censored) + len(unreported)} were NOT MEASURED and "
            f"{len(errored)} failed inside their own wall budget."
        ),
        "",
        "| Outcome | Count | Meaning |",
        "|---------|------:|---------|",
        f"| measured | {measured} | Ran to a verdict; its rows are in the table above. |",
        (
            f"| censored | {len(censored)} | Cut short at its wall budget (GNU `timeout` "
            "124/137). **Measured nothing** -- neither a pass nor a fail. |"
        ),
        (
            f"| errored | {len(errored)} | Failed INSIDE its budget. A real defect, and "
            "not a budget case. |"
        ),
        (
            f"| unreported | {len(unreported)} | Published neither rows nor an outcome "
            "marker (the job never reached the shard step). Also not measured. |"
        ),
    ]
    for label, names in (
        ("Censored (not measured)", censored),
        ("Errored inside budget (defects)", errored),
        ("Unreported (not measured)", unreported),
    ):
        if names:
            lines.extend(["", f"{label}: `" + "`, `".join(names) + "`"])
    return lines


def format_results():
    from datetime import datetime, timezone

    def fmt_num(v, digits=4):
        if v is None:
            return "—"
        try:
            return f"{float(v):.{digits}f}"
        except Exception:
            return "—"

    def fmt_status(row):
        status = str(row.get("status", "unknown"))
        if status == "ok":
            return "ok"
        # A contender's error text is multi-line and contains `|` (the fit
        # log lines it quotes do), both of which terminate a markdown table
        # cell — so every failed row used to shred the rest of the table it
        # appeared in. Flatten to one cell; the full text is in the shard
        # artifact and in `results.nightly.json`.
        error = " ".join(str(row.get("error", "unknown error")).split())
        return f"failed: {error.replace('|', '/')}"

    # Each `bench-<scenario>` shard artifact extracts to a `<scenario>.json`
    # file holding ONE shard payload of the shape `bench/run_suite.py` writes:
    #   {"created_at_utc": ..., "evaluation": {...}, "results": [<row>, ...]}
    # where every row is one per-contender measurement (scenario_name,
    # contender, status, fit_sec, predict_sec, metric columns). The merged
    # `results.nightly.json` must therefore be a dict carrying the FLATTENED
    # list of those rows under a "results" key — that is exactly what
    # `bench/generate_figures.py` (`payload["results"]`) and the nightly
    # dashboard consume. Recurse + shape-filter the downloaded tree so the
    # merge is robust to however upload-artifact nested the file inside its
    # zip, and so a stray non-shard JSON cannot derail the merge.
    root = pathlib.Path("bench/artifacts")
    rows = []
    shard_files = 0
    shard_outcomes = []
    for p in sorted(root.rglob("*.json")):
        try:
            payload = json.loads(p.read_text())
        except Exception as e:
            print(f"Failed to load {p}: {e}")
            continue
        if not isinstance(payload, dict):
            continue
        # #2737: every shard publishes an outcome marker whether it finished, was
        # censored at its wall budget, or failed inside it. The marker is the only
        # thing a censored shard publishes at all -- its result JSON was never
        # written -- so it is what keeps a cut-short scenario from being counted
        # as an ordinary red.
        if payload.get("schema") == SHARD_OUTCOME_SCHEMA:
            shard_outcomes.append(payload)
            continue
        if not isinstance(payload.get("results"), list):
            continue
        rows.extend(payload["results"])
        shard_files += 1

    ratio_pairs = fit_sec_ratio_rows(rows)
    benchmark_verdict = matched_benchmark_verdict(rows, shard_outcomes=shard_outcomes)
    merged = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "results": rows,
        # #2623: carried alongside `results` (which `bench/generate_figures.py`
        # and the dashboard read) so the speed comparison is in the published
        # artifact, not only in a step summary that expires with the run.
        "fit_sec_ratios": ratio_pairs,
        "benchmark_verdict": benchmark_verdict,
        # #2737: carried into the published artifact so a later reader can tell
        # which scenarios this run actually measured without re-deriving it from
        # job durations.
        "shard_outcomes": shard_outcomes,
    }
    with open("bench/results.nightly.json", "w") as f:
        json.dump(merged, f, indent=2)
    with open("bench/benchmark-verdict.json", "w") as f:
        json.dump(benchmark_verdict, f, indent=2)
    print(
        f"merged {len(rows)} contender rows from {shard_files} shard file(s) "
        "into bench/results.nightly.json"
    )

    run_url = f"https://github.com/{os.environ.get('GITHUB_REPOSITORY')}/actions/runs/{os.environ.get('GITHUB_RUN_ID')}"
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    lines = [
        "## Benchmark Summary",
        f"Run: {run_url}",
        f"Generated: {timestamp}",
        f"Merged {len(rows)} contender rows from {shard_files} scenario shard(s).",
        "",
        "| Scenario | Contender | Status | Fit (s) | Predict (s) |",
        "|----------|-----------|--------|---------|-------------|",
    ]

    for r in sorted(
        rows, key=lambda r: (str(r.get("scenario_name", "")), str(r.get("contender", "")))
    ):
        scen = r.get("scenario_name", "unknown")
        contender = r.get("contender", "unknown")
        stat = fmt_status(r)
        fit_s = fmt_num(r.get("fit_sec"), digits=2)
        pred_s = fmt_num(r.get("predict_sec"), digits=2)
        lines.append(f"| {scen} | {contender} | {stat} | {fit_s} | {pred_s} |")

    lines.extend(_shard_outcome_summary_lines(benchmark_verdict))
    lines.extend(_fit_sec_ratio_summary_lines(ratio_pairs))
    lines.extend(
        [
            "",
            "### Strict matched benchmark verdict",
            "",
            (
                "**Observed scope "
                f"{'PASSED' if benchmark_verdict['observed_scope_certified'] else 'FAILED'}; "
                "full suite "
                f"{'CERTIFIED' if benchmark_verdict['certified'] else 'NOT CERTIFIED'}** — "
                f"{benchmark_verdict['observed_scenario_count']}/"
                f"{benchmark_verdict['configured_scenario_count']} scenarios observed; "
                f"{len(benchmark_verdict['comparisons'])} matched comparison(s)."
            ),
        ]
    )
    worst_measure = benchmark_verdict["worst_performance_measure"]
    if worst_measure is not None:
        lines.append(
            f"Worst measured performance ratio: {worst_measure['gam_over_reference']:.3f}x "
            f"on `{worst_measure['scenario_name']}` / `{worst_measure['measure']}` "
            f"({worst_measure['gam_contender']}` vs `{worst_measure['reference_contender']}`)."
        )

    summary = "\n".join(lines)
    print(summary)

    step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if step_summary:
        with open(step_summary, "a") as f:
            f.write(summary + "\n")

    # Lets the figure steps distinguish "the nightly measured nothing" from
    # "the nightly measured something" now that the job runs with failed legs.
    step_output = os.environ.get("GITHUB_OUTPUT")
    if step_output:
        with open(step_output, "a") as f:
            f.write(f"merged_rows={len(rows)}\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: task_runner.py <task> [args]")
    
    task = sys.argv[1]
    if task == "validate_schemas":
        validate_schemas()
    elif task == "validate_geo_subpop":
        validate_geo_subpop()
    elif task == "build_matrix":
        parser = argparse.ArgumentParser()
        parser.add_argument("--scenarios")
        build_matrix(parser.parse_args(sys.argv[2:]).scenarios)
    elif task == "extract_maturin_wheel":
        out_dir = sys.argv[2] if len(sys.argv) > 2 else "gamfit"
        extract_maturin_wheel(out_dir)
    elif task == "download_artifacts":
        download_artifacts(sys.argv[2], sys.argv[3])
    elif task == "check_python_deps":
        check_python_deps()
    elif task == "record_shard_outcome":
        parser = argparse.ArgumentParser()
        parser.add_argument("--scenario", required=True)
        parser.add_argument("--exit-code", required=True, type=int)
        parser.add_argument("--budget-seconds", required=True, type=int)
        parser.add_argument("--out", required=True)
        args = parser.parse_args(sys.argv[2:])
        record_shard_outcome(args.scenario, args.exit_code, args.budget_seconds, args.out)
    elif task == "print_wall_budgets":
        for name, budget in sorted(scenario_wall_budgets().items()):
            print(
                f"{name}\t{budget['budget_seconds']}s\t{budget['timeout_minutes']}m\t"
                f"{budget['basis']}"
            )
    elif task == "format_results":
        format_results()
    elif task == "enforce_verdict":
        parser = argparse.ArgumentParser()
        parser.add_argument("--require-full-suite", action="store_true")
        enforce_verdict(parser.parse_args(sys.argv[2:]).require_full_suite)
    else:
        sys.exit(f"Unknown task: {task}")
