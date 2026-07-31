import argparse
import glob
import importlib.util
import json
import os
import pathlib
import shutil
import subprocess
import sys
import zipfile


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_RUN_SUITE_PATH = _REPO_ROOT / "bench" / "run_suite.py"
_SCENARIOS_PATH = _REPO_ROOT / "bench" / "scenarios.json"


def _load_scenario_config():
    return json.loads(_SCENARIOS_PATH.read_text())


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

    if is_nightly:
        if requested_scenarios:
            raise SystemExit("Scheduled benchmark runs cannot select a scenario subset")
        selected = names
    else:
        if not requested_scenarios:
            raise SystemExit(
                "workflow_dispatch requires --scenarios with one or more comma-separated names"
            )
        selected = list(
            dict.fromkeys(
                name.strip()
                for name in requested_scenarios.split(",")
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
    serial = [s for s in selected if s in SERIAL_SCENARIOS]
    parallel = [s for s in selected if s not in SERIAL_SCENARIOS]
    parallel_matrix = {"scenario": parallel}
    serial_matrix = {"scenario": serial}
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


def _finite_number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and number not in (float("inf"), float("-inf")) else None


def matched_benchmark_verdict(rows, *, maximum_slowdown=1.2):
    """Strict #2623 performance/accuracy verdict for genuinely matched arms."""

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
    comparisons = []

    for scenario in sorted(observed_scenarios):
        for gam_contender, reference_contender in MATCHED_BENCHMARK_CONTENDERS.items():
            gam_row = by_scenario_contender.get((scenario, gam_contender))
            reference_row = by_scenario_contender.get((scenario, reference_contender))
            if gam_row is None and reference_row is None:
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
                tolerance = 1e-12 * max(1.0, abs(gam_value), abs(reference_value))
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
    certified = complete and bool(comparisons) and all(c["passed"] for c in comparisons)
    return {
        "contract": {
            "maximum_slowdown": maximum_slowdown,
            "accuracy": "no loss on every shared reported accuracy measure",
            "missing_or_failed_pairs": "fail",
            "full_suite_required": True,
        },
        "configured_scenario_count": len(expected_scenarios),
        "observed_scenario_count": len(observed_scenarios),
        "missing_scenarios": missing_scenarios,
        "full_suite": complete,
        "comparisons": comparisons,
        "worst_performance_measure": worst_performance,
        "certified": certified,
    }


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
    for p in sorted(root.rglob("*.json")):
        try:
            payload = json.loads(p.read_text())
        except Exception as e:
            print(f"Failed to load {p}: {e}")
            continue
        if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
            continue
        rows.extend(payload["results"])
        shard_files += 1

    ratio_pairs = fit_sec_ratio_rows(rows)
    benchmark_verdict = matched_benchmark_verdict(rows)
    merged = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "results": rows,
        # #2623: carried alongside `results` (which `bench/generate_figures.py`
        # and the dashboard read) so the speed comparison is in the published
        # artifact, not only in a step summary that expires with the run.
        "fit_sec_ratios": ratio_pairs,
        "benchmark_verdict": benchmark_verdict,
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

    lines.extend(_fit_sec_ratio_summary_lines(ratio_pairs))
    lines.extend(
        [
            "",
            "### Strict matched benchmark verdict",
            "",
            (
                f"**{'CERTIFIED' if benchmark_verdict['certified'] else 'NOT CERTIFIED'}** — "
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
    elif task == "format_results":
        format_results()
    else:
        sys.exit(f"Unknown task: {task}")
