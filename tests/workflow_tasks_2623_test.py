import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


_REPO_ROOT = Path(__file__).resolve().parents[1]
_TASKS_PATH = _REPO_ROOT / ".github" / "scripts" / "workflow_tasks.py"
_BENCHMARK_WORKFLOW_PATH = _REPO_ROOT / ".github" / "workflows" / "benchmark.yml"
_SPEC = importlib.util.spec_from_file_location("workflow_tasks_2623", _TASKS_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"failed to load {_TASKS_PATH}")
_TASKS = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_TASKS)


class WorkflowTasks2623Tests(unittest.TestCase):
    def test_aggregate_sparse_checkout_contains_its_repository_dependencies(self) -> None:
        workflow = _BENCHMARK_WORKFLOW_PATH.read_text()
        aggregate = workflow.split("  aggregate:", 1)[1]
        checkout = aggregate.split("- name: Download all shard artifacts", 1)[0]
        for required_path in (
            "bench",
            ".github/scripts/workflow_tasks.py",
            ".github/actions/publish-gha-results",
        ):
            self.assertIn(
                required_path,
                checkout,
                f"aggregate checkout omits repository dependency {required_path}",
            )

    def test_targeted_matrix_preserves_requested_order_and_rejects_unknown_names(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            output = Path(td) / "output"
            with patch.dict(
                os.environ,
                {"GITHUB_EVENT_NAME": "workflow_dispatch", "GITHUB_OUTPUT": str(output)},
                clear=False,
            ):
                _TASKS.build_matrix("wine_temp_vs_year,papuan_oce4_psperpc_k6")
            values = dict(
                line.split("=", 1)
                for line in output.read_text().splitlines()
            )
            self.assertEqual(
                json.loads(values["parallel_matrix"])["scenario"],
                ["wine_temp_vs_year", "papuan_oce4_psperpc_k6"],
            )
            with patch.dict(
                os.environ,
                {"GITHUB_EVENT_NAME": "workflow_dispatch", "GITHUB_OUTPUT": str(output)},
                clear=False,
            ):
                with self.assertRaisesRegex(SystemExit, "Unknown benchmark scenario"):
                    _TASKS.build_matrix("does_not_exist")

    def test_matched_verdict_checks_both_speed_measures_and_every_accuracy_direction(self) -> None:
        rows = [
            {
                "scenario_name": "wine_temp_vs_year",
                "contender": "rust_gam",
                "status": "ok",
                "fit_sec": 1.2,
                "predict_sec": 0.6,
                "rmse": 0.4,
                "r2": 0.8,
            },
            {
                "scenario_name": "wine_temp_vs_year",
                "contender": "r_mgcv",
                "status": "ok",
                "fit_sec": 1.0,
                "predict_sec": 0.5,
                "rmse": 0.4,
                "r2": 0.8,
            },
        ]
        verdict = _TASKS.matched_benchmark_verdict(rows)
        comparison = verdict["comparisons"][0]
        self.assertTrue(comparison["passed"])
        self.assertFalse(verdict["certified"], "one targeted scenario cannot certify the full suite")

        rows[0]["rmse"] = 0.4000001
        verdict = _TASKS.matched_benchmark_verdict(rows)
        comparison = verdict["comparisons"][0]
        failed = [m["measure"] for m in comparison["accuracy"] if not m["passed"]]
        self.assertEqual(failed, ["rmse"])
        self.assertFalse(comparison["passed"])

        rows[0]["rmse"] = 0.4
        rows[0]["r2"] = 0.7999999
        verdict = _TASKS.matched_benchmark_verdict(rows)
        comparison = verdict["comparisons"][0]
        failed = [m["measure"] for m in comparison["accuracy"] if not m["passed"]]
        self.assertEqual(failed, ["r2"])
        self.assertFalse(comparison["passed"])


if __name__ == "__main__":
    unittest.main()
