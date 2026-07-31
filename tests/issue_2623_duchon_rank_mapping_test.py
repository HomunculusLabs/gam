import importlib.util
import re
import typing
import unittest
from pathlib import Path

import numpy as np


_RUN_SUITE_PATH = Path(__file__).resolve().parents[1] / "bench" / "run_suite.py"
_SPEC = importlib.util.spec_from_file_location("issue_2623_run_suite", _RUN_SUITE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"failed to load benchmark runner from {_RUN_SUITE_PATH}")
_RUN_SUITE: typing.Any = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_RUN_SUITE)


class DuchonRankMappingTests(unittest.TestCase):
    def test_joint_duchon_k_is_retained_rank_not_knot_count(self) -> None:
        formula = _RUN_SUITE._rust_joint_spatial_term(
            "duchon",
            ["pc1", "pc2", "pc3", "pc4"],
            6,
            "",
        )
        self.assertEqual(
            formula,
            "duchon(pc1, pc2, pc3, pc4, rank=6, order=0, power=1.5)",
        )
        self.assertNotIn("centers=", formula)

    def test_external_metrics_are_not_truncated_before_accuracy_comparison(self) -> None:
        source = (_RUN_SUITE_PATH.parent / "_run_suite_external.py").read_text()
        encoders = re.findall(r"toJSON\(out,[^)]*\)", source)
        self.assertTrue(encoders)
        self.assertTrue(
            all("digits=NA" in encoder for encoder in encoders),
            "every R result path must preserve full double precision",
        )

    def test_duchon_reference_uses_the_shared_landmark_experiment(self) -> None:
        source = (_RUN_SUITE_PATH.parent / "_run_suite_external.py").read_text()
        self.assertIn('"duchon_landmark_cols"', source)
        self.assertIn("landmark_frame <- landmark_frame[!duplicated(landmark_frame)", source)
        self.assertIn("set.seed(duchon_landmark_seed)", source)
        self.assertIn("list(knots=duchon_knots)", source)

    def test_prediction_timing_is_warmed_bounded_and_output_checked(self) -> None:
        calls = 0

        def predict() -> list[float]:
            nonlocal calls
            calls += 1
            return [1.0, 2.0, 3.0]

        prediction, seconds = _RUN_SUITE._time_stable_mean_prediction(predict)
        self.assertEqual(prediction.tolist(), [1.0, 2.0, 3.0])
        self.assertGreaterEqual(seconds, 0.0)
        self.assertGreaterEqual(
            calls,
            1 + _RUN_SUITE.PREDICTION_TIMING_MIN_REPETITIONS,
        )
        self.assertLessEqual(
            calls,
            1 + _RUN_SUITE.PREDICTION_TIMING_MAX_REPETITIONS,
        )

        changing_call = 0

        def changing_prediction() -> list[float]:
            nonlocal changing_call
            changing_call += 1
            return [float(changing_call)]

        with self.assertRaisesRegex(RuntimeError, "changed its output"):
            _RUN_SUITE._time_stable_mean_prediction(changing_prediction)

    def test_cv_design_and_penalty_diagnostics_stay_in_their_fold_charts(self) -> None:
        rust_fold_0 = np.column_stack((np.ones(4), [-1.0, -0.2, 0.4, 1.1]))
        rust_fold_1 = np.column_stack((np.ones(4), [-1.2, -0.4, 0.5, 1.3]))
        mgcv_fold_0 = rust_fold_0 @ np.diag([1.0, 2.0])
        mgcv_fold_1 = rust_fold_1 @ np.diag([1.0, 3.0])

        def result(
            contender: str,
            designs: list[np.ndarray],
            penalties: list[float],
            smoothing_parameters: list[float],
        ) -> dict[str, typing.Any]:
            rows = np.vstack(designs)
            per_fold = []
            for penalty, smoothing_parameter in zip(
                penalties, smoothing_parameters, strict=True
            ):
                quality: dict[str, typing.Any] = {
                    "smoothing_parameters": [smoothing_parameter],
                }
                if contender == "rust_gam":
                    quality["duchon_fitted_primary_penalty_matrices"] = [
                        [[penalty]]
                    ]
                else:
                    quality["smooth_penalty_matrices"] = [
                        {"S": [[penalty]]}
                    ]
                per_fold.append(quality)
            return {
                "status": "ok",
                "scenario_name": "fold_chart_regression",
                "contender": contender,
                "fit_quality": {"per_fold": per_fold},
                "plot_payload": {
                    "linear_predictor": [0.0] * rows.shape[0],
                    "_diagnostic_eta_variance": [0.0] * rows.shape[0],
                    "_diagnostic_design_rows": rows.tolist(),
                    "_diagnostic_design_columns": rows.shape[1],
                    "_diagnostic_design_fold_sizes": [len(design) for design in designs],
                },
            }

        results = [
            result("rust_gam", [rust_fold_0, rust_fold_1], [2.0, 2.0], [0.1, 0.1]),
            # chart.T @ [[0,0],[0,2]] @ chart is 8 and 18 respectively;
            # multiplying by 0.5 gives these reference penalties. Therefore
            # 0.2 * 0.5 maps exactly to Rust lambda 0.1 on both folds.
            result("r_mgcv", [mgcv_fold_0, mgcv_fold_1], [4.0, 9.0], [0.2, 0.2]),
        ]
        _RUN_SUITE._attach_design_subspace_diagnostics(results)

        quality = results[0]["fit_quality"]
        self.assertLess(
            quality["design_relative_projection_residuals"]["rust_outside_mgcv"],
            1e-12,
        )
        self.assertLess(
            quality["design_relative_projection_residuals"]["mgcv_outside_rust"],
            1e-12,
        )
        self.assertEqual(quality["design_numerical_ranks"]["rust_gam"], [2, 2])
        self.assertEqual(len(quality["design_per_fold"]), 2)
        self.assertLess(
            quality["duchon_penalty_congruence"][
                "max_relative_residual_after_scalar"
            ],
            1e-12,
        )
        self.assertLess(
            quality["duchon_penalty_congruence"]["max_abs_lambda_difference"],
            1e-12,
        )
        for entry in results:
            self.assertNotIn(
                "_diagnostic_design_fold_sizes", entry["plot_payload"]
            )
            for fold in entry["fit_quality"]["per_fold"]:
                self.assertNotIn("duchon_fitted_primary_penalty_matrices", fold)
                self.assertNotIn("smooth_penalty_matrices", fold)


if __name__ == "__main__":
    unittest.main()
