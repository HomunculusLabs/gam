import importlib.util
import re
import typing
import unittest
from pathlib import Path


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


if __name__ == "__main__":
    unittest.main()
