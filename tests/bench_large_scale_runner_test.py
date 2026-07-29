import typing
import csv
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[1]
_RUNNER_PATH = _REPO_ROOT / "bench" / "large_scale" / "runner.py"
_SPEC = importlib.util.spec_from_file_location("bench_large_scale_runner", _RUNNER_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"failed to load large-scale benchmark runner from {_RUNNER_PATH}")
_RUNNER: typing.Any = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _RUNNER
_SPEC.loader.exec_module(_RUNNER)


def _write_csv(path: Path, rows: typing.Sequence[typing.Mapping[str, object]]) -> None:
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class LargeScaleRunnerTests(unittest.TestCase):
    def test_terminal_output_sanitizer_removes_cursor_controls_across_chunks(self) -> None:
        sanitizer = _RUNNER._TerminalOutputSanitizer()
        text = (
            sanitizer.feed("progress\r        [1s] ok \x1b[")
            + sanitizer.feed("2K next\x1b]0;title")
            + sanitizer.feed("\x07 done\n")
        )
        self.assertEqual(text, "progress\n[1s] ok  next done\n")

    def test_default_large_scale_matrix_keeps_both_400k_binomial_marginal_slope_lanes(self) -> None:
        cfg = _RUNNER.load_config(_RUNNER.DEFAULT_CONFIG)

        self.assertEqual(int(cfg["target_n"]), 400000)

        specs = _RUNNER.build_method_specs(cfg)

        marginal_slope_disease = [
            s
            for s in specs
            if s.dataset == "disease"
            and s.backend == "rust_gam"
            and s.family == "binomial"
            and s.marginal_slope
        ]
        self.assertEqual(
            len(marginal_slope_disease),
            2,
            "expected rigid and warped disease + Rust + binomial + marginal-slope lanes in the "
            "default large-scale matrix; found "
            f"{[s.name for s in marginal_slope_disease]}",
        )
        lanes = {lane.name: lane for lane in marginal_slope_disease}
        rigid = lanes["rust_margslope_aniso_duchon16d_rigid"]
        warped = lanes["rust_margslope_aniso_duchon16d_linkwiggle_scorewarp_fast"]
        for lane in (rigid, warped):
            self.assertEqual(lane.spatial_basis, "duchon")
            self.assertEqual(lane.pc_count, 16, f"{lane.name} must run on 16 PCs")
            self.assertTrue(lane.scale_dimensions, f"{lane.name} must enable per-axis scales")
            self.assertEqual(lane.z_column, "pgs_ctn_z", f"{lane.name} must read CTN z column")
            self.assertEqual(lane.centers, 24)
        self.assertIsNone(rigid.mean_linkwiggle_knots)
        self.assertIsNone(rigid.logslope_linkwiggle_knots)
        self.assertEqual(warped.mean_linkwiggle_knots, 8)
        self.assertEqual(warped.logslope_linkwiggle_knots, 8)

    def test_marginal_slope_formula_supports_linkwiggle_and_scorewarp(self) -> None:
        spec = _RUNNER.MethodSpec(
            name="margslope_variant",
            dataset="disease",
            backend="rust_gam",
            family="binomial",
            spatial_basis="duchon",
            marginal_slope=True,
            scale_dimensions=True,
            z_column="pgs_ctn_z",
            mean_linkwiggle_knots=8,
            logslope_linkwiggle_knots=7,
        )
        mean_formula, logslope_formula = _RUNNER.rust_marginal_slope_formula_classification(spec, centers=20)
        self.assertIn("duchon(pc1_std, pc2_std", mean_formula)
        self.assertIn("centers=20", mean_formula)
        self.assertIn("order=0", mean_formula)
        self.assertIn("power=9", mean_formula)
        self.assertIn("length_scale=1", mean_formula)
        self.assertNotIn("pgs_ctn_z", mean_formula)
        self.assertIn("linkwiggle(internal_knots=8)", mean_formula)
        self.assertIn("linkwiggle(internal_knots=7)", logslope_formula)

    def test_large_scale_preflight_rejects_unsafe_dense_duchon_width_before_allocation(self) -> None:
        report = _RUNNER.preflight_marginal_slope_large_scale(
            n_train=400000,
            d_pc=16,
            centers=1400,
        )
        self.assertEqual(report.status, "FAIL")
        text = "\n".join(report.lines)
        self.assertIn("anisotropic derivative dense estimate", text)
        self.assertIn("status: FAIL", text)

    def test_large_scale_preflight_accepts_production_marginal_slope_width(self) -> None:
        report = _RUNNER.preflight_marginal_slope_large_scale(
            n_train=400000,
            d_pc=16,
            centers=24,
            linkwiggle_knots=8,
            scorewarp_knots=8,
        )
        self.assertEqual(report.status, "PASS")
        text = "\n".join(report.lines)
        self.assertIn("Duchon tuple: order=0, power=9, length_scale=1", text)
        self.assertIn("Duchon smooth: lazy chunked", text)
        self.assertIn("anisotropy derivatives: implicit streaming", text)

    def test_ctn_preflight_uses_factored_kronecker_not_dense_rowwise_product(self) -> None:
        report = _RUNNER.preflight_ctn_score_warp(
            n_train=400000,
            p_response=12,
            p_cov=50,
        )
        self.assertEqual(report.status, "PASS")
        text = "\n".join(report.lines)
        self.assertIn("CTN Kronecker: factored", text)
        self.assertIn("avoided dense rowwise Kronecker", text)
        self.assertLess(report.largest_single_allocation_bytes, 400000 * 600 * 8)

    def test_survival_prediction_preflight_chunks_large_horizon_grid(self) -> None:
        report = _RUNNER.preflight_survival_prediction(
            n_rows=400000,
            grid_points=1000,
        )
        self.assertEqual(report.status, "PASS")
        self.assertEqual(report.chunk_rows, _RUNNER.LARGE_SCALE_SURVIVAL_PREDICTION_CHUNK_ROWS)
        self.assertLess(report.largest_single_allocation_bytes, 400000 * 1000 * 8)

    def test_marginal_slope_preflight_status_is_grep_friendly(self) -> None:
        report = _RUNNER.preflight_marginal_slope_large_scale(
            n_train=400000,
            d_pc=16,
            centers=20,
            linkwiggle_knots=8,
            scorewarp_knots=7,
        )
        text = "\n".join(report.lines)
        self.assertIn("status: PASS", text)

    def test_run_method_subparser_exposes_emit_routing_log_flag(self) -> None:
        parser = _RUNNER.build_parser()
        args = parser.parse_args(
            [
                "run-method",
                "--prep-dir",
                "/tmp/p",
                "--method",
                "x",
                "--out-dir",
                "/tmp/o",
                "--out-json",
                "/tmp/o.json",
                "--emit-routing-log",
            ]
        )
        self.assertTrue(getattr(args, "emit_routing_log", False))

    def test_prepare_ctn_subparser_requires_explicit_artifact_boundaries(self) -> None:
        parser = _RUNNER.build_parser()
        args = parser.parse_args(
            [
                "prepare-ctn",
                "--prep-dir",
                "/tmp/raw-prepared",
                "--out-dir",
                "/tmp/ctn-prepared",
            ]
        )
        self.assertEqual(args.prep_dir, Path("/tmp/raw-prepared"))
        self.assertEqual(args.out_dir, Path("/tmp/ctn-prepared"))

    def test_default_marginal_slope_lanes_share_one_ctn_contract(self) -> None:
        cfg = _RUNNER.load_config(_RUNNER.DEFAULT_CONFIG)
        shared = _RUNNER.shared_ctn_spec(cfg)
        consumers = [spec for spec in _RUNNER.build_method_specs(cfg) if spec.marginal_slope]
        self.assertEqual(len(consumers), 3)
        self.assertEqual(shared.pc_count, 16)
        self.assertEqual(shared.centers, 24)
        self.assertEqual(shared.z_column, "pgs_ctn_z")
        self.assertEqual(
            {(spec.pc_count, spec.centers, spec.z_column) for spec in consumers},
            {(16, 24, "pgs_ctn_z")},
        )

    def test_run_method_refuses_raw_cohort_without_shared_ctn_artifact(self) -> None:
        spec = _RUNNER.shared_ctn_spec(_RUNNER.load_config(_RUNNER.DEFAULT_CONFIG))
        with tempfile.TemporaryDirectory() as raw_dir:
            root = Path(raw_dir)
            train = root / "train.csv"
            heldout = root / "heldout.csv"
            _write_csv(train, [{"pgs_raw": 0.1, "pc1_std": 0.0}])
            _write_csv(heldout, [{"pgs_raw": -0.1, "pc1_std": 0.2}])
            with self.assertRaisesRegex(RuntimeError, "shared CTN preprocessing artifact"):
                _RUNNER.require_shared_ctn_columns(spec, train, heldout)

    def test_add_standardized_columns_returns_replayable_training_statistics(self) -> None:
        train_rows: list[dict[str, object]] = []
        test_rows: list[dict[str, object]] = []
        for target, base in ((train_rows, 10.0), (test_rows, 20.0)):
            for idx in range(3):
                row: dict[str, object] = {
                    "age_entry": base + idx,
                    "lat_final": base + idx + 1.0,
                    "lon_final": base + idx + 2.0,
                    "pgs_raw": base + idx + 3.0,
                }
                row.update({f"pc{i}": base + idx + i for i in range(1, 17)})
                target.append(row)

        standardization = _RUNNER.add_standardized_columns(train_rows, test_rows)

        expected_columns = {
            "age_entry",
            "lat_final",
            "lon_final",
            "pgs_raw",
            *[f"pc{i}" for i in range(1, 17)],
        }
        self.assertEqual(set(standardization), expected_columns)
        self.assertAlmostEqual(standardization["age_entry"]["mean"], 11.0)
        self.assertGreater(standardization["age_entry"]["sd"], 0.0)
        self.assertIn("pgs_std", train_rows[0])
        self.assertIn("pc16_std", test_rows[0])

    def test_routing_log_scraper_captures_outer_plan_lines_only(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            tmp = Path(raw_dir) / "lane.routing.log"
            stderr = (
                "[HEARTBEAT] elapsed=1.2s cmd='gam ...' pid=42 cpu=10% mem=2%\n"
                "[OUTER] reml outer: n_params=6, gradient=Analytic, hessian=Analytic"
                " -> solver=Arc, hessian_source=Analytic"
                " [solver=Arc;hessian=Analytic;matrix-free=true]\n"
                "some unrelated stderr noise mentioning solver=Cheese\n"
                "[OUTER] aux outer: n_params=2, gradient=Analytic, hessian=Unavailable"
                " -> solver=Bfgs, hessian_source=BfgsApprox"
                " [solver=Bfgs;hessian=BfgsApprox;matrix-free=false] [no Hessian: BFGS approximation]\n"
            )
            _RUNNER._append_routing_lines(tmp, stderr)
            captured = tmp.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(captured), 2, captured)
            self.assertIn(
                "solver=Arc;hessian=Analytic;matrix-free=true", captured[0]
            )
            self.assertIn(
                "solver=Bfgs;hessian=BfgsApprox;matrix-free=false", captured[1]
            )
            self.assertNotIn("HEARTBEAT", "\n".join(captured))
            self.assertNotIn("Cheese", "\n".join(captured))

    def test_build_method_specs_rejects_pc_count_above_generated_columns(self) -> None:
        cfg = {
            "methods": [
                {
                    "name": "too_many_pcs",
                    "dataset": "disease",
                    "backend": "rust_gam",
                    "family": "binomial",
                    "spatial_basis": "duchon",
                    "pc_count": 17,
                }
            ]
        }
        with self.assertRaisesRegex(RuntimeError, "pc_count in \\[1, 16\\]"):
            _RUNNER.build_method_specs(cfg)

    def test_marginal_slope_formula_supports_matern_basis(self) -> None:
        spec = _RUNNER.MethodSpec(
            name="margslope_matern",
            dataset="disease",
            backend="rust_gam",
            family="binomial",
            spatial_basis="matern",
            marginal_slope=True,
            pc_count=4,
        )
        mean_formula, logslope_formula = _RUNNER.rust_marginal_slope_formula_classification(
            spec,
            centers=18,
        )
        self.assertIn("matern(pc1_std, pc2_std, pc3_std, pc4_std, centers=18)", mean_formula)
        self.assertIn("smooth(age_entry_std)", logslope_formula)

    def _survival_contract_train_rows(self) -> list[dict[str, float]]:
        return [
            {
                "time": 4.0,
                "event": 1.0,
                "pgs_std": 0.1,
                "sex": 0.0,
                "age_entry_std": -1.0,
                "lat_final_std": 0.2,
                "lon_final_std": -0.3,
                "pc1_std": 0.1,
                "pc2_std": 0.2,
                "pc3_std": 0.3,
                "pc4_std": 0.4,
            },
            {
                "time": 10.0,
                "event": 0.0,
                "pgs_std": -0.2,
                "sex": 1.0,
                "age_entry_std": 0.5,
                "lat_final_std": -0.1,
                "lon_final_std": 0.6,
                "pc1_std": -0.1,
                "pc2_std": -0.2,
                "pc3_std": -0.3,
                "pc4_std": -0.4,
            },
        ]

    def _survival_contract_test_rows(self) -> list[dict[str, float]]:
        return [
            {
                "time": 6.0,
                "event": 1.0,
                "pgs_std": 0.4,
                "sex": 1.0,
                "age_entry_std": -0.4,
                "lat_final_std": 0.8,
                "lon_final_std": 0.1,
                "pc1_std": 0.5,
                "pc2_std": 0.4,
                "pc3_std": 0.3,
                "pc4_std": 0.2,
            }
        ]

    def test_run_rust_survival_uses_explicit_survival_contract(self) -> None:
        spec = _RUNNER.MethodSpec(
            name="rust_gamlss_survival_ps",
            dataset="survival",
            backend="rust_survival",
            family="survival",
            spatial_basis="duchon",
            centers=24,
            survival_likelihood="location-scale",
            survival_distribution="probit",
        )
        train_rows = self._survival_contract_train_rows()
        test_rows = self._survival_contract_test_rows()
        snapshots: dict[str, typing.Any] = {}
        orig_load_bin = _RUNNER.load_or_build_rust_binary
        orig_run_cmd = _RUNNER.run_cmd_stream
        try:
            _RUNNER.load_or_build_rust_binary = lambda: Path("/tmp/fake-gam")

            def _fake_run_cmd(cmd: typing.Any, cwd: typing.Any = None) -> typing.Any:
                self.assertIsNotNone(cwd)
                if cmd[1] == "fit":
                    fit_input = Path(cmd[-2])
                    snapshots["fit_formula"] = cmd[-1]
                    snapshots["fit_cmd"] = list(cmd)
                    snapshots["fit_rows"] = _RUNNER.read_csv_rows(fit_input)
                    Path(cmd[cmd.index("--out") + 1]).write_text("{}", encoding="utf-8")
                    return 0, "", ""
                if cmd[1] == "predict":
                    input_path = Path(cmd[3])
                    out_path = Path(cmd[cmd.index("--out") + 1])
                    input_rows = _RUNNER.read_csv_rows(input_path)
                    snapshots.setdefault("predict_inputs", []).append(input_rows)
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    n = max(len(input_rows), 1)
                    with out_path.open("w", encoding="utf-8", newline="") as fh:
                        writer = csv.DictWriter(fh, fieldnames=["survival_prob"])
                        writer.writeheader()
                        for idx in range(len(input_rows)):
                            writer.writerow(
                                {"survival_prob": float(0.99 - 0.05 * idx / max(n - 1, 1))}
                            )
                    return 0, "", ""
                raise AssertionError(f"unexpected command: {cmd}")

            _RUNNER.run_cmd_stream = _fake_run_cmd

            with tempfile.TemporaryDirectory() as td:
                td_path = Path(td)
                train_csv = td_path / "train.csv"
                test_csv = td_path / "test.csv"
                _write_csv(train_csv, train_rows)
                _write_csv(test_csv, test_rows)
                result = _RUNNER.run_rust_survival(spec, train_csv, test_csv, td_path)
        finally:
            _RUNNER.load_or_build_rust_binary = orig_load_bin
            _RUNNER.run_cmd_stream = orig_run_cmd

        self.assertIn("Surv(__entry, time, event)", snapshots["fit_formula"])
        self.assertIn(
            "survmodel(spec=net, distribution=probit)",
            snapshots["fit_formula"],
        )
        self.assertIn("survival-likelihood=location-scale", result["model_spec"])
        fit_rows = snapshots["fit_rows"]
        self.assertTrue(all(float(row["__entry"]) == 0.0 for row in fit_rows))
        self.assertEqual([float(row["time"]) for row in fit_rows], [4.0, 10.0])

        predict_inputs = snapshots["predict_inputs"]
        self.assertEqual(len(predict_inputs), 2)

        horizon = _RUNNER.survival_eval_horizon_from_rows(train_rows)
        self.assertEqual(len(predict_inputs[0]), len(test_rows))
        self.assertTrue(
            all(abs(float(row["time"]) - horizon) < 1e-12 for row in predict_inputs[0])
        )

        import numpy as np

        grid = _RUNNER._survival_score_grid(
            np.array([float(r["time"]) for r in train_rows], dtype=float)
        )
        expected_native_rows = len(test_rows) * grid.shape[0]
        self.assertEqual(
            len(predict_inputs[1]),
            expected_native_rows,
            f"native survival grid must stack {len(test_rows)} test rows × "
            f"{grid.shape[0]} grid points = {expected_native_rows}; "
            f"got {len(predict_inputs[1])}",
        )

        for invocation_idx, rows in enumerate(predict_inputs):
            for row_idx, row in enumerate(rows):
                self.assertIn(
                    "__entry",
                    row,
                    f"predict invocation {invocation_idx} row {row_idx} missing __entry",
                )
                self.assertEqual(
                    float(row["__entry"]),
                    0.0,
                    f"predict invocation {invocation_idx} row {row_idx} has non-zero __entry",
                )

        fit_cmd = snapshots["fit_cmd"]
        self.assertIn("--survival-likelihood", fit_cmd)
        self.assertEqual(
            fit_cmd[fit_cmd.index("--survival-likelihood") + 1],
            "location-scale",
        )

    def test_run_rust_survival_rejects_invalid_native_grid_columns(self) -> None:
        spec = _RUNNER.MethodSpec(
            name="rust_gamlss_survival_ps",
            dataset="survival",
            backend="rust_survival",
            family="survival",
            spatial_basis="duchon",
            centers=24,
            survival_likelihood="location-scale",
            survival_distribution="probit",
        )
        train_rows = self._survival_contract_train_rows()
        test_rows = self._survival_contract_test_rows()

        orig_load_bin = _RUNNER.load_or_build_rust_binary
        orig_run_cmd = _RUNNER.run_cmd_stream
        try:
            _RUNNER.load_or_build_rust_binary = lambda: Path("/tmp/fake-gam")

            def _fake_run_cmd(cmd: typing.Any, cwd: typing.Any = None) -> typing.Any:
                self.assertIsNotNone(cwd)
                if cmd[1] == "fit":
                    Path(cmd[cmd.index("--out") + 1]).write_text("{}", encoding="utf-8")
                    return 0, "", ""
                if cmd[1] == "predict":
                    input_path = Path(cmd[3])
                    out_path = Path(cmd[cmd.index("--out") + 1])
                    input_rows = _RUNNER.read_csv_rows(input_path)
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    with out_path.open("w", encoding="utf-8", newline="") as fh:
                        writer = csv.DictWriter(fh, fieldnames=["risk_score"])
                        writer.writeheader()
                        for idx in range(len(input_rows)):
                            writer.writerow({"risk_score": float(idx)})
                    return 0, "", ""
                raise AssertionError(f"unexpected command: {cmd}")

            _RUNNER.run_cmd_stream = _fake_run_cmd

            with tempfile.TemporaryDirectory() as td:
                td_path = Path(td)
                train_csv = td_path / "train.csv"
                test_csv = td_path / "test.csv"
                _write_csv(train_csv, train_rows)
                _write_csv(test_csv, test_rows)
                with self.assertRaises(RuntimeError) as ctx:
                    _RUNNER.run_rust_survival(spec, train_csv, test_csv, td_path)
                self.assertIn(spec.name, str(ctx.exception))
        finally:
            _RUNNER.load_or_build_rust_binary = orig_load_bin
            _RUNNER.run_cmd_stream = orig_run_cmd

    def test_survival_formula_rhs_supports_linkwiggle_and_timewiggle(self) -> None:
        spec = _RUNNER.MethodSpec(
            name="surv_variant",
            dataset="survival",
            backend="rust_survival",
            family="survival",
            spatial_basis="duchon",
            centers=24,
            survival_likelihood="transformation",
            survival_distribution="gaussian",
            mean_linkwiggle_knots=8,
            timewiggle_knots=8,
        )
        rhs = _RUNNER.rust_survival_formula_rhs(spec)
        self.assertIn("linkwiggle(internal_knots=8)", rhs)
        self.assertIn("timewiggle(internal_knots=8)", rhs)

    def test_generate_raw_cohort_populates_pc_columns_from_each_row(self) -> None:
        cfg = {
            "seed": 1,
            "raw_subpop_n": 20,
            "observed_latlon_fraction": 0.5,
            "split_seed": 2,
            "target_n": 100,
            "smoke_target_n": 50,
        }
        with tempfile.TemporaryDirectory() as td:
            rows = _RUNNER.generate_raw_cohort(cfg, Path(td), smoke=False)[0]
        self.assertGreater(len(rows), 20)
        pc1 = [float(r["pc1"]) for r in rows[:30]]
        pc2 = [float(r["pc2"]) for r in rows[:30]]
        self.assertGreater(len(set(round(v, 6) for v in pc1)), 10)
        self.assertGreater(len(set(round(v, 6) for v in pc2)), 10)


class MarkerPatternTests(unittest.TestCase):


    def test_bfgs_summary_pattern_covers_all_outcome_variants(self) -> None:
        cases = [
            (
                "[OUTER summary] BFGS converged in 12 iters elapsed=145.234s "
                "final_value=1.23e3",
                "converged", "12", "145.234",
            ),
            (
                "[OUTER summary] BFGS hit max_iter in 100 iters elapsed=2398.0s "
                "final_value=1.23e3",
                "hit max_iter", "100", "2398.0",
            ),
            (
                "[OUTER summary] BFGS line-search failed in 47 iters "
                "elapsed=87.654s final_value=1.23e3",
                "line-search failed", "47", "87.654",
            ),
            (
                "[OUTER summary] BFGS failed elapsed=12.0s err=SomeErr",
                "failed", None, "12.0",
            ),
        ]
        for line, expected_status, expected_iters, expected_elapsed in cases:
            matches = _RUNNER._BFGS_SUMMARY_PATTERN.findall(line)
            self.assertEqual(
                len(matches),
                1,
                f"BFGS outcome {expected_status!r} did not parse: {line}",
            )
            status, iters, elapsed = matches[0]
            self.assertEqual(status, expected_status)
            if expected_iters is None:
                self.assertIn(iters, ("", None))
            else:
                self.assertEqual(iters, expected_iters)
            self.assertEqual(elapsed, expected_elapsed)


class PhaseSummaryAggregationTests(unittest.TestCase):
    def _run_summary(self, captured_stderr: str) -> str:
        import io
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            _RUNNER._emit_phase_summary(captured_stderr, "cmd-preview", timed_out=False, rc=0)
        return buf.getvalue()


    def test_phase_summary_tolerates_off_by_one_tangent_marker_drift(self) -> None:
        stderr = "\n".join([
            "[TANGENT-PREDICT] alpha=1.000e+00 cap=1.500e+00 drho_step_norm_sq=2.0e-02 drho_prev_norm_sq=2.0e-02",
            "[TANGENT-PREDICT] alpha=1.100e+00 cap=1.500e+00 drho_step_norm_sq=2.0e-02 drho_prev_norm_sq=2.0e-02",
            "[TANGENT-QUALITY] residual=1.0e-03 converged_norm=1.0 predicted_norm=1.0 iters=4",
            "[PHASE] my-fit fit end elapsed=10.0s",
        ])
        out = self._run_summary(stderr)
        self.assertNotIn("tangent_marker_drift", out)


    def test_phase_summary_omits_outer_nonfinite_when_count_is_zero(self) -> None:
        stderr = "[PHASE] my-fit fit end elapsed=10.0s\n"
        out = self._run_summary(stderr)
        self.assertNotIn("outer_nonfinite", out)


class DefaultBenchmarkConfigTests(unittest.TestCase):
    """RED tests for issue #221.

    The default config ships methods that the runner's spec validator must
    accept. Today `r_mgcv_jointpc_duchon60` is a `disease` method with
    `backend: r_mgcv`, but `validate_method_spec` rejects every disease
    backend except `rust_gam`, so the default config cannot be loaded
    end-to-end.
    """

    def test_default_config_loads_and_builds_specs(self) -> None:
        cfg = _RUNNER.load_config(_RUNNER.DEFAULT_CONFIG)
        specs = _RUNNER.build_method_specs(cfg)
        self.assertGreater(len(specs), 0, "default config must define at least one method")

    def test_every_default_method_passes_spec_validation(self) -> None:
        cfg = _RUNNER.load_config(_RUNNER.DEFAULT_CONFIG)
        raw_methods = list(cfg.get("methods", []))
        self.assertGreater(len(raw_methods), 0)
        failures: list[str] = []
        for entry in raw_methods:
            name = entry.get("name", "<unnamed>")
            try:
                spec = _RUNNER.MethodSpec(**entry)
                _RUNNER.validate_method_spec(spec)
            except Exception as exc:
                failures.append(f"{name}: {exc}")
        self.assertEqual(failures, [], f"default-config methods failed validation: {failures}")


if __name__ == "__main__":
    unittest.main()
