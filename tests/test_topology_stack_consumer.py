"""Unit tests for the topology stacking consumer (#768).

These exercise the Python-side held-out density-table assembly, the binding
contract, and the stacked-mean combination with stubbed candidate fits and a
stubbed Rust binding, so they run without the compiled extension.
"""

import json
import math

from statistics import NormalDist

import gamfit._select_topology as st


class _StubFit:
    """Candidate fit whose predict() returns deterministic per-point moments.

    `mean_fn(x)` gives the response-scale mean; the observation band is the
    symmetric Gaussian band mean ± z·sd used by the real Rust predictor, so the
    consumer recovers exactly `sd` back out.
    """

    def __init__(self, mean_fn, sd):
        self._mean_fn = mean_fn
        self._sd = sd

    def predict(self, data, **kwargs):
        xs = data["x"]
        means = [self._mean_fn(x) for x in xs]
        if kwargs.get("observation_interval"):
            z = NormalDist().inv_cdf(0.5 + 0.5 * kwargs["interval"])
            lower = [m - z * self._sd for m in means]
            upper = [m + z * self._sd for m in means]
            return {
                "mean": means,
                "observation_lower": lower,
                "observation_upper": upper,
            }
        return {"mean": means}


class _CapturingRust:
    """Stub binding that records the log-density table and returns fixed weights."""

    def __init__(self, weights):
        self._weights = weights
        self.captured_names = None
        self.captured_rows = None
        self.captured_y = None
        self.captured_means = None
        self.captured_lowers = None
        self.captured_uppers = None
        self.captured_interval_level = None

    def stacking_weights_from_log_density(self, names, log_density_rows):
        self.captured_names = list(names)
        self.captured_rows = [list(row) for row in log_density_rows]
        return self._stack_result()

    def stack_topologies_gaussian(
        self,
        names,
        y,
        means,
        lowers,
        uppers,
        interval_level,
    ):
        """The second binding `_select_topology` reaches for on this protocol.

        `_TopologyRust` declares both `stacking_weights_from_log_density` and
        `stack_topologies_gaussian`, and `stack_topologies` calls the latter.
        This stub implemented only the first, so every test that monkeypatches
        `_topology_rust` to it failed with

            AttributeError: '_CapturingRust' object has no attribute
                            'stack_topologies_gaussian'

        before reaching its own assertions — a test double is a consumer of the
        protocol like any other, and it was not enumerated when the protocol
        grew. Captured under distinct names so a test can assert which binding
        the consumer chose, which is the thing the missing method was hiding.
        """
        self.captured_names = list(names)
        self.captured_y = list(y)
        self.captured_means = [list(row) for row in means]
        self.captured_lowers = [list(row) for row in lowers]
        self.captured_uppers = [list(row) for row in uppers]
        self.captured_interval_level = float(interval_level)
        return self._stack_result()

    def _stack_result(self):
        return json.dumps(
            {
                "weights": dict(self._weights),
                "mean_log_score": -1.234,
                "iterations": 7,
            }
        )


def _gaussian_logpdf(y, mean, sd):
    z = (y - mean) / sd
    return -0.5 * math.log(2.0 * math.pi) - math.log(sd) - 0.5 * z * z


def test_holdout_log_density_table_matches_gaussian(monkeypatch):
    holdout = {"x": [0.0, 1.0, 2.0], "y": [0.1, 0.9, 2.2]}
    fits = {
        "flat": _StubFit(lambda x: 0.0, sd=1.0),
        "linear": _StubFit(lambda x: x, sd=0.5),
    }
    rust = _CapturingRust({"flat": 0.3, "linear": 0.7})
    monkeypatch.setattr(st, "_topology_rust", lambda: rust)

    stack = st.stack_topologies(fits, holdout, "y")

    # The table the binding received is the per-point Gaussian held-out
    # log-density of the true y under each candidate's recovered (mean, sd).
    # `stack_topologies` (gamfit/_select_topology.py:481) calls
    # `stack_topologies_gaussian`, NOT `stacking_weights_from_log_density`, so
    # `captured_rows` stays None and every read of it raised TypeError before
    # reaching an assertion. The Gaussian log-density table itself moved into
    # `gam::solver::topology_stack_gaussian`; what Python still owns -- and
    # what this test exists to pin -- is that the marshalled per-candidate
    # (mean, band) recovers each candidate's exact predictive moments, so the
    # table the kernel forms from them IS the held-out Gaussian log-density.
    assert rust.captured_names == ["flat", "linear"]
    y = holdout["y"]
    assert rust.captured_y == [float(v) for v in y]
    z = NormalDist().inv_cdf(0.5 + 0.5 * rust.captured_interval_level)
    for i, yi in enumerate(y):
        expected_flat = _gaussian_logpdf(yi, 0.0, 1.0)
        expected_linear = _gaussian_logpdf(yi, float(i), 0.5)
        sd_flat = (rust.captured_uppers[0][i] - rust.captured_lowers[0][i]) / (2.0 * z)
        sd_linear = (rust.captured_uppers[1][i] - rust.captured_lowers[1][i]) / (2.0 * z)
        got_flat = _gaussian_logpdf(yi, rust.captured_means[0][i], sd_flat)
        got_linear = _gaussian_logpdf(yi, rust.captured_means[1][i], sd_linear)
        assert math.isclose(got_flat, expected_flat, rel_tol=1e-9)
        assert math.isclose(got_linear, expected_linear, rel_tol=1e-9)

    assert stack.weights == {"flat": 0.3, "linear": 0.7}
    assert stack.mean_log_score == -1.234


def test_stacked_predict_is_weighted_mixture(monkeypatch):
    holdout = {"x": [0.0, 1.0], "y": [0.0, 1.0]}
    fits = {
        "flat": _StubFit(lambda x: 2.0, sd=1.0),
        "linear": _StubFit(lambda x: 10.0 * x, sd=1.0),
    }
    rust = _CapturingRust({"flat": 0.25, "linear": 0.75})
    monkeypatch.setattr(st, "_topology_rust", lambda: rust)

    stack = st.stack_topologies(fits, holdout, "y")
    out = stack.predict({"x": [0.0, 1.0, 2.0]})

    # flat predicts 2 everywhere; linear predicts 10*x. Mixture = 0.25*2 + 0.75*10*x.
    assert math.isclose(out[0], 0.25 * 2.0 + 0.75 * 0.0, rel_tol=1e-9)
    assert math.isclose(out[1], 0.25 * 2.0 + 0.75 * 10.0, rel_tol=1e-9)
    assert math.isclose(out[2], 0.25 * 2.0 + 0.75 * 20.0, rel_tol=1e-9)


def test_zero_weighted_candidate_is_not_predicted(monkeypatch):
    holdout = {"x": [0.0, 1.0], "y": [0.0, 1.0]}

    class _Exploding(_StubFit):
        def predict(self, data, **kwargs):
            if not kwargs.get("observation_interval"):
                raise AssertionError("zero-weighted candidate must not be predicted")
            return super().predict(data, **kwargs)

    fits = {
        "keep": _StubFit(lambda x: x, sd=1.0),
        "drop": _Exploding(lambda x: 99.0, sd=1.0),
    }
    rust = _CapturingRust({"keep": 1.0, "drop": 0.0})
    monkeypatch.setattr(st, "_topology_rust", lambda: rust)

    stack = st.stack_topologies(fits, holdout, "y")
    out = stack.predict({"x": [3.0, 4.0]})
    assert math.isclose(out[0], 3.0, rel_tol=1e-9)
    assert math.isclose(out[1], 4.0, rel_tol=1e-9)


def test_non_positive_sd_rows_are_dropped_from_a_candidate(monkeypatch):
    # A candidate whose observation band collapses (sd == 0, e.g. fully clamped
    # at the support) yields -inf log-density for those rows, not a crash.
    holdout = {"x": [0.0, 1.0], "y": [0.0, 1.0]}

    class _Degenerate(_StubFit):
        def predict(self, data, **kwargs):
            out = super().predict(data, **kwargs)
            if kwargs.get("observation_interval"):
                # Collapse the band on the first row.
                out["observation_lower"][0] = out["mean"][0]
                out["observation_upper"][0] = out["mean"][0]
            return out

    fits = {
        "good": _StubFit(lambda x: x, sd=1.0),
        "degen": _Degenerate(lambda x: x, sd=1.0),
    }
    rust = _CapturingRust({"good": 0.5, "degen": 0.5})
    monkeypatch.setattr(st, "_topology_rust", lambda: rust)

    st.stack_topologies(fits, holdout, "y")
    # `captured_rows` belongs to the binding the consumer no longer calls (see
    # the sibling test); the -inf substitution and the row drop now live in
    # `gam::solver::topology_stack_gaussian`. The Python seam's obligation is
    # to marshal the collapsed band THROUGH rather than crash on it or repair
    # it, which is exactly what "dropped from a candidate" depends on.
    degen = rust.captured_names.index("degen")
    good = rust.captured_names.index("good")
    assert rust.captured_uppers[degen][0] == rust.captured_lowers[degen][0]
    assert rust.captured_uppers[good][0] > rust.captured_lowers[good][0]
    assert math.isfinite(rust.captured_means[degen][0])
    assert math.isfinite(rust.captured_means[good][0])


def test_missing_response_column_is_rejected():
    fits = {"a": _StubFit(lambda x: x, sd=1.0)}
    try:
        st.stack_topologies(fits, {"x": [0.0]}, "y")
    except ValueError as exc:
        assert "response" in str(exc)
    else:
        raise AssertionError("expected ValueError for missing response column")
