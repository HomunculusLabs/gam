"""#2411: the fitted-model summary must carry the optimizer's own convergence
certificate.

The library's contract is that returning a fit *is* the convergence verdict:
a non-certified optimization raises instead of minting a model. That contract
is enforced by a sealed ``FitConvergenceEvidence`` the fit owns, which the Rust
and CLI surfaces already read. The Python summary did not expose it, so a
caller wanting to gate on stationarity — for instance to impose a tolerance
stricter than the library's own data-scaled bound — had no field to read and
fell back to parsing the log stream, where routine scheduled events are easy to
mistake for fit verdicts (#2410).

What is asserted here:

* the certificate reaches Python at all, on a model with smoothing parameters;
* it is internally consistent — ``certified`` agrees with the projected
  residual actually clearing the bound it is reported against, so a caller can
  apply their own threshold to the same two numbers;
* it survives save/load, which is where the gap would otherwise reappear;
* the "no smoothing coordinate was optimized" case stays distinguishable from
  "the projected gradient was zero" rather than being flattened into it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import gamfit

_STATIONARITY_KINDS = {"analytic_gradient", "fixed_point", "asymptote_rail"}


def _smooth_frame() -> pd.DataFrame:
    rng = np.random.default_rng(2411)
    n = 400
    x = rng.uniform(0.0, 1.0, n)
    y = np.sin(2.0 * np.pi * x) + 0.2 * rng.standard_normal(n)
    return pd.DataFrame({"x": x, "y": y})


def _check_outer_block(outer: dict) -> None:
    """A reported outer certificate must be self-describing and checkable."""
    assert outer["kind"] in _STATIONARITY_KINDS, outer["kind"]

    grad = float(outer["gradient_norm"])
    projected = float(outer["projected_gradient_norm"])
    bound = float(outer["stationarity_bound"])
    assert np.isfinite(grad) and grad >= 0.0, grad
    assert np.isfinite(projected) and projected >= 0.0, projected
    assert np.isfinite(bound) and bound >= 0.0, bound

    # Projection removes components; it cannot manufacture them.
    assert projected <= grad + 1e-12 * max(1.0, grad), (projected, grad)

    railed = outer["lambdas_railed"]
    assert isinstance(railed, list), type(railed)
    assert all(isinstance(index, int) and index >= 0 for index in railed), railed
    assert len(set(railed)) == len(railed), railed

    psd = outer.get("hessian_psd")
    assert psd is None or isinstance(psd, bool), psd


def test_summary_exposes_the_outer_convergence_certificate() -> None:
    """A penalized fit reports the outer stationarity certificate it was minted on."""
    model = gamfit.fit(_smooth_frame(), "y ~ s(x)")
    convergence = model.summary().convergence

    assert convergence is not None, "penalized fit reported no convergence certificate"
    assert convergence["certified"] is True
    assert isinstance(convergence["inner_status"], str)
    assert convergence["inner_status"] != ""
    assert int(convergence["outer_iterations"]) >= 0

    outer = convergence["outer"]
    assert outer is not None, "a fit with smoothing parameters must carry an outer certificate"
    _check_outer_block(outer)


def test_certified_verdict_agrees_with_the_reported_bound() -> None:
    """``certified`` must be checkable from the numbers reported alongside it.

    This is the property that makes the field usable as an API: a caller can
    re-derive the verdict, and therefore can also impose a stricter tolerance
    of their own, without trusting a boolean they cannot audit.
    """
    convergence = gamfit.fit(_smooth_frame(), "y ~ s(x)").summary().convergence
    outer = convergence["outer"]

    projected = float(outer["projected_gradient_norm"])
    bound = float(outer["stationarity_bound"])
    assert projected <= bound, (projected, bound)
    assert convergence["certified"] is True

    # The same two numbers support a stricter user gate; assert only that the
    # comparison is well-posed, not that this particular fit passes it.
    strict = projected <= 1e-3
    assert strict in (True, False)


def test_certificate_survives_save_and_load(tmp_path) -> None:
    """The certificate must round-trip; a reloaded model is where the gap bit."""
    model = gamfit.fit(_smooth_frame(), "y ~ s(x)")
    before = model.summary().convergence

    path = tmp_path / "certificate_roundtrip.gam"
    model.save(path)
    after = gamfit.load(path).summary().convergence

    assert after is not None
    assert after["certified"] == before["certified"]
    assert after["inner_status"] == before["inner_status"]
    assert after["outer_iterations"] == before["outer_iterations"]

    outer_before, outer_after = before["outer"], after["outer"]
    assert outer_after["kind"] == outer_before["kind"]
    assert outer_after["lambdas_railed"] == outer_before["lambdas_railed"]
    assert outer_after["hessian_psd"] == outer_before["hessian_psd"]
    for key in ("gradient_norm", "projected_gradient_norm", "stationarity_bound"):
        assert float(outer_after[key]) == float(outer_before[key]), key


def test_unpenalized_fit_reports_no_outer_equation_rather_than_a_zero_gradient() -> None:
    """With no smoothing coordinate there is no outer stationarity equation.

    Reporting that as a projected gradient of 0.0 against a bound of 0.0 would
    read as "certified at machine precision", which is a different and much
    stronger claim than "there was nothing to certify".
    """
    rng = np.random.default_rng(24110)
    n = 300
    x = rng.uniform(0.0, 1.0, n)
    y = 1.0 + 0.5 * x + 0.1 * rng.standard_normal(n)
    model = gamfit.fit(pd.DataFrame({"x": x, "y": y}), "y ~ x")

    convergence = model.summary().convergence
    assert convergence is not None
    assert convergence["certified"] is True
    assert convergence["outer"] is None, convergence["outer"]
    assert int(convergence["outer_iterations"]) == 0
