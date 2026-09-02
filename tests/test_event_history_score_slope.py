"""An observed risk score enters an event-history model as one penalised
slope surface, the forecast tiers are one model conditioned on more, and a
score calibrated by a conditional transformation-normal Stage 1 composes in
front of the fit. All through the public Python API (``docs/event-history.md``).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import gamfit


def _simulate(n: int, follow_up: float, intercept: float, slope, bound_slope: float, seed: int,
              score=None):
    """Thinning simulation of a single-mark cohort with log-intensity
    ``intercept + slope(t) * g`` for a subject-level score ``g`` (standard
    normal unless given) and no latent state."""
    rng = np.random.default_rng(seed)
    g = rng.standard_normal(n) if score is None else np.asarray(score, dtype=float)
    ids, times = [], []
    for i in range(n):
        bound = np.exp(intercept + bound_slope * abs(g[i]))
        t = 0.0
        while True:
            t -= np.log(rng.uniform()) / bound
            if t >= follow_up:
                break
            if rng.uniform() * bound < np.exp(intercept + slope(t) * g[i]):
                ids.append(f"s{i}")
                times.append(t)
    subjects = pd.DataFrame({"id": [f"s{i}" for i in range(n)], "entry": 0.0, "exit": follow_up})
    events = pd.DataFrame({"id": ids, "time": times, "mark": "event"})
    covariates = pd.DataFrame({"id": [f"s{i}" for i in range(n)], "start": 0.0, "g": g})
    return subjects, events, covariates


def _slope_probe(model, times, width: float = 0.05):
    """``b(t)`` read through the public forecast: the log ratio of the
    expected count over a short window at score one and score zero."""
    out = []
    for t in times:
        one = model.population_forecast({"g": 1.0}, start=t, horizons=[t + width])
        zero = model.population_forecast({"g": 0.0}, start=t, horizons=[t + width])
        out.append(float(np.log(one["expected_counts"][0, 0] / zero["expected_counts"][0, 0])))
    return np.asarray(out)


def test_a_score_enters_as_one_penalised_slope_surface() -> None:
    truth = lambda t: 1.0 - 0.15 * t  # noqa: E731
    subjects, events, covariates = _simulate(300, 6.0, -0.5, truth, 1.0, seed=19)
    model = gamfit.fit_event_history(subjects, events, covariates, "s(time, by=g)")
    assert model.rank == 0, f"a cohort with no latent structure must stay at rank 0, got {model.rank}"
    assert model.covariate_names == ["g"]
    times = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
    fitted = _slope_probe(model, times)
    assert fitted[0] - fitted[-1] > 0.35, f"the effect declines by 0.75; fitted {fitted}"
    np.testing.assert_allclose(fitted, truth(times), atol=0.3)

    # The same surface on a score that carries nothing collapses.
    subjects, events, covariates = _simulate(300, 6.0, -0.5, lambda t: 0.0, 0.0, seed=23)
    null = gamfit.fit_event_history(subjects, events, covariates, "s(time, by=g)")
    amplitude = float(np.max(np.abs(_slope_probe(null, times))))
    assert amplitude < 0.15, f"an uninformative score must collapse; fitted amplitude {amplitude}"


def test_forecast_tiers_are_one_model_conditioned_on_more() -> None:
    subjects, events, covariates = _simulate(300, 6.0, -0.5, lambda t: 0.6, 0.6, seed=5)
    model = gamfit.fit_event_history(subjects, events, covariates, "s(time, by=g)")
    horizons = [7.0, 8.0]
    population = model.population_forecast({"g": 0.0}, start=6.0, horizons=horizons)
    high = model.population_forecast([1.0], start=6.0, horizons=horizons)
    low = model.population_forecast([-1.0], start=6.0, horizons=horizons)
    # No absorbing marks: survival is one; a positive effect orders the tiers.
    np.testing.assert_allclose(population["survival"], 1.0, atol=1e-12)
    assert high["expected_counts"][1, 0] > population["expected_counts"][1, 0] > low["expected_counts"][1, 0]
    # With no latent state the history adds nothing: a subject's own forecast
    # is its score-only tier, to roundoff.
    own = model.forecast("s0", horizons=horizons)
    alone = model.population_forecast({"g": float(covariates.loc[0, "g"])}, start=6.0, horizons=horizons)
    np.testing.assert_allclose(own["expected_counts"], alone["expected_counts"], rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(own["survival"], alone["survival"], rtol=1e-12)


def test_a_ctn_calibrated_score_composes_in_front_of_the_fit() -> None:
    # A raw score confounded by ancestry: prs = 0.8 * pc1 + liability. The
    # disease is driven by the liability alone, with a constant effect 0.8.
    rng = np.random.default_rng(11)
    n = 300
    pc1 = rng.uniform(-1.0, 1.0, n)
    liability = rng.standard_normal(n)
    prs = 0.8 * pc1 + liability
    stage1 = pd.DataFrame({"prs": prs, "pc1": pc1})
    calib = gamfit.fit(stage1, "prs ~ s(pc1)", transformation_normal=True)
    z = np.asarray(calib.transformation_score(stage1), dtype=float)
    assert abs(z.mean()) < 0.15 and abs(z.std() - 1.0) < 0.15
    assert abs(np.corrcoef(prs, pc1)[0, 1]) > 0.35, "the raw score is confounded by construction"
    assert abs(np.corrcoef(z, pc1)[0, 1]) < 0.15, "the calibrated score is conditioned on ancestry"
    assert np.corrcoef(z, liability)[0, 1] > 0.85

    subjects, events, covariates = _simulate(n, 6.0, -0.5, lambda t: 0.8, 0.8, seed=29, score=liability)
    covariates = covariates.assign(g=z)
    model = gamfit.fit_event_history(subjects, events, covariates, "s(time, by=g)")
    fitted = _slope_probe(model, np.array([1.0, 3.0, 5.0]))
    np.testing.assert_allclose(fitted, 0.8, atol=0.3)
    assert float(np.ptp(fitted)) < 0.4, f"a constant effect must not bend: {fitted}"
