"""A categorical covariate is coded once, by the native cohort encoder, when the
model is fitted and again when a forecast names one of its levels."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import gamfit


def _simulate_two_groups(n: int, follow_up: float, seed: int):
    """Thinning simulation of a single-mark cohort whose log-intensity is -0.4
    in group "low" and 0.6 in group "high", with no latent state."""
    rng = np.random.default_rng(seed)
    group = np.where(rng.uniform(size=n) < 0.5, "low", "high")
    log_rate = {"low": -0.4, "high": 0.6}
    bound = np.exp(max(log_rate.values()))
    ids, times = [], []
    for i in range(n):
        t = 0.0
        while True:
            t -= np.log(rng.uniform()) / bound
            if t >= follow_up:
                break
            if rng.uniform() * bound < np.exp(log_rate[group[i]]):
                ids.append(f"s{i}")
                times.append(t)
    subjects = pd.DataFrame({"id": [f"s{i}" for i in range(n)], "entry": 0.0, "exit": follow_up})
    events = pd.DataFrame({"id": ids, "time": times, "mark": "event"})
    covariates = pd.DataFrame({"id": [f"s{i}" for i in range(n)], "start": 0.0, "group": group})
    return subjects, events, covariates


def test_a_string_covariate_is_a_categorical_with_sorted_levels_that_forecasts_accept() -> None:
    subjects, events, covariates = _simulate_two_groups(240, 4.0, seed=41)
    model = gamfit.fit_event_history(subjects, events, covariates, "group")
    assert model.covariate_levels == {"group": ["high", "low"]}
    high = model.population_forecast({"group": "high"}, start=1.0, horizons=[1.5])
    low = model.population_forecast({"group": "low"}, start=1.0, horizons=[1.5])
    high_count = float(high["expected_counts"][0, 0])
    low_count = float(low["expected_counts"][0, 0])
    assert np.isfinite(high_count) and np.isfinite(low_count)
    # The planted log-rate gap is one; the fitted forecasts must order the groups.
    assert high_count > low_count
    with pytest.raises(ValueError, match="unknown level"):
        model.population_forecast({"group": "medium"}, start=1.0, horizons=[1.5])
