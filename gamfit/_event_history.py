"""Event histories: marked counting processes with smooth covariate and time
effects and an evidence-selected per-subject latent state.

The fit is exact-marginal over the latent chain (adaptive Gauss-Hermite
filtering); forecasts and the predictive PIT are exact expectations under the
filtered state. See ``docs/event-history.md``.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from ._binding import rust_module


def _column(frame: Any, name: str) -> np.ndarray:
    if isinstance(frame, dict):
        if name not in frame:
            raise KeyError(f"missing column {name!r}")
        return np.asarray(frame[name])
    if hasattr(frame, "columns") and name not in list(frame.columns):
        raise KeyError(f"missing column {name!r}")
    column = frame[name]
    if hasattr(column, "to_numpy"):
        return column.to_numpy()
    return np.asarray(column)


def _column_names(frame: Any) -> list[str]:
    if isinstance(frame, dict):
        return list(frame.keys())
    return [str(c) for c in frame.columns]


class EventHistoryModel:
    """A fitted event-history model."""

    def __init__(self, native: Any) -> None:
        self._native = native
        self._subject_index = {sid: i for i, sid in enumerate(native.subject_ids())}

    @property
    def mark_names(self) -> list[str]:
        return list(self._native.mark_names())

    @property
    def covariate_names(self) -> list[str]:
        return list(self._native.covariate_names())

    @property
    def subject_ids(self) -> list[str]:
        return list(self._native.subject_ids())

    @property
    def atoms(self) -> int:
        return int(self._native.atoms())

    @property
    def loadings(self) -> np.ndarray:
        """Loadings, shape ``(marks, atoms)``."""
        return np.asarray(self._native.loadings())

    @property
    def rates(self) -> np.ndarray:
        """Atom rates in the data's time unit."""
        return np.asarray(self._native.rates())

    @property
    def atom_log_lambdas(self) -> np.ndarray:
        """REML log smoothing parameter of each atom's ridge."""
        return np.asarray(self._native.atom_log_lambdas())

    @property
    def log_likelihood(self) -> float:
        return float(self._native.log_likelihood())

    @property
    def reml_score(self) -> float | None:
        return self._native.reml_score()

    @property
    def quadrature(self) -> dict[str, Any]:
        return dict(self._native.quadrature())

    def coefficients(self, mark: int | str) -> np.ndarray:
        """Coefficients of one mark's population log-intensity surface: the
        latent term enters as the deviation from a population rate, so
        ``exp(η⁰)`` is the intensity averaged over the latent state."""
        if isinstance(mark, str):
            mark = self.mark_names.index(mark)
        return np.asarray(self._native.coefficients(int(mark)))

    def _absorbing_mask(self, absorbing: Sequence[str] | Sequence[bool] | None) -> list[bool]:
        marks = self.mark_names
        if absorbing is None:
            return [False] * len(marks)
        if all(isinstance(a, (bool, np.bool_)) for a in absorbing):
            return [bool(a) for a in absorbing]
        chosen = set(absorbing)
        return [name in chosen for name in marks]

    def _subject(self, subject: int | str) -> int:
        if isinstance(subject, str):
            return self._subject_index[subject]
        return int(subject)

    def forecast(
        self,
        subject: int | str,
        horizons: Sequence[float],
        absorbing: Sequence[str] | Sequence[bool] | None = None,
        future_row: int | None = None,
    ) -> dict[str, Any]:
        """Forecast one subject beyond its exit: ``survival`` is the
        probability that no absorbing mark has fired by each horizon and
        ``expected_counts`` (horizons × marks) is the expected count of each
        mark, its cumulative incidence when absorbing."""
        out = self._native.forecast(
            self._subject(subject),
            [float(h) for h in horizons],
            self._absorbing_mask(absorbing),
            future_row,
        )
        return {
            "horizons": np.asarray(out["horizons"]),
            "survival": np.asarray(out["survival"]),
            "expected_counts": np.asarray(out["expected_counts"]),
        }

    def population_forecast(
        self,
        covariates: Mapping[str, float] | Sequence[float],
        start: float,
        horizons: Sequence[float],
        absorbing: Sequence[str] | Sequence[bool] | None = None,
    ) -> dict[str, Any]:
        """Forecast a subject with no observed history from covariate values
        alone: the latent state starts at its stationary prior. Population
        covariate values give the population tier; a subject's own score gives
        what the model says before its history is seen. ``covariates`` is a
        mapping by covariate name or a sequence in ``covariate_names`` order;
        ``start`` is the time the window opens."""
        names = self.covariate_names
        if isinstance(covariates, Mapping):
            missing = [c for c in names if c not in covariates]
            if missing:
                raise KeyError(f"population_forecast is missing covariates {missing}")
            values = [float(covariates[c]) for c in names]
        else:
            values = [float(v) for v in covariates]
            if len(values) != len(names):
                raise ValueError(
                    f"population_forecast needs {len(names)} covariate values, got {len(values)}"
                )
        out = self._native.population_forecast(
            values, float(start), [float(h) for h in horizons], self._absorbing_mask(absorbing)
        )
        return {
            "horizons": np.asarray(out["horizons"]),
            "survival": np.asarray(out["survival"]),
            "expected_counts": np.asarray(out["expected_counts"]),
        }

    def pit(self, subject: int | str) -> np.ndarray:
        """Predictive PIT of every event of one subject; uniform under the model."""
        return np.asarray(self._native.pit(self._subject(subject)))

    def pit_ks(self) -> float:
        """Kolmogorov–Smirnov distance of all predictive PITs from uniform."""
        return float(self._native.pit_ks())


def fit_event_history(
    subjects: Any,
    events: Any,
    covariates: Any,
    formula: str,
    *,
    atoms: int = 1,
    id_column: str = "id",
) -> EventHistoryModel:
    """Fit an event-history model.

    ``subjects`` has columns ``id, entry, exit``; ``events`` has ``id, time,
    mark``; ``covariates`` has ``id, start`` and the covariate columns, one row
    per covariate segment (a subject's covariates are constant from ``start``
    until its next segment). ``formula`` is the right-hand side of a gam
    formula over the covariate columns and ``time``, e.g. ``"x + s(time)"``.
    An observed risk score enters as a penalised slope surface,
    ``"s(time, by=prs, identifiability=none)"``: its effect on every mark
    may bend with time as much as the evidence supports and collapses to
    zero when the score carries nothing. ``atoms`` is the maximum number of
    latent atoms; the evidence switches off the ones the data do not support.
    """
    rust = rust_module()
    subject_ids = [str(v) for v in _column(subjects, id_column)]
    index = {sid: i for i, sid in enumerate(subject_ids)}
    entry = _column(subjects, "entry").astype(float).tolist()
    exit_ = _column(subjects, "exit").astype(float).tolist()
    mark_values = [str(v) for v in _column(events, "mark")]
    mark_names = sorted(set(mark_values))
    mark_index = {name: i for i, name in enumerate(mark_names)}
    event_subject = [index[str(v)] for v in _column(events, id_column)]
    event_time = _column(events, "time").astype(float).tolist()
    event_mark = [mark_index[m] for m in mark_values]
    covariate_names = [
        c for c in _column_names(covariates) if c not in (id_column, "start")
    ]
    if not covariate_names:
        raise ValueError("covariates needs at least one covariate column besides id and start")
    table = np.column_stack(
        [_column(covariates, c).astype(float) for c in covariate_names]
    )
    segment_subject = [index[str(v)] for v in _column(covariates, id_column)]
    segment_start = _column(covariates, "start").astype(float).tolist()
    segment_row = list(range(len(segment_subject)))
    native = rust.fit_event_history(
        mark_names,
        covariate_names,
        np.ascontiguousarray(table, dtype=np.float64),
        subject_ids,
        entry,
        exit_,
        event_subject,
        event_time,
        event_mark,
        segment_subject,
        segment_start,
        segment_row,
        str(formula),
        int(atoms),
    )
    return EventHistoryModel(native)
