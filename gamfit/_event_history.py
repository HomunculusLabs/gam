"""Event histories: marked counting processes with smooth covariate and time
effects, per-mark risk sets, and a per-subject latent state whose covariance
rank the evidence grows from zero.

The latent path is integrated out by a Laplace approximation on its Markov
structure; forecasts are per-mark first-occurrence probabilities under the
smoothed state at exit. See ``docs/event-history.md``.
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
    def mark_kinds(self) -> dict[str, str]:
        """``recurrent``, ``once`` or ``terminal`` per mark."""
        return dict(zip(self.mark_names, self._native.mark_kinds()))

    @property
    def covariate_names(self) -> list[str]:
        return list(self._native.covariate_names())

    @property
    def subject_ids(self) -> list[str]:
        return list(self._native.subject_ids())

    @property
    def rank(self) -> int:
        """Rank of the latent covariance the evidence supports."""
        return int(self._native.rank())

    @property
    def loadings(self) -> np.ndarray:
        """Factor coordinates of the latent covariance, shape ``(marks, rank)``.
        The covariance itself is the reported object: see
        :meth:`disease_covariance`."""
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
    def atom_evidence(self) -> np.ndarray:
        """Decrease of the outer LAML criterion each accepted atom brought."""
        return np.asarray(self._native.atom_evidence())

    @property
    def rank_path(self) -> list[dict[str, Any]]:
        """Every rank step tried: the score's top eigenvalue, the proposed
        log-rate, the evidence gain, and whether it was accepted."""
        return [dict(step) for step in self._native.rank_path()]

    @property
    def log_likelihood(self) -> float:
        return float(self._native.log_likelihood())

    @property
    def reml_score(self) -> float | None:
        return self._native.reml_score()

    def coefficients(self, mark: int | str) -> np.ndarray:
        """Coefficients of one mark's population log-intensity surface: the
        latent term enters as the deviation from a population rate, so
        ``exp(η⁰)`` is the intensity averaged over the latent state."""
        return np.asarray(self._native.coefficients(self._mark(mark)))

    def disease_covariance(self) -> np.ndarray:
        """``C(0) = A Aᵀ``: the covariance across marks of the latent
        log-intensity deviations at one time, shape ``(marks, marks)``."""
        return np.asarray(self._native.disease_covariance())

    def temporal_covariance(self, lag: float) -> np.ndarray:
        """``C(Δ) = A diag(exp(-r Δ)) Aᵀ`` across a lag of ``lag`` time units."""
        return np.asarray(self._native.temporal_covariance(float(lag)))

    def eigenmodes(self) -> tuple[np.ndarray, np.ndarray]:
        """Eigenvalues (descending) and eigenvectors (columns) of ``C(0)``."""
        values, vectors = self._native.eigenmodes()
        return np.asarray(values), np.asarray(vectors)

    def _mark(self, mark: int | str) -> int:
        if isinstance(mark, str):
            return self.mark_names.index(mark)
        return int(mark)

    def _subject(self, subject: int | str) -> int:
        if isinstance(subject, str):
            return self._subject_index[subject]
        return int(subject)

    @staticmethod
    def _risk(out: Any) -> dict[str, Any]:
        return {
            "horizons": np.asarray(out["horizons"]),
            "marks": list(out["marks"]),
            "risk": np.asarray(out["risk"]),
            "survival": np.asarray(out["survival"]),
            "expected_counts": np.asarray(out["expected_counts"]),
        }

    def forecast(
        self,
        subject: int | str,
        horizons: Sequence[float],
        future_row: int | None = None,
    ) -> dict[str, Any]:
        """Per-mark risks of one subject beyond its exit at absolute
        ``horizons``, given its history: ``risk`` (horizons × marks) is the
        probability that the mark first occurs by the horizon before any
        terminal event, ``NaN`` for a once-only mark the subject already has;
        ``survival`` is the probability of no terminal event by the horizon;
        ``expected_counts`` the expected number of events per mark."""
        return self._risk(
            self._native.forecast(self._subject(subject), [float(h) for h in horizons], future_row)
        )

    def population_forecast(
        self,
        covariates: Mapping[str, float] | Sequence[float],
        start: float,
        horizons: Sequence[float],
    ) -> dict[str, Any]:
        """The same risks for a subject with no observed history, from
        covariate values alone: the latent state starts at its stationary
        prior. Population covariate values give the population tier; a
        subject's own score gives what the model says before its history is
        seen. ``covariates`` is a mapping by covariate name or a sequence in
        ``covariate_names`` order; ``start`` is the time the window opens."""
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
        return self._risk(
            self._native.population_forecast(values, float(start), [float(h) for h in horizons])
        )

    def latent_state(self, subject: int | str) -> dict[str, np.ndarray]:
        """The smoothed latent state of one subject on its own nodes:
        ``times``, ``means`` (nodes × rank) and ``covariances``
        (nodes × rank × rank), the Laplace posterior given the whole history."""
        out = self._native.latent_state(self._subject(subject))
        return {key: np.asarray(out[key]) for key in ("times", "means", "covariances")}

    def latent_exposure(self, subject: int | str) -> dict[str, np.ndarray]:
        """The follow-up average of one subject's latent state as a posterior
        Gaussian: ``mean`` (rank) and ``covariance`` (rank × rank)."""
        out = self._native.latent_exposure(self._subject(subject))
        return {"mean": np.asarray(out["mean"]), "covariance": np.asarray(out["covariance"])}

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
    once: Sequence[str] = (),
    terminal: Sequence[str] = (),
    id_column: str = "id",
) -> EventHistoryModel:
    """Fit an event-history model.

    ``subjects`` has columns ``id, entry, exit``; ``events`` has ``id, time,
    mark`` (an event at or before a subject's entry is prior history: it is
    not modelled, but it removes the subject from the risk set of a mark that
    happens once); ``covariates`` has ``id, start`` and the covariate columns,
    one row per covariate segment (a subject's covariates are constant from
    ``start`` until its next segment). ``formula`` is the right-hand side of a
    gam formula over the covariate columns and ``time``, e.g. ``"x + s(time)"``.
    ``once`` names the marks that happen at most once per subject (a first
    diagnosis); ``terminal`` the marks that end follow-up (death); every
    other mark may recur. The rank of the latent covariance is grown from
    zero by the evidence. An observed risk score enters as a penalised slope
    surface, ``"s(time, by=prs, identifiability=none)"``: its effect on every
    mark may bend with time as much as the evidence supports and collapses to
    zero when the score carries nothing.
    """
    rust = rust_module()
    subject_ids = [str(v) for v in _column(subjects, id_column)]
    index = {sid: i for i, sid in enumerate(subject_ids)}
    entry = _column(subjects, "entry").astype(float).tolist()
    exit_ = _column(subjects, "exit").astype(float).tolist()
    mark_values = [str(v) for v in _column(events, "mark")]
    mark_names = sorted(set(mark_values) | set(once) | set(terminal))
    mark_index = {name: i for i, name in enumerate(mark_names)}
    if set(once) & set(terminal):
        raise ValueError("a mark cannot be both once-only and terminal")
    mark_kinds = [
        "terminal" if name in set(terminal) else "once" if name in set(once) else "recurrent"
        for name in mark_names
    ]
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
        mark_kinds,
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
    )
    return EventHistoryModel(native)
