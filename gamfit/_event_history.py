"""Event histories: marked counting processes with smooth covariate and time
effects and an evidence-selected per-subject latent state.

The latent chain is marginalised by adaptive Gauss-Hermite filtering, and the
fit carries a certificate that its coefficients are stationary under a
refinement of the quadrature and of the time mesh. Forecasts are expectations
under the filtered state, every probability the chronological integral of a
killed process. See ``docs/event-history.md``.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from ._binding import rust_module

MARK_KINDS = ("recurrent", "once", "terminal")


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


def _labels(values: np.ndarray, what: str) -> list[str]:
    """Identifiers as strings, refusing two source values that are unequal
    yet spell the same string (an integer ``1`` and a string ``"1"`` name two
    subjects, and silently merging them would attach one's events to the
    other)."""
    out = [str(v) for v in values]
    first: dict[str, Any] = {}
    for raw, label in zip(values, out):
        seen = first.setdefault(label, raw)
        if seen is not raw and not _same_value(seen, raw):
            raise ValueError(
                f"{what} {seen!r} and {raw!r} are different values that spell the same identifier {label!r}"
            )
    return out


def _same_value(a: Any, b: Any) -> bool:
    try:
        return bool(a == b)
    except Exception:
        return False


def _is_categorical(values: np.ndarray) -> bool:
    """A covariate column is categorical when it is not numeric: strings,
    booleans, pandas categoricals. Numeric codes stay continuous; declare
    such a column categorical by giving it string labels."""
    return values.dtype.kind in ("O", "U", "S", "b")


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
        """Kind of every mark: ``recurrent``, ``once`` or ``terminal``."""
        return dict(zip(self._native.mark_names(), self._native.mark_kinds()))

    @property
    def covariate_names(self) -> list[str]:
        return list(self._native.covariate_names())

    @property
    def covariate_levels(self) -> dict[str, list[str]]:
        """Level labels of every categorical covariate (empty for continuous)."""
        return dict(zip(self._native.covariate_names(), self._native.covariate_levels()))

    @property
    def subject_ids(self) -> list[str]:
        return list(self._native.subject_ids())

    @property
    def rank(self) -> int:
        """Rank of the latent covariance the evidence supports: the fit grows
        it from zero and keeps an atom only when the outer criterion
        improves."""
        return int(self._native.rank())

    @property
    def atom_evidence(self) -> np.ndarray:
        """Decrease of the outer LAML criterion each accepted atom brought."""
        return np.asarray(self._native.atom_evidence())

    @property
    def rank_path(self) -> list[dict[str, Any]]:
        """Every rank step tried: the covariance score's top eigenvalue, the
        proposed log-rate, whether that rate sat at the mesh's resolution
        limit, whether the candidate reached a certified optimum, the
        evidence gain and whether it was accepted."""
        return [dict(step) for step in self._native.rank_path()]

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
    def log_likelihood(self) -> float:
        return float(self._native.log_likelihood())

    @property
    def reml_score(self) -> float | None:
        return self._native.reml_score()

    def coefficients(self, mark: int | str) -> np.ndarray:
        """Coefficients of one mark's population log-intensity surface: the
        latent term enters as the deviation from a population rate, so
        ``exp(η⁰)`` is the intensity averaged over the latent state."""
        if isinstance(mark, str):
            mark = self.mark_names.index(mark)
        return np.asarray(self._native.coefficients(int(mark)))

    def disease_covariance(self) -> np.ndarray:
        """``C(0) = A Aᵀ``: the covariance across marks of the latent
        log-intensity deviations at one time, shape ``(marks, marks)``. This
        is the reported latent object; the loadings are its factor
        coordinates, which two atoms of equal rate could rotate."""
        return np.asarray(self._native.disease_covariance())

    def temporal_covariance(self, lag: float) -> np.ndarray:
        """``C(Δ) = A diag(exp(-r Δ)) Aᵀ`` across a lag of ``lag`` time units."""
        return np.asarray(self._native.temporal_covariance(float(lag)))

    def eigenmodes(self) -> tuple[np.ndarray, np.ndarray]:
        """Eigenvalues (descending) and eigenvectors (columns) of ``C(0)``."""
        values, vectors = self._native.eigenmodes()
        return np.asarray(values), np.asarray(vectors)

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

    def _subject(self, subject: int | str) -> int:
        if isinstance(subject, str):
            return self._subject_index[subject]
        return int(subject)

    def _covariate_values(self, covariates: Mapping[str, Any] | Sequence[Any]) -> list[float]:
        names = self.covariate_names
        levels = self.covariate_levels
        if isinstance(covariates, Mapping):
            missing = [c for c in names if c not in covariates]
            if missing:
                raise KeyError(f"missing covariates {missing}")
            raw = [covariates[c] for c in names]
        else:
            raw = list(covariates)
            if len(raw) != len(names):
                raise ValueError(f"expected {len(names)} covariate values, got {len(raw)}")
        values = []
        for name, value in zip(names, raw):
            if levels[name]:
                label = str(value)
                if label not in levels[name]:
                    raise ValueError(
                        f"unknown level {label!r} for categorical covariate {name!r}; levels: {levels[name]}"
                    )
                values.append(float(levels[name].index(label)))
            else:
                values.append(float(value))
        return values

    def _future(
        self,
        future: Mapping[str, Any] | Sequence[Any] | Sequence[tuple[float, Any]] | None,
        start: float,
    ) -> list[tuple[float, list[float]]]:
        """A covariate path over a window opening at ``start``: ``None`` (the
        current row holds), one covariate record (constant over the window),
        or a sequence of ``(start, record)`` pairs."""
        if future is None:
            return []
        if isinstance(future, Mapping):
            return [(float(start), self._covariate_values(future))]
        pairs = list(future)
        if pairs and all(isinstance(p, tuple) and len(p) == 2 for p in pairs):
            return [(float(t), self._covariate_values(record)) for t, record in pairs]
        return [(float(start), self._covariate_values(pairs))]

    @staticmethod
    def _forecast_dict(out: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "horizons": np.asarray(out["horizons"]),
            "survival": np.asarray(out["survival"]),
            "expected_counts": np.asarray(out["expected_counts"]),
        }

    def forecast(
        self,
        subject: int | str,
        horizons: Sequence[float],
        future: Mapping[str, Any] | Sequence[Any] | Sequence[tuple[float, Any]] | None = None,
    ) -> dict[str, Any]:
        """Forecast one subject beyond its exit: ``survival`` is the
        probability that no terminal mark has fired by each horizon and
        ``expected_counts`` (horizons × marks) is the expected count of each
        mark — its cumulative incidence when terminal, its first-occurrence
        probability when once-only. ``future`` is the covariate path over the
        window: absent, the row in force at exit holds; a record holds
        constant; ``[(start, record), ...]`` changes at the given times."""
        index = self._subject(subject)
        exit_ = float(self._native.subject_exits()[index])
        path = self._future(future, exit_)
        out = self._native.forecast(index, [float(h) for h in horizons], path)
        return self._forecast_dict(out)

    def population_forecast(
        self,
        covariates: Mapping[str, Any] | Sequence[Any] | Sequence[tuple[float, Any]],
        start: float,
        horizons: Sequence[float],
    ) -> dict[str, Any]:
        """Forecast a subject with no observed history from covariate values
        alone: the latent state starts at its stationary prior at ``start``.
        Population covariate values give the population tier; a subject's own
        score gives what the model says before its history is seen.
        ``covariates`` is one record (constant over the window) or a sequence
        of ``(start, record)`` pairs whose first start is at or before
        ``start``."""
        path = self._future(covariates, float(start))
        if not path:
            raise ValueError("population_forecast needs covariate values")
        out = self._native.population_forecast(float(start), [float(h) for h in horizons], path)
        return self._forecast_dict(out)

    def pit(self, subject: int | str) -> dict[str, np.ndarray]:
        """Predictive PIT of every event of one subject (uniform under the
        model), with the event times and marks and the predictive probability
        of each mark at each event (``mark_probabilities``, events × marks)."""
        out = self._native.pit(self._subject(subject))
        return {
            "time": np.asarray(out["time"]),
            "mark": np.asarray(out["mark"]),
            "pit": np.asarray(out["pit"]),
            "mark_probabilities": np.asarray(out["mark_probabilities"]),
        }

    def pit_ks(self) -> float | None:
        """Kolmogorov–Smirnov distance of all predictive PITs from uniform, or
        ``None`` when the cohort has no events."""
        return self._native.pit_ks()


def fit_event_history(
    subjects: Any,
    events: Any,
    covariates: Any,
    formula: str,
    *,
    marks: Mapping[str, str] | Sequence[str] | None = None,
    id_column: str = "id",
) -> EventHistoryModel:
    """Fit an event-history model.

    ``subjects`` has columns ``id, entry, exit``; ``events`` has ``id, time,
    mark``; ``covariates`` has ``id, start`` and the covariate columns, one row
    per covariate segment (a subject's covariates are constant from ``start``
    until its next segment). String, boolean or categorical covariate columns
    are categorical covariates; numeric columns are continuous. ``formula``
    is the right-hand side of a gam formula over the covariate columns and
    ``time``, e.g. ``"x + s(time)"``, or ``"1"`` for an intercept alone.

    ``marks`` declares the mark vocabulary and each mark's kind — a mapping
    ``{"relapse": "recurrent", "death": "terminal"}``, or a sequence of names
    that are all recurrent. Without it the names are the distinct values of
    the events' ``mark`` column, all recurrent, which needs at least one event.
    A terminal mark ends follow-up (the subject's ``exit`` is its time), a
    once-only mark removes the subject from that mark's risk set, a recurrent
    mark can fire any number of times.

    The rank of the latent covariance is grown from zero by the evidence.
    """
    rust = rust_module()
    subject_values = _column(subjects, id_column)
    subject_ids = _labels(subject_values, "subject identifiers")
    if len(set(subject_ids)) != len(subject_ids):
        raise ValueError("subject identifiers must be distinct")
    index = {sid: i for i, sid in enumerate(subject_ids)}
    entry = _column(subjects, "entry").astype(float).tolist()
    exit_ = _column(subjects, "exit").astype(float).tolist()
    mark_values = _labels(_column(events, "mark"), "mark names")
    if marks is None:
        if not mark_values:
            raise ValueError(
                "the events table has no rows, so the mark vocabulary must be given: marks={name: kind}"
            )
        mark_names = sorted(set(mark_values))
        mark_kinds = ["recurrent"] * len(mark_names)
    elif isinstance(marks, Mapping):
        mark_names = [str(k) for k in marks.keys()]
        mark_kinds = [str(v).lower() for v in marks.values()]
    else:
        mark_names = [str(m) for m in marks]
        mark_kinds = ["recurrent"] * len(mark_names)
    for kind in mark_kinds:
        if kind not in MARK_KINDS:
            raise ValueError(f"unknown mark kind {kind!r}; expected one of {MARK_KINDS}")
    mark_index = {name: i for i, name in enumerate(mark_names)}
    unknown = sorted(set(mark_values) - set(mark_names))
    if unknown:
        raise ValueError(f"events carry marks {unknown} that are not in the mark vocabulary {mark_names}")
    event_subject = []
    for v in _column(events, id_column):
        label = str(v)
        if label not in index:
            raise ValueError(f"event subject {label!r} is not in the subjects table")
        event_subject.append(index[label])
    event_time = _column(events, "time").astype(float).tolist()
    event_mark = [mark_index[m] for m in mark_values]
    covariate_names = [
        c for c in _column_names(covariates) if c not in (id_column, "start")
    ]
    columns = []
    covariate_levels: list[list[str]] = []
    for name in covariate_names:
        values = _column(covariates, name)
        if _is_categorical(values):
            labels = [str(v) for v in values]
            levels = sorted(set(labels))
            covariate_levels.append(levels)
            code = {level: float(i) for i, level in enumerate(levels)}
            columns.append(np.asarray([code[v] for v in labels], dtype=float))
        else:
            covariate_levels.append([])
            columns.append(values.astype(float))
    n_segments = len(_column(covariates, "start"))
    table = (
        np.column_stack(columns)
        if columns
        else np.zeros((n_segments, 0), dtype=float)
    )
    segment_subject = []
    for v in _column(covariates, id_column):
        label = str(v)
        if label not in index:
            raise ValueError(f"covariate subject {label!r} is not in the subjects table")
        segment_subject.append(index[label])
    segment_start = _column(covariates, "start").astype(float).tolist()
    segment_row = list(range(len(segment_subject)))
    native = rust.fit_event_history(
        mark_names,
        mark_kinds,
        covariate_names,
        covariate_levels,
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
