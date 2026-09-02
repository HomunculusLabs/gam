# Event histories

`gam_models::event_history` fits marked counting processes: each subject is
observed over a follow-up window, events of several marks may recur, a
terminal mark ends follow-up, a once-only mark removes the subject from that
mark's risk set, and covariates may change over time.

## The model

For subject `i` and mark `d` the intensity is

```text
log λ_{i,d}(t) = η⁰_{i,d}(t) − ½ Σ_k a_{d,k}² + Σ_k a_{d,k} z_{i,k}(t)
```

`η⁰` is an ordinary gam linear predictor built from the covariate table plus
the node time, so every term type in the formula language applies: smooths of
age or calendar time, smooths of covariates, tensor terms, factors, random
effects, varying-coefficient terms. `z_{i,k}` are independent unit-variance
Ornstein–Uhlenbeck atoms with rates `r_k`, shared by all marks through the
loadings `a_{d,k}`.

The latent term is the individual's deviation from a population rate, not an
addition to it. The atoms are stationary and standard at every time, so
`E_z exp(Σ_k a_{d,k} z_{i,k}) = exp(½ Σ_k a_{d,k}²)`, and the shift cancels
it exactly:

```text
E_z λ_{i,d}(t) = exp(η⁰_{i,d}(t))
```

whatever the loadings. `η⁰` is therefore the population-average
log-intensity surface — the coefficients, `mark_eta`, and the CLI's
`coefficients` all describe population rates — and raising the latent
heterogeneity does not raise the population rate behind the baseline's back.
(Marginal survival is not `exp(−∫ exp(η⁰))`: the expectation of the
exponential of an integral is not the exponential of the integral of the
expectation. Forecasts integrate the latent state, so they need no such
identity.)

Nothing about the latent structure is chosen by hand beyond the number of
atoms offered:

- every atom carries one REML ridge over its loadings and its log-rate, so an
  atom the evidence does not support is switched off (its loadings shrink to
  zero under a large ridge). The fit reports the atoms it was offered; which
  ones the data use is read from `loadings` and `atom_log_lambdas`;
- rates are parameterised as `ln(r_k · T̄)` with `T̄` the mean follow-up, so
  the ridge on a log-rate is an empirical-Bayes prior centred on the cohort's
  own time scale, whose strength REML learns per atom;
- the atoms start apart — every loading at one latent standard deviation per
  unit of log-intensity, the log-rates spaced by `ln 2` around the ridge
  centre — which fixes the sign, permutation and (at equal rates) rotation
  gauge of the loading matrix at the initial point. A permutation-symmetric
  start would keep every atom identical under a deterministic Newton.

## Marks and risk sets

Every mark has a kind, declared with the cohort:

| kind | when it fires |
| --- | --- |
| `recurrent` | any number of times; the subject stays at risk |
| `once` | at most once; the subject leaves this mark's risk set, follow-up continues |
| `terminal` | at most once; follow-up ends (the subject's `exit` is its time) |

The compensator of mark `d` integrates `λ_{i,d}(t) R_{i,d}(t)` with `R` the
mark's risk indicator, so competing risks (several terminal marks), a
first-occurrence outcome beside recurrent ones, and a survival outcome are
all the same likelihood with different kinds. Validation enforces the kinds:
a terminal event must end follow-up, a once-only or terminal mark fires at
most once per subject.

Covariates are predictable: at an event node the row in force is the left
limit `X(t⁻)`, so a covariate that changes at the instant of an event never
explains the event that changed it. Categorical covariates carry their level
labels, so factor terms, `by=` gates and random effects resolve against the
labels the user supplied.

## Marginalisation and its certificate

The latent chain is marginalised by adaptive product Gauss-Hermite filtering.
The predict step is a Gaussian convolution of `envelope × polynomial`,
evaluated through the Lagrange interpolant on the grid, exact for any gap
length. Every node's grid sits at the posterior mean with the predictive
spread. The gradient the inner Newton uses is the exact derivative of the
computed log-likelihood: the forward filter replayed on forward-mode duals
seeded with the design rows. The Hessian is Louis' identity accumulated in
coefficient space by one forward sweep of the smoothed complete-data scores
(linear in the node count and in the coefficient count; no all-pairs table),
evaluated by the same quadrature as the value, so it agrees with the second
derivative of the value to the quadrature error. The directional Hessian
derivatives the outer LAML solve needs come from the same code run on a dual
scalar.

Two discretisations enter: the Gauss-Hermite order per latent axis and the
time mesh the latent path is sampled on (the mesh cuts every follow-up at
entry, exit, covariate changes and events, and integrates each cell by
Gauss-Legendre at the order that resolves the spline basis exactly). The fit
refines both until the fitted coefficients are stationary under refinement:
at the fitted smoothing parameters the coefficients are re-solved at the next
order (`2G − 1`) and on the halved mesh, and if any coefficient moves by more
than `quadrature_tolerance` posterior standard deviations (default `0.01`)
the fit repeats at the refined setting. The certificate the fit carries
records both checks. An order whose Lagrange interpolant would amplify
roundoff above the tolerance (the Lebesgue constant of interpolation on
Hermite nodes grows exponentially) is refused rather than tried. The
transient memory of an evaluation (`G^{2K}` for one backward kernel, `P·G^K`
for the carried expectations) is checked against the machine's budget before
any grid is built.

## Fitting

```rust
use gam_models::event_history::*;

let mut cohort = EventHistoryCohort {
    mark_names, mark_kinds, covariate_names, covariate_levels, covariates, subjects,
};
let spec = EventHistorySpec::new(2, vec![term_collection_spec]);
let fit = fit_event_history(&mut cohort, &spec)?;
fit.loadings;          // marks × atoms
fit.rates;             // per atom, in the data's time unit
fit.atom_log_lambdas;  // the ridge each atom ended on
fit.quadrature;        // the refinement certificate
```

`covariates` holds one term collection shared by every mark, or one per mark.
Feature columns index the covariate table's columns followed by the node time,
so a smooth of column `n_cov` is a smooth of time. The bases are built on the
outcome-free design rows (entry, exit, covariate changes and an event-free
quadrature of every follow-up), so a data-adaptive basis never depends on
where the events fell and the time basis spans every window to its ends.

## Observed risk scores

An observed subject-level score `g` — a polygenic score, a calibrated
biomarker — enters as a penalised slope surface, the varying-coefficient
smooth of the formula language:

```text
s(time, by=prs)
```

This is `b_d(t) · g_i` added to mark `d`'s log-intensity. A continuous
by-smooth keeps its constant, so `b_d` is one surface with two REML-selected
ridges: the wiggliness ridge decides how much the score's effect bends with
time, and the null-space ridge decides whether the effect exists at all. A
score that carries nothing collapses to `b_d ≡ 0`; a useful, age-invariant
score becomes a constant; an effect that genuinely fades with age becomes a
smooth decline. With several marks every mark gets its own `b_d(t)` under the
same rule.

The score sits *beside* the latent state, not inside it: `g` is observed, so
the intensity is conditional on it, while the latent state carries what the
event history reveals.

## Python

```python
import gamfit
model = gamfit.fit_event_history(
    subjects, events, covariates, "x + s(time)",
    atoms=2, marks={"relapse": "recurrent", "death": "terminal"},
)
model.loadings, model.rates, model.atom_log_lambdas, model.quadrature
f = model.forecast("subject-17", horizons=[6.0, 7.0])
f["survival"], f["expected_counts"]
f = model.forecast("subject-17", horizons=[6.0, 7.0], future=[(5.5, {"x": 1.0}), (6.5, {"x": 0.0})])
p = model.population_forecast({"x": 0.0}, start=5.0, horizons=[6.0, 7.0])
model.pit("subject-17"); model.pit_ks()
```

`subjects` has columns `id, entry, exit`; `events` has `id, time, mark`;
`covariates` has `id, start` and the covariate columns, one row per segment
during which a subject's covariates are constant. String, boolean or
categorical columns are categorical covariates; numeric columns are
continuous. Frames may be pandas, polars, or dicts of columns. `marks` is the
mark vocabulary with kinds; without it the observed marks are all recurrent
(and a cohort without events must declare its marks). A forecast's `future`
is the covariate path over the window: absent, the row in force at exit
holds; one record holds constant; `(start, record)` pairs change at the given
times. `population_forecast` is the same window from the stationary prior,
for a subject with no history.

## CLI

```bash
gam fit-events --subjects s.csv --events e.csv --covariates c.csv \
    --formula "x + s(time)" --atoms 2 \
    --marks relapse:recurrent,death:terminal \
    --horizons-after-exit 1,2,5 --out summary.json
```

Covariate columns that do not parse as numbers are categorical. The summary
carries the mark kinds, categorical levels, loadings, rates, per-atom ridges,
coefficients, the refinement certificate, the predictive-PIT
Kolmogorov–Smirnov distance over every event, and per-subject forecasts at
the given offsets after exit, each beside the same window run without the
subject's history (`without_history`).

## Forecasting and calibration

```rust
let request = ForecastRequest { history: &subject, horizons: &[6.0, 7.0], future: &[] };
let f = forecast(&fit, &cohort, &request)?;
f.survival;         // P(no terminal event by horizon | history)
f.expected_counts;  // horizons × marks
```

The forecast filters the subject's own history into its latent state, then
integrates the killed process forward along the same latent path: the
survival to a horizon is `E[exp(−∫ Λ_T)]` over the terminal marks' total
intensity, and the expected count of a mark by a horizon is the chronological
integral of `E[S(t) λ_d(t)]`. For a terminal mark that is its cumulative
incidence, for a once-only mark the probability of a first occurrence before
termination (its own hazard joins the killing), for a recurrent mark the
expected number of events before termination. The chronology is real: the
survival at every quadrature time is its own Gauss-Legendre integral of the
elapsed hazard from the cell's start, so every reported probability lies in
`[0, 1]` and the terminal incidences sum to `1 − survival` to quadrature
accuracy. A subject whose follow-up ended with a terminal event has survival
zero and no further counts.

`population_forecast` runs the same window from the stationary prior at
given covariate values: the population tier with population values, the
score-only tier with a subject's own score. No weight between the tiers is
chosen by hand — each is the same probability model conditioned on more.

The predictive PIT of every event, `predictive_pit`, is the Rosenblatt
transform of the event times under the model: independent uniforms across
events and subjects when the model is right (the time-rescaling theorem).
Each carries the predictive probability of every mark at that event, the
diagnostic of the mark model. `kolmogorov_smirnov_uniform` summarises the
PITs and is `None` for a cohort without events; with parameters estimated
from the same data it is a summary, not a calibrated test.
