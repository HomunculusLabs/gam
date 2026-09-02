# Event histories

`gam_models::event_history` fits marked counting processes: each subject is
observed over a follow-up window, events of several marks may recur, an
absorbing mark ends follow-up, and covariates may change over time.

## The model

For subject `i` and mark `d` the intensity is

```text
log λ_{i,d}(t) = η⁰_{i,d}(t) + Σ_k a_{d,k} z_{i,k}(t)
```

`η⁰` is an ordinary gam linear predictor built from the covariate table plus
the node time, so every term type in the formula language applies: smooths of
age or calendar time, smooths of covariates, tensor terms, random effects.
`z_{i,k}` are independent unit-variance Ornstein–Uhlenbeck atoms with rates
`r_k`, shared by all marks through the loadings `a_{d,k}`.

Nothing about the latent structure is chosen by hand beyond the maximum number
of atoms:

- every atom carries one REML ridge over its loadings and its log-rate, so an
  atom the evidence does not support is switched off;
- rates are parameterised as `ln(r_k · T̄)` with `T̄` the mean follow-up, so
  their prior centre is the cohort's own time scale, whatever the unit;
- the Gauss-Hermite order is raised until the fitted marginal log-likelihood
  is stable, and the fit carries that certificate.

The latent chain is marginalised exactly (to quadrature tolerance) by adaptive
product Gauss-Hermite filtering. The predict and condition steps are Gaussian
convolutions of `envelope × polynomial`, evaluated through the Lagrange
interpolant on the grid, so they are exact for any gap length. Every node's
grid sits at the posterior mean with the predictive spread: the prediction
is first made onto the predictive grid, the posterior mean is read off it,
and the exact prediction and conditioning are redone on a grid centred
there. A tilted Gaussian is a shifted Gaussian, so that keeps the ratio of
density to envelope a bounded mild tilt, and both the interpolation and the
quadrature stay benign at every order the certificate reaches. The gradient
the inner Newton uses is the exact derivative of the computed log-likelihood:
the forward filter replayed on forward-mode duals seeded with the design rows.
Hessians come from Louis' identity, and the directional Hessian derivatives the
outer LAML solve needs come from the same code run on a dual scalar.

Survival with a static frailty is one atom with rate zero; competing risks is
absorbing marks; recurrent events and history-conditioned prediction are the
same family with different rows.

## Fitting

```rust
use gam_models::event_history::*;

let mut cohort = EventHistoryCohort { mark_names, covariates, subjects };
let spec = EventHistorySpec::new(2, vec![term_collection_spec]);
let fit = fit_event_history(&mut cohort, &spec)?;
fit.loadings;          // marks × atoms
fit.rates;             // per atom, in the data's time unit
fit.atom_log_lambdas;  // the ridge each atom ended on
fit.quadrature;        // Gauss-Hermite certificate
```

`covariates` holds one term collection shared by every mark, or one per mark.
Feature columns index the covariate table's columns followed by the node time,
so a smooth of column `n_cov` is a smooth of time.

## Python

```python
import gamfit
model = gamfit.fit_event_history(subjects, events, covariates, "x + s(time)", atoms=2)
model.loadings, model.rates, model.atom_log_lambdas, model.quadrature
f = model.forecast("subject-17", horizons=[6.0, 7.0], absorbing=["death"])
f["survival"], f["expected_counts"]
model.pit("subject-17"); model.pit_ks()
```

`subjects` has columns `id, entry, exit`; `events` has `id, time, mark`;
`covariates` has `id, start` and the covariate columns, one row per segment
during which a subject's covariates are constant. Frames may be pandas,
polars, or dicts of columns.

## CLI

```bash
gam fit-events --subjects s.csv --events e.csv --covariates c.csv \
    --formula "x + s(time)" --atoms 2 --horizons 1,2,5 --absorbing death \
    --out summary.json
```

The summary carries the loadings, rates, per-atom ridges, coefficients, the
quadrature certificate, the predictive-PIT Kolmogorov–Smirnov distance over
every event, and per-subject forecasts at the given offsets after exit.

## Forecasting and calibration

```rust
let request = ForecastRequest { history: &subject, horizons: &[6.0, 7.0], absorbing: &[true, false], future_row: 0 };
let f = forecast(&fit, &cohort, &request)?;
f.survival;         // P(no absorbing event by horizon | history)
f.expected_counts;  // horizons × marks; the cumulative incidence for absorbing marks
```

The forecast filters the subject's own history into its latent state, then
runs the same filter over the future with zero counts. The predictive PIT of
every event, `predictive_pit`, is uniform when the model is right;
`kolmogorov_smirnov_uniform` summarises it.
