"""Bug hunt: ``PosteriorSamples.method`` reports ``'nuts'`` for every standard
model class, including the two that documentedly never run NUTS.

``docs/posterior-sampling.md:100-107`` publishes the dispatch table:

    | Gaussian-identity standard GLM                     | Laplace (closed form) |
    | Standard GLM (probit/cloglog, Poisson, Tweedie...) | NUTS                  |
    | Bernoulli-logit standard GLM (no Firth, no offset, | Polya-Gamma Gibbs     |
    |   unit weights)                                    |                       |

and ``crates/gam-inference/src/sample.rs`` implements it: the Gaussian-identity
route is ``StandardPosteriorRoute::GaussianClosedForm ->
laplace_gaussian_fallback`` (~line 829), and Bernoulli-logit goes to the
Polya-Gamma Gibbs sampler.

The badge does not follow. ``nuts_method_label``
(``crates/gam-pyffi/src/manifold/manifold_and_posterior_ffi.rs:308-332``) maps
``PredictModelClass::Standard`` unconditionally to ``"nuts"`` --

    fn nuts_method_label(model: &FittedModel) -> &'static str {
        match model.predict_model_class() {
            PredictModelClass::Standard => "nuts",
            ...

-- even though its own docstring says it exists to "Mirror the dispatch in
``gam::inference::sample::sample_saved_model``" and correctly resolves
``"laplace"`` for the location-scale and latent-survival classes. Only the
``Standard`` arm, which is where all three samplers live, collapses to one
label.

Measured (n=400, ``y ~ s(x)``, 2 chains x 800 draws):

    gaussian        method='nuts'  rhat=1.000000  ess=1600.0  (= n_draws)
    binomial-logit  method='nuts'  rhat=1.004081  ess=866.1
    poisson         method='nuts'  rhat=1.036801  ess=103.1

Three samplers, one label. The Gaussian row is ``laplace_gaussian_fallback``'s
hard-coded ``rhat: 1.0, ess: n_total`` signature (sample.rs ~line 445): those
are not measured MCMC diagnostics, because no chain ran -- yet a caller who
believes ``method == 'nuts'`` reads them as if they were.

The mislabel also propagates into a public property: ``gamfit/_sampling.py:185``
defines ``is_exact = (method == "nuts")``, so ``is_exact`` is driven by a string
that is wrong for two of the three standard routes.

Observed: ``PosteriorSamples.method == 'nuts'`` for Gaussian-identity (Laplace)
and Bernoulli-logit (Polya-Gamma Gibbs) standard fits.

Expected: the badge names the sampler that ran, matching the documented table --
in particular ``'laplace'`` for a Gaussian-identity standard GLM -- and the three
routes do not all report the same string.
"""

from __future__ import annotations

import importlib
from typing import Any

pytest: Any = importlib.import_module("pytest")
np = pytest.importorskip("numpy")
pytest.importorskip("gamfit._rust")

import gamfit


def _sample(family: str) -> Any:
    rng = np.random.default_rng(5)
    n = 400
    x = np.linspace(0.0, 1.0, n)
    eta = 1.5 * np.sin(2.0 * np.pi * x)
    response = {
        "gaussian": eta + 0.3 * rng.standard_normal(n),
        "binomial-logit": (rng.random(n) < 1.0 / (1.0 + np.exp(-eta))).astype(float),
        "poisson": rng.poisson(np.exp(0.8 + eta)).astype(float),
    }[family]
    data = {"x": x, "y": response}
    model = gamfit.fit(data, "y ~ s(x)", family=family)
    return model.sample(data, seed=3, samples=800, chains=2)


def test_gaussian_identity_standard_fit_is_badged_laplace() -> None:
    """docs/posterior-sampling.md:101 — Gaussian-identity standard GLM => Laplace."""
    posterior = _sample("gaussian")
    assert posterior.method == "laplace", (
        "a Gaussian-identity standard GLM is sampled by laplace_gaussian_fallback "
        f"(closed form, iid) but is badged {posterior.method!r}"
    )


def test_bernoulli_logit_standard_fit_is_not_badged_nuts() -> None:
    """docs/posterior-sampling.md:105 — Bernoulli-logit standard GLM => Polya-Gamma Gibbs."""
    posterior = _sample("binomial-logit")
    assert posterior.method != "nuts", (
        "a Bernoulli-logit standard GLM is sampled by Polya-Gamma Gibbs but is "
        "badged 'nuts'"
    )


def test_three_distinct_standard_samplers_do_not_share_one_badge() -> None:
    """Whatever the names, the badge has to distinguish the routes."""
    labels = {family: _sample(family).method for family in
              ("gaussian", "binomial-logit", "poisson")}
    assert len(set(labels.values())) >= 2, (
        "Laplace, Polya-Gamma Gibbs and NUTS all report the same method badge: "
        f"{labels}"
    )


def test_hardcoded_laplace_diagnostics_are_not_presented_as_mcmc() -> None:
    """rhat == 1.0 exactly and ess == n_draws exactly are constants, not measurements."""
    posterior = _sample("gaussian")
    fabricated = (
        posterior.rhat == 1.0 and posterior.ess == float(posterior.n_draws)
    )
    assert not (fabricated and posterior.method == "nuts"), (
        "the closed-form path's hard-coded rhat=1.0 / ess=n_draws are surfaced "
        f"under method='nuts': rhat={posterior.rhat!r}, ess={posterior.ess!r}, "
        f"n_draws={posterior.n_draws}"
    )
