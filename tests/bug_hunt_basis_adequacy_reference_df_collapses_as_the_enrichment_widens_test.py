"""Bug hunt: the #2774 basis-adequacy test loses reference d.f. as its
alternative gets WIDER, and with it loses the power the check exists for. A 1-D
smooth that captures 5% of the function it was asked to model is reported
adequate.

``crates/gam-models/src/fit_orchestration/drivers/basis_adequacy.rs:105-108``
describes ``enrichment_rank`` as

    Estimable enrichment directions left after the design was projected out
    -- the reference d.f. of the test, and a direct measure of how much NEW
    resolution the alternative carried.

and lines 128-135 explain why the alternative is built four times wider than the
realized basis:

    4x is the smallest multiple that showed usable power

Measured (``y = sin(30*pi*x) + N(0, 0.1)`` -- 15 cycles -- at n=4000, one seed;
``R2`` is the fitted curve against the noiseless truth):

    k    basis_dim   enrichment_dim   enrichment_rank   R2 vs truth   p_value
   10        9            36                8              0.030      9.6e-05
   12       11            44                6              0.033      7.9e-03
   14       13            52                5              0.039      1.3e-02
   16       15            60                3              0.043      6.3e-01
   18       17            68                1              0.039      1.8e-01

The alternative nearly doubles in width (36 -> 68) and the reference d.f. of the
test built on it falls from 8 to 1. Over five seeds at ``k=16`` (rank 3 in every
replicate, ``R2 <= 0.053`` in every replicate) the p-values are
0.060, 0.212, 0.0024, 0.021, 0.610 -- median 0.060, against the engine's own
``BASIS_ADEQUACY_NOTE_LEVEL = 1e-3``. Nothing in that fit is close to
representing the truth, and the check meant to say so does not.

The size of the test is fine, so this is power, not calibration: on an adequate
fixture (``y = sin(2*pi*x) + N(0, 0.3)``, ``s(x, k=10)``, 150 seeds) the reported
p-values are uniform -- P(p<0.01)=0.000, P(p<0.05)=0.040, P(p<0.25)=0.233,
P(p<0.5)=0.507, mean 0.491.

Mechanism (same root cause as the ``statistic_unavailable`` cliff in the sibling
bug hunt). ``crates/gam-terms/src/inference/basis_adequacy.rs:145`` sets

    const ESTIMABLE_DIRECTION_FLOOR: f64 = 1.0e-9;

and lines 336-346 keep only the eigenvalues of the residualized enrichment Gram
``V = Z~^T W_F Z~`` above ``energy_scale * 1e-9``, where ``energy_scale`` is the
largest diagonal of the RAW ``Z^T W_F Z``. The residual spectrum of a smooth 1-D
radial kernel is a Karhunen-Loeve tail -- it decays geometrically with no gap --
while ``energy_scale`` grows with the center count, so a wider alternative
truncates HARDER. Redoing the identical algebra outside the engine and swapping
only the tolerance for an ordinary rank tolerance against ``V``'s own largest
eigenvalue (``lambda_max * q * eps``) on the same fits:

    basis_dim   shipped rank / T / p        full rank / T / p
        9          7 /   57.8 / 4.9e-10      34 / 3735.3 / 0.0
       15          3 /    1.2 / 7.5e-01      58 / 3900.4 / 0.0
       19          0 /   -    / -            74 / 3879.1 / 0.0

At ``basis_dim=9`` the retained subspace carries 57.8 of a non-centrality of
3735: 1.5% of the evidence. By ``basis_dim=15`` it carries none of it.

Observed: ``enrichment_rank`` shrinks monotonically while ``enrichment_dim``
grows, and a fit that explains 4% of its target is reported adequate.

Expected: a 4x-wider alternative supplies MORE reference d.f., not less, and a
smooth whose residuals still carry the entire signal in its own covariate is
flagged at the level the engine itself notes at.

Related: #2788 (the same truncation drives the rank to zero past 18 columns and
the report goes dark entirely).
"""

from __future__ import annotations

import importlib
from typing import Any

pytest: Any = importlib.import_module("pytest")
np = pytest.importorskip("numpy")
pytest.importorskip("gamfit._rust")

import gamfit

#: The engine's own fit-time note level
#: (``BASIS_ADEQUACY_NOTE_LEVEL`` in ``drivers/basis_adequacy.rs``).
NOTE_LEVEL = 1.0e-3

#: Widths below the ``statistic_unavailable`` cliff, so every rung here still
#: reports a p-value today and the failures are about the verdict, not its
#: absence.
_LADDER = [10, 12, 14, 16]

N_ROWS = 4000
CYCLES = 15.0


def _frame(seed: int) -> tuple[dict[str, Any], Any]:
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, N_ROWS)
    truth = np.sin(2.0 * CYCLES * np.pi * x)
    return {"x": x, "y": truth + 0.1 * rng.standard_normal(N_ROWS)}, truth


def _measure(seed: int, k: int) -> tuple[dict[str, Any], float]:
    data, truth = _frame(seed)
    model = gamfit.fit(data, f"y ~ s(x, k={k})", family="gaussian")
    row = dict(model.summary().basis_checks[0])
    fitted = np.asarray(model.predict({"x": data["x"]}), dtype=float)
    r_squared = 1.0 - float(np.var(truth - fitted) / np.var(truth))
    return row, r_squared


def test_control_the_ladder_cannot_represent_the_truth() -> None:
    """Green today, and the anchor for everything below: every basis in the
    ladder captures a few percent of a 15-cycle sine observed at SNR 10."""
    for k in _LADDER:
        _, r_squared = _measure(101, k)
        assert r_squared < 0.15, f"k={k}: R^2 {r_squared:.3f} -- fixture is not inadequate"


def test_control_the_narrowest_rung_still_detects_it() -> None:
    """Green today: at 9 columns the test does reject, so the signal is
    detectable through this exact surface."""
    row, _ = _measure(101, 10)
    assert row["provenance"] == "radial_enrichment"
    assert row["p_value"] < NOTE_LEVEL


def test_reference_df_does_not_shrink_as_the_alternative_widens() -> None:
    widths: list[tuple[int, int, int]] = []
    for k in _LADDER:
        row, _ = _measure(101, k)
        assert row["provenance"] == "radial_enrichment", f"k={k}: {row['provenance']}"
        widths.append((row["basis_dim"], row["enrichment_dim"], row["enrichment_rank"]))

    enrichment_dims = [w[1] for w in widths]
    assert enrichment_dims == sorted(enrichment_dims) and len(set(enrichment_dims)) == len(
        enrichment_dims
    ), f"fixture assumption broken: enrichment widths {enrichment_dims} are not increasing"

    ranks = [w[2] for w in widths]
    assert ranks == sorted(ranks), (
        "the reference d.f. falls as the alternative widens: "
        + ", ".join(f"basis_dim={b} enrichment_dim={e} -> rank={r}" for b, e, r in widths)
    )


@pytest.mark.parametrize("k", _LADDER)
def test_a_four_times_wider_alternative_carries_new_resolution(k: int) -> None:
    """``enrichment_dim`` is ``4 * basis_dim``, so after projecting out a design
    of ``basis_dim`` columns at least ``basis_dim`` new directions must remain
    estimable. Today the count runs the other way."""
    row, _ = _measure(101, k)
    assert row["provenance"] == "radial_enrichment"
    assert row["enrichment_rank"] >= row["basis_dim"], (
        f"k={k}: a {row['enrichment_dim']}-column alternative against a "
        f"{row['basis_dim']}-column design kept only {row['enrichment_rank']} "
        "estimable directions"
    )


def test_the_check_flags_a_smooth_that_captures_none_of_its_target() -> None:
    """Five seeds at the widest still-reporting rung. Every replicate has
    ``R^2 < 0.15``; the median p-value must clear the engine's own note level."""
    p_values: list[float] = []
    for seed in range(101, 106):
        row, r_squared = _measure(seed, 16)
        assert row["provenance"] == "radial_enrichment", f"seed={seed}: {row['provenance']}"
        assert r_squared < 0.15, f"seed={seed}: R^2 {r_squared:.3f}"
        p_values.append(float(row["p_value"]))

    median = float(np.median(p_values))
    assert median < NOTE_LEVEL, (
        "s(x, k=16) explains <15% of a 15-cycle sine and the basis-adequacy "
        f"check calls it adequate: p-values {p_values!r}, median {median:.3g}"
    )
