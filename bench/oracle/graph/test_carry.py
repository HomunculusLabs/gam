"""Real MPS/CUDA/CPU checks against the independent pathpatch prototype.

MPD_MEM_GIB=24 ~/mpd-data/venv/bin/python -B bench/oracle/graph/test_carry.py
"""
import math
import json
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import carry
import native


@pytest.fixture(scope="module")
def case():
    nat = native.Native()
    ids, targets = native.text(native.TEXTS / "vpd4l/hard9308.json")
    prompts = carry.alternatives(nat, ids, 322)
    study = carry.Carry(nat, ids, targets, prompts)
    space = carry.Space(nat, len(ids), [322, 332])
    active = torch.ones(space.size, dtype=torch.bool, device=nat.dev)
    diag = carry.diagnostics(study)
    print("diagnostics", diag, flush=True)
    return nat, ids, targets, prompts, study, space, active, diag


@torch.no_grad()
def prototype(nat, ids, targets, prompts, nodes):
    """Literal slow prototype; deliberately independent mask construction."""
    rec = nat._model_record(ids)[0]
    scores = []
    for x in prompts:
        patch = {}
        for n in nat.names:
            m = torch.ones(1, len(ids), nat.C[n], device=nat.dev, dtype=torch.bool)
            for nn, pos, c in nodes:
                if nn == n:
                    m[0, pos, c] = False
            patch[n] = (m, rec[n])
        lp = nat.reference([x], targets)
        lq = nat.run(x, targets, nat._full_plan(len(ids), 1), *nat._one(), patch=patch)
        scores.append(float((lp.exp() * (lp - lq)).sum(-1).sum(-1)) / math.log(2))
    return sum(scores) / len(scores)


def test_empty_movement_and_remainder_leak(case):
    *_, diag = case
    # This checks the claimed empty==movement, while making any actual bypass
    # contribution explicit rather than silently absorbing it into a tolerance.
    delta = abs(diag["empty_bits"] - diag["movement_bits"])
    print("empty versus movement discrepancy, bits:", delta)
    assert delta < 1e-4
    assert abs(diag["empty_frozen_remainders_and_embedding_bits"] - diag["native_movement_bits"]) < 1e-5
    assert diag["frozen_vs_native_original_max_logprob_error"] == 0


def test_everything(case):
    *_, diag = case
    assert abs(diag["all_nodes_bits"]) < 1e-5
    assert 0 <= diag["all_nodes_quadratic_kl_bits"] < 1e-9
    assert diag["all_nodes_max_logprob_error"] < 1e-4


def test_batched_equals_single(case):
    nat, ids, targets, prompts, study, space, active, _ = case
    half = active.clone()
    half[::2] = False
    sets = [set(), space.expand(active), space.expand(half)]
    batch = carry.carry_kl(nat, ids, targets, prompts, sets, chunk_size=3)
    single = carry.carry_kl(nat, ids, targets, prompts, sets, chunk_size=1)
    print("batch", batch.tolist(), "single", single.tolist())
    torch.testing.assert_close(batch, single, atol=1e-5, rtol=0)
    assert batch.device == nat.target.wte.device
    chunked = carry.carry_kl(nat, ids, targets, prompts, sets, chunk_size=2)
    torch.testing.assert_close(chunked, single, atol=1e-5, rtol=0)


def test_hard9308_prototype(case):
    nat, ids, targets, prompts, study, space, active, _ = case
    nodes = space.nodes(active)
    expected = [prototype(nat, ids, targets, prompts, s) for s in (set(), nodes)]
    actual = study.measure([set(), space.expand(active)]).tolist()
    print("hard9308 prototype", expected, "carry", actual)
    assert actual == pytest.approx(expected, abs=1e-5)
    assert expected[0] == pytest.approx(8.42, abs=.02)
    assert abs(expected[1]) < .01


def test_mix_endpoints_and_gradient(case):
    nat, ids, targets, prompts, study, space, active, _ = case
    half = active.clone()
    half[::2] = False
    with torch.no_grad():
        for live in (torch.zeros_like(active), half, active):
            coeff = {n: z.float()[None] for n, z in space.expand(live).items()}
            mixed = torch.stack([carry.kl(lp, study.mixed(x, coeff))[0] for x, lp in zip(prompts, study.lp)]).mean()
            hard = study.measure([space.expand(live)])[0]
            torch.testing.assert_close(mixed, hard, atol=1e-5, rtol=0)
    cost = carry.attribution(study, space, half)
    assert torch.isfinite(cost).all()
    for n in nat.names:
        if n.startswith("h.3.") and n.rsplit(".", 1)[1] not in ("k_proj", "v_proj"):
            # Last-layer source writers/queries cannot reach a later target.
            assert torch.count_nonzero(cost[space.slices[n]][:nat.C[n]]) == 0
    j = int(cost.abs().masked_fill(~half, 0).argmax())
    # A strong live derivative is much less vulnerable to float32 cancellation
    # than an arbitrary component at the near-zero full-set endpoint.
    a = half.float()
    step = .01
    with torch.no_grad():
        losses = []
        for sign in (-1, 1):
            coeff = a.clone()
            coeff[j] += sign * step
            expanded = {n: z[None] for n, z in space.expand(coeff).items()}
            losses.append(torch.stack([carry.kl(lp, study.mixed(x, expanded))[0]
                                       for x, lp in zip(prompts, study.lp)]).mean())
    finite = (losses[1] - losses[0]) / (2 * step)
    print("gradient", float(cost[j]), "finite difference", float(finite))
    torch.testing.assert_close(cost[j], finite, atol=2e-3, rtol=.03)


def test_masks_validate_and_index(case):
    nat, ids, _, _, _, space, active, _ = case
    active[:] = True
    active[1::2] = False
    nodes = space.nodes(active)
    actual = carry.masks(nat, len(ids), nodes)
    for n, expected in space.expand(active).items():
        assert torch.equal(actual[n], expected)
    with pytest.raises(ValueError):
        carry.masks(nat, len(ids), {(nat.names[0], len(ids), 0)})
    with pytest.raises(ValueError):
        carry.masks(nat, len(ids), {("unknown", 0, 0)})


@pytest.mark.parametrize("task,source,target", [("hard9302", 4, 465), ("hard9318", 398, 411)])
def test_nonzero_source_target_controls(task, source, target):
    nat = native.Native()
    ids, targets = native.text(native.TEXTS / "vpd4l" / f"{task}.json")
    prompts = carry.alternatives(nat, ids, source)
    space = carry.Space(nat, len(ids), [source, target])
    active = torch.ones(space.size, dtype=torch.bool, device=nat.dev)
    expected = prototype(nat, ids, targets, prompts, space.nodes(active))
    actual = float(carry.carry_kl(nat, ids, targets, prompts, [space.expand(active)])[0])
    print(task, "source/target-only prototype", expected, "carry", actual)
    assert actual == pytest.approx(expected, abs=1e-5)
    assert expected > .03


def test_finite_removal_versus_local_derivative(case):
    nat, ids, targets, prompts, study, space, _, _ = case
    active = torch.ones(space.size, dtype=torch.bool, device=nat.dev)
    grad = carry.attribution(study, space, active)
    j = int(grad.argmax())
    assert grad[j] > 0
    values = []
    with torch.no_grad():
        for level in (1., .999, .99, 0.):
            a = active.float()
            a[j] = level
            expanded = {n: z[None] for n, z in space.expand(a).items()}
            values.append(float(torch.stack([carry.kl(lp, study.mixed(x, expanded))[0]
                                             for x, lp in zip(prompts, study.lp)]).mean()))
    only = torch.zeros_like(active)
    only[j] = True
    print("signed-rank first removal", space.nodes(only), "gradient", float(grad[j]),
          "KL at a=1,.999,.99,0", values)
    # Verify the full ablation, whose direction may differ from local influence.
    active[j] = False
    assert values[-1] == pytest.approx(float(study.measure([space.expand(active)])[0]), abs=1e-5)


def test_prototype_explanations_and_vpd(case):
    nat, ids, targets, prompts, study, _, _, _ = case
    task_path = native.TEXTS / "vpd4l/hard9308.json"
    behavior = {**json.loads(task_path.read_text()), "path": str(task_path)}
    paths = {
        "reads": Path("/private/tmp/claude-501/-Users-user/b1063ff3-6750-418e-8406-23ae82bf968a/scratchpad/lead/vr4_hard/hard9308.py"),
        "vpdnode": Path.home() / "mpd-data/runpod/oracle-graph-search-vpdn1/search_hard/hard9308.py",
    }
    sets = {}
    for label, path in paths.items():
        if path.exists():
            # Trusted local reference answers; vpd4l's tokenizer is a local file.
            ir = native.mech.trace_inline(path.read_text(), "vpd4l", behavior=behavior)
            assert ir["valid"], ir.get("error")
            sets[label] = native.prefix(ir, ir["graph"]["steps"])[0].node_set()
    steps = nat.vpd_steps(ids, targets)
    for k in (6, 9, 12):
        assert k < len(steps)
        sets[f"vpd_top{2**k}"] = steps[k].node_set()
    expected = [prototype(nat, ids, targets, prompts, s) for s in sets.values()]
    actual = study.measure(list(sets.values())).tolist()
    print("prototype explanations and VPD", dict(zip(sets, actual)),
          "sizes", {n: len(s) for n, s in sets.items()})
    assert actual == pytest.approx(expected, abs=1e-5)
    # They round to the empty result at two decimals, but do not equal it at
    # full precision. The literal prototype above is the correctness check.
    assert actual[list(sets).index("vpd_top4096")] == pytest.approx(8.420146, abs=1e-5)


def test_oom_chunk_backoff(case, monkeypatch):
    nat, ids, targets, prompts, _, space, active, _ = case
    study = carry.Carry(nat, ids, targets, prompts, chunk_size=3)
    selected = space.expand(active)
    sets = [set(), selected, selected]
    expected = carry.Carry(nat, ids, targets, prompts, chunk_size=1).measure(sets)
    original_run = nat.run
    failed = []

    def constrained_run(x, targets, plan, *args, **kwargs):
        if plan["B"] > 1:
            failed.append(plan["B"])
            raise RuntimeError("out of memory: simulated batch limit")
        return original_run(x, targets, plan, *args, **kwargs)

    monkeypatch.setattr(nat, "run", constrained_run)
    actual = study.measure(sets)
    assert failed == [3]
    assert study.chunk_size == 1
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=0)


if __name__ == "__main__":
    # The local memory guard recognizes script paths, but reserves only 2 GiB
    # for `python -m pytest` regardless of MPD_MEM_GIB.
    raise SystemExit(pytest.main([__file__, "-s", "-p", "no:cacheprovider"]))
