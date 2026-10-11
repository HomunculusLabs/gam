"""Carry a prompt change using only selected VPD nodes (#2951).

All omitted activations are held at their values on the original prompt. Token
embeddings and W - V U still run; ``diagnostics`` separates these bypasses.
KL sums target positions, averages changed prompts, and is reported in bits.
Run: MPD_MEM_GIB=24 ~/mpd-data/venv/bin/python carry.py search --task hard9308
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import plistlib
import subprocess
import threading
import time

import numpy as np
import torch
from torch.overrides import TorchFunctionMode

import native

LN2 = math.log(2)
COPYCHECK = Path.home() / "mpd-data/graph_oracle/copycheck_vr4.jsonl"


def kl(lp, lq):
    # Do not clamp: small negative values expose floating point disagreement.
    return (lp.exp() * (lp - lq)).sum(-1).sum(-1) / LN2


def synchronize(dev):
    if str(dev).startswith("mps"):
        torch.mps.synchronize()
    elif str(dev).startswith("cuda"):
        torch.cuda.synchronize()


def masks(nat, T, nodes):
    """Node triples or matrix -> GPU bool [T,C] to live-node masks.

    Triple metadata is encoded once on the host. Assignment is advanced tensor
    indexing, never one device assignment per Python node.
    """
    if isinstance(nodes, dict):
        if set(nodes) != set(nat.names):
            raise ValueError("mask dictionary must contain every matrix")
        for n in nat.names:
            if nodes[n].shape != (T, nat.C[n]) or nodes[n].dtype != torch.bool:
                raise ValueError(f"invalid mask for {n}")
            if nodes[n].device != nat.target.site(n).V.device:
                raise ValueError("masks must already be on the model device")
        return nodes
    out = {n: torch.zeros(T, nat.C[n], dtype=torch.bool, device=nat.dev) for n in nat.names}
    rows = np.asarray(list(nodes), dtype=object).reshape(-1, 3)
    if not len(rows):
        return out
    if not set(rows[:, 0]).issubset(nat.names):
        raise ValueError("unknown matrix")
    for n in nat.names:
        pairs = rows[rows[:, 0] == n, 1:].astype(np.int64)
        if len(pairs):
            if (pairs < 0).any() or (pairs[:, 0] >= T).any() or (pairs[:, 1] >= nat.C[n]).any():
                raise ValueError(f"node outside {n}'s [T,C]")
            ix = torch.as_tensor(pairs, device=nat.dev)
            out[n][ix[:, 0], ix[:, 1]] = True
    return out


class _Mix(TorchFunctionMode):
    """Replace precisely Native.fixed's hard patches by differentiable mixes.

    Native has no soft-patch API. The identity lookup matches ONLY the patch
    mask objects supplied by us, with no global monkeypatch or model mutation.
    No reuse/window slicing is used in this mode. The live activation remains
    in the autograd graph, including its dependence on earlier mixed nodes.
    """
    def __init__(self, patch, coefficients):
        self.by_id = {id(patch[n][0]): coefficients[n] for n in patch}
        self.seen = set()

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func is torch.where and id(args[0]) in self.by_id:
            self.seen.add(id(args[0]))
            a = self.by_id[id(args[0])]
            base, live = args[1:]
            return a * live + (1 - a) * base
        return func(*args, **kwargs)


class _Remainders(TorchFunctionMode):
    """Capture or freeze each original remainder product z @ delta_T."""
    def __init__(self, nat, values=None, embedding=None):
        self.names = {id(nat.target.site(n).delta_T): n for n in nat.names}
        self.values = {} if values is None else values
        self.capture = values is None
        self.embedding = embedding
        self.wte = nat.target.wte
        self.seen = set()

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func.__name__ in ("matmul", "__matmul__") and id(args[1]) in self.names:
            n = self.names[id(args[1])]
            self.seen.add(n)
            if not self.capture:
                return self.values[n].expand(args[0].shape[0], -1, -1)
            result = func(*args, **kwargs)
            self.values[n] = result.detach()
            return result
        if self.embedding is not None and func.__name__ == "__getitem__" and args[0] is self.wte:
            return self.embedding
        return func(*args, **kwargs)


class Carry:
    """Cache the original record and changed-prompt references for a study."""
    def __init__(self, nat, ids, targets, prompts, chunk_size=None):
        if not prompts or not targets:
            raise ValueError("need at least one prompt and target")
        if any(len(x) != len(ids) for x in prompts):
            raise ValueError("changed prompts must have the original length")
        self.nat, self.ids, self.targets, self.prompts = nat, ids, targets, prompts
        self.T = len(ids)
        self.rec = nat._model_record(ids)[0]
        with torch.no_grad():
            self.lp = [nat.reference([x], targets) for x in prompts]
        # Bound inference working memory, chiefly activations and attention.
        # This is a resource limit, not a search/scoring hyperparameter.
        H = nat.target.n_head
        per = 4 * (8 * H * self.T**2 + 12 * self.T * sum(nat.C.values()))
        self.chunk_size = max(1, (512 * 1024**2) // per) if chunk_size is None else chunk_size
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be positive")

    @torch.no_grad()
    def measure(self, sets):
        if not len(sets):
            return torch.empty(0, device=self.nat.dev)
        nat = self.nat
        results, start = [], 0
        while start < len(sets):
            B = min(self.chunk_size, len(sets) - start)
            try:
                live = [masks(nat, self.T, s) for s in sets[start:start + B]]
                patch = {n: (~torch.stack([s[n] for s in live]), self.rec[n]) for n in nat.names}
                del live
                total = torch.zeros(B, device=nat.dev)
                for x, lp in zip(self.prompts, self.lp):
                    lq = nat.run(x, self.targets, nat._full_plan(self.T, B), *nat._one(), patch=patch)
                    total += kl(lp, lq)
                results.append(total / len(self.prompts))
                start += B
                del patch, lq
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower() or B == 1:
                    raise
                self.chunk_size = max(1, B // 2)
                # An OOM traceback can retain the entire failed forward graph.
                exc.__traceback__ = None
                live = patch = lq = total = None
                native.free(nat.dev)
                if str(nat.dev).startswith("cuda"):
                    torch.cuda.empty_cache()
        return torch.cat(results)

    def mixed(self, x, coefficients):
        patch = {n: (torch.ones_like(self.rec[n], dtype=torch.bool), self.rec[n]) for n in self.nat.names}
        mode = _Mix(patch, coefficients)
        with mode:
            lq = self.nat.run(x, self.targets, self.nat._full_plan(self.T, 1), *self.nat._one(), patch=patch)
        if len(mode.seen) != len(patch):
            raise RuntimeError("Native's patch operation changed; soft patches were not all intercepted")
        return lq


def carry_kl(nat, ids, targets, prompts, sets, *, chunk_size=None):
    """GPU tensor: one mean changed-prompt carry KL in bits per node set.

    Sets are batched, with bounded working memory and OOM chunk backoff. Accepts
    triples or GPU mask dictionaries. Only metadata starts on the host; model
    activations, masks, distributions and KL arithmetic stay on the device.
    """
    return Carry(nat, ids, targets, prompts, chunk_size).measure(sets)


class Space:
    """Compact, deterministic node order at candidate positions, on the GPU."""
    def __init__(self, nat, T, positions):
        positions = sorted(set(positions))
        if not positions or min(positions) < 0 or max(positions) >= T:
            raise ValueError("positions must be nonempty and inside the text")
        self.nat, self.T, self.positions = nat, T, positions
        self.pos = torch.tensor(positions, device=nat.dev)
        self.slices, offset = {}, 0
        for n in nat.names:
            size = len(positions) * nat.C[n]
            self.slices[n] = slice(offset, offset + size)
            offset += size
        self.size = offset

    def expand(self, a):
        out = {}
        for n, sl in self.slices.items():
            # index_copy is differentiable with respect to a.
            z = torch.zeros(self.T, self.nat.C[n], device=a.device, dtype=a.dtype)
            out[n] = z.index_copy(0, self.pos, a[sl].view(len(self.positions), -1))
        return out

    def nodes(self, active):
        out = set()
        for n, sl in self.slices.items():
            pairs = active[sl].view(len(self.positions), -1).nonzero().cpu().tolist()
            out.update((n, self.positions[p], c) for p, c in pairs)
        return out


class Curve(list):
    """Public (size, KL) curve, with compact measured sets for inspection."""
    def __init__(self, space):
        super().__init__()
        self.space, self.active = space, []

    def record(self, active, score):
        self.append((int(active.sum()), float(score)))
        self.active.append(active.clone())

    def node_set(self, i):
        return self.space.nodes(self.active[i])


def attribution(study, space, active):
    """d(KL)/da at a=1 for live nodes; removed coefficients are held at zero.

    Removed nodes have coefficient zero. Prompts share coefficients; backward
    runs one prompt at a time to bound graph memory. Upstream dependencies are
    differentiated, not detached replacement activations from a reference run.
    """
    a = active.float().detach().requires_grad_(True)
    grad = torch.zeros_like(a)
    for x, lp in zip(study.prompts, study.lp):
        coefficients = {n: v[None] for n, v in space.expand(a).items()}
        loss = kl(lp, study.mixed(x, coefficients)).sum() / len(study.prompts)
        grad += torch.autograd.grad(loss, a)[0]
        del loss, coefficients
    return grad


def _search(study, positions):
    space = Space(study.nat, study.T, positions)
    active = torch.ones(space.size, dtype=torch.bool, device=study.nat.dev)
    curve = Curve(space)
    curve.record(active, study.measure([space.expand(active)])[0])
    batch = 1
    remaining = space.size
    while remaining:
        # Magnitude identifies locally uninfluential nodes. A large positive
        # derivative predicts improvement only for an infinitesimal removal;
        # dropping its node entirely can overshoot and increase KL substantially.
        cost = attribution(study, space, active).abs()
        cost = cost.masked_fill(~active, float("inf"))
        # Grow removals geometrically, then halve the remaining set down to zero.
        count = min(batch, max(1, remaining // 2))
        remove = torch.argsort(cost, stable=True)[:count]
        active[remove] = False
        remaining -= count
        curve.record(active, study.measure([space.expand(active)])[0])
        batch *= 2
        print(f"  {remaining:6d} {curve[-1][1]:.9g}", flush=True)
    return curve


def search(nat, ids, targets, prompts, positions):
    """Measured geometric pruning; return all (node count, carry KL) points.

    Rank by gradient magnitude. Every removal is committed and measured, even
    if KL worsens. There is no
    acceptance threshold, tuned stopping rule or assumption of monotonicity.
    """
    return _search(Carry(nat, ids, targets, prompts), positions)


@torch.no_grad()
def diagnostics(study):
    nat, T = study.nat, study.T
    full = {n: torch.ones(T, nat.C[n], device=nat.dev, dtype=torch.bool) for n in nat.names}
    empty, all_nodes = study.measure([set(), full]).tolist()
    original = nat.reference([study.ids], study.targets)
    movement = torch.stack([kl(lp, original)[0] for lp in study.lp]).mean()
    capture = _Remainders(nat)
    with capture:
        native_original = nat.run(study.ids, study.targets, nat._full_plan(T, 1), *nat._one())
    if capture.seen != set(nat.names):
        raise RuntimeError("did not capture all remainder products")
    patch = {n: (torch.ones_like(study.rec[n], dtype=torch.bool), study.rec[n]) for n in nat.names}
    frozen, frozen_embedding, shifts, frozen_errors, stable_full, quadratic, full_errors = [], [], [], [], [], [], []
    emb = nat.target.wte[torch.tensor(study.ids, device=nat.dev)]
    for x, lp in zip(study.prompts, study.lp):
        full_lq = nat.run(x, study.targets, nat._full_plan(T, 1), *nat._one())
        d = full_lq - lp
        # For normalized p,q, sum p*(expm1(d)-d) equals KL(p||q).
        # This form exposes the float32 normalization/cancellation floor of
        # the prototype's sum p*(-d), without changing the reported metric.
        stable_full.append((lp.exp() * (torch.expm1(d) - d)).sum() / LN2)
        quadratic.append((lp.exp() * d.square()).sum() / (2 * LN2))
        full_errors.append(d.abs().max())
        mode = _Remainders(nat, capture.values)
        with mode:
            lq = nat.run(x, study.targets, nat._full_plan(T, 1), *nat._one(), patch=patch)
        if mode.seen != set(nat.names):
            raise RuntimeError("did not freeze all remainder products")
        frozen.append(kl(lp, lq)[0])
        live = nat.run(x, study.targets, nat._full_plan(T, 1), *nat._one(), patch=patch)
        shifts.append(kl(lq, live)[0])
        mode = _Remainders(nat, capture.values, emb)
        with mode:
            lq = nat.run(x, study.targets, nat._full_plan(T, 1), *nat._one(), patch=patch)
        frozen_embedding.append(kl(lp, lq)[0])
        frozen_errors.append((lq - native_original).abs().max())
    fixed = float(torch.stack(frozen).mean())
    return {"movement_bits": float(movement), "empty_bits": empty,
            "native_movement_bits": float(torch.stack([kl(lp, native_original)[0] for lp in study.lp]).mean()),
            "all_nodes_bits": all_nodes, "empty_frozen_remainders_bits": fixed,
            "all_nodes_stable_kl_bits": float(torch.stack(stable_full).mean()),
            "all_nodes_quadratic_kl_bits": float(torch.stack(quadratic).mean()),
            "all_nodes_max_logprob_error": float(torch.stack(full_errors).max()),
            "empty_frozen_remainders_and_embedding_bits": float(torch.stack(frozen_embedding).mean()),
            "remainder_kl_reduction_bits": fixed - empty,
            "remainder_distribution_shift_bits": float(torch.stack(shifts).mean()),
            "frozen_vs_native_original_max_logprob_error": float(torch.stack(frozen_errors).max()),
            "note": "Remainder effect is a KL difference with the changed embedding present, not an additive information decomposition."}


@torch.no_grad()
def alternatives(nat, ids, source):
    if not 0 < source < len(ids):
        raise ValueError("source must have a preceding prediction position")
    lp = nat.reference([ids[:source]], [source - 1])[0, 0]
    order = lp.topk(4).indices.tolist()
    alts = [int(z) for z in order if z != ids[source]][:3]
    return [ids[:source] + [z] + ids[source + 1:] for z in alts]


def report(curve, diag):
    baseline = diag["empty_bits"]
    cutoff = curve[0][1] + .1 * (baseline - curve[0][1])
    eligible = [i for i, (size, score) in enumerate(curve) if score <= cutoff]
    if not eligible:
        return {"cutoff_bits": cutoff, "smallest": None}
    i = min(eligible, key=lambda j: curve[j][0])
    nodes = sorted(curve.node_set(i))
    counts = Counter((n, p) for n, p, c in nodes)
    return {"cutoff_bits": cutoff,
            "smallest": {"size": len(nodes), "kl_bits": curve[i][1],
                         "matrix_positions": [[n, p, k] for (n, p), k in sorted(counts.items())],
                         "nodes": nodes}}


class _Utilization:
    """Best-effort system GPU utilization on macOS, sampled once per second.

    IORegistry is read-only and needs no privileges. These samples include
    other GPU users and are not a process-specific measurement.
    """
    def __init__(self, dev):
        self.samples, self.stop = [], threading.Event()
        self.thread = None
        if str(dev).startswith("mps"):
            self.thread = threading.Thread(target=self._sample, daemon=True)
            self.thread.start()

    def _sample(self):
        while not self.stop.is_set():
            try:
                raw = subprocess.run(["ioreg", "-a", "-r", "-c", "AGXAccelerator", "-d", "1"],
                                     capture_output=True, timeout=2, check=True).stdout
                for entry in plistlib.loads(raw):
                    value = entry.get("PerformanceStatistics", {}).get("Device Utilization %")
                    if value is not None:
                        self.samples.append(float(value))
            except (OSError, subprocess.SubprocessError, ValueError, plistlib.InvalidFileException):
                return
            self.stop.wait(1)

    def finish(self):
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=3)
        if not self.samples:
            return None
        return {"mean_percent": sum(self.samples) / len(self.samples),
                "min_percent": min(self.samples), "max_percent": max(self.samples),
                "samples": len(self.samples), "scope": "system GPU, 1 second IORegistry samples"}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("search")
    p.add_argument("--task", required=True, help="copy question id, or first6")
    p.add_argument("--positions", default="source,target", help="comma-separated source,target, integer positions, or all")
    p.add_argument("--out", type=Path, help="JSON result path (otherwise only prints)")
    args = ap.parse_args()
    rows = [json.loads(line) for line in COPYCHECK.read_text().splitlines()]
    rows = rows[:6] if args.task == "first6" else [r for r in rows if r["task"] == args.task]
    if not rows:
        ap.error("task not found in copycheck_vr4.jsonl")
    nat = native.Native()
    output = []
    for row in rows:
        ids, targets = native.text(native.TEXTS / "vpd4l" / f"{row['task']}.json")
        prompts = alternatives(nat, ids, row["source"])
        named = {"source": row["source"], "target": row["target"]}
        positions = list(range(len(ids))) if args.positions == "all" else [named[z] if z in named else int(z) for z in args.positions.split(",")]
        study = Carry(nat, ids, targets, prompts)
        diag = diagnostics(study)
        print(row["task"], json.dumps(diag), flush=True)
        synchronize(nat.dev)
        utilization = _Utilization(nat.dev)
        start = time.perf_counter()
        curve = _search(study, positions)
        synchronize(nat.dev)
        seconds = time.perf_counter() - start
        gpu = utilization.finish()
        result = {"task": row["task"], "source": row["source"], "target": row["target"],
                  "targets": targets, "positions": positions, "alternatives": [x[row["source"]] for x in prompts],
                  "device": str(nat.dev), "diagnostics": diag, "curve": curve, "seconds": seconds,
                  "gpu_utilization": gpu, **report(curve, diag)}
        if str(nat.dev).startswith("mps"):
            result["mps_driver_allocated_bytes"] = torch.mps.driver_allocated_memory()
        output.append(result)
        printable = dict(result)
        if result["smallest"] is not None:
            printable["smallest"] = {k: v for k, v in result["smallest"].items() if k != "nodes"}
        print(json.dumps(printable), flush=True)
        if args.out:
            # Output is explicitly requested by --out; no auxiliary artifacts.
            args.out.write_text(json.dumps(output if args.task == "first6" else result, indent=2) + "\n")
        del study, curve
        native.free(nat.dev)


if __name__ == "__main__":
    main()
