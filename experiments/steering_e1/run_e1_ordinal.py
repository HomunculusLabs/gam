#!/usr/bin/env python3
"""E1-ORDINAL — on-manifold steering of a NON-CYCLIC (ordinal / interval) chart (gam#2234).

Why this file exists
--------------------
`run_e1.py` steers the day-of-week CIRCLE. A circle is the easy case for a
manifold SAE: the chart is periodic, the group action is a rotation, and a flat
decoder direction cannot possibly track it. The owner's direction on gam#2234 is
explicit — *"do not overfit to cyclic"* — so the #2234 claim ("chart-coordinate
interventions with dose in the chart's own units beat a matched-norm flat
direction") has to be tested on a structure with **no periodicity at all**.

This harness is the same protocol on an **ordinal magnitude line**: the number
words ` one` … ` twelve`. The fitted chart is `atom_topology="euclidean"`,
`d_atom=1` — an interval, not a circle — and the steering dose is a signed step
along that interval, in chart units per ordinal rank (estimated once from the
train split by regressing the fitted coordinate on the rank). Base labels and
target shifts are chosen so the target NEVER wraps past ` twelve`; a wrap would
smuggle cyclic structure back in.

Everything downstream is deliberately identical to `run_e1.py` so the two
structures are comparable and so `analyze_collateral.py` (E2) consumes this
output unchanged:

  * effect      = full-softmax probability mass moved onto the intended target
                  token (the rank AFTER the intervention, `source + shift + 1`),
  * collateral  = target-excluded `KL(patched || base)` over the full vocabulary,
  * control     = a matched-L2-norm fixed-direction addition from a flat SAE
                  (`gamfit.sparse_dictionary_fit`), the latent whose code best
                  tracks rank,
  * chart       = a train-only PCA-`r` chart; steering deltas are lifted back to
                  ambient through the orthonormal rows, so the intervention is
                  exact and norm-preserving,
  * assignment  = `softmax` (the steer entry's routed assignment).

Model / capture / patch helpers are imported from `run_e1` rather than copied,
so the two harnesses cannot drift apart on the measurement contract.

Launch:
    python3 experiments/steering_e1/run_e1_ordinal.py \
        --model Qwen/Qwen3.5-4B-Base --layer-index 16 \
        --out-dir experiments/steering_e1/out_ordinal
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def _load_run_e1():
    """Import the sibling `run_e1` module by path (works from any cwd)."""
    if "run_e1" in sys.modules:
        return sys.modules["run_e1"]
    path = Path(__file__).resolve().parent / "run_e1.py"
    spec = importlib.util.spec_from_file_location("run_e1", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load sibling harness at {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_e1"] = module
    spec.loader.exec_module(module)
    return module


E1 = _load_run_e1()

# The ordinal ladder. Non-cyclic by construction: there is no edge joining
# ` twelve` back to ` one`, and the base/shift grid never asks for one.
ORDINALS = (
    "one", "two", "three", "four", "five", "six",
    "seven", "eight", "nine", "ten", "eleven", "twelve",
)

FIT_TEMPLATES = (
    "Counting up, the number after {label} is",
    "In order: {label} is followed by",
    "The next whole number after {label} is",
    "Adding one to {label} gives",
    "On the number line, immediately right of {label} lies",
    "If you have {label} and take one more you have",
    "Ascending sequence, the term after {label} is",
    "Successor of {label} is",
    "Step forward once from {label} and you reach",
    "One greater than {label} is",
)
BASE_TEMPLATES = (
    "Starting at {label}, the next number is",
    "Sequence note: after {label} comes",
)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def candidate_token_ids(tokenizer: Any, prefix: str) -> list[int]:
    ids: list[int] = []
    for label in ORDINALS:
        enc = tokenizer.encode(prefix + label, add_special_tokens=False)
        if len(enc) != 1:
            raise ValueError(
                f"candidate {prefix + label!r} tokenized to {len(enc)} tokens; "
                "choose a different --candidate-prefix"
            )
        ids.append(int(enc[0]))
    return ids


def label_token_probabilities(logits: Any, candidate_ids: list[int]) -> np.ndarray:
    """Full-softmax probabilities for the ordinal label tokens (never renormalized)."""
    logp = E1._log_softmax(logits)
    ids = np.asarray(candidate_ids, dtype=np.int64)
    if ids.shape != (len(ORDINALS),) or np.any(ids < 0) or np.any(ids >= logp.size):
        raise ValueError("candidate ordinal token ids must be in-vocabulary ids, one per label")
    return np.exp(logp[ids])


@dataclass(eq=False)
class CleanExample:
    template_index: int
    rank: int
    prompt: str
    activation: Any
    logits: Any


def collect_cloud(model, tokenizer, layer, templates, ranks):
    examples: list[CleanExample] = []
    for ti, template in enumerate(templates):
        for rank in ranks:
            prompt = template.format(label=ORDINALS[rank])
            act, logits = E1.run_clean(model, tokenizer, layer, prompt)
            examples.append(CleanExample(ti, rank, prompt, act, logits))
    return examples


def continuation_target_rank(base_rank: int, shift: int) -> int:
    """Rank of the next-token answer after moving the source rank by `shift`.

    No modular arithmetic anywhere: this is the ordinal line, and the caller is
    responsible for keeping `base_rank + shift + 1` inside the ladder.
    """
    target = base_rank + shift + 1
    if not (0 <= target < len(ORDINALS)):
        raise ValueError(
            f"target rank {target} leaves the ordinal ladder [0, {len(ORDINALS)}); "
            "the non-cyclic protocol forbids wrapping"
        )
    return target


def parse_shifts(spec: str) -> list[int]:
    try:
        shifts = [int(v.strip()) for v in spec.split(",") if v.strip()]
    except ValueError as error:
        raise ValueError("--target-shifts must be comma-separated integers") from error
    if not shifts or len(set(shifts)) != len(shifts) or any(s < 1 for s in shifts):
        raise ValueError("--target-shifts must contain unique positive integers")
    return sorted(shifts)


def _linear_r2(signal: np.ndarray, ranks: np.ndarray) -> tuple[float, float]:
    """`(R^2, slope)` of a 1-D signal regressed on rank (the ordinal design)."""
    d = np.column_stack([np.ones(len(ranks)), ranks.astype(np.float64)])
    coef, *_ = np.linalg.lstsq(d, signal, rcond=None)
    resid = signal - d @ coef
    tss = float(np.sum((signal - signal.mean()) ** 2))
    return 1.0 - float(np.sum(resid**2)) / max(tss, 1e-30), float(coef[1])


def select_ordinal_atom(model, ranks: np.ndarray) -> tuple[int, float, float]:
    """Atom whose fitted coordinate is most linear in rank.

    Returns `(atom, chart_units_per_rank, r2)`. The sign of the slope IS the
    chart orientation — on an interval there is no separate orientation gauge to
    resolve, unlike the circle's conjugation ambiguity.
    """
    best_k, best_r2, best_slope = 0, -np.inf, 1.0
    for k in range(len(model.coords)):
        c = np.asarray(model.coords[k], dtype=float)
        coord = c[:, 0] if c.ndim == 2 else c
        r2, slope = _linear_r2(coord, ranks)
        if r2 > best_r2:
            best_k, best_r2, best_slope = k, r2, slope
    if not np.isfinite(best_slope) or abs(best_slope) < 1e-12:
        raise ValueError(
            "no fitted atom carries a nonzero rank gradient; the ordinal chart did not form"
        )
    log(f"ordinal atom = {best_k} (rank R2={best_r2:.4f}, chart units/rank={best_slope:+.6f})")
    return best_k, best_slope, best_r2


def select_flat_direction(flat_fit, X, ranks):
    """Flat-SAE ordinal latent's unit decoder direction (fixed-direction control)."""
    tr = flat_fit.transform(X)
    k = int(flat_fit.decoder.shape[0])
    codes = np.zeros((X.shape[0], k), dtype=np.float64)
    rows = np.arange(X.shape[0])[:, None]
    codes[rows, tr.indices.astype(np.int64)] = tr.codes.astype(np.float64)
    best_lat, best = 0, -np.inf
    for lat in range(k):
        col = codes[:, lat]
        if np.allclose(col, 0.0):
            continue
        r2, _ = _linear_r2(col, ranks)
        if r2 > best:
            best, best_lat = r2, lat
    w = np.asarray(flat_fit.decoder[best_lat], dtype=np.float64)
    w = w / max(np.linalg.norm(w), 1e-30)
    log(f"flat ordinal latent = {best_lat} (code rank R2={best:.4f})")
    return w, best_lat


def steer_records(lm_model, sae_model, tokenizer, layer, atom, units_per_rank,
                  base_examples, metric_rows, base_coords, base_amplitudes,
                  candidate_ids, flat_dir, target_shifts, dose_fractions, lift=None):
    import torch

    records: list[dict[str, Any]] = []
    for base, metric_row, t0_in, amplitude in zip(
        base_examples, metric_rows, base_coords, base_amplitudes
    ):
        base_probs = label_token_probabilities(base.logits, candidate_ids)
        t0 = np.atleast_1d(np.asarray(t0_in, dtype=np.float64)).reshape(-1)
        for shift in target_shifts:
            if base.rank + shift + 1 >= len(ORDINALS):
                continue  # would wrap; the non-cyclic protocol drops it
            target_rank = continuation_target_rank(base.rank, shift)
            target_token_id = candidate_ids[target_rank]
            base_target_probability = float(base_probs[target_rank])
            for dose_fraction in dose_fractions:
                dose_ranks = float(shift) * dose_fraction
                dcoord = units_per_rank * dose_ranks
                t_to = t0.copy()
                t_to[0] = t0[0] + dcoord
                plan = sae_model.steer(int(atom), int(metric_row), float(amplitude), t0, t_to)
                delta = np.asarray(plan["delta"], dtype=np.float64)
                if lift is not None:
                    delta = delta @ lift

                patched = base.activation + torch.from_numpy(delta.astype(np.float32))
                manifold_logits = E1.run_patched(lm_model, tokenizer, layer, base.prompt, patched)

                flat_delta = np.linalg.norm(delta) * flat_dir
                patched_flat = base.activation + torch.from_numpy(flat_delta.astype(np.float32))
                flat_logits = E1.run_patched(lm_model, tokenizer, layer, base.prompt, patched_flat)

                for arm, patched_logits in (
                    ("manifold", manifold_logits),
                    ("flat", flat_logits),
                ):
                    probs = label_token_probabilities(patched_logits, candidate_ids)
                    top = int(np.argmax(probs))
                    target_probability = float(probs[target_rank])
                    collateral = E1.target_excluded_kl_model_to_base(
                        patched_logits, base.logits, target_token_id
                    )
                    records.append({
                        "structure": "ordinal",
                        "arm": arm,
                        "base_template": base.template_index,
                        "base_label": ORDINALS[base.rank],
                        "base_day_index": base.rank,      # schema-compatible with run_e1
                        "target_shift_days": int(shift),  # "shift" in ordinal ranks
                        "target_label": ORDINALS[target_rank],
                        "target_day_index": target_rank,
                        "target_token_id": int(target_token_id),
                        "dose_fraction": float(dose_fraction),
                        "coordinate_delta_chart_units": float(dcoord),
                        "chart_units_per_rank": float(units_per_rank),
                        "delta_norm": float(np.linalg.norm(delta)),
                        "steer_off_manifold_norm": (
                            float(plan["off_manifold_norm"])
                            if arm == "manifold" and plan.get("off_manifold_norm") is not None
                            else None
                        ),
                        "steer_predicted_nats": (
                            float(plan["predicted_nats"])
                            if arm == "manifold" and plan.get("predicted_nats") is not None
                            else None
                        ),
                        "realized_top_label": ORDINALS[top],
                        "realized_top_weekday_index": top,
                        "target_token_probability": target_probability,
                        "base_target_token_probability": base_target_probability,
                        "target_probability_mass_moved": (
                            target_probability - base_target_probability
                        ),
                        "collateral_kl_model_to_base_non_target": collateral,
                        "label_token_probabilities": [float(x) for x in probs],
                    })
    return records


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B-Base")
    ap.add_argument("--cache-dir", default="")
    ap.add_argument("--layer-index", type=int, default=16)
    ap.add_argument("--k-atoms", type=int, default=1)
    ap.add_argument("--flat-k", type=int, default=32)
    ap.add_argument("--n-iter", type=int, default=60)
    ap.add_argument("--base-ranks", default="0,1,2,3,4,5,6,7",
                    help="comma-separated 0-based ordinal ranks used as sources (must not wrap)")
    ap.add_argument("--target-shifts", default="1,2,3")
    ap.add_argument("--dose-fractions", default="0,0.25,0.5,0.75,1")
    ap.add_argument("--candidate-prefix", default=" ")
    ap.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="fp32")
    ap.add_argument("--seed", type=int, default=20260731)
    ap.add_argument("--pca-dim", type=int, default=64)
    ap.add_argument("--per-template-center", action="store_true",
                    help="remove each prompt template's mean before charting (see run_e1.py: the "
                         "cloud's leading variance is template identity, not the ordinal rank). "
                         "Chart-construction only; the steering delta still edits the RAW "
                         "activation.")
    ap.add_argument("--out-dir", default="experiments/steering_e1/out_ordinal")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    target_shifts = parse_shifts(args.target_shifts)
    dose_fractions = E1.parse_dose_fractions(args.dose_fractions)
    base_ranks = [int(v) for v in args.base_ranks.split(",") if v.strip()]
    if not base_ranks or any(not 0 <= r < len(ORDINALS) for r in base_ranks):
        raise ValueError("--base-ranks must be 0-based ranks inside the ordinal ladder")
    np.random.seed(args.seed)
    import gamfit

    log(f"loading {args.model}")
    model_lm, tok = E1.load_model_and_tokenizer(args.model, args.cache_dir, args.dtype)
    layers = E1.resolve_layers(model_lm)
    if not (0 <= args.layer_index < len(layers)):
        raise ValueError(f"--layer-index must be in [0,{len(layers)}); got {args.layer_index}")
    layer = layers[args.layer_index]
    candidate_ids = candidate_token_ids(tok, args.candidate_prefix)
    log(f"candidate ids: {dict(zip(ORDINALS, candidate_ids))}")

    fit_ranks = list(range(len(ORDINALS)))  # the chart is fit on the whole ladder
    log("collecting disjoint fit and held-out ordinal activation clouds")
    fit_examples = collect_cloud(model_lm, tok, layer, FIT_TEMPLATES, fit_ranks)
    base_examples = collect_cloud(model_lm, tok, layer, BASE_TEMPLATES, base_ranks)
    X_fit_ambient = np.ascontiguousarray(
        np.stack([ex.activation.numpy().astype(np.float64) for ex in fit_examples]))
    X_base_ambient = np.ascontiguousarray(
        np.stack([ex.activation.numpy().astype(np.float64) for ex in base_examples]))
    fit_rank_array = np.asarray([ex.rank for ex in fit_examples])
    log(f"fit X shape {X_fit_ambient.shape}; held-out X shape {X_base_ambient.shape}")

    if args.per_template_center:
        Xf = X_fit_ambient.copy()
        for ti in {ex.template_index for ex in fit_examples}:
            rows = [i for i, ex in enumerate(fit_examples) if ex.template_index == ti]
            Xf[rows] -= Xf[rows].mean(0, keepdims=True)
        Xb = X_base_ambient.copy()
        for ti in {ex.template_index for ex in base_examples}:
            rows = [i for i, ex in enumerate(base_examples) if ex.template_index == ti]
            Xb[rows] -= Xb[rows].mean(0, keepdims=True)
        X_fit_ambient, X_base_ambient = Xf, Xb
        log("per-template centering applied to the chart inputs")

    if args.pca_dim and args.pca_dim < X_fit_ambient.shape[1]:
        mu = X_fit_ambient.mean(0, keepdims=True)
        X_fit_centered = X_fit_ambient - mu
        _, svals, vt = np.linalg.svd(X_fit_centered, full_matrices=False)
        r = int(min(args.pca_dim, vt.shape[0]))
        lift = np.ascontiguousarray(vt[:r])
        X_fit = np.ascontiguousarray(X_fit_centered @ lift.T)
        X_base = np.ascontiguousarray((X_base_ambient - mu) @ lift.T)
        evr = float((svals[:r] ** 2).sum() / max((svals**2).sum(), 1e-30))
        log(f"train-only PCA chart: {X_fit.shape} (fit explained variance {evr:.4f})")
    else:
        lift, X_fit, X_base, evr = None, X_fit_ambient, X_base_ambient, 1.0

    log("fitting gamfit.sae_manifold_fit (EUCLIDEAN interval chart, softmax assignment)")
    sae_model = gamfit.sae_manifold_fit(
        X_fit, K=args.k_atoms, d_atom=1, atom_topology="euclidean", assignment="softmax",
        n_iter=args.n_iter, random_state=args.seed)
    fit_ev = float(1.0 - np.sum((X_fit - np.asarray(sae_model.fitted)) ** 2)
                   / max(np.sum((X_fit - X_fit.mean(0)) ** 2), 1e-30))
    atom, units_per_rank, rank_r2 = select_ordinal_atom(sae_model, fit_rank_array)

    log("fitting flat-SAE control (gamfit.sparse_dictionary_fit)")
    flat_fit = gamfit.sparse_dictionary_fit(
        X_fit.astype(np.float32), min(args.flat_k, X_fit.shape[0] - 1),
        active=1, max_epochs=40)
    flat_dir, flat_lat = select_flat_direction(
        flat_fit, X_fit.astype(np.float32), fit_rank_array)
    if lift is not None:
        flat_dir = np.asarray(flat_dir, dtype=np.float64) @ lift
        norm = np.linalg.norm(flat_dir)
        if norm > 0:
            flat_dir = flat_dir / norm

    base_latents = sae_model.converged_latents(X_base)
    base_coords_array = np.asarray(base_latents["coords"][atom], dtype=float)
    base_assignments = np.asarray(base_latents["assignments"], dtype=float)
    fit_coords_array = np.asarray(sae_model.coords[atom], dtype=float)
    fit_line = fit_coords_array[:, 0]
    metric_rows = [int(np.argmin(np.abs(fit_line - t))) for t in base_coords_array[:, 0]]
    base_coords = [base_coords_array[row] for row in range(len(base_examples))]
    base_amplitudes = [float(base_assignments[row, atom]) for row in range(len(base_examples))]

    log(f"steering {len(base_examples)} base contexts × shifts {target_shifts} × "
        f"dose fractions {dose_fractions} (manifold + flat)")
    records = steer_records(
        model_lm, sae_model, tok, layer, atom, units_per_rank,
        base_examples, metric_rows, base_coords, base_amplitudes,
        candidate_ids, flat_dir, target_shifts, dose_fractions, lift=lift)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "e1_records.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    meta = {
        "structure": "ordinal (non-cyclic interval chart)",
        "model": args.model, "layer_index": args.layer_index, "k_atoms": args.k_atoms,
        "atom_topology": "euclidean",
        "flat_k": int(flat_fit.decoder.shape[0]), "flat_latent": int(flat_lat),
        "ordinal_atom": int(atom), "chart_units_per_rank": float(units_per_rank),
        "atom_rank_r2": float(rank_r2), "fit_ev": fit_ev, "pca_explained_variance": evr,
        "n_fit_rows": int(len(fit_examples)), "n_base_rows": int(len(base_examples)),
        "base_ranks": base_ranks, "target_shifts": target_shifts,
        "dose_fractions": dose_fractions, "seed": args.seed,
    }
    endpoint = {}
    for arm in ("manifold", "flat"):
        rs = [r for r in records if r["arm"] == arm and r["dose_fraction"] == 1.0]
        if rs:
            endpoint[arm] = {
                "endpoint_target_accuracy": float(np.mean([
                    r["realized_top_weekday_index"] == r["target_day_index"] for r in rs])),
                "mean_endpoint_target_token_probability": float(np.mean([
                    r["target_token_probability"] for r in rs])),
                "mean_endpoint_target_probability_mass_moved": float(np.mean([
                    r["target_probability_mass_moved"] for r in rs])),
                "mean_endpoint_collateral_kl_model_to_base_non_target": float(np.mean([
                    r["collateral_kl_model_to_base_non_target"] for r in rs])),
            }
    (out_dir / "e1_summary.json").write_text(
        json.dumps({"meta": meta, "summary": endpoint}, indent=2) + "\n")

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import analyze_collateral

    analyze_collateral.run(out_dir)
    for arm, s in endpoint.items():
        print(
            f"E1ORD_{arm.upper()} endpoint_accuracy={s['endpoint_target_accuracy']:.4f} "
            f"endpoint_target_prob={s['mean_endpoint_target_token_probability']:.6f} "
            f"endpoint_mass_moved={s['mean_endpoint_target_probability_mass_moved']:.6f} "
            f"endpoint_collateral={s['mean_endpoint_collateral_kl_model_to_base_non_target']:.6f}",
            flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
