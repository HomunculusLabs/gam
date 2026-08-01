#!/usr/bin/env python3
"""Chart-configuration probe for gam#2234 E1 (harvest once, sweep cheaply).

The first real-model E1 run (Qwen3.5-4B-Base L16, 2026-07-31) minted a fit — the
historical `1e12` wall is gone — but the fitted chart was empty: `fit_ev`
0.0152 and weekday circular `R2` **0.0000**, so every steering delta was ~1e-9
and both arms measured zero. The cause is not the solver: a 70-row cloud built
from 10 prompt templates × 7 weekdays has its leading singular directions
spanned by TEMPLATE identity, and a PCA-64 chart of 70 rows is then essentially
noise. The calendar circle is a rank-2 object hiding under that.

This script separates the two costs so the chart config can be chosen on
evidence rather than guessed at 10 minutes per guess:

  `harvest`  one GPU pass; dumps the ambient activation cloud + labels + template
             indices to an `.npz` (no fit).
  `sweep`    CPU only; for each (per-template-centering, PCA dim) cell, fits
             `gamfit.sae_manifold_fit` exactly as the E1 harness does and reports
             the fit EV together with how much of the *known* structure the
             fitted coordinate recovers (circular R² against the day-of-week
             phase for the cyclic ladder, linear R² against rank for the ordinal
             one). That recovery number — not EV — is what decides whether E1 can
             measure anything at all.

    python3 experiments/steering_e1/probe_chart_2234.py harvest \
        --structure weekday --model Qwen/Qwen3.5-4B-Base --layer-index 16 \
        --out cloud_cyclic.npz
    python3 experiments/steering_e1/probe_chart_2234.py sweep \
        --npz cloud_weekday.npz --structure weekday --pca-dims 2,3,4,6,8,12
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np

TAU = 2.0 * math.pi


def _sibling(name: str):
    if name in sys.modules:
        return sys.modules[name]
    path = Path(__file__).resolve().parent / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load sibling harness at {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def harvest(args) -> int:
    E1 = _sibling("run_e1")
    structure = E1.STRUCTURES[args.structure]
    model, tok = E1.load_model_and_tokenizer(args.model, args.cache_dir, args.dtype)
    layers = E1.resolve_layers(model)
    layer = layers[args.layer_index]

    templates, labels = structure.fit_templates(), structure.labels
    label_ids = E1.label_token_ids(tok, labels, args.candidate_prefix)

    rows, label_idx, template_idx = [], [], []
    for ti, template in enumerate(templates):
        for li, label in enumerate(labels):
            if args.capture_at == "label":
                ids, pos = E1.build_label_prompt(tok, template, label_ids[li])
                act, _ = E1.run_clean_at(model, layer, ids, pos)
            else:
                act, _ = E1.run_clean(model, tok, layer, template.format(label=label))
            rows.append(act.numpy().astype(np.float64))
            label_idx.append(li)
            template_idx.append(ti)
    X = np.ascontiguousarray(np.stack(rows))
    np.savez(
        args.out, X=X,
        label_index=np.asarray(label_idx), template_index=np.asarray(template_idx),
        model=np.asarray(args.model), layer_index=np.asarray(args.layer_index),
        structure=np.asarray(args.structure),
    )
    print(f"harvested {X.shape} -> {args.out}", flush=True)
    return 0


def structure_recovery(coord: np.ndarray, label_idx: np.ndarray, cyclic: bool,
                       n_labels: int) -> float:
    """How much of the KNOWN generator the fitted 1-D coordinate recovers.

    Delegates to `run_e1` so the probe and the harness can never disagree about
    what "recovery" means. The chart PERIOD is a property of the fitted object,
    not of this script, so both the period-one and the period-2*pi conventions
    are scored and the better is kept: reading a good chart in the wrong unit
    winds the phase 2*pi times too fast and returns R^2 == 0, which is
    indistinguishable from "the fit found nothing".
    """
    E1 = _sibling("run_e1")
    if cyclic:
        return max(E1.circular_recovery(coord, label_idx, n_labels, p)[0]
                   for p in (1.0, TAU))
    return E1.linear_recovery(coord, label_idx)[0]


def sweep(args) -> int:
    import gamfit

    data = np.load(args.npz, allow_pickle=False)
    X0, label_idx, template_idx = data["X"], data["label_index"], data["template_index"]
    E1 = _sibling("run_e1")
    structure = E1.STRUCTURES[args.structure]
    n_labels = int(label_idx.max()) + 1
    topology = structure.topology
    dims = [int(v) for v in args.pca_dims.split(",") if v.strip()]
    k_atoms = [int(v) for v in args.k_atoms.split(",") if v.strip()]

    results = []
    for center in (False, True):
        X = X0.copy()
        if center:
            for ti in np.unique(template_idx):
                rows = np.flatnonzero(template_idx == ti)
                X[rows] -= X[rows].mean(0, keepdims=True)
        mu = X.mean(0, keepdims=True)
        Xc = X - mu
        _, svals, vt = np.linalg.svd(Xc, full_matrices=False)
        for dim in dims:
            r = int(min(dim, vt.shape[0]))
            Xr = np.ascontiguousarray(Xc @ vt[:r].T)
            evr = float((svals[:r] ** 2).sum() / max((svals**2).sum(), 1e-30))
            for K in k_atoms:
                try:
                    fit = gamfit.sae_manifold_fit(
                        Xr, K=K, d_atom=1, atom_topology=topology, assignment="softmax",
                        n_iter=args.n_iter, random_state=args.seed)
                    fit_ev = float(
                        1.0 - np.sum((Xr - np.asarray(fit.fitted)) ** 2)
                        / max(np.sum((Xr - Xr.mean(0)) ** 2), 1e-30))
                    recovery = -1.0
                    for k in range(K):
                        c = np.asarray(fit.coords[k], dtype=float)
                        coord = c[:, 0] if c.ndim == 2 else c
                        recovery = max(recovery, structure_recovery(
                            coord, label_idx, structure.cyclic, n_labels))
                    status = "ok"
                except Exception as error:  # a refusal is a datum, not a crash
                    fit_ev, recovery = float("nan"), float("nan")
                    status = f"{type(error).__name__}: {error}"
                row = {
                    "per_template_center": center, "pca_dim": r, "k_atoms": K,
                    "pca_explained_variance": evr, "fit_ev": fit_ev,
                    "structure_recovery_r2": recovery, "status": status[:200],
                }
                results.append(row)
                print(
                    f"center={int(center)} dim={r:3d} K={K} pca_evr={evr:.4f} "
                    f"fit_ev={fit_ev:.4f} recovery_r2={recovery:.4f} {row['status']}",
                    flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2) + "\n")
    ok = [r for r in results if np.isfinite(r["structure_recovery_r2"])]
    if ok:
        best = max(ok, key=lambda r: r["structure_recovery_r2"])
        print(f"BEST center={int(best['per_template_center'])} dim={best['pca_dim']} "
              f"K={best['k_atoms']} recovery_r2={best['structure_recovery_r2']:.4f} "
              f"fit_ev={best['fit_ev']:.4f}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    h = sub.add_parser("harvest")
    h.add_argument("--model", default="Qwen/Qwen3.5-4B-Base")
    h.add_argument("--cache-dir", default="")
    h.add_argument("--layer-index", type=int, default=16)
    h.add_argument("--structure", choices=("weekday", "ordinal"), default="weekday")
    h.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="fp32")
    h.add_argument("--capture-at", choices=("last", "label"), default="label",
                   help="'label' captures at the label token's own position (the representation); "
                        "'last' captures at the final token (a downstream trace, measured to carry "
                        "0.91%% of the cloud variance on Qwen3.5-4B-Base L16)")
    h.add_argument("--candidate-prefix", default=" ")
    h.add_argument("--out", required=True)
    h.set_defaults(func=harvest)

    s = sub.add_parser("sweep")
    s.add_argument("--npz", required=True)
    s.add_argument("--structure", choices=("weekday", "ordinal"), default="weekday")
    s.add_argument("--pca-dims", default="2,3,4,6,8,12,16")
    s.add_argument("--k-atoms", default="1,2")
    s.add_argument("--n-iter", type=int, default=60)
    s.add_argument("--seed", type=int, default=20260731)
    s.add_argument("--out", default="")
    s.set_defaults(func=sweep)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
