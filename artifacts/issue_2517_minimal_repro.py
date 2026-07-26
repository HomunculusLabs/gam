"""Smallest support-sparse manifold fit that still fails to reach its fixed point.

`probe_residual_size.py` showed the inner solve stalls even on noiseless data drawn
from a circle-atom dictionary — the model family the fit is built to recover. That
removes real activations, the harvest, the GPU and the scale of the problem from the
story, so the remaining question is how small the failing case can be made, and
whether this code path converges on anything at all.

Each cell is a fresh fit on noiseless synthetic data with the requested geometry.
A cell that returns `ok` converges; a cell that returns a raw KKT residual stalls.
"""

import argparse
import itertools
import json
import os
import re
import time

import numpy as np

KKT = re.compile(r"raw KKT max=([0-9.eE+-]+)")


def synth(n, p, k, top_k, seed):
    rng = np.random.default_rng(seed)
    decoders = rng.normal(size=(k, 3, p)) / np.sqrt(p)
    X = np.zeros((n, p))
    for row in range(n):
        for atom in rng.choice(k, size=top_k, replace=False):
            t = rng.uniform(0.0, 2.0 * np.pi)
            X[row] += np.array([1.0, np.cos(t), np.sin(t)]) @ decoders[atom]
    return np.ascontiguousarray(X)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-iter", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.expanduser("~/i2502-baselines/minimal_repro.jsonl"))
    args = ap.parse_args()

    import gamfit

    cells = []
    for p, k in ((2, 4), (4, 8), (8, 16), (16, 32)):
        for n in (25, 100, 400):
            for top_k in (1, 2):
                if top_k > k:
                    continue
                cells.append((n, p, k, top_k))

    records = []
    for n, p, k, top_k in cells:
        X = synth(n, p, k, top_k, args.seed)
        rec = dict(n=n, p=p, k=k, top_k=top_k, n_iter=args.n_iter,
                   gamfit=gamfit.__version__)
        t0 = time.perf_counter()
        try:
            model = gamfit.sae_manifold_fit(
                X, K=k, d_atom=1, atom_topology="circle", assignment="topk",
                n_iter=args.n_iter, random_state=args.seed, top_k=top_k,
                sparsity_weight=0.0, ard_per_atom=True, gpu="off",
            )
            rec.update(status="ok",
                       reconstruction_r2=float(model.reconstruction_r2()))
        except Exception as exc:  # noqa: BLE001 - the error text is the measurement
            m = KKT.search(str(exc))
            rec.update(status=type(exc).__name__,
                       raw_kkt_max=float(m.group(1)) if m else None,
                       error=str(exc)[:200])
        rec["wall_s"] = round(time.perf_counter() - t0, 1)
        records.append(rec)
        print("[cell] " + json.dumps(rec), flush=True)
        with open(args.out, "w") as fh:
            for r in records:
                fh.write(json.dumps(r) + "\n")

    ok = sum(1 for r in records if r["status"] == "ok")
    print(f"[done] {ok}/{len(records)} cells converged", flush=True)


if __name__ == "__main__":
    main()
