"""Is the support-sparse convergence test scale-free, or does it depend on units?

`solve_fixed_point` compares `raw_stationarity().max_abs()` against an absolute
tolerance (1e-6, hardcoded at the FFI boundary). That residual is the gradient of
the unnormalised objective: the decoder block sums `phi * residual` over every row
assigned to an atom, and the residual carries the units of the data.

If the test were scale-free, neither multiplying the activations by a constant nor
adding more rows would change the residual it reports. This measures both.

  arm "scale": same rows, activations multiplied by s
  arm "rows" : same activations, more rows

A residual proportional to s means whether a fit is declared converged depends on
the units the activations were stored in. A residual growing with the row count
means the same model on more data is judged further from stationary.
"""

import argparse
import json
import os
import re
import time

import numpy as np

KKT = re.compile(r"raw KKT max=([0-9.eE+-]+)")


def run(gamfit, X, k, top_k, n_iter, seed):
    t0 = time.perf_counter()
    try:
        model = gamfit.sae_manifold_fit(
            X, K=k, d_atom=1, atom_topology="circle", assignment="topk",
            n_iter=n_iter, random_state=seed, top_k=top_k,
            sparsity_weight=0.0, ard_per_atom=True, gpu="off",
        )
        return dict(status="ok", reconstruction_r2=float(model.reconstruction_r2()),
                    wall_s=round(time.perf_counter() - t0, 1))
    except Exception as exc:  # noqa: BLE001 - the error text is the measurement
        m = KKT.search(str(exc))
        return dict(status=type(exc).__name__,
                    raw_kkt_max=float(m.group(1)) if m else None,
                    error=str(exc)[:200],
                    wall_s=round(time.perf_counter() - t0, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", default=os.path.expanduser("~/i2502/prep_L16"))
    ap.add_argument("--p", type=int, default=64)
    ap.add_argument("--k", type=int, default=128)
    ap.add_argument("--top-k", type=int, default=4)
    ap.add_argument("--n-iter", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rows", type=int, default=4000)
    ap.add_argument("--out", default=os.path.expanduser("~/i2502-baselines/kkt_scale.jsonl"))
    args = ap.parse_args()

    import gamfit

    full = np.ascontiguousarray(
        np.load(f"{args.prep}/train.npy")[:, : args.p], dtype=np.float64
    )
    records = []

    for s in (0.01, 0.1, 1.0, 10.0):
        X = np.ascontiguousarray(full[: args.rows] * s)
        rec = dict(arm="scale", scale=s, rows=args.rows, p=args.p, k=args.k,
                   n_iter=args.n_iter, rms_row_norm=float(np.sqrt((X ** 2).sum(1).mean())))
        rec.update(run(gamfit, X, args.k, args.top_k, args.n_iter, args.seed))
        records.append(rec)
        print("[scale] " + json.dumps(rec), flush=True)
        with open(args.out, "w") as fh:
            for r in records:
                fh.write(json.dumps(r) + "\n")

    for n in (500, 1000, 2000, 4000, 8000):
        X = np.ascontiguousarray(full[:n])
        rec = dict(arm="rows", scale=1.0, rows=n, p=args.p, k=args.k,
                   n_iter=args.n_iter, rms_row_norm=float(np.sqrt((X ** 2).sum(1).mean())))
        rec.update(run(gamfit, X, args.k, args.top_k, args.n_iter, args.seed))
        records.append(rec)
        print("[rows] " + json.dumps(rec), flush=True)
        with open(args.out, "w") as fh:
            for r in records:
                fh.write(json.dumps(r) + "\n")

    print("[done]", flush=True)


if __name__ == "__main__":
    main()
