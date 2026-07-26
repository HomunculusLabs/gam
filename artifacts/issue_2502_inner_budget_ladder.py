"""Is the support-sparse inner fixed point budget-bound, or is it stuck?

Every #2502 support-sparse fit dies with

    SaeSupportSparseTerm::solve_fixed_point did not recur within N cycles
    (raw KKT max=...)

where N is exactly the `n_iter` the caller passed. The FFI passes one value to
both `max_outer_iter` (the REML smoothing search) and `max_inner_iter` (the
alternating decoder/coordinate fixed point), so `n_iter` is not the outer-search
budget it is documented to be.

If the inner solve is merely budget-starved, the reported raw KKT residual must
fall as n_iter rises, and the fit must eventually converge. If the KKT residual
plateaus instead, the budget is not the binding constraint and the conflation is
a real but separate defect. This ladder decides which.

Everything except n_iter is held fixed.
"""

import argparse
import json
import os
import re
import time

import numpy as np

KKT = re.compile(r"raw KKT max=([0-9.eE+-]+)")
CYCLES = re.compile(r"did not recur within (\d+) cycles")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", default=os.path.expanduser("~/i2502/prep_L16"))
    ap.add_argument("--rows", type=int, default=4000)
    ap.add_argument("--p", type=int, default=64)
    ap.add_argument("--k", type=int, default=128)
    ap.add_argument("--top-k", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ladder", default="2,4,8,16,32,64,128,256")
    ap.add_argument("--learning-rate", type=float, default=None,
                    help="maps to the inner coordinate trust radius")
    ap.add_argument("--out", default=os.path.expanduser("~/i2502-baselines/n_iter_ladder.jsonl"))
    args = ap.parse_args()

    import gamfit

    X = np.ascontiguousarray(
        np.load(f"{args.prep}/train.npy")[: args.rows, : args.p], dtype=np.float64
    )
    print(f"[ladder] gamfit {gamfit.__version__} X {X.shape} K={args.k} "
          f"top_k={args.top_k}", flush=True)

    records = []
    for n_iter in [int(v) for v in args.ladder.split(",")]:
        t0 = time.perf_counter()
        rec = dict(n_iter=n_iter, rows=args.rows, p=args.p, k=args.k,
                   top_k=args.top_k, learning_rate=args.learning_rate,
                   gamfit=gamfit.__version__)
        try:
            model = gamfit.sae_manifold_fit(
                X, K=args.k, d_atom=1, atom_topology="circle", assignment="topk",
                n_iter=n_iter, random_state=args.seed, top_k=args.top_k,
                sparsity_weight=0.0, ard_per_atom=True, gpu="off",
                **({} if args.learning_rate is None
                   else {"learning_rate": args.learning_rate}),
            )
            rec.update(status="ok", reconstruction_r2=float(model.reconstruction_r2()))
        except Exception as exc:  # noqa: BLE001 - the error text is the measurement
            text = str(exc)
            kkt = KKT.search(text)
            cycles = CYCLES.search(text)
            rec.update(
                status=type(exc).__name__,
                raw_kkt_max=float(kkt.group(1)) if kkt else None,
                cycles_reported=int(cycles.group(1)) if cycles else None,
                error=text[:400],
            )
        rec["wall_s"] = round(time.perf_counter() - t0, 1)
        records.append(rec)
        print("[ladder] " + json.dumps(rec), flush=True)
        with open(args.out, "w") as fh:
            for r in records:
                fh.write(json.dumps(r) + "\n")

    print("[ladder] DONE", flush=True)


if __name__ == "__main__":
    main()
