"""Where exactly does the support-sparse fixed point stop being reachable?

The one in-repo test that drives `solve_fixed_point` to convergence
(`support_outer.rs::build_objective`) is a 2-row fixture, and its own doc comment
says why it converges: "With one active row per atom the exact ridge decoder block
makes the fitted row **interpolate the target**". Zero residual by construction.

So the prediction is a boundary at one active row per atom: at one row the decoder
block interpolates and the fixed point is trivially reachable; at two or more the
block cannot interpolate both, a genuine residual survives, and the alternation has
to actually converge.

Arm "rows_per_atom" walks that boundary on noiseless data drawn from a circle-atom
dictionary. Arm "arguments" checks that the failure is not something I induced with
the arguments I chose, by varying each of them at the smallest failing geometry.
"""

import argparse
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


def fit(gamfit, X, k, top_k, n_iter, seed, **kwargs):
    t0 = time.perf_counter()
    try:
        model = gamfit.sae_manifold_fit(
            X, K=k, d_atom=1, atom_topology="circle", assignment="topk",
            n_iter=n_iter, random_state=seed, top_k=top_k, gpu="off", **kwargs
        )
        return dict(status="ok", reconstruction_r2=float(model.reconstruction_r2()),
                    wall_s=round(time.perf_counter() - t0, 1))
    except Exception as exc:  # noqa: BLE001 - the error text is the measurement
        m = KKT.search(str(exc))
        return dict(status=type(exc).__name__,
                    raw_kkt_max=float(m.group(1)) if m else None,
                    error=str(exc)[:200], wall_s=round(time.perf_counter() - t0, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-iter", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.expanduser("~/i2502-baselines/rows_per_atom.jsonl"))
    args = ap.parse_args()

    import gamfit

    records = []
    base = dict(sparsity_weight=0.0, ard_per_atom=True)

    for p, k in ((2, 4), (4, 8)):
        for mult in (1, 2, 3, 4, 8):
            n = k * mult
            X = synth(n, p, k, 1, args.seed)
            rec = dict(arm="rows_per_atom", n=n, p=p, k=k, top_k=1,
                       rows_per_atom=mult, n_iter=args.n_iter)
            rec.update(fit(gamfit, X, k, 1, args.n_iter, args.seed, **base))
            records.append(rec)
            print("[rpa] " + json.dumps(rec), flush=True)
            with open(args.out, "w") as fh:
                for r in records:
                    fh.write(json.dumps(r) + "\n")

    # Did my own arguments induce this? Vary each at a small failing geometry.
    variants = {
        "as_measured": dict(sparsity_weight=0.0, ard_per_atom=True),
        "ard_off": dict(sparsity_weight=0.0, ard_per_atom=False),
        "default_sparsity": dict(ard_per_atom=True),
        "all_defaults": dict(),
        "smoothness_0.01": dict(sparsity_weight=0.0, ard_per_atom=True,
                                smoothness_weight=0.01),
        "smoothness_10": dict(sparsity_weight=0.0, ard_per_atom=True,
                              smoothness_weight=10.0),
    }
    X = synth(16, 2, 4, 1, args.seed)
    for name, kwargs in variants.items():
        rec = dict(arm="arguments", variant=name, n=16, p=2, k=4, top_k=1,
                   n_iter=args.n_iter, kwargs=sorted(kwargs))
        rec.update(fit(gamfit, X, 4, 1, args.n_iter, args.seed, **kwargs))
        records.append(rec)
        print("[arg] " + json.dumps(rec), flush=True)
        with open(args.out, "w") as fh:
            for r in records:
                fh.write(json.dumps(r) + "\n")

    ok = sum(1 for r in records if r["status"] == "ok")
    print(f"[done] {ok}/{len(records)} converged", flush=True)


if __name__ == "__main__":
    main()
