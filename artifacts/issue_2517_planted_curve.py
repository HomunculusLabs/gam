"""What separates the converging in-repo fixtures from every fit that stalls?

`tiered/fit.rs` has two tests that drive the SAME support-sparse engine to a
recurred fixed point and assert it: `tier2_branch_constructs_the_support_sparse_path`
(n=96, P=4, K=8, support_k=2) and `coordinate_seed_carries_a_full_tiered_fit`
(n=240, P=16, K=24). So the engine does converge on something, and the question is
what their data has that mine does not.

Two candidate differences, tested as a 2x2:

  DATA     - their rows are a deterministic one-parameter curve: every row is a
             function of a single scalar phase `i * step`, so the whole cloud lies
             on a 1-D curve. My in-class generator draws an independent angle per
             atom per row, so the cloud fills a much larger set.
  ENTRY    - they reach the engine through the tiered driver, which fits Tier-2 on
             the Tier-1 residual with `max_inner_iter` decoupled from the outer
             budget. The public `sae_manifold_fit` entry has neither property.

Arm "planted_curve" replays their exact geometry and their exact data recipe
through the PUBLIC entry. If it converges, the discriminator is the data. If it
stalls, the discriminator is the entry point, and the two passing tests certify a
route that the public API cannot reach.
"""

import argparse
import json
import os
import re
import time

import numpy as np

KKT = re.compile(r"raw KKT max=([0-9.eE+-]+)")


def planted_circles(n, p, step, freqs):
    """The tiered tests' own recipe: one scalar phase per row, circles from it."""
    z = np.zeros((n, p))
    for i in range(n):
        ph = i * step
        for c, f in enumerate(freqs):
            z[i, 2 * c] = np.cos(f * ph + c)
            z[i, 2 * c + 1] = np.sin(f * ph + c)
    return np.ascontiguousarray(z)


def independent_angles(n, p, k, top_k, seed):
    """My generator: an independent angle per atom per row."""
    rng = np.random.default_rng(seed)
    decoders = rng.normal(size=(k, 3, p)) / np.sqrt(p)
    X = np.zeros((n, p))
    for row in range(n):
        for atom in rng.choice(k, size=top_k, replace=False):
            t = rng.uniform(0.0, 2.0 * np.pi)
            X[row] += np.array([1.0, np.cos(t), np.sin(t)]) @ decoders[atom]
    return np.ascontiguousarray(X)


def fit(gamfit, X, k, top_k, n_iter, seed):
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
                    error=str(exc)[:200], wall_s=round(time.perf_counter() - t0, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-iter", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.expanduser("~/i2502-baselines/planted_curve.jsonl"))
    args = ap.parse_args()

    import gamfit

    cases = [
        # (label, X, K, support_k) — the two tiered fixtures, replayed exactly.
        ("tier2_branch_n96_p4_K8", planted_circles(96, 4, 0.19, (1.0, 1.7)), 8, 2),
        ("coordinate_seed_n240_p16_K24",
         planted_circles(240, 16, 0.261799, tuple(1.0 + c * 0.37 for c in range(8))), 24, 2),
    ]
    records = []
    for label, X, k, top_k in cases:
        rec = dict(arm="planted_curve", case=label, n=len(X), p=X.shape[1],
                   k=k, top_k=top_k, n_iter=args.n_iter)
        rec.update(fit(gamfit, X, k, top_k, args.n_iter, args.seed))
        records.append(rec)
        print("[planted] " + json.dumps(rec), flush=True)

    # Same geometry, my generator: isolates the data recipe from the geometry.
    for label, n, p, k, top_k in (("n96_p4_K8", 96, 4, 8, 2),
                                  ("n240_p16_K24", 240, 16, 24, 2)):
        X = independent_angles(n, p, k, top_k, args.seed)
        rec = dict(arm="independent_angles", case=label, n=n, p=p, k=k,
                   top_k=top_k, n_iter=args.n_iter)
        rec.update(fit(gamfit, X, k, top_k, args.n_iter, args.seed))
        records.append(rec)
        print("[indep] " + json.dumps(rec), flush=True)

    with open(args.out, "w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    ok = sum(1 for r in records if r["status"] == "ok")
    print(f"[done] {ok}/{len(records)} converged", flush=True)


if __name__ == "__main__":
    main()
