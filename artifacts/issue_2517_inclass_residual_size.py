"""Does the inner solve stall because the residual is large?

The inner coordinate step is a Gauss-Newton trust-region step: `coordinate_sweep`
builds its Hessian model as `J Jᵀ` plus an ARD majorizer, with no term for the
curvature of the atom manifold times the residual. That model is accurate only
when the residual is small. On real activations at top_k=4 the model explains
roughly half the variance, so the residual is emphatically not small, and a bad
Hessian model forces the Armijo line search onto ever-shorter accepted steps —
which decreases the objective while leaving the gradient where it is. That is
exactly the plateau the budget ladder found.

If that is the mechanism, then feeding the SAME code path data drawn from its own
model family, where the achievable residual is set by a noise level I control,
should converge at low noise and reproduce the plateau at high noise.

Each arm reports the raw KKT residual the solver stalls at, against the noise
level of the data it was given. Nothing but the data changes.
"""

import argparse
import json
import os
import re
import time

import numpy as np

KKT = re.compile(r"raw KKT max=([0-9.eE+-]+)")


def synth(n, p, k, top_k, noise, seed):
    """Rows built from a circle-atom dictionary: each row picks `top_k` atoms,
    each contributing decoder(atom) applied to [1, cos t, sin t] at its own angle."""
    rng = np.random.default_rng(seed)
    decoders = rng.normal(size=(k, 3, p)) / np.sqrt(p)
    X = np.zeros((n, p))
    for row in range(n):
        for atom in rng.choice(k, size=top_k, replace=False):
            t = rng.uniform(0.0, 2.0 * np.pi)
            X[row] += np.array([1.0, np.cos(t), np.sin(t)]) @ decoders[atom]
    signal_rms = float(np.sqrt((X ** 2).sum(1).mean()))
    X = X + rng.normal(scale=noise * signal_rms / np.sqrt(p), size=X.shape)
    return np.ascontiguousarray(X), signal_rms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--p", type=int, default=64)
    ap.add_argument("--k", type=int, default=128)
    ap.add_argument("--top-k", type=int, default=4)
    ap.add_argument("--n-iter", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--noise", default="0.0,0.01,0.05,0.2,0.5,1.0")
    ap.add_argument("--out", default=os.path.expanduser("~/i2502-baselines/residual_size.jsonl"))
    args = ap.parse_args()

    import gamfit

    records = []
    for noise in [float(v) for v in args.noise.split(",")]:
        X, signal_rms = synth(args.n, args.p, args.k, args.top_k, noise, args.seed)
        rec = dict(noise=noise, n=args.n, p=args.p, k=args.k, top_k=args.top_k,
                   n_iter=args.n_iter, signal_rms=signal_rms,
                   gamfit=gamfit.__version__)
        t0 = time.perf_counter()
        try:
            model = gamfit.sae_manifold_fit(
                X, K=args.k, d_atom=1, atom_topology="circle", assignment="topk",
                n_iter=args.n_iter, random_state=args.seed, top_k=args.top_k,
                sparsity_weight=0.0, ard_per_atom=True, gpu="off",
            )
            rec.update(status="ok",
                       reconstruction_r2=float(model.reconstruction_r2()))
        except Exception as exc:  # noqa: BLE001 - the error text is the measurement
            m = KKT.search(str(exc))
            rec.update(status=type(exc).__name__,
                       raw_kkt_max=float(m.group(1)) if m else None,
                       error=str(exc)[:220])
        rec["wall_s"] = round(time.perf_counter() - t0, 1)
        records.append(rec)
        print("[resid] " + json.dumps(rec), flush=True)
        with open(args.out, "w") as fh:
            for r in records:
                fh.write(json.dumps(r) + "\n")

    print("[done]", flush=True)


if __name__ == "__main__":
    main()
