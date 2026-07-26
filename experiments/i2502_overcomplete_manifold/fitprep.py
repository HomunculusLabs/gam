"""Prep Qwen3.5-4B harvest for the #2502 overcomplete manifold-dictionary fit.

Steps (prior-art path from experiments/real_manifold_sae):
  1. load resid_L{L}, drop rows with pos_in_seq==0 IF the pos0 indicator absorbs
     variance beyond a permutation null (sink peel, causal test);
  2. center on train rows, PCA to P dims (train-only), save train/test + lift;
  3. save row->token metadata aligned with the kept rows.
"""
import argparse, json, os
import numpy as np


def absorbed_r2(Xc, Z):
    B = np.linalg.pinv(Z.T @ Z) @ (Z.T @ Xc)
    resid = Xc - Z @ B
    return 1.0 - float((resid ** 2).sum()) / float((Xc ** 2).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--harvest", default=os.path.expanduser("~/i2502/harvest"))
    ap.add_argument("--layer", type=int, default=16)
    ap.add_argument("--n-train", type=int, default=50000)
    ap.add_argument("--n-test", type=int, default=10000)
    ap.add_argument("--p", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--null-reps", type=int, default=50)
    ap.add_argument("--out", default=os.path.expanduser("~/i2502/prep_L16"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    X = np.load(f"{args.harvest}/resid_L{args.layer}.npy", mmap_mode="r")
    pos = np.load(f"{args.harvest}/pos_in_seq.npy")
    n = X.shape[0]
    rng = np.random.default_rng(args.seed)
    take = args.n_train + args.n_test
    idx = np.sort(rng.choice(n, size=min(take, n), replace=False))
    Xs = np.asarray(X[idx], dtype=np.float64)
    ps = pos[idx]

    # sink peel: pos0 indicator, causal permutation test (prior-art design)
    m0 = Xs.mean(0)
    Xc = Xs - m0
    Z = np.column_stack([np.ones(len(Xc)), (ps == 0).astype(np.float64)])
    obs = absorbed_r2(Xc, Z)
    null = [absorbed_r2(Xc, Z[rng.permutation(len(Z))]) for _ in range(args.null_reps)]
    peel = obs > max(null)
    B = np.zeros((2, Xc.shape[1]))
    if peel:
        B = np.linalg.pinv(Z.T @ Z) @ (Z.T @ Xc)
        Xc = Xc - Z @ B
    print(f"[prep] pos0 peel: causal={peel} obs={obs:.5f} null_max={max(null):.5f}", flush=True)

    # extreme-norm guard: drop rows beyond 6 sigma of log-norm (activation sinks)
    norms = np.linalg.norm(Xc, axis=1)
    ln = np.log(np.maximum(norms, 1e-12))
    keep = np.abs(ln - np.median(ln)) < 6.0 * (np.quantile(ln, 0.84) - np.quantile(ln, 0.16)) / 2.0
    print(f"[prep] norm guard keeps {keep.sum()}/{len(keep)}", flush=True)
    Xc, idx, ps = Xc[keep], idx[keep], ps[keep]

    perm = rng.permutation(len(Xc))
    tr, te = perm[: args.n_train], perm[args.n_train : args.n_train + args.n_test]
    mu = Xc[tr].mean(0, keepdims=True)
    Xtr_c = Xc[tr] - mu
    _, s, vt = np.linalg.svd(Xtr_c, full_matrices=False)
    r = min(args.p, vt.shape[0])
    lift = np.ascontiguousarray(vt[:r])
    train = np.ascontiguousarray(Xtr_c @ lift.T)
    test = np.ascontiguousarray((Xc[te] - mu) @ lift.T)
    evr = float((s[:r] ** 2).sum() / (s ** 2).sum())
    print(f"[prep] PCA chart P={r} train {train.shape} test {test.shape} ev_frac={evr:.4f}",
          flush=True)

    np.save(f"{args.out}/train.npy", train)
    np.save(f"{args.out}/test.npy", test)
    np.save(f"{args.out}/lift.npy", lift)
    np.save(f"{args.out}/mu.npy", mu)
    # affine splice map: ambient x at pos p>0 maps to chart via (x - c0) @ lift.T,
    # pos0 via (x - c1); inverse lift adds the same constant back.
    c0 = (m0 + B[0] + mu.ravel())
    c1 = c0 + B[1]
    np.save(f"{args.out}/c0.npy", c0)
    np.save(f"{args.out}/c1.npy", c1)
    np.save(f"{args.out}/rows_train.npy", idx[tr])   # row index into harvest arrays
    np.save(f"{args.out}/rows_test.npy", idx[te])
    meta = dict(layer=args.layer, p=int(r), pca_ev_frac=evr, peel_causal=bool(peel),
                peel_obs=obs, n_train=int(len(tr)), n_test=int(len(te)), seed=args.seed)
    json.dump(meta, open(f"{args.out}/meta.json", "w"), indent=2)
    print("[prep] DONE", json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
