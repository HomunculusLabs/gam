"""What the row-level train/test split was worth: #2502 split-contract measurement.

The #2502 harvest is 600,064 token rows drawn from 5,774 wikitext documents. The
prep in use until now split those rows by a uniform random permutation, so tokens
of the same document — often adjacent positions of the same sentence — sat on both
sides of the held-out boundary. This script prices that choice.

Both arms are byte-identical in every respect except which rows are held out:
same harvest, same pool sizes, same sink peel, same norm guard, same train-only
PCA chart width, same models, same seeds. The manipulated variable is the split
rule alone.

  arm "row" : 60,000 rows sampled uniformly, permuted, cut 50,000 / 10,000
              (the rule the campaign has been using)
  arm "doc" : 50,000 rows sampled from train documents and 10,000 from held-out
              documents, documents assigned by issue_2502_doc_split (the contract)

Scored models: PCA-M subspace reconstruction (deterministic, no seed) and the
TopK SAE of Gao et al. 2024 at K=32,000 / top_k=8, three seeds per arm.

Explained variance is reported both in the chart the arm built and in the ambient
2560-d residual stream, so the comparison does not depend on the arm's own chart.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from issue_2502_doc_split import assign_documents, split_manifest  # noqa: E402


def ev_of(X, R):
    return 1.0 - float(((X - R) ** 2).sum()) / float((X ** 2).sum())


def absorbed_r2(Xc, Z):
    B = np.linalg.pinv(Z.T @ Z) @ (Z.T @ Xc)
    return 1.0 - float(((Xc - Z @ B) ** 2).sum()) / float((Xc ** 2).sum())


def build_arm(name, rows_train, rows_test, X, pos, p):
    """Sink peel, norm guard and PCA chart, all estimated on train rows only."""
    tr = np.asarray(X[rows_train], dtype=np.float64)
    te = np.asarray(X[rows_test], dtype=np.float64)
    ptr, pte = pos[rows_train], pos[rows_test]

    m0 = tr.mean(0)
    tr, te = tr - m0, te - m0
    Ztr = np.column_stack([np.ones(len(tr)), (ptr == 0).astype(np.float64)])
    B = np.linalg.pinv(Ztr.T @ Ztr) @ (Ztr.T @ tr)
    tr = tr - Ztr @ B
    te = te - np.column_stack([np.ones(len(te)), (pte == 0).astype(np.float64)]) @ B

    ln = np.log(np.maximum(np.linalg.norm(tr, axis=1), 1e-12))
    med = np.median(ln)
    sig = (np.quantile(ln, 0.84) - np.quantile(ln, 0.16)) / 2.0
    keep_tr = np.abs(ln - med) < 6.0 * sig
    lnte = np.log(np.maximum(np.linalg.norm(te, axis=1), 1e-12))
    keep_te = np.abs(lnte - med) < 6.0 * sig
    tr, te = tr[keep_tr], te[keep_te]
    rows_train, rows_test = rows_train[keep_tr], rows_test[keep_te]

    mu = tr.mean(0, keepdims=True)
    tr, te = tr - mu, te - mu
    # Exact PCA via the train Gram matrix in feature space (2560 x 2560).
    evals, evecs = np.linalg.eigh(tr.T @ tr)
    order = np.argsort(evals)[::-1]
    evals, evecs = evals[order], evecs[:, order]
    lift = np.ascontiguousarray(evecs[:, :p].T)
    ev_frac = float(evals[:p].sum() / evals.sum())

    return dict(
        name=name,
        chart_train=np.ascontiguousarray(tr @ lift.T),
        chart_test=np.ascontiguousarray(te @ lift.T),
        ambient_test=te,
        lift=lift,
        rows_train=rows_train,
        rows_test=rows_test,
        pca_ev_frac=ev_frac,
        n_train=int(len(tr)),
        n_test=int(len(te)),
    )


def topk_sae(arm, K, top_k, epochs, batch, lr, seed):
    import torch

    torch.manual_seed(seed)
    dev = "cuda:0"
    Xtr = torch.from_numpy(arm["chart_train"]).float().to(dev)
    Xte = torch.from_numpy(arm["chart_test"]).float().to(dev)
    N, D = Xtr.shape

    W_enc = torch.nn.Parameter(torch.randn(D, K, device=dev) * (1.0 / np.sqrt(D)))
    W_dec = torch.nn.Parameter(W_enc.detach().T.clone())
    b_pre = torch.nn.Parameter(Xtr.mean(0).clone())
    b_enc = torch.nn.Parameter(torch.zeros(K, device=dev))
    opt = torch.optim.Adam([W_enc, W_dec, b_pre, b_enc], lr=lr)

    def forward(xb):
        pre = (xb - b_pre) @ W_enc + b_enc
        vals, idx = torch.topk(pre, top_k, dim=1)
        z = torch.zeros_like(pre).scatter_(1, idx, torch.relu(vals))
        return z @ W_dec + b_pre, z

    t0 = time.perf_counter()
    fired = torch.zeros(K, dtype=torch.bool, device=dev)
    for _ in range(epochs):
        perm = torch.randperm(N, device=dev)
        for i in range(0, N, batch):
            xb = Xtr[perm[i : i + batch]]
            recon, z = forward(xb)
            loss = ((recon - xb) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            with torch.no_grad():
                W_dec.data /= W_dec.data.norm(dim=1, keepdim=True).clamp_min(1e-8)
                fired |= (z != 0).any(0)
    wall = time.perf_counter() - t0

    with torch.no_grad():
        recs, nzs = [], []
        for i in range(0, len(Xte), 2048):
            r, z = forward(Xte[i : i + 2048])
            recs.append(r)
            nzs.append(z != 0)
        R = torch.cat(recs).cpu().numpy().astype(np.float64)
        nz = torch.cat(nzs)
        alive = int(nz.any(0).sum().item())
        l0 = float(nz.float().sum(1).mean().item())
        r_tr = []
        for i in range(0, len(Xtr), 2048):
            r, _ = forward(Xtr[i : i + 2048])
            r_tr.append(r)
        R_tr = torch.cat(r_tr).cpu().numpy().astype(np.float64)

    amb_recon = R @ arm["lift"]
    return dict(
        wall_s=round(wall, 1),
        train_ev_chart=ev_of(arm["chart_train"], R_tr),
        test_ev_chart=ev_of(arm["chart_test"], R),
        test_ev_ambient=ev_of(arm["ambient_test"], amb_recon),
        test_mean_l0=round(l0, 2),
        alive_atoms_test=alive,
        ever_fired_train=int(fired.sum().item()),
    )


def nearest_train_neighbour(arm, doc_ids):
    """Median distance from a test row to its nearest train row, and how often
    that neighbour comes from the test row's own document."""
    import torch

    dev = "cuda:0"
    A = torch.from_numpy(arm["chart_test"]).float().to(dev)
    B = torch.from_numpy(arm["chart_train"]).float().to(dev)
    dtr = torch.from_numpy(doc_ids[arm["rows_train"]].astype(np.int64)).to(dev)
    dte = doc_ids[arm["rows_test"]].astype(np.int64)
    scale = float(B.pow(2).sum(1).mean().sqrt().item())

    best_d, best_i = [], []
    for i in range(0, len(A), 512):
        d = torch.cdist(A[i : i + 512], B)
        v, j = d.min(dim=1)
        best_d.append(v)
        best_i.append(dtr[j])
    best_d = torch.cat(best_d).cpu().numpy()
    nn_doc = torch.cat(best_i).cpu().numpy()
    same = nn_doc == dte
    return dict(
        median_nn_distance=float(np.median(best_d)),
        median_nn_distance_relative=float(np.median(best_d) / scale),
        frac_nn_same_document=float(same.mean()),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--harvest", default=os.path.expanduser("~/i2502/harvest"))
    ap.add_argument("--layer", type=int, default=16)
    ap.add_argument("--p", type=int, default=128)
    ap.add_argument("--n-train", type=int, default=50000)
    ap.add_argument("--n-test", type=int, default=10000)
    ap.add_argument("--k", type=int, default=32000)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--pool-seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.expanduser("~/i2502-baselines/split_leakage.jsonl"))
    args = ap.parse_args()

    X = np.load(f"{args.harvest}/resid_L{args.layer}.npy", mmap_mode="r")
    doc_ids = np.load(f"{args.harvest}/doc_ids.npy")
    pos = np.load(f"{args.harvest}/pos_in_seq.npy")
    n = X.shape[0]

    manifest = split_manifest(doc_ids)
    print("[split] " + json.dumps(manifest), flush=True)

    rng = np.random.default_rng(args.pool_seed)
    take = args.n_train + args.n_test
    pool = np.sort(rng.choice(n, size=take, replace=False))
    perm = rng.permutation(take)
    rows = {
        "row": (np.sort(pool[perm[: args.n_train]]), np.sort(pool[perm[args.n_train :]])),
    }

    test_docs, train_docs = assign_documents(doc_ids)
    is_test_doc = np.isin(doc_ids, test_docs)
    train_pool = np.flatnonzero(~is_test_doc)
    test_pool = np.flatnonzero(is_test_doc)
    rng2 = np.random.default_rng(args.pool_seed)
    rows["doc"] = (
        np.sort(rng2.choice(train_pool, size=args.n_train, replace=False)),
        np.sort(rng2.choice(test_pool, size=args.n_test, replace=False)),
    )
    print(
        f"[split] doc arm draws from {len(train_pool)} train-doc rows and "
        f"{len(test_pool)} held-out-doc rows",
        flush=True,
    )

    out_records = [dict(record="split_manifest", **manifest)]
    for name in ("row", "doc"):
        rtr, rte = rows[name]
        arm = build_arm(name, rtr, rte, X, pos, args.p)
        base = dict(
            arm=name,
            layer=args.layer,
            p=args.p,
            n_train=arm["n_train"],
            n_test=arm["n_test"],
            pca_ev_frac=arm["pca_ev_frac"],
        )
        n_docs_tr = len(np.unique(doc_ids[arm["rows_train"]]))
        n_docs_te = len(np.unique(doc_ids[arm["rows_test"]]))
        shared = len(
            np.intersect1d(
                np.unique(doc_ids[arm["rows_train"]]), np.unique(doc_ids[arm["rows_test"]])
            )
        )
        rec = dict(record="arm_geometry", **base, n_docs_train=n_docs_tr,
                   n_docs_test=n_docs_te, n_docs_shared=shared,
                   **nearest_train_neighbour(arm, doc_ids))
        out_records.append(rec)
        print("[arm] " + json.dumps(rec), flush=True)

        Xtr, Xte = arm["chart_train"], arm["chart_test"]
        mu = Xtr.mean(0)
        evals, evecs = np.linalg.eigh((Xtr - mu).T @ (Xtr - mu))
        V_all = evecs[:, np.argsort(evals)[::-1]].T
        for M in (args.top_k, 16, 64, 256):
            if M > args.p:
                continue
            V = V_all[:M]
            R = ((Xte - mu) @ V.T) @ V + mu
            rec = dict(record=f"pca_M{M}", **base, test_ev_chart=ev_of(Xte, R),
                       test_ev_ambient=ev_of(arm["ambient_test"], R @ arm["lift"]),
                       test_mean_l0=M)
            out_records.append(rec)
            print("[pca] " + json.dumps(rec), flush=True)

        for seed in range(args.seeds):
            res = topk_sae(arm, args.k, args.top_k, args.epochs, args.batch, args.lr, seed)
            rec = dict(record="topk_sae", **base, k=args.k, top_k=args.top_k,
                       epochs=args.epochs, seed=seed, **res)
            out_records.append(rec)
            print("[topk] " + json.dumps(rec), flush=True)
            with open(args.out, "w") as fh:
                for r in out_records:
                    fh.write(json.dumps(r) + "\n")

    with open(args.out, "w") as fh:
        for r in out_records:
            fh.write(json.dumps(r) + "\n")
    print("[done] wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
