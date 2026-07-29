"""The SAE-duality test, end to end, on the pure-ReLU LM.

THE QUESTION. Sparse autoencoders (and our curved dictionaries) learn their
features by fitting activations. The tomograph reads the model's features
directly from its weights — every ReLU on/off wall, with the write direction
u it adds to the residual stream when crossed. The duality conjecture says
these must agree: SAE decoder directions should line up with (clusters of)
wall write-directions, because creases in the activation cloud are what
sparse reconstruction spends its atoms on. If they don't agree, dictionary
features are artifacts of fitting, not invariants of the model.

THE TEST, concretely:
  stage harvest : run the ReLU LM over corpus text and save the final
                  residual vector (the space the tomograph's u lives in) at
                  every position — the training data for the dictionaries.
  stage train   : fit a flat TopK dictionary and a curved a·(U+tV) dictionary
                  on those residuals (the same recipe as the Qwen arms,
                  shrunk to this model's scale).
  stage match   : for every decoder direction, find its best cosine match
                  among the tomograph's wall writes (u, weighted by write
                  strength); compare the match distribution against random
                  unit vectors in the same space — the null that alignment
                  is geometric accident.

Usage: python3 duality_pipeline.py harvest|train|match
"""

import json
import os
import sys

import numpy as np
import torch


def load_model(dev):
    ck = torch.load(os.path.expanduser("~/relu_lm_ckpt.pt"), map_location="cpu",
                    weights_only=False)
    sys.path.insert(0, os.path.expanduser("~"))
    from relu_lm import ReluLM
    model = ReluLM(len(ck["vocab_bytes"]))
    model.load_state_dict(ck["model"])
    model.eval()
    return model.to(dev), ck["vocab_bytes"]


def harvest():
    dev = "cuda:0"
    model, vocab_bytes = load_model(dev)
    text = open(os.path.expanduser("~/wikitext103_train.txt"), "rb").read()
    lut = {b: i for i, b in enumerate(vocab_bytes)}
    ids = np.array([lut[b] for b in text if b in lut], dtype=np.int64)
    ctx, rows_wanted = 64, 300_000
    out = []
    with torch.inference_mode():
        step = ctx  # non-overlapping windows: every position appears once
        for s in range(0, rows_wanted * step // ctx, step):
            seq = torch.tensor(ids[s:s + ctx][None, :], device=dev)
            if seq.shape[1] < ctx:
                break
            x = model.emb(seq) + model.pos(torch.arange(ctx, device=dev))
            mask = torch.triu(torch.full((ctx, ctx), float("-inf"), device=dev), 1)
            for blk in model.blocks:
                hn = blk.n1(x)
                a, _ = blk.attn(hn, hn, hn, attn_mask=mask, need_weights=False)
                x = x + a
                x = x + blk.down(torch.relu(blk.up(blk.n2(x))))
            out.append(model.norm(x)[0].float().cpu().numpy())
            if len(out) * ctx >= rows_wanted:
                break
    acts = np.concatenate(out)[:rows_wanted]
    np.save(os.path.expanduser("~/relu_lm_resid.npy"), acts)
    print(json.dumps({"rows": len(acts), "d": acts.shape[1]}), flush=True)
    print("HARVEST_DONE", flush=True)


def train():
    # Both dictionaries at equal decoder parameters: flat K=1024 (1 vector
    # each) vs curved K=512 (2 vectors each), d=256, TopK=8, AuxK — the same
    # recipe as the Qwen arms (see artifacts/issue_2502_curved_steelman.py),
    # inlined and shrunk.
    dev = "cuda:0"
    X = np.load(os.path.expanduser("~/relu_lm_resid.npy"))
    n_test = len(X) // 10
    Xtr, Xte = X[:-n_test], X[-n_test:]
    t_data = torch.tensor(Xtr, dtype=torch.float32, device=dev)
    results = {}
    for arm, K, curved in (("flat", 1024, False), ("curved", 512, True)):
        g = torch.Generator(device=dev).manual_seed(0)
        P = t_data.shape[1]
        U = torch.randn(K, P, generator=g, device=dev)
        U /= U.norm(dim=1, keepdim=True)
        U = torch.nn.Parameter(U)
        V = torch.randn(K, P, generator=g, device=dev)
        V.mul_(0.1 / V.norm(dim=1, keepdim=True))
        V = torch.nn.Parameter(V)
        W_enc = torch.nn.Parameter(U.detach().t().clone())
        b_enc = torch.nn.Parameter(torch.zeros(K, device=dev))
        W_t = torch.nn.Parameter(0.01 * torch.randn(P, K, generator=g, device=dev))
        b_t = torch.nn.Parameter(torch.zeros(K, device=dev))
        b_pre = torch.nn.Parameter(t_data.mean(0).clone())
        params = [U, W_enc, b_enc, b_pre] + ([V, W_t, b_t] if curved else [])
        opt = torch.optim.Adam(params, lr=1e-3)
        n, bs, k_act, epochs = len(t_data), 4096, 8, 60
        steps = (n + bs - 1) // bs * epochs
        warm = max(1, steps // 50)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: (s + 1) / warm if s < warm
            else 0.5 * (1 + np.cos(np.pi * (s - warm) / max(1, steps - warm))))
        last_fired = torch.zeros(K, dtype=torch.long, device=dev)
        spe = (n + bs - 1) // bs
        for _ep in range(epochs):
            perm = torch.randperm(n, generator=g, device=dev)
            for s in range(0, n, bs):
                xb = t_data[perm[s:s + bs]]
                xc = xb - b_pre
                pre = xc @ W_enc + b_enc
                tc = torch.tanh(xc @ W_t + b_t) if curved else torch.zeros_like(pre)
                val, idx = torch.topk(pre, k_act, dim=1)
                z = torch.zeros_like(pre).scatter_(1, idx, torch.relu(val))
                recon = z @ U + (z * tc) @ V + b_pre if curved else z @ U + b_pre
                residual = xb - recon
                loss = (residual ** 2).mean()
                with torch.no_grad():
                    fired = torch.zeros(K, dtype=torch.bool, device=dev)
                    fired.scatter_(0, idx.reshape(-1), True)
                    last_fired += 1
                    last_fired[fired] = 0
                    dead = last_fired > 8 * spe
                if dead.any():
                    dp = pre.masked_fill(~dead.unsqueeze(0), float("-inf"))
                    kk = int(min(512, int(dead.sum())))
                    av, ai = torch.topk(dp, kk, dim=1)
                    az = torch.zeros_like(pre).scatter_(1, ai, torch.relu(av))
                    aux = az @ U + ((az * tc) @ V if curved else 0)
                    loss = loss + (1 / 32) * ((residual.detach() - aux) ** 2).mean()
                opt.zero_grad(); loss.backward(); opt.step(); sched.step()
                with torch.no_grad():
                    U.data /= U.data.norm(dim=1, keepdim=True).clamp_min(1e-8)
        np.savez(os.path.expanduser(f"~/duality_{arm}.npz"),
                 U=U.detach().cpu().numpy(), V=V.detach().cpu().numpy(),
                 b_pre=b_pre.detach().cpu().numpy(), curved=curved)
        with torch.no_grad():
            xte = torch.tensor(Xte, dtype=torch.float32, device=dev) - b_pre
            pre = xte @ W_enc + b_enc
            tc = torch.tanh(xte @ W_t + b_t) if curved else torch.zeros_like(pre)
            val, idx = torch.topk(pre, k_act, dim=1)
            z = torch.zeros_like(pre).scatter_(1, idx, torch.relu(val))
            rec = z @ U + (z * tc) @ V if curved else z @ U
            evv = 1 - ((xte - rec) ** 2).sum() / (xte ** 2).sum()
        results[arm] = float(evv)
        print(json.dumps({arm: float(evv)}), flush=True)
    print("TRAIN_DONE", flush=True)


def match():
    d = np.load(os.path.expanduser("~/tomograph_atoms.npz"))
    u = d["u"].astype(np.float64)               # (A, dim) unit write directions
    w = d["u_norm"].astype(np.float64)          # write strengths
    rng = np.random.default_rng(0)
    out = {"atoms": int(len(u))}
    for arm in ("flat", "curved"):
        blob = np.load(os.path.expanduser(f"~/duality_{arm}.npz"))
        U = blob["U"].astype(np.float64)
        U /= np.linalg.norm(U, axis=1, keepdims=True)
        # Best |cosine| of each decoder direction to any wall write; the
        # random null draws the same number of unit directions in the same
        # space. Weighting: each atom counts by its write strength, so soft
        # numerical flips do not dominate the matchable set.
        keep = w > np.quantile(w, 0.5)          # the harder-writing half
        cos = np.abs(U @ u[keep].T).max(axis=1)
        null_dirs = rng.standard_normal(U.shape)
        null_dirs /= np.linalg.norm(null_dirs, axis=1, keepdims=True)
        cos_null = np.abs(null_dirs @ u[keep].T).max(axis=1)
        out[arm] = {
            "median_best_cos": float(np.median(cos)),
            "null_median_best_cos": float(np.median(cos_null)),
            "frac_above_null_p95": float((cos > np.quantile(cos_null, 0.95)).mean()),
        }
        print(json.dumps({arm: out[arm]}), flush=True)
    json.dump(out, open(os.path.expanduser("~/duality_match.json"), "w"), indent=1)
    print("MATCH_DONE", flush=True)


if __name__ == "__main__":
    {"harvest": harvest, "train": train, "match": match}[sys.argv[1]]()
