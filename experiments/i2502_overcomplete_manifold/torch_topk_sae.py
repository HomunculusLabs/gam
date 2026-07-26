"""Standard-method baseline for #2502: TopK SAE (Gao et al. 2024) trained with
Adam on GPU — the community-standard sparse autoencoder — on the IDENTICAL
train/test chart as the manifold dictionary.

Also evaluates PCA-M subspace baselines at matched per-row coefficient budgets.
"""
import argparse, json, os, time
import numpy as np


def ev_of(X, R):
    return 1.0 - float(((X - R) ** 2).sum()) / float((X ** 2).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", default=os.path.expanduser("~/i2502/prep_L16"))
    ap.add_argument("--k", type=int, default=32000)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=os.path.expanduser("~/i2502/fits"))
    args = ap.parse_args()
    out = os.path.join(args.out_dir, "fits.jsonl")

    import torch
    torch.manual_seed(args.seed)
    dev = "cuda:0"
    X_train = torch.from_numpy(np.load(f"{args.prep}/train.npy")).float().to(dev)
    X_test = torch.from_numpy(np.load(f"{args.prep}/test.npy")).float().to(dev)
    N, D = X_train.shape
    K, k = args.k, args.top_k

    W_enc = torch.nn.Parameter(torch.randn(D, K, device=dev) * (1.0 / np.sqrt(D)))
    W_dec = torch.nn.Parameter(W_enc.detach().T.clone())
    b_pre = torch.nn.Parameter(X_train.mean(0).clone())
    b_enc = torch.nn.Parameter(torch.zeros(K, device=dev))
    params = [W_enc, W_dec, b_pre, b_enc]
    opt = torch.optim.Adam(params, lr=args.lr)

    def forward(xb):
        pre = (xb - b_pre) @ W_enc + b_enc
        vals, idx = torch.topk(pre, k, dim=1)
        vals = torch.relu(vals)
        z = torch.zeros_like(pre).scatter_(1, idx, vals)
        return z @ W_dec + b_pre, z

    t0 = time.perf_counter()
    steps = 0
    fired = torch.zeros(K, dtype=torch.bool, device=dev)
    for ep in range(args.epochs):
        perm = torch.randperm(N, device=dev)
        for i in range(0, N, args.batch):
            xb = X_train[perm[i:i + args.batch]]
            recon, z = forward(xb)
            loss = ((recon - xb) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            with torch.no_grad():
                W_dec.data /= W_dec.data.norm(dim=1, keepdim=True).clamp_min(1e-8)
                fired |= (z != 0).any(0)
            steps += 1
        if ep % 20 == 0:
            with torch.no_grad():
                r, _ = forward(X_test[:2048])
                print(f"[topk] ep{ep} loss={loss.item():.5f} test_ev~="
                      f"{ev_of(X_test[:2048].cpu().numpy(), r.cpu().numpy()):.4f}",
                      flush=True)
    wall = time.perf_counter() - t0

    def forward_chunked(X, bs=2048):
        outs, zs = [], []
        with torch.no_grad():
            for i in range(0, len(X), bs):
                r, z = forward(X[i:i + bs])
                outs.append(r)
                zs.append(z != 0)
        return torch.cat(outs), torch.cat(zs)

    with torch.no_grad():
        r_tr, _ = forward_chunked(X_train)
        r_te, nz_te = forward_chunked(X_test)
        alive = int(nz_te.any(0).sum().item())
        l0 = float(nz_te.float().sum(1).mean().item())
    rec = dict(record=f"torch_topk_k{K}", status="ok", k=K, top_k=k,
               epochs=args.epochs, wall_s=round(wall, 1),
               train_ev=ev_of(X_train.cpu().numpy(), r_tr.cpu().numpy()),
               test_ev=ev_of(X_test.cpu().numpy(), r_te.cpu().numpy()),
               test_mean_l0=round(l0, 2), alive_atoms_test=alive,
               ever_fired_train=int(fired.sum().item()))
    with open(out, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print("[topk]", json.dumps(rec), flush=True)
    np.savez(os.path.join(args.out_dir, f"torch_topk_k{K}.npz"),
             W_enc=W_enc.detach().cpu().numpy(), W_dec=W_dec.detach().cpu().numpy(),
             b_pre=b_pre.detach().cpu().numpy(), b_enc=b_enc.detach().cpu().numpy())

    # PCA baselines on the same chart
    Xtr = X_train.cpu().numpy()
    Xte = X_test.cpu().numpy()
    _, _, vt = np.linalg.svd(Xtr - Xtr.mean(0), full_matrices=False)
    for M in (args.top_k, 16, 64, 256):
        V = vt[:M]
        R = ((Xte - Xtr.mean(0)) @ V.T) @ V + Xtr.mean(0)
        rec = dict(record=f"pca_M{M}", status="ok",
                   test_ev=ev_of(Xte, R), test_mean_l0=M)
        with open(out, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print("[topk]", json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
