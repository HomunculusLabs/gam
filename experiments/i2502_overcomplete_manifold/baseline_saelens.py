"""External-library SAE baseline for #2502 (no strawman): SAELens TopK SAE
trained on the IDENTICAL chart data at matched K and L0.
"""
import argparse, inspect, json, os, time
import numpy as np
import torch


def ev_of(X, R):
    return 1.0 - float(((X - R) ** 2).sum()) / float((X ** 2).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", default=os.path.expanduser("~/i2502/prep_L16_p128_25k"))
    ap.add_argument("--k", type=int, default=32000)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out-dir", default=os.path.expanduser("~/i2502/fits"))
    args = ap.parse_args()

    import sae_lens
    print("sae_lens", sae_lens.__version__, flush=True)
    from sae_lens.saes.topk_sae import (TopKTrainingSAE, TopKTrainingSAEConfig, TrainStepInput)

    Xtr = torch.from_numpy(np.load(f"{args.prep}/train.npy")).float().cuda()
    Xte = torch.from_numpy(np.load(f"{args.prep}/test.npy")).float().cuda()
    D = Xtr.shape[1]

    cfg = TopKTrainingSAEConfig(
        d_in=int(D), d_sae=int(args.k), k=int(args.top_k),
        device="cuda", dtype="float32")
    sae = TopKTrainingSAE(cfg)
    sae.to("cuda")

    opt = torch.optim.Adam(sae.parameters(), lr=args.lr)
    N = len(Xtr)
    t0 = time.time()
    n_steps = 0
    for ep in range(args.epochs):
        perm = torch.randperm(N, device="cuda")
        for i in range(0, N, args.batch):
            xb = Xtr[perm[i:i + args.batch]]
            out = sae.training_forward_pass(TrainStepInput(
                sae_in=xb, coefficients={}, dead_neuron_mask=None,
                n_training_steps=n_steps, is_logging_step=False))
            loss = out.loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            n_steps += 1
        if ep % 20 == 0:
            with torch.no_grad():
                r = sae.decode(sae.encode(Xte[:2048]))
                print(f"[saelens] ep{ep} loss={float(loss):.5f} test_ev~="
                      f"{ev_of(Xte[:2048].cpu().numpy(), r.cpu().numpy()):.4f}",
                      flush=True)
    wall = time.time() - t0

    with torch.no_grad():
        outs, l0s, alive = [], [], set()
        for i in range(0, len(Xte), 2048):
            f = sae.encode(Xte[i:i + 2048])
            outs.append(sae.decode(f))
            nz = f != 0
            l0s.append(nz.float().sum(1))
            alive.update(torch.nonzero(nz.any(0)).flatten().tolist())
        R = torch.cat(outs)
        rtr_chunks = [sae.decode(sae.encode(Xtr[i:i + 2048]))
                      for i in range(0, len(Xtr), 2048)]
        Rtr = torch.cat(rtr_chunks)
    rec = dict(record=f"saelens_topk_k{args.k}_p128", status="ok",
               library=f"sae-lens=={sae_lens.__version__}", k=args.k,
               top_k=args.top_k, epochs=args.epochs, wall_s=round(wall, 1),
               train_ev=ev_of(Xtr.cpu().numpy(), Rtr.cpu().numpy()),
               test_ev=ev_of(Xte.cpu().numpy(), R.cpu().numpy()),
               test_mean_l0=float(torch.cat(l0s).mean().item()),
               alive_atoms_test=len(alive))
    with open(os.path.join(args.out_dir, "fits.jsonl"), "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print("[saelens]", json.dumps(rec), flush=True)
    W = sae.W_dec.detach().cpu().numpy()
    np.savez(os.path.join(args.out_dir, f"saelens_topk_k{args.k}_p128.npz"),
             W_dec=W, b_dec=sae.b_dec.detach().cpu().numpy())


if __name__ == "__main__":
    main()
