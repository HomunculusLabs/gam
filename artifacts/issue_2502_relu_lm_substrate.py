"""Train a small ReLU-family LM as the substrate for curvature tomography.

The curvature-atoms duality test needs a language model whose MLP
nonlinearities are exactly piecewise-linear, so that facet crossings carry
exact rank-one Jacobian jumps (Theorem 1) and ray-walking is an exact
measuring instrument. SwiGLU/GeLU models (Qwen, GPT-2) only approximate this;
here the MLPs are pure ReLU. Attention remains smooth — its contribution is
the absolutely-continuous background the theory already prices in — so every
atom the tomograph finds is an MLP-owned facet with clean provenance.

Architecture: pre-norm transformer, RMSNorm, learned positions, 4 layers,
d_model 256, 4 heads, ReLU MLP x4 expansion, ctx 256, byte-pair-free
character-level vocab from wikitext-103 (keeps the input space continuous
after the embedding, which is where rays live).
"""

import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.n1 = nn.RMSNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.n2 = nn.RMSNorm(d)
        self.up = nn.Linear(d, 4 * d)
        self.down = nn.Linear(4 * d, d)

    def forward(self, x, mask):
        h = self.n1(x)
        a, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + a
        x = x + self.down(torch.relu(self.up(self.n2(x))))
        return x


class ReluLM(nn.Module):
    def __init__(self, vocab, d=256, heads=4, layers=4, ctx=256):
        super().__init__()
        self.ctx = ctx
        self.emb = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(ctx, d)
        self.blocks = nn.ModuleList([Block(d, heads) for _ in range(layers)])
        self.norm = nn.RMSNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.emb.weight

    def forward(self, idx):
        b, t = idx.shape
        x = self.emb(idx) + self.pos(torch.arange(t, device=idx.device))
        mask = torch.triu(torch.full((t, t), float("-inf"), device=idx.device), 1)
        for blk in self.blocks:
            x = blk(x, mask)
        return self.head(self.norm(x))


def main():
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    dev = "cuda:0"
    text_path = os.path.expanduser("~/wikitext103_train.txt")
    if not os.path.exists(text_path):
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
        with open(text_path, "w") as f:
            for row in ds:
                f.write(row["text"])
    raw = open(text_path, "rb").read()[: 200_000_000]
    arr = np.frombuffer(raw, dtype=np.uint8)
    vocab_bytes = sorted(set(arr.tolist()))
    lut = np.zeros(256, dtype=np.int64)
    for i, b in enumerate(vocab_bytes):
        lut[b] = i
    data = torch.from_numpy(lut[arr])
    n_val = len(data) // 100
    train, val = data[:-n_val], data[-n_val:]
    vocab = len(vocab_bytes)
    torch.manual_seed(0)
    model = ReluLM(vocab).to(dev)
    print(f"vocab={vocab} params={sum(p.numel() for p in model.parameters())/1e6:.2f}M",
          flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / 500, 0.5 * (1 + math.cos(math.pi * s / steps))))
    bs, ctx = 48, 256
    g = torch.Generator().manual_seed(0)
    t0 = time.time()
    for step in range(steps):
        ix = torch.randint(0, len(train) - ctx - 1, (bs,), generator=g)
        xb = torch.stack([train[i:i + ctx] for i in ix]).to(dev)
        yb = torch.stack([train[i + 1:i + ctx + 1] for i in ix]).to(dev)
        logits = model(xb)
        loss = F.cross_entropy(logits.reshape(-1, vocab), yb.reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 500 == 0 or step == steps - 1:
            model.eval()
            with torch.no_grad():
                vl = []
                for k in range(20):
                    i = k * (len(val) - ctx - 1) // 20
                    xv = val[i:i + ctx].unsqueeze(0).to(dev)
                    yv = val[i + 1:i + ctx + 1].unsqueeze(0).to(dev)
                    vl.append(float(F.cross_entropy(
                        model(xv).reshape(-1, vocab), yv.reshape(-1))))
            model.train()
            print(json.dumps({"step": step, "train_loss": float(loss),
                              "val_loss": float(np.mean(vl)),
                              "elapsed_s": round(time.time() - t0, 1)}), flush=True)
            torch.save({"model": model.state_dict(), "vocab_bytes": vocab_bytes,
                        "step": step}, os.path.expanduser("~/relu_lm_ckpt.pt"))
    print("RELU_LM_DONE", flush=True)


if __name__ == "__main__":
    main()
