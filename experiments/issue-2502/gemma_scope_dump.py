"""Dump a public Gemma Scope 2 SAE's decoder and codes for the Rust curl census.

Thin producer: this file downloads the artifact, runs Google's model, runs
Google's JumpReLU encoder, and writes raw arrays. Every statistic the census
reports -- antipodal coalescing, the joint amplitude law kappa with its
influence-function SE, the noise-debiased radius, the rate-distortion crossover,
the acceptance conjunction -- is computed by
`crates/gam-sae/examples/curl_census_foreign.rs` calling
`gam_sae::manifold::census_shattered_circles`. Nothing here decides anything.

The `null` arm replaces the real activations with a Gaussian matched to their
per-coordinate mean and standard deviation and pushes it through the SAME
encoder, so the census's accept rate has a matched false-positive reference.

    python gemma_scope_dump.py <outdir> [--layer 17] [--width 16k] [--l0 medium]
                               [--rows 100000] [--null]
"""

import argparse
import json
import os

import numpy as np
import torch
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

SAE_REPO = "google/gemma-scope-2-4b-pt"
# The SAE weights are public; google/gemma-3-4b-pt itself is access-gated, so the
# activations come from an open mirror of the same pretrained checkpoint. That the
# mirror IS the checkpoint the SAE was trained on is not taken on trust: the census
# reports the SAE's realised L0 and fraction-of-variance-unexplained on these
# activations, and a dictionary run on the wrong model reconstructs it badly.
MODEL_ID = "unsloth/gemma-3-4b-pt"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("--layer", type=int, default=17)
    ap.add_argument("--width", default="16k")
    ap.add_argument("--l0", default="medium")
    ap.add_argument("--rows", type=int, default=100_000)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--null", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    sae_dir = f"resid_post/layer_{args.layer}_width_{args.width}_l0_{args.l0}"
    cfg = json.load(
        open(hf_hub_download(SAE_REPO, f"{sae_dir}/config.json"))
    )
    print("SAE config:", cfg, flush=True)
    params = load_file(hf_hub_download(SAE_REPO, f"{sae_dir}/params.safetensors"))
    print("SAE tensors:", {k: tuple(v.shape) for k, v in params.items()}, flush=True)

    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map=dev
    ).eval()

    ds = load_dataset("NeelNanda/pile-10k", split="train")
    texts = [t for t in ds["text"] if len(t) > 2000]

    # Harvest the residual stream at the SAE's own hook point: hidden_states[l+1]
    # is the stream after block l. Positions 0..3 are dropped (the BOS attention
    # sink is not what the dictionary is about).
    want = args.rows
    chunks = []
    tok_chunks = []
    got = 0
    ti = 0
    with torch.no_grad():
        while got < want and ti < len(texts):
            batch = texts[ti : ti + 8]
            ti += 8
            enc = tok(
                batch,
                return_tensors="pt",
                truncation=True,
                max_length=args.seq,
                padding="max_length",
            ).to(dev)
            out = model(**enc, output_hidden_states=True)
            h = out.hidden_states[args.layer + 1][:, 4:, :]
            mask = enc["attention_mask"][:, 4:].bool()
            sel = h[mask].float().cpu()
            chunks.append(sel)
            tok_chunks.append(enc["input_ids"][:, 4:][mask].cpu())
            got += sel.shape[0]
            print(f"harvest {got}/{want}", flush=True)
    x = torch.cat(chunks, 0)[:want]
    token_ids = torch.cat(tok_chunks, 0)[:want].numpy().astype(np.int32)
    del model, chunks
    torch.cuda.empty_cache()
    n, p = x.shape
    print("activations", x.shape, flush=True)

    if args.null:
        g = torch.Generator().manual_seed(20502)
        x = torch.randn(x.shape, generator=g) * x.std(0, keepdim=True) + x.mean(
            0, keepdim=True
        )
        print("NULL arm: matched Gaussian substituted", flush=True)

    w_enc = params["w_enc"].float().to(dev)
    b_enc = params["b_enc"].float().to(dev)
    w_dec = params["w_dec"].float().to(dev)
    b_dec = params["b_dec"].float().to(dev)
    thresh = params["threshold"].float().to(dev)
    if w_enc.shape[0] != p:
        w_enc = w_enc.T.contiguous()
    if w_dec.shape[1] != p:
        w_dec = w_dec.T.contiguous()
    k = w_dec.shape[0]

    # Google's JumpReLU encoder, unchanged.
    rows_idx, rows_val, counts = [], [], []
    with torch.no_grad():
        for s in range(0, n, 8192):
            xb = x[s : s + 8192].to(dev)
            pre = xb @ w_enc + b_enc
            act = torch.where(pre > thresh, torch.relu(pre), torch.zeros_like(pre))
            nzr, nzc = act.nonzero(as_tuple=True)
            counts.append(torch.bincount(nzr, minlength=xb.shape[0]).cpu())
            rows_idx.append(nzc.to(torch.int32).cpu())
            rows_val.append(act[nzr, nzc].cpu())
    idx = torch.cat(rows_idx).numpy().astype(np.int32)
    val = torch.cat(rows_val).numpy().astype(np.float32)
    cnt = torch.cat(counts).numpy().astype(np.int64)
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(cnt, out=indptr[1:])
    print(f"codes: nnz={len(idx)} mean L0={len(idx)/n:.1f}", flush=True)

    d = args.outdir
    x.numpy().astype(np.float32).tofile(f"{d}/x.f32")
    w_dec.cpu().numpy().astype(np.float32).tofile(f"{d}/w.f32")
    b_dec.cpu().numpy().astype(np.float32).tofile(f"{d}/b.f32")
    token_ids.tofile(f"{d}/tokens.i32")
    json.dump(
        {int(t): tok.decode([int(t)]) for t in np.unique(token_ids)},
        open(f"{d}/vocab.json", "w"),
    )
    indptr.tofile(f"{d}/indptr.i64")
    idx.tofile(f"{d}/idx.i32")
    val.tofile(f"{d}/val.f32")
    json.dump(
        {
            "n": int(n),
            "p": int(p),
            "k": int(k),
            "nnz": int(len(idx)),
            "sae": f"{SAE_REPO}/{sae_dir}",
            "model": MODEL_ID,
            "arm": "null" if args.null else "real",
            "sae_config": cfg,
        },
        open(f"{d}/meta.json", "w"),
    )
    print("DUMPDONE", d, flush=True)


if __name__ == "__main__":
    main()
