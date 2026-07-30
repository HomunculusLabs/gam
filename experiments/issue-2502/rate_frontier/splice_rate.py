"""Causal rate frontier: splice ΔCE at matched scalars/token.

Replaces the in-chart component of the layer-16 residual with each SAE's decode
of it (orthogonal complement kept intact), and measures the CE increase on
held-out-document target tokens. Identity splice (decode = the chart projection
itself) is the control and prices the chart truncation; each arm's damage is
reported both raw (vs clean) and net of the identity control.

Same packing, same doc split, same layer as harvest.py. Within-run numbers only.
Usage: python3 splice_rate.py L0 name=weights.npz:form ...   (form: flat|field)
"""
import hashlib
import json
import os
import sys

import numpy as np

LAYER = 16
SEQ = 512
N_SEQ = 1400
SPLIT_KEY = b"gam-2502-docsplit-v1"
OUT = os.path.expanduser("~/i2502")


def doc_side_one(d, cache={}):
    if d not in cache:
        h = hashlib.blake2b(SPLIT_KEY + str(d).encode(), digest_size=8).digest()
        cache[d] = (int.from_bytes(h, "little") % 100) < 10
    return cache[d]


def main() -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    L0 = int(sys.argv[1])
    arms = []
    for a in sys.argv[2:]:
        name, rest = a.split("=", 1)
        path, form = rest.rsplit(":", 1)
        arms.append((name, dict(np.load(path)), form))

    dev = "cuda:0"
    lift = torch.tensor(np.load(f"{OUT}/lift.npy"), dtype=torch.float32, device=dev)  # P x D
    c0 = torch.tensor(np.load(f"{OUT}/c0.npy"), dtype=torch.float32, device=dev)      # D

    mdl = "Qwen/Qwen3.5-4B-Base"
    tok = AutoTokenizer.from_pretrained(mdl, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        mdl, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map=dev)
    model.eval()
    layers = None
    for _n, mod in model.named_modules():
        if isinstance(mod, torch.nn.ModuleList) and (layers is None or len(mod) > len(layers)):
            layers = mod
    block = layers[LAYER]

    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
    seqs, docs, buf, buf_doc = [], [], [], []
    for doc_i, ex in enumerate(ds):
        t = ex["text"].strip()
        if not t:
            continue
        ids = tok.encode(t, add_special_tokens=False)
        buf.extend(ids); buf_doc.extend([doc_i] * len(ids))
        while len(buf) >= SEQ:
            seqs.append(buf[:SEQ]); docs.append(buf_doc[:SEQ])
            buf, buf_doc = buf[SEQ:], buf_doc[SEQ:]
        if len(seqs) >= N_SEQ:
            break
    # only sequences that contain any held-out-document token are worth a forward
    keep = [i for i in range(len(seqs)) if any(doc_side_one(d) for d in docs[i])]
    print(f"sequences {len(seqs)}, with held-out tokens {len(keep)}", flush=True)

    def make_codec(w, form):
        U = torch.tensor(w["U"], dtype=torch.float32, device=dev)
        W_enc = torch.tensor(w["W_enc"], dtype=torch.float32, device=dev)
        b_enc = torch.tensor(w["b_enc"], dtype=torch.float32, device=dev)
        b_pre = torch.tensor(w["b_pre"], dtype=torch.float32, device=dev)
        extra = {}
        if form == "field":
            for k in ("MA", "MB", "g2"):
                extra[k] = torch.tensor(w[k], dtype=torch.float32, device=dev)

        def codec(zc):  # rows x P chart vectors -> decoded chart vectors, L0 firings
            pre = (zc - b_pre) @ W_enc + b_enc
            val, idx = torch.topk(pre, L0, dim=1)
            z = torch.zeros_like(pre).scatter_(1, idx, torch.relu(val))
            rec = z @ U
            if form == "field":
                V_eff = U @ extra["MA"] @ extra["MB"].t()
                rec = rec + ((z * z) * extra["g2"]) @ V_eff
            return rec + b_pre
        return codec

    codecs = {"identity": None}
    for name, w, form in arms:
        codecs[name] = make_codec(w, form)

    mode = {"arm": None}
    def hook(_m, _i, output):
        if mode["arm"] is None:
            return output
        h = output[0] if isinstance(output, tuple) else output
        B, S, D = h.shape
        x = h.reshape(-1, D).to(torch.float32)
        zc = (x - c0) @ lift.T
        dec = zc if mode["arm"] == "identity" else codecs[mode["arm"]](zc)
        x2 = x + (dec - zc) @ lift          # replace in-chart part, keep complement
        h2 = x2.to(h.dtype).reshape(B, S, D)
        return (h2,) + tuple(output[1:]) if isinstance(output, tuple) else h2
    block.register_forward_hook(hook)

    results = {}
    B = 4
    with torch.inference_mode():
        for arm in [None, "identity"] + [a[0] for a in arms]:
            mode["arm"] = arm
            tot, cnt = 0.0, 0
            for s in range(0, len(keep), B):
                batch = keep[s:s + B]
                ids = torch.tensor([seqs[i] for i in batch], dtype=torch.long, device=dev)
                logits = model(input_ids=ids, use_cache=False).logits
                ce = torch.stack([
                    torch.nn.functional.cross_entropy(
                        logits[b, :-1].float(), ids[b, 1:], reduction="none")
                    for b in range(logits.shape[0])])
                del logits
                m = torch.tensor([[doc_side_one(docs[i][j + 1]) for j in range(SEQ - 1)]
                                  for i in batch], device=dev)
                tot += float((ce * m).sum()); cnt += int(m.sum())
            results["clean" if arm is None else arm] = tot / cnt
            print(f"{'clean' if arm is None else arm}: CE={tot/cnt:.6f} over {cnt} held-out tokens", flush=True)

    clean = results["clean"]; ident = results["identity"]
    out = {"L0": L0, "clean_ce": clean, "identity_ce": ident,
           "identity_dce": ident - clean}
    for name, _, _ in arms:
        out[f"{name}_ce"] = results[name]
        out[f"{name}_dce"] = results[name] - clean
        out[f"{name}_dce_net"] = results[name] - ident
    print(json.dumps(out, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
