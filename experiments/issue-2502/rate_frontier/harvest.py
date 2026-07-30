"""Regenerate the #2502 harvest + charts (the originals died with the box I terminated).

Qwen3.5-4B-Base residual stream after block 16, wikitext-103-raw-v1, document-level
split, train-only PCA-128 chart. Writes the two f64 .bin files the Rust harness reads.

Deliberately larger than the original prep: the earlier campaign used 44,818 train rows
of the 537,237 available, which capped rows-per-atom and is why the guarded support move
never fired at K=8,000. This targets the high-rows-per-atom regime.
"""

import hashlib
import os
import sys

import numpy as np

OUT = os.path.expanduser("~/i2502")
LAYER = 16
SEQ = 512
N_SEQ = 1400          # ~716k tokens; trimmed to the doc-split train side below
P = 128
N_TRAIN = 250000
N_TEST = 12000
SPLIT_KEY = b"gam-2502-docsplit-v1"


def doc_side(doc_ids):
    """Document-level held-out mask by keyed hash — no RNG, stable across machines."""
    out = np.empty(len(doc_ids), dtype=bool)
    cache = {}
    for i, d in enumerate(doc_ids):
        d = int(d)
        if d not in cache:
            h = hashlib.blake2b(SPLIT_KEY + str(d).encode(), digest_size=8).digest()
            cache[d] = (int.from_bytes(h, "little") % 100) < 10   # ~10% held out
        out[i] = cache[d]
    return out


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    name = "Qwen/Qwen3.5-4B-Base"
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="cuda:0")
    model.eval()
    layers = None
    for _n, mod in model.named_modules():
        if isinstance(mod, torch.nn.ModuleList) and (layers is None or len(mod) > len(layers)):
            layers = mod
    block = layers[LAYER]
    print(f"model loaded, {len(layers)} blocks, tapping {LAYER}", flush=True)

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
    print(f"tokenized {len(seqs)} sequences of {SEQ}", flush=True)

    cap = {}
    def hook(_m, _i, output):
        cap["h"] = (output[0] if isinstance(output, tuple) else output).detach()
    h = block.register_forward_hook(hook)

    acts, doc_ids = [], []
    B = 8
    with torch.inference_mode():
        for s in range(0, len(seqs), B):
            ids = torch.tensor(seqs[s:s + B], dtype=torch.long, device="cuda:0")
            model(input_ids=ids, use_cache=False)
            acts.append(cap["h"].to(torch.float32).reshape(-1, cap["h"].shape[-1]).cpu().numpy())
            doc_ids.append(np.asarray(docs[s:s + B], dtype=np.int64).reshape(-1))
            if (s // B) % 20 == 0:
                print(f"  forward {s}/{len(seqs)}", flush=True)
    h.remove()
    X = np.concatenate(acts, 0).astype(np.float64)
    D = np.concatenate(doc_ids, 0)
    print("activations", X.shape, flush=True)

    held = doc_side(D)
    tr_idx = np.flatnonzero(~held)
    te_idx = np.flatnonzero(held)
    print(f"doc split: {len(tr_idx)} train-side rows, {len(te_idx)} held-out rows", flush=True)

    rng = np.random.default_rng(0)
    tr = np.sort(rng.choice(tr_idx, size=min(N_TRAIN, len(tr_idx)), replace=False))
    te = np.sort(rng.choice(te_idx, size=min(N_TEST, len(te_idx)), replace=False))

    # chart fitted on TRAIN ROWS ONLY -- fixes the provenance caveat the old prep carried
    c0 = X[tr].mean(0)
    Xc = X[tr] - c0
    _, _, vt = np.linalg.svd(Xc, full_matrices=False)
    lift = vt[:P]
    ev_frac = float((Xc @ lift.T).var(0).sum() / Xc.var(0).sum())
    print(f"train-only PCA-{P} chart holds {ev_frac:.4f} of variance", flush=True)

    Ztr = np.ascontiguousarray((X[tr] - c0) @ lift.T)
    Zte = np.ascontiguousarray((X[te] - c0) @ lift.T)
    open(f"{OUT}/doc_chart.bin", "wb").write(Ztr.tobytes())
    open(f"{OUT}/test_chart.bin", "wb").write(Zte.tobytes())
    np.save(f"{OUT}/lift.npy", lift); np.save(f"{OUT}/c0.npy", c0)
    np.save(f"{OUT}/train_chart.npy", Ztr); np.save(f"{OUT}/test_chart.npy", Zte)
    np.save(f"{OUT}/test_ambient.npy", X[te])
    print(f"HARVEST DONE train={Ztr.shape} test={Zte.shape} "
          f"rms={float(np.sqrt((Ztr**2).mean())):.4f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
