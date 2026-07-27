"""Confirmation harvest for #2502 on a FRESH split key.

`gam-2502-docsplit-v1` has been used to adjudicate dozens of adaptive decisions
and is a development set now. This builds a document-disjoint confirmation set
under `gam-2502-docsplit-v2` that has never been scored, and saves the extra
artifacts the confirmation currencies need and the v1 prep did not keep:

  ambient test activations   so reconstruction error can be reported in the
                             2560-d residual stream, not only inside the
                             128-d chart
  sequence grid + positions  so the reconstruction can be spliced back into
                             the model at layer 16 and the cross-entropy
                             change measured

The chart is fitted on v2 TRAIN rows only, as in v1.
"""

import hashlib
import os
import sys

import numpy as np

OUT = os.path.expanduser("~/i2502v2")
LAYER = 16
SEQ = 512
N_SEQ = 1400
P = 128
N_TRAIN = 250000
N_TEST = 12000
SPLIT_KEY = b"gam-2502-docsplit-v2"          # <-- rotated
SPLICE_SEQS = 96                              # whole sequences held out for the CE splice


def doc_side(doc_ids):
    out = np.empty(len(doc_ids), dtype=bool)
    cache = {}
    for i, d in enumerate(doc_ids):
        d = int(d)
        if d not in cache:
            h = hashlib.blake2b(SPLIT_KEY + str(d).encode(), digest_size=8).digest()
            cache[d] = (int.from_bytes(h, "little") % 100) < 10
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
        name, dtype=torch.bfloat16, trust_remote_code=True, device_map="cuda:0")
    model.eval()
    layers = None
    for _n, mod in model.named_modules():
        if isinstance(mod, torch.nn.ModuleList) and (layers is None or len(mod) > len(layers)):
            layers = mod
    block = layers[LAYER]
    print(f"model loaded, {len(layers)} blocks, tapping {LAYER}", flush=True)

    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train",
                      streaming=True)
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
    print(f"tokenized {len(seqs)} sequences", flush=True)

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
            if (s // B) % 25 == 0:
                print(f"  forward {s}/{len(seqs)}", flush=True)
    h.remove()
    X = np.concatenate(acts, 0)
    D = np.concatenate(doc_ids, 0)
    print("activations", X.shape, flush=True)

    held = doc_side(D)
    tr_idx = np.flatnonzero(~held)
    te_idx = np.flatnonzero(held)
    print(f"v2 split: {len(tr_idx)} train-side, {len(te_idx)} held-out rows", flush=True)

    # Overlap against v1's held-out documents, reported rather than assumed.
    v1 = np.empty(len(D), dtype=bool)
    cache = {}
    for i, d in enumerate(D):
        d = int(d)
        if d not in cache:
            hh = hashlib.blake2b(b"gam-2502-docsplit-v1" + str(d).encode(),
                                 digest_size=8).digest()
            cache[d] = (int.from_bytes(hh, "little") % 100) < 10
        v1[i] = cache[d]
    both = float((held & v1).sum()) / max(float(held.sum()), 1.0)
    print(f"fraction of v2 held-out rows that were ALSO v1 held-out: {both:.4f}", flush=True)

    rng = np.random.default_rng(0)
    tr = np.sort(rng.choice(tr_idx, size=min(N_TRAIN, len(tr_idx)), replace=False))
    te = np.sort(rng.choice(te_idx, size=min(N_TEST, len(te_idx)), replace=False))

    c0 = X[tr].astype(np.float64).mean(0)
    Xc = X[tr].astype(np.float64) - c0
    _, _, vt = np.linalg.svd(Xc, full_matrices=False)
    lift = vt[:P]
    print(f"train-only PCA-{P} chart holds "
          f"{float((Xc @ lift.T).var(0).sum() / Xc.var(0).sum()):.4f} of variance", flush=True)

    Ztr = np.ascontiguousarray((X[tr].astype(np.float64) - c0) @ lift.T)
    Zte = np.ascontiguousarray((X[te].astype(np.float64) - c0) @ lift.T)
    open(f"{OUT}/doc_chart.bin", "wb").write(Ztr.tobytes())
    open(f"{OUT}/test_chart.bin", "wb").write(Zte.tobytes())
    np.save(f"{OUT}/lift.npy", lift); np.save(f"{OUT}/c0.npy", c0)
    np.save(f"{OUT}/train_chart.npy", Ztr); np.save(f"{OUT}/test_chart.npy", Zte)

    # Ambient truth for the held-out rows: needed to score reconstruction in the
    # residual stream rather than inside the chart the model never uses.
    np.save(f"{OUT}/test_ambient.npy", X[te].astype(np.float32))

    # Whole held-out sequences for the CE splice, chosen from documents on the
    # held-out side so the splice never sees training text.
    seq_doc = np.asarray(docs, dtype=np.int64)[:, 0]
    seq_held = doc_side(seq_doc)
    splice = np.flatnonzero(seq_held)[:SPLICE_SEQS]
    np.save(f"{OUT}/splice_seqs.npy", np.asarray(seqs, dtype=np.int64)[splice])
    np.save(f"{OUT}/seqs.npy", np.asarray(seqs, dtype=np.int64))
    np.save(f"{OUT}/test_seq_pos.npy", np.stack([te // SEQ, te % SEQ], 1))
    print(f"HARVEST V2 DONE train={Ztr.shape} test={Zte.shape} "
          f"ambient={X[te].shape} splice_seqs={len(splice)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
