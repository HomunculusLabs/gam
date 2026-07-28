"""Recover the v2 TRAIN tokens and (sequence, position) index, without the GPU.

`harvest_v2` saved the chart and the ambient held-out activations but not the
train-side token identities, and the figures / interpretation / steering
deliverables need them. Every step that maps a token to a row is deterministic --
the streaming dataset is read in order, the tokenizer is pure, the split is a
keyed blake2b, and the subsample is `default_rng(0)` -- so replaying just those
steps reproduces the exact row->token map with no forward passes.

Guarded: the recovered row count must equal the chart's, or this refuses rather
than emitting a misaligned map.
"""

import hashlib
import os
import sys

import numpy as np

V2 = os.path.expanduser("~/i2502v2")
SEQ = 512
N_SEQ = 1400
N_TRAIN = 250000
P = 128
SPLIT_KEY = b"gam-2502-docsplit-v2"


def held_out(doc_ids):
    cache, out = {}, np.empty(len(doc_ids), dtype=bool)
    for i, d in enumerate(doc_ids):
        d = int(d)
        if d not in cache:
            h = hashlib.blake2b(SPLIT_KEY + str(d).encode(), digest_size=8).digest()
            cache[d] = (int.from_bytes(h, "little") % 100) < 10
        out[i] = cache[d]
    return out


def main() -> int:
    from transformers import AutoTokenizer
    from datasets import load_dataset

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B-Base", trust_remote_code=True)
    ds = load_dataset(
        "Salesforce/wikitext", "wikitext-103-raw-v1", split="train", streaming=True
    )

    seqs, docs, buf, buf_doc = [], [], [], []
    for doc_i, ex in enumerate(ds):
        t = ex["text"].strip()
        if not t:
            continue
        ids = tok.encode(t, add_special_tokens=False)
        buf.extend(ids)
        buf_doc.extend([doc_i] * len(ids))
        while len(buf) >= SEQ:
            seqs.append(buf[:SEQ])
            docs.append(buf_doc[:SEQ])
            buf, buf_doc = buf[SEQ:], buf_doc[SEQ:]
        if len(seqs) >= N_SEQ:
            break
    print(f"tokenized {len(seqs)} sequences", flush=True)

    tokens = np.asarray(seqs, dtype=np.int64).reshape(-1)
    doc_ids = np.asarray(docs, dtype=np.int64).reshape(-1)

    # The same split and the same subsample harvest_v2 used, so rows line up.
    train_idx = np.flatnonzero(~held_out(doc_ids))
    rng = np.random.default_rng(0)
    tr = np.sort(rng.choice(train_idx, size=min(N_TRAIN, len(train_idx)), replace=False))

    want = os.path.getsize(f"{V2}/doc_chart.bin") // P // 8
    if want != len(tr):
        print(f"REFUSED: doc_chart holds {want} rows, replay produced {len(tr)}")
        return 1

    np.save(f"{V2}/train_tokens.npy", tokens[tr])
    np.save(f"{V2}/train_seq_pos.npy", np.stack([tr // SEQ, tr % SEQ], 1))
    np.save(
        f"{V2}/train_chart.npy",
        np.fromfile(f"{V2}/doc_chart.bin", dtype=np.float64).reshape(-1, P),
    )
    vocab = np.array(
        [tok.decode([i]) for i in range(int(tokens.max()) + 1)], dtype=object
    )
    np.save(f"{V2}/vocab.npy", vocab, allow_pickle=True)
    print(f"V2 TRAIN TOKENS DONE rows={len(tr)} vocab={len(vocab)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
