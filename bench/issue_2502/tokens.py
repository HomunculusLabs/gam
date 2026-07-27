"""Recover the token id behind every harvested row, without touching the GPU.

harvest.py never saved the tokens, but every step that maps a token to a row is
deterministic: the streaming dataset is read in order, the tokenizer is pure, the
doc split is a keyed blake2b, and the subsample is default_rng(0). Replaying just
those steps reproduces the exact row->token map with no forward passes.

Guarded: the recovered row counts must equal the harvested chart row counts, or
this refuses rather than emitting a misaligned map.
"""

import hashlib
import os
import sys

import numpy as np

OUT = os.path.expanduser("~/i2502")
SEQ = 512
N_SEQ = 1400
N_TRAIN = 250000
N_TEST = 12000
SPLIT_KEY = b"gam-2502-docsplit-v1"


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
    from transformers import AutoTokenizer
    from datasets import load_dataset

    name = "Qwen/Qwen3.5-4B-Base"
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
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
    D = np.asarray(docs, dtype=np.int64).reshape(-1)

    held = doc_side(D)
    tr_idx = np.flatnonzero(~held)
    te_idx = np.flatnonzero(held)
    rng = np.random.default_rng(0)
    tr = np.sort(rng.choice(tr_idx, size=min(N_TRAIN, len(tr_idx)), replace=False))
    te = np.sort(rng.choice(te_idx, size=min(N_TEST, len(te_idx)), replace=False))

    # The chart files are the ground truth for how many rows exist; refuse on mismatch.
    for path, got in ((f"{OUT}/doc_chart.bin", len(tr)), (f"{OUT}/test_chart.bin", len(te))):
        want = os.path.getsize(path) // 128 // 8
        if want != got:
            print(f"REFUSED: {path} holds {want} rows, replay produced {got}")
            return 1

    np.save(f"{OUT}/train_tokens.npy", tokens[tr])
    np.save(f"{OUT}/test_tokens.npy", tokens[te])
    # The sampled rows are NOT contiguous text, so a steering pass cannot build a
    # context by slicing them. Save the full sequence grid plus each row's
    # (sequence, position) so real left-context can be reconstructed exactly.
    np.save(f"{OUT}/seqs.npy", np.asarray(seqs, dtype=np.int64))
    np.save(f"{OUT}/train_seq_pos.npy", np.stack([tr // SEQ, tr % SEQ], 1))
    np.save(f"{OUT}/test_seq_pos.npy", np.stack([te // SEQ, te % SEQ], 1))
    # Decoded strings for the interpretation pass, vocab-sized so it stays small.
    vocab = np.array(
        [tok.decode([i]) for i in range(int(tokens.max()) + 1)], dtype=object
    )
    np.save(f"{OUT}/vocab.npy", vocab, allow_pickle=True)
    print(
        f"TOKENS DONE train={tokens[tr].shape} test={tokens[te].shape} "
        f"vocab={len(vocab)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
