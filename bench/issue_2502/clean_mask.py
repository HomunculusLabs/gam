"""Mark the v2 confirmation rows that are genuinely fresh.

Two independent ~10% document splits overlap: 8.68% of the v2 held-out rows come
from documents that were ALSO v1 held-out, and those rows participated in the
adaptive decisions v1 adjudicated. They are not training contamination -- no
model ever trained on them -- but they are selection contamination, so the
confirmation should be scored on the complement.

Doc ids are recoverable on CPU: the streamed dataset is read in order and the
tokenizer is pure, so replaying the tokenisation reproduces the row->document
map exactly, with no forward passes.
"""

import hashlib
import os
import sys

import numpy as np

V2 = os.path.expanduser("~/i2502v2")
SEQ = 512
N_SEQ = 1400


def side(doc_ids, key):
    cache, out = {}, np.empty(len(doc_ids), dtype=bool)
    for i, d in enumerate(doc_ids):
        d = int(d)
        if d not in cache:
            h = hashlib.blake2b(key + str(d).encode(), digest_size=8).digest()
            cache[d] = (int.from_bytes(h, "little") % 100) < 10
        out[i] = cache[d]
    return out


def main() -> int:
    from transformers import AutoTokenizer
    from datasets import load_dataset

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B-Base", trust_remote_code=True)
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train",
                      streaming=True)
    docs, buf, buf_doc, n = [], [], [], 0
    for doc_i, ex in enumerate(ds):
        t = ex["text"].strip()
        if not t:
            continue
        ids = tok.encode(t, add_special_tokens=False)
        buf.extend(ids); buf_doc.extend([doc_i] * len(ids))
        while len(buf) >= SEQ:
            docs.append(buf_doc[:SEQ]); buf, buf_doc = buf[SEQ:], buf_doc[SEQ:]
            n += 1
        if n >= N_SEQ:
            break
    D = np.asarray(docs, dtype=np.int64).reshape(-1)

    pos = np.load(f"{V2}/test_seq_pos.npy")
    rows = pos[:, 0] * SEQ + pos[:, 1]
    test_docs = D[rows]

    v1_held = side(test_docs, b"gam-2502-docsplit-v1")
    clean = ~v1_held
    np.save(f"{V2}/test_clean_mask.npy", clean)
    print(f"v2 test rows={len(clean)}  fresh={int(clean.sum())} "
          f"({clean.mean():.4f})  selection-contaminated={int(v1_held.sum())}", flush=True)
    print("MASK DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
