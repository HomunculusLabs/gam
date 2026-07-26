"""Declared train/test split for the #2502 Qwen3.5-4B-Base residual harvest.

Every arm of the #2502 / #2283 campaign must train and score against ONE split,
or the arms are unrelated numbers rather than a comparison. This module derives
that split from the harvest's own metadata, with no state outside the harvest.

Why the split is at DOCUMENT level, not row level
-------------------------------------------------
A harvest row is one token position of one document. Rows from a single document
are not independent draws: neighbouring residual-stream vectors within a sequence
share the document's topic, its induction/copy structure, and often literal
repeated tokens. The #2502 harvest holds 600,064 rows over 5,774 documents
(median ~104 rows per document), so a uniform random split over ROWS puts tokens
of the same document on both sides of the boundary, and a held-out score then
partly measures memorisation of documents the model was fit on.

Splitting on the document makes the boundary a real one: no token of a test
document is ever seen in training, at any position, in any layer.

Determinism
-----------
A document's side is a keyed hash of its integer id, so the assignment
  * needs no RNG, no seed ordering, and no shuffle implementation;
  * is identical across machines, Python versions and numpy versions;
  * is stable when rows are added, subsampled or reordered;
  * is the same for layers 8/16/22, which share one row ordering, so a document
    held out at one layer is held out at all of them.

`split_hash` binds the assignment to the harvest it was derived from (via a
digest of the doc-id column) so an arm quoting the hash is quoting both the rule
and the data it was applied to.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os

import numpy as np

SPLIT_VERSION = "gam-2502-docsplit-v1"
# Domain-separation key: this rule's identity. Changing it changes every side.
SPLIT_KEY = b"gam#2502-docsplit-v1"
TEST_FRACTION = 0.10
_UNIT = float(1 << 64)


def doc_bucket(doc_id: int) -> float:
    """Uniform-in-[0,1) bucket for a document id, from a keyed 64-bit hash."""
    digest = hashlib.blake2b(
        int(doc_id).to_bytes(8, "little", signed=False),
        key=SPLIT_KEY,
        digest_size=8,
    ).digest()
    return int.from_bytes(digest, "little") / _UNIT


def assign_documents(doc_ids: np.ndarray, test_fraction: float = TEST_FRACTION):
    """Return (test_doc_ids, train_doc_ids) as sorted arrays of unique ids."""
    if not 0.0 < test_fraction < 1.0:
        raise ValueError(f"test_fraction must lie in (0, 1); got {test_fraction}")
    unique = np.unique(np.asarray(doc_ids))
    buckets = np.fromiter(
        (doc_bucket(d) for d in unique), dtype=np.float64, count=len(unique)
    )
    is_test = buckets < test_fraction
    return unique[is_test], unique[~is_test]


def row_side(doc_ids: np.ndarray, test_fraction: float = TEST_FRACTION) -> np.ndarray:
    """Boolean mask over rows: True where the row's document is held out."""
    test_docs, _ = assign_documents(doc_ids, test_fraction)
    return np.isin(np.asarray(doc_ids), test_docs)


def harvest_digest(doc_ids: np.ndarray) -> str:
    """Digest of the doc-id column: the identity of the harvest being split."""
    arr = np.ascontiguousarray(np.asarray(doc_ids, dtype=np.int64))
    return hashlib.sha256(arr.tobytes()).hexdigest()


def split_manifest(doc_ids: np.ndarray, test_fraction: float = TEST_FRACTION) -> dict:
    """The full declarable description of the split, including its hash."""
    doc_ids = np.asarray(doc_ids)
    test_docs, train_docs = assign_documents(doc_ids, test_fraction)
    mask = np.isin(doc_ids, test_docs)
    if np.intersect1d(test_docs, train_docs).size:
        raise AssertionError("document appears on both sides of the split")

    payload = "\n".join(
        [
            SPLIT_VERSION,
            f"key={SPLIT_KEY.decode()}",
            f"test_fraction={test_fraction!r}",
            f"harvest_doc_digest={harvest_digest(doc_ids)}",
            f"n_rows={doc_ids.size}",
            f"n_docs={test_docs.size + train_docs.size}",
            "test_docs=" + ",".join(str(int(d)) for d in test_docs),
        ]
    ).encode()

    return {
        "split_version": SPLIT_VERSION,
        "rule": (
            "held out iff blake2b(doc_id.to_bytes(8,'little'), key=%r, digest_size=8) "
            "read as a little-endian u64 and divided by 2**64 is < test_fraction"
            % SPLIT_KEY.decode()
        ),
        "test_fraction": test_fraction,
        "harvest_doc_digest": harvest_digest(doc_ids),
        "n_rows": int(doc_ids.size),
        "n_docs": int(test_docs.size + train_docs.size),
        "n_docs_test": int(test_docs.size),
        "n_docs_train": int(train_docs.size),
        "n_rows_test": int(mask.sum()),
        "n_rows_train": int((~mask).sum()),
        "split_hash": hashlib.blake2b(payload, digest_size=16).hexdigest(),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--harvest", default=os.path.expanduser("~/i2502/harvest"))
    ap.add_argument("--test-fraction", type=float, default=TEST_FRACTION)
    ap.add_argument("--out", default=None, help="write split_manifest.json here")
    args = ap.parse_args()

    doc_ids = np.load(f"{args.harvest}/doc_ids.npy")
    manifest = split_manifest(doc_ids, args.test_fraction)

    # The property the split exists to guarantee, asserted rather than assumed.
    mask = row_side(doc_ids, args.test_fraction)
    shared = np.intersect1d(np.unique(doc_ids[mask]), np.unique(doc_ids[~mask]))
    if shared.size:
        raise AssertionError(f"{shared.size} documents straddle the split")
    manifest["documents_straddling_split"] = 0

    print(json.dumps(manifest, indent=2))
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(manifest, fh, indent=2)


if __name__ == "__main__":
    main()
