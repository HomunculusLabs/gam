"""Build the doc-split evaluation matrices per the declared #2502 contract
(gam-2502-docsplit-v1, artifacts/issue_2502_doc_split.py @ main).

Produces prep_L16_p128_doc/: train.npy (rows from TRAIN-side docs only, PCA'd
through the EXISTING train-only chart), test_doc.npy (held-out-doc rows), and
a contamination report for the in-flight fit (how many of its train rows sit
on the held-out side).
"""
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.expanduser("~/lane-2502"))
from artifacts.issue_2502_doc_split import row_side, split_manifest  # noqa: E402

H = os.path.expanduser("~/i2502/harvest")
SRC = os.path.expanduser("~/i2502/prep_L16_p128")
DST = os.path.expanduser("~/i2502/prep_L16_p128_doc")
os.makedirs(DST, exist_ok=True)

doc_ids = np.load(f"{H}/doc_ids.npy")
held = row_side(doc_ids)
man = split_manifest(doc_ids)
print("manifest:", {k: man[k] for k in ("split_version", "split_hash")}, flush=True)

lift = np.load(f"{SRC}/lift.npy")
c0 = np.load(f"{SRC}/c0.npy")
rows_train = np.load(f"{SRC}/rows_train.npy")
rows_test = np.load(f"{SRC}/rows_test.npy")

contam_train = held[rows_train]
contam_test = held[rows_test]
print(f"in-flight fit: {contam_train.sum()}/{len(rows_train)} train rows are on the "
      f"held-out side; {contam_test.sum()}/{len(rows_test)} of the old test rows are "
      f"genuinely held-out-doc", flush=True)

X = np.load(f"{H}/resid_L16.npy", mmap_mode="r")

# doc-clean train pool: the current train rows that are NOT held out (so the
# chart and any refit share provenance with the in-flight fit)
tr_clean = rows_train[~contam_train]
Xtr = (np.asarray(X[tr_clean], dtype=np.float64) - c0) @ lift.T
np.save(f"{DST}/train.npy", np.ascontiguousarray(Xtr))
np.save(f"{DST}/rows_train.npy", tr_clean)

# doc-split test: a fixed, deterministic 12k-row sample of held-out-doc rows
ho_rows = np.flatnonzero(held)
sel = ho_rows[np.linspace(0, len(ho_rows) - 1, 12000).astype(int)]
Xte = (np.asarray(X[sel], dtype=np.float64) - c0) @ lift.T
np.save(f"{DST}/test_doc.npy", np.ascontiguousarray(Xte))
np.save(f"{DST}/rows_test_doc.npy", sel)

for name in ("lift", "c0", "c1", "mu"):
    np.save(f"{DST}/{name}.npy", np.load(f"{SRC}/{name}.npy"))
meta = json.load(open(f"{SRC}/meta.json"))
meta.update(split_version=man["split_version"], split_hash=man["split_hash"],
            n_train=int(len(tr_clean)), n_test=int(len(sel)),
            note="doc-split per gam-2502-docsplit-v1; chart inherited from row-split "
                 "train-only PCA (chart provenance predates the contract)")
json.dump(meta, open(f"{DST}/meta.json", "w"), indent=2)
print("DONE", json.dumps(meta)[:300], flush=True)
