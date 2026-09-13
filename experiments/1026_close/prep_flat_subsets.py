"""#2283: the acceptance cell's own training rows for flat-tier scaling points.

Calls driver_1026_arms.load_chunk_dir / make_split verbatim (seed 0, max_rows 120000,
test_frac 0.2), so an n-row subset is the first n rows of the 96,000-row training split.
The subsets feed crates/gam-sae/examples/flat_tier_scaling_2283.rs.

usage: python prep_flat_subsets.py CHUNK_DIR OUT_DIR N [N ...]
"""

import hashlib
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from driver_1026_arms import load_chunk_dir, make_split  # noqa: E402


def main() -> int:
    if len(sys.argv) < 4:
        print(__doc__, file=sys.stderr)
        return 2
    chunk_dir, out_dir = sys.argv[1], Path(sys.argv[2])
    sizes = [int(value) for value in sys.argv[3:]]
    X, row_ids, manifest = load_chunk_dir(chunk_dir, 120_000, 0)
    x_tr, _x_te, tr_ids, _te_ids = make_split(X, row_ids, 0.2, 0)
    print("manifest_sha256", manifest["sha256"], "train", x_tr.shape, x_tr.dtype, flush=True)
    for n in sizes:
        if not 0 < n <= x_tr.shape[0]:
            print(f"size {n} is outside 1..{x_tr.shape[0]}", file=sys.stderr)
            return 2
        subset = np.ascontiguousarray(x_tr[:n], dtype="<f4")
        path = out_dir / f"creditscope_l30_train_first{n}.f32.npy"
        np.save(path, subset)
        ids = np.ascontiguousarray(tr_ids[:n], dtype="<i8")
        print(
            "wrote", path, subset.shape,
            "data_sha256", hashlib.sha256(subset.tobytes()).hexdigest(),
            "row_ids_sha256", hashlib.sha256(ids.tobytes()).hexdigest(),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
