"""Is Qwen3.5-4B's local curvature REUSABLE, or is every neighbourhood its own shape?

The curvature census on this issue established that neighbourhoods of the residual
stream bend away from their own tangent plane beyond what a covariance-matched
Gaussian can produce. That licenses curved atoms. It does not license a
*dictionary* of curved atoms, which is a strictly stronger claim: that a small
number of reusable shapes explains the bending everywhere, rather than each
neighbourhood carrying its own idiosyncratic second fundamental form.

That distinction is measurable without fitting anything, which matters because the
support-sparse fit is blocked on #2517. The test is transfer:

  1. at landmark A, fit a local tangent frame and a quadratic normal model on A's
     own training rows;
  2. carry that quadratic to landmark B by the orthogonal map that best aligns A's
     tangent frame to B's (Procrustes), which is the only frame-free way to ask
     whether the *shape* is the same;
  3. score it on B's HELD-OUT rows, and compare against B's own locally fitted
     quadratic (the ceiling) and against two nulls.

Nulls, because a positive transfer score can be manufactured two ways:
  * "rotation" — carry A's quadratic to B under a RANDOM orthogonal map instead of
    the Procrustes one. Beating this says the alignment carries information, not
    merely that some quadratic fits;
  * "gaussian" — the whole pipeline on a covariance-matched Gaussian, which is
    locally flat by construction and supplies the finite-sample floor.

Train/test rows come from the declared document split (`split_hash`
6e3dbf8dcb8164bebc9d58eaacb56067), so no document contributes to both a chart and
the rows that score it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from issue_2502_doc_split import row_side, split_manifest  # noqa: E402


def farthest_point_landmarks(X, count, seed):
    """Deterministic farthest-point sampling: spread landmarks over the cloud."""
    rng = np.random.default_rng(seed)
    picked = [int(rng.integers(len(X)))]
    d = np.linalg.norm(X - X[picked[0]], axis=1)
    for _ in range(count - 1):
        nxt = int(np.argmax(d))
        picked.append(nxt)
        d = np.minimum(d, np.linalg.norm(X - X[nxt], axis=1))
    return np.array(picked)


def quadratic_features(T):
    """Monomials of the tangent coordinates, degree 2, without the constant."""
    d = T.shape[1]
    cols = [T]
    quad = [T[:, i] * T[:, j] for i in range(d) for j in range(i, d)]
    cols.append(np.column_stack(quad))
    return np.column_stack([np.ones(len(T))] + cols)


def local_chart(X, centre, neighbours, d):
    """Centre and top-d tangent frame from a neighbourhood's own rows."""
    P = X[neighbours]
    mu = P.mean(0)
    _, _, vt = np.linalg.svd(P - mu, full_matrices=False)
    return mu, np.ascontiguousarray(vt[:d])


def normal_excursion(X, rows, mu, frame):
    """Tangent coordinates and the MAGNITUDE of the normal remainder.

    The magnitude, not the normal vector, is the transferable target. A normal
    vector lives in the neighbourhood's own (D-d)-dimensional normal space, and
    those spaces differ between landmarks, so carrying a vector-valued model from
    one landmark to another would score A's normal directions against B's and
    measure frame mismatch rather than shape. The magnitude is frame-free in the
    output while remaining a direct reading of how far the patch bends away from
    its own tangent plane as a function of tangent direction.
    """
    C = X[rows] - mu
    T = C @ frame.T
    return T, np.linalg.norm(C - T @ frame, axis=1)


def fit_quadratic(X, rows, mu, frame):
    """Regress the normal excursion on quadratic monomials of tangent coords."""
    T, y = normal_excursion(X, rows, mu, frame)
    coef, *_ = np.linalg.lstsq(quadratic_features(T), y, rcond=None)
    return coef


def score_quadratic(X, rows, mu, frame, coef):
    """Out-of-sample R² of the quadratic excursion model on `rows`."""
    T, y = normal_excursion(X, rows, mu, frame)
    pred = quadratic_features(T) @ coef
    ss = float(((y - y.mean()) ** 2).sum())
    if ss <= 0.0:
        return 0.0
    return 1.0 - float(((y - pred) ** 2).sum()) / ss


def transfer_map(frame_a, frame_b):
    """Orthogonal map carrying frame A onto frame B (Procrustes on the frames)."""
    u, _, vt = np.linalg.svd(frame_b @ frame_a.T)
    return u @ vt  # d x d, orthogonal


def carry(coef_a, R, d):
    """Re-express A's quadratic in B's tangent coordinates under rotation R.

    The design column order is [1, t_1..t_d, t_i t_j for i<=j]. A rotation of the
    tangent coordinates t -> R t induces a linear map on that column basis, built
    here explicitly so the carried model predicts the same shape, read in B's frame.
    """
    n_quad = d * (d + 1) // 2
    M = np.zeros((1 + d + n_quad, 1 + d + n_quad))
    M[0, 0] = 1.0
    M[1 : 1 + d, 1 : 1 + d] = R
    pairs = [(i, j) for i in range(d) for j in range(i, d)]
    for col, (a, b) in enumerate(pairs):
        # (R t)_a (R t)_b expanded in the monomials of t
        outer = np.outer(R[a], R[b])
        sym = outer + outer.T
        for row, (i, j) in enumerate(pairs):
            M[1 + d + col, 1 + d + row] = sym[i, j] if i != j else outer[i, i]
    return M.T @ coef_a


def gaussian_twin(X, seed):
    rng = np.random.default_rng(seed)
    mu = X.mean(0)
    C = np.cov(X - mu, rowvar=False)
    vals, vecs = np.linalg.eigh(C)
    root = vecs @ np.diag(np.sqrt(np.maximum(vals, 0.0))) @ vecs.T
    return mu + rng.standard_normal(X.shape) @ root


def run_arm(X, landmarks_count, m, d, seed, label):
    rng = np.random.default_rng(seed + 991)
    lm = farthest_point_landmarks(X, landmarks_count, seed)
    charts = []
    for centre in lm:
        order = np.argsort(np.linalg.norm(X - X[centre], axis=1))[: 2 * m]
        # Split the neighbourhood at RANDOM, not by radius. Taking the inner half
        # to fit and the outer half to score would make every model extrapolate to
        # a radius regime it never saw, which is a property of the split rather
        # than of the manifold.
        shuffled = rng.permutation(order)
        fit_rows, test_rows = shuffled[:m], shuffled[m:]
        mu, frame = local_chart(X, X[centre], fit_rows, d)
        charts.append(dict(centre=centre, fit=fit_rows, test=test_rows,
                           mu=mu, frame=frame,
                           coef=fit_quadratic(X, fit_rows, mu, frame)))

    own, transferred, rotated = [], [], []
    for b_i, B in enumerate(charts):
        own.append(score_quadratic(X, B["test"], B["mu"], B["frame"], B["coef"]))
        for a_i, A in enumerate(charts):
            if a_i == b_i:
                continue
            R = transfer_map(A["frame"], B["frame"])
            transferred.append(
                score_quadratic(X, B["test"], B["mu"], B["frame"], carry(A["coef"], R, d))
            )
            Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
            rotated.append(
                score_quadratic(X, B["test"], B["mu"], B["frame"], carry(A["coef"], Q, d))
            )

    def stat(v):
        # The mean alone cannot distinguish "no pair transfers" from "most pairs
        # do not, but some do" — which is the difference between refuting a shared
        # shape and finding a sparse one. Report the best pair and how many clear
        # zero as well.
        v = np.asarray(v)
        return dict(mean=float(v.mean()),
                    se=float(v.std(ddof=1) / np.sqrt(len(v))),
                    best=float(v.max()), median=float(np.median(v)),
                    frac_above_zero=float((v > 0.0).mean()), n=int(len(v)))

    return dict(arm=label, landmarks=landmarks_count, m=m, d=d,
                own_neighbourhood=stat(own), transferred=stat(transferred),
                random_rotation=stat(rotated))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--harvest", default=os.path.expanduser("~/i2502/harvest"))
    ap.add_argument("--layer", type=int, default=16)
    ap.add_argument("--rows", type=int, default=30000)
    ap.add_argument("--chart", type=int, default=128)
    ap.add_argument("--landmarks", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.expanduser("~/i2502-baselines/curvature_transfer.jsonl"))
    args = ap.parse_args()

    Xall = np.load(f"{args.harvest}/resid_L{args.layer}.npy", mmap_mode="r")
    doc_ids = np.load(f"{args.harvest}/doc_ids.npy")
    manifest = split_manifest(doc_ids)
    print("[split] " + json.dumps(manifest), flush=True)

    held = row_side(doc_ids)
    rng = np.random.default_rng(args.seed)
    train_rows = np.sort(rng.choice(np.flatnonzero(~held), size=args.rows, replace=False))
    X = np.asarray(Xall[train_rows], dtype=np.float64)
    mu = X.mean(0)
    Xc = X - mu
    # Chart via the feature-space Gram eigendecomposition rather than a full SVD
    # of the row matrix: identical subspace, and 2560x2560 instead of 30000x2560.
    evals, evecs = np.linalg.eigh(Xc.T @ Xc)
    lift = np.ascontiguousarray(evecs[:, np.argsort(evals)[::-1][: args.chart]].T)
    X = Xc @ lift.T
    print(f"[prep] chart {X.shape} from {len(train_rows)} train-document rows", flush=True)

    G = gaussian_twin(X, args.seed)

    records = [dict(record="split_manifest", **manifest)]
    for m in (256, 512):
        for d in (2, 3):
            for label, data in (("real", X), ("gaussian", G)):
                rec = dict(record="transfer", layer=args.layer, chart=args.chart,
                           **run_arm(data, args.landmarks, m, d, args.seed, label))
                records.append(rec)
                print("[transfer] " + json.dumps(rec), flush=True)
                with open(args.out, "w") as fh:
                    for r in records:
                        fh.write(json.dumps(r) + "\n")
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
