"""Score the v2 confirmation in all three currencies, on the fresh rows only.

Chart EV is what the campaign has reported. It asks how well 128 PCA directions
are reconstructed, and the chart holds only ~68% of residual-stream variance, so
a chart-space win is not by itself evidence about the model's computation.

Reported side by side, never mixed:
  chart EV    inside the PCA-128 chart
  ambient EV  lifted back to the 2560-d residual stream

and separately (splice.py) the cross-entropy change from substituting the
reconstruction at layer 16.

Scored on the FRESH subset: 8.68% of v2 held-out rows come from documents that
were also v1 held-out, and v1 adjudicated every adaptive decision in this
campaign. Those rows are selection-contaminated, not training-contaminated, and
both numbers are printed so the difference is visible rather than asserted.
"""

import glob
import json
import os
import sys

import numpy as np

V2 = os.path.expanduser("~/i2502v2")
P = 128


def ev(x, r):
    return 1.0 - float(((x - r) ** 2).sum()) / float((x ** 2).sum())


def main() -> int:
    lift = np.load(f"{V2}/lift.npy")
    c0 = np.load(f"{V2}/c0.npy")
    chart = np.load(f"{V2}/test_chart.npy")
    amb = np.load(f"{V2}/test_ambient.npy").astype(np.float64)
    clean = np.load(f"{V2}/test_clean_mask.npy")
    print(f"v2 test rows={len(chart)} fresh={int(clean.sum())} "
          f"({clean.mean():.4f})", flush=True)

    rows = []
    for path in sorted(glob.glob(f"{V2}/a8_s*/heldout_recon.bin")):
        seed = path.split("a8_s")[1].split("/")[0]
        rec = np.frombuffer(open(path, "rb").read(), dtype=np.float64).reshape(-1, P)
        if len(rec) != len(chart):
            print(f"REFUSED {path}: {len(rec)} rows != {len(chart)}")
            continue
        rec_amb = rec @ lift + c0
        rows.append({
            "arm": "manifold", "seed": int(seed),
            "chart_ev_all": ev(chart, rec),
            "chart_ev_fresh": ev(chart[clean], rec[clean]),
            "ambient_ev_all": ev(amb, rec_amb),
            "ambient_ev_fresh": ev(amb[clean], rec_amb[clean]),
        })
        print(json.dumps(rows[-1]), flush=True)

    # Baselines were saved as decoders; rescore them on the same fresh subset.
    for path in sorted(glob.glob(f"{V2}/baseline_k*_s*.npz")):
        blob = np.load(path)
        W = blob["W_dec"].astype(np.float64)
        b = blob["b_pre"].astype(np.float64)
        k_act = int(blob["k_act"]) if "k_act" in blob else 8
        norms = (W * W).sum(1)
        R = chart - b
        taken = np.zeros((len(chart), len(W)), dtype=bool)
        picks = np.zeros((len(chart), k_act), dtype=np.int64)
        for s in range(k_act):
            gain = 2.0 * (R @ W.T) - norms
            gain[taken] = -np.inf
            p = gain.argmax(1)
            picks[:, s] = p
            taken[np.arange(len(chart)), p] = True
            coef = np.maximum((R * W[p]).sum(1) / np.maximum(norms[p], 1e-30), 0.0)
            R = R - coef[:, None] * W[p]
        rec = np.empty_like(chart)
        for i in range(len(chart)):
            A = W[picks[i]].T
            c, *_ = np.linalg.lstsq(A, chart[i] - b, rcond=None)
            rec[i] = A @ c
        rec = rec + b
        rec_amb = rec @ lift + c0
        name = os.path.basename(path).replace(".npz", "")
        rows.append({
            "arm": name,
            "chart_ev_all": ev(chart, rec),
            "chart_ev_fresh": ev(chart[clean], rec[clean]),
            "ambient_ev_all": ev(amb, rec_amb),
            "ambient_ev_fresh": ev(amb[clean], rec_amb[clean]),
        })
        print(json.dumps(rows[-1]), flush=True)

    json.dump(rows, open(f"{V2}/scores.json", "w"), indent=1)
    print("SCORE DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
