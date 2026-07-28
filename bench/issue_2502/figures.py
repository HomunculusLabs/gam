"""#2502 figures + interpretation from the Rust fit artifacts.

Everything plotted here is READ from what the Rust harness dumped -- curve_*.bin
are decoded by `SaeSupportSparseTerm::decode_atom_at` in Rust, not recomputed in
Python. Python only arranges the numbers.

Produces:
  manifolds.png     the fitted atom curves in the chart, with the real tokens
                    that route to each one plotted at their fitted coordinate
  interpret.md      per-atom top tokens along the coordinate (what varies)
  overcomplete.png  usage census + atom coherence vs the Welch bound
"""

import json
import os
import sys

import numpy as np

I2502 = os.path.expanduser("~/i2502v2")
P = 128


def load_f64(path, cols):
    return np.frombuffer(open(path, "rb").read(), dtype=np.float64).reshape(-1, cols)


def main() -> int:
    fit_dir = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else fit_dir
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    manifest = json.load(open(f"{fit_dir}/manifest.json"))
    train = np.load(f"{I2502}/train_chart.npy")
    tokens = np.load(f"{I2502}/train_tokens.npy")
    vocab = np.load(f"{I2502}/vocab.npy", allow_pickle=True)
    print(f"{len(manifest)} atoms dumped; chart {train.shape}", flush=True)

    # A shared 2-D view so every atom is drawn in the SAME coordinates and the
    # panels are comparable; PCA of the chart itself, not per-atom.
    c = train - train.mean(0)
    _u, _s, vt = np.linalg.svd(c[:20000], full_matrices=False)
    view = vt[:2]

    n = len(manifest)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.8 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")

    lines = ["# Fitted atom interpretation", ""]
    for entry in manifest:
        idx, atom_id, kind = entry["idx"], entry["atom"], entry["kind"]
        curve = load_f64(f"{fit_dir}/curve_{idx}.bin", P)
        toks = load_f64(f"{fit_dir}/tokens_{idx}.bin", 3)
        ax = axes.flat[idx]
        ax.axis("on")

        pts = (train[toks[:, 0].astype(int)] - train.mean(0)) @ view.T
        cur = (curve - train.mean(0)) @ view.T
        sc = ax.scatter(pts[:, 0], pts[:, 1], c=toks[:, 1], s=6, cmap="coolwarm",
                        alpha=0.55, linewidths=0)
        ax.plot(cur[:, 0], cur[:, 1], "k-", lw=2.0, zorder=3)
        ax.plot(cur[0, 0], cur[0, 1], "ko", ms=5, zorder=4)
        ax.set_title(f"atom {atom_id} · {kind} · used by {entry['usage']}", fontsize=9)
        ax.set_xlabel("chart PC1", fontsize=7)
        ax.set_ylabel("chart PC2", fontsize=7)
        ax.tick_params(labelsize=6)
        fig.colorbar(sc, ax=ax, label="coordinate t", fraction=0.046)

        # interpretation: tokens at the two ends of the fitted coordinate
        order = np.argsort(toks[:, 1])
        lo_rows = toks[order[:40], 0].astype(int)
        hi_rows = toks[order[-40:], 0].astype(int)

        def words(rs):
            seen, out_w = set(), []
            for r in rs:
                w = str(vocab[tokens[r]]).strip()
                if w and w not in seen:
                    seen.add(w)
                    out_w.append(repr(w))
                if len(out_w) == 10:
                    break
            return ", ".join(out_w)

        lines += [
            f"## atom {atom_id} — `{kind}`, used by {entry['usage']} rows",
            f"- **low t**: {words(lo_rows)}",
            f"- **high t**: {words(hi_rows)}",
            "",
        ]

    fig.suptitle("Fitted overcomplete manifold atoms (curves decoded in Rust) "
                 "with the real Qwen3.5-4B tokens routed to them", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(f"{out}/manifolds.png", dpi=140)
    open(f"{out}/interpret.md", "w").write("\n".join(lines))

    # ---- overcompleteness ------------------------------------------------
    usage = np.array([e["usage"] for e in manifest], dtype=float)
    gam = np.stack([load_f64(f"{fit_dir}/curve_{e['idx']}.bin", P).mean(0)
                    for e in manifest])
    g = gam / np.maximum(np.linalg.norm(gam, axis=1, keepdims=True), 1e-12)
    coh = np.abs(g @ g.T)
    np.fill_diagonal(coh, 0.0)
    # K comes from the census dump, not from parsing the directory name: the
    # name encodes the ARM (alpha, seed), and inferring a model dimension
    # from a path is how a rename silently changes a reported number.
    k_total = len(np.fromfile(f"{fit_dir}/census.bin", dtype=np.float64).reshape(-1, 4))
    welch = np.sqrt((k_total - P) / (P * (k_total - 1)))

    fig2, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.hist(usage, bins=30, color="#4477aa")
    a1.set_xlabel("rows routed to atom")
    a1.set_ylabel("atoms")
    a1.set_title(f"usage of dumped atoms (K={k_total}, P={P}, K/P={k_total/P:.1f}x)")
    a2.hist(coh[np.triu_indices_from(coh, 1)], bins=40, color="#cc6677")
    a2.axvline(welch, color="k", ls="--", label=f"Welch bound {welch:.3f}")
    a2.set_xlabel("|cos| between atom mean images")
    a2.set_title("pairwise coherence — overcomplete ⇒ cannot be orthogonal")
    a2.legend(fontsize=8)
    fig2.tight_layout()
    fig2.savefig(f"{out}/overcomplete.png", dpi=140)

    print(json.dumps({
        "k_total": k_total, "P": P, "overcompleteness": k_total / P,
        "welch_bound": float(welch),
        "median_coherence": float(np.median(coh[np.triu_indices_from(coh, 1)])),
        "max_coherence": float(coh.max()),
        "usage_min": float(usage.min()), "usage_median": float(np.median(usage)),
    }, indent=1), flush=True)
    print("FIGURES DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
