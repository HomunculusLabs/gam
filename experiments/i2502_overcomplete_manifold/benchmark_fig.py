"""#2502 benchmark figure: held-out EV at matched L0 across methods, from
~/i2502/fits/fits.jsonl. One chart per PCA chart width present.
"""
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, GRAY = "#0072B2", "#D55E00", "#8C8C8C"


def latest(records, name):
    out = None
    for r in records:
        if r.get("record") == name and r.get("status") == "ok":
            out = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fits", default=os.path.expanduser("~/i2502/fits/fits.jsonl"))
    ap.add_argument("--k", type=int, default=32000)
    ap.add_argument("--suffix", default="", help="'' for p512 chart, '_p128' for p128")
    ap.add_argument("--out", default=os.path.expanduser("~/i2502/fits/fig_benchmark.png"))
    args = ap.parse_args()
    records = [json.loads(x) for x in open(args.fits)]
    s = args.suffix

    bars = []
    man = latest(records, f"manifold_k{args.k}_p128") or latest(records, f"manifold_k{args.k}")
    if man:
        bars.append((f"gam manifold dictionary\nK={args.k} (topology-adjudicated, REML/LAML)",
                     man["test_ev"], BLUE, man.get("test_mean_l0")))
    lin = latest(records, f"linear_k{args.k}{s}") or latest(records, f"linear_k{args.k}")
    if lin:
        bars.append((f"gam linear TopK lane\nK={args.k}", lin["test_ev"], "#56B4E9",
                     lin.get("test_mean_l0")))
    tk = latest(records, f"torch_topk_k{args.k}{s}")
    if tk:
        bars.append(("TopK SAE, direct torch\n(Adam, Gao et al. 2024)", tk["test_ev"],
                     ORANGE, tk.get("test_mean_l0")))
    sl = latest(records, f"saelens_topk_k{args.k}{s}")
    if sl:
        bars.append(("TopK SAE, sae-lens 6.46\n(external library)", sl["test_ev"],
                     "#E69F00", sl.get("test_mean_l0")))
    pca = latest(records, f"pca_M8{s}")
    if pca:
        bars.append(("PCA, 8 components\n(matched coefficient budget)",
                     pca["test_ev"], GRAY, 8))

    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    xs = np.arange(len(bars))
    for i, (label, ev, color, _l0) in enumerate(bars):
        ax.bar(i, ev, width=0.62, color=color)
        ax.annotate(f"{ev:.3f}", (i, ev), ha="center", va="bottom", fontsize=10,
                    color="#333333")
    ax.set_xticks(xs, [b[0] for b in bars], fontsize=8.5)
    ax.set_ylabel("held-out explained variance (9,987 fresh rows)")
    ax.set_title("Reconstruction quality at matched sparsity (L0 = 8 active atoms/row)",
                 fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0, max(b[1] for b in bars) * 1.15)
    fig.tight_layout()
    fig.savefig(args.out, dpi=160)
    print("wrote", args.out, "bars:", [(b[0].split("\n")[0], round(b[1], 4)) for b in bars])


if __name__ == "__main__":
    main()
