"""Figures for the Gemma Scope 2 shattered-circle census.

Reads only what the Rust census wrote (`cen_<seed>.json` and the matching
`.planes.json`). No statistic is recomputed here; this file draws.

    python census_figs.py <dir-with-cen_*.json> <outdir> [<token-dump-dir>]
"""

import glob
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REAL = "#2a78d6"
NULL = "#eb6834"
INK = "#1a1a19"
MUTED = "#6b6a63"

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
        "figure.dpi": 150,
    }
)


FLAGSHIP = os.environ.get("FLAGSHIP", "17")


def load(d):
    real = json.load(open(os.path.join(d, f"cen_L{FLAGSHIP}.json")))
    nulls = [
        json.load(open(f))
        for f in sorted(glob.glob(os.path.join(d, "cen_null*.json")))
        if not f.endswith(".planes.json")
    ]
    layers = {}
    for f in sorted(glob.glob(os.path.join(d, "cen_L*.json"))):
        if f.endswith(".planes.json"):
            continue
        layers[int(os.path.basename(f)[5:].split(".")[0])] = json.load(open(f))
    return real, nulls, layers


def fig_layers(layers, out):
    ls = sorted(layers)
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    disc = [layers[l]["n_accepted"] for l in ls]
    scr = [layers[l]["n_pairs"] for l in ls]
    ax.bar([str(l) for l in ls], disc, color=REAL, width=0.55)
    for i, (dd, ss) in enumerate(zip(disc, scr)):
        ax.annotate(
            f"{dd}\nof {ss:,}", (i, dd), ha="center", va="bottom", fontsize=8.5, color=INK
        )
    ax.set_xlabel("Gemma 3 4B residual-stream layer (SAE hook point)")
    ax.set_ylabel("e-BH discoveries")
    ax.set_ylim(0, max(disc) * 1.35 + 1)
    ax.set_title(
        "Shattered circles by depth, one Gemma Scope 2 SAE per layer",
        loc="left",
        fontsize=12,
        color=INK,
    )
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    print("wrote", out)


def fig_kappa(real, nulls, out):
    kr = np.array([p["kappa"] for p in real["pairs"]])
    kn = np.concatenate([[p["kappa"] for p in n["pairs"]] for n in nulls])
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    bins = np.linspace(0.5, 8, 90)
    ax.hist(
        kr, bins=bins, density=True, color=REAL, alpha=0.85, label=f"Gemma Scope 2 ({len(kr):,} pairs)"
    )
    ax.hist(
        kn,
        bins=bins,
        density=True,
        histtype="step",
        linewidth=2,
        color=NULL,
        label=f"permutation null ({len(nulls)} draws, {len(kn):,} pairs)",
    )
    ax.axvline(2.0, color=INK, linewidth=1.2, linestyle="--")
    ax.annotate(
        "κ = 2\nGaussian fill",
        xy=(2.0, ax.get_ylim()[1] * 0.92),
        xytext=(2.35, ax.get_ylim()[1] * 0.92),
        color=INK,
        fontsize=9,
        va="top",
    )
    ax.axvline(1.0, color=INK, linewidth=1.2, linestyle=":")
    ax.annotate("κ = 1\nring", xy=(1.0, 0), xytext=(0.6, ax.get_ylim()[1] * 0.55), color=INK, fontsize=9)
    ax.set_xlabel("κ = m₄/m₂²  on the co-firing plane's joint radius")
    ax.set_ylabel("density")
    ax.set_title(
        "The amplitude law of every co-firing atom pair in a public SAE",
        loc="left",
        fontsize=12,
        color=INK,
    )
    na = real["n_accepted"]
    nn = np.mean([n["n_accepted"] for n in nulls])
    ax.text(
        0.99,
        0.62,
        f"accepted as rings\n  real  {na} / {real['n_pairs']:,}\n  null  {nn:.1f} / "
        f"{np.mean([n['n_pairs'] for n in nulls]):,.0f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color=INK,
        family="monospace",
    )
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    print("wrote", out)


def fig_planes(planes, out, toks=None, ncol=4, nrow=2):
    planes = sorted(planes, key=lambda p: p["kappa"])[: ncol * nrow]
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.1 * ncol, 3.25 * nrow))
    for ax, pl in zip(np.ravel(axes), planes):
        a = np.array(pl["alpha"])
        b = np.array(pl["beta"])
        th = np.arctan2(b, a)
        ax.scatter(a, b, c=th, cmap="twilight", s=3, alpha=0.65, linewidths=0)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ids = pl["members_a"] + pl["members_b"]
        ax.set_title(
            f"atoms {'+'.join(str(i) for i in ids)}\n"
            f"κ={pl['kappa']:.2f}  z={pl['z_below_gaussian']:.1f}σ  R̂={pl['radius_over_sigma']:.0f}σ",
            fontsize=8.5,
            color=INK,
        )
        if toks is not None:
            order = np.argsort(th)
            for q in np.linspace(0, len(order) - 1, 14).astype(int):
                i = order[q]
                ax.annotate(
                    toks(pl["rows"][i]),
                    (a[i], b[i]),
                    fontsize=6,
                    color=INK,
                    ha="center",
                )
    for ax in np.ravel(axes)[len(planes) :]:
        ax.axis("off")
    fig.suptitle(
        "Rings Google's SAE split into straight atoms — the plane parse of accepted pairs",
        fontsize=12,
        color=INK,
        x=0.02,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, bbox_inches="tight")
    print("wrote", out)


def fig_power(d, out):
    def curve(prefix):
        fs = sorted(
            (f for f in glob.glob(os.path.join(d, prefix + "*.json"))
             if not f.endswith(".planes.json")),
            key=lambda f: float(os.path.basename(f)[len(prefix):-5]),
        )
        rs = [float(os.path.basename(f)[len(prefix):-5]) for f in fs]
        return rs, [json.load(open(f))["n_accepted"] for f in fs]

    rc, dc = curve("power_comb_")
    rk, dk = curve("power_konly_")
    rr, dr = curve("power_rank_")
    if not rc and not rk and not rr:
        return
    fig, ax = plt.subplots(figsize=(6.8, 3.9))
    if rc:
        ax.plot(rc, dc, "-o", color=REAL, linewidth=2, markersize=7,
                label="kappa + fourth harmonic")
    if rk:
        ax.plot(rk, dk, "-s", color=NULL, linewidth=2, markersize=6,
                label="kappa alone")
    if rr:
        ax.plot(rr, dr, "-^", color="#1baf7a", linewidth=2, markersize=7,
                label="rank complementarity")
    ax.axvline(1.8138, color=INK, linewidth=1.2, linestyle=":")
    ax.annotate("derived rate-distortion\ncrossover  R = 1.814 sigma",
                (1.8138, ax.get_ylim()[1] * 0.72), xytext=(4, 0),
                textcoords="offset points", fontsize=8.5, color=INK, va="top")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("planted ring radius  (units of the dictionary's own reconstruction sigma)")
    ax.set_ylabel("e-BH discoveries")
    ax.set_title(
        "Power: a ring planted in the SAE's own code plane, re-encoded by its own encoder",
        loc="left", fontsize=11.5, color=INK,
    )
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    print("wrote", out)


def main():
    d, outdir = sys.argv[1], sys.argv[2]
    os.makedirs(outdir, exist_ok=True)
    real, nulls, layers = load(d)
    print(
        "real:",
        {k: v for k, v in real.items() if k != "pairs"},
    )
    for n in nulls:
        print("null:", n["permute_seed"], n["n_pairs"], n["n_accepted"])
    fig_kappa(real, nulls, os.path.join(outdir, "census_kappa.png"))
    fig_power(d, os.path.join(outdir, "census_power.png"))
    if len(layers) > 1:
        fig_layers(layers, os.path.join(outdir, "census_layers.png"))
    planes = json.load(open(os.path.join(d, f"cen_L{FLAGSHIP}.planes.json")))["planes"]
    toks = None
    if len(sys.argv) > 3:
        td = sys.argv[3]
        ids = np.fromfile(f"{td}/tokens.i32", dtype=np.int32)
        vocab = json.load(open(f"{td}/vocab.json"))
        toks = lambda r: repr(vocab.get(str(int(ids[r])), "?"))[1:-1][:10]
    if planes:
        fig_planes(planes, os.path.join(outdir, "census_rings.png"), toks)


if __name__ == "__main__":
    main()
