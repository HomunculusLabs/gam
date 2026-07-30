#!/usr/bin/env python3
"""gam-vs-reference view over every [QUALITY_PAIR] telemetry row of a
reference-quality run.

Left  : scatter, gam error vs reference error, both oriented lower-is-better.
Right : the named losses, as a ratio bar chart.

Pairs whose ratio is below 1e-3 are exact-arithmetic identities (geodesic
round-trips, closed-form CIF integrals) rather than accuracy comparisons; they
are counted but held off the scatter's axes, which they otherwise dominate by
fifteen orders of magnitude.
"""
import re, sys, math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

pat = re.compile(
    r"\[QUALITY_PAIR\] category=(\S+) test=(\S+) metric=(\S+) gam=(\S+) "
    r"reference=(\S+) reference_value=(\S+) lower_is_better=(\S+)")

rows, seen = [], set()
for line in open(sys.argv[1], errors="replace"):
    m = pat.search(line)
    if not m:
        continue
    cat, test, metric, g, ref, rv, lib = m.groups()
    try:
        g, rv = float(g), float(rv)
    except ValueError:
        continue
    if not (math.isfinite(g) and math.isfinite(rv)):
        continue
    if (test, metric) in seen:
        continue
    seen.add((test, metric))
    lower = lib == "true"
    gg, rr = (g, rv) if lower else (-g, -rv)
    if gg <= 0 or rr <= 0:
        continue
    rows.append((cat, test, metric, gg, rr, ref))

ratios = [(g / r, c, t, m) for c, t, m, g, r, _ in rows]
# Split on ABSOLUTE magnitude, not on the ratio: the exact-arithmetic pairs
# (geodesic round-trips, closed-form CIF integrals) sit at 1e-15 in BOTH
# coordinates, so a ratio filter leaves them on the axes and they stretch the
# log range over thirteen empty decades.
# A pair belongs on the accuracy scatter only if BOTH sides are in the regime
# where the comparison is about accuracy. Where gam is exact (1e-15) and the
# reference is approximate (1e0), the pair is a correctness win, not an accuracy
# datum, and plotting it stretches the log axes over eleven empty decades.
FLOOR = 1e-4
bulk = [x for x in rows if min(x[3], x[4]) >= FLOOR]
exact = [x for x in rows if min(x[3], x[4]) < FLOOR]
wins = sum(1 for r, *_ in ratios if r < 1.0)
print(f"{len(rows)} pairs | {wins} gam-better / {len(rows)-wins} reference-better "
      f"| {len(exact)} exact-identity pairs held off the scatter")

cats = sorted({r[0] for r in rows})
palette = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860", "#DA8BC3"]
colour = {c: palette[i % len(palette)] for i, c in enumerate(cats)}

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(16.5, 8.4),
                              gridspec_kw={"width_ratios": [1.15, 1.0]})

vals = [v for r in bulk for v in (r[3], r[4])]
lo, hi = min(vals) / 2.2, max(vals) * 2.2
# 10% tolerance band: most of these tests assert gam <= 1.10 * reference.
ax.fill_between([lo, hi], [lo, hi], [lo * 1.1, hi * 1.1],
                color="#C44E52", alpha=0.10, zorder=0)
ax.plot([lo, hi], [lo, hi], color="#333333", lw=1.4, zorder=1)
ax.plot([lo, hi], [lo * 1.1, hi * 1.1], color="#C44E52", lw=1.0, ls="--", zorder=1)
for c in cats:
    sub = [r for r in bulk if r[0] == c]
    if sub:
        ax.scatter([r[4] for r in sub], [r[3] for r in sub], s=62, alpha=0.85,
                   color=colour[c], edgecolor="white", linewidth=0.7,
                   label=f"{c} ({len(sub)})", zorder=3)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
ax.set_xlabel("mature reference tool — error (lower is better)")
ax.set_ylabel("gam — error (lower is better)")
ax.set_title(f"gam vs the mature reference, {len(bulk)} comparable pairs\n"
             "below the line = gam wins; shaded band = the 10% tolerance most tests allow",
             fontsize=11.5)
ax.legend(fontsize=8.5, loc="upper left", framealpha=0.92)
ax.grid(alpha=0.22, which="both", lw=0.5)
if exact:
    ax.text(0.985, 0.02,
            f"+{len(exact)} exact-arithmetic pairs off-scale "
            f"(one side < 1e-4; {sum(1 for e in exact if e[3] < e[4])} of them gam wins)",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color="#555555")

losses = sorted([x for x in ratios if x[0] > 1.0], reverse=True)[:22]
def short(t, m):
    head = t.split("::")[0]
    for pre in ("quality_vs_", "quality_"):
        if head.startswith(pre):
            head = head[len(pre):]
            break
    return f"{head[:36]}::{m[:22]}"
labels = [short(t, m) for _, c, t, m in losses]
ys = list(range(len(losses)))[::-1]
ax2.barh(ys, [r for r, *_ in losses],
         color=[colour[c] for _, c, _, _ in losses], height=0.78,
         edgecolor="white", linewidth=0.5)
ax2.axvline(1.0, color="#333333", lw=1.4)
ax2.axvline(1.1, color="#C44E52", lw=1.0, ls="--")
ax2.set_yticks(ys); ax2.set_yticklabels(labels, fontsize=7.6)
ax2.set_xlim(1.0, max(r for r, *_ in losses) * 1.06)
ax2.set_xlabel("gam error / reference error")
ax2.set_title(f"the {len(losses)} largest losses, named\n"
              f"(of {len(ratios)-wins} losses and {wins} wins overall)", fontsize=11.5)
ax2.grid(alpha=0.22, axis="x", lw=0.5)

fig.tight_layout()
fig.savefig(sys.argv[2], dpi=145)
print("wrote", sys.argv[2])
