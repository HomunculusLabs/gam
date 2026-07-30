#!/usr/bin/env python3
"""gam vs the mature reference across the whole quality suite, coloured by the
suite's own outcome class (PASS / METRIC_OFF / GAM_ERROR).

  argv[1]  quality_results.tsv   (one row per libtest case, with its outcome)
  argv[2]  pairs.log             ([QUALITY_PAIR] telemetry lines)
  argv[3]  out.png

A GAM_ERROR has no coordinates by construction: gam raised instead of producing
a fit, so it emitted no pair. They are therefore counted in their own panel
rather than silently dropped -- the whole point of the class split is that a
refusal is not an accuracy datum.
"""
import re, sys, math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

tsv, pairlog, out = sys.argv[1], sys.argv[2], sys.argv[3]

# --- outcome per (category, file stem) -------------------------------------
outcome_by_key, outcome_counts = {}, {}
for i, line in enumerate(open(tsv, errors="replace")):
    f = line.rstrip("\n").split("\t")
    if i == 0 or len(f) < 4:
        continue
    outcome, test = f[1], f[3]
    outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1
    parts = test.split("::")
    if len(parts) >= 2:
        # A file can hold several test fns; a GAM_ERROR in one must not repaint
        # a sibling that passed, so keep the worst outcome seen for the file.
        rank = {"PASS": 0, "REF_ERROR": 1, "METRIC_OFF": 2, "GAM_ERROR": 3}
        k = (parts[0], parts[1])
        if outcome_by_key.get(k) is None or rank.get(outcome, 1) > rank.get(outcome_by_key[k], 1):
            outcome_by_key[k] = outcome

pat = re.compile(
    r"\[QUALITY_PAIR\] category=(\S+) test=(\S+) metric=(\S+) gam=(\S+) "
    r"reference=(\S+) reference_value=(\S+) lower_is_better=(\S+)")

rows, seen = [], set()
for line in open(pairlog, errors="replace"):
    m = pat.search(line)
    if not m:
        continue
    cat, test, metric, g, ref, rv, lib = m.groups()
    if (test, metric) in seen:
        continue
    seen.add((test, metric))
    try:
        g, rv = float(g), float(rv)
    except ValueError:
        continue
    if not (math.isfinite(g) and math.isfinite(rv)):
        continue
    gg, rr = (g, rv) if lib == "true" else (-g, -rv)
    if min(gg, rr) <= 1e-4:          # exact-arithmetic identities, not accuracy
        continue
    stem = test.split("::")[0]
    rows.append((outcome_by_key.get((cat, stem), "unknown"), gg, rr, test, metric, ref))

style = {
    "PASS":       ("#2E8B57", "PASS"),
    "METRIC_OFF": ("#C44E52", "METRIC_OFF"),
    "REF_ERROR":  ("#B0A08A", "REF_ERROR"),
    "unknown":    ("#9AA0A6", "unmatched"),
}

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(16.5, 8.0),
                              gridspec_kw={"width_ratios": [1.45, 1.0]})

vals = [v for r in rows for v in (r[1], r[2])]
lo, hi = min(vals) / 2.2, max(vals) * 2.2
ax.fill_between([lo, hi], [lo, hi], [lo * 1.1, hi * 1.1],
                color="#C44E52", alpha=0.09, zorder=0)
ax.plot([lo, hi], [lo, hi], color="#333333", lw=1.4, zorder=1)
ax.plot([lo, hi], [lo * 1.1, hi * 1.1], color="#C44E52", lw=1.0, ls="--", zorder=1)
for key in ("PASS", "REF_ERROR", "unknown", "METRIC_OFF"):
    sub = [r for r in rows if r[0] == key]
    if not sub:
        continue
    c, lab = style[key]
    ax.scatter([r[2] for r in sub], [r[1] for r in sub], s=64, alpha=0.85,
               color=c, edgecolor="white", linewidth=0.7,
               label=f"{lab} ({len(sub)} pairs)",
               zorder=4 if key == "METRIC_OFF" else 3)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
ax.set_xlabel("mature reference tool — error (lower is better)")
ax.set_ylabel("gam — error (lower is better)")
ax.set_title("gam vs the mature reference, by outcome class\n"
             "below the line = gam wins; band = the 10% tolerance most tests allow",
             fontsize=12)
ax.legend(fontsize=9, loc="upper left", framealpha=0.93)
ax.grid(alpha=0.22, which="both", lw=0.5)
wins = sum(1 for r in rows if r[1] < r[2])
ax.text(0.985, 0.02, f"{wins} of {len(rows)} pairs below the diagonal",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=9, color="#555555")

# --- right panel: the suite by class, and where the GAM_ERRORs went ---------
order = ["PASS", "METRIC_OFF", "GAM_ERROR", "REF_ERROR"]
counts = [outcome_counts.get(k, 0) for k in order]
total = sum(outcome_counts.values())
cols = ["#2E8B57", "#C44E52", "#8B2E3F", "#B0A08A"]
bars = ax2.bar(range(len(order)), counts, color=cols, width=0.62, edgecolor="white")
for b, c in zip(bars, counts):
    ax2.text(b.get_x() + b.get_width() / 2, c + total * 0.012, f"{c}\n{100*c/total:.1f}%",
             ha="center", va="bottom", fontsize=10)
ax2.set_xticks(range(len(order)))
ax2.set_xticklabels(order, fontsize=10)
ax2.set_ylim(0, max(counts) * 1.22)
ax2.set_ylabel("libtest cases")
ax2.set_title(f"the whole suite: {total} tests by outcome\n"
              "a GAM_ERROR has no point on the left panel — it produced no fit",
              fontsize=12)
ax2.grid(alpha=0.22, axis="y", lw=0.5)

fig.tight_layout()
fig.savefig(out, dpi=145)
print(f"{len(rows)} plotted pairs, {wins} gam-better")
print("suite:", outcome_counts, "total", total)
print("unmatched pair keys:", sum(1 for r in rows if r[0] == "unknown"))
